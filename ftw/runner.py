"""Frontrun Fries' live loop: long-poll the state, answer every night inside the 2 s window.

    python3 -m ftw.runner            (or ./run.sh)

Each night: start the rival predictions, decide with the brain (or the fallback), post one order.
A watchdog posts the fallback's order at SAFE_AFTER seconds if the brain is not done by then.
control.json (optional, re-read every night): {"mode": "main"} or {"mode": "fallback"}.
The bot also switches itself to the fallback for the rest of a market when realised profit trails
the brain's own expectation badly over several nights.
"""
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

from . import game as G
from .brain import Brain, riot_risk
from .clean import sanitize
from .fallback import Fallback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("NIGHTSHIFT_BASE", "https://nightshift.joinbuilders.org").rstrip("/")
API = BASE + "/api/v1"
CONTROL = os.path.join(ROOT, "control.json")
LOGDIR = os.path.join(ROOT, "logs")
# Real bots (run from rivals/ under the shim) where they matched live at least as well as our ports did:
# BigBuck (its repo's saved data makes it exact), IBE and Wagyu (same orders as the ports). Cheapest: ports.
REAL_DEFAULT = "bigbuck,ibe,wagyu"
PREDICT_BUDGET = float(os.environ.get("FTW_PREDICT_S", "0.9"))
SAFE_AFTER = float(os.environ.get("FTW_SAFE_S", "1.35"))
SWITCH_NIGHTS, SWITCH_RATIO = 5, 0.4
# The final's day closes as soon as every truck has an order in ("waits up to 2 s"; practice plays on our first
# order), so an order posted at once could be the only one that counts. Default: one post, the brain's (or the
# safe order at SAFE_AFTER if the brain is not done). FTW_SAFE_FIRST=1 brings back the early safe order.
SAFE_FIRST = os.environ.get("FTW_SAFE_FIRST", "0") == "1"
REPLAN_BEFORE = float(os.environ.get("FTW_REPLAN_S", "0.55"))   # re-read planned counts this long before close
WINDOW_S = 2.0
REFETCH_S = float(os.environ.get("FTW_REFETCH_S", "60"))      # rival repo re-fetch period before the first night
REFETCH_UNTIL = os.environ.get("FTW_REFETCH_UNTIL", "0831")    # ET HHMM: stop re-fetching after this
RETRY_STATUS = (0, 502, 503, 504)   # transport failure (timeout, connection) or a gateway hiccup: send it again


def parse_ts(s):
    """Seconds from an ISO time like 2026-10-24T13:00:02.000Z (only differences are used)."""
    try:
        import datetime as dt
        s = str(s).strip().replace("Z", "")[:26]
        base, _, frac = s.partition(".")
        t = dt.datetime.strptime(base[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
        return t + (float("0." + frac.split("+")[0]) if frac.split("+")[0].isdigit() else 0.0)
    except Exception:
        return None


def window_left(state):
    """Seconds of order window left when the state arrived: phase_ends_at - server_time, else 2 s."""
    a, b = parse_ts(state.get("phase_ends_at")), parse_ts(state.get("server_time"))
    if a is not None and b is not None and 0.3 < a - b < 120:
        return a - b
    return WINDOW_S


def load_key():
    key = os.environ.get("NIGHTSHIFT_KEY")
    if key:
        return key.strip()
    try:
        for line in open(os.path.join(ROOT, ".env")):
            k, _, v = line.strip().partition("=")
            if k.strip().replace("export ", "") == "NIGHTSHIFT_KEY":
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def log(msg):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, "bot.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def save_state(state):
    try:
        with open(os.path.join(LOGDIR, "states.jsonl"), "a") as f:
            f.write(json.dumps(state) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def night_record(rec):
    """logs/nights.jsonl: one line per night for the dashboard. Fields: t, mode, market, day, bot (main/fallback/
    fallback-error), preds {rival id: order or null}, order (final posted), by (safe/safe-late/brain/replan/
    fallback/failed-*), posts / post_attempts (orders sent / HTTP tries with retries), predict_ms (state arrival
    to predictions), decide_ms (brain only), realised_day and realised {truck id: order with sold} =
    what every truck actually did on realised_day (yesterday)."""
    try:
        with open(os.path.join(LOGDIR, "nights.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass


class Api:
    def __init__(self, key):
        self.key = key
        self.api = API            # a practice market moves this to its own base

    def call(self, method, path, body=None, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        url = path if path.startswith("http") else self.api + path
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Authorization": "Bearer " + self.key,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode() or "null")
            except Exception:
                return e.code, None
        except Exception as e:      # network errors, timeouts, bad JSON: never raise into the night
            return 0, {"error": repr(e)[:200]}


def send_order(api, order, t_end, wanted=lambda: True):
    """POST /orders. On a transport failure or 502/503/504 the same order goes again every ~100 ms while the
    window is open (until t_end - 0.1) and wanted() holds (no newer order replaced it). Any other status,
    4xx included (409 day closed, 403 wrecked, 400), is final. Returns (status, response, attempts)."""
    attempts = 0
    while True:
        attempts += 1
        st, resp = api.call("POST", "/orders", order, timeout=max(0.15, min(0.6, t_end - time.time() + 0.1)))
        if st not in RETRY_STATUS or time.time() + 0.1 >= t_end - 0.1 or not wanted():
            return st, resp, attempts
        time.sleep(0.1)
        if not wanted():
            return st, resp, attempts


def read_mode():
    try:
        return json.load(open(CONTROL)).get("mode", "main")
    except (OSError, ValueError, AttributeError):
        return "main"


def clean_order(o, state):
    """A valid order or None."""
    if not isinstance(o, dict):
        return None
    rules = state.get("rules") if isinstance(state.get("rules"), dict) else {}
    spots = rules.get("spots") if isinstance(rules.get("spots"), list) else []
    q = o.get("quality")
    if not isinstance(o.get("spot"), str) or o.get("spot") not in spots or q not in G.COSTS:
        return None
    price = round(min(30.0, max(1.0, G.num(o.get("price"), 10.0))), 2)
    meals = int(max(0, min(G.num(o.get("meals"), 0), G.max_meals(q))))
    return {"spot": o["spot"], "price": price, "quality": q, "meals": meals}


class Bot:
    def __init__(self, api, predictor=None):
        self.api = api
        self.predictor = predictor
        self.brain = Brain(predictor=None)
        self.fallback = Fallback()
        self.done = set()            # (market, day) we have answered
        self.in_night = False        # inside night() (the rival re-fetch waits)
        self.final_answered = False  # answered a night of the final (mode "final"): the re-fetch stops
        self.market = None
        self.auto_fallback = False
        self.track = []              # (expected profit, realised profit) this market
        self.expected = {}           # day -> brain's expected profit for our order

    def night(self, state, t_got):
        """One night: a safe order at once, the brain's order when ready, and with planned counts live one
        re-read of forecast.planned late in the window and a re-order if it changes our answer."""
        mkey = (state.get("mode"), state.get("market_id") or state.get("market"))
        day = state.get("order_for_day")
        if mkey != self.market:
            self.market, self.auto_fallback, self.track, self.expected = mkey, False, [], {}
            log("market %s" % (mkey,))
        self.learn(state)
        mode = "fallback" if (read_mode() == "fallback" or self.auto_fallback) else "main"
        t_end = t_got + window_left(state)
        posted = {"order": None, "by": None, "posts": 0, "attempts": 0, "sent": None, "ack": None}
        lock = threading.Lock()

        def post(order, by):
            with lock:
                if order == posted["order"]:
                    return True
                prev = posted["order"]
                posted["order"], posted["by"] = order, by
                posted["posts"] += 1
            t_sent = time.time()
            st, resp, n = send_order(self.api, order, t_end, lambda: posted["order"] == order)
            with lock:
                posted["attempts"] += n
            log("day %s %s order %s -> %s %s (%.0f ms%s)" % (day, by, order, st, json.dumps(resp)[:120],
                                                             1000 * (time.time() - t_got),
                                                             ", %d tries" % n if n > 1 else ""))
            if st == 200:
                with lock:
                    if posted["order"] == order:
                        posted["sent"], posted["ack"] = t_sent, time.time()
            if st != 200:
                with lock:
                    if posted["order"] == order:
                        posted["order"], posted["by"] = prev, "failed-" + by
            return st == 200

        fb_order = clean_order(self.fallback.decide(state), state)
        riot = bool(riot_risk(sanitize(state)))
        practice = state.get("mode") == "practice"     # practice plays the day on our first order: post once
        if fb_order and not riot and SAFE_FIRST and not practice:
            # a safe order in at once (in its own thread) so the night is never lost; the brain replaces it
            threading.Thread(target=post, args=(fb_order, "safe"), daemon=True).start()
        timer = threading.Timer(max(0.0, min(SAFE_AFTER, t_end - t_got - 0.35) - (time.time() - t_got)),
                                lambda: posted["order"] is None and fb_order and post(fb_order, "safe-late"))
        timer.daemon = True
        if not practice:
            timer.start()
        order, preds = None, {}
        t_pred = t_dec = None
        try:
            if mode == "main":
                preds = self.predictor(state, deadline=min(t_got + PREDICT_BUDGET, t_end - 0.6)) \
                    if self.predictor else {}
                t_pred = time.time()
                order = self.brain.decide(state, predictions=preds)
                t_dec = time.time()
                info = getattr(self.brain, "last_info", None)
                log("day %s preds %s%s" % (day, {k[-4:]: (v and (v["spot"], v["quality"], v["price"], v["meals"]))
                                                for k, v in preds.items()}, self.sources()))
                if info:
                    self.expected[day] = info
                if order is None and posted["order"] is not None:
                    order = self.brain.last_best      # we already have an order in: replace it with the best
            else:
                order = fb_order
                if order:
                    post(order, "fallback")       # answer first, then keep the rival models and brain warm
                preds = self.predictor(state, deadline=min(t_got + PREDICT_BUDGET, t_end - 0.6)) \
                    if self.predictor else {}
                self.brain.observe(state)
                self.brain.pred[day] = dict(preds)       # keeps the trust in each rival's model current
                order = None
            order = clean_order(order, state) if order is not None else None
        except Exception:
            log("decide error: " + traceback.format_exc().replace("\n", " | ")[-600:])
            order = fb_order
            mode = "fallback-error"
        timer.cancel()
        if order is not None and time.time() > t_end - 0.05:
            log("day %s brain order too late (%.0f ms after the window): keeping %s" % (
                day, 1000 * (time.time() - t_end), posted["by"]))
        elif order is not None:
            post(order, "brain")
        elif mode == "main" and posted["order"] is None:
            log("day %s sit out" % day)
        if mode == "main" and (state.get("features") or {}).get("planned_counts") and not practice:
            self.replan(state, preds, posted, post, t_got, t_end)
        self.done.add((mkey, day))
        night_record({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": state.get("mode"), "market": mkey[1],
                      "day": day, "bot": mode, "preds": preds, "order": posted["order"], "by": posted["by"],
                      "predict_ms": round(1000 * (t_pred - t_got)) if t_pred else None,
                      "decide_ms": round(1000 * (t_dec - t_pred)) if t_dec else None,
                      # our last accepted POST: sent / acknowledged, ms after the state arrived; window_ms = the
                      # window left then; margin_ms = window close minus the acknowledgement
                      "posts": posted["posts"], "post_attempts": posted["attempts"],
                      "post_ms": round(1000 * (posted["sent"] - t_got)) if posted["sent"] else None,
                      "ack_ms": round(1000 * (posted["ack"] - t_got)) if posted["ack"] else None,
                      "window_ms": round(1000 * (t_end - t_got)),
                      "margin_ms": round(1000 * (t_end - posted["ack"])) if posted["ack"] else None,
                      "realised_day": (state.get("last_day") or {}).get("day"),
                      "realised": getattr(self.brain, "last_actual", None)})

    def replan(self, state, preds, posted, post, t_got, t_end):
        """Planned counts: re-read tonight's forecast.planned late in the window and re-decide once."""
        day = state.get("order_for_day")
        wait = t_end - REPLAN_BEFORE - time.time()
        if wait > 0:
            time.sleep(wait)
        if t_end - time.time() < 0.25:
            log("day %s replan: no time" % day)
            return
        st, fresh = self.api.call("GET", "/state", timeout=max(0.1, t_end - time.time() - 0.2))
        if st != 200 or not isinstance(fresh, dict) or fresh.get("order_for_day") != day or \
                not isinstance((fresh.get("forecast") or {}).get("planned"), dict):
            log("day %s replan: no planned counts (%s)" % (day, st))
            return
        s2 = dict(state, forecast=dict(state.get("forecast") or {}, planned=fresh["forecast"]["planned"]),
                  _posted=(posted["order"] or {}).get("spot"))
        step = self.brain.p["price_step"]
        self.brain.p["price_step"] = 0.5          # coarser grid: this must be quick
        try:
            o = clean_order(self.brain.decide(s2, predictions=preds) or self.brain.last_best, s2)
        except Exception:
            log("replan error: " + traceback.format_exc().replace("\n", " | ")[-300:])
            o = None
        finally:
            self.brain.p["price_step"] = step
        log("day %s replan planned %s -> %s (%.0f ms left)" % (day, fresh["forecast"]["planned"], o,
                                                               1000 * (t_end - time.time())))
        if o and o != posted["order"] and t_end - time.time() > 0.08:
            post(o, "replan")

    def sources(self):
        src = getattr(self.predictor, "last_source", None)
        return " src %s" % {k[-4:]: v for k, v in src.items()} if src else ""

    def learn(self, state):
        """Compare yesterday's realised profit with what the brain expected; maybe switch to fallback."""
        ld = state.get("last_day") or {}
        you = ld.get("you") or {}
        d = ld.get("day")
        if d in self.expected and you.get("status") == "played":
            # coupons dropped on us are not the brain's mistake: add what they cost back (coupon_loss, else from
            # coupon_share: revenue = sold * price * (1 - share / 2))
            share = min(0.6, max(0.0, G.num(you.get("coupon_share"), 0.0)))
            loss = abs(G.num(you.get("coupon_loss"), 0.0)) or G.num(you.get("revenue"), 0.0) * share / (2 - share)
            self.track.append((self.expected.pop(d), G.num(you.get("profit")) + loss))
            exp = sum(e for e, _ in self.track[-SWITCH_NIGHTS:])
            real = sum(r for _, r in self.track[-SWITCH_NIGHTS:])
            if len(self.track) >= SWITCH_NIGHTS and exp > 300 and real < SWITCH_RATIO * exp and not self.auto_fallback:
                self.auto_fallback = True
                log("AUTO-SWITCH to fallback: last %d nights realised %.0f vs expected %.0f" % (SWITCH_NIGHTS, real, exp))

    def run(self, practice=False):
        version, fails = 0, 0
        while True:
            try:
                st, state = self.api.call("GET", "/state?since=%s" % version, timeout=35)
                t_got = time.time()
                if st == 401:
                    log("401: bad key")
                    time.sleep(10)
                    continue
                if st != 200 or not isinstance(state, dict):
                    fails += 1
                    if fails in (1, 5) or fails % 60 == 0:
                        log("state poll failed x%d: %s %s" % (fails, st, json.dumps(state)[:160]))
                    time.sleep(1)
                    continue
                fails = 0
                version = state.get("version", version)
                if practice and state.get("phase") == "over":
                    you = state.get("you") or {}
                    log("practice over: cash %s; %s" % (you.get("cash"), sorted(
                        ((t.get("truck_name"), t.get("cash")) for t in state.get("trucks") or [] if isinstance(t, dict)),
                        key=lambda x: -G.num(x[1]))))
                    return
                if state.get("phase") == "night" and state.get("orders_open"):
                    mkey = (state.get("mode"), state.get("market_id") or state.get("market"))
                    if not isinstance(state.get("you"), dict):
                        continue                  # showcase without us, or not in this market: idle
                    if (mkey, state.get("order_for_day")) not in self.done:
                        self.in_night = True
                        try:
                            self.night(state, t_got)
                        finally:
                            self.in_night = False
                        if state.get("mode") == "final":
                            self.final_answered = True
                        save_state(state)
                        if practice:
                            version = 0           # the practice day was played on our order: re-read at once
            except KeyboardInterrupt:
                raise
            except Exception:
                log("loop error: " + traceback.format_exc().replace("\n", " | ")[-600:])
                time.sleep(1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", type=int, default=0, metavar="DAYS", help="play one private practice market")
    ap.add_argument("--seed", default=None)
    args = ap.parse_args()
    key = load_key()
    if not key:
        print("Set NIGHTSHIFT_KEY (or put it in .env)", file=sys.stderr)
        sys.exit(1)
    api = Api(key)
    if args.practice:
        body = dict({"days": args.practice}, **({"seed": args.seed} if args.seed else {}))
        st, res = api.call("POST", "/practice", body, timeout=15)
        if st not in (200, 201) or not isinstance(res, dict) or not res.get("base"):
            print("could not start practice: %s %s" % (st, json.dumps(res)[:200]), file=sys.stderr)
            sys.exit(1)
        b = res["base"]
        api.api = b if b.startswith("http") else BASE + b
        log("practice market %s at %s" % (res.get("id"), api.api))
    predictor = None
    if os.environ.get("FTW_NO_RIVALS") != "1":
        try:
            from .rivals import Predictor
            predictor = Predictor(budget=PREDICT_BUDGET)
            log("rival workers: %d" % len(predictor.workers))
            # FTW_REAL_RIVALS (default REAL_DEFAULT; empty = none): run those rivals' real bots (rivals/, under
            # ftw/shim.py) and prefer their orders; the adapters cover every rival and fill in for a late real bot.
            real = [n for n in os.environ.get("FTW_REAL_RIVALS", REAL_DEFAULT).split(",") if n]
            if real:
                from .rivalhost import ROSTER, Hybrid, RivalHost
                host = RivalHost(roster={n: r for n, r in ROSTER.items() if n in real}, budget=PREDICT_BUDGET)
                predictor = Hybrid(host, predictor)
                log("real rival bots: %s" % sorted(host.rivals))
        except Exception:
            log("no rival predictor: " + traceback.format_exc().replace("\n", " | ")[-300:])
    log("Frontrun Fries bot starting against %s" % BASE)
    bot = Bot(api, predictor)
    host = getattr(predictor, "host", None)
    if host is not None and not args.practice and os.environ.get("FTW_REFETCH", "1") == "1":
        threading.Thread(target=refetch_rivals, args=(bot, host), daemon=True).start()
    bot.run(practice=bool(args.practice))


def _et_hhmm():
    try:
        import datetime as dt
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).strftime("%H%M")
    except Exception:
        return time.strftime("%H%M", time.gmtime(time.time() - 4 * 3600))      # EDT on 2026-10-24


def refetch_rivals(bot, host, every=REFETCH_S, until=REFETCH_UNTIL):
    """Until our first night of the final is answered (or `until` ET, HHMM): every `every` s re-fetch the rival repos into
    rivals_live/ and restart any real rival whose snapshot changed, so a push just before the lock still counts.
    Never during a night; never fatal."""
    try:
        from . import fetch_rivals
    except Exception:
        return
    while True:
        time.sleep(every)
        try:
            if bot.final_answered or _et_hhmm() >= until:
                log("rival re-fetch: stopped (%s)" % ("first final night answered" if bot.final_answered
                                                       else "past " + until))
                return
            if bot.in_night:
                continue
            fetch_rivals.main(["--total", "20"])
            if bot.in_night or bot.final_answered:
                continue
            changed = host.refresh()
            if changed:
                log("rival re-fetch: restarted %s on new snapshots" % sorted(changed))
        except Exception:
            log("rival re-fetch error: " + traceback.format_exc().replace("\n", " | ")[-300:])


if __name__ == "__main__":
    main()
