"""Bot main loop for live, final and practice markets.

Reliability rules:
  * one order per (mode, market, order_for_day); day numbers repeat across final markets
  * every day's results are ingested once (Brain.processed, persisted across restarts)
  * compute deadline derived from the server's own clock (phase_ends_at - server_time)
  * logging and disk writes happen on a background thread, after the order is sent
  * live/final memory and practice memory are separate files
  * a file lock prevents two live bots fighting over the same truck
"""
import datetime as dt
import fcntl
import json
import os
import queue
import subprocess
import sys
import threading
import time

from .api import API, Client, NetError, ROOT
from .rules import Rules
from .strategy import Brain, choose_vote

STATE_DIR = os.path.join(ROOT, "state")
LOG_DIR = os.path.join(ROOT, "logs")


def _parse_ts(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


class Writer(threading.Thread):
    """Background writer: JSON-lines logs and atomic memory snapshots."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()

    def log(self, path, rec):
        self.q.put(("log", path, rec))

    def snapshot(self, path, obj):
        self.q.put(("snap", path, obj))

    def run(self):
        while True:
            kind, path, obj = self.q.get()
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if kind == "log":
                    with open(path, "a") as f:
                        f.write(json.dumps(obj, default=str) + "\n")
                else:
                    tmp = path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(obj, f)
                    os.replace(tmp, path)
            except Exception as e:  # never crash the bot over disk problems
                print("  (write failed: %s)" % e, file=sys.stderr)
            finally:
                self.q.task_done()

    def flush(self, timeout=5.0):
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.02)


def acquire_lock(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    f = open(os.path.join(STATE_DIR, name + ".lock"), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("Another %s bot is already running (lock %s). Stop it first." % (name, f.name))
    f.write(str(os.getpid()))
    f.flush()
    return f


def load_memory(path):
    try:
        with open(path) as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


class Bot:
    def __init__(self, client, base=API, memory_path=None, log_name="live", cfg=None, seed=None,
                 dry_run=False, quiet=False):
        self.client = client
        self.base = base
        self.memory_path = memory_path
        self.dry_run = dry_run
        self.quiet = quiet
        mem = load_memory(memory_path) if memory_path else {}
        seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
        self.brain = Brain(cfg=cfg, memory=mem, seed=seed)
        self._restore_market(mem.get("market"))
        self.writer = Writer()
        self.writer.start()
        self.log_path = os.path.join(LOG_DIR, log_name + ".jsonl")
        self.ordered = set()
        self.reported = set()
        self.sent = {}
        self.foreign = set()
        self.voted = set()
        self.stats = {"orders": 0, "rejected": 0, "net_errors": 0, "decide_ms": []}

    # ---- persistence of per-market progress so restarts do not double-count days
    def _restore_market(self, m):
        if not isinstance(m, dict) or not m.get("key"):
            return
        b = self.brain
        b.market_key = tuple(m["key"])
        b.processed = set(int(d) for d in m.get("processed", []) if isinstance(d, int))
        b.forecasts = {int(k): v for k, v in (m.get("forecasts") or {}).items()}
        b.last_days = {int(k): v for k, v in (m.get("last_days") or {}).items()}
        for tid, last in (m.get("last_spots") or {}).items():
            if tid in b.rivals:
                b.rivals[tid].last_spot = last

    def _save(self):
        if not self.memory_path:
            return
        b = self.brain
        mem = b.memory()
        recent = sorted(b.last_days)[-2:]
        mem["market"] = {"key": list(b.market_key) if b.market_key else None,
                         "processed": sorted(b.processed),
                         "forecasts": {str(k): v for k, v in b.forecasts.items() if k >= (max(b.forecasts) - 2 if b.forecasts else 0)},
                         "last_days": {str(k): b.last_days[k] for k in recent},
                         "last_spots": {tid: r.last_spot for tid, r in b.rivals.items() if r.last_spot}}
        self.writer.snapshot(self.memory_path, mem)

    def say(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    # ---- main loop
    def run(self, max_markets=None, stop_when_over=False):
        version = None
        backoff = 1.0
        state = None
        while True:
            if state is None:
                try:
                    status, state = self.client.state(base=self.base, since=version, timeout=35.0)
                except NetError as e:
                    self.stats["net_errors"] += 1
                    self.say("network error (%s), retry in %.0fs" % (e, backoff))
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    state = None
                    continue
                backoff = 1.0
                if status == 429:
                    time.sleep(2.0)
                    state = None
                    continue
                if status == 401:
                    sys.exit("401: missing or wrong NIGHTSHIFT_KEY")
                if status == 404:
                    if stop_when_over:
                        self.say("404: market not found (expired?)")
                        return None
                    time.sleep(5.0)
                    state = None
                    continue
                if status != 200:
                    self.say("state error %d: %s" % (status, state.get("error")))
                    time.sleep(3.0 if status != 403 else 10.0)
                    state = None
                    continue
            if not isinstance(state, dict):
                state = None
                continue
            v = state.get("version")
            version = v if isinstance(v, int) else version
            self.brain.observe(state)
            self._report_day(state)
            self._check_foreign_order(state)
            self._maybe_vote(state)
            self._maybe_staff(state)
            self._maybe_coupon(state)
            self._maybe_self_update(state)
            if state.get("phase") == "over":
                self._save()
                self.writer.flush()
                if stop_when_over:
                    return state
            nxt = self._maybe_order(state)
            state = nxt  # practice returns the next state with the order; otherwise poll

    def _maybe_self_update(self, state):
        """Live game only: every few minutes during the day phase (our order is locked), check GitHub
        for newer code; if found, flush and exit so run.sh pulls it and restarts the bot.
        Never runs in the final or in practice, so the final plays on settled code."""
        if self.dry_run or state.get("mode") != "live" or state.get("phase") != "day":
            return
        if os.environ.get("NS_NO_SELF_UPDATE"):
            return
        now = time.time()
        if now - getattr(self, "_last_update_check", 0.0) < 240:
            return
        self._last_update_check = now
        try:
            def git(*a):
                return subprocess.run(["git"] + list(a), cwd=ROOT, capture_output=True, text=True, timeout=20)
            if git("fetch", "-q", "origin").returncode != 0:
                return
            head = git("rev-parse", "HEAD").stdout.strip()
            remote = git("rev-parse", "@{u}").stdout.strip()
            if head and remote and head != remote and git("merge-base", "--is-ancestor", head, remote).returncode == 0:
                self.say("new code on GitHub (%s -> %s): restarting to pick it up" % (head[:7], remote[:7]))
                self._save()
                self.writer.flush()
                sys.exit(0)
        except Exception as e:  # never let updating break the bot
            self.say("update check failed: %r" % (e,))

    def _maybe_staff(self, state):
        """Staff rule: keep a crew big enough for our largest order at a wage rivals can't easily
        poach; raise the wage if we end up short. Settings are saved on the truck (carry into the final)."""
        if self.dry_run:
            return
        try:
            feats = state.get("features") or {}
            you = state.get("you") or {}
            st = you.get("staff") if isinstance(you.get("staff"), dict) else None
            if not feats.get("staff") or st is None:
                return
            cfg = self.brain.cfg
            want_n = int(cfg.get("staff_target", 3))
            want_w = float(cfg.get("staff_wage", 9.0))
            cur_w = float(st.get("wage") or 0)
            if (st.get("employees") or 0) < want_n and st.get("target") == want_n and cur_w >= want_w:
                want_w = min(20.0, cur_w + 2.0)  # short-handed: someone poached us, pay more
            want_w = max(want_w, cur_w)
            key = ("staff", state.get("mode"), state.get("market"), want_n, want_w)
            if key in self.voted or (st.get("target") == want_n and abs(cur_w - want_w) < 0.01):
                return
            self.voted.add(key)
            status, res = self.client.staff(want_n, want_w, base=self.base)
            self.say("staff -> %d at $%.2f (%s)" % (want_n, want_w, status))
        except Exception as e:
            self.say("staff error: %r" % (e,))

    def _maybe_coupon(self, state):
        """Coupons rule: once per night, $20 coupon on the truck just ahead of us (or the closest chaser if we lead)."""
        if self.dry_run or not self.brain.cfg.get("coupon_rule"):
            return
        try:
            if not (state.get("features") or {}).get("coupons") or state.get("phase") != "night":
                return
            key = ("coupon", state.get("mode"), state.get("market"), state.get("order_for_day"))
            if key in self.voted:
                return
            you = state.get("you") or {}
            # coupon_doubling: each drop costs 2x our last one ($20, $40, $80...) - only buy cheap ones
            rc = ((state.get("rules") or {}).get("coupons") or {})
            cost = you.get("coupon_cost", rc.get("cost", 20))
            try:
                cost = float(cost)
            except (TypeError, ValueError):
                cost = 20.0
            if cost > float(self.brain.cfg.get("coupon_max_cost", 20.0)):
                return
            me = you.get("id")
            others = [t for t in (state.get("trucks") or []) if isinstance(t, dict) and t.get("id") and t.get("id") != me]
            if not others:
                return
            mine = you.get("cash") or 0
            ahead = [t for t in others if (t.get("cash") or 0) > mine]
            # race the truck just ahead of us; if we lead, slow the closest chaser
            lead = min(ahead, key=lambda t: t.get("cash") or 0) if ahead else max(others, key=lambda t: t.get("cash") or 0)
            if abs((lead.get("cash") or 0) - mine) < float(self.brain.cfg.get("coupon_min_lead", 0.0)):
                return
            self.voted.add(key)
            status, res = self.client.coupon(lead["id"], base=self.base)
            self.say("coupon on %s (%s)" % (lead.get("truck_name"), status))
        except Exception as e:
            self.say("coupon error: %r" % (e,))

    def _maybe_vote(self, state):
        """Vote automatically in campus elections for the candidate that helps our home spot most."""
        if self.dry_run:
            return
        try:
            el = ((state.get("politics") or {}).get("election") or {})
            if not el.get("candidates"):
                return
            pick = choose_vote(state, self.brain.cfg)
            if not pick:
                return
            current = (state.get("you") or {}).get("vote")
            if isinstance(current, dict):
                current = current.get("candidate")
            if current is None:
                current = (el.get("votes") or {}).get(self.brain.my_id)
            key = (el.get("id"), pick)
            if current == pick or key in self.voted:
                return
            self.voted.add(key)
            status, res = self.client.vote(pick, base=self.base)
            self.say("vote %s in election %s -> %s" % (pick, el.get("id"), status))
        except Exception as e:  # never let voting break ordering
            self.say("vote error: %r" % (e,))

    def _maybe_order(self, state):
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if not (state.get("phase") == "night" and state.get("orders_open") and key[2] is not None):
            return None
        if key in self.ordered:
            return None
        rules = Rules(state.get("rules"))
        # time left in the night, by the server's clock
        left = None
        end, now = _parse_ts(state.get("phase_ends_at")), _parse_ts(state.get("server_time"))
        if end and now:
            left = end - now
        budget = self.brain.cfg["deadline"]
        if left is not None:
            budget = max(0.05, min(budget, 0.3 * left))
        t0 = time.perf_counter()
        order = self.brain.decide(state, deadline=budget)
        ms = (time.perf_counter() - t0) * 1000
        self.stats["decide_ms"].append(ms)
        if self.dry_run:
            self.ordered.add(key)
            self.say("  [dry-run] would order for %s: %s (%.0f ms)" % (key, order, ms))
            self._log_decision(state, order, None, ms)
            return None
        send_deadline = time.perf_counter() + (max(0.3, left - 0.2 - ms / 1000) if left is not None else 10.0)
        res, status = None, None
        attempt = 0
        while attempt < 4 and time.perf_counter() < send_deadline:
            attempt += 1
            try:
                status, res = self.client.order(order, base=self.base,
                                                timeout=max(0.5, min(5.0, send_deadline - time.perf_counter())))
            except NetError as e:
                self.stats["net_errors"] += 1
                self.say("  order network error (%s), retrying" % e)
                time.sleep(0.15)
                continue
            if status == 200:
                break
            if status == 400:
                self.stats["rejected"] += 1
                self.say("  order rejected 400: %s; sending fallback" % res.get("error"))
                order = rules.validate(self.brain._fallback(state, rules))
                continue
            if status == 429:
                time.sleep(0.5)
                continue
            if status == 401:
                sys.exit("401: missing or wrong NIGHTSHIFT_KEY")
            # 403 / 404 / 409: wrong time or market; re-read state
            self.say("  order refused %s: %s" % (status, res.get("error") if res else ""))
            break
        if status == 200:
            self.ordered.add(key)
            self.sent[key] = dict(order)
            self.stats["orders"] += 1
            dbg = self.brain.last_debug
            self.say("  ordered %s day %s: %s %s $%.2f x%d  (exp $%s, %.0f ms)" % (
                key[0], res.get("for_day", key[2]), order["spot"], order["quality"], order["price"],
                order["meals"], dbg.get("exp_profit"), ms))
            self._log_decision(state, order, res, ms)
            self._save()
            nxt = res.get("state") if isinstance(res, dict) else None
            return nxt if isinstance(nxt, dict) else None
        return None

    def _check_foreign_order(self, state):
        """Warn (never fight) if someone else replaced our order with the same key."""
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        mine = self.sent.get(key)
        cur = (state.get("you") or {}).get("order")
        if not mine or not isinstance(cur, dict):
            return
        same = all(cur.get(k) == mine.get(k) for k in ("spot", "quality", "meals")) and \
            abs(float(cur.get("price") or 0) - mine["price"]) < 0.005
        if not same and key not in self.foreign:
            self.foreign.add(key)
            self.say("  !!! WARNING: our order for %s was replaced by another client: %s (we sent %s). "
                     "Only ONE machine should run the live bot." % (key, cur, mine))

    def _log_decision(self, state, order, res, ms):
        self.writer.log(self.log_path, {
            "t": time.time(), "mode": state.get("mode"), "market": state.get("market"),
            "day": state.get("order_for_day"), "order": order, "ms": round(ms, 1),
            "debug": self.brain.last_debug, "forecast": state.get("forecast"),
            "ok": bool(res) or self.dry_run})

    def _report_day(self, state):
        ld = state.get("last_day") if isinstance(state.get("last_day"), dict) else {}
        mine = ld.get("you") if isinstance(ld.get("you"), dict) else {}
        k = ("report", state.get("mode"), state.get("market"), ld.get("day"))
        if not ld.get("day") or k in self.reported:
            return
        self.reported.add(k)
        you = state.get("you") or {}
        where = "%s m%s" % (state.get("mode"), state.get("market"))
        if mine.get("spot"):
            self.say("[%s] day %s: %s %s $%s sold %s/%s profit $%.2f cash $%.2f" % (
                where, ld.get("day"), mine.get("spot"), mine.get("quality"), mine.get("price"),
                mine.get("sold"), mine.get("meals"), mine.get("profit") or 0, you.get("cash") or 0))
        self.writer.log(self.log_path, {"t": time.time(), "mode": state.get("mode"), "market": state.get("market"),
                                        "result_day": ld.get("day"), "you": mine, "cash": you.get("cash"),
                                        "last_day": ld, "trucks": state.get("trucks")})
