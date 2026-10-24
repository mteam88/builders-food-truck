"""Frontrun Fries v3: a simple High Street camper. Python 3.9+, standard library only.

    ./run.sh                    # live game, then the final (leave it running)
    python3 bot.py --practice   # private practice market

v2 modeled every spot and every rival and lost about $800 on days 2-3. v3 parks at
High Street every night with premium food and lets the price follow sales.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# v3 strategy: the same order every night, only the price moves.
SPOT = "highst"        # High Street: the biggest crowd on campus
QUALITY = "premium"    # best value for customers, +5 reputation a day
MEALS = 90             # as many premium meals as the $500 budget buys
START_PRICE = 12.00    # first night at the spot, and at the start of every market
STEP = 0.25            # the price moves a quarter at a time
SLOW = 0.60            # sold under 60% of the meals: one step cheaper
FLOOR = 10.50          # never cheaper than this
CAP = 14.00            # never dearer than this

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "data")
LIVE = "/api/v1"


def load_env():
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


load_env()
URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")


def decide(state):
    """Tomorrow's order: same spot, food and meals every night; the price follows yesterday's sales."""
    last = state.get("last_day") or {}
    mine = last.get("you") or {}
    camped = (mine.get("spot") == SPOT and mine.get("quality") == QUALITY and mine.get("meals")
              and last.get("day") == state.get("day"))
    if not camped:
        price = START_PRICE                  # first night here (or a new market): the usual price
    elif mine.get("sold", 0) >= mine["meals"]:
        price = mine["price"] + STEP         # sold out: they would have paid more
    elif mine.get("sold", 0) < SLOW * mine["meals"]:
        price = mine["price"] - STEP         # slow day: too dear
    else:
        price = mine["price"]                # in between: hold
    price = min(CAP, max(FLOOR, round(price / STEP) * STEP))
    return {"spot": SPOT, "price": price, "quality": QUALITY, "meals": meals(state)}


def meals(state):
    """All the meals the budget buys, capped at MEALS."""
    rules = state.get("rules") or {}
    try:
        return min(MEALS, int(rules["budget"] / rules["costs"][QUALITY] + 1e-9))
    except (KeyError, TypeError, ZeroDivisionError):
        return MEALS


def call(method, path, body=None, timeout=35):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"error": str(e)}


def day_key(state):
    return state.get("mode"), state.get("market"), (state.get("last_day") or {}).get("day")


def log(name, state):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, name), "a") as f:
        f.write(json.dumps(state) + "\n")


def last_logged(name):
    """The day of the last state in the log, so a restart doesn't log it twice."""
    try:
        with open(os.path.join(LOG_DIR, name), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            return day_key(json.loads(f.read().splitlines()[-1]))
    except Exception:
        return None


def play(base, logname):
    """Long-poll the state and send one order per night. Works for live, final and practice."""
    version, ordered, reported, rejected = 0, None, None, {}
    logged, connected = last_logged(logname), False
    while True:
        try:
            status, state = call("GET", base + "/state?since=%d" % version)
            if status != 200:
                if status != 429:
                    print("error %d: %s" % (status, state.get("error")))
                if status == 404 and base != LIVE:
                    return None              # practice market expired
                time.sleep(1)
                continue
            if not connected:
                connected = True
                print("[%s] connected: day %s, %s, cash $%.2f" % (
                    state.get("mode"), state.get("day"), state.get("phase"), state["you"]["cash"]))
            version = state["version"]
            if state.get("last_day") and day_key(state) != logged:
                logged = day_key(state)
                log(logname, state)
            if state["phase"] == "over" and state["mode"] == "practice":
                report(state)
                return state
            key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
            if state["phase"] == "night" and state.get("orders_open") and key != ordered and rejected.get(key, 0) < 3:
                if key != reported:
                    reported = key
                    report(state)
                t0 = time.time()
                order = decide(state)
                status, res = call("POST", base + "/orders", order, timeout=3)
                if status == 200:
                    ordered = key
                    print("  ordered for day %s in %.2fs: %s" % (res.get("for_day"), time.time() - t0, order))
                    if "state" in res:       # practice plays the day right away
                        version = 0
                else:
                    print("  order rejected (%d): %s" % (status, res.get("error")))
                    if 400 <= status < 500 and status != 429:
                        rejected[key] = rejected.get(key, 0) + 1   # a real "no": don't keep asking
                    version = 0              # look again now and retry
                    time.sleep(1 if status == 429 else 0.25)
        except Exception as e:
            print("loop error (%s: %s), retrying in 1s" % (type(e).__name__, e))
            version = 0
            time.sleep(1)


def report(state):
    last = state.get("last_day") or {}
    mine = last.get("you") or {}
    where = state["mode"] + (" market %s" % state["market"] if state["mode"] == "final" else "")
    if mine.get("spot"):
        print("[%s] day %s: %s at %s $%s, sold %s/%s, profit $%.2f, cash $%.2f" % (
            where, last.get("day"), mine.get("quality"), mine.get("spot"), mine.get("price"),
            mine.get("sold"), mine.get("meals"), mine.get("profit", 0), state["you"]["cash"]))
    elif last.get("day"):
        print("[%s] day %s: sat out, cash $%.2f" % (where, last.get("day"), state["you"]["cash"]))
    else:
        print("[%s] night before day %s, cash $%.2f" % (where, state.get("order_for_day"), state["you"]["cash"]))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="Frontrun Fries v3")
    ap.add_argument("--practice", action="store_true", help="play a private practice market")
    ap.add_argument("--seed", help="practice seed, for repeatable markets")
    ap.add_argument("--days", type=int, default=30, help="practice market length")
    ap.add_argument("--rivals", type=int, default=3, help="practice house bots")
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY (or put it in .env).")
    if args.practice:
        body = {"days": args.days, "rivals": args.rivals}
        if args.seed:
            body["seed"] = args.seed
        status, res = call("POST", LIVE + "/practice", body)
        if status not in (200, 201):
            sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
        print("practice market %s started" % res["id"])
        final = play(res["base"], "practice-%s.jsonl" % res["id"])
        if final:
            print("practice over: cash $%.2f" % final["you"]["cash"])
            for t in sorted(final["trucks"], key=lambda t: -t["cash"]):
                print("  %-24s $%10.2f" % (t["truck_name"], t["cash"]))
    else:
        print("playing the live game at %s (Ctrl+C to stop)" % URL)
        play(LIVE, "live.jsonl")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
