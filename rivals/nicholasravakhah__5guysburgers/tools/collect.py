"""Calibration data collector: plays practice markets with randomized orders and
logs every day's full result to state/calib.jsonl.

Orders are chosen to measure demand, not to make money: we usually park where no
house bot parked yesterday and stock the maximum so sales are not censored.
Throttled to roughly two requests a second.

    python3 tools/collect.py --markets 20 --prefix calA
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw.api import Client, NetError, ROOT  # noqa: E402

CAP = {"cheap": 200, "standard": 133, "premium": 90}
VAL = {"cheap": 8, "standard": 12, "premium": 16}


def choose(st, rng, only=None):
    spots = st["rules"]["spots"]
    if only:
        spot = rng.choice(only)
        return _order(st, rng, spot)
    ld = st.get("last_day") or {}
    occupied = {s for s, v in (ld.get("spots") or {}).items()
                if any(t["id"] != st["you"]["id"] for t in v.get("trucks", []))}
    free = [s for s in spots if s not in occupied] or spots
    events = [e.get("spot") for e in (st.get("forecast") or {}).get("events", []) if e.get("spot") in spots]
    if events and rng.random() < 0.6:
        spot = rng.choice(events)
    else:
        spot = rng.choice(free) if rng.random() < 0.85 else rng.choice(spots)
    return _order(st, rng, spot)


def _order(st, rng, spot):
    q = rng.choice(["cheap", "standard", "premium", "premium"])
    rep = st["you"]["reputation"].get(spot, 50)
    vmax = VAL[q] * (0.7 + 0.006 * rep)
    lo = {"cheap": 2.0, "standard": 4.0, "premium": 5.0}[q]
    price = round(rng.uniform(lo, min(vmax - 0.01, 22.0)), 2)
    return {"spot": spot, "price": price, "quality": q, "meals": CAP[q]}


def get_state(c, base):
    for attempt in range(6):
        try:
            s, st = c.state(base=base, timeout=20)
            if s == 200:
                return st
        except NetError as e:
            print("net", e, flush=True)
        time.sleep(2 + 3 * attempt)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=10)
    ap.add_argument("--prefix", default="cal")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--spots", default="", help="comma list: only park at these spots")
    args = ap.parse_args()
    c = Client()
    path = os.path.join(ROOT, "state", "calib.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for m in range(args.markets):
        seed = "%s-%d" % (args.prefix, m)
        rng = random.Random(seed)
        try:
            s, r = c.practice(seed=seed, days=args.days, rivals=args.rivals)
        except NetError as e:
            print("net", e, flush=True)
            time.sleep(10)
            continue
        if s not in (200, 201):
            print("practice failed", s, r.get("error"), flush=True)
            time.sleep(10)
            continue
        base = r["base"]
        st = get_state(c, base)
        while st.get("phase") == "night":
            order = choose(st, rng, [x for x in args.spots.split(",") if x])
            prev_fc = st.get("forecast")
            prev_rep = dict(st["you"]["reputation"])
            prev_trucks = st.get("trucks")
            try:
                s, res = c.order(order, base=base, timeout=15)
            except NetError as e:
                print("net", e, flush=True)
                time.sleep(3)
                st = get_state(c, base)
                continue
            if s != 200:
                print("order", s, res.get("error"), flush=True)
                time.sleep(3)
                st = get_state(c, base)
                continue
            st = res["state"]
            rec = {"seed": seed, "rivals": args.rivals, "forecast": prev_fc, "rep_before": prev_rep,
                   "trucks_before": prev_trucks, "order": order, "last_day": st.get("last_day"),
                   "trucks_after": st.get("trucks")}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            time.sleep(0.45)
        print("market", seed, "done, cash", st.get("you", {}).get("cash"), flush=True)


if __name__ == "__main__":
    main()
