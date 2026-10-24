"""Benchmark strategies on real practice markets (server engine, house bots).

    python3 tools/practice_bench.py --seeds pb --markets 4 --cand starter,fixed_std,fixed_prem,crowd,adaptive

Same seed => same weather, events and crowds for every candidate. Results are
appended to state/practice_bench.jsonl. Throttled; uses one practice market at a time.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw.api import Client, NetError, ROOT  # noqa: E402
from ftw.rules import Rules  # noqa: E402
from bench import candidate  # noqa: E402


def get_state(c, base):
    for attempt in range(8):
        try:
            s, st = c.state(base=base, timeout=20)
            if s == 200:
                return st
            if s == 404:
                return {}
        except NetError:
            pass
        time.sleep(1 + 2 * attempt)
    return {}


def play(c, cand, seed, cfg):
    s, r = c.practice(seed=seed, days=30, rivals=3)
    if s not in (200, 201):
        raise RuntimeError("practice %s: %s" % (s, r.get("error")))
    base = r["base"]
    strat = candidate(cand, abs(hash(seed)) % 1000, cfg)
    st = get_state(c, base)
    calib = []
    rejected = missed = 0
    while st.get("phase") == "night":
        if hasattr(strat, "observe"):
            strat.observe(st)
        o = Rules(st.get("rules")).validate(strat.decide(st))
        exp = getattr(getattr(strat, "brain", None), "last_debug", {}).get("exp_profit")
        res = None
        for attempt in range(5):
            try:
                code, res = c.order(o, base=base, timeout=15)
            except NetError:
                time.sleep(1 + attempt)
                continue
            if code == 200:
                break
            rejected += 1
            time.sleep(1)
            res = None
        if not res:
            missed += 1
            st = get_state(c, base)
            continue
        st = res["state"]
        mine = (st.get("last_day") or {}).get("you") or {}
        if exp is not None:
            calib.append((exp, mine.get("profit", 0.0)))
        time.sleep(0.25)
    if hasattr(strat, "observe"):
        strat.observe(st)
    trucks = {t["truck_name"]: t["cash"] for t in st.get("trucks") or []}
    ms = getattr(strat, "ms", [])
    return {"cand": cand, "seed": seed, "cash": (st.get("you") or {}).get("cash"), "trucks": trucks,
            "rejected": rejected, "missed": missed, "calib": calib, "max_ms": max(ms) if ms else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="pb")
    ap.add_argument("--markets", type=int, default=3)
    ap.add_argument("--cand", default="starter,fixed_std,fixed_prem,crowd,adaptive")
    ap.add_argument("--cfg", default="{}")
    args = ap.parse_args()
    c = Client()
    cfg = json.loads(args.cfg)
    out = os.path.join(ROOT, "state", "practice_bench.jsonl")
    rows = []
    for i in range(args.markets):
        for cand in args.cand.split(","):
            seed = "%s-%d" % (args.seeds, i)
            try:
                r = play(c, cand, seed, cfg)
            except Exception as e:
                print("fail", cand, seed, e, flush=True)
                continue
            r["cfg"] = cfg
            rows.append(r)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print("%-10s %-8s cash %9.2f  rejected %d missed %d" % (cand, seed, r["cash"], r["rejected"], r["missed"]),
                  flush=True)
    for cand in args.cand.split(","):
        ps = [r["cash"] for r in rows if r["cand"] == cand]
        if ps:
            print("%-10s mean %9.2f sd %8.2f n=%d" % (cand, statistics.mean(ps),
                                                       statistics.stdev(ps) if len(ps) > 1 else 0, len(ps)))


if __name__ == "__main__":
    main()
