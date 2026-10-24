"""Local benchmark: each candidate strategy takes slot 0 in the same seeded markets
against several rival fields. Reports mean/sd profit, waste, missed orders, runtime.

    python3 tools/bench.py --markets 10 --seeds dev
    python3 tools/bench.py --markets 10 --seeds test --cand adaptive,crowd

Local results validate robustness, not exact server behaviour.
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw.baselines import (Adaptive, CrowdFollower, Fixed, PerCapChaser, Starter,  # noqa: E402
                           Undercutter, house_bots)
from ftw.localsim import Market  # noqa: E402


def field(name, seed):
    if name == "house":
        return house_bots()
    if name == "crowd":
        return [("c1", "Crowd std", CrowdFollower("standard", 10.0, 133)),
                ("c2", "Crowd prem", CrowdFollower("premium", 12.5, 90)),
                ("c3", "Crowd std2", CrowdFollower("standard", 9.5, 120)),
                ("c4", "Crowd cheap", CrowdFollower("cheap", 6.5, 200))]
    if name == "mixed":
        return [("m1", "Fixed union", Fixed("union", "standard", 10.0, 120)),
                ("m2", "Crowd prem", CrowdFollower("premium", 12.0, 90)),
                ("m3", "Undercut", Undercutter("standard", 10.5)),
                ("m4", "PerCap prem", PerCapChaser("premium", 12.0)),
                ("m5", "Starter", Starter(seed))]
    if name == "premium":
        return [("p1", "Prem highst", Fixed("highst", "premium", 13.0, 90)),
                ("p2", "PerCap prem", PerCapChaser("premium", 12.0)),
                ("p3", "Crowd prem", CrowdFollower("premium", 12.5, 90)),
                ("p4", "Undercut prem", Undercutter("premium", 13.0, 8.0))]
    if name == "smart":
        return [("a1", "Adaptive A", Adaptive(seed=seed * 7 + 1)),
                ("a2", "Adaptive B", Adaptive(seed=seed * 7 + 2)),
                ("a3", "Adaptive C", Adaptive(seed=seed * 7 + 3)),
                ("s1", "PerCap std", PerCapChaser("standard", 10.0)),
                ("s2", "Crowd prem", CrowdFollower("premium", 12.0, 90))]
    raise ValueError(name)


def candidate(name, seed, cfg=None):
    if name == "starter":
        return Starter(seed)
    if name == "fixed_std":
        return Fixed("union", "standard", 10.0, 133)
    if name == "fixed_prem":
        return Fixed("oval", "premium", 12.5, 90)
    if name == "crowd":
        return CrowdFollower("standard", 10.0, 133)
    if name == "adaptive":
        return Adaptive(seed=seed, cfg=cfg)
    raise ValueError(name)


def run_one(job):
    cand, fname, seed_tag, i, cfg = job
    seed = "%s-%d" % (seed_tag, i)
    strat = candidate(cand, i, cfg)
    strats = [("me", cand, strat)] + field(fname, i)
    t0 = time.time()
    st = Market(seed, strats).run()["me"]
    ms = getattr(strat, "ms", [])
    return {"cand": cand, "field": fname, "seed": seed, "profit": st["profit"], "cooked": st["cooked"],
            "sold": st["sold"], "missed": st["missed"], "invalid": st["invalid"],
            "max_ms": max(ms) if ms else 0.0, "mean_ms": statistics.mean(ms) if ms else 0.0,
            "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=8)
    ap.add_argument("--seeds", default="dev")
    ap.add_argument("--cand", default="starter,fixed_std,fixed_prem,crowd,adaptive")
    ap.add_argument("--fields", default="house,crowd,mixed,premium,smart")
    ap.add_argument("--cfg", default="{}", help="JSON overrides for the adaptive config")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    jobs = [(c, f, args.seeds, i, cfg) for c in args.cand.split(",") for f in args.fields.split(",")
            for i in range(args.markets)]
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f)
    fields = args.fields.split(",")
    print("mean profit per 30-day market (sd), %d markets per cell, seeds=%s" % (args.markets, args.seeds))
    print("%-11s" % "cand" + "".join("%17s" % f for f in fields) + "%11s%8s%7s%8s" % ("overall", "waste", "miss", "maxms"))
    for c in args.cand.split(","):
        line = "%-11s" % c
        allp = []
        cooked = sold = missed = 0
        mx = 0.0
        for f in fields:
            ps = [r["profit"] for r in rows if r["cand"] == c and r["field"] == f]
            allp += ps
            sd = statistics.stdev(ps) if len(ps) > 1 else 0.0
            line += "%11.0f (%4.0f)" % (statistics.mean(ps), sd)
            for r in rows:
                if r["cand"] == c and r["field"] == f:
                    cooked += r["cooked"]
                    sold += r["sold"]
                    missed += r["missed"]
                    mx = max(mx, r["max_ms"])
        waste = 1 - sold / cooked if cooked else 0.0
        print(line + "%11.0f%7.0f%%%7d%8.0f" % (statistics.mean(allp), 100 * waste, missed, mx))


if __name__ == "__main__":
    main()
