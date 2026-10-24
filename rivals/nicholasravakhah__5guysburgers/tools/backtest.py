"""Backtest strategies on the REAL game engine (practice markets), final-style.

Every strategy plays the same seeded practice markets (same crowds, weather and rival
house bots), so differences come from the strategy, not luck. Reports mean profit per
30-day market (the final's ranking metric), worst market, spread and missed orders.

    python3 tools/backtest.py                       # all strategies, 4 markets each
    python3 tools/backtest.py --markets 6 --only camp,claim
    python3 tools/backtest.py --list

Needs NIGHTSHIFT_KEY (export it or put it in .env). Does not touch the live bot or its memory.
"""
import argparse
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw.api import Client  # noqa: E402
from ftw.runner import Bot  # noqa: E402
from ftw.strategy import STYLES  # noqa: E402

CAMP = STYLES["camp"]

# Every approach we tried tonight, as optimizer settings.
STRATEGIES = {
    # what runs now: camp a home spot, game-day/late-night rules, premium guard, riot guard
    "camp": dict(CAMP),
    # claim a spot early and hold it harder (reputation pays for 30 days)
    "claim": dict(CAMP, rep_lambda=3.5, rep_horizon=15.0, rep_return=1.0),
    # undercut: price a quarter below the grid to win close deals at contested spots
    "undercut": dict(CAMP, price_offset=-0.26),
    # cheap pricing (Nick's earlier style)
    "cheap": dict(STYLES["cheap"]),
    # no special rules: pure optimizer (checks the rules actually help)
    "plain": dict(CAMP, gameday_rule=False, latenight_rule=False, premium_guard=None),
    # cook more: bigger meal counts on top of the optimum
    "volume": dict(CAMP, meals_mult=1.2),
    # no camping, chase the best spot each day
    "roam": {},
}


def run_market(client, cfg, seed, days, rivals, mem_path):
    status, res = client.practice(seed=seed, days=days, rivals=rivals)
    if status not in (200, 201):
        raise RuntimeError("practice start failed (%d): %s" % (status, res.get("error")))
    bot = Bot(client, base=res["base"], memory_path=mem_path, log_name="backtest", quiet=True, cfg=cfg)
    t0 = time.time()
    final = bot.run(stop_when_over=True) or {}
    bot.writer.flush()
    you = final.get("you") or {}
    trucks = sorted(final.get("trucks") or [], key=lambda t: -(t.get("cash") or 0))
    rank = next((i + 1 for i, t in enumerate(trucks) if t.get("id") == you.get("id")), None)
    return {"profit": you.get("cash") or 0.0, "rank": rank, "n": len(trucks),
            "missed": bot.stats.get("rejected", 0), "secs": time.time() - t0,
            "max_ms": max(bot.stats["decide_ms"]) if bot.stats["decide_ms"] else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=4)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=6)
    ap.add_argument("--seed", default="bt")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in STRATEGIES.items():
            print("%-9s %s" % (k, v))
        return
    client = Client()
    if not client.has_key:
        sys.exit("Set NIGHTSHIFT_KEY (export it, or put NIGHTSHIFT_KEY=... in .env).")
    names = [n for n in (a.only.split(",") if a.only else STRATEGIES) if n in STRATEGIES]
    rows = {}
    for name in names:
        mem = os.path.join(tempfile.mkdtemp(prefix="bt_"), "memory.json")  # rival memory per strategy
        out = []
        for i in range(a.markets):
            seed = "%s-%d" % (a.seed, i)
            try:
                r = run_market(client, STRATEGIES[name], seed, a.days, a.rivals, mem)
            except Exception as e:  # keep going; report it
                print("  %s %s failed: %r" % (name, seed, e))
                continue
            out.append(r)
            print("  %-9s %-8s profit $%8.2f  rank %s/%s  %.0fs" % (name, seed, r["profit"], r["rank"], r["n"], r["secs"]))
        rows[name] = out
    print("\n%-9s %10s %10s %10s %8s %7s %7s" % ("strategy", "mean", "worst", "sd", "avgrank", "missed", "maxms"))
    ranked = sorted(rows.items(), key=lambda kv: -statistics.mean([r["profit"] for r in kv[1]]) if kv[1] else 0)
    for name, out in ranked:
        if not out:
            print("%-9s  (no results)" % name)
            continue
        ps = [r["profit"] for r in out]
        rk = [r["rank"] for r in out if r["rank"]]
        print("%-9s %10.0f %10.0f %10.0f %8.1f %7d %7.0f" % (
            name, statistics.mean(ps), min(ps), statistics.stdev(ps) if len(ps) > 1 else 0.0,
            statistics.mean(rk) if rk else 0, sum(r["missed"] for r in out), max(r["max_ms"] for r in out)))
    print("\nPick the highest mean with an acceptable worst market. Same seeds = fair comparison.")


if __name__ == "__main__":
    main()
