"""Cheapest Eats (truck t_5f2743, formerly 5 Guys) Food Truck Wars bot. Python 3.8+, standard library only.

    python3 bot.py                      # live game, then the final (leave it running)
    python3 bot.py --dry-run            # live state, decide each night, send nothing
    python3 bot.py --practice           # one private practice market
    python3 bot.py --practice --games 5 --seed s   # several practice markets: s-0, s-1, ...

The key is read from $NIGHTSHIFT_KEY or a gitignored .env file (NIGHTSHIFT_KEY=...).
"""
import argparse
import os
import statistics
import sys
import time

from ftw.api import API, Client
from ftw.runner import STATE_DIR, Bot, acquire_lock
from ftw.strategy import STYLES


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="5 Guys Food Truck Wars bot")
    ap.add_argument("--practice", action="store_true", help="play private practice markets")
    ap.add_argument("--games", type=int, default=1, help="number of practice markets")
    ap.add_argument("--seed", help="practice seed prefix, for repeatable markets")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="practice: ignore saved practice memory")
    ap.add_argument("--dry-run", action="store_true", help="live: decide but never send orders")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--style", choices=sorted(STYLES), default="camp",
                    help="camp: build reputation at a home spot (default); max: plain profit; cheap: $9 cap")
    ap.add_argument("--avoid", default="", help="comma list of spots to never use, e.g. oval")
    ap.add_argument("--quality", choices=["cheap", "standard", "premium"], help="only use this quality")
    ap.add_argument("--max-price", type=float, help="never price above this")
    ap.add_argument("--price", type=float, help="fixed price; the bot still picks meals")
    ap.add_argument("--meals-mult", type=float, help="cook this multiple of the optimal meal count")
    args = ap.parse_args()
    manual = {"avoid_spots": [x for x in args.avoid.split(",") if x], "only_quality": args.quality}
    if args.max_price:
        manual["max_price"] = args.max_price
    if args.price:
        manual["fixed_price"] = args.price
    if args.meals_mult:
        manual["meals_mult"] = args.meals_mult
    # manual locks only apply in the live game; the final always runs the full optimizer
    cfg = dict(STYLES[args.style], live_overrides=manual)

    client = Client()
    if not client.has_key:
        sys.exit("Set NIGHTSHIFT_KEY (export it, or put NIGHTSHIFT_KEY=... in .env).")

    if args.practice:
        mem = None if args.fresh else os.path.join(STATE_DIR, "memory_practice.json")
        profits = []
        for g in range(args.games):
            seed = None if args.seed is None else ("%s-%d" % (args.seed, g) if args.games > 1 else args.seed)
            status, res = client.practice(seed=seed, days=args.days, rivals=args.rivals)
            if status not in (200, 201):
                sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
            print("practice market %s (seed %s) started" % (res.get("id"), res.get("seed")))
            bot = Bot(client, base=res["base"], memory_path=mem, log_name="practice", quiet=args.quiet,
                      cfg=cfg)
            final = bot.run(stop_when_over=True)
            if not final:
                continue
            you = final.get("you") or {}
            profits.append(you.get("cash") or 0.0)
            ms = bot.stats["decide_ms"]
            print("practice over: cash $%.2f  (orders %d, rejected %d, decide max %.0f ms)" % (
                you.get("cash") or 0, bot.stats["orders"], bot.stats["rejected"], max(ms) if ms else 0))
            for t in sorted(final.get("trucks") or [], key=lambda t: -(t.get("cash") or 0)):
                print("  %-24s $%10.2f" % (t.get("truck_name"), t.get("cash") or 0))
            bot.writer.flush()
            time.sleep(0.5)
        if len(profits) > 1:
            print("mean $%.2f  sd $%.2f  over %d markets" % (
                statistics.mean(profits), statistics.stdev(profits), len(profits)))
        return

    name = "live_dry" if args.dry_run else "live"
    lock = acquire_lock(name)  # noqa: F841  (held for the life of the process)
    print("Cheapest Eats bot (%s style) on %s %s(Ctrl+C to stop)" % (
        args.style, client.url, "[DRY RUN] " if args.dry_run else ""))
    bot = Bot(client, base=API, memory_path=os.path.join(STATE_DIR, "memory_%s.json" % name),
              log_name=name, dry_run=args.dry_run, quiet=args.quiet, cfg=cfg)
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
