# 5guysburgers: Cheapest Eats (truck `t_5f2743`)

Food Truck Wars bot for Builders Night Shift. Python 3.8+, standard library only.

## Setup (Mac)

```bash
git clone https://github.com/nicholasravakhah/5guysburgers.git
cd 5guysburgers
echo 'NIGHTSHIFT_KEY=fk_...' > .env && chmod 600 .env   # gitignored; or: export NIGHTSHIFT_KEY=...
python3 -m unittest discover -s tests                  # 15 tests, ~2 s
```

## Run

**Only ONE machine may run the live bot.** Every order replaces the last one, so two bots on one key overwrite each other.

```bash
python3 bot.py                    # live game, then the 8:30 AM final. Leave it running.
python3 bot.py --avoid oval       # default "camp" style; never park at the listed spots
python3 bot.py --style max        # plain profit, no camping bias  (--style cheap: $9 price cap)
python3 bot.py --dry-run          # reads live state and decides, but never sends. Safe anywhere.
python3 bot.py --practice --games 3 --seed try   # private practice markets
python3 tools/watch_orders.py     # read-only: logs every change to our live order
```

To restart after a code change: press Ctrl+C, run `git pull`, then start `python3 bot.py` again. Learned rival data
(`state/memory_live.json`) and the days already processed survive restarts. A second live
instance on the same Mac refuses to start (file lock). If another machine overwrites our order, the bot prints
`!!! WARNING ... replaced by another client`.

Keep the Mac awake during the final: `caffeinate -dims python3 bot.py`.

## How it decides

Each night the bot decides in about 30–110 ms, inside the final's 2-second window:

1. **Customer model** (`ftw/model.py`, `ftw/params.json`). Each customer has a price limit drawn from a lognormal
   distribution and a minimum quality, with spot-specific parameters fitted to about 900 practice days (`tools/fit.py`).
   Crowd sizes come from `forecast.expected_customers`. The weather is drawn once per scenario and its effect is
   renormalized by the forecast odds, because the forecast already averages over weather, so it is not counted twice.
2. **Opponent model** (`ftw/opponents.py`). For each rival it scores competing spot-choice explanations: stay put,
   follow the crowd, go where crowds per truck are biggest, chase hungry customers, follow events, favorite spot, or
   random. Older evidence decays, so a rival who changes code is noticed within a few days. It also tracks how often
   each rival keeps the same price, cuts its price, and actually shows up. What it learns carries from the live game
   into every final market; per-market state resets.
3. **Scenarios + exact sales** (`ftw/engine.py`). The bot samples about 48 joint scenarios (weather, crowds,
   customers, rival orders) and scores every option against the same set. Customers arrive one at a time, choose
   the best value, and move on when a truck sells out. Two facts make this exact and fast. Our sales with n meals
   are min(n, D), where D is our sales with full stock. Trucks ranked above us sell out independently of what we do.
4. **Search**. All 6 spots × 3 qualities, a coarse price grid, prices one cent better than likely rivals' value, and a
   refinement step. Quantity is chosen directly from simulated demand (the newsvendor rule: cook the nth meal only if
   price × P(demand ≥ n) > cost). A small reputation term is weighted by the days left (zero on the last day).
   Ties among near-equal options are broken at random so the bot isn't predictable.

## Benchmarks (actual runs)

Real practice server, against the 3 house bots, 4 seeds (`tools/practice_bench.py`), profit per 30-day market:

| strategy | mean | sd |
|---|---|---|
| adaptive (this bot) | **$11,246** | $788 |
| crowd-follower (standard $10) | $3,333 | $1,385 |
| fixed premium (oval $12.50) | $2,172 | $1,838 |
| fixed standard (union $10) | $783 | $332 |

Local simulator (`tools/bench.py`; *not* the server engine, used to check robustness, with customer parameters
perturbed in each market). Profit per market against five rival fields, 8 seeds each:

| candidate | house | crowd | mixed | premium | smart (3 copies of this bot) | overall | waste |
|---|---|---|---|---|---|---|---|
| starter | -3,779 | -2,235 | -8,960 | -4,412 | -5,865 | -5,050 | 75% |
| fixed std | -420 | 2,909 | -9,488 | 2,492 | -7,386 | -2,378 | 68% |
| fixed prem | -708 | 3,105 | 5,098 | 1,986 | 3,766 | 2,649 | 48% |
| crowd | 4,852 | -14,761 | -13,178 | -6,018 | -9,856 | -7,792 | 82% |
| **adaptive** | **11,237** | **11,295** | **11,450** | **9,950** | **6,639** | **10,114** | 16% |

Held-out seeds (5 per field), adaptive variants, overall mean / vs smart rivals:
camp (default: rep_return 0.9, rep_lambda 2, near_tol $0.50) **$10,529 / $7,536** · camp without the 2× reputation
bonus $10,384 / $6,861 · max $10,071 / $5,746 · cheap ($9 cap) $8,875 / $6,866.

## Assumptions and caveats

- The budget and quality distributions are fitted approximations, not the server's real parameters. Union and event
  days have the least data.
- Verified from data: a truck that parks at a spot and sells nothing keeps its reputation there (no drift).
- `rivals: 0` in practice is treated as the default (3 house bots).
- How the server breaks ties beyond "cheaper price wins" is unknown, so the bot avoids exact ties.

## Layout

`bot.py` entry point · `ftw/` bot package · `tools/` calibration, benchmarks, observers · `tests/` ·
`state/` and `logs/` hold private runtime data (gitignored).
