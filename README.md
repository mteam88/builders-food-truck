# Frontrun Fries

Matthew's bot for the Builders Night Shift Food Truck Wars final. Python 3.9+, standard library only.

```
git clone https://github.com/mteam88/builders-food-truck && cd builders-food-truck
echo 'NIGHTSHIFT_KEY=...' > .env      # or export NIGHTSHIFT_KEY
./run.sh
```

## What it does each night

1. **Predict the rivals.** For each rival with public code, we run a port of that rival's own decision code
   (`ftw/adapters/`, checked against its live orders) on tonight's state as that rival sees it, and take the
   order it would place. Optionally (`FTW_REAL_RIVALS=ibe,wagyu,...`) we also run the rivals' real bots,
   unmodified, from `rivals/`: a snapshot of their public repos (see `rivals/NOTICE` for repo, commit and
   licence), refreshed at start-up by `run.sh` from GitHub into `rivals_live/` when that
   works within a few seconds. Rival code always runs in separate processes with a clean environment (it never
   sees our key), a temporary home and working directory, its network redirected to a local stand-in server
   (`ftw/shim.py`, plus a private network namespace where Linux allows it), and a time limit. A rival whose
   code fails, times out, or guessed wrong yesterday is assumed to repeat yesterday's order (partly spread over
   other spots); rivals without public code are always handled that way. When the game shows tonight's planned
   counts (orders in per spot), those override the predictions.
2. **Best response.** For every spot, quality and price on a grid, we compute expected sales against the
   predicted orders with the game's customer model (fair choice, sell-outs, walk-over), over weather and crowd
   noise. Meals come from the newsvendor rule, capped by the food budget and staff (50 meals per employee).
3. **Profit and reputation.** Score = expected profit (minus moving cost, wages, expected riot bill) plus the
   future value of reputation gained. Leaving a spot we sold at costs a $150 penalty while our reputation there
   is below 70.
4. **Sit-outs.** If nothing is profitable, or the best spot is inside a warned riot's radius, we send no
   order; never two nights in a row (the crew would quit), so after a sit-out we open at the best safe spot.
5. **Safety.** A simple fallback order goes in at once (unless a riot is warned); the main decision replaces
   it about 0.1–0.5 s later, and with planned counts live we re-read them ~0.55 s before the window closes and
   re-order once if they change the answer. `control.json` (`{"mode": "fallback"}`) switches to the fallback bot at runtime, and the bot switches
   itself if realised profit trails its own forecast badly for 5 nights.

## Files

- `ftw/runner.py`: live loop (long-poll, watchdog, one order per night). `ftw/brain.py`: the decision.
- `ftw/game.py`: game constants and the expected-sales day model. `ftw/fallback.py`: the fallback bot.
- `ftw/rivals.py`, `ftw/adapters/`: rival prediction from ports. `ftw/rivalhost.py`, `ftw/shim.py`,
  `ftw/fetch_rivals.py`, `rivals/`: the rivals' real bots (opt-in).
- Logs go to `logs/` (gitignored).
