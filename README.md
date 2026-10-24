# Frontrun Fries

Food Truck Wars bot for Builders Night Shift. One file, Python 3.9+, standard library only.

    echo NIGHTSHIFT_KEY=fk_... > .env
    ./run.sh                    # live game and the final (restarts on crash, keeps the Mac awake)
    python3 bot.py --practice   # private practice market

## v3: simple High Street camper

v2 tried to model every spot and every rival and lost about $800 on days 2-3. v3 drops all of that:

- Park at High Street every night, the busiest spot on campus. Never move, never sit out.
- Premium food, 90 meals: all the $500 budget buys.
- Start at $12. Sold out: $0.25 more the next day. Sold under 60%: $0.25 less. Stay between $10.50 and $14.

The constants are at the top of `bot.py`. Every day's full state is logged to `data/`.
