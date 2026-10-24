"""Game constants and the expected-sales day model.

Everything here is a published or measured rule of Food Truck Wars. The bot (brain.py) and the
simulator (harness/sim.py) both use it, so the bot plans with the same model it is rehearsed in.

Customer model (fitted on thousands of practice days, round numbers): at each spot a customer has
a price limit ~ Normal(mu, sd) and a minimum quality (cheap ok with prob cheap_ok, standard ok with
prob standard_ok, else premium only). Among trucks they accept (price <= limit, quality ok, in stock,
value >= 0) they pick one with weight exp(value / 1.5), value = quality_value * rep_factor - price.
"""
import math

QUALITIES = ("cheap", "standard", "premium")
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
VALUES = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
REP_DELTA = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}
BUDGET = 500.0
FAIR_TEMP = 1.5
SALVAGE = 0.25
MEALS_PER_EMPLOYEE = 50
CROWD_SIGMA = 0.12

# Customer parameters per spot (mu, sd of the price limit; cheap_ok, standard_ok cumulative).
SPOTS = {
    "highst": (12.0, 4.0, 0.50, 0.85),
    "oval": (13.0, 3.0, 0.35, 0.80),
    "union": (13.0, 3.5, 0.35, 0.80),
    "northres": (11.0, 2.5, 0.50, 0.90),
    "thompson": (12.0, 3.0, 0.45, 0.85),
    "stadium": (13.0, 3.5, 0.30, 0.75),
    # New spots: docs' typical limit and premium share; the rest are guesses.
    "fisher": (14.0, 3.0, 0.30, 0.70),
    "kottman": (12.5, 3.0, 0.35, 0.75),
    "mirror": (12.0, 3.0, 0.40, 0.80),
    "medcenter": (13.0, 3.0, 0.35, 0.80),
    "lane": (12.0, 4.0, 0.50, 0.85),
    "rpac": (12.0, 3.0, 0.35, 0.75),
    "schott": (13.0, 3.5, 0.35, 0.80),
    "gateway": (11.5, 3.0, 0.45, 0.85),
}
DEFAULT_SPOT = (12.0, 3.0, 0.40, 0.80)
# Events replace a spot's customer parameters for the day.
EVENT_PARAMS = {
    ("gameday", "stadium"): (15.5, 3.5, 0.40, 0.80),
    ("championship", "stadium"): (15.5, 3.5, 0.40, 0.80),
    ("midterms", "thompson"): (12.0, 3.0, 0.35, 0.75),
    ("concert", "union"): (14.0, 3.5, 0.35, 0.80),
    ("latenight", "highst"): (12.0, 4.0, 0.60, 0.90),
}
# Weather multipliers on each spot's crowd (realized weather; the forecast gives probabilities).
WEATHER_MULT = {
    "clear": {},
    "cloudy": {"highst": .95, "oval": .90, "union": .97, "northres": .95, "thompson": .97, "stadium": .90},
    "rain": {"highst": .65, "oval": .45, "union": .75, "northres": .70, "thompson": .70, "stadium": .50},
    "cold": {"highst": .80, "oval": .70, "union": .85, "northres": .85, "thompson": .85, "stadium": .70},
}
WEATHERS = ("clear", "cloudy", "rain", "cold")
# Default positions (metres); the state's spot_info overrides them.
POSITIONS = {"oval": (100, 70), "thompson": (-12, 140), "union": (440, 388), "stadium": (-490, 48),
             "highst": (463, -186), "northres": (170, -478)}


def num(x, default=0.0):
    """A float from anything the server (or a dirty practice market) might send."""
    try:
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def rep_factor(rep):
    return 0.7 + 0.6 * rep / 100.0


def value_of(quality, rep, price):
    return VALUES[quality] * rep_factor(rep) - price


def weather_mult(weather, spot):
    return WEATHER_MULT.get(weather, {}).get(spot, 1.0)


def spot_params(spot, events=()):
    p = SPOTS.get(spot, DEFAULT_SPOT)
    for e in events or ():
        key = e.get("key") if isinstance(e, dict) else e
        p = EVENT_PARAMS.get((key, spot), p)
    return p


def walk_prob(metres):
    return 0.7 * math.exp(-metres / 400.0)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def max_meals(quality):
    return int(BUDGET / COSTS[quality] + 1e-9)


# ------------------------------------------------------------------ fluid day model
# A truck here is a dict: price, quality, value, stock (meals; float("inf") allowed).

def customer_groups(params, trucks, lo=-1e9, hi=1e9):
    """Split a spot's customers into groups that accept the same trucks.

    Returns [(fraction, acceptable truck indexes)], fractions over customers with limit in [lo, hi).
    """
    mu, sd, cheap_ok, std_ok = params
    qshare = (cheap_ok, std_ok - cheap_ok, 1.0 - std_ok)
    cuts = sorted({t["price"] for t in trucks if lo < t["price"] < hi})
    edges = [lo] + cuts + [hi]
    cdf = [phi((e - mu) / sd) if abs(e) < 1e8 else (0.0 if e < 0 else 1.0) for e in edges]
    out = []
    for k in range(len(edges) - 1):
        frac = cdf[k + 1] - cdf[k]
        if frac <= 1e-12:
            continue
        limit_lo = edges[k]
        for minq in range(3):
            acc = tuple(i for i, t in enumerate(trucks)
                        if t["price"] <= limit_lo + 1e-9 and QRANK[t["quality"]] >= minq and t["value"] >= 0)
            out.append((frac * qshare[minq], acc, (edges[k], edges[k + 1], minq)))
    return out


def fluid(groups, trucks, n, fair=True, sold=None):
    """Expected sales when n customers (split into groups) arrive in random order.

    Runs the arrival process as a fluid: each truck sells at a constant rate until it sells out,
    then its customers move on. Returns (sold per truck, unserved mass per group).
    """
    stock = [t["stock"] - (sold[i] if sold else 0.0) for i, t in enumerate(trucks)]
    sold = list(sold) if sold else [0.0] * len(trucks)
    w = [math.exp(t["value"] / FAIR_TEMP) if fair else 1.0 for t in trucks]
    if not fair:
        best = {}
    left = n
    hungry = [0.0] * len(groups)
    while left > 1e-9:
        rates = [0.0] * len(trucks)
        open_ = [s > 1e-9 for s in stock]
        starving = []
        for g, (frac, acc, _) in enumerate(groups):
            live = [i for i in acc if open_[i]]
            if not live:
                starving.append(g)
                continue
            if fair:
                z = sum(w[i] for i in live)
                for i in live:
                    rates[i] += frac * w[i] / z
            else:
                i = max(live, key=lambda j: (trucks[j]["value"], -trucks[j]["price"]))
                rates[i] += frac
        # customers until the next truck sells out
        step = left
        for i, r in enumerate(rates):
            if r > 1e-12 and open_[i]:
                step = min(step, stock[i] / r)
        for i, r in enumerate(rates):
            if r > 0:
                q = min(stock[i], r * step)
                stock[i] -= q
                sold[i] += q
        for g in starving:
            hungry[g] += groups[g][0] * step
        left -= step
        if step <= 1e-12:
            break
    return sold, hungry


def sales_curve(groups, trucks, me, n_max, fair=True):
    """Our cumulative expected sales after N customers, for N in [0, n_max], with our stock unlimited.

    Returns breakpoints [(N, our_sold)]; rivals' sell-outs bend the curve upward.
    """
    stock = [t["stock"] for t in trucks]
    w = [math.exp(t["value"] / FAIR_TEMP) if fair else 1.0 for t in trucks]
    pts = [(0.0, 0.0)]
    n, mine = 0.0, 0.0
    while n < n_max - 1e-9:
        rates = [0.0] * len(trucks)
        for frac, acc, _ in groups:
            live = [i for i in acc if stock[i] > 1e-9]
            if not live:
                continue
            if fair:
                z = sum(w[i] for i in live)
                for i in live:
                    rates[i] += frac * w[i] / z
            else:
                i = max(live, key=lambda j: (trucks[j]["value"], -trucks[j]["price"]))
                rates[i] += frac
        step = n_max - n
        for i, r in enumerate(rates):
            if i != me and r > 1e-12 and stock[i] > 1e-9:
                step = min(step, stock[i] / r)
        for i, r in enumerate(rates):
            if i != me:
                stock[i] -= min(stock[i], r * step)
        n += step
        mine += rates[me] * step
        pts.append((n, mine))
        if step <= 1e-12:
            break
    return pts


def curve_at(pts, n):
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        if n <= n1:
            return s0 + (s1 - s0) * (n - n0) / (n1 - n0) if n1 > n0 else s1
    (n0, s0), (n1, s1) = pts[-2], pts[-1]
    return s1 + ((s1 - s0) / (n1 - n0) if n1 > n0 else 0.0) * (n - n1)


def crowd_scenarios(expected, weather_probs, spot):
    """[(prob, crowd)] for a spot: forecast weather mixture times lognormal noise (3 points)."""
    probs = {w: max(0.0, num((weather_probs or {}).get(w), 0.0)) for w in WEATHERS}
    z = sum(probs.values())
    if z <= 0:
        probs, z = dict({w: 0.0 for w in WEATHERS}, clear=1.0), 1.0
    mean_mult = sum(probs[w] / z * weather_mult(w, spot) for w in WEATHERS) or 1.0
    noise = ((0.25, math.exp(-1.15 * CROWD_SIGMA)), (0.5, 1.0), (0.25, math.exp(1.15 * CROWD_SIGMA)))
    out = []
    for w in WEATHERS:
        if probs[w] <= 0:
            continue
        for pn, k in noise:
            out.append((probs[w] / z * pn, expected * weather_mult(w, spot) / mean_mult * k))
    return out


def best_meals(price, cost, demands, cap, mmax):
    """Newsvendor: meals maximizing E[revenue + salvage - food cost] over [(prob, demand)]."""
    cands = {0, int(min(cap, mmax))}
    for _, d in demands:
        for m in (math.floor(d), math.ceil(d)):
            cands.add(int(max(0, min(m, mmax))))
    best = (0.0, 0)
    for m in sorted(cands):
        ev = 0.0
        for p, d in demands:
            s = min(d, m, cap)
            ev += p * (price * s + SALVAGE * cost * (m - s))
        ev -= cost * m
        if ev > best[0] + 1e-9:
            best = (ev, m)
    return best
