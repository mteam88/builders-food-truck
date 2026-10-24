"""The fallback bot: non-predictive, never crashes, one decision per night.

Base: Fortress (zoo fortress-r1, written blind from the docs), wrapped with v3's riot escape.
1. It never guesses tonight's rival orders: it reads only yesterday's public results, reputations and the forecast.
2. Night 1 it locks one spot, the Union unless a better-scoring spot wins on long-run uncontested premium profit.
3. It leaves only on nights 2-3, and only if a premium truck camps the same spot (two nights, or no event to hold it).
4. It locks for good once reputation passes 60 or on day 4; a mid-market switch adopts a spot where rep is already 58+.
5. Premium every night until reputation caps at 100 (+5 a day is the moat); then standard only on huge uncontested days.
6. Price and meals maximise expected profit on a grid against yesterday's neighbours at the spot (fair choice
   exp(value/1.5), rep multiplier, Normal price limits), plus a hunter world when we led revenue or the spot is busiest.
7. Meals come from the same expected-profit grid over three crowd scenarios, capped by budget and staff.
8. A warned riot near tonight's spot (v3's expected bill >= $50) moves us for the night to the best safe spot (v3's model).
9. A wrecked truck sends nothing, never two nights running; laws clamp the price; any exception returns a safe order.
10. Fallback(base="v3") gives the second option instead: v3, the sticky-premium camper (class V3), homed indoors.
"""
import math

# ====================================================================== fortress-r1 (verbatim, NOISE -> F_NOISE)
# Fortress: lock one spot, sell premium until reputation caps, then charge what the moat is worth.
# 
# Customers draw a price limit ~ Normal(mu, sd) and a minimum quality. They buy from a truck they
# can afford with weight exp(value / 1.5), value = quality_value * (0.7 + 0.6 * rep/100) - price.
# Premium at one spot gains +5 reputation a day (50 -> 100 in ten selling days). That lead is worth
# about $4.80 of value against a newcomer, which is the whole point of staying put.
# 
# The price grid maximises expected profit against whoever was here yesterday (and against a
# chaser, on days this spot is the busiest). Quality stays premium while reputation is still
# climbing. Spot choice is once, with a short window to leave if another premium truck is
# camping the same ground.


QUALITIES = ("cheap", "standard", "premium")
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
CANON_COST = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
CANON_VALUE = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
BUDGET = 500.0
FAIR_TEMP = 1.5
CROWD_SIGMA = 0.12
KNOWN = ("highst", "oval", "union", "northres", "thompson", "stadium",
         "fisher", "kottman", "mirror", "medcenter", "lane", "rpac", "schott", "gateway")
SHARE = {"highst": 0.24, "oval": 0.20, "union": 0.18, "northres": 0.16, "thompson": 0.14, "stadium": 0.08,
         "fisher": 0.12, "kottman": 0.08, "mirror": 0.10, "medcenter": 0.14, "lane": 0.12, "rpac": 0.09,
         "schott": 0.07, "gateway": 0.11}
# (mu, sd, cheap_ok, standard_ok). Round-number fit to the published "about $X" limits and mix.
SPOTS = {
    "highst": (12.0, 4.0, 0.50, 0.85),
    "oval": (13.0, 3.0, 0.35, 0.80),
    "union": (13.0, 3.5, 0.35, 0.80),
    "northres": (11.0, 2.5, 0.50, 0.90),
    "thompson": (12.0, 3.0, 0.45, 0.85),
    "stadium": (13.0, 3.5, 0.30, 0.75),
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
EVENT_PARAMS = {
    ("gameday", "stadium"): (15.5, 3.5, 0.40, 0.80),
    ("championship", "stadium"): (15.5, 3.5, 0.40, 0.80),
    ("midterms", "thompson"): (12.0, 3.0, 0.35, 0.75),
    ("concert", "union"): (14.0, 3.5, 0.35, 0.80),
    ("latenight", "highst"): (12.0, 4.0, 0.60, 0.90),
}
WEATHERS = ("clear", "cloudy", "rain", "cold")
WEATHER_MULT = {
    "clear": {},
    "cloudy": {"highst": .95, "oval": .90, "union": .97, "northres": .95, "thompson": .97, "stadium": .90,
               "mirror": .90, "lane": .95, "gateway": .95, "kottman": .95, "fisher": .97, "medcenter": 1.0,
               "rpac": .97, "schott": .97},
    "rain": {"highst": .65, "oval": .45, "union": .75, "northres": .70, "thompson": .70, "stadium": .50,
             "mirror": .45, "lane": .70, "gateway": .75, "kottman": .75, "fisher": .80, "medcenter": .95,
             "rpac": .80, "schott": .85},
    "cold": {"highst": .80, "oval": .70, "union": .85, "northres": .85, "thompson": .85, "stadium": .70,
             "mirror": .70, "lane": .80, "gateway": .80, "kottman": .80, "fisher": .85, "medcenter": .95,
             "rpac": .85, "schott": .85},
}
# Long-run event multiplier (games, concerts, midterms, late night, rally) from the published schedule.
LONG_EVENT = {"highst": 1.23, "oval": 1.14, "union": 1.10, "northres": 1.10, "thompson": 1.07,
              "stadium": 1.57, "schott": 1.25, "fisher": 1.06, "medcenter": 1.04}
# How often a serious rival also wants this spot. Busiest outdoor corners attract chasers.
CONTEST = {"highst": 0.72, "stadium": 0.48, "oval": 0.42, "union": 0.28, "thompson": 0.24,
           "northres": 0.20, "fisher": 0.30, "schott": 0.34, "mirror": 0.22, "lane": 0.26,
           "gateway": 0.22, "kottman": 0.16, "rpac": 0.20, "medcenter": 0.18}
# Indoor spots have a tighter crowd; High Street and the Oval do not.
STABILITY = {"union": 1.06, "medcenter": 1.05, "fisher": 1.04, "thompson": 1.03, "rpac": 1.02,
             "schott": 1.01, "northres": 0.99, "kottman": 0.98, "gateway": 0.97, "lane": 0.97,
             "oval": 0.96, "mirror": 0.92, "highst": 0.90, "stadium": 0.86}
GAMEDAY = {"highst": 2.0, "oval": 1.6, "union": 1.5, "northres": 1.3, "thompson": 1.2, "stadium": 5.0, "schott": 1.5}
EVENT_MULT = {
    "gameday": GAMEDAY,
    "midterms": {"thompson": 1.7},
    "concert": {"union": 1.8},
    "latenight": {"highst": 1.9, "northres": 1.7, "oval": .5, "union": .5, "thompson": .5, "stadium": .5},
    "rally": {"oval": 2.0},
}
BUILDINGS = {"thompson", "union", "fisher", "rpac", "kottman", "medcenter", "northres", "schott"}
POSITIONS = {"oval": (100, 70), "thompson": (-12, 140), "union": (440, 388), "stadium": (-490, 48),
             "highst": (463, -186), "northres": (170, -478)}
_ROWS = {"clear": (.58, .26, .13, .03), "cloudy": (.29, .335, .305, .07),
         "rain": (.26, .25, .28, .21), "cold": (.20, .14, .10, .56)}


def _stationary():
    p = [0.25, 0.25, 0.25, 0.25]
    for _ in range(40):
        nxt = [0.0, 0.0, 0.0, 0.0]
        for i, a in enumerate(WEATHERS):
            for j in range(4):
                nxt[j] += p[i] * _ROWS[a][j]
        p = nxt
    return dict(zip(WEATHERS, p))


STAT_W = _stationary()
F_NOISE = ((0.25, math.exp(-1.15 * CROWD_SIGMA)), (0.50, 1.0), (0.25, math.exp(1.15 * CROWD_SIGMA)))


def num(x, default=0.0):
    try:
        if isinstance(x, bool):
            return default
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
            if not x:
                return default
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def as_dict(x):
    return x if isinstance(x, dict) else {}


def as_list(x):
    return x if isinstance(x, list) else []


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rep_factor(rep):
    return 0.7 + 0.6 * max(0.0, min(100.0, rep)) / 100.0


def weather_mult(weather, spot):
    return WEATHER_MULT.get(weather, {}).get(spot, 1.0)


def long_weather(spot):
    return sum(STAT_W[w] * weather_mult(w, spot) for w in WEATHERS) or 1.0


def event_mult(events, spot):
    m = 1.0
    for e in events or ():
        if not isinstance(e, dict):
            continue
        key = str(e.get("key") or "")
        table = EVENT_MULT.get(key)
        if key == "championship":
            table = {s: 1.2 * v for s, v in GAMEDAY.items()}
        if table:
            m *= table.get(spot, 1.0)
    return m if m > 0 else 1.0


def spot_params(spot, events):
    p = SPOTS.get(spot, DEFAULT_SPOT)
    for e in events or ():
        if isinstance(e, dict):
            p = EVENT_PARAMS.get((str(e.get("key") or ""), spot), p)
    return p


def qshare(params, quality):
    _mu, _sd, cheap_ok, std_ok = params
    if quality == "premium":
        return 1.0
    if quality == "standard":
        return max(0.0, min(1.0, std_ok))
    return max(0.0, min(1.0, cheap_ok))


def costs_of(state):
    raw = as_dict(as_dict(state.get("rules")).get("costs"))
    out = {}
    for q, c in CANON_COST.items():
        v = num(raw.get(q), c)
        out[q] = v if 0.5 <= v <= 30 else c
    return out


def values_of(state):
    raw = as_dict(as_dict(state.get("rules")).get("values"))
    out = {}
    for q, c in CANON_VALUE.items():
        v = num(raw.get(q), c)
        out[q] = v if 1 <= v <= 40 else c
    return out


def features_of(state):
    return as_dict(state.get("features"))


def open_spots(state):
    rules = as_dict(state.get("rules"))
    raw = rules.get("spots")
    found = []
    if isinstance(raw, list):
        for s in raw:
            if isinstance(s, str) and s in KNOWN and s not in found:
                found.append(s)
    if found:
        return found
    info = as_dict(state.get("spot_info"))
    for s in KNOWN:
        if s in info:
            found.append(s)
    return found or ["union"]


def forecast_of(state):
    fc = as_dict(state.get("forecast"))
    exp = {}
    raw = as_dict(fc.get("expected_customers"))
    for s, v in raw.items():
        if isinstance(s, str):
            exp[s] = max(0.0, num(v, 0.0))
    weather = {}
    raw_w = as_dict(fc.get("weather"))
    for w in WEATHERS:
        weather[w] = max(0.0, num(raw_w.get(w), 0.0))
    if sum(weather.values()) <= 0:
        weather = {"clear": 1.0, "cloudy": 0.0, "rain": 0.0, "cold": 0.0}
    events = [e for e in as_list(fc.get("events")) if isinstance(e, dict)]
    return exp, weather, events


def our_rep(state, spot):
    you = as_dict(state.get("you"))
    return num(as_dict(you.get("reputation")).get(spot), 50.0)


def rival_rep(state, tid, spot):
    for t in as_list(state.get("trucks")):
        if isinstance(t, dict) and t.get("id") == tid:
            return num(as_dict(t.get("reputation")).get(spot), 50.0)
    return 50.0


def rivals_here(state, spot):
    """Yesterday's other trucks at this spot, as pricing opponents."""
    you = as_dict(state.get("you"))
    me = you.get("id")
    block = as_dict(as_dict(as_dict(state.get("last_day")).get("spots")).get(spot))
    out = []
    for t in as_list(block.get("trucks")):
        if not isinstance(t, dict) or (me and t.get("id") == me):
            continue
        if t.get("status") in ("no_staff", "wrecked"):
            continue
        q = t.get("quality") if t.get("quality") in QRANK else "standard"
        meals = max(0.0, num(t.get("meals"), 0.0))
        sold = max(0.0, num(t.get("sold"), 0.0))
        if meals <= 0 and sold <= 0:
            continue
        stock = meals if meals > 0 else sold
        emp = t.get("employees")
        if emp is not None and not isinstance(emp, bool):
            stock = min(stock, max(0.0, num(emp, 0.0)) * 50.0)
        if sold >= meals > 0:
            stock *= 1.08  # they sold out; they may cook a little more tonight
        out.append({"price": max(1.0, num(t.get("price"), 10.0)), "quality": q,
                    "rep": rival_rep(state, t.get("id"), spot), "stock": max(1.0, stock),
                    "sold": sold, "id": t.get("id")})
    return out


def premium_campers(rivals):
    return [r for r in rivals if r["quality"] == "premium" and r["sold"] >= 12 and r["stock"] >= 20]


def event_at(state, spot):
    for e in as_list(as_dict(state.get("last_day")).get("events")):
        if not isinstance(e, dict):
            continue
        key = str(e.get("key") or "")
        if e.get("spot") == spot or (key in ("gameday", "championship") and spot == "stadium"):
            return True
    return False


def busiest(exp, spots):
    best, name = -1.0, None
    for s in spots:
        if exp.get(s, 0.0) > best:
            best, name = exp.get(s, 0.0), s
    return name


def weather_mean(weather, spot):
    z = sum(weather.values()) or 1.0
    return sum(weather[w] / z * weather_mult(w, spot) for w in WEATHERS) or 1.0


def typical_crowd(spot, exp, weather, events):
    """Strip today's event and weather so a game-day forecast does not make us move in forever."""
    seen = exp.get(spot)
    em = event_mult(events, spot)
    wm = weather_mean(weather, spot)
    if seen is None:
        seen = 110.0 * 6.0 * SHARE.get(spot, 0.10) * em * wm
    base = seen / em / wm
    return max(0.0, base * long_weather(spot) * LONG_EVENT.get(spot, 1.05))


def monopoly_profit(n, spot, events=()):
    """Rough long-run daily profit of an uncontested premium truck, for ranking spots."""
    if n <= 1:
        return 0.0
    mu, sd, _c, _s = spot_params(spot, events)
    ceiling = 16.0
    best = 0.0
    price = 8.0
    while price <= ceiling + 1e-9:
        dem = n * phi((mu - price) / max(sd, 0.5))
        meals = min(90, max(0, int(round(dem))))
        sold = min(dem, meals)
        profit = sold * price - meals * 5.5 + (meals - sold) * 5.5 * 0.25
        if profit > best:
            best = profit
        price = round(price + 0.5, 2)
    return best


def spot_score(spot, exp, weather, events):
    n = typical_crowd(spot, exp, weather, events)
    base = monopoly_profit(n, spot)
    contested = 1.0 - 0.40 * CONTEST.get(spot, 0.3)
    return base * contested * STABILITY.get(spot, 1.0)


def scenarios_for(expected, weather, spot):
    """Crowd distribution around the forecast. A 3% haircut covers giveaways and fire drills."""
    expected *= 0.965 if spot in BUILDINGS else 0.97
    z = sum(weather.values()) or 1.0
    mean_mult = weather_mean(weather, spot)
    out = []
    for w in WEATHERS:
        pw = weather[w] / z
        if pw <= 0:
            continue
        for pn, nk in F_NOISE:
            base = expected * weather_mult(w, spot) / mean_mult * nk
            out.append((pw * pn, max(0.0, base)))
    if not out:
        out = [(0.25, expected * 0.87), (0.5, expected), (0.25, expected * 1.15)]
    return out


def customer_groups(params, trucks):
    mu, sd, cheap_ok, std_ok = params
    sd = max(sd, 0.4)
    qshare_ = (cheap_ok, max(0.0, std_ok - cheap_ok), max(0.0, 1.0 - std_ok))
    cuts = sorted({t["price"] for t in trucks if -1e8 < t["price"] < 1e8})
    edges = [-1e9] + cuts + [1e9]
    cdf = [phi((e - mu) / sd) if abs(e) < 1e8 else (0.0 if e < 0 else 1.0) for e in edges]
    out = []
    for k in range(len(edges) - 1):
        frac = cdf[k + 1] - cdf[k]
        if frac <= 1e-12:
            continue
        limit_lo = edges[k]
        for minq in range(3):
            acc = tuple(i for i, t in enumerate(trucks)
                        if t["price"] <= limit_lo + 1e-9 and QRANK[t["quality"]] >= minq and t["value"] >= -1e-9)
            out.append((frac * qshare_[minq], acc))
    return out


def sales_curve(groups, trucks, me, n_max, fair):
    """Our cumulative sales against rival stocks, if we never sell out."""
    stock = [float(t["stock"]) for t in trucks]
    stock[me] = 1e18
    w = [math.exp(max(-40.0, min(40.0, t["value"] / FAIR_TEMP))) if fair else 1.0 for t in trucks]
    pts = [(0.0, 0.0)]
    n, mine = 0.0, 0.0
    guard = 0
    while n < n_max - 1e-6 and guard < 12:
        guard += 1
        rates = [0.0] * len(trucks)
        for frac, acc in groups:
            live = [i for i in acc if stock[i] > 1e-6]
            if not live:
                continue
            if fair:
                z = sum(w[i] for i in live) or 1.0
                for i in live:
                    rates[i] += frac * w[i] / z
            else:
                i = max(live, key=lambda j: (trucks[j]["value"], -trucks[j]["price"]))
                rates[i] += frac
        step = n_max - n
        for i, r in enumerate(rates):
            if i != me and r > 1e-12 and stock[i] > 1e-6:
                step = min(step, stock[i] / r)
        if step <= 1e-8:
            break
        for i, r in enumerate(rates):
            if i != me:
                stock[i] -= min(stock[i], r * step)
        n += step
        mine += rates[me] * step
        pts.append((n, mine))
    if pts[-1][0] < n_max:
        pts.append((n_max, mine))
    return pts


def curve_at(pts, n):
    if n <= 0:
        return 0.0
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        if n <= n1 + 1e-9:
            if n1 <= n0:
                return s1
            return s0 + (s1 - s0) * (n - n0) / (n1 - n0)
    return pts[-1][1]


def alone_demand(price, quality, n, params, values, rf):
    ceiling = values[quality] * rf
    if price > ceiling + 1e-9 or n <= 0:
        return 0.0
    mu, sd, _c, _s = params
    return n * qshare(params, quality) * phi((mu - price) / max(sd, 0.4))


def law_bounds(laws):
    """Posted-price window after a cap or a floor. Customers pay this price."""
    lo, hi = 1.0, 30.0
    for law in laws:
        if not isinstance(law, dict):
            continue
        key = str(law.get("key") or "")
        params = as_dict(law.get("params"))
        if key in ("price_cap", "cap"):
            hi = min(hi, num(params.get("max_price", params.get("cap", 12)), 12))
        elif key in ("price_floor", "floor"):
            lo = max(lo, num(params.get("min_price", params.get("floor", 7.5)), 7.5))
    return lo, hi


def coupon_share(you):
    n = 0
    for c in as_list(you.get("coupons")):
        if isinstance(c, dict) and c.get("applied_day") in (None, ""):
            n += 1
    return min(0.6, 0.2 * n)


def scenario_profit(pay, sold, meals, cost, salvage, laws, quality, food, coupon):
    net = pay * sold * (1.0 - 0.5 * coupon)
    profit = net - cost * meals + salvage * cost * (meals - sold)
    for law in laws:
        if not isinstance(law, dict):
            continue
        key = str(law.get("key") or "")
        params = as_dict(law.get("params"))
        pm = as_dict(params.get("per_meal"))
        if quality in pm:
            profit -= num(pm.get(quality), 0.0) * sold
        if food and food in pm:
            profit -= num(pm.get(food), 0.0) * sold
        if not pm and key == "luxury_tax":
            profit -= (0.5 if quality == "premium" else (-0.5 if quality == "cheap" else 0.0)) * sold
        elif not pm and key in ("junk_tax", "junk_food_tax"):
            profit -= (0.5 if quality == "cheap" else (-0.5 if quality == "premium" else 0.0)) * sold
        if key in ("tax_top", "flat_fee"):
            share = num(params.get("revenue_share"), -0.1 if key == "tax_top" else 0.1)
            above = num(params.get("above"), 500)
            flat = num(params.get("flat"), 20 if key == "tax_top" else -20)
            profit += share * max(0.0, net - above) + (flat if meals >= 1 else 0.0)
    return profit


def best_meals(pay, cost, dems, mmax, salvage, laws, quality, food, coupon):
    cands = {0, mmax}
    for _p, d in dems:
        cands.add(int(max(0, min(mmax, math.floor(d)))))
        cands.add(int(max(0, min(mmax, math.ceil(d)))))
    best = (-1e18, 0, 0.0)
    total_p = sum(p for p, _d in dems) or 1.0
    # Sorted, and a tie keeps the smaller cook. Set iteration would change the answer across processes.
    for m in sorted(cands):
        ev = 0.0
        sold_ev = 0.0
        for p, d in dems:
            s = min(d, m)
            ev += p * scenario_profit(pay, s, m, cost, salvage, laws, quality, food, coupon)
            sold_ev += p * s
        if ev > best[0] + 1e-4:
            best = (ev, m, sold_ev / total_p)
    return best


def price_candidates(lo, hi, rivals, values, rf, quality):
    lo_c = int(math.floor(lo * 100 + 0.5))
    hi_c = int(math.floor(min(hi, values[quality] * rf) * 100 + 0.5))
    hi_c = max(lo_c, min(hi_c, 3000))
    cands = set(range(lo_c, hi_c + 1, 50))
    cands.add(hi_c)
    for r in rivals:
        rc = int(math.floor(r["price"] * 100 + 0.5))
        for d in (0, -1, -25, -50):
            c = rc + d
            if lo_c <= c <= hi_c:
                cands.add(c)
        # A cent better on value than this rival, for the winner-take-all rule.
        tie = values[quality] * rf - (values[r["quality"]] * rep_factor(r["rep"]) - r["price"]) - 0.01
        tc = int(math.floor(tie * 100 + 0.5))
        if lo_c <= tc <= hi_c:
            cands.add(tc)
    return sorted(c / 100.0 for c in cands)


def protest_factor(state):
    you = as_dict(state.get("you"))
    vote = you.get("vote")
    office = as_dict(as_dict(state.get("politics")).get("office"))
    holder = as_dict(office.get("holder"))
    if vote and holder.get("id") and vote == holder.get("id"):
        p = 0.50 if office.get("crisis") else 0.30
        return 1.0 - p * 0.40
    return 1.0


def choose_price(state, spot, quality, cases, exp_n):
    values = values_of(state)
    costs = costs_of(state)
    feat = features_of(state)
    fair = bool(feat.get("fair_choice", True))
    salvage = 0.25 if feat.get("salvage", True) else 0.0
    laws = [L for L in as_list(as_dict(state.get("politics")).get("laws")) if isinstance(L, dict)]
    events = forecast_of(state)[2]
    params = spot_params(spot, events)
    rf = rep_factor(our_rep(state, spot))
    _exp, weather, _events = forecast_of(state)
    expected = exp_n * protest_factor(state)
    scens = scenarios_for(expected, weather, spot)
    n_max = max(n for _p, n in scens)
    cost = costs[quality]
    mmax = min(int(BUDGET / cost + 1e-9), int(BUDGET / CANON_COST[quality] + 1e-9))
    cap = staff_cap(state)
    mmax = max(0, min(mmax, cap))
    lo, hi = law_bounds(laws)
    rules = as_dict(state.get("rules"))
    lo = max(lo, num(rules.get("price_min"), 1.0) if num(rules.get("price_min"), 1) >= 1 else 1.0)
    hi = min(hi, num(rules.get("price_max"), 30.0) if 5 <= num(rules.get("price_max"), 30) <= 30 else 30.0)
    lo = max(lo, 7.0)
    you = as_dict(state.get("you"))
    coupon = coupon_share(you) if feat.get("coupons", False) else 0.0
    food = pick_food(state, spot) if feat.get("food", False) else None
    rivals_all = []
    for _w, rs in cases:
        rivals_all.extend(rs)
    worlds = [(w, rs) for w, rs in cases if w > 0]
    rep = our_rep(state, spot)
    days_left = max(0.0, num(state.get("days"), 30) - num(state.get("order_for_day") or state.get("day"), 0))

    def score_at(price):
        dems = []
        my_value = values[quality] * rf - price
        for w, rs in worlds:
            if not rs:
                for sp, n in scens:
                    dems.append((w * sp, alone_demand(price, quality, n, params, values, rf)))
                continue
            live = [{"price": r["price"], "quality": r["quality"],
                     "value": values[r["quality"]] * rep_factor(r["rep"]) - r["price"],
                     "stock": r["stock"]} for r in rs]
            live.append({"price": price, "quality": quality, "value": my_value, "stock": 1e18})
            pts = sales_curve(customer_groups(params, live), live, len(live) - 1, n_max, fair)
            for sp, n in scens:
                dems.append((w * sp, max(0.0, curve_at(pts, n))))
        ev, meals, sold_ev = best_meals(price, cost, dems, mmax, salvage, laws, quality, food, coupon)
        bonus = 40.0 * min(1.0, sold_ev) * min(1.0, days_left / 12.0) if quality == "premium" and rep < 99 else 0.0
        return ev + bonus, price, meals, ev

    best = None
    for price in price_candidates(lo, hi, rivals_all, values, rf, quality):
        row = score_at(price)
        if best is None or row[0] > best[0]:
            best = row
    if best is None:
        return 11.75, min(80, mmax), 0.0
    # Half-dollar grid, then check the quarters around the winner.
    anchor = int(math.floor(best[1] * 100 + 0.5))
    for delta in (-50, -25, 25, 50):
        c = anchor + delta
        if int(math.floor(lo * 100 + 0.5)) <= c <= int(math.floor(hi * 100 + 0.5)):
            row = score_at(c / 100.0)
            if row[0] > best[0]:
                best = row
    return best[1], best[2], best[3]


def staff_cap(state):
    feat = features_of(state)
    staff = as_dict(as_dict(state.get("you")).get("staff"))
    cap = num(staff.get("capacity"), 0.0)
    if feat.get("staff", False) and cap >= 50:
        return int(cap)
    return 200


def pick_food(state, spot):
    diets = as_dict(as_dict(as_dict(state.get("forecast")).get("diets")).get(spot))
    meat = num(diets.get("meat"), 0.15)
    veg = num(diets.get("veggie"), 0.08)
    return "veggie" if veg > meat + 0.02 else "meat"


def were_top_earner(state, spot):
    """True when we led the market in revenue yesterday from this spot. Followers show up the next night."""
    last = as_dict(state.get("last_day"))
    you = as_dict(last.get("you"))
    if you.get("spot") != spot:
        return False
    mine = num(you.get("sold"), 0) * num(you.get("price"), 0)
    if mine <= 0:
        return False
    me = as_dict(state.get("you")).get("id")
    best = 0.0
    for block in as_dict(last.get("spots")).values():
        for t in as_list(as_dict(block).get("trucks")):
            if isinstance(t, dict) and t.get("id") != me:
                best = max(best, num(t.get("sold"), 0) * num(t.get("price"), 0))
    return mine >= best - 0.5


def was_busiest_yesterday(state, spot):
    spots = as_dict(as_dict(state.get("last_day")).get("spots"))
    ours = num(as_dict(spots.get(spot)).get("customers"), -1)
    if ours < 0:
        return False
    for s, block in spots.items():
        if s != spot and num(as_dict(block).get("customers"), 0) > ours + 1:
            return False
    return True


def cases_for(state, spot, rivals, exp, spots):
    """Worlds to price against: quiet, yesterday's neighbours, and hunters who chase a payday."""
    busy = busiest(exp, spots) == spot
    was_busy = was_busiest_yesterday(state, spot)
    top = were_top_earner(state, spot)
    you = as_dict(as_dict(state.get("last_day")).get("you"))
    worlds = []
    if rivals:
        # A truck that sat a quiet spot with us is a camper. One that only came because it was packed may leave.
        worlds.append((0.84 if not was_busy else 0.32, list(rivals)))
    hunters = []
    if top and you.get("quality") in QRANK:
        hunters.append({"price": max(7.0, num(you.get("price"), 11) - 0.5), "quality": you.get("quality"),
                        "rep": 50.0, "stock": min(90.0 if you.get("quality") == "premium" else 130.0,
                                                  max(40.0, num(you.get("meals"), 80))), "sold": 0})
    if busy:
        hunters.append({"price": 6.5, "quality": "cheap", "rep": 48.0, "stock": 160.0, "sold": 0})
        if not top:
            hunters.append({"price": 10.0, "quality": "premium", "rep": 50.0, "stock": 90.0, "sold": 0})
    if hunters:
        base = list(rivals) if (rivals and not was_busy) else []
        extra = [h for h in hunters if not any(b["quality"] == h["quality"] and abs(b["price"] - h["price"]) < 0.6
                                               for b in base)]
        if extra:
            worlds.append((0.72, base + extra))
    if not worlds:
        return [(1.0, [])]
    worlds.append((0.30, []))
    z = sum(w for w, _rs in worlds) or 1.0
    return [(w / z, rs) for w, rs in worlds]


def choose_home(state, mem, spots):
    exp, weather, events = forecast_of(state)
    scores = {s: spot_score(s, exp, weather, events) for s in spots}
    ranked = sorted(spots, key=lambda s: scores[s], reverse=True)

    def best():
        return ranked[0]

    home = mem.get("home")
    if home not in spots:
        reps = as_dict(as_dict(state.get("you")).get("reputation"))
        carried = max(spots, key=lambda s: num(reps.get(s), 50))
        if num(reps.get(carried), 50) >= 58:
            mem["home"] = carried
            mem["locked"] = True
            return carried
        mem["home"] = best()
        mem["locked"] = False
        mem["hit"] = 0
        return mem["home"]
    if mem.get("locked"):
        return home
    day = int(num(state.get("order_for_day"), num(state.get("day"), 0) + 1))
    rep = our_rep(state, home)
    if rep >= 60 or day >= 4:
        mem["locked"] = True
        return home
    last = state.get("last_day")
    if not isinstance(last, dict):
        return home
    rivals = rivals_here(state, home)
    serious = premium_campers(rivals)
    if serious:
        mem["hit"] = mem.get("hit", 0) + 1
    else:
        mem["hit"] = 0
        if day >= 2:
            mem["locked"] = True
        return home
    flee = mem["hit"] >= 2 or not event_at(state, home)
    if not flee:
        return home
    hostile = set()
    yesterday = as_dict(last.get("spots"))
    for s in spots:
        block = as_dict(yesterday.get(s))
        me = as_dict(state.get("you")).get("id")
        for t in as_list(block.get("trucks")):
            if not isinstance(t, dict) or t.get("id") == me:
                continue
            if t.get("quality") == "premium" and num(t.get("sold"), 0) >= 12:
                hostile.add(s)
                break
    alts = [s for s in ranked if s != home and s not in hostile]
    if alts and scores[alts[0]] >= 0.70 * scores.get(home, 1.0):
        mem["home"] = alts[0]
        mem["hit"] = 0
        return alts[0]
    return home


def market_key(state):
    return (state.get("mode"), state.get("market"), state.get("market_id"))


def wrecked(state, day):
    you = as_dict(state.get("you"))
    until = you.get("wrecked_until_day")
    if until is None or isinstance(until, bool):
        return False
    return num(until, 0) >= day


def sanitize(order, state):
    spots = open_spots(state)
    spot = order.get("spot") if isinstance(order, dict) else None
    if spot not in spots:
        spot = "union" if "union" in spots else spots[0]
    quality = order.get("quality") if isinstance(order, dict) else None
    if quality not in QRANK:
        quality = "premium"
    costs = costs_of(state)
    mmax = min(int(BUDGET / costs[quality] + 1e-9), int(BUDGET / CANON_COST[quality] + 1e-9))
    mmax = max(0, min(mmax, staff_cap(state)))
    meals = int(math.floor(num(order.get("meals") if isinstance(order, dict) else 0, 0)))
    meals = max(0, min(meals, mmax if mmax > 0 else 0))
    if meals <= 0:
        meals = min(80, mmax) if mmax else 80
        if quality == "premium":
            meals = min(meals, 90)
    price = num(order.get("price") if isinstance(order, dict) else None, 11.75)
    price = max(1.0, min(30.0, price))
    price = math.floor(price * 100 + 0.5) / 100.0
    out = {"spot": spot, "price": price, "quality": quality, "meals": int(meals)}
    feat = features_of(state)
    if feat.get("food", False):
        food = order.get("food") if isinstance(order, dict) else None
        out["food"] = food if food in ("meat", "veggie") else pick_food(state, spot)
    return out


def fallback(state):
    state = state if isinstance(state, dict) else {}
    spots = open_spots(state)
    spot = "union" if "union" in spots else spots[0]
    return sanitize({"spot": spot, "price": 11.75, "quality": "premium", "meals": 80}, state)


def decide(state, mem):
    you = state.get("you")
    if not isinstance(you, dict):
        if state.get("mode") == "showcase":
            return None
        return fallback(state)
    day = int(num(state.get("order_for_day"), num(state.get("day"), 0) + 1))
    key = market_key(state)
    if key != mem.get("key") or (mem.get("day") and day < mem["day"]):
        mem.clear()
        mem["key"] = key
    mem["key"] = key
    mem["day"] = day
    if wrecked(state, day):
        return None
    spots = open_spots(state)
    home = choose_home(state, mem, spots)
    exp, _weather, _events = forecast_of(state)
    if home not in exp:
        # Forecast omitted this spot: scale a base share by the spots we can see.
        seen = [exp[s] for s in spots if s in exp and SHARE.get(s)]
        scale = (sum(seen) / sum(SHARE[s] for s in spots if s in exp and SHARE.get(s))) if seen else 110 * 6
        exp = dict(exp)
        exp[home] = scale * SHARE.get(home, 0.12)
    rivals = rivals_here(state, home)
    cases = cases_for(state, home, rivals, exp, spots)
    rep = our_rep(state, home)
    price, meals, _ev = choose_price(state, home, "premium", cases, exp.get(home, 0.0))
    quality = "premium"
    # Once the moat is capped, a huge day can pay more in standard meals than in 90 premium ones.
    if rep >= 99 and not premium_campers(rivals):
        p2, m2, ev2 = choose_price(state, home, "standard", cases, exp.get(home, 0.0))
        _p1, _m1, ev1 = price, meals, _ev
        if ev2 > ev1 + 60:
            price, meals, quality = p2, m2, "standard"
    order = {"spot": home, "price": price, "quality": quality, "meals": meals}
    return sanitize(order, state)


def make(seed=0):
    mem = {"seed": seed, "home": None, "locked": False, "hit": 0, "key": None, "day": None}

    def policy(state):
        try:
            return decide(state if isinstance(state, dict) else {}, mem)
        except Exception:
            return fallback(state if isinstance(state, dict) else {})

    policy.mem = mem
    return policy


# ====================================================================== v3
BASE = {"highst": .24, "oval": .20, "union": .18, "northres": .16, "thompson": .14, "stadium": .08}
# price limit mean, sd, P(cheap ok), P(standard ok) per spot; events replace a spot's customers
CUST = {"highst": (12.0, 4.0, .50, .85), "oval": (13.0, 3.0, .35, .80), "union": (13.0, 3.5, .35, .80),
        "northres": (11.0, 2.5, .50, .90), "thompson": (12.0, 3.0, .45, .85), "stadium": (13.0, 3.5, .30, .75)}
CUST_DEFAULT = (12.5, 3.0, .40, .80)
EVENT_CUST = {("gameday", "stadium"): (15.5, 3.5, .40, .80), ("championship", "stadium"): (15.5, 3.5, .40, .80),
              ("midterms", "thompson"): (12.0, 3.0, .35, .75), ("concert", "union"): (14.0, 3.5, .35, .80),
              ("latenight", "highst"): (12.0, 4.0, .60, .90)}
POS = {"oval": (100, 70), "thompson": (-12, 140), "union": (440, 388), "stadium": (-490, 48),
       "highst": (463, -186), "northres": (170, -478)}
VALUES = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
REP_DELTA = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}
TEMP, SALVAGE, WRECK_COST = 1.5, 0.25, 800.0
NOISE = ((0.82, .25), (1.0, .5), (1.18, .25))    # 12% lognormal crowd noise, three points


def _num(x, d=0.0):
    try:
        if isinstance(x, bool):
            return d
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def _d(x):
    return x if isinstance(x, dict) else {}


def _l(x):
    return x if isinstance(x, list) else []


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rf(rep):
    return 0.7 + 0.6 * max(0.0, min(100.0, rep)) / 100.0


def demand(cust, trucks):
    """Expected fraction of a spot's customers buying from each truck (dicts: price, rank, w), with no stock limits."""
    mu, sd, c_ok, s_ok = cust
    sd = max(sd, 0.1)
    qmass = (c_ok, s_ok - c_ok, 1.0 - s_ok)
    edges = sorted({t["price"] for t in trucks}) + [1e9]
    out = [0.0] * len(trucks)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        mass = (1.0 if hi >= 1e8 else _phi((hi - mu) / sd)) - _phi((lo - mu) / sd)
        if mass <= 1e-9:
            continue
        for minq in range(3):
            acc = [i for i, t in enumerate(trucks) if t["price"] <= lo + 1e-9 and t["rank"] >= minq and t["ok"]]
            tot = sum(trucks[i]["w"] for i in acc)
            for i in acc:
                out[i] += mass * qmass[minq] * trucks[i]["w"] / tot
    return out


class V3:
    """The fallback bot: non-predictive, never crashes, one decision per night.
    
    1. It never guesses tonight's rival orders: it only reads yesterday's public results, the public reputations
       and the forecast.
    2. For each rival it keeps a decayed history of where that rival parks (memory 0.7 a night; prior: spread by
       base share), so one night's move doesn't make a spot look empty.
    3. Each night it scores every spot and price for premium food with the game's customer model: per-spot price
       limits, fair choice by exp(value/1.5) with value = 16 x (0.7 + 0.6 rep/100) - price, and rivals selling out.
    4. The crowd is the forecast, corrected for the spend-weighted truck count (N_eff) we add by opening.
    5. Score = expected profit (food, salvage, wages) - moving cost - expected riot bill + $6 a rep point gained.
    6. It camps: the first night's best spot is home (+$60 a night to stay), and home moves only when another spot
       scores better 3 nights running. Chasing yesterday's empty spot herds with the rivals and loses.
    7. Premium only: +5 rep a night builds the per-spot moat fastest (50 rep points = x24.5 share).
    8. Meals: newsvendor on the expected sales (12% crowd noise), capped by budget and staff.
    9. A warned riot near home (Michigan loss: High St; title win: Stadium) moves it for the night to the best
       safe spot. A night expected to lose money is sat out, never two nights running (the crew would quit).
    10. Laws: a price cap or floor clamps the price; tailgate scales the riot bill; any exception returns a safe order.
    """

    def __init__(self, hop=1.3, stay_bonus=60.0, rep_pt=6.0, memory=0.7, patience=3, riot_escape=50.0,
                 qualities=("premium",), phantom=True, only=None, first=None, sellout_bump=0, lo=7.0, hi=16.0, step=0.25, **_):
        self.camp = hop is None           # the harness's "camp" variant: never leave the first home
        self.stay_bonus, self.rep_pt, self.memory, self.riot_escape = stay_bonus, rep_pt, memory, riot_escape
        self.patience = 10 ** 9 if self.camp else max(1, int(patience))
        self.qualities, self.phantom, self.only, self.first = tuple(qualities), phantom, only, first
        self.sellout_bump = max(0, int(_num(sellout_bump)))
        self.grid = [round(lo + step * i, 2) for i in range(int((hi - lo) / step + 1e-9) + 1)]
        self.cache = (None, None)
        self.market = None
        self.home = None
        self.streak, self.cand = 0, None
        self.sat_out = False
        self.last = None
        self.seen = {}        # rival id -> {spot: presence weight}, decayed nightly
        self.seen_day = None
        self.orders = {}      # (rival id, spot) -> its latest order there

    def __call__(self, state):
        return self.decide(state)

    def decide(self, state):
        try:
            key = (state.get("mode"), state.get("market_id") or state.get("market"), state.get("order_for_day"))
            if self.cache[0] == key:
                return dict(self.cache[1]) if self.cache[1] else None
            o = self._decide(state)
            self.cache = (key, o)
            self.sat_out = o is None
            if o:
                self.last = dict(o)
            return dict(o) if o else None
        except Exception:
            o = self.last or {"spot": self.home or "oval", "price": 11.0, "quality": "premium", "meals": 80}
            return dict(o)

    # ------------------------------------------------------------------ the night's decision
    def _decide(self, state):
        you = state.get("you")
        if not isinstance(you, dict):
            return None
        rules = _d(state.get("rules"))
        spots = [s for s in _l(rules.get("spots")) if isinstance(s, str)] or list(BASE)
        mkey = (state.get("mode"), state.get("market_id") or state.get("market"))
        if mkey != self.market:
            self.market, self.home, self.sat_out, self.last = mkey, None, False, None
            self.seen, self.seen_day, self.orders = {}, None, {}
            self.streak, self.cand = 0, None
        fc = _d(state.get("forecast"))
        exp = _d(fc.get("expected_customers"))
        events = [e for e in _l(fc.get("events")) if isinstance(e, dict)]
        me = you.get("id")
        myrep = _d(you.get("reputation"))
        budget = _num(rules.get("budget"), 500) or 500
        costs = dict(COSTS, **{k: _num(v, COSTS[k]) for k, v in _d(rules.get("costs")).items() if k in COSTS})
        values = dict(VALUES, **{k: _num(v, VALUES[k]) for k, v in _d(rules.get("values")).items() if k in VALUES})
        pmin, pmax = max(1.0, _num(rules.get("price_min"), 1)), min(30.0, _num(rules.get("price_max"), 30) or 30)
        cap = _num(_d(you.get("staff")).get("capacity"), 200) or 200
        staff = _d(you.get("staff"))
        wages = _num(staff.get("employees"), 4) * _num(staff.get("wage"), 6)
        days = int(_num(state.get("days"), 30) or 30)
        day = int(_num(state.get("order_for_day"), 1) or 1)
        left = max(1, days - day)
        cap_price, floor_price, riot_mult = None, None, 1.0
        for law in _l(_d(state.get("politics")).get("laws")):
            k, p = str(_d(law).get("key", "")), _d(_d(law).get("params"))
            if k == "price_cap":
                cap_price = _num(p.get("max_price"), 12.0) or 12.0
            elif k == "price_floor":
                floor_price = _num(p.get("min_price"), 7.5) or 7.5
            elif k == "tailgate":
                riot_mult *= _num(p.get("riot"), 1.5) or 1.5
            elif k == "crackdown":
                riot_mult = 0.0
        features = _d(state.get("features"))
        if "riots" in features and not features.get("riots"):
            riot_mult = 0.0

        # yesterday, as public results show it
        ld = _d(state.get("last_day"))
        ld_spots = _d(ld.get("spots"))
        reps = {}
        for t in _l(state.get("trucks")):
            if isinstance(t, dict) and t.get("id") != me:
                reps[t.get("id")] = _d(t.get("reputation"))
        rivals = {s: [] for s in spots}
        n_rival = 0.0
        my_sold_at, sold_out = None, None
        yest = {}
        for s, v in ld_spots.items():
            for r in _l(_d(v).get("trucks")):
                if not isinstance(r, dict):
                    continue
                q = r.get("quality") if r.get("quality") in QRANK else "premium"
                meals = _num(r.get("meals"))
                if r.get("id") == me:
                    if _num(r.get("sold")) > 0:
                        my_sold_at = s
                    if meals >= 1 and _num(r.get("sold")) >= meals:
                        sold_out = (s, int(meals))
                    continue
                n_rival += min(1.0, meals * costs[q] / 250.0)
                if meals > 0:
                    emp = _num(r.get("employees"), 4) or 4
                    yest[r.get("id")] = s
                    self.orders[(r.get("id"), s)] = {"price": _num(r.get("price"), 10.0), "quality": q,
                                                     "stock": min(meals, emp * 50)}
        if ld_spots and self.seen_day != ld.get("day"):
            # where each rival parks: a decayed history of public results (prior: spread by base share)
            self.seen_day = ld.get("day")
            tot = sum(BASE.get(s, .12) for s in spots)
            for rid in set(yest) | set(self.seen):
                h = self.seen.get(rid) or {s: BASE.get(s, .12) / tot for s in spots}
                h = {s: w * self.memory for s, w in h.items()}
                if rid in yest:
                    h[yest[rid]] = h.get(yest[rid], 0.0) + (1.0 - self.memory)
                self.seen[rid] = h
        for rid, h in self.seen.items():
            for s, k in h.items():
                if s in rivals and k > 0.03:
                    o = self.orders.get((rid, s)) or {"price": 10.5, "quality": "premium", "stock": 90.0}
                    rivals[s].append(dict(o, rep=_num(_d(reps.get(rid)).get(s), 50.0), weight=k,
                                          stock=o["stock"] * k))
        n_yest = 0.0
        for v in ld_spots.values():
            for r in _l(_d(v).get("trucks")):
                if isinstance(r, dict):
                    q = r.get("quality") if r.get("quality") in COSTS else "premium"
                    n_yest += min(1.0, _num(r.get("meals")) * costs[q] / 250.0)
        if not ld_spots:
            # night 1: nobody seen yet; phantom rivals spread by base share
            n_tr = sum(1 for t in _l(state.get("trucks")) if isinstance(t, dict) and t.get("id") != me) or 5
            if self.phantom:
                tot = sum(BASE.get(s, .12) for s in spots)
                for s in spots:
                    k = n_tr * BASE.get(s, .12) / tot
                    if k > 0:
                        rivals[s].append({"price": 10.5, "quality": "premium", "rep": 50.0, "stock": 90 * k,
                                          "weight": k})
            crowd_corr = 1.0
        else:
            # the forecast counted yesterday's spend; today's crowd follows today's (we open at full spend)
            crowd_corr = max(4.0, n_rival + 1.0) / max(4.0, n_yest) if n_yest > 0 else 1.0

        # home: where we camp and build the moat; it changes only after `patience` nights of a better spot
        last_spot = you.get("last_spot") if you.get("last_spot") in spots else None
        if self.home not in spots:
            self.home = None
        if self.home is None and last_spot and _num(myrep.get(last_spot), 50) > 50.5:
            self.home = last_spot          # switched in mid-market: adopt where we already sell
        spots_try = [self.home] if (self.camp and self.home) else spots
        if self.only:
            spots_try = [s for s in spots_try if s in self.only] or spots_try
        move_costs = _d(rules.get("move_costs"))

        per = {}           # spot -> (score, spot, quality, price, meals, net, riot)
        for s in spots_try:
            c0 = _num(exp.get(s)) * crowd_corr
            if c0 <= 0:
                continue
            cust = CUST.get(s, CUST_DEFAULT)
            for e in events:
                cust = EVENT_CUST.get((e.get("key"), s), cust)
            r0 = _num(myrep.get(s), 50)
            move = 0.0
            if last_spot and last_spot != s and features.get("moving_costs", True):
                move = _num(_d(move_costs.get(last_spot)).get(s), 2 + 0.01 * self._dist(state, last_spot, s))
            riot = self._riot(state, events, s) * riot_mult
            score_adj = -move - riot
            if s == self.home:
                score_adj += self.stay_bonus
            for q in self.qualities:
                if q not in QRANK:
                    continue
                v0 = values[q] * rf(r0)
                meals_max = int(min(budget / costs[q] + 1e-9, cap))
                if meals_max <= 0:
                    continue
                for p in self.grid:
                    if p < pmin or p > pmax or (cap_price and p > cap_price + 1e-9) or v0 - p < 0:
                        continue
                    if floor_price and p < floor_price - 1e-9:
                        continue
                    sales = self._sales(cust, c0, p, q, v0, rivals.get(s, []), values, meals_max)
                    meals = self._meals(sales, meals_max, p, costs[q])
                    exp_sold = sum(w * min(meals, sales * f) for f, w in NOISE)
                    profit = exp_sold * p - meals * costs[q] + (meals - exp_sold) * costs[q] * SALVAGE - wages
                    gain = max(-r0, min(100.0 - r0, REP_DELTA[q])) if exp_sold >= 1 else 0.0
                    score = profit + score_adj + self.rep_pt * gain * min(1.0, left / 10.0)
                    if s not in per or score > per[s][0]:
                        per[s] = (score, s, q, p, meals, profit - move - riot, riot)
        if not per:
            s = self.home or last_spot or ("oval" if "oval" in spots else spots[0])
            return {"spot": s, "price": 11.0, "quality": "premium", "meals": int(min(80, cap))}
        top = max(per.values())
        pick = top
        if self.home is None or self.home not in per:
            first = [per[s] for s in (self.first or ()) if s in per]
            if first:
                top = pick = max(first)
            self.home, self.streak = top[1], 0
        elif top[1] != self.home:
            here = per[self.home]
            self.streak = self.streak + 1 if top[1] == self.cand else 1
            self.cand = top[1]
            if self.streak >= self.patience:
                self.home, self.streak = top[1], 0          # a better spot, night after night: move home
            elif here[6] < self.riot_escape:
                pick = here                                  # stay home tonight (unless a riot is warned there)
        else:
            self.streak = 0
        score, s, q, p, meals, net, _ = pick
        if self.sellout_bump and sold_out and sold_out[0] == s:
            # sold out here last night: demand was censored, so cook more than last night
            meals = max(meals, min(int(min(budget / costs[q] + 1e-9, cap)), sold_out[1] + self.sellout_bump))
        # did we open last night? (the runner may have posted the main bot's order, or none)
        out_last = self.sat_out or (bool(ld) and _d(ld.get("you")).get("status") not in (None, "played"))
        if net < 0 and not out_last and day > 1:
            return None          # a losing night: sit it out (never two nights running: the crew quits)
        return {"spot": s, "price": round(p, 2), "quality": q, "meals": int(max(1, meals))}

    # ------------------------------------------------------------------ pieces
    def _sales(self, cust, crowd, p, q, v0, rivals, values, stock):
        """Our expected sales at a spot against yesterday's rivals there (their sell-outs spill to us)."""
        tr = [{"price": p, "rank": QRANK[q], "w": math.exp((v0 - p) / TEMP), "ok": True, "stock": stock, "k": 1.0}]
        for r in rivals:
            v = values[r["quality"]] * rf(r["rep"]) - r["price"]
            k = r.get("weight", 1.0)
            tr.append({"price": r["price"], "rank": QRANK[r["quality"]], "w": k * math.exp(v / TEMP), "ok": v >= 0,
                       "stock": r["stock"], "k": k})
        f = demand(cust, tr)
        mu, sd, c_ok, s_ok = cust
        surv = lambda x: 1.0 - _phi((x - mu) / max(sd, 0.1))
        out = [crowd * f[i] > tr[i]["stock"] for i in range(len(tr))]
        spill = 0.0
        for i in range(1, len(tr)):
            excess = crowd * f[i] - tr[i]["stock"]
            if excess <= 0:
                continue
            # the sold-out rival's leftover customers who would take our price and quality ...
            reach = surv(max(p, tr[i]["price"])) / max(surv(tr[i]["price"]), 1e-9)
            if tr[0]["rank"] < tr[i]["rank"]:
                reach *= (c_ok if tr[0]["rank"] == 0 else s_ok)
            # ... shared with the trucks still in stock by choice weight
            w_open = sum(t["w"] for j, t in enumerate(tr) if j and j != i and not out[j] and t["ok"])
            spill += excess * reach * tr[0]["w"] / (tr[0]["w"] + w_open)
        return crowd * f[0] + spill

    @staticmethod
    def _meals(sales, meals_max, p, cost):
        """Newsvendor on the 3-point noise: add meals while the next one is worth cooking."""
        under, over = p - cost, cost * (1.0 - SALVAGE)
        crit = under / (under + over) if under > 0 else 0.0
        m, acc = 0.0, 0.0
        for f, w in NOISE:
            acc += w
            if acc >= 1.0 - crit - 1e-9:
                m = sales * f
                break
        return int(max(1, min(meals_max, math.floor(m + 0.5))))

    def _dist(self, state, a, b):
        info = _d(state.get("spot_info"))
        pa = (_num(_d(info.get(a)).get("x"), POS.get(a, (0, 0))[0]), _num(_d(info.get(a)).get("z"), POS.get(a, (0, 0))[1]))
        pb = (_num(_d(info.get(b)).get("x"), POS.get(b, (0, 0))[0]), _num(_d(info.get(b)).get("z"), POS.get(b, (0, 0))[1]))
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def _riot(self, state, events, s):
        """Expected riot bill tonight at spot s from a warned riot (Michigan loss: High St; title win: Stadium)."""
        cost = 0.0
        for e in events:
            text = str(e.get("effect", "")).lower()
            if "riot" not in text:
                continue
            if "high street" in text:
                center, p, inten, radius = "highst", 0.45, 0.7, 260.0
            elif "stadium" in text:
                center, p, inten, radius = "stadium", 0.80, 0.95, 380.0
            else:
                continue
            x = math.exp(-self._dist(state, center, s) / radius)
            hit = min(1.0, p * inten * x)
            cost += hit * (0.3 * WRECK_COST + 0.7 * (30 + 120 * x))
        return cost


# ====================================================================== the shipped interface
def _riot_mult(state):
    mult = 1.0
    for law in _l(_d(state.get("politics")).get("laws")):
        k, p = str(_d(law).get("key", "")), _d(_d(law).get("params"))
        if k == "tailgate":
            mult *= _num(p.get("riot"), 1.5) or 1.5
        elif k == "crackdown":
            mult = 0.0
    feats = _d(state.get("features"))
    if "riots" in feats and not feats.get("riots"):
        mult = 0.0
    return mult


INDOOR = ("union", "northres", "thompson")    # v3's night-1 home candidates when it is the base


class Fallback:
    """Fortress with v3's riot escape. hop is accepted for the harness (Fortress always camps)."""

    def __init__(self, hop=1.3, base="fortress", riot_escape=50.0, **kw):
        self.base, self.riot_escape = base, riot_escape
        kw.setdefault("first", INDOOR)
        self.v3 = V3(hop=hop, riot_escape=riot_escape, **kw) if base == "v3" else None
        self.pol = make(0)
        self.cache = (None, None)
        self.sat_out = False
        self.last = None

    def __call__(self, state):
        return self.decide(state)

    def decide(self, state):
        if self.v3 is not None:
            return self.v3.decide(state)
        try:
            key = (state.get("mode"), state.get("market_id") or state.get("market"), state.get("order_for_day"))
            if self.cache[0] == key:
                return dict(self.cache[1]) if self.cache[1] else None
            o = self._decide(state)
            self.cache = (key, o)
            self.sat_out = o is None
            if o:
                self.last = dict(o)
            return dict(o) if o else None
        except Exception:
            try:
                o = self.last or fallback(state)
            except Exception:
                o = {"spot": "union", "price": 11.75, "quality": "premium", "meals": 80}
            return dict(o)

    def _decide(self, state):
        if not isinstance(state.get("you"), dict):
            return None                      # showcase: no truck, no order
        played = state["you"].get("last_spot")
        mem = getattr(self.pol, "mem", {})
        if self.last and played and played != self.last.get("spot") and mem.get("home") not in (None, played):
            mem["home"] = None               # the main bot's order was posted: re-adopt the spot our rep is built at
        o = self.pol(state)
        if not o:
            if not self.sat_out:
                return None                  # wrecked tonight (orders would bounce anyway)
            o = fallback(state)              # never two nights running
        mult = _riot_mult(state)
        if mult > 0:
            events = [e for e in _l(_d(state.get("forecast")).get("events")) if isinstance(e, dict)]
            helper = V3()
            if helper._riot(state, events, o["spot"]) * mult >= self.riot_escape:
                spots = [s for s in _l(_d(state.get("rules")).get("spots")) if isinstance(s, str)] or list(BASE)
                safe = [s for s in spots if helper._riot(state, events, s) * mult < self.riot_escape]
                if safe:
                    alt = V3(only=safe, riot_escape=self.riot_escape).decide(state)
                    if alt and alt.get("spot") in safe:
                        o = alt
                    elif alt is None and not self.sat_out:
                        return None
        return o
