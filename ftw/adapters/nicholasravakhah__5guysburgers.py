"""Port of nicholasravakhah/5guysburgers (Cheapest Eats, formerly 5 Guys, t_5f2743)
@ adf4fb68 (see notes.md for the full sha). Previous ports: policy_c030310e.py, policy_55ef7bf1.py.

What runs at adf4fb68 (`python3 bot.py`, default --style camp, --avoid ""):
  * d974743: new default style "camp" = {"rep_return": 0.9, "rep_lambda": 2.0, "near_tol_abs": 0.5}.
    No price cap / offset any more (max_price 24, grid $x.00/$x.50). --avoid drops spots from the search.
  * adf4fb68 deduped strategy.py/opponents.py (the duplicate-module bug is gone) and ported Arjun's
    event herd at EVENT_PULL = 0.3 (every rival's spot distribution mixed 30% onto forecast event spots,
    before the 3% uniform floor). shared_meal_frac exists but is 1.0 (off) in every style.

Earlier: 55ef7bf1df added ftw/params.json, which load_params merges over DEFAULT; it is inlined below
as PARAMS_JSON.

Faithful stdlib port: ftw/rules.py, engine.py, model.py, opponents.py, strategy.py inlined verbatim
(relative imports removed). runner.py/api.py are I/O only; the runner's per-night flow is: brain.observe(state) then
brain.decide(state, deadline=min(0.45, 0.3*time_left)). The live Brain seed is os.urandom (random);
here it is `seed`. Live compute deadline is 0.45 s wall clock; the sim default `deadline=None`
uses 0.45 too (set e.g. deadline=30 for a deterministic full search).
"""
import hashlib
import math
import random
import time
from bisect import bisect_left


# ==================== ftw/rules.py ====================

SPOTS = ["oval", "thompson", "union", "stadium", "highst", "northres"]
QUALITIES = ["cheap", "standard", "premium"]
QI = {"cheap": 0, "standard": 1, "premium": 2}
COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
VALUES = {"cheap": 8, "standard": 12, "premium": 16}
REP_DELTA = {"cheap": -3, "standard": 2, "premium": 5}
BUDGET = 500.0
PRICE_MIN, PRICE_MAX = 1.0, 30.0


class Rules:
    def __init__(self, raw=None):
        raw = raw if isinstance(raw, dict) else {}
        self.budget = _num(raw.get("budget"), BUDGET)
        self.costs = _merge(COSTS, raw.get("costs"))
        self.values = _merge(VALUES, raw.get("values"))
        spots = raw.get("spots")
        self.spots = [s for s in spots if isinstance(s, str)] if isinstance(spots, list) and spots else list(SPOTS)
        self.qualities = [q for q in QUALITIES if q in self.costs and q in self.values]
        self.price_min = _num(raw.get("price_min"), PRICE_MIN)
        self.price_max = _num(raw.get("price_max"), PRICE_MAX)

    def max_meals(self, quality):
        # integer cents avoid float edge cases (500 / 3.75 = 133.33)
        cost_c = int(round(self.costs[quality] * 100))
        return int(round(self.budget * 100)) // cost_c if cost_c > 0 else 0

    def clamp_price(self, price):
        p = min(max(float(price), self.price_min), self.price_max)
        return math.floor(p * 100 + 1e-6) / 100.0

    def validate(self, order):
        """Return a corrected legal order (never raises)."""
        spot = order.get("spot") if order.get("spot") in self.spots else self.spots[0]
        q = order.get("quality") if order.get("quality") in self.qualities else "standard"
        try:
            price = self.clamp_price(order.get("price", 10.0))
        except (TypeError, ValueError):
            price = 10.0
        try:
            meals = int(order.get("meals", 0))
        except (TypeError, ValueError):
            meals = 0
        meals = max(0, min(meals, self.max_meals(q)))
        return {"spot": spot, "price": price, "quality": q, "meals": meals}


def rep_factor(rep):
    return 0.7 + 0.006 * rep


def rep_after(rep, quality, sold):
    if sold <= 0:
        return rep
    return max(0.0, min(100.0, rep + REP_DELTA[quality]))


def _num(x, default):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _merge(base, raw):
    out = dict(base)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str):
                out[k] = _num(v, out.get(k, 0.0))
    return out


# ==================== ftw/engine.py ====================



class Truck:
    __slots__ = ("tid", "qi", "price", "value", "stock")

    def __init__(self, tid, quality, price, rep, stock, values):
        self.tid = tid
        self.qi = QI[quality]
        self.price = float(price)
        self.value = values[quality] * rep_factor(rep) - self.price
        self.stock = int(stock)


def rank_key(value, price):
    # higher value first, then cheaper price first
    return (-round(value, 9), round(price, 9))


def simulate_spot(limits, minq, trucks, rng=None, fair=None):
    """Full sequential simulation. Returns {tid: sold}. Used by the local market
    simulator and tests; trucks with identical (value, price) are randomized.
    fair=(delta, temp): customers pick among eligible trucks within delta of the best
    eligible value, weighted by exp(value/temp) (our guess at "fair choice")."""
    if fair and rng is not None:
        return _simulate_fair(limits, minq, trucks, rng, fair)
    live = [t for t in trucks if t.value >= 0 and t.stock > 0]
    if rng is not None:
        rng.shuffle(live)
    live.sort(key=lambda t: rank_key(t.value, t.price))  # stable: random among exact ties
    stock = [t.stock for t in live]
    sold = [0] * len(live)
    prices = [t.price for t in live]
    qis = [t.qi for t in live]
    for L, mq in zip(limits, minq):
        for i in range(len(live)):
            if stock[i] > 0 and prices[i] <= L and qis[i] >= mq:
                stock[i] -= 1
                sold[i] += 1
                break
    out = {t.tid: 0 for t in trucks}
    for t, s in zip(live, sold):
        out[t.tid] = s
    return out


def _simulate_fair(limits, minq, trucks, rng, fair):
    delta, temp = fair
    live = [t for t in trucks if t.value >= 0 and t.stock > 0]
    stock = {t.tid: t.stock for t in live}
    sold = {t.tid: 0 for t in trucks}
    for L, mq in zip(limits, minq):
        ok = [t for t in live if stock[t.tid] > 0 and t.price <= L and t.qi >= mq]
        if not ok:
            continue
        best = max(t.value for t in ok)
        band = [t for t in ok if t.value >= best - delta]
        ws = [math.exp(t.value / temp) for t in band]
        r = rng.random() * sum(ws)
        acc = 0.0
        pick = band[-1]
        for t, w in zip(band, ws):
            acc += w
            if r <= acc:
                pick = t
                break
        stock[pick.tid] -= 1
        sold[pick.tid] += 1
    return sold


class SpotScenario:
    """One sampled day at one spot: customers in arrival order plus rival trucks."""
    __slots__ = ("rivals", "_free", "limits", "minq")

    def __init__(self, limits, minq, rivals):
        self.limits = limits
        self.minq = minq
        self.rivals = sorted([t for t in rivals if t.value >= 0 and t.stock > 0],
                             key=lambda t: rank_key(t.value, t.price))
        self._free = {}

    def free_sorted(self, j):
        """Sorted limits of customers not captured by the top-j rivals, by min-quality class."""
        f = self._free.get(j)
        if f is not None:
            return f
        top = self.rivals[:j]
        stock = [t.stock for t in top]
        prices = [t.price for t in top]
        qis = [t.qi for t in top]
        buckets = ([], [], [])
        for L, mq in zip(self.limits, self.minq):
            taken = False
            for i in range(j):
                if stock[i] > 0 and prices[i] <= L and qis[i] >= mq:
                    stock[i] -= 1
                    taken = True
                    break
            if not taken:
                buckets[mq].append(L)
        for b in buckets:
            b.sort()
        self._free[j] = buckets
        return buckets

    def demand(self, qi, price, value, cap, fair=None):
        """Our sales with `cap` meals at (quality index, price, value).

        fair=(delta, temp) approximates the "fair choice" rule: rivals whose value is
        within delta of ours share customers with us by weight exp(value/temp). We mix
        the outcomes with those rivals ranked above us and below us by our share."""
        if value < 0:
            return 0
        if fair:
            delta, temp = fair
            above = [t for t in self.rivals if t.value > value + delta]
            band = [t for t in self.rivals if abs(t.value - value) <= delta]
            if band:
                j_lo = len(above)
                j_hi = j_lo + len(band)
                w_us = math.exp(value / temp)
                share = w_us / (w_us + sum(math.exp(t.value / temp) for t in band))
                d_lo = self._count(j_lo, qi, price)
                d_hi = self._count(j_hi, qi, price)
                d = int(round(share * d_lo + (1 - share) * d_hi))
                return d if d < cap else cap
        key = rank_key(value, price)
        j = 0
        for t in self.rivals:
            if rank_key(t.value, t.price) <= key:  # exact tie counts against us
                j += 1
            else:
                break
        d = self._count(j, qi, price)
        return d if d < cap else cap

    def _count(self, j, qi, price):
        buckets = self.free_sorted(j)
        d = 0
        for c in range(qi + 1):
            b = buckets[c]
            d += len(b) - bisect_left(b, price)
        return d


def best_quantity(demands, price, cost, cap, salvage=0.0):
    """Given sampled demands D_k (already capped), pick n maximizing
    price * E[min(n, D)] - cost * n + salvage * cost * E[n - min(n, D)].
    Returns (n, expected_profit, expected_sold). With salvage s this is the plain
    problem at price - s*cost and cost (1-s)*cost, so the same search applies.
    The objective is concave piecewise-linear in n, so the optimum is at a sample."""
    if salvage:
        price, cost = price - salvage * cost, (1.0 - salvage) * cost
    K = len(demands)
    if K == 0 or price <= cost:
        return 0, 0.0, 0.0
    ds = sorted(demands)
    best = (0, 0.0, 0.0)
    below = 0.0  # sum of samples strictly below the current n
    i = 0
    while i < K:
        n = ds[i]
        exp_sold = (below + (K - i) * n) / K
        profit = price * exp_sold - cost * n
        if n <= cap and profit > best[1]:
            best = (n, profit, exp_sold)
        while i < K and ds[i] == n:
            below += n
            i += 1
    return best


# ==================== ftw/model.py ====================


WEATHERS = ["clear", "cloudy", "rain", "cold"]

DEFAULT = {
    # price limit ~ LogNormal(log(median), sigma); mix = P(min quality = cheap/standard/premium)
    "spots": {
        "highst":   {"median": 11.4, "sigma": 0.34, "mix": [0.49, 0.42, 0.09]},
        "oval":     {"median": 12.7, "sigma": 0.30, "mix": [0.37, 0.45, 0.18]},
        "union":    {"median": 12.6, "sigma": 0.30, "mix": [0.15, 0.62, 0.23]},
        "northres": {"median": 10.7, "sigma": 0.27, "mix": [0.51, 0.41, 0.08]},
        "thompson": {"median": 11.7, "sigma": 0.26, "mix": [0.40, 0.40, 0.20]},
        "stadium":  {"median": 13.0, "sigma": 0.30, "mix": [0.32, 0.43, 0.25]},
    },
    # log crowd multiplier by realized weather (relative to clear)
    "weather": {
        "oval":     {"clear": 0.0, "cloudy": -0.13, "rain": -0.82, "cold": -0.37},
        "thompson": {"clear": 0.0, "cloudy": 0.01, "rain": -0.35, "cold": -0.15},
        "union":    {"clear": 0.0, "cloudy": -0.07, "rain": -0.27, "cold": -0.15},
        "stadium":  {"clear": 0.0, "cloudy": -0.09, "rain": -0.67, "cold": -0.33},
        "highst":   {"clear": 0.0, "cloudy": -0.04, "rain": -0.41, "cold": -0.18},
        "northres": {"clear": 0.0, "cloudy": -0.08, "rain": -0.37, "cold": -0.17},
    },
    "crowd_noise": 0.12,
    # event effects on budgets/quality (crowd size is already in the forecast)
    # "at": the event's own spot, "else": every other spot
    "events": {
        "gameday":   {"at": {"median_mult": 1.18, "mix": [0.42, 0.38, 0.20]},
                      "else": {"median_mult": 1.03, "mix_shift": [0.04, -0.02, -0.02]}},
        "midterms":  {"at": {"median_mult": 1.0, "mix_shift": [-0.05, -0.02, 0.07]}},
        "concert":   {"at": {"median_mult": 1.10}},
        "latenight": {"at": {"median_mult": 0.97, "mix_shift": [0.06, -0.03, -0.03]},
                      "spots": ["highst", "northres"]},
        "rally":     {"at": {"median_mult": 1.0}},
    },
    # per-scenario parameter jitter (model uncertainty)
    "median_jitter": 0.05,
    "mix_jitter": 0.04,
}

# c7704aeb model.py: new spots (docs "New spots and moving"): typical price limit and share wanting premium.
# mix = [cheap ok, needs standard, needs premium]
NEW_SPOT_PRIORS = {
    "fisher":    {"median": 14.0, "sigma": 0.30, "mix": [0.30, 0.40, 0.30]},
    "kottman":   {"median": 12.5, "sigma": 0.30, "mix": [0.35, 0.40, 0.25]},
    "mirror":    {"median": 12.0, "sigma": 0.30, "mix": [0.40, 0.40, 0.20]},
    "medcenter": {"median": 13.0, "sigma": 0.30, "mix": [0.35, 0.45, 0.20]},
    "lane":      {"median": 12.0, "sigma": 0.40, "mix": [0.50, 0.35, 0.15]},
    "rpac":      {"median": 12.0, "sigma": 0.30, "mix": [0.35, 0.40, 0.25]},
    "schott":    {"median": 13.0, "sigma": 0.30, "mix": [0.35, 0.45, 0.20]},
    "gateway":   {"median": 11.5, "sigma": 0.30, "mix": [0.45, 0.40, 0.15]},
}

_EVENT_ALIASES = [("game", "gameday"), ("midterm", "midterms"), ("concert", "concert"),
                  ("late", "latenight"), ("rush", "latenight"), ("rally", "rally")]


def event_key(ev):
    if not isinstance(ev, dict):
        return None
    k = ev.get("key")
    if isinstance(k, str) and k:
        return k
    name = str(ev.get("name", "")).lower()
    for frag, key in _EVENT_ALIASES:
        if frag in name:
            return key
    return None


PARAMS_JSON = {"_source": "tools/fit.py on ~900 practice days (state/calib.jsonl), 2026-10-24. Approximations, not server internals.", "spots": {"highst": {"median": 11.58, "sigma": 0.39, "mix": [0.49, 0.37, 0.14]}, "oval": {"median": 12.57, "sigma": 0.29, "mix": [0.37, 0.44, 0.19]}, "union": {"median": 13.58, "sigma": 0.39, "mix": [0.33, 0.48, 0.19]}, "northres": {"median": 10.7, "sigma": 0.27, "mix": [0.51, 0.41, 0.08]}, "thompson": {"median": 11.59, "sigma": 0.27, "mix": [0.46, 0.4, 0.14]}, "stadium": {"median": 13.03, "sigma": 0.3, "mix": [0.31, 0.44, 0.25]}}, "events": {"gameday": {"at": {"median_mult": 1.21, "mix": [0.41, 0.38, 0.21]}}, "midterms": {"at": {"median_mult": 1.03, "mix_shift": [-0.13, 0.01, 0.12]}}, "concert": {"at": {"median_mult": 1.05}}, "latenight": {"at": {"median_mult": 0.91, "mix_shift": [0.12, -0.05, -0.07]}, "spots": ["highst", "northres"]}, "rally": {"at": {"median_mult": 1.08}}}}


def load_params():
    p = _deepcopy(DEFAULT)
    try:
        over = _deepcopy(PARAMS_JSON)   # ftw/params.json @ 55ef7bf1 (same merge semantics as theirs)
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                for kk, vv in v.items():
                    p[k][kk] = vv
            else:
                p[k] = v
    except (OSError, ValueError):
        pass
    return p


class CustomerModel:
    def __init__(self, params=None):
        self.p = params or load_params()

    def profile(self, spot, events):
        """(median, sigma, mix) for a spot given tomorrow's events list."""
        base = self.p["spots"].get(spot) or NEW_SPOT_PRIORS.get(spot) or \
            {"median": 12.0, "sigma": 0.3, "mix": [0.35, 0.45, 0.20]}  # c7704aeb
        median, sigma, mix = base["median"], base["sigma"], list(base["mix"])
        for ev in events or []:
            key = event_key(ev)
            spec = self.p["events"].get(key) if key else None
            if not spec:
                continue
            at_spots = spec.get("spots") or [ev.get("spot")]
            eff = spec.get("at") if spot in at_spots else spec.get("else")
            if not eff:
                continue
            median *= eff.get("median_mult", 1.0)
            if "mix" in eff:
                mix = list(eff["mix"])
            if "mix_shift" in eff:
                mix = [m + d for m, d in zip(mix, eff["mix_shift"])]
        mix = [max(0.0, m) for m in mix]
        s = sum(mix) or 1.0
        return median, sigma, [m / s for m in mix]

    def sample_weather(self, forecast, rng):
        probs = (forecast or {}).get("weather") or {}
        r = rng.random()
        acc = 0.0
        tot = sum(max(0.0, float(probs.get(w, 0) or 0)) for w in WEATHERS) or 0.0
        if tot <= 0:
            return None
        for w in WEATHERS:
            acc += max(0.0, float(probs.get(w, 0) or 0)) / tot
            if r <= acc:
                return w
        return WEATHERS[-1]

    def crowd(self, spot, expected, weather, forecast, rng):
        """Sampled crowd size for a spot given a sampled weather outcome."""
        mult = 1.0
        wt = self.p["weather"].get(spot)
        probs = (forecast or {}).get("weather") or {}
        if weather and wt:
            norm = sum(float(probs.get(w, 0) or 0) * math.exp(wt.get(w, 0.0)) for w in WEATHERS)
            if norm > 0:
                mult = math.exp(wt.get(weather, 0.0)) / norm
        mult *= math.exp(rng.gauss(0.0, self.p["crowd_noise"]))
        lam = max(0.0, expected * mult)
        # Poisson-ish integer rounding
        n = int(lam)
        if rng.random() < lam - n:
            n += 1
        return n

    def customers(self, n, median, sigma, mix, rng, jitter=True):
        """n customers in arrival order: (limits, minq) lists."""
        if jitter:
            median *= math.exp(rng.gauss(0.0, self.p["median_jitter"]))
            mj = self.p["mix_jitter"]
            mix = [max(0.0, m + rng.gauss(0.0, mj)) for m in mix]
            s = sum(mix) or 1.0
            mix = [m / s for m in mix]
        mu = math.log(median)
        c0, c1 = mix[0], mix[0] + mix[1]
        limits = [0.0] * n
        minq = [0] * n
        gauss, rand = rng.gauss, rng.random
        for i in range(n):
            limits[i] = math.exp(mu + sigma * gauss(0.0, 1.0))
            r = rand()
            minq[i] = 0 if r < c0 else (1 if r < c1 else 2)
        return limits, minq


def default_expected(n_trucks):
    """Fallback crowd when the forecast is missing: ~110 per truck, at least 4 trucks."""
    share = {"highst": 0.24, "oval": 0.20, "union": 0.18, "northres": 0.16, "thompson": 0.14, "stadium": 0.08}
    total = 110 * max(4, n_trucks)
    return {s: total * share.get(s, 1.0 / len(SPOTS)) for s in SPOTS}


def make_rng(seed):
    return random.Random(seed)


# ==================== ftw/opponents.py ====================


EVENT_PULL = 0.3  # live day 3: all six trucks went to the Union concert (Arjun); learned per rival too
HYPS = ["stay", "crowd", "percap", "hungry", "event", "favorite", "uniform"]
PRIOR = {"stay": 0.6, "crowd": 0.5, "percap": 0.4, "hungry": 0.2, "event": 0.3, "favorite": 0.3, "uniform": 0.2}
DECAY = 0.85          # per observation
MARKET_SHRINK = 0.5   # when a new market starts
HIST = 24             # orders kept per rival


def _softmax(scores, beta):
    m = max(scores.values())
    ex = {k: math.exp(beta * (v - m)) for k, v in scores.items()}
    t = sum(ex.values())
    return {k: v / t for k, v in ex.items()}


def _norm(scores, spots):
    m = max(max(scores.get(s, 0.0) for s in spots), 1e-9)
    return {s: scores.get(s, 0.0) / m for s in spots}


def hypothesis_dists(spots, ctx, last_spot, favorite):
    """ctx: {'expected': {spot: n}, 'events': [spot], 'prev': {spot: {'trucks': k, 'hungry': h}}}"""
    exp_c = ctx.get("expected") or {}
    prev = ctx.get("prev") or {}
    uni = {s: 1.0 / len(spots) for s in spots}
    crowd = _softmax(_norm(exp_c, spots), 6.0) if exp_c else uni
    if exp_c:
        pc = {s: exp_c.get(s, 0.0) / (1.0 + prev.get(s, {}).get("trucks", 0)) for s in spots}
        percap = _softmax(_norm(pc, spots), 6.0)
    else:
        percap = uni
    hung = {s: float(prev.get(s, {}).get("hungry", 0)) for s in spots}
    hungry = _softmax(_norm(hung, spots), 5.0) if any(hung.values()) else uni
    evs = [s for s in (ctx.get("events") or []) if s in spots]
    event = {s: (1.0 / len(evs) if s in evs else 0.0) for s in spots} if evs else crowd
    stay = {s: (1.0 if s == last_spot else 0.0) for s in spots} if last_spot in spots else uni
    fav = favorite if favorite else uni
    return {"stay": stay, "crowd": crowd, "percap": percap, "hungry": hungry,
            "event": event, "favorite": fav, "uniform": uni}


class Rival:
    def __init__(self, tid, name=""):
        self.tid = tid
        self.name = name
        self.score = {h: 0.0 for h in HYPS}      # decayed log-likelihood
        self.orders = []                          # recent dicts: spot, quality, price, meals, sold, market, day
        self.active = [1.0, 1.0]                  # decayed (played, sat out) counts, Beta prior
        self.spot_freq = {}
        self.last_spot = None                     # last spot in the current market
        self.rep = {}                             # current per-spot reputation (state or tracked)

    # ---- persistence
    def to_json(self):
        return {"tid": self.tid, "name": self.name, "score": self.score, "orders": self.orders[-HIST:],
                "active": self.active, "spot_freq": self.spot_freq}

    @classmethod
    def from_json(cls, d):
        r = cls(d.get("tid", "?"), d.get("name", ""))
        sc = d.get("score") or {}
        r.score = {h: float(sc.get(h, 0.0)) for h in HYPS}
        r.orders = [o for o in (d.get("orders") or []) if isinstance(o, dict)][-HIST:]
        a = d.get("active") or [1.0, 1.0]
        r.active = [float(a[0]), float(a[1])]
        r.spot_freq = {k: float(v) for k, v in (d.get("spot_freq") or {}).items()}
        return r

    def new_market(self):
        self.score = {h: v * MARKET_SHRINK for h, v in self.score.items()}
        self.last_spot = None
        self.rep = {}

    # ---- learning
    def observe(self, spots, ctx, order, market=None, day=None):
        """order: dict with spot/quality/price/meals/sold, or None if they sat out."""
        self.active = [self.active[0] * 0.9 + (1.0 if order else 0.0),
                       self.active[1] * 0.9 + (0.0 if order else 1.0)]
        if not order:
            return
        spot = order.get("spot")
        if spot in spots:
            dists = hypothesis_dists(spots, ctx, self.last_spot, self.favorite(spots))
            for h in HYPS:
                p = 0.9 * dists[h].get(spot, 0.0) + 0.1 / len(spots)
                self.score[h] = self.score[h] * DECAY + math.log(p)
            self.spot_freq = {k: v * 0.95 for k, v in self.spot_freq.items()}
            self.spot_freq[spot] = self.spot_freq.get(spot, 0.0) + 1.0
            self.last_spot = spot
        rec = {k: order.get(k) for k in ("spot", "quality", "price", "meals", "sold")}
        rec["market"], rec["day"] = market, day
        self.orders.append(rec)
        del self.orders[:-HIST]

    def favorite(self, spots):
        tot = sum(self.spot_freq.get(s, 0.0) for s in spots)
        if tot <= 0:
            return None
        return {s: self.spot_freq.get(s, 0.0) / tot for s in spots}

    # ---- prediction
    def p_active(self):
        return self.active[0] / (self.active[0] + self.active[1])

    def weights(self):
        lw = {h: self.score[h] + math.log(PRIOR[h]) for h in HYPS}
        m = max(lw.values())
        w = {h: math.exp(v - m) for h, v in lw.items()}
        t = sum(w.values())
        return {h: v / t for h, v in w.items()}

    def spot_dist(self, spots, ctx):
        dists = hypothesis_dists(spots, ctx, self.last_spot, self.favorite(spots))
        w = self.weights()
        out = {s: 0.0 for s in spots}
        for h, wh in w.items():
            for s in spots:
                out[s] += wh * dists[h].get(s, 0.0)
        t = sum(out.values()) or 1.0
        out = {s: v / t for s, v in out.items()}
        evs = [s for s in (ctx.get("events") or []) if s in spots]
        if evs:  # rival bots herd to forecast events: pull every rival toward the event spot
            out = {s: (1 - EVENT_PULL) * v + EVENT_PULL * (1.0 / len(evs) if s in evs else 0.0)
                   for s, v in out.items()}
        return {s: 0.97 * v + 0.03 / len(spots) for s, v in out.items()}

    def price_stats(self):
        """(P(price unchanged between consecutive orders), P(price cut), price volatility)."""
        ps = [o["price"] for o in self.orders if isinstance(o.get("price"), (int, float))]
        if len(ps) < 2:
            return 0.5, 0.25, 0.8
        same = sum(1 for a, b in zip(ps, ps[1:]) if abs(a - b) < 0.005)
        cut = sum(1 for a, b in zip(ps, ps[1:]) if b < a - 0.005)
        n = len(ps) - 1
        mean = sum(ps) / len(ps)
        sd = (sum((p - mean) ** 2 for p in ps) / len(ps)) ** 0.5
        return same / n, cut / n, min(max(sd, 0.1), 2.0)

    def sample_order(self, spot, rng, max_meals):
        """Sample (quality, price, meals) for a given spot from recent history."""
        if not self.orders:
            return generic_order(rng, max_meals)
        n = len(self.orders)
        weights = []
        for i, o in enumerate(self.orders):
            w = 0.75 ** (n - 1 - i)
            if o.get("spot") == spot:
                w *= 2.0
            weights.append(w)
        r = rng.random() * sum(weights)
        acc = 0.0
        pick = self.orders[-1]
        for w, o in zip(weights, self.orders):
            acc += w
            if r <= acc:
                pick = o
                break
        q = pick.get("quality") if pick.get("quality") in QUALITIES else "standard"
        try:
            price = float(pick.get("price"))
        except (TypeError, ValueError):
            price = 10.0
        p_same, p_cut, sd = self.price_stats()
        if rng.random() > p_same:
            price += rng.gauss(-0.15 if p_cut > 0.4 else 0.0, 0.5 * sd)
        price = min(max(round(price, 2), 1.0), 30.0)
        try:
            meals = int(pick.get("meals"))
        except (TypeError, ValueError):
            meals = max_meals[q]
        return q, price, max(0, min(meals, max_meals[q]))

    def summary(self, spots):
        w = self.weights()
        top = sorted(w.items(), key=lambda kv: -kv[1])[:2]
        p_same, p_cut, _ = self.price_stats()
        return "%s act=%.2f %s fixedp=%.2f cut=%.2f" % (
            self.name or self.tid, self.p_active(), " ".join("%s:%.2f" % kv for kv in top), p_same, p_cut)


def generic_order(rng, max_meals):
    """Prior for a rival we know nothing about."""
    r = rng.random()
    if r < 0.2:
        q, price = "cheap", rng.uniform(5.0, 7.5)
    elif r < 0.65:
        q, price = "standard", rng.uniform(8.5, 11.5)
    else:
        q, price = "premium", rng.uniform(10.5, 14.5)
    meals = int(max_meals[q] * (1.0 if rng.random() < 0.7 else rng.uniform(0.5, 1.0)))
    return q, round(price, 2), meals


def day_context(forecast, prev_last_day, spots):
    """Context a rival saw when choosing a spot: tomorrow's forecast and yesterday's results."""
    fc = forecast if isinstance(forecast, dict) else {}
    exp_c = fc.get("expected_customers") if isinstance(fc.get("expected_customers"), dict) else {}
    expected = {}
    for s in spots:
        try:
            expected[s] = float(exp_c.get(s, 0) or 0)
        except (TypeError, ValueError):
            expected[s] = 0.0
    events = [e.get("spot") for e in (fc.get("events") or []) if isinstance(e, dict)]
    prev = {}
    ld_spots = (prev_last_day or {}).get("spots") if isinstance(prev_last_day, dict) else None
    if isinstance(ld_spots, dict):
        for s, v in ld_spots.items():
            if isinstance(v, dict):
                prev[s] = {"trucks": len(v.get("trucks") or []), "hungry": v.get("hungry") or 0,
                           "customers": v.get("customers") or 0}
    return {"expected": expected if any(expected.values()) else {}, "events": events, "prev": prev}


# ==================== ftw/strategy.py ====================


DEFAULT_CFG = {
    "scenarios": 150,  # e44bc9e0 (00:22 ET; was 48)
    "deadline": 0.45,         # seconds of compute before using the best order found so far
    "rep_lambda": 1.0,        # $ per reputation point per remaining-day weight
    "rep_horizon": 10.0,      # days over which reputation pays back
    "rep_return": 0.5,        # baseline chance we return to a spot we sell at
    "near_tol_abs": 3.0,      # randomize among options within this many dollars...
    "near_tol_rel": 0.015,    # ...or this fraction of the best expected profit
    "price_step": 0.5,
    "min_price": 2.0,
    "max_price": 24.0,
    "price_offset": 0.0,      # added to grid prices (e.g. -0.01 gives $8.99-style prices)
    "avoid_spots": [],        # spots never considered
    "only_quality": None,     # e.g. "premium" to consider one tier only (32433ccc --quality)
    "fixed_price": None,      # e.g. 9.24: only this price; the optimizer still picks meals (5bbe5557 --price)
    "live_overrides": {},     # manual settings applied in mode == "live" only (9ceafd47)
    "meals_mult": 1.0,        # scale the optimal meal count (e.g. 1.15 after repeated sellouts) (b14dd860)
    # optional server rules (state.features); exact rules for fair choice / walk over are not published (92932fab)
    "salvage_frac": 0.25,     # salvage: unsold meals refund 25% of food cost (documented)
    "fair_delta": 1.0,        # fair choice: rivals within this value of ours share customers (guess)
    "fair_temp": 0.5,         # fair choice: share weight exp(value / temp) (guess)
    "walk_frac": 0.15,        # walk over: share of hungry customers who walk (live day 16: 10-25%) (2aa3104f)
    "premium_guard": 0.5,     # skip cheap/standard at a spot where a better-value premium rival shows up in
                              # >= this share of scenarios (live day 21: bot wanted standard vs Frontrun/Wagyu premium) (77ad3ee0)
    "gameday_rule": True,     # game day: standard, max meals, biggest spot no rival owns (live day 21: +$763) (77ad3ee0)
    "gameday_price": 9.49,
    "stronghold_rep": 70.0,   # a rival with this reputation at a spot owns it
    "latenight_rule": True,   # late-night rush at our home spot: premium, max meals (live day 23: +$286) (ee74c4bb)
    "latenight_price": 9.25,
    "home_rep": 80.0,         # our reputation at a spot that makes it "home"
    "home_day_profit": 250.0, # typical daily profit at home, used to value crowd laws when voting
    "home_meals": 60.0,       # typical daily meals at home, for valuing per-meal taxes (c46a2d0f)
    "gameday_profit": 600.0,  # typical game-day profit at the Stadium, for valuing game-day laws (638d5930)
    "gameday_every": 8.0,     # roughly one game day every N days
    "riot_cost_day": 40.0,    # $/day we'd pay to remove riot risk (wrecked truck = lost days)
    "riot_guard": True,       # on a possible riot day keep away from the event spot and its neighbours
    "riot_radius": 650.0,     # metres around the Stadium treated as the riot zone (Thompson 487 m, Oval 590 m)
    "staff_target": 3,        # crew size: 3 x 50 = 150 meals >= our biggest order (133) (c7704aeb)
    "staff_wage": 9.0,        # rivals need >= $12 to poach (lure needs +$3); auto-raised if we lose staff
    "coupon_rule": True,      # while coupons are live: $20 coupon on the truck we are racing each day
    "coupon_min_lead": 0.0,   # only target a truck at least this far ahead of us in cash
    "shared_meal_frac": 1.0,  # <1 cooks less than the optimum when rivals are expected (Arjun's idea; off)
}

STYLES = {
    "max": {},
    # camp a home spot and build reputation there (best on held-out local seeds)
    "camp": {"rep_return": 0.9, "rep_lambda": 2.0, "near_tol_abs": 0.5},
    # Cheapest Eats: low prices, odd cents; the optimizer still picks spot/quality/meals
    "cheap": {"max_price": 9.0, "price_offset": -0.01},
}



def _stable_seed(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


class Brain:
    def __init__(self, cfg=None, model=None, memory=None, seed=0):
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(cfg or {})
        self.model = model or CustomerModel()
        self.seed = seed
        self.rng = random.Random(seed)
        self.rivals = {}
        if memory:
            for d in memory.get("rivals", []):
                try:
                    r = Rival.from_json(d)
                    self.rivals[r.tid] = r
                except Exception:
                    pass
        self.market_key = None
        self.forecasts = {}       # day -> forecast seen the night before
        self.last_days = {}       # day -> last_day result
        self.processed = set()    # days already ingested in this market
        self.my_id = None
        self.last_debug = {}

    # ------------------------------------------------------------ memory
    def memory(self):
        return {"version": 1, "rivals": [r.to_json() for r in self.rivals.values()]}

    def _ensure_market(self, state):
        key = (state.get("mode"), state.get("market"))
        if key != self.market_key:
            if self.market_key is not None:
                for r in self.rivals.values():
                    r.new_market()
            self.market_key = key
            self.forecasts, self.last_days, self.processed = {}, {}, set()

    # ------------------------------------------------------------ observation
    def observe(self, state):
        """Ingest a state. Safe to call on every state; each day is processed once."""
        if not isinstance(state, dict):
            return
        self._ensure_market(state)
        you = state.get("you") if isinstance(state.get("you"), dict) else {}
        self.my_id = you.get("id") or self.my_id
        rules = Rules(state.get("rules"))
        fc = state.get("forecast")
        if isinstance(fc, dict) and isinstance(fc.get("day"), int):
            self.forecasts[fc["day"]] = fc
        ld = state.get("last_day")
        day = ld.get("day") if isinstance(ld, dict) else None
        trucks = state.get("trucks") if isinstance(state.get("trucks"), list) else []
        for t in trucks:
            if not isinstance(t, dict) or not t.get("id") or t.get("id") == self.my_id:
                continue
            r = self.rivals.get(t["id"])
            if r is None:
                r = self.rivals[t["id"]] = Rival(t["id"], t.get("truck_name", ""))
            if isinstance(t.get("reputation"), dict):
                r.rep = {s: _f(v, 50.0) for s, v in t["reputation"].items()}
        if not isinstance(day, int) or day in self.processed:
            return
        self.processed.add(day)
        self.last_days[day] = ld
        ctx = day_context(self.forecasts.get(day), self.last_days.get(day - 1), rules.spots)
        played = {}
        spots = ld.get("spots") if isinstance(ld.get("spots"), dict) else {}
        for s, v in spots.items():
            for t in (v.get("trucks") or []) if isinstance(v, dict) else []:
                if isinstance(t, dict) and t.get("id"):
                    played[t["id"]] = dict(t, spot=s)
        for tid, r in self.rivals.items():
            o = played.get(tid)
            r.observe(rules.spots, ctx, o, market=state.get("market"), day=day)
            if not any(isinstance(t, dict) and t.get("id") == tid and isinstance(t.get("reputation"), dict)
                       for t in trucks):
                _track_rep(r, o, rules.spots)

    # ------------------------------------------------------------ decision
    def decide(self, state, deadline=None):
        t0 = time.perf_counter()
        budget = self.cfg["deadline"] if deadline is None else deadline
        rules = Rules(state.get("rules") if isinstance(state, dict) else None)
        try:
            self.observe(state)
            order, debug = self._optimize(state, rules, t0 + budget)
        except Exception as e:  # never fail to produce an order
            order, debug = self._fallback(state, rules), {"error": repr(e)}
        order = rules.validate(order)
        food = choose_food(state, order.get("spot"), self.my_id)  # c7704aeb
        if food:
            order["food"] = food
        debug["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.last_debug = debug
        return order

    def _fallback(self, state, rules):
        fc = (state or {}).get("forecast") or {}
        exp_c = fc.get("expected_customers") if isinstance(fc.get("expected_customers"), dict) else None
        trucks = (state or {}).get("trucks") or []
        if not exp_c:
            exp_c = default_expected(len(trucks))
        crowd_trucks = {}
        for tid, r in self.rivals.items():
            if r.last_spot and r.p_active() > 0.5:
                crowd_trucks[r.last_spot] = crowd_trucks.get(r.last_spot, 0) + 1
        spot = max(rules.spots, key=lambda s: _f(exp_c.get(s), 0) / (1 + crowd_trucks.get(s, 0)))
        share = _f(exp_c.get(spot), 60) / (1 + crowd_trucks.get(spot, 0))
        meals = int(min(rules.max_meals("standard"), max(20, 0.75 * share)))
        return {"spot": spot, "price": 10.0, "quality": "standard", "meals": meals}

    def _rival_list(self, state):
        out = []
        for tid, r in self.rivals.items():
            if tid == self.my_id:
                continue
            out.append(r)
        return out

    def _scenarios(self, state, rules, K, deadline):
        fc = state.get("forecast") if isinstance(state.get("forecast"), dict) else {}
        exp_c = fc.get("expected_customers") if isinstance(fc.get("expected_customers"), dict) else {}
        trucks = state.get("trucks") if isinstance(state.get("trucks"), list) else []
        if not exp_c:
            exp_c = default_expected(len(trucks))
        events = [e for e in (fc.get("events") or []) if isinstance(e, dict)]
        rng = random.Random(_stable_seed(self.seed, state.get("mode"), state.get("market"),
                                         state.get("order_for_day")))
        ctx = day_context(fc, state.get("last_day"), rules.spots)
        rivals = self._rival_list(state)
        rival_pred = [(r, r.p_active(), r.spot_dist(rules.spots, ctx)) for r in rivals]
        profiles = {s: self.model.profile(s, events) for s in rules.spots}
        walk_over = bool(_features(state).get("walk_over"))
        pos = spot_positions(state)
        maxm = {q: rules.max_meals(q) for q in rules.qualities}
        scen = []
        for k in range(K):
            if k >= 8 and time.perf_counter() > deadline:
                break
            w = self.model.sample_weather(fc, rng)
            at = {s: [] for s in rules.spots}
            for r, pa, sd in rival_pred:
                if rng.random() >= pa:
                    continue
                spot = _sample(sd, rng)
                q, p, m = r.sample_order(spot, rng, maxm)
                at[spot].append(Truck(r.tid, q, p, r.rep.get(spot, 50.0), m, rules.values))
            crowds = {}
            for s in rules.spots:
                n = self.model.crowd(s, _f(exp_c.get(s), 0.0), w, fc, rng)
                med, sig, mix = profiles[s]
                crowds[s] = self.model.customers(n, med, sig, mix, rng)
            per_spot = {}
            for s in rules.spots:
                lim, mq = crowds[s]
                if walk_over:
                    # customers at spots with no food walk to the nearest spot that has food
                    fed = {x for x in rules.spots if at[x]} | {s}
                    for e in rules.spots:
                        if e not in fed and nearest(e, fed, pos) == s:
                            wl, wq = crowds[e]
                            k = int(len(wl) * self.cfg["walk_frac"])
                            lim = lim + wl[:k]
                            mq = mq + wq[:k]
                per_spot[s] = SpotScenario(lim, mq, at[s])
            scen.append(per_spot)
        return scen, rival_pred

    def _optimize(self, state, rules, deadline):
        cfg = self.cfg
        if state.get("mode") == "live" and cfg.get("live_overrides"):
            # manual live-game settings (spot/quality/price locks) never reach the final or practice
            cfg = dict(cfg, **cfg["live_overrides"])
        you = state.get("you") if isinstance(state.get("you"), dict) else {}
        my_rep = you.get("reputation") if isinstance(you.get("reputation"), dict) else {}
        gd = self._gameday_order(state, rules, cfg) if cfg.get("gameday_rule") else None  # 77ad3ee0
        if gd:
            return gd, {"why": "gameday rule"}
        day = state.get("order_for_day") or ((state.get("day") or 0) + 1)
        days = state.get("days") or 30
        remaining = max(0, int(days) - int(day))   # days after this one
        scen, rival_pred = self._scenarios(state, rules, cfg["scenarios"], deadline)
        if not scen:
            return self._fallback(state, rules), {"why": "no scenarios"}
        feats = _features(state)
        salvage = cfg["salvage_frac"] if feats.get("salvage") else 0.0
        fair = (cfg["fair_delta"], cfg["fair_temp"]) if feats.get("fair_choice") else None
        best_by_combo = {}
        evals = 0
        exp_rivals = {}
        for _r, pa, sd in rival_pred:
            for s_, p_ in sd.items():
                exp_rivals[s_] = exp_rivals.get(s_, 0.0) + pa * p_

        guard = cfg.get("premium_guard")
        move_cost = _move_costs(state, rules)  # c2fda150

        def evaluate(spot, q, price):
            nonlocal evals
            rep = _f(my_rep.get(spot), 50.0)
            value = rules.values[q] * rep_factor(rep) - price
            if guard is not None and QI[q] < QI["premium"]:  # 77ad3ee0 premium guard
                hit = sum(1 for sc in scen if any(t.qi == QI["premium"] and t.value >= value
                                                  for t in sc[spot].rivals))
                if hit >= guard * len(scen):
                    return None
            cap = rules.max_meals(q)
            qi = QI[q]
            ds = [sc[spot].demand(qi, price, value, cap, fair) for sc in scen]
            evals += 1
            cost = rules.costs[q]
            n, prof, sold = best_quantity(ds, price, cost, cap, salvage)

            def at_n(n):
                sold = sum(min(n, d) for d in ds) / len(ds)
                return sold, price * sold - cost * n + salvage * cost * (n - sold)
            if n > 0 and cfg.get("meals_mult", 1.0) != 1.0:
                n = max(0, min(cap, int(round(n * cfg["meals_mult"]))))
                sold, prof = at_n(n)
            if n > 0 and cfg["shared_meal_frac"] < 1.0 and exp_rivals.get(spot, 0.0) >= 0.5:
                n = int(n * cfg["shared_meal_frac"])
                sold, prof = at_n(n)
            bonus = 0.0
            if n > 0 and remaining > 0 and cfg["rep_lambda"]:
                new_rep = max(0.0, min(100.0, rep + REP_DELTA[q]))
                p_sell = sum(1 for d in ds if d > 0) / len(ds)
                weight = min(remaining, cfg["rep_horizon"]) / cfg["rep_horizon"]
                bonus = cfg["rep_lambda"] * (new_rep - rep) * weight * cfg["rep_return"] * p_sell * \
                    rules.values[q] / 12.0
            if n > 0:
                prof -= move_cost.get(spot, 0.0)  # moving costs: $2 + $0.01/m to leave the last spot sold at (c2fda150)
            total = prof + bonus
            key = (spot, q)
            cur = best_by_combo.get(key)
            if cur is None or total > cur[0]:
                best_by_combo[key] = (total, {"spot": spot, "quality": q, "price": round(price, 2), "meals": n},
                                      prof, sold)
            return total

        # coarse pass
        avoid = set(cfg.get("avoid_spots") or [])
        if cfg.get("riot_guard"):  # 638d5930
            avoid |= riot_zone(state, rules, cfg["riot_radius"])
        if avoid >= set(rules.spots):
            avoid = set()
        quals = [cfg["only_quality"]] if cfg.get("only_quality") in rules.qualities else rules.qualities
        combos = [(s, q) for s in rules.spots if s not in avoid for q in quals]
        thresholds = self._threshold_prices(scen, rules, my_rep)
        for s, q in combos:
            if time.perf_counter() > deadline:
                break
            ceil_p = rules.values[q] * rep_factor(_f(my_rep.get(s), 50.0)) - 0.01
            hi = min(ceil_p, cfg["max_price"], rules.price_max)
            lo = max(cfg["min_price"], rules.costs[q] + 0.25, rules.price_min)
            p = lo
            prices = []
            while p <= hi + 1e-9:
                po = round(p + cfg["price_offset"], 2)
                if lo <= po <= hi:
                    prices.append(po)
                p += cfg["price_step"]
            prices.extend(tp for tp in thresholds.get((s, q), []) if lo <= tp <= hi)
            if cfg.get("fixed_price"):
                prices = [round(float(cfg["fixed_price"]), 2)]
            for p in sorted(set(prices)):
                evaluate(s, q, p)
        # refine around the best few combos
        ranked = sorted(best_by_combo.items(), key=lambda kv: -kv[1][0])
        for (s, q), (tot, order, _, _) in ([] if cfg.get("fixed_price") else ranked[:5]):
            if time.perf_counter() > deadline:
                break
            ceil_p = rules.values[q] * rep_factor(_f(my_rep.get(s), 50.0)) - 0.01
            base = order["price"]
            for dp in (-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4):
                p = round(base + dp, 2)
                if rules.price_min <= p <= min(ceil_p, rules.price_max, cfg["max_price"]):
                    evaluate(s, q, p)
        if not best_by_combo:
            return self._fallback(state, rules), {"why": "deadline before any evaluation"}
        ranked = sorted(best_by_combo.values(), key=lambda v: -v[0])
        best_total = ranked[0][0]
        if best_total <= 0:
            # nothing profitable in expectation: cook nothing (legal, no cost)
            o = dict(ranked[0][1])
            o["meals"] = 0
            return o, {"why": "no profitable option", "best": round(best_total, 2), "evals": evals}
        tol = max(cfg["near_tol_abs"], cfg["near_tol_rel"] * best_total)
        near = [v for v in ranked if v[0] >= best_total - tol]
        if len(near) > 1:
            ws = [math.exp((v[0] - best_total) / (tol / 2.0)) for v in near]
            pick = near[_sample_idx(ws, self.rng)]
        else:
            ws, near = [1.0], [ranked[0]]
            pick = ranked[0]
        # [Frontrun port] the options it was choosing among and their odds (its pick is a draw from these)
        self.last_near = [[dict(v[1]), w / sum(ws)] for v, w in zip(near, ws)]
        debug = {"evals": evals, "scenarios": len(scen), "exp_total": round(pick[0], 2),
                 "exp_profit": round(pick[2], 2), "exp_sold": round(pick[3], 1), "near": len(near),
                 "top": [(v[1]["spot"], v[1]["quality"], v[1]["price"], v[1]["meals"], round(v[0], 1))
                         for v in ranked[:5]],
                 "rivals": [(r.name or r.tid, round(pa, 2), max(sd, key=sd.get)) for r, pa, sd in rival_pred]}
        return dict(pick[1]), debug

    def _gameday_order(self, state, rules, cfg):
        """77ad3ee0. Game day: crowds outgrow every truck. Take the biggest spot that no rival owns
        (reputation >= stronghold_rep) and isn't ours already worth more, with the most standard meals."""
        fc = state.get("forecast") if isinstance(state.get("forecast"), dict) else {}
        keys = {event_key(e) for e in (fc.get("events") or []) if isinstance(e, dict)}
        if "latenight" in keys and cfg.get("latenight_rule") and "premium" in rules.qualities:  # ee74c4bb
            mine = (state.get("you") or {}).get("reputation") or {}
            home = max(rules.spots, key=lambda s: _f(mine.get(s), 50.0))
            if home in ("highst", "northres") and _f(mine.get(home), 50.0) >= cfg["home_rep"]:
                return {"spot": home, "quality": "premium", "price": float(cfg["latenight_price"]),
                        "meals": rules.max_meals("premium")}
        if "gameday" not in keys:
            return None
        if riot_risk(state) and cfg.get("riot_guard"):  # 638d5930
            return None  # possible riot: let the optimizer play a normal day away from the riot zone
        if "standard" not in rules.qualities:
            return None
        exp_c = fc.get("expected_customers") if isinstance(fc.get("expected_customers"), dict) else {}
        if not exp_c:
            return None
        owned = set()
        for r in self._rival_list(state):
            for s, v in (r.rep or {}).items():
                if _f(v, 50.0) >= cfg["stronghold_rep"]:
                    owned.add(s)
        free = [s for s in rules.spots if s not in owned and _f(exp_c.get(s), 0) > 0]
        if not free:
            return None
        spot = max(free, key=lambda s: _f(exp_c.get(s), 0))
        meals = rules.max_meals("standard")
        if _f(exp_c.get(spot), 0) < 0.8 * meals:
            return None  # not an overflow day at any free spot: let the optimizer decide
        price = min(float(cfg["gameday_price"]), rules.values["standard"] * rep_factor(
            _f(((state.get("you") or {}).get("reputation") or {}).get(spot), 50.0)) - 0.01)
        return {"spot": spot, "quality": "standard", "price": round(price, 2), "meals": meals}

    def _threshold_prices(self, scen, rules, my_rep):
        """Prices that just beat a likely rival's value at a spot (one cent better)."""
        counts = {}
        for sc in scen:
            for s, ss in sc.items():
                for t in ss.rivals:
                    k = (s, round(t.value, 3))
                    counts[k] = counts.get(k, 0) + 1
        out = {}
        for (s, v), c in counts.items():
            if c < max(2, len(scen) // 10):
                continue
            for q in rules.qualities:
                a = rules.values[q] * rep_factor(_f(my_rep.get(s), 50.0))
                p = math.floor((a - v - 0.01) * 100) / 100.0
                out.setdefault((s, q), []).append(p)
        return out


# campus map coordinates (x, z) from the TV's campus-map.js; new spots come from state.spot_info
LAYOUT = {"oval": (100, 70), "thompson": (-12, 140), "union": (440, 388), "stadium": (-490, 48),
          "highst": (463, -186), "northres": (170, -478)}


def spot_positions(state):
    pos = dict(LAYOUT)
    si = state.get("spot_info") if isinstance(state, dict) else None
    items = si.items() if isinstance(si, dict) else (
        [((o or {}).get("key") or (o or {}).get("spot") or (o or {}).get("id"), o) for o in si]
        if isinstance(si, list) else [])
    for k, o in items:
        if not k or not isinstance(o, dict):
            continue
        xy = o.get("pos") or o.get("position") or [o.get("x"), o.get("z")]
        try:
            pos[k] = (float(xy[0]), float(xy[1]))
        except (TypeError, ValueError, IndexError):
            pass
    return pos


def nearest(spot, candidates, pos=LAYOUT):
    a = pos.get(spot)
    if a is None:
        return None  # unknown position: assume nobody walks
    known = [c for c in candidates if c in pos]
    if not known:
        return None
    return min(known, key=lambda c: (pos[c][0] - a[0]) ** 2 + (pos[c][1] - a[1]) ** 2)


def _laws_in_force(state):  # 638d5930
    pol = state.get("politics") if isinstance(state, dict) else None
    pol = pol if isinstance(pol, dict) else {}
    laws = list(pol.get("laws") or [])
    holder = ((pol.get("office") or {}).get("holder") or {}) if isinstance(pol.get("office"), dict) else {}
    laws += list(holder.get("policies") or [])
    return [l for l in laws if isinstance(l, dict)]


def riot_risk(state):
    """True when tomorrow's game could end in a riot: riots are live, the game is Michigan or a
    national championship, and no law in force switches riots off."""
    if not _features(state).get("riots"):
        return False
    fc = state.get("forecast") if isinstance(state.get("forecast"), dict) else {}
    risky = False
    for e in fc.get("events") or []:
        if not isinstance(e, dict) or event_key(e) != "gameday":
            continue
        text = " ".join(str(e.get(k, "")) for k in ("name", "effect", "opponent", "title", "detail")).lower()
        if "michigan" in text or "championship" in text or "title" in text:
            risky = True
    if not risky:
        return False
    for law in _laws_in_force(state):
        pr = law.get("params") or {}
        if "riot" in pr and _f(pr.get("riot"), 1.0) <= 0:
            return False
    return True


def riot_zone(state, rules, radius):
    """Spots within radius metres of a possible riot (the game-day event spot)."""
    if not riot_risk(state):
        return set()
    fc = state.get("forecast") or {}
    centers = [e.get("spot") for e in (fc.get("events") or []) if isinstance(e, dict) and event_key(e) == "gameday"]
    pos = spot_positions(state)
    out = set()
    for c in centers:
        if c in pos:
            for s in rules.spots:
                if s in pos and math.hypot(pos[s][0] - pos[c][0], pos[s][1] - pos[c][1]) <= radius:
                    out.add(s)
        elif c:
            out.add(c)
    return out


def choose_food(state, spot, my_id=None):
    """c7704aeb. Meat or veggie (only while the food rule is live): serve the diet whose only-eaters we can
    win, i.e. its share divided by the rivals already serving it at that spot yesterday."""
    if not spot or not _features(state).get("food"):
        return None
    diets = ((state.get("forecast") or {}).get("diets") or {}).get(spot) or {}
    meat, veg = _f(diets.get("meat"), 0.12), _f(diets.get("veggie"), 0.07)
    same = {"meat": 0, "veggie": 0}
    ld = state.get("last_day") if isinstance(state.get("last_day"), dict) else {}
    for t in (((ld.get("spots") or {}).get(spot) or {}).get("trucks") or []):
        if isinstance(t, dict) and t.get("id") != my_id and t.get("food") in same:
            same[t["food"]] += 1
    return "veggie" if veg / (1 + same["veggie"]) > meat / (1 + same["meat"]) else "meat"


def runner_staff(state, cfg, sent=None):
    """c7704aeb runner._maybe_staff: (employees, wage) it would set, or None. sent = set of keys already sent."""
    feats = state.get("features") or {}
    you = state.get("you") or {}
    st = you.get("staff") if isinstance(you.get("staff"), dict) else None
    if not feats.get("staff") or st is None:
        return None
    want_n = int(cfg.get("staff_target", 3))
    want_w = float(cfg.get("staff_wage", 9.0))
    cur_w = float(st.get("wage") or 0)
    if (st.get("employees") or 0) < want_n and st.get("target") == want_n and cur_w >= want_w:
        want_w = min(20.0, cur_w + 2.0)  # short-handed: someone poached us, pay more
    want_w = max(want_w, cur_w)
    key = ("staff", state.get("mode"), state.get("market"), want_n, want_w)
    if (sent is not None and key in sent) or (st.get("target") == want_n and abs(cur_w - want_w) < 0.01):
        return None
    if sent is not None:
        sent.add(key)
    return (want_n, want_w)


def runner_coupon(state, cfg, sent=None):
    """c7704aeb runner._maybe_coupon: truck id it would coupon tonight, or None."""
    if not cfg.get("coupon_rule"):
        return None
    if not (state.get("features") or {}).get("coupons") or state.get("phase") != "night":
        return None
    key = ("coupon", state.get("mode"), state.get("market"), state.get("order_for_day"))
    if sent is not None and key in sent:
        return None
    you = state.get("you") or {}
    me = you.get("id")
    others = [t for t in (state.get("trucks") or []) if isinstance(t, dict) and t.get("id") and t.get("id") != me]
    if not others:
        return None
    mine = you.get("cash") or 0
    ahead = [t for t in others if (t.get("cash") or 0) > mine]
    lead = min(ahead, key=lambda t: t.get("cash") or 0) if ahead else max(others, key=lambda t: t.get("cash") or 0)
    if abs((lead.get("cash") or 0) - mine) < float(cfg.get("coupon_min_lead", 0.0)):
        return None
    if sent is not None:
        sent.add(key)
    return lead["id"]


def choose_vote(state, cfg=None):
    """ee74c4bb. Pick the election candidate whose laws help us most: crowd multipliers at our home spot
    (highest reputation) valued at a typical day's profit there, plus the daily fee effect."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    pol = state.get("politics") if isinstance(state, dict) else None
    el = (pol or {}).get("election") if isinstance(pol, dict) else None
    cands = (el or {}).get("candidates") if isinstance(el, dict) else None
    if not cands:
        return None
    mine = ((state.get("you") or {}).get("reputation") or {})
    if not mine:
        return None
    home = max(mine, key=lambda s: _f(mine.get(s), 50.0))
    base = float(cfg["home_day_profit"])
    rev = base + 300.0  # rough daily revenue at home
    best, best_score = None, None
    for c in cands:
        score = 0.0
        for pol_ in c.get("policies") or []:
            pr = pol_.get("params") or {}
            crowd = pr.get("crowd") if isinstance(pr.get("crowd"), dict) else {}
            if home in crowd:
                score += base * (_f(crowd[home], 1.0) - 1.0)
            gc = pr.get("game_crowd") if isinstance(pr.get("game_crowd"), dict) else {}
            if gc:  # game-day crowd law, valued at our game-day spot (the Stadium) per day (638d5930)
                mult = _f(gc.get("stadium", gc.get("other", 1.0)), 1.0)
                score += float(cfg["gameday_profit"]) / float(cfg["gameday_every"]) * (mult - 1.0)
            pm = pr.get("per_meal") if isinstance(pr.get("per_meal"), dict) else {}
            if pm:  # per-meal tax (+) or subsidy (-) on a quality, valued at the quality we usually sell (c46a2d0f)
                q = (((state.get("last_day") or {}).get("you") or {}).get("quality")
                     or ((state.get("you") or {}).get("order") or {}).get("quality") or "standard")
                score -= _f(pm.get(q), 0.0) * float(cfg.get("home_meals", 60.0))
            if "riot" in pr:  # riot 0 removes the risk; >1 makes it worse
                score -= float(cfg["riot_cost_day"]) * (_f(pr.get("riot"), 1.0) - 1.0)
            if "flat" in pr or "revenue_share" in pr:
                above = _f(pr.get("above"), 500.0)
                score += _f(pr.get("flat"), 0.0) + _f(pr.get("revenue_share"), 0.0) * max(0.0, rev - above)
        if best_score is None or score > best_score:
            best, best_score = c.get("id"), score
    return best


def _move_costs(state, rules):
    """c2fda150. {spot: cost to park there tomorrow} while moving costs are live: base + per_metre * distance
    from the spot we last sold at (documented: $2 plus $0.01 a metre). Empty when off or unknown."""
    if not _features(state).get("moving_costs"):
        return {}
    you = state.get("you") if isinstance(state.get("you"), dict) else {}
    last = you.get("last_spot")
    pos = spot_positions(state)
    if not last or last not in pos:
        return {}
    base, per_m = 2.0, 0.01
    mc = (state.get("rules") or {}).get("move_costs")
    if isinstance(mc, dict) and isinstance(mc.get(last), dict):  # exact table: move_costs[from][to] (c7704aeb)
        row = mc[last]
        exact = {s: _f(row.get(s), None) for s in rules.spots if s != last and s in row}
        exact = {s: v for s, v in exact.items() if v is not None}
        if exact:
            return exact
    if isinstance(mc, dict):
        for k, v in mc.items():
            k = str(k).lower()
            if "per" in k or "metre" in k or "meter" in k:
                per_m = _f(v, per_m)
            elif "base" in k or "fixed" in k or "flat" in k:
                base = _f(v, base)
    a = pos[last]
    out = {}
    for s in rules.spots:
        if s == last or s not in pos:
            continue
        d = math.hypot(pos[s][0] - a[0], pos[s][1] - a[1])
        out[s] = base + per_m * d
    return out


def _features(state):
    f = state.get("features") if isinstance(state, dict) else None
    return f if isinstance(f, dict) else {}


def _track_rep(r, order, spots):
    """Update a rival's reputation estimate when the state does not expose it."""
    for s in spots:
        cur = r.rep.get(s, 50.0)
        if order and order.get("spot") == s:
            if (order.get("sold") or 0) > 0 and order.get("quality") in REP_DELTA:
                cur = max(0.0, min(100.0, cur + REP_DELTA[order["quality"]]))
        else:
            cur = cur + 0.03 * (50.0 - cur)
        r.rep[s] = cur


def _f(x, default):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _sample(dist, rng):
    r = rng.random()
    acc = 0.0
    last = None
    for k, p in dist.items():
        acc += p
        last = k
        if r <= acc:
            return k
    return last


def _sample_idx(ws, rng):
    r = rng.random() * sum(ws)
    acc = 0.0
    for i, w in enumerate(ws):
        acc += w
        if r <= acc:
            return i
    return len(ws) - 1





def _deepcopy(x):
    if isinstance(x, dict):
        return {k: _deepcopy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deepcopy(v) for v in x]
    return x


def make(seed=0, deadline=None, style="camp", avoid=(), quality=None, max_price=None, price=None, meals_mult=None):
    """policy(state) -> order. Mirrors runner.Bot: observe, then decide with the nightly deadline.

    style: bot.py --style (default "camp" at adf4fb68; "cheap" = $9 cap, "max" = plain profit).
    avoid: bot.py --avoid spots (default none).
    quality: bot.py --quality lock (32433ccc; default None = all tiers).
    max_price: bot.py --max-price cap (06d6a7cd; default None = style's cap). Falsy values are ignored, like bot.py.
    price: bot.py --price fixed price (5bbe5557). Since 9ceafd47, avoid/quality/max_price/price apply only when
    state["mode"] == "live"; the final (and practice) always runs the plain optimizer.
    """
    manual = {"avoid_spots": [x for x in avoid if x], "only_quality": quality}
    if max_price:
        manual["max_price"] = max_price
    if price:
        manual["fixed_price"] = price
    if meals_mult:
        manual["meals_mult"] = meals_mult
    # 9ceafd47: manual locks only apply in the live game; the final always runs the full optimizer
    cfg = dict(STYLES[style], live_overrides=manual)
    brain = Brain(cfg=cfg, seed=seed)

    sent = set()  # c7704aeb runner self.voted (staff/coupon keys)

    def policy(state):
        brain.observe(state)
        brain.last_near = None
        order = brain.decide(state, deadline=deadline)
        if isinstance(order, dict) and brain.last_near and order.get("meals"):
            order = dict(order, _near=brain.last_near)   # [Frontrun port] its odds over the options (predictor only)
        # ee74c4bb runner._maybe_vote: votes choose_vote(state, brain.cfg) whenever an election has candidates
        # (separate API call; reported here as an extra key).
        if isinstance(order, dict) and ((state.get("politics") or {}).get("election") or {}).get("candidates"):
            order = dict(order, vote=choose_vote(state, brain.cfg))
        if isinstance(order, dict):  # c7704aeb runner side actions (separate API calls; extra keys here)
            stf = runner_staff(state, brain.cfg, sent)
            cpn = runner_coupon(state, brain.cfg, sent)
            if stf:
                order = dict(order, staff=stf)
            if cpn:
                order = dict(order, coupon=cpn)
        return order

    policy.brain = brain
    return policy
