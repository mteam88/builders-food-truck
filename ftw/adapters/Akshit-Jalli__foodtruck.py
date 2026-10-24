"""Port of Akshit-Jalli/foodtruck bot.py decide() at SHA 1ba32c79 (2026-10-24 01:51:17 ET,
"Safety: bad-data guard, last-good-order fallback, staff capacity"), on top of 6cc7afb
(00:42:05 ET, "Bad-data sanitizer, last-good-order fallback, geometric walk flag").
Previous port: policy_6244d308.py.

Verbatim copy of the pure decision code (constants, Memory, market model, rivals, decide, the
6cc7afb FINAL RULES flags choice/salvage/walk_uplift/walk_model, the 1ba32c7 feature gating and
SERVE_CAP staff capacity) plus play()'s Sanitizer and last-good-order fallback, emulated in the
make() closure. Plumbing (HTTP, logging, CLI, preload_memory) dropped. Nothing loaded at runtime.

Defaults are bot.py's: choice "best", salvage 0.0, walk_uplift 0.0, walk_model False. The
feature gate in decide() only switches flags OFF when the server reports a rule off, so with the
shipped defaults the new rules change nothing; staff capacity (employees x 50) binds only below
2 employees for premium. Live `--set` overrides can be emulated with make(params={...}).

Cross-day memory: bot.py keeps a module-global MEM (rival per-spot reputation replay, rival spot
history, our own spot history). Here each make() gets its own Memory, swapped into the module
global MEM for the duration of the call, so instances do not share state.

Live quirk (preload_memory): on every restart bot.py replays days.jsonl, whose rows carry
`you` = last_day.you, which has no "id", so preloaded days never land in MEM.mine (our own spot
history, used for my_revisit_p -> reputation value). Only days observed live since the last
restart count. make(seed, restarts=(order days)) emulates those restarts (live mode only); a
restart also resets the Sanitizer and the last-good-order table (both live in play()).
Seed unused (deterministic).
"""
import math
import re
import time
from collections import defaultdict

QUALITIES = ("cheap", "standard", "premium")
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
QVALUE = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
QCOST = {"cheap": 2.50, "standard": 3.75, "premium": 5.50}
QREP = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}
REP_SLOPE = 0.006          # d(reputation_factor)/d(reputation point)
REP_DRIFT = 0.03           # fraction of the gap to 50 closed per day away

# Per-spot customer priors, from the docs and refined by fit.py against
# observed play. limit_mean/sd describe the private price ceiling; qmin is
# P(minimum quality is cheap / standard / premium).
SPOTS = {
    "highst":   {"limit_mean": 12.0, "limit_sd": 3.0, "qmin": (0.50, 0.35, 0.15)},
    "oval":     {"limit_mean": 12.4, "limit_sd": 2.3, "qmin": (0.30, 0.50, 0.20)},
    "union":    {"limit_mean": 12.8, "limit_sd": 2.2, "qmin": (0.22, 0.47, 0.31)},
    "northres": {"limit_mean": 11.0, "limit_sd": 2.1, "qmin": (0.45, 0.45, 0.10)},
    "thompson": {"limit_mean": 12.0, "limit_sd": 2.3, "qmin": (0.45, 0.40, 0.15)},
    "stadium":  {"limit_mean": 13.0, "limit_sd": 2.6, "qmin": (0.25, 0.50, 0.25)},
}

# fit.py's maximum-likelihood priors, from 854 designed observations, cut
# out-of-sample demand error from 4.99 to 4.41 meals RMSE (better at 5 of 6
# spots) - and then lost $238/market over 8 paired seeds, 2 wins to 6. Better
# average demand prediction is not better decisions: profit turns on exactly
# where the price-limit curve falls away, not on the average fit. Not adopted.
FITTED_SPOTS = {
    "highst":   {"limit_mean": 12.12, "limit_sd": 3.16, "qmin": (0.50, 0.32, 0.18)},
    "oval":     {"limit_mean": 12.65, "limit_sd": 2.52, "qmin": (0.33, 0.48, 0.19)},
    "union":    {"limit_mean": 12.80, "limit_sd": 1.98, "qmin": (0.28, 0.41, 0.31)},
    "northres": {"limit_mean": 11.25, "limit_sd": 2.41, "qmin": (0.47, 0.42, 0.11)},
    "thompson": {"limit_mean": 12.00, "limit_sd": 2.83, "qmin": (0.41, 0.44, 0.15)},
    "stadium":  {"limit_mean": 13.50, "limit_sd": 2.63, "qmin": (0.34, 0.40, 0.27)},
}

# 341e1eff: named prior sets, selectable with P["priors"] (bot.py --set priors="'dfl'"); default "orig".
DFL_SPOTS = {
        "highst":    {"limit_mean": 12.77, "limit_sd": 3.02, "qmin": (0.5000, 0.3462, 0.1537)},
        "oval":      {"limit_mean": 13.18, "limit_sd": 2.33, "qmin": (0.3038, 0.4975, 0.1988)},
        "union":     {"limit_mean": 13.55, "limit_sd": 2.17, "qmin": (0.2275, 0.4625, 0.3100)},
        "northres":  {"limit_mean": 11.78, "limit_sd": 2.14, "qmin": (0.4525, 0.4462, 0.1013)},
        "thompson":  {"limit_mean": 12.75, "limit_sd": 2.37, "qmin": (0.4450, 0.4050, 0.1500)},
        "stadium":   {"limit_mean": 13.81, "limit_sd": 2.60, "qmin": (0.2609, 0.4869, 0.2522)},
}

PRIOR_SETS = {
    "orig": {s: dict(c) for s, c in SPOTS.items()},
    "rmse": {s: dict(c) for s, c in FITTED_SPOTS.items()},
    "dfl": {s: dict(c) for s, c in DFL_SPOTS.items()},
}
_ACTIVE_PRIORS = "orig"


def use_priors(name):
    """Make `name` the active customer priors. Returns the name."""
    global SPOTS, _ACTIVE_PRIORS
    if name not in PRIOR_SETS:
        raise KeyError("unknown prior set %r (have: %s)"
                       % (name, ", ".join(sorted(PRIOR_SETS))))
    SPOTS = {s: dict(c) for s, c in PRIOR_SETS[name].items()}
    _ACTIVE_PRIORS = name
    return name

P = {
    # which PRIOR_SETS entry decide() uses; re-read every night
    "priors": "dfl",  # 2cc9322 (23:34 ET): ship the decision-focused priors
    # risk
    "risk_lambda": 0.55,     # weight on the bad tail
    "risk_tail": 0.30,       # what counts as the bad tail
    # uncertainty
    "crowd_mults": ((0.70, 0.15), (0.85, 0.20), (1.00, 0.30), (1.15, 0.20), (1.35, 0.15)),
    "intruder_p": 0.45,      # chance an unpredicted rival shows up at our spot
    # night one: rivals' opening offers sit at value +1.50 at rep 50 (None = old $11.00/standard guess)
    "day1_iso_value": 1.50,
    # Rivals do not repeat their price to the cent, and winning the value
    # contest is a cliff: a rival shaving 25c can take every customer. So
    # carry a spread of rival prices and prefer a pitch that survives it.
    "rival_price_shifts": ((-0.60, 0.30), (0.0, 0.50), (0.45, 0.20)),
    # reputation
    "rep_revisit_floor": 0.25,
    "spot_repeat_penalty": 0.0,   # $ of score x my_revisit_p(spot); 0.0 = off (default)
    "rep_max_horizon": 15,
    "rep_discount": 0.97,
    "rep_bonus": 5.0,        # only used when use_repvalue is off
    "rep_horizon": 12.0,
    # search
    "time_budget_s": 1.0,    # the final allows 2s a night, including the round trip
    "price_step": 0.25,
    "shortlist": 18,
    "limit_cells": 18,
    # feature flags, for ablation
    "use_reptracker": True,     # reconstruct rivals' per-spot reputation
    "use_mobility": True,       # probabilistic rival locations
    "use_repvalue": True,       # value-function reputation credit
    "stage1_newsvendor": True,  # inventory-aware screening
    "dedup_intruder": True,     # do not clone a rival onto its own spot
    # Measured: costs ~3.6% against the house bots (0 wins in 8 paired seeds),
    # because it gives up margin to guard against undercutting they never try.
    # Kept for the live field, where rivals do move their prices.
    "use_price_jitter": False,
    "use_herding": True,        # rivals pile onto the night's obvious spot
    # FINAL RULES. "best" = documented winner-take-all (today's behaviour).
    # "share" = each customer picks among eligible trucks with probability
    # value_j / sum(values). salvage = fraction of food cost refunded on
    # unsold meals.
    "choice": "best",           # "best" | "share" | "logit"
    "choice_tau": 3.0,          # logit temperature, fitted on practice: tau~3
    "salvage": 0.0,
    # Walk-over: hungry customers from empty spots walk to occupied ones.
    # Practice showed served spots gaining ~20-60% extra; this inflates every
    # spot's expected crowd by this fraction. 0.0 = off.
    "walk_uplift": 0.0,
    # Geometric walk-over model (replaces walk_uplift when on). Fitted on 38
    # empty-spot practice days: the share of an empty spot's crowd that walks
    # to its nearest occupied spot is walk_a * exp(-dist / walk_scale), with
    # dist in projected-map pixels (rms 7.7 pts).
    "walk_model": False,
    "walk_a": 0.55,
    "walk_scale": 300.0,
    "herd_p": 0.40,
}


def repfactor(rep):
    return 0.7 + 0.6 * rep / 100.0


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def norm_ppf(p):
    """Inverse normal CDF (Acklam's approximation)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def cells_for(spot, n_cells):
    """Discretise the customers into (weight, min_rank, price_limit) cells."""
    cfg = SPOTS[spot]
    out = []
    for rank, pq in enumerate(cfg["qmin"]):
        if pq <= 0:
            continue
        for i in range(n_cells):
            q = (i + 0.5) / n_cells
            limit = cfg["limit_mean"] + cfg["limit_sd"] * norm_ppf(q)
            out.append((pq / n_cells, rank, max(1.0, limit)))
    return out


# Meals a truck can serve in a day, if the server imposes a limit ("each truck
# serves 50 meals a day"). None = no limit beyond the budget. Set per night by
# decide() from the state, so a rule that comes and goes is followed exactly.
SERVE_CAP = None


def cap_meals(n):
    return int(n) if SERVE_CAP is None else int(min(n, SERVE_CAP))


def budget_max(quality, budget=500.0):
    return cap_meals(budget // QCOST[quality])


def detect_serve_cap(state):
    """Meals we can serve tonight: our staff capacity (employees x
    meals_per_employee), as the server reports it. None when the staff rule
    is off or not reported. NOTE: '50 meals a day' is PER EMPLOYEE, not per
    truck; the default 4 staff serve 200, which never binds a $500 budget
    except at cheap quality."""
    try:
        if (state.get("features") or {}).get("staff") is False:
            return None
        staff = (state.get("you") or {}).get("staff") or {}
        cap = staff.get("capacity")
        if not _is_num(cap):
            cap = ((state.get("last_day") or {}).get("you") or {}).get("capacity")
        if _is_num(cap) and cap > 0:
            return int(cap)
    except Exception:  # noqa: BLE001
        pass
    return None


# Each spot's share of the crowd on an ordinary clear day, from the docs.
SHARES = {"highst": 0.24, "oval": 0.20, "union": 0.18,
          "northres": 0.16, "thompson": 0.14, "stadium": 0.08}


def expected_crowds(state, spots):
    """Tomorrow's crowd per spot, with a sane floor if the forecast is absent.

    A missing or all-zero forecast is a data fault, not an empty campus: the
    market is about 110 customers per truck over at least 4 trucks. Falling
    back to the documented shares beats both guessing zero (sit out for
    nothing) and the old blind 133-meal default (risk the whole budget).
    """
    exp = ((state.get("forecast") or {}).get("expected_customers")) or {}
    up = 1.0 + float(P["walk_uplift"])
    out = {s: float(exp.get(s) or 0.0) * up for s in spots}
    if sum(out.values()) > 0:
        return out
    trucks = max(4, len(state.get("trucks") or []) or 4)
    total = 110.0 * trucks
    return {s: total * SHARES.get(s, 1.0 / max(1, len(spots))) for s in spots}


# Spot positions on the projected campus map (isometric screen px, approximate).
SPOT_XY = {"stadium": (325, 900), "thompson": (610, 1105), "oval": (750, 1125),
           "highst": (1140, 1165), "northres": (1133, 968), "union": (778, 1468)}


def _dist(a, b):
    (x1, y1), (x2, y2) = SPOT_XY[a], SPOT_XY[b]
    return math.hypot(x1 - x2, y1 - y2)


def walk_adjust(expected, rivals, spots):
    """Crowd at each spot *if we park there*, counting walk-ins.

    A spot A with no truck sends a share of its crowd to the nearest occupied
    spot. Parking at s collects A's walkers when A is empty and every spot
    nearer to A than s is empty too. Rival occupancy comes from their parking
    history (rival_presence), treated as independent across spots.
    """
    if not P["walk_model"]:
        return expected
    occ = {}
    for a in spots:
        empty = 1.0
        for _, p in rival_presence(rivals, a):
            empty *= (1.0 - min(1.0, p))
        occ[a] = 1.0 - empty
    out = {}
    for s in spots:
        if s not in SPOT_XY:
            out[s] = expected.get(s, 0.0)
            continue
        gain = 0.0
        for a in spots:
            if a == s or a not in SPOT_XY:
                continue
            d = _dist(a, s)
            p = 1.0 - occ[a]
            for b in spots:
                if b not in (a, s) and b in SPOT_XY and _dist(a, b) < d:
                    p *= (1.0 - occ[b])
            gain += float(expected.get(a, 0.0)) * p * P["walk_a"] * math.exp(-d / P["walk_scale"])
        out[s] = float(expected.get(s, 0.0)) + gain
    return out


# ------------------------------------------------------------------ memory

class Memory:
    """What we have seen. Rivals' per-spot reputation and where they park.

    Reputation is private in the state we receive - `trucks` only reports an
    average across all six spots - but its dynamics are deterministic and
    every truck's spot and quality is public in last_day. So we replay the
    rules and recover each rival's reputation at each spot exactly.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.rep = defaultdict(dict)        # truck id -> spot -> reputation
        self.spots = defaultdict(list)      # truck id -> [spot, newest last]
        self.mine = []                      # our own spots, newest last
        self.key = None
        self.days = 0

    def _market_key(self, state):
        return (state.get("mode"), state.get("market"))

    def observe_day(self, last, me=None):
        """Fold one completed day into memory. `last` is a last_day object."""
        spots = last.get("spots") or {}
        sold_at = {}
        for spot, info in spots.items():
            for t in info.get("trucks", []):
                tid = t.get("id")
                if not tid:
                    continue
                sold_at[(tid, spot)] = (float(t.get("sold") or 0), t.get("quality", "standard"))
                self.spots[tid].append(spot)
                if me and tid == me:
                    self.mine.append(spot)
        seen = {tid for tid, _ in sold_at}
        for tid in set(list(self.rep.keys())) | seen:
            cur = self.rep[tid]
            for spot in SPOTS:
                r = cur.get(spot, 50.0)
                hit = sold_at.get((tid, spot))
                if hit is None:
                    r += (50.0 - r) * REP_DRIFT     # away: drift back toward 50
                elif hit[0] >= 1:
                    r = clamp(r + QREP[hit[1]], 0.0, 100.0)
                # Parked but sold nothing: reputation does not move at all.
                # The docs say unvisited spots drift, which reads as though a
                # shut-out day drifts too. It does not. Replaying 960 observed
                # truck-days: drifting on a shut-out mismatches the server
                # 175 times (max 12.07 points), holding steady mismatches 0.
                cur[spot] = r
        self.days += 1

    def sync(self, state):
        """Called each night. Resets between markets, folds in the new day."""
        k = self._market_key(state)
        last = state.get("last_day") or {}
        day = last.get("day")
        seen = getattr(self, "_last_seen", None)
        # 341e1eff: a day that runs backwards, or no history when we hold some, is a new market
        fresh = ((day is None and self.days > 0)
                 or (day is not None and seen is not None and day < seen))
        if k != self.key or fresh:
            self.reset()
            self.key = k
            seen = None
        if day and day != seen:
            self._last_seen = day
            self.observe_day(last, me=(state.get("you") or {}).get("id"))

    def rep_at(self, tid, spot, fallback=50.0):
        if not P["use_reptracker"]:
            return fallback
        got = self.rep.get(tid)
        if not got or spot not in got:
            return fallback
        return got[spot]

    def spot_probs(self, tid, half_life=4.0):
        """Where this truck parks, weighted to recent days, Laplace-smoothed."""
        hist = self.spots.get(tid) or []
        if not hist:
            return None
        w = defaultdict(float)
        total = 0.0
        for i, spot in enumerate(reversed(hist)):
            wt = 0.5 ** (i / half_life)
            w[spot] += wt
            total += wt
        alpha = 1.5
        n = len(SPOTS)
        return {s: (w.get(s, 0.0) + alpha / n) / (total + alpha) for s in SPOTS}

    def my_revisit_p(self, spot):
        """How often we come back to a spot, used to value reputation there."""
        if not self.mine:
            return 0.5
        recent = self.mine[-10:]
        p = sum(1 for s in recent if s == spot) / float(len(recent))
        return max(P["rep_revisit_floor"], p)




# ------------------------------------------------------------------ market model

class Truck:
    __slots__ = ("value", "price", "qrank", "cap", "tid")

    def __init__(self, value, price, qrank, cap, tid=None):
        self.value = value
        self.price = price
        self.qrank = qrank
        self.cap = cap
        self.tid = tid


def beats(a, b):
    """Customer prefers a over b: higher value, ties to the cheaper truck."""
    if a.value > b.value + 1e-9:
        return True
    if a.value < b.value - 1e-9:
        return False
    return a.price < b.price


def allocate_share(cells, trucks, n_customers):
    """Value-weighted choice: each customer splits over eligible trucks in
    proportion to value; customers whose pick sold out choose again."""
    sales = [0.0] * len(trucks)
    cap = [t.cap for t in trucks]
    weights = [w * n_customers for (w, _, _) in cells]
    logit = P["choice"] == "logit"
    tau = float(P["choice_tau"])
    for _ in range(6):
        demand = defaultdict(float)
        parts = {}
        for ci, (_, rank, limit) in enumerate(cells):
            w = weights[ci]
            if w <= 1e-9:
                continue
            el = [(ti, t.value) for ti, t in enumerate(trucks)
                  if cap[ti] > 1e-9 and t.qrank >= rank and t.price <= limit
                  and (t.value >= 0 if logit else t.value > 0)]
            if logit and el:
                top = max(v for _, v in el)
                el = [(ti, math.exp((v - top) / tau)) for ti, v in el]
            tot = sum(v for _, v in el)
            if tot <= 0:
                weights[ci] = 0.0
                continue
            parts[ci] = [(ti, v / tot) for ti, v in el]
            for ti, sh in parts[ci]:
                demand[ti] += w * sh
        if not demand:
            break
        frac = {ti: (1.0 if d <= cap[ti] else cap[ti] / d) for ti, d in demand.items()}
        for ti, d in demand.items():
            take = d * frac[ti]
            sales[ti] += take
            cap[ti] -= take
        spill = False
        for ci in range(len(weights)):
            if ci not in parts:
                weights[ci] = 0.0
                continue
            weights[ci] *= sum(sh * (1.0 - frac[ti]) for ti, sh in parts[ci])
            if weights[ci] > 1e-9:
                spill = True
        if not spill:
            break
    return sales


def allocate(cells, trucks, n_customers):
    """Assign customers to trucks, respecting sell-outs. Sales per truck."""
    if P["choice"] in ("share", "logit"):
        return allocate_share(cells, trucks, n_customers)
    sales = [0.0] * len(trucks)
    cap = [t.cap for t in trucks]
    weights = [w * n_customers for (w, _, _) in cells]

    for _ in range(5):
        demand = defaultdict(float)
        choice = {}
        for ci, (_, rank, limit) in enumerate(cells):
            w = weights[ci]
            if w <= 1e-9:
                continue
            best = -1
            for ti, t in enumerate(trucks):
                if cap[ti] <= 1e-9 or t.qrank < rank or t.price > limit or t.value < 0:
                    continue
                if best < 0 or beats(t, trucks[best]):
                    best = ti
            if best >= 0:
                demand[best] += w
                choice[ci] = best
        if not demand:
            break
        frac = {ti: (1.0 if d <= cap[ti] else cap[ti] / d) for ti, d in demand.items()}
        for ti, d in demand.items():
            take = d * frac[ti]
            sales[ti] += take
            cap[ti] -= take
        spill = False
        for ci in range(len(weights)):
            ti = choice.get(ci)
            if ti is None:
                weights[ci] = 0.0
                continue
            weights[ci] *= (1.0 - frac[ti])
            if weights[ci] > 1e-9:
                spill = True
        if not spill:
            break
    return sales


def tail_mean(outcomes, frac):
    """Average profit over the worst `frac` of the probability mass (CVaR)."""
    xs = sorted(outcomes)
    acc, tot = 0.0, 0.0
    for p, w in xs:
        take = min(w, frac - tot)
        if take <= 1e-12:
            break
        acc += p * take
        tot += take
    return acc / tot if tot > 0 else xs[0][0]


def meal_grid(demands, hardcap):
    """Expected profit is piecewise linear in meals, kinked at each scenario's
    demand, so the optimum sits on one of those points (or 0, or the cap)."""
    pts = {0, hardcap}
    for d, _ in demands:
        pts.add(int(clamp(round(d), 0, hardcap)))
    return sorted(pts)


def best_inventory(demands, price, cost, hardcap):
    """Newsvendor over the scenario demands, penalising the bad tail.
    Returns (meals, risk_adjusted_score, expected_profit)."""
    best = (0, -1e18, 0.0)
    for m in meal_grid(demands, hardcap):
        sv = P["salvage"] * cost
        outcomes = [(min(d, m) * price - m * cost + sv * max(0.0, m - d), w)
                    for d, w in demands]
        mean = sum(p * w for p, w in outcomes)
        score = mean - P["risk_lambda"] * (mean - tail_mean(outcomes, P["risk_tail"]))
        if score > best[1]:
            best = (m, score, mean)
    return best


# ------------------------------------------------------------------ rivals

def predict_rivals(state):
    """Each rival's likely pitch tomorrow, from what we have seen them do."""
    me = (state.get("you") or {}).get("id")
    rep_avg = {t["id"]: t.get("reputation_avg", 50.0) for t in state.get("trucks", [])}
    last = state.get("last_day") or {}
    out = []
    for spot, info in (last.get("spots") or {}).items():
        for t in info.get("trucks", []):
            if t["id"] == me:
                continue
            q = t.get("quality", "standard")
            out.append({
                "id": t["id"], "spot": spot, "price": float(t.get("price", 11.0)),
                "quality": q,
                "rep": MEM.rep_at(t["id"], spot, rep_avg.get(t["id"], 50.0)),
                "meals": float(t.get("meals") or budget_max(q)),
                "probs": MEM.spot_probs(t["id"]),
            })
    if not out:
        # Nothing observed yet: night one. Field spreads over the busy spots; every opening
        # offer sits on the value +1.50 line at rep 50 (cheap $6.50, standard $10.50, premium $14.50).
        others = [t for t in state.get("trucks", []) if t["id"] != me]
        ring = ["highst", "oval", "union", "northres", "thompson", "stadium"]
        opening = QUALITIES if P["day1_iso_value"] is not None else None
        for i, t in enumerate(others):
            if opening:
                q = opening[i % len(opening)]
                price = QVALUE[q] * repfactor(50.0) - float(P["day1_iso_value"])
            else:
                q, price = "standard", 11.0
            out.append({"id": t["id"], "spot": ring[i % len(ring)], "price": price,
                        "quality": q, "rep": 50.0, "meals": 133.0, "probs": None})
    return out


def rival_presence(rivals, spot, herd=False, magnet=None):
    """(rival, probability it is at this spot) for everyone who plausibly is.

    Rivals do not choose independently. They all read the same forecast, so
    on a night with an obvious draw - a concert, a game day - the field piles
    onto it together and leaves the rest of campus unserved. Treating their
    choices as independent makes every spot look mildly contested and hides
    the empty ones, which on such a night are where the money is.
    """
    out = []
    for r in rivals:
        if herd and magnet:
            p = 0.85 if spot == magnet else 0.05
        elif P["use_mobility"] and r.get("probs"):
            p = r["probs"].get(spot, 0.0)
        else:
            p = 1.0 if r["spot"] == spot else 0.0
        if p >= 0.12:
            out.append((r, p))
    return out


def strongest_rival(rivals):
    """The most dangerous truck in the field: what it costs us if it moves in."""
    best = None
    for r in rivals:
        v = QVALUE[r["quality"]] * repfactor(r["rep"]) - r["price"]
        if best is None or v > best[0]:
            best = (v, r)
    if best is None:
        return Truck(QVALUE["standard"] - 10.5, 10.5, 1, 133.0, tid="__generic__")
    v, r = best
    return Truck(v, r["price"], QRANK[r["quality"]], r["meals"], tid=r["id"])


def rival_trucks_at(rivals, spot, intruder=None, shift=0.0, herd=False, magnet=None):
    """The field we expect to face at a spot.

    A rival that only sometimes parks here is carried at a fraction of its
    capacity: over the scenarios that is the competition it actually removes.
    `shift` moves every rival's price, to test whether our pitch survives a
    field that turns out cheaper than it was yesterday.
    """
    ts = []
    here = set()
    for r, p in rival_presence(rivals, spot, herd, magnet):
        rep = MEM.rep_at(r["id"], spot, r["rep"])
        price = max(1.0, r["price"] + shift)
        v = QVALUE[r["quality"]] * repfactor(rep) - price
        ts.append(Truck(v, price, QRANK[r["quality"]], r["meals"] * p, tid=r["id"]))
        here.add(r["id"])
    if intruder is not None:
        # Do not invent a second copy of a truck that is already standing here.
        if not (P["dedup_intruder"] and intruder.tid in here):
            ip = max(1.0, intruder.price + shift)
            ts.append(Truck(intruder.value + (intruder.price - ip), ip,
                            intruder.qrank, intruder.cap, tid=intruder.tid))
    return ts


# ------------------------------------------------------------------ the decision

def scenarios():
    """(crowd multiplier, probability, intruder?, rival price shift, herd?)."""
    shifts = P["rival_price_shifts"] if P["use_price_jitter"] else ((0.0, 1.0),)
    herds = ((False, 1.0 - P["herd_p"]), (True, P["herd_p"])) if P["use_herding"] else ((False, 1.0),)
    out = []
    for mult, w in P["crowd_mults"]:
        for ds, ws in shifts:
            for hd, wh in herds:
                out.append((mult, w * (1 - P["intruder_p"]) * ws * wh, False, ds, hd))
                out.append((mult, w * P["intruder_p"] * ws * wh, True, ds, hd))
    return out


def magnet_spot(state, spots):
    """The spot the whole field can see is tonight's draw."""
    fc = state.get("forecast") or {}
    for ev in (fc.get("events") or []):
        if ev.get("spot") in spots:
            return ev["spot"]
    exp = fc.get("expected_customers") or {}
    cand = [(exp.get(s, 0), s) for s in spots]
    if not cand:
        return None
    n, best = max(cand)
    total = sum(x for x, _ in cand) or 1.0
    # Only a genuinely outsized crowd pulls the field together.
    return best if n > 0.28 * total else None


def rival_best_per_cell(cells, trucks):
    """The offer each kind of customer would otherwise take. Under value-
    weighted choice: the summed value of the rivals they would consider."""
    if P["choice"] == "logit":
        tau = float(P["choice_tau"])
        return [sum(math.exp(t.value / tau) for t in trucks
                    if t.qrank >= rank and t.price <= limit and t.value >= 0 and t.cap > 1e-9)
                for (_, rank, limit) in cells]
    if P["choice"] == "share":
        return [sum(t.value for t in trucks
                    if t.qrank >= rank and t.price <= limit and t.value > 0 and t.cap > 1e-9)
                for (_, rank, limit) in cells]
    arr = []
    for (_, rank, limit) in cells:
        best = None
        for t in trucks:
            if t.qrank < rank or t.price > limit or t.value < 0:
                continue
            if best is None or beats(t, best):
                best = t
        arr.append(best)
    return arr


def fast_demand(cells, rbest, mine, n):
    """Customers who would pick us, ignoring who runs out of food first."""
    d = 0.0
    if P["choice"] == "logit":
        if mine.value < 0:
            return 0.0
        e = math.exp(mine.value / float(P["choice_tau"]))
        for ci, (cw, rank, limit) in enumerate(cells):
            if mine.qrank < rank or mine.price > limit:
                continue
            d += cw * n * e / (e + rbest[ci])
        return d
    if P["choice"] == "share":
        if mine.value <= 0:
            return 0.0
        for ci, (cw, rank, limit) in enumerate(cells):
            if mine.qrank < rank or mine.price > limit:
                continue
            d += cw * n * mine.value / (mine.value + rbest[ci])
        return d
    for ci, (cw, rank, limit) in enumerate(cells):
        if mine.qrank < rank or mine.price > limit:
            continue
        b = rbest[ci]
        if b is None or beats(mine, b):
            d += cw * n
    return d


def best_at_rep(spot, cells, rbest_modal, n_modal, rep, budget, pmin, pmax):
    """Best achievable day-profit at a spot for a given reputation.

    Used twice - at today's reputation and at tomorrow's - so the worth of a
    reputation point is read off the game's own mechanics instead of guessed.
    """
    best = 0.0
    for quality in QUALITIES:
        qv = QVALUE[quality] * repfactor(rep)
        cost = QCOST[quality]
        hardcap = cap_meals(budget // cost)
        price = max(pmin, 5.0)
        top = min(pmax, qv)
        while price <= top + 1e-9:
            mine = Truck(qv - price, price, QRANK[quality], hardcap)
            d = fast_demand(cells, rbest_modal, mine, n_modal)
            m = int(clamp(round(d), 0, hardcap))
            profit = min(d, m) * price - m * cost
            if profit > best:
                best = profit
            price += 0.5
    return best


# Each new-rule flag only applies when the server says that rule is on. If a
# market reports a rule as off (features.<rule> == false), the matching flag
# falls back to the documented behaviour for that night, so a wrong guess
# about which rules the final uses cannot cost us anything.
_FEATURE_FLAGS = (("salvage", "salvage", 0.0),
                  ("fair_choice", "choice", "best"),
                  ("walk_over", "walk_model", False))


def decide(state):
    global SERVE_CAP
    SERVE_CAP = detect_serve_cap(state) if isinstance(state, dict) else None
    feats = state.get("features") if isinstance(state, dict) else None
    saved = {}
    if isinstance(feats, dict):
        for feat, key, off in _FEATURE_FLAGS:
            if feats.get(feat) is False:
                saved[key] = P[key]
                P[key] = off
        # The refund rate is announced in text ("...sold back for 25% of their
        # food cost"). If the server ever changes it, follow the announcement.
        if feats.get("salvage") and P["salvage"] > 0:
            pct = _announced_salvage(state)
            if pct is not None:
                saved.setdefault("salvage", P["salvage"])
                P["salvage"] = pct
    try:
        return _decide_core(state)
    finally:
        P.update(saved)


def _announced_salvage(state):
    try:
        latest = None
        for rc in state.get("rule_changes") or []:
            if rc.get("name") == "salvage" and rc.get("on"):
                if latest is None or str(rc.get("at", "")) > str(latest.get("at", "")):
                    latest = rc
        if latest:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", latest.get("text") or "")
            if m and 0 < float(m.group(1)) < 100:
                return float(m.group(1)) / 100.0
    except Exception:  # noqa: BLE001
        pass
    return None


def _decide_core(state):
    """Pick tomorrow's spot, price, quality and meal count. Runs to a deadline (time_budget_s)."""
    if P["priors"] != _ACTIVE_PRIORS:
        use_priors(P["priors"])
    t0 = time.time()
    deadline = t0 + float(P["time_budget_s"])
    MEM.sync(state)
    rules = state["rules"]
    budget = float(rules.get("budget", 500))
    spots = [s for s in rules.get("spots", list(SPOTS)) if s in SPOTS]
    pmin = float(rules.get("price_min", 1))
    pmax = float(rules.get("price_max", 30))
    expected = expected_crowds(state, spots)
    myrep = (state["you"].get("reputation") or {})
    rivals = predict_rivals(state)
    intruder = strongest_rival(rivals)
    expected = walk_adjust(expected, rivals, spots)

    # order_for_day is null outside a night, and `days` can be missing.
    day_now = int(state.get("order_for_day") or state.get("day") or 1)
    days_left = max(1, int(state.get("days") or 30) - day_now + 1)
    scen = scenarios()
    magnet = magnet_spot(state, spots) if P["use_herding"] else None

    # ---- stage 1: screen every spot/quality/price without capacity feedback.
    shortlist = []
    cellcache, rbestcache, modal = {}, {}, {}
    for spot in spots:
        n_base = float(expected.get(spot, 150))
        if n_base <= 0:
            continue
        # Half the budget for screening; leave the rest for the real search.
        if shortlist and time.time() > t0 + 0.5 * (deadline - t0):
            break
        cells = cellcache.setdefault(spot, cells_for(spot, P["limit_cells"]))
        rep = float(myrep.get(spot, 50.0))
        rf = repfactor(rep)
        rbest = {k: rival_best_per_cell(
                     cells, rival_trucks_at(rivals, spot, intruder if k[0] else None,
                                            k[1], k[2], magnet))
                 for k in {(e, ds, hd) for _, _, e, ds, hd in scen}}
        rbestcache[spot] = rbest
        modal[spot] = (cells, rival_best_per_cell(
            cells, rival_trucks_at(rivals, spot, None, 0.0)), n_base)
        for quality in QUALITIES:
            qv = QVALUE[quality] * rf
            cost = QCOST[quality]
            hardcap = cap_meals(budget // cost)
            price = max(pmin, 4.0)
            top = min(pmax, qv)
            while price <= top + 1e-9:
                mine = Truck(qv - price, price, QRANK[quality], hardcap)
                demands = [(fast_demand(cells, rbest[(e, ds, hd)], mine, n_base * mult), w)
                           for mult, w, e, ds, hd in scen]
                if P["stage1_newsvendor"]:
                    _, score, _ = best_inventory(demands, price, cost, hardcap)
                else:
                    exp_sales = sum(w * min(d, hardcap) for d, w in demands)
                    score = exp_sales * price - hardcap * cost
                shortlist.append((score, spot, quality, price))
                price += P["price_step"]
    if not shortlist:
        return {"spot": "union", "price": 11.0, "quality": "standard", "meals": 133}
    shortlist.sort(key=lambda x: -x[0])
    shortlist = shortlist[:P["shortlist"]]

    # ---- how much is a day's reputation gain worth at each shortlisted spot?
    repvalue = {}
    if P["use_repvalue"]:
        horizon = int(min(days_left - 1, P["rep_max_horizon"]))
        for spot in {s for _, s, _, _ in shortlist}:
            cells, rbest0, n_base = modal[spot]
            rep = float(myrep.get(spot, 50.0))
            base = best_at_rep(spot, cells, rbest0, n_base, rep, budget, pmin, pmax)
            visits = sum(MEM.my_revisit_p(spot) * (P["rep_discount"] ** k)
                         for k in range(horizon))
            for quality in QUALITIES:
                nxt = clamp(rep + QREP[quality], 0.0, 100.0)
                if abs(nxt - rep) < 1e-9 or visits <= 0:
                    repvalue[(spot, quality)] = 0.0
                    continue
                gain = best_at_rep(spot, cells, rbest0, n_base, nxt, budget, pmin, pmax) - base
                repvalue[(spot, quality)] = gain * visits
    else:
        w = P["rep_bonus"] * min(days_left, P["rep_horizon"]) / P["rep_horizon"]
        for _, spot, quality, _ in shortlist:
            repvalue[(spot, quality)] = w * QREP[quality]

    # ---- stage 2: full simulation with sell-outs, then choose the stock.
    best = None
    for _, spot, quality, price in shortlist:
        if best is not None and time.time() > deadline:
            break
        n_base = float(expected.get(spot, 150))
        cells = cellcache[spot]
        rep = float(myrep.get(spot, 50.0))
        myv = QVALUE[quality] * repfactor(rep) - price
        cost = QCOST[quality]
        hardcap = cap_meals(budget // cost)

        demands = []
        for mult, w, extra, ds, hd in scen:
            rts = rival_trucks_at(rivals, spot, intruder if extra else None, ds, hd, magnet)
            mine = Truck(myv, price, QRANK[quality], 1e9)
            demands.append((allocate(cells, [mine] + rts, n_base * mult)[0], w))

        meals, score, mean = best_inventory(demands, price, cost, hardcap)
        if meals > 0:
            score += repvalue.get((spot, quality), 0.0)
        if P["spot_repeat_penalty"]:
            score -= P["spot_repeat_penalty"] * MEM.my_revisit_p(spot)
        if best is None or score > best[0]:
            best = (score, mean, spot, quality, price, meals)

    _, mean, spot, quality, price, meals = best
    if mean < 0 and meals > 0:
        # Every option loses money today; sitting out costs nothing.
        return {"spot": spot, "price": round(price, 2), "quality": quality, "meals": 0}
    return {"spot": spot, "price": round(price, 2), "quality": quality, "meals": int(meals)}


# --- bad-data guard ---------------------------------------------------------
# Some practice markets (state.data_quality.bad_data) serve corrupted fields:
# numbers as None or as "$10.50" strings. Repair them from the last clean state
# of the same market before anything reads the state. Clean input passes
# through unchanged (tested on practice_states.jsonl).
_NUM_RE = re.compile(r'^\s*-?\$?\s*-?[\d,]*\.?\d+\s*%?\s*$')


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _list_key(items):
    if items and all(isinstance(d, dict) for d in items):
        for k in ("id", "spot", "key"):
            if all(isinstance(d.get(k), str) for d in items):
                return k
    return None


def _fillable(path):
    # Top-level fields (order_for_day, phase_ends_at, ...) and our own last-day
    # record legitimately go None; only `version` is worth carrying forward.
    if path.count(".") == 1:
        return path == ".version"
    if path.startswith(".politics") or path.startswith(".labor"):
        return False
    return not path.startswith(".last_day.you")


def _repair(s, prev, path, fixes):
    if isinstance(s, dict):
        p = prev if isinstance(prev, dict) else {}
        return {k: _repair(v, p.get(k), path + "." + k, fixes) for k, v in s.items()}
    if isinstance(s, list):
        p = prev if isinstance(prev, list) else []
        k = _list_key(s)
        if k and _list_key(p) == k:
            by = {d[k]: d for d in p}
            out = [_repair(v, by.get(v[k]), "%s[%s]" % (path, v[k]), fixes) for v in s]
        else:
            out = [_repair(v, p[i] if i < len(p) else None, "%s[%d]" % (path, i), fixes)
                   for i, v in enumerate(s)]
        if k and _fillable(path + ".x"):
            # A rival with no history here (it moved spots): borrow the median
            # of its neighbours rather than leave a None for the model to trip on.
            for f in {f for d in out for f, v in d.items() if v is None}:
                nums = sorted(d[f] for d in out if _is_num(d.get(f)))
                if nums:
                    med = nums[len(nums) // 2]
                    for d in out:
                        if d.get(f) is None:
                            d[f] = med
                            fixes.append("%s[%s].%s None->median" % (path, d[k], f))
        return out
    if s is None and _is_num(prev) and _fillable(path):
        fixes.append(path + " None")
        return prev
    if isinstance(s, str) and _NUM_RE.match(s) and (_is_num(prev) or "$" in s):
        try:
            v = float(s.replace("$", "").replace(",", "").replace("%", "").strip())
        except ValueError:
            return s
        fixes.append("%s %r" % (path, s))
        return int(v) if isinstance(prev, int) and v.is_integer() else v
    return s


class Sanitizer:
    def __init__(self):
        self.prev, self.key = None, None

    def __call__(self, state):
        try:
            key = (state.get("mode"), state.get("market"))
            prev = self.prev if key == self.key else None
            # Every practice market says market=1: a day running backwards
            # means a new market, whose numbers must not leak from the last.
            day, pday = state.get("day"), (prev or {}).get("day")
            if _is_num(day) and _is_num(pday) and day < pday:
                prev = None
            fixes = []
            clean = _repair(state, prev, "", fixes)
            self.prev, self.key = clean, key
            return clean, fixes
        except Exception as e:  # noqa: BLE001 - never let the guard cost a night
            return state, ["sanitize failed: %s" % e]


def fallback_order(state):
    """A modest order for a night on which decide() itself blew up."""
    try:
        exp = ((state.get("forecast") or {}).get("expected_customers")) or {}
        spot = max(exp, key=lambda s: float(exp[s] or 0)) if exp else "highst"
    except Exception:  # noqa: BLE001
        spot = "highst"
    if spot not in SPOTS:
        spot = "highst"
    return {"spot": spot, "price": 10.5, "quality": "premium", "meals": 50}


MEM = None

# Live restarts (supervise.py re-running bot.py -> preload_memory on days.jsonl) right before the
# order for these days. Their console shows restarts before days 3, 4 and 5, but days 3-4 ran
# earlier local code (decide 0.09-0.18 s vs 0.27-0.30 s now; with a restart at 4 this port says
# highst $10 x90, they sent union $10.25 x87). Day 5's order was sent twice: union $10.75 x79,
# then after the 21:26:35 restart highst $9.00 x90 (the one that counted, exact here). Only the
# latest restart matters for later days. Append a day when their console shows another restart.
# Day 6 (union premium $9.50 x90) is exact only with a restart before it too (22:00 ET run).
# 13: restarted in night 12 (~23:25 ET, same minute their recorder.py came back) onto dfl priors
# (shipped as 2cc9322 at 23:34), preloading a days.jsonl with days 6-11 missing: days 13-27 all exact.
# Restarts onto 6cc7afb (first possible order: day 20) and 1ba32c7 (day 27) leave every order
# unchanged (rival replay identical, our recent spots all union), so they are not listed.
LIVE_RESTARTS = (5, 6, 10, 13)  # 10: restarted after skipping day 9


def archived_days(order_day):
    """Days in IBE's days.jsonl (written by their recorder.py, a separate process) when a restart
    before `order_day` runs preload_memory(). From the copies committed in their repo: the
    recorder archived days 1-5 (last at 21:30 ET), died on a 502, and only came back at 23:25:45 ET
    (night 12), archiving day 12 and everything after it. Days 6-11 are missing for good, so every
    restart from day 7 on rebuilds rival reputation/spot history with a hole in it."""
    if int(order_day) <= 12:
        # Empirical: the day-10 restart (night 9, ~22:40 ET) reproduces exactly only with full rival
        # history (days 1-9), so the local file still had days 6-9 then; by the 23:25 restart it had
        # been replaced by the 1-5 copy (e.g. a git checkout/pull of the committed file).
        return set(range(1, int(order_day)))
    return set(range(1, 6)) | set(range(12, int(order_day)))


def make(seed=0, restarts=LIVE_RESTARTS, params=None, archive=archived_days):
    """params: optional P overrides for this instance (like bot.py --set K=V), applied per call.
    archive: order_day -> set of days preload_memory() finds in days.jsonl at that restart
    (None = every day seen so far)."""
    mem = Memory()
    done = set()
    seen = {}  # day -> last_day, everything this "process tree" has observed (= the server's days)
    # play() state, per process: rebuilt on every (emulated) restart.
    proc = {"sanitize": Sanitizer(), "last_good": {}}

    def policy(state):
        global MEM
        day = state.get("order_for_day")
        ld = state.get("last_day") or {}
        if ld.get("day"):
            seen[ld["day"]] = ld
        if state.get("mode") == "live" and day in restarts and day not in done:
            # preload_memory(): MEM.reset(), then observe_day() every archived row with me=None
            # (days.jsonl rows carry `you` = last_day.you, which has no id), so our own spot
            # history (MEM.mine, used by my_revisit_p) is empty after every restart, and rival
            # history only covers the archived days. decide()'s sync then folds tonight's
            # last_day in (with our id) if the archive does not have it yet.
            # The new process also starts with a fresh Sanitizer and an empty last-good-order table.
            done.add(day)
            keep = set(seen) if archive is None else archive(day)
            mem.reset()
            mem.key = (state.get("mode"), state.get("market"))
            for d in sorted(seen):
                if d in keep:
                    mem.observe_day(seen[d], me=None)
                    mem._last_seen = d
            proc["sanitize"], proc["last_good"] = Sanitizer(), {}
        # 6cc7afb play(): every fetched state goes through the bad-data guard first. (Live, the
        # Sanitizer also sees the day-phase polls; here it only sees nights, so `prev` is the
        # previous night. Identical on clean data, which is all the live market has served.)
        state, _fixes = proc["sanitize"](state)
        state = _validate_shims(state)
        saved = {}
        if params:
            saved = {k: P[k] for k in params}
            P.update(params)
        prev, MEM = MEM, mem
        mkey = (state.get("mode"), state.get("market"))
        try:
            order = decide(state)
            proc["last_good"][mkey] = dict(order)
        except Exception:  # noqa: BLE001 - 6cc7afb: a model bug must not cost a night
            prev_order = proc["last_good"].get(mkey)
            order = dict(prev_order) if prev_order else fallback_order(state)
        finally:
            MEM = prev
            P.update(saved)
        return order
    return policy


def _validate_shims(state):
    """Harness-only: validate.py's as_seen_by() rebuilds `you` without the live `staff` block.
    Live, the server sends you.staff.capacity = employees x meals_per_employee; rebuild it from
    the public trucks row so detect_serve_cap() sees what IBE saw. No-op when staff is present,
    the rule is off, or the trucks row has no employees (sim)."""
    try:
        if (state.get("features") or {}).get("staff") is not True:
            return state
        you = state.get("you") or {}
        if you.get("staff"):
            return state
        row = next((t for t in state.get("trucks") or [] if t.get("id") == you.get("id")), None)
        emp = (row or {}).get("employees")
        per = (((state.get("rules") or {}).get("staff") or {}).get("meals_per_employee")) or 50
        if _is_num(emp):
            you = dict(you, staff={"employees": emp, "wage": row.get("wage"), "capacity": int(emp * per)})
            state = dict(state, you=you)
    except Exception:  # noqa: BLE001
        pass
    return state
