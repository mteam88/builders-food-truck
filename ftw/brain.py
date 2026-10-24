"""Frontrun Fries' decision, in ten sentences.

1. Each night we predict every rival's order (its own code run on tonight's state as it sees it; a random
   rival as a spread over the options its code draws from); if a rival's prediction for yesterday was
   wrong, or it did not answer, we assume it repeats yesterday.
2. For every spot, quality and price on a grid, we compute our expected sales against those orders
   with the game's customer model (fair choice, sell-outs, walk-over), over weather and crowd noise.
3. Meals come from the newsvendor rule: the count that maximises expected revenue + salvage - food cost,
   capped by the food budget and our staff (50 meals per employee).
4. Profit = that, minus the moving cost from our last spot, minus wages, minus the expected riot bill
   when the forecast warns of a riot near the spot.
5. Reputation is a moat: each point at a spot is worth 0.006 x quality value per meal sold there on
   every later day, so a gain today is valued as points x 0.006 x 16 x expected sales x horizon days.
6. Points stop counting at 100 and cheap food's -3 is a cost.
7. Commitment: leaving the spot we sold at yesterday costs a $150 score penalty while our reputation
   there is below 70 (we are still building it), and `stay_margin` dollars once it is above.
8. If nothing has positive expected profit, or the best spot is inside a warned riot's radius, we sit
   out (no order: safe from riots, keeps our spot), but never two nights in a row, because a truck with
   no order for 2 nights loses its crew; after a sit-out we open at the best spot outside any riot.
9. Laws in force (price cap/floor, per-meal taxes) adjust the price we are charged and our margin.
10. Memory is per market and is reset when the market changes; any error falls back to a safe order.
"""
import math

from . import game as G
from .clean import sanitize

DEFAULTS = {
    "rep_horizon": 3.0,      # days a reputation point keeps paying (capped by days left)
    "rep_value": "model",    # "model": rep's future value from the day model; "legacy": fixed formula + penalties
    "rep_scale": 1.0,        # multiplier on the model's reputation value
    "rep_top": 12,           # candidates whose reputation value is computed exactly (plus the best per spot/quality)
    "vfit_g": 30.4,          # vfit: $/pt (30 nights left) when far ahead of the best rival at the spot (7-night fit)
    "vfit_lin": -2.1,        # vfit: $/pt regardless of rivals
    "stay_margin": 40.0,     # legacy only: $ an alternative must beat staying by, once our rep there is >= commit_rep
    "commit_penalty": 150.0, # legacy only: $ while our rep there is < commit_rep
    "commit_rep": 70.0,
    "unsure_spread": 0.5,    # weight an unpredicted rival spreads over other spots instead of repeating yesterday
    "trust_mode": "last",    # "last": code prediction used iff it was right last night (spot, quality, price);
                             # "spot": iff its spot was right last night; "rate": mixed with persistence
                             # by its running spot hit rate; "sticky": used while that hit rate >= trust_min
    "trust_prior": (2.0, 1.0),  # rate/sticky: Beta prior (hits, misses) on a rival's code spot hit rate
    "trust_min": 0.5,        # sticky: hit rate needed to keep using the code prediction
    "use_dist": True,        # a randomized rival (Cheapest Eats): spread it over the options its samples drew from, by
                             # their odds (rehearsal 2026-10-24, seeds 1-20: +704 +-268 with 6 seeds; peek +758)
    "dist_trust": True,      # a randomized rival counts as right if one of those options has its real spot and quality
                             # (else it typed by hand: assume it repeats yesterday)
    "use_planned": True,     # use tonight's planned counts (features.planned_counts) when the state has them
    "late_weight": 0.15,     # weight of a rival not yet in the planned counts at its likely spot
    "riot_sitout": "safe",   # warned riot: open at the best spot outside its radius (True: sit out; False: EV only)
    "wreck_value": 400.0,    # $ per night lost to a wrecked truck, or "auto": tonight's best expected profit
    "sit_negative": True,    # sit out when the best order's expected profit after wages is negative
    "move_hurdle": 0.0,      # $ a new spot must beat staying at last night's (sold-at) spot by
    "first_spots": None,     # night 1 (no last spot yet): only these spots (e.g. the indoor ones), if open
    "price_swing": None,     # max |price change| vs last night while staying at the same spot (None: free)
    "price_lo": 5.0,
    "price_hi": 16.0,
    "price_step": 0.25,
    "qualities": ("standard", "premium", "cheap"),
}

RIOTS = {  # center, P(trigger), intensity, radius
    "highst": (0.45, 0.7, 260.0),
    "stadium": (0.80, 0.95, 380.0),
}


def _laws(state):
    cap, floor, tax = None, None, {}
    for law in ((state.get("politics") or {}).get("laws") or []):
        if not isinstance(law, dict):
            continue
        p = law.get("params") or {}
        if "max_price" in p:
            cap = G.num(p["max_price"], None)
        if "min_price" in p:
            floor = G.num(p["min_price"], None)
        for q, v in (p.get("per_meal") or {}).items():
            if q in G.COSTS:
                tax[q] = tax.get(q, 0.0) + G.num(v)
    return cap, floor, tax


def riot_risk(state):
    """{spot: (p_trigger, intensity, radius, centre)} for riots warned in tomorrow's forecast."""
    out = []
    laws = [str((l or {}).get("key")) for l in ((state.get("politics") or {}).get("laws") or []) if isinstance(l, dict)]
    if "crackdown" in laws:
        return out
    boost = 1.5 if "tailgate" in laws else 1.0
    for e in ((state.get("forecast") or {}).get("events") or []):
        if not isinstance(e, dict):
            continue
        text = str(e.get("effect", "")).lower()
        if "riot" not in text:
            continue
        centre = "highst" if "high street" in text else "stadium"
        p, inten, rad = RIOTS[centre]
        out.append((p, min(1.0, inten * boost), rad, centre))
    return out


class Brain:
    """policy(state) -> order or None. Keeps per-market memory of rival predictions."""

    wants_peek = True   # in the simulator, we may be handed the rivals' real orders
    posts_last = True   # in the simulator, we decide after the rivals (planned counts)

    def __init__(self, params=None, predictor=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.predictor = predictor
        self.market = None
        self.reset()

    def reset(self):
        self.pred = {}          # day -> {rival id: predicted order}
        self.used = {}          # day -> {rival id: (order we assumed, "code" | "persist")}
        self.trust = {}         # rival id -> bool: was its last prediction right?
        self.hitrate = {}       # rival id -> [spot hits, nights, last night hit] of its code prediction (nights it answered)
        self.sat_out = 0        # consecutive nights we sent no order
        self.day = None         # the night being decided (deciding it again, e.g. with planned counts, is fine)
        self.sat_before = 0
        self.sold_last = False
        self.last_order = None
        self.notes = []

    # ------------------------------------------------------------------ state helpers
    def __call__(self, state):
        return self.decide(state)

    def _market_key(self, state):
        return (state.get("mode"), state.get("market_id") or state.get("market"))

    def observe(self, state):
        if "_peek" not in state:
            state = sanitize(state)
        key = self._market_key(state)
        day = G.num(state.get("order_for_day"), 0)
        if key != self.market or (self.pred and day < max(self.pred)):
            self.market = key          # a new market (or the day went back): forget everything
            self.reset()
        ld = state.get("last_day") or {}
        d = ld.get("day")
        if d is None:
            return {}
        actual = {}
        for spot, v in (ld.get("spots") or {}).items():
            for t in (v or {}).get("trucks") or []:
                if isinstance(t, dict) and t.get("id") and G.num(t.get("meals"), 0) > 0:
                    actual[t["id"]] = {"spot": spot, "price": G.num(t.get("price"), 10.0),
                                       "quality": t.get("quality"), "meals": int(G.num(t.get("meals"), 0)),
                                       "sold": G.num(t.get("sold"), 0)}
        guessed = self.pred.get(d)
        if guessed is not None:
            for tid in set(guessed) | set(actual):
                g, a = guessed.get(tid), actual.get(tid)
                if tid == ((state.get("you") or {}).get("id")):
                    continue
                dist = g.get("_dist") if isinstance(g, dict) and self.p["dist_trust"] else None
                if g is None and a is None:
                    self.trust[tid] = True
                elif g is None or a is None:
                    self.trust[tid] = False
                elif isinstance(dist, list) and dist:
                    # randomized rival: right if some sample chose its spot and quality (else it typed by hand)
                    self.trust[tid] = any(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], dict)
                                          and x[0].get("spot") == a["spot"] and x[0].get("quality") == a["quality"]
                                          for x in dist)
                else:
                    self.trust[tid] = (g.get("spot") == a["spot"] and g.get("quality") == a["quality"]
                                       and abs(G.num(g.get("price")) - a["price"]) <= 0.30)
                if tid in guessed:
                    h = self.hitrate.setdefault(tid, [0, 0, 1])
                    hit = int((g is None and a is None) or (isinstance(g, dict) and a is not None
                                                            and g.get("spot") == a["spot"]))
                    h[0], h[1], h[2] = h[0] + hit, h[1] + 1, hit
        return actual

    # ------------------------------------------------------------------ decision
    def decide(self, state, predictions=None):
        raw, state = state, sanitize(state)
        if not isinstance(predictions, dict):
            predictions = None
        else:
            predictions = {k: v for k, v in predictions.items() if isinstance(k, str)}
        you = state.get("you")
        if not isinstance(you, dict) or not state.get("orders_open", True):
            return None
        actual = self.observe(state)
        self.last_actual = actual
        if state.get("order_for_day") == self.day:
            self.sat_out = self.sat_before
        else:
            self.day, self.sat_before = state.get("order_for_day"), self.sat_out
        me = you.get("id")
        ld_day = (state.get("last_day") or {}).get("day")
        seen_open = set(actual)
        spots = [s for s in ((state.get("rules") or {}).get("spots") or []) if isinstance(s, str)]
        if not spots:
            return None
        if predictions is None:
            predictions = state.get("_peek")
        if predictions is None and self.predictor is not None:
            predictions = self.predictor(raw)      # as live: rivals' code sees the raw state, not our cleaned copy
        predictions = dict(predictions or {})
        day = state.get("order_for_day")
        self.pred[day] = {k: v for k, v in predictions.items() if k != me}
        rivals = {}
        others = [t for t in (state.get("trucks") or []) if isinstance(t, dict) and t.get("id") not in (None, me)]
        exp = (state.get("forecast") or {}).get("expected_customers") or {}
        tot = sum(max(0.0, G.num(exp.get(s), 0.0)) for s in spots) or 1.0
        unsure = self.p["unsure_spread"]
        n_today = 1.0
        plans = []      # (truck row, meal cap, [(order, weight)], template order)
        for t in others:
            tid = t["id"]
            emp = t.get("employees")
            cap = G.num(emp, 4) * G.MEALS_PER_EMPLOYEE if emp is not None else 1e9
            o = predictions.get(tid, "missing")
            pieces = []
            w_code = 1.0 if o != "missing" and self.trust.get(tid, True) else 0.0
            mode = self.p["trust_mode"]
            if o != "missing" and mode == "spot":
                w_code = float(self.hitrate.get(tid, (0, 0, 1))[2])
            elif o != "missing" and mode in ("rate", "sticky"):
                a0, b0 = self.p["trust_prior"]
                h, n = self.hitrate.get(tid, (0, 0, 1))[:2]
                rate = (h + a0) / max(1e-9, n + a0 + b0)
                w_code = rate if mode == "rate" else float(rate >= self.p["trust_min"] or self.trust.get(tid, True))
            if w_code >= 1.0:
                pieces = self._code_pieces(o, 1.0)
            else:
                # unknown: it repeats yesterday with weight 1 - unsure, else spreads over spots by crowd
                o = actual.get(tid)
                if not (isinstance(o, dict) and o.get("spot") in spots):
                    o, w0 = {"price": 10.5, "quality": "premium", "meals": 90}, 0.0
                    if ld_day is not None and tid not in seen_open:
                        w0 = 1.0 - unsure          # closed yesterday: probably closed again
                        o = None
                else:
                    w0 = 1.0 - unsure
                if o is not None:
                    pieces = [(o, w0)] if w0 > 0 else []
                    rest = 1.0 - w0
                    if rest > 0:
                        pieces += [(dict(o, spot=s2), rest * max(0.0, G.num(exp.get(s2), 0.0)) / tot)
                                   for s2 in spots if s2 != o.get("spot") or w0 <= 0]
                if w_code > 0:
                    code = predictions.get(tid)
                    pieces = (self._code_pieces(code, w_code)
                              + [(x, w * (1.0 - w_code)) for x, w in pieces])
                    if isinstance(code, dict):
                        o = code
                    pieces.sort(key=lambda pw: -pw[1])
            plans.append((t, cap, pieces, o))
        planned = (state.get("forecast") or {}).get("planned")
        if isinstance(planned, dict) and self.p["use_planned"]:
            plans = self.place_planned(plans, planned, spots, state.get("_posted"))
        # what we assumed for each rival tonight (heaviest piece): for hit-rate reports only
        self.used[day] = {t["id"]: (max(pieces, key=lambda pw: pw[1])[0] if pieces else None,
                                    "code" if predictions.get(t["id"], "missing") != "missing"
                                    and (self.trust.get(t["id"], True) or self.p["trust_mode"] != "last")
                                    else "persist")
                          for t, _, pieces, _ in plans}
        for t, cap, pieces, _ in plans:
            tid = t["id"]
            for k, (o, w) in enumerate(pieces):
                if isinstance(o, dict) and o.get("spot") in spots and o.get("quality") in G.COSTS and w > 1e-3:
                    stock = min(G.num(o.get("meals"), 0), cap)
                    if k == 0:
                        n_today += min(1.0, stock * G.COSTS[o["quality"]] / 250.0)
                    rivals["%s#%d" % (tid, k)] = {
                        "spot": o["spot"], "price": G.num(o.get("price"), 10.0), "quality": o["quality"],
                        "stock": stock * w, "rep": G.num((t.get("reputation") or {}).get(o["spot"]), 50.0)}
        # The forecast counts trucks by yesterday's food spend; today's spend sets the real crowd.
        spend = lambda q, m: min(1.0, G.num(m) * G.COSTS.get(q, 5.5) / 250.0)
        n_yest = sum(spend(o["quality"], o["meals"]) for o in actual.values()) if actual else 0.0
        self.crowd_scale = max(4.0, n_today) / max(4.0, n_yest) if actual else 1.0
        mine = actual.get(me)
        self.sold_last = bool(mine) and mine["spot"] == you.get("last_spot") and mine.get("sold", 0) > 0
        self.mine = mine
        order, info = self.best_order(state, spots, rivals)
        self.last_info = info.get("score", (0, 0))[1] if order is not None else None
        self.last_best = info.get("best") or info.get("fallback")    # what we would open with if we must
        if order is None:
            if self.sat_out >= 1 and info.get("fallback"):
                order = info["fallback"]     # never two nights without an order: the crew would quit
            else:
                self.sat_out += 1
                self.notes.append(("sit", day, info.get("best"), info.get("why")))
                return None
        self.sat_out = 0
        self.last_order = order
        return order

    def _code_pieces(self, o, w):
        """[(order, weight)] for a rival's code prediction o (None = closed) given total weight w."""
        if not isinstance(o, dict):
            return []
        dist = o.get("_dist") if self.p["use_dist"] else None
        if isinstance(dist, list):
            out = [(x[0], w * G.num(x[1], 0.0)) for x in dist
                   if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], dict)]
            out = [(x, wx) for x, wx in out if wx > 0]
            if out:
                return out
        return [(o, w)]

    def place_planned(self, plans, planned, spots, posted):
        """Tonight's planned counts (orders in per spot) beat predictions: trusted predictions keep their spot
        while it has room, the remaining rivals fill the remaining counts (preferring a spot they were
        expected at), and rivals not counted anywhere get a small late-order weight at their likely spot."""
        slots = {s: int(G.num(planned.get(s), 0)) for s in spots}
        if posted in slots:
            slots[posted] = max(0, slots[posted] - 1)      # our own order is in the count
        out, pool = [], []
        for t, cap, pieces, tmpl in plans:
            if len(pieces) == 1 and pieces[0][1] >= 1.0 and slots.get(pieces[0][0].get("spot"), 0) > 0:
                slots[pieces[0][0]["spot"]] -= 1
                out.append((t, cap, pieces, tmpl))
            else:
                likely = max(pieces, key=lambda pw: pw[1])[0] if pieces else None
                pool.append((t, cap, likely if isinstance(likely, dict) else None))
        for s in sorted(spots, key=lambda s: -slots[s]):
            while slots[s] > 0:
                k = next((i for i, (_, _, o) in enumerate(pool) if o and o.get("spot") == s), 0 if pool else None)
                if k is None:      # more orders than known rivals: an unknown truck
                    out.append(({"id": "_x%s%d" % (s, slots[s]), "reputation": {}}, 1e9,
                                [({"spot": s, "price": 10.5, "quality": "premium", "meals": 90}, 1.0)], None))
                else:
                    t, cap, o = pool.pop(k)
                    o = dict(o or {"price": 10.5, "quality": "premium", "meals": 90}, spot=s)
                    out.append((t, cap, [(o, 1.0)], o))
                slots[s] -= 1
        late = self.p["late_weight"]
        for t, cap, o in pool:
            out.append((t, cap, [(o, late)] if o and late > 0 else [], o))
        return out

    def best_order(self, state, spots, rivals):
        P = self.p
        you = state["you"]
        fc = state.get("forecast") or {}
        exp = fc.get("expected_customers") or {}
        events = fc.get("events") or []
        events = [e for e in events if isinstance(e, dict)]
        feats = state.get("features") or {}
        fair = feats.get("fair_choice", True) is not False
        walk = bool(feats.get("walk_over", False))
        rep = you.get("reputation") or {}
        staff = you.get("staff") or {}
        cap = G.num(staff.get("capacity"), 0) or G.num(staff.get("employees"), 4) * G.MEALS_PER_EMPLOYEE
        if not feats.get("staff"):
            cap = 1e9
        wages = G.num(staff.get("employees"), 0) * G.num(staff.get("wage"), 0) if feats.get("staff") else 0.0
        if not cap:     # no crew tonight (all quit after 2 nights without an order, e.g. back from a wreck): the
            heads = G.num(staff.get("target"), 4)      # morning hires up to target before sales, so open with it
            cap, wages = heads * G.MEALS_PER_EMPLOYEE, heads * G.num(staff.get("wage"), 0)
        last = you.get("last_spot")
        move_costs = (state.get("rules") or {}).get("move_costs") or {}
        days_left = max(0, int(G.num(state.get("days"), 30)) - int(G.num(state.get("order_for_day"), 1)))
        horizon = min(P["rep_horizon"], days_left)
        price_cap, price_floor, tax = _laws(state)
        riots = riot_risk(state)
        pos = {}
        for s, v in (state.get("spot_info") or {}).items():
            if isinstance(v, dict):
                pos[s] = (G.num(v.get("x")), G.num(v.get("z")))
        for s in spots:
            pos.setdefault(s, G.POSITIONS.get(s, (0.0, 0.0)))

        def dist(a, b):
            pa, pb = pos.get(a, (0.0, 0.0)), pos.get(b, (0.0, 0.0))     # junk last_spot must not crash
            return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

        by_spot = {s: [] for s in spots}
        for tid, r in rivals.items():
            by_spot[r["spot"]].append(dict(r, id=tid, value=G.value_of(r["quality"], r["rep"], r["price"])))
        crowd = {s: max(0.0, G.num(exp.get(s), 0.0)) * getattr(self, "crowd_scale", 1.0) for s in spots}
        # hungry customers each spot leaves behind with the predicted rivals (walk-over pool)
        hungry = {}
        if walk:
            for s in spots:
                params = G.spot_params(s, events)
                tr = by_spot[s]
                groups = G.customer_groups(params, tr)
                _, h = G.fluid(groups, tr, crowd[s], fair)
                hungry[s] = [(h[i], g[2], params) for i, g in enumerate(groups) if h[i] > 0.05]

        def walk_pool(s):
            """Walkers from other spots who could reach s: [(mass*walk_p, lo, hi, minq, mu, sd, rival weight)]."""
            out = []
            if not walk:
                return out
            for s2, hs in hungry.items():
                if s2 == s:
                    continue
                d = dist(s, s2)
                for mass, (lo, hi, minq), (mu, sd, _, _) in hs:
                    mid = min(max(lo, mu - 2 * sd), hi) if hi < 1e8 else max(lo, mu)
                    # a nearer spot with a truck they would accept takes them first
                    if any(dist(s2, s3) < d and any(t["price"] <= mid and G.QRANK[t["quality"]] >= minq
                                                    and t["value"] >= 0 for t in by_spot[s3])
                           for s3 in spots if s3 not in (s, s2)):
                        continue
                    w_riv = sum(math.exp(t["value"] / G.FAIR_TEMP) for t in by_spot[s]
                                if t["price"] <= mid and G.QRANK[t["quality"]] >= minq and t["value"] >= 0)
                    plo = G.phi((lo - mu) / sd) if lo > -1e8 else 0.0
                    phi_ = G.phi((hi - mu) / sd) if hi < 1e8 else 1.0
                    out.append((mass * G.walk_prob(d), lo, hi, minq, mu, sd, w_riv, max(1e-9, phi_ - plo)))
            return out

        def walk_in(pool, q, price, value):
            """Expected walkers from other spots who would pick us."""
            if value < 0:
                return 0.0
            total = 0.0
            w_me = math.exp(value / G.FAIR_TEMP)
            for mass, lo, hi, minq, mu, sd, w_riv, den in pool:
                if G.QRANK[q] < minq:
                    continue
                a = max(lo, price)
                if a >= hi:
                    continue
                pb = G.phi((hi - mu) / sd) if hi < 1e8 else 1.0
                frac = (pb - G.phi((a - mu) / sd)) / den
                total += mass * frac * (w_me / (w_me + w_riv) if fair else 1.0)
            return total

        home = last if (last in spots and self.sold_last) else None
        leave = 0.0
        if home is not None and P["rep_value"] == "legacy":
            leave = P["commit_penalty"] if G.num(rep.get(home), 50) < P["commit_rep"] else P["stay_margin"]
        ctx = {}

        def evaluate(s, q, price, r):
            """(expected food profit, meals, expected sold) at spot s with our reputation r there."""
            c = ctx[s]
            value = G.value_of(q, r, price)
            if value < 0:
                return 0.0, 0, 0.0
            me = {"price": price, "quality": q, "value": value, "stock": float("inf")}
            trucks = c["tr"] + [me]
            groups = G.customer_groups(c["params"], trucks)
            curve = G.sales_curve(groups, trucks, len(trucks) - 1, c["mean_c"] * 1.6 + 1, fair)
            w_in = walk_in(c["pool"], q, price, value)
            demands = [(p, G.curve_at(curve, x) + w_in * x / c["mean_c"]) for p, x in c["scen"]]
            charged = price_floor if price_floor is not None and price < price_floor else price
            ev, meals = G.best_meals(charged - tax.get(q, 0.0), G.COSTS[q], demands, cap, G.max_meals(q))
            return ev, meals, sum(p * min(d, meals, cap) for p, d in demands)

        cands = []      # (profit, legacy rep bonus, spot, quality, price, meals, sold, in riot, food profit)
        wreck_w = {}
        first = [s for s in (P["first_spots"] or ()) if s in spots and crowd[s] > 0] if not last else []
        for s in spots:
            if crowd[s] <= 0 or (first and s not in first):
                continue
            tr = by_spot[s]
            my_rep = max(0.0, min(100.0, G.num(rep.get(s), 50.0)))
            move = 0.0
            if feats.get("moving_costs") and last and last != s:
                move = G.num((move_costs.get(last) or {}).get(s), 2 + 0.01 * dist(last, s))
            riot = 0.0
            in_riot = False
            wreck_w[s] = 0.0
            for p_trig, inten, rad, centre in riots:
                if centre in pos:
                    in_riot = in_riot or dist(s, centre) < rad
                    hit = min(1.0, inten * math.exp(-dist(s, centre) / rad))
                    riot += p_trig * hit * 0.7 * (30 + 120 * math.exp(-dist(s, centre) / rad))
                    wreck_w[s] += p_trig * hit * 0.3          # P(wrecked) at s, times the wreck's $ below
            scen = G.crowd_scenarios(crowd[s], fc.get("weather"), s)
            ctx[s] = {"tr": tr, "params": G.spot_params(s, events), "scen": scen, "pool": walk_pool(s),
                      "mean_c": sum(p * c for p, c in scen) or 1.0, "rep": my_rep}
            for q in P["qualities"]:
                for price in self._prices(tr, price_cap):
                    ev, meals, sold = evaluate(s, q, price, my_rep)
                    if meals <= 0:
                        continue
                    gain = max(-my_rep, min(G.REP_DELTA[q], 100.0 - my_rep))
                    legacy = gain * 0.006 * G.VALUES["premium"] * sold * horizon
                    cands.append((ev - move - wages - riot, legacy, s, q, price, meals, sold, in_riot, ev))
        if not cands:
            return None, {"fallback": None}
        # a wreck costs the next min(2, days left) nights: either a flat $/night or ("auto") the profit we
        # expect tonight at our best spot, which is what those nights would have made
        per_night = P["wreck_value"] if P["wreck_value"] != "auto" else max(0.0, max(c[0] for c in cands))
        wreck = per_night * min(2, days_left)
        if any(wreck_w.values()):
            cands = [(c[0] - wreck_w[c[2]] * wreck,) + c[1:] for c in cands]
        # Reputation's future value, from the same day model: tonight's sale moves our rep at the spot by
        # REP_DELTA[q]; that is worth (profit at the new rep - profit at the old rep) on each of the next
        # `rep_horizon` nights we come back (same field, same order). Leaving our last spot lets its rep drift
        # 3% toward 50, valued the same way. Exact values for the strongest candidates only (time).
        swing = None
        mine = getattr(self, "mine", None)
        if P["price_swing"] is not None and home is not None and mine and mine.get("spot") == home:
            swing = (mine["price"], P["price_swing"])
        future = {}
        if P["rep_value"] == "vfit":
            # VFIT (whole-market rep-shock fit): with D nights left, a rep point at spot s is worth
            # D/30 * (g * share(lead) + lin), lead = our rep - the best rival's rep there. Selling at s moves
            # s by REP_DELTA[q] and spares it the 3% decay toward 50 every other spot takes (score up to a constant).
            best_riv = {}
            for t in state.get("trucks") or []:
                if isinstance(t, dict) and t.get("id") != you.get("id"):
                    for sp, r in (t.get("reputation") or {}).items():
                        best_riv[sp] = max(best_riv.get(sp, 0.0), G.num(r, 50.0))
            d = days_left / 30.0
            for c in cands:
                if c[6] <= 0.5:
                    continue
                r0 = ctx[c[2]]["rep"]
                lead = max(-60.0, min(60.0, (r0 - best_riv.get(c[2], 50.0)) * 0.096 / 1.5))
                per_pt = d * (P["vfit_g"] / (1.0 + math.exp(-lead)) + P["vfit_lin"])
                gain = max(-r0, min(G.REP_DELTA[c[3]], 100.0 - r0)) + 0.03 * (r0 - 50.0)
                future[(c[2], c[3], c[4])] = gain * per_pt * P["rep_scale"]
            drift = 0.0
        elif P["rep_value"] == "model" and horizon > 0:
            top = sorted(cands, key=lambda c: -(c[0] + c[1]))[:P["rep_top"]]
            best_sq = {}
            for c in cands:
                if swing is not None and c[2] == home and abs(c[4] - swing[0]) > swing[1] + 1e-9:
                    continue
                if (c[2], c[3]) not in best_sq or c[0] + c[1] > best_sq[(c[2], c[3])][0] + best_sq[(c[2], c[3])][1]:
                    best_sq[(c[2], c[3])] = c
            for c in top + list(best_sq.values()):
                key = (c[2], c[3], c[4])
                if key in future:
                    continue
                r1 = max(0.0, min(100.0, ctx[c[2]]["rep"] + G.REP_DELTA[c[3]]))
                future[key] = (evaluate(c[2], c[3], c[4], r1)[0] - c[8]) * horizon * P["rep_scale"] if c[6] > 0.5 else 0.0
            drift = 0.0
            if last in ctx and ctx[last]["rep"] > 50.5:
                hb = max((c for c in cands if c[2] == last), key=lambda c: c[0], default=None)
                if hb is not None:
                    r0 = ctx[last]["rep"]
                    drift = (hb[8] - evaluate(last, hb[3], hb[4], r0 - 0.03 * (r0 - 50.0))[0]) * horizon * P["rep_scale"]
        best, best_score, best_info = None, -1e18, None
        sit_fallback = safe_best = None
        for profit, legacy, s, q, price, meals, sold, in_riot, _ in cands:
            if P["rep_value"] == "vfit":
                rep_bonus = future.get((s, q, price), 0.0)
            elif P["rep_value"] == "model":
                key = (s, q, price)
                if horizon <= 0:          # last night: reputation is worth nothing any more
                    rep_bonus = 0.0
                elif key not in future:
                    continue
                else:
                    rep_bonus = future[key] - (drift if s != last else 0.0)
            else:
                rep_bonus = legacy - (leave if home is not None and s != home else 0.0)
            if home is not None and s != home:
                rep_bonus -= P["move_hurdle"]
            elif swing is not None and abs(price - swing[0]) > swing[1] + 1e-9:
                continue
            score = profit + rep_bonus
            cand = {"spot": s, "price": round(price, 2), "quality": q, "meals": int(meals)}
            info = (score, profit, sold, rep_bonus)
            if score > best_score:
                best, best_score, best_info = cand, score, info
            if not in_riot and (safe_best is None or score > safe_best[1][0]):
                safe_best = (cand, info)
            if sit_fallback is None or (not in_riot, profit) > sit_fallback[0]:
                sit_fallback = ((not in_riot, profit), cand)
        if best is None:
            return None, {"fallback": sit_fallback[1] if sit_fallback else None}
        mode = P["riot_sitout"]      # True: sit out; "safe": open at the best spot outside the radius; False: EV only
        if mode and riots and (safe_best is None or safe_best[0] is not best):
            if mode == "safe" and safe_best is not None:
                best, best_info = safe_best
            else:
                # the best spot is inside a warned riot: sit out; after a sit-out, the best safe spot
                return None, {"best": best, "fallback": safe_best[0] if safe_best else sit_fallback[1], "riot": True,
                              "why": "riot"}
        if best_info[1] < 0 and P["sit_negative"]:      # expected profit after wages is negative: sit out
            return None, {"best": best, "fallback": safe_best[0] if safe_best else best, "why": "negative"}
        return best, {"score": best_info}

    def _prices(self, tr, cap):
        P = self.p
        ps = set()
        x = P["price_lo"]
        while x <= P["price_hi"] + 1e-9:
            ps.add(round(x, 2))
            x += P["price_step"]
        for t in tr:   # just under / at each rival's price
            for d in (-0.01, -0.26):
                v = round(t["price"] + d, 2)
                if P["price_lo"] <= v <= P["price_hi"]:
                    ps.add(v)
        if cap is not None:
            ps = {min(p, cap) for p in ps}
        return sorted(ps)
