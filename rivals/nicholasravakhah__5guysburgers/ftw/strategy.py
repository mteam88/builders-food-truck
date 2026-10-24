"""Decision making: scenario sampling + joint search over spot, quality, price, meals.

Brain.observe(state) ingests each day's results exactly once per (mode, market, day).
Brain.decide(state) returns a legal order within a deadline, falling back to a
simple heuristic if anything goes wrong.
"""
import hashlib
import math
import random
import time

from . import engine
from .model import CustomerModel, default_expected, event_key
from .opponents import Rival, day_context
from .rules import QI, REP_DELTA, Rules, rep_factor

DEFAULT_CFG = {
    "scenarios": 150,
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
    "only_quality": None,     # e.g. "premium" to consider one tier only
    "fixed_price": None,      # e.g. 9.24: only this price; the optimizer still picks meals
    "live_overrides": {},     # manual settings applied in mode == "live" only
    "meals_mult": 1.0,        # scale the optimal meal count (e.g. 1.15 after repeated sellouts)
    # optional server rules (state.features); exact rules for fair choice / walk over are not published
    "salvage_frac": 0.25,     # salvage: unsold meals refund 25% of food cost (documented)
    "fair_delta": 1.0,        # fair choice: rivals within this value of ours share customers (guess)
    "fair_temp": 0.5,         # fair choice: share weight exp(value / temp) (guess)
    "walk_frac": 0.15,        # walk over: share of hungry customers who walk (live day 16: 10-25%)
    "premium_guard": 0.5,     # skip cheap/standard at a spot where a better-value premium rival shows up in
                              # >= this share of scenarios (live day 21: bot wanted standard vs Frontrun/Wagyu premium)
    "gameday_rule": True,     # game day: standard, max meals, biggest spot no rival owns (live day 21: +$763)
    "gameday_price": 9.49,
    "stronghold_rep": 70.0,   # a rival with this reputation at a spot owns it
    "latenight_rule": True,   # late-night rush at our home spot: premium, max meals (live day 23: +$286)
    "latenight_price": 9.25,
    "home_rep": 80.0,         # our reputation at a spot that makes it "home"
    "home_day_profit": 250.0, # typical daily profit at home, used to value crowd laws when voting
    "home_meals": 60.0,       # typical daily meals at home, for valuing per-meal taxes
    "gameday_profit": 600.0,  # typical game-day profit at the Stadium, for valuing game-day laws
    "gameday_every": 8.0,     # roughly one game day every N days
    "riot_cost_day": 40.0,    # $/day we'd pay to remove riot risk (wrecked truck = lost days)
    "riot_guard": True,       # on a possible riot day keep away from the event spot and its neighbours
    "riot_radius": 650.0,     # metres around the Stadium treated as the riot zone (Thompson 487 m, Oval 590 m)
    "staff_target": 3,        # crew size: 3 x 50 = 150 meals >= our biggest order (133)
    "staff_wage": 9.0,        # rivals need >= $12 to poach (lure needs +$3); auto-raised if we lose staff
    "coupon_rule": True,      # while coupons are live: $20 coupon on the truck we are racing each day
    "coupon_min_lead": 0.0,   # only target a truck at least this far ahead of us in cash
    "shared_meal_frac": 1.0,  # <1 cooks less than the optimum when rivals are expected (Arjun's idea; off)
}

STYLES = {
    "max": {},
    # camp a home spot and build reputation there (best on held-out local seeds)
    "camp": {"rep_return": 0.9, "rep_lambda": 2.0, "near_tol_abs": 0.5, "meals_mult": 1.2},  # 1.2: won 3/3 paired backtests (+$840/market)
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
        food = choose_food(state, order.get("spot"), self.my_id)
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
                at[spot].append(engine.Truck(r.tid, q, p, r.rep.get(spot, 50.0), m, rules.values))
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
                per_spot[s] = engine.SpotScenario(lim, mq, at[s])
            scen.append(per_spot)
        return scen, rival_pred

    def _optimize(self, state, rules, deadline):
        cfg = self.cfg
        if state.get("mode") == "live" and cfg.get("live_overrides"):
            # manual live-game settings (spot/quality/price locks) never reach the final or practice
            cfg = dict(cfg, **cfg["live_overrides"])
        you = state.get("you") if isinstance(state.get("you"), dict) else {}
        my_rep = you.get("reputation") if isinstance(you.get("reputation"), dict) else {}
        gd = self._gameday_order(state, rules, cfg) if cfg.get("gameday_rule") else None
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
        move_cost = _move_costs(state, rules)

        def evaluate(spot, q, price):
            nonlocal evals
            rep = _f(my_rep.get(spot), 50.0)
            value = rules.values[q] * rep_factor(rep) - price
            if guard is not None and QI[q] < QI["premium"]:
                hit = sum(1 for sc in scen if any(t.qi == QI["premium"] and t.value >= value
                                                  for t in sc[spot].rivals))
                if hit >= guard * len(scen):
                    return None
            cap = rules.max_meals(q)
            qi = QI[q]
            ds = [sc[spot].demand(qi, price, value, cap, fair) for sc in scen]
            evals += 1
            cost = rules.costs[q]
            n, prof, sold = engine.best_quantity(ds, price, cost, cap, salvage)

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
                prof -= move_cost.get(spot, 0.0)  # moving costs: $2 + $0.01/m to leave the last spot sold at
            total = prof + bonus
            key = (spot, q)
            cur = best_by_combo.get(key)
            if cur is None or total > cur[0]:
                best_by_combo[key] = (total, {"spot": spot, "quality": q, "price": round(price, 2), "meals": n},
                                      prof, sold)
            return total

        # coarse pass
        avoid = set(cfg.get("avoid_spots") or [])
        if cfg.get("riot_guard"):
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
            pick = ranked[0]
        debug = {"evals": evals, "scenarios": len(scen), "exp_total": round(pick[0], 2),
                 "exp_profit": round(pick[2], 2), "exp_sold": round(pick[3], 1), "near": len(near),
                 "top": [(v[1]["spot"], v[1]["quality"], v[1]["price"], v[1]["meals"], round(v[0], 1))
                         for v in ranked[:5]],
                 "rivals": [(r.name or r.tid, round(pa, 2), max(sd, key=sd.get)) for r, pa, sd in rival_pred]}
        return dict(pick[1]), debug

    def _gameday_order(self, state, rules, cfg):
        """Game day: crowds outgrow every truck. Take the biggest spot that no rival owns
        (reputation >= stronghold_rep) and isn't ours already worth more, with the most standard meals."""
        fc = state.get("forecast") if isinstance(state.get("forecast"), dict) else {}
        keys = {event_key(e) for e in (fc.get("events") or []) if isinstance(e, dict)}
        if "latenight" in keys and cfg.get("latenight_rule") and "premium" in rules.qualities:
            mine = (state.get("you") or {}).get("reputation") or {}
            home = max(rules.spots, key=lambda s: _f(mine.get(s), 50.0))
            if home in ("highst", "northres") and _f(mine.get(home), 50.0) >= cfg["home_rep"]:
                return {"spot": home, "quality": "premium", "price": float(cfg["latenight_price"]),
                        "meals": rules.max_meals("premium")}
        if "gameday" not in keys:
            return None
        if riot_risk(state) and cfg.get("riot_guard"):
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


def _laws_in_force(state):
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
    """Meat or veggie (only while the food rule is live): serve the diet whose only-eaters we can
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


def choose_vote(state, cfg=None):
    """Pick the election candidate whose laws help us most: crowd multipliers at our home spot
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
            if gc:  # game-day crowd law, valued at our game-day spot (the Stadium) per day
                mult = _f(gc.get("stadium", gc.get("other", 1.0)), 1.0)
                score += float(cfg["gameday_profit"]) / float(cfg["gameday_every"]) * (mult - 1.0)
            pm = pr.get("per_meal") if isinstance(pr.get("per_meal"), dict) else {}
            if pm:  # per-meal tax (+) or subsidy (-) on a quality: valued at the quality we usually sell
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
    """{spot: cost to park there tomorrow} while moving costs are live: base + per_metre * distance
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
    if isinstance(mc, dict) and isinstance(mc.get(last), dict):  # exact table: move_costs[from][to]
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


__all__ = ["Brain", "DEFAULT_CFG", "event_key", "choose_vote", "riot_risk", "choose_food"]
