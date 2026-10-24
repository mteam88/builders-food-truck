"""Port of jtanSE/wagyuexpress bot.py (Wagyu Express / Oriental Dishes, t_96a96a),
SHA bf998d3625135056e52c45dbc905c621f7959bf5 (2026-10-23T21:36:40-04:00).
Decision logic last changed in 691a2cd (21:34, price discount) and ad96588 (21:23, risk .5).
Previous ports: policy_967d6fc7.py, policy_4799e1b4.py.

Decision code (bot.py lines 26-600: constants, customer model, fluid demand, newsvendor,
selection_limits/own_budget/discounted_price/fallback_order/inventory_diagnostics, Policy)
copied verbatim. Live launch (PRICE_ADJUSTMENT.md, SELF_IMPROVEMENT.md recovery command):
  bot.py --live --allow-orders --inventory-mode profit --price-discount 0.40   (cecf1c90, 21:58 ET; live from day 7)
-> Policy(memory=..., risk=0.0, price_discount=.40); event rule defaults unchanged. Before day 7: risk .5 (policy_28fef421.py)
(avoid_event_hotspots=True, event_price_cap=9.99); regular_rank/regular_price_cap/
own_budget_limit stay None (experiments only). make() defaults to those live values.
opponent_model.py is an offline advisory CLI; bot.py does not import it. responsive_policy.py
is still unused live.

Exceptions in decide fall back to bot.py's policy_fallback (fallback_order with the policy's
event/regular/budget/discount settings). Randomness: deterministic given (your id,
mode/market, order_for_day); `seed` unused; only the 0.60 s think budget is wall-clock.
"""
import hashlib
import math
import random
import statistics
import time
from collections import Counter

SPOTS = ("oval", "thompson", "union", "stadium", "highst", "northres")
QUALITIES = ("cheap", "standard", "premium")
COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
VALUES = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
DELTAS = {"cheap": -3, "standard": 2, "premium": 5}
# Priors inferred from the docs' approximate descriptions. Learned from results.
PROFILES = {
    "oval": (13.0, 2.6, (.35, .45, .20)),
    "thompson": (12.0, 2.3, (.45, .42, .13)),
    "union": (13.0, 2.5, (.30, .52, .18)),
    "stadium": (13.0, 2.5, (.30, .45, .25)),
    "highst": (12.0, 3.4, (.50, .36, .14)),
    "northres": (11.0, 2.2, (.50, .40, .10)),
}
Z = tuple(statistics.NormalDist().inv_cdf((i + .5) / 11) for i in range(11))


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big")


def factor(reputation):
    return .7 + .006 * max(0.0, min(100.0, float(reputation)))


def rules_for(state):
    rules = state.get("rules") or {}
    return {
        "spots": list(rules.get("spots") or SPOTS),
        "budget": float(rules.get("budget", 500)),
        "costs": dict(rules.get("costs") or COSTS),
        "values": dict(rules.get("values") or VALUES),
        "price_min": float(rules.get("price_min", 1)),
        "price_max": float(rules.get("price_max", 30)),
    }


def valid_order(order, state):
    r = rules_for(state)
    try:
        q, n, p = order["quality"], order["meals"], order["price"]
        return (set(order) == {"spot", "quality", "price", "meals"}
                and order["spot"] in r["spots"] and q in QUALITIES
                and isinstance(n, int) and not isinstance(n, bool) and n >= 0
                and isinstance(p, (int, float)) and not isinstance(p, bool)
                and math.isfinite(p) and r["price_min"] <= p <= r["price_max"]
                and n * float(r["costs"][q]) <= r["budget"] + 1e-8)
    except (KeyError, TypeError, ValueError):
        return False


def selection_limits(state, rules, avoid_event_hotspots=True, event_price_cap=9.99,
                     regular_rank=None, regular_price_cap=None):
    forecast = state.get("forecast") or {}
    has_event = any(
        event.get("spot") in rules["spots"] for event in forecast.get("events") or [])
    event_day = avoid_event_hotspots and has_event
    allowed_spots, ceiling = list(rules["spots"]), rules["price_max"]
    if event_day:
        expected = forecast.get("expected_customers") or {}
        ranked = sorted(rules["spots"], key=lambda spot: -float(expected.get(spot, 0)))
        allowed_spots = ranked[1:3] or ranked
        ceiling = max(rules["price_min"], min(ceiling, float(event_price_cap)))
    elif not has_event:
        if regular_rank is not None:
            if isinstance(regular_rank, bool) or not isinstance(regular_rank, int) or regular_rank < 1:
                raise ValueError("regular_rank must be a positive integer or None")
            expected = forecast.get("expected_customers") or {}
            ranked = sorted(rules["spots"], key=lambda spot: -float(expected.get(spot, 0)))
            allowed_spots = [ranked[min(regular_rank - 1, len(ranked) - 1)]]
        if regular_price_cap is not None:
            price_cap = float(regular_price_cap)
            if not math.isfinite(price_cap):
                raise ValueError("regular_price_cap must be finite or None")
            ceiling = max(rules["price_min"], min(ceiling, price_cap))
    return bool(event_day), allowed_spots, ceiling


def own_budget(rules, own_budget_limit=None):
    if own_budget_limit is None:
        return rules["budget"]
    limit = float(own_budget_limit)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("own_budget_limit must be finite and nonnegative")
    return min(rules["budget"], limit)


def discounted_price(price, rules, discount):
    discount = float(discount)
    if not math.isfinite(discount) or discount < 0:
        raise ValueError("price discount must be finite and nonnegative")
    return round(max(rules["price_min"], min(rules["price_max"], price - discount)), 2)


def fallback_order(state, avoid_event_hotspots=True, event_price_cap=9.99,
                   regular_rank=None, regular_price_cap=None, own_budget_limit=None, price_discount=0.0):
    """Cheap computation for malformed models or an unexpectedly short deadline."""
    r = rules_for(state)
    _, allowed_spots, ceiling = selection_limits(
        state, r, avoid_event_hotspots, event_price_cap, regular_rank, regular_price_cap)
    crowds = (state.get("forecast") or {}).get("expected_customers") or {}
    spot = max(allowed_spots, key=lambda s: float(crowds.get(s, 0)))
    rep = (state.get("you") or {}).get("reputation") or {}
    price = min(10.49, r["values"]["standard"] * factor(rep.get(spot, 50)) - .01)
    price = round(max(r["price_min"], min(ceiling, price)), 2)
    price = discounted_price(price, r, price_discount)
    cap = int((own_budget(r, own_budget_limit) + 1e-9) / r["costs"]["standard"])
    meals = min(cap, max(0, int(float(crowds.get(spot, 100)) * .55)))
    if price <= r["costs"]["standard"]:
        meals = 0
    return {"spot": spot, "quality": "standard", "price": price, "meals": meals}


def inventory_diagnostics(order, demands, rules):
    """Selected stock exposure; modeled sales are capped at our own inventory.

    A stockout only establishes a lower bound on actual demand. These scenario
    estimates do not turn observed sold/meals into an uncensored demand sample.
    """
    quantity = order["meals"]
    food_cost = quantity * float(rules["costs"][order["quality"]])
    result = {"expected_sales": None, "expected_spoiled": None,
              "loss_probability": None, "worst_scenario_profit": None,
              "max_food_cost": round(food_cost, 2)}
    if quantity == 0:
        result.update(expected_sales=0.0, expected_spoiled=0.0,
                      loss_probability=0.0, worst_scenario_profit=0.0)
    elif demands:
        sales = [min(quantity, max(0.0, demand)) for demand in demands]
        expected = sum(sales) / len(sales)
        profits = [order["price"] * sold - food_cost for sold in sales]
        result.update(expected_sales=round(expected, 2),
                      expected_spoiled=round(quantity - expected, 2),
                      loss_probability=round(sum(profit < -1e-9 for profit in profits) / len(profits), 4),
                      worst_scenario_profit=round(min(profits), 2))
    return result


def customer_segments(spot, forecast, rng, variant, calibration=0.0):
    """A small probability distribution; not a claim about the server generator."""
    mean, spread, tastes = PROFILES.get(spot, PROFILES["union"])
    tastes = list(tastes)
    for event in forecast.get("events") or []:
        name = str(event.get("name", "")).lower()
        if "game" in name:
            mean += .7
            tastes[0] += .10
            tastes[2] -= .06
        if "midterm" in name and spot == "thompson":
            tastes[2] += .20
            tastes[0] -= .15
        if "concert" in name and spot == "union":
            mean += 1.0
        if "rush" in name and spot in ("highst", "northres"):
            tastes[0] += .16
            tastes[2] -= .07
    mean += calibration + rng.gauss(0, .45)
    spread *= (.85, 1.15, 1.0)[variant % 3]
    tastes = [max(.025, p + rng.uniform(-.035, .035)) for p in tastes]
    total = sum(tastes)
    result = []
    for i, z in enumerate(Z):
        # Alternate a light-tailed and a heavy-tailed budget prior.
        quantile = z if variant % 3 != 1 else math.log((i + .5) / (11 - i - .5)) / 1.7
        budget = max(1.0, mean + spread * quantile)
        for minimum, weight in enumerate(tastes):
            result.append((budget, minimum, weight / total / len(Z)))
    return result


def fluid_demand(segments, crowd, rivals, quality, price, value):
    """Potential sales with unlimited own stock and finite rival stocks.

    Simulate continuous arrivals, jumping directly to the next rival stockout.
    With our inventory n, own sales are min(n, this potential demand). Before
    our stockout the unlimited and limited systems have identical trajectories.
    """
    if value < -1e-9 or crowd <= 0:
        return 0.0
    eligible = []
    for budget, minimum, weight in segments:
        if quality < minimum or price > budget:
            own = False
        else:
            own = True
        prefs = []
        for j, rival in enumerate(rivals):
            # rival = (quality index, price, value, meals)
            if rival[0] >= minimum and rival[1] <= budget and rival[2] >= -1e-9:
                prefs.append((rival[2], -rival[1], j))
        if own:
            # Equal score AND equal price has no documented tiebreaker.
            # The final tuple field favors a rival only after comparing price.
            prefs.append((value, -price, -1))
        if prefs:
            prefs.sort(reverse=True)
            eligible.append((weight * crowd, tuple(x[2] for x in prefs)))
    remaining = [max(0.0, float(x[3])) for x in rivals]
    active = {i for i, n in enumerate(remaining) if n > 1e-8}
    elapsed, demand = 0.0, 0.0
    for _ in range(len(rivals) + 1):
        rates = [0.0] * len(rivals)
        own_rate = 0.0
        for count, prefs in eligible:
            for j in prefs:
                if j == -1:
                    own_rate += count
                    break
                if j in active:
                    rates[j] += count
                    break
        dt = 1.0 - elapsed
        for j in active:
            if rates[j] > 1e-9:
                dt = min(dt, remaining[j] / rates[j])
        demand += own_rate * dt
        elapsed += dt
        if elapsed >= 1.0 - 1e-9:
            break
        for j in tuple(active):
            remaining[j] -= rates[j] * dt
            if remaining[j] < 1e-7:
                active.remove(j)
    return demand


def inventory_choice(demands, price, cost, cap, rep_bonus, risk):
    """Optimize a discrete newsvendor objective over all demand breakpoints."""
    choices = {0, 1, cap}
    for d in demands:
        choices.update((min(cap, max(0, int(d))), min(cap, max(0, int(d) + 1))))
    best = (-float("inf"), 0, 0.0, 0.0)
    for n in choices:
        if n < 0 or n > cap:
            continue
        profits = [price * min(n, d) - cost * n for d in demands]
        mean = sum(profits) / len(profits)
        sd = math.sqrt(sum((p - mean) ** 2 for p in profits) / len(profits))
        sale_probability = sum(d >= 1 for d in demands) / len(demands) if n else 0.0
        score = mean - risk * sd + rep_bonus * sale_probability
        candidate = (score, n, mean, sd)
        if candidate[0] > best[0]:
            best = candidate
    return best


class Policy:
    def __init__(self, scenarios=27, think_seconds=.60, risk=.5, reputation_weight=.45, memory=None, demand_scale=1.0,
                 avoid_event_hotspots=True, event_price_cap=9.99,
                 regular_rank=None, regular_price_cap=None, own_budget_limit=None, price_discount=0.0):
        self.scenarios = max(3, int(scenarios))
        self.think_seconds = think_seconds
        self.risk = risk
        self.reputation_weight = reputation_weight
        self.demand_scale = demand_scale
        self.avoid_event_hotspots = avoid_event_hotspots
        self.event_price_cap = float(event_price_cap)
        self.price_discount = float(price_discount)
        discounted_price(10, {"price_min": 1, "price_max": 30}, self.price_discount)
        self.regular_rank = regular_rank
        self.regular_price_cap = regular_price_cap
        self.own_budget_limit = own_budget_limit
        self.memory = memory or {"schema": 1, "opponents": {}, "calibration": {}}
        self.context = self.memory.get("context")
        self.seen_day = int(self.memory.get("seen_day", 0))
        self.explanation = {}

    def observe(self, state):
        context = [state.get("mode", "offline"), state.get("market", 1)]
        opponents = self.memory.setdefault("opponents", {})
        spots = rules_for(state)["spots"]
        if self.context != context or int(state.get("day", 0)) < self.seen_day:
            for rival in opponents.values():
                rival["rep"] = {s: 50.0 for s in spots}
                rival.pop("last", None)
                rival["opportunities"] = 0
                rival["played"] = 0
            self.context, self.seen_day = context, 0
            for field in ("previous_forecast", "current_forecast", "forecast_day"):
                self.memory.pop(field, None)
        mine = (state.get("you") or {}).get("id")
        for truck in state.get("trucks") or []:
            tid = truck.get("id")
            if tid and tid != mine:
                opponents.setdefault(tid, {"actions": [], "rep": {s: 50.0 for s in spots}})
        last = state.get("last_day") or {}
        day = int(last.get("day", 0) or 0)
        if day > self.seen_day:
            observed = {}
            for spot, detail in (last.get("spots") or {}).items():
                for action in detail.get("trucks") or []:
                    tid = action.get("id")
                    if tid and tid != mine:
                        observed[tid] = dict(action, spot=spot, day=day)
            missing_days = max(0, day - self.seen_day - 1)
            for tid, rival in opponents.items():
                reps = rival.setdefault("rep", {s: 50.0 for s in spots})
                action = observed.get(tid)
                # Count observed rounds, not unobserved gaps. A roster entry is
                # not proof the truck is currently running and placing orders.
                rival["opportunities"] = rival.get("opportunities", 0) + 1
                if action and float(action.get("meals") or 0) > 0:
                    rival["played"] = rival.get("played", 0) + 1
                for spot in spots:
                    reps[spot] = 50 + (reps.get(spot, 50) - 50) * .97 ** missing_days
                    if not action or spot != action["spot"]:
                        reps[spot] = 50 + (reps[spot] - 50) * .97
                if action:
                    if float(action.get("sold", 0)) > 0:
                        q = action.get("quality", "standard")
                        reps[action["spot"]] = max(0, min(100, reps[action["spot"]] + DELTAS.get(q, 0)))
                    compact = {k: action.get(k) for k in ("spot", "quality", "price", "meals", "sold", "day")}
                    rival.setdefault("actions", []).append(compact)
                    rival["actions"] = rival["actions"][-48:]
                    rival["last"] = compact
            self._calibrate(last, state)
            self.seen_day = day
        # Exact public average corrects drift after missing observations, without
        # pretending that a rival's average is its reputation at every spot.
        for truck in state.get("trucks") or []:
            rival = opponents.get(truck.get("id"))
            if rival and isinstance(truck.get("reputation"), dict):
                # The live API supplies this richer field even though the docs'
                # abbreviated example only shows the public average.
                rival["rep"] = {s: max(0, min(100, float(truck["reputation"].get(s, rival["rep"].get(s, 50)))))
                                for s in spots}
            elif rival and "reputation_avg" in truck:
                reps = rival["rep"]
                shift = float(truck["reputation_avg"]) - sum(reps.get(s, 50) for s in spots) / len(spots)
                for s in spots:
                    reps[s] = max(0, min(100, reps.get(s, 50) + shift))
        self.memory.update(context=self.context, seen_day=self.seen_day)
        forecast_day = (state.get("forecast") or {}).get("day", state.get("order_for_day"))
        if forecast_day != self.memory.get("forecast_day"):
            self.memory["previous_forecast"] = self.memory.get("current_forecast") or {}
            self.memory["current_forecast"] = state.get("forecast") or {}
            self.memory["forecast_day"] = forecast_day

    def _calibrate(self, last, state):
        # Infer a gentle budget correction only from nearly isolated offers.
        # Stockouts censor demand, so they are NOT treated as exact demand.
        for spot, detail in (last.get("spots") or {}).items():
            rows = detail.get("trucks") or []
            crowd = float(detail.get("customers", 0))
            if len(rows) != 1 or crowd < 30:
                continue
            row = rows[0]
            sold, meals = float(row.get("sold", 0)), float(row.get("meals", 0))
            q, price = row.get("quality"), float(row.get("price", 0))
            if q != "premium" or sold >= meals - 1 or price <= 0:
                continue
            tid = row.get("id")
            if tid == (state.get("you") or {}).get("id"):
                rep = ((state.get("you") or {}).get("reputation") or {}).get(spot, 50) - 5
            else:
                rep = self.memory.get("opponents", {}).get(tid, {}).get("rep", {}).get(spot, 50) - 5
            if price > VALUES["premium"] * factor(rep) - .2:
                continue
            mean, spread, _ = PROFILES.get(spot, PROFILES["union"])
            # For an isolated premium offer all minimum-quality groups qualify.
            share = max(.025, min(.975, sold / crowd))
            inferred = price - spread * statistics.NormalDist().inv_cdf(1 - share) - mean
            old = self.memory.setdefault("calibration", {}).get(spot, 0.0)
            self.memory["calibration"][spot] = max(-1.5, min(1.5, .9 * old + .1 * inferred))

    def _worlds(self, state, rules):
        forecast = state.get("forecast") or {}
        expected = forecast.get("expected_customers") or {}
        mine = (state.get("you") or {}).get("id", "offline")
        seed = stable_seed(mine, self.context, state.get("order_for_day"))
        rivals = [(t.get("id"), self.memory["opponents"].get(t.get("id"), {}))
                  for t in state.get("trucks") or [] if t.get("id") != mine]
        # Offline partial states may omit the roster. Do not invent competitors.
        result = {s: [] for s in rules["spots"]}
        totals = sum(max(0.0, float(expected.get(s, 0))) for s in rules["spots"])
        scenario_count = min(self.scenarios, max(9, 540 // max(1, len(rivals))))
        for k in range(scenario_count):
            rng = random.Random(seed + k * 104729)
            assignments = {s: [] for s in rules["spots"]}
            # Correlated attraction models account for several rivals following
            # the same obvious forecast/event, as well as quiet-market hunting.
            attraction_power = (.7, 1.2, 1.8)[k % 3]
            for tid, model in rivals:
                actions, last = model.get("actions", []), model.get("last")
                opportunities = model.get("opportunities", 0)
                if opportunities:
                    active_probability = ((2 + model.get("played", 0)) / (3 + opportunities))
                    if last and last.get("day") == self.seen_day and float(last.get("meals") or 0) > 0:
                        active_probability = 1.0
                    if active_probability < 1 and rng.random() > max(.1, active_probability):
                        continue
                reps = model.get("rep", {})
                weights = []
                freq = Counter(a.get("spot") for a in actions[-20:])
                for spot in rules["spots"]:
                    share = max(1.0, float(expected.get(spot, 0))) / max(1.0, totals)
                    prior = share ** attraction_power
                    habit = (freq[spot] + 1) / (len(actions[-20:]) + len(rules["spots"]))
                    rep_boost = math.exp((reps.get(spot, 50) - 50) / 55)
                    weights.append((.65 * prior + .35 * habit) * rep_boost)
                persistence = .64
                if last and float(last.get("meals") or 0) > 0:
                    ratio = float(last.get("sold") or 0) / float(last["meals"])
                    persistence = .78 if ratio > .93 else .30 if ratio < .55 else .55
                    recent = actions[-16:]
                    transitions = [(a, b) for a, b in zip(recent, recent[1:])
                                   if b.get("day") == a.get("day", -2) + 1]
                    if transitions:
                        repeats = sum(a.get("spot") == b.get("spot") for a, b in transitions)
                        empirical = (repeats + 2) / (len(transitions) + 4)
                        persistence = .35 * persistence + .65 * empirical
                    old_crowd = (self.memory.get("previous_forecast") or {}).get("expected_customers", {}).get(last["spot"])
                    if old_crowd and float(expected.get(last["spot"], 0)) < .55 * old_crowd:
                        persistence *= .6
                if last and last.get("spot") in assignments and rng.random() < persistence:
                    spot = last["spot"]
                else:
                    spot = rng.choices(rules["spots"], weights=weights)[0]
                quality_counts = Counter(a.get("quality") for a in actions[-20:])
                quality = rng.choices(QUALITIES, weights=[quality_counts[q] + w for q, w in zip(QUALITIES, (1, 4, 3))])[0]
                if last and rng.random() < .72:
                    quality = last.get("quality", "standard")
                qi = QUALITIES.index(quality)
                matching = [float(a["price"]) for a in actions[-12:] if a.get("quality") == quality and a.get("price")]
                price = statistics.median(matching) if matching else (6.8, 10.25, 12.25)[qi]
                if last and quality == last.get("quality"):
                    price = .65 * float(last["price"]) + .35 * price
                price_noise = max(.4, min(1.4, statistics.pstdev(matching))) if len(matching) > 2 else .55
                price += rng.gauss(0, price_noise if actions else 1.15)
                price = min(rules["price_max"], max(rules["price_min"], price))
                cap = int((rules["budget"] + 1e-9) / rules["costs"][quality])
                stock_fraction = 1.0
                if last and last.get("meals") and last.get("quality") in rules["costs"]:
                    old_cap = int(rules["budget"] / rules["costs"][last["quality"]])
                    stock_fraction = min(1.0, max(.2, float(last["meals"]) / max(1, old_cap)))
                meals = max(1, int(cap * stock_fraction))
                value = rules["values"][quality] * factor(reps.get(spot, 50)) - price
                assignments[spot].append((qi, price, value, meals))
            for spot in rules["spots"]:
                crowd = max(0.0, float(expected.get(spot, 0)))
                # Separate crowd noise and persistent taste/budget uncertainty.
                crowd *= (.72, .88, 1.0, 1.12, 1.28, .8, .95, 1.08, 1.20)[k % 9]
                calibration = self.memory.get("calibration", {}).get(spot, 0.0)
                segments = customer_segments(spot, forecast, rng, k, calibration)
                result[spot].append((segments, crowd, assignments[spot]))
        return result

    def decide(self, state):
        started = time.perf_counter()
        self.observe(state)
        r = rules_for(state)
        worlds = self._worlds(state, r)
        event_day, allowed_spots, price_ceiling = selection_limits(
            state, r, self.avoid_event_hotspots, self.event_price_cap,
            self.regular_rank, self.regular_price_cap)
        rep = (state.get("you") or {}).get("reputation") or {}
        remaining_days = max(0, int(state.get("days", 30)) - int(state.get("order_for_day", 1)))
        horizon = (1 - .87 ** min(remaining_days, 12)) / .13
        leaders, evaluated, cache = [], 0, set()

        def consider(spot, quality, price):
            nonlocal evaluated
            if spot not in allowed_spots:
                return
            price = round(max(r["price_min"], min(price_ceiling, price)), 2)
            key = (spot, quality, price)
            if key in cache:
                return
            cache.add(key)
            cost = float(r["costs"][quality])
            value = float(r["values"][quality]) * factor(rep.get(spot, 50)) - price
            if value < -1e-9 or price <= cost:
                return
            cap = int((own_budget(r, self.own_budget_limit) + 1e-9) / cost)
            demands = [self.demand_scale * fluid_demand(seg, n, others, QUALITIES.index(quality), price, value)
                       for seg, n, others in worlds[spot]]
            typical_sales = min(cap, sum(demands) / len(demands))
            delta = max(0, min(100, float(rep.get(spot, 50)) + DELTAS[quality])) - float(rep.get(spot, 50))
            # Approximate future value of reputation. Taper to zero at the end;
            # cap at 100; never confuse average reputation with local reputation.
            bonus = (self.reputation_weight * horizon * .55 * typical_sales
                     * float(r["values"][quality]) * .006 * delta)
            score, meals, mean, sd = inventory_choice(demands, price, cost, cap, bonus, self.risk)
            order = {"spot": spot, "quality": quality, "price": price, "meals": meals}
            row = (score, mean, sd, order, demands, bonus)
            leaders.append(row)
            evaluated += 1

        # Complete the same coarse grid across every spot before refinement.
        # This avoids a deadline favoring whichever spot was enumerated first.
        for price in (9.49, 11.49, 12.99, 7.49, 14.49, 10.49, 5.99, 15.99, 17.49):
            for quality in ("standard", "premium", "cheap"):
                for spot in r["spots"]:
                    consider(spot, quality, price)
        for spot in r["spots"]:
            for quality in QUALITIES:
                ceiling = r["values"][quality] * factor(rep.get(spot, 50))
                consider(spot, quality, math.floor((ceiling - .005) * 100) / 100)
        leaders.sort(key=lambda row: row[0], reverse=True)
        # Refine the best offer for every location/tier, not twelve nearly
        # identical prices at one location. This prevents coarse-grid lock-in.
        diverse = {}
        for row in leaders:
            diverse.setdefault((row[3]["spot"], row[3]["quality"]), row)
        seeds = list(diverse.values()) + leaders[:4]
        for row in seeds:
            order = row[3]
            for shift in (-.75, -.5, -.25, -.1, .1, .25, .5, .75):
                if time.perf_counter() - started > self.think_seconds:
                    break
                consider(order["spot"], order["quality"], order["price"] + shift)
            # Cent-level best-response thresholds against sampled rival offers.
            own_base = r["values"][order["quality"]] * factor(rep.get(order["spot"], 50))
            thresholds = {round(own_base - rival[2] - .01, 2)
                          for _, _, rivals in worlds[order["spot"]] for rival in rivals}
            for threshold in sorted(thresholds, key=lambda p: abs(p - order["price"]))[:8]:
                if time.perf_counter() - started > self.think_seconds:
                    break
                consider(order["spot"], order["quality"], threshold)
        if not leaders:
            order = fallback_order(state, self.avoid_event_hotspots, self.event_price_cap,
                                   self.regular_rank, self.regular_price_cap, self.own_budget_limit, self.price_discount)
            self.explanation = {"fallback": True, "reason": "no profitable modeled offer",
                                **inventory_diagnostics(order, None, r)}
            return order
        leaders.sort(key=lambda row: row[0], reverse=True)
        best = leaders[0]
        order = best[3]
        assert valid_order(order, state), "optimizer produced an invalid order"
        self.explanation = {
            "order": order, "estimated_profit": round(best[1], 2),
            "scenario_profit_sd": round(best[2], 2), "reputation_bonus": round(best[5], 2),
            "potential_sales_by_scenario": [round(d, 1) for d in best[4]], "offers_evaluated": evaluated,
            "compute_ms": round((time.perf_counter() - started) * 1000, 2),
            "scenarios": len(worlds[order["spot"]]),
            "event_preference": {"active": bool(event_day), "allowed_spots": allowed_spots,
                                 "price_ceiling": price_ceiling},
            "regular_preference": {"rank": self.regular_rank, "price_cap": self.regular_price_cap,
                                   "active": not any(event.get("spot") in r["spots"]
                                                     for event in (state.get("forecast") or {}).get("events") or [])
                                   and (self.regular_rank is not None or self.regular_price_cap is not None)},
            **inventory_diagnostics(order, best[4], r),
            "inventory_note": "Scenario estimates; stockout sales are a lower bound on demand.",
            "alternatives": [{"order": x[3], "estimated_profit": round(x[1], 2)} for x in leaders[1:5]],
            "warning": "Estimates use approximate customer and rival models; not server predictions.",
        }
        if self.price_discount:
            # User-directed price adjustment happens once, after location/quality
            # selection. Recompute demand and stock at the actual selling price.
            base_order = dict(order)
            order = dict(order, price=discounted_price(order["price"], r, self.price_discount))
            spot, quality, price = order["spot"], order["quality"], order["price"]
            cost = float(r["costs"][quality])
            value = float(r["values"][quality]) * factor(rep.get(spot, 50)) - price
            demands = [self.demand_scale * fluid_demand(seg, n, others, QUALITIES.index(quality), price, value)
                       for seg, n, others in worlds[spot]]
            cap = int((own_budget(r, self.own_budget_limit) + 1e-9) / cost)
            typical_sales = min(cap, sum(demands) / len(demands))
            delta = max(0, min(100, float(rep.get(spot, 50)) + DELTAS[quality])) - float(rep.get(spot, 50))
            bonus = self.reputation_weight * horizon * .55 * typical_sales * float(r["values"][quality]) * .006 * delta
            _, meals, mean, sd = inventory_choice(demands, price, cost, cap, bonus, self.risk)
            if price <= cost:
                meals, mean, sd, bonus = 0, 0.0, 0.0, 0.0
            order["meals"] = meals
            self.explanation["alternatives_before_discount"] = self.explanation.pop("alternatives")
            self.explanation.update(order=order, base_order=base_order,
                                    requested_price_discount=self.price_discount,
                                    applied_price_discount=round(base_order["price"] - price, 2),
                                    estimated_profit=round(mean, 2), scenario_profit_sd=round(sd, 2),
                                    reputation_bonus=round(bonus, 2),
                                    potential_sales_by_scenario=[round(d, 1) for d in demands],
                                    compute_ms=round((time.perf_counter() - started) * 1000, 2),
                                    **inventory_diagnostics(order, demands, r))
        return order



# ===== f2ec9669 (03:50-03:53 ET) bot.py DecoyPolicy: gameday_stadium, price bands, comments stripped (verbatim except a seedable rng) =====
class FinalReached(Exception):
    pass


class DecoyPolicy:
    PERSONAS = (("highst_fortress", .15), ("highst_reclaim", .13),
                ("turf_raid", .25), ("cheap_blitz", .08),
                ("price_spike", .07), ("rotation", .07), ("empty_premium", .1),
                ("inverse_crowd", .1), ("chaos", .05))

    def __init__(self, memory=None, seed=None):
        self.memory = {"decoy": True, "history": list((memory or {}).get("history", []))}
        self.explanation = {}
        self.rng = random.Random(seed)  # live: random.Random() (os entropy); port: seedable

    def observe(self, state):
        if state.get("mode") == "final":
            raise FinalReached()

    RIOT_RADIUS = 600

    def _order(self, r, spot, quality, price, share, crowds):
        price = round(max(r["price_min"], min(r["price_max"], price)), 2)
        cap = min(int((r["budget"] + 1e-9) / float(r["costs"][quality])), self.capacity)
        meals = max(1, min(cap, int(float(crowds.get(spot, 100)) * share)))
        return {"spot": spot, "quality": quality, "price": price, "meals": meals}

    @staticmethod
    def _game_day(state):
        for event in (state.get("forecast") or {}).get("events") or []:
            text = (str(event.get("key", "")) + " " + str(event.get("name", ""))).lower()
            if event.get("opponent") or "game" in text or "champion" in text:
                return event
        return None

    @staticmethod
    def _office_policies(state):
        office = (state.get("politics") or {}).get("office") or {}
        return (office.get("holder") or {}).get("policies") or []

    def _crowds(self, state, game_day):
        crowds = {s: float(n) for s, n in
                  ((state.get("forecast") or {}).get("expected_customers") or {}).items()}
        for policy in self._office_policies(state):
            params = policy.get("params") or {}
            multipliers = params.get("crowd")
            if game_day and isinstance(params.get("game_crowd"), dict):
                game = params["game_crowd"]
                multipliers = {s: game.get(s, game.get("other", 1.0)) for s in crowds}
            elif policy.get("key") in ("crackdown", "tailgate") and not game_day:
                continue
            if not isinstance(multipliers, dict):
                continue
            for spot, m in multipliers.items():
                if spot in crowds:
                    crowds[spot] *= float(m)
        return crowds

    def _riot_spots(self, state, spots, game_day):
        if not game_day or not (state.get("features") or {}).get("riots", True):
            return set()
        text = (str(game_day.get("opponent", "")) + " " + str(game_day.get("name", ""))
                + " " + str(game_day.get("key", ""))).lower()
        if "michigan" not in text and "champion" not in text and "title" not in text:
            return set()
        if any(p.get("key") == "crackdown" or (p.get("params") or {}).get("riot") == 0
               for p in self._office_policies(state)):
            return set()
        info = state.get("spot_info") or {}
        center = info.get(game_day.get("spot") or "stadium") or info.get("stadium")
        if not center:
            return {s for s in spots if s in ("stadium", "thompson", "oval")}
        return {s for s in spots if s in info and math.hypot(
            float(info[s]["x"]) - float(center["x"]),
            float(info[s]["z"]) - float(center["z"])) <= self.RIOT_RADIUS}

    def decide(self, state):
        if state.get("mode") == "final":
            raise FinalReached()
        r = rules_for(state)
        staff = (state.get("rules") or {}).get("staff") or {}
        employees = ((state.get("last_day") or {}).get("you") or {}).get("employees") \
            or staff.get("default_staff", 4)
        self.capacity = int(employees) * int(staff.get("meals_per_employee", 50)) \
            if (state.get("features") or {}).get("staff") else 10 ** 6
        game_day = self._game_day(state)
        crowds = self._crowds(state, game_day)
        riot = self._riot_spots(state, r["spots"], game_day)
        spots = [s for s in r["spots"] if s not in riot] or r["spots"]
        me = (state.get("you") or {}).get("id")
        rivals = [t for t in state.get("trucks") or [] if t.get("id") != me]
        strongholds = [s for t in rivals for s, rep in (t.get("reputation") or {}).items()
                       if rep >= 90 and s in spots and s != "highst"]
        last_spots = (state.get("last_day") or {}).get("spots") or {}
        visited = {s for s, v in last_spots.items() if isinstance(v, dict) and v.get("trucks")}
        empty = [s for s in spots if s not in visited and s != "highst"]
        leader = max(rivals, key=lambda t: float(t.get("cash", 0)), default={}).get("id")
        leader_play = next(((s, t) for s, v in last_spots.items() if isinstance(v, dict)
                            for t in v.get("trucks") or [] if t.get("id") == leader), None)
        names, weights = zip(*self.PERSONAS)
        persona = self.rng.choices(names, weights)[0]
        if persona == "turf_raid" and not strongholds:
            persona = "highst_fortress"
        if persona == "mirror_leader" and not (leader_play and leader_play[0] in spots):
            persona = "rotation"
        if persona == "empty_premium" and not empty:
            persona = "chaos"
        if game_day and "stadium" in spots and self.rng.random() < .3:
            persona = "gameday_stadium"
        if persona == "gameday_stadium":
            order = self._order(r, "stadium", "standard", self.rng.uniform(8.0, 9.5), .4, crowds)
        elif persona == "highst_fortress" and "highst" in spots:
            order = self._order(r, "highst", "standard", self.rng.uniform(5.75, 7.25), .9, crowds)
        elif persona == "highst_reclaim" and "highst" in spots:
            holders = (last_spots.get("highst") or {}).get("trucks") or []
            rival_price = min((float(t.get("price") or 99) for t in holders
                               if t.get("id") != me), default=10.5)
            order = self._order(r, "highst", "premium", min(rival_price, 10.75) - .25, .7, crowds)
        elif persona == "turf_raid":
            order = self._order(r, self.rng.choice(strongholds), "standard",
                                self.rng.uniform(5.0, 6.5), .5, crowds)
        elif persona == "mirror_leader":
            spot, truck = leader_play
            quality = truck.get("quality") if truck.get("quality") in QUALITIES else "standard"
            price = float(truck.get("price") or r["values"][quality] * .7) - .25
            order = self._order(r, spot, quality, price, .6, crowds)
        elif persona == "price_spike" and "highst" in spots:
            order = self._order(r, "highst", "premium", self.rng.uniform(12.75, 14.75), .2, crowds)
        elif persona == "rotation":
            day = int(state.get("order_for_day") or 0)
            order = self._order(r, spots[day % len(spots)], "standard", 7.49, .5, crowds)
        elif persona == "cheap_blitz":
            ranked = sorted(spots, key=lambda s: -float(crowds.get(s, 0)))
            spot = ranked[1] if len(ranked) > 1 else ranked[0]
            order = self._order(r, spot, "cheap", self.rng.uniform(3.5, 4.5), .6, crowds)
        elif persona == "inverse_crowd":
            spot = min(spots, key=lambda s: float(crowds.get(s, 0)))
            order = self._order(r, spot, "cheap", self.rng.uniform(4.0, 5.5), .55, crowds)
        elif persona == "empty_premium":
            spot = max(empty, key=lambda s: float(crowds.get(s, 0)))
            order = self._order(r, spot, "premium", self.rng.uniform(13, 15.5), .35, crowds)
        else:
            persona = "chaos"
            quality = self.rng.choice(QUALITIES)
            order = self._order(r, self.rng.choice(spots), quality,
                                float(r["values"][quality]) * self.rng.uniform(.4, 1.1),
                                self.rng.uniform(.1, 1), crowds)
        last_spot = ((state.get("last_day") or {}).get("you") or {}).get("spot")
        move_cost = ((state.get("rules") or {}).get("move_costs") or {}).get(last_spot, {}).get(order["spot"], 0)
        self.explanation = {"decoy_persona": persona, "order": order, "game_day": bool(game_day),
                            "riot_avoided": sorted(riot), "capacity": self.capacity,
                            "move_cost": move_cost}
        self.memory["history"].append([state.get("order_for_day"), persona, order])
        return order


def make(seed=0, think_seconds=.60, risk=0.0, price_discount=.40, decoy=False, **kw):
    if decoy:  # e86b04c8 --decoy (own --memory file). Raises FinalReached in mode final (bot stops, no order).
        d = DecoyPolicy(seed=seed)

        def policy(state):
            d.observe(state)
            return d.decide(state)
        policy.p = d
        return policy
    # cecf1c90 (21:58 ET): live launch is --inventory-mode profit (risk 0) --price-discount 0.40.
    p = Policy(think_seconds=think_seconds, risk=risk, price_discount=price_discount, **kw)

    def policy(state):
        try:
            return p.decide(state)
        except Exception:
            # bot.py play(): strategy exception -> policy_fallback(state)
            return fallback_order(state, p.avoid_event_hotspots, p.event_price_cap, p.regular_rank,
                                  p.regular_price_cap, p.own_budget_limit, p.price_discount)
    policy.p = p
    return policy
