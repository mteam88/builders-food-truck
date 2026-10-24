"""Local market simulator for benchmarking strategies against varied rivals.

This is NOT the server engine. Forecast/event/weather structure and customer
parameters are approximations fitted to practice observations, and each market
perturbs the "true" customer parameters so strategies are tested against model
error. Produces states with the same shape as the API.
"""
import json
import math
import random

from . import engine
from .model import WEATHERS, CustomerModel, load_params
from .rules import COSTS, QUALITIES, REP_DELTA, SPOTS, VALUES, Rules

SHARE = {"highst": 0.24, "oval": 0.20, "union": 0.18, "northres": 0.16, "thompson": 0.14, "stadium": 0.08}
# weather transitions (rows: today -> tomorrow), from practice logs
TRANS = {
    "clear": {"clear": 0.53, "cloudy": 0.28, "rain": 0.16, "cold": 0.03},
    "cloudy": {"clear": 0.27, "cloudy": 0.32, "rain": 0.35, "cold": 0.06},
    "rain": {"clear": 0.18, "cloudy": 0.29, "rain": 0.30, "cold": 0.23},
    "cold": {"clear": 0.21, "cloudy": 0.17, "rain": 0.14, "cold": 0.48},
}
# crowd multipliers by event (relative to an ordinary day), from practice forecasts
EVENT_CROWD = {
    "gameday": {"highst": 2.0, "northres": 1.3, "oval": 1.6, "stadium": 5.0, "thompson": 1.2, "union": 1.5},
    "latenight": {"highst": 1.9, "northres": 1.7, "oval": 0.5, "stadium": 0.5, "thompson": 0.5, "union": 0.5},
    "midterms": {"thompson": 1.7},
    "concert": {"union": 1.8},
    "rally": {"oval": 2.0},
}
EVENT_SPOT = {"gameday": "stadium", "latenight": "highst", "midterms": "thompson", "concert": "union", "rally": "oval"}
EVENT_NAME = {"gameday": "Game day", "latenight": "Late-night rush", "midterms": "Midterms",
              "concert": "Concert at the Union", "rally": "Rally on the Oval"}
EVENT_P = {"latenight": 0.07, "midterms": 0.10, "concert": 0.06, "rally": 0.07}


def perturbed_params(rng, scale=1.0):
    p = load_params()
    for s, v in p["spots"].items():
        v["median"] *= math.exp(rng.gauss(0, 0.06 * scale))
        v["sigma"] *= math.exp(rng.gauss(0, 0.10 * scale))
        mix = [max(0.01, m + rng.gauss(0, 0.04 * scale)) for m in v["mix"]]
        t = sum(mix)
        v["mix"] = [m / t for m in mix]
    p["median_jitter"] = 0.0
    p["mix_jitter"] = 0.0
    return p


class Market:
    def __init__(self, seed, strategies, days=30, model_error=1.0):
        """strategies: list of (truck_id, name, strategy) where strategy.decide(state) -> order."""
        self.rng = random.Random("market|%s" % seed)
        self.crowd_rng = random.Random("crowd|%s" % seed)
        self.days = days
        self.truth = CustomerModel(perturbed_params(random.Random("truth|%s" % seed), model_error))
        self.strats = strategies
        self.ids = [s[0] for s in strategies]
        self.names = {s[0]: s[1] for s in strategies}
        self.cash = {i: 0.0 for i in self.ids}
        self.rep = {i: {s: 50.0 for s in SPOTS} for i in self.ids}
        self.rules = {"budget": 500, "costs": dict(COSTS), "values": dict(VALUES), "spots": list(SPOTS),
                      "qualities": list(QUALITIES), "price_min": 1, "price_max": 30}
        self.R = Rules(self.rules)
        self.weather = "clear"
        self.next_game = self.rng.randint(4, 6)
        self.last_day = None
        self.version = 1
        self.stats = {i: {"profit": 0.0, "cooked": 0, "sold": 0, "missed": 0, "invalid": 0, "days": []}
                      for i in self.ids}

    def _forecast(self, day):
        rng = self.rng
        row = TRANS[self.weather]
        noisy = {w: max(0.005, row[w] * math.exp(rng.gauss(0, 0.5))) for w in WEATHERS}
        t = sum(noisy.values())
        probs = {w: round(v / t, 2) for w, v in noisy.items()}
        events = []
        if day == self.next_game:
            events.append("gameday")
            self.next_game = day + rng.randint(7, 9)
        for k, p in EVENT_P.items():
            if rng.random() < p:
                events.append(k)
        n = max(4, len(self.ids))
        exp_c = {}
        for s in SPOTS:
            m = 1.0
            for e in events:
                m *= EVENT_CROWD[e].get(s, 1.0)
            wt = self.truth.p["weather"][s]
            ew = sum(probs[w] * math.exp(wt[w]) for w in WEATHERS)
            exp_c[s] = int(round(110 * n * SHARE[s] * m * ew))
        labels = {}
        for s in SPOTS:
            r = exp_c[s] / (110 * n * SHARE[s])
            labels[s] = "huge" if r > 2.2 else "high" if r > 1.15 else "medium" if r > 0.7 else "low"
        return {"day": day, "weather": probs,
                "events": [{"key": e, "name": EVENT_NAME[e], "spot": EVENT_SPOT[e], "effect": ""} for e in events],
                "crowds": labels, "expected_customers": exp_c}

    def _state(self, tid, day, forecast, phase="night"):
        trucks = [{"id": i, "truck_name": self.names[i], "company_name": "", "color": "#000",
                   "cash": round(self.cash[i], 2), "reputation_avg": round(sum(self.rep[i].values()) / 6, 1),
                   "reputation": {s: round(v, 2) for s, v in self.rep[i].items()}} for i in self.ids]
        ld = None
        if self.last_day:
            ld = json.loads(json.dumps(self.last_day))
            ld["you"] = self.last_you.get(tid, {"status": "sat_out"})
        return {"version": self.version, "mode": "practice", "market": 1, "markets": 1,
                "day": day - 1, "days": self.days, "phase": phase,
                "orders_open": phase == "night", "order_for_day": day if phase == "night" else None,
                "rules": self.rules, "forecast": forecast,
                "you": {"id": tid, "truck_name": self.names[tid], "cash": round(self.cash[tid], 2),
                        "reputation": {s: round(v, 2) for s, v in self.rep[tid].items()}, "order": None},
                "last_day": ld, "trucks": trucks, "announcements": []}

    def run(self, timer=None):
        for day in range(1, self.days + 1):
            fc = self._forecast(day)
            orders = {}
            for tid, name, strat in self.strats:
                st = self._state(tid, day, fc)
                try:
                    o = strat.decide(st)
                except Exception:
                    o = None
                if o is None:
                    self.stats[tid]["missed"] += 1
                    continue
                v = self.R.validate(o)
                if any(v[k] != o.get(k) for k in v):
                    self.stats[tid]["invalid"] += 1
                orders[tid] = v
            self._play(day, fc, orders)
            self.version += 1
        for tid, name, strat in self.strats:
            if hasattr(strat, "observe"):
                strat.observe(self._state(tid, self.days + 1, None, phase="over"))
        return self.stats

    def _play(self, day, fc, orders):
        rng = self.crowd_rng
        w = self.truth.sample_weather(fc, rng)
        events = fc["events"]
        spots_out = {}
        you = {}
        for s in SPOTS:
            n = self.truth.crowd(s, fc["expected_customers"][s], w, fc, rng)
            med, sig, mix = self.truth.profile(s, events)
            lim, mq = self.truth.customers(n, med, sig, mix, rng, jitter=False)
            here = [(tid, o) for tid, o in orders.items() if o["spot"] == s]
            trucks = [engine.Truck(tid, o["quality"], o["price"], self.rep[tid][s], o["meals"], VALUES)
                      for tid, o in here]
            sold = engine.simulate_spot(lim, mq, trucks, rng) if trucks else {}
            tot = sum(sold.values())
            spots_out[s] = {"customers": n, "hungry": n - tot, "trucks": [
                {"id": tid, "truck_name": self.names[tid], "price": o["price"], "quality": o["quality"],
                 "meals": o["meals"], "sold": sold.get(tid, 0)} for tid, o in here]}
            for tid, o in here:
                k = sold.get(tid, 0)
                cost = o["meals"] * COSTS[o["quality"]]
                rev = k * o["price"]
                self.cash[tid] += rev - cost
                st = self.stats[tid]
                st["profit"] += rev - cost
                st["cooked"] += o["meals"]
                st["sold"] += k
                st["days"].append((day, s, o["quality"], o["price"], o["meals"], k, round(rev - cost, 2)))
                you[tid] = {"status": "played", "spot": s, "price": o["price"], "quality": o["quality"],
                            "meals": o["meals"], "sold": k, "revenue": rev, "cost": cost,
                            "profit": rev - cost, "spoiled": o["meals"] - k}
        for tid in self.ids:
            o = orders.get(tid)
            for s in SPOTS:
                r = self.rep[tid][s]
                if o and o["spot"] == s:
                    k = you[tid]["sold"]
                    if k > 0:
                        r = max(0.0, min(100.0, r + REP_DELTA[o["quality"]]))
                else:
                    r += 0.03 * (50.0 - r)
                self.rep[tid][s] = r
        self.weather = w
        self.last_day = {"day": day, "weather": w, "events": events, "spots": spots_out}
        self.last_you = you
