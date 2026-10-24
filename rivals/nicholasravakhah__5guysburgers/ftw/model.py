"""Customer model: crowd sizes and individual customers (price limit, minimum quality).

These are fitted approximations, not the server's engine. Parameters were fitted
to practice-market observations with tools/fit.py and can be overridden by
ftw/params.json. Crowds use forecast.expected_customers as the mean; weather is
sampled once per scenario and its multiplier is renormalized by the forecast odds
(the forecast already averages over weather, so we never double count it).
"""
import json
import math
import os
import random

from .rules import SPOTS

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

_EVENT_ALIASES = [("game", "gameday"), ("midterm", "midterms"), ("concert", "concert"),
                  ("late", "latenight"), ("rush", "latenight"), ("rally", "rally")]


# New spots (docs "New spots and moving"): typical price limit and share wanting premium.
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


def load_params():
    p = json.loads(json.dumps(DEFAULT))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
    try:
        with open(path) as f:
            over = json.load(f)
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
            {"median": 12.0, "sigma": 0.3, "mix": [0.35, 0.45, 0.20]}
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
