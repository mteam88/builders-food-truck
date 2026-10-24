"""Baseline and rival strategies for benchmarking. Each has decide(state) -> order."""
import random

from .strategy import Brain


class Starter:
    """The organizers' starter bot: random spot, standard at $10, full budget."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def decide(self, st):
        return {"spot": self.rng.choice(st["rules"]["spots"]), "price": 10.0, "quality": "standard", "meals": 133}


class Fixed:
    def __init__(self, spot, quality, price, meals):
        self.order = {"spot": spot, "quality": quality, "price": price, "meals": meals}

    def decide(self, st):
        return dict(self.order)


class CrowdFollower:
    """Parks at the biggest forecast crowd."""
    def __init__(self, quality="standard", price=10.0, meals=133):
        self.q, self.p, self.m = quality, price, meals

    def decide(self, st):
        ec = st["forecast"]["expected_customers"]
        return {"spot": max(ec, key=ec.get), "price": self.p, "quality": self.q, "meals": self.m}


class PerCapChaser:
    """Biggest forecast crowd per truck that parked there yesterday; sizes stock to it."""
    def __init__(self, quality="premium", price=12.0):
        self.q, self.p = quality, price

    def decide(self, st):
        ec = st["forecast"]["expected_customers"]
        ld = st.get("last_day") or {}
        cnt = {s: len(v["trucks"]) for s, v in (ld.get("spots") or {}).items()}
        me = st["you"]["id"]
        for s, v in (ld.get("spots") or {}).items():
            if any(t["id"] == me for t in v["trucks"]):
                cnt[s] -= 1
        spot = max(ec, key=lambda s: ec[s] / (1 + cnt.get(s, 0)))
        cap = {"cheap": 200, "standard": 133, "premium": 90}[self.q]
        return {"spot": spot, "price": self.p, "quality": self.q,
                "meals": int(min(cap, 0.8 * ec[spot] / (1 + cnt.get(spot, 0))))}


class Undercutter:
    """Stays at one spot (initially the biggest crowd) and undercuts yesterday's
    cheapest same-quality rival there by 25 cents."""
    def __init__(self, quality="standard", start_price=10.5, floor=6.0):
        self.q, self.p, self.floor, self.spot = quality, start_price, floor, None

    def decide(self, st):
        if self.spot is None:
            ec = st["forecast"]["expected_customers"]
            self.spot = max(ec, key=ec.get)
        ld = st.get("last_day") or {}
        rivals = [t for t in ((ld.get("spots") or {}).get(self.spot, {}).get("trucks") or [])
                  if t["id"] != st["you"]["id"] and t["quality"] == self.q]
        if rivals:
            self.p = max(self.floor, min(t["price"] for t in rivals) - 0.25)
        cap = {"cheap": 200, "standard": 133, "premium": 90}[self.q]
        return {"spot": self.spot, "price": round(self.p, 2), "quality": self.q, "meals": cap}


class Adaptive:
    """Our Brain, observing every state before deciding."""
    def __init__(self, seed=0, cfg=None, memory=None):
        self.brain = Brain(cfg=cfg, seed=seed, memory=memory)
        self.ms = []

    def decide(self, st):
        o = self.brain.decide(st)
        self.ms.append(self.brain.last_debug.get("ms", 0))
        return o

    def observe(self, st):
        self.brain.observe(st)


def house_bots():
    """The practice server's house bots, as observed."""
    return [("h1", "Steady Eddie", Fixed("union", "standard", 10.5, 110)),
            ("h2", "Discount Dan", Fixed("highst", "cheap", 6.5, 180)),
            ("h3", "Fancy Fran", Fixed("thompson", "premium", 14.5, 60))]
