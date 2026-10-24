"""Opponent model.

For each rival we keep recency-weighted evidence for several spot-choice
hypotheses and a short history of their orders. Hypothesis scores decay every
observation, so a rival that changes behaviour (new code overnight) is picked
up within a few days; scores are also shrunk at each market reset.

Spot hypotheses (each gives a distribution over spots for tomorrow):
  stay      same spot as their last order
  crowd     follows forecast.expected_customers
  percap    expected customers per truck seen there yesterday (underserved spots)
  hungry    yesterday's hungry counts
  event     the event spot when there is one, otherwise crowd
  favorite  their historical spot frequencies
  uniform   anything
"""
import math

from .rules import QUALITIES

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
