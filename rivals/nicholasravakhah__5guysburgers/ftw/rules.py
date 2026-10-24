"""Game constants and order validation. Live `state.rules` overrides the defaults."""
import math

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
