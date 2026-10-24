"""LIVE PORT for T.H.E Truck (t_d64e26): siyanshvir/TruckHomeEatsBot @ 7439109ba07f0c10fd0742b5029cefca437cf735
(2026-10-24 00:14:08 ET "Model feature-gated fair choice walking and salvage with separate learning regimes").
Previous port: policy_e2334345.py (= 5c96779e decisions).

`python3 bot.py` (no flags = live): make_brain(model=None, samples=32, small_crowds=True) -> Brain(Learner(prior.json),
seed=42, samples=32, risk=.08, small_crowds=True). small_crowds = order only at the 3 smallest forecast spots (bottom 2
preferred, 3rd needs +max($15, 15%) profit), price ceilings cheap 5.50 / standard 7.00 / premium 8.50, premium floor 7.25.
At startup (live) Runner restores data/runtime-v1.json, then history.warm_start replays data/live-observations/*.json.
Runner also observes day-phase states (learning before the night window); equivalent to observing in decide.
Deterministic (sha256 seed 42), no wall clock.

7439109b: state['features'] (fair_choice, walk_over, salvage) gates the demand model (mechanics.demand_levels: softmax
fair-choice shares with temperature 1.5, a campus walk-over fluid flow .7*exp(-d/400) to the nearest acceptable spot,
salvage at 25% of food cost in profit and downside loss), and Learner keeps one posterior bank per feature regime
('000' = the legacy prior.json bank; a new regime starts from a FRESH uniform bank, not the prior). Under walk_over the
learner skips the particle (preference) update; crowd calibration still learns. With small_crowds and walk_over, only
the 3 smallest-forecast spots are searched. A Runner order is re-decided if the feature flags change for the same key.

game.py, learning.py, mechanics.py, vendor/pricewars_strategies.py, pricing_models.py, opponents.py, strategy.py,
history.py inlined verbatim (only intra-package imports and the forecast_log.capture observability block removed);
bot.fallback verbatim; model/prior.json (unchanged) inlined as PRIOR_JSON.
"""

# ======================= game.py (verbatim) =======================
"""Published game contracts and an expected customer-flow model. No network or learning."""
import math
from functools import lru_cache
from dataclasses import dataclass

SPOTS = ('highst', 'oval', 'union', 'northres', 'thompson', 'stadium')
QUALITIES = ('cheap', 'standard', 'premium')
COSTS = dict(zip(QUALITIES, (2.5, 3.75, 5.5)))
VALUES = dict(zip(QUALITIES, (8., 12., 16.)))
REP_CHANGE = dict(zip(QUALITIES, (-3, 2, 5)))
# Priors are hypotheses, not engine constants. Historical observations update them.
PRIORS = {
    'highst': (12., 4., (.50, .35, .15), .24),
    'oval': (13., 3., (.40, .40, .20), .20),
    'union': (13., 3., (.20, .60, .20), .18),
    'northres': (11., 2.5, (.50, .42, .08), .16),
    'thompson': (12., 3., (.47, .40, .13), .14),
    'stadium': (13., 3., (.25, .50, .25), .08),
}
WEATHER = {
    'clear': (1., 1., 1., 1., 1., 1.),
    'cloudy': (.95, 1., .95, .94, .96, .98),
    'rain': (.71, .53, .90, .85, .90, .55),
    'cold': (.85, .79, .98, .90, .98, .78),
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def identity(truck):
    return str(truck.get('id') or truck.get('truck_name') or truck.get('name') or '?')


def events(raw):
    if isinstance(raw, dict):
        raw = [dict(v, key=k) if isinstance(v, dict) else k for k, v in raw.items()]
    if isinstance(raw, str):
        raw = [raw]
    text = ' '.join(str(x) for x in (raw or [])).lower()
    out = []
    for name, words in (('game', ('game', 'football')), ('midterm', ('midterm', 'exam')),
                        ('concert', ('concert',)), ('late', ('late', 'rush')),
                        ('rally', ('rally',))):
        if any(w in text for w in words):
            out.append(name)
    return out


def customer_model(spot, kinds, particle):
    mu, sd, mix, _ = PRIORS[spot]
    if 'game' in kinds:
        mu += 1. if spot == 'stadium' else .5
        if spot == 'stadium':
            mix = (.45, .40, .15)
    if 'midterm' in kinds and spot == 'thompson':
        mu += .5
        mix = (.10, .35, .55)
    if 'concert' in kinds and spot == 'union':
        mu += 1.
    if 'late' in kinds and spot in ('highst', 'northres'):
        mix = (.75, .20, .05)
    shift, spread, taste = particle
    move = clamp(taste, -mix[2], mix[0])
    return mu + shift, sd * spread, (mix[0] - move, mix[1], mix[2] + move)


def without_null_counts(state):
    """The API can send null for a spot's expected customers. Treat that as missing, not zero."""
    forecast = state.get('forecast') or {}
    counts = forecast.get('expected_customers') or {}
    if all(v is not None for v in counts.values()):
        return state
    known = {spot: v for spot, v in counts.items() if v is not None}
    return dict(state, forecast=dict(forecast, expected_customers=known))


def weather_probs(forecast):
    raw = forecast.get('weather') or {'clear': 1.}
    if isinstance(raw, str):
        raw = {raw: 1.}
    probs = {str(k).lower(): max(0., float(v)) for k, v in raw.items()}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items() if v > 0} if total else {'clear': 1.}


def weather_factor(weather, spot):
    return WEATHER.get(weather, WEATHER['clear'])[SPOTS.index(spot)]


def spot_results(last):
    raw = last.get('spots') or {}
    if isinstance(raw, list):
        return {x.get('spot') or x.get('id'): x for x in raw}
    return raw


@dataclass(frozen=True)
class Rules:
    budget: float
    costs: dict
    values: dict
    spots: tuple
    price_min: float
    price_max: float

    @classmethod
    def read(cls, state):
        r = state.get('rules') or {}
        return cls(float(r.get('budget', 500)), dict(r.get('costs') or COSTS),
                   dict(r.get('values') or VALUES), tuple(r.get('spots') or SPOTS),
                   float(r.get('price_min', 1)), float(r.get('price_max', 30)))

    def capacity(self, quality):
        return int((self.budget + 1e-8) / self.costs[quality])

    def validate(self, order):
        assert order['spot'] in self.spots
        assert order['quality'] in QUALITIES
        assert self.price_min <= order['price'] <= self.price_max
        assert isinstance(order['meals'], int) and 0 <= order['meals'] <= self.capacity(order['quality'])
        return order


def reputation_value(quality, reputation, values=VALUES):
    return values[quality] * (.7 + .006 * clamp(reputation, 0, 100))


@lru_cache(maxsize=4096)
def _customer_segments(trucks, model, value_items):
    """Cache crowd-independent preferences; retain the existing sequential flow model."""
    values = dict(value_items)
    mu, sd, mix = model
    scores = [reputation_value(q, r, values) - p for p, q, r, _ in trucks]
    segments = []
    for minimum, mass in enumerate(mix):
        eligible = [i for i, (p, q, _, stock) in enumerate(trucks)
                    if QUALITIES.index(q) >= minimum and scores[i] >= -1e-9 and stock > 0]
        prices = sorted({trucks[i][0] for i in eligible})
        for j, price in enumerate(prices):
            low = .5 * (1 + math.erf((price - mu) / (max(sd, .3) * math.sqrt(2))))
            high = (.5 * (1 + math.erf((prices[j+1] - mu) / (max(sd, .3) * math.sqrt(2))))
                    if j+1 < len(prices) else 1.)
            preference = sorted((i for i in eligible if trucks[i][0] <= price),
                                key=lambda i: (-scores[i], trucks[i][0], i))
            if high - low > 1e-10:
                segments.append((mass * (high - low), tuple(preference)))
    return tuple(segments)


def expected_sales(trucks, customers, model, values=VALUES, fair_choice=False):
    """Sequential choice with finite stock; exact ties favor earlier trucks."""
    if fair_choice:
        return list(_local_flow(tuple(tuple(t) for t in trucks), max(0., float(customers)),
                                (model[0], model[1], tuple(model[2])),
                                tuple(sorted(values.items())), True)[0])
    mu, sd, mix = model
    segments = _customer_segments(tuple(tuple(t) for t in trucks),
                                  (mu, sd, tuple(mix)), tuple(sorted(values.items())))
    remaining = [float(t[3]) for t in trucks]
    sold = [0.] * len(trucks)
    duration = 1.
    for _ in range(len(trucks) + 1):
        rates = [0.] * len(trucks)
        for mass, preference in segments:
            for i in preference:
                if remaining[i] > 1e-8:
                    rates[i] += mass * customers
                    break
        dt = min([duration] + [remaining[i] / rate for i, rate in enumerate(rates)
                                if rate > 0 and remaining[i] > 1e-8])
        for i, rate in enumerate(rates):
            bought = min(remaining[i], rate * dt)
            remaining[i] -= bought
            sold[i] += bought
        duration -= dt
        if duration <= 1e-8 or not any(rates):
            break
    return sold


# Official campus distances in metres, symmetric. Walking is modeled as an
# expected fluid flow; the live engine's individual random draws are not replayed.
_DISTANCE_PAIRS = {
    ('oval', 'thompson'): 130, ('oval', 'union'): 470, ('oval', 'stadium'): 590,
    ('oval', 'highst'): 440, ('oval', 'northres'): 550,
    ('thompson', 'union'): 520, ('thompson', 'stadium'): 490,
    ('thompson', 'highst'): 580, ('thompson', 'northres'): 640,
    ('union', 'stadium'): 990, ('union', 'highst'): 570, ('union', 'northres'): 910,
    ('stadium', 'highst'): 980, ('stadium', 'northres'): 840,
    ('highst', 'northres'): 410,
}
DISTANCES = {spot: {} for spot in SPOTS}
for (_left, _right), _distance in _DISTANCE_PAIRS.items():
    DISTANCES[_left][_right] = DISTANCES[_right][_left] = _distance


WALK_DISTANCES = DISTANCES


@lru_cache(maxsize=32768)
def _scalar_budget_cdf(price, mu, sd):
    return .5 * (1. + math.erf((price-mu) / (max(sd, .3)*math.sqrt(2.))))


def _budget_cdf(price, model):
    return _scalar_budget_cdf(price, model[0], model[1])


@lru_cache(maxsize=4096)
def _budget_cohorts(model, prices):
    edges = (float('-inf'),) + tuple(sorted(set(prices))) + (float('inf'),)
    return tuple((mass * (_budget_cdf(upper, model)-_budget_cdf(lower, model)), minimum, lower, upper)
                 for minimum, mass in enumerate(model[2]) if mass > 0
                 for lower, upper in zip(edges, edges[1:])
                 if _budget_cdf(upper, model)-_budget_cdf(lower, model) > 1e-12)


def _choices(trucks, minimum, budget, scores):
    return tuple(i for i, truck in enumerate(trucks)
                 if truck[3] > 0 and QUALITIES.index(truck[1]) >= minimum
                 and truck[0] <= budget and scores[i] >= -1e-9)


def _shares(eligible, remaining, scores, fair_choice):
    available = [i for i in eligible if remaining[i] > 1e-8]
    if not available:
        return ()
    if len(available) == 1:
        return ((available[0], 1.),)
    if not fair_choice:
        best = max(available, key=lambda i: (scores[i], -i))
        return ((best, 1.),)
    top = max(scores[i] for i in available)
    weighted = [(i, math.exp((scores[i]-top)/1.5)) for i in available]
    total = sum(weight for _, weight in weighted)
    return tuple((i, weight/total) for i, weight in weighted)


@lru_cache(maxsize=8192)
def _local_flow(trucks, customers, model, value_items, fair_choice):
    values = dict(value_items)
    cohorts = _budget_cohorts(model, tuple(t[0] for t in trucks))
    scores = [reputation_value(q, r, values)-p for p, q, r, _ in trucks]
    eligible = [_choices(trucks, minimum, lower, scores) for _, minimum, lower, _ in cohorts]
    # Deterministic legacy ties additionally favor a lower selling price.
    if not fair_choice:
        eligible = [tuple(sorted(options, key=lambda i: (-scores[i], trucks[i][0], i))) for options in eligible]
    remaining = [max(0., float(t[3])) for t in trucks]
    sold, served = [0.]*len(trucks), [0.]*len(cohorts)
    duration = 1.
    for _ in range(len(trucks)+1):
        rates, cohort_rates = [0.]*len(trucks), [0.]*len(cohorts)
        for k, (mass, _, _, _) in enumerate(cohorts):
            options = eligible[k]
            if fair_choice:
                shares = _shares(options, remaining, scores, True)
            else:
                first = next((i for i in options if remaining[i] > 1e-8), None)
                shares = () if first is None else ((first, 1.),)
            for i, weight in shares:
                rate = mass*customers*weight
                rates[i] += rate
                cohort_rates[k] += rate
        dt = min([duration]+[remaining[i]/rate for i, rate in enumerate(rates) if rate > 0 and remaining[i] > 1e-8])
        for i, rate in enumerate(rates):
            bought = min(remaining[i], rate*dt)
            remaining[i] -= bought
            sold[i] += bought
        for k, rate in enumerate(cohort_rates):
            served[k] += rate*dt
        duration -= dt
        if duration <= 1e-8 or not any(rates):
            break
    residual = tuple((max(0., mass*customers-served[k]), minimum, lower, upper)
                     for k, (mass, minimum, lower, upper) in enumerate(cohorts)
                     if mass*customers-served[k] > 1e-9)
    return tuple(sold), residual


def expected_market_sales(placements, crowds, models, values=VALUES, fair_choice=False, walk_over=False):
    """Expected sales aligned with each spot's truck list.

    First serve all original local crowds. If enabled, original hungry budget and
    quality cohorts may walk once to their nearest acceptable other spot, with
    probability .7*exp(-distance/400). Remaining inventory is shared across origins.
    This fluid approximation preserves customers and stock, rather than resampling
    a new budget/quality or counting a customer at multiple destinations.
    """
    spots = tuple(dict.fromkeys(tuple(placements)+tuple(crowds)))
    if not walk_over:
        return {spot: expected_sales(placements.get(spot, []), crowds.get(spot, 0.), models[spot],
                                     values, fair_choice=fair_choice) for spot in spots}
    items = tuple(sorted(values.items()))
    sold, remaining, residual = {}, {}, {}
    for spot in spots:
        trucks = tuple(tuple(t) for t in placements.get(spot, []))
        model = models[spot]
        result, hungry = _local_flow(trucks, max(0., float(crowds.get(spot, 0.))),
                                    (model[0], model[1], tuple(model[2])), items, fair_choice)
        sold[spot] = list(result)
        remaining[spot] = [max(0., float(t[3])-result[i]) for i, t in enumerate(trucks)]
        residual[spot] = hungry
    # Split residual budget intervals at all destination prices; this retains
    # each origin's budget distribution after its own local selection.
    all_prices = sorted({t[0] for spot in spots for t in placements.get(spot, [])})
    cohorts = []
    scores = {spot: [reputation_value(t[1], t[2], values)-t[0] for t in placements.get(spot, [])]
              for spot in spots}
    open_spots = tuple(spot for spot in spots if any(stock > 1e-8 for stock in remaining[spot]))
    choices_cache = {}
    for origin in spots:
        model = models[origin]
        for mass, minimum, lower, upper in residual[origin]:
            edges = [lower]+[price for price in all_prices if lower < price < upper]+[upper]
            denominator = _budget_cdf(upper, model)-_budget_cdf(lower, model)
            for left, right in zip(edges, edges[1:]):
                quantity = mass*(_budget_cdf(right, model)-_budget_cdf(left, model))/max(denominator, 1e-15)
                if quantity < 1e-10:
                    continue
                destinations = []
                for destination in open_spots:
                    distance = DISTANCES.get(origin, {}).get(destination)
                    if distance is None:
                        continue
                    key = (destination, minimum, left)
                    options = choices_cache.get(key)
                    if options is None:
                        options = tuple(i for i in _choices(placements.get(destination, []), minimum, left, scores[destination])
                                        if remaining[destination][i] > 1e-8)
                        if not fair_choice:
                            options = tuple(sorted(options, key=lambda i: (-scores[destination][i], placements[destination][i][0], i)))
                        choices_cache[key] = options
                    if options:
                        destinations.append((distance, destination, options))
                if destinations:
                    cohorts.append((quantity, tuple(sorted(destinations))))
    duration = 1.
    for _ in range(sum(len(remaining[spot]) for spot in spots)+1):
        rates = {spot: [0.]*len(remaining[spot]) for spot in spots}
        for quantity, destinations in cohorts:
            for distance, destination, eligible in destinations:
                if fair_choice:
                    shares = _shares(eligible, remaining[destination], scores[destination], True)
                else:
                    first = next((i for i in eligible if remaining[destination][i] > 1e-8), None)
                    shares = () if first is None else ((first, 1.),)
                if shares:
                    flow = quantity*.7*math.exp(-distance/400.)
                    for i, weight in shares:
                        rates[destination][i] += flow*weight
                    break
        if not any(rate > 0 for local in rates.values() for rate in local):
            break
        dt = min([duration]+[remaining[spot][i]/rate for spot in spots for i, rate in enumerate(rates[spot])
                             if rate > 0 and remaining[spot][i] > 1e-8])
        for spot in spots:
            for i, rate in enumerate(rates[spot]):
                bought = min(remaining[spot][i], rate*dt)
                remaining[spot][i] -= bought
                sold[spot][i] += bought
        duration -= dt
        if duration <= 1e-8:
            break
    return sold

# ======================= learning.py (verbatim) =======================
"""Online Bayesian model averaging, shared by training and the playing policy."""
import copy
import json
import math
from pathlib import Path

# Broad enough to represent uncertainty rather than fitting one guessed distribution.
PARTICLES = [(mu, sd, taste) for mu in (-3., -1.5, 0., 1.5, 3.)
             for sd in (.65, 1., 1.4) for taste in (-.12, 0., .12)]


FEATURE_NAMES = ('fair_choice', 'walk_over', 'salvage')


def feature_tuple(features):
    """Canonical realized engine flags; never read tomorrow's forecast."""
    if features is None:
        return (False, False, False)
    if isinstance(features, (tuple, list)) and len(features) == len(FEATURE_NAMES):
        return tuple(bool(flag) for flag in features)
    if not isinstance(features, dict):
        raise ValueError('Expected feature flags as a mapping or three-element sequence')
    def enabled(value):
        return bool(value.get('enabled', False)) if isinstance(value, dict) else bool(value)
    return tuple(enabled(features.get(name, False)) for name in FEATURE_NAMES)


class Learner:
    VERSION = 1

    def __init__(self):
        self.active_features = (False, False, False)
        self.regimes = {}
        self._bind(self._fresh_bank())
        self.regimes[self._regime_key(self.active_features)] = self._current_bank()

    @staticmethod
    def _fresh_bank():
        return {'weights': {s: [0.] * len(PARTICLES) for s in SPOTS},
                'counts': {s: 0 for s in SPOTS}, 'crowd': {s: {} for s in SPOTS},
                'crowd_variance': {s: {} for s in SPOTS}, 'rows': 0}

    @staticmethod
    def _regime_key(features):
        return ''.join('1' if flag else '0' for flag in features)

    def _current_bank(self):
        return {name: getattr(self, name) for name in
                ('weights', 'counts', 'crowd', 'crowd_variance', 'rows')}

    def _bind(self, bank):
        for name, value in bank.items():
            setattr(self, name, value)

    def activate(self, features):
        """Select a rules-specific posterior, retaining all other feature regimes.

        New engine behavior starts with the broad customer prior, rather than
        transferring likelihoods fitted under different customer-choice rules.
        Rival history lives outside Learner and is unaffected.
        """
        flags = feature_tuple(features)
        if flags == self.active_features:
            return False
        self.regimes[self._regime_key(self.active_features)] = self._current_bank()
        key = self._regime_key(flags)
        if key not in self.regimes:
            self.regimes[key] = self._fresh_bank()
        self._bind(self.regimes[key])
        self.active_features = flags
        return True

    def to_dict(self):
        # Detach serialized snapshots from both live state and other banks.
        banks = dict(self.regimes)
        banks[self._regime_key(self.active_features)] = self._current_bank()
        snapshot = copy.deepcopy(self._current_bank())
        snapshot.update(version=self.VERSION, active_features=list(self.active_features),
                        regimes=copy.deepcopy(banks))
        return snapshot

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != cls.VERSION:
            raise ValueError('Unsupported learned-model version')
        def read_bank(raw):
            bank = cls._fresh_bank()
            bank.update({name: copy.deepcopy(raw[name]) for name in bank if name in raw})
            for spot in SPOTS:
                weights = bank['weights'][spot]
                if len(weights) != len(PARTICLES) or not all(math.isfinite(x) for x in weights):
                    raise ValueError('Invalid learned weights')
            return bank
        obj = cls()
        flags = feature_tuple(data.get('active_features'))
        obj.regimes = {}
        for key, raw in (data.get('regimes') or {}).items():
            if len(key) != 3 or any(char not in '01' for char in key):
                raise ValueError('Invalid learning regime key')
            obj.regimes[key] = read_bank(raw)
        # The existing version-1 fields remain authoritative for the active bank.
        # Older files have no regime metadata and are therefore legacy (000).
        active = cls._regime_key(flags)
        obj.regimes[active] = read_bank(data)
        obj._bind(obj.regimes[active])
        obj.active_features = flags
        return obj

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def posterior(self, spot):
        raw = self.weights[spot]
        maximum = max(raw)
        weights = [math.exp(x - maximum) for x in raw]
        total = sum(weights)
        return [x / total for x in weights]

    def sample(self, spot, rng):
        return rng.choices(PARTICLES, weights=self.posterior(spot), k=1)[0]

    def confidence(self, spot):
        return self.counts[spot] / (self.counts[spot] + 20.)

    def learn(self, last, before_reps=None, forecast=None, rules=None, strength=1., features=None):
        if features is not None:
            self.activate(features)
        fair_choice, walk_over, _ = self.active_features
        rules = rules or Rules.read({})
        before_reps = before_reps or {}
        kinds = events(last.get('events'))
        weather = last.get('weather', 'clear')
        if isinstance(weather, dict):
            weather = weather.get('kind') or weather.get('type') or 'clear'
        for spot, info in spot_results(last).items():
            if spot not in SPOTS:
                continue
            customers = max(0., float(info.get('customers', 0)))
            records = info.get('trucks') or []
            trucks = [(float(t['price']), t['quality'],
                       float(before_reps.get(identity(t), {}).get(spot, 50)), int(t['meals']))
                      for t in records if t.get('quality') in QUALITIES]
            records = [t for t in records if t.get('quality') in QUALITIES]
            # Walk-over changes local arrivals after the origin crowd is drawn.
            # Until a joint customer-flow likelihood exists, avoid falsely fitting
            # those transferred sales as local preferences. Origin crowd residuals
            # below remain usable and segregated in this feature regime.
            if customers and trucks and not walk_over:
                losses = []
                for particle in PARTICLES:
                    predictions = expected_sales(trucks, customers, customer_model(spot, kinds, particle),
                                                 rules.values, fair_choice=True) if fair_choice else expected_sales(
                        trucks, customers, customer_model(spot, kinds, particle), rules.values)
                    loss = 0.
                    for predicted, t in zip(predictions, records):
                        actual = float(t.get('sold', 0))
                        # A sellout is a lower bound, never evidence that demand stops there.
                        error = max(0., actual - predicted) if actual >= t['meals'] else predicted - actual
                        loss += error * error / (actual + 12.)
                    losses.append(loss / max(1, len(records)))
                updated = [.98 * w - strength * min(loss, 80.) / 5.
                           for w, loss in zip(self.weights[spot], losses)]
                top = max(updated)
                self.weights[spot] = [max(-16., w - top) for w in updated]
                self.counts[spot] += 1
            if forecast:
                expected = (forecast.get('expected_customers') or {}).get(spot)
                if expected is not None and float(expected) > 0:
                    odds = weather_probs(forecast)
                    average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                    conditional = float(expected) * weather_factor(str(weather), spot) / max(average, .1)
                    ratio = clamp(customers / max(conditional, 1.), .25, 3.)
                    key = str(weather)
                    old = self.crowd[spot].get(key, [1., 0])
                    # Shrink toward 1 when data are sparse. Event/weather effects in the
                    # supplied expected counts are never multiplied in a second time.
                    self.crowd[spot][key] = [.9 * old[0] + .1 * ratio, old[1] + 1]
                    # Online variance of weather-normalized forecast errors, including
                    # zero-customer days. Older checkpoints start with the prior.
                    count, mean, m2 = self.crowd_variance[spot].get(key, [0, 0., 0.])
                    observation = clamp(customers / max(conditional, 1.), 0., 3.)
                    count += 1
                    delta = observation - mean
                    mean += delta / count
                    m2 += delta * (observation - mean)
                    self.crowd_variance[spot][key] = [count, mean, m2]
        self.rows += 1

    def crowd_multiplier(self, spot, weather):
        mean, count = self.crowd[spot].get(weather, [1., 0])
        return 1. + (mean - 1.) * count / (count + 12.)

    def crowd_noise(self, spot, weather):
        """Relative standard deviation; shrink sparse observations toward old +/-15% noise."""
        count, mean, m2 = self.crowd_variance[spot].get(weather, [0, 0., 0.])
        prior_variance = .15**2 / 3.
        relative_m2 = m2 / max(.25, mean)**2 if count else 0.
        variance = (relative_m2 + 12 * prior_variance) / (max(0, count-1) + 12)
        return clamp(math.sqrt(variance), .04, .40)


def training_rows(path):
    """Saved collection pairs include exact pre-day local reputations and forecasts."""
    with open(path) as source:
        for line in source:
            row = json.loads(line)
            if 'pre' in row and 'last_day' in row:
                yield row


def train(rows):
    learner = Learner()
    for row in rows:
        reps = {identity(t): t.get('rep') or t.get('reputation') or {} for t in row['pre'].get('trucks', [])}
        learner.learn(row['last_day'], reps, row['pre'].get('forecast'),
                      features=row['pre'].get('features') or {})
    return learner

# ======================= mechanics.py (verbatim) =======================
"""Feature-aware projection and food accounting shared by planning and logging."""

FEATURE_NAMES = ('fair_choice', 'walk_over', 'salvage')


def features(raw=None):
    return dict(zip(FEATURE_NAMES, feature_tuple(raw)))


def food_profit(sales, meals, price, cost, salvage=False):
    return sales*price-meals*cost + max(0., meals-sales)*cost*(.25 if salvage else 0.)


def demand_levels(state, rules, spot, quality, price, scenario):
    """Uncapped demand at three production levels; walking runs after local sales.

    Like the legacy planner, inventory optimization caps this uncapped flow.
    That is an approximation when our stock changes other trucks' leftovers;
    the independent simulator tests finite inventories directly.
    """
    placement, crowds, models, active = scenario
    flags = features(state.get('features'))
    rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
    trucks = list(placement[spot]) + [(price, quality, rep, float('inf'))]
    forecast = state.get('forecast') or {}
    supplied = spot in (forecast.get('expected_customers') or {})
    roster_basis = forecast.get('population_basis') == 'registered_trucks'
    baseline = max(4, len(state.get('trucks') or []))
    levels = ([1., 1., 1.] if supplied and not roster_basis else
              [max(4., active+x)/baseline for x in (0., .5, 1.)])
    demands = {}
    for scale in set(levels):
        if flags['walk_over']:
            market = dict(placement, **{spot: trucks})
            result = expected_market_sales(market, {s: n*scale for s, n in crowds.items()}, models,
                                           rules.values, fair_choice=flags['fair_choice'], walk_over=True)
            demands[scale] = result[spot][-1]
        else:
            demands[scale] = expected_sales(trucks, crowds[spot]*scale, models[spot], rules.values,
                                           fair_choice=flags['fair_choice'])[-1]
    return [demands[scale] for scale in levels]

# ======================= vendor/pricewars_strategies.py (verbatim) =======================
"""Price Wars reference strategies (MIT), vendored without network dependencies.
Source: https://github.com/hpi-epic/pricewars-merchant/blob/7981dad52034493f1f77e21e2aa297598c6ef769/merchant.py
"""

class CheapestStrategy:
    name = 'cheapest'

    settings = {
        'price_decrement': 0.05,
        'default_price': 30
    }

    # set color used in graphs for this merchant, if not set it will be a random color
    # e31a1c = red
    color = "#e31a1c"

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['default_price']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        return cheapest_offer.price - merchant.settings['price_decrement']

class TwoBoundStrategy:
    name = 'two_bound'

    settings = {
        'price_decrement': 0.10,
        'upper_price_bound': 30,
        'lower_price_bound': 20
    }

    # 33a02c = green
    color = "#33a02c"

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['upper_price_bound']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        if cheapest_offer.price <= merchant.settings['lower_price_bound'] or \
                cheapest_offer.price > merchant.settings['upper_price_bound']:
            return merchant.settings['upper_price_bound']
        else:
            return cheapest_offer.price - merchant.settings['price_decrement']

# ======================= pricing_models.py (verbatim) =======================
"""Adapt the vendored Price Wars reference merchants to Night Shift price forecasts."""
from types import SimpleNamespace


def reference_prices(competitor_prices, default, floor, upper):
    offers = [SimpleNamespace(offer_id='ours', product_id='meal', merchant_id='rival', price=default)]
    offers += [SimpleNamespace(offer_id=str(i), product_id='meal', merchant_id='competitor', price=price)
               for i, price in enumerate(competitor_prices)]
    settings = dict(CheapestStrategy.settings, default_price=default)
    cheapest = CheapestStrategy.calculate_price(SimpleNamespace(merchant_id='rival', settings=settings), 'ours', offers)
    settings = dict(TwoBoundStrategy.settings, lower_price_bound=floor, upper_price_bound=upper)
    cycle = TwoBoundStrategy.calculate_price(SimpleNamespace(merchant_id='rival', settings=settings), 'ours', offers)
    # Night Shift requires a food-cost floor for this rival hypothesis. The actual
    # optimizer remains free to price independently and pursue overflow.
    return max(floor, cheapest), max(floor, cycle)

# ======================= opponents.py (verbatim) =======================
"""Online probabilistic rival forecasts, scored before observing each new result.

A mixture learns which behavior models explain each rival. Contextual historical
neighbors use announced events, forecast crowd shares, and weather probabilities.
Price Wars' documented undercut/cycle behaviors are pricing hypotheses, not rules
that our own order must obey. No future or private orders are used.
"""
import math

LOCATION_MODELS = ('uniform', 'habit', 'repeat', 'crowd', 'context', 'opportunity')
PRICE_MODELS = ('context', 'repeat', 'undercut', 'cycle')


def normalize(values):
    total = sum(values)
    return [x/total for x in values] if total else [1/len(values)]*len(values)


def posterior(logs):
    maximum = max(logs)
    return normalize([math.exp(x-maximum) for x in logs])


def context(forecast):
    forecast = forecast or {}
    counts = forecast.get('expected_customers') or {}
    total = sum(float(v) for v in counts.values()) or 1.
    return {'events': events(forecast.get('events')), 'weather': weather_probs(forecast),
            'shares': {s: float(counts.get(s, 0))/total for s in SPOTS}}


def similarity(left, right):
    le, re = set(left.get('events', [])), set(right.get('events', []))
    event_distance = len(le ^ re)
    crowd_distance = sum(abs(left.get('shares', {}).get(s, 0)-right.get('shares', {}).get(s, 0)) for s in SPOTS)
    weather_distance = sum(abs(left.get('weather', {}).get(w, 0)-right.get('weather', {}).get(w, 0))
                           for w in ('clear', 'cloudy', 'rain', 'cold'))
    return math.exp(-1.5*event_distance-2.*crowd_distance-.5*weather_distance)


class OpponentModel:
    def __init__(self, data=None):
        self.data = data or {}

    def entry(self, name):
        return self.data.setdefault(name, {'location': [0.]*len(LOCATION_MODELS),
                                          'price': [0.]*len(PRICE_MODELS),
                                          'active': 0., 'absent': 0., 'scored': 0,
                                          'log_loss': 0., 'repeat_log_loss': 0.,
                                          'price_count': 0, 'price_error': 0., 'repeat_price_error': 0.})

    def attendance(self, name):
        e = self.entry(name)
        # Unknown trucks get an explicitly uncertain prior; a single absence is
        # weak evidence. Multi-day absences reduce both competition and demand.
        return clamp((3.+e['active'])/(4.+e['active']+e['absent']), .05, .98)

    def weights(self, history, forecast):
        current = context(forecast)
        return [.96**(len(history)-index-1) * (.05+.95*similarity(r.get('context', {}), current))
                for index, r in enumerate(history)]

    def location_experts(self, name, history, forecast, histories, spots=SPOTS):
        counts = (forecast or {}).get('expected_customers') or {}
        uniform = [1/len(spots)]*len(spots)
        habit, nearby = [.2]*len(spots), [.2]*len(spots)
        weights = self.weights(history, forecast)
        for age, (rec, weight) in enumerate(zip(history, weights)):
            if rec['spot'] in spots:
                i = spots.index(rec['spot'])
                habit[i] += .96**(len(history)-age-1)
                nearby[i] += weight*3.
        repeat = [.2/len(spots)]*len(spots)
        if history and history[-1]['spot'] in spots:
            repeat[spots.index(history[-1]['spot'])] += .8
        crowd = normalize([max(0., float(counts.get(s, 0)))**3 for s in spots])
        pressure = {s: 0. for s in spots}
        for rival, records in histories.items():
            if rival == name or not records:
                continue
            total = sum(.96**age for age in range(len(records)))
            for age, rec in enumerate(reversed(records)):
                if rec['spot'] in pressure:
                    pressure[rec['spot']] += rec['meals']*.96**age/total
        opportunity = normalize([float(counts.get(s, 0))/(1.+pressure[s]/max(1., float(counts.get(s, 0)))) for s in spots])
        return [uniform, normalize(habit), normalize(repeat), crowd, normalize(nearby), opportunity]

    def placement(self, name, history, forecast, histories, spots=SPOTS):
        experts = self.location_experts(name, history, forecast, histories, spots)
        weights = posterior(self.entry(name)['location'])
        # A uniform floor keeps unobserved moves possible as rivals change code.
        return normalize([.05/len(spots)+.95*sum(w*p[i] for w, p in zip(weights, experts)) for i in range(len(spots))])

    def quality(self, history, forecast):
        weights = dict.fromkeys(QUALITIES, .3)
        for r, w in zip(history, self.weights(history, forecast)):
            weights[r['quality']] += w*3.
        return normalize([weights[q] for q in QUALITIES])

    def price_experts(self, name, history, forecast, histories, quality, spot=None):
        paired = [(r, w) for r, w in zip(history, self.weights(history, forecast)) if r['quality'] == quality]
        default = {'cheap': 6.5, 'standard': 9., 'premium': 10.}[quality]
        if paired:
            total = sum(w for _, w in paired)
            mean = sum(r['price']*w for r, w in paired)/total
            variance = sum(w*(r['price']-mean)**2 for r, w in paired)/total
            stock = sum(r['meals']*w for r, w in paired)/total
            repeat = paired[-1][0]['price']
        else:
            mean, variance, stock, repeat = default, 1., 80., default
        competitors = [h[-1]['price'] for rival, h in histories.items() if rival != name and h
                       and h[-1]['quality'] == quality and (spot is None or h[-1]['spot'] == spot)]
        floor, upper = COSTS[quality]+.5, min(VALUES[quality]-.5, default+3.)
        undercut, cycle = reference_prices(competitors, mean, floor, upper)
        return [mean, repeat, undercut, cycle], max(.2, min(2., math.sqrt(variance+.16))), stock

    def pricing(self, name, history, forecast, histories, quality, spot=None):
        experts, spread, stock = self.price_experts(name, history, forecast, histories, quality, spot)
        weights = posterior(self.entry(name)['price'])
        mean = sum(w*x for w, x in zip(weights, experts))
        variance = sum(w*(x-mean)**2 for w, x in zip(weights, experts))
        return mean, min(2.5, math.sqrt(spread*spread+variance)), stock

    def observe(self, name, record, history, forecast, histories, spots=SPOTS):
        e = self.entry(name)
        # Gently forget old attendance so a previously inactive truck can return.
        e['active'] *= .97
        e['absent'] *= .97
        e['active' if record else 'absent'] += 1.
        if record is None:
            return
        if forecast and (forecast.get('expected_customers') or {}):
            experts = self.location_experts(name, history, forecast, histories, spots)
            probabilities = self.placement(name, history, forecast, histories, spots)
            index = spots.index(record['spot'])
            e['log_loss'] -= math.log(max(probabilities[index], 1e-9))
            e['repeat_log_loss'] -= math.log(max(experts[2][index], 1e-9))
            e['scored'] += 1
            updated = [.98*w+.7*math.log(max(p[index], 1e-6)) for w, p in zip(e['location'], experts)]
            maximum = max(updated)
            e['location'] = [max(-12., x-maximum) for x in updated]
        if history:
            quality = record['quality']
            experts, spread, _ = self.price_experts(name, history, forecast, histories, quality, record['spot'])
            predicted, _, _ = self.pricing(name, history, forecast, histories, quality, record['spot'])
            e['price_count'] += 1
            e['price_error'] += abs(predicted-record['price'])
            e['repeat_price_error'] += abs(experts[1]-record['price'])
            updated = [.98*w-.35*min(25., ((p-record['price'])/max(.8, spread))**2)
                       for w, p in zip(e['price'], experts)]
            maximum = max(updated)
            e['price'] = [max(-12., x-maximum) for x in updated]

    def summary(self, name, history, forecast, histories, spots=SPOTS):
        e = self.entry(name)
        return {'attendance': self.attendance(name), 'history_days': len(history),
                'locations': dict(zip(spots, self.placement(name, history, forecast, histories, spots))),
                'location_models': dict(zip(LOCATION_MODELS, posterior(e['location']))),
                'price_models': dict(zip(PRICE_MODELS, posterior(e['price'])))}

# ======================= strategy.py (verbatim) =======================
"""Learned sales forecasting plus exhaustive location/quality profit optimization.

Every spot receives the same fixed search effort. No crowd-ordered early exit,
forced rival undercut, premium-only switch, or arbitrary selling-price cap.
"""
import hashlib
import math
import random
import time


class Brain:
    def __init__(self, learner=None, seed=42, samples=12, risk=.15, price_cap=None, quiet_bonus=0., small_crowds=False):
        self.learner = learner or Learner()
        self.seed, self.samples, self.risk = seed, max(4, int(samples)), float(risk)
        # Optional steering: never charge above price_cap; reward spots where rivals rarely park.
        self.price_cap, self.quiet_bonus = price_cap, float(quiet_bonus)
        self.small_crowds = bool(small_crowds)
        self.low_price_caps = {'cheap': 5.50, 'standard': 7.00, 'premium': 8.50}
        self.market = None
        self.seen = None
        self.history = {}
        self.reputations = {}
        self.pending = None
        self.last_order = None
        self.diagnostics = {}
        self.opponents = OpponentModel()
        self.prediction_checks = []
        self.forecast_record = None
        self.scenario_rivals = []

    def dump(self):
        return {'learner': self.learner.to_dict(), 'market': self.market, 'seen': self.seen,
                'history': self.history, 'reputations': self.reputations,
                'pending': self.pending, 'last_order': self.last_order, 'opponents': self.opponents.data,
                'prediction_checks': self.prediction_checks}

    def restore(self, data):
        self.learner = Learner.from_dict(data['learner'])
        self.opponents = OpponentModel(data.get('opponents'))
        self.prediction_checks = data.get('prediction_checks', [])
        for key in ('market', 'seen', 'history', 'reputations', 'pending', 'last_order'):
            setattr(self, key, data[key])

    def observe(self, state):
        key = [state.get('mode'), state.get('market')]
        if key != self.market:
            self.market = key
            self.seen = None
            self.reputations = {}
            self.pending = None
            self.last_order = None
        last = state.get('last_day') or {}
        if last.get('day') is None or (self.seen is not None and last['day'] <= self.seen):
            return
        me = identity(state.get('you') or {})
        rules = Rules.read(state)
        before = {name: dict(reps) for name, reps in self.reputations.items()}
        # Our local reputation is public; undo this day's increment to fit pre-day sales.
        own_reps = dict((state.get('you') or {}).get('reputation') or {})
        own = last.get('you') or {}
        if own.get('sold', 0) > 0 and own.get('spot') in own_reps:
            s = own['spot']
            # At the cap the exact previous value is ambiguous; prefer tracked history.
            own_reps[s] = before.get(me, {}).get(s, clamp(own_reps[s] - REP_CHANGE[own['quality']], 0, 100))
        before[me] = own_reps
        forecast = self.pending['forecast'] if self.pending and self.pending['day'] == last['day'] else None
        # state.features can describe tomorrow after a toggle. Fit completed
        # results only with their captured pre-order flags or explicit result flags.
        realized_features = last.get('features')
        if realized_features is None and self.pending and self.pending['day'] == last['day']:
            realized_features = self.pending.get('features')
        if realized_features is None and 'features' not in state:
            realized_features = {}  # Legacy archives and fixtures before feature flags.
        prediction = (self.pending or {}).get('prediction') if forecast else None
        if prediction:
            order = prediction['order']
            actual = spot_results(last).get(order['spot'], {})
            matched = all(own.get(k) == v for k, v in order.items())
            sold = float(own.get('sold', 0))
            check = {'mode': key[0], 'market': key[1], 'day': last['day'],
                     'forecast_id': prediction.get('forecast_id'),
                     'rules_digest': prediction.get('rules_digest'), 'regime_id': prediction.get('regime_id'),
                     'features': prediction.get('features'),
                     'order': order, 'order_matches': matched,
                     'expected_sales': prediction['sales'], 'actual_sales': sold,
                     'expected_profit': prediction['profit'], 'actual_profit': own.get('profit'),
                     'expected_rivals': prediction['rivals'],
                     'actual_rivals': sum(identity(t) != me for t in actual.get('trucks', [])),
                     'sellout': matched and order['meals'] > 0 and sold >= order['meals']}
            # A sold-out truck reveals only a lower bound on uncapped demand.
            check['demand_error'] = (max(0., sold-prediction['demand']) if check['sellout']
                                     else abs(sold-prediction['demand'])) if matched else None
            self.prediction_checks = (self.prediction_checks + [check])[-48:]
        if realized_features is not None:
            self.learner.learn(last, before, forecast, rules, strength=.7, features=realized_features)
        observed_context = context(forecast or {'events': last.get('events') or []})
        visited = {}
        records = {}
        for spot, info in spot_results(last).items():
            if spot not in SPOTS:
                continue
            for truck in info.get('trucks') or []:
                name = identity(truck)
                visited[name] = (spot, truck)
                if truck.get('quality') in QUALITIES:
                    rec = {'spot': spot, 'price': float(truck['price']), 'quality': truck['quality'],
                           'meals': int(truck['meals']), 'sold': int(truck.get('sold', 0)),
                           'context': observed_context, 'day': last['day']}
                    rec['features'] = features(realized_features) if realized_features is not None else None
                    records[name] = rec
        # Score each prediction before adding ANY of today's orders to history.
        # This prevents both future-data leakage and truck iteration-order effects.
        roster = {identity(t) for t in state.get('trucks') or []} | set(records)
        for name in sorted(roster - {me}):
            self.opponents.observe(name, records.get(name), self.history.get(name, []),
                                   forecast, self.history, rules.spots)
        for name, rec in records.items():
            self.history.setdefault(name, []).append(rec)
            self.history[name] = self.history[name][-48:]
        for name in set(self.reputations) | set(visited):
            reps = self.reputations.setdefault(name, {s: 50. for s in SPOTS})
            for spot in SPOTS:
                if name in visited and visited[name][0] == spot:
                    t = visited[name][1]
                    if t.get('sold', 0) > 0:
                        reps[spot] = clamp(reps[spot] + REP_CHANGE.get(t.get('quality'), 0), 0, 100)
                else:
                    reps[spot] += .03 * (50 - reps[spot])
        for t in state.get('trucks') or []:
            if isinstance(t.get('reputation'), dict):
                self.reputations[identity(t)] = dict(t['reputation'])
        self.reputations[me] = dict((state.get('you') or {}).get('reputation') or {s: 50. for s in SPOTS})
        self.seen = last['day']

    def forecast(self, state, rules):
        fc = state.get('forecast') or {}
        given = fc.get('expected_customers') or {}
        kinds = events(fc.get('events'))
        odds = weather_probs(fc)
        counts = {}
        for s in rules.spots:
            if s in given:  # Zero is meaningful; do not replace it with a prior.
                counts[s] = max(0., float(given[s]))
                continue
            base = 110 * max(4, len(state.get('trucks') or [])) * PRIORS[s][3]
            for event in kinds:
                mult = {'game': {'stadium': 5, 'highst': 1.5, '*': 1.2},
                        'midterm': {'thompson': 1.7}, 'concert': {'union': 1.8},
                        'late': {'highst': 2, 'northres': 2, '*': .5}, 'rally': {'oval': 2}}[event]
                base *= mult.get(s, mult.get('*', 1.))
            counts[s] = base * sum(p * weather_factor(w, s) for w, p in odds.items())
        return counts

    def scenarios(self, state, rules, counts, rng):
        """Joint rival placements and customer uncertainty, shared across candidate orders."""
        me = identity(state.get('you') or {})
        roster = [identity(t) for t in state.get('trucks') or [] if identity(t) != me]
        fc = state.get('forecast') or {}
        odds = weather_probs(fc)
        kinds = events(fc.get('events'))
        field = [h[-1] for name, h in self.history.items() if name in roster and h]
        scenarios = []
        self.scenario_rivals = []
        for index in range(self.samples):
            # Stratified weather sampling reduces random ranking fluctuations.
            target, acc = (index + .5) / self.samples, 0.
            weather = list(odds)[-1]
            for w, p in sorted(odds.items()):
                acc += p
                if target <= acc:
                    weather = w
                    break
            placements = {s: [] for s in rules.spots}
            active = 0.
            named = []
            for name in roster:
                hist = self.history.get(name, [])
                if rng.random() > self.opponents.attendance(name):
                    continue
                weights = self.opponents.placement(name, hist, fc, self.history, rules.spots)
                spot = rng.choices(rules.spots, weights=weights, k=1)[0]
                if hist:
                    quality_weights = self.opponents.quality(hist, fc)
                    q = rng.choices(QUALITIES, weights=quality_weights, k=1)[0]
                    mean, spread, stock = self.opponents.pricing(name, hist, fc, self.history, q, spot)
                    price = rng.gauss(mean, spread)
                    meals = max(0, min(rules.capacity(q), int(stock * rng.uniform(.85, 1.15))))
                else:
                    rec = rng.choice(field) if field else None
                    q = rec['quality'] if rec else rng.choice(QUALITIES)
                    price = rec['price'] if rec else {'cheap': 6.5, 'standard': 9., 'premium': 10.}[q]
                    price += rng.gauss(0, .7)
                    meals = min(rules.capacity(q), int(rec['meals']) if rec else 80)
                price = round(clamp(price, rules.price_min, rules.price_max), 2)
                rep = self.reputations.get(name, {}).get(spot, 50.)
                rep = clamp(rep + rng.gauss(0, 5 if hist else 12), 0, 100)
                placements[spot].append((price, q, rep, meals))
                named.append({'id': name, 'spot': spot, 'price': price,
                              'quality': q, 'reputation': rep, 'meals': meals})
                active += min(1., meals * rules.costs[q] / 250.)
            crowds, models = {}, {}
            for spot in rules.spots:
                average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                conditional = weather_factor(weather, spot) / max(average, .1)
                correction = self.learner.crowd_multiplier(spot, weather)
                width = math.sqrt(3.) * self.learner.crowd_noise(spot, weather)
                crowds[spot] = counts[spot] * conditional * correction * rng.uniform(1-width, 1+width)
                models[spot] = customer_model(spot, kinds, self.learner.sample(spot, rng))
            scenarios.append((placements, crowds, models, active))
            self.scenario_rivals.append(named)
        return scenarios

    def evaluate(self, state, rules, spot, quality, price, scenarios):
        rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
        cost = rules.costs[quality]
        cap = rules.capacity(quality)
        flags = features(state.get('features'))
        distributions = [demand_levels(state, rules, spot, quality, price, scenario) for scenario in scenarios]
        remaining = max(0, int(state.get('days', 30)) - int(state.get('order_for_day', 1)))
        rep_delta = clamp(rep + REP_CHANGE[quality], 0, 100) - rep
        best = None
        inventory_options = []
        n = len(distributions)
        quiet = sum(1 for placement, _, _, _ in scenarios if not placement[spot]) / max(1, n)
        # Every feasible integer inventory is considered, including zero.
        for meals in range(cap + 1):
            contribution = min(1., meals * cost / 250.)
            left = 0 if contribution <= .5 else 1
            fraction = contribution * 2 if left == 0 else (contribution - .5) * 2
            demand = [d[left] * (1-fraction) + d[left+1] * fraction for d in distributions]
            sales = [min(meals, x) for x in demand]
            mean = sum(sales) / n
            profit = food_profit(mean, meals, price, cost, flags['salvage'])
            loss = sum(max(0., -food_profit(x, meals, price, cost, flags['salvage'])) for x in sales) / n
            # Small, bounded future benefit: no incentive to lose hundreds buying reputation.
            sale_probability = sum(x > .1 for x in sales) / n
            future = .006 * rules.values[quality] * rep_delta * mean * min(remaining, 6) * .15 * sale_probability
            future = clamp(future, -25., 25.)
            exploration = 3. * min(1., remaining / 15.) * (1-self.learner.confidence(spot)) * sale_probability
            score = profit - self.risk * loss + future + exploration
            if meals and self.quiet_bonus:
                score += self.quiet_bonus * quiet
            candidate = {'score': score, 'profit': profit, 'sales': mean, 'loss': loss,
                        'demand': sum(demand) / n,
                        'expected_unsold': meals - mean,
                        'expected_salvage': (meals-mean)*cost*(.25 if flags['salvage'] else 0.),
                        'sellout_probability': sum(x >= meals for x in demand) / n if meals else 0.,
                        'order': {'spot': spot, 'quality': quality, 'price': price, 'meals': meals}}
            inventory_options.append(candidate)
            if best is None or score > best['score']:
                best = candidate
        if self.small_crowds and best['order']['meals'] > 0 and best['profit'] > 0:
            base_meals = best['order']['meals']
            extra_limit = min(5, max(1, math.ceil(base_meals * .08)))
            tolerance = min(2., .03 * best['profit'])
            supported = []
            for option in inventory_options:
                extra = option['order']['meals'] - base_meals
                extra_sales = option['sales'] - best['sales']
                if (0 < extra <= extra_limit and extra_sales >= .60 * extra - 1e-9
                        and extra_sales * price >= .90 * extra * cost
                        and option['profit'] >= best['profit'] - tolerance
                        and option['score'] >= best['score'] - tolerance
                        and option['loss'] <= best['loss'] + 2.):
                    supported.append(option)
            if supported:
                best = max(supported, key=lambda r: r['order']['meals'])
                best = dict(best, inventory_cushion=best['order']['meals']-base_meals,
                            base_meals=base_meals)
        return best

    def location_restriction(self, results, counts, spots):
        if not self.small_crowds:
            return {'best': max(results, key=lambda r: r['score']), 'enabled': False}
        # User rule: the THREE SMALLEST CUSTOMER forecasts, not the lowest rival
        # occupancy. Prefer the bottom two; the third must clear a profit hurdle.
        ordered = sorted(spots, key=lambda s: (counts[s], s))
        eligible, preferred = ordered[:3], ordered[:2]
        allowed = [r for r in results if r['order']['spot'] in eligible]
        first_two = [r for r in allowed if r['order']['spot'] in preferred]
        best = max(first_two or allowed, key=lambda r: r['score'])
        third = [r for r in allowed if r['order']['spot'] not in preferred]
        hurdle = max(15., .15*max(0., best['profit']))
        if third:
            challenger = max(third, key=lambda r: r['score'])
            if challenger['profit'] > best['profit']+hurdle and challenger['score'] > best['score']:
                best = challenger
        return {'best': best, 'enabled': True, 'rank_by': 'forecast_customers',
                'eligible': eligible, 'preferred': preferred, 'third_profit_hurdle': hurdle,
                'price_caps': dict(self.low_price_caps)}

    def decide(self, state):
        started = time.perf_counter()
        state = without_null_counts(state)  # 57181513
        self.observe(state)
        active_features = features(state.get('features'))
        self.learner.activate(active_features)
        if state.get('order_for_day') is None:
            next_day = (state.get('forecast') or {}).get('day') or int(state.get('day', 0))+1
            state = dict(state, order_for_day=next_day)
        rules = Rules.read(state)
        counts = self.forecast(state, rules)
        # Walking couples all campus crowds, but only eligible destinations can
        # become our order. Avoid expensive candidate searches that policy forbids.
        search_spots = rules.spots
        if self.small_crowds and active_features['walk_over']:
            search_spots = tuple(sorted(rules.spots, key=lambda s: (counts[s], s))[:3])
        digest = hashlib.sha256(repr((self.seed, self.market, state.get('order_for_day'))).encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        self.scenario_rivals = []
        scenarios = self.scenarios(state, rules, counts, rng)
        results, coverage = [], {}
        by_pair = {}
        for spot in search_spots:
            rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
            coverage[spot] = []
            for quality in QUALITIES:
                coverage[spot].append(quality)
                ceiling = min(rules.price_max, self.price_cap or rules.price_max,
                              self.low_price_caps[quality] if self.small_crowds else rules.price_max,
                              reputation_value(quality, rep, rules.values))
                low = max(rules.price_min, rules.costs[quality] + .25,
                          7.25 if self.small_crowds and quality == 'premium' else rules.price_min)
                # Coarse prices plus value-equivalent rival thresholds; prices above
                # rivals remain eligible to capture overflow and reputation advantages.
                prices = {round(low + .75*k, 2) for k in range(int(max(0, ceiling-low) / .75)+1)}
                prices.add(round(ceiling, 2))
                thresholds = []
                for placements, _, _, _ in scenarios:
                    for rp, rq, rr, _ in placements[spot]:
                        thresholds.append(reputation_value(quality, rep, rules.values)
                                          - reputation_value(rq, rr, rules.values) + rp - .01)
                # Quantiles bound computation even in a large tournament field.
                thresholds.sort()
                if thresholds:
                    prices.update(round(thresholds[int((len(thresholds)-1)*q)], 2) for q in (0., .25, .5, .75, 1.))
                candidates = [self.evaluate(state, rules, spot, quality, price, scenarios)
                              for price in sorted(prices) if low <= price <= ceiling]
                if candidates:
                    result = max(candidates, key=lambda r: r['score'])
                    by_pair[spot, quality] = result
                    results.extend(candidates)
        # Same refinement allowance for each spot, after complete coarse coverage.
        for spot in search_spots:
            local = [r for (s, _), r in by_pair.items() if s == spot]
            if not local:
                continue
            top = max(local, key=lambda r: r['score'])
            quality, base = top['order']['quality'], top['order']['price']
            rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
            ceiling = min(rules.price_max, self.price_cap or rules.price_max,
                              self.low_price_caps[quality] if self.small_crowds else rules.price_max,
                              reputation_value(quality, rep, rules.values))
            for step in (-.5, -.25, .25, .5):
                price = round(base + step, 2)
                floor = max(rules.price_min, rules.costs[quality]+.01,
                            7.25 if self.small_crowds and quality == 'premium' else rules.price_min)
                if floor <= price <= ceiling:
                    results.append(self.evaluate(state, rules, spot, quality, price, scenarios))
        if not results:
            raise ValueError('No legal candidates for supplied rules')
        restriction = self.location_restriction(results, counts, rules.spots)
        best = restriction['best']
        order = rules.validate(best['order'])
        self.pending = {'day': state.get('order_for_day'), 'forecast': state.get('forecast') or {},
                        'features': active_features,
                        'prediction': {'order': dict(order), 'sales': best['sales'],
                                       'features': active_features,
                                       'demand': best['demand'], 'profit': best['profit'],
                                       'rivals': sum(len(p[order['spot']]) for p, _, _, _ in scenarios)/len(scenarios)}}
        self.last_order = order
        ranked = []
        for spot in rules.spots:
            local = [r for r in results if r['order']['spot'] == spot]
            if local:
                r = max(local, key=lambda r: r['score'])
                ranked.append({**r, 'forecast': counts[spot]})
        self.diagnostics = {'elapsed': time.perf_counter()-started, 'coverage': coverage,
                            'candidates': len(results), 'ranking': sorted(ranked, key=lambda r: -r['score']),
                            'chosen': best, 'location_rule': {k: v for k, v in restriction.items() if k != 'best'},
                            'scenario_count': len(scenarios), 'prediction_checks': self.prediction_checks,
                            'features': active_features,
                            'rivals': {identity(t): self.opponents.summary(identity(t),
                                self.history.get(identity(t), []), state.get('forecast') or {},
                                self.history, rules.spots) for t in state.get('trucks') or []
                                if identity(t) != identity(state.get('you') or {})}}
        # forecast_log.capture (observability only, cannot change the order) omitted in the port.
        self.diagnostics['elapsed'] = time.perf_counter()-started
        return order

# ======================= history.py (verbatim) =======================
"""Replay saved public live observations in order; never submit historical orders."""
import json
from pathlib import Path


def warm_start(brain, root):
    directory = Path(root) / 'data' / 'live-observations'
    if not directory.exists():
        return 0
    states = [json.loads(path.read_text()) for path in directory.glob('*.json')]
    states.sort(key=lambda s: (str(s.get('mode')), str(s.get('market')), (s.get('last_day') or {}).get('day', 0)))
    loaded = 0
    for state in states:
        key = [state.get('mode'), state.get('market')]
        if brain.market is not None and brain.market != key:
            continue  # Never replace a final-market checkpoint with old live data.
        day = (state.get('last_day') or {}).get('day')
        if day is None or (brain.seen is not None and day <= brain.seen):
            continue
        brain.observe(state)
        forecast = state.get('forecast') or {}
        next_day = forecast.get('day') or state.get('order_for_day')
        # Day-phase snapshots also contain tomorrow's forecast, even while orders
        # are closed and order_for_day is null. Preserve it for the next result.
        if next_day is not None and next_day > day:
            brain.pending = {'day': next_day, 'forecast': forecast}
            if 'features' in state:
                brain.pending['features'] = state['features']
        loaded += 1
    return loaded

# ======================= bot.py fallback (verbatim) =======================
def fallback(state):
    """Bound loss if policy fails: assess all spots using forecast less rival stock."""
    state = without_null_counts(state)  # 57181513
    rules = Rules.read(state)
    forecast = (state.get('forecast') or {}).get('expected_customers') or {}
    last = (state.get('last_day') or {}).get('spots') or {}
    def opportunity(spot):
        rivals = (last.get(spot) or {}).get('trucks') or []
        stock = sum(t.get('meals', 0) for t in rivals if t.get('id') != state.get('you', {}).get('id'))
        return max(0., float(forecast.get(spot, 0)) - stock)
    bottom_two = sorted(rules.spots, key=lambda spot: (float(forecast.get(spot, 0)), spot))[:2]
    spot = max(bottom_two, key=opportunity)
    meals = min(30, rules.capacity('standard'), int(opportunity(spot) * .35))
    return rules.validate({'spot': spot, 'price': max(rules.price_min, min(rules.price_max, 7.0)),
                           'quality': 'standard', 'meals': max(0, meals)})

# ======================= model/prior.json (verbatim) =======================
PRIOR_JSON = r'''{"version":1,"weights":{"highst":[-15.716097405513716,-15.82300568308542,-16.0,-15.68,-15.68,-15.856099069192558,-15.68,-15.68,-15.715340906743943,-13.570780653663864,-14.90932766429571,-15.68,-8.331213302170623,-12.904039086005872,-15.68,-3.418775126867456,-12.309534890755446,-15.68,-15.68,-9.359099654256536,-0.07616757312171257,-14.367712352768036,0.0,-4.136935100251859,-3.146782968073255,-0.0903431845559753,-10.720083210896593,-15.68,-15.68,-13.311066570154585,-15.68,-15.247364585412162,-2.5588108965259027,-15.577882306067043,-4.352046346633972,-3.2966095298485385,-15.68,-15.68,-15.57175370480364,-15.68,-15.68,-13.559072709298738,-15.68,-15.516989407490609,-5.549828647093943],"oval":[-16.0,-14.92501496184862,-15.081601803469939,-16.0,-14.448836129127534,-15.732474044646406,-15.267250654393363,-13.828159745196144,-16.0,-16.0,-15.144247137795942,-14.91294982080811,-16.0,-9.574481261394599,-14.84563328933995,-16.0,-4.292818203017029,-14.83080223813381,-16.0,-5.64451763690759,-8.440917597023365,-16.0,0.0,-5.775428859065305,-16.0,-0.8393275757479551,-12.14166645649772,-16.0,-11.210832907353627,-11.463590933275766,-16.0,-12.777782906955219,-14.907862920520278,-16.0,-14.809760877498748,-15.053582553923206,-16.0,-15.181077829364567,-14.88827440274184,-16.0,-15.175067843436429,-14.89210210157052,-16.0,-15.07884242937525,-14.953930711803089],"union":[-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-4.298862434804317,-3.8508417788051457,-15.879732514071783,-0.11847353010235112,-11.565083896532665,-16.0,-3.1572944516257437,-15.969423127148747,-16.0,-15.781062999342693,-12.234726693741274,-1.9721742163720768,-13.061681207442223,0.0,-7.095485446373006,-1.5584867922808707,-0.32956001166248666,-16.0,-15.996817730820517,-16.0,-8.147384656908176,-15.772940975339836,-13.815480256093032,-3.8050359016020145,-15.174798901981303,-2.5740466073886474,-5.95479707669366,-16.0,-16.0,-11.66183821781964,-15.959155680923342,-16.0,-8.098608961225356,-15.752829441201353,-13.937180212129208,-5.139228288772005],"northres":[-15.736686293296634,-15.736686293296634,-15.736686293296634,-15.7341925994597,-15.7341925994597,-15.7341925994597,-15.69403399987713,-15.69403399987713,-15.69403399987713,-15.736625197130722,-15.736625197130722,-15.736625197130722,-7.71363235955467,-7.513924271840447,-11.86027416527504,-2.571781374076156,-4.031490914883694,-10.436024729759087,-6.7519153088719515,-3.0332943763409883,-1.769400454310105,-3.272537833582349,0.0,-0.4791867503495321,-2.8397071856106946,-1.4390041851918687,-4.261299231181433,-15.696637230840905,-14.84596971441801,-11.389809281362092,-16.0,-14.36725464167959,-11.92932685160532,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0],"thompson":[-15.815969862338097,-15.815969862338097,-15.815969862338097,-15.621548388103124,-15.621548388103124,-15.621548388103124,-15.45758538508982,-15.45758538508982,-15.45758538508982,-15.666801504233367,-15.666801504233367,-15.666801504233367,-15.453205834955597,-15.453205834955597,-15.453205834955597,-0.5944320705626251,-5.719041293534741,-12.020919594901876,-14.934952823838904,-15.446243586020568,-15.446243586020568,0.0,-4.7251101455550515,-11.12979513325659,-14.878967913656433,-15.192682167150915,-15.192682167150915,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0],"stadium":[-15.454305465763744,-14.851299536099882,-15.757316924638783,-9.932805246299697,-10.088661627203404,-13.232122068849577,-6.99309113164204,-8.126532985175388,-11.778163312987687,-6.0582918534448424,-4.365457513744483,-5.99465844246056,-3.294762719558404,-2.316008141780043,-4.575822809761454,-2.2880591249451805,-2.0504080046452406,-4.970333047437496,-3.8136615103462583,-1.4054372406313673,-2.233246334449242,-2.0617578012868023,-0.020255092592381743,-1.434244066361976,-1.2614705828136765,0.0,-2.11431359906325,-7.107537411971021,-4.721815685440683,-5.266243721160507,-5.199470817748324,-2.89727988420013,-3.6741702873443916,-3.82698790676987,-1.7441140263048251,-3.144308629407937,-15.96353314801916,-13.85065009060152,-14.400870650633399,-11.724180470755755,-9.375207512070899,-9.924063448107649,-8.469207795639612,-6.2725618359524065,-7.136875867133414]},"counts":{"highst":408,"oval":105,"union":420,"northres":132,"thompson":420,"stadium":85},"crowd":{"highst":{"cloudy":[0.9920529094548451,136],"clear":[1.0423013075445515,161],"rain":[0.9693263295076181,88],"cold":[0.9583906895273422,35]},"oval":{"cloudy":[0.9925754662902104,136],"clear":[1.103528542474442,161],"rain":[0.9718379258934416,88],"cold":[0.9357123646185133,35]},"union":{"cloudy":[1.0266563393481678,136],"clear":[1.0251139900648758,161],"rain":[0.9273537379157114,88],"cold":[0.9456967329737556,35]},"northres":{"cloudy":[1.105774281561361,136],"clear":[1.077607613087599,161],"rain":[0.8588201149247786,88],"cold":[1.0385397103193477,35]},"thompson":{"cloudy":[1.0888682622798378,136],"clear":[1.0512911320879488,161],"rain":[0.8635793393547397,88],"cold":[0.9511298766292379,35]},"stadium":{"cloudy":[0.9680603763738579,136],"clear":[1.0988598798169666,161],"rain":[0.999566182184343,88],"cold":[0.9947582038962325,35]}},"rows":420}'''


# ======================= wrapper: bot.make_brain (samples=32, small_crowds=True) + history.warm_start + Runner =======================
def make(seed=0, **opts):
    """seed is ignored (live Brain uses seed=42). opts: samples (default 32 since 4b2abab5), small_crowds (default True, as
    make_brain), use_prior (default True), warm_states (list of public live states to replay like
    history.warm_start before the first decision; validate.py instead warms via PERSISTENT_MEMORY)."""
    learner = Learner.from_dict(json.loads(PRIOR_JSON)) if opts.get('use_prior', True) else Learner()
    small = opts.get('small_crowds', True)
    # bee01637 make_brain: risk .08 with small_crowds, else .15
    brain = Brain(learner=learner, samples=opts.get('samples', 32), small_crowds=small,
                  risk=.08 if small else .15)
    states = list(opts.get('warm_states') or [])
    states.sort(key=lambda s: (str(s.get('mode')), str(s.get('market')), (s.get('last_day') or {}).get('day', 0)))
    for st in states:  # history.warm_start, minus the file reading
        key = [st.get('mode'), st.get('market')]
        if brain.market is not None and brain.market != key:
            continue
        day = (st.get('last_day') or {}).get('day')
        if day is None or (brain.seen is not None and day <= brain.seen):
            continue
        brain.observe(st)
        forecast = st.get('forecast') or {}
        next_day = forecast.get('day') or st.get('order_for_day')
        # 5c96779e: day-phase snapshots also carry tomorrow's forecast; keep it for the next result.
        if next_day is not None and next_day > day:
            brain.pending = {'day': next_day, 'forecast': forecast}
            if 'features' in st:  # 7439109b history.warm_start
                brain.pending['features'] = st['features']
    memo = {'key': None, 'order': None}

    def policy(state):
        # 7439109b Runner: re-decide when the key or the active feature flags change.
        key = [state.get('mode'), state.get('market'), state.get('order_for_day'), features(state.get('features'))]
        if key == memo['key'] and memo['order'] is not None:
            return dict(memo['order'])  # Runner reuses pending_order for the same key and flags
        try:
            order = brain.decide(state)
        except Exception:
            order = fallback(state)
        memo['key'], memo['order'] = key, dict(order)
        return dict(order)

    policy.brain = brain
    return policy
