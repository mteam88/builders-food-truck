"""LIVE + FINAL PORT for T.H.E Truck (t_d64e26): siyanshvir/TruckHomeEatsBot @ d3b4b8d0
(2026-10-24 05:30 ET "Add the final playbook (start to end)"). Previous port: policy_57181513.py.

`python3 bot.py` (no flags): make_brain(samples=32, small_crowds=True) -> Brain(Learner(prior.json), seed=42, samples=32,
risk=.08, small_crowds=True, spot_price_lift={'thompson': 1.39}) + attach_policy(brain) (FinalPolicy, seed 0, priors from
model/final_priors.json). Runner.apply_profile: in mode 'final'/'showcase' it calls brain.use_final(True)
(small_crowds False -> all spots, no low price caps, no premium floor; spot_price_lift {}), and on the first final state
FinalPolicy.reset_for_final() (fresh controller/competition/rule adapter). Live markets run the baseline Brain, with the
FinalPolicy only shadow-learning. Kill switch: NIGHTSHIFT_PROFILE=baseline or data/profile.json {"profile": "baseline"}
(here: make(profile={...}) and os.environ).

After an accepted order the Runner votes (politics.choose_vote, once per election, never overriding a cast vote), sets the
crew (CrewModel.plan, POST /staff when changed) and, only in final markets (complaints.enabled), may file a health complaint
(POST /x/8243598d/complaints {"target": id}). The port returns those as extra keys: vote, staff {staff, wage},
complaint (target id) and complaint_info. In final markets it adds 'food' when features.food is on (rules_adapter).
If the final policy says we are wrecked it returns meals 0 with 'wrecked': True (the Runner submits nothing).

Modules inlined verbatim, package imports stripped (built by a text-concatenation script, never importing their code):
game, learning, mechanics, vendor/pricewars_strategies, pricing_models, opponents, politics, overheads, riots, posture,
staffing, calibration, competition, controller, rules_adapter, complaints, final_mode, strategy, history; bot.fallback.
Renamed to avoid clashes: calibration FORGET/PRIOR/LOW/HIGH -> CAL_*, controller FORGET -> CTRL_FORGET,
complaints.enabled -> complaints_enabled, posture.* -> posture_* (exposed as posture_mod like strategy.py expects).
forecast_log.capture is replaced by a light version that keeps only what decisions use (scenario sales for the crew plan).
Dead code for decisions: dataset.py, sim.py, status.py, tools/, forecast_log logging, final_mode.make_final_brain.

Determinism: sha256 seeds, no hash(). The one wall-clock input is FinalPolicy.begin(): in final markets the scenario count
shrinks x0.7 (min 12) when the previous decision took > 0.9 s and grows +4 (max 32) when < 0.4 s. make(elapsed=x) pins it.
"""
import copy
import hashlib
import json
import math
import os
import random
import time
from types import SimpleNamespace
from pathlib import Path

DEFAULT_OPTS = {}  # (make() defaults baked in for archived CLI variants)
PROFILE = {}  # stands in for their gitignored data/profile.json (kill switches); empty = the shipped defaults


def _profile():
    return PROFILE

# ======================= game.py (verbatim; package imports stripped) =======================
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


# Weather factors for spots registered at run time (new spots); the original six use WEATHER above.
SPOT_WEATHER = {}
_REGISTERED = []
# Mild default per weather for new spots; the Mirror is outdoors, the Med Center is rain-proof.
_NEW_SPOT_WEATHER = {
    'mirror': {'clear': 1., 'cloudy': .95, 'rain': .5, 'cold': .8},
    'medcenter': {'clear': 1., 'cloudy': 1., 'rain': 1., 'cold': 1.},
}
_MILD_WEATHER = {'clear': 1., 'cloudy': .97, 'rain': .9, 'cold': .95}


def known_spots():
    """The original six plus any spots registered at run time, in registration order."""
    return SPOTS + tuple(_REGISTERED)


def register_spots(priors, distances, weather=None):
    """Add new spots to PRIORS/DISTANCES/SPOT_WEATHER in place; all-or-nothing, original data untouched.

    priors: {spot: (mu, sd, mix, weight)} or {spot: {'mu','sd','mix','weight'}}.
    distances: {(a, b): metres} or {a: {b: metres}}; pairs between two original spots are ignored.
    weather: optional {spot: {weather: factor}}; default is the built-in new-spot table, else mild.
    """
    new_priors, new_weather, new_dist = {}, {}, {}
    for spot, raw in dict(priors or {}).items():
        if spot in SPOTS or not isinstance(spot, str):
            continue
        if isinstance(raw, dict):
            raw = (raw['mu'], raw['sd'], raw['mix'], raw['weight'])
        mu, sd, mix, weight = raw
        mix = tuple(float(x) for x in mix)
        values = [float(mu), float(sd), float(weight)] + list(mix)
        if len(mix) != 3 or not all(math.isfinite(x) for x in values) or sd <= 0:
            raise ValueError('Invalid prior for %s' % spot)
        new_priors[spot] = (float(mu), float(sd), mix, float(weight))
        table = dict(_NEW_SPOT_WEATHER.get(spot, _MILD_WEATHER))
        table.update(dict(weather or {}).get(spot) or {})
        new_weather[spot] = {str(k): float(v) for k, v in table.items()}
    allowed = set(SPOTS) | set(_REGISTERED) | set(new_priors)
    pairs = []
    for key, value in dict(distances or {}).items():
        if isinstance(value, dict):
            pairs.extend(((key, other), d) for other, d in value.items())
        else:
            pairs.append((key, value))
    for (left, right), distance in pairs:
        distance = float(distance)
        if left == right or left not in allowed or right not in allowed or not math.isfinite(distance) or distance < 0:
            continue
        if left in SPOTS and right in SPOTS:
            continue
        new_dist.setdefault(left, {})[right] = distance
        new_dist.setdefault(right, {})[left] = distance
    PRIORS.update(new_priors)
    SPOT_WEATHER.update(new_weather)
    for spot in new_priors:
        if spot not in _REGISTERED:
            _REGISTERED.append(spot)
    for spot, row in new_dist.items():
        DISTANCES.setdefault(spot, {}).update(row)
    for spot in new_priors:
        DISTANCES.setdefault(spot, {})


def reset_spots():
    """Drop every registered spot (tests and kill-switch); the original six are restored untouched."""
    for spot in _REGISTERED:
        PRIORS.pop(spot, None)
        SPOT_WEATHER.pop(spot, None)
        DISTANCES.pop(spot, None)
    for row in DISTANCES.values():
        for spot in _REGISTERED:
            row.pop(spot, None)
    del _REGISTERED[:]


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
    for name, words in (('game', ('game', 'football', 'championship')), ('midterm', ('midterm', 'exam')),
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
    if spot in SPOTS:
        return WEATHER.get(weather, WEATHER['clear'])[SPOTS.index(spot)]
    table = SPOT_WEATHER.get(spot) or {}
    return table.get(weather, table.get('clear', 1.))


def number(value):
    """Practice data can be deliberately bad: prices arrive as text like '$10.00'. Parse or leave as is."""
    if isinstance(value, str):
        try:
            return float(value.replace('$', '').replace(',', '').strip())
        except ValueError:
            return value
    return value


def clean_record(record):
    """A truck or result record with its numeric fields parsed."""
    if not isinstance(record, dict):
        return record
    out = dict(record)
    for key in ('price', 'meals', 'sold', 'revenue', 'cost', 'profit', 'customers', 'hungry', 'walked_in', 'walked_out'):
        if isinstance(out.get(key), str):
            out[key] = number(out[key])
    return out


def spot_results(last):
    raw = last.get('spots') or {}
    if isinstance(raw, list):
        raw = {x.get('spot') or x.get('id'): x for x in raw}
    cleaned = {}
    for spot, info in raw.items():
        if isinstance(info, dict):
            info = dict(clean_record(info), trucks=[clean_record(t) for t in info.get('trucks') or []])
        cleaned[spot] = info
    return cleaned


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
                   dict(r.get('values') or VALUES),
                   tuple(s for s in (r.get('spots') or SPOTS) if s in SPOTS or s in _REGISTERED),
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


# ======================= learning.py (verbatim; package imports stripped) =======================
"""Online Bayesian model averaging, shared by training and the playing policy."""
import copy
import json
import math
from pathlib import Path
# inlined: from game import (SPOTS, known_spots, QUALITIES, REP_CHANGE, PRIORS, Rules, clamp, customer_model, events, expected_sales, identity, spot_results, weather_factor, weather_probs)

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

    def _ensure(self, spot):
        """Lazily create learner state for a spot added after construction (new spots)."""
        if spot not in self.weights:
            self.weights[spot] = [0.] * len(PARTICLES)
        self.counts.setdefault(spot, 0)
        self.crowd.setdefault(spot, {})
        self.crowd_variance.setdefault(spot, {})

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
            for weights in bank['weights'].values():
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
        self._ensure(spot)
        raw = self.weights[spot]
        maximum = max(raw)
        weights = [math.exp(x - maximum) for x in raw]
        total = sum(weights)
        return [x / total for x in weights]

    def sample(self, spot, rng):
        return rng.choices(PARTICLES, weights=self.posterior(spot), k=1)[0]

    def confidence(self, spot):
        count = self.counts.get(spot, 0)
        return count / (count + 20.)

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
            if spot not in SPOTS and spot not in known_spots():
                continue
            self._ensure(spot)
            customers = max(0., float(info.get('customers', 0)))
            records = info.get('trucks') or []
            trucks = [(float(t['price']), t['quality'],
                       float(before_reps.get(identity(t), {}).get(spot, 50)), int(t['meals']))
                      for t in records if t.get('quality') in QUALITIES]
            records = [t for t in records if t.get('quality') in QUALITIES]
            # Under walk_over some sales are customers who walked in from other spots.
            # The API reports walked_in per spot; we remove it from the truck sales so
            # only local buyers are fitted. Allocation choice: walk-ins are spread over
            # the trucks here in proportion to their sales (a simple approximation; the
            # true split depends on stock and price). Locals walking out are already
            # non-buyers here, so they need no adjustment. If either walk field is
            # missing the split is unknowable and fitting is withheld, as before.
            # Origin crowd residuals below remain usable and segregated per regime.
            walk_known = (not walk_over) or (info.get('walked_in') is not None
                                             and info.get('walked_out') is not None)
            observed = [float(t.get('sold', 0)) for t in records]
            if walk_over and walk_known and observed:
                total = sum(observed)
                walked_in = clamp(float(info.get('walked_in') or 0), 0., total)
                if total > 0:
                    observed = [x * (1. - walked_in / total) for x in observed]
            if customers and trucks and walk_known:
                losses = []
                for particle in PARTICLES:
                    predictions = expected_sales(trucks, customers, customer_model(spot, kinds, particle),
                                                 rules.values, fair_choice=True) if fair_choice else expected_sales(
                        trucks, customers, customer_model(spot, kinds, particle), rules.values)
                    loss = 0.
                    for predicted, t, actual in zip(predictions, records, observed):
                        # A sellout is a lower bound, never evidence that demand stops there.
                        error = max(0., actual - predicted) if float(t.get('sold', 0)) >= t['meals'] else predicted - actual
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
        mean, count = self.crowd.get(spot, {}).get(weather, [1., 0])
        return 1. + (mean - 1.) * count / (count + 12.)

    def crowd_noise(self, spot, weather):
        """Relative standard deviation; shrink sparse observations toward old +/-15% noise."""
        count, mean, m2 = self.crowd_variance.get(spot, {}).get(weather, [0, 0., 0.])
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


# ======================= mechanics.py (verbatim; package imports stripped) =======================
"""Feature-aware projection and food accounting shared by planning and logging."""
# inlined: from game import expected_sales, expected_market_sales
# inlined: from learning import feature_tuple

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


# ======================= vendor/pricewars_strategies.py (verbatim; package imports stripped) =======================
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


# ======================= pricing_models.py (verbatim; package imports stripped) =======================
"""Adapt the vendored Price Wars reference merchants to Night Shift price forecasts."""
from types import SimpleNamespace
# inlined: from vendor.pricewars_strategies import CheapestStrategy, TwoBoundStrategy


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


# ======================= opponents.py (verbatim; package imports stripped) =======================
"""Online probabilistic rival forecasts, scored before observing each new result.

A mixture learns which behavior models explain each rival. Contextual historical
neighbors use announced events, forecast crowd shares, and weather probabilities.
Price Wars' documented undercut/cycle behaviors are pricing hypotheses, not rules
that our own order must obey. No future or private orders are used.
"""
import math
# inlined: from pricing_models import reference_prices
# inlined: from game import SPOTS, known_spots, QUALITIES, COSTS, VALUES, clamp, events, weather_probs

LOCATION_MODELS = ('uniform', 'habit', 'repeat', 'crowd', 'context', 'opportunity', 'follow')
PRICE_MODELS = ('context', 'repeat', 'undercut', 'cycle', 'follow')
# A new behavior starts well below the incumbents (log weight), and must earn its share.
NEW_EXPERT_LOG_WEIGHT = -2.
FOLLOW_DEPTH_PRIOR, FOLLOW_DEPTH_RANGE = 1., (-1., 6.)


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
    spots = SPOTS + tuple(s for s in known_spots()[len(SPOTS):] if s in counts)
    return {'events': events(forecast.get('events')), 'weather': weather_probs(forecast),
            'shares': {s: float(counts.get(s, 0))/total for s in spots}}


def similarity(left, right):
    le, re = set(left.get('events', [])), set(right.get('events', []))
    event_distance = len(le ^ re)
    extra = sorted((set(left.get('shares', {})) | set(right.get('shares', {}))) - set(SPOTS))
    crowd_distance = sum(abs(left.get('shares', {}).get(s, 0)-right.get('shares', {}).get(s, 0)) for s in SPOTS + tuple(extra))
    weather_distance = sum(abs(left.get('weather', {}).get(w, 0)-right.get('weather', {}).get(w, 0))
                           for w in ('clear', 'cloudy', 'rain', 'cold'))
    return math.exp(-1.5*event_distance-2.*crowd_distance-.5*weather_distance)


class OpponentModel:
    def __init__(self, data=None):
        self.data = data or {}
        self.ours = []  # our own recent records, oldest first: what a rival that follows us has seen

    def population(self):
        """Average follow weight and undercut depth over rivals with enough evidence: a new rival's prior."""
        weights, depths = [], []
        for e in self.data.values():
            if len(e.get('location', ())) == len(LOCATION_MODELS) and e.get('scored', 0) >= 3:
                weights.append(e['location'][-1])
            if e.get('follow_n', 0) > 0:
                depths.append(e['follow_depth'])
        return (sum(weights)/len(weights) if weights else NEW_EXPERT_LOG_WEIGHT,
                sum(depths)/len(depths) if depths else FOLLOW_DEPTH_PRIOR)

    def entry(self, name):
        start, depth = self.population() if name not in self.data else (NEW_EXPERT_LOG_WEIGHT, FOLLOW_DEPTH_PRIOR)
        e = self.data.setdefault(name, {'location': [0.]*(len(LOCATION_MODELS)-1) + [min(0., start)],
                                       'price': [0.]*(len(PRICE_MODELS)-1) + [min(0., start)],
                                       'active': 0., 'absent': 0., 'scored': 0,
                                       'log_loss': 0., 'repeat_log_loss': 0.,
                                       'price_count': 0, 'price_error': 0., 'repeat_price_error': 0.})
        # Checkpoints saved before the follow behavior existed keep their weights.
        for key, names in (('location', LOCATION_MODELS), ('price', PRICE_MODELS)):
            e[key] = list(e[key]) + [NEW_EXPERT_LOG_WEIGHT]*(len(names)-len(e[key]))
        e.setdefault('follow_depth', depth)
        e.setdefault('follow_n', 0)
        return e

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
        follow = [.1/len(spots)]*len(spots)  # mass where we have been recently, newest most
        for age, rec in enumerate(reversed(self.ours[-6:])):
            if rec.get('spot') in spots:
                follow[spots.index(rec['spot'])] += .6**age
        return [uniform, normalize(habit), normalize(repeat), crowd, normalize(nearby), opportunity,
                normalize(follow)]

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
        # Follow: undercut the price our truck charged last, by the depth this rival has shown.
        seen = [r for r in self.ours[-3:] if r.get('quality') == quality and (spot is None or r.get('spot') == spot)]
        follow = clamp(float(seen[-1]['price']) - self.entry(name)['follow_depth'], floor, upper) if seen else mean
        return [mean, repeat, undercut, cycle, follow], max(.2, min(2., math.sqrt(variance+.16))), stock

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
        if forecast and (forecast.get('expected_customers') or {}) and record['spot'] in spots:
            experts = self.location_experts(name, history, forecast, histories, spots)
            probabilities = self.placement(name, history, forecast, histories, spots)
            index = spots.index(record['spot'])
            e['log_loss'] -= math.log(max(probabilities[index], 1e-9))
            e['repeat_log_loss'] -= math.log(max(experts[2][index], 1e-9))
            e['scored'] += 1
            updated = [.98*w+.7*math.log(max(p[index], 1e-6)) for w, p in zip(e['location'], experts)]
            maximum = max(updated)
            e['location'] = [max(-12., x-maximum) for x in updated]
        if self.ours and record['spot'] == self.ours[-1].get('spot') and record['quality'] == self.ours[-1].get('quality'):
            low, high = FOLLOW_DEPTH_RANGE
            depth = clamp(float(self.ours[-1]['price']) - float(record['price']), low, high)
            e['follow_depth'] = depth if not e['follow_n'] else .5*e['follow_depth'] + .5*depth
            e['follow_n'] += 1
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


# ======================= politics.py (verbatim; package imports stripped) =======================
"""Election laws as the docs describe them, and a deliberately cautious vote."""
import random
# inlined: from game import COSTS

# Laws use the API's params when present, else the documented name. Crowd multipliers
# are scored for the vote only: whether expected_customers already includes a law in
# force is unverified, so planning leaves them to the learned crowd calibration.
# Game-day and meat/veggie laws are not modeled (food is not live).
PER_MEAL = 0.5
MONEY = {'flat fee': (-20., .1, 500.), 'tax top': (20., -.1, 500.)}  # flat, revenue share, threshold
CROWDS = {'curfew': {'highst': .75, 'northres': .75, 'oval': 1.1, 'thompson': 1.1, 'union': 1.1},
          'quiet core': {'oval': .85, 'thompson': .85, 'union': .85, 'highst': 1.2, 'northres': 1.2}}


def law_terms(policies):
    terms = {'floor': None, 'cap': None, 'per_meal': {}, 'money': None, 'crowd': {}, 'game_crowd': {}, 'riot': 1.}
    for policy in policies or []:
        name = str(policy.get('key') or str(policy.get('text', '')).split(':')[0]).lower().replace('_', ' ')
        params = policy.get('params') or {}
        # Any law may carry spot crowd multipliers, whatever it is called: read what the API says.
        if isinstance(params.get('crowd'), dict):
            terms['crowd'] = {s: float(m) for s, m in params['crowd'].items() if isinstance(m, (int, float))}
        if isinstance(params.get('game_crowd'), dict):
            terms['game_crowd'] = {s: float(m) for s, m in params['game_crowd'].items() if isinstance(m, (int, float))}
            terms['riot'] = float(params['riot']) if isinstance(params.get('riot'), (int, float)) else 1.
        if 'floor' in name:
            terms['floor'] = float(params.get('min_price', 7.5))
        elif 'price' in name and 'cap' in name:
            terms['cap'] = float(params.get('max_price', 12.))
        elif 'luxury' in name:
            terms['per_meal'] = {'premium': -PER_MEAL, 'cheap': PER_MEAL}
        elif 'junk' in name:
            terms['per_meal'] = {'cheap': -PER_MEAL, 'premium': PER_MEAL}
        elif 'top' in name or 'flat' in name:
            flat, share, above = MONEY['tax top' if 'top' in name else 'flat fee']
            terms['money'] = (float(params.get('flat', flat)), float(params.get('revenue_share', share)),
                              float(params.get('above', above)))
        elif ('curfew' in name or 'quiet' in name) and not terms['crowd']:
            terms['crowd'] = dict(CROWDS['curfew' if 'curfew' in name else 'quiet core'])
    return terms


def law_profit(terms, sales, price, quality):
    """Net effect of the laws on one day's profit, given the price customers pay."""
    net = sales * terms['per_meal'].get(quality, 0.)
    if terms['money']:
        flat, share, above = terms['money']
        net += flat + share * max(0., sales * price - above)
    return net


def charged_price(terms, price):
    if terms['floor'] is not None:
        price = max(price, terms['floor'])
    if terms['cap'] is not None:
        price = min(price, terms['cap'])
    return price


TERM_DAYS = 8
PROTEST_DAILY, PROTEST_SHARE = .30, .40  # chance of a protest a day; customers lost when targeted


def win_odds(ids, tally, unvoted, ours, samples=400, turnout=.7, seed=7):
    """Chance each candidate wins given our vote (None abstains); unvoted trucks vote at random.

    Ties are settled by a draw, as in the game."""
    rng = random.Random(seed)
    odds = dict.fromkeys(ids, 0.)
    for _ in range(samples):
        counts = {c: float(tally.get(c, 0)) for c in ids}
        for _n in range(unvoted):
            if rng.random() < turnout:
                counts[rng.choice(ids)] += 1
        if ours in counts:
            counts[ours] += 1
        top = max(counts.values())
        leaders = [c for c in ids if counts[c] == top]
        for c in leaders:
            odds[c] += 1. / len(leaders) / samples
    return odds


def vote_analysis(state, order, game_rate=.25, protest_rate=PROTEST_DAILY):
    """Per candidate: daily law value, win odds with and without our vote, protest cost, net gain."""
    election = (state.get('politics') or {}).get('election') or {}
    candidates = [c for c in election.get('candidates') or [] if isinstance(c, dict) and c.get('id')]
    if not candidates or not order or not order.get('meals'):
        return None
    ids = [c['id'] for c in candidates]
    you = state.get('you') or {}
    sales = .85 * order['meals']
    unit = order['price'] - .25 * COSTS[order['quality']]
    value = {}
    for candidate in candidates:
        terms = law_terms(candidate.get('policies'))
        price = charged_price(terms, order['price'])
        crowd = min(order['meals'], sales * terms['crowd'].get(order.get('spot'), 1.))
        daily = ((crowd - sales) * unit + crowd * (price - order['price'])
                 + law_profit(terms, crowd, price, order['quality']))
        game = terms['game_crowd']
        if game:  # on a game day our crowd changes; other days it does not
            on_game = min(order['meals'], crowd * game.get(order.get('spot'), game.get('other', 1.)))
            daily += game_rate * ((on_game - crowd) * unit
                                  + law_profit(terms, on_game, price, order['quality']) - law_profit(terms, crowd, price, order['quality']))
        value[candidate['id']] = daily
    tally = {c: int(n) for c, n in (election.get('tally') or {}).items()}
    own = you.get('vote') if you.get('vote') in ids else None
    if own:
        tally[own] = max(0, tally.get(own, 0) - 1)
    trucks = state.get('trucks') or []
    unvoted = max(0, len(trucks) - 1 - sum(tally.values())) if trucks else 0
    term = TERM_DAYS
    days, vote_day = state.get('days'), election.get('vote_day')
    if days and vote_day:
        term = max(0, min(TERM_DAYS, int(days) - int(vote_day)))
    abstain = win_odds(ids, tally, unvoted, None)
    cash = {t.get('id'): t.get('cash') or 0. for t in trucks}
    revenue = sales * order['price']
    rows = []
    for candidate in candidates:
        cid = candidate['id']
        voted = win_odds(ids, tally, unvoted, cid)
        gain = term * sum((voted[k] - abstain[k]) * value[k] for k in ids)
        voters = [t for t, v in (election.get('votes') or {}).items() if v == cid and t != you.get('id')]
        richest = all(cash.get(t, 0.) <= (you.get('cash') or 0.) for t in voters)
        target = min(1., (2. if richest else 1.) / (len(voters) + 1))  # protests favor the top earner
        protest = term * protest_rate * target * PROTEST_SHARE * revenue * voted[cid]
        rows.append({'id': cid, 'name': candidate.get('name'), 'daily_value': round(value[cid], 1),
                     'p_win_abstain': round(abstain[cid], 3), 'p_win_vote': round(voted[cid], 3),
                     'gain': round(gain, 1), 'protest_cost': round(protest, 1), 'net': round(gain - protest, 1)})
    margin = max(20., .05 * term * sales * unit * .5)
    return {'rows': rows, 'margin': round(margin, 1), 'term_days': term, 'unvoted': unvoted, 'own': own}


def choose_vote(state, order, game_rate=.25, protest_rate=PROTEST_DAILY):
    """Candidate id whose laws most help our usual order, or None to abstain."""
    analysis = vote_analysis(state, order, game_rate, protest_rate)
    if not analysis:
        return None
    best = max(analysis['rows'], key=lambda r: r['net'])
    return best['id'] if best['net'] > analysis['margin'] else None


# ======================= overheads.py (verbatim; package imports stripped) =======================
"""Daily costs and laws outside customer flow: wages, moving, and election laws."""
# inlined: from mechanics import food_profit
# inlined: from politics import law_terms, law_profit, charged_price


class Overheads:
    def __init__(self, state, spot, salvage):
        flags = state.get('features') or {}
        you = state.get('you') or {}
        self.salvage = salvage
        self.fixed = 0.
        self.wage, self.per, self.capacity = 0., 50, None
        # Coupons dropped on us and not yet used: each puts 20% of our buyers at half price (stacking to 60%).
        pending = sum(1 for c in (you.get('coupons') or []) if isinstance(c, dict) and c.get('applied_day') is None)
        self.coupon_factor = 1. - .5 * min(.6, .2 * pending)
        if flags.get('staff') and isinstance(you.get('staff'), dict):
            crew = you['staff']
            staff_rules = (state.get('rules') or {}).get('staff') or {}
            self.per = max(1, int(staff_rules.get('meals_per_employee') or 50))
            self.wage = float(crew.get('wage', staff_rules.get('default_wage', 0)) or 0)
            # Tomorrow's crew is set tonight, so today's crew is not a limit on tomorrow's order: wages follow
            # the meals cooked (one employee per 50) and only the game's headcount maximum bounds the order.
            self.capacity = int(staff_rules.get('staff_max') or 10) * self.per
        if flags.get('moving_costs') and you.get('last_spot') not in (None, spot):
            table = (state.get('rules') or {}).get('move_costs') or {}
            self.fixed += float((table.get(you['last_spot']) or {}).get(spot, 0))
        laws = (state.get('politics') or {}).get('laws') if flags.get('elections') else None
        self.terms = law_terms(laws)

    def charged(self, price):
        return charged_price(self.terms, price)

    def profit(self, sales, meals, price, quality, cost):
        """Profit after wages, moving and laws. Opening costs nothing extra with zero meals."""
        if not meals:
            return 0.
        charged = self.charged(price)
        wages = self.wage * max(1, -(-int(meals) // self.per))  # employees needed for these meals
        return (food_profit(sales, meals, charged * self.coupon_factor, cost, self.salvage)
                + law_profit(self.terms, sales, charged, quality) - self.fixed - wages)


# ======================= riots.py (verbatim; package imports stripped) =======================
"""Riot exposure from the documented formulas. Used only as a score cost, never as profit.

A Michigan loss starts a riot on High Street (intensity 0.7, radius 260 m); a national title starts
one around the Stadium (0.95, 380 m). A truck that opened d metres away is hit with probability
intensity * exp(-d / radius): wrecked 30% of the time (out the next 2 days), otherwise a repair
bill of 30 + 120 * exp(-d / radius). The forecast warns the night before, but its shape is not
documented, so warnings are read leniently from event text; with no warning the cost is zero.
"""
import math
# inlined: from game import DISTANCES
# inlined: from politics import law_terms

# (intensity, radius, centre spot, chance the game ends that way: Ohio State loses to Michigan 45%,
# wins the title 80%)
KINDS = {'michigan': (.7, 260., 'highst', .45), 'title': (.95, 380., 'stadium', .80)}
WRECK_SHARE = .3
OUT_DAYS = 2


def warnings(forecast):
    """Riot kinds the forecast mentions, as (kind, centre) pairs; the spot may be named in the entry."""
    found = []
    entries = list((forecast or {}).get('events') or [])
    for key in ('riot', 'riots'):
        extra = (forecast or {}).get(key)
        if extra:
            entries += extra if isinstance(extra, list) else [extra]
    for entry in entries:
        text = ' '.join(str(v) for v in entry.values()) if isinstance(entry, dict) else str(entry)
        text = text.lower()
        if 'riot' not in text:
            continue
        kind = 'title' if ('title' in text or 'champion' in text) else 'michigan'
        centre = entry.get('spot') if isinstance(entry, dict) and entry.get('spot') else KINDS[kind][2]
        found.append((kind, centre))
    return found


def law_multiplier(state):
    """How election laws in force scale riot fierceness: crackdown 0 (no riots), tailgate 1.5, else 1."""
    laws = (state.get('politics') or {}).get('laws')
    return law_terms(laws)['riot'] if laws else 1.


def riot_cost(state, spot, daily_profit=250.):
    """Expected dollars of riot cost for opening at this spot tomorrow."""
    if not (state.get('features') or {}).get('riots'):
        return 0.
    scale = law_multiplier(state)
    if scale <= 0:
        return 0.
    total = 0.
    for kind, centre in warnings(state.get('forecast')):
        intensity, radius, _, chance = KINDS[kind]
        distance = 0. if spot == centre else DISTANCES.get(centre, {}).get(spot)
        if distance is None:
            continue
        decay = math.exp(-distance / radius)
        hit = intensity * decay
        total += scale * chance * hit * (WRECK_SHARE * OUT_DAYS * daily_profit + (1 - WRECK_SHARE) * (30. + 120. * decay))
    return total


# ======================= posture.py (verbatim; package imports stripped, renames {'enabled': 'posture_enabled', 'read': 'posture_read', 'final_enabled': 'posture_final_enabled', 'WINDOW': 'POSTURE_WINDOW', 'FLOOR': 'POSTURE_FLOOR', 'MIN_DAYS': 'POSTURE_MIN_DAYS', 'NEUTRAL': 'POSTURE_NEUTRAL', 'RAMP': 'POSTURE_RAMP', 'MAX_UPSIDE': 'POSTURE_MAX_UPSIDE', 'MAX_EXPLORE': 'POSTURE_MAX_EXPLORE', '_rates': '_posture_rates'}) =======================
"""Self-adjusting posture: how much risk and exploration the planner takes, from how the field is doing.

Everything is posture_read from the API each night (every truck's cash). Nothing is set by hand:
- pressure: the fastest rival's recent daily earnings exceed ours. The further behind, the lower the
  weight on expected losses, the more weight on upside variance, and the more a quiet spot we do not
  usually use is worth trying.
- lead: we out-earn every rival. Then risk weight rises and exploration falls, so we protect a home
  spot where our reputation already pays.
All effects are bounded, scale in smoothly as evidence accumulates, and switch off with
NIGHTSHIFT_POSTURE=off or data/profile.json {"posture": "off"}.
"""
POSTURE_WINDOW = 7            # days of cash history used for recent daily earnings
POSTURE_MIN_DAYS = 4          # below this much history the posture stays neutral
POSTURE_FLOOR = 60.           # dollars/day: scale below which gaps are noise
POSTURE_MAX_UPSIDE = .5       # weight on E[max(0, profit - mean)] at full pressure
POSTURE_MAX_EXPLORE = 40.     # dollars of bonus for a quiet, rarely used spot at full pressure
POSTURE_NEUTRAL = {'pressure': 0., 'lead': 0., 'risk_scale': 1., 'upside': 0., 'explore': 0.}
POSTURE_RAMP = 35.            # reputation points above the starting 50 at which a market counts as established


def maturity(state):
    """0 right after a reset (reputation 50 everywhere), 1 once a home spot is established (85+).

    Reputation compounds about 5 points a day at the spot we keep selling at, so early in a market the
    right move is to stay and build it; roaming and upside-seeking only pay once a home exists."""
    try:
        reps = [float(v) for v in ((state.get('you') or {}).get('reputation') or {}).values()]
        return max(0., min(1., (max(reps) - 50.) / POSTURE_RAMP)) if reps else 1.   # unknown: do not hold aggression back
    except (TypeError, ValueError):
        return 1.


def _posture_rates(history, day):
    """Recent daily earnings per truck from {day: {truck: cash}}."""
    days = sorted(d for d in history if isinstance(d, int) and d <= day)
    if len(days) < 2:
        return {}, 0
    last = days[-1]
    base = next((d for d in days if d >= last - POSTURE_WINDOW), days[0])
    span = max(1, last - base)
    start, end = history[base], history[last]
    return {t: (end[t] - start[t]) / span for t in end if t in start}, span


def posture_enabled(root=None):
    """Kill switch: NIGHTSHIFT_POSTURE=off or data/profile.json {"posture": "off"}."""
    import json, os
    from pathlib import Path
    if os.environ.get('NIGHTSHIFT_POSTURE') == 'off':
        return False
    try:
        path = None
        return _profile().get('posture') != 'off'
    except (OSError, ValueError, AttributeError):
        return True


def posture_final_enabled(root=None):
    """The final scores average profit, and offline tests showed risk-seeking lowers it in tough markets,
    so the posture stays neutral in final markets unless data/profile.json has {"posture_final": "on"}."""
    import json, os
    from pathlib import Path
    if os.environ.get('NIGHTSHIFT_POSTURE') == 'final':
        return True
    try:
        path = None
        return _profile().get('posture_final') == 'on'
    except (OSError, ValueError, AttributeError):
        return False


def posture_read(state, history, me):
    """Posture for tonight. `history` maps day -> {truck id: cash}, newest included."""
    if state.get('mode') in ('final', 'showcase') and not posture_final_enabled():
        return dict(POSTURE_NEUTRAL)
    day = int(state.get('day') or 0)
    rates, span = _posture_rates(history, day)
    if span < POSTURE_MIN_DAYS - 1 or me not in rates or len(rates) < 2:
        return dict(POSTURE_NEUTRAL)
    ours = rates[me]
    best = max(r for t, r in rates.items() if t != me)
    trust = min(1., span / float(POSTURE_WINDOW))                    # fewer days, less weight
    scale = max(abs(best), abs(ours), POSTURE_FLOOR)
    pressure = trust * max(0., min(1., (best - ours) / scale))
    if state.get('mode') not in ('final', 'showcase'):
        # One shared market to the end (the live game): to overtake the cash leader we must out-earn
        # their pace by gap / days_left. In the final each market is scored on its own average profit,
        # so only the pace comparison above applies there.
        cash = history.get(max(d for d in history if d <= day), {})
        leader = max((t for t in cash if t != me), key=lambda t: cash[t], default=None)
        left = int(state.get('days') or 0) - day
        if leader and left > 0 and cash.get(me) is not None and cash[leader] > cash[me]:
            needed = rates.get(leader, 0.) + (cash[leader] - cash[me]) / left
            pressure = max(pressure, trust * max(0., min(1., (needed - ours) / max(needed, POSTURE_FLOOR))))
    lead = trust * max(0., min(1., (ours - best) / scale))
    grown = maturity(state)   # aggression waits until a home spot is established
    return {'pressure': pressure, 'lead': lead,
            'risk_scale': max(.4, 1. - .6 * pressure * grown + .5 * lead),
            'upside': POSTURE_MAX_UPSIDE * pressure * grown,
            'explore': POSTURE_MAX_EXPLORE * pressure * (1. - lead) * grown}


def note_cash(history, state):
    """Record every truck's cash for today (day -> {id: cash}); keep a short window."""
    try:
        day = int(state.get('day'))
        cash = {t['id']: float(t['cash']) for t in state.get('trucks') or [] if t.get('id') and t.get('cash') is not None}
        mine = state.get('you') or {}
        if mine.get('id') and mine.get('cash') is not None:
            cash[mine['id']] = float(mine['cash'])
        if cash:
            history[day] = cash
            for old in sorted(history)[:-(POSTURE_WINDOW + 3)]:
                del history[old]
    except (TypeError, ValueError, KeyError):
        pass
    return history


posture_mod = SimpleNamespace(NEUTRAL=POSTURE_NEUTRAL, read=posture_read, enabled=posture_enabled,
                              note_cash=note_cash, maturity=maturity)


# ======================= staffing.py (verbatim; package imports stripped) =======================
"""Crew size and wage planning from the public labor market.

Mechanics (docs): each employee serves 50 meals and is paid the daily wage on days
we open. Each morning employees asking more than our wage quit, short rivals paying
at least max($3, 10%) more may lure employees (40% move), then we hire unemployed
workers who accept our wage. A new headcount or wage applies from the next hiring.

Learned online and persisted in the checkpoint:
- the unemployed pool's asking-wage curve, from labor.median/lowest_asking_wage;
- how many of our employees rivals actually poach per exposed rival-day;
- predicted versus realized crew fill, kept as checks.
Retention when lowering wages relies on a prior over employed asking wages, because
the pool of unemployed is selected for high asks and cannot describe our own crew.
"""
import math
from statistics import NormalDist

WAGE_MIN, WAGE_MAX, PER_EMPLOYEE = 5., 60., 50
POACH_PRIOR = (.5, 10.)  # events, exposed rival-days
EMPLOYED_ASK = NormalDist(5.6, 1.0)


def _binomial(n, p):
    return [math.comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1)]


class CrewModel:
    def __init__(self, data=None):
        data = data or {}
        self.pool = data.get('pool', {})
        self.poach = data.get('poach', {'events': POACH_PRIOR[0], 'exposures': POACH_PRIOR[1]})
        self.pending = data.get('pending')
        self.checks = data.get('checks', [])

    def to_dict(self):
        return {'pool': self.pool, 'poach': self.poach, 'pending': self.pending, 'checks': self.checks[-48:]}

    # --- learning -------------------------------------------------------
    def see_labor(self, state):
        """Fold in the pool stats once per day; planning also calls this, so a restart is not blind."""
        labor = state.get('labor')
        day = state.get('order_for_day') or state.get('day')
        if isinstance(labor, dict) and labor.get('median_asking_wage') is not None and self.pool.get('day') != day:
            old = self.pool.get('median')
            median = float(labor['median_asking_wage'])
            self.pool = {'median': median if old is None else .6*old + .4*median, 'day': day,
                         'lowest': float(labor.get('lowest_asking_wage', median)),
                         'unemployed': int(labor.get('unemployed', 0))}

    def observe(self, state, me):
        last = state.get('last_day') or {}
        self.see_labor(state)
        pending = self.pending
        if not pending or last.get('day') != pending['day']:
            return
        events = sum(int(e.get('count', 0)) for e in last.get('labor') or [] if e.get('from') == me)
        self.poach['events'] += events
        self.poach['exposures'] += pending['exposure']
        employees = (last.get('you') or {}).get('employees')
        if employees is not None:
            self.checks = (self.checks + [{
                'day': pending['day'], 'target': pending['target'], 'wage': pending['wage'],
                'predicted_employees': pending['expected_employees'], 'actual_employees': employees,
                'poached': events}])[-48:]
        self.pending = None

    # --- prediction -----------------------------------------------------
    def pool_accepting(self, wage, unemployed):
        """Unemployed workers expected to take this wage (normal curve through median/lowest)."""
        median = self.pool.get('median', 5.5)
        n = max(2, self.pool.get('unemployed', unemployed))
        spread = max(.3, (median - self.pool.get('lowest', median - 1.)) / NormalDist().inv_cdf(1 - .5/n))
        return unemployed * NormalDist(median, spread).cdf(wage)

    def exposure(self, state, wage, me):
        count = 0
        for truck in state.get('trucks') or []:
            other = truck.get('wage')
            if truck.get('id') != me and other is not None and wage <= float(other) - max(3., .1*float(other)):
                count += 1
        return count

    def slot_probability(self, state, headcount, wage, current_wage, me):
        high = EMPLOYED_ASK.cdf(max(current_wage, wage))
        low = EMPLOYED_ASK.cdf(WAGE_MIN - 1.)
        keep = 1. if wage >= current_wage else max(0., (EMPLOYED_ASK.cdf(wage) - low) / max(high - low, 1e-6))
        rate = self.poach['events'] / self.poach['exposures']
        poached = min(.9, rate * self.exposure(state, wage, me) / max(1, headcount))
        retained = keep * (1. - poached)
        labor = state.get('labor') or {}
        trucks = max(1, len(state.get('trucks') or []))
        refill = min(.97, self.pool_accepting(wage, int(labor.get('unemployed', 0))) / trucks)
        return retained + (1. - retained) * refill

    def search_space(self, state, current_wage, needed):
        """Headcounts and wages to consider, from the rules and what the market shows (no fixed ceiling).

        Wages run from the game's minimum to a dollar above the highest wage anyone pays (or the pool
        would ask), plus our own; headcounts run from what the order needs up to four spares. The
        expected-cost model decides inside this space, so spare staff and a higher wage are chosen only
        when the learned quit/poach risk pays for them."""
        rules = ((state.get('rules') or {}).get('staff')) or {}
        low = float(rules.get('wage_min', WAGE_MIN))
        cap = float(rules.get('wage_max', WAGE_MAX))
        seen = [float(t['wage']) for t in state.get('trucks') or [] if isinstance(t.get('wage'), (int, float))]
        spread = max(.3, (self.pool.get('median', 5.5) - self.pool.get('lowest', 4.5)) / 1.9) if self.pool else 1.
        pool_high = self.pool.get('median', 0.) + 3 * spread if self.pool else 0.
        high = min(cap, max([current_wage, low + 2.] + seen + [pool_high]) + 1.)
        grid = {round(low + .5 * i, 2) for i in range(int((high - low) / .5) + 1)} | {round(current_wage, 2)}
        grid |= {round(round(w * 2) / 2, 2) for w in seen if low <= w <= cap}
        staff_max = int(rules.get('staff_max') or 10)
        return range(needed, min(staff_max, needed + 4) + 1), sorted(w for w in grid if low <= w <= cap)

    def plan(self, state, order, sales, me):
        """Best (headcount, wage) for the accepted order, given scenario sales."""
        self.see_labor(state)
        crew = (state.get('you') or {}).get('staff') or {}
        current_h, current_w = int(crew.get('target', 4)), float(crew.get('wage', 6))
        meals = int(order['meals'])
        needed = max(1, math.ceil(meals / PER_EMPLOYEE))
        headcounts, wages = self.search_space(state, current_w, needed)
        margin = float(order['price']) * .98
        sales = sales or [meals * .8]

        def value(headcount, wage):
            p = self.slot_probability(state, headcount, wage, current_w, me)
            pmf = _binomial(headcount, p)
            wages = sum(q * k for k, q in enumerate(pmf)) * wage
            lost = sum(q * sum(max(0., min(s, meals) - PER_EMPLOYEE*k) for s in sales) / len(sales) * margin
                       for k, q in enumerate(pmf))
            return -wages - lost, p, pmf
        best = None
        for headcount in headcounts:
            for wage in wages:
                score, p, pmf = value(headcount, wage)
                if best is None or score > best[0]:
                    best = (score, headcount, wage, p, pmf)
        stay = value(current_h, current_w)[0] if current_h >= 1 else float('-inf')
        if best[0] < stay + 1.:  # hysteresis: only change for a clear gain
            score, headcount, wage = stay, current_h, current_w
            p, pmf = value(headcount, wage)[1:]
        else:
            score, headcount, wage, p, pmf = best
        enough = sum(q for k, q in enumerate(pmf) if PER_EMPLOYEE*k >= meals)
        expected = sum(q * k for k, q in enumerate(pmf))
        exposure = self.exposure(state, wage, me)
        self.pending = {'day': state.get('order_for_day'), 'target': headcount, 'wage': wage,
                        'exposure': exposure, 'expected_employees': expected}
        return {'headcount': headcount, 'wage': wage, 'change': (headcount, wage) != (current_h, current_w),
                'current': (current_h, current_w), 'needed': needed, 'prob_enough_staff': enough,
                'expected_employees': expected, 'expected_daily_wages': expected * wage,
                'poach_rate_per_exposed_rival': self.poach['events'] / self.poach['exposures'],
                'rivals_exposed': exposure, 'pool_median_ask': self.pool.get('median'),
                'unemployed': (state.get('labor') or {}).get('unemployed')}


# ======================= calibration.py (verbatim; package imports stripped, renames {'FORGET': 'CAL_FORGET', 'PRIOR': 'CAL_PRIOR', 'LOW': 'CAL_LOW', 'HIGH': 'CAL_HIGH'}) =======================
"""Online correction of the planner's uncapped demand by price level.

Each observation gives target = actual_sales / expected_demand in a cell
(quality, price/value_ceiling bin). A cell shrinks toward the pooled estimate
of cells at the same or a lower price ratio (same quality, then all qualities),
so misses at high prices never drag down low-price factors.
"""
import math

EDGES = (.55, .7, .85)       # ratio bins: <=.55, .55-.7, .7-.85, >.85
QUALITIES = ('cheap', 'standard', 'premium')
CAL_FORGET = .93                 # per-observation decay of old evidence
CAL_PRIOR = 4.                   # pseudo-observations of weight at the parent value
CAL_LOW, CAL_HIGH = .3, 1.15


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _bin(ratio):
    return sum(ratio > e for e in EDGES)


class Calibrator:
    def __init__(self, data=None):
        self.cells = {}   # 'quality|bin' -> [weighted target sum, weight]
        self.count = 0
        try:
            for key, (s, w) in ((data or {}).get('cells') or {}).items():
                q, b = key.split('|')
                s, w = _num(s), _num(w)
                if q in QUALITIES and 0 <= int(b) <= len(EDGES) and s is not None and w is not None \
                        and s >= 0 and w >= 0:
                    self.cells[key] = [s, w]
            self.count = max(0, int((data or {}).get('count', 0)))
        except Exception:
            self.cells, self.count = {}, 0

    def to_dict(self):
        return {'cells': {k: list(v) for k, v in self.cells.items()}, 'count': self.count}

    def _pooled(self, qualities, upto):
        """Estimate from cells with bin <= upto, shrunk toward 1.0 (global) or its parent."""
        s = w = 0.
        for q in qualities:
            for b in range(upto + 1):
                cs, cw = self.cells.get('%s|%d' % (q, b), (0., 0.))
                s, w = s + cs, w + cw
        return s, w

    def _estimate(self, quality, b):
        gs, gw = self._pooled(QUALITIES, b)
        glob = (CAL_PRIOR + gs) / (CAL_PRIOR + gw)
        qs, qw = self._pooled((quality,), b)
        qual = (CAL_PRIOR * glob + qs) / (CAL_PRIOR + qw)
        cs, cw = self.cells.get('%s|%d' % (quality, b), (0., 0.))
        return (CAL_PRIOR * qual + cs) / (CAL_PRIOR + cw)

    def observe(self, check):
        try:
            quality = check['quality']
            price, ceiling = _num(check['price']), _num(check['value_ceiling'])
            expected, actual = _num(check['expected_demand']), _num(check['actual_sales'])
            meals = _num(check['meals'])
            if quality not in QUALITIES or None in (price, ceiling, expected, actual, meals):
                return
            if min(price, expected, actual, meals) < 0 or ceiling <= 0 or expected <= 0:
                return
            b = _bin(price / ceiling)
            target = min(actual / expected, 1.5)
            sold_out = bool(check.get('sellout')) or (meals > 0 and actual >= meals)
            # Decay first, then a sellout (lower bound) may only raise the estimate.
            for cell in self.cells.values():
                cell[0] *= CAL_FORGET
                cell[1] *= CAL_FORGET
            if sold_out and target <= self._estimate(quality, b):
                return
            cell = self.cells.setdefault('%s|%d' % (quality, b), [0., 0.])
            cell[0] += target
            cell[1] += 1.
            self.count += 1
        except Exception:
            return

    def factor(self, spot, quality, price, value_ceiling, weather=None):
        try:
            price, ceiling = _num(price), _num(value_ceiling)
            if quality not in QUALITIES or price is None or ceiling is None or ceiling <= 0 or price < 0:
                return 1.
            return min(CAL_HIGH, max(CAL_LOW, self._estimate(quality, _bin(price / ceiling))))
        except Exception:
            return 1.

    def report(self):
        out = {'observations': self.count, 'buckets': {}}
        for q in QUALITIES:
            for b in range(len(EDGES) + 1):
                cs, cw = self.cells.get('%s|%d' % (q, b), (0., 0.))
                out['buckets']['%s/r%d' % (q, b)] = {
                    'weight': round(cw, 2), 'factor': round(min(CAL_HIGH, max(CAL_LOW, self._estimate(q, b))), 3)}
        return out


# ======================= competition.py (verbatim; package imports stripped) =======================
"""Learned per-spot adjustment replacing a hard "smallest crowds" location cap.

penalty(spot, state) returns dollars/day to subtract from a spot's score (negative = bonus):
- shortfall: EMA of (our realized profit - predicted profit) at the spot, shrunk to 0 by a
  pseudo-count; usually underperforming the plan there is a cost;
- pressure: recent rival count, how often trucks showed up at a spot right after we left it
  (follow behaviour) and how deep they undercut our price;
- exploration bonus: spots not visited in this market while reputation is still ~50.
Everything is bounded and 0 with no data (except the small bonus for unvisited spots).
"""
import math

LIMIT = 60.
SHRINK = 2.            # pseudo-count for the shortfall EMA
ALPHA = .3             # EMA rate
SHORT_MAX, SHORT_MIN = 40., -15.
PRESSURE_MAX = 25.
EXPLORE_MAX = 12.
REP_BAND = 1.5
TOP_RIVALS = 1.


def _num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _dict(value):
    return value if isinstance(value, dict) else {}


def _ema(old, new, count):
    rate = max(ALPHA, 1. / (count + 1))
    return new if old is None else old + rate * (new - old)


class Competition:
    def __init__(self, data=None):
        data = _dict(data)
        self.spots = {}
        for spot, row in _dict(data.get('spots')).items():
            row = _dict(row)
            self.spots[str(spot)] = {
                'gap': _num(row.get('gap'), 0.), 'n': max(0, int(_num(row.get('n'), 0))),
                'rivals': _num(row.get('rivals'), 0.), 'rn': max(0, int(_num(row.get('rn'), 0))),
                'follow': _num(row.get('follow'), 0.), 'fn': max(0, int(_num(row.get('fn'), 0))),
                'undercut': _num(row.get('undercut'), 0.),
                'visits': max(0, int(_num(row.get('visits'), 0)))}
        self.prev_loc = {str(k): str(v) for k, v in _dict(data.get('prev_loc')).items()}
        self.last_spot = data.get('last_spot') if isinstance(data.get('last_spot'), str) else None
        self.last_price = _num(data.get('last_price'))

    def to_dict(self):
        return {'spots': {spot: dict(row) for spot, row in self.spots.items()},
                'prev_loc': dict(self.prev_loc), 'last_spot': self.last_spot,
                'last_price': self.last_price}

    def row(self, spot):
        return self.spots.setdefault(spot, {'gap': 0., 'n': 0, 'rivals': 0., 'rn': 0, 'follow': 0.,
                                            'fn': 0, 'undercut': 0., 'visits': 0})

    def new_market(self):
        """Visits and positional memory are per market; learned gaps and pressure carry (halved weight)."""
        for row in self.spots.values():
            row['visits'] = 0
            row['n'] = row['n'] // 2
            row['rn'] = row['rn'] // 2
            row['fn'] = row['fn'] // 2
        self.prev_loc = {}
        self.last_spot = None
        self.last_price = None

    # --- learning ---------------------------------------------------------
    @staticmethod
    def _trucks(day, spot):
        trucks = _dict(_dict(_dict(day).get('spots')).get(spot)).get('trucks')
        return [t for t in trucks if isinstance(t, dict)] if isinstance(trucks, list) else []

    def observe(self, last_day, our_order, predicted):
        try:
            self._observe(_dict(last_day), _dict(our_order), _dict(predicted))
        except (TypeError, ValueError, KeyError, AttributeError):
            pass

    def _observe(self, last_day, order, predicted):
        spot = order.get('spot')
        spots = _dict(last_day.get('spots'))
        if not isinstance(spot, str) or spot not in spots:
            return
        you = _dict(last_day.get('you'))
        my_id = you.get('id', order.get('id'))
        price = _num(order.get('price'))
        row = self.row(spot)
        row['visits'] += 1

        def ours(truck):
            if my_id is not None and truck.get('id') == my_id:
                return True
            return my_id is None and price is not None and _num(truck.get('price')) == price and \
                _num(truck.get('meals')) == _num(order.get('meals'))

        # (a) shortfall versus the planner's own number
        realized = _num(you.get('profit'), _num(last_day.get('profit')))
        expected = _num(predicted.get('profit'))
        if realized is not None and expected is not None:
            row['gap'] = _ema(row['gap'] if row['n'] else None, max(-300., min(300., realized - expected)), row['n'])
            row['n'] += 1

        # (b) rival pressure where we sold, and follow behaviour where we left
        rivals_here = [t for t in self._trucks(last_day, spot) if not ours(t)]
        row['rivals'] = _ema(row['rivals'] if row['rn'] else None, float(len(rivals_here)), row['rn'])
        row['rn'] += 1
        if price is not None:
            gaps = [price - p for p in (_num(t.get('price')) for t in rivals_here) if p is not None]
            if gaps:
                row['undercut'] = _ema(row['undercut'], max(0., max(gaps)), row['n'])
        left = self.last_spot
        if left and left != spot and left in spots:
            arrivals = [t for t in self._trucks(last_day, left)
                        if not ours(t) and t.get('id') is not None and self.prev_loc.get(str(t['id'])) not in (None, left)]
            lrow = self.row(left)
            lrow['follow'] = _ema(lrow['follow'] if lrow['fn'] else None, min(3., float(len(arrivals))), lrow['fn'])
            lrow['fn'] += 1
            last_price = self.last_price
            if last_price is not None:
                gaps = [last_price - p for p in (_num(t.get('price')) for t in arrivals) if p is not None]
                if gaps:
                    lrow['undercut'] = _ema(lrow['undercut'], max(0., max(gaps)), lrow['fn'])
        loc = {}
        for name in spots:
            for truck in self._trucks(last_day, name):
                if truck.get('id') is not None and not ours(truck):
                    loc[str(truck['id'])] = name
        self.prev_loc = loc
        self.last_spot = spot
        self.last_price = price

    # --- scoring ----------------------------------------------------------
    def penalty(self, spot, state=None):
        try:
            return self._penalty(spot, _dict(state))
        except (TypeError, ValueError, KeyError, AttributeError):
            return 0.

    def _penalty(self, spot, state):
        row = self.spots.get(spot)
        total = 0.
        if row:
            shortfall = -row['gap'] * row['n'] / (row['n'] + SHRINK)
            total += max(SHORT_MIN, min(SHORT_MAX, shortfall))
            rival_conf = row['rn'] / (row['rn'] + SHRINK)
            follow_conf = row['fn'] / (row['fn'] + SHRINK)
            pressure = 2. * max(0., row['rivals'] - TOP_RIVALS) * rival_conf \
                + 10. * row['follow'] * follow_conf \
                + 3. * min(3., row['undercut']) * max(rival_conf, follow_conf)
            total += min(PRESSURE_MAX, pressure)
        visits = row['visits'] if row else 0
        rep = _num(_dict(_dict(state.get('you')).get('reputation')).get(spot), 50.)
        if abs(rep - 50.) <= REP_BAND:
            day = _num(state.get('order_for_day'), _num(state.get('day'), 1.))
            days = max(1., _num(state.get('days'), 30.))
            remaining = max(0., min(1., (days - day + 1.) / days))
            total -= EXPLORE_MAX * remaining / (1. + visits) ** 2
        return max(-LIMIT, min(LIMIT, total))

    def scaled(self, spot, state, breadth):
        b = _num(breadth, 0.)
        return max(-LIMIT, min(LIMIT, b * self.penalty(spot, state)))


# ======================= controller.py (verbatim; package imports stripped, renames {'FORGET': 'CTRL_FORGET'}) =======================
"""Meta-controller: a bounded tuner of three coarse knobs.

Each knob is an independent discounted Thompson-sampling bandit over three arms with Gaussian
posteriors. The reward is the realized daily profit minus the model's own predicted profit,
divided by a running scale so it is roughly unit size: the bandits learn which settings the
planner over- or under-trusts. All three knobs are credited with the same reward.

Per arm we keep discounted sufficient statistics (n, s) on top of a prior (mean, var):
    precision = 1/prior_var + n/OBS_VAR,  mean = (prior_mean/prior_var + s/OBS_VAR)/precision
Every update decays the evidence of every arm, so unplayed arms grow uncertain again and the
controller can track regime shifts. It never leaves its arm sets.
"""
import json
import math
import random
from typing import NamedTuple


class Knobs(NamedTuple):
    price_aggr: float
    breadth: float
    risk: float


ARMS = {'price_aggr': (0.90, 1.00, 1.10), 'breadth': (0.0, 0.5, 0.75), 'risk': (0.03, 0.10, 0.30)}
NAMES = tuple(ARMS)
DEFAULT = Knobs(1.00, 0.5, 0.10)
GAMMA = 0.97          # evidence discount per update
CTRL_FORGET = 0.5          # evidence kept at a new market
OBS_VAR = 1.0
MID_PRIOR = (0.6, 0.16)    # (mean, var) of the middle arm: favoured, so it wins most draws
SIDE_PRIOR = (0.0, 0.16)
VAR_FLOOR, VAR_CEIL = 0.01, 4.0
SCALE0, SCALE_MIN = 30., 1.
CLIP = 3.


def _num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _key(arm):
    return repr(float(arm))


def load_priors(path):
    """Read model/final_priors.json's 'controller' section; {} when missing or malformed."""
    try:
        with open(path) as handle:
            data = json.load(handle)
        section = data.get('controller')
        return section if isinstance(section, dict) else {}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _prior(name, index, priors):
    mean, var = MID_PRIOR if index == 1 else SIDE_PRIOR
    arm = ARMS[name][index]
    entry = {}
    try:
        group = (priors or {}).get(name) or {}
        entry = group.get(str(arm)) or group.get(_key(arm)) or group.get('%g' % arm) or {}
    except AttributeError:
        pass
    if isinstance(entry, dict):
        mean = _num(entry.get('mean'), mean)
        var = _num(entry.get('var'), var)
    return mean, min(max(var, VAR_FLOOR), VAR_CEIL)


def _fresh(priors=None):
    out = {}
    for name in NAMES:
        out[name] = []
        for index in range(3):
            mean, var = _prior(name, index, priors)
            out[name].append({'pm': mean, 'pv': var, 'n': 0., 's': 0.})
    return out


class MetaController:
    def __init__(self, data=None, seed=0, priors=None):
        data = data if isinstance(data, dict) else {}
        self.arms = _fresh(priors)
        stored = data.get('arms')
        if isinstance(stored, dict):
            for name in NAMES:
                rows = stored.get(name)
                if not isinstance(rows, list) or len(rows) != 3:
                    continue
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    arm = self.arms[name][index]
                    for field in ('pm', 'pv', 'n', 's'):
                        value = _num(row.get(field))
                        if value is not None:
                            arm[field] = value
                    arm['pv'] = min(max(arm['pv'], VAR_FLOOR), VAR_CEIL)
                    arm['n'] = max(0., arm['n'])
        self.scale_sq = max(_num(data.get('scale_sq'), SCALE0 ** 2), SCALE_MIN ** 2)
        self.updates = max(0, int(_num(data.get('updates'), 0)))
        self.seed = int(_num(data.get('seed'), seed))
        self.rng = random.Random(self.seed)
        try:
            version, internal, gauss = data['rng']
            self.rng.setstate((int(version), tuple(int(x) for x in internal), gauss))
        except (KeyError, TypeError, ValueError):
            pass

    def to_dict(self):
        return {'arms': {name: [dict(row) for row in rows] for name, rows in self.arms.items()},
                'scale_sq': self.scale_sq, 'updates': self.updates, 'seed': self.seed,
                'rng': list(self.rng.getstate())}

    # --- posterior --------------------------------------------------------
    @staticmethod
    def posterior(row):
        precision = 1. / row['pv'] + row['n'] / OBS_VAR
        return (row['pm'] / row['pv'] + row['s'] / OBS_VAR) / precision, 1. / precision

    def means(self):
        return {name: [self.posterior(row)[0] for row in rows] for name, rows in self.arms.items()}

    # --- choosing ---------------------------------------------------------
    def choose(self, context=None, greedy=False):
        streak = 0
        if isinstance(context, dict):
            streak = min(3, max(0, int(_num(context.get('losing_streak'), 0))))
        picks = []
        for name in NAMES:
            scores = []
            for index, row in enumerate(self.arms[name]):
                mean, var = self.posterior(row)
                if not greedy:
                    mean = self.rng.gauss(mean, math.sqrt(var))
                if name == 'risk' and streak:   # recent losing streak: temporarily lean cautious
                    mean += 0.15 * streak / 3. * (index - 1)
                scores.append(mean)
            best = max(range(3), key=lambda i: (scores[i], i == 1))
            picks.append(ARMS[name][best])
        return Knobs(*picks)

    # --- learning ---------------------------------------------------------
    def normalize(self, reward):
        reward = _num(reward)
        if reward is None:
            return None
        scale = math.sqrt(self.scale_sq)
        out = max(-CLIP, min(CLIP, reward / scale))
        rate = max(0.05, 1. / (self.updates + 2))
        self.scale_sq = max(SCALE_MIN ** 2, (1 - rate) * self.scale_sq + rate * reward * reward)
        return out

    def update(self, knobs, reward):
        """Credit all three knobs with the (normalized) reward; returns the normalized value."""
        try:
            values = tuple(knobs)
            if len(values) != 3:
                return None
        except TypeError:
            return None
        norm = self.normalize(reward)
        if norm is None:
            return None
        for name, value in zip(NAMES, values):
            value = _num(value)
            if value is None:
                continue
            index = min(range(3), key=lambda i: abs(ARMS[name][i] - value))
            for row in self.arms[name]:
                row['n'] *= GAMMA
                row['s'] *= GAMMA
            self.arms[name][index]['n'] += 1.
            self.arms[name][index]['s'] += norm
        self.updates += 1
        return norm

    def new_market(self):
        """Partial forgetting: shrink evidence toward the prior; the learned means are kept."""
        for rows in self.arms.values():
            for row in rows:
                row['n'] *= CTRL_FORGET
                row['s'] *= CTRL_FORGET


# ======================= rules_adapter.py (verbatim; package imports stripped) =======================
"""Turns raw state/features into planner inputs so new rules never touch the planner.

Handles new spots, meat/veggie orders, riot exposure, coupon awareness and learned
unmodeled profit terms. Never raises on malformed input; missing data is neutral.
"""
import copy
import math

# inlined: from game import DISTANCES, PRIORS, SPOTS

# New-spot hypotheses: (price limit, premium share, crowd weight, meat-only, veggie-only).
NEW_SPOTS = {
    'fisher': (14., .30, .12, .40, .10),
    'kottman': (12.5, .25, .08, .10, .50),
    'mirror': (12., .20, .10, .25, .20),
    'medcenter': (13., .20, .14, .25, .15),
    'lane': (12., .15, .12, .30, .10),
    'rpac': (12., .25, .09, .45, .10),
    'schott': (13., .20, .07, .35, .10),
    'gateway': (11.5, .15, .11, .25, .15),
}
BASE_DIETS = {'oval': (.20, .12), 'union': (.20, .12), 'northres': (.20, .12), 'thompson': (.15, .20),
              'stadium': (.27, .03), 'highst': (.24, .06)}
RIOTS = {'loss': (.7, 260., 'highst'), 'title': (.95, 380., 'stadium')}
GENERIC_PRIOR = {'mu': 12., 'sd': 3., 'mix': (.45, .40, .15), 'weight': .10}


def _num(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _dict(x):
    return x if isinstance(x, dict) else {}


def _list(x):
    return x if isinstance(x, (list, tuple)) else []


def _open_spots(state):
    spots = _dict(state.get('rules')).get('spots')
    return [s for s in _list(spots) if isinstance(s, str)] or list(SPOTS)


def prior_for(spot):
    if spot in PRIORS:
        mu, sd, mix, weight = PRIORS[spot]
        return {'mu': mu, 'sd': sd, 'mix': tuple(mix), 'weight': weight}
    if spot in NEW_SPOTS:
        limit, premium, weight = NEW_SPOTS[spot][:3]
        cheap = max(.05, min(.6, .55 - premium))
        return {'mu': limit, 'sd': 3., 'mix': (cheap, round(1. - cheap - premium, 3), premium), 'weight': weight}
    return dict(GENERIC_PRIOR)


def _coords(state):
    out = {}
    for spot, info in _dict(state.get('spot_info')).items():
        info = _dict(info)
        x, z = _num(info.get('x')), _num(info.get('z'))
        if x is not None and z is not None:
            out[spot] = (x, z)
    return out


def distances_for(state):
    spots = _open_spots(state)
    coords = _coords(state)
    out = {}
    for a in spots:
        for b in spots:
            if a == b:
                continue
            if a in coords and b in coords:
                out[(a, b)] = math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
            elif b in DISTANCES.get(a, {}):
                out[(a, b)] = float(DISTANCES[a][b])
    return out


def _riot_law_scale(state):
    """Election laws scale riot fierceness (crackdown 0, tailgate 1.5); 1 when no law says so."""
    try:
        pass  # inlined: from riots import law_multiplier
        return max(0., float(law_multiplier(state)))
    except Exception:
        return 1.


class RuleAdapter:
    def __init__(self, data=None):
        data = _dict(data)
        self.residual = {k: float(v) for k, v in _dict(data.get('residual')).items() if _num(v) is not None}
        self.streak = {k: int(v) for k, v in _dict(data.get('streak')).items() if _num(v) is not None}
        self.profit_proxy = _num(data.get('profit_proxy'), 150.)
        self.coupons_seen = int(_num(data.get('coupons_seen'), 0))
        self.wrecks = int(_num(data.get('wrecks'), 0))
        self.hits = int(_num(data.get('hits'), 0))
        self.min_streak = 3
        self.trivial = 3.

    def to_dict(self):
        return {'residual': dict(self.residual), 'streak': dict(self.streak),
                'profit_proxy': self.profit_proxy, 'coupons_seen': self.coupons_seen,
                'wrecks': self.wrecks, 'hits': self.hits}

    # ---- state normalization -------------------------------------------------
    def normalize(self, state):
        try:
            out = copy.deepcopy(state) if isinstance(state, dict) else {}
            out['spot_priors'] = {s: prior_for(s) for s in _open_spots(out)}
            out['spot_distances'] = distances_for(out)
            fc = out.get('forecast')
            if isinstance(fc, dict) and isinstance(fc.get('expected_customers'), dict):
                fc['expected_customers'] = {k: v for k, v in fc['expected_customers'].items()
                                            if _num(v) is not None}
            return out
        except Exception:
            return dict(state) if isinstance(state, dict) else {}

    # ---- meat or veggie ------------------------------------------------------
    def order_extras(self, state, order):
        try:
            if not _dict(_dict(state).get('features')).get('food'):
                return {}
            spot = _dict(order).get('spot')
            diets = _dict(_dict(_dict(state).get('forecast')).get('diets')).get(spot)
            if isinstance(diets, dict):
                meat, veg = _num(diets.get('meat')), _num(diets.get('veggie'))
            else:
                meat = veg = None
            if meat is None or veg is None:
                base = BASE_DIETS.get(spot) or (NEW_SPOTS[spot][3:5] if spot in NEW_SPOTS else None)
                if not base:
                    return {'food': 'meat'}
                meat, veg = base
            rivals = {'meat': 0, 'veggie': 0}
            for t in _list(_dict(_dict(_dict(state).get('last_day')).get('spots')).get(spot, {}).get('trucks')
                           if isinstance(_dict(_dict(state).get('last_day')).get('spots'), dict) else []):
                f = _dict(t).get('food')
                if f in rivals:
                    rivals[f] += 1
                elif isinstance(t, dict):
                    rivals['meat'] += 1
            # share per truck already serving that diet (+1 for us); higher means underserved
            meat_score = meat / (rivals['meat'] + 1.)
            veg_score = veg / (rivals['veggie'] + 1.)
            return {'food': 'veggie' if veg_score > meat_score else 'meat'}
        except Exception:
            return {}

    # ---- riots ---------------------------------------------------------------
    def _riot_warnings(self, state):
        fc = _dict(_dict(state).get('forecast'))
        entries = []
        for key in ('riot', 'riots'):
            raw = fc.get(key)
            if isinstance(raw, dict):
                entries.append(raw)
            elif isinstance(raw, (list, tuple)):
                entries.extend(raw)
            elif isinstance(raw, str):
                entries.append(raw)
            elif raw:
                entries.append('riot')
        for e in _list(fc.get('events')):
            if 'riot' in str(e).lower():
                entries.append(e)
        out = []
        for e in entries:
            text = str(e).lower()
            kind = 'title' if any(w in text for w in ('title', 'national', 'champion')) else 'loss'
            intensity, radius, center = RIOTS[kind]
            if isinstance(e, dict):
                intensity = _num(e.get('intensity'), intensity)
                radius = _num(e.get('radius'), radius)
                spot = e.get('spot') or e.get('where') or e.get('location')
                if isinstance(spot, str) and spot:
                    center = spot
            out.append((intensity, radius or 1., center, kind))
        return out

    def _repair_cost(self, d, radius):
        return 30. + 120. * math.exp(-d / radius)

    def spot_cost(self, state, spot, include_residual=True):
        try:
            cost = 0.
            if _dict(_dict(state).get('features')).get('riots'):
                scale = _riot_law_scale(state)
                dist = distances_for(state if isinstance(state, dict) else {})
                for intensity, radius, center, _ in self._riot_warnings(state):
                    d = 0. if spot == center else dist.get((spot, center), dist.get((center, spot)))
                    if d is None:
                        continue
                    p = min(1., max(0., intensity * math.exp(-d / radius)))
                    cost += scale * p * (.3 * self.profit_proxy + .7 * self._repair_cost(d, radius))
            if include_residual and self.streak.get(spot, 0) >= self.min_streak:
                cost -= self.residual.get(spot, 0.)
            return cost
        except Exception:
            return 0.

    # ---- learning ------------------------------------------------------------
    def observe(self, last_day, our_order, modeled_profit):
        try:
            you = _dict(_dict(last_day).get('you'))
            spot = _dict(our_order).get('spot')
            if you.get('coupons') or _num(you.get('coupon_share'), 0.) > 0 or _num(you.get('coupon_loss'), 0.) > 0:
                self.coupons_seen += 1
            if you.get('wrecked_until_day') or you.get('wrecked'):
                self.wrecks += 1
            repairs = _num(you.get('repairs'), 0.) or 0.
            if repairs > 0:
                self.hits += 1
            reported, modeled = _num(you.get('profit')), _num(modeled_profit)
            if reported is None:
                return
            if reported > 0:
                self.profit_proxy = .8 * self.profit_proxy + .2 * reported
            if modeled is None or not isinstance(spot, str):
                return
            # repairs and coupon loss are explained terms; the rest is unmodeled
            res = reported - modeled + repairs + (_num(you.get('coupon_loss'), 0.) or 0.)
            if abs(res) <= self.trivial:
                self.streak[spot] = 0
                self.residual[spot] = .5 * self.residual.get(spot, 0.)
                return
            prev = self.residual.get(spot)
            same = prev is not None and prev * res > 0
            self.streak[spot] = self.streak.get(spot, 0) + 1 if same or prev is None else 1
            self.residual[spot] = res if prev is None or not same else .6 * prev + .4 * res
        except Exception:
            return

    # ---- coupons / wrecks ----------------------------------------------------
    def coupon_action(self, state):
        """Hook for offensive coupons. Defensive stance: always None until a measured gain exists."""
        return None

    def is_wrecked(self, state):
        try:
            state = _dict(state)
            until = _num(_dict(state.get('you')).get('wrecked_until_day'))
            day = _num(state.get('order_for_day'))
            return until is not None and day is not None and until >= day
        except Exception:
            return False


# ======================= complaints.py (verbatim; package imports stripped, renames {'enabled': 'complaints_enabled'}) =======================
"""Health complaints against a rival: a rarely used move, filed only when the numbers clearly favor it.

The game's tip line cuts the target's reputation 20% at every spot from the next day. Each complaint can
be caught: with k complaints in the last 3 days (this one included) the catch chance is 10% x k, and a
catch bans us (5 days live; the rest of the market in the final), throws the complaint out and restores
the targets. So the model weighs the gain against catch risk x loss:

    gain  = our extra expected profit tomorrow if the target's reputation were 20% lower, times how long
            the dent lasts (reputation recovers as the target sells)
    loss  = days we would miss x our recent daily profit (more days left, more to lose)
    file  = gain - p_catch * (loss + gain) clears a margin, at most one complaint per 3 days, and only
            against a rival that is likely to share our spot.

Because loss shrinks as the market ends, filing becomes attractive only late. Everything is switchable:
NIGHTSHIFT_COMPLAINTS=off or data/profile.json {"complaints": "off"}.
"""
import json
import os
from pathlib import Path

ENDPOINT = '/x/8243598d/complaints'
WINDOW_DAYS = 3
CATCH_STEP = .10
LIVE_BAN_DAYS = 5
PERSISTENCE = 2.0        # a 20% dent lasts about this many days of full effect as reputation recovers
OVERLAP_MIN = .35        # the target must likely share our spot
MIN_GAIN = 40.
MARGIN = 25.


def complaints_enabled(mode, root=None):
    """On in the final by default; elsewhere only when asked for."""
    flag = os.environ.get('NIGHTSHIFT_COMPLAINTS')
    if flag in ('on', 'off'):
        return flag == 'on'
    try:
        path = None
        value = _profile().get('complaints')
        if value in ('on', 'off'):
            return value == 'on'
    except (OSError, ValueError, AttributeError):
        pass
    return mode in ('final', 'showcase')


def catch_chance(filed_recently):
    return min(.95, CATCH_STEP * (filed_recently + 1))


def ban_days(state):
    """Days we would miss if caught: the rest of the market in the final, 5 in the live game."""
    left = max(0, int(state.get('days') or 0) - int(state.get('order_for_day') or state.get('day') or 0) + 1)
    return left if state.get('mode') in ('final', 'showcase', 'practice') else min(LIVE_BAN_DAYS, left)


class Complaints:
    def __init__(self, data=None):
        data = data or {}
        self.filed = [tuple(x) for x in data.get('filed', [])]     # (market key, day, target)
        self.caught = int(data.get('caught', 0))
        self.log = data.get('log', [])[-20:]

    def to_dict(self):
        return {'filed': self.filed[-30:], 'caught': self.caught, 'log': self.log[-20:]}

    def recent(self, market, day):
        return sum(1 for m, d, _ in self.filed if m == market and day - d < WINDOW_DAYS)

    def analyse(self, state, brain, order, daily_profit):
        """Best target and its expected value, or a reason not to file."""
        market, day = tuple((state.get('mode'), state.get('market'))), int(state.get('order_for_day') or 0)
        market = list(market)
        if self.recent(market, day):
            return {'file': False, 'reason': 'one complaint per %d days' % WINDOW_DAYS}
        if not order or not order.get('meals'):
            return {'file': False, 'reason': 'no order'}
        me = (state.get('you') or {}).get('id')
        rivals = [t['id'] for t in state.get('trucks') or [] if t.get('id') and t['id'] != me]
        if not rivals:
            return {'file': False, 'reason': 'no rivals'}
        p_catch = catch_chance(0)
        loss = ban_days(state) * max(0., daily_profit)
        base = brain.order_value(state, order, {})
        overlaps = dict(brain.last_overlap)
        likely = sorted((t for t in rivals if overlaps.get(t, 0.) >= OVERLAP_MIN), key=lambda t: -overlaps[t])[:3]
        best = None
        for target in likely:
            hit = brain.order_value(state, order, {target: .8})
            overlap = overlaps.get(target, 0.)
            gain = (hit['profit'] - base['profit']) * PERSISTENCE
            ev = gain - p_catch * (loss + gain)
            row = {'target': target, 'gain': gain, 'overlap': overlap, 'ev': ev, 'loss': loss, 'p_catch': p_catch}
            if overlap >= OVERLAP_MIN and (best is None or ev > best['ev']):
                best = row
        if best is None:
            return {'file': False, 'reason': 'no rival likely to share our spot'}
        best['file'] = best['gain'] >= MIN_GAIN and best['ev'] >= max(MARGIN, .25 * best['gain'])
        best['reason'] = 'gain $%.0f, catch risk x loss $%.0f, ev $%.0f' % (best['gain'], p_catch * (loss + best['gain']), best['ev'])
        return best

    def record(self, state, target, response):
        market = list(tuple((state.get('mode'), state.get('market'))))
        self.filed.append((market, int(state.get('order_for_day') or 0), target))
        caught = isinstance(response, dict) and any(
            bool(v) for k, v in response.items() if 'caught' in str(k).lower() or 'banned' in str(k).lower())
        self.caught += bool(caught)
        self.log = (self.log + [{'day': state.get('order_for_day'), 'target': target, 'caught': bool(caught),
                                 'response': str(response)[:200]}])[-20:]
        return caught


# ======================= final_mode.py (verbatim; package imports stripped) =======================
"""Adaptive policy for the final: learned calibration, dynamic location choice, bounded knobs.

The policy always observes (shadow learning on live days) but only steers decisions while
Brain.final_active. Every call is guarded: a failing component returns a neutral value and the
planner behaves like the baseline for that factor. Kill switch: NIGHTSHIFT_PROFILE=baseline or
data/profile.json {"profile": "baseline"} (see bot.profile_wanted).
"""
import json
from pathlib import Path
import time
# inlined: from game import reputation_value, VALUES
# inlined: from posture import maturity
# inlined: from complaints import Complaints

try:
    pass  # inlined: from calibration import Calibrator
except ImportError:  # modules are optional so the baseline never depends on them
    Calibrator = None
try:
    pass  # inlined: from competition import Competition
except ImportError:
    Competition = None
try:
    pass  # inlined: from controller import MetaController, Knobs, load_priors
except ImportError:
    MetaController, Knobs, load_priors = None, None, None
try:
    pass  # inlined: from rules_adapter import RuleAdapter
except ImportError:
    RuleAdapter = None

MIDDLE = {'price_aggr': 1.0, 'breadth': .5, 'risk': .10}
SAMPLE_RANGE = (12, 32)
SLOW, FAST = .9, .4  # seconds of the last decision: shrink or grow the scenario count


class FinalPolicy:
    def __init__(self, data=None, priors=None, seed=0):
        data, priors = data or {}, priors or {}
        self.calibrator = self._build(Calibrator, data.get('calibration') or calibrator_seed(priors))
        self.competition = self._build(Competition, data.get('competition'))
        self.adapter = self._build(RuleAdapter, data.get('rules'))
        self.controller = None
        if MetaController is not None:
            try:
                self.controller = MetaController(data.get('controller'), seed=seed, priors=priors.get('controller'))
            except Exception:
                self.controller = None
        self.knobs = dict(MIDDLE)
        self.knob_day = None
        self.profit_mean = data.get('profit_mean')
        self.final_started = bool(data.get('final_started'))
        self.complaints = Complaints(data.get('complaints'))
        self.seed, self.priors = seed, priors
        self.errors = {}

    @staticmethod
    def _build(cls, data):
        if cls is None:
            return None
        try:
            return cls(data) if data else cls()
        except Exception:
            return None

    def _guard(self, name, fn, default):
        try:
            return fn()
        except Exception as error:
            self.errors[name] = type(error).__name__
            return default

    # --- persistence ----------------------------------------------------
    def to_dict(self):
        parts = {'calibration': self.calibrator, 'competition': self.competition,
                 'controller': self.controller, 'rules': self.adapter}
        out = {name: self._guard(name, lambda part=part: part.to_dict(), None)
               for name, part in parts.items() if part is not None}
        return dict(out, knobs=self.knobs, knob_day=self.knob_day, profit_mean=self.profit_mean, final_started=self.final_started,
                    complaints=self.complaints.to_dict())

    def restore(self, data):
        for key, cls, attr in (('calibration', Calibrator, 'calibrator'), ('competition', Competition, 'competition'),
                               ('rules', RuleAdapter, 'adapter')):
            if data.get(key) and cls is not None:
                setattr(self, attr, self._build(cls, data[key]) or getattr(self, attr))
        if data.get('controller') and MetaController is not None:
            self.controller = self._guard('controller', lambda: MetaController(data['controller']), self.controller)
        self.knobs = data.get('knobs') or dict(MIDDLE)
        self.knob_day = data.get('knob_day')
        self.profit_mean = data.get('profit_mean')
        self.final_started = bool(data.get('final_started'))
        self.complaints = Complaints(data.get('complaints'))

    def reset_for_final(self):
        """First final state: the tuner, competition and unexplained-profit residuals were learned under
        live conditions (reputation 100, one market), so they start fresh. Customer, crowd, rival,
        calibration, crew and event-rate learning lives elsewhere and is kept. Runs once, even across restarts."""
        if self.final_started:
            return False
        if MetaController is not None:
            self.controller = self._guard('reset', lambda: MetaController(None, seed=self.seed, priors=self.priors.get('controller')), self.controller)
        if Competition is not None:
            self.competition = self._build(Competition, None) or self.competition
        if RuleAdapter is not None:
            self.adapter = self._build(RuleAdapter, None) or self.adapter
        self.knobs, self.knob_day, self.profit_mean = dict(MIDDLE), None, None
        self.final_started = True
        return True

    def new_market(self):
        for part in (self.competition, self.controller):
            if part is not None:
                self._guard('new_market', lambda part=part: part.new_market(), None)

    # --- before the order -------------------------------------------------
    def begin(self, brain, state):
        """Pick tonight's knobs and apply risk and scenario count to the planner."""
        streak = 0
        for check in reversed([c for c in brain.prediction_checks
                               if c.get('mode') == state.get('mode') and c.get('market') == state.get('market')]):
            if (check.get('actual_profit') or 0) < 0:
                streak += 1
            else:
                break
        knobs = dict(MIDDLE)
        if self.controller is not None:
            context = dict(self._context(state), losing_streak=streak)  # the controller leans cautious on streaks
            chosen = self._guard('choose', lambda: self.controller.choose(context), None)
            if chosen is not None:
                knobs = {name: float(getattr(chosen, name)) for name in MIDDLE}
        self.knobs, self.knob_day = knobs, state.get('order_for_day')
        brain.risk = knobs['risk']
        elapsed = (brain.diagnostics or {}).get('elapsed')
        low, high = SAMPLE_RANGE
        if elapsed:
            if elapsed > SLOW:
                brain.samples = max(low, int(brain.samples * .7))
            elif elapsed < FAST and brain.samples < high:
                brain.samples = min(high, brain.samples + 4)

    @staticmethod
    def _context(state):
        return {'day': state.get('order_for_day'), 'days': state.get('days'), 'mode': state.get('mode')}

    def demand_factor(self, state, spot, quality, price, charged, rep):
        if self.calibrator is None:
            return 1.
        def run():
            values = (state.get('rules') or {}).get('values') or VALUES
            ceiling = reputation_value(quality, rep, values)
            factor = float(self.calibrator.factor(spot, quality, charged, ceiling))
            return max(.3, min(1.15, factor ** (1. / self.knobs['price_aggr'])))
        return self._guard('factor', run, 1.)

    def score_adjust(self, state, spot):
        """Dollars per day to subtract from a spot's score (negative is a bonus)."""
        total = 0.
        if self.competition is not None:
            def competition():
                penalty = float(self.competition.scaled(spot, state, self.knobs['breadth']))
                return penalty * maturity(state) if penalty < 0 else penalty   # roaming bonuses wait for a home spot
            total += self._guard('competition', competition, 0.)
        if self.adapter is not None:
            total += self._guard('spot_cost', lambda: float(self.adapter.spot_cost(state, spot, include_residual=False)), 0.)
        return max(-80., min(80., total))

    def order_extras(self, state, order):
        if self.adapter is None:
            return {}
        return self._guard('extras', lambda: dict(self.adapter.order_extras(state, order) or {}), {})

    def complaint_decision(self, state, brain, order):
        """Whether to file a health complaint tonight (see complaints.py); never raises."""
        recent = [c.get('actual_profit') for c in brain.prediction_checks[-7:] if c.get('actual_profit') is not None]
        daily = sum(recent) / len(recent) if recent else float((brain.diagnostics.get('chosen') or {}).get('profit') or 0.)
        return self._guard('complaints', lambda: self.complaints.analyse(state, brain, order, daily),
                           {'file': False, 'reason': 'error'})

    def is_wrecked(self, state):
        return bool(self.adapter is not None and self._guard('wrecked', lambda: self.adapter.is_wrecked(state), False))

    # --- after the day ----------------------------------------------------
    def observe_day(self, state, last, prediction, check, rep_before):
        """Learn from the completed day. Runs on live days too (shadow learning)."""
        order = check.get('order') or {}
        if not check.get('order_matches') or not order or check.get('surprise'):
            return
        spot, quality = order.get('spot'), order.get('quality')
        calibration = prediction.get('calibration') or 1.
        raw = (prediction.get('demand') or 0.) / max(calibration, 1e-6)
        values = (state.get('rules') or {}).get('values') or VALUES
        rep = float((rep_before or {}).get(spot, 50.))
        if self.calibrator is not None and check.get('actual_sales') is not None:
            self._guard('calibrate', lambda: self.calibrator.observe({
                'spot': spot, 'quality': quality, 'price': prediction.get('charged') or order.get('price'),
                'value_ceiling': reputation_value(quality, rep, values), 'expected_demand': raw,
                'actual_sales': check['actual_sales'], 'meals': order.get('meals'),
                'sellout': check.get('sellout'), 'weather': last.get('weather')}), None)
        if self.competition is not None:
            self._guard('compete', lambda: self.competition.observe(last, order, prediction), None)
        if self.adapter is not None:
            self._guard('rules', lambda: self.adapter.observe(last, order, prediction.get('profit')), None)
        if (self.controller is not None and self.knob_day == last.get('day')
                and check.get('actual_profit') is not None):
            knobs = Knobs(**self.knobs) if Knobs is not None else None
            actual = float(check['actual_profit'])
            if knobs is not None:
                # Reward beats a running mean of real profit, which no knob can inflate by deflating the forecast.
                baseline = self.profit_mean if self.profit_mean is not None else actual
                self._guard('update', lambda: self.controller.update(knobs, actual - baseline), None)
            self.profit_mean = actual if self.profit_mean is None else .8 * self.profit_mean + .2 * actual

    def report(self):
        out = {'knobs': self.knobs, 'errors': dict(self.errors)}
        if self.calibrator is not None:
            out['calibration'] = self._guard('report', lambda: self.calibrator.report(), {})
        return out


def calibrator_seed(priors):
    """Calibrator state from train_final priors: a few pseudo-observations per price-ratio cell."""
    cells = {}
    try:
        pass  # inlined: from calibration import EDGES
        for quality, bins in ((priors.get('calibration') or {}).get('buckets') or {}).items():
            for span, cell in bins.items():
                low, high = (float(x) for x in span.split('-'))
                n = float(cell.get('n') or 0)
                if n <= 0:
                    continue
                key = '%s|%d' % (quality, sum((low + high) / 2 > edge for edge in EDGES))
                weight = min(n, 3.)
                total, old = cells.get(key, [0., 0.])
                cells[key] = [total + float(cell['mean']) * weight, old + weight]
    except Exception:
        return None
    return {'cells': cells, 'count': 0} if cells else None


def load_priors_file(path=None):
    try:  # port: model/final_priors.json inlined as FINAL_PRIORS_JSON
        return json.loads(FINAL_PRIORS_JSON)
    except (OSError, ValueError):
        return {}




def attach_policy(brain, seed=0, priors_path=None):
    """Attach a policy (inactive) so it can learn from live days before the final starts."""
    if brain.final_policy is None:
        brain.final_policy = FinalPolicy(priors=load_priors_file(priors_path), seed=seed)
    return brain.final_policy


# ======================= strategy.py (verbatim; package imports stripped) =======================
"""Learned sales forecasting plus exhaustive location/quality profit optimization.

Every spot receives the same fixed search effort. No crowd-ordered early exit,
forced rival undercut, premium-only switch, or arbitrary selling-price cap.
"""
import hashlib
import math
import random
import time
# inlined: import game
# inlined: from game import (SPOTS, known_spots, QUALITIES, PRIORS, COSTS, REP_CHANGE, Rules, clamp, customer_model, events, identity, reputation_value, spot_results, weather_factor, weather_probs, without_null_counts)
# inlined: from learning import Learner
# inlined: from opponents import OpponentModel, context
# inlined: from forecast_log import capture
# inlined: from mechanics import features, demand_levels
# inlined: from overheads import Overheads
# inlined: from riots import riot_cost
# inlined: import posture as posture_mod
# inlined: from staffing import CrewModel

# Documented surprise rates; building spots can have a fire drill (open ones today).
BUILDINGS = ('thompson', 'union', 'northres')
FIRE_DRILL_CHANCE, GIVEAWAY_CHANCE = .06, .10
# Chance an unforeseen rival shows up at a spot: events and weight of a weak prior, then clamped.
PHANTOM_PRIOR, PHANTOM_LIMITS = (.64, 8.), (.03, .30)


class Brain:
    def __init__(self, learner=None, seed=42, samples=12, risk=.15, price_cap=None, quiet_bonus=0., small_crowds=False, spot_price_lift=None):
        self.learner = learner or Learner()
        self.seed, self.samples, self.risk = seed, max(4, int(samples)), float(risk)
        # Optional steering: never charge above price_cap; reward spots where rivals rarely park.
        self.price_cap, self.quiet_bonus = price_cap, float(quiet_bonus)
        self.small_crowds = bool(small_crowds)
        # Optional adaptive policy for the final; it always observes (shadow learning) but only
        # steers decisions while final_active. See final_mode.py.
        self.final_policy, self.final_active, self._baseline = None, False, None
        # Per-spot multiplier on the low price ceilings, e.g. {'thompson': 1.10}.
        self.spot_price_lift = dict(spot_price_lift or {})
        self.low_price_caps = {'cheap': 5.50, 'standard': 7.00, 'premium': 8.50}
        self.market = None
        self.seen = None
        self.history = {}
        self.reputations = {}
        self.pending = None
        self.last_order = None
        self.diagnostics = {}
        self.opponents = OpponentModel()
        self.crew = CrewModel()
        self.cash_history = {}   # day -> {truck id: cash}, for recent daily earnings of every truck
        self._rep_hit, self.last_overlap = {}, {}
        self.posture, self._home = dict(posture_mod.NEUTRAL), set()
        self.phantom = {'events': PHANTOM_PRIOR[0], 'exposures': PHANTOM_PRIOR[1]}
        self.surprises = {'days': 0, 'fire': 0, 'give': 0}  # counted from last_day.moments
        self.protests = {'days': 0, 'protests': 0, 'against_us': 0}  # office days seen and protests among them
        self.prediction_checks = []
        self.forecast_record = None
        self.scenario_rivals = []

    def _final_state(self):
        try:
            return self.final_policy.to_dict() if self.final_policy else None
        except Exception:
            return None  # the policy must never break checkpointing

    def use_final(self, active):
        """Switch between the baseline policy and the adaptive final policy."""
        active = bool(active and self.final_policy is not None)
        if active == self.final_active:
            return
        if active:
            self._baseline = (self.small_crowds, self.risk, dict(self.spot_price_lift), self.samples)
            self.small_crowds, self.spot_price_lift = False, {}
        elif self._baseline:
            self.small_crowds, self.risk, self.spot_price_lift, self.samples = self._baseline
        self.final_active = active

    def dump(self):
        return {'learner': self.learner.to_dict(), 'market': self.market, 'seen': self.seen,
                'history': self.history, 'reputations': self.reputations,
                'pending': self.pending, 'last_order': self.last_order, 'opponents': self.opponents.data,
                'prediction_checks': self.prediction_checks, 'crew': self.crew.to_dict(), 'phantom': self.phantom, 'surprises': self.surprises, 'protests': self.protests,
                'cash_history': {str(d): v for d, v in self.cash_history.items()},
                'final': self._final_state()}

    def restore(self, data):
        self.learner = Learner.from_dict(data['learner'])
        self.opponents = OpponentModel(data.get('opponents'))
        self.prediction_checks = data.get('prediction_checks', [])
        self.crew = CrewModel(data.get('crew'))
        self.phantom = data.get('phantom') or self.bootstrap_phantom(self.prediction_checks)
        self.surprises = data.get('surprises') or {'days': 0, 'fire': 0, 'give': 0}
        self.protests = data.get('protests') or {'days': 0, 'protests': 0, 'against_us': 0}
        self.cash_history = {int(d): v for d, v in (data.get('cash_history') or {}).items()}
        if self.final_policy is not None and data.get('final'):
            self.final_policy.restore(data['final'])
        for key in ('market', 'seen', 'history', 'reputations', 'pending', 'last_order'):
            setattr(self, key, data[key])

    @staticmethod
    def bootstrap_phantom(checks):
        """Older checkpoints: learn the rate from the saved prediction checks."""
        phantom = {'events': PHANTOM_PRIOR[0], 'exposures': PHANTOM_PRIOR[1]}
        for check in checks:
            if check.get('order_matches') and not check.get('surprise'):
                phantom['exposures'] += 1
                phantom['events'] += min(1., max(0., check['actual_rivals'] - check['expected_rivals']))
        return phantom

    def game_day_rate(self):
        """Share of observed days with a game, shrunk toward 25%: how often game-day laws would bite."""
        days = {}
        for records in self.history.values():
            for record in records:
                days[record.get('day')] = days.get(record.get('day'), False) or (
                    'game' in ((record.get('context') or {}).get('events') or []))
        return (sum(days.values()) + 1.) / (len(days) + 4.)

    def surprise_rate(self, kind, prior):
        """Fire-drill / giveaway chance per day: the documented rate, updated by what we have seen."""
        weight = 30.
        return (self.surprises['days'] and (self.surprises[kind] + prior * weight) / (self.surprises['days'] + weight)) or prior

    def protest_rate(self, prior=.30):
        """Chance of a protest on a day someone holds office: the documented 30%, updated by what we see."""
        weight = 20.
        return (self.protests['protests'] + prior * weight) / (self.protests['days'] + weight)

    def phantom_rate(self):
        low, high = PHANTOM_LIMITS
        return min(high, max(low, self.phantom['events'] / self.phantom['exposures']))

    @staticmethod
    def register_new_spots(state):
        """Teach game.py about open spots beyond the original six; any failure leaves them ignored."""
        try:
            spots = (state.get('rules') or {}).get('spots') or ()
            extra = [s for s in spots if isinstance(s, str) and s not in SPOTS]
            if not extra or all(s in _REGISTERED for s in extra):
                return
            pass  # inlined: from rules_adapter import RuleAdapter
            normal = RuleAdapter().normalize(state)
            priors = {s: p for s, p in (normal.get('spot_priors') or {}).items() if s in extra}
            register_spots(priors, normal.get('spot_distances') or {})
        except Exception:
            pass

    def observe(self, state):
        self.register_new_spots(state)
        key = [state.get('mode'), state.get('market')]
        if key != self.market:
            if self.final_policy is not None:
                self.final_policy.new_market()
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
        own = clean_record(last.get('you') or {})
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
        # Fire drills and giveaways thin a spot's crowd after the forecast; do not learn that as bias.
        surprise_spots = {m.get('spot') for m in last.get('moments') or []
                          if isinstance(m, dict) and m.get('kind') in ('firedrill', 'giveaway', 'protest')}
        if (state.get('features') or {}).get('surprises'):  # learn how often surprises really happen
            kinds = {m.get('kind') for m in last.get('moments') or [] if isinstance(m, dict)}
            self.surprises['days'] += 1
            self.surprises['fire'] += 'firedrill' in kinds
            self.surprises['give'] += 'giveaway' in kinds
        if (state.get('politics') or {}).get('office'):  # an office holder is in: protests can happen
            protests = [m for m in last.get('moments') or [] if isinstance(m, dict) and m.get('kind') == 'protest']
            self.protests['days'] += 1
            self.protests['protests'] += bool(protests)
            self.protests['against_us'] += any(isinstance(t, dict) and t.get('id') == me and t.get('status') == 'protested'
                                               for m in protests for t in m.get('trucks') or [])
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
                     'sellout': matched and order['meals'] > 0 and sold >= order['meals'],
                     'surprise': order['spot'] in surprise_spots}
            # A sold-out truck reveals only a lower bound on uncapped demand.
            check['demand_error'] = (max(0., sold-prediction['demand']) if check['sellout']
                                     else abs(sold-prediction['demand'])) if matched else None
            self.prediction_checks = (self.prediction_checks + [check])[-48:]
            if self.final_policy is not None:
                self.final_policy.observe_day(state, last, prediction, check, before.get(me))
            if matched and not check['surprise']:
                self.phantom['exposures'] += 1
                self.phantom['events'] += min(1., max(0., check['actual_rivals'] - prediction['rivals']))
        if realized_features is not None:
            fit = last if not surprise_spots else dict(last, spots={s: v for s, v in spot_results(last).items()
                                                                    if s not in surprise_spots})
            self.learner.learn(fit, before, forecast, rules, strength=.7, features=realized_features)
        observed_context = context(forecast or {'events': last.get('events') or []})
        visited = {}
        records = {}
        for spot, info in spot_results(last).items():
            if spot not in known_spots():
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
        self.opponents.ours = self.history.get(me, [])[-8:]  # before today's records are added
        for name in sorted(roster - {me}):
            self.opponents.observe(name, records.get(name), self.history.get(name, []),
                                   forecast, self.history, rules.spots)
        for name, rec in records.items():
            self.history.setdefault(name, []).append(rec)
            self.history[name] = self.history[name][-48:]
        for name in set(self.reputations) | set(visited):
            reps = self.reputations.setdefault(name, {s: 50. for s in SPOTS})
            for spot in known_spots():
                reps.setdefault(spot, 50.)
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
        posture_mod.note_cash(self.cash_history, state)
        try:
            self.crew.observe(state, me)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            pass  # crew learning must never block ordering

    def crew_plan(self, state, order):
        """Headcount and wage for the accepted order; None when staff is not live."""
        if not (state.get('features') or {}).get('staff') or not isinstance(
                (state.get('you') or {}).get('staff'), dict):
            return None
        record = self.forecast_record if (self.forecast_record or {}).get('order') == order else None
        sales = [r['sales'] for r in (record or {}).get('scenarios', [])]
        return self.crew.plan(state, order, sales, identity(state.get('you') or {}))

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
        self.opponents.ours = self.history.get(me, [])[-8:]
        roster = [identity(t) for t in state.get('trucks') or [] if identity(t) != me]
        fc = state.get('forecast') or {}
        odds = weather_probs(fc)
        kinds = events(fc.get('events'))
        field = [h[-1] for name, h in self.history.items() if name in roster and h]
        scenarios = []
        self.scenario_rivals = []
        fire_rate = self.surprise_rate('fire', FIRE_DRILL_CHANCE)
        giveaway_rate = self.surprise_rate('give', GIVEAWAY_CHANCE)
        # Own stream: unforeseen rivals must not shift the draws used for everything else.
        phantoms = random.Random(hashlib.sha256(repr((self.seed, self.market, state.get('order_for_day'),
                                                      'phantom')).encode()).digest())
        rate = self.phantom_rate()
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
                rep = clamp(rep + rng.gauss(0, 5 if hist else 12), 0, 100) * self._rep_hit.get(name, 1.)
                placements[spot].append((price, q, rep, meals))
                named.append({'id': name, 'spot': spot, 'price': price,
                              'quality': q, 'reputation': rep, 'meals': meals})
                active += min(1., meals * rules.costs[q] / 250.)
            thinned = {}
            if (state.get('features') or {}).get('surprises'):
                buildings = [s for s in rules.spots if s in BUILDINGS]
                if buildings and rng.random() < fire_rate:
                    thinned[rng.choice(buildings)] = .6
                if rng.random() < giveaway_rate:
                    target = rng.choice(rules.spots)
                    thinned[target] = thinned.get(target, 1.) * rng.uniform(.6, .8)
            for spot in rules.spots:
                if field and phantoms.random() < rate:
                    rec = phantoms.choice(field)
                    q = rec['quality']
                    price = round(clamp(rec['price'] + phantoms.gauss(0, .7), rules.price_min, rules.price_max), 2)
                    rep = clamp(50. + phantoms.gauss(0, 12), 0, 100)
                    meals = min(rules.capacity(q), int(rec['meals']))
                    placements[spot].append((price, q, rep, meals))
                    named.append({'id': 'phantom', 'spot': spot, 'price': price, 'quality': q,
                                  'reputation': rep, 'meals': meals})
                    active += min(1., meals * rules.costs[q] / 250.)
            crowds, models = {}, {}
            for spot in rules.spots:
                average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                conditional = weather_factor(weather, spot) / max(average, .1)
                correction = self.learner.crowd_multiplier(spot, weather)
                width = math.sqrt(3.) * self.learner.crowd_noise(spot, weather)
                crowds[spot] = (counts[spot] * conditional * correction * rng.uniform(1-width, 1+width)
                                * thinned.get(spot, 1.))
                models[spot] = customer_model(spot, kinds, self.learner.sample(spot, rng))
            scenarios.append((placements, crowds, models, active))
            self.scenario_rivals.append(named)
        return scenarios

    def evaluate(self, state, rules, spot, quality, price, scenarios, fixed_meals=None):
        rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
        cost = rules.costs[quality]
        cap = rules.capacity(quality)
        flags = features(state.get('features'))
        overheads = Overheads(state, spot, flags['salvage'])
        if overheads.capacity is not None:
            cap = min(cap, overheads.capacity)  # each employee serves 50 meals a day
        charged = overheads.charged(price)
        # Expected riot damage near tomorrow's game (score only). The final policy prices it through
        # its rule adapter instead, so it is not counted twice.
        riots = 0. if self.final_active else riot_cost(state, spot)
        distributions = [demand_levels(state, rules, spot, quality, charged, scenario) for scenario in scenarios]
        factor, adjust = 1., 0.
        if self.final_active:
            factor = self.final_policy.demand_factor(state, spot, quality, price, charged, rep)
            adjust = self.final_policy.score_adjust(state, spot)
            if factor != 1.:
                distributions = [tuple(x * factor for x in d) for d in distributions]
        remaining = max(0, int(state.get('days', 30)) - int(state.get('order_for_day', 1)))
        rep_delta = clamp(rep + REP_CHANGE[quality], 0, 100) - rep
        best = None
        inventory_options = []
        n = len(distributions)
        quiet = sum(1 for placement, _, _, _ in scenarios if not placement[spot]) / max(1, n)
        # Every feasible integer inventory is considered, including zero.
        for meals in (range(cap + 1) if fixed_meals is None else [min(int(fixed_meals), cap)]):
            contribution = min(1., meals * cost / 250.)
            left = 0 if contribution <= .5 else 1
            fraction = contribution * 2 if left == 0 else (contribution - .5) * 2
            demand = [d[left] * (1-fraction) + d[left+1] * fraction for d in distributions]
            sales = [min(meals, x) for x in demand]
            mean = sum(sales) / n
            profit = overheads.profit(mean, meals, price, quality, cost)
            outcomes = [overheads.profit(x, meals, price, quality, cost) for x in sales]
            loss = sum(max(0., -p) for p in outcomes) / n
            upside = sum(max(0., p - profit) for p in outcomes) / n
            # Small, bounded future benefit: no incentive to lose hundreds buying reputation.
            sale_probability = sum(x > .1 for x in sales) / n
            future = .006 * rules.values[quality] * rep_delta * mean * min(remaining, 6) * .15 * sale_probability
            future = clamp(future, -25., 25.)
            exploration = 3. * min(1., remaining / 15.) * (1-self.learner.confidence(spot)) * sale_probability
            post = self.posture
            roam = post['explore'] * quiet * sale_probability if spot not in self._home else 0.
            score = (profit - self.risk * post['risk_scale'] * loss + future + exploration - adjust - riots
                     + post['upside'] * upside + roam)
            if meals and self.quiet_bonus:
                score += self.quiet_bonus * quiet
            candidate = {'score': score, 'profit': profit, 'sales': mean, 'loss': loss,
                        'demand': sum(demand) / n, 'calibration': factor, 'score_adjust': adjust, 'charged': charged,
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
                        and extra_sales * charged >= .90 * extra * cost
                        and option['profit'] >= best['profit'] - tolerance
                        and option['score'] >= best['score'] - tolerance
                        and option['loss'] <= best['loss'] + 2.):
                    supported.append(option)
            if supported:
                best = max(supported, key=lambda r: r['order']['meals'])
                best = dict(best, inventory_cushion=best['order']['meals']-base_meals,
                            base_meals=base_meals)
        return best

    def ceiling(self, rules, spot, quality, rep):
        cap = self.low_price_caps[quality] * self.spot_price_lift.get(spot, 1.) if self.small_crowds else rules.price_max
        return min(rules.price_max, self.price_cap or rules.price_max, cap,
                   reputation_value(quality, rep, rules.values))

    def order_value(self, state, order, hit=None):
        """Expected result of this exact order under the usual scenarios (same seed as decide), with the
        reputations of some rivals scaled by `hit` ({truck id: factor}). Also records, in last_overlap,
        how often each rival shares the order's spot."""
        state = without_null_counts(state)
        if state.get('order_for_day') is None:
            state = dict(state, order_for_day=(state.get('forecast') or {}).get('day') or int(state.get('day', 0)) + 1)
        rules = Rules.read(state)
        counts = self.forecast(state, rules)
        digest = hashlib.sha256(repr((self.seed, self.market, state.get('order_for_day'))).encode()).digest()
        self._rep_hit = dict(hit or {})
        try:
            scenarios = self.scenarios(state, rules, counts, random.Random(int.from_bytes(digest[:8], 'big')))
        finally:
            self._rep_hit = {}
        spot = order['spot']
        seen = {}
        for named in self.scenario_rivals:
            for rival in {r['id'] for r in named if r.get('spot') == spot}:
                seen[rival] = seen.get(rival, 0) + 1
        self.last_overlap = {rival: count / max(1, len(self.scenario_rivals)) for rival, count in seen.items()}
        return self.evaluate(state, rules, spot, order['quality'], order['price'], scenarios, fixed_meals=order['meals'])

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
        state = without_null_counts(state)
        self.register_new_spots(state)
        self.observe(state)
        active_features = features(state.get('features'))
        self.learner.activate(active_features)
        if state.get('order_for_day') is None:
            next_day = (state.get('forecast') or {}).get('day') or int(state.get('day', 0))+1
            state = dict(state, order_for_day=next_day)
        rules = Rules.read(state)
        counts = self.forecast(state, rules)
        me_id = identity(state.get('you') or {})
        self.posture = (posture_mod.read(state, self.cash_history, (state.get('you') or {}).get('id'))
                        if posture_mod.enabled() else dict(posture_mod.NEUTRAL))
        self._home = {r['spot'] for r in self.history.get(me_id, [])[-5:] if 'spot' in r}
        if self.final_active:
            self.final_policy.begin(self, state)
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
                ceiling = self.ceiling(rules, spot, quality, rep)
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
                legal = [price for price in sorted(prices) if low <= price <= ceiling]
                if self.final_active and len(scenarios) >= 12 and len(legal) > 4:
                    # Staged search keeps six spots inside the 2-second window: score every price on a
                    # thin scenario subset, then re-score only the best two on all scenarios.
                    thin = scenarios[::max(1, len(scenarios) // 6)]
                    coarse = sorted(((self.evaluate(state, rules, spot, quality, price, thin)['score'], price)
                                     for price in sorted(set(legal[::2]) | {legal[-1]})), reverse=True)
                    legal = sorted({price for _, price in coarse[:2]})
                candidates = [self.evaluate(state, rules, spot, quality, price, scenarios) for price in legal]
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
            ceiling = self.ceiling(rules, spot, quality, rep)
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
                                       'calibration': best.get('calibration', 1.), 'charged': best.get('charged'),
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
                            'posture': dict(self.posture), 'chosen': best, 'location_rule': {k: v for k, v in restriction.items() if k != 'best'},
                            'scenario_count': len(scenarios), 'prediction_checks': self.prediction_checks,
                            'features': active_features,
                            'rivals': {identity(t): self.opponents.summary(identity(t),
                                self.history.get(identity(t), []), state.get('forecast') or {},
                                self.history, rules.spots) for t in state.get('trucks') or []
                                if identity(t) != identity(state.get('you') or {})}}
        # Capture the same scenarios used for selection. This adds observability,
        # not another sample or a change to order scoring.
        evidence_model = dict(self.dump(), planner_config={
            'seed': self.seed, 'samples': self.samples, 'risk': self.risk,
            'small_crowds': self.small_crowds, 'price_cap': self.price_cap,
            'quiet_bonus': self.quiet_bonus, 'low_price_caps': dict(self.low_price_caps)})
        self.forecast_record = None
        try:
            self.forecast_record = capture(state, order, scenarios, self.scenario_rivals, evidence_model, self.diagnostics)
            self.pending['prediction']['forecast_id'] = self.forecast_record['forecast_id']
            self.pending['prediction']['rules_digest'] = self.forecast_record['rules_digest']
            self.pending['prediction']['regime_id'] = self.forecast_record['regime_id']
            self.diagnostics['risk'] = self.forecast_record['risk']
            self.diagnostics['forecast_id'] = self.forecast_record['forecast_id']
        except Exception as error:
            # Observability must never replace a valid order with a fallback.
            self.diagnostics['forecast_capture_error'] = type(error).__name__
        self.diagnostics['elapsed'] = time.perf_counter()-started
        return order


# ======================= history.py (verbatim; package imports stripped) =======================
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


# ======================= bot.fallback (verbatim) =======================
def fallback(state):
    """Bound loss if policy fails: assess all spots using forecast less rival stock."""
    state = without_null_counts(state)
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


PRIOR_JSON = r'''{"version":1,"weights":{"highst":[-15.716097405513716,-15.82300568308542,-16.0,-15.68,-15.68,-15.856099069192558,-15.68,-15.68,-15.715340906743943,-13.570780653663864,-14.90932766429571,-15.68,-8.331213302170623,-12.904039086005872,-15.68,-3.418775126867456,-12.309534890755446,-15.68,-15.68,-9.359099654256536,-0.07616757312171257,-14.367712352768036,0.0,-4.136935100251859,-3.146782968073255,-0.0903431845559753,-10.720083210896593,-15.68,-15.68,-13.311066570154585,-15.68,-15.247364585412162,-2.5588108965259027,-15.577882306067043,-4.352046346633972,-3.2966095298485385,-15.68,-15.68,-15.57175370480364,-15.68,-15.68,-13.559072709298738,-15.68,-15.516989407490609,-5.549828647093943],"oval":[-16.0,-14.92501496184862,-15.081601803469939,-16.0,-14.448836129127534,-15.732474044646406,-15.267250654393363,-13.828159745196144,-16.0,-16.0,-15.144247137795942,-14.91294982080811,-16.0,-9.574481261394599,-14.84563328933995,-16.0,-4.292818203017029,-14.83080223813381,-16.0,-5.64451763690759,-8.440917597023365,-16.0,0.0,-5.775428859065305,-16.0,-0.8393275757479551,-12.14166645649772,-16.0,-11.210832907353627,-11.463590933275766,-16.0,-12.777782906955219,-14.907862920520278,-16.0,-14.809760877498748,-15.053582553923206,-16.0,-15.181077829364567,-14.88827440274184,-16.0,-15.175067843436429,-14.89210210157052,-16.0,-15.07884242937525,-14.953930711803089],"union":[-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-4.298862434804317,-3.8508417788051457,-15.879732514071783,-0.11847353010235112,-11.565083896532665,-16.0,-3.1572944516257437,-15.969423127148747,-16.0,-15.781062999342693,-12.234726693741274,-1.9721742163720768,-13.061681207442223,0.0,-7.095485446373006,-1.5584867922808707,-0.32956001166248666,-16.0,-15.996817730820517,-16.0,-8.147384656908176,-15.772940975339836,-13.815480256093032,-3.8050359016020145,-15.174798901981303,-2.5740466073886474,-5.95479707669366,-16.0,-16.0,-11.66183821781964,-15.959155680923342,-16.0,-8.098608961225356,-15.752829441201353,-13.937180212129208,-5.139228288772005],"northres":[-15.736686293296634,-15.736686293296634,-15.736686293296634,-15.7341925994597,-15.7341925994597,-15.7341925994597,-15.69403399987713,-15.69403399987713,-15.69403399987713,-15.736625197130722,-15.736625197130722,-15.736625197130722,-7.71363235955467,-7.513924271840447,-11.86027416527504,-2.571781374076156,-4.031490914883694,-10.436024729759087,-6.7519153088719515,-3.0332943763409883,-1.769400454310105,-3.272537833582349,0.0,-0.4791867503495321,-2.8397071856106946,-1.4390041851918687,-4.261299231181433,-15.696637230840905,-14.84596971441801,-11.389809281362092,-16.0,-14.36725464167959,-11.92932685160532,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0],"thompson":[-15.815969862338097,-15.815969862338097,-15.815969862338097,-15.621548388103124,-15.621548388103124,-15.621548388103124,-15.45758538508982,-15.45758538508982,-15.45758538508982,-15.666801504233367,-15.666801504233367,-15.666801504233367,-15.453205834955597,-15.453205834955597,-15.453205834955597,-0.5944320705626251,-5.719041293534741,-12.020919594901876,-14.934952823838904,-15.446243586020568,-15.446243586020568,0.0,-4.7251101455550515,-11.12979513325659,-14.878967913656433,-15.192682167150915,-15.192682167150915,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0,-16.0],"stadium":[-15.454305465763744,-14.851299536099882,-15.757316924638783,-9.932805246299697,-10.088661627203404,-13.232122068849577,-6.99309113164204,-8.126532985175388,-11.778163312987687,-6.0582918534448424,-4.365457513744483,-5.99465844246056,-3.294762719558404,-2.316008141780043,-4.575822809761454,-2.2880591249451805,-2.0504080046452406,-4.970333047437496,-3.8136615103462583,-1.4054372406313673,-2.233246334449242,-2.0617578012868023,-0.020255092592381743,-1.434244066361976,-1.2614705828136765,0.0,-2.11431359906325,-7.107537411971021,-4.721815685440683,-5.266243721160507,-5.199470817748324,-2.89727988420013,-3.6741702873443916,-3.82698790676987,-1.7441140263048251,-3.144308629407937,-15.96353314801916,-13.85065009060152,-14.400870650633399,-11.724180470755755,-9.375207512070899,-9.924063448107649,-8.469207795639612,-6.2725618359524065,-7.136875867133414]},"counts":{"highst":408,"oval":105,"union":420,"northres":132,"thompson":420,"stadium":85},"crowd":{"highst":{"cloudy":[0.9920529094548451,136],"clear":[1.0423013075445515,161],"rain":[0.9693263295076181,88],"cold":[0.9583906895273422,35]},"oval":{"cloudy":[0.9925754662902104,136],"clear":[1.103528542474442,161],"rain":[0.9718379258934416,88],"cold":[0.9357123646185133,35]},"union":{"cloudy":[1.0266563393481678,136],"clear":[1.0251139900648758,161],"rain":[0.9273537379157114,88],"cold":[0.9456967329737556,35]},"northres":{"cloudy":[1.105774281561361,136],"clear":[1.077607613087599,161],"rain":[0.8588201149247786,88],"cold":[1.0385397103193477,35]},"thompson":{"cloudy":[1.0888682622798378,136],"clear":[1.0512911320879488,161],"rain":[0.8635793393547397,88],"cold":[0.9511298766292379,35]},"stadium":{"cloudy":[0.9680603763738579,136],"clear":[1.0988598798169666,161],"rain":[0.999566182184343,88],"cold":[0.9947582038962325,35]}},"rows":420}'''

FINAL_PRIORS_JSON = r'''{"calibration":{"buckets":{"premium":{"0.00-0.50":{"censored":4,"mean":0.903803,"n":5},"0.50-0.60":{"censored":1,"mean":1.0,"n":0}},"standard":{"0.00-0.50":{"censored":1,"mean":1.0,"n":0},"0.50-0.60":{"censored":2,"mean":0.861738,"n":5},"0.70-0.80":{"censored":0,"mean":0.628039,"n":1}}},"global":{"censored":8,"mean":0.859613,"n":11,"ratio_bins":["0.00-0.50","0.50-0.60","0.60-0.70","0.70-0.80","0.80-0.90","0.90-1.00","1.00-1.25","1.25-10.00"]}},"controller":{"breadth":{"0.9":{"mean":0,"n":0,"var":1},"1.0":{"mean":0,"n":0,"var":1},"1.1":{"mean":0,"n":0,"var":1}},"price_aggr":{"0.9":{"mean":0,"n":0,"var":1},"1.0":{"mean":0,"n":0,"var":1},"1.1":{"mean":0,"n":0,"var":1}},"risk":{"0.9":{"mean":0,"n":0,"var":1},"1.0":{"mean":0,"n":0,"var":1},"1.1":{"mean":0,"n":0,"var":1}}},"meta":{"calibration_rows":19,"days":570,"source_modes":{"live":30,"practice":540}},"version":1}'''



def capture(state, order, scenarios, named_rivals, model, diagnostics=None):
    """Port: forecast_log.capture minus logging/digests. Only 'scenarios'[i]['sales'] feeds a decision (crew plan)."""
    rules = Rules.read(state)
    spot, quality, price, meals = (order[k] for k in ('spot', 'quality', 'price', 'meals'))
    flags = features(state.get('features'))
    overheads = Overheads(state, spot, flags['salvage'])
    contribution = min(1., meals*rules.costs[quality]/250.)
    left = 0 if contribution <= .5 else 1
    fraction = contribution*2 if left == 0 else (contribution-.5)*2
    rows = []
    for placements, crowds, models, active in scenarios:
        demands = demand_levels(state, rules, spot, quality, overheads.charged(price), (placements, crowds, models, active))
        demand = demands[left]*(1-fraction) + demands[left+1]*fraction
        sales = min(meals, demand)
        rows.append({'demand': demand, 'sales': sales,
                     'profit': overheads.profit(sales, meals, price, quality, rules.costs[quality])})
    return {'order': dict(order), 'scenarios': rows, 'forecast_id': None, 'rules_digest': None, 'regime_id': None,
            'risk': None}


def _fill_you(state):
    """Port-side: our replay states only carry the public truck row as `you`. The live owner view also has
    staff and last_spot, which change the plan (wages per 50 meals, moving costs). Rebuild them from public data."""
    you = state.get('you')
    if not isinstance(you, dict):
        return state
    state = dict(state, you=dict(you))
    you = state['you']
    flags = state.get('features') or {}
    row = next((t for t in state.get('trucks') or [] if t.get('id') == you.get('id')), {})
    if flags.get('staff') and not isinstance(you.get('staff'), dict) and row.get('wage') is not None:
        n = int(row.get('employees') or 0)
        you['staff'] = {'employees': n, 'target': n or 4, 'wage': row.get('wage'), 'capacity': 50 * n}
    if 'last_spot' not in you:
        mine = (state.get('last_day') or {}).get('you') or {}
        if mine.get('spot'):
            you['last_spot'] = mine['spot']
    votes = ((state.get('politics') or {}).get('election') or {}).get('votes')
    if 'vote' not in you and isinstance(votes, dict) and you.get('id') in votes:
        you['vote'] = votes[you['id']]
    return state


def make(seed=0, **opts):
    """seed is ignored (live Brain uses seed=42, FinalPolicy seed 0). opts: samples (32), small_crowds (True),
    use_prior (True), warm_states (list of states to replay like history.warm_start), profile (dict standing in for
    data/profile.json), force_final (bot --profile final), elapsed (pin the decision time FinalPolicy.begin sees),
    fill_you (True: rebuild you.staff/last_spot/vote from public data when missing)."""
    opts = dict(DEFAULT_OPTS, **opts)
    PROFILE.clear()
    PROFILE.update(opts.get('profile') or {})
    learner = Learner.from_dict(json.loads(PRIOR_JSON)) if opts.get('use_prior', True) else Learner()
    small = opts.get('small_crowds', True)
    brain = Brain(learner=learner, samples=opts.get('samples', 32), small_crowds=small,
                  risk=.08 if small else .15, spot_price_lift={'thompson': 1.39},
                  **({'target_spot': opts['target_spot']} if opts.get('target_spot') else {}))
    attach_policy(brain)
    states = list(opts.get('warm_states') or [])
    states.sort(key=lambda s: (str(s.get('mode')), str(s.get('market')), (s.get('last_day') or {}).get('day', 0)))
    def warm(st):  # history.warm_start for one saved public state (observe only, no decision)
        st = _fill_you(st) if opts.get('fill_you', True) else st
        key = [st.get('mode'), st.get('market')]
        if brain.market is not None and brain.market != key:
            return
        day = (st.get('last_day') or {}).get('day')
        if day is None or (brain.seen is not None and day <= brain.seen):
            return
        brain.observe(st)
        forecast = st.get('forecast') or {}
        next_day = forecast.get('day') or st.get('order_for_day')
        if next_day is not None and next_day > day:
            brain.pending = {'day': next_day, 'forecast': forecast}
            if 'features' in st:
                brain.pending['features'] = st['features']

    for st in states:
        warm(st)
    run = {'pending_key': None, 'pending_order': None, 'pending_features': None, 'blocked': None,
           'voted': set(), 'staffed': set(), 'complained': set(), 'out_key': None, 'out': None}
    force_final = bool(opts.get('force_final'))

    def profile_wanted(state):  # bot.Runner.profile_wanted
        if os.environ.get('NIGHTSHIFT_PROFILE') == 'baseline' or PROFILE.get('profile') == 'baseline':
            return 'baseline'
        return 'final' if (state.get('mode') in ('final', 'showcase') or force_final) else 'baseline'

    def apply_profile(state):  # bot.Runner.apply_profile (checkpoint paths dropped)
        fp = brain.final_policy
        want = profile_wanted(state) == 'final'
        if want != brain.final_active:
            brain.use_final(want)
        if state.get('mode') in ('final', 'showcase'):
            fp.reset_for_final()

    def policy(state):
        if opts.get('fill_you', True):
            state = _fill_you(state)
        apply_profile(state)
        key = [state.get('mode'), state.get('market'), state.get('order_for_day')]
        active_features = features(state.get('features'))
        if key == run['out_key'] and active_features == run['pending_features'] and run['out'] is not None:
            return dict(run['out'])
        if key == run['blocked']:
            return {'spot': (run['pending_order'] or {}).get('spot', 'oval'), 'quality': 'standard', 'price': 0.,
                    'meals': 0, 'wrecked': True}
        if key != run['pending_key'] or active_features != run['pending_features']:
            try:
                order = brain.decide(state)
            except Exception:
                order = (fallback(state, getattr(brain, 'target_spot', None))
                         if 'target' in fallback.__code__.co_varnames else fallback(state))
            if opts.get('elapsed') is not None:
                brain.diagnostics['elapsed'] = float(opts['elapsed'])
            run['pending_key'], run['pending_order'], run['pending_features'] = key, order, active_features
        fp = brain.final_policy
        extras = {}
        if brain.final_active:
            try:
                extras = fp.order_extras(state, run['pending_order'])
            except Exception:
                extras = {}
        order = dict(run['pending_order'], **extras)
        if brain.final_active and fp.is_wrecked(state):
            run['blocked'] = key
            return dict(order, meals=0, wrecked=True)
        out = dict(order)  # assume HTTP 200; then Runner: vote, staff, complain
        election = (state.get('politics') or {}).get('election') or {}
        vkey = (state.get('mode'), state.get('market'), election.get('id'))
        if election.get('id') and vkey not in run['voted']:
            run['voted'].add(vkey)
            try:
                choice = choose_vote(state, order, brain.game_day_rate(), brain.protest_rate())
                if choice and not (state.get('you') or {}).get('vote'):
                    out['vote'] = choice
            except Exception:
                pass
        skey = tuple(key)
        if skey not in run['staffed']:
            run['staffed'].add(skey)
            try:
                plan = brain.crew_plan(state, order)
                if plan and plan['change']:
                    out['staff'] = {'staff': plan['headcount'], 'wage': plan['wage']}
            except Exception:
                pass
        if brain.final_active and skey not in run['complained'] and complaints_enabled(state.get('mode')):
            run['complained'].add(skey)
            try:
                decision = fp.complaint_decision(state, brain, order)
                out['complaint_info'] = {k: decision.get(k) for k in ('file', 'target', 'reason', 'gain', 'ev', 'overlap')}
                if decision.get('file'):
                    out['complaint'] = decision['target']
                    fp.complaints.record(state, decision['target'], {})  # response unknown: assume not caught
            except Exception:
                pass
        run['out_key'], run['out'] = key, dict(out)
        return out

    policy.brain = brain
    policy.warm = warm  # validate.py warms persistent memory with this (fast, like history.warm_start)
    return policy
