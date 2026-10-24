"""Unit tests. Run: python3 -m unittest discover -s tests -v"""
import copy
import json
import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw import engine  # noqa: E402
from ftw.api import NetError  # noqa: E402
from ftw.localsim import Market  # noqa: E402
from ftw.baselines import Adaptive, house_bots  # noqa: E402
from ftw.rules import VALUES, Rules, rep_after  # noqa: E402
from ftw.runner import Bot  # noqa: E402
from ftw.strategy import Brain  # noqa: E402


def sample_state(day=1, market=1, mode="practice"):
    m = Market("t", [("me", "me", None)] + house_bots())
    fc = m._forecast(day)
    st = m._state("me", day, fc)
    st["mode"], st["market"] = mode, market
    return st


class EngineTests(unittest.TestCase):
    def test_min_n_D_identity(self):
        """Our sales with n meals equal min(n, sales with full stock) under full simulation."""
        rng = random.Random(1)
        for trial in range(200):
            n = rng.randint(0, 150)
            lim = [rng.lognormvariate(2.4, 0.3) for _ in range(n)]
            mq = [rng.choice([0, 1, 2]) for _ in range(n)]
            rivals = []
            for k in range(rng.randint(0, 3)):
                q = rng.choice(["cheap", "standard", "premium"])
                rivals.append(engine.Truck("r%d" % k, q, round(rng.uniform(5, 14), 2), rng.uniform(20, 90),
                                           rng.randint(5, 90), VALUES))
            q = rng.choice(["cheap", "standard", "premium"])
            p = round(rng.uniform(5, 14), 2) + 0.003  # avoid exact ties
            rep = rng.uniform(20, 90)
            full = engine.simulate_spot(lim, mq, rivals + [engine.Truck("me", q, p, rep, 200, VALUES)])["me"]
            sc = engine.SpotScenario(lim, mq, rivals)
            me = engine.Truck("me", q, p, rep, 200, VALUES)
            self.assertEqual(sc.demand(me.qi, p, me.value, 200), full)
            for cap in (0, 5, 30, 90):
                sim = engine.simulate_spot(lim, mq, rivals + [engine.Truck("me", q, p, rep, cap, VALUES)])["me"]
                self.assertEqual(sim, min(cap, full))

    def test_rival_sellout_frees_customers(self):
        # rival is better value but has 3 meals; remaining eligible customers come to us
        lim, mq = [20.0] * 10, [0] * 10
        rival = engine.Truck("r", "premium", 5.0, 50, 3, VALUES)
        sc = engine.SpotScenario(lim, mq, [rival])
        me = engine.Truck("me", "standard", 9.0, 50, 200, VALUES)
        self.assertEqual(sc.demand(me.qi, 9.0, me.value, 200), 7)

    def test_negative_value_and_quality(self):
        sc = engine.SpotScenario([30.0] * 5, [2] * 5, [])
        self.assertEqual(sc.demand(1, 5.0, 7.0, 133), 0)   # all need premium
        sc = engine.SpotScenario([30.0] * 5, [0] * 5, [])
        self.assertEqual(sc.demand(0, 9.0, -1.0, 200), 0)  # value below zero

    def test_best_quantity(self):
        self.assertEqual(engine.best_quantity([0, 50, 100, 100], 10, 4, 133)[0], 100)
        self.assertEqual(engine.best_quantity([10] * 8, 3.0, 3.75, 133)[0], 0)  # price below cost
        self.assertEqual(engine.best_quantity([], 10, 4, 133)[0], 0)


class RulesTests(unittest.TestCase):
    def test_budget_and_bounds(self):
        r = Rules({"budget": 500, "costs": {"cheap": 2.5, "standard": 3.75, "premium": 5.5}})
        self.assertEqual(r.max_meals("cheap"), 200)
        self.assertEqual(r.max_meals("standard"), 133)
        self.assertEqual(r.max_meals("premium"), 90)
        o = r.validate({"spot": "nowhere", "price": 99, "quality": "gold", "meals": 999})
        self.assertIn(o["spot"], r.spots)
        self.assertEqual(o["price"], 30.0)
        self.assertEqual(o["quality"], "standard")
        self.assertLessEqual(o["meals"] * 3.75, 500)
        self.assertEqual(r.validate({"price": -5, "meals": -3})["price"], 1.0)
        self.assertEqual(r.validate({"price": "x", "meals": "y"})["meals"], 0)
        self.assertEqual(r.validate({"price": 9.999})["price"], 9.99)

    def test_reputation_limits(self):
        self.assertEqual(rep_after(98, "premium", 5), 100)
        self.assertEqual(rep_after(1, "cheap", 5), 0)
        self.assertEqual(rep_after(40, "premium", 0), 40)   # no sale, no change


class BrainTests(unittest.TestCase):
    def test_legal_and_fast(self):
        b = Brain(seed=3)
        st = sample_state()
        t0 = time.perf_counter()
        o = b.decide(st)
        self.assertLess(time.perf_counter() - t0, 0.6)
        r = Rules(st["rules"])
        self.assertEqual(o, r.validate(o))

    def test_missing_fields(self):
        b = Brain(seed=3)
        for mutate in ("forecast", "last_day", "trucks", "rules", "you"):
            st = sample_state()
            st[mutate] = None
            o = b.decide(st)
            self.assertEqual(o, Rules(None).validate(o), mutate)
        o = b.decide({})
        self.assertEqual(o, Rules(None).validate(o))
        st = sample_state()
        st["forecast"] = {"weather": "garbage", "events": [None, 3], "expected_customers": {"oval": "x"}}
        o = b.decide(st)
        self.assertEqual(o, Rules(None).validate(o))

    def test_live_overrides_do_not_reach_final(self):
        cfg = {"live_overrides": {"avoid_spots": ["oval", "thompson", "union", "stadium", "highst"],
                                  "only_quality": "cheap", "fixed_price": 3.33}}
        live = sample_state()
        live["mode"] = "live"
        o = Brain(seed=3, cfg=cfg).decide(live)
        self.assertEqual((o["spot"], o["quality"], o["price"]), ("northres", "cheap", 3.33))
        fin = sample_state()
        fin["mode"] = "final"
        o = Brain(seed=3, cfg=cfg).decide(fin)
        self.assertNotEqual(o["price"], 3.33)

    def test_features_and_new_spot(self):
        from ftw.strategy import nearest, LAYOUT
        self.assertEqual(nearest("highst", {"northres", "oval", "union"}), "northres")
        self.assertEqual(nearest("stadium", {"thompson", "oval"}), "thompson")
        for feats in ({"salvage": True}, {"fair_choice": True}, {"walk_over": True},
                      {"salvage": True, "fair_choice": True, "walk_over": True}, "garbage"):
            st = sample_state()
            st["features"] = feats
            o = Brain(seed=3).decide(st)
            self.assertEqual(o, Rules(st["rules"]).validate(o))
        st = sample_state()  # a spot opens overnight
        st["rules"] = dict(st["rules"], spots=st["rules"]["spots"] + ["rpac"])
        st["forecast"]["expected_customers"]["rpac"] = 400
        st["spot_info"] = {"rpac": {"x": 0, "z": 0}}
        st["features"] = {"walk_over": True}
        o = Brain(seed=3).decide(st)
        self.assertEqual(o["spot"], "rpac")  # a huge empty new spot is taken
        self.assertEqual(o, Rules(st["rules"]).validate(o))

    def test_salvage_quantity(self):
        ds = [20, 40, 60, 80]
        n0 = engine.best_quantity(ds, 9.0, 5.5, 90)[0]
        n1 = engine.best_quantity(ds, 9.0, 5.5, 90, salvage=0.25)[0]
        self.assertGreaterEqual(n1, n0)
        n, prof, sold = engine.best_quantity([50], 9.0, 5.5, 90, salvage=0.25)
        self.assertAlmostEqual(prof, 9.0 * 50 - 5.5 * 50)

    def test_deadline_fallback(self):
        b = Brain(seed=3)
        o = b.decide(sample_state(), deadline=0.0)
        self.assertEqual(o, Rules(None).validate(o))

    def test_reproducible(self):
        st = sample_state()
        self.assertEqual(Brain(seed=5).decide(copy.deepcopy(st)), Brain(seed=5).decide(copy.deepcopy(st)))

    def test_day_processed_once_and_market_reset(self):
        b = Brain(seed=1)
        m = Market("x", [("me", "me", Adaptive(seed=1))] + house_bots(), days=3)
        m.run()
        st = m._state("me", 4, None, phase="over")
        st["mode"], st["market"] = "final", 1
        b.observe(st)
        act = {tid: list(r.active) for tid, r in b.rivals.items()}
        for _ in range(3):
            b.observe(copy.deepcopy(st))  # repeated versions of the same day
        self.assertEqual(act, {tid: list(r.active) for tid, r in b.rivals.items()})
        st2 = copy.deepcopy(st)
        st2["market"] = 2  # same day number, new market => processed again, history reset
        b.observe(st2)
        self.assertEqual(b.market_key, ("final", 2))
        self.assertEqual(b.processed, {st["last_day"]["day"]})

    def test_zero_sales_and_sellout_states(self):
        b = Brain(seed=2)
        st = sample_state(day=2)
        st["last_day"] = {"day": 1, "weather": "rain", "events": [], "spots": {
            "union": {"customers": 50, "hungry": 0, "trucks": [
                {"id": "h1", "price": 10.5, "quality": "standard", "meals": 50, "sold": 50}]},
            "oval": {"customers": 10, "hungry": 10, "trucks": [
                {"id": "me", "price": 13, "quality": "premium", "meals": 90, "sold": 0}]}},
            "you": {"spot": "oval", "sold": 0, "meals": 90, "profit": -495}}
        o = b.decide(st)
        self.assertEqual(o, Rules(None).validate(o))


class FakeClient:
    """Scripted API: a final-like sequence with day numbers reused across markets."""
    def __init__(self, states, fail_first_order=False):
        self.states = states
        self.i = 0
        self.orders = []
        self.fail = fail_first_order

    def state(self, base=None, since=None, timeout=None):
        if self.i >= len(self.states):
            raise KeyboardInterrupt
        st = self.states[self.i]
        self.i += 1
        if st == "neterr":
            raise NetError("boom")
        return 200, st

    def order(self, order, base=None, timeout=None):
        if self.fail:
            self.fail = False
            raise NetError("timeout")
        self.orders.append(order)
        return 200, {"order": order, "for_day": None}


class RunnerTests(unittest.TestCase):
    def test_one_order_per_market_day_and_reuse(self):
        states = []
        v = 0
        for market in (1, 2):
            for day in (1, 2):
                st = sample_state(day=day, market=market, mode="final")
                v += 1
                st["version"] = v
                st["phase_ends_at"], st["server_time"] = "2026-10-24T12:30:02Z", "2026-10-24T12:30:00Z"
                states.append(st)
                dup = copy.deepcopy(st)
                v += 1
                dup["version"] = v  # same night, new version: must not order twice
                states.append(dup)
        states.insert(1, "neterr")
        client = FakeClient(states, fail_first_order=True)
        bot = Bot(client, memory_path=None, quiet=True, seed=1)
        try:
            bot.run()
        except KeyboardInterrupt:
            pass
        self.assertEqual(len(client.orders), 4)  # (final,1,1) (final,1,2) (final,2,1) (final,2,2)
        self.assertEqual(len(bot.ordered), 4)
        for o in client.orders:
            self.assertEqual(o, Rules(None).validate(o))

    def test_no_order_in_day_phase(self):
        st = sample_state()
        st["phase"], st["orders_open"] = "day", False
        client = FakeClient([st])
        bot = Bot(client, memory_path=None, quiet=True, seed=1)
        try:
            bot.run()
        except KeyboardInterrupt:
            pass
        self.assertEqual(client.orders, [])

    def test_memory_roundtrip_and_isolation(self):
        import tempfile
        d = tempfile.mkdtemp()
        live, prac = os.path.join(d, "memory_live.json"), os.path.join(d, "memory_practice.json")
        st = sample_state(day=2, mode="live")
        st["last_day"] = {"day": 1, "spots": {"union": {"customers": 9, "hungry": 0, "trucks": [
            {"id": "h1", "price": 10.5, "quality": "standard", "meals": 9, "sold": 9}]}}}
        bot = Bot(FakeClient([]), memory_path=live, quiet=True, seed=1)
        bot.brain.observe(st)
        bot._save()
        bot.writer.flush()
        with open(live) as f:
            mem = json.load(f)
        self.assertTrue(any(r["tid"] == "h1" for r in mem["rivals"]))
        self.assertEqual(mem["market"]["processed"], [1])
        self.assertFalse(os.path.exists(prac))
        bot2 = Bot(FakeClient([]), memory_path=live, quiet=True, seed=1)
        self.assertEqual(bot2.brain.processed, {1})  # restart does not re-ingest day 1


if __name__ == "__main__":
    unittest.main()
