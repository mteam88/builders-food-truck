"""Predict rivals by running their own decision code (vendored adapters in ftw/adapters/).

Each adapter runs in its own plain subprocess (ftw/rival_worker.py): scrubbed environment (never our
key), a temporary working directory, no network use by design (only the pure decision function is
called), and a deadline. Workers live for the whole run so each rival keeps its own memory, and they
see every night's state rebuilt the way the server shows it to that rival (you = that truck).
"""
import json
import os
import select
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.join(HERE, "adapters")

# truck id -> (adapter file in ftw/adapters, make() kwargs, sample seeds), from ftw/roster.py
try:
    from .roster import RIVALS as _RIVALS, ALT_PORTS as _ALT
except ImportError:      # imported as a plain module (rival_worker)
    from roster import RIVALS as _RIVALS, ALT_PORTS as _ALT
ROSTER = {v[0]: (v[4], dict(v[5]), list(v[6])) for v in _RIVALS.values() if v[4]}
# truck id -> [(adapter, kwargs)]: other candidate ports, trusted by hit rate (roster.ALT_PORTS)
ALTS = {_RIVALS[n][0]: [(a, dict(kw)) for a, kw, _src in alts] for n, alts in _ALT.items() if n in _RIVALS}
TRUST_DECAY = 0.8       # per scored night: a rival that restarts on other code is re-learned within a few nights
TRUST_GAP = 1.0         # a candidate this far below the best one is not used even if it is the only one in time


def _num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def rival_view(state, tid, memo):
    """The state as truck `tid` sees it. memo[tid] keeps its last opened spot across nights."""
    row = next((t for t in state.get("trucks") or [] if isinstance(t, dict) and t.get("id") == tid), None)
    if row is None:
        return None
    ld = state.get("last_day")
    mine = None
    if isinstance(ld, dict):
        for spot, v in (ld.get("spots") or {}).items():
            for t in (v or {}).get("trucks") or []:
                if isinstance(t, dict) and t.get("id") == tid:
                    mine = (spot, t)
    m = memo.setdefault(tid, {"last_spot": None})
    last_you = {"status": "no_order", "profit": 0}
    if mine:
        spot, t = mine
        costs = (state.get("rules") or {}).get("costs") or {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
        meals, sold, price = int(_num(t.get("meals"))), int(_num(t.get("sold"))), _num(t.get("price"))
        cpm = _num(costs.get(t.get("quality")), 5.5)
        emp = int(_num(t.get("employees"), _num(row.get("employees"), 4)))
        wage = _num(row.get("wage"), 6.0)
        move = 0
        mc = (state.get("rules") or {}).get("move_costs") or {}
        if m["last_spot"] and m["last_spot"] != spot:
            move = _num((mc.get(m["last_spot"]) or {}).get(spot), 0)
        salvage = round((meals - sold) * cpm * 0.25 + 1e-9, 2)
        rev = round(sold * price, 2)
        last_you = {"status": "played", "spot": spot, "price": price, "quality": t.get("quality"), "meals": meals,
                    "sold": sold, "revenue": rev, "cost": round(meals * cpm, 2), "spoiled": meals - sold,
                    "salvage": salvage, "move_cost": move, "employees": emp, "capacity": emp * 50,
                    "wages": round(emp * wage, 2), "profit": round(rev - meals * cpm + salvage - move - emp * wage, 2)}
        m["last_spot"] = spot
    view = dict(state)
    emp = int(_num(row.get("employees"), 4))
    badge = row.get("badge") or {}
    view["you"] = {"id": tid, "truck_name": row.get("truck_name"), "company_name": row.get("company_name"),
                   "cash": row.get("cash"), "reputation": dict(row.get("reputation") or {}), "order": None,
                   "last_spot": m["last_spot"],
                   "staff": {"employees": emp, "target": emp, "wage": row.get("wage"), "capacity": emp * 50,
                             "idle_days": 0},
                   "news": [], "coupons": [], "coupons_dropped": [],
                   "wrecked_until_day": badge.get("until_day") if isinstance(badge, dict) else None,
                   "vote": ((state.get("politics") or {}).get("election") or {}).get("votes", {}).get(tid)
                   if isinstance((state.get("politics") or {}).get("election"), dict) else None}
    if isinstance(ld, dict):
        view["last_day"] = dict(ld, you=last_you)
    return view


class Worker:
    def __init__(self, tid, adapter, kwargs, seed):
        self.tid, self.seed = tid, seed
        self.args = (tid, adapter, kwargs, seed)
        self.answered = True        # False while a request is out and unanswered
        self.dir = tempfile.mkdtemp(prefix="ftw-rival-")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "HOME": self.dir,
               "LANG": "C.UTF-8"}
        cmd = [sys.executable, "-I", os.path.join(HERE, "rival_worker.py"), os.path.join(ADAPTERS, adapter),
               json.dumps(dict(kwargs, seed=seed))]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  cwd=self.dir, env=env, bufsize=0)
        self.buf = b""
        self.want = None

    def send(self, req_id, view):
        self.want = req_id
        self.answered = False
        try:
            self.p.stdin.write((json.dumps({"id": req_id, "state": view}) + "\n").encode())
        except (BrokenPipeError, OSError, ValueError):
            self.want = None

    def poll_lines(self):
        """Read whatever is available; return the answer for the current request, if any."""
        out = None
        try:
            chunk = os.read(self.p.stdout.fileno(), 1 << 20)
        except (BlockingIOError, OSError):
            return None
        if not chunk:
            return None
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") == self.want:
                out = msg
                self.answered = True
        return out

    def alive(self):
        return self.p.poll() is None

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


class Predictor:
    """predictor(state) -> {truck id: order dict or None (closed)}; rivals that miss the deadline are absent."""

    def __init__(self, budget=0.9, roster=None, alts=None):
        self.budget = budget
        self.roster = roster if roster is not None else ROSTER
        alts = alts if alts is not None else (ALTS if roster is None else {})
        self.workers = []
        self.memo = {}
        self.n = 0
        for tid, (adapter, kw, seeds) in self.roster.items():
            for k, (ad, kw_) in enumerate([(adapter, kw)] + list(alts.get(tid, []))):
                if not os.path.exists(os.path.join(ADAPTERS, ad)):
                    continue
                for s in (seeds if k == 0 else seeds[:1]):
                    try:
                        w = Worker(tid, ad, kw_, s)
                        w.cand = k
                        self.workers.append(w)
                    except Exception:
                        pass
        self.last_timing = {}
        self.trust = {}         # (tid, candidate) -> decayed hit score on the rival's actual orders
        self.said = {}          # (tid, candidate) -> (market, day, order) predicted last night
        self.used = {}          # tid -> candidate used tonight
        self.asked = None       # (market, day) of the last request, to score late answers

    def score(self, state):
        """Credit each candidate's prediction for last night against what the rival actually did."""
        ld = state.get("last_day")
        if not isinstance(ld, dict) or not self.said:
            return
        acts = {}
        for spot, v in (ld.get("spots") or {}).items():
            for t in (v or {}).get("trucks") or []:
                if isinstance(t, dict) and t.get("id"):
                    acts[t["id"]] = (spot, _num(t.get("price")))
        for (tid, k), (market, day, o) in list(self.said.items()):
            if market != state.get("market") or day != ld.get("day"):
                continue
            a = acts.get(tid)
            if a is None:
                hit = 1.0 if o is None else 0.0
            elif not isinstance(o, dict):
                hit = 0.0
            else:
                hit = (o.get("spot") == a[0]) + 0.5 * (abs(_num(o.get("price")) - a[1]) <= 0.25)
            self.trust[(tid, k)] = TRUST_DECAY * self.trust.get((tid, k), 0.0) + hit
        self.said = {}

    def __call__(self, state, deadline=None):
        self.n += 1
        t0 = time.time()
        deadline = deadline or (t0 + self.budget)
        # a worker still busy with an earlier night would steal CPU from tonight: restart it (it loses its
        # memory; the brain's trust check covers the rival until its code is right again)
        for i, w in enumerate(self.workers):
            if not w.answered and w.alive():
                if select.select([w.p.stdout.fileno()], [], [], 0)[0]:
                    msg = w.poll_lines()
                    k = (w.tid, getattr(w, "cand", 0))
                    # answered after last night's deadline: too late to use, but it still earns (or loses) trust,
                    # so a slow candidate that is right can win over a fast one that is wrong
                    if msg is not None and msg.get("order") != "error" and k not in self.said and self.asked:
                        self.said[k] = self.asked + (combine([msg.get("order")]),)
                if not w.answered:
                    w.close()
                    try:
                        nw = Worker(*w.args)
                        nw.cand = getattr(w, "cand", 0)
                        self.workers[i] = nw
                        self.restarts = getattr(self, "restarts", 0) + 1
                    except Exception:
                        pass
        try:
            self.score(state)
        except Exception:
            self.said = {}
        self.asked = (state.get("market"), state.get("order_for_day"))
        live = []
        for w in self.workers:
            if not w.alive():
                continue
            try:
                v = rival_view(state, w.tid, self.memo.setdefault(w.seed, {}))
            except Exception:
                v = None
            if v is None:
                continue
            w.send(self.n, v)
            live.append(w)
        answers = {}
        pending = {w.p.stdout.fileno(): w for w in live}
        while pending and time.time() < deadline:
            r, _, _ = select.select(list(pending), [], [], max(0.0, deadline - time.time()))
            for fd in r:
                w = pending[fd]
                msg = w.poll_lines()
                if msg is not None:
                    if msg.get("order") != "error":
                        answers.setdefault((w.tid, getattr(w, "cand", 0)), []).append(msg.get("order"))
                    self.last_timing[(w.tid, w.seed) if not getattr(w, "cand", 0) else (w.tid, "alt%d" % w.cand)] = time.time() - t0
                    del pending[fd]
                elif not w.alive():
                    del pending[fd]
        out = {}
        best = {}
        for (tid, k), v in self.trust.items():
            best[tid] = max(best.get(tid, 0.0), v)
        for (tid, k), orders in sorted(answers.items()):
            o = combine(orders)
            self.said[(tid, k)] = (state.get("market"), state.get("order_for_day"), o)
            tr = self.trust.get((tid, k), 0.0)
            if tr < best.get(tid, 0.0) - TRUST_GAP:
                continue        # clearly worse than a candidate that missed tonight: leave it to persistence
            # candidates in roster order; a later one is used only with a strictly better hit score
            if tid not in out or tr > self.trust.get((tid, self.used[tid]), 0.0):
                out[tid], self.used[tid] = o, k
        for tid in set(t for t, _ in answers) - set(out):
            self.used.pop(tid, None)
        return out

    def close(self):
        for w in self.workers:
            w.close()


def combine(orders):
    """Samples of a random rival -> one order: the most common spot, then the median price there."""
    good = [o for o in orders if isinstance(o, dict) and o.get("spot")]
    if not good:
        return None
    spots = [o["spot"] for o in good]
    spot = max(set(spots), key=spots.count)
    at = sorted((o for o in good if o["spot"] == spot), key=lambda o: _num(o.get("price")))
    out = {k: v for k, v in at[len(at) // 2].items() if k not in ("_dist", "_near")}
    near = [o.get("_near") for o in good]
    if all(isinstance(x, list) and x for x in near):
        # each sample's own odds over the options it drew from (a randomized rival's exact choice rule), mixed
        # over the samples: (spot, quality) -> the likeliest option there and its total probability
        mass, best = {}, {}
        for x in near:
            for item in x:
                if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], dict)):
                    continue
                o, w = item[0], _num(item[1]) / len(orders)
                k = (o.get("spot"), o.get("quality"))
                mass[k] = mass.get(k, 0.0) + w
                if w > best.get(k, (None, -1.0))[1]:
                    best[k] = (o, w)
        keep = [k for k in mass if mass[k] >= 0.03]
        tot = sum(mass[k] for k in keep)
        alive = len(good) / float(len(orders))      # samples that opened
        if keep and tot > 0:
            out["_dist"] = [[dict(best[k][0]), mass[k] / tot * alive]
                            for k in sorted(keep, key=lambda k: -mass[k])]
    elif len(orders) > 1:
        # every spot the samples chose, with its share of all samples (closed samples place nothing)
        dist = []
        for s in sorted(set(spots), key=lambda s: (-spots.count(s), s)):
            at = sorted((o for o in good if o["spot"] == s), key=lambda o: _num(o.get("price")))
            o = {k: v for k, v in at[len(at) // 2].items() if k not in ("_dist", "_near")}
            dist.append([o, spots.count(s) / float(len(orders))])
        out["_dist"] = dist
    return out
