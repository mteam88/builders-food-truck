"""Predict rivals by running their real, unmodified bots against a loopback fake game server.

    host = RivalHost()                 # starts the server and every rival once
    orders = host(state, deadline)     # {truck id: order dict or None (sat out)}; late rivals are absent
    host.close()                       # kills every rival's process group

Each rival is its own repo snapshot (rivals_live/<x> if it smoke-starts, else the vendored rivals/<x>)
copied to a private temp dir and started through ftw/shim.py, so every connection it makes lands on our
server on 127.0.0.1. It gets a scrubbed env (PATH with only python, private HOME/TMPDIR, a fake key
unique to it; never our key), start_new_session=True, and its output in a per-rival log file. The
processes live for the whole run so each rival keeps its memory.

Each night we publish the real state; each rival's long-poll wakes with the night as it sees it
(ftw.rivals.rival_view: you = that truck), its bot decides, and POST /api/v1/orders is captured.
"""
import copy
import datetime as dt
import json
import os
import secrets
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIM = os.path.join(HERE, "shim.py")
sys.path.insert(0, ROOT)
from ftw.rivals import rival_view  # noqa: E402

LONG_POLL_S = 25.0
STARTUP_S = 20.0        # a snapshot that hasn't asked for /state by then is "not starting"
MAX_RESTARTS = 4
REAL_URL = "https://nightshift.joinbuilders.org"
COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}

# name -> (truck id, snapshot dir name, entry script, args), from ftw/roster.py
from ftw.roster import RIVALS, snap  # noqa: E402
ROSTER = {n: (v[0], snap(v[1]), v[2], list(v[3])) for n, v in RIVALS.items()}
REPOS = {v[1]: v[1].replace("__", "/", 1) for v in ROSTER.values()}


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_ts(s):
    try:
        return dt.datetime.strptime(str(s).replace("Z", "+0000")[:30], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    except (ValueError, TypeError):
        try:
            return dt.datetime.strptime(str(s).replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z").timestamp()
        except (ValueError, TypeError):
            return None


def validate_order(body, rules):
    """The real server's checks (copied from the vmmode fake server). (order, None) or (None, error)."""
    if not isinstance(body, dict):
        return None, "body must be a JSON object"
    spots = rules.get("spots") or []
    costs = rules.get("costs") or COSTS
    spot, quality = body.get("spot"), body.get("quality")
    if spot not in spots:
        return None, "spot must be one of: " + ", ".join(map(str, spots))
    if quality not in costs:
        return None, "quality must be one of: " + ", ".join(costs)
    price = body.get("price")
    if isinstance(price, bool):
        price = None
    try:
        price = round(float(price), 2)
    except (TypeError, ValueError):
        return None, "price must be a number"
    lo, hi = rules.get("price_min", 1), rules.get("price_max", 30)
    if not (price == price) or price < lo or price > hi:
        return None, "price must be between %s and %s" % (lo, hi)
    meals = body.get("meals")
    try:
        if isinstance(meals, bool):
            raise ValueError
        if isinstance(meals, str):
            meals = float(meals.strip())
        if isinstance(meals, float):
            if not meals.is_integer():
                raise ValueError
            meals = int(meals)
        if not isinstance(meals, int):
            raise ValueError
    except (TypeError, ValueError):
        return None, "meals must be a whole number"
    if meals < 0:
        return None, "meals must be 0 or more"
    budget = rules.get("budget", 500)
    if meals * costs[quality] > budget + 1e-9:
        return None, "order over budget"
    return {"spot": spot, "price": price, "quality": quality, "meals": meals}, None


class Rival:
    def __init__(self, name, tid, snap, entry, args):
        self.name, self.tid, self.snap, self.entry, self.args = name, tid, snap, entry, args
        self.key = "fk_rh_%s_%s" % (name, secrets.token_hex(8))
        self.proc = None
        self.src = None          # snapshot dir in use
        self.src_sha = None      # its .ftw-sha when copied
        self.cands = []          # snapshot dirs to try, best first
        self.dir = None
        self.log = None
        self.version = 0
        self.view = None         # current night's view (dict) or None before the first night
        self.seq = None          # night seq of self.view
        self.order = None        # (seq, order, t)
        self.sat_out = None      # seq it re-polled without ordering
        self.served = None       # seq it has been served
        self.last_req = 0.0
        self.first_req = None
        self.started = None
        self.restarts = 0
        self.crashes = 0
        self.late = 0
        self.dead = False


class RivalHost:
    def __init__(self, roster=None, port=0, root=None, log_dir=None, python=None, startup_s=STARTUP_S, dirs=None,
                 netns=None, budget=1.0):
        self.budget = budget
        self.cv = threading.Condition()
        self.start_lock = threading.RLock()     # _monitor and refresh() both (re)start rivals
        self.counter = 1
        self.seq = 0
        self.night_t = None
        self.base = None
        self.last_key = None
        self.want = []
        self.memo = {}
        self.closed = False
        self.python = python or sys.executable
        self.startup_s = startup_s
        self.root = root or ROOT
        self.tmp = tempfile.mkdtemp(prefix="ftw-rivalhost-")
        self.log_dir = log_dir or os.path.join(self.tmp, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.last_timing = {}
        roster = ROSTER if roster is None else roster
        self.rivals = {}
        self.dirs = dirs
        for name, (tid, snap, entry, args) in roster.items():
            r = Rival(name, tid, snap, entry, list(args))
            r.cands = self._cands(r)
            self.rivals[name] = r
        self.by_key = {r.key: r for r in self.rivals.values()}
        self.srv = self._serve(port)
        self.port = self.srv.server_address[1]
        self.url = "http://127.0.0.1:%d" % self.port
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.usock = os.path.join(self.tmp, "fake.sock")
        self.usrv = None
        self.netns = False
        if netns is not False:          # None = use it if this machine allows it
            self.netns = self._netns_probe()
        for r in self.rivals.values():
            self._start(r)
        self._mon = threading.Thread(target=self._monitor, daemon=True)
        self._mon.start()

    def _cands(self, r):
        if self.dirs and r.name in self.dirs:
            return [d for d in self.dirs[r.name] if os.path.isfile(os.path.join(d, r.entry))]
        return [d for d in (os.path.join(self.root, "rivals_live", r.snap), os.path.join(self.root, "rivals", r.snap))
                if os.path.isfile(os.path.join(d, r.entry))]

    @staticmethod
    def _sha(d):
        try:
            with open(os.path.join(d, ".ftw-sha")) as f:
                return f.read().strip()
        except OSError:
            return None

    def refresh(self):
        """Re-read the snapshots (after fetch_rivals refreshed rivals_live/) and restart every rival whose best
        snapshot is a different dir or commit than the one it runs. Call between nights only. Never raises;
        returns the names restarted."""
        out = []
        for r in list(self.rivals.values()):
            try:
                cands = self._cands(r)
                if not cands or self.closed:
                    continue
                if cands[0] == r.src and self._sha(cands[0]) == r.src_sha:
                    continue
                r.cands, r.dead, r.restarts = cands, False, 0
                r.src = None              # copy the new snapshot afresh
                self._start(r)
                out.append(r.name)
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ server
    def _serve(self, port):
        H = make_handler(self)
        tries = [port] + ([] if port else list(range(18765, 18800)))
        err = None
        for p in tries:
            try:
                s = _Server(("127.0.0.1", p), H)
                return s
            except OSError as e:
                err = e
        raise err

    def _netns_probe(self):
        """Can rivals run in `unshare -rn` (private net namespace, only our Unix socket out)? ~0.1 s."""
        unshare = shutil.which("unshare") or next((p for p in ("/usr/bin/unshare", "/bin/unshare")
                                                   if os.path.exists(p)), None)
        if not unshare:
            return False
        try:
            self.usrv = _UnixServer(self.usock, make_handler(self))
            threading.Thread(target=self.usrv.serve_forever, daemon=True).start()
            self.unshare = unshare
            out = subprocess.run([unshare, "-rn", "--", self.python, "-I", SHIM, "--redirect",
                                  "127.0.0.1:%d" % self.port, "--via-unix", self.usock, "--", "--selftest"],
                                 env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, capture_output=True, timeout=10,
                                 stdin=subprocess.DEVNULL)
            return out.returncode == 0 and b"SELFTEST 200" in out.stdout
        except Exception:
            return False

    # ------------------------------------------------------------------ processes
    def _env(self, r, home):
        bindir = os.path.join(self.tmp, "bin")
        if not os.path.isdir(bindir):
            os.makedirs(bindir, exist_ok=True)
            for n in ("python", "python3"):
                try:
                    os.symlink(self.python, os.path.join(bindir, n))
                except OSError:
                    pass
        os.makedirs(os.path.join(home, "tmp"), exist_ok=True)
        return {"PATH": bindir, "HOME": home, "TMPDIR": os.path.join(home, "tmp"), "LANG": "C.UTF-8",
                "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                # the real host: the shim sends every connection (https too) to our stand-in server, and some
                # bots refuse a non-https URL
                "NIGHTSHIFT_URL": REAL_URL, "NIGHTSHIFT_BASE": REAL_URL, "NIGHTSHIFT_API": REAL_URL + "/api/v1",
                "NIGHTSHIFT_KEY": r.key}

    def _start(self, r):
        """(Re)start r from its best candidate snapshot. Never raises."""
        with self.start_lock:
            self._start1(r)

    def _start1(self, r):
        try:
            self._kill(r)
            if not r.cands:
                r.dead = True
                return
            src = r.cands[0]
            if r.src != src or r.dir is None:
                r.dir = os.path.join(self.tmp, r.name, "code")
                if os.path.exists(r.dir):
                    shutil.rmtree(r.dir, ignore_errors=True)
                shutil.copytree(src, r.dir, symlinks=True, ignore=shutil.ignore_patterns(".git"))
                r.src, r.src_sha = src, self._sha(src)
            home = os.path.join(self.tmp, r.name, "home")
            if r.log is None:
                r.log = open(os.path.join(self.log_dir, r.name + ".log"), "ab", buffering=0)
            r.log.write(("\n=== %s start %s from %s\n" % (time.strftime("%H:%M:%S"), r.name, src)).encode())
            cmd = [self.python, "-I", SHIM, "--lifeline", "--redirect", "127.0.0.1:%d" % self.port]
            if self.netns:
                cmd = [self.unshare, "-rn", "--"] + cmd + ["--via-unix", self.usock]
            cmd += ["--", os.path.join(r.dir, r.entry)] + r.args
            r.first_req = None
            r.started = time.time()
            r.proc = subprocess.Popen(cmd, cwd=r.dir, env=self._env(r, home), stdin=subprocess.PIPE,
                                      stdout=r.log, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as e:
            try:
                r.log and r.log.write(("start failed: %r\n" % (e,)).encode())
            except Exception:
                pass
            r.proc = None

    def _kill(self, r):
        p = r.proc
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            p.kill()
            p.wait(timeout=2)
        except Exception:
            pass
        r.proc = None

    def _monitor(self):
        while not self.closed:
            time.sleep(0.5)
            for r in list(self.rivals.values()):
                if self.closed or r.dead:
                    continue
                p = r.proc
                died = p is None or p.poll() is not None
                stuck = (not died and r.first_req is None and r.started and time.time() - r.started > self.startup_s)
                if not (died or stuck):
                    continue
                r.crashes += 1
                if len(r.cands) > 1 and (r.first_req is None or r.restarts >= MAX_RESTARTS):
                    r.cands.pop(0)          # this snapshot never got going / keeps dying: fall back
                    r.restarts = 0
                elif r.restarts >= MAX_RESTARTS:
                    self._kill(r)
                    r.dead = True
                    continue
                else:
                    r.restarts += 1
                if not self.closed:
                    self._start(r)

    def close(self):
        self.closed = True
        for r in self.rivals.values():
            self._kill(r)
            try:
                r.log and r.log.close()
            except Exception:
                pass
        for srv in (self.srv, self.usrv):
            try:
                if srv:
                    srv.shutdown()
                    srv.server_close()
            except Exception:
                pass
        # drop code copies, homes, socket and bin; keep logs (in tmp/logs unless log_dir was given)
        for x in list(self.rivals) + ["bin"]:
            shutil.rmtree(os.path.join(self.tmp, x), ignore_errors=True)
        try:
            os.unlink(self.usock)
        except OSError:
            pass
        if os.path.abspath(self.log_dir) != os.path.join(self.tmp, "logs"):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def __del__(self):
        try:
            if not self.closed:
                self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ nights
    def predict(self, state, deadline=None, views=None):
        """views: optional {truck id: the exact state that truck should see} instead of rival_view()."""
        try:
            return self._predict(state, deadline, views)
        except Exception:
            return {}

    __call__ = predict

    def _predict(self, state, deadline, given=None):
        t0 = time.time()
        deadline = deadline or (t0 + self.budget)
        if not isinstance(state, dict):
            return {}
        left = None
        e, n = parse_ts(state.get("phase_ends_at")), parse_ts(state.get("server_time"))
        if e and n:
            left = max(0.0, e - n)
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        night = state.get("phase", "night") == "night" and state.get("orders_open", True) is not False \
            and state.get("order_for_day") is not None
        if night and key == self.last_key and self.seq:
            with self.cv:                   # same night again: keep waiting on what we already published
                seq, t0, want = self.seq, self.night_t, list(self.want)
        else:
            views = {}
            for r in self.rivals.values():
                try:
                    v = copy.deepcopy(given.get(r.tid)) if given is not None else rival_view(state, r.tid, self.memo)
                except Exception:
                    v = None
                if isinstance(v, dict):
                    views[r.name] = v
            with self.cv:
                self.seq += 1
                seq = self.seq
                self.night_t = t0
                self.base = state
                self.counter += 1
                for name, v in views.items():
                    r = self.rivals[name]
                    v["_left"] = left
                    r.view, r.seq, r.version = v, seq, self.counter
                self.cv.notify_all()
                self.want = want = [self.rivals[n] for n in views]
                self.last_key = key if night else None
            if not night:
                return {}
        with self.cv:
            def done(r):
                return (r.order and r.order[0] == seq) or r.sat_out == seq or r.dead or \
                    r.proc is None or r.proc.poll() is not None

            while time.time() < deadline and not all(done(r) for r in want):
                self.cv.wait(max(0.001, min(0.05, deadline - time.time())))
            out = {}
            for r in want:
                if r.order and r.order[0] == seq:
                    out[r.tid] = dict(r.order[1])
                    self.last_timing[r.tid] = r.order[2] - t0
                elif r.sat_out == seq:
                    out[r.tid] = None
                    self.last_timing[r.tid] = None
        return out

    def view_for(self, r):
        """Caller holds self.cv."""
        now = time.time()
        if r.view is None:
            return {"version": r.version, "mode": "final", "phase": "pregame", "orders_open": False,
                    "order_for_day": None, "phase_ends_at": None, "server_time": iso(now), "market": 1,
                    "markets": 20, "day": 0, "days": 30, "trucks": [], "last_day": None, "forecast": None,
                    "rules": {"budget": 500, "costs": dict(COSTS), "values": {"cheap": 8, "standard": 12, "premium": 16},
                              "spots": ["oval", "thompson", "union", "stadium", "highst", "northres"],
                              "qualities": ["cheap", "standard", "premium"], "price_min": 1, "price_max": 30},
                    "features": {}, "announcements": [],
                    "you": {"id": r.tid, "truck_name": r.name, "company_name": r.name, "cash": 0,
                            "reputation": {}, "order": None}}
        s = dict(r.view)
        left = s.pop("_left", None)
        s["version"] = r.version
        s["server_time"] = iso(now)
        if s.get("phase") == "night" and left is not None and self.night_t:
            s["phase_ends_at"] = iso(self.night_t + left)
        you = dict(s.get("you") or {})
        you["order"] = dict(r.order[1]) if r.order and r.order[0] == r.seq else None
        s["you"] = you
        return s


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass                    # rivals hang up mid-request all the time (killed, timeouts)


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass

    def get_request(self):
        req, _ = super().get_request()
        return req, ("unix", 0)         # BaseHTTPRequestHandler wants a (host, port) client address


def make_handler(host):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60
        server_version = "nightshift"
        sys_version = ""

        def log_message(self, *a):
            pass

        def setup(self):
            super().setup()
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        def send(self, code, obj):
            body = json.dumps(obj, separators=(",", ":")).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
            except OSError:
                self.close_connection = True

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def do_PUT(self):
            self.route("PUT")

        def do_PATCH(self):
            self.route("PATCH")

        def do_DELETE(self):
            self.route("DELETE")

        def route(self, method):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 1 << 22)) if n > 0 else b""
                u = urlsplit(self.path)
                path = u.path.rstrip("/") or "/"
                q = {k: v[-1] for k, v in parse_qs(u.query).items()}
                a = self.headers.get("Authorization") or ""
                key = a[7:].strip() if a[:7].lower() == "bearer " else (self.headers.get("X-Api-Key") or "").strip()
                r = host.by_key.get(key)
                if r is None and u.path == "/api/v1/_ping":
                    return self.send(200, {"ok": True})
                if r is None:
                    return self.send(401, {"error": "missing or wrong key"})
                with host.cv:
                    r.last_req = time.time()
                    if r.first_req is None:
                        r.first_req = r.last_req
                if path == "/api/v1/state" and method == "GET":
                    return self.state(r, q)
                if path == "/api/v1/_ping":
                    return self.send(200, {"ok": True})
                if path == "/api/v1/orders" and method == "POST":
                    return self.order(r, raw)
                if path == "/api/v1/trucks/me" and method == "GET":
                    with host.cv:
                        s = host.view_for(r)
                    return self.send(200, dict(s.get("you") or {}))
                if path in ("/api/v1/staff", "/api/v1/vote", "/api/v1/coupons") or path.startswith("/api/v1/coupons/"):
                    return self.send(200, {"ok": True})
                if path.startswith("/api/v1/practice"):
                    return self.send(403, {"error": "not allowed right now"})
                return self.send(404, {"error": "not found"})
            except Exception as e:
                return self.send(400, {"error": "bad request: %s" % type(e).__name__})

        def state(self, r, q):
            try:
                since = int(q["since"]) if "since" in q else None
            except ValueError:
                since = None
            end = time.time() + LONG_POLL_S
            with host.cv:
                if (since is not None and since == r.version and r.seq is not None and r.served == r.seq
                        and not (r.order and r.order[0] == r.seq) and r.sat_out != r.seq):
                    r.sat_out = r.seq           # saw tonight, went back to waiting without ordering
                    host.cv.notify_all()
                while since is not None and since == r.version and not host.closed:
                    left = end - time.time()
                    if left <= 0:
                        break
                    host.cv.wait(left)
                s = host.view_for(r)
                r.served = r.seq
            self.send(200, s)

        def order(self, r, raw):
            try:
                body = json.loads(raw or b"null")
            except ValueError:
                return self.send(400, {"error": "body is not JSON"})
            with host.cv:
                v = r.view
                if v is None or v.get("phase") != "night" or not v.get("orders_open", True):
                    return self.send(409, {"error": "orders are closed: it is not night"})
                o, err = validate_order(body, v.get("rules") or {})
                if err:
                    return self.send(400, {"error": err})
                for_day = v.get("order_for_day")
                if r.served != r.seq:
                    # it has not seen tonight yet: a late order for an earlier night. The real server would
                    # count it for tonight, but it says nothing about tonight's decision, so don't capture it.
                    r.late += 1
                    return self.send(200, {"order": o, "for_day": for_day})
                r.order = (r.seq, o, time.time())
                host.counter += 1
                r.version = host.counter
                host.cv.notify_all()
            self.send(200, {"order": o, "for_day": for_day})
    return H


class Hybrid:
    """Real rivals first, adapters for whoever the real bots didn't answer by the deadline.

        predictor = Hybrid(RivalHost(), ftw.rivals.Predictor(budget=...))
    Both run at once (the adapters in a thread); a real order wins over the adapter's for the same truck.
    """

    # Cheapest Eats is random (seeded per run): the adapters' 3-seed combine estimates it better than
    # one real sample (live replay 2026-10-24: real 4/38 exact, spot 15 vs adapter 20).
    PREFER_ADAPTER = ("t_5f2743",)

    def __init__(self, host, fallback, prefer_adapter=PREFER_ADAPTER):
        self.host, self.fallback = host, fallback
        self.prefer_adapter = set(prefer_adapter or ())
        self.last_source = {}

    def __call__(self, state, deadline=None):
        box = {}

        def fb():
            try:
                box["o"] = self.fallback(state, deadline=deadline) or {}
            except Exception:
                box["o"] = {}
        t = threading.Thread(target=fb, daemon=True)
        t.start()
        real = self.host(state, deadline=deadline)
        t.join(max(0.0, (deadline or time.time()) - time.time()) + 0.05)
        out = dict(box.get("o") or {})
        self.last_source = {k: "adapter" for k in out}
        for k, v in real.items():
            if k in self.prefer_adapter and k in out:
                continue
            out[k] = v
            self.last_source[k] = "real"
        return out

    def close(self):
        for x in (self.host, self.fallback):
            try:
                x.close()
            except Exception:
                pass
