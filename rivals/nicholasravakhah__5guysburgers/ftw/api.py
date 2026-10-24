"""HTTP client for the Night Shift API. Standard library only.

The truck key comes from $NIGHTSHIFT_KEY or a gitignored .env file next to the
repo root. It is never logged or included in exceptions.
"""
import json
import os
import socket
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
API = "/api/v1"


def load_key():
    key = os.environ.get("NIGHTSHIFT_KEY", "").strip()
    if key:
        return key
    path = os.path.join(ROOT, ".env")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("NIGHTSHIFT_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


class NetError(Exception):
    """Network-level failure (timeout, DNS, connection reset)."""


class Client:
    def __init__(self, key=None, url=URL):
        self._key = key if key is not None else load_key()
        self.url = url

    def __repr__(self):  # never expose the key
        return "Client(url=%r)" % self.url

    @property
    def has_key(self):
        return bool(self._key)

    def call(self, method, path, body=None, timeout=35.0):
        """Returns (status, json_dict). Raises NetError on transport failures."""
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
            except Exception:
                raw = b""
            status = e.code
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            raise NetError(type(e).__name__ + ": " + str(getattr(e, "reason", e))) from None
        try:
            payload = json.loads(raw.decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            payload = {"error": "non-json response (%d bytes)" % len(raw)}
        if not isinstance(payload, dict):
            payload = {"data": payload}
        return status, payload

    def state(self, base=API, since=None, timeout=35.0):
        path = base + "/state" + ("?since=%d" % since if since is not None else "")
        return self.call("GET", path, timeout=timeout)

    def order(self, order, base=API, timeout=5.0):
        return self.call("POST", base + "/orders", order, timeout=timeout)

    def vote(self, candidate, base=API, timeout=5.0):
        return self.call("POST", base + "/vote", {"candidate": candidate}, timeout=timeout)

    def staff(self, staff=None, wage=None, base=API, timeout=5.0):
        body = {}
        if staff is not None:
            body["staff"] = int(staff)
        if wage is not None:
            body["wage"] = float(wage)
        return self.call("POST", base + "/staff", body, timeout=timeout)

    def coupon(self, target, base=API, timeout=5.0):
        return self.call("POST", base + "/coupons", {"target": target}, timeout=timeout)

    def practice(self, seed=None, days=30, rivals=3):
        body = {"days": days, "rivals": rivals}
        if seed is not None:
            body["seed"] = str(seed)
        return self.call("POST", API + "/practice", body, timeout=20.0)

    def public_state(self, since=None, timeout=35.0):
        path = API + "/public/state" + ("?since=%d" % since if since is not None else "")
        return self.call("GET", path, timeout=timeout)
