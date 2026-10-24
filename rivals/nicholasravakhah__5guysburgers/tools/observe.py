"""Log every live-game day from the public state (no key needed).

Long-polls /public/state?since=<version> (about two requests a minute) and appends
one JSON line per completed day to state/live_days.jsonl. Used to calibrate the
customer model against real crowds, hungry counts and rival orders.

    python3 tools/observe.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw.api import Client, NetError, ROOT  # noqa: E402


def main():
    out_dir = os.path.join(ROOT, "state")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "live_days.jsonl")
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen.add((d.get("mode"), d.get("market"), d["last_day"]["day"]))
                except Exception:
                    pass
    client = Client(key="")
    version, backoff = None, 2
    while True:
        try:
            status, st = client.public_state(since=version)
        except NetError as e:
            print("net error:", e, flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        backoff = 2
        if status != 200:
            print("status", status, st.get("error"), flush=True)
            time.sleep(30 if status == 429 else 10)
            continue
        version = st.get("version", version)
        ld = st.get("last_day") or {}
        k = (st.get("mode"), st.get("market"), ld.get("day"))
        if ld.get("day") and k not in seen:
            seen.add(k)
            rec = {"mode": st.get("mode"), "market": st.get("market"), "server_time": st.get("server_time"),
                   "last_day": ld, "trucks": st.get("trucks"), "forecast": st.get("forecast"),
                   "rules": st.get("rules")}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("logged", k, flush=True)
        if st.get("phase") == "over":
            time.sleep(60)


if __name__ == "__main__":
    main()
