"""Read-only: print every change to our live order (you.order) with a timestamp.

Each POST /orders from ANY client replaces you.order, so this shows how many
clients are writing and when. Match the timestamps against teammates' terminals.

    python3 tools/watch_orders.py
"""
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ftw.api import Client, NetError  # noqa: E402


def main():
    c = Client()
    version, last = None, "unset"
    while True:
        try:
            s, st = c.state(since=version, timeout=35)
        except NetError as e:
            print("net", e, flush=True)
            time.sleep(2)
            continue
        if s != 200:
            print("status", s, st.get("error"), flush=True)
            time.sleep(5)
            continue
        version = st.get("version", version)
        o = (st.get("you") or {}).get("order")
        cur = json.dumps(o, sort_keys=True)
        if cur != last:
            now = dt.datetime.now().strftime("%H:%M:%S")
            print("%s  phase=%s for_day=%s  order=%s" % (now, st.get("phase"), st.get("order_for_day"), cur),
                  flush=True)
            last = cur


if __name__ == "__main__":
    main()
