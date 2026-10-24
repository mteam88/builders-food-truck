"""Runs one rival's decision function: reads {"id", "state"} lines, writes {"id", "order"} lines.

Started by ftw/rivals.py with a scrubbed environment and a temp working directory. Only the adapter's
make(**kwargs) -> policy(state) is called; nothing here touches the network.
"""
import importlib.util
import json
import os
import sys


def main():
    path, kwargs = sys.argv[1], json.loads(sys.argv[2])
    spec = importlib.util.spec_from_file_location("rival_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    devnull = open(os.devnull, "w")
    real_stdout = sys.stdout
    sys.stdout = devnull            # rival code may print; keep our pipe clean
    spec.loader.exec_module(mod)
    policy = mod.make(**kwargs)
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        try:
            order = policy(req["state"])
            if not isinstance(order, dict) or order.get("wrecked"):     # wrecked: it submits nothing
                order = None
            else:
                order = {k: order.get(k) for k in ("spot", "price", "quality", "meals", "_near") if k in order
                         or k != "_near"}
        except BaseException:   # a crashing rival is "no answer" tonight, and keeps running
            order = "error"
        real_stdout.write(json.dumps({"id": req.get("id"), "order": order}) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
