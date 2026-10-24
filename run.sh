#!/usr/bin/env bash
# Frontrun Fries bot. Needs Python 3.9+ and NIGHTSHIFT_KEY (env or .env). Restarts itself on exit.
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
# Newest rival repo snapshots into rivals_live/ for RivalHost (<= 20 s, never fatal; vendored rivals/ is the fallback).
"$PY" -I ftw/fetch_rivals.py --total 20 || true
while true; do
  "$PY" -m ftw.runner
  echo "bot exited ($?); restarting in 1 s" >&2
  sleep 1
done
