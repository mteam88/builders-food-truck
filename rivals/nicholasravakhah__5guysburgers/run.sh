#!/usr/bin/env bash
# Keeps the Cheapest Eats bot running all night: if it crashes or exits, it restarts in 3 seconds.
# Usage:  ./run.sh            (or: bash run.sh)   - extra args are passed to bot.py
# Stop it with Ctrl+C twice.
cd "$(dirname "$0")" || exit 1
while true; do
  git pull --ff-only -q 2>/dev/null || echo "git pull skipped (local changes or offline)"
  python3 bot.py "$@"
  code=$?
  echo "$(date '+%H:%M:%S') bot exited (code $code), restarting in 3s... (Ctrl+C again to stop)"
  sleep 3 || break
done
