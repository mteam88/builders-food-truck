#!/bin/bash
# Keeps the bot alive all night: restarts on crash, keeps the Mac awake.
cd "$(dirname "$0")"
while true; do
  caffeinate -dimsu python3 -u bot.py "$@" 2>&1 | tee -a data/bot.log
  echo "bot exited, restarting in 1s"; sleep 1
done
