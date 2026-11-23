#!/bin/bash
# r332 console supervisor: self-healing dev-server loop (the reaper's dense cadence countered)
cd /home/z/my-project
while true; do
  if [ ! -x node_modules/.bin/next ]; then
    bun install >> scratch/console-supervisor.log 2>&1
  fi
  bun run dev >> dev.log 2>&1
  sleep 3
done
