#!/usr/bin/env bash
# round-283 E2E runner: boots the LIVE backend, runs the DL driver,
# SIGTERMs the backend AT the await-sigterm marker (the shutdown leg
# is the driver's own wait), then reports the verdict.
set -u
cd /home/z/my-project/pharma-ml-toolkit
PY=/home/z/.venv/bin/python
HANDSHAKE=/tmp/handshake283.txt
DRIVER_LOG=/tmp/e2e283-driver.log

step() { echo ""; echo "=== $1 ==="; }

step "1. boot the backend (stdout -> the handshake line)"
rm -f "$HANDSHAKE" "$DRIVER_LOG"
PYTHONPATH=src $PY -m pharmaml.api > "$HANDSHAKE" 2> /tmp/e2e283-backend.log &
BACKEND_PID=$!
echo "backend pid=$BACKEND_PID"

for i in $(seq 1 100); do
  if rg -q "^PHARMAML-READY " "$HANDSHAKE" 2>/dev/null; then break; fi
  sleep 0.5
done
head -c 200 "$HANDSHAKE" | sed 's/"token": "[^"]*"/"token": "REDACTED"/'

step "2. run the DL E2E driver against the live backend (background)"
cd /home/z/my-project/pharma-ml-toolkit/frontend
timeout 150 bun ../scratch/e2e283/driver.ts "$HANDSHAKE" > "$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!
cd /home/z/my-project/pharma-ml-toolkit

step "3. watch the driver until it reaches the await-sigterm marker"
for i in $(seq 1 240); do
  if rg -q "await-sigterm" "$DRIVER_LOG" 2>/dev/null; then
    echo "driver reached the shutdown wait (${i}s)"
    break
  fi
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "driver exited before the marker (failure — log below)"
    break
  fi
  sleep 1
done

step "4. SIGTERM the backend (the graceful-shutdown leg fires now)"
kill -TERM $BACKEND_PID 2>/dev/null

step "5. the driver's verdict"
wait $DRIVER_PID 2>/dev/null
DRIVER_RC=$?
cat "$DRIVER_LOG"

step "6. reap the backend + verdict"
wait $BACKEND_PID 2>/dev/null
sleep 1
echo "driver rc=$DRIVER_RC"
if [ "$DRIVER_RC" -eq 0 ]; then echo "E2E-RUN-PASS"; else echo "E2E-RUN-FAIL"; fi
exit $DRIVER_RC
