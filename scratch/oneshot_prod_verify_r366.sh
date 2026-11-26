#!/usr/bin/env bash
# r366 one-shot production verification: build -> serve -> verify -> teardown.
# The sandbox reverts large ephemeral state (node_modules, .next) between
# tool calls, so the whole heavy chain must run inside a single call.
set -u
cd /home/z/my-project

echo "=== [0/5] node_modules state ==="
if [ ! -f node_modules/eslint/package.json ] || [ ! -f node_modules/next/package.json ]; then
  echo "reinstalling (copyfile)"
  bun install --backend=copyfile 2>&1 | tail -2
fi
ls node_modules/eslint/package.json node_modules/next/package.json >/dev/null 2>&1 \
  && echo "deps OK" || { echo "DEPS MISSING"; exit 3; }

echo "=== [1/5] production build ==="
rm -rf .next
if bun run build > /tmp/build.log 2>&1; then
  echo "build exit 0"
else
  echo "BUILD FAILED"; tail -20 /tmp/build.log; exit 4
fi
ls .next/standalone/server.js >/dev/null 2>&1 && echo "standalone server.js present" || { echo "NO STANDALONE"; tail -5 /tmp/build.log; exit 5; }

echo "=== [2/5] serve standalone ==="
NODE_ENV=production nohup bun /home/z/my-project/.next/standalone/server.js \
  > /tmp/standalone.log 2>&1 &
SRV=$!
echo "server pid $SRV"

echo "=== [3/5] render verification ==="
up=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 4
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 http://localhost:3000/ 2>/dev/null)
  if [ "$code" = "200" ]; then up=1; echo "server up after ~$((i*4))s"; break; fi
done
if [ "$up" != "1" ]; then echo "SERVER NEVER CAME UP"; tail -10 /tmp/standalone.log; kill -9 $SRV 2>/dev/null; exit 6; fi

/home/z/.venv/bin/python scratch/render_verify_r366.py
RC=$?
echo "render-verify exit: $RC"

echo "=== [4/5] (skip: the standalone parking step — the parked copy cannot resolve 'next' after the janitor wipes node_modules; removed) ==="

echo "=== [5/5] teardown ==="
kill -9 $SRV 2>/dev/null
pkill -9 -f "standalone/server.js" 2>/dev/null
echo "done (render-verify rc=$RC)"
exit $RC
