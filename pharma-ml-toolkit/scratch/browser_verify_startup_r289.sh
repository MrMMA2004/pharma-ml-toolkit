#!/bin/bash
# r289 — the part11 §14.1 startup bridge, browser verification (v2):
# the app boots ITSELF through its own StartupGate (NO configureBackend
# injection — the r288 pattern), the WS opens through the proxied dev
# fallback, and a dataset flows end-to-end (upload → preview) through
# the app's native boot path. Everything inside ONE call.
set -u
cd "$HOME/my-project/pharma-ml-toolkit"

FAILED=0
note() { echo "==[ $1 ]=="; }
check() { if eval "$2"; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAILED=$((FAILED+1)); fi }

# -- boot the real backend (port-0 handshake) --------------------------
rm -f /tmp/r289_backend.log /tmp/r289_vite.log
"$HOME/.venv/bin/python3" -m pharmaml.api > /tmp/r289_backend.log 2>&1 &
BACKEND_PID=$!

READY=""
for i in $(seq 1 40); do
  READY=$(rg -o 'PHARMAML-READY \{.*\}' /tmp/r289_backend.log 2>/dev/null | head -1 || true)
  [ -n "${READY}" ] && break
  sleep 1
done
PORT=$(echo "${READY}" | rg -o '"port": *[0-9]+' | rg -o '[0-9]+' | head -1)
TOKEN=$(echo "${READY}" | rg -o '"token": *"[^"]+"' | sed 's/.*"token": *"//; s/"$//')
check "backend handshake (port-0 loopback)" '[ -n "${PORT}" ] && [ -n "${TOKEN}" ]'

# -- vite with BOTH env channels: the proxy port + the dev token --------
(cd "$HOME/my-project/pharma-ml-toolkit/frontend" && \
  PHARMAML_DEV_BACKEND_PORT="${PORT}" VITE_PHARMAML_TOKEN="${TOKEN}" \
  bun x vite --port 5174 --host 127.0.0.1 > /tmp/r289_vite.log 2>&1 &)
for i in $(seq 1 30); do
  curl -s --max-time 2 -o /dev/null http://127.0.0.1:5174/ && break
  sleep 1
done
check "vite up with the dev token channel" 'curl -s --max-time 2 -o /dev/null http://127.0.0.1:5174/'

# -- the browser flow: the app boots ITSELF (NO injection) -------------
agent-browser open http://127.0.0.1:5174 > /dev/null 2>&1
timeout 25 agent-browser wait --load networkidle > /dev/null 2>&1

page_text() { agent-browser eval "document.body.innerText" 2>/dev/null || true; }
# wait until the gate resolves (connecting card -> shell) or timeout
for i in $(seq 1 25); do
  TXT=$(page_text)
  echo "${TXT}" | rg -q "Starting the local scientific backend" || break
  sleep 1
done
TXT=$(page_text)
check "startup gate resolved to the app shell" \
  'echo "${TXT}" | rg -q "PharmaML Studio" && echo "${TXT}" | rg -q "Offline computational pharmacy workbench"'
check "gate not stuck on the connecting card" \
  'test $(echo "${TXT}" | rg -c "Starting the local scientific backend" || echo 0) -eq 0'
check "no failed-startup card" \
  'test $(echo "${TXT}" | rg -c "The backend did not start" || echo 0) -eq 0'
check "Introduction page is the initial workspace" \
  'echo "${TXT}" | rg -q "The nine workspaces"'

# the sr-only ready marker carries the backend version via the bridge
READYMARK=$(agent-browser eval "document.querySelector(\"[data-testid=startup-ready]\") ? document.querySelector(\"[data-testid=startup-ready]\").textContent : \"MISSING\"" 2>/dev/null || true)
echo "  startup-ready marker: ${READYMARK}"
check "bridge delivered the backend version" \
  'echo "${READYMARK}" | rg -q "ready via dev-same-origin"'

# the app's own event stream state (through the proxied WS lane)
WSSTATE=$(agent-browser eval "import(\"/src/services/websocket/client.ts\").then(m => m.getEventStream().status.state)" 2>/dev/null || true)
echo "  event stream state: ${WSSTATE}"
check "event stream OPEN through the app's own boot path" \
  'echo "${WSSTATE}" | rg -q "open"'

# -- navigate to Datasets and run a real flow --------------------------
# the nav buttons live in the header nav (aria-current on Introduction)
NAVREF=$(agent-browser snapshot -i -c 2>/dev/null | rg '^\s*- (link|button) "Datasets"' | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
echo "  nav ref: ${NAVREF}"
if [ -n "${NAVREF}" ]; then agent-browser click "@${NAVREF}" > /dev/null 2>&1; fi
sleep 3
DTXT=$(page_text)
check "Datasets workspace mounts post-gate" 'echo "${DTXT}" | rg -qi "Import.*Inspect|upload|dataset"'

# upload a tiny CSV through the app's own file input (DataTransfer —
# the React value-tracker-safe pattern the r288 note documented)
CSV=/tmp/r289_upload.csv
printf 'smiles,pIC50\nCCOc1ccccc1O,4.2\nc1ccccc1,3.9\nCC(=O)Oc1ccccc1C(=O)O,5.1\n' > "${CSV}"
CSVJS=$(cat "${CSV}" | sed 's/"/\\"/g' | awk '{printf "%s\\n", $0}' | sed 's/\\/\\\\/g; s/\\\\n/\\n/g')
agent-browser eval "
(async () => {
  const input = document.querySelector('input[type=file]');
  if (!input) { return 'NO-INPUT'; }
  const dt = new DataTransfer();
  dt.items.add(new File([\"${CSVJS}\"], 'r289_upload.csv', { type: 'text/csv' }));
  input.files = dt.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return 'FILE-SET';
})()" 2>/dev/null
sleep 1
# click the Upload button (enabled now that the file state landed)
UPREF=$(agent-browser snapshot -i -c 2>/dev/null | rg '^\s*- button "Upload"' | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
echo "  upload button ref: ${UPREF}"
if [ -n "${UPREF}" ]; then agent-browser click "@${UPREF}" > /dev/null 2>&1; fi
# wait for the upload job + preview
for i in $(seq 1 40); do
  UTXT=$(page_text)
  echo "${UTXT}" | rg -q "CCOc1ccccc1O" && break
  sleep 1
done
UTXT=$(page_text)
check "dataset uploaded + preview rendered through the native boot path" \
  'echo "${UTXT}" | rg -q "CCOc1ccccc1O" && echo "${UTXT}" | rg -q "4\.2"'

# screenshot evidence
agent-browser screenshot /tmp/r289_datasets.png > /dev/null 2>&1 || true

# -- cleanup ------------------------------------------------------------
kill "${BACKEND_PID}" 2>/dev/null || true
pkill -f "vite --port 5174" 2>/dev/null || true
agent-browser close > /dev/null 2>&1 || true

note "SUMMARY"
echo "FAILURES=${FAILED}"
exit $((FAILED > 0 ? 1 : 0))
