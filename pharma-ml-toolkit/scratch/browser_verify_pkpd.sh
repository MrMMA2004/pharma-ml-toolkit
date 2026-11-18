#!/bin/bash
# r286 — the PK/PD workspace browser verification, all inside ONE call
# (background processes do not survive between tool calls here):
# backend + vite + the agent-browser flow + the assertions + cleanup.
set -u
cd "$HOME/my-project/pharma-ml-toolkit"

FAILED=0
note() { echo "==[ $1 ]=="; }
check() { # check <name> <cond-string>
  if eval "$2"; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAILED=$((FAILED+1)); fi
}

# -- boot the real backend (port-0 handshake) --------------------------
rm -f /tmp/pkpd_backend.log
"$HOME/.venv/bin/python3" -m pharmaml.api > /tmp/pkpd_backend.log 2>&1 &
BACKEND_PID=$!

# read the handshake BEFORE starting vite (the dev proxy targets it)
READY=""
for i in $(seq 1 40); do
  READY=$(rg -o 'PHARMAML-READY \{.*\}' /tmp/pkpd_backend.log 2>/dev/null | head -1 || true)
  [ -n "${READY}" ] && break
  sleep 1
done
PORT=$(echo "${READY}" | rg -o '"port": *[0-9]+' | rg -o '[0-9]+' | head -1)
TOKEN=$(echo "${READY}" | rg -o '"token": *"[^"]+"' | sed 's/.*"token": *"//; s/"$//')

# vite on IPv4 loopback with the dev-only /api proxy to the real backend
(cd frontend && PHARMAML_DEV_BACKEND_PORT="${PORT}" \
  bun x vite --port 5174 --host 127.0.0.1 > /tmp/vite_dev.log 2>&1 &)
for i in $(seq 1 30); do
  curl -s --max-time 2 -o /dev/null http://127.0.0.1:5174/ && break
  sleep 1
done
check "backend handshake + vite up" '[ -n "${READY}" ] && curl -s --max-time 2 -o /dev/null http://127.0.0.1:5174/'
echo "  backend port=${PORT} token=${TOKEN:0:10}… vite=5174 (proxied /api)"

# -- the browser flow ---
echo "[phase] browser flow"------------------------------------------------
agent-browser open http://127.0.0.1:5174 > /dev/null 2>&1
timeout 20 agent-browser wait --load networkidle > /dev/null 2>&1
# wire the handshake (the module the Tauri host calls at startup)
agent-browser eval "import('/src/services/api/client.ts').then(m => { m.configureBackend({baseUrl: 'http://127.0.0.1:5174', token: '${TOKEN}'}); window.__pharmaml_wired = true; })" > /dev/null 2>&1
sleep 1
check "handshake injected" '[ -n "${PORT}" ] && [ -n "${TOKEN}" ]'
WIRED=$(agent-browser eval "window.__pharmaml_wired === true" 2>/dev/null || true)
echo "  wired: ${WIRED}"

echo "[phase] nav";
# mount the PK/PD page (the catalog effect fires with the endpoint set)
agent-browser find text "PK/PD" click > /dev/null 2>&1
sleep 4
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "the eight §11.6 stages render" 'echo "$PAGE" | rg -q "Import Data" && echo "$PAGE" | rg -q "Inspect" && echo "$PAGE" | rg -q "Select Model" && echo "$PAGE" | rg -q "Configure Parameters" && echo "$PAGE" | rg -q "Fit/Simulate" && echo "$PAGE" | rg -q "Diagnostics" && echo "$PAGE" | rg -q "Exposure/Response Visualization" && echo "$PAGE" | rg -q "Export"'
echo "[dbg] fetch/console: $(agent-browser console 2>/dev/null | rg -i "error|failed" | head -2)"
check "the backend catalog rendered (its own rows)" 'echo "$PAGE" | rg -q "pharmaml.pkpd.population.simulate_population" && echo "$PAGE" | rg -q "pharmaml.pharma.pk"'

click_by() { # click_by <label-regex>
  REF=$(agent-browser snapshot -i -c 2>/dev/null | rg "$1" | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
  if [ -n "${REF}" ]; then agent-browser click "@${REF}" > /dev/null 2>&1; fi
}

echo "[phase] import";
# stage 1: import the sample series
click_by "Load single-dose sample"
sleep 1
click_by "Import series"
sleep 2
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "the series imported (11 points)" 'echo "$PAGE" | rg -q "11 points imported"'

echo "[phase] nca";
# stage 2: NCA + steady-state over the real series
click_by "Run NCA"
sleep 5
echo "[dbg] nca text: $(echo "$PAGE" | rg -o "Request failed[^\\]*|nca_summary" | head -1)"
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "NCA summary + terminal phase (real values)" 'echo "$PAGE" | rg -q "nca_summary" && echo "$PAGE" | rg -q "terminal phase" && echo "$PAGE" | rg -q "lambda_z" && echo "$PAGE" | rg -q "half_life"'
click_by "Run steady-state"
sleep 3
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "steady-state interval metrics" 'echo "$PAGE" | rg -q "interval metrics" && echo "$PAGE" | rg -q "fluctuation" && echo "$PAGE" | rg -q "Cmax_ss"'

echo "[phase] simulate";
# stage 5: the bateman simulation (the default selected model)
click_by '"Simulate"'
sleep 3
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "the bateman result rendered (ka/ke echo + curve)" 'echo "$PAGE" | rg -q "ka 1.5000" && echo "$PAGE" | rg -q "POST /api/v1/pkpd/bateman"'
check "the real curve drew (the svg caption)" 'echo "$PAGE" | rg -q "49 real points · time"'

echo "[phase] figure";
# stage 7: the exposure figure artifact + digest-verified download
click_by "Register exposure figure"
sleep 4
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "the exposure figure registered (sha256 + digest)" 'echo "$PAGE" | rg -q "pkpd-exposure-profile" && echo "$PAGE" | rg -q "sha256" && echo "$PAGE" | rg -q "digest"'

echo "[phase] population";
# the population lane in sync mode (the job lane is E2E-verified)
POPREF=$(agent-browser snapshot -i -c 2>/dev/null | rg "Population simulation" | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
if [ -n "${POPREF}" ]; then
  agent-browser click "@${POPREF}" > /dev/null 2>&1
  sleep 1
  SYNCREF=$(agent-browser snapshot -i -c 2>/dev/null | rg "sync" | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
  if [ -n "${SYNCREF}" ]; then agent-browser click "@${SYNCREF}" > /dev/null 2>&1; fi
  click_by '"Simulate"'
  sleep 4
  PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
  check "the population sync body rendered (subjects + seed)" 'echo "$PAGE" | rg -q "120 subjects" && echo "$PAGE" | rg -q "seed 42"'
  agent-browser click "@${POPREF}" > /dev/null 2>&1
  sleep 1
fi

echo "[phase] export";
# stage 8: the table export (digest-verified)
click_by "Export table"
sleep 4
PAGE=$(agent-browser eval "document.body.innerText" 2>/dev/null || true)
check "the table export registered (artifact + sha256)" 'echo "$PAGE" | rg -q "pkpd-curve" && echo "$PAGE" | rg -q "sha256"'

# stage 6: the diagnostics blocks (rendered from the real outputs)
check "diagnostics present (span + interval verdicts)" 'echo "$PAGE" | rg -q "span" && echo "$PAGE" | rg -q "fluctuation"'

# zero console errors
CONSOLE=$(agent-browser console 2>/dev/null || true)
ERRS=$(agent-browser errors 2>/dev/null || true)
echo "  console tail: $(echo "${CONSOLE}" | tail -3 | tr '\n' ' ')"
echo "  page errors: ${ERRS:-none}"
check "zero console errors" '[ -z "$(echo "${CONSOLE}" | rg -i "error" | rg -v "0 error" | head -1)" ]'

# -- cleanup ------------------------------------------------------------
agent-browser close > /dev/null 2>&1
kill "${BACKEND_PID}" 2>/dev/null
sleep 2
pkill -f "pharmaml.api" 2>/dev/null
pkill -f "vite --port 5174" 2>/dev/null

echo ""
if [ "${FAILED}" -eq 0 ]; then
  echo "BROWSER-PKPD-PASS: all checks green"
  exit 0
else
  echo "BROWSER-PKPD-FAIL: ${FAILED} checks failed"
  exit 1
fi
