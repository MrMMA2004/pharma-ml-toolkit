#!/bin/bash
# r288 — the part11 Phase Eighteen Three.js 3D lanes, browser
# verification (the §11.5 docking structure lane + the §11.7
# trajectory playback lane), all inside ONE call (background processes
# do not survive between tool calls here): backend + vite + the
# agent-browser flow + the assertions + cleanup.
#
# The headless chromium's WebGL may ride SwiftShader or be absent —
# either state is honestly asserted (the canvas lane OR the stated
# fallback card), never assumed.
set -u
cd "$HOME/my-project/pharma-ml-toolkit"

FAILED=0
note() { echo "==[ $1 ]=="; }
check() { # check <name> <cond-string>
  if eval "$2"; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAILED=$((FAILED+1)); fi
}

# -- boot the real backend (port-0 handshake) --------------------------
rm -f /tmp/3d_backend.log
"$HOME/.venv/bin/python3" -m pharmaml.api > /tmp/3d_backend.log 2>&1 &
BACKEND_PID=$!

READY=""
for i in $(seq 1 40); do
  READY=$(rg -o 'PHARMAML-READY \{.*\}' /tmp/3d_backend.log 2>/dev/null | head -1 || true)
  [ -n "${READY}" ] && break
  sleep 1
done
PORT=$(echo "${READY}" | rg -o '"port": *[0-9]+' | rg -o '[0-9]+' | head -1)
TOKEN=$(echo "${READY}" | rg -o '"token": *"[^"]+"' | sed 's/.*"token": *"//; s/"$//')

(cd "$HOME/my-project/pharma-ml-toolkit/frontend" && PHARMAML_DEV_BACKEND_PORT="${PORT}" \
  bun x vite --port 5174 --host 127.0.0.1 > /tmp/vite3d_dev.log 2>&1 &)
VITE_BG=$!
for i in $(seq 1 30); do
  curl -s --max-time 2 -o /dev/null http://127.0.0.1:5174/ && break
  sleep 1
done
check "backend handshake + vite up" '[ -n "${READY}" ] && curl -s --max-time 2 -o /dev/null http://127.0.0.1:5174/'
echo "  backend port=${PORT} token=${TOKEN:0:10}… vite=5174 (proxied /api)"

# -- the browser flow ---------------------------------------------------
agent-browser open http://127.0.0.1:5174 > /dev/null 2>&1
timeout 20 agent-browser wait --load networkidle > /dev/null 2>&1
agent-browser eval "import('/src/services/api/client.ts').then(m => { m.configureBackend({baseUrl: 'http://127.0.0.1:5174', token: '${TOKEN}'}); window.__pharmaml_wired = true; })" > /dev/null 2>&1
sleep 1
check "handshake injected" '[ -n "${PORT}" ] && [ -n "${TOKEN}" ]'

page_text() { agent-browser eval "document.body.innerText" 2>/dev/null || true; }
canvas_count() { agent-browser eval "document.querySelectorAll('canvas').length" 2>/dev/null | rg -o '[0-9]+' | head -1 || echo 0; }

click_button() { # click_button <name-regex>
  REF=$(agent-browser snapshot -i -c 2>/dev/null | rg '^\s*- button "'"$1" | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
  if [ -n "${REF}" ]; then agent-browser click "@${REF}" > /dev/null 2>&1; fi
}

wait_for() { # wait_for <pattern> <seconds>
  local pattern="$1" budget="$2" i
  for i in $(seq 1 "${budget}"); do
    if page_text | rg -q "${pattern}"; then return 0; fi
    sleep 1
  done
  return 1
}

# the WebGL capability of this headless chromium, probed through the
# page itself (either lane is honest — asserted, never assumed)
WEBGL_STATE=$(agent-browser eval "(() => { try { const c = document.createElement('canvas'); const gl = c.getContext('webgl2') || c.getContext('webgl'); return gl ? 'available' : 'unavailable'; } catch (e) { return 'unavailable'; } })()" 2>/dev/null | rg -o 'available|unavailable' | head -1 || echo unavailable)
echo "  headless WebGL: ${WEBGL_STATE}"

# ============ the Molecular Docking workspace (§11.5, the 3D lane) ============
note "Docking workspace: prepare the ligand, then the 3D structure lane"
agent-browser find text "Molecular Docking" click > /dev/null 2>&1
sleep 4
PAGE=$(page_text)
check "the §11.5 workspace renders (ten stages)" 'echo "$PAGE" | rg -q "Select Ligands" && echo "$PAGE" | rg -q "Prepare Ligands" && echo "$PAGE" | rg -q "Visualize" && echo "$PAGE" | rg -q "Export"'

click_button "Prepare ligand PDBQT"
check "the ligand PDBQT artifact registered" 'wait_for "artifact" 45'
sleep 3
PAGE=$(page_text)
check "the ligand artifact card rendered (size + sha)" 'echo "$PAGE" | rg -q "B ·" || echo "$PAGE" | rg -q "bytes"'

note "Docking workspace: load the ligand into the 3D view"
check "the 3D load button renders" 'page_text | rg -q "Load the prepared ligand in 3D"'
click_button "Load the prepared ligand in 3D"
check "the ligand 3D card rendered" 'wait_for "Ligand structure \(3D\)" 30'
sleep 2
PAGE=$(page_text)
if [ "${WEBGL_STATE}" = "available" ]; then
  check "the WebGL canvas mounted" '[ "$(canvas_count)" -ge 1 ]'
else
  check "the honest WebGL fallback rendered" 'echo "$PAGE" | rg -q "WebGL unavailable — the 3D view is disabled"'
fi
check "the display-bond honesty note rendered" 'echo "$PAGE" | rg -q "distance heuristic" && echo "$PAGE" | rg -q "never scientific output"'
check "the never-fabricated-structure note rendered" 'echo "$PAGE" | rg -q "Pose coordinates are not parsed"'
check "the receptor 3D fallback note rendered (empty paste lane)" 'echo "$PAGE" | rg -q "paste ATOM/HETATM records and the ball-and-stick scene draws"'

# ============ the MD Analysis workspace (§11.7, the playback lane) ============
note "MD workspace: the inline stack + the trajectory playback"
agent-browser find text "Molecular Dynamics Analysis" click > /dev/null 2>&1
sleep 4
PAGE=$(page_text)
check "the §11.7 workspace renders" 'echo "$PAGE" | rg -q "Load Topology/Trajectory" && echo "$PAGE" | rg -q "Validate"'

check "the playback fallback note renders before a stack is loaded" 'page_text | rg -q "load a trajectory source in stage one first"'
click_button "Inline coordinate stack"
sleep 1
click_button "Load & validate"
check "the inline stack validated (3 frames x 4 atoms)" 'wait_for "the inline stack validates: 3 frames" 20'
sleep 2
PAGE=$(page_text)
check "the trajectory playback card rendered" 'echo "$PAGE" | rg -q "Trajectory playback \(3D\)"'
check "the frame lane rendered (Frame 1 / 3)" 'echo "$PAGE" | rg -q "Frame 1 / 3"'
check "the uniform-element honesty note rendered" 'echo "$PAGE" | rg -q "honestly uniform"'
check "the display-bond honesty note rendered" 'echo "$PAGE" | rg -q "display only" && echo "$PAGE" | rg -q "coordinates that actually arrived"'
if [ "${WEBGL_STATE}" = "available" ]; then
  check "the playback canvas mounted" '[ "$(canvas_count)" -ge 1 ]'
  # step the frame slider through the native value setter (React's
  # value tracker dedupes a bare .value assignment — the r288 find)
  agent-browser eval "(() => { const s = document.querySelector(\"input[type='range'][aria-label='Trajectory frame']\"); if (!s) return 'no-slider'; const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set; setter.call(s, '1'); s.dispatchEvent(new Event('input', { bubbles: true })); return 'stepped'; })()" > /dev/null 2>&1
  sleep 2
  check "the frame stepper advanced (Frame 2 / 3)" 'page_text | rg -q "Frame 2 / 3"'
fi

# ============ the console + error sweep =====================================
note "the console error sweep"
CONSOLE_ERRORS=$(agent-browser errors 2>/dev/null | rg -c . || echo 0)
CONSOLE_LOGS=$(agent-browser console 2>/dev/null | rg -i "error|uncaught|failed to fetch" | rg -cv "^\s*$" || echo 0)
check "zero page errors" '[ "${CONSOLE_ERRORS}" = "0" ]'
check "zero error-class console messages" '[ "${CONSOLE_LOGS}" = "0" ]'

note "cleanup"
kill "${BACKEND_PID}" 2>/dev/null
agent-browser close > /dev/null 2>&1
sleep 1

echo ""
if [ "${FAILED}" = "0" ]; then
  echo "BROWSER-3D-PASS (WebGL: ${WEBGL_STATE})"
else
  echo "BROWSER-3D-FAILED: ${FAILED} check(s) failed (WebGL: ${WEBGL_STATE})"
fi
exit "${FAILED}"
