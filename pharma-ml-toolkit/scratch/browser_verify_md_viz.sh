#!/bin/bash
# r287 — the Molecular Dynamics Analysis + Visualization workspace
# browser verification (part11 §11.7/§11.8 rows 8/9), all inside ONE
# call (background processes do not survive between tool calls here):
# backend + vite + the agent-browser flow + the assertions + cleanup.
#
# The snapshot lists interactive elements as `button "Label" [ref=…]`,
# so every click matches the button line (never the same-text stage
# heading); long lanes (the study/report jobs, the figure renders)
# wait through polling loops, never fixed sleeps.
set -u
cd "$HOME/my-project/pharma-ml-toolkit"

FAILED=0
note() { echo "==[ $1 ]=="; }
check() { # check <name> <cond-string>
  if eval "$2"; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAILED=$((FAILED+1)); fi
}

# -- boot the real backend (port-0 handshake) --------------------------
rm -f /tmp/mdviz_backend.log
"$HOME/.venv/bin/python3" -m pharmaml.api > /tmp/mdviz_backend.log 2>&1 &
BACKEND_PID=$!

READY=""
for i in $(seq 1 40); do
  READY=$(rg -o 'PHARMAML-READY \{.*\}' /tmp/mdviz_backend.log 2>/dev/null | head -1 || true)
  [ -n "${READY}" ] && break
  sleep 1
done
PORT=$(echo "${READY}" | rg -o '"port": *[0-9]+' | rg -o '[0-9]+' | head -1)
TOKEN=$(echo "${READY}" | rg -o '"token": *"[^"]+"' | sed 's/.*"token": *"//; s/"$//')

(cd "$HOME/my-project/pharma-ml-toolkit/frontend" && PHARMAML_DEV_BACKEND_PORT="${PORT}" \
  bun x vite --port 5174 --host 127.0.0.1 > /tmp/vite_dev.log 2>&1 &)
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

# click the BUTTON whose accessible name matches the regex (never the
# same-text stage heading — the snapshot line starts with `button "`)
click_button() { # click_button <name-regex>
  REF=$(agent-browser snapshot -i -c 2>/dev/null | rg '^\s*- button "'"$1" | rg -o 'ref=e[0-9]+' | head -1 | rg -o 'e[0-9]+')
  if [ -n "${REF}" ]; then agent-browser click "@${REF}" > /dev/null 2>&1; fi
}

# poll the page text until a pattern appears (or the budget runs out)
wait_for() { # wait_for <pattern> <seconds>
  local pattern="$1" budget="$2" i
  for i in $(seq 1 "${budget}"); do
    if page_text | rg -q "${pattern}"; then return 0; fi
    sleep 1
  done
  return 1
}

# ================= the MD Analysis workspace (§11.7) ===================
note "MD workspace: navigation"
agent-browser find text "Molecular Dynamics Analysis" click > /dev/null 2>&1
sleep 4
PAGE=$(page_text)
check "the seven §11.7 stages render" 'echo "$PAGE" | rg -q "Load Topology/Trajectory" && echo "$PAGE" | rg -q "Validate" && echo "$PAGE" | rg -q "Select Analysis" && echo "$PAGE" | rg -q "Execute analysis" && echo "$PAGE" | rg -q "Inspect Results" && echo "$PAGE" | rg -q "Visualize" && echo "$PAGE" | rg -q "Export"'
check "the analysis-vs-simulation distinction renders" 'echo "$PAGE" | rg -q "does not run simulations" && echo "$PAGE" | rg -q "no MD engine is bundled or launched"'
check "the backend md catalog rendered (its own rows)" 'echo "$PAGE" | rg -q "pharmaml.md.trajectory \+ pharmaml.md.metrics" && echo "$PAGE" | rg -q "pharmaml.md.xvg" && echo "$PAGE" | rg -q "pharmaml.md.enm" && echo "$PAGE" | rg -q "viz.md_plots \(figure artifacts\)"'

note "MD workspace: load + validate the seeded demo trajectory"
click_button "Load & validate"
sleep 2
PAGE=$(page_text)
check "the seeded demo trajectory loaded" 'echo "$PAGE" | rg -q "the seeded demo trajectory — 80 frames over 12 residues"'

note "MD workspace: execute the trajectory metrics lane"
click_button "Execute analysis"
check "the metric bundle rendered (real values)" 'wait_for "metric bundle summary \(synthetic lane\)" 30'
sleep 1
PAGE=$(page_text)
check "the bundle summary + contact map + PCA rendered" 'echo "$PAGE" | rg -q "rmsd first / last" && echo "$PAGE" | rg -q "contact_map.strong_pair_fraction" && echo "$PAGE" | rg -q "pca.explained_variance_ratio\[0\]"'
check "the RMSD + Rg curves drew" 'echo "$PAGE" | rg -q "RMSD per frame" && echo "$PAGE" | rg -q "Radius of gyration per frame"'

note "MD workspace: the xvg stats lane"
click_button ".*series stats"
sleep 1
click_button "Execute analysis"
check "the xvg series + stats rendered" 'wait_for "11 points" 30'
sleep 1
PAGE=$(page_text)
check "the xvg stats table rendered (mean/std/cv)" 'echo "$PAGE" | rg -q "cv" && echo "$PAGE" | rg -q "std"'
check "the overlay curve caption rendered" 'echo "$PAGE" | rg -q "solid = the raw series"'

note "MD workspace: the md-series figure artifact"
click_button "Register md-series figure artifact"
check "the md-series figure registered (sha256 + digest)" 'wait_for "md-series.png" 30'
sleep 1
PAGE=$(page_text)
check "the md-series series stats rendered" 'echo "$PAGE" | rg -q "rmsd: 11 pts"'

note "MD workspace: the full study job lane (job mode, tracker-polled)"
click_button "Full study job"
sleep 1
click_button "Execute analysis"
check "the study job card rendered (operation + state)" 'wait_for "md.study · the full study bundle" 30'
check "the study job reached COMPLETED" 'wait_for "seed 7" 90'
sleep 1
PAGE=$(page_text)
check "the study body rendered (metrics + artifacts)" 'echo "$PAGE" | rg -q "study .MD study. . seed 7" && echo "$PAGE" | rg -q "rmsd first / last" && echo "$PAGE" | rg -q "md-rmsd.png" && echo "$PAGE" | rg -q "md-rg.png"'

note "MD workspace: the export lane (a fresh metrics run carries rows)"
click_button "Trajectory metrics"
sleep 1
click_button "Execute analysis"
check "the metric bundle re-rendered" 'wait_for "metric bundle summary \(synthetic lane\)" 30'
click_button "Export analysis rows"
check "the analysis rows exported (digest-verified)" 'wait_for "md-analysis.csv" 30'
sleep 1
PAGE=$(page_text)
check "the export digest verified" 'echo "$PAGE" | rg -q "digest verified"'

# ================= the Visualization workspace (§11.8) ================
note "Visualization workspace: navigation"
agent-browser find text "Visualization" click > /dev/null 2>&1
sleep 4
PAGE=$(page_text)
check "the seven §11.8 stages render" 'echo "$PAGE" | rg -q "Select Dataset/Analysis Result" && echo "$PAGE" | rg -q "Select Visualization" && echo "$PAGE" | rg -q "Configure Variables" && echo "$PAGE" | rg -q "Generate" && echo "$PAGE" | rg -q "Inspect" && echo "$PAGE" | rg -q "Filter/Compare" && echo "$PAGE" | rg -q "Export"'
check "the backend families catalog rendered (its own rows)" 'echo "$PAGE" | rg -q "pharmaml.viz.statplots" && echo "$PAGE" | rg -q "pharmaml.viz.pharma_plots" && echo "$PAGE" | rg -q "pharmaml.viz.md_plots" && echo "$PAGE" | rg -q "HTMLReport"'
check "the palette + export formats + themes rendered" 'echo "$PAGE" | rg -q "palette:" && echo "$PAGE" | rg -q "PNG, SVG, TIFF, JPEG" && echo "$PAGE" | rg -q "report themes:"'
check "the cross-workspace registry rendered (the MD artifacts)" 'echo "$PAGE" | rg -q "md-rmsd.png" || echo "$PAGE" | rg -q "md-series.png"'
check "the cross-workspace provenance note" 'echo "$PAGE" | rg -q "other workspaces"'

note "Visualization workspace: generate the stat histogram"
click_button "Generate"
check "the histogram stats rendered (the backend's own dict)" 'wait_for "histogram stats — 40 points" 30'
sleep 1
PAGE=$(page_text)
check "the stat figure artifact registered" 'echo "$PAGE" | rg -q "stat-histogram.png"'

note "Visualization workspace: the concentration profile family"
click_button "Concentration-time profile"
sleep 1
click_button "Generate"
check "the concentration profile rendered (10 points)" 'wait_for "concentration profile — 10 points" 30'
sleep 1
PAGE=$(page_text)
check "the concentration profile artifact registered" 'echo "$PAGE" | rg -q "concentration-profile.png"'

note "Visualization workspace: the dose accumulation family"
click_button "Dose accumulation"
sleep 1
click_button "Generate"
check "the dose accumulation rendered (tau + doses)" 'wait_for "dose accumulation — " 30'
sleep 1
PAGE=$(page_text)
check "the dose accumulation artifact registered" 'echo "$PAGE" | rg -q "dose-accumulation.png"'

note "Visualization workspace: the md-series family"
click_button "MD series"
sleep 1
click_button "Generate"
check "the md-series family rendered (1 series + stats)" 'wait_for "md-series — 1 series" 30'
sleep 1
PAGE=$(page_text)
check "the md-series stats row rendered" 'echo "$PAGE" | rg -q "rmsd" && echo "$PAGE" | rg -q "cv"'

note "Visualization workspace: the table family"
click_button "Table export"
sleep 1
click_button "Generate"
check "the table exported (10 rows)" 'wait_for "table — 10 rows" 30'
sleep 1
PAGE=$(page_text)
check "the table artifact registered" 'echo "$PAGE" | rg -q "visualization-table.csv"'

note "Visualization workspace: the report job lane (job mode, tracker-polled)"
click_button "HTML report"
sleep 1
click_button "Generate"
check "the report job card rendered (operation)" 'wait_for "visualization.report · the self-contained HTML report" 30'
check "the report job reached COMPLETED + the body relayed" 'wait_for "stages: " 90'
sleep 1
PAGE=$(page_text)
check "the report body rendered (title + figures + stages)" 'echo "$PAGE" | rg -q "PharmaML results" && echo "$PAGE" | rg -q "figures" && echo "$PAGE" | rg -q "stages: "'
check "the report artifact registered" 'echo "$PAGE" | rg -q "PharmaML results.html"'

note "Visualization workspace: the filter + compare pane"
click_button "Refresh"
sleep 3
PAGE=$(page_text)
check "the compare pane rendered (latest vs previous)" 'echo "$PAGE" | rg -q "latest —" && echo "$PAGE" | rg -q "previous —"'
check "the kind filter rendered (the registry's own kinds)" 'echo "$PAGE" | rg -q "figure \(" && echo "$PAGE" | rg -q "table \("'
echo "[dbg] console errors: $(agent-browser console 2>/dev/null | rg -i "error" | head -3)"

# -- cleanup ------------------------------------------------------------
[ -n "${BACKEND_PID}" ] && kill "${BACKEND_PID}" 2>/dev/null
pkill -f "vite --port 5174" 2>/dev/null
agent-browser close > /dev/null 2>&1 || true

note "summary"
echo "FAILED=${FAILED}"
exit ${FAILED}
