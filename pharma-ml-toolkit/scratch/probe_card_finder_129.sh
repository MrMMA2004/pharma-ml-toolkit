#!/usr/bin/env bash
# 100x-129 — the card-finder E2E probe: every catalog entry jumps to the
# RIGHT card (title verified), the exit-animation race stays fixed, the
# hotkey works from a foreign top tab. The walk script is generated per
# entry by Python (no sed — '&' in queries broke the first draft).
set -u
BASE=http://localhost:3000
LOG=scratch/probe_card_finder_129.log
PASS=0; FAIL=0

/home/z/.venv/bin/python - << 'PY' > /tmp/walks_129.jsonl
import json
CATALOG = [
 ("Creatinine clearance", "Calculators"),
 ("Body surface area", "Calculators"),
 ("Pediatric dose", "Calculators"),
 ("Loading & maintenance dose", "Calculators"),
 ("Half-life", "Calculators"),
 ("trapezoidal", "Calculators"),
 ("Emax / sigmoid Hill", "Calculators"),
 ("Unit converter", "Calculators"),
 ("Regimen parameters", "PK / PD"),
 ("Concentration–time profile", "PK / PD"),
 ("effect–time profile", "PK / PD"),
 ("terminal fit", "PK / PD"),
 ("Experiment setup", "ML Training"),
 ("Probability Calibration", "ML Training"),
 ("Biomarker discovery", "Cell Line Bio"),
 ("Docking result parser", "Docking Parser"),
 ("Torsion QA", "Docking Parser"),
 ("results-folder scan", "Docking Parser"),
 ("Consensus rescore", "Docking Parser"),
 ("Feature factory", "Features"),
 ("ligand prep", "Ligand Prep"),
 ("Batch GPF/DPF", "Ligand Prep"),
 ("reconciliation", "Ligand Sets"),
 ("Session report", "STRIP"),
]
WALK = """(async () => {
  const q = %s;
  const expectTab = %s;
  // clear any flash left by the previous walk BEFORE opening the finder
  document.querySelectorAll(".pg-card-flash").forEach(e => e.classList.remove("pg-card-flash"));
  document.dispatchEvent(new KeyboardEvent("keydown", {key: "f", ctrlKey: true, shiftKey: true, bubbles: true, cancelable: true}));
  await new Promise(r => setTimeout(r, 400));
  let input = document.querySelector("input[aria-label='Find a card by title or keyword']");
  if (!input) return {error: "finder input never appeared"};
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(input, q);
  input.dispatchEvent(new Event("input", {bubbles: true}));
  await new Promise(r => setTimeout(r, 300));
  input.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", bubbles: true, cancelable: true}));
  for (let i = 0; i < 45; i++) {
    await new Promise(r => setTimeout(r, 100));
    const flashed = document.querySelector(".pg-card-flash");
    if (flashed) {
      const title = flashed.querySelector("[data-slot=card-title]")?.textContent ?? "?";
      const nav = document.querySelector("header button[aria-current=page]")?.textContent.trim() ?? "?";
      const sub = Array.from(document.querySelectorAll("[role=tab]")).filter(t => t.getAttribute("data-state") === "active").map(t => t.textContent.trim());
      await new Promise(r => setTimeout(r, 700)); // smooth scroll settle
      const rect = flashed.getBoundingClientRect();
      const inVp = rect.top < window.innerHeight && rect.bottom > 0;
      return {flashTitle: title, nav, activeTabs: sub, inViewport: inVp};
    }
  }
  return {error: "flash never appeared for " + q};
})()"""
for q, tab in CATALOG:
    print(json.dumps({"q": q, "tab": tab, "script": WALK % (json.dumps(q), json.dumps(tab))}))
PY

echo "== 100x-129 card finder probe ==" | tee "$LOG"
agent-browser open "$BASE" >/dev/null 2>&1; sleep 2
agent-browser find text "Overview" click >/dev/null 2>&1; sleep 1.2

COUNT=0
while IFS= read -r line; do
  COUNT=$((COUNT+1))
  Q=$(/home/z/.venv/bin/python -c "import sys, json; print(json.loads(sys.argv[1])['q'])" "$line")
  TAB=$(/home/z/.venv/bin/python -c "import sys, json; print(json.loads(sys.argv[1])['tab'])" "$line")
  SCRIPT=$(/home/z/.venv/bin/python -c "import sys, json; print(json.loads(sys.argv[1])['script'])" "$line")
  RAW=$(agent-browser eval "$SCRIPT" 2>/dev/null | tr '\n' ' ')
  VERDICT=$(/home/z/.venv/bin/python - "$Q" "$TAB" "$RAW" << 'PY'
import json, re, sys
q, tab, raw = sys.argv[1], sys.argv[2], sys.argv[3]
m = re.search(r"\{.*\}", raw, re.S)
if not m:
    print("bad: no json"); sys.exit()
try:
    data = json.loads(m.group(0))
except Exception as e:
    print(f"bad: parse {e}"); sys.exit()
if "error" in data:
    print(f"bad: {data['error']}"); sys.exit()
title = data.get("flashTitle", "")
nav = data.get("nav", "")
subs = " ".join(data.get("activeTabs", []))
problems = []
if nav != "Playground":
    problems.append(f"nav={nav}")
if tab != "STRIP" and tab not in subs:
    problems.append(f"sub={subs}")
if q.lower().split()[0].rstrip("s") not in title.lower():
    problems.append(f"title={title[:40]}")
if problems:
    print("bad: " + "; ".join(problems))
else:
    print("ok")
PY
)
  if [[ "$VERDICT" == ok* ]]; then
    PASS=$((PASS+1)); echo "  ok   $COUNT. $Q -> $TAB" | tee -a "$LOG"
  else
    FAIL=$((FAIL+1)); echo "  FAIL $COUNT. $Q -> $TAB [$VERDICT] raw=$RAW" | tee -a "$LOG"
  fi
done < /tmp/walks_129.jsonl

echo "== result: $PASS passed, $FAIL failed (of $COUNT) ==" | tee -a "$LOG"
if [ "$FAIL" -eq 0 ]; then echo "PROBE129_OK" | tee -a "$LOG"; else echo "PROBE129_FAILED" | tee -a "$LOG"; fi
