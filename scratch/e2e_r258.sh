#!/usr/bin/env bash
# round-258 E2E — the COLVAR playground card + pins + mobile + theme + zip
set -u
cd /home/z/my-project
step() { echo ""; echo "=== $1 ==="; }

step "1. open / (warm server)"
agent-browser open http://localhost:3000/ 2>&1 | tail -2
sleep 2

step "2. version badge + hero stats"
agent-browser eval "(() => { const t = document.body.innerText; return {
  badge166: t.includes('1.2.166'),
  tests5373: t.includes('5,373'),
  lines333305: t.includes('333,305'),
  files1093: t.includes('1,093'),
}; })()" 2>&1 | head -3

step "3. clear error buffers, then click the Molecular Dynamics tab"
agent-browser errors --clear >/dev/null 2>&1
agent-browser console --clear >/dev/null 2>&1
agent-browser find text "Molecular Dynamics" click 2>&1 | tail -1
sleep 2

step "4. scroll to the CV card"
agent-browser scrollintoview "text=Collective variables (COLVAR)" 2>&1 | tail -1
sleep 1

step "5. textarea carries the default snippet?"
agent-browser eval "(() => { const ta = document.getElementById('cv-plumed'); return {
  present: !!ta,
  hasFive: !!ta && ta.value.includes('d_ca: DISTANCE') && ta.value.includes('fold: COORDINATION'),
  lines: ta ? ta.value.split('\\n').length : 0,
}; })()" 2>&1 | head -3

step "6. click Run collective variables"
agent-browser click "text=Run collective variables" 2>&1 | tail -1
sleep 5

step "7. the card answered?"
agent-browser eval "(() => { const pre = document.getElementById('cv-colvar');
  const t = document.body.innerText;
  return {
    colvarPre: !!pre,
    header: pre ? pre.textContent.split('\\n')[0] : null,
    rows: pre ? pre.textContent.trim().split('\\n').length : 0,
    infoLine: t.includes('30 atoms') && t.includes('5 CVs'),
    statsChips: (t.match(/mean /g) || []).length,
    svgChart: !!document.querySelector('[aria-label*=\\'collective variables over\\']'),
  }; })()" 2>&1 | head -3

step "8. error path — a malformed snippet"
agent-browser fill "#cv-plumed" "x: FOOBAR ATOMS=1,2" 2>&1 | tail -1
agent-browser click "text=Run collective variables" 2>&1 | tail -1
sleep 3
agent-browser eval "(() => { const t = document.body.innerText; return {
  refused: t.includes('unsupported action'),
  supportedListed: t.includes('DISTANCE, ANGLE, DIHEDRAL, COORDINATION, RMSD, GYRATION'),
}; })()" 2>&1 | head -3
agent-browser fill "#cv-plumed" "# the helix demo: 10 residues x (N, CA, C), atoms 1-30
d_ca: DISTANCE ATOMS=2,5
bend: ANGLE ATOMS=2,5,8
twist: DIHEDRAL ATOMS=1,2,5,8
rg: GYRATION ATOMS=1-30
fold: COORDINATION GROUPA=1-15 GROUPB=16-30 R_0=6.5" 2>&1 | tail -1

step "9. errors + console"
agent-browser errors 2>&1 | head -6
agent-browser console 2>&1 | grep -iE "error|warn" | head -6
echo "(empty above = zero errors)"

step "10. desktop screenshot"
agent-browser screenshot scratch/r258-cv-desktop.png 2>&1 | tail -1

step "11. mobile 390x844"
agent-browser set viewport 390 844 2>&1 | tail -1
sleep 2
agent-browser eval "(() => { const d = document.documentElement; return {
  scrollW: d.scrollWidth, clientW: d.clientWidth,
  noHScroll: d.scrollWidth <= d.clientWidth,
  footer: !!document.querySelector('footer'),
}; })()" 2>&1 | head -3
agent-browser screenshot scratch/r258-cv-mobile.png 2>&1 | tail -1

step "12. theme toggle probe"
agent-browser set viewport 1280 900 >/dev/null 2>&1
agent-browser eval "(() => { const btn = document.querySelector('button[aria-label*=\\'theme\\' i]'); return { found: !!btn, label: btn ? btn.getAttribute('aria-label') : null }; })()" 2>&1 | head -2

step "13. zip endpoint from inside the browser"
agent-browser eval "fetch('/api/toolkit/zip-size').then(r => r.json()).then(d => window.__zs = d).catch(e => window.__zs = {err: String(e)}); 'fetching'" >/dev/null 2>&1
sleep 2
agent-browser eval "JSON.stringify(window.__zs)" 2>&1 | head -2

step "14. close"
agent-browser close 2>&1 | tail -1
echo "E2E DONE"
