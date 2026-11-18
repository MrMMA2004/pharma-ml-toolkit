#!/usr/bin/env bash
# round-258 E2E (pass 2) — Playground tab first, then Molecular Dynamics
set -u
cd /home/z/my-project
step() { echo ""; echo "=== $1 ==="; }

step "0. server still up?"
curl -s -o /dev/null -w "GET / -> %{http_code}\n" --max-time 60 http://localhost:3000/

step "1. open / + click Playground top tab"
agent-browser open http://localhost:3000/ 2>&1 | tail -1
sleep 2
agent-browser find role tab "Playground" click 2>&1 | tail -1 || agent-browser find text "Playground" click 2>&1 | tail -1
sleep 2

step "2. click the Molecular Dynamics sub-tab"
agent-browser find text "Molecular Dynamics" click 2>&1 | tail -1
sleep 2

step "3. hero/badge stats (full JSON)"
agent-browser eval "(() => { const t = document.body.innerText; return {
  badge166: t.includes('1.2.166'),
  tests5373: t.includes('5,373'),
  lines333305: t.includes('333,305'),
}; })()" 2>&1

step "4. scroll to the CV card + textarea check"
agent-browser scrollintoview "text=Collective variables" 2>&1 | tail -1
sleep 1
agent-browser eval "(() => { const ta = document.getElementById('cv-plumed'); return {
  present: !!ta,
  hasFive: !!ta && ta.value.includes('d_ca: DISTANCE') && ta.value.includes('fold: COORDINATION'),
  lines: ta ? ta.value.split('\\n').length : 0,
}; })()" 2>&1

step "5. clear buffers + run the card"
agent-browser errors --clear >/dev/null 2>&1
agent-browser console --clear >/dev/null 2>&1
agent-browser click "text=Run collective variables" 2>&1 | tail -1
sleep 6

step "6. the card answered?"
agent-browser eval "(() => { const pre = document.getElementById('cv-colvar');
  const t = document.body.innerText;
  return {
    colvarPre: !!pre,
    header: pre ? pre.textContent.split('\\n')[0] : null,
    rows: pre ? pre.textContent.trim().split('\\n').length : 0,
    infoLine: t.includes('30 atoms') && t.includes('5 CVs'),
    chipCount: (t.match(/mean /g) || []).length,
    svgChart: !!document.querySelector('[aria-label*=\\'collective variables over\\']'),
    labels: ['d_ca','bend','twist','rg','fold'].every(l => t.includes(l)),
  }; })()" 2>&1

step "7. error path — malformed snippet"
agent-browser fill "#cv-plumed" "x: FOOBAR ATOMS=1,2" 2>&1 | tail -1
agent-browser click "text=Run collective variables" 2>&1 | tail -1
sleep 3
agent-browser eval "(() => { const t = document.body.innerText; return {
  refused: t.includes('unsupported action'),
  supportedListed: t.includes('DISTANCE, ANGLE, DIHEDRAL, COORDINATION, RMSD, GYRATION'),
}; })()" 2>&1

step "8. errors + console"
agent-browser errors 2>&1 | head -4
agent-browser console 2>&1 | grep -iE "error|warn" | head -4
echo "(empty = zero)"

step "9. screenshots + mobile"
agent-browser screenshot scratch/r258-cv-desktop.png 2>&1 | tail -1
agent-browser set viewport 390 844 2>&1 | tail -1
sleep 2
agent-browser eval "(() => { const d = document.documentElement; return {
  scrollW: d.scrollWidth, clientW: d.clientWidth,
  noHScroll: d.scrollWidth <= d.clientWidth,
  footer: !!document.querySelector('footer'),
}; })()" 2>&1
agent-browser screenshot scratch/r258-cv-mobile.png 2>&1 | tail -1

step "10. theme toggle both ways"
agent-browser set viewport 1280 900 >/dev/null 2>&1
agent-browser eval "(() => { const btn = document.querySelector('button[aria-label*=\\'theme\\' i]'); if (btn) btn.click(); return document.documentElement.classList.contains('dark'); })()" 2>&1
sleep 1
agent-browser eval "(() => { const btn = document.querySelector('button[aria-label*=\\'theme\\' i]'); if (btn) btn.click(); return document.documentElement.classList.contains('dark'); })()" 2>&1

step "11. zip endpoint from inside the browser"
agent-browser eval "fetch('/api/toolkit/zip-size').then(r => r.json()).then(d => window.__zs = d); 'ok'" >/dev/null 2>&1
sleep 2
agent-browser eval "JSON.stringify(window.__zs)" 2>&1 | head -2

agent-browser close 2>&1 | tail -1
echo "E2E-B DONE"
