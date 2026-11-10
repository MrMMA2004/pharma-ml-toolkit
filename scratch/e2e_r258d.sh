#!/usr/bin/env bash
set -u
cd /home/z/my-project
step() { echo ""; echo "=== $1 ==="; }

step "1. open + JS-click the exact Playground nav button"
agent-browser open http://localhost:3000/ 2>&1 | tail -1
sleep 2
agent-browser eval "(() => { const btn = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Playground'); if (btn) { btn.click(); return 'clicked'; } return 'notfound'; })()" 2>&1
sleep 3

step "2. JS-click the Molecular Dynamics sub-tab"
agent-browser eval "(() => { const btn = [...document.querySelectorAll('[role=tab], button')].find(b => b.textContent.trim() === 'Molecular Dynamics'); if (btn) { btn.click(); return 'clicked'; } return 'notfound'; })()" 2>&1
sleep 3

step "3. scroll to the CV card + textarea"
agent-browser eval "(() => { const el = [...document.querySelectorAll('[data-slot=card-title]')].find(t => t.textContent.includes('Collective variables')); if (el) el.scrollIntoView({block: 'center'}); return el ? 'scrolled' : 'card-notfound'; })()" 2>&1
sleep 1
agent-browser eval "(() => { const ta = document.getElementById('cv-plumed'); return {
  present: !!ta,
  hasFive: !!ta && ta.value.includes('d_ca: DISTANCE') && ta.value.includes('fold: COORDINATION'),
  lines: ta ? ta.value.split('\\n').length : 0,
}; })()" 2>&1

step "4. clear buffers + run the card"
agent-browser errors --clear >/dev/null 2>&1
agent-browser console --clear >/dev/null 2>&1
agent-browser eval "(() => { const btn = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Run collective variables'); if (btn) { btn.click(); return 'clicked'; } return 'notfound'; })()" 2>&1
sleep 7

step "5. the card answered?"
agent-browser eval "(() => { const pre = document.getElementById('cv-colvar');
  const t = document.body.innerText;
  return {
    colvarPre: !!pre,
    header: pre ? pre.textContent.split('\\n')[0] : null,
    rows: pre ? pre.textContent.trim().split('\\n').length : 0,
    infoLine: t.includes('30 atoms') && t.includes('5 CVs'),
    chipCount: (t.match(/mean /g) || []).length,
    svgChart: !!document.querySelector('[aria-label*=\\'collective variables over\\']'),
    allLabels: ['d_ca','bend','twist','rg','fold'].every(l => t.includes(l)),
    copyBtn: t.includes('Copy COLVAR'),
  }; })()" 2>&1

step "6. error path — malformed snippet"
agent-browser fill "#cv-plumed" "x: FOOBAR ATOMS=1,2" 2>&1 | tail -1
agent-browser eval "(() => { const btn = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Run collective variables'); if (btn) { btn.click(); return 'clicked'; } return 'notfound'; })()" 2>&1
sleep 3
agent-browser eval "(() => { const t = document.body.innerText; return {
  refused: t.includes('unsupported action'),
  supportedListed: t.includes('DISTANCE, ANGLE, DIHEDRAL, COORDINATION, RMSD, GYRATION'),
}; })()" 2>&1

step "7. errors + console"
agent-browser errors 2>&1 | head -4
agent-browser console 2>&1 | grep -iE "error|warn" | head -4
echo "(empty = zero)"

step "8. desktop screenshot"
agent-browser screenshot scratch/r258-cv-desktop.png 2>&1 | tail -1

step "9. mobile 390x844"
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
agent-browser eval "(() => { const btn = document.querySelector('button[aria-label*=\\'theme\\' i]'); if (btn) btn.click(); return 'now-dark=' + document.documentElement.classList.contains('dark'); })()" 2>&1
sleep 1
agent-browser eval "(() => { const btn = document.querySelector('button[aria-label*=\\'theme\\' i]'); if (btn) btn.click(); return 'now-dark=' + document.documentElement.classList.contains('dark'); })()" 2>&1

agent-browser close 2>&1 | tail -1
echo "E2E-D DONE"
