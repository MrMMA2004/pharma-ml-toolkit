#!/usr/bin/env python3
"""r366 continuation: append the correction sentence to the counter's
last_run_note (the no-unsupported-claims close-out)."""
import json

P = "/home/z/my-project/scratch/100runs-counter.json"
d = json.load(open(P))
d["last_run_note"] = (
    d["last_run_note"]
    + " R366-CONTINUATION CORRECTION (the same firing's close-out, the "
    "no-unsupported-claims rule): two committed gate figures failed "
    "independent reproduction and are corrected at source in all five "
    "carriers -- the drift guards 202 -> 130 (the live set is the "
    "16-file tests/test_frontend_*.py family; no natural set composes "
    "202) and the API contracts 28 -> 6 (the live file carries 6, exit "
    "0). Every other r366 number re-verified true live (collection "
    "9,123/503, penetration 67, supply-chain+hygiene 479, portability "
    "2, retirement 13/13, version-carriers 7/7, validator --static "
    "27/27, github-page 11/11, vitest 346/346 over 39, lint exit 0); "
    "the part18 phase-3/4+5/6 slices executed fresh 108/165/104, all "
    "exit 0; the release evidence re-polled live (the v1.2.242 assets "
    "intact, CI at HEAD green, the six-home zip parity 9522afdf at "
    "27,950,886 B); the render verification re-flown over the "
    "production build (bun run build exit 0 + the standalone serve "
    "via the new scratch/standalone_daemon.py) 13/13 PASS incl. the "
    "three-way pin = walk = paint re-true (the walk 594,843 lines at "
    "1,860 files after this continuation's own +7 counted ledger "
    "delta, SHIPPED_SOURCE_LINES re-trued measurement-last); the "
    "interactive-browser harness wedged this continuation (five "
    "attempts dead) so the screenshot leg rides the committed "
    "pre-compaction captures, re-inspected genuine. Version unchanged "
    "1.2.242, no re-ship; runs_completed stays 158 (one firing, one "
    "bump); the correction rides its own commit."
)
json.dump(d, open(P, "w"), indent=2, ensure_ascii=False)
print("counter note updated; runs_completed =", d["runs_completed"])
