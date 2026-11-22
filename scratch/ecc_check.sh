#!/usr/bin/env bash
# ecc_check.sh — the standing round-check for the owner's r335 directive
# (upload/skill.txt): ECC fully installed, verified BEFORE every round.
# Checks the two install legs: the full clone (reference library) and the
# ~/.claude harness install (core profile + the four project-relevant
# capabilities). Deterministic, read-only, exits non-zero on any failure.
set -euo pipefail

ECC_REPO="/home/z/ECC"
CLAUDE_SKILLS="/home/z/.claude/skills"
STATE="/home/z/.claude/ecc/install-state.json"
EXPECT_VERSION="2.2.3"
EXPECT_MODULES="14"
FAIL=0

# 1. the clone leg
if [ ! -d "$ECC_REPO/.git" ]; then
  echo "FAIL: ECC clone missing at $ECC_REPO"; FAIL=1
else
  HEAD="$(git -C "$ECC_REPO" rev-parse HEAD)"
  VER="$(cat "$ECC_REPO/VERSION" 2>/dev/null || echo none)"
  [ "$VER" = "$EXPECT_VERSION" ] || { echo "FAIL: clone VERSION '$VER' != $EXPECT_VERSION"; FAIL=1; }
  echo "OK clone: HEAD ${HEAD:0:12} VERSION $VER ($(ls "$ECC_REPO/skills" | wc -l) skills in repo)"
fi

# 2. the harness install leg
if [ ! -f "$STATE" ]; then
  echo "FAIL: install-state missing at $STATE"; FAIL=1
else
  PROFILE="$(python3 -c "import json;print(json.load(open('$STATE'))['request']['profile'])")"
  NMOD="$(python3 -c "import json;print(len(json.load(open('$STATE'))['resolution']['selectedModules']))")"
  NSKILL="$(ls "$CLAUDE_SKILLS" 2>/dev/null | wc -l)"
  [ "$PROFILE" = "core" ] || { echo "FAIL: profile '$PROFILE' != core"; FAIL=1; }
  [ "$NMOD" = "$EXPECT_MODULES" ] || { echo "FAIL: selected modules $NMOD != $EXPECT_MODULES"; FAIL=1; }
  [ "${NSKILL:-0}" -ge 190 ] || { echo "FAIL: only $NSKILL skills installed (<190)"; FAIL=1; }
  echo "OK harness: profile $PROFILE, $NMOD modules, $NSKILL skills"
fi

# 3. the domain skills this repo's rounds actually use
for s in mle-workflow pytorch-patterns scientific-db-pubmed-database \
         hipaa-compliance docs-governance security-review deep-research \
         delivery-gate; do
  [ -f "$CLAUDE_SKILLS/$s/SKILL.md" ] || { echo "FAIL: domain skill '$s' missing"; FAIL=1; }
done
echo "OK domain skills: 8/8 present"

# 4. the four capability modules resolved in install-state
python3 - "$STATE" <<'EOF' || FAIL=1
import json, sys
mods = set(json.load(open(sys.argv[1]))['resolution']['selectedModules'])
need = {'machine-learning', 'research-apis', 'security', 'documentation-governance'}
missing = need - mods
print(("FAIL: capability modules missing: %s" % sorted(missing)) if missing else "OK capabilities: machine-learning, research-apis, security, documentation-governance all resolved")
sys.exit(1 if missing else 0)
EOF

[ "$FAIL" = "0" ] && echo "ECC-CHECK: GREEN" || { echo "ECC-CHECK: RED"; exit 1; }
