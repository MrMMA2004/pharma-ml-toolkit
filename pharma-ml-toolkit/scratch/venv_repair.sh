#!/usr/bin/env bash
# venv_repair.sh -- the discoverability wrapper (round 217, the 14th rollback).
#
# The 14th-rollback recovery (2026-11-07) hit the gap this file closes: the
# runbook's `bash scratch/venv_repair.sh` was typed from pharma-ml-toolkit/
# (where every other round command runs) and died with "No such file or
# directory" -- the canonical repair script lives at the REPO ROOT's
# scratch/ (scratch/venv_repair.sh, the round-196 rewrite).  This wrapper
# makes BOTH relative paths work with zero content duplication: it exec's
# the canonical script, so there is exactly one repair recipe to maintain.
#
# Superseded copies stay for their round's history: venv_repair_9.sh (this
# directory) predates the round-196 resolver-proof rewrite -- do not run it.
#
# The canonical script is idempotent and MUST run in the FOREGROUND
# (background pip dies in this sandbox -- three losses across rounds 194-196).

set -euo pipefail

CANDIDATES=(
    "${HOME}/my-project/scratch/venv_repair.sh"
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scratch/venv_repair.sh"
)
for candidate in "${CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
        echo "venv_repair wrapper: exec'ing the canonical $candidate"
        exec bash "$candidate" "$@"
    fi
done

echo "venv_repair wrapper: the canonical script was not found at either of:" >&2
for candidate in "${CANDIDATES[@]}"; do
    echo "  $candidate" >&2
done
exit 127
