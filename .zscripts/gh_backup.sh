#!/usr/bin/env bash
# GitHub backup helper — auth via GIT_ASKPASS reading ~/.github_token (chmod 600).
# The token is NEVER written to .git/config, remote URLs, logs, or ps output.
set -euo pipefail
TOKEN_FILE="$HOME/.github_token"
[ -r "$TOKEN_FILE" ] || { echo "ERROR: $TOKEN_FILE missing — cannot authenticate" >&2; exit 1; }
ASKPASS="$HOME/.git_askpass_gh.sh"
printf '#!/usr/bin/env bash\ncase "$1" in *Username*) echo "MrMMA2004";; *) cat %s;; esac\n' "$TOKEN_FILE" > "$ASKPASS"
chmod 700 "$ASKPASS"
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 exec git push "${1:-origin}" "${2:-main}"
