#!/usr/bin/env bash
# Credential wrapper: git fetch/push against origin WITHOUT ever exposing
# the PAT in argv, env, logs, or remote URLs.  The token is read fresh
# from the 600-mode credential file into a askpass helper that lives in
# a root-only tmp dir and is deleted on exit.
set -euo pipefail
CRED=/home/z/my-project/.credentials/github_pat
[ -f "$CRED" ] || { echo "credential file missing" >&2; exit 1; }
[ "$(stat -c %a "$CRED")" = "600" ] || chmod 600 "$CRED"
TMPD=$(mktemp -d)          # mode 700 by default
cat > "$TMPD/askpass" <<EOF
#!/bin/bash
case "\$1" in
  *Username*) printf '%s' 'x-access-token' ;;
  *Password*) cat '$CRED' ;;
esac
EOF
chmod 700 "$TMPD/askpass"
trap 'rm -rf "$TMPD"' EXIT
cd /home/z/my-project
export GIT_ASKPASS="$TMPD/askpass"
export GIT_TERMINAL_PROMPT=0
git "$@"
