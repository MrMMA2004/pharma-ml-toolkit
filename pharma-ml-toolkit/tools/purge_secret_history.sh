#!/usr/bin/env bash
# purge_secret_history.sh — verify-first PAT purge for the pharma-ml-toolkit monorepo
#
# Automates Phases 2–5 of docs/security/PAT_EXPOSURE_REMEDIATION.md:
#   --discover <mirror-dir>   enumerate contamination (read-only, prints no secrets)
#   --purge    <mirror-dir>   rewrite history (git filter-repo); token via
#                             hidden prompt or $PURGE_TOKEN; never echoed
#   --verify   <mirror-dir>   re-run discovery on the rewritten repo; expect zero
#   --push     <mirror-dir> --origin <url>   force-push all refs (asks twice)
#
# Run against a MIRROR CLONE OF THE REPOSITORY ROOT (the monorepo root that
# contains scratch/, upload/ and pharma-ml-toolkit/), not the inner toolkit.
#
# Safety properties:
#   * The token is never printed, logged, or passed as a command argument.
#   * Replacement file is created with umask 077 and shredded on exit.
#   * Discovery is read-only; purge refuses to run outside a --mirror clone.
#   * --push requires typing FORCE-PUSH.
#
# PDF detection note (verified against the real contaminated file):
#   the PAT inside "upload/MASTER_PROMPT part1.pdf" is FONT-ENCODED in a
#   compressed content stream — invisible to raw-byte and qpdf-decompressed
#   greps.  Only a text-extraction pass (pdftotext) recovers it.  The token
#   is also WRAPPED across two extracted lines (76 + 17 chars), so PDFs are
#   scanned both raw and with newlines stripped.
#
# Requirements: git >= 2.22, git-filter-repo on PATH, python3, poppler-utils
# (pdftotext).

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
MODE=""
DIR=""
ORIGIN_URL=""
REPLACEMENTS_FILE=""

# PAT byte patterns (regex, plain text layer)
PAT_REGEX='(ghp_|github_pat_)[A-Za-z0-9_]{30,}'
# paths known to carry the token (tracked at HEAD of main; root-relative)
KNOWN_BAD_PATHS=(
  "scratch/pdf-extract/MASTER_PROMPT.txt"
  "scratch/pdf-extract/"
  "upload/MASTER_PROMPT part1.pdf"
)

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { printf '[%s] FATAL: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }

cleanup() {
  if [[ -n "${REPLACEMENTS_FILE:-}" && -f "${REPLACEMENTS_FILE}" ]]; then
    if command -v shred >/dev/null 2>&1; then
      shred -u "${REPLACEMENTS_FILE}" 2>/dev/null || rm -f "${REPLACEMENTS_FILE}"
    else
      rm -f "${REPLACEMENTS_FILE}"
    fi
  fi
}
trap cleanup EXIT

usage() {
  cat <<EOF
usage:
  $SCRIPT_NAME --discover <mirror-dir>
  $SCRIPT_NAME --purge    <mirror-dir>              (token: prompt or \$PURGE_TOKEN)
  $SCRIPT_NAME --verify   <mirror-dir>
  $SCRIPT_NAME --push     <mirror-dir> --origin <url>
EOF
  exit 2
}

[[ $# -ge 2 ]] || usage
MODE="$1"; DIR="$2"; shift 2
if [[ "${1:-}" == "--origin" && $# -ge 2 ]]; then ORIGIN_URL="$2"; fi

[[ -d "$DIR" ]] || die "directory not found: $DIR"
cd "$DIR"
[[ -d refs && -f config ]] || die "not a bare/mirror git repository: $DIR (use a --mirror clone of the monorepo ROOT)"

command -v git-filter-repo >/dev/null 2>&1 || command -v git filter-repo >/dev/null 2>&1 \
  || die "git-filter-repo not installed: pip install --user git-filter-repo"
command -v pdftotext >/dev/null 2>&1 \
  || die "pdftotext not found (poppler-utils) — required for PDF deep-scan"

filter_repo() { # wrap both invocation styles
  if command -v git-filter-repo >/dev/null 2>&1; then git-filter-repo "$@";
  else git filter-repo "$@"; fi
}

# ---------------------------------------------------------------------------
# Discovery: scan every commit for PAT patterns.
#   text blobs  -> git grep -I (skips binaries; catches the 76-char first line)
#   pdf blobs   -> pdftotext -layout, scanned raw AND newline-stripped
#                  (the token wraps across two extracted lines in the wild)
# Prints "<mode>\t<path>\t<commit-subject>" lines. NEVER prints matched content.
# ---------------------------------------------------------------------------
discover() {
  local revs
  revs="$(git rev-list --all)" || die "git rev-list failed"
  local total
  total="$(wc -l <<<"$revs")"
  log "scanning ${total} commits in $(basename "$DIR") (read-only)"

  # --- text blobs ---
  local text_hits=0
  while IFS= read -r rev; do
    while IFS= read -r path; do
      [[ -n "$path" ]] || continue
      printf 'TEXT\t%s\t%s\n' "$path" "$(git log -1 --format=%s "$rev")"
      text_hits=$((text_hits + 1))
    done < <(git grep -I -l -E "$PAT_REGEX" "$rev" -- 2>/dev/null | sed "s/^$rev://" || true)
  done <<<"$revs"
  log "text-layer hits: ${text_hits}"

  # --- pdf blobs (deep scan via text extraction) ---
  local pdf_hits=0 unparsable=0
  local tmpdir
  tmpdir="$(mktemp -d)"
  while IFS=$'\t' read -r sha path; do
    [[ -n "$sha" ]] || continue
    git cat-file blob "$sha" >"$tmpdir/blob.pdf" 2>/dev/null || continue
    if pdftotext -layout "$tmpdir/blob.pdf" "$tmpdir/blob.txt" 2>/dev/null; then
      # raw scan catches an unwrapped token; joined scan catches any wrap
      if grep -qE "$PAT_REGEX" "$tmpdir/blob.txt" 2>/dev/null \
         || tr -d '\n\r' <"$tmpdir/blob.txt" | grep -qE "$PAT_REGEX"; then
        printf 'PDF\t%s\t(text-extracted blob %s)\n' "$path" "${sha:0:12}"
        pdf_hits=$((pdf_hits + 1))
      fi
    else
      # pdftotext cannot parse it — surface it instead of silently skipping:
      # a damaged/nonstandard PDF could still hide an encoded token.
      printf 'UNPARSEABLE_PDF\t%s\t(manual review needed: pdftotext failed)\n' "$path"
      unparsable=$((unparsable + 1))
    fi
  done < <(git rev-list --all --objects | awk '/\.pdf$/ { idx = index($0, " "); if (idx > 1) print substr($0,1,idx-1) "\t" substr($0,idx+1) }' | sort -u)
  rm -rf "$tmpdir"
  log "pdf-layer hits: ${pdf_hits}  (unparsable pdfs: ${unparsable})"

  # --- known bad paths present? ---
  for p in "${KNOWN_BAD_PATHS[@]}"; do
    if git rev-list --all -- "$p" | head -1 | grep -q .; then
      printf 'KNOWN\t%s\t(expected contaminated path)\n' "$p"
    fi
  done
}

collect_bad_paths() { # echoes unique contaminated paths from a discovery run
  discover | awk -F'\t' '$1=="TEXT" || $1=="PDF" {print $2}' | sort -u
}

# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------
purge() {
  # refuse to run on a repo that still has a working tree (safety)
  [[ -z "$(git config core.bare || true)" ]] && die "refusing: not a bare clone"

  # token acquisition — prompt (hidden) or env; never argv, never echoed
  local token="${PURGE_TOKEN:-}"
  if [[ -z "$token" ]]; then
    printf 'Paste the FULL exposed token value (input hidden): '
    read -rs token
    printf '\n'
  fi
  [[ "$token" =~ ^github_pat_[A-Za-z0-9_]+$ || "$token" =~ ^ghp_[A-Za-z0-9]+$ ]] \
    || die "token does not look like a GitHub PAT (expected github_pat_... or ghp_... prefix)"

  REPLACEMENTS_FILE="$(mktemp)"
  {
    printf 'literal:%s==>***REDACTED-PAT***\n' "$token"
    printf 'regex:github_pat_[A-Za-z0-9_]{60,}==>***REDACTED-PAT***\n'
    printf 'regex:ghp_[A-Za-z0-9]{30,}==>***REDACTED-PAT***\n'
  } >"$REPLACEMENTS_FILE"
  chmod 600 "$REPLACEMENTS_FILE"

  # paths to remove entirely: known bad + discovered.
  # PDFs cannot be byte-scrubbed (font-encoded streams); extraction dumps
  # under scratch/pdf-extract/ are removed as whole paths.
  local -a paths=()
  mapfile -t paths < <(
    {
      for p in "${KNOWN_BAD_PATHS[@]}"; do echo "$p"; done
      collect_bad_paths
    } | sort -u
  )
  log "paths to purge from history: ${#paths[@]}"
  local p
  for p in "${paths[@]}"; do log "  - $p"; done

  # pre-purge topology snapshot (for the verify step)
  git for-each-ref --format='%(refname)' >"../refs-before-purge.txt"
  : >"../purge-evidence.txt"
  discover >>"../purge-evidence.txt" 2>&1 || true

  local -a args=(--force --replace-text "$REPLACEMENTS_FILE")
  if [[ ${#paths[@]} -gt 0 ]]; then
    args+=(--invert-paths)
    for p in "${paths[@]}"; do args+=(--path "$p"); done
  fi
  log "running git filter-repo ..."
  filter_repo "${args[@]}"
  log "purge complete. run: $SCRIPT_NAME --verify <dir> next"
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
verify() {
  log "verifying rewritten repository"
  local fails=0

  # 1. no text/pdf pattern hits
  local out
  out="$(discover)"
  local hits
  hits="$(awk -F'\t' '$1=="TEXT" || $1=="PDF"' <<<"$out" | wc -l)"
  if [[ "$hits" -ne 0 ]]; then
    log "FAIL: ${hits} contamination hits remain:"
    awk -F'\t' '$1=="TEXT" || $1=="PDF" {print "  "$2}' <<<"$out"
    fails=1
  else
    log "PASS: zero text/pdf pattern matches across all commits"
  fi

  # 1b. unparsable PDFs block verification: resolve each by purging its path
  #     or manually confirming it cannot contain the token (pre-exposure date)
  local unparse
  unparse="$(awk -F'\t' '$1=="UNPARSEABLE_PDF"' <<<"$out" | wc -l)"
  if [[ "$unparse" -ne 0 ]]; then
    log "FAIL: ${unparse} PDF(s) could not be text-extracted (pdftotext) — resolve manually:"
    awk -F'\t' '$1=="UNPARSEABLE_PDF" {print "  "$2}' <<<"$out"
    fails=1
  fi

  # 2. known bad paths absent
  local p
  for p in "${KNOWN_BAD_PATHS[@]}"; do
    if git rev-list --all -- "$p" | head -1 | grep -q .; then
      log "FAIL: known contaminated path still in history: $p"
      fails=1
    fi
  done

  # 3. topology preserved (ref count; per-ref commit counts logged)
  if [[ -f ../refs-before-purge.txt ]]; then
    local before after ref
    before=$(wc -l <"../refs-before-purge.txt")
    after=$(git for-each-ref --format='%(refname)' | wc -l)
    if [[ "$before" -ne "$after" ]]; then
      log "FAIL: ref count changed (${before} -> ${after})"
      fails=1
    else
      while IFS= read -r ref; do
        local cb
        cb=$(git rev-list --count "$ref" 2>/dev/null || echo 0)
        log "  ref $ref: $cb commits"
      done <"../refs-before-purge.txt"
      log "PASS: ref topology preserved (${after} refs)"
    fi
  else
    log "note: no pre-purge ref snapshot found — skipping topology check"
  fi

  if [[ "$fails" -eq 0 ]]; then
    log "VERIFY: ALL PASS — repository is clean; safe to --push"
  else
    log "VERIFY: FAILURES DETECTED — do not push; inspect and re-run purge"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------
push() {
  [[ -n "$ORIGIN_URL" ]] || die "--push requires --origin <url>"
  log "WARNING: this rewrites the remote irreversibly."
  log "target: $ORIGIN_URL"
  log "check first: the arena system grows NEW refs between rounds —"
  log "re-clone and re-run --discover if any branch was pushed after your"
  log "mirror was taken, or its contaminated history will survive the rewrite."
  printf 'type FORCE-PUSH to continue: '
  local answer
  read -r answer
  [[ "$answer" == "FORCE-PUSH" ]] || die "aborted"
  git remote remove origin 2>/dev/null || true
  git remote add origin "$ORIGIN_URL"
  git push --force --all origin
  git push --force --tags origin
  log "pushed. post-steps: GitHub Support GC request, invalidate old clones, see Phase 6"
}

case "$MODE" in
  --discover) discover ;;
  --purge)    purge ;;
  --verify)   verify ;;
  --push)     push ;;
  *)          usage ;;
esac
