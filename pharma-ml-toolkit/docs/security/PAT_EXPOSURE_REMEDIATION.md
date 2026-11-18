# PAT Exposure Remediation — pharma-ml-toolkit monorepo

**Status:** READY TO EXECUTE (needs an owner decision + ~15 minutes)
**Scope:** `MrMMA2004/pharma-ml-toolkit` (private) — full git history, all refs
**Severity:** CRITICAL — the exposed fine-grained PAT is **confirmed still active**
(a fresh clone authenticated with it on 2026-11-12) and is embedded in **two files
tracked at the HEAD of `main`**.
**Prepared:** 2026-11-12, `part5-continuation` round 1 (supersedes the staging-repo
draft; re-baselined onto `main` @ `e2f88d1`, v1.2.174, remote main-only).

---

## 1. Incident summary

A GitHub fine-grained PAT (`github_pat_` prefix, 93 characters) is exposed in
**five** places:

| # | Vector | Detail | Status |
| --- | --- | --- | --- |
| 1 | Chat transcript (twice) | Pasted into the assistant chat during the original backup session, then **re-pasted on 2026-11-12 to authorize this continuation** — same token, byte-for-byte | Cannot be unsent; revocation is the only remedy |
| 2 | `scratch/pdf-extract/MASTER_PROMPT.txt` (tracked at HEAD of `main`) | Clear-text, wrapped across two lines (76 + 17 chars), line ~688 | **Confirmed in tree** |
| 3 | `upload/MASTER_PROMPT part1.pdf` (tracked at HEAD of `main`) | §12.1; the token is **font-encoded inside a compressed content stream** — invisible to raw-byte scans and to `qpdf --qdf` + grep, recovered by any standard `pdftotext` extraction | **Confirmed in tree** |
| 4 | Git history blobs | Vectors 2–3 in historical commits as well | History rewrite required |
| 5 | Every existing clone / archive | The four release-zip homes and all working copies carry the tracked files | Invalidated by Phase 6 |

Two aggravating facts verified this round:

- The token is **the same one** embedded in the tree (byte-for-byte comparison
  against the reconstructed wrapped value — 93/93 characters).
- The root `.gitignore` already tried to prevent exactly this class of leak
  (`upload/Command.txt`, `upload/Command.docx`, `upload/MASTER_PROMPT.docx` are
  listed as "Token-bearing prompt files … never commit") but missed the **part1
  PDF** and its **text extraction**. Round 1 closes that gap (see §6).

## 2. Why this is urgent

Until the token is revoked, anyone holding the part1 PDF, the chat transcript,
or any clone of the repository can read — and depending on its permission set,
write — the private repository, including its full history and the release
archives. Every additional round of work pushed on top of the contaminated
history increases the rewrite cost. **The purge should happen before the next
release ship, not after.**

## 3. Execution order (strict)

```
Phase 0  Revoke the exposed token            (owner, GitHub web UI)     ~2 min
Phase 1  Prepare surgery environment          (mirror clone + bundle)  ~5 min
Phase 2  Discovery — enumerate contamination  (script, read-only)      ~2 min
Phase 3  Purge — rewrite history             (git filter-repo)        ~2 min
Phase 4  Verify — zero matches remain        (script, read-only)      ~2 min
Phase 5  Force-push all refs                 (explicit confirmation)  ~1 min
Phase 6  Post-purge cleanup + docs           (re-clone, gitignore, docs)
```

`tools/purge_secret_history.sh` (in this toolkit tree, but run against the
**monorepo-root mirror**) automates Phases 2–5 with confirmations and never
prints or logs the token value. It was re-verified this round against synthetic
fixtures reproducing both real contamination shapes (wrapped clear-text and a
font-encoded PyMuPDF PDF with a space in its filename): 13/13 assertions.

---

## Phase 0 — Revoke (do this first, before anything else)

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens**.
2. Identify the token by its target repository (`MrMMA2004/pharma-ml-toolkit`)
   and creation around the backup session (2026-11-04/05).
3. **Delete** it. Regenerate is not enough — delete, then create a fresh one
   for future work (Contents RW, Workflows RW, Metadata R; short expiry;
   stored in a credential manager, never in a file inside any clone, never in
   chat).
4. Verify revocation from any machine:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" \
     -H "Authorization: Bearer <the-revoked-token>" \
     https://api.github.com/repos/MrMMA2004/pharma-ml-toolkit
   # must print 401 once the token is dead
   ```

> The history rewrite below is still required after revocation: the exposed
> bytes must leave the repository, and GitHub retains unreachable objects and
> cached views (raw URLs, PR diffs) until asked to GC them.

## Phase 1 — Prepare the surgery environment

On a machine with push rights, in a fresh directory:

```bash
# 1a. git-filter-repo (pure Python)
pip install --user git-filter-repo

# 1b. fresh MIRROR clone of the MONOREPO ROOT with the NEW token
#     (authenticate via gh CLI or a credential helper — never a token in the URL)
gh auth login
git clone --mirror https://github.com/MrMMA2004/pharma-ml-toolkit.git
cd pharma-ml-toolkit.git

# 1c. byte-exact safety bundle of the pre-surgery state
git bundle create ../pharma-ml-toolkit-pre-purge.bundle --all
```

## Phase 2 — Discovery (read-only, safe any time)

```bash
bash pharma-ml-toolkit/tools/purge_secret_history.sh --discover .
```

Enumerates, across **every commit**:

1. **Text blobs** matching the PAT patterns (`git grep -I`).
2. **PDF blobs** via `pdftotext -layout` text extraction — scanned raw **and**
   newline-stripped, because the real token wraps across two extracted lines.
   This is the only method that recovers the font-encoded token from the
   tracked part1 PDF (byte-level and `qpdf --qdf` scans provably miss it).
   PDFs that fail extraction are surfaced as `UNPARSEABLE_PDF`, never silently
   skipped.
3. The **known contaminated paths** (see the script's `KNOWN_BAD_PATHS`).

Expected hits on the current repository: `scratch/pdf-extract/MASTER_PROMPT.txt`
(text) and `upload/MASTER_PROMPT part1.pdf` (PDF). Any additional hit extends
the purge automatically.

## Phase 3 — Purge

```bash
bash pharma-ml-toolkit/tools/purge_secret_history.sh --purge .
# token via hidden prompt, or: export PURGE_TOKEN=... beforehand
```

What it does:

1. Builds a replacements file (mode `0600`, shredded on exit):
   `literal:<token>`, plus `github_pat_…{60,}` and `ghp_…{30,}` regexes →
   `***REDACTED-PAT***`.
2. Removes the contaminated **paths** from all history
   (`--invert-paths --path …`): the known bad paths plus everything Phase 2
   discovered. The PDF is removed as a whole path — a font-encoded stream
   cannot be byte-scrubbed in place.
3. Everything else in `scratch/`, `upload/`, `download/` and
   `pharma-ml-toolkit/dist-archive/` is **left in history** — those trees carry
   project infrastructure (daemons, E2E evidence, release-zip homes) per
   `docs/RESUME.md`; the rewrite stays minimal.

**Expected effect:** all commit hashes change; subjects, authors, dates and
topology are preserved. Verify by subject after the purge:

```bash
git log --oneline -5 main
# expect the round-267 chain by name (e55c1dc, 4cfd953, 5396954 …) with new hashes
```

## Phase 4 — Verify

```bash
bash pharma-ml-toolkit/tools/purge_secret_history.sh --verify .
```

Pass criteria: zero text/PDF pattern hits across all commits; zero
unparseable PDFs; known bad paths absent everywhere; ref topology preserved.
The script exits nonzero on any failure and refuses to bless the repo.

## Phase 5 — Force-push

```bash
bash pharma-ml-toolkit/tools/purge_secret_history.sh --push . \
  --origin https://github.com/MrMMA2004/pharma-ml-toolkit.git
```

- Requires typing `FORCE-PUSH`.
- **Before pushing:** the arena system grows new refs between rounds — if any
  branch was pushed after your mirror was taken, re-clone and re-run
  `--discover`, or that branch's contaminated history survives the rewrite.
- If branch protection rejects the force-push, temporarily allow force pushes
  (Settings → Branches), push, re-enable.
- After the push: ask GitHub Support to run early GC on the orphaned objects
  and cached views (the standard post-rewrite step for secret removal).
- Keep `../pharma-ml-toolkit-pre-purge.bundle` (it is secret-bearing material)
  until Phase 6 is fully verified, then destroy it.

## Phase 6 — Post-purge cleanup

```bash
cd .. && rm -rf pharma-ml-toolkit.git
git clone https://github.com/MrMMA2004/pharma-ml-toolkit.git && cd pharma-ml-toolkit
```

1. Confirm the purged paths are gone; confirm the round-1 `.gitignore`
   additions (below) are present so the files cannot be re-committed by
   accident.
2. **Documentation corrections**:
   - `GITHUB_BACKUP_MANIFEST.md` / `GITHUB_LINKS_REGISTRY.md`: correct any
     "never enters the repository" claim with a short incident note (date,
     vectors, revocation date, rewrite date, pointer to this document).
   - Add a `worklog.md` line recording the remediation.
3. **Invalidate every old clone and archive home** (they still carry the
   token): the three/four release-zip homes, all working copies, and any
   bundle. Rebuild the release zip from the clean tree on the next ship.
4. Resume the `part5-continuation` rounds (rebase the branch onto the
   rewritten `main` — its commits only add files, so the rebase is trivial).

## Appendix A — Round-1 `.gitignore` additions (already applied)

The root `.gitignore`'s existing token-bearing block now also covers the two
vectors that actually leaked, plus generic secret carriers:

```gitignore
# Token-bearing prompt files (contain GitHub PATs) — never commit
upload/Command.txt
upload/Command.docx
upload/MASTER_PROMPT.docx
upload/MASTER_PROMPT part*.pdf      # ← the part1 PDF vector (this incident)
scratch/pdf-extract/MASTER_PROMPT*.txt  # ← its text extraction (this incident)
*.token
*.key
*.pid
```

Deliberately **not** ignored (project conventions, see `docs/RESUME.md`):
`scratch/` as a whole (daemons, E2E evidence, suite runners live there),
`download/` and `pharma-ml-toolkit/dist-archive/` (the release-zip safe-copy
homes), `db/custom.db` (template).

## Appendix B — Why byte-level PDF scrubbing is insufficient

`git filter-repo --replace-text` operates on raw blob bytes. It scrubs
plain-text blobs reliably (vector 2). It cannot scrub a PDF whose text is
font-encoded inside a zlib-compressed content stream (vector 3): the byte
pattern simply does not exist at rest. `qpdf --qdf` decompression does not
help either — the characters are font glyph codes, not ASCII. Only a true text
extraction (`pdftotext`) reconstructs the token — which is exactly how this
document confirms the contamination and how the script's discovery scans PDFs.

## Appendix C — Verification log template

```
date/time (local):                  ______
operator:                           ______
token revoked at (UTC):             ______
pre-purge ref count:                ______
discovery hits (paths):             ______
bundle created:                     ☐
purge executed:                     ☐
verify: text/pdf matches 0          ☐
verify: unparseable pdfs 0          ☐
verify: topology preserved          ☐
arena re-check (new refs?):         ☐
force-pushed all refs               ☐
GitHub Support GC request           ☐  ticket: ______
docs corrected                      ☐
old clones + zip homes invalidated  ☐
release zip rebuilt from clean tree ☐
```
