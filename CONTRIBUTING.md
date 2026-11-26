# Contributing to PharmaML Toolkit

Thank you for helping improve PharmaML. Contributions should make pharmaceutical and machine-learning workflows more reliable, reproducible and accessible without compromising the project's offline-first design.

## Before you start

- Search existing issues and pull requests before opening a duplicate.
- Open an issue for substantial API, dependency or interface changes so the approach can be discussed first.
- Keep pull requests focused. Separate bug fixes, features, documentation-only edits and release chores where practical.
- Do not submit credentials, private datasets, patient information or proprietary research data.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security concerns privately using [SECURITY.md](SECURITY.md).

## Repository layout

- `pharma-ml-toolkit/src/pharmaml/`: Python package and command-line interface.
- `pharma-ml-toolkit/frontend/` and `pharma-ml-toolkit/src-tauri/`: supported React/TypeScript + Tauri 2 desktop application.
- `pharma-ml-toolkit/legacy/`: retired PySide6 launcher and archival source; not a supported application entry point.
- `pharma-ml-toolkit/tests/`: Python and desktop-backend tests.
- `pharma-ml-toolkit/examples/`: executed notebooks, output-free variants and runnable examples.
- `src/` and root `tests/`: separate Next.js project console and web integration checks.
- `pharma-ml-toolkit/docs/`: package, workflow and methodology documentation.

## Python development

Use Python 3.10–3.12 and work from the toolkit directory:

```bash
cd pharma-ml-toolkit
python -m pip install -e ".[dev,reports]"
```

Install only the optional extras needed by your change, such as `chem`, `docking`, `boosting`, `explain` or `deep`. For the fully pinned QA environment, follow `requirements.txt`; do not mix its pinned installation with a second loose-extra installation in the same environment.

Run focused checks while developing:

```bash
python -m pytest tests/test_data_loading.py tests/test_pk.py -o addopts="-q --tb=short"
ruff check src/pharmaml
mypy src/pharmaml
```

Run the broader suite when your change affects shared behavior. Tests involving optional dependencies must degrade or skip explicitly when those dependencies are unavailable; do not hide unexpected failures with broad exception handling.

## Studio development

The supported desktop application is the React + TypeScript frontend hosted by Tauri 2, with the Python backend as a local sidecar. The retired PySide6 launcher is archived and is not a supported contribution target. See [`pharma-ml-toolkit/frontend/README.md`](pharma-ml-toolkit/frontend/README.md) for architecture details.

From the repository root, install the locked frontend dependencies and run the frontend checks:

```bash
cd pharma-ml-toolkit/frontend
npm ci
npm test
npm run typecheck
npm run build
```

A browser session with live workflows needs a local backend plus the dev-only Vite proxy variables; see the [frontend development notes](pharma-ml-toolkit/frontend/README.md). The native Tauri host also requires a compatible frozen sidecar staged under `pharma-ml-toolkit/src-tauri/resources/backend/`; a fresh checkout does not include that generated Windows resource. The [Windows release workflow](.github/workflows/windows-release.yml) performs this staging before packaging. Desktop changes should include focused frontend/backend contract tests and keep scientific computation in the Python backend.

## Web console development

From the repository root:

```bash
bun install
bunx prisma generate
bun run lint
bun run build
```

Use `bun run dev` for local development. Browser-facing code must use relative application URLs rather than assuming a local backend address.

## Documentation and examples

- Update public API documentation, notebook twins or the changelog when behavior changes.
- Keep examples deterministic and based on shipped or explicitly licensed data.
- Avoid committing generated output larger than necessary. Use GitHub Releases for downloadable release archives.
- Preserve reproducibility: document random seeds, units, input assumptions and environment-specific limitations.
- Never claim clinical validation or present research software as clinical decision support.
- Run `python scripts/check_github_page.py` and `python -m unittest discover -s tests -p 'test_github_page.py'` for landing-page changes. Label screenshots with their actual version or capture round and environment; distinguish browser previews from packaged-install or clean-machine evidence. If `CITATION.cff` names a version, keep it aligned with the checkout. Update links and evidence rather than relabeling old screenshots as current.
- Regenerate the decorative social card with `python scripts/build_social_preview.py` (requires Pillow). GitHub does not pick it up from a commit: upload `pharma-ml-toolkit/docs/assets/github-social-preview.png` under **Settings → General → Social preview** if you have repository admin access.

## Pull request checklist

Before requesting review:

1. Explain the user-visible problem and the chosen fix.
2. Add regression coverage for bug fixes and behavior coverage for new features.
3. Run relevant tests, Ruff and mypy; run web lint/build for frontend changes.
4. Update documentation and versioned artifacts only when the change requires them.
5. Note any unrun tests, optional dependencies or platform limitations honestly.
6. Confirm that no credentials, private data or unrelated generated files are included.

By contributing, you agree that your contribution may be distributed under the repository's [MIT License](LICENSE).
