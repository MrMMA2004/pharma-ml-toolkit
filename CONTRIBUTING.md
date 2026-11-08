# Contributing to PharmaML Toolkit

Thank you for helping improve PharmaML. Contributions should make pharmaceutical and machine-learning workflows more reliable, reproducible and accessible without compromising the project's offline-first design.

## Before you start

- Search existing issues and pull requests before opening a duplicate.
- Open an issue for substantial API, dependency or interface changes so the approach can be discussed first.
- Keep pull requests focused. Separate bug fixes, features, documentation-only edits and release chores where practical.
- Do not submit credentials, private datasets, patient information or proprietary research data.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security concerns privately using [SECURITY.md](SECURITY.md).

## Repository layout

- `pharma-ml-toolkit/src/pharmaml/`: typed Python package.
- `pharma-ml-toolkit/ui/` and `pharma-ml-toolkit/studio.py`: PySide6 Studio.
- `pharma-ml-toolkit/tests/`: Python tests, including Studio tests.
- `pharma-ml-toolkit/examples/`: executed notebooks, clean twins and runnable examples.
- `src/` and `tests/` at repository root: Next.js console and web integration checks.
- `pharma-ml-toolkit/docs/`: detailed package and methodology documentation.

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

Enable the pinned **Studio GUI** block in `requirements.txt`, reinstall the requirements and launch:

```bash
python studio.py
```

For displayless Linux CI:

```bash
QT_QPA_PLATFORM=offscreen python studio.py
```

Do not force the offscreen backend on Windows, macOS or Wayland desktops. Interface changes should include focused behavior tests and keep expensive work off the GUI thread.

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

## Pull request checklist

Before requesting review:

1. Explain the user-visible problem and the chosen fix.
2. Add regression coverage for bug fixes and behavior coverage for new features.
3. Run relevant tests, Ruff and mypy; run web lint/build for frontend changes.
4. Update documentation and versioned artifacts only when the change requires them.
5. Note any unrun tests, optional dependencies or platform limitations honestly.
6. Confirm that no credentials, private data or unrelated generated files are included.

By contributing, you agree that your contribution may be distributed under the repository's [MIT License](LICENSE).
