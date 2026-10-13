# Worklog — PharmaML Toolkit Upgrade & Codebase Explorer

---
Task ID: 1
Agent: Main orchestrator (Z.ai Code)
Task: Extract and analyze the uploaded RAR archive `کدهای ماشین لرنینگ.rar` (machine learning codes) and assess the current code quality.

Work Log:
- Extracted `/home/z/my-project/upload/کدهای ماشین لرنینگ.rar` (648 KB, 138 entries) with `unrar x` into `/home/z/my-project/extracted/` — 118 files recovered (1 entry failed with a seek error, appears to be a stray binary entry).
- Analyzed representative scripts: `Classification/RF.py`, `Convert IC50(nM) to pIC50.py`, `others/Docking/main.py`, `others/Docking/readerwithdetails.py`, `others/Docking/gpfer.py`, `others/Docking/smilestopdbqt.py`, `others/Docking/checker.py`, `Split CSV file.py`, `Tanimoto calculator.py`, `others/Ml/Descriptor calculator.py`.
- Environment recon: Python 3.12.15 (numpy 2.1.3, pandas 2.2.3, sklearn 1.5.2, scipy 1.14.1, pytest) — RDKit NOT installed. `unrar`, `zip`, `unzip` CLI available. Next.js dev server already running on port 3000 (HTTP 200).

Stage Summary:
- Original codebase is a computational-pharmacy / QSAR research toolkit (~30 near-duplicate model scripts for classification & regression: RF, SVM, XGBoost, LightGBM, CatBoost, AdaBoost, ANN, KNN, Stacking, Voting, Logistic, Ridge/Lasso/ElasticNet, Decision tree), plus:
  - Data utilities: IC50→pIC50 conversion, CSV↔SDF↔SMI converters, CSV splitter, non-numeric remover, Tanimoto similarity scan, RDKit descriptor calculator, ECFP/Morgan/Mordred calculators.
  - Docking: AutoDock4 batch orchestration (autogrid4/autodock4 via subprocess + ProcessPoolExecutor), GPF generator with atom-type extraction, DLG cluster-table parser, docking completeness checker, SMILES→PDBQT via Meeko with AD4 atom-type repair.
- Quality issues found: no package structure (flat scripts), massive duplication, hardcoded Windows file paths (`C:\Users\...`), no type hints, no docstrings, bare `except:`, print-based logging, `os.chdir` side effects, `shell=True` subprocess calls, no tests, no CLI, mixed Persian/English messages, no pharmaceutical PK/PD/dosage calculation modules (user explicitly requests these as a new domain).
- Upgrade plan: build `pharma-ml-toolkit/` — a typed, documented, tested Python 3.10+ package `pharmaml` with subpackages `chem`, `ml`, `pharma` (NEW: PK/PD/dosage), `docking`, `utils`, plus examples, pytest suite, pyproject, Dockerfile, CI; then a Next.js "Codebase Explorer" web app on `/` to browse/download the upgraded codebase.
