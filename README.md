<div align="center">

# PharmaML Toolkit

### Pharmaceutical machine learning, from compound table to reproducible report

**An offline-first Python toolkit, PySide6 Studio and web console for cheminformatics, QSAR, PK/PD, docking, explainable AI and scientific visualization.**

[![CI](https://github.com/MrMMA2004/pharma-ml-toolkit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MrMMA2004/pharma-ml-toolkit/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/MrMMA2004/pharma-ml-toolkit?display_name=tag&color=0f766e)](https://github.com/MrMMA2004/pharma-ml-toolkit/releases/latest)
[![Python 3.10+](https://img.shields.io/badge/python-%E2%89%A5%203.10-3776ab?logo=python&logoColor=white)](pharma-ml-toolkit/pyproject.toml)
[![MIT licensed](https://img.shields.io/badge/license-MIT-525252)](LICENSE)
[![8 executed notebooks](https://img.shields.io/badge/notebooks-8%20executed-6d28d9)](pharma-ml-toolkit/examples/)

[Quick start](#quick-start) ·
[Capabilities](#capabilities) ·
[Examples](#examples-and-documentation) ·
[Latest release](https://github.com/MrMMA2004/pharma-ml-toolkit/releases/latest) ·
[Cite this project](CITATION.cff) ·
[Full user guide](pharma-ml-toolkit/README.md)

</div>

---

PharmaML brings the repetitive parts of a computational-pharmacology workflow into one typed, testable package: load and profile a dataset, calculate molecular features, train and explain models, simulate concentration–time profiles, prepare docking campaigns, and export readable figures and reports. Core workflows run locally; there is no required cloud account, API key or hosted model service.

> **Research and education software.** PharmaML is not a clinical decision-support system. Validate models, calculations and generated outputs for your own data and intended use.

## Capabilities

| Area | What you can do |
| --- | --- |
| **Cheminformatics & QSAR** | Standardize SMILES, calculate RDKit descriptors and fingerprints, measure similarity, split datasets, train baselines and compare models. |
| **PK/PD & pharmacometrics** | Run non-compartmental analysis, compartment and PBPK simulations, population PK, sensitivity analysis and PK/PD diagnostics. |
| **Docking & molecular workflows** | Prepare ligands and receptors, calculate grid boxes, run AutoDock Vina-compatible workflows and analyze docking results. |
| **Explainable & generative ML** | Inspect feature effects and applicability domains, use deep or ensemble models, and evaluate generated molecular libraries. |
| **Reports & visualization** | Create accessible Matplotlib figures, HTML/SVG/PDF exports, model cards, dataset reports and reproducible run manifests. |
| **Desktop & web interfaces** | Explore the package in PharmaML Studio, including MD metric plots, or browse code, APIs, notebooks and downloads in the included web console. |

<p align="center">
  <a href="pharma-ml-toolkit/docs/assets/web-console-v1.2.161.png">
    <img src="pharma-ml-toolkit/docs/assets/web-console-v1.2.161.png" width="960" alt="The actual PharmaML v1.2.161 web console showing its verified source download, five thousand two hundred seven passing tests and MIT license">
  </a>
  <br>
  <sub>The actual <strong>v1.2.161 Next.js console</strong>: source downloads, executed notebooks, API reference and live package statistics. See also the <a href="pharma-ml-toolkit/examples/chart_readability.py">reproducible PK chart example</a>.</sub>
</p>

## Quick start

### Python package and PharmaML Studio

PharmaML supports **Python 3.10–3.12**. The project is currently distributed from GitHub and versioned ZIP archives, not as a PyPI release.

```bash
git clone https://github.com/MrMMA2004/pharma-ml-toolkit.git
cd pharma-ml-toolkit/pharma-ml-toolkit
# Install the verified core stack, then the package
python -m pip install -r requirements.txt
python -m pip install -e .

# Inspect the command-line interface
pharmaml --help

# Launch the PySide6 Studio after enabling the pinned
# "Studio GUI" block in requirements.txt and reinstalling it
python studio.py
```

Prefer a downloaded snapshot? Get the **[latest published source ZIP from GitHub Releases](https://github.com/MrMMA2004/pharma-ml-toolkit/releases/latest)**, extract it, enter its `pharma-ml-toolkit/` directory and run the same editable-install command. See the [installation guide](pharma-ml-toolkit/README.md#installation) for optional Studio GUI, cheminformatics, docking, boosting, explainability, deep-learning and development dependencies.

### Web console

The repository root contains the Next.js explorer for the toolkit, including source browsing, API reference, playground documentation and archive downloads.

```bash
# From the repository root
bun install
bun run dev
```

Open `http://localhost:3000`. Run `bun run build` for a production build. The Python package and the web console can be used independently.

## Examples and documentation

Start with an executed notebook or jump directly to the reference material:

| Resource | Purpose |
| --- | --- |
| [Walkthrough notebook](pharma-ml-toolkit/examples/pharmaml_walkthrough.ipynb) | Guided tour of the package and its main workflows. |
| [Complete ML workflow](pharma-ml-toolkit/examples/complete_ml_workflow.ipynb) | End-to-end data, feature, model and evaluation example. |
| [Deep learning workflow](pharma-ml-toolkit/examples/complete_deep_learning_workflow.ipynb) | Neural and molecular-model examples. |
| [PK workflow](pharma-ml-toolkit/examples/complete_pk_workflow.ipynb) | Pharmacokinetic calculations and simulations. |
| [Docking workflow](pharma-ml-toolkit/examples/complete_docking_workflow.ipynb) | Ligand preparation, docking and result analysis. |
| [Generative workflow](pharma-ml-toolkit/examples/complete_generative_workflow.ipynb) | Molecular generation and evaluation. |
| [SA-aware generation](pharma-ml-toolkit/docs/GENERATIVE_SA.md) | Add synthetic-accessibility reward and review raw scores and rejected candidates. |
| [MD visualization guide](pharma-ml-toolkit/docs/MD_VISUALIZATION.md) | Plot metric traces and aligned correlation heatmaps. |
| [Multi-objective workflow](pharma-ml-toolkit/docs/MULTIOBJECTIVE_WORKFLOW.md) | Plot study trade-offs and shortlist trials by crowding distance. |
| [Conformal regression intervals](pharma-ml-toolkit/docs/CONFORMAL_REGRESSION.md) | Calibrate and evaluate held-out prediction intervals. |
| [Conformal classification sets](pharma-ml-toolkit/docs/CONFORMAL_CLASSIFICATION.md) | Calibrate and evaluate held-out label sets. |
| [Data splitting guide](pharma-ml-toolkit/docs/DATA_SPLITTING.md) | Choose split strategies and check group and scaffold limitations. |
| [Classification evaluation](pharma-ml-toolkit/docs/CLASSIFICATION_EVALUATION.md) | Interpret ROC-AUC and unavailable scores. |
| [API & module index](pharma-ml-toolkit/docs/INDEX.md) | Package map and links into the detailed documentation. |
| [Full user guide](pharma-ml-toolkit/README.md) | Detailed domain coverage, CLI, deployment, testing and troubleshooting. |
| [Changelog](pharma-ml-toolkit/CHANGELOG.md) | Release history and migration notes. |

Every notebook also has a `*_no_output.ipynb` twin for a clean starting point. The committed executed versions make the examples readable directly on GitHub.

## Repository map

```text
pharma-ml-toolkit/
├── pharma-ml-toolkit/       Python package, PharmaML Studio and full manual
│   ├── src/pharmaml/        Typed library source
│   ├── examples/            Executed notebooks and runnable scripts
│   ├── tests/               Python unit and integration tests
│   ├── docs/                Module, reporting and methodology guides
│   └── download/            Toolkit-local versioned archives
├── src/                     Next.js web console
├── public/                  Web assets
├── tests/                   Web/toolkit integration and parity tests
├── download/                Current and historical downloadable releases
└── upload/                  Distribution copies and project documents
```

## Verification and reproducibility

The [v1.2.161 release record](pharma-ml-toolkit/CHANGELOG.md) describes the checked-in archive (which may be newer than the latest published GitHub release):

- **5,207 collected pytest and doctest cases**—4,354 tests plus 853 doctests—with all nine verification chunks green;
- **eight executed notebooks refreshed at v1.2.161**, together with their clean no-output twins;
- a **1,075-file, 329,046-line** release census; and
- archive integrity verification for the 23,420,835-byte release ZIP.

The test inventory covers domain calculations, data leakage, notebooks, CLI, visualization, web parity and Studio pages. See the [testing guide](pharma-ml-toolkit/README.md#testing), [GUI audit](pharma-ml-toolkit/docs/gui/GUI_AUDIT.md) and [changelog](pharma-ml-toolkit/CHANGELOG.md) for the complete release evidence and reproduction commands.

## Contributing and license

Bug reports and focused pull requests are welcome. See the [contribution guide](CONTRIBUTING.md), [code of conduct](CODE_OF_CONDUCT.md) and [security policy](SECURITY.md). Keep changes offline-friendly, deterministic where practical, covered by tests, and documented when they alter public APIs or generated artifacts.

PharmaML is available under the [MIT License](LICENSE).
