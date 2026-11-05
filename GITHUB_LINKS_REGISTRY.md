# GitHub Links Registry — individual coverage audit

**Audit date:** 2026-11-05  
**Input:** `GitHub_Links_Registry.pdf` (11 pages, 80 link annotations, 34 unique repository URLs)  
**Branch:** `arena/c44913ab-pharma-ml-toolkit`  
**Outcome:** entries 31–33, the only repositories the PDF explicitly marks **Deferred**, are now implemented as compact offline fold-ins in `pharmaml.ml.xai` and exposed in `playground.py`. Entries 1–30 retain their existing concrete coverage; entry 34 is the private project backup.

## Method and evidence limits

Each unique URL was checked individually with authenticated GitHub repository metadata and the recursive Git tree API for its advertised default branch. Every request succeeded and every tree returned `truncated=false`. README content was retrieved for all 33 external repositories. The live metadata/tree/README responses are stored outside Git under `/tmp/github-link-registry-audit/`; the deferred trio was additionally shallow-cloned for source/test inspection.

The counts below are recursive tree-entry counts, **not** claims that every current upstream blob was executed or copied. For entries 1–30, this audit maps the repository to local source/tests/cards and preserves the earlier absorption record; it does not relabel a catalog URL or a successful GUI launch as functional proof. For entries 31–33, implementation decisions came from local source and test files at these inspected commits:

- `johnmarktaylor91/torchlens` — `23dc5ee2cb4c4278a42015b2df2d9fc5cd47bb62`
- `SeldonIO/alibi` — `99c3421d7c971c85893625e37da1133e7ddf1779`
- `understandable-machine-intelligence-lab/Quantus` — `85bc29d137d8f42835dd392f5cc441ad61d6d1f1`

## One-by-one register

| # | Repository | Default branch | Tree entries | Local coverage and disposition |
|---:|---|:---:|---:|---|
| 1 | [jakevdp/pythondatasciencehandbook](https://github.com/jakevdp/pythondatasciencehandbook) | `master` | 264 | Existing `ml/unsupervised.py`, statistics/foundations surfaces and `tests/test_unsupervised.py`; no registry-driven gap found. |
| 2 | [hoffstadt/DearPyGui](https://github.com/hoffstadt/DearPyGui) | `master` | 261 | Existing playground table chrome, collapsible cards, plot interaction, font scaling and developer-tool affordances, with GUI tests; no new dependency added. |
| 3 | [shap/shap](https://github.com/shap/shap) | `main` | 616 | Existing `ml/explain.py` TreeExplainer/interactions/figure surfaces and `tests/test_explain.py`; no registry-driven gap found. |
| 4 | [optuna/optuna](https://github.com/optuna/optuna) | `master` | 484 | Existing `ml/hpo.py` + `ml/tuner.py`, pruning/Hyperband and `tests/test_hpo.py` / `tests/test_tuner.py`; no new change. |
| 5 | [the-matter-lab/selfies](https://github.com/the-matter-lab/selfies) | `master` | 98 | Existing `chem/selfies_codec.py`, round-trip/randomized tests and GUI codec surface; no new change. |
| 6 | [molecule-generator-collection/ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2) | `master` | 157 | Existing `generative/mcts.py`, RNN/RL generation and ChemTS-style playground controls/tests; no new change. |
| 7 | [MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4) | `main` | 3,290 | Existing scoring transforms, diversity filters, standardization and optimizer workflow plus playground cards/tests; no new change. |
| 8 | [gcorso/DiffDock](https://github.com/gcorso/DiffDock) | `main` | 108 | Existing `docking/pose_rmsd.py`, pose/pocket analysis and DiffDock playground cards/tests; no new change. |
| 9 | [fchollet/deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks) | `master` | 59 | Existing deep-learning Foundations notebook/course surfaces; coverage reference, not a runtime dependency. |
| 10 | [SENATOROVAI/Data-Science-For-Beginners-from-scratch-course](https://github.com/SENATOROVAI/Data-Science-For-Beginners-from-scratch-course) | `main` | 31 | Existing Foundations statistics and Git-course surfaces; coverage reference, no missing runtime item identified. |
| 11 | [lukasmasuch/best-of-ml-python](https://github.com/lukasmasuch/best-of-ml-python) | `main` | 438 | Existing `ml/catalog.py`, capability modules and Best-of ML playground review; no new change. |
| 12 | [rdkit/rdkit](https://github.com/rdkit/rdkit) | `master` | 6,518 | Existing descriptors, fingerprints, molecular graphs/featurization and visualization integration with RDKit-dependent tests/skips; no new change. Native RDKit drawing remains environment-limited in this sandbox. |
| 13 | [cbouy/mols2grid](https://github.com/cbouy/mols2grid) | `master` | 77 | Existing `chem/molgrid.py` and Molecule grid playground surface; no new change. |
| 14 | [nglviewer/nglview](https://github.com/nglviewer/nglview) | `master` | 265 | Existing `chem/graphics3d.py` NGL selection/representation concepts and 3D viewer card; no new change. |
| 15 | [3dmol/3Dmol.js](https://github.com/3dmol/3Dmol.js) | `master` | 1,178 | Existing `chem/graphics3d.py` 3Dmol selection/representation concepts and viewer surface; no new change. |
| 16 | [Open-Systems-Pharmacology/PK-Sim](https://github.com/Open-Systems-Pharmacology/PK-Sim) | `develop` | 4,120 | Existing `pharma/pk.py`, events/units/sensitivity lineage and PK/PD playground/tests; no new change. |
| 17 | [Open-Systems-Pharmacology/OSPSuite-R](https://github.com/Open-Systems-Pharmacology/OSPSuite-R) | `main` | 741 | Existing OSP sensitivity/protocol conventions in `pharma/` with numerical tests; no new change. |
| 18 | [metrumresearchgroup/mrgsolve](https://github.com/metrumresearchgroup/mrgsolve) | `main` | 439 | Existing dose/event/infusion/steady-state conventions and solver controls in `pharma/`; no new change. |
| 19 | [nlmixr2/rxode2](https://github.com/nlmixr2/rxode2) | `main` | 2,101 | Existing EVID/event-window and ODE-tolerance lineage in `pharma/` with PK tests; no new change. |
| 20 | [meta-pytorch/captum](https://github.com/meta-pytorch/captum) | `master` | 393 | Existing `ml/xai.py` attribution/gradient/perturbation families, conductance/TCAV/adversarial validators and playground card; no additional captum integration needed for this registry request. |
| 21 | [keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis) | `master` | 95 | Existing single-row saliency/SmoothGrad explanation workflow in `ml/xai.py`; no new change. |
| 22 | [PAIR-code/saliency](https://github.com/PAIR-code/saliency) | `master` | 57 | Existing grayscale/diverging mask palettes and SmoothGrad lineage in `ml/xai.py`; no new change. |
| 23 | [jessevig/bertviz](https://github.com/jessevig/bertviz) | `master` | 48 | Existing seeded layer×head attention grids and query/key pair digest in `ml/xai.py` + playground; no new change. |
| 24 | [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) | `master` | 1,934 | Existing `ml/runboard.py`, reservoir/EMA/hparams/embedding surfaces and Run board card/tests; no new change. |
| 25 | [lmcinnes/umap](https://github.com/lmcinnes/umap) | `master` | 349 | Existing `ml/umap.py` and UMAP map card/tests; no new change. |
| 26 | [voxel51/fiftyone](https://github.com/voxel51/fiftyone) | `main` | 7,168 | Existing `ml/curation.py`, immutable view/brain/evaluation surfaces and Data curation card/tests; no new change. |
| 27 | [plotly/plotly.py](https://github.com/plotly/plotly.py) | `main` | 1,992 | Existing `viz/interactive.py`, express figures/templates/hover HTML and Interactive figure card/tests; no new change. |
| 28 | [interpretml/interpret](https://github.com/interpretml/interpret) | `main` | 499 | Existing `ml/interpret.py` EBM/partial-dependence surfaces, Glassbox EBM card and `tests/test_interpret.py`; no new change. |
| 29 | [divelab/DIG](https://github.com/divelab/DIG) | `dig-stable` | 458 | Existing `ml/geometric.py` + `ml/gnnxai.py`, Graph explain card and graph tests; no new change. |
| 30 | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `master` | 1,548 | Existing `ml/gnn.py` Data/Batch and GCN/SAGE/GIN/GAT surfaces, Graph model card and tests; no new change. |
| 31 | [johnmarktaylor91/torchlens](https://github.com/johnmarktaylor91/torchlens) | `main` | 4,938 | **Implemented:** `model_activation_trace` + **Model graph** action; exact graph/parameter/shape/memory tests. Compact numpy LayerStack scope, not arbitrary-framework hook parity. |
| 32 | [SeldonIO/alibi](https://github.com/SeldonIO/alibi) | `master` | 751 | **Implemented:** `tabular_anchor` + **Anchor rule** action; precision/coverage/failure tests. Numeric quantile-bin, deterministic empirical beam search; no KL-LUCB/PAC claim. |
| 33 | [understandable-machine-intelligence-lab/Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) | `main` | 236 | **Implemented:** complexity, faithfulness and max-sensitivity APIs + **Quantus metrics** action; numerical/determinism/undefined-score tests. This is the planned compact metric trio, not all upstream metric classes. |
| 34 | [MrMMA2004/pharma-ml-toolkit](https://github.com/MrMMA2004/pharma-ml-toolkit) | `main` | 1,705 at check time | Existing private backup and PR #1. Prior recovery evidence is in `GITHUB_BACKUP_MANIFEST.md`; this branch is pushed normally after verification, without history rewriting. |

## Deferred-source review and implementation mapping

### torchlens → operation-level activation graph

Reviewed the current README, capture/finalization modules, trace data surfaces, examples and test inventory. The upstream project now intercepts operations across multiple frameworks and carries a very large metadata/visualization/intervention surface. The registry requested only **activation traces**, so the compact local implementation records the semantics useful in this toolkit: `input → linear → activation → output` nodes, parent links, row-specific input/output widths, parameter counts, retained activation bytes and activation statistics.

The existing `layer_activation_trace` returned hidden post-activation arrays only. The new `model_activation_trace` adds explicit input/final-output nodes and operation metadata. It deliberately refuses estimators without the numpy LayerStack instead of claiming torch/JAX/TensorFlow interception.

### alibi → numeric tabular anchors

Reviewed `alibi/explainers/anchors/anchor_base.py`, `anchor_tabular.py`, `anchor_explanation.py`, anchor tests and method documentation. The retained objective is Alibi's: find a conjunction whose precision meets `threshold`, then maximize reference coverage. Numeric columns use the row's reference quantile bin, candidates grow as conjunctions through a bounded beam, and the final result includes human-readable conditions, covered row indices/count, empirical precision/coverage and candidate count.

The upstream stochastic sampling, KL-LUCB confidence bounds, categorical/text/image samplers and distributed runtime are outside this compact fold-in. Local precision/coverage are measured directly on the supplied reference rows; `success=False` and the best failed candidate are returned when the threshold is unreachable.

### Quantus → explanation-quality metric trio

Reviewed the metric inventory and, in particular, `functions/complexity_func.py`, `metrics/complexity/{complexity,sparseness}.py`, `metrics/faithfulness/faithfulness_estimate.py`, and `metrics/robustness/max_sensitivity.py` with their evaluation formulas.

- **Complexity:** magnitude-normalized entropy, normalized entropy, `exp(entropy)` effective support, Gini and nonzero share. All-zero explanations have zero entropy/support instead of a division error.
- **FaithfulnessEstimate:** descending signed-attribution groups are replaced independently from the original row; attribution sums and actual score drops are correlated with Pearson's r. Fewer than two groups or zero variance yields NaN plus a note.
- **MaxSensitivity:** seeded uniform input perturbations recompute the same explanation callback; each relative Frobenius change is retained and maximum/mean are reported. Zero original attribution norm is rejected as undefined.

These are three compact representatives requested by the registry's “metrics” gap, not a claim to reproduce Quantus' full axiomatic/localisation/randomisation/robustness catalog.

## Playground and verification evidence

The Deep Learning **Attribution & attention** card now includes:

1. **Model graph** — node/operation/parent/shape/parameter/mean/dead/byte table and node-mean strip.
2. **Anchor rule** — adjustable `anchor τ`, readable conditions, precision, coverage, covered-row count and explicit success/failure.
3. **Quantus metrics** — adjustable `noise n`, five complexity rows, faithfulness and two robustness rows for the currently selected attribution method.

The shared table restores the ordinary attribution schema when **Explain** is pressed again. `tests/test_xai_registry.py` supplies 11 display-independent numerical tests. `TestDeferredXaiHandlersHeadless` executes graph, metric, anchor and schema-restoration callbacks without Tk. The real Tk tests remain display-dependent and are skipped in this sandbox; native layout is not claimed as verified.

Focused evidence: **24 passed, 10 display-only skipped** across the new numerical tests, existing XAI helpers, the headless handlers and the native XAI GUI class. The counted full applicable suite passed with **3,853 passed, 637 skipped, 22 deselected**; the independent correlation doctest also passed. The 22 deselections are the exact native-drawing failures plus their mixed module-doctest container, not XAI exclusions. Ruff, `git diff --check`, mypy over all 117 package source files, and wheel/sdist builds passed.
