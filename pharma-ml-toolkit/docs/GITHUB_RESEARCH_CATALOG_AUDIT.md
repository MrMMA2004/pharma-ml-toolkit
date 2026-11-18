# GitHub Research Repository Catalog — one-by-one disposition

Audit date: **2026-11-06**. Source: `GitHub_Research_Repository_Catalog.pdf` (15 pages, 170 link annotations, **140 unique repositories**).

## Method and evidence limits

- Every URL was checked live through the GitHub API. Metadata, the complete recursive default-branch tree and the README were retrieved for every repository. **140/140 checks passed; 140/140 trees were untruncated.** One repository (ECFP6-Highlighter-GUI) has no README.
- The PDF labels were reconstructed from link and text coordinates, not guessed from ordering: **108 conversation-derived** and **32 supplemental**. Supplemental links were evaluated for concrete value, not automatically commissioned.
- Existing source and `playground.py` were compared against each repository. New work inspected the relevant upstream implementation directly: DeepDTA `source/datahelper.py` and `emetrics.py`, DeepPurpose encoders, DOCKSTRING `target.py`, `benchmarks/original.py` and all 58 target configs, and the pregnancy README/PKML passive-transport formulas.
- File counts are live default-branch blob counts, not lines read and not a scientific-quality score. No claim is made that every blob in every large framework was executed. Repository presence, popularity or an absorbed formula is not scientific validation.
- **Status meanings:** `added` = new computation in this round; `GUI linked` = existing computation whose missing playground surface is now linked; `covered` = existing native coverage; `partial` = useful overlap with a stated framework boundary; `boundary` = external engine/application/dataset remains external; `guide` = discovery-only list.

## This round’s concrete changes

- `ml/dta.py`: exact DeepDTA sequence encodings, AAC/dipeptide features, aligned ECFP pairs and rm2; Deep Learning paired-data + ridge card.
- `pkpd/pregnancy.py`: separate maternal/fetal unbound-fraction transfer, 16-project index, PK-Sim age conversion and maternal–fetal GUI. It is explicitly a two-compartment lightweight model.
- `docking/dockstring.py`: 58 target boxes and three original objectives; DOCKSTRING lab and Campaign bridge.
- The previously library-only PK/PD engines gain regimen, extended-NCA, OSP-library and DDI playground surfaces.

## Disposition table

| # | Repository (canonical redirect noted) | PDF label | Live blobs | Status | Disposition / local evidence |
|---:|---|---|---:|---|---|
| 1 | `hsiaoyi0504/awesome-cheminformatics` | conversation | 3 | **guide** | Discovery list only; concrete repositories receive their own rows. |
| 2 | `dmlc/xgboost` | conversation | 1,462 | **covered** | Optional XGBoost registry estimators, training and tuning. |
| 3 | `pandas-dev/pandas` | conversation | 2,691 | **covered** | Pandas table loading, cleaning, merging and profiling throughout data.py/preprocessing. |
| 4 | `scikit-learn/scikit-learn` | conversation | 1,867 | **covered** | Sklearn model registry, pipelines, CV, metrics and model comparison. |
| 5 | `volkamerlab/teachopencadd` | supplemental | 538 | **covered** | CADD topics map to chemistry, docking, interaction and PK workflows; notebooks are not transplanted. |
| 6 | `lmmentel/awesome-python-chemistry` | supplemental | 3 | **guide** | Discovery list only; no feature commission by itself. |
| 7 | `hussius/deeplearning-biology` | supplemental | 7 | **guide** | Deep-biology discovery list; concrete DTA/GNN repositories are mapped separately. |
| 8 | `yangnianzu0515/awesome-molecular-docking` → `Thinklab-SJTU/awesome-molecular-docking` | conversation | 4 | **guide** | Docking discovery list; concrete engines and analysis repositories are mapped separately. |
| 9 | `benb111/awesome-small-molecule-ml` | conversation | 3 | **guide** | Small-molecule-ML discovery list; concrete QSAR/ADMET repositories are mapped separately. |
| 10 | `DistrictDataLabs/yellowbrick` | conversation | 1,064 | **covered** | ml/diagnostics.py and viz diagnostic plots: Rank1D/2D, thresholds, influence and elbow. |
| 11 | `bokeh/bokeh` | conversation | 6,770 | **covered** | viz interactive figures, templates, hover HTML and dashboard-style surfaces. |
| 12 | `plotly/dash` | conversation | 1,328 | **covered** | viz/dashlite.py plus native callback-style playground cards; no Dash server dependency. |
| 13 | `plotly/plotly.py` | conversation | 1,992 | **covered** | viz/interactive.py express scatter/line/bar/histogram, templates and hover export. |
| 14 | `mwaskom/seaborn` | conversation | 325 | **covered** | viz palettes, statistical plots and seaborn-compatible styling. |
| 15 | `matplotlib/matplotlib` | conversation | 4,606 | **covered** | Shared Matplotlib backend guards, figure hosts and export/report machinery. |
| 16 | `ydataai/ydata-profiling` → `Data-Centric-AI-Community/fg-data-profiling` | conversation | 462 | **covered** | data_profiling.py and profiling.py: column quality, correlations and report surfaces. |
| 17 | `raphaelvallat/pingouin` | conversation | 118 | **covered** | ml/inferential.py: repeated-measures/Welch ANOVA, post-hoc, Bayes factor and circular statistics. |
| 18 | `statsmodels/statsmodels` | conversation | 2,301 | **covered** | ml/formula.py and stats.py: OLS/GLM, ANOVA and inferential diagnostics. |
| 19 | `scipy/scipy` | conversation | 2,929 | **covered** | SciPy-backed optimization/integration plus native compartment and interpolation engines. |
| 20 | `shap/shap` | conversation | 616 | **covered** | SHAP summary/interactions and waterfall/force/dependence/heatmap/group-difference GUI actions. |
| 21 | `optuna/optuna` | conversation | 484 | **covered** | ml/hpo.py and tuner.py: TPE, median pruner, importances and study figures. |
| 22 | `catboost/catboost` | conversation | 42,885 | **covered** | Optional CatBoost registry estimators and tuning. |
| 23 | `datamol-io/molfeat` | supplemental | 145 | **covered** | chem/features.py unified descriptors/fingerprints; no MolFeat transformer downloads. |
| 24 | `datamol-io/datamol` | conversation | 174 | **covered** | Datamol standardization profile in chem/csv_standardize.py. |
| 25 | `bp-kelley/descriptastorus` | conversation | 671 | **covered** | Descriptor/fingerprint feature factory and batch CSV metadata. |
| 26 | `MLCIL/scikit-fingerprints` | conversation | 427 | **covered** | Multiple fingerprint families in chem/features.py and similarity workflows. |
| 27 | `mordred-descriptor/mordred` | conversation | 200 | **covered** | Optional Mordred descriptor integration in the feature factory. |
| 28 | `rdkit/rdkit` | conversation | 6,518 | **covered** | RDKit descriptors, MolStandardize, ECFP, canonicalization, 3D and standardization lab. |
| 29 | `ModelOriented/DALEX` | conversation | 319 | **covered** | ml/dalex.py: model parts, profiles, diagnostics and fairness-style group analysis. |
| 30 | `interpretml/interpret` | conversation | 499 | **covered** | ml/interpret.py and EBM glassbox card. |
| 31 | `marcotcr/lime` | conversation | 73 | **covered** | ml/lime.py plus local surrogate explanations. |
| 32 | `lmcinnes/umap` | conversation | 349 | **covered** | ml/umap.py and UMAP map card with trustworthiness and neighbours. |
| 33 | `chemprop/chemprop` | conversation | 354 | **covered** | ml/benchmark.py, MPNN, split strategies, calibration and uncertainty audits. |
| 34 | `deepchem/deepchem` | conversation | 1,526 | **covered** | ml/dataset.py, benchmark.py and benchmetrics.py: splits, MoleculeNet registry and metrics. |
| 35 | `cimm-kzn/CGRtools` | supplemental | 154 | **boundary** | Small-molecule graph overlap is covered by chem/molgraph.py; reaction/CGR engines are explicitly not ported. |
| 36 | `UnixJunkie/dimorphite_dl` | conversation | 11 | **covered** | Dimorphite 41-rule protonation enumeration in csv_standardize.py. |
| 37 | `mcs07/MolVS` | conversation | 51 | **covered** | MolVS standardize chain in csv_standardize.py. |
| 38 | `openbabel/openbabel` | conversation | 37,613 | **covered** | OpenBabel stays external; InChI round-trip is the disclosed cross-toolkit check. |
| 39 | `chembl/ChEMBL_Structure_Pipeline` | conversation | 19 | **covered** | ChEMBL pipeline rules, salts/solvents and fragment parent in csv_standardize.py. |
| 40 | `volkamerlab/opencadd` | supplemental | 181 | **covered** | OpenCADD teaching topics map to docking preparation, poses, interactions and PK; no notebook shell. |
| 41 | `volkamerlab/maxsmi` | supplemental | 118 | **covered** | chem/smiles_enumeration.py: atom-order augmentation and deterministic seeded sampling. |
| 42 | `chembl/FPSim2` | supplemental | 57 | **partial** | chem/similarity.py and ANN search cover similarity; FPSim2 database/IO engine is not vendored. |
| 43 | `budimir-ilic/qsar-machine-learning` | supplemental | 139 | **covered** | QSAR preprocessing, registry training, evaluation and external validation workflows. |
| 44 | `hachmannlab/chemml` | supplemental | 637 | **covered** | Feature generation, preprocessing and model workflows cover ChemML concepts; no deprecated API shim. |
| 45 | `awslabs/dgl-lifesci` | supplemental | 408 | **added** | MPNN already existed; ml/dta.py now adds paired protein/ligand data and a Deep Learning card. |
| 46 | `hkmztrk/DeepDTA` | supplemental | 42 | **added** | Exact 25-symbol label/one-hot encoding and rm2 in ml/dta.py, paired ridge GUI baseline. |
| 47 | `kexinhuang12345/DeepPurpose` | supplemental | 296 | **added** | Paired DTA feature/matrix surface; CNN/RNN/transformer checkpoints remain an explicit boundary. |
| 48 | `precisionmatics/ECFP6-Highlighter-GUI` | conversation | 3 | **covered** | chem/bit_highlights.py and ECFP bit-highlight GUI; upstream has no README file. |
| 49 | `swansonk14/admet_ai` | conversation | 205 | **covered** | ml/admet.py: 52 endpoints, model-quality table, radial score and low-confidence disclosure. |
| 50 | `idekerlab/drugcell` | conversation | 30 | **covered** | ml/visiblenet.py visible-network/DrugCell-style ontology model. |
| 51 | `DeepGraphLearning/torchdrug` | conversation | 323 | **added** | Existing GNN/MPNN coverage plus the new paired DTA data surface; protein LMs remain external. |
| 52 | `chembl/chembl_webresource_client` | supplemental | 31 | **boundary** | Live ChEMBL web client is intentionally not called; offline examples/registries preserve reproducibility. |
| 53 | `mims-harvard/TDC` | conversation | 605 | **covered** | TDC split, oracle, ADMET and evaluation semantics in benchmark/oracles/benchmetrics. |
| 54 | `gcorso/DiffDock` | conversation | 108 | **covered** | docking/pose_rmsd.py: symmetry-corrected RMSD, perturbation, pocket crops and pose reports. |
| 55 | `ccsb-scripps/AutoDock-GPU` → `forlilab/AutoDock-GPU` | conversation | 226 | **boundary** | CUDA AutoDock-GPU is external; native workflow covers AutoDock4/Vina without claiming GPU parity. |
| 56 | `gnina/gnina` | conversation | 850 | **boundary** | GNINA CNN weights/scoring are external; pose analysis and explainable consensus remain native. |
| 57 | `forlilab/Meeko` | conversation | 230 | **covered** | Meeko-backed ligand PDBQT preparation and torsion audits. |
| 58 | `ccsb-scripps/AutoDock-Vina` → `forlilab/AutoDock-Vina` | conversation | 309 | **covered** | Vina configs, campaign execution, parsing and batch progress. |
| 59 | `molecularsets/moses` | supplemental | 167 | **covered** | generative/moses_metrics.py: validity, uniqueness, novelty, diversity and distribution shifts. |
| 60 | `BenevolentAI/guacamol` | supplemental | 46 | **covered** | Guacamol-style oracles/rediscovery/similarity and QED in chem/oracles.py. |
| 61 | `aspuru-guzik-group/selfies` → `the-matter-lab/selfies` | conversation | 98 | **covered** | chem/selfies_codec.py robust codec, round-trip and corruption audits; canonical repo moved to the-matter-lab. |
| 62 | `microsoft/molecule-generation` **(archived)** | conversation | 107 | **boundary** | Archived; GA/RNN/RL/MCTS generation exists, but pretrained MOSES-GCM weights are not bundled. |
| 63 | `MolecularAI/REINVENT4` | conversation | 3,290 | **covered** | REINVENT scoring transforms, RL and transfer workflows. |
| 64 | `MDAnalysis/mdanalysis` | supplemental | 1,236 | **partial** | chem/structure.py covers trajectory alignment/RMSD/RMSF; full MDAnalysis topology/trajectory IO stays external. |
| 65 | `openmm/pdbfixer` | supplemental | 71 | **boundary** | Receptor cleaning/preparation exists; PDBFixer itself and its OpenMM runtime are not vendored. |
| 66 | `openmm/openmm` | supplemental | 2,406 | **boundary** | Pose/structure analyses exist; OpenMM force-field MD engine is not reimplemented. |
| 67 | `3dmol/3Dmol.js` | conversation | 1,178 | **covered** | chem/graphics3d.py standalone browser 3D rendering/export. |
| 68 | `nglviewer/nglview` | conversation | 265 | **covered** | graphics3d/molgrid interactive molecule displays and exports. |
| 69 | `RBVI/ChimeraX` | conversation | 3,560 | **boundary** | Kabsch/superposition and 3D exports cover overlap; ChimeraX application/UI stays external. |
| 70 | `schrodinger/pymol-open-source` | conversation | 1,584 | **boundary** | Structure alignment, pose analysis and report exports cover overlap; PyMOL application stays external. |
| 71 | `chemosim-lab/ProLIF` | conversation | 149 | **covered** | chem/interactions.py implements 11 ProLIF-style interaction families and fingerprints. |
| 72 | `pharmai/plip` | supplemental | 109 | **boundary** | ProLIF interaction coverage is native; PLIP executable/XML detection stack is not bundled. |
| 73 | `dockstring/benchmarks` | supplemental | 7 | **added** | Original F2/PPAR/selective-JAK2 formulas, including corrected LCK median, in docking/dockstring.py + GUI. |
| 74 | `dockstring/dockstring` | supplemental | 160 | **added** | All 58 upstream target boxes and objective evaluation in docking/dockstring.py + Campaign bridge. |
| 75 | `jiamingf09-tech/DockMeow` | conversation | 107 | **covered** | Stepwise ligand/receptor/pocket/run/results/report workflow already matches DockMeow concepts; no PySide rewrite. |
| 76 | `matthiaskoenig/pkpdutils` | conversation | 341 | **GUI linked** | pkpd/nca_ext.py and pkstats.py now have extended-NCA/DDI playground surfaces. |
| 77 | `nlmixr2/nlmixr2` | conversation | 182 | **GUI linked** | Population NLME library remains tested; advanced PK notebook now exposes its regimen/qualification context. |
| 78 | `nlmixr2/rxode2` | supplemental | 2,101 | **GUI linked** | Exact linCmt compartments/events now drive the regimen playground pane. |
| 79 | `nlmixrdevelopment/RxODE` **(archived)** | conversation | 831 | **covered** | Archived legacy RxODE semantics are covered by odesim/compartments/events. |
| 80 | `metrumresearchgroup/mrgsolve` | conversation | 439 | **GUI linked** | mrgsolve event grammar and exact repeated dosing are exposed in the regimen pane. |
| 81 | `Open-Systems-Pharmacology/MoBi` | conversation | 2,483 | **GUI linked** | PBPK/PK-Sim data boundary is linked; MoBi projects are not executed by the lightweight GUI. |
| 82 | `Open-Systems-Pharmacology/OSPSuite-R` | conversation | 741 | **GUI linked** | OSPSuite-R sensitivity/PK registry exists; OSP browser and qualification now have GUI surfaces. |
| 83 | `Open-Systems-Pharmacology/PK-Sim` | conversation | 4,121 | **covered** | pkpd/physiology.py and tissue.py: ontogeny, tissue composition, Kp methods and IVIVE. |
| 84 | `biopython/biopython` | supplemental | 2,419 | **partial** | Sequence composition and PDB/structure handling cover overlap; full Biopython IO/phylogeny is out of scope. |
| 85 | `prody/ProDy` | supplemental | 764 | **partial** | Alignment/RMSD/RMSF cover overlap; ProDy normal-mode and ensemble engines stay external. |
| 86 | `Open-Systems-Pharmacology/OSPSuite.ReportingFramework` | supplemental | 500 | **GUI linked** | viz/report.py HTML/PDF reporting is linked to the advanced PK results context. |
| 87 | `soutomas/RappPKPLT` | supplemental | 16 | **GUI linked** | viz/pharma_plots.py plus the new regimen/maternal–fetal plots. |
| 88 | `matthiaskoenig/canagliflozin-model` | supplemental | 1,986 | **boundary** | Generic PBPK/PD, sensitivity and event engines cover concepts; fitted canagliflozin SBML/parameters are not vendored. |
| 89 | `gunjansubedi02/pkpd_nlmixr2_r_project` | conversation | 5 | **GUI linked** | pharma/nlme.py and pkpd/population.py carry the project conventions; advanced PK GUI links the workflow. |
| 90 | `dcmgithb/Pharmacometrics_Portfolio` | conversation | 51 | **covered** | pharma/fih.py and poppk.py: FIH floors, shrinkage, VPC, bootstrap and diagnostics. |
| 91 | `Open-Systems-Pharmacology/OSP-PBPK-Model-Library` | conversation | 1,138 | **GUI linked** | pkpd/osplib.py’s 40-model registry now has a playground browser. |
| 92 | `Open-Systems-Pharmacology/Itraconazole-Model` | conversation | 24 | **covered** | Digoxin/itraconazole compound parameters, Weibull dissolution, enzyme turnover and DDI machinery. |
| 93 | `Open-Systems-Pharmacology/Qualification-DDI-P-gp` | conversation | 24 | **GUI linked** | Real P-gp records, GMFE/Guest/two-fold qualification now include a DDI GUI evaluator. |
| 94 | `Open-Systems-Pharmacology/Digoxin-Model` | conversation | 22 | **covered** | pkpd/compounds.py transcribes digoxin reference parameters and P-gp profile. |
| 95 | `blakeaw/pysb-pkpd` | supplemental | 38 | **GUI linked** | Rule/event simulation concepts are covered by pkpd events/ODE and the regimen GUI; PySB is not required. |
| 96 | `DavAug/chi` | supplemental | 130 | **GUI linked** | Population/NLME and diagnostics cover Chi concepts; its full inference framework stays external. |
| 97 | `aimhubio/aim` | conversation | 2,092 | **covered** | Experiments card and runboard JSONL comparison; no hosted Aim service. |
| 98 | `mlflow/mlflow` | conversation | 10,369 | **covered** | Run/persistence/experiment tracking; no MLflow tracking server dependency. |
| 99 | `wandb/wandb` | conversation | 7,096 | **covered** | Experiment comparison, flagging, metrics and local run logs; no W&B cloud calls. |
| 100 | `tensorflow/tensorboard` | conversation | 1,934 | **covered** | ml/runboard.py: scalar/histogram/text/embedding events, reservoir, smoothing and hparams. |
| 101 | `biolab/orange3` | conversation | 1,146 | **covered** | Native workflow/cards and data curation cover Orange concepts; Orange canvas is not embedded. |
| 102 | `voxel51/fiftyone` | conversation | 7,168 | **covered** | ml/curation.py view algebra, duplicates/outliers/uniqueness and evaluation. |
| 103 | `cbouy/mols2grid` | conversation | 77 | **covered** | chem/molgrid.py molecule grid, selection and export. |
| 104 | `Open-Systems-Pharmacology/Vancomycin-Model` | supplemental | 19 | **GUI linked** | Vancomycin is indexed in OSP_PBPK_LIBRARY and existing TDM dose adjustment covers its use case. |
| 105 | `Open-Systems-Pharmacology/Pregnancy-Models` | supplemental | 44 | **added** | pkpd/pregnancy.py adds separate-fu maternal/fetal ODE, 16-project index and GUI; 27-compartment parity is refused. |
| 106 | `VainF/Torch-Pruning` | conversation | 213 | **covered** | ml/pruning.py local/global magnitude, structured channels and gradual retraining; model-surgery card. |
| 107 | `microsoft/nni` **(archived)** | conversation | 2,179 | **boundary** | Archived NNI service is external; native HPO/TPE/pruning/Bayesian search covers the local workflow. |
| 108 | `huggingface/pytorch-image-models` | conversation | 555 | **boundary** | Native MLP/GNN architectures exist; timm image checkpoints and its full model zoo are not bundled. |
| 109 | `kozistr/pytorch_optimizer` | conversation | 526 | **partial** | Autograd SGD/Adagrad/RMSprop/Adam/AdamW cover the core; the large pytorch_optimizer zoo is not cloned. |
| 110 | `keras-team/keras-tuner` | conversation | 138 | **covered** | ml/tuner.py and bayesopt.py: keras-tuner-style slots, Random/Hyperband/TPE/Bayesian search. |
| 111 | `ray-project/ray` | conversation | 10,579 | **boundary** | Ray distributed runtime is external; deterministic local tuning and campaign execution remain native. |
| 112 | `johnmarktaylor91/torchlens` | conversation | 4,938 | **covered** | ml/xai.py model_activation_trace operation/state graph and GUI model-graph action. |
| 113 | `szagoruyko/pytorchviz` | conversation | 10 | **covered** | Architecture graph/export surface in the Deep Learning tab. |
| 114 | `mert-kurttutan/torchview` | conversation | 229 | **covered** | Layer shapes/statistics and architecture reporting in the Deep Learning tab. |
| 115 | `lutzroeder/netron` | conversation | 277 | **covered** | Architecture card and model-graph summaries; Netron application is not embedded. |
| 116 | `pytorch/pytorch` | conversation | 22,674 | **covered** | NumPy/torch estimators, autograd, pruning and quantization cover the used computation subset. |
| 117 | `SeldonIO/alibi` | conversation | 751 | **covered** | Deterministic empirical tabular anchors with precision/coverage reporting. |
| 118 | `chr5tphr/zennit` | conversation | 81 | **partial** | Attribution families cover the local XAI workflow; Zennit’s full LRP rule/composite engine is not ported. |
| 119 | `meta-pytorch/captum` | conversation | 393 | **covered** | Captum-style gradients, IG, SmoothGrad, occlusion, DeepLift, SHAP and infidelity/sensitivity. |
| 120 | `microsoft/onnxruntime` | conversation | 11,607 | **boundary** | Quantization reports exist; ONNX graph conversion/runtime execution is not claimed. |
| 121 | `Lightning-AI/pytorch-lightning` | conversation | 1,051 | **covered** | ml/harness.py early stopping and deterministic training-loop semantics. |
| 122 | `huggingface/accelerate` | conversation | 356 | **covered** | Harness memory/sharding calculations; Accelerate launcher is not vendored. |
| 123 | `deepspeedai/DeepSpeed` | conversation | 2,100 | **covered** | Harness ZeRO stage 0–3 memory arithmetic and shard plans; no distributed DeepSpeed runtime. |
| 124 | `pytorch/ao` | conversation | 976 | **covered** | ml/quantization.py NF4/int8 and model-surgery reports; torchao deployment kernels stay external. |
| 125 | `NVIDIA/Model-Optimizer` | conversation | 2,566 | **boundary** | Pruning/quantization analysis exists; NVIDIA deployment/TensorRT optimizer pipeline stays external. |
| 126 | `bitsandbytes-foundation/bitsandbytes` | conversation | 185 | **covered** | NF4 grid, block-64 double quantization, int8 absmax and outlier-aware matmul. |
| 127 | `streamlit/streamlit` | conversation | 10,236 | **covered** | Native metrics, run cards, flagging and workflow UX; no Streamlit server rewrite. |
| 128 | `zauberzeug/nicegui` | conversation | 1,639 | **covered** | Refreshable previews, nine-position notifications and GUI helpers. |
| 129 | `TomSchimansky/CustomTkinter` | conversation | 136 | **covered** | Theme/accent packs, scaling, collapsible cards and live retinting. |
| 130 | `hoffstadt/DearPyGui` | conversation | 261 | **covered** | Collapsible sections, widget finder and GUI metrics/palette inspector. |
| 131 | `qtproject/pyside-pyside-setup` | conversation | 4,730 | **boundary** | viz/qt_backend.py is an optional adapter; the Tk playground is not rewritten into PySide. |
| 132 | `jessevig/bertviz` | conversation | 48 | **covered** | Layer×head attention grid and strongest token-pair digest. |
| 133 | `keisen/tf-keras-vis` | conversation | 95 | **covered** | Attribution methods, sign palettes and activation views in the XAI card. |
| 134 | `PAIR-code/saliency` | conversation | 57 | **covered** | Grayscale/diverging attribution palettes and SmoothGrad/occlusion semantics. |
| 135 | `divelab/DIG` | conversation | 458 | **covered** | ml/gnnxai.py GNN/attention/SubgraphX/PG explainers with fidelity/sparsity. |
| 136 | `understandable-machine-intelligence-lab/Quantus` | conversation | 236 | **covered** | Complexity, faithfulness estimate, max-sensitivity and combined Quantus-style GUI report. |
| 137 | `Robterheine/ferxgui` | conversation | 88 | **GUI linked** | NLME/Population PK now has an advanced playground surface; the FeRx R engine is not vendored. |
| 138 | `pythonguis/pythonguis-examples` | conversation | 1,889 | **boundary** | Reusable GUI patterns were absorbed where relevant; example applications are not bundled wholesale. |
| 139 | `flet-dev/flet` | conversation | 6,239 | **boundary** | Native desktop workflow remains Tk; Flet web/mobile runtime is not added. |
| 140 | `gradio-app/gradio` | conversation | 3,108 | **covered** | Examples pagination, flag CSV and toast semantics in the playground. |

## Totals and archived repositories

**added**: 7; **GUI linked**: 15; **covered**: 88; **partial**: 6; **boundary**: 19; **guide**: 5.

Archived at audit time: `microsoft/molecule-generation`, `nlmixrdevelopment/RxODE`, and `microsoft/nni`. Their status is disclosed in the table; archived code was not treated as a maintained dependency.

## Explicit non-parity boundaries

No CUDA/GNINA/OpenMM/ONNX/TensorRT execution, no hosted tracking service, no live ChEMBL client, no pretrained DTA/timm/DrugCell checkpoints, no 27-compartment pregnancy execution, no full Biopython/ProDy/MDAnalysis framework, no reaction CGR engine, and no wholesale GUI-framework rewrite. These are deliberate dependency/scope decisions, not silently missing buttons.

Focused verification for the new computation is in `tests/test_catalog_foldins.py`; real Tk callback drives are in `TestResearchCatalogFoldinsGui` (display-dependent and therefore skipped where Xvfb is unavailable).
