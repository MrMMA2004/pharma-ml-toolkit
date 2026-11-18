# Round 187 — the six-repo review (captum · shap · interpret · DIG · pytorch_geometric · rdkit)

The user's instruction: "Check the contents and files in this GitHub links completely
and one by one and add the missing items to playground.py (Deep Learning)."

Method: each repo cloned `--depth 1 --filter=blob:none --no-checkout` to
`~/gh187/` (the home path, outside the repo) (OUTSIDE the repo — the round-186 rule-7 lesson), full file
inventories via `git ls-tree -r HEAD --name-only` (393 / 616 / 499 / 458 / 1548 /
6,518 files), subtree counts per directory, then per-subpackage dispositions below.
Dispositions: **ADDED** (missing from the toolkit, offline-feasible, Deep-Learning
relevant) · **NA** (already covered elsewhere in the toolkit, or not applicable to
a CPU/offline pharma toolkit) · **STRETCH** (feasible but deferred with reason).

Existing coverage this round diffs against:
- `ml/xai.py` (round 185): input_x_gradient, integrated_gradients, smoothgrad,
  occlusion, layer_activation_trace/summary (LayerActivation), attention grid
  (bertviz), the numpy LayerStack + analytic activation derivatives.
- `ml/explain.py` (rounds 42-43 + 160): shap_summary, shap_interactions,
  group_difference; viz/ml_plots.py: waterfall / force / dependence / heatmap /
  beeswarm / bar figures; the Results card's SHAP rows.
- `chem/` (rdkit-adjacent, all offline): molgraph.py (SMILES→Molecule graph,
  rings, kekulization), descriptors.py, features.py (ECFP/Morgan), similarity.py,
  standardize.py, tables.py, structure.py, converters.py, molgrid.py, interpret.py.
- `ml/autograd.py`: the reverse-mode tape (Tensor/Module/Linear/optimizers/
  gradient_check) — the GNN substrate.
- NO graph engine exists yet (no Data/edge_index, no message passing).

---

## 1. meta-pytorch/captum — 393 files

`captum/attr/_core` one by one:
| file | disposition |
|---|---|
| saliency.py | NA — smoothgrad wraps the same raw-gradient scorer; xai.py exposes it |
| input_x_gradient.py | ADDED round 185 |
| integrated_gradients.py | ADDED round 185 |
| occlusion.py | ADDED round 185 |
| smoothgrad | ADDED round 185 (noise_tunnel's smoothgrad op) |
| gradient_shap.py | **ADDED this round** — stochastic baseline-mixture attributions on the LayerStack |
| deep_lift.py (+ layer/layer_deep_lift.py) | **ADDED this round** — rescale rule over the stack |
| kernel_shap.py | **ADDED this round** — weighted-coalition least squares (the SHAP kernel), distinct from lime's similarity kernel |
| lime.py | **ADDED this round** — local linear surrogate, exponential similarity kernel |
| shapley_value.py | **ADDED this round** — permutation-sampling Shapley |
| feature_permutation.py / feature_ablation.py | **ADDED this round** (permutation; ablation ≡ occlusion, cross-linked) |
| noise_tunnel.py | **ADDED this round** — the smoothgrad/sqgrad/ablation gate around any base method |
| lrp.py (+ layer/layer_lrp.py) | **ADDED this round** — epsilon z·(w/Σz) relevance backprop |
| guided_backprop_deconvnet / guided_grad_cam | NA — needs ReLU-clipped backprop through conv nets; the toolkit's DL tab is dense/MLP |
| grad_cam (layer/grad_cam.py) | NA — no conv feature maps in the engine |
| layer_conductance.py / neuron/* | **ADDED this round** — per-neuron (act−act_base)·∂y/∂z on the traced stack |
| llm_attr.py, remote_provider.py, dataloader_attr.py | NA — LLM/remote/iterator plumbing |
| captum.influence (TracIn & co) | STRETCH — needs stored checkpoints per epoch; recorded for a later round |
| captum.concept (TCAV/CAV) | **ADDED this round** — CAV linear probes + directional derivatives at a chosen layer |
| captum.robust (FGSM/PGD, perturbation) | **ADDED this round** — fgsm_attack/pgd_attack on the numpy stack (input gradients exist) |
| captum.metrics (sensitivity_max, infidelity, effectiveness) | **ADDED this round** — the attribution validators |

## 2. shap/shap — 616 files

| area | disposition |
|---|---|
| explainers/_kernel | ADDED rounds 42-43 (shap_summary's weighted least squares) |
| explainers/_tree | **ADDED this round** — exact TreeSHAP (path-dependent) for the sklearn tree ensembles models.py ships |
| explainers/_linear | **ADDED this round** — covariance-shrinkage linear SHAP |
| explainers/_permutation | **ADDED this round** — antithetic permutation explainer |
| explainers/_exact | **ADDED this round** — all-coalitions explainer for ≤ ~12 features |
| explainers/_deep, _gradient, _partition, _sampling, _coalition, pytree | STRETCH — deep/gradient need torch backprops per-mask (recorded); partition masks are text/image-shaped |
| explainers/other/* | NA — wrappers around lime/maple (lime lands via captum above) |
| plots: waterfall/force/bar/beeswarm/heatmap/partial_dependence/group_difference | ADDED rounds 42-160 |
| plots: _decision | **ADDED this round** — the cumulative-cohort decision figure |
| plots: _monitoring | **ADDED this round** — the loss/label-drift monitoring figure |
| plots: _violin | **ADDED this round** — explanation violin per feature |
| plots: _embedding | **ADDED this round** — explanations projected through ml/umap.py |
| plots: _image/_text, colors/_colorconv | NA — image/text maskers out of scope |
| maskers/* | NA — the toolkit's tabular explainers mask by mean/median directly (documented) |
| links.py | **ADDED this round** — identity/logit links for the explainer API |
| benchmark/, javascript/, cext/ | NA — benchmark harness + JS dashboards + C extension |

## 3. interpretml/interpret — 499 files

| area | disposition |
|---|---|
| glassbox/_ebm_core (bin/boost/tensor/measure+rank interactions/multiclass) | **ADDED this round** — the ExplainableBoostingMachine: quantile binning, cyclic additive boosting, FAST pair interactions, shape functions, local contributions, multiclass OVR |
| glassbox/_linear.py | NA — the toolkit's models.py already ships plain linear models with coefficients |
| glassbox/_decisiontree.py | NA — sklearn DecisionTree already present |
| glassbox/_skoperules.py, _aplr.py | STRETCH — rule-list induction / APLR recorded for later |
| interpret/perf (_curve, _regression) | NA — viz/ml_plots.py already draws ROC/PR/calibration/learning curves |
| interpret/visual (dashboard/udash) | NA — playground.py IS the dashboard (Tk cards, not a web dash) |
| interpret/blackbox, greybox, provider, ext, powerlift | NA — SHAP/LIME wrappers + eval harness plumbing; the toolkit hosts its own explainers |
| interpret/data | NA — the toolkit's data.py + curation.py cover the load/audit path |

## 4. divelab/DIG — 458 files

| area | disposition |
|---|---|
| xgraph/method/gnnexplainer.py | **ADDED this round** — edge-mask gradient search on the new GNN |
| xgraph/method/pgexplainer.py | **ADDED this round** — parametric edge scorer, trained per model then explained per graph |
| xgraph/method/subgraphx.py | **ADDED this round** — Shapley-guided connected-subgraph search (teaching scale) |
| xgraph/method/gradcam.py, gnn_gi.py, gnn_lrp.py, deeplift.py | NA — gradcam needs conv grids; GI/LRP/DeepLIFT need graph autograd hooks the tape now has — **GI + LRP-style edge relevance folded into gnnxai's mask scorer** (documented) |
| xgraph/method/flowx.py, shapley.py | STRETCH — flow-based + full Shapley graph explainers recorded |
| xgraph/evaluation/metrics.py | **ADDED this round** — fidelity+/-, sparsity, accuracy@k |
| xgraph/dataset (syn/mol/nlp) | **ADDED (part)** — synthetic motif graphs for tests/GUI; the molecule route goes through chem/featurize.py; NLP NA |
| xgraph/models | NA — the GNN comes from pyg absorption below |
| ggraph (GraphAF/GraphDF) | STRETCH — autoregressive flow molecule generation is a round of its own |
| sslgraph (GRACE/GraphCL/InfoGraph) | **ADDED (part)** — the graph augmentations + InfoGraph-style MI objective for unsupervised node/graph embeddings on the tape |
| threedgraph, ggraph3D | NA — 3D convolutions (SchNet/DimeNet) need point clouds; the toolkit's docking side is score-based |
| auggraph, lsgraph, fairgraph, oodgraph | STRETCH/NA — augmentation-for-robustness, label-shift, fairness and OOD graph benchmarks recorded |

## 5. pyg-team/pytorch_geometric — 1,548 files

| area | disposition |
|---|---|
| data/data.py + batch.py + collate.py | **ADDED this round** — Data (x/edge_index/edge_attr/y), Batch collate with node-offset remap, to_undirected, add_self_loops, degree |
| data/hetero_data, hypergraph, dataset.py, database, datapipes, feature/graph_store, download/extract | NA — hetero/hypergraph/disk datasets out of scope |
| utils/scatter, utils/softmax (the segment ops) | **ADDED this round** — scatter_add/scatter_max/scatter_softmax as autograd-tape nodes with gradcheck |
| nn/aggr: basic (sum/mean/max), attention, set2set | **ADDED (core)** — sum/mean/max + gated attention aggregation; the exotic aggregators (GMT, patch_transformer, equilibrium...) STRETCH |
| nn/conv: gcn_conv | **ADDED this round** |
| nn/conv: sage_conv | **ADDED this round** |
| nn/conv: gin_conv | **ADDED this round** |
| nn/conv: gat_conv | **ADDED this round** — multi-head scaled attention over edges |
| nn/conv: edge_conv | **ADDED this round** |
| nn/conv: cheb_conv | STRETCH — Chebyshev recursion needs the scaled Laplacian; recorded |
| nn/conv: the other ~40 convs | NA — the five canonical message-passing families above carry the concept |
| nn/pool: glob.py (global sum/mean/max) | **ADDED this round** |
| nn/pool: topk.py (TopKPooling) | **ADDED this round** — score-select pooling with the pyg gradient path |
| nn/pool: sag_pool.py | **ADDED this round** — SAGPooling (TopK over attention scores) |
| nn/pool: the rest (asap/pan/graclus/cluster/mem/knn...) | STRETCH/NA — recorded |
| nn/models (autoencode etc.) | STRETCH |
| explain/algorithm: gnn_explainer, pg_explainer, attention_explainer | **ADDED this round** (merged with DIG's xgraph twins — one engine, both lineages credited) |
| explain/metric: faithfulness/fidelity | **ADDED this round** |
| explain/graphmask, captum bridge, config.py's hetero paths | NA/STRETCH |
| nn/conv cugraph/, llm/, distributed/, graphgym/, loader/, sampler/ | NA — CUDA/LLM/distributed/config-platform plumbing |
| transforms/ (62 files) | **ADDED (core)** — ToUndirected, AddSelfLoops, NormalizeFeatures, Sign, LocalDegreeProfile; the rest NA |

## 6. rdkit/rdkit — 6,518 files

| area | disposition |
|---|---|
| rdkit/Chem (293 py files: atoms/bonds/rings/kekulize/SMILES parse) | NA — chem/molgraph.py reimplements the offline core (rounds of hardening) |
| rdkit/DataStructs (fingerprints/similarity) | NA — chem/features.py (ECFP/Morgan) + similarity.py |
| rdkit/Chem/rdMolDescriptors | NA — chem/descriptors.py |
| rdkit/Chem/Scaffolds (Murcko) | NA — chem/molgraph.py's ring system + ml/splits.py scaffold split |
| rdkit/ML (75 files: descriptors, cluster, feature selectors) | **ADDED (part)** — the one piece the DL pipeline actually needs: **chem/featurize.py**, the molecule→graph bridge (atom/bond feature tables → ml/gnn Data), because rdkit's role in DL frameworks is exactly this featurization |
| rdkit/sping, VLib, Dbase, SimDivFilters, DataManip | NA — drawing/virtual-lib/db plumbing |

---

## The absorption plan (downstream of these tables)

1. `ml/autograd.py`: scatter_add / scatter_max / scatter_softmax tape nodes (gradcheck doctests).
2. `ml/gnn.py`: Data/Batch + the five convs + global/TopK/SAG pooling + GraphClassifier (sklearn-style) + synthetic motif graphs.
3. `chem/featurize.py`: Molecule → Data (atom/bond feature tables), the rdkit bridge.
4. `ml/gnnxai.py`: GNNExplainer / PGExplainer / AttentionExplainer / SubgraphX / fidelity-sparsity metrics / InfoGraph-style unsupervised embeddings.
5. `ml/glassbox.py`: the EBM.
6. `ml/xai.py` +: gradient_shap, deep_lift, kernel_shap, lime, shapley_value, feature_permutation, noise_tunnel, lrp, layer_conductance, CAV/TCAV, fgsm/pgd, sensitivity/infidelity/effectiveness.
7. `ml/explain.py` +: TreeSHAP (sklearn trees), LinearExplainer, PermutationExplainer, ExactExplainer, links.
8. `viz/`: graph-explanation figures + EBM shape functions + the new shap figures.
9. playground.py Deep Learning tab: Graph model / Graph explain / Glassbox EBM cards + the extended attribution registry.
