#!/usr/bin/env python3
"""Round 40 (100x-157), Task 1-a: walk the pythondatasciencehandbook clone in
the external handbook clone (via ``git ls-files``) and extract, for every notebook under
``notebooks/``:

1. the notebook title (first markdown cell that carries a ``#`` heading);
2. every code cell's imports and the KEY APIs used -- sklearn estimators,
   pandas/numpy/matplotlib/seaborn function calls -- by simple regex over the
   source lines (no execution, no nbconvert).

Output goes to stdout as compact per-notebook blocks, preceded by a repository
inventory summary.  Read-only with respect to the clone; the script writes
nothing (the deliverable -- the review deliverable (scratch/pdsh_review.md) -- is written by hand by the
reviewing agent from this output).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PDSH = Path(__file__).resolve().parent / "pdsh"

IMPORT_RE = re.compile(r"^\s*(?:from\s+([\.\w]+)\s+import\s+([^\n#]+)|import\s+([\w\.]+))")
SKLASS_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+)\s*\(")
ALIAS_CALL_RE = re.compile(r"\b(pd|np|plt|sns|ax|fig|sklearn)\.([A-Za-z_][A-Za-z0-9_]*)")
HEADING_RE = re.compile(r"^\s*#+\s+(.+?)\s*$")

# sklearn estimator / transformer / metric names we consider "key APIs".
KNOWN_SKLEARN = {
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "PolynomialFeatures",
    "LogisticRegression", "SVC", "SVR", "LinearSVC", "DecisionTreeClassifier",
    "DecisionTreeRegressor", "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoostingClassifier",
    "BaggingClassifier", "BaggingRegressor", "AdaBoostClassifier",
    "GaussianNB", "MultinomialNB", "BernoulliNB", "KNeighborsClassifier",
    "KNeighborsRegressor", "GaussianProcessRegressor", "GaussianProcessClassifier",
    "PCA", "KernelPCA", "Isomap", "LocallyLinearEmbedding", "MDS",
    "SpectralEmbedding", "TSNE", "KMeans", "MiniBatchKMeans", "GaussianMixture",
    "BayesianGaussianMixture", "KernelDensity", "NearestNeighbors",
    "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "cross_validate",
    "learning_curve", "validation_curve", "train_test_split", "StratifiedKFold",
    "KFold", "LeaveOneOut", "ShuffleSplit", "LabelEncoder", "OneHotEncoder",
    "CountVectorizer", "TfidfVectorizer", "SimpleImputer", "StandardScaler",
    "MinMaxScaler", "RobustScaler", "Normalizer", "make_pipeline", "Pipeline",
    "Silhouette", "silhouette_score", "accuracy_score", "mutual_info_score",
    "adjusted_rand_score", "homogeneity_score", "completeness_score",
    "v_measure_score", "adjusted_mutual_info_score", "fowlkes_mallows_score",
    "classification_report", "confusion_matrix", "mean_squared_error",
    "r2_score", "euclidean", "manhattan", "cosine", " pairwise_distances",
    "pairwise_distances", "manifold", "spectral_clustering", "make_blobs",
    "make_classification", "make_regression", "load_digits", "load_iris",
    "load_wine", "load_breast_cancer", "fetch_20newsgroups", "fetch_lfw_people",
    "fetch_olivetti_faces", "fetch_california_housing", "fetch_mldata",
    "Bunch", "LatentDirichletAllocation", "NMF", "TruncatedSVD", "LinearDiscriminantAnalysis",
    "IsotonicRegression", "HuberRegressor", "TheilSenRegressor", "RANSACRegressor",
    "ARDRegression", "BayesianRidge", "LogisticRegressionCV", "SGDClassifier",
    "Perceptron", "PassiveAggressiveClassifier", "MLPClassifier", "MLPRegressor",
    "AgglomerativeClustering", "DBSCAN", "MeanShift", "Birch", "AffinityPropagation",
    "SpectralBiclustering", "DictionaryLearning", "SparsePCA", "MiniBatchSparsePCA",
    "FactorAnalysis", "FastICA", "Nystroem", "RBFSampler", "SelectKBest",
    "SelectPercentile", "mutual_info_classif", "chi2", "f_classif",
    "RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector",
    "ColumnTransformer", "FeatureUnion", "TransformedTargetRegressor",
    "VotingClassifier", "VotingRegressor", "StackingClassifier", "StackingRegressor",
}


def git_ls_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(PDSH), "ls-files"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def notebook_source_cells(path: Path) -> tuple[list[str], list[str]]:
    """Return (markdown_lines, code_lines) for a notebook."""
    try:
        with path.open(encoding="utf-8") as fh:
            nb = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return [f"<UNREADABLE: {exc}>"], []
    md_lines: list[str] = []
    code_lines: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, str):
            src = [src]
        text = "".join(src)
        if cell.get("cell_type") == "markdown":
            md_lines.append(text)
        elif cell.get("cell_type") == "code":
            code_lines.append(text)
    return md_lines, code_lines


def notebook_title(md_cells: list[str]) -> str:
    for text in md_cells:
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                return m.group(1).strip()
    return "<no markdown heading found>"


def extract_apis(code_cells: list[str]) -> dict[str, list[str]]:
    imports: Counter[str] = Counter()
    sk_names: Counter[str] = Counter()
    alias_calls: Counter[str] = Counter()
    sk_imported: set[str] = set()
    for text in code_cells:
        for line in text.splitlines():
            m = IMPORT_RE.match(line)
            if m:
                if m.group(1):  # from X import ...
                    module = m.group(1)
                    names = [n.strip().split(" as ")[0].strip()
                             for n in m.group(2).replace("(", " ").replace(")", " ").split(",")]
                    imports[module] += 1
                    if module.startswith("sklearn") or module.startswith("."):
                        for n in names:
                            if n and n[0].isupper() or n in KNOWN_SKLEARN:
                                sk_imported.add(n)
                elif m.group(3):
                    imports[m.group(3)] += 1
            # alias calls (pd./np./plt./sns.)
            for alias, attr in ALIAS_CALL_RE.findall(line):
                if alias in ("pd", "np", "plt", "sns"):
                    alias_calls[f"{alias}.{attr}"] += 1
            # sklearn class instantiations
            for name in SKLASS_RE.findall(line):
                if name in KNOWN_SKLEARN or name in sk_imported:
                    sk_names[name] += 1
    for name in sk_imported:
        if name not in sk_names:
            sk_names[name] += 0  # imported but maybe used via other syntax
    return {
        "imports": sorted(imports),
        "sklearn": sorted(sk_names),
        "alias_calls": sorted(
            (k for k, v in alias_calls.items() if v >= 1),
        ),
    }


def main() -> int:
    files = git_ls_files()
    top: Counter[str] = Counter()
    for f in files:
        top[f.split("/")[0]] += 1

    print("=" * 78)
    print("REPOSITORY INVENTORY (git ls-files): %d tracked files" % len(files))
    print("=" * 78)
    for name, count in sorted(top.items()):
        print(f"  {name:<16} {count:>4} files")

    print("\nnotebooks/data/ contents:")
    for f in sorted(files):
        if f.startswith("notebooks/data/"):
            print(f"  {f}")

    nb_files = sorted(f for f in files if f.startswith("notebooks/") and f.endswith(".ipynb"))
    print(f"\nnotebooks/*.ipynb count: {len(nb_files)} "
          f"(content notebooks: {sum(1 for f in nb_files if 'Untitled' not in f)})")
    print("notebooks/figures/ count: %d" % sum(
        1 for f in files if f.startswith("notebooks/figures/")))
    print("notebooks_v1/*.ipynb count: %d" % sum(
        1 for f in files if f.startswith("notebooks_v1/") and f.endswith(".ipynb")))
    print("notebooks_v1/data/ count: %d" % sum(
        1 for f in files if f.startswith("notebooks_v1/data/")))

    print("\n" + "=" * 78)
    print("NOTEBOOK-BY-NOTEBOOK EXTRACTION (notebooks/, filename order)")
    print("=" * 78)
    for rel in nb_files:
        path = PDSH / rel
        md, code = notebook_source_cells(path)
        title = notebook_title(md)
        apis = extract_apis(code)
        print(f"\n### {rel}")
        print(f"    title : {title}")
        print(f"    cells : {len(md)} markdown / {len(code)} code")
        print(f"    imports: {'; '.join(apis['imports']) if apis['imports'] else '-'}")
        print(f"    sklearn: {'; '.join(apis['sklearn']) if apis['sklearn'] else '-'}")
        calls = apis["alias_calls"]
        print(f"    calls : {'; '.join(calls) if calls else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
