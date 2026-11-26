# Test profiles (Part 5, incremental)

Run the Python commands from the toolkit root. Use separate virtual environments
for each profile; start with `python -m pip install -r requirements.txt`,
`python -m pip install --no-deps -e .`, then the packages listed below. The
commands are **named targeted profiles**, not assertions that the entire backend,
GUI, or release is covered. Record passes, failures and skips separately.
`requirements.txt` remains the pinned core stack; optional version pins below
are the versions exercised in this phase, not claims about every supported version.

| Profile | Additional dependencies | Targeted command |
| --- | --- | --- |
| Core backend | `pytest==9.0.2`; no GUI, RDKit, Mordred, Meeko or gemmi | `python -m pytest tests/test_data_loading.py tests/test_pk.py tests/test_conformal.py -o addopts="-q --tb=short"` |
| scikit-learn compatibility | pytest; core pins with a deliberate `--no-deps` replacement of scikit-learn to 1.3.0, 1.5.2 or 1.8.0 in separate environments | `python -m pytest tests/test_estimator_compatibility.py tests/test_conformal.py tests/test_conformal_classification.py tests/test_label_conditional_conformal.py tests/test_coverage_diagnostics.py -o addopts="-q --tb=short"` |
| Cheminformatics | pytest; `rdkit==2026.9.1`, `mordred==1.2.0` | `python -m pytest tests/test_descriptors.py -o addopts="-q --tb=short"` |
| Docking preparation | pytest; `rdkit==2026.9.1`, `meeko==0.8.0`, `gemmi==0.7.5` | `python -m pytest tests/test_docking_prep.py -o addopts="-q --tb=short"` |
| Retired PySide6 UI (archival tests) | From the toolkit root, install `python -m pip install -e '.[gui-test]'`; pinned CI versions in workflow; `matplotlib==3.10.9`; OS Qt/GL libraries. This does not test the supported React/Tauri Studio. | `QT_QPA_PLATFORM=offscreen PYQTGRAPH_QT_LIB=PySide6 python -m pytest legacy/tests_ui/test_molecular_dynamics_page.py legacy/tests_ui/test_main_window.py -o addopts="-q --tb=short"` |
| Notebook and provenance | pytest; pinned RDKit; `nbformat` | `python -m pytest tests/test_data_provenance.py tests/test_notebook_variants.py tests/test_docs_index.py -o addopts="-q --tb=short"`; `python tools/build_example_data.py --check` |
| Web integration/parity | Full monorepo, Bun, Python core and pytest; RDKit and matplotlib plus system Xrender libraries for bridge rendering; some probes also depend on local scratch scripts and may write temporary probe output | `python -m pytest tests/test_web_parity.py tests/test_version_carriers.py -o addopts="-q --tb=short"`; web build/lint separately from repository root using the commands in CI |
| Standalone release ZIP | Isolated extraction, pytest, pinned RDKit and `nbformat`; no Git or web tree | See [standalone testing](STANDALONE_TESTING.md) for the explicit `--standalone-release` selection. |
| Full source/release | All required optional stacks, platform/system libraries, Bun, real release inputs, and engine runtimes | `python -m pytest` is a starting point, **not** a complete release validation by itself. Run the other profiles, archive verification and engine checks separately; see the audit for remaining blockers. |

Use `python -m pip install -c requirements.txt ...` for optional package pins to
avoid silently changing the core stack. Installing the retired Qt test wheels
alone does not supply system GL libraries; offscreen mode still loads Qt's
native libraries.
The docking preparation profile exercises Meeko-backed preparation, not docking
engine execution. Running engine mocks or checking binary hashes is not a real
engine invocation.

The root CI workflow currently includes a focused core matrix, chemistry and
docking-preparation selections, React/Tauri frontend tests and build, a separate
focused selection for the retired PySide6 source, and a web-console build. The
archived Qt job does not validate the current desktop app. The broader notebook,
web-parity, full legacy-Qt and full source/release profiles are not implied by
those jobs. For observed results and unexecuted work, see
[Part 5 audit](PART5_PHASE0_AUDIT.md).

## Windows bundled-engine smoke

CI invokes `python tools/windows_engine_smoke.py` on Windows: it verifies bundled
binary hashes, invokes AutoGrid4 and AutoDock4 help only, and attempts a Vina
docking run on deliberately artificial tiny PDBQT geometry. It checks process
exit, a finite score and a pose record, not chemical or biological validity,
reproducibility across engines, or the toolkit runner. Logs and CI results, not
this description, establish what actually ran. AutoGrid4 and AutoDock4 map
generation and docking require separate validation. The script requires Windows.

For interrupted release publication, see
[read-only recovery inventory](INTERRUPTED_RELEASE_RECOVERY.md).

The Windows docking integration job also uses the toolkit's `run_vina`,
`run_vina_batch`, `write_gpf`, `write_dpf`, and `run_autodock4` on synthetic
fixtures. It checks nonempty outputs, a finite reported score and generated
maps where applicable; see the job result before claiming success. These
fixtures are not a real receptor or ligand, and neither scientific validity
nor input/engine reproducibility has been established. Vina batch captures
console output as its log rather than relying on the unsupported Vina 1.2
`--log` option. A two-file pose/log promotion is not crash-atomic. Resume's
existing-file heuristic does not verify inputs or provenance: use a new output
directory if the configuration, ligand, receptor, engine or seed changes.

### AutoGrid and Vina geometry

The same `GridBox` historically writes `npts` to AutoGrid but uses
`(npts - 1) * spacing` for Vina. The bundled AutoGrid test showed `npts 30`
produces a map with `NELEMENTS 30` and 31 grid points; `npts 21` was silently
reduced to `NELEMENTS 20`. Other engine builds may differ. To explicitly request
the predicted AutoGrid span as Vina kwargs for **positive even** `npts`:

```python
from pharmaml.docking.preparation import GridBox
box = GridBox(0, 0, 0, npts=(30, 30, 30), spacing=0.375)
legacy = box.as_vina_kwargs()                         # 10.875 Å, unchanged
matched = box.as_vina_kwargs(convention="autogrid")   # 11.25 Å
```

The opt-in rejects odd/invalid values, but it predicts rather than validates an
engine map: inspect `NELEMENTS`, `SPACING` and `CENTER` in that actual map,
including engine size limits and any precision effects. `write_gpf` warns on
odd npts but preserves the requested file. Existing `size`, `compute_grid_box`,
and `dock_smiles_array` defaults remain unchanged; the opt-in does not silently
update them. Changing the search box can change docking results. Neither toy
geometry nor matching dimensions validates scientific docking accuracy.
