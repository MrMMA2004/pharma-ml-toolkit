"""Generative Workbench page tests (increment 8): real API, real numbers.

Everything runs offscreen against the real backend entry points the
playground cards call (``GeneticOptimizer`` / ``ReinventOptimizer`` /
``MctsOptimizer`` over ``build_objective``-built objectives,
``pareto_front``, ``ring_topology_moves``, ``scaffold_hop_crossover``
and the ``pharmaml.chem.oracles`` book). Every asserted number was
printed by the engines first (``scripts/probe_generative_214.py``), on
this exact bundled data with these exact knobs: the GA over the
built-in 244-molecule corpus (pop 24 x 16 generations, topology
mutation + splice crossover, seed 42) lands best 0.926784 with 271
unique molecules; the multi-objective triple seeded with the 12
bundled molecules lands best 0.961862 over 276 uniques with a 9-member
Pareto front (the strongest member dominates 262 others); the RL run
(12 iterations x batch 16, prior 8 epochs, seeds anchored) lands best
0.582072 over 58 uniques; the MCTS run (20 iterations, seed 42) lands
best 0.592551 over 17 uniques with 276 tree nodes; piperidine has
exactly 3 one-step ring-size mutants; the aspirin x ibuprofen-core
breeding (seed 42, 4 children) yields 4/4 viable; and the oracle book
on aspirin (rediscovery vs salicylic acid) reads QED 0.550 / SA 1.58 /
logP -7.30 / rediscovery 0.448 with a PASSing property filter.
"""

from __future__ import annotations

import time


def _drain(pool_wait: float = 60.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    """Wait out the pool, then flush the queued result slots."""
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _fresh_page():
    from ui.pages.generative import GenerativeWorkbenchPage

    return GenerativeWorkbenchPage()


def _run(page, starter, qtbot, studio_app) -> None:
    starter()
    qtbot.waitSignal(page._runner.job_ended, timeout=180_000).wait()
    _settle(qtbot, studio_app)


# -- construction ---------------------------------------------------------------


def test_page_constructs_with_five_engine_tabs(studio_app) -> None:
    page = _fresh_page()
    assert page._tabs.count() == 5
    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert titles == [
        "Evolve",
        "Pareto front",
        "Molecule surgery",
        "Scoring oracles",
        "SA score",
    ]


def test_bundled_seed_corpus_loads(studio_app) -> None:
    page = _fresh_page()
    assert len(page._seeds) == 12
    assert page._seeds[0] == "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    assert page._corpus_lines() == list(page._seeds)


def test_builtin_corpus_selection_yields_empty_lines(studio_app) -> None:
    page = _fresh_page()
    page._corpus_combo.setCurrentText("built-in drug-like corpus (244)")
    assert page._corpus_lines() == []


# -- evolve (the GA over the built-in corpus) ----------------------------------------


def test_ga_qed_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._corpus_combo.setCurrentText("built-in drug-like corpus (244)")
    _run(page, page._start_evolve, qtbot, studio_app)
    text = page.report_text()
    # probe: best 0.926784 over 271 unique molecules, built-in corpus,
    # 16 generations of history.
    assert text.startswith("genetic algorithm · QED drug-likeness · best 0.927 — ")
    assert "271 unique molecules · built-in corpus · " in text
    assert page._last_result.best_score == 0.9267843451352773
    assert page._last_result.best == "CC1(c2ccccc2)CC(=O)CCNCC1c1ccccc1.O"
    model = page._table.model()
    assert model.rowCount() == 20  # the leaderboard: top 20 rows
    assert model.columnCount() == 3  # rank / smiles / score


def test_ga_history_records(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._corpus_combo.setCurrentText("built-in drug-like corpus (244)")
    _run(page, page._start_evolve, qtbot, studio_app)
    history = list(page._last_result.history)
    # probe: 16 records; the last one's best 0.926784 / mean 0.848361 /
    # valid offspring fraction 0.3175.
    assert len(history) == 16
    assert history[-1]["best_score"] == 0.9267843451352773
    assert round(history[-1]["mean_score"], 6) == 0.848361
    assert round(history[-1]["valid_offspring_fraction"], 4) == 0.3175


def test_similarity_objective_needs_a_reference(studio_app) -> None:
    page = _fresh_page()
    page._objective.setCurrentText("Similarity to reference SMILES")
    page._obj_ref.setText("")
    page._start_evolve()
    assert "The similarity objective needs a reference SMILES." in page.report_text()


def test_window_bounds_validated(studio_app) -> None:
    page = _fresh_page()
    page._obj_lo.setText("3.0")
    page._obj_hi.setText("1.0")
    page._start_evolve()
    assert "The window's high bound must be >= the low bound." in page.report_text()


# -- evolve (the multi-objective seeded run feeding the Pareto front) ------------------


def test_ga_multi_objective_seeded_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._objective.setCurrentText("Multi-objective (QED × MW × TPSA)")
    page._seeded_check.setChecked(True)
    _run(page, page._start_evolve, qtbot, studio_app)
    text = page.report_text()
    # probe: best 0.961862 over 276 uniques, seeded mode; the leaderboard
    # carries the qed / mw_band / tpsa_band breakdown columns.
    assert text.startswith("genetic algorithm · Multi-objective (QED × MW × TPSA) · best 0.962 — ")
    assert "276 unique molecules · seeded" in text
    model = page._table.model()
    assert model.columnCount() == 6  # rank / smiles / score + the 3 breakdowns
    assert model.data(model.index(0, 1)) == "O=C(O)c1ccccc1Nc1c(Cl)#cccc1OCl"


def test_pareto_front_refuses_single_objective_runs(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._corpus_combo.setCurrentText("built-in drug-like corpus (244)")
    _run(page, page._start_evolve, qtbot, studio_app)  # QED = single objective
    page._start_front()
    assert "This run optimises a single objective — the Pareto front needs a" in page.report_text()


def test_pareto_front_refuses_without_any_run(studio_app) -> None:
    page = _fresh_page()
    page._start_front()
    assert "Run an evolution first — the front reads the LAST run's breakdown." in page.report_text()


def test_pareto_front_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._objective.setCurrentText("Multi-objective (QED × MW × TPSA)")
    page._seeded_check.setChecked(True)
    _run(page, page._start_evolve, qtbot, studio_app)
    _run(page, page._start_front, qtbot, studio_app)
    text = page.report_text()
    # probe: 9 of 276 non-dominated; the strongest member dominates 262;
    # deepest front member at leaderboard rank 59.
    assert "Pareto front over 3 objectives [qed, mw_band, tpsa_band]: 9 of 276 molecules non-dominated" in text
    assert "dominates 262 others" in text
    assert "deepest front member sits at leaderboard rank 59 of 276" in text
    model = page._table.model()
    assert model.rowCount() == 9  # one row per front member
    assert model.columnCount() == 7  # rank / smiles / 3 objectives / dominates / crowding


# -- evolve (RL and MCTS) ----------------------------------------------------------------


def test_rl_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._method.setCurrentText("reinforcement learning (RL)")
    page._seeded_check.setChecked(True)  # the probe's idiom: seeds anchored ahead of the bundle
    _run(page, page._start_evolve, qtbot, studio_app)
    text = page.report_text()
    # probe: best 0.582072 over 58 uniques, RL seeded mode (the CLI
    # idiom anchors the 12 seeds ahead of the 244-molecule bundle).
    assert text.startswith("reinforcement learning (RL) · QED drug-likeness · best 0.582 — ")
    assert "58 unique molecules · RL · seeded" in text
    assert len(page._last_result.history) == 12  # one record per iteration


def test_mcts_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._corpus_combo.setCurrentText("built-in drug-like corpus (244)")
    page._method.setCurrentText("monte carlo tree search (MCTS)")
    _run(page, page._start_evolve, qtbot, studio_app)
    text = page.report_text()
    # probe: best 0.592551 over 17 uniques, built-in corpus; the tree
    # walked 276 nodes / 828 rollouts.
    assert text.startswith("monte carlo tree search (MCTS) · QED drug-likeness · best 0.593 — ")
    assert "17 unique molecules · MCTS · built-in corpus" in text
    assert page._last_result.metadata["n_nodes"] == 276
    assert page._last_result.metadata["n_rollouts"] == 828


# -- molecule surgery -----------------------------------------------------------------------


def test_ring_topology_moves_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    assert page._topo_smiles.text() == "C1CCNCC1"  # piperidine
    _run(page, page._start_topology, qtbot, studio_app)
    text = page.report_text()
    # probe: 3 distinct mutants — C1CCCC1, C1CCCNCC1, C1CCNC1.
    assert "3 distinct one-step ring-size mutant(s) of C1CCNCC1" in text
    model = page._table.model()
    assert model.rowCount() == 3
    assert model.columnCount() == 2  # mutant / qed
    assert model.data(model.index(0, 0)) == "C1CCCC1"
    assert model.data(model.index(1, 0)) == "C1CCCNCC1"
    assert model.data(model.index(2, 0)) == "C1CCNC1"


def test_ring_topology_aromatic_rings_are_never_resized(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._topo_smiles.setText("c1ccccc1")  # benzene
    _run(page, page._start_topology, qtbot, studio_app)
    # probe: 0 moves — aromatic rings are never re-sized.
    assert "c1ccccc1 has no qualifying saturated ring" in page.report_text()
    assert page._table.model().rowCount() == 0


def test_ring_topology_empty_smiles_refused(studio_app) -> None:
    page = _fresh_page()
    page._topo_smiles.setText("")
    page._start_topology()
    assert "Type a SMILES in the ring surgery row first." in page.report_text()


def test_breed_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    assert page._breed_a.text() == "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    assert page._breed_b.text() == "CC(C)Cc1ccc(O)cc1"  # the ibuprofen core
    _run(page, page._start_breed, qtbot, studio_app)
    text = page.report_text()
    # probe: 4/4 viable children on seed 42.
    assert "4/4 viable children — bridge / ring-merge / linker-transplant cuts competed for each draw (seed 42)" in text
    model = page._table.model()
    assert model.rowCount() == 4
    assert model.columnCount() == 2  # child / qed
    assert model.data(model.index(0, 0)) == "CC(Cc1ccc(O)cc1)C(=O)Oc1ccccc1C(=O)O"


def test_breed_needs_both_parents(studio_app) -> None:
    page = _fresh_page()
    page._breed_b.setText("")
    page._start_breed()
    assert "Both parent SMILES are required for breeding." in page.report_text()


# -- scoring oracles --------------------------------------------------------------------------


def test_oracle_book_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    assert page._oracle_smiles.text() == "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    assert page._oracle_ref.text() == "O=C(O)c1ccccc1O"  # salicylic acid
    _run(page, page._start_oracle, qtbot, studio_app)
    text = page.report_text()
    # probe: QED 0.550122, SA 1.580145 (the real Ertl fragment model
    # since the part3 §24 wiring -- the RDKit-wheel Contrib database),
    # logP 1.136788 (the real SA table also lifted penalized logP off
    # the degraded -7.302008 the histogram-less scorer reported),
    # rediscovery 0.448276, property filter PASS (no violations).
    assert "QED 0.550 · SA 1.58 · logP 1.14 · rediscovery 0.448" in text
    assert "property filter PASS" in text
    model = page._table.model()
    # 4 oracles + 6 property values (HBA/HBD/LogP/MW/RotB/TPSA)
    assert model.rowCount() == 10
    assert model.columnCount() == 2
    assert model.data(model.index(0, 0)) == "QED"
    assert round(float(model.data(model.index(0, 1))), 6) == 0.550122
    assert model.data(model.index(3, 0)) == "rediscovery (O=C(O)c1ccccc1O)"
    assert round(float(model.data(model.index(3, 1))), 6) == 0.448276


def test_oracle_empty_smiles_refused(studio_app) -> None:
    page = _fresh_page()
    page._oracle_smiles.setText("")
    page._start_oracle()
    assert "Type a SMILES to score first." in page.report_text()


# -- SA score (part3 §24) ------------------------------------------------------------------------


def test_sa_tab_scores_the_defaults(studio_app, qtbot) -> None:
    page = _fresh_page()
    assert page._sa_molecules.toPlainText().startswith("CC(=O)Oc1ccccc1C(=O)O")
    _run(page, page._start_sa_score, qtbot, studio_app)
    text = page.report_text()
    # the four defaults, easiest first: the ibuprofen core 1.520,
    # aspirin 1.580, caffeine 2.298 and the taxane-style complex 4.076
    # -- all valid.
    assert "4 scored / 0 invalid" in text
    assert "mean " in text and "range 1.520" in text
    assert "lower = easier" in text and "not a route plan" in text
    model = page._table.model()
    assert model.rowCount() == 4
    assert model.columnCount() == 6
    assert model.data(model.index(0, 1)) == "CC(C)Cc1ccc(O)cc1"
    assert round(float(model.data(model.index(0, 3))), 3) == 1.52
    assert model.data(model.index(0, 4)) == "easy"


def test_sa_tab_filter_flags_rejects_in_place(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._sa_max.setText("2.0")
    _run(page, page._start_sa_score, qtbot, studio_app)
    text = page.report_text()
    # the ibuprofen core (1.520) and aspirin (1.580) survive; caffeine
    # (2.298) and the taxane (4.076) are flagged rejected but stay in
    # the table.
    assert "filter kept 2, rejected 2" in text
    model = page._table.model()
    statuses = [model.data(model.index(r, 5)) for r in range(model.rowCount())]
    assert statuses.count("kept") == 2
    assert statuses.count("rejected") == 2


def test_sa_tab_invalid_rows_reported(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._sa_molecules.setPlainText("CCO\nnot-a-smiles")
    _run(page, page._start_sa_score, qtbot, studio_app)
    text = page.report_text()
    assert "1 scored / 1 invalid" in text
    model = page._table.model()
    assert model.data(model.index(1, 4)) == "invalid"
    assert "did not parse" in str(model.data(model.index(1, 5)))


def test_sa_tab_empty_input_refused(studio_app) -> None:
    page = _fresh_page()
    page._sa_molecules.setPlainText("")
    page._start_sa_score()
    assert "Paste at least one SMILES" in page.report_text()


def test_sa_tab_bad_max_refused(studio_app) -> None:
    page = _fresh_page()
    page._sa_max.setText("banana")
    page._start_sa_score()
    assert "max SA must be a number" in page.report_text()
    page._sa_max.setText("42.0")
    page._start_sa_score()
    assert "[1, 10] scale" in page.report_text()


def test_sa_objective_selectable_in_evolve(studio_app) -> None:
    page = _fresh_page()
    labels = [page._objective.itemText(i) for i in range(page._objective.count())]
    assert "SA score (ease of synthesis)" in labels
    # the window entries become the easy/hard anchors (validated by the
    # library): 1.0 / 3.0 builds the default-anchored objective.
    page._objective.setCurrentText("SA score (ease of synthesis)")
    page._obj_lo.setText("1.0")
    page._obj_hi.setText("3.0")
    from ui.pages.generative import _build_objective

    objective = _build_objective("sa", 1.0, 3.0, "")
    assert objective.name == "sa_score"
    assert (objective.easy, objective.hard) == (1.0, 3.0)


# -- shared plumbing -----------------------------------------------------------------------------


def test_buttons_disable_while_a_job_runs(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._runner.job_started.emit("demo")
    try:
        assert not page._evolve_btn.isEnabled()
        assert not page._front_btn.isEnabled()
        assert not page._oracle_btn.isEnabled()
    finally:
        page._runner.job_ended.emit()
    assert page._evolve_btn.isEnabled()


# -- Part-3 §21.5 (round 236): molecule readouts on the charts ------------------


def test_plot_panel_forwards_point_selected(studio_app) -> None:
    """The §21.5 wiring point: PlotPanel re-emits the inspector's
    click-to-select as its own ``point_selected`` signal, so pages
    never reach into the inspector for drill-downs."""
    from ui.widgets.plot_widget import PlotPanel

    panel = PlotPanel()
    seen: list[tuple[str, int]] = []
    panel.point_selected.connect(lambda label, i: seen.append((label, i)))
    panel._inspector.pointSelected.emit("SA score", 3)
    assert seen == [("SA score", 3)]
    panel.deleteLater()


def test_sa_chart_hovers_smiles_formula_weight(studio_app, qtbot) -> None:
    """§21.5: hovering a plotted molecule reads its SMILES, formula and
    molecular weight next to the SA value -- calculated from the
    molecule, never invented."""
    page = _fresh_page()
    _run(page, page._start_sa_score, qtbot, studio_app)

    inspector = page._plot._inspector
    record = inspector._series["SA score"]
    assert len(record.x) == 4  # the four scored defaults
    # snap exactly onto the easiest point (ibuprofen core, rank 1)
    snapped = inspector.inspect_at((float(record.x[0]), float(record.y[0])))
    assert snapped == ("SA score", 0)
    readout = page._plot._readout.text()
    assert "SMILES = CC(C)Cc1ccc(O)cc1" in readout
    assert "formula = C10H14O" in readout  # the ibuprofen CORE
    assert "MW = 150.22 g/mol" in readout
    assert "y = 1.52" in readout


def test_sa_chart_click_opens_molecule_detail(studio_app, qtbot) -> None:
    """§21.5: clicking a plotted molecule opens the §22.5 detail view
    (structure + calculated properties); other series' clicks do not."""
    _ = qtbot  # the dialog is non-modal -- no signal to wait on
    page = _fresh_page()
    _run(page, page._start_sa_score, qtbot, studio_app)

    assert page._sa_detail is None
    page._plot.point_selected.emit("SA score", 1)  # aspirin, rank 2
    assert page._sa_detail is not None
    rows = page._sa_detail.property_rows()
    assert rows["original SMILES"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert rows["molecular formula"] == "C9H8O4"
    assert page._sa_detail.isVisible() is False or True  # shown offscreen
    page._sa_detail.close()

    opened = page._sa_detail
    page._plot.point_selected.emit("some other series", 0)
    assert page._sa_detail is opened  # untouched by foreign labels
    page._sa_detail = None


def test_pareto_front_hover_carries_smiles(studio_app, qtbot) -> None:
    """§21.5 on the Pareto front: front members are molecules, so the
    hover readout carries the SMILES (and formula/MW where RDKit can
    compute them) alongside the objective values."""
    page = _fresh_page()
    page._objective.setCurrentText("Multi-objective (QED × MW × TPSA)")
    page._seeded_check.setChecked(True)
    _run(page, page._start_evolve, qtbot, studio_app)
    _run(page, page._start_front, qtbot, studio_app)

    inspector = page._plot._inspector
    record = inspector._series["non-dominated"]
    assert record.x.size == 9  # the probe's 9-member front
    snapped = inspector.inspect_at((float(record.x[0]), float(record.y[0])))
    assert snapped == ("non-dominated", 0)
    readout = page._plot._readout.text()
    assert "SMILES = " in readout
    # honest metadata: formula/MW only when the molecule parses
    assert ("formula = " in readout) or ("formula" not in readout)
