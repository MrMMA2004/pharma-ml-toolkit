"""Review and preparation of real small receptor/ligand fixtures."""
import json
import os
from pathlib import Path

import pytest

from pharmaml.docking.campaign_preparation import build_preparation_plan, prepare_reviewed_inputs
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.provenance import fingerprint_file

pytest.importorskip("rdkit")
pytest.importorskip("meeko")

RECEPTOR = Path(__file__).parent / "data" / "1crn.pdb"


def inputs(tmp_path):
    receptor = tmp_path / "source.pdb"
    receptor.write_bytes(RECEPTOR.read_bytes())
    table = tmp_path / "library.csv"
    table.write_text("name,SMILES\nethanol,CCO\nbroken,C1CC\nbenzene,c1ccccc1\n", encoding="utf-8")
    return receptor, table, tmp_path / "prepared"


def test_real_review_prepare_and_status_accounting(tmp_path):
    receptor, table, output = inputs(tmp_path)
    original = {path: path.read_bytes() for path in (receptor, table)}
    plan = build_preparation_plan(receptor, table, output)
    assert plan["review"]["syntactically_valid"] == 2 and plan["review"]["invalid_smiles"] == 1
    assert not output.exists()
    report = prepare_reviewed_inputs(plan)
    assert report["status"] == "completed_with_failures"
    assert report["ligands"]["prepared"] == 2 and report["ligands"]["failed"] == 1
    assert (output / "ligands/failed_smiles.csv").is_file()
    assert report["receptor"]["fingerprint"] == fingerprint_file(output / "receptor.pdbqt")
    for row in report["ligands"]["rows"]:
        if row["ok"]:
            assert fingerprint_file(output / row["pdbqt"]) == row["fingerprint"]
    assert json.loads((output / "preparation.json").read_text())["status"] == "completed_with_failures"
    assert (output / "inputs/receptor_source.pdb").read_bytes() == original[receptor]
    assert (output / "inputs/ligands_source.csv").read_bytes() == original[table]
    assert {p: p.read_bytes() for p in original} == original
    with pytest.raises(DataValidationError, match="new, non-existing"):
        prepare_reviewed_inputs(plan)


def test_prepared_files_can_be_reviewed_for_vina_without_launch(tmp_path):
    from pharmaml.docking.campaign_preview import preview_vina_campaign

    receptor, table, output = inputs(tmp_path)
    result = prepare_reviewed_inputs(build_preparation_plan(receptor, table, output))
    config = output / "vina.conf"
    config.write_text("receptor = receptor.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\n"
                      "size_x = 10\nsize_y = 10\nsize_z = 10\n")
    review = preview_vina_campaign(config, output / "ligands", tmp_path / "run", vina_bin="not-an-engine")
    assert review["ligands"] == 2 and not review["executed"] and result["status"] == "completed_with_failures"
    assert not (tmp_path / "run").exists()


def test_review_detects_source_and_dependency_changes_before_writes(tmp_path, monkeypatch):
    import pharmaml.docking.campaign_preparation as preparation

    receptor, table, output = inputs(tmp_path)
    plan = build_preparation_plan(receptor, table, output)
    table.write_text(table.read_text() + "\n")
    with pytest.raises(DataValidationError, match="changed since review"):
        prepare_reviewed_inputs(plan)
    assert not output.exists()
    plan = build_preparation_plan(receptor, table, output)
    original = preparation._dependencies
    monkeypatch.setattr(preparation, "_dependencies", lambda suffix: {**original(suffix), "meeko": "changed"})
    with pytest.raises(DataValidationError, match="changed since review"):
        prepare_reviewed_inputs(plan)
    assert not output.exists()


def test_changed_during_snapshot_is_failed_without_preparation(tmp_path, monkeypatch):
    import pharmaml.docking.campaign_preparation as preparation

    receptor, table, output = inputs(tmp_path)
    plan = build_preparation_plan(receptor, table, output)
    original = preparation._snapshot
    invoked = []

    def mutate(source, dest, expected):
        if source == table:
            source.write_text(source.read_text() + "\n")
        return original(source, dest, expected)

    monkeypatch.setattr(preparation, "_snapshot", mutate)
    monkeypatch.setattr(preparation, "prepare_protein", lambda *a, **k: invoked.append(True))
    with pytest.raises(DataValidationError, match="changed since review"):
        prepare_reviewed_inputs(plan)
    assert not invoked and json.loads((output / "preparation.json").read_text())["status"] == "failed"


def test_receptor_error_recorded_and_ligands_never_started(tmp_path, monkeypatch):
    import pharmaml.docking.campaign_preparation as preparation

    receptor, table, output = inputs(tmp_path)
    plan = build_preparation_plan(receptor, table, output)
    invoked = []

    def failed(*_args, **_kwargs):
        raise DataValidationError("synthetic preparation failure")

    monkeypatch.setattr(preparation, "prepare_protein", failed)
    monkeypatch.setattr(preparation, "prepare_ligand_library", lambda *a, **k: invoked.append(True))
    with pytest.raises(DataValidationError, match="synthetic"):
        prepare_reviewed_inputs(plan)
    assert not invoked
    manifest = json.loads((output / "preparation.json").read_text())
    assert manifest["status"] == "failed" and "synthetic" in manifest["error"]


def test_review_rejects_existing_or_dangling_output(tmp_path):
    receptor, table, output = inputs(tmp_path)
    output.mkdir()
    (output / "keep").write_text("unchanged")
    with pytest.raises(DataValidationError, match="new, non-existing"):
        build_preparation_plan(receptor, table, output)
    assert (output / "keep").read_text() == "unchanged"
    link = tmp_path / "dangling"
    try:
        link.symlink_to(tmp_path / "missing", target_is_directory=True)
    except OSError:
        pytest.skip("symlink unavailable")
    with pytest.raises(DataValidationError, match="new, non-existing"):
        build_preparation_plan(receptor, table, link)


@pytest.mark.parametrize("options", [{"chain": "ABC"}, {"seed": -1}, {"seed": True},
                                      {"remove_waters": "yes"}, {"smiles_col": "missing"}])
def test_invalid_options_are_refused(tmp_path, options):
    receptor, table, output = inputs(tmp_path)
    with pytest.raises(DataValidationError):
        build_preparation_plan(receptor, table, output, **options)
    assert not output.exists()


def test_all_invalid_smiles_refused_before_preparation(tmp_path):
    receptor, table, output = inputs(tmp_path)
    table.write_text("SMILES\nC1CC\n")
    with pytest.raises(DataValidationError, match="no syntactically valid"):
        build_preparation_plan(receptor, table, output)
    assert not output.exists()


def test_every_ligand_preparation_failure_is_failed_not_complete(tmp_path):
    receptor, table, output = inputs(tmp_path)
    # Boron passes RDKit syntax but is unsupported by this AutoDock typing path.
    table.write_text("name,SMILES\nboron,OB(O)c1ccccc1\n")
    plan = build_preparation_plan(receptor, table, output)
    result = prepare_reviewed_inputs(plan)
    assert result["status"] == "failed" and result["ligands"]["prepared"] == 0
    assert result["ligands"]["failed"] == 1 and result["ligands"]["rows"][0]["error"]


def test_changed_receptor_rejected_but_plan_is_not_an_authorization_token(tmp_path):
    receptor, table, output = inputs(tmp_path)
    plan = build_preparation_plan(receptor, table, output)
    receptor.write_text(receptor.read_text() + "\n")
    with pytest.raises(DataValidationError, match="changed since review"):
        prepare_reviewed_inputs(plan)
    assert not output.exists()
    plan = build_preparation_plan(receptor, table, output)
    plan["options"]["seed"] = 17
    # A modified plan is also recomputed: the caller must not treat an edited
    # plan as an authenticated approval. The GUI invalidates edited controls.
    # Because the schema is unsigned, a caller can produce a self-consistent
    # plan by editing its options; this API is not an authorization boundary.
    result = prepare_reviewed_inputs(plan)
    assert result["plan"]["options"]["seed"] == 17


@pytest.mark.skipif(os.name == "nt", reason="POSIX synthetic shebang engine")
def test_prepared_inputs_launch_with_synthetic_engine(tmp_path):
    import os
    import sys

    from pharmaml.docking.campaign_launch import build_vina_launch_plan, launch_vina_campaign

    if os.name == "nt":
        pytest.skip("synthetic executable uses a POSIX shebang")
    receptor, table, output = inputs(tmp_path)
    report = prepare_reviewed_inputs(build_preparation_plan(receptor, table, output))
    assert report["ligands"]["prepared"] == 2
    config = output / "vina.conf"
    config.write_text("receptor = receptor.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\n"
                      "size_x = 10\nsize_y = 10\nsize_z = 10\nseed = 42\n")
    engine = tmp_path / "vina"
    engine.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
args = sys.argv[1:]
pathlib.Path(args[args.index('--out') + 1]).write_text('toy pose')
print('toy log')
''')
    engine.chmod(0o755)
    launched = launch_vina_campaign(build_vina_launch_plan(config, output / "ligands", tmp_path / "docked",
                                                            vina_bin=str(engine)))
    assert launched["status"] == "completed" and len(launched["results"]) == 2


def test_missing_dependency_is_reported_before_output(tmp_path, monkeypatch):
    from pharmaml.docking import campaign_preparation as preparation
    from pharmaml.exceptions import DependencyError

    receptor, table, output = inputs(tmp_path)
    original = preparation.importlib.util.find_spec
    monkeypatch.setattr(preparation.importlib.util, "find_spec",
                        lambda name: None if name == "meeko" else original(name))
    with pytest.raises(DependencyError, match="requires meeko"):
        build_preparation_plan(receptor, table, output)
    assert not output.exists()


def test_ligand_change_during_review_is_refused(tmp_path, monkeypatch):
    from pharmaml.docking import campaign_preparation as preparation

    receptor, table, output = inputs(tmp_path)
    original = preparation.load_ligand_csv

    def changed(path, **kwargs):
        result = original(path, **kwargs)
        table.write_text(table.read_text() + "\n")
        return result

    monkeypatch.setattr(preparation, "load_ligand_csv", changed)
    with pytest.raises(DataValidationError, match="changed during review"):
        build_preparation_plan(receptor, table, output)
    assert not output.exists()
