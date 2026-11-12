"""Safe-ish reviewed Vina launch using a disposable synthetic engine and inputs."""
import json
import os
import sys

import pytest

from pharmaml.docking.campaign_launch import build_vina_launch_plan, launch_vina_campaign
from pharmaml.exceptions import DataValidationError

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX synthetic shebang engine")


def setup_campaign(tmp_path):
    engine = tmp_path / "vina"
    engine.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
args = sys.argv[1:]
ligand = pathlib.Path(args[args.index('--ligand') + 1])
if 'FAIL' in ligand.read_text():
    print('synthetic failure', file=sys.stderr)
    sys.exit(3)
out = pathlib.Path(args[args.index('--out') + 1])
out.write_text('MODEL 1\\nREMARK VINA RESULT: -1.0 0 0\\nENDMDL\\n')
print('synthetic engine output')
''')
    engine.chmod(0o755)
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "a.pdbqt").write_text("ligand A")
    (ligands / "b.pdbqt").write_text("ligand B")
    config = tmp_path / "vina.conf"
    config.write_text("receptor = r.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\n"
                      "size_x = 10\nsize_y = 10\nsize_z = 10\nexhaustiveness = 1\ncpu = 1\nseed = 42\n")
    output = tmp_path / "new-output"
    return config, ligands, output, engine


def plan(parts):
    config, ligands, output, engine = parts
    return build_vina_launch_plan(config, ligands, output, vina_bin=str(engine))


def test_review_then_launch_preserves_inputs_and_saves_results(tmp_path):
    parts = setup_campaign(tmp_path)
    config, ligands, output, _engine = parts
    source = {p: p.read_bytes() for p in [config, tmp_path / "r.pdbqt", *ligands.glob("*.pdbqt")]}
    reviewed = plan(parts)
    assert not output.exists() and len(reviewed["ligands"]) == 2
    progress = []
    manifest = launch_vina_campaign(reviewed, on_progress=lambda done, total: progress.append((done, total)))
    assert manifest["status"] == "completed" and len(manifest["results"]) == 2
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert all((output / row["pose"]).is_file() and (output / row["log"]).is_file()
               for row in manifest["results"])
    assert json.loads((output / "campaign.json").read_text())["status"] == "completed"
    assert source == {path: path.read_bytes() for path in source}
    with pytest.raises(DataValidationError, match="new, non-existing"):
        launch_vina_campaign(reviewed)


def test_changed_input_refused_before_creating_output(tmp_path):
    parts = setup_campaign(tmp_path)
    reviewed = plan(parts)
    (parts[1] / "a.pdbqt").write_text("modified")
    with pytest.raises(DataValidationError, match="changed since review"):
        launch_vina_campaign(reviewed)
    assert not parts[2].exists()


def test_cancel_between_ligands_preserves_completed_result_without_resume(tmp_path):
    parts = setup_campaign(tmp_path)
    reviewed = plan(parts)
    result = launch_vina_campaign(reviewed, on_progress=lambda done, _total: done < 1)
    assert result["status"] == "cancelled" and len(result["results"]) == 1
    assert (parts[2] / result["results"][0]["pose"]).is_file()
    with pytest.raises(DataValidationError, match="new, non-existing"):
        launch_vina_campaign(reviewed)


def test_failure_stops_before_next_ligand(tmp_path):
    parts = setup_campaign(tmp_path)
    (parts[1] / "a.pdbqt").write_text("FAIL")
    result = launch_vina_campaign(plan(parts))
    assert result["status"] == "failed" and len(result["results"]) == 1
    assert not result["results"][0]["ok"]
    assert json.loads((parts[2] / "campaign.json").read_text())["status"] == "failed"


def test_unsupported_directives_and_existing_output_refused(tmp_path):
    parts = setup_campaign(tmp_path)
    config, _, output, _ = parts
    config.write_text(config.read_text() + "out = unwanted.pdbqt\n")
    with pytest.raises(DataValidationError, match="does not support"):
        plan(parts)
    config.write_text(config.read_text().replace("out = unwanted.pdbqt\n", ""))
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(DataValidationError, match="new, non-existing"):
        plan(parts)
    assert (output / "keep.txt").read_text() == "keep"


def test_snapshot_change_is_detected_before_any_engine_call(tmp_path, monkeypatch):
    import pharmaml.docking.campaign_launch as launch

    parts = setup_campaign(tmp_path)
    reviewed = plan(parts)
    original = launch._copy_verified
    invoked = []

    def mutate(source, dest, expected):
        if source.name == "a.pdbqt":
            source.write_text("mutated after check")
        return original(source, dest, expected)

    monkeypatch.setattr(launch, "_copy_verified", mutate)
    monkeypatch.setattr(launch, "run_vina_batch", lambda *a, **k: invoked.append(True))
    with pytest.raises(DataValidationError, match="changed since review"):
        launch_vina_campaign(reviewed)
    assert not invoked and parts[2].is_dir()
    assert json.loads((parts[2] / "campaign.json").read_text())["status"] == "preparing"


def test_config_options_require_valid_integers(tmp_path):
    parts = setup_campaign(tmp_path)
    path = parts[0]
    path.write_text(path.read_text().replace("cpu = 1", "cpu = -1"))
    with pytest.raises(DataValidationError, match="positive integer"):
        plan(parts)


def test_engine_changed_between_ligands_aborts_without_second_result(tmp_path):
    parts = setup_campaign(tmp_path)
    reviewed = plan(parts)

    def progress(done, _total):
        if done == 1:
            parts[3].write_text(parts[3].read_text() + "\n# changed\n")
        return True

    with pytest.raises(DataValidationError, match="engine changed"):
        launch_vina_campaign(reviewed, on_progress=progress)
    manifest = json.loads((parts[2] / "campaign.json").read_text())
    assert manifest["status"] == "running" and len(manifest["results"]) == 1
    assert not (parts[2] / "results" / "ligand_000002_out.pdbqt").exists()


def test_new_ligand_invalidates_review(tmp_path):
    parts = setup_campaign(tmp_path)
    reviewed = plan(parts)
    (parts[1] / "c.pdbqt").write_text("another ligand")
    with pytest.raises(DataValidationError, match="changed since review"):
        launch_vina_campaign(reviewed)
    assert not parts[2].exists()


def test_same_stem_in_distinct_directories_gets_unique_results(tmp_path):
    parts = setup_campaign(tmp_path)
    folder = parts[1] / "nested"
    folder.mkdir()
    (folder / "a.pdbqt").write_text("different ligand")
    result = launch_vina_campaign(plan(parts))
    assert result["status"] == "completed"
    assert len({row["pose"] for row in result["results"]}) == 3


def test_symlinked_ligand_is_rejected(tmp_path):
    parts = setup_campaign(tmp_path)
    (parts[1] / "link.pdbqt").symlink_to(parts[1] / "a.pdbqt")
    with pytest.raises(DataValidationError, match="symlinked ligand"):
        plan(parts)
