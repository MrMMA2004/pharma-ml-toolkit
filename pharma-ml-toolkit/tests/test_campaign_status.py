"""Read-only, conservative inspection of review-first Vina campaigns."""
import json
import os
import sys

import pytest

from pharmaml.docking.campaign_launch import build_vina_launch_plan, launch_vina_campaign
from pharmaml.docking.campaign_status import inspect_vina_campaign
from pharmaml.exceptions import DataValidationError

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX synthetic engine fixture")


def create_campaign(tmp_path, *, stop_after=None):
    engine = tmp_path / "vina"
    engine.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
args = sys.argv[1:]
pathlib.Path(args[args.index('--out') + 1]).write_text('toy pose')
print('toy log')
''')
    engine.chmod(0o755)
    (tmp_path / "r.pdbqt").write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "a.pdbqt").write_text("ligand A")
    (ligands / "b.pdbqt").write_text("ligand B")
    config = tmp_path / "vina.conf"
    config.write_text("receptor = r.pdbqt\ncenter_x = 0\ncenter_y = 1\ncenter_z = 2\n"
                      "size_x = 10\nsize_y = 11\nsize_z = 12\nseed = 42\n")
    output = tmp_path / "output"
    plan = build_vina_launch_plan(config, ligands, output, vina_bin=str(engine))
    callback = None if stop_after is None else lambda done, total: done < stop_after
    launch_vina_campaign(plan, on_progress=callback)
    return output


def test_completed_checks_no_side_effects_and_opt_in_originals(tmp_path):
    output = create_campaign(tmp_path)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    result = inspect_vina_campaign(output)
    assert result["recorded_status"] == "completed"
    assert result["progress"] == {"total": 2, "recorded_successes": 2, "recorded_failures": 0,
                                  "not_recorded": 0, "success_output_pairs_matching": 2}
    assert all(value == "match" for value in result["snapshots"].values())
    assert result["originals"]["status"] == "not checked"
    result = inspect_vina_campaign(output, check_originals=True)
    assert result["originals"]["ligand_census"]["status"] == "match"
    assert result["originals"]["config"] == "match"
    assert all(item["status"] == "match" for item in result["originals"]["ligands"])
    assert all(item["status"] == "match" for item in result["originals"]["engine_prefix"])
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert "No resume" in result["notice"]


def test_cancel_and_unrecorded_partial_files_not_promoted_to_success(tmp_path):
    output = create_campaign(tmp_path, stop_after=1)
    (output / "results/ligand_000002_out.pdbqt").write_text("orphaned pose")
    (output / "results/.vina-batch-interrupted").mkdir()
    result = inspect_vina_campaign(output)
    assert result["recorded_status"] == "cancelled"
    assert result["progress"]["recorded_successes"] == 1 and result["progress"]["not_recorded"] == 1
    assert result["ligands"][1]["recorded_result"] == "not_recorded"
    assert result["unrecorded_result_files"] == ["results/ligand_000002_out.pdbqt"]
    assert result["unrecorded_result_directories"] == ["results/.vina-batch-interrupted"]


def test_changed_missing_and_new_inputs_and_output(tmp_path):
    output = create_campaign(tmp_path)
    (tmp_path / "ligands/a.pdbqt").write_text("changed")
    (tmp_path / "ligands/b.pdbqt").unlink()
    (tmp_path / "ligands/new.pdbqt").write_text("new")
    (tmp_path / "vina").write_text("modified")
    (output / "inputs/receptor.pdbqt").write_text("modified")
    (output / "inputs/vina.conf").write_text("changed")
    (output / "results/ligand_000001_out.pdbqt").write_text("tampered pose")
    (output / "results/ligand_000002.log").unlink()
    result = inspect_vina_campaign(output, check_originals=True)
    assert result["progress"]["success_output_pairs_matching"] == 0
    assert result["ligands"][0]["pose"] == "changed" and result["ligands"][1]["log"] == "missing"
    assert result["snapshots"]["receptor"] == result["snapshots"]["generated_config"] == "changed"
    assert [row["status"] for row in result["originals"]["ligands"]] == ["changed", "missing"]
    assert result["originals"]["ligand_census"]["added"] == ["new.pdbqt"]
    assert result["originals"]["ligand_census"]["missing"] == ["b.pdbqt"]
    assert result["originals"]["engine_prefix"][0]["status"] == "changed"


def test_newline_translation_for_windows_config(tmp_path):
    output = create_campaign(tmp_path)
    config = output / "inputs/vina.conf"
    config.write_bytes(config.read_bytes().replace(b"\n", b"\r\n"))
    assert inspect_vina_campaign(output)["snapshots"]["generated_config"] == "match"


def test_manifest_corrupt_or_contradictory_refused(tmp_path):
    output = create_campaign(tmp_path)
    manifest = output / "campaign.json"
    original = json.loads(manifest.read_text())
    original["status"] = "preparing"
    manifest.write_text(json.dumps(original))
    with pytest.raises(DataValidationError, match="contradicts"):
        inspect_vina_campaign(output)
    manifest.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(DataValidationError, match="duplicate JSON key"):
        inspect_vina_campaign(output)
    manifest.write_text("{broken")
    with pytest.raises(DataValidationError, match="cannot read"):
        inspect_vina_campaign(output)


def test_forged_path_traversal_and_symlink_refused(tmp_path):
    output = create_campaign(tmp_path)
    manifest = output / "campaign.json"
    data = json.loads(manifest.read_text())
    data["results"][0]["pose"] = "../../secret"
    manifest.write_text(json.dumps(data))
    with pytest.raises(DataValidationError, match="invalid recorded output paths"):
        inspect_vina_campaign(output)
    data["results"][0]["pose"] = "results/ligand_000001_out.pdbqt"
    data["plan"]["ligands"][0]["relative_path"] = "../secret.pdbqt"
    manifest.write_text(json.dumps(data))
    with pytest.raises(DataValidationError, match="invalid recorded ligand list"):
        inspect_vina_campaign(output)
    data["plan"]["ligands"][0]["relative_path"] = "a.pdbqt"
    manifest.write_text(json.dumps(data))
    pose = output / "results/ligand_000001_out.pdbqt"
    pose.unlink()
    pose.symlink_to(tmp_path / "secret")
    with pytest.raises(DataValidationError, match="symlink"):
        inspect_vina_campaign(output)


def test_originals_not_read_unless_explicit(tmp_path, monkeypatch):
    from pharmaml.docking import campaign_status

    output = create_campaign(tmp_path)
    original = campaign_status.fingerprint_file

    def local_only(path):
        assert path.is_relative_to(output)
        return original(path)

    monkeypatch.setattr(campaign_status, "fingerprint_file", local_only)
    assert inspect_vina_campaign(output)["originals"]["status"] == "not checked"


def test_cli_status_is_read_only_and_reports_partial_run(tmp_path, capsys):
    from pharmaml.cli import main

    output = create_campaign(tmp_path, stop_after=1)
    before = (output / "campaign.json").read_bytes()
    assert main(["docking", "campaign-status", "--dir", str(output), "--check-originals"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recorded_status"] == "cancelled" and result["originals"]["status"] == "checked"
    assert (output / "campaign.json").read_bytes() == before


def test_failed_manifest_reports_failure_not_completion(tmp_path):
    output = create_campaign(tmp_path, stop_after=1)
    path = output / "campaign.json"
    record = json.loads(path.read_text())
    record["status"] = "failed"
    record["results"].append({"ligand": "b.pdbqt", "ok": False, "error": "synthetic failure"})
    path.write_text(json.dumps(record))
    result = inspect_vina_campaign(output)
    assert result["progress"]["recorded_failures"] == 1 and result["progress"]["recorded_successes"] == 1
    assert result["ligands"][1]["recorded_result"] == "failed"
    assert result["ligands"][1]["recorded_error"] == "synthetic failure"


def test_preparing_and_missing_snapshots_are_reported_not_success(tmp_path):
    output = create_campaign(tmp_path, stop_after=0)
    path = output / "campaign.json"
    record = json.loads(path.read_text())
    record["status"] = "preparing"
    path.write_text(json.dumps(record))
    (output / "inputs/vina.conf").unlink()
    (output / "inputs/ligands/000001/ligand_000001.pdbqt").unlink()
    result = inspect_vina_campaign(output)
    assert result["snapshots"]["generated_config"] == "missing"
    assert result["ligands"][0]["snapshot"] == "missing"
    assert result["progress"]["not_recorded"] == 2


def test_directory_symlink_in_snapshot_refused(tmp_path):
    output = create_campaign(tmp_path)
    old = output / "inputs/ligands/000001"
    old.rename(output / "inputs/ligands/real")
    old.symlink_to(output / "inputs/ligands/real", target_is_directory=True)
    with pytest.raises(DataValidationError, match="symlink"):
        inspect_vina_campaign(output)


def test_broken_original_root_not_followed(tmp_path):
    output = create_campaign(tmp_path)
    source = tmp_path / "ligands"
    source.rename(tmp_path / "old-ligands")
    source.symlink_to(tmp_path / "old-ligands", target_is_directory=True)
    result = inspect_vina_campaign(output, check_originals=True)
    assert result["originals"]["ligand_census"]["status"] == "unavailable"
    assert all(item["status"] == "unavailable" for item in result["originals"]["ligands"])


def test_running_can_contain_last_failed_record_after_interruption(tmp_path):
    output = create_campaign(tmp_path, stop_after=1)
    path = output / "campaign.json"
    data = json.loads(path.read_text())
    data["status"] = "running"
    data["results"].append({"ligand": "b.pdbqt", "ok": False, "error": "synthetic error"})
    path.write_text(json.dumps(data))
    result = inspect_vina_campaign(output)
    assert result["recorded_status"] == "running" and result["progress"]["recorded_failures"] == 1


def test_windows_style_recorded_output_paths_are_understood(tmp_path):
    output = create_campaign(tmp_path)
    path = output / "campaign.json"
    record = json.loads(path.read_text())
    for row in record["results"]:
        row["pose"] = row["pose"].replace("/", "\\")
        row["log"] = row["log"].replace("/", "\\")
    path.write_text(json.dumps(record))
    assert inspect_vina_campaign(output)["progress"]["success_output_pairs_matching"] == 2
