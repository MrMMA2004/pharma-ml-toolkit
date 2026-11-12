import pytest

from pharmaml.docking.campaign_preview import preview_vina_campaign
from pharmaml.exceptions import DataValidationError, DockingError


def fixture(tmp_path):
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "first.pdbqt").write_text("first")
    config = tmp_path / "vina.conf"
    config.write_text("receptor = receptor.pdbqt\ncenter_x = 0\ncenter_y = 1\ncenter_z = 2\n"
                      "size_x = 20\nsize_y = 21\nsize_z = 22\nexhaustiveness = 8\n")
    return config, ligands, tmp_path / "out"


def test_read_only_preview_and_engine_unavailable(tmp_path, monkeypatch):
    config, ligands, out = fixture(tmp_path)

    def missing(*args, **kwargs):
        raise DockingError("no engine")

    monkeypatch.setattr("pharmaml.docking.campaign_preview.engine_argv", missing)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = preview_vina_campaign(config, ligands, out)
    assert result["engine"]["status"] == "unavailable" and result["executed"] is False
    assert result["ligands"] == 1 and result["center_angstrom"] == [0, 1, 2]
    assert result["size_angstrom"] == [20, 21, 22] and not out.exists()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert any("not input/config/engine provenance" in text for text in result["warnings"])


def test_existing_outputs_are_only_census_not_provenance(tmp_path, monkeypatch):
    config, ligands, out = fixture(tmp_path)
    out.mkdir()
    (out / "first_out.pdbqt").write_text("pose")
    (out / "first.log").write_text("log")
    monkeypatch.setattr("pharmaml.docking.campaign_preview.engine_argv", lambda *a: ["/bin/sh"])
    result = preview_vina_campaign(config, ligands, out)
    assert result["matching_existing_pairs"] == 1 and result["remaining_if_skip_existing"] == 0
    assert result["engine"]["status"] == "launch prefix resolved; not executed"
    assert any("overwritten" in text for text in result["warnings"])


@pytest.mark.parametrize("extra", ["center_x = 10\n", "invalid config\n", "size_x = 2\n"])
def test_ambiguous_config_refused(tmp_path, extra):
    config, ligands, out = fixture(tmp_path)
    config.write_text(config.read_text() + extra)
    with pytest.raises(DataValidationError):
        preview_vina_campaign(config, ligands, out)


def test_bad_box_and_overlapping_dirs_and_receptor_refused(tmp_path):
    config, ligands, out = fixture(tmp_path)
    with pytest.raises(DataValidationError, match="must not overlap"):
        preview_vina_campaign(config, ligands, ligands / "out")
    config.write_text(config.read_text().replace("size_x = 20", "size_x = -1"))
    with pytest.raises(DataValidationError, match="positive"):
        preview_vina_campaign(config, ligands, out)
    config.write_text(config.read_text().replace("size_x = -1", "size_x = 20")
                      .replace("receptor.pdbqt", "ligands/first.pdbqt"))
    with pytest.raises(DataValidationError, match="receptor is inside"):
        preview_vina_campaign(config, ligands, out)


def test_empty_ligand_refused(tmp_path):
    config, ligands, out = fixture(tmp_path)
    (ligands / "empty.pdbqt").write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        preview_vina_campaign(config, ligands, out)
