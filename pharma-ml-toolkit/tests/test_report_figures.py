"""R27 — §17D figure embedding (HTMLReport.add_image + run reports).

Pins: an existing PNG referenced by a run record is embedded as a
base64 <img> page (self-contained HTML); a recorded path whose file
is missing stays an honest text line with the reason; a non-PNG file
is refused with the reason; the Plots section's text inventory still
leads (the embedding follows it, never replaces it).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from pharmaml.studio.reports import build_run_report
from pharmaml.viz.report import HTMLReport


@pytest.fixture()
def real_png(tmp_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 4])
    fig.savefig(tmp_path / "curve.png")
    plt.close(fig)
    return tmp_path / "curve.png"


def test_add_image_embeds_the_files_own_bytes(real_png, tmp_path) -> None:
    report = HTMLReport("t")
    report.add_image(real_png, caption="the curve")
    out = report.save(tmp_path / "r.html")
    html = out.read_text()
    expected = base64.b64encode(real_png.read_bytes()).decode()
    assert f"data:image/png;base64,{expected}" in html
    assert "<figcaption>the curve</figcaption>" in html
    # it rides the same anchored figure page add_figure builds
    assert "<figure class='page' id='pg-1'" in html


def test_add_image_refuses_a_missing_file(tmp_path) -> None:
    from pharmaml.exceptions import DataValidationError

    report = HTMLReport("t")
    with pytest.raises(DataValidationError, match="unreadable"):
        report.add_image(tmp_path / "nope.png")


def test_add_image_refuses_a_non_png(tmp_path) -> None:
    from pharmaml.exceptions import DataValidationError

    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not a png at all")
    report = HTMLReport("t")
    with pytest.raises(DataValidationError, match="not a PNG"):
        report.add_image(fake)


def test_run_report_embeds_recorded_figures(real_png, tmp_path) -> None:
    record = {
        "label": "demo run",
        "family": "qsar",
        "summary": {
            "figures": [str(real_png), str(tmp_path / "gone.png")],
        },
    }
    out = build_run_report(record).save(tmp_path / "report.html")
    html = out.read_text()
    # the real file is embedded
    expected = base64.b64encode(real_png.read_bytes()).decode()
    assert f"data:image/png;base64,{expected}" in html
    assert "exported figure: curve.png" in html
    # the missing file stays an honest text line, not a broken img
    assert "recorded figure file missing: " in html
    assert "gone.png" in html


def test_run_report_without_figures_unchanged(tmp_path) -> None:
    """A record with no figures must render exactly as before."""
    record = {"label": "bare", "family": "", "summary": {}}
    html = build_run_report(record).render()
    assert "no figures recorded for this run" in html
    assert "<img" not in html
