"""Generate a readable PK chart in all three supported export formats.

Run from the repository root after installing the reporting extra::

    python examples/chart_readability.py

Outputs are written to ``artifacts/chart-readability/`` (ignored by Git).
No downloaded data, display server, or non-deterministic fitting is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pharmaml.viz import PALETTE, figure_to_base64, new_figure, save_figure


def main() -> None:
    out_dir = Path("artifacts/chart-readability")
    out_dir.mkdir(parents=True, exist_ok=True)

    hours = np.linspace(0.0, 24.0, 240)
    fig, ax = new_figure()  # 9 x 6 inches; 120-DPI interactive display
    for index, (half_life, label) in enumerate(
        ((4.0, "fast clearance"), (8.0, "medium clearance"), (16.0, "slow clearance"))
    ):
        concentration = 100.0 * np.exp(-np.log(2.0) * hours / half_life)
        ax.plot(
            hours,
            concentration,
            color=PALETTE[index],
            linewidth=2.2,
            label=label,
        )
    ax.set(
        xlim=(0.0, 24.0),
        ylim=(0.0, 105.0),
        xlabel="Time after dose (h)",
        ylabel="Plasma concentration (mg/L)",
        title="Concentration–time profile",
    )
    ax.legend(title="Elimination", frameon=False)

    for extension in ("png", "svg", "pdf"):
        path = save_figure(fig, out_dir / f"concentration-profile.{extension}")
        print(f"wrote {path}")

    # Embed the same figure directly in a self-contained HTML report.
    uri = figure_to_base64(fig)
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>PharmaML chart example</title>
<body style="margin:2rem;background:#f4f6f8;font-family:sans-serif">
<main style="max-width:1100px;margin:auto;background:white;padding:1.5rem">
<h1>Concentration–time profile</h1>
<img style="display:block;width:100%;height:auto" src="{uri}" alt="{ax.get_title()}">
</main></body></html>
"""
    html_path = out_dir / "concentration-profile.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
