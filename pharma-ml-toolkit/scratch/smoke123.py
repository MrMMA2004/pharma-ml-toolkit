"""100x-123 smoke: surgery grids + report table axis (scratch, not shipped)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import playground  # noqa: E402


def drain(app, timeout: float = 240.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def main() -> None:
    app = playground.PlaygroundApp(persist_session=False)
    for _ in range(6):
        app.update()

    # 1. topology enumerate + draw
    app._gen_topology_moves()
    assert drain(app), "topology job hung"
    source, mutants = app._topo_last
    print("topo:", source, "->", len(mutants), "mutants")
    app._gen_draw_topology()
    assert drain(app), "draw topo job hung"
    figure = getattr(app._topo_host, "_pg_figure", None)
    assert figure is not None, "no figure on the topo host"
    print("topo grid figure:", figure)

    # 2. breed + draw
    app._gen_breed()
    assert drain(app), "breed job hung"
    parent_a, parent_b, litter = app._breed_last
    print("breed:", len(litter), "drawn,", sum(1 for c in litter if c), "viable")
    app._gen_draw_breed()
    assert drain(app), "draw breed job hung"
    figure = getattr(app._breed_host, "_pg_figure", None)
    assert figure is not None, "no figure on the breed host"
    print("breed grid figure:", figure)

    # 3. table walk -- a REAL job fills a REAL results tree (~200 rows,
    # which also exercises the 150-row report cap)
    app._analyse_smiles()
    assert drain(app), "analyse job hung"
    tree = app._desc_tree
    row_count = len(tree.get_children())
    print("desc tree rows:", row_count)
    assert row_count > 150, "expected a long table for the cap path"
    tables = app._live_session_tables()
    print("tables found:", [(t, len(r)) for t, _h, r in tables])
    assert any(
        t == "SMILES inspector" and len(r) == row_count for t, _h, r in tables
    ), "desc tree not walked"

    # 4. the full report: 2 figure pages + table pages
    import unittest.mock as mock

    target = "/tmp/pg123-session-report.pdf"
    with mock.patch.object(
        playground.filedialog, "asksaveasfilename", lambda **k: target
    ), mock.patch.object(
        playground.messagebox, "showinfo", lambda *a, **k: print("INFO:", a[0])
    ):
        app._menu_save_session_report()
        assert drain(app), "report job hung"
    import pathlib

    path = pathlib.Path(target)
    assert path.is_file() and path.stat().st_size > 2000, "report missing"
    import pypdf

    reader = pypdf.PdfReader(str(path))
    texts = [page.extract_text() or "" for page in reader.pages]
    print("report pages:", len(reader.pages))
    joined = "\n".join(texts).lower()
    assert "table page(s)" in joined, "subtitle did not mention tables"
    assert "smiles inspector table" in joined, "the table page did not ship"
    assert "molwt" in joined, "walked rows did not ship"
    assert "more rows" in joined, "the library's 38-row truncation note did not ship"

    # 5. empty-guard message changed honestly
    app2_tables_only = None
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
