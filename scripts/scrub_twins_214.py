"""Round 214 scrub: (1) drop stream outputs carrying absolute paths from the
executed notebooks (the round-212 rule: ANY stderr stream carrying an absolute
path is a leak — ConvergenceWarning narrowly is not the rule), (2) regenerate
the 7 no-output twins WITH the inserted note cell at index 1 (the
test_note_cell_inserted pin: cells[1] markdown, 'No-output variant', twin name)."""

from __future__ import annotations

import nbformat

EXAMPLES = "/home/z/my-project/pharma-ml-toolkit/examples"

VARIANTS = (
    "complete_ml_workflow",
    "complete_docking_workflow",
    "complete_pk_workflow",
    "complete_cellline_workflow",
    "complete_generative_workflow",
    "complete_deep_learning_workflow",
    "complete_tuning_workflow",
)

NEEDLES = ("/home/z/", "/home/jake", "/home/USER", "/tmp/nbwork", "github_pat")


def scrub_notebook(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    dropped = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        keep = []
        for out in cell.get("outputs", []):
            text = ""
            if out.output_type == "stream":
                text = out.get("text", "")
            elif out.output_type == "error":
                text = "\n".join(out.get("traceback", []))
            if text and any(n in text for n in NEEDLES):
                dropped += 1
                continue
            keep.append(out)
        cell.outputs = keep
    nbformat.write(nb, path)
    return dropped


def regen_twin(name: str) -> None:
    nb = nbformat.read(f"{EXAMPLES}/{name}.ipynb", as_version=4)
    twin = nbformat.v4.new_notebook()
    twin.metadata = nb.metadata
    for cell in nb.cells:
        if cell.cell_type == "code":
            fresh = nbformat.v4.new_code_cell(cell.source)
        elif cell.cell_type == "markdown":
            fresh = nbformat.v4.new_markdown_cell(cell.source)
        else:
            fresh = nbformat.v4.new_raw_cell(cell.source)
        fresh.id = cell.id
        twin.cells.append(fresh)
    # the note rides at index 1 with the pinned id (test_note_cell_inserted +
    # the variant-note filter in test_sources_match_original)
    note = nbformat.v4.new_markdown_cell(
        "No-output variant of `{0}.ipynb`: identical cell sources, every "
        "output stripped and every execution count cleared — the diff of the "
        "two files shows exactly what one full run produces. Re-execute the "
        "parent notebook to refresh this twin.".format(name)
    )
    note.id = "variant-note"
    twin.cells.insert(1, note)
    nbformat.write(twin, f"{EXAMPLES}/{name}_no_output.ipynb")


def main() -> None:
    total = 0
    for name in VARIANTS:
        path = f"{EXAMPLES}/{name}.ipynb"
        dropped = scrub_notebook(path)
        total += dropped
        print(f"{name}: dropped {dropped} leaking stream/error output(s)")
        regen_twin(name)
        print(f"  twin regenerated with note cell")
    print(f"total dropped: {total}")


if __name__ == "__main__":
    main()
