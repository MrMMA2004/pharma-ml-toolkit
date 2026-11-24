/**
 * part16 §12 (Phase 9) — the Grid box 3D view's own contract.
 *
 * jsdom has no WebGL, so the REAL component renders its honest
 * fallback lane — which is exactly the surface worth asserting:
 *
 *   the empty state      — no receptor atoms arrived → the honest
 *                          "never an imagined protein" card with the
 *                          numeric config still readable;
 *   the fallback lane    — the WebGL-unavailable card is stated, the
 *                          readouts/toggles/notes still render;
 *   the readouts         — center/size in Å, the volume, the in-box
 *                          censuses, the unit note;
 *   the warnings         — the malformed box raises the error chip;
 *                          a box missing the protein raises the
 *                          empty-box chip; a sane box is silent;
 *   the toggles          — protein / grid box / ligand / highlight,
 *                          the ligand toggle honestly disabled while
 *                          no ligand arrived;
 *   the pinned notes     — the reference-frame and spacing notes
 *                          verbatim (the drift guard pins them too);
 *   the diagnostics      — the parse diagnostics surface verbatim.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { GridBoxViewer } from "@/components/docking/GridBoxViewer";
import type { Molecule3DAtom } from "@/components/molecular3d/Molecule3D";

const RECEPTOR: readonly Molecule3DAtom[] = [
  { element: "C", x: 0, y: 0, z: 0, label: "CA ALA A 1" },
  { element: "N", x: 1.5, y: 0, z: 0, label: "N ALA A 1" },
  { element: "O", x: 3.2, y: 1.1, z: 0.4, label: "O GLY A 2" },
  { element: "C", x: 500, y: 500, z: 500, label: "FE far" },
];

const SANE_CONFIG = {
  centerX: 1,
  centerY: 0.5,
  centerZ: 0.2,
  sizeX: 22.5,
  sizeY: 22.5,
  sizeZ: 22.5,
} as const;


describe("GridBoxViewer — the empty state", () => {
  it("renders the honest no-receptor card, never an imagined protein", () => {
    render(
      <GridBoxViewer
        receptorAtoms={[]}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    expect(screen.getByText("No receptor atoms arrived yet.")).toBeInTheDocument();
    expect(
      screen.getByText(/renders only structures that actually arrived/),
    ).toBeInTheDocument();
    // the numeric config stays readable in the empty state
    expect(screen.getByText(/center \(1\.00, 0\.50, 0\.20\) Å/)).toBeInTheDocument();
    expect(screen.getByText(/size 22\.50 × 22\.50 × 22\.50 Å/)).toBeInTheDocument();
  });

  it("still pins the reference-frame + spacing notes in the empty state", () => {
    render(
      <GridBoxViewer
        receptorAtoms={[]}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    expect(screen.getAllByText(/never shifts, rescales, or rotates/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/not a DockRequest field/).length).toBeGreaterThan(0);
  });
});

describe("GridBoxViewer — the fallback + readout lane (jsdom: no WebGL)", () => {
  it("renders the WebGL-unavailable fallback as a stated card, not a blank canvas", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    expect(
      screen.getByText("WebGL unavailable — the 3D view is disabled."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/stated rather than a blank canvas/),
    ).toBeInTheDocument();
  });

  it("renders the readouts: center, size, volume, the in-box censuses, units", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    expect(screen.getByText(/center \(1\.00, 0\.50, 0\.20\) Å/)).toBeInTheDocument();
    expect(screen.getByText(/size 22\.50 × 22\.50 × 22\.50 Å/)).toBeInTheDocument();
    // 22.5³ = 11,390.625 → the readout rounds to 11,391
    expect(screen.getByText(/volume 11,390\.625 Å³/)).toBeInTheDocument();
    // the three near-origin atoms sit inside; the far FE atom does not
    expect(screen.getByText("receptor atoms in box: 3 / 4")).toBeInTheDocument();
    expect(screen.getByText(/All coordinates and dimensions are Ångström/)).toBeInTheDocument();
  });

  it("renders the four visibility toggles, the ligand one honestly disabled", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    const protein = screen.getByRole("checkbox", { name: "Protein" });
    const box = screen.getByRole("checkbox", { name: "Grid box" });
    const ligand = screen.getByRole("checkbox", { name: "Ligand (none arrived)" });
    const highlight = screen.getByRole("checkbox", { name: "Binding-site highlight" });
    expect(protein).toBeChecked();
    expect(box).toBeChecked();
    expect(ligand).toBeDisabled();
    expect(highlight).toBeChecked();
  });

  it("toggles flip without touching the readouts (the read-only viewer rule)", async () => {
    const user = userEvent.setup();
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    const protein = screen.getByRole("checkbox", { name: "Protein" });
    await user.click(protein);
    expect(protein).not.toBeChecked();
    // the numeric truth is untouched — the viewer cannot drift it
    expect(screen.getByText(/center \(1\.00, 0\.50, 0\.20\) Å/)).toBeInTheDocument();
    expect(screen.getByText("receptor atoms in box: 3 / 4")).toBeInTheDocument();
  });

  it("is silent (no chips) for a sane box over its receptor", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    expect(screen.queryByText(/must be positive/)).toBeNull();
    expect(screen.queryByText(/No receptor atoms fall inside/)).toBeNull();
  });

  it("raises the non-positive-size error chip for a malformed box", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={{ ...SANE_CONFIG, sizeX: 0 }}
      />,
    );
    expect(screen.getByText(/Every box dimension must be positive/)).toBeInTheDocument();
  });

  it("raises the empty-box warning chip when the box misses the protein", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={null}
        config={{ centerX: 400, centerY: 400, centerZ: 400, sizeX: 5, sizeY: 5, sizeZ: 5 }}
      />,
    );
    expect(screen.getByText(/No receptor atoms fall inside the box/)).toBeInTheDocument();
  });

  it("surfaces the parse diagnostics verbatim when they arrived", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics="2 malformed records skipped (counted, never silent)"
        ligand={null}
        config={SANE_CONFIG}
      />,
    );
    expect(
      screen.getByText(/receptor parse: 2 malformed records skipped/),
    ).toBeInTheDocument();
  });

  it("enables the ligand toggle once a ligand arrived and shows its census", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={{
          atoms: [
            { element: "C", x: 1, y: 0.5, z: 0.2, label: null },
            { element: "O", x: 2, y: 1, z: 0.4, label: null },
          ],
          bonds: [{ a: 0, b: 1 }],
          diagnostics: null,
        }}
        config={SANE_CONFIG}
      />,
    );
    const ligand = screen.getByRole("checkbox", { name: "Ligand" });
    expect(ligand).toBeEnabled();
    expect(screen.getByText("ligand atoms in box: 2 / 2")).toBeInTheDocument();
  });

  it("raises the ligand-outside warning when the reference ligand misses", () => {
    render(
      <GridBoxViewer
        receptorAtoms={RECEPTOR}
        receptorDiagnostics={null}
        ligand={{
          atoms: [{ element: "C", x: 900, y: 900, z: 900, label: null }],
          bonds: [],
          diagnostics: null,
        }}
        config={SANE_CONFIG}
      />,
    );
    expect(
      screen.getByText(/No ligand atoms fall inside the box/),
    ).toBeInTheDocument();
  });
});
