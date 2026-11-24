/**
 * part16 §15.4 (Phase 12) — the Data Quality card's own contract.
 *
 * The §15.4 surface as a component-level user flow:
 *
 *   the pre-scan state   — the honest "not scanned yet" line, the
 *                          button disabled with no dataset selected;
 *   the scan             — the click POSTs through the service, the
 *                          severity chips render with the counts,
 *                          the findings with their pinned codes, the
 *                          executed/skipped checks, the advisory;
 *   the clean scan       — zero findings renders the honest clean
 *                          state, never a fabricated row;
 *   the failure lane     — a rejected scan surfaces the error
 *                          banner, never a fabricated report;
 *   the no-mutation rule — the card offers NO fix/repair/apply
 *                          control anywhere (§15.4: never silently
 *                          alter user-provided scientific data).
 *
 * The quality service is mocked at the module boundary only; the
 * component, its state machine, and the DOM are the real ones.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const runMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/quality/client", () => ({
  qualityApi: { run: runMock },
}));

import { DataQualityCard } from "@/components/quality/DataQualityCard";
import type { DataQualityReport } from "@/services/quality/types";

function report(
  overrides: Partial<DataQualityReport> = {},
): DataQualityReport {
  return {
    rows: 10,
    columns: 3,
    target: "activity",
    smiles_column: "smiles",
    checks: [
      {
        check: "smiles_validation",
        label: "SMILES validation",
        status: "executed",
        detail: "over column 'smiles'",
      },
      {
        check: "molecular_structure_validation",
        label: "Molecular structure validation",
        status: "skipped",
        detail: "skipped: no SMILES column detected",
      },
    ],
    findings: [
      {
        code: "invalid_smiles",
        severity: "error",
        category: "chemistry",
        message: "1 of 10 SMILES values failed the grammar parse.",
        columns: ["smiles"],
        evidence: { invalid_count: 1 },
      },
      {
        code: "class_imbalance",
        severity: "warning",
        category: "distribution",
        message: "Target 'activity': the dominant class holds 90.0%.",
        columns: ["activity"],
        evidence: { majority_fraction: 0.9 },
      },
    ],
    counts: { error: 1, warning: 1, info: 0 },
    read_only: true,
    advisory:
      "Advisory report only -- PharmaML never modifies user-provided scientific data.",
    ...overrides,
  };
}

describe("DataQualityCard", () => {
  beforeEach(() => {
    runMock.mockReset();
  });

  it("renders the honest pre-scan state and disables without a dataset", () => {
    render(<DataQualityCard datasetId={null} />);
    expect(
      screen.getByText(/Not scanned yet/i),
    ).toBeInTheDocument();
    const button = screen.getByRole("button", {
      name: /Run the data-quality scan/i,
    });
    expect(button).toBeDisabled();
    expect(screen.getByText(/Select a dataset first/i)).toBeInTheDocument();
  });

  it("runs the scan and renders the findings with the counts", async () => {
    runMock.mockResolvedValue(report());
    render(<DataQualityCard datasetId="ds-001" target="activity" />);
    await userEvent.click(
      screen.getByRole("button", { name: /Run the data-quality scan/i }),
    );
    await waitFor(() => {
      expect(runMock).toHaveBeenCalledWith("ds-001", { target: "activity" });
    });
    expect(
      await screen.findByText(/1 of 10 SMILES values failed/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/1 error/i)).toBeInTheDocument();
    expect(screen.getByText(/1 warning/i)).toBeInTheDocument();
    expect(screen.getByText(/0 infos/i)).toBeInTheDocument();
    expect(screen.getByText(/10 rows × 3 columns/i)).toBeInTheDocument();
  });

  it("renders the executed/skipped checks and the advisory", async () => {
    runMock.mockResolvedValue(report());
    render(<DataQualityCard datasetId="ds-001" />);
    await userEvent.click(
      screen.getByRole("button", { name: /Run the data-quality scan/i }),
    );
    await screen.findByText(/1 of 10 SMILES values failed/i);
    expect(
      screen.getByText(/never modifies user-provided scientific data/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The executed §15\.4 checks \(2\)/i),
    ).toBeInTheDocument();
  });

  it("renders the honest clean state on a zero-finding scan", async () => {
    runMock.mockResolvedValue(
      report({ findings: [], counts: { error: 0, warning: 0, info: 0 } }),
    );
    render(<DataQualityCard datasetId="ds-001" />);
    await userEvent.click(
      screen.getByRole("button", { name: /Run the data-quality scan/i }),
    );
    expect(
      await screen.findByText(/No issues detected/i),
    ).toBeInTheDocument();
  });

  it("surfaces a rejected scan through the error banner", async () => {
    runMock.mockRejectedValue(new Error("network unreachable"));
    render(<DataQualityCard datasetId="ds-001" />);
    await userEvent.click(
      screen.getByRole("button", { name: /Run the data-quality scan/i }),
    );
    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("Scan failed:");
    expect(banner.textContent).toContain("network unreachable");
  });

  it("offers no mutation control anywhere (the §15.4 rule)", async () => {
    runMock.mockResolvedValue(report());
    render(<DataQualityCard datasetId="ds-001" />);
    await userEvent.click(
      screen.getByRole("button", { name: /Run the data-quality scan/i }),
    );
    await screen.findByText(/1 of 10 SMILES values failed/i);
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(1); // the scan button alone
    for (const word of ["fix", "repair", "apply", "delete", "clean"]) {
      expect(document.body.textContent ?? "").not.toMatch(
        new RegExp(`\\b${word}\\b`, "i"),
      );
    }
  });
});
