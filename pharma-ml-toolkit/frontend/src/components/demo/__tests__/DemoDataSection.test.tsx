import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DemoDataSection } from "@/components/demo/DemoDataSection";
import type { SampleView } from "@/services/samples/client";

/**
 * part16 §8 (Phase 5) — the shared Demo Data section as a component-level
 * user flow:
 *
 *   the catalog    — the workspace-filtered entries render with their
 *                    real metadata (title, file, size, kind label);
 *   preview-first  — the preview button fetches the RAW head and
 *                    renders the bounded table, then hides again;
 *   the confirm gate — "Load demo" never loads in one click: it arms
 *                    the explicit confirmation (stating what will be
 *                    imported and that nothing user-owned is replaced);
 *                    Confirm alone fires the load;
 *   per-kind payloads — the tabular result surfaces the registered
 *                    dataset_id and the host's onDatasetLoaded fires;
 *                    the series payload fires onSeriesLoaded with the
 *                    delivered arrays;
 *   the unavailable row — a missing file renders "Unavailable" and
 *                    refuses to arm.
 *
 * The samples service is mocked at the module boundary only; the
 * component, its state machine, and the DOM are the real ones.
 */

const samplesMocks = vi.hoisted(() => ({
  list: vi.fn(),
  preview: vi.fn(),
  load: vi.fn(),
}));

vi.mock("@/services/samples/client", () => ({
  samplesApi: {
    list: samplesMocks.list,
    preview: samplesMocks.preview,
    load: samplesMocks.load,
  },
}));

const catalog: SampleView[] = [
  {
    sample_id: "qsar-classification",
    title: "QSAR classification set",
    filename: "qsar_classification.csv",
    kind: "tabular",
    file_type: "csv",
    series_columns: null,
    library_columns: null,
    description: "Precomputed descriptors plus a binary activity label.",
    scientific_context: "Structure-activity screening.",
    workflows: ["datasets", "machine-learning"],
    target: "activity",
    task: "classification",
    size_bytes: 43883,
    sha256: "a".repeat(64),
    available: true,
    cleanup: "removable via the Datasets workspace",
  },
  {
    sample_id: "pk-profile",
    title: "PK concentration–time profile",
    filename: "pk_profile.csv",
    kind: "series",
    file_type: "csv",
    series_columns: { time: "time_h", value: "conc_ng_ml" },
    library_columns: null,
    description: "A time / concentration series.",
    scientific_context: "A single-subject profile.",
    workflows: ["pkpd"],
    target: null,
    task: null,
    size_bytes: 676,
    sha256: "b".repeat(64),
    available: true,
    cleanup: "nothing registered",
  },
  {
    sample_id: "lost-file",
    title: "A sample whose file is missing",
    filename: "missing.csv",
    kind: "tabular",
    file_type: "csv",
    series_columns: null,
    library_columns: null,
    description: "The runtime guard marks this unavailable.",
    scientific_context: "n/a",
    workflows: ["datasets"],
    target: null,
    task: null,
    size_bytes: null,
    sha256: null,
    available: false,
    cleanup: "n/a",
  },
];

beforeEach(() => {
  samplesMocks.list.mockReset();
  samplesMocks.preview.mockReset();
  samplesMocks.load.mockReset();
  samplesMocks.list.mockResolvedValue({
    count: catalog.length,
    workflow: "datasets",
    samples: catalog,
  });
});

describe("DemoDataSection (part16 Phase 5)", () => {
  it("renders the workspace catalog with real metadata", async () => {
    render(<DemoDataSection workflow="datasets" />);

    expect(samplesMocks.list).toHaveBeenCalledWith("datasets");
    const title = await screen.findByText("QSAR classification set");
    expect(title).toBeVisible();
    const row = title.closest("li");
    expect(row).not.toBeNull();
    expect(
      within(row as HTMLElement).getByText(/qsar_classification\.csv · 42\.9 KiB/),
    ).toBeVisible();
    expect(
      within(row as HTMLElement).getByText(/table → dataset/),
    ).toBeVisible();
    expect(
      within(row as HTMLElement).getByText(/target activity/),
    ).toBeVisible();
  });

  it("previews the raw head before any load, and never registers", async () => {
    samplesMocks.preview.mockResolvedValue({
      sample_id: "qsar-classification",
      kind: "tabular",
      columns: ["descr_00", "descr_01", "activity"],
      dtypes: { descr_00: "float64" },
      n_rows: 420,
      returned: 2,
      truncated: true,
      rows: [
        [1.5, 2.5, 1],
        [2.5, 3.5, 0],
      ],
    });
    render(<DemoDataSection workflow="datasets" />);

    const title = await screen.findByText("QSAR classification set");
    const row = title.closest("li") as HTMLElement;
    await userEvent.click(within(row).getByText("Preview"));

    expect(samplesMocks.preview).toHaveBeenCalledWith(
      "qsar-classification",
      8,
    );
    expect(await screen.findByText(/420 rows × 3 columns/)).toBeVisible();
    expect(screen.getByRole("columnheader", { name: "descr_00" })).toBeVisible();
    // a preview never loads anything
    expect(samplesMocks.load).not.toHaveBeenCalled();
  });

  it("gates the load behind an explicit confirm, then delivers the dataset", async () => {
    samplesMocks.load.mockResolvedValue({
      mode: "sync",
      kind: "tabular",
      sample_id: "qsar-classification",
      dataset: { dataset_id: "ds-demo123", shape: [420, 10] },
    });
    const onDatasetLoaded = vi.fn();
    render(
      <DemoDataSection workflow="datasets" onDatasetLoaded={onDatasetLoaded} />,
    );

    await userEvent.click(
      within(
        (await screen.findByText("QSAR classification set")).closest("li") as HTMLElement,
      ).getByText("Load demo"),
    );
    // the confirm panel states the no-replace semantics
    expect(
      screen.getByText(/NEW dataset record through the backend's own import engine/),
    ).toBeVisible();
    // nothing loaded yet — Confirm alone fires
    expect(samplesMocks.load).not.toHaveBeenCalled();

    await userEvent.click(screen.getByText("Confirm load"));
    await waitFor(() => {
      expect(samplesMocks.load).toHaveBeenCalledWith("qsar-classification");
    });
    expect(await screen.findByText(/registered dataset ds-demo123/)).toBeVisible();
    expect(onDatasetLoaded).toHaveBeenCalledTimes(1);
  });

  it("delivers the series payload to the host workflow", async () => {
    samplesMocks.list.mockResolvedValue({
      count: 1,
      workflow: "pkpd",
      samples: [catalog[1]],
    });
    samplesMocks.load.mockResolvedValue({
      mode: "sync",
      kind: "series",
      sample_id: "pk-profile",
      columns: { time: "time_h", value: "conc_ng_ml" },
      times: [0, 0.5, 1],
      conc: [0, 2.6, 3.6],
      n_points: 3,
    });
    const onSeriesLoaded = vi.fn();
    render(<DemoDataSection workflow="pkpd" onSeriesLoaded={onSeriesLoaded} />);

    await userEvent.click(
      within(
        (await screen.findByText("PK concentration–time profile")).closest("li") as HTMLElement,
      ).getByText("Load demo"),
    );
    await userEvent.click(screen.getByText("Confirm load"));
    await waitFor(() => {
      expect(onSeriesLoaded).toHaveBeenCalledTimes(1);
    });
    expect(onSeriesLoaded.mock.calls[0]?.[0]?.times).toEqual([0, 0.5, 1]);
    expect(
      await screen.findByText(/delivered 3 time\/concentration points/),
    ).toBeVisible();
  });

  it("renders the unavailable row honestly and refuses to arm it", async () => {
    render(<DemoDataSection workflow="datasets" />);
    const section = await screen.findByRole("list");
    const unavailable = within(section).getByText("Unavailable");
    expect(unavailable.closest("button")).toBeDisabled();
    expect(
      screen.getByText("A sample whose file is missing"),
    ).toBeVisible();
  });
});
