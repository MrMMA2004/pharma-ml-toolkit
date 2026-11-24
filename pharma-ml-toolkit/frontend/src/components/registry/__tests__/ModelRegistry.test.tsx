/**
 * part16 §15.3 (Phase 12) — the Model registry card's own contract.
 *
 * The registry surface as a component-level user flow:
 *
 *   the joined view   — the catalog section (every entry, the
 *                       availability badge, the honest `requires`
 *                       line on the unavailable rows) renders above
 *                       the trained registry (the version column, the
 *                       metrics, the actions), both over the one
 *                       refresh;
 *   the filters       — the catalog task filter and the
 *                       show-unavailable toggle filter the one full
 *                       listing client-side;
 *   the detail        — a row click fetches the trained view on
 *                       demand and expands the registry record,
 *                       caching per model_id;
 *   the export lane   — the export button posts the artifact and
 *                       rides the digest-verified download;
 *   the deletion lane — the two-click confirm evicts the row (the
 *                       status line, the row leaving the table) and
 *                       a rejection surfaces through the error
 *                       banner with an honest refresh;
 *   the failure lane  — a rejected listing surfaces through the
 *                       error banner, never a fabricated table.
 *
 * The services are mocked at the module boundary only; the component,
 * its state machine, and the DOM are the real ones.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mlMocks = vi.hoisted(() => ({
  catalog: vi.fn(),
  trained: vi.fn(),
  trainedView: vi.fn(),
  evictTrained: vi.fn(),
  exportTrained: vi.fn(),
}));
vi.mock("@/services/ml/client", () => ({
  mlApi: {
    catalog: mlMocks.catalog,
    trained: mlMocks.trained,
    trainedView: mlMocks.trainedView,
    evictTrained: mlMocks.evictTrained,
    exportTrained: mlMocks.exportTrained,
  },
}));
const downloadMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/artifacts/client", () => ({
  artifactsApi: { download: downloadMock },
}));

import { ModelRegistry } from "@/components/registry/ModelRegistry";
import type { TrainedModelView } from "@/services/ml/client";

const CATALOG = [
  {
    name: "logistic_regression",
    task: "classification",
    available: true,
    requires: null,
    notes: "the linear baseline",
  },
  {
    name: "ridge",
    task: "regression",
    available: true,
    requires: null,
    notes: "",
  },
  {
    name: "xgboost",
    task: "both",
    available: false,
    requires: "xgboost",
    notes: "the boosted lane",
  },
];

// the backend store's own contract: the listing is OLDEST-FIRST
// (sorted by created_at ascending); the card re-renders newest first
const TRAINED: TrainedModelView[] = [
  {
    model_id: "mdl-20250601-0001",
    name: "logistic_regression",
    task: "classification",
    dataset_id: "ds-20250601-0001",
    target: "activity",
    features: ["molwt", "logp", "tpsa"],
    n_train_rows: 96,
    best_params: { C: 1.0 },
    cv_scores: [0.9, 0.89],
    cv_mean: 0.895,
    cv_std: 0.02,
    fit_seconds: 0.19,
    created_at: "2025-06-01T10:00:03+00:00",
  },
  {
    model_id: "mdl-20250601-0002",
    name: "logistic_regression",
    task: "classification",
    dataset_id: "ds-20250601-0001",
    target: "activity",
    features: ["molwt", "logp", "tpsa"],
    n_train_rows: 96,
    best_params: { C: 0.5 },
    cv_scores: [0.91, 0.92],
    cv_mean: 0.915,
    cv_std: 0.01,
    fit_seconds: 0.21,
    created_at: "2025-06-01T10:00:04+00:00",
  },
];

const EXPORT_ARTIFACT = {
  artifact_id: "art-20250601-0001",
  kind: "model",
  filename: "logistic_regression-model.joblib",
  content_type: "application/octet-stream",
  size_bytes: 1837,
  sha256: "0f0f0f0f0f0f0f0f",
  created_at: "2025-06-01T10:01:00+00:00",
  meta: {},
  download_url: "/api/v1/artifacts/art-20250601-0001/download",
};

beforeEach(() => {
  mlMocks.catalog.mockReset();
  mlMocks.trained.mockReset();
  mlMocks.trainedView.mockReset();
  mlMocks.evictTrained.mockReset();
  mlMocks.exportTrained.mockReset();
  downloadMock.mockReset();
  vi.stubGlobal(
    "URL",
    Object.assign(URL, {
      createObjectURL: vi.fn(() => "blob:registry-test"),
      revokeObjectURL: vi.fn(),
    }),
  );
});

function primeListing() {
  mlMocks.catalog.mockResolvedValue({ models: [...CATALOG] });
  mlMocks.trained.mockResolvedValue({ models: [...TRAINED] });
}

describe("ModelRegistry (part16 §15.3)", () => {
  it("renders the joined registry — the catalog and the trained table", async () => {
    primeListing();

    render(<ModelRegistry />);

    await waitFor(() => {
      expect(screen.getByLabelText("Trained models")).toBeTruthy();
    });
    // the catalog section: every entry, the honest requires line
    const catalogTable = screen.getByLabelText("Model registry catalog");
    expect(within(catalogTable).getAllByText("logistic_regression")).toBeTruthy();
    expect(within(catalogTable).getByText("ridge")).toBeTruthy();
    expect(within(catalogTable).getByText("xgboost")).toBeTruthy();
    expect(within(catalogTable).getAllByText(/requires xgboost/)).toBeTruthy();
    expect(within(catalogTable).getAllByText("available").length).toBeGreaterThanOrEqual(2);

    // the trained section: the newest first, the derived versions
    const trainedTable = screen.getByLabelText("Trained models");
    expect(within(trainedTable).getAllByText(/v1 of 2/)).toBeTruthy();
    expect(within(trainedTable).getAllByText(/v2 of 2/)).toBeTruthy();
    expect(
      within(trainedTable).getByText("2025-06-01T10:00:04+00:00"),
    ).toBeTruthy();
    // the store bound is stated where it acts
    expect(screen.getByText("2 / 16")).toBeTruthy();
  });

  it("renders the honest empty state when no model trained yet", async () => {
    mlMocks.catalog.mockResolvedValue({ models: [...CATALOG] });
    mlMocks.trained.mockResolvedValue({ models: [] });

    render(<ModelRegistry />);

    await waitFor(() => {
      expect(
        screen.getByText(/No trained models in this session yet/),
      ).toBeTruthy();
    });
  });

  it("filters the catalog client-side — the task filter keeps the dual-task entries", async () => {
    primeListing();

    render(<ModelRegistry />);
    await waitFor(() => {
      expect(screen.getByText("ridge")).toBeTruthy();
    });

    await userEvent.click(screen.getByRole("button", { name: "Regression" }));
    const catalogTable = screen.getByLabelText("Model registry catalog");
    expect(within(catalogTable).queryByText("logistic_regression")).toBeNull();
    // the dual-task entry rides every task filter (the backend's own rule)
    expect(within(catalogTable).getByText("xgboost")).toBeTruthy();
  });

  it("hides the unavailable entries when the availability filter is armed", async () => {
    primeListing();

    render(<ModelRegistry />);
    await waitFor(() => {
      expect(screen.getByText("xgboost")).toBeTruthy();
    });

    // "show unavailable" checked by default (availableOnly=false)
    await userEvent.click(
      screen.getByLabelText("include unavailable catalog entries"),
    );
    const catalogTable = screen.getByLabelText("Model registry catalog");
    await waitFor(() => {
      expect(within(catalogTable).queryByText("xgboost")).toBeNull();
    });
    expect(within(catalogTable).getByText("ridge")).toBeTruthy();
  });

  it("expands the registry record on demand — one fetch per model_id, cached", async () => {
    primeListing();
    mlMocks.trainedView.mockResolvedValue(TRAINED[0]);

    render(<ModelRegistry />);
    await waitFor(() => {
      expect(screen.getByText("2 / 16")).toBeTruthy();
    });

    const trainedTable = screen.getByLabelText("Trained models");
    const row = within(trainedTable).getByLabelText(
      "trained model mdl-20250601-0001",
    );
    await userEvent.click(row);

    await waitFor(() => {
      expect(mlMocks.trainedView).toHaveBeenCalledWith("mdl-20250601-0001");
    });
    // the expanded record renders the identity + the best_params block
    await waitFor(() => {
      expect(screen.getByText("mdl-20250601-0001")).toBeTruthy();
    });
    expect(screen.getByText(/Training configuration/)).toBeTruthy();

    // collapse then re-expand: the record is cached, never refetched
    await userEvent.click(row);
    await userEvent.click(row);
    expect(mlMocks.trainedView).toHaveBeenCalledTimes(1);
  });

  it("exports through the digest-verified download lane and states the status", async () => {
    primeListing();
    mlMocks.exportTrained.mockResolvedValue(EXPORT_ARTIFACT);
    downloadMock.mockResolvedValue({
      blob: new Blob(["joblib-bytes"]),
      filename: "logistic_regression-model.joblib",
      sha256: "0f0f0f0f0f0f0f0f",
      actual_sha256: "0f0f0f0f0f0f0f0f",
      digest_verified: true,
    });

    render(<ModelRegistry />);
    await waitFor(() => {
      expect(screen.getByText("2 / 16")).toBeTruthy();
    });

    // the newest row rides first — its export button is the first match
    const exportButtons = screen.getAllByRole("button", {
      name: "export logistic_regression as a joblib artifact",
    });
    const exportButton = exportButtons[0];
    if (exportButton === undefined) {
      throw new Error("the export button did not render");
    }
    await userEvent.click(exportButton);

    await waitFor(() => {
      expect(mlMocks.exportTrained).toHaveBeenCalledWith(
        "mdl-20250601-0002",
      );
    });
    expect(downloadMock).toHaveBeenCalledWith("art-20250601-0001");
    await waitFor(() => {
      expect(
        screen.getByText(/exported logistic_regression-model\.joblib/),
      ).toBeTruthy();
    });
    expect(screen.getByText(/digest verified/)).toBeTruthy();
  });

  it("evicts on the two-click confirm — the row leaves, the status states the remainder", async () => {
    primeListing();
    mlMocks.evictTrained.mockResolvedValue({
      evicted: "mdl-20250601-0002",
      remaining: 1,
    });

    render(<ModelRegistry />);
    await waitFor(() => {
      expect(screen.getByText("2 / 16")).toBeTruthy();
    });

    const deleteButtons = screen.getAllByRole("button", {
      name: "delete logistic_regression from the session registry",
    });
    const deleteButton = deleteButtons[0];
    if (deleteButton === undefined) {
      throw new Error("the delete button did not render");
    }
    await userEvent.click(deleteButton);
    expect(mlMocks.evictTrained).not.toHaveBeenCalled();
    // the armed confirm — the same button re-clicked executes
    await userEvent.click(deleteButton);
    await waitFor(() => {
      expect(mlMocks.evictTrained).toHaveBeenCalledWith("mdl-20250601-0002");
    });
    await waitFor(() => {
      expect(screen.getByText(/evicted logistic_regression/)).toBeTruthy();
      expect(screen.getByText(/1 remaining/)).toBeTruthy();
      expect(screen.getByText("1 / 16")).toBeTruthy();
    });
    const trainedTable = screen.getByLabelText("Trained models");
    expect(
      within(trainedTable).queryByLabelText(
        "trained model mdl-20250601-0002",
      ),
    ).toBeNull();
  });

  it("surfaces a rejected eviction through the error banner and refreshes honestly", async () => {
    primeListing();
    mlMocks.evictTrained.mockRejectedValue(new Error("404 Unknown model_id."));

    render(<ModelRegistry />);
    await waitFor(() => {
      expect(screen.getByText("2 / 16")).toBeTruthy();
    });

    const deleteButtons = screen.getAllByRole("button", {
      name: "delete logistic_regression from the session registry",
    });
    const deleteButton = deleteButtons[0];
    if (deleteButton === undefined) {
      throw new Error("the delete button did not render");
    }
    await userEvent.click(deleteButton);
    await userEvent.click(deleteButton);

    await waitFor(() => {
      expect(
        screen.getByText(/404 Unknown model_id\./),
      ).toBeTruthy();
    });
    // the refresh re-listed the registry honestly (the row still there)
    expect(mlMocks.trained.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("2 / 16")).toBeTruthy();
  });

  it("surfaces a rejected listing through the error banner, never a fabricated table", async () => {
    mlMocks.catalog.mockRejectedValue(new Error("backend not booted"));
    mlMocks.trained.mockRejectedValue(new Error("backend not booted"));

    render(<ModelRegistry />);

    await waitFor(() => {
      expect(screen.getAllByText(/backend not booted/).length).toBeGreaterThanOrEqual(2);
    });
    expect(screen.queryByLabelText("Trained models")).toBeNull();
    expect(screen.queryByLabelText("Model registry catalog")).toBeNull();
  });
});
