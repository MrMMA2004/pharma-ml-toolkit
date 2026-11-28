import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VisualizationPage } from "@/features/visualization/VisualizationPage";
import { ApiError } from "@/services/api/client";
import type {
  ColumnValues,
  DatasetView,
} from "@/services/datasets/client";

/**
 * VIZ-1 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 9) — the registered-
 * dataset column bridge over the Visualization stat family's
 * paste-only values lane.
 *
 * The page-level user flow under test, exactly the roadmap's
 * acceptance clauses:
 *
 *   the identical  — dataset+column selection produces the identical
 *                    stat request the paste lane does for the same
 *                    numbers (one snapshot: the bridge's pulled
 *                    values and the pasted values ride the SAME
 *                    parser into the SAME request shape);
 *   the honest 422 — a backend 422 (a non-numeric column) surfaces
 *                    verbatim, never a silent failure;
 *   the paste lane — unchanged: the free-form values textarea still
 *                    parses verbatim through its own parser (the
 *                    regression clause);
 *   the null guard— a null cell (json_safe's NaN → null) is a
 *                    row-labeled error, never Number(null) === 0
 *                    fabricating a value the frame never measured;
 *   the bound     — a truncated pull names its 500-row bound (§17.3:
 *                    the bounded view never masquerades as the whole
 *                    column);
 *   the empty     — the honest empty state when the registry holds
 *                    no datasets, the button disarmed.
 *
 * Services are mocked at the network boundary only; the page
 * component, its state machine, and the DOM are the real ones.
 */

const datasetsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  view: vi.fn(),
  column: vi.fn(),
}));

vi.mock("@/services/datasets/client", () => ({
  datasetsApi: {
    list: datasetsMocks.list,
    view: datasetsMocks.view,
    column: datasetsMocks.column,
  },
}));

const artifactsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  view: vi.fn(),
  delete: vi.fn(),
  download: vi.fn(),
}));
vi.mock("@/services/artifacts/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/artifacts/client")>();
  return {
    ...actual,
    artifactsApi: {
      ...actual.artifactsApi,
      list: artifactsMocks.list,
      view: artifactsMocks.view,
      delete: artifactsMocks.delete,
      download: artifactsMocks.download,
    },
  };
});

const vizMocks = vi.hoisted(() => ({
  catalog: vi.fn(),
  stat: vi.fn(),
  concentrationProfile: vi.fn(),
  doseAccumulation: vi.fn(),
  mdSeries: vi.fn(),
  exportTable: vi.fn(),
  report: vi.fn(),
}));
vi.mock("@/services/visualization/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/visualization/client")>();
  return {
    ...actual,
    vizApi: {
      ...actual.vizApi,
      catalog: vizMocks.catalog,
      stat: vizMocks.stat,
      concentrationProfile: vizMocks.concentrationProfile,
      doseAccumulation: vizMocks.doseAccumulation,
      mdSeries: vizMocks.mdSeries,
      exportTable: vizMocks.exportTable,
      report: vizMocks.report,
    },
  };
});

vi.mock("@/hooks/useJobTracking", () => ({
  useJob: () => ({
    snapshot: null,
    cancel: vi.fn(),
    refresh: vi.fn(),
  }),
}));

vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: vi.fn() },
}));

vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => ({
    subscribe: () => () => undefined,
  }),
}));

vi.mock("@/services/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api/client")>();
  return { ...actual, api: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } };
});

const CATALOG = {
  families: [
    {
      family: "stat",
      kinds: ["box", "histogram", "ecdf", "regression"],
      formats: ["PNG", "SVG"],
      backend: "matplotlib",
    },
  ],
  palette: [],
  export_formats: ["csv"],
  report_themes: ["auto"],
};

/** The registered dataset the Datasets workflow would hold. */
const VIZ_DATASET: DatasetView = {
  dataset_id: "ds-solubility",
  name: "solubility.csv",
  source_filename: "solubility.csv",
  size_bytes: 2048,
  shape: [8, 4],
  columns: ["smiles", "molwt", "logp", "solubility"],
  target: "solubility",
  task: "regression",
  n_features: 3,
  feature_names: ["molwt", "logp"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: [],
  summary: {},
  created_at: "2026-11-28T00:00:00Z",
};

function columnPayload(
  column: string,
  values: (string | number | null)[],
  overrides: Partial<ColumnValues> = {},
): ColumnValues {
  return {
    column,
    dtype: "float64",
    n_rows: values.length,
    returned: values.length,
    truncated: false,
    values,
    ...overrides,
  };
}

const PULLED = [2.4, 2.9, 3.3, 3.7, 4.1, 4.4, 4.9, 5.2];

function statAnswer(values: readonly number[]) {
  return {
    kind: "histogram",
    n_points: values.length,
    stats: { mean: values.reduce((a, b) => a + b, 0) / values.length },
    artifact: null,
  };
}

/** Select the dataset, then its column, then click the pull. */
async function driveBridgeColumn() {
  await waitFor(() => {
    expect(
      screen.getByRole("option", { name: /solubility\.csv · 8 rows · 4 columns/ }),
    ).toBeInTheDocument();
  });
  await userEvent.selectOptions(
    screen.getByLabelText("the registered dataset"),
    "ds-solubility",
  );
  await waitFor(() => {
    expect(screen.getByLabelText("the dataset column")).toBeInTheDocument();
  });
  await userEvent.selectOptions(
    screen.getByLabelText("the dataset column"),
    "logp",
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Pull column" }),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vizMocks.catalog.mockResolvedValue(CATALOG);
  artifactsMocks.list.mockResolvedValue([]);
  datasetsMocks.list.mockResolvedValue({ datasets: [VIZ_DATASET] });
  datasetsMocks.view.mockResolvedValue(VIZ_DATASET);
});

describe("VisualizationPage stage 3 — the registered dataset column bridge (VIZ-1)", () => {
  it("produces the identical stat request the paste lane does for the same numbers (one snapshot)", async () => {
    datasetsMocks.column.mockResolvedValue(columnPayload("logp", PULLED));
    vizMocks.stat.mockImplementation(
      async (request: { values?: readonly number[] | null }) =>
        statAnswer(request.values ?? []),
    );

    // -- entry point one: the bridge --
    render(<VisualizationPage />);
    await driveBridgeColumn();

    // the pulled values fill the values textarea verbatim (the one
    // source of truth stays visible and editable)
    await waitFor(() => {
      expect(
        screen.getByPlaceholderText(/one value per line/),
      ).toHaveValue(PULLED.map((v) => String(v)).join("\n"));
    });
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => {
      expect(vizMocks.stat).toHaveBeenCalledTimes(1);
    });
    const bridgeRequest = vizMocks.stat.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;

    // -- entry point two: the paste lane, the SAME numbers --
    const pasted = PULLED.map((v) => String(v)).join("\n");
    const values = screen.getByPlaceholderText(/one value per line/);
    await userEvent.clear(values);
    await userEvent.type(values, pasted);
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => {
      expect(vizMocks.stat).toHaveBeenCalledTimes(2);
    });
    const pasteRequest = vizMocks.stat.mock.calls[1]?.[0] as Record<
      string,
      unknown
    >;

    // one snapshot: the two requests are the same request — the same
    // values array (parsed by the same parser), the same defaults
    expect(pasteRequest).toStrictEqual(bridgeRequest);
    expect(bridgeRequest["values"]).toEqual(PULLED);
    // the column rode the SAME bounded bridge lane
    expect(datasetsMocks.column).toHaveBeenCalledWith(
      "ds-solubility",
      "logp",
    );
  });

  it("surfaces the backend's 422 on a non-numeric column verbatim — never a silent failure", async () => {
    datasetsMocks.column.mockRejectedValueOnce(
      new ApiError(422, {
        detail: "the column logp carries non-numeric cells",
      }),
    );

    render(<VisualizationPage />);
    const before = (screen.getByPlaceholderText(/one value per line/) as HTMLTextAreaElement).value;
    await driveBridgeColumn();

    expect(
      await screen.findByText(/the column logp carries non-numeric cells/),
    ).toBeInTheDocument();
    // the paste lane's own content is never overwritten on a failed pull
    expect(
      (screen.getByPlaceholderText(/one value per line/) as HTMLTextAreaElement).value,
    ).toBe(before);
    expect(vizMocks.stat).not.toHaveBeenCalled();
  });

  it("keeps the paste lane unchanged — the free-form values textarea still parses verbatim (regression)", async () => {
    render(<VisualizationPage />);

    const values = screen.getByPlaceholderText(/one value per line/);
    await userEvent.clear(values);
    await userEvent.type(values, "1.5\n2.5\n3.5\n4.5");
    await userEvent.click(
      screen.getByRole("button", { name: "Load & validate" }),
    );
    expect(
      await screen.findByText(
        /4 values · \d+ time\/conc points · the \.xvg text is pasted/,
      ),
    ).toBeInTheDocument();

    // the paste parser's own labeled error lane still surfaces
    await userEvent.clear(values);
    await userEvent.type(values, "1.5\nx");
    await userEvent.click(
      screen.getByRole("button", { name: "Load & validate" }),
    );
    expect(
      await screen.findByText(
        /the values lane entry 2: "x" is not a finite number/,
      ),
    ).toBeInTheDocument();
  });

  it("treats a null cell as a row-labeled honest error — never Number(null) === 0", async () => {
    datasetsMocks.column.mockResolvedValue(
      columnPayload("logp", [2.4, 2.9, null, 3.7, 4.1, 4.4, 4.9, 5.2]),
    );

    render(<VisualizationPage />);
    const before = (screen.getByPlaceholderText(/one value per line/) as HTMLTextAreaElement).value;
    await driveBridgeColumn();

    expect(
      await screen.findByText(
        /row 3 of column "logp": a null value \(NaN\/Inf in the frame\) — never silently zero/,
      ),
    ).toBeInTheDocument();
    expect(
      (screen.getByPlaceholderText(/one value per line/) as HTMLTextAreaElement).value,
    ).toBe(before);
    expect(vizMocks.stat).not.toHaveBeenCalled();
  });

  it("names the bound honestly on a truncated pull — the bounded view never masquerades as the whole column", async () => {
    datasetsMocks.column.mockResolvedValue(
      columnPayload("logp", PULLED, {
        n_rows: 640,
        returned: 8,
        truncated: true,
      }),
    );

    render(<VisualizationPage />);
    await driveBridgeColumn();

    expect(
      await screen.findByText(
        /the column bridge is bounded at 500 rows .* the dataset's 640 rows/,
      ),
    ).toBeInTheDocument();
    // the values textarea still carries the pulled rows
    await waitFor(() => {
      expect(
        screen.getByPlaceholderText(/one value per line/),
      ).toHaveValue(PULLED.map((v) => String(v)).join("\n"));
    });
  });

  it("renders the honest empty state when the registry holds no datasets", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [] });

    render(<VisualizationPage />);

    expect(
      await screen.findByText(
        /No datasets registered yet — import one in the Datasets workspace to plot its columns\./,
      ),
    ).toBeInTheDocument();
    // the bridge button stays disarmed without a selection
    expect(
      screen.getByRole("button", { name: "Pull column" }),
    ).toBeDisabled();
  });
});
