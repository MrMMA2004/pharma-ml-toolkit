import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PKPDPage } from "@/features/pkpd/PKPDPage";
import { ApiError } from "@/services/api/client";
import type {
  ColumnValues,
  DatasetView,
} from "@/services/datasets/client";

/**
 * PK-1 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 8) — the registered-
 * dataset bridge over the PK/PD Import stage's paste-only lane.
 *
 * The page-level user flow under test, exactly the roadmap's
 * acceptance clauses:
 *
 *   the bridge      — selecting a dataset and its time/conc column
 *                     pair pulls both columns through the SAME
 *                     column() lane every other workspace rides and
 *                     populates the series view (row-count + the
 *                     t/C ranges asserted);
 *   the honest 422  — a backend 422 (a non-numeric column) surfaces
 *                     verbatim, never a silent failure;
 *   the paste lane  — unchanged: the free-form textarea still
 *                     imports verbatim (the regression clause);
 *   the null guard  — a null cell (json_safe's NaN → null) is a
 *                     row-labeled error, never Number(null) === 0
 *                     fabricating a concentration;
 *   the bound       — a truncated pull names its 500-row bound
 *                     (§17.3: the bounded view never masquerades);
 *   the alignment   — a length disagreement between the two columns
 *                     is the lane's own honest error.
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

const pkpdMocks = vi.hoisted(() => ({ catalog: vi.fn() }));
vi.mock("@/services/pkpd/client", async (importOriginal) => {
  // the real module keeps its types + validators (the page narrows
  // through the REAL parse helpers); only the mounted catalog call is
  // networked and mocked
  const actual = await importOriginal<
    typeof import("@/services/pkpd/client")
  >();
  return {
    ...actual,
    pkpdApi: { ...actual.pkpdApi, catalog: pkpdMocks.catalog },
  };
});

/** The registered PK profile the Datasets workflow would hold. */
const PK_DATASET: DatasetView = {
  dataset_id: "ds-pk-profile",
  name: "pk_profile.csv",
  source_filename: "pk_profile.csv",
  size_bytes: 2048,
  shape: [5, 3],
  columns: ["time", "conc", "dose"],
  target: null,
  task: null,
  n_features: 2,
  feature_names: ["time", "conc"],
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

const TIMES = [0, 0.25, 0.5, 1, 2];
const CONCS = [8.0, 6.9, 5.8, 4.4, 3.1];

/** Select the dataset, then its column pair, then click the import. */
async function driveBridgeImport() {
  // the registry effect resolves async — wait for the option before
  // selecting it (selectOptions refuses values not yet in the list)
  await waitFor(() => {
    expect(
      screen.getByRole("option", { name: /pk_profile\.csv · 5 rows · 3 columns/ }),
    ).toBeInTheDocument();
  });
  await userEvent.selectOptions(
    screen.getByLabelText("the registered dataset"),
    "ds-pk-profile",
  );
  await waitFor(() => {
    expect(screen.getByLabelText("the times column")).toBeInTheDocument();
  });
  await userEvent.selectOptions(
    screen.getByLabelText("the times column"),
    "time",
  );
  await userEvent.selectOptions(
    screen.getByLabelText("the concentration column"),
    "conc",
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Import from dataset" }),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  pkpdMocks.catalog.mockResolvedValue({
    groups: [
      {
        group: "nca",
        routes: ["/api/v1/pkpd/nca"],
        backend: "nca_summary",
      },
    ],
  });
  datasetsMocks.list.mockResolvedValue({ datasets: [PK_DATASET] });
  datasetsMocks.view.mockResolvedValue(PK_DATASET);
});

describe("PKPDPage stage 1 — the registered dataset bridge (PK-1)", () => {
  it("populates the series view from the dataset's own time/conc columns — row-count and t/C ranges asserted", async () => {
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string) => {
        if (column === "time") {
          return columnPayload("time", TIMES);
        }
        return columnPayload("conc", CONCS);
      },
    );

    render(<PKPDPage />);
    await driveBridgeImport();

    // the roadmap's own acceptance line: 5 points, t 0–2, C 3.1000–8
    expect(
      await screen.findByText(
        "5 points imported · t 0–2 · C 3.1000–8",
      ),
    ).toBeInTheDocument();
    // both columns rode the SAME bridge — the paired calls, verbatim
    expect(datasetsMocks.column).toHaveBeenCalledWith(
      "ds-pk-profile",
      "time",
    );
    expect(datasetsMocks.column).toHaveBeenCalledWith(
      "ds-pk-profile",
      "conc",
    );
    // the paired lines also fill the textarea (the one source of
    // truth stays visible and editable — the paste lane's own form;
    // the placeholder query normalizes newlines to spaces)
    const textarea = screen.getByPlaceholderText(
      /time conc 0 8\.0 0\.25 6\.9/,
    );
    expect(textarea).toHaveValue("0 8\n0.25 6.9\n0.5 5.8\n1 4.4\n2 3.1");
  });

  it("surfaces the backend's 422 on a non-numeric column verbatim — never a silent failure", async () => {
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string) => {
        if (column === "time") {
          return columnPayload("time", TIMES);
        }
        throw new ApiError(422, {
          detail: "the concentration column carries non-numeric cells",
        });
      },
    );

    render(<PKPDPage />);
    await driveBridgeImport();

    expect(
      await screen.findByText(
        /the concentration column carries non-numeric cells/,
      ),
    ).toBeInTheDocument();
    // no fabricated series on the failed pull
    expect(screen.queryByText(/points imported/)).toBeNull();
  });

  it("keeps the paste lane unchanged — the free-form textarea still imports verbatim (regression)", async () => {
    render(<PKPDPage />);

    const textarea = screen.getByPlaceholderText(
      /time conc 0 8\.0 0\.25 6\.9/,
    );
    await userEvent.type(textarea, "time conc\n0 5\n1 2.5");
    await userEvent.click(
      screen.getByRole("button", { name: "Import series" }),
    );

    expect(
      await screen.findByText("2 points imported · t 0–1 · C 2.5000–5"),
    ).toBeInTheDocument();

    // the paste parser's own labeled error lane still surfaces
    await userEvent.clear(textarea);
    await userEvent.type(textarea, "0 x");
    await userEvent.click(
      screen.getByRole("button", { name: "Import series" }),
    );
    expect(
      await screen.findByText(/is not a pair of finite numbers/),
    ).toBeInTheDocument();
  });

  it("treats a null cell as a row-labeled honest error — never Number(null) === 0", async () => {
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string) => {
        if (column === "time") {
          return columnPayload("time", TIMES);
        }
        return columnPayload("conc", [8.0, 6.9, null, 4.4, 3.1]);
      },
    );

    render(<PKPDPage />);
    await driveBridgeImport();

    expect(
      await screen.findByText(
        /row 3 of column "conc": a null value \(NaN\/Inf in the frame\) — never silently zero/,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/points imported/)).toBeNull();
  });

  it("names the bound honestly on a truncated pull — the bounded view never masquerades as the whole column", async () => {
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string) => {
        if (column === "time") {
          return columnPayload("time", TIMES, {
            n_rows: 640,
            truncated: true,
          });
        }
        return columnPayload("conc", CONCS, {
          n_rows: 640,
          truncated: true,
        });
      },
    );

    render(<PKPDPage />);
    await driveBridgeImport();

    expect(
      await screen.findByText(
        "5 points imported · t 0–2 · C 3.1000–8",
      ),
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        /the column bridge is bounded at 500 rows .* the dataset's 640 rows/,
      ),
    ).toBeInTheDocument();
  });

  it("refuses row-misaligned columns — a length disagreement is the lane's own honest error", async () => {
    datasetsMocks.column.mockImplementation(
      async (_datasetId: string, column: string) => {
        if (column === "time") {
          return columnPayload("time", TIMES);
        }
        return columnPayload("conc", CONCS.slice(0, 4));
      },
    );

    render(<PKPDPage />);
    await driveBridgeImport();

    expect(
      await screen.findByText(
        /the columns disagree in length \(time: 5 rows vs conc: 4 rows\)/,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/points imported/)).toBeNull();
  });

  it("renders the honest empty state when the registry holds no datasets", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [] });

    render(<PKPDPage />);

    expect(
      await screen.findByText(
        /No datasets registered yet — import one in the Datasets workspace\./,
      ),
    ).toBeInTheDocument();
    // the bridge button stays disarmed without a selection
    expect(
      screen.getByRole("button", { name: "Import from dataset" }),
    ).toBeDisabled();
  });
});
