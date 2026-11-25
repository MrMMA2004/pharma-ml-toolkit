import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JobCenterCard } from "@/components/system/JobCenterCard";
import { ApiError } from "@/services/api/client";
import type { JobView } from "@/services/jobs/types";

/**
 * part17 §3.2 (Round 360, the G1 lane) — the global Job Center's own
 * contract as a component-level user flow:
 *
 *   the ledger      — the mount fetches GET /api/v1/jobs through the
 *                     typed jobsApi.list and renders every row (the
 *                     state chip, the operation, the workspace, the
 *                     progress, the id);
 *   the empty state — a session with no jobs says so honestly;
 *   the result lane — only a COMPLETED row offers Result; the click
 *                     fetches through jobsApi.result and the payload
 *                     renders as bounded JSON;
 *   the result 409  — the backend's own honest "not completed"
 *                     answer surfaces verbatim;
 *   the cancel gate — the row action only ARMS the two-step
 *                     confirmation; only the confirm button fires the
 *                     typed jobsApi.cancel; the notice acknowledges;
 *   the cancel 404  — an already-gone job reconciles the ledger;
 *   the failure    — a rejected list surfaces the error banner;
 *   the honest split
 *                  — the card states the live-session vs
 *                     persisted-history distinction.
 *
 * The jobs service is mocked at the module boundary only; the
 * component, its state machine, and the DOM are the real ones.
 */

const jobsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  result: vi.fn(),
  cancel: vi.fn(),
}));
vi.mock("@/services/jobs/client", () => ({
  jobsApi: {
    list: jobsMocks.list,
    result: jobsMocks.result,
    cancel: jobsMocks.cancel,
  },
}));

const job = (overrides: Partial<JobView> = {}): JobView => ({
  job_id: "job-100",
  operation: "ml.train",
  workspace: "machine-learning",
  state: "RUNNING",
  created_at: "2026-11-25T09:00:00Z",
  started_at: "2026-11-25T09:00:01Z",
  completed_at: null,
  progress: 42,
  stage: "fitting",
  configuration: {},
  cancellation: "none",
  error: null,
  ...overrides,
});

describe("JobCenterCard", () => {
  beforeEach(() => {
    jobsMocks.list.mockReset();
    jobsMocks.result.mockReset();
    jobsMocks.cancel.mockReset();
  });

  it("loads the ledger on mount and renders the cross-workspace rows", async () => {
    jobsMocks.list.mockResolvedValue({
      jobs: [
        job(),
        job({
          job_id: "job-101",
          operation: "docking.dock",
          workspace: "docking",
          state: "COMPLETED",
          progress: 100,
          completed_at: "2026-11-25T09:10:00Z",
        }),
      ],
    });
    render(<JobCenterCard />);
    await waitFor(() => {
      expect(jobsMocks.list).toHaveBeenCalledTimes(1);
    });
    expect(await screen.findByText("ml.train")).toBeInTheDocument();
    expect(screen.getByText("docking.dock")).toBeInTheDocument();
    // the two workspaces side by side — the cross-workspace ledger
    expect(screen.getByText(/machine-learning · progress 42%/)).toBeInTheDocument();
    expect(screen.getByText(/docking · progress 100%/)).toBeInTheDocument();
    // the honest split: the live session vs the persisted history
    expect(
      screen.getByText(/persisted history that survives restarts/i),
    ).toBeInTheDocument();
  });

  it("renders the honest empty state for a session with no jobs", async () => {
    jobsMocks.list.mockResolvedValue({ jobs: [] });
    render(<JobCenterCard />);
    expect(
      await screen.findByText(/No jobs this session yet/i),
    ).toBeInTheDocument();
  });

  it("surfaces a rejected ledger fetch through the error banner", async () => {
    jobsMocks.list.mockRejectedValue(new Error("backend unreachable"));
    render(<JobCenterCard />);
    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("backend unreachable");
  });

  it("only a COMPLETED row offers Result — and the payload renders as JSON", async () => {
    jobsMocks.list.mockResolvedValue({
      jobs: [
        job({
          job_id: "job-101",
          state: "COMPLETED",
          progress: 100,
          completed_at: "2026-11-25T09:10:00Z",
        }),
        job({ job_id: "job-100" }),
      ],
    });
    render(<JobCenterCard />);
    expect(
      await screen.findByRole("button", { name: "Fetch the result of ml.train" }),
    ).toBeInTheDocument();
    // the RUNNING row never offers the result lane (the backend
    // answers 409 for anything not COMPLETED — the button is not
    // offered at all)
    expect(
      screen.queryByRole("button", { name: /Fetch the result of/ }),
    ).toHaveAttribute("aria-label", "Fetch the result of ml.train");

    jobsMocks.result.mockResolvedValue({
      job_id: "job-101",
      result: { cv_mean: 0.87, n_features: 4 },
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Fetch the result of ml.train" }),
    );
    await waitFor(() => {
      expect(jobsMocks.result).toHaveBeenCalledWith("job-101");
    });
    expect(await screen.findByText(/"cv_mean": 0\.87/)).toBeInTheDocument();
  });

  it("the result lane surfaces the backend's own 409 verbatim", async () => {
    jobsMocks.list.mockResolvedValue({
      jobs: [
        job({
          job_id: "job-101",
          state: "COMPLETED",
          progress: 100,
          completed_at: "2026-11-25T09:10:00Z",
        }),
      ],
    });
    jobsMocks.result.mockRejectedValueOnce(
      new ApiError(409, "Job state is RUNNING; not completed."),
    );
    render(<JobCenterCard />);
    await screen.findByText("ml.train");
    await userEvent.click(
      screen.getByRole("button", { name: "Fetch the result of ml.train" }),
    );
    expect(await screen.findByText(/409/)).toBeInTheDocument();
    expect(await screen.findByText(/not completed/)).toBeInTheDocument();
  });

  it("the row's Cancel action only ARMS the gate — no request yet", async () => {
    jobsMocks.list.mockResolvedValue({ jobs: [job()] });
    render(<JobCenterCard />);
    await screen.findByText("ml.train");

    await userEvent.click(screen.getByRole("button", { name: "Cancel ml.train" }));
    expect(await screen.findByText(/Cancel ml\.train \(RUNNING\)\?/)).toBeInTheDocument();
    expect(
      screen.getByText(/cooperative cancellation is requested/i),
    ).toBeInTheDocument();
    expect(jobsMocks.cancel).not.toHaveBeenCalled();
  });

  it("only the Confirm button fires the typed cancel and acknowledges it", async () => {
    jobsMocks.list.mockResolvedValue({ jobs: [job()] });
    jobsMocks.cancel.mockResolvedValue({
      ...job(),
      state: "CANCELLATION_REQUESTED",
      cancel_acknowledged: true,
    });
    render(<JobCenterCard />);
    await screen.findByText("ml.train");

    await userEvent.click(screen.getByRole("button", { name: "Cancel ml.train" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm cancel" }));
    await waitFor(() => {
      expect(jobsMocks.cancel).toHaveBeenCalledTimes(1);
      expect(jobsMocks.cancel).toHaveBeenCalledWith("job-100");
    });
    expect(
      await screen.findByText(/Cancellation acknowledged for ml\.train/),
    ).toBeInTheDocument();
  });

  it("an already-gone job (404) reconciles the ledger honestly", async () => {
    jobsMocks.list.mockResolvedValue({ jobs: [job()] });
    jobsMocks.cancel.mockRejectedValueOnce(new ApiError(404, "Unknown job_id."));
    render(<JobCenterCard />);
    await screen.findByText("ml.train");

    await userEvent.click(screen.getByRole("button", { name: "Cancel ml.train" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm cancel" }));
    expect(
      await screen.findByText(/already gone \(404\) — ledger refreshed\./),
    ).toBeInTheDocument();
    expect(jobsMocks.list).toHaveBeenCalledTimes(2);
  });

  it("a terminal state never offers cancellation", async () => {
    jobsMocks.list.mockResolvedValue({
      jobs: [
        job({
          job_id: "job-102",
          state: "FAILED",
          error: "RuntimeError: the seeded failure",
        }),
      ],
    });
    render(<JobCenterCard />);
    expect(await screen.findByText("ml.train")).toBeInTheDocument();
    // the FAILED row renders its error verbatim
    expect(screen.getByText(/RuntimeError: the seeded failure/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel ml.train" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Fetch the result of ml.train" })).toBeNull();
  });
});
