import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DatabasePreferences } from "@/components/system/DatabasePreferences";
import { ApiError } from "@/services/api/client";

/**
 * part17 §3.2 (Round 360, the G1 lane) — the durable user-preference
 * manager's own contract as a component-level user flow:
 *
 *   the list       — the mount fetches GET /api/v1/database/preferences
 *                    through the typed databaseApi.preferences and
 *                    renders the rows (key, JSON value, updated);
 *   the empty      — an empty store says so honestly;
 *   the write      — a strict-JSON value over a valid key fires the
 *                    typed setPreference and the refresh;
 *   the validation — an INVALID JSON value never leaves the browser
 *                    (no request fires; the honest error names the
 *                    lane); an out-of-bound key likewise;
 *   the delete     — the row action only ARMS the two-step gate;
 *                    only the confirm button fires the typed
 *                    deletePreference; the list refreshes;
 *   the 404 lane   — an already-gone key reconciles honestly;
 *   the 409 lane   — a non-durable instance renders the honest note;
 *   the split      — the card states the backend-store vs
 *                    localStorage-settings distinction.
 *
 * The database service is mocked at the module boundary only; the
 * component, its state machine, and the DOM are the real ones.
 */

const databaseMocks = vi.hoisted(() => ({
  status: vi.fn(),
  jobHistory: vi.fn(),
  backup: vi.fn(),
  restore: vi.fn(),
  preferences: vi.fn(),
  setPreference: vi.fn(),
  deletePreference: vi.fn(),
}));
vi.mock("@/services/database/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/database/client")>();
  return {
    ...actual,
    databaseApi: {
      ...actual.databaseApi,
      status: databaseMocks.status,
      jobHistory: databaseMocks.jobHistory,
      backup: databaseMocks.backup,
      restore: databaseMocks.restore,
      preferences: databaseMocks.preferences,
      setPreference: databaseMocks.setPreference,
      deletePreference: databaseMocks.deletePreference,
    },
  };
});

const ROWS = [
  {
    key: "analysis.default_units",
    value: { concentration: "mg/L" },
    updated_at: "2026-11-25T08:30:00Z",
  },
  {
    key: "report.theme",
    value: "auto",
    updated_at: "2026-11-24T16:05:00Z",
  },
];

describe("DatabasePreferences", () => {
  beforeEach(() => {
    databaseMocks.preferences.mockReset();
    databaseMocks.setPreference.mockReset();
    databaseMocks.deletePreference.mockReset();
  });

  it("loads the durable store on mount and renders the rows", async () => {
    databaseMocks.preferences.mockResolvedValue(ROWS);
    render(<DatabasePreferences />);
    await waitFor(() => {
      expect(databaseMocks.preferences).toHaveBeenCalledTimes(1);
    });
    expect(
      await screen.findByText("analysis.default_units"),
    ).toBeInTheDocument();
    expect(screen.getByText(/"concentration":"mg\/L"/)).toBeInTheDocument();
    expect(screen.getByText("report.theme")).toBeInTheDocument();
    expect(screen.getByText(/2 preferences in the durable store/)).toBeInTheDocument();
    // the honest split: the backend store vs the app's localStorage
    expect(
      screen.getByText(/settings panel&rsquo;s localStorage store|settings panel's localStorage store/i),
    ).toBeInTheDocument();
  });

  it("renders the honest empty state", async () => {
    databaseMocks.preferences.mockResolvedValue([]);
    render(<DatabasePreferences />);
    expect(
      await screen.findByText(/The durable store is empty/i),
    ).toBeInTheDocument();
  });

  it("surfaces a rejected fetch through the error banner", async () => {
    databaseMocks.preferences.mockRejectedValue(new Error("backend unreachable"));
    render(<DatabasePreferences />);
    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("backend unreachable");
  });

  it("a strict-JSON value over a valid key fires the typed write and refreshes", async () => {
    databaseMocks.preferences.mockResolvedValue([]);
    databaseMocks.setPreference.mockResolvedValue({
      key: "analysis.default_units",
      value: { concentration: "mg/L" },
      saved: true as const,
    });
    render(<DatabasePreferences />);
    await screen.findByText(/The durable store is empty/i);

    await userEvent.type(screen.getByLabelText("The preference key"), "analysis.default_units");
    await userEvent.type(
      screen.getByLabelText("The preference value (strict JSON)"),
      '{"concentration":"mg/L"}',
    );
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(databaseMocks.setPreference).toHaveBeenCalledWith(
        "analysis.default_units",
        { concentration: "mg/L" },
      );
    });
    expect(
      await screen.findByText(/Saved the preference "analysis\.default_units"\./),
    ).toBeInTheDocument();
    expect(databaseMocks.preferences).toHaveBeenCalledTimes(2);
  });

  it("an invalid JSON value never leaves the browser — no request fires", async () => {
    databaseMocks.preferences.mockResolvedValue([]);
    render(<DatabasePreferences />);
    await screen.findByText(/The durable store is empty/i);

    await userEvent.type(screen.getByLabelText("The preference key"), "bad.value");
    await userEvent.type(
      screen.getByLabelText("The preference value (strict JSON)"),
      "{not json}",
    );
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("strict JSON");
    expect(databaseMocks.setPreference).not.toHaveBeenCalled();
  });

  it("an out-of-bound key is refused client-side — no request fires", async () => {
    databaseMocks.preferences.mockResolvedValue([]);
    render(<DatabasePreferences />);
    await screen.findByText(/The durable store is empty/i);

    await userEvent.type(screen.getByLabelText("The preference value (strict JSON)"), "true");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("1–128");
    expect(databaseMocks.setPreference).not.toHaveBeenCalled();
  });

  it("the row's Delete action only ARMS the gate — no request yet", async () => {
    databaseMocks.preferences.mockResolvedValue(ROWS);
    render(<DatabasePreferences />);
    await screen.findByText("analysis.default_units");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete preference analysis.default_units" }),
    );
    expect(
      await screen.findByText(/Delete the preference “analysis\.default_units”\?/),
    ).toBeInTheDocument();
    // the honest metadata: the value + the updated stamp
    expect(screen.getByText(/value \{"concentration":"mg\/L"\}/)).toBeInTheDocument();
    expect(databaseMocks.deletePreference).not.toHaveBeenCalled();
  });

  it("Cancel disarms the gate without any request", async () => {
    databaseMocks.preferences.mockResolvedValue(ROWS);
    render(<DatabasePreferences />);
    await screen.findByText("analysis.default_units");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete preference analysis.default_units" }),
    );
    await screen.findByText(/Delete the preference “analysis\.default_units”\?/);
    await userEvent.click(screen.getByRole("button", { name: "Keep it" }));

    expect(databaseMocks.deletePreference).not.toHaveBeenCalled();
    expect(
      screen.queryByText(/Delete the preference “analysis\.default_units”\?/),
    ).toBeNull();
  });

  it("only the Confirm button fires the typed delete, refreshes, and reports it", async () => {
    databaseMocks.preferences.mockResolvedValueOnce(ROWS);
    databaseMocks.preferences.mockResolvedValue([ROWS[1]]);
    databaseMocks.deletePreference.mockResolvedValue({
      key: "analysis.default_units",
      deleted: true as const,
    });
    render(<DatabasePreferences />);
    await screen.findByText("analysis.default_units");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete preference analysis.default_units" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete preference" }));
    await waitFor(() => {
      expect(databaseMocks.deletePreference).toHaveBeenCalledWith("analysis.default_units");
    });
    expect(
      await screen.findByText(/Deleted the preference "analysis\.default_units"\./),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText("analysis.default_units")).toBeNull();
    });
  });

  it("an already-gone key (404) reconciles the list honestly", async () => {
    databaseMocks.preferences.mockResolvedValue(ROWS);
    databaseMocks.deletePreference.mockRejectedValueOnce(
      new ApiError(404, "Unknown preference: 'analysis.default_units'."),
    );
    render(<DatabasePreferences />);
    await screen.findByText("analysis.default_units");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete preference analysis.default_units" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete preference" }));
    expect(
      await screen.findByText(
        /"analysis\.default_units" was already gone \(404\) — the list refreshed\./,
      ),
    ).toBeInTheDocument();
    expect(databaseMocks.preferences).toHaveBeenCalledTimes(2);
  });

  it("a non-durable instance (409) renders the honest note", async () => {
    databaseMocks.preferences.mockRejectedValueOnce(
      new ApiError(409, "persistence is disabled for this app instance"),
    );
    render(<DatabasePreferences />);
    expect(
      await screen.findByText(/Non-durable instance — the preference store is unavailable/i),
    ).toBeInTheDocument();
  });
});
