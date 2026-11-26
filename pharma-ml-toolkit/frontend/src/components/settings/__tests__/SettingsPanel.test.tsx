import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { SettingsEntry } from "@/components/settings/SettingsEntry";
import { getSettingsStore } from "@/services/settings/store";
import { SETTINGS_STORAGE_KEY } from "@/services/settings/store";

/**
 * part16 §10 — the compact settings surface contract.
 *
 * §10's own rules pinned here: a COMPACT popover (never a full-screen
 * page), the §10 inventory of controls (theme / density / chart
 * preferences incl. export defaults / scientific execution defaults /
 * notification preferences / resource-monitor preferences / the
 * read-only diagnostics block / reset-to-default), validation errors
 * surfaced inline (role=alert) for out-of-band values, and the
 * persistence visible in the localStorage lane.
 */

describe("the §10 compact settings popover", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.className = "";
    delete document.documentElement.dataset.density;
    getSettingsStore().reset();
    window.localStorage.clear();
  });

  it("opens from the header entry and closes on Escape (compact popover, not a page)", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);

    const entry = screen.getByRole("button", { name: "Settings" });
    expect(entry).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(entry);
    expect(entry).toHaveAttribute("aria-expanded", "true");
    const dialog = screen.getByRole("dialog", { name: "Application settings" });
    expect(dialog).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(entry).toHaveAttribute("aria-expanded", "false");
  });

  it("carries the §10 inventory sections", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);
    await user.click(screen.getByRole("button", { name: "Settings" }));

    for (const section of [
      "Appearance",
      "Charts",
      "Scientific execution defaults",
      "Notifications",
      "Resource monitor",
      "Diagnostics",
    ]) {
      expect(screen.getByRole("heading", { name: section })).toBeInTheDocument();
    }
    expect(
      screen.getByRole("button", { name: "Reset all settings to defaults" }),
    ).toBeInTheDocument();
  });

  it("applies + persists a theme choice (light/dark/system, §10's first ask)", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);
    await user.click(screen.getByRole("button", { name: "Settings" }));

    await user.click(screen.getByRole("radio", { name: "Dark" }));
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY)!).theme).toBe("dark");

    await user.click(screen.getByRole("radio", { name: "Light" }));
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY)!).theme).toBe("light");
  });

  it("applies the density + the colorblind-safe palette + the export default", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);
    await user.click(screen.getByRole("button", { name: "Settings" }));

    await user.click(screen.getByRole("radio", { name: "Compact" }));
    expect(document.documentElement.dataset.density).toBe("compact");

    await user.click(
      screen.getByRole("checkbox", { name: "Colorblind-safe chart palette" }),
    );
    expect(document.documentElement.classList.contains("charts-cvd")).toBe(true);

    await user.click(screen.getByRole("radio", { name: "SVG" }));
    const persisted = JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY)!);
    expect(persisted.charts.defaultExportFormat).toBe("svg");
  });

  it("validates the execution default inline — out-of-band values are refused, never applied", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);
    await user.click(screen.getByRole("button", { name: "Settings" }));

    let parallelism = screen.getByRole("spinbutton", {
      name: "Default parallelism (n_jobs)",
    });
    await user.clear(parallelism);
    await user.type(parallelism, "8");
    await user.tab(); // commit on blur (the input remounts keyed to the new value)
    expect(
      JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY)!).execution
        .defaultParallelism,
    ).toBe(8);

    parallelism = screen.getByRole("spinbutton", {
      name: "Default parallelism (n_jobs)",
    });
    await user.clear(parallelism);
    await user.type(parallelism, "99");
    await user.tab(); // commit refused: -1..64
    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(
      JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY)!).execution
        .defaultParallelism,
    ).toBe(8);
  });

  it("diagnostics shows the app version + the honest not-connected backend state", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);
    await user.click(screen.getByRole("button", { name: "Settings" }));

    expect(screen.getByText("1.2.242")).toBeInTheDocument();
    expect(screen.getByText(/not connected/i)).toBeInTheDocument();
    expect(screen.getByText(/offline/i)).toBeInTheDocument();
  });

  it("reset restores every control to the defaults", async () => {
    const user = userEvent.setup();
    render(<SettingsEntry />);
    await user.click(screen.getByRole("button", { name: "Settings" }));

    await user.click(screen.getByRole("radio", { name: "Dark" }));
    await user.click(screen.getByRole("radio", { name: "Compact" }));
    await user.click(
      screen.getByRole("button", { name: "Reset all settings to defaults" }),
    );
    expect(screen.getByRole("radio", { name: "System" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Comfortable" })).toBeChecked();
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.dataset.density).toBe("comfortable");
  });
});
