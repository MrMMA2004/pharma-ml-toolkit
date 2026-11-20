import { describe, expect, it } from "vitest";

import {
  INITIAL_WORKSPACE,
  WORKSPACES,
} from "@/app/workspaces";

/**
 * part13 Phase 3 — the registry guard on the JS carrier.
 *
 * The pytest contract (tests/test_frontend_foundation.py) pins the
 * §10.1 navigation order against the same source file; this lane
 * asserts it at runtime, where a silent registry edit would otherwise
 * surface only as a reordered UI.
 */
describe("the nine-workspace registry (§10.1)", () => {
  it("carries exactly nine workspaces in the exact §10.1 order", () => {
    expect(WORKSPACES.map((w) => w.id)).toEqual([
      "introduction",
      "datasets",
      "machine-learning",
      "deep-learning",
      "generative-ai",
      "molecular-docking",
      "pkpd",
      "molecular-dynamics",
      "visualization",
    ]);
  });

  it("selects Introduction at startup — the nine-card home never returns", () => {
    expect(INITIAL_WORKSPACE).toBe("introduction");
    expect(WORKSPACES[0]?.id).toBe(INITIAL_WORKSPACE);
  });

  it("never repeats an id or an empty label", () => {
    const ids = new Set(WORKSPACES.map((w) => w.id));
    expect(ids.size).toBe(WORKSPACES.length);
    for (const workspace of WORKSPACES) {
      expect(workspace.label.trim().length).toBeGreaterThan(0);
      expect(workspace.summary.trim().length).toBeGreaterThan(0);
    }
  });

  it("states every row honestly as a connected workflow", () => {
    // `shell` rows were the part11 queue mechanism; every row is
    // workflow-connected now — a regression here is a silent fake
    // of connectedness, exactly what the audit retired.
    expect(WORKSPACES.every((w) => w.state === "workflow")).toBe(true);
  });
});
