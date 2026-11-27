/**
 * DS-1 — the split response parser's narrowing lane.
 *
 * The page tests mock `datasetsApi.split` at the module boundary, so
 * the parser itself is pinned here: the strategy/census fields narrow
 * honestly (unknown literals fall to null, malformed census entries
 * degrade per-side, shared_groups filters to strings) and the legacy
 * answer (no strategy, no census) still parses.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { datasetsApi } from "@/services/datasets/client";
import { api } from "@/services/api/client";

vi.mock("@/services/api/client", () => ({
  api: { get: vi.fn(), post: vi.fn(), del: vi.fn() },
}));

const post = vi.mocked(api.post);

const SIDES = {
  train: {
    rows: 96,
    columns: ["smiles", "y"],
    preview: {
      columns: ["smiles", "y"],
      dtypes: {},
      n_rows: 96,
      returned: 3,
      rows: [["CCO", 1]],
    },
  },
  test: {
    rows: 32,
    columns: ["smiles", "y"],
    preview: {
      columns: ["smiles", "y"],
      dtypes: {},
      n_rows: 32,
      returned: 3,
      rows: [["CCC", 0]],
    },
  },
};

/** the full SplitRequest the generated type requires */
const REQ = {
  test_size: 0.2,
  stratify: true,
  random_state: 42,
  mode: "sync" as const,
};

beforeEach(() => {
  post.mockReset();
});

describe("parseSplitResponse — the DS-1 strategy fields", () => {
  it("narrows the scaffold answer: strategy, smiles_column, census, verdict", async () => {
    post.mockResolvedValueOnce({
      target: "y",
      stratified: false,
      random_state: 42,
      strategy: "scaffold",
      smiles_column: "smiles",
      group_column: null,
      group_census: {
        train: { n_groups: 5, top_groups: [["c1ccccc1", 40]] },
        test: { n_groups: 2, top_groups: [["", 8]] },
        shared_groups: [],
      },
      sides: SIDES,
      n_rows: 128,
    });
    const r = await datasetsApi.split("ds-1", REQ);
    expect(r.strategy).toBe("scaffold");
    expect(r.smiles_column).toBe("smiles");
    expect(r.group_census).not.toBeNull();
    const train = r.group_census?.sides.train;
    expect(train).toBeDefined();
    expect(train?.n_groups).toBe(5);
    expect(train?.top_groups[0]).toEqual(["c1ccccc1", 40]);
    expect(r.group_census?.shared_groups).toEqual([]);
    expect(r.sides.train?.rows).toBe(96);
  });

  it("an unknown strategy literal degrades to null — never a fabricated one", async () => {
    post.mockResolvedValueOnce({
      target: null,
      stratified: true,
      random_state: 7,
      strategy: "definitely-not-a-strategy",
      smiles_column: null,
      group_column: null,
      group_census: null,
      sides: SIDES,
      n_rows: 128,
    });
    const r = await datasetsApi.split("ds-1", REQ);
    expect(r.strategy).toBeNull();
    expect(r.group_census).toBeNull();
  });

  it("the legacy answer (no strategy, no census) parses unchanged", async () => {
    post.mockResolvedValueOnce({
      target: "y",
      stratified: true,
      random_state: 42,
      strategy: null,
      smiles_column: null,
      group_column: null,
      group_census: null,
      sides: SIDES,
      n_rows: 128,
    });
    const r = await datasetsApi.split("ds-1", REQ);
    expect(r.strategy).toBeNull();
    expect(r.group_census).toBeNull();
    expect(r.stratified).toBe(true);
  });

  it("a malformed census degrades honestly: non-string shared entries dropped, odd sides skipped", async () => {
    post.mockResolvedValueOnce({
      target: null,
      stratified: false,
      random_state: 1,
      strategy: "group",
      smiles_column: null,
      group_column: "batch",
      group_census: {
        train: { n_groups: 3, top_groups: [["b1", 9], [7, "x"], ["b2", 4]] },
        "not-an-object": 12,
        shared_groups: ["b1", 42, null],
      },
      sides: SIDES,
      n_rows: 128,
    });
    const r = await datasetsApi.split("ds-1", REQ);
    const train = r.group_census?.sides.train;
    expect(train).toBeDefined();
    expect(train?.n_groups).toBe(3);
    // malformed top-group entries degrade to ("", 0) — never thrown
    const top = train?.top_groups ?? [];
    expect(top).toHaveLength(3);
    expect(top[1]).toEqual(["", 0]);
    expect(top[2]).toEqual(["b2", 4]);
    expect(r.group_census?.shared_groups).toEqual(["b1"]);
  });
});
