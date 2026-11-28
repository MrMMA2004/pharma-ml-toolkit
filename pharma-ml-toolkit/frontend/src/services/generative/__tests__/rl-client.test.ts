import { describe, expect, it } from "vitest";

import { parseRlOptimizeView } from "@/services/generative/client";

/**
 * GEN-3 (the roadmap's row 17) — the RL client pair: the strict
 * parser over the route's own answer.
 *
 * The honest clauses the roadmap pins:
 *
 *   the history rows' absent diagnostics stay null, never implied
 *   zeros; the leaderboard rows parse over the GA lane's own row
 *   parser; the prior view rides parsed (or stays null on the
 *   on-the-fly lane); a malformed answer is the parser's own
 *   refusal.
 */

const RL_ANSWER = {
  mode: "sync",
  optimization: {
    best: "CCCC(=O)Nc1ccc2c(n1)S2",
    best_score: 0.743,
    iterations: 10,
    history: [
      { iteration: 0, best_score: 0.617, mean_score: 0.156, valid_fraction: 0.83, drift: 0.01, entropy: 3.2, loss: 0.4 },
      { iteration: 1, best_score: 0.743, mean_score: 0.2, valid_fraction: 0.87, drift: 0.03, entropy: 3.1, loss: 0.38 },
    ],
    metadata: { method: "reinforce_rl", seed: 3, iterations: 10 },
    top: [
      { smiles: "CCCC(=O)Nc1ccc2c(n1)S2", score: 0.743, breakdown: {} },
    ],
  },
  prior: {
    prior_id: "prior-000111",
    name: "smiles-prior",
    n_train_smiles: 10,
    params: {},
    losses: [0.9, 0.5],
    corpus_source: { source: "inline", n: 10 },
    created_at: "2026-11-28T00:00:00Z",
  },
  note: "the REINVENT-style recipe over the REAL engine",
};

describe("parseRlOptimizeView — the GEN-3 client pair", () => {
  it("mirrors the answer field-for-field (the history + the prior view)", () => {
    const parsed = parseRlOptimizeView(RL_ANSWER, "rl-optimize");
    expect(parsed.mode).toEqual("sync");
    expect(parsed.optimization.best).toEqual("CCCC(=O)Nc1ccc2c(n1)S2");
    expect(parsed.optimization.best_score).toEqual(0.743);
    expect(parsed.optimization.iterations).toEqual(10);
    expect(parsed.optimization.history).toHaveLength(2);
    expect(parsed.optimization.history[0]).toEqual({
      iteration: 0,
      best_score: 0.617,
      mean_score: 0.156,
      valid_fraction: 0.83,
      drift: 0.01,
      entropy: 3.2,
      loss: 0.4,
    });
    expect(parsed.optimization.metadata["method"]).toEqual("reinforce_rl");
    expect(parsed.optimization.top[0]?.smiles).toEqual(
      "CCCC(=O)Nc1ccc2c(n1)S2",
    );
    expect(parsed.prior?.prior_id).toEqual("prior-000111");
    expect(parsed.note).toContain("REINVENT-style");
  });

  it("keeps absent diagnostics honestly null (never implied zeros)", () => {
    const parsed = parseRlOptimizeView(
      {
        ...RL_ANSWER,
        optimization: {
          ...RL_ANSWER.optimization,
          history: [
            { iteration: 0, mean_score: 0.1 },
            { iteration: 1, mean_score: 0.2, drift: 0.02 },
          ],
        },
      },
      "rl-optimize",
    );
    expect(parsed.optimization.history[0]?.drift).toBeNull();
    expect(parsed.optimization.history[1]?.drift).toEqual(0.02);
  });

  it("keeps the prior honestly null on the on-the-fly lane", () => {
    const parsed = parseRlOptimizeView(
      { ...RL_ANSWER, prior: null },
      "rl-optimize",
    );
    expect(parsed.prior).toBeNull();
  });

  it("refuses an answer without its optimization block", () => {
    expect(() =>
      parseRlOptimizeView({ mode: "sync" }, "rl-optimize"),
    ).toThrow(/the optimization block is missing/);
  });

  it("refuses an answer that lost its iterations/note", () => {
    const broken = {
      ...RL_ANSWER,
      optimization: { ...RL_ANSWER.optimization, iterations: null },
    };
    expect(() => parseRlOptimizeView(broken, "rl-optimize")).toThrow(
      /lost its iterations\/note/,
    );
  });

  it("refuses a non-object answer outright", () => {
    expect(() =>
      parseRlOptimizeView("not-an-object", "rl-optimize"),
    ).toThrow();
  });
});
