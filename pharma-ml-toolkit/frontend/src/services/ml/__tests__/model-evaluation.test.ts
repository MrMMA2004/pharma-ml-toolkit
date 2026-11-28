import { describe, expect, it } from "vitest";

import {
  MODEL_EVALUATION_CONDITIONAL_LITERALS,
  parseConformalClassificationResult,
  parseConformalRegressionResult,
  parseCoverageDiagnosticsBlock,
  parseModelEvaluationCatalog,
} from "@/services/ml/client";

/**
 * ML-1 — the model-evaluation family client's runtime parsers: every
 * answer narrows field-by-field (payload narrowing), a malformed field
 * throws instead of rendering as if it were honest (malformed-throw),
 * and the absent optional blocks stay null (the calibration-only lane).
 */

const REGRESSION = {
  alpha: 0.1,
  radius: 0.25,
  n_calibration: 40,
  rank: 36,
  mean_width: 0.5,
  feature_names: ["x1", "x2"],
  intervals: [
    { prediction: 1.2, lower: 0.95, upper: 1.45 },
    { prediction: 2.4, lower: 2.15, upper: 2.65 },
  ],
  empirical: { n_test: 20, n_covered: 19, coverage: 0.95 },
  coverage_diagnostics: {
    mode: "scaffold",
    confidence: 0.95,
    overall: {
      group: "overall",
      n_test: 20,
      n_covered: 19,
      coverage: 0.95,
      lower: 0.75,
      upper: 0.99,
      mean_size: 0.5,
    },
    groups: [
      {
        group: "c1ccccc1",
        n_test: 8,
        n_covered: 8,
        coverage: 1.0,
        lower: 0.67,
        upper: 1.0,
        mean_size: 0.5,
      },
    ],
  },
};

const CLASSIFICATION = {
  alpha: 0.1,
  conditional: "split",
  classes: [0, 1],
  n_calibration: 40,
  threshold: 0.3,
  class_thresholds: null,
  calibration_summary: null,
  sets: [
    { prediction_set: [0, 1], set_size: 2, covered: true },
    { prediction_set: [1], set_size: 1, covered: false },
  ],
  empirical: {
    n_test: 20,
    n_covered: 15,
    coverage: 0.75,
    mean_set_size: 1.6,
  },
  per_class: {
    mode: "groups",
    confidence: 0.95,
    overall: {
      group: "overall",
      n_test: 20,
      n_covered: 15,
      coverage: 0.75,
      lower: 0.53,
      upper: 0.9,
      mean_size: 1.6,
    },
    groups: [
      {
        group: 0,
        n_test: 10,
        n_covered: 9,
        coverage: 0.9,
        lower: 0.6,
        upper: 0.98,
        mean_size: 1.4,
      },
    ],
  },
};

describe("the ML-1 conformal regression parser", () => {
  it("narrows a full payload with intervals, empirical, and the coverage table", () => {
    const parsed = parseConformalRegressionResult(
      JSON.parse(JSON.stringify(REGRESSION)),
      "conformal-regression result",
    );
    expect(parsed.alpha).toBe(0.1);
    expect(parsed.radius).toBe(0.25);
    expect(parsed.n_calibration).toBe(40);
    expect(parsed.rank).toBe(36);
    expect(parsed.mean_width).toBe(0.5);
    expect(parsed.feature_names).toEqual(["x1", "x2"]);
    expect(parsed.intervals?.[0]).toEqual({
      prediction: 1.2,
      lower: 0.95,
      upper: 1.45,
    });
    expect(parsed.empirical?.n_covered).toBe(19);
    expect(parsed.coverage_diagnostics?.mode).toBe("scaffold");
    expect(parsed.coverage_diagnostics?.overall.group).toBe("overall");
    expect(parsed.coverage_diagnostics?.groups[0]?.group).toBe("c1ccccc1");
    expect(parsed.coverage_diagnostics?.groups[0]?.mean_size).toBe(0.5);
  });

  it("keeps the calibration-only lane honest (all optional blocks null)", () => {
    const payload = JSON.parse(JSON.stringify(REGRESSION));
    payload["intervals"] = null;
    payload["empirical"] = null;
    payload["coverage_diagnostics"] = null;
    const parsed = parseConformalRegressionResult(payload, "x");
    expect(parsed.intervals).toBeNull();
    expect(parsed.empirical).toBeNull();
    expect(parsed.coverage_diagnostics).toBeNull();
  });

  it("throws when an interval bound is not a number", () => {
    const hostile = JSON.parse(JSON.stringify(REGRESSION));
    hostile["intervals"][0]["lower"] = "0.95";
    expect(() =>
      parseConformalRegressionResult(hostile, "conformal-regression result"),
    ).toThrow(/bounds are not numbers/);
  });

  it("throws when the radius is not a number", () => {
    const hostile = JSON.parse(JSON.stringify(REGRESSION));
    hostile["radius"] = null;
    expect(() =>
      parseConformalRegressionResult(hostile, "conformal-regression result"),
    ).toThrow(/alpha\/radius\/n_calibration\/rank\/mean_width/);
  });

  it("throws when a coverage group row is malformed", () => {
    const hostile = JSON.parse(JSON.stringify(REGRESSION));
    hostile["coverage_diagnostics"]["groups"][0]["n_covered"] = "8";
    expect(() =>
      parseConformalRegressionResult(hostile, "conformal-regression result"),
    ).toThrow(/n_test\/n_covered are not numbers/);
  });
});

describe("the ML-1 conformal classification parser", () => {
  it("narrows a full payload with sets, empirical, and the per-class table", () => {
    const parsed = parseConformalClassificationResult(
      JSON.parse(JSON.stringify(CLASSIFICATION)),
      "conformal-classification result",
    );
    expect(parsed.alpha).toBe(0.1);
    expect(parsed.conditional).toBe("split");
    expect(parsed.classes).toEqual([0, 1]);
    expect(parsed.threshold).toBe(0.3);
    expect(parsed.sets?.[1]?.set_size).toBe(1);
    expect(parsed.sets?.[1]?.covered).toBe(false);
    expect(parsed.empirical?.mean_set_size).toBe(1.6);
    expect(parsed.per_class?.mode).toBe("groups");
    expect(parsed.per_class?.groups[0]?.group).toBe(0);
  });

  it("throws when conditional is outside the route's literals", () => {
    const hostile = JSON.parse(JSON.stringify(CLASSIFICATION));
    hostile["conditional"] = "marginal";
    expect(() =>
      parseConformalClassificationResult(
        hostile,
        "conformal-classification result",
      ),
    ).toThrow(/route's literals/);
  });

  it("throws when a set row's prediction_set is not an array", () => {
    const hostile = JSON.parse(JSON.stringify(CLASSIFICATION));
    hostile["sets"][0]["prediction_set"] = "0 | 1";
    expect(() =>
      parseConformalClassificationResult(
        hostile,
        "conformal-classification result",
      ),
    ).toThrow(/prediction_set\/set_size are malformed/);
  });

  it("keeps the label-conditional thresholds when the mode rides them", () => {
    const hostile = JSON.parse(JSON.stringify(CLASSIFICATION));
    hostile["conditional"] = "label";
    hostile["class_thresholds"] = { "0": 0.31, "1": 0.29 };
    const parsed = parseConformalClassificationResult(hostile, "x");
    expect(parsed.conditional).toBe("label");
    expect(parsed.class_thresholds?.["0"]).toBe(0.31);
  });

  it("the conditional literal list pins the route's own set", () => {
    expect(MODEL_EVALUATION_CONDITIONAL_LITERALS).toEqual([
      "split",
      "label",
    ]);
  });
});

describe("the ML-1 coverage-diagnostics block parser", () => {
  it("narrows the standalone route's shape (the fourth operation)", () => {
    const parsed = parseCoverageDiagnosticsBlock(
      JSON.parse(
        JSON.stringify(CLASSIFICATION["per_class"] ?? {}),
      ),
      "coverage-diagnostics result",
    );
    expect(parsed.mode).toBe("groups");
    expect(parsed.confidence).toBe(0.95);
    expect(parsed.overall.n_test).toBe(20);
    expect(parsed.groups[0]?.lower).toBe(0.6);
  });

  it("throws when mode is not a string", () => {
    expect(() =>
      parseCoverageDiagnosticsBlock(
        { mode: 7, confidence: 0.95, overall: {}, groups: [] },
        "coverage-diagnostics result",
      ),
    ).toThrow(/mode\/confidence are malformed/);
  });
});

describe("the ML-1 catalog parser", () => {
  it("narrows the family census with the honesty notes", () => {
    const parsed = parseModelEvaluationCatalog(
      {
        entry_points: [
          {
            route: "/conformal-regression",
            backend: "calibrate_split_conformal + predict_interval",
            defaults: { alpha: 0.1 },
          },
        ],
        interval_columns: ["prediction", "lower", "upper"],
        set_columns: ["prediction_set", "set_size", "covered"],
        coverage_columns: ["group", "n_test", "n_covered", "coverage"],
        notes: [
          "sync-only: one predict pass + a quantile",
          "empirical coverage is empirical, never the conformal guarantee",
        ],
      },
      "model-evaluation catalog",
    );
    expect(parsed.entry_points[0]?.route).toBe("/conformal-regression");
    expect(parsed.interval_columns).toEqual(["prediction", "lower", "upper"]);
    expect(parsed.notes[1]).toContain("never the conformal guarantee");
  });

  it("throws when entry_points is not an array", () => {
    expect(() =>
      parseModelEvaluationCatalog(
        { entry_points: "three" },
        "model-evaluation catalog",
      ),
    ).toThrow(/entry_points is not an array/);
  });
});
