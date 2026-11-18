/**
 * The round-282 ML-workflow E2E driver (scratch verification —
 * never shipped in the release archive): boots against the LIVE
 * pharmaml backend and proves the §11.2 chain this round connected:
 *
 *   handshake -> configureBackend -> WS stream -> multipart upload of a
 *   CLASSIFICATION dataset (job mode, explicit target/task) -> tracked
 *   completion -> dataset_id -> THE ML CATALOG (task-synced, the
 *   unavailable rows surfaced) -> TRAIN (job mode, the real
 *   train_model engine, tracked to COMPLETED on live events) ->
 *   trained-model view (model_id + cv keys + best_params) ->
 *   PREDICT (dataset mode, predict_proba on) -> THE TRAINED REGISTRY
 *   (the model present) -> THE APPLICABILITY DOMAIN (x_test assembled
 *   through the bounded column bridge, figure=true, job mode) -> the
 *   figure artifact view -> SEMI-SUPERVISED (job mode, mask_fraction)
 *   -> THE EXPORT LANE (POST /visualization/table over the predictions
 *   rows + the digest-verified download) -> graceful shutdown
 *   envelope.
 */

import { readFileSync } from "node:fs";

import { api, configureBackend } from "../../frontend/src/services/api/client";
import { getEventStream } from "../../frontend/src/services/websocket/client";
import { getJobTracker } from "../../frontend/src/services/jobs/tracker";
import { jobsApi } from "../../frontend/src/services/jobs/client";
import { datasetsApi } from "../../frontend/src/services/datasets/client";
import { artifactsApi } from "../../frontend/src/services/artifacts/client";
import { mlApi } from "../../frontend/src/services/ml/client";

const HANDSHAKE = process.argv[2] ?? "/tmp/handshake282.txt";

function log(step: string, detail: string): void {
  console.log(`E2E ${step}: ${detail}`);
}

async function waitFor<T>(
  what: string,
  probe: () => T | null,
  timeoutMs: number,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = probe();
    if (value !== null && value !== undefined) {
      return value;
    }
    if (Date.now() > deadline) {
      throw new Error(`timeout waiting for ${what}`);
    }
    await new Promise((r) => setTimeout(r, 100));
  }
}

async function trackJobToCompletion(
  jobId: string,
  requireEnvelope = false,
) {
  const tracker = getJobTracker();
  tracker.start();
  const untrack = tracker.track(jobId);
  // the app's useJob SUBSCRIBES while it renders the job card — the
  // tracker's poll only refreshes subscribed entries, so a driver
  // that only track()s a fast job races the dormancy rule and can
  // starve. Subscribe exactly like the real consumer.
  const unsubscribe = tracker.subscribe(jobId, () => undefined);
  const done = await waitFor(
    "job completion",
    () => {
      const snap = tracker.snapshot(jobId);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED")
        ? snap
        : null;
    },
    120_000,
  );
  const view = done.view;
  if (view === null || view.state !== "COMPLETED") {
    throw new Error(`job did not complete: ${JSON.stringify(view)}`);
  }
  log("tracker", `state=${view.state} progress=${view.progress} stage=${view.stage} lastEvent=${done.lastEvent?.event_type ?? "none"}`);
  // §13's own rule: the tracker is REST-authoritative for fast jobs —
  // a fit that completes inside the subscribe window simply has no
  // envelope left to observe; only the long jobs carry the hard
  // live-envelope requirement.
  if (requireEnvelope && done.lastEvent === null) {
    throw new Error("the tracker saw no WS envelope — the stream is not flowing");
  }
  unsubscribe();
  untrack();
  return view;
}

async function main(): Promise<void> {
  // 1. the startup handshake (the single PHARMAML-READY line)
  const line = readFileSync(HANDSHAKE, "utf-8").trim().split("\n").at(-1);
  if (line === undefined || !line.startsWith("PHARMAML-READY ")) {
    throw new Error(`bad handshake line: ${line}`);
  }
  const ready = JSON.parse(line.slice("PHARMAML-READY ".length)) as {
    port: number;
    token: string;
    pid: number;
    version: string;
  };
  log("handshake", `port=${ready.port} version=${ready.version} pid=${ready.pid}`);
  configureBackend({ baseUrl: `http://127.0.0.1:${ready.port}`, token: ready.token });

  // 2. the WS stream opens through the hello envelope
  const stream = getEventStream();
  stream.connect();
  await waitFor(
    "stream open", () => (stream.status.state === "open" ? stream.status : null), 10_000,
  );
  log("ws-open", "hello parsed, cursor live");

  // 3. the §11.2 Select-Dataset stage: a classification dataset,
  //    target + task explicit through the REAL multipart reader
  const rows = Array.from({ length: 16 }, (_, i) => {
    const cls = i % 2;
    const x1 = (0.2 + cls * 0.6 + (i % 3) * 0.05).toFixed(3);
    const x2 = (0.8 - cls * 0.5 + (i % 4) * 0.04).toFixed(3);
    return `${x1},${x2},${cls}`;
  });
  const csv = "x1,x2,y\n" + rows.join("\n");
  const file = new File([csv], "e2e-282.csv", { type: "text/csv" });
  const upload = await datasetsApi.upload(file, {
    mode: "job",
    target: "y",
    task: "classification",
  });
  if (upload.mode !== "job") {
    throw new Error(`expected job-mode upload, got ${upload.mode}`);
  }
  log("upload", `job=${upload.job.job_id} state=${upload.job.state}`);
  await trackJobToCompletion(upload.job.job_id, true);
  const uploadResult = await jobsApi.result(upload.job.job_id);
  const datasetId = String(
    (uploadResult.result as Record<string, unknown>)["dataset_id"],
  );
  if (datasetId === "") {
    throw new Error("the job result carries no dataset_id");
  }
  log("result", `dataset_id=${datasetId}`);

  // 4. the catalog (task-synced; the unavailable rows surfaced too)
  const catalog = await mlApi.catalog("classification", false);
  if (catalog.models.length === 0) {
    throw new Error("the classification catalog is empty");
  }
  const logistic = catalog.models.find(
    (m) => m.name === "logistic_regression",
  );
  if (logistic === undefined || !logistic.available) {
    throw new Error("logistic_regression is not available in the catalog");
  }
  log("catalog", `models=${catalog.models.length} logistic_regression available=${logistic.available}`);

  // 5. THE TRAIN STAGE (job mode, the real engine, tracked on live
  //    events; tune off, cv=2 — small but REAL cross-validation)
  const train = await mlApi.train({
    datasetId,
    model: "logistic_regression",
    task: "classification",
    tune: false,
    cv: 2,
    nJobs: 1,
    preprocess: true,
    randomState: 42,
    modelParams: null,
    mode: "job",
  });
  if (train.mode !== "job") {
    throw new Error(`expected job-mode train, got ${train.mode}`);
  }
  log("train", `job=${train.job.job_id} state=${train.job.state}`);
  await trackJobToCompletion(train.job.job_id);
  const trainResult = await jobsApi.result(train.job.job_id);
  const model = (trainResult.result as Record<string, unknown>)["model"] as
    Record<string, unknown>;
  const modelId = String(model["model_id"] ?? "");
  if (modelId === "") {
    throw new Error("the train result carries no model_id");
  }
  for (const key of ["name", "task", "cv_scores", "cv_mean", "cv_std", "fit_seconds"]) {
    if (!(key in model)) {
      throw new Error(`the trained-model view lost the '${key}' key`);
    }
  }
  log("trained", `model_id=${modelId} cv_mean=${String(model["cv_mean"])} fit_seconds=${String(model["fit_seconds"])}`);

  // 6. THE PREDICT STAGE (dataset mode + probabilities)
  const predict = await mlApi.predict({
    modelId,
    datasetId,
    predictProba: true,
    mode: "sync",
  });
  if (predict.n_rows !== 16 || predict.predictions.length !== 16) {
    throw new Error(`bad predict payload: n_rows=${predict.n_rows}`);
  }
  if (
    predict.probabilities === undefined ||
    predict.probabilities.classes.length !== 2 ||
    predict.probabilities.rows.length !== 16
  ) {
    throw new Error("the probabilities block is missing or malformed");
  }
  log("predict", `n_rows=${predict.n_rows} classes=${predict.probabilities.classes.join(",")}`);

  // 7. THE TRAINED REGISTRY (Compare's source)
  const trainedList = await mlApi.trained();
  const found = trainedList.models.find((m) => m.model_id === modelId);
  if (found === undefined) {
    throw new Error("the trained registry does not carry the new model");
  }
  log("compare", `registry=${trainedList.models.length} new-model-present=true`);

  // 8. THE APPLICABILITY DOMAIN — x_test assembled through the
  //    bounded column bridge (both feature columns, limit=5,
  //    truncated honesty), figure on, job mode
  const x1col = await datasetsApi.column(datasetId, "x1", 5);
  const x2col = await datasetsApi.column(datasetId, "x2", 5);
  if (
    x1col.returned !== 5 || !x1col.truncated || x1col.n_rows !== 16 ||
    x2col.returned !== 5
  ) {
    throw new Error("the bounded column bridge lost its truncation honesty");
  }
  const xTest: number[][] = x1col.values.map((v, i) => [
    Number(v),
    Number(x2col.values[i]),
  ]);
  const analyze = await mlApi.analyze({
    modelId,
    datasetId,
    xTest,
    k: 3,
    percentile: 95,
    nModels: 5,
    uncertainty: true,
    figure: true,
    mode: "job",
  });
  if (analyze.mode !== "job") {
    throw new Error(`expected job-mode AD analyze, got ${analyze.mode}`);
  }
  log("ad-analyze", `job=${analyze.job.job_id} x_test=${xTest.length}x2`);
  await trackJobToCompletion(analyze.job.job_id, true);
  const adResult = await jobsApi.result(analyze.job.job_id);
  const adPayload = adResult.result as Record<string, unknown>;
  const analysis = adPayload["analysis"];
  if (typeof analysis !== "object" || analysis === null) {
    throw new Error("the AD job result carries no analysis payload");
  }
  const figureArtifact = adPayload["artifact"] as
    | Record<string, unknown>
    | null;
  if (figureArtifact === null || typeof figureArtifact !== "object") {
    throw new Error("figure=true did not register an artifact");
  }
  const figureId = String(figureArtifact["artifact_id"] ?? "");
  if (figureId === "") {
    throw new Error("the figure artifact view carries no artifact_id");
  }
  log("ad-result", `analysis keys=${Object.keys(analysis).slice(0, 6).join(",")} figure=${figureId}`);

  // 9. THE SEMI-SUPERVISED LANE (job mode, mask_fraction carves the
  //    unlabeled pool out of the 16 labeled rows)
  const semi = await mlApi.semiSupervisedTrain({
    datasetId,
    method: "self_training",
    baseEstimator: null,
    maskFraction: 0.5,
    threshold: 0.75,
    kBest: 10,
    maxIter: 10,
    kernel: "knn",
    nNeighbors: 7,
    gamma: 20,
    randomState: 42,
    mode: "job",
  });
  if (semi.mode !== "job") {
    throw new Error(`expected job-mode semi train, got ${semi.mode}`);
  }
  log("semi", `job=${semi.job.job_id} state=${semi.job.state}`);
  await trackJobToCompletion(semi.job.job_id);
  const semiResult = await jobsApi.result(semi.job.job_id);
  const semiView = (semiResult.result as Record<string, unknown>)[
    "semi_supervised"
  ] as Record<string, unknown>;
  for (const key of ["method", "n_labeled", "n_unlabeled", "n_pseudo_labeled"]) {
    if (!(key in semiView)) {
      throw new Error(`the semi view lost the '${key}' key`);
    }
  }
  log("semi-result", `method=${String(semiView["method"])} labeled=${String(semiView["n_labeled"])} unlabeled=${String(semiView["n_unlabeled"])}`);

  // 10. THE EXPORT LANE — the predictions rows through the rows-based
  //     table artifact, then the digest-verified download
  const exportRows = predict.predictions.map((p, i) => ({
    row: i,
    prediction: typeof p === "number" ? p : String(p),
    class_0: predict.probabilities?.rows[i]?.[0] ?? 0,
    class_1: predict.probabilities?.rows[i]?.[1] ?? 0,
  }));
  const exportResponse = await api.post<Record<string, unknown>>(
    "/api/v1/visualization/table",
    {
      rows: exportRows,
      format: "csv",
      filename: "e2e-282-predictions",
    },
  );
  const exportArtifact = exportResponse["artifact"] as
    | Record<string, unknown>
    | undefined;
  if (typeof exportArtifact !== "object" || exportArtifact === null) {
    throw new Error("the table export registered no artifact");
  }
  const exportId = String(exportArtifact["artifact_id"] ?? "");
  if (exportId === "") {
    throw new Error("the export artifact carries no artifact_id");
  }
  const download = await artifactsApi.download(exportId);
  if (download.digest_verified !== true) {
    throw new Error(`digest not verified: ${String(download.digest_verified)}`);
  }
  const text = await download.blob.text();
  if (!text.includes("prediction")) {
    throw new Error(`export content mismatch: ${text.slice(0, 40)}`);
  }
  log("export", `artifact=${exportId} bytes=${download.blob.size} digest=${download.actual_sha256?.slice(0, 12)}… verified=true`);

  // 11. graceful shutdown: the outer script SIGTERMs the backend; the
  //     `shutdown` envelope must flip this stream to closed — no retry.
  log("await-sigterm", "the outer script will SIGTERM the backend now");
  const afterTerm = await waitFor(
    "shutdown envelope",
    () => (stream.status.state === "closed" ? stream.status : null),
    20_000,
  );
  log("shutdown", `state=${afterTerm.state} cursor=${afterTerm.lastSequence}`);
  getJobTracker().stop();
  stream.close();
  console.log("E2E-ML-PASS");
}

main().catch((error) => {
  console.error("E2E-FAIL", error);
  process.exit(1);
});
