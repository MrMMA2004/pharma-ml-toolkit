/**
 * The round-283 DL-workspace E2E driver (scratch verification — never
 * shipped in the release archive): boots against the LIVE pharmaml
 * backend and proves the §11.3 Deep Learning workflow on the REAL
 * service clients (the same lanes the page rides):
 *
 *   handshake -> configureBackend -> WS stream (hello) -> dataset
 *   upload (multipart, job mode, tracked) -> dl networks catalog (the
 *   live signature defaults) -> dl.train (job mode, tracked through
 *   the JobTracker) -> the per-epoch training.metrics telemetry
 *   accumulated EXACTLY like the Monitor stage does -> the COMPLETED
 *   model view relayed by the authoritative REST result -> predict on
 *   the single session path (mlApi.predict, probabilities block) ->
 *   the loss-curve figure artifact -> the checkpoint (the model
 *   artifact lane, digest-verified download) -> the predictions table
 *   export (the rows lane + the mirrored cap) -> SIGTERM -> the
 *   shutdown envelope.
 */

import { readFileSync } from "node:fs";

import { api, configureBackend } from "../../frontend/src/services/api/client";
import { getEventStream } from "../../frontend/src/services/websocket/client";
import { getJobTracker } from "../../frontend/src/services/jobs/tracker";
import { jobsApi } from "../../frontend/src/services/jobs/client";
import { datasetsApi } from "../../frontend/src/services/datasets/client";
import { artifactsApi } from "../../frontend/src/services/artifacts/client";
import { mlApi } from "../../frontend/src/services/ml/client";
import {
  dlApi,
  parseDLTrainResult,
  parseTrainingMetricsPayload,
  type TrainingMetricsPoint,
} from "../../frontend/src/services/dl/client";

const HANDSHAKE = process.argv[2] ?? "/tmp/handshake283.txt";

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
  log("ws-open", `hello cursor=${stream.status.lastSequence ?? -1}`);

  // 3. the Monitor's accumulation lane: subscribe exactly like the
  //    page's Monitor stage (job-scoped training.metrics envelopes)
  let trainJobId: string | null = null;
  const metricsPoints: TrainingMetricsPoint[] = [];
  stream.subscribe((envelope) => {
    if (
      trainJobId !== null &&
      envelope.job_id === trainJobId &&
      envelope.event_type === "training.metrics"
    ) {
      metricsPoints.push(
        parseTrainingMetricsPayload(envelope.payload, "training.metrics"),
      );
    }
  });

  // 4. the multipart upload through the REAL reader (job mode)
  const csv =
    "f1,f2,target\n" +
    Array.from(
      { length: 32 },
      (_, i) =>
        `${(i % 7) * 0.5},${((i * 3) % 5) * 0.25},${i % 2}`,
    ).join("\n");
  const file = new File([csv], "e2e-dl.csv", { type: "text/csv" });
  const upload = await datasetsApi.upload(file, { mode: "job" });
  if (upload.mode !== "job") {
    throw new Error(`expected job-mode upload, got ${upload.mode}`);
  }
  log("upload", `job=${upload.job.job_id} state=${upload.job.state}`);

  // 5. the tracker follows the upload job (subscribe + track, exactly
  //    like useJob does — the r282 honest find: a fast job inside the
  //    subscribe window races the tracker's dormancy rule, so the
  //    driver keeps a subscriber registered so pollTick keeps polling)
  const tracker = getJobTracker();
  tracker.start();
  const untrackUpload = tracker.track(upload.job.job_id);
  const unlistenUpload = tracker.subscribe(upload.job.job_id, () => undefined);
  const uploadDone = await waitFor(
    "upload completion",
    () => {
      const snap = tracker.snapshot(upload.job.job_id);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED")
        ? snap : null;
    },
    30_000,
  );
  if (uploadDone.view?.state !== "COMPLETED") {
    throw new Error(`upload job did not complete: ${JSON.stringify(uploadDone.view)}`);
  }
  const uploadResult = await jobsApi.result(upload.job.job_id);
  const datasetId = uploadResult.result["dataset_id"];
  if (typeof datasetId !== "string" || datasetId === "") {
    throw new Error("the upload result carries no dataset_id");
  }
  log("dataset", `dataset_id=${datasetId}`);

  // 6. the live network catalog (the four torch families)
  const catalog = await dlApi.networks();
  if (catalog.networks.length !== 4) {
    throw new Error(`expected 4 torch families, got ${catalog.networks.length}`);
  }
  const residual = catalog.networks.find((n) => n.network === "residual_mlp_classifier");
  if (residual === undefined) {
    throw new Error("the catalog lost residual_mlp_classifier");
  }
  log("catalog", `families=${catalog.networks.length} defaults=${Object.keys(residual.defaults).length} predict_url=${catalog.predict_url}`);

  // 7. dl.train in job mode (the page's lane), tracked live
  const trainResponse = await dlApi.train({
    datasetId,
    network: residual.network,
    hyperparams: {
      hidden_layers: [8],
      epochs: 3,
      patience: 3,
      batch_size: 8,
      val_split: 0.2,
    },
    mode: "job",
  });
  if (trainResponse.mode !== "job") {
    throw new Error(`expected job-mode train, got ${trainResponse.mode}`);
  }
  trainJobId = trainResponse.job.job_id;
  const untrackTrain = tracker.track(trainJobId);
  const unlistenTrain = tracker.subscribe(trainJobId, () => undefined);
  log("train", `job=${trainJobId} state=${trainResponse.job.state}`);

  const trainDone = await waitFor(
    "train completion",
    () => {
      const snap = tracker.snapshot(trainJobId as string);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED" ||
          snap.view.state === "CANCELLED")
        ? snap : null;
    },
    60_000,
  );
  if (trainDone.view?.state !== "COMPLETED") {
    throw new Error(`train job did not complete: ${JSON.stringify(trainDone.view)}`);
  }
  log("trained", `progress=${trainDone.view.progress} stage=${trainDone.view.stage} telemetry=${metricsPoints.length} epochs`);

  // the REAL per-epoch telemetry accumulated like the Monitor does
  if (metricsPoints.length < 3) {
    throw new Error(
      `the Monitor lane saw only ${metricsPoints.length} training.metrics envelopes (expected >= 3)`,
    );
  }
  if (!metricsPoints.every((p) => typeof p.epoch === "number")) {
    throw new Error("a telemetry point lost its epoch counter");
  }
  const withLosses = metricsPoints.filter(
    (p) => p.train_loss !== null && Number.isFinite(p.train_loss),
  );
  if (withLosses.length < 3) {
    throw new Error("the telemetry points lost their train losses");
  }
  log("monitor", `points=${metricsPoints.length} epochs=${withLosses.map((p) => p.epoch).join(",")} first_loss=${withLosses[0]?.train_loss?.toFixed(4)}`);

  // 8. the authoritative REST result relay (the page's lane)
  const trainResult = await jobsApi.result(trainJobId);
  const model = parseDLTrainResult(trainResult.result, "dl train result");
  if (model.engine !== "torch") {
    throw new Error(`the model engine is ${model.engine}, expected torch`);
  }
  if (model.predict_url !== "/api/v1/ml/predict") {
    throw new Error(`the predict_url is ${model.predict_url}`);
  }
  log("model", `model_id=${model.model_id} task=${model.task} rows=${model.n_train_rows} fit=${model.fit_seconds.toFixed(2)}s params=${Object.keys(model.best_params).length}`);

  // 9. predict on the SINGLE session prediction path
  const predict = await mlApi.predict({
    modelId: model.model_id,
    datasetId,
    predictProba: true,
  });
  if (predict.n_rows !== 32 || predict.predictions.length !== 32) {
    throw new Error(`predict answered ${predict.n_rows} rows (expected 32)`);
  }
  if (predict.probabilities === null || predict.probabilities.rows.length !== 32) {
    throw new Error("the probabilities block is missing its rows");
  }
  log("predict", `rows=${predict.n_rows} classes=${predict.probabilities?.classes.join("/")}`);

  // 10. the loss-curve figure artifact (the Visualize stage's lane)
  const figure = await dlApi.lossCurveFigure({
    trainLoss: metricsPoints.map((p) => p.train_loss),
    valLoss: metricsPoints.map((p) => p.val_loss),
    format: "PNG",
  });
  if (figure.kind !== "figure" || figure.size_bytes === 0) {
    throw new Error(`the figure artifact is ${figure.kind} / ${figure.size_bytes} bytes`);
  }
  log("figure", `artifact=${figure.artifact_id} bytes=${figure.size_bytes} sha256=${figure.sha256.slice(0, 12)}…`);

  // 11. the checkpoint (the model artifact lane) + the digest-verified download
  const bestVals = metricsPoints
    .map((p) => p.best_val)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const checkpoint = await dlApi.saveCheckpoint(
    model.model_id,
    bestVals.length > 0 ? { best_val_loss: Math.min(...bestVals) } : {},
  );
  if (checkpoint.kind !== "model") {
    throw new Error(`the checkpoint artifact is kind ${checkpoint.kind}`);
  }
  const checkpointDownload = await artifactsApi.download(checkpoint.artifact_id);
  if (checkpointDownload.digest_verified !== true) {
    throw new Error(
      `the checkpoint download digest verdict: ${String(checkpointDownload.digest_verified)}`,
    );
  }
  if (checkpointDownload.blob.size !== checkpoint.size_bytes) {
    throw new Error("the checkpoint blob size disagrees with the view");
  }
  log("checkpoint", `artifact=${checkpoint.artifact_id} bytes=${checkpoint.size_bytes} digest=${checkpointDownload.digest_verified ? "verified" : "FAILED"} meta_keys=${Object.keys(checkpoint.meta).length}`);

  // 12. the predictions table export (the rows lane + the mirrored cap)
  const exportRequest = {
    rows: predict.predictions.map((prediction, i) => {
      const row: Record<string, unknown> = { prediction };
      if (predict.probabilities !== null) {
        const probRow = predict.probabilities.rows[i];
        predict.probabilities.classes.forEach((cls, j) => {
          row[`probability_${String(cls)}`] = probRow?.[j] ?? null;
        });
      }
      return row;
    }),
    format: "csv" as const,
    filename: "dl-predictions",
  };
  const exportRaw = await api.post<Record<string, unknown>>(
    "/api/v1/visualization/table",
    exportRequest,
  );
  const artifactRaw = exportRaw["artifact"];
  if (typeof artifactRaw !== "object" || artifactRaw === null) {
    throw new Error("the table export answered without an artifact");
  }
  const exportArtifact = await artifactsApi.view(
    String((artifactRaw as Record<string, unknown>)["artifact_id"] ?? ""),
  );
  const exportDownload = await artifactsApi.download(exportArtifact.artifact_id);
  if (exportDownload.digest_verified !== true) {
    throw new Error("the export download digest was not verified");
  }
  log("export", `artifact=${exportArtifact.artifact_id} bytes=${exportArtifact.size_bytes} digest=verified rows=${exportRequest.rows.length}`);

  // 13. graceful shutdown: the outer script SIGTERMs the backend; the
  //     `shutdown` envelope must flip this stream to closed — no retry.
  log("await-sigterm", "the outer script will SIGTERM the backend now");
  const afterTerm = await waitFor(
    "shutdown envelope",
    () => (stream.status.state === "closed" ? stream.status : null),
    20_000,
  );
  log("shutdown", `state=${afterTerm.state} cursor=${afterTerm.lastSequence}`);
  unlistenUpload();
  untrackUpload();
  unlistenTrain();
  untrackTrain();
  tracker.stop();
  stream.close();
  console.log("E2E-DL-PASS");
}

main().catch((error) => {
  console.error("E2E-FAIL", error);
  process.exit(1);
});
