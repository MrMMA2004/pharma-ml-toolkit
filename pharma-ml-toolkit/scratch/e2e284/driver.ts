/**
 * The round-284 Generative-AI-workspace E2E driver (scratch
 * verification — never shipped in the release archive): boots against
 * the LIVE pharmaml backend and proves the §11.4 workflow on the REAL
 * service clients (the same lanes the page rides):
 *
 *   handshake -> configureBackend -> WS stream (hello) -> dataset
 *   upload (a REAL SMILES corpus, multipart, job mode, tracked) -> the
 *   objectives catalog (the nine kinds) -> the empty prior registry ->
 *   generative prior train (job mode, tracked; the training.metrics
 *   loss-curve envelope accumulated exactly like the page does) -> the
 *   prior view relayed by the authoritative REST result -> the prior
 *   registry now carries it -> rnn.sample (the sync lane, verbatim) ->
 *   generative optimize (job mode, tracked; the per-generation
 *   iteration.update telemetry accumulated exactly like the Generate
 *   stage does) -> the optimization view relayed by the authoritative
 *   REST result (the ranked top rows) -> the descriptors engine over
 *   the sampled working set (job mode, tracked; the frame's own index
 *   IS the validity verdict) -> the prior loss-curve figure artifact
 *   -> the ranked score-profile figure artifact -> the ranked table
 *   export (the rows lane + the digest-verified download) -> SIGTERM
 *   -> the shutdown envelope.
 */

import { readFileSync } from "node:fs";

import { configureBackend } from "../../frontend/src/services/api/client";
import { getEventStream } from "../../frontend/src/services/websocket/client";
import { getJobTracker } from "../../frontend/src/services/jobs/tracker";
import { jobsApi } from "../../frontend/src/services/jobs/client";
import { datasetsApi } from "../../frontend/src/services/datasets/client";
import { artifactsApi } from "../../frontend/src/services/artifacts/client";
import {
  chemistryApi,
  type DescriptorFrame,
} from "../../frontend/src/services/chemistry/client";
import {
  generativeApi,
  parseIterationUpdatePayload,
  parseOptimizeResult,
  parsePriorTrainResult,
  parseTrainingMetricsPayload,
  type IterationUpdatePayload,
  type TrainingMetricsPayload,
} from "../../frontend/src/services/generative/client";

const HANDSHAKE = process.argv[2] ?? "/tmp/handshake284.txt";

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

/** 24 small, genuinely valid SMILES — the prior corpus's own rows. */
const CORPUS_SMILES: readonly string[] = [
  "CCO", "CCN", "CC(=O)O", "c1ccccc1", "CC(C)Cc1ccc(O)cc1",
  "CCOC(=O)c1ccccc1", "CCN(CC)C(=O)N", "c1cccnc1", "CC(=O)Nc1ccccc1",
  "OC(=O)c1ccccc1", "CC(C)O", "CC(O)C", "NCCc1ccc(O)c(Cl)c1",
  "CCOC(=O)CC", "CC(C)(C)O", "c1ccncc1", "CC(=O)OCC",
  "OCc1ccccc1", "CCN1CCN(CC1)Cc1ccccc1", "COc1ccccc1",
  "CC(C)N", "CCCS(=O)(=O)O", "NC(=O)c1ccccc1", "CCCl",
];

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

  // 3. the Generate stage's accumulation lanes: job-scoped
  //    training.metrics (the prior lane) + iteration.update (the
  //    optimizer) + descriptor.progress (the Validate lane)
  let priorJobId: string | null = null;
  let optimizeJobId: string | null = null;
  let descriptorJobId: string | null = null;
  let trainingMetrics: TrainingMetricsPayload | null = null;
  const iterations: IterationUpdatePayload[] = [];
  let descriptorProgressSeen = false;
  stream.subscribe((envelope) => {
    if (
      priorJobId !== null &&
      envelope.job_id === priorJobId &&
      envelope.event_type === "training.metrics"
    ) {
      trainingMetrics = parseTrainingMetricsPayload(
        envelope.payload,
        "training.metrics",
      );
    }
    if (
      optimizeJobId !== null &&
      envelope.job_id === optimizeJobId &&
      envelope.event_type === "iteration.update"
    ) {
      iterations.push(
        parseIterationUpdatePayload(envelope.payload, "iteration.update"),
      );
    }
    if (
      descriptorJobId !== null &&
      envelope.job_id === descriptorJobId &&
      envelope.event_type === "descriptor.progress"
    ) {
      descriptorProgressSeen = true;
    }
  });

  // 4. the multipart upload of a REAL SMILES corpus (job mode)
  const csv =
    "smiles,value\n" +
    CORPUS_SMILES.map((smiles, i) => `${smiles},${(i % 5) * 0.5}`).join("\n");
  const file = new File([csv], "e2e-generative.csv", { type: "text/csv" });
  const upload = await datasetsApi.upload(file, {
    target: "value",
    mode: "job",
  });
  if (upload.mode !== "job") {
    throw new Error(`expected job-mode upload, got ${upload.mode}`);
  }
  log("upload", `job=${upload.job.job_id} state=${upload.job.state}`);

  // 5. the tracker follows the upload job (the r282 dormancy-race fix:
  //    track-first-then-subscribe keeps a listener registered)
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

  // 6. the live objective catalog (the nine kinds)
  const objectives = await generativeApi.objectives();
  if (objectives.objectives.length !== 9) {
    throw new Error(
      `expected the nine-objective catalog, got ${objectives.objectives.length}`,
    );
  }
  const qed = objectives.objectives.find((o) => o.kind === "qed");
  if (qed === undefined || qed.class !== "QedObjective") {
    throw new Error("the catalog lost the QED objective row");
  }
  log("objectives", `kinds=${objectives.objectives.map((o) => o.kind).join(",")}`);

  // 7. the prior registry starts empty
  const emptyRegistry = await generativeApi.priors();
  if (emptyRegistry.priors.length !== 0) {
    throw new Error(
      `expected an empty prior registry at boot, got ${emptyRegistry.priors.length}`,
    );
  }

  // 8. the prior train lane (job mode, the dataset corpus column).
  //     60 epochs — the E2E's own honesty find: a 3-epoch prior samples
  //     junk the descriptor engine rightly rejects (ALL 12 non-blank
  //     invalid), so the prior trains long enough to actually learn
  //     the corpus's grammar (final loss ~0.5) and the sampled set
  //     carries a REAL valid lane to validate
  const trainResponse = await generativeApi.trainPrior({
    smiles: null,
    datasetId,
    column: "smiles",
    name: "e2e-prior",
    embeddingDim: 24,
    hiddenDim: 32,
    nLayers: 1,
    epochs: 60,
    batchSize: 8,
    lr: 0.01,
    gradClip: 5,
    seed: 284,
  });
  if (trainResponse.mode !== "job") {
    throw new Error(`expected job-mode prior train, got ${trainResponse.mode}`);
  }
  priorJobId = trainResponse.job.job_id;
  const untrackPrior = tracker.track(priorJobId);
  const unlistenPrior = tracker.subscribe(priorJobId, () => undefined);
  log("prior-train", `job=${priorJobId} state=${trainResponse.job.state}`);

  const priorDone = await waitFor(
    "prior train completion",
    () => {
      const snap = tracker.snapshot(priorJobId as string);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED" ||
          snap.view.state === "CANCELLED")
        ? snap : null;
    },
    90_000,
  );
  if (priorDone.view?.state !== "COMPLETED") {
    throw new Error(`prior train did not complete: ${JSON.stringify(priorDone.view)}`);
  }
  // the prior lane's loss curve: ONE training.metrics envelope when
  // the training ends — accumulated exactly like the page does
  if (trainingMetrics === null) {
    throw new Error("the prior lane's training.metrics envelope never arrived");
  }
  if (
    trainingMetrics.epochs < 3 ||
    trainingMetrics.losses.length < 3 ||
    trainingMetrics.final_loss === null
  ) {
    throw new Error(
      `the loss-curve envelope is thin: ${JSON.stringify(trainingMetrics)}`,
    );
  }
  log(
    "prior-telemetry",
    `stage=${trainingMetrics.stage} epochs=${trainingMetrics.epochs} final_loss=${trainingMetrics.final_loss.toFixed(4)} losses=${trainingMetrics.losses.map((v) => (v === null ? "null" : v.toFixed(3))).join(",")}`,
  );

  // 9. the authoritative REST result relay — the prior view
  const priorResult = await jobsApi.result(priorJobId);
  const prior = parsePriorTrainResult(priorResult.result, "prior train result");
  if (prior.kind !== "smiles_rnn" || prior.n_train_smiles !== 24) {
    throw new Error(
      `the prior view is ${prior.kind} trained on ${prior.n_train_smiles} SMILES (expected smiles_rnn on 24)`,
    );
  }
  if (prior.losses.length < 3) {
    throw new Error("the prior view lost its loss curve");
  }
  log("prior", `prior_id=${prior.prior_id} epochs=${prior.n_epochs} final=${prior.final_loss?.toFixed(4)} corpus=${JSON.stringify(prior.corpus_source)}`);

  // 10. the prior registry now carries the trained prior
  const registry = await generativeApi.priors();
  if (registry.priors.length !== 1 || registry.priors[0]?.prior_id !== prior.prior_id) {
    throw new Error("the prior registry did not register the trained prior");
  }
  const priorViewById = await generativeApi.priorView(prior.prior_id);
  if (priorViewById.prior_id !== prior.prior_id) {
    throw new Error("the prior view-by-id lane disagrees with the relay");
  }

  // 11. the sample lane (sync by design — the route's own shape).
  //     temperature 0.6: the convergence-checked value whose sampled
  //     set carries a REAL valid lane (the E2E's own experiment — see
  //     the prior train comment above)
  const sample = await generativeApi.sample({
    priorId: prior.prior_id,
    n: 16,
    temperature: 0.6,
    topK: null,
    seed: 284,
    maxLength: 80,
  });
  if (sample.n_sampled !== 16 || sample.samples.length !== 16) {
    throw new Error(
      `the sample lane answered ${sample.n_sampled} samples (expected 16)`,
    );
  }
  if (sample.unique < 1) {
    throw new Error("the sample lane reported zero unique molecules");
  }
  log("sample", `n=${sample.n_sampled} unique=${sample.unique} params=${JSON.stringify(sample.params)}`);

  // 12. the optimize lane (job mode, tracked; the per-generation
  //     iteration.update telemetry accumulates live)
  const optimizeResponse = await generativeApi.optimize({
    objective: { kind: "qed" },
    seedSmiles: CORPUS_SMILES.slice(0, 8),
    datasetId: null,
    column: null,
    populationSize: 8,
    generations: 4,
    mutationRate: 0.08,
    crossoverRate: 0.7,
    elitism: 2,
    tournamentK: 3,
    maxSmilesLength: 80,
    seed: 284,
    topN: 8,
  });
  if (optimizeResponse.mode !== "job") {
    throw new Error(`expected job-mode optimize, got ${optimizeResponse.mode}`);
  }
  optimizeJobId = optimizeResponse.job.job_id;
  const untrackOptimize = tracker.track(optimizeJobId);
  const unlistenOptimize = tracker.subscribe(optimizeJobId, () => undefined);
  log("optimize", `job=${optimizeJobId} state=${optimizeResponse.job.state}`);

  const optimizeDone = await waitFor(
    "optimize completion",
    () => {
      const snap = tracker.snapshot(optimizeJobId as string);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED" ||
          snap.view.state === "CANCELLED")
        ? snap : null;
    },
    60_000,
  );
  if (optimizeDone.view?.state !== "COMPLETED") {
    throw new Error(`optimize did not complete: ${JSON.stringify(optimizeDone.view)}`);
  }
  // the optimizer's own per-generation telemetry (>= 2 of the 4)
  if (iterations.length < 2) {
    throw new Error(
      `the Generate lane saw only ${iterations.length} iteration.update envelopes (expected >= 2)`,
    );
  }
  log(
    "optimize-telemetry",
    `envelopes=${iterations.length} last=${iterations[iterations.length - 1]?.progress}% "${iterations[iterations.length - 1]?.stage}"`,
  );

  // 13. the authoritative REST result relay — the optimization view
  const optimizeResult = await jobsApi.result(optimizeJobId);
  const optimization = parseOptimizeResult(optimizeResult.result, "optimize result");
  if (optimization.top.length < 2) {
    throw new Error(
      `the optimization carried ${optimization.top.length} ranked molecules (expected >= 2)`,
    );
  }
  const scored = optimization.top.filter(
    (row) => row.score !== null && Number.isFinite(row.score),
  );
  if (scored.length < 2) {
    throw new Error("the ranked rows lost their objective scores");
  }
  log("optimization", `best=${optimization.best} best_score=${optimization.best_score?.toFixed(4)} generations=${optimization.generations} top=${optimization.top.length} meta_keys=${Object.keys(optimization.metadata).length}`);

  // 14. the Validate lane: the REAL descriptor engine over the sampled
  //     working set (job mode, tracked; the frame's index IS the verdict).
  //     The request list drops blank SMILES first — chemistry.py's own
  //     non-empty-string validator — exactly like the page does (the
  //     blanks stay in the verdict's invalid lane below)
  const workingSet = sample.samples;
  const descriptorList = workingSet.filter((smiles) => smiles.trim() !== "");
  const blankCount = workingSet.length - descriptorList.length;
  if (descriptorList.length === 0) {
    throw new Error("every sampled SMILES is blank — no descriptor request");
  }
  const descriptorResponse = await chemistryApi.descriptors(descriptorList, {
    descriptorSet: "rdkit_all",
  });
  if (descriptorResponse.mode !== "job") {
    throw new Error(`expected job-mode descriptors, got ${descriptorResponse.mode}`);
  }
  descriptorJobId = descriptorResponse.job.job_id;
  const untrackDescriptors = tracker.track(descriptorJobId);
  const unlistenDescriptors = tracker.subscribe(descriptorJobId, () => undefined);
  log("descriptors", `job=${descriptorJobId} smiles=${descriptorList.length} blanks=${blankCount}`);

  const descriptorDone = await waitFor(
    "descriptor completion",
    () => {
      const snap = tracker.snapshot(descriptorJobId as string);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED")
        ? snap : null;
    },
    60_000,
  );
  if (descriptorDone.view?.state !== "COMPLETED") {
    throw new Error(`descriptors did not complete: ${JSON.stringify(descriptorDone.view)}`);
  }
  const descriptorResult = await jobsApi.result(descriptorJobId);
  const frame = descriptorResult.result as DescriptorFrame;
  if (
    !Array.isArray(frame.columns) ||
    !Array.isArray(frame.rows) ||
    !Array.isArray(frame.index)
  ) {
    throw new Error("the descriptor frame lost its own shape");
  }
  const kept = new Set(
    frame.index.map((v) => Number.parseInt(String(v), 10)),
  );
  const validCount = descriptorList.filter((_, i) => kept.has(i)).length;
  if (validCount < 1) {
    throw new Error("the descriptor engine validated zero sampled molecules");
  }
  if (validCount !== frame.rows.length) {
    throw new Error(
      `the verdict disagrees: ${validCount} valid vs ${frame.rows.length} frame rows`,
    );
  }
  const hasMolWt = frame.columns.includes("MolWt");
  log("validate", `valid=${validCount} invalid=${descriptorList.length - validCount} blanks=${blankCount} columns=${frame.columns.length} molwt=${hasMolWt} descriptor.progress=${descriptorProgressSeen}`);

  // 15. the prior loss-curve figure artifact (the Visualize stage's lane)
  const lossFigure = await generativeApi.priorLossFigure(prior.losses, "PNG");
  if (lossFigure.kind !== "figure" || lossFigure.size_bytes === 0) {
    throw new Error(
      `the loss figure is ${lossFigure.kind} / ${lossFigure.size_bytes} bytes`,
    );
  }
  log("loss-figure", `artifact=${lossFigure.artifact_id} bytes=${lossFigure.size_bytes} sha256=${lossFigure.sha256.slice(0, 12)}…`);

  // 16. the ranked score-profile figure artifact
  const rankFigure = await generativeApi.rankFigure(
    optimization.top.map((row) => ({
      score: row.score,
      kept: true,
    })),
    "PNG",
    "objective score",
  );
  if (rankFigure.kind !== "figure" || rankFigure.size_bytes === 0) {
    throw new Error(
      `the rank figure is ${rankFigure.kind} / ${rankFigure.size_bytes} bytes`,
    );
  }
  log("rank-figure", `artifact=${rankFigure.artifact_id} bytes=${rankFigure.size_bytes} sha256=${rankFigure.sha256.slice(0, 12)}…`);

  // 17. the ranked table export (the rows lane) + the digest-verified download
  const exportArtifact = await generativeApi.exportTable({
    rows: optimization.top.map((row, position) => ({
      rank: position + 1,
      smiles: row.smiles,
      score: row.score,
      kept: true,
    })),
    format: "csv",
    filename: "e2e-generative-ranked",
  });
  if (exportArtifact.kind !== "table" || exportArtifact.size_bytes === 0) {
    throw new Error(
      `the export artifact is ${exportArtifact.kind} / ${exportArtifact.size_bytes} bytes`,
    );
  }
  const exportDownload = await artifactsApi.download(exportArtifact.artifact_id);
  if (exportDownload.digest_verified !== true) {
    throw new Error(
      `the export download digest verdict: ${String(exportDownload.digest_verified)}`,
    );
  }
  if (exportDownload.blob.size !== exportArtifact.size_bytes) {
    throw new Error("the export blob size disagrees with the view");
  }
  log("export", `artifact=${exportArtifact.artifact_id} bytes=${exportArtifact.size_bytes} digest=verified rows=${optimization.top.length}`);

  // 18. graceful shutdown: the outer script SIGTERMs the backend; the
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
  unlistenPrior();
  untrackPrior();
  unlistenOptimize();
  untrackOptimize();
  unlistenDescriptors();
  untrackDescriptors();
  tracker.stop();
  stream.close();
  console.log("E2E-GENERATIVE-PASS");
}

main().catch((error) => {
  console.error("E2E-FAIL", error);
  process.exit(1);
});
