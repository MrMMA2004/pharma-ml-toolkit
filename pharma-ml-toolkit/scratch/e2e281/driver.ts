/**
 * The round-281 Datasets-workflow E2E driver (scratch verification —
 * never shipped in the release archive): boots against the LIVE
 * pharmaml backend and proves the §11.1 chain this round connected:
 *
 *   handshake -> configureBackend -> WS stream -> multipart upload
 *   (job mode, real reader) -> tracked completion -> dataset_id ->
 *   THE COLUMN BRIDGE (this round's route: values + n_rows +
 *   truncated, both full and limit=5) -> CHEMISTRY DESCRIPTORS
 *   (job mode through the new chemistry client) -> FINGERPRINTS
 *   (sync mode) -> profile -> split -> THE EXPORT LANE
 *   (POST /artifacts/table + the digest-verified download) ->
 *   graceful shutdown envelope.
 */

import { readFileSync } from "node:fs";

import { api, configureBackend } from "../../frontend/src/services/api/client";
import { getEventStream } from "../../frontend/src/services/websocket/client";
import { getJobTracker } from "../../frontend/src/services/jobs/tracker";
import { jobsApi } from "../../frontend/src/services/jobs/client";
import { datasetsApi } from "../../frontend/src/services/datasets/client";
import { chemistryApi } from "../../frontend/src/services/chemistry/client";
import { artifactsApi } from "../../frontend/src/services/artifacts/client";

const HANDSHAKE = process.argv[2] ?? "/tmp/handshake281.txt";

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

async function trackJobToCompletion(jobId: string) {
  const tracker = getJobTracker();
  tracker.start();
  const untrack = tracker.track(jobId);
  const done = await waitFor(
    "job completion",
    () => {
      const snap = tracker.snapshot(jobId);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED")
        ? snap
        : null;
    },
    60_000,
  );
  const view = done.view;
  if (view === null || view.state !== "COMPLETED") {
    throw new Error(`job did not complete: ${JSON.stringify(view)}`);
  }
  log("tracker", `state=${view.state} progress=${view.progress} stage=${view.stage} lastEvent=${done.lastEvent?.event_type ?? "none"}`);
  if (done.lastEvent === null) {
    throw new Error("the tracker saw no WS envelope — the stream is not flowing");
  }
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

  // 3. the §11.1 Import stage: multipart upload through the REAL reader
  const csv =
    "smiles,pIC50\n" +
    Array.from(
      { length: 16 },
      (_, i) => `C${"C".repeat(i % 5)}N,${(4 + (i % 4) * 0.4).toFixed(2)}`,
    ).join("\n");
  const file = new File([csv], "e2e-281.csv", { type: "text/csv" });
  const upload = await datasetsApi.upload(file, { mode: "job" });
  if (upload.mode !== "job") {
    throw new Error(`expected job-mode upload, got ${upload.mode}`);
  }
  log("upload", `job=${upload.job.job_id} state=${upload.job.state}`);
  await trackJobToCompletion(upload.job.job_id);
  const result = await jobsApi.result(upload.job.job_id);
  const datasetId = String((result.result as Record<string, unknown>)["dataset_id"]);
  if (datasetId === "") {
    throw new Error("the job result carries no dataset_id");
  }
  log("result", `dataset_id=${datasetId}`);

  // 4. THIS ROUND'S ROUTE — the column bridge (full + bounded)
  const column = await datasetsApi.column(datasetId, "smiles");
  if (column.n_rows !== 16 || column.returned !== 16 || column.truncated) {
    throw new Error(`bad column view: ${JSON.stringify(column)}`);
  }
  if (column.values[0] !== "CN" || column.values[1] !== "CCN") {
    throw new Error(`unexpected smiles values: ${JSON.stringify(column.values.slice(0, 3))}`);
  }
  const bounded = await datasetsApi.column(datasetId, "smiles", 5);
  if (bounded.returned !== 5 || !bounded.truncated || bounded.n_rows !== 16) {
    throw new Error(`bad bounded column view: ${JSON.stringify(bounded)}`);
  }
  log("column-bridge", `values=${column.returned}/${column.n_rows} dtype=${column.dtype} truncated@5=${bounded.truncated}`);

  // 5. the chemistry descriptors lane (job mode, the new client)
  const smiles = column.values.filter((v): v is string => typeof v === "string");
  const descriptors = await chemistryApi.descriptors(smiles, {
    descriptorSet: "morgan",
    mode: "job",
  });
  if (descriptors.mode !== "job") {
    throw new Error(`expected job-mode descriptors, got ${descriptors.mode}`);
  }
  await trackJobToCompletion(descriptors.job.job_id);
  const descriptorResult = await jobsApi.result(descriptors.job.job_id);
  const frame = descriptorResult.result as Record<string, unknown>;
  const shape = frame["shape"] as [number, number];
  if (shape[0] !== 16 || shape[1] < 1) {
    throw new Error(`bad descriptor frame shape: ${JSON.stringify(shape)}`);
  }
  log("descriptors", `shape=[${shape[0]}, ${shape[1]}] columns=${(frame["columns"] as string[]).length}`);

  // 6. the fingerprints lane (sync mode for speed)
  const fingerprints = await chemistryApi.fingerprints(smiles, {
    kind: "morgan",
    radius: 2,
    nBits: 512,
    asCounts: false,
    mode: "sync",
  });
  if (fingerprints.mode !== "sync") {
    throw new Error(`expected sync fingerprints, got ${fingerprints.mode}`);
  }
  const matrix = fingerprints.result;
  if (matrix.shape[0] !== 16 || matrix.shape[1] !== 512) {
    throw new Error(`bad fingerprint shape: ${JSON.stringify(matrix.shape)}`);
  }
  log("fingerprints", `shape=[${matrix.shape[0]}, ${matrix.shape[1]}] dtype=${matrix.dtype} invalid=${matrix.invalid_indices.length}`);

  // 7. profile + split (the Validate and Split stages)
  const profile = await datasetsApi.profile(datasetId, { mode: "sync" });
  const variables = profile.profile["variables"] as Record<string, unknown> | undefined;
  log("profile", `variables=${variables === undefined ? "?" : Object.keys(variables).length}`);
  const split = await datasetsApi.split(datasetId, {
    target: null, test_size: 0.25, val_size: null, stratify: true, random_state: 42, mode: "sync",
  });
  log("split", `sides=${Object.keys(split.sides).join("+")} n_rows=${split.n_rows}`);

  // 8. the Export stage: a real table artifact + the digest-verified download
  const exportRaw = await api.post<Record<string, unknown>>(
    "/api/v1/artifacts/table",
    { dataset_id: datasetId, format: "csv" },
  );
  const artifactRaw = exportRaw["artifact"] as Record<string, unknown>;
  const artifactId = String(artifactRaw["artifact_id"]);
  const download = await artifactsApi.download(artifactId);
  if (download.blob.size !== Number(artifactRaw["size_bytes"])) {
    throw new Error(`download size mismatch: blob=${download.blob.size} view=${artifactRaw["size_bytes"]}`);
  }
  if (download.digest_verified !== true) {
    throw new Error(`digest not verified: ${JSON.stringify({
      expected: download.expected_sha256, actual: download.actual_sha256,
    })}`);
  }
  const text = await download.blob.text();
  if (!text.startsWith("smiles,pIC50")) {
    throw new Error(`export content mismatch: ${text.slice(0, 40)}`);
  }
  log("export", `artifact=${artifactId} bytes=${download.blob.size} digest=${download.actual_sha256?.slice(0, 12)}… verified=true`);

  // 9. graceful shutdown: the outer script SIGTERMs the backend; the
  //    `shutdown` envelope must flip this stream to closed — no retry.
  log("await-sigterm", "the outer script will SIGTERM the backend now");
  const afterTerm = await waitFor(
    "shutdown envelope",
    () => (stream.status.state === "closed" ? stream.status : null),
    20_000,
  );
  log("shutdown", `state=${afterTerm.state} cursor=${afterTerm.lastSequence}`);
  getJobTracker().stop();
  stream.close();
  console.log("E2E-DATASETS-PASS");
}

main().catch((error) => {
  console.error("E2E-FAIL", error);
  process.exit(1);
});
