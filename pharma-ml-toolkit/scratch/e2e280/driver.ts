/**
 * The round-280 services E2E driver (scratch verification — never
 * shipped in the release archive): boots against the LIVE pharmaml
 * backend and proves the Phase-Three-services-part-two stack:
 *
 *   handshake -> configureBackend -> WS stream (hello/cursor/
 *   heartbeats) -> dataset upload (multipart, real reader) -> job
 *   tracked through the JobTracker (WS events + REST polls) ->
 *   completed result -> registry/preview/profile/split -> artifact
 *   list/view/download (token-header raw lane) -> graceful shutdown.
 */

import { readFileSync } from "node:fs";

import { api, configureBackend } from "../../../frontend/src/services/api/client";
import { getEventStream } from "../../../frontend/src/services/websocket/client";
import { getJobTracker } from "../../../frontend/src/services/jobs/tracker";
import { jobsApi } from "../../../frontend/src/services/jobs/client";
import { datasetsApi } from "../../../frontend/src/services/datasets/client";
import { artifactsApi } from "../../../frontend/src/services/artifacts/client";

const HANDSHAKE = process.argv[2] ?? "/tmp/handshake280.txt";

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
  const openStatus = await waitFor(
    "stream open", () => (stream.status.state === "open" ? stream.status : null), 10_000,
  );
  log("ws-open", `hello=${JSON.stringify(openStatus.hello)} cursor=${openStatus.lastSequence}`);

  // 3. the multipart upload through the REAL reader (job mode)
  const csv =
    "smiles,pIC50\n" +
    Array.from(
      { length: 24 },
      (_, i) => `C${"C".repeat(i % 6)}N,${(4 + (i % 5) * 0.5).toFixed(2)}`,
    ).join("\n");
  const file = new File([csv], "e2e-280.csv", { type: "text/csv" });
  const upload = await datasetsApi.upload(file, { mode: "job" });
  if (upload.mode !== "job") {
    throw new Error(`expected job-mode upload, got ${upload.mode}`);
  }
  const jobId = upload.job.job_id;
  log("upload", `job=${jobId} state=${upload.job.state}`);

  // 4. the tracker follows the job: WS events + REST reconciliation
  const tracker = getJobTracker();
  tracker.start();
  const untrack = tracker.track(jobId);
  const done = await waitFor(
    "job completion",
    () => {
      const snap = tracker.snapshot(jobId);
      return snap !== null && snap.view !== null &&
        (snap.view.state === "COMPLETED" || snap.view.state === "FAILED")
        ? snap : null;
    },
    30_000,
  );
  const view = done.view;
  if (view === null || view.state !== "COMPLETED") {
    throw new Error(`job did not complete: ${JSON.stringify(view)}`);
  }
  log("tracker", `state=${view.state} progress=${view.progress} stage=${view.stage} lastEvent=${done.lastEvent?.event_type ?? "none"}`);
  if (done.lastEvent === null) {
    throw new Error("the tracker saw no WS envelope — the stream is not flowing");
  }

  // 5. the authoritative REST result fetch
  const result = await jobsApi.result(jobId);
  const datasetView = (result.result as Record<string, unknown>)["dataset_id"];
  const datasetId = typeof datasetView === "string" ? datasetView : "";
  if (datasetId === "") {
    throw new Error("the job result carries no dataset_id");
  }
  log("result", `dataset_id=${datasetId}`);

  // 6. the registry, preview, profile and split (all real backend)
  const listed = await datasetsApi.list();
  const found = listed.datasets.find((d) => d.dataset_id === datasetId);
  if (found === undefined) {
    throw new Error("the uploaded dataset is not in the registry");
  }
  log("registry", `datasets=${listed.datasets.length} shape=[${found.shape}] target=${found.target}`);
  const preview = await datasetsApi.preview(datasetId, 5);
  log("preview", `columns=${preview.columns.length} returned=${preview.returned}`);
  const profile = await datasetsApi.profile(datasetId, { mode: "sync" });
  const profileVars = profile.profile["variables"] as Record<string, unknown> | undefined;
  log("profile", `variables=${profileVars === undefined ? "?" : Object.keys(profileVars).length}`);
  const split = await datasetsApi.split(datasetId, {
    target: null, test_size: 0.25, val_size: null, stratify: true, random_state: 42, mode: "sync",
  });
  log("split", `sides=${Object.keys(split.sides).join("+")} n_rows=${split.n_rows}`);

  // 7. the artifact registry + the token-header download lane
  const backup = await api.post<Record<string, unknown>>(
    "/api/v1/database/backup", { filename: null },
  );
  const artifactRaw = backup["artifact"] as Record<string, unknown> | undefined;
  const backupArtifact = await artifactsApi.view(
    String(artifactRaw?.["artifact_id"] ?? ""),
  );
  const download = await artifactsApi.download(backupArtifact.artifact_id);
  if (download.blob.size !== backupArtifact.size_bytes) {
    throw new Error(`download size mismatch: blob=${download.blob.size} view=${backupArtifact.size_bytes}`);
  }
  const arts = await artifactsApi.list();
  log("artifacts", `list=${arts.artifacts.length} kind=${backupArtifact.kind} size=${backupArtifact.size_bytes} sha256=${download.sha256.slice(0, 12)}…`);

  // 8. the job ledger via the REST surface
  const ledger = await jobsApi.list("datasets");
  log("jobs-ledger", `workspace-jobs=${ledger.jobs.length}`);

  // 9. graceful shutdown: the outer script SIGTERMs the backend; the
  //    `shutdown` envelope must flip this stream to closed — no retry.
  log("await-sigterm", "the outer script will SIGTERM the backend now");
  const afterTerm = await waitFor(
    "shutdown envelope",
    () => (stream.status.state === "closed" ? stream.status : null),
    20_000,
  );
  log("shutdown", `state=${afterTerm.state} cursor=${afterTerm.lastSequence}`);
  untrack();
  tracker.stop();
  stream.close();
  console.log("E2E-SERVICES-PASS");
}

main().catch((error) => {
  console.error("E2E-FAIL", error);
  process.exit(1);
});
