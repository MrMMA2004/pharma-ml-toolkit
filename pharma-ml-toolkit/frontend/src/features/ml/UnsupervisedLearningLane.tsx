/**
 * The part18 frontend lanes: the Unsupervised Learning section
 * (DBSCAN / mean-shift / spectral clustering + Isolation Forest
 * anomaly detection over the REAL backend) and the clearly-labelled
 * EXPERIMENTAL Reinforcement Learning section (Q-learning / SARSA /
 * compact DQN / REINFORCE / actor-critic over the deterministic
 * grid world).
 *
 * Every control is wired to a real operation through
 * `services/ml/unsupervised.ts` (the generated-contract clients); every
 * result renders the parsed payload (diagnostics with honest nulls,
 * noise ratios, the anomaly honesty note, the RL experimental label,
 * reward curves and policies).  Nothing here renders fabricated
 * numbers: degenerate backend answers surface as their own states.
 */

import { useCallback, useState } from "react";
import { AlertTriangle, FlaskConical, Radar, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { DatasetView } from "@/services/datasets/client";
import {
  type AnomalyResultView,
  type ClusterResultView,
  type PcaResultView,
  type ReinforceRunView,
  reinforceApi,
  unsupervisedApi,
} from "@/services/ml/unsupervised";

const SELECT_CLASS =
  "w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm";
const INPUT_CLASS =
  "w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm";

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function fmtNumber(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function fmtPercent(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(1)}%`;
}

// ---------------------------------------------------------------------------
// the Unsupervised Learning lane (part18 §4.3 + §4.5 + ML-4)
// ---------------------------------------------------------------------------
type ClusterMethod =
  | "dbscan"
  | "meanshift"
  | "spectral"
  | "kmeans"
  | "gmm"
  | "hierarchical";

export function UnsupervisedLearningLane({
  selected,
}: {
  selected: DatasetView | null;
}) {
  const [method, setMethod] = useState<ClusterMethod>("dbscan");
  const [epsText, setEpsText] = useState("");
  const [minSamples, setMinSamples] = useState(5);
  const [bandwidthText, setBandwidthText] = useState("");
  const [nClustersText, setNClustersText] = useState("3");
  const [affinity, setAffinity] = useState<"rbf" | "nearest_neighbors">("rbf");
  // ML-4: the new methods' own controls
  const [kText, setKText] = useState("");
  const [gmmComponentsText, setGmmComponentsText] = useState("");
  const [covarianceType, setCovarianceType] = useState<
    "full" | "tied" | "diag" | "spherical"
  >("full");
  const [criterion, setCriterion] = useState<"bic" | "aic">("bic");
  const [linkage, setLinkage] = useState<
    "ward" | "single" | "complete" | "average"
  >("ward");
  const [contamination, setContamination] = useState("auto");
  const [clusterResult, setClusterResult] = useState<ClusterResultView | null>(null);
  const [anomalyResult, setAnomalyResult] = useState<AnomalyResultView | null>(null);
  const [clusterError, setClusterError] = useState<string | null>(null);
  const [anomalyError, setAnomalyError] = useState<string | null>(null);
  const [clusterBusy, setClusterBusy] = useState(false);
  const [anomalyBusy, setAnomalyBusy] = useState(false);
  // ML-4: the PCA sibling lane's own state
  const [pcaComponentsText, setPcaComponentsText] = useState("");
  const [pcaScale, setPcaScale] = useState(true);
  const [pcaResult, setPcaResult] = useState<PcaResultView | null>(null);
  const [pcaError, setPcaError] = useState<string | null>(null);
  const [pcaBusy, setPcaBusy] = useState(false);

  const datasetId = selected?.dataset_id ?? null;

  const runCluster = useCallback(async () => {
    if (!datasetId) {
      setClusterError("select a dataset first (the unsupervised fit runs on its full numeric matrix)");
      return;
    }
    setClusterBusy(true);
    setClusterError(null);
    setClusterResult(null);
    const body: Record<string, unknown> = {
      dataset_id: datasetId,
      method,
      mode: "sync",
    };
    if (method === "dbscan") {
      const eps = Number.parseFloat(epsText);
      body["eps"] = Number.isFinite(eps) && epsText.trim() !== "" ? eps : null;
      body["min_samples"] = minSamples;
    } else if (method === "meanshift") {
      const bandwidth = Number.parseFloat(bandwidthText);
      body["bandwidth"] =
        Number.isFinite(bandwidth) && bandwidthText.trim() !== "" ? bandwidth : null;
    } else if (method === "spectral") {
      const nClusters = Number.parseInt(nClustersText, 10);
      if (!Number.isFinite(nClusters) || nClusters < 2) {
        setClusterError("spectral clustering needs an explicit n_clusters >= 2");
        setClusterBusy(false);
        return;
      }
      body["n_clusters"] = nClusters;
      body["affinity"] = affinity;
    } else if (method === "kmeans") {
      // blank k = the engine's own silhouette scan (the honest default,
      // the sweep rides the response's scan table)
      const k = Number.parseInt(kText, 10);
      body["k"] = Number.isFinite(k) && kText.trim() !== "" ? k : null;
    } else if (method === "gmm") {
      const nComponents = Number.parseInt(gmmComponentsText, 10);
      body["n_components"] =
        Number.isFinite(nComponents) && gmmComponentsText.trim() !== ""
          ? nComponents
          : null;
      body["covariance_type"] = covarianceType;
      body["criterion"] = criterion;
    } else {
      // hierarchical: blank n_clusters rides the engine's own default
      // (3); the linkage select mirrors the engine's own vocabulary
      const nClusters = Number.parseInt(nClustersText, 10);
      body["n_clusters"] =
        Number.isFinite(nClusters) && nClustersText.trim() !== "" ? nClusters : null;
      body["linkage"] = linkage;
    }
    try {
      const answer = await unsupervisedApi.cluster(body);
      if (answer.job) {
        setClusterError(
          `submitted as job ${answer.job.job_id} — watch the Job Center; sync mode re-runs inline`,
        );
      } else if (answer.result) {
        setClusterResult(answer.result);
      }
    } catch (error) {
      setClusterError(describeError(error));
    } finally {
      setClusterBusy(false);
    }
  }, [
    datasetId,
    method,
    epsText,
    minSamples,
    bandwidthText,
    nClustersText,
    affinity,
    kText,
    gmmComponentsText,
    covarianceType,
    criterion,
    linkage,
  ]);

  const runAnomaly = useCallback(async () => {
    if (!datasetId) {
      setAnomalyError("select a dataset first");
      return;
    }
    setAnomalyBusy(true);
    setAnomalyError(null);
    setAnomalyResult(null);
    try {
      const answer = await unsupervisedApi.anomaly({
        dataset_id: datasetId,
        contamination:
          contamination.trim() === "" || contamination.trim() === "auto"
            ? "auto"
            : Number.parseFloat(contamination),
        mode: "sync",
      });
      if (answer.job) {
        setAnomalyError(
          `submitted as job ${answer.job.job_id} — watch the Job Center; sync mode re-runs inline`,
        );
      } else if (answer.result) {
        setAnomalyResult(answer.result);
      }
    } catch (error) {
      setAnomalyError(describeError(error));
    } finally {
      setAnomalyBusy(false);
    }
  }, [datasetId, contamination]);

  // ML-4: the PCA sibling — blank n_components rides the engine's
  // own cap (min(10, n_samples, n_features)); the answer carries the
  // explained-variance ladder + the loadings preview
  const runPca = useCallback(async () => {
    if (!datasetId) {
      setPcaError("select a dataset first (PCA runs on the full numeric matrix)");
      return;
    }
    setPcaBusy(true);
    setPcaError(null);
    setPcaResult(null);
    const nComponents = Number.parseInt(pcaComponentsText, 10);
    const body: Record<string, unknown> = {
      dataset_id: datasetId,
      n_components:
        Number.isFinite(nComponents) && pcaComponentsText.trim() !== ""
          ? nComponents
          : null,
      scale: pcaScale,
      mode: "sync",
    };
    try {
      const answer = await unsupervisedApi.pca(body);
      if (answer.job) {
        setPcaError(
          `submitted as job ${answer.job.job_id} — watch the Job Center; sync mode re-runs inline`,
        );
      } else if (answer.result) {
        setPcaResult(answer.result);
      }
    } catch (error) {
      setPcaError(describeError(error));
    } finally {
      setPcaBusy(false);
    }
  }, [datasetId, pcaComponentsText, pcaScale]);

  return (
    <section className="mt-8 space-y-6" aria-labelledby="unsupervised-lane">
      <Card id="unsupervised-lane-header">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Radar aria-hidden className="h-4 w-4" />
            Unsupervised Learning (part18 §4.3 + §4.5)
          </CardTitle>
          <CardDescription>
            density / mode-seeking / graph clustering and anomaly detection
            over the REAL backend — no target column, no train/test split,
            degenerate outcomes reported honestly (never fabricated scores)
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Clustering</CardTitle>
          <CardDescription>
            DBSCAN (epsilon + min-samples, first-class noise reporting) ·
            Mean Shift (bandwidth-driven modes, no k) · Spectral (affinity +
            explicit cluster count — the clustering, not the embedding) ·
            k-means (elbow AND silhouette, blank k scans the engine's own
            range) · GMM (AIC/BIC sweep, soft responsibilities) ·
            hierarchical (agglomerative k-cut, four linkage rules)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">Method</span>
              <select
                aria-label="the clustering method"
                value={method}
                onChange={(event) => setMethod(event.target.value as ClusterMethod)}
                className={SELECT_CLASS}
              >
                <option value="dbscan">DBSCAN</option>
                <option value="meanshift">Mean Shift</option>
                <option value="spectral">Spectral</option>
                <option value="kmeans">k-means</option>
                <option value="gmm">Gaussian Mixture</option>
                <option value="hierarchical">Hierarchical</option>
              </select>
            </label>
            {method === "dbscan" ? (
              <>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                    eps (blank = k-distance estimate)
                  </span>
                  <input
                    value={epsText}
                    onChange={(event) => setEpsText(event.target.value)}
                    placeholder="auto"
                    className={INPUT_CLASS}
                    inputMode="decimal"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">min_samples</span>
                  <input
                    value={minSamples}
                    onChange={(event) =>
                      setMinSamples(Number.parseInt(event.target.value, 10) || 1)
                    }
                    className={INPUT_CLASS}
                    inputMode="numeric"
                  />
                </label>
              </>
            ) : method === "meanshift" ? (
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                  bandwidth (blank = estimate)
                </span>
                <input
                  value={bandwidthText}
                  onChange={(event) => setBandwidthText(event.target.value)}
                  placeholder="auto"
                  className={INPUT_CLASS}
                  inputMode="decimal"
                />
              </label>
            ) : method === "kmeans" ? (
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                  k (blank = the silhouette scan)
                </span>
                <input
                  aria-label="the k-means k"
                  value={kText}
                  onChange={(event) => setKText(event.target.value)}
                  placeholder="scan"
                  className={INPUT_CLASS}
                  inputMode="numeric"
                />
              </label>
            ) : method === "gmm" ? (
              <>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                    components (blank = the BIC/AIC sweep)
                  </span>
                  <input
                    aria-label="the GMM components"
                    value={gmmComponentsText}
                    onChange={(event) => setGmmComponentsText(event.target.value)}
                    placeholder="sweep"
                    className={INPUT_CLASS}
                    inputMode="numeric"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">covariance</span>
                  <select
                    aria-label="the GMM covariance type"
                    value={covarianceType}
                    onChange={(event) =>
                      setCovarianceType(
                        event.target.value as "full" | "tied" | "diag" | "spherical",
                      )
                    }
                    className={SELECT_CLASS}
                  >
                    <option value="full">full</option>
                    <option value="tied">tied</option>
                    <option value="diag">diag</option>
                    <option value="spherical">spherical</option>
                  </select>
                </label>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">criterion</span>
                  <select
                    aria-label="the GMM criterion"
                    value={criterion}
                    onChange={(event) => setCriterion(event.target.value as "bic" | "aic")}
                    className={SELECT_CLASS}
                  >
                    <option value="bic">bic</option>
                    <option value="aic">aic</option>
                  </select>
                </label>
              </>
            ) : method === "hierarchical" ? (
              <>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                    k (blank = the engine's own 3)
                  </span>
                  <input
                    value={nClustersText}
                    onChange={(event) => setNClustersText(event.target.value)}
                    className={INPUT_CLASS}
                    inputMode="numeric"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">linkage</span>
                  <select
                    aria-label="the hierarchical linkage"
                    value={linkage}
                    onChange={(event) =>
                      setLinkage(
                        event.target.value as "ward" | "single" | "complete" | "average",
                      )
                    }
                    className={SELECT_CLASS}
                  >
                    <option value="ward">ward</option>
                    <option value="single">single</option>
                    <option value="complete">complete</option>
                    <option value="average">average</option>
                  </select>
                </label>
              </>
            ) : (
              <>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                    n_clusters
                  </span>
                  <input
                    value={nClustersText}
                    onChange={(event) => setNClustersText(event.target.value)}
                    className={INPUT_CLASS}
                    inputMode="numeric"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block font-medium text-[var(--color-foreground)]">Affinity</span>
                  <select
                    value={affinity}
                    onChange={(event) =>
                      setAffinity(event.target.value as "rbf" | "nearest_neighbors")
                    }
                    className={SELECT_CLASS}
                  >
                    <option value="rbf">rbf</option>
                    <option value="nearest_neighbors">nearest_neighbors</option>
                  </select>
                </label>
              </>
            )}
            <Button onClick={() => void runCluster()} disabled={clusterBusy}>
              {clusterBusy ? "fitting…" : "Fit clustering"}
            </Button>
          </div>
          {clusterError && (
            <p className="text-sm text-[var(--color-destructive)]" role="alert">{clusterError}</p>
          )}
          {clusterResult && (
            <div className="space-y-2 rounded-md border border-[var(--color-border)] p-4 text-sm">
              <p className="font-mono text-xs leading-relaxed whitespace-pre-wrap">
                {clusterResult.summary}
              </p>
              <ul className="list-disc pl-5 text-xs text-[var(--color-muted-foreground)]">
                <li>
                  eps {fmtNumber(clusterResult.eps, 4)} ({clusterResult.eps_source ?? "—"}
                  {clusterResult.min_samples !== null
                    ? `, min_samples ${clusterResult.min_samples}`
                    : ""}
                  )
                </li>
                <li>
                  bandwidth {fmtNumber(clusterResult.bandwidth, 4)} (
                  {clusterResult.bandwidth_source ?? "—"}
                  {clusterResult.n_modes !== null ? `, ${clusterResult.n_modes} modes` : ""})
                </li>
                <li>
                  labels: n={clusterResult.labels.n}, distinct clusters{" "}
                  {clusterResult.labels.unique.filter((v) => v >= 0).length}
                  {clusterResult.labels.unique.includes(-1) ? ", noise present" : ""}
                </li>
              </ul>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                diagnostics — silhouette {fmtNumber(clusterResult.diagnostics.silhouette)} ·
                Davies–Bouldin {fmtNumber(clusterResult.diagnostics.davies_bouldin)} ·
                Calinski–Harabasz {fmtNumber(clusterResult.diagnostics.calinski_harabasz, 1)} ·
                noise {clusterResult.diagnostics.noise_count} (
                {fmtPercent(clusterResult.diagnostics.noise_ratio)})
              </p>
              {clusterResult.method === "kmeans" ? (
                <div className="space-y-1">
                  <p className="text-xs">
                    k {clusterResult.k} ({clusterResult.k_source}) · inertia{" "}
                    {fmtNumber(clusterResult.inertia)} · silhouette{" "}
                    {fmtNumber(clusterResult.kmeans_silhouette)}
                  </p>
                  {clusterResult.scan.length > 0 ? (
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-[var(--color-muted-foreground)]">
                          <th className="text-left font-medium">k</th>
                          <th className="text-right font-medium">inertia</th>
                          <th className="text-right font-medium">silhouette</th>
                        </tr>
                      </thead>
                      <tbody>
                        {clusterResult.scan.map((row) => (
                          <tr
                            key={row.k}
                            className={
                              row.k === clusterResult.k
                                ? "border-t border-[var(--color-border)] font-medium"
                                : "border-t border-[var(--color-border)]"
                            }
                          >
                            <td className="py-1 text-left">{row.k}</td>
                            <td className="py-1 text-right">{fmtNumber(row.inertia)}</td>
                            <td className="py-1 text-right">{fmtNumber(row.silhouette)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                </div>
              ) : null}
              {clusterResult.method === "gmm" ? (
                <div className="space-y-1">
                  <p className="text-xs">
                    {clusterResult.n_components} components ({clusterResult.chosen_by} ·{" "}
                    {clusterResult.covariance_type}) · log-likelihood{" "}
                    {fmtNumber(clusterResult.log_likelihood)} ·{" "}
                    {clusterResult.converged ? "EM converged" : "EM did not converge"}
                  </p>
                  {clusterResult.weights !== null ? (
                    <p className="text-xs text-[var(--color-muted-foreground)]">
                      mixing weights:{" "}
                      {Object.entries(clusterResult.weights)
                        .map(([name, weight]) => `${name} ${fmtNumber(weight, 2)}`)
                        .join(" · ")}
                    </p>
                  ) : null}
                  {clusterResult.bic_table.length > 0 ? (
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-[var(--color-muted-foreground)]">
                          <th className="text-left font-medium">components</th>
                          <th className="text-right font-medium">aic</th>
                          <th className="text-right font-medium">bic</th>
                        </tr>
                      </thead>
                      <tbody>
                        {clusterResult.bic_table.map((row) => (
                          <tr
                            key={row.n_components}
                            className={
                              row.n_components === clusterResult.n_components
                                ? "border-t border-[var(--color-border)] font-medium"
                                : "border-t border-[var(--color-border)]"
                            }
                          >
                            <td className="py-1 text-left">{row.n_components}</td>
                            <td className="py-1 text-right">{fmtNumber(row.aic, 1)}</td>
                            <td className="py-1 text-right">{fmtNumber(row.bic, 1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                </div>
              ) : null}
              {clusterResult.method === "hierarchical" ? (
                <p className="text-xs">
                  k {clusterResult.k} ({clusterResult.k_source}) · linkage{" "}
                  {clusterResult.linkage} · inertia {fmtNumber(clusterResult.inertia)} ·
                  merge heights [{clusterResult.merge_heights
                    .map((height) => fmtNumber(height, 2))
                    .join(", ")}]
                </p>
              ) : null}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">PCA (dimensionality reduction)</CardTitle>
          <CardDescription>
            the first plot every chemist asks for on a descriptor matrix —
            the real pca_reduce engine: the explained-variance ladder, the
            cumulative share, and the loadings preview (the what-does-PC1-mean
            table); components are axes of retained variance, not causal
            directions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                n_components (blank = min(10, rows, features))
              </span>
              <input
                aria-label="the PCA components"
                value={pcaComponentsText}
                onChange={(event) => setPcaComponentsText(event.target.value)}
                placeholder="auto"
                className={INPUT_CLASS}
                inputMode="numeric"
              />
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={pcaScale}
                onChange={(event) => setPcaScale(event.target.checked)}
                className="h-4 w-4"
              />
              <span className="text-[var(--color-foreground)]">standardise columns first</span>
            </label>
            <Button onClick={() => void runPca()} disabled={pcaBusy}>
              {pcaBusy ? "reducing…" : "Reduce (PCA)"}
            </Button>
          </div>
          {pcaError && (
            <p className="text-sm text-[var(--color-destructive)]" role="alert">{pcaError}</p>
          )}
          {pcaResult && (
            <div className="space-y-2 rounded-md border border-[var(--color-border)] p-4 text-sm">
              <p className="font-mono text-xs leading-relaxed whitespace-pre-wrap">
                {pcaResult.summary}
              </p>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[var(--color-muted-foreground)]">
                    <th className="text-left font-medium">component</th>
                    <th className="text-right font-medium">explained variance</th>
                    <th className="text-right font-medium">cumulative</th>
                  </tr>
                </thead>
                <tbody>
                  {pcaResult.explained_variance_ratio.map((ratio, index) => (
                    <tr key={index} className="border-t border-[var(--color-border)]">
                      <td className="py-1 text-left">PC{index + 1}</td>
                      <td className="py-1 text-right">{fmtPercent(ratio)}</td>
                      <td className="py-1 text-right">
                        {fmtPercent(pcaResult.cumulative_variance[index] ?? null)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {pcaResult.loadings_preview.length > 0 ? (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[var(--color-muted-foreground)]">
                      <th className="text-left font-medium">feature</th>
                      {pcaResult.column_names.slice(0, 6).map((_, index) => (
                        <th key={index} className="text-right font-medium">
                          PC{index + 1}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {pcaResult.loadings_preview.slice(0, 12).map((row, index) => (
                      <tr key={index} className="border-t border-[var(--color-border)]">
                        <td className="py-1 text-left">{String(row[""] ?? row["index"] ?? index)}</td>
                        {Array.from({ length: Math.min(6, pcaResult.n_components) }).map(
                          (_, column) => (
                            <td key={column} className="py-1 text-right">
                              {fmtNumber(
                                typeof row[`PC${column + 1}`] === "number"
                                  ? (row[`PC${column + 1}`] as number)
                                  : null,
                              )}
                            </td>
                          ),
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
              <p className="text-xs text-[var(--color-muted-foreground)]">
                {pcaResult.honesty_note}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Anomaly Detection (Isolation Forest)</CardTitle>
          <CardDescription>
            unsupervised outlier detection — contamination is an analyst prior
            (a share, not a measured rate); scores follow sklearn&apos;s
            decision_function convention (lower = more anomalous)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                contamination (auto or 0.0–0.5)
              </span>
              <input
                value={contamination}
                onChange={(event) => setContamination(event.target.value)}
                placeholder="auto"
                className={INPUT_CLASS}
                inputMode="decimal"
              />
            </label>
            <Button onClick={() => void runAnomaly()} disabled={anomalyBusy}>
              {anomalyBusy ? "fitting…" : "Detect anomalies"}
            </Button>
          </div>
          {anomalyError && (
            <p className="text-sm text-[var(--color-destructive)]" role="alert">{anomalyError}</p>
          )}
          {anomalyResult && (
            <div className="space-y-2 rounded-md border border-[var(--color-border)] p-4 text-sm">
              <p className="font-mono text-xs leading-relaxed whitespace-pre-wrap">
                {anomalyResult.summary}
              </p>
              {anomalyResult.outlier_positions.length > 0 && (
                <p className="text-xs">
                  flagged rows (most anomalous first):{" "}
                  {anomalyResult.outlier_positions.slice(0, 12).join(", ")}
                  {anomalyResult.outlier_positions.length > 12 ? ", …" : ""}
                </p>
              )}
              {anomalyResult.feature_report.length > 0 && (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[var(--color-muted-foreground)]">
                      <th className="text-left font-medium">feature</th>
                      <th className="text-right font-medium">inlier mean</th>
                      <th className="text-right font-medium">outlier mean</th>
                      <th className="text-right font-medium">shift (σ)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {anomalyResult.feature_report.map((row) => (
                      <tr key={row.feature} className="border-t border-[var(--color-border)]">
                        <td className="py-1 text-left">{row.feature}</td>
                        <td className="py-1 text-right">{fmtNumber(row.inlier_mean, 2)}</td>
                        <td className="py-1 text-right">{fmtNumber(row.outlier_mean, 2)}</td>
                        <td className="py-1 text-right">{fmtNumber(row.shift_sigma, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <p className="flex items-start gap-2 text-xs text-[var(--color-muted-foreground)]">
                <AlertTriangle aria-hidden className="mt-0.5 h-3 w-3 shrink-0" />
                {anomalyResult.honesty_note}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}

// ---------------------------------------------------------------------------
// the EXPERIMENTAL Reinforcement Learning lane (part18 §6)
// ---------------------------------------------------------------------------
type RlMethod = "q_learning" | "sarsa" | "dqn" | "reinforce" | "actor_critic";

export function ReinforceLane() {
  const [method, setMethod] = useState<RlMethod>("q_learning");
  const [episodes, setEpisodes] = useState(150);
  const [seed, setSeed] = useState(42);
  const [runResult, setRunResult] = useState<ReinforceRunView | null>(null);
  const [rlError, setRlError] = useState<string | null>(null);
  const [rlBusy, setRlBusy] = useState(false);

  const runTraining = useCallback(async () => {
    setRlBusy(true);
    setRlError(null);
    setRunResult(null);
    try {
      const answer = await reinforceApi.train({
        method,
        rows: 4,
        cols: 5,
        episodes,
        seed,
        mode: "sync",
      });
      if (answer.job) {
        setRlError(
          `submitted as job ${answer.job.job_id} — watch the Job Center; sync mode re-runs inline`,
        );
      } else if (answer.result) {
        setRunResult(answer.result);
      }
    } catch (error) {
      setRlError(describeError(error));
    } finally {
      setRlBusy(false);
    }
  }, [method, episodes, seed]);

  return (
    <section className="mt-8" aria-labelledby="reinforce-lane">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles aria-hidden className="h-4 w-4" />
            Reinforcement Learning — experimental
          </CardTitle>
          <CardDescription>
            part18 §6 lane over the REAL engine: Q-learning · SARSA · a
            compact DQN (NumPy MLP + replay + frozen target) · REINFORCE ·
            actor-critic on the deterministic grid world — NOT supervised
            registry entries and NOT pharmaceutical models (the scientific
            honesty the lane carries)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">Agent</span>
              <select
                value={method}
                onChange={(event) => setMethod(event.target.value as RlMethod)}
                className={SELECT_CLASS}
              >
                <option value="q_learning">Q-Learning (off-policy TD)</option>
                <option value="sarsa">SARSA (on-policy TD)</option>
                <option value="dqn">DQN (compact, NumPy)</option>
                <option value="reinforce">REINFORCE (policy gradient)</option>
                <option value="actor_critic">Actor-Critic (tabular)</option>
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">Episodes</span>
              <input
                value={episodes}
                onChange={(event) =>
                  setEpisodes(Number.parseInt(event.target.value, 10) || 50)
                }
                className={INPUT_CLASS}
                inputMode="numeric"
              />
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">Seed</span>
              <input
                value={seed}
                onChange={(event) =>
                  setSeed(Number.parseInt(event.target.value, 10) || 42)
                }
                className={INPUT_CLASS}
                inputMode="numeric"
              />
            </label>
            <Button onClick={() => void runTraining()} disabled={rlBusy}>
              {rlBusy ? "training…" : "Train agent"}
            </Button>
          </div>
          {rlError && (
            <p className="text-sm text-[var(--color-destructive)]" role="alert">{rlError}</p>
          )}
          {runResult && (
            <div className="space-y-2 rounded-md border border-[var(--color-border)] p-4 text-sm">
              <p className="flex items-center gap-2 text-xs font-medium text-[var(--color-foreground)]">
                <FlaskConical aria-hidden className="h-3 w-3" />
                experimental result — {runResult.method} over {runResult.env}
              </p>
              <p className="text-xs">
                success rate {fmtPercent(runResult.success_rate)} · convergence
                episode{" "}
                {runResult.convergence_episode === null
                  ? "never settled (reported honestly)"
                  : runResult.convergence_episode}
              </p>
              <p className="text-xs">
                reward curve: {runResult.reward_curve.episodes} episodes ·
                first-20 mean {fmtNumber(runResult.reward_curve.first_20_mean)} →
                last-20 mean {fmtNumber(runResult.reward_curve.last_20_mean)} ·
                max {fmtNumber(runResult.reward_curve.max)}
              </p>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                greedy policy over {Object.keys(runResult.policy).length} states;
                actions {Array.from(new Set(Object.values(runResult.policy))).join("/")}
              </p>
              {runResult.notes.length > 0 && (
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  {runResult.notes.join("; ")}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
