"use client";

import { useMemo, useState } from "react";
import {
  CircleCheck,
  Copy,
  Download,
  ListOrdered,
  Loader2,
  Network,
  TriangleAlert,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { CHART } from "@/components/explorer/playground/calc-ui";
import { StatChip } from "@/components/explorer/stat-chip";
import { copyToClipboard, fmtNum } from "@/lib/format";
import type { MctsResponse } from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* MCTS molecule search — the Studio Generative method's web twin      */
/* ------------------------------------------------------------------ */
/**
 * The desktop Studio's Generative page runs three optimisation
 * methods (GA / RL / MCTS); the web twin carried the surgery cards
 * and the SA score but never the search itself. This card wires the
 * third leg: ChemTSv2-style Monte Carlo tree search over an on-the-fly
 * SMILES RNN prior — the objective registry (QED, LogP/MW/TPSA
 * windows, the QED+LogP blend, the druglike triple, similarity to a
 * reference, the SA ease-of-synthesis anchors) is verbatim the
 * Studio's, the corpus semantics (built-in / seeded / custom) mirror
 * the desktop's, and one POST runs the REAL `MctsOptimizer` with real
 * RDKit + torch on the server. The tree policy knob (UCT / PUCT) is
 * the card's honest extra over the Studio surface.
 */

const OBJECTIVE_OPTIONS: ReadonlyArray<{
  value: MctsResponse["objective"] | (string & {});
  label: string;
  needsWindow: boolean;
  needsReference: boolean;
}> = [
  { value: "qed", label: "QED drug-likeness", needsWindow: false, needsReference: false },
  { value: "logp", label: "LogP window", needsWindow: true, needsReference: false },
  { value: "mw", label: "Molecular weight window", needsWindow: true, needsReference: false },
  { value: "tpsa", label: "TPSA window", needsWindow: true, needsReference: false },
  { value: "blend", label: "QED + LogP blend", needsWindow: true, needsReference: false },
  { value: "multi", label: "Multi-objective (QED × MW × TPSA)", needsWindow: false, needsReference: false },
  { value: "similarity", label: "Similarity to reference", needsWindow: false, needsReference: true },
  { value: "sa", label: "SA score (ease of synthesis)", needsWindow: true, needsReference: false },
];

const DEFAULT_OBJECTIVE = "qed";
const DEFAULT_ITERATIONS = 12;
const DEFAULT_SEED = 42;
const DEFAULT_LO = "1.0";
const DEFAULT_HI = "3.0";
const DEFAULT_REFERENCE = "CC(=O)Oc1ccccc1C(=O)O"; // aspirin — the Studio default

type Status = "idle" | "running" | "done" | "error";

/** Save a data-URI figure through a throwaway anchor (the web twin of
 * the desktop cards' per-card "Save figure…"). */
function downloadFigure(dataUri: string, filename: string) {
  const link = document.createElement("a");
  link.href = dataUri;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/** The per-cycle progress chart: best + mean score on the one shared
 * [0, 1] scale every objective scores on (no fabricated second axis). */
function MctsHistoryChart({ response }: { response: MctsResponse }) {
  const { cycle, bestScore, meanScore } = response.history;
  const W = 560;
  const H = 190;
  const pad = { l: 34, r: 14, top: 12, b: 24 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;

  const c0 = cycle[0] ?? 0;
  const c1 = cycle.length > 1 ? (cycle[cycle.length - 1] ?? c0) : c0 + 1;
  const cSpan = c1 - c0 || 1;
  const px = (c: number) => pad.l + ((c - c0) / cSpan) * pw;
  const py = (v: number) => pad.top + ph - v * ph; // scores live in [0, 1]

  const line = (values: number[]) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"}${fmtNum(px(cycle[i] ?? 0), 1)},${fmtNum(py(v), 1)}`)
      .join(" ");

  const series = [
    { label: "best score", values: bestScore, color: CHART.emerald },
    { label: "mean score", values: meanScore, color: CHART.amber },
  ];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`MCTS progress over ${cycle.length} cycles: best and mean score`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={`gy-${f}`}>
          <line
            x1={pad.l}
            y1={pad.top + ph - f * ph}
            x2={pad.l + pw}
            y2={pad.top + ph - f * ph}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
        </g>
      ))}
      {[0, 0.5, 1].map((v) => (
        <text
          key={`ly-${v}`}
          x={pad.l - 6}
          y={pad.top + ph - v * ph + 3.5}
          textAnchor="end"
          className="fill-muted-foreground tabular-nums"
          style={CHART.axisTick}
        >
          {v.toFixed(1)}
        </text>
      ))}
      {cycle.map((c, i) =>
        i % Math.max(1, Math.ceil(cycle.length / 5)) === 0 || i === cycle.length - 1 ? (
          <text
            key={`lx-${i}`}
            x={px(c)}
            y={pad.top + ph + 14}
            textAnchor="middle"
            className="fill-muted-foreground tabular-nums"
            style={CHART.axisTick}
          >
            {c}
          </text>
        ) : null
      )}
      {series.map((s) => (
        <g key={s.label}>
          <path
            d={line(s.values)}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {s.values.map((v, i) => (
            <circle
              key={`pt-${s.label}-${i}`}
              cx={px(cycle[i] ?? 0)}
              cy={py(v)}
              r={2.6}
              fill={s.color}
            />
          ))}
        </g>
      ))}
      {series.map((s, i) => (
        <g key={`lg-${s.label}`}>
          <circle cx={pad.l + 16 + i * 120} cy={8} r={4} fill={s.color} />
          <text
            x={pad.l + 26 + i * 120}
            y={11.5}
            className="fill-muted-foreground"
            style={CHART.axisTick}
          >
            {s.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

export function MctsCard() {
  const [objective, setObjective] = useState<string>(DEFAULT_OBJECTIVE);
  const [iterations, setIterations] = useState(String(DEFAULT_ITERATIONS));
  const [treePolicy, setTreePolicy] = useState<string>("uct");
  const [seed, setSeed] = useState(String(DEFAULT_SEED));
  const [lo, setLo] = useState(DEFAULT_LO);
  const [hi, setHi] = useState(DEFAULT_HI);
  const [reference, setReference] = useState(DEFAULT_REFERENCE);
  const [corpus, setCorpus] = useState("");
  const [seeded, setSeeded] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<MctsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const option = useMemo(
    () => OBJECTIVE_OPTIONS.find((o) => o.value === objective) ?? OBJECTIVE_OPTIONS[0],
    [objective]
  );
  const iterationsParsed = Number.parseInt(iterations, 10);
  const iterationsValid =
    Number.isInteger(iterationsParsed) &&
    iterationsParsed >= 4 &&
    iterationsParsed <= 24;
  const seedParsed = Number.parseInt(seed, 10);
  const seedValid = Number.isInteger(seedParsed) && seedParsed >= 0 && seedParsed <= 999_999;
  const running = status === "running";

  async function run() {
    if (!iterationsValid || !seedValid) {
      setError("Iterations must be 4–24 and the seed a non-negative integer.");
      toast.error("Invalid search settings");
      return;
    }
    if (option.needsReference && !reference.trim()) {
      setError("The similarity objective needs a reference SMILES.");
      toast.error("Missing the reference molecule");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/gen/mcts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          objective,
          iterations: iterationsParsed,
          treePolicy,
          seed: seedParsed,
          lo: Number.parseFloat(lo) || 1.0,
          hi: Number.parseFloat(hi) || 3.0,
          reference: reference.trim(),
          corpus,
          seeded,
        }),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? String((data as { error: unknown }).error)
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const response = data as MctsResponse;
      if (response.objective !== objective) {
        throw new Error("the bridge answered with an unexpected objective");
      }
      setResult(response);
      setStatus("done");
      toast.success(
        `MCTS found ${response.leaderboard.length} molecule(s) — best score ${fmtNum(response.bestScore, 3)}`
      );
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "MCTS search failed.";
      setError(message);
      toast.error("MCTS search failed", { description: message });
    }
  }

  async function copyLeaderboard() {
    if (!result) return;
    const header = `# MCTS ${result.objectiveLabel} — ${result.iterations} cycles, seed ${result.seed}`;
    const columns = ["rank", "smiles", "score", ...result.breakdownColumns];
    const lines = result.leaderboard.map((row) =>
      [
        row.rank,
        row.smiles,
        row.score,
        ...result.breakdownColumns.map((c) => row.breakdown[c] ?? ""),
      ].join(",")
    );
    const ok = await copyToClipboard([header, columns.join(","), ...lines].join("\n"));
    if (ok) {
      setCopied(true);
      toast.success(`Copied ${result.leaderboard.length} leaderboard rows`);
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  return (
    <Card className="overflow-hidden py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Network className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">MCTS molecule search</CardTitle>
            <CardDescription className="text-xs">
              ChemTSv2-style Monte Carlo tree search over an on-the-fly SMILES RNN prior —
              UCT or PUCT tree policies, eight objectives, a deterministic per-seed tree —
              mirroring{" "}
              <code className="font-mono">pharmaml.generative.MctsOptimizer</code>{" "}
              (real RDKit + torch on the server)
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="mcts-objective" className="text-xs font-medium">
              Objective
            </Label>
            <Select value={objective} onValueChange={setObjective}>
              <SelectTrigger id="mcts-objective" className="h-10" aria-label="Objective">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {OBJECTIVE_OPTIONS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1.5">
              <Label htmlFor="mcts-iterations" className="text-xs font-medium">
                Cycles{" "}
                <span className="font-normal text-muted-foreground">(4–24)</span>
              </Label>
              <Input
                id="mcts-iterations"
                inputMode="numeric"
                value={iterations}
                onChange={(e) => setIterations(e.target.value)}
                aria-invalid={iterationsValid ? undefined : true}
                className={`h-10 font-mono text-sm tabular-nums ${
                  iterationsValid ? "" : "border-destructive/60"
                }`}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="mcts-seed" className="text-xs font-medium">
                Seed
              </Label>
              <Input
                id="mcts-seed"
                inputMode="numeric"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                aria-invalid={seedValid ? undefined : true}
                className={`h-10 font-mono text-sm tabular-nums ${
                  seedValid ? "" : "border-destructive/60"
                }`}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="mcts-policy" className="text-xs font-medium">
                Tree policy
              </Label>
              <Select value={treePolicy} onValueChange={setTreePolicy}>
                <SelectTrigger id="mcts-policy" className="h-10" aria-label="Tree policy">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="uct">UCT</SelectItem>
                  <SelectItem value="puct">PUCT</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {option.needsWindow ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-2">
            <div className="space-y-1.5 sm:col-span-1">
              <Label htmlFor="mcts-lo" className="text-xs font-medium">
                Window lo{" "}
                <span className="font-normal text-muted-foreground">
                  {objective === "sa" ? "(easy anchor)" : ""}
                </span>
              </Label>
              <Input
                id="mcts-lo"
                inputMode="decimal"
                value={lo}
                onChange={(e) => setLo(e.target.value)}
                className="h-10 font-mono text-sm tabular-nums"
              />
            </div>
            <div className="space-y-1.5 sm:col-span-1">
              <Label htmlFor="mcts-hi" className="text-xs font-medium">
                Window hi{" "}
                <span className="font-normal text-muted-foreground">
                  {objective === "sa" ? "(hard anchor)" : ""}
                </span>
              </Label>
              <Input
                id="mcts-hi"
                inputMode="decimal"
                value={hi}
                onChange={(e) => setHi(e.target.value)}
                className="h-10 font-mono text-sm tabular-nums"
              />
            </div>
            <p className="col-span-2 self-center text-xs text-muted-foreground sm:col-span-2">
              the objective&apos;s reward window (lo &lt; hi; the SA objective reads them as
              the easy/hard synthesizability anchors)
            </p>
          </div>
        ) : null}

        {option.needsReference ? (
          <div className="space-y-1.5">
            <Label htmlFor="mcts-reference" className="text-xs font-medium">
              Reference molecule (SMILES)
            </Label>
            <Input
              id="mcts-reference"
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              spellCheck={false}
              className="h-10 font-mono text-sm"
              placeholder="CC(=O)Oc1ccccc1C(=O)O"
              aria-label="Reference molecule SMILES"
            />
          </div>
        ) : null}

        <div className="space-y-1.5">
          <Label htmlFor="mcts-corpus" className="text-xs font-medium">
            Prior corpus{" "}
            <span className="font-normal text-muted-foreground">
              (optional — one SMILES per line; empty = the built-in 244-molecule corpus)
            </span>
          </Label>
          <Textarea
            id="mcts-corpus"
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            spellCheck={false}
            className="thin-scrollbar min-h-16 font-mono text-xs"
            placeholder={"CCO\nCCN\nc1ccccc1"}
            aria-label="Optional custom prior corpus"
          />
          <div className="flex items-center gap-2 pt-1">
            <Switch
              id="mcts-seeded"
              checked={seeded}
              onCheckedChange={setSeeded}
              disabled={!corpus.trim()}
              aria-label="Seed the corpus ahead of the built-in molecules"
            />
            <Label
              htmlFor="mcts-seeded"
              className={`text-xs font-medium ${!corpus.trim() ? "text-muted-foreground" : ""}`}
            >
              Seeded mode — anchor these lines ahead of the built-in corpus
            </Label>
          </div>
        </div>

        <Button type="button" onClick={run} disabled={running} className="h-10 gap-2">
          {running ? (
            <>
              <Loader2 className="size-4 animate-spin" aria-hidden />
              Searching…
            </>
          ) : (
            <>
              <Network className="size-4" aria-hidden />
              Run MCTS search
            </>
          )}
        </Button>

        {error ? (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden />
            <AlertTitle>MCTS search failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {result ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
              <StatChip
                icon={Network}
                value={fmtNum(result.bestScore, 3)}
                label={`best ${result.objectiveLabel.toLowerCase()} score`}
                hint={result.best ? "rank 1 molecule" : "nothing valid generated"}
              />
              <StatChip
                icon={ListOrdered}
                value={String(result.treeStats.nUnique)}
                label="unique valid molecules"
                hint={`${result.treeStats.nValid} valid draw(s), ${result.treeStats.nRollouts} rollout(s)`}
              />
              <StatChip
                icon={Network}
                value={String(result.treeStats.nNodes)}
                label="tree nodes"
                hint={`depth ≤ ${result.treeStats.maxDepth}, ${result.treeStats.nCacheHits} cache hit(s)`}
              />
              <StatChip
                icon={CircleCheck}
                value={`${fmtNum(result.elapsedSeconds, 1)} s`}
                label="compute time"
                hint={`real RDKit ${result.rdkitVersion} · ${result.treeStats.policy} prior`}
              />
            </div>

            <p className="text-xs text-muted-foreground">{result.summary}</p>

            <div className="space-y-1.5">
              <Label className="text-xs font-medium">
                Search progress — best vs mean score per cycle
              </Label>
              <div className="rounded-lg border bg-muted/20 p-2">
                <MctsHistoryChart response={result} />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">
                  Leaderboard — rank 1 is the best molecule found
                </Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={copyLeaderboard}
                >
                  {copied ? (
                    <>
                      <CircleCheck className="size-3.5 text-primary" aria-hidden />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="size-3.5" aria-hidden />
                      Copy
                    </>
                  )}
                </Button>
              </div>
              <div className="thin-scrollbar max-h-56 overflow-y-auto rounded-lg border">
                <table className="w-full border-collapse font-mono text-xs">
                  <thead className="sticky top-0 bg-muted/50 text-left">
                    <tr>
                      <th className="px-2 py-1.5 font-medium">rank</th>
                      <th className="px-2 py-1.5 font-medium">smiles</th>
                      <th className="px-2 py-1.5 text-right font-medium">score</th>
                      {result.breakdownColumns.map((column) => (
                        <th key={column} className="px-2 py-1.5 text-right font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.leaderboard.map((row) => (
                      <tr key={`${row.rank}-${row.smiles}`} className="border-t">
                        <td className="px-2 py-1 tabular-nums text-muted-foreground">
                          {row.rank}
                        </td>
                        <td className="px-2 py-1 break-all">{row.smiles}</td>
                        <td className="px-2 py-1 text-right tabular-nums">
                          {fmtNum(row.score, 3)}
                        </td>
                        {result.breakdownColumns.map((column) => (
                          <td key={column} className="px-2 py-1 text-right tabular-nums">
                            {fmtNum(row.breakdown[column] ?? 0, 3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">
                  Best molecules — one 2D panel each
                </Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={() =>
                    downloadFigure(
                      result.gridPng,
                      `mcts-leaderboard-${result.objective}-seed-${result.seed}.png`
                    )
                  }
                >
                  <Download className="size-3.5" aria-hidden />
                  Download PNG
                </Button>
              </div>
              <img
                src={result.gridPng}
                alt={`2D structure grid of the top ${result.gridPanels} molecules of the MCTS ${result.objectiveLabel} run at seed ${result.seed}`}
                className="w-full rounded-lg border bg-white"
              />
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            no search yet — the default (QED, 12 cycles, seed 42) grows a ~90-node tree in
            a few seconds; the same seed regrows the identical tree
          </p>
        )}
      </CardContent>
    </Card>
  );
}
