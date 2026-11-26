"use client";

import { useState } from "react";
import {
  CircleCheck,
  Copy,
  Dna,
  Download,
  Loader2,
  Scaling,
  Scissors,
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
import { StatChip } from "@/components/explorer/stat-chip";
import { copyToClipboard, fmtNum } from "@/lib/format";
import { useToolkitStore } from "@/lib/store";
import type {
  SurgeryBreedResponse,
  SurgeryTopologyResponse,
} from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Molecule-surgery demonstrations for the Next.js web console          */
/* ------------------------------------------------------------------ */
/**
 * These separate web-console cards expose ring-topology moves and
 * seeded scaffold-hop crossover through the `pharmaml.generative` API.
 * A server request runs the real operators with RDKit and returns SMILES
 * plus a captioned 2D structure grid. This describes only the web demo;
 * it is not a feature-parity claim for the React/Tauri Studio. Failed
 * crossover draws are returned as null rows, and molecules without
 * qualifying saturated rings receive an explanatory result.
 */

const DEFAULT_TOPOLOGY_SMILES = "C1CCNCC1"; // piperidine: 3 mutants
const DEFAULT_PARENT_A = "CC(=O)Oc1ccccc1C(=O)O"; // aspirin
const DEFAULT_PARENT_B = "CC(C)Cc1ccc(O)cc1"; // ibuprofen core
const DEFAULT_CHILDREN = 4;
const DEFAULT_SEED = 42;

type Status = "idle" | "running" | "done" | "error";

/** Save a data-URI figure from this web-console card via a temporary
 * download link. */
function downloadFigure(dataUri: string, filename: string) {
  const link = document.createElement("a");
  link.href = dataUri;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function postSurgery(body: unknown): Promise<
  SurgeryTopologyResponse | SurgeryBreedResponse
> {
  const res = await fetch("/api/gen/surgery", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data: unknown = await res.json();
  if (!res.ok) {
    const message =
      typeof data === "object" && data !== null && "error" in data
        ? String((data as { error: unknown }).error)
        : `HTTP ${res.status}`;
    throw new Error(message);
  }
  return data as SurgeryTopologyResponse | SurgeryBreedResponse;
}

/* ------------------------------------------------------------------ */
/* The topology card                                                    */
/* ------------------------------------------------------------------ */

export function TopologyMovesCard() {
  const [smiles, setSmiles] = useState(DEFAULT_TOPOLOGY_SMILES);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<SurgeryTopologyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  // (100x-131) a successful run feeds the session report — the latest
  // surgery run wins, exactly like mlResult does for training.
  const setSurgeryResult = useToolkitStore((s) => s.setSurgeryResult);

  const trimmed = smiles.trim();
  const running = status === "running";

  async function enumerate() {
    if (trimmed.length === 0) {
      setError("Type a SMILES in the molecule row first.");
      toast.error("Nothing to enumerate");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const data = await postSurgery({ mode: "topology", smiles: trimmed });
      if (data.mode !== "topology") {
        throw new Error("the bridge answered with an unexpected mode");
      }
      setResult(data);
      setSurgeryResult(data);
      setStatus("done");
      if (data.mutants.length > 0) {
        toast.success(`Enumerated ${data.mutants.length} ring-size mutants`);
      } else {
        toast.info("No qualifying saturated ring", {
          description: data.summary,
        });
      }
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Enumeration failed.";
      setError(message);
      toast.error("Ring topology moves failed", { description: message });
    }
  }

  async function copyMutants() {
    if (!result) return;
    const ok = await copyToClipboard(
      [`# mutants of ${result.source}`, ...result.mutants].join("\n")
    );
    if (ok) {
      setCopied(true);
      toast.success(`Copied ${result.mutants.length} mutant SMILES`);
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
            <Scaling className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">Ring topology moves</CardTitle>
            <CardDescription className="text-xs">
              Every distinct one-step ring-size mutant of one molecule — contraction and
              expansion over each saturated ring (fused/spiro rings are never touched) —
              mirroring <code className="font-mono">pharmaml.generative.ring_topology_moves</code>{" "}
              (real RDKit on the server)
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">
        <div className="space-y-1.5">
          <Label htmlFor="topo-smiles" className="text-xs font-medium">
            Molecule (SMILES)
          </Label>
          <div className="flex flex-wrap gap-2">
            <Input
              id="topo-smiles"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              spellCheck={false}
              className="h-10 min-w-0 flex-1 font-mono text-sm"
              placeholder="C1CCNCC1"
              aria-label="Molecule SMILES"
            />
            <Button
              type="button"
              onClick={enumerate}
              disabled={running}
              className="h-10 gap-2"
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Enumerating…
                </>
              ) : (
                <>
                  <Scissors className="size-4" aria-hidden />
                  Enumerate &amp; draw mutants
                </>
              )}
            </Button>
          </div>
        </div>

        {error ? (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden />
            <AlertTitle>Ring topology moves failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {result ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <StatChip
                icon={Scaling}
                value={String(result.mutants.length)}
                label="distinct mutants"
                hint={
                  result.mutants.length > 0
                    ? "contraction + expansion, sorted"
                    : "no qualifying saturated ring"
                }
              />
              <StatChip
                icon={CircleCheck}
                value={`${fmtNum(result.elapsedSeconds, 1)} s`}
                label="compute time"
                hint={`real RDKit ${result.rdkitVersion} on the server`}
              />
            </div>

            <p className="text-xs text-muted-foreground">{result.summary}</p>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">
                  Mutant SMILES — one per line, most first
                </Label>
                {result.mutants.length > 0 ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px]"
                    onClick={copyMutants}
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
                ) : null}
              </div>
              <div className="thin-scrollbar max-h-40 overflow-y-auto rounded-lg border bg-muted/20 p-3 font-mono text-xs leading-relaxed">
                {result.mutants.length > 0 ? (
                  result.mutants.map((mutant, index) => (
                    <div key={`${index}-${mutant}`} className="py-0.5">
                      <span className="mr-2 inline-block w-14 shrink-0 text-right text-muted-foreground tabular-nums">
                        #{index + 1}
                      </span>
                      {mutant}
                    </div>
                  ))
                ) : (
                  <p className="text-muted-foreground">
                    (nothing to enumerate — {result.summary})
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">
                  Mutant structures — one 2D panel each
                </Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={() =>
                    downloadFigure(
                      result.gridPng,
                      `ring-topology-mutants-${result.source.slice(0, 24)}.png`
                    )
                  }
                >
                  <Download className="size-3.5" aria-hidden />
                  Download PNG
                </Button>
              </div>
              <img
                src={result.gridPng}
                alt={`2D structure grid of the ${result.mutants.length} ring-size mutants of ${result.source}`}
                className="w-full rounded-lg border bg-white"
              />
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            no enumeration yet — piperidine (the default) has three mutants: two
            contractions and one expansion
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* The breed card                                                       */
/* ------------------------------------------------------------------ */

export function BreedMoleculesCard() {
  const [parentA, setParentA] = useState(DEFAULT_PARENT_A);
  const [parentB, setParentB] = useState(DEFAULT_PARENT_B);
  const [children, setChildren] = useState(String(DEFAULT_CHILDREN));
  const [seed, setSeed] = useState(String(DEFAULT_SEED));
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<SurgeryBreedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  // (100x-131) a successful run feeds the session report — the latest
  // surgery run wins, exactly like mlResult does for training.
  const setSurgeryResult = useToolkitStore((s) => s.setSurgeryResult);

  const childrenParsed = Number.parseInt(children, 10);
  const childrenValid =
    Number.isInteger(childrenParsed) && childrenParsed >= 1 && childrenParsed <= 8;
  const seedParsed = Number.parseInt(seed, 10);
  const seedValid = Number.isInteger(seedParsed) && seedParsed >= 0;
  const running = status === "running";

  async function breed() {
    if (!parentA.trim() || !parentB.trim()) {
      setError("Both parent SMILES are required.");
      toast.error("Missing a parent");
      return;
    }
    if (!childrenValid || !seedValid) {
      setError("Children must be 1–8 and the seed a non-negative integer.");
      toast.error("Invalid litter settings");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const data = await postSurgery({
        mode: "breed",
        parentA: parentA.trim(),
        parentB: parentB.trim(),
        children: childrenParsed,
        seed: seedParsed,
      });
      if (data.mode !== "breed") {
        throw new Error("the bridge answered with an unexpected mode");
      }
      setResult(data);
      setSurgeryResult(data);
      setStatus("done");
      if (data.viable > 0) {
        toast.success(`Bred ${data.viable}/${data.children} viable children`);
      } else {
        toast.info("No viable crossover on this draw", {
          description: "change the seed for a different litter",
        });
      }
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Breeding failed.";
      setError(message);
      toast.error("Scaffold-hop crossover failed", { description: message });
    }
  }

  async function copyLitter() {
    if (!result) return;
    const lines = result.litter.map(
      (child, index) => `child ${index + 1},${child ?? "(no viable crossover)"}`
    );
    const ok = await copyToClipboard(
      [`# litter of ${result.parentA} x ${result.parentB} (seed ${result.seed})`, ...lines].join(
        "\n"
      )
    );
    if (ok) {
      setCopied(true);
      toast.success(`Copied ${result.litter.length} litter rows`);
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
            <Dna className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">Breed two molecules</CardTitle>
            <CardDescription className="text-xs">
              Scaffold-hop crossover: homologous cuts on two parents breed hybrid children —
              bridge, ring-merge and linker-transplant geometries compete for each draw —
              mirroring{" "}
              <code className="font-mono">pharmaml.generative.scaffold_hop_crossover</code>{" "}
              (real RDKit on the server)
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">
        <div className="space-y-1.5">
          <Label htmlFor="breed-a" className="text-xs font-medium">
            Parent A (SMILES)
          </Label>
          <Input
            id="breed-a"
            value={parentA}
            onChange={(e) => setParentA(e.target.value)}
            spellCheck={false}
            className="h-10 font-mono text-sm"
            placeholder="CC(=O)Oc1ccccc1C(=O)O"
            aria-label="Parent A SMILES"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="breed-b" className="text-xs font-medium">
            Parent B (SMILES)
          </Label>
          <Input
            id="breed-b"
            value={parentB}
            onChange={(e) => setParentB(e.target.value)}
            spellCheck={false}
            className="h-10 font-mono text-sm"
            placeholder="CC(C)Cc1ccc(O)cc1"
            aria-label="Parent B SMILES"
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="breed-count" className="text-xs font-medium">
              Children{" "}
              <span className="font-normal text-muted-foreground">(1–8 draws)</span>
            </Label>
            <Input
              id="breed-count"
              inputMode="numeric"
              value={children}
              onChange={(e) => setChildren(e.target.value)}
              aria-invalid={childrenValid ? undefined : true}
              className={`h-10 font-mono text-sm tabular-nums ${
                childrenValid ? "" : "border-destructive/60"
              }`}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="breed-seed" className="text-xs font-medium">
              Random seed{" "}
              <span className="font-normal text-muted-foreground">(same seed, same litter)</span>
            </Label>
            <Input
              id="breed-seed"
              inputMode="numeric"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              aria-invalid={seedValid ? undefined : true}
              className={`h-10 font-mono text-sm tabular-nums ${
                seedValid ? "" : "border-destructive/60"
              }`}
            />
          </div>
        </div>
        <Button type="button" onClick={breed} disabled={running} className="h-10 gap-2">
          {running ? (
            <>
              <Loader2 className="size-4 animate-spin" aria-hidden />
              Breeding…
            </>
          ) : (
            <>
              <Dna className="size-4" aria-hidden />
              Breed &amp; draw litter
            </>
          )}
        </Button>

        {error ? (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden />
            <AlertTitle>Scaffold-hop crossover failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {result ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <StatChip
                icon={Dna}
                value={`${result.viable}/${result.children}`}
                label="viable children"
                hint={`seed ${result.seed} — same seed, same litter`}
              />
              <StatChip
                icon={CircleCheck}
                value={`${fmtNum(result.elapsedSeconds, 1)} s`}
                label="compute time"
                hint={`real RDKit ${result.rdkitVersion} on the server`}
              />
            </div>

            <p className="text-xs text-muted-foreground">{result.summary}</p>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">
                  Litter — one row per crossover draw
                </Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={copyLitter}
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
              <div className="thin-scrollbar max-h-40 overflow-y-auto rounded-lg border bg-muted/20 p-3 font-mono text-xs leading-relaxed">
                {result.litter.map((child, index) => (
                  <div key={`${index}-${child ?? "null"}`} className="py-0.5">
                    <span className="mr-2 inline-block w-14 shrink-0 text-right text-muted-foreground tabular-nums">
                      #{index + 1}
                    </span>
                    {child ?? (
                      <span className="italic text-muted-foreground">
                        (no viable crossover on this draw)
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">
                  Parents &amp; children — one 2D panel each
                </Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={() =>
                    downloadFigure(
                      result.gridPng,
                      `scaffold-hop-litter-seed-${result.seed}.png`
                    )
                  }
                >
                  <Download className="size-3.5" aria-hidden />
                  Download PNG
                </Button>
              </div>
              <img
                src={result.gridPng}
                alt={`2D structure grid of the two parents and the ${result.viable} viable children of the seed-${result.seed} litter`}
                className="w-full rounded-lg border bg-white"
              />
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            no children yet — each child is one crossover draw from the seeded rng; a draw
            can honestly fail (no viable cut pair) and says so
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/** The Next.js web-console Surgery sub-tab's two side-by-side cards. */
export function SurgeryCards() {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <TopologyMovesCard />
      <BreedMoleculesCard />
    </div>
  );
}
