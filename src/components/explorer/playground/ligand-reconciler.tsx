"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeftRight,
  CircleDot,
  Eraser,
  GitCompare,
  ListChecks,
  Sparkles,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { StatChip } from "@/components/explorer/stat-chip";

/** sessionStorage key for the reconciler inputs (survives reloads). */
const RECONCILE_STORAGE_KEY = "pharmaml.ligandReconciler";

interface ReconcilePersisted {
  tableA: string;
  tableB: string;
  reference: string;
}

/** Demo data echoing the legacy notcommon.py / smilesgetter.py workflow:
 * a docking summary column vs the ligand library, plus a SMILES lookup.
 * Note the mixed case and .pdbqt extensions — exactly what the
 * normalization flags are for. */
const SAMPLE_A = `aspirin.pdbqt
IBUPROFEN.pdbqt
Naproxen.pdbqt
paracetamol.pdbqt
caffeine.pdbqt
theophylline.pdbqt
benzocaine.pdbqt
propranolol.pdbqt
coumarin.pdbqt
vanillin.pdbqt`;

const SAMPLE_B = `aspirin
IBUPROFEN
naproxen
paracetamol
caffeine
theophylline
metformin
benzocaine
propranolol
nicotine
atenolol
lidocaine
coumarin
vanillin`;

const SAMPLE_REF = `aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O
naproxen,COc1ccc2cc(ccc2c1)C(C)C(=O)O
paracetamol,CC(=O)Nc1ccc(O)cc1
caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C
theophylline,Cn1c(=O)c2[nH]cnc2n(C)c1=O
metformin,CN(C)C(=N)N=C(N)N
propranolol,CC(C)NCC(O)COc1cccc2ccccc12
coumarin,O=C1Oc2ccccc2C=C1
vanillin,COc1cc(C=O)ccc1O`;

/** Read the persisted reconciler input (client only, failure-tolerant). */
function loadReconcileInput(): ReconcilePersisted {
  if (typeof window === "undefined") return { tableA: "", tableB: "", reference: "" };
  try {
    const raw = window.sessionStorage.getItem(RECONCILE_STORAGE_KEY);
    if (!raw) return { tableA: "", tableB: "", reference: "" };
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return { tableA: "", tableB: "", reference: "" };
    }
    const p = parsed as Partial<ReconcilePersisted>;
    return {
      tableA: typeof p.tableA === "string" ? p.tableA.slice(0, 100_000) : "",
      tableB: typeof p.tableB === "string" ? p.tableB.slice(0, 100_000) : "",
      reference: typeof p.reference === "string" ? p.reference.slice(0, 100_000) : "",
    };
  } catch {
    return { tableA: "", tableB: "", reference: "" };
  }
}

/** Split a pasted ligand list into trimmed, non-empty entries.
 * Accepts one-per-line or comma/semicolon separated values. */
function parseLines(raw: string): string[] {
  return raw
    .split(/[\n,;]+/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

/** Normalize one ligand name exactly like pharmaml.chem.compare_tables:
 * whitespace always stripped; the file extension and letter case
 * removed when the corresponding flags are set. */
function normalizeName(raw: string, stripExtension: boolean, lowercase: boolean): string {
  let text = raw.trim();
  if (stripExtension) {
    const dot = text.lastIndexOf(".");
    if (dot > 0) text = text.slice(0, dot);
  }
  if (lowercase) text = text.toLowerCase();
  return text;
}

interface ReconcileResult {
  common: Array<{ name: string; smiles: string | null }>;
  onlyA: Array<{ name: string; smiles: string | null }>;
  onlyB: Array<{ name: string; smiles: string | null }>;
}

/** The compare_tables set logic + annotate_with_smiles lookup, client-side. */
function reconcile(
  aRaw: string[],
  bRaw: string[],
  referenceRaw: string,
  stripExtension: boolean,
  lowercase: boolean
): ReconcileResult {
  const normalize = (value: string) => normalizeName(value, stripExtension, lowercase);

  const lookup = new Map<string, string>();
  for (const line of referenceRaw
    .split(/\n+/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0)) {
    const comma = line.indexOf(",");
    if (comma <= 0) continue;
    const key = normalize(line.slice(0, comma));
    const smiles = line.slice(comma + 1).trim();
    if (key) lookup.set(key, smiles);
  }

  const originalOf = new Map<string, string>();
  const setA = new Set<string>();
  for (const entry of aRaw) {
    const key = normalize(entry);
    if (!setA.has(key)) originalOf.set(key, entry);
    setA.add(key);
  }
  const setB = new Set<string>();
  for (const entry of bRaw) {
    const key = normalize(entry);
    if (!setB.has(key)) originalOf.set(key, entry);
    setB.add(key);
  }

  const annotate = (key: string): { name: string; smiles: string | null } => ({
    name: originalOf.get(key) ?? key,
    smiles: referenceRaw.trim() ? (lookup.get(key) ?? "") : null,
  });

  return {
    common: [...setA].filter((k) => setB.has(k)).sort().map(annotate),
    onlyA: [...setA].filter((k) => !setB.has(k)).sort().map(annotate),
    onlyB: [...setB].filter((k) => !setA.has(k)).sort().map(annotate),
  };
}

function ResultColumn({
  title,
  hint,
  items,
  accent,
  emptyText,
}: {
  title: string;
  hint: string;
  items: Array<{ name: string; smiles: string | null }>;
  accent: string;
  emptyText: string;
}) {
  return (
    <div className="flex min-w-0 flex-col rounded-lg border bg-card">
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {title}
          </p>
          <p className="truncate text-[11px] text-muted-foreground/80">{hint}</p>
        </div>
        <Badge variant="secondary" className="shrink-0 tabular-nums">
          {items.length}
        </Badge>
      </div>
      <ul className="thin-scrollbar max-h-72 overflow-y-auto p-2 text-sm">
        {items.length === 0 ? (
          <li className="px-2 py-3 text-center text-xs text-muted-foreground">
            {emptyText}
          </li>
        ) : (
          items.map((item, index) => (
            <li
              key={`${item.name}-${index}`}
              className={`flex flex-col gap-0.5 rounded-md px-2 py-1.5 transition-colors hover:bg-muted/40 ${
                item.smiles === "" ? "opacity-70" : ""
              }`}
            >
              <span className="truncate font-mono text-[13px]" title={item.name}>
                <span className={accent}>●</span> {item.name}
              </span>
              {item.smiles !== null && (
                <span
                  className="truncate font-mono text-[11px] text-muted-foreground"
                  title={item.smiles || "no SMILES in the reference library"}
                >
                  {item.smiles || "— no SMILES in reference"}
                </span>
              )}
            </li>
          ))
        )}
      </ul>
    </div>
  );
}

export function LigandReconciler() {
  const [restored] = useState(loadReconcileInput);
  const [tableA, setTableA] = useState(restored.tableA);
  const [tableB, setTableB] = useState(restored.tableB);
  const [reference, setReference] = useState(restored.reference);
  const [stripExtension, setStripExtension] = useState(true);
  const [lowercase, setLowercase] = useState(true);
  const [nameColumn, setNameColumn] = useState("Ligand");

  /* Persist inputs so a reload doesn't lose work (optional, quota-tolerant). */
  useEffect(() => {
    try {
      window.sessionStorage.setItem(
        RECONCILE_STORAGE_KEY,
        JSON.stringify({ tableA, tableB, reference })
      );
    } catch {
      /* quota exceeded or private mode — persistence is optional */
    }
  }, [tableA, tableB, reference]);

  const result = useMemo(
    () =>
      reconcile(
        parseLines(tableA),
        parseLines(tableB),
        reference,
        stripExtension,
        lowercase
      ),
    [tableA, tableB, reference, stripExtension, lowercase]
  );

  const total =
    result.common.length + result.onlyA.length + result.onlyB.length;

  const loadSample = () => {
    setTableA(SAMPLE_A);
    setTableB(SAMPLE_B);
    setReference(SAMPLE_REF);
  };

  const clearAll = () => {
    setTableA("");
    setTableB("");
    setReference("");
  };

  return (
    <div className="mx-auto grid w-full max-w-5xl gap-4 sm:gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-2 text-base">
            <GitCompare className="size-4 text-teal-600 dark:text-teal-400" aria-hidden />
            Ligand-set reconciliation
            <Badge variant="outline" className="font-normal text-muted-foreground">
              client-side mirror of pharmaml.chem.compare_tables
            </Badge>
          </CardTitle>
          <CardDescription>
            Paste two ligand-name columns (one entry per line, or comma
            separated) — the exact workflow of the legacy{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
              notcommon.py
            </code>{" "}
            /{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
              smilesgetter.py
            </code>{" "}
            scripts. Names are normalized before comparing, so{" "}
            <span className="font-mono text-[12px]">LIGAND1.pdbqt</span> equals{" "}
            <span className="font-mono text-[12px]">ligand1</span>. In Python:{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
              from pharmaml.chem import compare_tables, annotate_with_smiles
            </code>
            .
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="grid gap-1.5">
              <Label htmlFor="ligands-a" className="text-xs text-muted-foreground">
                Table A — {nameColumn} column (e.g. the docking summary)
              </Label>
              <Textarea
                id="ligands-a"
                value={tableA}
                onChange={(event) => setTableA(event.target.value)}
                placeholder={"01_aspirin.pdbqt\n02_ibuprofen.pdbqt\n…"}
                className="thin-scrollbar min-h-36 font-mono text-[13px] leading-relaxed"
                spellCheck={false}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="ligands-b" className="text-xs text-muted-foreground">
                Table B — {nameColumn} column (e.g. the ligand library)
              </Label>
              <Textarea
                id="ligands-b"
                value={tableB}
                onChange={(event) => setTableB(event.target.value)}
                placeholder={"aspirin\nibuprofen\n…"}
                className="thin-scrollbar min-h-36 font-mono text-[13px] leading-relaxed"
                spellCheck={false}
              />
            </div>
          </div>

          <div className="grid gap-1.5">
            <Label htmlFor="reference-lib" className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Sparkles className="size-3.5 text-amber-500" aria-hidden />
              Reference library — optional{" "}
              <span className="font-mono text-[11px]">name,SMILES</span> lines
              (annotates every result with SMILES, like smilesgetter.py)
            </Label>
            <Textarea
              id="reference-lib"
              value={reference}
              onChange={(event) => setReference(event.target.value)}
              placeholder={"aspirin,CC(=O)OC1=CC=CC=C1C(=O)O\n…"}
              className="thin-scrollbar min-h-20 font-mono text-[13px] leading-relaxed"
              spellCheck={false}
            />
          </div>

          <div className="flex flex-wrap items-end justify-between gap-3">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  id="strip-ext"
                  checked={stripExtension}
                  onCheckedChange={setStripExtension}
                  aria-label="Strip file extensions"
                />
                <Label htmlFor="strip-ext" className="text-xs text-muted-foreground">
                  strip{" "}
                  <span className="font-mono text-[11px]">.pdbqt</span> /
                  extensions
                </Label>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="lowercase"
                  checked={lowercase}
                  onCheckedChange={setLowercase}
                  aria-label="Fold names to lowercase"
                />
                <Label htmlFor="lowercase" className="text-xs text-muted-foreground">
                  case-insensitive
                </Label>
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="col-name" className="sr-only">
                  Column name
                </Label>
                <Input
                  id="col-name"
                  value={nameColumn}
                  onChange={(event) => setNameColumn(event.target.value.slice(0, 24) || "Ligand")}
                  className="h-8 w-28 font-mono text-xs"
                  aria-label="Compared column name (display only)"
                />
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={loadSample}>
                <ListChecks className="size-4" aria-hidden />
                Load sample
              </Button>
              <Button variant="ghost" size="sm" onClick={clearAll}>
                <Eraser className="size-4" aria-hidden />
                Clear
              </Button>
            </div>
          </div>

          <div className="flex flex-wrap gap-2 rounded-lg border bg-muted/30 p-3">
            <StatChip icon={CircleDot} value={String(result.common.length)} label="common" />
            <StatChip icon={ArrowLeftRight} value={String(result.onlyA.length)} label="only in A" />
            <StatChip icon={ArrowLeftRight} value={String(result.onlyB.length)} label="only in B" />
            <StatChip
              icon={GitCompare}
              value={String(total)}
              label="unique names"
            />
          </div>

          {/* Proportional overlap bar — one emerald/amber/rose segment per
           * bucket, sized by share of the unique-name union. */}
          {total > 0 && (
            <div className="grid gap-1.5">
              <div
                className="flex h-3 w-full overflow-hidden rounded-full border bg-muted/50"
                role="img"
                aria-label={`Overlap: ${result.common.length} common, ${result.onlyA.length} only in A, ${result.onlyB.length} only in B`}
              >
                <div
                  className="h-full bg-emerald-500/80 transition-[width] duration-500"
                  style={{ width: `${(result.common.length / total) * 100}%` }}
                  title={`common · ${result.common.length} (${Math.round((result.common.length / total) * 100)}%)`}
                />
                <div
                  className="h-full bg-amber-500/80 transition-[width] duration-500"
                  style={{ width: `${(result.onlyA.length / total) * 100}%` }}
                  title={`only in A · ${result.onlyA.length} (${Math.round((result.onlyA.length / total) * 100)}%)`}
                />
                <div
                  className="h-full bg-rose-500/80 transition-[width] duration-500"
                  style={{ width: `${(result.onlyB.length / total) * 100}%` }}
                  title={`only in B · ${result.onlyB.length} (${Math.round((result.onlyB.length / total) * 100)}%)`}
                />
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full bg-emerald-500/80" aria-hidden />
                  {Math.round((result.common.length / total) * 100)}% both
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full bg-amber-500/80" aria-hidden />
                  {Math.round((result.onlyA.length / total) * 100)}% only A
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full bg-rose-500/80" aria-hidden />
                  {Math.round((result.onlyB.length / total) * 100)}% only B
                </span>
              </div>
            </div>
          )}

          <div className="grid gap-3 lg:grid-cols-3">
            <ResultColumn
              title="Common"
              hint="docked AND in the library"
              items={result.common}
              accent="text-emerald-600 dark:text-emerald-400"
              emptyText="No overlap between the two sets"
            />
            <ResultColumn
              title="Only in A"
              hint="docked but missing from the library"
              items={result.onlyA}
              accent="text-amber-600 dark:text-amber-400"
              emptyText="Every docked ligand is in the library"
            />
            <ResultColumn
              title="Only in B"
              hint="in the library but never docked"
              items={result.onlyB}
              accent="text-rose-600 dark:text-rose-400"
              emptyText="Every library ligand was docked"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
