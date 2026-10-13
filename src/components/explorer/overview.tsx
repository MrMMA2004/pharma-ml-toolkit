"use client";

import { useEffect } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BookOpen,
  Boxes,
  Brain,
  CircleCheck,
  Download,
  Dna,
  FileArchive,
  FileCode2,
  Pill,
  ShieldCheck,
  Sigma,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CodeBlock } from "@/components/explorer/code-block";
import { StatChip } from "@/components/explorer/stat-chip";
import { useToolkitStore } from "@/lib/store";
import { fmtInt } from "@/lib/format";

const QUICKSTART = `# --- Pharmaceutical calculations -------------------------------
from pharmaml.pharma import (
    creatinine_clearance, bsa_mosteller,
    loading_dose, maintenance_dose, auc_trapezoid,
)
import numpy as np

crcl = creatinine_clearance(age_years=68, weight_kg=70,
                            serum_cr_mg_dl=1.1, sex="male")
bsa  = bsa_mosteller(height_cm=175, weight_kg=70)      # m^2
ld   = loading_dose(vd=45.0, target_concentration=12.0)  # mg
md   = maintenance_dose(clearance_=3.2, target_css=6.0, tau=8.0)

t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
c = np.array([0.0, 8.2, 6.1, 3.4, 1.1, 0.35])
auc = auc_trapezoid(t, c)                              # ug*h/mL

# --- QSAR machine learning --------------------------------------
from pharmaml.ml import (
    train_model, evaluate_model, compare_models,
    make_train_test_split,
)

X_tr, X_te, y_tr, y_te = make_train_test_split(X, y, test_size=0.2)
result = train_model("random_forest", "classification",
                     X_tr, y_tr, cv=5, tune=True)
print(result.cv_mean, result.cv_std)      # 5-fold ROC-AUC mean +/- std
report = evaluate_model(result.model, X_te, y_te, "classification")

# --- Docking -----------------------------------------------------
from pharmaml.docking import (
    parse_vina_output, parse_dlg_cluster_table,
    ligand_efficiency, consensus_rescore,
)

poses = parse_vina_output(open("vina.out").read())
le    = ligand_efficiency(affinity_kcal_mol=-9.6, num_heavy_atoms=32)
cons  = consensus_rescore(scores_df, columns=["vina", "ad4"])`;

interface DomainCardSpec {
  icon: LucideIcon;
  title: string;
  tagline: string;
  bullets: string[];
  apis: string[];
  accent: string;
}

const DOMAINS: DomainCardSpec[] = [
  {
    icon: Brain,
    title: "Machine Learning",
    tagline: "pharmaml.ml — QSAR end-to-end",
    bullets: [
      "Cross-validated training with optional grid-search tuning",
      "15-entry model registry: sklearn core + optional XGBoost / LightGBM / CatBoost",
      "Metrics, comparison leaderboards & joblib persistence",
    ],
    apis: ["train_model()", "compare_models()", "list_models()"],
    accent: "bg-primary/10 text-primary",
  },
  {
    icon: Pill,
    title: "Pharma Calculations",
    tagline: "pharmaml.pharma — PK · PD · dosage",
    bullets: [
      "Non-compartmental PK: trapezoidal AUC, clearance, half-life",
      "Renal function (Cockcroft-Gault) & BSA-based dosing",
      "Loading / maintenance regimen design with bioavailability",
    ],
    apis: ["creatinine_clearance()", "auc_trapezoid()", "maintenance_dose()"],
    accent: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  },
  {
    icon: Boxes,
    title: "Molecular Docking",
    tagline: "pharmaml.docking — AutoDock4 & Vina",
    bullets: [
      "Grid-box & GPF preparation, batch run orchestration",
      "Pure-Python Vina output & DLG cluster-table parsers",
      "Ligand efficiency + consensus z-score rescoring",
    ],
    apis: ["parse_vina_output()", "write_gpf()", "consensus_rescore()"],
    accent: "bg-teal-500/10 text-teal-600 dark:text-teal-400",
  },
  {
    icon: Dna,
    title: "Cheminformatics",
    tagline: "pharmaml.chem — lazy RDKit extras",
    bullets: [
      "IC50 / Ki → pIC50 / pKi conversions with unit handling",
      "RDKit descriptors & fingerprints behind lazy imports",
      "Tanimoto similarity scans across compound tables",
    ],
    apis: ["ic50_to_pic50()", "tanimoto_from_sets()", "similarity_scan()"],
    accent: "bg-rose-500/10 text-rose-600 dark:text-rose-400",
  },
];

const UPGRADE_ROWS: Array<{ aspect: string; before: string; after: string }> = [
  {
    aspect: "Structure",
    before: "30+ flat, near-duplicate scripts in one folder",
    after: "Typed package pharmaml with 5 subpackages + CLI",
  },
  {
    aspect: "Typing",
    before: "No type hints, no docstrings",
    after: "Full annotations (Python ≥ 3.10 syntax) + Google docstrings",
  },
  {
    aspect: "Logging",
    before: "Bare print() with mixed Persian/English messages",
    after: "Structured logger — console + optional file handler",
  },
  {
    aspect: "Paths",
    before: "Hardcoded C:\\Users\\... Windows paths, os.chdir side effects",
    after: "ToolConfig paths, CLI arguments, no working-directory mutation",
  },
  {
    aspect: "Testing",
    before: "0 tests — changes verified by eyeball",
    after: "159 passing pytest cases across every domain",
  },
  {
    aspect: "Distribution",
    before: "Copy-paste between machines",
    after: "pip-installable extras, Dockerfile, GitHub Actions CI matrix",
  },
];

const DESIGN_CHOICES: Array<{ title: string; rationale: string }> = [
  {
    title: "scikit-learn as the ML backbone",
    rationale:
      "QSAR tabular data is the sweet spot for sklearn — the registry covers every family from logistic to random forests, and ann (MLPClassifier) handles the neural case. Trade-off: no GPU training for tabular models, which is rarely needed at this data scale.",
  },
  {
    title: "Lazily-imported optional dependencies",
    rationale:
      "The package imports cleanly on a bare Python + numpy/pandas/sklearn environment; RDKit, Mordred, XGBoost/LightGBM/CatBoost and Meeko fail at call time with an informative DependencyError including install hints. Trade-off: a slightly more complex import path, worth it for portability.",
  },
  {
    title: "Pure-Python docking parsers",
    rationale:
      "DLG / Vina / PDBQT parsing has zero external dependencies, so result analysis works even on machines without AutoDock installed. Execution wrappers locate binaries via shutil.which and raise DockingError with guidance instead of shell=True surprises.",
  },
  {
    title: "Explicit unit conventions over a unit library",
    rationale:
      "PK functions document their units in the docstring instead of pulling in pint. Trade-off: caller responsibility — but zero-dependency math functions are trivially testable and fast.",
  },
  {
    title: "One shared validation layer",
    rationale:
      "pharmaml.utils.validation raises the same DataValidationError — with the parameter name in the message — from every domain, so one except PharmaMLError clause catches everything and error messages stay uniform.",
  },
];

export function Overview() {
  const setTab = useToolkitStore((s) => s.setTab);
  const ensureTree = useToolkitStore((s) => s.ensureTree);
  const stats = useToolkitStore((s) => s.stats);
  const treeStatus = useToolkitStore((s) => s.treeStatus);

  useEffect(() => {
    void ensureTree();
  }, [ensureTree]);

  const fileCount = stats ? fmtInt(stats.files) : "48";
  const loc = stats ? fmtInt(stats.totalLines) : "7,097";

  return (
    <div className="space-y-10">
      {/* ------------------------------ Hero ------------------------------ */}
      <section
        aria-labelledby="hero-title"
        className="hero-panel relative space-y-6 rounded-2xl border border-border/70 p-6 shadow-sm sm:p-10"
      >
        <div className="space-y-4">
          <Badge variant="secondary" className="gap-1.5 border border-primary/25 bg-primary/10 text-primary">
            <ShieldCheck className="size-3.5" aria-hidden />
            Python ≥ 3.10 · numpy / pandas / sklearn core
          </Badge>
          <h1
            id="hero-title"
            className="text-3xl font-bold tracking-tight text-balance sm:text-4xl lg:text-5xl"
          >
            One toolkit. Three scientific domains.
          </h1>
          <p className="max-w-2xl text-base text-muted-foreground sm:text-lg">
            A production-ready, typed, tested Python package for computational pharmacy — QSAR
            machine learning, pharmaceutical PK/PD &amp; dosage calculations, and molecular
            docking workflows, upgraded from a flat collection of research scripts.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatChip
            icon={FileCode2}
            value={treeStatus === "loading" ? "…" : fileCount}
            label="source files"
            hint="Whole package: src, tests, examples, CI"
          />
          <StatChip
            icon={Sigma}
            value={treeStatus === "loading" ? "…" : loc}
            label="lines of code"
            hint="Counted live from the file tree"
          />
          <StatChip
            icon={CircleCheck}
            value="159"
            label="passing tests"
            hint="pytest suite covering every domain"
          />
          <StatChip
            icon={Brain}
            value="15"
            label="registry models"
            hint="sklearn core + optional boosters"
          />
        </div>

        <div className="flex flex-wrap gap-3">
          <Button
            size="lg"
            className="h-11 gap-2"
            onClick={() => setTab("explorer")}
          >
            <BookOpen className="size-4" aria-hidden />
            Explore the code
            <ArrowRight className="size-4" aria-hidden />
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="h-11 gap-2"
            onClick={() => setTab("download")}
          >
            <FileArchive className="size-4" aria-hidden />
            Download ZIP
          </Button>
        </div>
      </section>

      {/* --------------------------- Domain cards --------------------------- */}
      <section aria-labelledby="domains-title" className="space-y-4">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">The package</p>
            <h2 id="domains-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
              Four subpackages, one import
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Every module ships with type hints, docstrings, structured logging and tests.
            </p>
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {DOMAINS.map((d, i) => (
            <motion.div
              key={d.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06, duration: 0.25 }}
              whileHover={{ y: -4 }}
              className="h-full"
            >
              <Card className="h-full border-border/80 shadow-sm">
                <CardHeader className="pb-3">
                  <span
                    className={`mb-2 flex size-10 items-center justify-center rounded-lg ${d.accent}`}
                  >
                    <d.icon className="size-5" aria-hidden />
                  </span>
                  <CardTitle className="text-base">{d.title}</CardTitle>
                  <CardDescription className="font-mono text-xs">{d.tagline}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 pb-4">
                  <ul className="space-y-1.5 text-sm text-muted-foreground">
                    {d.bullets.map((b) => (
                      <li key={b} className="flex gap-2">
                        <CircleCheck className="mt-0.5 size-3.5 shrink-0 text-primary" aria-hidden />
                        <span>{b}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {d.apis.map((api) => (
                      <code
                        key={api}
                        className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-foreground/80"
                      >
                        {api}
                      </code>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ------------------------- Upgrade comparison ------------------------- */}
      <section aria-labelledby="upgrade-title" className="space-y-4">
        <div>
          <p className="eyebrow">The re-engineering</p>
          <h2 id="upgrade-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            Before → After: the upgrade
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            What changed when the flat script collection was re-engineered into the{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">pharmaml</code>{" "}
            package.
          </p>
        </div>
        <Card className="overflow-hidden py-0 shadow-sm">
          <div className="thin-scrollbar overflow-x-auto">
            <Table className="data-table">
              <TableHeader>
                <TableRow className="bg-muted/40 hover:bg-muted/40">
                  <TableHead className="w-[110px] font-semibold">Aspect</TableHead>
                  <TableHead className="font-semibold text-rose-600 dark:text-rose-400">
                    Before — flat scripts
                  </TableHead>
                  <TableHead className="font-semibold text-primary">
                    After — pharmaml package
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {UPGRADE_ROWS.map((row) => (
                  <TableRow key={row.aspect}>
                    <TableCell className="whitespace-nowrap font-medium">{row.aspect}</TableCell>
                    <TableCell className="text-muted-foreground">{row.before}</TableCell>
                    <TableCell>{row.after}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>
      </section>

      {/* --------------------------- Design choices --------------------------- */}
      <section aria-labelledby="design-title" className="space-y-4">
        <div>
          <p className="eyebrow">Architecture rationale</p>
          <h2 id="design-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            Design choices &amp; trade-offs
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            The reasoning behind the architecture — straight from the README.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {DESIGN_CHOICES.map((choice, i) => (
            <motion.div
              key={choice.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.25 }}
              whileHover={{ y: -3 }}
            >
              <Card className="h-full border-border/80 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-start gap-2 text-sm leading-snug">
                    <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-[11px] font-semibold text-primary">
                      {i + 1}
                    </span>
                    {choice.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {choice.rationale}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ------------------------------ Quickstart ------------------------------ */}
      <section aria-labelledby="quickstart-title" className="space-y-4">
        <div>
          <p className="eyebrow">Get running</p>
          <h2 id="quickstart-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            Quickstart
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            One import per domain — deterministic examples, no external datasets required.
          </p>
        </div>
        <CodeBlock code={QUICKSTART} language="python" fileName="quickstart.py" showLineNumbers maxHeightClass="max-h-[520px]" />
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground sm:gap-x-4">
          <span>Want the full picture?</span>
          <Button variant="link" className="h-auto gap-1 p-0 text-sm" onClick={() => setTab("explorer")}>
            Browse the actual source <ArrowRight className="size-3.5" aria-hidden />
          </Button>
          <span aria-hidden className="hidden text-border sm:inline">·</span>
          <Button variant="link" className="h-auto gap-1 p-0 text-sm" onClick={() => setTab("reference")}>
            Browse the API reference <ArrowRight className="size-3.5" aria-hidden />
          </Button>
        </div>
      </section>

      {/* ------------------------------ Download CTA ------------------------------ */}
      <section
        aria-labelledby="dl-cta-title"
        className="rounded-xl border border-primary/20 bg-gradient-to-br from-primary/5 via-transparent to-amber-500/5 p-6 sm:p-8"
      >
        <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
          <div className="space-y-1">
            <h2 id="dl-cta-title" className="flex items-center gap-2 text-lg font-semibold">
              <Download className="size-5 text-primary" aria-hidden />
              Get the whole toolkit
            </h2>
            <p className="text-sm text-muted-foreground">
              Full source, tests, examples, Dockerfile and CI — packaged as a single ZIP.
            </p>
          </div>
          <Button size="lg" className="h-11 gap-2" onClick={() => setTab("download")}>
            <FileArchive className="size-4" aria-hidden />
            Go to download
          </Button>
        </div>
      </section>
    </div>
  );
}
