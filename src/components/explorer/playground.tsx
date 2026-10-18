"use client";

import {
  Activity,
  Boxes,
  Brain,
  Calculator,
  FlaskConical,
  GitCompare,
  Microscope,
  ScanSearch,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PharmaCalcs } from "@/components/explorer/playground/pharma-calcs";
import { MlPlayground } from "@/components/explorer/playground/ml-playground";
import { BiologyPlayground } from "@/components/explorer/playground/biology-playground";
import { DockingParser } from "@/components/explorer/playground/docking-parser";
import { FeatureFactory } from "@/components/explorer/playground/feature-factory";
import { LigandPrep } from "@/components/explorer/playground/ligand-prep";
import { PkSimulator } from "@/components/explorer/playground/pk-simulator";
import { LigandReconciler } from "@/components/explorer/playground/ligand-reconciler";
import { useToolkitStore } from "@/lib/store";

/** Playground: pharma calculators / PK-PD explorer / ML training / cell line
 * biology / docking parser + ligand prep / feature factory / ligand-set
 * reconciler. The active sub-tab and the PK/PD state live in the store so
 * both survive switching away and back. */
export function Playground() {
  const playgroundTab = useToolkitStore((s) => s.playgroundTab);
  const setPlaygroundTab = useToolkitStore((s) => s.setPlaygroundTab);

  return (
    <Tabs
      value={playgroundTab}
      onValueChange={setPlaygroundTab}
      className="w-full"
    >
      <TabsList className="thin-scrollbar mx-auto grid h-auto w-full max-w-3xl grid-cols-2 gap-1 overflow-x-auto p-1 sm:grid-cols-4 lg:grid-cols-8">
        <TabsTrigger
          value="calculators"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <Calculator className="size-4 shrink-0" aria-hidden />
          Calculators
        </TabsTrigger>
        <TabsTrigger
          value="pk"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <Activity className="size-4 shrink-0" aria-hidden />
          PK / PD
        </TabsTrigger>
        <TabsTrigger
          value="ml"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <Brain className="size-4 shrink-0" aria-hidden />
          ML Training
        </TabsTrigger>
        <TabsTrigger
          value="biology"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <Microscope className="size-4 shrink-0" aria-hidden />
          Cell Line Bio
        </TabsTrigger>
        <TabsTrigger
          value="docking"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <ScanSearch className="size-4 shrink-0" aria-hidden />
          Docking Parser
        </TabsTrigger>
        <TabsTrigger
          value="features"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <FlaskConical className="size-4 shrink-0" aria-hidden />
          Features
        </TabsTrigger>
        <TabsTrigger
          value="ligandprep"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <Boxes className="size-4 shrink-0" aria-hidden />
          Ligand Prep
        </TabsTrigger>
        <TabsTrigger
          value="reconciler"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-3 sm:text-sm"
        >
          <GitCompare className="size-4 shrink-0" aria-hidden />
          Ligand Sets
        </TabsTrigger>
      </TabsList>
      <TabsContent value="calculators" className="mt-4 sm:mt-6">
        <PharmaCalcs />
      </TabsContent>
      <TabsContent value="pk" className="mt-4 sm:mt-6">
        <PkSimulator />
      </TabsContent>
      <TabsContent value="ml" className="mt-4 sm:mt-6">
        <MlPlayground />
      </TabsContent>
      <TabsContent value="biology" className="mt-4 sm:mt-6">
        <BiologyPlayground />
      </TabsContent>
      <TabsContent value="docking" className="mt-4 sm:mt-6">
        <DockingParser />
      </TabsContent>
      <TabsContent value="features" className="mt-4 sm:mt-6">
        <FeatureFactory />
      </TabsContent>
      <TabsContent value="ligandprep" className="mt-4 sm:mt-6">
        <LigandPrep />
      </TabsContent>
      <TabsContent value="reconciler" className="mt-4 sm:mt-6">
        <LigandReconciler />
      </TabsContent>
    </Tabs>
  );
}
