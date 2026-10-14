"use client";

import { Activity, Brain, Calculator, ScanSearch } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PharmaCalcs } from "@/components/explorer/playground/pharma-calcs";
import { MlPlayground } from "@/components/explorer/playground/ml-playground";
import { DockingParser } from "@/components/explorer/playground/docking-parser";
import { PkSimulator } from "@/components/explorer/playground/pk-simulator";
import { useToolkitStore } from "@/lib/store";

/** Playground: pharma calculators / PK simulator / ML training / docking parser.
 * The active sub-tab and the PK regimen live in the store so both survive
 * switching away and back. */
export function Playground() {
  const playgroundTab = useToolkitStore((s) => s.playgroundTab);
  const setPlaygroundTab = useToolkitStore((s) => s.setPlaygroundTab);

  return (
    <Tabs
      value={playgroundTab}
      onValueChange={setPlaygroundTab}
      className="w-full"
    >
      <TabsList className="thin-scrollbar mx-auto grid h-auto w-full max-w-3xl grid-cols-2 gap-1 overflow-x-auto p-1 sm:grid-cols-4">
        <TabsTrigger
          value="calculators"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-4 sm:text-sm"
        >
          <Calculator className="size-4 shrink-0" aria-hidden />
          Calculators
        </TabsTrigger>
        <TabsTrigger
          value="pk"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-4 sm:text-sm"
        >
          <Activity className="size-4 shrink-0" aria-hidden />
          PK Simulator
        </TabsTrigger>
        <TabsTrigger
          value="ml"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-4 sm:text-sm"
        >
          <Brain className="size-4 shrink-0" aria-hidden />
          ML Training
        </TabsTrigger>
        <TabsTrigger
          value="docking"
          className="min-h-11 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm sm:px-4 sm:text-sm"
        >
          <ScanSearch className="size-4 shrink-0" aria-hidden />
          Docking Parser
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
      <TabsContent value="docking" className="mt-4 sm:mt-6">
        <DockingParser />
      </TabsContent>
    </Tabs>
  );
}
