"use client";

import { useMemo, useState } from "react";
import {
  Braces,
  ChevronDown,
  Hash,
  ListTree,
  Parentheses,
  SquareFunction,
} from "lucide-react";
import { motion } from "framer-motion";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  extractOutline,
  OUTLINE_KIND_LABEL,
  type OutlineKind,
  type OutlineSymbol,
} from "@/lib/outline";

/** Minimum symbols before the outline panel is worth rendering. */
const MIN_SYMBOLS = 3;

const KIND_ICON: Record<OutlineKind, typeof Braces> = {
  class: Braces,
  function: SquareFunction,
  method: Parentheses,
  heading: Hash,
  section: ChevronDown,
};

interface FileOutlineProps {
  code: string;
  language: string;
  /** Called with the 1-based line to jump to. */
  onJump: (line: number) => void;
}

/**
 * Collapsible table of contents for the currently open file — Python
 * classes/functions, Markdown headings, TOML/YAML sections. Clicking a
 * symbol scrolls the code pane to that line and pins the highlight.
 */
export function FileOutline({ code, language, onJump }: FileOutlineProps) {
  const [open, setOpen] = useState(true);
  const symbols = useMemo(() => extractOutline(code, language), [code, language]);

  if (symbols.length < MIN_SYMBOLS) return null;

  const counts = symbols.reduce<Record<string, number>>((acc, s) => {
    acc[s.kind] = (acc[s.kind] ?? 0) + 1;
    return acc;
  }, {});
  const kindSummary = (
    ["class", "function", "method", "heading", "section"] as OutlineKind[]
  )
    .filter((k) => counts[k])
    .map((k) => `${counts[k]} ${OUTLINE_KIND_LABEL[k]}${counts[k] === 1 ? "" : "s"}`)
    .join(" · ");

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-lg border bg-muted/20 shadow-sm"
    >
      <CollapsibleTrigger
        className="group flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left transition-colors hover:bg-primary/5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        aria-label={open ? "Collapse file outline" : "Expand file outline"}
      >
        <ListTree className="size-4 shrink-0 text-primary" aria-hidden />
        <span className="text-sm font-medium">Outline</span>
        <Badge variant="secondary" className="shrink-0 tabular-nums">
          {symbols.length}
        </Badge>
        <span className="hidden min-w-0 flex-1 truncate text-xs text-muted-foreground sm:block">
          {kindSummary}
        </span>
        <ChevronDown
          className="ml-auto size-4 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180"
          aria-hidden
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <nav
          aria-label="Symbol outline"
          className="thin-scrollbar max-h-52 overflow-y-auto border-t px-1.5 py-1.5"
        >
          <ul className="space-y-0.5">
            {symbols.map((s, i) => (
              <OutlineRow key={`${s.line}-${s.name}`} symbol={s} onJump={onJump} index={i} />
            ))}
          </ul>
        </nav>
      </CollapsibleContent>
    </Collapsible>
  );
}

function OutlineRow({
  symbol,
  onJump,
  index,
}: {
  symbol: OutlineSymbol;
  onJump: (line: number) => void;
  index: number;
}) {
  const Icon = KIND_ICON[symbol.kind];
  const tone =
    symbol.kind === "class"
      ? "text-primary"
      : symbol.kind === "function" || symbol.kind === "method"
        ? "text-foreground/70"
        : "text-muted-foreground";
  return (
    <li>
      <motion.button
        type="button"
        initial={{ opacity: 0, x: -4 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: Math.min(index * 0.012, 0.25), duration: 0.15 }}
        onClick={() => onJump(symbol.line)}
        aria-label={`Jump to ${OUTLINE_KIND_LABEL[symbol.kind]} ${symbol.name} on line ${symbol.line}`}
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-xs transition-colors",
          "hover:bg-primary/8 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          symbol.kind === "class" && "font-mono font-semibold",
          (symbol.kind === "function" || symbol.kind === "method") && "font-mono",
          symbol.kind === "heading" && "font-sans",
          symbol.kind === "section" && "font-mono text-muted-foreground"
        )}
        style={{ paddingLeft: `${0.5 + symbol.depth * 0.75}rem` }}
      >
        <Icon className={cn("size-3.5 shrink-0", tone)} aria-hidden />
        <span className="min-w-0 flex-1 truncate">{symbol.name}</span>
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground/70">
          {symbol.line}
        </span>
      </motion.button>
    </li>
  );
}
