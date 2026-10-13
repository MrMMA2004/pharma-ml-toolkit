"use client";

import { useEffect, useRef, useState } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import toml from "react-syntax-highlighter/dist/esm/languages/prism/toml";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import docker from "react-syntax-highlighter/dist/esm/languages/prism/docker";
import ini from "react-syntax-highlighter/dist/esm/languages/prism/ini";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import oneDark from "react-syntax-highlighter/dist/esm/styles/prism/one-dark";
import oneLight from "react-syntax-highlighter/dist/esm/styles/prism/one-light";
import { useTheme } from "next-themes";
import { Check, Copy, FileCode2, Link2, LocateFixed, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/format";
import { toast } from "sonner";

// Register only the grammars the toolkit actually contains.
SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("toml", toml);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("docker", docker);
SyntaxHighlighter.registerLanguage("ini", ini);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("json", json);

/** Grammar ids registered with Prism — anything else renders as plain text. */
const REGISTERED_GRAMMARS = new Set([
  "python",
  "toml",
  "yaml",
  "markdown",
  "docker",
  "ini",
  "bash",
  "json",
]);

const LANGUAGE_LABELS: Record<string, string> = {
  python: "Python",
  toml: "TOML",
  yaml: "YAML",
  markdown: "Markdown",
  docker: "Dockerfile",
  ini: "INI",
  text: "Text",
  bash: "Shell",
  json: "JSON",
};

interface CodeBlockProps {
  code: string;
  /** Grammar id — unregistered values render as plain text. */
  language?: string;
  /** Optional file name shown in the header bar. */
  fileName?: string;
  showLineNumbers?: boolean;
  /** Tailwind max-height class for the scroll container. */
  maxHeightClass?: string;
  /** 1-based line to highlight and scroll to (definition deep links). */
  highlightLine?: number | null;
  /** Called when the user dismisses the line-highlight chip. */
  onClearHighlight?: () => void;
  /**
   * Toolkit-relative file path. When set (with line numbers enabled),
   * clicking any line copies "<path>#L<line>" to the clipboard.
   */
  permalinkPath?: string;
  /** Called (rAF-throttled) with the topmost visible 1-based line while the
   * pane scrolls — powers the outline's active-symbol scroll-spy. */
  onVisibleLineChange?: (line: number) => void;
  className?: string;
}

export function CodeBlock({
  code,
  language = "text",
  fileName,
  showLineNumbers = false,
  maxHeightClass = "max-h-[70vh]",
  highlightLine = null,
  onClearHighlight,
  permalinkPath,
  onVisibleLineChange,
  className,
}: CodeBlockProps) {
  const { resolvedTheme } = useTheme();
  const [copied, setCopied] = useState(false);
  const [copiedLine, setCopiedLine] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const style = resolvedTheme === "dark" ? oneDark : oneLight;
  const grammar = REGISTERED_GRAMMARS.has(language) ? language : undefined;
  const label = LANGUAGE_LABELS[language] ?? language;
  const hasHighlight = highlightLine !== null && highlightLine > 0;
  const permalink = Boolean(permalinkPath && showLineNumbers);
  const wrapLines = hasHighlight || permalink;

  /* Scroll the highlighted line to the middle of the pane once the code
   * for the current highlight target has rendered. */
  useEffect(() => {
    if (!hasHighlight) return;
    const timer = window.setTimeout(() => {
      const container = scrollRef.current;
      if (!container) return;
      const line = container.querySelector<HTMLElement>(`.cl-${highlightLine}`);
      if (!line) return;
      const target = Math.max(0, line.offsetTop - container.clientHeight / 2);
      container.scrollTo({ top: target, behavior: "smooth" });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [hasHighlight, highlightLine, code]);

  /* Scroll-spy: report the topmost fully-relevant visible line (rAF-throttled)
   * so the file outline can track the active symbol while scrolling. Only
   * active when the pane wraps lines (`.code-line` wrappers exist). */
  useEffect(() => {
    const container = scrollRef.current;
    if (!container || !onVisibleLineChange || !wrapLines) return;
    let raf = 0;
    const report = () => {
      raf = 0;
      const lines = container.querySelectorAll<HTMLElement>(".code-line");
      for (const line of lines) {
        // First line whose bottom edge sits below the viewport top (+4px slack).
        if (line.offsetTop + line.offsetHeight > container.scrollTop + 4) {
          const match = /(?:^|\s)cl-(\d+)(?:\s|$)/.exec(line.className);
          if (match) onVisibleLineChange(Number(match[1]));
          break;
        }
      }
    };
    const onScroll = () => {
      if (raf === 0) raf = window.requestAnimationFrame(report);
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    report(); // report the initial position (line 1) without scrolling
    return () => {
      container.removeEventListener("scroll", onScroll);
      if (raf !== 0) window.cancelAnimationFrame(raf);
    };
  }, [code, wrapLines, onVisibleLineChange]);

  async function handleCopy() {
    const ok = await copyToClipboard(code);
    if (ok) {
      setCopied(true);
      toast.success(fileName ? `Copied ${fileName}` : "Code copied to clipboard");
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  /** Extract the 1-based line number from a `.code-line.cl-N` wrapper. */
  function lineFromTarget(target: HTMLElement): number | null {
    const line = target.closest<HTMLElement>(".code-line");
    if (!line) return null;
    const match = /(?:^|\s)cl-(\d+)(?:\s|$)/.exec(line.className);
    return match ? Number(match[1]) : null;
  }

  /* Event delegation: any click inside a line (gutter included — the line
   * number span is a child of the wrapped line) copies a path#LN reference.
   * Skipped while the user is mid-selection so word/paragraph selection
   * never triggers a copy. */
  function handlePaneClick(event: React.MouseEvent<HTMLDivElement>) {
    if (!permalink || !permalinkPath) return;
    const selection = window.getSelection();
    if (selection && !selection.isCollapsed) return;
    const lineNumber = lineFromTarget(event.target as HTMLElement);
    if (lineNumber === null) return;
    const reference = `${permalinkPath}#L${lineNumber}`;
    void copyToClipboard(reference).then((ok) => {
      if (!ok) {
        toast.error("Could not copy — your browser blocked clipboard access");
        return;
      }
      setCopiedLine(lineNumber);
      window.setTimeout(
        () => setCopiedLine((cur) => (cur === lineNumber ? null : cur)),
        1400
      );
      toast.success("Line reference copied", { description: reference });
    });
  }

  return (
    <div
      className={cn(
        "overflow-hidden rounded-lg border bg-card focus-within:ring-1 focus-within:ring-ring",
        className
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b bg-muted/40 px-3 py-1.5">
        <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
          {fileName ? <FileCode2 className="size-3.5 shrink-0 text-primary" aria-hidden /> : null}
          <span className="truncate font-mono">{fileName ?? "snippet"}</span>
          <Badge variant="secondary" className="hidden shrink-0 font-mono text-[10px] sm:inline-flex">
            {label}
          </Badge>
          {permalink ? (
            <span
              className="hidden shrink-0 items-center gap-1 font-mono text-[10px] text-muted-foreground/70 md:inline-flex"
              title="Click any line (or its number) to copy a path#LN reference to the clipboard"
            >
              <Link2 className="size-3" aria-hidden />
              click a line to copy its reference
            </span>
          ) : null}
          {hasHighlight ? (
            <Badge
              variant="outline"
              className="shrink-0 gap-1 border-primary/30 bg-primary/5 font-mono text-[10px] text-primary tabular-nums"
            >
              <LocateFixed className="size-3" aria-hidden />
              line {highlightLine}
              {onClearHighlight ? (
                <button
                  type="button"
                  onClick={onClearHighlight}
                  aria-label="Dismiss line highlight"
                  className="-mr-1 flex size-4 items-center justify-center rounded-full transition-colors hover:bg-primary/15"
                >
                  <X className="size-2.5" aria-hidden />
                </button>
              ) : null}
            </Badge>
          ) : null}
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-8 shrink-0 gap-1.5 px-2 text-xs"
          onClick={handleCopy}
          aria-label={fileName ? `Copy ${fileName} to clipboard` : "Copy code to clipboard"}
        >
          {copied ? (
            <Check className="size-3.5 text-primary" aria-hidden />
          ) : (
            <Copy className="size-3.5" aria-hidden />
          )}
          <span className="hidden sm:inline">{copied ? "Copied" : "Copy"}</span>
        </Button>
      </div>
      <div
        ref={scrollRef}
        onClick={permalink ? handlePaneClick : undefined}
        className={cn(
          "thin-scrollbar relative overflow-auto",
          maxHeightClass,
          permalink && "code-permalink"
        )}
      >
        <SyntaxHighlighter
          language={grammar}
          style={style}
          showLineNumbers={showLineNumbers}
          wrapLongLines={false}
          wrapLines={wrapLines}
          lineProps={
            wrapLines
              ? (lineNumber: number) => ({
                  className: cn(
                    "code-line",
                    `cl-${lineNumber}`,
                    lineNumber === highlightLine && "code-line-highlight",
                    lineNumber === copiedLine && "code-line-copied"
                  ),
                })
              : undefined
          }
          customStyle={{
            background: "transparent",
            margin: 0,
            padding: "0.875rem 1rem",
            fontSize: "0.8125rem",
            lineHeight: 1.6,
          }}
          codeTagProps={{ style: { fontFamily: "var(--font-geist-mono)" } }}
          lineNumberStyle={{
            minWidth: "2.25em",
            paddingRight: "0.75em",
            opacity: 0.45,
            fontSize: "0.75rem",
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
