"use client";

import { useState } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import toml from "react-syntax-highlighter/dist/esm/languages/prism/toml";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import docker from "react-syntax-highlighter/dist/esm/languages/prism/docker";
import ini from "react-syntax-highlighter/dist/esm/languages/prism/ini";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import oneDark from "react-syntax-highlighter/dist/esm/styles/prism/one-dark";
import oneLight from "react-syntax-highlighter/dist/esm/styles/prism/one-light";
import { useTheme } from "next-themes";
import { Check, Copy, FileCode2 } from "lucide-react";
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

/** Grammar ids registered with Prism — anything else renders as plain text. */
const REGISTERED_GRAMMARS = new Set([
  "python",
  "toml",
  "yaml",
  "markdown",
  "docker",
  "ini",
  "bash",
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
  className?: string;
}

export function CodeBlock({
  code,
  language = "text",
  fileName,
  showLineNumbers = false,
  maxHeightClass = "max-h-[70vh]",
  className,
}: CodeBlockProps) {
  const { resolvedTheme } = useTheme();
  const [copied, setCopied] = useState(false);

  const style = resolvedTheme === "dark" ? oneDark : oneLight;
  const grammar = REGISTERED_GRAMMARS.has(language) ? language : undefined;
  const label = LANGUAGE_LABELS[language] ?? language;

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
      <div className={cn("thin-scrollbar overflow-auto", maxHeightClass)}>
        <SyntaxHighlighter
          language={grammar}
          style={style}
          showLineNumbers={showLineNumbers}
          wrapLongLines={false}
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
