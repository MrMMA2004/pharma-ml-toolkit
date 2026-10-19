"use client";

import type { ReactNode } from "react";
import { fuzzyMatchIndices } from "@/lib/format";
import { cn } from "@/lib/utils";

interface HighlightTextProps {
  text: string;
  /** Fuzzy query — matched characters render emphasized. */
  query: string;
  className?: string;
}

/**
 * Renders `text` with the characters matched by the (subsequence fuzzy)
 * query visually emphasized via <mark>. Non-matching / empty queries render
 * the plain text.
 */
export function HighlightText({ text, query, className }: HighlightTextProps) {
  const trimmed = query.trim();
  const indices = trimmed.length > 0 ? fuzzyMatchIndices(trimmed, text) : null;
  if (!indices || indices.length === 0) {
    return <span className={className}>{text}</span>;
  }
  const matched = new Set(indices);
  const nodes: ReactNode[] = [];
  let plain = "";
  let key = 0;
  const flush = () => {
    if (plain.length > 0) {
      nodes.push(<span key={`p-${key++}`}>{plain}</span>);
      plain = "";
    }
  };
  for (let i = 0; i < text.length; i++) {
    if (matched.has(i)) {
      flush();
      nodes.push(
        <mark
          key={`m-${key++}`}
          className="rounded-[2px] bg-primary/15 px-px font-semibold text-primary"
        >
          {text[i]}
        </mark>
      );
    } else {
      plain += text[i];
    }
  }
  flush();
  return <span className={cn(className)}>{nodes}</span>;
}
