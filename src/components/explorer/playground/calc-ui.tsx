"use client";

import type { ReactNode } from "react";
import { Check, Copy, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/format";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Numeric input field                                                 */
/* ------------------------------------------------------------------ */

interface NumberFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  unit?: string;
  error?: string | null;
  placeholder?: string;
  optional?: boolean;
  className?: string;
}

/** Labelled decimal input with inline validation error and unit suffix. */
export function NumberField({
  id,
  label,
  value,
  onChange,
  unit,
  error,
  placeholder,
  optional,
  className,
}: NumberFieldProps) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <Label htmlFor={id} className="flex items-baseline gap-1 text-xs font-medium">
        <span>{label}</span>
        {optional ? (
          <span className="text-[10px] font-normal text-muted-foreground">(optional)</span>
        ) : null}
        {unit ? <span className="text-[10px] text-muted-foreground">{unit}</span> : null}
      </Label>
      <div className="relative">
        <Input
          id={id}
          type="text"
          inputMode="decimal"
          autoComplete="off"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? `${id}-error` : undefined}
          className={cn(
            "h-10 font-mono text-sm tabular-nums",
            error && "border-destructive/60 focus-visible:ring-destructive/30",
            unit && "pr-12"
          )}
        />
        {unit ? (
          <span
            className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-[10px] text-muted-foreground"
            aria-hidden
          >
            {unit}
          </span>
        ) : null}
      </div>
      {error ? (
        <p id={`${id}-error`} className="flex items-center gap-1 text-[11px] text-destructive">
          <TriangleAlert className="size-3 shrink-0" aria-hidden />
          {error}
        </p>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Result block                                                        */
/* ------------------------------------------------------------------ */

interface ResultBlockProps {
  label: string;
  value: string | null;
  unit?: string;
  formula?: string;
  copyText?: string | null;
  error?: string | null;
  children?: ReactNode;
}

/** Big mono value + formula + copy button. Renders an error line when invalid. */
export function ResultBlock({
  label,
  value,
  unit,
  formula,
  copyText,
  error,
  children,
}: ResultBlockProps) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    if (!copyText) return;
    const ok = await copyToClipboard(copyText);
    if (ok) {
      setCopied(true);
      toast.success(`Copied: ${copyText}`);
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  return (
    <div
      className={cn(
        "rounded-lg border bg-muted/30 p-3",
        error && "border-destructive/40 bg-destructive/5"
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        {copyText ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 gap-1 px-2 text-[11px]"
            onClick={handleCopy}
            aria-label={`Copy ${label} result`}
          >
            {copied ? (
              <Check className="size-3 text-primary" aria-hidden />
            ) : (
              <Copy className="size-3" aria-hidden />
            )}
            Copy
          </Button>
        ) : null}
      </div>

      {error ? (
        <p className="mt-1 flex items-center gap-1.5 text-sm text-destructive">
          <TriangleAlert className="size-3.5 shrink-0" aria-hidden />
          {error}
        </p>
      ) : (
        <p className="mt-0.5 font-mono text-2xl font-semibold tabular-nums">
          {value ?? "—"}
          {unit ? <span className="ml-1.5 text-sm font-normal text-muted-foreground">{unit}</span> : null}
        </p>
      )}

      {children}

      {formula ? (
        <p className="mt-2 font-mono text-[11px] leading-relaxed text-muted-foreground" aria-label="Formula used">
          {formula}
        </p>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Recharts shared styling (no blue/indigo anywhere)                   */
/* ------------------------------------------------------------------ */

export const CHART = {
  emerald: "#10b981",
  emeraldDark: "#059669",
  amber: "#f59e0b",
  rose: "#f43f5e",
  teal: "#14b8a6",
  stone: "#78716c",
  axisTick: { fill: "#78716c", fontSize: 11 } as const,
  gridStroke: "#78716c",
  gridStrokeOpacity: 0.18,
  tooltipStyle: {
    backgroundColor: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: "8px",
    color: "var(--popover-foreground)",
    fontSize: "12px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
  } as const,
} as const;
