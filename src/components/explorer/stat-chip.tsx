"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatChipProps {
  icon: LucideIcon;
  value: string;
  label: string;
  hint?: string;
  className?: string;
  /** When provided, the value counts up from 0 to this number on mount. */
  countTo?: number;
}

/**
 * Numeric value that counts up from 0 on mount (rAF, cubic ease-out).
 * Renders the final number first so SSR/hydration match, then animates.
 * Honors prefers-reduced-motion by staying at the final value.
 */
function AnimatedNumber({ target }: { target: number }) {
  const [display, setDisplay] = useState(target);

  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return; // keep the final value — no animation
    }
    const duration = 900;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setDisplay(Math.round(target * eased));
      if (t < 1) raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [target]);

  return <span className="tabular-nums">{display.toLocaleString("en-US")}</span>;
}

/** Compact stat card used in the hero and download panel. */
export function StatChip({
  icon: Icon,
  value,
  label,
  hint,
  className,
  countTo,
}: StatChipProps) {
  return (
    <motion.div
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={cn(
        "group flex items-center gap-3 rounded-lg border bg-card px-3 py-2.5 shadow-sm transition-colors hover:border-primary/40",
        className
      )}
    >
      <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary transition-all duration-200 group-hover:scale-105 group-hover:bg-primary/15">
        <Icon className="size-4.5" aria-hidden />
      </span>
      <div className="min-w-0">
        <p
          className="break-words text-lg font-semibold leading-tight tabular-nums"
          title={value}
        >
          {typeof countTo === "number" && Number.isFinite(countTo) ? (
            <AnimatedNumber target={countTo} />
          ) : (
            value
          )}
        </p>
        <p className="break-words text-xs leading-snug text-muted-foreground" title={hint ?? label}>
          {label}
        </p>
      </div>
    </motion.div>
  );
}
