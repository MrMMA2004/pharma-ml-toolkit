"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatChipProps {
  icon: LucideIcon;
  value: string;
  label: string;
  hint?: string;
  className?: string;
}

/** Compact stat card used in the hero and download panel. */
export function StatChip({ icon: Icon, value, label, hint, className }: StatChipProps) {
  return (
    <motion.div
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={cn(
        "flex items-center gap-3 rounded-lg border bg-card px-3 py-2.5 shadow-sm",
        className
      )}
    >
      <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
        <Icon className="size-4.5" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="break-words text-lg font-semibold leading-tight tabular-nums" title={value}>
          {value}
        </p>
        <p className="break-words text-xs leading-snug text-muted-foreground" title={hint ?? label}>
          {label}
        </p>
      </div>
    </motion.div>
  );
}
