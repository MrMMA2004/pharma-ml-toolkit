/**
 * part16 §10 — the compact Settings entry point.
 *
 * "A compact Settings entry point that is accessible throughout the
 * desktop application" — the header button (present on every
 * workspace) opening a popover panel, never a full-screen settings
 * page. The popover closes on Escape and on outside pointer-down,
 * and aria-expanded carries the state to assistive tech.
 */

import { useEffect, useRef, useState } from "react";
import { Settings as SettingsIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { SettingsPanel } from "@/components/settings/SettingsPanel";

export function SettingsEntry() {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const onPointerDown = (event: PointerEvent): void => {
      if (wrapperRef.current?.contains(event.target as Node) !== true) {
        setOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    dialogRef.current?.focus();
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={wrapperRef} className="relative">
      <Button
        variant="ghost"
        size="icon"
        aria-label="Settings"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => {
          setOpen((value) => !value);
        }}
      >
        <SettingsIcon className="h-4 w-4" aria-hidden="true" />
      </Button>
      {open ? (
        <div
          ref={dialogRef}
          role="dialog"
          aria-label="Application settings"
          tabIndex={-1}
          className="absolute right-0 top-[calc(100%+0.5rem)] z-50 max-h-[min(32rem,80vh)] w-[min(24rem,calc(100vw-2rem))] overflow-y-auto rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4 text-[var(--color-foreground)] shadow-xl focus:outline-none"
        >
          <SettingsPanel />
        </div>
      ) : null}
    </div>
  );
}
