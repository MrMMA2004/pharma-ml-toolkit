/**
 * The React binding for the settings store — part16 §10.
 *
 * The house pattern (part11 §8.2): `useSyncExternalStore` over the
 * framework-free singleton, no new state-management dependency.
 * `useSettings()` re-renders its consumer on every committed change;
 * updates go through the store's validate-before-apply `update()`.
 */

import { useSyncExternalStore } from "react";

import { getSettingsStore } from "@/services/settings/store";
import type { AppSettings } from "@/services/settings/types";

/** The live application settings. */
export function useSettings(): AppSettings {
  const store = getSettingsStore();
  return useSyncExternalStore(
    (listener) => store.subscribe(listener),
    () => store.snapshot(),
    () => store.snapshot(),
  );
}
