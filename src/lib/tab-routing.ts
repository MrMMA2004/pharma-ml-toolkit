"use client";

import { useEffect } from "react";

import { TABS, useToolkitStore } from "@/lib/store";
import type { TabId } from "@/lib/store";

/* r268 (review M-7): hash-based deep links for the Explorer tabs.
 *
 * The Explorer was a single URL with client-state tabs — sharing "look
 * at my ML run" was impossible and browser Back exited the app.  This
 * hook binds the tab (and the playground sub-tab) to the location hash:
 *
 *   #/playground            — the playground
 *   #/playground/ml         — the ML run, sub-tab included
 *   #/explorer              — the code explorer
 *
 * Rules:
 *  - the hash is the single source of truth ON LOAD; afterwards store
 *    changes write the hash (replaceState so tab clicks don't spam
 *    history) while hashchange (Back/Forward, manual paste, in-page
 *    anchors from other sites) drives the store.
 *  - unknown hashes fall back to the overview tab.
 *  - the initial render never rewrites an explicitly shared URL.
 */

const TAB_IDS: readonly TabId[] = TABS.map((t) => t.id);

export function tabFromHash(hash: string): TabId {
  const id = hash.replace(/^#\/?/, "").split("/")[0];
  return (TAB_IDS as readonly string[]).includes(id) ? (id as TabId) : "overview";
}

function subTabFromHash(hash: string): string | undefined {
  const parts = hash.replace(/^#\/?/, "").split("/");
  const sub = parts.length > 1 && parts[1] ? parts[1] : undefined;
  return sub;
}

export function useTabRouting() {
  const tab = useToolkitStore((s) => s.tab);
  const setTab = useToolkitStore((s) => s.setTab);
  const playgroundTab = useToolkitStore((s) => s.playgroundTab);
  const setPlaygroundTab = useToolkitStore((s) => s.setPlaygroundTab);

  /* ON LOAD: adopt the hash (a shared link wins over the default). */
  useEffect(() => {
    const fromHash = tabFromHash(window.location.hash);
    const sub = subTabFromHash(window.location.hash);
    if (fromHash !== "overview" || sub) {
      if (fromHash !== tab) setTab(fromHash);
      if (fromHash === "playground" && sub && sub !== playgroundTab) {
        setPlaygroundTab(sub);
      }
    }
  }, []);

  /* STORE -> HASH: tab switches update the URL without adding history
   * entries (Back means "previous tab", not "leave the app"). */
  useEffect(() => {
    const next =
      tab === "playground" && playgroundTab
        ? `#/playground/${playgroundTab}`
        : `#/${tab}`;
    if (window.location.hash !== next) {
      window.history.replaceState(null, "", next);
    }
  }, [tab, playgroundTab]);

  /* HASH -> STORE: Back/Forward and pasted links navigate the tabs. */
  useEffect(() => {
    const onHashChange = () => {
      const fromHash = tabFromHash(window.location.hash);
      const sub = subTabFromHash(window.location.hash);
      if (fromHash !== tab) setTab(fromHash);
      if (fromHash === "playground" && sub && sub !== playgroundTab) {
        setPlaygroundTab(sub);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [tab, playgroundTab, setTab, setPlaygroundTab]);
}
