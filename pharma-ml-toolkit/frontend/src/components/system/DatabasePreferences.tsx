/**
 * part17 §3.2 (Round 360, the G1 lane) — the durable user-preference
 * manager.
 *
 * The `/database/preferences` quartet (GET list / GET one / PUT one /
 * DELETE one) was the one /database sub-surface with NO consumer at
 * all — not even a client function. This card is the consumer:
 *
 * * the list lane — `GET /api/v1/database/preferences` (the rows
 *   ordered by key), rendered as key + JSON value + updated_at;
 * * the write lane — `PUT /api/v1/database/preferences/{key}` over a
 *   strict-JSON value the card validates BEFORE the request (an
 *   invalid value never leaves the browser — the honest error names
 *   the lane), with the key bound checked client-side against the
 *   backend's own 1–128 rule;
 * * the delete lane — `DELETE /api/v1/database/preferences/{key}`
 *   behind a two-step confirmation (the row action only arms; the
 *   confirm button fires), with the honest 404 already-gone lane
 *   reconciling the list;
 * * the ephemeral lane — a 409 from a non-durable instance renders
 *   the honest note (the shipped host is always durable).
 *
 * The honest split, stated plainly in the card: this is the BACKEND's
 * durable preference store (SQLite `user_preferences`); the app's own
 * session settings (theme/density, the settings panel) are a
 * DIFFERENT surface — the localStorage store — and neither pretends
 * to be the other.
 */

import { useCallback, useEffect, useState } from "react";
import {
  LoaderCircle,
  RefreshCw,
  Save,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ApiError } from "@/services/api/client";
import {
  databaseApi,
  type PreferenceRow,
} from "@/services/database/client";

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail;
    if (
      typeof detail === "object" &&
      detail !== null &&
      "detail" in detail &&
      typeof (detail as Record<string, unknown>)["detail"] === "string"
    ) {
      return (detail as Record<string, string>)["detail"] ?? `HTTP ${error.status}`;
    }
    if (typeof detail === "string") {
      return detail;
    }
    return `HTTP ${error.status}`;
  }
  return error instanceof Error ? error.message : String(error);
}

function formatValue(value: unknown): string {
  try {
    const text = JSON.stringify(value);
    return text === undefined ? String(value) : text;
  } catch {
    return String(value);
  }
}

export function DatabasePreferences() {
  const [rows, setRows] = useState<readonly PreferenceRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  // the write lane (validated client-side before any request leaves)
  const [keyText, setKeyText] = useState("");
  const [valueText, setValueText] = useState("");
  const [writeBusy, setWriteBusy] = useState(false);
  // the delete lane — the two-step gate, one armed row at a time
  const [deletePending, setDeletePending] = useState<PreferenceRow | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await databaseApi.preferences();
      setRows(next);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // the honest ephemeral answer: persistence is disabled for
        // this instance (the shipped host is always durable)
        setRows(null);
        setError(null);
        setNotice(
          "Non-durable instance — the preference store is unavailable (the shipped host always runs durable).",
        );
        return;
      }
      setError(describeError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleSave = useCallback(async () => {
    const key = keyText.trim();
    if (key.length === 0 || key.length > 128) {
      setError(
        "the preference key must be 1–128 characters (the backend's own bound)",
      );
      return;
    }
    let value: unknown;
    try {
      value = JSON.parse(valueText);
    } catch {
      setError(
        "the preference value must be strict JSON (a number, string, boolean, array or object)",
      );
      return;
    }
    setWriteBusy(true);
    setError(null);
    setNotice(null);
    try {
      await databaseApi.setPreference(key, value);
      setNotice(`Saved the preference "${key}".`);
      setValueText("");
      await refresh();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setWriteBusy(false);
    }
  }, [keyText, valueText, refresh]);

  const requestDelete = useCallback((row: PreferenceRow) => {
    setDeletePending(row);
    setError(null);
    setNotice(null);
  }, []);

  const cancelDelete = useCallback(() => {
    setDeletePending(null);
  }, []);

  const confirmDelete = useCallback(
    async (row: PreferenceRow) => {
      setDeleteBusy(true);
      setError(null);
      setNotice(null);
      try {
        await databaseApi.deletePreference(row.key);
        setDeletePending(null);
        await refresh();
        setNotice(`Deleted the preference "${row.key}".`);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          // the honest already-gone lane: reconcile the list
          setDeletePending(null);
          await refresh();
          setNotice(
            `"${row.key}" was already gone (404) — the list refreshed.`,
          );
          return;
        }
        setError(describeError(err));
      } finally {
        setDeleteBusy(false);
      }
    },
    [refresh],
  );

  return (
    <section className="mt-12" aria-label="Studio preferences">
      <div className="flex items-center gap-2">
        <SlidersHorizontal
          aria-hidden
          className="h-5 w-5 text-[var(--color-primary)]"
        />
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">
          The durable preference store
        </h2>
      </div>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--color-muted-foreground)]">
        The backend&rsquo;s own user-preference key/value store over the
        local SQLite database — one JSON value per key, persisted with
        everything else in the backup bundles. The app&rsquo;s session
        settings (theme, density) are a different surface — the
        settings panel&rsquo;s localStorage store — neither pretends to
        be the other.
      </p>

      {error ? (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        >
          {error}
        </div>
      ) : null}
      {notice ? (
        <p
          role="status"
          className="mt-4 text-sm text-[var(--color-muted-foreground)]"
        >
          {notice}
        </p>
      ) : null}

      <Card className="mt-4">
        <CardHeader>
          <CardTitle className="text-base">The stored preferences</CardTitle>
          <CardDescription>
            {loading
              ? "Reading the store…"
              : rows === null
                ? "The store is unavailable for this instance."
                : `${rows.length} preference${rows.length === 1 ? "" : "s"} in the durable store.`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refresh()}
              disabled={loading}
            >
              {loading ? (
                <LoaderCircle aria-hidden className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw aria-hidden className="h-4 w-4" />
              )}
              Refresh
            </Button>
          </div>
          {rows !== null && rows.length > 0 ? (
            <div className="mt-4 max-h-96 space-y-2 overflow-y-auto">
              {rows.map((row) => (
                <div
                  key={row.key}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-[var(--color-border)] p-3"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-xs font-medium text-[var(--color-foreground)]">
                      {row.key}
                    </p>
                    <p className="mt-1 break-all font-mono text-[11px] text-[var(--color-muted-foreground)]">
                      {formatValue(row.value)}
                    </p>
                    <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                      updated {row.updated_at.slice(0, 19).replace("T", " ")}
                    </p>
                  </div>
                  <div className="shrink-0">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => requestDelete(row)}
                      aria-label={`Delete preference ${row.key}`}
                    >
                      <Trash2 aria-hidden className="h-4 w-4" />
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : rows !== null && rows.length === 0 ? (
            <p className="mt-4 text-sm text-[var(--color-muted-foreground)]">
              The durable store is empty — save a preference below and it
              lands here (and in every backup bundle).
            </p>
          ) : null}

          <div className="mt-6 rounded-md border border-[var(--color-border)] p-4">
            <p className="text-xs font-semibold text-[var(--color-foreground)]">
              Write one preference
            </p>
            <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
              PUT /api/v1/database/preferences/&#123;key&#125; — an existing
              key is replaced, a new key is created. The value is validated
              as strict JSON before the request leaves this page.
            </p>
            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
              <label className="sr-only" htmlFor="preference-key">
                The preference key
              </label>
              <input
                id="preference-key"
                value={keyText}
                onChange={(event) => setKeyText(event.target.value)}
                placeholder="the key (1–128 characters)"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-1.5 font-mono text-xs text-[var(--color-foreground)] placeholder:text-[var(--color-muted-foreground)] sm:w-56"
              />
              <label className="sr-only" htmlFor="preference-value">
                The preference value (strict JSON)
              </label>
              <input
                id="preference-value"
                value={valueText}
                onChange={(event) => setValueText(event.target.value)}
                placeholder='the value (strict JSON — e.g. {"units": "mg/L"} or true)'
                className="min-w-0 flex-1 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-1.5 font-mono text-xs text-[var(--color-foreground)] placeholder:text-[var(--color-muted-foreground)]"
              />
              <Button
                size="sm"
                onClick={() => void handleSave()}
                disabled={writeBusy}
              >
                <Save aria-hidden className="h-4 w-4" />
                {writeBusy ? "Saving…" : "Save"}
              </Button>
            </div>
          </div>

          {deletePending !== null ? (
            <div className="mt-4 rounded-md border border-[var(--color-destructive)]/40 p-4">
              <p className="text-sm font-semibold text-[var(--color-foreground)]">
                <Trash2 aria-hidden className="mr-1 inline h-4 w-4" />
                Delete the preference “{deletePending.key}”?
              </p>
              <p className="mt-2 break-all font-mono text-xs text-[var(--color-muted-foreground)]">
                value {formatValue(deletePending.value)} · updated{" "}
                {deletePending.updated_at.slice(0, 19).replace("T", " ")}
              </p>
              <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
                This removes exactly this one row from the durable store.
                Every other preference, every dataset, model and job stays
                untouched — and the next backup bundle simply no longer
                carries this key.
              </p>
              <div className="mt-3 flex gap-2">
                <Button variant="outline" size="sm" onClick={cancelDelete} disabled={deleteBusy}>
                  Keep it
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => void confirmDelete(deletePending)}
                  disabled={deleteBusy}
                >
                  {deleteBusy ? "Deleting…" : "Delete preference"}
                </Button>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </section>
  );
}
