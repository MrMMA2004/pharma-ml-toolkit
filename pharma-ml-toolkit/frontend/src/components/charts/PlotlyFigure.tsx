import { useEffect, useRef, useState } from "react";

/**
 * part11 §22.2 — the Plotly.js interactive chart surface.
 *
 * The one shared lane every 2-D scientific chart in the workspaces
 * renders through (the §3.2 campaign: real Plotly traces in src,
 * replacing the hand-rolled SVG charts — the backend Matplotlib
 * figure artifacts stay their own §22.3 lane). The library loads
 * through a dynamic `import` so the Vite `manualChunks` split keeps
 * the plotly bundle a separate lazy chunk: the shell boots without
 * it, the first chart pays the one-time cost.
 *
 * §22.4 — the interaction contract: `hovermode: "closest"` gives the
 * per-point hover readout WITHOUT any mouse-following crosshair lines
 * (never `hovermode: "x"/"y"` — those are exactly the unwanted
 * automatic crosshair behaviour the migration retires); gridlines
 * and threshold/reference shapes stay, click-to-select rides
 * `plotly_click`.
 *
 * §22.5 — the point identity contract: every trace may carry
 * `customdata` (the stable per-point sample metadata); a click hands
 * the selected point's own `customdata` back through `onPointSelect`
 * so the caller's inspector shows the REAL sample row, never the
 * plot position or an array index.
 *
 * §23 resource rules — `Plotly.react` for updates (diffed, cheap),
 * `Plotly.Plots.resize` under a ResizeObserver, `Plotly.purge` on
 * unmount so no DOM listeners leak across workspace switches.
 */

/** A plotly trace — the minimal typed surface the house charts build. */
export interface PlotTrace {
  readonly type?: string;
  readonly mode?: string;
  readonly name?: string;
  readonly x?: readonly (number | string | null)[];
  readonly y?: readonly (number | string | null)[];
  readonly z?: readonly (readonly (number | null)[])[];
  readonly text?: readonly (string | number | null)[];
  readonly customdata?: readonly unknown[];
  readonly orientation?: string;
  readonly showlegend?: boolean;
  readonly hovertemplate?: string;
  readonly marker?: {
    readonly color?: string | readonly (string | number | null)[];
    readonly size?: number | readonly number[];
    readonly symbol?: string;
    readonly colorscale?: string;
    readonly showscale?: boolean;
    readonly line?: { readonly color?: string; readonly width?: number };
    [key: string]: unknown;
  };
  readonly line?: {
    readonly color?: string;
    readonly width?: number;
    readonly dash?: string;
    readonly shape?: string;
    [key: string]: unknown;
  };
  readonly fill?: string;
  readonly opacity?: number;
  readonly colorbar?: Record<string, unknown>;
  [key: string]: unknown;
}

/** The layout pieces the house charts set (merged over the theme). */
export interface PlotLayout {
  readonly title?: Record<string, unknown>;
  readonly xaxis?: Record<string, unknown>;
  readonly yaxis?: Record<string, unknown>;
  readonly margin?: { readonly l?: number; readonly r?: number; readonly t?: number; readonly b?: number };
  readonly legend?: Record<string, unknown>;
  readonly annotations?: readonly Record<string, unknown>[];
  readonly shapes?: readonly Record<string, unknown>[];
  readonly hovermode?: string;
  readonly dragmode?: string;
  [key: string]: unknown;
}

/** The plotly config surface the component pins. */
export interface PlotConfig {
  readonly responsive?: boolean;
  readonly displaylogo?: boolean;
  readonly displayModeBar?: boolean | string;
}

/** One selected point of a `plotly_click` (§22.5's payload). */
export interface PlotPoint {
  readonly pointIndex: number | null;
  readonly x: unknown;
  readonly y: unknown;
  readonly customdata: unknown;
}

/** The structural slice of the `plotly.js-dist-min` module used here. */
interface PlotlyModule {
  react: (
    el: HTMLElement,
    data: readonly PlotTrace[],
    layout: Record<string, unknown>,
    config: PlotConfig,
  ) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
  Plots: { resize: (el: HTMLElement) => void };
}

/** The plot div with plotly's own event-emitter surface. */
interface PlotlyDiv extends HTMLElement {
  on?: (
    event: string,
    handler: (payload: { points?: ReadonlyArray<Record<string, unknown>> }) => void,
  ) => void;
  removeAllListeners?: (event: string) => void;
}

/** The Mineral design tokens a themed chart reads at render time. */
interface ChartTheme {
  readonly foreground: string;
  readonly mutedForeground: string;
  readonly border: string;
  readonly fontFamily: string;
}

/**
 * Convert an OKLCH color token to an sRGB `rgb()/rgba()` string.
 *
 * Plotly's own color parser accepts rgb/rgba/hex/hsl — not the modern
 * `oklch()` space the Mineral tokens are authored in (it silently
 * falls back to its default #444/#eee, which the r291 browser pass
 * caught on the live axis text). The conversion is the standard
 * OKLab → linear-sRGB matrix, then the sRGB transfer curve, clamped.
 */
function oklchToRgb(token: string): string {
  const match = /^oklch\(\s*([\d.]+(?:e-?[\d.]+)?)\s+([\d.]+(?:e-?[\d.]+)?)\s+([\d.]+(?:e-?[\d.]+)?)(?:\s*\/\s*([\d.]+%?))?\s*\)$/i
    .exec(token.trim());
  if (match === null) {
    return token; // not a plain oklch() literal — pass it through
  }
  const L = Number(match[1]);
  const C = Number(match[2]);
  const H = (Number(match[3]) * Math.PI) / 180;
  let alpha = 1;
  const alphaPart = match[4];
  if (alphaPart !== undefined) {
    alpha = alphaPart.endsWith("%")
      ? Number(alphaPart.slice(0, -1)) / 100
      : Number(alphaPart);
  }
  const a = C * Math.cos(H);
  const b = C * Math.sin(H);
  const lp = L + 0.3963377774 * a + 0.2158037573 * b;
  const mp = L - 0.1055613458 * a - 0.0638541728 * b;
  const sp = L - 0.0894841775 * a - 1.291485548 * b;
  const lin = [lp ** 3, mp ** 3, sp ** 3];
  const matrix = [
    [4.0767416621, -3.3077115913, 0.230969043],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.707614701],
  ];
  const gamma = (c: number): number => {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  };
  const rgb = matrix.map(
    (row) =>
      gamma(row[0]! * lin[0]! + row[1]! * lin[1]! + row[2]! * lin[2]!),
  );
  return alpha < 1
    ? `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${Number(alpha.toFixed(4))})`
    : `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

/**
 * Resolve one color value: a `var(--token)` reads the live computed
 * token off the plot element, and any oklch result converts to sRGB
 * (plotly's parser accepts only the classic spaces). Anything else
 * passes through untouched — `customdata` strings are never colors
 * and never change.
 */
function resolveColor(value: unknown, div: HTMLElement): unknown {
  if (typeof value !== "string") {
    return value;
  }
  let token = value.trim();
  if (token.startsWith("var(")) {
    const name = token.slice(4, -1).trim();
    token = window.getComputedStyle(div).getPropertyValue(name).trim();
    if (token === "") {
      return value; // an unknown token — leave the string alone
    }
  }
  return token.toLowerCase().startsWith("oklch") ? oklchToRgb(token) : token;
}

/** Walk traces/layout/shape trees, resolving every color-ish string. */
function resolveTokens(node: unknown, div: HTMLElement): unknown {
  if (typeof node === "string") {
    return resolveColor(node, div);
  }
  if (Array.isArray(node)) {
    return node.map((item) => resolveTokens(item, div));
  }
  if (node !== null && typeof node === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(
      node as Record<string, unknown>,
    )) {
      out[key] = resolveTokens(child, div);
    }
    return out;
  }
  return node;
}

/** Read the live design tokens, already in plotly-parseable sRGB. */
function readTheme(el: HTMLElement): ChartTheme {
  const style = window.getComputedStyle(el);
  const token = (name: string, fallback: string): string => {
    const raw = style.getPropertyValue(name).trim();
    return raw === "" ? fallback : oklchToRgb(raw);
  };
  return {
    foreground: token("--foreground", "rgb(24, 24, 24)"),
    mutedForeground: token("--muted-foreground", "rgb(116, 116, 116)"),
    border: token("--border", "rgb(233, 233, 233)"),
    fontFamily: style.getPropertyValue("font-family").trim() || "system-ui, sans-serif",
  };
}

/** The house chart-token colors (readable names for trace styling). */
export const CHART_COLORS = {
  primary: "var(--chart-1)",
  secondary: "var(--chart-2)",
  tertiary: "var(--chart-3)",
  quaternary: "var(--chart-4)",
  neutral: "var(--chart-5)",
} as const;

export interface PlotlyFigureProps {
  /** The traces — each may carry `customdata` per point (§22.5). */
  readonly data: readonly PlotTrace[];
  /** Caller layout — merged over the themed defaults. */
  readonly layout?: PlotLayout;
  /** Config overrides (displaylogo stays false — no vendor logo). */
  readonly config?: PlotConfig;
  /** §22.5 — a click hands the point's own customdata back. */
  readonly onPointSelect?: (point: PlotPoint) => void;
  /** Fixed pixel height for the chart viewport. */
  readonly height?: number;
  /** The accessible description (role="img", aria-label). */
  readonly ariaLabel: string;
  /** Empty-state hint when `data` carries no renderable trace. */
  readonly emptyHint?: string;
}

export function PlotlyFigure({
  data,
  layout,
  config,
  onPointSelect,
  height = 280,
  ariaLabel,
  emptyHint = "No plottable rows yet — run the stage to chart its real result.",
}: PlotlyFigureProps) {
  const holderRef = useRef<HTMLDivElement | null>(null);
  const divRef = useRef<PlotlyDiv | null>(null);
  const moduleRef = useRef<PlotlyModule | null>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const renderable = data.some((trace) => {
    if (trace.z !== undefined) return trace.z.length > 0;
    if (trace.x === undefined || trace.y === undefined) return false;
    return Math.min(trace.x.length, trace.y.length) > 0;
  });

  // the click handler rides a ref: a new closure must not re-mount
  // the whole plot
  const selectRef = useRef(onPointSelect);
  selectRef.current = onPointSelect;

  // §23 — purge the plotly DOM/listener surface on unmount (the
  // later-declared effect cleans up last, after the render effect's
  // observers are gone)
  useEffect(() => {
    return () => {
      const plotly = moduleRef.current;
      const div = divRef.current;
      if (plotly !== null && div !== null) {
        try {
          plotly.purge(div);
        } catch {
          // already detached — nothing left to leak
        }
      }
      if (div !== null && div.parentElement === holderRef.current) {
        holderRef.current?.removeChild(div);
        divRef.current = null;
      }
      moduleRef.current = null;
    };
  }, []);

  // load the lazy chunk once
  useEffect(() => {
    if (!renderable || moduleRef.current !== null) {
      return undefined;
    }
    let cancelled = false;
    void import("plotly.js-dist-min")
      .then((mod: unknown) => {
        if (cancelled) return;
        // the dist-min UMD bundle arrives as the interop `default`
        // (Vite's CJS wrapping) — read both shapes, the structural
        // slice is the same either way
        const candidate = (mod as { readonly default?: PlotlyModule })
          .default;
        moduleRef.current = candidate ?? (mod as PlotlyModule);
        setReady(true);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setFailed(
          error instanceof Error
            ? `the plotly chunk failed to load: ${error.message}`
            : "the plotly chunk failed to load",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [renderable]);

  // the draw itself: Plotly.react over the themed layout
  useEffect(() => {
    const plotly = moduleRef.current;
    const holder = holderRef.current;
    if (plotly === null || holder === null || !ready || !renderable) {
      return undefined;
    }
    if (divRef.current === null) {
      const div = document.createElement("div") as PlotlyDiv;
      div.style.width = "100%";
      div.style.height = "100%";
      holder.appendChild(div);
      divRef.current = div;
    }
    const div = divRef.current;

    const draw = (): void => {
      const theme = readTheme(div);
      const themed: Record<string, unknown> = {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: theme.foreground, family: theme.fontFamily, size: 11 },
        margin: { l: 48, r: 16, t: 24, b: 36 },
        hovermode: "closest", // §22.4: per-point readout, never a crosshair
        dragmode: "zoom",
        legend: { orientation: "h", y: -0.16, font: { size: 10 } },
        ...(layout ?? {}),
        xaxis: {
          gridcolor: theme.border,
          zerolinecolor: theme.border,
          tickfont: { color: theme.mutedForeground, size: 10 },
          titlefont: { color: theme.foreground },
          ...(layout?.xaxis ?? {}),
        },
        yaxis: {
          gridcolor: theme.border,
          zerolinecolor: theme.border,
          tickfont: { color: theme.mutedForeground, size: 10 },
          titlefont: { color: theme.foreground },
          ...(layout?.yaxis ?? {}),
        },
      };
      // the theme pass: traces + layout carry `var(--chart-*)` tokens
      // (and the oklch tokens above); plotly parses only classic
      // color spaces, so every color-ish string resolves to live sRGB
      // — customdata strings pass through untouched
      const resolvedTraces = resolveTokens(data, div) as readonly PlotTrace[];
      const resolvedLayout = resolveTokens(themed, div) as Record<
        string,
        unknown
      >;
      void plotly
        .react(div, resolvedTraces, resolvedLayout, config ?? { responsive: true, displaylogo: false })
        .then(() => {
          if (selectRef.current !== undefined) {
            void div.removeAllListeners?.("plotly_click");
            div.on?.("plotly_click", (payload: { points?: ReadonlyArray<Record<string, unknown>> }) => {
              const point = payload.points?.[0];
              if (point === undefined) return;
              const index =
                typeof point["pointNumber"] === "number"
                  ? (point["pointNumber"] as number)
                  : typeof point["pointIndex"] === "number"
                    ? (point["pointIndex"] as number)
                    : null;
              selectRef.current?.({
                pointIndex: index,
                x: point["x"],
                y: point["y"],
                customdata: point["customdata"],
              });
            });
          }
        })
        .catch((error: unknown) => {
          setFailed(
            error instanceof Error
              ? `the chart failed to render: ${error.message}`
              : "the chart failed to render",
          );
        });
    };

    draw();

    let disposed = false;
    const observer = new ResizeObserver(() => {
      if (disposed) return;
      if (divRef.current !== null && moduleRef.current !== null) {
        moduleRef.current.Plots.resize(divRef.current);
      }
    });
    observer.observe(div);

    // re-theme when the `.dark` class flips (the tokens are CSS vars)
    let lastTheme = readTheme(div);
    const themeObserver = new MutationObserver(() => {
      if (disposed) return;
      const next = readTheme(div);
      if (
        next.foreground === lastTheme.foreground &&
        next.border === lastTheme.border &&
        next.fontFamily === lastTheme.fontFamily
      ) {
        return;
      }
      lastTheme = next;
      draw();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => {
      disposed = true;
      observer.disconnect();
      themeObserver.disconnect();
      void div.removeAllListeners?.("plotly_click");
    };
    // the pages build trace arrays with useMemo from their result
    // state, so identity-keyed deps re-run on genuine changes only
  }, [data, layout, config, ready, renderable]);

  return (
    <div className="w-full">
      {failed !== null ? (
        <p
          role="alert"
          className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-muted)] px-3 py-2 text-xs text-[var(--color-foreground)]"
        >
          {failed}
        </p>
      ) : !renderable ? (
        <p className="rounded-md border border-dashed border-[var(--color-border)] px-3 py-6 text-center text-xs text-[var(--color-muted-foreground)]">
          {emptyHint}
        </p>
      ) : !ready ? (
        <div
          className="flex items-center justify-center rounded-md border border-[var(--color-border)] text-xs text-[var(--color-muted-foreground)]"
          style={{ height }}
          role="status"
        >
          loading the plotly chart chunk…
        </div>
      ) : (
        <div
          ref={holderRef}
          role="img"
          aria-label={ariaLabel}
          className="w-full rounded-md"
          style={{ height }}
        />
      )}
    </div>
  );
}
