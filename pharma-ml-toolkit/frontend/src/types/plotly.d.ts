/**
 * The typed surface of `plotly.js-dist-min` (part11 §22.2's chart
 * library). The dist-min bundle is the UMD namespace object — under
 * Vite's CJS interop it arrives as the module's `default` export —
 * and this declaration types exactly the structural slice the house
 * charts use (`react` for diffed re-renders, `purge` for the §23
 * listener cleanup, `Plots.resize` for container resizes; the click
 * payload carries each point's own `customdata` per §22.5). Typed at
 * the boundary — no `any` leaks into the chart components.
 */
declare module "plotly.js-dist-min" {
  /** One point of a `plotly_click` event's `points` array. */
  export interface PlotlyClickPoint {
    readonly pointNumber?: number;
    readonly pointIndex?: number;
    readonly x?: unknown;
    readonly y?: unknown;
    readonly customdata?: unknown;
    readonly [key: string]: unknown;
  }

  const Plotly: {
    readonly react: (
      el: HTMLElement,
      data: readonly object[],
      layout: object,
      config?: object,
    ) => Promise<HTMLElement>;
    readonly purge: (el: HTMLElement) => void;
    readonly Plots: {
      readonly resize: (el: HTMLElement) => void;
    };
  };

  export default Plotly;
}
