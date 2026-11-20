import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";

/**
 * part13 Phase 3 — the §22 interactive chart surface.
 *
 * The plotly module is faked at the library boundary (its real bundle
 * needs WebGL/canvas the jsdom lane does not carry); everything under
 * test is the REAL component: the lazy-chunk loading state, the empty
 * state, the §22.4 interaction contract (hovermode "closest" — never
 * the crosshair modes), the §22.5 point-identity contract (a click
 * hands back the point's own customdata, never a position or index),
 * and the §23 resource rule (purge on unmount — no listener leaks).
 */

interface GraphDiv extends HTMLElement {
  on?: (event: string, handler: (payload: unknown) => void) => unknown;
  removeAllListeners?: (event: string) => void;
}

const plotlyState = vi.hoisted(() => {
  const state: {
    reactCalls: Array<{ el: GraphDiv; data: unknown; layout: Record<string, unknown>; config: unknown }>;
    purged: unknown[];
    resizes: unknown[];
    clickHandlers: Map<string, (payload: unknown) => void>;
  } = { reactCalls: [], purged: [], resizes: [], clickHandlers: new Map() };
  return state;
});

vi.mock("plotly.js-dist-min", () => {
  const module = {
    react: (el: unknown, data: unknown, layout: unknown, config: unknown) => {
      const div = el as GraphDiv;
      plotlyState.reactCalls.push({
        el: div,
        data,
        layout: layout as Record<string, unknown>,
        config,
      });
      // real plotly attaches its event-emitter surface to the graph
      // div; the fake reproduces exactly that contract
      div.on = (event: string, handler: (payload: unknown) => void) => {
        plotlyState.clickHandlers.set(event, handler);
        return div;
      };
      div.removeAllListeners = (event: string) => {
        plotlyState.clickHandlers.delete(event);
      };
      return Promise.resolve();
    },
    purge: (el: unknown) => {
      plotlyState.purged.push(el);
    },
    Plots: {
      resize: (el: unknown) => {
        plotlyState.resizes.push(el);
      },
    },
  };
  // the real dist-min bundle arrives as the interop `default` (Vite's
  // CJS wrapping) AND as named exports — the mock mirrors both shapes
  return { default: module, ...module };
});

beforeEach(() => {
  plotlyState.reactCalls.length = 0;
  plotlyState.purged.length = 0;
  plotlyState.resizes.length = 0;
  plotlyState.clickHandlers.clear();
  // jsdom carries no ResizeObserver — the resource-aware contract
  // (§23) is stubbed at the browser API boundary
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    },
  );
});

const SCATTER = [
  {
    type: "scatter",
    mode: "markers",
    x: [1, 2, 3],
    y: [4, 5, 6],
    customdata: [{ smiles: "CCO" }, { smiles: "CCC" }, { smiles: "CCCC" }],
  },
];

describe("PlotlyFigure", () => {
  it("shows the honest empty state when no trace is renderable", async () => {
    render(
      <PlotlyFigure data={[]} ariaLabel="the descriptor chart" emptyHint="run the stage first" />,
    );

    expect(screen.getByText("run the stage first")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    // the plotly chunk is never even loaded for an empty figure
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(plotlyState.reactCalls).toHaveLength(0);
  });

  it("loads the chunk, mounts the graph div, and draws through Plotly.react", async () => {
    render(<PlotlyFigure data={SCATTER} ariaLabel="the solubility chart" />);

    // the loading state while the lazy chunk resolves
    expect(screen.getByRole("status")).toHaveTextContent(/loading the plotly chart chunk/);

    const holder = await screen.findByRole("img", { name: "the solubility chart" });
    await waitFor(() => {
      expect(plotlyState.reactCalls).toHaveLength(1);
    });
    const call = plotlyState.reactCalls[0]!;
    expect(call.el).toBeInstanceOf(HTMLElement);
    expect(holder.contains(call.el)).toBe(true);
  });

  it("pins the §22.4 interaction contract: hovermode closest, never the crosshair modes", async () => {
    render(<PlotlyFigure data={SCATTER} ariaLabel="chart" />);

    await waitFor(() => {
      expect(plotlyState.reactCalls).toHaveLength(1);
    });
    const layout = plotlyState.reactCalls[0]!.layout;
    expect(layout["hovermode"]).toBe("closest");
    expect(layout["hovermode"]).not.toBe("x");
    expect(layout["hovermode"]).not.toBe("y");
    // the vendor logo stays off the offline surface
    const config = plotlyState.reactCalls[0]!.config as Record<string, unknown>;
    expect(config["displaylogo"]).toBe(false);
  });

  it("hands a click back the point's OWN customdata (§22.5 — never a position or index)", async () => {
    const onPointSelect = vi.fn();
    render(
      <PlotlyFigure data={SCATTER} ariaLabel="chart" onPointSelect={onPointSelect} />,
    );

    await waitFor(() => {
      expect(plotlyState.reactCalls).toHaveLength(1);
    });
    const fire = plotlyState.clickHandlers.get("plotly_click");
    expect(fire).toBeDefined();

    fire!({
      points: [{ pointNumber: 2, x: 3, y: 6, customdata: { smiles: "CCCC" } }],
    });
    expect(onPointSelect).toHaveBeenCalledWith({
      pointIndex: 2,
      x: 3,
      y: 6,
      customdata: { smiles: "CCCC" },
    });
  });

  it("purges the plotly surface on unmount — no DOM listeners leak (§23)", async () => {
    const { unmount } = render(<PlotlyFigure data={SCATTER} ariaLabel="chart" />);

    await waitFor(() => {
      expect(plotlyState.reactCalls).toHaveLength(1);
    });
    const drawnEl = plotlyState.reactCalls[0]!.el;

    unmount();
    expect(plotlyState.purged).toHaveLength(1);
    expect(plotlyState.purged[0]).toBe(drawnEl);
  });
});
