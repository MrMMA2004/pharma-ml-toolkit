# PharmaML Studio frontend

This is the React + TypeScript single-page application loaded by the Tauri 2 desktop host. Production builds are static assets bundled locally for WebView2; the production app does not use the Vite development server, a remote URL, or a CDN.

## Stack

TypeScript + React + Vite + Tailwind CSS + shadcn/ui primitives + Motion for React + Plotly.js + Three.js. Scientific computation remains in the Python FastAPI/WebSocket backend; the frontend is a typed client.

## Frontend commands

Run from `pharma-ml-toolkit/frontend/`:

```bash
npm ci
npm run generate:types # regenerate src/types/api.d.ts from the backend contract
npm run typecheck      # strict tsc --noEmit
npm test               # Vitest component and service tests
npm run build          # typecheck + production Vite build -> dist/
npm run dev            # dev-only Vite server on :5174
```

## Browser development with a live backend

A browser session uses a **development-only** same-origin fallback. Start the local API backend separately, then copy the port and per-session token from its `PHARMAML-READY` handshake into the Vite process environment:

```bash
# In another terminal, from pharma-ml-toolkit/ after installing API runtime dependencies:
python -m pharmaml.api

# In the frontend terminal, using the port and token from that handshake:
PHARMAML_DEV_BACKEND_PORT=<port> VITE_PHARMAML_TOKEN=<session-token> npm run dev
```

Vite proxies relative `/api` and WebSocket requests to the local backend. The token is session-scoped: keep it local, do not commit it or paste it into shared logs, and do not use this dev-only channel in a production build. Without a backend and the two environment values, the frontend startup gate will report that it cannot connect; it does not fabricate backend results.

## Native Tauri shell and sidecar

The Rust host starts a frozen Python sidecar from `src-tauri/resources/backend/pharmaml-backend/pharmaml-backend.exe`. That generated Windows resource directory is **not checked into the source tree**. The repository's [Windows release workflow](../../.github/workflows/windows-release.yml) builds and stages the sidecar on a Windows runner before bundling the app. Consequently, `cargo tauri dev` or `cargo tauri build` is not a complete fresh-checkout launch recipe: the native shell needs a compatible sidecar staged at that path first. A successful hosted build or install-smoke run also does not establish independent clean-machine verification.

## Layout

```text
src/
  app/          shell + nine-workspace registry
  components/   shared UI and layout primitives
  features/     eight API-backed workflows + the Introduction start page
  services/     typed REST and WebSocket clients
  types/        generated API types from ../contracts/openapi.json
  styles/       light/dark design tokens and chart palette
```

## Implementation boundaries

The workspace registry (`src/app/workspaces.ts`) records the nine navigation entries. The eight scientific workflow pages call the real backend through their service clients; Introduction is the informational start page. Tests cover the workspace registry, typed client contracts, and startup gate. See [`../docs/RESEARCH_WORKFLOWS.md`](../docs/RESEARCH_WORKFLOWS.md) for Python/CLI workflow assumptions and limits. That guide also preserves clearly labeled PySide6 interface notes for historical context; those retired screens are not current Studio features. Neither API documentation nor tests imply feature parity.
