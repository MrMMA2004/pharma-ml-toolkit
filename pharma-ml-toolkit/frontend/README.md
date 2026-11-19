# frontend — the PharmaML Studio SPA (part11 Phase Three)

The offline React + TypeScript single-page application that the Tauri 2
desktop host loads through WebView2. Production is the static `dist/`
build bundled as local application resources: no dev server, no remote
URL, no CDN resources (§4.1).

## The mandated stack (§3)

TypeScript + React + Vite + Tailwind CSS + shadcn/ui primitives +
Motion for React + Plotly.js + Three.js. The Python sidecar behind the
FastAPI/WebSocket contract stays the only scientific brain (§3's
boundary rule — no science in TypeScript, none in Rust).

## Commands

```
npm ci                 # restore exact dependencies from package-lock.json
npm run generate:types # regenerate src/types/api.d.ts from the backend contract
npm run typecheck      # strict tsc --noEmit
npm run build          # typecheck + the production Vite build -> dist/
npm run dev            # DEV-ONLY server on :5174 (never used in production)
```

## Layout

```
src/
  app/          the shell + the nine-workspace registry (the §10.1 order)
  components/
    ui/         the shadcn/ui primitives (button, card, tabs; Radix heads)
    layout/     WorkspaceNav, WorkspaceShell
  features/     nine API-backed workspaces (datasets, ML, DL, docking, PK/PD, MD, generative, visualization, introduction)
  services/     api/ (typed REST client), websocket/ (typed envelope)
  types/        api.d.ts — GENERATED from ../contracts/openapi.json (§12.2: one DTO source)
  styles/       the Mineral tokens (light/dark, chart palette)
  lib/          cn() and shared utilities
```

## Honesty rules carried over

The workspace registry (`src/app/workspaces.ts`) states each
workspace's build state; all nine rows are `workflow`-connected to
the real backend through their service clients (verified by the
`tests/test_frontend_services.py` route-surface guards and the
scratch browser drivers). The shell still renders the honest
queued-row placeholder for any registry row that regresses — no
fake controls, no fabricated data. The registry order is
drift-guarded by `tests/test_frontend_foundation.py` on the Python
side, and the whole app boots through the §14.1 `StartupGate`
(`tests/test_frontend_startup.py`).
