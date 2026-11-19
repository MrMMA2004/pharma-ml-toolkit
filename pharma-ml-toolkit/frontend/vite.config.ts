import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";

// part11 §9.2: the Vite dev server is a DEVELOPMENT-ONLY convenience.
// The production frontend is the static `dist/` build loaded locally by
// the Tauri 2 host (frontendDist) — no dev server, no remote URL.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  css: {
    // r297: an inline EMPTY postcss config SHADOWS vite's upward file
    // discovery. Tailwind 4 needs no PostCSS pipeline here (the
    // @tailwindcss/vite plugin owns the transform), but the discovery
    // walk finds the repository ROOT's postcss.config.mjs -- the Next.js
    // wrapper around this tree, whose plugin shape vite cannot load
    // ("Invalid PostCSS Plugin found at: plugins[0]", run 44525087953).
    // The empty config says: nothing to load, anywhere.
    postcss: {},
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir: "dist",
    // the offline bundle must be self-contained: no CDN fallbacks
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        // plotly and three are the mandated scientific visualization
        // stacks — kept as separate lazy chunks so the shell boots fast
        manualChunks: {
          plotly: ["plotly.js-dist-min"],
          three: ["three"],
          motion: ["motion"],
        },
      },
    },
  },
  server: {
    port: 5174,
    strictPort: true,
    host: process.env.PHARMAML_DEV_HOST ?? "localhost",
    // dev-only: with PHARMAML_DEV_BACKEND_PORT set, the /api lanes are
    // proxied server-side to the real backend's loopback port, so a
    // browser dev session rides the actual service same-origin (the
    // CORS gate never applies server-side; the X-Session-Token header
    // forwards as-is; ws: true carries the event stream through the
    // same proxy so the dev fallback's WS lane opens same-origin).
    // The production frontend never uses
    // this: the Tauri 2 host loads the static dist/ build.
    proxy: process.env.PHARMAML_DEV_BACKEND_PORT
      ? {
          "/api": {
            target: `http://127.0.0.1:${process.env.PHARMAML_DEV_BACKEND_PORT}`,
            changeOrigin: false,
            ws: true,
          },
        }
      : undefined,
  },
});
