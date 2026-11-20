import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// part13 Phase 3 — the component-test lane.
//
// Vitest + jsdom + React Testing Library over the same `@` alias the
// Vite build resolves, so a test imports exactly what the Tauri host
// loads. The tailwind plugin is intentionally absent: no test asserts
// compiled CSS, so the CSS pipeline never runs (vitest stubs style
// imports) and the suite stays fast. The production build stays the
// ONLY place CSS is verified.
//
// Scope rule (the §22.4 anti-fabrication ethic): these tests exercise
// the REAL component tree through user-facing queries (roles, labels,
// visible text) with service modules mocked at the network boundary —
// no component is tested against fabricated DOM it never renders.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/__tests__/**/*.test.{ts,tsx}"],
  },
});
