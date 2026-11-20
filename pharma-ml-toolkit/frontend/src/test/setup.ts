import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Explicit cleanup (vitest runs without `globals: true`, so the
// testing-library auto-cleanup registration never happens on its own).
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
