/// <reference types="vite/client" />

/**
 * The dev-only startup token channel (`services/api/startup.ts`):
 * supplied at dev-server start (`VITE_PHARMAML_TOKEN=… vite`),
 * never present in production builds, never persisted.
 */
interface ImportMetaEnv {
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly VITE_PHARMAML_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
