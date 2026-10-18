import { spawn } from "child_process";
import { existsSync } from "fs";
import path from "path";

/**
 * Shared stdin->stdout JSON runner for the repo-root Python bridge scripts
 * (scripts/*.py — outside pharma-ml-toolkit, so they never ship in the ZIP).
 *
 * Contract (both directions strict JSON on stdout):
 *  - exit 0 + parseable stdout  -> resolved with the parsed value
 *  - exit != 0 + {"error": ...} -> rejected with Error("BRIDGE_ERROR: ...")
 *    (a *client* error: bad input, invalid SMILES, degenerate request)
 *  - timeout                    -> rejected with Error("BRIDGE_TIMEOUT")
 *  - anything else              -> rejected with Error("BRIDGE_EXIT_<code>:<stderr tail>")
 */

/** Python interpreter candidates: env override -> toolkit venv -> PATH. */
function resolvePython(): string {
  const fromEnv = process.env.PHARMAML_PYTHON;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  const venv = "/home/z/.venv/bin/python";
  if (existsSync(venv)) return venv;
  return "python3";
}

export interface BridgeOptions {
  /** Bridge script file name inside the repo-root scripts/ directory. */
  script: string;
  /** JSON-serialisable request payload written to the bridge's stdin. */
  payload: unknown;
  /** Hard kill timeout in milliseconds (default 60 s). */
  timeoutMs?: number;
}

const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_STDOUT_BYTES = 20 * 1024 * 1024;
const MAX_STDERR_CHARS = 8_000;

/** Run a Python bridge script and return its parsed JSON response. */
export function runPythonBridge<T = unknown>({
  script,
  payload,
  timeoutMs = DEFAULT_TIMEOUT_MS,
}: BridgeOptions): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "scripts", script);
    if (!existsSync(scriptPath)) {
      reject(new Error(`BRIDGE_MISSING:${script}`));
      return;
    }

    const child = spawn(resolvePython(), [scriptPath], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        // Single-threaded worker caps: routes are one-shot per request and
        // neither bridge needs BLAS/joblib parallelism.
        LOKY_MAX_CPU_COUNT: "1",
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        child.kill("SIGKILL");
        reject(new Error("BRIDGE_TIMEOUT"));
      }
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => {
      if (stdout.length < MAX_STDOUT_BYTES) stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      if (stderr.length < MAX_STDERR_CHARS) stderr += chunk.toString("utf8");
    });
    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`Could not start the RDKit backend (${err.message}).`));
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code === 0) {
        try {
          resolve(JSON.parse(stdout) as T);
        } catch {
          reject(new Error("BRIDGE_BAD_JSON"));
        }
      } else {
        // Bridges print {"error": "..."} on stdout for clean failures.
        try {
          const parsed = JSON.parse(stdout) as { error?: unknown };
          if (typeof parsed.error === "string") {
            reject(new Error(`BRIDGE_ERROR: ${parsed.error}`));
            return;
          }
        } catch {
          // fall through to the generic message
        }
        reject(new Error(`BRIDGE_EXIT_${code ?? "unknown"}:${stderr.slice(-400)}`));
      }
    });

    child.stdin.on("error", () => {
      /* EPIPE when the bridge dies before reading stdin — close reports it. */
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

/** Map a runPythonBridge rejection to an HTTP status + clean message.
 *  Returns null when the error is not bridge-shaped (caller decides). */
export function bridgeErrorResponse(err: unknown): {
  status: number;
  body: { error: string };
} | null {
  const message = err instanceof Error ? err.message : String(err);
  if (message.startsWith("BRIDGE_ERROR:")) {
    return { status: 400, body: { error: message.slice("BRIDGE_ERROR:".length).trim() } };
  }
  if (message === "BRIDGE_TIMEOUT") {
    return {
      status: 504,
      body: { error: "The RDKit backend timed out — try fewer molecules or families." },
    };
  }
  if (message === "BRIDGE_BAD_JSON") {
    return { status: 500, body: { error: "The RDKit backend returned malformed output." } };
  }
  if (message.startsWith("BRIDGE_MISSING:")) {
    return {
      status: 500,
      body: {
        error: `The ${message.slice("BRIDGE_MISSING:".length)} bridge script is missing on the server.`,
      },
    };
  }
  const tail = message.startsWith("BRIDGE_EXIT_") ? ` ${message.slice(-400)}` : "";
  if (tail || message.startsWith("Could not start")) {
    return {
      status: 500,
      body: { error: `The RDKit backend failed to run.${tail}`.trim() },
    };
  }
  return null;
}
