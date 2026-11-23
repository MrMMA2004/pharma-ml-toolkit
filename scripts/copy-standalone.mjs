#!/usr/bin/env node
/* The PowerShell-safe standalone-copy step (part15 roadmap #3, r342).
 *
 * The default root "build" script chains POSIX `cp -r` — fine under
 * the sandbox's bun/POSIX shell, but a hard stop under Windows
 * cmd.exe/PowerShell where `cp` does not exist. This is the portable
 * variant: a pure-Node recursive copy with identical semantics
 * (.next/static -> .next/standalone/.next/static and
 * public -> .next/standalone/public), zero shell constructs.
 * Wired as "build:portable" (next build && node scripts/copy-standalone.mjs).
 * Dev-only lane: the console is sandbox-only today (the roadmap note).
 */
import fs from 'node:fs';

const steps = [
  ['.next/static', '.next/standalone/.next/static'],
  ['public', '.next/standalone/public'],
];

for (const [from, to] of steps) {
  const src = `${process.cwd()}/${from}`;
  const dst = `${process.cwd()}/${to}`;
  if (!fs.existsSync(src)) {
    console.error(`[copy-standalone] missing source: ${from} (run "next build" first)`);
    process.exit(1);
  }
  fs.cpSync(src, dst, { recursive: true });
  console.log(`[copy-standalone] ${from} -> ${to}`);
}
