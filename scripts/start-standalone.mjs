#!/usr/bin/env node
/* The PowerShell-safe standalone-server launcher (part15 roadmap #3, r342).
 *
 * The default root "start" script uses the POSIX env-prefix syntax
 * `NODE_ENV=production bun ... 2>&1 | tee server.log` — the
 * VAR=value prefix is unparseable under Windows cmd.exe/PowerShell.
 * This variant sets NODE_ENV in-process, spawns the standalone
 * server with inherited stdio and a faithful exit-code passthrough,
 * zero shell constructs. Wired as "start:portable".
 * Dev-only lane: the console is sandbox-only today (the roadmap note).
 */
import { spawn } from 'node:child_process';

const target = '.next/standalone/server.js';
const env = { ...process.env, NODE_ENV: 'production' };

const child = spawn(process.execPath, [target], {
  stdio: 'inherit',
  env,
  shell: false,
});

child.on('error', (err) => {
  console.error(`[start-standalone] spawn failed: ${err.message}`);
  process.exit(1);
});

child.on('close', (code, signal) => {
  if (signal) {
    console.error(`[start-standalone] child terminated by ${signal}`);
    process.exit(1);
  }
  process.exit(code ?? 0);
});
