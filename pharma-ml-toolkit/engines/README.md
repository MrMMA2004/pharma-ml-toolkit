# Engine binaries — the AutoDock trio this toolkit ships

This folder holds the three Windows console builds the docking campaign
runner drives. They are exactly the executables the user uploaded
(2026-10 round, tag 100x-171) and are picked up **automatically** — no
PATH setup, no configuration:

| file           | build            | role                                                    |
| -------------- | ---------------- | ------------------------------------------------------- |
| `vina.exe`     | PE32+ x86-64     | AutoDock Vina — the fast engine (single / batch / SMILES-array cards) |
| `autogrid4.exe`| PE32 i386        | AutoGrid4 — computes the affinity maps AutoDock4 docks into |
| `autodock4.exe`| PE32 i386        | AutoDock4 — the classic Lamarckian-GA engine (campaign / single folder) |

## Checksums (verify a copy against the upload)

```
052576a87455c30df8ef8bdd951d65ab  autodock4.exe
9ef694a508215d5c46ab072fba59e035  autogrid4.exe
51fd15a78c71da12958d55609c3720c7  vina.exe
```

## How the runner finds them

1. the folder picked with the Docking tab's **Engine binaries…**
   picker (the Campaign runner card) — searched first, per binary;
2. this bundled `engines/` folder next to `playground.py` — the
   default, so a fresh unzip just works on Windows;
3. `PATH` — native/conda installs still win for any binary the folder
   does not carry.

On Windows the `.exe` files run directly. On Linux/macOS a resolved PE
binary is routed through `wine` when installed (detected by the `MZ`
magic, not the file name); without wine the run fails with an
actionable error naming all three options.

## Licences (redistribution note)

* **AutoDock Vina** — Apache License 2.0 (redistributable).
* **AutoDock4 / AutoGrid4** — GNU GPL (the Scripps binaries,
  redistributed unmodified as received from the user's upload; the
  source is available from autodock.scripps.edu).
