# Engine binaries — the AutoDock trio this toolkit ships

This folder holds the three Windows console builds the docking campaign
runner drives. They are exactly the executables the user uploaded
(2026-10 round, tag 100x-171). Discovery does not prove that an engine is
runnable or that it produces valid results:

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

Backend discovery searches, in order: an explicit binary path, an explicitly
supplied `engine_dir` (bare name, then `.exe`), PATH (bare name, then `.exe`),
and finally the verified bundled copy. PATH precedes the bundle so a native
installation is not silently displaced. An explicit but unusable binary fails
rather than silently falling back. The toolkit-root `engines/` is detected in
source checkouts and extracted release ZIPs; it is not next to `playground.py`.
An installed wheel only sees bundled binaries if they are explicitly packaged
under `pharmaml/engines/`; the current wheel does not promise to include them.
The Studio Docking page is primarily preparation, inspection, parsing and
rescoring; it does not provide the Campaign runner picker described in older
documentation. Backend APIs can accept an `engine_dir`.

Bundled files are checked against fixed SHA-256 references and PE machine types.
Vina is x86-64; AutoGrid4 and AutoDock4 are x86 (32-bit). Windows x86-64 can
potentially run both; a 32-bit x86 host cannot run Vina's 64-bit binary. Other
host architectures are not assumed compatible. On Linux/macOS a PE binary needs
Wine; its presence alone does not prove that Wine, including 32-bit support,
or the engine works. Native installations can be used instead. Actual runtime
success must be checked separately; no docking score follows from discovery.

SHA-256 references:

```
36c0b16c04d7df8e6225737bae65bb04058d1f4c90accfaf2d230dbc913954bf  autodock4.exe
797efce687d1ae82df59726461e0e1966b3d8edb0f8b187f982fa1ab0c12da9e  autogrid4.exe
e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5  vina.exe
```

## Licences (redistribution note)

* **AutoDock Vina** — Apache License 2.0 (redistributable).
* **AutoDock4 / AutoGrid4** — GNU GPL (the Scripps binaries,
  redistributed unmodified as received from the user's upload; the
  source is available from autodock.scripps.edu).
