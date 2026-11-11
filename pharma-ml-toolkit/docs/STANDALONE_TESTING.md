# Standalone release testing

Release ZIPs intentionally exclude `scratch/`, `download/` and `dist-archive/`.
The original example-data and API-index generators now live in the shipped
`tools/` directory. From the toolkit root:

```bash
python tools/build_example_data.py --check
python tools/build_docs_index.py  # rewrites docs/INDEX.md
python tools/reexec_notebooks.py   # executes all notebooks; modifies their outputs
python tools/strip_notebook_outputs.py  # refreshes the seven no-output twins
```

The data check needs the scientific dependencies used to build the inventory;
for the verified v1.2.172 baseline use `requirements.txt` and RDKit 2026.9.1.
It is an audit, not a data-generation command. Other RDKit versions may report
legitimate byte drift and should not be treated as equivalent without evaluation.
The scratch paths in a full checkout are command-line compatibility forwarders;
they are not present in a ZIP.

A release tree has neither checkout-only scratch shims nor a Git origin. To run
tests against an extracted tree with its dependencies installed, explicitly use:

```bash
python -m pytest --standalone-release tests/test_data_provenance.py tests/test_checkpoint_tooling.py tests/test_docs_index.py tests/test_notebook_variants.py -o addopts="-q --tb=short"
```

This selection also requires pytest, nbformat and the pinned RDKit. The option
skips only tests marked `checkout_only`; they remain active by default in a full
checkout, so a missing shim or Git origin there is still a failure. Other tests
may independently skip when optional components or the web source are absent.
This selection is a targeted portability profile, **not** the full release suite
or proof of engine execution. See the [Part 5 audit](PART5_PHASE0_AUDIT.md) for
actual observed results and remaining work.

`tests/test_release_portability.py` makes a temporary candidate ZIP using the
release builder's file-selection rules and tests an isolated extraction; it does
not invoke the publication routine or overwrite/delete any release archive.
