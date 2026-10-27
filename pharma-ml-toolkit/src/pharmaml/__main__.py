"""``python -m pharmaml`` entry point.

Delegates to :func:`pharmaml.cli.main` so the CLI works both through the
installed console script (``pharmaml --help``) and through the module
invocation (``python -m pharmaml --help``) -- the latter matters inside
the Docker image and in ``pip install -e .`` checkouts where the
console script is not necessarily on ``PATH``.

Examples
--------
>>> import subprocess, sys
>>> proc = subprocess.run(
...     [sys.executable, "-m", "pharmaml", "--version"],
...     capture_output=True, text=True,
... )
>>> proc.returncode
0
>>> "pharmaml" in proc.stdout.lower()
True
"""

from __future__ import annotations

import sys

from pharmaml.cli import main

if __name__ == "__main__":
    sys.exit(main())
