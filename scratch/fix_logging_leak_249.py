#!/usr/bin/env python
"""round-249 -- convert the eight module-level ``logging.disable(CRITICAL)``
calls in tests/ into scoped autouse fixtures.

The module-level call leaked for the WHOLE pytest process: any sibling
module collected into the same pytest process lost all logging, so its
``caplog`` assertions saw empty text.  The nine-chunk gate (round 249's
size-balanced grouping) caught the first materialisation:
test_format_converters.py + test_numpy_neural_checkpoints.py in one
chunk -> the two §20.5 warning-discard tests failed.  The suite's own
``_visible_logs`` workarounds (test_descriptors, test_docking,
test_split_categorical) existed precisely because of this leak.

Same silence, correct scope: the fixture applies CRITICAL-disable around
each test of the SAME module only and restores the previous level after.
"""
from __future__ import annotations

from pathlib import Path

FILES = [
    "tests/test_approvals.py",
    "tests/test_chem_standardize.py",
    "tests/test_chem_tables.py",
    "tests/test_cli_features.py",
    "tests/test_csv_standardize.py",
    "tests/test_format_converters.py",
    "tests/test_generative.py",
    "tests/test_split_categorical.py",
]

OLD = "logging.disable(logging.CRITICAL)\n"
NEW = '''# (round-249) the module-level ``logging.disable(logging.CRITICAL)``
# this fixture replaces leaked for the WHOLE pytest process and silently
# broke ``caplog`` assertions in sibling modules sharing a pytest process
# (the nine-chunk gate caught it: test_format_converters before
# test_numpy_neural_checkpoints).  Same silence, correct scope: applied
# around each test of THIS module only, previous level restored after.
@pytest.fixture(autouse=True)
def _quiet_module_logging():
    prev = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(prev)
'''


def main() -> int:
    root = Path("/home/z/my-project/pharma-ml-toolkit")
    for rel in FILES:
        p = root / rel
        text = p.read_text()
        if OLD not in text:
            print(f"SKIP (pattern absent): {rel}")
            continue
        if text.count(OLD) != 1:
            print(f"SKIP (pattern not unique): {rel}")
            continue
        if "_quiet_module_logging" in text:
            print(f"SKIP (already converted): {rel}")
            continue
        p.write_text(text.replace(OLD, NEW, 1))
        print(f"converted: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
