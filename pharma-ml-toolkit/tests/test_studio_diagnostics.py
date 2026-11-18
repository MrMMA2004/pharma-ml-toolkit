import json

import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.studio_diagnostics import run_example_drift_diagnostic


def write_tool(tmp_path, body):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    script = tools / "diagnostic.py"
    script.write_text(body)
    return script


def test_child_diagnostic_preserves_strict_drift_status(tmp_path):
    payload = {"strict_check_exit_code": 1, "strict_drift": ["a.csv differs"], "scope": "strict"}
    script = write_tool(tmp_path, f"import json, sys\nprint(json.dumps({payload!r}))\nsys.exit(1)\n")
    result = run_example_drift_diagnostic(script)
    assert result == payload and result["strict_check_exit_code"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tools"]


@pytest.mark.parametrize("body", [
    "import sys; sys.exit(2)",
    "print('bad json')",
    "import json; print(json.dumps({'strict_check_exit_code': 0, 'strict_drift': ['x']}))",
    f"import json; print(json.dumps({json.dumps({'strict_check_exit_code': 1, 'strict_drift': []})}))",
])
def test_bad_child_result_refused(tmp_path, body):
    with pytest.raises(DataValidationError):
        run_example_drift_diagnostic(write_tool(tmp_path, body))


def test_missing_tool_reported(tmp_path):
    with pytest.raises(DataValidationError, match="unavailable"):
        run_example_drift_diagnostic(tmp_path / "absent.py")
