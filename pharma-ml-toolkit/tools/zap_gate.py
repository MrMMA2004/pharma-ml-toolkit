#!/usr/bin/env python
"""The ZAP severity gate (part17 §4.4, Round 356 P1).

Reads a ZAP JSON report, applies the project's documented
false-positive suppression list, and exits non-zero when any
non-suppressed finding is Critical or High — the workflow's agreed
failure policy ("Fail on agreed critical/high findings").

The suppression list (``packaging/security/zap-suppressions.json``)
is the ONLY suppression channel, and every entry must carry:

* ``pluginid`` + ``regex`` — what is suppressed;
* ``justification`` — why it is a false positive (never blanket);
* ``expires`` — an ISO date after which the entry is dead (the
  standing rule: expired suppressions are silently ignored, so the
  finding comes back and the gate fails again until re-reviewed).

Usage (from pharma-ml-toolkit/, exercised by the CI workflow):

    python tools/zap_gate.py \\
        --report "$RUNNER_TEMP/zap-report.json" \\
        --suppress packaging/security/zap-suppressions.json

Exit codes: 0 = no Critical/High survivors; 1 = survivors (the
workflow fails); 2 = malformed inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: riskcode -> severity name (ZAP's own mapping: 3=High, 4=Critical).
RISK_NAMES = {
    0: "Informational",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical",
}

#: the failure threshold: anything at or above this exits 1.
FAIL_AT_OR_ABOVE = 3


@dataclass
class Verdict:
    """The gate's decision record (printed as the workflow's summary)."""

    counts: dict[str, int] = field(default_factory=dict)
    suppressed: int = 0
    expired: int = 0
    exit_code: int = 0

    def summary(self) -> str:
        parts = [f"{name}={n}" for name, n in sorted(self.counts.items())]
        parts.append(f"suppressed={self.suppressed}")
        parts.append(f"expired_suppressions={self.expired}")
        return " ".join(parts)


def _today_iso() -> str:
    import datetime

    return datetime.date.today().isoformat()


def load_suppressions(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("suppression file must be a JSON list")
    for entry in raw:
        for key in ("pluginid", "regex", "justification", "expires"):
            if key not in entry:
                raise ValueError(f"suppression entry missing {key!r}: {entry}")
        if not re.search(r"\d{4}-\d{2}-\d{2}", entry["expires"]):
            raise ValueError(f"suppression expires must be ISO date: {entry}")
    return raw


def _suppression_applies(
    alert: dict, suppression: dict, today: str
) -> bool | None:
    """True = suppressed, False = not this entry, None = entry expired."""
    if str(alert.get("pluginid")) != str(suppression["pluginid"]):
        return False
    if not re.search(suppression["regex"], str(alert.get("name", ""))):
        return False
    if suppression["expires"] < today:
        return None  # expired: the finding comes back
    return True


def gate_report(
    report_path: Path, suppressions_path: Path | None = None
) -> Verdict:
    """Evaluate one ZAP JSON report against the severity policy."""
    today = _today_iso()
    suppressions = load_suppressions(suppressions_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    verdict = Verdict()
    sites = payload.get("site", []) if isinstance(payload, dict) else []
    for site in sites:
        for alert in site.get("alerts", []):
            risk = int(alert.get("riskcode", 0))
            name = RISK_NAMES.get(risk, f"riskcode-{risk}")
            verdict.counts[name] = verdict.counts.get(name, 0) + 1
            if risk < FAIL_AT_OR_ABOVE:
                continue
            state = None
            for suppression in suppressions:
                applies = _suppression_applies(alert, suppression, today)
                if applies is None:
                    verdict.expired += 1
                elif applies:
                    state = True
                    break
            if state:
                verdict.suppressed += 1
            else:
                verdict.exit_code = 1
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--suppress", type=Path, required=True,
        help="the documented suppression list (JSON)",
    )
    args = parser.parse_args(argv)
    try:
        verdict = gate_report(args.report, args.suppress)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"zap-gate malformed input: {exc}", file=sys.stderr)
        return 2
    print(
        f"zap-gate: {verdict.summary()} "
        f"-> {'FAIL' if verdict.exit_code else 'PASS'}"
    )
    return verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
