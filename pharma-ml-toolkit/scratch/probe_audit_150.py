"""Stdlib-only probe for the round-33 (100x-150) audit, Task 1-a.

The venv is mid-repair (rdkit missing), so this audit is STATIC; the ONE
runtime check below uses no pharmaml imports at all -- it pins the CPython
float-power overflow semantics behind audit_150.txt FINDING 2
(pic50_to_ic50's ``10.0 ** (exponent - pIC50)``).

Run: python scratch/probe_audit_150.py
"""

import math


def main() -> None:
    # The exact expression of converters.pic50_to_ic50 (nM exponent 9).
    cases = [
        (9.0, 6.0),     # normal: 1000.0
        (9.0, 0.0),     # pIC50 0 -> 1e9 nM
        (9.0, -299.25), # just inside the double range
        (9.0, -299.26), # just OUTSIDE -> ?
        (9.0, -300.0),  # clearly outside -> ?
        (9.0, 400.0),   # huge positive pIC50 (underflow direction)
        (12.0, -297.0), # pM unit boundary
    ]
    for exponent, pic in cases:
        try:
            value = 10.0 ** (exponent - pic)
            print(f"10.0**({exponent} - ({pic})) = {value!r}")
        except Exception as exc:  # noqa: BLE001 - the probe's whole point
            print(f"10.0**({exponent} - ({pic})) -> {type(exc).__name__}: {exc}")

    print("log10(DBL_MAX) =", math.log10(1.7976931348623157e308))

    # Reachability of the crash via the FORWARD converter's own outputs:
    # ic50_to_pic50 rejects non-finite values, so the largest |pIC50| it
    # can emit is log10(DBL_MAX * scale) -- check whether the round trip
    # can ever cross the crash boundary.
    for unit, scale in (("M", 1.0), ("nM", 1e-9), ("pM", 1e-12)):
        worst_pic50 = -math.log10(1.7976931348623157e308 * scale)
        exponent = {"M": 0.0, "nM": 9.0, "pM": 12.0}[unit]
        try:
            back = 10.0 ** (exponent - worst_pic50)
            print(
                f"unit={unit!r}: worst forward pIC50 {worst_pic50:.4f} -> "
                f"reverse {back!r} (no crash)"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"unit={unit!r}: worst forward pIC50 {worst_pic50:.4f} -> "
                f"{type(exc).__name__}"
            )


if __name__ == "__main__":
    main()
