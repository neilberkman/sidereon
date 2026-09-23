#!/usr/bin/env python3
"""Generate the TLE assumed-decimal spelling fixture from python-sgp4.

For B* and the second mean-motion derivative, python-sgp4's `export_tle`
formats `value * 10` with `'{: 4.4e}'` and rewrites the exponent
(`sgp4.exporter._abbreviate_rate`, with `'+0'` for B* and `'-0'` for the
second derivative). That spelling fits the eight TLE columns when its
exponent is from -9 to 9; the fixture records it for those values. Below
1e-10 python-sgp4 writes a two-digit exponent, so the fixture records the
spelling at exponent -9 with the digits rounded half to even from the exact
decimal expansion of the value, and zero of the value's sign when those
digits are 00000. A value whose exponent exceeds 9 is recorded as refused
(null).

Every B* expectation from `_abbreviate_rate` is checked against the full
`export_tle` line of a satellite initialized with that B*.

Run with python-sgp4 2.22:
    python fixtures-generators/generate_tle_assumed_decimal.py
"""

from __future__ import annotations

import json
import math
import random
import struct
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import sgp4
from sgp4.api import WGS72, Satrec
from sgp4.exporter import _abbreviate_rate, export_tle

OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/tle/assumed_decimal_sgp4_exporter.json"
FIELDS = {"bstar": "+0", "mean_motion_double_dot": "-0"}


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def spelling(value: float, zero_exponent: str) -> str | None:
    text = _abbreviate_rate(value * 10.0, zero_exponent)[:-1]
    formatted = "{0: 4.4e}".format(value * 10.0)
    exponent = int(formatted.split("e")[1])
    if exponent > 9:
        return None
    if exponent >= -9:
        assert len(text) == 8, (value, text)
        return text
    sign = "-" if math.copysign(1.0, value) < 0 else " "
    digits = Decimal(abs(value)).quantize(Decimal("1e-14"), rounding=ROUND_HALF_EVEN)
    digits = f"{digits:.14f}"[-5:]
    if digits == "00000":
        return f"{sign}00000{zero_exponent}"
    return f"{sign}{digits}-9"


def export_tle_bstar(value: float) -> str:
    sat = Satrec()
    sat.sgp4init(WGS72, "i", 5, 18441.785, value, 0.0, 0.0, 0.1859667,
                 5.7904160274885, 0.5980929187319, 0.3373093125574,
                 0.0472294454407, 6.0863854713832)
    return export_tle(sat)[0][53:61]


def exact_ties(rng: random.Random) -> list[float]:
    """Values `v` with `v * 10` exactly halfway between two five-digit mantissas."""
    ties = []
    while len(ties) < 400:
        if rng.random() < 0.5:
            t = (rng.randrange(10000, 100000) + 0.5) * 10.0 ** rng.randrange(0, 5)
        else:
            k = rng.randrange(1, 24)
            t = rng.randrange(1, 2 ** 20) * 2 + 1
            t = t / 2.0 ** k
        d = Decimal(t).normalize()
        if len(d.as_tuple().digits) != 6 or d.as_tuple().digits[-1] != 5:
            continue
        for v in (t / 10.0, -t / 10.0):
            if Decimal(v * 10.0) == Decimal(t) * (1 if v > 0 else -1):
                ties.append(v)
    return ties


def main() -> None:
    rng = random.Random(0x7E1E)
    values = [
        0.0, -0.0, 3.21675e-9, 8233.15, 2.5e-14, -2.5e-14, 1.5e-14, 3.5e-14,
        1.0e-10, 9.999996e-11, 9.99995e-11, 0.99999e-9, 0.999996e-9,
        -2.3456789e-12, 4.0e-15, -4.0e-15, 5.0e-15, 6.0e-15, 5.0e-11,
        1.009e-5, 3.1745e-5, 1.7172e-4, 0.5, -0.5, 0.099999, 0.0999996,
        0.999994e9, 0.999995e9, 0.999996e9, 1.0e9, 1.0e12,
        2.2250738585072014e-308, -5e-324,
    ]
    values += exact_ties(rng)
    for _ in range(4000):
        v = 10.0 ** rng.uniform(-20.0, 12.0) * rng.choice((1.0, -1.0))
        if rng.random() < 0.4:
            v = float(f"{v:.{rng.randrange(0, 7)}e}")
        values.append(v)

    cases = []
    for v in values:
        case = {"value": f64_bits(v)}
        for field, zero in FIELDS.items():
            case[field] = spelling(v, zero)
        if v == 0.0 or (case["bstar"] is not None and 1e-10 <= abs(v) < 1e9):
            exported = export_tle_bstar(v)
            assert exported == case["bstar"], (v, exported, case["bstar"])
        cases.append(case)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "generator": "fixtures-generators/generate_tle_assumed_decimal.py",
        "sgp4_version": sgp4.__version__,
    }
    lines = [json.dumps(case) for case in cases]
    body = json.dumps(header)[:-1] + ', "cases": [\n' + ",\n".join(lines) + "\n]}\n"
    OUT.write_text(body)


if __name__ == "__main__":
    main()
