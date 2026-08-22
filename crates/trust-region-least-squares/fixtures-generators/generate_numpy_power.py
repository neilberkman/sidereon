#!/usr/bin/env python3
"""Generate NumPy `power` fixtures for the host-numerics power seam.

Two oracles are recorded per case, because NumPy dispatches them differently:

* ``vector_bits`` -- ``np.power(values, np.float64(exponent))`` over a
  contiguous f64 array. Its inner loop applies the stride-0 scalar-exponent
  table before selecting the AVX-512 SVML kernel on a capable CPU or scalar
  ``npy_pow`` everywhere else. It is what ``HostNumerics::power`` must
  reproduce.
* ``scalar_bits`` -- ``np.float64(base) ** np.float64(exponent)`` element by
  element. NumPy scalars do not enter the ufunc's stride-0 table, so this is
  always ``npy_pow`` (the platform C ``pow``). It is what
  ``HostNumerics::power_scalar`` must reproduce.

The value set is deliberately adversarial: subnormals, the smallest normal,
values straddling 1, the extremes of the finite range, both signed zeros, both
infinities, and NaN. Exponents cover the two the robust losses dispatch
(``-0.5``, ``-1.5``), the unified hook's ``denom ** 3`` call (exponent ``3.0``),
and every stride-0 table row (``-1.0``, ``0.0``, ``0.5``, ``1.0``, ``2.0``).

Run this in the pinned virtualenv on the canonical reference host (see the
crate README's "Reproducibility scope"): the payload is platform-, CPU-, and
NumPy-version-specific, exactly like the other fixtures in this directory.

    python fixtures-generators/generate_numpy_power.py
"""

from __future__ import annotations

import json
import platform
import struct
from pathlib import Path

import numpy as np


F8 = np.dtype("f8")
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "numpy_power.json"

# Bases, in the exact order the Rust test replays them. The length crosses more
# than two eight-lane SVML groups so the tail handling is exercised too.
BASES: list[float] = [
    0.0,
    -0.0,
    float(np.finfo(F8).tiny) * 0.5,  # subnormal
    float(np.finfo(F8).tiny),  # smallest normal
    1e-300,
    0.25,
    0.9999999999999999,
    1.0,
    1.0000000000000002,
    2.0,
    4.0,
    9.0,
    1.5e10,
    1e300,
    float(np.finfo(F8).max),
    -1.0,
    -4.0,
    float("inf"),
    float("-inf"),
    float("nan"),
]

EXPONENTS: list[float] = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_array(values) -> list[str]:
    return [f64_bits(x) for x in np.asarray(values, dtype=F8).ravel(order="C")]


def case(exponent: float) -> dict[str, object]:
    values = np.asarray(BASES, dtype=F8, order="C")
    with np.errstate(all="ignore"):
        vector = np.power(values, np.float64(exponent))
        scalars = [np.float64(base) ** np.float64(exponent) for base in values]
    return {
        "name": f"exponent_{exponent!r}",
        "exponent": f64_bits(exponent),
        "vector_call": "np.power(values, np.float64(e))",
        "scalar_call": "np.float64(b) ** np.float64(e)",
        "values": bits_array(values),
        "vector_bits": bits_array(vector),
        "scalar_bits": bits_array(scalars),
    }


def main() -> None:
    document = {
        "schema": "trust-region-least-squares-numpy-power-v1",
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cases": [case(exponent) for exponent in EXPONENTS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {OUT} ({len(document['cases'])} cases)")


if __name__ == "__main__":
    main()
