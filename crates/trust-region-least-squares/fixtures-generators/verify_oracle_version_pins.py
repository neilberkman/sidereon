#!/usr/bin/env python3
"""Reproduce the cross-version SciPy and NumPy oracle-pin measurements."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys


PROBE = r"""
import gc
import hashlib
import json
import platform
import sys

import numpy as np
import scipy
from scipy.interpolate import splev, splrep


def hex_bytes(values):
    return np.asarray(values, dtype=np.float64).tobytes().hex()


x = np.linspace(0.0, 1.0, 9)
y = np.sin(7.0 * x) + 0.125 * x * x
queries = np.linspace(0.0, 1.0, 17)
spline_cases = {}
for smoothing in (0.0, 0.01):
    knots, coefficients, degree = splrep(x, y, s=smoothing)
    used = len(knots) - degree - 1
    values = splev(queries, (knots, coefficients, degree))
    spline_cases[str(smoothing)] = {
        "knots": hex_bytes(knots),
        "coefficients": hex_bytes(coefficients),
        "used": used,
        "degree": degree,
        "values": hex_bytes(values),
        "tail": [float(value).hex() for value in coefficients[used:]],
    }

nonzero_tails = 0
addresses = set()
for index in range(10_000):
    knots, coefficients, degree = splrep(x, y, s=0.01)
    used = len(knots) - degree - 1
    addresses.add(int(coefficients.ctypes.data))
    if np.any(coefficients[used:] != 0.0):
        nonzero_tails += 1
    coefficients[:] = np.float64.fromhex("0x1.23456789abcp+40")
    del knots, coefficients
    if index % 100 == 0:
        gc.collect()

rng = np.random.default_rng(20260821)
pinv_input = rng.normal(size=(9, 6))
pinv_output = np.linalg.pinv(pinv_input)

sweep = []
for kind in ("random", "hilbert", "near"):
    for n in range(2, 21):
        for seed in range(5):
            if kind == "hilbert":
                indices = np.arange(n, dtype=np.float64)
                matrix = 1.0 / (indices[:, None] + indices[None, :] + 1.0)
            else:
                case_rng = np.random.default_rng(seed)
                matrix = case_rng.normal(size=(n + 2, n))
                if kind == "near":
                    matrix[:, -1] = (
                        matrix[:, 0]
                        + np.float64(1e-8) * matrix[:, 1]
                        + np.float64(1e-10) * matrix[:, 2 % n]
                    )
            output = np.linalg.pinv(matrix)
            sweep.append(hashlib.sha256(output.tobytes()).hexdigest())

print(json.dumps({
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "spline_cases": spline_cases,
    "allocator_reuse": {
        "fits": 10_000,
        "addresses": len(addresses),
        "nonzero_tails": nonzero_tails,
    },
    "pinv": {
        "input": hex_bytes(pinv_input),
        "input_sha256": hashlib.sha256(pinv_input.tobytes()).hexdigest(),
        "output": hex_bytes(pinv_output),
        "output_sha256": hashlib.sha256(pinv_output.tobytes()).hexdigest(),
        "sweep_sha256": sweep,
    },
}))
"""


def run_probe(python: str) -> dict:
    output = subprocess.check_output([python, "-c", PROBE], text=True)
    return json.loads(output)


def floats(payload: str) -> list[float]:
    return list(memoryview(bytes.fromhex(payload)).cast("d"))


def max_ulp(left: list[float], right: list[float]) -> int:
    if len(left) != len(right):
        raise ValueError("float arrays have different lengths")
    maximum = 0
    for lhs, rhs in zip(left, right, strict=True):
        lhs_bits = int.from_bytes(memoryview(bytearray(struct_pack(lhs))), sys.byteorder)
        rhs_bits = int.from_bytes(memoryview(bytearray(struct_pack(rhs))), sys.byteorder)
        if (lhs_bits >> 63) != (rhs_bits >> 63):
            raise ValueError("float signs differ")
        maximum = max(maximum, abs(lhs_bits - rhs_bits))
    return maximum


def struct_pack(value: float) -> bytes:
    import struct

    return struct.pack("=d", value)


def compare(old: dict, new: dict) -> dict:
    spline = {}
    for smoothing in ("0.0", "0.01"):
        old_case = old["spline_cases"][smoothing]
        new_case = new["spline_cases"][smoothing]
        used = old_case["used"]
        old_coefficients = floats(old_case["coefficients"])
        new_coefficients = floats(new_case["coefficients"])
        spline[smoothing] = {
            "knots_bit_identical": old_case["knots"] == new_case["knots"],
            "used_coefficients": used,
            "old_tail": old_case["tail"],
            "new_tail": new_case["tail"],
            "coefficient_max_ulp": max_ulp(
                old_coefficients[:used], new_coefficients[:used]
            ),
            "evaluation_max_ulp": max_ulp(
                floats(old_case["values"]), floats(new_case["values"])
            ),
        }

    old_sweep = old["pinv"]["sweep_sha256"]
    new_sweep = new["pinv"]["sweep_sha256"]
    differing_sweep_cases = sum(
        old_digest != new_digest
        for old_digest, new_digest in zip(old_sweep, new_sweep, strict=True)
    )
    return {
        "environments": {
            "old": {
                key: old[key] for key in ("python", "platform", "numpy", "scipy")
            },
            "new": {
                key: new[key] for key in ("python", "platform", "numpy", "scipy")
            },
        },
        "scipy_splrep": {
            "cases": spline,
            "allocator_reuse": {
                "old": old["allocator_reuse"],
                "new": new["allocator_reuse"],
            },
        },
        "numpy_pinv": {
            "input_sha256": old["pinv"]["input_sha256"],
            "input_bit_identical": old["pinv"]["input"] == new["pinv"]["input"],
            "old_output_sha256": old["pinv"]["output_sha256"],
            "new_output_sha256": new["pinv"]["output_sha256"],
            "fixed_matrix_max_ulp": max_ulp(
                floats(old["pinv"]["output"]), floats(new["pinv"]["output"])
            ),
            "sweep_cases": len(old_sweep),
            "differing_sweep_cases": differing_sweep_cases,
        },
    }


def main() -> None:
    old_python = os.environ.get("SIDEREON_ORACLE_OLD_PYTHON")
    new_python = os.environ.get("SIDEREON_ORACLE_NEW_PYTHON")
    if not old_python or not new_python:
        raise SystemExit(
            "set SIDEREON_ORACLE_OLD_PYTHON and SIDEREON_ORACLE_NEW_PYTHON"
        )
    result = compare(run_probe(old_python), run_probe(new_python))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
