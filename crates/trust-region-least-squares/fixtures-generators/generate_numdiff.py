#!/usr/bin/env python3
"""Generate SciPy dense two-point finite-difference fixtures."""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import (
    _adjust_scheme_to_bounds,
    _compute_absolute_step,
    approx_derivative,
)


F8 = np.dtype("f8")


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_array(values) -> list[str]:
    return [f64_bits(v) for v in np.asarray(values, dtype=F8).ravel(order="C")]


def random_coeff(rng: random.Random, spread: float) -> float:
    return rng.uniform(-spread, spread)


def model_value(case, x):
    out = []
    n = case["n"]
    pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]
    for i in range(case["m"]):
        acc = case["bias"][i]
        for j in range(n):
            acc += case["linear"][i][j] * x[j]
        for j in range(n):
            acc += case["quadratic"][i][j] * x[j] * x[j]
        for p, (j, k) in enumerate(pairs):
            acc += case["cross"][i][p] * (x[j] - 0.25 * x[k]) * (x[k] + 0.5)
        out.append(acc)
    return out


def make_cases():
    rng = random.Random(0x54_33)
    cases = []
    special_x0 = [
        [0.0],
        [-0.0, 1.0],
        [-1.0, 0.5, 2.0],
        [1e-12, -1e-12, 10.0, -10.0],
        [1e8, -1e8, 3.25, -4.5, 0.0],
    ]

    for case_id in range(20):
        n = 1 + (case_id % 5)
        m = 1 + ((case_id * 3) % 6)
        if case_id < len(special_x0):
            x0 = list(special_x0[case_id])
        else:
            x0 = []
            for j in range(n):
                selector = (case_id + 3 * j) % 4
                if selector == 0:
                    x0.append(rng.uniform(-20.0, 20.0))
                elif selector == 1:
                    x0.append(rng.uniform(-1e-6, 1e-6))
                elif selector == 2:
                    x0.append(rng.uniform(-1e4, 1e4))
                else:
                    x0.append(rng.choice([-1.0, 1.0]) * 2.0 ** rng.randint(-20, 20))

        pairs = n * (n - 1) // 2
        case = {
            "id": case_id,
            "n": n,
            "m": m,
            "x0": x0,
            "bias": [random_coeff(rng, 3.0) for _ in range(m)],
            "linear": [[random_coeff(rng, 0.75) for _ in range(n)] for _ in range(m)],
            "quadratic": [[random_coeff(rng, 0.01) for _ in range(n)] for _ in range(m)],
            "cross": [[random_coeff(rng, 0.005) for _ in range(pairs)] for _ in range(m)],
        }

        x0_arr = np.asarray(case["x0"], dtype=F8)
        f0 = np.asarray(model_value(case, case["x0"]), dtype=F8)
        calls = []

        def fun(x):
            calls.append([float(v) for v in x])
            return np.asarray(model_value(case, [float(v) for v in x]), dtype=F8)

        jac = approx_derivative(
            fun,
            x0_arr,
            method="2-point",
            rel_step=None,
            f0=f0,
            bounds=(-np.inf, np.inf),
        )
        jac2d = np.atleast_2d(jac)
        h = _compute_absolute_step(None, x0_arr, f0, "2-point")
        h_adjusted, use_one_sided = _adjust_scheme_to_bounds(
            x0_arr,
            h,
            1,
            "1-sided",
            np.full_like(x0_arr, -np.inf),
            np.full_like(x0_arr, np.inf),
        )

        cases.append(
            {
                "id": case_id,
                "n": n,
                "m": m,
                "x0_bits": bits_array(case["x0"]),
                "bias_bits": bits_array(case["bias"]),
                "linear_bits": [bits_array(row) for row in case["linear"]],
                "quadratic_bits": [bits_array(row) for row in case["quadratic"]],
                "cross_bits": [bits_array(row) for row in case["cross"]],
                "f0_bits": bits_array(f0),
                "h_bits": bits_array(h_adjusted),
                "use_one_sided": [bool(v) for v in use_one_sided.tolist()],
                "evaluation_points_bits": [bits_array(point) for point in calls],
                "jacobian_bits": [bits_array(row) for row in jac2d],
            }
        )

    return cases


def main() -> None:
    import scipy

    payload = {
        "schema": "trust-region-least-squares-numdiff-v1",
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "cases": make_cases(),
    }
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "numdiff_2point.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
