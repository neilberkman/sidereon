#!/usr/bin/env python3
"""Generate SciPy trajectory fixtures for small dense NLLS problems."""

from __future__ import annotations

import importlib
import json
import platform
import struct
from pathlib import Path

import numpy as np
import scipy


F8 = np.dtype("f8")
SEED = 20260611
N_CASES = 25


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_flat(values) -> list[str]:
    return [f64_bits(v) for v in np.asarray(values, dtype=F8).ravel(order="C")]


def residual_values(matrix: np.ndarray, target: np.ndarray, x: np.ndarray) -> np.ndarray:
    linear = np.empty(matrix.shape[0], dtype=F8)
    for i in range(matrix.shape[0]):
        linear[i] = matrix[i, 0] * x[0] + matrix[i, 1] * x[1] + matrix[i, 2] * x[2]
    curved = np.array(
        [
            np.sin(0.25 * x[0]) + 0.01 * x[1] * x[1],
            np.cos(0.20 * x[1]) - 0.02 * x[0] * x[2],
            np.sin(0.15 * x[2]) + 0.03 * x[0],
        ],
        dtype=F8,
    )
    out = linear - target
    out[:3] += curved
    return np.asarray(out, dtype=F8)


def build_case(rng: np.random.Generator, case_index: int):
    m = 5 + case_index % 8
    matrix = rng.normal(0.0, 0.8, (m, 3)).astype(F8)
    truth = rng.normal(0.0, 4.0, 3).astype(F8)
    target = np.empty(m, dtype=F8)
    for i in range(m):
        target[i] = matrix[i, 0] * truth[0] + matrix[i, 1] * truth[1] + matrix[i, 2] * truth[2]
    target[:3] += np.array(
        [
            np.sin(0.25 * truth[0]) + 0.01 * truth[1] * truth[1],
            np.cos(0.20 * truth[1]) - 0.02 * truth[0] * truth[2],
            np.sin(0.15 * truth[2]) + 0.03 * truth[0],
        ],
        dtype=F8,
    )
    target += rng.normal(0.0, 1e-5, m)
    x0 = truth + rng.normal(0.0, 0.75, 3)
    return matrix, target.astype(F8), x0.astype(F8)


def main() -> None:
    lsq_mod = importlib.import_module("scipy.optimize._lsq.least_squares")
    trf_mod = importlib.import_module("scipy.optimize._lsq.trf")
    # scipy >=1.x computes the finite-difference Jacobian inside VectorFunction,
    # which calls approx_derivative imported into _differentiable_functions (now
    # returning a (J, dct) tuple), so intercept it there rather than on
    # least_squares, which no longer exposes the symbol.
    df_mod = importlib.import_module("scipy.optimize._differentiable_functions")

    original_approx_derivative = df_mod.approx_derivative
    original_svd = trf_mod.svd

    rng = np.random.default_rng(SEED)
    cases = []

    try:
        for case_index in range(N_CASES):
            matrix, target, x0 = build_case(rng, case_index)
            fun_calls = []
            jac_calls = []
            svd_calls = []

            def fun(x):
                f = residual_values(matrix, target, x)
                fun_calls.append({"x": bits_flat(x), "f": bits_flat(f)})
                return f

            def approx_derivative_wrapped(fun_arg, x, *args, **kwargs):
                result = original_approx_derivative(fun_arg, x, *args, **kwargs)
                # scipy >=1.x returns (J, dct); older returned a bare J.
                jacobian = result[0] if isinstance(result, tuple) else result
                jac_calls.append(
                    {
                        "x": bits_flat(x),
                        "f0": bits_flat(kwargs["f0"]),
                        "jac": bits_flat(jacobian),
                    }
                )
                return result

            def svd_wrapped(a, full_matrices=False):
                u, s, vt = original_svd(a, full_matrices=full_matrices)
                svd_calls.append(
                    {
                        "a": bits_flat(a),
                        "u": bits_flat(u),
                        "s": bits_flat(s),
                        "vt": bits_flat(vt),
                    }
                )
                return u, s, vt

            df_mod.approx_derivative = approx_derivative_wrapped
            trf_mod.svd = svd_wrapped

            result = lsq_mod.least_squares(fun, x0, gtol=1e-10)

            cases.append(
                {
                    "name": f"small_dense_{case_index:02d}",
                    "m": int(target.size),
                    "x0": bits_flat(x0),
                    "matrix": bits_flat(matrix),
                    "target": bits_flat(target),
                    "fun_calls": fun_calls,
                    "jac_calls": jac_calls,
                    "svd_calls": svd_calls,
                    "result": {
                        "x": bits_flat(result.x),
                        "cost": f64_bits(result.cost),
                        "fun": bits_flat(result.fun),
                        "jac": bits_flat(result.jac),
                        "grad": bits_flat(result.grad),
                        "optimality": f64_bits(result.optimality),
                        "nfev": int(result.nfev),
                        "njev": int(result.njev),
                        "status": int(result.status),
                    },
                }
            )
    finally:
        df_mod.approx_derivative = original_approx_derivative
        trf_mod.svd = original_svd

    payload = {
        "schema": "trust-region-least-squares-trf-small-dense-v1",
        "reference": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "method": "least_squares(fun, x0, gtol=1e-10)",
            "jac": "2-point",
            "loss": "linear",
            "x_scale": "1.0",
            "tr_solver": "exact",
        },
        "seed": SEED,
        "cases": cases,
    }

    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "trf_small_dense.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
