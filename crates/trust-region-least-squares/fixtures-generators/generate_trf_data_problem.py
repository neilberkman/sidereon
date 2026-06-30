#!/usr/bin/env python3
"""Generate SciPy ``least_squares`` (TRF) trajectory fixtures for the data-driven
``BuiltinResidual`` kinds.

This is the counterpart of ``generate_trf_general.py`` for the ``data`` module:
it spans each built-in residual kind (``linear``, ``polynomial``,
``exponential``) crossed with every SciPy loss (``linear``, ``soft_l1``,
``huber``, ``cauchy``, ``arctan``) and records the problem (the kind's data
arrays plus ``x0``) together with the converged ``result``. The Rust test
rebuilds each ``DataProblem`` and replays it through ``solve_data_problem_with``
with the host-LAPACK ``ThinSvd`` backend, asserting the full result is
bit-identical.

Each residual is defined to be reproducible scalar-for-scalar in both Python and
Rust (it mirrors ``BuiltinResidual::eval_residual`` in ``src/data.rs``): the
linear/polynomial combinations are sequential scalar accumulations (not
``np.dot``) so there is no BLAS/pairwise ambiguity in the user residual, and the
exponential uses ``np.exp`` elementwise. Bit-exactness of the SciPy *solver*
trajectory comes from the injected LAPACK/BLAS backend.

Reference: ``scipy/optimize/_lsq/least_squares.py`` (default ``jac='2-point'``).
Run inside the pinned environment in ``requirements.txt``.
"""

from __future__ import annotations

import importlib
import json
import platform
import struct
from pathlib import Path

import numpy as np
import scipy


F8 = np.dtype("f8")
SEED = 20260629
LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_flat(values) -> list[str]:
    return [f64_bits(v) for v in np.asarray(values, dtype=F8).ravel(order="C")]


def linear_residual(a, b, m, n, x):
    """Mirrors BuiltinResidual::Linear: residual_i = (sum_j a[i*n+j]*x[j]) - b[i]."""
    out = np.empty(m, dtype=F8)
    for i in range(m):
        acc = np.float64(0.0)
        for j in range(n):
            acc = acc + a[i * n + j] * x[j]
        out[i] = acc - b[i]
    return out


def poly_residual(degree, t, y, x):
    """Mirrors BuiltinResidual::Polynomial: Horner from the highest coefficient."""
    out = np.empty(len(t), dtype=F8)
    for i, ti in enumerate(t):
        acc = np.float64(x[degree])
        for k in range(degree - 1, -1, -1):
            acc = acc * ti + x[k]
        out[i] = acc - y[i]
    return out


def exp_residual(t, y, x):
    """Mirrors BuiltinResidual::Exponential: (x0*exp(x1*t)+x2) - y."""
    out = np.empty(len(t), dtype=F8)
    for i, ti in enumerate(t):
        out[i] = (x[0] * np.exp(x[1] * np.float64(ti)) + x[2]) - y[i]
    return out


def add_outliers(rng, values, loss):
    """Heavy-tailed contamination so robust losses actually bite. Linear loss is
    left clean (it ignores f_scale and would just chase the outliers)."""
    if loss == "linear":
        return values
    m = values.shape[0]
    contaminated = values + rng.normal(0.0, 1e-4, m).astype(F8)
    contaminated = contaminated + rng.standard_t(2.0, m).astype(F8) * np.float64(4.0)
    return contaminated.astype(F8)


def build_linear(rng, loss):
    n, m = 3, 9
    a = rng.normal(0.0, 0.8, m * n).astype(F8)
    truth = rng.normal(0.0, 3.0, n).astype(F8)
    base = linear_residual(a, np.zeros(m, dtype=F8), m, n, truth)  # = A @ truth
    b = add_outliers(rng, base + rng.normal(0.0, 1e-5, m).astype(F8), loss).astype(F8)
    x0 = (truth + rng.normal(0.0, 0.6, n)).astype(F8)
    case = {"kind": "linear", "m": int(m), "n": int(n), "a": bits_flat(a), "b": bits_flat(b)}
    fun = lambda x: linear_residual(a, b, m, n, x)
    return case, x0, fun


def build_polynomial(rng, loss):
    degree, m = 3, 12
    t = np.linspace(-1.5, 1.5, m).astype(F8)
    truth = rng.normal(0.0, 1.5, degree + 1).astype(F8)
    base = poly_residual(degree, t, np.zeros(m, dtype=F8), truth)  # = poly(truth, t)
    y = add_outliers(rng, base + rng.normal(0.0, 1e-5, m).astype(F8), loss).astype(F8)
    x0 = (truth + rng.normal(0.0, 0.4, degree + 1)).astype(F8)
    case = {"kind": "polynomial", "degree": int(degree), "t": bits_flat(t), "y": bits_flat(y)}
    fun = lambda x: poly_residual(degree, t, y, x)
    return case, x0, fun


def build_exponential(rng, loss):
    m = 12
    t = np.linspace(0.0, 2.0, m).astype(F8)
    truth = np.array([2.5, -0.6, 0.5], dtype=F8)
    base = exp_residual(t, np.zeros(m, dtype=F8), truth)  # = model(truth, t)
    y = add_outliers(rng, base + rng.normal(0.0, 1e-5, m).astype(F8), loss).astype(F8)
    x0 = (truth + np.array([0.3, 0.1, -0.2], dtype=F8)).astype(F8)
    case = {"kind": "exponential", "t": bits_flat(t), "y": bits_flat(y)}
    fun = lambda x: exp_residual(t, y, x)
    return case, x0, fun


BUILDERS = {
    "linear": build_linear,
    "polynomial": build_polynomial,
    "exponential": build_exponential,
}


def main() -> None:
    lsq_mod = importlib.import_module("scipy.optimize._lsq.least_squares")
    rng = np.random.default_rng(SEED)
    cases = []

    for kind in ("linear", "polynomial", "exponential"):
        for loss in LOSSES:
            f_scale = 1.0
            case, x0, fun = BUILDERS[kind](rng, loss)

            result = lsq_mod.least_squares(fun, x0, gtol=1e-10, loss=loss, f_scale=f_scale)

            case.update(
                {
                    "name": f"{kind}_{loss}",
                    "loss": loss,
                    "f_scale": f64_bits(f_scale),
                    "x0": bits_flat(x0),
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
            cases.append(case)

    payload = {
        "schema": "trust-region-least-squares-trf-data-problem-v1",
        "reference": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "method": "least_squares(fun, x0, gtol=1e-10, loss=..., f_scale=...)",
            "jac": "2-point",
            "kinds": ["linear", "polynomial", "exponential"],
            "losses": LOSSES,
            "x_scale": "1.0",
            "tr_solver": "exact",
        },
        "seed": SEED,
        "cases": cases,
    }

    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "trf_data_problem.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
