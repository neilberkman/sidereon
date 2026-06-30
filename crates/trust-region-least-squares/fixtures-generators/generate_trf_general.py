#!/usr/bin/env python3
"""Generate SciPy ``least_squares`` (TRF) trajectory fixtures across a range of
problem dimensions ``n``.

This is the general-``n`` counterpart of ``generate_trf_small_dense.py`` /
``generate_trf_loss.py``: it spans ``n`` in {2, 3, 4, 5, 6, 8} crossed with
every SciPy loss (``linear``, ``soft_l1``, ``huber``, ``cauchy``, ``arctan``)
and a couple of conditioning settings, and records the problem
(``matrix``/``target``/``x0``) together with the converged ``result``. The Rust
test replays each problem through ``trf_no_bounds`` with the host-LAPACK
``ThinSvd`` backend and asserts the full result is bit-identical.

The residual model is defined to be reproducible scalar-for-scalar in both
Python and Rust: the linear part is a sequential left-to-right accumulation and
the curvature uses only ``+``/``-``/``*`` and ``sin`` (verified bit-exact
against this target's libm). It is *not* ``np.dot`` so there is no BLAS/pairwise
ambiguity in the user residual itself; bit-exactness of the SciPy *solver*
trajectory comes from the injected LAPACK/BLAS backend.

Reference: ``scipy/optimize/_lsq/trf.py`` (``trf_no_bounds``) and
``scipy/optimize/_lsq/common.py``. Run inside the pinned environment in
``requirements.txt``.
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
SEED = 20260628
DIMS = [2, 3, 4, 5, 6, 8]
LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
# (f_scale, outlier_scale) per sub-case. outlier_scale > 0 seeds heavy-tailed
# residuals so robust reweighting actually bites and the huber mask exercises
# both the z<=1 and z>1 branches.
CONDITIONING = [
    (1.0, 0.0),
    (0.5, 6.0),
]


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_flat(values) -> list[str]:
    return [f64_bits(v) for v in np.asarray(values, dtype=F8).ravel(order="C")]


def residual_values(matrix: np.ndarray, target: np.ndarray, x: np.ndarray) -> np.ndarray:
    """General-n residual. Mirrors the Rust `residual_values` exactly.

    Linear part is a sequential accumulation (not np.dot) so the float-op order
    is identical to the Rust side; curvature uses sin and elementary ops only.
    """
    m, n = matrix.shape
    out = np.empty(m, dtype=F8)
    for i in range(m):
        acc = np.float64(0.0)
        for j in range(n):
            acc = acc + matrix[i, j] * x[j]
        a = x[i % n]
        b = x[(i + 1) % n]
        out[i] = (acc - target[i]) + np.sin(np.float64(0.25) * a) + np.float64(0.01) * b * b - np.float64(0.02) * a * b
    return out


def build_case(rng: np.random.Generator, n: int, m: int, outlier_scale: float):
    matrix = rng.normal(0.0, 0.8, (m, n)).astype(F8)
    truth = rng.normal(0.0, 3.0, n).astype(F8)
    # target is chosen so that residual_values(matrix, target, truth) is the
    # pure noise vector (the curvature cancels because it depends only on x).
    base = np.empty(m, dtype=F8)
    for i in range(m):
        acc = np.float64(0.0)
        for j in range(n):
            acc = acc + matrix[i, j] * truth[j]
        a = truth[i % n]
        b = truth[(i + 1) % n]
        base[i] = acc + np.sin(np.float64(0.25) * a) + np.float64(0.01) * b * b - np.float64(0.02) * a * b
    target = base + rng.normal(0.0, 1e-5, m).astype(F8)
    if outlier_scale > 0.0:
        target = target + (rng.standard_t(2.0, m).astype(F8) * outlier_scale)
    x0 = (truth + rng.normal(0.0, 0.6, n)).astype(F8)
    return matrix.astype(F8), target.astype(F8), x0.astype(F8)


def main() -> None:
    lsq_mod = importlib.import_module("scipy.optimize._lsq.least_squares")

    rng = np.random.default_rng(SEED)
    cases = []

    for n in DIMS:
        m = 2 * n + 3
        for loss in LOSSES:
            for sub_index, (f_scale, outlier_scale) in enumerate(CONDITIONING):
                # Linear loss ignores f_scale/outliers; keep one clean and one
                # noisier draw so each (n, loss) still has two distinct cases.
                eff_outlier = 0.0 if loss == "linear" else outlier_scale
                matrix, target, x0 = build_case(rng, n, m, eff_outlier)

                def fun(x):
                    return residual_values(matrix, target, x)

                result = lsq_mod.least_squares(
                    fun, x0, gtol=1e-10, loss=loss, f_scale=f_scale
                )

                cases.append(
                    {
                        "name": f"n{n}_{loss}_fs{f_scale:g}_{sub_index}",
                        "loss": loss,
                        "f_scale": f64_bits(f_scale),
                        "n": int(n),
                        "m": int(m),
                        "x0": bits_flat(x0),
                        "matrix": bits_flat(matrix),
                        "target": bits_flat(target),
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

    payload = {
        "schema": "trust-region-least-squares-trf-general-v1",
        "reference": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "method": "least_squares(fun, x0, gtol=1e-10, loss=..., f_scale=...)",
            "jac": "2-point",
            "dims": DIMS,
            "losses": LOSSES,
            "x_scale": "1.0",
            "tr_solver": "exact",
        },
        "seed": SEED,
        "cases": cases,
    }

    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "trf_general.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
