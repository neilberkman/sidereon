#!/usr/bin/env python3
"""Generate SciPy trajectory fixtures for small dense NLLS problems under each
robust loss function.

For every non-linear SciPy loss (``soft_l1``, ``huber``, ``cauchy``, ``arctan``)
this records, across a range of problem conditioning and ``f_scale`` values:

- the problem (``matrix``/``target``/``x0``) and the converged ``result``,
  enabling the host-LAPACK full-solve replay, and
- every robust-loss application: each ``scale_for_robust_loss_function`` call
  (the ``f``/``J`` inputs, the rho it was handed, the trf cost
  ``0.5 * sum(rho[0])``, and the reweighted outputs) and each ``cost_only``
  ``loss_function`` call, so the loss math can be replayed bit-for-bit without a
  LAPACK backend.

The residual, two-point Jacobian, and SVD machinery is unchanged by the loss
work and is already covered by the ``trf_small_dense``/``numdiff`` fixtures, so
those trajectories are intentionally not duplicated here.

Reference: ``scipy/optimize/_lsq/least_squares.py`` (``construct_loss_function``
+ ``IMPLEMENTED_LOSSES``) and ``scipy/optimize/_lsq/common.py``
(``scale_for_robust_loss_function``). Run inside the pinned environment in
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
LOSSES = ["soft_l1", "huber", "cauchy", "arctan"]
# (f_scale, outlier_scale): outlier_scale seeds residuals on both sides of the
# z<=1 / z>1 boundary so the huber mask exercises both branches.
CONDITIONING = [
    (1.0, 0.0),
    (0.5, 5.0),
    (2.0, 8.0),
]


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


def build_case(rng: np.random.Generator, case_index: int, outlier_scale: float):
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
    if outlier_scale > 0.0:
        # Inject heavy-tailed perturbations so robust reweighting actually bites.
        outliers = rng.standard_t(2.0, m).astype(F8) * outlier_scale
        target += outliers
    x0 = truth + rng.normal(0.0, 0.75, 3)
    return matrix, target.astype(F8), x0.astype(F8)


def main() -> None:
    lsq_mod = importlib.import_module("scipy.optimize._lsq.least_squares")
    trf_mod = importlib.import_module("scipy.optimize._lsq.trf")

    original_construct = lsq_mod.construct_loss_function
    original_scale = trf_mod.scale_for_robust_loss_function

    rng = np.random.default_rng(SEED)
    cases = []

    try:
        for loss in LOSSES:
            for cond_index, (f_scale, outlier_scale) in enumerate(CONDITIONING):
                for sub_index in range(2):
                    case_index = cond_index * 2 + sub_index
                    matrix, target, x0 = build_case(rng, case_index, outlier_scale)

                    cost_only_calls: list = []
                    scale_calls: list = []

                    def fun(x):
                        return residual_values(matrix, target, x)

                    def construct_wrapped(m_arg, loss_arg, f_scale_arg):
                        inner = original_construct(m_arg, loss_arg, f_scale_arg)
                        if inner is None:
                            return None

                        def wrapper(f, cost_only=False):
                            out = inner(f, cost_only=cost_only)
                            if cost_only:
                                cost_only_calls.append(
                                    {"f": bits_flat(f), "cost": f64_bits(out)}
                                )
                            return out

                        return wrapper

                    def scale_wrapped(j, f, rho):
                        # `rho` is the (3, m) array (already f_scale-rescaled)
                        # produced by the full loss_function call that trf made
                        # immediately before this; trf forms 0.5 * sum(rho[0]) as
                        # the cost right here, then reweights J and f.
                        snapshot = {
                            "j_in": bits_flat(j),
                            "f_in": bits_flat(f),
                            "rho0": bits_flat(rho[0]),
                            "rho1": bits_flat(rho[1]),
                            "rho2": bits_flat(rho[2]),
                            "cost": f64_bits(0.5 * np.sum(rho[0])),
                        }
                        j_out, f_out = original_scale(j, f, rho)
                        snapshot["j_out"] = bits_flat(j_out)
                        snapshot["f_out"] = bits_flat(f_out)
                        scale_calls.append(snapshot)
                        return j_out, f_out

                    lsq_mod.construct_loss_function = construct_wrapped
                    trf_mod.scale_for_robust_loss_function = scale_wrapped

                    result = lsq_mod.least_squares(
                        fun, x0, gtol=1e-10, loss=loss, f_scale=f_scale
                    )

                    lsq_mod.construct_loss_function = original_construct
                    trf_mod.scale_for_robust_loss_function = original_scale

                    cases.append(
                        {
                            "name": f"{loss}_fs{f_scale:g}_{case_index:02d}",
                            "loss": loss,
                            "f_scale": f64_bits(f_scale),
                            "m": int(target.size),
                            "x0": bits_flat(x0),
                            "matrix": bits_flat(matrix),
                            "target": bits_flat(target),
                            "cost_only_calls": cost_only_calls,
                            "scale_calls": scale_calls,
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
        lsq_mod.construct_loss_function = original_construct
        trf_mod.scale_for_robust_loss_function = original_scale

    payload = {
        "schema": "trust-region-least-squares-trf-loss-v1",
        "reference": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "method": "least_squares(fun, x0, gtol=1e-10, loss=..., f_scale=...)",
            "jac": "2-point",
            "losses": LOSSES,
            "x_scale": "1.0",
            "tr_solver": "exact",
        },
        "seed": SEED,
        "cases": cases,
    }

    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "trf_loss.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
