#!/usr/bin/env python3
"""Generate the shared benchmark problem set for the native-Rust-vs-SciPy timing
comparison.

This is NOT a bit-exact fixture: it is the common input both the criterion
benchmark (`benches/solve.rs`) and the SciPy timing harness (`bench_scipy.py`)
load, so they solve the *same* problems. Floating-point payloads are stored as
f64 hex bits so the inputs are identical down to the last bit on both sides.

The residual model is the same sequential-accumulation + curvature model used by
`generate_trf_general.py`, reproduced scalar-for-scalar in `benches/solve.rs`.

Run inside the pinned environment in `requirements.txt`:

    .venv/bin/python generate_bench_problems.py
"""

from __future__ import annotations

import json
import platform
import struct
from pathlib import Path

import numpy as np
import scipy

F8 = np.dtype("f8")
SEED = 20260628

# (name, n, m, loss, f_scale). The small/repeated regime (small n, modest m) is
# where eliminating SciPy's per-call Python orchestration should dominate. The
# `large_*` cases are bigger single solves where both sides are SVD-bound.
PROBLEMS = [
    ("small_n3_linear", 3, 9, "linear", 1.0),
    ("small_n4_linear", 4, 11, "linear", 1.0),
    ("small_n5_linear", 5, 13, "linear", 1.0),
    ("small_n3_soft_l1", 3, 9, "soft_l1", 1.0),
    ("small_n4_huber", 4, 11, "huber", 1.0),
    ("large_n20_m400_linear", 20, 400, "linear", 1.0),
    ("large_n40_m120_linear", 40, 120, "linear", 1.0),
]


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_flat(values) -> list[str]:
    return [f64_bits(v) for v in np.asarray(values, dtype=F8).ravel(order="C")]


def residual_values(matrix: np.ndarray, target: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Idiomatic vectorized residual (what a real SciPy user would write):
    a linear part plus a mild elementwise nonlinearity. The same model is
    implemented as a native loop in `benches/solve.rs`; the benchmark is a
    timing comparison, not a bit-exact one.
    """
    m, n = matrix.shape
    idx = np.arange(m) % n
    return matrix @ x - target + np.sin(np.float64(0.25) * x[idx])


def build_case(rng: np.random.Generator, n: int, m: int):
    matrix = rng.normal(0.0, 0.8, (m, n)).astype(F8)
    truth = rng.normal(0.0, 3.0, n).astype(F8)
    # Choose target so the residual at `truth` is pure small noise.
    base = residual_values(matrix, target=np.zeros(m, dtype=F8), x=truth)
    target = base + rng.normal(0.0, 1e-5, m).astype(F8)
    x0 = (truth + rng.normal(0.0, 0.6, n)).astype(F8)
    return matrix.astype(F8), target.astype(F8), x0.astype(F8)


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases = []
    for name, n, m, loss, f_scale in PROBLEMS:
        matrix, target, x0 = build_case(rng, n, m)
        cases.append(
            {
                "name": name,
                "loss": loss,
                "f_scale": f64_bits(f_scale),
                "n": int(n),
                "m": int(m),
                "x0": bits_flat(x0),
                "matrix": bits_flat(matrix),
                "target": bits_flat(target),
            }
        )

    payload = {
        "schema": "trust-region-least-squares-bench-problems-v1",
        "reference": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "seed": SEED,
        "cases": cases,
    }

    out = Path(__file__).resolve().parents[1] / "benches" / "bench_problems.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
