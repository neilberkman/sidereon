#!/usr/bin/env python3
"""Time ``scipy.optimize.least_squares`` (TRF, 2-point Jacobian) on the shared
benchmark problem set, to compare against the native-Rust criterion benchmark.

This solves exactly the same problems as ``benches/solve.rs`` (loaded from
``benches/bench_problems.json``), with the same options the crate reproduces:
``method='trf'``, ``jac='2-point'``, ``gtol=1e-10``, matching ``loss`` and
``f_scale``. It reports per-solve wall-clock time and solves/sec so the README
can quote a fair native-Rust-vs-SciPy number.

Run inside the pinned environment in ``requirements.txt``:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python bench_scipy.py
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

F8 = np.dtype("f8")


def bits_to_f64(s: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(s, 16)))[0]


def vec(values) -> np.ndarray:
    return np.array([bits_to_f64(v) for v in values], dtype=F8)


def residual_values(matrix: np.ndarray, target: np.ndarray, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Idiomatic vectorized residual, matching `benches/solve.rs`."""
    return matrix @ x - target + np.sin(np.float64(0.25) * x[idx])


def bench_case(case: dict) -> dict:
    n = case["n"]
    m = case["m"]
    matrix = vec(case["matrix"]).reshape(m, n)
    target = vec(case["target"])
    x0 = vec(case["x0"])
    loss = case["loss"]
    f_scale = bits_to_f64(case["f_scale"])
    idx = np.arange(m) % n

    def fun(x):
        return residual_values(matrix, target, x, idx)

    # Warm up (also confirms it converges).
    r = least_squares(fun, x0, method="trf", jac="2-point", gtol=1e-10, loss=loss, f_scale=f_scale)

    # Pick repetitions so each measured batch runs >= ~0.2 s, then take the best
    # of several batches (best = least contended).
    def timed(reps: int) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            least_squares(fun, x0, method="trf", jac="2-point", gtol=1e-10, loss=loss, f_scale=f_scale)
        return time.perf_counter() - t0

    reps = 1
    while timed(reps) < 0.05:
        reps *= 2
    batches = [timed(reps) / reps for _ in range(7)]
    per_solve = min(batches)

    return {
        "name": case["name"],
        "n": n,
        "m": m,
        "loss": loss,
        "status": int(r.status),
        "nfev": int(r.nfev),
        "per_solve_s": per_solve,
        "solves_per_s": 1.0 / per_solve,
    }


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "benches" / "bench_problems.json"
    doc = json.loads(path.read_text())
    print(f"scipy {scipy.__version__}, numpy {np.__version__}")
    print(f"{'case':28} {'n':>3} {'m':>5} {'loss':>8} {'nfev':>5} "
          f"{'per_solve':>14} {'solves/s':>12}")
    for case in doc["cases"]:
        r = bench_case(case)
        per = r["per_solve_s"]
        unit = f"{per * 1e6:.2f} us" if per < 1e-3 else f"{per * 1e3:.3f} ms"
        print(f"{r['name']:28} {r['n']:>3} {r['m']:>5} {r['loss']:>8} "
              f"{r['nfev']:>5} {unit:>14} {r['solves_per_s']:>12.1f}")


if __name__ == "__main__":
    main()
