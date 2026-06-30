#!/usr/bin/env python3
"""Generate the Huber/IRLS reweighting golden.

Hand-rolled explicit outer-loop IRLS (NOT scipy loss='huber') on a small
deterministic linear least-squares case A x = y with one injected large
outlier so the Huber psi engages. The recipe mirrors the Rust crate exactly:

  - base weights = identity (1.0), so effective weight = Huber multiplier
  - per OUTER iteration, recompute the UNWEIGHTED post-fit residual r = y - A x
  - MAD scale s = max(scale_floor, 1.4826 * median(|r - median(r)|))
  - per-row Huber weight w_i = 1 if |r_i/s| <= k else k / |r_i/s|
  - rebuild the weighted normal equations and re-solve, warm-started at prev x
  - stop when ||dx||_2 < outer_tol or max_outer reached

Iteration 0 is the unweighted (identity) warm-start solve. Each outer iteration
records the residual vector, the Huber weight vector, the sqrt-weighted
residual, and the converged x for that iteration.

Output is float.hex() lossless. Pinned: scipy 1.17.1 / numpy 2.4.6.
"""
import json
import numpy as np
import scipy

K = 1.345
MAD_CONST = 1.4826
SCALE_FLOOR = 1.0
MAX_OUTER = 5
OUTER_TOL = 1e-4


def hx(x):
    return float(x).hex()


def median(vals):
    # numpy median on a copy; matches a total-order sort + central average.
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def mad_scale(r):
    med = median(r)
    abs_dev = [abs(v - med) for v in r]
    mad = median(abs_dev)
    scaled = MAD_CONST * mad
    return scaled if scaled > SCALE_FLOOR else SCALE_FLOOR


def huber_w(u):
    a = abs(u)
    return 1.0 if a <= K else K / a


def solve_weighted(A, y, w):
    # Normal equations with diagonal weights W: (A^T W A) x = A^T W y.
    W = np.diag(w)
    lhs = A.T @ W @ A
    rhs = A.T @ W @ y
    return np.linalg.solve(lhs, rhs)


def main():
    # Deterministic over-determined linear system: a line a + b*t plus a small
    # quadratic column, 9 rows, true x = [2, -0.5, 0.1]. One injected outlier.
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    A = np.column_stack([np.ones_like(t), t, t * t]).astype(np.float64)
    x_true = np.array([2.0, -0.5, 0.1], dtype=np.float64)
    # Small deterministic perturbations so the fit is not exact.
    noise = np.array(
        [0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.04, 0.03],
        dtype=np.float64,
    )
    y = A @ x_true + noise
    # Inject one large outlier at row 5.
    y[5] += 12.0

    n = A.shape[0]
    iters = []

    # Iteration 0: unweighted warm start.
    w0 = np.ones(n, dtype=np.float64)
    x = solve_weighted(A, y, w0)
    r = y - A @ x
    iters.append(
        {
            "outer": 0,
            "scale_m": hx(SCALE_FLOOR),  # unused at warm start; record floor
            "residual": [hx(v) for v in r],
            "huber_weight": [hx(1.0)] * n,
            "sqrt_weighted_residual": [hx(v) for v in r],
            "x": [hx(v) for v in x],
        }
    )

    for it in range(1, MAX_OUTER + 1):
        r = y - A @ x
        s = mad_scale(list(r))
        w = np.array([huber_w(ri / s) for ri in r], dtype=np.float64)
        sqrt_w = np.sqrt(w)
        x_prev = x.copy()
        x = solve_weighted(A, y, w)
        swr = sqrt_w * (y - A @ x)
        iters.append(
            {
                "outer": it,
                "scale_m": hx(s),
                "residual": [hx(v) for v in r],
                "huber_weight": [hx(v) for v in w],
                "sqrt_weighted_residual": [hx(v) for v in swr],
                "x": [hx(v) for v in x],
            }
        )
        dpos = float(np.linalg.norm(x - x_prev))
        if dpos < OUTER_TOL:
            break

    doc = {
        "_comment": "Hand-rolled explicit outer-loop Huber IRLS golden. NOT scipy loss=huber.",
        "metadata": {
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "huber_k": hx(K),
            "mad_normal_const": hx(MAD_CONST),
            "scale_floor_m": hx(SCALE_FLOOR),
            "max_outer": MAX_OUTER,
            "outer_tol": hx(OUTER_TOL),
        },
        "problem": {
            "A": [[hx(v) for v in row] for row in A],
            "y": [hx(v) for v in y],
        },
        "iterations": iters,
        "final_x": [hx(v) for v in x],
    }
    out = __file__.rsplit("/", 1)[0] + "/huber_irls_trace.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote", out, "outer iters:", len(iters) - 1)


if __name__ == "__main__":
    main()
