#!/usr/bin/env python3
"""Generate SciPy thin-SVD fixtures for the host LAPACK bridge."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import scipy
from scipy import linalg


F8 = np.dtype("f8")
SEED = 20260611


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_array(values) -> list[str]:
    return [f64_bits(x) for x in np.asarray(values, dtype=F8).ravel(order="C")]


def fixture(name: str, a: np.ndarray) -> dict[str, object]:
    a = np.asarray(a, dtype=F8, order="C")
    u, s, vt = linalg.svd(a, full_matrices=False, lapack_driver="gesdd")
    return {
        "name": name,
        "m": int(a.shape[0]),
        "n": int(a.shape[1]),
        "a_bits": bits_array(a),
        "u_bits": bits_array(u),
        "s_bits": bits_array(s),
        "vt_bits": bits_array(vt),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases = []

    for i in range(20):
        m = 5 + (i % 6)
        cases.append(fixture(f"random_{i:02d}_{m}x3", rng.standard_normal((m, 3))))

    cases.append(fixture("zero_5x3", np.zeros((5, 3), dtype=F8)))

    col = np.linspace(-2.0, 2.0, 7, dtype=F8)
    cases.append(fixture("rank_one_7x3", col[:, None] @ np.array([[1.5, -2.0, 0.25]], dtype=F8)))

    base = np.arange(1.0, 25.0, dtype=F8).reshape(8, 3)
    base[:, 2] = base[:, 0] - 2.0 * base[:, 1]
    cases.append(fixture("rank_two_8x3", base))

    scaled = rng.standard_normal((10, 3)) * np.float_power(2.0, -40)
    scaled[:, 1] = scaled[:, 0]
    cases.append(fixture("tiny_duplicate_10x3", scaled))

    payload = {
        "schema": "trust-region-least-squares-hostlapack-svd-v1",
        "seed": SEED,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "layout": "row-major arrays encoded as f64 hex bits",
        "lapack_driver": "gesdd",
        "full_matrices": False,
        "cases": cases,
    }
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hostlapack_svd.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
