#!/usr/bin/env python3
"""Generate the residual-distribution (normality) golden values that
``crates/sidereon-core/src/quality/normality.rs`` checks against.

The Rust unit tests pin two fixed residual vectors (``V1``, ``V2``) and compare
``skewness``/``kurtosis``/``jarque_bera``/``shapiro_wilk`` (plus the mean and
biased variance) to ``scipy.stats`` to a tight tolerance (the folds are
left-to-right ``f64`` while NumPy reduces pairwise). This script recomputes those
references from the pinned ``scipy`` in ``requirements.txt`` and prints them in
the Rust literal form used by the test module, so the goldens are never
hand-edited.

Run inside the pinned environment in ``requirements.txt``.
"""

from __future__ import annotations

import numpy as np
import scipy
from scipy import stats

V1 = [
    0.12, -0.34, 0.05, 0.88, -1.21, 0.42, -0.07, 0.63, -0.55, 0.19, 0.27,
    -0.91, 1.04, -0.16, 0.33,
]
V2 = [1.0, -2.0, 0.5, 3.2, -1.1, 0.0, 2.3, -0.7, 4.5, -3.1, 0.9, -1.8]


def rust_lit(value: float) -> str:
    """Round-trippable f64 in the test's Rust literal style (digits grouped by
    three from the decimal point, scientific notation preserved)."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        mant, exp = text.lower().split("e")
        exp = str(int(exp))
        suffix = f"e{exp}" if int(exp) >= 0 else f"e-{abs(int(exp))}"
    else:
        mant, suffix = text, ""
    neg = mant.startswith("-")
    if neg:
        mant = mant[1:]
    if "." in mant:
        int_part, frac = mant.split(".")
    else:
        int_part, frac = mant, ""

    def group(digits: str, from_left: bool) -> str:
        if from_left:
            return "_".join(digits[i:i + 3] for i in range(0, len(digits), 3))
        rev = digits[::-1]
        grouped = "_".join(rev[i:i + 3] for i in range(0, len(rev), 3))
        return grouped[::-1]

    out = group(int_part, from_left=False)
    if frac:
        out += "." + group(frac, from_left=True)
    if neg:
        out = "-" + out
    return out + suffix


def main() -> None:
    print(f"# scipy {scipy.__version__} / numpy {np.__version__}")
    a = np.asarray(V1, dtype=np.float64)
    b = np.asarray(V2, dtype=np.float64)

    rows = [
        ("skew V1 bias=True", stats.skew(a, bias=True)),
        ("skew V1 bias=False", stats.skew(a, bias=False)),
        ("skew V2 bias=True", stats.skew(b, bias=True)),
        ("skew V2 bias=False", stats.skew(b, bias=False)),
        ("kurt V1 fisher=True bias=True", stats.kurtosis(a, fisher=True, bias=True)),
        ("kurt V1 fisher=False bias=True", stats.kurtosis(a, fisher=False, bias=True)),
        ("kurt V1 fisher=True bias=False", stats.kurtosis(a, fisher=True, bias=False)),
        ("kurt V2 fisher=True bias=True", stats.kurtosis(b, fisher=True, bias=True)),
        ("kurt V2 fisher=False bias=True", stats.kurtosis(b, fisher=False, bias=True)),
        ("kurt V2 fisher=True bias=False", stats.kurtosis(b, fisher=True, bias=False)),
        ("mean V1", float(np.mean(a))),
        ("var V1 (biased)", float(np.var(a))),
    ]
    jb1 = stats.jarque_bera(a)
    jb2 = stats.jarque_bera(b)
    rows += [
        ("jarque_bera V1 statistic", float(jb1.statistic)),
        ("jarque_bera V1 pvalue", float(jb1.pvalue)),
        ("jarque_bera V2 statistic", float(jb2.statistic)),
        ("jarque_bera V2 pvalue", float(jb2.pvalue)),
    ]
    sw1 = stats.shapiro(a)
    sw2 = stats.shapiro(b)
    rows += [
        ("shapiro V1 W", float(sw1.statistic)),
        ("shapiro V1 pvalue", float(sw1.pvalue)),
        ("shapiro V2 W", float(sw2.statistic)),
        ("shapiro V2 pvalue", float(sw2.pvalue)),
    ]

    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {rust_lit(value)}")


if __name__ == "__main__":
    main()
