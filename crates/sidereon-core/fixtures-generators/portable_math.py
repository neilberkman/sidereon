"""Binary64 operations rounded from a high-precision mpmath calculation.

Every helper returns a Python ``float`` immediately.  This is deliberate: the
broadcast reference recipe is a sequence of binary64 operations, not one
unrounded arbitrary-precision expression.
"""

from __future__ import annotations

import mpmath as mp


MPMATH_PRECISION_DIGITS = 200
mp.mp.dps = MPMATH_PRECISION_DIGITS


def _mp(value: float) -> mp.mpf:
    return mp.mpf(value)


def _f64(value: mp.mpf) -> float:
    return float(value)


def add(a: float, b: float) -> float:
    return _f64(_mp(a) + _mp(b))


def sub(a: float, b: float) -> float:
    return _f64(_mp(a) - _mp(b))


def mul(a: float, b: float) -> float:
    return _f64(_mp(a) * _mp(b))


def div(a: float, b: float) -> float:
    return _f64(_mp(a) / _mp(b))


def sqrt(a: float) -> float:
    return _f64(mp.sqrt(_mp(a)))


def sin(a: float) -> float:
    return _f64(mp.sin(_mp(a)))


def cos(a: float) -> float:
    return _f64(mp.cos(_mp(a)))


def tan(a: float) -> float:
    return _f64(mp.tan(_mp(a)))


def asin(a: float) -> float:
    return _f64(mp.asin(_mp(a)))


def acos(a: float) -> float:
    return _f64(mp.acos(_mp(a)))


def atan(a: float) -> float:
    return _f64(mp.atan(_mp(a)))


def atan2(y: float, x: float) -> float:
    return _f64(mp.atan2(_mp(y), _mp(x)))


def exp(a: float) -> float:
    return _f64(mp.exp(_mp(a)))


def ln(a: float) -> float:
    return _f64(mp.log(_mp(a)))


def log10(a: float) -> float:
    return _f64(mp.log10(_mp(a)))


def pow(a: float, b: float) -> float:
    return _f64(mp.power(_mp(a), _mp(b)))
