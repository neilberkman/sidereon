"""python-sgp4's pure-Python SGP4 in the operation order of Vallado's C++.

`sgp4.propagation` (python-sgp4 2.22) transcribes Vallado's C++ of 2020-07-13
line for line, except in three places where it evaluates the same expression
with different binary64 operations:

1. `x % twopi` for every angle reduction. Python's `%` returns a result with
   the sign of the divisor, so for a negative angle it returns
   `fmod(x, twopi) + twopi`: an angle 2 pi larger, rounded once more. The C++
   (and sidereon-core's port) calls `fmod`, which is exact and keeps the sign
   of `x`. The two agree for a non-negative angle.
2. The position scaling at the end of `sgp4`: Python forms
   `mrt * radiusearthkm` once and multiplies each unit-vector component by it;
   the C++ multiplies `(mrt * ux) * radiusearthkm` per component.
3. The Lyddane branch of `dpper`: Python sums
   `mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep` in one
   expression; the C++ forms `xls = mp + argpp + cosip * nodep`,
   `dls = pl + pgh - pinc * nodep * sinip` and `xls + dls`.

`vallado_order_module()` returns a copy of `sgp4.propagation` with those three
rewritten as the C++ writes them, and `sin`, `cos`, `atan2`, `pow` and `fmod`
taken from the module's globals, so a caller can supply a libm. With the Rust
`libm` crate's functions (`rust_libm_port.py`) it computes, operation for
operation, what sidereon-core's kernel computes; with the platform's it
computes what python-sgp4's compiled extension computes, except where the
compiler fused a `sin` and `cos` of one argument into one `sincos` call.
"""

from __future__ import annotations

import inspect
import math
import re
import types

import sgp4.propagation as _pure

_SCALING = """         _mr = mrt * satrec.radiusearthkm
         r = (_mr * ux, _mr * uy, _mr * uz)"""
_SCALING_C = (
    "         r = ((mrt * ux) * satrec.radiusearthkm, (mrt * uy) * satrec.radiusearthkm,"
    " (mrt * uz) * satrec.radiusearthkm)"
)
_LYDDANE = "           xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep\n"
_LYDDANE_C = (
    "           xls = mp + argpp + cosip * nodep\n"
    "           dls = pl + pgh - pinc * nodep * sinip\n"
    "           xls = xls + dls\n"
)


def vallado_order_source() -> str:
    src = inspect.getsource(_pure)
    # nodep and nodem already emulate fmod: x % twopi if x >= 0 else -(-x % twopi)
    src = re.sub(r"(\w+) % twopi if \1 >= 0\.0 else -\(-\1 % twopi\)", r"fmod(\1, twopi)", src)
    src = re.sub(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*%\s*twopi", r"fmod((\1), twopi)", src)
    src = re.sub(r"\b(\w+)\s*%\s*twopi", r"fmod(\1, twopi)", src)
    if "% twopi" in src:
        raise RuntimeError("an angle reduction was not rewritten")
    if src.count(_SCALING) != 1 or src.count(_LYDDANE) != 1:
        raise RuntimeError("unexpected sgp4.propagation source")
    src = src.replace(_SCALING, _SCALING_C).replace(_LYDDANE, _LYDDANE_C)
    return src


def vallado_order_module(
    name: str, sin, cos, atan2, pow, fmod=math.fmod, transform=None, extra=None
) -> types.ModuleType:
    """A fresh copy of the rewritten model with the given functions.

    `transform`, if given, rewrites the source once more before it is
    compiled; `extra` adds names to the module's globals.
    """
    src = vallado_order_source()
    if transform is not None:
        src = transform(src)
    module = types.ModuleType(name)
    module.__file__ = name
    module.__dict__.update(extra or {})
    exec(compile(src, name, "exec"), module.__dict__)
    module.sin, module.cos, module.atan2, module.pow, module.fmod = sin, cos, atan2, pow, fmod
    return module
