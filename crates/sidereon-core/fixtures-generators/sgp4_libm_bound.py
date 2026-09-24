"""First-order bound on how far two SGP4 builds can drift apart when they run
the same binary64 operations in the same order but call different libm
implementations.

The reference states come from python-sgp4's compiled extension, which calls
the platform's `sin`, `cos`, `atan2` and `pow`. sidereon-core runs the same
Vallado operations with the pure-Rust `libm` crate (a port of FreeBSD msun /
musl). `sqrt`, `fabs`, `fmod` and the four arithmetic operations are
correctly rounded or exact in both, so they give identical results on
identical inputs. Each `sin`, `cos`, `atan2` and `pow` result is documented
as within 1 ulp of the true value in both libraries ("nearly rounded" in
msun; under 1 ulp in the platform library), so the two results for the same
argument differ by less than 2 ulp.

`Tracked` carries a value together with a bound on the difference between
the two builds' values of the same quantity. The difference is kept as an
affine form, a sum of coefficients times independent symbols in [-1, 1]:

* one symbol for every libm call, scaled by 2 ulp of the largest magnitude
  either build's result can have (see `_libm_symbol`);
* one symbol for every operation whose inputs already differ between the
  builds, scaled by 1 ulp of its result: the two builds round two different
  exact results, and each rounding moves its result by at most half an ulp;

plus a non-negative remainder for the second-order terms of products,
quotients and the functions, bounded from their derivatives over the whole
uncertainty interval. The bound is the sum of the coefficients' magnitudes
and the remainder.

Keeping the symbols as signed coefficients lets the bound see the
cancellation that SGP4's own algebra performs (for example the Kepler
iteration, which contracts an error in its starting value); a plain
interval or absolute-value bound would not.

The model is python-sgp4's pure-Python `sgp4.propagation` rewritten into the
C++'s operation order (`sgp4_vallado_order.py`): its angle reductions call C
`fmod`, as the C++ and sidereon-core do, rather than Python's `%`, which for a
negative angle returns the angle 2 pi larger and rounds once more; and two
expressions (the final position scaling and the Lyddane longitude sum) are
evaluated in the C++'s order. It runs with every input wrapped in `Tracked`
and with `sin`, `cos`, `atan2`, `sqrt`, `fabs`, `pow` and `fmod` replaced.
Every comparison whose outcome could differ between the builds, and every
`fmod` whose argument's uncertainty spans a multiple of 2 pi, is recorded in
`BRANCH_FLAGS`; the generator gives no bound to a state with a flag.
"""

from __future__ import annotations

import math
from itertools import count

LIBM_PAIR_ULPS = 2.0
_symbols = count()
BRANCH_FLAGS: list[str] = []


def _ulp(x: float) -> float:
    return math.ulp(abs(x))


def _merge(a: dict, ka: float, b: dict, kb: float) -> dict:
    out = {s: c * ka for s, c in a.items()} if ka != 0.0 else {}
    if kb != 0.0:
        for s, c in b.items():
            out[s] = out.get(s, 0.0) + c * kb
    return out


class Tracked:
    __slots__ = ("v", "lin", "rem")

    def __init__(self, v, lin=None, rem=0.0):
        self.v = float(v)
        self.lin = lin if lin is not None else {}
        self.rem = rem

    # -- bookkeeping -----------------------------------------------------
    def radius(self) -> float:
        return sum(abs(c) for c in self.lin.values()) + self.rem

    def diverged(self) -> bool:
        return bool(self.lin) or self.rem > 0.0

    def __float__(self):
        return self.v

    def __format__(self, spec):
        return format(self.v, spec)

    def __repr__(self):
        return f"Tracked({self.v!r}, radius={self.radius():.3e})"

    # -- arithmetic ------------------------------------------------------
    @staticmethod
    def _finish(z, lin, rem, inputs_diverged, radius_hint):
        if inputs_diverged:
            lin[next(_symbols)] = math.ulp(abs(z) + radius_hint)
        return Tracked(z, lin, rem)

    def _add(self, o, sign):
        o = lift(o)
        z = self.v + o.v if sign > 0 else self.v - o.v
        lin = _merge(self.lin, 1.0, o.lin, float(sign))
        rem = self.rem + o.rem
        dv = self.diverged() or o.diverged()
        return Tracked._finish(z, lin, rem, dv, self.radius() + o.radius())

    def __add__(self, o):
        return self._add(o, 1)

    def __radd__(self, o):
        return lift(o)._add(self, 1)

    def __sub__(self, o):
        return self._add(o, -1)

    def __rsub__(self, o):
        return lift(o)._add(self, -1)

    def __mul__(self, o):
        o = lift(o)
        z = self.v * o.v
        bx, by = self.radius(), o.radius()
        lin = _merge(self.lin, o.v, o.lin, self.v)
        rem = abs(o.v) * self.rem + abs(self.v) * o.rem + bx * by
        return Tracked._finish(z, lin, rem, self.diverged() or o.diverged(), abs(o.v) * bx + abs(self.v) * by + bx * by)

    def __rmul__(self, o):
        return lift(o).__mul__(self)

    def __truediv__(self, o):
        o = lift(o)
        x, y = self.v, o.v
        z = x / y
        bx, by = self.radius(), o.radius()
        if by >= 0.5 * abs(y):
            raise ArithmeticError("divisor uncertainty too large")
        lin = _merge(self.lin, 1.0 / y, o.lin, -x / (y * y))
        second = (abs(x) * by * by / abs(y) ** 3 + bx * by / (y * y)) / (1.0 - by / abs(y))
        rem = self.rem / abs(y) + abs(x) * o.rem / (y * y) + second
        return Tracked._finish(z, lin, rem, self.diverged() or o.diverged(), bx / abs(y) + abs(x) * by / (y * y) + second)

    def __rtruediv__(self, o):
        return lift(o).__truediv__(self)

    def __neg__(self):
        return Tracked(-self.v, {s: -c for s, c in self.lin.items()}, self.rem)

    def __pos__(self):
        return self

    def __abs__(self):
        return t_fabs(self)

    def __mod__(self, m):
        m = float(m)
        b = self.radius()
        z = self.v % m
        if self.diverged() and math.floor((self.v - b) / m) != math.floor((self.v + b) / m):
            BRANCH_FLAGS.append(f"mod wrap v={self.v!r} radius={b:.3e}")
        # fmod is exact; Python's % adds m to a negative remainder, which rounds.
        lin = dict(self.lin)
        if self.v < 0.0 and self.diverged():
            lin[next(_symbols)] = math.ulp(abs(z) + b)
        return Tracked(z, lin, self.rem)

    def __pow__(self, e):
        return t_pow(self, e)

    # -- comparisons -----------------------------------------------------
    def _cmp(self, o, op):
        o = lift(o)
        if (self.diverged() or o.diverged()) and abs(self.v - o.v) <= self.radius() + o.radius():
            BRANCH_FLAGS.append(f"comparison {self.v!r} vs {o.v!r} radius {self.radius() + o.radius():.3e}")
        return op(self.v, o.v)

    def __lt__(self, o):
        return self._cmp(o, lambda a, b: a < b)

    def __le__(self, o):
        return self._cmp(o, lambda a, b: a <= b)

    def __gt__(self, o):
        return self._cmp(o, lambda a, b: a > b)

    def __ge__(self, o):
        return self._cmp(o, lambda a, b: a >= b)

    def __eq__(self, o):
        return self._cmp(o, lambda a, b: a == b)

    def __ne__(self, o):
        return self._cmp(o, lambda a, b: a != b)

    __hash__ = None


def lift(x) -> Tracked:
    return x if isinstance(x, Tracked) else Tracked(float(x))


def _libm_symbol(z: float) -> dict:
    # Each build's result is within 1 ulp of the true value; the other
    # build's result may lie in the next binade up, where an ulp is twice
    # this one's, so the pair's difference is bounded with the ulp of the
    # largest magnitude either can have.
    return {next(_symbols): LIBM_PAIR_ULPS * _ulp(abs(z) + LIBM_PAIR_ULPS * _ulp(z))}


def t_fmod(x, m):
    """C `fmod` against a constant divisor: exact in both builds."""
    x = lift(x)
    m = float(m)
    b = x.radius()
    z = math.fmod(x.v, m)
    if x.diverged() and math.trunc((x.v - b) / m) != math.trunc((x.v + b) / m):
        BRANCH_FLAGS.append(f"fmod wrap v={x.v!r} radius={b:.3e}")
    return Tracked(z, dict(x.lin), x.rem)


def t_sin(x):
    x = lift(x)
    z = math.sin(x.v)
    b = x.radius()
    lin = _merge(x.lin, math.cos(x.v), _libm_symbol(z), 1.0)
    return Tracked(z, lin, abs(math.cos(x.v)) * x.rem + 0.5 * b * b)


def t_cos(x):
    x = lift(x)
    z = math.cos(x.v)
    b = x.radius()
    lin = _merge(x.lin, -math.sin(x.v), _libm_symbol(z), 1.0)
    return Tracked(z, lin, abs(math.sin(x.v)) * x.rem + 0.5 * b * b)


def t_atan2(y, x):
    y, x = lift(y), lift(x)
    z = math.atan2(y.v, x.v)
    by, bx = y.radius(), x.radius()
    r2 = x.v * x.v + y.v * y.v
    rmin = math.sqrt(r2) - bx - by
    if rmin <= 0.0:
        raise ArithmeticError("atan2 argument uncertainty reaches the origin")
    if x.v < 0.0 and abs(y.v) <= by and (x.diverged() or y.diverged()):
        BRANCH_FLAGS.append("atan2 branch cut")
    lin = _merge(y.lin, x.v / r2, x.lin, -y.v / r2)
    lin = _merge(lin, 1.0, _libm_symbol(z), 1.0)
    second = 0.5 * (bx + by) ** 2 / (rmin * rmin)
    rem = abs(x.v) / r2 * y.rem + abs(y.v) / r2 * x.rem + second
    return Tracked(z, lin, rem)


def t_sqrt(x):
    x = lift(x)
    z = math.sqrt(x.v)
    b = x.radius()
    if b >= 0.5 * x.v:
        raise ArithmeticError("sqrt argument uncertainty too large")
    lin = _merge(x.lin, 0.5 / z, {}, 0.0)
    second = b * b / (8.0 * (x.v - b) ** 1.5)
    return Tracked._finish(z, lin, x.rem * 0.5 / z + second, x.diverged(), b * 0.5 / z + second)


def t_fabs(x):
    x = lift(x)
    b = x.radius()
    if abs(x.v) > b:
        s = 1.0 if x.v > 0.0 else -1.0
        return Tracked(abs(x.v), {k: s * c for k, c in x.lin.items()}, x.rem)
    return Tracked(abs(x.v), {}, b)


def t_pow(x, e):
    if isinstance(e, Tracked):
        if e.diverged():
            raise ArithmeticError("pow exponent carries a difference")
        e = e.v
    x = lift(x)
    c = float(e)
    z = math.pow(x.v, c)
    b = x.radius()
    if b >= 0.5 * x.v:
        raise ArithmeticError("pow base uncertainty too large")
    d1 = c * math.pow(x.v, c - 1.0)
    lo, hi = x.v - b, x.v + b
    d2max = abs(c * (c - 1.0)) * max(math.pow(lo, c - 2.0), math.pow(hi, c - 2.0))
    lin = _merge(x.lin, d1, _libm_symbol(z), 1.0)
    return Tracked(z, lin, abs(d1) * x.rem + 0.5 * d2max * b * b)


def symbol_mark() -> int:
    """The id the next symbol will get; every older symbol has a smaller id."""
    global _symbols
    mark = next(_symbols)
    _symbols = count(mark + 1)
    return mark


def condense(value, mark: int):
    """Enclose every symbol at or after `mark` in one new symbol.

    The resonance integrator of the deep-space model takes one fixed 720-minute
    step per call of the loop, so a propagation far from epoch makes thousands
    of steps and would otherwise carry thousands of symbols in `xli` and
    `xni`. After each step the symbols created during the integration are
    replaced by one fresh symbol whose radius is the sum of their magnitudes.
    The enclosure is sound; it only gives up the correlation between those
    step errors, so the bound can only grow.
    """
    value = lift(value)
    kept = {}
    radius = 0.0
    for s, c in value.lin.items():
        if s < mark:
            kept[s] = c
        else:
            radius += abs(c)
    if radius > 0.0:
        kept[next(_symbols)] = radius
    return Tracked(value.v, kept, value.rem)


_STEP = "                 atime = atime + delt;\n"
_FT = "     ft    = 0.0;\n"


def instrument_source(src: str) -> str:
    """Enclose the resonance integrator's per-step symbols (see `condense`)."""
    if src.count(_STEP) != 1 or src.count(_FT) != 1:
        raise RuntimeError("unexpected _dspace source")
    indent = " " * 17
    src = src.replace(
        _STEP, _STEP + f"{indent}xli = _condense(xli, _mark)\n{indent}xni = _condense(xni, _mark)\n"
    )
    return src.replace(_FT, _FT + "     _mark = _symbol_mark()\n")


def tracked_module(name: str = "sgp4_tracked"):
    """python-sgp4's model in the C++ operation order (`sgp4_vallado_order`),
    running on `Tracked` values."""
    from sgp4_vallado_order import vallado_order_module

    module = vallado_order_module(
        name,
        t_sin,
        t_cos,
        t_atan2,
        t_pow,
        fmod=t_fmod,
        transform=instrument_source,
        extra={"_condense": condense, "_symbol_mark": symbol_mark},
    )
    module.sqrt = t_sqrt
    module.fabs = t_fabs
    return module
