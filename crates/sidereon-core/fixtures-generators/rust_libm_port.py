"""Binary64 port of the Rust `libm` crate 0.2.16 `sin`, `cos` and `atan2`.

The `libm` crate ports musl, which ports FreeBSD msun; the functions below follow
its kernels (`k_sin`, `k_cos`, `rem_pio2` for arguments below 2^20 * pi/2, `atan`,
`atan2`) statement for statement, with every operation rounded to binary64 as
Python floats round. Square roots use `math.sqrt`, which is correctly rounded as
`libm::sqrt` is. The broadcast goldens built with this port are checked at 0 ULP
against the `libm` crate by the Rust tests that read them.
"""

import struct

def bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]
def fb(u):
    return struct.unpack('<d', struct.pack('<Q', u & 0xFFFFFFFFFFFFFFFF))[0]
def hi(x):
    return (bits(x) >> 32) & 0x7fffffff

S1 = -1.66666666666666324348e-01
S2 = 8.33333333332248946124e-03
S3 = -1.98412698298579493134e-04
S4 = 2.75573137070700676789e-06
S5 = -2.50507602534068634195e-08
S6 = 1.58969099521155010221e-10

def k_sin(x, y, iy):
    z = x * x
    w = z * z
    r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6)
    v = z * x
    if iy == 0:
        return x + v * (S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * S1)

C1 = 4.16666666666666019037e-02
C2 = -1.38888888888741095749e-03
C3 = 2.48015872894767294178e-05
C4 = -2.75573143513906633035e-07
C5 = 2.08757232129817482790e-09
C6 = -1.13596475577881948265e-11

def k_cos(x, y):
    z = x * x
    w = z * z
    r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6))
    hz = 0.5 * z
    w = 1.0 - hz
    return w + (((1.0 - w) - hz) + (z * r - x * y))

EPS = 2.2204460492503131e-16
TO_INT = 1.5 / EPS
INV_PIO2 = 6.36619772367581382433e-01
PIO2_1 = 1.57079632673412561417e+00
PIO2_1T = 6.07710050650619224932e-11
PIO2_2 = 6.07710050630396597660e-11
PIO2_2T = 2.02226624879595063154e-21
PIO2_3 = 2.02226624871116645580e-21
PIO2_3T = 8.47842766036889956997e-32

def medium(x, ix):
    tmp = x * INV_PIO2 + TO_INT
    f_n = tmp - TO_INT
    n = int(f_n)
    r = x - f_n * PIO2_1
    w = f_n * PIO2_1T
    y0 = r - w
    ey = (bits(y0) >> 52) & 0x7ff
    ex = ix >> 20
    if ex - ey > 16:
        t = r
        w = f_n * PIO2_2
        r = t - w
        w = f_n * PIO2_2T - ((t - r) - w)
        y0 = r - w
        ey = (bits(y0) >> 52) & 0x7ff
        if ex - ey > 49:
            t = r
            w = f_n * PIO2_3
            r = t - w
            w = f_n * PIO2_3T - ((t - r) - w)
            y0 = r - w
    y1 = (r - y0) - w
    return n, y0, y1

def rem_pio2(x):
    sign = bits(x) >> 63
    ix = hi(x)
    if ix <= 0x400f6a7a:
        if (ix & 0xfffff) == 0x921fb:
            return medium(x, ix)
        if ix <= 0x4002d97c:
            if sign == 0:
                z = x - PIO2_1; y0 = z - PIO2_1T; y1 = (z - y0) - PIO2_1T; return 1, y0, y1
            z = x + PIO2_1; y0 = z + PIO2_1T; y1 = (z - y0) + PIO2_1T; return -1, y0, y1
        if sign == 0:
            z = x - 2.0 * PIO2_1; y0 = z - 2.0 * PIO2_1T; y1 = (z - y0) - 2.0 * PIO2_1T; return 2, y0, y1
        z = x + 2.0 * PIO2_1; y0 = z + 2.0 * PIO2_1T; y1 = (z - y0) + 2.0 * PIO2_1T; return -2, y0, y1
    if ix <= 0x401c463b:
        if ix <= 0x4015fdbc:
            if ix == 0x4012d97c:
                return medium(x, ix)
            if sign == 0:
                z = x - 3.0 * PIO2_1; y0 = z - 3.0 * PIO2_1T; y1 = (z - y0) - 3.0 * PIO2_1T; return 3, y0, y1
            z = x + 3.0 * PIO2_1; y0 = z + 3.0 * PIO2_1T; y1 = (z - y0) + 3.0 * PIO2_1T; return -3, y0, y1
        if ix == 0x401921fb:
            return medium(x, ix)
        if sign == 0:
            z = x - 4.0 * PIO2_1; y0 = z - 4.0 * PIO2_1T; y1 = (z - y0) - 4.0 * PIO2_1T; return 4, y0, y1
        z = x + 4.0 * PIO2_1; y0 = z + 4.0 * PIO2_1T; y1 = (z - y0) + 4.0 * PIO2_1T; return -4, y0, y1
    if ix < 0x413921fb:
        return medium(x, ix)
    raise ValueError("arguments at or above 2^20 * pi/2 are not ported")

def sin(x):
    ix = hi(x)
    if ix <= 0x3fe921fb:
        if ix < 0x3e500000:
            return x
        return k_sin(x, 0.0, 0)
    n, y0, y1 = rem_pio2(x)
    n &= 3
    if n == 0: return k_sin(y0, y1, 1)
    if n == 1: return k_cos(y0, y1)
    if n == 2: return -k_sin(y0, y1, 1)
    return -k_cos(y0, y1)

def cos(x):
    ix = hi(x)
    if ix <= 0x3fe921fb:
        if ix < 0x3e46a09e:
            if int(x) == 0:
                return 1.0
        return k_cos(x, 0.0)
    n, y0, y1 = rem_pio2(x)
    n &= 3
    if n == 0: return k_cos(y0, y1)
    if n == 1: return -k_sin(y0, y1, 1)
    if n == 2: return -k_cos(y0, y1)
    return k_sin(y0, y1, 1)

ATANHI = [4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00]
ATANLO = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17]
AT = [3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01,
      -1.11111104054623557880e-01, 9.09088713343650656196e-02, -7.69187620504482999495e-02,
      6.66107313738753120669e-02, -5.83357013379057348645e-02, 4.97687799461593236017e-02,
      -3.65315727442169155270e-02, 1.62858201153657823623e-02]

def atan(x):
    ixf = bits(x) >> 32
    sign = ixf >> 31
    ix = ixf & 0x7fffffff
    if ix >= 0x44100000:
        z = ATANHI[3] + fb(0x0380000000000000)
        return -z if sign else z
    if ix < 0x3fdc0000:
        if ix < 0x3e400000:
            return x
        idx = -1
    else:
        x = abs(x)
        if ix < 0x3ff30000:
            if ix < 0x3fe60000:
                x = (2. * x - 1.) / (2. + x); idx = 0
            else:
                x = (x - 1.) / (x + 1.); idx = 1
        elif ix < 0x40038000:
            x = (x - 1.5) / (1. + 1.5 * x); idx = 2
        else:
            x = -1. / x; idx = 3
    z = x * x
    w = z * z
    s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))))
    s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))))
    if idx < 0:
        return x - x * (s1 + s2)
    z = ATANHI[idx] - (x * (s1 + s2) - ATANLO[idx] - x)
    return -z if sign else z

PI = 3.1415926535897931160E+00
PI_LO = 1.2246467991473531772E-16

def atan2(y, x):
    if x != x or y != y:
        return x + y
    ix = bits(x) >> 32
    lx = bits(x) & 0xffffffff
    iy = bits(y) >> 32
    ly = bits(y) & 0xffffffff
    if ((ix - 0x3ff00000) & 0xffffffff) | lx == 0:
        return atan(y)
    m = ((iy >> 31) & 1) | ((ix >> 30) & 2)
    ix &= 0x7fffffff
    iy &= 0x7fffffff
    if iy | ly == 0:
        return {0: y, 1: y, 2: PI}.get(m, -PI)
    if ix | lx == 0:
        return -PI / 2.0 if m & 1 else PI / 2.0
    if ix == 0x7ff00000:
        raise ValueError("inf")
    if ((ix + (64 << 20)) & 0xffffffff) < iy or iy == 0x7ff00000:
        return -PI / 2.0 if m & 1 else PI / 2.0
    if (m & 2) and ((iy + (64 << 20)) & 0xffffffff) < ix:
        z = 0.0
    else:
        z = atan(abs(y / x))
    if m == 0: return z
    if m == 1: return -z
    if m == 2: return PI - (z - PI_LO)
    return (z - PI_LO) - PI
