"""Binary64 port of the Rust `libm` crate 0.2.16 `sin`, `cos`, `atan2` and `pow`.

The `libm` crate ports musl, which ports FreeBSD msun; the functions below follow
its kernels (`k_sin`, `k_cos`, `rem_pio2` and `rem_pio2_large`, `atan`, `atan2`,
`pow`) statement for statement, special cases (NaN, infinities, zeros) included,
with every operation rounded to binary64 as Python floats round. Square roots use
`math.sqrt` and `scalbn` uses `math.ldexp`, which are correctly rounded as the
crate's are. The broadcast goldens built with this port are checked at 0 ULP
against the `libm` crate by the Rust tests that read them, as are the SGP4 states
`generate_sgp4_verification.py` builds with it.
"""

import math
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
    if ix >= 0x7ff00000:
        y0 = x - x
        return 0, y0, y0
    # set z = scalbn(|x|, -ilogb(x) + 23)
    ui = bits(x) & ((0xFFFFFFFFFFFFFFFF ^ 1) >> 12)
    ui |= (0x3ff + 23) << 52
    z = fb(ui)
    tx = [0.0, 0.0, 0.0]
    for i in range(2):
        tx[i] = float(_i32_trunc(z))
        z = (z - tx[i]) * X1P24
    tx[2] = z
    i = 2
    while i != 0 and tx[i] == 0.0:
        i -= 1
    n, y0, y1 = rem_pio2_large(tx[: i + 1], (ix >> 20) - (0x3ff + 23))
    if sign:
        return -n, -y0, -y1
    return n, y0, y1


X1P24 = fb(0x4170000000000000)
X1P_24 = fb(0x3E70000000000000)
PIO2_TABLE = [
    1.57079625129699707031e00,
    7.54978941586159635335e-08,
    5.39030252995776476554e-15,
    3.28200341580791294123e-22,
    1.27065575308067607349e-29,
    1.22933308981111328932e-36,
    2.73370053816464559624e-44,
    2.16741683877804819444e-51,
]
IPIO2 = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B, 0x47C419, 0xC367CD, 0xDCE809, 0x2A8359, 0xC4768B, 0x961CA6,
    0xDDAF44, 0xD15719, 0x053EA5, 0xFF0705, 0x3F7E33, 0xE832C2, 0xDE4F98, 0x327DBB, 0xC33D26,
    0xEF6B1E, 0x5EF89F, 0x3A1F35, 0xCAF27F, 0x1D87F1, 0x21907C, 0x7C246A, 0xFA6ED5, 0x772D30,
    0x433B15, 0xC614B5, 0x9D19C3, 0xC2C4AD, 0x414D2C, 0x5D000C, 0x467D86, 0x2D71E3, 0x9AC69B,
    0x006233, 0x7CD2B4, 0x97A7B4, 0xD55537, 0xF63ED7, 0x1810A3, 0xFC764D, 0x2A9D64, 0xABD770,
    0xF87C63, 0x57B07A, 0xE71517, 0x5649C0, 0xD9D63B, 0x3884A7, 0xCB2324, 0x778AD6, 0x23545A,
    0xB91F00, 0x1B0AF1, 0xDFCE19, 0xFF319F, 0x6A1E66, 0x615799, 0x47FBAC, 0xD87F7E, 0xB76522,
    0x89E832, 0x60BFE6, 0xCDC4EF, 0x09366C, 0xD43F5D, 0xD7DE16, 0xDE3B58, 0x929BDE, 0x2822D2,
    0xE88628, 0x4D58E2, 0x32CAC6, 0x16E308, 0xCB7DE0, 0x50C017, 0xA71DF3, 0x5BE018, 0x34132E,
    0x621283, 0x014883, 0x5B8EF5, 0x7FB0AD, 0xF2E91E, 0x434A48, 0xD36710, 0xD8DDAA, 0x425FAE,
    0xCE616A, 0xA4280A, 0xB499D3, 0xF2A606, 0x7F775C, 0x83C2A3, 0x883C61, 0x78738A, 0x5A8CAF,
    0xBDD76F, 0x63A62D, 0xCBBFF4, 0xEF818D, 0x67C126, 0x45CA55, 0x36D9CA, 0xD2A828, 0x8D61C2,
    0x77C912, 0x142604, 0x9B4612, 0xC459C4, 0x44C5C8, 0x91B24D, 0xF31700, 0xAD43D4, 0xE54929,
    0x10D5FD, 0xFCBE00, 0xCC941E, 0xEECE70, 0xF53E13, 0x80F1EC, 0xC3E7B3, 0x28F8C7, 0x940593,
    0x3E71C1, 0xB3092E, 0xF3450B, 0x9C1288, 0x7B20AB, 0x9FB52E, 0xC29247, 0x2F327B, 0x6D550C,
    0x90A772, 0x1FE76B, 0x96CB31, 0x4A1679, 0xE27941, 0x89DFF4, 0x9794E8, 0x84E6E2, 0x973199,
    0x6BED88, 0x365F5F, 0x0EFDBB, 0xB49A48, 0x6CA467, 0x427271, 0x325D8D, 0xB8159F, 0x09E5BC,
    0x25318D, 0x3974F7, 0x1C0530, 0x010C0D, 0x68084B, 0x58EE2C, 0x90AA47, 0x02E774, 0x24D6BD,
    0xA67DF7, 0x72486E, 0xEF169F, 0xA6948E, 0xF691B4, 0x5153D1, 0xF20ACF, 0x339820, 0x7E4BF5,
    0x6863B2, 0x5F3EDD, 0x035D40, 0x7F8985, 0x295255, 0xC06437, 0x10D86D, 0x324832, 0x754C5B,
    0xD4714E, 0x6E5445, 0xC1090B, 0x69F52A, 0xD56614, 0x9D0727, 0x50045D, 0xDB3BB4, 0xC576EA,
    0x17F987, 0x7D6B49, 0xBA271D, 0x296996, 0xACCCC6, 0x5414AD, 0x6AE290, 0x89D988, 0x50722C,
    0xBEA404, 0x940777, 0x7030F3, 0x27FC00, 0xA871EA, 0x49C266, 0x3DE064, 0x83DD97, 0x973FA3,
    0xFD9443, 0x8C860D, 0xDE4131, 0x9D3992, 0x8C70DD, 0xE7B717, 0x3BDF08, 0x2B3715, 0xA0805C,
    0x93805A, 0x921110, 0xD8E80F, 0xAF806C, 0x4BFFDB, 0x0F9038, 0x761859, 0x15A562, 0xBBCB61,
    0xB989C7, 0xBD4010, 0x04F2D2, 0x277549, 0xF6B6EB, 0xBB22DB, 0xAA140A, 0x2F2689, 0x768364,
    0x333B09, 0x1A940E, 0xAA3A51, 0xC2A31D, 0xAEEDAF, 0x12265C, 0x4DC26D, 0x9C7A2D, 0x9756C0,
    0x833F03, 0xF6F009, 0x8C402B, 0x99316D, 0x07B439, 0x15200C, 0x5BC3D8, 0xC492F5, 0x4BADC6,
    0xA5CA4E, 0xCD37A7, 0x36A9E6, 0x9492AB, 0x6842DD, 0xDE6319, 0xEF8C76, 0x528B68, 0x37DBFC,
    0xABA1AE, 0x3115DF, 0xA1AE00, 0xDAFB0C, 0x664D64, 0xB705ED, 0x306529, 0xBF5657, 0x3AFF47,
    0xB9F96A, 0xF3BE75, 0xDF9328, 0x3080AB, 0xF68C66, 0x15CB04, 0x0622FA, 0x1DE4D9, 0xA4B33D,
    0x8F1B57, 0x09CD36, 0xE9424E, 0xA4BE13, 0xB52333, 0x1AAAF0, 0xA8654F, 0xA5C1D2, 0x0F3F0B,
    0xCD785B, 0x76F923, 0x048B7B, 0x721789, 0x53A6C6, 0xE26E6F, 0x00EBEF, 0x584A9B, 0xB7DAC4,
    0xBA66AA, 0xCFCF76, 0x1D02D1, 0x2DF1B1, 0xC1998C, 0x77ADC3, 0xDA4886, 0xA05DF7, 0xF480C6,
    0x2FF0AC, 0x9AECDD, 0xBC5C3F, 0x6DDED0, 0x1FC790, 0xB6DB2A, 0x3A25A3, 0x9AAF00, 0x9353AD,
    0x0457B6, 0xB42D29, 0x7E804B, 0xA707DA, 0x0EAA76, 0xA1597B, 0x2A1216, 0x2DB7DC, 0xFDE5FA,
    0xFEDB89, 0xFDBE89, 0x6C76E4, 0xFCA906, 0x70803E, 0x156E85, 0xFF87FD, 0x073E28, 0x336761,
    0x86182A, 0xEABD4D, 0xAFE7B3, 0x6E6D8F, 0x396795, 0x5BBF31, 0x48D784, 0x16DF30, 0x432DC7,
    0x356125, 0xCE70C9, 0xB8CB30, 0xFD6CBF, 0xA200A4, 0xE46C05, 0xA0DD5A, 0x476F21, 0xD21262,
    0x845CB9, 0x496170, 0xE0566B, 0x015299, 0x375550, 0xB7D51E, 0xC4F133, 0x5F6E13, 0xE4305D,
    0xA92E85, 0xC3B21D, 0x3632A1, 0xA4B708, 0xD4B1EA, 0x21F716, 0xE4698F, 0x77FF27, 0x80030C,
    0x2D408D, 0xA0CD4F, 0x99A520, 0xD3A2B3, 0x0A5D2F, 0x42F9B4, 0xCBDA11, 0xD0BE7D, 0xC1DB9B,
    0xBD17AB, 0x81A2CA, 0x5C6A08, 0x17552E, 0x550027, 0xF0147F, 0x8607E1, 0x640B14, 0x8D4196,
    0xDEBE87, 0x2AFDDA, 0xB6256B, 0x34897B, 0xFEF305, 0x9EBFB9, 0x4F6A68, 0xA82A4A, 0x5AC44F,
    0xBCF82D, 0x985AD7, 0x95C7F4, 0x8D4D0D, 0xA63A20, 0x5F57A4, 0xB13F14, 0x953880, 0x0120CC,
    0x86DD71, 0xB6DEC9, 0xF560BF, 0x11654D, 0x6B0701, 0xACB08C, 0xD0C0B2, 0x485551, 0x0EFB1E,
    0xC37295, 0x3B06A3, 0x3540C0, 0x7BDC06, 0xCC45E0, 0xFA294E, 0xC8CAD6, 0x41F3E8, 0xDE647C,
    0xD8649B, 0x31BED9, 0xC397A4, 0xD45877, 0xC5E369, 0x13DAF0, 0x3C3ABA, 0x461846, 0x5F7555,
    0xF5BDD2, 0xC6926E, 0x5D2EAC, 0xED440E, 0x423E1C, 0x87C461, 0xE9FD29, 0xF3D6E7, 0xCA7C22,
    0x35916F, 0xC5E008, 0x8DD7FF, 0xE26A6E, 0xC6FDB0, 0xC10893, 0x745D7C, 0xB2AD6B, 0x9D6ECD,
    0x7B723E, 0x6A11C6, 0xA9CFF7, 0xDF7329, 0xBAC9B5, 0x5100B7, 0x0DB2E2, 0x24BA74, 0x607DE5,
    0x8AD874, 0x2C150D, 0x0C1881, 0x94667E, 0x162901, 0x767A9F, 0xBEFDFD, 0xEF4556, 0x367ED9,
    0x13D9EC, 0xB9BA8B, 0xFC97C4, 0x27A831, 0xC36EF1, 0x36C594, 0x56A8D8, 0xB5A8B4, 0x0ECCCF,
    0x2D8912, 0x34576F, 0x89562C, 0xE3CE99, 0xB920D6, 0xAA5E6B, 0x9C2A3E, 0xCC5F11, 0x4A0BFD,
    0xFBF4E1, 0x6D3B8E, 0x2C86E2, 0x84D4E9, 0xA9B4FC, 0xD1EEEF, 0xC9352E, 0x61392F, 0x442138,
    0xC8D91B, 0x0AFC81, 0x6A4AFB, 0xD81C2F, 0x84B453, 0x8C994E, 0xCC2254, 0xDC552A, 0xD6C6C0,
    0x96190B, 0xB8701A, 0x649569, 0x605A26, 0xEE523F, 0x0F117F, 0x11B5F4, 0xF5CBFC, 0x2DBC34,
    0xEEBC34, 0xCC5DE8, 0x605EDD, 0x9B8E67, 0xEF3392, 0xB817C9, 0x9B5861, 0xBC57E1, 0xC68351,
    0x103ED8, 0x4871DD, 0xDD1C2D, 0xA118AF, 0x462C21, 0xD7F359, 0x987AD9, 0xC0549E, 0xFA864F,
    0xFC0656, 0xAE79E5, 0x362289, 0x22AD38, 0xDC9367, 0xAAE855, 0x382682, 0x9BE7CA, 0xA40D51,
    0xB13399, 0x0ED7A9, 0x480569, 0xF0B265, 0xA7887F, 0x974C88, 0x36D1F9, 0xB39221, 0x4A827B,
    0x21CF98, 0xDC9F40, 0x5547DC, 0x3A74E1, 0x42EB67, 0xDF9DFE, 0x5FD45E, 0xA4677B, 0x7AACBA,
    0xA2F655, 0x23882B, 0x55BA41, 0x086E59, 0x862A21, 0x834739, 0xE6E389, 0xD49EE5, 0x40FB49,
    0xE956FF, 0xCA0F1C, 0x8A59C5, 0x2BFA94, 0xC5C1D3, 0xCFC50F, 0xAE5ADB, 0x86C547, 0x624385,
    0x3B8621, 0x94792C, 0x876110, 0x7B4C2A, 0x1A2C80, 0x12BF43, 0x902688, 0x893C78, 0xE4C4A8,
    0x7BDBE5, 0xC23AC4, 0xEAF426, 0x8A67F7, 0xBF920D, 0x2BA365, 0xB1933D, 0x0B7CBD, 0xDC51A4,
    0x63DD27, 0xDDE169, 0x19949A, 0x9529A8, 0x28CE68, 0xB4ED09, 0x209F44, 0xCA984E, 0x638270,
    0x237C7E, 0x32B90F, 0x8EF5A7, 0xE75614, 0x08F121, 0x2A9DB5, 0x4D7E6F, 0x5119A5, 0xABF9B5,
    0xD6DF82, 0x61DD96, 0x023616, 0x9F3AC4, 0xA1A283, 0x6DED72, 0x7A8D39, 0xA9B882, 0x5C326B,
    0x5B2746, 0xED3400, 0x7700D2, 0x55F4FC, 0x4D5901, 0x8071E0,
]


def _i32_trunc(v):
    """Rust `v as i32`: truncate toward zero, saturating."""
    if v != v:
        return 0
    if v >= 2147483647.0:
        return 2147483647
    if v <= -2147483648.0:
        return -2147483648
    return int(v)


def rem_pio2_large(x, e0):
    """`rem_pio2_large` with `prec = 1`: returns (n & 7, y0, y1)."""
    jk = 4
    jp = jk
    nx = len(x)
    jx = nx - 1
    jv = int((e0 - 3) / 24)  # Rust integer division truncates
    if jv < 0:
        jv = 0
    q0 = e0 - 24 * (jv + 1)
    f = [0.0] * 20
    fq = [0.0] * 20
    q = [0.0] * 20
    iq = [0] * 20
    j = jv - jx
    m = jx + jk
    for i in range(m + 1):
        f[i] = 0.0 if j < 0 else float(IPIO2[j])
        j += 1
    for i in range(jk + 1):
        fw = 0.0
        for j in range(jx + 1):
            fw += x[j] * f[jx + i - j]
        q[i] = fw
    jz = jk
    while True:
        i = 0
        z = q[jz]
        for j in range(jz, 0, -1):
            fw = float(_i32_trunc(X1P_24 * z))
            iq[i] = _i32_trunc(z - X1P24 * fw)
            z = q[j - 1] + fw
            i += 1
        z = math.ldexp(z, q0)
        z -= 8.0 * math.floor(z * 0.125)
        n = _i32_trunc(z)
        z -= float(n)
        ih = 0
        if q0 > 0:
            i = iq[jz - 1] >> (24 - q0)
            n += i
            iq[jz - 1] -= i << (24 - q0)
            ih = iq[jz - 1] >> (23 - q0)
        elif q0 == 0:
            ih = iq[jz - 1] >> 23
        elif z >= 0.5:
            ih = 2
        if ih > 0:
            n += 1
            carry = 0
            for i in range(jz):
                j = iq[i]
                if carry == 0:
                    if j != 0:
                        carry = 1
                        iq[i] = 0x1000000 - j
                else:
                    iq[i] = 0xFFFFFF - j
            if q0 > 0:
                if q0 == 1:
                    iq[jz - 1] &= 0x7FFFFF
                elif q0 == 2:
                    iq[jz - 1] &= 0x3FFFFF
            if ih == 2:
                z = 1.0 - z
                if carry != 0:
                    z -= math.ldexp(1.0, q0)
        if z == 0.0:
            j = 0
            for i in range(jz - 1, jk - 1, -1):
                j |= iq[i]
            if j == 0:
                k = 1
                while iq[jk - k] == 0:
                    k += 1
                for i in range(jz + 1, jz + k + 1):
                    f[jx + i] = float(IPIO2[jv + i])
                    fw = 0.0
                    for j in range(jx + 1):
                        fw += x[j] * f[jx + i - j]
                    q[i] = fw
                jz += k
                continue
        break
    if z == 0.0:
        jz -= 1
        q0 -= 24
        while iq[jz] == 0:
            jz -= 1
            q0 -= 24
    else:
        z = math.ldexp(z, -q0)
        if z >= X1P24:
            fw = float(_i32_trunc(X1P_24 * z))
            iq[jz] = _i32_trunc(z - X1P24 * fw)
            jz += 1
            q0 += 24
            iq[jz] = _i32_trunc(fw)
        else:
            iq[jz] = _i32_trunc(z)
    fw = math.ldexp(1.0, q0)
    for i in range(jz, -1, -1):
        q[i] = fw * float(iq[i])
        fw *= X1P_24
    for i in range(jz, -1, -1):
        fw = 0.0
        k = 0
        while k <= jp and k <= jz - i:
            fw += PIO2_TABLE[k] * q[i + k]
            k += 1
        fq[jz - i] = fw
    fw = 0.0
    for i in range(jz, -1, -1):
        fw += fq[i]
    y0 = fw if ih == 0 else -fw
    fw = fq[0] - fw
    for i in range(1, jz + 1):
        fw += fq[i]
    y1 = fw if ih == 0 else -fw
    return n & 7, y0, y1

def sin(x):
    ix = hi(x)
    if ix <= 0x3fe921fb:
        if ix < 0x3e500000:
            return x
        return k_sin(x, 0.0, 0)
    if ix >= 0x7ff00000:
        return x - x
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
    if ix >= 0x7ff00000:
        return x - x
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
        if x != x:
            return x
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
        if iy == 0x7ff00000:
            return {0: PI / 4.0, 1: -PI / 4.0, 2: 3.0 * PI / 4.0}.get(m, -3.0 * PI / 4.0)
        return {0: 0.0, 1: -0.0, 2: PI}.get(m, -PI)
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


# ---------------------------------------------------------------------------
# pow (msun e_pow.c as the crate ports it)

def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def _set_low(x, lo):
    return fb((bits(x) & 0xFFFFFFFF00000000) | (lo & 0xFFFFFFFF))


def _set_high(x, high):
    return fb(((high & 0xFFFFFFFF) << 32) | (bits(x) & 0xFFFFFFFF))


def _get_high(x):
    return (bits(x) >> 32) & 0xFFFFFFFF


BP = [1.0, 1.5]
DP_H = [0.0, 5.84962487220764160156e-01]
DP_L = [0.0, 1.35003920212974897128e-08]
TWO53 = 9007199254740992.0
HUGE = 1.0e300
TINY = 1.0e-300
PL1 = 5.99999999999994648725e-01
PL2 = 4.28571428578550184252e-01
PL3 = 3.33333329818377432918e-01
PL4 = 2.72728123808534006489e-01
PL5 = 2.30660745775561754067e-01
PL6 = 2.06975017800338417784e-01
PP1 = 1.66666666666666019037e-01
PP2 = -2.77777777770155933842e-03
PP3 = 6.61375632143793436117e-05
PP4 = -1.65339022054652515390e-06
PP5 = 4.13813679705723846039e-08
LG2 = 6.93147180559945286227e-01
LG2_H = 6.93147182464599609375e-01
LG2_L = -1.90465429995776804525e-09
OVT = 8.0085662595372944372e-017
CP = 9.61796693925975554329e-01
CP_H = 9.61796700954437255859e-01
CP_L = -7.02846165095275826516e-09
IVLN2 = 1.44269504088896338700e00
IVLN2_H = 1.44269502162933349609e00
IVLN2_L = 1.92596299112661746887e-08


def pow(x, y):
    bx, by = bits(x), bits(y)
    hx, lx = _i32(bx >> 32), bx & 0xFFFFFFFF
    hy, ly = _i32(by >> 32), by & 0xFFFFFFFF
    ix = hx & 0x7FFFFFFF
    iy = hy & 0x7FFFFFFF
    if (iy | ly) == 0:
        return 1.0
    if hx == 0x3FF00000 and lx == 0:
        return 1.0
    if ix > 0x7FF00000 or (ix == 0x7FF00000 and lx != 0) or iy > 0x7FF00000 or (
        iy == 0x7FF00000 and ly != 0
    ):
        return x + y
    yisint = 0
    if hx < 0:
        if iy >= 0x43400000:
            yisint = 2
        elif iy >= 0x3FF00000:
            k = (iy >> 20) - 0x3FF
            if k > 20:
                j = _i32(ly >> (52 - k))
                if _i32(j << (52 - k)) == _i32(ly):
                    yisint = 2 - (j & 1)
            elif ly == 0:
                j = iy >> (20 - k)
                if (j << (20 - k)) == iy:
                    yisint = 2 - (j & 1)
    if ly == 0:
        if iy == 0x7FF00000:
            if ((ix - 0x3FF00000) | _i32(lx)) == 0:
                return 1.0
            if ix >= 0x3FF00000:
                return y if hy >= 0 else 0.0
            return 0.0 if hy >= 0 else -y
        if iy == 0x3FF00000:
            return x if hy >= 0 else 1.0 / x
        if hy == 0x40000000:
            return x * x
        if hy == 0x3FE00000 and hx >= 0:
            return math.sqrt(x)
    ax = abs(x)
    if lx == 0 and (ix == 0x7FF00000 or ix == 0 or ix == 0x3FF00000):
        z = ax
        if hy < 0:
            z = 1.0 / z if z != 0.0 else math.inf
        if hx < 0:
            if ((ix - 0x3FF00000) | yisint) == 0:
                z = math.nan
            elif yisint == 1:
                z = -z
        return z
    s = 1.0
    if hx < 0:
        if yisint == 0:
            return math.nan
        if yisint == 1:
            s = -1.0
    if iy > 0x41E00000:
        if iy > 0x43F00000:
            if ix <= 0x3FEFFFFF:
                return HUGE * HUGE if hy < 0 else TINY * TINY
            if ix >= 0x3FF00000:
                return HUGE * HUGE if hy > 0 else TINY * TINY
        if ix < 0x3FEFFFFF:
            return s * HUGE * HUGE if hy < 0 else s * TINY * TINY
        if ix > 0x3FF00000:
            return s * HUGE * HUGE if hy > 0 else s * TINY * TINY
        t = ax - 1.0
        w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25))
        u = IVLN2_H * t
        v = t * IVLN2_L - w * IVLN2
        t1 = _set_low(u + v, 0)
        t2 = v - (t1 - u)
    else:
        n = 0
        if ix < 0x00100000:
            ax *= TWO53
            n -= 53
            ix = _i32(_get_high(ax))
        n += (ix >> 20) - 0x3FF
        j = ix & 0x000FFFFF
        ix = j | 0x3FF00000
        if j <= 0x3988E:
            k = 0
        elif j < 0xBB67A:
            k = 1
        else:
            k = 0
            n += 1
            ix -= 0x00100000
        ax = _set_high(ax, ix)
        u = ax - BP[k]
        v = 1.0 / (ax + BP[k])
        ss = u * v
        s_h = _set_low(ss, 0)
        t_h = _set_high(0.0, ((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18))
        t_l = ax - (t_h - BP[k])
        s_l = v * ((u - s_h * t_h) - s_h * t_l)
        s2 = ss * ss
        r = s2 * s2 * (PL1 + s2 * (PL2 + s2 * (PL3 + s2 * (PL4 + s2 * (PL5 + s2 * PL6)))))
        r += s_l * (s_h + ss)
        s2 = s_h * s_h
        t_h = _set_low(3.0 + s2 + r, 0)
        t_l = r - ((t_h - 3.0) - s2)
        u = s_h * t_h
        v = s_l * t_h + t_l * ss
        p_h = _set_low(u + v, 0)
        p_l = v - (p_h - u)
        z_h = CP_H * p_h
        z_l = CP_L * p_h + p_l * CP + DP_L[k]
        t = float(n)
        t1 = _set_low(((z_h + z_l) + DP_H[k]) + t, 0)
        t2 = z_l - (((t1 - t) - DP_H[k]) - z_h)
    y1 = _set_low(y, 0)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z = p_l + p_h
    j = _i32(bits(z) >> 32)
    i = _i32(bits(z) & 0xFFFFFFFF)
    if j >= 0x40900000:
        if ((j - 0x40900000) | i) != 0:
            return s * HUGE * HUGE
        if p_l + OVT > z - p_h:
            return s * HUGE * HUGE
    elif (j & 0x7FFFFFFF) >= 0x4090CC00:
        if (((j & 0xFFFFFFFF) - 0xC090CC00) & 0xFFFFFFFF) | (i & 0xFFFFFFFF) != 0:
            return s * TINY * TINY
        if p_l <= z - p_h:
            return s * TINY * TINY
    i = j & 0x7FFFFFFF
    k = (i >> 20) - 0x3FF
    n = 0
    if i > 0x3FE00000:
        n = _i32(j + (0x00100000 >> (k + 1)))
        k = ((n & 0x7FFFFFFF) >> 20) - 0x3FF
        t = _set_high(0.0, n & ~(0x000FFFFF >> k))
        n = ((n & 0x000FFFFF) | 0x00100000) >> (20 - k)
        if j < 0:
            n = -n
        p_h -= t
    t = _set_low(p_l + p_h, 0)
    u = t * LG2_H
    v = (p_l - (t - p_h)) * LG2 + t * LG2_L
    z = u + v
    w = v - (z - u)
    t = z * z
    t1 = z - t * (PP1 + t * (PP2 + t * (PP3 + t * (PP4 + t * PP5))))
    r = (z * t1) / (t1 - 2.0) - (w + z * w)
    z = 1.0 - (r - z)
    j = _i32(_get_high(z))
    j += n << 20
    if (j >> 20) <= 0:
        z = math.ldexp(z, n)
    else:
        z = _set_high(z, j)
    return s * z
