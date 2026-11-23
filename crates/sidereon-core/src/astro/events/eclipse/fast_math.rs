//! Port of the Hipparchus 4.0.3 `FastMath`, `Precision::SAFE_MIN`, and
//! `MathArrays::linearCombination` routines reached by Orekit 13.1.10's
//! occultation and solar radiation pressure models.

#![allow(clippy::excessive_precision, clippy::approx_constant)]

/// Hipparchus `FastMath.PI`.
pub(crate) const PI: f64 = 105414357.0 / 33554432.0 + 1.984187159361080883e-9;

/// Hipparchus `Precision.SAFE_MIN` (`2^-1022`).
pub(crate) const SAFE_MIN: f64 = f64::from_bits((1023_u64 - 1022_u64) << 52);

const HEX_40000000: f64 = 0x4000_0000_u64 as f64;
const MASK_30BITS: u64 = !0_u64 - (0x4000_0000_u64 - 1);
const MASK_NON_SIGN_LONG: u64 = 0x7fff_ffff_ffff_ffff;
const TWO_POWER_52: f64 = 4503599627370496.0;

const F_1_2: f64 = 1.0 / 2.0;
const F_1_4: f64 = 1.0 / 4.0;
const F_3_4: f64 = 3.0 / 4.0;

const SINE_TABLE_A: [f64; 14] = [
    0.0,
    0.1246747374534607,
    0.24740394949913025,
    0.366272509098053,
    0.4794255495071411,
    0.5850973129272461,
    0.6816387176513672,
    0.7675435543060303,
    0.8414709568023682,
    0.902267575263977,
    0.9489846229553223,
    0.9808930158615112,
    0.9974949359893799,
    0.9985313415527344,
];

const SINE_TABLE_B: [f64; 14] = [
    0.0,
    -4.068233003401932e-9,
    9.755392680573412e-9,
    1.9987994582857286e-8,
    -1.0902938113007961e-8,
    -3.9986783938944604e-8,
    4.23719669792332e-8,
    -5.207000323380292e-8,
    2.800552834259e-8,
    1.883511811213715e-8,
    -3.5997360512765566e-9,
    4.116164446561962e-8,
    5.0614674548127384e-8,
    -1.0129027912496858e-9,
];

const COSINE_TABLE_A: [f64; 14] = [
    1.0,
    0.9921976327896118,
    0.9689123630523682,
    0.9305076599121094,
    0.8775825500488281,
    0.8109631538391113,
    0.7316888570785522,
    0.6409968137741089,
    0.5403022766113281,
    0.4311765432357788,
    0.3153223395347595,
    0.19454771280288696,
    0.07073719799518585,
    -0.05417713522911072,
];

const COSINE_TABLE_B: [f64; 14] = [
    0.0,
    3.4439717236742845e-8,
    5.865827662008209e-8,
    -3.7999795083850525e-8,
    1.184154459111628e-8,
    -3.43338934259355e-8,
    1.1795268640216787e-8,
    4.438921624363781e-8,
    2.925681159240093e-8,
    -2.6437112632041807e-8,
    2.2860509143963117e-8,
    -4.813899778443457e-9,
    3.6725170580355583e-9,
    2.0217439756338078e-10,
];

const TANGENT_TABLE_A: [f64; 14] = [
    0.0,
    0.1256551444530487,
    0.25534194707870483,
    0.3936265707015991,
    0.5463024377822876,
    0.7214844226837158,
    0.9315965175628662,
    1.1974215507507324,
    1.5574076175689697,
    2.092571258544922,
    3.0095696449279785,
    5.041914939880371,
    14.101419448852539,
    -18.430862426757812,
];

const TANGENT_TABLE_B: [f64; 14] = [
    0.0,
    -7.877917738262007e-9,
    -2.5857668567479893e-8,
    5.2240336371356666e-9,
    5.206150291559893e-8,
    1.8307188599677033e-8,
    -5.7618793749770706e-8,
    7.848361555046424e-8,
    1.0708593250394448e-7,
    1.7827257129423813e-8,
    2.893485277253286e-8,
    3.1660099222737955e-7,
    4.983191803254889e-7,
    -3.356118100840571e-7,
];

const EIGHTHS: [f64; 14] = [
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625,
];

const RECIP_2PI: [u64; 18] = [
    (0x28be_60db_u64 << 32) | 0x9391_054a,
    (0x7f09_d5f4_u64 << 32) | 0x7d4d_3770,
    (0x36d8_a566_u64 << 32) | 0x4f10_e410,
    (0x7f94_58ea_u64 << 32) | 0xf7ae_f158,
    (0x6dc9_1b8e_u64 << 32) | 0x9093_74b8,
    (0x0192_4bba_u64 << 32) | 0x8274_6487,
    (0x3f87_7ac7_u64 << 32) | 0x2c4a_69cf,
    (0xba20_8d7d_u64 << 32) | 0x4bae_d121,
    (0x3a67_1c09_u64 << 32) | 0xad17_df90,
    (0x4e64_758e_u64 << 32) | 0x60d4_ce7d,
    (0x2721_17e2_u64 << 32) | 0xef7e_4a0e,
    (0xc7fe_25ff_u64 << 32) | 0xf781_6603,
    (0xfbcb_c462_u64 << 32) | 0xd682_9b47,
    (0xdb4d_9fb3_u64 << 32) | 0xc9f2_c26d,
    (0xd3d1_8fd9_u64 << 32) | 0xa797_fa8b,
    (0x5d49_eeb1_u64 << 32) | 0xfaf9_7c5e,
    (0xcf41_ce7d_u64 << 32) | 0xe294_a4ba,
    0x9afe_d7ec_u64 << 32,
];

const PI_O_4_BITS: [u64; 2] = [
    (0xc90f_daa2_u64 << 32) | 0x2168_c234,
    (0xc4c6_628b_u64 << 32) | 0x80dc_1cd1,
];

/// Java `Double.doubleToLongBits`: canonicalizes NaN to `0x7ff8000000000000`.
#[inline]
pub(crate) fn double_to_long_bits(x: f64) -> i64 {
    if x.is_nan() {
        0x7ff8_0000_0000_0000_u64 as i64
    } else {
        x.to_bits() as i64
    }
}

#[inline]
pub(crate) fn sqrt(a: f64) -> f64 {
    a.sqrt()
}

#[inline]
pub(crate) fn abs(x: f64) -> f64 {
    f64::from_bits(MASK_NON_SIGN_LONG & x.to_bits())
}

#[inline]
pub(crate) fn copy_sign(magnitude: f64, sign: f64) -> f64 {
    let m = magnitude.to_bits() as i64;
    let s = sign.to_bits() as i64;
    if (m ^ s) >= 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[inline]
pub(crate) fn get_exponent(d: f64) -> i32 {
    (((d.to_bits() >> 52) & 0x7ff) as i32) - 1023
}

#[inline]
pub(crate) fn min(a: f64, b: f64) -> f64 {
    if a > b {
        return b;
    }
    if a < b {
        return a;
    }
    if a != b {
        return f64::NAN;
    }
    let bits = a.to_bits();
    if bits == 0x8000_0000_0000_0000 {
        return a;
    }
    b
}

#[inline]
pub(crate) fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    if a < b {
        return b;
    }
    if a != b {
        return f64::NAN;
    }
    let bits = a.to_bits();
    if bits == 0x8000_0000_0000_0000 {
        return b;
    }
    a
}

#[inline]
pub(crate) fn next_after(d: f64, direction: f64) -> f64 {
    if d.is_nan() || direction.is_nan() {
        f64::NAN
    } else if d == direction {
        direction
    } else if d.is_infinite() {
        if d < 0.0 {
            -f64::MAX
        } else {
            f64::MAX
        }
    } else if d == 0.0 {
        if direction < 0.0 {
            -f64::from_bits(1)
        } else {
            f64::from_bits(1)
        }
    } else {
        let bits = d.to_bits();
        let sign = bits & 0x8000_0000_0000_0000;
        if (direction < d) ^ (sign == 0) {
            f64::from_bits(sign | ((bits & 0x7fff_ffff_ffff_ffff) + 1))
        } else {
            f64::from_bits(sign | ((bits & 0x7fff_ffff_ffff_ffff) - 1))
        }
    }
}

#[inline]
pub(crate) fn next_down(a: f64) -> f64 {
    next_after(a, f64::NEG_INFINITY)
}

pub(crate) fn scalb(d: f64, n: i32) -> f64 {
    if n > -1023 && n < 1024 {
        return d * f64::from_bits(((n + 1023) as u64) << 52);
    }
    if d.is_nan() || d.is_infinite() || d == 0.0 {
        return d;
    }
    if n < -2098 {
        return if d > 0.0 { 0.0 } else { -0.0 };
    }
    if n > 2097 {
        return if d > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }

    let bits = d.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mut mantissa = bits & 0x000f_ffff_ffff_ffff;
    let mut scaled_exponent = exponent + n;

    if n < 0 {
        if scaled_exponent > 0 {
            f64::from_bits(sign | ((scaled_exponent as u64) << 52) | mantissa)
        } else if scaled_exponent > -53 {
            mantissa |= 1_u64 << 52;
            let most_significant_lost_bit = mantissa & (1_u64 << (-scaled_exponent));
            mantissa >>= 1 - scaled_exponent;
            if most_significant_lost_bit != 0 {
                mantissa += 1;
            }
            f64::from_bits(sign | mantissa)
        } else if sign == 0 {
            0.0
        } else {
            -0.0
        }
    } else if exponent == 0 {
        while (mantissa >> 52) != 1 {
            mantissa <<= 1;
            scaled_exponent -= 1;
        }
        scaled_exponent += 1;
        mantissa &= 0x000f_ffff_ffff_ffff;
        if scaled_exponent < 2047 {
            f64::from_bits(sign | ((scaled_exponent as u64) << 52) | mantissa)
        } else if sign == 0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else if scaled_exponent < 2047 {
        f64::from_bits(sign | ((scaled_exponent as u64) << 52) | mantissa)
    } else if sign == 0 {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    }
}

pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        f64::INFINITY
    } else if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        let exp_x = get_exponent(x);
        let exp_y = get_exponent(y);
        if exp_x > exp_y + 27 {
            abs(x)
        } else if exp_y > exp_x + 27 {
            abs(y)
        } else {
            let middle_exp = (exp_x + exp_y) / 2;
            let scaled_x = scalb(x, -middle_exp);
            let scaled_y = scalb(y, -middle_exp);
            let scaled_h = sqrt(scaled_x * scaled_x + scaled_y * scaled_y);
            scalb(scaled_h, middle_exp)
        }
    }
}

#[inline]
fn double_high_part(d: f64) -> f64 {
    if d > -SAFE_MIN && d < SAFE_MIN {
        return d;
    }
    let xl = d.to_bits() & MASK_30BITS;
    f64::from_bits(xl)
}

fn atan_inner(mut xa: f64, mut xb: f64, left_plane: bool) -> f64 {
    if xa == 0.0 {
        return if left_plane { copy_sign(PI, xa) } else { xa };
    }

    let negate = if xa < 0.0 {
        xa = -xa;
        xb = -xb;
        true
    } else {
        false
    };

    if xa > 1.633123935319537e16 {
        return if negate ^ left_plane {
            -PI * F_1_2
        } else {
            PI * F_1_2
        };
    }

    let idx = if xa < 1.0 {
        (((-1.7168146928204136 * xa * xa + 8.0) * xa) + 0.5) as usize
    } else {
        let one_over_xa = 1.0 / xa;
        (-((-1.7168146928204136 * one_over_xa * one_over_xa + 8.0) * one_over_xa) + 13.07) as usize
    };

    let tt_a = TANGENT_TABLE_A[idx];
    let tt_b = TANGENT_TABLE_B[idx];

    let mut eps_a = xa - tt_a;
    let mut eps_b = -(eps_a - xa + tt_a);
    eps_b += xb - tt_b;

    let mut temp = eps_a + eps_b;
    eps_b = -(temp - eps_a - eps_b);
    eps_a = temp;

    temp = xa * HEX_40000000;
    let mut ya = xa + temp - temp;
    let mut yb = xb + xa - ya;
    xa = ya;
    xb += yb;

    if idx == 0 {
        let denom = 1.0 / (1.0 + (xa + xb) * (tt_a + tt_b));
        ya = eps_a * denom;
        yb = eps_b * denom;
    } else {
        let mut temp2 = xa * tt_a;
        let mut za = 1.0 + temp2;
        let mut zb = -(za - 1.0 - temp2);
        temp2 = xb * tt_a + xa * tt_b;
        temp = za + temp2;
        zb += -(temp - za - temp2);
        za = temp;

        zb += xb * tt_b;
        ya = eps_a / za;

        temp = ya * HEX_40000000;
        let yaa = (ya + temp) - temp;
        let yab = ya - yaa;

        temp = za * HEX_40000000;
        let zaa = (za + temp) - temp;
        let zab = za - zaa;

        yb = (eps_a - yaa * zaa - yaa * zab - yab * zaa - yab * zab) / za;
        yb += -eps_a * zb / za / za;
        yb += eps_b / za;
    }

    eps_a = ya;
    eps_b = yb;

    let eps_a2 = eps_a * eps_a;

    yb = 0.07490822288864472;
    yb = yb * eps_a2 - 0.09088450866185192;
    yb = yb * eps_a2 + 0.11111095942313305;
    yb = yb * eps_a2 - 0.1428571423679182;
    yb = yb * eps_a2 + 0.19999999999923582;
    yb = yb * eps_a2 - 0.33333333333333287;
    yb = yb * eps_a2 * eps_a;

    ya = eps_a;

    temp = ya + yb;
    yb = -(temp - ya - yb);
    ya = temp;

    yb += eps_b / (1.0 + eps_a * eps_a);

    let eighths = EIGHTHS[idx];

    let mut za = eighths + ya;
    let mut zb = -(za - eighths - ya);
    temp = za + yb;
    zb += -(temp - za - yb);
    za = temp;

    let mut result = za + zb;

    if left_plane {
        let resultb = -(result - za - zb);
        let pia = 1.5707963267948966 * 2.0;
        let pib = 6.123233995736766e-17 * 2.0;

        za = pia - result;
        zb = -(za - pia + result);
        zb += pib - resultb;

        result = za + zb;
    }

    if negate ^ left_plane {
        result = -result;
    }

    result
}

pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    if y == 0.0 {
        let result = x * y;
        let invx = 1.0 / x;

        if invx == 0.0 {
            if x > 0.0 {
                return y;
            } else {
                return copy_sign(PI, y);
            }
        }

        if x < 0.0 || invx < 0.0 {
            return copy_sign(PI, y);
        } else {
            return result;
        }
    }

    if y == f64::INFINITY {
        if x == f64::INFINITY {
            return PI * F_1_4;
        }
        if x == f64::NEG_INFINITY {
            return PI * F_3_4;
        }
        return PI * F_1_2;
    }

    if y == f64::NEG_INFINITY {
        if x == f64::INFINITY {
            return -PI * F_1_4;
        }
        if x == f64::NEG_INFINITY {
            return -PI * F_3_4;
        }
        return -PI * F_1_2;
    }

    if x == f64::INFINITY {
        return copy_sign(0.0, y);
    }

    if x == f64::NEG_INFINITY {
        return copy_sign(PI, y);
    }

    if x == 0.0 {
        return copy_sign(PI * F_1_2, y);
    }

    let r = y / x;
    if r.is_infinite() {
        return atan_inner(r, 0.0, x < 0.0);
    }

    let mut ra = double_high_part(r);
    let mut rb = r - ra;

    let xa = double_high_part(x);
    let xb = x - xa;

    rb += (y - ra * xa - ra * xb - rb * xa - rb * xb) / x;

    let temp = ra + rb;
    rb = -(temp - ra - rb);
    ra = temp;

    if ra == 0.0 {
        ra = copy_sign(0.0, y);
    }

    atan_inner(ra, rb, x < 0.0)
}

pub(crate) fn asin(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 1.0 {
        return PI / 2.0;
    }
    if x == -1.0 {
        return -PI / 2.0;
    }
    if x == 0.0 {
        return x;
    }

    let mut temp = x * HEX_40000000;
    let xa = x + temp - temp;
    let xb = x - xa;

    let mut ya = xa * xa;
    let mut yb = xa * xb * 2.0 + xb * xb;

    ya = -ya;
    yb = -yb;

    let mut za = 1.0 + ya;
    let mut zb = -(za - 1.0 - ya);

    temp = za + yb;
    zb += -(temp - za - yb);
    za = temp;

    let y = sqrt(za);
    temp = y * HEX_40000000;
    ya = y + temp - temp;
    yb = y - ya;

    yb += (za - ya * ya - 2.0 * ya * yb - yb * yb) / (2.0 * y);

    let dx = zb / (2.0 * y);

    let r = x / y;
    temp = r * HEX_40000000;
    let mut ra = r + temp - temp;
    let mut rb = r - ra;

    rb += (x - ra * ya - ra * yb - rb * ya - rb * yb) / y;
    rb += -x * dx / y / y;

    temp = ra + rb;
    rb = -(temp - ra - rb);
    ra = temp;

    atan_inner(ra, rb, false)
}

pub(crate) fn acos(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == -1.0 {
        return PI;
    }
    if x == 1.0 {
        return 0.0;
    }
    if x == 0.0 {
        return PI / 2.0;
    }

    let mut temp = x * HEX_40000000;
    let xa = x + temp - temp;
    let xb = x - xa;

    let mut ya = xa * xa;
    let mut yb = xa * xb * 2.0 + xb * xb;

    ya = -ya;
    yb = -yb;

    let mut za = 1.0 + ya;
    let mut zb = -(za - 1.0 - ya);

    temp = za + yb;
    zb += -(temp - za - yb);
    za = temp;

    let mut y = sqrt(za);
    temp = y * HEX_40000000;
    ya = y + temp - temp;
    yb = y - ya;

    yb += (za - ya * ya - 2.0 * ya * yb - yb * yb) / (2.0 * y);

    yb += zb / (2.0 * y);
    y = ya + yb;
    yb = -(y - ya - yb);

    let r = y / x;
    if r.is_infinite() {
        return PI / 2.0;
    }

    let mut ra = double_high_part(r);
    let mut rb = r - ra;

    rb += (y - ra * xa - ra * xb - rb * xa - rb * xb) / x;
    rb += yb / x;

    temp = ra + rb;
    rb = -(temp - ra - rb);
    ra = temp;

    atan_inner(ra, rb, x < 0.0)
}

fn poly_sine(x: f64) -> f64 {
    let x2 = x * x;
    let mut p = 2.7553817452272217e-6;
    p = p * x2 + -1.9841269659586505e-4;
    p = p * x2 + 0.008333333333329196;
    p = p * x2 + -0.16666666666666666;
    p = p * x2 * x;
    p
}

fn poly_cosine(x: f64) -> f64 {
    let x2 = x * x;
    let mut p = 2.479773539153719e-5;
    p = p * x2 + -0.0013888888689039883;
    p = p * x2 + 0.041666666666621166;
    p = p * x2 + -0.49999999999999994;
    p *= x2;
    p
}

fn sin_q(xa: f64, xb: f64) -> f64 {
    let idx = ((xa * 8.0) + 0.5) as usize;
    let epsilon = xa - EIGHTHS[idx];

    let sint_a = SINE_TABLE_A[idx];
    let sint_b = SINE_TABLE_B[idx];
    let cost_a = COSINE_TABLE_A[idx];
    let cost_b = COSINE_TABLE_B[idx];

    let mut sin_eps_a = epsilon;
    let mut sin_eps_b = poly_sine(epsilon);
    let cos_eps_a = 1.0;
    let cos_eps_b = poly_cosine(epsilon);

    let temp = sin_eps_a * HEX_40000000;
    let temp2 = (sin_eps_a + temp) - temp;
    sin_eps_b += sin_eps_a - temp2;
    sin_eps_a = temp2;

    let mut a = 0.0;
    let mut b = 0.0;

    let mut t = sint_a;
    let mut c = a + t;
    let mut d = -(c - a - t);
    a = c;
    b += d;

    t = cost_a * sin_eps_a;
    c = a + t;
    d = -(c - a - t);
    a = c;
    b += d;

    b = b + sint_a * cos_eps_b + cost_a * sin_eps_b;
    b = b + sint_b + cost_b * sin_eps_a + sint_b * cos_eps_b + cost_b * sin_eps_b;

    if xb != 0.0 {
        t = ((cost_a + cost_b) * (cos_eps_a + cos_eps_b)
            - (sint_a + sint_b) * (sin_eps_a + sin_eps_b))
            * xb;
        c = a + t;
        d = -(c - a - t);
        a = c;
        b += d;
    }

    a + b
}

fn cos_q(xa: f64, xb: f64) -> f64 {
    let pi2a = 1.5707963267948966;
    let pi2b = 6.123233995736766e-17;

    let a = pi2a - xa;
    let mut b = -(a - pi2a + xa);
    b += pi2b - xb;

    sin_q(a, b)
}

fn cody_waite(xa: f64) -> (i32, f64, f64) {
    let mut k = (xa * 0.6366197723675814) as i32;
    let mut rem_a;
    let mut rem_b;
    loop {
        let mut a = -(k as f64) * 1.570796251296997;
        rem_a = xa + a;
        rem_b = -(rem_a - xa - a);

        a = -(k as f64) * 7.549789948768648e-8;
        let mut b = rem_a;
        rem_a = a + b;
        rem_b += -(rem_a - b - a);

        a = -(k as f64) * 6.123233995736766e-17;
        b = rem_a;
        rem_a = a + b;
        rem_b += -(rem_a - b - a);

        if rem_a > 0.0 {
            break;
        }
        k -= 1;
    }
    (k, rem_a, rem_b)
}

fn reduce_payne_hanek(x: f64) -> (i32, f64, f64) {
    let mut inbits = x.to_bits();
    let mut exponent = (((inbits >> 52) & 0x7ff) as i32) - 1023;

    inbits &= 0x000f_ffff_ffff_ffff;
    inbits |= 0x0010_0000_0000_0000;

    exponent += 1;
    inbits <<= 11;

    let idx = (exponent >> 6) as usize;
    let shift = (exponent - ((idx as i32) << 6)) as u32;

    let (mut shpi0, shpi_a, shpi_b) = if shift != 0 {
        let s0 = if idx == 0 {
            0
        } else {
            RECIP_2PI[idx - 1] << shift
        };
        (
            s0 | (RECIP_2PI[idx] >> (64 - shift)),
            (RECIP_2PI[idx] << shift) | (RECIP_2PI[idx + 1] >> (64 - shift)),
            (RECIP_2PI[idx + 1] << shift) | (RECIP_2PI[idx + 2] >> (64 - shift)),
        )
    } else {
        (
            if idx == 0 { 0 } else { RECIP_2PI[idx - 1] },
            RECIP_2PI[idx],
            RECIP_2PI[idx + 1],
        )
    };
    let _ = &mut shpi0;

    let mut a = inbits >> 32;
    let mut b = inbits & 0xffff_ffff;

    let mut c = shpi_a >> 32;
    let mut d = shpi_a & 0xffff_ffff;

    let mut ac = a.wrapping_mul(c);
    let mut bd = b.wrapping_mul(d);
    let mut bc = b.wrapping_mul(c);
    let mut ad = a.wrapping_mul(d);

    let mut prod_b = bd.wrapping_add(ad << 32);
    let mut prod_a = ac.wrapping_add(ad >> 32);

    let mut bita = (bd & 0x8000_0000_0000_0000) != 0;
    let mut bitb = (ad & 0x8000_0000) != 0;
    let mut bitsum = (prod_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod_a = prod_a.wrapping_add(1);
    }

    bita = (prod_b & 0x8000_0000_0000_0000) != 0;
    bitb = (bc & 0x8000_0000) != 0;
    prod_b = prod_b.wrapping_add(bc << 32);
    prod_a = prod_a.wrapping_add(bc >> 32);
    bitsum = (prod_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod_a = prod_a.wrapping_add(1);
    }

    c = shpi_b >> 32;
    d = shpi_b & 0xffff_ffff;
    ac = a.wrapping_mul(c);
    bc = b.wrapping_mul(c);
    ad = a.wrapping_mul(d);
    ac = ac.wrapping_add(bc.wrapping_add(ad) >> 32);

    bita = (prod_b & 0x8000_0000_0000_0000) != 0;
    bitb = (ac & 0x8000_0000_0000_0000) != 0;
    prod_b = prod_b.wrapping_add(ac);
    bitsum = (prod_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod_a = prod_a.wrapping_add(1);
    }

    c = shpi0 >> 32;
    d = shpi0 & 0xffff_ffff;
    bd = b.wrapping_mul(d);
    bc = b.wrapping_mul(c);
    ad = a.wrapping_mul(d);
    prod_a = prod_a.wrapping_add(bd.wrapping_add(bc.wrapping_add(ad) << 32));

    let int_part = (prod_a >> 62) as i32;

    prod_a <<= 2;
    prod_a |= prod_b >> 62;
    prod_b <<= 2;

    a = prod_a >> 32;
    b = prod_a & 0xffff_ffff;
    c = PI_O_4_BITS[0] >> 32;
    d = PI_O_4_BITS[0] & 0xffff_ffff;

    ac = a.wrapping_mul(c);
    bd = b.wrapping_mul(d);
    bc = b.wrapping_mul(c);
    ad = a.wrapping_mul(d);

    let mut prod2_b = bd.wrapping_add(ad << 32);
    let mut prod2_a = ac.wrapping_add(ad >> 32);

    bita = (bd & 0x8000_0000_0000_0000) != 0;
    bitb = (ad & 0x8000_0000) != 0;
    bitsum = (prod2_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod2_a = prod2_a.wrapping_add(1);
    }

    bita = (prod2_b & 0x8000_0000_0000_0000) != 0;
    bitb = (bc & 0x8000_0000) != 0;
    prod2_b = prod2_b.wrapping_add(bc << 32);
    prod2_a = prod2_a.wrapping_add(bc >> 32);
    bitsum = (prod2_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod2_a = prod2_a.wrapping_add(1);
    }

    c = PI_O_4_BITS[1] >> 32;
    d = PI_O_4_BITS[1] & 0xffff_ffff;
    ac = a.wrapping_mul(c);
    bc = b.wrapping_mul(c);
    ad = a.wrapping_mul(d);
    ac = ac.wrapping_add(bc.wrapping_add(ad) >> 32);

    bita = (prod2_b & 0x8000_0000_0000_0000) != 0;
    bitb = (ac & 0x8000_0000_0000_0000) != 0;
    prod2_b = prod2_b.wrapping_add(ac);
    bitsum = (prod2_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod2_a = prod2_a.wrapping_add(1);
    }

    a = prod_b >> 32;
    b = prod_b & 0xffff_ffff;
    c = PI_O_4_BITS[0] >> 32;
    d = PI_O_4_BITS[0] & 0xffff_ffff;
    ac = a.wrapping_mul(c);
    bc = b.wrapping_mul(c);
    ad = a.wrapping_mul(d);
    ac = ac.wrapping_add(bc.wrapping_add(ad) >> 32);

    bita = (prod2_b & 0x8000_0000_0000_0000) != 0;
    bitb = (ac & 0x8000_0000_0000_0000) != 0;
    prod2_b = prod2_b.wrapping_add(ac);
    bitsum = (prod2_b & 0x8000_0000_0000_0000) != 0;
    if (bita && bitb) || ((bita || bitb) && !bitsum) {
        prod2_a = prod2_a.wrapping_add(1);
    }

    let tmp_a = ((prod2_a >> 12) as f64) / TWO_POWER_52;
    let tmp_b = ((((prod2_a & 0xfff) << 40).wrapping_add(prod2_b >> 24)) as f64)
        / TWO_POWER_52
        / TWO_POWER_52;

    let sum_a = tmp_a + tmp_b;
    let sum_b = -(sum_a - tmp_a - tmp_b);

    (int_part, sum_a * 2.0, sum_b * 2.0)
}

pub(crate) fn sin(x: f64) -> f64 {
    let mut negative = false;
    let mut quadrant = 0;
    let mut xa = x;
    let mut xb = 0.0;

    if x < 0.0 {
        negative = true;
        xa = -xa;
    }

    if xa == 0.0 {
        if (x.to_bits() as i64) < 0 {
            return -0.0;
        }
        return 0.0;
    }

    if xa.is_nan() || xa == f64::INFINITY {
        return f64::NAN;
    }

    if xa > 3294198.0 {
        let (k, ra, rb) = reduce_payne_hanek(xa);
        quadrant = k & 3;
        xa = ra;
        xb = rb;
    } else if xa > 1.5707963267948966 {
        let (k, ra, rb) = cody_waite(xa);
        quadrant = k & 3;
        xa = ra;
        xb = rb;
    }

    if negative {
        quadrant ^= 2;
    }

    match quadrant {
        0 => sin_q(xa, xb),
        1 => cos_q(xa, xb),
        2 => -sin_q(xa, xb),
        3 => -cos_q(xa, xb),
        _ => f64::NAN,
    }
}

pub(crate) fn sin_cos(x: f64) -> (f64, f64) {
    let mut negative = false;
    let mut quadrant = 0;
    let mut xa = x;
    let mut xb = 0.0;

    if x < 0.0 {
        negative = true;
        xa = -xa;
    }

    if xa == 0.0 {
        if (x.to_bits() as i64) < 0 {
            return (-0.0, 1.0);
        }
        return (0.0, 1.0);
    }

    if xa.is_nan() || xa == f64::INFINITY {
        return (f64::NAN, f64::NAN);
    }

    if xa > 3294198.0 {
        let (k, ra, rb) = reduce_payne_hanek(xa);
        quadrant = k & 3;
        xa = ra;
        xb = rb;
    } else if xa > 1.5707963267948966 {
        let (k, ra, rb) = cody_waite(xa);
        quadrant = k & 3;
        xa = ra;
        xb = rb;
    }

    match quadrant {
        0 => (
            if negative {
                -sin_q(xa, xb)
            } else {
                sin_q(xa, xb)
            },
            cos_q(xa, xb),
        ),
        1 => (
            if negative {
                -cos_q(xa, xb)
            } else {
                cos_q(xa, xb)
            },
            -sin_q(xa, xb),
        ),
        2 => (
            if negative {
                sin_q(xa, xb)
            } else {
                -sin_q(xa, xb)
            },
            -cos_q(xa, xb),
        ),
        3 => (
            if negative {
                cos_q(xa, xb)
            } else {
                -cos_q(xa, xb)
            },
            sin_q(xa, xb),
        ),
        _ => (f64::NAN, f64::NAN),
    }
}

/// Hipparchus `Precision.equals(x, y, 1)`.
pub(crate) fn precision_equals_1ulp(x: f64, y: f64) -> bool {
    const SGN_MASK: u64 = 0x8000_0000_0000_0000;
    let pos_zero_bits = 0.0_f64.to_bits() as i64;
    let neg_zero_bits = (-0.0_f64).to_bits() as i64;
    let x_int = x.to_bits() as i64;
    let y_int = y.to_bits() as i64;
    let is_equal = if (((x_int ^ y_int) as u64) & SGN_MASK) == 0 {
        x_int.wrapping_sub(y_int).unsigned_abs() <= 1
    } else {
        let (delta_plus, delta_minus) = if x_int < y_int {
            (
                y_int.wrapping_sub(pos_zero_bits),
                x_int.wrapping_sub(neg_zero_bits),
            )
        } else {
            (
                x_int.wrapping_sub(pos_zero_bits),
                y_int.wrapping_sub(neg_zero_bits),
            )
        };
        if delta_plus > 1 {
            false
        } else {
            delta_minus <= (1 - delta_plus)
        }
    };
    is_equal && !x.is_nan() && !y.is_nan()
}

/// Hipparchus `MathArrays.linearCombination(double, double, double, double)`.
#[inline]
pub(crate) fn linear_combination_2(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    const SPLIT_MASK: u64 = (!0_u64) << 27;

    let a1_high = f64::from_bits(a1.to_bits() & SPLIT_MASK);
    let a1_low = a1 - a1_high;
    let b1_high = f64::from_bits(b1.to_bits() & SPLIT_MASK);
    let b1_low = b1 - b1_high;
    let prod1_high = a1 * b1;
    let prod1_low = a1_low * b1_low
        - (((prod1_high - a1_high * b1_high) - a1_low * b1_high) - a1_high * b1_low);

    let a2_high = f64::from_bits(a2.to_bits() & SPLIT_MASK);
    let a2_low = a2 - a2_high;
    let b2_high = f64::from_bits(b2.to_bits() & SPLIT_MASK);
    let b2_low = b2 - b2_high;
    let prod2_high = a2 * b2;
    let prod2_low = a2_low * b2_low
        - (((prod2_high - a2_high * b2_high) - a2_low * b2_high) - a2_high * b2_low);

    let s12_high = prod1_high + prod2_high;
    let s12_prime = s12_high - prod2_high;
    let s12_low = (prod2_high - (s12_high - s12_prime)) + (prod1_high - s12_prime);

    let mut result = s12_high + (prod1_low + prod2_low + s12_low);
    if result.is_nan() || result == 0.0 {
        result = a1 * b1 + a2 * b2;
    }
    result
}

/// Hipparchus `MathArrays.linearCombination(double x6)`.
#[inline]
pub(crate) fn linear_combination_3(a1: f64, b1: f64, a2: f64, b2: f64, a3: f64, b3: f64) -> f64 {
    const SPLIT_MASK: u64 = (!0_u64) << 27;

    let a1_high = f64::from_bits(a1.to_bits() & SPLIT_MASK);
    let a1_low = a1 - a1_high;
    let b1_high = f64::from_bits(b1.to_bits() & SPLIT_MASK);
    let b1_low = b1 - b1_high;
    let prod1_high = a1 * b1;
    let prod1_low = a1_low * b1_low
        - (((prod1_high - a1_high * b1_high) - a1_low * b1_high) - a1_high * b1_low);

    let a2_high = f64::from_bits(a2.to_bits() & SPLIT_MASK);
    let a2_low = a2 - a2_high;
    let b2_high = f64::from_bits(b2.to_bits() & SPLIT_MASK);
    let b2_low = b2 - b2_high;
    let prod2_high = a2 * b2;
    let prod2_low = a2_low * b2_low
        - (((prod2_high - a2_high * b2_high) - a2_low * b2_high) - a2_high * b2_low);

    let a3_high = f64::from_bits(a3.to_bits() & SPLIT_MASK);
    let a3_low = a3 - a3_high;
    let b3_high = f64::from_bits(b3.to_bits() & SPLIT_MASK);
    let b3_low = b3 - b3_high;
    let prod3_high = a3 * b3;
    let prod3_low = a3_low * b3_low
        - (((prod3_high - a3_high * b3_high) - a3_low * b3_high) - a3_high * b3_low);

    let s12_high = prod1_high + prod2_high;
    let s12_prime = s12_high - prod2_high;
    let s12_low = (prod2_high - (s12_high - s12_prime)) + (prod1_high - s12_prime);

    let s123_high = s12_high + prod3_high;
    let s123_prime = s123_high - prod3_high;
    let s123_low = (prod3_high - (s123_high - s123_prime)) + (s12_high - s123_prime);

    let mut result = s123_high + (prod1_low + prod2_low + prod3_low + s12_low + s123_low);
    if result.is_nan() || result == 0.0 {
        result = a1 * b1 + a2 * b2 + a3 * b3;
    }
    result
}

/// Hipparchus `MathArrays.linearCombination(double x8)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn linear_combination_4(
    a1: f64,
    b1: f64,
    a2: f64,
    b2: f64,
    a3: f64,
    b3: f64,
    a4: f64,
    b4: f64,
) -> f64 {
    const SPLIT_MASK: u64 = (!0_u64) << 27;

    let a1_high = f64::from_bits(a1.to_bits() & SPLIT_MASK);
    let a1_low = a1 - a1_high;
    let b1_high = f64::from_bits(b1.to_bits() & SPLIT_MASK);
    let b1_low = b1 - b1_high;
    let prod1_high = a1 * b1;
    let prod1_low = a1_low * b1_low
        - (((prod1_high - a1_high * b1_high) - a1_low * b1_high) - a1_high * b1_low);

    let a2_high = f64::from_bits(a2.to_bits() & SPLIT_MASK);
    let a2_low = a2 - a2_high;
    let b2_high = f64::from_bits(b2.to_bits() & SPLIT_MASK);
    let b2_low = b2 - b2_high;
    let prod2_high = a2 * b2;
    let prod2_low = a2_low * b2_low
        - (((prod2_high - a2_high * b2_high) - a2_low * b2_high) - a2_high * b2_low);

    let a3_high = f64::from_bits(a3.to_bits() & SPLIT_MASK);
    let a3_low = a3 - a3_high;
    let b3_high = f64::from_bits(b3.to_bits() & SPLIT_MASK);
    let b3_low = b3 - b3_high;
    let prod3_high = a3 * b3;
    let prod3_low = a3_low * b3_low
        - (((prod3_high - a3_high * b3_high) - a3_low * b3_high) - a3_high * b3_low);

    let a4_high = f64::from_bits(a4.to_bits() & SPLIT_MASK);
    let a4_low = a4 - a4_high;
    let b4_high = f64::from_bits(b4.to_bits() & SPLIT_MASK);
    let b4_low = b4 - b4_high;
    let prod4_high = a4 * b4;
    let prod4_low = a4_low * b4_low
        - (((prod4_high - a4_high * b4_high) - a4_low * b4_high) - a4_high * b4_low);

    let s12_high = prod1_high + prod2_high;
    let s12_prime = s12_high - prod2_high;
    let s12_low = (prod2_high - (s12_high - s12_prime)) + (prod1_high - s12_prime);

    let s123_high = s12_high + prod3_high;
    let s123_prime = s123_high - prod3_high;
    let s123_low = (prod3_high - (s123_high - s123_prime)) + (s12_high - s123_prime);

    let s1234_high = s123_high + prod4_high;
    let s1234_prime = s1234_high - prod4_high;
    let s1234_low = (prod4_high - (s1234_high - s1234_prime)) + (s123_high - s1234_prime);

    let mut result = s1234_high
        + (prod1_low + prod2_low + prod3_low + prod4_low + s12_low + s123_low + s1234_low);
    if result.is_nan() || result == 0.0 {
        result = a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
    }
    result
}

/// Hipparchus `MathArrays.linearCombination(double[], double[])` for fixed `N >= 2`.
pub(crate) fn linear_combination_arr<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    const SPLIT_MASK: u64 = (!0_u64) << 27;

    let mut prod_high = [0.0; N];
    let mut prod_low_sum = 0.0;

    for i in 0..N {
        let ai = a[i];
        let a_high = f64::from_bits(ai.to_bits() & SPLIT_MASK);
        let a_low = ai - a_high;

        let bi = b[i];
        let b_high = f64::from_bits(bi.to_bits() & SPLIT_MASK);
        let b_low = bi - b_high;

        prod_high[i] = ai * bi;
        let prod_low =
            a_low * b_low - (((prod_high[i] - a_high * b_high) - a_low * b_high) - a_high * b_low);
        prod_low_sum += prod_low;
    }

    let prod_high_cur = prod_high[0];
    let mut prod_high_next = prod_high[1];
    let mut s_high_prev = prod_high_cur + prod_high_next;
    let mut s_prime = s_high_prev - prod_high_next;
    let mut s_low_sum = (prod_high_next - (s_high_prev - s_prime)) + (prod_high_cur - s_prime);

    for &ph in &prod_high[2..N] {
        prod_high_next = ph;
        let s_high_cur = s_high_prev + prod_high_next;
        s_prime = s_high_cur - prod_high_next;
        s_low_sum += (prod_high_next - (s_high_cur - s_prime)) + (s_high_prev - s_prime);
        s_high_prev = s_high_cur;
    }

    let mut result = s_high_prev + (prod_low_sum + s_low_sum);
    if result.is_nan() || result == 0.0 {
        result = a[0] * b[0];
        for i in 1..N {
            result += a[i] * b[i];
        }
    }
    result
}
