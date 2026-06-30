// Faithful line-by-line port of Vallado's SGP4 C++ implementation to Rust.
// Preserves exact floating-point computation order for 0 ULP parity with
// Python's sgp4 package (which compiles the same C++ source).
//
// Based on: SGP4 Version 2020-07-13, David Vallado
//
// DO NOT rearrange expressions, combine operations, or "optimize."
// Every +, *, -, / must happen in the same order as the C++.
// DO NOT use f64::mul_add() anywhere.

// This module IS the runtime SGP4 implementation: it backs the public
// `Satellite` / `propagate` / `propagate_elements` API. The Vallado C++ in
// tests/cpp is a development-only parity oracle (feature `sgp4-debug-oracle`,
// off by default and excluded from the published crate); no C++ runs in
// normal builds.
//
// Attribution: the SGP4/SDP4 model is the public AIAA/Spacetrack theory; this
// is a Rust port of David Vallado's reference C++ (companion code to
// "Fundamentals of Astrodynamics and Applications"), the same source the
// `sgp4` Python package ports. Credit to David Vallado and the 2006 AIAA
// paper (Vallado, Crawford, Hujsak, Kelso).

const PI_VAL: f64 = 3.14159265358979323846;

#[allow(dead_code)]
const TWOPI: f64 = 2.0 * PI_VAL;

// Gravity constant type -- we only use wgs72
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum GravConstType {
    Wgs72old,
    Wgs72,
    Wgs84,
}

/// The satellite record -- direct port of the C struct `elsetrec`.
/// All fields default to 0/0.0.
#[derive(Clone)]
pub struct ElsetRec {
    pub satnum: [u8; 6],
    pub epochyr: i32,
    pub epochtynumrev: i32,
    pub error: i32,
    pub operationmode: char,
    pub init: char,
    pub method: char,

    // Near Earth
    pub isimp: i32,
    pub aycof: f64,
    pub con41: f64,
    pub cc1: f64,
    pub cc4: f64,
    pub cc5: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
    pub delmo: f64,
    pub eta: f64,
    pub argpdot: f64,
    pub omgcof: f64,
    pub sinmao: f64,
    pub t: f64,
    pub t2cof: f64,
    pub t3cof: f64,
    pub t4cof: f64,
    pub t5cof: f64,
    pub x1mth2: f64,
    pub x7thm1: f64,
    pub mdot: f64,
    pub nodedot: f64,
    pub xlcof: f64,
    pub xmcof: f64,
    pub nodecf: f64,

    // Deep Space
    pub irez: i32,
    pub d2201: f64,
    pub d2211: f64,
    pub d3210: f64,
    pub d3222: f64,
    pub d4410: f64,
    pub d4422: f64,
    pub d5220: f64,
    pub d5232: f64,
    pub d5421: f64,
    pub d5433: f64,
    pub dedt: f64,
    pub del1: f64,
    pub del2: f64,
    pub del3: f64,
    pub didt: f64,
    pub dmdt: f64,
    pub dnodt: f64,
    pub domdt: f64,
    pub e3: f64,
    pub ee2: f64,
    pub peo: f64,
    pub pgho: f64,
    pub pho: f64,
    pub pinco: f64,
    pub plo: f64,
    pub se2: f64,
    pub se3: f64,
    pub sgh2: f64,
    pub sgh3: f64,
    pub sgh4: f64,
    pub sh2: f64,
    pub sh3: f64,
    pub si2: f64,
    pub si3: f64,
    pub sl2: f64,
    pub sl3: f64,
    pub sl4: f64,
    pub gsto: f64,
    pub xfact: f64,
    pub xgh2: f64,
    pub xgh3: f64,
    pub xgh4: f64,
    pub xh2: f64,
    pub xh3: f64,
    pub xi2: f64,
    pub xi3: f64,
    pub xl2: f64,
    pub xl3: f64,
    pub xl4: f64,
    pub xlamo: f64,
    pub zmol: f64,
    pub zmos: f64,
    pub atime: f64,
    pub xli: f64,
    pub xni: f64,

    pub a: f64,
    pub altp: f64,
    pub alta: f64,
    pub epochdays: f64,
    pub jdsatepoch: f64,
    pub jdsatepochF: f64,
    pub nddot: f64,
    pub ndot: f64,
    pub bstar: f64,
    pub rcse: f64,
    pub inclo: f64,
    pub nodeo: f64,
    pub ecco: f64,
    pub argpo: f64,
    pub mo: f64,
    pub no_kozai: f64,

    // sgp4fix add unkozai'd variable
    pub no_unkozai: f64,
    // sgp4fix add singly averaged variables
    pub am: f64,
    pub em: f64,
    pub im: f64,
    pub Om: f64,
    pub om: f64,
    pub mm: f64,
    pub nm: f64,
    // sgp4fix add constant parameters
    pub tumin: f64,
    pub mus: f64,
    pub radiusearthkm: f64,
    pub xke: f64,
    pub j2: f64,
    pub j3: f64,
    pub j4: f64,
    pub j3oj2: f64,
}

impl Default for ElsetRec {
    fn default() -> Self {
        ElsetRec {
            satnum: [0u8; 6],
            epochyr: 0,
            epochtynumrev: 0,
            error: 0,
            operationmode: 'i',
            init: 'n',
            method: 'n',
            isimp: 0,
            aycof: 0.0,
            con41: 0.0,
            cc1: 0.0,
            cc4: 0.0,
            cc5: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            delmo: 0.0,
            eta: 0.0,
            argpdot: 0.0,
            omgcof: 0.0,
            sinmao: 0.0,
            t: 0.0,
            t2cof: 0.0,
            t3cof: 0.0,
            t4cof: 0.0,
            t5cof: 0.0,
            x1mth2: 0.0,
            x7thm1: 0.0,
            mdot: 0.0,
            nodedot: 0.0,
            xlcof: 0.0,
            xmcof: 0.0,
            nodecf: 0.0,
            irez: 0,
            d2201: 0.0,
            d2211: 0.0,
            d3210: 0.0,
            d3222: 0.0,
            d4410: 0.0,
            d4422: 0.0,
            d5220: 0.0,
            d5232: 0.0,
            d5421: 0.0,
            d5433: 0.0,
            dedt: 0.0,
            del1: 0.0,
            del2: 0.0,
            del3: 0.0,
            didt: 0.0,
            dmdt: 0.0,
            dnodt: 0.0,
            domdt: 0.0,
            e3: 0.0,
            ee2: 0.0,
            peo: 0.0,
            pgho: 0.0,
            pho: 0.0,
            pinco: 0.0,
            plo: 0.0,
            se2: 0.0,
            se3: 0.0,
            sgh2: 0.0,
            sgh3: 0.0,
            sgh4: 0.0,
            sh2: 0.0,
            sh3: 0.0,
            si2: 0.0,
            si3: 0.0,
            sl2: 0.0,
            sl3: 0.0,
            sl4: 0.0,
            gsto: 0.0,
            xfact: 0.0,
            xgh2: 0.0,
            xgh3: 0.0,
            xgh4: 0.0,
            xh2: 0.0,
            xh3: 0.0,
            xi2: 0.0,
            xi3: 0.0,
            xl2: 0.0,
            xl3: 0.0,
            xl4: 0.0,
            xlamo: 0.0,
            zmol: 0.0,
            zmos: 0.0,
            atime: 0.0,
            xli: 0.0,
            xni: 0.0,
            a: 0.0,
            altp: 0.0,
            alta: 0.0,
            epochdays: 0.0,
            jdsatepoch: 0.0,
            jdsatepochF: 0.0,
            nddot: 0.0,
            ndot: 0.0,
            bstar: 0.0,
            rcse: 0.0,
            inclo: 0.0,
            nodeo: 0.0,
            ecco: 0.0,
            argpo: 0.0,
            mo: 0.0,
            no_kozai: 0.0,
            no_unkozai: 0.0,
            am: 0.0,
            em: 0.0,
            im: 0.0,
            Om: 0.0,
            om: 0.0,
            mm: 0.0,
            nm: 0.0,
            tumin: 0.0,
            mus: 0.0,
            radiusearthkm: 0.0,
            xke: 0.0,
            j2: 0.0,
            j3: 0.0,
            j4: 0.0,
            j3oj2: 0.0,
        }
    }
}

// ============================================================================
//                           procedure getgravconst
// ============================================================================
pub fn getgravconst(whichconst: GravConstType) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    // returns (tumin, mus, radiusearthkm, xke, j2, j3, j4, j3oj2)
    let tumin: f64;
    let mus: f64;
    let radiusearthkm: f64;
    let xke: f64;
    let j2: f64;
    let j3: f64;
    let j4: f64;
    let j3oj2: f64;

    match whichconst {
        GravConstType::Wgs72old => {
            mus = 398600.79964;
            radiusearthkm = 6378.135;
            xke = 0.0743669161;
            tumin = 1.0 / xke;
            j2 = 0.001082616;
            j3 = -0.00000253881;
            j4 = -0.00000165597;
            j3oj2 = j3 / j2;
        }
        GravConstType::Wgs72 => {
            mus = 398600.8;
            radiusearthkm = 6378.135;
            xke = 60.0 / (radiusearthkm * radiusearthkm * radiusearthkm / mus).sqrt();
            tumin = 1.0 / xke;
            j2 = 0.001082616;
            j3 = -0.00000253881;
            j4 = -0.00000165597;
            j3oj2 = j3 / j2;
        }
        GravConstType::Wgs84 => {
            mus = 398600.5;
            radiusearthkm = 6378.137;
            xke = 60.0 / (radiusearthkm * radiusearthkm * radiusearthkm / mus).sqrt();
            tumin = 1.0 / xke;
            j2 = 0.00108262998905;
            j3 = -0.00000253215306;
            j4 = -0.00000161098761;
            j3oj2 = j3 / j2;
        }
    }

    (tumin, mus, radiusearthkm, xke, j2, j3, j4, j3oj2)
}

// The SGP4 time math lives in the `vallado_time` parity-adapter submodule below;
// the calls in this file resolve through this re-export, and the rest of the
// SGP4 module reaches them as `vallado::jday_SGP4` etc.
pub use vallado_time::{days2mdhms_SGP4, gstime_SGP4, jday_SGP4};

/// SGP4 / Vallado time adapter.
///
/// These three routines are the time math of David Vallado's reference SGP4,
/// ported line-for-line for 0-ULP parity with Python's `sgp4` package. They are
/// deliberately NOT part of [`crate::astro::time::civil`]: `days2mdhms_SGP4`
/// uses Vallado's plain `year % 4` leap rule (no century correction) and
/// `jday_SGP4` uses Vallado's `367*year` closed form, both of which differ from
/// the rigorous proleptic-Gregorian civil conversions. Keeping them isolated in
/// this SGP4-named submodule marks them as a parity adapter so they are not
/// mistaken for a duplicate of the canonical time helpers and are never
/// consolidated into them.
///
/// DO NOT rearrange expressions, combine operations, or "optimize." Every
/// +, *, -, / must happen in the same order as the Vallado C++.
pub mod vallado_time {
    use super::PI_VAL;

    // ========================================================================
    //                           procedure gstime_SGP4
    // ========================================================================
    pub fn gstime_SGP4(jdut1: f64) -> f64 {
        let twopi = 2.0 * PI_VAL;
        let deg2rad = PI_VAL / 180.0;

        let tut1 = (jdut1 - 2451545.0) / 36525.0;
        let mut temp = -6.2e-6 * tut1 * tut1 * tut1
            + 0.093104 * tut1 * tut1
            + (876600.0 * 3600.0 + 8640184.812866) * tut1
            + 67310.54841; // sec
        temp = (temp * deg2rad / 240.0) % twopi; //360/86400 = 1/240, to deg, to rad

        // check quadrants
        if temp < 0.0 {
            temp = temp + twopi;
        }

        temp
    }

    // ========================================================================
    //                           procedure jday_SGP4
    // ========================================================================
    pub fn jday_SGP4(year: i32, mon: i32, day: i32, hr: i32, minute: i32, sec: f64) -> (f64, f64) {
        let mut jd = 367.0 * year as f64
            - (7.0 * (year as f64 + ((mon as f64 + 9.0) / 12.0).floor()) * 0.25).floor()
            + (275.0 * mon as f64 / 9.0).floor()
            + day as f64
            + 1721013.5;
        let mut jdFrac = (sec + minute as f64 * 60.0 + hr as f64 * 3600.0) / 86400.0;

        // check that the day and fractional day are correct
        if jdFrac.abs() > 1.0 {
            let dtt = jdFrac.floor();
            jd = jd + dtt;
            jdFrac = jdFrac - dtt;
        }

        (jd, jdFrac)
    }

    // ========================================================================
    //                           procedure days2mdhms_SGP4
    // ========================================================================
    pub fn days2mdhms_SGP4(year: i32, days: f64) -> (i32, i32, i32, i32, f64) {
        // returns (mon, day, hr, minute, sec)
        let mut lmonth: [i32; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let dayofyr = days.floor() as i32;
        if (year % 4) == 0 {
            lmonth[2] = 29;
        }

        let mut i: usize = 1;
        let mut inttemp: i32 = 0;
        while (dayofyr > inttemp + lmonth[i]) && (i < 12) {
            inttemp = inttemp + lmonth[i];
            i += 1;
        }
        let mon = i as i32;
        let day = dayofyr - inttemp;

        let mut temp = (days - dayofyr as f64) * 24.0;
        let hr = temp.floor() as i32;
        temp = (temp - hr as f64) * 60.0;
        let minute = temp.floor() as i32;
        let sec = (temp - minute as f64) * 60.0;

        (mon, day, hr, minute, sec)
    }
}

// ============================================================================
//                           procedure dpper
// ============================================================================
#[allow(unused_variables)]
fn dpper(
    e3: f64,
    ee2: f64,
    peo: f64,
    pgho: f64,
    pho: f64,
    pinco: f64,
    plo: f64,
    se2: f64,
    se3: f64,
    sgh2: f64,
    sgh3: f64,
    sgh4: f64,
    sh2: f64,
    sh3: f64,
    si2: f64,
    si3: f64,
    sl2: f64,
    sl3: f64,
    sl4: f64,
    t: f64,
    xgh2: f64,
    xgh3: f64,
    xgh4: f64,
    xh2: f64,
    xh3: f64,
    xi2: f64,
    xi3: f64,
    xl2: f64,
    xl3: f64,
    xl4: f64,
    zmol: f64,
    zmos: f64,
    inclo: f64,
    init: char,
    ep: &mut f64,
    inclp: &mut f64,
    nodep: &mut f64,
    argpp: &mut f64,
    mp: &mut f64,
    opsmode: char,
) {
    /* --------------------- local variables ------------------------ */
    let twopi = 2.0 * PI_VAL;
    let mut alfdp: f64 = 0.0;
    let mut betdp: f64 = 0.0;
    let mut cosip: f64 = 0.0;
    let mut cosop: f64 = 0.0;
    let mut dalf: f64 = 0.0;
    let mut dbet: f64 = 0.0;
    let mut dls: f64 = 0.0;
    let mut f2: f64 = 0.0;
    let mut f3: f64 = 0.0;
    let mut pe: f64 = 0.0;
    let mut pgh: f64 = 0.0;
    let mut ph: f64 = 0.0;
    let mut pinc: f64 = 0.0;
    let mut pl: f64 = 0.0;
    let mut sel: f64 = 0.0;
    let mut ses: f64 = 0.0;
    let mut sghl: f64 = 0.0;
    let mut sghs: f64 = 0.0;
    let mut shll: f64 = 0.0;
    let mut shs: f64 = 0.0;
    let mut sil: f64 = 0.0;
    let mut sinip: f64 = 0.0;
    let mut sinop: f64 = 0.0;
    let mut sinzf: f64 = 0.0;
    let mut sis: f64 = 0.0;
    let mut sll: f64 = 0.0;
    let mut sls: f64 = 0.0;
    let mut xls: f64 = 0.0;
    let mut xnoh: f64 = 0.0;
    let mut zf: f64 = 0.0;
    let mut zm: f64 = 0.0;
    let mut zel: f64 = 0.0;
    let mut zes: f64 = 0.0;
    let mut znl: f64 = 0.0;
    let mut zns: f64 = 0.0;

    /* ---------------------- constants ----------------------------- */
    zns = 1.19459e-5;
    zes = 0.01675;
    znl = 1.5835218e-4;
    zel = 0.05490;

    /* --------------- calculate time varying periodics ----------- */
    zm = zmos + zns * t;
    // be sure that the initial call has time set to zero
    if init == 'y' {
        zm = zmos;
    }
    zf = zm + 2.0 * zes * zm.sin();
    sinzf = zf.sin();
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * zf.cos();
    ses = se2 * f2 + se3 * f3;
    sis = si2 * f2 + si3 * f3;
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf;
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf;
    shs = sh2 * f2 + sh3 * f3;
    zm = zmol + znl * t;
    if init == 'y' {
        zm = zmol;
    }
    zf = zm + 2.0 * zel * zm.sin();
    sinzf = zf.sin();
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * zf.cos();
    sel = ee2 * f2 + e3 * f3;
    sil = xi2 * f2 + xi3 * f3;
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf;
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf;
    shll = xh2 * f2 + xh3 * f3;
    pe = ses + sel;
    pinc = sis + sil;
    pl = sls + sll;
    pgh = sghs + sghl;
    ph = shs + shll;

    if init == 'n' {
        pe = pe - peo;
        pinc = pinc - pinco;
        pl = pl - plo;
        pgh = pgh - pgho;
        ph = ph - pho;
        *inclp = *inclp + pinc;
        *ep = *ep + pe;
        sinip = (*inclp).sin();
        cosip = (*inclp).cos();

        /* ----------------- apply periodics directly ------------ */
        //  sgp4fix for lyddane choice
        //  strn3 used original inclination - this is technically feasible
        //  gsfc used perturbed inclination - also technically feasible
        //  probably best to readjust the 0.2 limit value and limit discontinuity
        //  0.2 rad = 11.45916 deg
        //  use next line for original strn3 approach and original inclination
        //  if (inclo >= 0.2)
        //  use next line for gsfc version and perturbed inclination
        if *inclp >= 0.2 {
            ph = ph / sinip;
            pgh = pgh - cosip * ph;
            *argpp = *argpp + pgh;
            *nodep = *nodep + ph;
            *mp = *mp + pl;
        } else {
            /* ---- apply periodics with lyddane modification ---- */
            sinop = (*nodep).sin();
            cosop = (*nodep).cos();
            alfdp = sinip * sinop;
            betdp = sinip * cosop;
            dalf = ph * cosop + pinc * cosip * sinop;
            dbet = -ph * sinop + pinc * cosip * cosop;
            let alfdp = alfdp + dalf;
            let betdp = betdp + dbet;
            *nodep = (*nodep) % twopi;
            //  sgp4fix for afspc written intrinsic functions
            // nodep used without a trigonometric function ahead
            if (*nodep < 0.0) && (opsmode == 'a') {
                *nodep = *nodep + twopi;
            }
            xls = *mp + *argpp + cosip * *nodep;
            dls = pl + pgh - pinc * *nodep * sinip;
            let xls = xls + dls;
            xnoh = *nodep;
            *nodep = alfdp.atan2(betdp);
            //  sgp4fix for afspc written intrinsic functions
            // nodep used without a trigonometric function ahead
            if (*nodep < 0.0) && (opsmode == 'a') {
                *nodep = *nodep + twopi;
            }
            if (xnoh - *nodep).abs() > PI_VAL {
                if *nodep < xnoh {
                    *nodep = *nodep + twopi;
                } else {
                    *nodep = *nodep - twopi;
                }
            }
            *mp = *mp + pl;
            *argpp = xls - *mp - cosip * *nodep;
        }
    } // if init == 'n'
}

// ============================================================================
//                           procedure dscom
// ============================================================================
struct DscomResult {
    snodm: f64,
    cnodm: f64,
    sinim: f64,
    cosim: f64,
    sinomm: f64,
    cosomm: f64,
    day: f64,
    e3: f64,
    ee2: f64,
    em: f64,
    emsq: f64,
    gam: f64,
    peo: f64,
    pgho: f64,
    pho: f64,
    pinco: f64,
    plo: f64,
    rtemsq: f64,
    se2: f64,
    se3: f64,
    sgh2: f64,
    sgh3: f64,
    sgh4: f64,
    sh2: f64,
    sh3: f64,
    si2: f64,
    si3: f64,
    sl2: f64,
    sl3: f64,
    sl4: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
    s6: f64,
    s7: f64,
    ss1: f64,
    ss2: f64,
    ss3: f64,
    ss4: f64,
    ss5: f64,
    ss6: f64,
    ss7: f64,
    sz1: f64,
    sz2: f64,
    sz3: f64,
    sz11: f64,
    sz12: f64,
    sz13: f64,
    sz21: f64,
    sz22: f64,
    sz23: f64,
    sz31: f64,
    sz32: f64,
    sz33: f64,
    xgh2: f64,
    xgh3: f64,
    xgh4: f64,
    xh2: f64,
    xh3: f64,
    xi2: f64,
    xi3: f64,
    xl2: f64,
    xl3: f64,
    xl4: f64,
    nm: f64,
    z1: f64,
    z2: f64,
    z3: f64,
    z11: f64,
    z12: f64,
    z13: f64,
    z21: f64,
    z22: f64,
    z23: f64,
    z31: f64,
    z32: f64,
    z33: f64,
    zmol: f64,
    zmos: f64,
}

#[allow(unused_variables)]
fn dscom(epoch: f64, ep: f64, argpp: f64, tc: f64, inclp: f64, nodep: f64, np: f64) -> DscomResult {
    /* -------------------------- constants ------------------------- */
    let zes: f64 = 0.01675;
    let zel: f64 = 0.05490;
    let c1ss: f64 = 2.9864797e-6;
    let c1l: f64 = 4.7968065e-7;
    let zsinis: f64 = 0.39785416;
    let zcosis: f64 = 0.91744867;
    let zcosgs: f64 = 0.1945905;
    let zsings: f64 = -0.98088458;
    let twopi = 2.0 * PI_VAL;

    /* --------------------- local variables ------------------------ */
    let mut a1: f64 = 0.0;
    let mut a2: f64 = 0.0;
    let mut a3: f64 = 0.0;
    let mut a4: f64 = 0.0;
    let mut a5: f64 = 0.0;
    let mut a6: f64 = 0.0;
    let mut a7: f64 = 0.0;
    let mut a8: f64 = 0.0;
    let mut a9: f64 = 0.0;
    let mut a10: f64 = 0.0;
    let betasq: f64;
    let mut cc: f64;
    let ctem: f64;
    let stem: f64;
    let mut x1: f64 = 0.0;
    let mut x2: f64 = 0.0;
    let mut x3: f64 = 0.0;
    let mut x4: f64 = 0.0;
    let mut x5: f64 = 0.0;
    let mut x6: f64 = 0.0;
    let mut x7: f64 = 0.0;
    let mut x8: f64 = 0.0;
    let xnodce: f64;
    let xnoi: f64;
    let mut zcosg: f64;
    let mut zcosgl: f64 = 0.0;
    let mut zcosh: f64;
    let mut zcoshl: f64;
    let mut zcosi: f64;
    let mut zcosil: f64;
    let mut zsing: f64;
    let mut zsingl: f64 = 0.0;
    let mut zsinh: f64;
    let mut zsinhl: f64;
    let mut zsini: f64;
    let mut zsinil: f64;

    let mut nm = np;
    let mut em = ep;
    let snodm = nodep.sin();
    let cnodm = nodep.cos();
    let sinomm = argpp.sin();
    let cosomm = argpp.cos();
    let sinim = inclp.sin();
    let cosim = inclp.cos();
    let emsq = em * em;
    betasq = 1.0 - emsq;
    let rtemsq = betasq.sqrt();

    /* ----------------- initialize lunar solar terms --------------- */
    let peo: f64 = 0.0;
    let pinco: f64 = 0.0;
    let plo: f64 = 0.0;
    let pgho: f64 = 0.0;
    let pho: f64 = 0.0;
    let day = epoch + 18261.5 + tc / 1440.0;
    xnodce = (4.5236020 - 9.2422029e-4 * day) % twopi;
    stem = xnodce.sin();
    ctem = xnodce.cos();
    zcosil = 0.91375164 - 0.03568096 * ctem;
    zsinil = (1.0 - zcosil * zcosil).sqrt();
    zsinhl = 0.089683511 * stem / zsinil;
    zcoshl = (1.0 - zsinhl * zsinhl).sqrt();
    let gam = 5.8351514 + 0.0019443680 * day;
    let zx = 0.39785416 * stem / zsinil;
    let zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
    let zx = zx.atan2(zy);
    let zx = gam + zx - xnodce;
    zcosgl = zx.cos();
    zsingl = zx.sin();

    /* ------------------------- do solar terms --------------------- */
    zcosg = zcosgs;
    zsing = zsings;
    zcosi = zcosis;
    zsini = zsinis;
    zcosh = cnodm;
    zsinh = snodm;
    cc = c1ss;
    xnoi = 1.0 / nm;

    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut s3: f64 = 0.0;
    let mut s4: f64 = 0.0;
    let mut s5: f64 = 0.0;
    let mut s6: f64 = 0.0;
    let mut s7: f64 = 0.0;
    let mut ss1: f64 = 0.0;
    let mut ss2: f64 = 0.0;
    let mut ss3: f64 = 0.0;
    let mut ss4: f64 = 0.0;
    let mut ss5: f64 = 0.0;
    let mut ss6: f64 = 0.0;
    let mut ss7: f64 = 0.0;
    let mut sz1: f64 = 0.0;
    let mut sz2: f64 = 0.0;
    let mut sz3: f64 = 0.0;
    let mut sz11: f64 = 0.0;
    let mut sz12: f64 = 0.0;
    let mut sz13: f64 = 0.0;
    let mut sz21: f64 = 0.0;
    let mut sz22: f64 = 0.0;
    let mut sz23: f64 = 0.0;
    let mut sz31: f64 = 0.0;
    let mut sz32: f64 = 0.0;
    let mut sz33: f64 = 0.0;
    let mut z1: f64 = 0.0;
    let mut z2: f64 = 0.0;
    let mut z3: f64 = 0.0;
    let mut z11: f64 = 0.0;
    let mut z12: f64 = 0.0;
    let mut z13: f64 = 0.0;
    let mut z21: f64 = 0.0;
    let mut z22: f64 = 0.0;
    let mut z23: f64 = 0.0;
    let mut z31: f64 = 0.0;
    let mut z32: f64 = 0.0;
    let mut z33: f64 = 0.0;

    for lsflg in 1..=2 {
        a1 = zcosg * zcosh + zsing * zcosi * zsinh;
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
        a8 = zsing * zsini;
        a9 = zsing * zsinh + zcosg * zcosi * zcosh;
        a10 = zcosg * zsini;
        a2 = cosim * a7 + sinim * a8;
        a4 = cosim * a9 + sinim * a10;
        a5 = -sinim * a7 + cosim * a8;
        a6 = -sinim * a9 + cosim * a10;

        x1 = a1 * cosomm + a2 * sinomm;
        x2 = a3 * cosomm + a4 * sinomm;
        x3 = -a1 * sinomm + a2 * cosomm;
        x4 = -a3 * sinomm + a4 * cosomm;
        x5 = a5 * sinomm;
        x6 = a6 * sinomm;
        x7 = a5 * cosomm;
        x8 = a6 * cosomm;

        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq;
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq;
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq;
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
        z12 = -6.0 * (a1 * a6 + a3 * a5)
            + emsq * (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5));
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
        z22 = 6.0 * (a4 * a5 + a2 * a6)
            + emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
        z1 = z1 + z1 + betasq * z31;
        z2 = z2 + z2 + betasq * z32;
        z3 = z3 + z3 + betasq * z33;
        s3 = cc * xnoi;
        s2 = -0.5 * s3 / rtemsq;
        s4 = s3 * rtemsq;
        s1 = -15.0 * em * s4;
        s5 = x1 * x3 + x2 * x4;
        s6 = x2 * x3 + x1 * x4;
        s7 = x2 * x4 - x1 * x3;

        /* ----------------------- do lunar terms ------------------- */
        if lsflg == 1 {
            ss1 = s1;
            ss2 = s2;
            ss3 = s3;
            ss4 = s4;
            ss5 = s5;
            ss6 = s6;
            ss7 = s7;
            sz1 = z1;
            sz2 = z2;
            sz3 = z3;
            sz11 = z11;
            sz12 = z12;
            sz13 = z13;
            sz21 = z21;
            sz22 = z22;
            sz23 = z23;
            sz31 = z31;
            sz32 = z32;
            sz33 = z33;
            zcosg = zcosgl;
            zsing = zsingl;
            zcosi = zcosil;
            zsini = zsinil;
            zcosh = zcoshl * cnodm + zsinhl * snodm;
            zsinh = snodm * zcoshl - cnodm * zsinhl;
            cc = c1l;
        }
    }

    let zmol = (4.7199672 + 0.22997150 * day - gam) % twopi;
    let zmos = (6.2565837 + 0.017201977 * day) % twopi;

    /* ------------------------ do solar terms ---------------------- */
    let se2 = 2.0 * ss1 * ss6;
    let se3 = 2.0 * ss1 * ss7;
    let si2 = 2.0 * ss2 * sz12;
    let si3 = 2.0 * ss2 * (sz13 - sz11);
    let sl2 = -2.0 * ss3 * sz2;
    let sl3 = -2.0 * ss3 * (sz3 - sz1);
    let sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes;
    let sgh2 = 2.0 * ss4 * sz32;
    let sgh3 = 2.0 * ss4 * (sz33 - sz31);
    let sgh4 = -18.0 * ss4 * zes;
    let sh2 = -2.0 * ss2 * sz22;
    let sh3 = -2.0 * ss2 * (sz23 - sz21);

    /* ------------------------ do lunar terms ---------------------- */
    let ee2 = 2.0 * s1 * s6;
    let e3 = 2.0 * s1 * s7;
    let xi2 = 2.0 * s2 * z12;
    let xi3 = 2.0 * s2 * (z13 - z11);
    let xl2 = -2.0 * s3 * z2;
    let xl3 = -2.0 * s3 * (z3 - z1);
    let xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel;
    let xgh2 = 2.0 * s4 * z32;
    let xgh3 = 2.0 * s4 * (z33 - z31);
    let xgh4 = -18.0 * s4 * zel;
    let xh2 = -2.0 * s2 * z22;
    let xh3 = -2.0 * s2 * (z23 - z21);

    DscomResult {
        snodm,
        cnodm,
        sinim,
        cosim,
        sinomm,
        cosomm,
        day,
        e3,
        ee2,
        em,
        emsq,
        gam,
        peo,
        pgho,
        pho,
        pinco,
        plo,
        rtemsq,
        se2,
        se3,
        sgh2,
        sgh3,
        sgh4,
        sh2,
        sh3,
        si2,
        si3,
        sl2,
        sl3,
        sl4,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        ss1,
        ss2,
        ss3,
        ss4,
        ss5,
        ss6,
        ss7,
        sz1,
        sz2,
        sz3,
        sz11,
        sz12,
        sz13,
        sz21,
        sz22,
        sz23,
        sz31,
        sz32,
        sz33,
        xgh2,
        xgh3,
        xgh4,
        xh2,
        xh3,
        xi2,
        xi3,
        xl2,
        xl3,
        xl4,
        nm,
        z1,
        z2,
        z3,
        z11,
        z12,
        z13,
        z21,
        z22,
        z23,
        z31,
        z32,
        z33,
        zmol,
        zmos,
    }
}

// ============================================================================
//                           procedure dsinit
// ============================================================================
struct DsinitResult {
    em: f64,
    argpm: f64,
    inclm: f64,
    mm: f64,
    nm: f64,
    nodem: f64,
    irez: i32,
    atime: f64,
    d2201: f64,
    d2211: f64,
    d3210: f64,
    d3222: f64,
    d4410: f64,
    d4422: f64,
    d5220: f64,
    d5232: f64,
    d5421: f64,
    d5433: f64,
    dedt: f64,
    didt: f64,
    dmdt: f64,
    dndt: f64,
    dnodt: f64,
    domdt: f64,
    del1: f64,
    del2: f64,
    del3: f64,
    xfact: f64,
    xlamo: f64,
    xli: f64,
    xni: f64,
}

#[allow(unused_variables)]
fn dsinit(
    xke: f64,
    cosim: f64,
    emsq: f64,
    argpo: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
    sinim: f64,
    ss1: f64,
    ss2: f64,
    ss3: f64,
    ss4: f64,
    ss5: f64,
    sz1: f64,
    sz3: f64,
    sz11: f64,
    sz13: f64,
    sz21: f64,
    sz23: f64,
    sz31: f64,
    sz33: f64,
    t: f64,
    tc: f64,
    gsto: f64,
    mo: f64,
    mdot: f64,
    no: f64,
    nodeo: f64,
    nodedot: f64,
    xpidot: f64,
    z1: f64,
    z3: f64,
    z11: f64,
    z13: f64,
    z21: f64,
    z23: f64,
    z31: f64,
    z33: f64,
    ecco: f64,
    eccsq: f64,
    mut em: f64,
    mut argpm: f64,
    mut inclm: f64,
    mut mm: f64,
    mut nm: f64,
    mut nodem: f64,
) -> DsinitResult {
    /* --------------------- local variables ------------------------ */
    let twopi = 2.0 * PI_VAL;

    let mut ainv2: f64 = 0.0;
    let mut aonv: f64 = 0.0;
    let mut cosisq: f64 = 0.0;
    let mut eoc: f64 = 0.0;
    let mut f220: f64 = 0.0;
    let mut f221: f64 = 0.0;
    let mut f311: f64 = 0.0;
    let mut f321: f64 = 0.0;
    let mut f322: f64 = 0.0;
    let mut f330: f64 = 0.0;
    let mut f441: f64 = 0.0;
    let mut f442: f64 = 0.0;
    let mut f522: f64 = 0.0;
    let mut f523: f64 = 0.0;
    let mut f542: f64 = 0.0;
    let mut f543: f64 = 0.0;
    let mut g200: f64 = 0.0;
    let mut g201: f64 = 0.0;
    let mut g211: f64 = 0.0;
    let mut g300: f64 = 0.0;
    let mut g310: f64 = 0.0;
    let mut g322: f64 = 0.0;
    let mut g410: f64 = 0.0;
    let mut g422: f64 = 0.0;
    let mut g520: f64 = 0.0;
    let mut g521: f64 = 0.0;
    let mut g532: f64 = 0.0;
    let mut g533: f64 = 0.0;
    let mut ses: f64 = 0.0;
    let mut sgs: f64 = 0.0;
    let mut sghl: f64 = 0.0;
    let mut sghs: f64 = 0.0;
    let mut shs: f64 = 0.0;
    let mut shll: f64 = 0.0;
    let mut sis: f64 = 0.0;
    let mut sini2: f64 = 0.0;
    let mut sls: f64 = 0.0;
    let mut temp: f64 = 0.0;
    let mut temp1: f64 = 0.0;
    let mut theta: f64 = 0.0;
    let mut xno2: f64 = 0.0;

    let q22: f64 = 1.7891679e-6;
    let q31: f64 = 2.1460748e-6;
    let q33: f64 = 2.2123015e-7;
    let root22: f64 = 1.7891679e-6;
    let root44: f64 = 7.3636953e-9;
    let root54: f64 = 2.1765803e-9;
    let rptim: f64 = 4.37526908801129966e-3;
    let root32: f64 = 3.7393792e-7;
    let root52: f64 = 1.1428639e-7;
    let x2o3: f64 = 2.0 / 3.0;
    let znl: f64 = 1.5835218e-4;
    let zns: f64 = 1.19459e-5;

    let mut irez: i32;
    let mut d2201: f64 = 0.0;
    let mut d2211: f64 = 0.0;
    let mut d3210: f64 = 0.0;
    let mut d3222: f64 = 0.0;
    let mut d4410: f64 = 0.0;
    let mut d4422: f64 = 0.0;
    let mut d5220: f64 = 0.0;
    let mut d5232: f64 = 0.0;
    let mut d5421: f64 = 0.0;
    let mut d5433: f64 = 0.0;
    let mut dedt: f64 = 0.0;
    let mut didt: f64 = 0.0;
    let mut dmdt: f64 = 0.0;
    let mut dndt: f64 = 0.0;
    let mut dnodt: f64 = 0.0;
    let mut domdt: f64 = 0.0;
    let mut del1: f64 = 0.0;
    let mut del2: f64 = 0.0;
    let mut del3: f64 = 0.0;
    let mut xfact: f64 = 0.0;
    let mut xlamo: f64 = 0.0;
    let mut xli: f64 = 0.0;
    let mut xni: f64 = 0.0;
    let mut atime: f64 = 0.0;

    /* -------------------- deep space initialization ------------ */
    irez = 0;
    if (nm < 0.0052359877) && (nm > 0.0034906585) {
        irez = 1;
    }
    if (nm >= 8.26e-3) && (nm <= 9.24e-3) && (em >= 0.5) {
        irez = 2;
    }

    /* ------------------------ do solar terms ------------------- */
    ses = ss1 * zns * ss5;
    sis = ss2 * zns * (sz11 + sz13);
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq);
    sghs = ss4 * zns * (sz31 + sz33 - 6.0);
    shs = -zns * ss2 * (sz21 + sz23);
    // sgp4fix for 180 deg incl
    if (inclm < 5.2359877e-2) || (inclm > PI_VAL - 5.2359877e-2) {
        shs = 0.0;
    }
    if sinim != 0.0 {
        shs = shs / sinim;
    }
    sgs = sghs - cosim * shs;

    /* ------------------------- do lunar terms ------------------ */
    dedt = ses + s1 * znl * s5;
    didt = sis + s2 * znl * (z11 + z13);
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq);
    sghl = s4 * znl * (z31 + z33 - 6.0);
    shll = -znl * s2 * (z21 + z23);
    // sgp4fix for 180 deg incl
    if (inclm < 5.2359877e-2) || (inclm > PI_VAL - 5.2359877e-2) {
        shll = 0.0;
    }
    domdt = sgs + sghl;
    dnodt = shs;
    if sinim != 0.0 {
        domdt = domdt - cosim / sinim * shll;
        dnodt = dnodt + shll / sinim;
    }

    /* ----------- calculate deep space resonance effects -------- */
    dndt = 0.0;
    theta = (gsto + tc * rptim) % twopi;
    em = em + dedt * t;
    inclm = inclm + didt * t;
    argpm = argpm + domdt * t;
    nodem = nodem + dnodt * t;
    mm = mm + dmdt * t;

    /* -------------- initialize the resonance terms ------------- */
    if irez != 0 {
        aonv = (nm / xke).powf(x2o3);

        /* ---------- geopotential resonance for 12 hour orbits ------ */
        if irez == 2 {
            cosisq = cosim * cosim;
            let emo = em;
            em = ecco;
            let emsqo = emsq;
            let emsq = eccsq;
            eoc = em * emsq;
            g201 = -0.306 - (em - 0.64) * 0.440;

            if em <= 0.65 {
                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq;
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc;
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc;
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc;
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc;
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc;
            } else {
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc;
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc;
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc;
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc;
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc;
                if em > 0.715 {
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc;
                } else {
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq;
                }
            }
            if em < 0.7 {
                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc;
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc;
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc;
            } else {
                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc;
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc;
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc;
            }

            sini2 = sinim * sinim;
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq);
            f221 = 1.5 * sini2;
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq);
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq);
            f441 = 35.0 * sini2 * f220;
            f442 = 39.3750 * sini2 * sini2;
            f522 = 9.84375
                * sinim
                * (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                    + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq));
            f523 = sinim
                * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                    + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq));
            f542 = 29.53125
                * sinim
                * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq));
            f543 = 29.53125
                * sinim
                * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq));
            xno2 = nm * nm;
            ainv2 = aonv * aonv;
            temp1 = 3.0 * xno2 * ainv2;
            temp = temp1 * root22;
            d2201 = temp * f220 * g201;
            d2211 = temp * f221 * g211;
            temp1 = temp1 * aonv;
            temp = temp1 * root32;
            d3210 = temp * f321 * g310;
            d3222 = temp * f322 * g322;
            temp1 = temp1 * aonv;
            temp = 2.0 * temp1 * root44;
            d4410 = temp * f441 * g410;
            d4422 = temp * f442 * g422;
            temp1 = temp1 * aonv;
            temp = temp1 * root52;
            d5220 = temp * f522 * g520;
            d5232 = temp * f523 * g532;
            temp = 2.0 * temp1 * root54;
            d5421 = temp * f542 * g521;
            d5433 = temp * f543 * g533;
            xlamo = (mo + nodeo + nodeo - theta - theta) % twopi;
            xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no;
            em = emo;
            // emsq = emsqo; -- restored via scope in C++, but em is what matters
        }

        /* ---------------- synchronous resonance terms -------------- */
        if irez == 1 {
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq);
            g310 = 1.0 + 2.0 * emsq;
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq);
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim);
            f330 = 1.0 + cosim;
            f330 = 1.875 * f330 * f330 * f330;
            del1 = 3.0 * nm * nm * aonv * aonv;
            del2 = 2.0 * del1 * f220 * g200 * q22;
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv;
            del1 = del1 * f311 * g310 * q31 * aonv;
            xlamo = (mo + nodeo + argpo - theta) % twopi;
            xfact = mdot + xpidot - rptim + dmdt + domdt + dnodt - no;
        }

        /* ------------ for sgp4, initialize the integrator ---------- */
        xli = xlamo;
        xni = no;
        atime = 0.0;
        nm = no + dndt;
    } else {
        // irez == 0
        xli = 0.0;
        xni = 0.0;
        atime = 0.0;
    }

    DsinitResult {
        em,
        argpm,
        inclm,
        mm,
        nm,
        nodem,
        irez,
        atime,
        d2201,
        d2211,
        d3210,
        d3222,
        d4410,
        d4422,
        d5220,
        d5232,
        d5421,
        d5433,
        dedt,
        didt,
        dmdt,
        dndt,
        dnodt,
        domdt,
        del1,
        del2,
        del3,
        xfact,
        xlamo,
        xli,
        xni,
    }
}

// ============================================================================
//                           procedure dspace
// ============================================================================
struct DspaceResult {
    atime: f64,
    em: f64,
    argpm: f64,
    inclm: f64,
    xli: f64,
    mm: f64,
    xni: f64,
    nodem: f64,
    dndt: f64,
    nm: f64,
}

fn dspace(
    irez: i32,
    d2201: f64,
    d2211: f64,
    d3210: f64,
    d3222: f64,
    d4410: f64,
    d4422: f64,
    d5220: f64,
    d5232: f64,
    d5421: f64,
    d5433: f64,
    dedt: f64,
    del1: f64,
    del2: f64,
    del3: f64,
    didt: f64,
    dmdt: f64,
    dnodt: f64,
    domdt: f64,
    argpo: f64,
    argpdot: f64,
    t: f64,
    tc: f64,
    gsto: f64,
    xfact: f64,
    xlamo: f64,
    no: f64,
    mut atime: f64,
    mut em: f64,
    mut argpm: f64,
    mut inclm: f64,
    mut xli: f64,
    mut mm: f64,
    mut xni: f64,
    mut nodem: f64,
    mut nm: f64,
) -> DspaceResult {
    let twopi = 2.0 * PI_VAL;
    let mut iretn: i32;
    let mut ft: f64 = 0.0;
    let theta: f64;
    let mut x2li: f64 = 0.0;
    let mut x2omi: f64 = 0.0;
    let mut xl: f64 = 0.0;
    let mut xldot: f64 = 0.0;
    let mut xnddt: f64 = 0.0;
    let mut xndt: f64 = 0.0;
    let mut xomi: f64 = 0.0;
    let mut delt: f64 = 0.0;
    let mut dndt: f64;

    let fasx2: f64 = 0.13130908;
    let fasx4: f64 = 2.8843198;
    let fasx6: f64 = 0.37448087;
    let g22: f64 = 5.7686396;
    let g32: f64 = 0.95240898;
    let g44: f64 = 1.8014998;
    let g52: f64 = 1.0508330;
    let g54: f64 = 4.4108898;
    let rptim: f64 = 4.37526908801129966e-3;
    let stepp: f64 = 720.0;
    let stepn: f64 = -720.0;
    let step2: f64 = 259200.0;

    /* ----------- calculate deep space resonance effects ----------- */
    dndt = 0.0;
    theta = (gsto + tc * rptim) % twopi;
    em = em + dedt * t;

    inclm = inclm + didt * t;
    argpm = argpm + domdt * t;
    nodem = nodem + dnodt * t;
    mm = mm + dmdt * t;

    /* - update resonances : numerical (euler-maclaurin) integration - */
    /* ------------------------- epoch restart ----------------------  */
    ft = 0.0;
    if irez != 0 {
        // sgp4fix streamline check
        if (atime == 0.0) || (t * atime <= 0.0) || (t.abs() < atime.abs()) {
            atime = 0.0;
            xni = no;
            xli = xlamo;
        }
        // sgp4fix move check outside loop
        if t > 0.0 {
            delt = stepp;
        } else {
            delt = stepn;
        }

        iretn = 381; // added for do loop
        while iretn == 381 {
            /* ------------------- dot terms calculated ------------- */
            /* ----------- near - synchronous resonance terms ------- */
            if irez != 2 {
                xndt = del1 * (xli - fasx2).sin()
                    + del2 * (2.0 * (xli - fasx4)).sin()
                    + del3 * (3.0 * (xli - fasx6)).sin();
                xldot = xni + xfact;
                xnddt = del1 * (xli - fasx2).cos()
                    + 2.0 * del2 * (2.0 * (xli - fasx4)).cos()
                    + 3.0 * del3 * (3.0 * (xli - fasx6)).cos();
                xnddt = xnddt * xldot;
            } else {
                /* --------- near - half-day resonance terms -------- */
                xomi = argpo + argpdot * atime;
                x2omi = xomi + xomi;
                x2li = xli + xli;
                xndt = d2201 * (x2omi + xli - g22).sin()
                    + d2211 * (xli - g22).sin()
                    + d3210 * (xomi + xli - g32).sin()
                    + d3222 * (-xomi + xli - g32).sin()
                    + d4410 * (x2omi + x2li - g44).sin()
                    + d4422 * (x2li - g44).sin()
                    + d5220 * (xomi + xli - g52).sin()
                    + d5232 * (-xomi + xli - g52).sin()
                    + d5421 * (xomi + x2li - g54).sin()
                    + d5433 * (-xomi + x2li - g54).sin();
                xldot = xni + xfact;
                xnddt = d2201 * (x2omi + xli - g22).cos()
                    + d2211 * (xli - g22).cos()
                    + d3210 * (xomi + xli - g32).cos()
                    + d3222 * (-xomi + xli - g32).cos()
                    + d5220 * (xomi + xli - g52).cos()
                    + d5232 * (-xomi + xli - g52).cos()
                    + 2.0
                        * (d4410 * (x2omi + x2li - g44).cos()
                            + d4422 * (x2li - g44).cos()
                            + d5421 * (xomi + x2li - g54).cos()
                            + d5433 * (-xomi + x2li - g54).cos());
                xnddt = xnddt * xldot;
            }

            /* ----------------------- integrator ------------------- */
            if (t - atime).abs() >= stepp {
                iretn = 381;
            } else {
                ft = t - atime;
                iretn = 0;
            }

            if iretn == 381 {
                xli = xli + xldot * delt + xndt * step2;
                xni = xni + xndt * delt + xnddt * step2;
                atime = atime + delt;
            }
        } // while iretn == 381

        nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
        xl = xli + xldot * ft + xndt * ft * ft * 0.5;
        if irez != 1 {
            mm = xl - 2.0 * nodem + 2.0 * theta;
            dndt = nm - no;
        } else {
            mm = xl - nodem - argpm + theta;
            dndt = nm - no;
        }
        nm = no + dndt;
    }

    DspaceResult {
        atime,
        em,
        argpm,
        inclm,
        xli,
        mm,
        xni,
        nodem,
        dndt,
        nm,
    }
}

// ============================================================================
//                           procedure initl
// ============================================================================
struct InitlResult {
    method: char,
    ainv: f64,
    ao: f64,
    con41: f64,
    con42: f64,
    cosio: f64,
    cosio2: f64,
    eccsq: f64,
    omeosq: f64,
    posq: f64,
    rp: f64,
    rteosq: f64,
    sinio: f64,
    gsto: f64,
    no_unkozai: f64,
}

fn initl(
    xke: f64,
    j2: f64,
    ecco: f64,
    epoch: f64,
    inclo: f64,
    no_kozai: f64,
    opsmode: char,
) -> InitlResult {
    /* --------------------- local variables ------------------------ */
    let x2o3 = 2.0 / 3.0;
    let twopi = 2.0 * PI_VAL;

    /* ------------- calculate auxillary epoch quantities ---------- */
    let eccsq = ecco * ecco;
    let omeosq = 1.0 - eccsq;
    let rteosq = omeosq.sqrt();
    let cosio = inclo.cos();
    let cosio2 = cosio * cosio;

    /* ------------------ un-kozai the mean motion ----------------- */
    let ak = (xke / no_kozai).powf(x2o3);
    let d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
    let mut del = d1 / (ak * ak);
    let adel = ak * (1.0 - del * del - del * (1.0 / 3.0 + 134.0 * del * del / 81.0));
    del = d1 / (adel * adel);
    let no_unkozai = no_kozai / (1.0 + del);
    let ao = (xke / (no_unkozai)).powf(x2o3);
    let sinio = inclo.sin();
    let po = ao * omeosq;
    let con42 = 1.0 - 5.0 * cosio2;
    let con41 = -con42 - cosio2 - cosio2;
    let ainv = 1.0 / ao;
    let posq = po * po;
    let rp = ao * (1.0 - ecco);
    let method = 'n';

    // sgp4fix modern approach to finding sidereal time
    //   if (opsmode == 'a')
    //      {
    // sgp4fix use old way of finding gst
    // count integer number of days from 0 jan 1970
    let _ts70 = epoch - 7305.0;
    let _ds70 = (_ts70 + 1.0e-8).floor();
    let _tfrac = _ts70 - _ds70;
    // find greenwich location at epoch
    let _c1 = 1.72027916940703639e-2;
    let _thgr70 = 1.7321343856509374;
    let _fk5r = 5.07551419432269442e-15;
    let _c1p2p = _c1 + twopi;
    let _gsto1 = (_thgr70 + _c1 * _ds70 + _c1p2p * _tfrac + _ts70 * _ts70 * _fk5r) % twopi;
    // if gsto1 < 0.0 { gsto1 = gsto1 + twopi; }
    //    }
    //    else
    let gsto = gstime_SGP4(epoch + 2433281.5);

    InitlResult {
        method,
        ainv,
        ao,
        con41,
        con42,
        cosio,
        cosio2,
        eccsq,
        omeosq,
        posq,
        rp,
        rteosq,
        sinio,
        gsto,
        no_unkozai,
    }
}

// ============================================================================
//                             procedure sgp4init
// ============================================================================
pub fn sgp4init(
    whichconst: GravConstType,
    opsmode: char,
    satn: &str,
    epoch: f64,
    xbstar: f64,
    xndot: f64,
    xnddot: f64,
    xecco: f64,
    xargpo: f64,
    xinclo: f64,
    xmo: f64,
    xno_kozai: f64,
    xnodeo: f64,
    satrec: &mut ElsetRec,
) -> bool {
    /* --------------------- local variables ------------------------ */
    let mut ao: f64 = 0.0;
    let mut ainv: f64 = 0.0;
    let mut con42: f64 = 0.0;
    let mut cosio2: f64 = 0.0;
    let mut eccsq: f64 = 0.0;
    let mut omeosq: f64 = 0.0;
    let mut posq: f64 = 0.0;
    let mut rp: f64 = 0.0;
    let mut rteosq: f64 = 0.0;
    let mut cnodm: f64 = 0.0;
    let mut snodm: f64 = 0.0;
    let mut cosim: f64 = 0.0;
    let mut sinim: f64 = 0.0;
    let mut cosomm: f64 = 0.0;
    let mut sinomm: f64 = 0.0;
    let mut cc1sq: f64 = 0.0;
    let mut cc2: f64 = 0.0;
    let mut coef: f64 = 0.0;
    let mut coef1: f64 = 0.0;
    let mut cosio4: f64 = 0.0;
    let mut day: f64 = 0.0;
    let mut dndt: f64 = 0.0;
    let mut em: f64 = 0.0;
    let mut emsq: f64 = 0.0;
    let mut eeta: f64 = 0.0;
    let mut etasq: f64 = 0.0;
    let mut gam: f64 = 0.0;
    let mut argpm: f64 = 0.0;
    let mut nodem: f64 = 0.0;
    let mut inclm: f64 = 0.0;
    let mut mm: f64 = 0.0;
    let mut nm: f64 = 0.0;
    let mut perige: f64 = 0.0;
    let mut pinvsq: f64 = 0.0;
    let mut psisq: f64 = 0.0;
    let mut qzms24: f64 = 0.0;
    let mut rtemsq: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut s3: f64 = 0.0;
    let mut s4: f64 = 0.0;
    let mut s5: f64 = 0.0;
    let mut s6: f64 = 0.0;
    let mut s7: f64 = 0.0;
    let mut sfour: f64 = 0.0;
    let mut ss1: f64 = 0.0;
    let mut ss2: f64 = 0.0;
    let mut ss3: f64 = 0.0;
    let mut ss4: f64 = 0.0;
    let mut ss5: f64 = 0.0;
    let mut ss6: f64 = 0.0;
    let mut ss7: f64 = 0.0;
    let mut sz1: f64 = 0.0;
    let mut sz2: f64 = 0.0;
    let mut sz3: f64 = 0.0;
    let mut sz11: f64 = 0.0;
    let mut sz12: f64 = 0.0;
    let mut sz13: f64 = 0.0;
    let mut sz21: f64 = 0.0;
    let mut sz22: f64 = 0.0;
    let mut sz23: f64 = 0.0;
    let mut sz31: f64 = 0.0;
    let mut sz32: f64 = 0.0;
    let mut sz33: f64 = 0.0;
    let mut tc: f64 = 0.0;
    let mut temp: f64 = 0.0;
    let mut temp1: f64 = 0.0;
    let mut temp2: f64 = 0.0;
    let mut temp3: f64 = 0.0;
    let mut tsi: f64 = 0.0;
    let mut xpidot: f64 = 0.0;
    let mut xhdot1: f64 = 0.0;
    let mut z1: f64 = 0.0;
    let mut z2: f64 = 0.0;
    let mut z3: f64 = 0.0;
    let mut z11: f64 = 0.0;
    let mut z12: f64 = 0.0;
    let mut z13: f64 = 0.0;
    let mut z21: f64 = 0.0;
    let mut z22: f64 = 0.0;
    let mut z23: f64 = 0.0;
    let mut z31: f64 = 0.0;
    let mut z32: f64 = 0.0;
    let mut z33: f64 = 0.0;
    let mut qzms2t: f64 = 0.0;
    let mut ss: f64 = 0.0;
    let x2o3: f64;
    let mut r: [f64; 3] = [0.0; 3];
    let mut v: [f64; 3] = [0.0; 3];
    let mut delmotemp: f64 = 0.0;
    let mut qzms2ttemp: f64 = 0.0;
    let mut qzms24temp: f64 = 0.0;

    /* ------------------------ initialization --------------------- */
    // sgp4fix divisor for divide by zero check on inclination
    let temp4: f64 = 1.5e-12;

    /* ----------- set all near earth variables to zero ------------ */
    satrec.isimp = 0;
    satrec.method = 'n';
    satrec.aycof = 0.0;
    satrec.con41 = 0.0;
    satrec.cc1 = 0.0;
    satrec.cc4 = 0.0;
    satrec.cc5 = 0.0;
    satrec.d2 = 0.0;
    satrec.d3 = 0.0;
    satrec.d4 = 0.0;
    satrec.delmo = 0.0;
    satrec.eta = 0.0;
    satrec.argpdot = 0.0;
    satrec.omgcof = 0.0;
    satrec.sinmao = 0.0;
    satrec.t = 0.0;
    satrec.t2cof = 0.0;
    satrec.t3cof = 0.0;
    satrec.t4cof = 0.0;
    satrec.t5cof = 0.0;
    satrec.x1mth2 = 0.0;
    satrec.x7thm1 = 0.0;
    satrec.mdot = 0.0;
    satrec.nodedot = 0.0;
    satrec.xlcof = 0.0;
    satrec.xmcof = 0.0;
    satrec.nodecf = 0.0;

    /* ----------- set all deep space variables to zero ------------ */
    satrec.irez = 0;
    satrec.d2201 = 0.0;
    satrec.d2211 = 0.0;
    satrec.d3210 = 0.0;
    satrec.d3222 = 0.0;
    satrec.d4410 = 0.0;
    satrec.d4422 = 0.0;
    satrec.d5220 = 0.0;
    satrec.d5232 = 0.0;
    satrec.d5421 = 0.0;
    satrec.d5433 = 0.0;
    satrec.dedt = 0.0;
    satrec.del1 = 0.0;
    satrec.del2 = 0.0;
    satrec.del3 = 0.0;
    satrec.didt = 0.0;
    satrec.dmdt = 0.0;
    satrec.dnodt = 0.0;
    satrec.domdt = 0.0;
    satrec.e3 = 0.0;
    satrec.ee2 = 0.0;
    satrec.peo = 0.0;
    satrec.pgho = 0.0;
    satrec.pho = 0.0;
    satrec.pinco = 0.0;
    satrec.plo = 0.0;
    satrec.se2 = 0.0;
    satrec.se3 = 0.0;
    satrec.sgh2 = 0.0;
    satrec.sgh3 = 0.0;
    satrec.sgh4 = 0.0;
    satrec.sh2 = 0.0;
    satrec.sh3 = 0.0;
    satrec.si2 = 0.0;
    satrec.si3 = 0.0;
    satrec.sl2 = 0.0;
    satrec.sl3 = 0.0;
    satrec.sl4 = 0.0;
    satrec.gsto = 0.0;
    satrec.xfact = 0.0;
    satrec.xgh2 = 0.0;
    satrec.xgh3 = 0.0;
    satrec.xgh4 = 0.0;
    satrec.xh2 = 0.0;
    satrec.xh3 = 0.0;
    satrec.xi2 = 0.0;
    satrec.xi3 = 0.0;
    satrec.xl2 = 0.0;
    satrec.xl3 = 0.0;
    satrec.xl4 = 0.0;
    satrec.xlamo = 0.0;
    satrec.zmol = 0.0;
    satrec.zmos = 0.0;
    satrec.atime = 0.0;
    satrec.xli = 0.0;
    satrec.xni = 0.0;

    /* ------------------------ earth constants ----------------------- */
    let (tumin, mus, radiusearthkm, xke, j2, j3, j4, j3oj2) = getgravconst(whichconst);
    satrec.tumin = tumin;
    satrec.mus = mus;
    satrec.radiusearthkm = radiusearthkm;
    satrec.xke = xke;
    satrec.j2 = j2;
    satrec.j3 = j3;
    satrec.j4 = j4;
    satrec.j3oj2 = j3oj2;

    satrec.error = 0;
    satrec.operationmode = opsmode;

    // copy satnum
    let satn_bytes = satn.as_bytes();
    let copy_len = satn_bytes.len().min(5);
    satrec.satnum[..copy_len].copy_from_slice(&satn_bytes[..copy_len]);
    satrec.satnum[copy_len] = 0;

    satrec.bstar = xbstar;
    satrec.ndot = xndot;
    satrec.nddot = xnddot;
    satrec.ecco = xecco;
    satrec.argpo = xargpo;
    satrec.inclo = xinclo;
    satrec.mo = xmo;
    satrec.no_kozai = xno_kozai;
    satrec.nodeo = xnodeo;

    // single averaged mean elements
    satrec.am = 0.0;
    satrec.em = 0.0;
    satrec.im = 0.0;
    satrec.Om = 0.0;
    satrec.mm = 0.0;
    satrec.nm = 0.0;

    /* ------------------------ earth constants ----------------------- */
    ss = 78.0 / satrec.radiusearthkm + 1.0;
    // sgp4fix use multiply for speed instead of pow
    qzms2ttemp = (120.0 - 78.0) / satrec.radiusearthkm;
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
    x2o3 = 2.0 / 3.0;

    satrec.init = 'y';
    satrec.t = 0.0;

    // sgp4fix remove satn as it is not needed in initl
    let il = initl(
        satrec.xke,
        satrec.j2,
        satrec.ecco,
        epoch,
        satrec.inclo,
        satrec.no_kozai,
        satrec.operationmode,
    );
    satrec.method = il.method;
    satrec.con41 = il.con41;
    con42 = il.con42;
    let cosio = il.cosio;
    cosio2 = il.cosio2;
    eccsq = il.eccsq;
    omeosq = il.omeosq;
    posq = il.posq;
    rp = il.rp;
    rteosq = il.rteosq;
    let sinio = il.sinio;
    satrec.gsto = il.gsto;
    satrec.no_unkozai = il.no_unkozai;
    ao = il.ao;
    ainv = il.ainv;

    satrec.a = (satrec.no_unkozai * satrec.tumin).powf(-2.0 / 3.0);
    satrec.alta = satrec.a * (1.0 + satrec.ecco) - 1.0;
    satrec.altp = satrec.a * (1.0 - satrec.ecco) - 1.0;
    satrec.error = 0;

    if (omeosq >= 0.0) || (satrec.no_unkozai >= 0.0) {
        satrec.isimp = 0;
        if rp < (220.0 / satrec.radiusearthkm + 1.0) {
            satrec.isimp = 1;
        }
        sfour = ss;
        qzms24 = qzms2t;
        perige = (rp - 1.0) * satrec.radiusearthkm;

        /* - for perigees below 156 km, s and qoms2t are altered - */
        if perige < 156.0 {
            sfour = perige - 78.0;
            if perige < 98.0 {
                sfour = 20.0;
            }
            // sgp4fix use multiply for speed instead of pow
            qzms24temp = (120.0 - sfour) / satrec.radiusearthkm;
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
            sfour = sfour / satrec.radiusearthkm + 1.0;
        }
        pinvsq = 1.0 / posq;

        tsi = 1.0 / (ao - sfour);
        satrec.eta = ao * satrec.ecco * tsi;
        etasq = satrec.eta * satrec.eta;
        eeta = satrec.ecco * satrec.eta;
        psisq = (1.0 - etasq).abs();
        coef = qzms24 * tsi.powf(4.0);
        coef1 = coef / psisq.powf(3.5);
        cc2 = coef1
            * satrec.no_unkozai
            * (ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
                + 0.375 * satrec.j2 * tsi / psisq
                    * satrec.con41
                    * (8.0 + 3.0 * etasq * (8.0 + etasq)));
        satrec.cc1 = satrec.bstar * cc2;
        let cc3 = if satrec.ecco > 1.0e-4 {
            -2.0 * coef * tsi * satrec.j3oj2 * satrec.no_unkozai * sinio / satrec.ecco
        } else {
            0.0
        };
        satrec.x1mth2 = 1.0 - cosio2;
        satrec.cc4 = 2.0
            * satrec.no_unkozai
            * coef1
            * ao
            * omeosq
            * (satrec.eta * (2.0 + 0.5 * etasq) + satrec.ecco * (0.5 + 2.0 * etasq)
                - satrec.j2 * tsi / (ao * psisq)
                    * (-3.0 * satrec.con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                        + 0.75
                            * satrec.x1mth2
                            * (2.0 * etasq - eeta * (1.0 + etasq))
                            * (2.0 * satrec.argpo).cos()));
        satrec.cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);
        cosio4 = cosio2 * cosio2;
        temp1 = 1.5 * satrec.j2 * pinvsq * satrec.no_unkozai;
        temp2 = 0.5 * temp1 * satrec.j2 * pinvsq;
        temp3 = -0.46875 * satrec.j4 * pinvsq * pinvsq * satrec.no_unkozai;
        satrec.mdot = satrec.no_unkozai
            + 0.5 * temp1 * rteosq * satrec.con41
            + 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
        satrec.argpdot = -0.5 * temp1 * con42
            + 0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4)
            + temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4);
        xhdot1 = -temp1 * cosio;
        satrec.nodedot = xhdot1
            + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio;
        xpidot = satrec.argpdot + satrec.nodedot;
        satrec.omgcof = satrec.bstar * cc3 * satrec.argpo.cos();
        satrec.xmcof = 0.0;
        if satrec.ecco > 1.0e-4 {
            satrec.xmcof = -x2o3 * coef * satrec.bstar / eeta;
        }
        satrec.nodecf = 3.5 * omeosq * xhdot1 * satrec.cc1;
        satrec.t2cof = 1.5 * satrec.cc1;
        // sgp4fix for divide by zero with xinco = 180 deg
        if (cosio + 1.0).abs() > 1.5e-12 {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio);
        } else {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4;
        }
        satrec.aycof = -0.5 * satrec.j3oj2 * sinio;
        // sgp4fix use multiply for speed instead of pow
        delmotemp = 1.0 + satrec.eta * satrec.mo.cos();
        satrec.delmo = delmotemp * delmotemp * delmotemp;
        satrec.sinmao = satrec.mo.sin();
        satrec.x7thm1 = 7.0 * cosio2 - 1.0;

        /* --------------- deep space initialization ------------- */
        if (2.0 * PI_VAL / satrec.no_unkozai) >= 225.0 {
            satrec.method = 'd';
            satrec.isimp = 1;
            tc = 0.0;
            inclm = satrec.inclo;

            let dc = dscom(
                epoch,
                satrec.ecco,
                satrec.argpo,
                tc,
                satrec.inclo,
                satrec.nodeo,
                satrec.no_unkozai,
            );
            snodm = dc.snodm;
            cnodm = dc.cnodm;
            sinim = dc.sinim;
            cosim = dc.cosim;
            sinomm = dc.sinomm;
            cosomm = dc.cosomm;
            day = dc.day;
            satrec.e3 = dc.e3;
            satrec.ee2 = dc.ee2;
            em = dc.em;
            emsq = dc.emsq;
            gam = dc.gam;
            satrec.peo = dc.peo;
            satrec.pgho = dc.pgho;
            satrec.pho = dc.pho;
            satrec.pinco = dc.pinco;
            satrec.plo = dc.plo;
            rtemsq = dc.rtemsq;
            satrec.se2 = dc.se2;
            satrec.se3 = dc.se3;
            satrec.sgh2 = dc.sgh2;
            satrec.sgh3 = dc.sgh3;
            satrec.sgh4 = dc.sgh4;
            satrec.sh2 = dc.sh2;
            satrec.sh3 = dc.sh3;
            satrec.si2 = dc.si2;
            satrec.si3 = dc.si3;
            satrec.sl2 = dc.sl2;
            satrec.sl3 = dc.sl3;
            satrec.sl4 = dc.sl4;
            s1 = dc.s1;
            s2 = dc.s2;
            s3 = dc.s3;
            s4 = dc.s4;
            s5 = dc.s5;
            s6 = dc.s6;
            s7 = dc.s7;
            ss1 = dc.ss1;
            ss2 = dc.ss2;
            ss3 = dc.ss3;
            ss4 = dc.ss4;
            ss5 = dc.ss5;
            ss6 = dc.ss6;
            ss7 = dc.ss7;
            sz1 = dc.sz1;
            sz2 = dc.sz2;
            sz3 = dc.sz3;
            sz11 = dc.sz11;
            sz12 = dc.sz12;
            sz13 = dc.sz13;
            sz21 = dc.sz21;
            sz22 = dc.sz22;
            sz23 = dc.sz23;
            sz31 = dc.sz31;
            sz32 = dc.sz32;
            sz33 = dc.sz33;
            satrec.xgh2 = dc.xgh2;
            satrec.xgh3 = dc.xgh3;
            satrec.xgh4 = dc.xgh4;
            satrec.xh2 = dc.xh2;
            satrec.xh3 = dc.xh3;
            satrec.xi2 = dc.xi2;
            satrec.xi3 = dc.xi3;
            satrec.xl2 = dc.xl2;
            satrec.xl3 = dc.xl3;
            satrec.xl4 = dc.xl4;
            nm = dc.nm;
            z1 = dc.z1;
            z2 = dc.z2;
            z3 = dc.z3;
            z11 = dc.z11;
            z12 = dc.z12;
            z13 = dc.z13;
            z21 = dc.z21;
            z22 = dc.z22;
            z23 = dc.z23;
            z31 = dc.z31;
            z32 = dc.z32;
            z33 = dc.z33;
            satrec.zmol = dc.zmol;
            satrec.zmos = dc.zmos;

            dpper(
                satrec.e3,
                satrec.ee2,
                satrec.peo,
                satrec.pgho,
                satrec.pho,
                satrec.pinco,
                satrec.plo,
                satrec.se2,
                satrec.se3,
                satrec.sgh2,
                satrec.sgh3,
                satrec.sgh4,
                satrec.sh2,
                satrec.sh3,
                satrec.si2,
                satrec.si3,
                satrec.sl2,
                satrec.sl3,
                satrec.sl4,
                satrec.t,
                satrec.xgh2,
                satrec.xgh3,
                satrec.xgh4,
                satrec.xh2,
                satrec.xh3,
                satrec.xi2,
                satrec.xi3,
                satrec.xl2,
                satrec.xl3,
                satrec.xl4,
                satrec.zmol,
                satrec.zmos,
                inclm,
                satrec.init,
                &mut satrec.ecco,
                &mut satrec.inclo,
                &mut satrec.nodeo,
                &mut satrec.argpo,
                &mut satrec.mo,
                satrec.operationmode,
            );

            argpm = 0.0;
            nodem = 0.0;
            mm = 0.0;

            let di = dsinit(
                satrec.xke,
                cosim,
                emsq,
                satrec.argpo,
                s1,
                s2,
                s3,
                s4,
                s5,
                sinim,
                ss1,
                ss2,
                ss3,
                ss4,
                ss5,
                sz1,
                sz3,
                sz11,
                sz13,
                sz21,
                sz23,
                sz31,
                sz33,
                satrec.t,
                tc,
                satrec.gsto,
                satrec.mo,
                satrec.mdot,
                satrec.no_unkozai,
                satrec.nodeo,
                satrec.nodedot,
                xpidot,
                z1,
                z3,
                z11,
                z13,
                z21,
                z23,
                z31,
                z33,
                satrec.ecco,
                eccsq,
                em,
                argpm,
                inclm,
                mm,
                nm,
                nodem,
            );
            // em = di.em; -- don't store back, matches C++ (local vars)
            // argpm = di.argpm; inclm = di.inclm; mm = di.mm; nm = di.nm; nodem = di.nodem;
            satrec.irez = di.irez;
            satrec.atime = di.atime;
            satrec.d2201 = di.d2201;
            satrec.d2211 = di.d2211;
            satrec.d3210 = di.d3210;
            satrec.d3222 = di.d3222;
            satrec.d4410 = di.d4410;
            satrec.d4422 = di.d4422;
            satrec.d5220 = di.d5220;
            satrec.d5232 = di.d5232;
            satrec.d5421 = di.d5421;
            satrec.d5433 = di.d5433;
            satrec.dedt = di.dedt;
            satrec.didt = di.didt;
            satrec.dmdt = di.dmdt;
            dndt = di.dndt;
            satrec.dnodt = di.dnodt;
            satrec.domdt = di.domdt;
            satrec.del1 = di.del1;
            satrec.del2 = di.del2;
            satrec.del3 = di.del3;
            satrec.xfact = di.xfact;
            satrec.xlamo = di.xlamo;
            satrec.xli = di.xli;
            satrec.xni = di.xni;
        }

        /* ----------- set variables if not deep space ----------- */
        if satrec.isimp != 1 {
            cc1sq = satrec.cc1 * satrec.cc1;
            satrec.d2 = 4.0 * ao * tsi * cc1sq;
            temp = satrec.d2 * tsi * satrec.cc1 / 3.0;
            satrec.d3 = (17.0 * ao + sfour) * temp;
            satrec.d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * satrec.cc1;
            satrec.t3cof = satrec.d2 + 2.0 * cc1sq;
            satrec.t4cof =
                0.25 * (3.0 * satrec.d3 + satrec.cc1 * (12.0 * satrec.d2 + 10.0 * cc1sq));
            satrec.t5cof = 0.2
                * (3.0 * satrec.d4
                    + 12.0 * satrec.cc1 * satrec.d3
                    + 6.0 * satrec.d2 * satrec.d2
                    + 15.0 * cc1sq * (2.0 * satrec.d2 + cc1sq));
        }
    } // if omeosq >= 0 ...

    /* finally propogate to zero epoch to initialize all others. */
    sgp4(satrec, 0.0, &mut r, &mut v);

    satrec.init = 'n';

    return true;
}

// ============================================================================
//                             procedure sgp4
// ============================================================================
pub fn sgp4(satrec: &mut ElsetRec, tsince: f64, r: &mut [f64; 3], v: &mut [f64; 3]) -> bool {
    let mut am: f64 = 0.0;
    let mut axnl: f64 = 0.0;
    let mut aynl: f64 = 0.0;
    let mut betal: f64 = 0.0;
    let mut cosim: f64 = 0.0;
    let mut cnod: f64 = 0.0;
    let mut cos2u: f64 = 0.0;
    let mut coseo1: f64 = 0.0;
    let mut cosi: f64 = 0.0;
    let mut cosip: f64 = 0.0;
    let mut cosisq: f64 = 0.0;
    let mut cossu: f64 = 0.0;
    let mut cosu: f64 = 0.0;
    let mut delm: f64 = 0.0;
    let mut delomg: f64 = 0.0;
    let mut em: f64 = 0.0;
    let mut emsq: f64 = 0.0;
    let mut ecose: f64 = 0.0;
    let mut el2: f64 = 0.0;
    let mut eo1: f64 = 0.0;
    let mut ep: f64 = 0.0;
    let mut esine: f64 = 0.0;
    let mut argpm: f64 = 0.0;
    let mut argpp: f64 = 0.0;
    let mut argpdf: f64 = 0.0;
    let mut pl: f64 = 0.0;
    let mut mrt: f64 = 0.0;
    let mut mvt: f64 = 0.0;
    let mut rdotl: f64 = 0.0;
    let mut rl: f64 = 0.0;
    let mut rvdot: f64 = 0.0;
    let mut rvdotl: f64 = 0.0;
    let mut sinim: f64 = 0.0;
    let mut sin2u: f64 = 0.0;
    let mut sineo1: f64 = 0.0;
    let mut sini: f64 = 0.0;
    let mut sinip: f64 = 0.0;
    let mut sinsu: f64 = 0.0;
    let mut sinu: f64 = 0.0;
    let mut snod: f64 = 0.0;
    let mut su: f64 = 0.0;
    let mut t2: f64 = 0.0;
    let mut t3: f64 = 0.0;
    let mut t4: f64 = 0.0;
    let mut tem5: f64 = 0.0;
    let mut temp: f64 = 0.0;
    let mut temp1: f64 = 0.0;
    let mut temp2: f64 = 0.0;
    let mut tempa: f64 = 0.0;
    let mut tempe: f64 = 0.0;
    let mut templ: f64 = 0.0;
    let mut u: f64 = 0.0;
    let mut ux: f64 = 0.0;
    let mut uy: f64 = 0.0;
    let mut uz: f64 = 0.0;
    let mut vx: f64 = 0.0;
    let mut vy: f64 = 0.0;
    let mut vz: f64 = 0.0;
    let mut inclm: f64 = 0.0;
    let mut mm: f64 = 0.0;
    let mut nm: f64 = 0.0;
    let mut nodem: f64 = 0.0;
    let mut xinc: f64 = 0.0;
    let mut xincp: f64 = 0.0;
    let mut xl: f64 = 0.0;
    let mut xlm: f64 = 0.0;
    let mut mp: f64 = 0.0;
    let mut xmdf: f64 = 0.0;
    let mut xmx: f64 = 0.0;
    let mut xmy: f64 = 0.0;
    let mut nodedf: f64 = 0.0;
    let mut xnode: f64 = 0.0;
    let mut nodep: f64 = 0.0;
    let mut tc: f64 = 0.0;
    let mut dndt: f64 = 0.0;
    let twopi: f64;
    let x2o3: f64;
    let vkmpersec: f64;
    let mut delmtemp: f64 = 0.0;
    let mut ktr: i32;

    /* ------------------ set mathematical constants --------------- */
    let temp4: f64 = 1.5e-12;
    twopi = 2.0 * PI_VAL;
    x2o3 = 2.0 / 3.0;
    vkmpersec = satrec.radiusearthkm * satrec.xke / 60.0;

    /* --------------------- clear sgp4 error flag ----------------- */
    satrec.t = tsince;
    satrec.error = 0;

    /* ------- update for secular gravity and atmospheric drag ----- */
    xmdf = satrec.mo + satrec.mdot * satrec.t;
    argpdf = satrec.argpo + satrec.argpdot * satrec.t;
    nodedf = satrec.nodeo + satrec.nodedot * satrec.t;
    argpm = argpdf;
    mm = xmdf;
    t2 = satrec.t * satrec.t;
    nodem = nodedf + satrec.nodecf * t2;
    tempa = 1.0 - satrec.cc1 * satrec.t;
    tempe = satrec.bstar * satrec.cc4 * satrec.t;
    templ = satrec.t2cof * t2;

    if satrec.isimp != 1 {
        delomg = satrec.omgcof * satrec.t;
        // sgp4fix use mutliply for speed instead of pow
        delmtemp = 1.0 + satrec.eta * xmdf.cos();
        delm = satrec.xmcof * (delmtemp * delmtemp * delmtemp - satrec.delmo);
        temp = delomg + delm;
        mm = xmdf + temp;
        argpm = argpdf - temp;
        t3 = t2 * satrec.t;
        t4 = t3 * satrec.t;
        tempa = tempa - satrec.d2 * t2 - satrec.d3 * t3 - satrec.d4 * t4;
        tempe = tempe + satrec.bstar * satrec.cc5 * (mm.sin() - satrec.sinmao);
        templ = templ + satrec.t3cof * t3 + t4 * (satrec.t4cof + satrec.t * satrec.t5cof);
    }

    nm = satrec.no_unkozai;
    em = satrec.ecco;
    inclm = satrec.inclo;
    if satrec.method == 'd' {
        tc = satrec.t;
        let ds = dspace(
            satrec.irez,
            satrec.d2201,
            satrec.d2211,
            satrec.d3210,
            satrec.d3222,
            satrec.d4410,
            satrec.d4422,
            satrec.d5220,
            satrec.d5232,
            satrec.d5421,
            satrec.d5433,
            satrec.dedt,
            satrec.del1,
            satrec.del2,
            satrec.del3,
            satrec.didt,
            satrec.dmdt,
            satrec.dnodt,
            satrec.domdt,
            satrec.argpo,
            satrec.argpdot,
            satrec.t,
            tc,
            satrec.gsto,
            satrec.xfact,
            satrec.xlamo,
            satrec.no_unkozai,
            satrec.atime,
            em,
            argpm,
            inclm,
            satrec.xli,
            mm,
            satrec.xni,
            nodem,
            nm,
        );
        satrec.atime = ds.atime;
        em = ds.em;
        argpm = ds.argpm;
        inclm = ds.inclm;
        satrec.xli = ds.xli;
        mm = ds.mm;
        satrec.xni = ds.xni;
        nodem = ds.nodem;
        dndt = ds.dndt;
        nm = ds.nm;
    } // if method = d

    if nm <= 0.0 {
        satrec.error = 2;
        return false;
    }
    am = (satrec.xke / nm).powf(x2o3) * tempa * tempa;
    nm = satrec.xke / am.powf(1.5);
    em = em - tempe;

    // fix tolerance for error recognition
    if (em >= 1.0) || (em < -0.001) {
        satrec.error = 1;
        return false;
    }
    // sgp4fix fix tolerance to avoid a divide by zero
    if em < 1.0e-6 {
        em = 1.0e-6;
    }
    mm = mm + satrec.no_unkozai * templ;
    xlm = mm + argpm + nodem;
    emsq = em * em;
    temp = 1.0 - emsq;

    nodem = nodem % twopi;
    argpm = argpm % twopi;
    xlm = xlm % twopi;
    mm = (xlm - argpm - nodem) % twopi;

    // sgp4fix recover singly averaged mean elements
    satrec.am = am;
    satrec.em = em;
    satrec.im = inclm;
    satrec.Om = nodem;
    satrec.om = argpm;
    satrec.mm = mm;
    satrec.nm = nm;

    /* ----------------- compute extra mean quantities ------------- */
    sinim = inclm.sin();
    cosim = inclm.cos();

    /* -------------------- add lunar-solar periodics -------------- */
    ep = em;
    xincp = inclm;
    argpp = argpm;
    nodep = nodem;
    mp = mm;
    sinip = sinim;
    cosip = cosim;
    if satrec.method == 'd' {
        dpper(
            satrec.e3,
            satrec.ee2,
            satrec.peo,
            satrec.pgho,
            satrec.pho,
            satrec.pinco,
            satrec.plo,
            satrec.se2,
            satrec.se3,
            satrec.sgh2,
            satrec.sgh3,
            satrec.sgh4,
            satrec.sh2,
            satrec.sh3,
            satrec.si2,
            satrec.si3,
            satrec.sl2,
            satrec.sl3,
            satrec.sl4,
            satrec.t,
            satrec.xgh2,
            satrec.xgh3,
            satrec.xgh4,
            satrec.xh2,
            satrec.xh3,
            satrec.xi2,
            satrec.xi3,
            satrec.xl2,
            satrec.xl3,
            satrec.xl4,
            satrec.zmol,
            satrec.zmos,
            satrec.inclo,
            'n',
            &mut ep,
            &mut xincp,
            &mut nodep,
            &mut argpp,
            &mut mp,
            satrec.operationmode,
        );
        if xincp < 0.0 {
            xincp = -xincp;
            nodep = nodep + PI_VAL;
            argpp = argpp - PI_VAL;
        }
        if (ep < 0.0) || (ep > 1.0) {
            satrec.error = 3;
            return false;
        }
    } // if method = d

    /* -------------------- long period periodics ------------------ */
    if satrec.method == 'd' {
        sinip = xincp.sin();
        cosip = xincp.cos();
        satrec.aycof = -0.5 * satrec.j3oj2 * sinip;
        // sgp4fix for divide by zero for xincp = 180 deg
        if (cosip + 1.0).abs() > 1.5e-12 {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
        } else {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4;
        }
    }
    axnl = ep * argpp.cos();
    temp = 1.0 / (am * (1.0 - ep * ep));
    aynl = ep * argpp.sin() + temp * satrec.aycof;
    xl = mp + argpp + nodep + temp * satrec.xlcof * axnl;

    /* --------------------- solve kepler's equation --------------- */
    u = (xl - nodep) % twopi;
    eo1 = u;
    tem5 = 9999.9;
    ktr = 1;
    //   sgp4fix for kepler iteration
    //   the following iteration needs better limits on corrections
    while (tem5.abs() >= 1.0e-12) && (ktr <= 10) {
        sineo1 = eo1.sin();
        coseo1 = eo1.cos();
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
        if tem5.abs() >= 0.95 {
            tem5 = if tem5 > 0.0 { 0.95 } else { -0.95 };
        }
        eo1 = eo1 + tem5;
        ktr = ktr + 1;
    }

    /* ------------- short period preliminary quantities ----------- */
    ecose = axnl * coseo1 + aynl * sineo1;
    esine = axnl * sineo1 - aynl * coseo1;
    el2 = axnl * axnl + aynl * aynl;
    pl = am * (1.0 - el2);
    if pl < 0.0 {
        satrec.error = 4;
        return false;
    } else {
        rl = am * (1.0 - ecose);
        rdotl = am.sqrt() * esine / rl;
        rvdotl = pl.sqrt() / rl;
        betal = (1.0 - el2).sqrt();
        temp = esine / (1.0 + betal);
        sinu = am / rl * (sineo1 - aynl - axnl * temp);
        cosu = am / rl * (coseo1 - axnl + aynl * temp);
        su = sinu.atan2(cosu);
        sin2u = (cosu + cosu) * sinu;
        cos2u = 1.0 - 2.0 * sinu * sinu;
        temp = 1.0 / pl;
        temp1 = 0.5 * satrec.j2 * temp;
        temp2 = temp1 * temp;

        /* -------------- update for short period periodics ------------ */
        if satrec.method == 'd' {
            cosisq = cosip * cosip;
            satrec.con41 = 3.0 * cosisq - 1.0;
            satrec.x1mth2 = 1.0 - cosisq;
            satrec.x7thm1 = 7.0 * cosisq - 1.0;
        }
        mrt = rl * (1.0 - 1.5 * temp2 * betal * satrec.con41) + 0.5 * temp1 * satrec.x1mth2 * cos2u;
        su = su - 0.25 * temp2 * satrec.x7thm1 * sin2u;
        xnode = nodep + 1.5 * temp2 * cosip * sin2u;
        xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
        mvt = rdotl - nm * temp1 * satrec.x1mth2 * sin2u / satrec.xke;
        rvdot = rvdotl + nm * temp1 * (satrec.x1mth2 * cos2u + 1.5 * satrec.con41) / satrec.xke;

        /* --------------------- orientation vectors ------------------- */
        sinsu = su.sin();
        cossu = su.cos();
        snod = xnode.sin();
        cnod = xnode.cos();
        sini = xinc.sin();
        cosi = xinc.cos();
        xmx = -snod * cosi;
        xmy = cnod * cosi;
        ux = xmx * sinsu + cnod * cossu;
        uy = xmy * sinsu + snod * cossu;
        uz = sini * sinsu;
        vx = xmx * cossu - cnod * sinsu;
        vy = xmy * cossu - snod * sinsu;
        vz = sini * cossu;

        /* --------- position and velocity (in km and km/sec) ---------- */
        r[0] = (mrt * ux) * satrec.radiusearthkm;
        r[1] = (mrt * uy) * satrec.radiusearthkm;
        r[2] = (mrt * uz) * satrec.radiusearthkm;
        v[0] = (mvt * ux + rvdot * vx) * vkmpersec;
        v[1] = (mvt * uy + rvdot * vy) * vkmpersec;
        v[2] = (mvt * uz + rvdot * vz) * vkmpersec;
    } // if pl > 0

    // sgp4fix for decaying satellites
    if mrt < 1.0 {
        satrec.error = 6;
        return false;
    }

    return true;
}

// ============================================================================
//  High-level helper: initialize + propagate (matches the C++ wrapper)
// ============================================================================
/// Initialize an ElsetRec from TLE elements and propagate to a given JD.
/// Returns (position_km, velocity_km_s) or an error code.
pub fn sgp4_propagate(
    catalog_number: &str,
    epochyr: i32,
    epochdays: f64,
    bstar: f64,
    ndot: f64,
    nddot: f64,
    ecco: f64,
    argpo_deg: f64,
    inclo_deg: f64,
    mo_deg: f64,
    no_kozai_revday: f64, // rev/day
    nodeo_deg: f64,
    jd_whole: f64,
    jd_fraction: f64,
) -> Result<([f64; 3], [f64; 3]), i32> {
    let whichconst = GravConstType::Wgs72;
    let opsmode = 'i';

    let mut satrec = ElsetRec::default();

    // Compute epoch JD matching Python's twoline2rv:
    let year_full = if epochyr < 57 {
        epochyr + 2000
    } else {
        epochyr + 1900
    };

    // JD of Jan 1.0 (midnight) of epoch year
    let (jan1_jd, _jan1_frac) = jday_SGP4(year_full, 1, 1, 0, 0, 0.0);

    // Python twoline2rv: satrec.jdsatepoch = jan1_jd + floor(epochdays) - 1
    //                    satrec.jdsatepochF = epochdays - floor(epochdays)
    let epoch_whole_days = epochdays.floor();
    let epoch_frac = epochdays - epoch_whole_days;

    let jd = jan1_jd + epoch_whole_days - 1.0; // -1 because day 1 = Jan 1
    let jdfrac = epoch_frac;

    let epoch_sgp4 = jd + jdfrac - 2433281.5;

    satrec.epochyr = epochyr;
    satrec.epochdays = epochdays;
    satrec.jdsatepoch = jd;
    satrec.jdsatepochF = jdfrac;

    // Convert angles to radians
    let deg2rad = PI_VAL / 180.0;
    let inclo = inclo_deg * deg2rad;
    let nodeo = nodeo_deg * deg2rad;
    let argpo = argpo_deg * deg2rad;
    let mo = mo_deg * deg2rad;

    // Convert mean motion to rad/minute
    let xpdotp = 1440.0 / (2.0 * PI_VAL);
    let no_kozai_rad = no_kozai_revday / xpdotp;

    sgp4init(
        whichconst,
        opsmode,
        catalog_number,
        epoch_sgp4,
        bstar,
        ndot,
        nddot,
        ecco,
        argpo,
        inclo,
        mo,
        no_kozai_rad,
        nodeo,
        &mut satrec,
    );

    if satrec.error != 0 {
        return Err(satrec.error);
    }

    // Restore the split epoch (sgp4init may have overwritten via initl)
    satrec.epochyr = epochyr;
    satrec.epochdays = epochdays;
    satrec.jdsatepoch = jd;
    satrec.jdsatepochF = jdfrac;

    // Split tsince matching Python sgp4's sgp4(jd, fr) method
    let tsince =
        (jd_whole - satrec.jdsatepoch) * 1440.0 + (jd_fraction - satrec.jdsatepochF) * 1440.0;

    let mut r = [0.0_f64; 3];
    let mut v = [0.0_f64; 3];

    sgp4(&mut satrec, tsince, &mut r, &mut v);

    if satrec.error != 0 {
        return Err(satrec.error);
    }

    Ok((r, v))
}

// ============================================================================
/// Parse raw TLE lines and propagate, exactly matching Python's twoline2rv.
/// This is the only way to achieve 0 ULP parity - the TLE parsing must use
/// the same string slicing and float conversion as Python's sgp4.
pub fn twoline2rv_propagate(
    longstr1: &str,
    longstr2: &str,
    jd_whole: f64,
    jd_fraction: f64,
) -> Result<([f64; 3], [f64; 3]), String> {
    let line1 = longstr1.trim_end();
    let line2 = longstr2.trim_end();

    if line1.len() < 64 || line2.len() < 68 {
        return Err("TLE lines too short".into());
    }

    let deg2rad = PI_VAL / 180.0;
    let xpdotp = 1440.0 / (2.0 * PI_VAL);

    // Parse line 1
    let two_digit_year: i32 = line1[18..20].trim().parse().map_err(|_| "bad epochyr")?;
    let epochdays: f64 = line1[20..32].trim().parse().map_err(|_| "bad epochdays")?;

    // ndot: read as float from positions 33-43
    let ndot_raw: f64 = line1[33..43].trim().parse().map_err(|_| "bad ndot")?;

    // nddot: Python does float(line[44] + '.' + line[45:50])
    // For space sign: float(' .00000') = 0.0
    let nddot_str = format!("{}.", &line1[44..45]);
    let nddot_str = format!("{}{}", nddot_str, &line1[45..50]);
    let nddot_mantissa: f64 = nddot_str.trim().parse().unwrap_or(0.0);
    let nexp: i32 = line1[50..52].trim().parse().unwrap_or(0);

    // bstar: Python does float(line[53] + '.' + line[54:59])
    let bstar_str = format!("{}.", &line1[53..54]);
    let bstar_str = format!("{}{}", bstar_str, &line1[54..59]);
    let bstar_mantissa: f64 = bstar_str.trim().parse().unwrap_or(0.0);
    let ibexp: i32 = line1[59..61].trim().parse().unwrap_or(0);

    // Parse line 2
    let inclo_deg: f64 = line2[8..16].trim().parse().map_err(|_| "bad inclo")?;
    let nodeo_deg: f64 = line2[17..25].trim().parse().map_err(|_| "bad nodeo")?;
    let ecco_str = format!("0.{}", line2[26..33].replace(' ', "0"));
    let ecco: f64 = ecco_str.parse().map_err(|_| "bad ecco")?;
    let argpo_deg: f64 = line2[34..42].trim().parse().map_err(|_| "bad argpo")?;
    let mo_deg: f64 = line2[43..51].trim().parse().map_err(|_| "bad mo")?;
    let no_kozai_revday: f64 = line2[52..63].trim().parse().map_err(|_| "bad no_kozai")?;

    // Convert to SGP4 units (EXACTLY matching Python's twoline2rv)
    let no_kozai = no_kozai_revday / xpdotp; // rad/min
    let nddot = nddot_mantissa * 10.0_f64.powi(nexp);
    let bstar = bstar_mantissa * 10.0_f64.powi(ibexp);

    // Convert ndot: Python does ndot / (xpdotp * 1440.0)
    let ndot = ndot_raw / (xpdotp * 1440.0);
    // Convert nddot: Python does nddot / (xpdotp * 1440.0 * 1440)
    let nddot = nddot / (xpdotp * 1440.0 * 1440.0);

    // Convert angles to radians
    let inclo = inclo_deg * deg2rad;
    let nodeo = nodeo_deg * deg2rad;
    let argpo = argpo_deg * deg2rad;
    let mo = mo_deg * deg2rad;

    // Epoch JD: compute via days2mdhms + jday, then round jdsatepochF
    // to 8 decimal places matching Python sgp4 v2.20+ behavior.
    // This rounding is applied when epochdays has ≤8 decimal digits,
    // which is always true for standard TLE format.
    let year_full = if two_digit_year < 57 {
        two_digit_year + 2000
    } else {
        two_digit_year + 1900
    };
    let (mon, day, hr, minute, sec) = days2mdhms_SGP4(year_full, epochdays);
    let (jd, jdfrac_raw) = jday_SGP4(year_full, mon, day, hr, minute, sec);
    // Round to 8 decimal places: matches Python's sgp4init epoch rounding
    let jdfrac = (jdfrac_raw * 100000000.0).round() / 100000000.0;
    let epoch_sgp4 = jd + jdfrac - 2433281.5;

    let satnum = line1[2..7].trim();

    let mut satrec = ElsetRec::default();
    satrec.epochyr = two_digit_year;
    satrec.epochdays = epochdays;
    satrec.jdsatepoch = jd;
    satrec.jdsatepochF = jdfrac;

    sgp4init(
        GravConstType::Wgs72,
        'i',
        satnum,
        epoch_sgp4,
        bstar,
        ndot,
        nddot,
        ecco,
        argpo,
        inclo,
        mo,
        no_kozai,
        nodeo,
        &mut satrec,
    );

    if satrec.error != 0 {
        return Err(format!("sgp4init error {}", satrec.error));
    }

    satrec.jdsatepoch = jd;
    satrec.jdsatepochF = jdfrac;

    let tsince =
        (jd_whole - satrec.jdsatepoch) * 1440.0 + (jd_fraction - satrec.jdsatepochF) * 1440.0;

    let mut r = [0.0_f64; 3];
    let mut v = [0.0_f64; 3];
    sgp4(&mut satrec, tsince, &mut r, &mut v);

    if satrec.error != 0 {
        return Err(format!("sgp4 propagation error {}", satrec.error));
    }

    Ok((r, v))
}
