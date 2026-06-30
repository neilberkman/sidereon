//! NRLMSISE-00 empirical neutral-atmosphere model (Picone et al., 2002).
//!
//! This is the full NRLMSISE-00 model: all eight species (He, O, N2, O2, Ar,
//! H, N, and anomalous oxygen), total mass density, and exospheric plus
//! altitude temperature, from the surface to the lower exosphere (~1000 km),
//! as a function of position, time, and solar/geomagnetic activity. It is the
//! standard model used for satellite-drag prediction.
//!
//! `gtd7` returns the total mass density excluding anomalous oxygen; `gtd7d`
//! returns the effective total mass density for drag, folding anomalous oxygen
//! into `d[5]` (relevant above ~500 km). The supporting call tree (`gts7`,
//! `globe7`/`glob7s`, `densu`/`densm`, `ccor`/`ccor2`, `scalh`, `dnet`,
//! `spline`/`splint`/`splini`, `zeta`) mirrors the reference.
//!
//! Provenance and license: the model was developed by Mike Picone, Alan Hedin,
//! and Doug Drob at the US Naval Research Laboratory and is in the public
//! domain. The model logic and the full coefficient tables in [`tables`] were
//! transcribed verbatim from Dominik Brodowski's public-domain C port
//! (`nrlmsise-00.c` / `nrlmsise-00_data.c`, release 20041227,
//! <https://www.brodo.de/space/nrlmsise/>). The coefficient tables are
//! reproduced bit-for-bit; the implementation reproduces the reference's own
//! constants (its `re`/`gsurf` datum, gas constant, and species masses) rather
//! than substituting this crate's WGS84/GM values, so output matches the
//! reference oracle. Reference: Picone, J.M., Hedin, A.E., Drob, D.P., and
//! Aikin, A.C., "NRLMSISE-00 empirical model of the atmosphere", J. Geophys.
//! Res., 107(A12), 1468, 2002.

// The model is a verbatim numeric port: many loops index parallel coefficient
// and term arrays by position (`t[i]` against `sw[i+1]`), which is clearer than
// an iterator and preserves the reference structure.
#![allow(clippy::needless_range_loop)]

mod tables;

/// Magnetic-activity history (the `ap_a` array), used when switch 9 is -1.
///
/// Elements: daily AP; 3 hr AP for current time; 3 hr AP at -3/-6/-9 hr;
/// average of eight 3 hr indices from 12 to 33 hr prior; average of eight from
/// 36 to 57 hr prior.
pub type ApArray = [f64; 7];

/// Documented upper bound of the NRLMSISE-00 altitude domain (km).
///
/// The model spans the surface to the lower exosphere; above this the profile
/// is an extrapolation rather than a fit, so the API rejects it.
pub const MAX_ALTITUDE_KM: f64 = 1000.0;

/// Error returned when an evaluation cannot be performed for the requested
/// inputs or switch configuration.
///
/// The model itself is total: it will emit a number for any input. These
/// variants guard the API boundary so a misconfigured call surfaces a typed
/// failure instead of a plausible-but-wrong or non-finite result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AtmosphereError {
    /// Switch 9 is set to -1 (Ap-history mode) but [`NrlmsiseInput::ap_array`]
    /// was not supplied. There is no defensible default Ap history, so the call
    /// is rejected rather than silently substituting zeros.
    #[error("ap_array is required when switch 9 is -1 (Ap-history mode)")]
    MissingApArray,
    /// A model input was not finite (NaN or infinity). The offending field name
    /// is carried for diagnostics.
    #[error("non-finite input: {0}")]
    NonFiniteInput(&'static str),
    /// A model input was outside the documented valid domain (altitude in
    /// `[0, MAX_ALTITUDE_KM]`; `f107`, `f107a`, `ap`, and every `ap_array`
    /// element finite and non-negative). The offending field name is carried.
    #[error("input out of domain: {0}")]
    OutOfDomain(&'static str),
}

/// Validate inputs and switch configuration at the API boundary.
///
/// Policy: reject Ap-history mode without an `ap_array`; reject any non-finite
/// numeric input; require altitude in `[0, MAX_ALTITUDE_KM]` and the solar and
/// geomagnetic indices (`f107`, `f107a`, `ap`, and any `ap_array` element) to
/// be finite and non-negative. Latitude, longitude, time, and local solar time
/// are only checked for finiteness; longitude keeps its `<= -1000` sentinel
/// (longitude variation off), which the finite check still admits.
fn validate(input: &NrlmsiseInput, flags: &Flags) -> Result<(), AtmosphereError> {
    use AtmosphereError::*;

    if flags.switches[9] == -1 && input.ap_array.is_none() {
        return Err(MissingApArray);
    }

    for (value, name) in [
        (input.alt, "alt"),
        (input.g_lat, "g_lat"),
        (input.g_long, "g_long"),
        (input.sec, "sec"),
        (input.lst, "lst"),
        (input.f107, "f107"),
        (input.f107a, "f107a"),
        (input.ap, "ap"),
    ] {
        if !value.is_finite() {
            return Err(NonFiniteInput(name));
        }
    }

    if !(0.0..=MAX_ALTITUDE_KM).contains(&input.alt) {
        return Err(OutOfDomain("alt"));
    }
    if input.f107 < 0.0 {
        return Err(OutOfDomain("f107"));
    }
    if input.f107a < 0.0 {
        return Err(OutOfDomain("f107a"));
    }
    if input.ap < 0.0 {
        return Err(OutOfDomain("ap"));
    }
    if let Some(ap_array) = input.ap_array {
        for v in ap_array {
            if !v.is_finite() {
                return Err(NonFiniteInput("ap_array"));
            }
            if v < 0.0 {
                return Err(OutOfDomain("ap_array"));
            }
        }
    }

    Ok(())
}

/// Inputs to the neutral-atmosphere evaluation (mirrors the reference
/// `nrlmsise_input`).
#[derive(Debug, Clone, Copy)]
pub struct NrlmsiseInput {
    /// Year (ignored by the model; kept for API stability).
    pub year: i32,
    /// Day of year (1-366).
    pub doy: i32,
    /// Seconds in day (UT).
    pub sec: f64,
    /// Geodetic altitude (km).
    pub alt: f64,
    /// Geodetic latitude (deg).
    pub g_lat: f64,
    /// Geodetic longitude (deg).
    pub g_long: f64,
    /// Local apparent solar time (hours); for consistency use
    /// `sec/3600 + g_long/15`.
    pub lst: f64,
    /// 81-day average of F10.7 flux (centered on `doy`).
    pub f107a: f64,
    /// Daily F10.7 flux for the previous day.
    pub f107: f64,
    /// Daily magnetic Ap index.
    pub ap: f64,
    /// Optional Ap history; required when switch 9 is set to -1.
    pub ap_array: Option<ApArray>,
}

/// Outputs of the neutral-atmosphere evaluation (mirrors the reference
/// `nrlmsise_output`).
///
/// `d` holds number densities (He, O, N2, O2, Ar, _total_, H, N, anomalous O);
/// index 5 is the total mass density. `t[0]` is exospheric temperature, `t[1]`
/// is temperature at altitude. Units follow the switch-0 flag: with metric
/// output, densities are m^-3, total mass density is kg/m^3; otherwise cm^-3
/// and g/cm^3.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NrlmsiseOutput {
    /// Densities: He, O, N2, O2, Ar, total mass density, H, N, anomalous O.
    pub d: [f64; 9],
    /// Temperatures: exospheric, at altitude.
    pub t: [f64; 2],
}

impl NrlmsiseOutput {
    /// Total mass density (`d[5]`).
    pub fn density(&self) -> f64 {
        self.d[5]
    }
    /// Exospheric temperature (`t[0]`).
    pub fn temperature_exo(&self) -> f64 {
        self.t[0]
    }
    /// Temperature at the requested altitude (`t[1]`).
    pub fn temperature_alt(&self) -> f64 {
        self.t[1]
    }
}

/// Variation switches (mirrors the reference `nrlmsise_flags`).
///
/// `switches[0]` selects metric output (m/kg) when nonzero; entries 1..=23
/// enable individual variations (0 off, 1 on, 2 main effects off but cross
/// terms on). Entry 9 set to -1 selects Ap-history mode and requires
/// [`NrlmsiseInput::ap_array`].
#[derive(Debug, Clone, Copy)]
pub struct Flags {
    /// Caller-set variation switches.
    pub switches: [i32; 24],
    sw: [f64; 24],
    swc: [f64; 24],
}

impl Flags {
    /// Build flags from a raw switch vector, deriving the internal `sw`/`swc`.
    pub fn new(switches: [i32; 24]) -> Self {
        let mut f = Flags {
            switches,
            sw: [0.0; 24],
            swc: [0.0; 24],
        };
        f.tselec();
        f
    }

    /// Reference standard flags: CGS output, all variations on (the
    /// configuration of the reference test program).
    pub fn standard() -> Self {
        let mut s = [1i32; 24];
        s[0] = 0;
        Flags::new(s)
    }

    /// All variations on with metric (m/kg) output.
    pub fn metric() -> Self {
        Flags::new([1i32; 24])
    }

    /// Derive `sw`/`swc` from `switches` (reference `tselec`).
    fn tselec(&mut self) {
        for i in 0..24 {
            if i != 9 {
                self.sw[i] = if self.switches[i] == 1 { 1.0 } else { 0.0 };
                self.swc[i] = if self.switches[i] > 0 { 1.0 } else { 0.0 };
            } else {
                self.sw[i] = self.switches[i] as f64;
                self.swc[i] = self.switches[i] as f64;
            }
        }
    }
}

/// Local apparent solar time (hours) from UT seconds and longitude (deg).
pub fn local_solar_time(sec: f64, g_long: f64) -> f64 {
    let lst = sec / 3600.0 + g_long / 15.0;
    ((lst % 24.0) + 24.0) % 24.0
}

/// Default daily F10.7 solar-radio flux (previous day), solar flux units.
///
/// This is the NRLMSISE-00 reference test-program standard input (release
/// 20041227) used throughout the oracle cases. Bindings offering a sane default
/// (rather than requiring live space-weather input) read this.
pub const DEFAULT_F107: f64 = 150.0;

/// Default 81-day centred average F10.7 solar-radio flux, solar flux units.
///
/// This is also the model's own reference baseline: the `globe7` flux term is
/// centred on `f107a - 150.0`, so 150 is the neutral value. Matches the
/// NRLMSISE-00 reference test-program standard input.
pub const DEFAULT_F107A: f64 = 150.0;

/// Default daily magnetic Ap index (dimensionless).
///
/// NRLMSISE-00 reference test-program standard input (release 20041227),
/// representing quiet geomagnetic conditions.
pub const DEFAULT_AP: f64 = 4.0;

const DGTR: f64 = 1.74533E-2;
const DR: f64 = 1.72142E-2;
const HR: f64 = 0.2618;
const SR: f64 = 7.2722E-5;
const RGAS: f64 = 831.4;

/// Chemistry/dissociation correction (reference `ccor`).
fn ccor(alt: f64, r: f64, h1: f64, zh: f64) -> f64 {
    let e = (alt - zh) / h1;
    if e > 70.0 {
        return (0.0_f64).exp();
    }
    if e < -70.0 {
        return r.exp();
    }
    let ex = e.exp();
    let e = r / (1.0 + ex);
    e.exp()
}

/// Chemistry/dissociation correction with two scale lengths (reference
/// `ccor2`).
fn ccor2(alt: f64, r: f64, h1: f64, zh: f64, h2: f64) -> f64 {
    let e1 = (alt - zh) / h1;
    let e2 = (alt - zh) / h2;
    if (e1 > 70.0) || (e2 > 70.0) {
        return (0.0_f64).exp();
    }
    if (e1 < -70.0) && (e2 < -70.0) {
        return r.exp();
    }
    let ex1 = e1.exp();
    let ex2 = e2.exp();
    let ccor2v = r / (1.0 + 0.5 * (ex1 + ex2));
    ccor2v.exp()
}

/// Turbopause density blend (reference `dnet`).
fn dnet(dd: f64, dm: f64, zhm: f64, xmm: f64, xm: f64) -> f64 {
    let a = zhm / (xmm - xm);
    let (mut dd, dm) = (dd, dm);
    if !((dm > 0.0) && (dd > 0.0)) {
        if (dd == 0.0) && (dm == 0.0) {
            dd = 1.0;
        }
        if dm == 0.0 {
            return dd;
        }
        if dd == 0.0 {
            return dm;
        }
    }
    let ylog = a * (dm / dd).ln();
    if ylog < -10.0 {
        return dd;
    }
    if ylog > 10.0 {
        return dm;
    }
    dd * (1.0 + ylog.exp()).powf(1.0 / a)
}

/// Integrate a cubic spline from `xa[0]` to `x` (reference `splini`).
fn splini(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut yi = 0.0;
    let mut klo = 0usize;
    let mut khi = 1usize;
    while (x > xa[klo]) && (khi < n) {
        let mut xx = x;
        if khi < (n - 1) {
            if x < xa[khi] {
                xx = x;
            } else {
                xx = xa[khi];
            }
        }
        let h = xa[khi] - xa[klo];
        let a = (xa[khi] - xx) / h;
        let b = (xx - xa[klo]) / h;
        let a2 = a * a;
        let b2 = b * b;
        yi += ((1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + ((-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi])
                * h
                * h
                / 6.0)
            * h;
        klo += 1;
        khi += 1;
    }
    yi
}

/// Cubic-spline interpolation (reference `splint`).
fn splint(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut klo = 0usize;
    let mut khi = n - 1;
    while (khi - klo) > 1 {
        let k = (khi + klo) / 2;
        if xa[k] > x {
            khi = k;
        } else {
            klo = k;
        }
    }
    let h = xa[khi] - xa[klo];
    let a = (xa[khi] - x) / h;
    let b = (x - xa[klo]) / h;
    a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0
}

/// Second derivatives of a cubic-spline interpolant (reference `spline`).
fn spline(x: &[f64], y: &[f64], n: usize, yp1: f64, ypn: f64, y2: &mut [f64]) {
    let mut u = [0.0_f64; 10];
    if yp1 > 0.99E30 {
        y2[0] = 0.0;
        u[0] = 0.0;
    } else {
        y2[0] = -0.5;
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
    }
    for i in 1..(n - 1) {
        let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (6.0
            * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1])
            / p;
    }
    let (qn, un) = if ypn > 0.99E30 {
        (0.0, 0.0)
    } else {
        (
            0.5,
            (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])),
        )
    };
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
    for k in (0..=(n - 2)).rev() {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
}

/// 3 hr magnetic-activity term (reference `g0`).
fn g0(a: f64, p: &[f64]) -> f64 {
    a - 4.0
        + (p[25] - 1.0)
            * (a - 4.0
                + ((-(p[24] * p[24]).sqrt() * (a - 4.0)).exp() - 1.0) / (p[24] * p[24]).sqrt())
}

/// 3 hr magnetic-activity sum normalizer (reference `sumex`).
fn sumex(ex: f64) -> f64 {
    1.0 + (1.0 - ex.powf(19.0)) / (1.0 - ex) * ex.powf(0.5)
}

/// 3 hr magnetic-activity weighted sum (reference `sg0`).
fn sg0(ex: f64, p: &[f64], ap: &[f64]) -> f64 {
    (g0(ap[1], p)
        + (g0(ap[2], p) * ex
            + g0(ap[3], p) * ex * ex
            + g0(ap[4], p) * ex.powf(3.0)
            + (g0(ap[5], p) * ex.powf(4.0) + g0(ap[6], p) * ex.powf(12.0)) * (1.0 - ex.powf(8.0))
                / (1.0 - ex)))
        / sumex(ex)
}

/// Working state for one evaluation: the reference's shared (static) variables,
/// scoped to a single call tree instead of file-global mutable state.
#[derive(Clone)]
struct State {
    gsurf: f64,
    re: f64,
    dd: f64,
    dm04: f64,
    dm16: f64,
    dm28: f64,
    dm32: f64,
    dm40: f64,
    dm01: f64,
    dm14: f64,
    meso_tn1: [f64; 5],
    meso_tn2: [f64; 4],
    meso_tn3: [f64; 5],
    meso_tgn1: [f64; 2],
    meso_tgn2: [f64; 2],
    meso_tgn3: [f64; 2],
    dfa: f64,
    plg: [[f64; 9]; 4],
    ctloc: f64,
    stloc: f64,
    c2tloc: f64,
    s2tloc: f64,
    s3tloc: f64,
    c3tloc: f64,
    apdf: f64,
    apt: [f64; 4],
}

impl State {
    fn new() -> Self {
        State {
            gsurf: 0.0,
            re: 0.0,
            dd: 0.0,
            dm04: 0.0,
            dm16: 0.0,
            dm28: 0.0,
            dm32: 0.0,
            dm40: 0.0,
            dm01: 0.0,
            dm14: 0.0,
            meso_tn1: [0.0; 5],
            meso_tn2: [0.0; 4],
            meso_tn3: [0.0; 5],
            meso_tgn1: [0.0; 2],
            meso_tgn2: [0.0; 2],
            meso_tgn3: [0.0; 2],
            dfa: 0.0,
            plg: [[0.0; 9]; 4],
            ctloc: 0.0,
            stloc: 0.0,
            c2tloc: 0.0,
            s2tloc: 0.0,
            s3tloc: 0.0,
            c3tloc: 0.0,
            apdf: 0.0,
            apt: [0.0; 4],
        }
    }

    /// Latitude variation of gravity; sets `gsurf` and `re` (reference
    /// `glatf`).
    fn glatf(&mut self, lat: f64) {
        let c2 = (2.0 * DGTR * lat).cos();
        self.gsurf = 980.616 * (1.0 - 0.0026373 * c2);
        self.re = 2.0 * self.gsurf / (3.085462E-6 + 2.27E-9 * c2) * 1.0E-5;
    }

    /// Geopotential difference (reference `zeta`).
    fn zeta(&self, zz: f64, zl: f64) -> f64 {
        (zz - zl) * (self.re + zl) / (self.re + zz)
    }

    /// Scale height (reference `scalh`).
    fn scalh(&self, alt: f64, xm: f64, temp: f64) -> f64 {
        let g = self.gsurf / (1.0 + alt / self.re).powf(2.0);
        RGAS * temp / (g * xm)
    }

    /// Temperature and density profiles for the lower atmosphere (reference
    /// `densm`). Returns density (or, when `xm == 0`, the temperature) and
    /// writes the temperature at altitude into `tz`.
    #[allow(clippy::too_many_arguments)]
    fn densm(
        &self,
        alt: f64,
        d0: f64,
        xm: f64,
        tz: &mut f64,
        mn3: usize,
        zn3: &[f64],
        tn3: &[f64],
        tgn3: &[f64],
        mn2: usize,
        zn2: &[f64],
        tn2: &[f64],
        tgn2: &[f64],
    ) -> f64 {
        let mut xs = [0.0_f64; 10];
        let mut ys = [0.0_f64; 10];
        let mut y2out = [0.0_f64; 10];
        let mut densm_tmp = d0;

        if alt > zn2[0] {
            if xm == 0.0 {
                return *tz;
            } else {
                return d0;
            }
        }

        // Stratosphere / mesosphere temperature.
        let z = if alt > zn2[mn2 - 1] {
            alt
        } else {
            zn2[mn2 - 1]
        };
        let mn = mn2;
        let z1 = zn2[0];
        let z2 = zn2[mn - 1];
        let t1 = tn2[0];
        let t2 = tn2[mn - 1];
        let zg = self.zeta(z, z1);
        let zgdif = self.zeta(z2, z1);

        for k in 0..mn {
            xs[k] = self.zeta(zn2[k], z1) / zgdif;
            ys[k] = 1.0 / tn2[k];
        }
        let yd1 = -tgn2[0] / (t1 * t1) * zgdif;
        let yd2 = -tgn2[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)).powf(2.0);

        spline(&xs, &ys, mn, yd1, yd2, &mut y2out);
        let x = zg / zgdif;
        let y = splint(&xs, &ys, &y2out, mn, x);

        *tz = 1.0 / y;
        if xm != 0.0 {
            let glb = self.gsurf / (1.0 + z1 / self.re).powf(2.0);
            let gamm = xm * glb * zgdif / RGAS;
            let yi = splini(&xs, &ys, &y2out, mn, x);
            let mut expl = gamm * yi;
            if expl > 50.0 {
                expl = 50.0;
            }
            densm_tmp = densm_tmp * (t1 / *tz) * (-expl).exp();
        }

        if alt > zn3[0] {
            if xm == 0.0 {
                return *tz;
            } else {
                return densm_tmp;
            }
        }

        // Troposphere / stratosphere temperature.
        let z = alt;
        let mn = mn3;
        let z1 = zn3[0];
        let z2 = zn3[mn - 1];
        let t1 = tn3[0];
        let t2 = tn3[mn - 1];
        let zg = self.zeta(z, z1);
        let zgdif = self.zeta(z2, z1);

        for k in 0..mn {
            xs[k] = self.zeta(zn3[k], z1) / zgdif;
            ys[k] = 1.0 / tn3[k];
        }
        let yd1 = -tgn3[0] / (t1 * t1) * zgdif;
        let yd2 = -tgn3[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)).powf(2.0);

        spline(&xs, &ys, mn, yd1, yd2, &mut y2out);
        let x = zg / zgdif;
        let y = splint(&xs, &ys, &y2out, mn, x);

        *tz = 1.0 / y;
        if xm != 0.0 {
            let glb = self.gsurf / (1.0 + z1 / self.re).powf(2.0);
            let gamm = xm * glb * zgdif / RGAS;
            let yi = splini(&xs, &ys, &y2out, mn, x);
            let mut expl = gamm * yi;
            if expl > 50.0 {
                expl = 50.0;
            }
            densm_tmp = densm_tmp * (t1 / *tz) * (-expl).exp();
        }
        if xm == 0.0 {
            *tz
        } else {
            densm_tmp
        }
    }

    /// Temperature and density profiles for the thermosphere (reference
    /// `densu`). Returns density (or temperature when `xm == 0`) and writes the
    /// temperature at altitude into `tz`. `tn1`/`tgn1` end nodes are updated in
    /// place, matching the reference.
    #[allow(clippy::too_many_arguments)]
    fn densu(
        &self,
        alt: f64,
        dlb: f64,
        tinf: f64,
        tlb: f64,
        xm: f64,
        alpha: f64,
        tz: &mut f64,
        zlb: f64,
        s2: f64,
        mn1: usize,
        zn1: &[f64],
        tn1: &mut [f64],
        tgn1: &mut [f64],
    ) -> f64 {
        let mut xs = [0.0_f64; 5];
        let mut ys = [0.0_f64; 5];
        let mut y2out = [0.0_f64; 5];
        let mut densu_temp;

        let za = zn1[0];
        let z = if alt > za { alt } else { za };

        let zg2 = self.zeta(z, zlb);

        let tt = tinf - (tinf - tlb) * (-s2 * zg2).exp();
        let ta = tt;
        *tz = tt;
        densu_temp = *tz;

        // Spline node bookkeeping reused for the sub-`za` density branch.
        let mut z1 = 0.0;
        let mut zgdif = 0.0;
        let mut mn = mn1;
        let mut x = 0.0;
        let mut t1 = 0.0;

        if alt < za {
            let dta = (tinf - ta) * s2 * ((self.re + zlb) / (self.re + za)).powf(2.0);
            tgn1[0] = dta;
            tn1[0] = ta;
            let z = if alt > zn1[mn1 - 1] {
                alt
            } else {
                zn1[mn1 - 1]
            };
            mn = mn1;
            z1 = zn1[0];
            let z2 = zn1[mn - 1];
            t1 = tn1[0];
            let t2 = tn1[mn - 1];
            let zg = self.zeta(z, z1);
            zgdif = self.zeta(z2, z1);
            for k in 0..mn {
                xs[k] = self.zeta(zn1[k], z1) / zgdif;
                ys[k] = 1.0 / tn1[k];
            }
            let yd1 = -tgn1[0] / (t1 * t1) * zgdif;
            let yd2 = -tgn1[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)).powf(2.0);
            spline(&xs, &ys, mn, yd1, yd2, &mut y2out);
            x = zg / zgdif;
            let y = splint(&xs, &ys, &y2out, mn, x);
            *tz = 1.0 / y;
            densu_temp = *tz;
        }
        if xm == 0.0 {
            return densu_temp;
        }

        let glb = self.gsurf / (1.0 + zlb / self.re).powf(2.0);
        let gamma = xm * glb / (s2 * RGAS * tinf);
        let mut expl = (-s2 * gamma * zg2).exp();
        if expl > 50.0 {
            expl = 50.0;
        }
        if tt <= 0.0 {
            expl = 50.0;
        }

        let densa = dlb * (tlb / tt).powf(1.0 + alpha + gamma) * expl;
        densu_temp = densa;
        if alt >= za {
            return densu_temp;
        }

        let glb = self.gsurf / (1.0 + z1 / self.re).powf(2.0);
        let gamm = xm * glb * zgdif / RGAS;

        let yi = splini(&xs, &ys, &y2out, mn, x);
        let mut expl = gamm * yi;
        if expl > 50.0 {
            expl = 50.0;
        }
        if *tz <= 0.0 {
            expl = 50.0;
        }

        densu_temp * (t1 / *tz).powf(1.0 + alpha) * (-expl).exp()
    }

    /// Upper-thermosphere G(L) expansion (reference `globe7`). Sets the shared
    /// Legendre/time/activity state used by [`State::glob7s`].
    fn globe7(&mut self, p: &[f64], input: &NrlmsiseInput, flags: &Flags) -> f64 {
        let mut t = [0.0_f64; 15];
        let tloc = input.lst;

        let c = (input.g_lat * DGTR).sin();
        let s = (input.g_lat * DGTR).cos();
        let c2 = c * c;
        let c4 = c2 * c2;
        let s2 = s * s;

        self.plg[0][1] = c;
        self.plg[0][2] = 0.5 * (3.0 * c2 - 1.0);
        self.plg[0][3] = 0.5 * (5.0 * c * c2 - 3.0 * c);
        self.plg[0][4] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0;
        self.plg[0][5] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0;
        self.plg[0][6] = (11.0 * c * self.plg[0][5] - 5.0 * self.plg[0][4]) / 6.0;
        self.plg[1][1] = s;
        self.plg[1][2] = 3.0 * c * s;
        self.plg[1][3] = 1.5 * (5.0 * c2 - 1.0) * s;
        self.plg[1][4] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s;
        self.plg[1][5] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s;
        self.plg[1][6] = (11.0 * c * self.plg[1][5] - 6.0 * self.plg[1][4]) / 5.0;
        self.plg[2][2] = 3.0 * s2;
        self.plg[2][3] = 15.0 * s2 * c;
        self.plg[2][4] = 7.5 * (7.0 * c2 - 1.0) * s2;
        self.plg[2][5] = 3.0 * c * self.plg[2][4] - 2.0 * self.plg[2][3];
        self.plg[2][6] = (11.0 * c * self.plg[2][5] - 7.0 * self.plg[2][4]) / 4.0;
        self.plg[2][7] = (13.0 * c * self.plg[2][6] - 8.0 * self.plg[2][5]) / 5.0;
        self.plg[3][3] = 15.0 * s2 * s;
        self.plg[3][4] = 105.0 * s2 * s * c;
        self.plg[3][5] = (9.0 * c * self.plg[3][4] - 7.0 * self.plg[3][3]) / 2.0;
        self.plg[3][6] = (11.0 * c * self.plg[3][5] - 8.0 * self.plg[3][4]) / 3.0;

        if !(((flags.sw[7] == 0.0) && (flags.sw[8] == 0.0)) && (flags.sw[14] == 0.0)) {
            self.stloc = (HR * tloc).sin();
            self.ctloc = (HR * tloc).cos();
            self.s2tloc = (2.0 * HR * tloc).sin();
            self.c2tloc = (2.0 * HR * tloc).cos();
            self.s3tloc = (3.0 * HR * tloc).sin();
            self.c3tloc = (3.0 * HR * tloc).cos();
        }

        let doy = input.doy as f64;
        let cd32 = (DR * (doy - p[31])).cos();
        let cd18 = (2.0 * DR * (doy - p[17])).cos();
        let cd14 = (DR * (doy - p[13])).cos();
        let cd39 = (2.0 * DR * (doy - p[38])).cos();

        // F10.7 effect.
        let df = input.f107 - input.f107a;
        self.dfa = input.f107a - 150.0;
        let dfa = self.dfa;
        t[0] = p[19] * df * (1.0 + p[59] * dfa)
            + p[20] * df * df
            + p[21] * dfa
            + p[29] * dfa.powf(2.0);
        let f1 = 1.0 + (p[47] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1];
        let f2 = 1.0 + (p[49] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1];

        // Time independent.
        t[1] = (p[1] * self.plg[0][2] + p[2] * self.plg[0][4] + p[22] * self.plg[0][6])
            + (p[14] * self.plg[0][2]) * dfa * flags.swc[1]
            + p[26] * self.plg[0][1];

        // Symmetrical annual.
        t[2] = p[18] * cd32;

        // Symmetrical semiannual.
        t[3] = (p[15] + p[16] * self.plg[0][2]) * cd18;

        // Asymmetrical annual.
        t[4] = f1 * (p[9] * self.plg[0][1] + p[10] * self.plg[0][3]) * cd14;

        // Asymmetrical semiannual.
        t[5] = p[37] * self.plg[0][1] * cd39;

        // Diurnal.
        if flags.sw[7] != 0.0 {
            let t71 = (p[11] * self.plg[1][2]) * cd14 * flags.swc[5];
            let t72 = (p[12] * self.plg[1][2]) * cd14 * flags.swc[5];
            t[6] = f2
                * ((p[3] * self.plg[1][1] + p[4] * self.plg[1][3] + p[27] * self.plg[1][5] + t71)
                    * self.ctloc
                    + (p[6] * self.plg[1][1]
                        + p[7] * self.plg[1][3]
                        + p[28] * self.plg[1][5]
                        + t72)
                        * self.stloc);
        }

        // Semidiurnal.
        if flags.sw[8] != 0.0 {
            let t81 = (p[23] * self.plg[2][3] + p[35] * self.plg[2][5]) * cd14 * flags.swc[5];
            let t82 = (p[33] * self.plg[2][3] + p[36] * self.plg[2][5]) * cd14 * flags.swc[5];
            t[7] = f2
                * ((p[5] * self.plg[2][2] + p[41] * self.plg[2][4] + t81) * self.c2tloc
                    + (p[8] * self.plg[2][2] + p[42] * self.plg[2][4] + t82) * self.s2tloc);
        }

        // Terdiurnal.
        if flags.sw[14] != 0.0 {
            t[13] = f2
                * ((p[39] * self.plg[3][3]
                    + (p[93] * self.plg[3][4] + p[46] * self.plg[3][6]) * cd14 * flags.swc[5])
                    * self.s3tloc
                    + (p[40] * self.plg[3][3]
                        + (p[94] * self.plg[3][4] + p[48] * self.plg[3][6]) * cd14 * flags.swc[5])
                        * self.c3tloc);
        }

        // Magnetic activity.
        if flags.sw[9] == -1.0 {
            // Ap-history mode. The public entry points validate that `ap_array`
            // is present whenever switch 9 is -1, so this never substitutes a
            // default; the expect documents that boundary invariant.
            let ap = input
                .ap_array
                .expect("ap_array must be present in Ap-history mode (validated at entry)");
            if p[51] != 0.0 {
                // The reference clamps p[24] in place; do so on a local copy
                // (the clamp never fires for the standard tables).
                let mut pc = [0.0_f64; 150];
                pc.copy_from_slice(p);
                let mut exp1 = (-10800.0 * (pc[51] * pc[51]).sqrt()
                    / (1.0 + pc[138] * (45.0 - (input.g_lat * input.g_lat).sqrt())))
                .exp();
                if exp1 > 0.99999 {
                    exp1 = 0.99999;
                }
                if pc[24] < 1.0E-4 {
                    pc[24] = 1.0E-4;
                }
                self.apt[0] = sg0(exp1, &pc, &ap);
                if flags.sw[9] != 0.0 {
                    t[8] = self.apt[0]
                        * (p[50]
                            + p[96] * self.plg[0][2]
                            + p[54] * self.plg[0][4]
                            + (p[125] * self.plg[0][1]
                                + p[126] * self.plg[0][3]
                                + p[127] * self.plg[0][5])
                                * cd14
                                * flags.swc[5]
                            + (p[128] * self.plg[1][1]
                                + p[129] * self.plg[1][3]
                                + p[130] * self.plg[1][5])
                                * flags.swc[7]
                                * (HR * (tloc - p[131])).cos());
                }
            }
        } else {
            let apd = input.ap - 4.0;
            let mut p44 = p[43];
            let p45 = p[44];
            if p44 < 0.0 {
                p44 = 1.0E-5;
            }
            self.apdf = apd + (p45 - 1.0) * (apd + ((-p44 * apd).exp() - 1.0) / p44);
            if flags.sw[9] != 0.0 {
                t[8] = self.apdf
                    * (p[32]
                        + p[45] * self.plg[0][2]
                        + p[34] * self.plg[0][4]
                        + (p[100] * self.plg[0][1]
                            + p[101] * self.plg[0][3]
                            + p[102] * self.plg[0][5])
                            * cd14
                            * flags.swc[5]
                        + (p[121] * self.plg[1][1]
                            + p[122] * self.plg[1][3]
                            + p[123] * self.plg[1][5])
                            * flags.swc[7]
                            * (HR * (tloc - p[124])).cos());
            }
        }

        if (flags.sw[10] != 0.0) && (input.g_long > -1000.0) {
            // Longitudinal.
            if flags.sw[11] != 0.0 {
                t[10] = (1.0 + p[80] * dfa * flags.swc[1])
                    * ((p[64] * self.plg[1][2]
                        + p[65] * self.plg[1][4]
                        + p[66] * self.plg[1][6]
                        + p[103] * self.plg[1][1]
                        + p[104] * self.plg[1][3]
                        + p[105] * self.plg[1][5]
                        + flags.swc[5]
                            * (p[109] * self.plg[1][1]
                                + p[110] * self.plg[1][3]
                                + p[111] * self.plg[1][5])
                            * cd14)
                        * (DGTR * input.g_long).cos()
                        + (p[90] * self.plg[1][2]
                            + p[91] * self.plg[1][4]
                            + p[92] * self.plg[1][6]
                            + p[106] * self.plg[1][1]
                            + p[107] * self.plg[1][3]
                            + p[108] * self.plg[1][5]
                            + flags.swc[5]
                                * (p[112] * self.plg[1][1]
                                    + p[113] * self.plg[1][3]
                                    + p[114] * self.plg[1][5])
                                * cd14)
                            * (DGTR * input.g_long).sin());
            }

            // UT and mixed UT/longitude.
            if flags.sw[12] != 0.0 {
                t[11] = (1.0 + p[95] * self.plg[0][1])
                    * (1.0 + p[81] * dfa * flags.swc[1])
                    * (1.0 + p[119] * self.plg[0][1] * flags.swc[5] * cd14)
                    * ((p[68] * self.plg[0][1] + p[69] * self.plg[0][3] + p[70] * self.plg[0][5])
                        * (SR * (input.sec - p[71])).cos());
                t[11] += flags.swc[11]
                    * (p[76] * self.plg[2][3] + p[77] * self.plg[2][5] + p[78] * self.plg[2][7])
                    * (SR * (input.sec - p[79]) + 2.0 * DGTR * input.g_long).cos()
                    * (1.0 + p[137] * dfa * flags.swc[1]);
            }

            // UT/longitude magnetic activity.
            if flags.sw[13] != 0.0 {
                if flags.sw[9] == -1.0 {
                    if p[51] != 0.0 {
                        t[12] = self.apt[0]
                            * flags.swc[11]
                            * (1.0 + p[132] * self.plg[0][1])
                            * ((p[52] * self.plg[1][2]
                                + p[98] * self.plg[1][4]
                                + p[67] * self.plg[1][6])
                                * (DGTR * (input.g_long - p[97])).cos())
                            + self.apt[0]
                                * flags.swc[11]
                                * flags.swc[5]
                                * (p[133] * self.plg[1][1]
                                    + p[134] * self.plg[1][3]
                                    + p[135] * self.plg[1][5])
                                * cd14
                                * (DGTR * (input.g_long - p[136])).cos()
                            + self.apt[0]
                                * flags.swc[12]
                                * (p[55] * self.plg[0][1]
                                    + p[56] * self.plg[0][3]
                                    + p[57] * self.plg[0][5])
                                * (SR * (input.sec - p[58])).cos();
                    }
                } else {
                    t[12] = self.apdf
                        * flags.swc[11]
                        * (1.0 + p[120] * self.plg[0][1])
                        * ((p[60] * self.plg[1][2]
                            + p[61] * self.plg[1][4]
                            + p[62] * self.plg[1][6])
                            * (DGTR * (input.g_long - p[63])).cos())
                        + self.apdf
                            * flags.swc[11]
                            * flags.swc[5]
                            * (p[115] * self.plg[1][1]
                                + p[116] * self.plg[1][3]
                                + p[117] * self.plg[1][5])
                            * cd14
                            * (DGTR * (input.g_long - p[118])).cos()
                        + self.apdf
                            * flags.swc[12]
                            * (p[83] * self.plg[0][1]
                                + p[84] * self.plg[0][3]
                                + p[85] * self.plg[0][5])
                            * (SR * (input.sec - p[75])).cos();
                }
            }
        }

        let mut tinf = p[30];
        for i in 0..14 {
            tinf += flags.sw[i + 1].abs() * t[i];
        }
        tinf
    }

    /// Lower-atmosphere G(L) expansion (reference `glob7s`). Reads shared state
    /// produced by [`State::globe7`].
    fn glob7s(&self, p: &[f64], input: &NrlmsiseInput, flags: &Flags) -> f64 {
        let pset = 2.0;
        let mut t = [0.0_f64; 14];
        let p99 = if p[99] == 0.0 { pset } else { p[99] };
        if p99 != pset {
            return -1.0;
        }
        let doy = input.doy as f64;
        let cd32 = (DR * (doy - p[31])).cos();
        let cd18 = (2.0 * DR * (doy - p[17])).cos();
        let cd14 = (DR * (doy - p[13])).cos();
        let cd39 = (2.0 * DR * (doy - p[38])).cos();

        // F10.7.
        t[0] = p[21] * self.dfa;

        // Time independent.
        t[1] = p[1] * self.plg[0][2]
            + p[2] * self.plg[0][4]
            + p[22] * self.plg[0][6]
            + p[26] * self.plg[0][1]
            + p[14] * self.plg[0][3]
            + p[59] * self.plg[0][5];

        // Symmetrical annual.
        t[2] = (p[18] + p[47] * self.plg[0][2] + p[29] * self.plg[0][4]) * cd32;

        // Symmetrical semiannual.
        t[3] = (p[15] + p[16] * self.plg[0][2] + p[30] * self.plg[0][4]) * cd18;

        // Asymmetrical annual.
        t[4] = (p[9] * self.plg[0][1] + p[10] * self.plg[0][3] + p[20] * self.plg[0][5]) * cd14;

        // Asymmetrical semiannual.
        t[5] = (p[37] * self.plg[0][1]) * cd39;

        // Diurnal.
        if flags.sw[7] != 0.0 {
            let t71 = p[11] * self.plg[1][2] * cd14 * flags.swc[5];
            let t72 = p[12] * self.plg[1][2] * cd14 * flags.swc[5];
            t[6] = (p[3] * self.plg[1][1] + p[4] * self.plg[1][3] + t71) * self.ctloc
                + (p[6] * self.plg[1][1] + p[7] * self.plg[1][3] + t72) * self.stloc;
        }

        // Semidiurnal.
        if flags.sw[8] != 0.0 {
            let t81 = (p[23] * self.plg[2][3] + p[35] * self.plg[2][5]) * cd14 * flags.swc[5];
            let t82 = (p[33] * self.plg[2][3] + p[36] * self.plg[2][5]) * cd14 * flags.swc[5];
            t[7] = (p[5] * self.plg[2][2] + p[41] * self.plg[2][4] + t81) * self.c2tloc
                + (p[8] * self.plg[2][2] + p[42] * self.plg[2][4] + t82) * self.s2tloc;
        }

        // Terdiurnal.
        if flags.sw[14] != 0.0 {
            t[13] = p[39] * self.plg[3][3] * self.s3tloc + p[40] * self.plg[3][3] * self.c3tloc;
        }

        // Magnetic activity.
        if flags.sw[9] != 0.0 {
            if flags.sw[9] == 1.0 {
                t[8] = self.apdf * (p[32] + p[45] * self.plg[0][2] * flags.swc[2]);
            }
            if flags.sw[9] == -1.0 {
                t[8] = p[50] * self.apt[0] + p[96] * self.plg[0][2] * self.apt[0] * flags.swc[2];
            }
        }

        // Longitudinal.
        if !((flags.sw[10] == 0.0) || (flags.sw[11] == 0.0) || (input.g_long <= -1000.0)) {
            t[10] = (1.0
                + self.plg[0][1]
                    * (p[80] * flags.swc[5] * (DR * (doy - p[81])).cos()
                        + p[85] * flags.swc[6] * (2.0 * DR * (doy - p[86])).cos())
                + p[83] * flags.swc[3] * (DR * (doy - p[84])).cos()
                + p[87] * flags.swc[4] * (2.0 * DR * (doy - p[88])).cos())
                * ((p[64] * self.plg[1][2]
                    + p[65] * self.plg[1][4]
                    + p[66] * self.plg[1][6]
                    + p[74] * self.plg[1][1]
                    + p[75] * self.plg[1][3]
                    + p[76] * self.plg[1][5])
                    * (DGTR * input.g_long).cos()
                    + (p[90] * self.plg[1][2]
                        + p[91] * self.plg[1][4]
                        + p[92] * self.plg[1][6]
                        + p[77] * self.plg[1][1]
                        + p[78] * self.plg[1][3]
                        + p[79] * self.plg[1][5])
                        * (DGTR * input.g_long).sin());
        }
        let mut tt = 0.0;
        for i in 0..14 {
            tt += flags.sw[i + 1].abs() * t[i];
        }
        tt
    }
}

impl State {
    /// Thermospheric portion of NRLMSISE-00 (reference `gts7`). Valid for
    /// `alt > 72.5 km`.
    fn gts7(&mut self, input: &NrlmsiseInput, flags: &Flags, output: &mut NrlmsiseOutput) {
        use tables::*;

        let mut zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
        let mn1 = 5usize;
        let alpha = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0];
        let altl = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0];

        let za = PDL[1][15];
        zn1[0] = za;
        for j in 0..9 {
            output.d[j] = 0.0;
        }

        // Tinf variations are unimportant below za / zn1[0].
        let tinf = if input.alt > zn1[0] {
            PTM[0] * PT[0] * (1.0 + flags.sw[16] * self.globe7(&PT, input, flags))
        } else {
            PTM[0] * PT[0]
        };
        output.t[0] = tinf;

        // Gradient variations are unimportant below zn1[4].
        let g0 = if input.alt > zn1[4] {
            PTM[3] * PS[0] * (1.0 + flags.sw[19] * self.globe7(&PS, input, flags))
        } else {
            PTM[3] * PS[0]
        };
        let tlb = PTM[1] * (1.0 + flags.sw[17] * self.globe7(&PD[3], input, flags)) * PD[3][0];
        let s = g0 / (tinf - tlb);

        // Lower-thermosphere temperature variations are insignificant for
        // density above 300 km.
        if input.alt < 300.0 {
            self.meso_tn1[1] =
                PTM[6] * PTL[0][0] / (1.0 - flags.sw[18] * self.glob7s(&PTL[0], input, flags));
            self.meso_tn1[2] =
                PTM[2] * PTL[1][0] / (1.0 - flags.sw[18] * self.glob7s(&PTL[1], input, flags));
            self.meso_tn1[3] =
                PTM[7] * PTL[2][0] / (1.0 - flags.sw[18] * self.glob7s(&PTL[2], input, flags));
            self.meso_tn1[4] = PTM[4] * PTL[3][0]
                / (1.0 - flags.sw[18] * flags.sw[20] * self.glob7s(&PTL[3], input, flags));
            self.meso_tgn1[1] = PTM[8]
                * PMA[8][0]
                * (1.0 + flags.sw[18] * flags.sw[20] * self.glob7s(&PMA[8], input, flags))
                * self.meso_tn1[4]
                * self.meso_tn1[4]
                / (PTM[4] * PTL[3][0]).powf(2.0);
        } else {
            self.meso_tn1[1] = PTM[6] * PTL[0][0];
            self.meso_tn1[2] = PTM[2] * PTL[1][0];
            self.meso_tn1[3] = PTM[7] * PTL[2][0];
            self.meso_tn1[4] = PTM[4] * PTL[3][0];
            self.meso_tgn1[1] = PTM[8] * PMA[8][0] * self.meso_tn1[4] * self.meso_tn1[4]
                / (PTM[4] * PTL[3][0]).powf(2.0);
        }

        // N2 variation factor at Zlb.
        let g28 = flags.sw[21] * self.globe7(&PD[2], input, flags);

        // Variation of turbopause height.
        let zhf = PDL[1][24]
            * (1.0
                + flags.sw[5]
                    * PDL[0][24]
                    * (DGTR * input.g_lat).sin()
                    * (DR * (input.doy as f64 - PT[13])).cos());
        output.t[0] = tinf;
        let xmm = PDM[2][4];
        let z = input.alt;

        // Extract meso end nodes so densu can borrow them mutably while we read
        // the shared state; copy back afterward.
        let mut tn1 = self.meso_tn1;
        let mut tgn1 = self.meso_tgn1;

        // N2 density.
        let db28 = PDM[2][0] * g28.exp() * PD[2][0];
        output.d[2] = self.densu(
            z,
            db28,
            tinf,
            tlb,
            28.0,
            alpha[2],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        let zh28 = PDM[2][2] * zhf;
        let zhm28 = PDM[2][3] * PDL[1][5];
        let xmd = 28.0 - xmm;
        let mut tz = 0.0;
        let b28 = self.densu(
            zh28,
            db28,
            tinf,
            tlb,
            xmd,
            alpha[2] - 1.0,
            &mut tz,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if (flags.sw[15] != 0.0) && (z <= altl[2]) {
            self.dm28 = self.densu(
                z, b28, tinf, tlb, xmm, alpha[2], &mut tz, PTM[5], s, mn1, &zn1, &mut tn1,
                &mut tgn1,
            );
            output.d[2] = dnet(output.d[2], self.dm28, zhm28, xmm, 28.0);
        }

        // He density.
        let g4 = flags.sw[21] * self.globe7(&PD[0], input, flags);
        let db04 = PDM[0][0] * g4.exp() * PD[0][0];
        output.d[0] = self.densu(
            z,
            db04,
            tinf,
            tlb,
            4.0,
            alpha[0],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if (flags.sw[15] != 0.0) && (z < altl[0]) {
            let zh04 = PDM[0][2];
            let b04 = self.densu(
                zh04,
                db04,
                tinf,
                tlb,
                4.0 - xmm,
                alpha[0] - 1.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            self.dm04 = self.densu(
                z,
                b04,
                tinf,
                tlb,
                xmm,
                0.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            let zhm04 = zhm28;
            output.d[0] = dnet(output.d[0], self.dm04, zhm04, xmm, 4.0);
            let rl = (b28 * PDM[0][1] / b04).ln();
            let zc04 = PDM[0][4] * PDL[1][0];
            let hc04 = PDM[0][5] * PDL[1][1];
            output.d[0] *= ccor(z, rl, hc04, zc04);
        }

        // O density.
        let g16 = flags.sw[21] * self.globe7(&PD[1], input, flags);
        let db16 = PDM[1][0] * g16.exp() * PD[1][0];
        output.d[1] = self.densu(
            z,
            db16,
            tinf,
            tlb,
            16.0,
            alpha[1],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if (flags.sw[15] != 0.0) && (z <= altl[1]) {
            let zh16 = PDM[1][2];
            let b16 = self.densu(
                zh16,
                db16,
                tinf,
                tlb,
                16.0 - xmm,
                alpha[1] - 1.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            self.dm16 = self.densu(
                z,
                b16,
                tinf,
                tlb,
                xmm,
                0.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            let zhm16 = zhm28;
            output.d[1] = dnet(output.d[1], self.dm16, zhm16, xmm, 16.0);
            let rl =
                PDM[1][1] * PDL[1][16] * (1.0 + flags.sw[1] * PDL[0][23] * (input.f107a - 150.0));
            let hc16 = PDM[1][5] * PDL[1][3];
            let zc16 = PDM[1][4] * PDL[1][2];
            let hc216 = PDM[1][5] * PDL[1][4];
            output.d[1] *= ccor2(z, rl, hc16, zc16, hc216);
            let hcc16 = PDM[1][7] * PDL[1][13];
            let zcc16 = PDM[1][6] * PDL[1][12];
            let rc16 = PDM[1][3] * PDL[1][14];
            output.d[1] *= ccor(z, rc16, hcc16, zcc16);
        }

        // O2 density.
        let g32 = flags.sw[21] * self.globe7(&PD[4], input, flags);
        let db32 = PDM[3][0] * g32.exp() * PD[4][0];
        output.d[3] = self.densu(
            z,
            db32,
            tinf,
            tlb,
            32.0,
            alpha[3],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if flags.sw[15] != 0.0 {
            if z <= altl[3] {
                let zh32 = PDM[3][2];
                let b32 = self.densu(
                    zh32,
                    db32,
                    tinf,
                    tlb,
                    32.0 - xmm,
                    alpha[3] - 1.0,
                    &mut output.t[1],
                    PTM[5],
                    s,
                    mn1,
                    &zn1,
                    &mut tn1,
                    &mut tgn1,
                );
                self.dm32 = self.densu(
                    z,
                    b32,
                    tinf,
                    tlb,
                    xmm,
                    0.0,
                    &mut output.t[1],
                    PTM[5],
                    s,
                    mn1,
                    &zn1,
                    &mut tn1,
                    &mut tgn1,
                );
                let zhm32 = zhm28;
                output.d[3] = dnet(output.d[3], self.dm32, zhm32, xmm, 32.0);
                let rl = (b28 * PDM[3][1] / b32).ln();
                let hc32 = PDM[3][5] * PDL[1][7];
                let zc32 = PDM[3][4] * PDL[1][6];
                output.d[3] *= ccor(z, rl, hc32, zc32);
            }
            let hcc32 = PDM[3][7] * PDL[1][22];
            let hcc232 = PDM[3][7] * PDL[0][22];
            let zcc32 = PDM[3][6] * PDL[1][21];
            let rc32 =
                PDM[3][3] * PDL[1][23] * (1.0 + flags.sw[1] * PDL[0][23] * (input.f107a - 150.0));
            output.d[3] *= ccor2(z, rc32, hcc32, zcc32, hcc232);
        }

        // Ar density.
        let g40 = flags.sw[21] * self.globe7(&PD[5], input, flags);
        let db40 = PDM[4][0] * g40.exp() * PD[5][0];
        output.d[4] = self.densu(
            z,
            db40,
            tinf,
            tlb,
            40.0,
            alpha[4],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if (flags.sw[15] != 0.0) && (z <= altl[4]) {
            let zh40 = PDM[4][2];
            let b40 = self.densu(
                zh40,
                db40,
                tinf,
                tlb,
                40.0 - xmm,
                alpha[4] - 1.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            self.dm40 = self.densu(
                z,
                b40,
                tinf,
                tlb,
                xmm,
                0.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            let zhm40 = zhm28;
            output.d[4] = dnet(output.d[4], self.dm40, zhm40, xmm, 40.0);
            let rl = (b28 * PDM[4][1] / b40).ln();
            let hc40 = PDM[4][5] * PDL[1][9];
            let zc40 = PDM[4][4] * PDL[1][8];
            output.d[4] *= ccor(z, rl, hc40, zc40);
        }

        // Hydrogen density.
        let g1 = flags.sw[21] * self.globe7(&PD[6], input, flags);
        let db01 = PDM[5][0] * g1.exp() * PD[6][0];
        output.d[6] = self.densu(
            z,
            db01,
            tinf,
            tlb,
            1.0,
            alpha[6],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if (flags.sw[15] != 0.0) && (z <= altl[6]) {
            let zh01 = PDM[5][2];
            let b01 = self.densu(
                zh01,
                db01,
                tinf,
                tlb,
                1.0 - xmm,
                alpha[6] - 1.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            self.dm01 = self.densu(
                z,
                b01,
                tinf,
                tlb,
                xmm,
                0.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            let zhm01 = zhm28;
            output.d[6] = dnet(output.d[6], self.dm01, zhm01, xmm, 1.0);
            let rl = (b28 * PDM[5][1] * (PDL[1][17] * PDL[1][17]).sqrt() / b01).ln();
            let hc01 = PDM[5][5] * PDL[1][11];
            let zc01 = PDM[5][4] * PDL[1][10];
            output.d[6] *= ccor(z, rl, hc01, zc01);
            let hcc01 = PDM[5][7] * PDL[1][19];
            let zcc01 = PDM[5][6] * PDL[1][18];
            let rc01 = PDM[5][3] * PDL[1][20];
            output.d[6] *= ccor(z, rc01, hcc01, zcc01);
        }

        // Atomic nitrogen density.
        let g14 = flags.sw[21] * self.globe7(&PD[7], input, flags);
        let db14 = PDM[6][0] * g14.exp() * PD[7][0];
        output.d[7] = self.densu(
            z,
            db14,
            tinf,
            tlb,
            14.0,
            alpha[7],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        if (flags.sw[15] != 0.0) && (z <= altl[7]) {
            let zh14 = PDM[6][2];
            let b14 = self.densu(
                zh14,
                db14,
                tinf,
                tlb,
                14.0 - xmm,
                alpha[7] - 1.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            self.dm14 = self.densu(
                z,
                b14,
                tinf,
                tlb,
                xmm,
                0.0,
                &mut output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut tn1,
                &mut tgn1,
            );
            let zhm14 = zhm28;
            output.d[7] = dnet(output.d[7], self.dm14, zhm14, xmm, 14.0);
            let rl = (b28 * PDM[6][1] * (PDL[0][2] * PDL[0][2]).sqrt() / b14).ln();
            let hc14 = PDM[6][5] * PDL[0][1];
            let zc14 = PDM[6][4] * PDL[0][0];
            output.d[7] *= ccor(z, rl, hc14, zc14);
            let hcc14 = PDM[6][7] * PDL[0][4];
            let zcc14 = PDM[6][6] * PDL[0][3];
            let rc14 = PDM[6][3] * PDL[0][5];
            output.d[7] *= ccor(z, rc14, hcc14, zcc14);
        }

        // Anomalous oxygen density.
        let g16h = flags.sw[21] * self.globe7(&PD[8], input, flags);
        let db16h = PDM[7][0] * g16h.exp() * PD[8][0];
        let tho = PDM[7][9] * PDL[0][6];
        let dd = self.densu(
            z,
            db16h,
            tho,
            tho,
            16.0,
            alpha[8],
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );
        let zsht = PDM[7][5];
        let zmho = PDM[7][4];
        let zsho = self.scalh(zmho, 16.0, tho);
        output.d[8] = dd * (-zsht / zsho * ((-(z - zmho) / zsht).exp() - 1.0)).exp();

        // Total mass density.
        output.d[5] = 1.66E-24
            * (4.0 * output.d[0]
                + 16.0 * output.d[1]
                + 28.0 * output.d[2]
                + 32.0 * output.d[3]
                + 40.0 * output.d[4]
                + output.d[6]
                + 14.0 * output.d[7]);

        // Temperature at altitude.
        let z = (input.alt * input.alt).sqrt();
        let _ = self.densu(
            z,
            1.0,
            tinf,
            tlb,
            0.0,
            0.0,
            &mut output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut tn1,
            &mut tgn1,
        );

        // Copy back the meso end nodes mutated by densu.
        self.meso_tn1 = tn1;
        self.meso_tgn1 = tgn1;

        if flags.sw[0] != 0.0 {
            for i in 0..9 {
                output.d[i] *= 1.0E6;
            }
            output.d[5] /= 1000.0;
        }
    }

    /// Full model from surface to lower exosphere (reference `gtd7`).
    fn gtd7(&mut self, input: &NrlmsiseInput, flags: &Flags, output: &mut NrlmsiseOutput) {
        use tables::*;

        let mn3 = 5usize;
        let zn3 = [32.5, 20.0, 15.0, 10.0, 0.0];
        let mn2 = 4usize;
        let zn2 = [72.5, 55.0, 45.0, 32.5];
        let zmix = 62.5;

        // Latitude variation of gravity (none for sw[2] == 0).
        let xlat = if flags.sw[2] == 0.0 {
            45.0
        } else {
            input.g_lat
        };
        self.glatf(xlat);

        let xmm = PDM[2][4];

        // Thermosphere / mesosphere (above zn2[0]).
        let altt = if input.alt > zn2[0] {
            input.alt
        } else {
            zn2[0]
        };

        let mut sinput = *input;
        sinput.alt = altt;
        let mut soutput = NrlmsiseOutput::default();
        self.gts7(&sinput, flags, &mut soutput);

        let dm28m = if flags.sw[0] != 0.0 {
            self.dm28 * 1.0E6
        } else {
            self.dm28
        };
        output.t[0] = soutput.t[0];
        output.t[1] = soutput.t[1];
        if input.alt >= zn2[0] {
            output.d = soutput.d;
            return;
        }

        // Lower mesosphere / upper stratosphere (between zn3[0] and zn2[0]).
        self.meso_tgn2[0] = self.meso_tgn1[1];
        self.meso_tn2[0] = self.meso_tn1[4];
        self.meso_tn2[1] =
            PMA[0][0] * PAVGM[0] / (1.0 - flags.sw[20] * self.glob7s(&PMA[0], input, flags));
        self.meso_tn2[2] =
            PMA[1][0] * PAVGM[1] / (1.0 - flags.sw[20] * self.glob7s(&PMA[1], input, flags));
        self.meso_tn2[3] = PMA[2][0] * PAVGM[2]
            / (1.0 - flags.sw[20] * flags.sw[22] * self.glob7s(&PMA[2], input, flags));
        self.meso_tgn2[1] = PAVGM[8]
            * PMA[9][0]
            * (1.0 + flags.sw[20] * flags.sw[22] * self.glob7s(&PMA[9], input, flags))
            * self.meso_tn2[3]
            * self.meso_tn2[3]
            / (PMA[2][0] * PAVGM[2]).powf(2.0);
        self.meso_tn3[0] = self.meso_tn2[3];

        if input.alt < zn3[0] {
            // Lower stratosphere and troposphere (below zn3[0]).
            self.meso_tgn3[0] = self.meso_tgn2[1];
            self.meso_tn3[1] =
                PMA[3][0] * PAVGM[3] / (1.0 - flags.sw[22] * self.glob7s(&PMA[3], input, flags));
            self.meso_tn3[2] =
                PMA[4][0] * PAVGM[4] / (1.0 - flags.sw[22] * self.glob7s(&PMA[4], input, flags));
            self.meso_tn3[3] =
                PMA[5][0] * PAVGM[5] / (1.0 - flags.sw[22] * self.glob7s(&PMA[5], input, flags));
            self.meso_tn3[4] =
                PMA[6][0] * PAVGM[6] / (1.0 - flags.sw[22] * self.glob7s(&PMA[6], input, flags));
            self.meso_tgn3[1] = PMA[7][0]
                * PAVGM[7]
                * (1.0 + flags.sw[22] * self.glob7s(&PMA[7], input, flags))
                * self.meso_tn3[4]
                * self.meso_tn3[4]
                / (PMA[6][0] * PAVGM[6]).powf(2.0);
        }

        // Linear transition to full mixing below zn2[0].
        let mut dmc = 0.0;
        if input.alt > zmix {
            dmc = 1.0 - (zn2[0] - input.alt) / (zn2[0] - zmix);
        }
        let dz28 = soutput.d[2];

        // Snapshot meso temperature arrays for densm reads.
        let tn2 = self.meso_tn2;
        let tgn2 = self.meso_tgn2;
        let tn3 = self.meso_tn3;
        let tgn3 = self.meso_tgn3;
        let mut tz = 0.0;

        // N2 density.
        let dmr = soutput.d[2] / dm28m - 1.0;
        output.d[2] = self.densm(
            input.alt, dm28m, xmm, &mut tz, mn3, &zn3, &tn3, &tgn3, mn2, &zn2, &tn2, &tgn2,
        );
        output.d[2] *= 1.0 + dmr * dmc;

        // He density.
        let dmr = soutput.d[0] / (dz28 * PDM[0][1]) - 1.0;
        output.d[0] = output.d[2] * PDM[0][1] * (1.0 + dmr * dmc);

        // O density.
        output.d[1] = 0.0;
        output.d[8] = 0.0;

        // O2 density.
        let dmr = soutput.d[3] / (dz28 * PDM[3][1]) - 1.0;
        output.d[3] = output.d[2] * PDM[3][1] * (1.0 + dmr * dmc);

        // Ar density.
        let dmr = soutput.d[4] / (dz28 * PDM[4][1]) - 1.0;
        output.d[4] = output.d[2] * PDM[4][1] * (1.0 + dmr * dmc);

        // Hydrogen density.
        output.d[6] = 0.0;

        // Atomic nitrogen density.
        output.d[7] = 0.0;

        // Total mass density.
        output.d[5] = 1.66E-24
            * (4.0 * output.d[0]
                + 16.0 * output.d[1]
                + 28.0 * output.d[2]
                + 32.0 * output.d[3]
                + 40.0 * output.d[4]
                + output.d[6]
                + 14.0 * output.d[7]);

        if flags.sw[0] != 0.0 {
            output.d[5] /= 1000.0;
        }

        // Temperature at altitude.
        self.dd = self.densm(
            input.alt, 1.0, 0.0, &mut tz, mn3, &zn3, &tn3, &tgn3, mn2, &zn2, &tn2, &tgn2,
        );
        output.t[1] = tz;
    }

    /// Full model with effective total mass density for drag (reference
    /// `gtd7d`): anomalous oxygen folded into `d[5]`.
    fn gtd7d(&mut self, input: &NrlmsiseInput, flags: &Flags, output: &mut NrlmsiseOutput) {
        self.gtd7(input, flags, output);
        output.d[5] = 1.66E-24
            * (4.0 * output.d[0]
                + 16.0 * output.d[1]
                + 28.0 * output.d[2]
                + 32.0 * output.d[3]
                + 40.0 * output.d[4]
                + output.d[6]
                + 14.0 * output.d[7]
                + 16.0 * output.d[8]);
        if flags.sw[0] != 0.0 {
            output.d[5] /= 1000.0;
        }
    }
}

/// Evaluate the full NRLMSISE-00 model (`gtd7`), excluding anomalous oxygen
/// from the total mass density (`d[5]`).
///
/// For satellite-drag total density (anomalous oxygen folded in) use [`gtd7d`].
/// Inputs are validated at the boundary; see [`AtmosphereError`].
pub fn gtd7(input: &NrlmsiseInput, flags: &Flags) -> Result<NrlmsiseOutput, AtmosphereError> {
    validate(input, flags)?;
    let mut output = NrlmsiseOutput::default();
    let mut state = State::new();
    state.gtd7(input, flags, &mut output);
    Ok(output)
}

/// Evaluate the full NRLMSISE-00 model with effective total mass density for
/// drag (`gtd7d`): anomalous oxygen is folded into the total mass density
/// (`d[5]`), which matters above ~500 km.
///
/// Inputs are validated at the boundary; see [`AtmosphereError`].
pub fn gtd7d(input: &NrlmsiseInput, flags: &Flags) -> Result<NrlmsiseOutput, AtmosphereError> {
    validate(input, flags)?;
    let mut output = NrlmsiseOutput::default();
    let mut state = State::new();
    state.gtd7d(input, flags, &mut output);
    Ok(output)
}

/// Convenience evaluation with all variations on and metric (m/kg) output.
///
/// Returns the full species set, total mass density (kg/m^3), and temperatures
/// (K). Uses [`gtd7d`], so `d[5]` is the drag-effective total mass density
/// including anomalous oxygen, which is the correct quantity for drag users.
/// Use [`gtd7`] explicitly if you need the total mass density without anomalous
/// oxygen.
pub fn nrlmsise00(input: &NrlmsiseInput) -> Result<NrlmsiseOutput, AtmosphereError> {
    gtd7d(input, &Flags::metric())
}

/// [`nrlmsise00`] with the local apparent solar time supplied or derived.
///
/// When `lst` is `Some`, that value is used as [`NrlmsiseInput::lst`]; when it is
/// `None`, the consistent value `local_solar_time(input.sec, input.g_long)` is
/// derived internally so a thin binding need not compute it itself. All other
/// inputs and the evaluation are exactly [`nrlmsise00`]'s, so with an explicit
/// `lst` equal to `input.lst` the result is bit-identical; this wrapper adds no
/// new numeric behaviour, only the optional derivation.
pub fn nrlmsise00_with_lst(
    input: &NrlmsiseInput,
    lst: Option<f64>,
) -> Result<NrlmsiseOutput, AtmosphereError> {
    let mut input = *input;
    input.lst = lst.unwrap_or_else(|| local_solar_time(input.sec, input.g_long));
    nrlmsise00(&input)
}

#[cfg(test)]
mod tests {
    // The oracle fixture holds full-precision f64 literals transcribed from the
    // reference test program output; keep them verbatim.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]
    use super::*;

    // gtd7 reference output, release 20041227, 17 standard cases.
    // Columns: d[0..8] (He,O,N2,O2,Ar,total,H,N,anom-O), t[0]=Tinf, t[1]=Talt.
    const GTD7_REF: [[f64; 11]; 17] = [
        [
            6.66517690495152026E+05,
            1.13880555975221708E+08,
            1.99821092557345442E+07,
            4.02276358571251098E+05,
            3.55746499451588579E+03,
            4.07471353275722314E-15,
            3.47531239971714167E+04,
            4.09591326829300169E+06,
            2.66727320933586889E+04,
            1.25053994356079943E+03,
            1.24141613001912060E+03,
        ],
        [
            3.40729322316091415E+06,
            1.58633336956916809E+08,
            1.39111736546111498E+07,
            3.26255950959554641E+05,
            1.55961815050122459E+03,
            5.00184572907224415E-15,
            4.85420846334025409E+04,
            4.38096671289862506E+06,
            6.95668195594226836E+03,
            1.16675438375720887E+03,
            1.16171045188704238E+03,
        ],
        [
            1.12376724403793560E+05,
            6.93413008676059981E+04,
            4.24710521747708185E+01,
            1.32275014147492764E-01,
            2.61884841823217900E-05,
            2.75677231926887105E-18,
            2.01674985432143185E+04,
            5.74125593414717332E+03,
            2.37439415198959796E+04,
            1.23989211171666511E+03,
            1.23989064013305870E+03,
        ],
        [
            5.41155437993667349E+07,
            1.91889344393930878E+11,
            6.11582559822463086E+12,
            1.22520105174012402E+12,
            6.02321197308486633E+10,
            3.58442630411333278E-10,
            1.05987969774054065E+07,
            2.61573669370513933E+05,
            2.81987935592833352E-42,
            1.02731846489999998E+03,
            2.06887776403605500E+02,
        ],
        [
            1.85112248619252769E+06,
            1.47655483792746186E+08,
            1.57935622826449610E+07,
            2.63379497731231386E+05,
            1.58878139838393008E+03,
            4.80963023940745105E-15,
            5.81616678078747354E+04,
            5.47898447906879056E+06,
            1.26444594176100850E+03,
            1.21239615212120930E+03,
            1.20813542521239174E+03,
        ],
        [
            8.67309523390615708E+05,
            1.27886176801412776E+08,
            1.82257662717170008E+07,
            2.92221419061824679E+05,
            2.40296243642370064E+03,
            4.35586564264464703E-15,
            3.68638924375054994E+04,
            3.89727550372696389E+06,
            2.66727320933586889E+04,
            1.22014641791503209E+03,
            1.21271208321180620E+03,
        ],
        [
            5.77625121602324420E+05,
            6.97913869366019815E+07,
            1.23681355982170273E+07,
            2.49286771542910225E+05,
            1.40573867417784300E+03,
            2.47065139166313234E-15,
            5.29198556706664021E+04,
            1.06981410936656618E+06,
            2.66727320933586780E+04,
            1.11638537604315161E+03,
            1.11299856821731100E+03,
        ],
        [
            3.74030410550766566E+05,
            4.78272012361134216E+07,
            5.24038003332420439E+06,
            1.75987464039060724E+05,
            5.50164877956996406E+02,
            1.57188873925484437E-15,
            8.89677572293503763E+04,
            1.97974083623295487E+06,
            9.12181487599149295E+03,
            1.03124744071455893E+03,
            1.02484849221300897E+03,
        ],
        [
            6.74833876662362367E+05,
            1.24531526044373140E+08,
            2.36900954105298519E+07,
            4.91158315474982315E+05,
            4.57878109905442034E+03,
            4.56442024536117137E-15,
            3.24459477516109328E+04,
            5.37083308708603773E+06,
            2.66727320933586889E+04,
            1.30605204202729215E+03,
            1.29337404038953400E+03,
        ],
        [
            5.52860084164518747E+05,
            1.19804132404135779E+08,
            3.49579776455820650E+07,
            9.33961835502814618E+05,
            1.09625476549342875E+04,
            4.97454311032222049E-15,
            2.68642785625980869E+04,
            4.88997423297139723E+06,
            2.80544483712566507E+04,
            1.36186802078492315E+03,
            1.34738918372970147E+03,
        ],
        [
            1.37548758418628516E+14,
            0.00000000000000000E+00,
            2.04968704429075456E+19,
            5.49869543371880755E+18,
            2.45173315802838592E+17,
            1.26106566111855011E-03,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            1.02731846489999998E+03,
            2.81464757663215607E+02,
        ],
        [
            4.42744258767709297E+13,
            0.00000000000000000E+00,
            6.59756715773731123E+18,
            1.76992934140618854E+18,
            7.89167995572748480E+16,
            4.05913937579917825E-04,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            1.02731846489999998E+03,
            2.27417980827261800E+02,
        ],
        [
            2.12782875620718823E+12,
            0.00000000000000000E+00,
            3.17079055035404288E+17,
            8.50627980943479040E+16,
            3.79274111680598850E+15,
            1.95082224517562141E-05,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            1.02731846489999998E+03,
            2.37438914587726885E+02,
        ],
        [
            1.41218354559285187E+11,
            0.00000000000000000E+00,
            2.10436964378315880E+16,
            5.64539244337708000E+15,
            2.51714174941122531E+14,
            1.29470901592856755E-06,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            1.02731846489999998E+03,
            2.79555112954127935E+02,
        ],
        [
            1.25488440027266960E+10,
            0.00000000000000000E+00,
            1.87453282921902300E+15,
            4.92305098078476250E+14,
            2.23968541385638359E+13,
            1.14766767151153664E-07,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            0.00000000000000000E+00,
            1.02731846489999998E+03,
            2.19073231364195721E+02,
        ],
        [
            5.19647740297288226E+05,
            1.27449407296046287E+08,
            4.85044986985335723E+07,
            1.72083798257490038E+06,
            2.35448659054442614E+04,
            5.88194044865163260E-15,
            2.50007839108092958E+04,
            6.27920982501879986E+06,
            2.66727320933586780E+04,
            1.42641166228242469E+03,
            1.40860779555326394E+03,
        ],
        [
            4.26085974879412130E+07,
            1.24134201554874313E+11,
            4.92956154248814258E+12,
            1.04840674909283203E+12,
            4.99346508305550461E+10,
            2.91430355030879247E-10,
            8.83122859257159382E+06,
            2.25251550862615462E+05,
            2.41524592964891382E-42,
            1.02731846489999998E+03,
            1.93407106257668147E+02,
        ],
    ];
    // gtd7d total mass density d[5] (anomalous O folded in).
    const GTD7D_RHO_REF: [f64; 17] = [
        4.07542196052162235E-15,
        5.00203049854499384E-15,
        3.38741140603730843E-18,
        3.58442630411333278E-10,
        4.80966382309166367E-15,
        4.35657407040904624E-15,
        2.47135981942753195E-15,
        1.57213101465795068E-15,
        4.56512867312557136E-15,
        4.97528823647096049E-15,
        1.26106566111855011E-03,
        4.05913937579917825E-04,
        1.95082224517562141E-05,
        1.29470901592856755E-06,
        1.14766767151153664E-07,
        5.88264887641603259E-15,
        2.91430355030879247E-10,
    ];

    /// Build the 17 reference test cases (reference `test_gtd7`).
    fn reference_cases() -> ([NrlmsiseInput; 17], usize, usize) {
        let aph: ApArray = [100.0; 7];
        let base = NrlmsiseInput {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt: 400.0,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a: 150.0,
            f107: 150.0,
            ap: 4.0,
            ap_array: None,
        };
        let mut input = [base; 17];
        input[1].doy = 81;
        input[2].sec = 75000.0;
        input[2].alt = 1000.0;
        input[3].alt = 100.0;
        input[10].alt = 0.0;
        input[11].alt = 10.0;
        input[12].alt = 30.0;
        input[13].alt = 50.0;
        input[14].alt = 70.0;
        input[16].alt = 100.0;
        input[4].g_lat = 0.0;
        input[5].g_long = 0.0;
        input[6].lst = 4.0;
        input[7].f107a = 70.0;
        input[8].f107 = 180.0;
        input[9].ap = 40.0;
        input[15].ap_array = Some(aph);
        input[16].ap_array = Some(aph);
        // Cases 0..15 use scalar daily ap (switch 9 = 1); 15..17 use the ap
        // history array (switch 9 = -1).
        (input, 15, 17)
    }

    fn rel_err(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            got.abs()
        } else {
            ((got - want) / want).abs()
        }
    }

    /// Reference-agreement gate: every output of the 17 standard `gtd7` cases
    /// must match Brodowski's release-20041227 oracle to within this relative
    /// bound. The reference oracle was produced at full f64 precision from the
    /// same model logic and tables. The only source of residual is
    /// floating-point operation ordering in the libm `powf`/`exp`/trig calls,
    /// not any model difference: the measured worst-case relative error is
    /// about 2.6e-16 (roughly one f64 ULP). This bound keeps generous margin
    /// for cross-platform libm variation while staying far tighter than any
    /// physical tolerance. Do not loosen it to mask a port defect; a real
    /// defect shifts results by orders of magnitude, not ULPs.
    const REF_TOL: f64 = 1.0e-13;

    #[test]
    fn gtd7_matches_reference_oracle() {
        let (input, n_scalar, n_total) = reference_cases();
        let mut worst = 0.0_f64;
        for (i, inp) in input.iter().enumerate() {
            let mut flags = Flags::standard();
            if i >= n_scalar && i < n_total {
                flags = Flags::new({
                    let mut s = [1i32; 24];
                    s[0] = 0;
                    s[9] = -1;
                    s
                });
            }
            let out = gtd7(inp, &flags).unwrap();
            for j in 0..9 {
                let want = GTD7_REF[i][j];
                let got = out.d[j];
                let e = rel_err(got, want);
                worst = worst.max(e);
                assert!(
                    e <= REF_TOL,
                    "case {i} d[{j}]: got {got:.17E} want {want:.17E} rel {e:.3E}"
                );
            }
            for k in 0..2 {
                let want = GTD7_REF[i][9 + k];
                let got = out.t[k];
                let e = rel_err(got, want);
                worst = worst.max(e);
                assert!(
                    e <= REF_TOL,
                    "case {i} t[{k}]: got {got:.17E} want {want:.17E} rel {e:.3E}"
                );
            }
        }
        assert!(worst <= REF_TOL, "worst relative error {worst:.3E}");
    }

    #[test]
    fn gtd7d_total_density_matches_reference_oracle() {
        let (input, n_scalar, n_total) = reference_cases();
        for (i, inp) in input.iter().enumerate() {
            let mut flags = Flags::standard();
            if i >= n_scalar && i < n_total {
                flags = Flags::new({
                    let mut s = [1i32; 24];
                    s[0] = 0;
                    s[9] = -1;
                    s
                });
            }
            let out = gtd7d(inp, &flags).unwrap();
            let want = GTD7D_RHO_REF[i];
            let e = rel_err(out.d[5], want);
            assert!(
                e <= REF_TOL,
                "case {i} gtd7d rho: got {:.17E} want {want:.17E} rel {e:.3E}",
                out.d[5]
            );
        }
    }

    #[test]
    fn nrlmsise00_metric_units() {
        // Metric convenience wrapper: sea-level total mass density in kg/m^3.
        let input = NrlmsiseInput {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt: 0.0,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a: 150.0,
            f107: 150.0,
            ap: 4.0,
            ap_array: None,
        };
        let out = nrlmsise00(&input).unwrap();
        // Reference case 10 RHO = 1.26106566E-03 g/cm^3 = 1.26106566 kg/m^3.
        // At sea level anomalous oxygen is zero, so gtd7d matches gtd7 here.
        assert!((out.density() - 1.26106566111855011).abs() < 1e-6);
        assert!(out.temperature_alt() > 270.0 && out.temperature_alt() < 290.0);
    }

    #[test]
    fn density_decreases_with_altitude() {
        let make = |alt: f64| NrlmsiseInput {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a: 150.0,
            f107: 150.0,
            ap: 4.0,
            ap_array: None,
        };
        let d0 = nrlmsise00(&make(0.0)).unwrap().density();
        let d200 = nrlmsise00(&make(200.0)).unwrap().density();
        let d400 = nrlmsise00(&make(400.0)).unwrap().density();
        let d800 = nrlmsise00(&make(800.0)).unwrap().density();
        assert!(d0 > d200 && d200 > d400 && d400 > d800);
    }

    #[test]
    fn solar_activity_increases_thermospheric_density() {
        let make = |f107a: f64| NrlmsiseInput {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt: 400.0,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a,
            f107: f107a,
            ap: 4.0,
            ap_array: None,
        };
        assert!(
            nrlmsise00(&make(250.0)).unwrap().density()
                > nrlmsise00(&make(70.0)).unwrap().density()
        );
    }

    #[test]
    fn local_solar_time_wraps() {
        assert!((local_solar_time(43200.0, 0.0) - 12.0).abs() < 0.001);
        assert!((local_solar_time(0.0, 0.0) - 0.0).abs() < 0.001);
        assert!((local_solar_time(0.0, 180.0) - 12.0).abs() < 0.001);
    }

    fn sample_input() -> NrlmsiseInput {
        NrlmsiseInput {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt: 400.0,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a: 150.0,
            f107: 150.0,
            ap: 4.0,
            ap_array: None,
        }
    }

    fn ap_history_flags() -> Flags {
        Flags::new({
            let mut s = [1i32; 24];
            s[0] = 0;
            s[9] = -1;
            s
        })
    }

    #[test]
    fn ap_history_mode_without_array_is_rejected() {
        let input = sample_input(); // ap_array: None
        let flags = ap_history_flags();
        assert_eq!(gtd7(&input, &flags), Err(AtmosphereError::MissingApArray));
        assert_eq!(gtd7d(&input, &flags), Err(AtmosphereError::MissingApArray));
    }

    #[test]
    fn ap_history_mode_with_array_succeeds() {
        let mut input = sample_input();
        input.ap_array = Some([100.0; 7]);
        assert!(gtd7(&input, &ap_history_flags()).is_ok());
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        type Mutate = fn(&mut NrlmsiseInput);
        let cases: [(Mutate, &str); 5] = [
            (|i| i.alt = f64::NAN, "alt"),
            (|i| i.f107 = f64::INFINITY, "f107"),
            (|i| i.f107a = f64::NAN, "f107a"),
            (|i| i.ap = f64::INFINITY, "ap"),
            (|i| i.g_lat = f64::NAN, "g_lat"),
        ];
        for (mutate, name) in cases {
            let mut input = sample_input();
            mutate(&mut input);
            assert_eq!(
                gtd7(&input, &Flags::metric()),
                Err(AtmosphereError::NonFiniteInput(name)),
                "expected non-finite rejection for {name}"
            );
        }
    }

    #[test]
    fn out_of_domain_inputs_are_rejected() {
        let below = {
            let mut i = sample_input();
            i.alt = -1.0;
            i
        };
        assert_eq!(
            gtd7(&below, &Flags::metric()),
            Err(AtmosphereError::OutOfDomain("alt"))
        );

        let above = {
            let mut i = sample_input();
            i.alt = MAX_ALTITUDE_KM + 1.0;
            i
        };
        assert_eq!(
            gtd7(&above, &Flags::metric()),
            Err(AtmosphereError::OutOfDomain("alt"))
        );

        let neg_f107 = {
            let mut i = sample_input();
            i.f107 = -1.0;
            i
        };
        assert_eq!(
            gtd7(&neg_f107, &Flags::metric()),
            Err(AtmosphereError::OutOfDomain("f107"))
        );

        let neg_ap = {
            let mut i = sample_input();
            i.ap = -1.0;
            i
        };
        assert_eq!(
            gtd7(&neg_ap, &Flags::metric()),
            Err(AtmosphereError::OutOfDomain("ap"))
        );
    }

    #[test]
    fn domain_boundaries_are_inclusive() {
        let sea_level = {
            let mut i = sample_input();
            i.alt = 0.0;
            i
        };
        assert!(gtd7(&sea_level, &Flags::metric()).is_ok());

        let top = {
            let mut i = sample_input();
            i.alt = MAX_ALTITUDE_KM;
            i
        };
        assert!(gtd7(&top, &Flags::metric()).is_ok());
    }

    #[test]
    fn nrlmsise00_uses_drag_effective_total_density() {
        // Above ~500 km anomalous oxygen is non-zero, so the drag-effective
        // total (gtd7d, used by nrlmsise00) must exceed the gtd7 total that
        // excludes it.
        let mut input = sample_input();
        input.alt = 800.0;
        let drag = nrlmsise00(&input).unwrap();
        let no_anom = gtd7(&input, &Flags::metric()).unwrap();
        assert!(drag.d[8] > 0.0, "expected non-zero anomalous oxygen");
        assert!(
            drag.density() > no_anom.density(),
            "nrlmsise00 (gtd7d) total {} should exceed gtd7 total {}",
            drag.density(),
            no_anom.density()
        );
    }
}
