//! GNSS attitude laws against Orekit's `org.orekit.gnss.attitude` providers.
//!
//! The fixture `tests/fixtures/attitude/gnss_attitude_orekit.json` comes from
//! `fixtures-generators/orekit_gnss_attitude_oracle/GnssAttitudeOracle.java`
//! (Orekit 13.1.10, Hipparchus 4.0.3; versions and the orekit-data commit are
//! in its `_source`). Each case is a circular or eccentric Keplerian orbit
//! (eccentricity 0.02 for every constellation, 0.1 and 0.3 for GPS) in an
//! inclined plane and a Sun moving uniformly on a circle in the ecliptic, in
//! closed form; this test evaluates the same closed forms from the parameters
//! in the fixture, checks them against the states written there, and feeds
//! them to `gnss_attitude`. Orekit evaluated one provider per case and block
//! at increasing times.
//!
//! Laws: Kouba (2009) and Kouba's `eclips.f` (December 2017) for GPS IIA, IIR
//! and IIF, with Bar-Sever (1996) for the GPS IIA shadow direction and
//! recovery and Dilssner (2010) for the IIF shadow crossing; Dilssner et al.
//! (2011) for GLONASS-M; the Galileo satellite metadata IOV law, also for GPS
//! III with the limits of Montenbruck et al. (2026); the BeiDou-2
//! yaw-steering/orbit-normal switch; BeiDou orbit-normal mode for GEO. The GPS
//! and GLONASS-M turns start at the geometric crossing the papers define,
//! which Orekit estimates from each requested instant instead: those blocks
//! are compared with an independent evaluation of the papers' formulas written
//! here (turn start and end instants to a millisecond off the fixture grid,
//! the yaw inside each turn, the turns counted and matched one to one with
//! sidereon's runs of turn samples), with Orekit at the nominal tolerance
//! outside turns, and inside turns within the drift of Orekit's per-instant
//! estimate from the geometric turn, computed here per sample from Orekit's
//! formulas and bounded on circular orbits by the limits the CHANGELOG states.
//! GPS IIA appears at PRNs 1, 11 and 14 and with a bias of -0.5 deg; where
//! Orekit's `GPSBlockIIA` turns otherwise (its shadow at `+R` whatever the
//! bias, ended where the yaw meets the nominal yaw inside the shadow, and kept
//! at `+R` after it) only the papers' evaluation applies. Galileo IOV and GPS
//! III are compared with the law written out here with their limits (15 deg
//! and 2 deg, and 15 deg and 5.8 deg), and with Orekit's `Galileo` where its
//! narrower region (and, on circular orbits, its `beta_y`) agrees; Orekit fixes
//! `beta_x` at 15 deg and takes `beta_y = atan(mu_rate / yaw_rate)`, so the
//! generator gives each series the yaw rate that makes it 2 deg or 5.8 deg on
//! its circular orbits.
//! BeiDou-2 is compared with the quadrature switch rule written out here and
//! with Orekit, on sweeps of beta through both limits falling and rising, of
//! either sign. For Galileo FOC and BeiDou-3, Orekit applies its Galileo (IOV)
//! law while sidereon applies the published FOC (Galileo satellite metadata
//! 3.1.2), CAST (Wang et al. 2018) and SECM (CSNO 2019) laws. Those blocks are
//! compared with an independent evaluation of the published law written here
//! from the closed forms (the window start and end instants, checked to a
//! millisecond off the fixture grid, and the yaw inside each window), and with
//! Orekit wherever Orekit's law is nominal; the windows are counted and
//! matched one to one with sidereon's runs of turn samples.

#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::astro::attitude::gnss::{
    gnss_attitude, gps_iia_default_yaw_rate_rad_s, AttitudeMode, AttitudeSample, GnssAttitude,
    GnssAttitudeLaw, GnssBlock, BEIDOU2_ON_TO_YS_BETA_RAD, BEIDOU2_YS_TO_ON_BETA_RAD,
    BEIDOU3_CAST_IGSO_TURN, BEIDOU3_CAST_MEO_TURN, BEIDOU3_SECM_BETA0_RAD, GALILEO_FOC_TURN,
};
use std::f64::consts::{PI, TAU};
use std::path::PathBuf;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/attitude/gnss_attitude_orekit.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    serde_json::from_str(&text).expect("fixture JSON")
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn wrap_pi(a: f64) -> f64 {
    let w = a - TAU * ((a + PI) / TAU).floor();
    if w <= -PI {
        w + TAU
    } else {
        w
    }
}

#[derive(Clone, Copy)]
struct Model {
    gm: f64,
    au: f64,
    obliquity: f64,
    sun_rate: f64,
}

/// The closed forms of `GnssAttitudeOracle.java`: a circular orbit
/// (`e = 0`, `u = u0 + n t`) or a Keplerian ellipse (mean anomaly
/// `u0 + n t`, argument of perigee `argp` from the node).
#[derive(Clone, Copy)]
struct Closed {
    model: Model,
    a: f64,
    n: f64,
    p: [f64; 3],
    q: [f64; 3],
    u0: f64,
    e: f64,
    argp: f64,
    lambda0: f64,
}

impl Closed {
    fn new(model: Model, case: &Value) -> Self {
        let orbit = &case["orbit"];
        let a = f(&orbit["a_m"]);
        let i = f(&orbit["inclination_rad"]);
        let raan = f(&orbit["raan_rad"]);
        Self {
            model,
            a,
            n: (model.gm / (a * a * a)).sqrt(),
            p: [libm::cos(raan), libm::sin(raan), 0.0],
            q: [
                -libm::cos(i) * libm::sin(raan),
                libm::cos(i) * libm::cos(raan),
                libm::sin(i),
            ],
            u0: f(&orbit["u0_rad"]),
            e: f(&orbit["e"]),
            argp: f(&orbit["argp_rad"]),
            lambda0: f(&case["sun_lambda0_rad"]),
        }
    }

    fn sample(&self, t: f64) -> AttitudeSample {
        let l = self.lambda0 + self.model.sun_rate * t;
        let (cl, sl) = (libm::cos(l), libm::sin(l));
        let (ce, se) = (
            libm::cos(self.model.obliquity),
            libm::sin(self.model.obliquity),
        );
        let au = self.model.au;
        let sun_position_m = [au * cl, au * sl * ce, au * sl * se];
        if self.e == 0.0 {
            let u = self.u0 + self.n * t;
            let (cu, su) = (libm::cos(u), libm::sin(u));
            let av = self.a * self.n;
            return AttitudeSample {
                position_m: std::array::from_fn(|k| {
                    self.a * cu * self.p[k] + self.a * su * self.q[k]
                }),
                velocity_m_s: std::array::from_fn(|k| -av * su * self.p[k] + av * cu * self.q[k]),
                sun_position_m,
            };
        }
        let (e, gm) = (self.e, self.model.gm);
        let m = self.u0 + self.n * t;
        let mut ea = m + e * libm::sin(m);
        for _ in 0..100 {
            let step = (ea - e * libm::sin(ea) - m) / (1.0 - e * libm::cos(ea));
            ea -= step;
            if step.abs() <= 1e-16 * ea.abs().max(1.0) {
                break;
            }
        }
        let nu = 2.0
            * libm::atan2(
                (1.0 + e).sqrt() * libm::sin(0.5 * ea),
                (1.0 - e).sqrt() * libm::cos(0.5 * ea),
            );
        let u = self.argp + nu;
        let semi_latus = self.a * (1.0 - e * e);
        let radius = semi_latus / (1.0 + e * libm::cos(nu));
        let h = (gm / semi_latus).sqrt();
        let (v_r, v_t) = (h * e * libm::sin(nu), h * (1.0 + e * libm::cos(nu)));
        let (cu, su) = (libm::cos(u), libm::sin(u));
        let e_r: [f64; 3] = std::array::from_fn(|k| cu * self.p[k] + su * self.q[k]);
        let e_t: [f64; 3] = std::array::from_fn(|k| -su * self.p[k] + cu * self.q[k]);
        AttitudeSample {
            position_m: e_r.map(|x| radius * x),
            velocity_m_s: std::array::from_fn(|k| v_r * e_r[k] + v_t * e_t[k]),
            sun_position_m,
        }
    }

    /// Orbital period.
    fn period(&self) -> f64 {
        TAU / self.n
    }
}

fn law_for(block: &str, law: &Value) -> GnssAttitudeLaw {
    let parsed: GnssBlock = block.parse().expect("IGS block name");
    match law["law"].as_str().expect("law name") {
        "GpsIIA" => GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s: f(&law["yaw_rate_rad_s"]),
            yaw_bias_rad: f(&law["yaw_bias_rad"]),
            shadow_limit_rad: f(&law["shadow_limit_rad"]),
        },
        "GpsIIR" => GnssAttitudeLaw::GpsIIR {
            yaw_rate_rad_s: f(&law["yaw_rate_rad_s"]),
        },
        "GpsIIF" => GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s: f(&law["yaw_rate_rad_s"]),
            yaw_bias_rad: f(&law["yaw_bias_rad"]),
            shadow_limit_rad: f(&law["shadow_limit_rad"]),
        },
        "GlonassM" => GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s: f(&law["yaw_rate_rad_s"]),
            shadow_limit_rad: f(&law["shadow_limit_rad"]),
        },
        "GalileoIov" => {
            // The block's own law, with the limits the fixture states (Java's
            // degree-to-radian conversion may differ in the last bit).
            let iov = GnssAttitudeLaw::for_block(parsed, None).expect("IOV law");
            let GnssAttitudeLaw::GalileoIov {
                beta_x_rad,
                beta_y_rad,
            } = iov
            else {
                panic!("{block}: {iov:?} is not the Galileo IOV law");
            };
            assert!(
                (beta_x_rad - f(&law["beta_x_rad"])).abs() <= 1e-16,
                "{block}"
            );
            assert!(
                (beta_y_rad - f(&law["beta_y_rad"])).abs() <= 1e-16,
                "{block}"
            );
            iov
        }
        "GalileoFoc" => {
            let law = GnssAttitudeLaw::for_block(parsed, None).expect("FOC law");
            assert_eq!(law, GnssAttitudeLaw::CosineTurn(GALILEO_FOC_TURN));
            law
        }
        "Beidou3Cast" => {
            let law = GnssAttitudeLaw::for_block(parsed, None).expect("CAST law");
            assert!(
                law == GnssAttitudeLaw::CosineTurn(BEIDOU3_CAST_MEO_TURN)
                    || law == GnssAttitudeLaw::CosineTurn(BEIDOU3_CAST_IGSO_TURN)
            );
            law
        }
        "Beidou3Secm" => {
            let law = GnssAttitudeLaw::for_block(parsed, None).expect("SECM law");
            assert_eq!(
                law,
                GnssAttitudeLaw::FixedBetaYawSteering {
                    beta0_rad: BEIDOU3_SECM_BETA0_RAD
                }
            );
            law
        }
        "Beidou2" => {
            let law = GnssAttitudeLaw::for_block(parsed, None).expect("BeiDou-2 law");
            assert_eq!(
                law,
                GnssAttitudeLaw::BeidouOrbitNormalSwitch {
                    ys_to_on_beta_rad: BEIDOU2_YS_TO_ON_BETA_RAD,
                    on_to_ys_beta_rad: BEIDOU2_ON_TO_YS_BETA_RAD,
                }
            );
            law
        }
        "OrbitNormal" => GnssAttitudeLaw::OrbitNormal,
        other => panic!("unknown law {other}"),
    }
}

/// Orekit's attitude at one instant: yaw, body X, body Y.
struct OrekitAttitude {
    yaw: f64,
    x: [f64; 3],
    y: [f64; 3],
}

fn orekit_row(row: &Value) -> OrekitAttitude {
    let v: Vec<f64> = row.as_array().expect("row").iter().map(f).collect();
    OrekitAttitude {
        yaw: v[0],
        x: [v[1], v[2], v[3]],
        y: [v[4], v[5], v[6]],
    }
}

fn axis_error(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

struct Comparison {
    case: String,
    block: String,
    t: f64,
    yaw_error: f64,
    axis_error: f64,
    rust: GnssAttitude,
    orekit_yaw: f64,
    fresh_yaw: Option<f64>,
    /// Rust axes against the axes Orekit's new-provider yaw defines.
    fresh_axis_error: Option<f64>,
    reference: Reference,
    closed: Closed,
    /// The yaw rate Orekit's Galileo provider was built with, if any.
    orekit_yaw_rate: Option<f64>,
    /// `(beta_x, beta_y)` of a series under the Galileo IOV law.
    iov_limits: Option<(f64, f64)>,
    /// The GPS or GLONASS-M law as the papers state it.
    paper: Option<PaperLaw>,
}

/// Body X and Y for a yaw angle: Z = -e_r, X . n = -sin(yaw), Y . n = -cos(yaw).
fn axes_for_yaw(sample: &AttitudeSample, yaw: f64) -> ([f64; 3], [f64; 3]) {
    let r = sample.position_m;
    let v = sample.velocity_m_s;
    let unit = |a: [f64; 3]| {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let e_r = unit(r);
    let n = unit(cross(r, v));
    let along = cross(n, e_r);
    let (s, c) = (libm::sin(yaw), libm::cos(yaw));
    (
        std::array::from_fn(|k| -s * n[k] + c * along[k]),
        std::array::from_fn(|k| -c * n[k] - s * along[k]),
    )
}

#[test]
fn closed_forms_reproduce_the_fixture_states() {
    let fixture = fixture();
    let model = model(&fixture);
    let mut worst = 0.0_f64;
    for case in fixture["cases"].as_array().expect("cases") {
        let closed = Closed::new(model, case);
        let times = case["times_s"].as_array().expect("times");
        let states = case["states"].as_array().expect("states");
        for (t, state) in times.iter().zip(states) {
            let s = closed.sample(f(t));
            let v: Vec<f64> = state.as_array().unwrap().iter().map(f).collect();
            // Position and Sun relative to their size, velocity to its size:
            // the two closed forms differ only in rounding.
            for k in 0..3 {
                worst = worst.max((s.position_m[k] - v[k]).abs() / closed.a);
                worst = worst.max(
                    (s.velocity_m_s[k] - v[3 + k]).abs() / (closed.a * closed.n * (1.0 + closed.e)),
                );
                worst = worst.max((s.sun_position_m[k] - v[6 + k]).abs() / model.au);
            }
        }
    }
    // The same closed forms evaluated with libm here and FastMath in Java,
    // with Hipparchus' compensated linear combinations, agree to a few units
    // in the last place of the angle arguments at t up to six days (Kepler's
    // equation solved by the same Newton iteration on both sides).
    assert!(worst < 1.0e-13, "closed-form states differ by {worst:e}");
}

fn model(fixture: &Value) -> Model {
    let m = &fixture["model"];
    Model {
        gm: f(&m["gm_m3_s2"]),
        au: f(&m["au_m"]),
        obliquity: f(&m["obliquity_rad"]),
        sun_rate: f(&m["sun_rate_rad_s"]),
    }
}

/// Which law a fixture block is checked against.
#[derive(Clone, Copy, PartialEq)]
enum Reference {
    /// Orekit implements the published law: compare with Orekit.
    Orekit,
    /// Orekit 13.1.10 applies its Galileo law to this block, which is not the
    /// published one: compare with an independent evaluation of the
    /// published law here, and with Orekit outside both laws' windows.
    Published(Published),
}

fn compare_all() -> Vec<Comparison> {
    let fixture = fixture();
    let model = model(&fixture);
    let mut out = Vec::new();
    for case in fixture["cases"].as_array().expect("cases") {
        let closed = Closed::new(model, case);
        let trajectory = move |t: f64| Some(closed.sample(t));
        let times: Vec<f64> = case["times_s"].as_array().unwrap().iter().map(f).collect();
        for block in case["blocks"].as_array().expect("blocks") {
            // The series label (the IGS name, with the PRN or bias where a
            // block appears more than once) and the IGS name.
            let name = block["block"].as_str().unwrap();
            let law = law_for(block["igs_block"].as_str().unwrap(), &block["law"]);
            let fresh: Option<Vec<f64>> = block
                .get("fresh_yaw")
                .map(|v| v.as_array().unwrap().iter().map(f).collect());
            if fresh.is_some() {
                // Orekit's axes equal the axes its yaw defines to this bound,
                // checked when the fixture was written.
                assert!(f(&block["fresh_axes_from_yaw_max_error"]) <= 1.0e-12);
            }
            let reference =
                published_law(name, &block["law"]).map_or(Reference::Orekit, Reference::Published);
            if matches!(reference, Reference::Published(_)) {
                assert_eq!(block["orekit_provider"], "Galileo", "{name}");
            }
            for (i, (t, row)) in times
                .iter()
                .zip(block["attitude"].as_array().unwrap())
                .enumerate()
            {
                let orekit = orekit_row(row);
                let rust = gnss_attitude(&law, &trajectory, *t)
                    .unwrap_or_else(|e| panic!("{} {name} t={t}: {e}", case["name"]));
                let yaw_error = wrap_pi(rust.yaw_rad - orekit.yaw).abs();
                let axes_error =
                    axis_error(rust.body_x, orekit.x).max(axis_error(rust.body_y, orekit.y));
                let fresh_yaw = fresh.as_ref().map(|v| v[i]);
                let fresh_axis_error = fresh_yaw.map(|yaw| {
                    let (x, y) = axes_for_yaw(&closed.sample(*t), yaw);
                    axis_error(rust.body_x, x).max(axis_error(rust.body_y, y))
                });
                out.push(Comparison {
                    case: case["name"].as_str().unwrap().to_string(),
                    block: name.to_string(),
                    t: *t,
                    yaw_error,
                    axis_error: axes_error,
                    rust,
                    orekit_yaw: orekit.yaw,
                    fresh_yaw,
                    fresh_axis_error,
                    reference,
                    closed,
                    orekit_yaw_rate: block["law"].get("orekit_yaw_rate_rad_s").map(f),
                    iov_limits: match law {
                        GnssAttitudeLaw::GalileoIov {
                            beta_x_rad,
                            beta_y_rad,
                        } => Some((beta_x_rad, beta_y_rad)),
                        _ => None,
                    },
                    paper: paper_law(&block["law"]),
                });
            }
        }
    }
    out
}

/// Nominal yaw steering, orbit-normal mode and every sample where both sides
/// are nominal: sidereon forms the nominal axes with RTKLIB's operation order
/// (beta and the orbit angle by `acos`, then `atan2`, sine and cosine) and
/// Orekit with cross products and normalisation; the two agree to a few units
/// in the last place of yaw angles up to pi.
const NOMINAL_TOLERANCE_RAD: f64 = 1.0e-13;

/// Galileo IOV: the in-plane angle to the turn centre is an `acos` of a
/// value close to 1 near the centre on both sides, which amplifies rounding by
/// `1 / sqrt(1 - x^2)`.
const GALILEO_IOV_TOLERANCE_RAD: f64 = 1.0e-10;

/// How far Orekit's per-instant turn estimate ([`orekit_estimate`]) moves from
/// the geometric turn of the papers ([`paper_turn`]) over every fixture sample
/// where Orekit turns: the start instant, and the yaw drift (the start yaw,
/// plus the rate difference times the time since the estimated start, which
/// the GPS IIF shadow rate formed from the estimated span adds, and for the
/// GLONASS-M shadow the held end yaw). Orekit takes beta and the half span at
/// the requested instant and extrapolates back to the start at that instant's
/// orbital rate, so its start moves with the Sun's in-plane motion and with
/// beta, and it forms the start yaw with a beta up to a turn later; its GPS
/// noon and IIR midnight half span is Kouba's small-angle form, which starts
/// up to 0.85 s after the nominal-rate crossing. The worst seen is 3.875 s and
/// 6.73e-3 rad, both GPS IIF; the CHANGELOG states these limits.
const OREKIT_START_DRIFT_S: f64 = 4.0;
const OREKIT_YAW_DRIFT_RAD: f64 = 7.0e-3;

/// What the estimate written here leaves of Orekit's per-instant yaw: GPS,
/// rounding of the same formulas (a few 1e-12 rad seen); GLONASS-M, Orekit's
/// Brent solve of the noon-turn end yaw to 1e-8 rad absolute accuracy, which
/// moves its half span and start yaw by up to 1e-8 rad.
const GPS_ESTIMATE_RESIDUAL_RAD: f64 = 1.0e-10;
const GLONASS_ESTIMATE_RESIDUAL_RAD: f64 = 1.0e-8;

/// sidereon against Orekit's per-instant yaw where Orekit turns, from the
/// drift of Orekit's estimate `e` from the geometric turn `w` at `t`. Both are
/// linear in time, so sidereon - Orekit =
/// `(psi0 - psi0') + rate (start' - start) - (rate' - rate) (t - start')`; a
/// held shadow yaw (GLONASS-M) differs by at most the larger of that and the
/// two held yaws' difference. Where sidereon is nominal (outside its turn), its
/// distance from its own extended line is added.
fn drift_bound(w: &PaperTurn, e: &OrekitEstimate, t: f64) -> f64 {
    let linear = wrap_pi(e.psi0 - w.psi0).abs()
        + w.rate.abs() * (e.start - w.start).abs()
        + (e.rate - w.rate).abs() * (t - e.start).abs();
    match (e.hold, w.hold) {
        (Some(a), Some(b)) => linear.max(wrap_pi(a - b).abs()),
        _ => linear,
    }
}

/// The yaw part of the drift: everything in [`drift_bound`] except the start
/// instant's.
fn yaw_drift(w: &PaperTurn, e: &OrekitEstimate, t: f64) -> f64 {
    let linear = wrap_pi(e.psi0 - w.psi0).abs() + (e.rate - w.rate).abs() * (t - e.start).abs();
    match (e.hold, w.hold) {
        (Some(a), Some(b)) => linear.max(wrap_pi(a - b).abs()),
        _ => linear,
    }
}

/// Orekit's sequential provider (one provider called at increasing times)
/// continues a turn past its trigger zone from the span it cached at an
/// earlier call: a per-instant estimate made earlier in the same turn. Its
/// difference from sidereon is bounded by the largest drift of any such
/// estimate: `OREKIT_YAW_DRIFT_RAD + R x OREKIT_START_DRIFT_S`, at the turn's
/// maximum yaw rate `R`.
fn sequential_bound(rate: f64) -> f64 {
    OREKIT_YAW_DRIFT_RAD + rate * OREKIT_START_DRIFT_S
}

/// Orekit's sequential provider continues a turn until its cached end plus
/// its 1800 s margin while the linear yaw has not met the nominal yaw; after
/// sidereon has reached the nominal yaw, Orekit's restarted or overrun turn
/// lasts at most that margin plus the turn's 3000 s (GPS IIA) span.
const MAX_SEQUENTIAL_ONLY_RUN_S: f64 = 1800.0 + 3000.0;

/// At most this many consecutive samples (at the 20 s dense step) where
/// Orekit's sequential provider has reached the nominal yaw and sidereon has
/// not: the two linear yaws differ by the drift of the cached estimate, so
/// their reach instants differ by seconds on these orbits (none occur here).
const MAX_SIDEREON_LATER_REACH_RUN: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Category {
    /// Orekit with a new provider is in a turn.
    FreshTurn,
    /// Nominal on every side (or orbit normal on both).
    Nominal,
    /// sidereon in a turn past Orekit's trigger zone, where Orekit's sequential
    /// provider continues the same turn from its cached span.
    CachedSpan,
    /// sidereon still in a turn after Orekit's sequential provider reached
    /// the nominal yaw.
    SidereonLaterReach,
    /// GLONASS-M noon turn between the turn start and Orekit's trigger zone.
    BeforeOrekitZone,
    /// Only Orekit's sequential provider is turning; sidereon and Orekit with a
    /// new provider are nominal.
    SequentialOnly,
    /// Blocks Orekit evaluates without cached spans, compared directly.
    Direct,
    /// GPS IIA where Orekit's `GPSBlockIIA` and the published law turn
    /// differently (a negative yaw bias, or the recovery after the shadow
    /// taking the shorter way against Orekit's `+R`): compared with the
    /// published law in `gps_and_glonass_turns_follow_the_papers`.
    PublishedOnly,
    /// The Galileo IOV law (Galileo IOV, GPS III) inside its region but outside
    /// Orekit's, or on an eccentric orbit (where Orekit's `beta_y`, `atan(mu_rate / yaw_rate)`,
    /// moves with the orbital rate): compared with the metadata's law written
    /// out here.
    IovRegionOnly,
    /// BeiDou-2 where Orekit's `BeidouMeo` takes its band shortcut to orbit
    /// normal: compared with the quadrature rule written out here.
    BeidouBand,
    /// GPS and GLONASS-M on eccentric orbits where Orekit's sequential
    /// provider or its per-instant estimate is in a turn other than the
    /// geometric one: Orekit tests a zone's existence and estimates its span
    /// with the requested instant's orbital rate, which varies over the arc,
    /// so near the existence limit `atan(mu_rate / R)` it opens or misses
    /// turns, and its cached spans carry that estimate. Compared with the
    /// papers' turns in `gps_and_glonass_turns_follow_the_papers`.
    EccentricEstimate,
}

/// Largest start drift of Orekit's per-instant estimate on an eccentric orbit
/// for which it still describes the geometric turn: beyond it the estimate
/// belongs to a turn the geometry places elsewhere or does not have.
const ECCENTRIC_SAME_TURN_S: f64 = 120.0;

/// The window of a GPS IIA paper turn in which Orekit's `GPSBlockIIA`
/// (shadow at `+R` whatever the bias, ended where the yaw meets the nominal
/// yaw also inside the shadow, the same rate after the shadow, noon turns
/// reversed for `0 < beta < bias` only) turns differently from the published
/// law, widened by Orekit's start drift (and a sample step) before and by its
/// longest sequential run after.
fn orekit_iia_differs(w: &PaperTurn, law: PaperLaw) -> Option<(f64, f64)> {
    let PaperLaw::GpsIia { rate, bias, .. } = law else {
        return None;
    };
    let after = w.zone_end + MAX_SEQUENTIAL_ONLY_RUN_S;
    let before = w.start - 2.0 * OREKIT_START_DRIFT_S;
    if w.noon {
        let orekit_rate = if w.beta > 0.0 && w.beta < bias {
            rate.copysign(w.beta)
        } else {
            -rate.copysign(w.beta)
        };
        (orekit_rate.signum() != w.rate.signum()).then_some((before, after))
    } else if bias < 0.0 {
        Some((before, after))
    } else if let Some(met) = w.met_in_shadow {
        Some((met - 2.0 * OREKIT_START_DRIFT_S - 60.0, after))
    } else {
        (w.after < 0.0).then_some((w.zone_end - 1.0, after))
    }
}

fn categorise(c: &Comparison) -> Category {
    let Some(fresh) = c.fresh_yaw else {
        return Category::Direct;
    };
    let fresh_turn = wrap_pi(fresh - c.rust.nominal_yaw_rad).abs() > 1e-9;
    let sequential_turn = wrap_pi(c.orekit_yaw - c.rust.nominal_yaw_rad).abs() > 1e-9;
    let rust_turn = c.rust.mode != AttitudeMode::Nominal;
    match (rust_turn, fresh_turn, sequential_turn) {
        (_, true, _) => Category::FreshTurn,
        (false, false, false) => Category::Nominal,
        (true, false, true) => Category::CachedSpan,
        (true, false, false)
            if c.block == "GLONASS-M"
                && c.rust.mode == AttitudeMode::NoonTurn
                && c.rust.orbit_angle_rad > 0.0 =>
        {
            Category::BeforeOrekitZone
        }
        (true, false, false) => Category::SidereonLaterReach,
        (false, false, true) => Category::SequentialOnly,
    }
}

fn glonass_noon_half_span(beta: f64, mu_rate: f64, rate: f64) -> f64 {
    // The positive root of y = |phi(beta, d) - phi(beta, -d)| / 2,
    // d = mu_rate y / rate (Dilssner et al. 2011), by bisection.
    let half_swing = |y: f64| {
        let d = mu_rate * y / rate;
        0.5 * (libm::atan2(-libm::tan(beta), libm::sin(d))
            - libm::atan2(-libm::tan(beta), -libm::sin(d)))
        .abs()
    };
    let (mut lo, mut hi) = (1e-12, PI);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid - half_swing(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    mu_rate * 0.5 * (lo + hi) / rate
}

/// The paper-law turns of every GPS and GLONASS-M series, keyed by case and
/// block.
fn paper_turns_by_series(
    all: &[Comparison],
) -> std::collections::HashMap<(String, String), (PaperLaw, Vec<PaperTurn>)> {
    let mut map = std::collections::HashMap::new();
    for series in all.chunk_by(|a, b| a.case == b.case && a.block == b.block) {
        let first = &series[0];
        if let Some(law) = first.paper {
            let turns = paper_turns(&first.closed, law, first.t, series.last().unwrap().t);
            map.insert((first.case.clone(), first.block.clone()), (law, turns));
        }
    }
    map
}

/// The turn of the pass nearest `t`: the one whose span `[start, end]` is
/// closest.
fn nearest_turn(turns: &[PaperTurn], t: f64) -> &PaperTurn {
    let distance = |w: &PaperTurn| (w.start - t).max(t - w.end()).max(0.0);
    turns
        .iter()
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))
        .expect("a turn")
}

#[test]
fn attitude_matches_orekit() {
    let all: Vec<Comparison> = compare_all()
        .into_iter()
        .filter(|c| c.reference == Reference::Orekit)
        .collect();
    let turns = paper_turns_by_series(&all);
    // The first instant of each series: Orekit's providers continue a turn
    // only from spans cached at earlier calls, so a turn under way at a
    // series' first instant is no turn to them.
    let mut first_t: std::collections::HashMap<(String, String), f64> = Default::default();
    for c in &all {
        first_t
            .entry((c.case.clone(), c.block.clone()))
            .or_insert(c.t);
    }
    let categories: Vec<Category> = all
        .iter()
        .map(|c| {
            let key = (c.case.clone(), c.block.clone());
            let differs = turns.get(&key).is_some_and(|(law, ws)| {
                ws.iter().any(|w| {
                    let begun_before = w.start < first_t[&key] && w.end() >= first_t[&key];
                    let window = if begun_before {
                        Some((first_t[&key], w.end() + MAX_SEQUENTIAL_ONLY_RUN_S))
                    } else {
                        orekit_iia_differs(w, *law)
                    };
                    window.is_some_and(|(a, b)| a <= c.t && c.t <= b)
                })
            });
            if differs {
                return Category::PublishedOnly;
            }
            let category = categorise(c);
            if c.closed.e == 0.0 || matches!(category, Category::Nominal | Category::Direct) {
                return category;
            }
            if category == Category::FreshTurn {
                if let Some((law, ws)) = turns.get(&key) {
                    let w = nearest_turn(ws, c.t);
                    let e = orekit_estimate(&c.closed, *law, w.noon, c.t);
                    if (e.start - w.start).abs() <= ECCENTRIC_SAME_TURN_S {
                        return category;
                    }
                }
            }
            Category::EccentricEstimate
        })
        .collect();
    // Orekit's drift from the geometric turn over each eccentric series (its
    // per-instant estimate takes the requested instant's orbital rate, which
    // varies there): the bound its sequential provider is held to in that
    // series. Circular series are held to the limits the CHANGELOG states.
    let mut series_drift: std::collections::HashMap<(String, String), (f64, f64)> =
        Default::default();
    for (c, &category) in all.iter().zip(&categories) {
        if category != Category::FreshTurn || c.closed.e == 0.0 {
            continue;
        }
        let (law, ws) = &turns[&(c.case.clone(), c.block.clone())];
        let w = nearest_turn(ws, c.t);
        let e = orekit_estimate(&c.closed, *law, w.noon, c.t);
        let entry = series_drift
            .entry((c.case.clone(), c.block.clone()))
            .or_default();
        entry.0 = entry.0.max((e.start - w.start).abs());
        entry.1 = entry.1.max(yaw_drift(w, &e, c.t));
    }
    // (block, category) -> (count, worst yaw error, worst axis error, where)
    let mut summary: std::collections::BTreeMap<(String, Category), (usize, f64, f64, String)> =
        Default::default();
    // block -> (worst start drift s, worst yaw drift rad, worst error / bound)
    let mut drifts: std::collections::BTreeMap<String, (f64, f64, f64)> = Default::default();
    let same_series =
        |i: usize, j: usize| all[i].case == all[j].case && all[i].block == all[j].block;
    // Index of the last sample of a turn both sides share, and the start of
    // the current run of SequentialOnly or SidereonLaterReach samples.
    let mut last_shared_turn: Option<usize> = None;
    let mut run_start: Option<usize> = None;
    // Galileo IOV law samples in the region compared with Orekit, by block.
    let mut iov_with_orekit: std::collections::BTreeMap<String, usize> = Default::default();

    for (index, (c, &category)) in all.iter().zip(&categories).enumerate() {
        let previous = (index > 0 && same_series(index - 1, index)).then(|| categories[index - 1]);
        if previous.is_none() {
            last_shared_turn = None;
        }
        if previous != Some(category) {
            run_start = Some(index);
        }
        let run_length = index - run_start.unwrap() + 1;
        let where_ = format!("{} {} t={}", c.case, c.block, c.t);
        let series_turns = turns.get(&(c.case.clone(), c.block.clone()));
        let residual = if c.block == "GLONASS-M" {
            GLONASS_ESTIMATE_RESIDUAL_RAD
        } else {
            GPS_ESTIMATE_RESIDUAL_RAD
        };
        // Compare against the reference that defines each category.
        let (yaw_error, axis_error, tolerance, axis_tolerance) = match category {
            Category::FreshTurn => {
                let (law, turns) = series_turns.expect("paper turns");
                let w = nearest_turn(turns, c.t);
                let e = orekit_estimate(&c.closed, *law, w.noon, c.t);
                if c.closed.e == 0.0 {
                    let drift = drifts.entry(c.block.clone()).or_default();
                    drift.0 = drift.0.max((e.start - w.start).abs());
                    drift.1 = drift.1.max(yaw_drift(w, &e, c.t));
                    assert!(
                        (e.start - w.start).abs() <= OREKIT_START_DRIFT_S
                            && yaw_drift(w, &e, c.t) <= OREKIT_YAW_DRIFT_RAD,
                        "{where_}: Orekit's estimate drifts {} s, {} rad",
                        e.start - w.start,
                        yaw_drift(w, &e, c.t)
                    );
                }
                let off_line = if w.contains(c.t) {
                    0.0
                } else {
                    let line = w.psi0 + w.rate * (c.t - w.start);
                    wrap_pi(c.rust.nominal_yaw_rad - line).abs()
                };
                let bound = drift_bound(w, &e, c.t) + off_line;
                let error = wrap_pi(c.rust.yaw_rad - c.fresh_yaw.unwrap()).abs();
                if c.closed.e == 0.0 {
                    let drift = drifts.entry(c.block.clone()).or_default();
                    drift.2 = drift.2.max(error / (bound + residual));
                }
                // Axes rebuilt from Orekit's yaw differ from Orekit's own axes by
                // up to 1e-12 (checked when the fixture was written).
                (
                    error,
                    c.fresh_axis_error.unwrap(),
                    bound + residual,
                    bound + residual + 1.0e-12,
                )
            }
            Category::Nominal => (
                c.yaw_error,
                c.axis_error,
                NOMINAL_TOLERANCE_RAD,
                NOMINAL_TOLERANCE_RAD,
            ),
            Category::CachedSpan | Category::SidereonLaterReach => {
                // Past Orekit's zone, continuing a turn both share.
                assert!(
                    matches!(
                        previous,
                        Some(
                            Category::FreshTurn
                                | Category::CachedSpan
                                | Category::SidereonLaterReach
                        )
                    ),
                    "{where_}: {category:?} sample not continuing a shared turn ({previous:?})"
                );
                if category == Category::SidereonLaterReach {
                    assert!(
                        run_length <= MAX_SIDEREON_LATER_REACH_RUN,
                        "{where_}: {run_length} samples where only sidereon is turning"
                    );
                }
                let (law, _) = series_turns.expect("paper turns");
                let bound = match series_drift.get(&(c.case.clone(), c.block.clone())) {
                    Some(&(start, yaw)) => yaw + law.rate() * start,
                    None => sequential_bound(law.rate()),
                } + residual;
                (c.yaw_error, c.axis_error, bound, bound)
            }
            Category::SequentialOnly => {
                // sidereon is nominal and so is Orekit with a new provider:
                // compare the two sharply. Orekit's sequential provider is
                // still in a turn it continues from its cached span (restarted
                // from a new start yaw, or run past nominal); such a run
                // follows a turn both sides share, starting at most two samples
                // after its last shared sample, and lasts at most the cached
                // end plus Orekit's margin.
                let follows_turn = previous == Some(Category::SequentialOnly)
                    || last_shared_turn.is_some_and(|last| index - last <= 2);
                assert!(
                    follows_turn,
                    "{where_}: Orekit turn without a preceding turn"
                );
                let run_s = c.t - all[run_start.unwrap()].t;
                assert!(
                    run_s <= MAX_SEQUENTIAL_ONLY_RUN_S,
                    "{where_}: Orekit's sequential turn runs {run_s} s past sidereon's"
                );
                let error = wrap_pi(c.rust.yaw_rad - c.fresh_yaw.unwrap()).abs();
                (
                    error,
                    c.fresh_axis_error.unwrap(),
                    NOMINAL_TOLERANCE_RAD,
                    NOMINAL_TOLERANCE_RAD + 1.0e-12,
                )
            }
            Category::BeforeOrekitZone => {
                // Which samples: inside the geometric noon turn, but the
                // Sun-satellite angle is still above the half span, Orekit's
                // trigger zone (`setUpTurnRegion` with cos(a_noon)).
                let (law, turns) = series_turns.expect("paper turns");
                let w = nearest_turn(turns, c.t);
                assert!(w.noon && w.contains(c.t), "{where_}: outside the noon turn");
                let sample = c.closed.sample(c.t);
                let rate = law.rate();
                let half_span = glonass_noon_half_span(c.rust.beta_rad, c.closed.n, rate);
                let sun_angle = libm::acos(cos_sun_satellite(&sample));
                assert!(
                    sun_angle > half_span,
                    "{where_}: Sun angle {sun_angle} within the half span {half_span}"
                );
                // Yaw: the next sample is in Orekit's zone; Orekit's line there
                // has sidereon's rate, so carried back to this sample it
                // differs from sidereon by the drift bound at that sample.
                let next = &all[index + 1];
                assert!(same_series(index, index + 1));
                assert_eq!(categories[index + 1], Category::FreshTurn, "{where_}");
                let e = orekit_estimate(&c.closed, *law, true, next.t);
                assert_eq!(e.rate, w.rate, "{where_}");
                let expected = next.fresh_yaw.unwrap() - w.rate * (next.t - c.t);
                let tolerance = drift_bound(w, &e, next.t) + residual;
                let (x, y) = axes_for_yaw(&sample, expected);
                (
                    wrap_pi(c.rust.yaw_rad - expected).abs(),
                    axis_error(c.rust.body_x, x).max(axis_error(c.rust.body_y, y)),
                    tolerance,
                    tolerance + AXIS_ROUNDING_RAD,
                )
            }
            Category::Direct if c.iov_limits.is_some() => {
                // The Galileo IOV law with the series' limits (Galileo
                // satellite metadata 3.1.1; Montenbruck et al. 2026 for GPS
                // III), written out here; where Orekit's narrower region also
                // applies, Orekit too.
                let (beta_x, beta_y) = c.iov_limits.unwrap();
                let sample = c.closed.sample(c.t);
                let orekit_in_region =
                    wrap_pi(c.orekit_yaw - geometry(&sample).nominal_yaw).abs() > 1e-9;
                match iov_yaw(&c.closed, c.t, beta_x, beta_y) {
                    Some(yaw) => {
                        assert_ne!(c.rust.mode, AttitudeMode::Nominal, "{where_}");
                        let error = wrap_pi(c.rust.yaw_rad - yaw).abs();
                        assert!(
                            error <= GALILEO_IOV_TOLERANCE_RAD,
                            "{where_}: IOV law yaw error {error:e}"
                        );
                        // Orekit's `Galileo` takes `sin(beta_y)` as
                        // `sin(atan(mu_rate / yaw_rate))`; the generator's yaw
                        // rate makes that `beta_y` at a circular orbit's rate,
                        // and on an eccentric orbit it moves with the rate.
                        if orekit_in_region && c.closed.e == 0.0 {
                            *iov_with_orekit.entry(c.block.clone()).or_default() += 1;
                            (
                                c.yaw_error,
                                c.axis_error,
                                GALILEO_IOV_TOLERANCE_RAD,
                                GALILEO_IOV_TOLERANCE_RAD,
                            )
                        } else {
                            let (x, y) = axes_for_yaw(&sample, yaw);
                            let axes =
                                axis_error(c.rust.body_x, x).max(axis_error(c.rust.body_y, y));
                            (
                                error,
                                axes,
                                GALILEO_IOV_TOLERANCE_RAD,
                                GALILEO_IOV_TOLERANCE_RAD + AXIS_ROUNDING_RAD,
                            )
                        }
                    }
                    None => {
                        assert_eq!(c.rust.mode, AttitudeMode::Nominal, "{where_}");
                        assert!(
                            !orekit_in_region,
                            "{where_}: Orekit beyond the IOV law's region"
                        );
                        (
                            c.yaw_error,
                            c.axis_error,
                            NOMINAL_TOLERANCE_RAD,
                            NOMINAL_TOLERANCE_RAD,
                        )
                    }
                }
            }
            Category::Direct if c.block.starts_with("BEIDOU-2") && c.block != "BEIDOU-2G" => {
                // The quadrature rule written out here decides the mode; where
                // Orekit agrees, Orekit is the reference for the attitude.
                let orbit_normal = beidou2_orbit_normal(&c.closed, c.t);
                let expected = if orbit_normal {
                    AttitudeMode::OrbitNormal
                } else {
                    AttitudeMode::Nominal
                };
                assert_eq!(c.rust.mode, expected, "{where_}");
                if (c.orekit_yaw.abs() < 1e-9) == orbit_normal {
                    (
                        c.yaw_error,
                        c.axis_error,
                        NOMINAL_TOLERANCE_RAD,
                        NOMINAL_TOLERANCE_RAD,
                    )
                } else {
                    let sample = c.closed.sample(c.t);
                    let yaw = if orbit_normal {
                        0.0
                    } else {
                        geometry(&sample).nominal_yaw
                    };
                    let (x, y) = axes_for_yaw(&sample, yaw);
                    (
                        wrap_pi(c.rust.yaw_rad - yaw).abs(),
                        axis_error(c.rust.body_x, x).max(axis_error(c.rust.body_y, y)),
                        INDEPENDENT_NOMINAL_TOLERANCE_RAD,
                        INDEPENDENT_NOMINAL_TOLERANCE_RAD + AXIS_ROUNDING_RAD,
                    )
                }
            }
            Category::Direct => (
                c.yaw_error,
                c.axis_error,
                NOMINAL_TOLERANCE_RAD,
                NOMINAL_TOLERANCE_RAD,
            ),
            Category::PublishedOnly
            | Category::IovRegionOnly
            | Category::BeidouBand
            | Category::EccentricEstimate => (0.0, 0.0, 0.0, 0.0),
        };
        // Recategorise the samples where Orekit's narrower law applies
        // differently, now checked against the published law.
        let category = match category {
            Category::Direct
                if c.iov_limits.is_some()
                    && c.rust.mode != AttitudeMode::Nominal
                    && (c.closed.e != 0.0
                        || wrap_pi(c.orekit_yaw - geometry(&c.closed.sample(c.t)).nominal_yaw)
                            .abs()
                            <= 1e-9) =>
            {
                Category::IovRegionOnly
            }
            Category::Direct
                if c.block.starts_with("BEIDOU-2")
                    && c.block != "BEIDOU-2G"
                    && (c.orekit_yaw.abs() < 1e-9)
                        != (c.rust.mode == AttitudeMode::OrbitNormal) =>
            {
                // Orekit's shortcut: orbit normal for |beta| within the band
                // between the two limits widened by the last orbit's change.
                assert!(
                    c.orekit_yaw.abs() < 1e-9 && beidou2_in_orekit_band(&c.closed, c.t),
                    "{where_}: Orekit's BeiDou-2 mode differs outside its band shortcut"
                );
                Category::BeidouBand
            }
            other => other,
        };
        assert!(
            yaw_error <= tolerance && axis_error <= axis_tolerance,
            "{where_} {category:?}: yaw error {yaw_error:e} (tolerance {tolerance:e}), axis error {axis_error:e} (sidereon {:?} yaw {}, Orekit {}, Orekit with a new provider {:?})",
            c.rust.mode,
            c.rust.yaw_rad,
            c.orekit_yaw,
            c.fresh_yaw
        );
        if matches!(category, Category::FreshTurn | Category::CachedSpan) {
            last_shared_turn = Some(index);
        }

        let entry =
            summary
                .entry((c.block.clone(), category))
                .or_insert((0, 0.0, 0.0, String::new()));
        entry.0 += 1;
        if yaw_error > entry.1 {
            entry.1 = yaw_error;
            entry.3 = format!("{} t={}", c.case, c.t);
        }
        entry.2 = entry.2.max(axis_error);
    }

    for ((block, category), (count, yaw, axis, at)) in &summary {
        println!(
            "{block:16} {category:?}: {count} samples, max yaw error {yaw:.3e} rad, max axis error {axis:.3e} at {at}"
        );
    }
    for (block, (start, yaw, ratio)) in &drifts {
        println!(
            "{block:16} Orekit's per-instant estimate drifts up to {start:.3} s and {yaw:.3e} rad; worst error / drift bound {ratio:.3}"
        );
    }

    // Every block has a nominal comparison, and the turn laws have turns.
    for block in ["BLOCK IIA", "BLOCK IIR-B", "BLOCK IIF", "GLONASS-M"] {
        let turns = summary
            .get(&(block.to_string(), Category::FreshTurn))
            .map_or(0, |s| s.0);
        assert!(turns > 100, "{block}: {turns} turn samples");
    }
    for (block, least) in [("GALILEO-1", 50), ("BLOCK IIIA", 500)] {
        let turns = all
            .iter()
            .filter(|c| c.block == block && c.rust.mode != AttitudeMode::Nominal)
            .count();
        let with_orekit = iov_with_orekit.get(block).copied().unwrap_or(0);
        println!(
            "{block:16} {turns} samples in the IOV law's region, {with_orekit} of them in Orekit's"
        );
        assert!(turns > least, "{block}: {turns} samples in the region");
        assert!(
            with_orekit > least / 2,
            "{block}: {with_orekit} region samples against Orekit"
        );
    }
    let on = all
        .iter()
        .filter(|c| c.block == "BEIDOU-2M" && c.rust.mode == AttitudeMode::OrbitNormal)
        .count();
    assert!(on > 50, "BeiDou-2 MEO: {on} orbit-normal samples");
    // Coverage: the bias of the other sign, the metadata's IOV region beyond
    // Orekit's, both BeiDou-2 modes in every switch sweep (|beta| falling and
    // rising, of either sign), and eccentric orbits compared with Orekit.
    let count = |block: &str, category: Category| {
        summary
            .get(&(block.to_string(), category))
            .map_or(0, |s| s.0)
    };
    assert!(count("BLOCK IIA PRN 23 BIAS -0.5 DEG", Category::PublishedOnly) > 500);
    assert!(count("GALILEO-1", Category::IovRegionOnly) > 50);
    assert!(count("BLOCK IIIA", Category::IovRegionOnly) > 20);
    for case in [
        "bds_meo_mode_switch",
        "bds_meo_mode_switch_rising",
        "bds_meo_mode_switch_negative",
        "bds_meo_mode_switch_negative_rising",
        "bds_igso_mode_switch",
        "bds_igso_mode_switch_rising",
        "bds_igso_mode_switch_negative",
        "bds_igso_mode_switch_negative_rising",
    ] {
        for mode in [AttitudeMode::OrbitNormal, AttitudeMode::Nominal] {
            let n = all
                .iter()
                .filter(|c| {
                    c.case == case && c.block.starts_with("BEIDOU-2") && c.rust.mode == mode
                })
                .count();
            assert!(n > 20, "{case}: {n} samples in {mode:?}");
        }
    }
    let eccentric = all
        .iter()
        .zip(&categories)
        .filter(|(c, category)| {
            c.closed.e > 0.0
                && matches!(
                    category,
                    Category::Nominal | Category::FreshTurn | Category::Direct
                )
        })
        .count();
    assert!(
        eccentric > 5000,
        "{eccentric} eccentric samples against Orekit"
    );
}

/// Paper-law turn yaw against sidereon: both solve the turn start (and end) on
/// the trajectory, sidereon to the last bit and the evaluation here to 1e-9 s.
/// The yaw moves with the start instant by the nominal yaw rate there minus
/// the turn rate: zero at a nominal-rate crossing, otherwise at most
/// `2 x 0.25 deg/s`, 8.7e-12 rad over 1e-9 s; the GPS IIF shadow rate moves by
/// the two boundary errors over a shadow of at least 1000 s, under 1e-11 rad
/// over a 3000 s turn. Beta, the orbit angle and the nominal yaw are formed
/// here with `asin` and `atan2` and in sidereon from vector angles, a few
/// units in the last place, which the yaw sensitivity near the centre
/// (`1 / sin(half span)`, about 10^2) raises to a few 1e-14 rad.
const PAPER_TURN_TOLERANCE_RAD: f64 = 1.0e-10;

#[test]
fn gps_and_glonass_turns_follow_the_papers() {
    let all: Vec<Comparison> = compare_all()
        .into_iter()
        .filter(|c| c.paper.is_some())
        .collect();
    let turns = paper_turns_by_series(&all);
    // block -> (turns with samples, samples in turns, worst in-turn error)
    let mut counts: std::collections::BTreeMap<String, (usize, usize, f64)> = Default::default();
    for series in all.chunk_by(|a, b| a.case == b.case && a.block == b.block) {
        let first = &series[0];
        let closed = first.closed;
        let trajectory = move |t: f64| Some(closed.sample(t));
        let law = law_for_block_name(&first.block, first.paper.unwrap());
        let (_, series_turns) = &turns[&(first.case.clone(), first.block.clone())];
        let entry = counts.entry(first.block.clone()).or_default();

        // Turn starts and ends to a millisecond, evaluated off the fixture
        // grid, with the yaw just inside.
        for w in series_turns {
            if w.end() <= w.start {
                continue;
            }
            for (t, inside) in [
                (w.start - 1e-3, false),
                (w.start + 1e-3, true),
                (w.end() - 1e-3, true),
                (w.end() + 1e-3, false),
            ] {
                let rust = gnss_attitude(&law, &trajectory, t).unwrap();
                let where_ = format!("{} {} t={t} {w:?}", first.case, first.block);
                if inside {
                    assert_eq!(rust.mode, w.mode, "{where_}");
                    let error = wrap_pi(rust.yaw_rad - w.yaw(t)).abs();
                    entry.2 = entry.2.max(error);
                    assert!(error <= PAPER_TURN_TOLERANCE_RAD, "{where_}: {error:e}");
                } else {
                    assert_eq!(rust.mode, AttitudeMode::Nominal, "{where_}");
                }
            }
        }

        // Every fixture sample, and the runs of turn samples against the
        // turns.
        let mut runs = 0;
        let mut in_run = false;
        let mut turns_with_samples = std::collections::BTreeSet::new();
        for c in series {
            let where_ = format!("{} {} t={}", c.case, c.block, c.t);
            let sample = closed.sample(c.t);
            match series_turns.iter().position(|w| w.contains(c.t)) {
                Some(k) => {
                    let w = &series_turns[k];
                    turns_with_samples.insert(k);
                    entry.1 += 1;
                    assert_eq!(c.rust.mode, w.mode, "{where_}");
                    let error = wrap_pi(c.rust.yaw_rad - w.yaw(c.t)).abs();
                    entry.2 = entry.2.max(error);
                    assert!(
                        error <= PAPER_TURN_TOLERANCE_RAD,
                        "{where_}: paper-law yaw error {error:e}"
                    );
                    let (x, y) = axes_for_yaw(&sample, w.yaw(c.t));
                    let axes = axis_error(c.rust.body_x, x).max(axis_error(c.rust.body_y, y));
                    assert!(
                        axes <= PAPER_TURN_TOLERANCE_RAD + AXIS_ROUNDING_RAD,
                        "{where_}: paper-law axis error {axes:e}"
                    );
                }
                None => {
                    assert_eq!(c.rust.mode, AttitudeMode::Nominal, "{where_}");
                    let error = wrap_pi(c.rust.yaw_rad - geometry(&sample).nominal_yaw).abs();
                    assert!(
                        error <= INDEPENDENT_NOMINAL_TOLERANCE_RAD,
                        "{where_}: nominal yaw error {error:e}"
                    );
                }
            }
            let turning = c.rust.mode != AttitudeMode::Nominal;
            if turning && !in_run {
                runs += 1;
            }
            in_run = turning;
        }
        assert_eq!(
            runs,
            turns_with_samples.len(),
            "{} {}: {runs} turn runs, {} turns with samples",
            first.case,
            first.block,
            turns_with_samples.len()
        );
        entry.0 += turns_with_samples.len();
    }
    for (block, (windows, inside, worst)) in &counts {
        println!(
            "{block:16} {windows} turns, {inside} samples in turns, worst yaw against the papers {worst:.3e} rad"
        );
    }
    // The turns the evaluation here finds on these cases (each matched above
    // by one run of sidereon turn samples), and the samples inside them.
    for (block, expected_turns, expected_inside) in [
        ("BLOCK IIA", 33, 3528),
        ("BLOCK IIA PRN 11", 8, 1008),
        ("BLOCK IIA PRN 14", 9, 1396),
        ("BLOCK IIA PRN 23 BIAS -0.5 DEG", 9, 1377),
        ("BLOCK IIR-B", 22, 638),
        ("BLOCK IIF", 33, 2929),
        ("GLONASS-M", 21, 1600),
    ] {
        let (windows, inside, _) = counts[block];
        assert_eq!(windows, expected_turns, "{block}: turns");
        assert_eq!(inside, expected_inside, "{block}: samples in turns");
    }
}

/// GPS IIA at the fastest published IIA yaw rates (PRNs 2, 9, 11, 12, 13, 15,
/// 17 and 21, 0.1278 to 0.2 deg/s), at Sun elevations within 0.6 deg of the
/// orbit plane, where the shadow (about 3170 s at beta = 0) lasts longer than a
/// full yaw turn at the rate (1800 s at 0.2 deg/s): against the independent
/// evaluation of the papers' turns here, on circular GPS orbits whose plane
/// holds the Sun on the first day (inclination 55 deg, node on the Sun's
/// direction at the vernal equinox), so that beta drifts through zero. Every
/// 20 s for two days, and each turn's start and end to a millisecond.
#[test]
fn gps_iia_fast_rates_follow_the_papers_at_low_beta() {
    let fixture_model = model(&fixture());
    let a = 26_559_700.0;
    let inclination = 55.0_f64.to_radians();
    let duration = 2.0 * 86_400.0;
    let mut long_shadows = 0;
    let mut turns_checked = 0;
    for (lambda0, u0) in [(-0.02, 0.3), (-0.0005, 2.0), (0.004, -1.0)] {
        let closed = Closed {
            model: fixture_model,
            a,
            n: (fixture_model.gm / (a * a * a)).sqrt(),
            p: [1.0, 0.0, 0.0],
            q: [0.0, libm::cos(inclination), libm::sin(inclination)],
            u0,
            e: 0.0,
            argp: 0.0,
            lambda0,
        };
        let trajectory = move |t: f64| Some(closed.sample(t));
        for prn in [2_u8, 9, 11, 12, 13, 15, 17, 21] {
            let rate = gps_iia_default_yaw_rate_rad_s(prn).unwrap();
            let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(prn)).unwrap();
            let GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s,
                yaw_bias_rad,
                shadow_limit_rad,
            } = law
            else {
                unreachable!()
            };
            assert_eq!(yaw_rate_rad_s, rate);
            let paper = PaperLaw::GpsIia {
                rate,
                bias: yaw_bias_rad,
                limit: shadow_limit_rad,
            };
            let turns = paper_turns(&closed, paper, 0.0, duration);
            for w in &turns {
                let where_ = format!("lambda0 {lambda0} PRN {prn} {w:?}");
                if !w.noon && (w.zone_end - w.start) * rate > TAU {
                    long_shadows += 1;
                }
                if w.end() <= w.start || w.end() < 0.0 || w.start > duration {
                    continue;
                }
                turns_checked += 1;
                for (t, inside) in [
                    (w.start - 1e-3, false),
                    (w.start + 1e-3, true),
                    (w.end() - 1e-3, true),
                    (w.end() + 1e-3, false),
                ] {
                    let rust = gnss_attitude(&law, &trajectory, t).unwrap();
                    if inside {
                        assert_eq!(rust.mode, w.mode, "{where_} t={t}");
                        let error = wrap_pi(rust.yaw_rad - w.yaw(t)).abs();
                        assert!(
                            error <= PAPER_TURN_TOLERANCE_RAD,
                            "{where_} t={t}: {error:e}"
                        );
                    } else {
                        assert_eq!(rust.mode, AttitudeMode::Nominal, "{where_} t={t}");
                    }
                }
            }
            let mut t = 0.0;
            while t <= duration {
                let where_ = format!("lambda0 {lambda0} PRN {prn} t={t}");
                let rust =
                    gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
                match turns.iter().find(|w| w.contains(t)) {
                    Some(w) => {
                        assert_eq!(rust.mode, w.mode, "{where_} {w:?}");
                        let error = wrap_pi(rust.yaw_rad - w.yaw(t)).abs();
                        assert!(error <= PAPER_TURN_TOLERANCE_RAD, "{where_}: {error:e}");
                    }
                    None => {
                        assert_eq!(rust.mode, AttitudeMode::Nominal, "{where_}");
                        let nominal = geometry(&closed.sample(t)).nominal_yaw;
                        let error = wrap_pi(rust.yaw_rad - nominal).abs();
                        assert!(
                            error <= INDEPENDENT_NOMINAL_TOLERANCE_RAD,
                            "{where_}: nominal yaw error {error:e}"
                        );
                    }
                }
                t += 20.0;
            }
        }
    }
    println!("{turns_checked} turns, {long_shadows} shadows longer than a full yaw turn");
    assert!(
        long_shadows >= 20,
        "{long_shadows} shadows longer than a full turn"
    );
}

/// The Galileo IOV law's yaw (Galileo satellite metadata 3.1.1, with
/// `beta_x = 15 deg` and `beta_y = 2 deg`; Montenbruck et al. 2026, eqs. 11,
/// 13, 17 and 18, with `gamma_x = sin(15 deg)` and `gamma_y = sin(5.8 deg)` for
/// GPS III) written out from the closed form, `None` outside the auxiliary
/// region: `S_x = -sin(eta) cos(beta)`, `S_y = -sin(beta)`, the region
/// `|S_x| < sin(beta_x)`, `|S_y| < sin(beta_y)`, `S_Hy = (sin(beta_y) G + S_y)
/// / 2 + (sin(beta_y) G - S_y) / 2 cos(pi |S_x| / sin(beta_x))` with `G` the
/// sign of `S_y` where the region began (found here on a 10 s scan back and
/// bisected to 1e-9 s), and the yaw `atan2(S_Hy, S_x)` (IGS convention,
/// Section 3.2).
fn iov_yaw(closed: &Closed, t: f64, beta_x: f64, beta_y: f64) -> Option<f64> {
    let (sin_x, sin_y) = (libm::sin(beta_x), libm::sin(beta_y));
    let components = |t: f64| {
        let g = geometry(&closed.sample(t));
        (
            -libm::sin(g.from_noon) * libm::cos(g.beta),
            -libm::sin(g.beta),
        )
    };
    let in_region = |t: f64| {
        let (s_x, s_y) = components(t);
        s_x.abs() < sin_x && s_y.abs() < sin_y
    };
    if !in_region(t) {
        return None;
    }
    let mut back = t;
    while in_region(back) {
        back -= 10.0;
    }
    let entry = bisect_predicate(in_region, back, back + 10.0);
    let gamma = 1.0_f64.copysign(components(entry + 1e-9).1);
    let (s_x, s_y) = components(t);
    let s_hy = 0.5 * (sin_y * gamma + s_y)
        + 0.5 * (sin_y * gamma - s_y) * libm::cos(PI * s_x.abs() / sin_x);
    Some(libm::atan2(s_hy, s_x))
}

/// BeiDou-2 limits: yaw steering to orbit normal below 4.1 deg, back above
/// 3.9 deg.
const BDS2_YS_TO_ON_DEG: f64 = 4.1;
const BDS2_ON_TO_YS_DEG: f64 = 3.9;

/// Whether the BeiDou-2 IGSO or MEO law is in orbit-normal mode at `t`,
/// written out from the closed form: the mode the satellite switched to at the
/// last quadrature (orbit angle 90 deg from midnight, argument from noon
/// -90 deg; found here on a 60 s scan back and bisected to 1e-9 s), orbit
/// normal there if `|beta|` was at most 4.1 deg while decreasing or at most
/// 3.9 deg while increasing, whether it decreases taken from its change over
/// the last orbit (one period of the closed form).
fn beidou2_orbit_normal(closed: &Closed, t: f64) -> bool {
    let beta = geometry(&closed.sample(t)).beta;
    let before = geometry(&closed.sample(t - closed.period())).beta;
    let decreasing = beta * (beta - before) <= 0.0;
    let (on, ys) = (
        BDS2_YS_TO_ON_DEG.to_radians(),
        BDS2_ON_TO_YS_DEG.to_radians(),
    );
    let from_noon = |x: f64| geometry(&closed.sample(x)).from_noon;
    let mut later = t;
    let quadrature = loop {
        let earlier = later - 60.0;
        let (a, b) = (from_noon(earlier), from_noon(later));
        if a < -PI / 2.0 && b >= -PI / 2.0 && b - a < 1.0 {
            break bisect_predicate(|x| from_noon(x) >= -PI / 2.0, earlier, later);
        }
        later = earlier;
        assert!(
            t - later < 2.0 * closed.period(),
            "no quadrature within two orbits"
        );
    };
    let at_quadrature = geometry(&closed.sample(quadrature)).beta.abs();
    if decreasing {
        at_quadrature <= on
    } else {
        at_quadrature <= ys
    }
}

/// Whether Orekit's `BeidouMeo` takes its shortcut to orbit normal at `t`:
/// `|beta| <= 4.1 deg - |db|`, with `db` Orekit's change of beta over
/// `2 pi / mu_rate` at `t`, which with `|db| < 0.1 deg` covers `|beta|` up to
/// the band between 3.9 and 4.1 deg where the quadrature decides.
fn beidou2_in_orekit_band(closed: &Closed, t: f64) -> bool {
    let sample = closed.sample(t);
    let beta = geometry(&sample).beta;
    let before = geometry(&closed.sample(t - TAU / orbital_rate(&sample))).beta;
    beta.abs() <= BDS2_YS_TO_ON_DEG.to_radians() - (beta - before).abs()
}

fn law_for_block_name(block: &str, law: PaperLaw) -> GnssAttitudeLaw {
    match law {
        PaperLaw::GpsIia { rate, bias, limit } => GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s: rate,
            yaw_bias_rad: bias,
            shadow_limit_rad: limit,
        },
        PaperLaw::GpsIir { rate } => GnssAttitudeLaw::GpsIIR {
            yaw_rate_rad_s: rate,
        },
        PaperLaw::GpsIif { rate, bias, limit } => GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s: rate,
            yaw_bias_rad: bias,
            shadow_limit_rad: limit,
        },
        PaperLaw::Glonass { rate, limit } => {
            assert_eq!(block, "GLONASS-M");
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: rate,
                shadow_limit_rad: limit,
            }
        }
    }
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Cosine of the Sun-Earth-satellite angle, `r . s` with unit vectors.
fn cos_sun_satellite(sample: &AttitudeSample) -> f64 {
    dot(unit(sample.position_m), unit(sample.sun_position_m))
}

/// Geometry of one closed-form sample, formed here independently of the
/// library: Sun elevation, argument from noon, nominal yaw.
struct Geometry {
    beta: f64,
    from_noon: f64,
    nominal_yaw: f64,
}

fn geometry(sample: &AttitudeSample) -> Geometry {
    let r = unit(sample.position_m);
    let n = unit(cross(sample.position_m, sample.velocity_m_s));
    let s = unit(sample.sun_position_m);
    let beta = libm::asin(dot(s, n));
    let s_plane = [
        s[0] - dot(s, n) * n[0],
        s[1] - dot(s, n) * n[1],
        s[2] - dot(s, n) * n[2],
    ];
    let from_noon = libm::atan2(dot(cross(s_plane, r), n), dot(s_plane, r));
    // Kouba (2009) with the orbit angle from midnight, from_noon - pi.
    let nominal_yaw = libm::atan2(-libm::tan(beta), libm::sin(from_noon - PI));
    Geometry {
        beta,
        from_noon,
        nominal_yaw,
    }
}

/// A published law Orekit does not implement.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Published {
    /// Galileo satellite metadata 3.1.2 (colinearity start) or Wang et al.
    /// (2018) (orbit-angle start): `psi = 90 s + (psi_0 - 90 s) cos(2 pi t /
    /// T)` for half a period from the start.
    Cosine {
        colinearity: bool,
        start: f64,
        beta_limit: f64,
        inclusive: bool,
        period: f64,
    },
    /// CSNO (2019): nominal yaw steering with beta held at +-beta_0 while
    /// `|beta| < beta_0`.
    FixedBeta { beta0: f64 },
}

/// The published law of a block where Orekit applies its Galileo law, with
/// the published values written out here.
fn published_law(block: &str, law: &Value) -> Option<Published> {
    let cast = |period: f64| Published::Cosine {
        colinearity: false,
        start: 6.0_f64.to_radians(),
        beta_limit: 3.0_f64.to_radians(),
        inclusive: true,
        period,
    };
    match (law["law"].as_str().unwrap(), block) {
        ("GalileoFoc", _) => Some(Published::Cosine {
            colinearity: true,
            start: 10.0_f64.to_radians(),
            beta_limit: 4.1_f64.to_radians(),
            inclusive: false,
            period: 5656.0,
        }),
        // Wang et al. (2018): 3090 s for MEO, 5740 s for IGSO.
        ("Beidou3Cast", "BEIDOU-3M-CAST") => Some(cast(3090.0)),
        ("Beidou3Cast", "BEIDOU-3I") => Some(cast(5740.0)),
        ("Beidou3Cast", other) => panic!("no CAST period for {other}"),
        ("Beidou3Secm", _) => Some(Published::FixedBeta {
            beta0: 3.0_f64.to_radians(),
        }),
        _ => None,
    }
}

/// One published turn window: start and end instants, the centre, and the
/// nominal yaw at the start.
#[derive(Debug, Clone, Copy)]
struct Window {
    start: f64,
    end: f64,
    noon: bool,
    psi0: f64,
}

impl Window {
    fn yaw(&self, t: f64, period: f64) -> f64 {
        let quarter = if self.psi0 >= 0.0 {
            PI / 2.0
        } else {
            -PI / 2.0
        };
        quarter + (self.psi0 - quarter) * libm::cos(TAU * (t - self.start) / period)
    }
}

/// The published cosine-turn windows of a closed-form case that overlap
/// `[t_from, t_to]`: every instant the start angle falls below its value with
/// `|beta|` within the limit there. The start angle is the colinearity angle
/// `acos(|r . s|)` or the in-plane angle to the nearer of noon and midnight;
/// crossings are found on a 5 s scan (the angles move by under 1e-3 rad in
/// 5 s) and bisected to 1e-9 s.
fn cosine_windows(closed: &Closed, law: Published, t_from: f64, t_to: f64) -> Vec<Window> {
    let Published::Cosine {
        colinearity,
        start,
        beta_limit,
        inclusive,
        period,
    } = law
    else {
        unreachable!()
    };
    let angle = |t: f64| {
        let sample = closed.sample(t);
        if colinearity {
            libm::acos(cos_sun_satellite(&sample).abs().min(1.0))
        } else {
            let u = geometry(&sample).from_noon.abs();
            u.min(PI - u)
        }
    };
    let mut windows = Vec::new();
    let step = 5.0;
    let mut t = t_from - period;
    let mut previous = angle(t);
    while t < t_to {
        let next = t + step;
        let current = angle(next);
        if previous > start && current < start {
            let (mut lo, mut hi) = (t, next);
            while hi - lo > 1e-9 {
                let mid = 0.5 * (lo + hi);
                if angle(mid) > start {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t_start = 0.5 * (lo + hi);
            let g = geometry(&closed.sample(t_start));
            let abs_beta = g.beta.abs();
            let starts = if inclusive {
                abs_beta <= beta_limit
            } else {
                abs_beta < beta_limit
            };
            let t_end = t_start + 0.5 * period;
            if starts && t_end >= t_from && t_start <= t_to {
                windows.push(Window {
                    start: t_start,
                    end: t_end,
                    noon: g.from_noon.abs() < PI / 2.0,
                    psi0: g.nominal_yaw,
                });
            }
        }
        previous = current;
        t = next;
    }
    windows
}

/// Published-law yaw inside a window against the independent evaluation: the
/// start instants agree to the 1e-9 s of both solvers, the yaw rate is at most
/// `pi / 2 x 2 pi / period` (2e-3 rad/s at 5656 s), and the start yaw is formed
/// with different arithmetic (sidereon from the body axes, here from
/// `atan2(-tan(beta), sin(mu))`), a few units in the last place.
const PUBLISHED_WINDOW_TOLERANCE_RAD: f64 = 1.0e-11;

/// Margin of an axis check over its yaw tolerance: a yaw error `e` moves each
/// body axis by `2 sin(e / 2)`, at most `e`, and the axes formed here and in
/// sidereon use different arithmetic, a few units in the last place.
const AXIS_ROUNDING_RAD: f64 = 1.0e-13;

/// Nominal yaw formed here from `asin` and `atan2` against sidereon's RTKLIB
/// arithmetic: a few units in the last place of angles up to pi.
const INDEPENDENT_NOMINAL_TOLERANCE_RAD: f64 = 1.0e-13;

#[test]
fn published_laws_match_their_formulas_and_orekit_outside_their_windows() {
    let all: Vec<Comparison> = compare_all()
        .into_iter()
        .filter(|c| c.reference != Reference::Orekit)
        .collect();
    // (block) -> (windows with samples, samples in windows, samples outside
    // compared with Orekit, samples in Orekit's Galileo windows)
    let mut counts: std::collections::BTreeMap<String, (usize, usize, usize, usize)> =
        Default::default();
    let mut series = all.chunk_by(|a, b| a.case == b.case && a.block == b.block);
    for chunk in &mut series {
        let first = &chunk[0];
        let Reference::Published(law) = first.reference else {
            unreachable!()
        };
        let closed = first.closed;
        let (t_from, t_to) = (first.t, chunk.last().unwrap().t);
        let entry = counts.entry(first.block.clone()).or_default();
        let (period, windows) = match law {
            Published::Cosine { period, .. } => {
                (period, cosine_windows(&closed, law, t_from, t_to))
            }
            Published::FixedBeta { .. } => (f64::NAN, Vec::new()),
        };
        let trajectory = move |t: f64| Some(closed.sample(t));
        let law_rust = law_for_block(&first.block);

        // Window boundaries to a millisecond, evaluated off the fixture grid.
        for w in &windows {
            let turn_mode = if w.noon {
                AttitudeMode::NoonTurn
            } else {
                AttitudeMode::MidnightTurn
            };
            for (t, inside) in [
                (w.start - 1e-3, false),
                (w.start + 1e-3, true),
                (w.end - 1e-3, true),
                (w.end + 1e-3, false),
            ] {
                let rust = gnss_attitude(&law_rust, &trajectory, t).unwrap();
                if inside {
                    assert_eq!(
                        rust.mode, turn_mode,
                        "{} {} t={t} {w:?}",
                        first.case, first.block
                    );
                    let yaw = w.yaw(t, period);
                    let error = wrap_pi(rust.yaw_rad - yaw).abs();
                    assert!(
                        error <= PUBLISHED_WINDOW_TOLERANCE_RAD,
                        "{} {} t={t}: {error:e}",
                        first.case,
                        first.block
                    );
                    let (x, y) = axes_for_yaw(&closed.sample(t), yaw);
                    let axes = axis_error(rust.body_x, x).max(axis_error(rust.body_y, y));
                    assert!(
                        axes <= PUBLISHED_WINDOW_TOLERANCE_RAD + AXIS_ROUNDING_RAD,
                        "{} {} t={t}: axis error {axes:e}",
                        first.case,
                        first.block
                    );
                } else {
                    assert_eq!(
                        rust.mode,
                        AttitudeMode::Nominal,
                        "{} {} t={t} {w:?}",
                        first.case,
                        first.block
                    );
                }
            }
        }

        // Every fixture sample, and the runs of turn samples against the
        // windows.
        let mut runs = 0;
        let mut in_run = false;
        let mut windows_with_samples = std::collections::BTreeSet::new();
        for c in chunk {
            let sample = closed.sample(c.t);
            let g = geometry(&sample);
            let where_ = format!("{} {} t={}", c.case, c.block, c.t);
            let window = windows.iter().position(|w| w.start <= c.t && c.t <= w.end);
            let expected = match (law, window) {
                (Published::Cosine { .. }, Some(k)) => {
                    windows_with_samples.insert(k);
                    let w = windows[k];
                    Some((
                        if w.noon {
                            AttitudeMode::NoonTurn
                        } else {
                            AttitudeMode::MidnightTurn
                        },
                        w.yaw(c.t, period),
                    ))
                }
                (Published::FixedBeta { beta0 }, _) if g.beta.abs() < beta0 => {
                    let held = if g.beta >= 0.0 { beta0 } else { -beta0 };
                    Some((
                        AttitudeMode::ModifiedYawSteering,
                        libm::atan2(-libm::tan(held), libm::sin(g.from_noon - PI)),
                    ))
                }
                _ => None,
            };
            let turning = c.rust.mode != AttitudeMode::Nominal;
            if turning && !in_run {
                runs += 1;
            }
            in_run = turning;
            match expected {
                Some((mode, yaw)) => {
                    entry.1 += 1;
                    assert_eq!(c.rust.mode, mode, "{where_}");
                    let error = wrap_pi(c.rust.yaw_rad - yaw).abs();
                    assert!(
                        error <= PUBLISHED_WINDOW_TOLERANCE_RAD,
                        "{where_}: published-law yaw error {error:e}"
                    );
                    let (x, y) = axes_for_yaw(&sample, yaw);
                    let axes = axis_error(c.rust.body_x, x).max(axis_error(c.rust.body_y, y));
                    assert!(
                        axes <= PUBLISHED_WINDOW_TOLERANCE_RAD + AXIS_ROUNDING_RAD,
                        "{where_}: published-law axis error {axes:e}"
                    );
                }
                None => {
                    assert_eq!(c.rust.mode, AttitudeMode::Nominal, "{where_}");
                    let error = wrap_pi(c.rust.yaw_rad - g.nominal_yaw).abs();
                    assert!(
                        error <= INDEPENDENT_NOMINAL_TOLERANCE_RAD,
                        "{where_}: nominal yaw error {error:e}"
                    );
                    let orekit_nominal = wrap_pi(c.orekit_yaw - g.nominal_yaw).abs() < 1e-9;
                    if orekit_nominal {
                        entry.2 += 1;
                        assert!(
                            c.yaw_error <= NOMINAL_TOLERANCE_RAD
                                && c.axis_error <= NOMINAL_TOLERANCE_RAD,
                            "{where_}: against Orekit yaw {:e} axes {:e}",
                            c.yaw_error,
                            c.axis_error
                        );
                    } else {
                        // Orekit's Galileo law departs from nominal only inside
                        // its own window: |beta| below atan(mu_rate / its yaw
                        // rate) and the Sun-satellite angle within beta_x =
                        // 15 deg of noon or midnight.
                        entry.3 += 1;
                        let beta0 = libm::atan(orbital_rate(&sample) / c.orekit_yaw_rate.unwrap());
                        let sun_angle = libm::acos(cos_sun_satellite(&sample).abs().min(1.0));
                        assert!(
                            g.beta.abs() < beta0 + 1e-9 && sun_angle < 15.0_f64.to_radians() + 1e-9,
                            "{where_}: Orekit off nominal outside its window (beta {}, Sun angle {})",
                            g.beta,
                            sun_angle
                        );
                    }
                }
            }
        }
        if matches!(law, Published::Cosine { .. }) {
            assert_eq!(
                runs,
                windows_with_samples.len(),
                "{} {}: {runs} turn runs, {} windows with samples",
                first.case,
                first.block,
                windows_with_samples.len()
            );
            entry.0 += windows_with_samples.len();
        }
    }
    for (block, (windows, inside, orekit, orekit_window)) in &counts {
        println!(
            "{block:16} {windows} windows, {inside} samples in the published law, {orekit} nominal samples against Orekit, {orekit_window} in Orekit's Galileo window"
        );
    }
    // The windows the independent evaluation finds on these cases (each
    // matched above by one run of sidereon turn samples), and the samples
    // inside the published laws.
    for (block, expected_windows, expected_inside) in [
        ("GALILEO-2", 19, 1264),
        ("BEIDOU-3M-CAST", 45, 252),
        ("BEIDOU-3I", 29, 199),
        ("BEIDOU-3M-SECM", 0, 1406),
    ] {
        let (windows, inside, orekit, _) = counts[block];
        assert_eq!(windows, expected_windows, "{block}: windows");
        assert_eq!(
            inside, expected_inside,
            "{block}: samples in the published law"
        );
        assert!(
            orekit > 500,
            "{block}: {orekit} nominal samples against Orekit"
        );
    }
}

fn law_for_block(block: &str) -> GnssAttitudeLaw {
    GnssAttitudeLaw::for_block(block.parse().expect("IGS block name"), None).expect("law")
}

/// A GPS or GLONASS-M law as Kouba (2009), Kouba's `eclips.f` (December
/// 2017), Bar-Sever (1996), Dilssner (2010) and Dilssner et al. (2011) state
/// it, with the fixture's parameters.
#[derive(Clone, Copy, Debug)]
enum PaperLaw {
    /// GPS IIA: noon turns at `-sign(beta) R`, at `sign(bias) R` for beta of
    /// the bias's sign with `|beta| <= |bias|` and at beta = 0 (`eclips.f`);
    /// the shadow crossed at `SIGN(R, bias)` (Bar-Sever 1996, Kouba 2009,
    /// `eclips.f`); after the exit the yaw turns at `SIGN(R, D)`, `D` the
    /// nominal minus the actual yaw at the exit taken in [-180, 180] deg
    /// (Bar-Sever 1996), until it equals the nominal yaw.
    GpsIia { rate: f64, bias: f64, limit: f64 },
    /// GPS IIR: noon and midnight turns at the maximum rate.
    GpsIir { rate: f64 },
    /// GPS IIF: noon turns as GPS IIA's with its bias, and the shadow crossed
    /// at the constant rate that links the nominal yaws at shadow entry and
    /// exit (Dilssner 2010).
    GpsIif { rate: f64, bias: f64, limit: f64 },
    /// GLONASS-M: the noon turn at the maximum rate over the Dilssner span,
    /// and the shadow crossed at the maximum rate, held at the exit's nominal
    /// yaw once reached.
    Glonass { rate: f64, limit: f64 },
}

fn paper_law(law: &Value) -> Option<PaperLaw> {
    let rate = || f(&law["yaw_rate_rad_s"]);
    Some(match law["law"].as_str()? {
        "GpsIIA" => PaperLaw::GpsIia {
            rate: rate(),
            bias: f(&law["yaw_bias_rad"]),
            limit: f(&law["shadow_limit_rad"]),
        },
        "GpsIIR" => PaperLaw::GpsIir { rate: rate() },
        "GpsIIF" => PaperLaw::GpsIif {
            rate: rate(),
            bias: f(&law["yaw_bias_rad"]),
            limit: f(&law["shadow_limit_rad"]),
        },
        "GlonassM" => PaperLaw::Glonass {
            rate: rate(),
            limit: f(&law["shadow_limit_rad"]),
        },
        _ => return None,
    })
}

impl PaperLaw {
    fn rate(self) -> f64 {
        match self {
            PaperLaw::GpsIia { rate, .. }
            | PaperLaw::GpsIir { rate }
            | PaperLaw::GpsIif { rate, .. }
            | PaperLaw::Glonass { rate, .. } => rate,
        }
    }

    /// The shadow limit, for the laws with a shadow crossing.
    fn limit(self) -> Option<f64> {
        match self {
            PaperLaw::GpsIia { limit, .. }
            | PaperLaw::GpsIif { limit, .. }
            | PaperLaw::Glonass { limit, .. } => Some(limit),
            PaperLaw::GpsIir { .. } => None,
        }
    }
}

/// Orbital rate `|r x v| / |r|^2` of a closed-form sample.
fn orbital_rate(sample: &AttitudeSample) -> f64 {
    let h = cross(sample.position_m, sample.velocity_m_s);
    dot(h, h).sqrt() / dot(sample.position_m, sample.position_m)
}

/// Kouba (2009) nominal yaw rate `mu_rate tan(beta) cos(mu) / (sin^2(mu) +
/// tan^2(beta))`, `mu` the orbit angle from midnight.
fn nominal_yaw_rate(sample: &AttitudeSample) -> f64 {
    let g = geometry(sample);
    let mu = g.from_noon - PI;
    let tb = libm::tan(g.beta);
    orbital_rate(sample) * tb * libm::cos(mu) / (libm::sin(mu).powi(2) + tb * tb)
}

/// One turn of a paper law: the zone it starts and ends with, and the yaw.
#[derive(Clone, Copy, Debug)]
struct PaperTurn {
    noon: bool,
    mode: AttitudeMode,
    /// The first instant in the zone.
    start: f64,
    /// The last instant in the zone.
    zone_end: f64,
    /// GPS: where the linear yaw meets the nominal yaw (`None` for
    /// GLONASS-M, whose turns end with the zone).
    reach: Option<f64>,
    /// Nominal yaw at the start.
    psi0: f64,
    /// Yaw rate of the turn.
    rate: f64,
    /// GLONASS-M shadow: the yaw held once reached.
    hold: Option<f64>,
    /// Sun elevation at the start.
    beta: f64,
    /// GPS: the yaw at the zone end and the rate after it.
    exit: f64,
    after: f64,
    /// GPS IIA shadow: the first instant inside the shadow at which the yaw
    /// meets the nominal yaw again, catching it up after the nominal yaw ran
    /// ahead or having gained a full turn on it (the turn goes on through the
    /// shadow; Orekit's `GPSBlockIIA` ends there).
    met_in_shadow: Option<f64>,
}

/// GPS turns continue at most this long past their zone while the yaw has
/// not met the nominal yaw (Kouba's `eclips.f` and Orekit, 1800 s).
const PAPER_GPS_MARGIN_S: f64 = 1800.0;

impl PaperTurn {
    /// The last instant of the turn.
    fn end(&self) -> f64 {
        match self.reach {
            Some(reach) => reach.min(self.zone_end + PAPER_GPS_MARGIN_S),
            None => self.zone_end,
        }
    }

    fn contains(&self, t: f64) -> bool {
        match self.reach {
            Some(reach) => self.start <= t && t < reach && t <= self.zone_end + PAPER_GPS_MARGIN_S,
            None => self.start <= t && t <= self.zone_end,
        }
    }

    fn yaw(&self, t: f64) -> f64 {
        if self.reach.is_some() && t > self.zone_end {
            return self.exit + self.after * (t - self.zone_end);
        }
        let linear = self.psi0 + self.rate * (t - self.start);
        match self.hold {
            Some(hold) if self.rate > 0.0 => linear.min(hold),
            Some(hold) => linear.max(hold),
            None => linear,
        }
    }
}

/// Bisect a predicate that is `false` at `lo` and `true` at `hi` (or the
/// reverse) to 1e-9 s.
fn bisect_predicate(p: impl Fn(f64) -> bool, mut lo: f64, mut hi: f64) -> f64 {
    let at_lo = p(lo);
    assert_ne!(at_lo, p(hi), "no sign change in [{lo}, {hi}]");
    while hi - lo > 1e-9 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if p(mid) == at_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The turn of a paper law on the pass through the noon (`noon == true`) or
/// midnight at `centre`, if the pass has one.
///
/// Zones: Kouba (2009) noon and midnight turns where the nominal yaw rate
/// exceeds the maximum rate R in magnitude (they exist when
/// `|beta| < atan(mu_rate / R)`); the shadow where the Sun-satellite angle
/// exceeds the shadow limit; the GLONASS-M noon turn where the in-plane angle
/// to noon is within the Dilssner et al. (2011) half span (which exists where
/// `|beta| < atan(mu_rate / R)`). The zone is looked for over the whole half
/// orbit between the quadratures, each condition with the instant's own
/// orbital rate and beta, and each boundary is bisected to 1e-9 s.
fn paper_turn(closed: &Closed, law: PaperLaw, noon: bool, centre: f64) -> Option<PaperTurn> {
    let sample = |t: f64| closed.sample(t);
    let rate = law.rate();
    let shadow = if noon { None } else { law.limit() };
    let sun_angle = |t: f64| libm::acos(cos_sun_satellite(&sample(t)).clamp(-1.0, 1.0));
    let c = sample(centre);
    let gc = geometry(&c);
    // Positive exactly inside the zone.
    let zone_margin = |t: f64| -> f64 {
        if let Some(limit) = shadow {
            return sun_angle(t) - limit;
        }
        let s = sample(t);
        match law {
            PaperLaw::Glonass { .. } => {
                // The span exists where |beta| < atan(mu_rate / R).
                let g = geometry(&s);
                let mu_rate = orbital_rate(&s);
                if g.beta.abs() < libm::atan(mu_rate / rate) {
                    glonass_noon_half_span(g.beta, mu_rate, rate) - g.from_noon.abs()
                } else {
                    -1.0
                }
            }
            _ => nominal_yaw_rate(&s).abs() - rate,
        }
    };
    let in_zone = |t: f64| zone_margin(t) > 0.0;
    // The zone anywhere in the half orbit between the quadratures around the
    // centre (on an eccentric orbit it need not hold at the centre): the
    // quadratures where the in-plane angle from the centre reaches 90 deg,
    // stepped out to by 60 s and bisected, then a 20 s scan of the half orbit
    // with the centre, one run of instants inside, its edges bisected. Where
    // no scan point is inside, the scan point of largest margin is zoomed in
    // on (a scan of twice the step around it at a twentieth of it, eleven
    // times, down to 4e-13 s): a zone narrower than the step is found there,
    // or there is none.
    let centre_u = if noon { 0.0 } else { PI };
    let from_centre = |t: f64| wrap_pi(geometry(&sample(t)).from_noon - centre_u);
    let quadrature = |direction: f64| {
        let mut t = centre;
        while (from_centre(t + direction * 60.0) * direction) < PI / 2.0 {
            t += direction * 60.0;
        }
        bisect_predicate(
            |x| from_centre(x) * direction >= PI / 2.0,
            t,
            t + direction * 60.0,
        )
    };
    let (t_lo, t_hi) = (quadrature(-1.0), quadrature(1.0));
    let step = 20.0;
    let mut scan: Vec<f64> = Vec::new();
    let mut t = t_lo;
    while t < t_hi {
        scan.push(t);
        t += step;
    }
    scan.push(centre);
    scan.sort_by(f64::total_cmp);
    let inside: Vec<usize> = (0..scan.len()).filter(|&i| in_zone(scan[i])).collect();
    let (start, zone_end) = if let (Some(&first), Some(&last)) = (inside.first(), inside.last()) {
        assert_eq!(last - first + 1, inside.len(), "one run of the zone");
        (
            bisect_predicate(in_zone, scan[first] - step, scan[first]),
            bisect_predicate(in_zone, scan[last], scan[last] + step),
        )
    } else {
        let mut best = scan
            .iter()
            .copied()
            .max_by(|a, b| zone_margin(*a).total_cmp(&zone_margin(*b)))
            .unwrap();
        let mut width = step;
        for _ in 0..11 {
            let (lo, fine) = (best - width, width / 20.0);
            for k in 0..=40 {
                let t = lo + fine * f64::from(k);
                if zone_margin(t) > zone_margin(best) {
                    best = t;
                }
            }
            width = fine;
        }
        if !in_zone(best) {
            let exists_at_centre = match shadow {
                Some(limit) => sun_angle(centre) > limit,
                None => gc.beta.abs() < libm::atan(orbital_rate(&c) / rate),
            };
            assert!(
                !exists_at_centre,
                "zone narrower than the centre instant at {centre}"
            );
            return None;
        }
        (
            bisect_predicate(in_zone, best - step, best),
            bisect_predicate(in_zone, best, best + step),
        )
    };
    let gs = geometry(&sample(start));
    let ge = geometry(&sample(zone_end));
    let (psi0, beta) = (gs.nominal_yaw, gs.beta);
    let (mode, turn_rate, hold) = match (law, noon) {
        (PaperLaw::Glonass { .. }, true) => (AttitudeMode::NoonTurn, -rate.copysign(beta), None),
        (PaperLaw::Glonass { .. }, false) => (
            AttitudeMode::Shadow,
            rate.copysign(beta),
            Some(ge.nominal_yaw),
        ),
        (PaperLaw::GpsIia { bias, .. } | PaperLaw::GpsIif { bias, .. }, true) => {
            let in_bias = beta == 0.0 || (beta * bias > 0.0 && beta.abs() <= bias.abs());
            let r = if bias != 0.0 && in_bias {
                rate.copysign(bias)
            } else {
                -rate.copysign(beta)
            };
            (AttitudeMode::NoonTurn, r, None)
        }
        (PaperLaw::GpsIir { .. }, true) => (AttitudeMode::NoonTurn, -rate.copysign(beta), None),
        (PaperLaw::GpsIia { bias, .. }, false) => (
            AttitudeMode::Shadow,
            if bias < 0.0 { -rate } else { rate },
            None,
        ),
        (PaperLaw::GpsIir { .. }, false) => (AttitudeMode::MidnightTurn, rate.copysign(beta), None),
        (PaperLaw::GpsIif { .. }, false) => (
            AttitudeMode::Shadow,
            (ge.nominal_yaw - psi0) / (zone_end - start),
            None,
        ),
    };
    let mut turn = PaperTurn {
        noon,
        mode,
        start,
        zone_end,
        reach: None,
        psi0,
        rate: turn_rate,
        hold,
        beta,
        exit: psi0 + turn_rate * (zone_end - start),
        after: turn_rate,
        met_in_shadow: None,
    };
    if let (PaperLaw::GpsIia { .. }, false) = (law, noon) {
        // How far the nominal yaw is ahead of the yaw, the way it turns,
        // followed on a 0.5 s step (the nominal yaw stays within half a turn
        // for beta of one sign): it meets the yaw where this returns to 0
        // after being positive, or reaches -2 pi.
        let nominal = |t: f64| geometry(&closed.sample(t)).nominal_yaw;
        let ahead =
            |t: f64| turn_rate.signum() * (nominal(t) - psi0) - turn_rate.abs() * (t - start);
        let mut was_ahead = false;
        let mut t = start;
        while t < zone_end {
            let next = (t + 0.5).min(zone_end);
            let d = ahead(next);
            if (was_ahead && d <= 0.0) || d <= -TAU {
                let target = if d <= -TAU { -TAU } else { 0.0 };
                turn.met_in_shadow = Some(bisect_predicate(|x| ahead(x) <= target, t, next));
                break;
            }
            was_ahead |= d > 1e-9;
            t = next;
        }
    }
    if !matches!(law, PaperLaw::Glonass { .. }) {
        let (reach, after) = paper_reach(closed, law, &turn);
        turn.reach = Some(reach);
        turn.after = after;
    }
    Some(turn)
}

/// Where a GPS linear turn ends, and its rate after the zone: the first
/// instant at or after the zone end at which the yaw, carried on from the
/// zone end, meets the nominal yaw; `f64::INFINITY` when that does not happen
/// within 1800 s (the turn then runs the full margin). Kouba (2009): a turn
/// the way the nominal yaw moves lags it at the zone end by less than pi and
/// continues at its rate; a noon turn reversed by the bias continues the long
/// way. Bar-Sever (1996): after the GPS IIA shadow the yaw turns at
/// `SIGN(R, D)`, `D = psi_n(t_e) - psi(t_e) - 360 NINT((psi_n(t_e) -
/// psi(t_e)) / 360)` deg. The GPS IIF shadow crossing ends at the zone end,
/// where its rate brings it to the exit's nominal yaw (Dilssner 2010).
///
/// Integrated forward in time from the zone end: the remaining distance is
/// followed continuously on a 0.5 s step, the nominal yaw unwrapped from step
/// to step (it moves by far less than pi in 0.5 s past a zone), and its first
/// zero is bisected to 1e-9 s within the step, the nominal yaw unwrapped
/// against the step's start.
fn paper_reach(closed: &Closed, law: PaperLaw, turn: &PaperTurn) -> (f64, f64) {
    if matches!(law, PaperLaw::GpsIif { .. }) && !turn.noon {
        return (turn.zone_end, turn.rate);
    }
    let nominal = |t: f64| geometry(&closed.sample(t)).nominal_yaw;
    let end = turn.zone_end;
    let to_nominal = nominal(end) - turn.exit;
    let (after, mut distance) = match (law, turn.noon) {
        (PaperLaw::GpsIia { rate, .. }, false) => {
            let d = to_nominal - TAU * libm::round(to_nominal / TAU);
            (rate.copysign(d), d.abs())
        }
        _ => {
            let forward = turn.rate.signum() * to_nominal;
            let reversed =
                turn.noon && turn.beta != 0.0 && turn.rate.signum() == turn.beta.signum();
            let wrapped = forward.rem_euclid(TAU);
            // Within a zone the nominal yaw runs ahead of the turn by less than
            // pi; a nominal yaw a rounding behind it (near 2 pi ahead) has met
            // it.
            let distance = if reversed || wrapped < 1.5 * PI {
                wrapped
            } else {
                0.0
            };
            (turn.rate, distance)
        }
    };
    if distance == 0.0 {
        return (end, after);
    }
    let sign = after.signum();
    let speed = after.abs();
    let limit = end + PAPER_GPS_MARGIN_S;
    let (mut t, mut psi) = (end, nominal(end));
    let step = 0.5;
    while t < limit {
        let next = (t + step).min(limit);
        // The distance at x in [t, next], from the step's start.
        let at = |x: f64| distance + sign * wrap_pi(nominal(x) - psi) - speed * (x - t);
        if at(next) <= 0.0 {
            return (bisect_predicate(|x| at(x) <= 0.0, t, next), after);
        }
        distance = at(next);
        psi = nominal(next);
        t = next;
    }
    (f64::INFINITY, after)
}

/// Every paper-law turn whose pass centre lies within half an orbit of
/// `[t_from, t_to]`, centres found on a 60 s scan of the argument from noon and
/// bisected to 1e-9 s.
fn paper_turns(closed: &Closed, law: PaperLaw, t_from: f64, t_to: f64) -> Vec<PaperTurn> {
    let period = TAU / closed.n;
    let from_noon = |t: f64| geometry(&closed.sample(t)).from_noon;
    let mut turns = Vec::new();
    let step = 60.0;
    let mut t = t_from - 0.5 * period;
    while t < t_to + 0.5 * period {
        let (a, b) = (from_noon(t), from_noon(t + step));
        // Noon: the argument from noon rises through 0; midnight: it wraps
        // from +pi to -pi.
        let noon = a < 0.0 && b >= 0.0 && b - a < 1.0;
        let midnight = a > 0.0 && b < 0.0 && a - b > 1.0;
        if noon || midnight {
            let centre = bisect_predicate(|x| from_noon(x) < 0.0, t, t + step);
            if let Some(turn) = paper_turn(closed, law, noon, centre) {
                turns.push(turn);
            }
        }
        t += step;
    }
    turns
}

/// Orekit's turn as it estimates it from the geometry at `t` alone
/// (`GNSSAttitudeContext.setHalfSpan` and the block providers): the start
/// `t + (delta - half_span) / mu_rate`, with `delta` the in-plane angle to the
/// centre (positive approaching it) and the half span at `t`'s beta (Kouba's
/// `|beta| sqrt(atan(mu_rate / R) / |beta| - 1)` at a GPS noon or IIR
/// midnight, `acos(cos(limit - pi) / cos(beta))` in the shadow, the Dilssner
/// half span at a GLONASS-M noon); the start yaw `atan2(-tan(beta), sin(+-half
/// span))`; beta from `t`, or from the estimated start below 0.07 deg
/// (`getSecuredBeta`).
#[derive(Clone, Copy, Debug)]
struct OrekitEstimate {
    start: f64,
    psi0: f64,
    rate: f64,
    hold: Option<f64>,
}

fn orekit_estimate(closed: &Closed, law: PaperLaw, noon: bool, t: f64) -> OrekitEstimate {
    let s = closed.sample(t);
    let g = geometry(&s);
    let mu_rate = orbital_rate(&s);
    let delta = if noon {
        -g.from_noon
    } else {
        wrap_pi(PI - g.from_noon)
    };
    let rate = law.rate();
    let shadow = if noon { None } else { law.limit() };
    let half_span = match (shadow, law) {
        (Some(limit), _) => libm::acos(libm::cos(limit - PI) / libm::cos(g.beta)),
        (None, PaperLaw::Glonass { .. }) => glonass_noon_half_span(g.beta, mu_rate, rate),
        (None, _) => {
            let b = g.beta.abs();
            b * (libm::atan(mu_rate / rate) / b - 1.0).sqrt()
        }
    };
    let start = t + (delta - half_span) / mu_rate;
    let beta = if g.beta.abs() < 0.07_f64.to_radians() {
        geometry(&closed.sample(start)).beta
    } else {
        g.beta
    };
    let side = if noon { 1.0 } else { -1.0 };
    let phi = |x: f64| libm::atan2(-libm::tan(beta), libm::sin(x));
    let psi0 = phi(side * half_span);
    let (turn_rate, hold) = match (law, noon) {
        (PaperLaw::GpsIia { bias, .. }, true) if beta > 0.0 && beta < bias => {
            (rate.copysign(beta), None)
        }
        (PaperLaw::GpsIif { bias, .. }, true) if beta > bias && beta < 0.0 => {
            (rate.copysign(beta), None)
        }
        (_, true) => (-rate.copysign(beta), None),
        (PaperLaw::GpsIia { .. }, false) => (rate, None),
        (PaperLaw::GpsIir { .. }, false) => (rate.copysign(beta), None),
        (PaperLaw::GpsIif { .. }, false) => {
            ((phi(half_span) - psi0) / (2.0 * half_span / mu_rate), None)
        }
        (PaperLaw::Glonass { .. }, false) => (rate.copysign(beta), Some(phi(half_span))),
    };
    OrekitEstimate {
        start,
        psi0,
        rate: turn_rate,
        hold,
    }
}
