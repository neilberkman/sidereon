//! GLONASS broadcast ephemeris: PZ-90.11 state-vector propagation (Phase 5b).
//!
//! GLONASS does not broadcast Keplerian elements. Each record carries an ECEF
//! state at a reference epoch - position, velocity, and the lunisolar
//! acceleration - plus the clock terms (−TauN, +GammaN). The position at another
//! epoch comes from numerically integrating the equation of motion (central
//! gravity + the J2 oblateness term + the Earth-rotation / Coriolis terms + the
//! broadcast lunisolar acceleration held constant) with a fixed-step fourth-order
//! Runge-Kutta integrator - the canonical GLONASS-ICD / RTKLIB `glorbit`/`deq`
//! algorithm.
//!
//! This follows the reference recipe in `fixtures-generators/glonass_golden.py`
//! statement-for-statement: plain `f64` arithmetic with no fused multiply-add and
//! integer powers written as explicit multiplies, so it is a bit-exact (0-ULP)
//! target against the committed `glonass_golden.json`.
//!
//! The step is pinned at 60 s with a final partial step to land exactly on the
//! requested epoch, and the integration direction follows the sign of the time
//! difference. A different step policy is a different (still valid) answer, so it
//! is part of the pinned contract.

pub use crate::astro::constants::models::pz90::{
    A_M as R_E, GM_M3_S2 as MU, J2, OMEGA_E_RAD_S as OMEGA_E,
};

use crate::tolerances::GLONASS_TIME_EPS_S;

/// Pinned RK4 fixed step (seconds).
pub const TSTEP_S: f64 = 60.0;
/// Most RK4 steps one propagation takes: the 30 full steps of the 1800 s broadcast age
/// limit (RTKLIB `MAXDTOE_GLO`), plus one for the remainder when the 1 ms step of a
/// record's velocity (RTKLIB `ephpos`) is added at that limit.
const MAX_PROPAGATION_STEPS: usize = 31;
const _: () = assert!(
    MAX_PROPAGATION_STEPS as f64 * TSTEP_S
        >= crate::rinex_nav::GLONASS_MAX_AGE_S + crate::rinex_nav::EPHPOS_STEP_S
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlonassError {
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

impl core::fmt::Display for GlonassError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid GLONASS input {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for GlonassError {}

/// State derivative for the GLONASS equation of motion.
///
/// `s` is `[x, y, z, vx, vy, vz]` in metres and metres/second (PZ-90 ECEF); `acc`
/// is the broadcast lunisolar acceleration `[ax, ay, az]` in m/s^2, held constant
/// over the step. Returns `[vx, vy, vz, ax, ay, az]`.
pub(crate) fn deq(s: &[f64; 6], acc: &[f64; 3]) -> [f64; 6] {
    let (x, y, z, vx, vy, vz) = (s[0], s[1], s[2], s[3], s[4], s[5]);
    let r2 = x * x + y * y + z * z;
    // RTKLIB `deq` returns a zero derivative for a state at the origin rather than
    // dividing by zero.
    if r2 <= 0.0 {
        return [0.0; 6];
    }
    let r = r2.sqrt();
    let r3 = r2 * r;
    // a = 3/2 * J2 * mu * Re^2 / r^5, formed as RTKLIB `deq` writes it:
    // a=1.5*J2_GLO*MU_GLO*SQR(RE_GLO)/r2/r3;
    let a = 1.5 * J2 * MU * (R_E * R_E) / r2 / r3;
    let b = 5.0 * z * z / r2;
    let c = -MU / r3 - a * (1.0 - b);
    let omg2 = OMEGA_E * OMEGA_E;
    let ax = (c + omg2) * x + 2.0 * OMEGA_E * vy + acc[0];
    let ay = (c + omg2) * y - 2.0 * OMEGA_E * vx + acc[1];
    let az = (c - 2.0 * a) * z + acc[2];
    [vx, vy, vz, ax, ay, az]
}

/// One classical RK4 step of length `step` seconds.
// explicit indices keep the RK4 stage updates legible and pinned
pub(crate) fn glorbit(step: f64, s: &[f64; 6], acc: &[f64; 3]) -> [f64; 6] {
    let k1 = deq(s, acc);
    let mut w = [0.0_f64; 6];
    for i in 0..6 {
        w[i] = s[i] + k1[i] * step / 2.0;
    }
    let k2 = deq(&w, acc);
    for i in 0..6 {
        w[i] = s[i] + k2[i] * step / 2.0;
    }
    let k3 = deq(&w, acc);
    for i in 0..6 {
        w[i] = s[i] + k3[i] * step;
    }
    let k4 = deq(&w, acc);
    let mut out = [0.0_f64; 6];
    for i in 0..6 {
        out[i] = s[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * step / 6.0;
    }
    out
}

/// Integrate `state0` by `tk` seconds with the pinned step policy: fixed 60 s
/// steps in the direction of `tk`, then a final partial step of the remainder.
pub fn propagate(state0: [f64; 6], acc: [f64; 3], tk: f64) -> Result<[f64; 6], GlonassError> {
    if !tk.is_finite() {
        return Err(invalid_input("tk", "not finite"));
    }

    let mut state = state0;
    let mut tt = tk;
    let mut steps = 0usize;
    while tt.abs() > GLONASS_TIME_EPS_S {
        if steps >= MAX_PROPAGATION_STEPS {
            return Err(invalid_input("tk", "out of range"));
        }
        let step = if tt.abs() < TSTEP_S {
            tt
        } else if tt > 0.0 {
            TSTEP_S
        } else {
            -TSTEP_S
        };
        state = glorbit(step, &state, &acc);
        tt -= step;
        steps += 1;
    }
    Ok(state)
}

fn invalid_input(field: &'static str, reason: &'static str) -> GlonassError {
    GlonassError::InvalidInput { field, reason }
}

/// GLONASS clock offset (seconds) at satellite clock time `tk` = t − toe: RTKLIB
/// `geph2clk`.
///
/// `clk_bias` is the broadcast line-0 field (which is −TauN) and `gamma_n` is
/// +GammaN. With `ts = tk`, the time argument is refined twice as
/// `t = ts - (clk_bias + gamma_n·t)`, which removes the satellite clock from a
/// time read on that clock, and the offset is `clk_bias + gamma_n·t`. There is no
/// relativistic eccentricity term and no group delay.
pub fn clock_offset_s(clk_bias: f64, gamma_n: f64, tk: f64) -> f64 {
    // t=ts=timediff(time,geph->toe);
    // for (i=0;i<2;i++) t=ts-(-geph->taun+geph->gamn*t);
    // return -geph->taun+geph->gamn*t;
    let ts = tk;
    let mut t = ts;
    for _ in 0..2 {
        t = ts - (clk_bias + gamma_n * t);
    }
    clk_bias + gamma_n * t
}

/// GLONASS clock offset (seconds) at system time `tk` = t − toe, evaluated with
/// the orbit as RTKLIB `geph2pos` does: `clk_bias + gamma_n·tk`, without
/// iteration.
pub fn position_clock_offset_s(clk_bias: f64, gamma_n: f64, tk: f64) -> f64 {
    // *dts=-geph->taun+geph->gamn*t;
    clk_bias + gamma_n * tk
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn state0() -> [f64; 6] {
        [
            10_908_942.0,
            -2_885_726.0,
            22_883_539.0,
            1407.8,
            2795.9,
            -317.0,
        ]
    }

    #[test]
    fn propagate_rejects_nonfinite_time() {
        for tk in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                propagate(state0(), [0.0, 0.0, 0.0], tk),
                Err(GlonassError::InvalidInput {
                    field: "tk",
                    reason: "not finite"
                })
            );
        }
    }

    /// A record's velocity at the 1800 s age limit propagates 1 ms past it, which takes a
    /// thirty-first step.
    #[test]
    fn propagate_reaches_the_velocity_step_past_the_age_limit() {
        let tk = crate::rinex_nav::GLONASS_MAX_AGE_S + crate::rinex_nav::EPHPOS_STEP_S;
        assert!(propagate(state0(), [0.0, 0.0, 0.0], tk).is_ok());
    }

    #[test]
    fn propagation_step_limit_covers_the_age_limit_and_velocity_step() {
        assert_eq!(MAX_PROPAGATION_STEPS, 31);
        assert_eq!(crate::rinex_nav::GLONASS_MAX_AGE_S, 1800.0);
        assert!(propagate(state0(), [0.0, 0.0, 0.0], -1800.001).is_ok());
    }

    /// RTKLIB `geph2clk` subtracts from the first time difference `ts` on each
    /// refinement; the second refinement is `ts - (clk_bias + gamma_n·t1)`.
    #[test]
    fn clock_offset_is_rtklib_geph2clk() {
        for clk_bias in [-1.0e-3, 5.8e-5, 0.0] {
            for gamma_n in [0.0, 9.1e-13, -2.7e-12] {
                for tk in [-1800.0, -0.5, 0.0, 900.0] {
                    let taun = -clk_bias;
                    let ts = tk;
                    let mut t = ts;
                    for _ in 0..2 {
                        t = ts - (-taun + gamma_n * t);
                    }
                    let want = -taun + gamma_n * t;
                    let got = clock_offset_s(clk_bias, gamma_n, tk);
                    assert_eq!(got.to_bits(), want.to_bits(), "{clk_bias} {gamma_n} {tk}");
                    let unrefined = -taun + gamma_n * tk;
                    assert_eq!(
                        position_clock_offset_s(clk_bias, gamma_n, tk).to_bits(),
                        unrefined.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn deq_at_the_origin_is_zero_as_rtklib() {
        assert_eq!(deq(&[0.0; 6], &[1.0, 2.0, 3.0]), [0.0; 6]);
    }

    #[test]
    fn propagate_rejects_too_many_steps() {
        let tk = TSTEP_S * (MAX_PROPAGATION_STEPS as f64 + 1.0);
        assert_eq!(
            propagate(state0(), [0.0, 0.0, 0.0], tk),
            Err(GlonassError::InvalidInput {
                field: "tk",
                reason: "out of range"
            })
        );
    }

    #[test]
    fn propagate_valid_time_matches_single_rk4_step() {
        let state0 = state0();
        let acc = [1.0e-9, -2.0e-9, 3.0e-9];
        let got = propagate(state0, acc, TSTEP_S).expect("valid GLONASS propagation");
        let want = glorbit(TSTEP_S, &state0, &acc);
        for (got, want) in got.iter().zip(want) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
