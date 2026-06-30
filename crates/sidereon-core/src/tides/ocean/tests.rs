//! Validation of [`ocean_tide_loading`](super::ocean_tide_loading) against the
//! IERS `ARG2` / HARDISP-BLQ convention.
//!
//! The end-to-end RTKLIB `tidedisp` oracle agreement lives in
//! `tests/ocean_loading_oracle.rs`; these are the in-crate unit checks of the
//! argument computation, the displacement magnitude, and input validation.

use super::{arg2_angles, day_of_year, ocean_tide_loading, OceanLoadingBlq};
use crate::tides::{TideError, TideInputErrorKind};

/// ZIM2 (Zimmerwald) ITRF2020 ECEF position, from `tests/ppp_decimeter_arc.rs`.
const ZIM2_ECEF_M: [f64; 3] = [
    4_331_299.584_071_246,
    567_537.707_032_023_1,
    4_633_133.964_520_6,
];

// ZIM2 ocean-loading BLQ coefficients (ocean tide model GOT4.7, long-period
// tides from FES99), computed by OLFG/OLMPP of H.-G. Scherneck, Onsala Space
// Observatory (holt.oso.chalmers.se ocean tide loading provider), 2020-Jun-25;
// obtained from the published BLQ block for ZIM2 (lon/lat 7.4650 46.8771,
// 956.425 m). BLQ column order M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa; row order
// amplitude radial/EW/NS (m) then phase radial/EW/NS (deg). Real provider
// values, not fabricated.
const ZIM2_BLQ: OceanLoadingBlq = OceanLoadingBlq {
    amplitude_m: [
        [
            0.00693, 0.00228, 0.00148, 0.00061, 0.00220, 0.00094, 0.00070, 0.00001, 0.00047,
            0.00025, 0.00019,
        ],
        [
            0.00272, 0.00076, 0.00061, 0.00020, 0.00036, 0.00025, 0.00011, 0.00005, 0.00004,
            0.00001, 0.00002,
        ],
        [
            0.00061, 0.00026, 0.00010, 0.00009, 0.00025, 0.00002, 0.00008, 0.00003, 0.00002,
            0.00000, 0.00001,
        ],
    ],
    phase_deg: [
        [
            -72.3, -44.2, -90.8, -44.1, -62.9, -94.5, -64.3, 171.0, 3.4, 3.6, 1.1,
        ],
        [
            84.3, 115.4, 63.3, 113.7, 98.6, 20.7, 94.2, -44.5, -170.0, -162.7, -177.8,
        ],
        [
            -29.3, 1.7, -44.0, -4.2, 44.2, -39.1, 43.7, 170.1, -93.3, -118.3, -176.4,
        ],
    ],
};

#[test]
fn day_of_year_matches_calendar() {
    assert_eq!(day_of_year(2026, 1, 1), 1);
    assert_eq!(day_of_year(2026, 5, 13), 133); // DOY 133 (the ZIM2 arc day).
    assert_eq!(day_of_year(2026, 12, 31), 365);
    // 2024 is a leap year, so DOY of Mar 1 is 61 (not 60).
    assert_eq!(day_of_year(2024, 3, 1), 61);
    assert_eq!(day_of_year(2024, 12, 31), 366);
}

#[test]
fn arg2_s2_advances_with_solar_time() {
    // S2 has zero ANGFAC, so its argument is purely SPEED_S2 * FDAY and must
    // advance by exactly the solar-semidiurnal rate (30 deg/hr = pi/6 rad/hr).
    let a0 = arg2_angles(2026, 5, 13, 0.0)[1];
    let a6 = arg2_angles(2026, 5, 13, 6.0)[1];
    let advance = (a6 - a0).rem_euclid(2.0 * std::f64::consts::PI);
    // 6 hours of S2 -> pi (180 deg). The residual ~2e-6 rad reflects ARG2's
    // truncated SPEED constant (1.45444e-4 rad/s), not a code error.
    assert!(
        (advance - std::f64::consts::PI).abs() < 1.0e-5,
        "S2 6 h advance {advance} rad"
    );
}

#[test]
fn arg2_angles_are_normalized() {
    let angles = arg2_angles(2026, 5, 13, 12.5);
    for a in angles {
        assert!(
            (0.0..2.0 * std::f64::consts::PI).contains(&a),
            "argument {a} outside [0, 2pi)"
        );
    }
}

#[test]
fn ocean_loading_zim2_real_blq_has_few_mm_magnitude() {
    let d = ocean_tide_loading(&ZIM2_ECEF_M, 2026, 5, 13, 12.5, &ZIM2_BLQ)
        .expect("valid ZIM2 ocean loading input");
    let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    // ZIM2 is deep inland (Switzerland); the OTL signal is at the few-mm level
    // and never exceeds ~1 cm even at the tidal extreme.
    assert!(
        mag < 1.0e-2,
        "ZIM2 ocean loading magnitude {mag:.4e} m exceeds the expected few-mm band"
    );
    assert!(d.iter().all(|c| c.is_finite()));
}

#[test]
fn ocean_loading_varies_over_the_day() {
    // The displacement is dominated by the semidiurnal M2/S2 band, so it must
    // change materially between epochs hours apart.
    let d0 = ocean_tide_loading(&ZIM2_ECEF_M, 2026, 5, 13, 0.0, &ZIM2_BLQ).expect("valid");
    let d6 = ocean_tide_loading(&ZIM2_ECEF_M, 2026, 5, 13, 6.0, &ZIM2_BLQ).expect("valid");
    let delta =
        ((d0[0] - d6[0]).powi(2) + (d0[1] - d6[1]).powi(2) + (d0[2] - d6[2]).powi(2)).sqrt();
    assert!(
        delta > 1.0e-3,
        "6 h displacement change {delta:.4e} m too small"
    );
}

#[test]
fn ocean_loading_rejects_degenerate_geometry() {
    assert_invalid(
        ocean_tide_loading(&[0.0, 0.0, 0.0], 2026, 5, 13, 12.0, &ZIM2_BLQ),
        "station radius",
        TideInputErrorKind::NotPositive,
    );
}

#[test]
fn ocean_loading_rejects_invalid_date_hour_and_nonfinite_blq() {
    assert_invalid(
        ocean_tide_loading(&ZIM2_ECEF_M, 2026, 13, 13, 12.0, &ZIM2_BLQ),
        "civil datetime",
        TideInputErrorKind::InvalidCivilDate,
    );
    assert_invalid(
        ocean_tide_loading(&ZIM2_ECEF_M, 2026, 5, 13, 24.0, &ZIM2_BLQ),
        "fractional hour",
        TideInputErrorKind::OutOfRange,
    );

    let mut bad_amp = ZIM2_BLQ;
    bad_amp.amplitude_m[0][0] = f64::NAN;
    assert_invalid(
        ocean_tide_loading(&ZIM2_ECEF_M, 2026, 5, 13, 12.0, &bad_amp),
        "ocean loading amplitude",
        TideInputErrorKind::NonFinite,
    );

    let mut bad_phase = ZIM2_BLQ;
    bad_phase.phase_deg[2][5] = f64::INFINITY;
    assert_invalid(
        ocean_tide_loading(&ZIM2_ECEF_M, 2026, 5, 13, 12.0, &bad_phase),
        "ocean loading phase",
        TideInputErrorKind::NonFinite,
    );
}

fn assert_invalid(got: Result<[f64; 3], TideError>, field: &'static str, kind: TideInputErrorKind) {
    assert_eq!(
        got.expect_err("invalid ocean loading input must error"),
        TideError::InvalidInput { field, kind }
    );
}
