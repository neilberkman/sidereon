//! Validation of [`solid_earth_pole_tide`](super::solid_earth_pole_tide)
//! against the IERS Conventions (2010) §7.1.4 reference.
//!
//! The displacement kernel (Eq. 7.24) is cross-checked against an independent
//! latitude-form transcription of the same equation (the form used by RTKLIB's
//! `tide_pole` and ESA Navipedia), and the end-to-end model is exercised at a
//! real IERS-EOP epoch (ZIM2, 2026-05-13) with the polar motion taken from the
//! IERS Bulletin A (`finals.daily`).

use super::{mean_pole_arcsec, solid_earth_pole_tide};
use crate::tides::{TideError, TideInputErrorKind};

// IERS (2018) conventional mean pole, hard-coded per displacement-test epoch
// from the published linear secular formula and held independently of the
// implementation's `mean_pole_arcsec`, so the displacement oracle below catches
// a wrong mean-pole epoch or model instead of cancelling it out:
//
//   x_bar(t) =  55.0 + 1.677 (t - 2000.0)  mas
//   y_bar(t) = 320.5 + 3.460 (t - 2000.0)  mas
//
// with `t` the epoch in Julian years from J2000.0
// (t - 2000.0 = (MJD - 51544.5 + fhr/24) / 365.25). Values are in arcseconds.
//
// 2026-05-13 12.5h (t - 2000.0 = 26.36282226793...): the ZIM2 real-EOP epoch.
const ZIM2_MEAN_POLE_X_ARCSEC: f64 = 0.099_210_452_943_189_61;
const ZIM2_MEAN_POLE_Y_ARCSEC: f64 = 0.411_715_365_046_771_64;
// 2003-07-01 06h (t - 2000.0 = 3.49555099246...).
const E2003_MEAN_POLE_X_ARCSEC: f64 = 0.060_862_039_014_373_72;
const E2003_MEAN_POLE_Y_ARCSEC: f64 = 0.332_594_606_433_949_4;
// 2031-11-20 18.25h (t - 2000.0 = 31.88572324892...).
const E2031_MEAN_POLE_X_ARCSEC: f64 = 0.108_472_357_888_432_58;
const E2031_MEAN_POLE_Y_ARCSEC: f64 = 0.430_824_602_441_250_26;

/// ZIM2 (Zimmerwald) ITRF2020 ECEF position, from `tests/ppp_decimeter_arc.rs`.
const ZIM2_ECEF_M: [f64; 3] = [
    4_331_299.584_071_246,
    567_537.707_032_023_1,
    4_633_133.964_520_6,
];

// IERS Bulletin A polar motion for 2026-05-13 (MJD 61173), from the IERS Rapid
// Service `finals.daily`:
//   26  5 13 61173.00 I  0.169051 0.000020  0.411760 0.000023 ...
const ZIM2_XP_ARCSEC: f64 = 0.169_051;
const ZIM2_YP_ARCSEC: f64 = 0.411_760;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Geocentric (radial, north, east) decomposition of an ECEF displacement at a
/// station, used to read the displacement back in the IERS local triad.
fn ecef_to_runeu(xsta: &[f64; 3], d: &[f64; 3]) -> (f64, f64, f64) {
    let r = (xsta[0] * xsta[0] + xsta[1] * xsta[1] + xsta[2] * xsta[2]).sqrt();
    let lon = xsta[1].atan2(xsta[0]);
    let lat_gc = (xsta[2] / r).asin();
    let (sinlat, coslat) = lat_gc.sin_cos();
    let (sinlon, coslon) = lon.sin_cos();

    let up_hat = [coslat * coslon, coslat * sinlon, sinlat];
    let east_hat = [-sinlon, coslon, 0.0];
    let north_hat = [-sinlat * coslon, -sinlat * sinlon, coslat];

    let dot = |a: &[f64; 3]| a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
    (dot(&up_hat), dot(&north_hat), dot(&east_hat))
}

/// Independent latitude-form transcription of IERS Eq. (7.24) (RTKLIB
/// `tide_pole` / Navipedia), returning (up, north, east) in metres. The mean
/// pole `(x_bar, y_bar)` is supplied by the caller from hard-coded, independently
/// computed values (not the implementation's `mean_pole_arcsec`), so a mean-pole
/// epoch or model regression surfaces here rather than cancelling; any remaining
/// disagreement isolates a trig/sign error in the ECEF kernel.
fn eq724_latitude_form(
    xsta: &[f64; 3],
    x_bar_arcsec: f64,
    y_bar_arcsec: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
) -> (f64, f64, f64) {
    let r = (xsta[0] * xsta[0] + xsta[1] * xsta[1] + xsta[2] * xsta[2]).sqrt();
    let lon = xsta[1].atan2(xsta[0]);
    let lat = (xsta[2] / r).asin();

    let m1 = xp_arcsec - x_bar_arcsec;
    let m2 = -(yp_arcsec - y_bar_arcsec);

    let (sinlon, coslon) = lon.sin_cos();
    let radial = m1 * coslon + m2 * sinlon;
    let lambda = m1 * sinlon - m2 * coslon;

    let up = -33.0e-3 * (2.0 * lat).sin() * radial;
    let north = -9.0e-3 * (2.0 * lat).cos() * radial;
    let east = 9.0e-3 * lat.sin() * lambda;
    (up, north, east)
}

#[test]
fn mean_pole_secular_matches_iers_2018_coefficients() {
    // J2000.0 (2000-01-01 12:00 UTC) is years = 0, so the mean pole equals the
    // IERS (2018) linear-secular intercepts: 55.0 mas and 320.5 mas.
    let (x_bar, y_bar) = mean_pole_arcsec(2000, 1, 1, 12.0);
    assert!(approx_eq(x_bar, 0.055_0, 1.0e-12), "x_bar = {x_bar}");
    assert!(approx_eq(y_bar, 0.320_5, 1.0e-12), "y_bar = {y_bar}");

    // One Julian-year-ish step recovers the secular rates (1.677, 3.460 mas/yr).
    // 2000 is a leap year, so the step is 366/365.25 = 1.002053 yr.
    let (x1, y1) = mean_pole_arcsec(2001, 1, 1, 12.0);
    let step_yr = 366.0 / 365.25;
    assert!(
        approx_eq((x1 - x_bar) * 1000.0, 1.677 * step_yr, 1.0e-9),
        "x rate {}",
        (x1 - x_bar) * 1000.0
    );
    assert!(
        approx_eq((y1 - y_bar) * 1000.0, 3.460 * step_yr, 1.0e-9),
        "y rate {}",
        (y1 - y_bar) * 1000.0
    );
}

#[test]
fn pole_tide_matches_iers_eq724_latitude_form() {
    // A spread of stations (latitude, longitude, radius) and epochs.
    let stations = [
        ZIM2_ECEF_M,
        [4_517_590.9, 837_910.5, 4_402_330.5], // ~mid-latitude N, +lon
        [1_130_773.0, -4_831_253.0, 3_994_200.0], // N America
        [-2_409_600.0, 5_384_500.0, 2_407_800.0], // SE Asia
        [3_183_000.0, 1_421_000.0, -5_322_000.0], // southern hemisphere
    ];
    // Each epoch carries its hard-coded IERS (2018) mean pole (x_bar, y_bar)
    // arcsec, computed independently of `mean_pole_arcsec`.
    let epochs = [
        (
            2003,
            7,
            1,
            6.0,
            0.10,
            0.25,
            E2003_MEAN_POLE_X_ARCSEC,
            E2003_MEAN_POLE_Y_ARCSEC,
        ),
        (
            2026,
            5,
            13,
            12.5,
            ZIM2_XP_ARCSEC,
            ZIM2_YP_ARCSEC,
            ZIM2_MEAN_POLE_X_ARCSEC,
            ZIM2_MEAN_POLE_Y_ARCSEC,
        ),
        (
            2031,
            11,
            20,
            18.25,
            -0.05,
            0.55,
            E2031_MEAN_POLE_X_ARCSEC,
            E2031_MEAN_POLE_Y_ARCSEC,
        ),
    ];

    let mut max_dev = 0.0_f64;
    for xsta in &stations {
        for &(y, mo, d, fhr, xp, yp, x_bar, y_bar) in &epochs {
            let got =
                solid_earth_pole_tide(xsta, y, mo, d, fhr, xp, yp).expect("valid pole tide input");
            let (u_got, n_got, e_got) = ecef_to_runeu(xsta, &got);
            let (u_ref, n_ref, e_ref) = eq724_latitude_form(xsta, x_bar, y_bar, xp, yp);

            for (g, r) in [(u_got, u_ref), (n_got, n_ref), (e_got, e_ref)] {
                let dev = (g - r).abs();
                max_dev = max_dev.max(dev);
                assert!(
                    dev < 1.0e-12,
                    "eq 7.24 mismatch at {y}-{mo}-{d}: got {g:.15e}, ref {r:.15e}, dev {dev:.3e}"
                );
            }
        }
    }
    // Confirms the colatitude-form ECEF kernel reproduces the independent
    // latitude-form transcription of IERS Eq. (7.24) to round-off.
    assert!(max_dev < 1.0e-12, "max deviation {max_dev:.3e}");
}

#[test]
fn pole_tide_zim2_real_eop_has_few_mm_magnitude() {
    let d = solid_earth_pole_tide(
        &ZIM2_ECEF_M,
        2026,
        5,
        13,
        12.5,
        ZIM2_XP_ARCSEC,
        ZIM2_YP_ARCSEC,
    )
    .expect("valid ZIM2 pole tide input");

    let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    // At this epoch the instantaneous pole sits ~0.07 arcsec from the secular
    // mean pole, giving a sub-cm displacement (radial ~ -2 mm). The pole tide
    // reaches ~2-2.5 cm only when the wobble approaches its ~0.3 arcsec extreme.
    assert!(
        (0.5e-3..5.0e-3).contains(&mag),
        "ZIM2 pole tide magnitude {mag:.4e} m outside the expected few-mm band"
    );

    // End-to-end agreement with the independent latitude-form reference.
    let (u_got, n_got, e_got) = ecef_to_runeu(&ZIM2_ECEF_M, &d);
    let (u_ref, n_ref, e_ref) = eq724_latitude_form(
        &ZIM2_ECEF_M,
        ZIM2_MEAN_POLE_X_ARCSEC,
        ZIM2_MEAN_POLE_Y_ARCSEC,
        ZIM2_XP_ARCSEC,
        ZIM2_YP_ARCSEC,
    );
    assert!(approx_eq(u_got, u_ref, 1.0e-12), "up {u_got} vs {u_ref}");
    assert!(approx_eq(n_got, n_ref, 1.0e-12), "north {n_got} vs {n_ref}");
    assert!(approx_eq(e_got, e_ref, 1.0e-12), "east {e_got} vs {e_ref}");

    // Radial term is the dominant component and is negative at this epoch.
    assert!(u_got < 0.0 && u_got > -3.0e-3, "radial {u_got}");
}

#[test]
fn pole_tide_rejects_degenerate_geometry() {
    assert_invalid(
        solid_earth_pole_tide(&[0.0, 0.0, 0.0], 2026, 5, 13, 12.0, 0.1, 0.4),
        "station radius",
        TideInputErrorKind::NotPositive,
    );
    assert_invalid(
        solid_earth_pole_tide(&[0.0, 0.0, 6_378_136.6], 2026, 5, 13, 12.0, 0.1, 0.4),
        "station horizontal radius",
        TideInputErrorKind::NotPositive,
    );
}

#[test]
fn pole_tide_rejects_invalid_date_hour_and_nonfinite_pole() {
    assert_invalid(
        solid_earth_pole_tide(&ZIM2_ECEF_M, 2026, 13, 13, 12.0, 0.1, 0.4),
        "civil datetime",
        TideInputErrorKind::InvalidCivilDate,
    );
    assert_invalid(
        solid_earth_pole_tide(&ZIM2_ECEF_M, 2026, 5, 13, 24.0, 0.1, 0.4),
        "fractional hour",
        TideInputErrorKind::OutOfRange,
    );
    assert_invalid(
        solid_earth_pole_tide(&ZIM2_ECEF_M, 2026, 5, 13, 12.0, f64::NAN, 0.4),
        "polar motion xp",
        TideInputErrorKind::NonFinite,
    );
    assert_invalid(
        solid_earth_pole_tide(&ZIM2_ECEF_M, 2026, 5, 13, 12.0, 0.1, f64::INFINITY),
        "polar motion yp",
        TideInputErrorKind::NonFinite,
    );
}

fn assert_invalid(got: Result<[f64; 3], TideError>, field: &'static str, kind: TideInputErrorKind) {
    assert_eq!(
        got.expect_err("invalid pole tide input must error"),
        TideError::InvalidInput { field, kind }
    );
}
