//! Golden-vector validation of [`solid_earth_tide`] against the staged IERS
//! DEHANTTIDEINEL reference cases.
//!
//! Fixture provenance: `tests/fixtures/tides/tides_dehant_golden.json` holds the
//! canonical test cases transcribed from the header comments of the IERS
//! Conventions routine `DEHANTTIDEINEL.F`
//! (https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel/DEHANTTIDEINEL.F).
//! Each case carries its own source citation, station/Sun/Moon vectors, UTC
//! date, fractional UTC hour `FHR`, and the expected geocentric ITRF
//! displacement vector (metres; hours for `FHR`). Only reference test-case data
//! is vendored, not the Fortran routine; the IERS Conventions Software License
//! grants free use including commercial use and distribution of derived work
//! with attribution to the IERS origin.
//!
//! `tests/fixtures/tides/dehanttideinel_oracle.json` holds the output of
//! `DEHANTTIDEINEL` itself, compiled from the IERS source by
//! `fixtures-generators/dehanttideinel_oracle/generate.sh`, for the four header
//! cases and a grid of eight stations over thirteen UTC dates from 1958 to 2040.

use std::path::PathBuf;

use serde_json::Value;

use super::{
    solid_earth_tide, solid_earth_tide_unchecked, solid_earth_tide_with_constants,
    tai_minus_utc_seconds, StationTideConstants, TideError, TideInputErrorKind,
    DIURNAL_BAND_CONVENTIONS, DIURNAL_BAND_IERS_ROUTINE,
};

fn fixture_path() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .join("tests/fixtures/tides/tides_dehant_golden.json")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures/tides/tides_dehant_golden.json from {}: {e}",
                crate_dir.display()
            )
        })
}

fn vec3(v: &Value) -> [f64; 3] {
    let a = v["values"].as_array().expect("values array");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

#[test]
fn solid_earth_tide_matches_iers_dehant_golden() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read tides_dehant_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse tides_dehant_golden.json");
    let cases = doc["cases"].as_array().expect("cases array");

    // Sub-nanometre tolerance: a faithful translation reproduces the IERS
    // reference displacement to far below any geodetically meaningful level.
    const TOL_M: f64 = 1.0e-9;

    let mut failures: Vec<String> = Vec::new();
    let mut max_dev = 0.0_f64;

    for case in cases {
        let id = case["id"].as_str().unwrap_or("?");
        assert!(
            case["source"].as_str().is_some_and(|source| {
                source.contains("IERS Conventions") && source.contains("DEHANTTIDEINEL")
            }),
            "{id} must cite its source row"
        );
        // The DEHANTTIDEINEL.F header prints case 3's output again under case
        // 4, whose `xsun` input (~0.06 AU) is not a physical Sun distance. The
        // routine built from source returns a different displacement for those
        // inputs; `solid_earth_tide_matches_dehanttideinel_built_from_source`
        // checks case 4 against that output instead.
        if id == "case_4_2017_01_15" {
            continue;
        }
        let inputs = &case["inputs"];
        let xsta = vec3(&inputs["xsta_m"]);
        let xsun = vec3(&inputs["xsun_m"]);
        let xmon = vec3(&inputs["xmon_m"]);
        let year = inputs["date_utc"]["year"].as_i64().unwrap() as i32;
        let month = inputs["date_utc"]["month"].as_i64().unwrap() as i32;
        let day = inputs["date_utc"]["day"].as_i64().unwrap() as i32;
        let fhr = inputs["fhr_hours"]["value"].as_f64().unwrap();

        let expected = vec3(&case["expected"]["dxtide_m"]);
        // The header values come from the routine, so its own constants.
        let got = solid_earth_tide_with_constants(
            &xsta,
            year,
            month,
            day,
            fhr,
            &xsun,
            &xmon,
            StationTideConstants::IersRoutine,
        )
        .expect("valid tide input");

        for k in 0..3 {
            let dev = (got[k] - expected[k]).abs();
            if dev > max_dev {
                max_dev = dev;
            }
            if dev > TOL_M {
                failures.push(format!(
                    "{id} component {k}: got {:.18e}, expected {:.18e}, dev {:.3e} m",
                    got[k], expected[k], dev
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "solid-earth tide golden mismatch (max dev {max_dev:.3e} m):\n{}",
        failures.join("\n")
    );
}

#[test]
fn solid_earth_tide_matches_dehanttideinel_built_from_source() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tides/dehanttideinel_oracle.json");
    let raw = std::fs::read_to_string(&path).expect("read dehanttideinel_oracle.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse dehanttideinel_oracle.json");
    let cases = doc["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 108, "4 header cases and 8 stations x 13 dates");

    let array3 = |v: &Value| -> [f64; 3] {
        let a = v.as_array().expect("3-vector");
        assert_eq!(a.len(), 3);
        [
            a[0].as_f64().unwrap(),
            a[1].as_f64().unwrap(),
            a[2].as_f64().unwrap(),
        ]
    };

    let mut failures = Vec::new();
    let mut max_dev = 0.0_f64;
    let mut max_bound = 0.0_f64;
    let mut max_bound_fraction = 0.0_f64;
    let mut exact = 0_usize;
    for case in cases {
        let id = case["id"].as_str().expect("case id");
        let xsta = array3(&case["xsta_m"]);
        let xsun = array3(&case["xsun_m"]);
        let xmon = array3(&case["xmon_m"]);
        let year = case["year"].as_i64().unwrap() as i32;
        let month = case["month"].as_i64().unwrap() as i32;
        let day = case["day"].as_i64().unwrap() as i32;
        let fhr = case["fhr_hours"].as_f64().unwrap();
        let expected = array3(&case["dxtide_m"]);
        assert!((1958..=2040).contains(&year));

        let got = solid_earth_tide_with_constants(
            &xsta,
            year,
            month,
            day,
            fhr,
            &xsun,
            &xmon,
            StationTideConstants::IersRoutine,
        )
        .expect("valid tide input");
        let tolerance = dehant_oracle_roundoff_bound(&xsun, &xmon);
        max_bound = max_bound.max(tolerance);
        for k in 0..3 {
            let dev = (got[k] - expected[k]).abs();
            if got[k] == expected[k] {
                exact += 1;
            }
            max_dev = max_dev.max(dev);
            max_bound_fraction = max_bound_fraction.max(dev / tolerance);
            if !dev.is_finite() || dev > tolerance {
                failures.push(format!(
                    "{id} component {k}: got {:.17e}, DEHANTTIDEINEL {:.17e}, dev {dev:.3e} m exceeds derived bound {tolerance:.3e} m",
                    got[k], expected[k],
                ));
            }
        }
    }
    println!(
        "DEHANTTIDEINEL: {exact} of {} components bit-identical, max dev {max_dev:.3e} m, \
         max derived bound {max_bound:.3e} m, max bound fraction {max_bound_fraction:.3e}",
        3 * cases.len()
    );
    assert!(
        failures.is_empty(),
        "DEHANTTIDEINEL mismatch (max dev {max_dev:.3e} m):\n{}",
        failures.join("\n")
    );
}

fn dehant_oracle_roundoff_bound(xsun: &[f64; 3], xmon: &[f64; 3]) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_378_136.6;
    const MASS_RATIO_SUN: f64 = 332_946.048_2;
    const MASS_RATIO_MOON: f64 = 0.0123000371;
    let distance = |vector: &[f64; 3]| {
        (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
    };
    let scale = |mass_ratio: f64, radius: f64| {
        let earth_to_body = EARTH_RADIUS_M / radius;
        mass_ratio * EARTH_RADIUS_M * earth_to_body * earth_to_body * earth_to_body
    };
    let sun_degree_two = scale(MASS_RATIO_SUN, distance(xsun));
    let moon_degree_two = scale(MASS_RATIO_MOON, distance(xmon));
    let magnitude_m = 8.0
        * (sun_degree_two
            + moon_degree_two
            + sun_degree_two * EARTH_RADIUS_M / distance(xsun)
            + moon_degree_two * EARTH_RADIUS_M / distance(xmon))
        + 0.1;
    let unit_roundoff = f64::EPSILON / 2.0;
    let gamma = |operations: f64| operations * unit_roundoff / (1.0 - operations * unit_roundoff);
    let phase_error = 2.0 * gamma(64.0) * (2.0e6_f64.to_radians()) + 8.0 * unit_roundoff;
    2.0 * gamma(512.0) * magnitude_m + 0.1 * phase_error
}

#[test]
fn conventions_diurnal_table_matches_the_table_7_2_derivation() {
    // Equation (7.12c): dR = -(3/2) sqrt(5/(24 pi)) Hf dh and
    // dT = -3 sqrt(5/(24 pi)) Hf dl, in and out of phase, with dh and dl the
    // Table 7.2 Love and Shida numbers less the nominal h2 = 0.6078,
    // hI = -0.0025, l2 = 0.0847 and lI = -0.0007, and Hf the
    // Cartwright-Tayler-Edden amplitude of the IERS routine ADMINT.F (m).
    // Rows: Doodson number, DATDI row (0-based), Hf, h(0)R, h(0)I, l(0)R,
    // l(0)I (NaN where Table 7.2 lists no l, for pi1).
    #[allow(clippy::approx_constant)]
    const ROWS: [(u32, usize, f64, f64, f64, f64, f64); 11] = [
        (135_655, 3, -0.050208, 0.6036, -0.0026, 0.0846, -0.0006),
        (145_545, 5, -0.04947, 0.6028, -0.0025, 0.0846, -0.0006),
        (145_555, 6, -0.262232, 0.6028, -0.0025, 0.0846, -0.0006),
        (155_655, 10, 0.020613, 0.6005, -0.0023, 0.0847, -0.0006),
        (162_556, 13, -0.007131, 0.5878, -0.0015, f64::NAN, f64::NAN),
        (163_555, 15, -0.121995, 0.5817, -0.0011, 0.0853, -0.0006),
        (165_545, 18, -0.0073, 0.5283, 0.0023, 0.0869, -0.0006),
        (165_555, 19, 0.368645, 0.5236, 0.0030, 0.0870, -0.0006),
        (165_565, 20, 0.050031, 0.5182, 0.0036, 0.0872, -0.0006),
        (166_554, 22, 0.002885, 1.0569, 0.0036, 0.0710, -0.0020),
        (167_555, 26, 0.005249, 0.6645, -0.0059, 0.0828, -0.0007),
    ];
    let c = (5.0 / (24.0 * std::f64::consts::PI)).sqrt();
    let mut routine_outside = Vec::new();
    for (doodson, row, hf, h_re, h_im, l_re, l_im) in ROWS {
        let derived = [
            -1.5 * c * hf * (h_re - 0.6078) * 1.0e3,
            -1.5 * c * hf * (h_im + 0.0025) * 1.0e3,
            -3.0 * c * hf * (l_re - 0.0847) * 1.0e3,
            -3.0 * c * hf * (l_im + 0.0007) * 1.0e3,
        ];
        // Table 7.2 prints four decimals, so each derived amplitude is known
        // to its factor times |Hf| times 5e-5, and the tables print two
        // decimals, 0.005 mm more.
        let bound = [
            1.5 * c * hf.abs() * 5.0e-5 * 1.0e3 + 0.005,
            1.5 * c * hf.abs() * 5.0e-5 * 1.0e3 + 0.005,
            3.0 * c * hf.abs() * 5.0e-5 * 1.0e3 + 0.005,
            3.0 * c * hf.abs() * 5.0e-5 * 1.0e3 + 0.005,
        ];
        for k in 0..4 {
            if derived[k].is_nan() {
                continue;
            }
            let conventions = DIURNAL_BAND_CONVENTIONS[row][5 + k];
            assert!(
                (conventions - derived[k]).abs() <= bound[k] + 1.0e-12,
                "{doodson} column {k}: table {conventions}, derived {:.4}, bound {:.4}",
                derived[k],
                bound[k]
            );
            let routine = DIURNAL_BAND_IERS_ROUTINE[row][5 + k];
            if (routine - derived[k]).abs() > bound[k] + 1.0e-12 {
                routine_outside.push((doodson, k));
            }
        }
    }
    // The routine departs from the derivation exactly where the Conventions
    // table was corrected: K1 and P1 out-of-phase radial.
    assert_eq!(routine_outside, vec![(163_555, 1), (165_555, 1)]);

    // Row 25 is tide 166,564 in the Conventions table: s, h, p, N', ps
    // multipliers 1, 1, 0, 1, -1 (Doodson 1 1 1 0 1 -1).
    assert_eq!(
        DIURNAL_BAND_CONVENTIONS[24][..5],
        [1.0, 1.0, 0.0, 1.0, -1.0]
    );
    assert_eq!(
        DIURNAL_BAND_IERS_ROUTINE[24][..5],
        [0.0, 1.0, 0.0, 1.0, -1.0]
    );
    // Every other entry is the routine's.
    for row in 0..31 {
        for col in 0..9 {
            if (row, col) != (15, 6) && (row, col) != (19, 6) && (row, col) != (24, 0) {
                assert_eq!(
                    DIURNAL_BAND_CONVENTIONS[row][col].to_bits(),
                    DIURNAL_BAND_IERS_ROUTINE[row][col].to_bits(),
                    "row {row} column {col}"
                );
            }
        }
    }
}

#[test]
fn conventions_and_routine_constants_differ_by_under_the_changed_amplitudes() {
    // The three corrected rows change amplitudes by 0.02 (K1), 0.14 (P1) and
    // 0.01 mm moved to another frequency (row 25, up to 0.02 mm), so the two
    // variants can differ by at most 0.18 mm in any component.
    const BOUND_M: f64 = 0.18e-3;
    let raw = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tides/dehanttideinel_oracle.json"),
    )
    .expect("read dehanttideinel_oracle.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse dehanttideinel_oracle.json");
    let mut max_diff = 0.0_f64;
    for case in doc["cases"].as_array().expect("cases array") {
        if case["id"] == "header_case_4" {
            continue;
        }
        let v = |key: &str| -> [f64; 3] {
            let a = case[key].as_array().expect("3-vector");
            [
                a[0].as_f64().unwrap(),
                a[1].as_f64().unwrap(),
                a[2].as_f64().unwrap(),
            ]
        };
        let (xsta, xsun, xmon) = (v("xsta_m"), v("xsun_m"), v("xmon_m"));
        let year = case["year"].as_i64().unwrap() as i32;
        let month = case["month"].as_i64().unwrap() as i32;
        let day = case["day"].as_i64().unwrap() as i32;
        let fhr = case["fhr_hours"].as_f64().unwrap();
        let [a, b] = [
            StationTideConstants::Conventions,
            StationTideConstants::IersRoutine,
        ]
        .map(|constants| {
            solid_earth_tide_with_constants(&xsta, year, month, day, fhr, &xsun, &xmon, constants)
                .expect("valid tide input")
        });
        for k in 0..3 {
            max_diff = max_diff.max((a[k] - b[k]).abs());
        }
    }
    println!("Conventions minus IersRoutine constants: max {max_diff:.3e} m");
    assert!(max_diff > 0.0, "the variants differ");
    assert!(max_diff <= BOUND_M, "max difference {max_diff:.3e} m");
}

#[test]
fn the_tide_delta_at_follows_sofa_dat_before_1972() {
    // SOFA DAT as distributed with DEHANTTIDEINEL: 0 before 1960, then the
    // 1960-1971 offsets plus drift from the reference MJD. 1965-03-10 is MJD
    // 38829, so 3.6401300 + (38829 + 0.5 - 38761) * 0.001296 at noon.
    assert_eq!(tai_minus_utc_seconds(1959, 12, 31, 0.5), 0.0);
    assert_eq!(
        tai_minus_utc_seconds(1965, 3, 10, 0.5),
        3.6401300 + (38829.0 + 0.5 - 38761.0) * 0.001296
    );
    assert_eq!(
        tai_minus_utc_seconds(1971, 12, 31, 0.0),
        4.2131700 + (41316.0 - 39126.0) * 0.002592
    );
    assert_eq!(tai_minus_utc_seconds(1972, 1, 1, 0.0), 10.0);
}

fn assert_invalid_input(
    got: Result<[f64; 3], TideError>,
    field: &'static str,
    kind: TideInputErrorKind,
) {
    assert_eq!(
        got.expect_err("invalid tide input must error"),
        TideError::InvalidInput { field, kind }
    );
}

fn valid_tide_inputs() -> ([f64; 3], [f64; 3], [f64; 3]) {
    (
        [3_512_900.0, 780_500.0, 5_248_700.0],
        [
            1.379_133_792_566_993e11,
            -5.521_095_241_319_248e10,
            -2.394_349_831_958_611e10,
        ],
        [
            1.749_761_742_158_154e8,
            -3.202_053_263_558_994e8,
            -1.746_291_411_625_388e8,
        ],
    )
}

#[test]
fn solid_earth_tide_rejects_degenerate_geometry() {
    let (station, sun, moon) = valid_tide_inputs();

    assert_invalid_input(
        solid_earth_tide(&[0.0, 0.0, 0.0], 2020, 6, 24, 12.0, &sun, &moon),
        "station radius",
        TideInputErrorKind::NotPositive,
    );
    assert_invalid_input(
        solid_earth_tide(&[0.0, 0.0, 6_378_136.6], 2020, 6, 24, 12.0, &sun, &moon),
        "station horizontal radius",
        TideInputErrorKind::NotPositive,
    );
    assert_invalid_input(
        solid_earth_tide(&station, 2020, 6, 24, 12.0, &[0.0, 0.0, 0.0], &moon),
        "sun radius",
        TideInputErrorKind::NotPositive,
    );
    assert_invalid_input(
        solid_earth_tide(&station, 2020, 6, 24, 12.0, &sun, &[0.0, 0.0, 0.0]),
        "moon radius",
        TideInputErrorKind::NotPositive,
    );
}

#[test]
fn solid_earth_tide_rejects_invalid_civil_date_and_hour() {
    let (station, sun, moon) = valid_tide_inputs();

    assert_invalid_input(
        solid_earth_tide(&station, 2020, 13, 24, 12.0, &sun, &moon),
        "civil datetime",
        TideInputErrorKind::InvalidCivilDate,
    );
    assert_invalid_input(
        solid_earth_tide(&station, 2021, 2, 31, 12.0, &sun, &moon),
        "civil datetime",
        TideInputErrorKind::InvalidCivilDate,
    );
    assert_invalid_input(
        solid_earth_tide(&station, 2020, 6, 24, 24.0, &sun, &moon),
        "fractional hour",
        TideInputErrorKind::OutOfRange,
    );
    assert_invalid_input(
        solid_earth_tide(&station, 2020, 6, 24, -0.25, &sun, &moon),
        "fractional hour",
        TideInputErrorKind::OutOfRange,
    );
}

#[test]
fn solid_earth_tide_valid_date_and_hour_matches_unchecked_result() {
    let (station, sun, moon) = valid_tide_inputs();

    let got = solid_earth_tide(&station, 2020, 6, 24, 23.5, &sun, &moon).expect("valid tide input");
    let expected = solid_earth_tide_unchecked(
        &station,
        2020,
        6,
        24,
        23.5,
        &sun,
        &moon,
        StationTideConstants::Conventions,
    );

    assert_eq!(got, expected);
}

#[test]
fn the_tide_leap_second_table_equals_the_main_table_month_by_month() {
    // The SOFA DAT table kept for the tides is checked against
    // `find_leap_seconds` on the first and last day of every month from
    // 1972 to 2035.
    for year in 1972..=2035 {
        for month in 1..=12 {
            for day in [
                1,
                crate::astro::time::civil::days_in_month(i64::from(year), i64::from(month)),
            ] {
                let jd = crate::astro::time::scales::julian_day_number(year, month, day as i32)
                    as f64
                    - 0.5;
                assert_eq!(
                    tai_minus_utc_seconds(year, month, day as i32, 0.0),
                    crate::astro::time::scales::find_leap_seconds(jd),
                    "{year}-{month:02}-{day:02}"
                );
            }
        }
    }
}
