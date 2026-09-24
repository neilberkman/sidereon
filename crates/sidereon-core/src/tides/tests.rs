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
    solid_earth_tide, solid_earth_tide_unchecked, tai_minus_utc_seconds, TideError,
    TideInputErrorKind,
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
        let got =
            solid_earth_tide(&xsta, year, month, day, fhr, &xsun, &xmon).expect("valid tide input");

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

        let got =
            solid_earth_tide(&xsta, year, month, day, fhr, &xsun, &xmon).expect("valid tide input");
        for k in 0..3 {
            // The two sides differ only by the last-bit rounding of the
            // platform trigonometric functions and of a few regrouped
            // products, far below 1e-15 m for displacements of 0.1 m. Header
            // case 4, with its Sun at 0.06 AU, displaces the station by tens of
            // metres, so the bound also scales with the value.
            let tolerance = 1.0e-15_f64.max(8.0 * f64::EPSILON * expected[k].abs());
            let dev = (got[k] - expected[k]).abs();
            max_dev = max_dev.max(dev);
            if dev > tolerance {
                failures.push(format!(
                    "{id} component {k}: got {:.17e}, DEHANTTIDEINEL {:.17e}, dev {dev:.3e} m",
                    got[k], expected[k]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "DEHANTTIDEINEL mismatch (max dev {max_dev:.3e} m):\n{}",
        failures.join("\n")
    );
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
    let expected = solid_earth_tide_unchecked(&station, 2020, 6, 24, 23.5, &sun, &moon);

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
