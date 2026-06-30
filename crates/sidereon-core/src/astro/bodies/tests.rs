//! Golden validation of [`sun_moon_ecef`] against a Skyfield + JPL DE440 ITRS
//! reference (see `tests/fixtures/bodies/gen_sun_moon_golden.py`).
//!
//! `sun_moon_ecef` is the RTKLIB-style low-precision analytic series rotated
//! GCRS -> ITRS by the crate's IAU 2000A/2006 transform, so the bar is the
//! model's own accuracy (sub-degree direction), not bit-exactness. The value of
//! the golden is catching frame / units / series-coefficient regressions: a
//! wrong ECI-frame interpretation or a flipped/duplicated series term shows up
//! as degree-to-huge ITRS deviations that this test gates out.

use std::path::PathBuf;

use serde_json::Value;

use super::{sun_moon_ecef, sun_moon_eci, sun_moon_eci_at};
use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD};
use crate::astro::time::scales::TimeScales;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bodies/sun_moon_skyfield_golden.json")
        .canonicalize()
        .expect("locate sun_moon_skyfield_golden.json")
}

fn vec3(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("vec3 array");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn angle_deg(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let c = (dot / (norm(a) * norm(b))).clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

#[test]
fn sun_moon_eci_at_matches_century_argument_bit_for_bit() {
    // The time-tagged ECI entry point must form exactly the Julian-century TT
    // argument the bare series takes, so the two agree to the bit. This pins the
    // refactor that routes sun_moon_ecef through sun_moon_eci_at.
    let ts = TimeScales::from_utc(2026, 4, 30, 9, 45, 0.0).expect("valid UTC instant");
    let t = (ts.jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
    let from_t = sun_moon_eci(t).expect("valid century argument");
    let from_ts = sun_moon_eci_at(&ts).expect("valid time scales");
    for i in 0..3 {
        assert_eq!(from_ts.sun[i].to_bits(), from_t.sun[i].to_bits());
        assert_eq!(from_ts.moon[i].to_bits(), from_t.moon[i].to_bits());
    }
}

#[test]
fn sun_moon_ecef_matches_skyfield_de440_golden() {
    // Tolerances justified by the analytic model accuracy (sub-degree direction,
    // sub-percent distance) plus margin; tight enough that a frame/units/series
    // regression (degrees+) fails, loose enough that the faithful model passes.
    const SUN_DIR_TOL_DEG: f64 = 0.10;
    const SUN_DIST_TOL: f64 = 0.003; // 0.3%
    const MOON_DIR_TOL_DEG: f64 = 0.60;
    const MOON_DIST_TOL: f64 = 0.012; // 1.2%

    let raw = std::fs::read_to_string(fixture_path()).expect("read golden");
    let doc: Value = serde_json::from_str(&raw).expect("parse golden");
    let cases = doc["cases"].as_array().expect("cases");

    let (mut max_sun_dir, mut max_sun_dist) = (0.0_f64, 0.0_f64);
    let (mut max_moon_dir, mut max_moon_dist) = (0.0_f64, 0.0_f64);
    let mut failures: Vec<String> = Vec::new();

    for case in cases {
        let utc = &case["utc"];
        let ts = TimeScales::from_utc(
            utc["year"].as_i64().unwrap() as i32,
            utc["month"].as_i64().unwrap() as i32,
            utc["day"].as_i64().unwrap() as i32,
            utc["hour"].as_i64().unwrap() as i32,
            utc["minute"].as_i64().unwrap() as i32,
            utc["second"].as_f64().unwrap(),
        )
        .expect("golden UTC instant is valid");
        let sm = sun_moon_ecef(&ts).expect("valid time scales");

        let sun_ref = vec3(&case["sun_itrs_m"]);
        let moon_ref = vec3(&case["moon_itrs_m"]);

        let sun_dir = angle_deg(&sm.sun, &sun_ref);
        let moon_dir = angle_deg(&sm.moon, &moon_ref);
        let sun_dist = (norm(&sm.sun) - norm(&sun_ref)).abs() / norm(&sun_ref);
        let moon_dist = (norm(&sm.moon) - norm(&moon_ref)).abs() / norm(&moon_ref);

        max_sun_dir = max_sun_dir.max(sun_dir);
        max_moon_dir = max_moon_dir.max(moon_dir);
        max_sun_dist = max_sun_dist.max(sun_dist);
        max_moon_dist = max_moon_dist.max(moon_dist);

        if sun_dir > SUN_DIR_TOL_DEG
            || sun_dist > SUN_DIST_TOL
            || moon_dir > MOON_DIR_TOL_DEG
            || moon_dist > MOON_DIST_TOL
        {
            failures.push(format!(
                "{}: sun_dir={sun_dir:.4}deg sun_dist={sun_dist:.4e} moon_dir={moon_dir:.4}deg moon_dist={moon_dist:.4e}",
                case["utc"]
            ));
        }
    }

    println!(
        "sun_moon golden max dev: sun_dir={max_sun_dir:.4}deg sun_dist={max_sun_dist:.4e} moon_dir={max_moon_dir:.4}deg moon_dist={max_moon_dist:.4e}"
    );

    assert!(
        failures.is_empty(),
        "sun_moon_ecef golden out of tolerance:\n{}",
        failures.join("\n")
    );
}
