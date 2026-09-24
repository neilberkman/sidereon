//! SGP4 against python-sgp4 over Vallado's verification set.
//!
//! `sgp4_verification.json` is python-sgp4 2.22's own output (the compiled
//! extension, Vallado's C++ of 2020-07-13, WGS72, opsmode 'i', built for arm64
//! macOS without fused multiply-add), written by
//! `fixtures-generators/generate_sgp4_verification.py`: the 33 element sets of
//! `SGP4-VER.TLE` at the times Vallado's verification driver prints and at 0,
//! 120, 360, 720, 1080 and 1440 minutes, 721 states, 21 of them python-sgp4
//! error codes.
//!
//! Each state is checked two ways, neither against this crate's own output:
//!
//! * Against python-sgp4. This crate runs the same operations with the
//!   portable `libm` crate, and python-sgp4 with the platform's libm. The two
//!   libraries' `sin`, `cos`, `atan2` and `pow` each stay within 1 ulp of the
//!   true value. Each error-free state carries a per-component bound on how
//!   far that alone can move it, derived by
//!   `fixtures-generators/sgp4_libm_bound.py` from SGP4's first-order
//!   sensitivity to every libm call and to every re-rounding downstream of
//!   one. Every component must lie within its bound, and every error state
//!   must be refused with python-sgp4's error code.
//! * Bit for bit against `rust_libm`: python-sgp4's model rewritten into the
//!   C++'s operation order (`fixtures-generators/sgp4_vallado_order.py`) and
//!   run with a statement-for-statement port of the `libm` crate's functions
//!   (`fixtures-generators/rust_libm_port.py`). That is the computation this
//!   crate performs, so every state must match it to the bit and every error
//!   code must be its code.

use sidereon_core::astro::sgp4::{
    propagate_elements, ElementSet, Error, JulianDate, MinutesSinceEpoch, OpsMode, Prediction,
    Satellite,
};
use sidereon_core::astro::tle;
use sidereon_core::astro::tle::TlePolicy;

/// Verification states python-sgp4 gives as numbers that this crate
/// reproduces bit for bit (the rest differ within the libm bound).
const VERIFICATION_EXACT: usize = 379;
/// Read a verification-set TLE as Vallado's `twoline2rv` does. The set's
/// element sets 33333, 33334 and 33335 carry checksum digits that disagree
/// with their lines, which the reference reader ignores, so they are read
/// under the lenient policy rather than the strict default.
fn vallado_satellite(line1: &str, line2: &str, opsmode: OpsMode) -> Satellite {
    Satellite::from_tle_with_policy(line1, line2, opsmode, TlePolicy::Lenient)
        .expect("verification TLE initializes")
        .0
}

fn hex_to_f64(s: &str) -> f64 {
    let (neg, rest) = if let Some(r) = s.strip_prefix("-0x") {
        (true, r)
    } else if let Some(r) = s.strip_prefix("0x") {
        (false, r)
    } else if s == "nan" {
        return f64::NAN;
    } else if s == "inf" {
        return f64::INFINITY;
    } else if s == "-inf" {
        return f64::NEG_INFINITY;
    } else {
        panic!("bad hex float: {s}");
    };
    let (mant_str, exp_str) = rest.split_once('p').unwrap();
    let (int_part, frac_part) = mant_str.split_once('.').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    let full = format!("{int_part}{frac_part}");
    let mant = u64::from_str_radix(&full, 16).unwrap();
    let frac_bits = frac_part.len() as i32 * 4;
    let val = mant as f64 * (2.0_f64).powi(exp - frac_bits);
    if neg {
        -val
    } else {
        val
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    // Order the bit patterns monotonically so the distance across zero counts.
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    (key(a) as i128 - key(b) as i128).unsigned_abs() as u64
}

const LABELS: [&str; 6] = ["px", "py", "pz", "vx", "vy", "vz"];

fn components(prediction: &Prediction) -> [f64; 6] {
    [
        prediction.position[0],
        prediction.position[1],
        prediction.position[2],
        prediction.velocity[0],
        prediction.velocity[1],
        prediction.velocity[2],
    ]
}

/// Tally of one set of states against python-sgp4 and the Rust-libm model.
#[derive(Default)]
struct Agreement {
    /// Error-free python-sgp4 states checked against their bound.
    bounded: usize,
    /// States python-sgp4 gives as numbers, bit-identical to them.
    exact: usize,
    /// States python-sgp4 gives as numbers.
    finite: usize,
    /// python-sgp4 error codes.
    errors: usize,
    /// python-sgp4 code-0 states that are not finite.
    non_finite: usize,
    max_ulp: u64,
    max_fraction_of_bound: f64,
    failures: Vec<String>,
}

impl Agreement {
    fn check(&mut self, what: &str, result: Result<Prediction, Error>, row: &serde_json::Value) {
        let rust = &row["rust_libm"];
        // The Rust-libm model: this crate's computation, bit for bit.
        match (&result, rust.get("error")) {
            (Err(Error::Sgp4 { code }), Some(want))
                if i64::from(*code) == want.as_i64().unwrap() => {}
            (Err(Error::NonFiniteOutput { .. }), None) if rust.get("non_finite").is_some() => {}
            (Ok(prediction), None) if rust.get("non_finite").is_none() => {
                let actual = components(prediction);
                for (i, label) in LABELS.iter().enumerate() {
                    let want = hex_to_f64(rust[*label].as_str().unwrap());
                    if actual[i].to_bits() != want.to_bits() {
                        self.failures.push(format!(
                            "{what} {label}: {:e} vs the Rust-libm model {want:e}, {} ulp",
                            actual[i],
                            ulp_distance(actual[i], want)
                        ));
                    }
                }
            }
            (other, _) => self.failures.push(format!(
                "{what}: got {other:?}, the Rust-libm model gives {rust}"
            )),
        }

        // python-sgp4.
        if let Some(code) = row.get("error") {
            self.errors += 1;
            let code = code.as_i64().unwrap();
            if !matches!(&result, Err(Error::Sgp4 { code: ours }) if i64::from(*ours) == code) {
                self.failures
                    .push(format!("{what}: python-sgp4 error {code}, got {result:?}"));
            }
            return;
        }
        if row.get("non_finite").is_some() {
            // python-sgp4 returns code 0 with a NaN state; this crate reports
            // the non-finite output as an error instead of returning it.
            self.non_finite += 1;
            if !matches!(&result, Err(Error::NonFiniteOutput { .. })) {
                self.failures.push(format!(
                    "{what}: python-sgp4 returns a non-finite state, got {result:?}"
                ));
            }
            return;
        }
        let Ok(prediction) = &result else {
            self.failures
                .push(format!("{what}: python-sgp4 propagates, got {result:?}"));
            return;
        };
        self.finite += 1;
        let actual = components(prediction);
        let expected: Vec<f64> = LABELS
            .iter()
            .map(|label| hex_to_f64(row[*label].as_str().unwrap()))
            .collect();
        if actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            self.exact += 1;
        }
        for (a, b) in actual.iter().zip(&expected) {
            self.max_ulp = self.max_ulp.max(ulp_distance(*a, *b));
        }
        let Some(bounds) = row["bound"].as_array() else {
            return;
        };
        self.bounded += 1;
        for (i, label) in LABELS.iter().enumerate() {
            let bound = bounds[i].as_f64().unwrap();
            let difference = (actual[i] - expected[i]).abs();
            self.max_fraction_of_bound = self.max_fraction_of_bound.max(difference / bound);
            if difference.is_nan() || difference > bound {
                self.failures.push(format!(
                    "{what} {label}: {} vs python-sgp4 {:e}, difference {difference:e} over bound {bound:e}",
                    actual[i], expected[i]
                ));
            }
        }
    }

    fn finish(&self, label: &str) {
        eprintln!(
            "{label}: {} python-sgp4 states, {} bit-identical, largest difference {} ulp; {} within the bound, at most {:.4} of it; {} error codes; {} non-finite",
            self.finite,
            self.exact,
            self.max_ulp,
            self.bounded,
            self.max_fraction_of_bound,
            self.errors,
            self.non_finite,
        );
        assert!(
            self.failures.is_empty(),
            "{} disagreements:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

fn fixture() -> serde_json::Value {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();
    assert_eq!(data["sgp4_version"], "2.22");
    data
}

fn check_rows(
    agreement: &mut Agreement,
    satellite: &Satellite,
    norad: &str,
    rows: &serde_json::Value,
) {
    for row in rows.as_array().unwrap() {
        let tsince = row["tsince"].as_f64().unwrap();
        let what = format!("{norad} t={tsince:e}");
        agreement.check(&what, satellite.propagate(MinutesSinceEpoch(tsince)), row);
    }
}

#[test]
fn verification_set_matches_python_sgp4_and_the_rust_libm_model() {
    let data = fixture();
    let mut agreement = Agreement::default();
    for sat in data["satellites"].as_array().unwrap() {
        let satellite = vallado_satellite(
            sat["line1"].as_str().unwrap(),
            sat["line2"].as_str().unwrap(),
            OpsMode::Improved,
        );
        check_rows(
            &mut agreement,
            &satellite,
            sat["norad"].as_str().unwrap(),
            &sat["propagations"],
        );
    }
    agreement.finish("verification set");
    assert_eq!(
        (agreement.finite, agreement.errors, agreement.bounded),
        (700, 21, 700)
    );
    assert_eq!(agreement.exact, VERIFICATION_EXACT);
}

/// The ISS at split Julian dates, against python-sgp4's `Satrec.sgp4(jd, fr)`.
#[test]
fn split_julian_dates_match_python_sgp4_and_the_rust_libm_model() {
    let data = fixture();
    let satellite = Satellite::from_tle(
        "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
        "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
    )
    .unwrap();
    let mut agreement = Agreement::default();
    for row in data["iss_split_jd_tests"].as_array().unwrap() {
        let jd = JulianDate(
            row["jd_whole"].as_f64().unwrap(),
            row["jd_fraction"].as_f64().unwrap(),
        );
        agreement.check(
            row["label"].as_str().unwrap(),
            satellite.propagate_jd(jd),
            row,
        );
    }
    agreement.finish("ISS split Julian dates");
    assert_eq!(agreement.bounded, 4);
}

#[test]
fn iss_basic_propagation() {
    let sat = Satellite::from_tle(
        "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
        "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
    )
    .unwrap();

    let pred = sat.propagate(MinutesSinceEpoch(0.0)).unwrap();

    // Position should be in the right ballpark (LEO, ~6700-7000 km from center)
    let r = (pred.position[0].powi(2) + pred.position[1].powi(2) + pred.position[2].powi(2)).sqrt();
    assert!(
        (6500.0..=7200.0).contains(&r),
        "ISS radius {r} km outside LEO range"
    );
}

#[test]
fn julian_date_propagation() {
    let sat = Satellite::from_tle(
        "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
        "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
    )
    .unwrap();

    // 2018-07-04 00:00:00 UTC = JD 2458303.5
    let pred = sat.propagate_jd(JulianDate(2458303.0, 0.5)).unwrap();

    let r = (pred.position[0].powi(2) + pred.position[1].powi(2) + pred.position[2].powi(2)).sqrt();
    assert!(
        (6500.0..=7200.0).contains(&r),
        "ISS radius {r} km outside LEO range"
    );
}

#[test]
fn invalid_tle_rejected() {
    assert!(Satellite::from_tle("garbage", "data").is_err());
    assert!(Satellite::from_tle("1 short", "2 short").is_err());
}

/// Verifies that `Satellite::from_elements` (the pre-parsed-elements path
/// added in 0.6.1) produces *bit-identical* propagation results to
/// `Satellite::from_tle` (the TLE-string path) for the entire 33-satellite
/// Vallado verification corpus. This locks down equivalence of the two
/// public constructors.
#[test]
fn from_elements_matches_from_tle_bit_exact() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    // The fixture's element bit-patterns are stored in *radians* (the SGP4
    // internal-units form), but `ElementSet` takes degrees. We can't recover
    // the exact deg → rad inputs from the radian bits without precision loss.
    // Instead, parse the same TLE strings the fixture uses, then independently
    // build an `ElementSet` directly from the TLE line slices and compare
    // outputs to the from_tle path.
    let mut total_checks = 0;
    let mut mismatches = Vec::new();

    for sat in data["satellites"].as_array().unwrap() {
        let line1 = sat["line1"].as_str().unwrap();
        let line2 = sat["line2"].as_str().unwrap();
        let norad = sat["norad"].as_str().unwrap();

        let from_tle = vallado_satellite(line1, line2, OpsMode::Improved);

        // Build the ElementSet by parsing the TLE the same way sidereon does
        // (Elixir-side string slicing into rev/day², rev/day³, deg, etc.).
        // Lenient for the verification set's checksum mismatches (see
        // `vallado_satellite`).
        let epoch = tle::parse_with_policy(line1, line2, TlePolicy::Lenient)
            .unwrap()
            .elements
            .to_element_set()
            .expect("valid TLE bridge")
            .epoch;
        let mean_motion_dot: f64 = line1[33..43].trim().parse().unwrap();
        let nddot_str = format!("{}.{}", &line1[44..45], &line1[45..50]);
        let nddot_mantissa: f64 = nddot_str.trim().parse().unwrap_or(0.0);
        let nexp: i32 = line1[50..52].trim().parse().unwrap_or(0);
        let mean_motion_double_dot = nddot_mantissa * 10.0_f64.powi(nexp);
        let bstar_str = format!("{}.{}", &line1[53..54], &line1[54..59]);
        let bstar_mantissa: f64 = bstar_str.trim().parse().unwrap_or(0.0);
        let ibexp: i32 = line1[59..61].trim().parse().unwrap_or(0);
        let bstar = bstar_mantissa * 10.0_f64.powi(ibexp);

        let inclination_deg: f64 = line2[8..16].trim().parse().unwrap();
        let right_ascension_deg: f64 = line2[17..25].trim().parse().unwrap();
        let ecco_str = format!("0.{}", line2[26..33].replace(' ', "0"));
        let eccentricity: f64 = ecco_str.parse().unwrap();
        let argument_of_perigee_deg: f64 = line2[34..42].trim().parse().unwrap();
        let mean_anomaly_deg: f64 = line2[43..51].trim().parse().unwrap();
        let mean_motion_rev_per_day: f64 = line2[52..63].trim().parse().unwrap();

        let elements = ElementSet {
            epoch,
            bstar,
            mean_motion_dot: Some(mean_motion_dot),
            mean_motion_double_dot: Some(mean_motion_double_dot),
            eccentricity,
            argument_of_perigee_deg,
            inclination_deg,
            mean_anomaly_deg,
            mean_motion_rev_per_day,
            right_ascension_deg,
            catalog_number: None,
            omm_epoch_days: None,
        };

        let from_elem = Satellite::from_elements(&elements).unwrap();

        for prop in sat["propagations"].as_array().unwrap() {
            if prop.get("error").is_some() {
                continue;
            }
            let tsince = prop["tsince"].as_f64().unwrap();
            let pa = match from_tle.propagate(MinutesSinceEpoch(tsince)) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let pb = match from_elem.propagate(MinutesSinceEpoch(tsince)) {
                Ok(p) => p,
                Err(_) => continue,
            };
            // Also exercise the one-shot propagate_elements free function.
            let pc = propagate_elements(&elements, MinutesSinceEpoch(tsince)).unwrap();

            for i in 0..3 {
                total_checks += 1;
                if pa.position[i].to_bits() != pb.position[i].to_bits()
                    || pa.position[i].to_bits() != pc.position[i].to_bits()
                {
                    mismatches.push(format!(
                        "{norad} t={tsince} pos[{i}] tle={:?} elem={:?} oneshot={:?}",
                        pa.position[i], pb.position[i], pc.position[i]
                    ));
                }
                if pa.velocity[i].to_bits() != pb.velocity[i].to_bits()
                    || pa.velocity[i].to_bits() != pc.velocity[i].to_bits()
                {
                    mismatches.push(format!(
                        "{norad} t={tsince} vel[{i}] tle={:?} elem={:?} oneshot={:?}",
                        pa.velocity[i], pb.velocity[i], pc.velocity[i]
                    ));
                }
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{}/{} mismatches between from_tle and from_elements:\n{}",
        mismatches.len(),
        total_checks * 2,
        mismatches
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Sanity check that AFSPC and Improved opsmodes produce *different* results
/// for at least some satellites. The two modes are not bit-equivalent - if
/// they ever produce identical output across the entire fixture, our opsmode
/// plumbing is broken (likely passing the same char in both branches).
#[test]
fn opsmode_afspc_differs_from_improved_for_some_sats() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    let mut any_diff = false;
    for sat in data["satellites"].as_array().unwrap() {
        let line1 = sat["line1"].as_str().unwrap();
        let line2 = sat["line2"].as_str().unwrap();
        let imp = vallado_satellite(line1, line2, OpsMode::Improved);
        let afspc = vallado_satellite(line1, line2, OpsMode::Afspc);

        for prop in sat["propagations"].as_array().unwrap() {
            if prop.get("error").is_some() {
                continue;
            }
            let tsince = prop["tsince"].as_f64().unwrap();
            let pi = imp.propagate(MinutesSinceEpoch(tsince));
            let pa = afspc.propagate(MinutesSinceEpoch(tsince));
            if let (Ok(i), Ok(a)) = (pi, pa) {
                if i.position[0].to_bits() != a.position[0].to_bits()
                    || i.position[1].to_bits() != a.position[1].to_bits()
                    || i.position[2].to_bits() != a.position[2].to_bits()
                {
                    any_diff = true;
                    break;
                }
            }
        }
        if any_diff {
            break;
        }
    }
    assert!(any_diff, "AFSPC and Improved produced identical output for ALL fixture rows - opsmode plumbing is broken");
}

#[test]
fn epoch_jd_accessor() {
    let sat = Satellite::from_tle(
        "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
        "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
    )
    .unwrap();
    let epoch = sat.epoch_jd();
    // TLE epoch field 184.80969102 → 2018 day-of-year 184.80969 = 2018-07-03
    // 19:25 UTC ≈ JD 2458303.31.
    let total = epoch.0 + epoch.1;
    assert!(
        (2458303.0..=2458304.0).contains(&total),
        "epoch JD {total} outside expected range"
    );
}
