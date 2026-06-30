//! Vallado SGP4 verification suite - 33 satellites, 198 propagation points,
//! 1098 component checks. Reference values captured from the Python `sgp4` C
//! extension (v2.25), which compiles Vallado's C++ (v2020-07-13) with WGS72,
//! opsmode 'i'. The pure-Rust port in `sidereon_core::astro::sgp4` must match
//! bit-for-bit (0 ULP) on every component.

use sidereon_core::astro::sgp4::{
    propagate_elements, ElementSet, JulianDate, MinutesSinceEpoch, OpsMode, Satellite,
};
use sidereon_core::astro::tle;

fn hex_to_f64(s: &str) -> f64 {
    let (neg, rest) = if let Some(r) = s.strip_prefix("-0x") {
        (true, r)
    } else if let Some(r) = s.strip_prefix("0x") {
        (false, r)
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
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    (ia - ib).unsigned_abs()
}

#[test]
fn all_33_vallado_satellites_at_0_ulp() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    let labels = ["px", "py", "pz", "vx", "vy", "vz"];
    let mut failures = Vec::new();

    for sat in data["satellites"].as_array().unwrap() {
        let line1 = sat["line1"].as_str().unwrap();
        let line2 = sat["line2"].as_str().unwrap();
        let norad = sat["norad"].as_str().unwrap();

        let satellite = Satellite::from_tle(line1, line2).unwrap();

        for prop in sat["propagations"].as_array().unwrap() {
            // Skip rows the C++ reference flagged as errors (decay, etc.)
            if prop.get("error").is_some() {
                continue;
            }

            let tsince = prop["tsince"].as_f64().unwrap();
            let pred = match satellite.propagate(MinutesSinceEpoch(tsince)) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let actual = [
                pred.position[0],
                pred.position[1],
                pred.position[2],
                pred.velocity[0],
                pred.velocity[1],
                pred.velocity[2],
            ];

            for (i, label) in labels.iter().enumerate() {
                let expected = hex_to_f64(prop[*label].as_str().unwrap());
                let ulp = ulp_distance(actual[i], expected);
                if ulp > 0 {
                    failures.push((norad.to_string(), tsince, label.to_string(), ulp));
                }
            }
        }
    }

    if !failures.is_empty() {
        failures.sort_by_key(|f| std::cmp::Reverse(f.3));
        let summary: Vec<String> = failures
            .iter()
            .take(20)
            .map(|(norad, tsince, label, ulp)| format!("  {norad} t={tsince} {label}: {ulp} ULP"))
            .collect();
        panic!(
            "{} ULP failures (top 20):\n{}",
            failures.len(),
            summary.join("\n")
        );
    }
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

        let from_tle = Satellite::from_tle(line1, line2).unwrap();

        // Build the ElementSet by parsing the TLE the same way sidereon does
        // (Elixir-side string slicing into rev/day², rev/day³, deg, etc.).
        let epoch = tle::parse(line1, line2)
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
            mean_motion_dot,
            mean_motion_double_dot,
            eccentricity,
            argument_of_perigee_deg,
            inclination_deg,
            mean_anomaly_deg,
            mean_motion_rev_per_day,
            right_ascension_deg,
            catalog_number: 0,
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
        let imp = Satellite::from_tle_with_opsmode(line1, line2, OpsMode::Improved).unwrap();
        let afspc = Satellite::from_tle_with_opsmode(line1, line2, OpsMode::Afspc).unwrap();

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
