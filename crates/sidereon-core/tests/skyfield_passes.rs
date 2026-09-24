#![cfg(sidereon_repo_tests)]
//! The pass and look-angle APIs against Skyfield 1.54.
//!
//! `fixtures/skyfield_passes/skyfield_passes.json` is Skyfield's own output,
//! written by `gen_skyfield_passes.py` there: three satellites (the 2018 ISS,
//! a Molniya and a low-inclination deep-space orbit on which SGP4's AFSPC and
//! improved modes differ by up to a kilometre within a day) from London and
//! Kampala, with the built-in timescale, which applies no polar motion.
//! Skyfield's `EarthSatellite` initialises SGP4 with python-sgp4's
//! `twoline2rv`, opsmode 'i', as these APIs do.
//!
//! Every comparison carries a bound derived from where the two computations
//! can legitimately differ, not a measured residual:
//!
//! * TEME state: SGP4 runs the same operations in both, with this crate's
//!   portable libm against the platform's; the fixture carries the
//!   per-component bound `fixtures-generators/sgp4_libm_bound.py` derives
//!   for that (see `sgp4_vallado_oracle.rs`).
//! * Frames: both evaluate the same IAU 2000A / 2006 rotations from the same
//!   time scales, in different operation orders and with different libms.
//!   `frames_and_station_match_skyfield` checks every entry of the
//!   TEME-to-GCRS and GCRS-to-ITRS rotations against Skyfield's to
//!   `MATRIX_TOL` = 2^-44, and the station's ITRS position to `MATRIX_TOL`
//!   of its magnitude; the look-angle and ground-track bounds carry those
//!   through, with the rounding of both chains (64 units of 2^-53 of the
//!   vectors' magnitudes) and of the east-north-up rotation, formed from
//!   four sines and cosines (8 units of 2^-53 per entry).
//! * Angles: an error `d` in the station-to-satellite vector moves the
//!   elevation by at most `d / range` and the azimuth by at most
//!   `d / (range cos(el))`, radians.
//! * Pass events: a crossing found to within `w` of the crossing of this
//!   crate's elevation lies within `w + delta / |rate|` of Skyfield's, for an
//!   elevation bound `delta` and Skyfield's elevation rate there. The
//!   culmination bounds (`culmination_bounds`) follow from each search's
//!   method and Skyfield's second and third derivatives of the elevation
//!   there, and the peak elevation within `delta + kappa t^2 / 2` for a time
//!   bound `t`.

use serde_json::Value;
use sidereon_core::astro::coverage::look_angles_batch;
use sidereon_core::astro::frames::transforms::{
    gcrs_to_itrs_matrix, geodetic_to_itrs, teme_to_gcrs_compute, TemeStateKm,
};
use sidereon_core::astro::passes::{
    find_passes, find_passes_batch_parallel, find_passes_batch_serial, find_passes_for_satellite,
    find_passes_with_opsmode, ground_track, look_angle, look_angle_arc, look_angle_batch_parallel,
    look_angle_batch_serial, predict_passes, predict_passes_with_opsmode, propagate_teme_arc,
    propagate_teme_batch_parallel, propagate_teme_batch_serial, visible_from_constellation,
    visible_from_satellites, ConstellationMember, GroundStation, LookAngle, PassFinderOptions,
    PassPredictionOptions, UtcInstant,
};
use sidereon_core::astro::sgp4::{ElementSet, OpsMode, Satellite};
use sidereon_core::astro::tle;

const FIXTURE: &str = include_str!("fixtures/skyfield_passes/skyfield_passes.json");

/// Unit roundoff of binary64.
const U: f64 = f64::EPSILON / 2.0;
/// Entry-wise agreement of each rotation with Skyfield's, 2^-44.
const MATRIX_TOL: f64 = f64::EPSILON * 256.0;
/// The smallest meridian radius of curvature of WGS84, a (1 - e^2), km: no
/// geodetic latitude above the ellipsoid moves faster than 1 / this per km.
const WGS84_MIN_MERIDIAN_RADIUS_KM: f64 = 6335.439;
/// `predict_passes` bisects each crossing 20 times within its coarse step.
const PREDICT_BISECTIONS: i32 = 20;
/// `predict_passes` narrows the culmination with 30 golden-section steps.
const PREDICT_GOLDEN_STEPS: i32 = 30;

fn fixture() -> Value {
    let fx: Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    assert_eq!(fx["skyfield"], "1.54");
    assert_eq!(fx["sgp4"], "2.22");
    fx
}

fn hexf(v: &Value) -> f64 {
    let s = v.as_str().expect("hex float");
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let rest = rest.strip_prefix("0x").expect("0x prefix");
    let (mant, exp) = rest.split_once('p').expect("exponent");
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let digits = format!("{int_part}{frac_part}");
    let mantissa = u64::from_str_radix(&digits, 16).expect("hex digits") as f64;
    let exp: i32 = exp.parse().expect("exponent digits");
    let value = mantissa * 2f64.powi(exp - 4 * frac_part.len() as i32);
    if neg {
        -value
    } else {
        value
    }
}

fn hex3(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("three values");
    [hexf(&a[0]), hexf(&a[1]), hexf(&a[2])]
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("numbers")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn deg(radians: f64) -> f64 {
    radians.to_degrees()
}

struct Case {
    satellite: Satellite,
    elements: ElementSet,
}

fn case(fx: &Value, name: &str) -> Case {
    let sat = &fx["satellites"][name];
    let (l1, l2) = (
        sat["line1"].as_str().unwrap(),
        sat["line2"].as_str().unwrap(),
    );
    Case {
        satellite: Satellite::from_tle(l1, l2).expect("TLE initialises"),
        elements: tle::parse(l1, l2)
            .expect("TLE parses")
            .elements
            .to_element_set()
            .expect("element set"),
    }
}

fn station(fx: &Value, name: &str) -> GroundStation {
    let s = &fx["stations"][name];
    GroundStation {
        latitude_deg: s["latitude_deg"].as_f64().unwrap(),
        longitude_deg: s["longitude_deg"].as_f64().unwrap(),
        altitude_m: s["altitude_m"].as_f64().unwrap(),
    }
}

fn station_itrs_km(ground: GroundStation) -> [f64; 3] {
    let (x, y, z) = geodetic_to_itrs(
        ground.latitude_deg,
        ground.longitude_deg,
        ground.altitude_m / 1000.0,
    )
    .unwrap();
    [x, y, z]
}

/// Bound on the difference between this crate's and Skyfield's
/// station-to-satellite vector, km, in the frame the angles are taken in.
fn line_of_sight_bound_km(
    teme_position_bound_km: f64,
    r_km: f64,
    station_km: f64,
    range_km: f64,
) -> f64 {
    teme_position_bound_km
        + 2.0 * 3.0 * MATRIX_TOL * r_km
        + 3.0 * MATRIX_TOL * station_km
        + 3.0 * 8.0 * U * range_km
        + 64.0 * U * (r_km + station_km)
}

struct AngleBounds {
    azimuth_deg: f64,
    elevation_deg: f64,
    range_km: f64,
}

fn angle_bounds(los_bound_km: f64, range_km: f64, elevation_deg: f64) -> AngleBounds {
    AngleBounds {
        azimuth_deg: deg(los_bound_km / (range_km * libm::cos(elevation_deg.to_radians())))
            + 16.0 * U * 360.0,
        elevation_deg: deg(los_bound_km / range_km) + 16.0 * U * 90.0,
        range_km: los_bound_km + 16.0 * U * range_km,
    }
}

fn azimuth_difference(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

fn group_rows<'a>(fx: &'a Value, sat: &str, station: &str) -> Vec<&'a Value> {
    fx["epochs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["satellite"] == sat && row["station"] == station)
        .collect()
}

fn groups(fx: &Value) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for row in fx["epochs"].as_array().unwrap() {
        let key = (
            row["satellite"].as_str().unwrap().to_string(),
            row["station"].as_str().unwrap().to_string(),
        );
        if !out.contains(&key) {
            out.push(key);
        }
    }
    out
}

fn instants(rows: &[&Value]) -> Vec<UtcInstant> {
    rows.iter()
        .map(|row| UtcInstant::from_unix_microseconds(row["unix_us"].as_i64().unwrap()))
        .collect()
}

#[test]
fn frames_and_station_match_skyfield() {
    let fx = fixture();
    let mut worst = 0.0_f64;
    for row in fx["epochs"].as_array().unwrap() {
        let ts = UtcInstant::from_unix_microseconds(row["unix_us"].as_i64().unwrap()).time_scales();

        let skyfield_itrs = row["gcrs_to_itrs"].as_array().unwrap();
        let ours_itrs = gcrs_to_itrs_matrix(&ts).unwrap();
        for (i, (ours_row, skyfield_row)) in ours_itrs.iter().zip(skyfield_itrs).enumerate() {
            let skyfield_row = skyfield_row.as_array().unwrap();
            for (j, (ours, skyfield)) in ours_row.iter().zip(skyfield_row).enumerate() {
                let d = (ours - hexf(skyfield)).abs();
                worst = worst.max(d);
                assert!(
                    d <= MATRIX_TOL,
                    "GCRS->ITRS [{i}][{j}] off by {d:e} at {row}"
                );
            }
        }

        // Column j of this crate's TEME->GCRS rotation is its image of the
        // unit vector e_j, exactly: the other two terms of each sum are zero.
        let skyfield_teme = &row["gcrs_to_teme"];
        for j in 0..3 {
            let unit: [f64; 3] = std::array::from_fn(|k| if k == j { 1.0 } else { 0.0 });
            let (column, _) = teme_to_gcrs_compute(
                &TemeStateKm {
                    position_km: unit,
                    velocity_km_s: [0.0; 3],
                },
                &ts,
                false,
            )
            .unwrap();
            let column = [column.0, column.1, column.2];
            for (i, value) in column.iter().enumerate() {
                // Skyfield's GCRS->TEME is the transpose of TEME->GCRS.
                let d = (value - hexf(&skyfield_teme[j][i])).abs();
                worst = worst.max(d);
                assert!(
                    d <= MATRIX_TOL,
                    "TEME->GCRS [{i}][{j}] off by {d:e} at {row}"
                );
            }
        }

        let ground = station(&fx, row["station"].as_str().unwrap());
        let ours = station_itrs_km(ground);
        let skyfield = hex3(&row["station_itrs_km"]);
        for axis in 0..3 {
            let d = (ours[axis] - skyfield[axis]).abs();
            assert!(
                d <= MATRIX_TOL * norm(skyfield),
                "station axis {axis} off by {d:e} km"
            );
        }
    }
    eprintln!("largest rotation entry difference from Skyfield: {worst:e}");
}

#[test]
fn teme_states_match_python_sgp4_under_skyfield() {
    let fx = fixture();
    let all: Vec<Satellite> = ["25544", "08195", "23599"]
        .iter()
        .map(|name| case(&fx, name).satellite)
        .collect();
    let mut states = 0;
    for (sat_name, station_name) in groups(&fx) {
        let rows = group_rows(&fx, &sat_name, &station_name);
        let grid = instants(&rows);
        let sat = case(&fx, &sat_name).satellite;
        let arc = propagate_teme_arc(&sat, &grid).unwrap();

        let index = ["25544", "08195", "23599"]
            .iter()
            .position(|n| *n == sat_name)
            .unwrap();
        for batch in [
            propagate_teme_batch_serial(&all, &grid),
            propagate_teme_batch_parallel(&all, &grid),
        ] {
            assert_eq!(batch[index].as_ref().unwrap(), &arc);
        }

        for (row, prediction) in rows.iter().zip(&arc) {
            let bound = f64s(&row["teme_bound"]);
            let want_r = hex3(&row["teme_position_km"]);
            let want_v = hex3(&row["teme_velocity_km_s"]);
            for axis in 0..3 {
                let dr = (prediction.position[axis] - want_r[axis]).abs();
                let dv = (prediction.velocity[axis] - want_v[axis]).abs();
                assert!(
                    dr <= bound[axis],
                    "{sat_name} {} position[{axis}] off by {dr:e}, bound {:e}",
                    row["unix_us"],
                    bound[axis]
                );
                assert!(
                    dv <= bound[3 + axis],
                    "{sat_name} {} velocity[{axis}] off by {dv:e}, bound {:e}",
                    row["unix_us"],
                    bound[3 + axis]
                );
            }
            states += 1;
        }
    }
    assert_eq!(states, 190);
}

fn check_look(what: &str, got: LookAngle, row: &Value, ground: GroundStation) -> [f64; 3] {
    let want_az = hexf(&row["azimuth_deg"]);
    let want_el = hexf(&row["elevation_deg"]);
    let want_range = hexf(&row["range_km"]);
    let teme_bound = f64s(&row["teme_bound"]);
    let pos_bound = norm([teme_bound[0], teme_bound[1], teme_bound[2]]);
    let r_km = norm(hex3(&row["teme_position_km"]));
    let los = line_of_sight_bound_km(pos_bound, r_km, norm(station_itrs_km(ground)), want_range);
    let bounds = angle_bounds(los, want_range, want_el);

    let d_az = azimuth_difference(got.azimuth_deg, want_az);
    let d_el = (got.elevation_deg - want_el).abs();
    let d_range = (got.range_km - want_range).abs();
    assert!(
        d_az <= bounds.azimuth_deg,
        "{what} azimuth off by {d_az:e}, bound {:e}",
        bounds.azimuth_deg
    );
    assert!(
        d_el <= bounds.elevation_deg,
        "{what} elevation off by {d_el:e}, bound {:e}",
        bounds.elevation_deg
    );
    assert!(
        d_range <= bounds.range_km,
        "{what} range off by {d_range:e}, bound {:e}",
        bounds.range_km
    );
    [
        d_az / bounds.azimuth_deg,
        d_el / bounds.elevation_deg,
        d_range / bounds.range_km,
    ]
}

#[test]
fn look_angles_match_skyfield() {
    let fx = fixture();
    let mut worst = [0.0_f64; 3];
    let mut checked = 0;
    for (sat_name, station_name) in groups(&fx) {
        let rows = group_rows(&fx, &sat_name, &station_name);
        let grid = instants(&rows);
        let c = case(&fx, &sat_name);
        let ground = station(&fx, &station_name);

        let arc = look_angle_arc(&c.satellite, ground, &grid).unwrap();
        let sats = std::slice::from_ref(&c.satellite);
        for batch in [
            look_angle_batch_serial(sats, ground, &grid),
            look_angle_batch_parallel(sats, ground, &grid),
        ] {
            assert_eq!(batch[0].as_ref().unwrap(), &arc);
        }

        for ((row, instant), from_arc) in rows.iter().zip(&grid).zip(&arc) {
            let single = look_angle(&c.elements, ground, *instant).unwrap();
            assert_eq!(&single, from_arc);
            let cell = look_angles_batch(sats, &[ground], *instant);
            assert_eq!(cell[0][0].as_ref().unwrap(), &single);

            let what = format!("{sat_name} from {station_name} at {}", row["unix_us"]);
            let fraction = check_look(&what, single, row, ground);
            for (w, f) in worst.iter_mut().zip(fraction) {
                *w = w.max(f);
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 190);
    eprintln!(
        "largest look-angle difference as a fraction of its bound: azimuth {:.3e}, elevation {:.3e}, range {:.3e}",
        worst[0], worst[1], worst[2]
    );
}

#[test]
fn ground_tracks_match_skyfield() {
    let fx = fixture();
    let mut checked = 0;
    for (sat_name, station_name) in groups(&fx) {
        let rows = group_rows(&fx, &sat_name, &station_name);
        let grid = instants(&rows);
        let c = case(&fx, &sat_name);
        let track = ground_track(&c.satellite, &grid).unwrap();
        for (row, point) in rows.iter().zip(&track) {
            let teme_bound = f64s(&row["teme_bound"]);
            let pos_bound = norm([teme_bound[0], teme_bound[1], teme_bound[2]]);
            let r_km = norm(hex3(&row["teme_position_km"]));
            let itrs_bound = pos_bound + 2.0 * 3.0 * MATRIX_TOL * r_km + 64.0 * U * r_km;
            let want_lat = hexf(&row["latitude_rad"]);
            let want_lon = hexf(&row["longitude_rad"]);
            let want_height = hexf(&row["height_m"]);
            let r_xy = r_km * libm::cos(want_lat);
            let lat_bound = itrs_bound / WGS84_MIN_MERIDIAN_RADIUS_KM + 16.0 * U * 2.0;
            let lon_bound = itrs_bound / r_xy + 16.0 * U * 4.0;
            let height_bound_m = (1.01 * itrs_bound + 16.0 * U * r_km) * 1000.0;

            let d_lat = (point.lat_rad - want_lat).abs();
            let d_lon = {
                let d = (point.lon_rad - want_lon).rem_euclid(std::f64::consts::TAU);
                d.min(std::f64::consts::TAU - d)
            };
            let d_height = (point.height_m - want_height).abs();
            let what = format!("{sat_name} at {}", row["unix_us"]);
            assert!(
                d_lat <= lat_bound,
                "{what} latitude off by {d_lat:e}, bound {lat_bound:e}"
            );
            assert!(
                d_lon <= lon_bound,
                "{what} longitude off by {d_lon:e}, bound {lon_bound:e}"
            );
            assert!(
                d_height <= height_bound_m,
                "{what} height off by {d_height:e} m, bound {height_bound_m:e}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 190);
}

#[test]
fn visibility_lists_match_skyfield() {
    let fx = fixture();
    let names = ["25544", "08195", "23599"];
    let cases: Vec<Case> = names.iter().map(|n| case(&fx, n)).collect();
    let members: Vec<ConstellationMember> = names
        .iter()
        .zip(&cases)
        .map(|(n, c)| ConstellationMember {
            catalog_number: n.to_string(),
            elements: c.elements.clone(),
        })
        .collect();
    let satellites: Vec<Satellite> = cases.iter().map(|c| c.satellite.clone()).collect();
    let ids: Vec<String> = names.iter().map(|n| n.to_string()).collect();

    for list in fx["visible"].as_array().unwrap() {
        let instant = UtcInstant::from_unix_microseconds(list["unix_us"].as_i64().unwrap());
        let station_name = list["station"].as_str().unwrap();
        let ground = station(&fx, station_name);
        let rows: Vec<&Value> = list["satellites"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row.get("error").is_none())
            .collect();

        for min_elevation in [-90.0, 0.0] {
            let mut expected: Vec<&&Value> = rows
                .iter()
                .filter(|row| hexf(&row["elevation_deg"]) >= min_elevation)
                .collect();
            expected.sort_by(|a, b| {
                hexf(&b["elevation_deg"])
                    .partial_cmp(&hexf(&a["elevation_deg"]))
                    .unwrap()
            });

            let from_members =
                visible_from_constellation(&members, ground, instant, min_elevation).unwrap();
            let from_satellites =
                visible_from_satellites(&satellites, &ids, ground, instant, min_elevation).unwrap();
            assert_eq!(from_members, from_satellites);
            assert_eq!(
                from_members
                    .iter()
                    .map(|v| v.catalog_number.as_str())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|row| row["satellite"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                "{station_name} at {}",
                list["unix_us"]
            );
            for (got, row) in from_members.iter().zip(&expected) {
                let what = format!(
                    "{} from {station_name} at {}",
                    got.catalog_number, list["unix_us"]
                );
                check_look(
                    &what,
                    LookAngle {
                        azimuth_deg: got.azimuth_deg,
                        elevation_deg: got.elevation_deg,
                        range_km: got.range_km,
                    },
                    row,
                    ground,
                );
                let bound = f64s(&row["teme_bound"]);
                let want = hex3(&row["teme_position_km"]);
                for axis in 0..3 {
                    let d = (got.position_km[axis] - want[axis]).abs();
                    assert!(d <= bound[axis], "{what} position[{axis}] off by {d:e}");
                }
            }
        }
    }
}

/// Elevation bound at a pass event, degrees.
fn event_elevation_bound_deg(event: &Value, ground: GroundStation) -> f64 {
    let range = event["range_km"].as_f64().unwrap();
    let station_km = norm(station_itrs_km(ground));
    // The satellite's geocentric distance is at most range + station distance.
    let r_km = range + station_km;
    let los = line_of_sight_bound_km(
        event["teme_position_bound_km"].as_f64().unwrap(),
        r_km,
        station_km,
        range,
    );
    deg(los / range) + 16.0 * U * 90.0
}

fn crossing_bound_s(event: &Value, ground: GroundStation, search_width_s: f64) -> f64 {
    let rate = event["elevation_rate_deg_s"].as_f64().unwrap().abs();
    search_width_s + event_elevation_bound_deg(event, ground) / rate + 2.0e-6
}

/// How a pass search places its culmination.
enum CulminationSearch {
    /// `find_passes`: bisection on the sign of the central-difference
    /// elevation rate with half-step `h` (1 s), to a bracket of `tolerance`,
    /// returning its midpoint.
    RateZero { tolerance_s: f64, half_step_s: f64 },
    /// `predict_passes`: golden-section search on the elevation, to a
    /// bracket of twice `half_width_s`, returning its midpoint. The search
    /// keeps one crest; on the one pass here that crests twice (08195 from
    /// Kampala, 36.7 and 21.9 degrees) it keeps the higher, which this test
    /// requires.
    Golden { half_width_s: f64 },
}

/// (time bound s, elevation bound deg) for a culmination.
///
/// Skyfield's elevation near its maximum is `el_max - kappa t^2 / 2 + c t^3`
/// with `kappa = |el''|` and `c = el''' / 6`. This crate's elevation is within
/// `delta` of it. A central difference with half-step `h` moves the zero of the
/// rate by at most `h^2 |el'''| / (6 kappa)`, and its `delta / h` error by
/// `delta / (h kappa)`; a comparison of two elevations that differ by less
/// than `2 delta` can keep either, which a golden-section search tolerates to
/// within `sqrt(4 delta / kappa)` of the maximum. `kappa` is taken at 0.9 and
/// `|el'''|` at 1.1 of the generator's finite differences.
fn culmination_bounds(
    event: &Value,
    ground: GroundStation,
    search: CulminationSearch,
) -> (f64, f64) {
    let kappa = 0.9
        * event["elevation_second_derivative_deg_s2"]
            .as_f64()
            .unwrap()
            .abs();
    let third = 1.1
        * event["elevation_third_derivative_deg_s3"]
            .as_f64()
            .unwrap()
            .abs();
    let delta = event_elevation_bound_deg(event, ground);
    let t = match search {
        CulminationSearch::RateZero {
            tolerance_s,
            half_step_s,
        } => {
            tolerance_s / 2.0
                + half_step_s * half_step_s * third / (6.0 * kappa)
                + delta / (half_step_s * kappa)
        }
        CulminationSearch::Golden { half_width_s } => half_width_s + (4.0 * delta / kappa).sqrt(),
    } + 2.0e-6;
    (t, delta + 0.5 * kappa * t * t)
}

fn seconds(instant: UtcInstant) -> f64 {
    instant.unix_microseconds() as f64 / 1e6
}

struct PassWindow<'a> {
    sat_name: String,
    station_name: String,
    ground: GroundStation,
    start: UtcInstant,
    end: UtcInstant,
    mask: f64,
    reference: Vec<&'a Value>,
}

fn pass_windows(fx: &Value) -> Vec<PassWindow<'_>> {
    fx["passes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let station_name = case["station"].as_str().unwrap().to_string();
            PassWindow {
                sat_name: case["satellite"].as_str().unwrap().to_string(),
                ground: station(fx, &station_name),
                station_name,
                start: UtcInstant::from_unix_microseconds(case["start_unix_us"].as_i64().unwrap()),
                end: UtcInstant::from_unix_microseconds(case["end_unix_us"].as_i64().unwrap()),
                mask: case["mask_deg"].as_f64().unwrap(),
                reference: case["passes"].as_array().unwrap().iter().collect(),
            }
        })
        .collect()
}

/// `find_events` refines each event to half a second (`half_second`, its
/// search epsilon) and reports a time within that of the event.
const FIND_EVENTS_EPSILON_S: f64 = 0.5;

/// Checks one event time against the refined Skyfield time within `bound`
/// seconds, and against the time `find_events` itself reported within its
/// epsilon plus that bound.
fn check_event_time(what: &str, kind: &str, got_s: f64, event: &Value, bound: f64) {
    let refined = event["unix_s"].as_f64().unwrap();
    let raw = event["skyfield_unix_s"].as_f64().unwrap();
    let d = (got_s - refined).abs();
    assert!(d <= bound, "{what} {kind} off by {d:e} s, bound {bound:e}");
    let d_raw = (got_s - raw).abs();
    assert!(
        d_raw <= FIND_EVENTS_EPSILON_S + bound,
        "{what} {kind} off find_events' time by {d_raw:e} s"
    );
}

/// Elevation bound at the window start of a clamped pass, degrees.
fn start_elevation_bound_deg(start: &Value, ground: GroundStation) -> f64 {
    event_elevation_bound_deg(start, ground)
}

#[test]
fn find_passes_match_skyfield() {
    let fx = fixture();
    let (mut checked, mut clamped) = (0, 0);
    let (mut normal_adapter_checked, mut partial_adapter_checked) = (false, false);
    for w in pass_windows(&fx) {
        let c = case(&fx, &w.sat_name);
        let window_seconds = w.end.diff_seconds(w.start);
        let adapter_step_seconds = if !normal_adapter_checked
            && w.sat_name == "25544"
            && w.station_name == "london"
            && w.mask == 0.0
            && window_seconds == 86_400
        {
            Some(30.0)
        } else if !partial_adapter_checked && window_seconds < 86_400 {
            Some(10.0)
        } else {
            None
        };
        for coarse_step_seconds in [30.0, 10.0] {
            let mut options = PassFinderOptions::default();
            options.elevation_mask_deg = w.mask;
            options.coarse_step_seconds = coarse_step_seconds;
            let tolerance = options.time_tolerance_seconds;

            let found = find_passes(&c.elements, w.ground, w.start, w.end, options).unwrap();
            if adapter_step_seconds == Some(coarse_step_seconds) {
                let one = std::slice::from_ref(&c.elements);
                assert_eq!(
                    find_passes_batch_serial(one, w.ground, w.start, w.end, options)[0],
                    Ok(found.clone())
                );
                assert_eq!(
                    find_passes_batch_parallel(one, w.ground, w.start, w.end, options)[0],
                    Ok(found.clone())
                );
                assert_eq!(
                    find_passes_for_satellite(&c.satellite, w.ground, w.start, w.end, options),
                    Ok(found.clone())
                );
                assert_eq!(
                    find_passes_with_opsmode(
                        &c.elements,
                        w.ground,
                        w.start,
                        w.end,
                        options,
                        OpsMode::Improved
                    ),
                    Ok(found.clone())
                );
                if window_seconds == 86_400 {
                    normal_adapter_checked = true;
                } else {
                    partial_adapter_checked = true;
                }
            }

            // A pass still up at the window end has no LOS here and no set in
            // Skyfield, so neither lists it.
            assert_eq!(
                found.len(),
                w.reference.len(),
                "{} from {} mask {}: {found:?}",
                w.sat_name,
                w.station_name,
                w.mask
            );
            for (got, want) in found.iter().zip(&w.reference) {
                let what = format!(
                    "{} from {} mask {} {}",
                    w.sat_name, w.station_name, w.mask, want["set"]["unix_s"]
                );
                if want.get("clamped").is_some() {
                    // Already above the mask at the window start: AOS is the start.
                    assert_eq!(got.aos, w.start, "{what}");
                    clamped += 1;
                } else {
                    let rise_bound = crossing_bound_s(&want["rise"], w.ground, tolerance / 2.0);
                    check_event_time(&what, "AOS", seconds(got.aos), &want["rise"], rise_bound);
                }
                let set_bound = crossing_bound_s(&want["set"], w.ground, tolerance / 2.0);
                check_event_time(&what, "LOS", seconds(got.los), &want["set"], set_bound);
                if want["culmination"].is_null() {
                    // The elevation only falls from the window start, so the
                    // culmination is the start itself.
                    let start = &want["start"];
                    assert_eq!(got.culmination, w.start, "{what}");
                    let d =
                        (got.max_elevation_deg - start["elevation_deg"].as_f64().unwrap()).abs();
                    let bound = start_elevation_bound_deg(start, w.ground);
                    assert!(d <= bound, "{what} peak off by {d:e} deg, bound {bound:e}");
                } else {
                    let (culm_bound, peak_bound) = culmination_bounds(
                        &want["culmination"],
                        w.ground,
                        CulminationSearch::RateZero {
                            tolerance_s: tolerance,
                            half_step_s: 1.0,
                        },
                    );
                    check_event_time(
                        &what,
                        "culmination",
                        seconds(got.culmination),
                        &want["culmination"],
                        culm_bound,
                    );
                    let d = (got.max_elevation_deg
                        - want["culmination"]["elevation_deg"].as_f64().unwrap())
                    .abs();
                    assert!(
                        d <= peak_bound,
                        "{what} peak off by {d:e} deg, bound {peak_bound:e}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    assert!(
        clamped > 0,
        "the partial windows hold passes cut at the start"
    );
    assert!(
        normal_adapter_checked,
        "normal-window adapters were not checked"
    );
    assert!(
        partial_adapter_checked,
        "partial-window adapters were not checked"
    );
}

#[test]
fn predict_passes_match_skyfield() {
    let fx = fixture();
    let (mut checked, mut clamped) = (0, 0);
    for w in pass_windows(&fx).into_iter().filter(|w| w.mask == 0.0) {
        let c = case(&fx, &w.sat_name);
        for min_elevation_deg in [0.0, 10.0] {
            let mut options = PassPredictionOptions::default();
            options.min_elevation_deg = min_elevation_deg;
            let step_s = options.step_seconds as f64;

            let predicted = predict_passes(&c.elements, w.ground, w.start, w.end, options).unwrap();
            assert_eq!(
                predict_passes_with_opsmode(
                    &c.elements,
                    w.ground,
                    w.start,
                    w.end,
                    options,
                    OpsMode::Improved
                ),
                Ok(predicted.clone())
            );
            let peak = |p: &Value| {
                if p["culmination"].is_null() {
                    p["start"]["elevation_deg"].as_f64().unwrap()
                } else {
                    p["culmination"]["elevation_deg"].as_f64().unwrap()
                }
            };
            let reference: Vec<&&Value> = w
                .reference
                .iter()
                .filter(|p| peak(p) >= min_elevation_deg)
                .collect();
            assert_eq!(
                predicted.len(),
                reference.len(),
                "{} from {} min {min_elevation_deg}: {predicted:?}",
                w.sat_name,
                w.station_name
            );
            // Bisection halves the coarse step 20 times and returns the
            // midpoint; each midpoint is floored to a microsecond.
            let crossing_width = step_s / 2f64.powi(PREDICT_BISECTIONS + 1) + 1.0e-6;
            for (got, want) in predicted.iter().zip(&reference) {
                let what = format!(
                    "{} from {} {}",
                    w.sat_name, w.station_name, want["set"]["unix_s"]
                );
                if want.get("clamped").is_some() {
                    assert_eq!(got.rise, w.start, "{what}");
                    clamped += 1;
                } else {
                    let rise_bound = crossing_bound_s(&want["rise"], w.ground, crossing_width);
                    check_event_time(&what, "rise", seconds(got.rise), &want["rise"], rise_bound);
                }
                let set_bound = crossing_bound_s(&want["set"], w.ground, crossing_width);
                check_event_time(&what, "set", seconds(got.set), &want["set"], set_bound);
                // Each golden-section step keeps 1 - 0.381966... (under 0.618034)
                // of the span plus under a microsecond of rounding, which sums
                // to under 3 microseconds; the result is the midpoint.
                let span_s = seconds(got.set) - seconds(got.rise);
                let golden_width =
                    (span_s * 0.618_034_f64.powi(PREDICT_GOLDEN_STEPS) + 3.0e-6) / 2.0;
                if want["culmination"].is_null() {
                    // The elevation only falls from the start: the search
                    // closes in on the start, within its final half width.
                    let start = &want["start"];
                    let rate = start["elevation_rate_deg_s"].as_f64().unwrap().abs();
                    let d_time = seconds(got.max_elevation_time) - seconds(w.start);
                    assert!(
                        (0.0..=golden_width + 1.0e-6).contains(&d_time),
                        "{what} culmination {d_time:e} s after the start"
                    );
                    let bound =
                        start_elevation_bound_deg(start, w.ground) + rate * (golden_width + 1.0e-6);
                    let d =
                        (got.max_elevation_deg - start["elevation_deg"].as_f64().unwrap()).abs();
                    assert!(d <= bound, "{what} peak off by {d:e} deg, bound {bound:e}");
                } else {
                    let (culm_bound, peak_bound) = culmination_bounds(
                        &want["culmination"],
                        w.ground,
                        CulminationSearch::Golden {
                            half_width_s: golden_width,
                        },
                    );
                    check_event_time(
                        &what,
                        "culmination",
                        seconds(got.max_elevation_time),
                        &want["culmination"],
                        culm_bound,
                    );
                    let d = (got.max_elevation_deg
                        - want["culmination"]["elevation_deg"].as_f64().unwrap())
                    .abs();
                    assert!(
                        d <= peak_bound,
                        "{what} peak off by {d:e} deg, bound {peak_bound:e}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    assert!(
        clamped > 0,
        "the partial windows hold passes cut at the start"
    );
}

/// On 23599 the AFSPC and improved modes put the satellite up to a kilometre
/// apart, so a pass search in AFSPC mode does not reproduce Skyfield's.
#[test]
fn the_afspc_mode_moves_the_low_inclination_deep_space_passes() {
    let fx = fixture();
    let w = pass_windows(&fx)
        .into_iter()
        .find(|w| w.sat_name == "23599" && w.station_name == "kampala" && w.mask == 0.0)
        .unwrap();
    let c = case(&fx, "23599");
    let options = PassFinderOptions::default();
    let improved = find_passes(&c.elements, w.ground, w.start, w.end, options).unwrap();
    let afspc = find_passes_with_opsmode(
        &c.elements,
        w.ground,
        w.start,
        w.end,
        options,
        OpsMode::Afspc,
    )
    .unwrap();
    assert_eq!(improved.len(), afspc.len());
    assert_ne!(improved, afspc);
}
