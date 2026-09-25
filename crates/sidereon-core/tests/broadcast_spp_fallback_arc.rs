//! Reference-arc validation for the first-class broadcast SPP path and the
//! precise-with-broadcast fallback entry, on the committed 2020 DOY177 IGS data:
//! ESBC00DNK GPS observations, the ESBC mixed broadcast navigation, and the COD
//! MGEX final precise SP3, all at the first observation epoch (2020-06-25
//! 00:00:00 GPST).
//!
//! What it pins:
//!
//! 1. Broadcast-vs-precise agreement: a broadcast-only SPP fix and a precise SPP
//!    fix on the same GPS C1C pseudoranges agree to within a LABELED 8 m bound at
//!    each epoch of the arc (5.66 m and 5.73 m measured). This is the physical
//!    broadcast signal-in-space accuracy delta, not a bit-exact claim (the
//!    broadcast orbit/clock is a fit/extrapolation where the precise product is
//!    post-processed, and the single-frequency model subtracts the broadcast TGD
//!    where the precise ionosphere-free clock has none; the per-satellite error
//!    partly absorbs into the receiver clock, leaving a few-metre position
//!    difference). The underlying orbit RMS is ~1-2 m and the clock RMS 0.69 m,
//!    measured directly by the `broadcast_comparison` SISRE gate.
//! 2. Precise-present byte identity: with a precise product covering the epoch,
//!    `solve_with_fallback` is bit-for-bit identical to `solve` on that SP3 and
//!    reports `FixSource::Precise` (exact).
//! 3. Fallback to broadcast: with no precise product (or none covering the
//!    epoch), `solve_with_fallback` produces the broadcast fix bit-for-bit and
//!    reports `FixSource::Broadcast` carrying the precise selection's rejection
//!    reason, never a silent substitution.
#![cfg(sidereon_repo_tests)]

use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::astro::time::ExactEpoch;
use sidereon_core::constants::{SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE};
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::observables::j2000_seconds_from_split;
use sidereon_core::positioning::{
    solve, solve_broadcast, solve_with_fallback, BroadcastReason, Corrections, EphemerisSource,
    FixSource, KlobucharCoeffs, Observation, ReceiverSolution, SolveInputs, SurfaceMet,
};
use sidereon_core::rinex::observations::{
    observation_values, ObsEpoch, ObsEpochTime, ObservationFilter, RinexObs,
};
use sidereon_core::staleness::{select_sp3, DegradationKind, SelectionError, StalenessPolicy};
use sidereon_core::GnssSystem;
use std::path::PathBuf;

fn fixture_path(parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for part in parts {
        path.push(part);
    }
    path
}

fn load_text(parts: &[&str]) -> String {
    let path = fixture_path(parts);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"))
}

fn broadcast_store() -> BroadcastEphemeris {
    let nav = load_text(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
    BroadcastEphemeris::from_nav(&nav).expect("parse ESBC broadcast NAV")
}

fn precise_sp3() -> Sp3 {
    let bytes = std::fs::read(fixture_path(&[
        "sp3",
        "COD0MGXFIN_20201770000_01D_05M_ORB.SP3",
    ]))
    .expect("read COD precise SP3");
    Sp3::parse(&bytes).expect("parse COD precise SP3")
}

fn precise_sp3_with_accuracy() -> Sp3 {
    let bytes = std::fs::read(fixture_path(&[
        "sp3",
        "COD0MGXFIN_20201770000_01D_05M_ORB.SP3",
    ]))
    .expect("read COD precise SP3");
    let mut with_accuracy = Vec::with_capacity(bytes.len());
    let mut records_with_accuracy = 0;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        if line.starts_with(b"PG01") {
            let mut record = line.strip_suffix(b"\n").unwrap_or(line).to_vec();
            record.resize(record.len().max(73), b' ');
            record[61..63].copy_from_slice(b" 0");
            record[64..66].copy_from_slice(b" 0");
            record[67..69].copy_from_slice(b" 0");
            record[70..73].copy_from_slice(b"  0");
            with_accuracy.extend_from_slice(&record);
            if line.ends_with(b"\n") {
                with_accuracy.push(b'\n');
            }
            records_with_accuracy += 1;
        } else {
            with_accuracy.extend_from_slice(line);
        }
    }
    assert!(
        records_with_accuracy > 0,
        "fixture must contain GPS G01 records"
    );
    Sp3::parse(&with_accuracy).expect("parse COD precise SP3 with G01 accuracy")
}

/// An SP3 whose coverage (2026 DOY120) lies entirely after the 2020 query epoch,
/// so the staleness selection finds no product at or before the epoch.
fn wrong_epoch_sp3() -> Sp3 {
    let bytes = std::fs::read(fixture_path(&[
        "sp3",
        "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3",
    ]))
    .expect("read IGS 2026 SP3");
    Sp3::parse(&bytes).expect("parse IGS 2026 SP3")
}

fn civil_to_julian_split(epoch: ObsEpochTime) -> JulianDateSplit {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date")
}

/// Build the GPS-only first-epoch SPP inputs from the ESBC observation file.
/// Troposphere-only corrections with zero Klobuchar, matching the deterministic
/// GPS C1C configuration the SPP unit tests use on this arc.
fn first_epoch_inputs() -> SolveInputs {
    let obs = esbc_obs();
    let epoch = obs.epochs().first().expect("at least one obs epoch");
    let inputs = epoch_inputs(&obs, epoch);
    assert!(
        inputs.observations.len() >= 5,
        "need a redundant GPS set, got {}",
        inputs.observations.len()
    );
    inputs
}

fn esbc_obs() -> RinexObs {
    let obs_text = load_text(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
    RinexObs::parse(&obs_text).expect("parse ESBC observation file")
}

/// The GPS C1C SPP inputs of one ESBC epoch, as [`first_epoch_inputs`] builds them.
fn epoch_inputs(obs: &RinexObs, epoch: &ObsEpoch) -> SolveInputs {
    let time = epoch.epoch.expect("the first epoch carries a time");
    let split = civil_to_julian_split(time);
    let t_rx_j2000_s =
        j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split");
    let sod = f64::from(time.hour) * SECONDS_PER_HOUR
        + f64::from(time.minute) * SECONDS_PER_MINUTE
        + time.second;

    let filter = ObservationFilter::from_entries([(GnssSystem::Gps, vec!["C1C".to_string()])]);
    let values = observation_values(obs, epoch, &filter).expect("observation values");
    let mut observations: Vec<Observation> = Vec::new();
    for (sat, rows) in values {
        if sat.system != GnssSystem::Gps {
            continue;
        }
        if let Some(code_m) = rows.iter().find(|r| r.code == "C1C").and_then(|r| r.value) {
            observations.push(Observation {
                satellite_id: sat,
                pseudorange_m: code_m,
            });
        }
    }

    let approx = obs.header().approx_position_m.expect("APPROX POSITION XYZ");

    SolveInputs {
        observations,
        t_rx_j2000_s,
        t_rx_second_of_day_s: sod,
        day_of_year: 177.0 + sod / SECONDS_PER_DAY,
        initial_guess: [approx[0], approx[1], approx[2], 0.0],
        corrections: Corrections {
            ionosphere: false,
            troposphere: true,
        },
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met: SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
        pseudorange_code: sidereon_core::positioning::PseudorangeCode::SingleFrequency,
        qzss_clock: sidereon_core::positioning::QzssClock::Gps,
        troposphere_model: sidereon_core::positioning::TroposphereModel::Rtklib,
    }
}

fn position_delta_m(a: &ReceiverSolution, b: &ReceiverSolution) -> f64 {
    let pa = a.position.as_array();
    let pb = b.position.as_array();
    ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt()
}

/// Full byte-for-byte equality of two receiver solutions: every field, with the
/// float fields compared by bit pattern. The fallback's precise-present and
/// broadcast paths must reproduce the corresponding direct solve exactly, so this
/// checks the whole solution, not just the position.
fn assert_solution_bits_eq(a: &ReceiverSolution, b: &ReceiverSolution) {
    assert_eq!(a.position.x_m.to_bits(), b.position.x_m.to_bits());
    assert_eq!(a.position.y_m.to_bits(), b.position.y_m.to_bits());
    assert_eq!(a.position.z_m.to_bits(), b.position.z_m.to_bits());
    assert_eq!(a.geodetic, b.geodetic);
    assert_eq!(a.rx_clock_s.to_bits(), b.rx_clock_s.to_bits());
    assert_eq!(a.rx_clock_drift_s_s, b.rx_clock_drift_s_s);
    assert_eq!(a.system_clocks_s.len(), b.system_clocks_s.len());
    for ((a_sys, a_clk), (b_sys, b_clk)) in a.system_clocks_s.iter().zip(b.system_clocks_s.iter()) {
        assert_eq!(a_sys, b_sys);
        assert_eq!(a_clk.to_bits(), b_clk.to_bits());
    }
    assert_eq!(a.dop, b.dop);
    assert_eq!(
        a.position_covariance
            .ecef_m2
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        b.position_covariance
            .ecef_m2
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        a.position_covariance
            .enu_m2
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        b.position_covariance
            .enu_m2
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        a.residuals_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        b.residuals_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(a.used_sats, b.used_sats);
    assert_eq!(a.rejected_sats, b.rejected_sats);
    assert_eq!(a.metadata, b.metadata);
}

/// LABELED broadcast-vs-precise accuracy delta. On identical L1 C1C pseudoranges
/// the broadcast-only and precise SPP fixes differ by the broadcast signal-in-space
/// error mapped through the geometry: the broadcast orbit error (~1-2 m RMS 3D) and
/// the broadcast clock's scatter about the precise clock (0.69 m RMS, 0.64 m with
/// the common datum removed, measured by the `broadcast_comparison` SISRE gate), plus
/// the remaining difference between the sources, the broadcast TGD the model
/// subtracts for single-frequency code where the precise ionosphere-free clock has
/// none. Both clocks carry the relativistic term. The measured delta is 5.66 m and
/// 5.73 m at the arc's two epochs; the 8 m bound is 40% above the larger, not a
/// bit-exact claim (two orbit/clock sources legitimately differ).
const BROADCAST_VS_PRECISE_POSITION_BOUND_M: f64 = 8.0;

#[test]
fn broadcast_spp_agrees_with_precise_spp_within_labeled_bound() {
    let inputs = first_epoch_inputs();
    let store = broadcast_store();
    let sp3 = precise_sp3();

    let broadcast = solve_broadcast(&store, &inputs, true).expect("broadcast-only SPP");
    let precise = solve(&sp3, &inputs, true).expect("precise SPP");

    assert!(
        broadcast.metadata.converged,
        "broadcast solve must converge"
    );
    assert!(precise.metadata.converged, "precise solve must converge");

    let delta = position_delta_m(&broadcast, &precise);
    eprintln!("broadcast-vs-precise SPP position delta = {delta:.4} m");
    // Non-tautological lower bound: a degenerate/zeroed source would collapse the
    // two solutions onto each other; a real broadcast-vs-precise pair differs.
    assert!(
        delta > 0.01,
        "broadcast and precise SPP are implausibly identical ({delta} m)"
    );
    assert!(
        delta < BROADCAST_VS_PRECISE_POSITION_BOUND_M,
        "broadcast SPP disagrees with precise SPP by {delta:.4} m \
         (> {BROADCAST_VS_PRECISE_POSITION_BOUND_M} m)"
    );
}

/// The broadcast-vs-precise SPP position delta at every epoch of the arc with at least
/// five GPS observations (two epochs, 5.66 m and 5.73 m measured), under the same
/// labeled bound. Every such epoch must solve on both sources.
#[test]
fn broadcast_vs_precise_spp_delta_over_the_arc() {
    let obs = esbc_obs();
    let store = broadcast_store();
    let sp3 = precise_sp3();
    let mut worst_m = 0.0_f64;
    let mut count = 0usize;
    for epoch in obs.epochs() {
        if epoch.epoch.is_none() {
            continue;
        }
        let inputs = epoch_inputs(&obs, epoch);
        if inputs.observations.len() < 5 {
            continue;
        }
        let broadcast = solve_broadcast(&store, &inputs, true).expect("broadcast-only SPP");
        let precise = solve(&sp3, &inputs, true).expect("precise SPP");
        let delta = position_delta_m(&broadcast, &precise);
        worst_m = worst_m.max(delta);
        count += 1;
    }
    assert_eq!(
        count, 2,
        "the arc has two epochs with five or more GPS observations"
    );
    assert!(
        worst_m < BROADCAST_VS_PRECISE_POSITION_BOUND_M,
        "max broadcast-vs-precise delta {worst_m:.4} m"
    );
}

#[test]
fn fallback_uses_precise_byte_identically_when_it_covers_the_epoch() {
    let inputs = first_epoch_inputs();
    let store = broadcast_store();
    let sp3 = precise_sp3();

    let direct = solve(&sp3, &inputs, true).expect("precise SPP");
    let products = [sp3];
    let sourced = solve_with_fallback(&products, &store, &inputs, StalenessPolicy::days(3.0), true)
        .expect("fallback solve");

    match &sourced.source {
        FixSource::Precise(meta) => {
            assert_eq!(meta.kind, DegradationKind::Exact);
            assert_eq!(meta.staleness_s, 0.0);
        }
        other => panic!("expected precise-exact source, got {other:?}"),
    }
    assert!(sourced.source.is_precise_exact());
    // The precise-present path must change no output bit versus solving the SP3
    // directly: the fallback is purely additive.
    assert_solution_bits_eq(&sourced.solution, &direct);
}

#[test]
fn sp3_selection_preserves_scalar_and_exact_accuracy_variance() {
    let inputs = first_epoch_inputs();
    let products = [precise_sp3_with_accuracy()];
    let selection = select_sp3(&products, inputs.t_rx_j2000_s, StalenessPolicy::days(3.0))
        .expect("exact SP3 selection");
    let satellite = sidereon_core::GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS G01");
    let raw_accuracy = products[0]
        .record_accuracy_codes(satellite, 0)
        .expect("G01 raw record accuracy")
        .p
        .expect("G01 P-record accuracy");
    assert_eq!(raw_accuracy.axis_exponents, [Some(0); 3]);
    assert_eq!(raw_accuracy.clock_exponent, Some(0));
    let selection_epoch =
        ExactEpoch::from_binary_j2000_seconds(inputs.t_rx_j2000_s).expect("binary receive epoch");
    let state_epoch = selection_epoch
        .clone()
        .checked_sub_binary_seconds(0.071_234_567_890_123)
        .expect("binary transmit offset");

    let direct_scalar = products[0].ephemeris_variance_m2(
        satellite,
        state_epoch.j2000_seconds(),
        selection_epoch.j2000_seconds(),
    );
    let selected_scalar = selection.ephemeris_variance_m2(
        satellite,
        state_epoch.j2000_seconds(),
        selection_epoch.j2000_seconds(),
    );
    assert!(
        direct_scalar > 0.0,
        "fixture must exercise retained SP3 accuracy"
    );
    assert_eq!(selected_scalar.to_bits(), direct_scalar.to_bits());

    let direct_exact =
        products[0].ephemeris_variance_at_epoch_query(satellite, &state_epoch, &selection_epoch);
    let selected_exact =
        selection.ephemeris_variance_at_epoch_query(satellite, &state_epoch, &selection_epoch);
    assert!(
        direct_exact > 0.0,
        "fixture must exercise exact retained accuracy"
    );
    assert_eq!(selected_exact.to_bits(), direct_exact.to_bits());
}

#[test]
fn fallback_drops_to_broadcast_when_no_precise_product_is_supplied() {
    let inputs = first_epoch_inputs();
    let store = broadcast_store();

    let broadcast = solve_broadcast(&store, &inputs, true).expect("broadcast-only SPP");
    let sourced = solve_with_fallback(&[], &store, &inputs, StalenessPolicy::days(3.0), true)
        .expect("fallback solve");

    match &sourced.source {
        FixSource::Broadcast(BroadcastReason::PreciseUnavailable(rejection)) => {
            assert_eq!(*rejection, SelectionError::EmptyProductSet);
        }
        other => panic!("expected broadcast (precise-unavailable) source, got {other:?}"),
    }
    assert!(sourced.source.is_broadcast());
    assert_eq!(sourced.source.staleness(), None);
    // The broadcast fix is bit-for-bit the broadcast-only solve.
    assert_solution_bits_eq(&sourced.solution, &broadcast);
}

#[test]
fn fallback_drops_to_broadcast_when_precise_does_not_cover_the_epoch() {
    let inputs = first_epoch_inputs();
    let store = broadcast_store();

    let broadcast = solve_broadcast(&store, &inputs, true).expect("broadcast-only SPP");
    // The only precise product is from 2026; the 2020 epoch precedes it, so the
    // staleness layer has no product at or before the epoch and declines.
    let products = [wrong_epoch_sp3()];
    let sourced = solve_with_fallback(&products, &store, &inputs, StalenessPolicy::days(3.0), true)
        .expect("fallback solve");

    match &sourced.source {
        FixSource::Broadcast(BroadcastReason::PreciseUnavailable(rejection)) => {
            assert!(
                matches!(rejection, SelectionError::NoPriorProduct { .. }),
                "expected NoPriorProduct, got {rejection:?}"
            );
        }
        other => panic!("expected broadcast (precise-unavailable) source, got {other:?}"),
    }
    assert_solution_bits_eq(&sourced.solution, &broadcast);
}

/// A precise SP3 for the prior day (2020 DOY176) whose last epoch precedes the
/// DOY177 00:00 query by one 15-minute step. The staleness layer selects it as a
/// within-cap nearest-prior product, and SP3 interpolation still serves the epoch
/// one step past coverage, so the degraded precise product produces the fix.
fn prior_day_sp3() -> Sp3 {
    let bytes = std::fs::read(fixture_path(&["sp3", "GAP_G01_20201760000_15M.sp3"]))
        .expect("read prior-day SP3");
    Sp3::parse(&bytes).expect("parse prior-day SP3")
}

#[test]
fn fallback_uses_degraded_precise_when_a_stale_product_still_serves_the_epoch() {
    let inputs = first_epoch_inputs();
    let store = broadcast_store();

    // A within-cap nearest-prior precise product is selected (it precedes the
    // epoch by under the 3-day cap) and can still serve the epoch, so the degraded
    // precise product is used: the fallback does not over-eagerly drop to broadcast
    // when stale precise data is usable.
    let products = [prior_day_sp3()];
    let direct = solve(&products[0], &inputs, true).expect("degraded precise SPP");
    let sourced = solve_with_fallback(&products, &store, &inputs, StalenessPolicy::days(3.0), true)
        .expect("fallback solve");

    match &sourced.source {
        FixSource::Precise(meta) => {
            assert_eq!(meta.kind, DegradationKind::NearestPrior);
            assert!(meta.staleness_s > 0.0);
            assert!(meta.staleness_s < StalenessPolicy::days(3.0).max_staleness_s);
        }
        other => panic!("expected precise-degraded source, got {other:?}"),
    }
    assert!(sourced.source.is_precise());
    assert!(!sourced.source.is_precise_exact());
    // The degraded-precise path is bit-for-bit identical to solving the selected
    // SP3 directly.
    assert_solution_bits_eq(&sourced.solution, &direct);
}
