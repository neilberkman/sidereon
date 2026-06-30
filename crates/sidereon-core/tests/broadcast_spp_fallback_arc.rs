//! Reference-arc validation for the first-class broadcast SPP path and the
//! precise-with-broadcast fallback entry, on the committed 2020 DOY177 IGS data:
//! ESBC00DNK GPS observations, the ESBC mixed broadcast navigation, and the COD
//! MGEX final precise SP3, all at the first observation epoch (2020-06-25
//! 00:00:00 GPST).
//!
//! What it pins:
//!
//! 1. Broadcast-vs-precise agreement: a broadcast-only SPP fix and a precise SPP
//!    fix on the same GPS C1C pseudoranges agree to within a LABELED few-meter
//!    bound. This is the physical broadcast signal-in-space accuracy delta, not a
//!    bit-exact claim (the broadcast orbit/clock is a fit/extrapolation where the
//!    precise product is post-processed; the per-satellite error partly absorbs
//!    into the receiver clock, leaving a few-meter position difference). The
//!    underlying orbit/clock RMS is ~1-2 m, measured directly by the
//!    `broadcast_comparison` SISRE gate.
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
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::observables::j2000_seconds_from_split;
use sidereon_core::positioning::{
    solve, solve_broadcast, solve_with_fallback, BroadcastReason, Corrections, FixSource,
    KlobucharCoeffs, Observation, ReceiverSolution, SolveInputs, SurfaceMet,
};
use sidereon_core::rinex::observations::{
    observation_values, ObsEpochTime, ObservationFilter, RinexObs,
};
use sidereon_core::staleness::{DegradationKind, SelectionError, StalenessPolicy};
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
    let obs_text = load_text(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
    let obs = RinexObs::parse(&obs_text).expect("parse ESBC observation file");
    let epoch = obs.epochs().first().expect("at least one obs epoch");

    let split = civil_to_julian_split(epoch.epoch);
    let t_rx_j2000_s =
        j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split");
    let sod = f64::from(epoch.epoch.hour) * 3600.0
        + f64::from(epoch.epoch.minute) * 60.0
        + epoch.epoch.second;

    let filter = ObservationFilter::from_entries([(GnssSystem::Gps, vec!["C1C".to_string()])]);
    let values = observation_values(&obs, epoch, &filter).expect("observation values");
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
    assert!(
        observations.len() >= 5,
        "need a redundant GPS set, got {}",
        observations.len()
    );

    let approx = obs.header().approx_position_m.expect("APPROX POSITION XYZ");

    SolveInputs {
        observations,
        t_rx_j2000_s,
        t_rx_second_of_day_s: sod,
        day_of_year: 177.0 + sod / 86_400.0,
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
        glonass_channels: std::collections::BTreeMap::new(),
        met: SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
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
    assert_eq!(a.system_clocks_s.len(), b.system_clocks_s.len());
    for ((a_sys, a_clk), (b_sys, b_clk)) in a.system_clocks_s.iter().zip(b.system_clocks_s.iter()) {
        assert_eq!(a_sys, b_sys);
        assert_eq!(a_clk.to_bits(), b_clk.to_bits());
    }
    assert_eq!(a.dop, b.dop);
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

/// LABELED broadcast-vs-precise accuracy delta. The broadcast orbit error is
/// ~1-2 m RMS (3D), but at a single epoch the position difference between a
/// broadcast-only and a precise SPP fix on identical L1 C1C pseudoranges is
/// larger: the broadcast L1 satellite clock (polynomial fit, relativity, minus
/// TGD) differs from the precise SP3 ionosphere-free clock (no TGD applied) by a
/// per-satellite amount that does not fully absorb into the receiver clock, so
/// the geometry maps the orbit plus clock-scatter difference to the ~10 m level.
/// The observed delta on this machine is ~13 m; the 20 m bound is that documented
/// delta with margin, not a bit-exact claim (two orbit/clock sources legitimately
/// differ). The underlying ~1-2 m orbit and clock RMS is measured directly by the
/// `broadcast_comparison` SISRE gate.
const BROADCAST_VS_PRECISE_POSITION_BOUND_M: f64 = 20.0;

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
