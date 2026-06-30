//! GLONASS-included single-point-positioning CROSS-IMPLEMENTATION agreement
//! against an RTKLIB-demo5 `rnx2rtkp` reference solution.
//!
//! This is a smoke / integration cross-check, NOT the validation of the FDMA
//! ionosphere scaling. It drives the SPP solver on a real multi-GNSS RINEX
//! observation with GLONASS pseudoranges and the ionosphere correction enabled
//! and asserts the solved ECEF position agrees with the committed RTKLIB-demo5
//! `.pos` to a meter-level bound. That bound is the honest cross-implementation
//! floor: two independent broadcast SPP implementations legitimately differ at
//! the ~meter level from their elevation weighting, GPS TGD handling, and
//! satellite-clock/relativity/troposphere internals. The position is also not
//! bit-portable across BLAS builds.
//!
//! It deliberately does NOT prove the per-satellite FDMA `(f_L1 / f_k)^2`
//! ionosphere scaling: that scaling's effect (~3% of the iono delay, ~10-16 cm)
//! is absorbed by the free per-system GLONASS receiver clock and is far below
//! this meter-level position floor. The scaling is validated DIRECTLY and
//! deterministically by the unit tests in `src/spp/tests.rs`
//! (`glonass_iono_is_exactly_fdma_scaled_gps_l1_delay` and
//! `glonass_iono_changes_monotonically_with_channel`). What this test DOES
//! confirm end-to-end: a GPS+GLONASS solve runs, uses the GLONASS satellites,
//! carries one receiver clock per system (the GLO-GPS inter-system offset), and
//! lands within meters of an independent solver on real data.
//!
//! Fixture provenance: `tests/fixtures/rtk/esbc_gps_glonass_spp_l1_demo5.pos`
//! records `rnx2rtkp RTKLIB EX 2.5.1` (rtklibexplorer/demo5, commit 968da9a) single-
//! point GPS+GLONASS L1 output. Inputs: obs
//! `test/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx` (RINEX 3.05 MIXED,
//! 2 epochs from 2020-06-25 00:00:00 GPST, carries GLONASS SLOT/FRQ #); GPS nav +
//! GPSA/GPSB Klobuchar `nav/ESBC00DNK_R_20201770000_01D_MN.rnx`; GLONASS nav
//! `nav/ESBC00DNK_R_20201770000_01D_RN.rnx`. The committed RN file is genuine RINEX
//! 3.05 with the five-physical-line layout (epoch + 4 orbit lines); its orbit-4 ΔτN
//! field is gfzrnx's "unavailable" sentinel `.999999999999e+09`. sidereon's
//! `parse_glonass` consumes only the epoch + first three orbit lines, never reading
//! ΔτN (correct for an L1-only single-freq user); locked by
//! `committed_rn_fixture_is_rinex_305_five_line_layout_parsed_correctly` in
//! `src/rinex_nav/tests.rs`. Reference-generation workaround (the committed file is
//! NEVER modified): RTKLIB EX 2.5.1 parses ΔτN from a 3.05 header and the sentinel
//! corrupts the GLONASS pseudorange, so the `.pos` was generated against a throwaway
//! copy with only its header version edited 3.05 -> 3.04 (`sed '1 s/^     3.05/
//! 3.04/'`), leaving ΔτN=0 as sidereon uses. Config `glo_spp.conf`:
//! `pos1-posmode=single`, `pos1-frequency=l1`, `pos1-navsys=5` (GPS+GLONASS),
//! `pos1-elmask=10`, `pos1-ionoopt=brdc`, `pos1-tropopt=off`, `pos1-sateph=brdc`,
//! `pos2-armode=off`, `out-solformat=xyz` (tropo off isolates the GLONASS
//! measurement model). Used sats epoch 1: 9 GPS (G05 G07 G09 G13 G15 G18 G27 G28
//! G30) + 8 GLONASS (R01 R02 R08 R09 R10 R11 R17 R18); RTKLIB ECEF
//! 3582110.6334/532590.1127/5232764.8971 m, sidereon delta ~1.41 m (the honest
//! cross-implementation floor). Repro at
//! `test/fixtures/rtk/generators/glo_spp_repro.sh`.
#![cfg(sidereon_repo_tests)]

use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::ephemeris::BroadcastEphemeris;
use sidereon_core::observables::j2000_seconds_from_split;
use sidereon_core::positioning::{
    solve, Corrections, KlobucharCoeffs, Observation, SolveInputs, SurfaceMet,
};
use sidereon_core::rinex::observations::{
    observation_values, ObsEpochTime, ObservationFilter, RinexObs,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::path::PathBuf;

/// RTKLIB-demo5 reference ECEF position for the first epoch (2020-06-25
/// 00:00:00 GPST), from `tests/fixtures/rtk/esbc_gps_glonass_spp_l1_demo5.pos`.
const RTKLIB_EPOCH: &str = "2111 345600.000";

/// The satellites RTKLIB-demo5 used in the first-epoch single-point solution
/// (9 GPS + 8 GLONASS, read from the `.pos.stat` `$SAT` records). The sidereon
/// solve is restricted to this set so the comparison isolates the measurement
/// model and the GLONASS ionosphere scaling from selection differences.
const RTKLIB_USED_SATS: &[&str] = &[
    "G05", "G07", "G09", "G13", "G15", "G18", "G27", "G28", "G30", "R01", "R02", "R08", "R09",
    "R10", "R11", "R17", "R18",
];

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

fn satellite_id(token: &str) -> GnssSatelliteId {
    let mut chars = token.chars();
    let system = GnssSystem::from_letter(chars.next().expect("system char"))
        .expect("known GNSS system code");
    let prn = chars.as_str().parse::<u8>().expect("PRN integer");
    GnssSatelliteId::new(system, prn).expect("valid satellite id")
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

fn j2000_seconds(epoch: ObsEpochTime) -> f64 {
    let split = civil_to_julian_split(epoch);
    j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split Julian date")
}

/// Build a single broadcast ephemeris source carrying both the GPS Keplerian
/// records (multi-GNSS `MN` nav) and the GLONASS state vectors (`RN` nav) by
/// concatenating the GLONASS record body onto the GPS file.
fn combined_broadcast_store() -> BroadcastEphemeris {
    let gps = load_text(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
    let glo = load_text(&["nav", "ESBC00DNK_R_20201770000_01D_RN.rnx"]);
    let glo_body = glo
        .split_once("END OF HEADER")
        .map(|(_, body)| body.trim_start_matches(['\r', '\n']))
        .expect("GLONASS nav END OF HEADER");
    let combined = format!("{gps}{glo_body}");
    BroadcastEphemeris::from_nav(&combined).expect("parse combined GPS+GLONASS nav")
}

/// Read the RTKLIB-demo5 reference ECEF position for the first epoch.
fn rtklib_reference_xyz() -> [f64; 3] {
    let pos = load_text(&["rtk", "esbc_gps_glonass_spp_l1_demo5.pos"]);
    let line = pos
        .lines()
        .find(|line| line.starts_with(RTKLIB_EPOCH))
        .unwrap_or_else(|| panic!("RTKLIB reference epoch {RTKLIB_EPOCH} not found"));
    let cols: Vec<&str> = line.split_whitespace().collect();
    [
        cols[2].parse().expect("x-ecef"),
        cols[3].parse().expect("y-ecef"),
        cols[4].parse().expect("z-ecef"),
    ]
}

#[test]
fn glonass_spp_agrees_with_rtklib_demo5_l1_single() {
    let obs_text = load_text(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
    let obs = RinexObs::parse(&obs_text).expect("parse ESBC observation file");
    let store = combined_broadcast_store();

    let epoch = obs.epochs().first().expect("at least one obs epoch");
    let t_rx_j2000_s = j2000_seconds(epoch.epoch);
    let sod = f64::from(epoch.epoch.hour) * 3600.0
        + f64::from(epoch.epoch.minute) * 60.0
        + epoch.epoch.second;
    let day_of_year = 177.0 + sod / 86_400.0;

    // Broadcast Klobuchar coefficients (GPSA/GPSB) from the nav header; the same
    // L1 set RTKLIB scales per carrier, GLONASS G1 included.
    let gps_klob = store
        .iono_corrections()
        .gps
        .expect("GPS Klobuchar coefficients in nav header");
    let klobuchar = KlobucharCoeffs {
        alpha: gps_klob.alpha,
        beta: gps_klob.beta,
    };

    // GLONASS FDMA channel numbers from the observation header.
    let glonass_channels = obs.header().glonass_slots.clone();

    // Real C1C pseudoranges for exactly the satellites RTKLIB used.
    let filter = ObservationFilter::from_entries([
        (GnssSystem::Gps, vec!["C1C".to_string()]),
        (GnssSystem::Glonass, vec!["C1C".to_string()]),
    ]);
    let values = observation_values(&obs, epoch, &filter).expect("observation values");
    let wanted: Vec<GnssSatelliteId> = RTKLIB_USED_SATS.iter().map(|s| satellite_id(s)).collect();

    let mut observations = Vec::new();
    for (sat, rows) in values {
        if !wanted.contains(&sat) {
            continue;
        }
        if let Some(code_m) = rows.iter().find(|r| r.code == "C1C").and_then(|r| r.value) {
            observations.push(Observation {
                satellite_id: sat,
                pseudorange_m: code_m,
            });
        }
    }
    assert_eq!(
        observations.len(),
        RTKLIB_USED_SATS.len(),
        "every RTKLIB-used satellite must have a C1C pseudorange"
    );
    let glonass_count = observations
        .iter()
        .filter(|o| o.satellite_id.system == GnssSystem::Glonass)
        .count();
    assert_eq!(
        glonass_count, 8,
        "the solve must include the 8 GLONASS sats"
    );

    let approx = obs.header().approx_position_m.expect("APPROX POSITION XYZ");

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s,
        t_rx_second_of_day_s: sod,
        day_of_year,
        initial_guess: [approx[0], approx[1], approx[2], 0.0],
        corrections: Corrections::IONO,
        klobuchar,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels,
        met: SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
    };

    let solution = solve(&store, &inputs, true).expect("GLONASS-included SPP solve");

    // The solve must actually use GLONASS, and carry one receiver clock per
    // system (GPS reference + the GLO-GPS inter-system offset).
    assert!(
        solution
            .used_sats
            .iter()
            .any(|s| s.system == GnssSystem::Glonass),
        "the solution must use GLONASS satellites"
    );
    assert_eq!(
        solution.system_clocks_s.len(),
        2,
        "GPS+GLONASS solve carries two per-system receiver clocks"
    );

    let reference = rtklib_reference_xyz();
    let p = solution.position;
    let dx = p.x_m - reference[0];
    let dy = p.y_m - reference[1];
    let dz = p.z_m - reference[2];
    let delta = (dx * dx + dy * dy + dz * dz).sqrt();
    eprintln!(
        "sidereon=({:.4},{:.4},{:.4}) rtklib=({:.4},{:.4},{:.4}) delta_3d={delta:.4} m",
        p.x_m, p.y_m, p.z_m, reference[0], reference[1], reference[2]
    );

    // Cross-implementation agreement bound (NOT a 0-ULP claim, and NOT a proof
    // of the FDMA scaling). Two independent broadcast SPP implementations differ
    // at the ~meter level on the same satellites and corrections from their
    // elevation weighting, GPS TGD, and satellite-clock/relativity internals;
    // the converged least-squares position is also not bit-portable across BLAS
    // builds. The observed agreement on this machine is ~1.41 m (corrections on)
    // and the 2.0 m bound is that floor with margin. The FDMA `(f_L1 / f_k)^2`
    // scaling -- whose ~10-16 cm effect is absorbed by the GLONASS clock here and
    // sits below this floor -- is validated directly by the deterministic unit
    // tests in `src/spp/tests.rs`, not by this position-level delta. See the
    // module-level fixture provenance note.
    assert!(
        delta < 2.0,
        "GLONASS SPP position disagrees with RTKLIB-demo5 by {delta:.4} m (> 2.0 m)"
    );
}
