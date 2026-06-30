//! Decimeter real-arc PPP-static truth test (the headline PPP signal).
//!
//! Builds and enables the FULL existing PPP correction stack -- relativistic
//! satellite clock, ZTD estimation, receiver AND satellite antenna PCO/PCV from
//! a real ANTEX (`igs20.atx`), solid-earth tide, phase wind-up, and elevation
//! weighting -- fed by an IGS final 30 s satellite clock, and asserts the
//! recovered STATIC coordinate lands within a decimeter of the published ITRF2020
//! truth. This is the correction-stack-on counterpart to the `disabled()`/5 m
//! ESBC float test in `ppp_real_arc.rs`, which is a code/phase model regression
//! signal, NOT a PPP accuracy signal.
//!
//! Fixture provenance (all open-access IGS / IGN holdings, fetched 2026-06-27):
//!   * Station: ZIM2 (Zimmerwald, CH; DOMES 14001M008; receiver TRM59800.00 NONE).
//!   * Day/window: 2026-05-13 (GPS week 2418, DOY 133), first 1 h, 30 s, GPS L1/L2
//!     ionosphere-free (C1C/C2W/L1C/L2W).
//!   * Obs: igs.bkg.bund.de/root_ftp/IGS/obs/2026/133/ZIM200CHE_R_20261330000_01D_30S_MO.crx.gz
//!     (Hatanaka; decompressed, trimmed to the first 120 epochs).
//!   * Orbit: BKG IGS/products/2418/IGS0OPSFIN_20261330000_01D_15M_ORB.SP3.gz
//!     (IGS final 15 min; trimmed to the first 13 epochs + EOF).
//!   * Clock: BKG IGS/products/2418/IGS0OPSFIN_20261330000_01D_30S_CLK.CLK.gz
//!     (IGS final 30 s; trimmed to AS records through 01:30:00).
//!   * Antenna: files.igs.org/pub/station/general/igs20.atx (IGS20; trimmed to the
//!     ZIM2 receiver block + the observed-GPS satellite blocks valid at the epoch).
//!   * Truth: itrf.ign.fr/.../ITRF2020-IGS-TRF.SSC, ZIM2 soln 3 (ref epoch 2015.0)
//!     propagated to 2026-05-13; see `ZIM2_TRUTH_ECEF_M`.
//!
//! Cross-validation: RTKLIB 2.4.3 `rnx2rtkp -p 8` (ppp-static, est-ztd, precise
//! eph, solid tide + rx/sat antenna + wind-up on, 7 deg mask, ANTEX igs20.atx) on
//! the SAME obs+SP3+CLK+ANTEX lands 0.088 m from this truth (final converged
//! epoch); see `RTKLIB_PPP_STATIC_ECEF_M`. sidereon's full-stack PPP is asserted
//! within a decimeter of both truth and RTKLIB (achieved ~0.04 m / ~0.06 m).

#![cfg(sidereon_repo_tests)]

use sidereon_core::antex::{Antex, AntexDateTime, PcvGrid};
use sidereon_core::astro::time::civil::civil_from_split_julian_date;
use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::atmosphere::troposphere::Met;
use sidereon_core::combinations::{ionosphere_free, ionosphere_free_phase_cycles};
use sidereon_core::constants::{F_L1_HZ, F_L2_HZ};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::frame::{itrf_to_geodetic, ItrfPositionM};
use sidereon_core::observables::{j2000_seconds_from_split, predict, PredictOptions};
use sidereon_core::ppp_corrections::{
    self, CivilDateTime, PppCorrectionEpoch, PppCorrectionObservation, PppCorrectionsOptions,
    SatelliteAntenna, SatelliteAntennaFrequency, SatelliteAntennaOptions,
};
use sidereon_core::precise_positioning::{
    solve_float_epochs, FloatEpoch, FloatObservation, FloatSolveConfig, FloatSolveOptions,
    FloatState, MeasurementWeights, PcvSample, PppCorrectionLookup, RangeCorrections,
    ReceiverAntennaFrequency, ReceiverAntennaOptions, SatelliteClockCorrections, TropoMapping,
    TroposphereOptions, VmfSiteSample, VmfSiteSeries,
};
use sidereon_core::rinex::clock::RinexClock;
use sidereon_core::rinex::observations::{
    observation_values, ObsEpoch, ObsEpochTime, ObservationFilter, RinexObs,
};
use sidereon_core::tides::OceanLoadingBlq;
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

// Published ITRF2020 (IGS realization) truth for ZIM2 on 2026-05-13, soln 3
// position+velocity at epoch 2015.0 propagated to 2026-05-13 (dt = 11.36164 y):
//   X0=4331299.74060566 VX=-0.0137774441653608
//   Y0= 567537.501384189 VY= 0.0181001831307067
//   Z0=4633133.83227038 VZ= 0.0116400603135022
const ZIM2_TRUTH_ECEF_M: [f64; 3] = [
    4_331_299.584_071_246,
    567_537.707_032_023_1,
    4_633_133.964_520_6,
];

// ZIM2 ocean-loading BLQ coefficients (ocean tide model GOT4.7, long-period
// tides from FES99), computed by OLFG/OLMPP of H.-G. Scherneck, Onsala Space
// Observatory (holt.oso.chalmers.se ocean tide loading provider), 2020-Jun-25;
// published BLQ block for ZIM2 (lon/lat 7.4650 46.8771, 956.425 m). BLQ column
// order M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa; rows amplitude radial/EW/NS (m) then
// phase radial/EW/NS (deg). Real provider values, not fabricated.
const ZIM2_OCEAN_LOADING_BLQ: OceanLoadingBlq = OceanLoadingBlq {
    amplitude_m: [
        [
            0.00693, 0.00228, 0.00148, 0.00061, 0.00220, 0.00094, 0.00070, 0.00001, 0.00047,
            0.00025, 0.00019,
        ],
        [
            0.00272, 0.00076, 0.00061, 0.00020, 0.00036, 0.00025, 0.00011, 0.00005, 0.00004,
            0.00001, 0.00002,
        ],
        [
            0.00061, 0.00026, 0.00010, 0.00009, 0.00025, 0.00002, 0.00008, 0.00003, 0.00002,
            0.00000, 0.00001,
        ],
    ],
    phase_deg: [
        [
            -72.3, -44.2, -90.8, -44.1, -62.9, -94.5, -64.3, 171.0, 3.4, 3.6, 1.1,
        ],
        [
            84.3, 115.4, 63.3, 113.7, 98.6, 20.7, 94.2, -44.5, -170.0, -162.7, -177.8,
        ],
        [
            -29.3, 1.7, -44.0, -4.2, 44.2, -39.1, 43.7, 170.1, -93.3, -118.3, -176.4,
        ],
    ],
};

// RTKLIB 2.4.3 `rnx2rtkp -p 8` (ppp-static, forward) final converged epoch on the
// identical obs+SP3+CLK+ANTEX inputs used by this test. Used only as a documented
// cross-validation reference; sidereon must agree with it to within a decimeter.
const RTKLIB_PPP_STATIC_ECEF_M: [f64; 3] = [4_331_299.543_7, 567_537.656_0, 4_633_133.905_0];

// Elevation cutoff (deg). Low-elevation satellites carry large unmodeled errors
// (the cm-level corrections we deliberately omit here -- ocean loading, higher-
// order mapping -- plus multipath); without a mask they corrupt the float
// solution. 7 deg matches the RTKLIB cross-validation run.
const ELEVATION_MASK_DEG: f64 = 7.0;

// Decimeter gate for a ~1 h GPS-only L1/L2 IF float arc. The achieved error is
// ~0.04 m vs truth and ~0.06 m vs RTKLIB; the 0.10 m gate enforces a genuine
// decimeter (it would fail above 10 cm) while leaving headroom for platform-libm
// drift in the transcendental correction models.
const DECIMETER_TRUTH_BOUND_M: f64 = 0.10;
// sidereon-vs-RTKLIB cross-validation gate (two PPP engines on the same data).
const SIDEREON_VS_RTKLIB_BOUND_M: f64 = 0.10;

fn fixture_path(parts: &[&str]) -> PathBuf {
    parts.iter().fold(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        |path, part| path.join(part),
    )
}

fn load_text(parts: &[&str]) -> String {
    let path = fixture_path(parts);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"))
}

fn load_sp3() -> Sp3 {
    let path = fixture_path(&["sp3", "IGS0OPSFIN_20261330000_03H_15M_ORB.SP3"]);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"));
    Sp3::parse(&bytes).unwrap_or_else(|err| panic!("parse SP3 {path:?}: {err}"))
}

fn load_obs() -> RinexObs {
    RinexObs::parse(&load_text(&[
        "obs",
        "ZIM200CHE_R_20261330000_01H_30S_MO_120epoch.rnx",
    ]))
    .expect("parse ZIM2 observation fixture")
}

fn load_clock() -> RinexClock {
    RinexClock::parse(&load_text(&[
        "clk",
        "IGS0OPSFIN_20261330000_90M_30S_CLK.CLK",
    ]))
    .expect("parse IGS final 30 s clock fixture")
}

fn load_antex() -> Antex {
    Antex::parse(&load_text(&["antex", "igs20_zim2_gps_trim.atx"])).expect("parse ANTEX fixture")
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

fn civil_datetime(epoch: ObsEpochTime) -> CivilDateTime {
    CivilDateTime {
        year: epoch.year,
        month: epoch.month,
        day: epoch.day,
        hour: epoch.hour,
        minute: epoch.minute,
        second: epoch.second,
    }
}

fn gps_l1_l2_filter() -> ObservationFilter {
    ObservationFilter::from_entries([(
        GnssSystem::Gps,
        vec![
            "C1C".to_string(),
            "C2W".to_string(),
            "L1C".to_string(),
            "L2W".to_string(),
        ],
    )])
}

fn float_observations(epoch: &ObsEpoch, obs: &RinexObs) -> Vec<FloatObservation> {
    let mut out = observation_values(obs, epoch, &gps_l1_l2_filter())
        .expect("valid observation values")
        .into_iter()
        .filter_map(|(sat, rows)| {
            let token = sat.to_string();
            let mut values = BTreeMap::new();
            for row in rows {
                values.insert(row.code, row.value);
            }
            let code_m = ionosphere_free(
                values.get("C1C").and_then(|v| *v)?,
                values.get("C2W").and_then(|v| *v)?,
                F_L1_HZ,
                F_L2_HZ,
            )
            .expect("ionosphere-free code");
            let phase_m = ionosphere_free_phase_cycles(
                values.get("L1C").and_then(|v| *v)?,
                values.get("L2W").and_then(|v| *v)?,
                F_L1_HZ,
                F_L2_HZ,
            )
            .expect("ionosphere-free carrier phase");
            Some(FloatObservation {
                sat,
                satellite_id: token.clone(),
                ambiguity_id: token,
                code_m,
                phase_m,
                freq1_hz: F_L1_HZ,
                freq2_hz: F_L2_HZ,
            })
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));
    out
}

fn float_epoch(epoch: ObsEpochTime, observations: Vec<FloatObservation>) -> FloatEpoch {
    let split = civil_to_julian_split(epoch);
    FloatEpoch {
        epoch: civil_datetime(epoch),
        jd_whole: split.jd_whole,
        jd_fraction: split.fraction,
        t_rx_j2000_s: j2000_seconds_from_split(split.jd_whole, split.fraction)
            .expect("valid split Julian date"),
        observations,
    }
}

/// Build the GPS L1/L2 ionosphere-free float epochs, applying an elevation mask
/// at the operationally available approximate position (satellite elevation from
/// the SP3 orbit). Standard PPP preprocessing; mirrors RTKLIB's mask angle.
fn gps_float_epochs(sp3: &Sp3, obs: &RinexObs, approx: [f64; 3]) -> Vec<FloatEpoch> {
    obs.epochs()
        .iter()
        .map(|epoch| {
            let split = civil_to_julian_split(epoch.epoch);
            let t_rx = j2000_seconds_from_split(split.jd_whole, split.fraction)
                .expect("valid split Julian date");
            let observations = float_observations(epoch, obs)
                .into_iter()
                .filter(|obs| elevation_deg(sp3, obs.sat, approx, t_rx) >= ELEVATION_MASK_DEG)
                .collect::<Vec<_>>();
            assert!(
                observations.len() >= 6,
                "fixture epoch {:?} has only {} complete masked GPS L1/L2 rows",
                epoch.epoch,
                observations.len()
            );
            float_epoch(epoch.epoch, observations)
        })
        .collect()
}

fn elevation_deg(sp3: &Sp3, sat: GnssSatelliteId, approx: [f64; 3], t_rx_j2000_s: f64) -> f64 {
    predict(
        sp3,
        sat,
        approx,
        t_rx_j2000_s,
        PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        },
    )
    .map(|p| p.elevation_deg)
    .unwrap_or(f64::NEG_INFINITY)
}

fn initial_state(epochs: &[FloatEpoch], start: [f64; 3]) -> FloatState {
    FloatState {
        position_m: start,
        clocks_m: vec![0.0; epochs.len()],
        ambiguities_m: initial_ambiguities(epochs),
        ztd_m: 0.0,
    }
}

fn initial_ambiguities(epochs: &[FloatEpoch]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for obs in epochs.iter().flat_map(|epoch| &epoch.observations) {
        out.entry(obs.ambiguity_id.clone())
            .or_insert(obs.phase_m - obs.code_m);
    }
    out
}

fn position_error_m(position_m: [f64; 3], truth_m: [f64; 3]) -> f64 {
    ((position_m[0] - truth_m[0]).powi(2)
        + (position_m[1] - truth_m[1]).powi(2)
        + (position_m[2] - truth_m[2]).powi(2))
    .sqrt()
}

fn antex_epoch() -> AntexDateTime {
    AntexDateTime::new(2026, 5, 13, 0, 30, 0).expect("valid ANTEX epoch")
}

fn observed_prns(epochs: &[FloatEpoch]) -> Vec<String> {
    epochs
        .iter()
        .flat_map(|e| e.observations.iter().map(|o| o.satellite_id.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Receiver antenna correction options from the ZIM2 ANTEX block. Maps the
/// ANTEX `G01`/`G02` PCO/PCV grids to the PPP receiver-antenna option struct.
fn receiver_antenna_options(antex: &Antex) -> ReceiverAntennaOptions {
    let antenna = antex
        .antenna_at("TRM59800.00     NONE", antex_epoch())
        .or_else(|| antex.antenna("TRM59800.00     NONE"))
        .expect("ZIM2 receiver antenna block present in ANTEX trim");
    let frequencies = ["G01", "G02"]
        .into_iter()
        .map(|label| {
            let freq = antenna
                .frequencies
                .get(label)
                .unwrap_or_else(|| panic!("ANTEX receiver frequency {label}"));
            ReceiverAntennaFrequency {
                label: label.to_string(),
                pco_m: freq.pco_m,
                pcv_samples: freq
                    .pcv_samples
                    .iter()
                    .map(|s| PcvSample {
                        azimuth_deg: s.azimuth_deg,
                        zenith_deg: s.zenith_deg,
                        value_m: s.value_m,
                    })
                    .collect(),
            }
        })
        .collect();
    ReceiverAntennaOptions {
        freq1_label: "G01".to_string(),
        freq1_hz: F_L1_HZ,
        freq2_label: "G02".to_string(),
        freq2_hz: F_L2_HZ,
        frequencies,
    }
}

/// Satellite antenna correction options from the ANTEX satellite blocks for the
/// observed GPS PRNs. Satellites carry NOAZI PCV only.
fn satellite_antenna_options(antex: &Antex, prns: &[String]) -> SatelliteAntennaOptions {
    let antennas =
        prns.iter()
            .map(|prn| {
                let antenna = antex
                    .satellite_antenna(prn, antex_epoch())
                    .unwrap_or_else(|| panic!("ANTEX satellite block for {prn}"));
                let frequencies = ["G01", "G02"]
                    .into_iter()
                    .map(|label| {
                        let freq = antenna.frequencies.get(label).unwrap_or_else(|| {
                            panic!("ANTEX satellite frequency {label} for {prn}")
                        });
                        SatelliteAntennaFrequency {
                            label: label.to_string(),
                            pco_m: freq.pco_m,
                            noazi_pcv_m: freq
                                .pcv_samples
                                .iter()
                                .filter(|s| s.grid == PcvGrid::NoAzimuth)
                                .map(|s| (s.zenith_deg, s.value_m))
                                .collect(),
                        }
                    })
                    .collect();
                SatelliteAntenna {
                    sat: gps_id(prn),
                    valid_from: None,
                    valid_until: None,
                    frequencies,
                }
            })
            .collect();
    SatelliteAntennaOptions {
        freq1_label: "G01".to_string(),
        freq1_hz: F_L1_HZ,
        freq2_label: "G02".to_string(),
        freq2_hz: F_L2_HZ,
        antennas,
    }
}

fn gps_id(token: &str) -> GnssSatelliteId {
    let prn = token
        .strip_prefix('G')
        .unwrap_or_else(|| panic!("expected GPS token, got {token:?}"))
        .parse::<u8>()
        .unwrap_or_else(|_| panic!("bad GPS token {token:?}"));
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
}

fn satellite_clock_corrections(clock: &RinexClock) -> SatelliteClockCorrections {
    let mut series = BTreeMap::new();
    for (token, points) in clock.series_rows() {
        let token: String = token;
        if let Some(stripped) = token.strip_prefix('G') {
            if let Ok(prn) = stripped.parse::<u8>() {
                if let Ok(sat) = GnssSatelliteId::new(GnssSystem::Gps, prn) {
                    series.insert(sat, points);
                }
            }
        }
    }
    SatelliteClockCorrections { series }
}

fn ppp_correction_epochs(epochs: &[FloatEpoch]) -> Vec<PppCorrectionEpoch> {
    epochs
        .iter()
        .map(|epoch| PppCorrectionEpoch {
            // The PPP correction stack consumes this civil field as UTC (Sun/Moon
            // and solid-earth tide need Earth orientation), but RINEX observation
            // epochs are labelled in GPST. Convert GPST -> UTC so the tide / Sun-
            // Moon / wind-up geometry is anchored to the right Earth rotation.
            // `t_rx_j2000_s` stays GPST (it drives the SP3 ephemeris lookup).
            epoch: gpst_civil_to_utc(epoch.jd_whole, epoch.jd_fraction),
            t_rx_j2000_s: epoch.t_rx_j2000_s,
            observations: epoch
                .observations
                .iter()
                .map(|o| PppCorrectionObservation {
                    sat: o.sat,
                    freq1_hz: F_L1_HZ,
                    freq2_hz: F_L2_HZ,
                })
                .collect(),
        })
        .collect()
}

/// GPS-UTC leap-second offset for the fixture epoch (2026). No leap second is
/// scheduled across the 1 h arc, so this is constant over every epoch here.
const GPS_MINUS_UTC_S: f64 = 18.0;

/// Convert a GPST Julian-date split (`jd_whole` ending in .5, `jd_fraction` the
/// day fraction) to a UTC `CivilDateTime` by removing the GPS-UTC leap offset and
/// expanding the resulting JD back to the Gregorian calendar via the canonical
/// `civil_from_split_julian_date`. The leap-shifted JD is re-split onto the
/// `*.5` civil-midnight boundary (`z - 0.5`, `jd - z`) the helper expects.
fn gpst_civil_to_utc(jd_whole: f64, jd_fraction: f64) -> CivilDateTime {
    let jd = jd_whole + jd_fraction - GPS_MINUS_UTC_S / 86_400.0 + 0.5;
    let z = jd.floor();
    let (year, month, day, hour, minute, second) = civil_from_split_julian_date(z - 0.5, jd - z);
    CivilDateTime {
        year: year as i32,
        month: month as u8,
        day: day as u8,
        hour: hour as u8,
        minute: minute as u8,
        second,
    }
}

fn full_corrections(
    sp3: &Sp3,
    antex: &Antex,
    clock: &RinexClock,
    epochs: &[FloatEpoch],
    receiver_ecef_m: [f64; 3],
) -> RangeCorrections {
    let prns = observed_prns(epochs);
    let options = PppCorrectionsOptions {
        solid_earth_tide: true,
        // Pole tide intentionally off: this arc validates the cm/dm-dominant
        // stack; pole tide is a sub-cm refinement out of scope here.
        pole_tide: None,
        // Ocean tide loading ON with the real ZIM2 BLQ. ZIM2 is deep inland, so
        // OTL is only a few mm here; the bar is no decimeter regression (a slight
        // change vs OTL-off is expected and benign).
        ocean_loading: Some(ZIM2_OCEAN_LOADING_BLQ),
        phase_windup: true,
        satellite_antenna: Some(satellite_antenna_options(antex, &prns)),
    };
    let precomputed = ppp_corrections::build(
        sp3,
        &ppp_correction_epochs(epochs),
        receiver_ecef_m,
        &options,
    )
    .expect("build full PPP correction tables");
    RangeCorrections {
        receiver_antenna: Some(receiver_antenna_options(antex)),
        sat_clock_relativity: true,
        satellite_clock: Some(satellite_clock_corrections(clock)),
        ppp: PppCorrectionLookup::from_options(precomputed, &options),
    }
}

/// A priori troposphere meteorology from the station's standard atmosphere at
/// its own height. ZIM2 sits at ~956 m, so a sea-level (1013.25 hPa) a priori
/// over-models the zenith hydrostatic delay by ~0.25 m; since the hydrostatic
/// and wet mapping functions differ, the estimated wet ZTD cannot fully absorb
/// the bias and it leaks into the recovered height. Anchoring the a priori at
/// the station height removes that vertical bias, matching RTKLIB's a priori.
fn station_met(ecef_m: [f64; 3]) -> Met {
    let geodetic = itrf_to_geodetic(
        ItrfPositionM::new(ecef_m[0], ecef_m[1], ecef_m[2]).expect("finite station ECEF"),
    )
    .expect("station geodetic");
    Met::standard(geodetic.height_m, 0.5).expect("valid station standard-atmosphere met")
}

fn full_stack_config(corrections: RangeCorrections, met: Met) -> FloatSolveConfig {
    full_stack_config_mapping(corrections, met, TropoMapping::Niell)
}

fn full_stack_config_mapping(
    corrections: RangeCorrections,
    met: Met,
    mapping: TropoMapping,
) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: MeasurementWeights {
            code: 1.0,
            phase: 100.0,
            elevation_weighting: true,
        },
        tropo: TroposphereOptions {
            enabled: true,
            estimate_ztd: true,
            met,
            mapping,
        },
        corrections,
        opts: FloatSolveOptions {
            max_iterations: 12,
            position_tolerance_m: 1.0e-4,
            clock_tolerance_m: 1.0e-4,
            ambiguity_tolerance_m: 1.0e-4,
            ztd_tolerance_m: 1.0e-4,
        },
        residual_screen: false,
    }
}

/// ZIM2 VMF1 site-wise mapping `a` coefficients for 2026-05-13 (MJD 61173),
/// the four 6-hourly nodes 00/06/12/18 UT.
///
/// Provenance: TU Wien VMF data server, GNSS site-wise VMF1 (operational),
/// `https://vmf.geo.tuwien.ac.at/trop_products/GNSS/VMF1/VMF1_OP/daily/2026/2026133.vmf1_g`,
/// fetched 2026-06-27; the ZIM2 rows (file columns: station, MJD, ah, aw, zhd,
/// zwd, orography, pressure, temperature, water-vapour pressure, station height).
/// ZIM2 ellipsoidal coordinates 46.8771 N, 7.4650 E, 956.40 m
/// (`station_coord_files/gnss.ell`). Real provider values, not fabricated.
fn zim2_vmf1_series() -> VmfSiteSeries {
    VmfSiteSeries::new(&[
        VmfSiteSample {
            mjd: 61173.00,
            ah: 0.00121738,
            aw: 0.00058796,
        },
        VmfSiteSample {
            mjd: 61173.25,
            ah: 0.00121388,
            aw: 0.00053850,
        },
        VmfSiteSample {
            mjd: 61173.50,
            ah: 0.00121315,
            aw: 0.00048897,
        },
        VmfSiteSample {
            mjd: 61173.75,
            ah: 0.00121222,
            aw: 0.00052133,
        },
    ])
    .expect("valid ZIM2 VMF1 site series")
}

/// THE HEADLINE: full correction stack ON reaches decimeter truth and matches
/// RTKLIB PPP-static, on a real IGS station arc fed by a 30 s CLK.
#[test]
fn zim2_full_stack_ppp_static_reaches_decimeter_truth() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let antex = load_antex();
    let clock = load_clock();

    let approx = obs
        .header()
        .approx_position_m
        .expect("ZIM2 approx position");
    let epochs = gps_float_epochs(&sp3, &obs, approx);
    assert_eq!(epochs.len(), 120);

    // Correction precompute (tide/wind-up/satellite-antenna geometry) is anchored
    // at the operationally available RINEX approximate position, not the truth.
    let corrections = full_corrections(&sp3, &antex, &clock, &epochs, approx);
    let solution = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, approx),
        full_stack_config(corrections, station_met(approx)),
    )
    .expect("full-stack PPP float solve");

    let truth_err = position_error_m(solution.position_m, ZIM2_TRUTH_ECEF_M);
    let rtklib_err = position_error_m(solution.position_m, RTKLIB_PPP_STATIC_ECEF_M);
    let rtklib_vs_truth = position_error_m(RTKLIB_PPP_STATIC_ECEF_M, ZIM2_TRUTH_ECEF_M);

    eprintln!(
        "ZIM2 full-stack PPP-static: pos={:?}\n  vs ITRF2020 truth = {truth_err:.4} m\n  vs RTKLIB ppp-static = {rtklib_err:.4} m\n  (RTKLIB vs truth = {rtklib_vs_truth:.4} m)",
        solution.position_m
    );

    assert!(
        truth_err < DECIMETER_TRUTH_BOUND_M,
        "full-stack PPP truth error {truth_err} m exceeded decimeter bound {DECIMETER_TRUTH_BOUND_M} m"
    );
    assert!(
        rtklib_err < SIDEREON_VS_RTKLIB_BOUND_M,
        "full-stack PPP vs RTKLIB ppp-static {rtklib_err} m exceeded {SIDEREON_VS_RTKLIB_BOUND_M} m"
    );
}

/// VMF1 mapping (Vienna, driven by the ZIM2 site-wise `a` coefficients) holds
/// the decimeter truth and stays within a few millimetres of the Niell mapping
/// on this inland, well-conditioned arc -- the no-regression bar for swapping the
/// mapping function. Niell remains the default and is unchanged.
#[test]
fn zim2_full_stack_ppp_static_vmf1_matches_niell() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let antex = load_antex();
    let clock = load_clock();

    let approx = obs
        .header()
        .approx_position_m
        .expect("ZIM2 approx position");
    let epochs = gps_float_epochs(&sp3, &obs, approx);
    let met = station_met(approx);

    let solve = |mapping: TropoMapping| -> [f64; 3] {
        let corrections = full_corrections(&sp3, &antex, &clock, &epochs, approx);
        solve_float_epochs(
            &sp3,
            &epochs,
            initial_state(&epochs, approx),
            full_stack_config_mapping(corrections, met, mapping),
        )
        .expect("full-stack PPP float solve")
        .position_m
    };

    let niell_pos = solve(TropoMapping::Niell);
    let vmf1_pos = solve(TropoMapping::Vmf1(zim2_vmf1_series()));

    let niell_err = position_error_m(niell_pos, ZIM2_TRUTH_ECEF_M);
    let vmf1_err = position_error_m(vmf1_pos, ZIM2_TRUTH_ECEF_M);
    let vmf1_vs_niell = position_error_m(vmf1_pos, niell_pos);

    eprintln!(
        "ZIM2 full-stack PPP-static mapping comparison:\n  Niell truth err = {niell_err:.4} m\n  VMF1  truth err = {vmf1_err:.4} m\n  VMF1 vs Niell    = {vmf1_vs_niell:.4} m"
    );

    assert!(
        vmf1_err < DECIMETER_TRUTH_BOUND_M,
        "VMF1 full-stack PPP truth error {vmf1_err} m exceeded decimeter bound {DECIMETER_TRUTH_BOUND_M} m"
    );
    // Inland, high-elevation-dominated arc: the two mapping functions agree to a
    // few millimetres. Generous bound so this stays a no-regression guard, not a
    // brittle exact-match.
    assert!(
        vmf1_vs_niell < 0.03,
        "VMF1 vs Niell position difference {vmf1_vs_niell} m exceeded 0.03 m no-regression bound"
    );
}

/// Per-correction contribution: the dominant relativistic-clock term collapses
/// the meters-level error, ZTD estimation refines it, and the full stack holds a
/// decimeter. Demonstrates WHY `disabled()` cannot reach decimeter. (The final
/// antenna/tide/wind-up terms are sub-centimeter on this masked arc, so this is a
/// staged contribution breakdown, not a strict monotonic ordering at every step.)
#[test]
fn zim2_correction_stack_progressively_approaches_truth() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let antex = load_antex();
    let clock = load_clock();
    let approx = obs
        .header()
        .approx_position_m
        .expect("ZIM2 approx position");
    let epochs = gps_float_epochs(&sp3, &obs, approx);

    let met = station_met(approx);
    let solve = |corrections: RangeCorrections, ztd: bool, elev: bool| -> f64 {
        let mut config = full_stack_config(corrections, met);
        config.tropo.estimate_ztd = ztd;
        config.weights.elevation_weighting = elev;
        let solution = solve_float_epochs(&sp3, &epochs, initial_state(&epochs, approx), config)
            .expect("staged PPP float solve");
        position_error_m(solution.position_m, ZIM2_TRUTH_ECEF_M)
    };

    // CLK + a priori troposphere, NO relativity (and no antenna/tide/wind-up): the
    // missing relativistic-clock term alone leaves a meters-level error. (The
    // a priori hydrostatic troposphere is on throughout via `full_stack_config`;
    // only ZTD estimation is toggled by the `ztd` argument.)
    let clock_only = RangeCorrections {
        sat_clock_relativity: false,
        satellite_clock: Some(satellite_clock_corrections(&clock)),
        ..RangeCorrections::disabled()
    };
    let e_clock_only = solve(clock_only, false, false);

    // + relativistic clock.
    let with_relativity = RangeCorrections {
        sat_clock_relativity: true,
        satellite_clock: Some(satellite_clock_corrections(&clock)),
        ..RangeCorrections::disabled()
    };
    let e_relativity = solve(with_relativity.clone(), false, false);

    // + ZTD estimation.
    let e_ztd = solve(with_relativity, true, false);

    // + full stack (antenna PCO/PCV, tide, wind-up) + elevation weighting.
    let full = full_corrections(&sp3, &antex, &clock, &epochs, approx);
    let e_full = solve(full, true, true);

    eprintln!(
        "ZIM2 staged truth error (m): clk+apriori-tropo(no-rel)={e_clock_only:.3} +relativity={e_relativity:.3} +ZTD={e_ztd:.3} +full-stack={e_full:.3}"
    );

    // The missing relativistic-clock term is the dominant, meters-level error;
    // enabling it collapses the error by ~two orders of magnitude. ZTD estimation
    // then refines it further. The remaining antenna/tide/wind-up terms are
    // sub-centimeter on this masked arc (their headline value is robustness across
    // stations/geometry, not this single number), so the full stack is asserted to
    // hold the decimeter bound rather than to strictly beat the already-excellent
    // ZTD-only stage.
    assert!(
        e_clock_only > 2.0,
        "missing relativity should be meters-level"
    );
    assert!(
        e_relativity < e_clock_only / 10.0,
        "relativistic clock must collapse the meters-level error ({e_relativity} !<< {e_clock_only})"
    );
    assert!(
        e_ztd < e_relativity,
        "ZTD estimation must refine the relativity-corrected solution ({e_ztd} !< {e_relativity})"
    );
    assert!(
        e_full < DECIMETER_TRUTH_BOUND_M,
        "full stack must reach decimeter ({e_full} m)"
    );
}
