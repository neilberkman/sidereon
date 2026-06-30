//! RINEX NAV parser coverage against the committed multi-GNSS fixture.
//!
//! Parsing is deterministic byte-to-record translation, so these are round-trip
//! and schema assertions (counts, field ranges, message classification, a
//! physical sanity check via the evaluator), NOT a 0-ULP parity claim.
//!
//! Fixture provenance (all under `tests/fixtures/nav/`; the nav-solutions/data
//! repo redistributes public IGS/MGEX products, original source CDDIS/BKG/IGS):
//!
//!   * `ESBC00DNK_R_20201770000_01D_MN.rnx` — IGS MGEX daily merged broadcast nav
//!     (RINEX 3.05 MIXED, station ESBC00DNK Esbjerg DK, 2020 DOY 177, GPS week
//!     2111). From
//!     `https://raw.githubusercontent.com/nav-solutions/data/main/NAV/V3/ESBC00DNK_R_20201770000_01D_MN.rnx.gz`
//!     (gz 285554 B sha256 3b930e79ec15c384622425a61f21f1f13f5980b9025f0a788de2882cf8898274;
//!     decompressed 2359118 B sha256
//!     ad6af3c21d2f97a0cb538a77fcf0acad5a59ade9d0987fd523b0b7d483317a4b). Committed
//!     copy is the decompressed product filtered to GPS+Galileo+BeiDou records (the
//!     Keplerian constellations) with the header verbatim through END OF HEADER, via
//!     a deterministic awk pass keeping `^[GEC]` records (1452728 B sha256
//!     069f73afc10e9c1a8b87b7fbbb774f3eb9be94fb4da4ac365cfd4356c6ebfd36; 257 GPS,
//!     1602 Galileo, 357 BeiDou records; BeiDou C05-C37 exercises GEO/IGSO/MEO).
//!   * `ESBC00DNK_R_20201770000_01D_RN.rnx` — GLONASS (`^R`) records of the same
//!     original ESBC00DNK product (decompressed sha256 ad6af3c2…), header verbatim,
//!     same awk pass keeping `^R`. 510 GLONASS broadcast records (5-line PZ-90.11
//!     3.05 layout); header LEAP SECONDS = 18.
//!   * `KMS300DNK_R_20221591000_01H_MN.rnx` — RINEX 4.00 MIXED nav, 1 hour (2022 DOY
//!     159), committed verbatim (decompressed) from nav-solutions/data NAV/V4
//!     (gz sha256 2bae4217cb71ad4a2b9c0067bd1c5b56915e42d2007a94e91eb408468cc4763f).
//!     Tests version-4 frame-marker parsing; 174 supported Keplerian records parsed,
//!     GLONASS/QZSS/SBAS/STO/ION skipped.
//!   * `BRDC00GOP_R_20210010000_01D_MN.rnx` — merged BRDC header (GOP/Pecny),
//!     header-only, from nav-solutions/data NAV/V3 (gz sha256
//!     1bb7bb0ca70fb1e11e366abd9126881d62b238b687ace7fba360002b61a12f09). Carries
//!     IONOSPHERIC CORR for GPS/Galileo/QZSS/NavIC and — the reason it is committed —
//!     BeiDou (BDSA/BDSB Klobuchar-8). No orbit records.

use super::*;
use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::broadcast::satellite_state;

fn fixture_text() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read NAV fixture {path}: {e}"))
}

fn records() -> Vec<BroadcastRecord> {
    parse_nav(&fixture_text()).expect("parse NAV fixture")
}

fn broadcast_time(system: GnssSystem, week: u32, sow: f64) -> GnssWeekTow {
    GnssWeekTow::new(
        match system {
            GnssSystem::Galileo => TimeScale::Gst,
            GnssSystem::BeiDou => TimeScale::Bdt,
            _ => TimeScale::Gpst,
        },
        week,
        sow,
    )
    .expect("valid week/TOW")
    .normalized()
    .expect("valid normalized week/TOW")
}

fn v4_fixture_text() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/KMS300DNK_R_20221591000_01H_MN.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read v4 NAV fixture {path}: {e}"))
}

fn glonass_fixture_text() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_RN.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read GLONASS fixture {path}: {e}"))
}

#[test]
fn parses_and_evaluates_glonass_records() {
    use crate::spp::EphemerisSource;

    let text = glonass_fixture_text();
    let recs = parse_glonass(&text).expect("parse GLONASS records");
    assert_eq!(recs.len(), 510, "GLONASS record count");
    assert_eq!(
        parse_leap_seconds(&text).expect("parse leap seconds"),
        Some(18.0),
        "GPS-UTC leap seconds"
    );

    // Every record's broadcast state sits on the GLONASS orbit (~25,510 km).
    for r in &recs {
        let radius_km =
            (r.pos_m[0].powi(2) + r.pos_m[1].powi(2) + r.pos_m[2].powi(2)).sqrt() / 1000.0;
        assert!(
            (25_000.0..26_000.0).contains(&radius_km),
            "{:?} GLONASS radius {radius_km} km out of band",
            r.satellite_id
        );
    }

    // The store evaluates a GLONASS satellite through the RK4 propagator. At the
    // record's own reference epoch (tk = 0) the position is the broadcast state,
    // so the radius is the GLONASS orbit radius.
    let store = BroadcastStore::from_nav(&text).expect("parse GLONASS NAV");
    assert_eq!(store.glonass_records().len(), 510);
    let r0 = store.glonass_records()[0];
    let t_toe_gpst = r0.toe_utc_j2000_s + 18.0; // leap seconds for 2020
    let (pos, _clk) = store
        .position_clock_at_j2000_s(r0.satellite_id, t_toe_gpst)
        .expect("GLONASS position at its toe");
    let radius_km = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt() / 1000.0;
    assert!(
        (25_000.0..26_000.0).contains(&radius_km),
        "evaluated GLONASS radius {radius_km} km out of band"
    );
    // tk = 0 means no integration, so the evaluated position equals the state.
    assert_eq!(
        [pos[0], pos[1], pos[2]],
        r0.pos_m,
        "tk=0 returns the broadcast state"
    );

    // A query far outside the product's coverage (a day before any record) has
    // no record within the +/-15 min validity window, so no ephemeris. (A query
    // an hour later would instead be served by the next half-hourly record.)
    assert!(
        store
            .position_clock_at_j2000_s(r0.satellite_id, t_toe_gpst - 86_400.0)
            .is_none(),
        "a query a day before any record is outside every validity window"
    );
}

/// The committed `ESBC00DNK_R_20201770000_01D_RN.rnx` fixture is the real
/// RINEX 3.05 GLONASS layout: a `3.05` header and FIVE physical lines per
/// record (the epoch/clock line plus FOUR broadcast-orbit lines). The fourth
/// orbit line is the one RINEX 3.05 added over 3.04 (status flags, the L1/L2
/// group-delay difference dtaun, URAI, health), and gfzrnx wrote its dtaun
/// field as the "unavailable" sentinel `.999999999999e+09`.
///
/// This locks the provenance: the file is NOT a three-line 3.04 layout, the
/// fourth orbit line IS present, and sidereon parses the file correctly because
/// `parse_glonass` delimits records by record-start lines and consumes only the
/// epoch + first three orbit lines, IGNORING the fourth orbit line entirely.
/// Ignoring dtaun is correct for an L1-only single-frequency user (no L1/L2
/// inter-frequency group delay is applied).
#[test]
fn committed_rn_fixture_is_rinex_305_five_line_layout_parsed_correctly() {
    let text = glonass_fixture_text();

    // Header declares RINEX 3.05.
    let version_line = text
        .lines()
        .find(|l| l.contains("RINEX VERSION / TYPE"))
        .expect("version line");
    assert!(
        version_line.trim_start().starts_with("3.05"),
        "committed RN header must declare 3.05, got {version_line:?}"
    );

    // Re-block the body exactly as the parser does (a record starts on an
    // alpha-digit-digit line) and confirm the first GLONASS record is FIVE
    // physical lines with the 3.05 fourth-orbit dtaun sentinel on line 5.
    let body = text
        .split_once("END OF HEADER")
        .map(|(_, b)| b.trim_start_matches(['\r', '\n']))
        .expect("END OF HEADER");
    let is_record_start = |line: &str| {
        let b = line.as_bytes();
        b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1].is_ascii_digit() && b[2].is_ascii_digit()
    };
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for line in body.lines() {
        if is_record_start(line) {
            blocks.push(vec![line]);
        } else if let Some(last) = blocks.last_mut() {
            last.push(line);
        }
    }
    let first_glonass = blocks
        .iter()
        .find(|b| b[0].starts_with('R'))
        .expect("a GLONASS record");
    assert_eq!(
        first_glonass.len(),
        5,
        "RINEX 3.05 GLONASS record is 5 physical lines (epoch + 4 orbit lines), \
         not the 4-line 3.04 layout; got {first_glonass:?}"
    );
    assert!(
        first_glonass[4].contains(".999999999999e+09"),
        "the 3.05 fourth orbit line carries the gfzrnx 'unavailable' dtaun \
         sentinel, got {:?}",
        first_glonass[4]
    );

    // The fourth orbit line is genuinely IGNORED: parsing still yields the
    // correct R01 broadcast state from the first three orbit lines.
    let recs = parse_glonass(&text).expect("parse GLONASS records");
    let r01 = recs
        .iter()
        .find(|r| r.satellite_id.system == GnssSystem::Glonass && r.satellite_id.prn == 1)
        .expect("R01 present");
    assert_eq!(r01.freq_channel, 1, "R01 FDMA channel from orbit-2 field 4");
    assert_eq!(r01.sv_health, 0.0, "R01 health from orbit-1 field 4");
    assert!(r01.gamma_n.is_finite(), "R01 gamma_n parsed");
    assert!(
        r01.toe_utc_j2000_s.is_finite(),
        "R01 epoch parsed (4th orbit line did not corrupt the record stream)"
    );
}

#[test]
fn spp_solves_from_broadcast_glonass() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, ELEVATION_MASK_RAD,
    };

    // GLONASS-only store from the RN fixture.
    let store = BroadcastStore::from_nav(&glonass_fixture_text()).expect("parse GLONASS NAV");

    // 2020-06-25 12:00 GPST, mid-day so GLONASS satellites have a near-epoch
    // record. The ionosphere correction is unsupported for GLONASS (no modeled
    // single-frequency carrier), so this geometry-only solve leaves it off.
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.0;
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0, 0.0];
    let corr = Corrections::NONE;
    let kl = KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    let mut sats: Vec<_> = store
        .glonass_records()
        .iter()
        .map(|r| r.satellite_id)
        .collect();
    sats.sort_unstable();
    sats.dedup();
    let mut observations = Vec::new();
    for sat in sats {
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: corr,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) = test_support::sat_model_for_test(
            &env,
            sat,
            [x_true[0], x_true[1], x_true[2]],
            x_true[3],
            20_000_000.0,
            &kl,
        ) {
            if m.el_rad >= ELEVATION_MASK_RAD {
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
            }
        }
    }
    assert!(
        observations.len() >= 4,
        "need >=4 visible GLONASS sats, got {}",
        observations.len()
    );

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: corr,
        klobuchar: kl,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    let sol = solve(&store, &inputs, true).expect("GLONASS broadcast SPP solve");
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(err < 1.0e-3, "recovered position off by {err} m");
    // A single-system GLONASS solve carries one receiver clock.
    assert_eq!(sol.system_clocks_s.len(), 1, "one GLONASS clock");
    assert_eq!(sol.system_clocks_s[0].0, GnssSystem::Glonass);
}

#[test]
fn beidou_uses_its_own_klobuchar_coefficients() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, ELEVATION_MASK_RAD,
    };

    // BeiDou-only store.
    let store = BroadcastStore::new(
        records()
            .into_iter()
            .filter(|r| r.satellite_id.system == GnssSystem::BeiDou)
            .collect(),
    )
    .expect("valid manual BeiDou broadcast store");
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.0;
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0];
    // The broadcast BeiDou Klobuchar-8 set (BDSA/BDSB).
    let bds = KlobucharCoeffs {
        alpha: [1.1180e-08, 2.9800e-08, -4.1720e-07, 6.5570e-07],
        beta: [1.4130e05, -5.2430e05, 1.6380e06, -4.5880e05],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    // Synthesize BeiDou observations with the ionosphere applied using the BeiDou
    // coefficients (sat_model scales the L1 delay to B1I for BeiDou).
    let mut sats: Vec<_> = store.records().iter().map(|r| r.satellite_id).collect();
    sats.sort_unstable();
    sats.dedup();
    let mut observations = Vec::new();
    for sat in sats {
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: Corrections::IONO,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) =
            test_support::sat_model_for_test(&env, sat, x_true, 0.0, 22_000_000.0, &bds)
        {
            if m.el_rad >= ELEVATION_MASK_RAD {
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
            }
        }
    }
    assert!(
        observations.len() >= 4,
        "need >=4 BeiDou sats, got {}",
        observations.len()
    );

    let base = |beidou_klobuchar| SolveInputs {
        observations: observations.clone(),
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: Corrections::IONO,
        // Zero GPS-side coefficients: if BeiDou wrongly used these, no ionosphere
        // would be applied and the synthesized delay would bias the solution.
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    // With the BeiDou coefficients supplied, BeiDou uses them and the truth is
    // recovered (the applied ionosphere matches the synthesized one).
    let sol = solve(&store, &base(Some(bds)), false).expect("BeiDou-native iono solve");
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(
        err < 1.0e-3,
        "with BDSA/BDSB the solve recovers; off by {err} m"
    );

    // Without them, BeiDou falls back to the (zero) shared set, so the modelled
    // ionosphere is missing and the solution is biased - proving the per-system
    // coefficients are actually used.
    let sol0 = solve(&store, &base(None), false).expect("fallback solve");
    let p0 = sol0.position;
    let err0 = ((p0.x_m - x_true[0]).powi(2)
        + (p0.y_m - x_true[1]).powi(2)
        + (p0.z_m - x_true[2]).powi(2))
    .sqrt();
    assert!(
        err0 > 0.1,
        "without BeiDou coeffs the unmodelled ionosphere biases the fix; off by {err0} m"
    );
}

fn brdc_gop_text() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/BRDC00GOP_R_20210010000_01D_MN.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read BRDC00GOP fixture {path}: {e}"))
}

#[test]
fn parses_broadcast_ionosphere_coefficients() {
    // The main fixture carries GPS (and Galileo NeQuick) coefficients but no
    // BeiDou set.
    let esbc = parse_iono_corrections(&fixture_text()).expect("parse ESBC ionosphere header");
    let gps = esbc.gps.expect("ESBC has GPSA/GPSB");
    assert!(
        (gps.alpha[0] - 4.6566e-09).abs() < 1e-19,
        "GPSA a0 {}",
        gps.alpha[0]
    );
    assert!(
        (gps.beta[0] - 8.1920e04).abs() < 1e-3,
        "GPSB b0 {}",
        gps.beta[0]
    );
    let gal = esbc.galileo.expect("ESBC has GAL NeQuick coefficients");
    assert!((gal.ai0 - 2.8250e01).abs() < 1e-10, "GAL ai0 {}", gal.ai0);
    assert!((gal.ai1 - 7.8125e-03).abs() < 1e-12, "GAL ai1 {}", gal.ai1);
    assert!((gal.ai2 - 1.0071e-02).abs() < 1e-12, "GAL ai2 {}", gal.ai2);
    assert!(esbc.beidou.is_none(), "ESBC has no BDSA/BDSB");

    // The merged BRDC header carries the BeiDou Klobuchar-8 set.
    let brdc = parse_iono_corrections(&brdc_gop_text()).expect("parse BRDC ionosphere header");
    let bds = brdc.beidou.expect("BRDC00GOP has BDSA/BDSB");
    assert!(
        (bds.alpha[0] - 1.1180e-08).abs() < 1e-18,
        "BDSA a0 {}",
        bds.alpha[0]
    );
    assert!(
        (bds.alpha[2] - -4.1720e-07).abs() < 1e-17,
        "BDSA a2 {}",
        bds.alpha[2]
    );
    assert!(
        (bds.beta[0] - 1.4130e05).abs() < 1e-3,
        "BDSB b0 {}",
        bds.beta[0]
    );
    assert!(
        (bds.beta[1] - -5.2430e05).abs() < 1e-3,
        "BDSB b1 {}",
        bds.beta[1]
    );
    assert!(brdc.gps.is_some(), "BRDC00GOP also has GPSA/GPSB");
    assert!(brdc.galileo.is_some(), "BRDC00GOP also has GAL");
}

#[test]
fn broadcast_store_exposes_header_ionosphere_coefficients() {
    // from_nav captures the header coefficients; new() leaves them empty.
    let store = BroadcastStore::from_nav(&brdc_gop_text()).expect("parse BRDC00GOP");
    assert!(
        store.iono_corrections().beidou.is_some(),
        "BeiDou coeffs from header"
    );
    assert!(
        store.iono_corrections().galileo.is_some(),
        "Galileo coeffs from header"
    );

    let bare = BroadcastStore::new(vec![]).expect("empty manual broadcast store");
    assert_eq!(
        bare.iono_corrections(),
        Default::default(),
        "new() has no coeffs"
    );
}

#[test]
fn parses_rinex_v4_body_ionosphere_frames() {
    let text = v4_fixture_text();
    let parsed = parse_iono_corrections(&text).expect("parse v4 ionosphere body frames");
    let gps = parsed.gps.expect("KMS RINEX 4 fixture has GPS ION frame");
    assert!(
        (gps.alpha[0] - 1.024454832077e-08).abs() < 1e-20,
        "GPS alpha0 {}",
        gps.alpha[0]
    );
    assert!(
        (gps.alpha[3] - -1.192092895508e-07).abs() < 1e-19,
        "GPS alpha3 {}",
        gps.alpha[3]
    );
    assert!(
        (gps.beta[0] - 9.6256e04).abs() < 1e-6,
        "GPS beta0 {}",
        gps.beta[0]
    );
    assert!(
        (gps.beta[3] - -5.89824e05).abs() < 1e-5,
        "GPS beta3 {}",
        gps.beta[3]
    );

    let bds = parsed
        .beidou
        .expect("KMS RINEX 4 fixture has BeiDou ION frame");
    assert!(
        (bds.alpha[0] - 2.142041921616e-08).abs() < 1e-20,
        "BDS alpha0 {}",
        bds.alpha[0]
    );
    assert!(
        (bds.alpha[3] - 1.549720764160e-06).abs() < 1e-18,
        "BDS alpha3 {}",
        bds.alpha[3]
    );
    assert!(
        (bds.beta[0] - 1.20832e05).abs() < 1e-6,
        "BDS beta0 {}",
        bds.beta[0]
    );
    assert!(
        (bds.beta[3] - -6.5536e04).abs() < 1e-6,
        "BDS beta3 {}",
        bds.beta[3]
    );

    let gal = parsed
        .galileo
        .expect("KMS RINEX 4 fixture has Galileo ION frame");
    assert!((gal.ai0 - 7.85e01).abs() < 1e-10, "GAL ai0 {}", gal.ai0);
    assert!(
        (gal.ai1 - 5.390625e-01).abs() < 1e-12,
        "GAL ai1 {}",
        gal.ai1
    );
    assert!(
        (gal.ai2 - 2.713012695312e-02).abs() < 1e-14,
        "GAL ai2 {}",
        gal.ai2
    );

    let store = BroadcastStore::from_nav(&text).expect("parse KMS RINEX 4 fixture");
    assert_eq!(store.iono_corrections(), parsed);
}

#[test]
fn parses_a_real_rinex_v4_file() {
    let recs = parse_nav(&v4_fixture_text()).expect("parse v4 NAV fixture");
    let count = |sys| recs.iter().filter(|r| r.satellite_id.system == sys).count();
    let msg = |m| recs.iter().filter(|r| r.message == m).count();

    // Supported Keplerian records only: GPS LNAV, Galileo I/NAV + F/NAV, BeiDou
    // D1 + D2. GLONASS (FDMA), QZSS, SBAS, STO and ION frames are skipped.
    assert_eq!(count(GnssSystem::Gps), 30, "GPS LNAV count");
    assert_eq!(count(GnssSystem::Galileo), 108, "Galileo count");
    assert_eq!(count(GnssSystem::BeiDou), 36, "BeiDou count");
    assert_eq!(recs.len(), 174, "only G/E/C are parsed");
    assert_eq!(
        count(GnssSystem::Glonass) + count(GnssSystem::Qzss) + count(GnssSystem::Sbas),
        0,
        "GLONASS/QZSS/SBAS must be skipped"
    );

    // Message type comes from the v4 marker token.
    assert_eq!(msg(NavMessage::GpsLnav), 30);
    assert_eq!(msg(NavMessage::GalileoInav), 55);
    assert_eq!(msg(NavMessage::GalileoFnav), 53);
    assert_eq!(msg(NavMessage::BeidouD1), 33);
    assert_eq!(msg(NavMessage::BeidouD2), 3);

    // Parsed records evaluate to physical orbit radii (parser-to-evaluator sanity
    // on real v4 bytes), MEO/IGSO/GEO bands across the constellations.
    for sys in [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou] {
        let r = recs.iter().find(|r| r.satellite_id.system == sys).unwrap();
        let st = satellite_state(
            &r.elements,
            &r.clock,
            &r.constants(),
            r.elements.toe_sow,
            r.broadcast_clock_group_delay_s(),
            crate::rinex_nav::is_beidou_geo(r.satellite_id),
        )
        .expect("valid parsed v4 broadcast record");
        let p = st.orbit.position().expect("valid orbit position");
        let radius_km = (p.x_m * p.x_m + p.y_m * p.y_m + p.z_m * p.z_m).sqrt() / 1000.0;
        assert!(
            (20_000.0..50_000.0).contains(&radius_km),
            "{sys:?} v4 radius {radius_km} km out of band"
        );
    }
}

#[test]
fn parses_gps_galileo_and_beidou_records() {
    let recs = records();
    let count = |sys| recs.iter().filter(|r| r.satellite_id.system == sys).count();
    let gps = count(GnssSystem::Gps);
    let gal = count(GnssSystem::Galileo);
    let bds = count(GnssSystem::BeiDou);
    // The committed fixture is filtered to GPS + Galileo + BeiDou.
    assert_eq!(gps, 257, "GPS record count");
    assert_eq!(gal, 1602, "Galileo record count");
    assert_eq!(bds, 357, "BeiDou record count");
    assert_eq!(
        recs.len(),
        gps + gal + bds,
        "only GPS+Galileo+BeiDou are returned"
    );
}

#[test]
fn gps_record_fields_are_in_range() {
    let recs = records();
    let g01 = recs
        .iter()
        .find(|r| {
            r.satellite_id == GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id")
        })
        .expect("a G01 record");

    assert_eq!(g01.message, NavMessage::GpsLnav);
    assert_eq!(g01.week, 2111, "GPS week 2111 for this product");
    // GPS semi-major axis ~26560 km => sqrt(a) ~ 5153.6 sqrt(m).
    assert!(
        (5100.0..5200.0).contains(&g01.elements.sqrt_a),
        "sqrt_a {}",
        g01.elements.sqrt_a
    );
    assert!(
        (0.0..0.05).contains(&g01.elements.e),
        "e {}",
        g01.elements.e
    );
    // For this record the clock and ephemeris reference epochs coincide.
    assert_eq!(g01.clock.toc_sow, g01.elements.toe_sow);
    assert_eq!(g01.sv_health, 0.0, "G01 is healthy");
    assert!(
        g01.group_delays
            .get(BroadcastGroupDelayTerm::GpsTgd)
            .expect("GPS TGD")
            .abs()
            < 1.0e-6,
        "TGD is a small delay"
    );
}

#[test]
fn galileo_messages_are_classified() {
    let recs = records();
    let gal: Vec<_> = recs
        .iter()
        .filter(|r| r.satellite_id.system == GnssSystem::Galileo)
        .collect();
    let inav = gal
        .iter()
        .filter(|r| r.message == NavMessage::GalileoInav)
        .count();
    let fnav = gal
        .iter()
        .filter(|r| r.message == NavMessage::GalileoFnav)
        .count();
    assert_eq!(inav, 821, "Galileo I/NAV record count");
    assert_eq!(fnav, 781, "Galileo F/NAV record count");
    assert_eq!(inav + fnav, gal.len(), "every Galileo record is classified");
}

#[test]
fn galileo_inav_uses_e5b_e1_bgd_for_clock() {
    use crate::spp::EphemerisSource;

    const BGD_E5A_E1_S: f64 = 1.0e-8;
    const BGD_E5B_E1_S: f64 = 2.5e-8;

    let mut lines = e01_lines();
    lines[5] = replace_orbit_field(&lines[5], 1, "1.000000000000e+00");
    lines[6] = replace_orbit_field(&lines[6], 2, "1.000000000000e-08");
    lines[6] = replace_orbit_field(&lines[6], 3, "2.500000000000e-08");
    let text = nav_text(&lines);

    let recs = parse_nav(&text).expect("parse Galileo I/NAV record");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].message, NavMessage::GalileoInav);
    assert_eq!(
        recs[0]
            .group_delays
            .get(BroadcastGroupDelayTerm::GalileoBgdE5aE1)
            .expect("Galileo BGD E5a/E1")
            .to_bits(),
        BGD_E5A_E1_S.to_bits(),
        "Galileo BGD E5a/E1 must be preserved"
    );
    assert_eq!(
        recs[0]
            .group_delays
            .get(BroadcastGroupDelayTerm::GalileoBgdE5bE1)
            .expect("Galileo BGD E5b/E1")
            .to_bits(),
        BGD_E5B_E1_S.to_bits(),
        "Galileo BGD E5b/E1 must be preserved"
    );
    assert!(
        (recs[0].broadcast_clock_group_delay_s() - BGD_E5B_E1_S).abs() < 1.0e-20,
        "I/NAV must use BGD E5b/E1"
    );

    let store = BroadcastStore::from_nav(&text).expect("default Galileo store");
    let rec = &store.records()[0];
    let (_, clock_s) = store
        .position_clock_at_j2000_s(rec.satellite_id, toe_as_j2000_s(rec))
        .expect("I/NAV record evaluates at toe");
    let expected_inav_clock_s = satellite_state(
        &rec.elements,
        &rec.clock,
        &rec.constants(),
        rec.elements.toe_sow,
        BGD_E5B_E1_S,
        false,
    )
    .expect("valid Galileo I/NAV broadcast state")
    .clock
    .dt_clock_total_s;
    let fnav_bgd_clock_s = satellite_state(
        &rec.elements,
        &rec.clock,
        &rec.constants(),
        rec.elements.toe_sow,
        BGD_E5A_E1_S,
        false,
    )
    .expect("valid Galileo F/NAV broadcast state")
    .clock
    .dt_clock_total_s;
    assert!(
        (clock_s - expected_inav_clock_s).abs() < 1.0e-18,
        "store clock must use the I/NAV BGD"
    );
    assert!(
        (clock_s - fnav_bgd_clock_s).abs() > 1.0e-9,
        "using the F/NAV BGD would leave a visible clock bias"
    );
}

#[test]
fn galileo_fnav_source_bit_uses_e5a_e1_bgd_for_clock() {
    const BGD_E5A_E1_S: f64 = 1.0e-8;
    const BGD_E5B_E1_S: f64 = 2.5e-8;

    let mut lines = e01_lines();
    lines[5] = replace_orbit_field(&lines[5], 1, "2.000000000000e+00");
    lines[6] = replace_orbit_field(&lines[6], 2, "1.000000000000e-08");
    lines[6] = replace_orbit_field(&lines[6], 3, "2.500000000000e-08");
    let text = nav_text(&lines);

    let recs = parse_nav(&text).expect("parse Galileo F/NAV record");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].message, NavMessage::GalileoFnav);
    assert!(
        (recs[0].broadcast_clock_group_delay_s() - BGD_E5A_E1_S).abs() < 1.0e-20,
        "F/NAV must use BGD E5a/E1"
    );

    let store = BroadcastStore::from_nav(&text).expect("default Galileo store");
    assert!(
        store.records().is_empty(),
        "default store must still exclude Galileo F/NAV records"
    );
    assert_ne!(
        recs[0].broadcast_clock_group_delay_s().to_bits(),
        BGD_E5B_E1_S.to_bits(),
        "F/NAV source bit must not select the I/NAV BGD"
    );
}

#[test]
fn beidou_record_preserves_tgd1_and_tgd2_terms() {
    const TGD1_S: f64 = -3.25e-9;
    const TGD2_S: f64 = 7.75e-9;

    let mut lines = satellite_lines(G01_LINES, "C19");
    lines[6] = replace_orbit_field(&lines[6], 2, "-3.250000000000e-09");
    lines[6] = replace_orbit_field(&lines[6], 3, "7.750000000000e-09");
    let text = nav_text(&lines);

    let recs = parse_nav(&text).expect("parse BeiDou record");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].message, NavMessage::BeidouD1);
    assert_eq!(
        recs[0]
            .group_delays
            .get(BroadcastGroupDelayTerm::BeidouTgd1)
            .expect("BeiDou TGD1")
            .to_bits(),
        TGD1_S.to_bits()
    );
    assert_eq!(
        recs[0]
            .group_delays
            .get(BroadcastGroupDelayTerm::BeidouTgd2)
            .expect("BeiDou TGD2")
            .to_bits(),
        TGD2_S.to_bits()
    );
    assert_eq!(
        recs[0].broadcast_clock_group_delay_s().to_bits(),
        TGD1_S.to_bits(),
        "default broadcast-clock path keeps prior TGD1 behavior"
    );
}

#[test]
fn parsed_records_evaluate_to_physical_orbit_radii() {
    let recs = records();
    // Evaluate each constellation's first record at its toe and check the ECEF
    // radius is in the expected MEO band (parser-to-evaluator sanity).
    for (system, lo_km, hi_km) in [
        (GnssSystem::Gps, 25_000.0, 27_500.0),
        (GnssSystem::Galileo, 29_000.0, 30_500.0),
    ] {
        let r = recs
            .iter()
            .find(|r| r.satellite_id.system == system)
            .expect("a record");
        let state = satellite_state(
            &r.elements,
            &r.clock,
            &r.constants(),
            r.elements.toe_sow,
            r.broadcast_clock_group_delay_s(),
            false,
        )
        .expect("valid parsed broadcast record");
        let p = state.orbit.position().expect("valid orbit position");
        let radius_km = (p.x_m * p.x_m + p.y_m * p.y_m + p.z_m * p.z_m).sqrt() / 1000.0;
        assert!(
            (lo_km..hi_km).contains(&radius_km),
            "{system:?} radius {radius_km} km out of band"
        );
    }
}

#[test]
fn spp_solves_from_broadcast_gps() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, ELEVATION_MASK_RAD,
    };

    // GPS-only store (avoids any Galileo I/NAV vs F/NAV selection ambiguity).
    let store = BroadcastStore::new(
        records()
            .into_iter()
            .filter(|r| r.satellite_id.system == GnssSystem::Gps)
            .collect(),
    )
    .expect("valid manual GPS broadcast store");

    // 2020-06-25 12:00 GPST (DOY 177 noon), as a J2000 second; mid-day so every
    // GPS satellite has a near-toe record.
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.0;
    // A true receiver on the ground near the ESBC station (Esbjerg, Denmark).
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0, 0.0];
    let corr = Corrections::NONE;
    let kl = KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    // Synthesize one pseudorange per visible GPS satellite with the same forward
    // model the solver inverts, so the true state is the zero-residual solution.
    let mut sats: Vec<_> = store.records().iter().map(|r| r.satellite_id).collect();
    sats.sort_unstable();
    sats.dedup();
    let mut observations = Vec::new();
    for sat in sats {
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: corr,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) = test_support::sat_model_for_test(
            &env,
            sat,
            [x_true[0], x_true[1], x_true[2]],
            x_true[3],
            22_000_000.0,
            &kl,
        ) {
            if m.el_rad >= ELEVATION_MASK_RAD {
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
            }
        }
    }
    assert!(
        observations.len() >= 4,
        "need >=4 visible GPS sats, got {}",
        observations.len()
    );

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: corr,
        klobuchar: kl,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    let sol = solve(&store, &inputs, true).expect("broadcast SPP solve");
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(err < 1.0e-3, "recovered position off by {err} m");
}

#[test]
fn rejects_a_non_navigation_header() {
    let bogus = "     3.05           OBSERVATION DATA   M                   RINEX VERSION / TYPE\n\
                 END OF HEADER\n";
    assert!(matches!(
        parse_nav(bogus),
        Err(NavParseError::UnsupportedHeader(_))
    ));
}

#[test]
fn reports_missing_header_end() {
    let truncated =
        "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n";
    assert_eq!(parse_nav(truncated), Err(NavParseError::MissingHeaderEnd));
}

#[test]
fn parse_glonass_rejects_a_non_navigation_header() {
    let bogus = "     3.05           OBSERVATION DATA   M                   RINEX VERSION / TYPE\n\
                 END OF HEADER\n";
    assert!(matches!(
        parse_glonass(bogus),
        Err(NavParseError::UnsupportedHeader(_))
    ));
}

#[test]
fn parse_glonass_reports_missing_header_end() {
    let truncated =
        "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n";
    assert_eq!(
        parse_glonass(truncated),
        Err(NavParseError::MissingHeaderEnd)
    );
}

// An exact GPS LNAV record block copied from the committed fixture (the v3 parser
// is already proven on this data), reused to build inline v3 and v4 inputs so the
// v4 path can be cross-checked against the v3 result. Continuation lines keep
// their fixed-column leading spaces.
const G01_LINES: &[&str] = &[
    "G01 2020 06 25 04 00 00 1.604342833161e-05 7.048583938740e-12 0.000000000000e+00",
    "     5.800000000000e+01-3.968750000000e+01 4.304822170265e-09 6.342094507864e-01",
    "    -2.177432179451e-06 1.000394229777e-02 1.937150955200e-06 5.153707128525e+03",
    "     3.600000000000e+05-1.508742570877e-07 2.572838528869e+00 1.359730958939e-07",
    "     9.806518601091e-01 3.539687500000e+02 7.941703015008e-01-8.384634967987e-09",
    "    -5.714523747137e-11 1.000000000000e+00 2.111000000000e+03 0.000000000000e+00",
    "     2.000000000000e+00 0.000000000000e+00 5.122274160385e-09 5.800000000000e+01",
    "     3.561060000000e+05 4.000000000000e+00",
];

// An exact Galileo record block whose data-source word (orbit-5 field 2 = 258,
// source bit 1 set) infers F/NAV under the v3 rule, used to show the v4 marker
// token is authoritative over that inference.
const E01_LINES: &[&str] = &[
    "E01 2020 06 24 23 30 00-8.846927667037e-04-7.972289495228e-12 0.000000000000e+00",
    "     6.100000000000e+01 1.865625000000e+01 2.656539226950e-09-1.832282909549e+00",
    "     8.568167686462e-07 9.650341235101e-05 1.049041748047e-05 5.440602037430e+03",
    "     3.438000000000e+05 1.862645149231e-09 2.123282284601e-01-1.452863216400e-07",
    "     9.828296477370e-01 1.298750000000e+02-2.778709093141e+00-5.216288707934e-09",
    "    -6.996720012901e-10 2.580000000000e+02 2.111000000000e+03",
    "     3.120000000000e+00 0.000000000000e+00-1.862645149231e-09 0.000000000000e+00",
    "     3.445400000000e+05",
];

const R01_GLONASS_LINES: &[&str] = &[
    "R01 2020 06 24 23 15 00 6.355904042721e-05 0.000000000000e+00 3.420000000000e+05",
    "     1.090894238281e+04 1.407806396484e+00-1.862645149231e-09 0.000000000000e+00",
    "    -2.885726074219e+03 2.795855522156e+00-0.000000000000e+00 1.000000000000e+00",
    "     2.288353955078e+04-3.169984817505e-01-2.793967723846e-09 0.000000000000e+00",
];

const V4_NAV_HEADER: &str =
    "     4.00           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n\
     XXX                                                         END OF HEADER\n";

const V3_NAV_HEADER: &str =
    "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n\
     XXX                                                         END OF HEADER\n";

fn join(lines: &[&str]) -> String {
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

fn gps_nav_text_with_epoch_field(start: usize, end: usize, value: &str) -> String {
    let mut lines: Vec<String> = G01_LINES.iter().map(ToString::to_string).collect();
    lines[0].replace_range(start..end, value);

    let mut text = String::from(V3_NAV_HEADER);
    for line in lines {
        text.push_str(&line);
        text.push('\n');
    }
    text
}

fn gps_nav_text_with_month(month: &str) -> String {
    gps_nav_text_with_epoch_field(9, 11, month)
}

fn glonass_text(lines: &[String]) -> String {
    let mut text = String::from(V3_NAV_HEADER);
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    text
}

fn r01_glonass_lines() -> Vec<String> {
    R01_GLONASS_LINES.iter().map(ToString::to_string).collect()
}

fn nav_text_with_version(version: &str, lines: &[String]) -> String {
    let mut text = format!(
        "{version:>9}           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n\
     XXX                                                         END OF HEADER\n"
    );
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    text
}

fn nav_text(lines: &[String]) -> String {
    nav_text_with_version("3.05", lines)
}

fn g01_lines() -> Vec<String> {
    G01_LINES.iter().map(ToString::to_string).collect()
}

fn e01_lines() -> Vec<String> {
    E01_LINES.iter().map(ToString::to_string).collect()
}

fn satellite_lines(template: &[&str], token: &str) -> Vec<String> {
    assert_eq!(token.len(), 3);
    let mut lines: Vec<String> = template.iter().map(ToString::to_string).collect();
    lines[0].replace_range(0..3, token);
    lines
}

fn replace_orbit_field(line: &str, field_index: usize, value: &str) -> String {
    let ranges = [(4, 23), (23, 42), (42, 61), (61, 80)];
    let (start, end) = ranges[field_index];
    let field = format!("{value:>width$}", width = end - start);
    let mut out = format!("{line:<80}");
    out.replace_range(start..end, &field);
    out
}

fn replace_fourth_orbit_field(line: &str, value: &str) -> String {
    assert_eq!(line.len(), 80);
    let field = format!("{value:>19}");
    assert_eq!(field.len(), 19);
    let mut out = line.to_string();
    out.replace_range(61..80, &field);
    out
}

fn nav_text_with_header_line(header_line: &str) -> String {
    format!(
        "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n\
{header_line}\n\
     XXX                                                         END OF HEADER\n{}",
        join(G01_LINES)
    )
}

#[test]
fn parse_glonass_valid_nav_without_glonass_records_is_empty() {
    let recs = parse_glonass(&nav_text(&g01_lines())).expect("valid non-GLONASS NAV");
    assert!(recs.is_empty());
}

#[test]
fn glonass_missing_health_is_bad_field() {
    let mut lines = r01_glonass_lines();
    lines[1] = replace_fourth_orbit_field(&lines[1], "");

    let err = parse_glonass(&glonass_text(&lines))
        .expect_err("missing GLONASS health must not default to healthy");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "R01".to_string(),
            field: "health",
        }
    );
}

#[test]
fn glonass_bad_frequency_channel_is_bad_field() {
    let mut lines = r01_glonass_lines();
    lines[2] = replace_fourth_orbit_field(&lines[2], "not-a-number");

    let err = parse_glonass(&glonass_text(&lines))
        .expect_err("bad GLONASS frequency channel must not default to channel 0");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "R01".to_string(),
            field: "frequency channel",
        }
    );
}

#[test]
fn glonass_nonintegral_frequency_channel_is_bad_field() {
    let mut lines = r01_glonass_lines();
    lines[2] = replace_fourth_orbit_field(&lines[2], "1.5");

    let err = parse_glonass(&glonass_text(&lines))
        .expect_err("fractional GLONASS frequency channel must be a bad field");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "R01".to_string(),
            field: "frequency channel",
        }
    );
}

#[test]
fn glonass_out_of_range_frequency_channel_is_bad_field() {
    for value in ["-8", "7"] {
        let mut lines = r01_glonass_lines();
        lines[2] = replace_fourth_orbit_field(&lines[2], value);

        let err = parse_glonass(&glonass_text(&lines))
            .expect_err("out-of-range GLONASS frequency channel must be a bad field");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: "R01".to_string(),
                field: "frequency channel",
            }
        );
    }
}

#[test]
fn glonass_integral_frequency_channel_parses() {
    let mut lines = r01_glonass_lines();
    lines[2] = replace_fourth_orbit_field(&lines[2], "-7");

    let recs = parse_glonass(&glonass_text(&lines)).expect("valid GLONASS frequency channel");
    assert_eq!(recs[0].freq_channel, -7);
}

#[test]
fn out_of_range_glonass_nav_slots_are_skipped_not_rejected() {
    // A slot the engine cannot represent (an extended GLONASS slot beyond the
    // PRN cap, e.g. R28 in real BKG/IGS products, or a nonsense R00/R99) must be
    // skipped, not reject the whole file. A real R01 record alongside it still
    // loads.
    for token in ["R00", "R28", "R99"] {
        let mut lines = satellite_lines(R01_GLONASS_LINES, token);
        lines.extend(r01_glonass_lines());

        let store = BroadcastStore::from_nav(&glonass_text(&lines)).unwrap_or_else(|err| {
            panic!("slot {token} must be skipped, not reject the file: {err}")
        });
        assert_eq!(
            store.glonass_records().len(),
            1,
            "only the representable R01 record is kept alongside {token}"
        );
        assert_eq!(
            store.glonass_records()[0].satellite_id,
            GnssSatelliteId::new(GnssSystem::Glonass, 1).expect("valid satellite id"),
            "kept record is R01 (with {token} skipped)"
        );
    }
}

#[test]
fn valid_edge_glonass_nav_prn_parses_into_broadcast_store() {
    let lines = satellite_lines(R01_GLONASS_LINES, "R27");

    let store = BroadcastStore::from_nav(&glonass_text(&lines)).expect("valid GLONASS PRN parses");
    assert_eq!(store.glonass_records().len(), 1);
    assert_eq!(
        store.glonass_records()[0].satellite_id,
        GnssSatelliteId::new(GnssSystem::Glonass, 27).expect("valid satellite id")
    );
}

#[test]
fn rejects_out_of_range_keplerian_nav_prns() {
    for (token, template) in [("G33", G01_LINES), ("E37", E01_LINES), ("C64", G01_LINES)] {
        let lines = satellite_lines(template, token);

        let err = parse_nav(&nav_text(&lines))
            .expect_err("out-of-range NAV satellite PRN must be rejected");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: token.to_string(),
                field: "prn",
            }
        );
    }
}

#[test]
fn valid_edge_keplerian_nav_prns_parse_into_broadcast_store() {
    let mut lines = satellite_lines(G01_LINES, "G32");

    let mut galileo = satellite_lines(E01_LINES, "E36");
    galileo[5] = replace_orbit_field(&galileo[5], 1, "5.120000000000e+02");
    lines.extend(galileo);
    lines.extend(satellite_lines(G01_LINES, "C63"));

    let store = BroadcastStore::from_nav(&nav_text(&lines)).expect("valid edge PRNs parse");
    let sats: Vec<_> = store
        .records()
        .iter()
        .map(|record| record.satellite_id)
        .collect();

    assert_eq!(sats.len(), 3);
    assert!(sats.contains(&GnssSatelliteId::new(GnssSystem::Gps, 32).expect("valid satellite id")));
    assert!(
        sats.contains(&GnssSatelliteId::new(GnssSystem::Galileo, 36).expect("valid satellite id"))
    );
    assert!(
        sats.contains(&GnssSatelliteId::new(GnssSystem::BeiDou, 63).expect("valid satellite id"))
    );
}

#[test]
fn rejects_nonfinite_orbital_field() {
    let mut lines = g01_lines();
    lines[2] = replace_orbit_field(&lines[2], 1, "NaN");

    let err = parse_nav(&nav_text(&lines)).expect_err("NaN eccentricity must be a bad field");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "G01".to_string(),
            field: "e",
        }
    );
}

#[test]
fn rejects_nonintegral_nonfinite_or_oversized_week_field() {
    for value in ["2.111500000000e+03", "NaN", "4.294967296000e+09"] {
        let mut lines = g01_lines();
        lines[5] = replace_orbit_field(&lines[5], 2, value);

        let err =
            parse_nav(&nav_text(&lines)).expect_err("invalid broadcast week must be a bad field");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: "G01".to_string(),
                field: "week",
            }
        );
    }
}

#[test]
fn rejects_malformed_galileo_data_source_word() {
    for value in ["-1.000000000000e+00", "", "not-a-number"] {
        let mut lines = e01_lines();
        lines[5] = replace_orbit_field(&lines[5], 1, value);

        let err = parse_nav(&nav_text(&lines))
            .expect_err("malformed Galileo data-source word must not cast to u32");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: "E01".to_string(),
                field: "data sources",
            }
        );
    }
}

#[test]
fn from_nav_rejects_malformed_header_ionosphere_coefficients() {
    let text = nav_text_with_header_line(
        "GPSA not-a-float                                             IONOSPHERIC CORR",
    );

    let err = match BroadcastStore::from_nav(&text) {
        Ok(_) => panic!("malformed IONOSPHERIC CORR field must be an error"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        NavParseError::BadHeaderField {
            field: "ionospheric correction",
        }
    );
}

#[test]
fn public_helper_rejects_malformed_header_ionosphere_coefficients() {
    let text = nav_text_with_header_line(
        "GPSA not-a-float                                             IONOSPHERIC CORR",
    );

    let err = parse_iono_corrections(&text)
        .expect_err("malformed IONOSPHERIC CORR field must be an error");
    assert_eq!(
        err,
        NavParseError::BadHeaderField {
            field: "ionospheric correction",
        }
    );
}

#[test]
fn from_nav_rejects_malformed_leap_seconds() {
    let text = nav_text_with_header_line(
        "bad                                                        LEAP SECONDS",
    );

    let err = match BroadcastStore::from_nav(&text) {
        Ok(_) => panic!("malformed LEAP SECONDS field must be an error"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        NavParseError::BadHeaderField {
            field: "leap seconds",
        }
    );
}

#[test]
fn public_helper_rejects_malformed_leap_seconds() {
    let text = nav_text_with_header_line(
        "bad                                                        LEAP SECONDS",
    );

    let err = parse_leap_seconds(&text).expect_err("malformed LEAP SECONDS field must be an error");
    assert_eq!(
        err,
        NavParseError::BadHeaderField {
            field: "leap seconds",
        }
    );
}

#[test]
fn parses_rinex_v4_eph_frames_and_skips_the_rest() {
    // One GPS LNAV EPH frame, plus a CNAV EPH frame and an STO frame that must be
    // skipped (CNAV reorders the orbit columns; STO carries no ephemeris).
    let mut text = String::from(V4_NAV_HEADER);
    text.push_str("> EPH G01 LNAV\n");
    text.push_str(&join(G01_LINES));
    text.push_str("> EPH G03 CNAV\n");
    text.push_str(&join(G01_LINES)); // body content is irrelevant; it must be skipped, not parsed
    text.push_str("> STO G01 LNAV\n");
    text.push_str("    2020 06 25 00 00 00 GPUT 0.0 0.0 0 0\n");

    let recs = parse_nav(&text).expect("parse v4 NAV");
    assert_eq!(recs.len(), 1, "only the LNAV EPH frame is parsed");
    assert_eq!(
        recs[0].satellite_id,
        GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id")
    );
    assert_eq!(recs[0].message, NavMessage::GpsLnav);

    // The v4 record must equal the same block parsed as v3, field for field.
    let v3_text = format!("{V3_NAV_HEADER}{}", join(G01_LINES));
    let v3 = parse_nav(&v3_text).expect("parse v3 NAV");
    assert_eq!(v3.len(), 1);
    assert_eq!(recs[0].elements, v3[0].elements, "elements differ v4 vs v3");
    assert_eq!(recs[0].clock, v3[0].clock, "clock differs v4 vs v3");
    assert_eq!(recs[0].week, v3[0].week);
    assert_eq!(recs[0].fit_interval_s, v3[0].fit_interval_s);
}

#[test]
fn rinex_v4_empty_eph_frame_is_truncated_record() {
    let text = format!("{V4_NAV_HEADER}> EPH G01 LNAV\n");

    assert!(matches!(
        parse_nav(&text),
        Err(NavParseError::TruncatedRecord(_))
    ));
}

#[test]
fn rinex_v4_rejects_marker_body_satellite_mismatch() {
    let text = format!("{V4_NAV_HEADER}> EPH E01 INAV\n{}", join(G01_LINES));

    let err = parse_nav(&text).expect_err("marker SV must match body SV");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "E01".to_string(),
            field: "frame marker",
        }
    );
}

#[test]
fn rinex_v4_rejects_marker_message_for_body_system_mismatch() {
    let text = format!("{V4_NAV_HEADER}> EPH E01 D1\n{}", join(E01_LINES));

    let err = parse_nav(&text).expect_err("marker message must match body constellation");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "E01".to_string(),
            field: "message",
        }
    );
}

#[test]
fn rejects_out_of_range_toc_epoch_month() {
    for month in ["00", "13"] {
        let err = parse_nav(&gps_nav_text_with_month(month))
            .expect_err("out-of-range TOC epoch month must be a parse error");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: "G01".to_string(),
                field: "toc epoch",
            }
        );
    }
}

#[test]
fn rejects_out_of_range_toc_epoch_date_time() {
    for (start, end, value) in [(12, 14, "31"), (15, 17, "24"), (21, 23, "60")] {
        let err = parse_nav(&gps_nav_text_with_epoch_field(start, end, value))
            .expect_err("out-of-range TOC epoch field must be a parse error");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: "G01".to_string(),
                field: "toc epoch",
            }
        );
    }
}

#[test]
fn rejects_out_of_range_glonass_utc_epoch() {
    let mut lines = r01_glonass_lines();
    lines[0].replace_range(12..14, "31");

    let err = parse_glonass(&glonass_text(&lines))
        .expect_err("out-of-range GLONASS UTC epoch must be a parse error");
    assert_eq!(
        err,
        NavParseError::BadField {
            satellite: "R01".to_string(),
            field: "epoch",
        }
    );
}

#[test]
fn glonass_utc_epoch_accepts_leap_second_label() {
    let mut lines = r01_glonass_lines();
    lines[0].replace_range(4..8, "2016");
    lines[0].replace_range(9..11, "12");
    lines[0].replace_range(12..14, "31");
    lines[0].replace_range(15..17, "23");
    lines[0].replace_range(18..20, "59");
    lines[0].replace_range(21..23, "60");

    let recs = parse_glonass(&glonass_text(&lines)).expect("GLONASS leap-second epoch");
    assert_eq!(recs.len(), 1);
    assert_eq!(
        recs[0].toe_utc_j2000_s,
        j2000_seconds_utc(2016, 12, 31, 23, 59, 60)
    );
}

#[test]
fn glonass_utc_epoch_rejects_invalid_leap_second_range() {
    for second in ["61", "-1"] {
        let mut lines = r01_glonass_lines();
        lines[0].replace_range(21..23, second);
        let err = parse_glonass(&glonass_text(&lines))
            .expect_err("invalid GLONASS UTC seconds must be a parse error");
        assert_eq!(
            err,
            NavParseError::BadField {
                satellite: "R01".to_string(),
                field: "epoch",
            }
        );
    }
}

#[test]
fn rinex_v4_message_type_comes_from_the_marker() {
    // The v3 data-source-word rule infers F/NAV for this Galileo block...
    let v3_text = format!("{V3_NAV_HEADER}{}", join(E01_LINES));
    let v3 = parse_nav(&v3_text).expect("parse v3 NAV");
    assert_eq!(
        v3[0].message,
        NavMessage::GalileoFnav,
        "v3 infers F/NAV here"
    );
    assert_eq!(
        v3[0].broadcast_clock_group_delay_s(),
        -1.862645149231e-09,
        "F/NAV uses Galileo BGD E5a/E1"
    );

    // ...but a v4 marker that says INAV is authoritative.
    let v4_text = format!("{V4_NAV_HEADER}> EPH E01 INAV\n{}", join(E01_LINES));
    let v4 = parse_nav(&v4_text).expect("parse v4 NAV");
    assert_eq!(v4.len(), 1);
    assert_eq!(
        v4[0].message,
        NavMessage::GalileoInav,
        "v4 message must come from the marker token, not the data-source word"
    );
    assert_eq!(
        v4[0].broadcast_clock_group_delay_s(),
        0.0,
        "v4 INAV marker uses Galileo BGD E5b/E1"
    );
}

#[test]
fn broadcast_reference_times_are_scale_tagged_by_constellation() {
    let recs = records();

    for (system, expected_scale) in [
        (GnssSystem::Gps, TimeScale::Gpst),
        (GnssSystem::Galileo, TimeScale::Gst),
        (GnssSystem::BeiDou, TimeScale::Bdt),
    ] {
        let rec = recs
            .iter()
            .find(|rec| rec.satellite_id.system == system)
            .unwrap_or_else(|| panic!("fixture should contain {system:?} records"));

        assert_eq!(rec.time_scale(), expected_scale, "{system:?} record scale");
        assert_eq!(rec.toe.system, expected_scale, "{system:?} toe scale");
        assert_eq!(rec.toc.system, expected_scale, "{system:?} toc scale");
        assert_eq!(rec.toe.week, rec.week, "{system:?} toe week");
        assert_eq!(rec.toc.week, rec.week, "{system:?} toc week");
        assert_eq!(rec.toe.tow_s.to_bits(), rec.elements.toe_sow.to_bits());
        assert_eq!(rec.toc.tow_s.to_bits(), rec.clock.toc_sow.to_bits());
    }
}

#[test]
fn toc_week_comes_from_clock_epoch_across_rollover() {
    let mut lines = g01_lines();
    lines[0].replace_range(4..23, "2020 06 28 00 00 00");
    lines[3].replace_range(4..23, " 6.045000000000e+05");

    let recs = parse_nav(&nav_text(&lines)).expect("parse week-rollover NAV record");
    let rec = &recs[0];

    assert_eq!(rec.week, 2111, "broadcast toe week remains from ORBIT-5");
    assert_eq!(rec.toe.week, 2111, "toe uses broadcast week");
    assert_eq!(rec.toe.tow_s.to_bits(), 604_500.0_f64.to_bits());
    assert_eq!(
        rec.toc.week, 2112,
        "toc week must be derived from the clock epoch line"
    );
    assert_eq!(rec.toc.tow_s.to_bits(), 0.0_f64.to_bits());
    assert_eq!(rec.clock.toc_sow.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn accepts_v4_nav_header_rejects_v4_non_nav() {
    // A 4.00 NAV header with one frame parses.
    let ok = format!("{V4_NAV_HEADER}> EPH G01 LNAV\n{}", join(G01_LINES));
    assert_eq!(parse_nav(&ok).expect("v4 NAV header accepted").len(), 1);

    // A 4.00 header that is not a navigation file (column 20 != 'N') is rejected.
    let bogus = "     4.00           OBSERVATION DATA   M                   RINEX VERSION / TYPE\n\
                 END OF HEADER\n";
    assert!(matches!(
        parse_nav(bogus),
        Err(NavParseError::UnsupportedHeader(_))
    ));
}

#[test]
fn from_nav_keeps_only_healthy_supported_messages() {
    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    let recs = store.records();
    assert!(!recs.is_empty());
    // Every kept record is healthy and a supported single-frequency message:
    // GPS LNAV, Galileo I/NAV, or BeiDou D1/D2. Galileo F/NAV and unhealthy
    // satellites are dropped.
    assert!(
        recs.iter().all(|r| r.sv_health == 0.0),
        "unhealthy record kept"
    );
    assert!(
        recs.iter().all(|r| matches!(
            r.message,
            NavMessage::GpsLnav
                | NavMessage::GalileoInav
                | NavMessage::BeidouD1
                | NavMessage::BeidouD2
        )),
        "an unsupported message type was kept"
    );
    for sys in [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou] {
        assert!(
            recs.iter().any(|r| r.satellite_id.system == sys),
            "no {sys:?} records kept"
        );
    }
    assert!(
        recs.iter().all(|r| r.message != NavMessage::GalileoFnav),
        "Galileo F/NAV must be excluded"
    );
    // The fixture's BeiDou set includes the geostationary C05 (a D2 message).
    assert!(
        recs.iter().any(|r| r.satellite_id
            == GnssSatelliteId::new(GnssSystem::BeiDou, 5).expect("valid satellite id")
            && r.message == NavMessage::BeidouD2),
        "expected the geostationary C05 (D2) record"
    );
}

#[test]
fn a_wrong_week_epoch_has_no_ephemeris() {
    use crate::spp::EphemerisSource;
    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    let sat = store.records()[0].satellite_id;

    // 2020-06-25 12:00 GPST as a J2000 second: a usable epoch for this product.
    let t_ok = 646_358_400.0_f64;
    assert!(
        store.position_clock_at_j2000_s(sat, t_ok).is_some(),
        "expected ephemeris at a valid epoch"
    );

    // The same wall-clock one week earlier: the nearest record is a week stale,
    // so the store must report no ephemeris rather than extrapolating a wrong
    // week's elements.
    let t_wrong_week = t_ok - 604_800.0;
    assert!(
        store.position_clock_at_j2000_s(sat, t_wrong_week).is_none(),
        "a wrong-week epoch must not silently produce an ephemeris"
    );
}

/// The J2000 second at which a record's reference epoch (`toe`) occurs, given the
/// satellite's timescale. BeiDou runs on BDT (= GPST - 14 s) with its week epoch
/// 1356 weeks after the GPS epoch; GPS/Galileo are GPST-aligned.
fn toe_as_j2000_s(rec: &BroadcastRecord) -> f64 {
    let toe_continuous = f64::from(rec.week) * 604_800.0 + rec.elements.toe_sow;
    let gps_epoch_to_j2000 = 630_763_200.0;
    if rec.satellite_id.system == GnssSystem::BeiDou {
        toe_continuous + 14.0 + 1356.0 * 604_800.0 - gps_epoch_to_j2000
    } else {
        toe_continuous - gps_epoch_to_j2000
    }
}

#[test]
fn broadcast_store_evaluates_beidou_including_geo() {
    use crate::spp::EphemerisSource;

    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    // The geostationary C05 and a MEO (C19+) BeiDou satellite, evaluated at each
    // one's own reference epoch through the store's BDT timescale mapping.
    let geo = GnssSatelliteId::new(GnssSystem::BeiDou, 5).expect("valid satellite id");
    let meo = store
        .records()
        .iter()
        .map(|r| r.satellite_id)
        .find(|s| s.system == GnssSystem::BeiDou && s.prn >= 19)
        .expect("a BeiDou MEO satellite");

    for (sat, lo_km, hi_km) in [(geo, 41_000.0, 43_000.0), (meo, 27_000.0, 29_000.0)] {
        let rec = store
            .records()
            .iter()
            .find(|r| r.satellite_id == sat)
            .unwrap();
        let t = toe_as_j2000_s(rec);
        let (pos, _clk) = store
            .position_clock_at_j2000_s(sat, t)
            .unwrap_or_else(|| panic!("{sat:?} should evaluate at its toe"));
        let radius_km = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt() / 1000.0;
        assert!(
            (lo_km..hi_km).contains(&radius_km),
            "{sat:?} radius {radius_km} km out of band"
        );
    }
    // The geostationary satellite sits near the equatorial plane.
    let c05 = store
        .records()
        .iter()
        .find(|r| r.satellite_id == geo)
        .unwrap();
    let (geo_pos, _) = store
        .position_clock_at_j2000_s(geo, toe_as_j2000_s(c05))
        .unwrap();
    let radius = (geo_pos[0].powi(2) + geo_pos[1].powi(2) + geo_pos[2].powi(2)).sqrt();
    assert!(
        geo_pos[2].abs() / radius < 0.2,
        "GEO should be near-equatorial"
    );
}

#[test]
fn broadcast_store_rejects_unsupported_systems() {
    use crate::broadcast::{ClockPolynomial, KeplerianElements};
    use crate::spp::EphemerisSource;

    // `BroadcastStore::new` accepts arbitrary records; a GLONASS satellite (a
    // non-Keplerian state-vector model) must report no ephemeris rather than be
    // evaluated with the wrong model.
    let sat = GnssSatelliteId::new(GnssSystem::Glonass, 1).expect("valid satellite id");
    let rec = BroadcastRecord {
        satellite_id: sat,
        message: NavMessage::GpsLnav,
        week: 2111,
        toe: broadcast_time(sat.system, 2111, 0.0),
        toc: broadcast_time(sat.system, 2111, 0.0),
        elements: KeplerianElements {
            sqrt_a: 5153.0,
            e: 0.001,
            m0: 0.0,
            delta_n: 0.0,
            omega0: 0.0,
            i0: 0.9,
            omega: 0.0,
            omega_dot: 0.0,
            idot: 0.0,
            cuc: 0.0,
            cus: 0.0,
            crc: 0.0,
            crs: 0.0,
            cic: 0.0,
            cis: 0.0,
            toe_sow: 0.0,
        },
        clock: ClockPolynomial {
            af0: 0.0,
            af1: 0.0,
            af2: 0.0,
            toc_sow: 0.0,
        },
        group_delays: BroadcastGroupDelays::default(),
        sv_health: 0.0,
        sv_accuracy_m: 0.0,
        fit_interval_s: None,
    };
    let store = BroadcastStore::new(vec![rec]).expect("valid unsupported-system manual store");
    assert!(
        store
            .position_clock_at_j2000_s(sat, 646_358_400.0)
            .is_none(),
        "an unsupported system must report no ephemeris"
    );
}

#[test]
fn broadcast_store_rejects_invalid_manual_ephemerides() {
    let mut rec = records()[0];

    rec.elements.sqrt_a = f64::NAN;
    let err = match BroadcastStore::new(vec![rec]) {
        Ok(_) => panic!("non-finite manual ephemeris must be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(err, crate::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );

    let mut rec = records()[0];
    rec.fit_interval_s = Some(f64::INFINITY);
    let err = match BroadcastStore::new(vec![rec]) {
        Ok(_) => panic!("non-finite fit interval must be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(err, crate::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );

    let mut rec = records()[0];
    rec.group_delays = BroadcastGroupDelays::gps_lnav(f64::NAN);
    let err = match BroadcastStore::new(vec![rec]) {
        Ok(_) => panic!("non-finite group delay must be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(err, crate::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn gps_fit_interval_bounds_record_validity() {
    use crate::broadcast::{ClockPolynomial, KeplerianElements};
    use crate::spp::EphemerisSource;

    // A minimal but evaluable record; only the system and fit interval matter to
    // selection (the orbit values just have to produce a finite state).
    let make = |system, fit_interval_s| BroadcastRecord {
        satellite_id: GnssSatelliteId::new(system, 1).expect("valid satellite id"),
        message: NavMessage::GpsLnav,
        week: 2111,
        toe: broadcast_time(system, 2111, 0.0),
        toc: broadcast_time(system, 2111, 0.0),
        elements: KeplerianElements {
            sqrt_a: 5153.0,
            e: 0.001,
            m0: 0.0,
            delta_n: 0.0,
            omega0: 0.0,
            i0: 0.9,
            omega: 0.0,
            omega_dot: 0.0,
            idot: 0.0,
            cuc: 0.0,
            cus: 0.0,
            crc: 0.0,
            crs: 0.0,
            cic: 0.0,
            cis: 0.0,
            toe_sow: 0.0,
        },
        clock: ClockPolynomial {
            af0: 0.0,
            af1: 0.0,
            af2: 0.0,
            toc_sow: 0.0,
        },
        group_delays: BroadcastGroupDelays::default(),
        sv_health: 0.0,
        sv_accuracy_m: 0.0,
        fit_interval_s,
    };

    // `toe` at GPS week 2111, second-of-week 0, expressed as a J2000 second.
    let toe_j2000 = 2111.0 * 604_800.0 - 630_763_200.0;

    // GPS with a four-hour fit interval is valid within +/-2 h of toe only.
    let g = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let gps = BroadcastStore::new(vec![make(GnssSystem::Gps, Some(4.0 * 3600.0))])
        .expect("valid manual GPS fit store");
    assert!(
        gps.position_clock_at_j2000_s(g, toe_j2000 + 3600.0)
            .is_some(),
        "1 h after toe is inside the 4 h fit interval"
    );
    assert!(
        gps.position_clock_at_j2000_s(g, toe_j2000 + 3.0 * 3600.0)
            .is_none(),
        "3 h after toe is outside the 4 h fit interval"
    );

    // A record with no broadcast fit interval (Galileo/BeiDou) falls back to the
    // coarse age bound, so the same 3 h offset is still accepted.
    let e = GnssSatelliteId::new(GnssSystem::Galileo, 1).expect("valid satellite id");
    let gal = BroadcastStore::new(vec![make(GnssSystem::Galileo, None)])
        .expect("valid manual Galileo fit store");
    assert!(
        gal.position_clock_at_j2000_s(e, toe_j2000 + 3.0 * 3600.0)
            .is_some(),
        "without a fit interval the coarse 4 h bound applies"
    );
}

#[test]
fn select_prefers_a_valid_farther_record_over_an_expired_nearer_one() {
    use crate::broadcast::{ClockPolynomial, KeplerianElements};
    use crate::spp::EphemerisSource;

    let elements = |toe_sow| KeplerianElements {
        sqrt_a: 5153.0,
        e: 0.001,
        m0: 0.0,
        delta_n: 0.0,
        omega0: 0.0,
        i0: 0.9,
        omega: 0.0,
        omega_dot: 0.0,
        idot: 0.0,
        cuc: 0.0,
        cus: 0.0,
        crc: 0.0,
        crs: 0.0,
        cic: 0.0,
        cis: 0.0,
        toe_sow,
    };
    let rec = |toe_sow, fit_interval_s| BroadcastRecord {
        satellite_id: GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id"),
        message: NavMessage::GpsLnav,
        week: 2111,
        toe: broadcast_time(GnssSystem::Gps, 2111, toe_sow),
        toc: broadcast_time(GnssSystem::Gps, 2111, toe_sow),
        elements: elements(toe_sow),
        clock: ClockPolynomial {
            af0: 0.0,
            af1: 0.0,
            af2: 0.0,
            toc_sow: toe_sow,
        },
        group_delays: BroadcastGroupDelays::default(),
        sv_health: 0.0,
        sv_accuracy_m: 0.0,
        fit_interval_s,
    };

    // Two records for one satellite: a nearer one with the nominal 4 h fit
    // (valid +/-2 h) and a farther one (toe 3 h earlier) with an extended 26 h
    // fit (valid +/-13 h).
    let near = rec(10_800.0, Some(4.0 * 3600.0));
    let far = rec(0.0, Some(26.0 * 3600.0));
    let store = BroadcastStore::new(vec![near, far]).expect("valid manual nearest-store records");

    let g = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    // 3 h after the nearer record's toe: outside its +/-2 h window, but inside
    // the farther record's +/-13 h window. Selecting nearest-then-checking would
    // wrongly reject this; filtering by validity first serves it from the farther
    // record.
    let near_toe_j2000 = 2111.0 * 604_800.0 + 10_800.0 - 630_763_200.0;
    let q = near_toe_j2000 + 3.0 * 3600.0;
    assert!(
        store.position_clock_at_j2000_s(g, q).is_some(),
        "a query past the nearest record's fit interval must fall back to a \
         farther record whose own window still covers it"
    );
}

#[test]
fn rinex_302_gps_fit_interval_flag_one_keeps_extended_validity() {
    use crate::spp::EphemerisSource;

    let mut lines = g01_lines();
    lines[7] = replace_orbit_field(&lines[7], 1, "1.000000000000e+00");
    let text = nav_text_with_version("3.02", &lines);

    let recs = parse_nav(&text).expect("parse RINEX 3.02 GPS record");
    assert_eq!(recs.len(), 1);
    let rec = recs[0];
    let fit = rec.fit_interval_s.expect("GPS fit interval");
    assert!(
        fit > GPS_NOMINAL_FIT_INTERVAL_S,
        "legacy flag 1 must decode as more than four hours, got {fit}"
    );

    let sat = rec.satellite_id;
    let query = toe_as_j2000_s(&rec) + 2.5 * 3600.0;
    let store = BroadcastStore::new(recs).expect("valid manual fit-boundary records");
    assert!(
        store.position_clock_at_j2000_s(sat, query).is_some(),
        "legacy flag 1 must not collapse the fit window to +/-30 minutes"
    );
}

#[test]
fn modern_gps_fit_interval_field_remains_hours_valued() {
    use crate::spp::EphemerisSource;

    let mut lines = g01_lines();
    lines[7] = replace_orbit_field(&lines[7], 1, "6.000000000000e+00");
    let text = nav_text_with_version("3.05", &lines);

    let recs = parse_nav(&text).expect("parse modern GPS record");
    assert_eq!(recs.len(), 1);
    let rec = recs[0];
    assert_eq!(
        rec.fit_interval_s,
        Some(6.0 * 3600.0),
        "modern fit interval is hours"
    );

    let sat = rec.satellite_id;
    let query = toe_as_j2000_s(&rec) + 2.5 * 3600.0;
    let store = BroadcastStore::new(recs).expect("valid manual fallback-boundary records");
    assert!(
        store.position_clock_at_j2000_s(sat, query).is_some(),
        "a 6 h modern fit interval is valid +/-3 h from toe"
    );
}

#[test]
fn gps_fit_interval_field_distinguishes_blank_zero_value_and_malformed() {
    // Place a value in ORBIT-7 field 2 (columns 23..42): 23 leading blanks then
    // the field, so `field(line, 23, 42)` reads exactly the value.
    let with_field2 = |val: &str| format!("{:23}{:<19}", "", val);
    let legacy = RinexVersion { major: 3, minor: 2 };
    let modern = RinexVersion { major: 3, minor: 5 };

    // Blank/absent -> the nominal four hours.
    assert_eq!(
        gps_fit_interval_s(&with_field2(""), modern),
        Ok(GPS_NOMINAL_FIT_INTERVAL_S)
    );
    // Explicit zero -> the nominal four hours.
    assert_eq!(
        gps_fit_interval_s(&with_field2("0.000000000000e+00"), modern),
        Ok(GPS_NOMINAL_FIT_INTERVAL_S)
    );
    // Legacy RINEX may carry the broadcast fit flag: 1 means more than four
    // hours, not a one-hour fit interval.
    assert_eq!(
        gps_fit_interval_s(&with_field2("1.000000000000e+00"), legacy),
        Ok(GPS_LEGACY_EXTENDED_FIT_INTERVAL_S)
    );
    // Modern RINEX keeps the same numeric field hours-valued.
    assert_eq!(
        gps_fit_interval_s(&with_field2("1.000000000000e+00"), modern),
        Ok(3600.0)
    );
    // A nonzero interval is taken verbatim (hours -> seconds).
    assert_eq!(
        gps_fit_interval_s(&with_field2("6.000000000000e+00"), modern),
        Ok(6.0 * 3600.0)
    );
    // Present but non-numeric -> an error, not a silent nominal substitution.
    assert!(gps_fit_interval_s(&with_field2("garbage"), modern).is_err());
}

#[test]
fn mixed_constellation_solve_recovers_the_receiver() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, ELEVATION_MASK_RAD,
    };

    // The default store carries both GPS LNAV and Galileo I/NAV (healthy).
    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.5;
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0];
    let corr = Corrections::NONE;
    let kl = KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    let mut sats: Vec<_> = store.records().iter().map(|r| r.satellite_id).collect();
    sats.sort_unstable();
    sats.dedup();

    let mut observations = Vec::new();
    let (mut have_gps, mut have_gal) = (false, false);
    for sat in sats {
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: corr,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) = test_support::sat_model_for_test(&env, sat, x_true, 0.0, 22_000_000.0, &kl)
        {
            if m.el_rad >= ELEVATION_MASK_RAD {
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
                have_gps |= sat.system == GnssSystem::Gps;
                have_gal |= sat.system == GnssSystem::Galileo;
            }
        }
    }
    assert!(
        have_gps && have_gal,
        "fixture must yield both GPS and Galileo observations"
    );

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: corr,
        klobuchar: kl,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    // The combined GPS+Galileo solve carries a per-system clock (a reference
    // clock plus the GPS/Galileo inter-system bias), so it recovers the receiver
    // from the mixed set. The geometry also yields a multi-system DOP.
    let sol = solve(&store, &inputs, true).expect("mixed-constellation solve");
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(
        err < 1.0e-3,
        "mixed solve recovered position off by {err} m"
    );

    let used_gps = sol.used_sats.iter().any(|s| s.system == GnssSystem::Gps);
    let used_gal = sol
        .used_sats
        .iter()
        .any(|s| s.system == GnssSystem::Galileo);
    assert!(
        used_gps && used_gal,
        "the solve must use both constellations"
    );
    let dop = sol
        .dop
        .expect("multi-system DOP present for the mixed solve");
    for (v, name) in [
        (dop.gdop, "GDOP"),
        (dop.pdop, "PDOP"),
        (dop.hdop, "HDOP"),
        (dop.vdop, "VDOP"),
        (dop.tdop, "TDOP"),
    ] {
        assert!(
            v.is_finite() && v > 0.0,
            "multi-system {name} not finite/positive: {v}"
        );
    }

    // Per-constellation TDOP: one entry per GNSS, in the same order as the
    // per-system clocks, with the reference (first) entry equal to the scalar
    // TDOP. This pins the system<->clock-column mapping at the solution level.
    assert_eq!(
        sol.system_tdops.len(),
        sol.system_clocks_s.len(),
        "one per-system TDOP per receiver clock"
    );
    assert!(
        sol.system_tdops.len() >= 2,
        "GPS+Galileo solve must carry at least two per-system TDOPs"
    );
    for ((sys_t, _), (sys_c, _)) in sol.system_tdops.iter().zip(sol.system_clocks_s.iter()) {
        assert_eq!(
            sys_t, sys_c,
            "per-system TDOP order must match the per-system clock order"
        );
    }
    assert_eq!(
        sol.system_tdops[0].1.to_bits(),
        dop.tdop.to_bits(),
        "reference-system TDOP must equal the scalar TDOP"
    );
    for (sys, v) in &sol.system_tdops {
        assert!(
            v.is_finite() && *v > 0.0,
            "per-system TDOP for {sys:?} not finite/positive: {v}"
        );
    }
}

#[test]
fn mixed_constellation_solve_recovers_a_nonzero_inter_system_bias() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, C_M_S, ELEVATION_MASK_RAD,
    };

    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.5;
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0];
    // The Galileo receiver clock leads the GPS one by a real inter-system bias.
    let gal_bias_m = 50.0_f64;
    let corr = Corrections::NONE;
    let kl = KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    let mut sats: Vec<_> = store.records().iter().map(|r| r.satellite_id).collect();
    sats.sort_unstable();
    sats.dedup();

    // Synthesize each pseudorange at the true position with the receiver clock
    // its own system sees: 0 for GPS (the reference), gal_bias_m for Galileo.
    let mut observations = Vec::new();
    let (mut have_gps, mut have_gal) = (false, false);
    for sat in sats {
        let b = if sat.system == GnssSystem::Galileo {
            gal_bias_m
        } else {
            0.0
        };
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: corr,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) = test_support::sat_model_for_test(&env, sat, x_true, b, 22_000_000.0, &kl) {
            if m.el_rad >= ELEVATION_MASK_RAD {
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
                have_gps |= sat.system == GnssSystem::Gps;
                have_gal |= sat.system == GnssSystem::Galileo;
            }
        }
    }
    assert!(
        have_gps && have_gal,
        "need both GPS and Galileo observations"
    );

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: corr,
        klobuchar: kl,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    let sol = solve(&store, &inputs, false).expect("mixed solve with inter-system bias");

    // Position is still recovered despite the inter-system bias.
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(err < 1.0e-3, "recovered position off by {err} m");

    // The per-system clocks are recovered: GPS ~ 0, Galileo ~ the injected bias.
    let clk = |sys| {
        sol.system_clocks_s
            .iter()
            .find(|(s, _)| *s == sys)
            .map(|(_, c)| *c * C_M_S)
            .unwrap_or_else(|| panic!("no {sys:?} clock"))
    };
    assert!(
        clk(GnssSystem::Gps).abs() < 1.0e-3,
        "GPS clock {} m",
        clk(GnssSystem::Gps)
    );
    assert!(
        (clk(GnssSystem::Galileo) - gal_bias_m).abs() < 1.0e-3,
        "Galileo clock {} m, expected ~{gal_bias_m}",
        clk(GnssSystem::Galileo)
    );
}

#[test]
fn mixed_solve_recovers_with_gps_galileo_and_beidou() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, C_M_S, ELEVATION_MASK_RAD,
    };

    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.5;
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0];
    let corr = Corrections::NONE;
    let kl = KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    // A distinct receiver-clock bias per system (GPS is the reference).
    let bias_m = |sys| match sys {
        GnssSystem::Galileo => 50.0,
        GnssSystem::BeiDou => 120.0,
        _ => 0.0,
    };

    let mut sats: Vec<_> = store.records().iter().map(|r| r.satellite_id).collect();
    sats.sort_unstable();
    sats.dedup();

    let mut observations = Vec::new();
    let (mut g, mut e, mut c) = (false, false, false);
    for sat in sats {
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: corr,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) = test_support::sat_model_for_test(
            &env,
            sat,
            x_true,
            bias_m(sat.system),
            22_000_000.0,
            &kl,
        ) {
            if m.el_rad >= ELEVATION_MASK_RAD {
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
                g |= sat.system == GnssSystem::Gps;
                e |= sat.system == GnssSystem::Galileo;
                c |= sat.system == GnssSystem::BeiDou;
            }
        }
    }
    assert!(
        g && e && c,
        "need GPS, Galileo, and BeiDou observations (got {g} {e} {c})"
    );

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: corr,
        klobuchar: kl,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    let sol = solve(&store, &inputs, false).expect("three-constellation solve");
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(err < 1.0e-3, "recovered position off by {err} m");

    let clk = |sys| {
        sol.system_clocks_s
            .iter()
            .find(|(s, _)| *s == sys)
            .map(|(_, v)| *v * C_M_S)
            .unwrap_or_else(|| panic!("no {sys:?} clock"))
    };
    assert!(
        clk(GnssSystem::Gps).abs() < 1.0e-3,
        "GPS clock {}",
        clk(GnssSystem::Gps)
    );
    assert!(
        (clk(GnssSystem::Galileo) - 50.0).abs() < 1.0e-3,
        "GAL clock {}",
        clk(GnssSystem::Galileo)
    );
    assert!(
        (clk(GnssSystem::BeiDou) - 120.0).abs() < 1.0e-3,
        "BDS clock {}",
        clk(GnssSystem::BeiDou)
    );
}

#[test]
fn ionosphere_correction_is_applied_to_beidou_b1i() {
    use crate::spp::{
        solve, test_support, Corrections, KlobucharCoeffs, Observation, SatModelEnv, SolveInputs,
        SppModelRecipe, SurfaceMet, ELEVATION_MASK_RAD,
    };

    let store = BroadcastStore::from_nav(&fixture_text()).expect("parse NAV");
    let t_rx = 646_358_400.0_f64;
    let sod = 12.0 * 3600.0;
    let doy = 177.5;
    let x_true = [3_512_900.0, 780_500.0, 5_248_700.0];
    // Ionosphere on. The broadcast Klobuchar L1 delay is scaled to each carrier
    // by (f_L1/f)^2 - exactly 1 for GPS L1 / Galileo E1, and scaled for BeiDou
    // B1I - so a BeiDou-bearing iono-corrected solve is now supported (not
    // rejected) and recovers the truth from observations synthesized with the
    // same frequency-aware model.
    let corr = Corrections::IONO;
    let kl = KlobucharCoeffs {
        alpha: [1.0e-8, 0.0, 0.0, 0.0],
        beta: [9.0e4, 0.0, 0.0, 0.0],
    };
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };

    let mut sats: Vec<_> = store.records().iter().map(|r| r.satellite_id).collect();
    sats.sort_unstable();
    sats.dedup();

    let mut observations = Vec::new();
    let mut saw_beidou = false;
    for sat in sats {
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env = SatModelEnv {
            eph: &store,
            t_rx_j2000_s: t_rx,
            t_rx_second_of_day_s: sod,
            day_of_year: doy,
            corrections: corr,
            met: &met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        if let Some(m) = test_support::sat_model_for_test(&env, sat, x_true, 0.0, 22_000_000.0, &kl)
        {
            if m.el_rad >= ELEVATION_MASK_RAD {
                saw_beidou |= sat.system == GnssSystem::BeiDou;
                observations.push(Observation {
                    satellite_id: sat,
                    pseudorange_m: m.p_hat_m,
                });
            }
        }
    }
    assert!(
        saw_beidou,
        "the iono-corrected set must include a BeiDou satellite"
    );

    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: sod,
        day_of_year: doy,
        initial_guess: [
            x_true[0] + 1000.0,
            x_true[1] - 1000.0,
            x_true[2] + 1000.0,
            0.0,
        ],
        corrections: corr,
        klobuchar: kl,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met,
        robust: None,
    };

    let sol = solve(&store, &inputs, false).expect("BeiDou-bearing iono-corrected solve");
    let p = sol.position;
    let err =
        ((p.x_m - x_true[0]).powi(2) + (p.y_m - x_true[1]).powi(2) + (p.z_m - x_true[2]).powi(2))
            .sqrt();
    assert!(err < 1.0e-3, "recovered position off by {err} m");
}

#[test]
fn galileo_broadcast_clock_group_delay_selects_e5b_for_inav_e5a_for_fnav() {
    // The broadcast-clock group delay must follow the message's reference
    // signal: I/NAV (E1/E5b) uses BGD E5b/E1, F/NAV (E5a) uses BGD E5a/E1. The
    // golden-recipe test reads a baked tgd, so this is the coverage that pins
    // the for_message accessor the bindings call on real records.
    let records = records();
    let e01_id = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
    let e01 = |message: NavMessage| {
        records
            .iter()
            .find(|r| r.satellite_id == e01_id && r.message == message)
            .unwrap_or_else(|| panic!("ESBC fixture carries an E01 {message:?} record"))
    };

    let inav = e01(NavMessage::GalileoInav);
    let fnav = e01(NavMessage::GalileoFnav);

    let inav_delay = inav.broadcast_clock_group_delay_s();
    let fnav_delay = fnav.broadcast_clock_group_delay_s();

    assert_eq!(
        inav_delay.to_bits(),
        inav.group_delays
            .galileo_bgd_e5b_e1_s
            .expect("I/NAV carries BGD E5b/E1")
            .to_bits(),
        "I/NAV broadcast clock must apply BGD E5b/E1"
    );
    assert_eq!(
        fnav_delay.to_bits(),
        fnav.group_delays
            .galileo_bgd_e5a_e1_s
            .expect("F/NAV carries BGD E5a/E1")
            .to_bits(),
        "F/NAV broadcast clock must apply BGD E5a/E1"
    );
    // The two messages reference different signals, so the delays must differ.
    assert_ne!(inav_delay.to_bits(), fnav_delay.to_bits());
}

// ---------------------------------------------------------------------------
// Decoded-LNAV -> BroadcastRecord glue (the `lnav::decode -> source` half of the
// real-time pipeline). Deterministic: it checks the unit conversions and a
// physical position-eval sanity, not a parser round-trip.
// ---------------------------------------------------------------------------

fn sample_lnav_decoded() -> crate::navigation::lnav::LnavDecoded {
    // Realistic healthy GPS ephemeris in transmitted units: the angular elements
    // (m0, delta_n, omega0, i0, omega, omega_dot, idot) are semicircles or
    // semicircles/second; the harmonic terms are radians and crc/crs meters.
    crate::navigation::lnav::LnavDecoded {
        // Must reduce to full_week % 1024; the tests unroll with full_week 2110
        // (2110 % 1024 = 62), so from_lnav's week-residue check passes.
        week_number: 62,
        l2_code: 1,
        ura_index: 0,
        sv_health: 0,
        iodc: 12,
        tgd: -5.0e-9,
        toc: 345_600,
        af0: 1.0e-4,
        af1: 1.0e-12,
        af2: 0.0,
        iode: 12,
        crs: 20.0,
        delta_n: 1.5e-9,
        m0: 0.3,
        cuc: 1.0e-6,
        eccentricity: 0.005,
        cus: 2.0e-6,
        sqrt_a: 5153.6,
        toe: 345_600,
        fit_interval_flag: 0,
        aodo: 0,
        cic: 1.0e-7,
        omega0: -0.8,
        cis: -1.0e-7,
        i0: 0.31,
        crc: 200.0,
        omega: 0.5,
        omega_dot: -2.5e-9,
        idot: 1.0e-10,
    }
}

#[test]
fn from_lnav_scales_semicircles_to_radians_and_passes_radian_terms_through() {
    let decoded = sample_lnav_decoded();
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid GPS id");
    let record = BroadcastRecord::from_lnav(&decoded, sat, 2110).expect("GPS LNAV record");

    let pi = core::f64::consts::PI;
    // Angular elements: semicircles -> radians (bit-exact multiply by PI).
    assert_eq!(record.elements.m0.to_bits(), (decoded.m0 * pi).to_bits());
    assert_eq!(
        record.elements.delta_n.to_bits(),
        (decoded.delta_n * pi).to_bits()
    );
    assert_eq!(
        record.elements.omega0.to_bits(),
        (decoded.omega0 * pi).to_bits()
    );
    assert_eq!(record.elements.i0.to_bits(), (decoded.i0 * pi).to_bits());
    assert_eq!(
        record.elements.omega.to_bits(),
        (decoded.omega * pi).to_bits()
    );
    assert_eq!(
        record.elements.omega_dot.to_bits(),
        (decoded.omega_dot * pi).to_bits()
    );
    assert_eq!(
        record.elements.idot.to_bits(),
        (decoded.idot * pi).to_bits()
    );

    // Radian/meter terms pass through unchanged.
    assert_eq!(record.elements.cuc.to_bits(), decoded.cuc.to_bits());
    assert_eq!(record.elements.cus.to_bits(), decoded.cus.to_bits());
    assert_eq!(record.elements.cic.to_bits(), decoded.cic.to_bits());
    assert_eq!(record.elements.cis.to_bits(), decoded.cis.to_bits());
    assert_eq!(record.elements.crc.to_bits(), decoded.crc.to_bits());
    assert_eq!(record.elements.crs.to_bits(), decoded.crs.to_bits());
    assert_eq!(record.elements.e.to_bits(), decoded.eccentricity.to_bits());
    assert_eq!(record.elements.sqrt_a.to_bits(), decoded.sqrt_a.to_bits());

    // Epoch, clock, and metadata.
    assert_eq!(record.elements.toe_sow, decoded.toe as f64);
    assert_eq!(record.clock.toc_sow, decoded.toc as f64);
    assert_eq!(record.clock.af0.to_bits(), decoded.af0.to_bits());
    assert_eq!(record.week, 2110);
    assert_eq!(record.toe.week, 2110);
    assert_eq!(record.message, NavMessage::GpsLnav);
    assert_eq!(record.sv_health, 0.0);
    assert_eq!(record.sv_accuracy_m, 2.4); // URA index 0 (IS-GPS-200N 20.3.3.3.1.3)
    assert_eq!(record.fit_interval_s, Some(4.0 * 3600.0));
    assert_eq!(
        record.group_delays.gps_tgd_s.map(f64::to_bits),
        Some(decoded.tgd.to_bits())
    );
}

#[test]
fn from_lnav_rejects_non_gps_satellite() {
    let decoded = sample_lnav_decoded();
    let sat = GnssSatelliteId::new(GnssSystem::Galileo, 7).expect("valid Galileo id");
    assert_eq!(
        BroadcastRecord::from_lnav(&decoded, sat, 2110),
        Err(LnavRecordError::NotGps(sat))
    );
}

#[test]
fn from_lnav_record_evaluates_to_a_physical_gps_position() {
    use crate::spp::EphemerisSource;

    let decoded = sample_lnav_decoded();
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid GPS id");
    let record = BroadcastRecord::from_lnav(&decoded, sat, 2110).expect("GPS LNAV record");
    let store = BroadcastStore::new(vec![record]).expect("store from decoded record");

    // Query at the record's reference epoch (tk = 0): GPS continuous time is
    // J2000 seconds plus the GPS-epoch offset, so invert that for the query.
    let t_j2000_s = 2110.0 * crate::constants::SECONDS_PER_WEEK + decoded.toe as f64
        - crate::constants::GPS_EPOCH_TO_J2000_S;
    let (pos, clock) = store
        .position_clock_at_j2000_s(sat, t_j2000_s)
        .expect("decoded record yields a position at its toe");

    let radius = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    assert!(
        (2.0e7..2.7e7).contains(&radius),
        "GPS orbital radius out of range: {radius} m"
    );
    assert!(clock.is_finite());
}

#[test]
fn from_lnav_rejects_week_residue_mismatch() {
    // Sample decodes 10-bit week 62; unrolling with a full_week whose low 10 bits
    // are not 62 means the wrong rollover epoch, which must be rejected.
    let decoded = sample_lnav_decoded();
    assert_eq!(decoded.week_number, 62);
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid GPS id");

    // 2109 % 1024 = 61 != 62.
    assert_eq!(
        BroadcastRecord::from_lnav(&decoded, sat, 2109),
        Err(LnavRecordError::WeekMismatch {
            full_week: 2109,
            decoded_week: 62,
        })
    );

    // A different rollover epoch with the SAME residue is accepted (62, 1086,
    // 2110 all reduce to 62).
    for full_week in [62u32, 1086, 2110, 3134] {
        let record = BroadcastRecord::from_lnav(&decoded, sat, full_week)
            .expect("matching week residue is accepted");
        assert_eq!(record.week, full_week);
    }
}

#[test]
fn gps_ura_index_table_matches_is_gps_200n() {
    // IS-GPS-200N 20.3.3.3.1.3 URA index -> meters (band upper bound).
    let expected = [
        (0, 2.4),
        (1, 3.4),
        (2, 4.85),
        (3, 6.85),
        (4, 9.65),
        (5, 13.65),
        (6, 24.0),
        (7, 48.0),
        (8, 96.0),
        (9, 192.0),
        (10, 384.0),
        (11, 768.0),
        (12, 1536.0),
        (13, 3072.0),
        (14, 6144.0),
    ];
    for (index, meters) in expected {
        assert_eq!(
            gps_ura_index_to_meters(index),
            Some(meters),
            "URA index {index}"
        );
    }
    // Index 15 = no accuracy prediction / not to be used: distinct, not a bogus
    // finite value. Out-of-range indices are also None.
    assert_eq!(gps_ura_index_to_meters(15), None);
    assert_eq!(gps_ura_index_to_meters(16), None);
    assert_eq!(gps_ura_index_to_meters(-1), None);
}

#[test]
fn from_lnav_rejects_ura_index_15() {
    let mut decoded = sample_lnav_decoded();
    decoded.ura_index = 15;
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid GPS id");
    assert_eq!(
        BroadcastRecord::from_lnav(&decoded, sat, 2110),
        Err(LnavRecordError::NoUraPrediction(15))
    );
}

#[test]
fn gps_fit_interval_mapping_matches_is_gps_200n_table_20_xii() {
    // flag 0 -> 4 hours regardless of IODE/IODC (IODE values are in the 0-239
    // normal-operations range, IS-GPS-200N 20.3.3.4.3.1 / Table 20-XII).
    assert_eq!(gps_fit_interval_from_flag(0, 12, 12), Ok(4.0 * 3600.0));

    // flag 1, short-term extended (IODE < 240) -> 6 hours.
    assert_eq!(gps_fit_interval_from_flag(1, 12, 12), Ok(6.0 * 3600.0));
    assert_eq!(gps_fit_interval_from_flag(1, 239, 239), Ok(6.0 * 3600.0));

    // flag 1, long-term extended (IODE 240-255) -> IODC selects the fit length.
    assert_eq!(gps_fit_interval_from_flag(1, 240, 240), Ok(8.0 * 3600.0));
    assert_eq!(gps_fit_interval_from_flag(1, 247, 247), Ok(8.0 * 3600.0));
    assert_eq!(gps_fit_interval_from_flag(1, 248, 248), Ok(14.0 * 3600.0));
    assert_eq!(gps_fit_interval_from_flag(1, 255, 496), Ok(14.0 * 3600.0));
    assert_eq!(gps_fit_interval_from_flag(1, 250, 497), Ok(26.0 * 3600.0));
    assert_eq!(gps_fit_interval_from_flag(1, 250, 1023), Ok(26.0 * 3600.0));

    // Reserved IODC for long-term extended (504-511, 752-767, 1008-1020).
    assert_eq!(
        gps_fit_interval_from_flag(1, 250, 504),
        Err(LnavRecordError::FitIntervalUnsupported {
            fit_interval_flag: 1,
            iode: 250,
            iodc: 504,
        })
    );

    // IODE above the defined extended range is not a valid flag-1 combination.
    assert_eq!(
        gps_fit_interval_from_flag(1, 256, 256),
        Err(LnavRecordError::FitIntervalUnsupported {
            fit_interval_flag: 1,
            iode: 256,
            iodc: 256,
        })
    );

    // A flag outside {0, 1} cannot be a defined interval.
    assert_eq!(
        gps_fit_interval_from_flag(2, 12, 12),
        Err(LnavRecordError::FitIntervalUnsupported {
            fit_interval_flag: 2,
            iode: 12,
            iodc: 12,
        })
    );

    // A negative IODE is not a real 8-bit decode and must not be read as a
    // short-term extended (6-hour) fit.
    assert_eq!(
        gps_fit_interval_from_flag(1, -1, 12),
        Err(LnavRecordError::FitIntervalUnsupported {
            fit_interval_flag: 1,
            iode: -1,
            iodc: 12,
        })
    );
}

#[test]
fn from_lnav_uses_extended_fit_interval_for_flag_1() {
    let mut decoded = sample_lnav_decoded();
    decoded.fit_interval_flag = 1; // short-term extended, IODE 12 < 240
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid GPS id");
    let record = BroadcastRecord::from_lnav(&decoded, sat, 2110).expect("GPS LNAV record");
    assert_eq!(record.fit_interval_s, Some(6.0 * 3600.0));
}

#[test]
fn parse_glonass_skips_extended_slot_and_keeps_others() {
    // R28 is an extended GLONASS slot beyond the engine's PRN cap (R01-R27),
    // as seen in real BKG/IGS broadcast-nav files. It must be skipped while a
    // valid R01 record in the same file still parses - one unrepresentable
    // record must not reject the whole file.
    let mut lines = r01_glonass_lines();
    lines.extend(satellite_lines(R01_GLONASS_LINES, "R28"));

    let recs = parse_glonass(&glonass_text(&lines))
        .expect("an extended GLONASS slot must not reject the file");
    assert_eq!(recs.len(), 1, "only the representable R01 record is kept");
    assert_eq!(recs[0].satellite_id.prn, 1);
}

#[test]
fn parse_glonass_lenient_surfaces_skipped_extended_slots() {
    // The lenient parser keeps the representable records and reports the slots it
    // dropped (here R28) with their tokens, rather than discarding them silently.
    let mut lines = r01_glonass_lines();
    lines.extend(satellite_lines(R01_GLONASS_LINES, "R28"));

    let parsed = parse_glonass_lenient(&glonass_text(&lines))
        .expect("an extended GLONASS slot must not reject the file");
    assert_eq!(parsed.records.len(), 1, "only R01 is representable");
    assert_eq!(parsed.records[0].satellite_id.prn, 1);
    assert_eq!(
        parsed.skipped,
        vec![SkippedGlonass {
            token: "R28".to_string()
        }],
        "the dropped extended slot is surfaced, not silent"
    );
}

#[test]
fn extended_glonass_slot_does_not_discard_other_systems() {
    // A mixed file: a GPS Keplerian record, a healthy GLONASS R01, and an
    // extended R28. The R28 slot is skipped; GPS and R01 both survive.
    let mut lines = g01_lines();
    lines.extend(r01_glonass_lines());
    lines.extend(satellite_lines(R01_GLONASS_LINES, "R28"));

    let store = BroadcastStore::from_nav(&glonass_text(&lines))
        .expect("an extended GLONASS slot must not reject the whole file");
    let gps = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    assert!(
        store.records().iter().any(|r| r.satellite_id == gps),
        "GPS record still loads alongside the skipped R28"
    );
    assert_eq!(
        store.glonass_records().len(),
        1,
        "R01 kept, extended R28 skipped"
    );
    assert_eq!(store.glonass_records()[0].satellite_id.prn, 1);
}

#[test]
fn galileo_iono_corr_with_three_coefficients_parses() {
    // Real/merged headers carry a `GAL IONOSPHERIC CORR` line with only the
    // three NeQuick-G coefficients (a0,a1,a2); the fourth column (the
    // disturbance flag) is blank. A short iono line must not reject the header.
    let four = "GAL    2.8250e+01  7.8125e-03  1.0071e-02  0.0000E+00       IONOSPHERIC CORR";
    // Blank the fourth coefficient column (41..53) to model the 3-coefficient
    // line, leaving the three coefficients in columns 5..41 intact.
    let mut three = format!("{four:<53}");
    three.replace_range(41..53, &" ".repeat(12));

    let iono = parse_iono_corrections(&nav_text_with_header_line(&three))
        .expect("a 3-coefficient GAL iono line must not reject the header");
    let gal = iono
        .galileo
        .expect("Galileo NeQuick coefficients still parsed");
    assert!((gal.ai0 - 2.8250e01).abs() < 1e-10, "ai0 {}", gal.ai0);
    assert!((gal.ai1 - 7.8125e-03).abs() < 1e-12, "ai1 {}", gal.ai1);
    assert!((gal.ai2 - 1.0071e-02).abs() < 1e-12, "ai2 {}", gal.ai2);
}

#[test]
fn gps_iono_corr_with_three_coefficients_is_rejected() {
    // GPS Klobuchar requires all four coefficients (alpha0..alpha3). A truncated
    // 3-column GPSA row is malformed and must error, not be silently accepted
    // with alpha3 defaulted to 0 (which would corrupt the ionospheric model).
    let four = "GPSA   4.6566e-09  1.4901e-08 -5.9605e-08 -1.1921E-07       IONOSPHERIC CORR";
    let mut three = format!("{four:<53}");
    three.replace_range(41..53, &" ".repeat(12));

    let err = parse_iono_corrections(&nav_text_with_header_line(&three))
        .expect_err("a truncated 3-coefficient GPS Klobuchar line must be rejected");
    assert!(
        matches!(err, NavParseError::BadHeaderField { .. }),
        "expected a malformed-header error, got {err:?}"
    );
}

#[test]
fn broadcast_store_sources_glonass_channels_from_nav() {
    // When an OBS file lacks `GLONASS SLOT / FRQ #` records, the per-satellite
    // FDMA channel numbers are obtainable from the broadcast nav GLONASS
    // records via the convenience accessor.
    let store =
        BroadcastStore::from_nav(&glonass_text(&r01_glonass_lines())).expect("GLONASS NAV parses");
    let channels = store.glonass_frequency_channels();
    assert_eq!(
        channels.get(&1).copied(),
        Some(1),
        "R01 FDMA channel sourced from nav"
    );
    assert_eq!(
        store.glonass_records()[0].freq_channel,
        1,
        "accessor matches the per-record channel"
    );
}

#[test]
fn encode_nav_round_trips_through_parse() {
    // The canonical IR is the parsed record set. Encoding it and re-parsing must
    // reproduce every BroadcastRecord: satellite, message, week, scale-tagged
    // toe/toc, Keplerian elements, clock polynomial, group delays, health,
    // accuracy, and the GPS fit interval. The fixture carries GPS, Galileo, and
    // BeiDou (GEO/IGSO/MEO) records, so all three column layouts are exercised.
    let original = records();
    assert!(
        original.len() > 2000,
        "fixture should carry the full multi-GNSS record set"
    );
    let encoded = encode_nav(&original);
    let reparsed = parse_nav(&encoded).expect("re-parse encoded NAV");
    assert_eq!(
        reparsed, original,
        "encode_nav must round-trip through parse"
    );

    // Deterministic: the same records always serialize byte-identically.
    assert_eq!(encode_nav(&original), encoded);
}
