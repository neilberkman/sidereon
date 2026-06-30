//! Parser unit + property tests for the SP3-c / SP3-d reader.
//!
//! These are supplemental coverage (round-trip, missing-value, boundary,
//! flags, unit conversion). They are NOT the 0-ULP parity acceptance gate -
//! that is the gnssanalysis/scipy golden-vector suite for the *interpolation*,
//! which is a separate deliverable. A parser is not a contested float recipe,
//! so these prove correctness of the byte/record decoding and the
//! unit/frame/flag contracts.
//!
//! SP3 fixture provenance (all under `tests/fixtures/sp3/`):
//!
//!   * `GRG0MGXFIN_20201760000_01D_15M_ORB.SP3` (+ `.gz`) — IGS MGEX final combined
//!     precise orbit+clock, AC CNES/CLS/GRGS, SP3-c, 2020 DOY 176 (GPS week 2111),
//!     96 epochs at 900 s, 75 sats (GPS/GLONASS/Galileo). From
//!     `https://raw.githubusercontent.com/nav-solutions/data/main/SP3/C/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3.gz`
//!     (redistributes the public IGS product; original CDDIS/IGS MGEX). gz 198894 B
//!     sha256 1787c5922d54c38bff1fae2e9c9a32fd8bb401419d0643cf0dbe934d9a720cc7;
//!     decompressed 443618 B sha256
//!     e123cedd659bf83eadaf40aa3845d58457e1f2596ffbd196de433a79898cffab. The SP3
//!     position-interpolation parity golden in `interp/interp_tests.rs` is generated
//!     from RTKLIB `peph2pos`/`interppol` (degree-10 sliding-window Lagrange,
//!     opt=0), certified sats G01/G15/G32; the clock channel uses scipy CubicSpline.
//!   * `GAP_G01_20201760000_15M.sp3` — derived from the GRG product by removing
//!     G01's position records for the contiguous block 07:30..10:00 (~2.5 h gap);
//!     all other sats and the header verbatim, no values altered. Proves the
//!     Lagrange window never interpolates across a gap.
//!   * `degenerate_coincident_5sat.sp3` — hand-authored (no external source), valid
//!     SP3-c with five GPS sats (G01-G05) at identical ECEF (26560,0,0) km, zero
//!     clock, two 15-min epochs. Rank-deficient geometry; proves graceful degrade
//!     (position with no DOP, no panic).
//!   * `GBM0MGXRAP_20201770000_01D_05M_ORB.SP3` — GFZ rapid MGEX, 2020 DOY 177,
//!     5-min, 122 sats incl. BeiDou C01-C60. From
//!     `ftp://ftp.gfz-potsdam.de/pub/GNSS/products/mgex/2111/GBM0MGXRAP_20201770000_01D_05M_ORB.SP3.gz`
//!     (gz sha256 51971877df4b4bb6c43bb13ff5c850752100d38048526d6bf39ecd98b54aaf27).
//!     The committed GRG product carries no BeiDou; GBM is the BeiDou physical
//!     anchor. Full day not vendored into the crate; trims below are committed.
//!   * `GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3` — first 120 epochs + the
//!     records the Wettzell RTK real-arc harness needs (243328 B sha256
//!     769e61ab9153cac0c9103df1b1721cda8a8e04457188b862a5f63c431ca3cba2); no values
//!     altered.
//!   * `GBM_BDS_C21_C08_trim.sp3` — header + position records for BeiDou C21 (MEO)
//!     and C08 (IGSO) across all 288 epochs, other sats dropped (72293 B sha256
//!     f77d83a0da91e7112c2890ba7aae29326b8c621cfee58ac18e4243d86e40238b); the
//!     BeiDou drift gate for the reduced-orbit eccentric_secular model.
//!   * `IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3` — IGS official combined final
//!     (GPS-only legacy combination), SP3-c, frame IGc20, 15-min, 2026 DOY 120 (GPS
//!     week 2416), 11 epochs 09:45..12:15, all 31 GPS sats verbatim. From Wuhan
//!     University IGS mirror
//!     `ftp://igs.gnsswhu.cn/pub/gps/products/2416/IGS0OPSFIN_20261200000_01D_15M_ORB.SP3.gz`
//!     (also BKG `https://igs.bkg.bund.de/root_ftp/IGS/products/2416/`). Used as the
//!     multi-center combine oracle (see `combine.rs`) and by
//!     `sp3_bodies_python_fixture.rs`/`rtklib_oracles.rs`.

use super::*;

/// A minimal but standards-shaped SP3-c position+clock file with two GPS sats,
/// two epochs, a missing-orbit record, a bad-clock record, and assorted flags.
const SP3C_FILE: &str = "\
#cP2020  6 24  0  0  0.00000000       2 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-c FIXTURE
*  2020  6 24  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    123.456789
PG02  -1234.567890   2345.678901  -3456.789012 999999.999999
*  2020  6 24  0 15  0.00000000
PG01  15100.000000 -20100.000000   5100.000000   -987.654321              E
PG02      0.000000      0.000000      0.000000    100.000000
EOF
";

/// A minimal SP3-d multi-GNSS file with position+velocity records, GPS +
/// Galileo + BeiDou, and predicted/maneuver flags.
const SP3D_FILE: &str = "\
#dV2022  1  2  3  4  5.00000000       1 ORBIT IGS20 FIT  TST
## 2191 270245.00000000   300.00000000 59581 0.1281597222222
+    3   G05E11C30  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-d FIXTURE
*  2022  1  2  3  4  5.00000000
PG05  10000.000000  20000.000000  30000.000000    -50.000000
VG05  10000.000000 -20000.000000  30000.000000      1.000000
PE11 -11111.111111  22222.222222 -33333.333333    250.000000                   P
VE11  -5000.000000   5000.000000  -5000.000000      2.500000
PC30   1000.000000   2000.000000   3000.000000    -10.000000                  MP
VC30   1234.000000   5678.000000   9012.000000     -1.000000
EOF
";

fn id(sys: GnssSystem, prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(sys, prn).expect("valid satellite id")
}

fn assert_parse_error_contains(text: &str, needle: &str) {
    let err = Sp3::parse(text.as_bytes()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains(needle)),
        "expected parse error containing {needle:?}; got {err:?}"
    );
}

#[test]
fn parses_sp3c_header() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).expect("parse SP3-c");
    let h = &sp3.header;
    assert_eq!(h.version, Sp3Version::C);
    assert_eq!(h.data_type, Sp3DataType::Position);
    assert_eq!(h.num_epochs, 2);
    assert_eq!(h.coordinate_system, "IGS14");
    assert_eq!(h.orbit_type, "FIT");
    assert_eq!(h.agency, "TST");
    assert_eq!(h.gnss_week, 2111);
    assert_eq!(h.seconds_of_week, 432000.0);
    assert_eq!(h.epoch_interval_s, 900.0);
    assert_eq!(h.mjd, 59024);
    assert_eq!(h.time_system, Sp3TimeSystem::Gps);
    assert_eq!(h.time_scale, TimeScale::Gpst);
    assert_eq!(
        h.satellites,
        vec![id(GnssSystem::Gps, 1), id(GnssSystem::Gps, 2)]
    );
    assert_eq!(h.satellite_accuracy_codes, vec![0, 0]);
    assert_eq!(sp3.epoch_count(), 2);
    assert_eq!(sp3.comments, vec!["TEST SP3-c FIXTURE".to_string()]);
}

#[test]
fn parses_sp3c_position_and_clock_units() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    let st = sp3.state(id(GnssSystem::Gps, 1), 0).unwrap();
    // km -> m is a single *1000 multiply.
    assert_eq!(st.position.x_m, 15000.000000 * 1_000.0);
    assert_eq!(st.position.y_m, -20000.000000 * 1_000.0);
    assert_eq!(st.position.z_m, 5000.000000 * 1_000.0);
    // us -> s is a single *1e-6 multiply.
    assert_eq!(st.clock_s, Some(123.456789 * 1.0e-6));
    assert!(st.velocity.is_none());
    assert!(st.clock_rate_s_s.is_none());
    assert_eq!(st.flags, Sp3Flags::default());
}

#[test]
fn missing_clock_sentinel_is_none_but_position_kept() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    // G02 epoch 0 has a valid position but the 999999.999999 bad-clock sentinel.
    let st = sp3.state(id(GnssSystem::Gps, 2), 0).unwrap();
    assert_eq!(st.position.x_m, -1234.567890 * 1_000.0);
    assert_eq!(st.clock_s, None, "bad-clock sentinel must surface as None");
}

#[test]
fn missing_position_record_is_dropped() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    // G02 epoch 1 is the all-zero (missing-orbit) sentinel: no state recorded.
    let err = sp3.state(id(GnssSystem::Gps, 2), 1).unwrap_err();
    assert_eq!(err, Error::UnknownSatellite(id(GnssSystem::Gps, 2)));
    // G01 at the same epoch is still present.
    assert!(sp3.state(id(GnssSystem::Gps, 1), 1).is_ok());
}

#[test]
fn clock_event_flag_parsed() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    let st = sp3.state(id(GnssSystem::Gps, 1), 1).unwrap();
    assert!(st.flags.clock_event, "E flag in clock-event column");
    assert!(!st.flags.orbit_predicted);
    assert_eq!(st.clock_s, Some(-987.654321 * 1.0e-6));
}

#[test]
fn parses_sp3d_multignss_velocity() {
    let sp3 = Sp3::parse(SP3D_FILE.as_bytes()).expect("parse SP3-d");
    assert_eq!(sp3.header.version, Sp3Version::D);
    assert_eq!(sp3.header.data_type, Sp3DataType::Velocity);
    assert_eq!(
        sp3.header.satellites,
        vec![
            id(GnssSystem::Gps, 5),
            id(GnssSystem::Galileo, 11),
            id(GnssSystem::BeiDou, 30),
        ]
    );

    // GPS sat: position + velocity, distinct per-axis (guards the refs/sp3
    // X/Y axis bug).
    let g = sp3.state(id(GnssSystem::Gps, 5), 0).unwrap();
    assert_eq!(g.position.x_m, 10000.0 * 1_000.0);
    let v = g.velocity.expect("velocity present");
    // dm/s -> m/s is *0.1, and each axis is read independently.
    assert_eq!(v.vx_m_s, 10000.0 * 0.1);
    assert_eq!(v.vy_m_s, -20000.0 * 0.1);
    assert_eq!(v.vz_m_s, 30000.0 * 0.1);
    assert_ne!(v.vx_m_s, v.vy_m_s, "X and Y velocity must not be aliased");
    // clock-rate: 1e-4 us/s field -> s/s is *1e-10.
    assert_eq!(g.clock_rate_s_s, Some(1.0 * 1.0e-10));
}

#[test]
fn predicted_and_maneuver_flags_sp3d() {
    let sp3 = Sp3::parse(SP3D_FILE.as_bytes()).unwrap();
    let e = sp3.state(id(GnssSystem::Galileo, 11), 0).unwrap();
    assert!(e.flags.orbit_predicted, "trailing P = predicted orbit");
    let c = sp3.state(id(GnssSystem::BeiDou, 30), 0).unwrap();
    assert!(c.flags.maneuver, "M = maneuver");
    assert!(c.flags.orbit_predicted, "P after M = predicted orbit");
}

#[test]
fn epoch_index_out_of_range_errors() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    assert_eq!(
        sp3.state(id(GnssSystem::Gps, 1), 99),
        Err(Error::EpochOutOfRange)
    );
    assert!(sp3.states_at(99).is_err());
}

#[test]
fn epoch_julian_split_is_consistent() {
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    // 2020-06-24 00:00:00 -> the day boundary should be on a *.5 JD whole, and
    // the second epoch is 900 s = 0.625 hours later, i.e. fraction differs by
    // exactly 900/86400.
    let e0 = sp3.epochs[0].julian_date().unwrap();
    let e1 = sp3.epochs[1].julian_date().unwrap();
    assert_eq!(e0.jd_whole, e1.jd_whole, "same civil day");
    assert_eq!(e1.fraction - e0.fraction, 900.0 / 86_400.0);
    // 2020-06-24 is JD 2459024.5 at midnight.
    assert_eq!(e0.jd_whole + e0.fraction, 2_459_024.5);
}

// --- Boundary / malformed-input tests ------------------------------------

#[test]
fn missing_header_line1_errors() {
    let no_h1 = "\
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
EOF
";
    let err = Sp3::parse(no_h1.as_bytes()).unwrap_err();
    assert!(matches!(err, Error::Parse(_)));
}

#[test]
fn missing_header_line2_errors() {
    let no_h2 = "\
#cP2020  6 24  0  0  0.00000000      2 ORBIT IGS14 FIT  TST
EOF
";
    let err = Sp3::parse(no_h2.as_bytes()).unwrap_err();
    assert!(matches!(err, Error::Parse(_)));
}

#[test]
fn missing_pc_descriptor_errors_for_sp3c() {
    // SP3-c with NO %c line at all: the time system is unknown and MUST NOT
    // silently default to GPST. finish() rejects it.
    let no_pc = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
*  2020  6 24  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    123.456789
EOF
";
    let err = Sp3::parse(no_pc.as_bytes()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("time system")),
        "missing %c must error, not default to GPST; got {err:?}"
    );
}

#[test]
fn short_pc_descriptor_errors_for_sp3c() {
    // SP3-c with a %c line too short to carry the time-system field: error,
    // never a GPST fallback.
    let short_pc = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G
EOF
";
    let err = Sp3::parse(short_pc.as_bytes()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("too short")),
        "short %c must error; got {err:?}"
    );
}

#[test]
fn blank_pc_time_system_errors_for_sp3c() {
    // SP3-c %c line long enough but with a blank time-system field (cols 9-12):
    // a blank scale is malformed, not GPST.
    let blank_pc = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc     ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
EOF
";
    let err = Sp3::parse(blank_pc.as_bytes()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("blank")),
        "blank %c time system must error; got {err:?}"
    );
}

#[test]
fn sp3a_with_no_pc_descriptor_is_gpst() {
    // SP3-a predates %c and is implicitly GPST; a file with no %c line still
    // parses and resolves to GPST.
    let sp3a = "\
#aP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
*  2020  6 24  0  0  0.00000000
P  1  15000.000000 -20000.000000   5000.000000    123.456789
EOF
";
    let sp3 = Sp3::parse(sp3a.as_bytes()).expect("SP3-a parses without %c");
    assert_eq!(sp3.header.version, Sp3Version::A);
    assert_eq!(sp3.header.time_system, Sp3TimeSystem::Gps);
    assert_eq!(sp3.header.time_scale, TimeScale::Gpst);
    assert!(sp3.state(id(GnssSystem::Gps, 1), 0).is_ok());
}

#[test]
fn sp3a_ignores_pc_descriptor_and_stays_gpst() {
    // Even if an SP3-a file carries a %c line with a non-GPS label, SP3-a is
    // implicitly GPST and the descriptor is ignored (no error, no aliasing).
    let sp3a = "\
#aP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c R  cc GLO ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
*  2020  6 24  0  0  0.00000000
P  1  15000.000000 -20000.000000   5000.000000    123.456789
EOF
";
    let sp3 = Sp3::parse(sp3a.as_bytes()).expect("SP3-a parses, ignoring %c");
    assert_eq!(sp3.header.time_system, Sp3TimeSystem::Gps);
    assert_eq!(sp3.header.time_scale, TimeScale::Gpst);
}

#[test]
fn valid_gps_pc_descriptor_parses() {
    // A valid GPS %c on an SP3-c file resolves to GPST (the SP3C_FILE fixture).
    let sp3 = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    assert_eq!(sp3.header.time_system, Sp3TimeSystem::Gps);
    assert_eq!(sp3.header.time_scale, TimeScale::Gpst);
}

fn sp3_fixture_with_time_system(label: &str) -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    );
    let text = std::fs::read_to_string(path).expect("read committed SP3 fixture");
    text.replacen("%c M  cc GPS", &format!("%c M  cc {label}"), 1)
}

#[test]
fn standard_sp3_time_system_labels_parse_from_committed_fixture() {
    for (label, system, scale) in [
        ("GLO", Sp3TimeSystem::Glonass, TimeScale::Utc),
        ("QZS", Sp3TimeSystem::Qzss, TimeScale::Qzsst),
        ("IRN", Sp3TimeSystem::Irnss, TimeScale::Gpst),
        ("GAL", Sp3TimeSystem::Galileo, TimeScale::Gst),
        ("BDT", Sp3TimeSystem::Beidou, TimeScale::Bdt),
        ("TAI", Sp3TimeSystem::Tai, TimeScale::Tai),
        ("UTC", Sp3TimeSystem::Utc, TimeScale::Utc),
    ] {
        let text = sp3_fixture_with_time_system(label);
        let sp3 = Sp3::parse(text.as_bytes())
            .unwrap_or_else(|err| panic!("{label} SP3 fixture should parse: {err}"));
        assert_eq!(sp3.header.time_system, system, "{label} label");
        assert_eq!(sp3.header.time_scale, scale, "{label} core scale");
        assert_eq!(sp3.header.time_system.label(), label);
        assert_eq!(sp3.epoch_count(), 96);
    }
}

#[test]
fn bogus_sp3_time_system_label_errors() {
    let text = sp3_fixture_with_time_system("BAD");
    let err = Sp3::parse(text.as_bytes()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("BAD")),
        "bogus time-system label must error; got {err:?}"
    );
}

#[test]
fn malformed_mjd_fraction_errors() {
    let bad = SP3C_FILE.replace("59024 0.0000000000000", "59024 NOT_A_NUMBER");
    assert_parse_error_contains(&bad, "mjd_fraction");
}

#[test]
fn epoch_line_rejects_invalid_civil_month() {
    let bad = SP3C_FILE.replacen(
        "*  2020  6 24  0  0  0.00000000",
        "*  2020 13 24  0  0  0.00000000",
        1,
    );
    assert_parse_error_contains(&bad, "valid civil date");
}

#[test]
fn epoch_line_rejects_invalid_civil_hour() {
    let bad = SP3C_FILE.replacen(
        "*  2020  6 24  0  0  0.00000000",
        "*  2020  6 24 24  0  0.00000000",
        1,
    );
    assert_parse_error_contains(&bad, "valid civil time");
}

#[test]
fn gps_time_system_rejects_leap_second_epoch_label() {
    let bad = SP3C_FILE.replacen(
        "*  2020  6 24  0  0  0.00000000",
        "*  2016 12 31 23 59 60.00000000",
        1,
    );
    assert_parse_error_contains(&bad, "valid civil time");
}

#[test]
fn utc_time_system_accepts_leap_second_epoch_label() {
    let utc = SP3C_FILE
        .replacen("%c G  cc GPS", "%c G  cc UTC", 1)
        .replacen(
            "*  2020  6 24  0  0  0.00000000",
            "*  2016 12 31 23 59 60.00000000",
            1,
        );
    let sp3 = Sp3::parse(utc.as_bytes()).expect("UTC SP3 leap-second epoch");
    assert_eq!(sp3.header.time_system, Sp3TimeSystem::Utc);
    assert_eq!(sp3.epochs[0].scale, TimeScale::Utc);
}

#[test]
fn utc_time_system_accepts_fractional_leap_second_epoch_label() {
    let utc = SP3C_FILE
        .replacen("%c G  cc GPS", "%c G  cc UTC", 1)
        .replacen(
            "*  2020  6 24  0  0  0.00000000",
            "*  2016 12 31 23 59 60.50000000",
            1,
        );

    let sp3 = Sp3::parse(utc.as_bytes()).expect("UTC SP3 fractional leap-second epoch");
    let split = sp3.epochs[0]
        .julian_date()
        .expect("SP3 epoch stored as split JD");

    assert_eq!(sp3.header.time_system, Sp3TimeSystem::Utc);
    assert_eq!(sp3.epochs[0].scale, TimeScale::Utc);
    assert_eq!(split.jd_whole, 2_457_754.5);
    assert!((split.fraction - 0.5 / 86_400.0).abs() < 1.0e-15);
}

#[test]
fn utc_time_system_rejects_malformed_leap_second_epoch_without_panic() {
    let bad = SP3C_FILE
        .replacen("%c G  cc GPS", "%c G  cc UTC", 1)
        .replacen(
            "*  2020  6 24  0  0  0.00000000",
            "*  2016 12 31 23 59 61.00000000",
            1,
        );

    assert_parse_error_contains(&bad, "valid civil time");
}

#[test]
fn truncated_position_record_errors() {
    let bad = SP3C_FILE.replace(
        "PG01  15000.000000 -20000.000000   5000.000000    123.456789",
        "PG01  15000.000000 -20000.000000",
    );
    assert_parse_error_contains(&bad, "position record truncated");
}

#[test]
fn truncated_velocity_record_errors() {
    let bad = SP3D_FILE.replace(
        "VG05  10000.000000 -20000.000000  30000.000000      1.000000",
        "VG05  10000.000000 -20000.000000",
    );
    assert_parse_error_contains(&bad, "velocity record truncated");
}

#[test]
fn non_finite_position_coordinate_errors() {
    let bad_line = format!(
        "PG01{:>14}{:>14}{:>14}{:>14}",
        "NaN", "-20000.000000", "5000.000000", "123.456789"
    );
    let bad = SP3C_FILE.replace(
        "PG01  15000.000000 -20000.000000   5000.000000    123.456789",
        &bad_line,
    );
    assert_parse_error_contains(&bad, "coordinate is not a finite number");
}

#[test]
fn non_finite_clock_errors() {
    let bad_line = format!(
        "PG01{:>14}{:>14}{:>14}{:>14}",
        "15000.000000", "-20000.000000", "5000.000000", "NaN"
    );
    let bad = SP3C_FILE.replace(
        "PG01  15000.000000 -20000.000000   5000.000000    123.456789",
        &bad_line,
    );
    assert_parse_error_contains(&bad, "clock is not a finite number");
}

#[test]
fn non_finite_velocity_coordinate_errors() {
    let bad_line = format!(
        "VG05{:>14}{:>14}{:>14}{:>14}",
        "NaN", "-20000.000000", "30000.000000", "1.000000"
    );
    let bad = SP3D_FILE.replace(
        "VG05  10000.000000 -20000.000000  30000.000000      1.000000",
        &bad_line,
    );
    assert_parse_error_contains(&bad, "coordinate is not a finite number");
}

#[test]
fn non_finite_clock_rate_errors() {
    let bad_line = format!(
        "VG05{:>14}{:>14}{:>14}{:>14}",
        "10000.000000", "-20000.000000", "30000.000000", "NaN"
    );
    let bad = SP3D_FILE.replace(
        "VG05  10000.000000 -20000.000000  30000.000000      1.000000",
        &bad_line,
    );
    assert_parse_error_contains(&bad, "clock is not a finite number");
}

#[test]
fn velocity_only_record_produces_no_state() {
    // A V-record with no preceding P-record for that sat at the epoch must NOT
    // synthesize a (0,0,0) geocenter position. The satellite is left absent.
    let vel_only = "\
#dV2022  1  2  3  4  5.00000000       1 ORBIT IGS20 FIT  TST
## 2191 270245.00000000   300.00000000 59581 0.1281597222222
+    1   G05  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
*  2022  1  2  3  4  5.00000000
VG05  10000.000000 -20000.000000  30000.000000      1.000000
EOF
";
    let sp3 = Sp3::parse(vel_only.as_bytes()).expect("parse velocity-only");
    // No state for G05: a fabricated (0,0,0) must never be exposed.
    let err = sp3.state(id(GnssSystem::Gps, 5), 0).unwrap_err();
    assert_eq!(err, Error::UnknownSatellite(id(GnssSystem::Gps, 5)));
    // states_at must likewise be empty for this epoch.
    assert!(
        sp3.states_at(0).unwrap().is_empty(),
        "no (0,0,0) state leaked"
    );
}

#[test]
fn position_then_velocity_augments_velocity() {
    // The normal P-then-V case still augments the existing position state with
    // the velocity (regression guard for the #4 fix).
    let sp3 = Sp3::parse(SP3D_FILE.as_bytes()).unwrap();
    let st = sp3.state(id(GnssSystem::Gps, 5), 0).unwrap();
    // Real position (not the fabricated geocenter).
    assert_eq!(st.position.x_m, 10000.0 * 1_000.0);
    let v = st
        .velocity
        .expect("velocity augmented onto the P-record state");
    assert_eq!(v.vx_m_s, 10000.0 * 0.1);
    assert_eq!(v.vy_m_s, -20000.0 * 0.1);
    assert_eq!(v.vz_m_s, 30000.0 * 0.1);
}

#[test]
fn position_record_before_epoch_errors() {
    let bad = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
PG01  15000.000000 -20000.000000   5000.000000    123.456789
EOF
";
    let err = Sp3::parse(bad.as_bytes()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("before any epoch")),
        "got {err:?}"
    );
}

#[test]
fn non_utf8_input_errors() {
    let bytes = [0xffu8, 0xfe, 0x00, 0x01];
    let err = Sp3::parse(&bytes).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("UTF-8")),
        "got {err:?}"
    );
}

#[test]
fn multibyte_comment_text_errors_without_panicking() {
    let err = Sp3::parse(b"/*\xc3\xa9\n").unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref m) if m.contains("ASCII")),
        "got {err:?}"
    );
}

#[test]
fn trailing_truncation_after_eof_tolerated() {
    // Lines after EOF are ignored; parse still succeeds with the prior epochs.
    let truncated = format!("{SP3C_FILE}garbage line that should be ignored\n");
    let sp3 = Sp3::parse(truncated.as_bytes()).unwrap();
    assert_eq!(sp3.epoch_count(), 2);
}

// --- Property-style tests (round-trip on the value contracts) ------------

#[test]
fn sv_token_round_trips_through_display() {
    // Valid system+PRN pairs parse back to the same id, which is the parser's
    // identity contract for the satellite list.
    for (sys, prns) in [
        (GnssSystem::Gps, &[1u8, 5, 9, 12, 30, 32][..]),
        (GnssSystem::Glonass, &[1u8, 5, 9, 12, 26, 27][..]),
        (GnssSystem::Galileo, &[1u8, 5, 9, 12, 30, 36][..]),
        (GnssSystem::BeiDou, &[1u8, 5, 9, 12, 30, 63][..]),
        (GnssSystem::Qzss, &[1u8, 5, 9][..]),
        (GnssSystem::Navic, &[1u8, 5, 9, 14][..]),
        (GnssSystem::Sbas, &[20u8, 23, 36, 58][..]),
    ] {
        for &prn in prns {
            let want = id(sys, prn);
            let token = want.to_string(); // e.g. "G01"
            let got = super::parse_sv_token(&token, Some(Sp3Version::D))
                .unwrap_or_else(|| panic!("token {token:?} failed to parse"));
            assert_eq!(got, want);
        }
    }
}

#[test]
fn sp3a_bare_numeric_prn_is_gps() {
    // SP3-a omits the constellation letter: a bare PRN is GPS.
    assert_eq!(
        super::parse_sv_token(" 7", Some(Sp3Version::A)),
        Some(id(GnssSystem::Gps, 7))
    );
    assert_eq!(
        super::parse_sv_token("23", Some(Sp3Version::A)),
        Some(id(GnssSystem::Gps, 23))
    );
}

#[test]
fn multibyte_line_does_not_panic() {
    // SP3 is ASCII, but a stray multibyte char must not panic the fixed-column
    // slicer (it should surface as a parse error, not an abort).
    let file = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
*  2020  6 24  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    123.456789 \u{00e9}\u{00e9}\u{00e9}
EOF
";
    // Must not panic; either parses (ignoring the trailing garbage) or errors.
    let _ = Sp3::parse(file.as_bytes());
}

#[test]
fn coordinate_sign_and_magnitude_preserved() {
    // A small fuzz over signed magnitudes parsed through fixed columns: build a
    // position record for each, parse, and require the meters back equals
    // km*1000 exactly (no rounding in the unit step).
    for &km in &[0.000001f64, -12345.678901, 26560.123456, -26560.999999] {
        // Position record columns: P(0) sv(1..4) x(4..18) y(18..32) z(32..46)
        // clk(46..60). Each numeric field is exactly 14 wide.
        let line = format!("PG01{:14.6}{:14.6}{:14.6}{:14.6}", km, km, km, 0.0);
        let file = format!(
            "#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST\n\
## 2111 432000.00000000   900.00000000 59024 0.0000000000000\n\
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
*  2020  6 24  0  0  0.00000000\n\
{line}\n\
EOF\n"
        );
        let sp3 = Sp3::parse(file.as_bytes()).unwrap();
        let st = sp3.state(id(GnssSystem::Gps, 1), 0).unwrap();
        assert_eq!(st.position.x_m, km * 1_000.0, "km={km}");
        assert_eq!(st.position.z_m, km * 1_000.0, "km={km}");
    }
}

// --- writer round-trip ------------------------------------------------------

/// Re-parse the written SP3 and assert it is semantically equal to `original`:
/// same epochs, satellite set, positions (mm), clocks (sub-ns), velocities.
fn assert_round_trip(original: &Sp3) {
    let text = original.to_sp3_string();
    let reparsed = Sp3::parse(text.as_bytes())
        .unwrap_or_else(|e| panic!("re-parse of written SP3 failed: {e}\n--- written ---\n{text}"));

    // Header derived from the product, not hardcoded.
    assert_eq!(reparsed.header.version, original.header.version, "version");
    assert_eq!(
        reparsed.header.time_scale, original.header.time_scale,
        "time scale"
    );
    assert_eq!(
        reparsed.header.time_system, original.header.time_system,
        "time system"
    );
    assert_eq!(
        reparsed.header.coordinate_system, original.header.coordinate_system,
        "coordinate system"
    );
    assert_eq!(
        reparsed.header.satellites, original.header.satellites,
        "satellite list"
    );
    assert_eq!(
        reparsed.header.satellite_accuracy_codes, original.header.satellite_accuracy_codes,
        "satellite accuracy codes"
    );
    assert_eq!(
        reparsed.header.num_epochs, original.header.num_epochs,
        "header epoch count"
    );
    assert_eq!(reparsed.comments, original.comments, "comments");
    assert_eq!(reparsed.epochs.len(), original.epochs.len(), "epoch count");

    for i in 0..original.epochs.len() {
        let ja = original.epochs[i].julian_date().unwrap();
        let jb = reparsed.epochs[i].julian_date().unwrap();
        assert!(
            ((ja.jd_whole + ja.fraction) - (jb.jd_whole + jb.fraction)).abs() < 1.0e-9,
            "epoch {i} time differs"
        );

        let sa = original.states_at(i).unwrap();
        let sb = reparsed.states_at(i).unwrap();
        let ka: Vec<_> = sa.keys().copied().collect();
        let kb: Vec<_> = sb.keys().copied().collect();
        assert_eq!(ka, kb, "satellite set at epoch {i}");

        for (sat, a) in sa {
            let b = &sb[sat];
            let (pa, pb) = (a.position.as_array(), b.position.as_array());
            for k in 0..3 {
                assert!(
                    (pa[k] - pb[k]).abs() < 1.0e-3,
                    "epoch {i} {sat:?} pos[{k}] {} vs {}",
                    pa[k],
                    pb[k]
                );
            }
            match (a.clock_s, b.clock_s) {
                (Some(x), Some(y)) => {
                    assert!(
                        (x - y).abs() < 1.0e-12,
                        "epoch {i} {sat:?} clock {x} vs {y}"
                    )
                }
                (None, None) => {}
                _ => panic!("epoch {i} {sat:?} clock presence mismatch"),
            }
            match (a.velocity, b.velocity) {
                (Some(x), Some(y)) => {
                    let (xa, yb) = (x.as_array(), y.as_array());
                    for k in 0..3 {
                        assert!((xa[k] - yb[k]).abs() < 1.0e-3, "epoch {i} {sat:?} vel[{k}]");
                    }
                }
                (None, None) => {}
                _ => panic!("epoch {i} {sat:?} velocity presence mismatch"),
            }
            assert_eq!(b.flags, a.flags, "epoch {i} {sat:?} flags");
        }
    }
}

#[test]
fn writer_round_trips_sp3c_position_clock() {
    assert_round_trip(&Sp3::parse(SP3C_FILE.as_bytes()).unwrap());
}

#[test]
fn writer_round_trips_sp3d_multignss_velocity() {
    assert_round_trip(&Sp3::parse(SP3D_FILE.as_bytes()).unwrap());
}

#[test]
fn writer_preserves_satellite_accuracy_codes() {
    let file = SP3C_FILE.replacen("++         0  0", "++         5 17", 1);
    let original = Sp3::parse(file.as_bytes()).expect("parse SP3 accuracy codes");
    assert_eq!(original.header.satellite_accuracy_codes, vec![5, 17]);

    let text = original.to_sp3_string();
    let first_accuracy_line = text
        .lines()
        .find(|line| line.starts_with("++"))
        .expect("written accuracy line");
    assert!(
        first_accuracy_line.starts_with("++         5 17"),
        "writer must preserve non-default accuracy code fields:\n{text}"
    );

    let reparsed = Sp3::parse(text.as_bytes()).expect("reparse written SP3 accuracy codes");
    assert_eq!(
        reparsed.header.satellite_accuracy_codes,
        original.header.satellite_accuracy_codes
    );
    assert_eq!(reparsed, original);
}

fn assert_exact_parse_write_parse_round_trip(file: &str) {
    let original = Sp3::parse(file.as_bytes()).expect("parse source SP3");
    let text = original.to_sp3_string();
    let reparsed = Sp3::parse(text.as_bytes())
        .unwrap_or_else(|err| panic!("reparse written SP3: {err}\n--- written ---\n{text}"));
    assert_eq!(
        reparsed, original,
        "parse -> write -> parse changed product"
    );
}

#[test]
fn writer_does_not_inject_comment_into_no_comment_product() {
    let no_comment = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
*  2020  6 24  0  0  0.00000000
PG01      1.000000      2.000000      3.000000 999999.999999
EOF
";
    let original = Sp3::parse(no_comment.as_bytes()).expect("parse no-comment SP3");
    assert!(original.comments.is_empty());

    let text = original.to_sp3_string();
    assert!(
        !text.lines().any(|line| line.starts_with("/*")),
        "writer must not synthesize a provenance comment:\n{text}"
    );
    assert_exact_parse_write_parse_round_trip(no_comment);
}

#[test]
fn parser_canonicalizes_declared_epoch_count_to_body_count() {
    let mismatched = "\
#cP2020  6 24  0  0  0.00000000       9 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* DECLARED COUNT MISMATCH
*  2020  6 24  0  0  0.00000000
PG01      1.000000      2.000000      3.000000 999999.999999
EOF
";
    let original = Sp3::parse(mismatched.as_bytes()).expect("parse mismatched-count SP3");
    assert_eq!(original.epochs.len(), 1);
    assert_eq!(original.header.num_epochs, original.epochs.len() as u64);

    assert_exact_parse_write_parse_round_trip(mismatched);
}

#[test]
fn writer_round_trips_utc_like_leap_second_epoch_without_hour_24() {
    for (label, expected_system) in [("UTC", Sp3TimeSystem::Utc), ("GLO", Sp3TimeSystem::Glonass)] {
        let file = SP3C_FILE
            .replacen("%c G  cc GPS", &format!("%c G  cc {label}"), 1)
            .replacen(
                "*  2020  6 24  0  0  0.00000000",
                "*  2016 12 31 23 59 60.00000000",
                1,
            );
        let original = Sp3::parse(file.as_bytes()).expect("parse UTC-like leap-second SP3");
        assert_eq!(original.header.time_system, expected_system);

        let text = original.to_sp3_string();
        assert!(
            text.contains("#cP2016 12 31 23 59 60.00000000"),
            "line-1 epoch must preserve the accepted leap-second label for {label}:\n{text}"
        );
        assert!(
            text.contains("*  2016 12 31 23 59 60.00000000\n"),
            "epoch line must preserve the accepted leap-second label for {label}:\n{text}"
        );
        assert!(
            !text.contains("2016 12 31 24  0  0.00000000"),
            "writer must not emit hour 24 for {label}:\n{text}"
        );
        assert!(
            text.contains("*  2020  6 24  0 15  0.00000000\n"),
            "ordinary epoch formatting must stay unchanged for {label}:\n{text}"
        );

        let reparsed = Sp3::parse(text.as_bytes())
            .unwrap_or_else(|err| panic!("reparse written {label} leap-second SP3: {err}"));
        assert_eq!(reparsed.epochs, original.epochs);
        assert_eq!(reparsed.header.time_system, expected_system);
    }
}

#[test]
fn writer_is_deterministic() {
    let sp3 = Sp3::parse(SP3D_FILE.as_bytes()).unwrap();
    assert_eq!(
        sp3.to_sp3_string(),
        sp3.to_sp3_string(),
        "byte-identical output"
    );
}

#[test]
fn writer_round_trips_record_flags() {
    let flags_file = "\
#dP2022  1  2  3  4  5.00000000       1 ORBIT IGS20 FIT  TST
## 2191 270245.00000000   300.00000000 59581 0.1281597222222
+    1   G05  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3 FLAGS
*  2022  1  2  3  4  5.00000000
PG05  10000.000000  20000.000000  30000.000000    -50.000000              EP  MP
EOF
";
    let original = Sp3::parse(flags_file.as_bytes()).expect("parse SP3 with all flags");
    let expected = Sp3Flags {
        clock_event: true,
        clock_predicted: true,
        maneuver: true,
        orbit_predicted: true,
    };
    assert_eq!(
        original.state(id(GnssSystem::Gps, 5), 0).unwrap().flags,
        expected
    );

    let text = original.to_sp3_string();
    let p_line = text
        .lines()
        .find(|line| line.starts_with("PG05"))
        .expect("written P record");
    assert_eq!(p_line.as_bytes().get(74), Some(&b'E'));
    assert_eq!(p_line.as_bytes().get(75), Some(&b'P'));
    assert_eq!(p_line.as_bytes().get(78), Some(&b'M'));
    assert_eq!(p_line.as_bytes().get(79), Some(&b'P'));

    let reparsed = Sp3::parse(text.as_bytes()).expect("reparse written SP3 flags");
    assert_eq!(
        reparsed.state(id(GnssSystem::Gps, 5), 0).unwrap().flags,
        expected
    );
}

#[test]
fn writer_emits_velocity_records_for_missing_velocity_cells() {
    let mut sp3 = Sp3::parse(SP3D_FILE.as_bytes()).unwrap();
    let missing_velocity = id(GnssSystem::Galileo, 11);
    let absent_sat = id(GnssSystem::BeiDou, 30);

    {
        let state = sp3.states[0]
            .get_mut(&missing_velocity)
            .expect("fixture state");
        state.velocity = None;
        state.clock_rate_s_s = None;
    }
    sp3.states[0]
        .remove(&absent_sat)
        .expect("fixture absent-sat state");

    let text = sp3.to_sp3_string();
    let lines: Vec<_> = text.lines().collect();
    let data_record_count = lines
        .iter()
        .filter(|line| line.starts_with('P') || line.starts_with('V'))
        .count();
    assert_eq!(
        data_record_count,
        sp3.header.satellites.len() * 2,
        "each header satellite must have a P+V record:\n{text}"
    );

    for (p, v) in [("PG05", "VG05"), ("PE11", "VE11"), ("PC30", "VC30")] {
        let p_idx = lines
            .iter()
            .position(|line| line.starts_with(p))
            .unwrap_or_else(|| panic!("missing {p} in:\n{text}"));
        assert!(
            lines.get(p_idx + 1).is_some_and(|line| line.starts_with(v)),
            "{p} must be followed by {v} in:\n{text}"
        );
    }

    for prefix in ["VE11", "VC30"] {
        let line = lines
            .iter()
            .find(|line| line.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing {prefix} in:\n{text}"));
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(
            fields.as_slice(),
            &[prefix, "0.000000", "0.000000", "0.000000", "999999.999999"],
            "{prefix} must use the SP3 missing-velocity and bad-clock-rate sentinels"
        );
    }

    let reparsed = Sp3::parse(text.as_bytes()).expect("reparse velocity sentinels");
    let state = reparsed.state(missing_velocity, 0).unwrap();
    assert!(
        state.velocity.is_none(),
        "missing velocity must not reparse as zero velocity"
    );
    assert!(state.clock_rate_s_s.is_none());
    assert!(
        reparsed.state(absent_sat, 0).is_err(),
        "absent satellite must remain absent after P+V sentinel records"
    );
}

#[test]
fn writer_emits_missing_satellite_as_sentinel_not_fabricated() {
    // SP3C epoch 1: G02 is the missing-orbit record (dropped on parse). The
    // writer must re-emit it as the 0,0,0 sentinel, so it re-reads as absent -
    // never a fabricated position.
    let original = Sp3::parse(SP3C_FILE.as_bytes()).unwrap();
    let text = original.to_sp3_string();
    assert!(
        text.contains("PG02      0.000000      0.000000      0.000000"),
        "absent G02 must be the 0,0,0 missing sentinel:\n{text}"
    );
    let reparsed = Sp3::parse(text.as_bytes()).unwrap();
    assert!(
        reparsed.state(id(GnssSystem::Gps, 2), 1).is_err(),
        "G02 must read back as absent at epoch 1"
    );
    assert!(
        reparsed.state(id(GnssSystem::Gps, 2), 0).is_ok(),
        "G02 is present at epoch 0"
    );
}

#[test]
fn writer_round_trips_a_real_full_day_fixture() {
    // CNES/CLS/GRGS MGEX final - 75 satellites (GPS + Galileo + GLONASS), 96
    // epochs at 15 min. The real-shape correctness bar.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read GRG fixture {path}: {e}"));
    let original = Sp3::parse(&bytes).expect("parse GRG fixture");

    assert_eq!(
        original.header.satellites.len(),
        75,
        "fixture satellite count"
    );
    assert_eq!(original.epochs.len(), 96, "fixture epoch count");

    assert_round_trip(&original);
}

#[test]
fn rejects_position_record_for_undeclared_satellite() {
    // Regression (fuzz sp3_round_trip): an SP3 whose header declares no
    // satellites but whose body carries a G01 position record. Accepting it
    // stored a state the writer (which emits only declared satellites) cannot
    // reproduce, breaking parse/encode/parse round-tripping. The parser must
    // reject the undeclared-satellite record.
    let bytes = include_bytes!(
        "../../../../fuzz/corpus/sp3_round_trip/regression-undeclared-sat-roundtrip"
    );
    let err = super::Sp3::parse(bytes).expect_err("undeclared-satellite SP3 must be rejected");
    match err {
        crate::error::Error::Parse(msg) => {
            assert!(
                msg.contains("not in the header satellite list"),
                "got: {msg}"
            );
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

/// A position record whose satellite token is out of range / unrepresentable
/// (an extended GLONASS slot `R28`, beyond the engine's 1..=27 PRN cap, as seen
/// in real BKG/IGS products) must be skipped and counted, not reject the whole
/// file. The surrounding GPS records must all survive.
#[test]
fn skips_out_of_range_satellite_position_record_and_counts_it() {
    const FILE: &str = "\
#cP2020  6 24  0  0  0.00000000       2 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-c FIXTURE
*  2020  6 24  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    123.456789
PR28  16000.000000 -21000.000000   6000.000000    222.222222
PG02  -1234.567890   2345.678901  -3456.789012    111.111111
EOF
";
    let sp3 = Sp3::parse(FILE.as_bytes()).expect("file with one out-of-range sat must still parse");
    assert_eq!(sp3.skipped_records, 1, "the R28 record must be counted");
    let states = sp3.states_at(0).expect("epoch 0 present");
    let present: Vec<u8> = states.keys().map(|s| s.prn).collect();
    assert_eq!(present, vec![1, 2], "both GPS records must survive");
    assert!(sp3.state(id(GnssSystem::Gps, 1), 0).is_ok());
    assert!(sp3.state(id(GnssSystem::Gps, 2), 0).is_ok());
}

/// An unrepresentable satellite token declared in the `+` header satellite list
/// (`R28`) must be dropped from the list but counted, not skipped silently - and
/// the positional `++` accuracy codes must stay aligned with the surviving
/// satellites (the dropped slot's column is skipped, not inherited by a
/// neighbour).
#[test]
fn skips_out_of_range_satellite_header_declaration_and_counts_it() {
    const FILE: &str = "\
#cP2020  6 24  0  0  0.00000000       2 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    3   G01R28G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         5  9 17  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-c FIXTURE
*  2020  6 24  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    123.456789
PG02  -1234.567890   2345.678901  -3456.789012    111.111111
EOF
";
    let sp3 = Sp3::parse(FILE.as_bytes())
        .expect("file with one out-of-range header sat must still parse");
    assert_eq!(
        sp3.skipped_records, 1,
        "the R28 header declaration must be counted"
    );
    let states = sp3.states_at(0).expect("epoch 0 present");
    let present: Vec<u8> = states.keys().map(|s| s.prn).collect();
    assert_eq!(present, vec![1, 2], "both representable GPS sats survive");
    // The dropped R28 column is skipped: G01 keeps 5, G02 keeps 17 (not 9).
    assert_eq!(
        sp3.header.satellite_accuracy_codes,
        vec![5, 17],
        "accuracy codes stay aligned with the surviving satellites"
    );
}

/// A velocity record whose satellite token is out of range / unrepresentable
/// (`R28`) must be skipped and counted, leaving the valid records intact.
#[test]
fn skips_out_of_range_satellite_velocity_record_and_counts_it() {
    const FILE: &str = "\
#dV2022  1  2  3  4  5.00000000       1 ORBIT IGS20 FIT  TST
## 2191 270245.00000000   300.00000000 59581 0.1281597222222
+    1   G05  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-d FIXTURE
*  2022  1  2  3  4  5.00000000
PG05  10000.000000  20000.000000  30000.000000    -50.000000
VG05  10000.000000 -20000.000000  30000.000000      1.000000
VR28  10000.000000 -20000.000000  30000.000000      1.000000
EOF
";
    let sp3 = Sp3::parse(FILE.as_bytes()).expect("file with one out-of-range V record must parse");
    assert_eq!(sp3.skipped_records, 1, "the VR28 record must be counted");
    let state = sp3.state(id(GnssSystem::Gps, 5), 0).expect("G05 present");
    assert!(state.velocity.is_some(), "G05 velocity must be applied");
}

/// Regression (fuzz `sp3_round_trip`): a real GBM multi-GNSS file whose unused
/// `+`-header satellite slots are zero-filled with ` 00` (not the canonical
/// `  0`). The parser must treat the all-zero tokens as padding, not as
/// unrepresentable satellites - otherwise they inflate `skipped_records` (to 13
/// here), which the writer (emitting canonical `  0` padding) cannot reproduce,
/// so the reparse diverges on `skipped_records`. Asserts full structural
/// `parse == parse(encode(parse))` equality.
#[test]
fn round_trips_plus_line_padded_with_00_zero_fill() {
    let bytes = include_bytes!("../../../../fuzz/corpus/sp3_round_trip/valid-gbm-bds-trim.sp3");
    let original = Sp3::parse(bytes).expect("GBM fixture must parse");
    assert_eq!(
        original.skipped_records, 0,
        "` 00` zero-fill slots are padding, not skipped satellites"
    );
    let encoded = original.to_sp3_string();
    let reparsed = Sp3::parse(encoded.as_bytes()).expect("re-encoded GBM fixture must reparse");
    assert_eq!(
        reparsed, original,
        "parse -> write -> parse changed product"
    );
}
