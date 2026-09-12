//! RINEX 3 observation parser tests against the committed ESBC00DNK fixture.

use super::*;
use crate::constants::{C_M_S, F_B1I_HZ, F_E5A_HZ, F_L1_HZ, F_L2_HZ};
use crate::crinex;

fn esbc_rnx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read RINEX fixture {path}: {e}"))
}

fn esbc_crx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read CRINEX fixture {path}: {e}"))
}

fn algo_v1_crx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.crx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read CRINEX v1 fixture {path}: {e}"))
}

fn algo_v1_rnx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read RINEX v2 fixture {path}: {e}"))
}

fn header_line(body: &str, label: &str) -> String {
    format!("{body:<60}{label}")
}

fn minimal_obs(extra_headers: &[String], body: &str) -> String {
    let mut lines = vec![
        header_line(
            "     3.05           OBSERVATION DATA    M (MIXED)",
            "RINEX VERSION / TYPE",
        ),
        header_line("G    1 C1C", "SYS / # / OBS TYPES"),
    ];
    lines.extend(extra_headers.iter().cloned());
    lines.push(header_line("", "END OF HEADER"));
    if !body.is_empty() {
        lines.extend(body.lines().map(str::to_string));
    }
    lines.join("\n")
}

fn obs_with_code_headers(code_headers: &[String], body: &str) -> String {
    obs_with_version_and_code_headers(3.05, code_headers, body)
}

fn obs_with_version_and_code_headers(version: f64, code_headers: &[String], body: &str) -> String {
    let version_line = format!("{version:9.2}           OBSERVATION DATA    M (MIXED)");
    let mut lines = vec![header_line(&version_line, "RINEX VERSION / TYPE")];
    lines.extend(code_headers.iter().cloned());
    lines.push(header_line("", "END OF HEADER"));
    if !body.is_empty() {
        lines.extend(body.lines().map(str::to_string));
    }
    lines.join("\n")
}

fn minimal_obs_with_phase_shift(body: &str) -> String {
    minimal_obs(&[header_line(body, "SYS / PHASE SHIFT")], "")
}

fn wrapped_obs_header() -> String {
    header_line("G    6 C1C L1C D1C S1C C2W L2W", "SYS / # / OBS TYPES")
}

fn obs_field(value: f64, lli: u8, ssi: u8) -> String {
    format!("{value:14.3}{lli}{ssi}")
}

fn blank_obs_field() -> String {
    " ".repeat(OBS_FIELD_WIDTH)
}

fn v2_epoch_line(epoch: ObsEpochTime, flag: u8, count: usize, sats: &str) -> String {
    let year = epoch.year % 100;
    format!(
        " {year:>2}{:>3}{:>3}{:>3}{:>3}{:11.7}{flag:>3}{count:>3}{sats}",
        epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second
    )
}

fn obs_fields(base: f64) -> Vec<String> {
    (0_u8..6)
        .map(|idx| obs_field(base + f64::from(idx), idx + 1, idx + 2))
        .collect()
}

fn wrapped_sat_record(sat: &str, fields: &[String]) -> String {
    format!(
        "{sat}{}{}{}{}\n   {}{}",
        fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]
    )
}

fn assert_wrapped_values(values: &[ObsValue], base: f64) {
    assert_eq!(values.len(), 6);
    for (idx, value) in values.iter().enumerate() {
        assert_eq!(value.value, Some(base + idx as f64));
        assert_eq!(value.lli, Some(idx as u8 + 1));
        assert_eq!(value.ssi, Some(idx as u8 + 2));
    }
}

fn only_phase_row(obs: &RinexObs) -> CarrierPhaseRow {
    let rows = carrier_phase_rows(obs, &obs.epochs()[0], &ObservationFilter::all())
        .expect("valid carrier-phase rows");
    assert_eq!(rows.len(), 1);
    let phases = &rows[0].1;
    assert_eq!(phases.len(), 1);
    phases[0].clone()
}

fn assert_parse_err(text: String) {
    let err = RinexObs::parse(&text).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{err}");
}

#[test]
fn parses_header_fields() {
    let obs = RinexObs::parse(&esbc_rnx()).expect("parse RINEX OBS");
    let h = obs.header();
    assert!((h.version - 3.05).abs() < 1e-9);
    let pos = h.approx_position_m.expect("approx position present");
    assert!((pos[0] - 3582105.2910).abs() < 1e-3);
    assert!((pos[1] - 532589.7313).abs() < 1e-3);
    assert!((pos[2] - 5232754.8054).abs() < 1e-3);
    let delta = h.antenna_delta_hen_m.expect("antenna delta H/E/N present");
    assert!((delta[0] - 0.2160).abs() < 1e-9);
    assert_eq!(delta[1], 0.0);
    assert_eq!(delta[2], 0.0);
    assert_eq!(h.marker_name.as_deref(), Some("ESBC00DNK"));
    assert_eq!(h.interval_s, Some(30.0));
    assert!(h.phase_shifts.len() >= 20);
    let gps_l1c = h
        .phase_shifts
        .iter()
        .find(|shift| shift.system == GnssSystem::Gps && shift.code == "L1C")
        .expect("GPS L1C phase shift");
    assert_eq!(gps_l1c.correction_cycles, 0.0);
    assert!(gps_l1c.satellites.is_empty());
    let gal_l5q = h
        .phase_shifts
        .iter()
        .find(|shift| shift.system == GnssSystem::Galileo && shift.code == "L5Q")
        .expect("Galileo L5Q phase shift");
    assert_eq!(gal_l5q.correction_cycles, 0.0);
    let (t0, scale) = h.time_of_first_obs.expect("time of first obs");
    assert_eq!(t0.year, 2020);
    assert_eq!(t0.month, 6);
    assert_eq!(t0.day, 25);
    assert_eq!(scale, TimeScale::Gpst);
}

#[test]
fn parses_per_system_obs_codes_in_order() {
    let obs = RinexObs::parse(&esbc_rnx()).expect("parse RINEX OBS");
    // GPS: 18 codes, first C1C.
    let gps = obs.obs_codes(GnssSystem::Gps).expect("GPS codes");
    assert_eq!(gps.len(), 18);
    assert_eq!(gps[0], "C1C");
    // BeiDou: 12 codes, first C2I (this 3.05 file uses the band-2 B1I label).
    let bds = obs.obs_codes(GnssSystem::BeiDou).expect("BeiDou codes");
    assert_eq!(bds.len(), 12);
    assert_eq!(bds[0], "C2I");
    // Galileo: 20 codes, first C1C.
    let gal = obs.obs_codes(GnssSystem::Galileo).expect("Galileo codes");
    assert_eq!(gal.len(), 20);
    assert_eq!(gal[0], "C1C");
}

#[test]
fn rejects_obs_type_count_mismatch_before_next_system() {
    assert_parse_err(obs_with_code_headers(
        &[
            header_line("G    3 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("R    1 C1C", "SYS / # / OBS TYPES"),
        ],
        "",
    ));
}

#[test]
fn rejects_obs_type_count_mismatch_at_header_end() {
    assert_parse_err(obs_with_code_headers(
        &[header_line("G    3 C1C L1C", "SYS / # / OBS TYPES")],
        "",
    ));
}

#[test]
fn accepts_obs_type_count_match() {
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("G    3 C1C L1C D1C", "SYS / # / OBS TYPES"),
            header_line("R    1 C1C", "SYS / # / OBS TYPES"),
        ],
        "",
    ))
    .expect("parse matching OBS type counts");

    assert_eq!(obs.obs_codes(GnssSystem::Gps).expect("GPS codes").len(), 3);
    assert_eq!(
        obs.obs_codes(GnssSystem::Glonass)
            .expect("GLONASS codes")
            .len(),
        1
    );
}

#[test]
fn parses_two_epochs_with_satellites() {
    let obs = RinexObs::parse(&esbc_rnx()).expect("parse RINEX OBS");
    assert_eq!(obs.epochs().len(), 2);
    let e0 = &obs.epochs()[0];
    assert_eq!(e0.flag, 0);
    assert_eq!(e0.sats.len(), 43);
    // A known GPS satellite carries a finite C1C pseudorange.
    let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).expect("valid satellite id");
    let g02_vals = e0.sats.get(&g02).expect("G02 present");
    assert!(g02_vals[0].value.unwrap() > 2.0e7);
}

#[test]
fn parses_wrapped_observation_record_for_one_satellite() {
    let fields = obs_fields(1001.0);
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\n{}",
        wrapped_sat_record("G01", &fields)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(&[wrapped_obs_header()], &body))
        .expect("parse wrapped one-satellite OBS");

    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let values = obs.epochs()[0].sats.get(&g01).expect("G01 present");
    assert_wrapped_values(values, 1001.0);
}

#[test]
fn parses_wrapped_observation_record_with_short_final_continuation_field() {
    let fields = obs_fields(1501.0);
    let short_final = format!("{:14.3}", 1506.0);
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}{}{}\n   {}{}",
        fields[0], fields[1], fields[2], fields[3], fields[4], short_final
    );
    let obs = RinexObs::parse(&obs_with_code_headers(&[wrapped_obs_header()], &body))
        .expect("parse wrapped OBS with short final continuation field");

    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let values = obs.epochs()[0].sats.get(&g01).expect("G01 present");
    assert_eq!(values.len(), 6);
    for (idx, value) in values.iter().take(5).enumerate() {
        assert_eq!(value.value, Some(1501.0 + idx as f64));
        assert_eq!(value.lli, Some(idx as u8 + 1));
        assert_eq!(value.ssi, Some(idx as u8 + 2));
    }
    assert_eq!(values[5].value, Some(1506.0));
    assert_eq!(values[5].lli, None);
    assert_eq!(values[5].ssi, None);
}

#[test]
fn parses_wrapped_observation_records_for_multiple_satellites() {
    let g01_fields = obs_fields(2001.0);
    let g02_fields = obs_fields(3001.0);
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  2\n{}\n{}",
        wrapped_sat_record("G01", &g01_fields),
        wrapped_sat_record("G02", &g02_fields)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(&[wrapped_obs_header()], &body))
        .expect("parse wrapped multi-satellite OBS");

    let epoch = &obs.epochs()[0];
    assert_eq!(epoch.sats.len(), 2);
    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).expect("valid satellite id");
    assert_wrapped_values(epoch.sats.get(&g01).expect("G01 present"), 2001.0);
    assert_wrapped_values(epoch.sats.get(&g02).expect("G02 present"), 3001.0);
}

#[test]
fn parses_wrapped_observation_record_with_non_ascii_column_without_panic() {
    let fields = obs_fields(4001.0);
    let mut first_line = format!("G01{}{}{}{}", fields[0], fields[1], fields[2], fields[3]);
    first_line.pop();
    first_line.push('é');
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\n{first_line}\n   {}{}",
        fields[4], fields[5]
    );
    let text = obs_with_code_headers(&[wrapped_obs_header()], &body);

    let result = std::panic::catch_unwind(|| RinexObs::parse(&text));
    assert!(result.is_ok(), "non-ASCII OBS column must not panic");
    let obs = result
        .unwrap()
        .expect("non-ASCII OBS column is replaced with a blank column");

    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let values = obs.epochs()[0].sats.get(&g01).expect("G01 present");
    assert_eq!(values.len(), 6);
    for (idx, value) in values.iter().enumerate() {
        assert_eq!(value.value, Some(4001.0 + idx as f64));
        assert_eq!(value.lli, Some(idx as u8 + 1));
        let expected_ssi = if idx == 3 { None } else { Some(idx as u8 + 2) };
        assert_eq!(value.ssi, expected_ssi);
    }
}

#[test]
fn pseudoranges_select_default_gps_code() {
    let obs = RinexObs::parse(&esbc_rnx()).expect("parse RINEX OBS");
    let policy = SignalPolicy::default_for(obs.header().version).expect("valid RINEX version");
    let prs = pseudoranges(&obs, &obs.epochs()[0], &policy).expect("valid pseudoranges");
    // Every returned satellite must be in the policy systems and carry a
    // plausible Earth-orbit pseudorange (1.9e7..4.2e7 m).
    assert!(!prs.is_empty());
    for (sat, range_m) in &prs {
        assert!(
            *range_m > 1.9e7 && *range_m < 4.3e7,
            "{sat} range {range_m}"
        );
    }
    // GPS-only override yields only GPS satellites.
    let gps_only = SignalPolicy {
        codes: [(GnssSystem::Gps, vec!["C1C".to_string()])]
            .into_iter()
            .collect(),
    };
    let gps_prs = pseudoranges(&obs, &obs.epochs()[0], &gps_only).expect("valid pseudoranges");
    assert!(gps_prs.iter().all(|(s, _)| s.system == GnssSystem::Gps));
    assert!(gps_prs.len() >= 8);
}

#[test]
fn beidou_default_is_version_aware() {
    // C2I in 3.01, C1I in 3.02, back to C2I in 3.03 and later.
    let v301 = SignalPolicy::default_for(3.01).expect("valid RINEX version");
    assert_eq!(v301.codes[&GnssSystem::BeiDou][0], "C2I");
    let v302 = SignalPolicy::default_for(3.02).expect("valid RINEX version");
    assert_eq!(v302.codes[&GnssSystem::BeiDou][0], "C1I");
    let v303 = SignalPolicy::default_for(3.03).expect("valid RINEX version");
    assert_eq!(v303.codes[&GnssSystem::BeiDou][0], "C2I");
    let v305 = SignalPolicy::default_for(3.05).expect("valid RINEX version");
    assert_eq!(v305.codes[&GnssSystem::BeiDou][0], "C2I");
}

#[test]
fn convenience_helpers_reject_non_finite_versions() {
    assert!(matches!(
        SignalPolicy::default_for(f64::NAN),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        observation_frequency_hz(GnssSystem::Gps, "L1C", f64::INFINITY, None),
        Err(Error::InvalidInput(_))
    ));

    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(10.0, 0, 0)
    );
    let mut obs = RinexObs::parse(&obs_with_code_headers(
        &[header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
        &body,
    ))
    .expect("parse carrier-phase OBS");
    obs.header.version = f64::NAN;

    assert!(matches!(
        carrier_phase_rows(&obs, &obs.epochs()[0], &ObservationFilter::all()),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn convenience_helpers_reject_non_finite_values() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(10.0, 0, 0)
    );
    let mut obs = RinexObs::parse(&obs_with_code_headers(
        &[header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
        &body,
    ))
    .expect("parse observation OBS");
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    obs.epochs[0].sats.get_mut(&sat).expect("G01 present")[0].value = Some(f64::NAN);
    let policy = SignalPolicy {
        codes: [(GnssSystem::Gps, vec!["C1C".to_string()])]
            .into_iter()
            .collect(),
    };

    assert!(matches!(
        observation_values(&obs, &obs.epochs()[0], &ObservationFilter::all()),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        pseudoranges(&obs, &obs.epochs()[0], &policy),
        Err(Error::InvalidInput(_))
    ));

    obs.epochs[0].sats.get_mut(&sat).expect("G01 present")[0].value = Some(22_000_000.0);
    obs.epochs[0].sats.get_mut(&sat).expect("G01 present")[1].value = Some(f64::INFINITY);
    assert!(matches!(
        carrier_phase_rows(&obs, &obs.epochs()[0], &ObservationFilter::all()),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn carrier_phase_rows_use_beidou_version_aware_wavelengths() {
    let body_302 = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nC01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(10.0, 0, 0)
    );
    let obs_302 = RinexObs::parse(&obs_with_version_and_code_headers(
        3.02,
        &[header_line("C    2 C1I L1I", "SYS / # / OBS TYPES")],
        &body_302,
    ))
    .expect("parse RINEX 3.02 BeiDou B1I OBS");
    let row_302 = only_phase_row(&obs_302);
    let lambda_302 = C_M_S / F_B1I_HZ;

    assert_eq!(row_302.code, "L1I");
    assert_eq!(
        row_302.frequency_hz.map(f64::to_bits),
        Some(F_B1I_HZ.to_bits())
    );
    assert_eq!(
        row_302.wavelength_m.map(f64::to_bits),
        Some(lambda_302.to_bits())
    );
    assert_eq!(
        row_302.value_m.map(f64::to_bits),
        Some((10.0 * lambda_302).to_bits())
    );

    let body_303 = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nC01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(10.0, 0, 0)
    );
    let obs_303 = RinexObs::parse(&obs_with_version_and_code_headers(
        3.03,
        &[header_line("C    2 C1X L1X", "SYS / # / OBS TYPES")],
        &body_303,
    ))
    .expect("parse RINEX 3.03 BeiDou B1C OBS");
    let row_303 = only_phase_row(&obs_303);

    assert_eq!(row_303.code, "L1X");
    assert_eq!(
        row_303.frequency_hz.map(f64::to_bits),
        Some(F_L1_HZ.to_bits())
    );
    assert_eq!(
        row_303.wavelength_m.map(f64::to_bits),
        Some((C_M_S / F_L1_HZ).to_bits())
    );
}

#[test]
fn carrier_phase_rows_include_qzss_l1_l2_l5_metadata() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nJ01{}{}{}{}{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(10.0, 1, 2),
        obs_field(22_000_001.0, 0, 0),
        obs_field(20.0, 3, 4),
        obs_field(22_000_002.0, 0, 0),
        obs_field(30.0, 5, 6)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[header_line(
            "J    6 C1C L1C C2L L2L C5Q L5Q",
            "SYS / # / OBS TYPES",
        )],
        &body,
    ))
    .expect("parse QZSS carrier-phase OBS");

    let rows = carrier_phase_rows(&obs, &obs.epochs()[0], &ObservationFilter::all())
        .expect("valid carrier-phase rows");
    assert_eq!(rows.len(), 1);
    let (sat, phases) = &rows[0];
    assert_eq!(
        *sat,
        GnssSatelliteId::new(GnssSystem::Qzss, 1).expect("valid satellite id")
    );
    assert_eq!(phases.len(), 3);

    for (row, expected_code, expected_cycles, expected_lli, expected_ssi, expected_frequency) in [
        (&phases[0], "L1C", 10.0_f64, Some(1), Some(2), F_L1_HZ),
        (&phases[1], "L2L", 20.0_f64, Some(3), Some(4), F_L2_HZ),
        (&phases[2], "L5Q", 30.0_f64, Some(5), Some(6), F_E5A_HZ),
    ] {
        let expected_wavelength = C_M_S / expected_frequency;
        assert_eq!(row.code, expected_code);
        assert_eq!(
            row.value_cycles.map(f64::to_bits),
            Some(expected_cycles.to_bits())
        );
        assert_eq!(row.lli, expected_lli);
        assert_eq!(row.ssi, expected_ssi);
        assert_eq!(
            row.frequency_hz.map(f64::to_bits),
            Some(expected_frequency.to_bits())
        );
        assert_eq!(
            row.wavelength_m.map(f64::to_bits),
            Some(expected_wavelength.to_bits())
        );
        assert_eq!(
            row.value_m.map(f64::to_bits),
            Some((expected_cycles * expected_wavelength).to_bits())
        );
    }
}

#[test]
fn carrier_phase_rows_use_recorded_cycles_when_phase_shift_header_is_nonzero() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(123_456.25, 1, 7)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("G L1C  0.25000", "SYS / PHASE SHIFT"),
        ],
        &body,
    ))
    .expect("parse shifted carrier-phase OBS");

    let row = only_phase_row(&obs);

    assert_eq!(row.code, "L1C");
    assert_eq!(row.phase_shift_cycles.to_bits(), 0.25_f64.to_bits());
    assert_eq!(
        row.value_cycles.map(f64::to_bits),
        Some(123_456.25_f64.to_bits())
    );
    assert_eq!(
        row.value_m.map(f64::to_bits),
        Some((123_456.25 * (C_M_S / F_L1_HZ)).to_bits())
    );
    assert_eq!(row.lli, Some(1));
    assert_eq!(row.ssi, Some(7));
}

#[test]
fn carrier_phase_rows_without_phase_shift_header_use_recorded_cycles() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(123_456.25, 1, 7)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
        &body,
    ))
    .expect("parse unshifted carrier-phase OBS");

    let row = only_phase_row(&obs);

    assert_eq!(row.code, "L1C");
    assert_eq!(row.phase_shift_cycles.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        row.value_cycles.map(f64::to_bits),
        Some(123_456.25_f64.to_bits())
    );
    assert_eq!(
        row.value_m.map(f64::to_bits),
        Some((123_456.25 * (C_M_S / F_L1_HZ)).to_bits())
    );
    assert_eq!(row.lli, Some(1));
    assert_eq!(row.ssi, Some(7));
}

#[test]
fn obs_scale_factor_divides_selected_observation_values() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(123_456.0, 1, 7)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("G   10  1 L1C", "SYS / SCALE FACTOR"),
        ],
        &body,
    ))
    .expect("parse selected scale-factor OBS");

    let values = &obs.epochs()[0].sats
        [&GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id")];
    assert_eq!(values[0].value, Some(22_000_000.0));
    assert!((values[1].value.unwrap() - 12_345.6).abs() < 1e-9);

    let scale = &obs.header().scale_factors[0];
    assert_eq!(scale.system, GnssSystem::Gps);
    assert_eq!(scale.factor.to_bits(), 10.0_f64.to_bits());
    assert_eq!(scale.codes, vec![String::from("L1C")]);
}

#[test]
fn obs_scale_factor_count_zero_divides_all_system_observation_values() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nG01{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(123_456.0, 1, 7)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("G  100  0", "SYS / SCALE FACTOR"),
        ],
        &body,
    ))
    .expect("parse all-code scale-factor OBS");

    let values = &obs.epochs()[0].sats
        [&GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id")];
    assert_eq!(values[0].value, Some(220_000.0));
    assert!((values[1].value.unwrap() - 1_234.56).abs() < 1e-9);

    let scale = &obs.header().scale_factors[0];
    assert_eq!(scale.system, GnssSystem::Gps);
    assert_eq!(scale.factor.to_bits(), 100.0_f64.to_bits());
    assert!(scale.codes.is_empty());
}

#[test]
fn parses_crinex_decoded_text_identically() {
    // Decoding the CRINEX and parsing the result must agree with parsing the
    // committed reference RINEX (the full chain the sidereon loader runs).
    let decoded = crinex::decode(&esbc_crx()).expect("decode CRINEX");
    let from_crx = RinexObs::parse(&decoded).expect("parse decoded");
    let from_rnx = RinexObs::parse(&esbc_rnx()).expect("parse reference");
    assert_eq!(from_crx, from_rnx);
}

#[test]
fn parses_plain_rinex2_observation_file() {
    let body = format!(
        "{}\n{}{}{}{}{}\n{}",
        v2_epoch_line(
            ObsEpochTime {
                year: 2020,
                month: 1,
                day: 2,
                hour: 3,
                minute: 4,
                second: 5.0,
            },
            0,
            1,
            "G 1",
        ),
        obs_field(123_456.789, 1, 2),
        obs_field(234_567.891, 3, 4),
        blank_obs_field(),
        obs_field(20_200_000.125, 0, 0),
        obs_field(45.0, 0, 5),
        blank_obs_field(),
    );
    let text = [
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line(
            "     6    L1    L2    C1    P1    S1    S2",
            "# / TYPES OF OBSERV",
        ),
        header_line(
            "  2020     1     2     3     4    5.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
        header_line("", "END OF HEADER"),
        body,
    ]
    .join("\n");

    let obs = RinexObs::parse(&text).expect("parse hand-built RINEX 2 OBS");
    assert!((obs.header().version - 2.11).abs() < 1e-9);
    assert_eq!(
        obs.obs_codes(GnssSystem::Gps).expect("GPS code table"),
        &[
            "L1C".to_string(),
            "L2W".to_string(),
            "C1C".to_string(),
            "C1W".to_string(),
            "S1C".to_string(),
            "S2W".to_string(),
        ]
    );
    assert_eq!(obs.epochs().len(), 1);
    let epoch = &obs.epochs()[0];
    assert_eq!(epoch.epoch.year, 2020);
    assert_eq!(epoch.declared_record_count, 1);
    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let values = epoch.sats.get(&g01).expect("G01 present");
    assert_eq!(values.len(), 6);
    assert_eq!(values[0].value, Some(123_456.789));
    assert_eq!(values[0].lli, Some(1));
    assert_eq!(values[0].ssi, Some(2));
    assert_eq!(values[2].value, None);
    assert_eq!(values[3].value, Some(20_200_000.125));
    assert_eq!(values[4].value, Some(45.0));
    assert_eq!(values[4].ssi, Some(5));
    assert_eq!(values[5].value, None);
}

#[test]
fn parses_crinex_v1_decoded_rinex2_into_observations() {
    let decoded = crinex::decode(&algo_v1_crx()).expect("decode CRINEX v1");
    let from_crx = RinexObs::parse(&decoded).expect("parse decoded RINEX 2");
    let from_rnx = RinexObs::parse(&algo_v1_rnx()).expect("parse reference RINEX 2");
    assert_eq!(from_crx, from_rnx);

    assert!((from_crx.header().version - 2.11).abs() < 1e-9);
    assert_eq!(from_crx.epochs().len(), 2);
    assert_eq!(from_crx.epochs()[0].declared_record_count, 20);
    assert_eq!(from_crx.epochs()[0].sats.len(), 20);
    assert_eq!(from_crx.epochs()[1].declared_record_count, 19);
    assert_eq!(from_crx.epochs()[1].sats.len(), 19);
    assert_eq!(
        from_crx.obs_codes(GnssSystem::Gps).expect("GPS code table"),
        &[
            "L1C".to_string(),
            "L2W".to_string(),
            "C1C".to_string(),
            "C2C".to_string(),
            "C2W".to_string(),
            "C1W".to_string(),
            "S1C".to_string(),
            "S2W".to_string(),
        ]
    );
    assert_eq!(
        from_crx
            .obs_codes(GnssSystem::Glonass)
            .expect("GLONASS code table"),
        &[
            "L1C".to_string(),
            "L2P".to_string(),
            "C1C".to_string(),
            "C2C".to_string(),
            "C2P".to_string(),
            "C1P".to_string(),
            "S1C".to_string(),
            "S2P".to_string(),
        ]
    );

    let g08 = GnssSatelliteId::new(GnssSystem::Gps, 8).expect("valid satellite id");
    let g08_values = from_crx.epochs()[0].sats.get(&g08).expect("G08 present");
    assert_eq!(g08_values[0].value, Some(118_504_127.181));
    assert_eq!(g08_values[0].lli, Some(4));
    assert_eq!(g08_values[0].ssi, Some(7));
    assert_eq!(g08_values[3].value, None);
    assert_eq!(g08_values[4].value, Some(22_550_574.970));
    assert_eq!(g08_values[5].value, Some(22_550_575.149));
    assert_eq!(g08_values[6].value, Some(47.250));
    assert_eq!(g08_values[7].value, Some(37.250));

    let r07 = GnssSatelliteId::new(GnssSystem::Glonass, 7).expect("valid satellite id");
    let r07_values = from_crx.epochs()[0].sats.get(&r07).expect("R07 present");
    assert_eq!(r07_values[2].value, Some(21_290_875.138));
    assert_eq!(r07_values[3].value, Some(21_290_870.931));
    assert_eq!(r07_values[4].value, Some(21_290_871.206));
    assert_eq!(r07_values[5].value, Some(21_290_874.848));
}

#[test]
fn rejects_v2_epoch_count_that_exceeds_its_i3_field() {
    // Exact input from scheduled fuzz run 29188441671. Before the count bound,
    // the parser passed 155_444_444_444_444 to Vec::with_capacity and aborted
    // under AddressSanitizer before it could report the malformed epoch.
    const CRASH: &[u8] =
        b"5 10 5 5 0 5 0 155444444444444\xff\xff\xaa\xaa\xaa\xaa\xaa\xaa\xaa\n4444444445\0\0\0\0\
000000000000000000000000000031770ttttv";
    assert_eq!(CRASH.len(), 92, "scheduled-fuzz artifact length");
    let body = String::from_utf8_lossy(CRASH);
    let text = [
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line(
            "     6    L1    L2    C1    P1    S1    S2",
            "# / TYPES OF OBSERV",
        ),
        header_line("", "END OF HEADER"),
        body.into_owned(),
    ]
    .join("\n");

    let err = RinexObs::parse(&text).expect_err("an over-width epoch count must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message) if message.contains("I3 field maximum of 999")),
        "{err}"
    );
}

#[test]
fn epoch_record_count_accepts_i3_maximum_and_rejects_overflow() {
    assert_eq!(
        parse_epoch_record_count("999", "epoch"),
        Ok(MAX_EPOCH_RECORD_COUNT)
    );
    for token in ["1000", "155444444444444"] {
        let err = parse_epoch_record_count(token, "epoch")
            .expect_err("a count wider than the I3 field must be rejected");
        assert!(
            matches!(err, Error::Parse(ref message) if message.contains("I3 field maximum of 999")),
            "{token}: {err}"
        );
    }
}

#[test]
fn rejects_v3_epoch_count_that_exceeds_its_i3_field() {
    let err = RinexObs::parse(&minimal_obs(&[], "> 2020 06 25 00 00 00.0000000  0  1000"))
        .expect_err("an over-width RINEX-3 epoch count must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message) if message.contains("I3 field maximum of 999")),
        "{err}"
    );
}

#[test]
fn rejects_obs_type_code_wider_than_its_a3_field() {
    // Exact shape from scheduled fuzz run 30197879510: a first SYS / # / OBS
    // TYPES record whose single "code" spills across the whole content area,
    // then a second record for the same system. The over-width descriptor
    // cannot be written back into a 1X,A3 field, so the re-emitted record
    // overran its 60 columns and re-parsed one code short of its own count.
    let long_code = format!("C{}", "t".repeat(51));
    let text = obs_with_code_headers(
        &[
            header_line(&format!("G    1 {long_code}"), "SYS / # / OBS TYPES"),
            header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        ],
        "",
    );

    let err = RinexObs::parse(&text).expect_err("an over-width obs descriptor must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("SYS / # / OBS TYPES code")
                && message.contains("exceeds the A3 field width")),
        "{err}"
    );
}

#[test]
fn accepts_obs_type_codes_up_to_the_a3_field_width() {
    let text = obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("R    1  C1", "SYS / # / OBS TYPES"),
        ],
        "",
    );
    let obs = RinexObs::parse(&text).expect("A3-width descriptors stay accepted");
    assert_eq!(
        obs.obs_codes(GnssSystem::Gps).expect("GPS codes"),
        ["C1C".to_string(), "L1C".to_string()]
    );
    assert_eq!(
        obs.obs_codes(GnssSystem::Glonass).expect("GLONASS codes"),
        ["C1".to_string()]
    );
}

#[test]
fn rejects_scale_factor_code_wider_than_its_a3_field() {
    let header = header_line("G    1   1 C1CX", "SYS / SCALE FACTOR");
    let err = RinexObs::parse(&minimal_obs(&[header], ""))
        .expect_err("an over-width scale-factor descriptor must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("SYS / SCALE FACTOR code")
                && message.contains("exceeds the A3 field width")),
        "{err}"
    );
}

#[test]
fn rejects_phase_shift_code_wider_than_its_a3_field() {
    let err = RinexObs::parse(&minimal_obs_with_phase_shift("G L1CX 0.0 1 G01"))
        .expect_err("an over-width phase-shift descriptor must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("SYS / PHASE SHIFT code")
                && message.contains("exceeds the A3 field width")),
        "{err}"
    );
}

#[test]
fn rejects_v2_obs_type_code_wider_than_its_a3_field() {
    let text = obs_with_version_and_code_headers(
        2.11,
        &[header_line("     1    C1CX", "# / TYPES OF OBSERV")],
        "",
    );
    let err =
        RinexObs::parse(&text).expect_err("an over-width RINEX-2 descriptor must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("# / TYPES OF OBSERV code")
                && message.contains("exceeds the A3 field width")),
        "{err}"
    );
}

#[test]
fn rejects_obs_type_count_that_exceeds_its_i3_field() {
    // The count sits in a 3-column field, so an over-wide count only reaches
    // the parser through the whitespace-tolerant form of the record.
    let text = obs_with_code_headers(&[header_line("G      1000 C1C", "SYS / # / OBS TYPES")], "");
    let err = RinexObs::parse(&text).expect_err("an over-width obs-type count must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("declares 1000 codes")
                && message.contains("I3 field maximum of 999")),
        "{err}"
    );
}

#[test]
fn rejects_v2_obs_type_count_that_exceeds_the_i3_field_it_is_written_to() {
    let text = obs_with_version_and_code_headers(
        2.11,
        &[header_line("  1000    C1", "# / TYPES OF OBSERV")],
        "",
    );
    let err = RinexObs::parse(&text)
        .expect_err("a RINEX-2 count beyond the RINEX-3 I3 field must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("# / TYPES OF OBSERV declares 1000 codes")
                && message.contains("SYS / # / OBS TYPES I3 field can carry")),
        "{err}"
    );
}

#[test]
fn conforming_phase_shift_corrections_keep_their_plain_decimal() {
    for (body, expected) in [
        ("G L1C 0.0 1 G01", "G L1C 0 1 G01"),
        ("G L1C 0.25000", "G L1C 0.25"),
        ("G L1C -0.25000", "G L1C -0.25"),
        ("G L1C 0.12345", "G L1C 0.12345"),
    ] {
        let obs = RinexObs::parse(&minimal_obs_with_phase_shift(body)).expect("parse phase shift");
        let written = obs.to_rinex_string();
        let line = written
            .lines()
            .find(|line| line.contains("SYS / PHASE SHIFT"))
            .expect("phase-shift record written");
        assert_eq!(
            line.trim_end_matches("SYS / PHASE SHIFT").trim_end(),
            expected
        );
    }
}

#[test]
fn phase_shift_correction_far_from_unity_round_trips_in_exponent_form() {
    // `Display` renders 1e-300 as 302 columns of plain decimal, which the
    // 60-column content area used to truncate: the value collapsed to zero and
    // the satellite list vanished, so the product no longer round-tripped.
    let obs = RinexObs::parse(&minimal_obs_with_phase_shift("G L1C 1e-300 1 G01"))
        .expect("parse phase shift far from unity");
    let written = obs.to_rinex_string();
    let line = written
        .lines()
        .find(|line| line.contains("SYS / PHASE SHIFT"))
        .expect("phase-shift record written");
    assert_eq!(line.len(), 60 + "SYS / PHASE SHIFT".len());

    let reparsed = RinexObs::parse(&written).expect("reparse phase shift far from unity");
    let shift = &reparsed.header().phase_shifts[0];
    assert_eq!(shift.correction_cycles, 1e-300);
    assert_eq!(shift.satellites.len(), 1);
    assert_eq!(reparsed.to_rinex_string().as_bytes(), written.as_bytes());
}

#[test]
fn rejects_phase_shift_satellite_list_that_cannot_be_written() {
    // Single-digit PRNs arrive in two-column tokens, so 14 satellites fit the
    // 60 columns the parser reads; re-emitted as `1X,A3` fields they need 66.
    let sats: Vec<String> = (1..=14).map(|prn| format!("G{prn}")).collect();
    let body = format!("G C1C 0 14 {}", sats.join(" "));
    assert_eq!(body.len(), 57, "the record must be readable in 60 columns");

    let err = RinexObs::parse(&minimal_obs_with_phase_shift(&body))
        .expect_err("an unwritable phase-shift record must be rejected");
    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("SYS / PHASE SHIFT record needs")
                && message.contains("exceeding the 60")),
        "{err}"
    );
}

#[test]
fn rejects_malformed_phase_shift_headers() {
    for body in [
        "G L1C bad",
        "G L1C 0.0 count",
        "G L1C NaN",
        "G L1C 0.0 2 G01",
        "G L1C 0.0 1 BAD",
    ] {
        let err = RinexObs::parse(&minimal_obs_with_phase_shift(body)).unwrap_err();
        assert!(matches!(err, Error::Parse(_)), "{body}: {err}");
    }
}

#[test]
fn rejects_malformed_receiver_metadata_numbers() {
    for header in [
        header_line(
            "  3582105.2910   not-a-number  5232754.8054",
            "APPROX POSITION XYZ",
        ),
        header_line(
            "        NaN        0.0000        0.0000",
            "ANTENNA: DELTA H/E/N",
        ),
        header_line("    bad", "INTERVAL"),
    ] {
        assert_parse_err(minimal_obs(&[header], ""));
    }
}

#[test]
fn blank_optional_interval_is_parsed_as_unavailable() {
    let obs = RinexObs::parse(&minimal_obs(&[header_line("", "INTERVAL")], ""))
        .expect("RINEX permits an unknown optional header item to be blank");
    assert_eq!(obs.header().interval_s, None);
}

#[test]
fn zero_optional_interval_is_retained_as_unavailable_metadata() {
    let obs = RinexObs::parse(&minimal_obs(&[header_line("     0.000", "INTERVAL")], ""))
        .expect("RINEX permits an unknown optional header item to be zero");
    assert_eq!(obs.header().interval_s, Some(0.0));
}

#[test]
fn rejects_header_numbers_a_fixed_column_field_cannot_re_emit() {
    // Every one of these parses as a finite f64 but cannot survive the writer's
    // own fixed-column format, so accepting it would mean emitting a file that
    // reads back as a different product. INTERVAL is F10.3, the position and
    // antenna-delta components are F14.4, and the header seconds are F13.7.
    for header in [
        // Below the field's resolution: the writer would emit 0.000, which
        // RINEX reads back as the "unknown" zero rather than the tiny value.
        header_line("    1e-300", "INTERVAL"),
        header_line("    0.0004", "INTERVAL"),
        // Too wide for the field: the writer would overrun the ten columns.
        header_line("     1e300", "INTERVAL"),
        header_line(
            "        1e-300         0.0         0.0",
            "APPROX POSITION XYZ",
        ),
        header_line(
            "       0.00004         0.0         0.0",
            "APPROX POSITION XYZ",
        ),
        header_line(
            "         1e300         0.0         0.0",
            "APPROX POSITION XYZ",
        ),
        header_line(
            "        1e-300      0.0000      0.0000",
            "ANTENNA: DELTA H/E/N",
        ),
        header_line(
            "         1e300      0.0000      0.0000",
            "ANTENNA: DELTA H/E/N",
        ),
        header_line(
            "  2020     6    24     0     0   0.00000001     GPS",
            "TIME OF FIRST OBS",
        ),
    ] {
        let err = RinexObs::parse(&minimal_obs(std::slice::from_ref(&header), ""))
            .expect_err("a value its field cannot re-emit must not parse");
        let message = err.to_string();
        assert!(
            message.contains("is not representable in its F"),
            "{header:?} was rejected for an unrelated reason: {message}"
        );
    }
}

#[test]
fn header_numbers_on_their_field_grid_round_trip_exactly() {
    for header in [
        header_line("     0.000", "INTERVAL"),
        header_line("     0.001", "INTERVAL"),
        header_line("    30.000", "INTERVAL"),
        header_line(
            "  3582105.2910   532589.7313  5232754.8054",
            "APPROX POSITION XYZ",
        ),
        header_line(
            "        0.0000      0.0000      0.0000",
            "ANTENNA: DELTA H/E/N",
        ),
        header_line(
            "  2020     6    24     0     0    0.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
    ] {
        let text = minimal_obs(std::slice::from_ref(&header), "");
        let obs = RinexObs::parse(&text).expect("a value on the field grid parses");
        let reparsed =
            RinexObs::parse(&obs.to_rinex_string()).expect("re-encoded RINEX OBS must reparse");
        assert_eq!(reparsed, obs, "{header:?} did not survive a round trip");
    }
}

#[test]
fn rejects_record_numbers_a_fixed_column_field_cannot_re_emit() {
    // The remaining fixed-format numbers the writer re-emits: the version
    // (F20.2), a GLONASS bias (F8.3), and an epoch record's seconds (F11.7),
    // receiver clock offset (F15.12) and observation values (F14.3). Each of
    // these parses as a finite f64 but cannot survive its own field.
    let version = minimal_obs(&[], "").replace("     3.05  ", "     3.999 ");
    let cases = [
        ("version", version),
        (
            "glonass_code_phase_bias",
            minimal_obs(&[header_line(" C1C   1e-300", "GLONASS COD/PHS/BIS")], ""),
        ),
        (
            "epoch.second",
            minimal_obs(
                &[],
                "> 2020 06 24 00 00 59.99999999  0  1\nG01        23000000.000",
            ),
        ),
        (
            "epoch.rcv_clock_offset_s",
            minimal_obs(
                &[],
                "> 2020 06 24 00 00  0.0000000  0  1 1e-13\nG01        23000000.000",
            ),
        ),
        (
            // Wider than its fifteen columns: the offset would run into the
            // satellite count on the next read.
            "epoch.rcv_clock_offset_s",
            minimal_obs(
                &[],
                "> 2020 06 24 00 00  0.0000000  0  1 -123.456789012345\nG01        23000000.000",
            ),
        ),
        (
            "observation value",
            minimal_obs(
                &[],
                "> 2020 06 24 00 00  0.0000000  0  1\nG01        1e-300",
            ),
        ),
    ];
    for (field, text) in cases {
        let err =
            RinexObs::parse(&text).expect_err("a value its field cannot re-emit must not parse");
        let message = err.to_string();
        assert!(
            message.contains("is not representable in its F") && message.contains(field),
            "{field} was rejected for an unrelated reason: {message}"
        );
    }
}

#[test]
fn adjacent_vector_header_columns_are_read_as_the_writer_wrote_them() {
    // Three F14.4 columns leave no separator when a component fills its field,
    // so a -10,000,000 m coordinate writes as `0.0000-10000000.0000`. Reading
    // the writer's columns recovers it; whitespace splitting saw one token.
    let text = minimal_obs(
        &[header_line(
            "           0.0  -10000000.0           0.0",
            "APPROX POSITION XYZ",
        )],
        "",
    );
    let obs = RinexObs::parse(&text).expect("parse an adjacent-column position");
    assert_eq!(
        obs.header().approx_position_m,
        Some([0.0, -10_000_000.0, 0.0])
    );

    let encoded = obs.to_rinex_string();
    let line = encoded
        .lines()
        .find(|line| line.contains("APPROX POSITION XYZ"))
        .expect("the position is written");
    assert!(
        line.contains("0.0000-10000000.0000"),
        "the columns really do abut: {line:?}"
    );
    let reparsed = RinexObs::parse(&encoded).expect("re-encoded RINEX OBS must reparse");
    assert_eq!(reparsed, obs);
}

#[test]
fn scaled_observations_round_trip_at_their_own_scale() {
    // Scaling is not exactly invertible in binary floating point: a file value
    // of 123456.001 at scale 10 stores 12345.6001 and multiplies back to
    // 123456.00099999999. What has to round trip is the value the next read
    // recovers, not that intermediate product.
    //
    // `SYS / SCALE FACTOR` puts the system at column 0, the factor in columns
    // 2..6 and the code count in columns 8..10, and an observation record puts
    // its F14.3 value in the fourteen columns after the satellite id, so these
    // fixtures are laid out by column rather than by eye.
    for (scale_header, scale, file_value, stored) in [
        (
            "G 1000  1 C1C",
            1000.0_f64,
            "133379507.327",
            133_379.507_327_f64,
        ),
        ("G   10  1 C1C", 10.0_f64, "123456.001", 12_345.600_1_f64),
    ] {
        let text = minimal_obs(
            &[header_line(scale_header, "SYS / SCALE FACTOR")],
            &format!("> 2020 06 24 00 00  0.0000000  0  1\nG01{file_value:>14}"),
        );
        let obs = RinexObs::parse(&text)
            .unwrap_or_else(|error| panic!("scale {scale} value {file_value} must parse: {error}"));

        // Prove the fixture is read as intended before trusting the round trip.
        let factor = &obs.header().scale_factors[0];
        assert_eq!(factor.system, GnssSystem::Gps);
        assert_eq!(factor.factor.to_bits(), scale.to_bits());
        assert_eq!(factor.codes, vec![String::from("C1C")]);
        let values = &obs.epochs()[0].sats
            [&GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id")];
        assert_eq!(values[0].value, Some(stored), "scale {scale}");
        assert_eq!(
            values[0].lli, None,
            "scale {scale}: the value must not spill into LLI"
        );
        assert_eq!(
            values[0].ssi, None,
            "scale {scale}: the value must not spill into SSI"
        );

        let reparsed =
            RinexObs::parse(&obs.to_rinex_string()).expect("re-encoded RINEX OBS must reparse");
        assert_eq!(
            reparsed, obs,
            "scale {scale} value {file_value} did not round trip"
        );
    }
}

#[test]
fn a_full_width_clock_offset_keeps_its_reserved_columns() {
    // RINEX reserves six columns between the satellite count and the clock
    // offset. Without them a full-width negative offset abuts the count and the
    // epoch line no longer reads back.
    let text = minimal_obs(
        &[],
        "> 2020 06 24 00 00  0.0000000  0  1 -0.000000000001\nG01        23000000.000",
    );
    let obs = RinexObs::parse(&text).expect("parse a full-width clock offset");
    let encoded = obs.to_rinex_string();
    let epoch_line = encoded
        .lines()
        .find(|line| line.starts_with('>'))
        .expect("the epoch line is written");
    assert!(
        epoch_line.contains("  1      -0.000000000001"),
        "the reserved columns are missing: {epoch_line:?}"
    );
    let reparsed = RinexObs::parse(&encoded).expect("re-encoded RINEX OBS must reparse");
    assert_eq!(reparsed, obs);
}

#[test]
fn loosely_spaced_vector_headers_are_still_accepted() {
    // Files that do not lay the components on the writer's columns have always
    // parsed, and still do.
    for (body, expected) in [
        ("1.0 2.0 3.0", [1.0, 2.0, 3.0]),
        // Fifteen-wide columns: every fourteen-column slice would also parse as
        // a number, so reading the writer's columns first would silently return
        // [1, 2, 34].
        (
            "             12             34             56",
            [12.0, 34.0, 56.0],
        ),
    ] {
        let text = minimal_obs(&[header_line(body, "APPROX POSITION XYZ")], "");
        let obs = RinexObs::parse(&text).expect("parse a loosely spaced position");
        assert_eq!(obs.header().approx_position_m, Some(expected), "{body:?}");
    }
}

/// Satellite ids across four constellations, and the `SYS / # / OBS TYPES`
/// headers that declare them.
fn multi_constellation_fixture(count: usize) -> (Vec<String>, Vec<String>) {
    let satellites: Vec<String> = (1..=32)
        .map(|prn| format!("G{prn:02}"))
        .chain((1..=24).map(|prn| format!("R{prn:02}")))
        .chain((1..=36).map(|prn| format!("E{prn:02}")))
        .chain((1..=63).map(|prn| format!("C{prn:02}")))
        .take(count)
        .collect();
    assert_eq!(satellites.len(), count);
    let systems = ["G", "R", "E", "C"]
        .iter()
        .map(|system| header_line(&format!("{system}    1 C1C"), "SYS / # / OBS TYPES"))
        .collect();
    (satellites, systems)
}

fn epoch_of(count: usize, flag: u8, picoseconds: &str, clock: &str) -> String {
    let (satellites, systems) = multi_constellation_fixture(count);
    let mut body = format!("> 2020 06 24 00 00  0.0000000{picoseconds}  {flag}{count:3}{clock}");
    for satellite in &satellites {
        body.push_str(&format!("\n{satellite}   23000000.000"));
    }
    obs_with_code_headers(&systems, &body)
}

#[test]
fn an_epoch_of_a_hundred_satellites_separates_its_flag_from_its_count() {
    // The epoch flag is `I1` and the satellite count `I3`, in adjacent columns,
    // so a count of 100 or more leaves no separator: a conforming line reads
    // `0100`, which whitespace splitting sees as one token. Multi-constellation
    // files reach this count routinely. The picosecond and clock-offset fields
    // shift the flag away from its usual column, so each combination is covered.
    for (picoseconds, clock, expected_picoseconds, expected_clock) in [
        ("", "", None, None),
        (" 12345", "", Some(12_345), None),
        ("", "      -0.000000000001", None, Some(-0.000_000_000_001)),
        (
            " 12345",
            "      -0.000000000001",
            Some(12_345),
            Some(-0.000_000_000_001),
        ),
    ] {
        let text = epoch_of(100, 0, picoseconds, clock);
        let obs = RinexObs::parse(&text).expect("an epoch of a hundred satellites must parse");
        let epoch = &obs.epochs()[0];
        assert_eq!(epoch.flag, 0);
        assert_eq!(epoch.sats.len(), 100);
        assert_eq!(
            epoch.epoch_picoseconds, expected_picoseconds,
            "{picoseconds:?}"
        );
        assert_eq!(epoch.rcv_clock_offset_s, expected_clock, "{clock:?}");

        let reparsed =
            RinexObs::parse(&obs.to_rinex_string()).expect("re-encoded RINEX OBS must reparse");
        assert_eq!(reparsed, obs);
    }
}

#[test]
fn epoch_satellite_counts_read_the_same_either_side_of_the_merge() {
    // 99 is written with a leading space and never merges; 100 and above fill
    // the field. A non-zero flag merges the same way.
    for (count, flag) in [(9_usize, 0_u8), (99, 0), (100, 1), (155, 0)] {
        let obs = RinexObs::parse(&epoch_of(count, flag, "", ""))
            .unwrap_or_else(|error| panic!("{count} satellites, flag {flag}: {error}"));
        let epoch = &obs.epochs()[0];
        assert_eq!(epoch.flag, flag, "{count} satellites");
        assert_eq!(epoch.sats.len(), count, "{count} satellites");
    }
}

#[test]
fn nonconforming_epoch_field_shapes_keep_their_existing_reading() {
    // Splitting is bounded to counts that actually merge, so these two keep the
    // readings they have always had. A zero-padded flag followed by a separate
    // count is not a merge, and must not be read as a count of zero with the
    // real count taken for a clock offset.
    let (_, systems) = multi_constellation_fixture(1);
    let padded_flag = obs_with_code_headers(
        &systems,
        "> 2020 06 24 00 00  0.0000000  0000 1\nG01   23000000.000",
    );
    let obs = RinexObs::parse(&padded_flag).expect("a zero-padded flag still parses");
    assert_eq!(obs.epochs()[0].flag, 0);
    assert_eq!(obs.epochs()[0].sats.len(), 1);
    assert_eq!(obs.epochs()[0].rcv_clock_offset_s, None);

    // A count past the I3 field stays rejected rather than becoming a
    // picosecond field with an invented flag and count.
    let overflowing = obs_with_code_headers(
        &systems,
        "> 2020 06 24 00 00  0.0000000  01000 0000\nG01   23000000.000",
    );
    assert!(
        RinexObs::parse(&overflowing).is_err(),
        "a count past the I3 field must not parse"
    );

    // A RINEX 4 record carries its picoseconds after the clock offset, where
    // this reader does not look for them. Such a line keeps its rejection
    // rather than parsing with the field silently dropped, whether the clock is
    // written or left blank so the picoseconds land in its token.
    for trailing in ["      -0.000000000001 12345", " 00001"] {
        let (satellites, systems) = multi_constellation_fixture(100);
        let mut body = format!("> 2020 06 24 00 00  0.0000000  0100{trailing}");
        for satellite in &satellites {
            body.push_str(&format!("\n{satellite}   23000000.000"));
        }
        assert!(
            RinexObs::parse(&obs_with_code_headers(&systems, &body)).is_err(),
            "a record whose trailing field this reader cannot place must not parse: {trailing:?}"
        );
    }

    // `0100` followed by a separate count reads as an out-of-range flag with no
    // satellites, which is what this reader has always made of it. Separating
    // the token here would invent a hundred-satellite epoch and take the count
    // for a clock offset. The picosecond variant reads the same way.
    for picoseconds in ["", " 12345"] {
        let flag_shaped = obs_with_code_headers(
            &systems,
            &format!("> 2020 06 24 00 00  0.0000000{picoseconds}  0100 0"),
        );
        let obs = RinexObs::parse(&flag_shaped).expect("an out-of-range flag still parses");
        assert_eq!(obs.epochs()[0].flag, 100, "{picoseconds:?}");
        assert_eq!(obs.epochs()[0].sats.len(), 0, "{picoseconds:?}");
        assert_eq!(obs.epochs()[0].rcv_clock_offset_s, None, "{picoseconds:?}");
    }
}

#[test]
fn reading_by_column_leaves_every_looser_shape_as_it_was() {
    // The layout is read first, but each looser reading below it still applies
    // in turn, so a line that is not laid out in columns keeps the reading it
    // has always had.
    let (satellites, systems) = multi_constellation_fixture(100);

    // Loosely spaced, with the flag and count run together: no layout, but the
    // merged-token reading still recovers the hundred satellites.
    let mut loose = String::from("> 2020 6 24 0 0 0 0100");
    for satellite in &satellites {
        loose.push_str(&format!("\n{satellite}   23000000.000"));
    }
    let obs = RinexObs::parse(&obs_with_code_headers(&systems, &loose))
        .expect("a loosely spaced merged count still parses");
    assert_eq!(obs.epochs()[0].sats.len(), 100);

    // A seconds field holding a space is handed over as one field, not re-split:
    // `0 1` is one second, as the looser reading has always made it.
    let spaced = obs_with_code_headers(&systems, &format!("> 2020 06 24 00 00{:11}  0  0", "0 1"));
    let obs = RinexObs::parse(&spaced).expect("a seconds field with a space still parses");
    assert_eq!(obs.epochs()[0].epoch.second, 1.0);

    // Column-shaped but not column-valued: the layout read parses badly, so the
    // looser reading takes over rather than the line being rejected.
    let odd_time = minimal_obs(
        &[header_line(
            &format!(
                "{:7}{:7}{:7}{:7}{:7} 0.0000                GPS",
                2020, 6, 24, 0, 0
            ),
            "TIME OF FIRST OBS",
        )],
        "",
    );
    let obs = RinexObs::parse(&odd_time).expect("a time header off its columns still parses");
    assert_eq!(
        obs.header().time_of_first_obs.expect("present").0.second,
        0.0
    );

    // Components on their columns with something trailing them: no layout, and
    // whitespace cannot separate the first two, but the columns still read.
    let trailing = minimal_obs(
        &[header_line(
            &format!("{:14.4}{:14.4}{:14.4} extra", 0.0, -10_000_000.0, 0.0),
            "APPROX POSITION XYZ",
        )],
        "",
    );
    let obs = RinexObs::parse(&trailing).expect("trailing content still parses");
    assert_eq!(
        obs.header().approx_position_m,
        Some([0.0, -10_000_000.0, 0.0])
    );
}

#[test]
fn a_version_two_epoch_reads_a_satellite_count_that_fills_its_field() {
    // The RINEX 2 epoch line abuts its flag and count exactly as the RINEX 3 one
    // does, and had no repair at all before the columns were read. Its satellite
    // list carries twelve to a line and continues from column 32 on the next.
    let satellites: Vec<String> = (1..=32)
        .map(|prn| format!("G{prn:02}"))
        .chain((1..=24).map(|prn| format!("R{prn:02}")))
        .chain((1..=36).map(|prn| format!("E{prn:02}")))
        .chain((1..=8).map(|prn| format!("C{prn:02}")))
        .collect();
    assert_eq!(satellites.len(), 100);

    let mut chunks = satellites.chunks(12);
    let mut body = format!(
        " 20  6 24  0  0  0.0000000  0{:3}{}",
        satellites.len(),
        chunks.next().expect("the first twelve").concat()
    );
    for chunk in chunks {
        body.push_str(&format!("\n{}{}", " ".repeat(32), chunk.concat()));
    }
    for satellite in &satellites {
        let _ = satellite;
        body.push_str("\n   23000000.000");
    }
    let text = obs_with_version_and_code_headers(
        2.11,
        &[header_line("     1    C1", "# / TYPES OF OBSERV")],
        &body,
    );

    let obs = RinexObs::parse(&text).expect("a version 2 epoch of a full count must parse");
    assert_eq!(obs.epochs()[0].flag, 0);
    assert_eq!(obs.epochs()[0].sats.len(), 100);
}

#[test]
fn an_antenna_delta_whose_components_abut_is_read() {
    // The same three adjacent F14.4 columns as the position, and the same merge.
    // This one guards the reading rather than reproducing a break: the delta's
    // components were already recovered by the reader's lenient last tier.
    let text = minimal_obs(
        &[header_line(
            "           0.0  -10000000.0           0.0",
            "ANTENNA: DELTA H/E/N",
        )],
        "",
    );
    let obs = RinexObs::parse(&text).expect("parse an adjacent-column antenna delta");
    assert_eq!(
        obs.header().antenna_delta_hen_m,
        Some([0.0, -10_000_000.0, 0.0])
    );
    let reparsed =
        RinexObs::parse(&obs.to_rinex_string()).expect("re-encoded RINEX OBS must reparse");
    assert_eq!(reparsed, obs);
}

#[test]
fn a_blank_glonass_bias_record_clears_the_one_before_it() {
    // A blank record means the biases are unknown, so it replaces what came
    // before rather than leaving it standing.
    let text = minimal_obs(
        &[
            header_line(" C1C  -71.940", "GLONASS COD/PHS/BIS"),
            header_line("", "GLONASS COD/PHS/BIS"),
        ],
        "",
    );
    let obs = RinexObs::parse(&text).expect("parse a cleared GLONASS bias record");
    assert_eq!(obs.header().glonass_cod_phs_bis, Some(Vec::new()));
}

#[test]
fn rinex2_code_round_trips_through_its_canonical_form() {
    // A version 2 file's codes are kept canonically, and the writer maps them
    // back to version 2 names. The two do not have to agree on the text - the
    // mapping into canonical form is not injective, so `C1` and `P1` can share
    // a canonical code and only one of them comes back. They do have to agree
    // on the signal, or a product would name a different one every time it was
    // written and read, and the fuzz round trip would find it.
    const KINDS: [char; 5] = ['C', 'P', 'L', 'D', 'S'];
    const BANDS: [char; 9] = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];
    for system in [
        GnssSystem::Gps,
        GnssSystem::Glonass,
        GnssSystem::Galileo,
        GnssSystem::BeiDou,
        GnssSystem::Qzss,
        GnssSystem::Navic,
        GnssSystem::Sbas,
    ] {
        for kind in KINDS {
            for band in BANDS {
                let declared = format!("{kind}{band}");
                let canonical = canonical_rinex2_obs_code(system, &declared);
                let candidates = rinex2_obs_code_candidates(system, &canonical);
                let written = candidates
                    .first()
                    .unwrap_or_else(|| panic!("{system:?} {declared} has no version 2 name"));
                for alternative in &candidates {
                    assert_eq!(
                        canonical_rinex2_obs_code(system, alternative),
                        canonical,
                        "{system:?} {declared} lists {alternative} as an inverse"
                    );
                }
                assert_eq!(
                    canonical_rinex2_obs_code(system, written),
                    canonical,
                    "{system:?} {declared} became {canonical}, written back as {written}"
                );
            }
        }
    }
}

#[test]
fn a_version_two_header_names_every_position_any_constellation_uses() {
    // Version 2 names its codes once for the whole file. A product a caller
    // assembled can give two constellations lists of different lengths, and the
    // one record has to name the longer, or the values past its end are written
    // where no reader looks for them.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.rnx"
    );
    let text = std::fs::read_to_string(path).expect("read the committed RINEX 2 fixture");
    let mut obs = RinexObs::parse(&text).expect("parse the RINEX 2 fixture");
    obs.header.obs_codes.clear();
    obs.header
        .obs_codes
        .insert(GnssSystem::Gps, vec!["C1C".to_string(), "L1C".to_string()]);
    obs.header.obs_codes.insert(
        GnssSystem::Galileo,
        vec!["C1C".to_string(), "L1C".to_string(), "C5Q".to_string()],
    );

    let encoded = obs.to_rinex_string();
    let declared = encoded
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the version 2 header names its types");
    assert_eq!(
        &declared[..24],
        "     3    C1    L1    C5",
        "every position is named: {declared:?}"
    );
}

#[test]
fn a_version_two_file_names_a_code_version_two_cannot_spell() {
    // A caller can put a version 3 code on a product and ask for version 2
    // output. Version 2 names a kind and a band and has nowhere to put the
    // tracking attribute, so `C1X` is written `C1` and reads back as this
    // system's default tracking on band 1. A digit in that field, which is what
    // holding the column with the code's position would write, is not a code
    // any reader takes.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.rnx"
    );
    let text = std::fs::read_to_string(path).expect("read the committed RINEX 2 fixture");
    let mut obs = RinexObs::parse(&text).expect("parse the RINEX 2 fixture");
    let codes = obs
        .header
        .obs_codes
        .get_mut(&GnssSystem::Gps)
        .expect("the fixture carries GPS codes");
    codes[0] = "C1X".to_string();

    let encoded = obs.to_rinex_string();
    let declared = encoded
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the version 2 header names its types");
    assert_eq!(
        &declared[6..12],
        "    C1",
        "the code keeps its kind and band: {declared:?}"
    );

    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(
        reparsed.header().obs_codes[&GnssSystem::Gps][0],
        "C1C",
        "reading it back gives band 1 at this system's default tracking"
    );
}

#[test]
fn prn_observation_counts_are_read_from_the_columns_they_are_written_in() {
    // `PRN / # OF OBS` is `3X,A1,I2,9I6`: three blanks, then the satellite, then
    // the counts. Reading the satellite from the first three columns found them
    // blank on every real file, so the record was dropped and every count with
    // it. The committed WTZR fixture carries the record as the format writes it.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
    );
    let text = std::fs::read_to_string(path).expect("read the committed fixture");
    let obs = RinexObs::parse(&text).expect("parse the fixture");

    let counts = obs
        .header()
        .prn_obs_counts
        .get(&GnssSatelliteId {
            system: GnssSystem::BeiDou,
            prn: 2,
        })
        .expect("C02 declares its counts");
    assert_eq!(
        counts.iter().take(3).copied().collect::<Vec<_>>(),
        vec![Some(1628), Some(1266), Some(2215)],
        "the counts are read from column 7 onward"
    );

    // Written back, the record lands where it was read from.
    let encoded = obs.to_rinex_string();
    let line = encoded
        .lines()
        .find(|line| line.contains("PRN / # OF OBS"))
        .expect("the record is written");
    assert_eq!(&line[..3], "   ", "the satellite sits at columns 4 to 6");
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(
        reparsed.header().prn_obs_counts,
        obs.header().prn_obs_counts
    );
}

fn version_two_fixture_text() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.rnx"
    );
    std::fs::read_to_string(path).expect("read the committed RINEX 2 fixture")
}

fn version_two_fixture() -> RinexObs {
    RinexObs::parse(&version_two_fixture_text()).expect("parse the RINEX 2 fixture")
}

#[test]
fn a_version_two_header_gives_a_conflicting_signal_its_own_column() {
    // GPS `C1W` and GLONASS `C1C` are `P1` and `C1`. Version 2 names its codes
    // once, so one column cannot carry both: writing `P1` makes GLONASS read
    // back as `C1P`, a different signal. The position is split instead, and
    // each constellation leaves the other's column blank.
    let mut obs = version_two_fixture();
    obs.header.obs_codes.clear();
    obs.header
        .obs_codes
        .insert(GnssSystem::Gps, vec!["C1W".to_string()]);
    obs.header
        .obs_codes
        .insert(GnssSystem::Glonass, vec!["C1C".to_string()]);
    let epoch = obs.epochs.first_mut().expect("the fixture has an epoch");
    epoch.sats.clear();
    epoch.sats.insert(
        GnssSatelliteId {
            system: GnssSystem::Gps,
            prn: 1,
        },
        vec![ObsValue {
            value: Some(123.0),
            lli: None,
            ssi: None,
        }],
    );
    epoch.sats.insert(
        GnssSatelliteId {
            system: GnssSystem::Glonass,
            prn: 2,
        },
        vec![ObsValue {
            value: Some(456.0),
            lli: None,
            ssi: None,
        }],
    );
    obs.epochs.truncate(1);

    let encoded = obs.to_rinex_string();
    let declared = encoded
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the version 2 header names its types");
    assert_eq!(
        &declared[..18],
        "     2    P1    C1",
        "the conflict gets two columns: {declared:?}"
    );

    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    let read = &reparsed.epochs()[0];
    assert_eq!(
        reparsed.header().obs_codes[&GnssSystem::Gps],
        vec!["C1W".to_string(), "C1C".to_string()],
        "no signal is renamed"
    );
    assert_eq!(
        reparsed.header().obs_codes[&GnssSystem::Glonass],
        vec!["C1P".to_string(), "C1C".to_string()],
        "no signal is renamed"
    );
    let gps = &read.sats[&GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    }];
    let glonass = &read.sats[&GnssSatelliteId {
        system: GnssSystem::Glonass,
        prn: 2,
    }];
    // Each value keeps the signal it was measured on: GPS holds `C1W` in the
    // `P1` column, GLONASS holds `C1C` in the `C1` column, and neither reads
    // the other's.
    assert_eq!(
        (gps[0].value, gps[1].value),
        (Some(123.0), None),
        "GPS keeps its value under P1"
    );
    assert_eq!(
        (glonass[0].value, glonass[1].value),
        (None, Some(456.0)),
        "GLONASS keeps its value under C1"
    );
}

#[test]
fn a_version_two_header_shares_one_column_where_the_constellations_agree() {
    // The split is only for a conflict. A file read at version 2 gave every
    // constellation its list from the same record, so one name serves them all
    // and the header keeps the width it had.
    let mut obs = version_two_fixture();
    obs.header.obs_codes.clear();
    obs.header
        .obs_codes
        .insert(GnssSystem::Gps, vec!["C1W".to_string(), "C1C".to_string()]);
    obs.header.obs_codes.insert(
        GnssSystem::Glonass,
        vec!["C1P".to_string(), "C1C".to_string()],
    );

    let encoded = obs.to_rinex_string();
    let declared = encoded
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the version 2 header names its types");
    assert_eq!(
        &declared[..18],
        "     2    P1    C1",
        "both constellations read P1 and C1 back as what they hold: {declared:?}"
    );
}

#[test]
fn version_two_prn_observation_counts_are_read() {
    // A version 2 header names its codes once, and the per-constellation lists
    // are not built until the body is read. Taking the count from those lists
    // found none while the header was still being read, so every count was
    // dropped. A version 2 file may also leave the constellation letter blank.
    let mut text = String::new();
    for line in version_two_fixture_text().lines() {
        if line.contains("END OF HEADER") {
            text.push_str(&format!(
                "{:<60}{:<20}\n",
                format!("     1{:6}{:6}{:6}{:6}", 11, 22, 33, 44),
                "PRN / # OF OBS"
            ));
        }
        text.push_str(line);
        text.push('\n');
    }
    let obs = RinexObs::parse(&text).expect("parse the fixture with a count record");
    let counts = obs
        .header()
        .prn_obs_counts
        .get(&GnssSatelliteId {
            system: GnssSystem::Gps,
            prn: 1,
        })
        .expect("the blank constellation letter means the one the header names");
    assert_eq!(
        counts.iter().take(4).copied().collect::<Vec<_>>(),
        vec![Some(11), Some(22), Some(33), Some(44)]
    );
}

#[test]
fn a_version_two_clock_offset_is_written_in_its_own_columns() {
    // The clock offset is an `F12.9` field at columns 69 to 80. Letting it
    // follow the last satellite put it wherever the count happened to end, so
    // an epoch of fewer than twelve satellites wrote it into the satellite
    // list, where this crate and every other reader lose it.
    let mut obs = version_two_fixture();
    let epoch = obs.epochs.first_mut().expect("the fixture has an epoch");
    let kept: Vec<_> = epoch.sats.keys().copied().take(3).collect();
    epoch.sats.retain(|sat, _| kept.contains(sat));
    epoch.rcv_clock_offset_s = Some(0.123_456_789);

    let encoded = obs.to_rinex_string();
    let line = encoded
        .lines()
        .find(|line| line.starts_with(" 15  1  1  0  0  0.0000000"))
        .expect("the epoch record is written");
    assert_eq!(
        &line[68..80],
        " 0.123456789",
        "the clock sits in columns 69 to 80: {line:?}"
    );

    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs()[0].rcv_clock_offset_s, Some(0.123_456_789));
}

#[test]
fn a_version_two_record_runs_to_the_count_the_header_declares() {
    // A version 2 reader takes a fixed number of observation lines for every
    // satellite. Writing only as many as a satellite happened to hold put the
    // next satellite's values under this one, and the epoch then ran out of
    // lines before its last satellite.
    let mut obs = version_two_fixture();
    let epoch = obs.epochs.first_mut().expect("the fixture has an epoch");
    let kept: Vec<_> = epoch.sats.keys().copied().take(3).collect();
    epoch.sats.retain(|sat, _| kept.contains(sat));
    let short = kept[1];
    epoch
        .sats
        .get_mut(&short)
        .expect("the second satellite")
        .truncate(2);
    let expected: Vec<_> = epoch.sats.values().map(Vec::len).collect();
    assert_eq!(expected, vec![8, 2, 8], "one satellite holds fewer values");

    let encoded = obs.to_rinex_string();
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    let read = &reparsed.epochs()[0];
    assert_eq!(read.sats.len(), 3, "every satellite is found");
    for (sat, values) in &read.sats {
        assert_eq!(values.len(), 8, "{sat:?} gets the declared count back");
    }
    // The short satellite keeps the two values it held; the rest read as blank.
    let short_values = &read.sats[&short];
    assert!(short_values[0].value.is_some() && short_values[1].value.is_some());
    assert!(short_values[2..].iter().all(|value| value.value.is_none()));
    // Its neighbours are untouched, which is what a misplaced record broke.
    for sat in [kept[0], kept[2]] {
        assert_eq!(read.sats[&sat], obs.epochs[0].sats[&sat]);
    }
}

#[test]
fn a_version_two_file_writes_values_a_scale_factor_would_have_declared() {
    // `SYS / SCALE FACTOR` arrived with version 3, so a version 2 file has no
    // way to say a value was scaled. Applying the factor anyway wrote a number
    // the reader had no reason to divide back.
    let mut obs = version_two_fixture();
    let system = *obs
        .header
        .obs_codes
        .keys()
        .next()
        .expect("the fixture names a constellation");
    obs.header.scale_factors.push(super::ObsScaleFactor {
        system,
        factor: 10.0,
        codes: Vec::new(),
    });
    let (first_sat, first_values) = obs.epochs[0]
        .sats
        .iter()
        .next()
        .expect("the fixture's first epoch has a satellite");
    let (first_sat, first_value) = (
        *first_sat,
        first_values[0].value.expect("its first value is present"),
    );

    let encoded = obs.to_rinex_string();
    assert!(!encoded.contains("SYS / SCALE FACTOR"));
    // Scaling it would have written ten times this, well inside `F14.3`, so the
    // written field is what tells the two apart.
    let written = encoded
        .lines()
        .find(|line| line.starts_with(&format!("{first_value:14.3}")))
        .unwrap_or_else(|| panic!("{first_sat:?} is written unscaled as {first_value:14.3}"));
    assert!(!written.starts_with(&format!("{:14.3}", first_value * 10.0)));

    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(
        reparsed.epochs()[0].sats,
        obs.epochs[0].sats,
        "the values come back as they went in, unscaled"
    );
}

#[test]
fn an_event_epoch_keeps_the_records_that_followed_it() {
    // A flag 3 epoch is followed by the header records for a new site
    // occupation. They used to be counted and thrown away, and the epoch then
    // written back declaring zero, so a file that changed site lost the marker,
    // antenna and position it changed to.
    let mut text = String::new();
    for line in version_two_fixture_text().lines() {
        text.push_str(line);
        text.push('\n');
        if line.contains("END OF HEADER") {
            text.push_str(" 15  1  1  0  0  0.0000000  3  2\n");
            text.push_str(&format!("{:<60}{:<20}\n", "NEWSITE", "MARKER NAME"));
            text.push_str(&format!(
                "{:<60}{:<20}\n",
                "  1234567.0000  -4567890.0000   4321098.0000", "APPROX POSITION XYZ"
            ));
        }
    }

    let obs = RinexObs::parse(&text).expect("parse a file with a site occupation");
    let event = obs
        .epochs()
        .iter()
        .find(|epoch| epoch.flag == 3)
        .expect("the event epoch is kept");
    assert_eq!(event.special_records.len(), 2);
    assert!(event.special_records[0].contains("MARKER NAME"));
    assert!(event.special_records[1].contains("APPROX POSITION XYZ"));

    // Written back, the epoch declares them and carries them.
    let encoded = obs.to_rinex_string();
    let line = encoded
        .lines()
        .find(|line| line.contains("  3  2"))
        .expect("the event declares its two records");
    assert_eq!(line.trim_end(), " 15  1  1  0  0  0.0000000  3  2");
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

#[test]
fn a_version_three_event_epoch_keeps_its_records_too() {
    let text = concat!(
        "     3.05           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n",
        "G    1 C1C                                                  SYS / # / OBS TYPES\n",
        "                                                            END OF HEADER\n",
        "> 2020 01 01 00 00  0.0000000  4  1\n",
        "a comment carried by the event                              COMMENT\n",
        "> 2020 01 01 00 00 30.0000000  0  1\n",
        "G01      20000000.000\n",
    );
    let obs = RinexObs::parse(text).expect("parse a version 3 file with an event");
    assert_eq!(
        obs.epochs()[0].special_records,
        vec!["a comment carried by the event                              COMMENT".to_string()]
    );
    let encoded = obs.to_rinex_string();
    assert!(encoded.contains("a comment carried by the event"));
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

#[test]
fn a_version_two_event_epoch_names_no_satellites() {
    // An event record keeps only its flag and epoch here. Writing a satellite
    // list beside a declared count of zero left text in the columns the clock
    // offset occupies, which the reader then took for a clock.
    let mut obs = version_two_fixture();
    let epoch = obs.epochs.first_mut().expect("the fixture has an epoch");
    epoch.flag = 4;

    let encoded = obs.to_rinex_string();
    let line = encoded
        .lines()
        .find(|line| line.starts_with(" 15  1  1  0  0  0.0000000"))
        .expect("the event record is written");
    assert_eq!(
        line.trim_end(),
        " 15  1  1  0  0  0.0000000  4  0",
        "an event names no satellites, and this one carries no records either"
    );
}

#[test]
fn a_galileo_band_five_code_keeps_its_band() {
    // Galileo's `C5` and `P2` both canonicalise to `C5X`, so an inverse that
    // takes whichever it meets first can write `P2` for a band 5 pseudorange.
    // No reader outside this crate defines `P2` for Galileo.
    let candidates = rinex2_obs_code_candidates(GnssSystem::Galileo, "C5X");
    assert_eq!(
        candidates.first().map(String::as_str),
        Some("C5"),
        "the code keeps the band it was measured on: {candidates:?}"
    );
    assert!(
        !candidates.iter().any(|name| name == "P2"),
        "version 2 gives Galileo no `P` observable: {candidates:?}"
    );

    // `C5Q` is band 5 too. Version 2 has no field for the tracking attribute,
    // so it is written `C5` and reads back as `C5X`, losing the `Q`. That is the
    // most version 2 can say; naming a different band to keep the attribute
    // would not be.
    assert_eq!(
        rinex2_obs_code_candidates(GnssSystem::Galileo, "C5Q"),
        vec!["C5".to_string()]
    );
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::Galileo, "C5"), "C5X");
}

#[test]
fn version_two_gives_only_gps_and_glonass_a_p_observable() {
    // Version 2 says "P: Pseudorange GPS and Glonass: P code". Galileo and
    // BeiDou carried `P` rows anyway, holding the legacy
    // differential-code-bias labels, where `P1` and `P2` mean the first and
    // second frequency whatever the constellation. For BeiDou that made `C2`
    // B2I and `P2` B3I: two spellings of the same digit naming different bands.
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::Gps, "P1"), "C1W");
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::Gps, "P2"), "C2W");
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::Glonass, "P1"), "C1P");
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::Glonass, "P2"), "C2P");
    // BeiDou's `C` rows stay: version 2 has no BeiDou at all, and the receivers
    // that wrote it numbered B1, B2, B3 as 1, 2, 3.
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::BeiDou, "C1"), "C2I");
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::BeiDou, "C2"), "C7I");
    assert_ne!(canonical_rinex2_obs_code(GnssSystem::BeiDou, "P2"), "C6I");
    // And the digit means the same band whatever the kind. It used to be
    // remapped only for `C`, so a file's `C1` was B1I and its `L1` was B1C:
    // one measurement pair read as two different signals.
    for (declared, canonical) in [
        ("C1", "C2I"),
        ("L1", "L2I"),
        ("D1", "D2I"),
        ("S1", "S2I"),
        ("C2", "C7I"),
        ("L2", "L7I"),
    ] {
        assert_eq!(
            canonical_rinex2_obs_code(GnssSystem::BeiDou, declared),
            canonical,
            "BeiDou {declared}"
        );
    }
}

#[test]
fn galileo_version_two_codes_are_the_ones_the_format_defines() {
    // RINEX 2.11 gives Galileo `C1`, `C5`, `C6`, `C7` and `C8`, whose digits are
    // the bands E1, E5a, E6, E5b and E5a+b, and no `P` observable. This table
    // used to carry the legacy differential-code-bias labels instead, reading
    // `C2` as E5a-Q and `P2` as E5a-X, so a conforming Galileo `C5` and an
    // invented `C2` both claimed band 5 and a band 2 code was read as band 5.
    for (declared, canonical) in [
        ("C1", "C1C"),
        ("C5", "C5X"),
        ("C6", "C6C"),
        ("C7", "C7X"),
        ("C8", "C8X"),
        ("L5", "L5X"),
    ] {
        assert_eq!(
            canonical_rinex2_obs_code(GnssSystem::Galileo, declared),
            canonical,
            "Galileo {declared}"
        );
    }
    // A `C2` version 2 never should have carried is read as the band it names.
    assert_eq!(canonical_rinex2_obs_code(GnssSystem::Galileo, "C2"), "C2X");
}

#[test]
fn a_version_two_file_carries_only_the_current_leap_second_count() {
    // Version 2 defines one field in `LEAP SECONDS`. The future count, week and
    // day arrived with version 3 and occupy columns version 2 leaves blank.
    let mut obs = version_two_fixture();
    obs.header.leap_seconds = Some(super::ObsLeapSeconds {
        current: 17,
        delta_future: Some(18),
        week: Some(2000),
        day: Some(3),
    });

    let encoded = obs.to_rinex_string();
    let line = encoded
        .lines()
        .find(|line| line.contains("LEAP SECONDS"))
        .expect("the record is written");
    assert_eq!(&line[..60], format!("{:6}{:54}", 17, ""), "{line:?}");
}

#[test]
fn a_version_two_file_carries_no_version_three_records() {
    // A caller can put version 3 records on a product and ask for version 2
    // output. Writing them would produce a file neither version accepts, so
    // they are left out, and the product loses them.
    let mut obs = version_two_fixture();
    let system = *obs
        .header
        .obs_codes
        .keys()
        .next()
        .expect("the fixture names a constellation");
    obs.header.marker_type = Some("GEODETIC".to_string());
    obs.header.signal_strength_unit = Some("DBHZ".to_string());
    obs.header.glonass_cod_phs_bis = Some(vec![("C1C".to_string(), -71.940)]);
    obs.header.phase_shifts.push(super::ObsPhaseShift {
        system,
        code: "L1C".to_string(),
        correction_cycles: 0.25,
        satellites: Vec::new(),
    });
    obs.header.scale_factors.push(super::ObsScaleFactor {
        system,
        factor: 1000.0,
        codes: Vec::new(),
    });

    let encoded = obs.to_rinex_string();
    for label in [
        "MARKER TYPE",
        "SIGNAL STRENGTH UNIT",
        "GLONASS COD/PHS/BIS",
        "SYS / PHASE SHIFT",
        "SYS / SCALE FACTOR",
    ] {
        assert!(
            !encoded.contains(label),
            "{label} is not a version 2 record"
        );
    }
    RinexObs::parse(&encoded).expect("the version 2 file still reads back");

    // The same product at version 3 keeps them.
    obs.header.version = 3.05;
    let encoded = obs.to_rinex_string();
    for label in ["MARKER TYPE", "SIGNAL STRENGTH UNIT", "GLONASS COD/PHS/BIS"] {
        assert!(
            encoded.contains(label),
            "{label} belongs in a version 3 file"
        );
    }
}

#[test]
fn a_version_two_product_is_written_as_version_two() {
    // A version 2 file used to be re-emitted through the version 3 record
    // writer, so its own output declared version 2 while carrying `>` epoch
    // records: a file that was neither version. It is now written in the
    // records its version names, which is a file other readers accept too.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.rnx"
    );
    let text = std::fs::read_to_string(path).expect("read the committed RINEX 2 fixture");
    let obs = RinexObs::parse(&text).expect("parse the RINEX 2 fixture");
    assert!((obs.header().version - 2.11).abs() < 1e-9);
    assert_eq!(obs.epochs().len(), 2);

    let encoded = obs.to_rinex_string();
    assert!(
        !encoded.lines().any(|line| line.starts_with('>')),
        "no version 3 epoch record is written"
    );
    assert!(
        encoded.contains("# / TYPES OF OBSERV") && !encoded.contains("SYS / # / OBS TYPES"),
        "the observation types are named the way version 2 names them"
    );
    // The codes come back exactly as the file declared them, which is what the
    // mapping back from canonical codes has to achieve for the values to stay
    // aligned to the header that names them.
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the fixture declares its types");
    assert!(
        encoded.lines().any(|line| line == declared),
        "the observation types are written back as they were read"
    );

    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
    assert!((reparsed.header().version - 2.11).abs() < 1e-9);
    assert_eq!(reparsed.header().obs_codes, obs.header().obs_codes);
}

#[test]
fn a_glonass_bias_record_longer_than_one_line_survives_a_round_trip() {
    // Each entry takes thirteen of the sixty columns, so a fifth would be cut
    // off the end of the line. The record continues on another line instead, and
    // the reader adds to what it already has rather than replacing it.
    let entries = [
        ("C1C", -71.940),
        ("C1P", -71.940),
        ("C2C", -71.940),
        ("C2P", -71.940),
        ("C3C", -12.500),
    ];
    let mut first = String::new();
    for (code, value) in &entries[..4] {
        first.push_str(&format!(" {code:>3} {value:8.3}"));
    }
    let text = minimal_obs(
        &[
            header_line(first.trim_start(), "GLONASS COD/PHS/BIS"),
            header_line(" C3C  -12.500", "GLONASS COD/PHS/BIS"),
        ],
        "",
    );

    let obs = RinexObs::parse(&text).expect("parse a continued GLONASS bias record");
    let read = obs
        .header()
        .glonass_cod_phs_bis
        .as_ref()
        .expect("the record is present");
    assert_eq!(read.len(), 5, "every entry is kept: {read:?}");
    assert_eq!(read[4].0, "C3C");

    let encoded = obs.to_rinex_string();
    assert_eq!(
        encoded
            .lines()
            .filter(|line| line.contains("GLONASS COD/PHS/BIS"))
            .count(),
        2,
        "the record is written across two lines rather than truncated"
    );
    let reparsed = RinexObs::parse(&encoded).expect("re-encoded RINEX OBS must reparse");
    assert_eq!(reparsed, obs);
}

#[test]
fn rejects_malformed_glonass_slot_records() {
    for header in [
        header_line("  1 R01 bad", "GLONASS SLOT / FRQ #"),
        header_line("  1 G01  1", "GLONASS SLOT / FRQ #"),
        header_line("  2 R01  1", "GLONASS SLOT / FRQ #"),
    ] {
        assert_parse_err(minimal_obs(&[header], ""));
    }
}

#[test]
fn rejects_out_of_range_glonass_slot_channel() {
    let header = header_line("  1 R01 99", "GLONASS SLOT / FRQ #");
    let err = RinexObs::parse(&minimal_obs(&[header], ""))
        .expect_err("out-of-range GLONASS slot channel must be rejected");

    assert!(
        matches!(err, Error::Parse(ref message)
            if message.contains("glonass_slot.channel")
                && message.contains("out of range")),
        "{err}"
    );
}

#[test]
fn glonass_slot_channel_drives_g1_g2_frequency_metadata() {
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  1\nR01{}{}{}{}",
        obs_field(22_000_000.0, 0, 0),
        obs_field(10.0, 1, 2),
        obs_field(22_000_001.0, 0, 0),
        obs_field(20.0, 3, 4)
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("R    4 C1C L1C C2C L2C", "SYS / # / OBS TYPES"),
            header_line("  1 R01 -7", "GLONASS SLOT / FRQ #"),
        ],
        &body,
    ))
    .expect("in-range GLONASS slot channel should parse");

    assert_eq!(obs.header().glonass_slots.get(&1), Some(&-7));
    let rows = carrier_phase_rows(&obs, &obs.epochs()[0], &ObservationFilter::all())
        .expect("valid carrier-phase rows");
    assert_eq!(rows.len(), 1);
    let (sat, phases) = &rows[0];
    assert_eq!(
        *sat,
        GnssSatelliteId::new(GnssSystem::Glonass, 1).expect("valid satellite id")
    );
    assert_eq!(phases.len(), 2);

    let channel = -7.0_f64;
    let expected_g1_hz = 1_602_000_000.0 + channel * 562_500.0;
    let expected_g2_hz = 1_246_000_000.0 + channel * 437_500.0;
    for (row, expected_code, expected_cycles, expected_lli, expected_ssi, expected_frequency) in [
        (
            &phases[0],
            "L1C",
            10.0_f64,
            Some(1),
            Some(2),
            expected_g1_hz,
        ),
        (
            &phases[1],
            "L2C",
            20.0_f64,
            Some(3),
            Some(4),
            expected_g2_hz,
        ),
    ] {
        let expected_wavelength = C_M_S / expected_frequency;
        assert_eq!(row.code, expected_code);
        assert_eq!(
            row.value_cycles.map(f64::to_bits),
            Some(expected_cycles.to_bits())
        );
        assert_eq!(row.lli, expected_lli);
        assert_eq!(row.ssi, expected_ssi);
        assert_eq!(
            row.frequency_hz.map(f64::to_bits),
            Some(expected_frequency.to_bits())
        );
        assert_eq!(
            row.wavelength_m.map(f64::to_bits),
            Some(expected_wavelength.to_bits())
        );
        assert_eq!(
            row.value_m.map(f64::to_bits),
            Some((expected_cycles * expected_wavelength).to_bits())
        );
    }
}

#[test]
fn rejects_unknown_time_of_first_obs_scale() {
    let header = header_line(
        "  2020     6    25     0     0    0.0000000     XYZ",
        "TIME OF FIRST OBS",
    );
    assert_parse_err(minimal_obs(&[header], ""));
}

#[test]
fn accepts_qzss_time_of_first_obs_as_qzsst() {
    let header = header_line(
        "  2020     6    25     0     0    0.0000000     QZS",
        "TIME OF FIRST OBS",
    );
    let body = "> 2020 06 25 00 00 00.0000000  0  1\nJ01  12345678.000";
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[header_line("J    1 C1C", "SYS / # / OBS TYPES"), header],
        body,
    ))
    .expect("QZSS OBS should parse with QZS time system");

    let (t0, scale) = obs.header.time_of_first_obs.expect("time of first obs");
    assert_eq!(scale, TimeScale::Qzsst);
    assert_eq!(t0.year, 2020);
    assert!(
        obs.epochs()[0]
            .sats
            .contains_key(&GnssSatelliteId::new(GnssSystem::Qzss, 1).expect("valid satellite id")),
        "QZSS satellite row should load"
    );
}

#[test]
fn to_rinex_string_omits_unsupported_time_header_labels() {
    let first = header_line(
        "  2020     6    25     0     0    0.0000000     GPS",
        "TIME OF FIRST OBS",
    );
    let last = header_line(
        "  2020     6    25     0    30    0.0000000     GPS",
        "TIME OF LAST OBS",
    );
    let mut obs = RinexObs::parse(&minimal_obs(&[first, last], "")).expect("parse OBS");
    let first_epoch = obs.header.time_of_first_obs.expect("first stamp").0;
    let last_epoch = obs.header.time_of_last_obs.expect("last stamp").0;
    obs.header.time_of_first_obs = Some((first_epoch, TimeScale::Tcg));
    obs.header.time_of_last_obs = Some((last_epoch, TimeScale::Tcb));

    let serialized = obs.to_rinex_string();

    assert!(!serialized.contains("TCG"));
    assert!(!serialized.contains("TCB"));
    assert!(!serialized.contains("TIME OF FIRST OBS"));
    assert!(!serialized.contains("TIME OF LAST OBS"));
    let reparsed = RinexObs::parse(&serialized).expect("parse serialized OBS");
    assert_eq!(reparsed.header.time_of_first_obs, None);
    assert_eq!(reparsed.header.time_of_last_obs, None);
}

#[test]
fn rejects_invalid_civil_epoch_fields() {
    let header = header_line(
        "  2020     2    30     0     0    0.0000000     GPS",
        "TIME OF FIRST OBS",
    );
    assert_parse_err(minimal_obs(&[header], ""));

    for body in [
        "> 2020 02 30 00 00 00.0000000  0  0",
        "> 2020 06 25 24 00 00.0000000  0  0",
        "> 2020 06 25 23 59 60.0000000  0  0",
    ] {
        assert_parse_err(minimal_obs(&[], body));
    }
}

#[test]
fn accepts_utc_leap_second_epoch_fields() {
    let header = header_line(
        "  2016    12    31    23    59   60.0000000     UTC",
        "TIME OF FIRST OBS",
    );
    let body = "> 2016 12 31 23 59 60.0000000  0  0";
    let obs = RinexObs::parse(&minimal_obs(&[header], body)).expect("UTC leap-second OBS");

    let (t0, scale) = obs.header.time_of_first_obs.expect("time of first obs");
    assert_eq!(scale, TimeScale::Utc);
    assert_eq!(t0.second, 60.0);
    assert_eq!(obs.epochs()[0].epoch.second, 60.0);
}

#[test]
fn accepts_glonass_utc_leap_second_epoch_fields() {
    let header = header_line(
        "  2016    12    31    23    59   60.0000000     GLO",
        "TIME OF FIRST OBS",
    );
    let body = "> 2016 12 31 23 59 60.0000000  0  0";
    let obs = RinexObs::parse(&minimal_obs(&[header], body)).expect("GLONASS UTC OBS");

    let (t0, scale) = obs.header.time_of_first_obs.expect("time of first obs");
    assert_eq!(scale, TimeScale::Utc);
    assert_eq!(t0.second, 60.0);
    assert_eq!(obs.epochs()[0].epoch.second, 60.0);
}

#[test]
fn gps_context_rejects_leap_second_epoch_fields() {
    let header = header_line(
        "  2016    12    31    23    59   60.0000000     GPS",
        "TIME OF FIRST OBS",
    );
    assert_parse_err(minimal_obs(&[header], ""));
}

#[test]
fn utc_context_still_rejects_invalid_leap_second_range() {
    let header = header_line(
        "  2016    12    31    23    59   59.0000000     UTC",
        "TIME OF FIRST OBS",
    );
    for second in ["61.0000000", "-1.0000000"] {
        let body = format!("> 2016 12 31 23 59 {second}  0  0");
        assert_parse_err(minimal_obs(std::slice::from_ref(&header), &body));
    }
}

#[test]
fn rejects_malformed_epoch_flag() {
    assert_parse_err(minimal_obs(&[], "> 2020 06 25 00 00 00.0000000  X  0"));
}

#[test]
fn rejects_truncated_event_record() {
    let text = minimal_obs(&[], "> 2020 06 25 00 00 00.0000000  2  2\nCOMMENT");
    let err = RinexObs::parse(&text).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref msg) if msg.contains("RINEX OBS event record truncated")),
        "{err}"
    );
}

#[test]
fn rejects_satellite_from_undeclared_system() {
    assert_parse_err(minimal_obs(
        &[],
        "> 2020 06 25 00 00 00.0000000  0  1\nR01  12345678.000",
    ));
}

#[test]
fn rejects_non_finite_observation_values() {
    assert_parse_err(minimal_obs(
        &[],
        &format!("> 2020 06 25 00 00 00.0000000  0  1\nG01{:>14}", "NaN"),
    ));
}

#[test]
fn rejects_non_observation_file() {
    let nav = "     3.05           N: GNSS NAV DATA    M (MIXED)           RINEX VERSION / TYPE\n";
    assert!(RinexObs::parse(nav).is_err());
}

#[test]
fn rejects_unsupported_observation_file_version() {
    let v1 = "     1.00           OBSERVATION DATA    G (GPS)             RINEX VERSION / TYPE\n";
    assert!(RinexObs::parse(v1).is_err());
}

#[test]
fn skips_out_of_range_glonass_slot_entry_and_counts_it() {
    // A GLONASS SLOT / FRQ # table that declares two slots, one of which (R28)
    // is an extended slot beyond the engine's 1..=27 PRN cap. The out-of-range
    // entry must be skipped and counted, not reject the whole header; the
    // representable slot (R01) must survive with its channel.
    let header = header_line("  2 R01  1 R28 -3", "GLONASS SLOT / FRQ #");
    let obs = RinexObs::parse(&minimal_obs(&[header], ""))
        .expect("one out-of-range GLONASS slot must not reject the header");
    assert_eq!(obs.skipped_records, 1, "the R28 slot entry must be counted");
    assert_eq!(obs.header().glonass_slots.len(), 1, "only R01 stored");
    assert_eq!(obs.header().glonass_slots.get(&1), Some(&1));
    assert_eq!(obs.header().glonass_slots.get(&28), None);
}

#[test]
fn skips_unknown_satellite_epoch_record_and_counts_it() {
    // An epoch advertising three satellite records, the middle of which is an
    // out-of-range GLONASS slot (R28). The unknown record must be skipped and
    // counted, leaving the valid GPS and GLONASS records intact - no
    // observation values fabricated, no epoch lost.
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  3\nG01{}\nR28{}\nR01{}",
        obs_field(20_000_000.0, 0, 0),
        obs_field(21_000_000.0, 0, 0),
        obs_field(22_000_000.0, 0, 0),
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("G    1 C1C", "SYS / # / OBS TYPES"),
            header_line("R    1 C1C", "SYS / # / OBS TYPES"),
        ],
        &body,
    ))
    .expect("one unknown satellite record must not reject the epoch");
    assert_eq!(obs.skipped_records, 1, "the R28 record must be counted");
    assert_eq!(obs.epochs().len(), 1);
    let sats = &obs.epochs()[0].sats;
    assert_eq!(sats.len(), 2, "both representable records must survive");
    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let r01 = GnssSatelliteId::new(GnssSystem::Glonass, 1).expect("valid satellite id");
    assert_eq!(
        sats.get(&g01).expect("G01 present")[0].value,
        Some(20_000_000.0)
    );
    assert_eq!(
        sats.get(&r01).expect("R01 present")[0].value,
        Some(22_000_000.0)
    );
}

#[test]
fn unrepresentable_record_at_continuation_boundary_is_not_eaten_as_continuation() {
    // G01 omits its trailing observation (RINEX permits dropping trailing blank
    // fields), so the parser is still awaiting a continuation line when it reaches
    // the *next* record, which is an out-of-range GLONASS slot (R28). R28 is a new
    // satellite record, not continuation data: it must terminate G01's record,
    // then be skipped and counted - never spliced into G01's missing L1C.
    let short_g01 = format!("G01{}", obs_field(20_000_000.0, 0, 0));
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  3\n{short_g01}\nR28{}{}\nR01{}{}",
        obs_field(21_000_000.0, 0, 0),
        obs_field(21_000_001.0, 0, 0),
        obs_field(22_000_000.0, 0, 0),
        obs_field(22_000_001.0, 0, 0),
    );
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("R    2 C1C L1C", "SYS / # / OBS TYPES"),
        ],
        &body,
    ))
    .expect("an unrepresentable record at a continuation boundary must not reject the epoch");

    assert_eq!(
        obs.skipped_records, 1,
        "R28 must be counted as a skipped record, not absorbed as continuation"
    );
    assert_eq!(obs.epochs().len(), 1);
    let sats = &obs.epochs()[0].sats;
    assert_eq!(sats.len(), 2, "G01 and R01 must both survive");

    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let r01 = GnssSatelliteId::new(GnssSystem::Glonass, 1).expect("valid satellite id");
    let g01_vals = sats.get(&g01).expect("G01 present");
    assert_eq!(g01_vals[0].value, Some(20_000_000.0), "G01 C1C intact");
    assert_eq!(
        g01_vals[1].value, None,
        "G01 L1C stays empty - R28 data must not leak in as continuation"
    );
    assert_eq!(
        sats.get(&r01).expect("R01 present")[0].value,
        Some(22_000_000.0)
    );
}

#[test]
fn epoch_overrunning_numsat_errors_instead_of_eating_next_epoch() {
    // The first epoch declares THREE satellite records but only one is present;
    // the next line is the FOLLOWING epoch's header. The overrun must be a
    // structural (truncated-epoch) error - NOT silently treated as an unknown
    // satellite, which would swallow the next epoch's header and lose its data.
    let body = format!(
        "> 2020 06 25 00 00 00.0000000  0  3\nG01{}\n> 2020 06 25 00 00 30.0000000  0  1\nG01{}",
        obs_field(20_000_000.0, 0, 0),
        obs_field(20_000_100.0, 0, 0),
    );
    let err = RinexObs::parse(&obs_with_code_headers(
        &[header_line("G    1 C1C", "SYS / # / OBS TYPES")],
        &body,
    ))
    .expect_err("an epoch overrunning its numsat must be a structural error, not a silent skip");
    let msg = format!("{err}");
    assert!(
        msg.contains("truncated") || msg.contains("expected satellite record"),
        "expected a truncated-epoch structural error, got: {msg}"
    );
}

fn zim_rnx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ZIM200CHE_R_20261330000_01H_30S_MO_120epoch.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read RINEX fixture {path}: {e}"))
}

#[test]
fn to_rinex_string_round_trips_through_parse() {
    // The canonical IR is the parsed product (header + epochs). Serializing it
    // and re-parsing must reproduce both. The fixture is multi-GNSS (GPS, GLONASS
    // with a slot/frequency table, Galileo, BeiDou, SBAS), carries phase-shift
    // records, an interval, time-of-first-obs, approximate position and antenna
    // delta, and 120 epochs - so the header and body serializers are exercised
    // broadly.
    let obs = RinexObs::parse(&zim_rnx()).expect("parse ZIM200 RINEX OBS");
    assert_eq!(
        obs.skipped_records, 0,
        "fixture should have no unrepresentable records, so the product re-parses identically"
    );
    assert!(obs.epochs.len() >= 100, "fixture should carry many epochs");

    let serialized = obs.to_rinex_string();
    let reparsed = RinexObs::parse(&serialized).expect("re-parse serialized RINEX OBS");
    let mut expected = obs;
    expected.header.unretained_header_labels.clear();
    assert_eq!(
        reparsed, expected,
        "to_rinex_string must round-trip through parse"
    );
    // Deterministic output.
    assert_eq!(reparsed.to_rinex_string(), serialized);
}
