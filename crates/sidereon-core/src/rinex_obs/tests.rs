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
    let rows = carrier_phase_rows(obs.header(), &obs.epochs()[0], &ObservationFilter::all())
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
        .find(|shift| shift.system == GnssSystem::Gps && shift.code.as_deref() == Some("L1C"))
        .expect("GPS L1C phase shift");
    assert_eq!(gps_l1c.correction_cycles, None);
    assert!(gps_l1c.satellites.is_empty());
    let gal_l5q = h
        .phase_shifts
        .iter()
        .find(|shift| shift.system == GnssSystem::Galileo && shift.code.as_deref() == Some("L5Q"))
        .expect("Galileo L5Q phase shift");
    assert_eq!(gal_l5q.correction_cycles, Some(0.0));
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
        carrier_phase_rows(obs.header(), &obs.epochs()[0], &ObservationFilter::all()),
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
        carrier_phase_rows(obs.header(), &obs.epochs()[0], &ObservationFilter::all()),
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

    let rows = carrier_phase_rows(obs.header(), &obs.epochs()[0], &ObservationFilter::all())
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
    assert_eq!(
        row.phase_shift_cycles.map(f64::to_bits),
        Ok(0.25_f64.to_bits())
    );
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
    assert_eq!(
        row.phase_shift_cycles.map(f64::to_bits),
        Ok(0.0_f64.to_bits())
    );
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
    assert_eq!(epoch.epoch.map(|time| time.year), Some(2020));
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
            // Version 2 added `C2` for the L2C pseudorange, which RINEX 3
            // spells `C2S`, `C2L` or `C2X` by channel. `C2C` is L2 C/A.
            "C2X".to_string(),
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
fn phase_shift_corrections_are_written_in_their_f8_5_field() {
    // A record read by its fields is written in its columns.
    for (body, expected) in [
        ("G L1C 0.0 1 G01", "G L1C  0.00000  01 G01"),
        ("G L1C 0.25000", "G L1C  0.25000"),
        ("G L1C -0.25000", "G L1C -0.25000"),
        ("G L1C 0.12345", "G L1C  0.12345"),
        ("G L1C 2.5e-1", "G L1C  0.25000"),
    ] {
        let obs = RinexObs::parse(&minimal_obs_with_phase_shift(body)).expect("parse phase shift");
        let written = obs.to_rinex_string().expect("serialize RINEX OBS");
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
fn a_phase_shift_correction_its_f8_5_field_cannot_hold_is_refused() {
    // Written in its `F8.5` columns, a correction with more decimals, or wider
    // than eight columns, reads back as another number. The reader refuses it,
    // and the writer refuses a product holding one.
    for body in [
        "G L1C 1e-300 1 G01",
        "G L1C 0.123456",
        "G L1C 1000.00000",
        "G C1C 1.2345678901234567e-300 8 G1 G2 G3 G4 G5 G6 G7 G8",
    ] {
        let err = RinexObs::parse(&minimal_obs_with_phase_shift(body)).expect_err(body);
        assert!(err.to_string().contains("F8.5"), "{body}: {err}");
    }
    let mut obs = RinexObs::parse(&minimal_obs_with_phase_shift("G L1C 0.25000")).expect("parse");
    obs.header.phase_shifts[0].correction_cycles = Some(1e-300);
    assert!(
        matches!(
            obs.to_rinex_string(),
            Err(RinexObsWriteError::ReadBackMismatch { .. })
        ),
        "a correction its field cannot hold was written"
    );
}

#[test]
fn a_phase_shift_list_read_on_one_record_is_written_ten_to_a_record() {
    // Single-digit PRNs arrive in two-column tokens, so 14 satellites fit the
    // 60 columns the parser reads. Re-emitted as `1X,A3` fields they need 66,
    // which used to be refused; RINEX continues a list past ten satellites on
    // `18X,10(1X,A3)` records, and the writer now does.
    let sats: Vec<String> = (1..=14).map(|prn| format!("G{prn}")).collect();
    let body = format!("G C1C 0 14 {}", sats.join(" "));
    assert_eq!(body.len(), 57, "the record must be readable in 60 columns");
    let obs = RinexObs::parse(&minimal_obs_with_phase_shift(&body)).expect("parse");
    assert_eq!(obs.header().phase_shifts[0].satellites.len(), 14);
    let written = obs.to_rinex_string().expect("serialize");
    assert_eq!(
        written
            .lines()
            .filter(|line| line.ends_with("SYS / PHASE SHIFT"))
            .count(),
        2,
        "{written}"
    );
    assert_eq!(
        RinexObs::parse(&written).expect("reads back").header(),
        obs.header()
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
        let reparsed = RinexObs::parse(&obs.to_rinex_string().expect("serialize RINEX OBS"))
            .expect("re-encoded RINEX OBS must reparse");
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

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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

        let reparsed = RinexObs::parse(&obs.to_rinex_string().expect("serialize RINEX OBS"))
            .expect("re-encoded RINEX OBS must reparse");
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
    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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

        // A version 3 epoch record has no picosecond field, so the writer
        // refuses one there; a version 4 record carries it.
        let mut obs = obs;
        if expected_picoseconds.is_some() {
            assert_eq!(
                obs.to_rinex_string(),
                Err(RinexObsWriteError::EpochPicosecondsNotInVersion {
                    epoch_index: 0,
                    version: obs.header().version,
                }),
                "{picoseconds:?}"
            );
            obs.header.version = 4.02;
        }
        let reparsed = RinexObs::parse(&obs.to_rinex_string().expect("serialize RINEX OBS"))
            .expect("re-encoded RINEX OBS must reparse");
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

    // RINEX 4.02 carries five more digits of the second after the clock
    // offset, `1X,I5.5`, and a hundred-satellite line laid out that way is
    // read with them, the clock written or its columns left blank.
    for (trailing, clock) in [
        (
            format!("      {:15.12} {:05}", -0.000_000_000_001, 12_345),
            Some(-0.000_000_000_001),
        ),
        (format!("{}{:05}", " ".repeat(22), 12_345), None),
    ] {
        let (satellites, systems) = multi_constellation_fixture(100);
        let mut body = format!("> 2020 06 24 00 00  0.0000000  0100{trailing}");
        for satellite in &satellites {
            body.push_str(&format!("\n{satellite}   23000000.000"));
        }
        let obs = RinexObs::parse(&obs_with_code_headers(&systems, &body))
            .unwrap_or_else(|error| panic!("{trailing:?}: {error}"));
        let epoch = &obs.epochs()[0];
        assert_eq!(epoch.flag, 0, "{trailing:?}");
        assert_eq!(epoch.sats.len(), 100, "{trailing:?}");
        assert_eq!(epoch.epoch_picoseconds, Some(12_345), "{trailing:?}");
        assert_eq!(epoch.rcv_clock_offset_s, clock, "{trailing:?}");
    }

    // Digits inside the reserved columns straight after the count are no field
    // the format defines, so that line keeps its rejection rather than parsing
    // with them dropped.
    {
        let (satellites, systems) = multi_constellation_fixture(100);
        let mut body = "> 2020 06 24 00 00  0.0000000  0100 00001".to_string();
        for satellite in &satellites {
            body.push_str(&format!("\n{satellite}   23000000.000"));
        }
        assert!(
            RinexObs::parse(&obs_with_code_headers(&systems, &body)).is_err(),
            "digits in the reserved columns must not parse"
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
    assert_eq!(obs.epochs()[0].epoch.map(|time| time.second), Some(1.0));

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
    let reparsed = RinexObs::parse(&obs.to_rinex_string().expect("serialize RINEX OBS"))
        .expect("re-encoded RINEX OBS must reparse");
    assert_eq!(reparsed, obs);
}

#[test]
fn a_blank_glonass_bias_record_clears_the_biases_before_it() {
    // A blank record means the biases are unknown. Beside a record giving a
    // bias in the same header block, which gives its records no order, it
    // contradicts it; in a later block it replaces the biases before it.
    let text = minimal_obs(
        &[
            header_line(" C1C  -71.940", "GLONASS COD/PHS/BIS"),
            header_line("", "GLONASS COD/PHS/BIS"),
        ],
        "",
    );
    let error = RinexObs::parse(&text).expect_err("a blank record beside a bias");
    assert!(error.to_string().contains("contradict"), "{error}");

    let text = minimal_obs(
        &[header_line(" C1C  -71.940", "GLONASS COD/PHS/BIS")],
        &blank_event(4, &[header_line("", "GLONASS COD/PHS/BIS")]),
    );
    let obs = RinexObs::parse(&text).expect("parse a cleared GLONASS bias record");
    assert_eq!(
        obs.header().glonass_cod_phs_bis,
        Some(vec![("C1C".to_string(), Some(-71.94))])
    );
    assert_eq!(
        obs.header_at(0)
            .expect("header in effect")
            .glonass_cod_phs_bis,
        Some(Vec::new())
    );
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
                let canonical = canonical_rinex2_obs_code(system, &declared, 2.11);
                let candidates = rinex2_obs_code_candidates(system, &canonical, 2.11);
                let written = candidates
                    .first()
                    .unwrap_or_else(|| panic!("{system:?} {declared} has no version 2 name"));
                for alternative in &candidates {
                    assert_eq!(
                        canonical_rinex2_obs_code(system, alternative, 2.11),
                        canonical,
                        "{system:?} {declared} lists {alternative} as an inverse"
                    );
                }
                assert_eq!(
                    canonical_rinex2_obs_code(system, written, 2.11),
                    canonical,
                    "{system:?} {declared} became {canonical}, written back as {written}"
                );
            }
        }
    }
}

#[test]
fn a_version_two_header_names_every_position_any_constellation_uses() {
    // Lists of different lengths are not one version 2 list, so writing refuses
    // and says which constellation holds the shorter one. The downgrade names
    // every position any constellation uses, and the shorter list gains a blank
    // code for the extra one.
    let product = two_system_product(
        2.11,
        &(
            GnssSystem::Gps,
            'G',
            1,
            vec!["C1C".to_string(), "L1C".to_string()],
        ),
        &(
            GnssSystem::Galileo,
            'E',
            11,
            vec!["C1X".to_string(), "L1X".to_string(), "C5X".to_string()],
        ),
        true,
    );
    assert!(matches!(
        product.to_rinex_string(),
        Err(RinexObsWriteError::CodeListsNotVersionTwo {
            system: GnssSystem::Gps,
            position: 2,
            code: None,
        })
    ));
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeAdded {
            system: GnssSystem::Gps,
            code: "C5X".to_string(),
        }]
    );
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..24], "     3    C1    L1    C5", "{declared:?}");
}

#[test]
fn a_version_two_code_with_no_version_two_name_is_refused_or_downgraded() {
    // Version 2 has no field for a tracking attribute, so no name reads back as
    // GPS `C1X`. The writer used to write `C1` and return a file that read back
    // as `C1C`. Writing refuses now, and the downgrade makes the rename and says
    // so.
    let text = concat!(
        "     2.11           OBSERVATION DATA    G (GPS)             RINEX VERSION / TYPE\n",
        "     2    C1    L1                                          # / TYPES OF OBSERV\n",
        "  2015     1     1     0     0    0.0000000     GPS         TIME OF FIRST OBS\n",
        "                                                            END OF HEADER\n",
        " 15  1  1  0  0  0.0000000  0  1G 1\n",
        "  20000001.000    100000002.000\n",
    );
    let mut obs = RinexObs::parse(text).expect("parse the version 2 file");
    obs.header
        .obs_codes
        .get_mut(&GnssSystem::Gps)
        .expect("GPS codes")[0] = "C1X".to_string();

    let error = obs
        .to_rinex_string()
        .expect_err("C1X has no version 2 name");
    assert!(
        matches!(
            error,
            RinexObsWriteError::CodeListsNotVersionTwo {
                system: GnssSystem::Gps,
                position: 0,
                ..
            }
        ),
        "{error}"
    );

    let (downgraded, changes) = obs.downgrade_to_rinex2(2.11).expect("downgrade");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeRenamed {
            system: GnssSystem::Gps,
            from: "C1X".to_string(),
            to: "C1C".to_string(),
        }]
    );
    let encoded = downgraded
        .to_rinex_string()
        .expect("the downgraded product writes");
    let declared = encoded
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..18], "     2    C1    L1", "{declared:?}");
    assert_eq!(
        RinexObs::parse(&encoded).expect("reads back").epochs(),
        downgraded.epochs()
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
    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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
    // GPS `C1W` and GLONASS `C1C` are `P1` and `C1`, so no one version 2 list
    // holds both at one position and writing refuses. The downgrade gives each
    // its own column: each constellation gains a blank code for the other's, and
    // every value keeps its own code.
    let product = two_system_product(
        2.11,
        &(GnssSystem::Gps, 'G', 1, vec!["C1W".to_string()]),
        &(GnssSystem::Glonass, 'R', 2, vec!["C1C".to_string()]),
        true,
    );
    assert!(matches!(
        product.to_rinex_string(),
        Err(RinexObsWriteError::CodeListsNotVersionTwo { .. })
    ));
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..18], "     2    P1    C1", "{declared:?}");
    assert!(
        changes.contains(&ObsDowngradeChange::CodeAdded {
            system: GnssSystem::Gps,
            code: "C1C".to_string(),
        }) && changes.contains(&ObsDowngradeChange::CodeAdded {
            system: GnssSystem::Glonass,
            code: "C1P".to_string(),
        }),
        "{changes:?}"
    );
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(
        read.header().obs_codes[&GnssSystem::Gps],
        vec!["C1W".to_string(), "C1C".to_string()]
    );
    assert_eq!(
        read.header().obs_codes[&GnssSystem::Glonass],
        vec!["C1P".to_string(), "C1C".to_string()]
    );
    let values = |system: GnssSystem, prn: u8| -> Vec<Option<f64>> {
        read.epochs()[0].sats[&GnssSatelliteId { system, prn }]
            .iter()
            .map(|v| v.value)
            .collect()
    };
    assert_eq!(values(GnssSystem::Gps, 1), vec![Some(1000.0), None]);
    assert_eq!(values(GnssSystem::Glonass, 2), vec![None, Some(2000.0)]);
}

#[test]
fn a_split_column_carries_its_observation_counts_too() {
    // `PRN / # OF OBS` counts are aligned to a constellation's own code list.
    // When the downgrade gives a code a column of its own, its count moves with
    // it, or the count would sit under the name beside the one it counts.
    let mut product = two_system_product(
        2.11,
        &(GnssSystem::Gps, 'G', 1, vec!["C1W".to_string()]),
        &(GnssSystem::Glonass, 'R', 2, vec!["C1C".to_string()]),
        true,
    );
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let glonass = GnssSatelliteId {
        system: GnssSystem::Glonass,
        prn: 2,
    };
    product.header.prn_obs_counts.insert(gps, vec![Some(5)]);
    product.header.prn_obs_counts.insert(glonass, vec![Some(7)]);
    assert!(matches!(
        product.to_rinex_string(),
        Err(RinexObsWriteError::CodeListsNotVersionTwo { .. })
    ));

    let (downgraded, _) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..18], "     2    P1    C1");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.header().prn_obs_counts[&gps], vec![Some(5), None]);
    assert_eq!(
        read.header().prn_obs_counts[&glonass],
        vec![None, Some(7)],
        "the count sits under C1, the column GLONASS uses"
    );
}

#[test]
fn a_version_two_header_does_not_repeat_a_column_it_already_has() {
    // Two constellations holding the same two signals in opposite order are not
    // one version 2 list, so writing refuses. The downgrade lays them out in two
    // columns, not four, moving GLONASS's codes and saying so, with nothing
    // renamed or added.
    let product = two_system_product(
        2.11,
        &(
            GnssSystem::Gps,
            'G',
            1,
            vec!["C1C".to_string(), "C1W".to_string()],
        ),
        &(
            GnssSystem::Glonass,
            'R',
            2,
            vec!["C1P".to_string(), "C1C".to_string()],
        ),
        true,
    );
    assert!(matches!(
        product.to_rinex_string(),
        Err(RinexObsWriteError::CodeListsNotVersionTwo { .. })
    ));
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    assert!(
        changes
            .iter()
            .all(|change| matches!(change, ObsDowngradeChange::CodeMoved { .. })),
        "{changes:?}"
    );
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..18], "     2    C1    P1", "{declared:?}");

    // A code version 2 cannot spell shares the column of the one it becomes.
    let product = two_system_product(
        2.11,
        &(GnssSystem::Gps, 'G', 1, vec!["C1X".to_string()]),
        &(GnssSystem::Glonass, 'R', 2, vec!["C1C".to_string()]),
        true,
    );
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeRenamed {
            system: GnssSystem::Gps,
            from: "C1X".to_string(),
            to: "C1C".to_string(),
        }]
    );
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..12], "     1    C1", "{declared:?}");
}

#[test]
fn a_version_two_observation_type_wider_than_its_field_is_rejected() {
    // `# / TYPES OF OBSERV` is `9(4X,A2)`. A three-character token means the
    // line is not in that layout, and the code could not be written back into a
    // two-character field: it was kept, written as two characters, and read
    // back as a different code, which the fuzz round trip fails on.
    let text = version_two_fixture_text().replace(
        "     8    L1    L2    C1    C2    P2    P1    S1    S2",
        "     8   L1X    L2    C1    C2    P2    P1    S1    S2",
    );
    let error = RinexObs::parse(&text).expect_err("a three-character version 2 code is rejected");
    assert!(
        error.to_string().contains("field width"),
        "the error names the field: {error}"
    );
}

#[test]
fn a_shared_column_is_a_name_every_constellation_in_it_may_carry() {
    // GPS `C1W` is `P1`, and Galileo reads `P1` back as the code it holds, but
    // version 2 gives Galileo no `P` observable, so no single column carries
    // both and writing refuses. The downgrade gives each its own column, and
    // each value comes back under its own code with nothing renamed.
    let product = two_system_product(
        2.11,
        &(GnssSystem::Gps, 'G', 1, vec!["C1W".to_string()]),
        &(GnssSystem::Galileo, 'E', 11, vec!["C1X".to_string()]),
        true,
    );
    assert!(matches!(
        product.to_rinex_string(),
        Err(RinexObsWriteError::CodeListsNotVersionTwo { .. })
    ));
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    assert!(
        !changes
            .iter()
            .any(|change| matches!(change, ObsDowngradeChange::CodeRenamed { .. })),
        "{changes:?}"
    );
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..18], "     2    P1    C1", "{declared:?}");
    let read = RinexObs::parse(&text).expect("reads back");
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let galileo = GnssSatelliteId {
        system: GnssSystem::Galileo,
        prn: 11,
    };
    assert_eq!(read.header().obs_codes[&GnssSystem::Gps][0], "C1W");
    assert_eq!(read.header().obs_codes[&GnssSystem::Galileo][1], "C1X");
    let values = |sat: GnssSatelliteId| -> Vec<Option<f64>> {
        read.epochs()[0].sats[&sat]
            .iter()
            .map(|v| v.value)
            .collect()
    };
    assert_eq!(values(gps), vec![Some(1000.0), None]);
    assert_eq!(values(galileo), vec![None, Some(2000.0)]);
}

#[test]
fn a_beidou_band_version_two_cannot_name_is_a_known_limit() {
    // Version 2 numbers by frequency slot, and no slot names BeiDou B1C. A
    // product holding one, written as version 2, comes back as B1I: the slot
    // that digit does name. There is nothing else the file can say, and
    // `to_rinex_string` has no way to refuse.
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::BeiDou, "C1", 2.11),
        "C2I"
    );
    assert_eq!(
        rinex2_obs_code_candidates(GnssSystem::BeiDou, "C1P", 2.11),
        vec!["C1".to_string()],
        "no version 2 name reads back as B1C"
    );
}

#[test]
fn a_version_two_header_shares_one_column_where_the_constellations_agree() {
    // GPS `[C1W, C1C]` beside GLONASS `[C1P, C1C]` is exactly what the version 2
    // list `P1 C1` reads as for each of them, so it writes as it is, with no
    // downgrade and no columns split.
    let product = two_system_product(
        2.11,
        &(
            GnssSystem::Gps,
            'G',
            1,
            vec!["C1W".to_string(), "C1C".to_string()],
        ),
        &(
            GnssSystem::Glonass,
            'R',
            2,
            vec!["C1P".to_string(), "C1C".to_string()],
        ),
        true,
    );
    let text = product
        .to_rinex_string()
        .expect("one version 2 list states both");
    let declared = text
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(&declared[..18], "     2    P1    C1", "{declared:?}");
    assert_eq!(
        RinexObs::parse(&text).expect("reads back").epochs(),
        product.epochs()
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

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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
    // A version 2 reader takes a value for every code, for every satellite. A
    // satellite holding fewer values than its constellation has codes would read
    // back with blanks the product does not hold, so writing it as it is refuses
    // and names the satellite. Holding the blanks explicitly writes, and the
    // satellites beside it are untouched.
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

    let error = obs
        .to_rinex_string()
        .expect_err("a short record is refused");
    assert!(error.to_string().contains(&short.to_string()), "{error}");

    obs.epochs[0]
        .sats
        .get_mut(&short)
        .expect("the second satellite")
        .resize(
            8,
            ObsValue {
                value: None,
                lli: None,
                ssi: None,
            },
        );
    let encoded = obs.to_rinex_string().expect("blanks held explicitly write");
    let reparsed = RinexObs::parse(&encoded).expect("reads back");
    assert_eq!(reparsed.epochs()[0].sats, obs.epochs()[0].sats);
}

#[test]
fn counts_declared_before_their_observation_types_are_kept() {
    // RINEX 3.05 does not fix the order of `PRN / # OF OBS` and
    // `SYS / # / OBS TYPES`. Counts read before their constellation's types
    // used to parse as no counts at all, which the writer then refused.
    let text = obs_with_code_headers(
        &[
            header_line("   G01     5", "PRN / # OF OBS"),
            header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        ],
        "> 2015 01 01 00 00  0.0000000  0  1\nG01  20000000.000\n",
    );
    let obs = RinexObs::parse(&text).expect("parse");
    assert_eq!(obs.skipped_records, 0);
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    assert_eq!(obs.header().prn_obs_counts[&gps], vec![Some(5)]);
    let written = obs.to_rinex_string().expect("writes");
    let read = RinexObs::parse(&written).expect("reads back");
    assert_eq!(read.header().prn_obs_counts, obs.header().prn_obs_counts);
}

#[test]
fn a_malformed_count_is_reported_where_its_types_are_known() {
    // Counts after their types are read at once, so a malformed one is the
    // error the parse reports, not the header's missing end that follows.
    let text = [
        header_line(
            "     3.05           OBSERVATION DATA    M (MIXED)",
            "RINEX VERSION / TYPE",
        ),
        header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        header_line("   G01    x5", "PRN / # OF OBS"),
    ]
    .join("\n");
    let error = RinexObs::parse(&text).expect_err("a malformed count is refused");
    assert!(error.to_string().contains("prn_obs_count"), "{error}");
}

#[test]
fn a_malformed_count_is_not_held_back_by_another_constellations_types() {
    // GLONASS declares fourteen types and has given thirteen when GPS, whose
    // one type is complete, gives a malformed count. That count is the error,
    // not the missing end of a header GLONASS never finished.
    let text = [
        header_line(
            "     3.05           OBSERVATION DATA    M (MIXED)",
            "RINEX VERSION / TYPE",
        ),
        header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        header_line(
            "R   14 C1C C1P C2C C2P L1C L1P L2C L2P D1C D1P D2C D2P S1C",
            "SYS / # / OBS TYPES",
        ),
        header_line("   G01    x5", "PRN / # OF OBS"),
    ]
    .join("\n");
    let error = RinexObs::parse(&text).expect_err("a malformed count is refused");
    assert!(error.to_string().contains("prn_obs_count"), "{error}");
}

#[test]
fn a_version_two_value_with_more_than_three_decimals_is_refused() {
    // An F14.3 field holds three decimals. A version 2 value with a fourth was
    // accepted and written back as a different number.
    let line = |content: &str, label: &str| format!("{content:<60}{label}\n");
    let text = [
        line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        line("     1    C1", "# / TYPES OF OBSERV"),
        line("", "END OF HEADER"),
        " 15  1  1  0  0  0.0000000  0  1G 1\n".to_string(),
        "        0.0001\n".to_string(),
    ]
    .concat();
    let error = RinexObs::parse(&text).expect_err("a fourth decimal is refused");
    assert!(
        error
            .to_string()
            .contains("is not representable in its F14.3"),
        "{error}"
    );
}

#[test]
fn a_version_two_scale_factor_is_refused_by_the_writer_and_removed_by_the_downgrade() {
    // Version 2 values read through a `SYS / SCALE FACTOR` are divided by it.
    // A version 2 reader that does not know the record, RTKLIB among them,
    // would take scaled numbers written back as physical ones, so the writer
    // refuses the product. The downgrade removes the record, writes the
    // physical values, and rounds and reports any value carrying more decimals
    // than a field without the factor holds.
    let line = |content: &str, label: &str| format!("{content:<60}{label}\n");
    let text = [
        line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        line("     2    C1    L1", "# / TYPES OF OBSERV"),
        line("G   10   0", "SYS / SCALE FACTOR"),
        line(
            "  2015     1     1     0     0    0.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
        line("", "END OF HEADER"),
        " 15  1  1  0  0  0.0000000  0  1G 1\n".to_string(),
        " 200000010.000        1230.001\n".to_string(),
    ]
    .concat();
    let obs = RinexObs::parse(&text).expect("parse a scaled version 2 file");
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    assert_eq!(obs.epochs()[0].sats[&gps][0].value, Some(20_000_001.0));
    assert!(matches!(
        obs.to_rinex_string(),
        Err(RinexObsWriteError::ScaleFactorsInVersionTwo { count: 1 })
    ));

    let (downgraded, changes) = obs.downgrade_to_rinex2(2.11).expect("downgrade");
    let held = obs.epochs()[0].sats[&gps][1].value.expect("L1 value");
    assert_eq!(
        changes,
        vec![
            ObsDowngradeChange::ScaleFactorsRemoved { count: 1 },
            ObsDowngradeChange::ValueRounded {
                epoch_index: 0,
                satellite: gps,
                code: "L1C".to_string(),
                from: held,
                to: 123.0,
            },
        ]
    );
    let plain = downgraded.to_rinex_string().expect("the downgrade writes");
    assert!(!plain.contains("SYS / SCALE FACTOR"));
    assert!(plain.contains("  20000001.000         123.000"), "{plain}");
    assert_eq!(
        RinexObs::parse(&plain).expect("reads back").epochs(),
        downgraded.epochs()
    );

    let mut fixture = version_two_fixture();
    fixture.header.scale_factors.push(super::ObsScaleFactor {
        system: GnssSystem::Gps,
        factor: 10.0,
        codes: Vec::new(),
    });
    assert!(matches!(
        fixture.to_rinex_string(),
        Err(RinexObsWriteError::ScaleFactorsInVersionTwo { count: 1 })
    ));
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
    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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
    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    assert!(encoded.contains("a comment carried by the event"));
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

#[test]
fn a_version_two_event_epoch_names_no_satellites() {
    // An event epoch declares the records that follow it and holds no
    // observations. One that still holds satellites says two things at once, and
    // the writer used to write the event and drop the satellites. It refuses now,
    // and an event that holds none is written as one.
    let mut obs = version_two_fixture();
    obs.epochs
        .first_mut()
        .expect("the fixture has an epoch")
        .flag = 4;
    let error = obs
        .to_rinex_string()
        .expect_err("an event holding observations is refused");
    assert!(error.to_string().contains("satellites"), "{error}");

    obs.epochs[0].sats.clear();
    let encoded = obs
        .to_rinex_string()
        .expect("an event without observations writes");
    let line = encoded
        .lines()
        .find(|line| line.starts_with(" 15  1  1  0  0  0.0000000"))
        .expect("the event record is written");
    assert_eq!(line.trim_end(), " 15  1  1  0  0  0.0000000  4  0");
}

#[test]
fn a_version_three_cycle_slip_epoch_reads_its_records_as_slips() {
    // RINEX 3.05 Table A3: flag 6 is followed by cycle slip records in the
    // observation record layout, a slip in place of an observation. They were
    // kept as uninterpreted event text; they are slips, held apart from the
    // observations, and written back in the same layout.
    let slip_line = format!("G01{}{:14.3}", blank_obs_field(), 1.0);
    let text = [
        "     3.05           OBSERVATION DATA    M                   RINEX VERSION / TYPE"
            .to_string(),
        "G    2 C1C L1C                                              SYS / # / OBS TYPES"
            .to_string(),
        "                                                            END OF HEADER".to_string(),
        "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
        format!(
            "G01{}{}",
            obs_field(20_000_000.0, 0, 7),
            obs_field(100_000.125, 0, 7)
        ),
        "> 2020 01 01 00 00 30.0000000  6  1".to_string(),
        slip_line.clone(),
        "> 2020 01 01 00 01  0.0000000  0  1".to_string(),
        format!(
            "G01{}{}",
            obs_field(20_000_100.0, 0, 7),
            obs_field(100_300.25, 1, 7)
        ),
    ]
    .join("\n");
    let obs = RinexObs::parse(&text).expect("parse a file with cycle slip records");
    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("G01");
    let slips = &obs.epochs()[1];
    assert_eq!(slips.flag, CYCLE_SLIP_FLAG);
    assert!(slips.sats.is_empty());
    assert!(slips.special_records.is_empty());
    assert_eq!(slips.declared_record_count, 1);
    assert_eq!(
        slips.cycle_slips[&g01],
        vec![
            ObsValue {
                value: None,
                lli: None,
                ssi: None
            },
            ObsValue {
                value: Some(1.0),
                lli: None,
                ssi: None
            },
        ]
    );
    assert!(obs.epochs()[0].cycle_slips.is_empty());
    assert!(obs.epochs()[2].cycle_slips.is_empty());

    // Slips feed no observation consumer.
    let policy = SignalPolicy::default_for(3.05).expect("policy");
    assert!(pseudoranges(&obs, slips, &policy)
        .expect("pseudoranges")
        .is_empty());
    let report = crate::observation_qc::observation_qc(&obs);
    assert_eq!(report.observation_epochs, 2);
    assert_eq!(report.event_records, 1);

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    assert!(encoded.contains("> 2020 01 01 00 00 30.0000000  6  1\n"));
    assert!(encoded.contains(&format!("{slip_line}\n")), "{encoded}");
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

#[test]
fn a_version_two_cycle_slip_epoch_reads_its_satellite_list_and_records() {
    // RINEX 2.11: flag 6's count is satellites, listed on the epoch line, each
    // followed by its record in the observation layout. Taken as that many
    // verbatim lines, a record continued past five types, or a list continued
    // past twelve satellites, left lines to be read as epochs.
    let names = ["C1", "L1", "D1", "S1", "C2", "L2"];
    let types = format!(
        "{:6}{}",
        names.len(),
        names
            .iter()
            .map(|name| format!("{name:>6}"))
            .collect::<String>()
    );
    let t0 = ObsEpochTime {
        year: 2015,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let t1 = ObsEpochTime { second: 30.0, ..t0 };
    let record = |base: f64| {
        let fields: Vec<String> = (0..6)
            .map(|index| obs_field(base + f64::from(index), 0, 6))
            .collect();
        format!("{}\n{}", fields[..5].concat(), fields[5])
    };
    let slip_record = format!("{}{:14.3}\n{:14.3}", blank_obs_field(), -2.0, 3.0);
    let text = [
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line(&types, "# / TYPES OF OBSERV"),
        header_line("", "END OF HEADER"),
        v2_epoch_line(t0, 0, 2, "G01G02"),
        record(20_000_000.0),
        record(21_000_000.0),
        v2_epoch_line(t1, 6, 1, "G02"),
        slip_record,
    ]
    .join("\n");
    let obs = RinexObs::parse(&text).expect("parse a version 2 file with cycle slip records");
    assert_eq!(obs.epochs().len(), 2);
    let slips = &obs.epochs()[1];
    let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).expect("G02");
    assert_eq!(slips.flag, CYCLE_SLIP_FLAG);
    assert!(slips.sats.is_empty());
    let values: Vec<Option<f64>> = slips.cycle_slips[&g02]
        .iter()
        .map(|value| value.value)
        .collect();
    assert_eq!(values, vec![None, Some(-2.0), None, None, None, Some(3.0)]);

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

#[test]
fn a_downgrade_moves_and_rounds_cycle_slips_with_their_codes() {
    // GPS `C1W` and GLONASS `C1C` take one version 2 column each, so GLONASS's
    // code moves, and its slip has to move with it. The GLONASS slip was read
    // through a scale factor of 10 and carries four decimals, which a version 2
    // field cannot hold: it is rounded and reported as a slip.
    let text = concat!(
        "     3.05           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n",
        "G    1 C1W                                                  SYS / # / OBS TYPES\n",
        "R    1 C1C                                                  SYS / # / OBS TYPES\n",
        "R   10   1 C1C                                              SYS / SCALE FACTOR\n",
        "                                                            END OF HEADER\n",
        "> 2020 01 01 00 00  0.0000000  0  2\n",
        "G01  20000000.000\n",
        "R02 200000001.230\n",
        "> 2020 01 01 00 00 30.0000000  6  2\n",
        "G01         1.000\n",
        "R02        12.345\n",
    );
    let obs = RinexObs::parse(text).expect("parse");
    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("G01");
    let r02 = GnssSatelliteId::new(GnssSystem::Glonass, 2).expect("R02");
    let held = obs.epochs()[1].cycle_slips[&r02][0]
        .value
        .expect("the GLONASS slip");
    assert_eq!(obs.epochs()[1].cycle_slips[&g01][0].value, Some(1.0));

    let (downgraded, changes) = obs.downgrade_to_rinex2(2.11).expect("downgrade");
    let rounded: f64 = format!("{held:.3}").parse().expect("rounded slip");
    assert!(
        changes.contains(&ObsDowngradeChange::CycleSlipRounded {
            epoch_index: 1,
            satellite: r02,
            code: "C1C".to_string(),
            from: held,
            to: rounded,
        }),
        "{changes:?}"
    );
    let at = |system: GnssSystem, code: &str| {
        downgraded.header().obs_codes[&system]
            .iter()
            .position(|held| held == code)
            .unwrap_or_else(|| panic!("{system} holds {code}"))
    };
    let glonass = &downgraded.epochs()[1].cycle_slips[&r02];
    for (index, value) in glonass.iter().enumerate() {
        let expected = (index == at(GnssSystem::Glonass, "C1C")).then_some(rounded);
        assert_eq!(
            value.value, expected,
            "GLONASS slip at {index}: {glonass:?}"
        );
    }
    let gps = &downgraded.epochs()[1].cycle_slips[&g01];
    for (index, value) in gps.iter().enumerate() {
        let expected = (index == at(GnssSystem::Gps, "C1W")).then_some(1.0);
        assert_eq!(value.value, expected, "GPS slip at {index}: {gps:?}");
    }
    let written = downgraded.to_rinex_string().expect("the downgrade writes");
    let read = RinexObs::parse(&written).expect("reads back");
    assert_eq!(read.epochs(), downgraded.epochs());
}

#[test]
fn an_event_without_a_significant_epoch_leaves_its_epoch_fields_blank() {
    // RINEX 3.05 Table A3: "For events without significant epoch the epoch
    // fields in the EPOCH RECORD can be left blank". The reader refused every
    // such record. It holds no time now, and is written back blank.
    let comment = |text: &str| format!("{text:<60}COMMENT");
    let text = [
        "     3.05           OBSERVATION DATA    M                   RINEX VERSION / TYPE"
            .to_string(),
        "G    1 C1C                                                  SYS / # / OBS TYPES"
            .to_string(),
        "                                                            END OF HEADER".to_string(),
        "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
        "G01  20000000.000".to_string(),
        // In the record's columns: the flag in column 32, the count after it.
        format!(">{:30}4  1", ""),
        comment("written in the record's columns"),
        // Spaced as the specification's own example writes it.
        format!(">{:33}4 1", ""),
        comment("written as the example writes it"),
        "> 2020 01 01 00 00 30.0000000  5  0".to_string(),
        "> 2020 01 01 00 01  0.0000000  0  1".to_string(),
        "G01  20000100.000".to_string(),
    ]
    .join("\n");
    let obs = RinexObs::parse(&text).expect("parse events with blank epoch fields");
    let read: Vec<(u8, bool, usize)> = obs
        .epochs()
        .iter()
        .map(|epoch| {
            (
                epoch.flag,
                epoch.epoch.is_some(),
                epoch.special_records.len(),
            )
        })
        .collect();
    assert_eq!(
        read,
        vec![
            (0, true, 0),
            (4, false, 1),
            (4, false, 1),
            (5, true, 0),
            (0, true, 0)
        ]
    );

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    assert_eq!(
        encoded
            .lines()
            .filter(|line| *line == format!(">{:30}4  1", ""))
            .count(),
        2,
        "{encoded}"
    );
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());

    // An observation or cycle slip record is tagged with its time.
    for flag in [0, 1, CYCLE_SLIP_FLAG] {
        let text = minimal_obs(&[], &format!(">{:30}{flag}  1\nG01  20000000.000", ""));
        let error = RinexObs::parse(&text).expect_err("blank epoch fields are refused");
        assert!(
            error.to_string().contains("leaves its epoch fields blank"),
            "flag {flag}: {error}"
        );
    }
    let mut untimed = obs.clone();
    untimed.epochs[0].epoch = None;
    assert_eq!(
        untimed.to_rinex_string(),
        Err(RinexObsWriteError::EpochTimeMissing {
            epoch_index: 0,
            flag: 0
        })
    );
}

#[test]
fn a_version_two_event_without_a_significant_epoch_leaves_its_epoch_fields_blank() {
    let t0 = ObsEpochTime {
        year: 2015,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let t1 = ObsEpochTime { second: 30.0, ..t0 };
    let text = [
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line("     1    C1", "# / TYPES OF OBSERV"),
        header_line("", "END OF HEADER"),
        v2_epoch_line(t0, 0, 1, "G01"),
        "  20000000.000".to_string(),
        format!("{:28}4  1", ""),
        header_line("a comment after a blank event", "COMMENT"),
        v2_epoch_line(t1, 0, 1, "G01"),
        "  20000100.000".to_string(),
    ]
    .join("\n");
    let obs = RinexObs::parse(&text).expect("parse a version 2 blank event");
    assert_eq!(obs.epochs().len(), 3);
    assert_eq!(obs.epochs()[1].epoch, None);
    assert_eq!(obs.epochs()[1].flag, 4);
    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    assert!(
        encoded.lines().any(|line| line == format!("{:28}4  1", "")),
        "{encoded}"
    );
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

#[test]
fn a_blank_event_declaring_a_hundred_records_or_more_reads_and_writes_back() {
    // RINEX 2.11 Table A2 allows an event up to 999 special records, and a
    // count of 100 or more fills its I3 field, so it abuts the flag: a blank
    // version 2 event line reads `4100` as one token after 28 blank columns.
    let comments: Vec<String> = (0..100)
        .map(|index| header_line(&format!("record {index}"), "COMMENT"))
        .collect();
    let t0 = ObsEpochTime {
        year: 2015,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let mut lines = vec![
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line("     1    C1", "# / TYPES OF OBSERV"),
        header_line("", "END OF HEADER"),
        format!("{}4100", " ".repeat(28)),
    ];
    lines.extend(comments.iter().cloned());
    lines.push(v2_epoch_line(t0, 0, 1, "G01"));
    lines.push("  20000000.000".to_string());
    let obs = RinexObs::parse(&lines.join("\n")).expect("parse a version 2 event of 100 records");
    assert_eq!(obs.epochs()[0].epoch, None);
    assert_eq!(obs.epochs()[0].flag, 4);
    assert_eq!(obs.epochs()[0].special_records, comments);
    let encoded = obs.to_rinex_string().expect("serialize");
    assert!(
        encoded
            .lines()
            .any(|line| line == format!("{}4100", " ".repeat(28))),
        "{encoded}"
    );
    assert_eq!(
        RinexObs::parse(&encoded).expect("reads back").epochs(),
        obs.epochs()
    );

    // The version 3 line holds its count in its own columns.
    let mut body = vec![format!(">{:30}4100", "")];
    body.extend(comments.iter().cloned());
    body.push("> 2020 01 01 00 00  0.0000000  0  1".to_string());
    body.push("G01  20000000.000".to_string());
    let obs = RinexObs::parse(&minimal_obs(&[], &body.join("\n")))
        .expect("parse a version 3 event of 100 records");
    assert_eq!(obs.epochs()[0].epoch, None);
    assert_eq!(obs.epochs()[0].special_records.len(), 100);
    let encoded = obs.to_rinex_string().expect("serialize");
    assert_eq!(
        RinexObs::parse(&encoded).expect("reads back").epochs(),
        obs.epochs()
    );
}

fn rinex211_table_a7_example() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/rinex211_table_a7_example.rnx"
    );
    std::fs::read_to_string(path).expect("read the RINEX 2.11 Table A7 example")
}

#[test]
fn the_rinex_211_example_file_reads_and_writes_back() {
    // Table A7 of RINEX 2.11 as the specification prints it: a flag 2 event, a
    // new site occupation (flag 3) and header records (flag 4) with blank epoch
    // fields, an external event (flag 5) with a significant epoch and no
    // records, and cycle slip records (flag 6).
    let obs = RinexObs::parse(&rinex211_table_a7_example())
        .expect("parse the specification's example file");
    let read: Vec<(u8, bool)> = obs
        .epochs()
        .iter()
        .map(|epoch| (epoch.flag, epoch.epoch.is_some()))
        .collect();
    assert_eq!(
        read,
        vec![
            (0, true),
            (4, true),
            (0, true),
            (2, true),
            (0, true),
            (3, false),
            (0, true),
            (5, true),
            (4, false),
            (0, true),
            (4, false),
            (6, true),
            (4, false),
            (0, true),
            (4, false),
        ]
    );
    let slips = &obs.epochs()[11].cycle_slips;
    let values = |prn: u8| -> Vec<Option<f64>> {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, prn).expect("satellite");
        slips[&sat].iter().map(|value| value.value).collect()
    };
    assert_eq!(
        values(16),
        vec![None, Some(123_456_789.0), Some(-9_876_543.5), None, None]
    );
    assert_eq!(values(9), vec![None, Some(0.0), Some(-0.5), None, None]);

    let encoded = obs.to_rinex_string().expect("serialize the example");
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(reparsed.epochs(), obs.epochs());
}

/// Write a product, read the text back, and require the same product.
fn assert_writes_back(obs: &RinexObs) -> String {
    let encoded = obs.to_rinex_string().expect("the product writes");
    let reparsed = RinexObs::parse(&encoded).expect("its own output must read back");
    assert_eq!(&reparsed, obs);
    encoded
}

fn obs_record(sat: &str, values: &[Option<f64>]) -> String {
    let mut line = sat.to_string();
    for value in values {
        match value {
            Some(value) => line.push_str(&format!("{value:14.3}  ")),
            None => line.push_str(&blank_obs_field()),
        }
    }
    line.trim_end().to_string()
}

fn blank_event(flag: u8, records: &[String]) -> String {
    let mut lines = vec![format!(">{:30}{flag}{:3}", "", records.len())];
    lines.extend(records.iter().cloned());
    lines.join("\n")
}

fn values_of(obs: &RinexObs, epoch: usize, sat: &str) -> Vec<Option<f64>> {
    let sat: GnssSatelliteId = sat.parse().expect("satellite");
    obs.epochs()[epoch].sats[&sat]
        .iter()
        .map(|value| value.value)
        .collect()
}

fn codes(list: &[&str]) -> Vec<String> {
    list.iter().map(|code| (*code).to_string()).collect()
}

/// A version 3 file whose flag 4 events declare a longer, reordered GPS list
/// and then a shorter one, and leave GLONASS's alone.
fn version_three_type_change_text() -> String {
    obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("R    1 C1C", "SYS / # / OBS TYPES"),
        ],
        &[
            "> 2020 01 01 00 00  0.0000000  0  2".to_string(),
            obs_record("G01", &[Some(20_000_000.0), Some(100_000.0)]),
            obs_record("R02", &[Some(21_000_000.0)]),
            blank_event(
                4,
                &[header_line("G    3 L1C C1C S1C", "SYS / # / OBS TYPES")],
            ),
            "> 2020 01 01 00 00 30.0000000  0  2".to_string(),
            obs_record("G01", &[Some(100_030.0), Some(20_000_030.0), Some(45.0)]),
            obs_record("R02", &[Some(21_000_030.0)]),
            blank_event(4, &[header_line("G    1 C1C", "SYS / # / OBS TYPES")]),
            "> 2020 01 01 00 01  0.0000000  0  1".to_string(),
            obs_record("G01", &[Some(20_000_060.0)]),
        ]
        .join("\n"),
    )
}

#[test]
fn an_event_declaring_a_type_list_reads_the_epochs_after_it_by_that_list() {
    // RINEX 3.05 section 5.3.2: flag 4, "header information follows". The
    // epochs after a SYS / # / OBS TYPES record read by it. They were read by
    // the file header's list, so a longer list left values unread, a reordered
    // one put them under the wrong codes, and a shorter one read blanks as
    // missing values.
    let obs = RinexObs::parse(&version_three_type_change_text()).expect("parse");
    let header = obs.header();
    assert_eq!(
        header.obs_codes[&GnssSystem::Gps],
        codes(&["C1C", "L1C", "S1C"])
    );
    assert_eq!(
        header.declared_obs_codes[&GnssSystem::Gps],
        codes(&["C1C", "L1C"])
    );
    assert_eq!(header.obs_codes[&GnssSystem::Glonass], codes(&["C1C"]));
    assert_eq!(
        values_of(&obs, 0, "G01"),
        vec![Some(20_000_000.0), Some(100_000.0), None]
    );
    assert_eq!(
        values_of(&obs, 2, "G01"),
        vec![Some(20_000_030.0), Some(100_030.0), Some(45.0)]
    );
    assert_eq!(
        values_of(&obs, 4, "G01"),
        vec![Some(20_000_060.0), None, None]
    );
    assert_eq!(values_of(&obs, 2, "R02"), vec![Some(21_000_030.0)]);

    let declared = |index: usize| {
        obs.header_at(index)
            .expect("header in effect")
            .declared_obs_codes[&GnssSystem::Gps]
            .clone()
    };
    assert_eq!(declared(0), codes(&["C1C", "L1C"]));
    assert_eq!(declared(1), codes(&["L1C", "C1C", "S1C"]));
    assert_eq!(declared(2), codes(&["L1C", "C1C", "S1C"]));
    assert_eq!(declared(3), codes(&["C1C"]));
    assert_eq!(declared(4), codes(&["C1C"]));
    let timeline = obs.header_timeline().expect("timeline");
    for index in 0..obs.epochs().len() {
        let at = obs.header_at(index).expect("header in effect");
        assert_eq!(at.obs_codes, header.obs_codes, "the union at {index}");
        assert_eq!(
            at.declared_obs_codes[&GnssSystem::Glonass],
            codes(&["C1C"]),
            "{index}"
        );
        assert_eq!(timeline.at(index), &at, "epoch {index}");
    }
    assert_eq!(
        timeline
            .segments()
            .map(|(first, _)| first)
            .collect::<Vec<_>>(),
        vec![0, 1, 3]
    );
    assert!(obs.header_at(obs.epochs().len()).is_err());

    // Written back, the header declares its own list, the events carry theirs,
    // and each epoch is written by the list in effect at it.
    let encoded = assert_writes_back(&obs);
    assert!(encoded.contains("\nG    2 C1C L1C"), "{encoded}");
    assert!(
        encoded.contains(&format!(
            "\n{}\n",
            obs_record("G01", &[Some(100_030.0), Some(20_000_030.0), Some(45.0)])
        )),
        "{encoded}"
    );

    // Observation QC expects the codes in effect at each epoch only.
    let report = crate::observation_qc::observation_qc(&obs);
    let gps = report
        .systems
        .iter()
        .find(|system| system.system == GnssSystem::Gps)
        .expect("GPS");
    assert_eq!(gps.expected_observations, 2 + 3 + 1);
}

#[test]
fn a_version_two_event_declaring_types_reads_the_epochs_after_it_by_them() {
    let at = |minute: u8, second: f64| ObsEpochTime {
        year: 2015,
        month: 1,
        day: 1,
        hour: 0,
        minute,
        second,
    };
    let values = |values: &[Option<f64>]| obs_record("", values);
    let text = [
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line("     2    C1    L1", "# / TYPES OF OBSERV"),
        header_line("", "END OF HEADER"),
        v2_epoch_line(at(0, 0.0), 0, 1, "G01"),
        values(&[Some(20_000_000.0), Some(100_000.0)]),
        format!("{:28}4  1", ""),
        header_line("     3    L1    C1    S1", "# / TYPES OF OBSERV"),
        v2_epoch_line(at(0, 30.0), 0, 1, "G01"),
        values(&[Some(100_030.0), Some(20_000_030.0), Some(45.0)]),
        format!("{:28}4  1", ""),
        header_line("     1    C1", "# / TYPES OF OBSERV"),
        v2_epoch_line(at(1, 0.0), 0, 1, "G01"),
        values(&[Some(20_000_060.0)]),
    ]
    .join("\n");
    let obs = RinexObs::parse(&text).expect("parse");
    assert_eq!(obs.header().rinex2_types, codes(&["C1", "L1"]));
    assert_eq!(
        obs.header().obs_codes[&GnssSystem::Gps],
        codes(&["C1C", "L1C", "S1C"])
    );
    assert_eq!(
        obs.header().declared_obs_codes[&GnssSystem::Gps],
        codes(&["C1C", "L1C"])
    );
    assert_eq!(
        values_of(&obs, 0, "G01"),
        vec![Some(20_000_000.0), Some(100_000.0), None]
    );
    assert_eq!(
        values_of(&obs, 2, "G01"),
        vec![Some(20_000_030.0), Some(100_030.0), Some(45.0)]
    );
    assert_eq!(
        values_of(&obs, 4, "G01"),
        vec![Some(20_000_060.0), None, None]
    );
    let after = obs.header_at(2).expect("header in effect");
    assert_eq!(after.rinex2_types, codes(&["L1", "C1", "S1"]));
    assert_eq!(
        after.declared_obs_codes[&GnssSystem::Gps],
        codes(&["L1C", "C1C", "S1C"])
    );
    assert_eq!(
        obs.header_at(4).expect("header in effect").rinex2_types,
        codes(&["C1"])
    );
    let encoded = assert_writes_back(&obs);
    assert!(encoded.contains("     2    C1    L1"), "{encoded}");
}

#[test]
fn a_scale_factor_an_event_declares_applies_to_the_values_read_after_it() {
    let text = obs_with_code_headers(
        &[
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("G   10   1 L1C", "SYS / SCALE FACTOR"),
        ],
        &[
            "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
            obs_record("G01", &[Some(20_000_000.0), Some(1_234_567.891)]),
            blank_event(4, &[header_line("G  100   1 L1C", "SYS / SCALE FACTOR")]),
            "> 2020 01 01 00 00 30.0000000  0  1".to_string(),
            obs_record("G01", &[Some(20_000_030.0), Some(12_345_678.912)]),
        ]
        .join("\n"),
    );
    let obs = RinexObs::parse(&text).expect("parse");
    // Values stay physical: each read through the factor in effect.
    assert_eq!(
        values_of(&obs, 0, "G01"),
        vec![Some(20_000_000.0), Some(1_234_567.891 / 10.0)]
    );
    assert_eq!(
        values_of(&obs, 2, "G01"),
        vec![Some(20_000_030.0), Some(12_345_678.912 / 100.0)]
    );
    assert_eq!(obs.header().scale_factors.len(), 1);
    // The event's record for L1C replaces the file header's.
    let after = obs.header_at(2).expect("header in effect");
    assert_eq!(
        after
            .scale_factors
            .iter()
            .map(|record| record.factor)
            .collect::<Vec<_>>(),
        vec![100.0]
    );
    // Each value is written through the factor in effect at its epoch.
    let encoded = assert_writes_back(&obs);
    assert!(encoded.contains("  12345678.912"), "{encoded}");

    // Version 2 readers read scaled numbers as physical ones, and an event's
    // scale factor is refused as the file header's is.
    let mut version_two = obs.clone();
    version_two.header.version = 2.11;
    assert!(matches!(
        version_two.to_rinex_string(),
        Err(RinexObsWriteError::ScaleFactorsInVersionTwo { .. })
    ));
}

#[test]
fn phase_shifts_and_glonass_channels_an_event_declares_apply_from_its_epoch() {
    let records = |g01: f64, g02: f64, r02: f64| {
        vec![
            obs_record("G01", &[Some(g01)]),
            obs_record("G02", &[Some(g02)]),
            obs_record("R02", &[Some(r02)]),
        ]
    };
    let mut body = vec!["> 2020 01 01 00 00  0.0000000  0  3".to_string()];
    body.extend(records(100.0, 200.0, 300.0));
    // A shift naming G02 replaces the shift for G02 only; the GLONASS slot's
    // channel changes.
    body.push(blank_event(
        4,
        &[
            header_line("G L2W 0.50000  1 G02", "SYS / PHASE SHIFT"),
            header_line("  1 R02 -4", "GLONASS SLOT / FRQ #"),
        ],
    ));
    body.push("> 2020 01 01 00 00 30.0000000  0  3".to_string());
    body.extend(records(101.0, 201.0, 301.0));
    // A shift for every satellite replaces every earlier one for the code.
    body.push(blank_event(
        4,
        &[header_line("G L2W 0.00000", "SYS / PHASE SHIFT")],
    ));
    body.push("> 2020 01 01 00 01  0.0000000  0  3".to_string());
    body.extend(records(102.0, 202.0, 302.0));
    let text = obs_with_code_headers(
        &[
            header_line("G    1 L2W", "SYS / # / OBS TYPES"),
            header_line("R    1 L1C", "SYS / # / OBS TYPES"),
            header_line("G L2W -0.25000", "SYS / PHASE SHIFT"),
            header_line("  1 R02  1", "GLONASS SLOT / FRQ #"),
        ],
        &body.join("\n"),
    );
    let obs = RinexObs::parse(&text).expect("parse");
    let timeline = obs.header_timeline().expect("timeline");
    let rows_at = |index: usize| -> BTreeMap<String, CarrierPhaseRow> {
        carrier_phase_rows(
            timeline.at(index),
            &obs.epochs()[index],
            &ObservationFilter::all(),
        )
        .expect("carrier phase rows")
        .into_iter()
        .map(|(sat, rows)| (sat.to_string(), rows[0].clone()))
        .collect()
    };
    let (before, after, last) = (rows_at(0), rows_at(2), rows_at(4));
    assert_eq!(before["G01"].phase_shift_cycles, Ok(-0.25));
    assert_eq!(before["G02"].phase_shift_cycles, Ok(-0.25));
    assert_eq!(after["G01"].phase_shift_cycles, Ok(-0.25));
    assert_eq!(after["G02"].phase_shift_cycles, Ok(0.5));
    assert_eq!(last["G01"].phase_shift_cycles, Ok(0.0));
    assert_eq!(last["G02"].phase_shift_cycles, Ok(0.0));
    assert_eq!(
        before["R02"].frequency_hz,
        rinex_observation_frequency_hz(GnssSystem::Glonass, "L1C", 3.05, Some(1))
    );
    assert_eq!(
        after["R02"].frequency_hz,
        rinex_observation_frequency_hz(GnssSystem::Glonass, "L1C", 3.05, Some(-4))
    );
    assert_eq!(obs.header().glonass_slots[&2], 1);
    assert_eq!(timeline.at(4).glonass_slots[&2], -4);
    assert_writes_back(&obs);
}

struct NoAssemblyCorrections;

impl crate::positioning::RinexSppAssemblySource for NoAssemblyCorrections {
    fn rinex_spp_broadcast_corrections(&self) -> crate::positioning::RinexSppBroadcastCorrections {
        crate::positioning::RinexSppBroadcastCorrections::default()
    }
}

#[test]
fn a_new_site_occupation_takes_effect_for_the_epochs_after_it() {
    // RINEX 2.11: flag 3, "new site occupation (end of kinem. data) (at least
    // MARKER NAME record follows)". Flag 2 and flag 5 events with no header
    // records, and a flag 5 event with only a comment, change nothing.
    let position_1 = "  3582105.2910   532589.7313  5232754.8054";
    let position_2 = "  3582205.2910   532489.7313  5232654.8054";
    let text = obs_with_code_headers(
        &[
            header_line("SITE1", "MARKER NAME"),
            header_line(position_1, "APPROX POSITION XYZ"),
            header_line(
                "        0.1000        0.0000        0.0000",
                "ANTENNA: DELTA H/E/N",
            ),
            header_line("G    1 C1C", "SYS / # / OBS TYPES"),
            header_line("R    1 C1C", "SYS / # / OBS TYPES"),
            header_line("  1 R02  1", "GLONASS SLOT / FRQ #"),
        ],
        &[
            "> 2020 01 01 00 00  0.0000000  2  0".to_string(),
            "> 2020 01 01 00 00  0.0000000  0  2".to_string(),
            obs_record("G01", &[Some(20_000_000.0)]),
            obs_record("R02", &[Some(21_000_000.0)]),
            "> 2020 01 01 00 00 15.0000000  5  1".to_string(),
            header_line("external event", "COMMENT"),
            blank_event(
                3,
                &[
                    header_line("SITE2", "MARKER NAME"),
                    header_line("12345", "MARKER NUMBER"),
                    header_line(position_2, "APPROX POSITION XYZ"),
                    header_line(
                        "        0.2500        0.0100        0.0200",
                        "ANTENNA: DELTA H/E/N",
                    ),
                    header_line("  1 R02 -4", "GLONASS SLOT / FRQ #"),
                    header_line("the new site", "COMMENT"),
                ],
            ),
            "> 2020 01 01 00 00 30.0000000  0  2".to_string(),
            obs_record("G01", &[Some(20_000_030.0)]),
            obs_record("R02", &[Some(21_000_030.0)]),
        ]
        .join("\n"),
    );
    let obs = RinexObs::parse(&text).expect("parse");
    let timeline = obs.header_timeline().expect("timeline");
    assert_eq!(
        timeline
            .segments()
            .map(|(first, _)| first)
            .collect::<Vec<_>>(),
        vec![0, 3]
    );
    let before = obs.header_at(2).expect("header in effect");
    assert_eq!(&before, obs.header());
    let after = obs.header_at(4).expect("header in effect");
    assert_eq!(after.marker_name.as_deref(), Some("SITE2"));
    assert_eq!(after.marker_number.as_deref(), Some("12345"));
    assert_eq!(
        after.approx_position_m,
        Some([3_582_205.291, 532_489.731_3, 5_232_654.805_4])
    );
    assert_eq!(after.antenna_delta_hen_m, Some([0.25, 0.01, 0.02]));
    assert_eq!(after.glonass_slots[&2], -4);
    assert_eq!(obs.header().marker_name.as_deref(), Some("SITE1"));

    // Positioning seeds each epoch from the position in effect at it, with the
    // GLONASS channels in effect.
    let options = crate::positioning::RinexSppOptions::new(
        SignalPolicy::default_for(3.05).expect("signal policy"),
    );
    let inputs =
        crate::positioning::spp_inputs_from_rinex_obs(&obs, &NoAssemblyCorrections, &options)
            .expect("assemble");
    assert_eq!(inputs.len(), 2);
    assert_eq!(
        inputs[0].inputs.initial_guess,
        [3_582_105.291, 532_589.731_3, 5_232_754.805_4, 0.0]
    );
    assert_eq!(
        inputs[1].inputs.initial_guess,
        [3_582_205.291, 532_489.731_3, 5_232_654.805_4, 0.0]
    );
    assert_eq!(inputs[0].inputs.glonass_channels[&2], 1);
    assert_eq!(inputs[1].inputs.glonass_channels[&2], -4);
    assert_writes_back(&obs);
}

#[test]
fn a_malformed_header_record_after_an_event_is_refused_as_the_file_header_refuses_it() {
    let bad = header_line("  not a position", "APPROX POSITION XYZ");
    let body = |record: &str| {
        [
            "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
            "G01  20000000.000".to_string(),
            blank_event(3, &[record.to_string()]),
        ]
        .join("\n")
    };
    let error = RinexObs::parse(&minimal_obs(&[], &body(&bad))).expect_err("refused");
    assert!(
        error.to_string().contains("epoch 1 event records"),
        "{error}"
    );
    assert_parse_err(minimal_obs(std::slice::from_ref(&bad), ""));
    // A count its records do not supply is refused at the end of the event.
    let short = header_line("G    2 C1C", "SYS / # / OBS TYPES");
    assert!(RinexObs::parse(&minimal_obs(&[], &body(&short))).is_err());

    // A product whose record was changed after reading reads no header past it.
    let mut obs = RinexObs::parse(&minimal_obs(
        &[],
        &body(&header_line("SITE2", "MARKER NAME")),
    ))
    .expect("parse");
    obs.epochs[1].special_records[0] = bad;
    assert!(obs.header_at(0).is_ok());
    assert!(obs.header_at(1).is_err());
    assert!(obs.header_timeline().is_err());
}

#[test]
fn the_strict_writer_refuses_what_the_lists_in_effect_cannot_hold() {
    let obs = RinexObs::parse(&version_three_type_change_text()).expect("parse");
    let g01: GnssSatelliteId = "G01".parse().expect("G01");

    // S1C is declared only after the first event, so the first epoch has no
    // field for a value under it.
    let mut outside = obs.clone();
    outside.epochs[0].sats.get_mut(&g01).expect("G01")[2].value = Some(45.0);
    assert_eq!(
        outside.to_rinex_string(),
        Err(RinexObsWriteError::ValueOutsideDeclaredList {
            epoch_index: 0,
            satellite: g01,
            code: Some("S1C".to_string()),
        })
    );

    // A code no list declares is not in the union a reader builds.
    let mut widened = obs.clone();
    widened
        .header
        .obs_codes
        .get_mut(&GnssSystem::Gps)
        .expect("GPS")
        .push("C5Q".to_string());
    assert_eq!(
        widened.to_rinex_string(),
        Err(RinexObsWriteError::CodeListsNotUnion {
            system: GnssSystem::Gps
        })
    );

    // PRN / # OF OBS has fields for the file header's codes only.
    let mut counted = obs.clone();
    counted
        .header
        .prn_obs_counts
        .insert(g01, vec![Some(3), Some(2), Some(1)]);
    assert_eq!(
        counted.to_rinex_string(),
        Err(RinexObsWriteError::CountsWithoutCodes {
            satellite: g01,
            codes: 2,
            counts: 3,
        })
    );

    // An event record that no longer reads leaves the lists after it unknown.
    let mut unreadable = obs.clone();
    unreadable.epochs[1].special_records[0] = header_line("G    3 C1C", "SYS / # / OBS TYPES");
    assert!(matches!(
        unreadable.to_rinex_string(),
        Err(RinexObsWriteError::EventRecordsUnreadable { .. })
    ));

    // The downgrade does not lay out the lists an event declares.
    assert_eq!(
        obs.downgrade_to_rinex2(2.11).map(|_| ()),
        Err(RinexObsWriteError::MidFileChangesNotDowngraded { epoch_index: 1 })
    );
}

/// The phase shift a satellite's L1 carrier reads with at an epoch.
fn l1_shift_at(obs: &RinexObs, epoch_index: usize, sat: &str, code: &str) -> f64 {
    let header = obs.header_at(epoch_index).expect("header in effect");
    let sat: GnssSatelliteId = sat.parse().expect("satellite");
    carrier_phase_rows(
        &header,
        &obs.epochs()[epoch_index],
        &ObservationFilter::all(),
    )
    .expect("carrier phase rows")
    .into_iter()
    .find(|(held, _)| *held == sat)
    .and_then(|(_, rows)| rows.into_iter().find(|row| row.code == code))
    .map(|row| {
        row.phase_shift_cycles
            .unwrap_or_else(|error| panic!("{sat} {code} at epoch {epoch_index}: {error:?}"))
    })
    .unwrap_or_else(|| panic!("{sat} {code} at epoch {epoch_index}"))
}

fn g_epoch(minute: u8, code_values: usize) -> String {
    let sats: Vec<String> = (1..=11).map(|prn| format!("G{prn:02}")).collect();
    let mut lines = vec![format!(
        "> 2020 01 01 00 {minute:02}  0.0000000  0 {:2}",
        sats.len()
    )];
    for sat in &sats {
        let values: Vec<Option<f64>> = (0..code_values).map(|_| Some(100.0)).collect();
        lines.push(obs_record(sat, &values));
    }
    lines.join("\n")
}

#[test]
fn a_phase_shift_satellite_list_continues_past_ten_in_the_header_and_after_an_event() {
    // RINEX 3.05 Table A2: SYS / PHASE SHIFT lists `10(1X,A3)` satellites and
    // continues on records of `18X,10(1X,A3)`. A list of eleven was refused as
    // a count mismatch, in the header and after an event alike.
    let satellites: String = (1..=10).map(|prn| format!(" G{prn:02}")).collect();
    let first = header_line(
        &format!("G L1C 0.25000 11{satellites}"),
        "SYS / PHASE SHIFT",
    );
    let continuation = header_line(&format!("{:18} G11", ""), "SYS / PHASE SHIFT");
    let types = header_line("G    1 L1C", "SYS / # / OBS TYPES");

    let in_header = obs_with_code_headers(
        &[types.clone(), first.clone(), continuation.clone()],
        &g_epoch(0, 1),
    );
    let obs = RinexObs::parse(&in_header).expect("a continued list in the header");
    assert_eq!(obs.header().phase_shifts[0].satellites.len(), 11);
    assert_eq!(l1_shift_at(&obs, 0, "G11", "L1C"), 0.25);
    assert_writes_back(&obs);

    let after_event = obs_with_code_headers(
        std::slice::from_ref(&types),
        &[
            g_epoch(0, 1),
            blank_event(4, &[first, continuation]),
            g_epoch(1, 1),
        ]
        .join("\n"),
    );
    let obs = RinexObs::parse(&after_event).expect("a continued list after an event");
    assert_eq!(l1_shift_at(&obs, 0, "G11", "L1C"), 0.0);
    assert_eq!(l1_shift_at(&obs, 2, "G11", "L1C"), 0.25);
    assert_eq!(l1_shift_at(&obs, 2, "G01", "L1C"), 0.25);
    assert_writes_back(&obs);

    // A list whose records end short of its count is still refused.
    let short = obs_with_code_headers(
        &[
            types,
            header_line(
                &format!("G L1C 0.25000 11{satellites}"),
                "SYS / PHASE SHIFT",
            ),
        ],
        "",
    );
    assert_parse_err(short);
}

#[test]
fn phase_shift_records_in_one_block_apply_whatever_their_order() {
    // RINEX 3.05 and 4.02 section 5.2.1: "RINEX allows the free ordering of
    // the header records". Within one block, the file header or one event's
    // records, a record naming satellites applies to them over a record for
    // every satellite of its code, whatever their order. Across blocks, an
    // event's record for every satellite replaces the default and every
    // earlier satellite exception for its code, as section 6.5 has it: "Each
    // value remains valid until changed by an additional header record".
    let types = header_line("G    1 L1W", "SYS / # / OBS TYPES");
    let specific = header_line("G L1W  0.25000  01 G01", "SYS / PHASE SHIFT");
    let general = header_line("G L1W  0.50000", "SYS / PHASE SHIFT");
    for records in [
        vec![specific.clone(), general.clone()],
        vec![general.clone(), specific.clone()],
    ] {
        let mut headers = vec![types.clone()];
        headers.extend(records.iter().cloned());
        let in_header =
            RinexObs::parse(&obs_with_code_headers(&headers, &g_epoch(0, 1))).expect("parse");
        assert_eq!(l1_shift_at(&in_header, 0, "G01", "L1W"), 0.25);
        assert_eq!(l1_shift_at(&in_header, 0, "G02", "L1W"), 0.5);

        let one_event = RinexObs::parse(&obs_with_code_headers(
            std::slice::from_ref(&types),
            &[g_epoch(0, 1), blank_event(4, &records), g_epoch(1, 1)].join("\n"),
        ))
        .expect("parse");
        assert_eq!(l1_shift_at(&one_event, 2, "G01", "L1W"), 0.25);
        assert_eq!(l1_shift_at(&one_event, 2, "G02", "L1W"), 0.5);
        assert_writes_back(&one_event);
    }

    // Across events: a later record for every satellite clears the exception;
    // a later exception replaces only its satellites.
    let across = RinexObs::parse(&obs_with_code_headers(
        &[
            types.clone(),
            specific.clone(),
            header_line("G L1W  0.10000", "SYS / PHASE SHIFT"),
        ],
        &[
            g_epoch(0, 1),
            blank_event(4, std::slice::from_ref(&general)),
            g_epoch(1, 1),
            blank_event(
                4,
                &[header_line("G L1W  0.30000  01 G02", "SYS / PHASE SHIFT")],
            ),
            g_epoch(2, 1),
        ]
        .join("\n"),
    ))
    .expect("parse");
    assert_eq!(l1_shift_at(&across, 0, "G01", "L1W"), 0.25);
    assert_eq!(l1_shift_at(&across, 0, "G02", "L1W"), 0.1);
    assert_eq!(l1_shift_at(&across, 2, "G01", "L1W"), 0.5);
    assert_eq!(l1_shift_at(&across, 2, "G02", "L1W"), 0.5);
    assert_eq!(l1_shift_at(&across, 4, "G01", "L1W"), 0.5);
    assert_eq!(l1_shift_at(&across, 4, "G02", "L1W"), 0.3);
    assert_writes_back(&across);
}

#[test]
fn contradictory_records_in_one_block_are_refused() {
    // With no order in a block, two records giving one key different values
    // say two things at once. Identical duplicates say one. Records that change
    // how observations are read are refused; phase shifts and GLONASS biases,
    // which do not, are kept and read as ambiguous, as tested apart.
    let types = [
        header_line("G    2 C1C L1W", "SYS / # / OBS TYPES"),
        header_line("R    1 C1C", "SYS / # / OBS TYPES"),
    ];
    let contradicted = |records: &[&str], label: &str| {
        let mut headers = types.to_vec();
        headers.extend(records.iter().map(|record| header_line(record, label)));
        let in_header = RinexObs::parse(&obs_with_code_headers(&headers, ""));
        let event = RinexObs::parse(&obs_with_code_headers(
            &types,
            &blank_event(
                4,
                &records
                    .iter()
                    .map(|record| header_line(record, label))
                    .collect::<Vec<_>>(),
            ),
        ));
        (in_header, event)
    };
    for (records, label) in [
        (
            vec!["G   10   1 L1W", "G  100   1 L1W"],
            "SYS / SCALE FACTOR",
        ),
        (vec!["G   10   0", "G  100   0"], "SYS / SCALE FACTOR"),
        (vec!["  2 R01  1 R01  2"], "GLONASS SLOT / FRQ #"),
    ] {
        let (in_header, event) = contradicted(&records, label);
        for (what, result) in [("header", in_header), ("event", event)] {
            let error = result.expect_err("contradictory records are refused");
            assert!(
                error.to_string().contains("contradict"),
                "{label} {records:?} in the {what}: {error}"
            );
        }
    }
    for (records, label) in [
        (
            vec!["G L1W  0.25000", "G L1W  0.25000"],
            "SYS / PHASE SHIFT",
        ),
        (
            vec!["G   10   1 L1W", "G   10   1 L1W"],
            "SYS / SCALE FACTOR",
        ),
        (vec!["  2 R01  1 R01  1"], "GLONASS SLOT / FRQ #"),
    ] {
        let (in_header, event) = contradicted(&records, label);
        assert!(in_header.is_ok(), "{label} {records:?}: {in_header:?}");
        assert!(event.is_ok(), "{label} {records:?}: {event:?}");
    }
}

#[test]
fn scale_factors_in_one_block_apply_whatever_their_order() {
    // A record naming observation types applies to them over a record for
    // every type of its constellation, whatever their order; a later block's
    // record for every type replaces the earlier ones.
    let types = header_line("G    2 C1C L1W", "SYS / # / OBS TYPES");
    let named = header_line("G   10   1 L1W", "SYS / SCALE FACTOR");
    let every = header_line("G  100   0", "SYS / SCALE FACTOR");
    let record = format!("G01{:14.3}  {:14.3}", 2_000_000_000.0, 1_000.0);
    let epoch = |minute: u8| format!("> 2020 01 01 00 {minute:02}  0.0000000  0  1\n{record}");
    for order in [
        vec![named.clone(), every.clone()],
        vec![every.clone(), named.clone()],
    ] {
        let mut headers = vec![types.clone()];
        headers.extend(order.iter().cloned());
        let obs = RinexObs::parse(&obs_with_code_headers(&headers, &epoch(0))).expect("parse");
        assert_eq!(
            values_of(&obs, 0, "G01"),
            vec![Some(20_000_000.0), Some(100.0)]
        );
        assert_writes_back(&obs);
    }
    let across = RinexObs::parse(&obs_with_code_headers(
        &[types, named],
        &[
            epoch(0),
            blank_event(4, std::slice::from_ref(&every)),
            epoch(1),
        ]
        .join("\n"),
    ))
    .expect("parse");
    assert_eq!(
        values_of(&across, 0, "G01"),
        vec![Some(2_000_000_000.0), Some(100.0)]
    );
    assert_eq!(
        values_of(&across, 2, "G01"),
        vec![Some(20_000_000.0), Some(10.0)]
    );
    assert_writes_back(&across);
}

#[test]
fn every_code_a_declaration_names_is_in_the_union_however_declarations_are_grouped() {
    // A repeated SYS / # / OBS TYPES for one constellation within one block
    // adds to its list, as the file header reads it; a later block's
    // declaration replaces the list. The union holds every code any
    // declaration names, however the declarations are grouped into events.
    let types = header_line("G    1 C1C", "SYS / # / OBS TYPES");
    let l2w = header_line("G    1 L2W", "SYS / # / OBS TYPES");
    let l1c = header_line("G    1 L1C", "SYS / # / OBS TYPES");
    let epoch = |values: &[Option<f64>]| {
        [
            "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
            obs_record("G01", values),
        ]
        .join("\n")
    };
    let grouped = RinexObs::parse(&obs_with_code_headers(
        std::slice::from_ref(&types),
        &[
            blank_event(4, &[l2w.clone(), l1c.clone()]),
            epoch(&[Some(1.0), Some(2.0)]),
        ]
        .join("\n"),
    ))
    .expect("parse");
    let split = RinexObs::parse(&obs_with_code_headers(
        &[types],
        &[
            blank_event(4, &[l2w]),
            blank_event(4, &[l1c]),
            epoch(&[Some(2.0)]),
        ]
        .join("\n"),
    ))
    .expect("parse");
    assert_eq!(
        grouped.header().obs_codes[&GnssSystem::Gps],
        codes(&["C1C", "L2W", "L1C"])
    );
    assert_eq!(split.header().obs_codes, grouped.header().obs_codes);
    assert_eq!(
        grouped
            .header_at(1)
            .expect("header in effect")
            .declared_obs_codes[&GnssSystem::Gps],
        codes(&["L2W", "L1C"])
    );
    assert_eq!(
        split
            .header_at(2)
            .expect("header in effect")
            .declared_obs_codes[&GnssSystem::Gps],
        codes(&["L1C"])
    );
    assert_writes_back(&grouped);
    assert_writes_back(&split);
}

#[test]
fn phase_shift_records_are_written_in_their_columns() {
    // RINEX 3.05 and 4.02 Table A2: `A1,1X,A3,1X,F8.5,2X,I2.2,10(1X,A3)`,
    // continued as `18X,10(1X,A3)`; the correction is blank if none, and the
    // count 0 or blank for every satellite of the system.
    let satellites: String = (1..=10).map(|prn| format!(" G{prn:02}")).collect();
    let expected = [
        format!("G L1C  0.25000  11{satellites}"),
        format!("{:18} G11", ""),
        "G L2S -0.25000".to_string(),
        "G L1W".to_string(),
        "G L2W           02 G01 G02".to_string(),
    ];
    for version in [2.11, 3.05, 4.02] {
        let types = if version < 3.0 {
            header_line("     1    C1", "# / TYPES OF OBSERV")
        } else {
            header_line("G    1 C1C", "SYS / # / OBS TYPES")
        };
        let mut headers = vec![types];
        headers.extend(
            expected
                .iter()
                .map(|content| header_line(content, "SYS / PHASE SHIFT")),
        );
        let obs = RinexObs::parse(&obs_with_version_and_code_headers(version, &headers, ""))
            .unwrap_or_else(|error| panic!("{version}: {error}"));
        let written = obs.to_rinex_string().expect("serialize");
        let lines: Vec<&str> = written
            .lines()
            .filter(|line| line.ends_with("SYS / PHASE SHIFT"))
            .collect();
        let wanted: Vec<String> = expected
            .iter()
            .map(|content| header_line(content, "SYS / PHASE SHIFT"))
            .collect();
        assert_eq!(lines, wanted, "{version}");
        assert_eq!(RinexObs::parse(&written).expect("reads back"), obs);
    }
    // A correction its `F8.5` field cannot hold is refused.
    assert_parse_err(minimal_obs_with_phase_shift("G L1C 1e-300 01 G01"));
}

#[test]
fn a_rinex_4_phase_shift_record_is_kept_and_not_applied() {
    // RINEX 4.00, 4.01 and 4.02 Table A2: SYS / PHASE SHIFT "is strongly
    // deprecated. It is allowed in this version for compatibility with previous
    // RINEX versions but the lines should be ignored by RINEX decoders and
    // encoders." The records are kept and written back, and no correction is
    // reported; contradictory records are not refused either.
    let types = header_line("G    1 L1W", "SYS / # / OBS TYPES");
    for (version, expected) in [(3.05, 0.25), (4.00, 0.0), (4.02, 0.0)] {
        let obs = RinexObs::parse(&obs_with_version_and_code_headers(
            version,
            &[
                types.clone(),
                header_line("G L1W  0.25000", "SYS / PHASE SHIFT"),
            ],
            &g_epoch(0, 1),
        ))
        .expect("parse");
        assert_eq!(l1_shift_at(&obs, 0, "G01", "L1W"), expected, "{version}");
        let written = assert_writes_back(&obs);
        assert!(written.contains("G L1W  0.25000"), "{version}");
    }
    let contradictory = RinexObs::parse(&obs_with_version_and_code_headers(
        4.02,
        &[
            types,
            header_line("G L1W  0.25000", "SYS / PHASE SHIFT"),
            header_line("G L1W  0.50000", "SYS / PHASE SHIFT"),
        ],
        &g_epoch(0, 1),
    ))
    .expect("ignored records are not refused");
    assert_writes_back(&contradictory);
}

#[test]
fn the_strict_writer_refuses_declared_lists_a_reader_would_not_rebuild() {
    // A reader rebuilds `declared_obs_codes` from the header's type records, so
    // a product holding other lists there would read back differently.
    let at = ObsEpochTime {
        year: 2015,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let version_two = [
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line("     1    C1", "# / TYPES OF OBSERV"),
        header_line("", "END OF HEADER"),
        v2_epoch_line(at, 0, 1, "G01"),
        "  20000000.000".to_string(),
    ]
    .join("\n");
    let mut cleared = RinexObs::parse(&version_two).expect("parse");
    assert_eq!(
        cleared.header().declared_obs_codes[&GnssSystem::Gps],
        codes(&["C1C"])
    );
    cleared.header.declared_obs_codes.clear();
    assert_eq!(
        cleared.to_rinex_string(),
        Err(RinexObsWriteError::DeclaredListNotStated {
            system: GnssSystem::Gps
        })
    );
    // A list other than the one the names read as is refused too.
    let mut renamed = RinexObs::parse(&version_two).expect("parse");
    renamed
        .header
        .declared_obs_codes
        .insert(GnssSystem::Gps, codes(&["C1W"]));
    assert_eq!(
        renamed.to_rinex_string(),
        Err(RinexObsWriteError::DeclaredListNotStated {
            system: GnssSystem::Gps
        })
    );

    let mut cleared = RinexObs::parse(&minimal_obs(
        &[],
        "> 2020 01 01 00 00  0.0000000  0  1\nG01  20000000.000",
    ))
    .expect("parse");
    cleared.header.declared_obs_codes.clear();
    assert!(
        cleared.to_rinex_string().is_err(),
        "cleared declared lists at version 3 were written"
    );
}

/// Whether an event record continues the declaration of the record before it:
/// a record carrying the same label with the blank lead columns that label's
/// continuation records open with.
fn continues_declaration(label: &str, previous_label: &str, record: &str) -> bool {
    if label != previous_label {
        return false;
    }
    let blank = |end: usize| record.get(..end).is_none_or(|head| head.trim().is_empty());
    match label {
        "SYS / # / OBS TYPES" | "SYS / SCALE FACTOR" => blank(1),
        "# / TYPES OF OBSERV" => blank(6),
        "SYS / PHASE SHIFT" => blank(18),
        "GLONASS SLOT / FRQ #" => blank(3),
        _ => false,
    }
}

/// An event's records as declarations: each record with its continuations.
fn record_declarations(records: &[String]) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut previous = String::new();
    for record in records {
        let label = event_record_label(record);
        match groups.last_mut() {
            Some(group) if continues_declaration(&label, &previous, record) => {
                group.push(record.clone());
            }
            _ => groups.push(vec![record.clone()]),
        }
        previous = label;
    }
    groups
}

/// One epoch's values and slips by satellite and code, the `n`th copy of a
/// code counted apart, blank fields left out.
fn epoch_values_by_code(obs: &RinexObs, index: usize) -> BTreeMap<(String, String), ObsValue> {
    let mut held = BTreeMap::new();
    let epoch = &obs.epochs()[index];
    for (kind, records) in [("obs", &epoch.sats), ("slip", &epoch.cycle_slips)] {
        for (sat, values) in records {
            let codes = &obs.header().obs_codes[&sat.system];
            let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
            for (code, value) in codes.iter().zip(values) {
                let copy = seen.entry(code.as_str()).or_default();
                *copy += 1;
                if value.value.is_some() || value.lli.is_some() || value.ssi.is_some() {
                    held.insert((format!("{kind} {sat}"), format!("{code}#{copy}")), *value);
                }
            }
        }
    }
    held
}

/// A key a declaration gives a value: its label, what the value is for (the
/// constellation of a type list or scale factor, the constellation and code of
/// a phase shift), and the satellite, code, slot or bias code it names, `None`
/// where it gives every one of them.
type DeclarationKey = (String, String, Option<String>);

/// Labels whose declarations give values to satellites, codes, slots or bias
/// codes, each declaration to some of them.
const KEYED_LABELS: [&str; 4] = [
    "SYS / PHASE SHIFT",
    "SYS / SCALE FACTOR",
    "GLONASS SLOT / FRQ #",
    "GLONASS COD/PHS/BIS",
];

/// The keys a declaration gives values, read as an event's records.
fn declaration_keys(obs: &RinexObs, declaration: &[String]) -> Vec<DeclarationKey> {
    let label = event_record_label(&declaration[0]);
    let key = |scope: String, member: Option<String>| (label.clone(), scope, member);
    match label.as_str() {
        "SYS / # / OBS TYPES" => vec![key(declaration[0].chars().take(1).collect(), None)],
        keyed if KEYED_LABELS.contains(&keyed) => {
            let mut header = obs.header().clone();
            header.phase_shifts.clear();
            header.scale_factors.clear();
            header.glonass_slots.clear();
            header.glonass_cod_phs_bis = None;
            apply_event_records(&mut header, declaration).expect("a declaration reads alone");
            let mut keys = Vec::new();
            // A phase shift's member is `<code>|<satellite>`, `*` for every
            // code or every satellite.
            for shift in &header.phase_shifts {
                let scope = shift.system.to_string();
                let Some(code) = &shift.code else {
                    keys.push(key(scope, None));
                    continue;
                };
                if shift.covers_every_satellite() {
                    keys.push(key(scope.clone(), Some(format!("{code}|*"))));
                }
                for satellite in shift
                    .satellites
                    .iter()
                    .map(ToString::to_string)
                    .chain(shift.unrepresentable_satellites.iter().cloned())
                {
                    keys.push(key(scope.clone(), Some(format!("{code}|{satellite}"))));
                }
            }
            for factor in &header.scale_factors {
                let scope = factor.system.to_string();
                if factor.codes.is_empty() {
                    keys.push(key(scope.clone(), None));
                }
                for code in &factor.codes {
                    keys.push(key(scope.clone(), Some(code.clone())));
                }
            }
            for prn in header.glonass_slots.keys() {
                keys.push(key(String::new(), Some(prn.to_string())));
            }
            match &header.glonass_cod_phs_bis {
                Some(entries) if entries.is_empty() => keys.push(key(String::new(), None)),
                Some(entries) => {
                    for (code, _) in entries {
                        keys.push(key(String::new(), Some(code.clone())));
                    }
                }
                None => {}
            }
            keys
        }
        "# / TYPES OF OBSERV" => vec![key(String::new(), None)],
        single if SINGLE_VALUE_LABELS.contains(&single) => vec![key(String::new(), None)],
        // A comment, or a record whose value nothing applies, gives no key.
        _ => Vec::new(),
    }
}

/// Whether two declarations give a value to one key. A member `None` covers
/// every member, and a member's `|`-separated parts match part by part, `*`
/// matching any.
fn keys_overlap(one: &[DeclarationKey], other: &[DeclarationKey]) -> bool {
    let members_overlap = |one: &Option<String>, other: &Option<String>| match (one, other) {
        (Some(one), Some(other)) => one
            .split('|')
            .zip(other.split('|'))
            .all(|(one, other)| one == other || one == "*" || other == "*"),
        _ => true,
    };
    one.iter().any(|(label, scope, member)| {
        other
            .iter()
            .any(|(other_label, other_scope, other_member)| {
                label == other_label
                    && scope == other_scope
                    && members_overlap(member, other_member)
            })
    })
}

/// Every event holding more than one declaration, no two of which give a value
/// to one key, gives, at every epoch from it on, the header its declarations
/// give split across consecutive events, and the file written from each reads
/// the same union, headers and values. Declarations sharing a key are not
/// split: within one event they have no order, and across events the later one
/// replaces the earlier. Returns how many events were split.
fn assert_split_events_agree(obs: &RinexObs, what: &str) -> usize {
    let timeline = obs.header_timeline().expect(what);
    let text = obs.to_rinex_string().ok();
    let mut split_events = 0;
    for (index, epoch) in obs.epochs().iter().enumerate() {
        if !applies_header_records(epoch.flag) {
            continue;
        }
        let groups = record_declarations(&epoch.special_records);
        if groups.len() < 2 {
            continue;
        }
        let keys: Vec<Vec<DeclarationKey>> = groups
            .iter()
            .map(|group| declaration_keys(obs, group))
            .collect();
        let shared = keys.iter().enumerate().any(|(position, one)| {
            keys[position + 1..]
                .iter()
                .any(|other| keys_overlap(one, other))
        });
        if shared {
            continue;
        }
        split_events += 1;
        let extra = groups.len() - 1;
        let copies: Vec<ObsEpoch> = groups
            .into_iter()
            .map(|group| ObsEpoch {
                declared_record_count: group.len(),
                special_records: group,
                ..epoch.clone()
            })
            .collect();
        let mut split = obs.clone();
        split.epochs.splice(index..=index, copies);
        let split_timeline = split
            .header_timeline()
            .unwrap_or_else(|error| panic!("{what}: split event {index}: {error}"));
        for later in index..obs.epochs().len() {
            assert_eq!(
                split_timeline.at(later + extra),
                timeline.at(later),
                "{what}: the header at epoch {later} after splitting event {index}"
            );
        }
        if let Some(text) = &text {
            let split_text = split
                .to_rinex_string()
                .unwrap_or_else(|error| panic!("{what}: write split event {index}: {error}"));
            let read = RinexObs::parse(&split_text).expect("read the split file");
            let original = RinexObs::parse(text).expect("read the file");
            assert_eq!(
                read.header().obs_codes,
                original.header().obs_codes,
                "{what}: the union after splitting event {index}"
            );
            let read_timeline = read.header_timeline().expect("split file timeline");
            let original_timeline = original.header_timeline().expect("file timeline");
            for later in index..obs.epochs().len() {
                assert_eq!(
                    read_timeline.at(later + extra),
                    original_timeline.at(later),
                    "{what}: the header read at epoch {later} after splitting event {index}"
                );
                assert_eq!(
                    epoch_values_by_code(&read, later + extra),
                    epoch_values_by_code(&original, later),
                    "{what}: values at epoch {later} after splitting event {index}"
                );
            }
        }
    }
    split_events
}

/// An event with header records placed before the first epoch gives the header
/// those records give moved into the file header, and the same values. A file
/// header declaration the event's records replace, a type list for the same
/// constellation or a record of a label holding one value, is taken out of the
/// header. Where a file header's phase shift, scale factor, GLONASS slot or
/// GLONASS bias shares a key with one the event declares, the move is not
/// checked: in the event it is replaced, and in one header block the two would
/// apply by their keys or contradict each other. Returns whether it was checked.
fn assert_leading_event_moves_into_header(obs: &RinexObs, what: &str) -> bool {
    let Some(first) = obs.epochs().first() else {
        return false;
    };
    if !applies_header_records(first.flag) {
        return false;
    }
    let version = obs.header().version;
    let text = obs.to_rinex_string().expect(what);
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    let end = lines
        .iter()
        .position(|line| line.ends_with("END OF HEADER"))
        .expect("END OF HEADER");
    let moved_records: Vec<String> = first
        .special_records
        .iter()
        .filter(|record| event_record_takes_effect(version, record))
        .cloned()
        .collect();
    let event_keys: Vec<DeclarationKey> = record_declarations(&moved_records)
        .iter()
        .flat_map(|declaration| declaration_keys(obs, declaration))
        .collect();
    let mut moved_lines: Vec<String> = Vec::new();
    for declaration in record_declarations(&lines[..end]) {
        let keys = declaration_keys(obs, &declaration);
        if !keys_overlap(&keys, &event_keys) {
            moved_lines.extend(declaration);
        } else if KEYED_LABELS.contains(&event_record_label(&declaration[0]).as_str()) {
            return false;
        }
    }
    moved_lines.extend(moved_records);
    moved_lines.push(lines[end].clone());
    let body = end + 2 + first.special_records.len();
    moved_lines.extend(lines[body..].iter().cloned());
    let moved = RinexObs::parse(&moved_lines.join("\n"))
        .unwrap_or_else(|error| panic!("{what}: the records moved into the header: {error}"));
    let at = obs.header_at(0).expect("header in effect");
    let file = moved.header();
    macro_rules! same {
        ($($field:ident),+ $(,)?) => {
            $(
                assert_eq!(
                    file.$field, at.$field,
                    "{what}: {} with the leading event's records in the header",
                    stringify!($field)
                );
            )+
        };
    }
    same!(
        approx_position_m,
        antenna_delta_hen_m,
        declared_obs_codes,
        rinex2_types,
        marker_name,
        marker_number,
        marker_type,
        observer,
        agency,
        receiver,
        antenna,
        interval_s,
        phase_shifts,
        scale_factors,
        glonass_slots,
        glonass_cod_phs_bis,
        signal_strength_unit,
    );
    assert_eq!(moved.epochs().len() + 1, obs.epochs().len(), "{what}");
    for index in 1..obs.epochs().len() {
        assert_eq!(
            epoch_values_by_code(&moved, index - 1),
            epoch_values_by_code(obs, index),
            "{what}: values at epoch {index} with the leading event in the header"
        );
    }
    true
}

/// A file whose events declare header records, and whether the split and move
/// properties are claimed for it.
struct EventCase {
    what: String,
    text: String,
    splits: bool,
    moves: bool,
}

/// Files whose events declare header records, with every event also moved to
/// the start of the body.
fn header_record_event_cases() -> Vec<EventCase> {
    let l1w = header_line("G    1 L1W", "SYS / # / OBS TYPES");
    let l1c = header_line("G    1 L1C", "SYS / # / OBS TYPES");
    // A phase shift naming G01 and one for every satellite share a key.
    let shift_records = vec![
        header_line("G L1W 0.25000  1 G01", "SYS / PHASE SHIFT"),
        header_line("G L1W 0.50000  0", "SYS / PHASE SHIFT"),
    ];
    let apart_shift_records = vec![
        header_line("G L1W  0.25000  01 G01", "SYS / PHASE SHIFT"),
        header_line("G L1W  0.50000  01 G02", "SYS / PHASE SHIFT"),
    ];
    let satellites: String = (1..=10).map(|prn| format!(" G{prn:02}")).collect();
    let continued_records = vec![
        header_line(
            &format!("G L1C 0.25000 11{satellites}"),
            "SYS / PHASE SHIFT",
        ),
        header_line(&format!("{:18} G11", ""), "SYS / PHASE SHIFT"),
        header_line("    30.000", "INTERVAL"),
        header_line("a comment", "COMMENT"),
    ];
    let lists_records = vec![
        header_line("G    3 L1C C1C S1C", "SYS / # / OBS TYPES"),
        header_line("R    2 C1C L1C", "SYS / # / OBS TYPES"),
        header_line("G   10   1 L1C", "SYS / SCALE FACTOR"),
    ];
    // Two declarations of one constellation's types share a key.
    let repeated_records = vec![
        header_line("G    1 L1C", "SYS / # / OBS TYPES"),
        header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
        header_line("G  100   1 C1C", "SYS / SCALE FACTOR"),
    ];
    let slots: String = (1..=8)
        .map(|prn| format!("R{prn:02} {:2} ", prn as i8 - 7))
        .collect();
    let glonass_records = vec![
        header_line(&format!("  9 {}", slots.trim_end()), "GLONASS SLOT / FRQ #"),
        header_line("    R09  2", "GLONASS SLOT / FRQ #"),
        header_line("C1C  -10.000 C1P  -10.123", "GLONASS COD/PHS/BIS"),
        header_line("C2C  -11.000", "GLONASS COD/PHS/BIS"),
        header_line("SITE2", "MARKER NAME"),
        header_line(
            "  3582105.2910   532589.7313  5232754.8054",
            "APPROX POSITION XYZ",
        ),
    ];
    let lists_epoch = |minute: u8, after: bool| {
        let g = if after {
            obs_record("G01", &[Some(100.0), Some(20_000_000.0), Some(45.0)])
        } else {
            obs_record("G01", &[Some(20_000_000.0), Some(100.0)])
        };
        let r = if after {
            obs_record("R02", &[Some(21_000_000.0), Some(200.0)])
        } else {
            obs_record("R02", &[Some(21_000_000.0)])
        };
        format!("> 2020 01 01 00 {minute:02}  0.0000000  0  2\n{g}\n{r}")
    };
    let repeated_epoch = |minute: u8| {
        format!(
            "> 2020 01 01 00 {minute:02}  0.0000000  0  1\n{}",
            obs_record("G01", &[Some(20_000_000.0), Some(100.0)])
        )
    };
    let glonass_epoch = |minute: u8| {
        format!(
            "> 2020 01 01 00 {minute:02}  0.0000000  0  2\n{}\n{}",
            obs_record("R01", &[Some(100.0)]),
            obs_record("R09", &[Some(200.0)])
        )
    };
    let v2_at = |minute: u8| ObsEpochTime {
        year: 2015,
        month: 1,
        day: 1,
        hour: 0,
        minute,
        second: 0.0,
    };
    let v2_header = |names: &str| {
        [
            header_line(
                "     2.11           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            header_line(names, "# / TYPES OF OBSERV"),
            header_line("", "END OF HEADER"),
        ]
        .join("\n")
    };
    let v2_records = vec![
        header_line("     3    L1    C1    S1", "# / TYPES OF OBSERV"),
        header_line("    30.000", "INTERVAL"),
        header_line("a comment", "COMMENT"),
    ];
    let v2_blank_event = |records: &[String]| {
        let mut lines = vec![format!("{:28}4{:3}", "", records.len())];
        lines.extend(records.iter().cloned());
        lines.join("\n")
    };

    let mut cases = Vec::new();
    for leading in [false, true] {
        let v3 = |types: &[String], before: String, records: &[String], after: String| {
            let body = if leading {
                [blank_event(4, records), after].join("\n")
            } else {
                [before, blank_event(4, records), after].join("\n")
            };
            obs_with_code_headers(types, &body)
        };
        let mut case = |what: &str, text: String, splits: bool, moves: bool| {
            cases.push(EventCase {
                what: if leading {
                    format!("leading {what}")
                } else {
                    what.to_string()
                },
                text,
                splits,
                moves: leading && moves,
            });
        };
        case(
            "phase shifts",
            v3(
                std::slice::from_ref(&l1w),
                g_epoch(0, 1),
                &shift_records,
                g_epoch(1, 1),
            ),
            false,
            true,
        );
        case(
            "phase shifts apart",
            v3(
                std::slice::from_ref(&l1w),
                g_epoch(0, 1),
                &apart_shift_records,
                g_epoch(1, 1),
            ),
            true,
            true,
        );
        case(
            "continued list",
            v3(
                std::slice::from_ref(&l1c),
                g_epoch(0, 1),
                &continued_records,
                g_epoch(1, 1),
            ),
            true,
            true,
        );
        case(
            "lists",
            v3(
                &[
                    header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
                    header_line("R    1 C1C", "SYS / # / OBS TYPES"),
                ],
                lists_epoch(0, false),
                &lists_records,
                lists_epoch(1, true),
            ),
            true,
            true,
        );
        case(
            "repeated list",
            v3(
                &[header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
                repeated_epoch(0),
                &repeated_records,
                repeated_epoch(1),
            ),
            false,
            true,
        );
        case(
            "glonass",
            v3(
                &[
                    header_line("R    1 L1C", "SYS / # / OBS TYPES"),
                    header_line("  1 R10  3", "GLONASS SLOT / FRQ #"),
                    header_line("C2P   -9.000", "GLONASS COD/PHS/BIS"),
                    header_line("SITE1", "MARKER NAME"),
                ],
                glonass_epoch(0),
                &glonass_records,
                glonass_epoch(1),
            ),
            true,
            true,
        );
        // The event gives R01, R09 and C1C values the file header gives.
        case(
            "glonass replaced",
            v3(
                &[
                    header_line("R    1 L1C", "SYS / # / OBS TYPES"),
                    header_line("  2 R01  1 R09 -4", "GLONASS SLOT / FRQ #"),
                    header_line("C1C   -9.000", "GLONASS COD/PHS/BIS"),
                ],
                glonass_epoch(0),
                &glonass_records,
                glonass_epoch(1),
            ),
            true,
            false,
        );
        let before = [
            v2_epoch_line(v2_at(0), 0, 1, "G01"),
            obs_record("", &[Some(20_000_000.0), Some(100.0)]),
        ]
        .join("\n");
        let after = [
            v2_epoch_line(v2_at(1), 0, 1, "G01"),
            obs_record("", &[Some(100.0), Some(20_000_000.0), Some(45.0)]),
        ]
        .join("\n");
        let body = if leading {
            [v2_blank_event(&v2_records), after].join("\n")
        } else {
            [before, v2_blank_event(&v2_records), after].join("\n")
        };
        case(
            "version 2 names",
            [v2_header("     2    C1    L1"), body].join("\n"),
            true,
            true,
        );
    }
    cases
}

#[test]
fn header_records_give_one_header_however_they_are_grouped_or_placed() {
    // The header in effect cannot depend on whether records giving values to
    // different keys share an event or sit in consecutive ones, or on whether
    // the records of an event before the first epoch are in it or in the file
    // header. Each case says whether its records are split and moved.
    for case in header_record_event_cases() {
        let what = case.what.as_str();
        let obs = RinexObs::parse(&case.text).unwrap_or_else(|error| panic!("{what}: {error}"));
        assert_writes_back(&obs);
        assert_eq!(
            assert_split_events_agree(&obs, what) > 0,
            case.splits,
            "{what}: split"
        );
        assert_eq!(
            assert_leading_event_moves_into_header(&obs, what),
            case.moves,
            "{what}: moved"
        );
    }
}

#[test]
fn single_value_records_in_one_block_that_differ_are_refused() {
    // A header block gives its records no order, so two records of a label
    // holding one value that read as different values say two things at once.
    // Identical records, and records whose text differs but which read as one
    // value, say one. A value in the file header and another after an event
    // are a value and the one replacing it.
    let position = |x: f64, decimals: usize| {
        format!(
            "{x:14.decimals$}{:14.4}{:14.4}",
            532_589.731_3, 5_232_754.805_4
        )
    };
    let delta = |h: f64, decimals: usize| format!("{h:14.decimals$}{:14.4}{:14.4}", 0.0, 0.0);
    let pair = |one: &str, two: &str| format!("{one:<20}{two:<20}");
    let receiver =
        |number: &str, version: &str| format!("{number:<20}{:<20}{version:<20}", "SEPT POLARX5");
    // A label, a record, a record reading as another value, and a record whose
    // text differs from the first's and reads as its value.
    let layered: Vec<(&str, String, String, String)> = vec![
        (
            "INTERVAL",
            "    30.000".into(),
            "    15.000".into(),
            "      30.0".into(),
        ),
        (
            "MARKER NAME",
            "SITE1".into(),
            "SITE2".into(),
            "   SITE1".into(),
        ),
        (
            "MARKER NUMBER",
            "10001M001".into(),
            "10002M001".into(),
            "  10001M001".into(),
        ),
        (
            "MARKER TYPE",
            "GEODETIC".into(),
            "NON_GEODETIC".into(),
            " GEODETIC".into(),
        ),
        (
            "APPROX POSITION XYZ",
            position(3_582_105.291, 4),
            position(3_582_105.292, 4),
            position(3_582_105.291, 3),
        ),
        (
            "ANTENNA: DELTA H/E/N",
            delta(0.1, 4),
            delta(0.2, 4),
            delta(0.1, 3),
        ),
        (
            "ANT # / TYPE",
            pair("1234", "TRM59800.00"),
            pair("5678", "TRM59800.00"),
            pair(" 1234", "TRM59800.00"),
        ),
        (
            "REC # / TYPE / VERS",
            receiver("5001", "5.3.2"),
            receiver("5001", "5.4.0"),
            receiver(" 5001", "5.3.2"),
        ),
        (
            "OBSERVER / AGENCY",
            pair("OBSERVER", "AGENCY"),
            pair("OBSERVER", "OTHER AGENCY"),
            pair("  OBSERVER", "AGENCY"),
        ),
        (
            "SIGNAL STRENGTH UNIT",
            "DBHZ".into(),
            "DB".into(),
            "  DBHZ".into(),
        ),
    ];
    let time = |minute: u8| {
        format!(
            "{:6}{:6}{:6}{:6}{minute:6}{:13.7}{:5}GPS",
            2020, 1, 1, 0, 0.0, ""
        )
    };
    // Labels only the file header applies.
    let header_only: Vec<(&str, String, String)> = vec![
        ("TIME OF FIRST OBS", time(0), time(1)),
        ("TIME OF LAST OBS", time(30), time(31)),
        ("LEAP SECONDS", "    18".into(), "    17".into()),
        ("# OF SATELLITES", "     5".into(), "     6".into()),
    ];
    let mut labels: Vec<&str> = layered.iter().map(|(label, ..)| *label).collect();
    labels.push("RINEX VERSION / TYPE");
    labels.extend(header_only.iter().map(|(label, ..)| *label));
    assert_eq!(
        labels,
        SINGLE_VALUE_LABELS[..]
            .iter()
            .copied()
            .filter(|label| labels.contains(label))
            .collect::<Vec<_>>()
    );
    assert_eq!(labels.len(), SINGLE_VALUE_LABELS.len());

    for version in [2.11, 3.05] {
        let types = if version < 3.0 {
            header_line("     1    C1", "# / TYPES OF OBSERV")
        } else {
            header_line("G    1 C1C", "SYS / # / OBS TYPES")
        };
        let event = |records: &[String]| {
            if version < 3.0 {
                let mut lines = vec![format!("{:28}4{:3}", "", records.len())];
                lines.extend(records.iter().cloned());
                lines.join("\n")
            } else {
                blank_event(4, records)
            }
        };
        let read = |headers: &[String], body: &str| {
            let mut all = vec![types.clone()];
            all.extend(headers.iter().cloned());
            RinexObs::parse(&obs_with_version_and_code_headers(version, &all, body))
                .map_err(|error| error.to_string())
        };
        let refused = |result: std::result::Result<RinexObs, String>, label: &str, what: &str| {
            let message = result.expect_err(what);
            assert!(
                message.contains(&format!("{label} records in one header block contradict")),
                "{what}: {message}"
            );
        };
        for (label, one, other, same) in &layered {
            let record = |content: &str| header_line(content, label);
            let what = format!("{version} {label}");
            refused(
                read(&[record(one), record(other)], ""),
                label,
                &format!("{what} in the header"),
            );
            refused(
                read(&[], &event(&[record(one), record(other)])),
                label,
                &format!("{what} in an event"),
            );
            for records in [[record(one), record(one)], [record(one), record(same)]] {
                read(&records, "").unwrap_or_else(|error| panic!("{what} in the header: {error}"));
                let obs = read(&[], &event(&records))
                    .unwrap_or_else(|error| panic!("{what} in an event: {error}"));
                assert_writes_back(&obs);
            }
            let obs = read(&[record(one)], &event(&[record(other)]))
                .unwrap_or_else(|error| panic!("{what} in the header and an event: {error}"));
            assert_writes_back(&obs);
        }
        for (label, one, other) in &header_only {
            let record = |content: &str| header_line(content, label);
            let what = format!("{version} {label}");
            refused(
                read(&[record(one), record(other)], ""),
                label,
                &format!("{what} in the header"),
            );
            read(&[record(one), record(one)], "")
                .unwrap_or_else(|error| panic!("{what} twice in the header: {error}"));
            // An event's record of the label takes no effect.
            read(&[], &event(&[record(one), record(other)]))
                .unwrap_or_else(|error| panic!("{what} in an event: {error}"));
        }
        let version_record = |at: f64| {
            header_line(
                &format!("{at:9.2}           OBSERVATION DATA    M (MIXED)"),
                "RINEX VERSION / TYPE",
            )
        };
        refused(
            read(&[version_record(version - 0.01)], ""),
            "RINEX VERSION / TYPE",
            &format!("{version} a second version"),
        );
        read(&[version_record(version)], "")
            .unwrap_or_else(|error| panic!("{version} the version twice: {error}"));
        // Records the product keeps as a list, or applies nothing of, are read
        // as before.
        let program = |name: &str| {
            header_line(
                &format!("{name:<20}{:<20}{:<20}", "RUN BY", "20200101 000000 UTC"),
                "PGM / RUN BY / DATE",
            )
        };
        read(
            &[
                program("ONE"),
                program("TWO"),
                header_line("one", "COMMENT"),
                header_line("two", "COMMENT"),
            ],
            "",
        )
        .unwrap_or_else(|error| panic!("{version} programs and comments: {error}"));
    }
}

#[test]
fn signed_zeros_in_one_block_are_one_value() {
    // -0.0 and 0.0 read as one number, so two records giving them do not
    // contradict each other.
    let triple = |x: f64| format!("{x:14.4}{:14.4}{:14.4}", 1.0, 2.0);
    let cases: Vec<(&str, String, String)> = vec![
        ("INTERVAL", "     0.000".into(), "    -0.000".into()),
        ("APPROX POSITION XYZ", triple(0.0), triple(-0.0)),
        ("ANTENNA: DELTA H/E/N", triple(0.0), triple(-0.0)),
        (
            "SYS / PHASE SHIFT",
            "G L1C  0.00000".into(),
            "G L1C -0.00000".into(),
        ),
        (
            "GLONASS COD/PHS/BIS",
            " C1C    0.000".into(),
            " C1C   -0.000".into(),
        ),
    ];
    let types = header_line("G    1 L1C", "SYS / # / OBS TYPES");
    for (label, zero, negative) in &cases {
        let records = [header_line(zero, label), header_line(negative, label)];
        let mut headers = vec![types.clone()];
        headers.extend(records.iter().cloned());
        RinexObs::parse(&obs_with_code_headers(&headers, ""))
            .unwrap_or_else(|error| panic!("{label} in the header: {error}"));
        let obs = RinexObs::parse(&obs_with_code_headers(
            std::slice::from_ref(&types),
            &blank_event(4, &records),
        ))
        .unwrap_or_else(|error| panic!("{label} in an event: {error}"));
        assert_writes_back(&obs);
    }
}

#[test]
fn a_loosely_spaced_phase_shift_record_is_read_by_the_reading_its_fields_agree_with() {
    let read = |content: &str| {
        RinexObs::parse(&minimal_obs_with_phase_shift(content))
            .map(|obs| obs.header().phase_shifts[0].clone())
            .map_err(|error| error.to_string())
    };
    let satellites = |shift: &ObsPhaseShift| {
        shift
            .satellites
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    };
    // Only a blank correction agrees: the number is the satellite count.
    for (content, count) in [
        ("G L1C            01 G01", 1),
        ("G  L1C           01 G01", 1),
        ("G  L1C  02 G01 G02", 2),
    ] {
        let shift = read(content).unwrap_or_else(|error| panic!("{content}: {error}"));
        assert_eq!(shift.correction_cycles, None, "{content}");
        assert_eq!(satellites(&shift).len(), count, "{content}");
    }
    // Only a correction agrees.
    for (content, correction, count) in [
        ("G  L1C  0.250  02 G01 G02", 0.25, 2),
        ("G  L1C  -0.5", -0.5, 0),
        ("G  L1C  0.0", 0.0, 0),
        ("G  L1C  1 0", 1.0, 0),
    ] {
        let shift = read(content).unwrap_or_else(|error| panic!("{content}: {error}"));
        assert_eq!(shift.correction_cycles, Some(correction), "{content}");
        assert_eq!(satellites(&shift).len(), count, "{content}");
    }
    // Both agree, a correction of 0 or a blank correction with no satellites,
    // and no decimal point tells them apart.
    for content in ["G  L1C  0", "G  L1C  00"] {
        let error = read(content).expect_err(content);
        assert!(error.contains("ambiguous"), "{content}: {error}");
    }
    // Neither agrees.
    for content in ["G  L1C  0.25  2 G01", "G  L1C  bad", "G  L1C  01 G01 G02"] {
        read(content).expect_err(content);
    }
}

/// The satellites a phase shift names by a designator the satellite id does
/// not hold, as written.
fn unrepresentable_tokens(shift: &ObsPhaseShift) -> Vec<String> {
    shift.unrepresentable_satellites.clone()
}

#[test]
fn real_igs_headers_naming_r28_in_phase_shift_lists_are_read_and_written_back() {
    // The BADG00RUS and ARHT00ATA headers of IGS 2026 day 001, expanded from
    // the CRINEX RNX2CRX 4.1.0 wrote, with PRN / # OF OBS kept for R01 and R28
    // only. Each names R28, which `GnssSatelliteId` does not hold, in its
    // GLONASS slot table, in four SYS / PHASE SHIFT lists and in a
    // PRN / # OF OBS record. Both were refused.
    for (name, header) in [
        (
            "BADG00RUS",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/obs/BADG00RUS_R_20260010000_01D_30S_MO_header_trim.rnx"
            )),
        ),
        (
            "ARHT00ATA",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/obs/ARHT00ATA_R_20260010000_01D_30S_MO_header_trim.rnx"
            )),
        ),
    ] {
        let obs = RinexObs::parse(header).unwrap_or_else(|error| panic!("{name}: {error}"));
        // R28 in the slot table, the four lists and the count record.
        assert_eq!(obs.skipped_records, 6, "{name}");
        let glonass: Vec<&ObsPhaseShift> = obs
            .header()
            .phase_shifts
            .iter()
            .filter(|shift| shift.system == GnssSystem::Glonass)
            .collect();
        for code in ["L1C", "L1P", "L2C", "L2P"] {
            let shift = glonass
                .iter()
                .find(|shift| shift.code.as_deref() == Some(code))
                .unwrap_or_else(|| panic!("{name} {code}"));
            assert_eq!(unrepresentable_tokens(shift), ["R28"], "{name} {code}");
            assert!(!shift.satellites.is_empty(), "{name} {code}");
        }
        let l3x = glonass
            .iter()
            .find(|shift| shift.code.as_deref() == Some("L3X"))
            .expect("L3X");
        assert!(unrepresentable_tokens(l3x).is_empty(), "{name}");

        // Written back whole: the R28 tokens stay in their lists, and the
        // product reads back as itself. The slot and count R28 names are not
        // written, as before, so the skip count differs.
        let written = obs
            .to_rinex_string()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            written
                .lines()
                .filter(|line| line.ends_with("SYS / PHASE SHIFT") && line.contains(" R28"))
                .count(),
            4,
            "{name}"
        );
        let mut reread = RinexObs::parse(&written).expect("reads back");
        assert_eq!(
            reread.header().phase_shifts,
            obs.header().phase_shifts,
            "{name}"
        );
        assert_eq!(reread.skipped_records, 4, "{name}");
        let mut original = obs.clone();
        for product in [&mut original, &mut reread] {
            product.skipped_records = 0;
            product.header.unretained_header_labels.clear();
        }
        assert_eq!(reread, original, "{name}");

        // An epoch of R01 and R28: the corrections apply to R01, and R28's
        // record is skipped as epoch records naming it already were.
        let width = obs.header().obs_codes[&GnssSystem::Glonass].len();
        let values = vec![Some(1_000.0); width];
        let body = format!(
            "> 2026 01 01 00 00  0.0000000  0  2\n{}\n{}",
            obs_record("R01", &values),
            obs_record("R28", &values)
        );
        let with_epoch = RinexObs::parse(&format!("{}\n{body}", header.trim_end()))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(with_epoch.skipped_records, 7, "{name}");
        assert_eq!(l1_shift_at(&with_epoch, 0, "R01", "L1C"), 0.0, "{name}");
        assert_eq!(l1_shift_at(&with_epoch, 0, "R01", "L1P"), -0.25, "{name}");
        assert_eq!(l1_shift_at(&with_epoch, 0, "R01", "L2C"), 0.25, "{name}");
        assert_eq!(l1_shift_at(&with_epoch, 0, "R01", "L3X"), 0.25, "{name}");
    }
}

#[test]
fn a_phase_shift_satellite_no_satellite_id_holds_is_kept_and_counted() {
    let types = header_line("R    2 C1C L1C", "SYS / # / OBS TYPES");
    let shift = |content: &str| header_line(content, "SYS / PHASE SHIFT");
    let record = |sat: &str| obs_record(sat, &[Some(20_000_000.0), Some(100.0)]);
    let epoch = |minute: u8| {
        format!(
            "> 2020 01 01 00 {minute:02}  0.0000000  0  2\n{}\n{}",
            record("R01"),
            record("R02")
        )
    };
    let body = [
        epoch(0),
        blank_event(4, &[shift("R L1C  0.50000  03 R28 R02 R29")]),
        epoch(1),
    ]
    .join("\n");
    let text = obs_with_code_headers(&[types.clone(), shift("R L1C  0.25000  02 R01 R28")], &body);
    let obs = RinexObs::parse(&text).expect("parse");
    // R28 in the file header, R28 and R29 in the event.
    assert_eq!(obs.skipped_records, 3);
    let held = &obs.header().phase_shifts[0];
    assert_eq!(
        held.satellites,
        ["R01".parse::<GnssSatelliteId>().expect("R01")]
    );
    assert_eq!(unrepresentable_tokens(held), ["R28"]);
    // The corrections apply to the satellites held, and a record naming only
    // R28 besides them is not one for every satellite.
    assert_eq!(l1_shift_at(&obs, 0, "R01", "L1C"), 0.25);
    assert_eq!(l1_shift_at(&obs, 0, "R02", "L1C"), 0.0);
    assert_eq!(l1_shift_at(&obs, 2, "R01", "L1C"), 0.25);
    assert_eq!(l1_shift_at(&obs, 2, "R02", "L1C"), 0.5);
    // The event's record replaces R28 by its designator and keeps R01's.
    let after = obs.header_at(2).expect("header in effect");
    assert_eq!(after.phase_shifts.len(), 2);
    assert!(unrepresentable_tokens(&after.phase_shifts[0]).is_empty());
    assert_eq!(
        unrepresentable_tokens(&after.phase_shifts[1]),
        ["R28", "R29"]
    );
    assert_writes_back(&obs);

    // Records giving one designator two corrections in one block contradict
    // each other, keyed by the designator: both are kept, and the
    // contradiction is counted beside the two designators. Identical records,
    // and records for different designators, count only the designators.
    let block = |second: &str| {
        RinexObs::parse(&obs_with_code_headers(
            std::slice::from_ref(&types),
            &blank_event(4, &[shift("R L1C  0.25000  01 R28"), shift(second)]),
        ))
        .unwrap_or_else(|error| panic!("{second}: {error}"))
    };
    let contradicted = block("R L1C  0.50000  01 R28");
    assert_eq!(contradicted.skipped_records, 3);
    assert_eq!(
        contradicted
            .header_at(0)
            .expect("header in effect")
            .phase_shifts
            .len(),
        2
    );
    assert_writes_back(&contradicted);
    assert_eq!(block("R L1C  0.25000  01 R28").skipped_records, 2);
    assert_eq!(block("R L1C  0.50000  01 R29").skipped_records, 2);

    // A token that names no satellite is still refused.
    for token in ["R2X", "X28", "R", "28"] {
        assert_parse_err(minimal_obs_with_phase_shift(&format!(
            "R L1C  0.25000  01 {token}"
        )));
    }
}

#[test]
fn glonass_slots_and_biases_in_a_later_block_replace_only_their_own() {
    let types = [
        header_line("R    1 L1C", "SYS / # / OBS TYPES"),
        header_line("  2 R01  1 R09 -4", "GLONASS SLOT / FRQ #"),
        header_line("C1C   -9.000 C1P   -9.500", "GLONASS COD/PHS/BIS"),
    ];
    let epoch = |minute: u8| {
        format!(
            "> 2020 01 01 00 {minute:02}  0.0000000  0  1\n{}",
            obs_record("R01", &[Some(100.0)])
        )
    };
    let obs = RinexObs::parse(&obs_with_code_headers(
        &types,
        &[
            epoch(0),
            blank_event(
                4,
                &[
                    header_line("  1 R01 -6", "GLONASS SLOT / FRQ #"),
                    header_line("C1C  -10.000", "GLONASS COD/PHS/BIS"),
                ],
            ),
            epoch(1),
            blank_event(4, &[header_line("", "GLONASS COD/PHS/BIS")]),
            epoch(2),
        ]
        .join("\n"),
    ))
    .expect("parse");
    let timeline = obs.header_timeline().expect("timeline");
    assert_eq!(
        timeline.at(2).glonass_slots,
        BTreeMap::from([(1, -6), (9, -4)])
    );
    assert_eq!(
        timeline.at(2).glonass_cod_phs_bis,
        Some(vec![
            ("C1P".to_string(), Some(-9.5)),
            ("C1C".to_string(), Some(-10.0))
        ])
    );
    assert_eq!(timeline.at(4).glonass_cod_phs_bis, Some(Vec::new()));
    assert_writes_back(&obs);
}

/// What the round-trip fuzz target checks of an input: it reads, skips no
/// record, writes, and reads back as itself. An input that stops short of the
/// comparison checks nothing.
fn assert_round_trip_target_reaches_its_comparison(text: &str, what: &str) {
    let original = RinexObs::parse(text).unwrap_or_else(|error| panic!("{what}: {error}"));
    assert_eq!(original.skipped_records, 0, "{what}");
    let encoded = original
        .to_rinex_string()
        .unwrap_or_else(|error| panic!("{what}: {error}"));
    let mut reparsed = RinexObs::parse(&encoded).expect("reads back");
    let mut original = original;
    original.header.unretained_header_labels.clear();
    reparsed.header.unretained_header_labels.clear();
    assert_eq!(reparsed, original, "{what}");
    assert_split_events_agree(&original, what);
}

#[test]
fn inputs_the_review_found_refused_reach_the_round_trip_comparison() {
    let comments: Vec<String> = (0..100)
        .map(|index| header_line(&format!("record {index}"), "COMMENT"))
        .collect();
    let mut hundred = vec![
        header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        header_line("     1    C1", "# / TYPES OF OBSERV"),
        header_line("", "END OF HEADER"),
        format!("{}4100", " ".repeat(28)),
    ];
    hundred.extend(comments);
    assert_round_trip_target_reaches_its_comparison(&hundred.join("\n"), "a hundred records");
    for case in header_record_event_cases() {
        assert_round_trip_target_reaches_its_comparison(&case.text, &case.what);
    }
}

#[test]
fn the_rinex_211_example_new_site_records_take_effect() {
    // The example's flag 4 events carry WAVELENGTH FACT L1/2 and comments,
    // which change nothing a later epoch reads; its flag 3 event carries a
    // marker, a marker number and an antenna offset.
    let obs = RinexObs::parse(&rinex211_table_a7_example()).expect("parse");
    let timeline = obs.header_timeline().expect("timeline");
    assert_eq!(
        timeline
            .segments()
            .map(|(first, _)| first)
            .collect::<Vec<_>>(),
        vec![0, 5]
    );
    let site = timeline.at(5);
    assert_eq!(site.marker_name.as_deref(), Some("A 9080"));
    assert_eq!(site.marker_number.as_deref(), Some("9080.1.34"));
    assert_eq!(site.antenna_delta_hen_m, Some([0.903, 0.0, 0.0]));
}

#[test]
fn a_galileo_band_five_code_keeps_its_band() {
    // Galileo's `C5` and `P2` both canonicalise to `C5X`, so an inverse that
    // takes whichever it meets first can write `P2` for a band 5 pseudorange.
    // No reader outside this crate defines `P2` for Galileo.
    let candidates = rinex2_obs_code_candidates(GnssSystem::Galileo, "C5X", 2.11);
    assert_eq!(
        candidates.first().map(String::as_str),
        Some("C5"),
        "the code keeps the band it was measured on: {candidates:?}"
    );
    assert!(
        !candidates.iter().any(|name| name.starts_with('P')),
        "version 2 gives Galileo no `P` observable: {candidates:?}"
    );

    // `C5Q` is band 5 too. Version 2 has no field for the tracking attribute,
    // so it is written `C5` and reads back as `C5X`, losing the `Q`. That is the
    // most version 2 can say; naming a different band to keep the attribute
    // would not be.
    assert_eq!(
        rinex2_obs_code_candidates(GnssSystem::Galileo, "C5Q", 2.11),
        vec!["C5".to_string()]
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Galileo, "C5", 2.11),
        "C5X"
    );
}

#[test]
fn version_two_point_twelve_names_the_civil_signals_by_letter() {
    // 2.12 gave the L1 and L2 civil signals their own letters and left the
    // digits to the P code, so `L1` there is the P(Y) phase and `LA` the C/A
    // one. Reading a letter as a band produced codes like `CAX`, which name no
    // signal at all.
    for (system, declared, canonical) in [
        (GnssSystem::Gps, "CA", "C1C"),
        (GnssSystem::Gps, "LA", "L1C"),
        (GnssSystem::Gps, "CB", "C1X"),
        (GnssSystem::Gps, "CC", "C2X"),
        (GnssSystem::Gps, "L1", "L1W"),
        (GnssSystem::Gps, "S1", "S1W"),
        (GnssSystem::Glonass, "CD", "C2C"),
        (GnssSystem::Glonass, "L1", "L1P"),
        (GnssSystem::Qzss, "CC", "C2X"),
    ] {
        assert_eq!(
            canonical_rinex2_obs_code(system, declared, 2.12),
            canonical,
            "{system:?} {declared} at 2.12"
        );
    }
    // And a 2.12 product is written back under those letters. Offering only
    // digits wrote `LA L1 CB CC` as `L1 L1 C1 C2`, which reads as four
    // different signals from the ones the file held.
    let text = concat!(
        "     2.12           OBSERVATION DATA    G (GPS)             RINEX VERSION / TYPE\n",
        "     4    LA    L1    CB    CC                              # / TYPES OF OBSERV\n",
        "  2015     1     1     0     0    0.0000000     GPS         TIME OF FIRST OBS\n",
        "                                                            END OF HEADER\n",
        " 15  1  1  0  0  0.0000000  0  1G 1\n",
        "         1.000         2.000         3.000         4.000\n",
    );
    let obs = RinexObs::parse(text).expect("parse the 2.12 file");
    assert_eq!(
        obs.header().obs_codes[&GnssSystem::Gps],
        vec!["L1C", "L1W", "C1X", "C2X"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );
    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    let declared = encoded
        .lines()
        .find(|line| line.contains("# / TYPES OF OBSERV"))
        .expect("the header names its types");
    assert_eq!(
        &declared[..30],
        "     4    LA    L1    CB    CC",
        "{declared:?}"
    );
    let reparsed = RinexObs::parse(&encoded).expect("reads back");
    assert_eq!(reparsed.header().obs_codes, obs.header().obs_codes);
    assert_eq!(reparsed.epochs(), obs.epochs());
    // The digits a letter replaced are no longer names at 2.12: `C1` is
    // refused outright, and `L1` names the P(Y) phase only where there is one.
    // Offering them first wrote a GPS `CA` as `C1` and a QZSS `LA` as `L1`,
    // which the reference reader does not recognise at 2.12. GLONASS `C2` is
    // still its G2 C/A at 2.12, so `CD` is an alias there, not a replacement.
    for (system, canonical, letter) in [
        (GnssSystem::Gps, "C1C", "CA"),
        (GnssSystem::Glonass, "C1C", "CA"),
        (GnssSystem::Qzss, "C1C", "CA"),
        (GnssSystem::Sbas, "C1C", "CA"),
        (GnssSystem::Qzss, "L1C", "LA"),
        (GnssSystem::Sbas, "S1C", "SA"),
        (GnssSystem::Gps, "C1X", "CB"),
    ] {
        assert_eq!(
            rinex2_obs_code_candidates(system, canonical, 2.12).first(),
            Some(&letter.to_string()),
            "{system:?} {canonical} at 2.12"
        );
    }

    // At 2.11 the digits still name the civil signals.
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Gps, "L1", 2.11),
        "L1C"
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Glonass, "L1", 2.11),
        "L1C"
    );
}

#[test]
fn a_version_two_name_has_to_name_a_band_the_constellation_measures() {
    // Version 2 shares one digit space across every constellation, so a digit
    // is only this one's name where it names a band this one measures. GPS has
    // no L5 P code and Galileo no band 2, and both were offered as names.
    assert!(!rinex2_obs_code_candidates(GnssSystem::Gps, "C5X", 2.11)
        .iter()
        .any(|name| name == "P5"));
    for (system, name) in [
        (GnssSystem::Gps, "P5"),
        (GnssSystem::Glonass, "P3"),
        (GnssSystem::Galileo, "C2"),
        (GnssSystem::Sbas, "C2"),
        (GnssSystem::Navic, "C5"),
    ] {
        assert!(
            !rinex2_name_allowed(system, name, 2.11),
            "{system:?} has no {name}"
        );
    }
    for (system, name) in [
        (GnssSystem::Gps, "P2"),
        (GnssSystem::Galileo, "C7"),
        (GnssSystem::BeiDou, "C6"),
    ] {
        assert!(
            rinex2_name_allowed(system, name, 2.11),
            "{system:?} has {name}"
        );
    }
}

#[test]
fn version_two_gives_only_gps_and_glonass_a_p_observable() {
    // Version 2 says "P: Pseudorange GPS and Glonass: P code". Galileo and
    // BeiDou carried `P` rows anyway, holding the legacy
    // differential-code-bias labels, where `P1` and `P2` mean the first and
    // second frequency whatever the constellation. For BeiDou that made `C2`
    // B2I and `P2` B3I: two spellings of the same digit naming different bands.
    // 2.11 section 10.1.1 added `C2` for the L2C pseudorange, which RINEX 3
    // spells by channel. `C2C` is L2 C/A, a different signal.
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Gps, "C2", 2.11),
        "C2X"
    );
    // From 2.12 the same name is L2P(Y): 2.12 gave L2C its own names.
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Gps, "C2", 2.12),
        "C2W"
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Gps, "P1", 2.11),
        "C1W"
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Gps, "P2", 2.11),
        "C2W"
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Glonass, "P1", 2.11),
        "C1P"
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Glonass, "P2", 2.11),
        "C2P"
    );
    // BeiDou's `C` rows stay: version 2 has no BeiDou at all, and the receivers
    // that wrote it numbered B1, B2, B3 as 1, 2, 3.
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::BeiDou, "C1", 2.11),
        "C2I"
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::BeiDou, "C2", 2.11),
        "C2I"
    );
    assert_ne!(
        canonical_rinex2_obs_code(GnssSystem::BeiDou, "P2", 2.11),
        "C6I"
    );
    for system in [GnssSystem::Galileo, GnssSystem::BeiDou, GnssSystem::Qzss] {
        for canonical in ["C1C", "C2I", "C5X", "C7I", "L1C"] {
            let candidates = rinex2_obs_code_candidates(system, canonical, 2.11);
            assert!(
                !candidates.iter().any(|name| name.starts_with('P')),
                "{system:?} {canonical} offers a `P` name: {candidates:?}"
            );
        }
    }
    // Dropping an attribute version 2 cannot carry must not also move the band.
    // `C2Q` is B1I with Q tracking; `C2` would read back as B2I.
    assert_eq!(
        rinex2_obs_code_candidates(GnssSystem::BeiDou, "C2Q", 2.11),
        vec!["C2".to_string()]
    );
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::BeiDou, "C1", 2.11),
        "C2I"
    );
    // Version 2 numbers by frequency slot across every constellation, so B1I is
    // slot 2 and some writers use slot 1 for it, while B2I and B3I sit in slots
    // 7 and 6, which RINEX 3 numbers the same. There is no slot 3.
    for (declared, canonical) in [
        ("C1", "C2I"),
        ("C2", "C2I"),
        ("L2", "L2I"),
        ("C7", "C7I"),
        ("C6", "C6I"),
        ("C3", "C3X"),
    ] {
        assert_eq!(
            canonical_rinex2_obs_code(GnssSystem::BeiDou, declared, 2.11),
            canonical,
            "BeiDou {declared}"
        );
    }
    // And the digit means the same band whatever the kind. It used to be
    // remapped only for `C`, so a file's `C1` was B1I and its `L1` was B1C:
    // one measurement pair read as two different signals.
    for (declared, canonical) in [
        ("C1", "C2I"),
        ("L1", "L2I"),
        ("D1", "D2I"),
        ("S1", "S2I"),
        ("C2", "C2I"),
        ("L2", "L2I"),
    ] {
        assert_eq!(
            canonical_rinex2_obs_code(GnssSystem::BeiDou, declared, 2.11),
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
    // Every band comes back combined: version 2 names no Galileo channel, so
    // claiming one would say more than the file did.
    for (declared, canonical) in [
        ("C1", "C1X"),
        ("C5", "C5X"),
        ("C6", "C6X"),
        ("C7", "C7X"),
        ("C8", "C8X"),
        ("L5", "L5X"),
    ] {
        assert_eq!(
            canonical_rinex2_obs_code(GnssSystem::Galileo, declared, 2.11),
            canonical,
            "Galileo {declared}"
        );
    }
    // A `C2` version 2 never should have carried is read as the band it names.
    assert_eq!(
        canonical_rinex2_obs_code(GnssSystem::Galileo, "C2", 2.11),
        "C2X"
    );
    assert_eq!(
        rinex2_obs_code_candidates(GnssSystem::Galileo, "C5X", 2.11),
        vec!["C5".to_string()]
    );
}

#[test]
fn a_version_two_file_keeps_the_leap_second_extras() {
    // The future count, week and day arrived with version 3, and the writer used
    // to leave them out of a version 2 file. This reader takes them from their
    // columns at any version and a version 2 reader that knows only the first
    // field ignores the rest, so they are written and read back.
    let mut obs = version_two_fixture();
    let leap = super::ObsLeapSeconds {
        current: 17,
        delta_future: Some(18),
        week: Some(2000),
        day: Some(3),
    };
    obs.header.leap_seconds = Some(leap);

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
    let reparsed = RinexObs::parse(&encoded).expect("reads back");
    assert_eq!(reparsed.header().leap_seconds, Some(leap));
}

#[test]
fn a_version_two_file_keeps_version_three_records_as_extension_records() {
    // `MARKER TYPE`, `SIGNAL STRENGTH UNIT`, `GLONASS COD/PHS/BIS` and
    // `SYS / PHASE SHIFT` arrived with version 3. The writer used to leave them
    // out of a version 2 file and lose them. This reader keeps them at any
    // version and a reader that does not know one skips it, so they are written
    // as extension records, the way `GLONASS SLOT / FRQ #` already was, and read
    // back.
    let mut obs = version_two_fixture();
    obs.header.marker_type = Some("GEODETIC".to_string());
    obs.header.signal_strength_unit = Some("DBHZ".to_string());
    obs.header.glonass_cod_phs_bis = Some(vec![("C1C".to_string(), Some(-71.940))]);
    obs.header.phase_shifts.push(super::ObsPhaseShift {
        system: GnssSystem::Gps,
        code: Some("L1C".to_string()),
        correction_cycles: Some(0.25),
        satellites: Vec::new(),
        unrepresentable_satellites: Vec::new(),
    });

    let text = obs.to_rinex_string().expect("serialize RINEX OBS");
    for label in [
        "MARKER TYPE",
        "SIGNAL STRENGTH UNIT",
        "GLONASS COD/PHS/BIS",
        "SYS / PHASE SHIFT",
    ] {
        assert!(text.contains(label), "{label} is written");
    }
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.header().marker_type, obs.header().marker_type);
    assert_eq!(read.header().phase_shifts, obs.header().phase_shifts);

    // A scale factor is not one of them: a version 2 reader that does not
    // apply it would read the scaled numbers as physical, so it is refused.
    obs.header.scale_factors.push(super::ObsScaleFactor {
        system: GnssSystem::Gps,
        factor: 1000.0,
        codes: Vec::new(),
    });
    assert!(matches!(
        obs.to_rinex_string(),
        Err(RinexObsWriteError::ScaleFactorsInVersionTwo { count: 1 })
    ));
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

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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

    let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
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
    let rows = carrier_phase_rows(obs.header(), &obs.epochs()[0], &ObservationFilter::all())
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
fn to_rinex_string_refuses_a_time_scale_a_header_cannot_name() {
    // TIME OF FIRST OBS labels its time system with three letters, and TCG has
    // none. The writer used to leave the record out and return the text anyway,
    // which read back with no first epoch at all. It refuses now, and names the
    // field that would have changed.
    let first = header_line(
        "  2020     6    25     0     0    0.0000000     GPS",
        "TIME OF FIRST OBS",
    );
    let mut obs = RinexObs::parse(&minimal_obs(&[first], "")).expect("parse OBS");
    let first_epoch = obs.header.time_of_first_obs.expect("first stamp").0;
    obs.header.time_of_first_obs = Some((first_epoch, TimeScale::Tcg));

    let error = obs
        .to_rinex_string()
        .expect_err("a time scale with no label is refused");
    assert!(
        matches!(error, RinexObsWriteError::ReadBackMismatch { ref what } if what.contains("time_of_first_obs")),
        "{error}"
    );
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
    assert_eq!(obs.epochs()[0].epoch.map(|time| time.second), Some(60.0));
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
    assert_eq!(obs.epochs()[0].epoch.map(|time| time.second), Some(60.0));
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

    let serialized = obs.to_rinex_string().expect("serialize RINEX OBS");
    let reparsed = RinexObs::parse(&serialized).expect("re-parse serialized RINEX OBS");
    let mut expected = obs;
    expected.header.unretained_header_labels.clear();
    assert_eq!(
        reparsed, expected,
        "to_rinex_string must round-trip through parse"
    );
    // Deterministic output.
    assert_eq!(
        reparsed.to_rinex_string().expect("serialize RINEX OBS"),
        serialized
    );
}

#[test]
fn a_real_mixed_version_three_file_survives_being_written_as_version_two() {
    // A real mixed version 3 file, several constellations across 120 epochs, has
    // per-constellation code lists that no one version 2 list reads as, so
    // writing it as version 2 is refused. The downgrade lays it out, reports
    // every change, and the result writes and reads back exactly, twice. Every
    // measurement keeps its kind, band, value and both indicators: the tracking
    // attribute is the one thing version 2 may lose, so it is the one thing not
    // compared.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
    );
    let text = std::fs::read_to_string(path).expect("read the committed fixture");
    let original = RinexObs::parse(&text).expect("parse the version 3 fixture");
    assert!(
        original.header().obs_codes.len() >= 3,
        "the fixture is mixed"
    );

    let mut relabelled = original.clone();
    relabelled.header.version = 2.11;
    assert!(
        matches!(
            relabelled.to_rinex_string(),
            Err(RinexObsWriteError::CodeListsNotVersionTwo { .. })
        ),
        "a version 3 product is not a version 2 file by relabelling it"
    );

    let (downgraded, changes) = original.downgrade_to_rinex2(2.11).expect("downgrade");
    assert!(
        changes.iter().all(|change| matches!(
            change,
            ObsDowngradeChange::CodeRenamed { .. }
                | ObsDowngradeChange::CodeMoved { .. }
                | ObsDowngradeChange::CodeAdded { .. }
        )),
        "only codes renamed, moved or added: {changes:?}"
    );
    let once = downgraded.to_rinex_string().expect("the downgrade writes");
    assert!(!once.lines().any(|line| line.starts_with('>')));
    let reparsed = RinexObs::parse(&once).expect("reads back");
    let twice = reparsed.to_rinex_string().expect("and writes again");
    assert_eq!(twice, once, "a second write is byte-identical");

    type Measurement = (GnssSatelliteId, char, char, u64, Option<u8>, Option<u8>);
    fn measurements(obs: &RinexObs, epoch: &ObsEpoch) -> Vec<Measurement> {
        let mut out = Vec::new();
        for (sat, values) in &epoch.sats {
            let codes = &obs.header().obs_codes[&sat.system];
            for (code, value) in codes.iter().zip(values) {
                let (Some(v), Some(kind), Some(band)) =
                    (value.value, code.chars().next(), code.chars().nth(1))
                else {
                    continue;
                };
                out.push((*sat, kind, band, v.to_bits(), value.lli, value.ssi));
            }
        }
        out.sort();
        out
    }
    assert_eq!(reparsed.epochs().len(), original.epochs().len());
    let mut compared = 0_usize;
    for (before, after) in original.epochs().iter().zip(reparsed.epochs()) {
        let expected = measurements(&original, before);
        assert_eq!(measurements(&reparsed, after), expected);
        compared += expected.len();
    }
    assert!(
        compared > 10_000,
        "the fixture carries real data to compare, not {compared} values"
    );
}

#[test]
fn a_name_a_constellation_lacks_does_not_swallow_the_one_beside_it() {
    // With the header ordered `P1 C1`, Galileo's `P1` used to read as its
    // `C1X`, the same code its `C1` reads as, and a writer choosing between
    // two positions holding one code kept the first: the blank one. The name
    // Galileo has no observable under now stays as the file wrote it, so its
    // measurement has one position and keeps it.
    let mut text = String::new();
    text.push_str(
        "     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n",
    );
    text.push_str(&format!(
        "{:<60}{:<20}\n",
        "     2    P1    C1", "# / TYPES OF OBSERV"
    ));
    text.push_str(&format!(
        "{:<60}{:<20}\n",
        "  2015     1     1     0     0    0.0000000     GPS", "TIME OF FIRST OBS"
    ));
    text.push_str(&format!("{:<60}{:<20}\n", "", "END OF HEADER"));
    text.push_str(" 15  1  1  0  0  0.0000000  0  2G 1E11\n");
    text.push_str("       123.000       234.000\n");
    text.push_str("                     456.000\n");
    let first = RinexObs::parse(&text).expect("parse");
    assert_eq!(
        first.header().obs_codes[&GnssSystem::Galileo],
        vec!["P1".to_string(), "C1X".to_string()],
        "Galileo has no P1, and the name stays as written"
    );
    let galileo = GnssSatelliteId {
        system: GnssSystem::Galileo,
        prn: 11,
    };
    let mut obs = first.clone();
    for round in 0..3 {
        let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
        obs = RinexObs::parse(&encoded).expect("reads back");
        assert_eq!(
            obs.epochs()[0].sats[&galileo][1].value,
            Some(456.0),
            "rewrite {round} lost Galileo's measurement"
        );
        assert_eq!(obs.epochs(), first.epochs(), "rewrite {round}");
    }
}

#[test]
fn a_second_name_for_a_code_already_held_stays_as_written() {
    // BeiDou's version 2 `C1` and `C2` are both B1I, and a file that carries
    // both declares one signal twice, which no convention produces. Reading
    // both as `C2I` put one code at two positions, and with NavIC beside it,
    // which has no version 2 names at all, the header grew a column on every
    // rewrite while the second value walked across. The second name now stays
    // as the file wrote it, so each value has a code of its own. The first
    // write may still order the columns differently from the file; after that
    // the header holds, and every value stays under the code it was read from.
    let mut text = String::new();
    text.push_str(
        "     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n",
    );
    text.push_str(&format!(
        "{:<60}{:<20}\n",
        "     2    C1    C2", "# / TYPES OF OBSERV"
    ));
    text.push_str(&format!(
        "{:<60}{:<20}\n",
        "  2015     1     1     0     0    0.0000000     GPS", "TIME OF FIRST OBS"
    ));
    text.push_str(&format!("{:<60}{:<20}\n", "", "END OF HEADER"));
    text.push_str(" 15  1  1  0  0  0.0000000  0  2C 5I 1\n");
    text.push_str("  20000000.000  20000001.000\n");
    text.push('\n');
    let first = RinexObs::parse(&text).expect("parse");
    assert_eq!(
        first.header().obs_codes[&GnssSystem::BeiDou],
        vec!["C2I".to_string(), "C2".to_string()],
        "the second name for B1I stays as written"
    );
    let beidou = GnssSatelliteId {
        system: GnssSystem::BeiDou,
        prn: 5,
    };
    // What each BeiDou code holds, whatever column it lands in.
    fn by_code(obs: &RinexObs, sat: GnssSatelliteId) -> Vec<(String, f64)> {
        let mut pairs: Vec<(String, f64)> = obs.header().obs_codes[&sat.system]
            .iter()
            .zip(&obs.epochs()[0].sats[&sat])
            .filter_map(|(code, value)| Some((code.clone(), value.value?)))
            .collect();
        pairs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        pairs
    }
    let expected = by_code(&first, beidou);
    assert_eq!(
        expected,
        vec![
            ("C2".to_string(), 20_000_001.0),
            ("C2I".to_string(), 20_000_000.0)
        ]
    );
    let mut obs = first.clone();
    let mut settled: Option<String> = None;
    for round in 0..4 {
        let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
        let declared = encoded
            .lines()
            .find(|line| line.contains("# / TYPES OF OBSERV"))
            .expect("the header names its types")[..60]
            .trim_end()
            .to_string();
        if let Some(previous) = &settled {
            assert_eq!(&declared, previous, "rewrite {round} changed the header");
        }
        settled = Some(declared);
        obs = RinexObs::parse(&encoded).expect("reads back");
        assert_eq!(by_code(&obs, beidou), expected, "rewrite {round}");
    }
}

#[test]
fn every_mixed_version_two_pair_keeps_its_measurements_by_code() {
    // The class of input that has broken this writer, swept rather than
    // sampled: every ordered pair of distinct version 2 names, for every pair
    // of constellations, with the second constellation holding both values, only
    // the first, or only the second. Each file is written and read four times.
    // Every populated measurement has to come back under the code it was read
    // with, loss-of-lock and signal strength included, and the header has to
    // match from the first write on. Tracking attributes cannot change here,
    // because every code is compared as read. `D` and `S` map exactly as `L`
    // does, so `C`, `P` and `L` cover every mapping class at a third the cost.
    let systems = [
        ('G', GnssSystem::Gps),
        ('R', GnssSystem::Glonass),
        ('E', GnssSystem::Galileo),
        ('C', GnssSystem::BeiDou),
        ('J', GnssSystem::Qzss),
        ('S', GnssSystem::Sbas),
        ('I', GnssSystem::Navic),
    ];
    type Measurement = (GnssSatelliteId, String, f64, Option<u8>, Option<u8>);
    fn by_code(obs: &RinexObs) -> Vec<Measurement> {
        let mut out = Vec::new();
        for epoch in obs.epochs() {
            for (sat, values) in &epoch.sats {
                let codes = &obs.header().obs_codes[&sat.system];
                for (code, value) in codes.iter().zip(values) {
                    if let Some(v) = value.value {
                        out.push((*sat, code.clone(), v, value.lli, value.ssi));
                    }
                }
            }
        }
        out.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        out
    }
    let mut cases = 0_usize;
    for version in [2.11, 2.12] {
        let letters: &[char] = if version >= 2.12 {
            &['A', 'B', 'C', 'D']
        } else {
            &[]
        };
        let names: Vec<String> = ['C', 'P', 'L']
            .iter()
            .flat_map(|kind| {
                ['1', '2', '3', '5', '6', '7', '8']
                    .iter()
                    .chain(letters)
                    .map(move |band| format!("{kind}{band}"))
            })
            .filter(|name| {
                systems
                    .iter()
                    .any(|(_, system)| rinex2_name_allowed(*system, name, version))
            })
            .collect();
        for (i, (letter_a, _)) in systems.iter().enumerate() {
            for (letter_b, _) in &systems[i + 1..] {
                for first in &names {
                    for second in &names {
                        if first == second {
                            continue;
                        }
                        for pattern in 0..3 {
                            // Each value takes sixteen columns: F14.3, then a loss-of-lock
                            // and a signal-strength digit.
                            let b_values = match pattern {
                                0 => "  20000001.000    20000002.00012",
                                1 => "  20000001.000",
                                _ => "                  20000002.00012",
                            };
                            let types = format!("     2    {first}    {second}");
                            let epoch_line =
                                format!(" 15  1  1  0  0  0.0000000  0  2{letter_a} 1{letter_b} 2");
                            let text = format!(
                                "{:9.2}{:11}{:<20}{:<20}RINEX VERSION / TYPE\n\
                                 {:<60}# / TYPES OF OBSERV\n\
                                 {:<60}TIME OF FIRST OBS\n\
                                 {:<60}END OF HEADER\n\
                                 {}\n  10000001.000    10000002.000\n{b_values}\n",
                                version,
                                "",
                                "OBSERVATION DATA",
                                "M (MIXED)",
                                types,
                                "  2015     1     1     0     0    0.0000000     GPS",
                                "",
                                epoch_line,
                            );
                            let label = format!(
                                "{version} {letter_a}{letter_b} {first} {second} pattern {pattern}"
                            );
                            let original = RinexObs::parse(&text)
                                .unwrap_or_else(|e| panic!("{label}: parse: {e}"));
                            let expected = by_code(&original);
                            let mut obs = original;
                            let mut settled: Option<String> = None;
                            for round in 0..4 {
                                let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
                                let header: String = encoded
                                    .lines()
                                    .filter(|line| line.contains("# / TYPES OF OBSERV"))
                                    .collect();
                                if round > 0 {
                                    assert_eq!(
                                        Some(&header),
                                        settled.as_ref(),
                                        "{label}: rewrite {round} changed the header"
                                    );
                                }
                                settled = Some(header);
                                obs = RinexObs::parse(&encoded)
                                    .unwrap_or_else(|e| panic!("{label}: rewrite {round}: {e}"));
                                assert_eq!(
                                    by_code(&obs),
                                    expected,
                                    "{label}: rewrite {round} moved a measurement off its code"
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 10_000, "the sweep ran {cases} cases");
}

#[test]
fn a_header_with_no_observations_reads_its_names_by_the_same_rule() {
    // A file with no epochs builds its constellation's list from the header
    // alone, down a separate path that read every name as a code: Galileo's
    // `P1 C1` as `C1X C1X`, and BeiDou's `C1 C2` as `C2I C2I`, one code at two
    // positions again. Both paths now share one rule.
    for (letter, system, names, expected) in [
        (
            'E',
            GnssSystem::Galileo,
            "     2    P1    C1",
            ["P1", "C1X"],
        ),
        ('C', GnssSystem::BeiDou, "     2    C1    C2", ["C2I", "C2"]),
    ] {
        let text = format!(
            "{:9.2}{:11}{:<20}{:<20}RINEX VERSION / TYPE\n\
             {names:<60}# / TYPES OF OBSERV\n\
             {:<60}TIME OF FIRST OBS\n\
             {:<60}END OF HEADER\n",
            2.11,
            "",
            "OBSERVATION DATA",
            letter,
            "  2015     1     1     0     0    0.0000000     GPS",
            "",
        );
        let obs = RinexObs::parse(&text).expect("a header-only file parses");
        assert_eq!(
            obs.header().obs_codes[&system],
            expected.map(String::from).to_vec(),
            "{system:?}"
        );
        let reparsed = RinexObs::parse(&obs.to_rinex_string().expect("serialize RINEX OBS"))
            .expect("and reads back");
        assert_eq!(
            reparsed.header().obs_codes,
            obs.header().obs_codes,
            "{system:?}"
        );
    }
}

/// One constellation's part of a small version 2 product: its system, the letter
/// and PRN its satellite is written with, and its code list.
type SmallList = (GnssSystem, char, u8, Vec<String>);

/// A two-constellation version 2 product with exactly these code lists and one
/// satellite each. Values are blank, or distinct per satellite and position when
/// `filled`, so a value that moves can be told apart from one that stays.
fn two_system_product(version: f64, a: &SmallList, b: &SmallList, filled: bool) -> RinexObs {
    let width = a.3.len().max(b.3.len()).max(1);
    // Each record is its own argument: a `\` continuation inside a string
    // literal strips the next line's leading spaces, and the epoch line's first
    // column is a space.
    let header = |content: &str, label: &str| format!("{content:<60}{label}\n");
    let text = [
        header(
            &format!(
                "{version:9.2}{:11}{:<20}{:<20}",
                "", "OBSERVATION DATA", "M (MIXED)"
            ),
            "RINEX VERSION / TYPE",
        ),
        header(
            &format!("{width:6}{}", "    C5".repeat(width)),
            "# / TYPES OF OBSERV",
        ),
        header(
            "  2015     1     1     0     0    0.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
        header("", "END OF HEADER"),
        format!(
            " 15  1  1  0  0  0.0000000  0  2{}{:2}{}{:2}\n",
            a.1, a.2, b.1, b.2
        ),
        "\n\n".to_string(),
    ]
    .concat();
    let mut product = RinexObs::parse(&text).expect("parse the small version 2 template");
    product.header.obs_codes.clear();
    product.header.declared_obs_codes.clear();
    let epoch = &mut product.epochs[0];
    epoch.sats.clear();
    for (base, (system, _, prn, list)) in [(1000.0, a), (2000.0, b)] {
        product.header.obs_codes.insert(*system, list.clone());
        product
            .header
            .declared_obs_codes
            .insert(*system, list.clone());
        let values = (0..list.len())
            .map(|index| ObsValue {
                value: filled.then_some(base + index as f64),
                lli: None,
                ssi: None,
            })
            .collect();
        epoch.sats.insert(
            GnssSatelliteId {
                system: *system,
                prn: *prn,
            },
            values,
        );
    }
    product
}

/// Every version 2 name the reader or writer can meet: the five kinds on every
/// digit band and every 2.12 letter. Names a constellation cannot carry stay in,
/// because the reader keeps them as written.
fn every_version_two_name() -> Vec<String> {
    let mut names: Vec<String> = ['C', 'P', 'L', 'D', 'S']
        .iter()
        .flat_map(|kind| {
            "123456789ABCD"
                .chars()
                .map(move |band| format!("{kind}{band}"))
        })
        .collect();
    // A type field is two characters wide, and a one-character name in it is
    // kept as written too.
    names.extend(["Z".to_string(), "C".to_string()]);
    names
}

const SMALL_PAIRS: [(GnssSystem, char, u8, GnssSystem, char, u8); 4] = [
    (GnssSystem::Gps, 'G', 1, GnssSystem::Glonass, 'R', 2),
    (GnssSystem::Gps, 'G', 1, GnssSystem::Galileo, 'E', 11),
    (GnssSystem::Gps, 'G', 1, GnssSystem::BeiDou, 'C', 5),
    (GnssSystem::Glonass, 'R', 2, GnssSystem::BeiDou, 'C', 5),
];

#[test]
fn version_two_writer_refuses_exactly_the_lists_no_name_list_states() {
    // The strict writer looks for a version 2 name list a position at a time.
    // This checks that search against brute force over every two-name sequence
    // drawn from the whole name space: for small two-constellation products,
    // writing succeeds exactly when some sequence reads back as both lists, and
    // otherwise refuses with the error that says no list exists. A refusal of a
    // product some sequence could state would be a writer saying no when it
    // could have said yes.
    let space = every_version_two_name();
    let mut checked = 0_usize;
    let mut refused = 0_usize;
    for version in [2.11, 2.12] {
        let mut sample: Vec<&str> = vec!["C1", "P1", "L1", "C2", "P2", "C5", "Z", "C"];
        if version >= 2.12 {
            sample.extend(["CA", "CC"]);
        }
        for (sa, la, pa, sb, lb, pb) in SMALL_PAIRS {
            let stated: std::collections::BTreeSet<(Vec<String>, Vec<String>)> = space
                .iter()
                .flat_map(|first| {
                    space
                        .iter()
                        .map(move |second| vec![first.clone(), second.clone()])
                })
                .map(|names| {
                    (
                        rinex2_system_obs_codes(sa, &names, version),
                        rinex2_system_obs_codes(sb, &names, version),
                    )
                })
                .collect();
            let readings = |system: GnssSystem| -> Vec<Vec<String>> {
                let mut lists: std::collections::BTreeSet<Vec<String>> = sample
                    .iter()
                    .flat_map(|first| {
                        sample
                            .iter()
                            .map(move |second| vec![(*first).to_string(), (*second).to_string()])
                    })
                    .map(|names| rinex2_system_obs_codes(system, &names, version))
                    .collect();
                // And lists holding a code no version 2 name reads back as.
                let spellable = lists.iter().next().cloned().expect("a reading");
                lists.insert(vec!["C9X".to_string(), spellable[1].clone()]);
                lists.insert(vec![spellable[0].clone(), "C9X".to_string()]);
                lists.into_iter().collect()
            };
            for list_a in readings(sa) {
                for list_b in readings(sb) {
                    let expected = stated.contains(&(list_a.clone(), list_b.clone()));
                    let product = two_system_product(
                        version,
                        &(sa, la, pa, list_a.clone()),
                        &(sb, lb, pb, list_b.clone()),
                        false,
                    );
                    let result = product.to_rinex_string();
                    match (&result, expected) {
                        (Ok(_), true) => {}
                        (Err(RinexObsWriteError::CodeListsNotVersionTwo { .. }), false) => {
                            refused += 1;
                        }
                        _ => panic!(
                            "{version} {sa:?} {list_a:?} beside {sb:?} {list_b:?}: some name \
                             list states it: {expected}, but writing gave {result:?}"
                        ),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 1_000 && refused > 100,
        "{checked} products checked, {refused} refused"
    );
}

#[test]
fn downgrade_states_every_small_mixed_product_and_names_every_change() {
    // The downgrade is the path for a product version 2 cannot state as it is.
    // For every small two-constellation product - lists of different lengths,
    // codes no version 2 name spells, names kept as written - its result has to
    // write, every value has to come back exactly once, and a value that comes
    // back under a code other than its own has to sit under a rename the change
    // list names. Anything else is a change nobody was told about.
    let alphabet = |system: GnssSystem| -> Vec<&'static str> {
        match system {
            GnssSystem::Gps => vec!["C1C", "C1W", "C2X", "L1C", "C9X"],
            GnssSystem::Glonass => vec!["C1C", "C1P", "C2C", "L1C", "C9X"],
            GnssSystem::Galileo => vec!["C1X", "C5X", "L1X", "P1", "C9X"],
            _ => vec!["C2I", "C7I", "C6I", "C2", "C9X"],
        }
    };
    let lists = |system: GnssSystem| -> Vec<Vec<String>> {
        let codes = alphabet(system);
        let mut out: Vec<Vec<String>> =
            codes.iter().map(|code| vec![(*code).to_string()]).collect();
        for first in &codes {
            for second in &codes {
                out.push(vec![(*first).to_string(), (*second).to_string()]);
            }
        }
        out
    };
    let mut products = 0_usize;
    for version in [2.11, 2.12] {
        for (sa, la, pa, sb, lb, pb) in SMALL_PAIRS {
            for list_a in lists(sa) {
                for list_b in lists(sb) {
                    let label = format!("{version} {sa:?} {list_a:?} beside {sb:?} {list_b:?}");
                    let original = two_system_product(
                        version,
                        &(sa, la, pa, list_a.clone()),
                        &(sb, lb, pb, list_b.clone()),
                        true,
                    );
                    let (downgraded, changes) = original
                        .downgrade_to_rinex2(version)
                        .unwrap_or_else(|e| panic!("{label}: downgrade: {e}"));
                    if original.to_rinex_string().is_ok() {
                        assert!(
                            changes.is_empty(),
                            "{label}: a product version 2 already states was changed: {changes:?}"
                        );
                    }
                    let text = downgraded
                        .to_rinex_string()
                        .unwrap_or_else(|e| panic!("{label}: the downgrade does not write: {e}"));
                    let read = RinexObs::parse(&text).expect("reads back");
                    for (sat, values) in &original.epochs()[0].sats {
                        let held = &original.header().obs_codes[&sat.system];
                        let now = &read.header().obs_codes[&sat.system];
                        let read_values = &read.epochs()[0].sats[sat];
                        for (index, value) in values.iter().enumerate() {
                            let Some(v) = value.value else { continue };
                            let columns: Vec<usize> = read_values
                                .iter()
                                .enumerate()
                                .filter(|(_, found)| found.value == Some(v))
                                .map(|(column, _)| column)
                                .collect();
                            assert_eq!(
                                columns.len(),
                                1,
                                "{label}: {sat} value {v} comes back {} times",
                                columns.len()
                            );
                            let (from, to) = (&held[index], &now[columns[0]]);
                            assert!(
                                from == to
                                    || changes.contains(&ObsDowngradeChange::CodeRenamed {
                                        system: sat.system,
                                        from: from.clone(),
                                        to: to.clone(),
                                    }),
                                "{label}: {sat} {from} came back under {to} with no change naming it: {changes:?}"
                            );
                        }
                    }
                    assert_moves_and_additions_are_itemized(
                        &label, version, &original, &text, &read, &changes,
                    );
                    products += 1;
                }
            }
        }
    }
    assert!(products > 1_000, "{products} products");
}

/// A version 2 product holding exactly these constellations' code lists, one
/// satellite each, every value distinct.
fn mixed_product(version: f64, lists: &[SmallList]) -> RinexObs {
    let mut product = two_system_product(
        version,
        &(GnssSystem::Gps, 'G', 1, vec!["C1C".to_string()]),
        &(GnssSystem::Glonass, 'R', 2, vec!["C1C".to_string()]),
        true,
    );
    product.header.obs_codes.clear();
    product.header.declared_obs_codes.clear();
    let epoch = &mut product.epochs[0];
    epoch.sats.clear();
    for (slot, (system, _, prn, list)) in lists.iter().enumerate() {
        product.header.obs_codes.insert(*system, list.clone());
        product
            .header
            .declared_obs_codes
            .insert(*system, list.clone());
        let base = 1_000.0 * (slot + 1) as f64;
        let values = (0..list.len())
            .map(|index| ObsValue {
                value: Some(base + index as f64),
                lli: (index % 2 == 0).then_some(1 + (slot % 7) as u8),
                ssi: Some(1 + ((slot + index) % 9) as u8),
            })
            .collect();
        epoch.sats.insert(
            GnssSatelliteId {
                system: *system,
                prn: *prn,
            },
            values,
        );
        // A second satellite whose odd positions hold only indicators, which
        // have to land under their codes too.
        let indicators = (0..list.len())
            .map(|index| ObsValue {
                value: (index % 2 == 0).then_some(base + 500.0 + index as f64),
                lli: (index % 2 == 1).then_some(1 + (index % 7) as u8),
                ssi: Some(9 - (index % 9) as u8),
            })
            .collect();
        epoch.sats.insert(
            GnssSatelliteId {
                system: *system,
                prn: *prn + 1,
            },
            indicators,
        );
    }
    epoch.declared_record_count = epoch.sats.len();
    // A second epoch, its values distinct from the first's.
    let mut later = product.epochs[0].clone();
    later.epoch.as_mut().expect("a timed epoch").minute += 1;
    for values in later.sats.values_mut() {
        for value in values.iter_mut() {
            value.value = value.value.map(|held| held + 100_000.0);
        }
    }
    product.epochs.push(later);
    product
}

/// Downgrade a product and check everything the downgrade promises: the
/// result writes, a product already writable is untouched, every value comes
/// back exactly once, under its own code or a reported rename, and every move
/// and addition is reported and no more moves are made than an order of the
/// same columns needs.
fn assert_downgrade_states(label: &str, version: f64, original: &RinexObs) {
    let (downgraded, changes) = original
        .downgrade_to_rinex2(version)
        .unwrap_or_else(|e| panic!("{label}: downgrade: {e}"));
    if original.to_rinex_string().is_ok() {
        assert!(
            changes.is_empty(),
            "{label}: a writable product was changed: {changes:?}"
        );
    }
    let text = downgraded
        .to_rinex_string()
        .unwrap_or_else(|e| panic!("{label}: the downgrade does not write: {e}"));
    let read = RinexObs::parse(&text).unwrap_or_else(|e| panic!("{label}: reads back: {e}"));
    assert_eq!(
        read.epochs().len(),
        original.epochs().len(),
        "{label}: epochs"
    );
    for (epoch_index, epoch) in original.epochs().iter().enumerate() {
        let read_epoch = &read.epochs()[epoch_index];
        // Where each code landed, from the first satellite of its
        // constellation, whose values are all present and distinct.
        let mut landed: BTreeMap<GnssSystem, Vec<Option<usize>>> = BTreeMap::new();
        for (sat, values) in &epoch.sats {
            if landed.contains_key(&sat.system) {
                continue;
            }
            let read_values = &read_epoch.sats[sat];
            let row = values
                .iter()
                .map(|value| {
                    let v = value.value?;
                    let columns: Vec<usize> = read_values
                        .iter()
                        .enumerate()
                        .filter(|(_, found)| found.value == Some(v))
                        .map(|(column, _)| column)
                        .collect();
                    assert_eq!(
                        columns.len(),
                        1,
                        "{label}: epoch {epoch_index} {sat} value {v} comes back {} times",
                        columns.len()
                    );
                    Some(columns[0])
                })
                .collect();
            landed.insert(sat.system, row);
        }
        for (sat, values) in &epoch.sats {
            let held = &original.header().obs_codes[&sat.system];
            let now = &read.header().obs_codes[&sat.system];
            let read_values = &read_epoch.sats[sat];
            let row = &landed[&sat.system];
            for (index, value) in values.iter().enumerate() {
                let Some(column) = row.get(index).copied().flatten() else {
                    continue;
                };
                assert_eq!(
                    read_values[column], *value,
                    "{label}: epoch {epoch_index} {sat} {} came back changed",
                    held[index]
                );
                let (from, to) = (&held[index], &now[column]);
                assert!(
                    from == to
                        || changes.contains(&ObsDowngradeChange::CodeRenamed {
                            system: sat.system,
                            from: from.clone(),
                            to: to.clone(),
                        }),
                    "{label}: {sat} {from} came back under {to} with no change naming it: {changes:?}"
                );
            }
            for (column, found) in read_values.iter().enumerate() {
                if !row.contains(&Some(column)) {
                    assert_eq!(
                        *found,
                        ObsValue {
                            value: None,
                            lli: None,
                            ssi: None,
                        },
                        "{label}: epoch {epoch_index} {sat} column {column} holds what no code put there"
                    );
                }
            }
        }
    }
    assert_moves_and_additions_are_itemized(label, version, original, &text, &read, &changes);
}

/// A deterministic stream of numbers for sweeps too large to enumerate.
struct SweepRng(u64);

impl SweepRng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

/// Every constellation a version 2 file can hold, with a satellite and the
/// codes a sweep draws for it: codes version 2 names, codes it names under
/// another attribute, codes no version 2 name spells, and names kept as
/// written.
fn sweep_constellations() -> Vec<(GnssSystem, char, u8, Vec<&'static str>)> {
    vec![
        (
            GnssSystem::Gps,
            'G',
            1,
            vec![
                "Z", "C1C", "C1W", "C2X", "C2W", "L1C", "L2W", "D1C", "S1C", "C5X", "L5X", "C9X",
                "P1", "C1",
            ],
        ),
        (
            GnssSystem::Glonass,
            'R',
            2,
            vec!["C1C", "C1P", "C2C", "C2P", "L1C", "L2P", "C3X", "P2", "C4A"],
        ),
        (
            GnssSystem::Galileo,
            'E',
            11,
            vec!["C1X", "C5X", "C7X", "C8X", "L1X", "L5X", "C6X", "C1C", "P1"],
        ),
        (
            GnssSystem::BeiDou,
            'C',
            5,
            vec!["C2I", "C7I", "C6I", "L2I", "C1X", "C2", "L7I"],
        ),
        (
            GnssSystem::Qzss,
            'J',
            1,
            vec!["C1C", "C2X", "C5X", "L1C", "C6X", "C2L"],
        ),
        (GnssSystem::Sbas, 'S', 20, vec!["C1C", "L1C", "C5X", "C1W"]),
        (
            GnssSystem::Navic,
            'I',
            3,
            vec!["C5A", "L5A", "C9A", "C5", "Z"],
        ),
    ]
}

#[test]
fn downgrade_states_three_constellation_products_across_every_constellation() {
    // The exhaustive sweep covers two constellations and two codes each. These
    // products draw three constellations of all seven, lists of up to four
    // codes from wider alphabets, at both version 2 revisions.
    let constellations = sweep_constellations();
    let mut rng = SweepRng(0x9E37_79B9_7F4A_7C15);
    for round in 0..4_000 {
        let version = if round % 2 == 0 { 2.11 } else { 2.12 };
        let mut picked: Vec<usize> = Vec::new();
        while picked.len() < 3 {
            let at = rng.below(constellations.len());
            if !picked.contains(&at) {
                picked.push(at);
            }
        }
        let lists: Vec<SmallList> = picked
            .iter()
            .map(|&at| {
                let (system, letter, prn, alphabet) = &constellations[at];
                let length = 1 + rng.below(4);
                let list = (0..length)
                    .map(|_| alphabet[rng.below(alphabet.len())].to_string())
                    .collect();
                (*system, *letter, *prn, list)
            })
            .collect();
        let label = format!("{version} {lists:?}");
        assert_downgrade_states(&label, version, &mixed_product(version, &lists));
    }
}

#[test]
fn downgrade_orders_a_wide_product_and_reports_every_change() {
    // Forty codes a constellation, drawn with repeats, lay out wider than any
    // order can be checked exhaustively; the order search has to finish on it
    // and the result still has to state every change.
    let constellations = sweep_constellations();
    let mut rng = SweepRng(0xD1B5_4A32_D192_ED03);
    for version in [2.11, 2.12] {
        let lists: Vec<SmallList> = constellations[..3]
            .iter()
            .map(|(system, letter, prn, alphabet)| {
                let list = (0..40)
                    .map(|_| alphabet[rng.below(alphabet.len())].to_string())
                    .collect();
                (*system, *letter, *prn, list)
            })
            .collect();
        let label = format!("{version} wide");
        super::write::LAST_ORDER_SEARCH.with(|last| last.set((0, false, 0)));
        assert_downgrade_states(&label, version, &mixed_product(version, &lists));
        let (placements, exhausted, _) = super::write::LAST_ORDER_SEARCH.with(std::cell::Cell::get);
        assert!(
            !exhausted,
            "{label}: the order search stopped at its budget after {placements} placements"
        );
    }
}

#[test]
fn downgrade_orders_shuffled_distinct_codes_exactly() {
    // Twenty distinct codes a constellation, shuffled independently, so the
    // positions constellations want for a shared name disagree everywhere.
    let alphabets: [(GnssSystem, char, u8, [&str; 20]); 3] = [
        (
            GnssSystem::Gps,
            'G',
            1,
            [
                "C1C", "C1W", "C2X", "C2W", "L1C", "L2W", "D1C", "S1C", "C5X", "L5X", "C9X", "L2X",
                "D2W", "S2W", "D5X", "S5X", "C1X", "L1W", "D1W", "S1W",
            ],
        ),
        (
            GnssSystem::Glonass,
            'R',
            2,
            [
                "C1C", "C1P", "C2C", "C2P", "L1C", "L2P", "C3X", "D1C", "D2P", "S1C", "S2P", "L3X",
                "D3X", "S3X", "L2C", "D2C", "S2C", "L1P", "D1P", "S1P",
            ],
        ),
        (
            GnssSystem::Galileo,
            'E',
            11,
            [
                "C1X", "C5X", "C7X", "C8X", "L1X", "L5X", "C6X", "L7X", "L8X", "D1X", "D5X", "S1X",
                "S5X", "D7X", "S7X", "L6X", "D8X", "S8X", "D6X", "S6X",
            ],
        ),
    ];
    let mut rng = SweepRng(0x2545_F491_4F6C_DD1D);
    for version in [2.11, 2.12] {
        let lists: Vec<SmallList> = alphabets
            .iter()
            .map(|(system, letter, prn, codes)| {
                let mut list: Vec<String> = codes.iter().map(|code| (*code).to_string()).collect();
                for at in (1..list.len()).rev() {
                    list.swap(at, rng.below(at + 1));
                }
                (*system, *letter, *prn, list)
            })
            .collect();
        let label = format!("{version} shuffled");
        super::write::LAST_ORDER_SEARCH.with(|last| last.set((0, false, 0)));
        assert_downgrade_states(&label, version, &mixed_product(version, &lists));
        let (placements, exhausted, _) = super::write::LAST_ORDER_SEARCH.with(std::cell::Cell::get);
        assert!(
            !exhausted,
            "{label}: the order search stopped at its budget after {placements} placements"
        );
    }
}

#[test]
fn downgrade_keeps_the_best_order_its_search_has_solved() {
    // Review found the search, stopped at its budget, returning an order with 24
    // moves while an assignment it had already solved gave a valid order with
    // 10: feasible orders waiting behind higher bounds were dropped.
    let codes =
        "C1 C5 C6 C7 L5X C8 L5 L6 L7 L8 L7X D5 C1X C5X C6X C7X C8X L1X L1 D1 L6X L8X D1X D5X"
            .split_whitespace()
            .map(str::to_string)
            .collect();
    let product = mixed_product(2.11, &[(GnssSystem::Galileo, 'E', 11, codes)]);
    let (_, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    let moves = changes
        .iter()
        .filter(|change| matches!(change, ObsDowngradeChange::CodeMoved { .. }))
        .count();
    assert!(moves <= 10, "{moves} moves: {changes:?}");
    assert_downgrade_states("Galileo literals", 2.11, &product);
}

/// Each constellation's names kept as written followed by the codes they stand
/// for: GPS and GLONASS on every kind and band they have, Galileo on four
/// kinds and five bands.
fn names_before_their_codes() -> Vec<SmallList> {
    let list = |system: GnssSystem, kinds: &[char], bands: &str| -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for &kind in kinds {
            for band in bands.chars() {
                if kind == 'P' && band != '1' && band != '2' {
                    continue;
                }
                names.push(format!("{kind}{band}"));
            }
        }
        let canonical: Vec<String> = names
            .iter()
            .map(|name| canonical_rinex2_obs_code(system, name, 2.11))
            .collect();
        names.extend(canonical);
        names
    };
    vec![
        (
            GnssSystem::Gps,
            'G',
            1,
            list(GnssSystem::Gps, &['C', 'P', 'L', 'D', 'S'], "125"),
        ),
        (
            GnssSystem::Glonass,
            'R',
            2,
            list(GnssSystem::Glonass, &['C', 'P', 'L', 'D', 'S'], "123"),
        ),
        (
            GnssSystem::Galileo,
            'E',
            11,
            list(GnssSystem::Galileo, &['C', 'L', 'D', 'S'], "15678"),
        ),
    ]
}

fn downgrade_moves(product: &RinexObs, version: f64) -> usize {
    let (_, changes) = product.downgrade_to_rinex2(version).expect("downgrade");
    changes
        .iter()
        .filter(|change| matches!(change, ObsDowngradeChange::CodeMoved { .. }))
        .count()
}

#[test]
fn downgrade_improves_orders_where_its_search_cannot_finish() {
    // Every name kept as written wants a position before the column it has to
    // follow, so the search cannot prove an order best within its work. Review
    // found valid orders moving 58 codes here, and 52 with each list's first 28
    // codes rotated by four, where the downgrade reported 84 and 85.
    let lists = names_before_their_codes();
    let product = mixed_product(2.11, &lists);
    let moves = downgrade_moves(&product, 2.11);
    assert!(moves <= 58, "{moves} moves");
    assert_downgrade_states("names before their codes", 2.11, &product);

    let rotated: Vec<SmallList> = lists
        .into_iter()
        .map(|(system, letter, prn, mut list)| {
            list[..28].rotate_left(4);
            (system, letter, prn, list)
        })
        .collect();
    let product = mixed_product(2.11, &rotated);
    let moves = downgrade_moves(&product, 2.11);
    assert!(moves <= 52, "rotated: {moves} moves");
    assert_downgrade_states("rotated names before their codes", 2.11, &product);
}

#[test]
fn downgrade_bounds_its_work_on_the_widest_header() {
    // 500 GPS and 499 Galileo copies lay out as 999 columns. The assignment
    // method alone took more than a second there, whatever the search's budget.
    let product = mixed_product(
        2.11,
        &[
            (GnssSystem::Gps, 'G', 1, vec!["C1C".to_string(); 500]),
            (GnssSystem::Galileo, 'E', 11, vec!["L1X".to_string(); 499]),
        ],
    );
    super::write::LAST_ORDER_SEARCH.with(|last| last.set((0, false, 0)));
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    // The first assignment alone would examine far more candidates than the
    // search may; the search stops within its work and says it did not finish.
    let (_, unfinished, spent) = super::write::LAST_ORDER_SEARCH.with(std::cell::Cell::get);
    assert!(unfinished);
    assert!(
        spent <= super::write::LAYOUT_ORDER_SEARCH_WORK,
        "{spent} steps"
    );
    // No order keeps more than one code at each of the 500 positions both
    // constellations' codes held, so 499 moves is the fewest possible.
    let moves = changes
        .iter()
        .filter(|change| matches!(change, ObsDowngradeChange::CodeMoved { .. }))
        .count();
    assert_eq!(moves, 499, "{changes:?}");
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.header().obs_codes[&GnssSystem::Gps].len(), 999);
}

#[test]
fn downgrade_refuses_a_layout_wider_than_a_header_before_searching_it() {
    // Seven constellations of 999 copies each fit their own lists but lay out
    // as 6,993 columns. The order search used to build matrices that wide, for
    // seconds and most of a gigabyte, before the width was refused.
    let lists: Vec<SmallList> = [
        (GnssSystem::Gps, 'G', 1, "C1C"),
        (GnssSystem::Glonass, 'R', 2, "L1C"),
        (GnssSystem::Galileo, 'E', 11, "D1X"),
        (GnssSystem::BeiDou, 'C', 5, "S2I"),
        (GnssSystem::Qzss, 'J', 1, "C5X"),
        (GnssSystem::Sbas, 'S', 20, "L5X"),
        (GnssSystem::Navic, 'I', 3, "D5A"),
    ]
    .into_iter()
    .map(|(system, letter, prn, code)| (system, letter, prn, vec![code.to_string(); 999]))
    .collect();
    let product = mixed_product(2.11, &lists);
    super::write::LAST_ORDER_SEARCH.with(|last| last.set((0, false, 0)));
    assert!(matches!(
        product.downgrade_to_rinex2(2.11),
        Err(RinexObsWriteError::TooManyObservationTypes { .. })
    ));
    assert_eq!(
        super::write::LAST_ORDER_SEARCH.with(std::cell::Cell::get),
        (0, false, 0),
        "the order search ran"
    );
}

#[test]
fn downgrade_keeps_a_valid_assignment_its_search_ran_out_of_work_to_check() {
    // GPS and GLONASS hold 329 and 317 names kept as written, sharing two. The
    // first assignment keeps 329 codes in place and keeps every reading rule,
    // but checking it when it was solved ran out of work, so it was dropped
    // and the emptied search reported the built order, moving every code, as
    // proven best. Placing all of GPS's names first and GLONASS's after moves
    // 317.
    let lists: Vec<Vec<usize>> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/downgrade_order_proof_lists.json"
    )))
    .expect("the committed lists");
    let names: Vec<String> = "0123456789ABEFGHIJKMNOQRTUVWXYZ"
        .chars()
        .flat_map(|first| {
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                .chars()
                .map(move |second| format!("{first}{second}"))
        })
        .take(644)
        .collect();
    let list =
        |at: usize| -> Vec<String> { lists[at].iter().map(|&name| names[name].clone()).collect() };
    let product = mixed_product(
        2.11,
        &[
            (GnssSystem::Gps, 'G', 1, list(0)),
            (GnssSystem::Glonass, 'R', 2, list(1)),
        ],
    );
    let moves = downgrade_moves(&product, 2.11);
    assert!(moves <= 317, "{moves} moves");
    assert_downgrade_states("names kept as written", 2.11, &product);
}

/// The layout's columns as they were built before construction was indexed:
/// every question about what a constellation reads answered by reading the
/// names again. Kept as the reference the indexed construction has to match.
fn reference_rinex2_obs_columns(
    product: &RinexObs,
) -> (Vec<String>, BTreeMap<GnssSystem, Vec<Option<usize>>>) {
    let version = product.header.version;
    let lists = &product.header.obs_codes;
    let reads_as = |system: GnssSystem, name: &str| -> Option<String> {
        rinex2_name_allowed(system, name, version)
            .then(|| canonical_rinex2_obs_code(system, name, version))
    };
    let covers = |names: &[String], system: GnssSystem, code: &str| {
        names
            .iter()
            .any(|name| reads_as(system, name).as_deref() == Some(code))
    };
    let mut providers: Vec<String> = Vec::new();
    for (system, codes) in lists {
        for (index, code) in codes.iter().enumerate() {
            if rinex2_kept_as_written(code)
                || codes[..index].contains(code)
                || covers(&providers, *system, code)
            {
                continue;
            }
            let mut best: Option<(String, usize)> = None;
            for name in rinex2_obs_code_candidates(*system, code, version) {
                if reads_as(*system, &name).as_deref() != Some(code.as_str()) {
                    continue;
                }
                let gain = lists
                    .iter()
                    .filter(|(other, other_codes)| {
                        reads_as(**other, &name).is_some_and(|read| {
                            other_codes.contains(&read) && !covers(&providers, **other, &read)
                        })
                    })
                    .count();
                if best.as_ref().is_none_or(|(_, most)| gain > *most) {
                    best = Some((name, gain));
                }
            }
            if let Some((name, _)) = best {
                providers.push(name);
            }
        }
    }
    let mut kept_as_written: Vec<String> = Vec::new();
    for (system, codes) in lists {
        let mut wanted: BTreeMap<&str, usize> = BTreeMap::new();
        for code in codes.iter().filter(|code| rinex2_kept_as_written(code)) {
            *wanted.entry(code.as_str()).or_default() += 1;
        }
        for (name, count) in wanted {
            if let Some(stands_for) = reads_as(*system, name) {
                if !covers(&providers, *system, &stands_for) {
                    providers.push(name.to_string());
                }
            }
            let mut columns = providers.clone();
            columns.extend(kept_as_written.iter().cloned());
            let have = rinex2_system_obs_codes(*system, &columns, version)
                .iter()
                .filter(|read| read.as_str() == name)
                .count();
            for _ in have..count {
                kept_as_written.push(name.to_string());
            }
        }
    }
    let mut names = providers;
    names.extend(kept_as_written);
    let mut slots: BTreeMap<GnssSystem, Vec<Option<usize>>> = BTreeMap::new();
    let mut renamed: Vec<(GnssSystem, usize)> = Vec::new();
    for (system, codes) in lists {
        let read = rinex2_system_obs_codes(*system, &names, version);
        let mut row: Vec<Option<usize>> = vec![None; names.len()];
        for (index, code) in codes.iter().enumerate() {
            match (0..read.len()).find(|&column| read[column] == *code && row[column].is_none()) {
                Some(column) => row[column] = Some(index),
                None => renamed.push((*system, index)),
            }
        }
        slots.insert(*system, row);
    }
    for (system, index) in renamed {
        if names.len() > MAX_OBS_TYPE_COUNT {
            break;
        }
        let code = &lists[&system][index];
        let spelling = rinex2_obs_code_candidates(system, code, version)
            .into_iter()
            .next()
            .unwrap_or_else(|| code.chars().take(2).collect());
        let duplicate = lists[&system][..index].contains(code);
        let read = rinex2_system_obs_codes(system, &names, version);
        let free = |column: usize| slots[&system].get(column).copied().flatten().is_none();
        let target = match reads_as(system, &spelling) {
            Some(target) if !duplicate && !lists[&system].contains(&target) => {
                (0..read.len()).find(|&column| free(column) && read[column] == target)
            }
            _ => None,
        };
        let same_name = || {
            let mut appended = names.clone();
            appended.push(spelling.clone());
            let would_read = rinex2_system_obs_codes(system, &appended, version)
                .pop()
                .unwrap_or_default();
            (0..read.len()).find(|&column| {
                free(column) && names[column] == spelling && read[column] == would_read
            })
        };
        let column = match target.or_else(same_name) {
            Some(column) => column,
            None => {
                names.push(spelling);
                for row in slots.values_mut() {
                    row.resize(names.len(), None);
                }
                names.len() - 1
            }
        };
        let row = slots.entry(system).or_default();
        row.resize(names.len(), None);
        row[column] = Some(index);
    }
    for row in slots.values_mut() {
        row.resize(names.len(), None);
    }
    (names, slots)
}

#[test]
fn indexed_layout_columns_match_reading_the_names_again() {
    // The layout's columns used to be built by reading the names again for
    // every question, which was quadratic in the width; they are built from
    // indexes now, and have to come out the same, column for column.
    let check = |label: &str, product: &RinexObs| {
        assert_eq!(
            product.rinex2_obs_columns(),
            reference_rinex2_obs_columns(product),
            "{label}"
        );
    };
    let mut products = 0_usize;
    for version in [2.11, 2.12] {
        for (sa, la, pa, sb, lb, pb) in SMALL_PAIRS {
            for list_a in small_code_lists(sa) {
                for list_b in small_code_lists(sb) {
                    let product = two_system_product(
                        version,
                        &(sa, la, pa, list_a.clone()),
                        &(sb, lb, pb, list_b.clone()),
                        true,
                    );
                    check(&format!("{version} {list_a:?} {list_b:?}"), &product);
                    products += 1;
                }
            }
        }
    }
    let constellations = sweep_constellations();
    let literal_heavy = [
        "C1", "C2", "P1", "P2", "L1", "L2", "D1", "S1", "CA", "CB", "C5", "L5", "ZZ", "X1", "C1C",
        "C1W", "L1C", "C5X", "C9X", "C2", "C7",
    ];
    let mut rng = SweepRng(0x5851_F42D_4C95_7F2D);
    for round in 0..3_000 {
        let version = if round % 2 == 0 { 2.11 } else { 2.12 };
        let count = 2 + rng.below(3);
        let mut picked: Vec<usize> = Vec::new();
        while picked.len() < count {
            let at = rng.below(constellations.len());
            if !picked.contains(&at) {
                picked.push(at);
            }
        }
        let lists: Vec<SmallList> = picked
            .iter()
            .map(|&at| {
                let (system, letter, prn, alphabet) = &constellations[at];
                let length = 1 + rng.below(8);
                let list = (0..length)
                    .map(|_| {
                        if round % 3 == 0 {
                            literal_heavy[rng.below(literal_heavy.len())].to_string()
                        } else {
                            alphabet[rng.below(alphabet.len())].to_string()
                        }
                    })
                    .collect();
                (*system, *letter, *prn, list)
            })
            .collect();
        check(
            &format!("{version} {lists:?}"),
            &mixed_product(version, &lists),
        );
        products += 1;
    }
    let names = names_before_their_codes();
    check("names before their codes", &mixed_product(2.11, &names));
    let proof: Vec<Vec<usize>> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/downgrade_order_proof_lists.json"
    )))
    .expect("the committed lists");
    let two_character: Vec<String> = "0123456789ABEFGHIJKMNOQRTUVWXYZ"
        .chars()
        .flat_map(|first| {
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                .chars()
                .map(move |second| format!("{first}{second}"))
        })
        .take(644)
        .collect();
    let list = |at: usize| -> Vec<String> {
        proof[at]
            .iter()
            .map(|&name| two_character[name].clone())
            .collect()
    };
    check(
        "names kept as written",
        &mixed_product(
            2.11,
            &[
                (GnssSystem::Gps, 'G', 1, list(0)),
                (GnssSystem::Glonass, 'R', 2, list(1)),
            ],
        ),
    );
    assert!(products > 3_000, "{products} products");
}

#[test]
fn downgrade_lays_out_seven_constellations_of_shuffled_names_kept_as_written() {
    // Seven constellations each holding the same 999 two-character names kept
    // as written, shuffled differently. Building the columns for them read
    // every name again for every distinct name, over a second in a release
    // build before the search began.
    let names: Vec<String> = "0123456789ABEFGHIJKMNOQRTUVWXYZ"
        .chars()
        .flat_map(|first| {
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                .chars()
                .map(move |second| format!("{first}{second}"))
        })
        .take(999)
        .collect();
    let mut rng = SweepRng(7);
    let lists: Vec<SmallList> = [
        (GnssSystem::Gps, 'G', 1_u8),
        (GnssSystem::Glonass, 'R', 3),
        (GnssSystem::Galileo, 'E', 5),
        (GnssSystem::BeiDou, 'C', 7),
        (GnssSystem::Qzss, 'J', 1),
        (GnssSystem::Sbas, 'S', 20),
        (GnssSystem::Navic, 'I', 3),
    ]
    .into_iter()
    .map(|(system, letter, prn)| {
        let mut codes = names.clone();
        for at in (1..codes.len()).rev() {
            codes.swap(at, rng.below(at + 1));
        }
        (system, letter, prn, codes)
    })
    .collect();
    let product = mixed_product(2.11, &lists);
    let (downgraded, _) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.header().obs_codes[&GnssSystem::Gps].len(), 999);
}

#[test]
fn a_wide_version_two_product_of_repeated_names_writes_unchanged() {
    // Seven constellations each reading 499 `ZZ` then 500 `C1`: a version 2
    // list states it, so the writer writes it and the downgrade changes
    // nothing. Finding that list asked, for every repeated `C1`, whether the
    // codes before it held its canonical code by scanning all of them.
    // Each constellation holds what the reader gives it for those names.
    let mut names = vec!["ZZ".to_string(); 499];
    names.extend(vec!["C1".to_string(); 500]);
    let codes = |system: GnssSystem| rinex2_system_obs_codes(system, &names, 2.11);
    let lists: Vec<SmallList> = [
        (GnssSystem::Gps, 'G', 1_u8),
        (GnssSystem::Glonass, 'R', 2),
        (GnssSystem::Galileo, 'E', 11),
        (GnssSystem::BeiDou, 'C', 5),
        (GnssSystem::Qzss, 'J', 1),
        (GnssSystem::Sbas, 'S', 20),
        (GnssSystem::Navic, 'I', 3),
    ]
    .into_iter()
    .map(|(system, letter, prn)| (system, letter, prn, codes(system)))
    .collect();
    let product = mixed_product(2.11, &lists);
    let text = product
        .to_rinex_string()
        .expect("a version 2 list states it");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.header().obs_codes, product.header().obs_codes);
    let (downgraded, changes) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    assert!(changes.is_empty(), "{changes:?}");
    assert_eq!(
        downgraded.to_rinex_string().expect("the downgrade writes"),
        text
    );
}

#[test]
fn a_version_two_file_with_a_one_character_type_writes_back() {
    // A type field is two characters wide and a file may put one character in
    // it. The reader keeps `Z` as written and the writers carry it, but the
    // search for a name list only took two-byte names as kept as written, so
    // the file could neither be written back nor downgraded.
    let line = |content: &str, label: &str| format!("{content:<60}{label}\n");
    let text = [
        line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        line("     2     Z    C1", "# / TYPES OF OBSERV"),
        line(
            "  2015     1     1     0     0    0.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
        line("", "END OF HEADER"),
        " 15  1  1  0  0  0.0000000  0  1G 1\n".to_string(),
        "      1234.56715  20000000.125 7\n".to_string(),
    ]
    .concat();
    let obs = RinexObs::parse(&text).expect("parse a one-character type");
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    assert_eq!(obs.header().obs_codes[&GnssSystem::Gps][0], "Z");
    assert_eq!(obs.epochs()[0].sats[&gps][0].lli, Some(1));
    let written = obs.to_rinex_string().expect("a one-character type writes");
    let read = RinexObs::parse(&written).expect("reads back");
    assert_eq!(read.header().obs_codes, obs.header().obs_codes);
    assert_eq!(read.epochs(), obs.epochs());
    let (downgraded, changes) = obs.downgrade_to_rinex2(2.11).expect("downgrade");
    assert!(changes.is_empty(), "{changes:?}");
    assert_eq!(downgraded.to_rinex_string().expect("writes"), written);
}

#[test]
fn a_one_character_type_is_kept_whichever_side_of_its_field_it_sits() {
    // An `A2` type field holding one character may put it in either column;
    // both read as the same name kept as written and write back.
    for field in [" Z", "Z "] {
        let line = |content: &str, label: &str| format!("{content:<60}{label}\n");
        let text = [
            line(
                "     2.11           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            line(&format!("     1    {field}"), "# / TYPES OF OBSERV"),
            line("", "END OF HEADER"),
            " 15  1  1  0  0  0.0000000  0  1G 1\n".to_string(),
            "      1234.567 1\n".to_string(),
        ]
        .concat();
        let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("{field:?}: {e}"));
        assert_eq!(obs.header().obs_codes[&GnssSystem::Gps], ["Z"], "{field:?}");
        let written = obs.to_rinex_string().expect("writes back");
        let read = RinexObs::parse(&written).expect("reads back");
        assert_eq!(read.epochs(), obs.epochs(), "{field:?}");
    }
}

#[test]
fn a_version_two_file_keeps_its_own_type_list_over_version_three_type_records() {
    // Version 2 lays out its observation records by `# / TYPES OF OBSERV`, and
    // `SYS / # / OBS TYPES` is a version 3 record. Its codes used to replace a
    // version 2 file's own list, so a file whose two records disagreed read its
    // observations against the wrong list. The version 3 record is reported as
    // not retained, before or after the version record.
    let line = |content: &str, label: &str| format!("{content:<60}{label}\n");
    let version = line(
        "     2.11           OBSERVATION DATA    G (GPS)",
        "RINEX VERSION / TYPE",
    );
    let types = line("     2    C1    L1", "# / TYPES OF OBSERV");
    let sys = line("G    1 ZZZ", "SYS / # / OBS TYPES");
    let body = [
        line("", "END OF HEADER"),
        " 15  1  1  0  0  0.0000000  0  1G 1\n".to_string(),
        format!("{:14.3}  {:14.3}1\n", 20_000_000.125, 1234.567),
    ]
    .concat();
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let expected =
        rinex2_system_obs_codes(GnssSystem::Gps, &["C1".to_string(), "L1".to_string()], 2.11);
    for text in [
        [version.clone(), types.clone(), sys.clone(), body.clone()].concat(),
        [sys.clone(), version.clone(), types.clone(), body.clone()].concat(),
    ] {
        let obs = RinexObs::parse(&text).expect("parse");
        assert_eq!(obs.header().obs_codes[&GnssSystem::Gps], expected);
        assert!(
            obs.header()
                .unretained_header_labels
                .iter()
                .any(|label| label == "SYS / # / OBS TYPES"),
            "{:?}",
            obs.header().unretained_header_labels
        );
        let values = &obs.epochs()[0].sats[&gps];
        assert_eq!(values[0].value, Some(20_000_000.125));
        assert_eq!(values[1].value, Some(1234.567));
        assert_eq!(values[1].lli, Some(1));
        let written = obs.to_rinex_string().expect("writes back");
        let read = RinexObs::parse(&written).expect("reads back");
        assert_eq!(read.epochs(), obs.epochs());
    }
}

/// A version 2 header record: content padded to its label.
fn v2_record(content: &str, label: &str) -> String {
    format!("{content:<60}{label}\n")
}

#[test]
fn version_two_files_holding_what_version_two_cannot_are_refused_downgraded_or_written_as_they_should(
) {
    // Review built these: version 2 headers the reader takes with version 3
    // type records, epoch records, scale factors, repeated declarations and
    // counts between them. Each is written, refused, or downgraded according
    // to what it holds, never by accident of which record came first.
    let version2 = v2_record(
        "     2.11           OBSERVATION DATA    G (GPS)",
        "RINEX VERSION / TYPE",
    );
    let end = v2_record("", "END OF HEADER");
    let z_types = v2_record("     1     Z", "# / TYPES OF OBSERV");
    let z_sys = v2_record("G    1 Z", "SYS / # / OBS TYPES");
    let v2_epoch = " 15  1  1  0  0  0.0000000  0  1G 1\n      1234.567\n";
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let parse = |label: &str, text: String| {
        let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(obs.skipped_records, 0, "{label}");
        obs
    };
    let writes_back = |label: &str, obs: &RinexObs| {
        let text = obs
            .to_rinex_string()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let read = RinexObs::parse(&text).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(read.epochs(), obs.epochs(), "{label}");
        assert_eq!(
            read.header().prn_obs_counts,
            obs.header().prn_obs_counts,
            "{label}"
        );
    };

    let native = parse(
        "native",
        [
            version2.clone(),
            z_types.clone(),
            end.clone(),
            v2_epoch.to_string(),
        ]
        .concat(),
    );
    writes_back("native", &native);

    let scale = parse(
        "scale",
        [
            version2.clone(),
            z_types.clone(),
            v2_record("G   10  0", "SYS / SCALE FACTOR"),
            end.clone(),
            v2_epoch.to_string(),
        ]
        .concat(),
    );
    assert!(matches!(
        scale.to_rinex_string(),
        Err(RinexObsWriteError::ScaleFactorsInVersionTwo { count: 1 })
    ));
    // The file's 1234.567 under a factor of 10 is 123.4567, which a field with
    // no factor to divide by holds only to three decimals.
    let (downgraded, changes) = scale.downgrade_to_rinex2(2.11).expect("scale downgrade");
    assert_eq!(
        changes,
        vec![
            ObsDowngradeChange::ScaleFactorsRemoved { count: 1 },
            ObsDowngradeChange::ValueRounded {
                epoch_index: 0,
                satellite: gps,
                code: "Z".to_string(),
                from: 123.4567,
                to: 123.457,
            },
        ]
    );
    writes_back("scale downgraded", &downgraded);

    let clock = parse(
        "clock",
        [
            version2.clone(),
            z_types.clone(),
            z_sys.clone(),
            end.clone(),
            "> 2020 06 24 00 00  0.0000000  0  1      -0.000000000001\nG01      1234.567\n"
                .to_string(),
        ]
        .concat(),
    );
    assert_eq!(clock.header().obs_codes[&GnssSystem::Gps], ["Z"]);
    assert!(clock.to_rinex_string().is_err());
    let (downgraded, changes) = clock.downgrade_to_rinex2(2.11).expect("clock downgrade");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::ClockOffsetRounded {
            epoch_index: 0,
            from: -0.000_000_000_001,
            to: 0.0,
        }]
    );
    writes_back("clock downgraded", &downgraded);

    let pico = parse(
        "pico",
        [
            version2.clone(),
            z_types.clone(),
            z_sys.clone(),
            end.clone(),
            "> 2020 06 24 00 00  0.0000000 12345  0  1\nG01      1234.567\n".to_string(),
        ]
        .concat(),
    );
    assert!(pico.to_rinex_string().is_err());
    let (downgraded, changes) = pico.downgrade_to_rinex2(2.11).expect("pico downgrade");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::EpochPicosecondsRemoved {
            epoch_index: 0,
            picoseconds: 12345,
        }]
    );
    writes_back("pico downgraded", &downgraded);

    // A year outside the two-digit window has no version 2 spelling, with or
    // without scale factors to remove first.
    for (label, extra) in [
        ("year", String::new()),
        ("scale-year", v2_record("G   10  0", "SYS / SCALE FACTOR")),
    ] {
        let obs = parse(
            label,
            [
                version2.clone(),
                z_types.clone(),
                z_sys.clone(),
                extra,
                end.clone(),
                "> 2080 01 01 00 00  0.0000000  0  1\nG01      1234.567\n".to_string(),
            ]
            .concat(),
        );
        assert!(obs.to_rinex_string().is_err(), "{label}");
        assert!(obs.downgrade_to_rinex2(2.11).is_err(), "{label}");
    }

    // Counts are read against the lists the header ends with.
    let counts = parse(
        "counts",
        [
            version2.clone(),
            v2_record("     2     Z     Y", "# / TYPES OF OBSERV"),
            v2_record("   G01     1     2", "PRN / # OF OBS"),
            z_types.clone(),
            end.clone(),
            v2_epoch.to_string(),
        ]
        .concat(),
    );
    assert_eq!(counts.header().prn_obs_counts[&gps], vec![Some(1)]);
    writes_back("counts", &counts);
    let partial = parse(
        "v3-partial-counts",
        [
            v2_record(
                "     3.05           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            v2_record("G    1 C1C", "SYS / # / OBS TYPES"),
            v2_record("   G01     1", "PRN / # OF OBS"),
            v2_record("G    1 L1C", "SYS / # / OBS TYPES"),
            end.clone(),
        ]
        .concat(),
    );
    assert_eq!(partial.header().prn_obs_counts[&gps], vec![Some(1), None]);
    writes_back("v3-partial-counts", &partial);

    // A version 3 type record does not replace a version 2 file's own list.
    let c1 = rinex2_system_obs_codes(GnssSystem::Gps, &["C1".to_string()], 2.11);
    for (label, code) in [("signal-override", "C5X"), ("two-char-override", "C1")] {
        let obs = parse(
            label,
            [
                version2.clone(),
                v2_record("     1    C1", "# / TYPES OF OBSERV"),
                v2_record(&format!("G    1 {code}"), "SYS / # / OBS TYPES"),
                end.clone(),
                v2_epoch.to_string(),
            ]
            .concat(),
        );
        assert_eq!(obs.header().obs_codes[&GnssSystem::Gps], c1, "{label}");
        writes_back(label, &obs);
    }
}

#[test]
fn version_two_headers_review_found_the_reader_took_without_what_writing_needs() {
    // Review built these: a flag too wide for its field, counts for a
    // constellation no observation names, and observations with no type list.
    let mixed = v2_record(
        "     2.11           OBSERVATION DATA    M (MIXED)",
        "RINEX VERSION / TYPE",
    );
    let end = v2_record("", "END OF HEADER");
    let c1 = v2_record("     1    C1", "# / TYPES OF OBSERV");
    let r01_count = v2_record("   R01     1", "PRN / # OF OBS");
    let r01 = GnssSatelliteId {
        system: GnssSystem::Glonass,
        prn: 1,
    };

    // A flag of 10 does not fit the one-digit field either version writes it
    // in. The reader keeps the reading it has always given such a line, and
    // the writer and downgrade refuse it rather than write it across the count.
    let flag_ten = [
        v2_record(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        v2_record("     1     Z", "# / TYPES OF OBSERV"),
        end.clone(),
        "> 2020 06 24 00 00  0.0000000  10  1\nEvent record\n".to_string(),
    ]
    .concat();
    let obs = RinexObs::parse(&flag_ten).expect("a flag of 10 is read");
    assert_eq!(obs.epochs()[0].flag, 10);
    assert!(obs.to_rinex_string().is_err());
    assert!(obs.downgrade_to_rinex2(2.11).is_err());

    // Counts for a constellation no observation names are read against the
    // file's list, with no epochs at all and with only an event epoch: they
    // write back, and the downgrade measures them against that list rather
    // than refusing them as counts with no codes.
    for (label, body) in [
        ("header only", String::new()),
        (
            "event only",
            " 15  1  1  0  0  0.0000000  4  0\n".to_string(),
        ),
    ] {
        let obs = RinexObs::parse(
            &[
                mixed.clone(),
                c1.clone(),
                r01_count.clone(),
                end.clone(),
                body,
            ]
            .concat(),
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(obs.header().prn_obs_counts[&r01], vec![Some(1)], "{label}");
        let text = obs
            .to_rinex_string()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let read = RinexObs::parse(&text).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(read.header(), obs.header(), "{label}");
        let (downgraded, changes) = obs
            .downgrade_to_rinex2(2.11)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(changes.is_empty(), "{label}: {changes:?}");
        assert_eq!(
            downgraded.to_rinex_string().expect("writes"),
            text,
            "{label}"
        );
    }

    // With a scale factor to remove, the downgrade has codes for those counts.
    let scaled = RinexObs::parse(
        &[
            mixed.clone(),
            c1.clone(),
            r01_count.clone(),
            v2_record("G   10  0", "SYS / SCALE FACTOR"),
            end.clone(),
        ]
        .concat(),
    )
    .expect("parse");
    assert!(matches!(
        scaled.to_rinex_string(),
        Err(RinexObsWriteError::ScaleFactorsInVersionTwo { count: 1 })
    ));
    let (downgraded, _) = scaled.downgrade_to_rinex2(2.11).expect("downgrade");
    assert_eq!(downgraded.header().prn_obs_counts[&r01], vec![Some(1)]);
    downgraded.to_rinex_string().expect("the downgrade writes");

    // Observations with no type list have nothing to be read by.
    let unlisted = [
        v2_record(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        end.clone(),
        "> 2020 06 24 00 00  0.0000000  0  1\nG01      1234.567\n".to_string(),
    ]
    .concat();
    let error = RinexObs::parse(&unlisted).expect_err("observations with no list are refused");
    assert!(
        error.to_string().contains("no # / TYPES OF OBSERV"),
        "{error}"
    );
}

#[test]
fn version_two_type_names_keep_what_counts_and_flags_mean() {
    // Review built these. A version 2 count for a constellation with no
    // observation reads by the file's type names, and the writer and downgrade
    // used to lose which names those were; a flag of 10 wrote at version 3.
    let mixed = v2_record(
        "     2.11           OBSERVATION DATA    M (MIXED)",
        "RINEX VERSION / TYPE",
    );
    let end = v2_record("", "END OF HEADER");

    // A GPS count beside only a BeiDou observation keeps meaning C1C: the
    // file is written with the name it was read with, not one that reads the
    // same for BeiDou and differently for GPS.
    let alias = RinexObs::parse(
        &[
            mixed.clone(),
            v2_record("     1    C1", "# / TYPES OF OBSERV"),
            v2_record("   G01     7", "PRN / # OF OBS"),
            end.clone(),
            "> 2020 06 24 00 00  0.0000000  0  1\nC01      1234.567\n".to_string(),
        ]
        .concat(),
    )
    .expect("parse");
    assert_eq!(alias.header().rinex2_types, ["C1"]);
    let written = alias.to_rinex_string().expect("writes");
    let read = RinexObs::parse(&written).expect("reads back");
    assert_eq!(read.header().rinex2_types, ["C1"]);
    assert_eq!(read.header(), alias.header());
    assert_eq!(
        rinex2_system_obs_codes(GnssSystem::Gps, &read.header().rinex2_types, 2.11),
        ["C1C"]
    );

    // A GLONASS count converted to 2.12 follows the code it counted: either
    // the names still read as C2C for GLONASS, or the rename is reported.
    let glonass = RinexObs::parse(
        &[
            mixed.clone(),
            v2_record("     1    C2", "# / TYPES OF OBSERV"),
            v2_record("   R01     7", "PRN / # OF OBS"),
            v2_record("  1 R01 -7", "GLONASS SLOT / FRQ #"),
            end.clone(),
        ]
        .concat(),
    )
    .expect("parse");
    let r01 = GnssSatelliteId {
        system: GnssSystem::Glonass,
        prn: 1,
    };
    let (converted, changes) = glonass.downgrade_to_rinex2(2.12).expect("downgrade");
    let reads =
        rinex2_system_obs_codes(GnssSystem::Glonass, &converted.header().rinex2_types, 2.12);
    let column = converted.header().prn_obs_counts[&r01]
        .iter()
        .position(|count| *count == Some(7))
        .expect("the count survives");
    assert!(
        reads[column] == "C2C"
            || changes.contains(&ObsDowngradeChange::CodeRenamed {
                system: GnssSystem::Glonass,
                from: "C2C".to_string(),
                to: reads[column].clone(),
            }),
        "the count now reads as {} with no rename reported: {changes:?}",
        reads[column]
    );
    let text = converted.to_rinex_string().expect("the downgrade writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(
        read.header().prn_obs_counts,
        converted.header().prn_obs_counts
    );

    // A flag of 10 does not fit at version 3 either.
    let wide = RinexObs::parse(
        &[
            v2_record(
                "     3.05           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            v2_record("G    1 C1C", "SYS / # / OBS TYPES"),
            end.clone(),
            "> 2020 06 24 00 00  0.0000000  10  1\nEvent record\n".to_string(),
        ]
        .concat(),
    )
    .expect("a flag of 10 is read");
    assert_eq!(
        wide.to_rinex_string(),
        Err(RinexObsWriteError::EpochFlagTooWide {
            epoch_index: 0,
            flag: 10,
        })
    );

    // Two records for one satellite are declared as the one kept.
    let twice = RinexObs::parse(
        &[
            mixed.clone(),
            v2_record("     1    C1", "# / TYPES OF OBSERV"),
            end.clone(),
            "> 2020 06 24 00 00  0.0000000  0  2\nG01      1234.567\nG01      1234.567\n"
                .to_string(),
        ]
        .concat(),
    )
    .expect("parse");
    let text = twice.to_rinex_string().expect("writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.epochs()[0].sats, twice.epochs()[0].sats);
    assert_eq!(read.epochs()[0].declared_record_count, 1);
}

#[test]
fn a_file_is_compared_with_what_its_text_states() {
    // Review built these. Version 3 epoch records have no picosecond field;
    // a version 2 file states a list for each constellation something names,
    // read from its type names, and nothing more.
    let end = v2_record("", "END OF HEADER");

    // Picoseconds at 3.05 would push the flag and count out of their columns.
    let pico = RinexObs::parse(
        &[
            v2_record(
                "     3.05           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            v2_record("G    1 C1C", "SYS / # / OBS TYPES"),
            end.clone(),
            "> 2020 06 24 00 00  0.0000000 12345  0  1\nG01      1234.567\n".to_string(),
        ]
        .concat(),
    )
    .expect("parse");
    assert_eq!(
        pico.to_rinex_string(),
        Err(RinexObsWriteError::EpochPicosecondsNotInVersion {
            epoch_index: 0,
            version: 3.05,
        })
    );

    // A product whose lists a caller cleared still has its type names, which
    // give GPS its C1C; the file states that list and reads back as it.
    let mixed = v2_record(
        "     2.11           OBSERVATION DATA    M (MIXED)",
        "RINEX VERSION / TYPE",
    );
    let mut cleared = RinexObs::parse(
        &[
            mixed.clone(),
            v2_record("     1    C1", "# / TYPES OF OBSERV"),
            v2_record("   G01     7", "PRN / # OF OBS"),
            end.clone(),
            " 15  1  1  0  0  0.0000000  0  1G01\n      1234.567\n".to_string(),
        ]
        .concat(),
    )
    .expect("parse");
    cleared.header.obs_codes.clear();
    let text = cleared
        .to_rinex_string()
        .expect("the type names state the list");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.epochs(), cleared.epochs());
    assert_eq!(
        read.header().prn_obs_counts,
        cleared.header().prn_obs_counts
    );
    assert_eq!(read.header().obs_codes[&GnssSystem::Gps], ["C1C"]);

    // Repair removes observations that hold nothing; the GLONASS list they
    // named then names nothing, and the header-only file states the list its
    // version record names.
    let empty = RinexObs::parse(
        &[
            mixed,
            v2_record("     1    C1", "# / TYPES OF OBSERV"),
            end,
            "> 2020 06 24 00 00  0.0000000  0  2\nG01\nR01\n".to_string(),
        ]
        .concat(),
    )
    .expect("parse");
    let native = RinexObs::parse(&empty.to_rinex_string().expect("writes")).expect("reads back");
    let options = crate::rinex_qc::RepairOptions {
        drop_empty_records: true,
        set_obs_counts: true,
        ..crate::rinex_qc::RepairOptions::default()
    };
    let repaired = crate::rinex_qc::repair_obs(&native, &options).repaired;
    let text = repaired
        .to_rinex_string()
        .expect("a repair that empties the body writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert!(read.epochs().iter().all(|epoch| epoch.sats.is_empty()));
}

/// A one-satellite observation file at `version`, of one constellation, with
/// these header records and body.
fn obs_file(version: &str, letter: char, headers: &[String], body: &str) -> String {
    let mut text = v2_record(
        &format!("{version:>9}           OBSERVATION DATA    {letter}"),
        "RINEX VERSION / TYPE",
    );
    for header in headers {
        text.push_str(header);
    }
    text.push_str(&v2_record("", "END OF HEADER"));
    text.push_str(body);
    text
}

#[test]
fn picoseconds_sit_where_rinex_4_02_puts_them() {
    // RINEX 4.02 added five more digits of the second after the clock offset,
    // `1X,I5.5`. They used to be written between the seconds and the flag,
    // which put the flag and count in the wrong columns for every other
    // reader, and a line carrying them where the format does was read with
    // them dropped. Earlier versions have no such field.
    let types = [v2_record("G    1 C1C", "SYS / # / OBS TYPES")];
    let old_placement = "> 2020 06 24 00 00  0.0000000 12345  0  1\nG01      1234.567\n";
    for version in ["3.05", "4.00", "4.01"] {
        let obs = RinexObs::parse(&obs_file(version, 'G', &types, old_placement)).expect("parse");
        assert_eq!(obs.epochs()[0].epoch_picoseconds, Some(12345), "{version}");
        assert_eq!(
            obs.to_rinex_string(),
            Err(RinexObsWriteError::EpochPicosecondsNotInVersion {
                epoch_index: 0,
                version: obs.header().version,
            }),
            "{version}"
        );
    }

    // At 4.02 the old placement is still read, and written after the clock,
    // whose columns stay blank with no offset.
    let obs = RinexObs::parse(&obs_file("4.02", 'G', &types, old_placement)).expect("parse");
    let text = obs.to_rinex_string().expect("writes at 4.02");
    let epoch_line = text
        .lines()
        .find(|line| line.starts_with('>'))
        .expect("epoch");
    assert_eq!(
        epoch_line,
        format!(
            "> 2020 06 24 00 00  0.0000000  0  1{} 12345",
            " ".repeat(21)
        )
    );
    assert_eq!(RinexObs::parse(&text).expect("reads back"), obs);

    // Where 4.02 puts them, with a clock offset, positive or negative, the
    // line is read with them and written back as it was.
    for (clock, picoseconds) in [(0.125, 12345_u32), (-0.000_000_000_001, 45)] {
        let line =
            format!("> 2020 06 24 00 00  0.0000000  0  1      {clock:15.12} {picoseconds:05}");
        let obs = RinexObs::parse(&obs_file(
            "4.02",
            'G',
            &types,
            &format!("{line}\nG01      1234.567\n"),
        ))
        .expect("parse");
        let epoch = &obs.epochs()[0];
        assert_eq!(epoch.epoch_picoseconds, Some(picoseconds), "{line}");
        assert_eq!(epoch.rcv_clock_offset_s, Some(clock), "{line}");
        let text = obs.to_rinex_string().expect("writes");
        assert!(text.contains(&line), "{text}");
        assert_eq!(RinexObs::parse(&text).expect("reads back"), obs);
    }
}

#[test]
fn version_two_files_are_compared_with_what_a_reader_builds_from_them() {
    // Review built these: a product with no lists left, a repair that empties
    // the body, and a header with counts only for another constellation.
    let c1 = [v2_record("     1    C1", "# / TYPES OF OBSERV")];

    // Header-only and events-only products whose lists were cleared write
    // their retained type names.
    for (label, body) in [
        ("header only", ""),
        ("events only", " 15  1  1  0  0  0.0000000  4  0\n"),
    ] {
        let mut obs = RinexObs::parse(&obs_file("2.11", 'G', &c1, body)).expect("parse");
        obs.header.obs_codes.clear();
        let text = obs
            .to_rinex_string()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let read = RinexObs::parse(&text).expect("reads back");
        assert_eq!(read.header().rinex2_types, ["C1"], "{label}");
        assert_eq!(
            read.header().obs_codes[&GnssSystem::Gps],
            ["C1C"],
            "{label}"
        );
    }

    // A repair that empties the body, written and repaired again, gives the
    // same bytes: the version record names the list the file states, not the
    // lists the product held before.
    let empty = RinexObs::parse(&obs_file(
        "2.11",
        'M',
        &c1,
        " 20  6 24  0  0  0.0000000  0  2G 1R 1\n\n\n",
    ))
    .expect("parse");
    let options = crate::rinex_qc::RepairOptions {
        set_interval: true,
        set_time_of_last_obs: true,
        set_obs_counts: true,
        drop_empty_records: true,
        drop_unsupported: true,
        ..crate::rinex_qc::RepairOptions::default()
    };
    let first = crate::rinex_qc::repair_obs(&empty, &options)
        .repaired
        .to_rinex_string()
        .expect("the repair writes");
    let again =
        crate::rinex_qc::repair_obs(&RinexObs::parse(&first).expect("reads back"), &options)
            .repaired
            .to_rinex_string()
            .expect("the repair writes again");
    assert_eq!(again, first);

    // A GPS header with counts only for GLONASS reads with the GPS list its
    // version record names, and writes back to the same header.
    let counts_only = RinexObs::parse(&obs_file(
        "2.11",
        'G',
        &[c1[0].clone(), v2_record("   R01     1", "PRN / # OF OBS")],
        "",
    ))
    .expect("parse");
    assert_eq!(counts_only.header().obs_codes[&GnssSystem::Gps], ["C1C"]);
    let text = counts_only.to_rinex_string().expect("writes");
    assert_eq!(
        RinexObs::parse(&text).expect("reads back").header(),
        counts_only.header()
    );
}

#[test]
fn picoseconds_are_read_after_the_clock_however_the_line_is_spaced() {
    // Review built these. A tab after correctly placed picoseconds took the
    // line out of its layout, and the looser reading then took the digits for
    // a clock offset, or dropped them after a real one.
    let types = [v2_record("G    1 C1C", "SYS / # / OBS TYPES")];
    let observation = "\nG01      1234.567\n";
    for (epoch_line, clock) in [
        (
            format!(
                "> 2020 06 24 00 00  0.0000000  0  1{}00001\t",
                " ".repeat(22)
            ),
            None,
        ),
        (
            format!(
                "> 2020 06 24 00 00  0.0000000  0  1      {:15.12} 00001\t",
                0.125
            ),
            Some(0.125),
        ),
        (
            "> 2020 06 24 00 00 0.0000000 0 1 0.125 00001".to_string(),
            Some(0.125),
        ),
    ] {
        let obs = RinexObs::parse(&obs_file(
            "4.02",
            'G',
            &types,
            &format!("{epoch_line}{observation}"),
        ))
        .unwrap_or_else(|error| panic!("{epoch_line:?}: {error}"));
        let epoch = &obs.epochs()[0];
        assert_eq!(epoch.epoch_picoseconds, Some(1), "{epoch_line:?}");
        assert_eq!(epoch.rcv_clock_offset_s, clock, "{epoch_line:?}");
        assert_eq!(epoch.sats.len(), 1, "{epoch_line:?}");
    }
}

#[test]
fn a_loosely_spaced_epoch_line_keeps_the_reading_its_tokens_always_had() {
    // Review built these. A lone five-digit token after the count may be a
    // clock offset written as an integer, and has always been read as one; a
    // flag and count run together must not be read differently because digits
    // follow a clock offset.
    for version in ["2.11", "3.05", "4.02"] {
        let header = if version.starts_with('2') {
            vec![v2_record("     1    C1", "# / TYPES OF OBSERV")]
        } else {
            vec![v2_record("G    1 C1C", "SYS / # / OBS TYPES")]
        };
        let obs = RinexObs::parse(&obs_file(
            version,
            'G',
            &header,
            "> 2020 06 24 00 00 0.0000000 0 1 00001\nG01      1234.567\n",
        ))
        .unwrap_or_else(|error| panic!("{version}: {error}"));
        let epoch = &obs.epochs()[0];
        assert_eq!(epoch.rcv_clock_offset_s, Some(1.0), "{version}");
        assert_eq!(epoch.epoch_picoseconds, None, "{version}");
    }
    let (satellites, systems) = multi_constellation_fixture(100);
    let mut body = "> 2020 06 24 00 00  0.0000000  3100 0.125 00001".to_string();
    for satellite in &satellites {
        body.push_str(&format!("\n{satellite}   23000000.000"));
    }
    assert!(
        RinexObs::parse(&obs_with_code_headers(&systems, &body)).is_err(),
        "a merged flag and count followed by a clock and digits stays rejected"
    );
}

#[test]
fn a_header_only_version_two_file_keeps_its_list_when_counts_name_another_constellation() {
    // Review built this: GPS `C2` with only a GLONASS count and no
    // observations. The file states the GPS list its version record names, so
    // a conversion to 2.12 has to keep that list or report its change; it used
    // to write `C2`, which reads as GPS C2W, with no change reported.
    let obs = RinexObs::parse(&obs_file(
        "2.11",
        'G',
        &[
            v2_record("     1    C2", "# / TYPES OF OBSERV"),
            v2_record("   R01     7", "PRN / # OF OBS"),
        ],
        "",
    ))
    .expect("parse");
    let held = obs.header().obs_codes[&GnssSystem::Gps].clone();
    let (converted, changes) = obs.downgrade_to_rinex2(2.12).expect("downgrade");
    let text = converted.to_rinex_string().expect("writes");
    let read = RinexObs::parse(&text).expect("reads back");
    let now = read.header().obs_codes[&GnssSystem::Gps].clone();
    // Each code the list held reads back where it was, or its rename is
    // reported; any further GPS column the conversion needed is reported as
    // added.
    for (index, code) in held.iter().enumerate() {
        assert!(
            now.get(index) == Some(code)
                || changes.contains(&ObsDowngradeChange::CodeRenamed {
                    system: GnssSystem::Gps,
                    from: code.clone(),
                    to: now.get(index).cloned().unwrap_or_default(),
                }),
            "GPS {code} at {index} reads back as {now:?} with no change reported: {changes:?}"
        );
    }
    let added = changes
        .iter()
        .filter(|change| {
            matches!(change, ObsDowngradeChange::CodeAdded { system, .. } if *system == GnssSystem::Gps)
        })
        .count();
    assert_eq!(added, now.len() - held.len(), "{changes:?}");
}

#[test]
fn lists_nothing_names_do_not_change_a_downgrade() {
    // Review built these: a GPS observation holding C1X, with and without an
    // unused GLONASS list. The unused list took a column, moved the GPS code
    // and reported changes; at 1,000 codes it made the conversion refuse.
    let base = RinexObs::parse(&obs_file(
        "3.05",
        'G',
        &[v2_record("G    1 C1X", "SYS / # / OBS TYPES")],
        "> 2020 06 24 00 00  0.0000000  0  1\nG01      1234.567\n",
    ))
    .expect("parse");
    let (plain, plain_changes) = base.downgrade_to_rinex2(2.11).expect("downgrade");
    for unused in [vec!["L1C".to_string()], vec!["L1C".to_string(); 1000]] {
        let mut obs = base.clone();
        obs.header
            .obs_codes
            .insert(GnssSystem::Glonass, unused.clone());
        let (converted, changes) = obs
            .downgrade_to_rinex2(2.11)
            .unwrap_or_else(|error| panic!("{} unused codes: {error}", unused.len()));
        // The file does not state the unused list, so its removal is the one
        // change the list adds.
        let mut expected = plain_changes.clone();
        expected.push(ObsDowngradeChange::CodeListRemoved {
            system: GnssSystem::Glonass,
            codes: unused.clone(),
        });
        assert_eq!(changes, expected, "{} unused codes", unused.len());
        assert_eq!(
            converted.header().rinex2_types,
            plain.header().rinex2_types,
            "{} unused codes",
            unused.len()
        );
        assert_eq!(
            converted.epochs(),
            plain.epochs(),
            "{} unused codes",
            unused.len()
        );
    }
}

/// Whether a downgrade kept each code a list held where it was, or reported its
/// move or rename, and reported every column it added.
fn kept_or_reported(
    system: GnssSystem,
    held: &[String],
    now: &[String],
    changes: &[ObsDowngradeChange],
) {
    for (index, code) in held.iter().enumerate() {
        let moved = changes.iter().any(|change| {
            matches!(change, ObsDowngradeChange::CodeMoved { system: moved, code: moved_code, from, to }
                if *moved == system && moved_code == code && *from == index && now.get(*to) == Some(code))
        });
        assert!(
            now.get(index) == Some(code)
                || moved
                || changes.contains(&ObsDowngradeChange::CodeRenamed {
                    system,
                    from: code.clone(),
                    to: now.get(index).cloned().unwrap_or_default(),
                }),
            "{system:?} {code} at {index} reads back as {now:?} with no change reported: {changes:?}"
        );
    }
    let added = changes
        .iter()
        .filter(|change| {
            matches!(change, ObsDowngradeChange::CodeAdded { system: added, .. } if *added == system)
        })
        .count();
    assert_eq!(added, now.len() - held.len(), "{system:?}: {changes:?}");
}

#[test]
fn a_header_only_file_keeps_the_list_its_version_record_names_through_a_downgrade() {
    // Review built these. With no observations, a version 2 file states the
    // list of the constellation its version record names; a downgrade took
    // that list from whatever lists it had added, and left it out when the
    // product held none.
    let c2 = v2_record("     1    C2", "# / TYPES OF OBSERV");

    // GPS C2, a GLONASS count, and the lists cleared: GPS still reads C2X by
    // the names, and a conversion to 2.12 has to keep it or say so.
    let mut cleared = RinexObs::parse(&obs_file(
        "2.11",
        'G',
        &[c2.clone(), v2_record("   R01     7", "PRN / # OF OBS")],
        "",
    ))
    .expect("parse");
    cleared.header.obs_codes.clear();
    let held = rinex2_system_obs_codes(GnssSystem::Gps, &["C2".to_string()], 2.11);
    let (converted, changes) = cleared.downgrade_to_rinex2(2.12).expect("downgrade");
    let read = RinexObs::parse(&converted.to_rinex_string().expect("writes")).expect("reads back");
    kept_or_reported(
        GnssSystem::Gps,
        &held,
        &read.header().obs_codes[&GnssSystem::Gps],
        &changes,
    );

    // GLONASS C2 with a GPS count converts, keeping the GLONASS list.
    let glonass = RinexObs::parse(&obs_file(
        "2.11",
        'R',
        &[c2, v2_record("   G01     7", "PRN / # OF OBS")],
        "",
    ))
    .expect("parse");
    let held = glonass.header().obs_codes[&GnssSystem::Glonass].clone();
    let (converted, changes) = glonass
        .downgrade_to_rinex2(2.12)
        .expect("a GLONASS header with a GPS count converts");
    let read = RinexObs::parse(&converted.to_rinex_string().expect("writes")).expect("reads back");
    kept_or_reported(
        GnssSystem::Glonass,
        &held,
        &read.header().obs_codes[&GnssSystem::Glonass],
        &changes,
    );
    let g01 = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    assert!(read.header().prn_obs_counts[&g01].contains(&Some(7)));
}

#[test]
fn an_unused_list_does_not_change_the_constellation_a_version_record_names() {
    // A header-only GLONASS file with an unused Galileo list is still written
    // as a GLONASS file. The file does not state the Galileo list, so the
    // writer refuses it and a conversion removes it and says so.
    let mut obs = RinexObs::parse(&obs_file(
        "2.11",
        'R',
        &[v2_record("     1    C2", "# / TYPES OF OBSERV")],
        "",
    ))
    .expect("parse");
    assert_eq!(obs.header().rinex2_system, Some(GnssSystem::Glonass));
    obs.header
        .obs_codes
        .insert(GnssSystem::Galileo, vec!["C1X".to_string()]);
    assert_eq!(
        obs.to_rinex_string(),
        Err(RinexObsWriteError::CodeListNotStated {
            system: GnssSystem::Galileo
        })
    );
    let (converted, changes) = obs.downgrade_to_rinex2(2.11).expect("converts");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeListRemoved {
            system: GnssSystem::Galileo,
            codes: vec!["C1X".to_string()],
        }]
    );
    let text = converted.to_rinex_string().expect("writes");
    let version_record = text.lines().next().expect("version record");
    assert_eq!(&version_record[40..41], "R", "{version_record:?}");
    assert_eq!(
        RinexObs::parse(&text)
            .expect("reads back")
            .header()
            .rinex2_system,
        Some(GnssSystem::Glonass)
    );
}

#[test]
fn a_glonass_code_bias_code_wider_than_its_field_is_refused() {
    // The code is `A3`; a longer one would be cut when written and read back
    // as another code.
    let text = obs_file(
        "3.05",
        'R',
        &[
            v2_record("R    1 C1C", "SYS / # / OBS TYPES"),
            v2_record(
                "LONGCODE 0 LONGCODE 0 LONGCODE 0 LONGCODE 0",
                "GLONASS COD/PHS/BIS",
            ),
        ],
        "",
    );
    let error = RinexObs::parse(&text).expect_err("a long code is refused");
    assert!(
        error.to_string().contains("GLONASS COD/PHS/BIS code"),
        "{error}"
    );
}

#[test]
fn a_type_continuation_record_that_continues_no_list_is_refused() {
    // Review found a blank-count record before any count skipped with its
    // codes. The same held after a complete list, and for a version 3 record
    // with a blank system field before any system.
    let types = |content: &str| v2_record(content, "# / TYPES OF OBSERV");
    let sys_types = |content: &str| v2_record(content, "SYS / # / OBS TYPES");
    let cases = [
        (
            "2.11",
            "before any count",
            vec![types("          S2"), types("     1    C1")],
        ),
        (
            "2.11",
            "after a complete list",
            vec![types("     1    C1"), types("          S2")],
        ),
        (
            "3.05",
            "before any system",
            vec![sys_types("       C1C"), sys_types("G    1 C1C")],
        ),
    ];
    for (version, what, headers) in cases {
        let error = RinexObs::parse(&obs_file(version, 'G', &headers, "")).expect_err(what);
        assert!(
            error.to_string().contains("continues no declared list"),
            "{what}: {error}"
        );
    }

    // A continuation record with no codes on it adds nothing and is passed over.
    RinexObs::parse(&obs_file(
        "2.11",
        'G',
        &[types("     1    C1"), types("")],
        "",
    ))
    .expect("an empty continuation record is passed over");
}

#[test]
fn lists_nothing_names_do_not_hold_a_version_two_file_back() {
    // A GLONASS list no observation or count names is nothing a version 2 file
    // states, so it cannot keep the names a GPS observation needs. The writer
    // refuses to leave it out unsaid, and a conversion removes it and changes
    // nothing else.
    let mut obs = RinexObs::parse(&obs_file(
        "2.11",
        'G',
        &[v2_record("     1    C1", "# / TYPES OF OBSERV")],
        " 20  6 24  0  0  0.0000000  0  1G 1\n      1234.567\n",
    ))
    .expect("parse");
    obs.header
        .obs_codes
        .insert(GnssSystem::Glonass, vec!["L1C".to_string()]);
    assert_eq!(
        obs.to_rinex_string(),
        Err(RinexObsWriteError::CodeListNotStated {
            system: GnssSystem::Glonass
        })
    );
    let (converted, changes) = obs
        .downgrade_to_rinex2(2.11)
        .expect("the unused list does not refuse the conversion");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeListRemoved {
            system: GnssSystem::Glonass,
            codes: vec!["L1C".to_string()],
        }]
    );
    let text = converted.to_rinex_string().expect("writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.epochs(), obs.epochs());
}

#[test]
fn a_list_a_version_two_file_does_not_state_is_reported_by_a_conversion() {
    // Review built the first: a 3.05 header declaring GPS C1C and GLONASS L1C,
    // with no observations or counts. A version 2 file with no observations
    // states one list, and a conversion to 2.11 left GLONASS's out with no
    // change reported.
    let both = RinexObs::parse(&obs_file(
        "3.05",
        'M',
        &[
            v2_record("G    1 C1C", "SYS / # / OBS TYPES"),
            v2_record("R    1 L1C", "SYS / # / OBS TYPES"),
        ],
        "",
    ))
    .expect("parse");
    let (converted, changes) = both.downgrade_to_rinex2(2.11).expect("converts");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeListRemoved {
            system: GnssSystem::Glonass,
            codes: vec!["L1C".to_string()],
        }]
    );
    let read = RinexObs::parse(&converted.to_rinex_string().expect("writes")).expect("reads back");
    assert_eq!(read.header().obs_codes, converted.header().obs_codes);

    // With no GPS list the file is named for the first constellation whose
    // list no count keeps, and a counted one is kept beside it.
    let g01 = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let mut named = both.clone();
    named.header.obs_codes.clear();
    named
        .header
        .obs_codes
        .insert(GnssSystem::Gps, vec!["C1C".to_string()]);
    named
        .header
        .obs_codes
        .insert(GnssSystem::Glonass, vec!["C1C".to_string()]);
    // Galileo reads `C1` as C1X, so an L5Q list is one the names do not say.
    named
        .header
        .obs_codes
        .insert(GnssSystem::Galileo, vec!["L5Q".to_string()]);
    named.header.prn_obs_counts.insert(g01, vec![Some(7)]);
    let (converted, changes) = named.downgrade_to_rinex2(2.11).expect("converts");
    assert_eq!(
        changes,
        vec![ObsDowngradeChange::CodeListRemoved {
            system: GnssSystem::Galileo,
            codes: vec!["L5Q".to_string()],
        }]
    );
    let read = RinexObs::parse(&converted.to_rinex_string().expect("writes")).expect("reads back");
    assert_eq!(read.header().rinex2_system, Some(GnssSystem::Glonass));
    assert_eq!(
        read.header().obs_codes[&GnssSystem::Glonass],
        vec!["C1C".to_string()]
    );
    assert_eq!(read.header().prn_obs_counts[&g01], vec![Some(7)]);
}

#[test]
fn names_that_also_read_as_an_unused_list_keep_it() {
    // Review built these: GPS C2W and GLONASS C2P with no observations. At 2.12
    // `C2` reads as GPS C2W but not as GLONASS C2P, and `P2` reads as both. A
    // conversion to 2.12 chose `C2` and removed the GLONASS list, and a 2.12
    // product holding `C2` and both lists was refused.
    let v3 = RinexObs::parse(&obs_file(
        "3.05",
        'M',
        &[
            v2_record("G    1 C2W", "SYS / # / OBS TYPES"),
            v2_record("R    1 C2P", "SYS / # / OBS TYPES"),
        ],
        "",
    ))
    .expect("parse");
    for version in [2.10, 2.11, 2.12] {
        let (converted, changes) = v3
            .downgrade_to_rinex2(version)
            .unwrap_or_else(|error| panic!("{version}: {error}"));
        assert!(changes.is_empty(), "{version}: {changes:?}");
        assert_eq!(
            converted.header().obs_codes,
            v3.header().obs_codes,
            "{version}"
        );
        assert_eq!(converted.header().rinex2_types, vec!["P2".to_string()]);
    }

    let mut held = RinexObs::parse(&obs_file(
        "2.12",
        'M',
        &[v2_record("     1    C2", "# / TYPES OF OBSERV")],
        "",
    ))
    .expect("parse");
    assert_eq!(
        held.header().obs_codes[&GnssSystem::Gps],
        vec!["C2W".to_string()]
    );
    held.header
        .obs_codes
        .insert(GnssSystem::Glonass, vec!["C2P".to_string()]);
    let text = held
        .to_rinex_string()
        .expect("P2 states both lists, so the product writes");
    let read = RinexObs::parse(&text).expect("reads back");
    assert_eq!(read.header().rinex2_types, vec!["P2".to_string()]);
}

#[test]
fn a_blank_leap_second_field_keeps_its_columns() {
    // Review built the version 3 record: the future count blank, the week and
    // day present. The writer closed the blank field up, putting the week where
    // the future count goes and the day where the week goes, and reading that
    // back refused the file.
    let leap = super::ObsLeapSeconds {
        current: 18,
        delta_future: None,
        week: Some(2300),
        day: Some(1),
    };
    let obs = RinexObs::parse(&obs_file(
        "3.05",
        'G',
        &[
            v2_record("G    1 C1C", "SYS / # / OBS TYPES"),
            v2_record("    18        2300     1", "LEAP SECONDS"),
        ],
        "> 2020 06 24 00 00  0.0000000  0  1\nG01      1234.567\n",
    ))
    .expect("parse");
    assert_eq!(obs.header().leap_seconds, Some(leap));
    let read = RinexObs::parse(&obs.to_rinex_string().expect("writes")).expect("reads back");
    assert_eq!(read.header().leap_seconds, Some(leap));

    // A version 2 file with only the week keeps it in the week's columns.
    let mut v2 = version_two_fixture();
    let week_only = super::ObsLeapSeconds {
        current: 18,
        delta_future: None,
        week: Some(2300),
        day: None,
    };
    v2.header.leap_seconds = Some(week_only);
    let read = RinexObs::parse(&v2.to_rinex_string().expect("writes")).expect("reads back");
    assert_eq!(read.header().leap_seconds, Some(week_only));
}

#[test]
fn repair_keeps_epochs_that_differ_only_in_picoseconds() {
    // Two epochs at the same seven-decimal second with picoseconds 1 and 2 are
    // different instants; repair merged them as duplicates and dropped the
    // second measurement.
    let text = obs_file(
        "4.02",
        'G',
        &[v2_record("G    1 C1C", "SYS / # / OBS TYPES")],
        &format!(
            "> 2020 06 24 00 00  0.0000000  0  1{blank}00001\nG01      1234.567\n\
             > 2020 06 24 00 00  0.0000000  0  1{blank}00002\nG01      9876.543\n",
            blank = " ".repeat(22)
        ),
    );
    let obs = RinexObs::parse(&text).expect("parse");
    assert_eq!(obs.epochs().len(), 2);
    let options = crate::rinex_qc::RepairOptions {
        set_interval: true,
        set_time_of_last_obs: true,
        set_obs_counts: true,
        drop_empty_records: true,
        drop_unsupported: true,
        ..crate::rinex_qc::RepairOptions::default()
    };
    let repaired = crate::rinex_qc::repair_obs(&obs, &options).repaired;
    assert_eq!(repaired.epochs().len(), 2);
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let values: Vec<Option<f64>> = repaired
        .epochs()
        .iter()
        .map(|epoch| epoch.sats[&gps][0].value)
        .collect();
    assert_eq!(values, [Some(1234.567), Some(9876.543)]);
}

#[test]
fn downgrade_refuses_values_and_counts_past_the_codes() {
    // A value past its constellation's codes names no observable. Laying the
    // codes out used to drop it, indicators and all, with nothing reported.
    let gps = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 1,
    };
    let product = || {
        two_system_product(
            2.11,
            &(GnssSystem::Gps, 'G', 1, vec!["C1C".to_string()]),
            &(GnssSystem::Glonass, 'R', 2, vec!["C2C".to_string()]),
            true,
        )
    };
    let mut extra_value = product();
    extra_value.epochs[0]
        .sats
        .get_mut(&gps)
        .expect("GPS satellite")
        .push(ObsValue {
            value: Some(999.0),
            lli: Some(1),
            ssi: Some(5),
        });
    assert_eq!(
        extra_value.downgrade_to_rinex2(2.11).map(|_| ()),
        Err(RinexObsWriteError::ValuesWithoutCodes {
            epoch_index: 0,
            satellite: gps,
            codes: 1,
            values: 2,
        })
    );
    let mut extra_count = product();
    extra_count
        .header
        .prn_obs_counts
        .insert(gps, vec![Some(1), Some(2)]);
    assert_eq!(
        extra_count.downgrade_to_rinex2(2.11).map(|_| ()),
        Err(RinexObsWriteError::CountsWithoutCodes {
            satellite: gps,
            codes: 1,
            counts: 2,
        })
    );
}

/// Every move and addition a downgrade reports is one its layout made, every
/// one its layout made is reported, and no order of the same columns under
/// which every constellation reads its placed codes the same moves fewer.
fn assert_moves_and_additions_are_itemized(
    label: &str,
    version: f64,
    original: &RinexObs,
    text: &str,
    read: &RinexObs,
    changes: &[ObsDowngradeChange],
) {
    let names: Vec<String> = text
        .lines()
        .filter(|line| line.get(60..).unwrap_or("").trim() == "# / TYPES OF OBSERV")
        .flat_map(|line| {
            line[6..60]
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    // Where each held code landed, per constellation, from the values.
    let mut placed: BTreeMap<GnssSystem, Vec<(usize, usize)>> = BTreeMap::new();
    for (sat, values) in &original.epochs()[0].sats {
        if placed.contains_key(&sat.system) {
            continue;
        }
        let read_values = &read.epochs()[0].sats[sat];
        let row = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                let v = value.value?;
                let column = read_values
                    .iter()
                    .position(|found| found.value == Some(v))?;
                Some((index, column))
            })
            .collect();
        placed.insert(sat.system, row);
    }
    let mut expected_moves = 0_usize;
    for (system, row) in &placed {
        let now = &read.header().obs_codes[system];
        for &(index, column) in row {
            if index != column {
                expected_moves += 1;
                let change = ObsDowngradeChange::CodeMoved {
                    system: *system,
                    code: now[column].clone(),
                    from: index,
                    to: column,
                };
                assert!(
                    changes.contains(&change),
                    "{label}: {change:?} happened and is not reported: {changes:?}"
                );
            }
        }
        for (column, code) in now.iter().enumerate() {
            if row.iter().any(|&(_, at)| at == column) {
                continue;
            }
            let reported = changes
                .iter()
                .filter(|change| {
                    matches!(change, ObsDowngradeChange::CodeAdded { system: s, code: c }
                        if s == system && c == code)
                })
                .count();
            let happened = now
                .iter()
                .enumerate()
                .filter(|(at, c)| *c == code && !row.iter().any(|&(_, p)| p == *at))
                .count();
            assert_eq!(
                reported, happened,
                "{label}: {system:?} {code} was added {happened} times and reported {reported}: {changes:?}"
            );
        }
        let added = changes
            .iter()
            .filter(|change| matches!(change, ObsDowngradeChange::CodeAdded { system: s, .. } if s == system))
            .count();
        assert_eq!(
            added,
            now.len() - row.len(),
            "{label}: {system:?} additions reported beyond the columns added: {changes:?}"
        );
    }
    let reported_moves = changes
        .iter()
        .filter(|change| matches!(change, ObsDowngradeChange::CodeMoved { .. }))
        .count();
    assert_eq!(
        reported_moves, expected_moves,
        "{label}: moves reported beyond the moves made: {changes:?}"
    );
    if names.len() > 6 {
        return;
    }
    let mut fewest = usize::MAX;
    for order in permutations(names.len()) {
        let ordered: Vec<String> = order.iter().map(|&column| names[column].clone()).collect();
        let mut moves = 0_usize;
        let valid = placed.iter().all(|(system, row)| {
            let now = &read.header().obs_codes[system];
            let reordered = rinex2_system_obs_codes(*system, &ordered, version);
            row.iter().all(|&(index, column)| {
                let position = order.iter().position(|&c| c == column).expect("in order");
                if position != index {
                    moves += 1;
                }
                reordered[position] == now[column]
            })
        });
        if valid {
            fewest = fewest.min(moves);
        }
    }
    assert!(
        expected_moves <= fewest,
        "{label}: {expected_moves} moves where an order of the same columns moves {fewest}: {names:?}"
    );
}

/// Every ordering of `0..n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for shorter in permutations(n - 1) {
        for at in 0..=shorter.len() {
            let mut order = shorter.clone();
            order.insert(at, n - 1);
            out.push(order);
        }
    }
    out
}

#[test]
fn downgrade_shares_columns_for_the_same_second_copies_across_constellations() {
    // Two constellations each holding 500 copies of a code no version 2 name
    // spells need 500 columns between them, not 1,000, which is more than the
    // 999 types a version 2 header can declare.
    // The text template declares at most nine types, so the product is
    // widened in memory.
    let one = vec!["C9X".to_string()];
    let mut product = two_system_product(
        2.11,
        &(GnssSystem::Gps, 'G', 1, one.clone()),
        &(GnssSystem::Glonass, 'R', 2, one),
        true,
    );
    for codes in product
        .header
        .obs_codes
        .values_mut()
        .chain(product.header.declared_obs_codes.values_mut())
    {
        *codes = vec!["C9X".to_string(); 500];
    }
    for epoch in &mut product.epochs {
        for (sat, values) in &mut epoch.sats {
            let base = f64::from(sat.prn) * 1_000.0;
            *values = (0..500)
                .map(|index| ObsValue {
                    value: Some(base + index as f64),
                    lli: None,
                    ssi: None,
                })
                .collect();
        }
    }
    let (downgraded, _) = product.downgrade_to_rinex2(2.11).expect("downgrade");
    let text = downgraded.to_rinex_string().expect("the downgrade writes");
    let read = RinexObs::parse(&text).expect("reads back");
    for system in [GnssSystem::Gps, GnssSystem::Glonass] {
        assert_eq!(read.header().obs_codes[&system].len(), 500, "{system:?}");
    }
}

/// Every code list of one or two codes a small mixed-product test draws for a
/// constellation: codes version 2 names, a code no version 2 name spells
/// (`C9X`), and a name kept as written.
fn small_code_lists(system: GnssSystem) -> Vec<Vec<String>> {
    let codes: &[&str] = match system {
        GnssSystem::Gps => &["C1C", "C1W", "C2X", "L1C", "C9X"],
        GnssSystem::Glonass => &["C1C", "C1P", "C2C", "L1C", "C9X"],
        GnssSystem::Galileo => &["C1X", "C5X", "L1X", "P1", "C9X"],
        _ => &["C2I", "C7I", "C6I", "C2", "C9X"],
    };
    let mut lists: Vec<Vec<String>> = codes.iter().map(|code| vec![(*code).to_string()]).collect();
    for first in codes {
        for second in codes {
            lists.push(vec![(*first).to_string(), (*second).to_string()]);
        }
    }
    lists
}

/// The fewest codes any version 2 layout has to rename for these two
/// constellations' lists, by brute force: every sequence of the names that can
/// read back as one of their codes, at every width from the longer list to both
/// lists side by side plus one column for each name kept as written, since such
/// a name needs a column before it reading as the code it stands for. Each code
/// is kept where some column reads back as it.
fn fewest_forced_renames(
    version: f64,
    (system_a, list_a): (GnssSystem, &[String]),
    (system_b, list_b): (GnssSystem, &[String]),
) -> usize {
    let reads_back_as = |system: GnssSystem, name: &str, code: &str| {
        if rinex2_kept_as_written(code) {
            name == code
                || (rinex2_name_allowed(system, name, version)
                    && canonical_rinex2_obs_code(system, name, version)
                        == canonical_rinex2_obs_code(system, code, version))
        } else {
            rinex2_name_allowed(system, name, version)
                && canonical_rinex2_obs_code(system, name, version) == code
        }
    };
    let names: Vec<String> = every_version_two_name()
        .into_iter()
        .filter(|name| {
            list_a
                .iter()
                .any(|code| reads_back_as(system_a, name, code))
                || list_b
                    .iter()
                    .any(|code| reads_back_as(system_b, name, code))
        })
        .collect();
    // Codes a reading keeps in place: the multiset intersection, since each code
    // needs a column of its own that reads back as it.
    let kept = |list: &[String], read: &[String]| -> usize {
        let mut pool: Vec<&String> = read.iter().collect();
        list.iter()
            .filter(|code| {
                pool.iter()
                    .position(|found| found == code)
                    .map(|at| pool.swap_remove(at))
                    .is_some()
            })
            .count()
    };
    let total = list_a.len() + list_b.len();
    let mut fewest = total;
    if names.is_empty() {
        return fewest;
    }
    let kept_as_written = list_a
        .iter()
        .chain(list_b)
        .filter(|code| rinex2_kept_as_written(code))
        .count();
    for width in list_a.len().max(list_b.len())..=total + kept_as_written {
        let mut digits = vec![0_usize; width];
        'sequences: loop {
            let sequence: Vec<String> = digits.iter().map(|&d| names[d].clone()).collect();
            let read_a = rinex2_system_obs_codes(system_a, &sequence, version);
            let read_b = rinex2_system_obs_codes(system_b, &sequence, version);
            fewest = fewest.min(total - kept(list_a, &read_a) - kept(list_b, &read_b));
            if fewest == 0 {
                return 0;
            }
            for place in (0..width).rev() {
                digits[place] += 1;
                if digits[place] < names.len() {
                    continue 'sequences;
                }
                digits[place] = 0;
            }
            break;
        }
    }
    fewest
}

/// How many codes the downgrade renamed for a product, after checking the
/// result writes.
fn downgrade_renames(product: &RinexObs, version: f64, label: &str) -> usize {
    let (downgraded, changes) = product
        .downgrade_to_rinex2(version)
        .unwrap_or_else(|e| panic!("{label}: downgrade: {e}"));
    downgraded
        .to_rinex_string()
        .unwrap_or_else(|e| panic!("{label}: the downgrade does not write: {e}"));
    changes
        .iter()
        .filter(|change| matches!(change, ObsDowngradeChange::CodeRenamed { .. }))
        .count()
}

#[test]
fn downgrade_renames_no_more_codes_than_any_layout_must() {
    // The downgrade may rename a code only when no version 2 layout keeps it.
    // For every small two-constellation product it has to rename exactly as many
    // codes as the fewest any layout must. More would be a change made when a
    // layout without it existed.
    let mut products = 0_usize;
    let mut forced = 0_usize;
    for version in [2.11, 2.12] {
        for (sa, la, pa, sb, lb, pb) in SMALL_PAIRS {
            for list_a in small_code_lists(sa) {
                for list_b in small_code_lists(sb) {
                    let label = format!("{version} {sa:?} {list_a:?} beside {sb:?} {list_b:?}");
                    let fewest = fewest_forced_renames(version, (sa, &list_a), (sb, &list_b));
                    let product = two_system_product(
                        version,
                        &(sa, la, pa, list_a.clone()),
                        &(sb, lb, pb, list_b.clone()),
                        true,
                    );
                    let renamed = downgrade_renames(&product, version, &label);
                    assert_eq!(
                        renamed, fewest,
                        "{label}: the downgrade renamed {renamed} codes where a layout renames {fewest}"
                    );
                    forced += usize::from(fewest > 0);
                    products += 1;
                }
            }
        }
    }
    assert!(
        products > 1_000 && forced > 0,
        "{products} products, {forced} with a rename no layout avoids"
    );
}

#[test]
fn downgrade_keeps_every_code_in_the_products_review_found_it_renaming() {
    // The small sweep's alphabets do not hold the products review found the
    // greedy layout mishandling at 2.12, so they are checked here by name,
    // against the same brute force at the widths their longer lists need: GPS
    // `[L1W, C2W]` beside GLONASS `[C2P, C2C]`, where the writer put GPS `C2W`
    // under a column GLONASS claimed; the same GPS list beside GLONASS
    // `[C2P, L1P, C2C]`, where every candidate for GPS looked unsafe; and GPS
    // `[C2W, C2]` beside BeiDou `[P2, C2I]`, two names kept as written.
    type NamedCase = (
        GnssSystem,
        char,
        u8,
        &'static [&'static str],
        GnssSystem,
        char,
        u8,
        &'static [&'static str],
    );
    let cases: [NamedCase; 3] = [
        (
            GnssSystem::Gps,
            'G',
            1,
            &["L1W", "C2W"],
            GnssSystem::Glonass,
            'R',
            2,
            &["C2P", "C2C"],
        ),
        (
            GnssSystem::Gps,
            'G',
            1,
            &["L1W", "C2W"],
            GnssSystem::Glonass,
            'R',
            2,
            &["C2P", "L1P", "C2C"],
        ),
        (
            GnssSystem::Gps,
            'G',
            1,
            &["C2W", "C2"],
            GnssSystem::BeiDou,
            'C',
            5,
            &["P2", "C2I"],
        ),
    ];
    for (sa, la, pa, codes_a, sb, lb, pb, codes_b) in cases {
        let list_a: Vec<String> = codes_a.iter().map(|code| (*code).to_string()).collect();
        let list_b: Vec<String> = codes_b.iter().map(|code| (*code).to_string()).collect();
        let label = format!("2.12 {sa:?} {list_a:?} beside {sb:?} {list_b:?}");
        let fewest = fewest_forced_renames(2.12, (sa, &list_a), (sb, &list_b));
        let product = two_system_product(
            2.12,
            &(sa, la, pa, list_a.clone()),
            &(sb, lb, pb, list_b.clone()),
            true,
        );
        let renamed = downgrade_renames(&product, 2.12, &label);
        assert_eq!(
            renamed, fewest,
            "{label}: the downgrade renamed {renamed} codes where a layout renames {fewest}"
        );
    }
}

#[test]
fn a_version_two_file_keeps_its_columns_through_repeated_rewrites() {
    // Version 2 names its codes once for every constellation at once, so a name
    // one of them has no observable for lands on a code it already holds:
    // Galileo has no `P1`, and both `C1` and `P1` read as its `C1X`. Giving that
    // a column of its own added one to the header on every rewrite, and the next
    // read named it again, so the file grew without bound.
    let mut text = String::new();
    text.push_str(
        "     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n",
    );
    text.push_str(&format!(
        "{:<60}{:<20}\n",
        "     2    C1    P1", "# / TYPES OF OBSERV"
    ));
    text.push_str(&format!(
        "{:<60}{:<20}\n",
        "  2015     1     1     0     0    0.0000000     GPS", "TIME OF FIRST OBS"
    ));
    text.push_str(&format!("{:<60}{:<20}\n", "", "END OF HEADER"));
    text.push_str(" 15  1  1  0  0  0.0000000  0  2G 1E11\n");
    text.push_str("  20000000.000  20000001.000\n");
    text.push_str("  21000000.000\n");

    let first = RinexObs::parse(&text).expect("parse the mixed version 2 file");
    let mut obs = first.clone();
    for round in 0..4 {
        let encoded = obs.to_rinex_string().expect("serialize RINEX OBS");
        let declared = encoded
            .lines()
            .find(|line| line.contains("# / TYPES OF OBSERV"))
            .expect("the header names its types");
        assert_eq!(
            &declared[..18],
            "     2    C1    P1",
            "rewrite {round} changed the header: {declared:?}"
        );
        obs = RinexObs::parse(&encoded).expect("its own output must read back");
        assert_eq!(
            obs.epochs(),
            first.epochs(),
            "rewrite {round} moved a value"
        );
    }
}

#[test]
fn real_igs_headers_with_unknown_alignment_contradictory_shifts_and_blank_biases_are_read() {
    // IGS 2026 day 001 headers the reader refused, PRN / # OF OBS left out:
    // JOZ200POL gives C, E and S phase shift records with no code, MET300FIN
    // gives C02 and C05 two L2I and two L6I corrections, and NICO00CYP leaves
    // C1P's GLONASS bias blank.
    for (name, header) in [
        (
            "JOZ200POL",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/obs/JOZ200POL_R_20260010000_01D_30S_MO_header_trim.rnx"
            )),
        ),
        (
            "MET300FIN",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/obs/MET300FIN_R_20260010000_01D_30S_MO_header_trim.rnx"
            )),
        ),
        (
            "NICO00CYP",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/obs/NICO00CYP_R_20260010000_01D_30S_MO_header_trim.rnx"
            )),
        ),
    ] {
        let obs = RinexObs::parse(header).unwrap_or_else(|error| panic!("{name}: {error}"));
        // Written back and read as itself. NICO00CYP's RCV CLOCK OFFS APPL is a
        // record this reader keeps no value of, reported as not retained, and
        // not written.
        let written = obs.to_rinex_string().expect("write");
        let mut reread = RinexObs::parse(&written).expect("reads back");
        let mut original = obs.clone();
        original.header.unretained_header_labels.clear();
        reread.header.unretained_header_labels.clear();
        assert_eq!(reread, original, "{name}");
        let header = obs.header();
        let shift = |sat: &str, code: &str| {
            phase_shift_cycles(header, sat.parse().expect("satellite"), code)
        };
        match name {
            "JOZ200POL" => {
                assert_eq!(obs.skipped_records, 0);
                let unknown: Vec<GnssSystem> = header
                    .phase_shifts
                    .iter()
                    .filter(|shift| shift.code.is_none())
                    .map(|shift| shift.system)
                    .collect();
                assert_eq!(
                    unknown,
                    [GnssSystem::BeiDou, GnssSystem::Galileo, GnssSystem::Sbas]
                );
                assert_eq!(shift("G01", "L2X"), Ok(-0.25));
                assert_eq!(shift("G01", "L1C"), Ok(0.0));
                assert_eq!(shift("R01", "L1P"), Ok(0.25));
                assert_eq!(shift("C19", "L2I"), Err(CorrectionUnavailable::Unknown));
                assert_eq!(shift("E11", "L1X"), Err(CorrectionUnavailable::Unknown));
                assert_eq!(
                    header.glonass_code_phase_bias("C1C"),
                    Err(CorrectionUnavailable::Unknown)
                );
                for system in ["C", "E", "S"] {
                    let line = header_line(system, "SYS / PHASE SHIFT");
                    assert!(written.lines().any(|written| written == line), "{line}");
                }
            }
            "MET300FIN" => {
                // Four satellite designators the id does not hold, and C02 and
                // C05 given two L2I and two L6I corrections.
                assert_eq!(obs.skipped_records, 8);
                for sat in ["C02", "C05"] {
                    for code in ["L2I", "L6I"] {
                        assert_eq!(
                            shift(sat, code),
                            Err(CorrectionUnavailable::Ambiguous {
                                corrections: vec![Some(0.0), Some(0.5)]
                            }),
                            "{sat} {code}"
                        );
                    }
                    assert_eq!(shift(sat, "L1X"), Ok(0.0));
                    assert_eq!(shift(sat, "L7I"), Ok(0.0));
                }
                assert_eq!(shift("C60", "L2I"), Ok(0.0));
                assert_eq!(shift("C06", "L2I"), Ok(0.5));
                assert_eq!(shift("C06", "L7I"), Ok(0.5));
                assert_eq!(shift("G01", "L1W"), Ok(-0.25));
            }
            _ => {
                assert_eq!(obs.skipped_records, 0);
                assert_eq!(
                    header.glonass_cod_phs_bis,
                    Some(vec![
                        ("C1C".to_string(), Some(0.0)),
                        ("C1P".to_string(), None),
                        ("C2C".to_string(), Some(0.0)),
                        ("C2P".to_string(), Some(0.0)),
                    ])
                );
                assert_eq!(
                    header.glonass_code_phase_bias("C1P"),
                    Err(CorrectionUnavailable::Unknown)
                );
                assert_eq!(header.glonass_code_phase_bias("C2C"), Ok(Some(0.0)));
                let line = header_line(
                    &format!(
                        " C1C {:8.3} C1P {:8} C2C {:8.3} C2P {:8.3}",
                        0.0, "", 0.0, 0.0
                    ),
                    "GLONASS COD/PHS/BIS",
                );
                assert!(written.lines().any(|written| written == line), "{written}");
            }
        }
    }
}

#[test]
fn contradictory_phase_shifts_and_glonass_biases_are_kept_and_read_as_ambiguous() {
    // A block giving G01's L1W two corrections and C1C two biases. Neither
    // changes how observations are read, so the file is read, the records are
    // kept and written back, and those two signals read as ambiguous.
    let types = [
        header_line("G    2 C1C L1W", "SYS / # / OBS TYPES"),
        header_line("R    1 C1C", "SYS / # / OBS TYPES"),
    ];
    let records = [
        header_line("G L1W  0.25000  01 G01", "SYS / PHASE SHIFT"),
        header_line("G L1W  0.50000  02 G02 G01", "SYS / PHASE SHIFT"),
        header_line(
            &format!(" C1C {:8.3} C1C {:8.3}", -9.0, -10.0),
            "GLONASS COD/PHS/BIS",
        ),
    ];
    let later = header_line("G L1W  0.75000  01 G01", "SYS / PHASE SHIFT");
    let g01: GnssSatelliteId = "G01".parse().expect("G01");
    let g02: GnssSatelliteId = "G02".parse().expect("G02");
    for in_event in [false, true] {
        let mut headers = types.to_vec();
        let body = if in_event {
            [
                blank_event(4, &records),
                blank_event(4, std::slice::from_ref(&later)),
            ]
            .join("\n")
        } else {
            headers.extend(records.iter().cloned());
            blank_event(4, std::slice::from_ref(&later))
        };
        let obs = RinexObs::parse(&obs_with_code_headers(&headers, &body))
            .unwrap_or_else(|error| panic!("event {in_event}: {error}"));
        assert_eq!(obs.skipped_records, 2, "event {in_event}");
        let header = if in_event {
            obs.header_at(0).expect("header in effect")
        } else {
            obs.header().clone()
        };
        assert_eq!(
            phase_shift_cycles(&header, g01, "L1W"),
            Err(CorrectionUnavailable::Ambiguous {
                corrections: vec![Some(0.25), Some(0.5)]
            })
        );
        assert_eq!(phase_shift_cycles(&header, g02, "L1W"), Ok(0.5));
        assert_eq!(
            header.glonass_code_phase_bias("C1C"),
            Err(CorrectionUnavailable::Ambiguous {
                corrections: vec![Some(-9.0), Some(-10.0)]
            })
        );
        // A later block's record for G01 settles it.
        let last = obs
            .header_at(obs.epochs().len() - 1)
            .expect("header in effect");
        assert_eq!(phase_shift_cycles(&last, g01, "L1W"), Ok(0.75));
        assert_eq!(phase_shift_cycles(&last, g02, "L1W"), Ok(0.5));
        assert_writes_back(&obs);
    }
    // Identical records, and a blank correction beside a zero one, say one
    // thing and count nothing.
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            types[0].clone(),
            header_line("G L1W  0.25000", "SYS / PHASE SHIFT"),
            header_line("G L1W  0.25000", "SYS / PHASE SHIFT"),
            header_line("G L1C", "SYS / PHASE SHIFT"),
            header_line("G L1C  0.00000", "SYS / PHASE SHIFT"),
        ],
        "",
    ))
    .expect("parse");
    assert_eq!(obs.skipped_records, 0);
    assert_eq!(phase_shift_cycles(obs.header(), g01, "L1W"), Ok(0.25));
    assert_eq!(phase_shift_cycles(obs.header(), g01, "L1C"), Ok(0.0));
}

#[test]
fn a_phase_shift_record_naming_only_its_constellation_declares_the_alignment_unknown() {
    // RINEX 3.05 section 5.2.12: "If the applied phase corrections or the phase
    // alignment is unknown, then the observation code field and the rest of
    // the SYS / PHASE SHIFT header record field of the respective satellite
    // system(s) are left blank."
    let types = header_line("G    2 L1C L2W", "SYS / # / OBS TYPES");
    let g01: GnssSatelliteId = "G01".parse().expect("G01");
    let obs = RinexObs::parse(&obs_with_code_headers(
        &[
            types.clone(),
            header_line("G", "SYS / PHASE SHIFT"),
            header_line("G L2W -0.25000", "SYS / PHASE SHIFT"),
        ],
        &[
            blank_event(4, &[header_line("G", "SYS / PHASE SHIFT")]),
            blank_event(4, &[header_line("G L1C  0.50000", "SYS / PHASE SHIFT")]),
        ]
        .join("\n"),
    ))
    .expect("parse");
    // In one block a record for a code applies to it over the constellation's.
    assert_eq!(
        phase_shift_cycles(obs.header(), g01, "L1C"),
        Err(CorrectionUnavailable::Unknown)
    );
    assert_eq!(phase_shift_cycles(obs.header(), g01, "L2W"), Ok(-0.25));
    // A later block's record naming only the constellation replaces them all,
    // and a later record for a code covers that code again.
    let after = obs.header_at(0).expect("header in effect");
    assert_eq!(
        phase_shift_cycles(&after, g01, "L2W"),
        Err(CorrectionUnavailable::Unknown)
    );
    let last = obs.header_at(1).expect("header in effect");
    assert_eq!(phase_shift_cycles(&last, g01, "L1C"), Ok(0.5));
    assert_eq!(
        phase_shift_cycles(&last, g01, "L2W"),
        Err(CorrectionUnavailable::Unknown)
    );
    assert_writes_back(&obs);
    let written = obs.to_rinex_string().expect("write");
    assert!(
        written.contains(&header_line("G", "SYS / PHASE SHIFT")),
        "{written}"
    );
    // A constellation letter with a correction and no code is not that record.
    assert_parse_err(minimal_obs_with_phase_shift("G 0.25000"));
    // From RINEX 4.00 the record is ignored.
    let version_four = RinexObs::parse(&obs_with_version_and_code_headers(
        4.02,
        &[types, header_line("G", "SYS / PHASE SHIFT")],
        "",
    ))
    .expect("parse");
    assert_eq!(
        phase_shift_cycles(version_four.header(), g01, "L1C"),
        Ok(0.0)
    );
}

#[test]
fn glonass_biases_are_read_in_their_columns_with_a_blank_bias_kept_blank() {
    let columns = format!(" C1C {:8.3} C1P {:8} C2C {:8.3}", -10.0, "", -10.432);
    let obs = RinexObs::parse(&minimal_obs(
        &[header_line(&columns, "GLONASS COD/PHS/BIS")],
        "",
    ))
    .expect("parse");
    assert_eq!(
        obs.header().glonass_cod_phs_bis,
        Some(vec![
            ("C1C".to_string(), Some(-10.0)),
            ("C1P".to_string(), None),
            ("C2C".to_string(), Some(-10.432)),
        ])
    );
    let written = assert_writes_back(&obs);
    assert!(
        written.contains(&header_line(&columns, "GLONASS COD/PHS/BIS")),
        "{written}"
    );
    // A blank last bias, and a record not laid out in the columns, as RINEX
    // 4.00's Table A2 example is, read by its fields.
    for (content, expected) in [
        (
            format!(" C1C {:8.3} C1P", -10.0),
            vec![("C1C", Some(-10.0)), ("C1P", None)],
        ),
        (
            "  C1C -10.000 C1P -10.123 C2C -10.432 C2P -10.634".to_string(),
            vec![
                ("C1C", Some(-10.0)),
                ("C1P", Some(-10.123)),
                ("C2C", Some(-10.432)),
                ("C2P", Some(-10.634)),
            ],
        ),
    ] {
        let obs = RinexObs::parse(&minimal_obs(
            &[header_line(&content, "GLONASS COD/PHS/BIS")],
            "",
        ))
        .unwrap_or_else(|error| panic!("{content}: {error}"));
        assert_eq!(
            obs.header().glonass_cod_phs_bis,
            Some(
                expected
                    .into_iter()
                    .map(|(code, bias)| (code.to_string(), bias))
                    .collect()
            ),
            "{content}"
        );
        assert_writes_back(&obs);
    }
}
