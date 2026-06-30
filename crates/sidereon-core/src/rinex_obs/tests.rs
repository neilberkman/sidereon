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
fn rejects_non_v3_observation_file() {
    let v2 = "     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n";
    assert!(RinexObs::parse(v2).is_err());
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
    assert_eq!(
        reparsed, obs,
        "to_rinex_string must round-trip through parse"
    );
    // Deterministic output.
    assert_eq!(reparsed.to_rinex_string(), serialized);
}
