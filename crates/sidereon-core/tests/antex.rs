#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::antex::{
    AntennaKind, Antex, AntexDateTime, AntexError, Calibration, OuterComment, PcvGrid, PcvType,
};

// Fixture provenance:
// `igs20_wettzell_trim.atx` is a trimmed derivative of the IGS ANTEX `igs20.atx`
// (https://files.igs.org/pub/station/general/igs20.atx, downloaded 2026-06-12,
// 56564581 bytes, sha256
// 70e963f66ca46c801a9fc8b37b0a0023c8e5213a724d7f26972ae81a80ce9699, release marker
// IGS20_2417). Attribution to IGS (open access); the committed copy is a trimmed
// derivative: 364261 bytes, sha256
// 5c30f41a7cb75564eb379fcbc10e123ebafce86f8c305836413cdeaa129cfa02. Retained the
// ANTEX header verbatim, the complete satellite antenna blocks valid at the WTZ
// fixture epoch 2020-06-25 UTC for PRNs G05,G08,G09,G13,G15,G18,G27,G28,G30, and
// the receiver antenna block `LEIAR25.R3      LEIT` found in both the WTZR and WTZZ
// 120-epoch RINEX headers. No retained ANTEX lines were altered.
// `antex_golden.json` holds PCO values and selected PCV grid samples transcribed
// from the retained blocks, recorded as numeric values and as source decimal text.
const ATX: &str = include_str!("fixtures/antex/igs20_wettzell_trim.atx");
const GOLDEN: &str = include_str!("fixtures/antex/antex_golden.json");

#[test]
fn parses_fixture_and_matches_golden_pco_pcv_bits() {
    let antex = Antex::parse(ATX).expect("parse ANTEX fixture");
    let golden: Value = serde_json::from_str(GOLDEN).expect("parse ANTEX golden");
    let antennas = golden["antennas"].as_array().expect("golden antennas");

    let matching = antennas
        .iter()
        .filter(|antenna| antex.antenna(antenna["id"].as_str().unwrap()).is_some())
        .count();
    assert_eq!(antex.antennas.len(), matching);
    assert_eq!(matching, antennas.len());

    for antenna_golden in antennas {
        let id = antenna_golden["id"].as_str().unwrap();
        let antenna = antex.antenna(id).unwrap_or_else(|| panic!("missing {id}"));

        assert_eq!(antenna.id, id);
        assert_eq!(
            antenna.kind,
            if antenna_golden["kind"].as_str().unwrap() == "satellite" {
                AntennaKind::Satellite
            } else {
                AntennaKind::Receiver
            }
        );
        if let Some(prn) = antenna_golden["prn"].as_str() {
            assert_eq!(
                antenna.antenna_type,
                antenna_golden["antenna_type"].as_str().unwrap()
            );
            assert_eq!(antenna.serial, prn);
        }
        assert_eq!(
            antenna.dazi_deg.unwrap().to_bits(),
            antenna_golden["dazi_deg"].as_f64().unwrap().to_bits()
        );

        let grid = &antenna_golden["zenith_grid_deg"];
        assert_eq!(
            antenna.zenith_grid.unwrap().start_deg.to_bits(),
            grid["start"].as_f64().unwrap().to_bits()
        );
        assert_eq!(
            antenna.zenith_grid.unwrap().end_deg.to_bits(),
            grid["end"].as_f64().unwrap().to_bits()
        );
        assert_eq!(
            antenna.zenith_grid.unwrap().step_deg.to_bits(),
            grid["step"].as_f64().unwrap().to_bits()
        );

        for frequency_golden in antenna_golden["frequencies"].as_array().unwrap() {
            let frequency = frequency_golden["frequency"].as_str().unwrap();
            let pco = antenna.pco(frequency).expect("pco");
            let pco_golden = &frequency_golden["pco_neu_mm"];
            assert_eq!(
                pco[0].to_bits(),
                (pco_golden["north"].as_f64().unwrap() * 1e-3).to_bits()
            );
            assert_eq!(
                pco[1].to_bits(),
                (pco_golden["east"].as_f64().unwrap() * 1e-3).to_bits()
            );
            assert_eq!(
                pco[2].to_bits(),
                (pco_golden["up"].as_f64().unwrap() * 1e-3).to_bits()
            );

            for sample in frequency_golden["pcv_samples_mm"].as_array().unwrap() {
                let zenith = sample["zenith_deg"].as_f64().unwrap();
                let azimuth = sample["azimuth_deg"].as_f64();
                let got = antenna.pcv(frequency, zenith, azimuth).expect("pcv");
                let want = sample["value"].as_f64().unwrap() * 1e-3;
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    }
}

#[test]
fn encode_round_trips_fixture_through_struct() {
    // The serializer is the inverse of the parser at the canonical-IR level:
    // parse -> encode -> parse must reproduce an equal product, and encoding is
    // deterministic (byte-identical for an equal product).
    let antex = Antex::parse(ATX).expect("parse ANTEX fixture");
    let encoded = antex.encode().expect("encode fixture");
    let reparsed = Antex::parse(&encoded).expect("re-parse encoded ANTEX");
    assert_eq!(antex, reparsed);
    assert_eq!(encoded, reparsed.encode().expect("re-encode fixture"));
    assert_eq!(
        antex.skipped_records(),
        0,
        "the IGS fixture has no malformed grid values"
    );
}

#[test]
fn selects_satellite_antenna_by_prn_and_validity() {
    let antex = Antex::parse(ATX).expect("parse ANTEX fixture");
    let epoch = AntexDateTime::new(2020, 6, 25, 0, 0, 0).unwrap();

    let g05 = antex
        .satellite_antenna("G05", epoch)
        .expect("G05 active antenna");
    assert_eq!(g05.serial, "G05");
    assert_eq!(g05.kind, AntennaKind::Satellite);

    assert!(antex.satellite_antenna("G99", epoch).is_none());
}

#[test]
fn duplicate_satellite_id_selects_epoch_valid_interval() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    fn type_serial(antenna_type: &str, serial: &str) -> String {
        line(
            &format!("{antenna_type:<20}{serial:<20}"),
            "TYPE / SERIAL NO",
        )
    }

    fn block(valid_from: &str, valid_until: &str, pco_north_mm: f64) -> Vec<String> {
        vec![
            line("", "START OF ANTENNA"),
            type_serial("BLOCK TEST", "G01"),
            line("     0.0   0.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line(valid_from, "VALID FROM"),
            line(valid_until, "VALID UNTIL"),
            line("G01", "START OF FREQUENCY"),
            line(
                &format!("{pco_north_mm:8.1}      2.0      3.0"),
                "NORTH / EAST / UP",
            ),
            line("NOAZI    4.0", ""),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
    }

    let text = [
        block(
            "  2020     1     1     0     0    0.0000000",
            "  2020    12    31    23    59   59.0000000",
            1.0,
        ),
        block(
            "  2021     1     1     0     0    0.0000000",
            "  2021    12    31    23    59   59.0000000",
            10.0,
        ),
    ]
    .concat()
    .join("\n");

    let antex = Antex::parse(&text).expect("parse duplicate ANTEX blocks");
    let id = format!("{:<20}{}", "BLOCK TEST", "G01");
    assert_eq!(antex.antenna_intervals(&id).count(), 2);

    let first = antex
        .satellite_antenna("G01", AntexDateTime::new(2020, 6, 1, 0, 0, 0).unwrap())
        .expect("first validity interval");
    assert_eq!(first.pco("G01").unwrap()[0].to_bits(), 0.001_f64.to_bits());

    let second = antex
        .satellite_antenna("G01", AntexDateTime::new(2021, 6, 1, 0, 0, 0).unwrap())
        .expect("second validity interval");
    assert_eq!(second.pco("G01").unwrap()[0].to_bits(), 0.010_f64.to_bits());

    assert!(antex
        .satellite_antenna("G01", AntexDateTime::new(2022, 1, 1, 0, 0, 0).unwrap())
        .is_none());
}

fn invalid_validity(record: &'static str, field: &'static str, value: &str) -> AntexError {
    AntexError::InvalidField {
        antenna_id: Some("TESTANT             TESTSER".to_string()),
        record,
        field,
        value: value.to_string(),
    }
}

fn validity_block(record_body: &str, record: &str) -> String {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }
    [
        line("", "START OF ANTENNA"),
        line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
        line(record_body, record),
        line("", "END OF ANTENNA"),
    ]
    .join("\n")
}

// ANTEX 1.4 states VALID FROM / VALID UNTIL "in GPS time". GPS time has no
// leap-second label, so 23:59:60 names no GPS instant, on a UTC leap-second
// day or any other. The UTC 2016-12-31 23:59:60 label is GPS 2017-01-01
// 00:00:17.
#[test]
fn valid_from_refuses_utc_leap_second_label_in_gps_time() {
    let text = validity_block("  2016    12    31    23    59   60.0000000", "VALID FROM");
    assert_eq!(
        Antex::parse(&text),
        Err(invalid_validity("VALID FROM", "second", "60.0000000"))
    );
    assert_eq!(
        AntexDateTime::new(2016, 12, 31, 23, 59, 60),
        Err(AntexError::InvalidDateTime)
    );
}

#[test]
fn valid_from_rejects_invalid_leap_second_range() {
    for second in ["61.0000000", "-1.0000000"] {
        let text = validity_block(
            &format!("  2016    12    31    23    59   {second}"),
            "VALID FROM",
        );
        assert_eq!(
            Antex::parse(&text),
            Err(invalid_validity("VALID FROM", "second", second))
        );
    }
}

#[test]
fn valid_from_rejects_invalid_civil_date() {
    let text = validity_block("  2026    13    31    23    59    0.0000000", "VALID FROM");
    assert_eq!(
        Antex::parse(&text),
        Err(invalid_validity("VALID FROM", "month", "13"))
    );
}

#[test]
fn pcv_interpolates_zenith_and_azimuth_with_frozen_bits() {
    let antex = Antex::parse(ATX).expect("parse ANTEX fixture");

    let (antenna, frequency) = antex
        .antennas
        .values()
        .find_map(|antenna| {
            antenna
                .frequencies
                .iter()
                .find(|frequency| {
                    frequency
                        .pcv_samples
                        .iter()
                        .any(|sample| sample.grid == PcvGrid::Azimuth)
                })
                .map(|frequency| (antenna, frequency))
        })
        .expect("fixture has azimuth-dependent PCV");

    let mut noazi: Vec<_> = frequency
        .pcv_samples
        .iter()
        .filter(|sample| sample.grid == PcvGrid::NoAzimuth)
        .collect();
    noazi.sort_by(|a, b| a.zenith_deg.total_cmp(&b.zenith_deg));
    let low = noazi[0];
    let high = noazi[1];
    let mid_zenith = (low.zenith_deg + high.zenith_deg) / 2.0;
    let want_mid = low.value_m + (high.value_m - low.value_m) * 0.5;
    let got_mid = antenna
        .pcv(&frequency.frequency, mid_zenith, None)
        .expect("mid zenith pcv");
    assert_eq!(got_mid.to_bits(), want_mid.to_bits());

    let mut azimuths: Vec<f64> = frequency
        .pcv_samples
        .iter()
        .filter_map(|sample| sample.azimuth_deg)
        .collect();
    azimuths.sort_by(|a, b| a.total_cmp(b));
    azimuths.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let az0 = azimuths[0];
    let az1 = azimuths[1];
    let sample_zenith = frequency
        .pcv_samples
        .iter()
        .find(|sample| sample.azimuth_deg == Some(az0))
        .expect("first azimuth sample")
        .zenith_deg;
    let value0 = antenna
        .pcv(&frequency.frequency, sample_zenith, Some(az0))
        .expect("az0 pcv");
    let value1 = antenna
        .pcv(&frequency.frequency, sample_zenith, Some(az1))
        .expect("az1 pcv");
    let az_mid = (az0 + az1) / 2.0;
    let want_az_mid = value0 + (value1 - value0) * 0.5;
    let got_az_mid = antenna
        .pcv(&frequency.frequency, sample_zenith, Some(az_mid))
        .expect("mid azimuth pcv");
    assert_eq!(got_az_mid.to_bits(), want_az_mid.to_bits());

    let wrap_a = antenna
        .pcv(&frequency.frequency, sample_zenith, Some(359.0))
        .expect("359 pcv");
    let wrap_b = antenna
        .pcv(&frequency.frequency, sample_zenith, Some(-1.0))
        .expect("-1 pcv");
    assert_eq!(wrap_a.to_bits(), wrap_b.to_bits());
}

#[test]
fn missing_frequency_is_an_explicit_error() {
    let antex = Antex::parse(ATX).expect("parse ANTEX fixture");
    let antenna = antex.antennas.values().next().expect("fixture antenna");
    let err = antenna.pco("UNKNOWN").expect_err("unknown frequency");
    assert!(err.to_string().contains("unknown frequency"));
}

#[test]
fn frequency_without_pco_row_is_rejected() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("", "START OF ANTENNA"),
        line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
        line("     0.0  90.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("NOAZI    0.0    0.0    0.0", ""),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let err = Antex::parse(&text).expect_err("missing PCO must fail");
    assert!(err.to_string().contains("PCO"));
}

#[test]
fn public_round_trip_preserves_trailing_and_interior_pcv_gaps() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("", "START OF ANTENNA"),
        line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
        line("     5.0", "DAZI"),
        line("     0.0  90.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.00    2.00    3.00            5.00    6.00    7.00    8.00    9.00   10.00   11.00   12.00   13.00   14.00   15.00".to_string(),
        "     0.0    1.10    2.10    3.10            5.10    6.10    7.10    8.10    9.10   10.10   11.10   12.10   13.10   14.10   15.10".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parse grid with trailing blanks");
    assert_eq!(antex.skipped_records(), 0);

    let encoded = antex.encode().expect("encode trailing blanks");
    let noazi = encoded
        .lines()
        .find(|l| l.starts_with("   NOAZI"))
        .expect("NOAZI line present");
    // Exactly 19 fields of 8 chars (152 chars total after 8-char head)
    assert_eq!(noazi[8..].len(), 152);
    // Last four 8-char fields (zeniths 75, 80, 85, 90) must be blank
    assert_eq!(&noazi[128..160], "                                ");

    let reparsed = Antex::parse(&encoded).expect("re-parse encoded grid");
    assert_eq!(antex, reparsed);
    assert_eq!(encoded, reparsed.encode().expect("re-encode grid"));
}

#[test]
fn public_dazi_layout_vs_compact_compatibility_matches() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    fn block(dazi: &str) -> String {
        [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line(dazi, "DAZI"),
            line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI    1.00    2.00    3.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n")
    }

    let std_5 = Antex::parse(&block("     5.0")).unwrap();
    let compact_5 = Antex::parse(&block("5.0")).unwrap();
    let compact_sp_5 = Antex::parse(&block(" 5.0")).unwrap();
    let std_10 = Antex::parse(&block("    10.0")).unwrap();
    let compact_10 = Antex::parse(&block("10.0")).unwrap();
    let std_0 = Antex::parse(&block("     0.0")).unwrap();
    let compact_0 = Antex::parse(&block("0.0")).unwrap();

    let ant_id = "TESTANT             TESTSER";
    assert_eq!(std_5.antenna(ant_id).unwrap().dazi_deg, Some(5.0));
    assert_eq!(compact_5.antenna(ant_id).unwrap().dazi_deg, Some(5.0));
    assert_eq!(compact_sp_5.antenna(ant_id).unwrap().dazi_deg, Some(5.0));

    assert_eq!(std_10.antenna(ant_id).unwrap().dazi_deg, Some(10.0));
    assert_eq!(compact_10.antenna(ant_id).unwrap().dazi_deg, Some(10.0));

    assert_eq!(std_0.antenna(ant_id).unwrap().dazi_deg, Some(0.0));
    assert_eq!(compact_0.antenna(ant_id).unwrap().dazi_deg, Some(0.0));

    // Re-encoding compact inputs produces identical canonical output
    assert_eq!(compact_5.encode().unwrap(), std_5.encode().unwrap());
    assert_eq!(compact_sp_5.encode().unwrap(), std_5.encode().unwrap());
    assert_eq!(compact_10.encode().unwrap(), std_10.encode().unwrap());
    assert_eq!(compact_0.encode().unwrap(), std_0.encode().unwrap());
}

#[test]
fn public_encode_refuses_divergent_public_antennas_mutation() {
    let mut antex = Antex::parse(ATX).expect("parse ANTEX fixture");
    let ant_id = "LEIAR25.R3      LEIT";
    antex
        .antennas
        .get_mut(ant_id)
        .unwrap()
        .zenith_grid
        .as_mut()
        .unwrap()
        .step_deg = 1.0;
    let err = antex
        .encode()
        .expect_err("mutated public antenna must refuse encode");
    match err {
        AntexError::Unwritable { field, reason } => {
            assert_eq!(field, "antennas");
            assert!(reason.contains("diverged"));
        }
        other => panic!("expected Unwritable, got {other:?}"),
    }
}

#[test]
fn public_encode_refuses_removed_or_inserted_public_antenna() {
    let mut antex = Antex::parse(ATX).expect("parse ANTEX fixture");
    antex.antennas.remove("LEIAR25.R3      LEIT");
    let err = antex
        .encode()
        .expect_err("removed public antenna must refuse encode");
    assert!(matches!(
        err,
        AntexError::Unwritable {
            field: "antennas",
            ..
        }
    ));
}

#[test]
fn public_encode_round_trips_large_grid_beyond_65536_cells() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    // Maximum standard grid from 0.0 to 9999.9 with step 0.1 -> exactly 100,000 cells.
    // In IEEE 754: 0.1 * 99999.0 = 9999.900000000001 != 9999.9 (parsed ZEN2).
    let mut noazi_row = String::with_capacity(8 + 100000 * 8);
    noazi_row.push_str("   NOAZI");
    noazi_row.push_str("    1.00");
    noazi_row.push_str("    2.00");
    for _ in 2..99999 {
        noazi_row.push_str("        ");
    }
    noazi_row.push_str("    5.00");

    let text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        line("", "START OF ANTENNA"),
        line("LARGEGRID           TESTSER", "TYPE / SERIAL NO"),
        line("     0.0", "DAZI"),
        line("     0.09999.9   0.1", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        noazi_row,
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parse large grid");
    assert_eq!(antex.skipped_records(), 0);

    let ant = antex
        .antenna("LARGEGRID           TESTSER")
        .expect("parsed antenna");
    assert_eq!(ant.zenith_grid.unwrap().start_deg, 0.0);
    assert_eq!(ant.zenith_grid.unwrap().end_deg, 9999.9);
    assert_eq!(ant.zenith_grid.unwrap().step_deg, 0.1);

    let samples = &ant.frequency("G01").unwrap().pcv_samples;
    assert_eq!(samples.len(), 3);
    let final_sample_zenith = samples[2].zenith_deg;
    let reader_reconstructed = 0.0 + 0.1 * 99999.0;
    assert_eq!(final_sample_zenith, reader_reconstructed);
    assert_eq!(final_sample_zenith, 9999.900000000001);
    assert!(final_sample_zenith > ant.zenith_grid.unwrap().end_deg);
    assert_ne!(
        final_sample_zenith.to_bits(),
        ant.zenith_grid.unwrap().end_deg.to_bits(),
        "final sample parser coordinate differs in bits from decimal ZEN2"
    );

    let encoded = antex
        .encode()
        .expect("encode large grid at maximum F6.1 bounds");
    let noazi = encoded
        .lines()
        .find(|l| l.starts_with("   NOAZI"))
        .expect("NOAZI line present");
    assert_eq!(noazi[8..].len(), 100000 * 8);

    let reparsed = Antex::parse(&encoded).expect("re-parse large grid");
    assert_eq!(antex, reparsed);
    assert_eq!(encoded, reparsed.encode().expect("re-encode large grid"));
}

#[test]
fn public_encode_refuses_sample_beyond_permissible_maximum_grid_index() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    // Grid 9999.8..9999.9 / 0.1 with 3 samples: indices 0 (9999.8), 1 (9999.9),
    // and 2 (10000.0). The permissible maximum index for F6.1 representability is
    // (99999 - 99998) / 1 = 1.
    // Parser accepts all 3 samples without alteration, but serializer refuses sample at index 2.
    let text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        line("", "START OF ANTENNA"),
        line("OVERMAXGRID         TESTSER", "TYPE / SERIAL NO"),
        line("     0.0", "DAZI"),
        line("  9999.89999.9   0.1", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.00    2.00    3.00".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parser accepts row beyond permissible maximum");
    assert_eq!(antex.skipped_records(), 0);
    let ant = antex
        .antenna("OVERMAXGRID         TESTSER")
        .expect("parsed antenna");
    let samples = &ant.frequency("G01").unwrap().pcv_samples;
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[2].zenith_deg, 10000.0);

    let err = antex
        .encode()
        .expect_err("encode beyond permissible maximum must fail");
    assert!(matches!(
        err,
        AntexError::Unwritable {
            field: "pcv_samples",
            ..
        }
    ));
}

#[test]
fn public_encode_round_trips_standard_decimal_step_grids() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    struct DecimalStepCase {
        id: &'static str,
        zen_line: &'static str,
        start: f64,
        end: f64,
        step: f64,
        expected_final_zenith: f64,
    }

    let cases = [
        // Standard grid 0.0..0.3 / 0.1
        // In IEEE 754: 0.1 * 3.0 = 0.30000000000000004 != 0.3 (parsed ZEN2)
        DecimalStepCase {
            id: "DECIMALGRID03       TESTSER",
            zen_line: "     0.0   0.3   0.1",
            start: 0.0,
            end: 0.3,
            step: 0.1,
            expected_final_zenith: 0.30000000000000004,
        },
        // Nonzero decimal start grid 0.1..0.7 / 0.2
        // In IEEE 754: 0.1 + 0.2 * 3.0 = 0.7000000000000001 != 0.7 (parsed ZEN2)
        DecimalStepCase {
            id: "DECIMALGRID07       TESTSER",
            zen_line: "     0.1   0.7   0.2",
            start: 0.1,
            end: 0.7,
            step: 0.2,
            expected_final_zenith: 0.7000000000000001,
        },
    ];

    for case in cases {
        let text = [
            line("     1.4            M", "ANTEX VERSION / SYST"),
            line("", "END OF HEADER"),
            line("", "START OF ANTENNA"),
            line(case.id, "TYPE / SERIAL NO"),
            line("     0.0", "DAZI"),
            line(case.zen_line, "ZEN1 / ZEN2 / DZEN"),
            line("G01", "START OF FREQUENCY"),
            line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
            "   NOAZI    1.00    2.00    3.00    4.00".to_string(),
            line("", "END OF FREQUENCY"),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");

        let antex = Antex::parse(&text).expect("parse standard decimal grid");
        assert_eq!(antex.skipped_records(), 0);
        let ant = antex.antenna(case.id).expect("parsed antenna");
        assert_eq!(ant.zenith_grid.unwrap().start_deg, case.start);
        assert_eq!(ant.zenith_grid.unwrap().end_deg, case.end);
        assert_eq!(ant.zenith_grid.unwrap().step_deg, case.step);

        // Reader arithmetic computes final sample coordinate as start + step * 3.0
        let samples = &ant.frequency("G01").unwrap().pcv_samples;
        assert_eq!(samples.len(), 4);
        let final_sample_zenith = samples[3].zenith_deg;
        assert_eq!(final_sample_zenith, case.start + case.step * 3.0);
        assert_eq!(final_sample_zenith, case.expected_final_zenith);
        assert_ne!(
            final_sample_zenith.to_bits(),
            ant.zenith_grid.unwrap().end_deg.to_bits(),
            "final sample parser coordinate differs in bits from decimal ZEN2"
        );

        let encoded = antex.encode().expect("encode standard decimal grid");
        let reparsed = Antex::parse(&encoded).expect("re-parse standard decimal grid");
        assert_eq!(antex, reparsed);
        assert_eq!(
            encoded,
            reparsed.encode().expect("re-encode standard decimal grid")
        );
    }
}

#[test]
fn public_calibration_method_records_routed_in_header_and_frequency() {
    fn line(prefix: &str, tag: &str) -> String {
        assert!(prefix.len() <= 60);
        format!("{prefix:<60}{tag}")
    }

    // Exercises calibration method records (`METH / BY / # / DATE` and legacy
    // `METH / BY / DATE`) routed by label in antenna header and inside frequency
    // blocks during PCV phase without generating diagnostic skips or corrupting
    // subsequent PCV samples.
    let text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        line("", "START OF ANTENNA"),
        line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
        // Antenna header placement with Table 1 formatted columns
        line(
            "CHAMBER             GEO++                    1    2026-09-22",
            "METH / BY / # / DATE",
        ),
        line("     0.0", "DAZI"),
        line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        // NORTH / EAST / UP transitions frequency into PCV phase
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        // In-frequency compatibility method records during PCV phase
        line(
            "ROBOT               IGS                      1    2026-09-22",
            "METH / BY / # / DATE",
        ),
        line(
            "FIELD               GEO                      1    2026-09-22",
            "METH / BY / DATE",
        ),
        // Subsequent PCV row following method records to verify it survives intact
        "   NOAZI    1.00    2.00    3.00".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parse ANTEX with calibration method records");
    assert_eq!(antex.skipped_records(), 0);

    let ant = antex
        .antenna("TESTANT             TESTSER")
        .expect("parsed antenna");
    let freq = ant.frequency("G01").expect("parsed frequency");

    // Phase transition PCO is intact
    assert_eq!(freq.pco_m, [0.0, 0.0, 0.0]);

    // Verify the real PCV row survived without corruption or spurious sample insertion
    assert_eq!(freq.pcv_samples.len(), 3);
    assert_eq!(freq.pcv_samples[0].grid, PcvGrid::NoAzimuth);
    assert_eq!(freq.pcv_samples[0].zenith_deg, 0.0);
    assert_eq!(freq.pcv_samples[0].value_m, 0.001);
    assert_eq!(freq.pcv_samples[1].zenith_deg, 5.0);
    assert_eq!(freq.pcv_samples[1].value_m, 0.002);
    assert_eq!(freq.pcv_samples[2].zenith_deg, 10.0);
    assert_eq!(freq.pcv_samples[2].value_m, 0.003);

    // All three method records are retained in file order, and the writer
    // restates them in the antenna header.
    let methods: Vec<&str> = ant.calibrations.iter().map(|c| c.method.as_str()).collect();
    assert_eq!(methods, ["CHAMBER", "ROBOT", "FIELD"]);
    assert_eq!(ant.calibrations[0].agency, "GEO++");
    assert_eq!(ant.calibrations[0].antennas_calibrated, Some(1));
    assert_eq!(ant.calibrations[0].date, "2026-09-22");
    let reparsed = Antex::parse(&antex.encode().expect("encode")).expect("re-parse");
    assert_eq!(reparsed, antex);
}

#[test]
fn public_round_trip_preserves_negative_zero_headers_multicell_grid() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        line("", "START OF ANTENNA"),
        line("MULTICELL           TESTSER", "TYPE / SERIAL NO"),
        line("    -0.0", "DAZI"),
        line("    -0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.00    2.00    3.00".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parse multicell grid with negative zero headers");
    assert_eq!(antex.skipped_records(), 0);

    let ant = antex
        .antenna("MULTICELL           TESTSER")
        .expect("parsed multicell antenna");

    // Header values: negative zero preserved exactly via to_bits (ordinary PartialEq treats -0.0 == +0.0)
    assert_eq!(ant.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert!(ant.dazi_deg.unwrap().is_sign_negative());
    assert_eq!(
        ant.zenith_grid.unwrap().start_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(ant.zenith_grid.unwrap().start_deg.is_sign_negative());
    assert_eq!(
        ant.zenith_grid.unwrap().end_deg.to_bits(),
        10.0_f64.to_bits()
    );
    assert_eq!(
        ant.zenith_grid.unwrap().step_deg.to_bits(),
        5.0_f64.to_bits()
    );

    // Reader arithmetic: sample zenith at k=0 is computed as ZEN1 + k * DZEN:
    // in IEEE 754, -0.0 + 5.0 * 0.0 evaluates to +0.0, so the sample coordinate
    // is positive zero rather than retaining negative zero.
    let samples = &ant.frequency("G01").unwrap().pcv_samples;
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].zenith_deg.to_bits(), 0.0_f64.to_bits());
    assert!(samples[0].zenith_deg.is_sign_positive());
    assert_eq!(samples[0].value_m, 0.001);

    assert_eq!(samples[1].zenith_deg.to_bits(), 5.0_f64.to_bits());
    assert_eq!(samples[1].value_m, 0.002);

    assert_eq!(samples[2].zenith_deg.to_bits(), 10.0_f64.to_bits());
    assert_eq!(samples[2].value_m, 0.003);

    let encoded = antex
        .encode()
        .expect("encode multicell grid with negative zero headers");
    let dazi_line = encoded
        .lines()
        .find(|l| l.ends_with("DAZI"))
        .expect("DAZI line present");
    assert_eq!(&dazi_line[..8], "    -0.0");
    assert_eq!(&dazi_line[60..], "DAZI");
    assert!(encoded.contains(&line("    -0.0", "DAZI")));

    let zen_line = encoded
        .lines()
        .find(|l| l.ends_with("ZEN1 / ZEN2 / DZEN"))
        .expect("ZEN1 line present");
    assert_eq!(&zen_line[..20], "    -0.0  10.0   5.0");
    assert_eq!(&zen_line[60..], "ZEN1 / ZEN2 / DZEN");
    assert!(encoded.contains(&line("    -0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN")));

    let reparsed = Antex::parse(&encoded).expect("re-parse encoded multicell block");
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(antex, reparsed);

    let rep_ant = reparsed
        .antenna("MULTICELL           TESTSER")
        .expect("reparsed multicell antenna");
    assert_eq!(rep_ant.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert!(rep_ant.dazi_deg.unwrap().is_sign_negative());
    assert_eq!(
        rep_ant.zenith_grid.unwrap().start_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(rep_ant.zenith_grid.unwrap().start_deg.is_sign_negative());
    assert_eq!(
        rep_ant.zenith_grid.unwrap().end_deg.to_bits(),
        10.0_f64.to_bits()
    );
    assert_eq!(
        rep_ant.zenith_grid.unwrap().step_deg.to_bits(),
        5.0_f64.to_bits()
    );

    let rep_samples = &rep_ant.frequency("G01").unwrap().pcv_samples;
    assert_eq!(rep_samples.len(), 3);
    assert_eq!(rep_samples[0].zenith_deg.to_bits(), 0.0_f64.to_bits());
    assert!(rep_samples[0].zenith_deg.is_sign_positive());
    assert_eq!(rep_samples[0].value_m, 0.001);
    assert_eq!(rep_samples[1].zenith_deg.to_bits(), 5.0_f64.to_bits());
    assert_eq!(rep_samples[1].value_m, 0.002);
    assert_eq!(rep_samples[2].zenith_deg.to_bits(), 10.0_f64.to_bits());
    assert_eq!(rep_samples[2].value_m, 0.003);

    assert_eq!(
        encoded,
        reparsed.encode().expect("re-encode multicell block stably")
    );
}

#[test]
fn public_round_trip_preserves_negative_zero_single_node_and_pco_only() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        // Single-node declared grid covering both zero bounds (start -0.0, end -0.0, step 5.0)
        line("", "START OF ANTENNA"),
        line("SINGLENODE          TESTSER", "TYPE / SERIAL NO"),
        line("    -0.0", "DAZI"),
        line("    -0.0  -0.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.50".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
        // PCO-only compatibility covering DZEN=-0.0 where no PCV data exists
        line("", "START OF ANTENNA"),
        line("PCOONLY             TESTSER", "TYPE / SERIAL NO"),
        line("    -0.0", "DAZI"),
        line("    -0.0  -0.0  -0.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      1.00      2.00      3.00", "NORTH / EAST / UP"),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parse single-node and PCO-only blocks");
    assert_eq!(antex.skipped_records(), 0);

    // 1. Single-node declared grid assertions
    let ant_single = antex
        .antenna("SINGLENODE          TESTSER")
        .expect("parsed single-node antenna");
    assert_eq!(ant_single.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        ant_single.zenith_grid.unwrap().start_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        ant_single.zenith_grid.unwrap().end_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        ant_single.zenith_grid.unwrap().step_deg.to_bits(),
        5.0_f64.to_bits()
    );

    let single_samples = &ant_single.frequency("G01").unwrap().pcv_samples;
    assert_eq!(single_samples.len(), 1);
    // Reader arithmetic: -0.0 + 5.0 * 0.0 = +0.0
    assert_eq!(single_samples[0].zenith_deg.to_bits(), 0.0_f64.to_bits());
    assert!(single_samples[0].zenith_deg.is_sign_positive());
    assert_eq!(single_samples[0].value_m, 0.0015);

    // 2. PCO-only compatibility assertions
    let ant_pco = antex
        .antenna("PCOONLY             TESTSER")
        .expect("parsed PCO-only antenna");
    assert_eq!(ant_pco.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        ant_pco.zenith_grid.unwrap().start_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        ant_pco.zenith_grid.unwrap().end_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        ant_pco.zenith_grid.unwrap().step_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(ant_pco.frequency("G01").unwrap().pcv_samples.is_empty());
    assert_eq!(
        ant_pco.frequency("G01").unwrap().pco_m,
        [0.001, 0.002, 0.003]
    );

    let encoded = antex
        .encode()
        .expect("encode single-node and PCO-only blocks");
    let zen_lines: Vec<&str> = encoded
        .lines()
        .filter(|l| l.ends_with("ZEN1 / ZEN2 / DZEN"))
        .collect();
    assert_eq!(zen_lines.len(), 2);

    let mut antenna_blocks = Vec::new();
    let mut current_block: Option<Vec<&str>> = None;
    for l in encoded.lines() {
        if l.ends_with("START OF ANTENNA") {
            current_block = Some(Vec::new());
        } else if l.ends_with("END OF ANTENNA") {
            if let Some(b) = current_block.take() {
                antenna_blocks.push(b);
            }
        } else if let Some(ref mut b) = current_block {
            b.push(l);
        }
    }
    assert_eq!(antenna_blocks.len(), 2);

    let single_block = antenna_blocks
        .iter()
        .find(|b| {
            b.iter().any(|l| {
                l.starts_with("SINGLENODE          TESTSER") && l.ends_with("TYPE / SERIAL NO")
            })
        })
        .expect("SINGLENODE block present");

    let pco_block = antenna_blocks
        .iter()
        .find(|b| {
            b.iter().any(|l| {
                l.starts_with("PCOONLY             TESTSER") && l.ends_with("TYPE / SERIAL NO")
            })
        })
        .expect("PCOONLY block present");

    let single_type = single_block
        .iter()
        .copied()
        .find(|l| l.ends_with("TYPE / SERIAL NO"))
        .expect("single-node TYPE / SERIAL NO line");
    assert_eq!(
        single_type,
        &line("SINGLENODE          TESTSER", "TYPE / SERIAL NO")
    );
    assert_eq!(&single_type[60..], "TYPE / SERIAL NO");

    let single_dazi = single_block
        .iter()
        .copied()
        .find(|l| l.ends_with("DAZI"))
        .expect("single-node DAZI line");
    assert_eq!(single_dazi, &line("    -0.0", "DAZI"));
    assert_eq!(&single_dazi[..8], "    -0.0");
    assert_eq!(&single_dazi[60..], "DAZI");

    let single_zen = single_block
        .iter()
        .copied()
        .find(|l| l.ends_with("ZEN1 / ZEN2 / DZEN"))
        .expect("single-node ZEN1 / ZEN2 / DZEN line");
    assert_eq!(
        single_zen,
        &line("    -0.0  -0.0   5.0", "ZEN1 / ZEN2 / DZEN")
    );
    assert_eq!(&single_zen[..20], "    -0.0  -0.0   5.0");
    assert_eq!(&single_zen[60..], "ZEN1 / ZEN2 / DZEN");

    let pco_type = pco_block
        .iter()
        .copied()
        .find(|l| l.ends_with("TYPE / SERIAL NO"))
        .expect("PCO-only TYPE / SERIAL NO line");
    assert_eq!(
        pco_type,
        &line("PCOONLY             TESTSER", "TYPE / SERIAL NO")
    );
    assert_eq!(&pco_type[60..], "TYPE / SERIAL NO");

    let pco_dazi = pco_block
        .iter()
        .copied()
        .find(|l| l.ends_with("DAZI"))
        .expect("PCO-only DAZI line");
    assert_eq!(pco_dazi, &line("    -0.0", "DAZI"));
    assert_eq!(&pco_dazi[..8], "    -0.0");
    assert_eq!(&pco_dazi[60..], "DAZI");

    let pco_zen = pco_block
        .iter()
        .copied()
        .find(|l| l.ends_with("ZEN1 / ZEN2 / DZEN"))
        .expect("PCO-only ZEN1 / ZEN2 / DZEN line");
    assert_eq!(pco_zen, &line("    -0.0  -0.0  -0.0", "ZEN1 / ZEN2 / DZEN"));
    assert_eq!(&pco_zen[..20], "    -0.0  -0.0  -0.0");
    assert_eq!(&pco_zen[60..], "ZEN1 / ZEN2 / DZEN");

    assert_ne!(single_zen, pco_zen);
    assert!(encoded.contains(&line("    -0.0", "DAZI")));
    assert!(encoded.contains(&line("    -0.0  -0.0   5.0", "ZEN1 / ZEN2 / DZEN")));
    assert!(encoded.contains(&line("    -0.0  -0.0  -0.0", "ZEN1 / ZEN2 / DZEN")));

    let reparsed =
        Antex::parse(&encoded).expect("re-parse encoded single-node and PCO-only blocks");
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(antex, reparsed);

    let rep_single = reparsed
        .antenna("SINGLENODE          TESTSER")
        .expect("reparsed single-node antenna");
    assert_eq!(rep_single.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        rep_single.zenith_grid.unwrap().start_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        rep_single.zenith_grid.unwrap().end_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        rep_single.zenith_grid.unwrap().step_deg.to_bits(),
        5.0_f64.to_bits()
    );
    let rep_single_samples = &rep_single.frequency("G01").unwrap().pcv_samples;
    assert_eq!(rep_single_samples.len(), 1);
    assert_eq!(
        rep_single_samples[0].zenith_deg.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(rep_single_samples[0].value_m, 0.0015);

    let rep_pco = reparsed
        .antenna("PCOONLY             TESTSER")
        .expect("reparsed PCO-only antenna");
    assert_eq!(rep_pco.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        rep_pco.zenith_grid.unwrap().start_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        rep_pco.zenith_grid.unwrap().end_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        rep_pco.zenith_grid.unwrap().step_deg.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(rep_pco.frequency("G01").unwrap().pcv_samples.is_empty());
    assert_eq!(
        rep_pco.frequency("G01").unwrap().pco_m,
        [0.001, 0.002, 0.003]
    );

    assert_eq!(
        encoded,
        reparsed
            .encode()
            .expect("re-encode single-node and PCO-only stably")
    );

    // Serializer must refuse nonpositive DZEN (including -0.0) when PCV data exists
    let invalid_step_text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        line("", "START OF ANTENNA"),
        line("BADSTEP             TESTSER", "TYPE / SERIAL NO"),
        line("    -0.0", "DAZI"),
        line("    -0.0  -0.0  -0.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.50".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");
    let invalid_antex =
        Antex::parse(&invalid_step_text).expect("parse accepts nonpositive step in header");
    let err = invalid_antex
        .encode()
        .expect_err("serializer must refuse nonpositive DZEN when PCV samples exist");
    assert!(matches!(
        err,
        AntexError::Unwritable {
            field: "zenith_step_deg",
            ..
        }
    ));
}

#[test]
fn public_encode_refuses_public_signed_zero_float_mutations() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("     1.4            M", "ANTEX VERSION / SYST"),
        line("", "END OF HEADER"),
        line("", "START OF ANTENNA"),
        line("SYNCZERO            TESTSER", "TYPE / SERIAL NO"),
        line("     0.0", "DAZI"),
        line("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    0.00    1.00    2.00".to_string(),
        "     0.0    0.10    1.10    2.10".to_string(),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("parse valid representable product with +0.0 floats");
    assert_eq!(antex.skipped_records(), 0);

    // Baseline encode succeeds
    let baseline_encoded = antex.encode().expect("baseline encode succeeds");
    assert!(!baseline_encoded.is_empty());

    let ant_id = "SYNCZERO            TESTSER";

    // 1. Header float regression: change only public antenna's dazi_deg zero to -0.0
    {
        let mut mutated = antex.clone();
        let pub_ant = mutated.antennas.get_mut(ant_id).expect("public antenna");
        assert_eq!(pub_ant.dazi_deg.unwrap().to_bits(), 0.0_f64.to_bits());
        pub_ant.dazi_deg = Some(-0.0);
        assert_eq!(pub_ant.dazi_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_ne!(pub_ant.dazi_deg.unwrap().to_bits(), 0.0_f64.to_bits());

        let err = mutated
            .encode()
            .expect_err("public-only -0.0 DAZI edit must refuse encode rather than silently serialize private state");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "antennas");
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected Unwritable with field antennas, got {other:?}"),
        }
    }

    // 2. Header float regression: zenith_start_deg zero to -0.0
    {
        let mut mutated = antex.clone();
        let pub_ant = mutated.antennas.get_mut(ant_id).expect("public antenna");
        assert_eq!(
            pub_ant.zenith_grid.unwrap().start_deg.to_bits(),
            0.0_f64.to_bits()
        );
        pub_ant.zenith_grid.as_mut().unwrap().start_deg = -0.0;
        assert_eq!(
            pub_ant.zenith_grid.unwrap().start_deg.to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_ne!(
            pub_ant.zenith_grid.unwrap().start_deg.to_bits(),
            0.0_f64.to_bits()
        );

        let err = mutated
            .encode()
            .expect_err("public-only -0.0 zenith_start_deg edit must refuse encode");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "antennas");
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected Unwritable with field antennas, got {other:?}"),
        }
    }

    // 3. Nested float location: PCO component zero to -0.0
    {
        let mut mutated = antex.clone();
        let pub_ant = mutated.antennas.get_mut(ant_id).expect("public antenna");
        let freq = &mut pub_ant.frequencies[0];
        assert_eq!(freq.pco_m[0].to_bits(), 0.0_f64.to_bits());
        freq.pco_m[0] = -0.0;
        assert_eq!(freq.pco_m[0].to_bits(), (-0.0_f64).to_bits());
        assert_ne!(freq.pco_m[0].to_bits(), 0.0_f64.to_bits());

        let err = mutated
            .encode()
            .expect_err("public-only -0.0 PCO edit must refuse encode");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "antennas");
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected Unwritable with field antennas, got {other:?}"),
        }
    }

    // 4. Nested float location: PCV sample value_m zero to -0.0
    {
        let mut mutated = antex.clone();
        let pub_ant = mutated.antennas.get_mut(ant_id).expect("public antenna");
        let freq = &mut pub_ant.frequencies[0];
        let sample = &mut freq.pcv_samples[0];
        assert_eq!(sample.value_m.to_bits(), 0.0_f64.to_bits());
        sample.value_m = -0.0;
        assert_eq!(sample.value_m.to_bits(), (-0.0_f64).to_bits());
        assert_ne!(sample.value_m.to_bits(), 0.0_f64.to_bits());

        let err = mutated
            .encode()
            .expect_err("public-only -0.0 PCV sample value edit must refuse encode");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "antennas");
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected Unwritable with field antennas, got {other:?}"),
        }
    }

    // 5. Nested float location: PCV sample azimuth_deg zero to -0.0
    {
        let mut mutated = antex.clone();
        let pub_ant = mutated.antennas.get_mut(ant_id).expect("public antenna");
        let freq = &mut pub_ant.frequencies[0];
        let sample = freq
            .pcv_samples
            .iter_mut()
            .find(|s| s.grid == PcvGrid::Azimuth)
            .expect("azimuth sample");
        assert_eq!(sample.azimuth_deg, Some(0.0));
        assert_eq!(sample.azimuth_deg.unwrap().to_bits(), 0.0_f64.to_bits());
        sample.azimuth_deg = Some(-0.0);
        assert_eq!(sample.azimuth_deg.unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_ne!(sample.azimuth_deg.unwrap().to_bits(), 0.0_f64.to_bits());

        let err = mutated
            .encode()
            .expect_err("public-only -0.0 PCV sample azimuth edit must refuse encode");
        match err {
            AntexError::Unwritable { field, reason } => {
                assert_eq!(field, "antennas");
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected Unwritable with field antennas, got {other:?}"),
        }
    }
}

// `igs_01_relative_trim.atx`: lines 1-24 and 473-484 of the IGS relative
// model `igs_01.atx` (https://files.igs.org/pub/station/general/igs_01.atx,
// retrieved 2026-09-22, 116100 bytes, sha256
// e053a7de3fb0845421642d263e940c86644eeb8c8b3c4c014ab7573ba383de4d), no line
// altered: the version 1.3 header with PCV TYPE `R` and reference antenna
// `AOAD/M_T`, the `BLOCK I` block, and the `ASH700699.L1    NONE` block whose
// METH record leaves the antenna count blank, through its END OF ANTENNA.
// 3092 bytes, sha256
// 7f6ce5ecdf66beecaad75f67e22c51c55c317561cdef3f7aab9f71c8c1f51d26.
const IGS01_RELATIVE: &str = include_str!("fixtures/antex/igs_01_relative_trim.atx");

// `igs20_block_i_g03_trim.atx`: lines 1-4, 568-569 and 736-753 of `igs20.atx`
// (https://files.igs.org/pub/station/general/igs20.atx, retrieved 2026-09-22,
// 61300492 bytes, sha256
// fdd9cac36186d0a8c23f3fdbac9f3e42e2fc7a1b79e68c7981c3913f39b45f2f), no line
// altered: the version and PCV TYPE records, three of the 566 header comments,
// END OF HEADER, and the `BLOCK I G03` block, whose VALID UNTIL is
// `1994 4 17 23 59 59.9999999`. That file carries 199 VALID FROM/UNTIL records
// with a fractional second, all `59.9999999`. 2040 bytes, sha256
// 5451d52a4a6ce22ad7c6878bf3adc03ac57853bf60a63e73b3e0fd2826f46755.
const IGS20_G03: &str = include_str!("fixtures/antex/igs20_block_i_g03_trim.atx");

fn rec(prefix: &str, tag: &str) -> String {
    format!("{prefix:<60}{tag}")
}

fn trimmed_lines(text: &str) -> Vec<&str> {
    text.lines().map(str::trim_end).collect()
}

fn frequency_labels(text: &str, label: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.get(60..).map(str::trim_end) == Some(label))
        .map(|l| l[..60].trim().to_string())
        .collect()
}

fn one_antenna(records: &[String]) -> String {
    let mut lines = vec![
        rec("", "START OF ANTENNA"),
        rec("TESTANT             TESTSER", "TYPE / SERIAL NO"),
    ];
    lines.extend_from_slice(records);
    lines.push(rec("", "END OF ANTENNA"));
    lines.join("\n")
}

const TEST_ID: &str = "TESTANT             TESTSER";

#[test]
fn relative_pcv_type_reference_antenna_and_metadata_are_retained_and_restated() {
    let antex = Antex::parse(IGS01_RELATIVE).expect("parse igs_01 excerpt");
    assert_eq!(antex.skipped_records(), 0);

    let version = antex.header.version.expect("version record");
    assert_eq!(version.version.to_bits(), 1.3_f64.to_bits());
    assert_eq!(version.system, Some('M'));
    let pcv = antex.header.pcv_type.as_ref().expect("PCV TYPE record");
    assert_eq!(pcv.pcv_type, PcvType::Relative);
    assert_eq!(pcv.reference_antenna_type, "AOAD/M_T");
    assert_eq!(pcv.reference_antenna_serial, "");
    assert_eq!(pcv.reference_antenna(), Some("AOAD/M_T"));
    assert_eq!(antex.header.comments.len(), 5);
    assert_eq!(
        antex.header.comments[1],
        "igs_01.pcv (version from July 2007) converted to ANTEX"
    );

    let block_i = antex.antenna("BLOCK I").expect("BLOCK I");
    assert_eq!(
        block_i.calibrations,
        vec![Calibration {
            method: String::new(),
            agency: String::new(),
            antennas_calibrated: Some(0),
            date: "21-APR-04".to_string(),
        }]
    );
    let ash = antex.antenna("ASH700699.L1    NONE").expect("ASH700699.L1");
    assert_eq!(
        ash.calibrations,
        vec![Calibration {
            method: "FIELD".to_string(),
            agency: "IGEX".to_string(),
            antennas_calibrated: None,
            date: "12-NOV-98".to_string(),
        }]
    );

    // 51.50 mm is read as RTKLIB `decodef` reads it, 51.50 * 1e-3, which is one
    // unit in the last place away from 51.50 / 1000.
    let up = ash.pco("G01").expect("G01 PCO")[2];
    assert_eq!(up.to_bits(), (51.5_f64 * 1e-3).to_bits());
    assert_ne!(up.to_bits(), (51.5_f64 / 1000.0).to_bits());

    // Blocks keep file order (BLOCK I before ASH700699.L1, the reverse of id
    // order), and the writer restates every record the excerpt carries.
    let order: Vec<&str> = antex.antenna_blocks().map(|a| a.id.as_str()).collect();
    assert_eq!(order, ["BLOCK I", "ASH700699.L1    NONE"]);
    let encoded = antex.encode().expect("encode igs_01 excerpt");
    assert_eq!(trimmed_lines(&encoded), trimmed_lines(IGS01_RELATIVE));
    assert_eq!(Antex::parse(&encoded).expect("re-parse"), antex);
}

#[test]
fn absolute_pcv_type_and_fractional_validity_are_retained_and_restated() {
    let antex = Antex::parse(IGS20_G03).expect("parse igs20 excerpt");
    assert_eq!(antex.skipped_records(), 0);
    let pcv = antex.header.pcv_type.as_ref().expect("PCV TYPE record");
    assert_eq!(pcv.pcv_type, PcvType::Absolute);
    assert_eq!(pcv.reference_antenna(), None);

    let g03 = antex
        .antenna("BLOCK I             G03                 G011      1985-093A")
        .expect("G03 block");
    assert_eq!(
        g03.valid_from,
        Some(AntexDateTime::new(1985, 10, 9, 0, 0, 0).unwrap())
    );
    let until = AntexDateTime::new_with_nanosecond(1994, 4, 17, 23, 59, 59, 999_999_900).unwrap();
    assert_eq!(g03.valid_until, Some(until));

    // The bound is the instant 0.1 microsecond before midnight, not 23:59:59.
    let inside = AntexDateTime::new_with_nanosecond(1994, 4, 17, 23, 59, 59, 500_000_000).unwrap();
    let after = AntexDateTime::new_with_nanosecond(1994, 4, 17, 23, 59, 59, 999_999_950).unwrap();
    assert!(g03.valid_at(inside));
    assert!(g03.valid_at(until));
    assert!(!g03.valid_at(after));
    assert!(antex.satellite_antenna("G03", inside).is_some());

    // -2.60 mm reads as -2.60 * 1e-3 (RTKLIB), not -2.60 / 1000.
    let sample = &g03.frequency("G01").unwrap().pcv_samples[1];
    assert_eq!(sample.value_m.to_bits(), (-2.6_f64 * 1e-3).to_bits());
    assert_ne!(sample.value_m.to_bits(), (-2.6_f64 / 1000.0).to_bits());

    let encoded = antex.encode().expect("encode igs20 excerpt");
    assert_eq!(trimmed_lines(&encoded), trimmed_lines(IGS20_G03));
    assert!(encoded.contains(&rec(
        "  1994     4    17    23    59   59.9999999",
        "VALID UNTIL"
    )));
    assert_eq!(Antex::parse(&encoded).expect("re-parse"), antex);
}

/// Build a one-block file whose `VALID FROM` seconds field (columns 31-43)
/// holds `seconds`, right-aligned.
fn valid_from_with_seconds(seconds: &str) -> String {
    one_antenna(&[rec(
        &format!("  2020     1     1     0     0{seconds:>13}"),
        "VALID FROM",
    )])
}

fn read_valid_from(text: &str) -> Antex {
    Antex::parse(text).unwrap_or_else(|err| panic!("{text:?}: {err}"))
}

// F13.7 input holds up to twelve decimals (`.123456789012`) and, as Fortran and
// RTKLIB's sscanf read it, an exponent that moves the point. Every such
// seconds value is kept exactly and written back to the same instant.
#[test]
fn validity_seconds_are_kept_exactly_and_restated() {
    let cases = [
        // (field text, whole second, fraction digits, scale, written text)
        ("0.1234567890", 0, 123_456_789, 9, "0.123456789"),
        ("0.1234567891", 0, 1_234_567_891, 10, "0.1234567891"),
        (".123456789012", 0, 123_456_789_012, 12, ".123456789012"),
        ("59.99D0", 59, 99, 2, "59.9900000"),
        ("5.999E1", 59, 99, 2, "59.9900000"),
        ("1.2345678E-9", 0, 12_345_678, 16, "1.2345678E-9"),
        // Exponent forms always carry a decimal point, so a Fortran reader does
        // not apply the implied seven decimals to them.
        (".123456789E-9", 0, 123_456_789, 18, ".123456789E-9"),
        ("7E-13", 0, 7, 13, "7.E-13"),
        ("12345678E-18", 0, 12_345_678, 18, "1.2345678E-11"),
        ("-0.0000000", 0, 0, 0, "0.0000000"),
    ];
    for (field, second, digits, scale, written) in cases {
        let antex = read_valid_from(&valid_from_with_seconds(field));
        let from = antex.antenna(TEST_ID).unwrap().valid_from.unwrap();
        assert_eq!(from.second, second, "{field}");
        assert_eq!(from.fraction.digits(), digits, "{field}");
        assert_eq!(from.fraction.scale(), scale, "{field}");
        let encoded = antex.encode().expect("encode seconds");
        assert!(
            encoded.contains(&rec(
                &format!("  2020     1     1     0     0{written:>13}"),
                "VALID FROM"
            )),
            "{field}: {encoded}"
        );
        assert_eq!(Antex::parse(&encoded).unwrap(), antex, "{field}");
    }

    // A value whose digits fit the field only without a decimal point is
    // refused by name rather than written in a form Fortran reads differently.
    let antex = read_valid_from(&valid_from_with_seconds("123456789E-99"));
    let from = antex.antenna(TEST_ID).unwrap().valid_from.unwrap();
    assert_eq!(from.fraction.digits(), 123_456_789);
    assert_eq!(from.fraction.scale(), 99);
    assert!(matches!(
        antex.encode(),
        Err(AntexError::Unwritable {
            field: "valid_from",
            ..
        })
    ));

    // 59.99 read from either exponent form is 59 s and 990_000_000 ns.
    let from = read_valid_from(&valid_from_with_seconds("59.99D0"))
        .antenna(TEST_ID)
        .unwrap()
        .valid_from
        .unwrap();
    assert_eq!(from.fraction.nanoseconds(), Some(990_000_000));
    assert_eq!(
        from,
        AntexDateTime::new_with_nanosecond(2020, 1, 1, 0, 0, 59, 990_000_000).unwrap()
    );

    // Fractions order by value whatever their scale.
    let at = |seconds: &str| {
        read_valid_from(&valid_from_with_seconds(seconds))
            .antenna(TEST_ID)
            .unwrap()
            .valid_from
            .unwrap()
    };
    assert!(at("0.1234567891") > at("0.123456789"));
    assert!(at("0.05") < at("0.5"));
    assert!(at("1.2345678E-9") < at("0.000000002"));
    assert!(at("1.2345678E-9") > at("0.000000001"));
    assert_eq!(at("0.50"), at(".5"));
}

#[test]
fn short_or_malformed_validity_records_are_refused_by_field_name() {
    let cases = [
        ("  2020     1", "day", ""),
        ("  2020     1     1     0     0", "second", ""),
        ("2020 1 1 0 0 0.0", "year", "2020 1"),
        (
            "  2020     1     1     0     0   59.99X0",
            "second",
            "59.99X0",
        ),
        (
            "  2020     1     1     0     0  -1.0000000",
            "second",
            "-1.0000000",
        ),
        (
            "  2020     1     1     0     0        5.9E",
            "second",
            "5.9E",
        ),
        ("  2020     1     1    24     0    0.0000000", "hour", "24"),
        ("  2020     2    30     0     0    0.0000000", "day", "30"),
        (
            "  2020     1     1     0    60    0.0000000",
            "minute",
            "60",
        ),
        (
            "2020.0     1     1     0     0    0.0000000",
            "year",
            "2020.0",
        ),
    ];
    for record in ["VALID FROM", "VALID UNTIL"] {
        for (body, field, value) in cases {
            let text = one_antenna(&[rec(body, record)]);
            assert_eq!(
                Antex::parse(&text),
                Err(invalid_validity(record, field, value)),
                "{record} {body:?}"
            );
        }
    }
}

fn grid_records(dazi: &str, zen: &str) -> Vec<String> {
    vec![rec(dazi, "DAZI"), rec(zen, "ZEN1 / ZEN2 / DZEN")]
}

fn frequency_section(label: &str, pco: &str, rows: &[&str]) -> Vec<String> {
    let mut lines = vec![
        rec(&format!("   {label}"), "START OF FREQUENCY"),
        rec(pco, "NORTH / EAST / UP"),
    ];
    lines.extend(rows.iter().map(|row| row.to_string()));
    lines.push(rec(&format!("   {label}"), "END OF FREQUENCY"));
    lines
}

#[test]
fn repeated_frequency_labels_keep_every_section_and_refuse_ambiguous_lookup() {
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.push(rec("     3", "# OF FREQUENCIES"));
    records.extend(frequency_section(
        "G01",
        "      1.00      1.00      1.00",
        &["   NOAZI    1.00    1.00    1.00"],
    ));
    records.extend(frequency_section(
        "G02",
        "      3.00      3.00      3.00",
        &["   NOAZI    3.00    3.00    3.00"],
    ));
    records.extend(frequency_section(
        "G01",
        "      2.00      2.00      2.00",
        &["   NOAZI    2.00    2.00    2.00"],
    ));
    let text = one_antenna(&records);

    let antex = Antex::parse(&text).expect("repeated labels are retained");
    assert_eq!(antex.skipped_records(), 0);
    let antenna = antex.antenna(TEST_ID).unwrap();
    let labels: Vec<&str> = antenna
        .frequencies
        .iter()
        .map(|f| f.frequency.as_str())
        .collect();
    assert_eq!(labels, ["G01", "G02", "G01"]);
    assert_eq!(antenna.frequencies[0].pco_m, [0.001, 0.001, 0.001]);
    assert_eq!(antenna.frequencies[2].pco_m, [0.002, 0.002, 0.002]);
    assert_eq!(
        antenna.pco("G01"),
        Err(AntexError::AmbiguousFrequency {
            antenna_id: TEST_ID.to_string(),
            frequency: "G01".to_string(),
            sections: 2,
        })
    );
    assert!(matches!(
        antenna.pcv("G01", 5.0, None),
        Err(AntexError::AmbiguousFrequency { .. })
    ));
    assert_eq!(antenna.pco("G02").unwrap(), [0.003, 0.003, 0.003]);

    let encoded = antex.encode().expect("encode repeated labels");
    assert_eq!(
        frequency_labels(&encoded, "START OF FREQUENCY"),
        ["G01", "G02", "G01"]
    );
    assert_eq!(Antex::parse(&encoded).unwrap(), antex);

    // Identical sections leave nothing ambiguous.
    let mut same = grid_records("     0.0", "     0.0  10.0   5.0");
    for _ in 0..2 {
        same.extend(frequency_section(
            "G01",
            "      1.00      1.00      1.00",
            &["   NOAZI    1.00    1.00    1.00"],
        ));
    }
    let antex = Antex::parse(&one_antenna(&same)).unwrap();
    assert_eq!(
        antex.antenna(TEST_ID).unwrap().pco("G01").unwrap(),
        [0.001, 0.001, 0.001]
    );
}

// igs20.atx (sha256 fdd9cac3...5f2f) has 217 antenna blocks whose frequency
// sections are not in label order, e.g. every GALILEO-2 block lists E05, E07,
// E06, E01, E08.
#[test]
fn frequency_sections_keep_file_order() {
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    for label in ["E05", "E07", "E06", "E01", "E08"] {
        records.extend(frequency_section(
            label,
            "      0.00      0.00      0.00",
            &["   NOAZI    0.00    0.00    0.00"],
        ));
    }
    let antex = Antex::parse(&one_antenna(&records)).unwrap();
    let labels: Vec<&str> = antex
        .antenna(TEST_ID)
        .unwrap()
        .frequencies
        .iter()
        .map(|f| f.frequency.as_str())
        .collect();
    assert_eq!(labels, ["E05", "E07", "E06", "E01", "E08"]);
    let encoded = antex.encode().unwrap();
    assert_eq!(
        frequency_labels(&encoded, "START OF FREQUENCY"),
        ["E05", "E07", "E06", "E01", "E08"]
    );
    // START and END OF FREQUENCY carry the code in columns 4-6 (`3X,A1,I2`),
    // where RTKLIB reads the system flag at byte 3.
    assert!(encoded.contains(&rec("   E05", "START OF FREQUENCY")));
    assert!(encoded.contains(&rec("   E05", "END OF FREQUENCY")));
    assert_eq!(Antex::parse(&encoded).unwrap(), antex);
}

fn degenerate(result: Result<Antex, AntexError>) -> String {
    match result {
        Err(AntexError::DegenerateGrid {
            antenna_id,
            frequency,
            reason,
        }) => {
            assert_eq!(antenna_id, TEST_ID);
            assert_eq!(frequency, "G01");
            reason
        }
        other => panic!("expected DegenerateGrid, got {other:?}"),
    }
}

#[test]
fn degenerate_zenith_grids_are_refused_by_name() {
    let pco = "      0.00      0.00      0.00";

    // DZEN 0.0 would place all three values at ZEN1.
    let mut records = grid_records("     0.0", "     0.0  10.0   0.0");
    records.extend(frequency_section(
        "G01",
        pco,
        &["   NOAZI    1.00    2.00    3.00"],
    ));
    let reason = degenerate(Antex::parse(&one_antenna(&records)));
    assert!(reason.contains("not positive"), "{reason}");

    // A negative DZEN runs the zeniths below ZEN1.
    let mut records = grid_records("     0.0", "     0.0  10.0  -5.0");
    records.extend(frequency_section("G01", pco, &["   NOAZI    1.00    2.00"]));
    degenerate(Antex::parse(&one_antenna(&records)));

    // A second NOAZI row refills zeniths the first one filled.
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.extend(frequency_section(
        "G01",
        pco,
        &["   NOAZI    1.00    2.00    3.00", "   NOAZI    4.00"],
    ));
    let reason = degenerate(Antex::parse(&one_antenna(&records)));
    assert!(reason.contains("zenith index 0"), "{reason}");

    // A repeated azimuth row does the same within that azimuth.
    let mut records = grid_records("     5.0", "     0.0  10.0   5.0");
    records.extend(frequency_section(
        "G01",
        pco,
        &[
            "   NOAZI    1.00    2.00    3.00",
            "     5.0    1.00    2.00    3.00",
            "     5.0            2.50",
        ],
    ));
    let reason = degenerate(Antex::parse(&one_antenna(&records)));
    assert!(reason.contains("azimuth 5"), "{reason}");

    // A row read before the grid record has no zenith to take.
    let mut records = vec![rec("     0.0", "DAZI")];
    records.extend(frequency_section("G01", pco, &["   NOAZI    1.00    2.00"]));
    records.push(rec("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"));
    let reason = degenerate(Antex::parse(&one_antenna(&records)));
    assert!(reason.contains("precedes"), "{reason}");

    // Rows at 0 and 360 degrees are distinct rows, and DZEN 0.0 with a single
    // value per row places nothing twice.
    let mut records = grid_records("   180.0", "     0.0  10.0   5.0");
    records.extend(frequency_section(
        "G01",
        pco,
        &[
            "   NOAZI    1.00    2.00    3.00",
            "     0.0    1.00    2.00    3.00",
            "   180.0    1.00    2.00    3.00",
            "   360.0    1.00    2.00    3.00",
        ],
    ));
    assert!(Antex::parse(&one_antenna(&records)).is_ok());
    let mut records = grid_records("     0.0", "     0.0   0.0   0.0");
    records.extend(frequency_section("G01", pco, &["   NOAZI    1.00"]));
    assert!(Antex::parse(&one_antenna(&records)).is_ok());
}

#[test]
fn malformed_grid_records_are_refused_by_name() {
    let invalid = |record, field, value: &str| AntexError::InvalidField {
        antenna_id: Some(TEST_ID.to_string()),
        record,
        field,
        value: value.to_string(),
    };
    let text = one_antenna(&grid_records("     0.0", "     0.0  10.0   X.0"));
    assert_eq!(
        Antex::parse(&text),
        Err(invalid("ZEN1 / ZEN2 / DZEN", "dzen", "X.0"))
    );
    let text = one_antenna(&grid_records("     0.0", "     0.0  10.0"));
    assert_eq!(
        Antex::parse(&text),
        Err(invalid("ZEN1 / ZEN2 / DZEN", "dzen", ""))
    );
    let text = one_antenna(&grid_records("", "     0.0  10.0   5.0"));
    assert_eq!(Antex::parse(&text), Err(invalid("DAZI", "dazi", "")));
}

#[test]
fn rms_sections_are_retained_and_written_after_their_frequency() {
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.extend(frequency_section(
        "G01",
        "      1.00      2.00      3.00",
        &["   NOAZI    1.00    2.00    3.00"],
    ));
    records.extend([
        rec("   G01", "START OF FREQ RMS"),
        rec("      0.10      0.20      0.30", "NORTH / EAST / UP"),
        "   NOAZI    0.05    0.06    0.07".to_string(),
        rec("   G01", "END OF FREQ RMS"),
    ]);
    let antex = Antex::parse(&one_antenna(&records)).expect("parse RMS section");
    assert_eq!(antex.skipped_records(), 0);
    let frequency = antex.antenna(TEST_ID).unwrap().frequency("G01").unwrap();
    let rms = frequency.rms.as_ref().expect("RMS section retained");
    assert_eq!(rms.pco_m, Some([0.1 * 1e-3, 0.2 * 1e-3, 0.3 * 1e-3]));
    let values: Vec<f64> = rms.pcv_samples.iter().map(|s| s.value_m).collect();
    assert_eq!(values, [0.05 * 1e-3, 0.06 * 1e-3, 0.07 * 1e-3]);
    let zeniths: Vec<f64> = rms.pcv_samples.iter().map(|s| s.zenith_deg).collect();
    assert_eq!(zeniths, [0.0, 5.0, 10.0]);
    // The values section is unchanged by its RMS section.
    assert_eq!(frequency.pco_m, [0.001, 0.002, 0.003]);
    assert_eq!(frequency.pcv_samples.len(), 3);

    let encoded = antex.encode().expect("encode RMS section");
    let tail: Vec<&str> = encoded
        .lines()
        .skip_while(|l| !l.ends_with("END OF FREQUENCY"))
        .take(5)
        .collect();
    assert_eq!(
        tail,
        [
            rec("   G01", "END OF FREQUENCY"),
            rec("   G01", "START OF FREQ RMS"),
            rec("      0.10      0.20      0.30", "NORTH / EAST / UP"),
            "   NOAZI    0.05    0.06    0.07".to_string(),
            rec("   G01", "END OF FREQ RMS"),
        ]
    );
    assert_eq!(Antex::parse(&encoded).unwrap(), antex);
}

#[test]
fn inconsistent_records_are_counted_not_passed_over() {
    let base = || {
        let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
        records.extend(frequency_section(
            "G01",
            "      0.00      0.00      0.00",
            &["   NOAZI    1.00"],
        ));
        records
    };

    // A line that is no record of the format, outside any frequency section.
    let mut records = base();
    records.insert(0, "stray text".to_string());
    assert_eq!(
        Antex::parse(&one_antenna(&records))
            .unwrap()
            .skipped_records(),
        1
    );

    // A declared frequency count the sections do not match.
    let mut records = base();
    records.push(rec("     2", "# OF FREQUENCIES"));
    assert_eq!(
        Antex::parse(&one_antenna(&records))
            .unwrap()
            .skipped_records(),
        1
    );

    // An end record naming another frequency.
    let mut records = base();
    let end = records.len() - 1;
    records[end] = rec("   G02", "END OF FREQUENCY");
    let antex = Antex::parse(&one_antenna(&records)).unwrap();
    assert_eq!(antex.skipped_records(), 1);
    assert!(antex.antenna(TEST_ID).unwrap().frequency("G01").is_ok());

    // An RMS section with no frequency section of its label.
    let mut records = base();
    records.extend([
        rec("   G02", "START OF FREQ RMS"),
        "   NOAZI    0.05".to_string(),
        rec("   G02", "END OF FREQ RMS"),
    ]);
    assert_eq!(
        Antex::parse(&one_antenna(&records))
            .unwrap()
            .skipped_records(),
        1
    );
}

#[test]
fn repeated_records_with_different_values_are_refused() {
    let from = rec("  2020     1     1     0     0    0.0000000", "VALID FROM");
    let text = one_antenna(&[from.clone(), from.clone()]);
    assert!(Antex::parse(&text).is_ok(), "identical repeat");

    let other = rec("  2021     1     1     0     0    0.0000000", "VALID FROM");
    assert_eq!(
        Antex::parse(&one_antenna(&[from, other])),
        Err(AntexError::RepeatedRecord {
            antenna_id: Some(TEST_ID.to_string()),
            record: "VALID FROM",
        })
    );

    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.push(rec("     5.0", "DAZI"));
    assert_eq!(
        Antex::parse(&one_antenna(&records)),
        Err(AntexError::RepeatedRecord {
            antenna_id: Some(TEST_ID.to_string()),
            record: "DAZI",
        })
    );

    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.extend(frequency_section(
        "G01",
        "      0.00      0.00      0.00",
        &[],
    ));
    records.insert(
        records.len() - 1,
        rec("      1.00      0.00      0.00", "NORTH / EAST / UP"),
    );
    assert_eq!(
        Antex::parse(&one_antenna(&records)),
        Err(AntexError::RepeatedRecord {
            antenna_id: Some(TEST_ID.to_string()),
            record: "NORTH / EAST / UP",
        })
    );
}

#[test]
fn malformed_header_records_are_refused_by_name() {
    let invalid = |record, field, value: &str| AntexError::InvalidField {
        antenna_id: None,
        record,
        field,
        value: value.to_string(),
    };
    let text = [
        rec("     1.4            M", "ANTEX VERSION / SYST"),
        rec("X", "PCV TYPE / REFANT"),
        rec("", "END OF HEADER"),
    ]
    .join("\n");
    assert_eq!(
        Antex::parse(&text),
        Err(invalid("PCV TYPE / REFANT", "pcv type", "X"))
    );
    let text = rec("     abc            M", "ANTEX VERSION / SYST");
    assert_eq!(
        Antex::parse(&text),
        Err(invalid("ANTEX VERSION / SYST", "version", "abc"))
    );
}

#[test]
fn comments_and_block_order_are_retained_in_place_and_nothing_is_invented() {
    let source = [
        rec("header note", "COMMENT"),
        rec("", "END OF HEADER"),
        rec("after the header", "COMMENT"),
        rec("", "START OF ANTENNA"),
        rec("before the type", "COMMENT"),
        rec("ZZTEST              SER", "TYPE / SERIAL NO"),
        rec("     0.0", "DAZI"),
        rec("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        rec("     1", "# OF FREQUENCIES"),
        rec("IGS20", "SINEX CODE"),
        rec("about ZZ", "COMMENT"),
        rec("G01", "START OF FREQUENCY"),
        rec("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        rec("  inside the frequency section", "COMMENT"),
        "   NOAZI    1.00    2.00    3.00".to_string(),
        rec("", "END OF FREQUENCY"),
        rec("", "END OF ANTENNA"),
        rec("between blocks", "COMMENT"),
        // A block with no DAZI, grid, count, SINEX CODE or METH record.
        rec("", "START OF ANTENNA"),
        rec("AATEST              SER", "TYPE / SERIAL NO"),
        rec("   G01", "START OF FREQUENCY"),
        rec("      1.00      2.00      3.00", "NORTH / EAST / UP"),
        rec("   G01", "END OF FREQUENCY"),
        rec("", "END OF ANTENNA"),
        rec("at the end", "COMMENT"),
    ]
    .join("\n");

    let antex = Antex::parse(&source).expect("parse two blocks");
    assert_eq!(antex.skipped_records(), 0);
    assert_eq!(antex.header.comments, ["header note"]);
    assert!(antex.header.end_of_header);
    assert_eq!(antex.header.version, None);
    assert_eq!(antex.header.pcv_type, None);
    let outer = |blocks_before, text: &str| OuterComment {
        blocks_before,
        text: text.to_string(),
    };
    assert_eq!(
        antex.outer_comments,
        [
            outer(0, "after the header"),
            outer(1, "between blocks"),
            outer(2, "at the end"),
        ]
    );
    let zz = antex.antenna("ZZTEST              SER").unwrap();
    assert_eq!(zz.leading_comments, ["before the type"]);
    assert_eq!(zz.comments, ["about ZZ", "  inside the frequency section"]);
    assert!(zz.has_frequency_count);
    let aa = antex.antenna("AATEST              SER").unwrap();
    assert_eq!(aa.dazi_deg, None);
    assert_eq!(aa.zenith_grid, None);
    assert!(!aa.has_frequency_count);
    let order: Vec<&str> = antex.antenna_blocks().map(|a| a.id.as_str()).collect();
    assert_eq!(
        order,
        ["ZZTEST              SER", "AATEST              SER"]
    );

    // Every record is restated where it stood. The only moves are the ones
    // the format prescribes: the comment read inside the frequency section
    // follows SINEX CODE, and the frequency code sits in columns 4-6 of both
    // frequency records. Nothing the source lacks is written.
    let expected = [
        rec("header note", "COMMENT"),
        rec("", "END OF HEADER"),
        rec("after the header", "COMMENT"),
        rec("", "START OF ANTENNA"),
        rec("before the type", "COMMENT"),
        rec("ZZTEST              SER", "TYPE / SERIAL NO"),
        rec("     0.0", "DAZI"),
        rec("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        rec("     1", "# OF FREQUENCIES"),
        rec("IGS20", "SINEX CODE"),
        rec("about ZZ", "COMMENT"),
        rec("  inside the frequency section", "COMMENT"),
        rec("   G01", "START OF FREQUENCY"),
        rec("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.00    2.00    3.00".to_string(),
        rec("   G01", "END OF FREQUENCY"),
        rec("", "END OF ANTENNA"),
        rec("between blocks", "COMMENT"),
        rec("", "START OF ANTENNA"),
        rec("AATEST              SER", "TYPE / SERIAL NO"),
        rec("   G01", "START OF FREQUENCY"),
        rec("      1.00      2.00      3.00", "NORTH / EAST / UP"),
        rec("   G01", "END OF FREQUENCY"),
        rec("", "END OF ANTENNA"),
        rec("at the end", "COMMENT"),
    ];
    let encoded = antex.encode().expect("encode two blocks");
    assert_eq!(encoded, format!("{}\n", expected.join("\n")));
    assert_eq!(Antex::parse(&encoded).unwrap(), antex);
}

#[test]
fn blocks_and_sections_not_closed_by_their_end_records_are_kept_and_reported() {
    let complete = Antex::parse(IGS01_RELATIVE).expect("parse igs_01 excerpt");
    assert_eq!(complete.skipped_records(), 0);

    // The same text without its final END OF ANTENNA: the block the end of the
    // file closes is kept whole and reported once.
    let (truncated, last) = IGS01_RELATIVE
        .trim_end_matches('\n')
        .rsplit_once('\n')
        .expect("more than one line");
    assert_eq!(last.trim(), "END OF ANTENNA");
    let antex = Antex::parse(truncated).expect("parse without the last line");
    assert_eq!(antex.skipped_records(), 1);
    assert_eq!(antex.header, complete.header);
    assert_eq!(antex.outer_comments, complete.outer_comments);
    assert_eq!(antex.antennas, complete.antennas);
    assert!(antex.antenna_blocks().eq(complete.antenna_blocks()));
    assert_eq!(antex.encode().unwrap(), complete.encode().unwrap());

    // A block closed by the next START OF ANTENNA.
    let pco = "      0.00      0.00      0.00";
    let mut block = vec![
        rec("", "START OF ANTENNA"),
        rec("TESTANT             TESTSER", "TYPE / SERIAL NO"),
    ];
    block.extend(grid_records("     0.0", "     0.0  10.0   5.0"));
    block.extend(frequency_section("G01", pco, &["   NOAZI    1.00"]));
    let mut other = block.clone();
    other[1] = rec("OTHERANT            TESTSER", "TYPE / SERIAL NO");
    other.push(rec("", "END OF ANTENNA"));
    let unclosed = [block.clone(), other.clone()].concat().join("\n");
    block.push(rec("", "END OF ANTENNA"));
    let closed = [block, other].concat().join("\n");
    let antex = Antex::parse(&unclosed).unwrap();
    let reference = Antex::parse(&closed).unwrap();
    assert_eq!(antex.skipped_records(), 1);
    assert_eq!(reference.skipped_records(), 0);
    assert_eq!(antex.antennas, reference.antennas);

    // A frequency section closed by the next START OF FREQUENCY, and one
    // closed by END OF ANTENNA.
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.extend([
        rec("   G01", "START OF FREQUENCY"),
        rec(pco, "NORTH / EAST / UP"),
        "   NOAZI    1.00".to_string(),
        rec("   G02", "START OF FREQUENCY"),
        rec(pco, "NORTH / EAST / UP"),
        "   NOAZI    2.00".to_string(),
    ]);
    let antex = Antex::parse(&one_antenna(&records)).unwrap();
    assert_eq!(antex.skipped_records(), 2);
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.extend(frequency_section("G01", pco, &["   NOAZI    1.00"]));
    records.extend(frequency_section("G02", pco, &["   NOAZI    2.00"]));
    let reference = Antex::parse(&one_antenna(&records)).unwrap();
    assert_eq!(reference.skipped_records(), 0);
    assert_eq!(antex.antennas, reference.antennas);
}

#[test]
fn frequency_labels_that_are_not_a1_i2_read_but_are_not_written() {
    let mut records = grid_records("     0.0", "     0.0  10.0   5.0");
    records.extend(frequency_section(
        "G01X",
        "      0.00      0.00      0.00",
        &["   NOAZI    1.00"],
    ));
    let antex = Antex::parse(&one_antenna(&records)).expect("the reader keeps G01X");
    assert_eq!(antex.skipped_records(), 0);
    assert!(antex.antenna(TEST_ID).unwrap().frequency("G01X").is_ok());
    assert!(matches!(
        antex.encode(),
        Err(AntexError::Unwritable {
            field: "frequency",
            ..
        })
    ));
}
