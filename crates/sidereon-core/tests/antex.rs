#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::antex::{AntennaKind, Antex, AntexDateTime, AntexError, PcvGrid};

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
            antenna.dazi_deg.to_bits(),
            antenna_golden["dazi_deg"].as_f64().unwrap().to_bits()
        );

        let grid = &antenna_golden["zenith_grid_deg"];
        assert_eq!(
            antenna.zenith_start_deg.to_bits(),
            grid["start"].as_f64().unwrap().to_bits()
        );
        assert_eq!(
            antenna.zenith_end_deg.to_bits(),
            grid["end"].as_f64().unwrap().to_bits()
        );
        assert_eq!(
            antenna.zenith_step_deg.to_bits(),
            grid["step"].as_f64().unwrap().to_bits()
        );

        for frequency_golden in antenna_golden["frequencies"].as_array().unwrap() {
            let frequency = frequency_golden["frequency"].as_str().unwrap();
            let pco = antenna.pco(frequency).expect("pco");
            let pco_golden = &frequency_golden["pco_neu_mm"];
            assert_eq!(
                pco[0].to_bits(),
                (pco_golden["north"].as_f64().unwrap() / 1000.0).to_bits()
            );
            assert_eq!(
                pco[1].to_bits(),
                (pco_golden["east"].as_f64().unwrap() / 1000.0).to_bits()
            );
            assert_eq!(
                pco[2].to_bits(),
                (pco_golden["up"].as_f64().unwrap() / 1000.0).to_bits()
            );

            for sample in frequency_golden["pcv_samples_mm"].as_array().unwrap() {
                let zenith = sample["zenith_deg"].as_f64().unwrap();
                let azimuth = sample["azimuth_deg"].as_f64();
                let got = antenna.pcv(frequency, zenith, azimuth).expect("pcv");
                let want = sample["value"].as_f64().unwrap() / 1000.0;
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

#[test]
fn valid_from_accepts_utc_leap_second_label() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("", "START OF ANTENNA"),
        line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
        line("  2016    12    31    23    59   60.0000000", "VALID FROM"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let antex = Antex::parse(&text).expect("ANTEX leap-second VALID FROM");
    let antenna = antex.antennas.values().next().expect("parsed antenna");
    assert_eq!(
        antenna.valid_from,
        Some(AntexDateTime::new(2016, 12, 31, 23, 59, 60).unwrap())
    );
}

#[test]
fn valid_from_rejects_invalid_leap_second_range() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    for second in ["61.0000000", "-1.0000000"] {
        let text = [
            line("", "START OF ANTENNA"),
            line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
            line(
                &format!("  2016    12    31    23    59   {second}"),
                "VALID FROM",
            ),
            line("", "END OF ANTENNA"),
        ]
        .join("\n");
        assert_eq!(
            Antex::parse(&text),
            Err(sidereon_core::antex::AntexError::InvalidDateTime)
        );
    }
}

#[test]
fn valid_from_rejects_invalid_civil_date() {
    fn line(prefix: &str, tag: &str) -> String {
        format!("{prefix:<60}{tag}")
    }

    let text = [
        line("", "START OF ANTENNA"),
        line("TESTANT             TESTSER", "TYPE / SERIAL NO"),
        line("  2026    13    31    23    59    0.0000000", "VALID FROM"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");
    assert_eq!(
        Antex::parse(&text),
        Err(sidereon_core::antex::AntexError::InvalidDateTime)
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
                .values()
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
    assert_eq!(std_5.antenna(ant_id).unwrap().dazi_deg, 5.0);
    assert_eq!(compact_5.antenna(ant_id).unwrap().dazi_deg, 5.0);
    assert_eq!(compact_sp_5.antenna(ant_id).unwrap().dazi_deg, 5.0);

    assert_eq!(std_10.antenna(ant_id).unwrap().dazi_deg, 10.0);
    assert_eq!(compact_10.antenna(ant_id).unwrap().dazi_deg, 10.0);

    assert_eq!(std_0.antenna(ant_id).unwrap().dazi_deg, 0.0);
    assert_eq!(compact_0.antenna(ant_id).unwrap().dazi_deg, 0.0);

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
    antex.antennas.get_mut(ant_id).unwrap().zenith_step_deg = 1.0;
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
    assert_eq!(ant.zenith_start_deg, 0.0);
    assert_eq!(ant.zenith_end_deg, 9999.9);
    assert_eq!(ant.zenith_step_deg, 0.1);

    let samples = &ant.frequencies["G01"].pcv_samples;
    assert_eq!(samples.len(), 3);
    let final_sample_zenith = samples[2].zenith_deg;
    let reader_reconstructed = 0.0 + 0.1 * 99999.0;
    assert_eq!(final_sample_zenith, reader_reconstructed);
    assert_eq!(final_sample_zenith, 9999.900000000001);
    assert!(final_sample_zenith > ant.zenith_end_deg);
    assert_ne!(
        final_sample_zenith.to_bits(),
        ant.zenith_end_deg.to_bits(),
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
    let samples = &ant.frequencies["G01"].pcv_samples;
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
        assert_eq!(ant.zenith_start_deg, case.start);
        assert_eq!(ant.zenith_end_deg, case.end);
        assert_eq!(ant.zenith_step_deg, case.step);

        // Reader arithmetic computes final sample coordinate as start + step * 3.0
        let samples = &ant.frequencies["G01"].pcv_samples;
        assert_eq!(samples.len(), 4);
        let final_sample_zenith = samples[3].zenith_deg;
        assert_eq!(final_sample_zenith, case.start + case.step * 3.0);
        assert_eq!(final_sample_zenith, case.expected_final_zenith);
        assert_ne!(
            final_sample_zenith.to_bits(),
            ant.zenith_end_deg.to_bits(),
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
    let freq = ant.frequencies.get("G01").expect("parsed frequency");

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
    assert_eq!(ant.dazi_deg.to_bits(), (-0.0_f64).to_bits());
    assert!(ant.dazi_deg.is_sign_negative());
    assert_eq!(ant.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
    assert!(ant.zenith_start_deg.is_sign_negative());
    assert_eq!(ant.zenith_end_deg.to_bits(), 10.0_f64.to_bits());
    assert_eq!(ant.zenith_step_deg.to_bits(), 5.0_f64.to_bits());

    // Reader arithmetic: sample zenith at k=0 is computed as ZEN1 + k * DZEN:
    // in IEEE 754, -0.0 + 5.0 * 0.0 evaluates to +0.0, so the sample coordinate
    // is positive zero rather than retaining negative zero.
    let samples = &ant.frequencies["G01"].pcv_samples;
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
    assert_eq!(rep_ant.dazi_deg.to_bits(), (-0.0_f64).to_bits());
    assert!(rep_ant.dazi_deg.is_sign_negative());
    assert_eq!(rep_ant.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
    assert!(rep_ant.zenith_start_deg.is_sign_negative());
    assert_eq!(rep_ant.zenith_end_deg.to_bits(), 10.0_f64.to_bits());
    assert_eq!(rep_ant.zenith_step_deg.to_bits(), 5.0_f64.to_bits());

    let rep_samples = &rep_ant.frequencies["G01"].pcv_samples;
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
    assert_eq!(ant_single.dazi_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ant_single.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ant_single.zenith_end_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ant_single.zenith_step_deg.to_bits(), 5.0_f64.to_bits());

    let single_samples = &ant_single.frequencies["G01"].pcv_samples;
    assert_eq!(single_samples.len(), 1);
    // Reader arithmetic: -0.0 + 5.0 * 0.0 = +0.0
    assert_eq!(single_samples[0].zenith_deg.to_bits(), 0.0_f64.to_bits());
    assert!(single_samples[0].zenith_deg.is_sign_positive());
    assert_eq!(single_samples[0].value_m, 0.0015);

    // 2. PCO-only compatibility assertions
    let ant_pco = antex
        .antenna("PCOONLY             TESTSER")
        .expect("parsed PCO-only antenna");
    assert_eq!(ant_pco.dazi_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ant_pco.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ant_pco.zenith_end_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ant_pco.zenith_step_deg.to_bits(), (-0.0_f64).to_bits());
    assert!(ant_pco.frequencies["G01"].pcv_samples.is_empty());
    assert_eq!(ant_pco.frequencies["G01"].pco_m, [0.001, 0.002, 0.003]);

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
    assert_eq!(rep_single.dazi_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rep_single.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rep_single.zenith_end_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rep_single.zenith_step_deg.to_bits(), 5.0_f64.to_bits());
    let rep_single_samples = &rep_single.frequencies["G01"].pcv_samples;
    assert_eq!(rep_single_samples.len(), 1);
    assert_eq!(
        rep_single_samples[0].zenith_deg.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(rep_single_samples[0].value_m, 0.0015);

    let rep_pco = reparsed
        .antenna("PCOONLY             TESTSER")
        .expect("reparsed PCO-only antenna");
    assert_eq!(rep_pco.dazi_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rep_pco.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rep_pco.zenith_end_deg.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rep_pco.zenith_step_deg.to_bits(), (-0.0_f64).to_bits());
    assert!(rep_pco.frequencies["G01"].pcv_samples.is_empty());
    assert_eq!(rep_pco.frequencies["G01"].pco_m, [0.001, 0.002, 0.003]);

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
        assert_eq!(pub_ant.dazi_deg.to_bits(), 0.0_f64.to_bits());
        pub_ant.dazi_deg = -0.0;
        assert_eq!(pub_ant.dazi_deg.to_bits(), (-0.0_f64).to_bits());
        assert_ne!(pub_ant.dazi_deg.to_bits(), 0.0_f64.to_bits());

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
        assert_eq!(pub_ant.zenith_start_deg.to_bits(), 0.0_f64.to_bits());
        pub_ant.zenith_start_deg = -0.0;
        assert_eq!(pub_ant.zenith_start_deg.to_bits(), (-0.0_f64).to_bits());
        assert_ne!(pub_ant.zenith_start_deg.to_bits(), 0.0_f64.to_bits());

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
        let freq = pub_ant.frequencies.get_mut("G01").expect("frequency G01");
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
        let freq = pub_ant.frequencies.get_mut("G01").expect("frequency G01");
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
        let freq = pub_ant.frequencies.get_mut("G01").expect("frequency G01");
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
