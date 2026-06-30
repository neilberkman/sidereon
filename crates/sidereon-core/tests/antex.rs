#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::antex::{AntennaKind, Antex, AntexDateTime, PcvGrid};

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
    let encoded = antex.encode();
    let reparsed = Antex::parse(&encoded).expect("re-parse encoded ANTEX");
    assert_eq!(antex, reparsed);
    assert_eq!(encoded, reparsed.encode());
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
            line("     0.0      0.0      5.0", "ZEN1 / ZEN2 / DZEN"),
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
        line("     0.0     90.0      5.0", "ZEN1 / ZEN2 / DZEN"),
        line("G01", "START OF FREQUENCY"),
        line("NOAZI    0.0    0.0    0.0", ""),
        line("", "END OF FREQUENCY"),
        line("", "END OF ANTENNA"),
    ]
    .join("\n");

    let err = Antex::parse(&text).expect_err("missing PCO must fail");
    assert!(err.to_string().contains("PCO"));
}
