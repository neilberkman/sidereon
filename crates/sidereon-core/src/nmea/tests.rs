use super::*;

const GGA_SAMPLE: &str = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n";

#[test]
fn parses_gga_sentence_fields() {
    let parsed = parse_sentence(GGA_SAMPLE).expect("parse GGA");
    assert!(parsed.diagnostics.is_empty());
    assert_eq!(
        parsed.value.talker,
        NmeaTalker::System(crate::GnssSystem::Gps)
    );
    let NmeaBody::Gga(gga) = parsed.value.body else {
        panic!("expected GGA");
    };
    assert_eq!(
        gga.time,
        Some(NmeaTime {
            hour: 12,
            minute: 35,
            second: 19,
            nanos: 0,
            decimals: 0,
        })
    );
    assert_eq!(gga.quality, Some(GgaQuality::GpsSps));
    assert_eq!(gga.satellites_used, Some(8));
    assert_eq!(gga.hdop, Some(0.9));
    assert_eq!(gga.altitude_msl_m, Some(545.4));
    assert_eq!(gga.geoid_separation_m, Some(46.9));
    assert_eq!(gga.latitude.expect("latitude").degrees, 48);
    assert_eq!(gga.longitude.expect("longitude").degrees, 11);
}

#[test]
fn rejects_checksum_mismatch_before_decoding() {
    let error = parse_sentence("$GPGGA,123519,4807.038,N,01131.000,E,9,08,0.9,545.4,M,46.9,M,,*47")
        .expect_err("checksum mismatch");
    assert!(matches!(error, NmeaError::ChecksumMismatch { .. }));

    let parsed =
        parse_nmea_str("$GPGGA,123519,4807.038,N,01131.000,E,9,08,0.9,545.4,M,46.9,M,,*47\n");
    assert!(parsed.value.sentences.is_empty());
    assert_eq!(
        parsed.diagnostics.skips[0].reason,
        SkipReason::InconsistentRecord("checksum mismatch")
    );
}

#[test]
fn rejects_every_wrong_checksum_byte_before_decoding() {
    let body = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,";
    let computed = body.bytes().fold(0, |acc, byte| acc ^ byte);
    for stated in 0..=u8::MAX {
        if stated == computed {
            continue;
        }
        let sentence = format!("${body}*{stated:02X}");
        let error = parse_sentence(&sentence).expect_err("checksum mismatch");
        assert!(matches!(
            error,
            NmeaError::ChecksumMismatch {
                computed: got_computed,
                stated: got_stated,
            } if got_computed == computed && got_stated == stated
        ));
    }
}

#[test]
fn unsupported_and_proprietary_sentences_are_typed_skips() {
    let parsed = parse_nmea_str("$GPTXT,01,01,02,message*26\n$PUBX,00*33\n");
    assert!(parsed.value.sentences.is_empty());
    assert_eq!(parsed.diagnostics.skips.len(), 2);
    assert_eq!(
        parsed.diagnostics.skips[0].reason,
        SkipReason::UnsupportedRecordType("unsupported sentence type")
    );
    assert_eq!(
        parsed.diagnostics.skips[1].reason,
        SkipReason::UnsupportedRecordType("proprietary sentence")
    );
}

#[test]
fn encapsulated_and_malformed_checksum_have_specific_skips() {
    let parsed = parse_nmea_str("!AIVDM,1,1,,A,15Muq?002>G?svP00<:O?vN60<0u,0\n$GPGGA,1*ZZ\n");
    assert!(parsed.value.sentences.is_empty());
    assert_eq!(
        parsed.diagnostics.skips[0].reason,
        SkipReason::UnsupportedRecordType("encapsulated sentence")
    );
    assert_eq!(
        parsed.diagnostics.skips[1].reason,
        SkipReason::InconsistentRecord("malformed checksum")
    );
}

#[test]
fn parses_core_sentence_set_and_resolution_tables() {
    let text = "$GPRMC,123520,A,4807.038,N,01131.000,E,22.4,84.4,230394,3.1,W,A,S*72\n\
        $GNGSA,A,3,01,02,03,04,,,,,,,,,1.5,0.9,1.2,1*3B\n\
        $GPGSV,2,1,05,01,45,083,42,02,17,308,40,03,25,120,39,04,10,200,35,1*68\n\
        $GPGSV,2,2,05,05,05,010,30,,,,,,,,,1*53\n\
        $GPGST,123520,1.2,3.4,2.3,45.0,0.5,0.6,0.7*4E\n\
        $GPVTG,84.4,T,83.1,M,22.4,N,41.5,K,A*25\n\
        $GPGLL,4807.038,N,01131.000,E,123520,A,A*42\n\
        $GPZDA,123520,23,03,1994,00,00*48\n";
    let parsed = parse_nmea_str(text);
    assert!(
        parsed.diagnostics.skips.is_empty(),
        "{:?}",
        parsed.diagnostics
    );
    assert_eq!(parsed.value.sentences.len(), 8);

    let NmeaBody::Rmc(rmc) = &parsed.value.sentences[0].body else {
        panic!("expected RMC");
    };
    assert_eq!(
        rmc.date,
        Some(NmeaDate {
            year: 1994,
            month: 3,
            day: 23,
        })
    );
    assert_eq!(rmc.magnetic_variation_deg, Some(-3.1));

    let NmeaBody::Gsa(gsa) = &parsed.value.sentences[1].body else {
        panic!("expected GSA");
    };
    assert_eq!(gsa.system, Some(crate::GnssSystem::Gps));
    assert_eq!(gsa.satellites[0].resolved.unwrap().to_string(), "G01");

    let NmeaBody::Gsv(gsv) = &parsed.value.sentences[2].body else {
        panic!("expected GSV");
    };
    assert_eq!(
        gsv.signal.unwrap().carrier_band(),
        Some(crate::frequencies::CarrierBand::L1)
    );
    assert_eq!(gsv.satellites[0].sat_number.unwrap().raw, 1);

    let epochs = group_epochs(&parsed.value);
    assert_eq!(epochs.len(), 1);
    assert_eq!(epochs[0].gsa.len(), 1);
    assert_eq!(epochs[0].gsv.len(), 1);
    assert!(epochs[0].gsv[0].complete);
    assert_eq!(epochs[0].satellites_in_view(), 5);
    assert_eq!(epochs[0].hdop(), Some(0.9));
    assert_eq!(epochs[0].date.unwrap().year, 1994);
}

#[test]
fn missing_checksum_decodes_with_warning() {
    let parsed = parse_sentence("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,")
        .expect("parse without checksum");
    assert_eq!(parsed.diagnostics.warnings.len(), 1);
    assert_eq!(
        parsed.diagnostics.warnings[0].kind,
        WarningKind::MissingMetadata
    );
}

#[test]
fn write_gga_round_trips_and_emits_checksum() {
    let gga = Gga {
        time: Some(NmeaTime {
            hour: 12,
            minute: 35,
            second: 19,
            nanos: 0,
            decimals: 2,
        }),
        latitude: Some(NmeaCoordinate::parse("4807.038", "N", true).unwrap()),
        longitude: Some(NmeaCoordinate::parse("01131.000", "E", false).unwrap()),
        quality: Some(GgaQuality::GpsSps),
        satellites_used: Some(8),
        hdop: Some(0.9),
        altitude_msl_m: Some(545.4),
        geoid_separation_m: Some(46.9),
        differential_age_s: None,
        differential_station_id: None,
    };
    let sentence = write_gga(NmeaTalker::System(crate::GnssSystem::Gps), &gga).unwrap();
    assert_eq!(
        sentence,
        "$GPGGA,123519.00,4807.038,N,01131.000,E,1,08,0.90,545.4,M,46.9,M,,*59\r\n"
    );
    let round_trip = parse_sentence(&sentence).expect("parse written GGA");
    let NmeaBody::Gga(parsed_gga) = round_trip.value.body else {
        panic!("expected GGA");
    };
    assert_eq!(parsed_gga, gga);
    assert_eq!(
        write_gga(round_trip.value.talker, &parsed_gga).unwrap(),
        sentence
    );
}

#[test]
fn write_gga_rejects_time_without_exact_two_decimal_contract() {
    let mut gga = Gga {
        time: Some(NmeaTime {
            hour: 12,
            minute: 35,
            second: 19,
            nanos: 0,
            decimals: 0,
        }),
        latitude: Some(NmeaCoordinate::parse("4807.038", "N", true).unwrap()),
        longitude: Some(NmeaCoordinate::parse("01131.000", "E", false).unwrap()),
        quality: Some(GgaQuality::GpsSps),
        satellites_used: Some(8),
        hdop: Some(0.9),
        altitude_msl_m: Some(545.4),
        geoid_separation_m: Some(46.9),
        differential_age_s: None,
        differential_station_id: None,
    };
    let error = write_gga(NmeaTalker::System(crate::GnssSystem::Gps), &gga)
        .expect_err("decimal count is part of writer contract");
    assert!(matches!(
        error,
        NmeaError::InvalidInput {
            field: "time",
            reason: "GGA writer requires NmeaTime.decimals == 2",
        }
    ));

    gga.time.as_mut().unwrap().decimals = 2;
    gga.time.as_mut().unwrap().nanos = 123_000_000;
    let error = write_gga(NmeaTalker::System(crate::GnssSystem::Gps), &gga)
        .expect_err("centisecond grid is part of writer contract");
    assert!(matches!(
        error,
        NmeaError::InvalidInput {
            field: "time",
            reason: "GGA writer emits exactly two fractional decimals",
        }
    ));
}

#[test]
fn coordinate_degrees_f64_keeps_expected_bit_pattern() {
    let latitude = NmeaCoordinate::parse("4807.038", "N", true).unwrap();
    let longitude = NmeaCoordinate::parse("01131.000", "W", false).unwrap();
    assert_eq!(latitude.degrees_f64().to_bits(), 0x4048_0f03_afb7_e910);
    assert_eq!(longitude.degrees_f64().to_bits(), 0xc027_0888_8888_8889);
}

#[test]
fn vrs_position_carries_coordinate_rounding_into_next_degree() {
    let position = crate::Wgs84Geodetic::new(
        (12.0_f64 + 59.999_9 / 60.0).to_radians(),
        -(1.0_f64 + 59.999_9 / 60.0).to_radians(),
        10.0,
    )
    .unwrap();
    let gga = Gga::vrs_position(
        position,
        NmeaTime {
            hour: 1,
            minute: 2,
            second: 3,
            nanos: 450_000_000,
            decimals: 2,
        },
        GgaQuality::GpsSps,
        8,
        0.9,
        2,
    )
    .unwrap();
    let latitude = gga.latitude.unwrap();
    let longitude = gga.longitude.unwrap();
    assert_eq!((latitude.degrees, latitude.minutes_scaled), (13, 0));
    assert_eq!((longitude.degrees, longitude.minutes_scaled), (2, 0));
    assert!(!latitude.negative);
    assert!(longitude.negative);
}

#[test]
fn accumulator_splits_gga_epochs_by_exact_time() {
    let log = parse_nmea_str(
        "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\n\
         $GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*4D\n",
    );
    assert!(log.diagnostics.is_empty());
    let epochs = group_epochs(&log.value);
    assert_eq!(epochs.len(), 2);
    assert_eq!(epochs[0].time_of_day.expect("time").second, 19);
    assert_eq!(epochs[1].time_of_day.expect("time").second, 20);
}

#[test]
fn accumulator_reports_duplicate_budget_and_midnight_diagnostics() {
    let first =
        parse_sentence("$GPGGA,123519.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*69")
            .unwrap()
            .value;
    let next =
        parse_sentence("$GPGGA,123520.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*63")
            .unwrap()
            .value;
    let mut accumulator = NmeaAccumulator::new();
    assert!(accumulator.push(&first).is_none());
    assert!(accumulator.push(&first).is_none());
    let completed = accumulator.push(&next).unwrap();
    assert_eq!(completed.diagnostics.warnings.len(), 1);

    let mut accumulator = NmeaAccumulator::new().with_max_sentences_per_epoch(16);
    for _ in 0..16 {
        assert!(accumulator.push(&first).is_none());
    }
    let completed = accumulator.push(&first).unwrap();
    assert!(!completed.diagnostics.warnings.is_empty());

    let late = parse_sentence("$GPGGA,235959,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*4B")
        .unwrap()
        .value;
    let early = parse_sentence("$GPGGA,000000,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*4A")
        .unwrap()
        .value;
    let date = NmeaDate {
        year: 2026,
        month: 7,
        day: 2,
    };
    let mut accumulator = NmeaAccumulator::with_date(date);
    accumulator.push(&late);
    let completed = accumulator.push(&early).unwrap();
    assert_eq!(completed.date, Some(date));
    assert_eq!(accumulator.finish().unwrap().date.unwrap().day, 3);
}

#[test]
fn push_bytes_is_bounded_line_numbered_and_split_independent() {
    let mut accumulator = NmeaAccumulator::new();
    let output = accumulator.push_bytes(&vec![b'X'; 1100]);
    assert_eq!(accumulator.retained_len(), 0);
    assert_eq!(output.diagnostics.skips[0].at.line, Some(1));
    assert_eq!(
        output.diagnostics.skips[0].reason,
        SkipReason::InconsistentRecord("sentence over length cap")
    );

    let text = b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n$GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*4D\r\n";
    let mut whole = NmeaAccumulator::new();
    let whole_out = whole.push_bytes(text);
    let whole_tail = whole.finish();

    let mut split = NmeaAccumulator::new();
    let mut split_sentences = Vec::new();
    let mut split_snapshots = Vec::new();
    for chunk in text.chunks(7) {
        let out = split.push_bytes(chunk);
        split_sentences.extend(out.sentences);
        split_snapshots.extend(out.snapshots);
    }
    let split_tail = split.finish();
    assert_eq!(whole_out.sentences, split_sentences);
    assert_eq!(whole_out.snapshots, split_snapshots);
    assert_eq!(whole_tail, split_tail);
}

#[test]
fn finish_with_output_preserves_final_parse_skips_and_open_epoch() {
    let mut accumulator = NmeaAccumulator::new();
    accumulator.push_bytes(b"bad line");

    let output = accumulator.finish_with_output();
    assert_eq!(output.snapshots.len(), 0);
    assert_eq!(output.sentences.len(), 0);
    assert_eq!(output.diagnostics.skips.len(), 1);
    assert_eq!(output.diagnostics.skips[0].at.line, Some(1));
    assert_eq!(
        output.diagnostics.skips[0].reason,
        SkipReason::UnknownBlock("no NMEA start delimiter".to_string())
    );
    assert_eq!(accumulator.retained_len(), 0);
}

#[test]
fn finish_with_output_keeps_prior_and_final_epochs_and_final_warning() {
    let mut accumulator = NmeaAccumulator::new();
    let first = b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n";
    let final_line = b"$GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,";
    assert!(accumulator.push_bytes(first).snapshots.is_empty());
    accumulator.push_bytes(final_line);

    let output = accumulator.finish_with_output();
    assert_eq!(output.snapshots.len(), 2);
    assert_eq!(output.sentences.len(), 1);
    assert_eq!(
        output.snapshots[0].time_of_day,
        Some(NmeaTime {
            hour: 12,
            minute: 35,
            second: 19,
            nanos: 0,
            decimals: 0,
        })
    );
    assert_eq!(
        output.snapshots[1].time_of_day,
        Some(NmeaTime {
            hour: 12,
            minute: 35,
            second: 20,
            nanos: 0,
            decimals: 0,
        })
    );
    assert_eq!(output.diagnostics.warnings.len(), 1);
    assert_eq!(output.diagnostics.warnings[0].at.line, Some(2));
    assert_eq!(
        output.diagnostics.warnings[0].kind,
        WarningKind::MissingMetadata
    );
}

#[test]
fn finish_with_output_without_remainder_is_idempotent_and_legacy_finish_keeps_contract() {
    let mut accumulator = NmeaAccumulator::new();
    accumulator.push_bytes(GGA_SAMPLE.as_bytes());
    let output = accumulator.finish_with_output();
    assert_eq!(output.snapshots.len(), 1);
    assert!(output.sentences.is_empty());
    assert!(output.diagnostics.is_empty());
    assert!(accumulator.finish_with_output().snapshots.is_empty());

    let mut legacy = NmeaAccumulator::new();
    legacy.push_bytes(GGA_SAMPLE.as_bytes());
    assert!(legacy.finish().is_some());
    assert!(legacy.finish().is_none());

    let mut legacy_with_unterminated_epoch = NmeaAccumulator::new();
    legacy_with_unterminated_epoch
        .push_bytes(b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n");
    legacy_with_unterminated_epoch
        .push_bytes(b"$GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    assert_eq!(
        legacy_with_unterminated_epoch
            .finish()
            .and_then(|epoch| epoch.time_of_day),
        Some(NmeaTime {
            hour: 12,
            minute: 35,
            second: 20,
            nanos: 0,
            decimals: 0,
        })
    );
}
