#![cfg(sidereon_repo_tests)]
//! CCSDS TDM KVN validation against CCSDS 503.0-B-2 Annex E examples.
//!
//! The source oracle is the public CCSDS 503.0-B-2 Cor. 1 "Tracking Data
//! Message" publication, Annex E. Expected values below are copied from the
//! Annex E KVN examples, then checked as exact decimal tokens and as `f64` bit
//! equality after parsing.

use sha2::{Digest, Sha256};
use sidereon_core::astro::tdm::{
    self, Tdm, TdmDataRecord, TdmError, TdmField, TdmInputErrorKind, TdmObservable, TdmUnit,
};

const ANNEX_E_ALL_KVN: &[(&str, &str, usize, usize)] = &[
    ("E-1", include_str!("fixtures/tdm/annex_e_01.kvn"), 1, 31),
    ("E-2", include_str!("fixtures/tdm/annex_e_02.kvn"), 1, 42),
    ("E-3", include_str!("fixtures/tdm/annex_e_03.kvn"), 1, 50),
    ("E-4", include_str!("fixtures/tdm/annex_e_04.kvn"), 1, 43),
    ("E-5", include_str!("fixtures/tdm/annex_e_05.kvn"), 1, 41),
    ("E-6", include_str!("fixtures/tdm/annex_e_06.kvn"), 1, 40),
    ("E-7", include_str!("fixtures/tdm/annex_e_07.kvn"), 3, 6),
    ("E-8", include_str!("fixtures/tdm/annex_e_08.kvn"), 2, 31),
    ("E-9", include_str!("fixtures/tdm/annex_e_09.kvn"), 1, 41),
    ("E-10", include_str!("fixtures/tdm/annex_e_10.kvn"), 1, 20),
    ("E-11", include_str!("fixtures/tdm/annex_e_11.kvn"), 3, 6),
    ("E-12", include_str!("fixtures/tdm/annex_e_12.kvn"), 1, 14),
    ("E-13", include_str!("fixtures/tdm/annex_e_13.kvn"), 2, 24),
    ("E-14", include_str!("fixtures/tdm/annex_e_14.kvn"), 1, 39),
    ("E-15", include_str!("fixtures/tdm/annex_e_15.kvn"), 3, 21),
    ("E-16", include_str!("fixtures/tdm/annex_e_16.kvn"), 2, 18),
    ("E-17", include_str!("fixtures/tdm/annex_e_17.kvn"), 1, 15),
    ("E-18", include_str!("fixtures/tdm/annex_e_18.kvn"), 2, 20),
    ("E-19", include_str!("fixtures/tdm/annex_e_19.kvn"), 1, 16),
    ("E-20", include_str!("fixtures/tdm/annex_e_20.kvn"), 1, 16),
    ("E-22", include_str!("fixtures/tdm/annex_e_21.kvn"), 1, 9),
];

const PUBLISHED_ANNEX_SAMPLE_VALUES: &[(&str, &str, &str, &str, &str)] = &[
    (
        "E-1",
        include_str!("fixtures/tdm/annex_e_01.kvn"),
        "TRANSMIT_FREQ_2",
        "2005-159T17:41:00",
        "32023442781.733",
    ),
    (
        "E-2",
        include_str!("fixtures/tdm/annex_e_02.kvn"),
        "RECEIVE_FREQ_1",
        "2005-159T17:41:00",
        "-409.2735",
    ),
    (
        "E-3",
        include_str!("fixtures/tdm/annex_e_03.kvn"),
        "RECEIVE_FREQ_1",
        "2005-184T13:59:43.27",
        "8429749418.986191",
    ),
    (
        "E-4",
        include_str!("fixtures/tdm/annex_e_04.kvn"),
        "PR_N0",
        "2005-191T00:31:51",
        "28.52538",
    ),
    (
        "E-5",
        include_str!("fixtures/tdm/annex_e_05.kvn"),
        "RECEIVE_FREQ_3",
        "2005-184T13:59:27.27",
        "8429753135.986102",
    ),
    (
        "E-6",
        include_str!("fixtures/tdm/annex_e_06.kvn"),
        "RECEIVE_FREQ",
        "1998-06-10T00:57:44",
        "2287487999.0",
    ),
    (
        "E-7",
        include_str!("fixtures/tdm/annex_e_07.kvn"),
        "RECEIVE_FREQ_1",
        "2006-347T06:17:49",
        "2299322650.01",
    ),
    (
        "E-8",
        include_str!("fixtures/tdm/annex_e_08.kvn"),
        "RANGE",
        "2007-08-29T12:00:02.000",
        "2.81439006334980E+04",
    ),
    (
        "E-9",
        include_str!("fixtures/tdm/annex_e_09.kvn"),
        "RANGE",
        "2005-09-17T00:42:58.000000",
        "3270.46440460551",
    ),
    (
        "E-10",
        include_str!("fixtures/tdm/annex_e_10.kvn"),
        "RECEIVE_FREQ",
        "2003-07-08T04:45:25.0000",
        "8.738750457763670E+00",
    ),
    (
        "E-11",
        include_str!("fixtures/tdm/annex_e_11.kvn"),
        "VLBI_DELAY",
        "2004-136T15:52:00.0000",
        "-1.911896106591159E-03",
    ),
    (
        "E-12",
        include_str!("fixtures/tdm/annex_e_12.kvn"),
        "ANGLE_2",
        "2004-216T07:45:00",
        "-71.93750",
    ),
    (
        "E-13",
        include_str!("fixtures/tdm/annex_e_13.kvn"),
        "STEC",
        "2005-281T00:00:00",
        "22.2",
    ),
    (
        "E-14",
        include_str!("fixtures/tdm/annex_e_14.kvn"),
        "RHUMIDITY",
        "2005-156T00:03:00",
        "12.0",
    ),
    (
        "E-15",
        include_str!("fixtures/tdm/annex_e_15.kvn"),
        "CLOCK_DRIFT",
        "2005-144T12:00:00",
        "8.102e-14",
    ),
    (
        "E-16",
        include_str!("fixtures/tdm/annex_e_16.kvn"),
        "MAG",
        "2012-10-29T18:01:28.02",
        "13.1",
    ),
    (
        "E-17",
        include_str!("fixtures/tdm/annex_e_17.kvn"),
        "CARRIER_POWER",
        "2011-05-11T10:26:33.2613",
        "-36.73723984",
    ),
    (
        "E-18",
        include_str!("fixtures/tdm/annex_e_18.kvn"),
        "RECEIVE_PHASE_CT_1",
        "2005-184T13:59:36.27",
        "84297497967.680710",
    ),
    (
        "E-19",
        include_str!("fixtures/tdm/annex_e_19.kvn"),
        "PR_N0",
        "2010-215T20:53:24.000",
        "30.0224",
    ),
    (
        "E-20",
        include_str!("fixtures/tdm/annex_e_20.kvn"),
        "RECEIVE_FREQ_1",
        "2010-049T17:04:43.000",
        "60527.50426",
    ),
    (
        "E-22",
        include_str!("fixtures/tdm/annex_e_21.kvn"),
        "ANGLE_1",
        "2019-10-21T19:00:39.023021",
        "333.89958508",
    ),
];

const ANNEX_E6_FOUR_WAY: &str = "\
CCSDS_TDM_VERS = 2.0
COMMENT TDM example created by yyyyy-nnnA Nav Team (JAXA)
CREATION_DATE = 1998-06-10T01:00:00
ORIGINATOR = JAXA
META_START
TIME_SYSTEM = UTC
START_TIME = 1998-06-10T00:57:37
STOP_TIME = 1998-06-10T00:57:44
PARTICIPANT_1 = NORTH
PARTICIPANT_2 = F07R07
PARTICIPANT_3 = E7
MODE = SEQUENTIAL
PATH = 1,2,3,2,1
INTEGRATION_INTERVAL = 1.0
INTEGRATION_REF = MIDDLE
RANGE_MODE = CONSTANT
RANGE_MODULUS = 0
RANGE_UNITS = km
ANGLE_TYPE = AZEL
META_STOP
DATA_START
RANGE     =  1998-06-10T00:57:37          80452.7542
ANGLE_1      =    1998-06-10T00:57:37           256.64002393
ANGLE_2      =    1998-06-10T00:57:37            13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:37    2106395199.07917
RECEIVE_FREQ =    1998-06-10T00:57:37    2287487999.0
RANGE    =   1998-06-10T00:57:38          80452.7368
ANGLE_1      =   1998-06-10T00:57:38            256.64002393
ANGLE_2      =   1998-06-10T00:57:38             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:38    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:38     2287487999.0
RANGE    =   1998-06-10T00:57:39          80452.7197
ANGLE_1      =   1998-06-10T00:57:39            256.64002393
ANGLE_2      =   1998-06-10T00:57:39             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:39    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:39     2287487999.0
RANGE    =   1998-06-10T00:57:40          80452.7025
ANGLE_1      =   1998-06-10T00:57:40            256.64002393
ANGLE_2      =   1998-06-10T00:57:40             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:40    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:40     2287487999.0
RANGE    =   1998-06-10T00:57:41          80452.6854
ANGLE_1      =   1998-06-10T00:57:41            256.64002393
ANGLE_2      =   1998-06-10T00:57:41             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:41    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:41     2287487999.0
RANGE    =   1998-06-10T00:57:42          80452.6680
ANGLE_1      =   1998-06-10T00:57:42            256.64002393
ANGLE_2      =   1998-06-10T00:57:42             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:42    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:42     2287487999.0
RANGE    =   1998-06-10T00:57:43          80452.6503
ANGLE_1      =   1998-06-10T00:57:43            256.64002393
ANGLE_2      =   1998-06-10T00:57:43             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:43    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:43     2287487999.0
RANGE     =  1998-06-10T00:57:44          80452.6331
ANGLE_1      =   1998-06-10T00:57:44            256.64002393
ANGLE_2      =   1998-06-10T00:57:44             13.38100016
TRANSMIT_FREQ_1 = 1998-06-10T00:57:44    2106395199.07917
RECEIVE_FREQ =   1998-06-10T00:57:44     2287487999.0
DATA_STOP";

const ANNEX_E9_RANGE_TRANSMIT_TAG: &str = "\
CCSDS_TDM_VERS = 2.0
COMMENT This TDM example contains range data timetagged at transmit time
CREATION_DATE = 2005-09-17T23:59:59
ORIGINATOR = JAXA
META_START
TIME_SYSTEM = UTC
START_TIME = 2005-09-17T00:41:38.0000
STOP_TIME = 2005-09-17T00:42:58.0000
PARTICIPANT_1 = yyyy-nnnA
PARTICIPANT_2 = USC1
MODE = SEQUENTIAL
PATH = 2,1,2
TRANSMIT_BAND = S
RECEIVE_BAND = S
TIMETAG_REF = TRANSMIT
INTEGRATION_REF = START
RANGE_MODE = CONSTANT
RANGE_MODULUS = 1.0E7
RANGE_UNITS = km
DATA_QUALITY = VALIDATED
CORRECTION_RANGE = 0.0
CORRECTIONS_APPLIED = YES
META_STOP
DATA_START
RANGE = 2005-09-17T00:41:38.000000 3198.03679519614
RANGE = 2005-09-17T00:41:40.000000 3199.82505720811
RANGE = 2005-09-17T00:41:42.000000 3201.61631714467
RANGE = 2005-09-17T00:41:44.000000 3203.40832656236
RANGE = 2005-09-17T00:41:46.000000 3205.20108546120
RANGE = 2005-09-17T00:41:48.000000 3206.99384436004
RANGE = 2005-09-17T00:41:50.000000 3208.79110014575
RANGE = 2005-09-17T00:41:52.000000 3210.58535800688
RANGE = 2005-09-17T00:41:54.000000 3212.38336327374
RANGE = 2005-09-17T00:41:56.000000 3214.18136854059
RANGE = 2005-09-17T00:41:58.000000 3215.98012328859
RANGE = 2005-09-17T00:42:00.000000 3217.78037699888
RANGE = 2005-09-17T00:42:02.000000 3219.58287915260
RANGE = 2005-09-17T00:42:04.000000 3221.38613078747
RANGE = 2005-09-17T00:42:06.000000 3223.19013190349
RANGE = 2005-09-17T00:42:08.000000 3224.99488250065
RANGE = 2005-09-17T00:42:10.000000 3226.80113206010
RANGE = 2005-09-17T00:42:12.000000 3228.60963006298
RANGE = 2005-09-17T00:42:14.000000 3230.41587962244
RANGE = 2005-09-17T00:42:16.000000 3232.22587658761
RANGE = 2005-09-17T00:42:18.000000 3234.03662303393
RANGE = 2005-09-17T00:42:20.000000 3235.84886844254
RANGE = 2005-09-17T00:42:22.000000 3237.65961488886
RANGE = 2005-09-17T00:42:24.000000 3239.47560770319
RANGE = 2005-09-17T00:42:26.000000 3241.28860259295
RANGE = 2005-09-17T00:42:28.000000 3243.10384592614
RANGE = 2005-09-17T00:42:30.000000 3244.92133770276
RANGE = 2005-09-17T00:42:32.000000 3246.73882947939
RANGE = 2005-09-17T00:42:34.000000 3248.55856969945
RANGE = 2005-09-17T00:42:36.000000 3250.37681095722
RANGE = 2005-09-17T00:42:38.000000 3252.19879962071
RANGE = 2005-09-17T00:42:40.000000 3254.02003880307
RANGE = 2005-09-17T00:42:42.000000 3255.84352642885
RANGE = 2005-09-17T00:42:44.000000 3257.66851301693
RANGE = 2005-09-17T00:42:46.000000 3259.49125116157
RANGE = 2005-09-17T00:42:48.000000 3261.31848619307
RANGE = 2005-09-17T00:42:50.000000 3263.14572122459
RANGE = 2005-09-17T00:42:52.000000 3264.97295625609
RANGE = 2005-09-17T00:42:54.000000 3266.80169024990
RANGE = 2005-09-17T00:42:56.000000 3268.63267268713
RANGE = 2005-09-17T00:42:58.000000 3270.46440460551
DATA_STOP";

const ANNEX_E10_DIFFERENCED_DOPPLER: &str = "\
CCSDS_TDM_VERS = 2.0
COMMENT This TDM example contains single differenced Doppler data.
CREATION_DATE = 2006-354T01:38:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
START_TIME = 2003-07-08T04:45:25.0000
STOP_TIME = 2003-07-08T04:48:25.0000
PARTICIPANT_1 = yyyy-nnnA
PARTICIPANT_2 = DSS-24
PARTICIPANT_3 = DSS-25
MODE = SINGLE_DIFF
PATH_1 = 1,2
PATH_2 = 1,3
TRANSMIT_BAND = X
RECEIVE_BAND = X
INTEGRATION_INTERVAL = 10.0
INTEGRATION_REF = MIDDLE
RECEIVE_DELAY_2 = 0.00007732
RECEIVE_DELAY_3 = 0.00007732
DATA_QUALITY = VALIDATED
META_STOP
DATA_START
COMMENT Transmit frequency is S/C beacon one OWLT prior to receive time
TRANSMIT_FREQ_1 = 2003-07-08T04:10:0000 8.435360E+09
RECEIVE_FREQ = 2003-07-08T04:45:25.0000 8.738750457763670E+00
RECEIVE_FREQ = 2003-07-08T04:45:35.0000 8.320683479309080E+00
RECEIVE_FREQ = 2003-07-08T04:45:45.0000 7.909399032592770E+00
RECEIVE_FREQ = 2003-07-08T04:45:55.0000 7.490205764770500E+00
RECEIVE_FREQ = 2003-07-08T04:46:05.0000 7.149572372436510E+00
RECEIVE_FREQ = 2003-07-08T04:46:15.0000 6.808938980102530E+00
RECEIVE_FREQ = 2003-07-08T04:46:25.0000 6.481011390686030E+00
RECEIVE_FREQ = 2003-07-08T04:46:35.0000 6.167441368103020E+00
RECEIVE_FREQ = 2003-07-08T04:46:45.0000 5.865190505981440E+00
RECEIVE_FREQ = 2003-07-08T04:46:55.0000 5.590643882751460E+00
RECEIVE_FREQ = 2003-07-08T04:47:05.0000 5.330531120300290E+00
RECEIVE_FREQ = 2003-07-08T04:47:15.0000 5.083267211914060E+00
RECEIVE_FREQ = 2003-07-08T04:47:25.0000 4.850607872009270E+00
RECEIVE_FREQ = 2003-07-08T04:47:35.0000 4.643701979796000E+00
RECEIVE_FREQ = 2003-07-08T04:47:45.0000 4.453802272725000E+00
RECEIVE_FREQ = 2003-07-08T04:47:55.0000 4.281702585856000E+00
RECEIVE_FREQ = 2003-07-08T04:48:05.0000 4.127402919189000E+00
RECEIVE_FREQ = 2003-07-08T04:48:15.0000 3.990903272724000E+00
RECEIVE_FREQ = 2003-07-08T04:48:25.0000 3.872203646461000E+00
DATA_STOP";

const ANNEX_E22_TRACK_ID: &str = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2019-10-21T22:17:21
ORIGINATOR = GSOC
META_START
TRACK_ID = S_191021_18593902_3
TIME_SYSTEM = UTC
START_TIME = 2019-10-21T18:59:38.869008
STOP_TIME = 2019-10-21T19:00:39.023021
PARTICIPANT_1 = SMARTNET-01-A-SUTH
PARTICIPANT_2 = UNKNOWN
MODE = SEQUENTIAL
PATH = 2,1
ANGLE_TYPE = RADEC
REFERENCE_FRAME = EME2000
CORRECTION_RECEIVE = -0.145
CORRECTION_ABERRATION_YEARLY = 0.0056932
CORRECTIONS_APPLIED = YES
META_STOP
DATA_START
ANGLE_1 = 2019-10-21T18:59:38.869008 333.64830529
ANGLE_2 = 2019-10-21T18:59:38.869008 5.23646136
MAG = 2019-10-21T18:59:38.869008 10.66
ANGLE_1 = 2019-10-21T19:00:24.405696 333.83841725
ANGLE_2 = 2019-10-21T19:00:24.405696 5.23617947
MAG = 2019-10-21T19:00:24.405696 10.77
ANGLE_1 = 2019-10-21T19:00:39.023021 333.89958508
ANGLE_2 = 2019-10-21T19:00:39.023021 5.23604417
MAG = 2019-10-21T19:00:39.023021 10.80
DATA_STOP";

const SYNTHETIC_DOPPLER: &str = "\
CCSDS_TDM_VERS=2.0
CREATION_DATE=2020-001T00:00:00
ORIGINATOR=TEST
META_START
TIME_SYSTEM=UTC
PARTICIPANT_1=TX
PARTICIPANT_2=RX
MODE=SEQUENTIAL
PATH=1,2
RANGE_UNITS=s
META_STOP
DATA_START
RANGE=2020-001T00:00:00 1.25
DOPPLER_INSTANTANEOUS=2020-001T00:00:00 -0.0125
DOPPLER_INTEGRATED=2020-001T00:00:00 -0.0126
TRANSMIT_FREQ_1=2020-001T00:00:00 8435360000.125
TRANSMIT_FREQ_RATE_1=2020-001T00:00:00 -0.125
RECEIVE_FREQ=2020-001T00:00:00 8435359991.38625
DATA_STOP";

const TABLE_3_5_EXTRA: &str = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2020-001T00:00:00
ORIGINATOR = TEST
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = TX
PARTICIPANT_2 = RX
MODE = SEQUENTIAL
PATH = 1,2
META_STOP
DATA_START
PC_N0 = 2020-001T00:00:00 41.5
DOPPLER_COUNT = 2020-001T00:00:01 0
DATA_STOP";

const SYNTHETIC_CANONICAL: &str = "\
CCSDS_TDM_VERS = 2.0
CREATION_DATE = 2020-001T00:00:00
ORIGINATOR = TEST
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = TX
PARTICIPANT_2 = RX
MODE = SEQUENTIAL
PATH = 1,2
RANGE_UNITS = s
META_STOP
DATA_START
RANGE = 2020-001T00:00:00 1.25
DOPPLER_INSTANTANEOUS = 2020-001T00:00:00 -0.0125
DOPPLER_INTEGRATED = 2020-001T00:00:00 -0.0126
TRANSMIT_FREQ_1 = 2020-001T00:00:00 8435360000.125
TRANSMIT_FREQ_RATE_1 = 2020-001T00:00:00 -0.125
RECEIVE_FREQ = 2020-001T00:00:00 8435359991.38625
DATA_STOP";

const SYNTHETIC_CANONICAL_FNV1A64: u64 = 6279402006941602445;

#[test]
fn all_annex_e_kvn_examples_parse_and_canonicalize() {
    for (label, example, expected_segments, expected_records) in ANNEX_E_ALL_KVN {
        let parsed =
            tdm::parse_kvn(example).unwrap_or_else(|err| panic!("{label} failed parse: {err}"));
        assert_eq!(
            parsed.segments.len(),
            *expected_segments,
            "{label} segments"
        );
        let records = parsed
            .segments
            .iter()
            .map(|segment| segment.data.records.len())
            .sum::<usize>();
        assert_eq!(records, *expected_records, "{label} records");
        let encoded =
            tdm::encode_kvn(&parsed).unwrap_or_else(|err| panic!("{label} failed encode: {err}"));
        let reparsed = tdm::parse_kvn(&encoded)
            .unwrap_or_else(|err| panic!("{label} encoded form failed parse: {err}"));
        assert_eq!(reparsed, parsed, "{label} must reparse to the same IR");
        assert_eq!(
            tdm::encode_kvn(&reparsed)
                .unwrap_or_else(|err| panic!("{label} re-encode failed: {err}")),
            encoded,
            "{label} canonical KVN must be byte-stable"
        );
    }
}

#[test]
fn annex_e_fixture_files_preserve_published_sample_values() {
    for (label, fixture, keyword, epoch, value) in PUBLISHED_ANNEX_SAMPLE_VALUES {
        assert!(
            fixture_contains_record(fixture, keyword, epoch, value),
            "{label} fixture missing {keyword} {epoch} {value}"
        );
    }
}

#[test]
fn annex_e_table_3_5_units_are_pinned() {
    let e4 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_04.kvn")).unwrap();
    assert_record(
        &e4,
        "PR_N0",
        "2005-191T00:31:51",
        "28.52538",
        TdmUnit::DecibelHertz,
    );

    let e11 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_11.kvn")).unwrap();
    assert_record(
        &e11,
        "DOR",
        "2004-136T15:42:00.0000",
        "-4.911896106591159E-03",
        TdmUnit::Seconds,
    );
    assert_record(
        &e11,
        "VLBI_DELAY",
        "2004-136T15:52:00.0000",
        "-1.911896106591159E-03",
        TdmUnit::Seconds,
    );

    let e13 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_13.kvn")).unwrap();
    assert_record(
        &e13,
        "TROPO_DRY",
        "2005-274T12:00:00",
        "2.0526",
        TdmUnit::Meters,
    );
    assert_record(
        &e13,
        "TROPO_WET",
        "2005-274T12:00:00",
        "0.1139",
        TdmUnit::Meters,
    );
    assert_record(
        &e13,
        "STEC",
        "2005-280T21:45:00",
        "23.1",
        TdmUnit::TotalElectronContentUnits,
    );

    let e14 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_14.kvn")).unwrap();
    assert_record(
        &e14,
        "TEMPERATURE",
        "2005-156T00:03:00",
        "302.95",
        TdmUnit::Kelvin,
    );
    assert_record(
        &e14,
        "PRESSURE",
        "2005-156T00:03:00",
        "896.2",
        TdmUnit::Hectopascals,
    );
    assert_record(
        &e14,
        "RHUMIDITY",
        "2005-156T00:03:00",
        "12.0",
        TdmUnit::Percent,
    );

    let e15 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_15.kvn")).unwrap();
    assert_record(
        &e15,
        "CLOCK_BIAS",
        "2005-142T12:00:00",
        "9.56e-7",
        TdmUnit::Seconds,
    );
    assert_record(
        &e15,
        "CLOCK_DRIFT",
        "2005-142T12:00:00",
        "6.944e-14",
        TdmUnit::SecondsPerSecond,
    );

    let e17 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_17.kvn")).unwrap();
    assert_record(
        &e17,
        "CARRIER_POWER",
        "2011-05-11T10:26:33.2613",
        "-36.73723984",
        TdmUnit::DecibelWatts,
    );
    assert_record(
        &e17,
        "RCS",
        "2011-05-11T10:26:33.2613",
        "2.984",
        TdmUnit::SquareMeters,
    );

    let e18 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_18.kvn")).unwrap();
    assert_record(
        &e18,
        "TRANSMIT_PHASE_CT_1",
        "2005-184T11:12:23",
        "7175173383.615373",
        TdmUnit::Dimensionless,
    );
    assert_record(
        &e18,
        "RECEIVE_PHASE_CT_1",
        "2005-184T13:59:27.27",
        "8429753135.986102",
        TdmUnit::Dimensionless,
    );

    let e21 = tdm::parse_kvn(include_str!("fixtures/tdm/annex_e_21.kvn")).unwrap();
    assert_record(
        &e21,
        "MAG",
        "2019-10-21T18:59:38.869008",
        "10.66",
        TdmUnit::Dimensionless,
    );
}

#[test]
fn annex_e_examples_parse_to_pinned_values() {
    let e6 = tdm::parse_kvn(ANNEX_E6_FOUR_WAY).unwrap();
    assert_eq!(e6.version, "2.0");
    assert_eq!(e6.originator.as_deref(), Some("JAXA"));
    assert_eq!(e6.segments.len(), 1);
    assert_eq!(e6.segments[0].metadata.participants.len(), 3);
    assert_eq!(e6.segments[0].metadata.mode.as_deref(), Some("SEQUENTIAL"));
    assert_eq!(
        e6.segments[0].metadata.paths[0].participants,
        vec![1, 2, 3, 2, 1]
    );
    assert_eq!(e6.segments[0].metadata.range_units, TdmUnit::Kilometers);
    assert_eq!(e6.segments[0].data.records.len(), 40);
    assert_record(
        &e6,
        "RANGE",
        "1998-06-10T00:57:37",
        "80452.7542",
        TdmUnit::Kilometers,
    );
    assert_record(
        &e6,
        "TRANSMIT_FREQ_1",
        "1998-06-10T00:57:37",
        "2106395199.07917",
        TdmUnit::Hertz,
    );
    assert_record(
        &e6,
        "RECEIVE_FREQ",
        "1998-06-10T00:57:44",
        "2287487999.0",
        TdmUnit::Hertz,
    );

    let e9 = tdm::parse_kvn(ANNEX_E9_RANGE_TRANSMIT_TAG).unwrap();
    assert_eq!(
        e9.segments[0].metadata.timetag_ref.as_deref(),
        Some("TRANSMIT")
    );
    assert_eq!(e9.segments[0].data.records.len(), 41);
    assert_record(
        &e9,
        "RANGE",
        "2005-09-17T00:41:38.000000",
        "3198.03679519614",
        TdmUnit::Kilometers,
    );
    assert_record(
        &e9,
        "RANGE",
        "2005-09-17T00:42:58.000000",
        "3270.46440460551",
        TdmUnit::Kilometers,
    );

    let e10 = tdm::parse_kvn(ANNEX_E10_DIFFERENCED_DOPPLER).unwrap();
    assert_eq!(
        e10.segments[0].metadata.mode.as_deref(),
        Some("SINGLE_DIFF")
    );
    assert_eq!(e10.segments[0].metadata.paths[0].key, "PATH_1");
    assert_eq!(e10.segments[0].metadata.paths[1].key, "PATH_2");
    assert_record(
        &e10,
        "TRANSMIT_FREQ_1",
        "2003-07-08T04:10:0000",
        "8.435360E+09",
        TdmUnit::Hertz,
    );
    assert_record(
        &e10,
        "RECEIVE_FREQ",
        "2003-07-08T04:45:25.0000",
        "8.738750457763670E+00",
        TdmUnit::Hertz,
    );
    assert_record(
        &e10,
        "RECEIVE_FREQ",
        "2003-07-08T04:48:25.0000",
        "3.872203646461000E+00",
        TdmUnit::Hertz,
    );

    let e22 = tdm::parse_kvn(ANNEX_E22_TRACK_ID).unwrap();
    assert_eq!(
        e22.segments[0].metadata.get_last("TRACK_ID"),
        Some("S_191021_18593902_3")
    );
    assert_eq!(e22.segments[0].data.records.len(), 9);
    assert_record(
        &e22,
        "ANGLE_1",
        "2019-10-21T19:00:39.023021",
        "333.89958508",
        TdmUnit::Degrees,
    );
}

#[test]
fn canonical_encoding_is_byte_stable_for_annex_examples() {
    for example in [
        ANNEX_E6_FOUR_WAY,
        ANNEX_E9_RANGE_TRANSMIT_TAG,
        ANNEX_E10_DIFFERENCED_DOPPLER,
        ANNEX_E22_TRACK_ID,
    ] {
        let parsed = tdm::parse_kvn(example).unwrap();
        let encoded = tdm::encode_kvn(&parsed).unwrap();
        let reparsed = tdm::parse_kvn(&encoded).unwrap();
        assert_eq!(reparsed, parsed);
        assert_eq!(tdm::encode_kvn(&reparsed).unwrap(), encoded);
    }
}

#[test]
fn doppler_range_and_frequency_units_are_assigned_from_ccsds_table() {
    let parsed = tdm::parse_kvn(SYNTHETIC_DOPPLER).unwrap();
    assert_record(
        &parsed,
        "RANGE",
        "2020-001T00:00:00",
        "1.25",
        TdmUnit::Seconds,
    );
    assert_record(
        &parsed,
        "DOPPLER_INSTANTANEOUS",
        "2020-001T00:00:00",
        "-0.0125",
        TdmUnit::KilometersPerSecond,
    );
    assert_record(
        &parsed,
        "DOPPLER_INTEGRATED",
        "2020-001T00:00:00",
        "-0.0126",
        TdmUnit::KilometersPerSecond,
    );
    assert_record(
        &parsed,
        "TRANSMIT_FREQ_RATE_1",
        "2020-001T00:00:00",
        "-0.125",
        TdmUnit::HertzPerSecond,
    );
    assert_record(
        &parsed,
        "RECEIVE_FREQ",
        "2020-001T00:00:00",
        "8435359991.38625",
        TdmUnit::Hertz,
    );
}

#[test]
fn frequency_records_preserve_pinned_ieee_bits() {
    let e6 = tdm::parse_kvn(ANNEX_E6_FOUR_WAY).unwrap();
    assert_record_bits(
        &e6,
        "TRANSMIT_FREQ_1",
        "1998-06-10T00:57:37",
        "2106395199.07917",
        0x41df_6342_8fc5_111f,
        TdmUnit::Hertz,
    );
    assert_record_bits(
        &e6,
        "RECEIVE_FREQ",
        "1998-06-10T00:57:44",
        "2287487999.0",
        0x41e1_0b09_7fe0_0000,
        TdmUnit::Hertz,
    );

    let e10 = tdm::parse_kvn(ANNEX_E10_DIFFERENCED_DOPPLER).unwrap();
    assert_record_bits(
        &e10,
        "TRANSMIT_FREQ_1",
        "2003-07-08T04:10:0000",
        "8.435360E+09",
        0x41ff_6c96_1000_0000,
        TdmUnit::Hertz,
    );
    assert_record_bits(
        &e10,
        "RECEIVE_FREQ",
        "2003-07-08T04:45:25.0000",
        "8.738750457763670E+00",
        0x4021_7a3d_7fff_ffff,
        TdmUnit::Hertz,
    );

    let synthetic = tdm::parse_kvn(SYNTHETIC_DOPPLER).unwrap();
    assert_record_bits(
        &synthetic,
        "RECEIVE_FREQ",
        "2020-001T00:00:00",
        "8435359991.38625",
        0x41ff_6c96_0f76_2e14,
        TdmUnit::Hertz,
    );
}

#[test]
fn remaining_table_3_5_units_are_assigned_from_ccsds_table() {
    let parsed = tdm::parse_kvn(TABLE_3_5_EXTRA).unwrap();
    assert_record(
        &parsed,
        "PC_N0",
        "2020-001T00:00:00",
        "41.5",
        TdmUnit::DecibelHertz,
    );
    assert_record(
        &parsed,
        "DOPPLER_COUNT",
        "2020-001T00:00:01",
        "0",
        TdmUnit::Dimensionless,
    );
}

#[test]
fn canonical_synthetic_encoding_matches_pinned_bytes_and_hash() {
    let encoded = tdm::encode_kvn(&tdm::parse_kvn(SYNTHETIC_DOPPLER).unwrap()).unwrap();
    assert_eq!(encoded, SYNTHETIC_CANONICAL);
    assert_eq!(fnv1a64(encoded.as_bytes()), SYNTHETIC_CANONICAL_FNV1A64);
}

#[test]
fn malformed_inputs_yield_typed_errors() {
    assert_eq!(
        tdm::parse_kvn("CREATION_DATE = 2020-001T00:00:00"),
        Err(TdmError::MissingVersion)
    );

    let missing_value = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RECEIVE_FREQ = 2020-001T00:00:00
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(missing_value),
        Err(TdmError::MalformedRecord {
            line: 6,
            keyword: "RECEIVE_FREQ".to_string(),
        })
    );

    let invalid_angle = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
ANGLE_1 = 2020-001T00:00:00 360.0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(invalid_angle),
        Err(TdmError::InvalidField {
            field: "ANGLE_1".to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        })
    );

    let non_finite = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RECEIVE_FREQ = 2020-001T00:00:00 NaN
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(non_finite),
        Err(TdmError::InvalidField {
            field: "RECEIVE_FREQ".to_string(),
            kind: TdmInputErrorKind::NonFinite,
        })
    );

    for value in ["1", ".5", "1.", "1234567890123456.7"] {
        let bad_numeric = format!(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2020-001T00:00:00 {value}
DATA_STOP"
        );
        assert_eq!(
            tdm::parse_kvn(&bad_numeric),
            Err(TdmError::InvalidField {
                field: "RANGE".to_string(),
                kind: TdmInputErrorKind::FloatParse,
            })
        );
    }

    for value in ["1.0E-400", "4.0E-324"] {
        let underflow = format!(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2020-001T00:00:00 {value}
DATA_STOP"
        );
        assert_eq!(
            tdm::parse_kvn(&underflow),
            Err(TdmError::InvalidField {
                field: "RANGE".to_string(),
                kind: TdmInputErrorKind::OutOfRange,
            })
        );
    }

    let minimum_positive_double = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2020-001T00:00:00 4.94E-324
DATA_STOP";
    let parsed_minimum = tdm::parse_kvn(minimum_positive_double).unwrap();
    assert_eq!(
        find_record(&parsed_minimum, "RANGE", "2020-001T00:00:00")
            .value
            .value
            .to_bits(),
        0x0000_0000_0000_0001
    );

    let inline_key_unit = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
ANGLE_1 [Hz] = 2020-001T00:00:00 1.0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(inline_key_unit),
        Err(TdmError::InvalidField {
            field: "ANGLE_1 [Hz]".to_string(),
            kind: TdmInputErrorKind::UnexpectedUnit,
        })
    );

    let inline_value_unit = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
ANGLE_1 = 2020-001T00:00:00 1.0 [Hz]
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(inline_value_unit),
        Err(TdmError::InvalidField {
            field: "ANGLE_1".to_string(),
            kind: TdmInputErrorKind::UnexpectedUnit,
        })
    );

    let unknown_keyword = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
UNKNOWN_OBS = 2020-001T00:00:00 1.0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(unknown_keyword),
        Err(TdmError::InvalidField {
            field: "UNKNOWN_OBS".to_string(),
            kind: TdmInputErrorKind::UnknownKeyword,
        })
    );

    let invalid_index = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RECEIVE_FREQ_6 = 2020-001T00:00:00 1.0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(invalid_index),
        Err(TdmError::InvalidField {
            field: "RECEIVE_FREQ_6".to_string(),
            kind: TdmInputErrorKind::InvalidIndex,
        })
    );

    for field in [
        "TRANSMIT_FREQ_0",
        "TRANSMIT_FREQ_RATE_6",
        "RECEIVE_PHASE_CT_0",
        "TRANSMIT_PHASE_CT_6",
    ] {
        let invalid_indexed_keyword = format!(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
{field} = 2020-001T00:00:00 1.0
DATA_STOP"
        );
        assert_eq!(
            tdm::parse_kvn(&invalid_indexed_keyword),
            Err(TdmError::InvalidField {
                field: field.to_string(),
                kind: TdmInputErrorKind::InvalidIndex,
            })
        );
    }

    for value in ["1.0E+3", "+1.0", "-1.0"] {
        let invalid_phase_count = format!(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RECEIVE_PHASE_CT_1 = 2020-001T00:00:00 {value}
DATA_STOP"
        );
        assert_eq!(
            tdm::parse_kvn(&invalid_phase_count),
            Err(TdmError::InvalidField {
                field: "RECEIVE_PHASE_CT_1".to_string(),
                kind: TdmInputErrorKind::FloatParse,
            })
        );
    }

    for (field, value, kind) in [
        ("RCS", "0.0", TdmInputErrorKind::NotPositive),
        ("STEC", "0.0", TdmInputErrorKind::NotPositive),
        ("TROPO_DRY", "-0.1", TdmInputErrorKind::Negative),
        ("TROPO_WET", "-0.1", TdmInputErrorKind::Negative),
        ("RHUMIDITY", "-0.1", TdmInputErrorKind::OutOfRange),
        ("RHUMIDITY", "100.1", TdmInputErrorKind::OutOfRange),
        ("TEMPERATURE", "0.0", TdmInputErrorKind::NotPositive),
    ] {
        let invalid_domain = format!(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
{field} = 2020-001T00:00:00 {value}
DATA_STOP"
        );
        assert_eq!(
            tdm::parse_kvn(&invalid_domain),
            Err(TdmError::InvalidField {
                field: field.to_string(),
                kind,
            })
        );
    }

    let negative_zero = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2020-001T00:00:00 -0.0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(negative_zero),
        Err(TdmError::InvalidField {
            field: "RANGE".to_string(),
            kind: TdmInputErrorKind::NegativeZero,
        })
    );

    let fractional_doppler_count = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
DOPPLER_COUNT = 2020-001T00:00:00 1.5
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(fractional_doppler_count),
        Err(TdmError::InvalidField {
            field: "DOPPLER_COUNT".to_string(),
            kind: TdmInputErrorKind::NonInteger,
        })
    );

    let decimal_doppler_count = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
DOPPLER_COUNT = 2020-001T00:00:00 1.0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(decimal_doppler_count),
        Err(TdmError::InvalidField {
            field: "DOPPLER_COUNT".to_string(),
            kind: TdmInputErrorKind::NonInteger,
        })
    );

    let exponent_doppler_count = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
DOPPLER_COUNT = 2020-001T00:00:00 1E+0
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(exponent_doppler_count),
        Err(TdmError::InvalidField {
            field: "DOPPLER_COUNT".to_string(),
            kind: TdmInputErrorKind::NonInteger,
        })
    );

    let negative_doppler_count = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
DOPPLER_COUNT = 2020-001T00:00:00 -1
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(negative_doppler_count),
        Err(TdmError::InvalidField {
            field: "DOPPLER_COUNT".to_string(),
            kind: TdmInputErrorKind::Negative,
        })
    );

    let large_doppler_count = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
DOPPLER_COUNT = 2020-001T00:00:00 2147483648
DATA_STOP";
    assert_eq!(
        tdm::parse_kvn(large_doppler_count),
        Err(TdmError::InvalidField {
            field: "DOPPLER_COUNT".to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        })
    );
}

fn assert_record<'a>(
    tdm: &'a Tdm,
    keyword: &str,
    epoch: &str,
    text: &str,
    unit: TdmUnit,
) -> &'a TdmDataRecord {
    let record = find_record(tdm, keyword, epoch);
    assert_eq!(record.value.text, text);
    assert_eq!(record.unit, unit);
    record
}

fn assert_record_bits(tdm: &Tdm, keyword: &str, epoch: &str, text: &str, bits: u64, unit: TdmUnit) {
    let record = assert_record(tdm, keyword, epoch, text, unit);
    assert_eq!(record.value.value.to_bits(), bits);
}

fn find_record<'a>(tdm: &'a Tdm, keyword: &str, epoch: &str) -> &'a TdmDataRecord {
    tdm.segments
        .iter()
        .flat_map(|segment| &segment.data.records)
        .find(|record| record.keyword == keyword && record.epoch == epoch)
        .unwrap_or_else(|| panic!("missing {keyword} at {epoch}"))
}

fn fixture_contains_record(fixture: &str, keyword: &str, epoch: &str, value: &str) -> bool {
    fixture.lines().any(|line| {
        let Some((key, raw_value)) = line.split_once('=') else {
            return false;
        };
        if key.trim() != keyword {
            return false;
        }
        let mut parts = raw_value.split_whitespace();
        parts.next() == Some(epoch) && parts.next() == Some(value) && parts.next().is_none()
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[test]
fn observable_variants_are_specific_for_required_keywords() {
    let parsed = tdm::parse_kvn(SYNTHETIC_DOPPLER).unwrap();
    let observables: Vec<&TdmObservable> = parsed.segments[0]
        .data
        .records
        .iter()
        .map(|record| &record.observable)
        .collect();
    assert!(matches!(observables[0], TdmObservable::Range));
    assert!(matches!(
        observables[1],
        TdmObservable::DopplerInstantaneous
    ));
    assert!(matches!(observables[2], TdmObservable::DopplerIntegrated));
    assert!(matches!(
        observables[3],
        TdmObservable::TransmitFreq {
            participant: Some(1)
        }
    ));
    assert!(matches!(
        observables[4],
        TdmObservable::TransmitFreqRate {
            participant: Some(1)
        }
    ));
    assert!(matches!(
        observables[5],
        TdmObservable::ReceiveFreq { participant: None }
    ));
}

fn decode_hex(encoded: &[u8]) -> Vec<u8> {
    let digits: Vec<u8> = encoded
        .iter()
        .copied()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    let (pairs, rest) = digits.as_chunks::<2>();
    assert!(rest.is_empty(), "hex fixture must have complete bytes");
    pairs
        .iter()
        .map(|[high, low]| {
            let high = (*high as char).to_digit(16).expect("hex high nibble");
            let low = (*low as char).to_digit(16).expect("hex low nibble");
            ((high << 4) | low) as u8
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The scheduled bounded fuzz run on the `tdm_round_trip` target failed on this
/// input. Line 112 of it is `COMMENT=`, inside a metadata block; that parsed as
/// a field keyed `COMMENT`, the writer emitted it as `COMMENT = `, and the
/// reparse read that line as a comment whose text is `=`. The reparsed value
/// then held one more comment and one fewer field than the value it came from.
///
/// CCSDS 503.0-B-2 4.2.5 c) excepts `COMMENT` from the KVN syntax and 4.5.3
/// requires at least one space after the keyword, so the line is not an
/// assignment in any section and is refused where it appears.
#[test]
fn scheduled_fuzz_comment_assignment_is_refused() {
    let encoded = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tdm/",
        "tdm_round_trip-crash-cb1c6e75.hex"
    ));
    let input = decode_hex(encoded);
    assert_eq!(input.len(), 1071);
    assert_eq!(
        sha256_hex(&input),
        "384540c19b79b5e7ee69373def4b6746e7a21cd1aa4dfb4e0c8c866f2dd77c35"
    );

    let text = String::from_utf8_lossy(&input);
    assert_eq!(
        tdm::parse_kvn(&text),
        Err(TdmError::MalformedLine {
            line: 112,
            text: "COMMENT=".to_string(),
        })
    );
}

/// The smallest message carrying the refused line, so the property is pinned on
/// a file with nothing else wrong with it rather than only on the fuzz payload.
/// CCSDS 503.0-B-2 4.2.5 c) excepts `COMMENT` from the KVN syntax and 4.5.3
/// requires at least one space after the keyword, so `COMMENT=value` is an
/// assignment the standard does not define, in a header or in metadata.
#[test]
fn a_comment_keyed_assignment_is_refused_in_both_sections() {
    let header =
        "CCSDS_TDM_VERS = 2.0\nCOMMENT=note\nMETA_START\nMETA_STOP\nDATA_START\nDATA_STOP\n";
    assert_eq!(
        tdm::parse_kvn(header),
        Err(TdmError::MalformedLine {
            line: 2,
            text: "COMMENT=note".to_string(),
        })
    );

    let metadata = "CCSDS_TDM_VERS = 2.0\nMETA_START\nCOMMENT=\nMETA_STOP\nDATA_START\nDATA_STOP\n";
    assert_eq!(
        tdm::parse_kvn(metadata),
        Err(TdmError::MalformedLine {
            line: 3,
            text: "COMMENT=".to_string(),
        })
    );
}

/// The refusal turns on the space 4.5.3 requires, not on what the text holds: a
/// comment whose own text opens with an equals sign still reads as a comment.
#[test]
fn a_comment_line_with_a_space_still_reads_as_a_comment() {
    let text = "CCSDS_TDM_VERS = 2.0\nCOMMENT note\nMETA_START\nCOMMENT =value\nMETA_STOP\nDATA_START\nDATA_STOP\n";
    let parsed = tdm::parse_kvn(text).expect("comment lines parse");
    // A comment's text is everything after the keyword, so `COMMENT = note`
    // would read as the text `= note` rather than as an assignment.
    assert_eq!(parsed.comments, vec!["note".to_string()]);
    assert_eq!(
        parsed.segments[0].metadata.comments,
        vec!["=value".to_string()]
    );
}

/// `TdmField` is public, so a caller can hold the state the parser refuses.
/// Encoding it would write `COMMENT = note`, which reads back as a comment
/// rather than as the field, so the writer refuses it in either section.
#[test]
fn encoding_refuses_a_field_keyed_comment() {
    let text = "CCSDS_TDM_VERS = 2.0\nMETA_START\nMETA_STOP\nDATA_START\nDATA_STOP\n";
    let base = tdm::parse_kvn(text).expect("the minimal message parses");
    tdm::encode_kvn(&base).expect("the minimal message encodes");

    let field = TdmField {
        key: "COMMENT".to_string(),
        value: "note".to_string(),
    };

    let mut header = base.clone();
    header.header_fields.push(field.clone());
    match tdm::encode_kvn(&header) {
        Err(TdmError::MalformedLine { text, .. }) => assert_eq!(text, "COMMENT = note"),
        other => panic!("a header field keyed COMMENT must be refused, got {other:?}"),
    }

    let mut metadata = base.clone();
    metadata.segments[0].metadata.fields.push(field);
    match tdm::encode_kvn(&metadata) {
        Err(TdmError::MalformedLine { text, .. }) => assert_eq!(text, "COMMENT = note"),
        other => panic!("a metadata field keyed COMMENT must be refused, got {other:?}"),
    }

    // A padded key writes the same line, so it is refused on the same footing.
    let mut padded = base;
    padded.header_fields.push(TdmField {
        key: "COMMENT ".to_string(),
        value: "note".to_string(),
    });
    match tdm::encode_kvn(&padded) {
        Err(TdmError::MalformedLine { .. }) => {}
        other => panic!("a padded COMMENT key must be refused, got {other:?}"),
    }
}
