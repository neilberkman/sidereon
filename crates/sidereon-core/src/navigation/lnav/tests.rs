//! Unit tests for the GPS LNAV codec, mirroring the authoritative round-trip,
//! parity, structure, and error-path behavior. The 0-ULP golden against the
//! Python reference generator lives in `tests/lnav.rs`.

use super::*;

fn n(v: f64) -> LnavNumber {
    LnavNumber::Float(v)
}
fn i(v: i64) -> LnavNumber {
    LnavNumber::Int(v)
}

/// A representative MEO GPS SV, matching `Sidereon.GNSS.Navigation.LNAV.Ephemeris.example/0`.
fn example() -> LnavParams {
    LnavParams {
        week_number: i(290),
        l2_code: i(1),
        l2_p_data_flag: i(0),
        ura_index: i(0),
        sv_health: i(0),
        iodc: i(0x2AB),
        tgd: n(-5.587_935_447_692_871e-9),
        toc: i(504_000),
        af0: n(-1.234e-4),
        af1: n(-3.5e-12),
        af2: n(0.0),
        iode: i(0xAB),
        crs: n(-55.625),
        delta_n: n(1.56e-9),
        m0: n(-0.35),
        cuc: n(-1.2e-6),
        eccentricity: n(0.012),
        cus: n(8.3e-6),
        sqrt_a: n(5153.65),
        toe: i(504_000),
        fit_interval_flag: i(0),
        aodo: i(0),
        cic: n(5.0e-8),
        omega0: n(-0.78),
        cis: n(-2.1e-7),
        i0: n(0.305),
        crc: n(250.625),
        omega: n(0.95),
        omega_dot: n(-8.1e-9),
        idot: n(1.5e-10),
    }
}

fn default_opts() -> LnavOptions {
    LnavOptions {
        tow: i(0),
        alert: i(0),
        anti_spoof: i(0),
        integrity: i(0),
        tlm_message: i(0),
    }
}

fn quantize(value: f64, lsb: f64) -> f64 {
    (value / lsb).round() * lsb
}

#[test]
fn round_trip_recovers_scaled_fields_within_lsb() {
    let p = example();
    let opts = LnavOptions {
        tow: i(12_345),
        ..default_opts()
    };
    let [sf1, sf2, sf3] = encode(&p, &opts).unwrap();
    let d = decode(&sf1, &sf2, &sf3).unwrap();

    let cases: &[(f64, f64, f64)] = &[
        (d.tgd, p.tgd.as_f64(), TWO_POW_M31),
        (d.af0, p.af0.as_f64(), TWO_POW_M31),
        (d.af1, p.af1.as_f64(), TWO_POW_M43),
        (d.af2, p.af2.as_f64(), TWO_POW_M55),
        (d.crs, p.crs.as_f64(), TWO_POW_M5),
        (d.delta_n, p.delta_n.as_f64(), TWO_POW_M43),
        (d.m0, p.m0.as_f64(), TWO_POW_M31),
        (d.cuc, p.cuc.as_f64(), TWO_POW_M29),
        (d.eccentricity, p.eccentricity.as_f64(), TWO_POW_M33),
        (d.cus, p.cus.as_f64(), TWO_POW_M29),
        (d.sqrt_a, p.sqrt_a.as_f64(), TWO_POW_M19),
        (d.cic, p.cic.as_f64(), TWO_POW_M29),
        (d.omega0, p.omega0.as_f64(), TWO_POW_M31),
        (d.cis, p.cis.as_f64(), TWO_POW_M29),
        (d.i0, p.i0.as_f64(), TWO_POW_M31),
        (d.crc, p.crc.as_f64(), TWO_POW_M5),
        (d.omega, p.omega.as_f64(), TWO_POW_M31),
        (d.omega_dot, p.omega_dot.as_f64(), TWO_POW_M43),
        (d.idot, p.idot.as_f64(), TWO_POW_M43),
    ];
    for &(decoded, input, lsb) in cases {
        assert!((decoded - quantize(input, lsb)).abs() <= lsb / 2.0);
    }
    // Scale-16 integer fields recover exactly.
    assert_eq!(d.toc, 504_000);
    assert_eq!(d.toe, 504_000);
}

#[test]
fn round_trip_recovers_integer_fields_exactly() {
    let p = example();
    let [sf1, sf2, sf3] = encode(&p, &default_opts()).unwrap();
    let d = decode(&sf1, &sf2, &sf3).unwrap();

    assert_eq!(d.week_number, 290);
    assert_eq!(d.l2_code, 1);
    assert_eq!(d.ura_index, 0);
    assert_eq!(d.sv_health, 0);
    assert_eq!(d.iodc, 0x2AB);
    assert_eq!(d.iode, 0xAB);
    assert_eq!(d.fit_interval_flag, 0);
    assert_eq!(d.aodo, 0);
}

#[test]
fn negative_signed_values_keep_their_sign() {
    let p = example();
    let [sf1, sf2, sf3] = encode(&p, &default_opts()).unwrap();
    let d = decode(&sf1, &sf2, &sf3).unwrap();
    for v in [
        d.tgd,
        d.af0,
        d.af1,
        d.crs,
        d.m0,
        d.cuc,
        d.cis,
        d.omega0,
        d.omega_dot,
    ] {
        assert!(v < 0.0);
    }
}

#[test]
fn near_full_scale_signed_values_round_trip() {
    let p = LnavParams {
        m0: n(-0.9999999),
        omega0: n(0.9999999),
        idot: n(-3.0e-10),
        ..example()
    };
    let [sf1, sf2, sf3] = encode(&p, &default_opts()).unwrap();
    let d = decode(&sf1, &sf2, &sf3).unwrap();
    assert!((d.m0 - quantize(-0.9999999, TWO_POW_M31)).abs() <= TWO_POW_M31 / 2.0);
    assert!((d.omega0 - quantize(0.9999999, TWO_POW_M31)).abs() <= TWO_POW_M31 / 2.0);
    assert!((d.idot - quantize(-3.0e-10, TWO_POW_M43)).abs() <= TWO_POW_M43 / 2.0);
}

#[test]
fn iodc_split_across_words_recovers_exactly() {
    let p = LnavParams {
        iodc: i(0x2AB),
        ..example()
    };
    let [sf1, sf2, sf3] = encode(&p, &default_opts()).unwrap();
    let d = decode(&sf1, &sf2, &sf3).unwrap();
    assert_eq!(d.iodc, 0x2AB);
}

#[test]
fn parity_matches_table_20_xiv_with_nonzero_prior() {
    let data = [
        1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1,
    ];
    assert_eq!(parity(&data, 1, 0).unwrap(), [0, 1, 1, 0, 1, 1]);
    assert_eq!(parity(&data, 0, 1).unwrap(), [1, 0, 0, 1, 0, 0]);
}

#[test]
fn parity_rejects_bad_source_word_length() {
    assert_eq!(
        parity(&[1, 0, 1], 0, 0),
        Err(LnavError::BadWordLength {
            expected: 24,
            actual: 3
        })
    );
}

#[test]
fn all_words_satisfy_parity_with_chained_prior() {
    let opts = LnavOptions {
        tow: i(777),
        ..default_opts()
    };
    let sfs = encode(&example(), &opts).unwrap();
    for sf in &sfs {
        let (mut d29, mut d30) = (0u8, 0u8);
        for word in sf.chunks(WORD_LENGTH) {
            assert!(parity_valid(word, d29, d30));
            d29 = word[28];
            d30 = word[29];
        }
    }
}

#[test]
fn flipping_a_single_data_bit_fails_parity() {
    let sfs = encode(&example(), &default_opts()).unwrap();
    let sf2 = &sfs[1];
    let w2: Vec<u8> = sf2[WORD_LENGTH..2 * WORD_LENGTH].to_vec();
    let w3: Vec<u8> = sf2[2 * WORD_LENGTH..3 * WORD_LENGTH].to_vec();
    let (d29p, d30p) = (w2[28], w2[29]);
    for &pos in &[0usize, 5, 12, 23] {
        let mut corrupted = w3.clone();
        corrupted[pos] ^= 1;
        assert!(!parity_valid(&corrupted, d29p, d30p));
    }
}

#[test]
fn decode_reports_parity_failed_on_corruption() {
    let mut sfs = encode(&example(), &default_opts()).unwrap();
    sfs[0][60] ^= 1;
    assert_eq!(
        decode(&sfs[0], &sfs[1], &sfs[2]),
        Err(LnavError::ParityFailed {
            subframe: 1,
            word: 3
        })
    );
}

#[test]
fn how_and_word10_trailing_parity_bits_are_zero() {
    let sfs = encode(&example(), &default_opts()).unwrap();
    for sf in &sfs {
        assert_eq!(&sf[58..60], &[0, 0]); // HOW (word 2) t-bits
        assert_eq!(&sf[298..300], &[0, 0]); // word 10 t-bits
    }
}

#[test]
fn structure_preamble_tow_and_subframe_id() {
    assert_eq!(PREAMBLE, 0x8B);
    let opts = LnavOptions {
        tow: i(54_321),
        ..default_opts()
    };
    let sfs = encode(&example(), &opts).unwrap();
    for (idx, sf) in sfs.iter().enumerate() {
        assert_eq!(bits_to_uint(&sf[0..8]), 0x8B);
        assert_eq!(tow(sf), Some(54_321));
        assert_eq!(subframe_id(sf), Some((idx + 1) as u64));
        assert_eq!(sf.len(), SUBFRAME_LENGTH);
    }
}

#[test]
fn out_of_range_fields_return_tagged_errors() {
    let p = LnavParams {
        week_number: i(2000),
        ..example()
    };
    assert_eq!(
        encode(&p, &default_opts()),
        Err(LnavError::OutOfRange {
            field: LnavField::WeekNumber,
            value: i(2000)
        })
    );

    let p = LnavParams {
        ura_index: i(99),
        ..example()
    };
    assert_eq!(
        encode(&p, &default_opts()),
        Err(LnavError::OutOfRange {
            field: LnavField::UraIndex,
            value: i(99)
        })
    );

    let p = LnavParams {
        l2_p_data_flag: i(2),
        ..example()
    };
    assert_eq!(
        encode(&p, &default_opts()),
        Err(LnavError::OutOfRange {
            field: LnavField::L2PDataFlag,
            value: i(2)
        })
    );

    let p = LnavParams {
        l2_p_data_flag: n(0.5),
        ..example()
    };
    assert_eq!(
        encode(&p, &default_opts()),
        Err(LnavError::OutOfRange {
            field: LnavField::L2PDataFlag,
            value: n(0.5)
        })
    );

    let p = LnavParams {
        m0: n(5.0),
        ..example()
    };
    assert_eq!(
        encode(&p, &default_opts()),
        Err(LnavError::OutOfRange {
            field: LnavField::M0,
            value: n(5.0)
        })
    );

    let opts = LnavOptions {
        alert: i(2),
        ..default_opts()
    };
    assert_eq!(
        encode(&example(), &opts),
        Err(LnavError::OutOfRange {
            field: LnavField::Alert,
            value: i(2)
        })
    );
}

#[test]
fn l2_p_data_flag_accepts_only_one_bit_integer_values() {
    let flag_zero = LnavParams {
        l2_p_data_flag: i(0),
        ..example()
    };
    let flag_one = LnavParams {
        l2_p_data_flag: i(1),
        ..example()
    };

    let sf_zero = encode(&flag_zero, &default_opts()).unwrap();
    let sf_one = encode(&flag_one, &default_opts()).unwrap();

    assert_eq!(source_words(&sf_zero[0])[1][0], 0);
    assert_eq!(source_words(&sf_one[0])[1][0], 1);
}
