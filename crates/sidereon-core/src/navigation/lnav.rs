//! GPS L1 C/A LNAV navigation message synthesis and decoding (subframes 1-3).
//!
//! The legacy navigation (LNAV) message is the data stream modulated onto the
//! GPS L1 C/A signal at 50 bits per second. Its structure is defined in
//! IS-GPS-200 (Section 20.3): the message is organized into 1500-bit *frames*,
//! each frame being five 300-bit *subframes*, and each subframe being ten 30-bit
//! *words*. Every word carries 24 source data bits (most significant first)
//! followed by 6 parity bits.
//!
//! This module covers the clock and ephemeris subframes:
//!
//!   * Subframe 1 - SV clock correction and health (IS-GPS-200 Table 20-I).
//!   * Subframe 2 - first half of the ephemeris (IS-GPS-200 Table 20-II).
//!   * Subframe 3 - second half of the ephemeris (IS-GPS-200 Table 20-III).
//!
//! The first word of every subframe is the telemetry (TLM) word; the second is
//! the hand-over word (HOW). Both are described in IS-GPS-200 Section 20.3.3.
//!
//! [`encode`] and [`decode`] exchange engineering-unit parameter values (the
//! products of the transmitted integers and their IS-GPS-200 scale factors).
//! Angular ephemeris quantities are in semicircles (and semicircles/second),
//! the harmonic correction terms are in radians, distances are in meters, and
//! clock/time quantities are in seconds, exactly as tabulated in IS-GPS-200.
//!
//! The codec is integer / exact-power-of-two arithmetic throughout, so it is a
//! 0-ULP target: a given set of parameters encodes to one exact bit pattern. The
//! authoritative golden is the `lnav` section of
//! `tests/fixtures/orbis_gnss_application_golden.json` (the Python reference
//! generator), asserted bit-for-bit in `tests/lnav.rs`.

use crate::validate;

/// Bit length of a single LNAV word (IS-GPS-200 Section 20.3.2).
pub const WORD_LENGTH: usize = 30;
/// Bit length of a single LNAV subframe (IS-GPS-200 Section 20.3.2).
pub const SUBFRAME_LENGTH: usize = 300;
/// The 8-bit TLM preamble `1000 1011` as an integer (IS-GPS-200 Section 20.3.3.1).
pub const PREAMBLE: u32 = 0b1000_1011;

// IS-GPS-200 per-field LSB scale factors (Tables 20-I/II/III). Each is an exact
// power of two; `1.0 / 2^n` is the exact f64 value and matches the reference
// generator's `2 ** -n` bit-for-bit, so `round(value / scale)` is deterministic.
const TWO_POW_4: f64 = 16.0;
const TWO_POW_M5: f64 = 1.0 / 32.0;
const TWO_POW_M19: f64 = 1.0 / 524_288.0;
const TWO_POW_M29: f64 = 1.0 / 536_870_912.0;
const TWO_POW_M31: f64 = 1.0 / 2_147_483_648.0;
const TWO_POW_M33: f64 = 1.0 / 8_589_934_592.0;
const TWO_POW_M43: f64 = 1.0 / 8_796_093_022_208.0;
const TWO_POW_M55: f64 = 1.0 / 36_028_797_018_963_968.0;

/// A numeric parameter value preserving the integer-vs-float distinction of the
/// caller's input, so range-validation failures can echo the value back in its
/// original type (an integer field reports an integer, a scaled field a float).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LnavNumber {
    /// An integer-typed value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
}

impl LnavNumber {
    fn as_f64(self) -> f64 {
        match self {
            LnavNumber::Int(i) => i as f64,
            LnavNumber::Float(f) => f,
        }
    }

    fn as_i64_truncated(self) -> i64 {
        match self {
            LnavNumber::Int(i) => i,
            LnavNumber::Float(f) => f as i64,
        }
    }

    fn is_int(self) -> bool {
        matches!(self, LnavNumber::Int(_))
    }
}

/// An LNAV parameter field, used to tag a range-validation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LnavField {
    Tow,
    Alert,
    AntiSpoof,
    Integrity,
    TlmMessage,
    WeekNumber,
    L2Code,
    L2PDataFlag,
    UraIndex,
    SvHealth,
    Iodc,
    Tgd,
    Toc,
    Af2,
    Af1,
    Af0,
    Iode,
    Crs,
    DeltaN,
    M0,
    Cuc,
    Eccentricity,
    Cus,
    SqrtA,
    Toe,
    FitIntervalFlag,
    Aodo,
    Cic,
    Omega0,
    Cis,
    I0,
    Crc,
    Omega,
    OmegaDot,
    Idot,
}

impl LnavField {
    /// The snake_case field name, matching the Elixir error-tuple atom.
    pub fn name(self) -> &'static str {
        match self {
            LnavField::Tow => "tow",
            LnavField::Alert => "alert",
            LnavField::AntiSpoof => "anti_spoof",
            LnavField::Integrity => "integrity",
            LnavField::TlmMessage => "tlm_message",
            LnavField::WeekNumber => "week_number",
            LnavField::L2Code => "l2_code",
            LnavField::L2PDataFlag => "l2_p_data_flag",
            LnavField::UraIndex => "ura_index",
            LnavField::SvHealth => "sv_health",
            LnavField::Iodc => "iodc",
            LnavField::Tgd => "tgd",
            LnavField::Toc => "toc",
            LnavField::Af2 => "af2",
            LnavField::Af1 => "af1",
            LnavField::Af0 => "af0",
            LnavField::Iode => "iode",
            LnavField::Crs => "crs",
            LnavField::DeltaN => "delta_n",
            LnavField::M0 => "m0",
            LnavField::Cuc => "cuc",
            LnavField::Eccentricity => "eccentricity",
            LnavField::Cus => "cus",
            LnavField::SqrtA => "sqrt_a",
            LnavField::Toe => "toe",
            LnavField::FitIntervalFlag => "fit_interval_flag",
            LnavField::Aodo => "aodo",
            LnavField::Cic => "cic",
            LnavField::Omega0 => "omega0",
            LnavField::Cis => "cis",
            LnavField::I0 => "i0",
            LnavField::Crc => "crc",
            LnavField::Omega => "omega",
            LnavField::OmegaDot => "omega_dot",
            LnavField::Idot => "idot",
        }
    }
}

/// An LNAV codec failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LnavError {
    /// A parameter does not fit its transmitted field; echoes the offending
    /// value in its original numeric type.
    OutOfRange { field: LnavField, value: LnavNumber },
    /// A word's recomputed parity did not match (1-based subframe and word).
    ParityFailed { subframe: u8, word: u8 },
    /// A parity source word was not exactly 24 data bits.
    BadWordLength { expected: usize, actual: usize },
    /// A subframe was not exactly [`SUBFRAME_LENGTH`] bits.
    BadSubframeLength { subframe: u8 },
}

/// Clock and ephemeris parameters in engineering units (the per-field input to
/// [`encode`]). Values preserve their numeric type for faithful error echoing.
#[derive(Clone, Copy, Debug)]
pub struct LnavParams {
    pub week_number: LnavNumber,
    pub l2_code: LnavNumber,
    pub l2_p_data_flag: LnavNumber,
    pub ura_index: LnavNumber,
    pub sv_health: LnavNumber,
    pub iodc: LnavNumber,
    pub tgd: LnavNumber,
    pub toc: LnavNumber,
    pub af0: LnavNumber,
    pub af1: LnavNumber,
    pub af2: LnavNumber,
    pub iode: LnavNumber,
    pub crs: LnavNumber,
    pub delta_n: LnavNumber,
    pub m0: LnavNumber,
    pub cuc: LnavNumber,
    pub eccentricity: LnavNumber,
    pub cus: LnavNumber,
    pub sqrt_a: LnavNumber,
    pub toe: LnavNumber,
    pub fit_interval_flag: LnavNumber,
    pub aodo: LnavNumber,
    pub cic: LnavNumber,
    pub omega0: LnavNumber,
    pub cis: LnavNumber,
    pub i0: LnavNumber,
    pub crc: LnavNumber,
    pub omega: LnavNumber,
    pub omega_dot: LnavNumber,
    pub idot: LnavNumber,
}

/// TLM/HOW options accompanying an [`encode`] (defaults applied by the caller).
#[derive(Clone, Copy, Debug)]
pub struct LnavOptions {
    pub tow: LnavNumber,
    pub alert: LnavNumber,
    pub anti_spoof: LnavNumber,
    pub integrity: LnavNumber,
    pub tlm_message: LnavNumber,
}

/// Decoded clock and ephemeris parameters (the typed output of [`decode`]). The
/// integer-typed fields are recovered exactly; the scaled fields are the
/// transmitted integer times the IS-GPS-200 LSB. (`l2_p_data_flag` is an
/// encode-only flag in word 4 and is not recovered.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LnavDecoded {
    pub week_number: i64,
    pub l2_code: i64,
    pub ura_index: i64,
    pub sv_health: i64,
    pub iodc: i64,
    pub tgd: f64,
    pub toc: i64,
    pub af0: f64,
    pub af1: f64,
    pub af2: f64,
    pub iode: i64,
    pub crs: f64,
    pub delta_n: f64,
    pub m0: f64,
    pub cuc: f64,
    pub eccentricity: f64,
    pub cus: f64,
    pub sqrt_a: f64,
    pub toe: i64,
    pub fit_interval_flag: i64,
    pub aodo: i64,
    pub cic: f64,
    pub omega0: f64,
    pub cis: f64,
    pub i0: f64,
    pub crc: f64,
    pub omega: f64,
    pub omega_dot: f64,
    pub idot: f64,
}

// --- field descriptors for range validation ---------------------------------

#[derive(Clone, Copy)]
enum FieldKind {
    /// Pure unsigned integer field (must be a non-negative integer).
    Uint { bits: u32 },
    /// Scaled unsigned field (`round(value / scale)` must fit, value >= 0).
    UintScaled { bits: u32, scale: f64 },
    /// Scaled signed field (`round(value / scale)` in two's-complement range).
    SintScaled { bits: u32, scale: f64 },
}

/// Validates one field exactly as IS-GPS-200 / the Elixir reference does, with
/// the same first-failure semantics.
fn validate_field(field: LnavField, value: LnavNumber, kind: FieldKind) -> Result<(), LnavError> {
    let in_range = match kind {
        FieldKind::Uint { bits } => {
            value.is_int() && {
                let v = value.as_i64_truncated();
                v >= 0 && v < (1i64 << bits)
            }
        }
        FieldKind::UintScaled { bits, scale } => {
            value.as_f64() >= 0.0 && {
                let n = round_half_away(value.as_f64() / scale);
                n >= 0 && n < (1i64 << bits)
            }
        }
        FieldKind::SintScaled { bits, scale } => {
            // `is_finite` reproduces the Elixir `is_number/1` guard: a non-number
            // (e.g. a missing `nil` field) is out of range, not a silent zero.
            value.as_f64().is_finite() && {
                let n = round_half_away(value.as_f64() / scale);
                let limit = 1i64 << (bits - 1);
                n >= -limit && n < limit
            }
        }
    };

    if in_range {
        Ok(())
    } else {
        Err(LnavError::OutOfRange { field, value })
    }
}

/// IEEE round-half-away-from-zero, matching Elixir `round/1`.
fn round_half_away(x: f64) -> i64 {
    x.round() as i64
}

// --- public API -------------------------------------------------------------

/// Extracts the 17-bit time-of-week count from a hand-over word.
///
/// Accepts either a 30-bit HOW word or a full 300-bit subframe (whose word 2 is
/// the HOW). Returns `None` on any other length.
pub fn tow(bits: &[u8]) -> Option<u64> {
    how_word(bits).map(|how| bits_to_uint(&how[0..17]))
}

/// Extracts the 3-bit subframe ID from a hand-over word.
///
/// Accepts a 30-bit HOW word or a full 300-bit subframe. Returns `None` on any
/// other length.
pub fn subframe_id(bits: &[u8]) -> Option<u64> {
    how_word(bits).map(|how| bits_to_uint(&how[19..22]))
}

fn how_word(bits: &[u8]) -> Option<Vec<u8>> {
    match bits.len() {
        WORD_LENGTH => Some(bits.to_vec()),
        SUBFRAME_LENGTH => Some(bits[WORD_LENGTH..2 * WORD_LENGTH].to_vec()),
        _ => None,
    }
}

/// Computes the 6 parity bits of a word (IS-GPS-200 Table 20-XIV).
///
/// `data24` is the 24 *source* data bits (most significant first, before the
/// `D30*` complementation). `d29_prev`/`d30_prev` are the two trailing parity
/// bits of the previous word. Returns `[D25, D26, D27, D28, D29, D30]`.
pub fn parity(data24: &[u8], d29_prev: u8, d30_prev: u8) -> Result<[u8; 6], LnavError> {
    validate::exact_len(data24, 24, "lnav parity data bits").map_err(|error| {
        LnavError::BadWordLength {
            expected: error.expected,
            actual: error.actual,
        }
    })?;

    // 1-based indexing of the source bits, matching IS-GPS-200 Table 20-XIV.
    let d = |n: usize| data24[n - 1];

    let d25 = xor(&[
        d29_prev,
        d(1),
        d(2),
        d(3),
        d(5),
        d(6),
        d(10),
        d(11),
        d(12),
        d(13),
        d(14),
        d(17),
        d(18),
        d(20),
        d(23),
    ]);
    let d26 = xor(&[
        d30_prev,
        d(2),
        d(3),
        d(4),
        d(6),
        d(7),
        d(11),
        d(12),
        d(13),
        d(14),
        d(15),
        d(18),
        d(19),
        d(21),
        d(24),
    ]);
    let d27 = xor(&[
        d29_prev,
        d(1),
        d(3),
        d(4),
        d(5),
        d(7),
        d(8),
        d(12),
        d(13),
        d(14),
        d(15),
        d(16),
        d(19),
        d(20),
        d(22),
    ]);
    let d28 = xor(&[
        d30_prev,
        d(2),
        d(4),
        d(5),
        d(6),
        d(8),
        d(9),
        d(13),
        d(14),
        d(15),
        d(16),
        d(17),
        d(20),
        d(21),
        d(23),
    ]);
    let d29 = xor(&[
        d30_prev,
        d(1),
        d(3),
        d(5),
        d(6),
        d(7),
        d(9),
        d(10),
        d(14),
        d(15),
        d(16),
        d(17),
        d(18),
        d(21),
        d(22),
        d(24),
    ]);
    let d30 = xor(&[
        d29_prev,
        d(3),
        d(5),
        d(6),
        d(8),
        d(9),
        d(10),
        d(11),
        d(13),
        d(15),
        d(19),
        d(22),
        d(23),
        d(24),
    ]);

    Ok([d25, d26, d27, d28, d29, d30])
}

/// Verifies the parity of a single 30-bit word.
///
/// `word30` is the 30-bit word as transmitted (data bits possibly complemented
/// by `D30*`, followed by 6 received parity bits). `d29_prev`/`d30_prev` are the
/// previous word's trailing parity bits.
pub fn parity_valid(word30: &[u8], d29_prev: u8, d30_prev: u8) -> bool {
    if word30.len() != WORD_LENGTH {
        return false;
    }
    let source: Vec<u8> = word30[0..24].iter().map(|b| b ^ d30_prev).collect();
    let received = &word30[24..30];
    parity(&source, d29_prev, d30_prev).is_ok_and(|par| par.as_slice() == received)
}

/// Encodes clock and ephemeris parameters into LNAV subframes 1-3.
///
/// Returns the three 300-bit subframes (most significant first) keyed 1/2/3.
/// Out-of-range parameters yield [`LnavError::OutOfRange`].
pub fn encode(params: &LnavParams, opts: &LnavOptions) -> Result<[Vec<u8>; 3], LnavError> {
    validate_field(LnavField::Tow, opts.tow, FieldKind::Uint { bits: 17 })?;
    validate_field(LnavField::Alert, opts.alert, FieldKind::Uint { bits: 1 })?;
    validate_field(
        LnavField::AntiSpoof,
        opts.anti_spoof,
        FieldKind::Uint { bits: 1 },
    )?;
    validate_field(
        LnavField::Integrity,
        opts.integrity,
        FieldKind::Uint { bits: 1 },
    )?;
    validate_field(
        LnavField::TlmMessage,
        opts.tlm_message,
        FieldKind::Uint { bits: 14 },
    )?;

    let w1 = subframe1_words(params)?;
    let w2 = subframe2_words(params)?;
    let w3 = subframe3_words(params)?;

    let tlm = tlm_data(
        opts.tlm_message.as_i64_truncated(),
        opts.integrity.as_i64_truncated(),
    );

    let sf1 = assemble_subframe(&prepend_headers(&tlm, opts, 1, w1))?;
    let sf2 = assemble_subframe(&prepend_headers(&tlm, opts, 2, w2))?;
    let sf3 = assemble_subframe(&prepend_headers(&tlm, opts, 3, w3))?;

    Ok([sf1, sf2, sf3])
}

/// Decodes LNAV subframes 1-3 back into the engineering-unit parameter struct.
///
/// Parity is verified on all 30 words first; a failure returns
/// [`LnavError::ParityFailed`] (1-based word).
pub fn decode(sf1: &[u8], sf2: &[u8], sf3: &[u8]) -> Result<LnavDecoded, LnavError> {
    verify_subframe(sf1, 1)?;
    verify_subframe(sf2, 2)?;
    verify_subframe(sf3, 3)?;

    let w1 = source_words(sf1);
    let w2 = source_words(sf2);
    let w3 = source_words(sf3);

    let mut d = LnavDecoded {
        week_number: 0,
        l2_code: 0,
        ura_index: 0,
        sv_health: 0,
        iodc: 0,
        tgd: 0.0,
        toc: 0,
        af0: 0.0,
        af1: 0.0,
        af2: 0.0,
        iode: 0,
        crs: 0.0,
        delta_n: 0.0,
        m0: 0.0,
        cuc: 0.0,
        eccentricity: 0.0,
        cus: 0.0,
        sqrt_a: 0.0,
        toe: 0,
        fit_interval_flag: 0,
        aodo: 0,
        cic: 0.0,
        omega0: 0.0,
        cis: 0.0,
        i0: 0.0,
        crc: 0.0,
        omega: 0.0,
        omega_dot: 0.0,
        idot: 0.0,
    };

    decode_subframe1(&mut d, &w1);
    decode_subframe2(&mut d, &w2);
    decode_subframe3(&mut d, &w3);

    Ok(d)
}

// --- Subframe 1 (clock/health), IS-GPS-200 Table 20-I -----------------------

/// One word entry awaiting parity: its 24 source data bits, and whether the two
/// trailing data bits must be solved so the word's `D29`/`D30` parity is zero.
struct WordEntry {
    data: Vec<u8>,
    solve: bool,
}

impl WordEntry {
    fn raw(data: Vec<u8>) -> Self {
        WordEntry { data, solve: false }
    }
    fn solved(data: Vec<u8>) -> Self {
        WordEntry { data, solve: true }
    }
}

fn subframe1_words(p: &LnavParams) -> Result<Vec<WordEntry>, LnavError> {
    validate_field(
        LnavField::WeekNumber,
        p.week_number,
        FieldKind::Uint { bits: 10 },
    )?;
    validate_field(LnavField::L2Code, p.l2_code, FieldKind::Uint { bits: 2 })?;
    validate_field(
        LnavField::L2PDataFlag,
        p.l2_p_data_flag,
        FieldKind::Uint { bits: 1 },
    )?;
    validate_field(
        LnavField::UraIndex,
        p.ura_index,
        FieldKind::Uint { bits: 4 },
    )?;
    validate_field(
        LnavField::SvHealth,
        p.sv_health,
        FieldKind::Uint { bits: 6 },
    )?;
    validate_field(LnavField::Iodc, p.iodc, FieldKind::Uint { bits: 10 })?;
    validate_field(
        LnavField::Tgd,
        p.tgd,
        FieldKind::SintScaled {
            bits: 8,
            scale: TWO_POW_M31,
        },
    )?;
    validate_field(
        LnavField::Toc,
        p.toc,
        FieldKind::UintScaled {
            bits: 16,
            scale: TWO_POW_4,
        },
    )?;
    validate_field(
        LnavField::Af2,
        p.af2,
        FieldKind::SintScaled {
            bits: 8,
            scale: TWO_POW_M55,
        },
    )?;
    validate_field(
        LnavField::Af1,
        p.af1,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M43,
        },
    )?;
    validate_field(
        LnavField::Af0,
        p.af0,
        FieldKind::SintScaled {
            bits: 22,
            scale: TWO_POW_M31,
        },
    )?;

    let iodc = p.iodc.as_i64_truncated();
    let iodc_msb = (iodc >> 8) & 0x3;
    let iodc_lsb = iodc & 0xFF;
    let l2_p_data_flag = p.l2_p_data_flag.as_i64_truncated();

    // Word 3: WN(10) L2(2) URA(4) health(6) IODC-MSB(2).
    let mut word3 = pack_uint(p.week_number.as_i64_truncated(), 10);
    word3.extend(pack_uint(p.l2_code.as_i64_truncated(), 2));
    word3.extend(pack_uint(p.ura_index.as_i64_truncated(), 4));
    word3.extend(pack_uint(p.sv_health.as_i64_truncated(), 6));
    word3.extend(pack_uint(iodc_msb, 2));

    // Word 4: L2-P data flag (bit 1) + 23 reserved bits.
    let mut word4 = pack_uint(l2_p_data_flag, 1);
    word4.extend(zeros(23));
    // Words 5, 6: reserved.
    let word5 = zeros(24);
    let word6 = zeros(24);
    // Word 7: 16 reserved bits then TGD(8).
    let mut word7 = zeros(16);
    word7.extend(pack_sint(p.tgd.as_f64(), 8, TWO_POW_M31));
    // Word 8: IODC-LSB(8) toc(16).
    let mut word8 = pack_uint(iodc_lsb, 8);
    word8.extend(pack_uint_scaled(p.toc.as_f64(), 16, TWO_POW_4));
    // Word 9: af2(8) af1(16).
    let mut word9 = pack_sint(p.af2.as_f64(), 8, TWO_POW_M55);
    word9.extend(pack_sint(p.af1.as_f64(), 16, TWO_POW_M43));
    // Word 10: af0(22) + 2 solved bits.
    let word10 = pack_sint(p.af0.as_f64(), 22, TWO_POW_M31);

    Ok(vec![
        WordEntry::raw(word3),
        WordEntry::raw(word4),
        WordEntry::raw(word5),
        WordEntry::raw(word6),
        WordEntry::raw(word7),
        WordEntry::raw(word8),
        WordEntry::raw(word9),
        WordEntry::solved(word10),
    ])
}

fn decode_subframe1(p: &mut LnavDecoded, w: &[Vec<u8>]) {
    let word3 = &w[0];
    let word7 = &w[4];
    let word8 = &w[5];
    let word9 = &w[6];
    let word10 = &w[7];

    p.week_number = bits_to_uint(slice(word3, 1, 10)) as i64;
    p.l2_code = bits_to_uint(slice(word3, 11, 2)) as i64;
    p.ura_index = bits_to_uint(slice(word3, 13, 4)) as i64;
    p.sv_health = bits_to_uint(slice(word3, 17, 6)) as i64;
    let iodc_msb = bits_to_uint(slice(word3, 23, 2)) as i64;

    p.tgd = unpack_sint(slice(word7, 17, 8), TWO_POW_M31);
    let iodc_lsb = bits_to_uint(slice(word8, 1, 8)) as i64;
    p.toc = unpack_uint_scaled_int(slice(word8, 9, 16), TWO_POW_4);
    p.af2 = unpack_sint(slice(word9, 1, 8), TWO_POW_M55);
    p.af1 = unpack_sint(slice(word9, 9, 16), TWO_POW_M43);
    p.af0 = unpack_sint(slice(word10, 1, 22), TWO_POW_M31);

    p.iodc = (iodc_msb << 8) | iodc_lsb;
}

// --- Subframe 2 (ephemeris part 1), IS-GPS-200 Table 20-II ------------------

fn subframe2_words(p: &LnavParams) -> Result<Vec<WordEntry>, LnavError> {
    validate_field(LnavField::Iode, p.iode, FieldKind::Uint { bits: 8 })?;
    validate_field(
        LnavField::Crs,
        p.crs,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M5,
        },
    )?;
    validate_field(
        LnavField::DeltaN,
        p.delta_n,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M43,
        },
    )?;
    validate_field(
        LnavField::M0,
        p.m0,
        FieldKind::SintScaled {
            bits: 32,
            scale: TWO_POW_M31,
        },
    )?;
    validate_field(
        LnavField::Cuc,
        p.cuc,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M29,
        },
    )?;
    validate_field(
        LnavField::Eccentricity,
        p.eccentricity,
        FieldKind::UintScaled {
            bits: 32,
            scale: TWO_POW_M33,
        },
    )?;
    validate_field(
        LnavField::Cus,
        p.cus,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M29,
        },
    )?;
    validate_field(
        LnavField::SqrtA,
        p.sqrt_a,
        FieldKind::UintScaled {
            bits: 32,
            scale: TWO_POW_M19,
        },
    )?;
    validate_field(
        LnavField::Toe,
        p.toe,
        FieldKind::UintScaled {
            bits: 16,
            scale: TWO_POW_4,
        },
    )?;
    validate_field(
        LnavField::FitIntervalFlag,
        p.fit_interval_flag,
        FieldKind::Uint { bits: 1 },
    )?;
    validate_field(LnavField::Aodo, p.aodo, FieldKind::Uint { bits: 5 })?;

    let m0 = pack_sint(p.m0.as_f64(), 32, TWO_POW_M31);
    let ecc = pack_uint_scaled(p.eccentricity.as_f64(), 32, TWO_POW_M33);
    let sqrt_a = pack_uint_scaled(p.sqrt_a.as_f64(), 32, TWO_POW_M19);

    // Word 3: IODE(8) Crs(16).
    let mut word3 = pack_uint(p.iode.as_i64_truncated(), 8);
    word3.extend(pack_sint(p.crs.as_f64(), 16, TWO_POW_M5));
    // Word 4: Delta-n(16) M0-MSB(8).
    let mut word4 = pack_sint(p.delta_n.as_f64(), 16, TWO_POW_M43);
    word4.extend_from_slice(&m0[0..8]);
    // Word 5: M0-LSB(24).
    let word5 = m0[8..32].to_vec();
    // Word 6: Cuc(16) e-MSB(8).
    let mut word6 = pack_sint(p.cuc.as_f64(), 16, TWO_POW_M29);
    word6.extend_from_slice(&ecc[0..8]);
    // Word 7: e-LSB(24).
    let word7 = ecc[8..32].to_vec();
    // Word 8: Cus(16) sqrtA-MSB(8).
    let mut word8 = pack_sint(p.cus.as_f64(), 16, TWO_POW_M29);
    word8.extend_from_slice(&sqrt_a[0..8]);
    // Word 9: sqrtA-LSB(24).
    let word9 = sqrt_a[8..32].to_vec();
    // Word 10: toe(16) fit(1) AODO(5) + 2 solved bits.
    let mut word10 = pack_uint_scaled(p.toe.as_f64(), 16, TWO_POW_4);
    word10.extend(pack_uint(p.fit_interval_flag.as_i64_truncated(), 1));
    word10.extend(pack_uint(p.aodo.as_i64_truncated(), 5));

    Ok(vec![
        WordEntry::raw(word3),
        WordEntry::raw(word4),
        WordEntry::raw(word5),
        WordEntry::raw(word6),
        WordEntry::raw(word7),
        WordEntry::raw(word8),
        WordEntry::raw(word9),
        WordEntry::solved(word10),
    ])
}

fn decode_subframe2(p: &mut LnavDecoded, w: &[Vec<u8>]) {
    let (word3, word4, word5, word6, word7, word8, word9, word10) =
        (&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]);

    p.iode = bits_to_uint(slice(word3, 1, 8)) as i64;
    p.crs = unpack_sint(slice(word3, 9, 16), TWO_POW_M5);
    p.delta_n = unpack_sint(slice(word4, 1, 16), TWO_POW_M43);
    let mut m0_bits = slice(word4, 17, 8).to_vec();
    m0_bits.extend_from_slice(slice(word5, 1, 24));
    p.m0 = unpack_sint(&m0_bits, TWO_POW_M31);
    p.cuc = unpack_sint(slice(word6, 1, 16), TWO_POW_M29);
    let mut ecc_bits = slice(word6, 17, 8).to_vec();
    ecc_bits.extend_from_slice(slice(word7, 1, 24));
    p.eccentricity = unpack_uint_scaled(&ecc_bits, TWO_POW_M33);
    p.cus = unpack_sint(slice(word8, 1, 16), TWO_POW_M29);
    let mut sqrt_a_bits = slice(word8, 17, 8).to_vec();
    sqrt_a_bits.extend_from_slice(slice(word9, 1, 24));
    p.sqrt_a = unpack_uint_scaled(&sqrt_a_bits, TWO_POW_M19);
    p.toe = unpack_uint_scaled_int(slice(word10, 1, 16), TWO_POW_4);
    p.fit_interval_flag = bits_to_uint(slice(word10, 17, 1)) as i64;
    p.aodo = bits_to_uint(slice(word10, 18, 5)) as i64;
}

// --- Subframe 3 (ephemeris part 2), IS-GPS-200 Table 20-III -----------------

fn subframe3_words(p: &LnavParams) -> Result<Vec<WordEntry>, LnavError> {
    validate_field(
        LnavField::Cic,
        p.cic,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M29,
        },
    )?;
    validate_field(
        LnavField::Omega0,
        p.omega0,
        FieldKind::SintScaled {
            bits: 32,
            scale: TWO_POW_M31,
        },
    )?;
    validate_field(
        LnavField::Cis,
        p.cis,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M29,
        },
    )?;
    validate_field(
        LnavField::I0,
        p.i0,
        FieldKind::SintScaled {
            bits: 32,
            scale: TWO_POW_M31,
        },
    )?;
    validate_field(
        LnavField::Crc,
        p.crc,
        FieldKind::SintScaled {
            bits: 16,
            scale: TWO_POW_M5,
        },
    )?;
    validate_field(
        LnavField::Omega,
        p.omega,
        FieldKind::SintScaled {
            bits: 32,
            scale: TWO_POW_M31,
        },
    )?;
    validate_field(
        LnavField::OmegaDot,
        p.omega_dot,
        FieldKind::SintScaled {
            bits: 24,
            scale: TWO_POW_M43,
        },
    )?;
    validate_field(LnavField::Iode, p.iode, FieldKind::Uint { bits: 8 })?;
    validate_field(
        LnavField::Idot,
        p.idot,
        FieldKind::SintScaled {
            bits: 14,
            scale: TWO_POW_M43,
        },
    )?;

    let omega0 = pack_sint(p.omega0.as_f64(), 32, TWO_POW_M31);
    let i0 = pack_sint(p.i0.as_f64(), 32, TWO_POW_M31);
    let omega = pack_sint(p.omega.as_f64(), 32, TWO_POW_M31);

    // Word 3: Cic(16) OMEGA0-MSB(8).
    let mut word3 = pack_sint(p.cic.as_f64(), 16, TWO_POW_M29);
    word3.extend_from_slice(&omega0[0..8]);
    // Word 4: OMEGA0-LSB(24).
    let word4 = omega0[8..32].to_vec();
    // Word 5: Cis(16) i0-MSB(8).
    let mut word5 = pack_sint(p.cis.as_f64(), 16, TWO_POW_M29);
    word5.extend_from_slice(&i0[0..8]);
    // Word 6: i0-LSB(24).
    let word6 = i0[8..32].to_vec();
    // Word 7: Crc(16) omega-MSB(8).
    let mut word7 = pack_sint(p.crc.as_f64(), 16, TWO_POW_M5);
    word7.extend_from_slice(&omega[0..8]);
    // Word 8: omega-LSB(24).
    let word8 = omega[8..32].to_vec();
    // Word 9: OMEGADOT(24).
    let word9 = pack_sint(p.omega_dot.as_f64(), 24, TWO_POW_M43);
    // Word 10: IODE(8) IDOT(14) + 2 solved bits.
    let mut word10 = pack_uint(p.iode.as_i64_truncated(), 8);
    word10.extend(pack_sint(p.idot.as_f64(), 14, TWO_POW_M43));

    Ok(vec![
        WordEntry::raw(word3),
        WordEntry::raw(word4),
        WordEntry::raw(word5),
        WordEntry::raw(word6),
        WordEntry::raw(word7),
        WordEntry::raw(word8),
        WordEntry::raw(word9),
        WordEntry::solved(word10),
    ])
}

fn decode_subframe3(p: &mut LnavDecoded, w: &[Vec<u8>]) {
    let (word3, word4, word5, word6, word7, word8, word9, word10) =
        (&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]);

    p.cic = unpack_sint(slice(word3, 1, 16), TWO_POW_M29);
    let mut omega0_bits = slice(word3, 17, 8).to_vec();
    omega0_bits.extend_from_slice(slice(word4, 1, 24));
    p.omega0 = unpack_sint(&omega0_bits, TWO_POW_M31);
    p.cis = unpack_sint(slice(word5, 1, 16), TWO_POW_M29);
    let mut i0_bits = slice(word5, 17, 8).to_vec();
    i0_bits.extend_from_slice(slice(word6, 1, 24));
    p.i0 = unpack_sint(&i0_bits, TWO_POW_M31);
    p.crc = unpack_sint(slice(word7, 1, 16), TWO_POW_M5);
    let mut omega_bits = slice(word7, 17, 8).to_vec();
    omega_bits.extend_from_slice(slice(word8, 1, 24));
    p.omega = unpack_sint(&omega_bits, TWO_POW_M31);
    p.omega_dot = unpack_sint(slice(word9, 1, 24), TWO_POW_M43);
    p.idot = unpack_sint(slice(word10, 9, 14), TWO_POW_M43);
}

// --- TLM / HOW --------------------------------------------------------------

fn tlm_data(tlm_message: i64, integrity: i64) -> Vec<u8> {
    // IS-GPS-200 Section 20.3.3.1: preamble(8) message(14) integrity(1) reserved(1).
    let mut bits = pack_uint(PREAMBLE as i64, 8);
    bits.extend(pack_uint(tlm_message, 14));
    bits.extend(pack_uint(integrity, 1));
    bits.push(0);
    bits
}

fn how_data(tow: i64, alert: i64, anti_spoof: i64, sf_id: i64) -> WordEntry {
    // IS-GPS-200 Section 20.3.3.2: TOW(17) alert(1) A-S(1) SF-ID(3) + 2 solved.
    let mut base = pack_uint(tow, 17);
    base.extend(pack_uint(alert, 1));
    base.extend(pack_uint(anti_spoof, 1));
    base.extend(pack_uint(sf_id, 3));
    base.extend(zeros(2));
    WordEntry::solved(base)
}

/// Prepends the TLM and HOW header words to the eight data words of a subframe.
fn prepend_headers(
    tlm: &[u8],
    opts: &LnavOptions,
    sf_id: i64,
    words: Vec<WordEntry>,
) -> Vec<WordEntry> {
    let how = how_data(
        opts.tow.as_i64_truncated(),
        opts.alert.as_i64_truncated(),
        opts.anti_spoof.as_i64_truncated(),
        sf_id,
    );
    let mut entries = Vec::with_capacity(words.len() + 2);
    entries.push(WordEntry::raw(tlm.to_vec()));
    entries.push(how);
    entries.extend(words);
    entries
}

// --- word/subframe assembly with parity -------------------------------------

/// Builds the 300-bit subframe from ten word entries (TLM, HOW, then words
/// 3..10), chaining parity through all ten words seeded with `D29* = D30* = 0`.
fn assemble_subframe(entries: &[WordEntry]) -> Result<Vec<u8>, LnavError> {
    let mut bits = Vec::with_capacity(SUBFRAME_LENGTH);
    let (mut d29_prev, mut d30_prev) = (0u8, 0u8);

    for entry in entries {
        let data = if entry.solve {
            solve_tbits(&entry.data, d29_prev, d30_prev)?
        } else {
            pad24(&entry.data)
        };
        let source = pad24(&data);
        let par = parity(&source, d29_prev, d30_prev)?;
        for b in &source {
            bits.push(b ^ d30_prev);
        }
        bits.extend_from_slice(&par);
        d29_prev = par[4];
        d30_prev = par[5];
    }

    Ok(bits)
}

/// Solve the two trailing data bits (positions 23, 24) so `D29 = D30 = 0`.
fn solve_tbits(data24: &[u8], d29_prev: u8, d30_prev: u8) -> Result<Vec<u8>, LnavError> {
    let mut base = pad24(data24);
    base[22] = 0;
    base[23] = 0;
    let par = parity(&base, d29_prev, d30_prev)?;
    let d24 = par[4];
    let d23 = par[5] ^ d24;
    base[22] = d23;
    base[23] = d24;
    Ok(base)
}

fn pad24(bits: &[u8]) -> Vec<u8> {
    let mut out = bits.to_vec();
    out.resize(24, 0);
    out
}

fn verify_subframe(bits: &[u8], sf: u8) -> Result<(), LnavError> {
    if bits.len() != SUBFRAME_LENGTH {
        return Err(LnavError::BadSubframeLength { subframe: sf });
    }

    let (mut d29_prev, mut d30_prev) = (0u8, 0u8);
    for (idx, word) in bits.chunks(WORD_LENGTH).enumerate() {
        if parity_valid(word, d29_prev, d30_prev) {
            d29_prev = word[28];
            d30_prev = word[29];
        } else {
            return Err(LnavError::ParityFailed {
                subframe: sf,
                word: (idx + 1) as u8,
            });
        }
    }
    Ok(())
}

/// Returns words 3..10 as 24-bit source words (with `D30*` uncomplemented).
fn source_words(bits: &[u8]) -> Vec<Vec<u8>> {
    let mut decoded = Vec::with_capacity(10);
    let mut d30_prev = 0u8;
    for word in bits.chunks(WORD_LENGTH) {
        let source: Vec<u8> = word[0..24].iter().map(|b| b ^ d30_prev).collect();
        d30_prev = word[29];
        decoded.push(source);
    }
    // Drop TLM (word 1) and HOW (word 2); keep words 3..10.
    decoded.split_off(2)
}

// --- packing helpers --------------------------------------------------------

fn pack_uint(value: i64, bits: u32) -> Vec<u8> {
    (0..bits).rev().map(|i| ((value >> i) & 1) as u8).collect()
}

fn pack_uint_scaled(value: f64, bits: u32, scale: f64) -> Vec<u8> {
    pack_uint(round_half_away(value / scale), bits)
}

fn pack_sint(value: f64, bits: u32, scale: f64) -> Vec<u8> {
    let int = round_half_away(value / scale);
    pack_twos_complement(int, bits)
}

fn pack_twos_complement(int: i64, bits: u32) -> Vec<u8> {
    let mask = (1i64 << bits) - 1;
    pack_uint(int & mask, bits)
}

fn bits_to_uint(bits: &[u8]) -> u64 {
    bits.iter().fold(0u64, |acc, &b| (acc << 1) | u64::from(b))
}

fn unpack_uint_scaled(bits: &[u8], scale: f64) -> f64 {
    bits_to_uint(bits) as f64 * scale
}

fn unpack_uint_scaled_int(bits: &[u8], scale: f64) -> i64 {
    // The scale-16 fields (toc, toe) recover an exact integer.
    round_half_away(bits_to_uint(bits) as f64 * scale)
}

fn unpack_sint(bits: &[u8], scale: f64) -> f64 {
    bits_to_sint(bits) as f64 * scale
}

fn bits_to_sint(bits: &[u8]) -> i64 {
    let n = bits.len() as u32;
    let raw = bits_to_uint(bits) as i64;
    if raw & (1i64 << (n - 1)) == 0 {
        raw
    } else {
        raw - (1i64 << n)
    }
}

fn zeros(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

/// 1-based slice (`start` is the IS-GPS-200 bit position within the 24-bit word).
fn slice(word: &[u8], start_1based: usize, len: usize) -> &[u8] {
    &word[start_1based - 1..start_1based - 1 + len]
}

fn xor(bits: &[u8]) -> u8 {
    bits.iter().fold(0u8, |acc, &b| acc ^ b)
}

#[cfg(test)]
mod tests;
