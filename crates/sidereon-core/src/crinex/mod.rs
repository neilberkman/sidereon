//! Hatanaka (CRINEX) observation-file decoder and encoder.
//!
//! Reconstructs the plain RINEX observation **text** from a Compact RINEX
//! (CRINEX) stream, reproducing the `CRX2RNX` algorithm, and the inverse:
//! recompacts plain RINEX observation text back to a CRINEX stream
//! (`RNX2CRX`-style). Two stream revisions are handled: **CRINEX 1.0** (which
//! compacts a RINEX 2 observation file) and **CRINEX 3.0** (which compacts a
//! RINEX 3 observation file). The expanded text is what
//! [`crate::rinex_obs::RinexObs`] then parses.
//!
//! The round-trip is closed at the RINEX-text level through the canonical
//! [`ObsStream`] intermediate representation: [`decode`] takes CRINEX to plain
//! RINEX text, [`encode_crinex`] takes plain RINEX text back to CRINEX, and
//! [`parse_stream`] / [`encode_stream`] expose the IR boundary directly. CRINEX
//! compression is not unique, so the encoder emits a canonical all-reset form
//! (see [`encode_stream`]); the guarantee is that decoding the re-emitted CRINEX
//! reproduces the original observations byte-for-byte.
//!
//! # What CRINEX is
//!
//! CRINEX is a lossless, line-oriented ASCII recompression of a RINEX
//! observation file. The plain RINEX header is copied through unchanged (it is
//! never compressed); only the data body is differenced. The body uses two
//! difference engines:
//!
//! - a per-character **text** difference (epoch descriptor line and the trailing
//!   LLI/SSI flag string of each satellite line); and
//! - a per-observation higher-order **integer** difference (each observation
//!   column, and the receiver clock offset), with arc (re)initialization marked
//!   inline by an `order&value` token.
//!
//! The algorithm is fully specified by Hatanaka (2008) and the RNXCMP toolset.
//! This is a deterministic byte-to-text transform, not a float recipe - there is
//! no 0-ULP claim here, exactly as for the SP3 and RINEX-NAV readers. The
//! reconstructed numbers are formatted with the same fixed-decimal layout the
//! reference `crx2rnx` emits (value scaled back by `10^-decimals`, right-aligned
//! in the field) and each output line has its trailing blanks trimmed, which is
//! what makes the expansion reproduce the reference byte-for-byte.
//!
//! # Memory
//!
//! The difference state is bounded: the engines hold only the previous epoch's
//! per-satellite reference values, not the whole stream. [`decode_to`] is the
//! line-at-a-time form - it pushes each reconstructed line to a sink as it is
//! produced, so the *decoder itself* never buffers the full expansion. Note that
//! the [`decode`] convenience does collect the entire expanded text into one
//! `String`; for a multi-megabyte daily file prefer `decode_to` with a streaming
//! sink (e.g. feeding a record consumer) so the expansion is processed
//! incrementally.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::format::columns::{raw_field as field, raw_field_from as field_from};
use crate::validate::{self, FieldError};
use crate::{Error, Result};

/// CRINEX stream revision (the `CRINEX VERS / TYPE` line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrinexVersion {
    /// CRINEX 1.0 - compacts a RINEX 2 observation file.
    V1,
    /// CRINEX 3.0 - compacts a RINEX 3 observation file.
    V3,
}

/// The compression order used by the historical `RNX2CRX` (and the only order a
/// reset token may request without exceeding the classic `M = 5` history). It is
/// carried per token, so this is only a sanity ceiling.
const MAX_ORDER: usize = 6;

/// Canonical, wire-format-agnostic observation stream recovered from a CRINEX
/// file. This is the IR the decoder produces: the difference engines are undone,
/// leaving the plain RINEX header verbatim and each epoch's recovered
/// observations as scaled integers plus reconstructed flag strings.
///
/// Two serializers consume it. [`Decoder`]'s RINEX path renders it back to plain
/// RINEX observation text (the [`decode`] output). [`encode_stream`] renders it
/// back to CRINEX. Because CRINEX compression is not unique, `encode_stream`
/// emits the canonical all-reset form rather than reproducing the original
/// CRINEX bytes; the round-trip guarantee is at the IR / RINEX-text level (see
/// [`encode_stream`]).
///
/// [`parse_stream`] builds it from a CRINEX stream; the [`encode_crinex`] entry
/// builds it from plain RINEX observation text, so the same container backs both
/// the decode and encode directions.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsStream {
    /// Stream revision (selects the epoch grammar).
    pub version: CrinexVersion,
    /// The embedded plain RINEX header lines, verbatim, up to and including
    /// `END OF HEADER` (the two CRINEX header lines are not part of the IR).
    pub header: Vec<String>,
    /// Epoch records in file order.
    pub epochs: Vec<EpochRecord>,
}

/// One decoded epoch: either an observation epoch or a verbatim event block.
#[derive(Debug, Clone, PartialEq)]
pub enum EpochRecord {
    /// An observation epoch (epoch flag 0 or 1).
    Obs(ObsEpoch),
    /// An event record (epoch flag > 1): header/comment lines copied verbatim,
    /// carrying no differenced observations.
    Event {
        /// The reconstructed epoch descriptor line.
        descriptor: String,
        /// The `numsat` event lines following the descriptor, verbatim.
        lines: Vec<String>,
    },
}

/// A decoded observation epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsEpoch {
    /// The reconstructed full epoch descriptor (V3: leading `>` plus the SV list;
    /// V1: leading space plus the SV list).
    pub descriptor: String,
    /// Recovered receiver clock offset as the scaled integer the stream carried,
    /// or `None` when the epoch carried no clock token.
    pub clock: Option<i64>,
    /// Per-satellite recovered observations, in epoch SV-list order.
    pub sats: Vec<SatRecord>,
}

/// One satellite's recovered observations at an epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct SatRecord {
    /// SV token (e.g. `G05`), already expanded for mono-system RINEX-2 streams.
    pub sv: String,
    /// Recovered scaled-integer observation values; `None` is a blanked column.
    pub values: Vec<Option<i64>>,
    /// The reconstructed LLI/SSI flag string (the text-difference engine state).
    pub flags: String,
}

/// Width of one RINEX-3 observation field: `F14.3` value + LLI + SSI.
const OBS_FIELD_WIDTH: usize = 16;
/// Width of the numeric part of one observation field (`F14.3`).
const OBS_VALUE_WIDTH: usize = 14;

/// Decode a CRINEX (Hatanaka) observation stream into the plain RINEX
/// observation text it expands to, returning the whole text as a `String`.
///
/// Supports CRINEX 1.0 (RINEX 2 host) and CRINEX 3.0 (RINEX 3 host). Returns
/// [`Error::Parse`] with a human-readable reason on a malformed stream.
pub fn decode(crinex_text: &str) -> Result<String> {
    let mut out = String::with_capacity(crinex_text.len() * 4);
    decode_to(crinex_text, |line| {
        out.push_str(line);
        out.push('\n');
    })?;
    Ok(out)
}

/// Streaming decode: reconstruct the plain RINEX observation text one line at a
/// time, pushing each line (without its trailing newline) to `emit`.
///
/// This is the bounded-memory form: the difference engines retain only the
/// previous epoch's state, so a multi-megabyte daily file never holds its full
/// expansion in a single buffer. [`decode`] is the collecting convenience.
pub fn decode_to<W: FnMut(&str)>(crinex_text: &str, mut emit: W) -> Result<()> {
    let mut decoder = Decoder::new();
    let mut lines = crinex_text.lines();
    decoder.read_crinex_header(&mut lines, &mut emit)?;
    decoder.read_body(&mut lines, &mut emit)?;
    Ok(())
}

/// Encode plain RINEX observation text into a CRINEX (Hatanaka) stream, the
/// inverse of [`decode`].
///
/// Supports RINEX 2 (encoded as CRINEX 1.0) and RINEX 3 (encoded as CRINEX 3.0),
/// selected from the embedded `RINEX VERSION / TYPE` header line. The text is
/// parsed into the canonical [`ObsStream`] IR and serialized by [`encode_stream`].
/// Because CRINEX compression is not unique, the output is the canonical
/// all-reset form (see [`encode_stream`]); it is not byte-identical to an
/// arbitrary `RNX2CRX` stream, but `decode(encode_crinex(rinex)) == rinex` for
/// any RINEX observation text this round-trips. Returns [`Error::Parse`] on a
/// malformed input.
pub fn encode_crinex(rinex_text: &str) -> Result<String> {
    let stream = parse_rinex_obs(rinex_text)?;
    Ok(encode_stream(&stream))
}

/// Per-observation / clock higher-order integer difference engine.
///
/// Mirrors the Hatanaka `NumDiff`: a value is the original decimal scaled to an
/// integer; the engine holds a small history and reconstructs the next value
/// from a delta using the signed binomial (Pascal) coefficients for the current
/// order. The order ramps up to `level` as samples arrive and is reset by an
/// `order&value` token (arc reinitialization).
#[derive(Debug, Clone)]
struct NumDiff {
    /// Iteration counter (current effective order), clamped to `level`.
    m: usize,
    /// Target compression level (order) for this arc.
    level: usize,
    /// History buffer, most-recent first.
    buf: [i64; MAX_ORDER],
}

impl NumDiff {
    /// Initialize a fresh arc seeded with `data` at the given `level`. The seed
    /// is the recovered value of the init sample.
    fn new(data: i64, level: usize) -> Self {
        let mut buf = [0i64; MAX_ORDER];
        buf[0] = data;
        Self { m: 0, level, buf }
    }

    /// Reinitialize the arc (the `order&value` reset token): clear the order,
    /// set the new level, and seed the history with the recovered value.
    fn force_init(&mut self, data: i64, level: usize) {
        self.m = 0;
        self.level = level;
        self.rotate(data);
    }

    /// Push a recovered value into the history buffer (most-recent first).
    fn rotate(&mut self, data: i64) {
        self.buf.copy_within(0..MAX_ORDER - 1, 1);
        self.buf[0] = data;
    }

    /// Recover the next value from its delta, advancing the order toward
    /// `level`.
    fn decompress(&mut self, delta: i64) -> core::result::Result<i64, validate::ArithmeticError> {
        let m = if self.m < self.level {
            self.m + 1
        } else {
            self.m
        };
        let b = &self.buf;
        let new = match m {
            1 => checked_diff_sum(delta, &[(1, b[0])])?,
            2 => checked_diff_sum(delta, &[(2, b[0]), (-1, b[1])])?,
            3 => checked_diff_sum(delta, &[(3, b[0]), (-3, b[1]), (1, b[2])])?,
            4 => checked_diff_sum(delta, &[(4, b[0]), (-6, b[1]), (4, b[2]), (-1, b[3])])?,
            5 => checked_diff_sum(
                delta,
                &[(5, b[0]), (-10, b[1]), (10, b[2]), (-5, b[3]), (1, b[4])],
            )?,
            6 => checked_diff_sum(
                delta,
                &[
                    (6, b[0]),
                    (-15, b[1]),
                    (20, b[2]),
                    (-15, b[3]),
                    (6, b[4]),
                    (-1, b[5]),
                ],
            )?,
            // m starts at 0 and is incremented before use, and `level` is
            // capped at MAX_ORDER, so m is always in 1..=MAX_ORDER here.
            _ => checked_diff_sum(delta, &[(1, b[0])])?,
        };
        self.m = m;
        self.rotate(new);
        Ok(new)
    }
}

fn checked_diff_sum(
    delta: i64,
    terms: &[(i64, i64)],
) -> core::result::Result<i64, validate::ArithmeticError> {
    const FIELD: &str = "crinex numeric difference";
    let mut sum = delta;
    for &(coefficient, value) in terms {
        let term = validate::checked_i64_mul(coefficient.abs(), value, FIELD)?;
        sum = if coefficient >= 0 {
            validate::checked_i64_add(sum, term, FIELD)?
        } else {
            validate::checked_i64_sub(sum, term, FIELD)?
        };
    }
    Ok(sum)
}

/// Per-character text difference engine (Hatanaka `TextDiff`), used for the
/// epoch descriptor line and each satellite's LLI/SSI flag string.
///
/// State is the last reconstructed string. A space keeps the buffered byte; an
/// `&` blanks it; any other byte overwrites it. Input longer than the buffer
/// appends verbatim.
#[derive(Debug, Default, Clone)]
struct TextDiff {
    buffer: Vec<u8>,
}

impl TextDiff {
    /// Replace the buffer wholesale (a forced reinit / first sample).
    fn force_init(&mut self, data: &str) {
        self.buffer = data.as_bytes().to_vec();
    }

    /// Apply a compressed line against the current buffer and return the
    /// reconstructed string.
    fn decompress(&mut self, data: &str) -> String {
        let bytes = data.as_bytes();
        if bytes.len() > self.buffer.len() {
            self.buffer.extend_from_slice(&bytes[self.buffer.len()..]);
        }
        for (i, &byte) in bytes.iter().enumerate() {
            if byte == b' ' {
                continue;
            }
            if let Some(slot) = self.buffer.get_mut(i) {
                *slot = if byte == b'&' { b' ' } else { byte };
            }
        }
        // CRINEX text is ASCII; lossy is unreachable for valid input but keeps
        // this panic-free on a stray byte.
        String::from_utf8_lossy(&self.buffer).into_owned()
    }
}

/// CRINEX decoder state machine.
struct Decoder {
    version: CrinexVersion,
    /// Number of observation codes declared per constellation letter, used to
    /// know how many observation fields each satellite line carries.
    obs_count: HashMap<char, usize>,
    /// Mono-constellation letter for a RINEX-2 file whose SV tokens omit the
    /// system letter (set from the header `RINEX VERSION / TYPE`).
    default_system: Option<char>,
    /// Epoch descriptor text-diff engine.
    epoch_diff: TextDiff,
    /// Receiver-clock-offset difference engine.
    clock_diff: Option<NumDiff>,
    /// Per-satellite observation difference engines, keyed by SV token.
    obs_diff: HashMap<String, Vec<Option<NumDiff>>>,
    /// Per-satellite flag (LLI/SSI) text-diff engines.
    flag_diff: HashMap<String, TextDiff>,
}

impl Decoder {
    fn new() -> Self {
        Self {
            version: CrinexVersion::V3,
            obs_count: HashMap::new(),
            default_system: None,
            epoch_diff: TextDiff::default(),
            clock_diff: None,
            obs_diff: HashMap::new(),
            flag_diff: HashMap::new(),
        }
    }

    /// Consume the two CRINEX header lines (dropped) and then copy the embedded
    /// plain RINEX header through verbatim up to and including `END OF HEADER`,
    /// recording the per-system observation-code counts and the file version
    /// along the way.
    fn read_crinex_header<'a, I, W>(&mut self, lines: &mut I, emit: &mut W) -> Result<()>
    where
        I: Iterator<Item = &'a str>,
        W: FnMut(&str),
    {
        // Line 1: CRINEX VERS / TYPE - selects the stream grammar.
        let l1 = lines
            .next()
            .ok_or_else(|| Error::Parse("CRINEX stream is empty".into()))?;
        let crx_ver = field(l1, 0, 20).trim();
        self.version = match crx_ver {
            v if v.starts_with("1.0") || v.starts_with("1.") => CrinexVersion::V1,
            v if v.starts_with("3.0") || v.starts_with("3.") => CrinexVersion::V3,
            other => {
                return Err(Error::Parse(format!(
                    "unsupported CRINEX version {other:?} (expected 1.0 or 3.0)"
                )))
            }
        };
        if !l1.contains("CRINEX VERS") {
            return Err(Error::Parse(
                "missing CRINEX VERS / TYPE header line".into(),
            ));
        }
        // Line 2: CRINEX PROG / DATE - dropped (it is the compaction stamp, not
        // part of the reconstructed RINEX).
        lines
            .next()
            .ok_or_else(|| Error::Parse("CRINEX header missing PROG / DATE line".into()))?;

        // Copy the embedded plain RINEX header verbatim, tracking obs counts.
        let mut saw_end = false;
        for raw in lines.by_ref() {
            let line = raw.trim_end_matches(['\r', '\n']);
            emit(line);

            let label = field(line, 60, 80).trim();
            self.classify_header_label(line, label)?;
            if label == "END OF HEADER" {
                saw_end = true;
                break;
            }
        }
        if !saw_end {
            return Err(Error::Parse(
                "CRINEX embedded RINEX header has no END OF HEADER".into(),
            ));
        }
        Ok(())
    }

    /// Record the per-system observation-code counts and the mono-system
    /// constellation letter from one labelled RINEX header record. Shared by the
    /// CRINEX-stream header reader and the plain-RINEX header scanner so both
    /// resolve observation widths identically.
    fn classify_header_label(&mut self, line: &str, label: &str) -> Result<()> {
        match label {
            "RINEX VERSION / TYPE" => {
                // RINEX 2 SV tokens may omit the constellation letter for a
                // single-system file; capture it for V1 streams.
                let sys_field = field(line, 40, 41).trim();
                if let Some(c) = sys_field.chars().next() {
                    if c != 'M' {
                        self.default_system = Some(c);
                    }
                }
            }
            "# / TYPES OF OBSERV" => {
                // RINEX 2 observation-code count (shared across systems).
                let n = strict_obs_count(line, 0, 6, "rinex2.obs_type_count")?;
                if let Some(sys) = self.default_system {
                    self.obs_count.insert(sys, n);
                }
                // RINEX 2 has one shared list; record under a sentinel so a
                // mono-system file resolves even if the letter was 'M'.
                self.obs_count.entry(' ').or_insert(n);
            }
            "SYS / # / OBS TYPES" => {
                let sys_field = field(line, 0, 1).trim();
                if let Some(c) = sys_field.chars().next() {
                    let n = strict_obs_count(line, 3, 6, "rinex3.obs_type_count")?;
                    self.obs_count.insert(c, n);
                }
                // Continuation lines (blank system field) carry no count.
            }
            _ => {}
        }
        Ok(())
    }

    /// Scan a plain RINEX observation header (no CRINEX wrapper): collect the
    /// header lines verbatim up to and including `END OF HEADER`, set the stream
    /// revision from `RINEX VERSION / TYPE`, and record the per-system
    /// observation-code counts via [`Self::classify_header_label`].
    fn scan_rinex_header<'a, I>(&mut self, lines: &mut I, header: &mut Vec<String>) -> Result<()>
    where
        I: Iterator<Item = &'a str>,
    {
        let mut saw_version = false;
        let mut saw_end = false;
        for raw in lines.by_ref() {
            let line = raw.trim_end_matches(['\r', '\n']);
            let label = field(line, 60, 80).trim();
            if label == "RINEX VERSION / TYPE" {
                let version = field(line, 0, 9).trim();
                self.version = match version.chars().next() {
                    Some('2') => CrinexVersion::V1,
                    Some('3') => CrinexVersion::V3,
                    _ => {
                        return Err(Error::Parse(format!(
                            "unsupported RINEX version {version:?} (expected 2 or 3)"
                        )))
                    }
                };
                saw_version = true;
            }
            self.classify_header_label(line, label)?;
            header.push(line.to_string());
            if label == "END OF HEADER" {
                saw_end = true;
                break;
            }
        }
        if !saw_version {
            return Err(Error::Parse(
                "plain RINEX header missing RINEX VERSION / TYPE".into(),
            ));
        }
        if !saw_end {
            return Err(Error::Parse(
                "plain RINEX observation header has no END OF HEADER".into(),
            ));
        }
        Ok(())
    }

    /// Decode the epoch records following the header into plain RINEX text.
    fn read_body<'a, I, W>(&mut self, lines: &mut I, emit: &mut W) -> Result<()>
    where
        I: Iterator<Item = &'a str>,
        W: FnMut(&str),
    {
        let version = self.version;
        loop {
            let record = match version {
                CrinexVersion::V3 => self.next_epoch_v3(lines)?,
                CrinexVersion::V1 => self.next_epoch_v1(lines)?,
            };
            let Some(record) = record else { break };
            match version {
                CrinexVersion::V3 => serialize_rinex_epoch_v3(&record, emit),
                CrinexVersion::V1 => serialize_rinex_epoch_v1(&record, emit),
            }
        }
        Ok(())
    }

    // ----------------------------------------------------------------- V3 ---

    /// Parse the next V3 epoch into the canonical [`EpochRecord`], advancing the
    /// difference engines. Returns `Ok(None)` at end of stream. Stray blank lines
    /// between records are skipped.
    fn next_epoch_v3<'a, I>(&mut self, lines: &mut I) -> Result<Option<EpochRecord>>
    where
        I: Iterator<Item = &'a str>,
    {
        let raw = loop {
            match lines.next() {
                None => return Ok(None),
                Some(raw) => {
                    let line = raw.trim_end_matches(['\r', '\n']);
                    if !line.is_empty() {
                        break line;
                    }
                }
            }
        };

        // Epoch descriptor. A reset is marked by a leading '>' (the rest of the
        // line is taken literally), otherwise the line is a TextDiff delta of the
        // previous descriptor. The leading '>' is kept in the text-diff buffer so
        // the delta lines' column offsets line up with the full RINEX-3 epoch line
        // (the seconds digits sit one column to the right of where they would be
        // in a '>'-stripped buffer).
        let descriptor = if raw.starts_with('>') {
            self.epoch_diff.force_init(raw);
            self.epoch_diff.decompress("")
        } else {
            self.epoch_diff.decompress(raw)
        };

        // The reconstructed full epoch line is
        // "> YYYY MM DD HH MM SS.sssssss  F NN<svlist>": the epoch flag is at
        // column 31, the satellite count at columns 32..35, and the 3-char SV
        // tokens begin at column 41.
        let numsat = strict_int_field::<usize>(&descriptor, 32, 35, "v3.epoch.satellite_count")?;
        let flag = strict_int_field::<u8>(&descriptor, 31, 32, "v3.epoch.flag")?;

        // Event records (flag > 1) carry header/comment lines rather than
        // observation lines, and the clock-offset record is omitted for them
        // entirely. Capture the `numsat` event lines verbatim, skip differencing.
        if flag > 1 {
            let mut event_lines = Vec::with_capacity(numsat);
            for _ in 0..numsat {
                let extra = lines
                    .next()
                    .ok_or_else(|| Error::Parse("CRINEX V3 event record truncated".into()))?;
                event_lines.push(extra.trim_end_matches(['\r', '\n']).to_string());
            }
            return Ok(Some(EpochRecord::Event {
                descriptor,
                lines: event_lines,
            }));
        }

        // The clock offset is its own line (a NumDiff token, possibly blank).
        let clock_line = lines
            .next()
            .ok_or_else(|| Error::Parse("CRINEX V3 epoch missing clock line".into()))?
            .trim_end_matches(['\r', '\n']);
        let clock = self.decode_clock_value(clock_line)?;

        let sv_list = self.sv_tokens_v3(&descriptor, numsat)?;
        let mut sats = Vec::with_capacity(sv_list.len());
        for sv in &sv_list {
            let data_line = lines.next().ok_or_else(|| {
                Error::Parse("CRINEX V3 epoch truncated: missing satellite line".into())
            })?;
            let n_obs = self.obs_count_for(sv)?;
            let (values, flags) =
                self.decode_sat_values(sv, data_line.trim_end_matches(['\r', '\n']), n_obs)?;
            sats.push(SatRecord {
                sv: sv.clone(),
                values,
                flags,
            });
        }
        Ok(Some(EpochRecord::Obs(ObsEpoch {
            descriptor,
            clock,
            sats,
        })))
    }

    /// Extract `numsat` 3-character SV tokens from the V3 epoch descriptor.
    fn sv_tokens_v3(&self, descriptor: &str, numsat: usize) -> Result<Vec<String>> {
        // The RINEX-3 epoch line pads the satellite list to column 41 of the
        // full line (the '>' is kept in the descriptor buffer); the 3-char SV
        // tokens run from there.
        let list = field_from(descriptor, 41);
        let bytes = list.as_bytes();
        let mut out = Vec::with_capacity(numsat);
        for i in 0..numsat {
            out.push(fixed_sv_token(bytes, "V3", numsat, i)?.to_string());
        }
        Ok(out)
    }

    /// Observation-code count for an SV token's constellation.
    fn obs_count_for(&self, sv: &str) -> Result<usize> {
        let sys = sv.chars().next().unwrap_or(' ');
        let count = self
            .obs_count
            .get(&sys)
            .or_else(|| self.obs_count.get(&' '))
            .copied()
            .ok_or_else(|| {
                Error::Parse(format!(
                    "CRINEX satellite {sv:?} has no declared observation count"
                ))
            })?;
        if count == 0 {
            return Err(Error::Parse(format!(
                "CRINEX satellite {sv:?} has zero declared observations"
            )));
        }
        Ok(count)
    }

    /// Recover one satellite's observations from a CRINEX data line: `n_obs`
    /// difference-coded observation tokens followed by the TextDiff flag string.
    /// The on-wire data-line grammar is identical for V1 and V3 (the SV token is
    /// carried by the epoch descriptor, not the data line), so both revisions
    /// share this recovery; only the RINEX-text layout of the result differs.
    fn decode_sat_values(
        &mut self,
        sv: &str,
        line: &str,
        n_obs: usize,
    ) -> Result<(Vec<Option<i64>>, String)> {
        // The observation tokens are whitespace-separated; the remainder after
        // the last consumed token is the flag string. We walk the line token by
        // token, tracking byte offsets so we know where the flags begin.
        let engines = self
            .obs_diff
            .entry(sv.to_string())
            .or_insert_with(|| vec![None; n_obs]);
        if engines.len() < n_obs {
            engines.resize(n_obs, None);
        }

        let mut values: Vec<Option<i64>> = Vec::with_capacity(n_obs);
        let mut cursor = 0usize;
        let bytes = line.as_bytes();

        for obs_index in 0..n_obs {
            // Skip the single separating blank between fields (the compressor
            // writes exactly one space between tokens; a doubled space marks a
            // blanked observation).
            if obs_index > 0 {
                if cursor < bytes.len() && bytes[cursor] == b' ' {
                    cursor += 1;
                } else if cursor >= bytes.len() {
                    // No more tokens on the line: the rest are blank.
                    values.push(None);
                    continue;
                }
            }
            // A blanked observation: the field is empty (immediately another
            // separator or end of the data section).
            if cursor >= bytes.len() || bytes[cursor] == b' ' {
                values.push(None);
                continue;
            }
            // Read the token up to the next space.
            let tok_start = cursor;
            while cursor < bytes.len() && bytes[cursor] != b' ' {
                cursor += 1;
            }
            let token = &line[tok_start..cursor];
            let recovered = self.apply_obs_token(sv, obs_index, token)?;
            values.push(Some(recovered));
        }

        // The flag string is whatever remains. In RNX2CRX output the flags are
        // separated from the last observation token by a single space.
        let flag_raw = if cursor < bytes.len() {
            let rest = &line[cursor..];
            rest.strip_prefix(' ').unwrap_or(rest)
        } else {
            ""
        };
        let flags = self
            .flag_diff
            .entry(sv.to_string())
            .or_default()
            .decompress(flag_raw);

        Ok((values, flags))
    }

    /// Apply one observation token (reset `order&value`, or a plain delta) and
    /// return the recovered scaled integer.
    fn apply_obs_token(&mut self, sv: &str, obs_index: usize, token: &str) -> Result<i64> {
        let engines = self.obs_diff.get_mut(sv).expect("engines inserted above");
        let slot = &mut engines[obs_index];
        if let Some((order, value)) = parse_reset(token)? {
            match slot {
                Some(e) => e.force_init(value, order),
                None => *slot = Some(NumDiff::new(value, order)),
            }
            Ok(value)
        } else {
            let delta = token.trim().parse::<i64>().map_err(|_| {
                Error::Parse(format!(
                    "CRINEX observation delta {token:?} is not an integer"
                ))
            })?;
            let Some(engine) = slot else {
                return Err(Error::Parse(format!(
                    "CRINEX observation {sv}[{obs_index}] has a delta before any arc init"
                )));
            };
            engine.decompress(delta).map_err(map_arithmetic_error)
        }
    }

    /// Recover the per-epoch receiver clock offset from its line as the scaled
    /// integer the stream carried (`None` when no clock token is present),
    /// advancing the clock difference engine. The picosecond/nanosecond scaling
    /// to text is applied by the RINEX-text serializer, not here.
    fn decode_clock_value(&mut self, line: &str) -> Result<Option<i64>> {
        let token = line.trim();
        if token.is_empty() {
            return Ok(None);
        }
        let value = if let Some((order, v)) = parse_reset(token)? {
            match &mut self.clock_diff {
                Some(e) => e.force_init(v, order),
                None => self.clock_diff = Some(NumDiff::new(v, order)),
            }
            v
        } else {
            let delta = token.parse::<i64>().map_err(|_| {
                Error::Parse(format!("CRINEX clock delta {token:?} is not an integer"))
            })?;
            match &mut self.clock_diff {
                Some(e) => e.decompress(delta).map_err(map_arithmetic_error)?,
                None => {
                    return Err(Error::Parse(
                        "CRINEX clock delta before any clock arc init".into(),
                    ))
                }
            }
        };
        Ok(Some(value))
    }

    // ----------------------------------------------------------------- V1 ---

    /// Parse the next V1 epoch into the canonical [`EpochRecord`]. See
    /// [`Self::next_epoch_v3`] for the streaming contract.
    fn next_epoch_v1<'a, I>(&mut self, lines: &mut I) -> Result<Option<EpochRecord>>
    where
        I: Iterator<Item = &'a str>,
    {
        let raw = loop {
            match lines.next() {
                None => return Ok(None),
                Some(raw) => {
                    let line = raw.trim_end_matches(['\r', '\n']);
                    if !line.is_empty() {
                        break line;
                    }
                }
            }
        };

        // CRINEX 1.0 stores the epoch line without RINEX-2's leading blank
        // column, but crx2rnx restores it on output and keeps the restored,
        // space-prefixed line as the text-difference base for the next epoch.
        // Mirror that: seed the engine with the leading space (on reset) so both
        // the reconstruction and the standard column offsets are right. A V1
        // epoch descriptor reset is marked by a leading '&'.
        let descriptor = if let Some(stripped) = raw.strip_prefix('&') {
            self.epoch_diff.force_init(&format!(" {stripped}"));
            self.epoch_diff.decompress("")
        } else {
            self.epoch_diff.decompress(raw)
        };

        // V1 epoch line: " YY MM DD HH MM SS.sssssss  F NN<svlist>". numsat is at
        // cols 29..32 of the reconstructed RINEX-2 epoch line (which the
        // descriptor mirrors, leading space included).
        let numsat = strict_int_field::<usize>(&descriptor, 29, 32, "v1.epoch.satellite_count")?;
        let flag = strict_int_field::<u8>(&descriptor, 26, 29, "v1.epoch.flag")?;

        // Event records (flag > 1): capture the `numsat` event lines verbatim.
        if flag > 1 {
            let mut event_lines = Vec::with_capacity(numsat);
            for _ in 0..numsat {
                let extra = lines
                    .next()
                    .ok_or_else(|| Error::Parse("CRINEX V1 event record truncated".into()))?;
                event_lines.push(extra.trim_end_matches(['\r', '\n']).to_string());
            }
            return Ok(Some(EpochRecord::Event {
                descriptor,
                lines: event_lines,
            }));
        }

        // Clock line (its own NumDiff line, possibly blank).
        let clock_line = lines
            .next()
            .ok_or_else(|| Error::Parse("CRINEX V1 epoch missing clock line".into()))?
            .trim_end_matches(['\r', '\n']);
        let clock = self.decode_clock_value(clock_line)?;

        let sv_list = self.sv_tokens_v1(&descriptor, numsat)?;
        let mut sats = Vec::with_capacity(sv_list.len());
        for sv in &sv_list {
            let data_line = lines.next().ok_or_else(|| {
                Error::Parse("CRINEX V1 epoch truncated: missing satellite line".into())
            })?;
            let n_obs = self.obs_count_for(sv)?;
            let (values, flags) =
                self.decode_sat_values(sv, data_line.trim_end_matches(['\r', '\n']), n_obs)?;
            sats.push(SatRecord {
                sv: sv.clone(),
                values,
                flags,
            });
        }
        Ok(Some(EpochRecord::Obs(ObsEpoch {
            descriptor,
            clock,
            sats,
        })))
    }

    fn sv_tokens_v1(&self, descriptor: &str, numsat: usize) -> Result<Vec<String>> {
        // RINEX-2 SV list starts at col 32 of the epoch line; tokens are 3 chars
        // and may omit the constellation letter for a mono-system file.
        let list = field_from(descriptor, 32);
        let bytes = list.as_bytes();
        let mut out = Vec::with_capacity(numsat);
        for i in 0..numsat {
            let mut tok = fixed_sv_token(bytes, "V1", numsat, i)?.to_string();
            if tok.starts_with(' ') {
                if let Some(sys) = self.default_system {
                    let prn = tok.trim();
                    tok = format!("{sys}{prn:>2}");
                }
            }
            out.push(tok);
        }
        Ok(out)
    }

    // ---------------------------------------------------- plain RINEX -> IR ---

    /// Parse the body of a plain RINEX-3 observation file into canonical epoch
    /// records (the inverse of [`serialize_rinex_epoch_v3`]).
    fn parse_rinex_epochs_v3<'a, I>(&self, lines: &mut I) -> Result<Vec<EpochRecord>>
    where
        I: Iterator<Item = &'a str>,
    {
        let mut epochs = Vec::new();
        loop {
            let Some(line) = next_nonblank(lines) else {
                return Ok(epochs);
            };
            if !line.starts_with('>') {
                return Err(Error::Parse(format!(
                    "RINEX-3 epoch line must start with '>': {line:?}"
                )));
            }
            let flag = strict_int_field::<u8>(&line, 31, 32, "v3.epoch.flag")?;
            let numsat = strict_int_field::<usize>(&line, 32, 35, "v3.epoch.satellite_count")?;

            if flag > 1 {
                let event_lines = read_event_lines(lines, numsat, "RINEX-3")?;
                epochs.push(EpochRecord::Event {
                    descriptor: line,
                    lines: event_lines,
                });
                continue;
            }

            let clock = parse_clock_field(&line, 41, 56, 12, "v3.epoch.clock")?;
            let mut sats = Vec::with_capacity(numsat);
            let mut sv_tokens = Vec::with_capacity(numsat);
            for _ in 0..numsat {
                let raw = lines.next().ok_or_else(|| {
                    Error::Parse("RINEX-3 epoch truncated: missing satellite line".into())
                })?;
                let sat_line = raw.trim_end_matches(['\r', '\n']);
                let sv = field(sat_line, 0, 3).to_string();
                let n_obs = self.obs_count_for(&sv)?;
                let (values, flags) = parse_sat_obs_v3(sat_line, n_obs)?;
                sv_tokens.push(sv.clone());
                sats.push(SatRecord { sv, values, flags });
            }
            let descriptor = build_descriptor_v3(&line, &sv_tokens);
            epochs.push(EpochRecord::Obs(ObsEpoch {
                descriptor,
                clock,
                sats,
            }));
        }
    }

    /// Parse the body of a plain RINEX-2 observation file into canonical epoch
    /// records (the inverse of [`serialize_rinex_epoch_v1`]). Handles the
    /// 12-satellite epoch-line wrap and the 5-observation data-line wrap.
    fn parse_rinex_epochs_v1<'a, I>(&self, lines: &mut I) -> Result<Vec<EpochRecord>>
    where
        I: Iterator<Item = &'a str>,
    {
        let mut epochs = Vec::new();
        loop {
            let Some(first) = next_nonblank(lines) else {
                return Ok(epochs);
            };
            let flag = strict_int_field::<u8>(&first, 26, 29, "v1.epoch.flag")?;
            let numsat = strict_int_field::<usize>(&first, 29, 32, "v1.epoch.satellite_count")?;

            if flag > 1 {
                let event_lines = read_event_lines(lines, numsat, "RINEX-2")?;
                epochs.push(EpochRecord::Event {
                    descriptor: first,
                    lines: event_lines,
                });
                continue;
            }

            let clock = parse_clock_field(&first, 68, 80, 9, "v1.epoch.clock")?;

            // The SV list begins at column 32 and wraps after 12 satellites onto
            // continuation lines, each padded with 32 leading blanks.
            let mut sv_tokens: Vec<String> = Vec::with_capacity(numsat);
            collect_sv_tokens_v1(&first, numsat.min(12), &mut sv_tokens);
            while sv_tokens.len() < numsat {
                let raw = lines.next().ok_or_else(|| {
                    Error::Parse("RINEX-2 epoch SV continuation truncated".into())
                })?;
                let cont = raw.trim_end_matches(['\r', '\n']);
                let need = (numsat - sv_tokens.len()).min(12);
                collect_sv_tokens_v1(cont, need, &mut sv_tokens);
            }
            let sv_tokens: Vec<String> = sv_tokens
                .into_iter()
                .map(|tok| self.normalize_v1_sv(tok))
                .collect();

            let mut sats = Vec::with_capacity(numsat);
            for sv in &sv_tokens {
                let n_obs = self.obs_count_for(sv)?;
                let row_count = n_obs.div_ceil(5);
                let mut obs_lines = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    let raw = lines.next().ok_or_else(|| {
                        Error::Parse("RINEX-2 epoch truncated: missing observation line".into())
                    })?;
                    obs_lines.push(raw.trim_end_matches(['\r', '\n']).to_string());
                }
                let (values, flags) = parse_sat_obs_v1(&obs_lines, n_obs)?;
                sats.push(SatRecord {
                    sv: sv.clone(),
                    values,
                    flags,
                });
            }
            let descriptor = build_descriptor_v1(&first, &sv_tokens);
            epochs.push(EpochRecord::Obs(ObsEpoch {
                descriptor,
                clock,
                sats,
            }));
        }
    }

    /// Re-attach the mono-system constellation letter to a RINEX-2 SV token that
    /// omits it, matching [`Self::sv_tokens_v1`].
    fn normalize_v1_sv(&self, token: String) -> String {
        if token.starts_with(' ') {
            if let Some(sys) = self.default_system {
                let prn = token.trim();
                return format!("{sys}{prn:>2}");
            }
        }
        token
    }
}

// ── Plain RINEX -> IR ─────────────────────────────────────────────────────────

/// Parse plain RINEX observation text into the canonical [`ObsStream`] IR (the
/// inverse of the RINEX serializers driving [`decode`]). The embedded header is
/// captured verbatim, the revision is taken from `RINEX VERSION / TYPE`, and each
/// epoch's fixed-decimal observation fields are read back into scaled integers.
fn parse_rinex_obs(rinex_text: &str) -> Result<ObsStream> {
    let mut decoder = Decoder::new();
    let mut header: Vec<String> = Vec::new();
    let mut lines = rinex_text.lines();
    decoder.scan_rinex_header(&mut lines, &mut header)?;

    let version = decoder.version;
    let epochs = match version {
        CrinexVersion::V3 => decoder.parse_rinex_epochs_v3(&mut lines)?,
        CrinexVersion::V1 => decoder.parse_rinex_epochs_v1(&mut lines)?,
    };
    Ok(ObsStream {
        version,
        header,
        epochs,
    })
}

/// Pull the next non-blank line from the body, returning it without its trailing
/// line ending, or `None` at end of stream.
fn next_nonblank<'a, I>(lines: &mut I) -> Option<String>
where
    I: Iterator<Item = &'a str>,
{
    for raw in lines.by_ref() {
        let line = raw.trim_end_matches(['\r', '\n']);
        if !line.is_empty() {
            return Some(line.to_string());
        }
    }
    None
}

/// Read the `count` verbatim records that follow an event epoch descriptor.
fn read_event_lines<'a, I>(lines: &mut I, count: usize, revision: &str) -> Result<Vec<String>>
where
    I: Iterator<Item = &'a str>,
{
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let raw = lines
            .next()
            .ok_or_else(|| Error::Parse(format!("{revision} event record truncated")))?;
        out.push(raw.trim_end_matches(['\r', '\n']).to_string());
    }
    Ok(out)
}

/// Read an optional receiver-clock field (`Fw.d`) as the scaled integer the
/// CRINEX clock engine carries, or `None` when the field is blank.
fn parse_clock_field(
    line: &str,
    start: usize,
    end: usize,
    decimals: usize,
    field_name: &'static str,
) -> Result<Option<i64>> {
    let text = field(line, start, end);
    if text.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(parse_scaled_decimal(text, decimals, field_name)?))
    }
}

/// Recover one RINEX-3 satellite line's observations: `n_obs` fixed 16-column
/// fields, each a 14-column `F14.3` value plus a 2-column LLI/SSI pair. A blank
/// value column is a `None`; trailing blanks may be trimmed from the line.
fn parse_sat_obs_v3(line: &str, n_obs: usize) -> Result<(Vec<Option<i64>>, String)> {
    let mut values = Vec::with_capacity(n_obs);
    let mut flags = String::with_capacity(n_obs * 2);
    for i in 0..n_obs {
        let base = 3 + i * OBS_FIELD_WIDTH;
        read_obs_field(line, base, &mut values, &mut flags)?;
    }
    Ok((values, flags))
}

/// Recover one RINEX-2 satellite's observations from its wrapped data lines (five
/// 16-column fields per line). See [`parse_sat_obs_v3`] for the field layout.
fn parse_sat_obs_v1(obs_lines: &[String], n_obs: usize) -> Result<(Vec<Option<i64>>, String)> {
    let mut values = Vec::with_capacity(n_obs);
    let mut flags = String::with_capacity(n_obs * 2);
    for i in 0..n_obs {
        let line = obs_lines.get(i / 5).map_or("", String::as_str);
        let base = (i % 5) * OBS_FIELD_WIDTH;
        read_obs_field(line, base, &mut values, &mut flags)?;
    }
    Ok((values, flags))
}

/// Read one observation field at column `base` of `line`, pushing the recovered
/// value (or `None` for a blank column) and its two LLI/SSI flag characters.
fn read_obs_field(
    line: &str,
    base: usize,
    values: &mut Vec<Option<i64>>,
    flags: &mut String,
) -> Result<()> {
    let value_text = field(line, base, base + OBS_VALUE_WIDTH);
    if value_text.trim().is_empty() {
        values.push(None);
        flags.push(' ');
        flags.push(' ');
    } else {
        values.push(Some(parse_scaled_decimal(value_text, 3, "observation")?));
        flags.push(char_at_or_space(line, base + OBS_VALUE_WIDTH));
        flags.push(char_at_or_space(line, base + OBS_VALUE_WIDTH + 1));
    }
    Ok(())
}

/// Parse a fixed-decimal field into the scaled integer it represents, exactly:
/// `value * 10^decimals`. The fraction is read digit-by-digit (not through a
/// float) so the three-decimal observation layout and the `-.920`-style dropped
/// leading zero round-trip without binary-float rounding.
fn parse_scaled_decimal(text: &str, decimals: usize, field_name: &'static str) -> Result<i64> {
    let trimmed = text.trim();
    let (negative, body) = trimmed.strip_prefix('-').map_or_else(
        || (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
        |rest| (true, rest),
    );
    let (integer_part, fraction_part) = match body.split_once('.') {
        Some((integer, fraction)) => (integer, fraction),
        None => (body, ""),
    };
    let integer_text = if integer_part.is_empty() {
        "0"
    } else {
        integer_part
    };
    // Right-pad (or clip) the fraction to exactly `decimals` digits.
    let mut fraction = String::with_capacity(decimals);
    fraction.extend(fraction_part.chars().take(decimals));
    while fraction.len() < decimals {
        fraction.push('0');
    }
    let scale = 10i64.pow(decimals as u32);
    let integer_value = parse_scaled_component(integer_text, text, field_name)?;
    let fraction_value = if decimals == 0 {
        0
    } else {
        parse_scaled_component(&fraction, text, field_name)?
    };
    let magnitude = validate::checked_i64_mul(integer_value, scale, field_name)
        .and_then(|scaled| validate::checked_i64_add(scaled, fraction_value, field_name))
        .map_err(map_arithmetic_error)?;
    Ok(if negative { -magnitude } else { magnitude })
}

/// Parse one all-digit component of a scaled-decimal field.
fn parse_scaled_component(token: &str, text: &str, field_name: &'static str) -> Result<i64> {
    token
        .parse::<i64>()
        .map_err(|_| Error::Parse(format!("CRINEX invalid {field_name}: {text:?}")))
}

/// Build the canonical V3 epoch descriptor from the RINEX-3 epoch line and the
/// SV list gathered from the satellite data lines: the cols 0..35 head, padded to
/// column 41, then the concatenated 3-character SV tokens (where the CRINEX epoch
/// line and [`Decoder::sv_tokens_v3`] expect them).
fn build_descriptor_v3(epoch_line: &str, sv_tokens: &[String]) -> String {
    let mut descriptor = pad_to(field(epoch_line, 0, 35), 41);
    for token in sv_tokens {
        descriptor.push_str(token);
    }
    descriptor
}

/// Build the canonical V1 epoch descriptor: the cols 0..32 head (leading space
/// included) then the full concatenated SV list, matching the single-line CRINEX
/// epoch record [`Decoder::sv_tokens_v1`] reads back.
fn build_descriptor_v1(epoch_line: &str, sv_tokens: &[String]) -> String {
    let mut descriptor = pad_to(field(epoch_line, 0, 32), 32);
    for token in sv_tokens {
        descriptor.push_str(token);
    }
    descriptor
}

/// Append up to `count` 3-character SV tokens read from column 32 onward.
fn collect_sv_tokens_v1(line: &str, count: usize, out: &mut Vec<String>) {
    for i in 0..count {
        let start = 32 + i * 3;
        out.push(field(line, start, start + 3).to_string());
    }
}

/// The ASCII byte at `index` as a `char`, or a space when the line is shorter
/// (trailing blanks are trimmed from RINEX output lines).
fn char_at_or_space(line: &str, index: usize) -> char {
    line.as_bytes().get(index).map_or(' ', |&byte| byte as char)
}

/// Right-pad a field with spaces to at least `width` columns (never truncating).
fn pad_to(text: &str, width: usize) -> String {
    let mut out = text.to_string();
    while out.len() < width {
        out.push(' ');
    }
    out
}

// ── Canonical-IR parse and serializers ───────────────────────────────────────

/// Parse a CRINEX stream into the canonical [`ObsStream`] IR (the inverse of
/// [`encode_stream`]). The plain RINEX header is captured verbatim and every
/// epoch's difference engines are undone into recovered integers and flag
/// strings.
pub fn parse_stream(crinex_text: &str) -> Result<ObsStream> {
    let mut decoder = Decoder::new();
    let mut lines = crinex_text.lines();
    let mut header: Vec<String> = Vec::new();
    decoder.read_crinex_header(&mut lines, &mut |line: &str| header.push(line.to_string()))?;

    let version = decoder.version;
    let mut epochs = Vec::new();
    loop {
        let record = match version {
            CrinexVersion::V3 => decoder.next_epoch_v3(&mut lines)?,
            CrinexVersion::V1 => decoder.next_epoch_v1(&mut lines)?,
        };
        match record {
            Some(record) => epochs.push(record),
            None => break,
        }
    }
    Ok(ObsStream {
        version,
        header,
        epochs,
    })
}

/// Serialize a canonical [`ObsStream`] back to CRINEX text (the inverse of
/// [`parse_stream`]).
///
/// CRINEX compression is not unique, so this emits the **canonical all-reset**
/// form: every observation and the receiver clock are written as `1&value`
/// arc-init tokens (no higher-order differencing) and every epoch descriptor is
/// written as a text-diff reset. Only the per-satellite LLI/SSI flag strings are
/// genuinely text-differenced, because the flag grammar has no inline reset
/// marker. The result is therefore not byte-identical to an arbitrary source
/// CRINEX, but it is a valid CRINEX stream that decodes to exactly the same plain
/// RINEX text. The round-trip guarantee is `decode(encode_stream(parse_stream(x)))
/// == decode(x)` and `parse_stream(encode_stream(s)) == s`.
pub fn encode_stream(stream: &ObsStream) -> String {
    let mut out = String::new();
    let version_label = match stream.version {
        CrinexVersion::V3 => "3.0",
        CrinexVersion::V1 => "1.0",
    };
    push_crinex_line(
        &mut out,
        &labeled_crinex(version_label, "CRINEX VERS   / TYPE"),
    );
    push_crinex_line(
        &mut out,
        &labeled_crinex("sidereon", "CRINEX PROG   / DATE"),
    );
    for header_line in &stream.header {
        push_crinex_line(&mut out, header_line);
    }

    let mut flag_state: HashMap<String, String> = HashMap::new();
    for epoch in &stream.epochs {
        encode_epoch(epoch, stream.version, &mut flag_state, &mut out);
    }
    out
}

/// Emit one epoch (observation or event) in canonical all-reset CRINEX form.
fn encode_epoch(
    epoch: &EpochRecord,
    version: CrinexVersion,
    flag_state: &mut HashMap<String, String>,
    out: &mut String,
) {
    match epoch {
        EpochRecord::Event { descriptor, lines } => {
            encode_descriptor(descriptor, version, out);
            for line in lines {
                push_crinex_line(out, line);
            }
        }
        EpochRecord::Obs(ObsEpoch {
            descriptor,
            clock,
            sats,
        }) => {
            encode_descriptor(descriptor, version, out);
            // An observation epoch always carries a clock line (possibly blank).
            match clock {
                Some(value) => push_crinex_line(out, &format!("1&{value}")),
                None => push_crinex_line(out, ""),
            }
            for sat in sats {
                let previous = flag_state.entry(sat.sv.clone()).or_default();
                let delta = text_diff_delta(previous.as_str(), &sat.flags);
                previous.clone_from(&sat.flags);
                push_crinex_line(out, &encode_sat_line(&sat.values, &delta));
            }
        }
    }
}

/// Emit the epoch descriptor as a text-diff reset for the given revision.
fn encode_descriptor(descriptor: &str, version: CrinexVersion, out: &mut String) {
    match version {
        // The reconstructed V3 descriptor already begins with '>'; a leading '>'
        // is exactly the V3 reset marker, so it re-emits verbatim.
        CrinexVersion::V3 => push_crinex_line(out, descriptor),
        // The reconstructed V1 descriptor begins with the restored leading space;
        // the V1 reset marker '&' replaces it and the decoder re-prepends a space.
        CrinexVersion::V1 => push_crinex_line(out, &format!("&{}", &descriptor[1..])),
    }
}

/// Build one CRINEX satellite data line: the observation tokens (each a `1&value`
/// arc-init reset, blank columns left empty) then the text-diff flag delta.
fn encode_sat_line(values: &[Option<i64>], flag_delta: &str) -> String {
    let mut line = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            line.push(' ');
        }
        if let Some(value) = value {
            let _ = write!(line, "1&{value}");
        }
    }
    // The flag string follows the last observation token after a single space.
    line.push(' ');
    line.push_str(flag_delta);
    line
}

/// Compute the CRINEX text-difference delta that transforms `previous` into
/// `current` under [`TextDiff::decompress`]: a space keeps the buffered byte, an
/// `&` blanks it, any other byte overwrites it, and bytes past the buffer extend
/// it verbatim. Per-satellite flag strings never shrink across epochs (the
/// buffer only grows), so no shortening case is needed.
fn text_diff_delta(previous: &str, current: &str) -> String {
    let prev = previous.as_bytes();
    let curr = current.as_bytes();
    let mut delta = Vec::with_capacity(curr.len());
    for (index, &byte) in curr.iter().enumerate() {
        let out = match prev.get(index) {
            Some(&previous_byte) if byte == previous_byte => b' ',
            Some(_) if byte == b' ' => b'&',
            // New non-space byte, or a position past the previous buffer (which
            // the decoder extends verbatim): emit the byte itself.
            _ => byte,
        };
        delta.push(out);
    }
    // Inputs are ASCII LLI/SSI flag strings, so this is always valid UTF-8.
    String::from_utf8(delta).unwrap_or_default()
}

/// Push a line plus its newline to a CRINEX output buffer.
fn push_crinex_line(out: &mut String, line: &str) {
    out.push_str(line);
    out.push('\n');
}

/// A labeled CRINEX/RINEX header record: body left-justified into the tag column.
fn labeled_crinex(body: &str, label: &str) -> String {
    format!("{body:<60}{label}")
}

/// Serialize one decoded V3 epoch back to plain RINEX-3 observation text.
fn serialize_rinex_epoch_v3<W: FnMut(&str)>(record: &EpochRecord, emit: &mut W) {
    match record {
        EpochRecord::Event { descriptor, lines } => {
            emit(trim_end(field(descriptor, 0, 35)));
            for line in lines {
                emit(line);
            }
        }
        EpochRecord::Obs(ObsEpoch {
            descriptor,
            clock,
            sats,
        }) => {
            let clock_text = format_clock_v3(*clock);
            // Everything before the SV list (cols 0..35) plus the clock. The SV
            // list is not part of a RINEX-3 epoch line. The optional receiver
            // clock offset is an `F15.12` field at columns 41..56, with columns
            // 35..41 reserved blank, so the head is padded to column 41 first.
            let head = field(descriptor, 0, 35);
            let mut epoch_out = head.to_string();
            if !clock_text.is_empty() {
                while epoch_out.len() < 41 {
                    epoch_out.push(' ');
                }
            }
            epoch_out.push_str(&clock_text);
            emit(trim_end(&epoch_out));
            for sat in sats {
                let out = format_sat_line(&sat.sv, &sat.values, &sat.flags);
                emit(trim_end(&out));
            }
        }
    }
}

/// Serialize one decoded V1 epoch back to plain RINEX-2 observation text.
fn serialize_rinex_epoch_v1<W: FnMut(&str)>(record: &EpochRecord, emit: &mut W) {
    match record {
        EpochRecord::Event { descriptor, lines } => {
            emit(trim_end(field(descriptor, 0, 32)));
            for line in lines {
                emit(line);
            }
        }
        EpochRecord::Obs(ObsEpoch {
            descriptor,
            clock,
            sats,
        }) => {
            let clock_text = format_clock_v1(*clock);
            // The SV list wraps after 12 satellites with a 32-space pad.
            let sv_list: Vec<String> = sats.iter().map(|sat| sat.sv.clone()).collect();
            for line in &format_epoch_v1(descriptor, &sv_list, &clock_text) {
                emit(trim_end(line));
            }
            for sat in sats {
                for line in format_sat_lines_v1(&sat.values, &sat.flags) {
                    emit(trim_end(&line));
                }
            }
        }
    }
}

/// Format the recovered V3 receiver clock offset (scaled by 10^12) as the
/// `%15.12f` field appended to the epoch line; empty when no clock is carried.
fn format_clock_v3(clock: Option<i64>) -> String {
    match clock {
        Some(value) => format!("{:15.12}", value as f64 / 1.0e12),
        None => String::new(),
    }
}

/// Format the recovered V1 receiver clock offset (scaled by 10^9) as the RINEX-2
/// `%12.9f` field; empty when no clock is carried.
fn format_clock_v1(clock: Option<i64>) -> String {
    match clock {
        Some(value) => format!("{:12.9}", value as f64 / 1.0e9),
        None => String::new(),
    }
}

fn strict_obs_count(
    line: &str,
    start: usize,
    end: usize,
    field_name: &'static str,
) -> Result<usize> {
    let count = strict_int_field::<usize>(line, start, end, field_name)?;
    if count == 0 {
        return Err(Error::Parse(format!(
            "CRINEX invalid {field_name}: observation count must be positive in {line:?}"
        )));
    }
    Ok(count)
}

fn strict_int_field<T>(line: &str, start: usize, end: usize, field_name: &'static str) -> Result<T>
where
    T: core::str::FromStr,
{
    strict_int_token(field(line, start, end), field_name, line)
}

fn strict_int_token<T>(token: &str, field_name: &'static str, line: &str) -> Result<T>
where
    T: core::str::FromStr,
{
    validate::strict_int::<T>(token, field_name).map_err(|error| map_field_error(error, line))
}

fn fixed_sv_token<'a>(
    sv_list: &'a [u8],
    crinex_version: &str,
    numsat: usize,
    index: usize,
) -> Result<&'a str> {
    let start = index * 3;
    let end = start + 3;
    if end > sv_list.len() {
        return Err(Error::Parse(format!(
            "CRINEX {crinex_version} epoch SV list shorter than {numsat} satellites"
        )));
    }
    let token = &sv_list[start..end];
    if !token.is_ascii() {
        return Err(Error::Parse(format!(
            "CRINEX {crinex_version} epoch SV token {} contains non-ASCII bytes",
            index + 1
        )));
    }
    std::str::from_utf8(token).map_err(|_| {
        Error::Parse(format!(
            "CRINEX {crinex_version} epoch SV token {} is not valid UTF-8",
            index + 1
        ))
    })
}

fn map_field_error(error: FieldError, line: &str) -> Error {
    Error::Parse(format!(
        "CRINEX invalid {}: {error} in {line:?}",
        error.field()
    ))
}

fn map_arithmetic_error(error: validate::ArithmeticError) -> Error {
    Error::Parse(format!("CRINEX {error}"))
}

/// Parse a reset token `order&value` (e.g. `3&126298057858`). Returns
/// `Ok(Some((order, value)))` for a reset, `Ok(None)` for a plain delta, and an
/// error for a malformed reset.
fn parse_reset(token: &str) -> Result<Option<(usize, i64)>> {
    let token = token.trim();
    if let Some(amp) = token.find('&') {
        let order = token[..amp]
            .parse::<usize>()
            .map_err(|_| Error::Parse(format!("CRINEX reset order in {token:?} invalid")))?;
        if order == 0 || order > MAX_ORDER {
            return Err(Error::Parse(format!(
                "CRINEX reset order {order} out of range 1..={MAX_ORDER}"
            )));
        }
        let value = token[amp + 1..]
            .parse::<i64>()
            .map_err(|_| Error::Parse(format!("CRINEX reset value in {token:?} invalid")))?;
        Ok(Some((order, value)))
    } else {
        Ok(None)
    }
}

/// Format one reconstructed V3 satellite line: the SV token, then each
/// observation as a 16-column field (`F14.3` value + LLI + SSI), with the flag
/// string supplying the LLI/SSI characters.
fn format_sat_line(sv: &str, values: &[Option<i64>], flags: &str) -> String {
    let mut out = String::with_capacity(3 + values.len() * OBS_FIELD_WIDTH);
    out.push_str(sv);
    let flag_bytes = flags.as_bytes();
    for (i, value) in values.iter().enumerate() {
        match value {
            Some(v) => out.push_str(&format_value(*v)),
            None => {
                for _ in 0..OBS_VALUE_WIDTH {
                    out.push(' ');
                }
            }
        }
        // LLI + SSI from the flag string (2 chars per observation).
        let lli = flag_bytes.get(i * 2).copied().unwrap_or(b' ');
        let ssi = flag_bytes.get(i * 2 + 1).copied().unwrap_or(b' ');
        if value.is_some() {
            out.push(lli as char);
            out.push(ssi as char);
        } else {
            out.push(' ');
            out.push(' ');
        }
    }
    out
}

/// Format a single scaled integer observation as the RINEX `F14.3` text the
/// reference `crx2rnx` emits: the value `value * 1e-3` right-aligned in 14
/// columns. A **negative** value in `(-1, 0)` drops its leading zero (`-0.920`
/// is written `-.920`) - the documented RNXCMP formatting idiosyncrasy; a
/// non-negative sub-one value keeps the zero (`0.216`, `0.000`). Formatting from
/// the scaled integer keeps the three decimals exact (no binary-float rounding
/// of the fractional part).
fn format_value(scaled: i64) -> String {
    let negative = scaled < 0;
    let magnitude = scaled.unsigned_abs();
    let whole = magnitude / 1000;
    let frac = magnitude % 1000;
    let body = if negative && whole == 0 {
        format!("-.{frac:03}")
    } else {
        format!("{}{}.{:03}", if negative { "-" } else { "" }, whole, frac)
    };
    format!("{body:>14}")
}

/// Format the RINEX-2 epoch line(s) from the reconstructed descriptor, SV list,
/// and clock text, wrapping the SV list after 12 satellites.
fn format_epoch_v1(descriptor: &str, sv_list: &[String], clock_text: &str) -> Vec<String> {
    // The fixed epoch header (date + flag + count) is cols 0..32 of the
    // descriptor.
    let head = field(descriptor, 0, 32).to_string();
    let mut lines = Vec::new();
    let mut first = head;
    for sv in sv_list.iter().take(12) {
        first.push_str(sv);
    }
    if !clock_text.is_empty() {
        // The RINEX-2 receiver clock offset sits at columns 68..80 of the first
        // epoch line, regardless of satellite count: pad the (up-to-12) SV slots
        // to column 68 before appending it.
        while first.len() < 68 {
            first.push(' ');
        }
        first.push_str(clock_text);
    }
    lines.push(first);
    let mut idx = 12;
    while idx < sv_list.len() {
        let chunk = sv_list[idx..(idx + 12).min(sv_list.len())].join("");
        lines.push(format!("{:32}{chunk}", ""));
        idx += 12;
    }
    lines
}

/// Format the RINEX-2 observation line(s) for one satellite, wrapping after 5
/// observations per line (RINEX 2 layout: 16-col fields).
fn format_sat_lines_v1(values: &[Option<i64>], flags: &str) -> Vec<String> {
    let flag_bytes = flags.as_bytes();
    let mut lines = Vec::new();
    let mut line = String::new();
    for (i, value) in values.iter().enumerate() {
        if i > 0 && i % 5 == 0 {
            lines.push(std::mem::take(&mut line));
        }
        match value {
            Some(v) => line.push_str(&format_value(*v)),
            None => {
                for _ in 0..OBS_VALUE_WIDTH {
                    line.push(' ');
                }
            }
        }
        let lli = flag_bytes.get(i * 2).copied().unwrap_or(b' ');
        let ssi = flag_bytes.get(i * 2 + 1).copied().unwrap_or(b' ');
        if value.is_some() {
            line.push(lli as char);
            line.push(ssi as char);
        } else {
            line.push(' ');
            line.push(' ');
        }
    }
    lines.push(line);
    lines
}

/// Trim trailing spaces from a reconstructed line (the reference `crx2rnx`
/// strips trailing blanks from every output line).
fn trim_end(line: &str) -> &str {
    line.trim_end_matches(' ')
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
