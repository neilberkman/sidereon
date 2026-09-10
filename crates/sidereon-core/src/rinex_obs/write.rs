//! RINEX 3 observation serialization - the inverse of [`super::RinexObs::parse`].
//!
//! Pure and deterministic: the same product always produces byte-identical text
//! and no I/O is performed. The output is a well-formed RINEX 3 observation file
//! whose header carries every record the parser reconstructs the [`super::ObsHeader`]
//! from, and whose body writes one fixed-column record per satellite, so
//! re-parsing reproduces the same [`super::RinexObs`] (header and epochs).
//!
//! Round-trip scope. The canonical IR is the parsed product. Header records the
//! reader does not retain (free-text `COMMENT`, `MARKER NUMBER`, the unparsed
//! GLONASS code/phase bias table) are not re-emitted; they carry no IR state, so
//! their absence does not change the re-parsed product. Observation values use
//! the `F14.3` width the files carry, and any `SYS / SCALE FACTOR` in force is
//! re-applied before formatting (the inverse of the parser's divide), so a value
//! read from a real file re-encodes to the same `f64`. Event records (epoch flag
//! greater than one) retain only their flag and civil epoch, so they are written
//! with a zero special-record count.

use core::fmt::Write as _;

use crate::id::GnssSystem;

use super::{ObsEpoch, ObsEpochTime, ObsValue, RinexObs, OBS_FIELD_WIDTH, OBS_VALUE_WIDTH};

/// Columns a header record's content occupies before its 20-column label.
pub(super) const HEADER_CONTENT_WIDTH: usize = 60;
/// Width of the `SYS / PHASE SHIFT` correction field (`F8.5`).
const PHASE_SHIFT_CORRECTION_WIDTH: usize = 8;
/// RINEX-3 observation codes per `SYS / # / OBS TYPES` line before continuation.
const OBS_CODES_PER_LINE: usize = 13;
/// RINEX-3 observation codes per `SYS / SCALE FACTOR` line after its 10-column prefix.
const SCALE_FACTOR_CODES_PER_LINE: usize = 12;
/// RINEX-3 `PRN / # OF OBS` count fields per line after its 3-column satellite field.
const PRN_OBS_COUNTS_PER_LINE: usize = 9;
/// GLONASS slot/channel pairs per `GLONASS SLOT / FRQ #` line.
const GLONASS_SLOTS_PER_LINE: usize = 8;

impl RinexObs {
    /// Serialize this product to standard RINEX 3 observation text - the inverse
    /// of [`RinexObs::parse`].
    ///
    /// Pure and deterministic. See this module's documentation for the round-trip
    /// scope: re-parsing the output reproduces the same header and epochs.
    pub fn to_rinex_string(&self) -> String {
        let mut out = String::new();
        self.write_header(&mut out);
        self.write_body(&mut out);
        out
    }

    fn write_header(&self, out: &mut String) {
        let h = &self.header;
        push_header_line(
            out,
            &format!(
                "{:<20}{:<40}",
                format!("{:.2}", h.version),
                "OBSERVATION DATA    M (MIXED)"
            ),
            "RINEX VERSION / TYPE",
        );
        if let Some(pgm) = &h.program_run_by_date {
            push_header_line(
                out,
                &format!("{:<20}{:<20}{:<20}", pgm.program, pgm.run_by, pgm.date),
                "PGM / RUN BY / DATE",
            );
        }
        for comment in &h.comments {
            push_header_line(out, comment, "COMMENT");
        }
        if let Some(name) = &h.marker_name {
            push_header_line(out, &format!("{name:<60}"), "MARKER NAME");
        }
        if let Some(number) = &h.marker_number {
            push_header_line(out, number, "MARKER NUMBER");
        }
        if let Some(marker_type) = &h.marker_type {
            push_header_line(out, marker_type, "MARKER TYPE");
        }
        if h.observer.is_some() || h.agency.is_some() {
            push_header_line(
                out,
                &format!(
                    "{:<20}{:<40}",
                    h.observer.as_deref().unwrap_or(""),
                    h.agency.as_deref().unwrap_or("")
                ),
                "OBSERVER / AGENCY",
            );
        }
        if let Some(receiver) = &h.receiver {
            push_header_line(
                out,
                &format!(
                    "{:<20}{:<20}{:<20}",
                    receiver.number, receiver.receiver_type, receiver.version
                ),
                "REC # / TYPE / VERS",
            );
        }
        if let Some(antenna) = &h.antenna {
            push_header_line(
                out,
                &format!("{:<20}{:<20}", antenna.number, antenna.antenna_type),
                "ANT # / TYPE",
            );
        }
        if let Some(pos) = h.approx_position_m {
            push_header_line(out, &format_vec3(pos), "APPROX POSITION XYZ");
        }
        if let Some(delta) = h.antenna_delta_hen_m {
            push_header_line(out, &format_vec3(delta), "ANTENNA: DELTA H/E/N");
        }
        for (system, codes) in &h.obs_codes {
            write_obs_types(out, *system, codes);
        }
        if let Some(unit) = &h.signal_strength_unit {
            push_header_line(out, unit, "SIGNAL STRENGTH UNIT");
        }
        // A cadence the F10.3 field cannot carry is omitted rather than written
        // into a line that overruns its columns and reads back as a different
        // number. Parsed products never hold one; a caller-built product can.
        if let Some(interval) = h
            .interval_s
            .filter(|interval| crate::rinex_common::writable_obs_interval_s(*interval))
        {
            push_header_line(out, &format!("{interval:10.3}"), "INTERVAL");
        }
        if let Some((epoch, scale)) = h.time_of_first_obs {
            if let Some(label) = crate::rinex_common::time_scale_rinex_label(scale) {
                push_header_line(out, &format_first_obs(epoch, label), "TIME OF FIRST OBS");
            }
        }
        if let Some((epoch, scale)) = h.time_of_last_obs {
            if let Some(label) = crate::rinex_common::time_scale_rinex_label(scale) {
                push_header_line(out, &format_first_obs(epoch, label), "TIME OF LAST OBS");
            }
        }
        for shift in &h.phase_shifts {
            write_phase_shift(out, shift);
        }
        for factor in &h.scale_factors {
            write_scale_factor(out, factor);
        }
        if !h.glonass_slots.is_empty() {
            write_glonass_slots(out, &h.glonass_slots);
        }
        if let Some(entries) = &h.glonass_cod_phs_bis {
            write_glonass_cod_phs_bis(out, entries);
        }
        if let Some(leap) = h.leap_seconds {
            write_leap_seconds(out, leap);
        }
        if let Some(count) = h.n_satellites {
            push_header_line(out, &format!("{count:6}"), "# OF SATELLITES");
        }
        for (sat, counts) in &h.prn_obs_counts {
            write_prn_obs_counts(out, *sat, counts);
        }
        push_header_line(out, "", "END OF HEADER");
    }

    fn write_body(&self, out: &mut String) {
        for epoch in &self.epochs {
            self.write_epoch(out, epoch);
        }
    }

    fn write_epoch(&self, out: &mut String, epoch: &ObsEpoch) {
        let t = epoch.epoch;
        // Event records (flag > 1) keep only their flag and epoch in the IR, so
        // no special records follow; flag 0/1 carry the satellite observations.
        let count = if epoch.flag > 1 { 0 } else { epoch.sats.len() };
        let picoseconds = epoch
            .epoch_picoseconds
            .map(|value| format!(" {value:05}"))
            .unwrap_or_default();
        // RINEX reserves six columns between the satellite count and the clock
        // offset. Without them a full-width negative offset abuts the count and
        // the line no longer reads back.
        let clock = epoch
            .rcv_clock_offset_s
            .map(|value| format!("      {value:15.12}"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "> {:04} {:02} {:02} {:02} {:02}{:11.7}{picoseconds}  {}{:3}{clock}",
            t.year, t.month, t.day, t.hour, t.minute, t.second, epoch.flag, count
        );
        if epoch.flag > 1 {
            return;
        }
        for (sat, values) in &epoch.sats {
            self.write_sat_record(out, *sat, values);
        }
    }

    fn write_sat_record(
        &self,
        out: &mut String,
        sat: crate::id::GnssSatelliteId,
        values: &[ObsValue],
    ) {
        let codes = self.header.obs_codes.get(&sat.system).map(Vec::as_slice);
        let mut line = format!("{:<3}", sat.to_string());
        for (index, value) in values.iter().enumerate() {
            let code = codes.and_then(|c| c.get(index)).map(String::as_str);
            push_obs_value(&mut line, *value, self.scale_for(sat.system, code));
        }
        // Trailing blank observations carry no information; drop them.
        let trimmed = line.trim_end();
        out.push_str(trimmed);
        out.push('\n');
    }

    /// The `SYS / SCALE FACTOR` divisor in force for a system/code, mirroring the
    /// parser's lookup so a value re-multiplies back to its stored ASCII.
    fn scale_for(&self, system: GnssSystem, code: Option<&str>) -> f64 {
        let Some(code) = code else {
            return 1.0;
        };
        self.header
            .scale_factors
            .iter()
            .rev()
            .find(|record| {
                record.system == system
                    && (record.codes.is_empty() || record.codes.iter().any(|c| c == code))
            })
            .map_or(1.0, |record| record.factor)
    }
}

/// Append a header line: content padded into the first 60 columns, then the
/// 20-column record label.
fn push_header_line(out: &mut String, content: &str, label: &str) {
    let content = header_content_60(content);
    let _ = writeln!(out, "{content:<HEADER_CONTENT_WIDTH$}{label}");
}

fn header_content_60(content: &str) -> std::borrow::Cow<'_, str> {
    if content.len() <= HEADER_CONTENT_WIDTH {
        return std::borrow::Cow::Borrowed(content);
    }
    let mut end = HEADER_CONTENT_WIDTH;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    std::borrow::Cow::Owned(content[..end].to_string())
}

/// Format an `F14.4` ECEF / antenna triple into the leading columns.
fn format_vec3(values: [f64; 3]) -> String {
    format!("{:14.4}{:14.4}{:14.4}", values[0], values[1], values[2])
}

/// Format the `TIME OF FIRST OBS` record body (civil epoch then the 3-column
/// time-system label at columns 48-50).
fn format_first_obs(epoch: ObsEpochTime, scale_label: &str) -> String {
    format!(
        "{:6}{:6}{:6}{:6}{:6}{:13.7}{:>8}",
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second, scale_label
    )
}

/// Write the `SYS / # / OBS TYPES` record(s) for one constellation, wrapping the
/// declared codes across continuation lines as the parser expects.
///
/// The 60-column content area holds the `A1,2X,I3` prefix plus `13(1X,A3)` code
/// fields, so a chunk fits exactly when each descriptor is at most `A3` wide and
/// the count fits its `I3` field. [`RinexObs::parse`] rejects a header that
/// breaks either bound, since a wider record would be truncated here and
/// re-parse with fewer codes than its count declares.
fn write_obs_types(out: &mut String, system: GnssSystem, codes: &[String]) {
    let count = codes.len();
    for (chunk_index, chunk) in codes.chunks(OBS_CODES_PER_LINE).enumerate() {
        let mut content = if chunk_index == 0 {
            format!("{}  {:>3}", system.letter(), count)
        } else {
            " ".repeat(6)
        };
        for code in chunk {
            let _ = write!(content, " {code:>3}");
        }
        push_header_line(out, &content, "SYS / # / OBS TYPES");
    }
    // A zero-code system still needs its declaration line.
    if codes.is_empty() {
        push_header_line(
            out,
            &format!("{}  {:>3}", system.letter(), count),
            "SYS / # / OBS TYPES",
        );
    }
}

/// Write one `SYS / PHASE SHIFT` record. The optional satellite list is emitted
/// with its count when present (otherwise the correction applies system-wide).
fn write_phase_shift(out: &mut String, shift: &super::ObsPhaseShift) {
    push_header_line(out, &phase_shift_content(shift), "SYS / PHASE SHIFT");
}

/// The 60-column content of one `SYS / PHASE SHIFT` record.
///
/// [`RinexObs::parse`] measures a record with this before accepting it, so a
/// list it could not re-emit inside the content area is rejected there instead
/// of being truncated here.
pub(super) fn phase_shift_content(shift: &super::ObsPhaseShift) -> String {
    let mut content = format!(
        "{} {} {}",
        shift.system.letter(),
        shift.code,
        fmt_shortest(shift.correction_cycles)
    );
    if !shift.satellites.is_empty() {
        let _ = write!(content, " {}", shift.satellites.len());
        for sat in &shift.satellites {
            let _ = write!(content, " {sat}");
        }
    }
    content
}

/// Write one `SYS / SCALE FACTOR` record (factor at columns 2-5, code count at
/// 8-9, affected codes from column 10), wrapping codes across continuation lines.
fn write_scale_factor(out: &mut String, factor: &super::ObsScaleFactor) {
    let divisor = factor.factor as u32;
    let count = factor.codes.len();
    if factor.codes.is_empty() {
        push_header_line(
            out,
            &format!("{} {:>4}  {:>2}", factor.system.letter(), divisor, count),
            "SYS / SCALE FACTOR",
        );
        return;
    }
    for (chunk_index, chunk) in factor.codes.chunks(SCALE_FACTOR_CODES_PER_LINE).enumerate() {
        let mut content = if chunk_index == 0 {
            format!("{} {:>4}  {:>2}", factor.system.letter(), divisor, count)
        } else {
            " ".repeat(10)
        };
        for code in chunk {
            let _ = write!(content, " {code:>3}");
        }
        push_header_line(out, &content, "SYS / SCALE FACTOR");
    }
}

/// Write the `GLONASS SLOT / FRQ #` table (count at columns 0-2, then `Rnn k`
/// slot/channel pairs from column 4), wrapping across continuation lines.
fn write_glonass_slots(out: &mut String, slots: &std::collections::BTreeMap<u8, i8>) {
    let entries: Vec<(u8, i8)> = slots
        .iter()
        .map(|(&prn, &channel)| (prn, channel))
        .collect();
    let count = entries.len();
    for (chunk_index, chunk) in entries.chunks(GLONASS_SLOTS_PER_LINE).enumerate() {
        let mut content = if chunk_index == 0 {
            format!("{count:3} ")
        } else {
            " ".repeat(4)
        };
        for (prn, channel) in chunk {
            let _ = write!(content, "R{prn:02} {channel:2} ");
        }
        push_header_line(out, content.trim_end(), "GLONASS SLOT / FRQ #");
    }
}

fn write_glonass_cod_phs_bis(out: &mut String, entries: &[(String, f64)]) {
    let mut content = String::new();
    for (code, value) in entries {
        let _ = write!(content, " {code:>3} {value:8.3}");
    }
    push_header_line(out, content.trim_start(), "GLONASS COD/PHS/BIS");
}

fn write_leap_seconds(out: &mut String, leap: super::ObsLeapSeconds) {
    let mut content = format!("{:6}", leap.current);
    if let Some(value) = leap.delta_future {
        let _ = write!(content, "{value:6}");
    }
    if let Some(value) = leap.week {
        let _ = write!(content, "{value:6}");
    }
    if let Some(value) = leap.day {
        let _ = write!(content, "{value:6}");
    }
    push_header_line(out, &content, "LEAP SECONDS");
}

fn write_prn_obs_counts(
    out: &mut String,
    sat: crate::id::GnssSatelliteId,
    counts: &[Option<usize>],
) {
    if counts.is_empty() {
        push_header_line(out, &format!("{sat:<3}"), "PRN / # OF OBS");
        return;
    }
    for (chunk_index, chunk) in counts.chunks(PRN_OBS_COUNTS_PER_LINE).enumerate() {
        let mut content = if chunk_index == 0 {
            format!("{sat:<3}")
        } else {
            " ".repeat(3)
        };
        for count in chunk {
            match count {
                Some(value) => {
                    let _ = write!(content, "{value:6}");
                }
                None => content.push_str("      "),
            }
        }
        push_header_line(out, &content, "PRN / # OF OBS");
    }
}

/// Append one 16-column observation field: the `F14.3` value (or blanks) re-scaled
/// by `scale`, then the loss-of-lock and signal-strength indicator digits.
fn push_obs_value(line: &mut String, value: ObsValue, scale: f64) {
    match value.value {
        Some(v) => {
            let _ = write!(line, "{:width$.3}", v * scale, width = OBS_VALUE_WIDTH);
        }
        None => line.push_str(&" ".repeat(OBS_VALUE_WIDTH)),
    }
    push_indicator(line, value.lli);
    push_indicator(line, value.ssi);
}

fn push_indicator(line: &mut String, indicator: Option<u8>) {
    match indicator {
        Some(digit) => {
            let _ = write!(line, "{digit}");
        }
        None => line.push(' '),
    }
}

/// Shortest spelling that round-trips back to the same `f64`, keeping the plain
/// decimal whenever it fits the record's `F8.5` field.
///
/// Rust's `Display` never switches to an exponent, so a correction far from
/// unity renders as hundreds of digits - wider than the content area, where it
/// would be truncated into a different value and take the satellite list with
/// it. Exponent form is the shortest round-tripping spelling at those
/// magnitudes and the parser reads it back exactly. A conforming correction
/// fits the `F8.5` field and is unaffected.
fn fmt_shortest(value: f64) -> String {
    let plain = format!("{value}");
    if plain.len() <= PHASE_SHIFT_CORRECTION_WIDTH {
        return plain;
    }
    let exponent = format!("{value:e}");
    if exponent.len() < plain.len() {
        exponent
    } else {
        plain
    }
}

const _: () = assert!(OBS_FIELD_WIDTH == OBS_VALUE_WIDTH + 2);
