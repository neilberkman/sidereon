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
use std::collections::BTreeMap;

use crate::id::GnssSystem;

use super::{ObsEpoch, ObsEpochTime, ObsValue, RinexObs, OBS_FIELD_WIDTH, OBS_VALUE_WIDTH};

/// Columns a header record's content occupies before its 20-column label.
pub(super) const HEADER_CONTENT_WIDTH: usize = 60;
/// Columns one satellite id occupies in a version 2 epoch line, `A1,I2`.
const RINEX2_SATELLITE_FIELD_WIDTH: usize = 3;
/// Satellite ids a version 2 epoch line carries before continuing, `12(A1,I2)`.
const RINEX2_EPOCH_SATELLITES_PER_LINE: usize = 12;
/// Observation values a version 2 record carries before continuing, `5(F14.3,I1,I1)`.
const RINEX2_OBS_VALUES_PER_LINE: usize = 5;
/// Observation codes a `# / TYPES OF OBSERV` record carries, `9(4X,A2)`.
const RINEX2_OBS_TYPES_PER_LINE: usize = 9;
/// `GLONASS COD/PHS/BIS` entries that fit the sixty-column body: each takes
/// thirteen columns, `1X,A3,1X,F8.3`.
pub(super) const GLONASS_BIAS_ENTRIES_PER_LINE: usize = 4;
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

/// How a version 2 file's one observation-code list lines up with each
/// constellation's own list. Built by [`RinexObs::rinex2_obs_layout`].
struct Rinex2ObsLayout {
    /// The names the `# / TYPES OF OBSERV` record carries, in order.
    names: Vec<String>,
    /// For each constellation, the index into its own value list that each name
    /// draws from, or `None` where that constellation has nothing to put there.
    slots: BTreeMap<GnssSystem, Vec<Option<usize>>>,
}

/// The constellations among `holders` that read `name` back as the code they
/// hold, so one column can carry it for all of them.
fn served_by(name: &str, holders: &[(GnssSystem, &String)]) -> Vec<GnssSystem> {
    holders
        .iter()
        .filter(|(system, canonical)| {
            super::canonical_rinex2_obs_code(*system, name) == **canonical
        })
        .map(|(system, _)| *system)
        .collect()
}

impl RinexObs {
    /// Serialize this product to standard RINEX observation text - the inverse
    /// of [`RinexObs::parse`].
    ///
    /// The version the header carries decides the records written: below 3.0
    /// the file is version 2 throughout, otherwise version 3.
    ///
    /// Pure and deterministic. See this module's documentation for the round-trip
    /// scope: re-parsing the output reproduces the same header and epochs.
    ///
    /// A version 2 file cannot say everything a product can hold, and what it
    /// cannot say is dropped rather than written where no reader looks:
    /// `MARKER TYPE`, `SIGNAL STRENGTH UNIT`, `SYS / PHASE SHIFT`,
    /// `SYS / SCALE FACTOR` and `GLONASS COD/PHS/BIS`, the future count, week
    /// and day of `LEAP SECONDS`, an observation code's tracking attribute, and
    /// the picoseconds of an epoch. Its year field holds two digits, so an
    /// epoch outside 1980 to 2079 reads back in that window.
    pub fn to_rinex_string(&self) -> String {
        let mut out = String::new();
        let rinex2_layout = self.is_rinex2().then(|| self.rinex2_obs_layout());
        self.write_header(&mut out, rinex2_layout.as_ref());
        self.write_body(&mut out, rinex2_layout.as_ref());
        out
    }

    fn write_header(&self, out: &mut String, rinex2_layout: Option<&Rinex2ObsLayout>) {
        let h = &self.header;
        if self.is_rinex2() {
            // A version 2 file names its constellation in the version record and
            // lists one set of observation codes for all of them.
            push_header_line(
                out,
                &format!(
                    "{:9.2}{:11}{:<20}{:<20}",
                    h.version,
                    "",
                    "OBSERVATION DATA",
                    self.rinex2_system_field()
                ),
                "RINEX VERSION / TYPE",
            );
        } else {
            push_header_line(
                out,
                &format!(
                    "{:<20}{:<40}",
                    format!("{:.2}", h.version),
                    "OBSERVATION DATA    M (MIXED)"
                ),
                "RINEX VERSION / TYPE",
            );
        }
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
        // `MARKER TYPE` arrived with version 3.
        if let Some(marker_type) = h.marker_type.as_ref().filter(|_| !self.is_rinex2()) {
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
        if let Some(layout) = rinex2_layout {
            write_obs_types_v2(out, &layout.names);
        } else {
            for (system, codes) in &h.obs_codes {
                write_obs_types(out, *system, codes);
            }
        }
        // `SIGNAL STRENGTH UNIT` is a version 3 record, as are the four below.
        // Writing one into a version 2 file would make it a file neither
        // version accepts, which is the very thing this writer exists to stop.
        // A product parsed from version 2 never carries them; one built by a
        // caller can, and loses them here.
        if let Some(unit) = h
            .signal_strength_unit
            .as_ref()
            .filter(|_| !self.is_rinex2())
        {
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
        if !self.is_rinex2() {
            for shift in &h.phase_shifts {
                write_phase_shift(out, shift);
            }
            for factor in &h.scale_factors {
                write_scale_factor(out, factor);
            }
        }
        // `GLONASS SLOT / FRQ #` arrived with version 3 too, but it carries the
        // frequency channel of every slot, which nothing else in the file says.
        // It stays, as an extension a version 2 reader skips like any label it
        // does not know, rather than being dropped and taking the table with it.
        if !h.glonass_slots.is_empty() {
            write_glonass_slots(out, &h.glonass_slots);
        }
        if let Some(entries) = h.glonass_cod_phs_bis.as_ref().filter(|_| !self.is_rinex2()) {
            write_glonass_cod_phs_bis(out, entries);
        }
        if let Some(leap) = h.leap_seconds {
            // Version 2 defines one field here. The future count, week and day
            // arrived with version 3, and a version 2 reader takes the columns
            // they occupy as blank comment space at best.
            write_leap_seconds(out, leap, self.is_rinex2());
        }
        if let Some(count) = h.n_satellites {
            push_header_line(out, &format!("{count:6}"), "# OF SATELLITES");
        }
        for (sat, counts) in &h.prn_obs_counts {
            write_prn_obs_counts(out, *sat, counts);
        }
        push_header_line(out, "", "END OF HEADER");
    }

    fn write_body(&self, out: &mut String, rinex2_layout: Option<&Rinex2ObsLayout>) {
        if let Some(layout) = rinex2_layout {
            for epoch in &self.epochs {
                self.write_epoch_v2(out, epoch, layout);
            }
            return;
        }
        for epoch in &self.epochs {
            self.write_epoch(out, epoch);
        }
    }

    /// Whether this product is written in the version 2 record layout.
    ///
    /// A product carries the version its file declared, and is written back in
    /// that version's records. Writing version 3 records under a version 2
    /// header would produce a file that is neither.
    fn is_rinex2(&self) -> bool {
        self.header.version < 3.0
    }

    /// The constellation a version 2 header names: one letter, or `M` for a file
    /// carrying more than one. The name beside it is the convention these files
    /// follow, and sits in the columns the record leaves free.
    fn rinex2_system_field(&self) -> String {
        let mut systems = self.header.obs_codes.keys();
        match (systems.next(), systems.next()) {
            (Some(only), None) => only.letter().to_string(),
            _ => "M (MIXED)".to_string(),
        }
    }

    /// How a version 2 file's one observation-code list lines up with each
    /// constellation's own list.
    ///
    /// Version 2 names its codes once for the whole file, so position `i` means
    /// the same signal for every constellation. A file read at version 2 gave
    /// every constellation its list from that one record, so one name per
    /// position serves them all and the layout is the identity.
    ///
    /// A product a caller assembled can hold codes at one position that no
    /// single version 2 name spells for every constellation holding it - GPS
    /// `C1W` beside GLONASS `C1C`, which are `P1` and `C1`. Rather than write
    /// one of them and let the others read back as a different signal, the
    /// position is split: each name gets its own column, and a constellation
    /// that name does not serve leaves that column blank. Nothing is renamed,
    /// and the file grows by the columns the conflict needs.
    fn rinex2_obs_layout(&self) -> Rinex2ObsLayout {
        let longest = self
            .header
            .obs_codes
            .values()
            .map(Vec::len)
            .max()
            .unwrap_or_default();
        let mut names: Vec<String> = Vec::new();
        let mut slots: BTreeMap<GnssSystem, Vec<Option<usize>>> = self
            .header
            .obs_codes
            .keys()
            .map(|system| (*system, Vec::new()))
            .collect();
        for index in 0..longest {
            let mut remaining: Vec<(GnssSystem, &String)> = self
                .header
                .obs_codes
                .iter()
                .filter_map(|(system, codes)| Some((*system, codes.get(index)?)))
                .collect();
            while let Some(&(first_system, first_code)) = remaining.first() {
                let candidates = super::rinex2_obs_code_candidates(first_system, first_code);
                // The name that serves the most of what is left, preferring the
                // earlier candidate on a tie, since that is the one that keeps
                // the band the signal was measured on.
                let mut best: Option<(&String, usize)> = None;
                for name in &candidates {
                    let served = served_by(name, &remaining).len();
                    if best.is_none_or(|(_, most)| served > most) {
                        best = Some((name, served));
                    }
                }
                let name = match best {
                    Some((name, _)) => name.clone(),
                    // A code of some other shape than RINEX writes is held to
                    // two characters, so the columns after it stay aligned.
                    None => first_code.chars().take(2).collect(),
                };
                let mut served = served_by(&name, &remaining);
                if served.is_empty() {
                    // No version 2 name spells this code. The best one still
                    // stands for the constellation it was built from; every
                    // other constellation here gets a column of its own.
                    served.push(first_system);
                }
                let position = names.len();
                names.push(name);
                for (system, row) in &mut slots {
                    row.resize(position + 1, None);
                    if served.contains(system) {
                        row[position] = Some(index);
                    }
                }
                remaining.retain(|(system, _)| !served.contains(system));
            }
        }
        for row in slots.values_mut() {
            row.resize(names.len(), None);
        }
        Rinex2ObsLayout { names, slots }
    }

    /// Write one version 2 epoch: the record, its satellite list continued
    /// twelve to a line, then each satellite's observations five to a line.
    fn write_epoch_v2(&self, out: &mut String, epoch: &ObsEpoch, layout: &Rinex2ObsLayout) {
        let t = epoch.epoch;
        // An event record keeps only its flag and epoch here, so it names no
        // satellites and no observation records follow it.
        let event = epoch.flag > 1;
        // `12(A1,I2)`: the constellation letter then the number, space padded,
        // which is what a version 2 reader expects.
        let satellites: Vec<String> = if event {
            Vec::new()
        } else {
            epoch
                .sats
                .keys()
                .map(|sat| format!("{}{:2}", sat.system.letter(), sat.prn))
                .collect()
        };
        let mut chunks = satellites.chunks(RINEX2_EPOCH_SATELLITES_PER_LINE);
        let first: String = chunks.next().unwrap_or_default().concat();
        // The clock offset is an `F12.9` field at columns 69 to 80, so the
        // satellite field is held to all twelve of its slots even when fewer
        // satellites fill it. Letting the clock follow the last satellite put
        // it wherever the count happened to end, where no reader looks.
        let tail = match epoch.rcv_clock_offset_s {
            Some(value) => format!(
                "{first:<width$}{value:12.9}",
                width = RINEX2_EPOCH_SATELLITES_PER_LINE * RINEX2_SATELLITE_FIELD_WIDTH
            ),
            None => first,
        };
        let _ = writeln!(
            out,
            "{}{:>3}{tail}",
            format_args!(
                " {:02} {:2} {:2} {:2} {:2}{:11.7}  {}",
                t.year.rem_euclid(100),
                t.month,
                t.day,
                t.hour,
                t.minute,
                t.second,
                epoch.flag
            ),
            satellites.len()
        );
        for chunk in chunks {
            let _ = writeln!(out, "{:32}{}", "", chunk.concat());
        }
        if !event {
            // A constellation the header does not name keeps its own order,
            // which is all there is to go on.
            let straight: Vec<Option<usize>> = (0..layout.names.len()).map(Some).collect();
            for (sat, values) in &epoch.sats {
                let row = layout.slots.get(&sat.system).unwrap_or(&straight);
                write_sat_record_v2(out, values, row);
            }
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

/// Write one satellite's observations in the version 2 layout: five to a line,
/// with no satellite id, since the epoch's list names them in order.
///
/// The record runs to the number of names the header carries, blank where this
/// satellite has nothing for one, because a reader takes that many lines for
/// every satellite. A shorter record would put the next satellite's values
/// under this one.
///
/// The values are written as they stand. A version 2 file carries no
/// `SYS / SCALE FACTOR` record, so scaling them by a factor the file has no way
/// to declare would change what a reader gets back.
fn write_sat_record_v2(out: &mut String, values: &[ObsValue], row: &[Option<usize>]) {
    const BLANK: ObsValue = ObsValue {
        value: None,
        lli: None,
        ssi: None,
    };
    for chunk in row.chunks(RINEX2_OBS_VALUES_PER_LINE) {
        let mut line = String::new();
        for slot in chunk {
            let value = slot
                .and_then(|index| values.get(index))
                .copied()
                .unwrap_or(BLANK);
            push_obs_value(&mut line, value, 1.0);
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
}

/// Write the version 2 `# / TYPES OF OBSERV` record, continued when the codes
/// do not fit one line. The count is written once, on the first line only.
fn write_obs_types_v2(out: &mut String, codes: &[String]) {
    let mut chunks = codes.chunks(RINEX2_OBS_TYPES_PER_LINE);
    let first = chunks.next().unwrap_or_default();
    let mut content = format!("{:6}", codes.len());
    for code in first {
        let _ = write!(content, "    {code:>2}");
    }
    push_header_line(out, &content, "# / TYPES OF OBSERV");
    for chunk in chunks {
        let mut content = " ".repeat(6);
        for code in chunk {
            let _ = write!(content, "    {code:>2}");
        }
        push_header_line(out, &content, "# / TYPES OF OBSERV");
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
    if entries.is_empty() {
        push_header_line(out, "", "GLONASS COD/PHS/BIS");
        return;
    }
    // Each entry takes thirteen of the sixty columns, so a fifth one would be
    // cut off the end of the line and lost. The record continues on another line
    // instead, which is how the reader takes it back.
    for chunk in entries.chunks(GLONASS_BIAS_ENTRIES_PER_LINE) {
        let mut content = String::new();
        for (code, value) in chunk {
            let _ = write!(content, " {code:>3} {value:8.3}");
        }
        push_header_line(out, content.trim_start(), "GLONASS COD/PHS/BIS");
    }
}

fn write_leap_seconds(out: &mut String, leap: super::ObsLeapSeconds, current_only: bool) {
    let mut content = format!("{:6}", leap.current);
    if !current_only {
        if let Some(value) = leap.delta_future {
            let _ = write!(content, "{value:6}");
        }
        if let Some(value) = leap.week {
            let _ = write!(content, "{value:6}");
        }
        if let Some(value) = leap.day {
            let _ = write!(content, "{value:6}");
        }
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
