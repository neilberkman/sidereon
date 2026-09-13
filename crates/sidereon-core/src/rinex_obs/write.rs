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
//! read from a real file re-encodes to the same `f64`. An event record (epoch
//! flag greater than one) keeps the records that followed it as they were
//! written, and they are written back unchanged under their own count.

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
/// Column a `PRN / # OF OBS` record's satellite id begins at, after its `3X`.
pub(super) const PRN_OBS_SATELLITE_COLUMN: usize = 3;
/// Column a `PRN / # OF OBS` record's counts begin at, `A1,I2` past the blanks.
pub(super) const PRN_OBS_COUNTS_COLUMN: usize = 6;
/// Width of one `PRN / # OF OBS` count field (`I6`).
pub(super) const PRN_OBS_COUNT_WIDTH: usize = 6;
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

/// Why a product could not be written as RINEX text that reads back as the
/// product itself.
///
/// The observation writer never changes what a product says to make it fit a
/// file.
#[derive(Debug, Clone, PartialEq)]
pub enum RinexObsWriteError {
    /// A version 2 product whose constellations' code lists are not what one
    /// list of version 2 names reads as. Version 2 names its codes once for
    /// every constellation, and its reader gives each of them a code for every
    /// name, so each list has to be exactly that reading.
    CodeListsNotVersionTwo {
        /// The constellation whose list no name fits.
        system: GnssSystem,
        /// Zero-based position in that list, or its length when the lists
        /// differ in length.
        position: usize,
        /// The code held there, or `None` when the lists differ in length.
        code: Option<String>,
    },
    /// A version 2 product carrying `SYS / SCALE FACTOR` records. This reader
    /// applies them, but version 2 readers that do not know the record, RTKLIB
    /// among them, read the scaled numbers as physical ones, so the text would
    /// say something else to them.
    ScaleFactorsInVersionTwo {
        /// How many records the product holds.
        count: usize,
    },
    /// A satellite holding more values than its constellation has codes. The
    /// values past the codes name no observable, so no file can say what they
    /// are.
    ValuesWithoutCodes {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The satellite.
        satellite: crate::id::GnssSatelliteId,
        /// How many codes its constellation has.
        codes: usize,
        /// How many values it holds.
        values: usize,
    },
    /// A `PRN / # OF OBS` record holding more counts than its constellation
    /// has codes.
    CountsWithoutCodes {
        /// The satellite.
        satellite: crate::id::GnssSatelliteId,
        /// How many codes its constellation has.
        codes: usize,
        /// How many counts it holds.
        counts: usize,
    },
    /// A version 2 product holding a code list for a constellation that no
    /// observation or `PRN / # OF OBS` count names, that the version record of
    /// a file with no observations does not name either, and that the type
    /// names do not read as. A version 2 reader builds no list for it and
    /// nothing in the file says it, so the list would be lost.
    CodeListNotStated {
        /// The constellation.
        system: GnssSystem,
    },
    /// An epoch flag no one-digit flag field holds. Both versions write the
    /// flag in an `I1` field; a flag of 10 or more would be written across the
    /// satellite count beside it.
    EpochFlagTooWide {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The flag.
        flag: u8,
    },
    /// Epoch picoseconds in a product below version 4.02, which added them as
    /// five digits after the receiver clock offset. Earlier epoch records have
    /// no field for them.
    EpochPicosecondsNotInVersion {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The product's version.
        version: f64,
    },
    /// The written text would read back as a different product.
    ReadBackMismatch {
        /// The first field that would change, with its value before and after.
        what: String,
    },
}

impl core::fmt::Display for RinexObsWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CodeListsNotVersionTwo {
                system,
                position,
                code: Some(code),
            } => write!(
                f,
                "RINEX OBS version 2 has no observation type that every constellation reads back \
                 as its own code at position {position}, where {system} holds {code:?}"
            ),
            Self::CodeListsNotVersionTwo {
                system,
                position,
                code: None,
            } => write!(
                f,
                "RINEX OBS version 2 names one list of codes for every constellation, and {system} \
                 holds {position} codes where another constellation holds a different number"
            ),
            Self::ScaleFactorsInVersionTwo { count } => write!(
                f,
                "RINEX OBS version 2 would carry {count} SYS / SCALE FACTOR records, which version 2 \
                 readers that do not apply them read as physical values"
            ),
            Self::ValuesWithoutCodes {
                epoch_index,
                satellite,
                codes,
                values,
            } => write!(
                f,
                "RINEX OBS epoch {epoch_index} satellite {satellite} holds {values} values for \
                 {codes} observation codes"
            ),
            Self::CountsWithoutCodes {
                satellite,
                codes,
                counts,
            } => write!(
                f,
                "RINEX OBS PRN / # OF OBS for {satellite} holds {counts} counts for {codes} \
                 observation codes"
            ),
            Self::CodeListNotStated { system } => write!(
                f,
                "RINEX OBS version 2 would not state {system}'s code list: no observation or \
                 PRN / # OF OBS count names {system}, so a reader builds no list for it, and \
                 the type names do not read as it"
            ),
            Self::EpochFlagTooWide { epoch_index, flag } => write!(
                f,
                "RINEX OBS epoch {epoch_index} flag {flag} does not fit the one-digit flag field"
            ),
            Self::EpochPicosecondsNotInVersion {
                epoch_index,
                version,
            } => write!(
                f,
                "RINEX OBS epoch {epoch_index} carries picoseconds, which a version {version} epoch \
                 record has no field for"
            ),
            Self::ReadBackMismatch { what } => {
                write!(
                    f,
                    "RINEX OBS text would not read back as the product: {what}"
                )
            }
        }
    }
}

impl std::error::Error for RinexObsWriteError {}

impl From<RinexObsWriteError> for crate::Error {
    fn from(error: RinexObsWriteError) -> Self {
        crate::Error::InvalidInput(error.to_string())
    }
}

impl RinexObs {
    /// Serialize this product to standard RINEX observation text - the inverse
    /// of [`RinexObs::parse`].
    ///
    /// The version the header carries decides the records written: below 3.0
    /// the file is version 2 throughout, otherwise version 3.
    ///
    /// Pure and deterministic. The text is returned only when reading it back
    /// gives this product: every header record, code, value, indicator and
    /// event record, compared field by field. Diagnostics describing the text a
    /// product was parsed from - labels read and not retained, records skipped,
    /// and the counts an epoch line declared - are not part of what is written.
    ///
    /// Nothing is dropped, rounded, wrapped or truncated to make a product fit.
    /// A value its column cannot hold exactly, text wider than its record, a
    /// time scale a record cannot name, a version 2 product whose
    /// constellations' code lists are not what one list of version 2 names
    /// reads as, or one holding a list its file would not state, is refused
    /// with the first field that would change.
    ///
    /// # Errors
    ///
    /// [`RinexObsWriteError`] names what the text could not carry.
    pub fn to_rinex_string(&self) -> Result<String, RinexObsWriteError> {
        if let Some((epoch_index, epoch)) = self
            .epochs
            .iter()
            .enumerate()
            .find(|(_, epoch)| epoch.flag > 9)
        {
            return Err(RinexObsWriteError::EpochFlagTooWide {
                epoch_index,
                flag: epoch.flag,
            });
        }
        if self.header.version < 4.02 {
            if let Some(epoch_index) = self
                .epochs
                .iter()
                .position(|epoch| epoch.epoch_picoseconds.is_some())
            {
                return Err(RinexObsWriteError::EpochPicosecondsNotInVersion {
                    epoch_index,
                    version: self.header.version,
                });
            }
        }
        if self.is_rinex2() && !self.header.scale_factors.is_empty() {
            return Err(RinexObsWriteError::ScaleFactorsInVersionTwo {
                count: self.header.scale_factors.len(),
            });
        }
        let rinex2_names = if self.is_rinex2() {
            let names = self.rinex2_names()?;
            if let Some(system) = self.rinex2_unstated_systems(&names).into_iter().next() {
                return Err(RinexObsWriteError::CodeListNotStated { system });
            }
            Some(names)
        } else {
            None
        };
        let mut out = String::new();
        self.write_header(&mut out, rinex2_names.as_deref());
        self.write_body(&mut out, rinex2_names.as_deref());
        self.check_reads_back(&out, rinex2_names.as_deref())?;
        Ok(out)
    }

    fn write_header(&self, out: &mut String, rinex2_names: Option<&[String]>) {
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
        if let Some(names) = rinex2_names {
            write_obs_types_v2(out, names);
        } else {
            for (system, codes) in &h.obs_codes {
                write_obs_types(out, *system, codes);
            }
        }
        if let Some(unit) = &h.signal_strength_unit {
            push_header_line(out, unit, "SIGNAL STRENGTH UNIT");
        }
        if let Some(interval) = h.interval_s {
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
            write_leap_seconds(out, leap, false);
        }
        if let Some(count) = h.n_satellites {
            push_header_line(out, &format!("{count:6}"), "# OF SATELLITES");
        }
        for (sat, counts) in &h.prn_obs_counts {
            write_prn_obs_counts(out, *sat, counts);
        }
        push_header_line(out, "", "END OF HEADER");
    }

    fn write_body(&self, out: &mut String, rinex2_names: Option<&[String]>) {
        if let Some(names) = rinex2_names {
            for epoch in &self.epochs {
                self.write_epoch_v2(out, epoch, names.len());
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
    /// The constellation a version 2 file's version record names: the one the
    /// product holds, while its observations are all from it, and `M (MIXED)`
    /// for a mixed product or one whose observations say otherwise. With no
    /// observations and no constellation held, the one whose list the file
    /// then states, which a reader takes from this field: `M (MIXED)` when that
    /// is GPS.
    fn rinex2_system_field(&self) -> String {
        let observed = self.rinex2_observed_systems();
        match self.header.rinex2_system {
            Some(system) if observed.iter().all(|seen| *seen == system) => {
                system.letter().to_string()
            }
            Some(_) => "M (MIXED)".to_string(),
            None if !observed.is_empty() => "M (MIXED)".to_string(),
            None => match self.rinex2_fallback_system() {
                GnssSystem::Gps => "M (MIXED)".to_string(),
                other => other.letter().to_string(),
            },
        }
    }

    /// Constellations this product's observations are from.
    fn rinex2_observed_systems(&self) -> std::collections::BTreeSet<GnssSystem> {
        self.epochs
            .iter()
            .flat_map(|epoch| epoch.sats.keys().map(|sat| sat.system))
            .collect()
    }

    /// The constellation a version 2 file with no observations states a list
    /// for: the one the product's version record names, and without one the
    /// product's one list's, or GPS when it holds none or several.
    fn rinex2_fallback_system(&self) -> GnssSystem {
        if let Some(system) = self.header.rinex2_system {
            return system;
        }
        let mut systems = self.header.obs_codes.keys();
        match (systems.next(), systems.next()) {
            (Some(only), None) => *only,
            _ => GnssSystem::Gps,
        }
    }

    /// Write one version 2 epoch: the record, its satellite list continued
    /// twelve to a line, then each satellite's observations five to a line.
    fn write_epoch_v2(&self, out: &mut String, epoch: &ObsEpoch, width: usize) {
        let t = epoch.epoch;
        // An event names no satellites. Its own records follow the epoch line,
        // and its declared count is how many of them there are.
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
            if event {
                epoch.special_records.len()
            } else {
                satellites.len()
            }
        );
        for chunk in chunks {
            let _ = writeln!(out, "{:32}{}", "", chunk.concat());
        }
        for record in &epoch.special_records {
            let _ = writeln!(out, "{record}");
        }
        if !event {
            // Every satellite's record runs to the number of names the header
            // carries, and its values are in the order its constellation's list
            // names them, which is the header's own order.
            for values in epoch.sats.values() {
                write_sat_record_v2(out, values, width);
            }
        }
    }

    fn write_epoch(&self, out: &mut String, epoch: &ObsEpoch) {
        let t = epoch.epoch;
        // An event (flag > 1) declares how many of its own records follow;
        // flag 0 and 1 declare their satellites and carry observations.
        // An event declares the records that follow it.
        let count = if epoch.flag > 1 {
            epoch.special_records.len()
        } else {
            epoch.sats.len()
        };
        // RINEX reserves six columns between the satellite count and the clock
        // offset. Without them a full-width negative offset abuts the count and
        // the line no longer reads back. Version 4.02 appends five more digits
        // of the second after the clock, `1X,I5.5`, with the clock's columns
        // left blank when there is no offset.
        let picoseconds = epoch
            .epoch_picoseconds
            .map(|value| format!(" {value:05}"))
            .unwrap_or_default();
        let clock = match epoch.rcv_clock_offset_s {
            Some(value) => format!("      {value:15.12}"),
            None if !picoseconds.is_empty() => " ".repeat(21),
            None => String::new(),
        };
        let _ = writeln!(
            out,
            "> {:04} {:02} {:02} {:02} {:02}{:11.7}  {}{:3}{clock}{picoseconds}",
            t.year, t.month, t.day, t.hour, t.minute, t.second, epoch.flag, count
        );
        if epoch.flag > 1 {
            for record in &epoch.special_records {
                let _ = writeln!(out, "{record}");
            }
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
/// satellite holds no value, because a reader takes that many lines for every
/// satellite. Values are written as they are stored: the writer refuses a
/// version 2 product with scale factors, since other version 2 readers would not
/// apply them.
fn write_sat_record_v2(out: &mut String, values: &[ObsValue], width: usize) {
    const BLANK: ObsValue = ObsValue {
        value: None,
        lli: None,
        ssi: None,
    };
    for start in (0..width).step_by(RINEX2_OBS_VALUES_PER_LINE) {
        let end = (start + RINEX2_OBS_VALUES_PER_LINE).min(width);
        let mut line = String::new();
        for column in start..end {
            push_obs_value(&mut line, values.get(column).copied().unwrap_or(BLANK), 1.0);
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
        // Each field has its own six columns, so a blank one before a value is
        // written blank; leaving it out would put the next value in its place.
        let fields = [leap.delta_future, leap.week, leap.day];
        let written = fields
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |last| last + 1);
        for field in &fields[..written] {
            match field {
                Some(value) => {
                    let _ = write!(content, "{value:6}");
                }
                None => content.push_str("      "),
            }
        }
    }
    push_header_line(out, &content, "LEAP SECONDS");
}

fn write_prn_obs_counts(
    out: &mut String,
    sat: crate::id::GnssSatelliteId,
    counts: &[Option<usize>],
) {
    // `3X,A1,I2,9I6`, continued as `6X,9I6`: the satellite id sits at columns 4
    // to 6, not at the start of the line, and the counts follow from column 7.
    let heading = format!("{:blanks$}{sat:<3}", "", blanks = PRN_OBS_SATELLITE_COLUMN);
    if counts.is_empty() {
        push_header_line(out, &heading, "PRN / # OF OBS");
        return;
    }
    for (chunk_index, chunk) in counts.chunks(PRN_OBS_COUNTS_PER_LINE).enumerate() {
        let mut content = if chunk_index == 0 {
            heading.clone()
        } else {
            " ".repeat(PRN_OBS_COUNTS_COLUMN)
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

impl RinexObs {
    /// The single list of observation types a version 2 header carries for this
    /// product.
    ///
    /// Version 2 names its codes once, and its reader gives every constellation
    /// a code for every name, so a product is a version 2 file only when every
    /// constellation's list is exactly what one sequence of names reads as. The
    /// sequence is found a position at a time: a name fits when every
    /// constellation reads it as the code it holds there. What a constellation
    /// reads at a position depends only on the codes it holds before it, and
    /// those are the product's own, so no name chosen earlier changes what fits
    /// later. Taking a fitting name at each position therefore finds a list
    /// whenever one exists, and a position where none fits proves there is none.
    ///
    /// A list the file does not state is lost unless the names read as it, so
    /// the names are also found for the stated lists together with such lists:
    /// the largest set of them some names read as too, larger sets tried
    /// first. With none, the stated lists alone decide the names.
    fn rinex2_names(&self) -> Result<Vec<String>, RinexObsWriteError> {
        let stated = self.rinex2_stated_lists()?;
        let unused: Vec<(GnssSystem, &Vec<String>)> = self
            .header
            .obs_codes
            .iter()
            .filter(|(system, _)| !stated.contains_key(system))
            .map(|(system, codes)| (*system, codes))
            .collect();
        let mut sets: Vec<u32> = (1..1u32 << unused.len()).collect();
        sets.sort_by_key(|set| std::cmp::Reverse(set.count_ones()));
        for set in sets {
            let mut lists = stated.clone();
            for (index, (system, codes)) in unused.iter().enumerate() {
                if set & (1 << index) != 0 {
                    lists.insert(*system, (*codes).clone());
                }
            }
            if let Ok(names) = self.rinex2_names_for(&lists) {
                return Ok(names);
            }
        }
        self.rinex2_names_for(&stated)
    }

    /// The names that read as exactly `lists`, found a position at a time as
    /// `rinex2_names` describes.
    fn rinex2_names_for(
        &self,
        lists: &BTreeMap<GnssSystem, Vec<String>>,
    ) -> Result<Vec<String>, RinexObsWriteError> {
        use std::collections::HashSet;

        let version = self.header.version;
        // With no list to state, the names the product was read with are the
        // file's names, and a header-only file reads its fallback list by them.
        let Some(width) = lists.values().map(Vec::len).max() else {
            return Ok(self.header.rinex2_types.clone());
        };
        if let Some((system, codes)) = lists.iter().find(|(_, codes)| codes.len() != width) {
            return Err(RinexObsWriteError::CodeListsNotVersionTwo {
                system: *system,
                position: codes.len(),
                code: None,
            });
        }
        // The names the product was read with state every list when each is
        // what they read as, and then they are the file's names.
        let types = &self.header.rinex2_types;
        if !types.is_empty()
            && lists.iter().all(|(system, codes)| {
                *codes == super::rinex2_system_obs_codes(*system, types, version)
            })
        {
            return Ok(types.clone());
        }
        // What `system` reads `name` as, given the codes it holds before the
        // position: the same rule the reader applies.
        let reads_as = |system: GnssSystem, before: &HashSet<&str>, name: &str| {
            let canonical = super::canonical_rinex2_obs_code(system, name, version);
            if super::rinex2_name_allowed(system, name, version)
                && !before.contains(canonical.as_str())
            {
                canonical
            } else {
                name.to_string()
            }
        };
        // Every name a constellation reads back as the code it holds: the code
        // itself when it is a name kept as written, otherwise the names that
        // map to it.
        let fitting = |system: GnssSystem, code: &String, before: &HashSet<&str>| -> Vec<String> {
            let names = if super::rinex2_kept_as_written(code) {
                vec![code.clone()]
            } else {
                super::rinex2_obs_code_candidates(system, code, version)
            };
            names
                .into_iter()
                .filter(|name| reads_as(system, before, name) == *code)
                .collect()
        };
        // Each constellation's codes before the position, added to as the
        // positions pass rather than scanned again at each.
        let mut before: BTreeMap<GnssSystem, HashSet<&str>> = lists
            .keys()
            .map(|system| (*system, HashSet::new()))
            .collect();
        let mut names = Vec::with_capacity(width);
        for position in 0..width {
            let mut shared: Option<Vec<String>> = None;
            for (system, codes) in lists {
                let own = fitting(*system, &codes[position], &before[system]);
                let narrowed: Vec<String> = match shared {
                    None => own,
                    Some(previous) => previous
                        .into_iter()
                        .filter(|name| own.contains(name))
                        .collect(),
                };
                if narrowed.is_empty() {
                    return Err(RinexObsWriteError::CodeListsNotVersionTwo {
                        system: *system,
                        position,
                        code: Some(codes[position].clone()),
                    });
                }
                shared = Some(narrowed);
            }
            if let Some(name) = shared.and_then(|fits| fits.into_iter().next()) {
                names.push(name);
            }
            for (system, codes) in lists {
                if let Some(held) = before.get_mut(system) {
                    held.insert(codes[position].as_str());
                }
            }
        }
        Ok(names)
    }

    /// Constellations holding a code list a version 2 file of this product
    /// written with `names` does not state: no observation or count names them,
    /// a file with observations, or whose version record names another
    /// constellation, builds no list for them, and the names do not read as
    /// their list either. A list the names read as is still in the file, as a
    /// list read from a version 2 file whose body a repair emptied is.
    fn rinex2_unstated_systems(&self, names: &[String]) -> std::collections::BTreeSet<GnssSystem> {
        let mut stated = self.rinex2_observed_systems();
        if stated.is_empty() {
            stated.insert(self.rinex2_fallback_system());
        }
        stated.extend(
            self.header
                .prn_obs_counts
                .iter()
                .filter(|(_, counts)| !counts.is_empty())
                .map(|(sat, _)| sat.system),
        );
        self.header
            .obs_codes
            .iter()
            .filter(|(system, codes)| {
                !stated.contains(system)
                    && super::rinex2_system_obs_codes(**system, names, self.header.version)
                        != **codes
            })
            .map(|(system, _)| *system)
            .collect()
    }

    /// The code lists a version 2 file of this product has to state: for each
    /// constellation an observation or a count names, its list, or with none
    /// of its own what the product's version 2 type names read as for it; and
    /// with no observations, the list of the constellation the file's version
    /// record names, which a reader builds whatever counts there are.
    /// A list nothing names is nothing the file states, so it does not hold the
    /// names back. A count or observation with neither a list nor names to read
    /// names no observable and is refused.
    fn rinex2_stated_lists(&self) -> Result<BTreeMap<GnssSystem, Vec<String>>, RinexObsWriteError> {
        let mut systems = self.rinex2_observed_systems();
        systems.extend(
            self.header
                .prn_obs_counts
                .iter()
                .filter(|(_, counts)| !counts.is_empty())
                .map(|(sat, _)| sat.system),
        );
        // With no observations a file also states the list its version record
        // names, which a reader then builds; counts do not change that.
        let named = !systems.is_empty();
        let observed = !self.rinex2_observed_systems().is_empty();
        let fallback = self.rinex2_fallback_system();
        if !observed {
            systems.insert(fallback);
        }
        let mut lists = BTreeMap::new();
        for system in systems {
            if let Some(codes) = self.header.obs_codes.get(&system) {
                lists.insert(system, codes.clone());
                continue;
            }
            if !self.header.rinex2_types.is_empty() {
                lists.insert(
                    system,
                    super::rinex2_system_obs_codes(
                        system,
                        &self.header.rinex2_types,
                        self.header.version,
                    ),
                );
                continue;
            }
            if !named || (!observed && system == fallback) {
                continue;
            }
            if let Some((sat, counts)) = self
                .header
                .prn_obs_counts
                .iter()
                .find(|(sat, counts)| sat.system == system && !counts.is_empty())
            {
                return Err(RinexObsWriteError::CountsWithoutCodes {
                    satellite: *sat,
                    codes: 0,
                    counts: counts.len(),
                });
            }
            for (epoch_index, epoch) in self.epochs.iter().enumerate() {
                if let Some((sat, values)) = epoch.sats.iter().find(|(sat, _)| sat.system == system)
                {
                    return Err(RinexObsWriteError::ValuesWithoutCodes {
                        epoch_index,
                        satellite: *sat,
                        codes: 0,
                        values: values.len(),
                    });
                }
            }
        }
        Ok(lists)
    }

    /// The code lists a version 2 file of this product reads back as, as the
    /// reader builds them: what the type names read as for each constellation
    /// an observation is from, and with no observations, for the constellation
    /// the version record names. A constellation only a count names reads by
    /// the names without a list of its own, and those names are compared
    /// themselves. With no names, the lists are the product's own.
    fn rinex2_read_lists(&self) -> BTreeMap<GnssSystem, Vec<String>> {
        let names = &self.header.rinex2_types;
        if names.is_empty() {
            return self.header.obs_codes.clone();
        }
        let mut systems = self.rinex2_observed_systems();
        if systems.is_empty() {
            systems.insert(self.rinex2_fallback_system());
        }
        systems
            .into_iter()
            .map(|system| {
                (
                    system,
                    super::rinex2_system_obs_codes(system, names, self.header.version),
                )
            })
            .collect()
    }

    /// Refuse text that would read back as anything but this product.
    fn check_reads_back(
        &self,
        text: &str,
        rinex2_names: Option<&[String]>,
    ) -> Result<(), RinexObsWriteError> {
        let reparsed =
            RinexObs::parse(text).map_err(|error| RinexObsWriteError::ReadBackMismatch {
                what: format!("the written text does not parse: {error}"),
            })?;
        let expected = self.as_written(rinex2_names);
        let found = reparsed.as_written(None);
        if expected == found {
            return Ok(());
        }
        Err(RinexObsWriteError::ReadBackMismatch {
            what: describe_difference(&expected, &found),
        })
    }

    /// This product as written text can state it. The labels of records read
    /// and not retained, the count of records skipped, and the count an epoch
    /// line declared describe the text a product was parsed from; the writer
    /// states the records it has, and a count of them. A version 2 file states
    /// the type names it is written with, `rinex2_names`, and a version 3 file
    /// states none.
    fn as_written(&self, rinex2_names: Option<&[String]>) -> RinexObs {
        let mut product = self.clone();
        product.header.unretained_header_labels.clear();
        product.skipped_records = 0;
        for epoch in &mut product.epochs {
            epoch.declared_record_count = if epoch.flag > 1 {
                epoch.special_records.len()
            } else {
                epoch.sats.len()
            };
        }
        if product.is_rinex2() {
            if let Some(names) = rinex2_names {
                product.header.rinex2_types = names.to_vec();
            }
            let field = product.rinex2_system_field();
            product.header.rinex2_system = field
                .chars()
                .next()
                .filter(|letter| *letter != 'M')
                .and_then(GnssSystem::from_letter);
            product.header.obs_codes = product.rinex2_read_lists();
        } else {
            product.header.rinex2_types.clear();
            product.header.rinex2_system = None;
        }
        product
    }
}

/// The first field on which two products differ, with both values.
fn describe_difference(expected: &RinexObs, found: &RinexObs) -> String {
    fn clipped(text: String) -> String {
        const LIMIT: usize = 400;
        if text.len() <= LIMIT {
            return text;
        }
        let mut end = LIMIT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &text[..end])
    }
    macro_rules! header_field {
        ($($name:ident),+ $(,)?) => {
            $(
                if expected.header.$name != found.header.$name {
                    return clipped(format!(
                        "header {}: {:?} reads back as {:?}",
                        stringify!($name),
                        expected.header.$name,
                        found.header.$name
                    ));
                }
            )+
        };
    }
    header_field!(
        version,
        program_run_by_date,
        comments,
        marker_name,
        marker_number,
        marker_type,
        observer,
        agency,
        receiver,
        antenna,
        approx_position_m,
        antenna_delta_hen_m,
        obs_codes,
        signal_strength_unit,
        interval_s,
        time_of_first_obs,
        time_of_last_obs,
        phase_shifts,
        scale_factors,
        glonass_slots,
        glonass_cod_phs_bis,
        leap_seconds,
        n_satellites,
        prn_obs_counts,
    );
    if expected.epochs.len() != found.epochs.len() {
        return format!(
            "{} epochs read back as {}",
            expected.epochs.len(),
            found.epochs.len()
        );
    }
    for (index, (before, after)) in expected.epochs.iter().zip(&found.epochs).enumerate() {
        if before == after {
            continue;
        }
        macro_rules! epoch_field {
            ($($name:ident),+ $(,)?) => {
                $(
                    if before.$name != after.$name {
                        return clipped(format!(
                            "epoch {index} {}: {:?} reads back as {:?}",
                            stringify!($name),
                            before.$name,
                            after.$name
                        ));
                    }
                )+
            };
        }
        epoch_field!(
            epoch,
            flag,
            rcv_clock_offset_s,
            epoch_picoseconds,
            declared_record_count,
            special_records,
        );
        let satellites_before: Vec<_> = before.sats.keys().collect();
        let satellites_after: Vec<_> = after.sats.keys().collect();
        if satellites_before != satellites_after {
            return clipped(format!(
                "epoch {index} satellites {satellites_before:?} read back as {satellites_after:?}"
            ));
        }
        for (sat, values) in &before.sats {
            let Some(read) = after.sats.get(sat) else {
                return format!("epoch {index} {sat} does not read back");
            };
            if values.len() != read.len() {
                return format!(
                    "epoch {index} {sat} holds {} values and reads back with {}",
                    values.len(),
                    read.len()
                );
            }
            if let Some(at) = values
                .iter()
                .zip(read)
                .position(|(value, back)| value != back)
            {
                return clipped(format!(
                    "epoch {index} {sat} value {at}: {:?} reads back as {:?}",
                    values[at], read[at]
                ));
            }
        }
    }
    "the product reads back differently".to_string()
}
