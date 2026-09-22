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
//! flag greater than one, other than six) keeps the records that followed it as
//! they were written, and they are written back unchanged under their own count.
//! A cycle slip epoch (flag six) is written in the observation record layout,
//! one record per satellite that reports a slip.

use core::fmt::Write as _;
use std::collections::BTreeMap;

use crate::id::GnssSystem;

use super::{
    is_event_flag, ObsEpoch, ObsEpochTime, ObsValue, RinexObs, SatRecords, CYCLE_SLIP_FLAG,
    OBS_FIELD_WIDTH, OBS_VALUE_WIDTH,
};

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
/// Satellites a `SYS / PHASE SHIFT` record lists before continuing,
/// `10(1X,A3)`.
const PHASE_SHIFT_SATELLITES_PER_LINE: usize = 10;
/// Blank columns a `SYS / PHASE SHIFT` continuation record opens with, `18X`.
pub(super) const PHASE_SHIFT_CONTINUATION_COLUMN: usize = 18;
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

/// Why a product could not be written as RINEX text that reads back as the
/// product itself.
///
/// The observation writer never changes what a product says to make it fit a
/// file. [`RinexObs::downgrade_to_rinex2`] is the explicit path for a product
/// that has to lose something to become a version 2 file, and it returns every
/// change it made.
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
    /// A downgrade asked for a version that is not a version 2.
    NotVersionTwo {
        /// The version asked for.
        version: f64,
    },
    /// A version 2 product carrying `SYS / SCALE FACTOR` records. This reader
    /// applies them, but version 2 readers that do not know the record, RTKLIB
    /// among them, read the scaled numbers as physical ones, so the text would
    /// say something else to them. [`RinexObs::downgrade_to_rinex2`] removes the
    /// records and writes the physical values.
    ScaleFactorsInVersionTwo {
        /// How many records the product holds.
        count: usize,
    },
    /// A satellite holding more observations or cycle slips than its
    /// constellation has codes. The values past the codes name no observable,
    /// so no file can say what they are, and a downgrade laying the codes out
    /// would drop them.
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
    /// [`RinexObs::downgrade_to_rinex2`] removes it and reports that.
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
    /// An observation or cycle slip epoch with no epoch time. RINEX lets an
    /// event without a significant epoch leave its epoch fields blank; the
    /// records of every other epoch are tagged with the time they were taken
    /// at, and a reader refuses the line without one.
    EpochTimeMissing {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The flag.
        flag: u8,
    },
    /// Epoch picoseconds in a product below version 4.02, which added them as
    /// five digits after the receiver clock offset. Earlier epoch records have
    /// no field for them. A downgrade to version 2 removes them.
    EpochPicosecondsNotInVersion {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The product's version.
        version: f64,
    },
    /// More observation types than the record's three-digit count can declare.
    TooManyObservationTypes {
        /// How many a version 2 list would need, at least: the layout stops
        /// once it is wider than any header.
        count: usize,
    },
    /// A product whose `obs_codes` is not the union of the lists its file
    /// header and its events declare: the file header's codes first, then each
    /// code a later list declares, in the order first declared. A reader builds
    /// that union from the text, so any other list reads back as it.
    CodeListsNotUnion {
        /// The constellation.
        system: GnssSystem,
    },
    /// An observation or cycle slip under a code the list in effect at its
    /// epoch does not declare, or of a constellation with no list in effect
    /// there. The epoch's records are written by that list, which has no field
    /// for it.
    ValueOutsideDeclaredList {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The satellite.
        satellite: crate::id::GnssSatelliteId,
        /// The code, or `None` when the constellation has no list in effect.
        code: Option<String>,
    },
    /// A version 2 product holding a `declared_obs_codes` list that its file's
    /// type names do not state: a version 2 header declares names, and a reader
    /// rebuilds each constellation's declared list as what they read as.
    DeclaredListNotStated {
        /// The constellation.
        system: GnssSystem,
    },
    /// An event's header record that does not read, so the lists and scale
    /// factors in effect after it are unknown. A product read from text never
    /// holds one.
    EventRecordsUnreadable {
        /// The reader's error.
        message: String,
    },
    /// A code on a physical carrier that the target version cannot represent.
    /// Downgrading would change the carrier frequency while keeping the numeric
    /// observation, which moves the measurement to another signal.
    ObservableNotRepresentable {
        /// The constellation.
        system: GnssSystem,
        /// The original code.
        code: String,
        /// The target version.
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
            Self::NotVersionTwo { version } => {
                write!(f, "RINEX OBS version {version} is not a version 2")
            }
            Self::ScaleFactorsInVersionTwo { count } => write!(
                f,
                "RINEX OBS version 2 would carry {count} SYS / SCALE FACTOR records, which version 2 \
                 readers that do not apply them read as physical values; downgrade_to_rinex2 \
                 removes them"
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
                 the type names do not read as it; downgrade_to_rinex2 removes the list"
            ),
            Self::EpochFlagTooWide { epoch_index, flag } => write!(
                f,
                "RINEX OBS epoch {epoch_index} flag {flag} does not fit the one-digit flag field"
            ),
            Self::EpochTimeMissing { epoch_index, flag } => write!(
                f,
                "RINEX OBS epoch {epoch_index} with flag {flag} has no epoch time, which only an \
                 event may leave blank"
            ),
            Self::EpochPicosecondsNotInVersion {
                epoch_index,
                version,
            } => write!(
                f,
                "RINEX OBS epoch {epoch_index} carries picoseconds, which a version {version} epoch \
                 record has no field for"
            ),
            Self::TooManyObservationTypes { count } => write!(
                f,
                "RINEX OBS version 2 would need at least {count} observation types, more than the \
                 999 its count field declares"
            ),
            Self::CodeListsNotUnion { system } => write!(
                f,
                "RINEX OBS {system} code list is not the union of the lists the header and its \
                 events declare"
            ),
            Self::ValueOutsideDeclaredList {
                epoch_index,
                satellite,
                code: Some(code),
            } => write!(
                f,
                "RINEX OBS epoch {epoch_index} satellite {satellite} holds a value under {code:?}, \
                 which the list in effect at that epoch does not declare"
            ),
            Self::ValueOutsideDeclaredList {
                epoch_index,
                satellite,
                code: None,
            } => write!(
                f,
                "RINEX OBS epoch {epoch_index} satellite {satellite} is of a constellation with no \
                 code list in effect at that epoch"
            ),
            Self::DeclaredListNotStated { system } => write!(
                f,
                "RINEX OBS {system} declared code list is not what the version 2 type names \
                 state for it"
            ),
            Self::EventRecordsUnreadable { message } => {
                write!(f, "RINEX OBS event header records do not read: {message}")
            }
            Self::ObservableNotRepresentable {
                system,
                code,
                version,
            } => write!(
                f,
                "RINEX OBS {system} code {code} is on a carrier that version {version} cannot represent"
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

/// One change [`RinexObs::downgrade_to_rinex2`] made to turn a product into
/// one a version 2 file can state exactly.
#[derive(Debug, Clone, PartialEq)]
pub enum ObsDowngradeChange {
    /// A constellation's code became the one its version 2 column reads as:
    /// version 2 has no field for the tracking attribute it named, or the
    /// column's name reads as a different code for this constellation.
    CodeRenamed {
        /// The constellation.
        system: GnssSystem,
        /// The code it held.
        from: String,
        /// The code its column reads back as.
        to: String,
    },
    /// A constellation's code moved to a different position in its list,
    /// because version 2 names one list for every constellation. Its values
    /// moved with it, so a value is still under its code, but not at the index
    /// it had.
    CodeMoved {
        /// The constellation.
        system: GnssSystem,
        /// The code, as the constellation holds it after the downgrade.
        code: String,
        /// Its zero-based position before.
        from: usize,
        /// Its zero-based position after.
        to: usize,
    },
    /// A constellation gained a code, with no values, because version 2 names
    /// one list for every constellation and another one needed this column.
    CodeAdded {
        /// The constellation.
        system: GnssSystem,
        /// The code its new column reads as.
        code: String,
    },
    /// A constellation's code list was removed. A version 2 file states lists
    /// only for constellations an observation or a `PRN / # OF OBS` count
    /// names, and in a file with no observations for the one its version
    /// record names, so a reader would build none for this one, and the type
    /// names do not read as it.
    CodeListRemoved {
        /// The constellation.
        system: GnssSystem,
        /// The codes it held.
        codes: Vec<String>,
    },
    /// A value rounded to the three decimals an observation field holds. Values
    /// are stored physical, so one read through a scale factor can carry more
    /// precision than a version 2 file, which has no scale factor, can write.
    ValueRounded {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The satellite.
        satellite: crate::id::GnssSatelliteId,
        /// The value's code, as its constellation held it before the downgrade.
        code: String,
        /// The value it held.
        from: f64,
        /// The value it holds now.
        to: f64,
    },
    /// A cycle slip rounded to the three decimals a slip field holds, as
    /// [`ObsDowngradeChange::ValueRounded`] reports for an observation. A slip
    /// is written in the observation record layout, so a slip read through a
    /// scale factor can carry more precision than a version 2 file can write.
    CycleSlipRounded {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The satellite.
        satellite: crate::id::GnssSatelliteId,
        /// The slip's code, as its constellation held it before the downgrade.
        code: String,
        /// The slip it held.
        from: f64,
        /// The slip it holds now.
        to: f64,
    },
    /// The `SYS / SCALE FACTOR` records were removed. Values are stored
    /// physical, so none of them change; a version 2 reader that does not know
    /// the record would otherwise read scaled numbers as physical ones.
    ScaleFactorsRemoved {
        /// How many records there were.
        count: usize,
    },
    /// An epoch's picoseconds were removed, since a version 2 epoch line has no
    /// field for them.
    EpochPicosecondsRemoved {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The picoseconds it held.
        picoseconds: u32,
    },
    /// A receiver clock offset rounded to the nine decimals the version 2
    /// epoch record's `F12.9` field holds; a version 3 epoch record holds
    /// twelve.
    ClockOffsetRounded {
        /// Zero-based epoch index.
        epoch_index: usize,
        /// The offset it held, in seconds.
        from: f64,
        /// The offset it holds now, in seconds.
        to: f64,
    },
    /// A change to the code lists in effect from an event's epoch, laid out
    /// for version 2 as the file header's lists are and reported as theirs
    /// would be: a code renamed, moved or added, or a list removed.
    InEventLists {
        /// Zero-based index of the event epoch.
        epoch_index: usize,
        /// The change.
        change: Box<ObsDowngradeChange>,
    },
    /// The `SYS / PHASE SHIFT` or `GLONASS COD/PHS/BIS` records of a product read
    /// from RINEX 4.00 or later were removed. RINEX 4.00, 4.01 and 4.02 say of
    /// both records that "the lines should be ignored by RINEX decoders and
    /// encoders", so the product applies none of them; a version 2 file would
    /// give them the meaning the version 4 file declared them not to have.
    DeprecatedRecordsRemoved {
        /// The records' label.
        label: String,
        /// Zero-based index of the event epoch that carried them, or `None` for
        /// the file header.
        epoch_index: Option<usize>,
        /// The records removed: an event's as it carried them, the file
        /// header's as it would write them.
        records: Vec<String>,
    },
    /// An event's special records were rewritten. Its type records became the
    /// version 2 `# / TYPES OF OBSERV` records its lists lay out as; a
    /// `SYS / SCALE FACTOR` record was removed, as the file header's are; or a
    /// version 2 type record a version 3 event carried without effect was
    /// removed, since a version 2 reader applies it.
    EventRecordsRewritten {
        /// Zero-based index of the event epoch.
        epoch_index: usize,
        /// The records it held.
        from: Vec<String>,
        /// The records it holds now.
        to: Vec<String>,
    },
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
    /// with the first field that would change. A product
    /// that has to lose something to become a version 2 file goes through
    /// [`RinexObs::downgrade_to_rinex2`], which returns every change it made.
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
        if let Some((epoch_index, epoch)) = self
            .epochs
            .iter()
            .enumerate()
            .find(|(_, epoch)| epoch.epoch.is_none() && !is_event_flag(epoch.flag))
        {
            return Err(RinexObsWriteError::EpochTimeMissing {
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
        // The lists, names and scale factors an event declares are in effect for
        // the epochs after it, and each epoch is written by them.
        let timeline =
            self.header_timeline()
                .map_err(|error| RinexObsWriteError::EventRecordsUnreadable {
                    message: error.to_string(),
                })?;
        if self.is_rinex2()
            && timeline
                .segments()
                .any(|(_, header)| !header.scale_factors.is_empty())
        {
            // Every record the file header and the events declare.
            let count = self.header.scale_factors.len()
                + self
                    .epochs
                    .iter()
                    .filter(|epoch| super::applies_header_records(epoch.flag))
                    .flat_map(|epoch| &epoch.special_records)
                    .filter(|record| {
                        super::event_record_label(record) == "SYS / SCALE FACTOR"
                            && !record.starts_with(' ')
                    })
                    .count();
            return Err(RinexObsWriteError::ScaleFactorsInVersionTwo { count });
        }
        let event_names = self.rinex2_event_names();
        let rinex2_names = if self.is_rinex2() {
            let names = if event_names.is_empty() {
                self.rinex2_names()?
            } else {
                self.rinex2_event_header_names(&event_names)?
            };
            if let Some(system) = self.rinex2_unstated_systems(&names).into_iter().next() {
                return Err(RinexObsWriteError::CodeListNotStated { system });
            }
            self.check_rinex2_declared(&names)?;
            Some(names)
        } else {
            self.check_code_union(&timeline)?;
            self.check_counts_declared()?;
            None
        };
        let mut layouts = EpochLayouts {
            product: self,
            timeline: &timeline,
            header_names: rinex2_names.as_deref(),
            event_names: &event_names,
            cache: std::collections::HashMap::new(),
        };
        self.check_values_declared(&mut layouts)?;
        let mut out = String::new();
        self.write_header(&mut out, rinex2_names.as_deref());
        self.write_body(&mut out, &mut layouts);
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
            // The lists the file header declares; an event declares its own.
            for (system, codes) in &h.declared_obs_codes {
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

    fn write_body(&self, out: &mut String, layouts: &mut EpochLayouts<'_>) {
        for (epoch_index, epoch) in self.epochs.iter().enumerate() {
            if layouts.header_names.is_some() {
                self.write_epoch_v2(out, epoch_index, epoch, layouts);
            } else {
                self.write_epoch(out, epoch_index, epoch, layouts);
            }
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

    /// Constellations this product's observations and cycle slips are from. A
    /// version 2 epoch names each satellite whose record follows, a slip's as
    /// much as an observation's, and a reader builds a code list for its
    /// constellation either way.
    fn rinex2_observed_systems(&self) -> std::collections::BTreeSet<GnssSystem> {
        self.epochs
            .iter()
            .flat_map(|epoch| epoch_record_satellites(epoch).map(|sat| sat.system))
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

    /// How this product's constellations' code lists lay out against one
    /// version 2 list, renaming as few codes as any layout can.
    ///
    /// The version 2 reader gives a constellation a canonical code at the first
    /// column whose name reads as it, and the name itself at every later one. So
    /// whether a canonical code is kept does not depend on column order: it is
    /// kept exactly when some column's name reads as it, and adding a column
    /// never takes a code away. The fewest renames therefore come from naming a
    /// column for every code any name reads as. Only a code no name spells, or a
    /// second copy of a code its constellation already holds, has to be renamed.
    ///
    /// A name kept as written - Galileo's `P1`, a second BeiDou `C2` - needs, for
    /// its constellation, a column that reads as the code it stands for before
    /// it. Every naming column comes first and every kept-as-written column
    /// after, which satisfies that for every constellation at once.
    ///
    /// A code that must be renamed gets a column of its own, named with its own
    /// spelling and placed after every other, so it reads back as that name or
    /// as the code the name stands for. A column already reading as the same
    /// kind and band would say more than the file can: a second GPS `C1C` there
    /// reads back as the P code.
    fn rinex2_obs_layout(&self) -> Rinex2ObsLayout {
        let (names, slots) = self.rinex2_obs_columns();
        // A layout wider than a header declares is refused, and ordering it
        // would build matrices as wide as it for nothing.
        if names.len() > super::MAX_OBS_TYPE_COUNT {
            return Rinex2ObsLayout { names, slots };
        }
        self.ordered_by_original_positions(names, slots)
    }

    /// The layout's columns in the order they are built, and the column each
    /// constellation's code sits at in them, before ordering.
    ///
    /// Every step asks what a constellation reads a list of names as, which the
    /// indexes here answer without reading the names again. A name a
    /// constellation carries is two ASCII characters and reads as a code of
    /// three, and any other name reads as itself; a name kept as written is one
    /// or two bytes, so no code is one, and a column reads as it only when named
    /// it.
    /// Which column comes first among those reading as each code settles the
    /// rest. Spellings for codes no name spells, which may be any length, are
    /// added only after the names kept as written are counted.
    pub(super) fn rinex2_obs_columns(
        &self,
    ) -> (Vec<String>, BTreeMap<GnssSystem, Vec<Option<usize>>>) {
        use std::collections::{HashMap, HashSet};

        /// A constellation's reading of the names so far, extended as names are
        /// added: the codes, the codes given, and the columns reading as each
        /// code and as each name and code, each list with the position before
        /// which no column is free.
        #[derive(Default)]
        struct Reading {
            codes: Vec<String>,
            given: HashSet<String>,
            by_code: HashMap<String, (Vec<usize>, usize)>,
            by_name: HashMap<(String, String), (Vec<usize>, usize)>,
        }

        /// The first free column in a list of columns taken in order. A column
        /// once taken stays taken, so the position moves past those.
        fn first_free(
            entry: Option<&mut (Vec<usize>, usize)>,
            free: impl Fn(usize) -> bool,
        ) -> Option<usize> {
            let (columns, next) = entry?;
            while columns.get(*next).is_some_and(|&column| !free(column)) {
                *next += 1;
            }
            columns.get(*next).copied()
        }

        let version = self.header.version;
        let stated = self.rinex2_layout_lists();
        let lists = &stated;
        let reads_as = |system: GnssSystem, name: &str| -> Option<String> {
            super::rinex2_name_allowed(system, name, version)
                .then(|| super::canonical_rinex2_obs_code(system, name, version))
        };
        let held: BTreeMap<GnssSystem, HashSet<&str>> = lists
            .iter()
            .map(|(system, codes)| (*system, codes.iter().map(String::as_str).collect()))
            .collect();
        // For each constellation and code, the first provider reading as it and
        // the first column kept as written reading as it; and how many columns
        // carry each name.
        let mut first_provider: HashMap<(GnssSystem, String), String> = HashMap::new();
        let mut first_kept: HashMap<(GnssSystem, String), String> = HashMap::new();
        let mut named: HashMap<String, usize> = HashMap::new();
        let note = |first: &mut HashMap<(GnssSystem, String), String>,
                    named: &mut HashMap<String, usize>,
                    name: &str| {
            for system in lists.keys() {
                if let Some(read) = reads_as(*system, name) {
                    first
                        .entry((*system, read))
                        .or_insert_with(|| name.to_string());
                }
            }
            *named.entry(name.to_string()).or_default() += 1;
        };

        // Name a column for every canonical code some name reads as, choosing
        // the name that reads as the most codes still without one, and the
        // earlier candidate on a tie, since that keeps the band it was measured
        // on.
        let mut providers: Vec<String> = Vec::new();
        for (system, codes) in lists {
            let mut seen: HashSet<&str> = HashSet::new();
            for code in codes {
                if super::rinex2_kept_as_written(code)
                    || !seen.insert(code.as_str())
                    || first_provider.contains_key(&(*system, code.clone()))
                {
                    continue;
                }
                let mut best: Option<(String, usize)> = None;
                for name in super::rinex2_obs_code_candidates(*system, code, version) {
                    if reads_as(*system, &name).as_deref() != Some(code.as_str()) {
                        continue;
                    }
                    let gain = lists
                        .keys()
                        .filter(|other| {
                            reads_as(**other, &name).is_some_and(|read| {
                                held[*other].contains(read.as_str())
                                    && !first_provider.contains_key(&(**other, read))
                            })
                        })
                        .count();
                    if best.as_ref().is_none_or(|(_, most)| gain > *most) {
                        best = Some((name, gain));
                    }
                }
                if let Some((name, _)) = best {
                    note(&mut first_provider, &mut named, &name);
                    providers.push(name);
                }
            }
        }

        // Columns for names kept as written, each after a column that reads, for
        // its constellation, as the code the name stands for.
        let mut kept_as_written: Vec<String> = Vec::new();
        for (system, codes) in lists {
            let mut wanted: BTreeMap<&str, usize> = BTreeMap::new();
            for code in codes
                .iter()
                .filter(|code| super::rinex2_kept_as_written(code))
            {
                *wanted.entry(code.as_str()).or_default() += 1;
            }
            for (name, count) in wanted {
                let stands_for = reads_as(*system, name);
                if let Some(stands_for) = &stands_for {
                    if !first_provider.contains_key(&(*system, stands_for.clone())) {
                        note(&mut first_provider, &mut named, name);
                        providers.push(name.to_string());
                    }
                }
                // A column already reading as this name for this constellation
                // counts, whether it was named for this code or another one:
                // every column named it, but the first column reading as the
                // code it stands for when that column is named it. Providers
                // all come before columns kept as written.
                let columns = named.get(name).copied().unwrap_or(0);
                let first_reader = stands_for.and_then(|code| {
                    first_provider
                        .get(&(*system, code.clone()))
                        .or_else(|| first_kept.get(&(*system, code)))
                });
                let have = columns
                    - usize::from(columns > 0 && first_reader.is_some_and(|first| first == name));
                for _ in have..count {
                    note(&mut first_kept, &mut named, name);
                    kept_as_written.push(name.to_string());
                }
            }
        }
        let mut names = providers;
        names.extend(kept_as_written);

        // Place every code where its column reads back as it, each taking the
        // next column reading as it.
        let mut slots: BTreeMap<GnssSystem, Vec<Option<usize>>> = BTreeMap::new();
        let mut renamed: Vec<(GnssSystem, usize)> = Vec::new();
        for (system, codes) in lists {
            let read = super::rinex2_system_obs_codes(*system, &names, version);
            let mut reading: HashMap<&str, (Vec<usize>, usize)> = HashMap::new();
            for (column, code) in read.iter().enumerate() {
                reading.entry(code.as_str()).or_default().0.push(column);
            }
            let mut row: Vec<Option<usize>> = vec![None; names.len()];
            for (index, code) in codes.iter().enumerate() {
                let column = reading.get_mut(code.as_str()).and_then(|(columns, next)| {
                    let column = columns.get(*next).copied();
                    *next += 1;
                    column
                });
                match column {
                    Some(column) => row[column] = Some(index),
                    None => renamed.push((*system, index)),
                }
            }
            slots.insert(*system, row);
        }

        // What no layout keeps. A code no name spells keeps its kind and band:
        // it shares a free column that already reads, for its constellation, as
        // the code its own spelling would, provided the constellation holds no
        // such code itself. Otherwise the code takes a column named with its own
        // spelling, sharing one another constellation's code already took under
        // that name wherever it reads the same there, so constellations holding
        // the same second copies need one column between them rather than one
        // each. It never reads as a different tracking attribute.
        let first_index: BTreeMap<GnssSystem, HashMap<&str, usize>> = lists
            .iter()
            .map(|(system, codes)| {
                let mut first = HashMap::new();
                for (index, code) in codes.iter().enumerate() {
                    first.entry(code.as_str()).or_insert(index);
                }
                (*system, first)
            })
            .collect();
        let mut readings: BTreeMap<GnssSystem, Reading> = BTreeMap::new();
        for (system, index) in renamed {
            // Past the widest header there is no layout to finish.
            if names.len() > super::MAX_OBS_TYPE_COUNT {
                break;
            }
            let code = &lists[&system][index];
            let spelling = super::rinex2_obs_code_candidates(system, code, version)
                .into_iter()
                .next()
                .unwrap_or_else(|| code.chars().take(2).collect());
            let duplicate = first_index[&system][code.as_str()] < index;
            let reading = readings.entry(system).or_default();
            for column in reading.codes.len()..names.len() {
                let next =
                    super::rinex2_next_obs_code(system, &names[column], version, &reading.given);
                reading.given.insert(next.clone());
                reading
                    .by_code
                    .entry(next.clone())
                    .or_default()
                    .0
                    .push(column);
                reading
                    .by_name
                    .entry((names[column].clone(), next.clone()))
                    .or_default()
                    .0
                    .push(column);
                reading.codes.push(next);
            }
            let row = &slots[&system];
            let free = |column: usize| row.get(column).copied().flatten().is_none();
            let target = match reads_as(system, &spelling) {
                Some(target) if !duplicate && !held[&system].contains(target.as_str()) => {
                    first_free(reading.by_code.get_mut(&target), free)
                }
                _ => None,
            };
            let found = target.or_else(|| {
                let would_read =
                    super::rinex2_next_obs_code(system, &spelling, version, &reading.given);
                first_free(
                    reading.by_name.get_mut(&(spelling.clone(), would_read)),
                    free,
                )
            });
            let column = match found {
                Some(column) => column,
                None => {
                    names.push(spelling);
                    for row in slots.values_mut() {
                        row.resize(names.len(), None);
                    }
                    names.len() - 1
                }
            };
            let row = slots.entry(system).or_default();
            row.resize(names.len(), None);
            row[column] = Some(index);
        }
        for row in slots.values_mut() {
            row.resize(names.len(), None);
        }
        (names, slots)
    }

    /// The same layout with its columns reordered to move as few codes as the
    /// reading rule allows.
    ///
    /// The layout is built with naming columns first and everything else after,
    /// an order the reader reads as intended. A code is reported moved when its
    /// column's position differs from the position it held, so the order moving
    /// the fewest keeps the most codes in place. Setting the reading rule aside,
    /// that is an assignment of columns to positions, solved exactly. The rule
    /// adds ordering: a group of names a constellation reads as one code gives
    /// the code to its first column, so a held first column has to precede the
    /// rest of its group, and a group whose held columns read as written has to
    /// begin with an unheld one. Those are met by branch and bound: an
    /// assignment breaking one splits into subproblems that narrow the positions
    /// the two columns may take, or choose the group's first column, and
    /// subproblems are taken most codes kept first, so the first assignment
    /// keeping every rule keeps the most codes any order does. Every
    /// assignment solved is also repaired into an order keeping the rules -
    /// columns in its sequence, each once the columns it follows are placed,
    /// or with each pulled to just before its earliest follower - and the best
    /// such order is kept as the search goes. The search has half of
    /// [`LAYOUT_ORDER_SEARCH_WORK`] steps, counted inside the assignment method.
    /// When it cannot prove an order best in them, the best distinct orders it
    /// saw are improved with the rest: one column moved or two swapped at a
    /// time, while that keeps the rules and more codes in place. The best order
    /// found is used; it reads back exactly and every move it makes is
    /// reported, but another order could move fewer.
    fn ordered_by_original_positions(
        &self,
        names: Vec<String>,
        slots: BTreeMap<GnssSystem, Vec<Option<usize>>>,
    ) -> Rinex2ObsLayout {
        let version = self.header.version;
        let rows: Vec<&Vec<Option<usize>>> = slots.values().collect();
        let search = OrderSearch::new(
            slots
                .keys()
                .map(|system| {
                    let built = super::rinex2_system_obs_codes(*system, &names, version);
                    names
                        .iter()
                        .enumerate()
                        .map(|(column, name)| {
                            super::rinex2_name_allowed(*system, name, version).then(|| {
                                let canonical =
                                    super::canonical_rinex2_obs_code(*system, name, version);
                                let first = built[column] == canonical;
                                (canonical, first)
                            })
                        })
                        .collect()
                })
                .collect(),
            &rows,
            &names,
        );
        let chosen = search.run();
        let ordered_names = chosen.iter().map(|&column| names[column].clone()).collect();
        let ordered_slots = slots
            .iter()
            .map(|(system, row)| (*system, chosen.iter().map(|&column| row[column]).collect()))
            .collect();
        Rinex2ObsLayout {
            names: ordered_names,
            slots: ordered_slots,
        }
    }

    /// Write one version 2 epoch: the record, its satellite list continued
    /// twelve to a line, then each satellite's observations five to a line.
    fn write_epoch_v2(
        &self,
        out: &mut String,
        epoch_index: usize,
        epoch: &ObsEpoch,
        layouts: &mut EpochLayouts<'_>,
    ) {
        // Every record runs to the number of names in effect at the epoch.
        let width = layouts.names_at(epoch_index).len();
        // An event without a significant epoch leaves the epoch fields, the
        // first 28 columns, blank.
        let head = match epoch.epoch {
            Some(t) => format!(
                " {:02} {:2} {:2} {:2} {:2}{:11.7}  {}",
                t.year.rem_euclid(100),
                t.month,
                t.day,
                t.hour,
                t.minute,
                t.second,
                epoch.flag
            ),
            None => format!("{:28}{}", "", epoch.flag),
        };
        // An event names no satellites. Its own records follow the epoch line,
        // and its declared count is how many of them there are.
        let event = is_event_flag(epoch.flag);
        let records = epoch_records(epoch);
        // `12(A1,I2)`: the constellation letter then the number, space padded,
        // which is what a version 2 reader expects.
        let satellites: Vec<String> = records
            .keys()
            .map(|sat| format!("{}{:2}", sat.system.letter(), sat.prn))
            .collect();
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
            "{head}{:>3}{tail}",
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
        // Each satellite's values are in the order its constellation reads the
        // names in effect as, placed by code from the union they are held under.
        // A cycle slip record is laid out as an observation record is.
        for (sat, values) in records {
            let stretch = layouts.prepare(epoch_index, sat.system);
            match layouts.get(stretch, sat.system) {
                Some(layout) if !layout.identity => {
                    write_sat_record_v2(out, &placed_values(values, &layout.positions), width);
                }
                _ => write_sat_record_v2(out, values, width),
            }
        }
    }

    fn write_epoch(
        &self,
        out: &mut String,
        epoch_index: usize,
        epoch: &ObsEpoch,
        layouts: &mut EpochLayouts<'_>,
    ) {
        // An event without a significant epoch leaves the epoch fields, the 28
        // columns after the `>`, blank.
        let time = match epoch.epoch {
            Some(t) => format!(
                " {:04} {:02} {:02} {:02} {:02}{:11.7}",
                t.year, t.month, t.day, t.hour, t.minute, t.second
            ),
            None => " ".repeat(28),
        };
        // An event declares how many of its own records follow; every other
        // epoch declares the satellites whose observations or slips follow.
        let count = declared_count(epoch);
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
            ">{time}  {}{:3}{clock}{picoseconds}",
            epoch.flag, count
        );
        if is_event_flag(epoch.flag) {
            for record in &epoch.special_records {
                let _ = writeln!(out, "{record}");
            }
            return;
        }
        // Each satellite's values are written by the list in effect at the
        // epoch, placed by code from the union, and scaled by the factors in
        // effect.
        let timeline = layouts.timeline;
        let scale_factors = &timeline.at(epoch_index).scale_factors;
        for (sat, values) in epoch_records(epoch) {
            let stretch = layouts.prepare(epoch_index, sat.system);
            match layouts.get(stretch, sat.system) {
                Some(layout) if !layout.identity => {
                    let placed = placed_values(values, &layout.positions);
                    write_sat_record(out, *sat, &placed, &layout.list, scale_factors);
                }
                Some(layout) => write_sat_record(out, *sat, values, &layout.list, scale_factors),
                None => write_sat_record(out, *sat, values, &[], scale_factors),
            }
        }
    }
}

/// Write one satellite's record in the version 3 layout, its values in the
/// order of `codes`, each scaled by the factor in effect for its code.
fn write_sat_record(
    out: &mut String,
    sat: crate::id::GnssSatelliteId,
    values: &[ObsValue],
    codes: &[String],
    scale_factors: &[super::ObsScaleFactor],
) {
    let mut line = format!("{:<3}", sat.to_string());
    for (index, value) in values.iter().enumerate() {
        let code = codes.get(index).map(String::as_str);
        push_obs_value(
            &mut line,
            *value,
            scale_for(scale_factors, sat.system, code),
        );
    }
    // Trailing blank observations carry no information; drop them.
    let trimmed = line.trim_end();
    out.push_str(trimmed);
    out.push('\n');
}

/// The `SYS / SCALE FACTOR` divisor in force for a system/code, mirroring the
/// parser's lookup so a value re-multiplies back to its stored ASCII.
fn scale_for(
    scale_factors: &[super::ObsScaleFactor],
    system: GnssSystem,
    code: Option<&str>,
) -> f64 {
    code.map_or(1.0, |code| {
        super::scale_factor_in(scale_factors, system, code)
    })
}

/// A blank observation field.
const BLANK_VALUE: ObsValue = ObsValue {
    value: None,
    lli: None,
    ssi: None,
};

/// Values held under the union, laid out by a list: each at its code's
/// position in the union, blank where the list's code has none.
fn placed_values(values: &[ObsValue], positions: &[Option<usize>]) -> Vec<ObsValue> {
    positions
        .iter()
        .map(|position| {
            position
                .and_then(|position| values.get(position))
                .copied()
                .unwrap_or(BLANK_VALUE)
        })
        .collect()
}

/// Rewrite an event's special records for a version 2 file. A
/// `SYS / SCALE FACTOR` record is removed, as the file header's are. The type
/// records a version 2 reader applies - a version 3 event's
/// `SYS / # / OBS TYPES`, and a `# / TYPES OF OBSERV` in any event - are
/// replaced by `names`, the version 2 records its lists lay out as, where the
/// first of them stood; with `names` `None`, a version 2 event keeps its own
/// and a version 3 event's are removed, since at version 3 they took no
/// effect. The rewrite is reported when the records change.
fn rewrite_event_records(
    source_version: f64,
    epoch_index: usize,
    epoch: &mut ObsEpoch,
    names: Option<Vec<String>>,
    changes: &mut Vec<ObsDowngradeChange>,
) {
    if !super::applies_header_records(epoch.flag) {
        return;
    }
    let source_rinex2 = source_version.floor() as i64 == 2;
    let mut records = Vec::with_capacity(epoch.special_records.len());
    let mut placed = false;
    for record in &epoch.special_records {
        let label = super::event_record_label(record);
        if label == "SYS / SCALE FACTOR" {
            continue;
        }
        let types =
            label == "# / TYPES OF OBSERV" || (!source_rinex2 && label == "SYS / # / OBS TYPES");
        if !types {
            records.push(record.clone());
            continue;
        }
        match &names {
            Some(lines) => {
                if !placed {
                    records.extend(lines.iter().cloned());
                    placed = true;
                }
            }
            None if source_rinex2 => records.push(record.clone()),
            None => {}
        }
    }
    if records != epoch.special_records {
        changes.push(ObsDowngradeChange::EventRecordsRewritten {
            epoch_index,
            from: epoch.special_records.clone(),
            to: records.clone(),
        });
        epoch.declared_record_count = records.len();
        epoch.special_records = records;
    }
}

/// Labels of the records RINEX 4.00 and later declare to be ignored.
const DEPRECATED_LABELS: [&str; 2] = ["SYS / PHASE SHIFT", "GLONASS COD/PHS/BIS"];

/// Remove the `SYS / PHASE SHIFT` and `GLONASS COD/PHS/BIS` records of a
/// product read from RINEX 4.00 or later, from its file header and from each
/// event, and report each removal. Version 4 declares them to be ignored, and a
/// version 2 file would apply them.
fn remove_deprecated_records(product: &mut RinexObs, changes: &mut Vec<ObsDowngradeChange>) {
    let lines = |text: String| text.lines().map(str::to_string).collect::<Vec<_>>();
    if !product.header.phase_shifts.is_empty() {
        let mut text = String::new();
        for shift in &product.header.phase_shifts {
            write_phase_shift(&mut text, shift);
        }
        changes.push(ObsDowngradeChange::DeprecatedRecordsRemoved {
            label: DEPRECATED_LABELS[0].to_string(),
            epoch_index: None,
            records: lines(text),
        });
        product.header.phase_shifts.clear();
    }
    if let Some(entries) = product.header.glonass_cod_phs_bis.take() {
        let mut text = String::new();
        write_glonass_cod_phs_bis(&mut text, &entries);
        changes.push(ObsDowngradeChange::DeprecatedRecordsRemoved {
            label: DEPRECATED_LABELS[1].to_string(),
            epoch_index: None,
            records: lines(text),
        });
    }
    for (epoch_index, epoch) in product.epochs.iter_mut().enumerate() {
        if !super::applies_header_records(epoch.flag) {
            continue;
        }
        for label in DEPRECATED_LABELS {
            let (removed, kept): (Vec<String>, Vec<String>) = epoch
                .special_records
                .iter()
                .cloned()
                .partition(|record| super::event_record_label(record) == label);
            if removed.is_empty() {
                continue;
            }
            epoch.declared_record_count = kept.len();
            epoch.special_records = kept;
            changes.push(ObsDowngradeChange::DeprecatedRecordsRemoved {
                label: label.to_string(),
                epoch_index: Some(epoch_index),
                records: removed,
            });
        }
    }
}

/// Whether a field holds nothing: no value and no indicator.
fn is_blank(value: &ObsValue) -> bool {
    value.value.is_none() && value.lli.is_none() && value.ssi.is_none()
}

/// The code list in effect for one constellation over a stretch of epochs, and
/// where each of its codes sits in the product's union.
struct Layout {
    list: Vec<String>,
    positions: Vec<Option<usize>>,
    /// Whether the list is the union itself, so values are written as held.
    identity: bool,
}

/// The code lists each epoch is written by: at version 3 the lists the header
/// in effect declares, at version 2 what the names in effect read as.
struct EpochLayouts<'a> {
    product: &'a RinexObs,
    timeline: &'a super::ObsHeaderTimeline,
    /// The names a version 2 file header is written with; `None` at version 3.
    header_names: Option<&'a [String]>,
    /// The names each version 2 event declaring them carries, by epoch index.
    event_names: &'a [(usize, Vec<String>)],
    cache: std::collections::HashMap<(usize, GnssSystem), Option<Layout>>,
}

impl<'a> EpochLayouts<'a> {
    /// The stretch of epochs sharing a list, identified by the lists declared
    /// at or before the epoch.
    fn stretch(&self, epoch_index: usize) -> usize {
        match self.header_names {
            Some(_) => self
                .event_names
                .partition_point(|(first, _)| *first <= epoch_index),
            None => self.timeline.segment_index(epoch_index),
        }
    }

    /// The version 2 names in effect at an epoch.
    fn names_at(&self, epoch_index: usize) -> &'a [String] {
        let stretch = self
            .event_names
            .partition_point(|(first, _)| *first <= epoch_index);
        match stretch
            .checked_sub(1)
            .and_then(|index| self.event_names.get(index))
        {
            Some((_, names)) => names,
            None => self.header_names.unwrap_or_default(),
        }
    }

    /// Work out the layout of a constellation at an epoch, once per stretch,
    /// and return the stretch to look it up by.
    fn prepare(&mut self, epoch_index: usize, system: GnssSystem) -> usize {
        let stretch = self.stretch(epoch_index);
        let key = (stretch, system);
        if !self.cache.contains_key(&key) {
            let list = match self.header_names {
                Some(_) => Some(super::rinex2_system_obs_codes(
                    system,
                    self.names_at(epoch_index),
                    self.product.header.version,
                )),
                None => self
                    .timeline
                    .at(epoch_index)
                    .declared_obs_codes
                    .get(&system)
                    .cloned(),
            };
            let union = self.union_of(system);
            let layout = list.map(|list| {
                let positions = super::union_positions(&list, &union);
                let identity = list.as_slice() == &*union;
                Layout {
                    list,
                    positions,
                    identity,
                }
            });
            self.cache.insert(key, layout);
        }
        stretch
    }

    /// The union a constellation's values are held under: its list, or at
    /// version 2, for a constellation holding none, what the file's names read
    /// as for it.
    fn union_of(&self, system: GnssSystem) -> std::borrow::Cow<'a, [String]> {
        let product = self.product;
        match (product.header.obs_codes.get(&system), self.header_names) {
            (Some(list), _) => std::borrow::Cow::Borrowed(list.as_slice()),
            (None, Some(names)) => {
                std::borrow::Cow::Owned(product.rinex2_union_read(system, names, self.event_names))
            }
            (None, None) => std::borrow::Cow::Borrowed(&[]),
        }
    }

    fn get(&self, stretch: usize, system: GnssSystem) -> Option<&Layout> {
        self.cache.get(&(stretch, system)).and_then(Option::as_ref)
    }
}

/// The records an epoch writes in the observation record layout: a cycle slip
/// epoch's slips, an event's none, and every other epoch's observations.
fn epoch_records(epoch: &ObsEpoch) -> &SatRecords {
    static NONE: SatRecords = BTreeMap::new();
    if epoch.flag == CYCLE_SLIP_FLAG {
        &epoch.cycle_slips
    } else if is_event_flag(epoch.flag) {
        &NONE
    } else {
        &epoch.sats
    }
}

/// The count an epoch line declares for the records written after it.
fn declared_count(epoch: &ObsEpoch) -> usize {
    if is_event_flag(epoch.flag) {
        epoch.special_records.len()
    } else {
        epoch_records(epoch).len()
    }
}

/// Every satellite an epoch holds observations or cycle slips for.
fn epoch_record_satellites(
    epoch: &ObsEpoch,
) -> impl Iterator<Item = &crate::id::GnssSatelliteId> + '_ {
    epoch.sats.keys().chain(epoch.cycle_slips.keys())
}

/// Append a header line: content padded into the first 60 columns, then the
/// 20-column record label.
fn push_header_line(out: &mut String, content: &str, label: &str) {
    out.push_str(&header_line(content, label));
    out.push('\n');
}

/// A header record: content padded into the first 60 columns, then the label.
fn header_line(content: &str, label: &str) -> String {
    let content = header_content_60(content);
    format!("{content:<HEADER_CONTENT_WIDTH$}{label}")
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
    for line in obs_types_v2_lines(codes) {
        out.push_str(&line);
        out.push('\n');
    }
}

/// The version 2 `# / TYPES OF OBSERV` records for a list of names: the count
/// on the first record, the names continued nine to a record.
fn obs_types_v2_lines(codes: &[String]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut chunks = codes.chunks(RINEX2_OBS_TYPES_PER_LINE);
    let first = chunks.next().unwrap_or_default();
    let mut content = format!("{:6}", codes.len());
    for code in first {
        let _ = write!(content, "    {code:>2}");
    }
    lines.push(header_line(&content, "# / TYPES OF OBSERV"));
    for chunk in chunks {
        let mut content = " ".repeat(6);
        for code in chunk {
            let _ = write!(content, "    {code:>2}");
        }
        lines.push(header_line(&content, "# / TYPES OF OBSERV"));
    }
    lines
}

/// Write one `SYS / PHASE SHIFT` record. The optional satellite list is emitted
/// with its count when present (otherwise the correction applies system-wide).
fn write_phase_shift(out: &mut String, shift: &super::ObsPhaseShift) {
    push_header_line(out, &phase_shift_content(shift), "SYS / PHASE SHIFT");
    // Satellites past the first record's ten continue, ten to a record.
    for chunk in phase_shift_tokens(shift)
        .chunks(PHASE_SHIFT_SATELLITES_PER_LINE)
        .skip(1)
    {
        let mut content = " ".repeat(PHASE_SHIFT_CONTINUATION_COLUMN);
        for token in chunk {
            let _ = write!(content, " {token}");
        }
        push_header_line(out, &content, "SYS / PHASE SHIFT");
    }
}

/// The satellites a phase shift names, as a record writes them: the ones
/// [`crate::id::GnssSatelliteId`] holds, then the designators it does not hold,
/// as they were written.
fn phase_shift_tokens(shift: &super::ObsPhaseShift) -> Vec<String> {
    shift
        .satellites
        .iter()
        .map(ToString::to_string)
        .chain(shift.unrepresentable_satellites.iter().cloned())
        .collect()
}

/// The 60-column content of the first `SYS / PHASE SHIFT` record for a shift,
/// holding its count and first ten satellites, in the record's columns.
pub(super) fn phase_shift_content(shift: &super::ObsPhaseShift) -> String {
    // `A1,1X,A3,1X,F8.5,2X,I2.2,10(1X,A3)`: the correction blank if none, the
    // count blank for every satellite of the system.
    let correction = shift.correction_cycles.map_or_else(
        || " ".repeat(PHASE_SHIFT_CORRECTION_WIDTH),
        |correction| format!("{correction:8.5}"),
    );
    // A record naming only its constellation leaves every other field blank.
    let Some(code) = &shift.code else {
        return shift.system.letter().to_string();
    };
    let mut content = format!("{} {:<3} {correction}", shift.system.letter(), code);
    if !shift.covers_every_satellite() {
        let tokens = phase_shift_tokens(shift);
        let _ = write!(content, "  {:02}", tokens.len());
        for token in tokens.iter().take(PHASE_SHIFT_SATELLITES_PER_LINE) {
            let _ = write!(content, " {token}");
        }
    }
    content.trim_end().to_string()
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

fn write_glonass_cod_phs_bis(out: &mut String, entries: &[(String, Option<f64>)]) {
    if entries.is_empty() {
        push_header_line(out, "", "GLONASS COD/PHS/BIS");
        return;
    }
    // Each entry takes thirteen of the sixty columns, `1X,A3,1X,F8.3`, so a
    // fifth one would be cut off the end of the line and lost. The record
    // continues on another line instead, which is how the reader takes it
    // back. A blank bias is written blank in its columns.
    for chunk in entries.chunks(GLONASS_BIAS_ENTRIES_PER_LINE) {
        let mut content = String::new();
        for (code, value) in chunk {
            match value {
                Some(value) => {
                    let _ = write!(content, " {code:>3} {value:8.3}");
                }
                None => {
                    let _ = write!(content, " {code:>3} {:8}", "");
                }
            }
        }
        push_header_line(out, content.trim_end(), "GLONASS COD/PHS/BIS");
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

    /// The version 2 type names each event declaring them carries, with the
    /// index of its epoch, in file order. Empty at version 3 and for a product
    /// whose event records do not read.
    fn rinex2_event_names(&self) -> Vec<(usize, Vec<String>)> {
        if !self.is_rinex2() {
            return Vec::new();
        }
        let version = self.header.version;
        let Ok(timeline) = self.header_timeline() else {
            return Vec::new();
        };
        self.epochs
            .iter()
            .enumerate()
            .filter(|(_, epoch)| {
                super::applies_header_records(epoch.flag)
                    && super::event_declares_label(
                        version,
                        &epoch.special_records,
                        "# / TYPES OF OBSERV",
                    )
            })
            .map(|(index, _)| (index, timeline.at(index).rinex2_types.clone()))
            .collect()
    }

    /// What a constellation reads a version 2 file's header names and every
    /// event's names as, together: the union of those readings.
    fn rinex2_union_read(
        &self,
        system: GnssSystem,
        names: &[String],
        event_names: &[(usize, Vec<String>)],
    ) -> Vec<String> {
        let version = self.header.version;
        let mut read = super::rinex2_system_obs_codes(system, names, version);
        for (_, event) in event_names {
            super::extend_code_union(
                &mut read,
                &super::rinex2_system_obs_codes(system, event, version),
            );
        }
        read
    }

    /// Header names for a version 2 product whose events declare type names:
    /// names reading as the lists the file header declares, found as
    /// `rinex2_names_for` finds them, whose reading together with the events'
    /// names is each stated constellation's union.
    fn rinex2_event_header_names(
        &self,
        event_names: &[(usize, Vec<String>)],
    ) -> Result<Vec<String>, RinexObsWriteError> {
        let stated = self.rinex2_stated_lists_from(&self.header.declared_obs_codes)?;
        let names = self.rinex2_names_for(&stated)?;
        for (system, held) in &self.header.obs_codes {
            if stated.contains_key(system)
                && self.rinex2_union_read(*system, &names, event_names) != *held
            {
                return Err(RinexObsWriteError::CodeListsNotUnion { system: *system });
            }
        }
        Ok(names)
    }

    /// Refuse a version 3 product whose `obs_codes` is not the union of the
    /// lists its file header and events declare.
    fn check_code_union(
        &self,
        timeline: &super::ObsHeaderTimeline,
    ) -> Result<(), RinexObsWriteError> {
        let mut union: BTreeMap<GnssSystem, Vec<String>> = BTreeMap::new();
        for (_, header) in timeline.segments() {
            for (system, list) in &header.declared_obs_codes {
                super::extend_code_union(union.entry(*system).or_default(), list);
            }
        }
        for system in union.keys().chain(self.header.obs_codes.keys()) {
            if union.get(system) != self.header.obs_codes.get(system) {
                return Err(RinexObsWriteError::CodeListsNotUnion { system: *system });
            }
        }
        Ok(())
    }

    /// Refuse `PRN / # OF OBS` counts past the codes the file header declares:
    /// the record holds a count for each, and no field for any other.
    fn check_counts_declared(&self) -> Result<(), RinexObsWriteError> {
        for (sat, counts) in &self.header.prn_obs_counts {
            let codes = self
                .header
                .declared_obs_codes
                .get(&sat.system)
                .map_or(0, Vec::len);
            if counts.len() > codes {
                return Err(RinexObsWriteError::CountsWithoutCodes {
                    satellite: *sat,
                    codes,
                    counts: counts.len(),
                });
            }
        }
        Ok(())
    }

    /// Refuse a value past its constellation's union, and a value under a code
    /// the list in effect at its epoch does not declare: the epoch's records are
    /// written by that list.
    fn check_values_declared(
        &self,
        layouts: &mut EpochLayouts<'_>,
    ) -> Result<(), RinexObsWriteError> {
        for (epoch_index, epoch) in self.epochs.iter().enumerate() {
            for (sat, values) in epoch_records(epoch) {
                let union = layouts.union_of(sat.system);
                if values.len() > union.len() {
                    return Err(RinexObsWriteError::ValuesWithoutCodes {
                        epoch_index,
                        satellite: *sat,
                        codes: union.len(),
                        values: values.len(),
                    });
                }
                let stretch = layouts.prepare(epoch_index, sat.system);
                let Some(layout) = layouts.get(stretch, sat.system) else {
                    return Err(RinexObsWriteError::ValueOutsideDeclaredList {
                        epoch_index,
                        satellite: *sat,
                        code: None,
                    });
                };
                if layout.identity {
                    continue;
                }
                for (position, value) in values.iter().enumerate() {
                    if !is_blank(value) && !layout.positions.contains(&Some(position)) {
                        return Err(RinexObsWriteError::ValueOutsideDeclaredList {
                            epoch_index,
                            satellite: *sat,
                            code: union.get(position).cloned(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Constellations a `PRN / # OF OBS` count or an observation names that
    /// hold no code list.
    fn rinex2_unlisted_systems(&self) -> std::collections::BTreeSet<GnssSystem> {
        let mut systems: std::collections::BTreeSet<GnssSystem> = self
            .header
            .prn_obs_counts
            .iter()
            .filter(|(_, counts)| !counts.is_empty())
            .map(|(sat, _)| sat.system)
            .collect();
        systems.extend(
            self.epochs
                .iter()
                .flat_map(|epoch| epoch_record_satellites(epoch).map(|sat| sat.system)),
        );
        systems.retain(|system| !self.header.obs_codes.contains_key(system));
        systems
    }

    /// Constellations holding a code list a version 2 file of this product
    /// written with `names` does not state: no observation or count names them,
    /// a file with observations, or whose version record names another
    /// constellation, builds no list for them, and the names do not read as
    /// their list either. A list the names read as is still in the file, as a
    /// list read from a version 2 file whose body a repair emptied is.
    fn rinex2_unstated_systems(&self, names: &[String]) -> std::collections::BTreeSet<GnssSystem> {
        let events = self.rinex2_event_names();
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
                    && self.rinex2_union_read(**system, names, &events) != **codes
            })
            .map(|(system, _)| *system)
            .collect()
    }

    /// Remove, and report, the lists a version 2 file of this product written
    /// with `names` does not state.
    fn remove_unstated_lists(&mut self, names: &[String], changes: &mut Vec<ObsDowngradeChange>) {
        for system in self.rinex2_unstated_systems(names) {
            if let Some(codes) = self.header.obs_codes.remove(&system) {
                changes.push(ObsDowngradeChange::CodeListRemoved { system, codes });
            }
        }
    }

    /// The lists a version 2 layout of this product places: the ones it holds
    /// for constellations an observation or a count names, and with no
    /// observations, the fallback constellation's. A list nothing names holds
    /// nothing a file states, so it does not take columns or move codes.
    fn rinex2_layout_lists(&self) -> BTreeMap<GnssSystem, Vec<String>> {
        let mut systems = self.rinex2_observed_systems();
        if systems.is_empty() {
            systems.insert(self.rinex2_fallback_system());
        }
        systems.extend(
            self.header
                .prn_obs_counts
                .iter()
                .filter(|(_, counts)| !counts.is_empty())
                .map(|(sat, _)| sat.system),
        );
        self.header
            .obs_codes
            .iter()
            .filter(|(system, _)| systems.contains(system))
            .map(|(system, codes)| (*system, codes.clone()))
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
        self.rinex2_stated_lists_from(&self.header.obs_codes)
    }

    /// The lists [`Self::rinex2_stated_lists`] gives, taken from `lists`: the
    /// product's own, or the lists its file header declares.
    fn rinex2_stated_lists_from(
        &self,
        source: &BTreeMap<GnssSystem, Vec<String>>,
    ) -> Result<BTreeMap<GnssSystem, Vec<String>>, RinexObsWriteError> {
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
            if let Some(codes) = source.get(&system) {
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
                if let Some((sat, values)) = epoch
                    .sats
                    .iter()
                    .chain(&epoch.cycle_slips)
                    .find(|(sat, _)| sat.system == system)
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

    /// Leave the lists a downgrade made explicit for constellations only a
    /// count names implied again, and keep the names that state them.
    fn leave_implied(
        &mut self,
        implied: &std::collections::BTreeSet<GnssSystem>,
        names: Vec<String>,
    ) {
        for system in implied {
            self.header.obs_codes.remove(system);
        }
        self.header.rinex2_types = names;
    }

    /// The code lists a version 2 file of this product reads back as, as the
    /// reader builds them: what the type names read as for each constellation
    /// an observation is from, and with no observations, for the constellation
    /// the version record names. A constellation only a count names reads by
    /// the names without a list of its own, and those names are compared
    /// themselves. With no names, the lists are the product's own.
    fn rinex2_read_lists(&self) -> BTreeMap<GnssSystem, Vec<String>> {
        let names = &self.header.rinex2_types;
        let events = self.rinex2_event_names();
        if names.is_empty() && events.is_empty() {
            return self.header.obs_codes.clone();
        }
        let mut systems = self.rinex2_observed_systems();
        if systems.is_empty() {
            systems.insert(self.rinex2_fallback_system());
        }
        systems
            .into_iter()
            .map(|system| (system, self.rinex2_union_read(system, names, &events)))
            .collect()
    }

    /// The lists a reader of this version 2 product's file declares in its
    /// header, written with `names`: what the names read as for each
    /// constellation the file states a list for.
    fn rinex2_rebuilt_declared(&self, names: &[String]) -> BTreeMap<GnssSystem, Vec<String>> {
        let version = self.header.version;
        let mut systems = self.rinex2_observed_systems();
        if systems.is_empty() {
            systems.insert(self.rinex2_fallback_system());
        }
        systems
            .into_iter()
            .map(|system| {
                (
                    system,
                    super::rinex2_system_obs_codes(system, names, version),
                )
            })
            .collect()
    }

    /// The declared lists of a version 2 product a downgrade leaves: its lists,
    /// with each list its file states what its type names read as, which is
    /// the list a reader rebuilds, including the list of the constellation a
    /// file with no observations names.
    fn rinex2_declared_after_downgrade(&self) -> BTreeMap<GnssSystem, Vec<String>> {
        let mut declared = self.header.obs_codes.clone();
        declared.extend(self.rinex2_rebuilt_declared(&self.header.rinex2_types));
        declared
    }

    /// Refuse a version 2 product whose `declared_obs_codes` a reader would not
    /// rebuild from the file written with `names`. A version 2 header declares
    /// type names, not lists: each list the file states is what the names read
    /// as for its constellation, and a list the file does not state is held
    /// only where the names read as it, as for `obs_codes`.
    fn check_rinex2_declared(&self, names: &[String]) -> Result<(), RinexObsWriteError> {
        let version = self.header.version;
        let rebuilt = self.rinex2_rebuilt_declared(names);
        for (system, list) in &rebuilt {
            if self.header.declared_obs_codes.get(system) != Some(list) {
                return Err(RinexObsWriteError::DeclaredListNotStated { system: *system });
            }
        }
        for (system, list) in &self.header.declared_obs_codes {
            if !rebuilt.contains_key(system)
                && super::rinex2_system_obs_codes(*system, names, version) != *list
            {
                return Err(RinexObsWriteError::DeclaredListNotStated { system: *system });
            }
        }
        Ok(())
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
            epoch.declared_record_count = declared_count(epoch);
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
            // A reader rebuilds `declared_obs_codes` from the type names for the
            // lists the file states; the writer has checked the product's own
            // against the same, so both sides are compared as a reader holds them.
            product.header.declared_obs_codes =
                product.rinex2_rebuilt_declared(&product.header.rinex2_types.clone());
        } else {
            product.header.rinex2_types.clear();
            product.header.rinex2_system = None;
        }
        product
    }
}

/// How many steps the layout order search takes before it uses the best order
/// found: each candidate the assignment method examines, each cell of a cost
/// matrix it builds, each column step local improvement weighs, each column it
/// re-reads to check an order, and the sorting that repairs one.
pub(crate) const LAYOUT_ORDER_SEARCH_WORK: u64 = 200_000_000;

/// How many of the best distinct orders the search has seen are kept, and local
/// improvement starts from when the search could not prove one best.
const LAYOUT_ORDER_IMPROVEMENT_STARTS: usize = 16;

#[cfg(test)]
thread_local! {
    /// The assignment problems the last order search on this thread solved,
    /// whether it ended without proving its order moves the fewest codes, and
    /// how many steps of [`LAYOUT_ORDER_SEARCH_WORK`] it took.
    pub(crate) static LAST_ORDER_SEARCH: std::cell::Cell<(usize, bool, u64)> =
        const { std::cell::Cell::new((0, false, 0)) };
}

/// One subproblem of the order search: the positions each column may take, and
/// for each group with no held first column, which column was chosen to come
/// before the group's held ones.
#[derive(Clone, Default)]
struct OrderNode {
    lowest: Vec<usize>,
    highest: Vec<usize>,
    chosen: Vec<Option<usize>>,
}

/// A reading rule an assignment breaks, and so how its subproblem splits.
enum OrderBranch {
    /// The first column sits after the second, which it has to precede.
    Before(usize, usize),
    /// Every held column of this group precedes all its unheld ones.
    Choose(usize),
}

/// The work allowed for the order search ran out.
struct WorkSpent;

/// The search for the column order that moves the fewest codes, see
/// `RinexObs::ordered_by_original_positions`. Its matrices are square in the
/// header's width, which the layout holds to the 999 types a header declares
/// before searching.
struct OrderSearch {
    width: usize,
    /// Per constellation, per column: the group of names reading as one code
    /// the column belongs to, and whether it has to be the group's first
    /// column. `None` for a name the constellation reads as written.
    groups: Vec<Vec<Option<(usize, bool)>>>,
    /// Per constellation, per column: whether a code sits there.
    occupied: Vec<Vec<bool>>,
    /// Per group, the held column that has to come first, when one does.
    leader: Vec<Option<usize>>,
    group_count: usize,
    /// Per column, per position: how many codes the column keeps in place there.
    gains: Vec<Vec<i64>>,
    /// Per column, the positions it keeps a code in place at.
    keeps: Vec<Vec<usize>>,
    /// Pairs of columns the first of which has to precede the second: a group's
    /// held first column before every other column of the group.
    precedences: Vec<(usize, usize)>,
    /// Groups with held columns reading as written and no held first column:
    /// some unheld column of the group has to precede every held one.
    choices: Vec<(Vec<usize>, Vec<usize>)>,
    /// The order keeping every reading rule with the most codes in place seen
    /// so far, and how many it keeps.
    best: Vec<usize>,
    best_kept: i64,
    /// The best distinct orders keeping every reading rule seen so far, by how
    /// many codes they keep in place: local improvement's starting points.
    pool: std::collections::BTreeSet<(i64, Vec<usize>)>,
    solves: usize,
    /// Steps left, and how many of them the current phase has to leave.
    work_left: u64,
    work_floor: u64,
    exhausted: bool,
}

impl OrderSearch {
    fn new(
        readings: Vec<Vec<Option<(String, bool)>>>,
        rows: &[&Vec<Option<usize>>],
        names: &[String],
    ) -> Self {
        let width = names.len();
        let mut group_ids: BTreeMap<(usize, String), usize> = BTreeMap::new();
        let mut leader: Vec<Option<usize>> = Vec::new();
        let mut members: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut groups = Vec::with_capacity(readings.len());
        for (constellation, reading) in readings.into_iter().enumerate() {
            let mut row_groups = Vec::with_capacity(width);
            for (column, entry) in reading.into_iter().enumerate() {
                row_groups.push(entry.map(|(canonical, first)| {
                    let next = group_ids.len();
                    let id = *group_ids.entry((constellation, canonical)).or_insert(next);
                    if id == leader.len() {
                        leader.push(None);
                        members.push((constellation, Vec::new()));
                    }
                    members[id].1.push(column);
                    if first && rows[constellation][column].is_some() {
                        leader[id] = Some(column);
                    }
                    (id, first)
                }));
            }
            groups.push(row_groups);
        }
        let occupied: Vec<Vec<bool>> = rows
            .iter()
            .map(|row| row.iter().map(Option::is_some).collect())
            .collect();
        let mut gains = vec![vec![0_i64; width]; width];
        let mut keeps = vec![Vec::new(); width];
        for row in rows {
            for (column, slot) in row.iter().enumerate() {
                if let Some(index) = slot.filter(|&index| index < width) {
                    if gains[column][index] == 0 {
                        keeps[column].push(index);
                    }
                    gains[column][index] += 1;
                }
            }
        }
        let mut precedences = std::collections::BTreeSet::new();
        let mut choices = Vec::new();
        for (id, (constellation, columns)) in members.iter().enumerate() {
            match leader[id] {
                Some(first) => {
                    for &then in columns.iter().filter(|&&then| then != first) {
                        precedences.insert((first, then));
                    }
                }
                None => {
                    let (held, free): (Vec<usize>, Vec<usize>) = columns
                        .iter()
                        .partition(|&&column| occupied[*constellation][column]);
                    if !held.is_empty() {
                        choices.push((free, held));
                    }
                }
            }
        }
        Self {
            width,
            groups,
            occupied,
            group_count: leader.len(),
            leader,
            gains,
            keeps,
            precedences: precedences.into_iter().collect(),
            choices,
            best: (0..width).collect(),
            best_kept: 0,
            pool: std::collections::BTreeSet::new(),
            solves: 0,
            work_left: LAYOUT_ORDER_SEARCH_WORK,
            work_floor: 0,
            exhausted: false,
        }
    }

    /// Take `steps` from the work left, or report that the phase has used all
    /// it may.
    fn charge(&mut self, steps: u64) -> bool {
        if self.work_left < self.work_floor.saturating_add(steps) {
            self.exhausted = true;
            return false;
        }
        self.work_left -= steps;
        true
    }

    /// The steps repairing an order takes: two sorts of the columns under every
    /// reading rule.
    fn repair_steps(&self) -> u64 {
        let width = self.width as u64;
        let rules = self.precedences.len()
            + self
                .choices
                .iter()
                .map(|(free, held)| free.len() + held.len())
                .sum::<usize>();
        2 * (width * (1 + u64::from(64 - width.leading_zeros())) + rules as u64)
    }

    fn kept_in_place(&self, order: &[usize]) -> i64 {
        order
            .iter()
            .enumerate()
            .map(|(position, &column)| self.gains[column][position])
            .sum()
    }

    /// Whether `column` can go next, given which groups have begun.
    fn fits(&self, column: usize, begun: &[bool], placed: &[bool]) -> bool {
        self.groups.iter().zip(&self.occupied).all(|(row, filled)| {
            let Some((group, first)) = row[column] else {
                return true;
            };
            if let Some(leader) = self.leader[group] {
                if leader != column && !placed[leader] {
                    return false;
                }
            }
            !filled[column] || first != begun[group]
        })
    }

    /// Whether every constellation reads every placed code as built in `order`.
    /// Without the work to check, it is not taken to.
    fn reads_as_built(&mut self, order: &[usize]) -> bool {
        self.charge((self.width * (1 + self.groups.len())) as u64) && self.keeps_every_rule(order)
    }

    /// Whether every constellation reads every placed code as built in `order`,
    /// whatever work is left.
    fn keeps_every_rule(&self, order: &[usize]) -> bool {
        let mut placed = vec![false; self.width];
        let mut begun = vec![false; self.group_count];
        for &column in order {
            if !self.fits(column, &begun, &placed) {
                return false;
            }
            for row in &self.groups {
                if let Some((group, _)) = row[column] {
                    begun[group] = true;
                }
            }
            placed[column] = true;
        }
        true
    }

    /// The order keeping the most codes in place with each column inside its
    /// allowed positions and the reading rule set aside. `Ok(None)` when the
    /// allowed positions admit no order, `Err` when the work runs out first.
    fn solve(&mut self, node: &OrderNode) -> Result<Option<(i64, Vec<usize>)>, WorkSpent> {
        let width = self.width;
        if !self.charge((width * width) as u64) {
            return Err(WorkSpent);
        }
        self.solves += 1;
        // More than every code kept in place together, so an order using a
        // disallowed position never costs less than one that does not.
        let disallowed = 8 * width as i64 + 1;
        let cost: Vec<Vec<i64>> = (0..width)
            .map(|column| {
                (0..width)
                    .map(|position| {
                        if (node.lowest[column]..=node.highest[column]).contains(&position) {
                            -self.gains[column][position]
                        } else {
                            disallowed
                        }
                    })
                    .collect()
            })
            .collect();
        let mut steps = self.work_left - self.work_floor;
        let assignment = least_cost_assignment(&cost, &mut steps);
        self.work_left = self.work_floor + steps;
        let Some(assignment) = assignment else {
            self.exhausted = true;
            return Err(WorkSpent);
        };
        let mut order = vec![0_usize; width];
        for (column, position) in assignment.into_iter().enumerate() {
            if !(node.lowest[column]..=node.highest[column]).contains(&position) {
                return Ok(None);
            }
            order[position] = column;
        }
        Ok(Some((self.kept_in_place(&order), order)))
    }

    /// Orders keeping every reading rule built from `order`: columns taken in
    /// its sequence, each once the columns it has to follow are placed, either
    /// as they come or with each column pulled to just before the earliest
    /// column that has to follow it. Empty when the rules, with `node`'s
    /// choices and each other group begun by its earliest unheld column, admit
    /// no order.
    fn repaired(&self, node: &OrderNode, order: &[usize]) -> Vec<Vec<usize>> {
        let width = self.width;
        let mut position = vec![0_usize; width];
        for (at, &column) in order.iter().enumerate() {
            position[column] = at;
        }
        let mut followers: Vec<Vec<usize>> = vec![Vec::new(); width];
        for &(first, then) in &self.precedences {
            followers[first].push(then);
        }
        for (index, (free, held)) in self.choices.iter().enumerate() {
            let first = node.chosen[index]
                .or_else(|| free.iter().copied().min_by_key(|&column| position[column]));
            if let Some(first) = first {
                for &then in held {
                    followers[first].push(then);
                }
            }
        }
        let sorted = |key: &[usize]| -> Vec<usize> {
            let mut waiting = vec![0_usize; width];
            for &next in followers.iter().flatten() {
                waiting[next] += 1;
            }
            let mut ready: std::collections::BTreeSet<(usize, usize, usize)> = (0..width)
                .filter(|&column| waiting[column] == 0)
                .map(|column| (key[column], position[column], column))
                .collect();
            let mut out = Vec::with_capacity(width);
            while let Some((_, _, column)) = ready.pop_first() {
                out.push(column);
                for &next in &followers[column] {
                    waiting[next] -= 1;
                    if waiting[next] == 0 {
                        ready.insert((key[next], position[next], next));
                    }
                }
            }
            out
        };
        let as_they_come = sorted(&position);
        if as_they_come.len() < width {
            return Vec::new();
        }
        let mut pulled = position.clone();
        for &column in as_they_come.iter().rev() {
            for &next in &followers[column] {
                pulled[column] = pulled[column].min(pulled[next]);
            }
        }
        let pulled_early = sorted(&pulled);
        vec![as_they_come, pulled_early]
    }

    /// Keep an order keeping every reading rule among the best distinct ones seen.
    fn remember(&mut self, kept: i64, order: Vec<usize>) {
        self.pool.insert((kept, order));
        if self.pool.len() > LAYOUT_ORDER_IMPROVEMENT_STARTS {
            self.pool.pop_first();
        }
    }

    /// Keep `order`, or an order repairing it into one keeping every reading
    /// rule, when it keeps more codes in place than the best order so far, and
    /// remember the best ones keeping the rules as starts for local improvement.
    fn offer(&mut self, node: &OrderNode, order: &[usize]) {
        if !self.charge(self.repair_steps()) {
            return;
        }
        let mut candidates = vec![order.to_vec()];
        candidates.extend(self.repaired(node, order));
        for candidate in candidates {
            if candidate.len() != self.width || !self.charge(self.width as u64) {
                continue;
            }
            let kept = self.kept_in_place(&candidate);
            if self.pool.contains(&(kept, candidate.clone())) || !self.reads_as_built(&candidate) {
                continue;
            }
            if kept > self.best_kept {
                self.best_kept = kept;
                self.best = candidate.clone();
            }
            self.remember(kept, candidate);
        }
    }

    /// The order reached from `order` by steps that keep more codes in place and
    /// keep every reading rule, directly or once repaired: one column moved to
    /// another position, two swapped, or three moved around a cycle in which
    /// one goes to a position it keeps a code at. Every step moving three
    /// columns that keeps more codes in place moves one of them to such a
    /// position, so the cycles are all of them. Passes over every step continue
    /// past each one taken, and end when a whole pass takes none or the work
    /// runs out; each step taken keeps more codes in place, so the passes end.
    fn improved(&mut self, mut order: Vec<usize>) -> (i64, Vec<usize>) {
        let width = self.width;
        let mut kept = self.kept_in_place(&order);
        let free = OrderNode {
            lowest: Vec::new(),
            highest: Vec::new(),
            chosen: vec![None; self.choices.len()],
        };
        let mut stepped = true;
        while stepped {
            stepped = false;
            for a in 0..width {
                if !self.charge(width as u64) {
                    return (kept, order);
                }
                for b in a + 1..width {
                    let (first, second) = (order[a], order[b]);
                    let gain = self.gains[first][b] + self.gains[second][a]
                        - self.gains[first][a]
                        - self.gains[second][b];
                    if gain > 0 {
                        order.swap(a, b);
                        match self.accepted(&free, &order, kept, gain) {
                            Some(better) => {
                                (kept, order) = better;
                                stepped = true;
                            }
                            None => order.swap(a, b),
                        }
                    }
                }
            }
            for a in 0..width {
                if !self.charge(width as u64) {
                    return (kept, order);
                }
                // Moving the column from `a` to `b` shifts every column between
                // them one place toward `a`; the running sum carries that. A
                // step taken changes the order the sum was built on, so the
                // pass goes on to the next position.
                let column = order[a];
                let mut shifted = 0_i64;
                let mut taken = false;
                for b in a + 1..width {
                    shifted += self.gains[order[b]][b - 1] - self.gains[order[b]][b];
                    let gain = shifted + self.gains[column][b] - self.gains[column][a];
                    if gain > 0 {
                        let moved = order.remove(a);
                        order.insert(b, moved);
                        if let Some(better) = self.accepted(&free, &order, kept, gain) {
                            (kept, order) = better;
                            taken = true;
                            break;
                        }
                        let moved = order.remove(b);
                        order.insert(a, moved);
                    }
                }
                if taken {
                    stepped = true;
                    continue;
                }
                let mut shifted = 0_i64;
                for b in (0..a).rev() {
                    shifted += self.gains[order[b]][b + 1] - self.gains[order[b]][b];
                    let gain = shifted + self.gains[column][b] - self.gains[column][a];
                    if gain > 0 {
                        let moved = order.remove(a);
                        order.insert(b, moved);
                        if let Some(better) = self.accepted(&free, &order, kept, gain) {
                            (kept, order) = better;
                            stepped = true;
                            break;
                        }
                        let moved = order.remove(b);
                        order.insert(a, moved);
                    }
                }
            }
            'cycles: for a in 0..width {
                let column = order[a];
                let targets: Vec<usize> = self.keeps[column]
                    .iter()
                    .copied()
                    .filter(|&target| target != a)
                    .collect();
                if !self.charge((width * (1 + targets.len())) as u64) {
                    return (kept, order);
                }
                for p in targets {
                    let displaced = order[p];
                    for q in 0..width {
                        if q == a || q == p {
                            continue;
                        }
                        let third = order[q];
                        let gain =
                            self.gains[column][p] + self.gains[displaced][q] + self.gains[third][a]
                                - self.gains[column][a]
                                - self.gains[displaced][p]
                                - self.gains[third][q];
                        if gain > 0 {
                            order[p] = column;
                            order[q] = displaced;
                            order[a] = third;
                            if let Some(better) = self.accepted(&free, &order, kept, gain) {
                                (kept, order) = better;
                                stepped = true;
                                continue 'cycles;
                            }
                            order[a] = column;
                            order[p] = displaced;
                            order[q] = third;
                        }
                    }
                }
            }
        }
        (kept, order)
    }

    /// The order a step from an order keeping `kept` codes in place to `stepped`,
    /// gaining `gain`, leads to: `stepped` itself when it keeps every reading
    /// rule, otherwise the best of its repairs keeping more than `kept`.
    fn accepted(
        &mut self,
        free: &OrderNode,
        stepped: &[usize],
        kept: i64,
        gain: i64,
    ) -> Option<(i64, Vec<usize>)> {
        if self.reads_as_built(stepped) {
            return Some((kept + gain, stepped.to_vec()));
        }
        if !self.charge(self.repair_steps()) {
            return None;
        }
        let mut best: Option<(i64, Vec<usize>)> = None;
        for repair in self.repaired(free, stepped) {
            if !self.charge(self.width as u64) {
                break;
            }
            let repaired_kept = self.kept_in_place(&repair);
            if repaired_kept > best.as_ref().map_or(kept, |(held, _)| *held)
                && self.reads_as_built(&repair)
            {
                best = Some((repaired_kept, repair));
            }
        }
        best
    }

    /// The first reading rule `order` breaks under `node`'s choices.
    fn broken_rule(&self, node: &OrderNode, order: &[usize]) -> Option<OrderBranch> {
        let mut position = vec![0_usize; self.width];
        for (at, &column) in order.iter().enumerate() {
            position[column] = at;
        }
        if let Some(&(first, then)) = self
            .precedences
            .iter()
            .find(|(first, then)| position[*first] > position[*then])
        {
            return Some(OrderBranch::Before(first, then));
        }
        for (index, (free, held)) in self.choices.iter().enumerate() {
            match node.chosen[index] {
                Some(first) => {
                    if let Some(&then) = held.iter().find(|&&then| position[first] > position[then])
                    {
                        return Some(OrderBranch::Before(first, then));
                    }
                }
                None => {
                    let earliest_held = held.iter().map(|&column| position[column]).min();
                    let earliest_free = free.iter().map(|&column| position[column]).min();
                    if earliest_free
                        .is_none_or(|free| earliest_held.is_some_and(|held| free > held))
                    {
                        return Some(OrderBranch::Choose(index));
                    }
                }
            }
        }
        None
    }

    /// Subproblems that together hold every order `node` holds that keeps the
    /// broken rule, none of them holding `order`.
    fn split(&self, node: &OrderNode, order: &[usize], branch: OrderBranch) -> Vec<OrderNode> {
        match branch {
            OrderBranch::Before(first, then) => {
                let at = order.iter().position(|&column| column == then).unwrap_or(0);
                let mut children = Vec::with_capacity(2);
                // Either the first column goes before where the second sits...
                if at > 0 {
                    let mut child = node.clone();
                    child.highest[first] = child.highest[first].min(at - 1);
                    if child.lowest[first] <= child.highest[first] {
                        children.push(child);
                    }
                }
                // ...or it goes there or later, and the second after that.
                let mut child = node.clone();
                child.lowest[then] = child.lowest[then].max(at + 1);
                child.lowest[first] = child.lowest[first].max(at);
                if child.lowest[then] <= child.highest[then]
                    && child.lowest[first] <= child.highest[first]
                {
                    children.push(child);
                }
                children
            }
            OrderBranch::Choose(index) => self.choices[index]
                .0
                .iter()
                .map(|&first| {
                    let mut child = node.clone();
                    child.chosen[index] = Some(first);
                    child
                })
                .collect(),
        }
    }

    fn run(mut self) -> Vec<usize> {
        let width = self.width;
        // The seed and bound below scan every column at every position once.
        self.charge((2 * width * width) as u64);
        let built: Vec<usize> = (0..width).collect();
        self.best_kept = self.kept_in_place(&built);
        self.best = built.clone();
        self.remember(self.best_kept, built);
        // Columns sorted by the earliest position they keep a code at, and that
        // order's repairs, start the search off with an order near the one a
        // wide header wants, which the search may not have the work to solve.
        let earliest: Vec<usize> = (0..width)
            .map(|column| self.keeps[column].iter().copied().min().unwrap_or(width))
            .collect();
        let mut by_earliest: Vec<usize> = (0..width).collect();
        by_earliest.sort_by_key(|&column| (earliest[column], column));
        let free = OrderNode {
            lowest: Vec::new(),
            highest: Vec::new(),
            chosen: vec![None; self.choices.len()],
        };
        self.offer(&free, &by_earliest);
        // Every position's best column bounds what any order keeps.
        let ceiling: i64 = (0..width)
            .map(|position| {
                (0..width)
                    .map(|column| self.gains[column][position])
                    .max()
                    .unwrap_or(0)
            })
            .sum();
        if self.best_kept < ceiling {
            // The search may take half the work left; improvement has the rest.
            self.work_floor = self.work_left / 2;
            let proved = self.search();
            self.work_floor = 0;
            self.exhausted = !proved;
            if !proved {
                let starts: Vec<Vec<usize>> = std::mem::take(&mut self.pool)
                    .into_iter()
                    .rev()
                    .map(|(_, order)| order)
                    .collect();
                for start in starts {
                    let (kept, order) = self.improved(start);
                    if kept > self.best_kept {
                        self.best_kept = kept;
                        self.best = order;
                    }
                }
                self.exhausted = true;
            }
        }
        #[cfg(test)]
        LAST_ORDER_SEARCH.with(|last| {
            last.set((
                self.solves,
                self.exhausted,
                LAYOUT_ORDER_SEARCH_WORK - self.work_left,
            ));
        });
        self.best
    }

    /// Branch and bound down to the work floor, returning whether it proved the
    /// best order it found keeps the most codes in place any order does.
    fn search(&mut self) -> bool {
        let root = OrderNode {
            lowest: vec![0; self.width],
            highest: vec![self.width.saturating_sub(1); self.width],
            chosen: vec![None; self.choices.len()],
        };
        // Subproblems by the most codes they could keep, the latest first among
        // equals. Every assignment solved is offered as it is and repaired, so
        // the best order seen is kept however the search ends; once no open
        // subproblem could keep more than it, it is the order keeping the most
        // codes in place.
        let mut open: Vec<(OrderNode, Vec<usize>)> = Vec::new();
        let mut queue = std::collections::BinaryHeap::new();
        match self.solve(&root) {
            Err(WorkSpent) => return false,
            Ok(None) => return true,
            Ok(Some((kept, order))) => {
                self.offer(&root, &order);
                if kept > self.best_kept {
                    open.push((root, order));
                    queue.push((kept, 0_usize));
                }
            }
        }
        while let Some((bound, at)) = queue.pop() {
            if bound <= self.best_kept {
                return true;
            }
            if !self.charge((self.width + self.precedences.len() + self.choices.len()) as u64) {
                return false;
            }
            let (node, order) = std::mem::take(&mut open[at]);
            let Some(branch) = self.broken_rule(&node, &order) else {
                // An assignment keeping every rule, with the most codes any open
                // subproblem could keep. Offering it when it was solved may have
                // run out of work before checking it, and dropping it then would
                // let an empty queue claim a worse order best, so it is checked
                // here whatever work is left: once per such subproblem, and the
                // search ends at the next one popped.
                if bound > self.best_kept && self.keeps_every_rule(&order) {
                    self.best_kept = bound;
                    self.best = order.clone();
                    self.remember(bound, order);
                }
                continue;
            };
            for child in self.split(&node, &order, branch) {
                match self.solve(&child) {
                    Err(WorkSpent) => return false,
                    Ok(None) => {}
                    Ok(Some((kept, child_order))) => {
                        self.offer(&child, &child_order);
                        if kept > self.best_kept {
                            open.push((child, child_order));
                            queue.push((kept, open.len() - 1));
                        }
                    }
                }
            }
        }
        true
    }
}

/// The assignment of rows to columns of a square matrix costing the least in
/// total, by the Hungarian method. Returns, for each row, its column, or `None`
/// when examining candidates would take more than `steps`, which counts down
/// by each one examined.
#[allow(clippy::needless_range_loop)] // the method walks parallel one-based arrays by index
fn least_cost_assignment(cost: &[Vec<i64>], steps: &mut u64) -> Option<Vec<usize>> {
    let n = cost.len();
    let infinity = i64::MAX / 4;
    let mut u = vec![0_i64; n + 1];
    let mut v = vec![0_i64; n + 1];
    let mut assigned_row = vec![0_usize; n + 1];
    let mut way = vec![0_usize; n + 1];
    for row in 1..=n {
        assigned_row[0] = row;
        let mut column = 0_usize;
        let mut slack = vec![infinity; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            *steps = steps.checked_sub(n as u64 + 1)?;
            used[column] = true;
            let current_row = assigned_row[column];
            let mut delta = infinity;
            let mut next = 0_usize;
            for candidate in 1..=n {
                if used[candidate] {
                    continue;
                }
                let reduced = cost[current_row - 1][candidate - 1] - u[current_row] - v[candidate];
                if reduced < slack[candidate] {
                    slack[candidate] = reduced;
                    way[candidate] = column;
                }
                if slack[candidate] < delta {
                    delta = slack[candidate];
                    next = candidate;
                }
            }
            for candidate in 0..=n {
                if used[candidate] {
                    u[assigned_row[candidate]] += delta;
                    v[candidate] -= delta;
                } else {
                    slack[candidate] -= delta;
                }
            }
            column = next;
            if assigned_row[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            assigned_row[column] = assigned_row[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    let mut assignment = vec![0_usize; n];
    for column in 1..=n {
        if assigned_row[column] != 0 {
            assignment[assigned_row[column] - 1] = column - 1;
        }
    }
    Some(assignment)
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
        declared_obs_codes,
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
        for (what, records_before, records_after) in [
            ("satellites", &before.sats, &after.sats),
            (
                "cycle slip satellites",
                &before.cycle_slips,
                &after.cycle_slips,
            ),
        ] {
            if let Some(difference) =
                describe_records_difference(index, what, records_before, records_after)
            {
                return difference;
            }
        }
    }
    "the product reads back differently".to_string()
}

/// The first difference between an epoch's observation or cycle slip records
/// before and after reading back, named with `what` they are.
fn describe_records_difference(
    index: usize,
    what: &str,
    before: &SatRecords,
    after: &SatRecords,
) -> Option<String> {
    const LIMIT: usize = 400;
    let clipped = |text: String| {
        if text.len() <= LIMIT {
            return text;
        }
        let mut end = LIMIT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &text[..end])
    };
    let satellites_before: Vec<_> = before.keys().collect();
    let satellites_after: Vec<_> = after.keys().collect();
    if satellites_before != satellites_after {
        return Some(clipped(format!(
            "epoch {index} {what} {satellites_before:?} read back as {satellites_after:?}"
        )));
    }
    for (sat, values) in before {
        let Some(read) = after.get(sat) else {
            return Some(format!("epoch {index} {sat} does not read back"));
        };
        if values.len() != read.len() {
            return Some(format!(
                "epoch {index} {what} {sat} holds {} values and reads back with {}",
                values.len(),
                read.len()
            ));
        }
        if let Some(at) = values
            .iter()
            .zip(read)
            .position(|(value, back)| value != back)
        {
            return Some(clipped(format!(
                "epoch {index} {what} {sat} value {at}: {:?} reads back as {:?}",
                values[at], read[at]
            )));
        }
    }
    None
}

impl RinexObs {
    /// This product as one a version 2 file can state exactly, with every change
    /// that took.
    ///
    /// Version 2 names one list of observation types for every constellation,
    /// so each constellation's codes are laid out against that one list: a code
    /// keeps its column where a name reads back as it, and moves to a column of
    /// its own where one would read it as a different code. Values and
    /// `PRN / # OF OBS` counts move with their codes. What the list cannot keep
    /// is returned as a change rather than dropped: a code renamed to what its
    /// column reads as, a code moved to another position, a blank code added so
    /// the lists match, a code list for a constellation no observation or count
    /// names that the file would not state, scale factor
    /// records removed from values that are already physical, and picoseconds
    /// a version 2 epoch has no field for. A version 3 product with no
    /// observations is named for a constellation whose list no count keeps,
    /// GPS first, so the file keeps every list it can.
    ///
    /// Columns are ordered to move as few codes as any order of them can: the
    /// order keeping the most codes at the positions they held is found by
    /// branch and bound over assignments of columns to positions. When proving an order
    /// best would take more than a fixed amount of work (200 million steps of
    /// the search, a fraction of a second in a release build), the best orders
    /// it found are improved by moving or swapping columns one at a time and the
    /// best result is used; it still reads back exactly and every move it makes
    /// is reported, but a different order could move fewer.
    ///
    /// Values or `PRN / # OF OBS` counts past their constellation's codes name
    /// no observable and are refused rather than dropped.
    ///
    /// A receiver clock offset carrying more than the nine decimals a version 2
    /// epoch record holds is rounded and reported.
    ///
    /// A product read from RINEX 4.00 or later loses its `SYS / PHASE SHIFT` and
    /// `GLONASS COD/PHS/BIS` records, from the file header and from its events,
    /// each removal reported as [`ObsDowngradeChange::DeprecatedRecordsRemoved`].
    /// Version 4 declares them to be ignored, so the product applies none of
    /// them, and a version 2 file would.
    /// Values carrying more than the three decimals an observation field holds
    /// are rounded and reported. The result is returned only once
    /// [`RinexObs::to_rinex_string`] writes it, so it always reads back exactly;
    /// what version 2 still cannot state - more than 999 observation types, a
    /// year outside the 1980 to 2079 window, a value too wide for its field - is
    /// refused instead.
    ///
    /// # Errors
    ///
    /// [`RinexObsWriteError::NotVersionTwo`] when `version` is not a version 2,
    /// [`RinexObsWriteError::TooManyObservationTypes`] when the layout needs more
    /// than 999 types, and any error writing the result gives.
    pub fn downgrade_to_rinex2(
        &self,
        version: f64,
    ) -> Result<(RinexObs, Vec<ObsDowngradeChange>), RinexObsWriteError> {
        if !(2.0..3.0).contains(&version) {
            return Err(RinexObsWriteError::NotVersionTwo { version });
        }
        let mut product = self.clone();
        product.header.version = version;
        let mut changes = Vec::new();
        if !product.header.scale_factors.is_empty() {
            changes.push(ObsDowngradeChange::ScaleFactorsRemoved {
                count: product.header.scale_factors.len(),
            });
            product.header.scale_factors.clear();
        }
        if super::records_deprecated_in_rinex4(self.header.version) {
            remove_deprecated_records(&mut product, &mut changes);
        }
        for (epoch_index, epoch) in product.epochs.iter_mut().enumerate() {
            if let Some(picoseconds) = epoch.epoch_picoseconds.take() {
                changes.push(ObsDowngradeChange::EpochPicosecondsRemoved {
                    epoch_index,
                    picoseconds,
                });
            }
            if let Some(held) = epoch.rcv_clock_offset_s.filter(|held| held.is_finite()) {
                if let Ok(rounded) = format!("{held:.9}").parse::<f64>() {
                    if rounded != held {
                        // `-0.000000000` is a zero offset, not a negative one.
                        let to = if rounded == 0.0 { 0.0 } else { rounded };
                        changes.push(ObsDowngradeChange::ClockOffsetRounded {
                            epoch_index,
                            from: held,
                            to,
                        });
                        epoch.rcv_clock_offset_s = Some(to);
                    }
                }
            }
        }
        let timeline =
            self.header_timeline()
                .map_err(|error| RinexObsWriteError::EventRecordsUnreadable {
                    message: error.to_string(),
                })?;
        let list_events = self.list_event_indices();
        if list_events.is_empty() {
            for (epoch_index, epoch) in product.epochs.iter_mut().enumerate() {
                rewrite_event_records(self.header.version, epoch_index, epoch, None, &mut changes);
            }
            return self.downgrade_lists(product, version, changes);
        }
        self.downgrade_stretches(product, version, changes, &timeline, &list_events)
    }

    /// The epochs whose events declare code lists, or at version 2 type names,
    /// that take effect.
    fn list_event_indices(&self) -> Vec<usize> {
        let version = self.header.version;
        let label = if self.is_rinex2() {
            "# / TYPES OF OBSERV"
        } else {
            "SYS / # / OBS TYPES"
        };
        self.epochs
            .iter()
            .enumerate()
            .filter(|(_, epoch)| {
                super::applies_header_records(epoch.flag)
                    && super::event_declares_label(version, &epoch.special_records, label)
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// Downgrade a product whose events declare code lists or type names. Each
    /// stretch of epochs sharing the lists in effect is laid out as a product of
    /// its own, holding every epoch so that every constellation the file names
    /// is named, with the values of the other stretches blank. The file header
    /// is written with the first stretch's names, each event's type records with
    /// its stretch's, and the values are held under the union of what every
    /// stretch's names read as.
    fn downgrade_stretches(
        &self,
        product: RinexObs,
        version: f64,
        mut changes: Vec<ObsDowngradeChange>,
        timeline: &super::ObsHeaderTimeline,
        list_events: &[usize],
    ) -> Result<(RinexObs, Vec<ObsDowngradeChange>), RinexObsWriteError> {
        // A value the list in effect at its epoch does not declare has no
        // field in the stretch that epoch is laid out in.
        let event_names = self.rinex2_event_names();
        let mut layouts = EpochLayouts {
            product: self,
            timeline,
            header_names: self
                .is_rinex2()
                .then_some(self.header.rinex2_types.as_slice()),
            event_names: &event_names,
            cache: std::collections::HashMap::new(),
        };
        self.check_values_declared(&mut layouts)?;

        let source_version = self.header.version;
        let starts: Vec<usize> = core::iter::once(0)
            .chain(list_events.iter().copied())
            .collect();
        let mut results: Vec<RinexObs> = Vec::with_capacity(starts.len());
        for (stretch, &first) in starts.iter().enumerate() {
            let end = starts
                .get(stretch + 1)
                .copied()
                .unwrap_or(product.epochs.len());
            // The first stretch lays out the file header's own lists, which
            // stay the header's to declare even where an event before the
            // first epoch declares others.
            let in_effect = if stretch == 0 {
                &self.header
            } else {
                timeline.at(first)
            };
            let mut lists = in_effect.declared_obs_codes.clone();
            for system in self.header.obs_codes.keys() {
                lists.entry(*system).or_default();
            }
            let mut source = product.clone();
            source.header.version = source_version;
            source.header.obs_codes = lists.clone();
            source.header.declared_obs_codes = lists.clone();
            source.header.rinex2_types = if self.is_rinex2() {
                in_effect.rinex2_types.clone()
            } else {
                Vec::new()
            };
            if stretch > 0 {
                source.header.prn_obs_counts.clear();
            }
            for (index, epoch) in source.epochs.iter_mut().enumerate() {
                epoch.special_records.clear();
                epoch.declared_record_count = 0;
                let inside = (first..end).contains(&index);
                for (sat, values) in epoch.sats.iter_mut().chain(epoch.cycle_slips.iter_mut()) {
                    let list = lists
                        .get(&sat.system)
                        .map(Vec::as_slice)
                        .unwrap_or_default();
                    *values = if inside {
                        let union = layouts.union_of(sat.system);
                        placed_values(values, &super::union_positions(list, &union))
                    } else {
                        vec![BLANK_VALUE; list.len()]
                    };
                }
            }
            let mut target = source.clone();
            target.header.version = version;
            let (result, stretch_changes) = source.downgrade_lists(target, version, Vec::new())?;
            for change in stretch_changes {
                let per_list = matches!(
                    change,
                    ObsDowngradeChange::CodeRenamed { .. }
                        | ObsDowngradeChange::CodeMoved { .. }
                        | ObsDowngradeChange::CodeAdded { .. }
                        | ObsDowngradeChange::CodeListRemoved { .. }
                );
                // A list this stretch declares none for was held empty only so
                // the stretch names its constellation; removing it removes
                // nothing.
                if let ObsDowngradeChange::CodeListRemoved { system, codes } = &change {
                    if codes.is_empty() && !in_effect.declared_obs_codes.contains_key(system) {
                        continue;
                    }
                }
                changes.push(if per_list && stretch > 0 {
                    ObsDowngradeChange::InEventLists {
                        epoch_index: first,
                        change: Box::new(change),
                    }
                } else {
                    change
                });
            }
            results.push(result);
        }

        let Some(header_result) = results.first() else {
            return Err(RinexObsWriteError::ReadBackMismatch {
                what: "a downgrade laid out no stretch of epochs".to_string(),
            });
        };
        let mut assembled = product;
        assembled.header.rinex2_types = header_result.header.rinex2_types.clone();
        assembled.header.rinex2_system = header_result.header.rinex2_system;
        assembled.header.prn_obs_counts = header_result.header.prn_obs_counts.clone();
        let mut union: BTreeMap<GnssSystem, Vec<String>> = BTreeMap::new();
        for result in &results {
            for (system, list) in &result.header.obs_codes {
                super::extend_code_union(union.entry(*system).or_default(), list);
            }
        }
        for (index, epoch) in assembled.epochs.iter_mut().enumerate() {
            let stretch = starts
                .partition_point(|first| *first <= index)
                .saturating_sub(1);
            let Some(result) = results.get(stretch) else {
                continue;
            };
            let Some(laid_epoch) = result.epochs.get(index) else {
                continue;
            };
            for (records, laid_records) in [
                (&mut epoch.sats, &laid_epoch.sats),
                (&mut epoch.cycle_slips, &laid_epoch.cycle_slips),
            ] {
                for (sat, values) in records.iter_mut() {
                    let (Some(laid), Some(held)) = (laid_records.get(sat), union.get(&sat.system))
                    else {
                        continue;
                    };
                    let list = result
                        .header
                        .obs_codes
                        .get(&sat.system)
                        .map(Vec::as_slice)
                        .unwrap_or_default();
                    let placed = super::values_in_union(
                        laid,
                        &super::union_positions(list, held),
                        held.len(),
                    )
                    .ok_or(RinexObsWriteError::ValuesWithoutCodes {
                        epoch_index: index,
                        satellite: *sat,
                        codes: list.len(),
                        values: laid.len(),
                    })?;
                    *values = placed;
                }
            }
        }
        let header_names = assembled.header.rinex2_types.clone();
        assembled.header.declared_obs_codes = union
            .keys()
            .map(|system| {
                (
                    *system,
                    super::rinex2_system_obs_codes(*system, &header_names, version),
                )
            })
            .collect();
        assembled.header.obs_codes = union;
        for (index, epoch) in assembled.epochs.iter_mut().enumerate() {
            let names = starts
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, first)| **first == index)
                .map(|(stretch, _)| stretch)
                .and_then(|stretch| results.get(stretch))
                .map(|result| result.header.rinex2_types.clone());
            // A version 2 event whose names are laid out unchanged keeps its
            // records as written.
            let names = names
                .filter(|names| !(self.is_rinex2() && *names == timeline.at(index).rinex2_types));
            rewrite_event_records(
                source_version,
                index,
                epoch,
                names.as_deref().map(obs_types_v2_lines),
                &mut changes,
            );
        }
        // The result is only returned once it writes.
        assembled.to_rinex_string()?;
        Ok((assembled, changes))
    }

    /// Lay a product's code lists out for version 2, as
    /// [`RinexObs::downgrade_to_rinex2`] describes, with `self` the product
    /// before the downgrade and `product` it with its version, scale factors,
    /// picoseconds and clock offsets already downgraded.
    fn downgrade_lists(
        &self,
        mut product: RinexObs,
        version: f64,
        mut changes: Vec<ObsDowngradeChange>,
    ) -> Result<(RinexObs, Vec<ObsDowngradeChange>), RinexObsWriteError> {
        const BLANK: ObsValue = BLANK_VALUE;
        // A version 2 product's type names are every constellation's, so a
        // constellation a count or an observation names that holds no list has
        // the codes those names read as at the product's own version. They are
        // laid out and reported like any list; a list only counts named is left
        // implied again once the names are chosen. A value or count past its
        // constellation's codes names no observable.
        let mut implied = std::collections::BTreeSet::new();
        // With no observations, the list a version 2 file states is the one its
        // version record names. That is taken from the source before any list
        // is added here, which would otherwise decide it again.
        let source_systems = self.rinex2_observed_systems();
        let source_observed = !source_systems.is_empty();
        let source_fallback = self.rinex2_fallback_system();
        if self.is_rinex2() {
            // A mixed header whose fallback is GPS already says so; any other
            // fallback is held, since added lists would decide it again.
            if !source_observed && source_fallback != GnssSystem::Gps {
                product.header.rinex2_system = Some(source_fallback);
            }
        } else {
            // A version 3 product has no version 2 constellation; a file of one
            // constellation's observations is named for it. A file with none
            // states the list of the constellation it is named for besides those
            // counts name, so it is named for one whose list no count keeps,
            // GPS first, which `M (MIXED)` already names.
            let mut systems = source_systems.iter();
            product.header.rinex2_system = match (systems.next(), systems.next()) {
                (Some(only), None) => Some(*only),
                (None, _) => {
                    let counted: std::collections::BTreeSet<GnssSystem> = self
                        .header
                        .prn_obs_counts
                        .iter()
                        .filter(|(_, counts)| !counts.is_empty())
                        .map(|(sat, _)| sat.system)
                        .collect();
                    self.header
                        .obs_codes
                        .keys()
                        .find(|system| !counted.contains(system))
                        .copied()
                        .filter(|system| *system != GnssSystem::Gps)
                }
                _ => None,
            };
        }
        if self.is_rinex2() && !self.header.rinex2_types.is_empty() {
            let observed = self.rinex2_observed_systems();
            if !source_observed && !product.header.obs_codes.contains_key(&source_fallback) {
                product.header.obs_codes.insert(
                    source_fallback,
                    super::rinex2_system_obs_codes(
                        source_fallback,
                        &self.header.rinex2_types,
                        self.header.version,
                    ),
                );
                implied.insert(source_fallback);
            }
            for system in self.rinex2_unlisted_systems() {
                product.header.obs_codes.insert(
                    system,
                    super::rinex2_system_obs_codes(
                        system,
                        &self.header.rinex2_types,
                        self.header.version,
                    ),
                );
                if !observed.contains(&system) {
                    implied.insert(system);
                }
            }
        }
        // Laying the codes out would drop it, so it is refused before anything
        // is transformed.
        for (epoch_index, epoch) in product.epochs.iter().enumerate() {
            for (sat, values) in epoch.sats.iter().chain(&epoch.cycle_slips) {
                let codes = product
                    .header
                    .obs_codes
                    .get(&sat.system)
                    .map_or(0, Vec::len);
                if values.len() > codes {
                    return Err(RinexObsWriteError::ValuesWithoutCodes {
                        epoch_index,
                        satellite: *sat,
                        codes,
                        values: values.len(),
                    });
                }
            }
        }
        for (sat, counts) in &product.header.prn_obs_counts {
            let codes = product
                .header
                .obs_codes
                .get(&sat.system)
                .map_or(0, Vec::len);
            if counts.len() > codes {
                return Err(RinexObsWriteError::CountsWithoutCodes {
                    satellite: *sat,
                    codes,
                    counts: counts.len(),
                });
            }
        }
        // Three decimals are what an observation field holds, and a value read
        // through a scale factor can carry more once the factor is gone. A
        // cycle slip sits in the same field.
        for (epoch_index, epoch) in product.epochs.iter_mut().enumerate() {
            let records = epoch
                .sats
                .iter_mut()
                .map(|record| (false, record))
                .chain(epoch.cycle_slips.iter_mut().map(|record| (true, record)));
            for (slip, (sat, values)) in records {
                let codes = product.header.obs_codes.get(&sat.system);
                for (index, value) in values.iter_mut().enumerate() {
                    let Some(held) = value.value.filter(|held| held.is_finite()) else {
                        continue;
                    };
                    let Ok(rounded) = format!("{held:.3}").parse::<f64>() else {
                        continue;
                    };
                    if rounded != held {
                        let code = codes
                            .and_then(|list| list.get(index))
                            .cloned()
                            .unwrap_or_default();
                        changes.push(if slip {
                            ObsDowngradeChange::CycleSlipRounded {
                                epoch_index,
                                satellite: *sat,
                                code,
                                from: held,
                                to: rounded,
                            }
                        } else {
                            ObsDowngradeChange::ValueRounded {
                                epoch_index,
                                satellite: *sat,
                                code,
                                from: held,
                                to: rounded,
                            }
                        });
                        value.value = Some(rounded);
                    }
                }
            }
        }
        // A product version 2 already states needs no layout, and laying it out
        // anyway could only report moves nobody needed. Once the names are
        // chosen, a list the file does not state is removed and reported rather
        // than left out of the file unsaid.
        if let Ok(names) = product.rinex2_names() {
            product.remove_unstated_lists(&names, &mut changes);
            for (system, held) in &product.header.obs_codes {
                let read = super::rinex2_system_obs_codes(*system, &names, version);
                for (original, proposed) in held.iter().zip(read.iter()) {
                    check_carrier_preservation(
                        *system,
                        original,
                        proposed,
                        self.header.version,
                        version,
                    )?;
                }
            }
            product.leave_implied(&implied, names);
            product.header.declared_obs_codes = product.rinex2_declared_after_downgrade();
            product.to_rinex_string()?;
            return Ok((product, changes));
        }
        let layout = product.rinex2_obs_layout();
        if layout.names.len() > super::MAX_OBS_TYPE_COUNT {
            return Err(RinexObsWriteError::TooManyObservationTypes {
                count: layout.names.len(),
            });
        }
        for (system, row) in &layout.slots {
            let read = super::rinex2_system_obs_codes(*system, &layout.names, version);
            let held = &product.header.obs_codes[system];
            for (column, slot) in row.iter().enumerate() {
                if let Some(index) = slot {
                    let original = &held[*index];
                    let proposed = &read[column];
                    check_carrier_preservation(
                        *system,
                        original,
                        proposed,
                        self.header.version,
                        version,
                    )?;
                }
            }
        }
        let mut obs_codes = BTreeMap::new();
        for (system, row) in &layout.slots {
            let read = super::rinex2_system_obs_codes(*system, &layout.names, version);
            let held = &product.header.obs_codes[system];
            for (column, slot) in row.iter().enumerate() {
                if let Some(index) = slot {
                    if *index != column {
                        changes.push(ObsDowngradeChange::CodeMoved {
                            system: *system,
                            code: read[column].clone(),
                            from: *index,
                            to: column,
                        });
                    }
                }
                match slot {
                    Some(index) if held[*index] != read[column] => {
                        changes.push(ObsDowngradeChange::CodeRenamed {
                            system: *system,
                            from: held[*index].clone(),
                            to: read[column].clone(),
                        });
                    }
                    Some(_) => {}
                    None => changes.push(ObsDowngradeChange::CodeAdded {
                        system: *system,
                        code: read[column].clone(),
                    }),
                }
            }
            obs_codes.insert(*system, read);
        }
        for epoch in &mut product.epochs {
            for (sat, values) in epoch.sats.iter_mut().chain(epoch.cycle_slips.iter_mut()) {
                if let Some(row) = layout.slots.get(&sat.system) {
                    let placed: Vec<ObsValue> = row
                        .iter()
                        .map(|slot| {
                            slot.and_then(|index| values.get(index).copied())
                                .unwrap_or(BLANK)
                        })
                        .collect();
                    *values = placed;
                }
            }
        }
        for (sat, counts) in &mut product.header.prn_obs_counts {
            if counts.is_empty() {
                continue;
            }
            if let Some(row) = layout.slots.get(&sat.system) {
                let placed: Vec<Option<usize>> = row
                    .iter()
                    .map(|slot| slot.and_then(|index| counts.get(index).copied()).flatten())
                    .collect();
                *counts = placed;
            }
        }
        let types = obs_codes.values().map(Vec::len).max().unwrap_or_default();
        // A list the layout gave no row stays as it was.
        for (system, codes) in &product.header.obs_codes {
            obs_codes.entry(*system).or_insert_with(|| codes.clone());
        }
        product.header.obs_codes = obs_codes;
        // The laid-out lists are what the layout's names read as, so names are
        // found for them; other names reading as them too may also read as a
        // list nothing states, which then stays.
        let names = product
            .rinex2_names()
            .unwrap_or_else(|_| layout.names.clone());
        product.remove_unstated_lists(&names, &mut changes);
        product.leave_implied(&implied, names);
        product.header.declared_obs_codes = product.rinex2_declared_after_downgrade();
        if types > super::MAX_OBS_TYPE_COUNT {
            return Err(RinexObsWriteError::TooManyObservationTypes { count: types });
        }
        // The result is only returned once it writes: anything version 2 still
        // cannot state is refused here, with the field that would change.
        product.to_rinex_string()?;
        Ok((product, changes))
    }
}

fn check_carrier_preservation(
    system: GnssSystem,
    original: &str,
    proposed: &str,
    source_version: f64,
    target_version: f64,
) -> Result<(), RinexObsWriteError> {
    let source_freq = crate::frequencies::rinex_observation_frequency_hz(
        system,
        original,
        source_version,
        Some(0),
    );
    let target_freq = crate::frequencies::rinex_observation_frequency_hz(
        system,
        proposed,
        target_version,
        Some(0),
    );
    if let (Some(f_src), Some(f_tgt)) = (source_freq, target_freq) {
        if f_src != f_tgt {
            return Err(RinexObsWriteError::ObservableNotRepresentable {
                system,
                code: original.to_string(),
                version: target_version,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod order_search_tests {
    use super::OrderSearch;

    #[test]
    fn a_search_out_of_work_to_check_an_assignment_does_not_prove_a_worse_order() {
        // Two columns, each keeping a code at the other's position, with work
        // for the cost matrix, the assignment and the pop but not for checking
        // the assignment when it is offered. The search used to drop that
        // valid assignment, empty its queue and call the built order, keeping
        // nothing in place, the best.
        let rows = vec![Some(1), Some(0)];
        let names = vec!["X1".to_string(), "X2".to_string()];
        let mut search = OrderSearch::new(vec![vec![None, None]], &[&rows], &names);
        search.work_floor = 100_000_000;
        search.work_left = search.work_floor + 12;
        let proved = search.search();
        assert!(
            !proved || search.best_kept == 2,
            "proved an order keeping {} codes in place, where swapping keeps 2",
            search.best_kept
        );
    }
}
