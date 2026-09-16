//! IONEX header records a product carries beside its grid, and the findings the
//! reader reports without refusing a file.

use crate::astro::time::model::Instant;

/// The `MAPPING FUNCTION` an IONEX product declares for its TEC determination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IonexMappingFunction {
    /// `NONE`: no mapping function was used, as for altimetry.
    NoMapping,
    /// `COSZ`: `1/cos(z)`.
    CosZ,
    /// `QFAC`: Q-factor.
    QFactor,
    /// Another code, kept as written. The spec lists `NONE`, `COSZ` and `QFAC`
    /// and says "Others might be introduced".
    Other(String),
}

impl IonexMappingFunction {
    /// The record's code: `NONE`, `COSZ`, `QFAC`, or the other code as written.
    pub fn code(&self) -> &str {
        match self {
            Self::NoMapping => "NONE",
            Self::CosZ => "COSZ",
            Self::QFactor => "QFAC",
            Self::Other(code) => code,
        }
    }

    /// The mapping function a code names, or `None` for a blank code.
    pub(crate) fn from_code(code: &str) -> Option<Self> {
        match code.trim() {
            "" => None,
            "NONE" => Some(Self::NoMapping),
            "COSZ" => Some(Self::CosZ),
            "QFAC" => Some(Self::QFactor),
            other => Some(Self::Other(other.to_string())),
        }
    }
}

/// Descriptive IONEX header records, kept so a product written back carries
/// them.
///
/// The records a product's grid determines, `EPOCH OF FIRST MAP`,
/// `EPOCH OF LAST MAP`, `# OF MAPS IN FILE` and `MAP DIMENSION`, are not held
/// here: the writer derives them from the maps. A file that leaves out one of
/// the records held here reads as that record's value in [`IonexHeader::new`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IonexHeader {
    /// `IONEX VERSION / TYPE` format version.
    pub version: f64,
    /// `IONEX VERSION / TYPE` satellite system or theoretical model, such as
    /// `GPS` or `MIX`; blank where the file names none.
    pub satellite_system: String,
    /// `PGM / RUN BY / DATE` program name.
    pub program: String,
    /// `PGM / RUN BY / DATE` agency name.
    pub run_by: String,
    /// `PGM / RUN BY / DATE` file creation date, as written.
    pub date: String,
    /// `DESCRIPTION` records, in file order.
    pub descriptions: Vec<String>,
    /// `COMMENT` records of the header outside auxiliary data blocks, in file
    /// order.
    pub comments: Vec<String>,
    /// `INTERVAL` between maps, in seconds; `0` where it may vary.
    pub interval_s: u32,
    /// `MAPPING FUNCTION`; `None` where the file gives no code.
    pub mapping_function: Option<IonexMappingFunction>,
    /// `ELEVATION CUTOFF`, in degrees; `0.0` where it is unknown.
    ///
    /// This is the lowest elevation the producer used to determine the maps.
    /// It describes the product and does not limit the elevations a slant
    /// delay may be evaluated at.
    pub elevation_cutoff_deg: f64,
    /// `OBSERVABLES USED`; blank for a theoretical model.
    pub observables_used: String,
    /// `# OF STATIONS`, where the file gives it.
    pub station_count: Option<u32>,
    /// `# OF SATELLITES`, where the file gives it.
    pub satellite_count: Option<u32>,
    /// `# OF MAPS IN FILE`, where the file gives it.
    ///
    /// IONEX 1 gives this as the total number of TEC, RMS and height maps,
    /// while CODE, IGS and UPC write the number of TEC maps: 25 with 25 RMS
    /// maps, 97 with 97. The value the file carried is kept so a product
    /// written back carries the same one.
    pub maps_in_file: Option<u32>,
}

impl IonexHeader {
    /// A header declaring `mapping_function`, with every other record at the
    /// value the spec gives for an unstated one: version `1.0`, a blank
    /// satellite system, program, agency and date, no description or comment,
    /// an `INTERVAL` of `0` (may vary), an `ELEVATION CUTOFF` of `0.0`
    /// (unknown), blank `OBSERVABLES USED` (a theoretical model), and no station
    /// or satellite count.
    pub fn new(mapping_function: IonexMappingFunction) -> Self {
        Self {
            mapping_function: Some(mapping_function),
            ..Self::unstated()
        }
    }

    /// The header a file with none of these records reads as.
    pub(crate) fn unstated() -> Self {
        Self {
            version: 1.0,
            satellite_system: String::new(),
            program: String::new(),
            run_by: String::new(),
            date: String::new(),
            descriptions: Vec::new(),
            comments: Vec::new(),
            interval_s: 0,
            mapping_function: None,
            elevation_cutoff_deg: 0.0,
            observables_used: String::new(),
            station_count: None,
            satellite_count: None,
            maps_in_file: None,
        }
    }
}

/// A finding the IONEX reader reports without refusing the file.
///
/// Most concern a header record that summarizes or describes the maps. The
/// values read do not depend on it: every map carries its own epoch and bands,
/// so the file is read the same with or without the record.
/// [`IonexWarning::NotANumberValue`] concerns a data field that gives no number.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum IonexWarning {
    /// A record IONEX 1 marks mandatory is absent; the value names its label.
    MissingRecord(&'static str),
    /// `IONEX VERSION / TYPE`, which IONEX 1 places first, follows another
    /// record.
    VersionRecordNotFirst {
        /// One-based line of the version record.
        line: usize,
    },
    /// `EPOCH OF FIRST MAP` or `EPOCH OF LAST MAP` names another epoch than
    /// the maps carry.
    EpochMismatch {
        /// The record's label.
        label: &'static str,
        /// One-based line of the record.
        line: usize,
        /// The epoch the record names.
        declared: Instant,
        /// The epoch of the first or last TEC map.
        maps: Instant,
    },
    /// `# OF MAPS IN FILE` counts neither the TEC maps nor every TEC, RMS and
    /// height map present. The spec describes it as the "Total number of
    /// TEC/RMS/HGT maps", and producers write the TEC map count, so either is
    /// read as agreeing.
    MapCountMismatch {
        /// One-based line of the record.
        line: usize,
        /// The count the record gives.
        declared: u64,
        /// The TEC maps present.
        tec_maps: usize,
        /// Every TEC, RMS and height map present.
        all_maps: usize,
    },
    /// A data field gives `nan`, which IONEX 1 does not define; it writes
    /// `9999` for a non-available value. The node reads as non-available.
    NotANumberValue {
        /// `TEC`, `RMS` or `HEIGHT`.
        kind: &'static str,
        /// The map's number in its `START OF ... MAP` record.
        map_number: usize,
        /// One-based line of the data record.
        line: usize,
        /// Latitude of the node, degrees.
        lat_deg: f64,
        /// Longitude of the node, degrees.
        lon_deg: f64,
    },
    /// A nonzero `INTERVAL` is not the spacing between two consecutive TEC
    /// maps.
    IntervalMismatch {
        /// One-based line of the record.
        line: usize,
        /// The interval the record gives, in seconds.
        declared_s: u32,
        /// The later map of the first pair spaced otherwise, counting from 1.
        map_number: usize,
        /// That pair's spacing, in seconds.
        spacing_s: i64,
    },
    /// A map gives no `EXPONENT` record before its first band while one set in
    /// an earlier map is still in effect, so that exponent is the unit of its
    /// values.
    ///
    /// IONEX 1 says of the header records that "Each value remains valid until
    /// changed by an additional header record", which carries an exponent
    /// across a map boundary. The map that inherits one is reported so the
    /// carry is never silent.
    ExponentCarriedIntoMap {
        /// `TEC`, `RMS` or `HEIGHT`.
        kind: &'static str,
        /// The map's number in its `START OF ... MAP` record.
        map_number: usize,
        /// One-based line of the band record that reads at the carried
        /// exponent.
        line: usize,
        /// The exponent in effect.
        exponent: i32,
        /// One-based line of the map that set it.
        set_by_line: usize,
    },
}

impl core::fmt::Display for IonexWarning {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingRecord(label) => write!(f, "IONEX has no {label} record"),
            Self::VersionRecordNotFirst { line } => write!(
                f,
                "IONEX VERSION / TYPE at line {line} is not the first record"
            ),
            Self::EpochMismatch {
                label,
                line,
                declared,
                maps,
            } => write!(
                f,
                "IONEX {label} at line {line} names {declared:?}, but the maps give {maps:?}"
            ),
            Self::MapCountMismatch {
                line,
                declared,
                tec_maps,
                all_maps,
            } => write!(
                f,
                "IONEX # OF MAPS IN FILE at line {line} gives {declared}, but the file has \
                 {tec_maps} TEC maps and {all_maps} maps in all"
            ),
            Self::NotANumberValue {
                kind,
                map_number,
                line,
                lat_deg,
                lon_deg,
            } => write!(
                f,
                "IONEX {kind} map {map_number} gives nan at latitude {lat_deg} longitude \
                 {lon_deg} (line {line}); it reads as non-available"
            ),
            Self::IntervalMismatch {
                line,
                declared_s,
                map_number,
                spacing_s,
            } => write!(
                f,
                "IONEX INTERVAL at line {line} gives {declared_s} s, but TEC map {map_number} \
                 follows the one before it by {spacing_s} s"
            ),
            Self::ExponentCarriedIntoMap {
                kind,
                map_number,
                line,
                exponent,
                set_by_line,
            } => write!(
                f,
                "IONEX {kind} map {map_number} gives no EXPONENT before its band at line \
                 {line}, so it reads at the EXPONENT {exponent} the map at line {set_by_line} \
                 set, which remains valid until another record changes it"
            ),
        }
    }
}
