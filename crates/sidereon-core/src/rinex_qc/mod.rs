//! RINEX observation/navigation lint and mechanical repair.
//!
//! This module is a sans-I/O layer over the existing RINEX and CRINEX readers.
//! It does not implement a parser. Text entry points decode through the owning
//! modules, then report typed findings derived from the parsed products.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::time::model::TimeScale;
use crate::crinex;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::rinex_common::{dominant_obs_interval_s, obs_epoch_seconds, usable_obs_interval_s};
use crate::rinex_nav::{
    parse_iono_corrections, parse_leap_seconds, parse_nav, parse_nav_lenient, BroadcastRecord,
    IonoCorrections, NavMessage, NavParseError,
};
use crate::rinex_obs::{
    AntennaInfo, ObsEpoch, ObsEpochTime, ObsHeader, PgmRunByDate, ReceiverInfo, RinexObs,
};
use crate::Result;

const EARTH_FIXED_RADIUS_MIN_M: f64 = 6_300_000.0;
const EARTH_FIXED_RADIUS_MAX_M: f64 = 6_400_000.0;

/// Severity assigned to a lint finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// The file cannot be represented by the existing parsed product.
    Fatal,
    /// The parsed product violates a standard-required invariant.
    Error,
    /// The parsed product is suspicious or will lose information in this slice.
    Warning,
    /// A useful fact about product scope or content.
    Info,
}

/// Location associated with a lint finding when known.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FindingRef {
    /// Zero-based epoch index.
    pub epoch_index: Option<usize>,
    /// Satellite token.
    pub satellite: Option<String>,
    /// Header or record field name.
    pub field: Option<&'static str>,
}

impl FindingRef {
    fn field(field: &'static str) -> Self {
        Self {
            field: Some(field),
            ..Self::default()
        }
    }

    fn epoch(epoch_index: usize) -> Self {
        Self {
            epoch_index: Some(epoch_index),
            ..Self::default()
        }
    }

    fn sat(epoch_index: usize, sat: GnssSatelliteId) -> Self {
        Self {
            epoch_index: Some(epoch_index),
            satellite: Some(sat.to_string()),
            ..Self::default()
        }
    }
}

/// A typed RINEX lint finding.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Finding {
    /// OBS parse failed before a parsed product existed.
    ObsFatalParse {
        /// Defaults to an empty [`FindingRef`] because no parsed location exists for the failure.
        at: FindingRef,
        /// Stringified CRINEX decoder or OBS parser error that prevented a parsed product.
        message: String,
    },
    /// OBS version is not one of the published versions covered here.
    ObsUnpublishedVersion {
        /// Points to `RINEX VERSION / TYPE` in the OBS header.
        at: FindingRef,
        /// Header version rejected by the published-version check.
        version: f64,
    },
    /// A mandatory OBS header retained by the current product is absent.
    ObsMissingHeader {
        /// Points to the absent header record.
        at: FindingRef,
        /// RINEX label of the absent header record.
        label: &'static str,
    },
    /// OBS header has no observation-code table.
    ObsMissingObsTypes {
        /// Points to `SYS / # / OBS TYPES`.
        at: FindingRef,
    },
    /// OBS code syntax is not valid for this first slice.
    ObsInvalidObsCode {
        /// Points to `SYS / # / OBS TYPES`.
        at: FindingRef,
        /// GNSS system owning the rejected code table.
        system: GnssSystem,
        /// Complete observation code rejected by the system-, band-, and attribute-specific check.
        code: String,
    },
    /// OBS code is duplicated in one system table.
    ObsDuplicateObsCode {
        /// Points to `SYS / # / OBS TYPES`.
        at: FindingRef,
        /// GNSS system whose header list repeats the code.
        system: GnssSystem,
        /// Observation code repeated in that system's header list.
        code: String,
    },
    /// TIME OF FIRST OBS disagrees with the body.
    ObsTimeOfFirstMismatch {
        /// Points to `TIME OF FIRST OBS`.
        at: FindingRef,
        /// Epoch copied from the `TIME OF FIRST OBS` header.
        declared: ObsEpochTime,
        /// Time scale copied from the `TIME OF FIRST OBS` header.
        declared_scale: TimeScale,
        /// First body epoch with a normal observation flag.
        observed: ObsEpochTime,
        /// Body time scale used for comparison with `declared_scale`.
        observed_scale: TimeScale,
    },
    /// TIME OF LAST OBS disagrees with the body epoch or with the file time
    /// system declared by TIME OF FIRST OBS (RINEX 3.05: TIME OF FIRST OBS
    /// defines the time system; TIME OF LAST OBS must agree with it).
    ObsTimeOfLastMismatch {
        /// Points to `TIME OF LAST OBS`.
        at: FindingRef,
        /// Epoch copied from the `TIME OF LAST OBS` header.
        declared: ObsEpochTime,
        /// Time scale copied from the `TIME OF LAST OBS` header.
        declared_scale: TimeScale,
        /// Last body epoch with a normal observation flag.
        observed: ObsEpochTime,
        /// Body time scale used for comparison with `declared_scale`.
        observed_scale: TimeScale,
    },
    /// INTERVAL disagrees with the dominant epoch spacing.
    ObsIntervalMismatch {
        /// Points to `INTERVAL`.
        at: FindingRef,
        /// Usable interval declared by the OBS header, in seconds.
        declared_s: f64,
        /// Dominant spacing inferred from normal epochs, in seconds.
        observed_s: f64,
    },
    /// # OF SATELLITES disagrees with the body.
    ObsSatelliteCountMismatch {
        /// Points to `# OF SATELLITES`.
        at: FindingRef,
        /// Nonzero satellite count declared by the OBS header.
        declared: usize,
        /// Number of distinct satellites in normal body records.
        observed: usize,
    },
    /// PRN / # OF OBS disagrees with body tallies.
    ObsPrnObsCountMismatch {
        /// Carries the affected satellite and points to `PRN / # OF OBS`.
        at: FindingRef,
        /// Satellite key of the header count entry.
        satellite: GnssSatelliteId,
        /// Observation code at the mismatching count index, or empty when absent.
        code: String,
        /// Optional count declared by the header at that index.
        declared: Option<usize>,
        /// Number of present values observed in the body at that index.
        observed: usize,
    },
    /// GLONASS observations need a valid slot/frequency table.
    ObsGlonassSlotIssue {
        /// Carries the affected satellite and points to `GLONASS SLOT / FRQ #`.
        at: FindingRef,
        /// GLONASS satellite with an invalid or missing slot entry.
        satellite: GnssSatelliteId,
        /// `invalid channel` for an out-of-range header channel, or `missing slot` for a body satellite without an entry.
        issue: &'static str,
    },
    /// SYS / PHASE SHIFT names a code absent from SYS / # / OBS TYPES.
    ObsPhaseShiftUndeclaredCode {
        /// Points to `SYS / PHASE SHIFT`.
        at: FindingRef,
        /// GNSS system named by the phase-shift entry.
        system: GnssSystem,
        /// Phase-shift code absent from that system's declared code list.
        code: String,
    },
    /// SYS / SCALE FACTOR is invalid or names an undeclared code.
    ObsScaleFactorIssue {
        /// Points to `SYS / SCALE FACTOR`.
        at: FindingRef,
        /// GNSS system named by the scale-factor entry.
        system: GnssSystem,
        /// `None` for an unsupported factor; otherwise the undeclared code.
        code: Option<String>,
    },
    /// MARKER TYPE is not a RINEX Table 8 keyword.
    ObsMarkerTypeIssue {
        /// Points to `MARKER TYPE`.
        at: FindingRef,
        /// Original marker-type text copied from the header before trimming for validation.
        marker_type: String,
    },
    /// Identity/header field exceeds width or has non-printable ASCII.
    ObsIdentityFieldIssue {
        /// Points to the identity field passed to the header checker.
        at: FindingRef,
        /// Static RINEX label of the invalid identity field.
        label: &'static str,
        /// Original identity value that exceeded its width or contained a non-printable byte.
        value: String,
    },
    /// Approximate position is implausible for a fixed marker.
    ObsImplausibleApproxPosition {
        /// Points to `APPROX POSITION XYZ`.
        at: FindingRef,
        /// Nonzero ECEF radius computed from the header position, in meters.
        radius_m: f64,
    },
    /// Antenna height/east/north offset is implausible.
    ObsImplausibleAntennaDelta {
        /// Points to `ANTENNA: DELTA H/E/N`.
        at: FindingRef,
        /// Zero-based H, E, or N component index that exceeded the limit.
        component: usize,
        /// Offending antenna offset component, in meters.
        value_m: f64,
    },
    /// Epoch times are not strictly increasing.
    ObsEpochOrder {
        /// Carries the zero-based index of the current out-of-order epoch.
        at: FindingRef,
        /// Previous normal epoch retained by the order scan.
        previous: ObsEpochTime,
        /// Current normal epoch whose rounded key is earlier than `previous`.
        current: ObsEpochTime,
    },
    /// Two normal epochs carry the same timestamp.
    ObsDuplicateEpoch {
        /// Carries the zero-based index of the later duplicate epoch.
        at: FindingRef,
        /// Later normal epoch whose rounded key was already seen.
        epoch: ObsEpochTime,
    },
    /// The parser skipped satellite records it could not represent.
    ObsSkippedRecords {
        /// Defaults to an empty reference because the parser exposes no more specific location.
        at: FindingRef,
        /// Positive parser counter for records that could not be represented.
        count: usize,
    },
    /// Epoch record count disagreed with retained satellite records.
    ObsEpochSatCountMismatch {
        /// Carries the zero-based epoch index.
        at: FindingRef,
        /// `NUM SAT` count declared in the epoch record.
        declared: usize,
        /// Number of satellite entries retained in the epoch map.
        retained: usize,
    },
    /// A retained event epoch had special records that are not retained.
    ObsEventSpecialRecords {
        /// Carries the zero-based event epoch index.
        at: FindingRef,
        /// Number of special records recorded for the event epoch.
        count: usize,
    },
    /// Header record is outside the retained OBS product.
    ObsUnretainedHeader {
        /// Uses the literal `header` location because only the unsupported label is retained.
        at: FindingRef,
        /// Header label recorded as unretained by the parser.
        label: String,
    },
    /// A pseudorange value is outside the configured plausibility window.
    ObsPseudorangeOutOfRange {
        /// Carries the normal epoch and satellite containing the value.
        at: FindingRef,
        /// Observation code selected at the value's index, or empty when unavailable.
        code: String,
        /// Pseudorange value outside the 15,000,000 to 50,000,000 meter window.
        value_m: f64,
    },
    /// LLI digit is outside the three defined bits.
    ObsLossOfLockOutOfRange {
        /// Carries the normal epoch and satellite containing the LLI.
        at: FindingRef,
        /// Observation code selected at the LLI field's index, or empty when unavailable.
        code: String,
        /// Raw loss-of-lock indicator greater than 7.
        lli: u8,
    },
    /// Event epoch retained with no special records.
    ObsEventEpoch {
        /// Carries the zero-based event epoch index.
        at: FindingRef,
        /// Parsed event flag greater than 1.
        flag: u8,
    },
    /// Satellite record has all observation fields blank.
    ObsEmptySatelliteRecord {
        /// Carries the normal epoch and satellite whose observation values are all absent.
        at: FindingRef,
    },
    /// Epoch gap is larger than 1.5 times the dominant interval.
    ObsEpochGap {
        /// Carries the zero-based current normal epoch index.
        at: FindingRef,
        /// Current-minus-previous normal epoch difference, in seconds.
        gap_s: f64,
        /// Dominant normal-epoch interval used for the threshold, in seconds.
        interval_s: f64,
    },
    /// NAV parse failed before a parsed product existed.
    NavFatalParse {
        /// Defaults to an empty reference because no reliable NAV record location exists.
        at: FindingRef,
        /// Stringified error returned by lenient NAV parsing.
        message: String,
    },
    /// NAV header has no LEAP SECONDS record.
    NavLeapSecondsAbsent {
        /// Points to `LEAP SECONDS`.
        at: FindingRef,
    },
    /// NAV ionospheric correction records are malformed.
    NavIonoMalformed {
        /// Points to `IONOSPHERIC CORR`.
        at: FindingRef,
        /// Stringified ionospheric-correction parse error.
        message: String,
    },
    /// NAV record block was dropped by lenient parsing.
    NavDroppedBlock {
        /// Carries the skipped block's satellite token.
        at: FindingRef,
        /// Satellite token stored by the lenient parser for the dropped block.
        satellite: String,
        /// Lenient parser explanation for skipping the block.
        message: String,
    },
    /// Duplicate NAV records share an identity.
    NavDuplicateRecord {
        /// Carries the zero-based index of the later duplicate record.
        at: FindingRef,
        /// Satellite ID of the later duplicate broadcast record.
        satellite: GnssSatelliteId,
        /// Whether the later record equals the first record with this NAV identity.
        same_payload: bool,
    },
    /// NAV records are not in canonical order.
    NavUnsortedRecords {
        /// Defaults to an empty reference because the check compares adjacent records.
        at: FindingRef,
    },
    /// NAV broadcast fields are outside this slice's plausibility limits.
    NavImplausibleRecord {
        /// Carries the zero-based NAV record index.
        at: FindingRef,
        /// Satellite ID of the record containing the out-of-range element.
        satellite: GnssSatelliteId,
        /// Static label identifying the checked element: `eccentricity` or `sqrt_a`.
        field: &'static str,
        /// Raw broadcast value that failed its range check.
        value: f64,
    },
    /// NAV records include unhealthy satellite records.
    NavUnhealthyRecords {
        /// Defaults to an empty reference because this finding summarizes a system.
        at: FindingRef,
        /// GNSS system in the unhealthy-record tally.
        system: GnssSystem,
        /// Number of that system's records with nonzero `sv_health`.
        count: usize,
    },
    /// NAV records outside the retained/writable scope are present.
    NavOutOfScopeRecords {
        /// Defaults to an empty reference because the text scan retains no record index.
        at: FindingRef,
        /// Deterministic class of unsupported frame, message, or constellation.
        class: String,
        /// Number of text records in the reported scope class.
        count: usize,
    },
    /// INTERVAL is zero, a standards-defined representation of unavailable metadata.
    ObsIntervalUnavailable {
        /// Points to `INTERVAL`.
        at: FindingRef,
    },
    /// INTERVAL is negative, or non-finite in a caller-constructed product.
    ObsInvalidInterval {
        /// Points to `INTERVAL`.
        at: FindingRef,
        /// Raw interval value rejected as negative or non-finite, in seconds.
        declared_s: f64,
    },
}

impl Finding {
    /// Stable rule identifier.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::ObsFatalParse { .. } => "OBS-H01",
            Self::ObsUnpublishedVersion { .. } => "OBS-H02",
            Self::ObsMissingHeader { .. } => "OBS-H03",
            Self::ObsMissingObsTypes { .. } => "OBS-H04",
            Self::ObsInvalidObsCode { .. } => "OBS-H05",
            Self::ObsDuplicateObsCode { .. } => "OBS-H06",
            Self::ObsTimeOfFirstMismatch { .. } => "OBS-H07",
            Self::ObsTimeOfLastMismatch { .. } => "OBS-H08",
            Self::ObsIntervalMismatch { .. } => "OBS-H09",
            Self::ObsSatelliteCountMismatch { .. } => "OBS-H10",
            Self::ObsPrnObsCountMismatch { .. } => "OBS-H11",
            Self::ObsGlonassSlotIssue { .. } => "OBS-H12",
            Self::ObsPhaseShiftUndeclaredCode { .. } => "OBS-H13",
            Self::ObsScaleFactorIssue { .. } => "OBS-H14",
            Self::ObsMarkerTypeIssue { .. } => "OBS-H15",
            Self::ObsIdentityFieldIssue { .. } => "OBS-H16",
            Self::ObsImplausibleApproxPosition { .. } => "OBS-H17",
            Self::ObsImplausibleAntennaDelta { .. } => "OBS-H18",
            Self::ObsIntervalUnavailable { .. } => "OBS-H19",
            Self::ObsInvalidInterval { .. } => "OBS-H20",
            Self::ObsUnretainedHeader { .. } => "OBS-H90",
            Self::ObsEpochOrder { .. } => "OBS-B01",
            Self::ObsDuplicateEpoch { .. } => "OBS-B02",
            Self::ObsEpochSatCountMismatch { .. } => "OBS-B03",
            Self::ObsSkippedRecords { .. } => "OBS-B04",
            Self::ObsPseudorangeOutOfRange { .. } => "OBS-B05",
            Self::ObsLossOfLockOutOfRange { .. } => "OBS-B06",
            Self::ObsEventEpoch { .. } => "OBS-B07",
            Self::ObsEmptySatelliteRecord { .. } => "OBS-B08",
            Self::ObsEpochGap { .. } => "OBS-B09",
            Self::ObsEventSpecialRecords { .. } => "OBS-B11",
            Self::NavFatalParse { .. } => "NAV-H01",
            Self::NavLeapSecondsAbsent { .. } => "NAV-H02",
            Self::NavIonoMalformed { .. } => "NAV-H03",
            Self::NavDroppedBlock { .. } => "NAV-B01",
            Self::NavDuplicateRecord { .. } => "NAV-B02",
            Self::NavUnsortedRecords { .. } => "NAV-B03",
            Self::NavImplausibleRecord { .. } => "NAV-B04",
            Self::NavUnhealthyRecords { .. } => "NAV-B05",
            Self::NavOutOfScopeRecords { .. } => "NAV-B06",
        }
    }

    /// Rule severity.
    pub const fn severity(&self) -> Severity {
        match self {
            Self::ObsFatalParse { .. }
            | Self::ObsMissingObsTypes { .. }
            | Self::NavFatalParse { .. } => Severity::Fatal,
            Self::ObsUnpublishedVersion { .. }
            | Self::ObsSkippedRecords { .. }
            | Self::ObsPseudorangeOutOfRange { .. }
            | Self::ObsLossOfLockOutOfRange { .. }
            | Self::ObsIntervalMismatch { .. }
            | Self::ObsPhaseShiftUndeclaredCode { .. }
            | Self::ObsMarkerTypeIssue { .. }
            | Self::ObsIdentityFieldIssue { .. }
            | Self::ObsImplausibleApproxPosition { .. }
            | Self::ObsImplausibleAntennaDelta { .. }
            | Self::ObsEventSpecialRecords { .. }
            | Self::NavIonoMalformed { .. }
            | Self::NavImplausibleRecord { .. } => Severity::Warning,
            Self::ObsEventEpoch { .. }
            | Self::ObsEmptySatelliteRecord { .. }
            | Self::ObsEpochGap { .. }
            | Self::ObsIntervalUnavailable { .. }
            | Self::ObsUnretainedHeader { .. }
            | Self::NavLeapSecondsAbsent { .. }
            | Self::NavUnsortedRecords { .. }
            | Self::NavUnhealthyRecords { .. }
            | Self::NavOutOfScopeRecords { .. } => Severity::Info,
            Self::NavDuplicateRecord { same_payload, .. } => {
                if *same_payload {
                    Severity::Warning
                } else {
                    Severity::Error
                }
            }
            _ => Severity::Error,
        }
    }

    /// Standard or policy reference for the rule.
    pub const fn spec_ref(&self) -> &'static str {
        match self {
            Self::ObsFatalParse { .. } => "RINEX 3.05/4.02 Table A2",
            Self::ObsUnpublishedVersion { .. } => "RINEX version history",
            Self::ObsMissingHeader { .. } => "RINEX 3.05/4.02 Table A2",
            Self::ObsMissingObsTypes { .. } => "RINEX 3.05/4.02 Table A2",
            Self::ObsInvalidObsCode { .. } => "RINEX 3.05 Tables 13-20",
            Self::ObsDuplicateObsCode { .. } => "RINEX 3.05 section 5.2",
            Self::ObsTimeOfFirstMismatch { .. } => "RINEX 3.05 Table A2",
            Self::ObsTimeOfLastMismatch { .. } => "RINEX 3.05 Table A2, TIME OF LAST OBS",
            Self::ObsIntervalMismatch { .. } => "RINEX 3.05 Table A2",
            Self::ObsSatelliteCountMismatch { .. } => "RINEX 3.05 Table A2, # OF SATELLITES",
            Self::ObsPrnObsCountMismatch { .. } => "RINEX 3.05 Table A2, PRN / # OF OBS",
            Self::ObsGlonassSlotIssue { .. } => "RINEX 3.05 Table A2",
            Self::ObsPhaseShiftUndeclaredCode { .. } => "RINEX 3.05 Table A2",
            Self::ObsScaleFactorIssue { .. } => "RINEX 3.05 Table A2",
            Self::ObsMarkerTypeIssue { .. } => "RINEX 3.05 Table 8",
            Self::ObsIdentityFieldIssue { .. } => "RINEX 3.05 Table A2 identity fields",
            Self::ObsImplausibleApproxPosition { .. } => "RINEX 3.05 Table A2",
            Self::ObsImplausibleAntennaDelta { .. } => "RINEX 3.05 Table A2",
            Self::ObsIntervalUnavailable { .. } => {
                "RINEX 2.11 section 5.3; RINEX 3.05/4.02 section 6.5 and Table A2, INTERVAL"
            }
            Self::ObsInvalidInterval { .. } => {
                "RINEX 2.11 Table A2; RINEX 3.05/4.02 Table A2, INTERVAL"
            }
            Self::ObsUnretainedHeader { .. } => "RINEX 3.05 section 6.6",
            Self::ObsEpochOrder { .. } => "RINEX 3.05 Table A3",
            Self::ObsDuplicateEpoch { .. } => "RINEX 3.05 Table A3",
            Self::ObsEpochSatCountMismatch { .. } => "RINEX 3.05 Table A3, NUM SAT",
            Self::ObsSkippedRecords { .. } => "parser diagnostic",
            Self::ObsPseudorangeOutOfRange { .. } => "RINEX QC policy",
            Self::ObsLossOfLockOutOfRange { .. } => "RINEX 3.05 Table A3 note 1",
            Self::ObsEventEpoch { .. } => "RINEX 3.05 Table A3",
            Self::ObsEmptySatelliteRecord { .. } => "RINEX QC policy",
            Self::ObsEpochGap { .. } => "RINEX QC policy",
            Self::ObsEventSpecialRecords { .. } => "RINEX 3.05/4.02 Table A3",
            Self::NavFatalParse { .. } => "RINEX 3.05 Table A5 / RINEX 4.02 Table A7",
            Self::NavLeapSecondsAbsent { .. } => "RINEX 3.05 Table A5",
            Self::NavIonoMalformed { .. } => "RINEX 3.05 Table A5",
            Self::NavDroppedBlock { .. } => "RINEX 3.05/4.02 navigation record layout",
            Self::NavDuplicateRecord { .. } => "RINEX 3.05 section 6.12",
            Self::NavUnsortedRecords { .. } => "RINEX QC policy",
            Self::NavImplausibleRecord { .. } => "RINEX QC policy",
            Self::NavUnhealthyRecords { .. } => "RINEX 3.05 broadcast record layout",
            Self::NavOutOfScopeRecords { .. } => "RINEX QC parse-scope disclosure",
        }
    }

    /// Finding location.
    pub const fn at(&self) -> &FindingRef {
        match self {
            Self::ObsFatalParse { at, .. }
            | Self::ObsUnpublishedVersion { at, .. }
            | Self::ObsMissingHeader { at, .. }
            | Self::ObsMissingObsTypes { at }
            | Self::ObsInvalidObsCode { at, .. }
            | Self::ObsDuplicateObsCode { at, .. }
            | Self::ObsTimeOfFirstMismatch { at, .. }
            | Self::ObsTimeOfLastMismatch { at, .. }
            | Self::ObsIntervalMismatch { at, .. }
            | Self::ObsSatelliteCountMismatch { at, .. }
            | Self::ObsPrnObsCountMismatch { at, .. }
            | Self::ObsGlonassSlotIssue { at, .. }
            | Self::ObsPhaseShiftUndeclaredCode { at, .. }
            | Self::ObsScaleFactorIssue { at, .. }
            | Self::ObsMarkerTypeIssue { at, .. }
            | Self::ObsIdentityFieldIssue { at, .. }
            | Self::ObsImplausibleApproxPosition { at, .. }
            | Self::ObsImplausibleAntennaDelta { at, .. }
            | Self::ObsIntervalUnavailable { at }
            | Self::ObsInvalidInterval { at, .. }
            | Self::ObsUnretainedHeader { at, .. }
            | Self::ObsEpochOrder { at, .. }
            | Self::ObsDuplicateEpoch { at, .. }
            | Self::ObsEpochSatCountMismatch { at, .. }
            | Self::ObsSkippedRecords { at, .. }
            | Self::ObsPseudorangeOutOfRange { at, .. }
            | Self::ObsLossOfLockOutOfRange { at, .. }
            | Self::ObsEventEpoch { at, .. }
            | Self::ObsEmptySatelliteRecord { at }
            | Self::ObsEpochGap { at, .. }
            | Self::ObsEventSpecialRecords { at, .. }
            | Self::NavFatalParse { at, .. }
            | Self::NavLeapSecondsAbsent { at }
            | Self::NavIonoMalformed { at, .. }
            | Self::NavDroppedBlock { at, .. }
            | Self::NavDuplicateRecord { at, .. }
            | Self::NavUnsortedRecords { at }
            | Self::NavImplausibleRecord { at, .. }
            | Self::NavUnhealthyRecords { at, .. }
            | Self::NavOutOfScopeRecords { at, .. } => at,
        }
    }

    /// Whether this finding can be changed by the repair helpers in this slice.
    pub const fn is_repairable(&self) -> bool {
        matches!(
            self,
            Self::ObsTimeOfFirstMismatch { .. }
                | Self::ObsTimeOfLastMismatch { .. }
                | Self::ObsIntervalMismatch { .. }
                | Self::ObsIntervalUnavailable { .. }
                | Self::ObsInvalidInterval { .. }
                | Self::ObsSatelliteCountMismatch { .. }
                | Self::ObsPrnObsCountMismatch { .. }
                | Self::ObsEpochOrder { .. }
                | Self::ObsDuplicateEpoch { .. }
                | Self::ObsEpochSatCountMismatch { .. }
                | Self::ObsEmptySatelliteRecord { .. }
                | Self::NavDuplicateRecord {
                    same_payload: true,
                    ..
                }
                | Self::NavUnsortedRecords { .. }
        )
    }
}

/// Lint result.
#[derive(Debug, Clone, PartialEq)]
pub struct LintReport {
    /// Findings in deterministic rule order.
    pub findings: Vec<Finding>,
    /// Whether a CRINEX input was decoded before linting.
    pub decoded_from_crinex: bool,
}

impl LintReport {
    /// Clean means no fatal or error findings.
    pub fn is_clean(&self) -> bool {
        self.findings
            .iter()
            .all(|f| !matches!(f.severity(), Severity::Fatal | Severity::Error))
    }

    /// Count findings by severity.
    pub fn count(&self, severity: Severity) -> usize {
        self.findings
            .iter()
            .filter(|finding| finding.severity() == severity)
            .count()
    }
}

/// Repair options for the first core slice.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RepairOptions {
    /// Caller-supplied PGM/RUN BY/DATE stamp for A8.
    pub file_stamp: Option<PgmRunByDate>,
    /// Set `INTERVAL` to the dominant normal-epoch spacing.
    ///
    /// When no cadence can be inferred, an unusable present value is removed.
    /// When false, source interval metadata is preserved.
    pub set_interval: bool,
    /// Set `TIME OF LAST OBS` when absent or wrong.
    pub set_time_of_last_obs: bool,
    /// Recompute `# OF SATELLITES` and `PRN / # OF OBS`.
    pub set_obs_counts: bool,
    /// Drop satellite rows whose observation fields are all blank.
    pub drop_empty_records: bool,
    /// Sort NAV records by satellite and toc.
    pub sort_records: bool,
    /// Allow text repair to drop records outside the retained product scope.
    pub drop_unsupported: bool,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            file_stamp: None,
            set_interval: false,
            set_time_of_last_obs: false,
            set_obs_counts: false,
            drop_empty_records: false,
            sort_records: true,
            drop_unsupported: false,
        }
    }
}

/// One mechanical repair action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairAction {
    /// Catalog action id, e.g. `A3`.
    pub id: &'static str,
    /// Short action description.
    pub message: String,
}

/// Observation repair result.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsRepair {
    /// Repaired parsed product.
    pub repaired: RinexObs,
    /// Actions applied.
    pub actions: Vec<RepairAction>,
    /// Lint report after repair.
    pub remaining: LintReport,
    /// Whether text input was decoded from CRINEX.
    pub decoded_from_crinex: bool,
}

/// Navigation repair result.
#[derive(Debug, Clone, PartialEq)]
pub struct NavRepair {
    /// Repaired broadcast records.
    pub records: Vec<BroadcastRecord>,
    /// Header ionospheric corrections parsed from text, if available.
    pub iono: Option<IonoCorrections>,
    /// Header leap-second count parsed from text, if available.
    pub leap_seconds: Option<f64>,
    /// Actions applied.
    pub actions: Vec<RepairAction>,
    /// Lint report after repair.
    pub remaining: LintReport,
}

/// Validated observation-header edit builder.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObsHeaderEdit {
    marker_name: Option<String>,
    marker_number: Option<Option<String>>,
    marker_type: Option<String>,
    observer: Option<String>,
    agency: Option<String>,
    receiver: Option<ReceiverInfo>,
    antenna: Option<AntennaInfo>,
    antenna_height_m: Option<f64>,
    antenna_eccentricity_en_m: Option<(f64, f64)>,
    approx_position_m: Option<[f64; 3]>,
}

/// One field changed by [`ObsHeaderEdit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedEdit {
    /// Field label.
    pub field: &'static str,
    /// Previous value.
    pub old_value: Option<String>,
    /// New value.
    pub new_value: Option<String>,
    /// Warning text for accepted suspicious values.
    pub warning: Option<String>,
}

/// Header edit validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderEditError {
    /// A staged field failed validation.
    InvalidField {
        /// Field label.
        field: &'static str,
        /// Validation reason.
        reason: &'static str,
    },
}

impl core::fmt::Display for HeaderEditError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidField { field, reason } => {
                write!(f, "invalid RINEX OBS header field {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for HeaderEditError {}

impl ObsHeaderEdit {
    /// Create an empty edit builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage marker name.
    pub fn marker_name(mut self, v: &str) -> Self {
        self.marker_name = Some(v.to_string());
        self
    }

    /// Stage marker number.
    pub fn marker_number(mut self, v: &str) -> Self {
        self.marker_number = Some(Some(v.to_string()));
        self
    }

    /// Stage marker-number removal.
    pub fn clear_marker_number(mut self) -> Self {
        self.marker_number = Some(None);
        self
    }

    /// Stage marker type.
    pub fn marker_type(mut self, v: &str) -> Self {
        self.marker_type = Some(v.to_string());
        self
    }

    /// Stage observer.
    pub fn observer(mut self, v: &str) -> Self {
        self.observer = Some(v.to_string());
        self
    }

    /// Stage agency.
    pub fn agency(mut self, v: &str) -> Self {
        self.agency = Some(v.to_string());
        self
    }

    /// Stage receiver fields.
    pub fn receiver(mut self, number: &str, receiver_type: &str, version: &str) -> Self {
        self.receiver = Some(ReceiverInfo {
            number: number.to_string(),
            receiver_type: receiver_type.to_string(),
            version: version.to_string(),
        });
        self
    }

    /// Stage antenna fields.
    pub fn antenna(mut self, number: &str, antenna_type: &str) -> Self {
        self.antenna = Some(AntennaInfo {
            number: number.to_string(),
            antenna_type: antenna_type.to_string(),
        });
        self
    }

    /// Stage antenna height.
    pub fn antenna_height_m(mut self, v: f64) -> Self {
        self.antenna_height_m = Some(v);
        self
    }

    /// Stage antenna east/north eccentricities.
    pub fn antenna_eccentricity_en_m(mut self, east: f64, north: f64) -> Self {
        self.antenna_eccentricity_en_m = Some((east, north));
        self
    }

    /// Stage approximate position.
    pub fn approx_position_m(mut self, xyz: [f64; 3]) -> Self {
        self.approx_position_m = Some(xyz);
        self
    }

    /// Validate and apply all staged changes atomically.
    pub fn apply(
        self,
        header: &mut ObsHeader,
    ) -> std::result::Result<Vec<AppliedEdit>, HeaderEditError> {
        self.validate()?;
        let original = header.clone();
        let mut edited = header.clone();
        let mut applied = Vec::new();

        if let Some(value) = self.marker_name {
            push_edit(
                &mut applied,
                "MARKER NAME",
                edited.marker_name.clone(),
                Some(value.clone()),
                None,
            );
            edited.marker_name = Some(value);
        }
        if let Some(value) = self.marker_number {
            push_edit(
                &mut applied,
                "MARKER NUMBER",
                edited.marker_number.clone(),
                value.clone(),
                None,
            );
            edited.marker_number = value;
        }
        if let Some(value) = self.marker_type {
            let warning = (!is_valid_marker_type(&value))
                .then(|| "not a RINEX Table 8 marker type".to_string());
            push_edit(
                &mut applied,
                "MARKER TYPE",
                edited.marker_type.clone(),
                Some(value.clone()),
                warning,
            );
            edited.marker_type = Some(value);
        }
        if let Some(value) = self.observer {
            push_edit(
                &mut applied,
                "OBSERVER",
                edited.observer.clone(),
                Some(value.clone()),
                None,
            );
            edited.observer = Some(value);
        }
        if let Some(value) = self.agency {
            push_edit(
                &mut applied,
                "AGENCY",
                edited.agency.clone(),
                Some(value.clone()),
                None,
            );
            edited.agency = Some(value);
        }
        if let Some(value) = self.receiver {
            push_edit(
                &mut applied,
                "REC # / TYPE / VERS",
                edited.receiver.as_ref().map(format_receiver),
                Some(format_receiver(&value)),
                None,
            );
            edited.receiver = Some(value);
        }
        if let Some(value) = self.antenna {
            push_edit(
                &mut applied,
                "ANT # / TYPE",
                edited.antenna.as_ref().map(format_antenna),
                Some(format_antenna(&value)),
                None,
            );
            edited.antenna = Some(value);
        }
        if let Some(value) = self.approx_position_m {
            push_edit(
                &mut applied,
                "APPROX POSITION XYZ",
                edited.approx_position_m.map(|v| format!("{v:?}")),
                Some(format!("{value:?}")),
                None,
            );
            edited.approx_position_m = Some(value);
        }
        if self.antenna_height_m.is_some() || self.antenna_eccentricity_en_m.is_some() {
            let mut delta = edited.antenna_delta_hen_m.unwrap_or([0.0; 3]);
            if let Some(height) = self.antenna_height_m {
                delta[0] = height;
            }
            if let Some((east, north)) = self.antenna_eccentricity_en_m {
                delta[1] = east;
                delta[2] = north;
            }
            push_edit(
                &mut applied,
                "ANTENNA: DELTA H/E/N",
                edited.antenna_delta_hen_m.map(|v| format!("{v:?}")),
                Some(format!("{delta:?}")),
                None,
            );
            edited.antenna_delta_hen_m = Some(delta);
        }

        if edited == original {
            return Ok(Vec::new());
        }
        *header = edited;
        Ok(applied)
    }

    fn validate(&self) -> std::result::Result<(), HeaderEditError> {
        if let Some(value) = &self.marker_name {
            validate_text_field("MARKER NAME", value, 60, false)?;
        }
        if let Some(Some(value)) = &self.marker_number {
            validate_text_field("MARKER NUMBER", value, 20, true)?;
        }
        if let Some(value) = &self.marker_type {
            validate_text_field("MARKER TYPE", value, 20, false)?;
        }
        if let Some(value) = &self.observer {
            validate_text_field("OBSERVER", value, 20, false)?;
        }
        if let Some(value) = &self.agency {
            validate_text_field("AGENCY", value, 40, false)?;
        }
        if let Some(value) = &self.receiver {
            validate_text_field("REC #", &value.number, 20, true)?;
            validate_text_field("REC TYPE", &value.receiver_type, 20, false)?;
            validate_text_field("REC VERS", &value.version, 20, true)?;
        }
        if let Some(value) = &self.antenna {
            validate_text_field("ANT #", &value.number, 20, true)?;
            validate_text_field("ANT TYPE", &value.antenna_type, 20, false)?;
        }
        if let Some(value) = self.antenna_height_m {
            validate_antenna_delta("ANTENNA HEIGHT", value)?;
        }
        if let Some((east, north)) = self.antenna_eccentricity_en_m {
            validate_antenna_delta("ANTENNA EAST", east)?;
            validate_antenna_delta("ANTENNA NORTH", north)?;
        }
        if let Some(xyz) = self.approx_position_m {
            if !xyz.iter().all(|value| value.is_finite()) {
                return Err(HeaderEditError::InvalidField {
                    field: "APPROX POSITION XYZ",
                    reason: "must be finite",
                });
            }
            let radius = (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt();
            if radius != 0.0
                && !(EARTH_FIXED_RADIUS_MIN_M..=EARTH_FIXED_RADIUS_MAX_M).contains(&radius)
            {
                return Err(HeaderEditError::InvalidField {
                    field: "APPROX POSITION XYZ",
                    reason: "radius outside earth-fixed range",
                });
            }
        }
        Ok(())
    }
}

/// Lint an already parsed observation product.
pub fn lint_obs(obs: &RinexObs) -> LintReport {
    LintReport {
        findings: obs_findings(obs),
        decoded_from_crinex: false,
    }
}

/// CRINEX-transparent observation lint entry point.
pub fn lint_obs_text(text: &str) -> LintReport {
    let (decoded_from_crinex, text) = match decode_if_crinex(text) {
        Ok(v) => v,
        Err(error) => {
            return LintReport {
                findings: vec![Finding::ObsFatalParse {
                    at: FindingRef::default(),
                    message: error.to_string(),
                }],
                decoded_from_crinex: true,
            };
        }
    };
    match RinexObs::parse(&text) {
        Ok(obs) => LintReport {
            findings: obs_findings(&obs),
            decoded_from_crinex,
        },
        Err(error) => LintReport {
            findings: vec![classify_obs_parse_error(&error.to_string())],
            decoded_from_crinex,
        },
    }
}

fn classify_obs_parse_error(message: &str) -> Finding {
    if message.contains("no SYS / # / OBS TYPES") || message.contains("no # / TYPES OF OBSERV") {
        Finding::ObsMissingObsTypes {
            at: FindingRef::field("SYS / # / OBS TYPES"),
        }
    } else {
        Finding::ObsFatalParse {
            at: FindingRef::default(),
            message: message.to_string(),
        }
    }
}

/// Lint navigation text with the existing NAV parser and header readers.
pub fn lint_nav_text(text: &str) -> LintReport {
    let mut findings = Vec::new();
    match parse_nav_lenient(text) {
        Ok(parsed) => {
            findings.extend(nav_findings(&parsed.records));
            for skipped in parsed.skipped {
                findings.push(Finding::NavDroppedBlock {
                    at: FindingRef {
                        satellite: Some(skipped.satellite.clone()),
                        ..FindingRef::default()
                    },
                    satellite: skipped.satellite,
                    message: skipped.message,
                });
            }
        }
        Err(error) => {
            findings.push(Finding::NavFatalParse {
                at: FindingRef::default(),
                message: error.to_string(),
            });
        }
    }
    for (class, count) in nav_scope_tallies(text) {
        findings.push(Finding::NavOutOfScopeRecords {
            at: FindingRef::default(),
            class,
            count,
        });
    }
    if matches!(parse_leap_seconds(text), Ok(None)) {
        findings.push(Finding::NavLeapSecondsAbsent {
            at: FindingRef::field("LEAP SECONDS"),
        });
    }
    if let Err(error) = parse_iono_corrections(text) {
        findings.push(Finding::NavIonoMalformed {
            at: FindingRef::field("IONOSPHERIC CORR"),
            message: error.to_string(),
        });
    }
    LintReport {
        findings,
        decoded_from_crinex: false,
    }
}

/// Repair an already parsed observation product.
pub fn repair_obs(obs: &RinexObs, options: &RepairOptions) -> ObsRepair {
    let mut repaired = obs.clone();
    let mut actions = Vec::new();
    repair_obs_order_and_duplicates(&mut repaired, &mut actions);
    repair_obs_times(&mut repaired, options, &mut actions);
    repair_obs_counts(&mut repaired, options, &mut actions);
    repair_obs_file_stamp(&mut repaired, options, &mut actions);
    repair_obs_unsupported_records(&mut repaired, options, &mut actions);
    if options.set_interval {
        repair_obs_interval(&mut repaired, &mut actions);
    }
    if options.drop_empty_records {
        repair_obs_empty_records(&mut repaired, &mut actions);
    }
    let remaining = lint_obs(&repaired);
    ObsRepair {
        repaired,
        actions,
        remaining,
        decoded_from_crinex: false,
    }
}

/// CRINEX-transparent observation repair entry point.
pub fn repair_obs_text(text: &str, options: &RepairOptions) -> Result<ObsRepair> {
    let (decoded_from_crinex, text) = decode_if_crinex(text)?;
    let obs = RinexObs::parse(&text)?;
    if !options.drop_unsupported && !obs.header.unretained_header_labels.is_empty() {
        return Err(crate::Error::InvalidInput(
            "RINEX OBS text repair would drop unretained header records".to_string(),
        ));
    }
    if !options.drop_unsupported
        && obs
            .epochs
            .iter()
            .any(|epoch| epoch.flag > 1 && epoch.special_record_count > 0)
    {
        return Err(crate::Error::InvalidInput(
            "RINEX OBS text repair would drop event special records".to_string(),
        ));
    }
    let mut repaired = repair_obs(&obs, options);
    repaired.decoded_from_crinex = decoded_from_crinex;
    repaired.remaining.decoded_from_crinex = decoded_from_crinex;
    Ok(repaired)
}

/// Encode an observation repair product as CRINEX through the existing codec.
pub fn repair_obs_to_crinex_string(repair: &ObsRepair) -> Result<String> {
    crinex::encode_crinex(&repair.repaired.to_rinex_string())
}

/// Repair parsed navigation records.
pub fn repair_nav(records: &[BroadcastRecord], options: &RepairOptions) -> NavRepair {
    let mut records = records.to_vec();
    let mut actions = Vec::new();
    repair_nav_duplicates(&mut records, &mut actions);
    if options.sort_records {
        repair_nav_order(&mut records, &mut actions);
    }
    let remaining = LintReport {
        findings: nav_findings(&records),
        decoded_from_crinex: false,
    };
    NavRepair {
        records,
        iono: None,
        leap_seconds: None,
        actions,
        remaining,
    }
}

/// Repair navigation text through the existing parser.
pub fn repair_nav_text(
    text: &str,
    options: &RepairOptions,
) -> std::result::Result<NavRepair, NavParseError> {
    let scope_tallies = nav_scope_tallies(text);
    if !scope_tallies.is_empty() && !options.drop_unsupported {
        return Err(NavParseError::UnsupportedHeader(format!(
            "RINEX NAV text repair would drop out-of-scope records: {scope_tallies:?}"
        )));
    }
    let records = parse_nav(text)?;
    let mut repair = repair_nav(&records, options);
    if !scope_tallies.is_empty() {
        for (class, count) in scope_tallies {
            repair.actions.push(RepairAction {
                id: "NAV-B06",
                message: format!("dropped {count} out-of-scope NAV records in {class}"),
            });
        }
    }
    repair.iono = parse_iono_corrections(text).ok();
    repair.leap_seconds = parse_leap_seconds(text).ok().flatten();
    Ok(repair)
}

fn decode_if_crinex(text: &str) -> Result<(bool, String)> {
    let is_crinex = text
        .lines()
        .next()
        .is_some_and(|line| line.get(60..).unwrap_or("").contains("CRINEX VERS"));
    if is_crinex {
        Ok((true, crinex::decode(text)?))
    } else {
        Ok((false, text.to_string()))
    }
}

fn obs_findings(obs: &RinexObs) -> Vec<Finding> {
    let mut findings = Vec::new();
    lint_obs_header(&obs.header, &mut findings);
    lint_obs_body(obs, &mut findings);
    findings
}

fn lint_obs_header(header: &ObsHeader, findings: &mut Vec<Finding>) {
    if !matches!(published_obs_version(header.version), Some(())) {
        findings.push(Finding::ObsUnpublishedVersion {
            at: FindingRef::field("RINEX VERSION / TYPE"),
            version: header.version,
        });
    }
    if header.marker_name.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("MARKER NAME"),
            label: "MARKER NAME",
        });
    }
    if header.program_run_by_date.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("PGM / RUN BY / DATE"),
            label: "PGM / RUN BY / DATE",
        });
    }
    if header.observer.is_none() || header.agency.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("OBSERVER / AGENCY"),
            label: "OBSERVER / AGENCY",
        });
    }
    if header.receiver.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("REC # / TYPE / VERS"),
            label: "REC # / TYPE / VERS",
        });
    }
    if header.antenna.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("ANT # / TYPE"),
            label: "ANT # / TYPE",
        });
    }
    if header.antenna_delta_hen_m.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("ANTENNA: DELTA H/E/N"),
            label: "ANTENNA: DELTA H/E/N",
        });
    }
    if header.approx_position_m.is_none()
        && header
            .marker_type
            .as_deref()
            .is_none_or(is_earth_fixed_marker_type)
    {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("APPROX POSITION XYZ"),
            label: "APPROX POSITION XYZ",
        });
    }
    if header.time_of_first_obs.is_none() {
        findings.push(Finding::ObsMissingHeader {
            at: FindingRef::field("TIME OF FIRST OBS"),
            label: "TIME OF FIRST OBS",
        });
    }
    if header.obs_codes.is_empty() {
        findings.push(Finding::ObsMissingObsTypes {
            at: FindingRef::field("SYS / # / OBS TYPES"),
        });
    }
    if let Some(declared_s) = header.interval_s {
        if declared_s == 0.0 {
            findings.push(Finding::ObsIntervalUnavailable {
                at: FindingRef::field("INTERVAL"),
            });
        } else if !usable_obs_interval_s(declared_s) {
            findings.push(Finding::ObsInvalidInterval {
                at: FindingRef::field("INTERVAL"),
                declared_s,
            });
        }
    }
    for (&system, codes) in &header.obs_codes {
        let mut seen = BTreeSet::new();
        for code in codes {
            if !is_valid_obs_code(system, code, header.version) {
                findings.push(Finding::ObsInvalidObsCode {
                    at: FindingRef::field("SYS / # / OBS TYPES"),
                    system,
                    code: code.clone(),
                });
            }
            if !seen.insert(code.as_str()) {
                findings.push(Finding::ObsDuplicateObsCode {
                    at: FindingRef::field("SYS / # / OBS TYPES"),
                    system,
                    code: code.clone(),
                });
            }
        }
    }
    if let Some(marker_type) = &header.marker_type {
        if !is_valid_marker_type(marker_type) {
            findings.push(Finding::ObsMarkerTypeIssue {
                at: FindingRef::field("MARKER TYPE"),
                marker_type: marker_type.clone(),
            });
        }
    }
    lint_identity_field(findings, "MARKER NAME", header.marker_name.as_deref(), 60);
    lint_identity_field(
        findings,
        "MARKER NUMBER",
        header.marker_number.as_deref(),
        20,
    );
    lint_identity_field(findings, "MARKER TYPE", header.marker_type.as_deref(), 20);
    lint_identity_field(findings, "OBSERVER", header.observer.as_deref(), 20);
    lint_identity_field(findings, "AGENCY", header.agency.as_deref(), 40);
    if let Some(receiver) = &header.receiver {
        lint_identity_field(findings, "REC #", Some(&receiver.number), 20);
        lint_identity_field(findings, "REC TYPE", Some(&receiver.receiver_type), 20);
        lint_identity_field(findings, "REC VERS", Some(&receiver.version), 20);
    }
    if let Some(antenna) = &header.antenna {
        lint_identity_field(findings, "ANT #", Some(&antenna.number), 20);
        lint_identity_field(findings, "ANT TYPE", Some(&antenna.antenna_type), 20);
    }
    for shift in &header.phase_shifts {
        if !header
            .obs_codes
            .get(&shift.system)
            .is_some_and(|codes| codes.iter().any(|code| code == &shift.code))
        {
            findings.push(Finding::ObsPhaseShiftUndeclaredCode {
                at: FindingRef::field("SYS / PHASE SHIFT"),
                system: shift.system,
                code: shift.code.clone(),
            });
        }
    }
    for factor in &header.scale_factors {
        if !matches!(factor.factor as i64, 1 | 10 | 100 | 1000) {
            findings.push(Finding::ObsScaleFactorIssue {
                at: FindingRef::field("SYS / SCALE FACTOR"),
                system: factor.system,
                code: None,
            });
        }
        for code in &factor.codes {
            if !header
                .obs_codes
                .get(&factor.system)
                .is_some_and(|codes| codes.iter().any(|declared| declared == code))
            {
                findings.push(Finding::ObsScaleFactorIssue {
                    at: FindingRef::field("SYS / SCALE FACTOR"),
                    system: factor.system,
                    code: Some(code.clone()),
                });
            }
        }
    }
    if let Some(pos) = header.approx_position_m {
        let radius = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        if radius != 0.0 && !(EARTH_FIXED_RADIUS_MIN_M..=EARTH_FIXED_RADIUS_MAX_M).contains(&radius)
        {
            findings.push(Finding::ObsImplausibleApproxPosition {
                at: FindingRef::field("APPROX POSITION XYZ"),
                radius_m: radius,
            });
        }
    }
    if let Some(delta) = header.antenna_delta_hen_m {
        for (idx, value) in delta.into_iter().enumerate() {
            if value.abs() > 100.0 {
                findings.push(Finding::ObsImplausibleAntennaDelta {
                    at: FindingRef::field("ANTENNA: DELTA H/E/N"),
                    component: idx,
                    value_m: value,
                });
            }
        }
    }
    for label in &header.unretained_header_labels {
        findings.push(Finding::ObsUnretainedHeader {
            at: FindingRef::field("header"),
            label: label.clone(),
        });
    }
}

fn lint_obs_body(obs: &RinexObs, findings: &mut Vec<Finding>) {
    if obs.skipped_records > 0 {
        findings.push(Finding::ObsSkippedRecords {
            at: FindingRef::default(),
            count: obs.skipped_records,
        });
    }
    if let Some(first) = first_normal_epoch(obs) {
        if let Some((declared, declared_scale)) = obs.header.time_of_first_obs {
            let observed_scale = obs_body_time_scale(obs);
            if !same_epoch_time(declared, first.epoch) || declared_scale != observed_scale {
                findings.push(Finding::ObsTimeOfFirstMismatch {
                    at: FindingRef::field("TIME OF FIRST OBS"),
                    declared,
                    declared_scale,
                    observed: first.epoch,
                    observed_scale,
                });
            }
        }
    }
    if let Some(last) = last_normal_epoch(obs) {
        if let Some((declared, declared_scale)) = obs.header.time_of_last_obs {
            let observed_scale = obs_body_time_scale(obs);
            if !same_epoch_time(declared, last.epoch) || declared_scale != observed_scale {
                findings.push(Finding::ObsTimeOfLastMismatch {
                    at: FindingRef::field("TIME OF LAST OBS"),
                    declared,
                    declared_scale,
                    observed: last.epoch,
                    observed_scale,
                });
            }
        }
    }
    lint_obs_counts(obs, findings);
    lint_obs_epoch_order(obs, findings);
    if let Some(observed) = dominant_interval_for_epochs(&obs.epochs) {
        if let Some(declared) = obs
            .header
            .interval_s
            .filter(|interval_s| usable_obs_interval_s(*interval_s))
        {
            if (declared - observed).abs() > 1.0e-6 {
                findings.push(Finding::ObsIntervalMismatch {
                    at: FindingRef::field("INTERVAL"),
                    declared_s: declared,
                    observed_s: observed,
                });
            }
        }
        lint_obs_gaps(obs, observed, findings);
    }
    lint_obs_glonass_slots(obs, findings);
    lint_obs_values(obs, findings);
}

fn lint_obs_epoch_order(obs: &RinexObs, findings: &mut Vec<Finding>) {
    let mut previous: Option<(usize, ObsEpochTime)> = None;
    let mut seen = BTreeMap::new();
    for (idx, epoch) in obs.epochs.iter().enumerate().filter(|(_, e)| e.flag <= 1) {
        let key = epoch_key(epoch.epoch);
        if let Some((_, prev)) = previous {
            if key < epoch_key(prev) {
                findings.push(Finding::ObsEpochOrder {
                    at: FindingRef::epoch(idx),
                    previous: prev,
                    current: epoch.epoch,
                });
            }
        }
        if seen.insert(key, idx).is_some() {
            findings.push(Finding::ObsDuplicateEpoch {
                at: FindingRef::epoch(idx),
                epoch: epoch.epoch,
            });
        }
        previous = Some((idx, epoch.epoch));
    }
}

fn lint_obs_counts(obs: &RinexObs, findings: &mut Vec<Finding>) {
    let body_counts = body_obs_counts(obs);
    let distinct_sats = body_counts.keys().copied().collect::<BTreeSet<_>>();
    if let Some(declared) = obs.header.n_satellites {
        let observed = distinct_sats.len();
        if declared != 0 && declared != observed {
            findings.push(Finding::ObsSatelliteCountMismatch {
                at: FindingRef::field("# OF SATELLITES"),
                declared,
                observed,
            });
        }
    }
    for (&sat, declared_counts) in &obs.header.prn_obs_counts {
        let Some(codes) = obs.header.obs_codes.get(&sat.system) else {
            continue;
        };
        let observed_counts = body_counts.get(&sat);
        for (idx, declared) in declared_counts.iter().enumerate() {
            let code = codes.get(idx).cloned().unwrap_or_default();
            let observed = observed_counts
                .and_then(|counts| counts.get(idx).copied())
                .unwrap_or(0);
            if declared.unwrap_or(0) != observed {
                findings.push(Finding::ObsPrnObsCountMismatch {
                    at: FindingRef {
                        satellite: Some(sat.to_string()),
                        field: Some("PRN / # OF OBS"),
                        ..FindingRef::default()
                    },
                    satellite: sat,
                    code,
                    declared: *declared,
                    observed,
                });
            }
        }
    }
}

fn lint_obs_glonass_slots(obs: &RinexObs, findings: &mut Vec<Finding>) {
    let has_glonass_codes = obs.header.obs_codes.contains_key(&GnssSystem::Glonass);
    if !has_glonass_codes {
        return;
    }
    let mut reported_missing = BTreeSet::new();
    for (&prn, &channel) in &obs.header.glonass_slots {
        if !crate::rinex_nav::valid_glonass_frequency_channel(i32::from(channel)) {
            if let Ok(satellite) = GnssSatelliteId::new(GnssSystem::Glonass, prn) {
                findings.push(Finding::ObsGlonassSlotIssue {
                    at: FindingRef {
                        satellite: Some(satellite.to_string()),
                        field: Some("GLONASS SLOT / FRQ #"),
                        ..FindingRef::default()
                    },
                    satellite,
                    issue: "invalid channel",
                });
            }
        }
    }
    for epoch in &obs.epochs {
        for sat in epoch
            .sats
            .keys()
            .filter(|sat| sat.system == GnssSystem::Glonass)
        {
            if !obs.header.glonass_slots.contains_key(&sat.prn) && reported_missing.insert(*sat) {
                findings.push(Finding::ObsGlonassSlotIssue {
                    at: FindingRef {
                        satellite: Some(sat.to_string()),
                        field: Some("GLONASS SLOT / FRQ #"),
                        ..FindingRef::default()
                    },
                    satellite: *sat,
                    issue: "missing slot",
                });
            }
        }
    }
}

fn lint_obs_values(obs: &RinexObs, findings: &mut Vec<Finding>) {
    for (epoch_index, epoch) in obs.epochs.iter().enumerate() {
        if epoch.flag > 1 {
            findings.push(Finding::ObsEventEpoch {
                at: FindingRef::epoch(epoch_index),
                flag: epoch.flag,
            });
            if epoch.special_record_count > 0 {
                findings.push(Finding::ObsEventSpecialRecords {
                    at: FindingRef::epoch(epoch_index),
                    count: epoch.special_record_count,
                });
            }
            continue;
        }
        if epoch.declared_record_count != epoch.sats.len() {
            findings.push(Finding::ObsEpochSatCountMismatch {
                at: FindingRef::epoch(epoch_index),
                declared: epoch.declared_record_count,
                retained: epoch.sats.len(),
            });
        }
        for (&sat, values) in &epoch.sats {
            let all_blank = values.iter().all(|value| value.value.is_none());
            if all_blank {
                findings.push(Finding::ObsEmptySatelliteRecord {
                    at: FindingRef::sat(epoch_index, sat),
                });
            }
            let codes = obs.header.obs_codes.get(&sat.system).map(Vec::as_slice);
            for (idx, value) in values.iter().enumerate() {
                let code = codes
                    .and_then(|codes| codes.get(idx))
                    .map_or("", String::as_str);
                if code.starts_with('C') {
                    if let Some(v) = value.value {
                        if !(15_000_000.0..=50_000_000.0).contains(&v) {
                            findings.push(Finding::ObsPseudorangeOutOfRange {
                                at: FindingRef::sat(epoch_index, sat),
                                code: code.to_string(),
                                value_m: v,
                            });
                        }
                    }
                }
                if let Some(lli) = value.lli {
                    if lli > 7 {
                        findings.push(Finding::ObsLossOfLockOutOfRange {
                            at: FindingRef::sat(epoch_index, sat),
                            code: code.to_string(),
                            lli,
                        });
                    }
                }
            }
        }
    }
}

fn lint_obs_gaps(obs: &RinexObs, interval_s: f64, findings: &mut Vec<Finding>) {
    let mut previous: Option<ObsEpochTime> = None;
    for (idx, epoch) in obs.epochs.iter().enumerate().filter(|(_, e)| e.flag <= 1) {
        if let Some(prev) = previous {
            let gap = obs_epoch_seconds(epoch.epoch) - obs_epoch_seconds(prev);
            if gap > interval_s * 1.5 {
                findings.push(Finding::ObsEpochGap {
                    at: FindingRef::epoch(idx),
                    gap_s: gap,
                    interval_s,
                });
            }
        }
        previous = Some(epoch.epoch);
    }
}

fn body_obs_counts(obs: &RinexObs) -> BTreeMap<GnssSatelliteId, Vec<usize>> {
    let mut counts: BTreeMap<GnssSatelliteId, Vec<usize>> = BTreeMap::new();
    for epoch in obs.epochs.iter().filter(|epoch| epoch.flag <= 1) {
        for (&sat, values) in &epoch.sats {
            let Some(codes) = obs.header.obs_codes.get(&sat.system) else {
                continue;
            };
            let entry = counts.entry(sat).or_insert_with(|| vec![0; codes.len()]);
            if entry.len() < codes.len() {
                entry.resize(codes.len(), 0);
            }
            for (idx, value) in values.iter().enumerate() {
                if value.value.is_some() {
                    if let Some(count) = entry.get_mut(idx) {
                        *count += 1;
                    }
                }
            }
        }
    }
    counts
}

fn is_earth_fixed_marker_type(marker_type: &str) -> bool {
    matches!(
        marker_type.trim(),
        "" | "GEODETIC" | "NON_GEODETIC" | "FIXED_BUOY"
    )
}

fn is_valid_marker_type(marker_type: &str) -> bool {
    matches!(
        marker_type.trim(),
        "GEODETIC"
            | "NON_GEODETIC"
            | "NON_PHYSICAL"
            | "SPACEBORNE"
            | "AIRBORNE"
            | "WATER_CRAFT"
            | "GROUND_CRAFT"
            | "FIXED_BUOY"
            | "FLOATING_BUOY"
            | "FLOATING_ICE"
            | "GLACIER"
            | "BALLOON"
            | "ANIMAL"
            | "HUMAN"
    )
}

fn lint_identity_field(
    findings: &mut Vec<Finding>,
    label: &'static str,
    value: Option<&str>,
    max_width: usize,
) {
    let Some(value) = value else {
        return;
    };
    if value.len() > max_width || !value.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        findings.push(Finding::ObsIdentityFieldIssue {
            at: FindingRef::field(label),
            label,
            value: value.to_string(),
        });
    }
}

fn validate_text_field(
    field: &'static str,
    value: &str,
    max_width: usize,
    allow_empty: bool,
) -> std::result::Result<(), HeaderEditError> {
    let trimmed = value.trim();
    if !allow_empty && trimmed.is_empty() {
        return Err(HeaderEditError::InvalidField {
            field,
            reason: "must not be empty",
        });
    }
    if trimmed.len() > max_width {
        return Err(HeaderEditError::InvalidField {
            field,
            reason: "too wide for RINEX field",
        });
    }
    if !trimmed.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return Err(HeaderEditError::InvalidField {
            field,
            reason: "must be printable ASCII",
        });
    }
    Ok(())
}

fn validate_antenna_delta(
    field: &'static str,
    value: f64,
) -> std::result::Result<(), HeaderEditError> {
    if !value.is_finite() {
        return Err(HeaderEditError::InvalidField {
            field,
            reason: "must be finite",
        });
    }
    if value.abs() > 100.0 {
        return Err(HeaderEditError::InvalidField {
            field,
            reason: "magnitude exceeds 100 m",
        });
    }
    Ok(())
}

fn push_edit(
    applied: &mut Vec<AppliedEdit>,
    field: &'static str,
    old_value: Option<String>,
    new_value: Option<String>,
    warning: Option<String>,
) {
    if old_value != new_value || warning.is_some() {
        applied.push(AppliedEdit {
            field,
            old_value,
            new_value,
            warning,
        });
    }
}

fn format_receiver(value: &ReceiverInfo) -> String {
    format!("{}/{}/{}", value.number, value.receiver_type, value.version)
}

fn format_antenna(value: &AntennaInfo) -> String {
    format!("{}/{}", value.number, value.antenna_type)
}

fn nav_findings(records: &[BroadcastRecord]) -> Vec<Finding> {
    let mut findings = Vec::new();
    lint_nav_duplicates(records, &mut findings);
    lint_nav_order(records, &mut findings);
    lint_nav_plausibility(records, &mut findings);
    findings
}

fn nav_scope_tallies(text: &str) -> BTreeMap<String, usize> {
    let mut body = false;
    let mut version_major = 3_u8;
    let mut tallies = BTreeMap::new();
    for line in text.lines() {
        if line.contains("RINEX VERSION / TYPE")
            && line.get(0..9).unwrap_or("").trim().starts_with('4')
        {
            version_major = 4;
        }
        if !body {
            if line.contains("END OF HEADER") {
                body = true;
            }
            continue;
        }
        if version_major >= 4 {
            if let Some(rest) = line.strip_prefix('>') {
                let fields: Vec<_> = rest.split_whitespace().collect();
                if fields.len() < 3 {
                    continue;
                }
                let frame = fields[0];
                let sv = fields[1];
                let msg = fields[2];
                let system = sv.chars().next();
                let class = if frame != "EPH" {
                    Some(format!("v4 {frame} frame"))
                } else if !matches!(system, Some('G' | 'E' | 'C')) {
                    Some(format!(
                        "unsupported constellation {}",
                        system.unwrap_or('?')
                    ))
                } else if !matches!(msg, "LNAV" | "INAV" | "FNAV" | "D1" | "D2") {
                    Some(format!("unsupported message {msg}"))
                } else {
                    None
                };
                if let Some(class) = class {
                    *tallies.entry(class).or_default() += 1;
                }
            }
        } else if is_nav_record_start_text(line) {
            let system = line.as_bytes()[0] as char;
            if !matches!(system, 'G' | 'E' | 'C') {
                *tallies
                    .entry(format!("unsupported constellation {system}"))
                    .or_default() += 1;
            }
        }
    }
    tallies
}

fn is_nav_record_start_text(line: &str) -> bool {
    let b = line.as_bytes();
    b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1].is_ascii_digit() && b[2].is_ascii_digit()
}

fn lint_nav_duplicates(records: &[BroadcastRecord], findings: &mut Vec<Finding>) {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (idx, record) in records.iter().enumerate() {
        let key = nav_identity(record);
        if let Some(first_idx) = seen.get(&key).copied() {
            findings.push(Finding::NavDuplicateRecord {
                at: FindingRef::epoch(idx),
                satellite: record.satellite_id,
                same_payload: records[first_idx] == *record,
            });
        } else {
            seen.insert(key, idx);
        }
    }
}

fn lint_nav_order(records: &[BroadcastRecord], findings: &mut Vec<Finding>) {
    if records
        .windows(2)
        .any(|pair| nav_sort_key(&pair[0]) > nav_sort_key(&pair[1]))
    {
        findings.push(Finding::NavUnsortedRecords {
            at: FindingRef::default(),
        });
    }
}

fn lint_nav_plausibility(records: &[BroadcastRecord], findings: &mut Vec<Finding>) {
    let mut unhealthy: BTreeMap<GnssSystem, usize> = BTreeMap::new();
    for (idx, record) in records.iter().enumerate() {
        if !(0.0..=0.1).contains(&record.elements.e) {
            findings.push(Finding::NavImplausibleRecord {
                at: FindingRef::epoch(idx),
                satellite: record.satellite_id,
                field: "eccentricity",
                value: record.elements.e,
            });
        }
        if !(4_000.0..=8_000.0).contains(&record.elements.sqrt_a) {
            findings.push(Finding::NavImplausibleRecord {
                at: FindingRef::epoch(idx),
                satellite: record.satellite_id,
                field: "sqrt_a",
                value: record.elements.sqrt_a,
            });
        }
        if record.sv_health != 0.0 {
            *unhealthy.entry(record.satellite_id.system).or_default() += 1;
        }
    }
    for (system, count) in unhealthy {
        findings.push(Finding::NavUnhealthyRecords {
            at: FindingRef::default(),
            system,
            count,
        });
    }
}

fn repair_obs_order_and_duplicates(obs: &mut RinexObs, actions: &mut Vec<RepairAction>) {
    if obs.epochs.iter().any(|epoch| epoch.flag > 1) {
        return;
    }
    let before = obs.epochs.clone();
    obs.epochs.sort_by_key(|epoch| epoch_key(epoch.epoch));
    let mut merged: Vec<ObsEpoch> = Vec::new();
    let mut discarded = Vec::new();
    for epoch in obs.epochs.drain(..) {
        if let Some(last) = merged.last_mut() {
            if same_epoch_time(last.epoch, epoch.epoch) {
                for (sat, values) in epoch.sats {
                    match last.sats.entry(sat) {
                        std::collections::btree_map::Entry::Vacant(slot) => {
                            slot.insert(values);
                        }
                        std::collections::btree_map::Entry::Occupied(_) => {
                            discarded.push(sat);
                        }
                    }
                }
                last.declared_record_count = last.sats.len();
                continue;
            }
        }
        merged.push(epoch);
    }
    obs.epochs = merged;
    if obs.epochs != before {
        let discarded = discarded
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        actions.push(RepairAction {
            id: "A3",
            message: format!(
                "sorted epochs and merged duplicate epochs, discarded duplicate satellite rows [{discarded}]"
            ),
        });
    }
}

fn repair_obs_times(obs: &mut RinexObs, options: &RepairOptions, actions: &mut Vec<RepairAction>) {
    let Some(first) = first_normal_epoch(obs).map(|epoch| epoch.epoch) else {
        return;
    };
    let scale = obs_body_time_scale(obs);
    if obs
        .header
        .time_of_first_obs
        .is_none_or(|(declared, declared_scale)| {
            !same_epoch_time(declared, first) || declared_scale != scale
        })
    {
        obs.header.time_of_first_obs = Some((first, scale));
        actions.push(RepairAction {
            id: "A4",
            message: "recomputed TIME OF FIRST OBS".to_string(),
        });
    }
    let Some(last) = last_normal_epoch(obs).map(|epoch| epoch.epoch) else {
        return;
    };
    // TIME OF FIRST OBS is the time-system authority (RINEX 3.05); a
    // disagreeing TIME OF LAST OBS is rewritten to match it.
    if options.set_time_of_last_obs
        || obs
            .header
            .time_of_last_obs
            .is_some_and(|(declared, declared_scale)| {
                !same_epoch_time(declared, last) || declared_scale != scale
            })
    {
        obs.header.time_of_last_obs = Some((last, scale));
        actions.push(RepairAction {
            id: "A4",
            message: "recomputed TIME OF LAST OBS".to_string(),
        });
    }
}

fn repair_obs_counts(obs: &mut RinexObs, options: &RepairOptions, actions: &mut Vec<RepairAction>) {
    if !options.set_obs_counts
        && obs.header.n_satellites.is_none()
        && obs.header.prn_obs_counts.is_empty()
    {
        return;
    }
    let counts = body_obs_counts(obs);
    obs.header.n_satellites = Some(counts.len());
    obs.header.prn_obs_counts = counts
        .into_iter()
        .map(|(sat, values)| (sat, values.into_iter().map(Some).collect()))
        .collect();
    actions.push(RepairAction {
        id: "A5",
        message: "recomputed observation count headers".to_string(),
    });
}

fn repair_obs_file_stamp(
    obs: &mut RinexObs,
    options: &RepairOptions,
    actions: &mut Vec<RepairAction>,
) {
    if let Some(stamp) = &options.file_stamp {
        if obs.header.program_run_by_date.as_ref() != Some(stamp) {
            obs.header.program_run_by_date = Some(stamp.clone());
            actions.push(RepairAction {
                id: "A8",
                message: "set PGM / RUN BY / DATE".to_string(),
            });
        }
    }
}

fn repair_obs_unsupported_records(
    obs: &mut RinexObs,
    options: &RepairOptions,
    actions: &mut Vec<RepairAction>,
) {
    if !options.drop_unsupported {
        return;
    }
    let mut dropped = 0_usize;
    for epoch in &mut obs.epochs {
        if epoch.flag > 1 && epoch.special_record_count > 0 {
            dropped += epoch.special_record_count;
            epoch.special_record_count = 0;
            epoch.declared_record_count = 0;
        }
    }
    if dropped > 0 {
        actions.push(RepairAction {
            id: "OBS-B11",
            message: format!("dropped {dropped} event special records"),
        });
    }
    let labels = std::mem::take(&mut obs.header.unretained_header_labels);
    if !labels.is_empty() {
        actions.push(RepairAction {
            id: "OBS-H90",
            message: format!("dropped {} unretained header records", labels.len()),
        });
    }
}

fn repair_obs_interval(obs: &mut RinexObs, actions: &mut Vec<RepairAction>) {
    let Some(interval) = dominant_interval_for_epochs(&obs.epochs) else {
        if obs
            .header
            .interval_s
            .is_some_and(|declared| !usable_obs_interval_s(declared))
        {
            obs.header.interval_s = None;
            actions.push(RepairAction {
                id: "A6",
                message: "removed unusable INTERVAL because no cadence could be inferred"
                    .to_string(),
            });
        }
        return;
    };
    if obs.header.interval_s.is_none_or(|declared| {
        !usable_obs_interval_s(declared) || (declared - interval).abs() > 1.0e-6
    }) {
        obs.header.interval_s = Some(interval);
        actions.push(RepairAction {
            id: "A6",
            message: format!("set INTERVAL to {interval:.3} seconds"),
        });
    }
}

fn repair_obs_empty_records(obs: &mut RinexObs, actions: &mut Vec<RepairAction>) {
    let mut dropped = 0_usize;
    for epoch in &mut obs.epochs {
        let before = epoch.sats.len();
        epoch
            .sats
            .retain(|_, values| values.iter().any(|value| value.value.is_some()));
        let removed = before - epoch.sats.len();
        if removed > 0 {
            epoch.declared_record_count = epoch.sats.len();
        }
        dropped += removed;
    }
    if dropped > 0 {
        actions.push(RepairAction {
            id: "A7",
            message: format!("dropped {dropped} empty satellite records"),
        });
    }
}

fn repair_nav_duplicates(records: &mut Vec<BroadcastRecord>, actions: &mut Vec<RepairAction>) {
    let mut seen: BTreeMap<String, Vec<BroadcastRecord>> = BTreeMap::new();
    let mut out = Vec::with_capacity(records.len());
    let mut dropped = 0_usize;
    for record in records.drain(..) {
        let key = nav_identity(&record);
        let family = seen.entry(key).or_default();
        if family.contains(&record) {
            dropped += 1;
        } else {
            family.push(record);
            out.push(record);
        }
    }
    *records = out;
    if dropped > 0 {
        actions.push(RepairAction {
            id: "A11",
            message: format!("dropped {dropped} identical duplicate NAV records"),
        });
    }
}

fn repair_nav_order(records: &mut [BroadcastRecord], actions: &mut Vec<RepairAction>) {
    let before = records.to_vec();
    records.sort_by_key(nav_sort_key);
    if records != before {
        actions.push(RepairAction {
            id: "A12",
            message: "sorted NAV records".to_string(),
        });
    }
}

fn published_obs_version(version: f64) -> Option<()> {
    let scaled = (version * 100.0).round() as i64;
    matches!(
        scaled,
        200 | 201 | 202 | 210 | 211 | 212 | 300 | 301 | 302 | 303 | 304 | 305 | 400 | 401 | 402
    )
    .then_some(())
}

fn is_valid_obs_code(system: GnssSystem, code: &str, version: f64) -> bool {
    let mut chars = code.chars();
    let Some(kind) = chars.next() else {
        return false;
    };
    let Some(band) = chars.next() else {
        return false;
    };
    let Some(attr) = chars.next() else {
        return false;
    };
    if chars.next().is_some() || !"CLDSX".contains(kind) || !band.is_ascii_digit() {
        return false;
    }
    obs_code_band_attr_allowed(system, band, attr, version)
}

fn obs_code_band_attr_allowed(system: GnssSystem, band: char, attr: char, _version: f64) -> bool {
    match system {
        // RINEX 3.05 Tables 14-20 plus RINEX 4.02 additions used by the
        // committed fixtures. Band checks are explicit so GPS C9C is rejected.
        GnssSystem::Gps => match band {
            '1' => "CWPYMSLXN".contains(attr),
            '2' => "CWPYMSLDXN".contains(attr),
            '5' => "IQX".contains(attr),
            _ => false,
        },
        GnssSystem::Glonass => match band {
            '1' | '2' => "CP".contains(attr),
            '3' => "IQX".contains(attr),
            '4' | '6' => "ABX".contains(attr),
            _ => false,
        },
        GnssSystem::Galileo => match band {
            '1' => "ABCXZ".contains(attr),
            '5' | '7' | '8' => "IQX".contains(attr),
            '6' => "ABCXZ".contains(attr),
            _ => false,
        },
        GnssSystem::BeiDou => match band {
            '1' => "DPXAN".contains(attr),
            '2' => "IQX".contains(attr),
            '5' => "DPX".contains(attr),
            '6' => "IQX".contains(attr),
            '7' => "IQXDPZ".contains(attr),
            '8' => "DPX".contains(attr),
            _ => false,
        },
        GnssSystem::Qzss => match band {
            '1' => "CSLXZ".contains(attr),
            '2' => "SLX".contains(attr),
            '5' => "IQX".contains(attr),
            '6' => "SLXEZ".contains(attr),
            _ => false,
        },
        GnssSystem::Navic => match band {
            '5' | '9' => "ABCX".contains(attr),
            _ => false,
        },
        GnssSystem::Sbas => match band {
            '1' => "C".contains(attr),
            '5' => "IQX".contains(attr),
            _ => false,
        },
    }
}

fn first_normal_epoch(obs: &RinexObs) -> Option<&ObsEpoch> {
    obs.epochs.iter().find(|epoch| epoch.flag <= 1)
}

fn last_normal_epoch(obs: &RinexObs) -> Option<&ObsEpoch> {
    obs.epochs.iter().rev().find(|epoch| epoch.flag <= 1)
}

/// RINEX 3.05 Table A2: TIME OF FIRST OBS carries the file's time system, so
/// it is authoritative; TIME OF LAST OBS must agree with it and is only
/// consulted when TIME OF FIRST OBS is absent.
fn obs_body_time_scale(obs: &RinexObs) -> TimeScale {
    match (obs.header.time_of_first_obs, obs.header.time_of_last_obs) {
        (Some((_, scale)), _) | (None, Some((_, scale))) => scale,
        _ => TimeScale::Gpst,
    }
}

fn dominant_interval_for_epochs(epochs: &[ObsEpoch]) -> Option<f64> {
    let normal: Vec<_> = epochs
        .iter()
        .filter(|epoch| epoch.flag <= 1)
        .map(|epoch| epoch.epoch)
        .collect();
    dominant_obs_interval_s(&normal)
}

fn epoch_key(epoch: ObsEpochTime) -> (i32, u8, u8, u8, u8, i64) {
    (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        (epoch.second * 10_000_000.0).round() as i64,
    )
}

fn same_epoch_time(a: ObsEpochTime, b: ObsEpochTime) -> bool {
    epoch_key(a) == epoch_key(b)
}

fn nav_identity(record: &BroadcastRecord) -> String {
    format!(
        "{}:{:?}:{}:{:016x}:{}",
        record.satellite_id,
        record.message,
        record.toc.week,
        record.toc.tow_s.to_bits(),
        record.issue_of_data.issue
    )
}

fn nav_sort_key(record: &BroadcastRecord) -> (GnssSystem, u8, u32, u64, u8) {
    (
        record.satellite_id.system,
        record.satellite_id.prn,
        record.toc.week,
        record.toc.tow_s.to_bits(),
        nav_message_rank(record.message),
    )
}

const fn nav_message_rank(message: NavMessage) -> u8 {
    match message {
        NavMessage::GpsLnav => 0,
        NavMessage::GpsCnav => 1,
        NavMessage::GpsCnav2 => 2,
        NavMessage::QzssLnav => 3,
        NavMessage::QzssCnav => 4,
        NavMessage::QzssCnav2 => 5,
        NavMessage::GalileoInav => 6,
        NavMessage::GalileoFnav => 7,
        NavMessage::BeidouD1 => 8,
        NavMessage::BeidouD2 => 9,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
