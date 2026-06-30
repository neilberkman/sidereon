//! GNSS constellation identity catalog and validation helpers.
//!
//! This is a data/catalog layer: it builds normalized satellite identity
//! records from public sources and compares those records with GNSS products.
//! It does not alter positioning solves or infer application-specific health
//! rules. It is deterministic and performs no network access; fetching the
//! source bytes is the caller's (binding's) job.
//!
//! GPS, Galileo, GLONASS, BeiDou, and QZSS are supported. The base source for
//! every system is a CelesTrak OMM/JSON group (`gps-ops`, `galileo`, `glo-ops`,
//! `beidou`, and the QZSS members of the `gnss` group); the within-system PRN /
//! slot is parsed from `OBJECT_NAME` and rendered as the SP3/RINEX id (`"G13"`,
//! `"E07"`, `"R13"`, `"C19"`, `"J02"`) via [`gnss_sp3_id`]. Each constellation
//! names its satellites differently, so [`from_celestrak_omm`] dispatches on
//! [`GnssSystem`] to a per-system identity adapter:
//!
//! - **GPS** — `(PRN nn)` in the object name is the PRN directly.
//! - **BeiDou** — `(Cnn)` in the object name is the PRN directly.
//! - **QZSS** — `(QZSS/PRN nnn)` carries the broadcast PRN (193..=201); the
//!   RINEX slot is `nnn - 192` (`J01`..`J09`), per RINEX 3.0x.
//! - **Galileo** — the object name is the `GSATdddd` build id, which carries no
//!   PRN; the SVID/PRN is resolved from the published GSAT->SVID table
//!   [`galileo_prn_for_gsat`].
//! - **GLONASS** — the parenthesized number is the GLONASS (Uragan) number, not
//!   the orbital slot; the slot is resolved from the published slot table
//!   [`glonass_slot_for_number`], and the FDMA frequency-channel number (which
//!   is not in OMM at all) from [`glonass_fdma_channel`].
//!
//! NAVCEN's GPS constellation status page can be parsed and merged as an
//! optional overlay for SVN and NANU usability details. There is no clean
//! equivalent health oracle for the other systems, so usability overlays are
//! GPS-only; the OMM identity round-trip and the GLONASS FDMA check are the
//! gates for the rest.
//!
//! The OMM input is the canonical [`Omm`](crate::astro::omm::Omm) produced by
//! the core OMM parser (`crate::astro::omm::{parse_json, parse_json_array}`):
//! this module does not re-parse OMM from scratch, it reads `OBJECT_NAME` and
//! `NORAD_CAT_ID` off already-parsed records.
//!
//! ```
//! use sidereon_core::constellation::{to_csv, BoolStyle, Record, RecordSource};
//! use sidereon_core::GnssSystem;
//!
//! let record = Record {
//!     system: GnssSystem::Gps,
//!     prn: 3,
//!     svn: None,
//!     norad_id: 40294,
//!     sp3_id: "G03".to_string(),
//!     fdma_channel: None,
//!     active: true,
//!     usable: true,
//!     source: RecordSource::default(),
//! };
//! assert_eq!(
//!     to_csv(&[record], BoolStyle::Lower),
//!     "prn,norad_cat_id,active,sp3_id\n3,40294,true,G03\n"
//! );
//! ```

use crate::astro::omm::Omm;
use crate::ephemeris::Sp3;
use crate::id::GnssSystem;
use core::fmt::{self, Write as _};

/// The CelesTrak GP group each system's identity base is fetched from.
///
/// These mirror the live CelesTrak group names (`gps-ops`, `galileo`, `glo-ops`,
/// `beidou`); QZSS has no dedicated group and is carried in the combined `gnss`
/// group, so its records are filtered out of that feed by the caller.
const fn celestrak_group(system: GnssSystem) -> &'static str {
    match system {
        GnssSystem::Gps => "gps-ops",
        GnssSystem::Galileo => "galileo",
        GnssSystem::Glonass => "glo-ops",
        GnssSystem::BeiDou => "beidou",
        GnssSystem::Qzss => "gnss",
        GnssSystem::Navic | GnssSystem::Sbas => "gnss",
    }
}

/// Failure modes of the constellation catalog builders.
///
/// Mirrors the typed error pattern used by the core parsers (for example
/// `astro::omm::OmmError`): a small enum with a `Display` and `std::error::Error`
/// implementation, never a panic on malformed input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstellationError {
    /// A CelesTrak `OBJECT_NAME` did not contain a parseable `(PRN nn)` block,
    /// or the OMM carried no object name at all. Holds the offending name.
    MissingPrn(Option<String>),
    /// The NAVCEN status bytes were not valid UTF-8.
    NavcenNotUtf8,
    /// The NAVCEN status HTML contained no GPS constellation rows.
    NavcenNoRows,
    /// A required NAVCEN integer cell could not be parsed. Holds the field name
    /// and the offending text.
    NavcenBadField {
        /// The NAVCEN field whose cell failed to parse (for example `gps-prn`).
        field: &'static str,
        /// The raw cell text that failed to parse.
        value: String,
    },
    /// A catalog failed SP3 validation. Holds a description of the findings.
    Sp3Validation(String),
}

impl fmt::Display for ConstellationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstellationError::MissingPrn(Some(name)) => {
                write!(f, "CelesTrak OBJECT_NAME has no PRN: {name:?}")
            }
            ConstellationError::MissingPrn(None) => {
                write!(f, "CelesTrak record has no OBJECT_NAME")
            }
            ConstellationError::NavcenNotUtf8 => write!(f, "NAVCEN bytes are not valid UTF-8"),
            ConstellationError::NavcenNoRows => write!(f, "NAVCEN HTML has no GPS rows"),
            ConstellationError::NavcenBadField { field, value } => {
                write!(f, "NAVCEN field {field} has invalid integer {value:?}")
            }
            ConstellationError::Sp3Validation(msg) => {
                write!(f, "GNSS catalog failed SP3 validation: {msg}")
            }
        }
    }
}

impl std::error::Error for ConstellationError {}

/// Per-source provenance kept on a [`Record`].
///
/// `active` in a record means the satellite is present in the base identity
/// source. `usable` is an advisory health flag; for the current GPS path it is
/// `true` unless a compatible merged NAVCEN row carries an active NANU that
/// marks the PRN unusable or decommissioned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordSource {
    /// CelesTrak `gps-ops` identity provenance.
    pub celestrak: Option<CelestrakSource>,
    /// NAVCEN overlay that was merged into this record.
    pub navcen: Option<NavcenSource>,
    /// A NAVCEN row that matched the PRN but was not merged because its block
    /// type was incompatible with the CelesTrak identity (a PRN transition).
    pub navcen_conflict: Option<NavcenSource>,
}

/// CelesTrak `gps-ops` provenance fields preserved on a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CelestrakSource {
    /// CelesTrak GP group the record came from (`gps-ops`).
    pub group: String,
    /// The OMM `OBJECT_NAME`.
    pub object_name: Option<String>,
    /// The OMM `OBJECT_ID` (international designator).
    pub object_id: Option<String>,
    /// The OMM `EPOCH`, ISO-8601.
    pub epoch: Option<String>,
    /// Block type parsed from the object name (`IIF`, `IIR`, `IIR-M`, `III`).
    pub block_type: Option<String>,
}

/// NAVCEN status provenance fields preserved on a record or conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavcenSource {
    /// Space Vehicle Number.
    pub svn: Option<u16>,
    /// Block type as reported by NAVCEN.
    pub block_type: Option<String>,
    /// Orbital plane letter.
    pub plane: Option<String>,
    /// Slot within the plane.
    pub slot: Option<String>,
    /// Clock type.
    pub clock: Option<String>,
    /// NANU type code (for example `FCSTSUMM`, `UNUSABLE`, `DECOM`).
    pub nanu_type: Option<String>,
    /// NANU subject line.
    pub nanu_subject: Option<String>,
    /// Whether the row carried an active NANU.
    pub active_nanu: bool,
}

/// A normalized GNSS satellite identity record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The constellation. GPS today; the type is system-tagged for extension.
    pub system: GnssSystem,
    /// The within-constellation PRN.
    pub prn: u16,
    /// Space Vehicle Number, when known (CelesTrak alone leaves this `None`).
    pub svn: Option<u16>,
    /// NORAD catalog id.
    pub norad_id: u32,
    /// Canonical SP3/RINEX satellite token (`G03`).
    pub sp3_id: String,
    /// GLONASS FDMA L1/L2 frequency-channel number (`k`, in `-7..=6`), `None`
    /// for the CDMA constellations. This is the one identity datum that is not
    /// present in any OMM feed; it is resolved from the orbital slot via the
    /// published IGS/MCC slot-channel table ([`glonass_fdma_channel`]).
    pub fdma_channel: Option<i8>,
    /// Present in the base identity source.
    pub active: bool,
    /// Advisory usability flag.
    pub usable: bool,
    /// Source provenance.
    pub source: RecordSource,
}

/// A parsed row from NAVCEN's GPS constellation status table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavcenStatus {
    /// The constellation (GPS).
    pub system: GnssSystem,
    /// The within-constellation PRN.
    pub prn: u16,
    /// Space Vehicle Number, when present.
    pub svn: Option<u16>,
    /// Whether the satellite is usable per the active NANU (if any).
    pub usable: bool,
    /// Whether the row carried an active NANU.
    pub active_nanu: bool,
    /// NANU type code.
    pub nanu_type: Option<String>,
    /// NANU subject line.
    pub nanu_subject: Option<String>,
    /// Orbital plane letter.
    pub plane: Option<String>,
    /// Slot within the plane.
    pub slot: Option<String>,
    /// Block type.
    pub block_type: Option<String>,
    /// Clock type.
    pub clock: Option<String>,
}

/// Validation report for a constellation catalog.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Validation {
    /// Active+usable catalog SP3 ids absent from the compared product.
    pub missing_sp3_ids: Vec<String>,
    /// `(system, PRN)` pairs that appear in more than one record. Keyed by
    /// system so a legitimate multi-system catalog (GPS PRN 1 and Galileo PRN 1)
    /// is not reported as a false duplicate.
    pub duplicate_prns: Vec<(GnssSystem, u16)>,
    /// NORAD ids that appear in more than one record.
    pub duplicate_norad_ids: Vec<u32>,
    /// `(system, PRN)` pairs that are inactive or unusable.
    pub inactive_unusable_prns: Vec<(GnssSystem, u16)>,
    /// SP3 ids present in the product but absent from the active+usable catalog.
    pub extra_sp3_ids: Vec<String>,
}

/// A single field change on a PRN that exists in both diffed snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldChange<T> {
    /// The constellation.
    pub system: GnssSystem,
    /// The PRN.
    pub prn: u16,
    /// The value in the previous snapshot.
    pub from: T,
    /// The value in the current snapshot.
    pub to: T,
}

/// Change report between two catalog snapshots, keyed by `(system, prn)`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Diff {
    /// PRNs present only in the current snapshot.
    pub added: Vec<Record>,
    /// PRNs present only in the previous snapshot.
    pub removed: Vec<Record>,
    /// NORAD id reassignments on a held PRN.
    pub norad_reassigned: Vec<FieldChange<u32>>,
    /// SP3 id changes on a held PRN.
    pub sp3_id_changed: Vec<FieldChange<String>>,
    /// SVN changes on a held PRN.
    pub svn_changed: Vec<FieldChange<Option<u16>>>,
    /// GLONASS FDMA frequency-channel corrections on a held slot.
    pub fdma_channel_changed: Vec<FieldChange<Option<i8>>>,
    /// Activity flips on a held PRN.
    pub activity_changed: Vec<FieldChange<bool>>,
    /// Usability flips on a held PRN.
    pub usability_changed: Vec<FieldChange<bool>>,
}

/// How the CSV `active` column renders booleans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoolStyle {
    /// `true` / `false` (the conventional CSV form).
    #[default]
    Lower,
    /// `True` / `False` (for a consumer that reads Python booleans).
    Title,
}

/// Render the canonical SP3/RINEX satellite token for a constellation + PRN
/// (`(Gps, 7)` -> `"G07"`, `(Glonass, 13)` -> `"R13"`).
#[must_use]
pub fn gnss_sp3_id(system: GnssSystem, prn: u16) -> String {
    format!("{}{prn:02}", system.letter())
}

/// The within-system identity an OMM `OBJECT_NAME` resolves to for a system.
struct Identity {
    /// The within-constellation PRN / orbital slot (the `nn` in the SP3 token).
    prn: u16,
    /// GLONASS FDMA channel, when applicable.
    fdma_channel: Option<i8>,
}

/// An OMM record that [`from_celestrak_omm_lenient`] could not resolve to a
/// [`Record`] for the requested system.
///
/// Carries the entry's identity (not just a count) so the caller can triage why
/// it was skipped: a record from another constellation in a combined feed (QZSS,
/// or anything else, living in the `gnss` group), versus a satellite of the
/// requested system whose name does not yet resolve (a freshly launched
/// GLONASS/Galileo not yet in the published slot/SVID table).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedOmm {
    /// The OMM `OBJECT_NAME`, when present.
    pub object_name: Option<String>,
    /// The OMM `NORAD_CAT_ID`.
    pub norad_id: u32,
}

/// The result of a lenient constellation catalog build: the records that
/// resolved, plus the OMM entries that did not.
///
/// Mirrors the partial-success convention of
/// [`crate::astro::omm::OmmArray`] / [`crate::astro::sgp4::TleFile`], but keeps
/// the skipped entries' identities (rather than a bare count) because, unlike a
/// malformed JSON element, an unresolved OMM here carries a meaningful
/// `OBJECT_NAME`/`NORAD_CAT_ID` the caller needs to act on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Catalog {
    /// Records built from resolvable OMM entries, sorted by `(system, prn)`.
    pub records: Vec<Record>,
    /// Entries whose `OBJECT_NAME` did not resolve to a PRN for the requested
    /// system, in input order.
    pub skipped: Vec<SkippedOmm>,
}

/// Build records for `system` from already-parsed CelesTrak OMM records, failing
/// on the first unresolvable entry.
///
/// The OMM source carries no SVN, so records built from it alone have
/// `svn: None`; GPS can be enriched afterwards with [`merge_navcen`]. Records
/// are returned sorted by `(system, prn)`. Fails with
/// [`ConstellationError::MissingPrn`] when an `OBJECT_NAME` cannot be resolved to
/// a PRN for `system` (an unparseable name, or a GLONASS/Galileo satellite not
/// in the published slot/SVID table).
///
/// Use this for a single-system feed already filtered to `system` (`gps-ops`,
/// `glo-ops`, ...), where an unresolvable name is a genuine error. To ingest a
/// raw combined feed (the `gnss` group carries QZSS plus the other systems, and
/// freshly launched satellites resolve to `None`) without aborting, use
/// [`from_celestrak_omm_lenient`].
pub fn from_celestrak_omm(
    system: GnssSystem,
    omms: &[Omm],
) -> Result<Vec<Record>, ConstellationError> {
    let mut records = Vec::with_capacity(omms.len());
    for omm in omms {
        records.push(record_from_omm(system, omm)?);
    }
    records.sort_by_key(|r| (r.system, r.prn));
    Ok(records)
}

/// Build records for `system` from already-parsed CelesTrak OMM records,
/// skipping (rather than aborting on) entries that do not resolve.
///
/// The lenient sibling of [`from_celestrak_omm`]: every OMM whose `OBJECT_NAME`
/// resolves to a PRN for `system` becomes a [`Record`]; every entry that does
/// not is collected into [`Catalog::skipped`] with its identity. This is what a
/// binding feeds a raw combined CelesTrak `gnss` feed: filter to one system by
/// keeping `records` and discarding the `skipped` entries that belong to other
/// constellations, while still seeing which satellites of `system` failed to
/// resolve. Resolvable records are returned sorted by `(system, prn)`; no
/// fabricated record is emitted for a skipped entry.
#[must_use]
pub fn from_celestrak_omm_lenient(system: GnssSystem, omms: &[Omm]) -> Catalog {
    let mut records = Vec::with_capacity(omms.len());
    let mut skipped = Vec::new();
    for omm in omms {
        match record_from_omm(system, omm) {
            Ok(record) => records.push(record),
            Err(_) => skipped.push(SkippedOmm {
                object_name: omm.object_name.clone(),
                norad_id: omm.norad_cat_id,
            }),
        }
    }
    records.sort_by_key(|r| (r.system, r.prn));
    Catalog { records, skipped }
}

fn record_from_omm(system: GnssSystem, omm: &Omm) -> Result<Record, ConstellationError> {
    let object_name = omm.object_name.as_deref();
    let identity = system_identity(system, object_name)
        .ok_or_else(|| ConstellationError::MissingPrn(omm.object_name.clone()))?;

    Ok(Record {
        system,
        prn: identity.prn,
        svn: None,
        norad_id: omm.norad_cat_id,
        sp3_id: gnss_sp3_id(system, identity.prn),
        fdma_channel: identity.fdma_channel,
        active: true,
        usable: true,
        source: RecordSource {
            celestrak: Some(CelestrakSource {
                group: celestrak_group(system).to_string(),
                object_name: omm.object_name.clone(),
                object_id: omm.object_id.clone(),
                epoch: Some(epoch_iso8601(omm)),
                block_type: block_type_from_object_name(system, object_name),
            }),
            navcen: None,
            navcen_conflict: None,
        },
    })
}

/// Resolve the per-system within-constellation identity from an `OBJECT_NAME`.
///
/// Each constellation names its satellites differently in the CelesTrak feeds,
/// so the adapter is dispatched on [`GnssSystem`]. Returns `None` when the name
/// cannot be resolved to a valid PRN for the system.
///
/// NavIC/IRNSS and SBAS are **deliberately unsupported** here and always return
/// `None` (no name will resolve), matching the module-level scope: NavIC OMM
/// names (`IRNSS-1A`, `NVS-01`) carry no PRN and have no published
/// build-id->PRN table comparable to Galileo's GSAT map, and SBAS PRNs (120..)
/// are payload assignments not derivable from the geostationary host's name.
/// Adding either would require a new identity source, not just a name parser; a
/// caller passing `Navic`/`Sbas` gets an empty catalog rather than a fabricated
/// record. This is intentional, not an oversight.
fn system_identity(system: GnssSystem, name: Option<&str>) -> Option<Identity> {
    match system {
        GnssSystem::Gps => prn_from_object_name(name).map(|prn| Identity {
            prn,
            fdma_channel: None,
        }),
        GnssSystem::BeiDou => paren_letter_prn(name, 'C').map(|prn| Identity {
            prn,
            fdma_channel: None,
        }),
        GnssSystem::Qzss => qzss_slot_from_object_name(name).map(|prn| Identity {
            prn,
            fdma_channel: None,
        }),
        GnssSystem::Galileo => {
            let gsat = gsat_from_object_name(name)?;
            galileo_prn_for_gsat(gsat).map(|prn| Identity {
                prn,
                fdma_channel: None,
            })
        }
        GnssSystem::Glonass => {
            let number = paren_number(name)?;
            let slot = glonass_slot_for_number(number)?;
            Some(Identity {
                prn: slot,
                fdma_channel: glonass_fdma_channel(slot),
            })
        }
        // NavIC/IRNSS and SBAS are out of scope (see the doc comment above): no
        // name resolves, so these systems yield an empty catalog rather than a
        // guessed PRN.
        GnssSystem::Navic | GnssSystem::Sbas => None,
    }
}

fn epoch_iso8601(omm: &Omm) -> String {
    let e = &omm.epoch;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}",
        e.year, e.month, e.day, e.hour, e.minute, e.second, e.microsecond
    )
}

/// Parse `(PRN nn)` from a CelesTrak object name, stripping leading zeros.
///
/// Matches the reference regex `\(PRN\s*0*([0-9]{1,3})\)` (case-insensitive),
/// including its *search* semantics: every `(PRN` occurrence is tried, so a
/// later valid `(PRN nn)` is found even if an earlier `(PRN ...)` does not
/// parse. The PRN is up to three significant digits and must be positive.
fn prn_from_object_name(name: Option<&str>) -> Option<u16> {
    let name = name?;
    let mut from = 0;
    while let Some(rel) = find_ci(&name[from..], "(PRN") {
        let after = from + rel + "(PRN".len();
        if let Some(prn) = prn_at(&name[after..]) {
            return Some(prn);
        }
        from = after;
    }
    None
}

/// Parse `\s*0*([0-9]{1,3})\)` at the start of `rest`.
fn prn_at(rest: &str) -> Option<u16> {
    let rest = rest.trim_start();
    let bytes = rest.as_bytes();

    let mut i = 0;
    while i < bytes.len() && bytes[i] == b'0' {
        i += 1;
    }
    let digit_start = i;
    let mut count = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() && count < 3 {
        i += 1;
        count += 1;
    }
    if i >= bytes.len() || bytes[i] != b')' || digit_start == i {
        return None;
    }
    let value: u16 = rest[digit_start..i].parse().ok()?;
    (value > 0).then_some(value)
}

/// Parse a parenthesized `(<letter>nn)` PRN from a CelesTrak object name.
///
/// BeiDou names the PRN inline, e.g. `BEIDOU-3 M1 (C19)`; the leading letter is
/// the RINEX system letter. Reuses the GPS [`prn_at`] digit reader (leading
/// zeros stripped, up to three significant digits, positive) and the same
/// search semantics, so a later valid group wins over an earlier bad one.
fn paren_letter_prn(name: Option<&str>, letter: char) -> Option<u16> {
    let name = name?;
    let needle = format!("({letter}");
    let mut from = 0;
    while let Some(rel) = find_ci(&name[from..], &needle) {
        let after = from + rel + needle.len();
        if let Some(prn) = prn_at(&name[after..]) {
            return Some(prn);
        }
        from = after;
    }
    None
}

/// Parse the first parenthesized integer `(ddd)` from a CelesTrak object name.
///
/// GLONASS names carry the GLONASS (Uragan) number this way, e.g.
/// `COSMOS 2456 (730)`.
fn paren_number(name: Option<&str>) -> Option<u16> {
    let name = name?;
    let open = name.find('(')?;
    let rest = &name[open + 1..];
    let close = rest.find(')')?;
    let digits = rest[..close].trim();
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Parse the QZSS RINEX slot from a CelesTrak object name.
///
/// QZSS names carry the broadcast PRN, e.g. `QZS-2 (QZSS/PRN 194)`; the RINEX
/// slot is `PRN - 192` (`J01`..`J09`), per RINEX 3.0x. Broadcast PRNs outside
/// `193..=201` are rejected.
fn qzss_slot_from_object_name(name: Option<&str>) -> Option<u16> {
    let name = name?;
    let mut from = 0;
    while let Some(rel) = find_ci(&name[from..], "PRN") {
        let after = from + rel + "PRN".len();
        if let Some(prn) = leading_uint(&name[after..]) {
            if (193..=201).contains(&prn) {
                return Some(prn - 192);
            }
        }
        from = after;
    }
    None
}

/// Parse the `GSATdddd` build id from a Galileo CelesTrak object name
/// (`GSAT0210 (GALILEO 13)` -> `210`).
fn gsat_from_object_name(name: Option<&str>) -> Option<u16> {
    let name = name?;
    let rel = find_ci(name, "GSAT")?;
    leading_uint(&name[rel + "GSAT".len()..])
}

/// Read the leading run of ASCII digits (after optional whitespace) as a `u16`.
fn leading_uint(rest: &str) -> Option<u16> {
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest.get(..end).filter(|s| !s.is_empty())?.parse().ok()
}

/// GSAT build id -> Galileo SVID (E-number).
///
/// The SVID is fixed per satellite at commissioning and is published in the EU
/// GSC constellation information / Galileo metadata, so this is a stable
/// identity table rather than a status snapshot. It carries no PRN in the OMM
/// feed (the name is the `GSATdddd` build id), hence this lookup. Satellites
/// with no SVID assigned yet (GIOVE prototypes, freshly launched FOC) are
/// absent and resolve to `None`. Cross-checked against the broadcasting
/// E-PRN set in the 2026-06-24 IGS broadcast navigation file. Source: EU GNSS
/// Service Centre, <https://www.gsc-europa.eu/system-service-status/constellation-information>.
#[must_use]
pub fn galileo_prn_for_gsat(gsat: u16) -> Option<u16> {
    let prn = match gsat {
        101 => 11,
        102 => 12,
        103 => 19,
        104 => 20,
        201 => 18,
        202 => 14,
        203 => 26,
        204 => 22,
        205 => 24,
        206 => 30,
        207 => 7,
        208 => 8,
        209 => 9,
        210 => 1,
        211 => 2,
        212 => 3,
        213 => 4,
        214 => 5,
        215 => 21,
        216 => 25,
        217 => 27,
        218 => 31,
        219 => 36,
        220 => 13,
        221 => 15,
        222 => 33,
        223 => 34,
        224 => 10,
        225 => 29,
        226 => 23,
        227 => 6,
        _ => return None,
    };
    Some(prn)
}

/// GLONASS (Uragan) number -> orbital slot (`1..=24`) for the operational
/// constellation.
///
/// The OMM `OBJECT_NAME` carries the GLONASS number, not the slot, and slot
/// occupancy rotates as satellites are replaced, so this is a point-in-time
/// snapshot of the published IGS/MCC / IAC constellation status matching the
/// committed `glonass_ops_sample.json` epoch (2026-06). Regenerate the two
/// together when the constellation changes. Source: IAC GLONASS constellation
/// status / List of GLONASS satellites; cross-checked against the CelesTrak
/// `glo-ops` NORAD ids.
#[must_use]
pub fn glonass_slot_for_number(number: u16) -> Option<u16> {
    let slot = match number {
        730 => 1,
        747 => 2,
        744 => 3,
        759 => 4,
        756 => 5,
        704 => 6,
        745 => 7,
        743 => 8,
        702 => 9,
        723 => 10,
        705 => 11,
        758 => 12,
        721 => 13,
        752 => 14,
        757 => 15,
        761 => 16,
        751 => 17,
        754 => 18,
        707 => 19,
        708 => 20,
        755 => 21,
        706 => 22,
        732 => 23,
        760 => 24,
        _ => return None,
    };
    Some(slot)
}

/// GLONASS orbital slot (`1..=24`) -> FDMA L1/L2 frequency-channel number `k`.
///
/// This is the published IGS/MCC slot<->channel assignment: antipodal slots
/// (same plane, 180 deg apart in argument of latitude) share a channel, so only
/// 14 of the channels in `-7..=6` are in use. The mapping is stable over time
/// (verified identical between the 2018 UNB/IAC published table and the
/// 2026-06-24 IGS merged broadcast navigation file), and is the bit-exact golden
/// for the FDMA datum, which appears in no OMM feed. Sources:
/// - GLONASS Constellation Status, R. B. Langley, UNB (IAC Moscow / IGS),
///   <https://gge.ext.unb.ca/Resources/GLONASSConstellationStatus.txt>.
/// - IGS daily merged broadcast navigation (GLONASS frequency-number field).
#[must_use]
pub fn glonass_fdma_channel(slot: u16) -> Option<i8> {
    let channel = match slot {
        1 => 1,
        2 => -4,
        3 => 5,
        4 => 6,
        5 => 1,
        6 => -4,
        7 => 5,
        8 => 6,
        9 => -2,
        10 => -7,
        11 => 0,
        12 => -1,
        13 => -2,
        14 => -7,
        15 => 0,
        16 => -1,
        17 => 4,
        18 => -3,
        19 => 3,
        20 => 2,
        21 => 4,
        22 => -3,
        23 => 3,
        24 => 2,
        _ => return None,
    };
    Some(channel)
}

/// Parse the satellite block/generation from a CelesTrak object name token.
///
/// GPS mirrors the reference patterns, matched as whole words in the order
/// `IIR-M`, `III`, `IIF`, `IIR` so `BIIRM` is not caught by `BIIR`. The other
/// systems carry their generation in the name too (`BEIDOU-3S`, `BEIDOU-2`;
/// Galileo IOV `GSAT01xx` vs FOC `GSAT02xx`); GLONASS does not, so it is `None`.
fn block_type_from_object_name(system: GnssSystem, name: Option<&str>) -> Option<String> {
    let name = name?;
    match system {
        GnssSystem::Gps => {
            if contains_word_ci(name, "BIIRM") || contains_word_ci(name, "BIIR-M") {
                Some("IIR-M".to_string())
            } else if contains_word_ci(name, "BIII") {
                Some("III".to_string())
            } else if contains_word_ci(name, "BIIF") {
                Some("IIF".to_string())
            } else if contains_word_ci(name, "BIIR") {
                Some("IIR".to_string())
            } else {
                None
            }
        }
        GnssSystem::BeiDou => {
            if contains_word_ci(name, "BEIDOU-3S") {
                Some("BDS-3S".to_string())
            } else if contains_word_ci(name, "BEIDOU-3") {
                Some("BDS-3".to_string())
            } else if contains_word_ci(name, "BEIDOU-2") {
                Some("BDS-2".to_string())
            } else {
                None
            }
        }
        GnssSystem::Galileo => match gsat_from_object_name(Some(name)) {
            Some(gsat) if gsat < 200 => Some("IOV".to_string()),
            Some(_) => Some("FOC".to_string()),
            None => None,
        },
        _ => None,
    }
}

/// Parse NAVCEN's GPS constellation status HTML from raw bytes.
///
/// The parser targets the Drupal table-field classes NAVCEN's public GPS
/// constellation page uses, scanned without an HTML crate. Returns status rows
/// sorted by PRN; merge them into CelesTrak records with [`merge_navcen`].
pub fn parse_navcen(bytes: &[u8]) -> Result<Vec<NavcenStatus>, ConstellationError> {
    let html = core::str::from_utf8(bytes).map_err(|_| ConstellationError::NavcenNotUtf8)?;

    let mut statuses = Vec::new();
    for row in tr_blocks(html) {
        if find_ci(row, "views-field-field-gps-prn").is_none() || find_ci(row, "<td").is_none() {
            continue;
        }
        statuses.push(navcen_status_from_row(row)?);
    }

    if statuses.is_empty() {
        return Err(ConstellationError::NavcenNoRows);
    }
    statuses.sort_by_key(|s| s.prn);
    Ok(statuses)
}

fn navcen_status_from_row(row: &str) -> Result<NavcenStatus, ConstellationError> {
    let prn = navcen_required_int(row, "gps-prn")?;
    let svn = navcen_optional_int(row, "gps-svn")?;
    let nanu_type = navcen_text(row, "nanu-type");
    let active_nanu = navcen_active(row);
    let usable = !(active_nanu && unusable_nanu_type(nanu_type.as_deref()));

    Ok(NavcenStatus {
        system: GnssSystem::Gps,
        prn,
        svn,
        usable,
        active_nanu,
        nanu_type: blank_to_none(nanu_type),
        nanu_subject: blank_to_none(navcen_text(row, "nanu-subject")),
        plane: blank_to_none(navcen_text(row, "gps-con-plane")),
        slot: blank_to_none(navcen_text(row, "gps-con-slot")),
        block_type: blank_to_none(navcen_text(row, "gps-con-block-type")),
        clock: blank_to_none(navcen_text(row, "gps-con-clock")),
    })
}

fn navcen_required_int(row: &str, field: &'static str) -> Result<u16, ConstellationError> {
    let text = navcen_text(row, field);
    parse_positive_int(text.as_deref().unwrap_or(""), field)
}

fn navcen_optional_int(row: &str, field: &'static str) -> Result<Option<u16>, ConstellationError> {
    match navcen_text(row, field).as_deref() {
        None | Some("") => Ok(None),
        Some(text) => parse_positive_int(text, field).map(Some),
    }
}

fn parse_positive_int(text: &str, field: &'static str) -> Result<u16, ConstellationError> {
    let trimmed = text.trim();
    match trimmed.parse::<u16>() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err(ConstellationError::NavcenBadField {
            field,
            value: trimmed.to_string(),
        }),
    }
}

fn navcen_text(row: &str, field: &str) -> Option<String> {
    let needle = format!("views-field-field-{field}");
    td_inner(row, &needle).map(clean_html)
}

fn navcen_active(row: &str) -> bool {
    td_inner(row, "nanu-active-check")
        .map(clean_html)
        .as_deref()
        == Some("1")
}

fn unusable_nanu_type(nanu_type: Option<&str>) -> bool {
    nanu_type.is_some_and(|text| {
        let upper = text.trim().to_ascii_uppercase();
        matches!(
            upper.as_str(),
            "UNUSABLE" | "DECOM" | "FCSTDV" | "FCSTMX" | "FCSTEXTD"
        )
    })
}

/// Merge NAVCEN status rows into normalized records by PRN.
///
/// NAVCEN does not publish NORAD ids, so CelesTrak stays the identity base. When
/// a PRN exists in both sources and the block types are compatible, this fills
/// `svn`, updates `usable`, and records the NAVCEN provenance. A NAVCEN row that
/// matches the PRN but carries an incompatible block type (a PRN transition) is
/// recorded under `navcen_conflict` rather than merged. Returns records sorted
/// by `(system, prn)`.
///
/// NAVCEN's status page is GPS-only, so the overlay is keyed by `(system, PRN)`
/// and only ever lands on GPS records. Keying by PRN alone would splice GPS SVN /
/// usability / provenance onto a same-PRN record of another constellation
/// (`R01`, `J01`, ...), corrupting cross-system identity.
///
/// As in the reference (`Map.new(statuses, &{&1.prn, &1})`), at most one status
/// is kept per `(system, PRN)`; if the input carries duplicates the last wins.
#[must_use]
pub fn merge_navcen(records: &[Record], statuses: &[NavcenStatus]) -> Vec<Record> {
    let mut by_key: std::collections::HashMap<(GnssSystem, u16), &NavcenStatus> =
        std::collections::HashMap::with_capacity(statuses.len());
    for status in statuses {
        by_key.insert((status.system, status.prn), status);
    }

    let mut merged: Vec<Record> = records
        .iter()
        .map(|record| {
            by_key
                .get(&(record.system, record.prn))
                .map_or_else(|| record.clone(), |status| merge_status(record, status))
        })
        .collect();
    merged.sort_by_key(|r| (r.system, r.prn));
    merged
}

fn merge_status(record: &Record, status: &NavcenStatus) -> Record {
    let mut out = record.clone();
    if navcen_compatible(record, status) {
        out.svn = status.svn;
        out.usable = status.usable;
        out.source.navcen = Some(navcen_source(status));
    } else {
        out.source.navcen_conflict = Some(navcen_source(status));
    }
    out
}

fn navcen_source(status: &NavcenStatus) -> NavcenSource {
    NavcenSource {
        svn: status.svn,
        block_type: status.block_type.clone(),
        plane: status.plane.clone(),
        slot: status.slot.clone(),
        clock: status.clock.clone(),
        nanu_type: status.nanu_type.clone(),
        nanu_subject: status.nanu_subject.clone(),
        active_nanu: status.active_nanu,
    }
}

fn navcen_compatible(record: &Record, status: &NavcenStatus) -> bool {
    let celestrak_block = record
        .source
        .celestrak
        .as_ref()
        .and_then(|c| c.block_type.as_deref());
    let navcen_block = status
        .block_type
        .as_deref()
        .map(|b| b.trim().to_ascii_uppercase());

    match (celestrak_block, navcen_block) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// Export records as the compact mapping CSV.
///
/// The header is `prn,norad_cat_id,active,sp3_id`. The `active` column is `true`
/// only when both `active` and `usable` hold. Records are sorted by
/// `(system, prn)`; the system is encoded in the `sp3_id` letter, so equal PRNs
/// across systems are ordered deterministically without a separate column.
#[must_use]
pub fn to_csv(records: &[Record], booleans: BoolStyle) -> String {
    let mut sorted: Vec<&Record> = records.iter().collect();
    sorted.sort_by_key(|r| (r.system, r.prn));

    let mut out = String::from("prn,norad_cat_id,active,sp3_id\n");
    for record in sorted {
        let active = format_bool(operational(record), booleans);
        let _ = writeln!(
            out,
            "{},{},{},{}",
            record.prn, record.norad_id, active, record.sp3_id
        );
    }
    out
}

fn format_bool(value: bool, style: BoolStyle) -> &'static str {
    match (style, value) {
        (BoolStyle::Lower, true) => "true",
        (BoolStyle::Lower, false) => "false",
        (BoolStyle::Title, true) => "True",
        (BoolStyle::Title, false) => "False",
    }
}

fn operational(record: &Record) -> bool {
    record.active && record.usable
}

/// Validate catalog identity without an SP3 product.
///
/// Reports duplicate PRNs, duplicate NORAD ids, and PRNs that are inactive or
/// unusable.
#[must_use]
pub fn validate(records: &[Record]) -> Validation {
    validation(records, None)
}

/// Validate catalog identity against a loaded SP3 product.
///
/// `missing_sp3_ids` reports active+usable catalog ids absent from the product;
/// `extra_sp3_ids` reports product ids absent from the active+usable catalog,
/// restricted to the constellations the catalog covers (so a single-system
/// catalog is not flagged against a multi-GNSS product's other systems).
#[must_use]
pub fn validate_against_sp3(records: &[Record], sp3: &Sp3) -> Validation {
    let ids: Vec<String> = sp3
        .header
        .satellites
        .iter()
        .map(ToString::to_string)
        .collect();
    validation(records, Some(&ids))
}

/// Validate catalog identity against a plain list of SP3/RINEX satellite tokens.
#[must_use]
pub fn validate_against_sp3_ids(records: &[Record], sp3_ids: &[&str]) -> Validation {
    let ids: Vec<String> = sp3_ids.iter().map(|id| (*id).to_string()).collect();
    validation(records, Some(&ids))
}

fn validation(records: &[Record], sp3_ids: Option<&[String]>) -> Validation {
    let mut report = Validation {
        missing_sp3_ids: Vec::new(),
        duplicate_prns: duplicates(records.iter().map(|r| (r.system, r.prn))),
        duplicate_norad_ids: duplicates(records.iter().map(|r| r.norad_id)),
        inactive_unusable_prns: inactive_unusable_prns(records),
        extra_sp3_ids: Vec::new(),
    };

    if let Some(sp3_ids) = sp3_ids {
        // Only compare against the systems this catalog actually covers, so a
        // GPS-only catalog is not flagged for the Galileo/GLONASS ids in a
        // multi-GNSS product. The hardcoded `'G'` filter generalizes to the set
        // of system letters present in the records.
        let letters: std::collections::HashSet<char> =
            records.iter().map(|r| r.system.letter()).collect();
        let catalog: Vec<String> = records
            .iter()
            .filter(|r| operational(r))
            .map(|r| r.sp3_id.to_ascii_uppercase())
            .collect();
        let product: Vec<String> = sp3_ids
            .iter()
            .map(|id| id.to_ascii_uppercase())
            .filter(|id| id.chars().next().is_some_and(|c| letters.contains(&c)))
            .collect();

        report.missing_sp3_ids = set_difference(&catalog, &product);
        report.extra_sp3_ids = set_difference(&product, &catalog);
    }

    report
}

fn duplicates<T>(values: impl Iterator<Item = T>) -> Vec<T>
where
    T: Ord + Copy,
{
    let mut seen: Vec<T> = values.collect();
    seen.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < seen.len() {
        let mut j = i + 1;
        while j < seen.len() && seen[j] == seen[i] {
            j += 1;
        }
        if j - i > 1 {
            out.push(seen[i]);
        }
        i = j;
    }
    out
}

fn inactive_unusable_prns(records: &[Record]) -> Vec<(GnssSystem, u16)> {
    let mut prns: Vec<(GnssSystem, u16)> = records
        .iter()
        .filter(|r| !operational(r))
        .map(|r| (r.system, r.prn))
        .collect();
    prns.sort_unstable();
    prns.dedup();
    prns
}

fn set_difference(left: &[String], right: &[String]) -> Vec<String> {
    let mut out: Vec<String> = left
        .iter()
        .filter(|id| !right.contains(id))
        .cloned()
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Returns `true` when a validation report has no findings.
#[must_use]
pub fn is_valid(report: &Validation) -> bool {
    report.missing_sp3_ids.is_empty()
        && report.duplicate_prns.is_empty()
        && report.duplicate_norad_ids.is_empty()
        && report.inactive_unusable_prns.is_empty()
        && report.extra_sp3_ids.is_empty()
}

/// Validate against a plain SP3 id list and fail unless the catalog is clean.
///
/// A build-time gate: returns `Ok(())` when the report has no findings, otherwise
/// [`ConstellationError::Sp3Validation`] describing them.
pub fn validate_against_sp3_ids_strict(
    records: &[Record],
    sp3_ids: &[&str],
) -> Result<(), ConstellationError> {
    let report = validate_against_sp3_ids(records, sp3_ids);
    if is_valid(&report) {
        Ok(())
    } else {
        Err(ConstellationError::Sp3Validation(describe_findings(
            &report,
        )))
    }
}

fn describe_findings(report: &Validation) -> String {
    let mut parts = Vec::new();
    if !report.missing_sp3_ids.is_empty() {
        parts.push(format!("missing_sp3_ids: {:?}", report.missing_sp3_ids));
    }
    if !report.extra_sp3_ids.is_empty() {
        parts.push(format!("extra_sp3_ids: {:?}", report.extra_sp3_ids));
    }
    if !report.duplicate_prns.is_empty() {
        parts.push(format!("duplicate_prns: {:?}", report.duplicate_prns));
    }
    if !report.duplicate_norad_ids.is_empty() {
        parts.push(format!(
            "duplicate_norad_ids: {:?}",
            report.duplicate_norad_ids
        ));
    }
    if !report.inactive_unusable_prns.is_empty() {
        parts.push(format!(
            "inactive_unusable_prns: {:?}",
            report.inactive_unusable_prns
        ));
    }
    parts.join("; ")
}

/// Compare two catalog snapshots by `(system, prn)` identity.
///
/// Assumes each input has at most one record per `(system, prn)`; run
/// [`validate`] first on hand-edited catalogs and treat duplicate findings as
/// malformed input rather than a constellation change.
#[must_use]
pub fn diff(previous: &[Record], current: &[Record]) -> Diff {
    let key = |r: &Record| (r.system, r.prn);

    let added: Vec<Record> = current
        .iter()
        .filter(|c| !previous.iter().any(|p| key(p) == key(c)))
        .cloned()
        .collect();
    let removed: Vec<Record> = previous
        .iter()
        .filter(|p| !current.iter().any(|c| key(c) == key(p)))
        .cloned()
        .collect();

    let mut added = added;
    let mut removed = removed;
    added.sort_by_key(|r| (r.system, r.prn));
    removed.sort_by_key(|r| (r.system, r.prn));

    let mut common: Vec<(GnssSystem, u16)> = previous
        .iter()
        .filter_map(|p| current.iter().find(|c| key(c) == key(p)).map(|_| key(p)))
        .collect();
    common.sort_unstable();

    let pairs: Vec<(&Record, &Record)> = common
        .iter()
        .map(|k| {
            let p = previous.iter().find(|r| key(r) == *k).expect("common key");
            let c = current.iter().find(|r| key(r) == *k).expect("common key");
            (p, c)
        })
        .collect();

    Diff {
        added,
        removed,
        norad_reassigned: changes(&pairs, |r| r.norad_id),
        sp3_id_changed: changes(&pairs, |r| r.sp3_id.clone()),
        svn_changed: changes(&pairs, |r| r.svn),
        fdma_channel_changed: changes(&pairs, |r| r.fdma_channel),
        activity_changed: changes(&pairs, |r| r.active),
        usability_changed: changes(&pairs, |r| r.usable),
    }
}

fn changes<T, F>(pairs: &[(&Record, &Record)], field: F) -> Vec<FieldChange<T>>
where
    T: PartialEq,
    F: Fn(&Record) -> T,
{
    pairs
        .iter()
        .filter_map(|(p, c)| {
            let from = field(p);
            let to = field(c);
            if from == to {
                None
            } else {
                Some(FieldChange {
                    system: p.system,
                    prn: p.prn,
                    from,
                    to,
                })
            }
        })
        .collect()
}

/// Returns `true` when a diff has any findings.
#[must_use]
pub fn changed(diff: &Diff) -> bool {
    !diff.added.is_empty()
        || !diff.removed.is_empty()
        || !diff.norad_reassigned.is_empty()
        || !diff.sp3_id_changed.is_empty()
        || !diff.svn_changed.is_empty()
        || !diff.fdma_channel_changed.is_empty()
        || !diff.activity_changed.is_empty()
        || !diff.usability_changed.is_empty()
}

// ── HTML/text scanning helpers (dependency-light) ────────────────────────────

fn blank_to_none(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

/// Case-insensitive ASCII substring search returning the byte offset.
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let hay = haystack.as_bytes();
    let need = needle.as_bytes();
    if need.is_empty() {
        return Some(0);
    }
    if hay.len() < need.len() {
        return None;
    }
    (0..=hay.len() - need.len()).find(|&i| {
        hay[i..i + need.len()]
            .iter()
            .zip(need)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    })
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Case-insensitive whole-word match, mirroring regex `\bword\b` boundaries.
fn contains_word_ci(haystack: &str, word: &str) -> bool {
    let hay = haystack.as_bytes();
    let need = word.as_bytes();
    let n = need.len();
    if n == 0 || hay.len() < n {
        return false;
    }
    (0..=hay.len() - n).any(|i| {
        let matched = hay[i..i + n]
            .iter()
            .zip(need)
            .all(|(a, b)| a.eq_ignore_ascii_case(b));
        if !matched {
            return false;
        }
        let left_ok = i == 0 || !is_word_byte(hay[i - 1]);
        let right_ok = i + n == hay.len() || !is_word_byte(hay[i + n]);
        left_ok && right_ok
    })
}

/// Split HTML into the inner text of each `<tr>...</tr>` block.
fn tr_blocks(html: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(start) = find_ci(rest, "<tr") {
        let Some(gt) = rest[start..].find('>') else {
            break;
        };
        let content_start = start + gt + 1;
        let Some(close) = find_ci(&rest[content_start..], "</tr>") else {
            break;
        };
        out.push(&rest[content_start..content_start + close]);
        rest = &rest[content_start + close + "</tr>".len()..];
    }
    out
}

/// Inner text of the first `<td>` whose attributes contain `class_needle`.
fn td_inner<'a>(row: &'a str, class_needle: &str) -> Option<&'a str> {
    let mut rest = row;
    loop {
        let start = find_ci(rest, "<td")?;
        let gt = rest[start..].find('>')?;
        let attrs = &rest[start..start + gt];
        let content_start = start + gt + 1;
        let close = find_ci(&rest[content_start..], "</td>")?;
        let inner = &rest[content_start..content_start + close];
        if find_ci(attrs, class_needle).is_some() {
            return Some(inner);
        }
        rest = &rest[content_start + close + "</td>".len()..];
    }
}

/// Strip tags, unescape entities, and collapse whitespace, matching the
/// reference `clean_html`.
fn clean_html(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => stripped.push(c),
            _ => {}
        }
    }
    let unescaped = html_unescape(&stripped);
    unescaped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Decode HTML entities: the named set the reference handles plus numeric
/// character references (`&#160;`, `&#xA0;`). Numeric decoding is a superset of
/// the reference's named-only set, so it never changes a reference-covered case
/// but keeps generated markup (numeric `&nbsp;`, `&apos;`) from leaking literal
/// `&#160;` into a cell and breaking, for example, optional-integer parsing.
fn html_unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        if let Some((decoded, consumed)) = decode_entity(tail) {
            out.push(decoded);
            rest = &tail[consumed..];
        } else {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Decode a single entity at the start of `s` (which begins with `&`), returning
/// the decoded char and the number of bytes consumed, or `None` if `s` does not
/// start with a recognized entity.
fn decode_entity(s: &str) -> Option<(char, usize)> {
    for (entity, decoded) in [
        ("&amp;", '&'),
        ("&lt;", '<'),
        ("&gt;", '>'),
        ("&quot;", '"'),
        ("&#39;", '\''),
        ("&apos;", '\''),
        ("&nbsp;", ' '),
    ] {
        if s.starts_with(entity) {
            return Some((decoded, entity.len()));
        }
    }

    // Numeric character reference: &#DDD; or &#xHHH;
    let body = s.strip_prefix("&#")?;
    let semi = body.find(';')?;
    let (digits, radix) = match body.strip_prefix(['x', 'X']) {
        Some(hex) => (&hex[..semi - 1], 16),
        None => (&body[..semi], 10),
    };
    if digits.is_empty() {
        return None;
    }
    let code = u32::from_str_radix(digits, radix).ok()?;
    let decoded = char::from_u32(code)?;
    Some((decoded, "&#".len() + semi + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prn_parses_padded_and_multi_digit() {
        assert_eq!(prn_from_object_name(Some("GPS BIIF-8  (PRN 03)")), Some(3));
        assert_eq!(prn_from_object_name(Some("GPS BIII-10 (PRN 13)")), Some(13));
        assert_eq!(prn_from_object_name(Some("X (PRN 003)")), Some(3));
    }

    #[test]
    fn prn_search_skips_unparseable_earlier_occurrence() {
        // A leading "(PRN ...)" that does not parse must not block a later valid
        // one, matching the reference regex's search semantics.
        assert_eq!(
            prn_from_object_name(Some("GPS (PRN X) BIIF (PRN 07)")),
            Some(7)
        );
        assert_eq!(prn_from_object_name(Some("GPS WITHOUT PRN")), None);
        assert_eq!(prn_from_object_name(Some("(PRN 000)")), None);
    }

    #[test]
    fn html_unescape_decodes_named_and_numeric_entities() {
        assert_eq!(html_unescape("a &amp; b"), "a & b");
        assert_eq!(html_unescape("&#39;x&#39;"), "'x'");
        // Numeric references for NBSP (decimal and hex) decode to spaces.
        assert_eq!(html_unescape("&#160;"), "\u{a0}");
        assert_eq!(html_unescape("&#xA0;"), "\u{a0}");
        // An unrecognized "&" is left literal rather than dropped.
        assert_eq!(html_unescape("AT&T"), "AT&T");
    }

    #[test]
    fn optional_int_treats_numeric_nbsp_cell_as_blank() {
        // A cell whose only content is a numeric NBSP cleans to whitespace and
        // collapses to "", so it is absent rather than a parse error.
        let row = r#"<td class="views-field-field-gps-svn">&#160;</td>"#;
        assert_eq!(navcen_optional_int(row, "gps-svn"), Ok(None));
    }

    #[test]
    fn beidou_prn_parses_from_parenthesized_letter_group() {
        assert_eq!(paren_letter_prn(Some("BEIDOU-3 M1 (C19)"), 'C'), Some(19));
        assert_eq!(paren_letter_prn(Some("BEIDOU-2 G8 (C01)"), 'C'), Some(1));
        assert_eq!(paren_letter_prn(Some("BEIDOU-3 G2 (C60)"), 'C'), Some(60));
        assert_eq!(paren_letter_prn(Some("NO LETTER GROUP"), 'C'), None);
    }

    #[test]
    fn qzss_slot_is_broadcast_prn_minus_192() {
        // RINEX 3.0x: J-slot = broadcast PRN - 192, valid only for 193..=201.
        assert_eq!(
            qzss_slot_from_object_name(Some("QZS-2 (QZSS/PRN 194)")),
            Some(2)
        );
        assert_eq!(
            qzss_slot_from_object_name(Some("QZS-3 (QZSS/PRN 199)")),
            Some(7)
        );
        assert_eq!(
            qzss_slot_from_object_name(Some("QZS-6 (QZSS/PRN 200)")),
            Some(8)
        );
        // Out-of-band broadcast PRN (e.g. an SBAS-style 122) is rejected.
        assert_eq!(qzss_slot_from_object_name(Some("X (PRN 122)")), None);
    }

    #[test]
    fn galileo_gsat_parses_and_maps_to_svid() {
        assert_eq!(
            gsat_from_object_name(Some("GSAT0210 (GALILEO 13)")),
            Some(210)
        );
        assert_eq!(
            gsat_from_object_name(Some("GSAT0101 (GALILEO-PFM)")),
            Some(101)
        );
        assert_eq!(gsat_from_object_name(Some("COSMOS 2456 (730)")), None);
        // GSAT0210 ("GALILEO 13") is SVID E01, not E13 - the table, not the name.
        assert_eq!(galileo_prn_for_gsat(210), Some(1));
        assert_eq!(galileo_prn_for_gsat(211), Some(2));
        assert_eq!(galileo_prn_for_gsat(101), Some(11));
        assert_eq!(galileo_prn_for_gsat(228), None);
    }

    #[test]
    fn glonass_number_resolves_to_slot_and_channel() {
        assert_eq!(paren_number(Some("COSMOS 2456 (730)")), Some(730));
        assert_eq!(glonass_slot_for_number(730), Some(1));
        assert_eq!(glonass_slot_for_number(721), Some(13));
        assert_eq!(glonass_slot_for_number(999), None);
        // FDMA channels: antipodal slots (180 deg apart in plane) share a
        // channel, e.g. slots 1 and 5 are both +1, 2 and 6 both -4.
        assert_eq!(glonass_fdma_channel(1), Some(1));
        assert_eq!(glonass_fdma_channel(5), Some(1));
        assert_eq!(glonass_fdma_channel(2), Some(-4));
        assert_eq!(glonass_fdma_channel(6), Some(-4));
        assert_eq!(glonass_fdma_channel(13), Some(-2));
        assert_eq!(glonass_fdma_channel(0), None);
        assert_eq!(glonass_fdma_channel(25), None);
    }

    #[test]
    fn gnss_sp3_id_renders_per_system_token() {
        assert_eq!(gnss_sp3_id(GnssSystem::Gps, 7), "G07");
        assert_eq!(gnss_sp3_id(GnssSystem::Galileo, 7), "E07");
        assert_eq!(gnss_sp3_id(GnssSystem::Glonass, 13), "R13");
        assert_eq!(gnss_sp3_id(GnssSystem::BeiDou, 19), "C19");
        assert_eq!(gnss_sp3_id(GnssSystem::Qzss, 2), "J02");
    }

    /// Minimal OMM carrying only the identity fields the constellation builders
    /// read (`OBJECT_NAME`, `NORAD_CAT_ID`); the orbital elements are unused here.
    fn omm_named(object_name: &str, norad_cat_id: u32) -> Omm {
        Omm {
            ccsds_omm_vers: String::new(),
            creation_date: None,
            originator: None,
            object_name: Some(object_name.to_string()),
            object_id: None,
            center_name: None,
            ref_frame: None,
            time_system: None,
            mean_element_theory: None,
            epoch: crate::astro::omm::OmmEpoch {
                year: 2026,
                month: 6,
                day: 24,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
            },
            mean_motion: 0.0,
            eccentricity: 0.0,
            inclination_deg: 0.0,
            ra_of_asc_node_deg: 0.0,
            arg_of_pericenter_deg: 0.0,
            mean_anomaly_deg: 0.0,
            ephemeris_type: 0,
            classification_type: String::new(),
            norad_cat_id,
            element_set_no: 0,
            rev_at_epoch: 0,
            bstar: 0.0,
            mean_motion_dot: 0.0,
            mean_motion_ddot: 0.0,
        }
    }

    #[test]
    fn lenient_builder_returns_partial_success_with_skipped_identities() {
        // A raw combined `gnss`-style slice viewed as GPS: two resolvable GPS
        // names, one QZSS member (lives in the same combined group, no GPS PRN),
        // and one GPS-looking name with no parseable PRN block.
        let omms = [
            omm_named("GPS BIIF-8  (PRN 03)", 40294),
            omm_named("QZS-2 (QZSS/PRN 194)", 42738),
            omm_named("GPS BIII-1  (PRN 04)", 43873),
            omm_named("GPS WITHOUT PRN", 99999),
        ];

        // Strict path aborts on the first unresolvable entry, naming it.
        assert_eq!(
            from_celestrak_omm(GnssSystem::Gps, &omms),
            Err(ConstellationError::MissingPrn(Some(
                "QZS-2 (QZSS/PRN 194)".to_string()
            )))
        );

        // Lenient path keeps the resolvable GPS records (sorted by prn) and
        // reports each skipped entry's identity, in input order.
        let catalog = from_celestrak_omm_lenient(GnssSystem::Gps, &omms);
        assert_eq!(
            catalog.records.iter().map(|r| r.prn).collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert!(catalog.records.iter().all(|r| r.system == GnssSystem::Gps));
        assert_eq!(
            catalog.skipped,
            vec![
                SkippedOmm {
                    object_name: Some("QZS-2 (QZSS/PRN 194)".to_string()),
                    norad_id: 42738,
                },
                SkippedOmm {
                    object_name: Some("GPS WITHOUT PRN".to_string()),
                    norad_id: 99999,
                },
            ]
        );
    }

    #[test]
    fn lenient_builder_partitions_a_realistic_combined_gnss_feed() {
        // The combined CelesTrak `gnss` group carries every system, each with its
        // own naming convention. Lenient build for one system must keep only that
        // system's records and skip the rest - the per-system identity adapters
        // are what distinguish them (only GPS uses the bare `(PRN nn)` form, QZSS
        // uses `(QZSS/PRN nnn)`, GLONASS a bare `(number)`, etc.).
        let feed = [
            omm_named("GPS BIIF-8  (PRN 03)", 40294),
            omm_named("COSMOS 2456 (730)", 37139), // GLONASS slot 1
            omm_named("GSAT0210 (GALILEO 13)", 41859), // Galileo E01
            omm_named("BEIDOU-3 M1 (C19)", 43001), // BeiDou C19
            omm_named("QZS-2 (QZSS/PRN 194)", 42738), // QZSS J02
        ];

        let gps = from_celestrak_omm_lenient(GnssSystem::Gps, &feed);
        assert_eq!(
            gps.records
                .iter()
                .map(|r| r.sp3_id.as_str())
                .collect::<Vec<_>>(),
            vec!["G03"]
        );
        assert_eq!(gps.skipped.len(), 4, "the four non-GPS names are skipped");

        let glonass = from_celestrak_omm_lenient(GnssSystem::Glonass, &feed);
        assert_eq!(
            glonass
                .records
                .iter()
                .map(|r| r.sp3_id.as_str())
                .collect::<Vec<_>>(),
            vec!["R01"]
        );
        assert_eq!(glonass.skipped.len(), 4);

        // The partitions are disjoint: each system claims exactly one record and
        // skips the other four, so no name is double-counted across systems.
        for system in [
            GnssSystem::Gps,
            GnssSystem::Glonass,
            GnssSystem::Galileo,
            GnssSystem::BeiDou,
            GnssSystem::Qzss,
        ] {
            let cat = from_celestrak_omm_lenient(system, &feed);
            assert_eq!(cat.records.len(), 1, "{system:?}: one record");
            assert_eq!(cat.skipped.len(), 4, "{system:?}: four skipped");
            assert!(cat.records.iter().all(|r| r.system == system));
        }
    }

    /// Build a minimal record for a given system/prn carrying a CelesTrak source
    /// (so block-type compatibility is exercised), used by the merge tests.
    fn record_for(system: GnssSystem, prn: u16, norad_id: u32) -> Record {
        Record {
            system,
            prn,
            svn: None,
            norad_id,
            sp3_id: gnss_sp3_id(system, prn),
            fdma_channel: None,
            active: true,
            usable: true,
            source: RecordSource::default(),
        }
    }

    fn navcen_gps(prn: u16, svn: u16, usable: bool) -> NavcenStatus {
        NavcenStatus {
            system: GnssSystem::Gps,
            prn,
            svn: Some(svn),
            usable,
            active_nanu: !usable,
            nanu_type: None,
            nanu_subject: None,
            plane: None,
            slot: None,
            block_type: None,
            clock: None,
        }
    }

    #[test]
    fn merge_navcen_does_not_cross_systems() {
        // GPS PRN 1 and GLONASS slot 1 (R01) share the integer PRN. A GPS-only
        // NAVCEN row for PRN 1 must merge onto the GPS record and leave R01/J01
        // untouched - keying by PRN alone corrupted the GLONASS/QZSS records.
        let records = [
            record_for(GnssSystem::Gps, 1, 40000),
            record_for(GnssSystem::Glonass, 1, 50000),
            record_for(GnssSystem::Qzss, 1, 60000),
        ];
        let statuses = [navcen_gps(1, 63, false)];

        let merged = merge_navcen(&records, &statuses);

        let gps = merged.iter().find(|r| r.system == GnssSystem::Gps).unwrap();
        assert_eq!(gps.svn, Some(63), "GPS record gets the NAVCEN SVN");
        assert!(!gps.usable, "GPS usability follows NAVCEN");
        assert!(gps.source.navcen.is_some());

        for system in [GnssSystem::Glonass, GnssSystem::Qzss] {
            let other = merged.iter().find(|r| r.system == system).unwrap();
            assert_eq!(other.svn, None, "{system:?} must not inherit GPS SVN");
            assert!(other.usable, "{system:?} usability untouched");
            assert!(
                other.source.navcen.is_none(),
                "{system:?} must carry no NAVCEN provenance"
            );
        }
    }

    #[test]
    fn merge_navcen_sorts_by_system_then_prn() {
        let records = [
            record_for(GnssSystem::Glonass, 2, 50002),
            record_for(GnssSystem::Gps, 5, 40005),
            record_for(GnssSystem::Gps, 1, 40001),
        ];
        let merged = merge_navcen(&records, &[]);
        let order: Vec<(GnssSystem, u16)> = merged.iter().map(|r| (r.system, r.prn)).collect();
        assert_eq!(
            order,
            vec![
                (GnssSystem::Gps, 1),
                (GnssSystem::Gps, 5),
                (GnssSystem::Glonass, 2),
            ]
        );
    }

    #[test]
    fn lenient_builder_all_resolvable_has_empty_skipped() {
        let omms = [
            omm_named("GPS BIIF-8  (PRN 03)", 40294),
            omm_named("GPS BIII-1  (PRN 04)", 43873),
        ];
        let catalog = from_celestrak_omm_lenient(GnssSystem::Gps, &omms);
        assert_eq!(catalog.records.len(), 2);
        assert!(catalog.skipped.is_empty());
        // Matches the strict builder exactly when nothing is skipped.
        assert_eq!(
            catalog.records,
            from_celestrak_omm(GnssSystem::Gps, &omms).unwrap()
        );
    }
}
