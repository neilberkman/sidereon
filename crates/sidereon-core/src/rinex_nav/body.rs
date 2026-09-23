//! Reading a navigation file into its header and body entries.
//!
//! The body is split into blocks before any block is decoded, so one block that
//! cannot be read never moves another's boundaries: a RINEX 4 block runs from its
//! frame marker to the next marker, a RINEX 2/3 block from its record-start line to
//! the next record-start line. Every line of the file belongs to the header or to
//! exactly one entry, which keeps its text.

use crate::id::{GnssSatelliteId, GnssSystem};

use super::frames::{parse_eop_frame, parse_ion_frame, parse_sto_frame};
use super::geo::parse_sbas_block;
use super::header::{self, HeaderIssue, NavHeader};
use super::{
    is_record_start, is_v2_numeric_record_start, nav_message_from_v4_token, parse_cnav_block,
    parse_glonass_block, parse_keplerian_block, parse_v4_eph_marker, parse_v4_marker,
    satellites_match, validate_v4_ephemeris_marker, BroadcastRecord, EarthOrientation,
    GlonassParse, GlonassRecord, IonosphereFrame, Layout, NavDiagnostic, NavParse, NavParseError,
    NavVersion, OtherNavBlock, OtherNavBlockKind, SbasRecord, SkippedGlonass, SkippedNavBlock,
    SystemTimeOffset, V4MarkerHeader,
};

/// What part of the format an entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavEntryKind {
    /// A navigation record: a RINEX 2/3 record, or a RINEX 4 `> EPH` frame (and a
    /// frame whose type is not recognized).
    Ephemeris,
    /// A RINEX 4 `> STO` frame.
    SystemTimeOffset,
    /// A RINEX 4 `> EOP` frame.
    EarthOrientation,
    /// A RINEX 4 `> ION` frame.
    Ionosphere,
    /// Body lines that belong to no record or frame (before the first record, for
    /// instance).
    Stray,
}

/// What an entry holds.
// Records are held by value, as the parsers return them.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum NavItem {
    /// A Keplerian record (GPS, QZSS, Galileo, BeiDou, NavIC).
    Ephemeris(BroadcastRecord),
    /// A GLONASS FDMA record.
    Glonass(GlonassRecord),
    /// An SBAS record.
    Sbas(SbasRecord),
    /// A RINEX 4 system time offset frame.
    SystemTimeOffset(SystemTimeOffset),
    /// A RINEX 4 Earth orientation frame.
    EarthOrientation(EarthOrientation),
    /// A RINEX 4 ionosphere frame.
    Ionosphere(IonosphereFrame),
    /// A block that is not decoded; the entry's text holds it.
    Undecoded(UndecodedBlock),
}

/// A block kept as text only.
#[derive(Debug, Clone, PartialEq)]
pub struct UndecodedBlock {
    /// Satellite token of the block, as read; empty for stray lines.
    pub satellite: String,
    /// The RINEX 4 message token, where the block has one.
    pub message_token: Option<String>,
    /// Why the block is not decoded.
    pub reason: UndecodedReason,
}

/// Why a block is kept as text only.
#[derive(Debug, Clone, PartialEq)]
pub enum UndecodedReason {
    /// A message this crate recognizes and does not decode (BeiDou CNAV-1/2/3,
    /// NavIC L1, GLONASS and SBAS messages other than `FDMA` and `SBAS`).
    NotDecoded,
    /// The block could not be read.
    Malformed(NavParseError),
    /// Body lines that belong to no record or frame.
    Stray,
}

/// One block of a navigation file's body, with its text.
#[derive(Debug, Clone, PartialEq)]
pub struct NavEntry {
    /// 1-based line number of the entry's first line (the frame marker in RINEX 4).
    pub line: usize,
    /// What part of the format the entry is.
    pub kind: NavEntryKind,
    /// The system of the entry's satellite, where its letter names one.
    pub system: Option<GnssSystem>,
    /// What the entry holds.
    pub item: NavItem,
    /// Departures from the format the reader read through in this entry.
    pub departures: Vec<NavParseError>,
    /// The entry's lines as read (line terminators removed, the frame marker included);
    /// empty for an entry built in code. The writer restates them when they still read
    /// as [`Self::item`].
    pub text: Vec<String>,
}

impl NavEntry {
    /// An entry built in code, which the writer formats from `item`.
    pub fn new(item: NavItem) -> Self {
        let (kind, system) = match &item {
            NavItem::Ephemeris(record) => {
                (NavEntryKind::Ephemeris, Some(record.satellite_id.system))
            }
            NavItem::Glonass(record) => (NavEntryKind::Ephemeris, Some(record.satellite_id.system)),
            NavItem::Sbas(record) => (NavEntryKind::Ephemeris, Some(record.satellite_id.system)),
            NavItem::SystemTimeOffset(frame) => (
                NavEntryKind::SystemTimeOffset,
                Some(frame.satellite_id.system),
            ),
            NavItem::EarthOrientation(frame) => (
                NavEntryKind::EarthOrientation,
                Some(frame.satellite_id.system),
            ),
            NavItem::Ionosphere(frame) => {
                (NavEntryKind::Ionosphere, Some(frame.satellite_id.system))
            }
            NavItem::Undecoded(_) => (NavEntryKind::Stray, None),
        };
        Self {
            line: 0,
            kind,
            system,
            item,
            departures: Vec::new(),
            text: Vec::new(),
        }
    }

    fn satellite_token(&self) -> String {
        match &self.item {
            NavItem::Ephemeris(record) => record.satellite_id.to_string(),
            NavItem::Glonass(record) => record.satellite_id.to_string(),
            NavItem::Sbas(record) => record.satellite_id.to_string(),
            NavItem::SystemTimeOffset(frame) => frame.satellite_id.to_string(),
            NavItem::EarthOrientation(frame) => frame.satellite_id.to_string(),
            NavItem::Ionosphere(frame) => frame.satellite_id.to_string(),
            NavItem::Undecoded(block) => block.satellite.clone(),
        }
    }

    fn message_token(&self) -> Option<String> {
        match &self.item {
            NavItem::SystemTimeOffset(frame) => Some(frame.message_token.clone()),
            NavItem::EarthOrientation(frame) => Some(frame.message_token.clone()),
            NavItem::Ionosphere(frame) => Some(frame.message_token.clone()),
            NavItem::Undecoded(block) => block.message_token.clone(),
            _ => None,
        }
    }

    fn has_text(&self) -> bool {
        self.text.iter().any(|line| !line.trim().is_empty())
    }
}

/// A whole RINEX navigation file: its header, and every block of its body in order.
#[derive(Debug, Clone, PartialEq)]
pub struct NavFile {
    /// The header.
    pub header: NavHeader,
    /// The body's blocks, in file order.
    pub entries: Vec<NavEntry>,
    /// Whether the file's lines end in `CR LF`; the writer ends lines the same way.
    pub crlf: bool,
    /// Whether the last line ends with a line terminator.
    pub final_newline: bool,
    header_issues: Vec<HeaderIssue>,
}

/// Which records a strict reader reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrictScope {
    Keplerian,
    Glonass,
    Sbas,
}

impl NavFile {
    /// A file with `header` and no entries, ending lines with `LF` and a final newline.
    pub fn new(header: NavHeader) -> Self {
        Self {
            header,
            entries: Vec::new(),
            crlf: false,
            final_newline: true,
            header_issues: Vec::new(),
        }
    }

    /// Header records whose values could not be read. Each record is kept verbatim in
    /// [`NavHeader::other_records`] and its values are absent.
    pub fn header_departures(&self) -> Vec<NavDiagnostic> {
        self.header_issues
            .iter()
            .map(|issue| NavDiagnostic {
                line: issue.line,
                satellite: String::new(),
                error: issue.error.clone(),
            })
            .collect()
    }

    /// Every departure the reader read through, header records first, each with its
    /// line.
    pub fn departures(&self) -> Vec<NavDiagnostic> {
        let mut out = self.header_departures();
        for entry in &self.entries {
            let satellite = entry.satellite_token();
            out.extend(entry.departures.iter().map(|error| NavDiagnostic {
                line: entry.line,
                satellite: satellite.clone(),
                error: error.clone(),
            }));
        }
        out
    }

    /// The Keplerian records, in file order.
    pub fn keplerian_records(&self) -> impl Iterator<Item = BroadcastRecord> + '_ {
        self.entries.iter().filter_map(|entry| match &entry.item {
            NavItem::Ephemeris(record) => Some(*record),
            _ => None,
        })
    }

    /// The GLONASS records, in file order.
    pub fn glonass_records(&self) -> impl Iterator<Item = GlonassRecord> + '_ {
        self.entries.iter().filter_map(|entry| match &entry.item {
            NavItem::Glonass(record) => Some(*record),
            _ => None,
        })
    }

    /// The SBAS records, in file order.
    pub fn sbas_records(&self) -> impl Iterator<Item = SbasRecord> + '_ {
        self.entries.iter().filter_map(|entry| match &entry.item {
            NavItem::Sbas(record) => Some(*record),
            _ => None,
        })
    }

    /// The RINEX 4 ionosphere frames, in file order.
    pub fn ionosphere_frames(&self) -> impl Iterator<Item = &IonosphereFrame> + '_ {
        self.entries.iter().filter_map(|entry| match &entry.item {
            NavItem::Ionosphere(frame) => Some(frame),
            _ => None,
        })
    }

    /// The first problem a strict reader of `scope` refuses: a block of its kind that
    /// could not be read or departs from the format, or a line that belongs to no
    /// record.
    pub(crate) fn first_error(&self, scope: StrictScope) -> Option<NavParseError> {
        for entry in &self.entries {
            if entry.kind == NavEntryKind::Stray {
                if entry.has_text() {
                    return Some(NavParseError::UnexpectedLine { line: entry.line });
                }
                continue;
            }
            let relevant = match scope {
                StrictScope::Keplerian => {
                    entry.kind == NavEntryKind::Ephemeris
                        && entry.system.is_none_or(is_keplerian_system)
                }
                StrictScope::Glonass => {
                    entry.kind == NavEntryKind::Ephemeris
                        && entry.system == Some(GnssSystem::Glonass)
                }
                StrictScope::Sbas => {
                    entry.kind == NavEntryKind::Ephemeris && entry.system == Some(GnssSystem::Sbas)
                }
            };
            if !relevant {
                continue;
            }
            if let NavItem::Undecoded(UndecodedBlock {
                reason: UndecodedReason::Malformed(error),
                ..
            }) = &entry.item
            {
                // A GLONASS slot token that names no satellite is skipped, not refused.
                if scope == StrictScope::Glonass && is_unrepresentable_satellite(error) {
                    continue;
                }
                return Some(error.clone());
            }
            if let Some(error) = entry.departures.first() {
                return Some(error.clone());
            }
        }
        None
    }

    /// The lenient Keplerian view of the file.
    pub(crate) fn nav_parse(&self) -> NavParse {
        let mut parse = NavParse {
            records: Vec::new(),
            skipped: Vec::new(),
            departures: self.header_departures(),
            other: Vec::new(),
        };
        for entry in &self.entries {
            let other = |kind: OtherNavBlockKind| OtherNavBlock {
                line: entry.line,
                satellite: entry.satellite_token(),
                message_token: entry.message_token(),
                kind,
            };
            match &entry.item {
                NavItem::Ephemeris(record) => {
                    parse.records.push(*record);
                    let satellite = entry.satellite_token();
                    parse
                        .departures
                        .extend(entry.departures.iter().map(|error| NavDiagnostic {
                            line: entry.line,
                            satellite: satellite.clone(),
                            error: error.clone(),
                        }));
                }
                NavItem::Glonass(_) => parse.other.push(other(OtherNavBlockKind::Glonass)),
                NavItem::Sbas(_) => parse.other.push(other(OtherNavBlockKind::Sbas)),
                NavItem::SystemTimeOffset(_) => {
                    parse.other.push(other(OtherNavBlockKind::SystemTimeOffset))
                }
                NavItem::EarthOrientation(_) => {
                    parse.other.push(other(OtherNavBlockKind::EarthOrientation))
                }
                NavItem::Ionosphere(_) => parse.other.push(other(OtherNavBlockKind::Ionosphere)),
                NavItem::Undecoded(block) => match &block.reason {
                    UndecodedReason::NotDecoded => {
                        parse.other.push(other(OtherNavBlockKind::NotDecoded))
                    }
                    UndecodedReason::Malformed(error) => parse.skipped.push(SkippedNavBlock {
                        satellite: block.satellite.clone(),
                        message: error.to_string(),
                        line: entry.line,
                    }),
                    UndecodedReason::Stray => {
                        if entry.has_text() {
                            parse.skipped.push(SkippedNavBlock {
                                satellite: String::new(),
                                message: NavParseError::UnexpectedLine { line: entry.line }
                                    .to_string(),
                                line: entry.line,
                            });
                        }
                    }
                },
            }
        }
        parse
    }

    /// The lenient GLONASS view of the file.
    pub(crate) fn glonass_parse(&self) -> GlonassParse {
        let mut parse = GlonassParse::default();
        for entry in &self.entries {
            if entry.system != Some(GnssSystem::Glonass) || entry.kind != NavEntryKind::Ephemeris {
                continue;
            }
            match &entry.item {
                NavItem::Glonass(record) => {
                    parse.records.push(*record);
                    let satellite = entry.satellite_token();
                    parse
                        .departures
                        .extend(entry.departures.iter().map(|error| NavDiagnostic {
                            line: entry.line,
                            satellite: satellite.clone(),
                            error: error.clone(),
                        }));
                }
                NavItem::Undecoded(UndecodedBlock {
                    satellite,
                    reason: UndecodedReason::Malformed(error),
                    ..
                }) => {
                    if is_unrepresentable_satellite(error) {
                        parse.skipped.push(SkippedGlonass {
                            token: satellite.clone(),
                            line: entry.line,
                        });
                    } else {
                        parse.invalid.push(SkippedNavBlock {
                            satellite: satellite.clone(),
                            message: error.to_string(),
                            line: entry.line,
                        });
                    }
                }
                _ => {}
            }
        }
        parse
    }
}

fn is_keplerian_system(system: GnssSystem) -> bool {
    matches!(
        system,
        GnssSystem::Gps
            | GnssSystem::Galileo
            | GnssSystem::BeiDou
            | GnssSystem::Qzss
            | GnssSystem::Navic
    )
}

fn is_unrepresentable_satellite(error: &NavParseError) -> bool {
    matches!(error, NavParseError::BadField { field: "prn", .. })
}

/// Split `text` into lines, noting whether every terminated line ends in `CR LF` and
/// whether the last line is terminated. Where the endings are uniformly `CR LF` the
/// `CR`s are removed; otherwise any `CR` stays in its line's text.
fn split_lines(text: &str) -> (Vec<&str>, bool, bool) {
    if text.is_empty() {
        return (Vec::new(), false, false);
    }
    let final_newline = text.ends_with('\n');
    let mut raw: Vec<&str> = text.split('\n').collect();
    if final_newline {
        raw.pop();
    }
    let terminated = if final_newline {
        raw.len()
    } else {
        raw.len().saturating_sub(1)
    };
    let crlf = terminated > 0 && raw[..terminated].iter().all(|line| line.ends_with('\r'));
    let lines = if crlf {
        raw.iter()
            .enumerate()
            .map(|(index, &line)| {
                if index < terminated {
                    line.strip_suffix('\r').unwrap_or(line)
                } else {
                    line
                }
            })
            .collect()
    } else {
        raw
    };
    (lines, crlf, final_newline)
}

/// A line as the field readers see it: a trailing `CR` removed.
fn content(line: &str) -> &str {
    line.strip_suffix('\r').unwrap_or(line)
}

pub(crate) fn read_nav_file(text: &str) -> Result<NavFile, NavParseError> {
    let (raw_lines, crlf, final_newline) = split_lines(text);
    let lines: Vec<&str> = raw_lines.iter().map(|line| content(line)).collect();
    let header_read = header::read_header(&lines)?;
    let mut header = header_read.header;
    header.text = raw_lines[..header_read.body_start]
        .iter()
        .map(|line| (*line).to_string())
        .collect();
    let version = header.version;
    let file_type = header.file_type;
    let body_start = header_read.body_start;
    let groups = group_body(&lines[body_start..], version.major);
    let mut entries = Vec::with_capacity(groups.len());
    for group in groups {
        let start = body_start + group.start;
        let end = body_start + group.end;
        let block = &lines[start..end];
        let mut entry = if group.stray {
            NavEntry {
                line: start + 1,
                kind: NavEntryKind::Stray,
                system: None,
                item: NavItem::Undecoded(UndecodedBlock {
                    satellite: String::new(),
                    message_token: None,
                    reason: UndecodedReason::Stray,
                }),
                departures: Vec::new(),
                text: Vec::new(),
            }
        } else {
            decode_block(block, version, file_type)
        };
        entry.line = start + 1;
        entry.text = raw_lines[start..end]
            .iter()
            .map(|line| (*line).to_string())
            .collect();
        entries.push(entry);
    }
    Ok(NavFile {
        header,
        entries,
        crlf,
        final_newline,
        header_issues: header_read.issues,
    })
}

/// The RINEX 4 ionosphere frames of a body, in file order, as
/// [`crate::rinex_nav::parse_iono_corrections`] reads them: a frame that cannot be read
/// is an error; a frame of a model not decoded is passed over.
pub(crate) fn body_ionosphere_frames(body: &[&str]) -> Result<Vec<IonosphereFrame>, NavParseError> {
    let mut frames = Vec::new();
    for group in group_body(body, 4) {
        if group.stray {
            continue;
        }
        let block = &body[group.start..group.end];
        let marker = block.first().copied().unwrap_or("");
        let frame_type = marker
            .strip_prefix('>')
            .unwrap_or(marker)
            .split_whitespace()
            .next()
            .unwrap_or("");
        if frame_type != "ION" {
            continue;
        }
        let entry = decode_v4_data_frame(frame_type, marker, block.get(1..).unwrap_or(&[]));
        match entry.item {
            NavItem::Ionosphere(frame) => frames.push(frame),
            NavItem::Undecoded(UndecodedBlock {
                reason: UndecodedReason::Malformed(error),
                ..
            }) => return Err(error),
            _ => {}
        }
    }
    Ok(frames)
}

/// A block's lines, as indices into the body.
struct Group {
    start: usize,
    end: usize,
    stray: bool,
}

/// Split the body into blocks: RINEX 4 at frame markers, RINEX 2/3 at record starts.
/// Lines before the first block form a stray group.
fn group_body(body: &[&str], major: u8) -> Vec<Group> {
    let starts_block = |line: &str| match major {
        4 => is_v4_frame_marker(line),
        3 => is_record_start(line),
        _ => is_v2_numeric_record_start(line) || is_record_start(line),
    };
    let mut groups: Vec<Group> = Vec::new();
    for (index, line) in body.iter().enumerate() {
        if starts_block(line) {
            groups.push(Group {
                start: index,
                end: index + 1,
                stray: false,
            });
        } else if let Some(last) = groups.last_mut() {
            last.end = index + 1;
        } else {
            groups.push(Group {
                start: index,
                end: index + 1,
                stray: true,
            });
        }
    }
    groups
}

/// Whether a version-4 line is a frame marker (`> ...`).
pub(crate) fn is_v4_frame_marker(line: &str) -> bool {
    line.starts_with("> ")
}

/// Decode one block (not stray) of a file of `version` and `file_type`. The block's
/// line numbers and text are set by the caller.
pub(crate) fn decode_block(block: &[&str], version: NavVersion, file_type: char) -> NavEntry {
    if version.major >= 4 {
        decode_v4_frame(block, version)
    } else {
        decode_v2_v3_record(block, version, file_type)
    }
}

fn entry(
    kind: NavEntryKind,
    system: Option<GnssSystem>,
    item: NavItem,
    departures: Vec<NavParseError>,
) -> NavEntry {
    NavEntry {
        line: 0,
        kind,
        system,
        item,
        departures,
        text: Vec::new(),
    }
}

fn malformed(
    kind: NavEntryKind,
    system: Option<GnssSystem>,
    satellite: &str,
    message_token: Option<&str>,
    error: NavParseError,
) -> NavEntry {
    entry(
        kind,
        system,
        NavItem::Undecoded(UndecodedBlock {
            satellite: satellite.to_string(),
            message_token: message_token.map(str::to_string),
            reason: UndecodedReason::Malformed(error),
        }),
        Vec::new(),
    )
}

fn not_decoded(
    kind: NavEntryKind,
    system: Option<GnssSystem>,
    satellite: &str,
    message_token: &str,
) -> NavEntry {
    entry(
        kind,
        system,
        NavItem::Undecoded(UndecodedBlock {
            satellite: satellite.to_string(),
            message_token: Some(message_token.to_string()),
            reason: UndecodedReason::NotDecoded,
        }),
        Vec::new(),
    )
}

/// The satellite a RINEX 2 record with a numeric PRN names, by the file type: `N` GPS
/// (PRN 93-97 are QZSS 193-197, as RTKLIB reads them), `G` GLONASS, `H` SBAS (the PRN
/// less 100), `J` QZSS, `L` Galileo.
fn v2_numeric_satellite(prn_field: &str, file_type: char) -> Result<GnssSatelliteId, ()> {
    let prn = prn_field.trim().parse::<u8>().map_err(|_| ())?;
    let (system, prn) = match file_type {
        'G' => (GnssSystem::Glonass, prn),
        'H' => (GnssSystem::Sbas, prn),
        'J' => (GnssSystem::Qzss, prn),
        'L' => (GnssSystem::Galileo, prn),
        _ if (93..=97).contains(&prn) => (GnssSystem::Qzss, prn - 92),
        _ => (GnssSystem::Gps, prn),
    };
    GnssSatelliteId::new(system, prn).map_err(|_| ())
}

fn decode_v2_v3_record(block: &[&str], version: NavVersion, file_type: char) -> NavEntry {
    let kind = NavEntryKind::Ephemeris;
    let l0 = block.first().copied().unwrap_or("");
    let lettered = l0
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic());
    let (layout, sat, parsed) = if lettered {
        let sat = l0.get(0..3).unwrap_or("").trim();
        let layout = if version.major == 2 {
            Layout::V2Lettered
        } else {
            Layout::V3
        };
        let letter = l0.as_bytes()[0] as char;
        let Some(system) = GnssSystem::from_letter(letter) else {
            return malformed(
                kind,
                None,
                sat,
                None,
                NavParseError::BadField {
                    satellite: sat.to_string(),
                    field: "system",
                },
            );
        };
        (
            layout,
            sat,
            sat.parse::<GnssSatelliteId>().map_err(|_| system),
        )
    } else {
        let sat = l0.get(0..2).unwrap_or("").trim();
        let system = match file_type {
            'G' => GnssSystem::Glonass,
            'H' => GnssSystem::Sbas,
            'J' => GnssSystem::Qzss,
            'L' => GnssSystem::Galileo,
            _ => GnssSystem::Gps,
        };
        (
            Layout::V2,
            sat,
            v2_numeric_satellite(sat, file_type).map_err(|()| system),
        )
    };
    let satellite_id = match parsed {
        Ok(id) => id,
        Err(system) => {
            return malformed(
                kind,
                Some(system),
                sat,
                None,
                NavParseError::BadField {
                    satellite: sat.to_string(),
                    field: "prn",
                },
            )
        }
    };
    let system = satellite_id.system;
    match system {
        GnssSystem::Glonass => {
            match parse_glonass_block(block, satellite_id, sat, version, layout) {
                Ok(decoded) => entry(
                    kind,
                    Some(system),
                    NavItem::Glonass(decoded.value),
                    decoded.departures,
                ),
                Err(error) => malformed(kind, Some(system), sat, None, error),
            }
        }
        GnssSystem::Sbas => match parse_sbas_block(block, satellite_id, sat, version, layout) {
            Ok(decoded) => entry(
                kind,
                Some(system),
                NavItem::Sbas(decoded.value),
                decoded.departures,
            ),
            Err(error) => malformed(kind, Some(system), sat, None, error),
        },
        _ => match parse_keplerian_block(block, satellite_id, sat, None, version, layout) {
            Ok(decoded) => entry(
                kind,
                Some(system),
                NavItem::Ephemeris(decoded.value),
                decoded.departures,
            ),
            Err(error) => malformed(kind, Some(system), sat, None, error),
        },
    }
}

fn system_of_token(token: &str) -> Option<GnssSystem> {
    token.chars().next().and_then(GnssSystem::from_letter)
}

fn body_satellite(body: &[&str]) -> String {
    body.first()
        .and_then(|line| line.get(0..3))
        .unwrap_or("")
        .trim()
        .to_string()
}

fn decode_v4_frame(block: &[&str], version: NavVersion) -> NavEntry {
    let marker = block.first().copied().unwrap_or("");
    let body = block.get(1..).unwrap_or(&[]);
    let frame_type = marker
        .strip_prefix('>')
        .unwrap_or(marker)
        .split_whitespace()
        .next()
        .unwrap_or("");
    match frame_type {
        "STO" | "EOP" | "ION" => decode_v4_data_frame(frame_type, marker, body),
        _ => decode_v4_eph_frame(marker, body, version),
    }
}

fn decode_v4_data_frame(frame_type: &str, marker: &str, body: &[&str]) -> NavEntry {
    let kind = match frame_type {
        "STO" => NavEntryKind::SystemTimeOffset,
        "EOP" => NavEntryKind::EarthOrientation,
        _ => NavEntryKind::Ionosphere,
    };
    let Some((_, sv, msg_token)) = parse_v4_marker(marker) else {
        let sat = body_satellite(body);
        return malformed(
            kind,
            None,
            &sat,
            None,
            NavParseError::BadField {
                satellite: sat.clone(),
                field: "frame marker",
            },
        );
    };
    let system = system_of_token(sv);
    let Ok(satellite_id) = sv.parse::<GnssSatelliteId>() else {
        return malformed(
            kind,
            system,
            sv,
            Some(msg_token),
            NavParseError::BadField {
                satellite: sv.to_string(),
                field: "prn",
            },
        );
    };
    let decoded = match kind {
        NavEntryKind::SystemTimeOffset => parse_sto_frame(body, satellite_id, sv, msg_token)
            .map(|d| (NavItem::SystemTimeOffset(d.value), d.departures)),
        NavEntryKind::EarthOrientation => parse_eop_frame(body, satellite_id, sv, msg_token)
            .map(|d| (NavItem::EarthOrientation(d.value), d.departures)),
        _ => match parse_ion_frame(body, satellite_id, sv, msg_token) {
            Ok(Some(d)) => Ok((NavItem::Ionosphere(d.value), d.departures)),
            Ok(None) => return not_decoded(kind, system, sv, msg_token),
            Err(error) => Err(error),
        },
    };
    match decoded {
        Ok((item, departures)) => entry(kind, system, item, departures),
        Err(error) => malformed(kind, system, sv, Some(msg_token), error),
    }
}

fn decode_v4_eph_frame(marker: &str, body: &[&str], version: NavVersion) -> NavEntry {
    let kind = NavEntryKind::Ephemeris;
    let (sv, msg_token) = match parse_v4_eph_marker(marker, body) {
        Ok(V4MarkerHeader::Eph { sv, msg_token }) => (sv, msg_token),
        // `parse_v4_eph_marker` reports STO/EOP/ION as recognized; they are routed
        // before this point.
        Ok(V4MarkerHeader::RecognizedNonEph) => {
            let sat = body_satellite(body);
            return malformed(
                kind,
                None,
                &sat,
                None,
                NavParseError::BadField {
                    satellite: sat.clone(),
                    field: "frame marker",
                },
            );
        }
        Err(error) => {
            let satellite = match &error {
                NavParseError::BadField { satellite, .. } => satellite.clone(),
                _ => body_satellite(body),
            };
            let system = system_of_token(&satellite);
            return malformed(kind, system, &satellite, None, error);
        }
    };
    let Some(system) = system_of_token(sv) else {
        return malformed(
            kind,
            None,
            sv,
            Some(msg_token),
            NavParseError::BadField {
                satellite: sv.to_string(),
                field: "system",
            },
        );
    };
    let Ok(satellite_id) = sv.parse::<GnssSatelliteId>() else {
        return malformed(
            kind,
            Some(system),
            sv,
            Some(msg_token),
            NavParseError::BadField {
                satellite: sv.to_string(),
                field: "prn",
            },
        );
    };
    if let Some(body_sv) = body
        .first()
        .and_then(|line| line.get(0..3))
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if !satellites_match(sv, body_sv) {
            return malformed(
                kind,
                Some(system),
                sv,
                Some(msg_token),
                NavParseError::BadField {
                    satellite: sv.to_string(),
                    field: "frame marker",
                },
            );
        }
    }
    if let Some(message) = nav_message_from_v4_token(msg_token, system) {
        let decoded = validate_v4_ephemeris_marker(sv, message, body).and_then(|()| {
            if message.is_cnav_family() {
                parse_cnav_block(body, satellite_id, sv, message)
            } else {
                parse_keplerian_block(body, satellite_id, sv, Some(message), version, Layout::V3)
            }
        });
        return match decoded {
            Ok(decoded) => entry(
                kind,
                Some(system),
                NavItem::Ephemeris(decoded.value),
                decoded.departures,
            ),
            Err(error) => malformed(kind, Some(system), sv, Some(msg_token), error),
        };
    }
    match (system, msg_token) {
        (GnssSystem::Glonass, "FDMA") => {
            match parse_glonass_block(body, satellite_id, sv, version, Layout::V3) {
                Ok(decoded) => entry(
                    kind,
                    Some(system),
                    NavItem::Glonass(decoded.value),
                    decoded.departures,
                ),
                Err(error) => malformed(kind, Some(system), sv, Some(msg_token), error),
            }
        }
        (GnssSystem::Sbas, "SBAS") => {
            match parse_sbas_block(body, satellite_id, sv, version, Layout::V3) {
                Ok(decoded) => entry(
                    kind,
                    Some(system),
                    NavItem::Sbas(decoded.value),
                    decoded.departures,
                ),
                Err(error) => malformed(kind, Some(system), sv, Some(msg_token), error),
            }
        }
        (GnssSystem::BeiDou, "CNV1" | "CNV2" | "CNV3")
        | (GnssSystem::Navic, "L1NV")
        | (GnssSystem::Glonass | GnssSystem::Sbas, _) => {
            not_decoded(kind, Some(system), sv, msg_token)
        }
        _ => malformed(
            kind,
            Some(system),
            sv,
            Some(msg_token),
            NavParseError::BadField {
                satellite: sv.to_string(),
                field: "message",
            },
        ),
    }
}
