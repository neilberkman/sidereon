//! Physical signal identity of SSR biases.
//!
//! Galileo HAS and RTCM SSR each name a biased signal by a small index into a
//! table of their own, per GNSS. The two tables disagree: HAS signal 0 of
//! Galileo is E1-B, RTCM SSR signal 0 of Galileo is E1-A. A bias is therefore
//! identified here by the physical signal the index names, a system plus its
//! RINEX 3 band and tracking attribute ([`GnssSignal`]), and an index a table
//! leaves reserved or unassigned keeps its source, system and raw value
//! ([`SsrRawSignal`]) instead of being guessed.

use core::fmt;

use crate::id::GnssSystem;

use super::SsrSource;

/// Band and tracking attribute of a GNSS signal: the two characters that follow
/// the observation type in a RINEX 3 observation code (`1C` in `C1C` and `L1C`).
///
/// Codes follow RINEX 3.04 and later, where BeiDou B1I is band `2` and band `1`
/// is B1C. [`SignalCode::from_rinex`] reads a RINEX 3.02 BeiDou code into that
/// convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SignalCode {
    band: char,
    attribute: char,
}

impl SignalCode {
    /// Signal code from a RINEX band digit (`1` to `9`) and an upper-case
    /// tracking attribute letter, or `None` outside those.
    pub const fn new(band: char, attribute: char) -> Option<Self> {
        if matches!(band, '1'..='9') && attribute.is_ascii_uppercase() {
            Some(Self { band, attribute })
        } else {
            None
        }
    }

    /// Parse a band and attribute (`1C`) or a full RINEX 3 observation code
    /// whose type letter is `C`, `L`, `D` or `S` (`C1C`, `L2W`). The code is
    /// taken as written, in the RINEX 3.04 convention.
    pub fn parse(code: &str) -> Option<Self> {
        let chars: Vec<char> = code.chars().collect();
        match chars.as_slice() {
            [band, attribute] => Self::new(*band, *attribute),
            ['C' | 'L' | 'D' | 'S', band, attribute] => Self::new(*band, *attribute),
            _ => None,
        }
    }

    /// Signal code of a RINEX observation code read from a file of
    /// `rinex_version`.
    ///
    /// A RINEX 3.02 file writes BeiDou B1I as band `1`, which RINEX 3.03 and
    /// later write as band `2`; the band is moved to `2` for a BeiDou code of a
    /// 3.02 file, as RTKLIB `decode_obsh` moves it. A RINEX 2 code carries no
    /// tracking attribute, so it names no signal code and `None` is returned.
    pub fn from_rinex(system: GnssSystem, code: &str, rinex_version: f64) -> Option<Self> {
        if rinex_version.is_nan() || rinex_version < 3.0 {
            return None;
        }
        let parsed = Self::parse(code)?;
        if system == GnssSystem::BeiDou
            && crate::frequencies::is_rinex_302(rinex_version)
            && parsed.band == '1'
        {
            return Self::new('2', parsed.attribute);
        }
        Some(parsed)
    }

    /// Signal code of an RTCM MSM signal-mask id, from RTKLIB `msm_sig_*`,
    /// including its tentative extensions
    /// ([`crate::rtcm::msm_signal_rinex_code`]); `None` for an id the tables leave
    /// unassigned.
    pub fn from_msm_signal(system: GnssSystem, signal_id: u8) -> Option<Self> {
        Self::parse(crate::rtcm::msm_signal_rinex_code(system, signal_id)?)
    }

    /// RINEX band digit.
    pub const fn band(self) -> char {
        self.band
    }

    /// RINEX tracking attribute letter.
    pub const fn attribute(self) -> char {
        self.attribute
    }
}

impl fmt::Display for SignalCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.band, self.attribute)
    }
}

/// A physical GNSS signal: a system and the RINEX 3 band and tracking attribute
/// of the signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GnssSignal {
    system: GnssSystem,
    code: SignalCode,
}

impl GnssSignal {
    /// The signal `code` of `system`.
    pub const fn new(system: GnssSystem, code: SignalCode) -> Self {
        Self { system, code }
    }

    /// The system.
    pub const fn system(self) -> GnssSystem {
        self.system
    }

    /// The band and tracking attribute.
    pub const fn code(self) -> SignalCode {
        self.code
    }

    /// Carrier frequency in hertz of the signal's band, from
    /// [`crate::frequencies::rinex_band_frequency_hz`]. A GLONASS G1 or G2 FDMA
    /// signal resolves only with the satellite's frequency channel.
    pub fn carrier_frequency_hz(self, glonass_channel: Option<i8>) -> Option<f64> {
        crate::frequencies::rinex_band_frequency_hz(self.system, self.code.band, glonass_channel)
    }
}

impl fmt::Display for GnssSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.system, self.code)
    }
}

/// A bias signal index as its source transmitted it: the stream, the GNSS whose
/// table the index belongs to, and the raw index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SsrRawSignal {
    source: SsrSource,
    system: GnssSystem,
    index: u8,
}

impl SsrRawSignal {
    /// Raw signal `index` of `system` transmitted by `source`.
    pub const fn new(source: SsrSource, system: GnssSystem, index: u8) -> Self {
        Self {
            source,
            system,
            index,
        }
    }

    /// A Galileo HAS signal index (HAS SIS ICD Table 20).
    pub const fn galileo_has(system: GnssSystem, index: u8) -> Self {
        Self::new(SsrSource::GalileoHas, system, index)
    }

    /// An RTCM SSR signal and tracking mode identifier.
    pub const fn rtcm_ssr(system: GnssSystem, index: u8) -> Self {
        Self::new(SsrSource::RtcmSsr, system, index)
    }

    /// The stream whose table the index belongs to.
    pub const fn source(self) -> SsrSource {
        self.source
    }

    /// The GNSS whose table the index belongs to.
    pub const fn system(self) -> GnssSystem {
        self.system
    }

    /// The raw index.
    pub const fn index(self) -> u8 {
        self.index
    }

    /// The physical signal the source's table assigns the index, or `None` for
    /// an index the table leaves reserved or unassigned.
    pub fn physical(self) -> Option<GnssSignal> {
        match self.source {
            SsrSource::GalileoHas => has_signal(self.system, self.index),
            SsrSource::RtcmSsr => rtcm_ssr_signal(self.system, self.index),
        }
    }

    /// The key a bias on this signal is stored under: its physical signal, or
    /// the raw signal itself when the source's table assigns none.
    pub fn key(self) -> SsrSignalKey {
        match self.physical() {
            Some(signal) => SsrSignalKey::Physical(signal),
            None => SsrSignalKey::Unknown(self),
        }
    }
}

impl fmt::Display for SsrRawSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let source = match self.source {
            SsrSource::GalileoHas => "Galileo HAS",
            SsrSource::RtcmSsr => "RTCM SSR",
        };
        write!(f, "{source} {} signal {}", self.system, self.index)
    }
}

/// The key an SSR bias is stored and queried under.
///
/// A bias whose source table assigns its index a physical signal is keyed by
/// that signal, so Galileo HAS and RTCM SSR records of one physical signal share
/// an entry and are arbitrated by arrival, while records of different physical
/// signals never share one. A bias on an index the table leaves reserved or
/// unassigned is keyed by its raw source-qualified signal: it is stored, and
/// never applied to an observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SsrSignalKey {
    /// A physical signal.
    Physical(GnssSignal),
    /// A raw signal index its source's table assigns no physical signal.
    Unknown(SsrRawSignal),
}

impl SsrSignalKey {
    /// The physical signal, when the key names one.
    pub const fn physical(self) -> Option<GnssSignal> {
        match self {
            Self::Physical(signal) => Some(signal),
            Self::Unknown(_) => None,
        }
    }

    /// The GNSS of the signal.
    pub const fn system(self) -> GnssSystem {
        match self {
            Self::Physical(signal) => signal.system,
            Self::Unknown(raw) => raw.system,
        }
    }

    /// The key a store holds this signal under: a raw signal whose source table
    /// assigns it a physical signal is that physical signal.
    pub fn canonical(self) -> Self {
        match self {
            Self::Physical(_) => self,
            Self::Unknown(raw) => raw.key(),
        }
    }
}

impl From<GnssSignal> for SsrSignalKey {
    fn from(signal: GnssSignal) -> Self {
        Self::Physical(signal)
    }
}

impl From<SsrRawSignal> for SsrSignalKey {
    fn from(raw: SsrRawSignal) -> Self {
        raw.key()
    }
}

impl fmt::Display for SsrSignalKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Physical(signal) => write!(f, "{signal}"),
            Self::Unknown(raw) => write!(f, "{raw} (unassigned)"),
        }
    }
}

const fn code(band: char, attribute: char) -> Option<SignalCode> {
    Some(SignalCode { band, attribute })
}

const N: Option<SignalCode> = None;

/// Galileo HAS signal index table for GPS, HAS SIS ICD Issue 1.0 Table 20.
/// Indices 1, 2, 10, 14 and 15 are reserved.
#[rustfmt::skip]
const HAS_GPS: [Option<SignalCode>; 16] = [
    code('1', 'C'), // L1 C/A
    N,              // reserved
    N,              // reserved
    code('1', 'S'), // L1C(D)
    code('1', 'L'), // L1C(P)
    code('1', 'X'), // L1C(D+P)
    code('2', 'S'), // L2 CM
    code('2', 'L'), // L2 CL
    code('2', 'X'), // L2 CM+CL
    code('2', 'P'), // L2 P
    N,              // reserved
    code('5', 'I'), // L5 I
    code('5', 'Q'), // L5 Q
    code('5', 'X'), // L5 I+Q
    N,              // reserved
    N,              // reserved
];

/// Galileo HAS signal index table for Galileo, HAS SIS ICD Issue 1.0 Table 20.
/// Index 15 is reserved.
#[rustfmt::skip]
const HAS_GALILEO: [Option<SignalCode>; 16] = [
    code('1', 'B'), // E1-B I/NAV OS
    code('1', 'C'), // E1-C
    code('1', 'X'), // E1-B + E1-C
    code('5', 'I'), // E5a-I F/NAV OS
    code('5', 'Q'), // E5a-Q
    code('5', 'X'), // E5a-I + E5a-Q
    code('7', 'I'), // E5b-I I/NAV OS
    code('7', 'Q'), // E5b-Q
    code('7', 'X'), // E5b-I + E5b-Q
    code('8', 'I'), // E5-I
    code('8', 'Q'), // E5-Q
    code('8', 'X'), // E5-I + E5-Q
    code('6', 'B'), // E6-B C/NAV HAS
    code('6', 'C'), // E6-C
    code('6', 'X'), // E6-B + E6-C
    N,              // reserved
];

/// RTCM SSR GPS signal and tracking mode identifiers, RTKLIB `ssr_sig_gps`.
#[rustfmt::skip]
const RTCM_GPS: [Option<SignalCode>; 16] = [
    code('1', 'C'), code('1', 'P'), code('1', 'W'), code('1', 'S'),
    code('1', 'L'), code('2', 'C'), code('2', 'D'), code('2', 'S'),
    code('2', 'L'), code('2', 'X'), code('2', 'P'), code('2', 'W'),
    N,              N,              code('5', 'I'), code('5', 'Q'),
];

/// RTCM SSR GLONASS signal and tracking mode identifiers, RTKLIB `ssr_sig_glo`.
#[rustfmt::skip]
const RTCM_GLONASS: [Option<SignalCode>; 10] = [
    code('1', 'C'), code('1', 'P'), code('2', 'C'), code('2', 'P'),
    code('4', 'A'), code('4', 'B'), code('6', 'A'), code('6', 'B'),
    code('3', 'I'), code('3', 'Q'),
];

/// RTCM SSR Galileo signal and tracking mode identifiers, RTKLIB `ssr_sig_gal`.
#[rustfmt::skip]
const RTCM_GALILEO: [Option<SignalCode>; 17] = [
    code('1', 'A'), code('1', 'B'), code('1', 'C'), N,
    N,              code('5', 'I'), code('5', 'Q'), N,
    code('7', 'I'), code('7', 'Q'), N,              code('8', 'I'),
    code('8', 'Q'), N,              code('6', 'A'), code('6', 'B'),
    code('6', 'C'),
];

/// RTCM SSR QZSS signal and tracking mode identifiers, RTKLIB `ssr_sig_qzs`.
#[rustfmt::skip]
const RTCM_QZSS: [Option<SignalCode>; 18] = [
    code('1', 'C'), code('1', 'S'), code('1', 'L'), code('2', 'S'),
    code('2', 'L'), N,              code('5', 'I'), code('5', 'Q'),
    N,              code('6', 'S'), code('6', 'L'), N,
    N,              N,              N,              N,
    N,              code('6', 'E'),
];

/// RTCM SSR BeiDou signal and tracking mode identifiers, RTKLIB `ssr_sig_cmp`,
/// in the RINEX 3.04 band convention (B1I is band `2`).
#[rustfmt::skip]
const RTCM_BEIDOU: [Option<SignalCode>; 19] = [
    code('2', 'I'), code('2', 'Q'), N,              code('6', 'I'),
    code('6', 'Q'), N,              code('7', 'I'), code('7', 'Q'),
    N,              code('1', 'D'), code('1', 'P'), N,
    code('5', 'D'), code('5', 'P'), N,              code('1', 'A'),
    N,              N,              code('6', 'A'),
];

/// RTCM SSR SBAS signal and tracking mode identifiers, RTKLIB `ssr_sig_sbs`.
#[rustfmt::skip]
const RTCM_SBAS: [Option<SignalCode>; 3] = [code('1', 'C'), code('5', 'I'), code('5', 'Q')];

fn table_entry(table: &[Option<SignalCode>], system: GnssSystem, index: u8) -> Option<GnssSignal> {
    let code = table.get(usize::from(index)).copied().flatten()?;
    Some(GnssSignal::new(system, code))
}

/// Physical signal of a Galileo HAS signal index, HAS SIS ICD Issue 1.0
/// Table 20. HAS corrects GPS and Galileo only; a reserved index, an index
/// outside the 16-entry table and any other system give `None`.
pub fn has_signal(system: GnssSystem, index: u8) -> Option<GnssSignal> {
    match system {
        GnssSystem::Gps => table_entry(&HAS_GPS, system, index),
        GnssSystem::Galileo => table_entry(&HAS_GALILEO, system, index),
        _ => None,
    }
}

/// Physical signal of an RTCM SSR signal and tracking mode identifier, from the
/// RTKLIB `ssr_sig_*` tables of RTCM 10403.3. An identifier those tables leave
/// unassigned, and NavIC, give `None`.
pub fn rtcm_ssr_signal(system: GnssSystem, index: u8) -> Option<GnssSignal> {
    let table: &[Option<SignalCode>] = match system {
        GnssSystem::Gps => &RTCM_GPS,
        GnssSystem::Glonass => &RTCM_GLONASS,
        GnssSystem::Galileo => &RTCM_GALILEO,
        GnssSystem::Qzss => &RTCM_QZSS,
        GnssSystem::BeiDou => &RTCM_BEIDOU,
        GnssSystem::Sbas => &RTCM_SBAS,
        GnssSystem::Navic => return None,
    };
    table_entry(table, system, index)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn codes(
        lookup: fn(GnssSystem, u8) -> Option<GnssSignal>,
        system: GnssSystem,
        width: u8,
    ) -> Vec<Option<String>> {
        (0..width)
            .map(|index| {
                lookup(system, index).map(|signal| {
                    assert_eq!(signal.system(), system);
                    signal.code().to_string()
                })
            })
            .collect()
    }

    fn expected(entries: &[Option<&str>]) -> Vec<Option<String>> {
        entries.iter().map(|e| e.map(str::to_string)).collect()
    }

    /// Every HAS index of HAS SIS ICD Issue 1.0 Table 20, and nothing past it.
    #[test]
    fn has_signal_index_table_matches_icd_table_20() {
        assert_eq!(
            codes(has_signal, GnssSystem::Gps, 17),
            expected(&[
                Some("1C"), // L1 C/A
                None,
                None,
                Some("1S"), // L1C(D)
                Some("1L"), // L1C(P)
                Some("1X"), // L1C(D+P)
                Some("2S"), // L2 CM
                Some("2L"), // L2 CL
                Some("2X"), // L2 CM+CL
                Some("2P"), // L2 P
                None,
                Some("5I"), // L5 I
                Some("5Q"), // L5 Q
                Some("5X"), // L5 I+Q
                None,
                None,
                None, // outside the 16-bit signal mask
            ])
        );
        assert_eq!(
            codes(has_signal, GnssSystem::Galileo, 17),
            expected(&[
                Some("1B"), // E1-B
                Some("1C"), // E1-C
                Some("1X"), // E1-B + E1-C
                Some("5I"), // E5a-I
                Some("5Q"), // E5a-Q
                Some("5X"), // E5a-I + E5a-Q
                Some("7I"), // E5b-I
                Some("7Q"), // E5b-Q
                Some("7X"), // E5b-I + E5b-Q
                Some("8I"), // E5-I
                Some("8Q"), // E5-Q
                Some("8X"), // E5-I + E5-Q
                Some("6B"), // E6-B
                Some("6C"), // E6-C
                Some("6X"), // E6-B + E6-C
                None,
                None,
            ])
        );
        for system in [
            GnssSystem::Glonass,
            GnssSystem::BeiDou,
            GnssSystem::Qzss,
            GnssSystem::Navic,
            GnssSystem::Sbas,
        ] {
            assert!((0..=u8::MAX).all(|index| has_signal(system, index).is_none()));
        }
    }

    /// The RTCM SSR tables are RTKLIB's `ssr_sig_*`, entry for entry, through the
    /// 32 identifiers a 5-bit field holds.
    #[test]
    fn rtcm_ssr_signal_tables_match_rtklib() {
        let pad = |mut v: Vec<Option<&'static str>>| {
            v.resize(32, None);
            expected(&v)
        };
        assert_eq!(
            codes(rtcm_ssr_signal, GnssSystem::Gps, 32),
            pad(vec![
                Some("1C"),
                Some("1P"),
                Some("1W"),
                Some("1S"),
                Some("1L"),
                Some("2C"),
                Some("2D"),
                Some("2S"),
                Some("2L"),
                Some("2X"),
                Some("2P"),
                Some("2W"),
                None,
                None,
                Some("5I"),
                Some("5Q"),
            ])
        );
        assert_eq!(
            codes(rtcm_ssr_signal, GnssSystem::Glonass, 32),
            pad(vec![
                Some("1C"),
                Some("1P"),
                Some("2C"),
                Some("2P"),
                Some("4A"),
                Some("4B"),
                Some("6A"),
                Some("6B"),
                Some("3I"),
                Some("3Q"),
            ])
        );
        assert_eq!(
            codes(rtcm_ssr_signal, GnssSystem::Galileo, 32),
            pad(vec![
                Some("1A"),
                Some("1B"),
                Some("1C"),
                None,
                None,
                Some("5I"),
                Some("5Q"),
                None,
                Some("7I"),
                Some("7Q"),
                None,
                Some("8I"),
                Some("8Q"),
                None,
                Some("6A"),
                Some("6B"),
                Some("6C"),
            ])
        );
        assert_eq!(
            codes(rtcm_ssr_signal, GnssSystem::Qzss, 32),
            pad(vec![
                Some("1C"),
                Some("1S"),
                Some("1L"),
                Some("2S"),
                Some("2L"),
                None,
                Some("5I"),
                Some("5Q"),
                None,
                Some("6S"),
                Some("6L"),
                None,
                None,
                None,
                None,
                None,
                None,
                Some("6E"),
            ])
        );
        assert_eq!(
            codes(rtcm_ssr_signal, GnssSystem::BeiDou, 32),
            pad(vec![
                Some("2I"),
                Some("2Q"),
                None,
                Some("6I"),
                Some("6Q"),
                None,
                Some("7I"),
                Some("7Q"),
                None,
                Some("1D"),
                Some("1P"),
                None,
                Some("5D"),
                Some("5P"),
                None,
                Some("1A"),
                None,
                None,
                Some("6A"),
            ])
        );
        assert_eq!(
            codes(rtcm_ssr_signal, GnssSystem::Sbas, 32),
            pad(vec![Some("1C"), Some("5I"), Some("5Q")])
        );
        assert!((0..32).all(|index| rtcm_ssr_signal(GnssSystem::Navic, index).is_none()));
    }

    /// The same raw index names different physical signals in the two sources,
    /// and one physical signal can be named by different raw indices.
    #[test]
    fn raw_indices_key_by_physical_signal_across_sources() {
        let has_e1b = SsrRawSignal::galileo_has(GnssSystem::Galileo, 0);
        let rtcm_e1a = SsrRawSignal::rtcm_ssr(GnssSystem::Galileo, 0);
        let rtcm_e1b = SsrRawSignal::rtcm_ssr(GnssSystem::Galileo, 1);
        assert_ne!(has_e1b.key(), rtcm_e1a.key());
        assert_eq!(has_e1b.key(), rtcm_e1b.key());

        let has_l2p = SsrRawSignal::galileo_has(GnssSystem::Gps, 9);
        let rtcm_l2x = SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 9);
        let rtcm_l2p = SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 10);
        let rtcm_l2w = SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 11);
        assert_ne!(has_l2p.key(), rtcm_l2x.key());
        assert_eq!(has_l2p.key(), rtcm_l2p.key());
        assert_ne!(has_l2p.key(), rtcm_l2w.key());

        let reserved = SsrRawSignal::galileo_has(GnssSystem::Gps, 1);
        assert_eq!(reserved.key(), SsrSignalKey::Unknown(reserved));
        assert_eq!(SsrSignalKey::Unknown(has_l2p).canonical(), has_l2p.key());
        assert_eq!(SsrSignalKey::Unknown(reserved).canonical(), reserved.key());
        // RTCM SSR GPS index 12 is unassigned while HAS GPS index 12 is L5 Q.
        let has_12 = SsrRawSignal::galileo_has(GnssSystem::Gps, 12);
        let rtcm_12 = SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 12);
        assert_eq!(rtcm_12.key(), SsrSignalKey::Unknown(rtcm_12));
        assert_eq!(
            has_12.key().physical().map(|s| s.code().to_string()),
            Some("5Q".to_string())
        );
        assert_ne!(has_12.key(), rtcm_12.key());
        // Index 16 is past the HAS table and unassigned in RTCM SSR: each keeps its own
        // source, so the two never share an entry.
        let has_16 = SsrRawSignal::galileo_has(GnssSystem::Gps, 16);
        let rtcm_16 = SsrRawSignal::rtcm_ssr(GnssSystem::Gps, 16);
        assert_eq!(has_16.key(), SsrSignalKey::Unknown(has_16));
        assert_eq!(rtcm_16.key(), SsrSignalKey::Unknown(rtcm_16));
        assert_ne!(has_16.key(), rtcm_16.key());
    }

    #[test]
    fn signal_codes_parse_rinex_observation_codes() {
        let c1c = SignalCode::new('1', 'C').unwrap();
        assert_eq!(SignalCode::parse("1C"), Some(c1c));
        assert_eq!(SignalCode::parse("C1C"), Some(c1c));
        assert_eq!(SignalCode::parse("L1C"), Some(c1c));
        assert_eq!(SignalCode::parse("C1"), None);
        assert_eq!(SignalCode::parse("P2"), None);
        assert_eq!(SignalCode::parse("X1C"), None);
        assert_eq!(SignalCode::parse("C0C"), None);
        assert_eq!(SignalCode::parse("C1c"), None);
        assert_eq!(SignalCode::from_rinex(GnssSystem::Gps, "C2W", 2.11), None);
        assert_eq!(
            SignalCode::from_rinex(GnssSystem::Gps, "C2W", 3.04),
            SignalCode::new('2', 'W')
        );
        // RINEX 3.02 BeiDou B1I is band 1; later versions write it as band 2.
        assert_eq!(
            SignalCode::from_rinex(GnssSystem::BeiDou, "C1I", 3.02),
            SignalCode::new('2', 'I')
        );
        assert_eq!(
            SignalCode::from_rinex(GnssSystem::BeiDou, "C1P", 3.04),
            SignalCode::new('1', 'P')
        );
        assert_eq!(
            SignalCode::from_rinex(GnssSystem::Gps, "C1C", 3.02),
            Some(c1c)
        );
        assert_eq!(SignalCode::from_msm_signal(GnssSystem::Gps, 2), Some(c1c));
        assert_eq!(
            SignalCode::from_msm_signal(GnssSystem::Gps, 10),
            SignalCode::new('2', 'W')
        );
        assert_eq!(SignalCode::from_msm_signal(GnssSystem::Gps, 0), None);
    }

    #[test]
    fn carrier_frequency_follows_the_band_and_glonass_channel() {
        let gps_l2p = GnssSignal::new(GnssSystem::Gps, SignalCode::new('2', 'P').unwrap());
        assert_eq!(
            gps_l2p.carrier_frequency_hz(None),
            Some(crate::constants::F_L2_HZ)
        );
        let glo_g1 = GnssSignal::new(GnssSystem::Glonass, SignalCode::new('1', 'C').unwrap());
        assert_eq!(glo_g1.carrier_frequency_hz(None), None);
        assert_eq!(
            glo_g1.carrier_frequency_hz(Some(-3)),
            Some(1_602_000_000.0 - 3.0 * 562_500.0)
        );
        let glo_g2 = GnssSignal::new(GnssSystem::Glonass, SignalCode::new('2', 'P').unwrap());
        assert_eq!(
            glo_g2.carrier_frequency_hz(Some(5)),
            Some(1_246_000_000.0 + 5.0 * 437_500.0)
        );
    }
}
