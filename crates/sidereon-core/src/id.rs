//! GNSS satellite identification.
//!
//! Foundational identifier types only - no domain numerics live here.

use core::fmt;

/// A GNSS constellation (satellite system).
///
/// Variants follow the RINEX / IGS single-letter system identifiers, which are
/// the canonical keys used throughout SP3, RINEX, and IONEX products:
///
/// | Letter | Variant                  | System                          |
/// |--------|--------------------------|---------------------------------|
/// | `G`    | [`GnssSystem::Gps`]      | GPS (US)                        |
/// | `R`    | [`GnssSystem::Glonass`]  | GLONASS (RU)                    |
/// | `E`    | [`GnssSystem::Galileo`]  | Galileo (EU)                    |
/// | `C`    | [`GnssSystem::BeiDou`]   | BeiDou (CN)                     |
/// | `J`    | [`GnssSystem::Qzss`]     | QZSS (JP)                       |
/// | `I`    | [`GnssSystem::Navic`]    | NavIC / IRNSS (IN)              |
/// | `S`    | [`GnssSystem::Sbas`]     | SBAS (geostationary augmentation) |
///
/// Note that timekeeping is constellation-tagged separately (`TimeScale`):
/// GPS/Galileo/BeiDou each run their own system time, and GNSS week numbers are
/// **not** cross-comparable between systems.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum GnssSystem {
    /// GPS (United States), RINEX letter `G`.
    Gps,
    /// GLONASS (Russia), RINEX letter `R`.
    Glonass,
    /// Galileo (European Union), RINEX letter `E`.
    Galileo,
    /// BeiDou (China), RINEX letter `C`.
    BeiDou,
    /// QZSS (Japan), RINEX letter `J`.
    Qzss,
    /// NavIC / IRNSS (India), RINEX letter `I`.
    Navic,
    /// SBAS geostationary augmentation, RINEX letter `S`.
    Sbas,
}

impl GnssSystem {
    /// The canonical display label (constellation acronyms uppercase, proper
    /// names as styled): GPS, GLONASS, Galileo, BeiDou, QZSS, NavIC, SBAS.
    pub const fn as_str(&self) -> &'static str {
        match *self {
            GnssSystem::Gps => "GPS",
            GnssSystem::Glonass => "GLONASS",
            GnssSystem::Galileo => "Galileo",
            GnssSystem::BeiDou => "BeiDou",
            GnssSystem::Qzss => "QZSS",
            GnssSystem::Navic => "NavIC",
            GnssSystem::Sbas => "SBAS",
        }
    }

    /// The canonical RINEX / IGS single-letter system identifier.
    pub const fn letter(self) -> char {
        match self {
            GnssSystem::Gps => 'G',
            GnssSystem::Glonass => 'R',
            GnssSystem::Galileo => 'E',
            GnssSystem::BeiDou => 'C',
            GnssSystem::Qzss => 'J',
            GnssSystem::Navic => 'I',
            GnssSystem::Sbas => 'S',
        }
    }

    /// Parse a RINEX / IGS single-letter system identifier.
    ///
    /// Returns `None` for an unrecognized letter. Accepts uppercase letters
    /// only, as emitted by SP3/RINEX/IONEX products.
    pub const fn from_letter(letter: char) -> Option<Self> {
        match letter {
            'G' => Some(GnssSystem::Gps),
            'R' => Some(GnssSystem::Glonass),
            'E' => Some(GnssSystem::Galileo),
            'C' => Some(GnssSystem::BeiDou),
            'J' => Some(GnssSystem::Qzss),
            'I' => Some(GnssSystem::Navic),
            'S' => Some(GnssSystem::Sbas),
            _ => None,
        }
    }
}

impl fmt::Display for GnssSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A satellite identifier: a constellation plus its within-system PRN/slot.
///
/// This is the `GnssSatelliteId { system, prn }` foundational type from the
/// spec (line 112). The `prn` is the within-constellation satellite number as
/// it appears in the product (e.g. the `01` in the SP3/RINEX token `G01`); it
/// is only meaningful in combination with [`GnssSatelliteId::system`].
///
/// [`GnssSatelliteId::new`] and the [`FromStr`](core::str::FromStr) impl accept
/// the shared SP3-d satellite-token range `01..=99` for every constellation.
/// That range is file-token syntax, not a roster of on-orbit satellites and not
/// a wire-field width: a `GnssSatelliteId` says only that the token was
/// spellable. Narrower domains - the SBAS broadcast PRN window, the Galileo HAS
/// 40-satellite mask, the RTCM raw satellite-id field widths - are checked by
/// the modules that own them, at the point of conversion.
///
/// Both fields are public and the derived `Deserialize` writes them directly,
/// so a value that never passed [`GnssSatelliteId::new`] can exist: a struct
/// literal or a deserialized record may hold `prn` `0`, `100` or `255`. There
/// is therefore no constructor-enforced invariant to rely on. Code that
/// converts an identifier back to a bounded wire field must re-check the value
/// itself rather than assume the constructor already did.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct GnssSatelliteId {
    /// The constellation this satellite belongs to.
    pub system: GnssSystem,
    /// The within-constellation PRN / slot number (e.g. `1` for `G01`).
    pub prn: u8,
}

/// Error returned when constructing a GNSS satellite identifier from invalid input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SatelliteIdError {
    /// The PRN is outside the shared satellite-token range.
    #[error("invalid GNSS satellite {field}: {reason}")]
    InvalidInput {
        /// The rejected input name; [`GnssSatelliteId::new`] sets this to
        /// `"prn"` for its shared-token range check.
        field: &'static str,
        /// The diagnostic from [`GnssSatelliteId::new`] when the PRN is not a
        /// spellable two-digit token: `"outside the 1..=99 satellite-token
        /// range"`.
        reason: &'static str,
    },
}

const fn invalid_input(field: &'static str, reason: &'static str) -> SatelliteIdError {
    SatelliteIdError::InvalidInput { field, reason }
}

impl GnssSatelliteId {
    /// Construct an identifier from a constellation and PRN.
    ///
    /// Accepts the shared SP3-d satellite-token range `1..=99` for every
    /// constellation and rejects `0` and `100..=255`, which no two-digit token
    /// can spell. It does not check the PRN against an on-orbit roster or
    /// against any protocol's field width; see the type-level note.
    pub const fn new(system: GnssSystem, prn: u8) -> Result<Self, SatelliteIdError> {
        if !is_shared_token_prn(prn) {
            return Err(invalid_input(
                "prn",
                "outside the 1..=99 satellite-token range",
            ));
        }
        Ok(Self { system, prn })
    }
}

impl fmt::Display for GnssSatelliteId {
    /// Renders the canonical SP3/RINEX token, e.g. `G01`, `E12`, `C30`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:02}", self.system.letter(), self.prn)
    }
}

/// Error returned when a string cannot be parsed as a [`GnssSatelliteId`].
///
/// Produced by the [`FromStr`](core::str::FromStr) implementation when the token
/// is empty, has no recognized constellation letter, or lacks a numeric
/// within-system PRN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseSatelliteIdError;

impl fmt::Display for ParseSatelliteIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid GNSS satellite token")
    }
}

impl std::error::Error for ParseSatelliteIdError {}

impl core::str::FromStr for GnssSatelliteId {
    type Err = ParseSatelliteIdError;

    /// Parse a SP3/RINEX satellite token (`G01`, `G 1`, `G1`, `E12`, `C30`): a
    /// constellation letter followed by the within-system PRN. Whitespace around
    /// the token and around the PRN is ignored, matching the SP3/RINEX field
    /// readers. This is the single canonical satellite-token parser; the
    /// SP3/RINEX/DGNSS readers delegate to it.
    ///
    /// The PRN must be one or two ASCII digits in `1..=99`: `G00`, `G001` and
    /// `G100` are all rejected, so no token can name `prn` `0` or `>= 100`. The
    /// unpadded `G1` and space-padded `G 1` spellings stay accepted for the
    /// fixed-column readers even though SP3-d itself requires `G01` on output;
    /// [`Display`](fmt::Display) always writes the padded canonical form.
    fn from_str(token: &str) -> Result<Self, Self::Err> {
        let token = token.trim();
        let first = token.chars().next().ok_or(ParseSatelliteIdError)?;
        let system = GnssSystem::from_letter(first).ok_or(ParseSatelliteIdError)?;
        let prn_token = token[first.len_utf8()..].trim();
        if !(1..=2).contains(&prn_token.len()) || !prn_token.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseSatelliteIdError);
        }
        let prn = prn_token.parse::<u8>().map_err(|_| ParseSatelliteIdError)?;
        Self::new(system, prn).map_err(|_| ParseSatelliteIdError)
    }
}

/// The shared satellite-token PRN range: `1..=99`, for every constellation.
///
/// SP3-d defines a satellite identifier as "a letter followed by a 2-digit
/// integer between 01 and 99" and names `Gnn`/`Rnn`/`Snn`/`Enn`/`Cnn`/`Inn`/
/// `Jnn` as the system spellings, so `01..=99` is the token syntax every
/// SP3/RINEX/IONEX/bias reader has to be able to hold. Products really do carry
/// tokens above the operational roster (extended GLONASS slots `R28`/`R29`,
/// GPS `G34` in CODE DCB tables), and dropping them loses real data.
///
/// This range is deliberately not a constellation roster and not a wire-field
/// width. It says which tokens a product may legally spell - not which
/// satellites are on orbit, and not which values a given protocol field can
/// carry. Every narrower domain is enforced at the boundary that owns it, with
/// its own primary source:
///
/// - SBAS broadcast PRN `120..=158` <-> stored slot `20..=58`, and the DO-229
///   PRN mask layout (`sbas::store`).
/// - Galileo HAS satellites `1..=40` for GPS and Galileo, Table 19 of the HAS
///   SIS ICD: the 40-bit mask bounds decoding, and the mask entries are checked
///   when a satellite is formed from one and when a message is encoded (`has`).
/// - RTCM ephemeris raw satellite fields: 6 bits for 1019/1020/1042/1045/1046,
///   4 bits for QZSS 1044, checked on conversion and on encoding; 1019 values
///   40..=63 name SBAS satellites (`rtcm::ephemeris`).
/// - RTCM MSM satellite and signal masks, ids `1..=64` and `1..=32`, checked on
///   encoding (`rtcm::msm`); an SBAS MSM number `n` is PRN `119 + n`
///   (`positioning`).
/// - RTCM SSR raw satellite fields: 5 bits for GLONASS, 6 bits for the GPS,
///   Galileo and BeiDou families, 4 bits for native QZSS (`ssr`).
///
/// Neither `0` nor `>= 100` is a spellable two-digit token, so both are
/// rejected here.
pub(crate) const fn is_shared_token_prn(prn: u8) -> bool {
    prn >= 1 && prn <= 99
}

/// The leading constellation letter of a satellite or single/double-difference
/// ambiguity id token, as a borrowed slice (`"G01"` -> `"G"`, `"G01~ra1"` ->
/// `"G"`, `""` -> `""`).
///
/// This is the single canonical first-letter extractor used for per-system
/// grouping of stringly-keyed ids. Satellite tokens are ASCII (the RINEX/IGS
/// system letters `G/R/E/C/J/I/S`), so the leading byte is the constellation
/// letter. Modules that need owned keys call `.to_string()` on the result; this
/// replaces the per-module first-character parsing the `satellite_system`
/// helpers used to duplicate.
pub(crate) fn constellation_letter(id: &str) -> &str {
    id.get(..1).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_round_trips() {
        for sys in [
            GnssSystem::Gps,
            GnssSystem::Glonass,
            GnssSystem::Galileo,
            GnssSystem::BeiDou,
            GnssSystem::Qzss,
            GnssSystem::Navic,
            GnssSystem::Sbas,
        ] {
            assert_eq!(GnssSystem::from_letter(sys.letter()), Some(sys));
        }
        assert_eq!(GnssSystem::from_letter('X'), None);
    }

    #[test]
    fn system_labels_are_canonical() {
        let cases = [
            (GnssSystem::Gps, "GPS"),
            (GnssSystem::Glonass, "GLONASS"),
            (GnssSystem::Galileo, "Galileo"),
            (GnssSystem::BeiDou, "BeiDou"),
            (GnssSystem::Qzss, "QZSS"),
            (GnssSystem::Navic, "NavIC"),
            (GnssSystem::Sbas, "SBAS"),
        ];
        for (system, label) in cases {
            assert_eq!(system.as_str(), label);
            assert_eq!(system.to_string(), label);
        }
    }

    #[test]
    fn satellite_token_formats_padded() {
        let id = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        assert_eq!(id.to_string(), "G01");
        assert_eq!(
            GnssSatelliteId::new(GnssSystem::BeiDou, 30)
                .expect("valid satellite id")
                .to_string(),
            "C30"
        );
    }

    const EVERY_SYSTEM: [GnssSystem; 7] = [
        GnssSystem::Gps,
        GnssSystem::Glonass,
        GnssSystem::Galileo,
        GnssSystem::BeiDou,
        GnssSystem::Qzss,
        GnssSystem::Navic,
        GnssSystem::Sbas,
    ];

    #[test]
    fn satellite_constructor_takes_the_shared_token_range() {
        let id = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        assert_eq!(id.system, GnssSystem::Gps);
        assert_eq!(id.prn, 1);

        // SP3-d gives the identifier as a system letter plus a 2-digit integer
        // 01..99, and says nothing about how many satellites a constellation
        // flies. Every system takes the whole range.
        for system in EVERY_SYSTEM {
            for prn in 1..=99u8 {
                let id = GnssSatelliteId::new(system, prn)
                    .unwrap_or_else(|e| panic!("{system:?} {prn}: {e}"));
                assert_eq!(id.system, system);
                assert_eq!(id.prn, prn);
            }
        }
    }

    #[test]
    fn satellite_constructor_rejects_unspellable_prns() {
        // 0 and anything from 100 up cannot be written as a two-digit token.
        for system in EVERY_SYSTEM {
            for prn in [0u8, 100, 101, 199, 200, 254, 255] {
                assert_eq!(
                    GnssSatelliteId::new(system, prn),
                    Err(SatelliteIdError::InvalidInput {
                        field: "prn",
                        reason: "outside the 1..=99 satellite-token range"
                    }),
                    "{system:?} {prn}"
                );
            }
        }
    }

    /// The shared range is token syntax, not a roster and not a wire-field
    /// width. This pins that it stays uniform across constellations: a
    /// narrower bound belongs to the boundary that owns it, not here.
    #[test]
    fn shared_token_range_is_the_same_for_every_constellation() {
        for prn in 0..=255u8 {
            let accepted: Vec<bool> = EVERY_SYSTEM
                .iter()
                .map(|system| GnssSatelliteId::new(*system, prn).is_ok())
                .collect();
            assert!(
                accepted.iter().all(|ok| *ok == accepted[0]),
                "prn {prn} is accepted for some constellations and not others"
            );
            assert_eq!(accepted[0], (1..=99).contains(&prn), "prn {prn}");
        }
    }

    /// Every accepted identifier writes a canonical padded token that parses
    /// back to itself.
    #[test]
    fn every_accepted_identifier_round_trips_through_display_and_from_str() {
        for system in EVERY_SYSTEM {
            for prn in 1..=99u8 {
                let id = GnssSatelliteId::new(system, prn).expect("valid satellite id");
                let token = id.to_string();
                assert_eq!(token.len(), 3, "{token}");
                assert_eq!(token.chars().next(), Some(system.letter()), "{token}");
                assert_eq!(&token[1..], format!("{prn:02}"), "{token}");
                assert_eq!(token.parse(), Ok(id), "{token}");
                // The unpadded and space-padded spellings parse to the same id.
                assert_eq!(
                    format!("{}{prn}", system.letter()).parse(),
                    Ok(id),
                    "unpadded {token}"
                );
                if prn < 10 {
                    assert_eq!(
                        format!("{} {prn}", system.letter()).parse(),
                        Ok(id),
                        "space-padded {token}"
                    );
                }
            }
        }
    }

    #[test]
    fn satellite_token_parses_via_from_str() {
        assert_eq!(
            "G01".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id"))
        );
        assert_eq!(
            "G 1".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id"))
        );
        assert_eq!(
            "G1".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id"))
        );
        assert_eq!(
            "G32".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Gps, 32).expect("valid satellite id"))
        );
        assert_eq!(
            "R27".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Glonass, 27).expect("valid satellite id"))
        );
        assert_eq!(
            "E36".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Galileo, 36).expect("valid satellite id"))
        );
        assert_eq!(
            "C30".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::BeiDou, 30).expect("valid satellite id"))
        );
        assert_eq!(
            "C63".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::BeiDou, 63).expect("valid satellite id"))
        );
        assert_eq!(
            "J09".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Qzss, 9).expect("valid satellite id"))
        );
        assert_eq!(
            "I14".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Navic, 14).expect("valid satellite id"))
        );
        assert_eq!(
            "S20".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Sbas, 20).expect("valid satellite id"))
        );
        assert_eq!(
            "S58".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Sbas, 58).expect("valid satellite id"))
        );
        // Tokens real products carry above the operational roster: extended
        // GLONASS slots, and the GPS number CODE's DCB tables use.
        assert_eq!(
            "R28".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Glonass, 28).expect("valid satellite id"))
        );
        assert_eq!(
            "G34".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Gps, 34).expect("valid satellite id"))
        );
        // Surrounding whitespace and a padded PRN both parse, matching the
        // SP3/RINEX field readers.
        assert_eq!(
            " E12 ".parse(),
            Ok(GnssSatelliteId::new(GnssSystem::Galileo, 12).expect("valid satellite id"))
        );
        // The Display round-trips through FromStr.
        let id = GnssSatelliteId::new(GnssSystem::Qzss, 7).expect("valid satellite id");
        assert_eq!(id.to_string().parse(), Ok(id));
        // Rejections: empty, unknown letter, missing PRN, non-numeric PRN.
        assert_eq!("".parse::<GnssSatelliteId>(), Err(ParseSatelliteIdError));
        assert_eq!("X01".parse::<GnssSatelliteId>(), Err(ParseSatelliteIdError));
        assert_eq!("G".parse::<GnssSatelliteId>(), Err(ParseSatelliteIdError));
        assert_eq!("GAB".parse::<GnssSatelliteId>(), Err(ParseSatelliteIdError));
        // The SP3-d `Lnn` Low-Earth Orbiter identifier names no GNSS
        // constellation and is not a satellite id here.
        assert_eq!("L09".parse::<GnssSatelliteId>(), Err(ParseSatelliteIdError));
        // Lowercase system letters are not the canonical spelling.
        assert_eq!("g01".parse::<GnssSatelliteId>(), Err(ParseSatelliteIdError));
    }

    #[test]
    fn satellite_token_rejects_bad_prn_width_and_range() {
        // `nn = 00` names no satellite, one to three digits cannot reach 100,
        // and a three-digit PRN field is not a satellite token at all.
        for token in [
            "G0", "G00", "G000", "G001", "G100", "G255", "R00", "E00", "C00", "J00", "I00", "S00",
            "S100", "G 0", "G 00", "G+1", "G-1", "G 1 1",
        ] {
            assert_eq!(
                token.parse::<GnssSatelliteId>(),
                Err(ParseSatelliteIdError),
                "{token}"
            );
        }
        // And every system letter rejects 00 while taking 01 and 99.
        for system in EVERY_SYSTEM {
            let letter = system.letter();
            assert_eq!(
                format!("{letter}00").parse::<GnssSatelliteId>(),
                Err(ParseSatelliteIdError),
                "{letter}00"
            );
            assert!(format!("{letter}01").parse::<GnssSatelliteId>().is_ok());
            assert!(format!("{letter}99").parse::<GnssSatelliteId>().is_ok());
        }
    }

    /// `system` and `prn` are public and the derived `Deserialize` writes them
    /// straight through, so an identifier that never passed the constructor can
    /// exist. There is no constructor-enforced invariant to rely on; each wire
    /// conversion has to re-check the value itself.
    #[test]
    fn public_fields_and_serde_can_bypass_the_constructor() {
        for prn in [0u8, 100, 255] {
            let bypass = GnssSatelliteId {
                system: GnssSystem::Gps,
                prn,
            };
            assert_eq!(bypass.prn, prn);
            assert!(GnssSatelliteId::new(GnssSystem::Gps, prn).is_err());

            let json = serde_json::to_string(&bypass).expect("serialize");
            let back: GnssSatelliteId = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, bypass, "deserialization applies no range check");
        }
        // Display still formats a bypassing value; it does not normalize it,
        // and the token it writes does not parse back.
        let hundred = GnssSatelliteId {
            system: GnssSystem::Gps,
            prn: 100,
        };
        assert_eq!(hundred.to_string(), "G100");
        assert_eq!(
            hundred.to_string().parse::<GnssSatelliteId>(),
            Err(ParseSatelliteIdError)
        );
    }

    #[test]
    fn constellation_letter_extracts_leading_token_byte() {
        assert_eq!(constellation_letter("G01"), "G");
        assert_eq!(constellation_letter("C30"), "C");
        assert_eq!(constellation_letter("E12~ra1"), "E");
        assert_eq!(constellation_letter("R07:base=R07,rover=R07"), "R");
        assert_eq!(constellation_letter(""), "");
    }
}
