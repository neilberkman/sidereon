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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
        f.write_str(match self {
            GnssSystem::Gps => "GPS",
            GnssSystem::Glonass => "GLO",
            GnssSystem::Galileo => "GAL",
            GnssSystem::BeiDou => "BDS",
            GnssSystem::Qzss => "QZSS",
            GnssSystem::Navic => "NavIC",
            GnssSystem::Sbas => "SBAS",
        })
    }
}

/// A satellite identifier: a constellation plus its within-system PRN/slot.
///
/// This is the `GnssSatelliteId { system, prn }` foundational type from the
/// spec (line 112). The `prn` is the within-constellation satellite number as
/// it appears in the product (e.g. the `01` in the SP3/RINEX token `G01`); it
/// is only meaningful in combination with [`GnssSatelliteId::system`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GnssSatelliteId {
    /// The constellation this satellite belongs to.
    pub system: GnssSystem,
    /// The within-constellation PRN / slot number (e.g. `1` for `G01`).
    pub prn: u8,
}

/// Error returned when constructing a GNSS satellite identifier from invalid input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SatelliteIdError {
    /// The PRN is outside the documented range for its constellation.
    #[error("invalid GNSS satellite {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

const fn invalid_input(field: &'static str, reason: &'static str) -> SatelliteIdError {
    SatelliteIdError::InvalidInput { field, reason }
}

impl GnssSatelliteId {
    /// Construct an identifier from a constellation and PRN.
    pub const fn new(system: GnssSystem, prn: u8) -> Result<Self, SatelliteIdError> {
        if !is_valid_prn(system, prn) {
            return Err(invalid_input("prn", "out of range for constellation"));
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

    /// Parse a canonical SP3/RINEX satellite token (`G01`, `E12`, `C30`): a
    /// constellation letter followed by the within-system PRN. Whitespace around
    /// the token and around the PRN is ignored, matching the SP3/RINEX field
    /// readers. This is the single canonical satellite-token parser; the
    /// SP3/RINEX/DGNSS readers delegate to it.
    fn from_str(token: &str) -> Result<Self, Self::Err> {
        let token = token.trim();
        let first = token.chars().next().ok_or(ParseSatelliteIdError)?;
        let system = GnssSystem::from_letter(first).ok_or(ParseSatelliteIdError)?;
        let prn_token = token[first.len_utf8()..].trim();
        if prn_token.len() != 2 || !prn_token.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseSatelliteIdError);
        }
        let prn = prn_token.parse::<u8>().map_err(|_| ParseSatelliteIdError)?;
        if !is_valid_prn(system, prn) {
            return Err(ParseSatelliteIdError);
        }
        Self::new(system, prn).map_err(|_| ParseSatelliteIdError)
    }
}

pub(crate) const fn is_valid_prn(system: GnssSystem, prn: u8) -> bool {
    match system {
        GnssSystem::Gps => prn >= 1 && prn <= 32,
        GnssSystem::Glonass => prn >= 1 && prn <= 27,
        GnssSystem::Galileo => prn >= 1 && prn <= 36,
        GnssSystem::BeiDou => prn >= 1 && prn <= 63,
        GnssSystem::Qzss => prn >= 1 && prn <= 9,
        GnssSystem::Navic => prn >= 1 && prn <= 14,
        GnssSystem::Sbas => prn >= 20 && prn <= 58,
    }
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

    #[test]
    fn satellite_constructor_validates_prn_range() {
        let id = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        assert_eq!(id.system, GnssSystem::Gps);
        assert_eq!(id.prn, 1);

        assert_eq!(
            GnssSatelliteId::new(GnssSystem::Gps, 0),
            Err(SatelliteIdError::InvalidInput {
                field: "prn",
                reason: "out of range for constellation"
            })
        );
        assert_eq!(
            GnssSatelliteId::new(GnssSystem::Sbas, 19),
            Err(SatelliteIdError::InvalidInput {
                field: "prn",
                reason: "out of range for constellation"
            })
        );
    }

    #[test]
    fn satellite_token_parses_via_from_str() {
        assert_eq!(
            "G01".parse(),
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
    }

    #[test]
    fn satellite_token_rejects_bad_prn_width_and_range() {
        for token in [
            "G0", "G1", "G001", "G00", "G33", "G255", "R28", "E37", "C64", "J10", "I15", "S01",
            "S19", "S59",
        ] {
            assert_eq!(
                token.parse::<GnssSatelliteId>(),
                Err(ParseSatelliteIdError),
                "{token}"
            );
        }
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
