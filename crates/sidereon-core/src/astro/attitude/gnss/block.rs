//! GNSS satellite block identifiers and their parsers.

use crate::GnssSystem;
use core::fmt;
use core::str::FromStr;

/// Manufacturer of a BeiDou-3 MEO satellite. The two follow different
/// published small-beta yaw laws (see [`super::GnssAttitudeLaw`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BeidouManufacturer {
    /// China Academy of Space Technology.
    Cast,
    /// Shanghai Engineering Center for Microsatellites.
    Secm,
}

/// Orbit class of a BeiDou satellite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BeidouOrbitClass {
    /// Geostationary orbit.
    Geo,
    /// Inclined geosynchronous orbit.
    Igso,
    /// Medium Earth orbit.
    Meo,
}

/// A GNSS satellite block with a published attitude law.
///
/// Parse it from an IGS/ANTEX block name with [`GnssBlock::from_igs_name`] (or
/// [`str::parse`]), or from the `block_type` strings of
/// [`crate::constellation`] with [`GnssBlock::from_constellation_block_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GnssBlock {
    /// GPS Block IIA (IGS `BLOCK IIA`).
    GpsIIA,
    /// GPS Block IIR-A and IIR-B (IGS `BLOCK IIR-A`, `BLOCK IIR-B`, and the
    /// earlier `BLOCK IIR`).
    GpsIIR,
    /// GPS Block IIR-M (IGS `BLOCK IIR-M`).
    GpsIIRM,
    /// GPS Block IIF (IGS `BLOCK IIF`).
    GpsIIF,
    /// GPS III (IGS `BLOCK IIIA`).
    GpsIII,
    /// GLONASS-M (IGS `GLONASS-M`).
    GlonassM,
    /// Galileo In-Orbit Validation (IGS `GALILEO-1`).
    GalileoIov,
    /// Galileo Full Operational Capability (IGS `GALILEO-2`).
    GalileoFoc,
    /// BeiDou-2 geostationary (IGS `BEIDOU-2G`, `BEIDOU-2G-CAST`).
    Beidou2Geo,
    /// BeiDou-2 inclined geosynchronous (IGS `BEIDOU-2I`).
    Beidou2Igso,
    /// BeiDou-2 medium Earth orbit (IGS `BEIDOU-2M`).
    Beidou2Meo,
    /// BeiDou-3 geostationary (IGS `BEIDOU-3G-CAST`).
    Beidou3Geo,
    /// BeiDou-3 inclined geosynchronous, built by CAST (IGS `BEIDOU-3I`,
    /// `BEIDOU-3I-CAST`).
    Beidou3Igso,
    /// BeiDou-3 medium Earth orbit (IGS `BEIDOU-3M-CAST`, `BEIDOU-3M-SECM`).
    /// `manufacturer` is `None` when the name does not state it (`BEIDOU-3M`).
    Beidou3Meo {
        /// The manufacturer, when known.
        manufacturer: Option<BeidouManufacturer>,
    },
}

/// Why a block name could not be turned into a [`GnssBlock`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum GnssBlockError {
    /// The name is not an IGS or constellation block name.
    #[error("unknown GNSS block name {name:?}")]
    Unknown {
        /// The name as given.
        name: String,
    },
    /// The name is a known block with no attitude law implemented here
    /// (for example `BLOCK I`, `GLONASS-M+`, `GLONASS-K1`, the BeiDou-3
    /// experimental satellites, QZSS and NavIC).
    #[error("GNSS block {name:?} has no attitude law here")]
    Unsupported {
        /// The name as given.
        name: String,
    },
    /// A BeiDou `block_type` (`BDS-2`, `BDS-3`) does not state the orbit
    /// class, which decides the attitude law; pass it explicitly.
    #[error("BeiDou block type {block_type:?} needs an orbit class")]
    BeidouOrbitClassRequired {
        /// The block type as given.
        block_type: String,
    },
}

/// IGS names of blocks known to IGS with no attitude law implemented here.
const UNSUPPORTED_IGS_NAMES: &[&str] = &[
    "BLOCK I",
    "BLOCK II",
    "GLONASS",
    "GLONASS-M+",
    "GLONASS-K1",
    "GLONASS-K2",
    "GALILEO-0A",
    "GALILEO-0B",
    "BEIDOU-3SI-SECM",
    "BEIDOU-3SI-CAST",
    "BEIDOU-3SM-CAST",
    "BEIDOU-3SM-SECM",
    "QZSS",
    "QZSS-2A",
    "QZSS-2I",
    "QZSS-2G",
    "QZSS-3G",
    "QZSS-3I",
    "IRNSS-1GEO",
    "IRNSS-1IGSO",
    "IRNSS-2GEO",
    "IRNSS-2IGSO",
];

impl GnssBlock {
    /// Parse an IGS/ANTEX satellite block name (the `TYPE / SERIAL NO` antenna
    /// type of a satellite antenna in ANTEX 1.4). Surrounding blanks are
    /// ignored and letters may be in either case; otherwise the name must match
    /// exactly.
    pub fn from_igs_name(name: &str) -> Result<Self, GnssBlockError> {
        let key = name.trim().to_ascii_uppercase();
        let block = match key.as_str() {
            "BLOCK IIA" => Self::GpsIIA,
            "BLOCK IIR" | "BLOCK IIR-A" | "BLOCK IIR-B" => Self::GpsIIR,
            "BLOCK IIR-M" => Self::GpsIIRM,
            "BLOCK IIF" => Self::GpsIIF,
            "BLOCK IIIA" => Self::GpsIII,
            "GLONASS-M" => Self::GlonassM,
            "GALILEO-1" => Self::GalileoIov,
            "GALILEO-2" => Self::GalileoFoc,
            "BEIDOU-2G" | "BEIDOU-2G-CAST" => Self::Beidou2Geo,
            "BEIDOU-2I" => Self::Beidou2Igso,
            "BEIDOU-2M" => Self::Beidou2Meo,
            "BEIDOU-3G" | "BEIDOU-3G-CAST" => Self::Beidou3Geo,
            "BEIDOU-3I" | "BEIDOU-3I-CAST" => Self::Beidou3Igso,
            "BEIDOU-3M" => Self::Beidou3Meo { manufacturer: None },
            "BEIDOU-3M-CAST" => Self::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            "BEIDOU-3M-SECM" => Self::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Secm),
            },
            _ if UNSUPPORTED_IGS_NAMES.contains(&key.as_str()) => {
                return Err(GnssBlockError::Unsupported {
                    name: name.to_string(),
                })
            }
            _ => {
                return Err(GnssBlockError::Unknown {
                    name: name.to_string(),
                })
            }
        };
        Ok(block)
    }

    /// Parse a `block_type` as [`crate::constellation`] records carry it:
    /// GPS `IIA`, `IIR`, `IIR-M`, `IIF`, `III` (also `IIIA`); Galileo `IOV`,
    /// `FOC`; BeiDou `BDS-2`, `BDS-3`. Blanks are ignored and letters may be in
    /// either case.
    ///
    /// BeiDou block types do not carry the orbit class, so `beidou_orbit` must
    /// give it; `None` is [`GnssBlockError::BeidouOrbitClassRequired`]. A
    /// `BDS-3` MEO parses with an unknown manufacturer. The BeiDou-3
    /// experimental satellites (`BDS-3S`) are [`GnssBlockError::Unsupported`].
    /// GLONASS records carry no block type.
    pub fn from_constellation_block_type(
        system: GnssSystem,
        block_type: &str,
        beidou_orbit: Option<BeidouOrbitClass>,
    ) -> Result<Self, GnssBlockError> {
        let key = block_type.trim().to_ascii_uppercase();
        let unknown = || GnssBlockError::Unknown {
            name: block_type.to_string(),
        };
        match system {
            GnssSystem::Gps => match key.as_str() {
                "IIA" => Ok(Self::GpsIIA),
                "IIR" => Ok(Self::GpsIIR),
                "IIR-M" => Ok(Self::GpsIIRM),
                "IIF" => Ok(Self::GpsIIF),
                "III" | "IIIA" => Ok(Self::GpsIII),
                _ => Err(unknown()),
            },
            GnssSystem::Galileo => match key.as_str() {
                "IOV" => Ok(Self::GalileoIov),
                "FOC" => Ok(Self::GalileoFoc),
                _ => Err(unknown()),
            },
            GnssSystem::BeiDou => {
                let generation = match key.as_str() {
                    "BDS-2" => 2,
                    "BDS-3" => 3,
                    "BDS-3S" => {
                        return Err(GnssBlockError::Unsupported {
                            name: block_type.to_string(),
                        })
                    }
                    _ => return Err(unknown()),
                };
                let orbit =
                    beidou_orbit.ok_or_else(|| GnssBlockError::BeidouOrbitClassRequired {
                        block_type: block_type.to_string(),
                    })?;
                Ok(match (generation, orbit) {
                    (2, BeidouOrbitClass::Geo) => Self::Beidou2Geo,
                    (2, BeidouOrbitClass::Igso) => Self::Beidou2Igso,
                    (2, BeidouOrbitClass::Meo) => Self::Beidou2Meo,
                    (_, BeidouOrbitClass::Geo) => Self::Beidou3Geo,
                    (_, BeidouOrbitClass::Igso) => Self::Beidou3Igso,
                    (_, BeidouOrbitClass::Meo) => Self::Beidou3Meo { manufacturer: None },
                })
            }
            _ => Err(unknown()),
        }
    }

    /// The constellation the block belongs to.
    #[must_use]
    pub fn system(self) -> GnssSystem {
        match self {
            Self::GpsIIA | Self::GpsIIR | Self::GpsIIRM | Self::GpsIIF | Self::GpsIII => {
                GnssSystem::Gps
            }
            Self::GlonassM => GnssSystem::Glonass,
            Self::GalileoIov | Self::GalileoFoc => GnssSystem::Galileo,
            Self::Beidou2Geo
            | Self::Beidou2Igso
            | Self::Beidou2Meo
            | Self::Beidou3Geo
            | Self::Beidou3Igso
            | Self::Beidou3Meo { .. } => GnssSystem::BeiDou,
        }
    }

    /// The IGS/ANTEX name of the block. GPS IIR returns `BLOCK IIR-B`, the
    /// later and larger of its two antenna groups; a BeiDou-3 MEO with an
    /// unknown manufacturer returns `BEIDOU-3M`.
    #[must_use]
    pub fn igs_name(self) -> &'static str {
        match self {
            Self::GpsIIA => "BLOCK IIA",
            Self::GpsIIR => "BLOCK IIR-B",
            Self::GpsIIRM => "BLOCK IIR-M",
            Self::GpsIIF => "BLOCK IIF",
            Self::GpsIII => "BLOCK IIIA",
            Self::GlonassM => "GLONASS-M",
            Self::GalileoIov => "GALILEO-1",
            Self::GalileoFoc => "GALILEO-2",
            Self::Beidou2Geo => "BEIDOU-2G",
            Self::Beidou2Igso => "BEIDOU-2I",
            Self::Beidou2Meo => "BEIDOU-2M",
            Self::Beidou3Geo => "BEIDOU-3G-CAST",
            Self::Beidou3Igso => "BEIDOU-3I",
            Self::Beidou3Meo { manufacturer: None } => "BEIDOU-3M",
            Self::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            } => "BEIDOU-3M-CAST",
            Self::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Secm),
            } => "BEIDOU-3M-SECM",
        }
    }
}

impl FromStr for GnssBlock {
    type Err = GnssBlockError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::from_igs_name(name)
    }
}

impl fmt::Display for GnssBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.igs_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn igs_names_round_trip() {
        for block in [
            GnssBlock::GpsIIA,
            GnssBlock::GpsIIR,
            GnssBlock::GpsIIRM,
            GnssBlock::GpsIIF,
            GnssBlock::GpsIII,
            GnssBlock::GlonassM,
            GnssBlock::GalileoIov,
            GnssBlock::GalileoFoc,
            GnssBlock::Beidou2Geo,
            GnssBlock::Beidou2Igso,
            GnssBlock::Beidou2Meo,
            GnssBlock::Beidou3Geo,
            GnssBlock::Beidou3Igso,
            GnssBlock::Beidou3Meo { manufacturer: None },
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Secm),
            },
        ] {
            assert_eq!(GnssBlock::from_igs_name(block.igs_name()), Ok(block));
        }
    }

    #[test]
    fn igs_names_parse_with_blanks_and_case() {
        assert_eq!(" block iir-a ".parse(), Ok(GnssBlock::GpsIIR));
        assert_eq!("BLOCK IIR-B".parse(), Ok(GnssBlock::GpsIIR));
        assert_eq!("BEIDOU-3G".parse(), Ok(GnssBlock::Beidou3Geo));
    }

    #[test]
    fn unknown_and_unsupported_names_are_errors() {
        assert_eq!(
            GnssBlock::from_igs_name("BLOCK IIX"),
            Err(GnssBlockError::Unknown {
                name: "BLOCK IIX".into()
            })
        );
        assert_eq!(
            GnssBlock::from_igs_name("GLONASS-K1"),
            Err(GnssBlockError::Unsupported {
                name: "GLONASS-K1".into()
            })
        );
        assert_eq!(
            GnssBlock::from_igs_name(""),
            Err(GnssBlockError::Unknown {
                name: String::new()
            })
        );
    }

    /// Names in the IGS satellite antenna table and in IGS ANTEX files.
    #[test]
    fn every_igs_satellite_block_name_parses_or_is_unsupported() {
        for (name, block) in [
            ("BLOCK IIR", GnssBlock::GpsIIR),
            ("BLOCK IIIA", GnssBlock::GpsIII),
            ("BEIDOU-2G-CAST", GnssBlock::Beidou2Geo),
            ("BEIDOU-3I-CAST", GnssBlock::Beidou3Igso),
        ] {
            assert_eq!(GnssBlock::from_igs_name(name), Ok(block), "{name}");
        }
        for name in ["GLONASS-M+", "QZSS-3G", "QZSS-3I", "IRNSS-2IGSO"] {
            assert_eq!(
                GnssBlock::from_igs_name(name),
                Err(GnssBlockError::Unsupported { name: name.into() }),
                "{name}"
            );
        }
    }

    #[test]
    fn constellation_block_types_parse() {
        let parse = GnssBlock::from_constellation_block_type;
        assert_eq!(
            parse(GnssSystem::Gps, "IIR-M", None),
            Ok(GnssBlock::GpsIIRM)
        );
        for name in ["iii", "III", " IIIA "] {
            assert_eq!(parse(GnssSystem::Gps, name, None), Ok(GnssBlock::GpsIII));
        }
        assert_eq!(parse(GnssSystem::Gps, "IIF", None), Ok(GnssBlock::GpsIIF));
        assert_eq!(parse(GnssSystem::Gps, "IIR", None), Ok(GnssBlock::GpsIIR));
        assert_eq!(
            parse(GnssSystem::Galileo, "IOV", None),
            Ok(GnssBlock::GalileoIov)
        );
        assert_eq!(
            parse(GnssSystem::Galileo, "FOC", None),
            Ok(GnssBlock::GalileoFoc)
        );
        assert_eq!(
            parse(GnssSystem::BeiDou, "BDS-2", Some(BeidouOrbitClass::Igso)),
            Ok(GnssBlock::Beidou2Igso)
        );
        assert_eq!(
            parse(GnssSystem::BeiDou, "BDS-3", Some(BeidouOrbitClass::Meo)),
            Ok(GnssBlock::Beidou3Meo { manufacturer: None })
        );
        assert_eq!(
            parse(GnssSystem::BeiDou, "BDS-3", None),
            Err(GnssBlockError::BeidouOrbitClassRequired {
                block_type: "BDS-3".into()
            })
        );
        assert_eq!(
            parse(GnssSystem::BeiDou, "BDS-3S", Some(BeidouOrbitClass::Meo)),
            Err(GnssBlockError::Unsupported {
                name: "BDS-3S".into()
            })
        );
        assert_eq!(
            parse(GnssSystem::Gps, "FOC", None),
            Err(GnssBlockError::Unknown { name: "FOC".into() })
        );
        assert_eq!(
            parse(GnssSystem::Glonass, "M", None),
            Err(GnssBlockError::Unknown { name: "M".into() })
        );
    }
}
