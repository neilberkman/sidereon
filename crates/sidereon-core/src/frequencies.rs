//! Canonical GNSS carrier-frequency table.
//!
//! All GNSS carrier lookup paths route through this module so signal frequency
//! policy cannot drift between combinations, RINEX observation decoding, SPP,
//! and ionosphere models.

use crate::constants::{C_M_S, F_B1I_HZ, F_B3I_HZ, F_E1_HZ, F_E5A_HZ, F_L1_HZ, F_L2_HZ};
use crate::validate;
use crate::GnssSystem;

const F_E6_HZ: f64 = 1_278_750_000.0;
const F_E5B_HZ: f64 = 1_207_140_000.0;
const F_E5_HZ: f64 = 1_191_795_000.0;
const F_GLONASS_G1_BASE_HZ: f64 = 1_602_000_000.0;
const F_GLONASS_G1_STEP_HZ: f64 = 562_500.0;
const F_GLONASS_G2_BASE_HZ: f64 = 1_246_000_000.0;
const F_GLONASS_G2_STEP_HZ: f64 = 437_500.0;

/// GNSS carrier band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CarrierBand {
    /// GPS/QZSS L1.
    L1,
    /// GPS/QZSS L2.
    L2,
    /// GPS/QZSS L5.
    L5,
    /// Galileo E1.
    E1,
    /// Galileo E5a.
    E5a,
    /// Galileo E5b.
    E5b,
    /// Galileo E5 AltBOC.
    E5,
    /// Galileo E6.
    E6,
    /// BeiDou B1C.
    B1c,
    /// BeiDou B1I.
    B1i,
    /// BeiDou B2a.
    B2a,
    /// BeiDou B2b.
    B2b,
    /// BeiDou B2.
    B2,
    /// BeiDou B3I.
    B3i,
    /// GLONASS G1 FDMA.
    G1,
    /// GLONASS G2 FDMA.
    G2,
}

impl CarrierBand {
    /// Parse a lower-case carrier-band token.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "l1" => Some(Self::L1),
            "l2" => Some(Self::L2),
            "l5" => Some(Self::L5),
            "e1" => Some(Self::E1),
            "e5a" => Some(Self::E5a),
            "e5b" => Some(Self::E5b),
            "e5" => Some(Self::E5),
            "e6" => Some(Self::E6),
            "b1c" => Some(Self::B1c),
            "b1i" => Some(Self::B1i),
            "b2a" => Some(Self::B2a),
            "b2b" => Some(Self::B2b),
            "b2" => Some(Self::B2),
            "b3i" => Some(Self::B3i),
            "g1" => Some(Self::G1),
            "g2" => Some(Self::G2),
            _ => None,
        }
    }

    /// Parse only the carrier-band tokens supported by the ionosphere-free API.
    pub fn from_iono_free_name(name: &str) -> Option<Self> {
        match name {
            "l1" => Some(Self::L1),
            "l2" => Some(Self::L2),
            "e1" => Some(Self::E1),
            "e5a" => Some(Self::E5a),
            "b1i" => Some(Self::B1i),
            "b3i" => Some(Self::B3i),
            _ => None,
        }
    }

    /// The canonical lower-case band token.
    pub const fn name(self) -> &'static str {
        match self {
            Self::L1 => "l1",
            Self::L2 => "l2",
            Self::L5 => "l5",
            Self::E1 => "e1",
            Self::E5a => "e5a",
            Self::E5b => "e5b",
            Self::E5 => "e5",
            Self::E6 => "e6",
            Self::B1c => "b1c",
            Self::B1i => "b1i",
            Self::B2a => "b2a",
            Self::B2b => "b2b",
            Self::B2 => "b2",
            Self::B3i => "b3i",
            Self::G1 => "g1",
            Self::G2 => "g2",
        }
    }
}

/// A standard two-carrier ionosphere-free pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarrierPair {
    /// First carrier band in the affine combination.
    pub band1: CarrierBand,
    /// Second carrier band in the affine combination.
    pub band2: CarrierBand,
}

impl CarrierPair {
    /// Construct a pair from two carrier bands.
    pub const fn new(band1: CarrierBand, band2: CarrierBand) -> Self {
        Self { band1, band2 }
    }
}

/// One fixed-frequency carrier-table entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarrierFrequency {
    /// GNSS constellation.
    pub system: GnssSystem,
    /// Carrier band.
    pub band: CarrierBand,
    /// Carrier frequency in hertz.
    pub frequency_hz: f64,
}

/// Fixed-frequency carrier entries. GLONASS FDMA carriers are channel-derived
/// through [`rinex_band_frequency_hz`] and therefore do not appear here.
pub const fn fixed_carrier_frequencies() -> [CarrierFrequency; 17] {
    [
        CarrierFrequency {
            system: GnssSystem::Gps,
            band: CarrierBand::L1,
            frequency_hz: F_L1_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Gps,
            band: CarrierBand::L2,
            frequency_hz: F_L2_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Gps,
            band: CarrierBand::L5,
            frequency_hz: F_E5A_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Qzss,
            band: CarrierBand::L1,
            frequency_hz: F_L1_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Qzss,
            band: CarrierBand::L2,
            frequency_hz: F_L2_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Qzss,
            band: CarrierBand::L5,
            frequency_hz: F_E5A_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E1,
            frequency_hz: F_E1_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E5a,
            frequency_hz: F_E5A_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E6,
            frequency_hz: F_E6_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E5b,
            frequency_hz: F_E5B_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E5,
            frequency_hz: F_E5_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B1c,
            frequency_hz: F_L1_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B1i,
            frequency_hz: F_B1I_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B2a,
            frequency_hz: F_E5A_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B3i,
            frequency_hz: F_B3I_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B2b,
            frequency_hz: F_E5B_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B2,
            frequency_hz: F_E5_HZ,
        },
    ]
}

/// Carrier entries used by the current ionosphere-free public API.
pub const fn iono_free_carrier_frequencies() -> [CarrierFrequency; 6] {
    [
        CarrierFrequency {
            system: GnssSystem::Gps,
            band: CarrierBand::L1,
            frequency_hz: F_L1_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Gps,
            band: CarrierBand::L2,
            frequency_hz: F_L2_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E1,
            frequency_hz: F_E1_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::Galileo,
            band: CarrierBand::E5a,
            frequency_hz: F_E5A_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B1i,
            frequency_hz: F_B1I_HZ,
        },
        CarrierFrequency {
            system: GnssSystem::BeiDou,
            band: CarrierBand::B3i,
            frequency_hz: F_B3I_HZ,
        },
    ]
}

/// Carrier frequency in hertz for a constellation and canonical carrier band.
pub const fn frequency_hz(system: GnssSystem, band: CarrierBand) -> Option<f64> {
    match (system, band) {
        (GnssSystem::Gps, CarrierBand::L1) => Some(F_L1_HZ),
        (GnssSystem::Gps, CarrierBand::L2) => Some(F_L2_HZ),
        (GnssSystem::Gps, CarrierBand::L5) => Some(F_E5A_HZ),
        (GnssSystem::Qzss, CarrierBand::L1) => Some(F_L1_HZ),
        (GnssSystem::Qzss, CarrierBand::L2) => Some(F_L2_HZ),
        (GnssSystem::Qzss, CarrierBand::L5) => Some(F_E5A_HZ),
        (GnssSystem::Galileo, CarrierBand::E1) => Some(F_E1_HZ),
        (GnssSystem::Galileo, CarrierBand::E5a) => Some(F_E5A_HZ),
        (GnssSystem::Galileo, CarrierBand::E6) => Some(F_E6_HZ),
        (GnssSystem::Galileo, CarrierBand::E5b) => Some(F_E5B_HZ),
        (GnssSystem::Galileo, CarrierBand::E5) => Some(F_E5_HZ),
        (GnssSystem::BeiDou, CarrierBand::B1c) => Some(F_L1_HZ),
        (GnssSystem::BeiDou, CarrierBand::B1i) => Some(F_B1I_HZ),
        (GnssSystem::BeiDou, CarrierBand::B2a) => Some(F_E5A_HZ),
        (GnssSystem::BeiDou, CarrierBand::B3i) => Some(F_B3I_HZ),
        (GnssSystem::BeiDou, CarrierBand::B2b) => Some(F_E5B_HZ),
        (GnssSystem::BeiDou, CarrierBand::B2) => Some(F_E5_HZ),
        _ => None,
    }
}

/// Carrier wavelength in meters for a constellation and canonical carrier band.
pub fn wavelength_m(system: GnssSystem, band: CarrierBand) -> Option<f64> {
    frequency_hz(system, band).and_then(wavelength_for_frequency)
}

/// RINEX observation band frequency in hertz for a system and band digit.
///
/// GLONASS G1/G2 carriers require the FDMA channel number from the observation
/// file's `GLONASS SLOT / FRQ #` records.
pub fn rinex_band_frequency_hz(
    system: GnssSystem,
    band: char,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    rinex_signal_frequency_hz(system, band, None, None, glonass_channel)
}

/// RINEX observation-code frequency in hertz for a system and full code.
///
/// BeiDou's band labels changed across RINEX 3 minor versions: in RINEX 3.02
/// `C1I`/`L1I` are B1I (1561.098 MHz), while in RINEX 3.03 and later band 1 is
/// B1C (1575.42 MHz). Use this helper when the observation code and file
/// version are available instead of reducing the code to a band digit first.
pub fn rinex_observation_frequency_hz(
    system: GnssSystem,
    code: &str,
    rinex_version: f64,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    let mut chars = code.chars();
    let _kind = chars.next()?;
    let band = chars.next()?;
    let tracking = chars.next();
    rinex_signal_frequency_hz(system, band, tracking, Some(rinex_version), glonass_channel)
}

fn rinex_signal_frequency_hz(
    system: GnssSystem,
    band: char,
    tracking: Option<char>,
    rinex_version: Option<f64>,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    let frequency_hz = match (system, band, glonass_channel) {
        (GnssSystem::Gps, '1', _) => frequency_hz(system, CarrierBand::L1),
        (GnssSystem::Gps, '2', _) => frequency_hz(system, CarrierBand::L2),
        (GnssSystem::Gps, '5', _) => frequency_hz(system, CarrierBand::L5),
        (GnssSystem::Qzss, '1', _) => frequency_hz(system, CarrierBand::L1),
        (GnssSystem::Qzss, '2', _) => frequency_hz(system, CarrierBand::L2),
        (GnssSystem::Qzss, '5', _) => frequency_hz(system, CarrierBand::L5),
        (GnssSystem::Galileo, '1', _) => frequency_hz(system, CarrierBand::E1),
        (GnssSystem::Galileo, '5', _) => frequency_hz(system, CarrierBand::E5a),
        (GnssSystem::Galileo, '6', _) => frequency_hz(system, CarrierBand::E6),
        (GnssSystem::Galileo, '7', _) => frequency_hz(system, CarrierBand::E5b),
        (GnssSystem::Galileo, '8', _) => frequency_hz(system, CarrierBand::E5),
        (GnssSystem::BeiDou, _, _) => {
            frequency_hz(system, rinex_beidou_band(band, tracking, rinex_version)?)
        }
        (GnssSystem::Glonass, '1', Some(channel)) => {
            Some(F_GLONASS_G1_BASE_HZ + f64::from(channel) * F_GLONASS_G1_STEP_HZ)
        }
        (GnssSystem::Glonass, '2', Some(channel)) => {
            Some(F_GLONASS_G2_BASE_HZ + f64::from(channel) * F_GLONASS_G2_STEP_HZ)
        }
        _ => None,
    }?;
    valid_frequency_hz(frequency_hz)
}

fn rinex_beidou_band(
    band: char,
    tracking: Option<char>,
    rinex_version: Option<f64>,
) -> Option<CarrierBand> {
    match band {
        '1' if tracking == Some('I') && rinex_version.is_some_and(is_rinex_302) => {
            Some(CarrierBand::B1i)
        }
        '1' => Some(CarrierBand::B1c),
        '2' => Some(CarrierBand::B1i),
        '5' => Some(CarrierBand::B2a),
        '6' => Some(CarrierBand::B3i),
        '7' => Some(CarrierBand::B2b),
        '8' => Some(CarrierBand::B2),
        _ => None,
    }
}

fn is_rinex_302(version: f64) -> bool {
    (3.015..3.025).contains(&version)
}

/// RINEX observation band wavelength in meters for a system and band digit.
pub fn rinex_band_wavelength_m(
    system: GnssSystem,
    band: char,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    rinex_band_frequency_hz(system, band, glonass_channel).and_then(wavelength_for_frequency)
}

/// RINEX observation-code wavelength in meters for a system and full code.
pub fn rinex_observation_wavelength_m(
    system: GnssSystem,
    code: &str,
    rinex_version: f64,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    rinex_observation_frequency_hz(system, code, rinex_version, glonass_channel)
        .and_then(wavelength_for_frequency)
}

fn valid_frequency_hz(frequency_hz: f64) -> Option<f64> {
    validate::finite_positive(frequency_hz, "frequency_hz").ok()
}

pub(crate) fn wavelength_for_frequency(frequency_hz: f64) -> Option<f64> {
    valid_frequency_hz(frequency_hz).map(|frequency_hz| C_M_S / frequency_hz)
}

/// Standard dual-frequency ionosphere-free carrier pair for a constellation.
pub const fn default_iono_free_pair(system: GnssSystem) -> Option<CarrierPair> {
    match system {
        GnssSystem::Gps => Some(CarrierPair::new(CarrierBand::L1, CarrierBand::L2)),
        GnssSystem::Galileo => Some(CarrierPair::new(CarrierBand::E1, CarrierBand::E5a)),
        GnssSystem::BeiDou => Some(CarrierPair::new(CarrierBand::B1i, CarrierBand::B3i)),
        _ => None,
    }
}

/// GLONASS G1 FDMA carrier frequency in hertz for an FDMA channel number `k`.
///
/// GLONASS is frequency-division multiplexed, so it has no single fixed
/// single-frequency carrier and is therefore absent from
/// [`default_spp_frequency_hz`]. The SPP ionosphere-scaling policy resolves the
/// GLONASS carrier per satellite from its broadcast / observation FDMA channel
/// instead: `1602.0 MHz + k * 562.5 kHz`, the same table as
/// [`rinex_band_frequency_hz`] uses for G1. This is the carrier the broadcast
/// Klobuchar L1 delay is scaled to by `(f_L1 / f_k)^2` for a GLONASS satellite.
pub const fn glonass_g1_frequency_hz(channel: i8) -> f64 {
    F_GLONASS_G1_BASE_HZ + (channel as f64) * F_GLONASS_G1_STEP_HZ
}

/// Single-frequency carrier used by the SPP ionosphere-scaling policy.
pub const fn default_spp_carrier(system: GnssSystem) -> Option<CarrierBand> {
    match system {
        GnssSystem::Gps => Some(CarrierBand::L1),
        GnssSystem::Galileo => Some(CarrierBand::E1),
        GnssSystem::BeiDou => Some(CarrierBand::B1i),
        _ => None,
    }
}

/// Single-frequency carrier frequency used by the SPP ionosphere-scaling policy.
pub const fn default_spp_frequency_hz(system: GnssSystem) -> Option<f64> {
    match default_spp_carrier(system) {
        Some(band) => frequency_hz(system, band),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iono_free_table_matches_supported_pairs() {
        assert_eq!(
            default_iono_free_pair(GnssSystem::Gps),
            Some(CarrierPair::new(CarrierBand::L1, CarrierBand::L2))
        );
        assert_eq!(
            default_iono_free_pair(GnssSystem::Galileo),
            Some(CarrierPair::new(CarrierBand::E1, CarrierBand::E5a))
        );
        assert_eq!(
            default_iono_free_pair(GnssSystem::BeiDou),
            Some(CarrierPair::new(CarrierBand::B1i, CarrierBand::B3i))
        );
        assert_eq!(default_iono_free_pair(GnssSystem::Glonass), None);
        assert_eq!(iono_free_carrier_frequencies().len(), 6);
    }

    #[test]
    fn rinex_band_table_matches_existing_frequency_bits() {
        let cases: [(GnssSystem, char, Option<i8>, f64); 16] = [
            (GnssSystem::Gps, '1', None, 1_575_420_000.0),
            (GnssSystem::Gps, '2', None, 1_227_600_000.0),
            (GnssSystem::Gps, '5', None, 1_176_450_000.0),
            (GnssSystem::Qzss, '1', None, 1_575_420_000.0),
            (GnssSystem::Qzss, '2', None, 1_227_600_000.0),
            (GnssSystem::Qzss, '5', None, 1_176_450_000.0),
            (GnssSystem::Galileo, '6', None, 1_278_750_000.0),
            (GnssSystem::Galileo, '7', None, 1_207_140_000.0),
            (GnssSystem::Galileo, '8', None, 1_191_795_000.0),
            (GnssSystem::BeiDou, '1', None, 1_575_420_000.0),
            (GnssSystem::BeiDou, '2', None, 1_561_098_000.0),
            (GnssSystem::BeiDou, '6', None, 1_268_520_000.0),
            (GnssSystem::BeiDou, '7', None, 1_207_140_000.0),
            (GnssSystem::BeiDou, '8', None, 1_191_795_000.0),
            (GnssSystem::Glonass, '1', Some(1), 1_602_562_500.0),
            (GnssSystem::Glonass, '2', Some(1), 1_246_437_500.0),
        ];
        for (system, band, channel, expected) in cases {
            assert_eq!(
                rinex_band_frequency_hz(system, band, channel).map(f64::to_bits),
                Some(expected.to_bits())
            );
        }
        assert_eq!(
            rinex_band_frequency_hz(GnssSystem::Glonass, '1', None),
            None
        );
    }

    #[test]
    fn rinex_observation_code_resolves_beidou_302_b1i() {
        for code in ["C1I", "L1I"] {
            assert_eq!(
                rinex_observation_frequency_hz(GnssSystem::BeiDou, code, 3.02, None)
                    .map(f64::to_bits),
                Some(F_B1I_HZ.to_bits())
            );
        }
        assert_eq!(
            rinex_observation_frequency_hz(GnssSystem::BeiDou, "L1X", 3.03, None).map(f64::to_bits),
            Some(F_L1_HZ.to_bits())
        );
    }

    #[test]
    fn wavelength_helpers_use_c_over_frequency() {
        let wavelength = wavelength_m(GnssSystem::Gps, CarrierBand::L1).unwrap();
        assert_eq!(wavelength.to_bits(), (C_M_S / F_L1_HZ).to_bits());

        let glonass_wavelength = rinex_band_wavelength_m(GnssSystem::Glonass, '1', Some(-7))
            .expect("GLONASS G1 channel");
        let frequency = 1_602_000_000.0 + f64::from(-7) * 562_500.0;
        assert_eq!(glonass_wavelength.to_bits(), (C_M_S / frequency).to_bits());
    }

    #[test]
    fn glonass_g1_spp_carrier_matches_rinex_band_table() {
        for channel in [-7_i8, -1, 0, 1, 6] {
            assert_eq!(
                glonass_g1_frequency_hz(channel).to_bits(),
                rinex_band_frequency_hz(GnssSystem::Glonass, '1', Some(channel))
                    .expect("GLONASS G1 channel frequency")
                    .to_bits()
            );
        }
    }

    #[test]
    fn frequency_validation_rejects_invalid_runtime_values() {
        assert_eq!(valid_frequency_hz(f64::NAN), None);
        assert_eq!(valid_frequency_hz(f64::INFINITY), None);
        assert_eq!(valid_frequency_hz(0.0), None);
        assert_eq!(valid_frequency_hz(-1.0), None);
        assert_eq!(wavelength_for_frequency(f64::NAN), None);
    }
}
