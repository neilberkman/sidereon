//! GNSS observable linear combinations.
//!
//! The ionosphere-free code and carrier-phase combinations are pure
//! frequency-domain algebra. The operation order is intentionally simple and
//! pinned to the original Sidereon/SciPy oracle recipe: square each carrier first,
//! form `gamma = f1^2 / (f1^2 - f2^2)`, then evaluate
//! `gamma * obs1 - (gamma - 1) * obs2` with normal Rust arithmetic.

use std::collections::{BTreeMap, BTreeSet};

use crate::constants::C_M_S;
use crate::frequencies;
pub use crate::frequencies::{CarrierBand, CarrierFrequency, CarrierPair};
use crate::validate;
use crate::GnssSystem;

/// Error produced by ionosphere-free combination helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IonosphereFreeError {
    /// The constellation has no standard ionosphere-free carrier pair.
    UnknownSystem(char),
    /// The requested band is not known for the constellation.
    UnknownBand { system: char, band: String },
    /// Equal carrier frequencies make the denominator vanish.
    EqualFrequencies,
    /// Cycle-to-meter conversion requires positive carrier frequencies.
    InvalidFrequency,
    /// Observation values must be finite, and the combined result must remain finite.
    InvalidObservation,
}

impl core::fmt::Display for IonosphereFreeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownSystem(system) => {
                write!(f, "unknown ionosphere-free constellation {system}")
            }
            Self::UnknownBand { system, band } => {
                write!(f, "unknown carrier band {band} for constellation {system}")
            }
            Self::EqualFrequencies => write!(f, "equal carrier frequencies"),
            Self::InvalidFrequency => write!(f, "carrier frequencies must be positive"),
            Self::InvalidObservation => write!(f, "observations must be finite"),
        }
    }
}

impl std::error::Error for IonosphereFreeError {}

/// Reason a satellite was not included in a paired pseudorange result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PseudorangeDropReason {
    /// Present in band 2 only.
    MissingBand1,
    /// Present in band 1 only.
    MissingBand2,
    /// The satellite appeared more than once within at least one band.
    DuplicateObservation,
    /// The satellite's constellation or requested band pair is unsupported.
    UnknownSystem,
}

/// A satellite-tokened pseudorange observation in meters.
pub type PseudorangeObservation = (String, f64);

/// Ionosphere-free pseudoranges paired by satellite token.
pub type CombinedPseudoranges = Vec<PseudorangeObservation>;

/// Per-satellite reasons for dropped pseudorange combinations.
pub type DroppedPseudoranges = Vec<(String, PseudorangeDropReason)>;

/// Result returned by [`ionosphere_free_pseudoranges`].
pub type PseudorangeCombinationResult =
    Result<(CombinedPseudoranges, DroppedPseudoranges), IonosphereFreeError>;

/// The full carrier-frequency table for the supported standard pairs.
pub const fn carrier_frequencies() -> [CarrierFrequency; 6] {
    frequencies::iono_free_carrier_frequencies()
}

/// The standard carrier frequency in hertz for a constellation and band.
pub fn carrier_frequency_hz(system: GnssSystem, band: CarrierBand) -> Option<f64> {
    frequencies::frequency_hz(system, band)
}

/// Carrier frequency lookup by RINEX/IGS system letter and lower-case band name.
pub fn frequency_hz(system: char, band: &str) -> Result<f64, IonosphereFreeError> {
    let Some(system_id) = GnssSystem::from_letter(system) else {
        return Err(IonosphereFreeError::UnknownSystem(system));
    };
    let Some(carrier_band) = CarrierBand::from_iono_free_name(band) else {
        return Err(IonosphereFreeError::UnknownBand {
            system,
            band: band.to_owned(),
        });
    };
    carrier_frequency_hz(system_id, carrier_band).ok_or_else(|| IonosphereFreeError::UnknownBand {
        system,
        band: band.to_owned(),
    })
}

/// Standard dual-frequency ionosphere-free carrier pair for a constellation.
pub fn default_pair(system: char) -> Result<CarrierPair, IonosphereFreeError> {
    match GnssSystem::from_letter(system).and_then(frequencies::default_iono_free_pair) {
        Some(pair) => Ok(pair),
        None => Err(IonosphereFreeError::UnknownSystem(system)),
    }
}

/// Ionosphere-free coefficient `gamma = f1^2 / (f1^2 - f2^2)`.
pub fn gamma(f1_hz: f64, f2_hz: f64) -> Result<f64, IonosphereFreeError> {
    let (f1_hz, f2_hz) = validate_distinct_frequencies(f1_hz, f2_hz)?;
    let f1sq = finite_frequency_product(f1_hz * f1_hz)?;
    let f2sq = finite_frequency_product(f2_hz * f2_hz)?;
    let denominator = finite_frequency_product(f1sq - f2sq)?;
    if denominator == 0.0 {
        return Err(IonosphereFreeError::EqualFrequencies);
    }
    let gamma = f1sq / denominator;
    validate::finite(gamma, "gamma").map_err(|_| IonosphereFreeError::InvalidFrequency)
}

/// Equal-variance noise amplification of the ionosphere-free combination.
pub fn noise_amplification(f1_hz: f64, f2_hz: f64) -> Result<f64, IonosphereFreeError> {
    let g = gamma(f1_hz, f2_hz)?;
    let amplification = (g * g + (g - 1.0) * (g - 1.0)).sqrt();
    validate::finite(amplification, "noise_amplification")
        .map_err(|_| IonosphereFreeError::InvalidFrequency)
}

/// Ionosphere-free code or meter-valued phase combination.
pub fn ionosphere_free(
    obs1_m: f64,
    obs2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, IonosphereFreeError> {
    let (f1_hz, f2_hz) = validate_distinct_frequencies(f1_hz, f2_hz)?;
    let obs1_m = validate_observation(obs1_m, "obs1_m")?;
    let obs2_m = validate_observation(obs2_m, "obs2_m")?;
    validate_observation(
        ionosphere_free_unchecked(obs1_m, obs2_m, f1_hz, f2_hz),
        "ionosphere_free_m",
    )
}

/// Ionosphere-free carrier-phase combination from meter-valued phase inputs.
pub fn ionosphere_free_phase_m(
    phase1_m: f64,
    phase2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, IonosphereFreeError> {
    ionosphere_free(phase1_m, phase2_m, f1_hz, f2_hz)
}

/// Ionosphere-free carrier-phase combination from cycle-valued phase inputs.
pub fn ionosphere_free_phase_cycles(
    phi1_cycles: f64,
    phi2_cycles: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, IonosphereFreeError> {
    let (f1_hz, f2_hz) = validate_distinct_frequencies(f1_hz, f2_hz)?;
    let phi1_cycles = validate_observation(phi1_cycles, "phi1_cycles")?;
    let phi2_cycles = validate_observation(phi2_cycles, "phi2_cycles")?;
    let phase1_m = C_M_S / f1_hz * phi1_cycles;
    let phase2_m = C_M_S / f2_hz * phi2_cycles;
    let phase1_m = validate_observation(phase1_m, "phase1_m")?;
    let phase2_m = validate_observation(phase2_m, "phase2_m")?;
    validate_observation(
        ionosphere_free_unchecked(phase1_m, phase2_m, f1_hz, f2_hz),
        "ionosphere_free_phase_m",
    )
}

fn validate_distinct_frequencies(
    f1_hz: f64,
    f2_hz: f64,
) -> Result<(f64, f64), IonosphereFreeError> {
    let f1_hz = validate_frequency(f1_hz, "f1_hz")?;
    let f2_hz = validate_frequency(f2_hz, "f2_hz")?;
    if f1_hz == f2_hz {
        Err(IonosphereFreeError::EqualFrequencies)
    } else {
        Ok((f1_hz, f2_hz))
    }
}

fn validate_frequency(f_hz: f64, field: &'static str) -> Result<f64, IonosphereFreeError> {
    validate::finite_positive(f_hz, field).map_err(|_| IonosphereFreeError::InvalidFrequency)
}

fn finite_frequency_product(value: f64) -> Result<f64, IonosphereFreeError> {
    validate::finite(value, "frequency_product").map_err(|_| IonosphereFreeError::InvalidFrequency)
}

fn validate_observation(value: f64, field: &'static str) -> Result<f64, IonosphereFreeError> {
    validate::finite(value, field).map_err(|_| IonosphereFreeError::InvalidObservation)
}

fn gamma_unchecked(f1_hz: f64, f2_hz: f64) -> f64 {
    let f1sq = f1_hz * f1_hz;
    let f2sq = f2_hz * f2_hz;
    f1sq / (f1sq - f2sq)
}

fn ionosphere_free_unchecked(obs1_m: f64, obs2_m: f64, f1_hz: f64, f2_hz: f64) -> f64 {
    let g = gamma_unchecked(f1_hz, f2_hz);
    g * obs1_m - (g - 1.0) * obs2_m
}

/// Combine two satellite-keyed pseudorange bands into ionosphere-free ranges.
///
/// `overrides` is a list of `(system_letter, band1_name, band2_name)` entries.
/// An invalid override band is treated the same way the original Sidereon wrapper
/// treated a failed combination for a paired satellite: that satellite is
/// reported as [`PseudorangeDropReason::UnknownSystem`].
pub fn ionosphere_free_pseudoranges(
    band1: &[PseudorangeObservation],
    band2: &[PseudorangeObservation],
    overrides: &[(char, String, String)],
) -> PseudorangeCombinationResult {
    let (m1, dups1) = map_and_duplicates(band1)?;
    let (m2, dups2) = map_and_duplicates(band2)?;

    let dups = dups1.union(&dups2).cloned().collect::<BTreeSet<_>>();
    let ids1 = m1.keys().cloned().collect::<BTreeSet<_>>();
    let ids2 = m2.keys().cloned().collect::<BTreeSet<_>>();

    let mut combined = Vec::new();
    let mut dropped = Vec::new();

    for sat in ids1.intersection(&ids2) {
        if dups.contains(sat) {
            continue;
        }
        match combine_satellite(sat, m1[sat], m2[sat], overrides) {
            Ok(range_m) => combined.push((sat.clone(), range_m)),
            Err(IonosphereFreeError::UnknownSystem(_))
            | Err(IonosphereFreeError::UnknownBand { .. }) => {
                dropped.push((sat.clone(), PseudorangeDropReason::UnknownSystem))
            }
            Err(error) => return Err(error),
        }
    }

    for sat in ids1.difference(&ids2) {
        if !dups.contains(sat) {
            dropped.push((sat.clone(), PseudorangeDropReason::MissingBand2));
        }
    }

    for sat in ids2.difference(&ids1) {
        if !dups.contains(sat) {
            dropped.push((sat.clone(), PseudorangeDropReason::MissingBand1));
        }
    }

    for sat in dups {
        dropped.push((sat, PseudorangeDropReason::DuplicateObservation));
    }

    dropped.sort();
    Ok((combined, dropped))
}

fn map_and_duplicates(
    observations: &[(String, f64)],
) -> Result<(BTreeMap<String, f64>, BTreeSet<String>), IonosphereFreeError> {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut map = BTreeMap::<String, f64>::new();
    for (sat, range_m) in observations {
        let range_m = validate_observation(*range_m, "pseudorange_m")?;
        *counts.entry(sat.clone()).or_insert(0) += 1;
        map.insert(sat.clone(), range_m);
    }
    let dups = counts
        .into_iter()
        .filter_map(|(sat, count)| (count > 1).then_some(sat))
        .collect();
    Ok((map, dups))
}

fn combine_satellite(
    sat: &str,
    pr1_m: f64,
    pr2_m: f64,
    overrides: &[(char, String, String)],
) -> Result<f64, IonosphereFreeError> {
    let system = sat
        .chars()
        .next()
        .ok_or(IonosphereFreeError::UnknownSystem('\0'))?;
    let pair = pair_for(system, overrides)?;
    let f1 = frequency_hz(system, pair.band1.name())?;
    let f2 = frequency_hz(system, pair.band2.name())?;
    ionosphere_free(pr1_m, pr2_m, f1, f2)
}

fn pair_for(
    system: char,
    overrides: &[(char, String, String)],
) -> Result<CarrierPair, IonosphereFreeError> {
    if let Some((_system, band1, band2)) = overrides.iter().find(|(s, _, _)| *s == system) {
        let Some(band1) = CarrierBand::from_iono_free_name(band1) else {
            return Err(IonosphereFreeError::UnknownBand {
                system,
                band: band1.clone(),
            });
        };
        let Some(band2) = CarrierBand::from_iono_free_name(band2) else {
            return Err(IonosphereFreeError::UnknownBand {
                system,
                band: band2.clone(),
            });
        };
        Ok(CarrierPair::new(band1, band2))
    } else {
        default_pair(system)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{F_B1I_HZ, F_B3I_HZ, F_E1_HZ, F_E5A_HZ, F_L1_HZ, F_L2_HZ};

    struct OracleCase {
        f1_hz: u64,
        f2_hz: u64,
        pr1_m: u64,
        pr2_m: u64,
        gamma: u64,
        noise: u64,
        iono_free_m: u64,
        phi1_cycles: u64,
        phi2_cycles: u64,
        phase1_m: u64,
        phase2_m: u64,
        phase_if_m: u64,
        phase_if_cycles_m: u64,
    }

    fn f(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    #[test]
    fn frequency_table_matches_standard_carriers() {
        assert_eq!(frequency_hz('G', "l1"), Ok(F_L1_HZ));
        assert_eq!(frequency_hz('G', "l2"), Ok(F_L2_HZ));
        assert_eq!(frequency_hz('E', "e1"), Ok(F_E1_HZ));
        assert_eq!(frequency_hz('E', "e5a"), Ok(F_E5A_HZ));
        assert_eq!(frequency_hz('C', "b1i"), Ok(F_B1I_HZ));
        assert_eq!(frequency_hz('C', "b3i"), Ok(F_B3I_HZ));
        assert_eq!(carrier_frequencies().len(), 6);
    }

    #[test]
    fn frequency_lookup_classifies_system_and_band_errors() {
        assert_eq!(
            frequency_hz('X', "l1"),
            Err(IonosphereFreeError::UnknownSystem('X'))
        );
        assert_eq!(
            frequency_hz('G', "bad"),
            Err(IonosphereFreeError::UnknownBand {
                system: 'G',
                band: "bad".to_owned(),
            })
        );
        assert_eq!(frequency_hz('G', "l1"), Ok(F_L1_HZ));
    }

    #[test]
    fn default_pairs_match_supported_systems() {
        assert_eq!(
            default_pair('G'),
            Ok(CarrierPair::new(CarrierBand::L1, CarrierBand::L2))
        );
        assert_eq!(
            default_pair('E'),
            Ok(CarrierPair::new(CarrierBand::E1, CarrierBand::E5a))
        );
        assert_eq!(
            default_pair('C'),
            Ok(CarrierPair::new(CarrierBand::B1i, CarrierBand::B3i))
        );
        assert_eq!(
            default_pair('X'),
            Err(IonosphereFreeError::UnknownSystem('X'))
        );
    }

    #[test]
    fn scipy_oracle_cases_are_zero_ulp() {
        let cases = [
            OracleCase {
                f1_hz: 0x41d779c018000000,
                f2_hz: 0x41d24aec20000000,
                pr1_m: 0x4175ef3c40772a36,
                pr2_m: 0x4175ef3c6a2bcbb5,
                gamma: 0x40045da686c28e3c,
                noise: 0x4007d3777c503ebc,
                iono_free_m: 0x4175ef3c00000000,
                phi1_cycles: 0x419cd8990a6a993b,
                phi2_cycles: 0x419682ad3bea73b9,
                phase1_m: 0x4175f4f80ddd7ecd,
                phase2_m: 0x4175fd37d057d184,
                phase_if_m: 0x4175e837d93b3cba,
                phase_if_cycles_m: 0x4175e837d93b3cba,
            },
            OracleCase {
                f1_hz: 0x41d779c018000000,
                f2_hz: 0x41d187ccf4000000,
                pr1_m: 0x41775d7280ee546c,
                pr2_m: 0x41775d72e7354588,
                gamma: 0x400215b7b8bf1d8d,
                noise: 0x4004b4e6a9e28198,
                iono_free_m: 0x41775d71ffffffff,
                phi1_cycles: 0x419eb9b5c28924dd,
                phi2_cycles: 0x4196fa6edb5f553e,
                phase1_m: 0x4177632debd8c260,
                phase2_m: 0x41776c09259441ae,
                phase_if_m: 0x41775803d8d388ae,
                phase_if_cycles_m: 0x41775803d8d388ae,
            },
            OracleCase {
                f1_hz: 0x41d7431dc4000000,
                f2_hz: 0x41d2e70510000000,
                pr1_m: 0x41753821627b0be3,
                pr2_m: 0x417538219525d0a5,
                gamma: 0x40078ca90724ddf1,
                noise: 0x400c384adb005afd,
                iono_free_m: 0x4175382100000000,
                phi1_cycles: 0x419ba72a23131776,
                phi2_cycles: 0x419680982c673023,
                phase1_m: 0x41753deaa1cd1743,
                phase2_m: 0x417545a9733bf939,
                phase_if_m: 0x41752edcaa262834,
                phase_if_cycles_m: 0x41752edcaa262834,
            },
        ];

        for case in cases {
            let f1 = f(case.f1_hz);
            let f2 = f(case.f2_hz);
            assert_eq!(gamma(f1, f2).unwrap().to_bits(), case.gamma);
            assert_eq!(noise_amplification(f1, f2).unwrap().to_bits(), case.noise);
            assert_eq!(
                ionosphere_free(f(case.pr1_m), f(case.pr2_m), f1, f2)
                    .unwrap()
                    .to_bits(),
                case.iono_free_m
            );
            assert_eq!(
                ionosphere_free_phase_m(f(case.phase1_m), f(case.phase2_m), f1, f2)
                    .unwrap()
                    .to_bits(),
                case.phase_if_m
            );
            assert_eq!(
                ionosphere_free_phase_cycles(f(case.phi1_cycles), f(case.phi2_cycles), f1, f2)
                    .unwrap()
                    .to_bits(),
                case.phase_if_cycles_m
            );
        }
    }

    #[test]
    fn invalid_frequency_modes_are_tagged() {
        assert_eq!(
            gamma(F_L1_HZ, F_L1_HZ),
            Err(IonosphereFreeError::EqualFrequencies)
        );
        assert_eq!(
            gamma(f64::NAN, F_L2_HZ),
            Err(IonosphereFreeError::InvalidFrequency)
        );
        assert_eq!(
            gamma(f64::MAX, F_L2_HZ),
            Err(IonosphereFreeError::InvalidFrequency)
        );
        assert_eq!(
            noise_amplification(F_L1_HZ, f64::INFINITY),
            Err(IonosphereFreeError::InvalidFrequency)
        );
        assert_eq!(
            ionosphere_free(1.0, 2.0, F_L1_HZ, 0.0),
            Err(IonosphereFreeError::InvalidFrequency)
        );
        assert_eq!(
            ionosphere_free_phase_cycles(1.0, 2.0, F_L1_HZ, F_L1_HZ),
            Err(IonosphereFreeError::EqualFrequencies)
        );
        assert_eq!(
            ionosphere_free_phase_cycles(1.0, 2.0, 0.0, F_L2_HZ),
            Err(IonosphereFreeError::InvalidFrequency)
        );
        assert_eq!(
            ionosphere_free_phase_cycles(1.0, 2.0, -F_L1_HZ, F_L2_HZ),
            Err(IonosphereFreeError::InvalidFrequency)
        );
    }

    #[test]
    fn invalid_observations_are_rejected() {
        assert_eq!(
            ionosphere_free(f64::NAN, 2.0, F_L1_HZ, F_L2_HZ),
            Err(IonosphereFreeError::InvalidObservation)
        );
        assert_eq!(
            ionosphere_free_phase_m(1.0, f64::INFINITY, F_L1_HZ, F_L2_HZ),
            Err(IonosphereFreeError::InvalidObservation)
        );
        assert_eq!(
            ionosphere_free_phase_cycles(f64::NAN, 2.0, F_L1_HZ, F_L2_HZ),
            Err(IonosphereFreeError::InvalidObservation)
        );
        assert_eq!(
            ionosphere_free(f64::MAX, -f64::MAX, F_L1_HZ, F_L2_HZ),
            Err(IonosphereFreeError::InvalidObservation)
        );
    }

    #[test]
    fn pseudorange_pairing_reports_missing_unknown_and_duplicate_satellites() {
        let band1 = vec![
            ("G01".to_string(), 23_000_000.0),
            ("G01".to_string(), 23_000_010.0),
            ("G02".to_string(), 22_000_000.0),
            ("G03".to_string(), 21_000_000.0),
            ("X01".to_string(), 20_000_000.0),
        ];
        let band2 = vec![
            ("G01".to_string(), 23_000_000.0),
            ("G02".to_string(), 22_000_000.0),
            ("G04".to_string(), 24_000_000.0),
            ("X01".to_string(), 20_000_000.0),
        ];

        let (combined, dropped) =
            ionosphere_free_pseudoranges(&band1, &band2, &[]).expect("valid pseudoranges");

        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0].0, "G02");
        assert_eq!(
            dropped,
            vec![
                (
                    "G01".to_string(),
                    PseudorangeDropReason::DuplicateObservation
                ),
                ("G03".to_string(), PseudorangeDropReason::MissingBand2),
                ("G04".to_string(), PseudorangeDropReason::MissingBand1),
                ("X01".to_string(), PseudorangeDropReason::UnknownSystem),
            ]
        );
    }

    #[test]
    fn pseudorange_pairing_rejects_non_finite_values() {
        let band1 = vec![("G01".to_string(), f64::NAN)];
        let band2 = vec![("G01".to_string(), 23_000_000.0)];

        assert_eq!(
            ionosphere_free_pseudoranges(&band1, &band2, &[]),
            Err(IonosphereFreeError::InvalidObservation)
        );
    }
}
