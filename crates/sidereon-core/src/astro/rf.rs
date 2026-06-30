//! RF link-budget primitives.
//!
//! Pure physics with no system-specific assumptions: free-space path loss,
//! EIRP, carrier-to-noise-density ratio, link margin, wavelength, and parabolic
//! dish gain. Callers combine these with geometry outputs (slant range,
//! elevation) to build a complete link budget for a specific system. The sidereon
//! Elixir binding is a thin marshaling layer; no formula lives there.

use crate::astro::constants::physics::SPEED_OF_LIGHT_M_S;
use crate::validate;
use std::f64::consts::PI;

/// Free-space path-loss reference constant for kilometre range and megahertz
/// frequency inputs (dB).
const FSPL_KM_MHZ_CONSTANT_DB: f64 = 32.45;
/// Decibel scaling for an amplitude (field) ratio: 20 dB per decade.
const DB_FIELD_DECADE: f64 = 20.0;
/// Decibel scaling for a power ratio: 10 dB per decade.
const DB_POWER_DECADE: f64 = 10.0;
/// dBm-to-dBW conversion offset (1 W = 30 dBm).
const DBM_TO_DBW_OFFSET_DB: f64 = 30.0;
/// Boltzmann constant as the positive dBW/Hz/K offset of the conventional
/// link-budget equation (-228.6 dBW/Hz/K).
const BOLTZMANN_K_DBW_HZ_K: f64 = 228.6;

/// Error returned by RF link-budget helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RfError {
    /// A public RF input was non-finite or outside its physical domain.
    #[error("invalid RF input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

/// Inputs to [`link_margin`], mirroring the self-documenting Elixir map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkBudget {
    /// Transmitter EIRP (dBW).
    pub eirp_dbw: f64,
    /// Free-space path loss (dB).
    pub fspl_db: f64,
    /// Receiver figure of merit G/T (dB/K).
    pub receiver_gt_dbk: f64,
    /// Sum of miscellaneous losses (dB).
    pub other_losses_db: f64,
    /// Minimum C/N0 for demodulation (dB-Hz).
    pub required_cn0_dbhz: f64,
}

/// Free-space path loss in dB for a slant range in km and frequency in MHz:
/// `FSPL = 32.45 + 20*log10(f_MHz) + 20*log10(d_km)`.
///
/// Operation order (frequency term before range term) is fixed to reproduce the
/// prior Elixir reference bit-for-bit.
pub fn fspl(distance_km: f64, frequency_mhz: f64) -> Result<f64, RfError> {
    let distance_km = rf_positive(distance_km, "distance_km")?;
    let frequency_mhz = rf_positive(frequency_mhz, "frequency_mhz")?;
    rf_finite_output(
        FSPL_KM_MHZ_CONSTANT_DB
            + DB_FIELD_DECADE * frequency_mhz.log10()
            + DB_FIELD_DECADE * distance_km.log10(),
        "fspl_db",
    )
}

/// Effective isotropic radiated power in dBW: `EIRP = P_tx(dBm) + G_tx(dBi) - 30`.
pub fn eirp(tx_power_dbm: f64, tx_antenna_gain_dbi: f64) -> Result<f64, RfError> {
    let tx_power_dbm = rf_finite(tx_power_dbm, "tx_power_dbm")?;
    let tx_antenna_gain_dbi = rf_finite(tx_antenna_gain_dbi, "tx_antenna_gain_dbi")?;
    rf_finite_output(
        tx_power_dbm + tx_antenna_gain_dbi - DBM_TO_DBW_OFFSET_DB,
        "eirp_dbw",
    )
}

/// Carrier-to-noise-density ratio (C/N0) in dB-Hz:
/// `C/N0 = EIRP + G/T - FSPL + 228.6 - other_losses`.
pub fn cn0(
    eirp_dbw: f64,
    fspl_db: f64,
    receiver_gt_dbk: f64,
    other_losses_db: f64,
) -> Result<f64, RfError> {
    let eirp_dbw = rf_finite(eirp_dbw, "eirp_dbw")?;
    let fspl_db = rf_finite(fspl_db, "fspl_db")?;
    let receiver_gt_dbk = rf_finite(receiver_gt_dbk, "receiver_gt_dbk")?;
    let other_losses_db = rf_finite(other_losses_db, "other_losses_db")?;
    rf_finite_output(
        eirp_dbw + receiver_gt_dbk - fspl_db + BOLTZMANN_K_DBW_HZ_K - other_losses_db,
        "cn0_dbhz",
    )
}

/// Link margin in dB: the achieved C/N0 minus the required C/N0. Positive means
/// the link closes.
pub fn link_margin(budget: &LinkBudget) -> Result<f64, RfError> {
    let cn0_dbhz = cn0(
        budget.eirp_dbw,
        budget.fspl_db,
        budget.receiver_gt_dbk,
        budget.other_losses_db,
    )?;
    let required_cn0_dbhz = rf_finite(budget.required_cn0_dbhz, "required_cn0_dbhz")?;
    rf_finite_output(cn0_dbhz - required_cn0_dbhz, "link_margin_db")
}

/// Wavelength in metres for a frequency in Hz.
pub fn wavelength(frequency_hz: f64) -> Result<f64, RfError> {
    let frequency_hz = rf_positive(frequency_hz, "frequency_hz")?;
    rf_finite_output(SPEED_OF_LIGHT_M_S / frequency_hz, "wavelength_m")
}

/// Parabolic-dish antenna gain in dBi: `G = 10*log10(eta * (pi*D/lambda)^2)`.
///
/// The squaring uses libm `pow` (`powf(2.0)`), matching the Erlang `**`
/// operator the prior Elixir reference used, for bit-for-bit parity.
pub fn dish_gain(diameter_m: f64, frequency_hz: f64, efficiency: f64) -> Result<f64, RfError> {
    let diameter_m = rf_positive(diameter_m, "diameter_m")?;
    let lambda = wavelength(frequency_hz)?;
    let efficiency = rf_unit_efficiency(efficiency)?;
    rf_finite_output(
        DB_POWER_DECADE * (efficiency * (PI * diameter_m / lambda).powf(2.0)).log10(),
        "dish_gain_dbi",
    )
}

/// Batch free-space path loss wrapper.
///
/// Each output element is produced by [`fspl`] with the corresponding distance
/// and shared frequency, so every element is bit-identical to the scalar helper.
pub fn fspl_batch(distances_km: &[f64], frequency_mhz: f64) -> Vec<Result<f64, RfError>> {
    distances_km
        .iter()
        .map(|&distance_km| fspl(distance_km, frequency_mhz))
        .collect()
}

/// Batch link-margin wrapper.
///
/// Each output element is produced by [`link_margin`] with the corresponding
/// budget, so every element is bit-identical to the scalar helper.
pub fn link_margin_batch(budgets: &[LinkBudget]) -> Vec<Result<f64, RfError>> {
    budgets.iter().map(link_margin).collect()
}

fn rf_finite(x: f64, field: &'static str) -> Result<f64, RfError> {
    validate::finite(x, field).map_err(map_rf_input)
}

fn rf_positive(x: f64, field: &'static str) -> Result<f64, RfError> {
    validate::finite_positive(x, field).map_err(map_rf_input)
}

fn rf_unit_efficiency(efficiency: f64) -> Result<f64, RfError> {
    let efficiency = rf_positive(efficiency, "efficiency")?;
    if efficiency <= 1.0 {
        Ok(efficiency)
    } else {
        Err(invalid_rf_input("efficiency", "out of range"))
    }
}

fn rf_finite_output(value: f64, field: &'static str) -> Result<f64, RfError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_rf_input(field, "out of range"))
    }
}

fn map_rf_input(error: validate::FieldError) -> RfError {
    invalid_rf_input(error.field(), error.reason())
}

fn invalid_rf_input(field: &'static str, reason: &'static str) -> RfError {
    RfError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen output bits captured from the prior Elixir `Sidereon.RF` reference
    // (the public doctest values), proving cross-language 0-ULP parity.

    #[test]
    fn fspl_matches_frozen_elixir_bits() {
        assert_eq!(
            fspl(1200.0, 1616.0).unwrap().to_bits(),
            158.20245204972383_f64.to_bits()
        );
    }

    #[test]
    fn eirp_matches_frozen_elixir_bits() {
        assert_eq!(eirp(27.0, 3.0).unwrap().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn cn0_matches_frozen_elixir_bits() {
        assert_eq!(
            cn0(0.0, 165.0, -12.0, 3.0).unwrap().to_bits(),
            48.599999999999994_f64.to_bits()
        );
    }

    #[test]
    fn link_margin_matches_frozen_elixir_bits() {
        let budget = LinkBudget {
            eirp_dbw: 0.0,
            fspl_db: 165.0,
            receiver_gt_dbk: -12.0,
            other_losses_db: 3.0,
            required_cn0_dbhz: 35.0,
        };
        assert_eq!(
            link_margin(&budget).unwrap().to_bits(),
            13.599999999999994_f64.to_bits()
        );
    }

    #[test]
    fn wavelength_matches_frozen_elixir_bits() {
        assert_eq!(
            wavelength(1616.0e6).unwrap().to_bits(),
            0.1855151349009901_f64.to_bits()
        );
    }

    #[test]
    fn dish_gain_matches_frozen_elixir_bits() {
        assert_eq!(
            dish_gain(1.0, 1616.0e6, 0.55).unwrap().to_bits(),
            21.97903741903791_f64.to_bits()
        );
    }

    #[test]
    fn rf_helpers_reject_invalid_physical_domains() {
        assert_invalid(
            fspl(f64::NAN, 1616.0).unwrap_err(),
            "distance_km",
            "not finite",
        );
        assert_invalid(
            fspl(0.0, 1616.0).unwrap_err(),
            "distance_km",
            "not positive",
        );
        assert_invalid(
            fspl(1200.0, -1.0).unwrap_err(),
            "frequency_mhz",
            "not positive",
        );
        assert_invalid(
            wavelength(f64::INFINITY).unwrap_err(),
            "frequency_hz",
            "not finite",
        );
        assert_invalid(wavelength(0.0).unwrap_err(), "frequency_hz", "not positive");
        assert_invalid(
            dish_gain(0.0, 1616.0e6, 0.55).unwrap_err(),
            "diameter_m",
            "not positive",
        );
        assert_invalid(
            dish_gain(1.0, 1616.0e6, 0.0).unwrap_err(),
            "efficiency",
            "not positive",
        );
        assert_invalid(
            dish_gain(1.0, 1616.0e6, 1.1).unwrap_err(),
            "efficiency",
            "out of range",
        );
        assert_invalid(
            eirp(f64::NAN, 3.0).unwrap_err(),
            "tx_power_dbm",
            "not finite",
        );
        assert_invalid(
            cn0(0.0, f64::INFINITY, -12.0, 3.0).unwrap_err(),
            "fspl_db",
            "not finite",
        );

        let budget = LinkBudget {
            eirp_dbw: 0.0,
            fspl_db: 165.0,
            receiver_gt_dbk: -12.0,
            other_losses_db: 3.0,
            required_cn0_dbhz: f64::NEG_INFINITY,
        };
        assert_invalid(
            link_margin(&budget).unwrap_err(),
            "required_cn0_dbhz",
            "not finite",
        );
    }

    #[test]
    fn fspl_batch_equals_scalar() {
        let distances_km = [1200.0, 1.0, 0.0, 42.5];
        let frequency_mhz = 1616.0;

        let batch = fspl_batch(&distances_km, frequency_mhz);

        assert_eq!(batch.len(), distances_km.len());
        for (actual, &distance_km) in batch.iter().zip(&distances_km) {
            let expected = fspl(distance_km, frequency_mhz);
            assert_rf_result_bits_eq(*actual, expected);
        }
    }

    #[test]
    fn link_margin_batch_equals_scalar() {
        let budgets = [
            LinkBudget {
                eirp_dbw: 0.0,
                fspl_db: 165.0,
                receiver_gt_dbk: -12.0,
                other_losses_db: 3.0,
                required_cn0_dbhz: 35.0,
            },
            LinkBudget {
                eirp_dbw: 8.0,
                fspl_db: 155.5,
                receiver_gt_dbk: -8.25,
                other_losses_db: 1.5,
                required_cn0_dbhz: 40.0,
            },
            LinkBudget {
                eirp_dbw: 0.0,
                fspl_db: 165.0,
                receiver_gt_dbk: -12.0,
                other_losses_db: 3.0,
                required_cn0_dbhz: f64::NAN,
            },
        ];

        let batch = link_margin_batch(&budgets);

        assert_eq!(batch.len(), budgets.len());
        for (actual, budget) in batch.iter().zip(&budgets) {
            let expected = link_margin(budget);
            assert_rf_result_bits_eq(*actual, expected);
        }
    }

    fn assert_invalid(error: RfError, field: &'static str, reason: &'static str) {
        assert_eq!(error, RfError::InvalidInput { field, reason });
    }

    fn assert_rf_result_bits_eq(actual: Result<f64, RfError>, expected: Result<f64, RfError>) {
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => assert_eq!(actual.to_bits(), expected.to_bits()),
            (Err(actual), Err(expected)) => assert_eq!(actual, expected),
            _ => panic!("actual {actual:?} did not match expected {expected:?}"),
        }
    }
}
