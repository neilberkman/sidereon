//! SBAS single-hypothesis protection levels.
//!
//! This module is sans-IO. Callers supply snapshot geometry and an externally
//! supplied range-error model. Message decoding and correction storage remain
//! in [`crate::sbas`].

pub mod error_model;

pub use crate::araim::{AraimGeometry as ProtectionGeometry, AraimRow as ProtectionRow};
pub use error_model::{
    give_variance_m2_for_givei, sbas_obliquity_factor, sigma_air_multipath_m,
    sigma_flt_m_for_udrei, sigma_tropo_m, udre_variance_m2_for_udrei, AirborneModel,
    DegradationParams, SbasErrorModel, SbasSisError, SBAS_IONOSPHERE_SHELL_HEIGHT_KM,
};

use crate::araim::protection::gain_matrix_enu;
use crate::araim::{AraimError, ProtectionModel};
use crate::integrity::{error_ellipse_2x2_unit, metric_cross, metric_sigma, IntegrityError};

/// Fixed SBAS protection-level multipliers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbasKMultipliers {
    /// Horizontal multiplier applied to the horizontal semi-major axis.
    pub k_h: f64,
    /// Vertical multiplier applied to the vertical one-sigma standard deviation.
    pub k_v: f64,
}

impl SbasKMultipliers {
    /// DO-229 precision-approach multipliers.
    pub const PRECISION_APPROACH: Self = Self {
        k_h: 6.0,
        k_v: 5.33,
    };

    /// DO-229 en-route through non-precision-approach multipliers.
    pub const EN_ROUTE_NPA: Self = Self {
        k_h: 6.18,
        k_v: 5.33,
    };
}

/// SBAS protection-level output for one geometry snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbasProtection {
    /// Horizontal protection level, meters.
    pub hpl_m: f64,
    /// Vertical protection level, meters.
    pub vpl_m: f64,
    /// Horizontal one-sigma semi-major axis, meters.
    pub d_major_m: f64,
    /// Vertical one-sigma standard deviation, meters.
    pub sigma_u_m: f64,
    /// East one-sigma standard deviation, meters.
    pub d_east_m: f64,
    /// North one-sigma standard deviation, meters.
    pub d_north_m: f64,
    /// East-north covariance term, square meters.
    pub d_en_m2: f64,
}

/// SBAS protection-level input or numerical failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SbasPlError {
    /// The geometry does not have enough independent rows for the active clocks.
    #[error("insufficient SBAS protection-level geometry")]
    InsufficientGeometry,
    /// A matrix operation or covariance projection failed.
    #[error("SBAS protection-level numerical failure")]
    NumericalFailure,
    /// The supplied error model is missing, non-finite, or outside its domain.
    #[error("invalid SBAS protection-level error model")]
    InvalidErrorModel,
    /// A satellite position was refused because producing it reads UT1
    /// outside the UT1 table under a strict UT1 policy.
    #[error("SBAS protection-level satellite position refused: {0}")]
    Ut1OutsideCoverage(crate::astro::time::DegradeReason),
}

/// Compute DO-229 SBAS HPL and VPL from geometry and supplied range sigmas.
///
/// The protection model supplies one externally determined range sigma per
/// geometry row. The function forms the same ENU gain matrix used by ARAIM,
/// projects the diagonal range covariance through that matrix, routes the
/// horizontal 2x2 covariance through [`error_ellipse_2x2_unit`], and applies the
/// fixed SBAS K multipliers.
pub fn sbas_protection_levels(
    geometry: &ProtectionGeometry,
    model: &dyn ProtectionModel,
    k: SbasKMultipliers,
) -> Result<SbasProtection, SbasPlError> {
    validate_k(k)?;
    if geometry.rows.is_empty() {
        return Err(SbasPlError::InsufficientGeometry);
    }

    let mut sigmas_m = Vec::with_capacity(geometry.rows.len());
    for row in &geometry.rows {
        let sigma_m = model.sigma_int_m(row).map_err(map_model_error)?;
        if !valid_positive_finite(sigma_m) {
            return Err(SbasPlError::InvalidErrorModel);
        }
        sigmas_m.push(sigma_m);
    }
    let weights = sigmas_m
        .iter()
        .map(|sigma_m| 1.0 / (sigma_m * sigma_m))
        .collect::<Vec<_>>();
    let gain = gain_matrix_enu(geometry, &weights).map_err(map_araim_error)?;

    let d_east_m = metric_sigma(&gain.enu_rows[0], &sigmas_m);
    let d_north_m = metric_sigma(&gain.enu_rows[1], &sigmas_m);
    let sigma_u_m = metric_sigma(&gain.enu_rows[2], &sigmas_m);
    let d_en_m2 = metric_cross(&gain.enu_rows[0], &gain.enu_rows[1], &sigmas_m);
    if [d_east_m, d_north_m, sigma_u_m, d_en_m2]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(SbasPlError::NumericalFailure);
    }

    let ellipse = error_ellipse_2x2_unit([
        [d_east_m * d_east_m, d_en_m2],
        [d_en_m2, d_north_m * d_north_m],
    ])
    .map_err(map_integrity_error)?;
    let d_major_m = ellipse.semi_major;
    Ok(SbasProtection {
        hpl_m: k.k_h * d_major_m,
        vpl_m: k.k_v * sigma_u_m,
        d_major_m,
        sigma_u_m,
        d_east_m,
        d_north_m,
        d_en_m2,
    })
}

fn validate_k(k: SbasKMultipliers) -> Result<(), SbasPlError> {
    if valid_positive_finite(k.k_h) && valid_positive_finite(k.k_v) {
        Ok(())
    } else {
        Err(SbasPlError::InvalidErrorModel)
    }
}

fn valid_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn map_model_error(error: AraimError) -> SbasPlError {
    match error {
        AraimError::InsufficientGeometry => SbasPlError::InsufficientGeometry,
        AraimError::InvalidIsm | AraimError::InvalidAllocation => SbasPlError::InvalidErrorModel,
        AraimError::UnmonitorableFaultMass | AraimError::NumericalFailure => {
            SbasPlError::NumericalFailure
        }
        AraimError::Ut1OutsideCoverage(reason) => SbasPlError::Ut1OutsideCoverage(reason),
    }
}

fn map_araim_error(error: AraimError) -> SbasPlError {
    match error {
        AraimError::InsufficientGeometry => SbasPlError::InsufficientGeometry,
        AraimError::InvalidIsm | AraimError::InvalidAllocation => SbasPlError::InvalidErrorModel,
        AraimError::UnmonitorableFaultMass | AraimError::NumericalFailure => {
            SbasPlError::NumericalFailure
        }
        AraimError::Ut1OutsideCoverage(reason) => SbasPlError::Ut1OutsideCoverage(reason),
    }
}

fn map_integrity_error(error: IntegrityError) -> SbasPlError {
    match error {
        IntegrityError::Singular => SbasPlError::InsufficientGeometry,
        IntegrityError::InvalidInput { .. }
        | IntegrityError::NonFinite
        | IntegrityError::NotPositiveSemidefinite
        | IntegrityError::InvalidProbability { .. } => SbasPlError::NumericalFailure,
    }
}
