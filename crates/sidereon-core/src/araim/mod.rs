//! Advanced RAIM multi-hypothesis snapshot integrity.
//!
//! This module is sans-IO: callers provide line-of-sight geometry plus an
//! externally supplied integrity support message, and the solver returns
//! protection levels without reading products, global state, or residuals.
//! ISM records can use the default local pseudorange variance model or direct
//! per-satellite effective sigmas for reference cases that publish `Cint` and
//! `Cacc` diagonals.

/// Enumerates the fault hypotheses used by the ARAIM multi-hypothesis solve.
/// [`crate::araim::enumerate_fault_modes`] returns a fault-free hypothesis
/// first, followed by allowed satellite and constellation events and then
/// higher-order satellite candidates ordered by prior mass.
pub mod fault_modes;
/// Holds the integrity support models consumed by ARAIM.
/// [`crate::araim::Ism`] combines constellation defaults with optional
/// satellite overrides; its effective-model path either uses supplied
/// effective sigmas or adds local pseudorange variance to the URA/URE
/// variances before taking their square roots.
pub mod ism;
mod mhss;
/// Supplies the range-error interface and gain-matrix primitives used by ARAIM
/// and single-hypothesis protection calculations.
/// [`crate::araim::ProtectionModel`] implementations provide external
/// per-row sigmas; the module projects weighted line-of-sight designs into ENU
/// and applies the WG-C protection equations to derive protection levels.
pub mod protection;
pub mod reliability;

#[cfg(test)]
mod tests;

pub use fault_modes::{enumerate_fault_modes, FaultHypothesis};
pub use ism::{ConstellationIsm, Ism, SatelliteIsm, SatelliteIsmModel};
pub use mhss::{araim, AraimResult, FaultMode};
pub use protection::ProtectionModel;
pub use reliability::{
    reliability_araim, reliability_design, wtest_noncentrality, wtest_noncentrality_components,
    ObservationReliability, RangeReliabilityRow, ReliabilityOptions, ReliabilityReport,
    ReliabilitySummary, WtestNoncentralityComponents,
};

use crate::astro::frames::transforms::geodetic_from_ecef_proj;
use crate::dop::{ecef_to_enu_rotation, LineOfSight};
use crate::frame::Wgs84Geodetic;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::spp::{EphemerisSource, ReceiverSolution};

/// One satellite row in an ARAIM geometry snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AraimRow {
    /// Satellite identity used for ISM lookup and satellite-fault modes.
    pub id: GnssSatelliteId,
    /// Receiver-to-satellite ECEF unit vector.
    pub line_of_sight: LineOfSight,
    /// Constellation owning the signal and constellation-fault mode.
    pub system: GnssSystem,
    /// Elevation angle at the receiver, radians.
    pub elevation_rad: f64,
}

/// A snapshot geometry and clock-column convention for ARAIM.
#[derive(Debug, Clone, PartialEq)]
pub struct AraimGeometry {
    /// Satellite rows, index-aligned through all gain matrices.
    pub rows: Vec<AraimRow>,
    /// Receiver geodetic position for ENU rotation.
    pub receiver: Wgs84Geodetic,
    /// Receiver-clock columns, in the same order as the SPP state.
    pub clock_systems: Vec<GnssSystem>,
}

impl AraimGeometry {
    /// Build ARAIM geometry from an SPP solution.
    ///
    /// The SPP solution carries the final receiver state and used satellite IDs.
    /// `t_j2000_s` is the receive epoch used to query the ephemeris source.
    pub fn from_receiver_solution(
        solution: &ReceiverSolution,
        eph: &dyn EphemerisSource,
        t_j2000_s: f64,
    ) -> Result<Self, AraimError> {
        if !t_j2000_s.is_finite() {
            return Err(AraimError::InsufficientGeometry);
        }
        let receiver = match solution.geodetic {
            Some(receiver) => receiver,
            None => geodetic_from_position(solution.position.as_array())?,
        };
        let clock_systems = receiver_solution_clock_systems(solution)?;
        if solution.used_sats.len() < 3 + clock_systems.len() {
            return Err(AraimError::InsufficientGeometry);
        }

        let rx_ecef_m = solution.position.as_array();
        let enu = ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad);
        let mut rows = Vec::with_capacity(solution.used_sats.len());
        for &id in &solution.used_sats {
            let (sat_ecef_m, _) = eph
                .position_clock_at_j2000_s(id, t_j2000_s)
                .ok_or(AraimError::InsufficientGeometry)?;
            let dx = sat_ecef_m[0] - rx_ecef_m[0];
            let dy = sat_ecef_m[1] - rx_ecef_m[1];
            let dz = sat_ecef_m[2] - rx_ecef_m[2];
            let range_m = (dx * dx + dy * dy + dz * dz).sqrt();
            if !range_m.is_finite() || range_m <= 0.0 {
                return Err(AraimError::InsufficientGeometry);
            }

            let line_of_sight = LineOfSight::new(dx / range_m, dy / range_m, dz / range_m);
            let up = enu[2][0] * line_of_sight.e_x
                + enu[2][1] * line_of_sight.e_y
                + enu[2][2] * line_of_sight.e_z;
            let elevation_rad = libm::asin(up.clamp(-1.0, 1.0));
            if !elevation_rad.is_finite() {
                return Err(AraimError::InsufficientGeometry);
            }

            rows.push(AraimRow {
                id,
                line_of_sight,
                system: id.system,
                elevation_rad,
            });
        }

        Ok(Self {
            rows,
            receiver,
            clock_systems,
        })
    }
}

fn geodetic_from_position(position_m: [f64; 3]) -> Result<Wgs84Geodetic, AraimError> {
    let [lon_deg, lat_deg, height_m] =
        geodetic_from_ecef_proj(position_m[0], position_m[1], position_m[2])
            .map_err(|_| AraimError::InsufficientGeometry)?;
    Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), height_m)
        .map_err(|_| AraimError::InsufficientGeometry)
}

fn receiver_solution_clock_systems(
    solution: &ReceiverSolution,
) -> Result<Vec<GnssSystem>, AraimError> {
    let clock_systems = if !solution.system_clocks_s.is_empty() {
        solution
            .system_clocks_s
            .iter()
            .map(|&(system, _)| system)
            .collect()
    } else if !solution.metadata.systems.is_empty() {
        solution.metadata.systems.clone()
    } else {
        crate::spp::clock_systems(&solution.used_sats)
    };
    if clock_systems.is_empty() {
        return Err(AraimError::InsufficientGeometry);
    }
    for (idx, system) in clock_systems.iter().enumerate() {
        if clock_systems[..idx].contains(system) {
            return Err(AraimError::InsufficientGeometry);
        }
    }
    Ok(clock_systems)
}

/// Integrity and continuity risk allocation for one ARAIM solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrityAllocation {
    /// Total probability of hazardous misleading information.
    pub phmi_total: f64,
    /// Vertical PHMI allocation.
    pub phmi_vert: f64,
    /// Horizontal PHMI allocation.
    pub phmi_hor: f64,
    /// Vertical false-alert allocation.
    pub pfa_vert: f64,
    /// Horizontal false-alert allocation.
    pub pfa_hor: f64,
    /// Maximum acceptable unmonitored fault probability mass.
    pub p_threshold_unmonitored: f64,
    /// Fault-prior threshold used for the effective monitor threshold.
    pub p_emt: f64,
    /// Maximum enumerated satellite-fault order. Zero keeps only fault-free.
    pub max_fault_order: usize,
}

impl IntegrityAllocation {
    /// LPV-200 allocation from Blanch et al. 2015 and WG-C Milestone 3.
    pub const fn lpv_200() -> Self {
        Self {
            phmi_total: 1.0e-7,
            phmi_vert: 9.8e-8,
            phmi_hor: 2.0e-9,
            pfa_vert: 3.9e-6,
            pfa_hor: 9.0e-8,
            // WG-C Reference ADD v3.0 Table 3, LPV-200 PTHRES.
            p_threshold_unmonitored: 8.0e-8,
            // WG-C Reference ADD v3.0 Table 2, LPV-200 PEMT.
            p_emt: 1.0e-5,
            max_fault_order: 2,
        }
    }
}

/// ARAIM input or numerical failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AraimError {
    /// The full or subset geometry does not have enough independent rows.
    #[error("insufficient ARAIM geometry")]
    InsufficientGeometry,
    /// The unmonitorable fault probability exceeds the allocation.
    #[error("unmonitorable ARAIM fault probability exceeds allocation")]
    UnmonitorableFaultMass,
    /// A matrix operation or root solve failed.
    #[error("ARAIM numerical failure")]
    NumericalFailure,
    /// The ISM is missing, non-finite, or outside its valid domain.
    #[error("invalid ARAIM ISM")]
    InvalidIsm,
    /// The integrity allocation is missing, non-finite, or outside its domain.
    #[error("invalid ARAIM allocation")]
    InvalidAllocation,
}

pub(crate) fn clock_system_for_row(system: GnssSystem) -> GnssSystem {
    match system {
        GnssSystem::Sbas => GnssSystem::Gps,
        other => other,
    }
}

pub(crate) fn validate_probability(value: f64, allow_zero: bool) -> bool {
    value.is_finite()
        && if allow_zero {
            (0.0..1.0).contains(&value) || value == 0.0
        } else {
            (0.0..1.0).contains(&value)
        }
}

pub(crate) fn validate_nonneg_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

pub(crate) fn validate_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
