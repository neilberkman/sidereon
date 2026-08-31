//! DO-229 SBAS range-error model.

use crate::araim::{AraimError, AraimRow, ProtectionModel};
use crate::constants::MEAN_EARTH_RADIUS_KM;
use crate::dop::ecef_to_enu_rotation;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::ionex::pierce_point;
use crate::sbas::SbasCorrectionStore;
pub use crate::sbas::{give_variance_m2_for_givei, udre_variance_m2_for_udrei};

use super::{ProtectionGeometry, SbasPlError};

/// SBAS ionospheric shell height used by DO-229, in kilometers.
pub const SBAS_IONOSPHERE_SHELL_HEIGHT_KM: f64 = 350.0;

/// One satellite's DO-229 one-sigma range-error budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbasSisError {
    /// Satellite identity matching a protection geometry row.
    pub id: GnssSatelliteId,
    /// Fast and long-term correction residual sigma, meters.
    pub sigma_flt_m: f64,
    /// User ionospheric range-error sigma, meters.
    pub sigma_uire_m: f64,
    /// Airborne receiver noise, divergence, and multipath sigma, meters.
    pub sigma_air_m: f64,
    /// Tropospheric residual sigma, meters.
    pub sigma_tropo_m: f64,
}

impl SbasSisError {
    /// Sum-of-squares range variance for this satellite, in square meters.
    pub fn variance_m2(self) -> Option<f64> {
        if !valid_nonnegative_finite(self.sigma_flt_m)
            || !valid_nonnegative_finite(self.sigma_uire_m)
            || !valid_nonnegative_finite(self.sigma_air_m)
            || !valid_nonnegative_finite(self.sigma_tropo_m)
        {
            return None;
        }
        let variance_m2 = self.sigma_flt_m * self.sigma_flt_m
            + self.sigma_uire_m * self.sigma_uire_m
            + self.sigma_air_m * self.sigma_air_m
            + self.sigma_tropo_m * self.sigma_tropo_m;
        (variance_m2.is_finite() && variance_m2 > 0.0).then_some(variance_m2)
    }

    /// Total one-sigma range error, meters.
    pub fn sigma_m(self) -> Option<f64> {
        self.variance_m2().map(f64::sqrt)
    }
}

/// Index-aligned SBAS error model for protection-level geometry rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SbasErrorModel {
    /// Per-satellite range-error rows.
    pub rows: Vec<SbasSisError>,
}

impl SbasErrorModel {
    /// Construct an SBAS error model from supplied per-satellite rows.
    pub fn new(rows: Vec<SbasSisError>) -> Self {
        Self { rows }
    }

    /// Build an SBAS error model from decoded SBAS correction storage.
    ///
    /// `epoch_j2000_s` is the receive epoch used for freshness checks. Missing
    /// fast corrections, missing GIVE variance, stale corrections, or invalid
    /// degradation inputs return [`SbasPlError::InvalidErrorModel`].
    pub fn from_store(
        store: &SbasCorrectionStore,
        geo: GnssSatelliteId,
        geometry: &ProtectionGeometry,
        airborne: &AirborneModel,
        epoch_j2000_s: f64,
        degradation: &DegradationParams,
    ) -> Result<Self, SbasPlError> {
        if geo.system != GnssSystem::Sbas || !epoch_j2000_s.is_finite() || !degradation.is_valid() {
            return Err(SbasPlError::InvalidErrorModel);
        }
        let iono_grid = store
            .fresh_iono_grid(geo, epoch_j2000_s)
            .ok_or(SbasPlError::InvalidErrorModel)?;
        let mut rows = Vec::with_capacity(geometry.rows.len());
        for row in &geometry.rows {
            let fast = store
                .fresh_fast(geo, row.id, epoch_j2000_s)
                .ok_or(SbasPlError::InvalidErrorModel)?;
            let sigma_flt_m = sigma_flt_m_for_udrei(fast.udrei, degradation)
                .ok_or(SbasPlError::InvalidErrorModel)?;
            let (ipp_lat_deg, ipp_lon_deg, mapping) =
                pierce_point_for_row(geometry, row).ok_or(SbasPlError::InvalidErrorModel)?;
            let uive_variance_m2 = iono_grid
                .variance_at_ipp(ipp_lat_deg, ipp_lon_deg)
                .ok_or(SbasPlError::InvalidErrorModel)?;
            let sigma_uire_m = mapping * uive_variance_m2.sqrt() + degradation.eps_iono_m;
            let sigma_air_m = airborne
                .sigma_air_m(row.elevation_rad)
                .ok_or(SbasPlError::InvalidErrorModel)?;
            let sigma_tropo_m =
                sigma_tropo_m(row.elevation_rad).ok_or(SbasPlError::InvalidErrorModel)?;
            let sis = SbasSisError {
                id: row.id,
                sigma_flt_m,
                sigma_uire_m,
                sigma_air_m,
                sigma_tropo_m,
            };
            if sis.sigma_m().is_none() {
                return Err(SbasPlError::InvalidErrorModel);
            }
            rows.push(sis);
        }
        Ok(Self { rows })
    }

    /// Return the range-error row for one satellite.
    pub fn row_for(&self, id: GnssSatelliteId) -> Option<&SbasSisError> {
        self.rows.iter().find(|row| row.id == id)
    }
}

impl ProtectionModel for SbasErrorModel {
    fn sigma_int_m(&self, row: &AraimRow) -> Result<f64, AraimError> {
        self.row_for(row.id)
            .and_then(|sis| sis.sigma_m())
            .ok_or(AraimError::InvalidIsm)
    }

    fn sigma_acc_m(&self, row: &AraimRow) -> Result<f64, AraimError> {
        self.sigma_int_m(row)
    }
}

/// Airborne receiver and multipath contribution model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirborneModel {
    /// Receiver noise and code-carrier divergence sigma, meters.
    pub sigma_noise_divergence_m: f64,
}

impl AirborneModel {
    /// Construct an airborne model from a supplied receiver noise term.
    pub const fn new(sigma_noise_divergence_m: f64) -> Self {
        Self {
            sigma_noise_divergence_m,
        }
    }

    /// DO-229 AAD-A airborne model.
    pub const fn aad_a() -> Self {
        Self::new(0.36)
    }

    /// Airborne receiver, divergence, and multipath sigma at an elevation.
    pub fn sigma_air_m(self, elevation_rad: f64) -> Option<f64> {
        if !valid_nonnegative_finite(self.sigma_noise_divergence_m) {
            return None;
        }
        let sigma_mp_m = sigma_air_multipath_m(elevation_rad)?;
        let sigma_air_m = (self.sigma_noise_divergence_m * self.sigma_noise_divergence_m
            + sigma_mp_m * sigma_mp_m)
            .sqrt();
        sigma_air_m.is_finite().then_some(sigma_air_m)
    }
}

impl Default for AirborneModel {
    fn default() -> Self {
        Self::aad_a()
    }
}

/// Supplied DO-229 degradation terms not yet decoded from MT10, MT27, or MT28.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegradationParams {
    /// Variance multiplier applied to the UDRE variance table.
    pub delta_udre: f64,
    /// Fast-correction degradation term, meters.
    pub eps_fc_m: f64,
    /// Range-rate-correction degradation term, meters.
    pub eps_rrc_m: f64,
    /// Long-term-correction degradation term, meters.
    pub eps_ltc_m: f64,
    /// En-route degradation term, meters.
    pub eps_er_m: f64,
    /// Ionospheric degradation term added to UIRE, meters.
    pub eps_iono_m: f64,
    /// True when UDRE degradation terms are combined by root-sum-square.
    pub rss_udre: bool,
}

impl DegradationParams {
    /// No extra degradation and no UDRE inflation.
    pub const fn none() -> Self {
        Self {
            delta_udre: 1.0,
            eps_fc_m: 0.0,
            eps_rrc_m: 0.0,
            eps_ltc_m: 0.0,
            eps_er_m: 0.0,
            eps_iono_m: 0.0,
            rss_udre: false,
        }
    }

    /// True when every supplied degradation term is finite and non-negative.
    pub fn is_valid(self) -> bool {
        valid_positive_finite(self.delta_udre)
            && valid_nonnegative_finite(self.eps_fc_m)
            && valid_nonnegative_finite(self.eps_rrc_m)
            && valid_nonnegative_finite(self.eps_ltc_m)
            && valid_nonnegative_finite(self.eps_er_m)
            && valid_nonnegative_finite(self.eps_iono_m)
    }
}

impl Default for DegradationParams {
    fn default() -> Self {
        Self::none()
    }
}

/// Compute fast and long-term residual sigma from a UDREI and degradation terms.
pub fn sigma_flt_m_for_udrei(udrei: u8, degradation: &DegradationParams) -> Option<f64> {
    if !degradation.is_valid() {
        return None;
    }
    let udre_variance_m2 = udre_variance_m2_for_udrei(udrei)?;
    let base_variance_m2 = degradation.delta_udre * udre_variance_m2;
    let eps_variance_m2 = if degradation.rss_udre {
        degradation.eps_fc_m * degradation.eps_fc_m
            + degradation.eps_rrc_m * degradation.eps_rrc_m
            + degradation.eps_ltc_m * degradation.eps_ltc_m
            + degradation.eps_er_m * degradation.eps_er_m
    } else {
        let eps_m = degradation.eps_fc_m
            + degradation.eps_rrc_m
            + degradation.eps_ltc_m
            + degradation.eps_er_m;
        eps_m * eps_m
    };
    let variance_m2 = base_variance_m2 + eps_variance_m2;
    (variance_m2.is_finite() && variance_m2 >= 0.0).then_some(variance_m2.sqrt())
}

/// DO-229 tropospheric residual sigma at an elevation angle, meters.
pub fn sigma_tropo_m(elevation_rad: f64) -> Option<f64> {
    if !elevation_rad.is_finite() {
        return None;
    }
    let sin_el = libm::sin(elevation_rad);
    let sigma_m = 0.12 * 1.001 / (0.002001 + sin_el * sin_el).sqrt();
    sigma_m.is_finite().then_some(sigma_m)
}

/// DO-229 SBAS obliquity factor at an elevation angle.
pub fn sbas_obliquity_factor(elevation_rad: f64) -> Option<f64> {
    if !elevation_rad.is_finite() {
        return None;
    }
    let s = MEAN_EARTH_RADIUS_KM * libm::cos(elevation_rad)
        / (MEAN_EARTH_RADIUS_KM + SBAS_IONOSPHERE_SHELL_HEIGHT_KM);
    let denominator = 1.0 - s * s;
    (denominator.is_finite() && denominator > 0.0).then_some(1.0 / denominator.sqrt())
}

/// DO-229 airborne multipath sigma at an elevation angle, meters.
pub fn sigma_air_multipath_m(elevation_rad: f64) -> Option<f64> {
    if !elevation_rad.is_finite() {
        return None;
    }
    let elevation_deg = elevation_rad.to_degrees();
    let sigma_m = 0.13 + 0.53 * libm::exp(-elevation_deg / 10.0);
    sigma_m.is_finite().then_some(sigma_m)
}

fn pierce_point_for_row(geometry: &ProtectionGeometry, row: &AraimRow) -> Option<(f64, f64, f64)> {
    let azimuth_rad = azimuth_for_row(geometry, row)?;
    let ipp = pierce_point(
        geometry.receiver.lat_rad,
        geometry.receiver.lon_rad,
        azimuth_rad,
        row.elevation_rad,
        MEAN_EARTH_RADIUS_KM,
        SBAS_IONOSPHERE_SHELL_HEIGHT_KM,
    );
    let mapping = sbas_obliquity_factor(row.elevation_rad)?;
    Some((
        ipp.phi_ipp_deg,
        normalize_lon_deg(ipp.lambda_ipp_deg),
        mapping,
    ))
}

fn azimuth_for_row(geometry: &ProtectionGeometry, row: &AraimRow) -> Option<f64> {
    let los = row.line_of_sight;
    if !los.e_x.is_finite() || !los.e_y.is_finite() || !los.e_z.is_finite() {
        return None;
    }
    let r = ecef_to_enu_rotation(geometry.receiver.lat_rad, geometry.receiver.lon_rad);
    let east = r[0][0] * los.e_x + r[0][1] * los.e_y + r[0][2] * los.e_z;
    let north = r[1][0] * los.e_x + r[1][1] * los.e_y + r[1][2] * los.e_z;
    let horizontal = libm::hypot(east, north);
    if !horizontal.is_finite() {
        return None;
    }
    if horizontal <= f64::EPSILON {
        Some(0.0)
    } else {
        Some(libm::atan2(east, north))
    }
}

fn valid_nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn valid_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn normalize_lon_deg(mut lon_deg: f64) -> f64 {
    while lon_deg < -180.0 {
        lon_deg += 360.0;
    }
    while lon_deg > 180.0 {
        lon_deg -= 360.0;
    }
    lon_deg
}
