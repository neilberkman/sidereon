use std::collections::BTreeSet;

use crate::quality::{pseudorange_variance, PseudorangeVarianceOptions};

use super::{
    validate_nonneg_finite, validate_positive_finite, validate_probability, AraimError, AraimRow,
};
use crate::id::{GnssSatelliteId, GnssSystem};

/// Per-satellite integrity and accuracy model without an identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatelliteIsmModel {
    /// Integrity one-sigma SIS range error, meters.
    pub sigma_ura_m: f64,
    /// Accuracy and continuity one-sigma SIS range error, meters.
    pub sigma_ure_m: f64,
    /// Effective integrity one-sigma range error after local terms, meters.
    pub effective_sigma_int_m: Option<f64>,
    /// Effective accuracy one-sigma range error after local terms, meters.
    pub effective_sigma_acc_m: Option<f64>,
    /// Nominal SIS bias bound, meters.
    pub b_nom_m: f64,
    /// Prior probability for a satellite fault.
    pub p_sat: f64,
}

impl SatelliteIsmModel {
    /// Construct a per-satellite model.
    pub const fn new(sigma_ura_m: f64, sigma_ure_m: f64, b_nom_m: f64, p_sat: f64) -> Self {
        Self {
            sigma_ura_m,
            sigma_ure_m,
            effective_sigma_int_m: None,
            effective_sigma_acc_m: None,
            b_nom_m,
            p_sat,
        }
    }

    /// Construct a per-satellite model with direct effective range sigmas.
    pub const fn new_with_effective_sigmas(
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
        effective_sigma_int_m: f64,
        effective_sigma_acc_m: f64,
    ) -> Self {
        Self {
            sigma_ura_m,
            sigma_ure_m,
            effective_sigma_int_m: Some(effective_sigma_int_m),
            effective_sigma_acc_m: Some(effective_sigma_acc_m),
            b_nom_m,
            p_sat,
        }
    }

    /// Construct the ADD v3.0 Eq. (3) / Appendix A dual-frequency model at a
    /// row elevation. The returned effective sigmas combine the supplied SIS
    /// terms with the v3.0 troposphere, airborne-noise, and multipath variance.
    ///
    /// `elevation_rad` must be finite and within the Appendix A table range,
    /// 5 through 90 degrees inclusive. Galileo values at the five-degree table
    /// nodes are exact; between nodes this library uses linear interpolation,
    /// because the ADD specifies no interpolation rule.
    /// ADD v3.1 changes the airborne-noise coefficients; this constructor is
    /// intentionally pinned to v3.0 worked examples.
    pub fn add_v3_0(
        system: GnssSystem,
        elevation_rad: f64,
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
    ) -> Result<Self, AraimError> {
        let local_variance_m2 = add_v3_0_local_variance_m2(system, elevation_rad)?;
        Ok(Self::new_with_effective_sigmas(
            sigma_ura_m,
            sigma_ure_m,
            b_nom_m,
            p_sat,
            (sigma_ura_m * sigma_ura_m + local_variance_m2).sqrt(),
            (sigma_ure_m * sigma_ure_m + local_variance_m2).sqrt(),
        ))
    }
}

fn add_v3_0_local_variance_m2(system: GnssSystem, elevation_rad: f64) -> Result<f64, AraimError> {
    if !elevation_rad.is_finite()
        || !(5.0_f64.to_radians()..=90.0_f64.to_radians()).contains(&elevation_rad)
    {
        return Err(AraimError::InvalidIsm);
    }
    let elevation_deg = elevation_rad.to_degrees();
    let tropo = 0.12 * 1.001 / (0.002001 + libm::sin(elevation_rad).powi(2)).sqrt();
    let user_sigma = match system {
        GnssSystem::Gps => {
            let noise = 0.15 + 0.43 * libm::exp(-elevation_deg / 6.9);
            let multipath = 0.13 + 0.53 * libm::exp(-elevation_deg / 10.0);
            let l1 = 1575.42_f64;
            let l5 = 1176.45_f64;
            let gamma = (l1.powi(4) + l5.powi(4)) / (l1.powi(2) - l5.powi(2)).powi(2);
            (gamma * (noise * noise + multipath * multipath)).sqrt()
        }
        GnssSystem::Galileo => galileo_user_sigma_m(elevation_deg),
        _ => return Err(AraimError::UnsupportedConstellation(system)),
    };
    Ok(tropo * tropo + user_sigma * user_sigma)
}

fn galileo_user_sigma_m(elevation_deg: f64) -> f64 {
    const TABLE: [f64; 18] = [
        0.4529, 0.3553, 0.3063, 0.2638, 0.2593, 0.2555, 0.2504, 0.2438, 0.2396, 0.2359, 0.2339,
        0.2302, 0.2295, 0.2278, 0.2297, 0.2310, 0.2274, 0.2277,
    ];
    let position = ((elevation_deg.clamp(5.0, 90.0) - 5.0) / 5.0).min(17.0);
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(TABLE.len() - 1);
    TABLE[lower] + (TABLE[upper] - TABLE[lower]) * (position - lower as f64)
}

/// Per-satellite ISM override.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatelliteIsm {
    /// Satellite identity for this override.
    pub id: GnssSatelliteId,
    /// Integrity one-sigma SIS range error, meters.
    pub sigma_ura_m: f64,
    /// Accuracy and continuity one-sigma SIS range error, meters.
    pub sigma_ure_m: f64,
    /// Effective integrity one-sigma range error after local terms, meters.
    pub effective_sigma_int_m: Option<f64>,
    /// Effective accuracy one-sigma range error after local terms, meters.
    pub effective_sigma_acc_m: Option<f64>,
    /// Nominal SIS bias bound, meters.
    pub b_nom_m: f64,
    /// Prior probability for a satellite fault.
    pub p_sat: f64,
}

impl SatelliteIsm {
    /// Construct a satellite-specific model.
    pub const fn new(
        id: GnssSatelliteId,
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
    ) -> Self {
        Self {
            id,
            sigma_ura_m,
            sigma_ure_m,
            effective_sigma_int_m: None,
            effective_sigma_acc_m: None,
            b_nom_m,
            p_sat,
        }
    }

    /// Construct a satellite-specific model with direct effective range sigmas.
    pub const fn new_with_effective_sigmas(
        id: GnssSatelliteId,
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
        effective_sigma_int_m: f64,
        effective_sigma_acc_m: f64,
    ) -> Self {
        Self {
            id,
            sigma_ura_m,
            sigma_ure_m,
            effective_sigma_int_m: Some(effective_sigma_int_m),
            effective_sigma_acc_m: Some(effective_sigma_acc_m),
            b_nom_m,
            p_sat,
        }
    }

    pub(crate) const fn model(self) -> SatelliteIsmModel {
        SatelliteIsmModel {
            sigma_ura_m: self.sigma_ura_m,
            sigma_ure_m: self.sigma_ure_m,
            effective_sigma_int_m: self.effective_sigma_int_m,
            effective_sigma_acc_m: self.effective_sigma_acc_m,
            b_nom_m: self.b_nom_m,
            p_sat: self.p_sat,
        }
    }
}

/// Per-constellation fault prior and default satellite model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstellationIsm {
    /// Constellation identity.
    pub system: GnssSystem,
    /// Prior probability for a constellation-wide fault.
    pub p_const: f64,
    /// Default satellite model for rows in this constellation.
    pub default_sat: SatelliteIsmModel,
}

impl ConstellationIsm {
    /// Construct a per-constellation model.
    pub const fn new(system: GnssSystem, p_const: f64, default_sat: SatelliteIsmModel) -> Self {
        Self {
            system,
            p_const,
            default_sat,
        }
    }
}

/// Parsed integrity support message used by ARAIM.
#[derive(Debug, Clone, PartialEq)]
pub struct Ism {
    /// Per-constellation defaults and constellation-wide fault priors.
    pub constellations: Vec<ConstellationIsm>,
    /// Per-satellite overrides.
    pub satellites: Vec<SatelliteIsm>,
}

impl Ism {
    /// Construct an ISM from parsed records.
    pub fn new(constellations: Vec<ConstellationIsm>, satellites: Vec<SatelliteIsm>) -> Self {
        Self {
            constellations,
            satellites,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), AraimError> {
        if self.constellations.is_empty() {
            return Err(AraimError::InvalidIsm);
        }

        let mut systems = BTreeSet::new();
        for constellation in &self.constellations {
            if !systems.insert(constellation.system) {
                return Err(AraimError::InvalidIsm);
            }
            if !validate_probability(constellation.p_const, true)
                || !validate_sat_model(constellation.default_sat)
            {
                return Err(AraimError::InvalidIsm);
            }
        }

        let mut satellites = BTreeSet::new();
        for sat in &self.satellites {
            if !satellites.insert(sat.id) || !validate_sat_model(sat.model()) {
                return Err(AraimError::InvalidIsm);
            }
            if self.constellation(sat.id.system).is_none() {
                return Err(AraimError::InvalidIsm);
            }
        }
        Ok(())
    }

    pub(crate) fn constellation(&self, system: GnssSystem) -> Option<&ConstellationIsm> {
        self.constellations
            .iter()
            .find(|constellation| constellation.system == system)
    }

    pub(crate) fn model_for(&self, id: GnssSatelliteId) -> Option<SatelliteIsmModel> {
        self.satellites
            .iter()
            .find(|sat| sat.id == id)
            .map(|sat| sat.model())
            .or_else(|| self.constellation(id.system).map(|c| c.default_sat))
    }

    pub(crate) fn effective_for(&self, row: &AraimRow) -> Result<EffectiveIsm, AraimError> {
        if !row.elevation_rad.is_finite() || row.elevation_rad <= 0.0 {
            return Err(AraimError::InvalidIsm);
        }
        let model = self.model_for(row.id).ok_or(AraimError::InvalidIsm)?;
        let (sigma_int_m, sigma_acc_m) = if let (Some(sigma_int_m), Some(sigma_acc_m)) =
            (model.effective_sigma_int_m, model.effective_sigma_acc_m)
        {
            (sigma_int_m, sigma_acc_m)
        } else {
            let elevation_deg = row.elevation_rad.to_degrees();
            let local_variance_m2 =
                pseudorange_variance(elevation_deg, PseudorangeVarianceOptions::default())
                    .map_err(|_| AraimError::InvalidIsm)?;
            let sigma_int_m2 = model.sigma_ura_m * model.sigma_ura_m + local_variance_m2;
            let sigma_acc_m2 = model.sigma_ure_m * model.sigma_ure_m + local_variance_m2;
            if !validate_positive_finite(sigma_int_m2) || !validate_positive_finite(sigma_acc_m2) {
                return Err(AraimError::InvalidIsm);
            }
            (sigma_int_m2.sqrt(), sigma_acc_m2.sqrt())
        };
        if !validate_positive_finite(sigma_int_m)
            || !validate_positive_finite(sigma_acc_m)
            || sigma_acc_m > sigma_int_m
        {
            return Err(AraimError::InvalidIsm);
        }
        Ok(EffectiveIsm {
            sigma_int_m,
            sigma_acc_m,
            b_nom_m: model.b_nom_m,
            p_sat: model.p_sat,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EffectiveIsm {
    pub sigma_int_m: f64,
    pub sigma_acc_m: f64,
    pub b_nom_m: f64,
    pub p_sat: f64,
}

fn validate_sat_model(model: SatelliteIsmModel) -> bool {
    let valid_effective_sigmas = match (model.effective_sigma_int_m, model.effective_sigma_acc_m) {
        (Some(sigma_int_m), Some(sigma_acc_m)) => {
            validate_positive_finite(sigma_int_m)
                && validate_positive_finite(sigma_acc_m)
                && sigma_acc_m <= sigma_int_m
        }
        (None, None) => true,
        _ => false,
    };
    validate_positive_finite(model.sigma_ura_m)
        && validate_positive_finite(model.sigma_ure_m)
        && model.sigma_ure_m <= model.sigma_ura_m
        && validate_nonneg_finite(model.b_nom_m)
        && validate_probability(model.p_sat, true)
        && valid_effective_sigmas
}
