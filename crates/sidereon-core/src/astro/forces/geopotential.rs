//! Spherical-harmonic geopotential perturbation force.
//!
//! The model evaluates fully normalized real coefficients in a body-fixed
//! frame, beginning at degree 2 so the central two-body term remains separate.
//! If a propagation context supplies an Earth-orientation provider, the input
//! GCRF position is rotated into ITRF and the resulting acceleration is rotated
//! back to GCRF. Non-zonal terms require that provider; zonal-only models can
//! be evaluated directly in the input frame.

use crate::astro::constants::{MU_EARTH, RE_EARTH};
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

/// EGM96 gravitational parameter used with the embedded coefficient table.
pub const EGM96_MU_KM3_S2: f64 = 398_600.441_5;
/// EGM96 reference equatorial radius used with the embedded coefficient table.
pub const EGM96_REFERENCE_RADIUS_KM: f64 = 6_378.136_3;
/// Maximum embedded EGM96 degree.
pub const EGM96_EMBEDDED_MAX_DEGREE: u16 = 36;
/// Maximum embedded EGM96 order.
pub const EGM96_EMBEDDED_MAX_ORDER: u16 = 36;
/// Embedded EGM96 fully normalized coefficient table through degree and order 36.
///
/// Each non-comment row contains `degree order Cnm Snm sigmaC sigmaS`. The
/// parser ignores the sigma columns. Coefficients are dimensionless and use the
/// fully normalized real convention for `Pnm(sin latitude)`.
pub const EGM96_DEGREE_ORDER_36: &str = include_str!("egm96_to_36.txt");

#[derive(Debug, Clone, Copy, PartialEq)]
struct CoefficientPair {
    c: f64,
    s: f64,
}

impl CoefficientPair {
    const ZERO: Self = Self { c: 0.0, s: 0.0 };
}

/// Fully normalized real spherical-harmonic coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalHarmonicCoefficient {
    /// Harmonic degree `n`.
    pub degree: u16,
    /// Harmonic order `m`.
    pub order: u16,
    /// Fully normalized cosine coefficient `Cnm`.
    pub c: f64,
    /// Fully normalized sine coefficient `Snm`.
    pub s: f64,
}

/// Copyable embedded-table selector for propagation force composition.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SphericalHarmonicGravityConfig {
    /// Gravitational parameter, km^3/s^2.
    pub mu_km3_s2: f64,
    /// Reference equatorial radius, km.
    pub reference_radius_km: f64,
    /// Highest active degree.
    pub max_degree: u16,
    /// Highest active order.
    pub max_order: u16,
}

impl SphericalHarmonicGravityConfig {
    /// Build a selector from explicit gravity constants and degree/order limits.
    ///
    /// All four fields are required because the constants and active harmonic
    /// limits directly determine the resulting perturbation.
    pub fn new(
        mu_km3_s2: f64,
        reference_radius_km: f64,
        max_degree: u16,
        max_order: u16,
    ) -> Result<Self, PropagationError> {
        validate_finite_positive(mu_km3_s2, "mu_km3_s2")?;
        validate_finite_positive(reference_radius_km, "reference_radius_km")?;
        validate_degree_order(max_degree, max_order)?;
        if max_degree > EGM96_EMBEDDED_MAX_DEGREE || max_order > EGM96_EMBEDDED_MAX_ORDER {
            return Err(PropagationError::InvalidInput(
                "EGM96 embedded degree/order must not exceed 36".to_string(),
            ));
        }
        Ok(Self {
            mu_km3_s2,
            reference_radius_km,
            max_degree,
            max_order,
        })
    }

    /// Build an EGM96 embedded-table selector.
    pub fn egm96(max_degree: u16, max_order: u16) -> Result<Self, PropagationError> {
        validate_degree_order(max_degree, max_order)?;
        if max_degree > EGM96_EMBEDDED_MAX_DEGREE || max_order > EGM96_EMBEDDED_MAX_ORDER {
            return Err(PropagationError::InvalidInput(
                "EGM96 embedded degree/order must not exceed 36".to_string(),
            ));
        }
        Ok(Self {
            mu_km3_s2: EGM96_MU_KM3_S2,
            reference_radius_km: EGM96_REFERENCE_RADIUS_KM,
            max_degree,
            max_order,
        })
    }

    /// Build a WGS84-constant selector over the embedded EGM96 table.
    pub fn earth(max_degree: u16, max_order: u16) -> Result<Self, PropagationError> {
        validate_degree_order(max_degree, max_order)?;
        if max_degree > EGM96_EMBEDDED_MAX_DEGREE || max_order > EGM96_EMBEDDED_MAX_ORDER {
            return Err(PropagationError::InvalidInput(
                "EGM96 embedded degree/order must not exceed 36".to_string(),
            ));
        }
        Ok(Self {
            mu_km3_s2: MU_EARTH,
            reference_radius_km: RE_EARTH,
            max_degree,
            max_order,
        })
    }

    /// Build the force model selected by this config.
    pub fn build(self) -> Result<SphericalHarmonicGravity, PropagationError> {
        SphericalHarmonicGravity::from_normalized_ascii_table(
            self.mu_km3_s2,
            self.reference_radius_km,
            self.max_degree,
            self.max_order,
            EGM96_DEGREE_ORDER_36,
        )
    }
}

/// Fully normalized spherical-harmonic gravity perturbation.
///
/// The force is perturbation-only: degree 0 and degree 1 rows are ignored, and
/// acceleration sums begin at degree 2.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalHarmonicGravity {
    mu_km3_s2: f64,
    reference_radius_km: f64,
    max_degree: u16,
    max_order: u16,
    coefficients: Vec<CoefficientPair>,
    has_non_zonal_terms: bool,
}

impl Default for SphericalHarmonicGravity {
    fn default() -> Self {
        Self::egm96_degree_order_36().expect("embedded EGM96 degree/order 36 is valid")
    }
}

impl SphericalHarmonicGravity {
    /// Build the embedded EGM96 degree and order 36 model.
    pub fn egm96_degree_order_36() -> Result<Self, PropagationError> {
        SphericalHarmonicGravityConfig::egm96(EGM96_EMBEDDED_MAX_DEGREE, EGM96_EMBEDDED_MAX_ORDER)?
            .build()
    }

    /// Build the embedded EGM96 model truncated to `max_degree` and `max_order`.
    pub fn egm96_truncated(max_degree: u16, max_order: u16) -> Result<Self, PropagationError> {
        SphericalHarmonicGravityConfig::egm96(max_degree, max_order)?.build()
    }

    /// Build an Earth-constant model over the embedded EGM96 coefficient table.
    pub fn earth_egm96_truncated(
        max_degree: u16,
        max_order: u16,
    ) -> Result<Self, PropagationError> {
        SphericalHarmonicGravityConfig::earth(max_degree, max_order)?.build()
    }

    /// Build from caller-supplied fully normalized coefficients.
    ///
    /// Coefficients above the requested degree or order are ignored. Degree 0
    /// and 1 coefficients are ignored because this force contains only the
    /// perturbation sum from degree 2 upward.
    pub fn from_normalized_coefficients(
        mu_km3_s2: f64,
        reference_radius_km: f64,
        max_degree: u16,
        max_order: u16,
        coefficients: &[SphericalHarmonicCoefficient],
    ) -> Result<Self, PropagationError> {
        validate_finite_positive(mu_km3_s2, "mu_km3_s2")?;
        validate_finite_positive(reference_radius_km, "reference_radius_km")?;
        validate_degree_order(max_degree, max_order)?;

        let len = triangular_len(max_degree);
        let mut dense = vec![CoefficientPair::ZERO; len];
        let mut seen = vec![false; len];
        let mut has_non_zonal_terms = false;

        for coefficient in coefficients {
            validate_coefficient(*coefficient)?;
            if coefficient.degree < 2
                || coefficient.degree > max_degree
                || coefficient.order > max_order
            {
                continue;
            }
            let idx = coefficient_index(coefficient.degree, coefficient.order);
            if seen[idx] {
                return Err(PropagationError::InvalidInput(format!(
                    "duplicate spherical harmonic coefficient {},{}",
                    coefficient.degree, coefficient.order
                )));
            }
            seen[idx] = true;
            dense[idx] = CoefficientPair {
                c: coefficient.c,
                s: coefficient.s,
            };
            if coefficient.order > 0 && (coefficient.c != 0.0 || coefficient.s != 0.0) {
                has_non_zonal_terms = true;
            }
        }

        Ok(Self {
            mu_km3_s2,
            reference_radius_km,
            max_degree,
            max_order,
            coefficients: dense,
            has_non_zonal_terms,
        })
    }

    /// Build from an ASCII coefficient table.
    ///
    /// Accepted data rows are either `n m Cnm Snm ...` or `gfc n m Cnm Snm ...`.
    /// Blank lines and lines beginning with `#` are ignored.
    pub fn from_normalized_ascii_table(
        mu_km3_s2: f64,
        reference_radius_km: f64,
        max_degree: u16,
        max_order: u16,
        table: &str,
    ) -> Result<Self, PropagationError> {
        let coefficients = parse_ascii_coefficients(table)?;
        Self::from_normalized_coefficients(
            mu_km3_s2,
            reference_radius_km,
            max_degree,
            max_order,
            &coefficients,
        )
    }

    /// Gravitational parameter, km^3/s^2.
    pub fn mu_km3_s2(&self) -> f64 {
        self.mu_km3_s2
    }

    /// Reference equatorial radius, km.
    pub fn reference_radius_km(&self) -> f64 {
        self.reference_radius_km
    }

    /// Highest active degree.
    pub fn max_degree(&self) -> u16 {
        self.max_degree
    }

    /// Highest active order.
    pub fn max_order(&self) -> u16 {
        self.max_order
    }

    /// Return the coefficient at `degree, order`, if it lies inside the model.
    pub fn coefficient(&self, degree: u16, order: u16) -> Option<SphericalHarmonicCoefficient> {
        if degree > self.max_degree || order > degree || order > self.max_order {
            return None;
        }
        let pair = self.coefficients[coefficient_index(degree, order)];
        Some(SphericalHarmonicCoefficient {
            degree,
            order,
            c: pair.c,
            s: pair.s,
        })
    }

    /// Evaluate the perturbation acceleration in the body-fixed frame.
    pub fn body_fixed_acceleration_km_s2(
        &self,
        position_body_fixed_km: [f64; 3],
    ) -> Result<[f64; 3], PropagationError> {
        validate_vec3(position_body_fixed_km, "position_body_fixed_km")?;

        let x = position_body_fixed_km[0];
        let y = position_body_fixed_km[1];
        let z = position_body_fixed_km[2];
        let rho2 = x * x + y * y;
        let r2 = rho2 + z * z;
        if r2 == 0.0 {
            return Err(PropagationError::NumericalFailure(
                "zero position magnitude".to_string(),
            ));
        }
        let r = r2.sqrt();
        let rho = rho2.sqrt();
        let sin_lat = z / r;
        let cos_lat = rho / r;

        if rho == 0.0 {
            return Ok(self.pole_acceleration_km_s2(r, sin_lat));
        }

        let legendre = normalized_legendre(self.max_degree, self.max_order, sin_lat, cos_lat)?;
        let (cos_m, sin_m) = longitude_trig(self.max_order, x, y, rho)?;

        let mut radial_accel = 0.0;
        let mut lat_accel = 0.0;
        let mut lon_accel = 0.0;
        let mut radius_power = 1.0;
        let mu_over_r2 = self.mu_km3_s2 / r2;

        for degree in 1..=self.max_degree {
            radius_power *= self.reference_radius_km / r;
            if degree < 2 {
                continue;
            }
            let factor = mu_over_r2 * radius_power;
            let order_limit = degree.min(self.max_order);
            let mut w = 0.0;
            let mut d_w_d_s = 0.0;
            let mut d_w_d_lambda = 0.0;

            for order in 0..=order_limit {
                let coefficient = self.coefficients[coefficient_index(degree, order)];
                if coefficient.c == 0.0 && coefficient.s == 0.0 {
                    continue;
                }
                let idx = coefficient_index(degree, order);
                let trig = coefficient.c * cos_m[usize::from(order)]
                    + coefficient.s * sin_m[usize::from(order)];
                w += legendre.p[idx] * trig;
                d_w_d_s += legendre.dp_ds[idx] * trig;
                if order > 0 {
                    let order_f64 = f64::from(order);
                    d_w_d_lambda += legendre.p[idx]
                        * order_f64
                        * (coefficient.s * cos_m[usize::from(order)]
                            - coefficient.c * sin_m[usize::from(order)]);
                }
            }

            radial_accel -= f64::from(degree + 1) * factor * w;
            lat_accel += factor * cos_lat * d_w_d_s;
            if d_w_d_lambda != 0.0 {
                if cos_lat == 0.0 {
                    return Err(PropagationError::NumericalFailure(
                        "longitude derivative is singular at the pole".to_string(),
                    ));
                }
                lon_accel += factor * d_w_d_lambda / cos_lat;
            }
        }

        if rho == 0.0 {
            return Ok([0.0, 0.0, radial_accel * sin_lat]);
        }

        let er = [x / r, y / r, z / r];
        let e_lat = [-sin_lat * x / rho, -sin_lat * y / rho, cos_lat];
        let e_lon = [-y / rho, x / rho, 0.0];
        Ok([
            radial_accel * er[0] + lat_accel * e_lat[0] + lon_accel * e_lon[0],
            radial_accel * er[1] + lat_accel * e_lat[1] + lon_accel * e_lon[1],
            radial_accel * er[2] + lat_accel * e_lat[2] + lon_accel * e_lon[2],
        ])
    }

    fn pole_acceleration_km_s2(&self, r: f64, sin_lat: f64) -> [f64; 3] {
        let mut radial_accel = 0.0;
        let mut transverse_accel = [0.0_f64; 2];
        let mut radius_power = 1.0;
        let mu_over_r2 = self.mu_km3_s2 / (r * r);
        let is_south_pole = sin_lat < 0.0;

        for degree in 1..=self.max_degree {
            radius_power *= self.reference_radius_km / r;
            if degree < 2 {
                continue;
            }
            let factor = mu_over_r2 * radius_power;
            let zonal = self.coefficients[coefficient_index(degree, 0)];
            if zonal.c != 0.0 {
                let pole_sign = if is_south_pole && degree % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                let p_n0 = pole_sign * f64::from(2 * degree + 1).sqrt();
                radial_accel -= f64::from(degree + 1) * factor * p_n0 * zonal.c;
            }

            if self.max_order >= 1 {
                let tesseral = self.coefficients[coefficient_index(degree, 1)];
                if tesseral.c != 0.0 || tesseral.s != 0.0 {
                    let pole_sign = if is_south_pole && degree % 2 == 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let q_n1 = pole_sign
                        * (f64::from(degree) * f64::from(degree + 1) * f64::from(2 * degree + 1)
                            / 2.0)
                            .sqrt();
                    transverse_accel[0] += factor * q_n1 * tesseral.c;
                    transverse_accel[1] += factor * q_n1 * tesseral.s;
                }
            }
        }

        [
            transverse_accel[0],
            transverse_accel[1],
            radial_accel * sin_lat,
        ]
    }
}

impl ForceModel for SphericalHarmonicGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let orientation = match ctx.body_fixed_frame_provider() {
            Some(provider) => Some(
                provider
                    .orientation_at_tdb_seconds(state.epoch_tdb_seconds)
                    .map_err(|error| {
                        PropagationError::ForceModelFailure(format!(
                            "body-fixed frame evaluation failed: {error}"
                        ))
                    })?,
            ),
            None => {
                if self.has_non_zonal_terms {
                    return Err(PropagationError::InvalidInput(
                        "non-zonal spherical-harmonic gravity requires a body-fixed frame provider"
                            .to_string(),
                    ));
                }
                None
            }
        };

        let position_body_fixed_km = match orientation {
            Some(orientation) => orientation
                .gcrf_to_itrf_position_km(state.position_array())
                .map_err(|error| {
                    PropagationError::ForceModelFailure(format!(
                        "body-fixed position rotation failed: {error}"
                    ))
                })?,
            None => state.position_array(),
        };
        let accel_body_fixed = self.body_fixed_acceleration_km_s2(position_body_fixed_km)?;
        let accel_gcrf = match orientation {
            Some(orientation) => orientation
                .itrf_to_gcrf_position_km(accel_body_fixed)
                .map_err(|error| {
                    PropagationError::ForceModelFailure(format!(
                        "inertial acceleration rotation failed: {error}"
                    ))
                })?,
            None => accel_body_fixed,
        };
        Ok(Vector3::new(accel_gcrf[0], accel_gcrf[1], accel_gcrf[2]))
    }
}

struct LegendreValues {
    p: Vec<f64>,
    dp_ds: Vec<f64>,
}

fn parse_ascii_coefficients(
    table: &str,
) -> Result<Vec<SphericalHarmonicCoefficient>, PropagationError> {
    let mut coefficients = Vec::new();
    for (line_idx, line) in table.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let first = fields.next().expect("trimmed line has a first field");
        let degree_token =
            if first.eq_ignore_ascii_case("gfc") || first.eq_ignore_ascii_case("gcoef") {
                fields
                    .next()
                    .ok_or_else(|| table_error(line_idx, "missing degree"))?
            } else {
                first
            };
        let order_token = fields
            .next()
            .ok_or_else(|| table_error(line_idx, "missing order"))?;
        let c_token = fields
            .next()
            .ok_or_else(|| table_error(line_idx, "missing Cnm"))?;
        let s_token = fields
            .next()
            .ok_or_else(|| table_error(line_idx, "missing Snm"))?;

        let degree = parse_u16_token(degree_token, line_idx, "degree")?;
        let order = parse_u16_token(order_token, line_idx, "order")?;
        let c = parse_f64_token(c_token, line_idx, "Cnm")?;
        let s = parse_f64_token(s_token, line_idx, "Snm")?;
        coefficients.push(SphericalHarmonicCoefficient {
            degree,
            order,
            c,
            s,
        });
    }
    Ok(coefficients)
}

fn normalized_legendre(
    max_degree: u16,
    max_order: u16,
    sin_lat: f64,
    cos_lat: f64,
) -> Result<LegendreValues, PropagationError> {
    let len = triangular_len(max_degree);
    let mut p = vec![0.0; len];
    let mut dp_ds = vec![0.0; len];
    p[0] = 1.0;

    if cos_lat == 0.0 && max_order > 0 {
        return Err(PropagationError::NumericalFailure(
            "associated Legendre derivative is singular at the pole".to_string(),
        ));
    }
    let du_ds = if max_order > 0 {
        -sin_lat / cos_lat
    } else {
        0.0
    };

    initialize_low_degree_legendre(
        max_degree, max_order, sin_lat, cos_lat, du_ds, &mut p, &mut dp_ds,
    );

    for degree in 4..=max_degree {
        let diag_order = degree;
        if diag_order <= max_order {
            let coeff = ((f64::from(2 * degree + 1)) / f64::from(2 * degree)).sqrt();
            let idx = coefficient_index(degree, diag_order);
            let prev = coefficient_index(degree - 1, diag_order - 1);
            p[idx] = coeff * cos_lat * p[prev];
            dp_ds[idx] = coeff * (du_ds * p[prev] + cos_lat * dp_ds[prev]);
        }

        let order_limit = (degree - 1).min(max_order);
        for order in 0..=order_limit {
            let idx = coefficient_index(degree, order);
            if degree == order + 1 {
                let coeff = f64::from(2 * degree + 1).sqrt();
                let prev = coefficient_index(degree - 1, order);
                p[idx] = coeff * sin_lat * p[prev];
                dp_ds[idx] = coeff * (p[prev] + sin_lat * dp_ds[prev]);
            } else {
                let a = ((f64::from(2 * degree + 1) * f64::from(2 * degree - 1))
                    / (f64::from(degree - order) * f64::from(degree + order)))
                .sqrt();
                let b = ((f64::from(2 * degree + 1)
                    * f64::from(degree + order - 1)
                    * f64::from(degree - order - 1))
                    / (f64::from(2 * degree - 3)
                        * f64::from(degree - order)
                        * f64::from(degree + order)))
                .sqrt();
                let prev1 = coefficient_index(degree - 1, order);
                let prev2 = coefficient_index(degree - 2, order);
                p[idx] = a * sin_lat * p[prev1] - b * p[prev2];
                dp_ds[idx] = a * (p[prev1] + sin_lat * dp_ds[prev1]) - b * dp_ds[prev2];
            }
        }
    }

    Ok(LegendreValues { p, dp_ds })
}

fn initialize_low_degree_legendre(
    max_degree: u16,
    max_order: u16,
    sin_lat: f64,
    cos_lat: f64,
    du_ds: f64,
    p: &mut [f64],
    dp_ds: &mut [f64],
) {
    let sqrt3 = 3.0_f64.sqrt();
    if max_degree >= 1 {
        let idx = coefficient_index(1, 0);
        p[idx] = sqrt3 * sin_lat;
        dp_ds[idx] = sqrt3;
        if max_order >= 1 {
            let idx = coefficient_index(1, 1);
            p[idx] = sqrt3 * cos_lat;
            dp_ds[idx] = sqrt3 * du_ds;
        }
    }

    let sqrt5 = 5.0_f64.sqrt();
    let sqrt15 = 15.0_f64.sqrt();
    if max_degree >= 2 {
        let idx = coefficient_index(2, 0);
        p[idx] = 0.5 * sqrt5 * (3.0 * sin_lat * sin_lat - 1.0);
        dp_ds[idx] = 3.0 * sqrt5 * sin_lat;
        if max_order >= 1 {
            let idx = coefficient_index(2, 1);
            p[idx] = sqrt15 * sin_lat * cos_lat;
            dp_ds[idx] = sqrt15 * (cos_lat + sin_lat * du_ds);
        }
        if max_order >= 2 {
            let idx = coefficient_index(2, 2);
            p[idx] = 0.5 * sqrt15 * cos_lat * cos_lat;
            dp_ds[idx] = sqrt15 * cos_lat * du_ds;
        }
    }

    if max_degree >= 3 {
        let sqrt7 = 7.0_f64.sqrt();
        let sqrt42 = 42.0_f64.sqrt();
        let sqrt70 = 70.0_f64.sqrt();
        let sqrt105 = 105.0_f64.sqrt();
        let sin2 = sin_lat * sin_lat;
        let cos2 = cos_lat * cos_lat;

        let idx = coefficient_index(3, 0);
        p[idx] = 0.5 * sqrt7 * (5.0 * sin_lat * sin2 - 3.0 * sin_lat);
        dp_ds[idx] = 0.5 * sqrt7 * (15.0 * sin2 - 3.0);
        if max_order >= 1 {
            let idx = coefficient_index(3, 1);
            let term = 5.0 * sin2 - 1.0;
            p[idx] = 0.25 * sqrt42 * cos_lat * term;
            dp_ds[idx] = 0.25 * sqrt42 * (du_ds * term + cos_lat * 10.0 * sin_lat);
        }
        if max_order >= 2 {
            let idx = coefficient_index(3, 2);
            p[idx] = 0.5 * sqrt105 * sin_lat * cos2;
            dp_ds[idx] = 0.5 * sqrt105 * (cos2 + 2.0 * sin_lat * cos_lat * du_ds);
        }
        if max_order >= 3 {
            let idx = coefficient_index(3, 3);
            p[idx] = 0.25 * sqrt70 * cos_lat * cos2;
            dp_ds[idx] = 0.75 * sqrt70 * cos2 * du_ds;
        }
    }
}

fn longitude_trig(
    max_order: u16,
    x: f64,
    y: f64,
    rho: f64,
) -> Result<(Vec<f64>, Vec<f64>), PropagationError> {
    let len = usize::from(max_order) + 1;
    let mut cos_m = vec![0.0; len];
    let mut sin_m = vec![0.0; len];
    cos_m[0] = 1.0;
    if max_order == 0 {
        return Ok((cos_m, sin_m));
    }
    if rho == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "longitude is undefined at the pole".to_string(),
        ));
    }
    let cos_1 = x / rho;
    let sin_1 = y / rho;
    cos_m[1] = cos_1;
    sin_m[1] = sin_1;
    for order in 2..=usize::from(max_order) {
        cos_m[order] = cos_m[order - 1] * cos_1 - sin_m[order - 1] * sin_1;
        sin_m[order] = sin_m[order - 1] * cos_1 + cos_m[order - 1] * sin_1;
    }
    Ok((cos_m, sin_m))
}

fn validate_coefficient(coefficient: SphericalHarmonicCoefficient) -> Result<(), PropagationError> {
    if coefficient.order > coefficient.degree {
        return Err(PropagationError::InvalidInput(format!(
            "coefficient order {} exceeds degree {}",
            coefficient.order, coefficient.degree
        )));
    }
    validate_finite(coefficient.c, "coefficient.c")?;
    validate_finite(coefficient.s, "coefficient.s")
}

fn validate_degree_order(max_degree: u16, max_order: u16) -> Result<(), PropagationError> {
    if max_degree < 2 {
        return Err(PropagationError::InvalidInput(
            "max_degree must be at least 2".to_string(),
        ));
    }
    if max_order > max_degree {
        return Err(PropagationError::InvalidInput(
            "max_order must not exceed max_degree".to_string(),
        ));
    }
    Ok(())
}

fn validate_finite_positive(value: f64, field: &'static str) -> Result<(), PropagationError> {
    validate_finite(value, field)?;
    if value <= 0.0 {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be positive"
        )));
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), PropagationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PropagationError::InvalidInput(format!(
            "{field} must be finite"
        )))
    }
}

fn validate_vec3(value: [f64; 3], field: &'static str) -> Result<(), PropagationError> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(())
    } else {
        Err(PropagationError::InvalidInput(format!(
            "{field} components must be finite"
        )))
    }
}

fn parse_u16_token(
    token: &str,
    line_idx: usize,
    field: &'static str,
) -> Result<u16, PropagationError> {
    token
        .parse::<u16>()
        .map_err(|_| table_error(line_idx, field))
}

fn parse_f64_token(
    token: &str,
    line_idx: usize,
    field: &'static str,
) -> Result<f64, PropagationError> {
    let normalized;
    let parsed = if token.contains('D') || token.contains('d') {
        normalized = token.replace(['D', 'd'], "E");
        normalized.parse::<f64>()
    } else {
        token.parse::<f64>()
    };
    parsed.map_err(|_| table_error(line_idx, field))
}

fn table_error(line_idx: usize, reason: &'static str) -> PropagationError {
    PropagationError::InvalidInput(format!("coefficient table line {}: {reason}", line_idx + 1))
}

fn triangular_len(max_degree: u16) -> usize {
    let n = usize::from(max_degree);
    (n + 1) * (n + 2) / 2
}

fn coefficient_index(degree: u16, order: u16) -> usize {
    let degree = usize::from(degree);
    degree * (degree + 1) / 2 + usize::from(order)
}

#[cfg(test)]
mod tests {
    //! Clean-room validation anchors:
    //!
    //! The coefficient pins use the public EGM96 table published through the
    //! NASA GSFC and NIMA model archive. The Legendre values are hard-coded
    //! f64 values independently evaluated from the fully normalized low-degree
    //! polynomial definitions in Montenbruck and Gill section 3.2.4. The
    //! degree-2 tensor reference differentiates the closed-form Cartesian
    //! quadratic potential corresponding to Vallado section 8.7.

    use super::*;
    use crate::astro::constants::{
        J2_EARTH, J3_EARTH, J4_EARTH, J5_EARTH, J6_EARTH, MU_EARTH, RE_EARTH,
    };
    use crate::astro::forces::ForceModel;
    use crate::astro::forces::{ZonalCoefficients, ZonalDegrees, ZonalGravity};
    use crate::astro::frames::TdbEarthOrientationProvider;
    use crate::astro::propagator::{ForceModelKind, IntegratorKind, IntegratorOptions};
    use crate::astro::propagator::{PropagationContext, StatePropagator};
    use crate::astro::state::CartesianState;
    use std::sync::Arc;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} diff={}",
            actual - expected
        );
    }

    fn assert_close_vec(actual: [f64; 3], expected: [f64; 3], tolerance: f64) {
        for axis in 0..3 {
            assert_close(actual[axis], expected[axis], tolerance);
        }
    }

    fn zonal_coefficients(max_degree: u16) -> Vec<SphericalHarmonicCoefficient> {
        let values = [
            (2_u16, J2_EARTH),
            (3_u16, J3_EARTH),
            (4_u16, J4_EARTH),
            (5_u16, J5_EARTH),
            (6_u16, J6_EARTH),
        ];
        values
            .into_iter()
            .filter(|(degree, _)| *degree <= max_degree)
            .map(|(degree, j)| SphericalHarmonicCoefficient {
                degree,
                order: 0,
                c: -j / f64::from(2 * degree + 1).sqrt(),
                s: 0.0,
            })
            .collect()
    }

    fn zonal_gravity(max_degree: u8) -> ZonalGravity {
        ZonalGravity::new(
            MU_EARTH,
            RE_EARTH,
            ZonalDegrees::through(max_degree).expect("zonal degree"),
            ZonalCoefficients::default(),
        )
    }

    #[test]
    fn egm96_coefficients_pin_published_values() {
        let gravity = SphericalHarmonicGravity::egm96_degree_order_36().expect("EGM96");

        let c20 = gravity.coefficient(2, 0).expect("C20");
        assert_eq!(c20.c.to_bits(), (-0.484_165_371_736e-3_f64).to_bits());
        assert_eq!(c20.s.to_bits(), 0.0_f64.to_bits());

        let c21 = gravity.coefficient(2, 1).expect("C21");
        assert_eq!(c21.c.to_bits(), (-0.186_987_635_955e-9_f64).to_bits());
        assert_eq!(c21.s.to_bits(), 0.119_528_012_031e-8_f64.to_bits());

        let c22 = gravity.coefficient(2, 2).expect("C22");
        assert_eq!(c22.c.to_bits(), 0.243_914_352_398e-5_f64.to_bits());
        assert_eq!(c22.s.to_bits(), (-0.140_016_683_654e-5_f64).to_bits());

        let c36 = gravity.coefficient(36, 36).expect("C36");
        assert_eq!(c36.c.to_bits(), 0.460_146_465_720e-8_f64.to_bits());
        assert_eq!(c36.s.to_bits(), (-0.594_245_336_314e-8_f64).to_bits());
    }

    #[test]
    fn legendre_recursion_matches_low_degree_polynomial_bits() {
        let legendre =
            normalized_legendre(3, 3, 0.3125, 0.949_917_759_598_166_5).expect("Legendre values");
        let expected = [
            (0_u16, 0_u16, 1.0_f64),
            (1, 0, 0.541_265_877_365_274_1),
            (1, 1, 1.645_305_822_636_022_9),
            (2, 0, -0.790_484_968_608_324_2),
            (2, 1, 1.149_692_394_746_987_3),
            (2, 2, 1.747_381_158_152_174_5),
            (3, 0, -1.038_341_121_224_689_8),
            (3, 1, -0.787_556_991_899_063_4),
            (3, 2, 1.444_729_996_909_586_8),
            (3, 3, 1.792_862_776_822_135_2),
        ];
        for (degree, order, expected_value) in expected {
            assert_eq!(
                legendre.p[coefficient_index(degree, order)].to_bits(),
                expected_value.to_bits(),
                "degree={degree} order={order}"
            );
        }
    }

    #[test]
    fn zonal_only_matches_landed_closed_forms() {
        let states = [
            CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [0.0, 0.0, 0.0]),
            CartesianState::new(0.0, [-4200.0, 5320.0, 2500.0], [0.0, 0.0, 0.0]),
            CartesianState::new(0.0, [1200.0, -6800.0, -1700.0], [0.0, 0.0, 0.0]),
        ];
        for max_degree in 2_u16..=6 {
            let gravity = SphericalHarmonicGravity::from_normalized_coefficients(
                MU_EARTH,
                RE_EARTH,
                max_degree,
                0,
                &zonal_coefficients(max_degree),
            )
            .expect("zonal harmonic gravity");
            let zonal = zonal_gravity(max_degree as u8);
            for state in states {
                let actual = gravity
                    .acceleration(&state, &PropagationContext::default())
                    .expect("harmonic acceleration");
                let expected = zonal
                    .acceleration(&state, &PropagationContext::default())
                    .expect("zonal acceleration");
                assert_close_vec(
                    [actual.x, actual.y, actual.z],
                    [expected.x, expected.y, expected.z],
                    2.0e-20,
                );
            }
        }
    }

    #[test]
    fn degree_two_matches_closed_form_tensor() {
        let coefficients = [
            SphericalHarmonicCoefficient {
                degree: 2,
                order: 0,
                c: -J2_EARTH / 5.0_f64.sqrt(),
                s: 0.0,
            },
            SphericalHarmonicCoefficient {
                degree: 2,
                order: 1,
                c: -1.25e-9,
                s: 2.75e-9,
            },
            SphericalHarmonicCoefficient {
                degree: 2,
                order: 2,
                c: 2.4e-6,
                s: -1.4e-6,
            },
        ];
        let gravity = SphericalHarmonicGravity::from_normalized_coefficients(
            MU_EARTH,
            RE_EARTH,
            2,
            2,
            &coefficients,
        )
        .expect("degree two gravity");
        let position = [7033.0, -1221.0, 1329.0];
        let actual = gravity
            .body_fixed_acceleration_km_s2(position)
            .expect("body acceleration");
        let expected = degree_two_closed_form(position, MU_EARTH, RE_EARTH, &coefficients);
        assert_close_vec(actual, expected, 5.0e-21);
    }

    #[test]
    fn degree_two_pole_limit_matches_closed_form_tensor() {
        let coefficients = [
            SphericalHarmonicCoefficient {
                degree: 2,
                order: 0,
                c: -J2_EARTH / 5.0_f64.sqrt(),
                s: 0.0,
            },
            SphericalHarmonicCoefficient {
                degree: 2,
                order: 1,
                c: -1.25e-9,
                s: 2.75e-9,
            },
            SphericalHarmonicCoefficient {
                degree: 2,
                order: 2,
                c: 2.4e-6,
                s: -1.4e-6,
            },
        ];
        let gravity = SphericalHarmonicGravity::from_normalized_coefficients(
            MU_EARTH,
            RE_EARTH,
            2,
            2,
            &coefficients,
        )
        .expect("degree two gravity");
        let position = [0.0, 0.0, 7000.0];
        let actual = gravity
            .body_fixed_acceleration_km_s2(position)
            .expect("pole acceleration");
        let expected = degree_two_closed_form(position, MU_EARTH, RE_EARTH, &coefficients);
        assert_close_vec(actual, expected, 5.0e-21);
    }

    #[test]
    fn ascii_loader_accepts_gfc_rows_and_truncates() {
        let table = "\
gfc 0 0 1.0 0.0 0.0 0.0
gfc 2 0 -0.484165371736D-03 0.000000000000D+00 0.0 0.0
gfc 2 1 -0.186987635955D-09 0.119528012031D-08 0.0 0.0
gfc 3 0 0.957161207093D-06 0.000000000000D+00 0.0 0.0
";
        let gravity = SphericalHarmonicGravity::from_normalized_ascii_table(
            EGM96_MU_KM3_S2,
            EGM96_REFERENCE_RADIUS_KM,
            2,
            0,
            table,
        )
        .expect("ASCII loader");
        assert!(gravity.coefficient(3, 0).is_none());
        assert_eq!(
            gravity.coefficient(2, 0).expect("C20").c.to_bits(),
            (-0.484_165_371_736e-3_f64).to_bits()
        );
        assert_eq!(gravity.max_order(), 0);
    }

    #[test]
    fn leo_n8_propagation_is_bounded_and_separates_from_j2() {
        let initial = CartesianState::new(
            0.0,
            [6_878.137, 0.0, 0.0],
            [0.0, (MU_EARTH / 6_878.137_f64).sqrt(), 0.9],
        );
        let options = IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };
        let j2_prop = StatePropagator {
            initial,
            force_model: ForceModelKind::two_body_j2(),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        };
        let n8_prop = StatePropagator {
            initial,
            force_model: ForceModelKind::earth_phase_b(8, 8, None).expect("phase B"),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        };
        let span = 900.0;
        let j2_final = j2_prop.propagate_to(span).expect("J2").final_state;
        let ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::new()));
        let n8_final = n8_prop
            .propagate_to_with_context(span, &ctx)
            .expect("N8")
            .final_state;
        let diff = (n8_final.position_km - j2_final.position_km).norm();

        assert!(n8_final.position_km.norm() > RE_EARTH + 450.0);
        assert!(diff > 1.0e-3, "diff={diff}");
        assert!(diff < 5.0, "diff={diff}");
    }

    #[test]
    fn non_zonal_force_requires_body_fixed_provider() {
        let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [1.0, 7.2, 0.5]);
        let gravity = SphericalHarmonicGravity::earth_egm96_truncated(2, 2).expect("EGM96");
        let err = gravity
            .acceleration(&state, &PropagationContext::default())
            .expect_err("non-zonal model should require a frame provider");

        match err {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains("body-fixed frame provider"), "{message}");
            }
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    #[test]
    fn existing_j2_output_is_unchanged_when_unselected() {
        let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [1.0, 7.2, 0.5]);
        let options = IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };
        let propagator = StatePropagator {
            initial: state,
            force_model: ForceModelKind::two_body_j2(),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        };
        let selected = propagator
            .propagate_to(120.0)
            .expect("selected")
            .final_state;
        assert_eq!(
            [
                selected.position_km.x.to_bits(),
                selected.position_km.y.to_bits(),
                selected.position_km.z.to_bits(),
                selected.velocity_km_s.x.to_bits(),
                selected.velocity_km_s.y.to_bits(),
                selected.velocity_km_s.z.to_bits(),
            ],
            [
                4_664_491_415_796_994_328,
                13_868_042_735_578_520_184,
                4_653_651_704_383_841_626,
                4_591_955_084_532_107_724,
                4_619_903_849_988_711_109,
                4_599_620_700_962_266_984,
            ]
        );
    }

    fn degree_two_closed_form(
        position: [f64; 3],
        mu: f64,
        radius: f64,
        coefficients: &[SphericalHarmonicCoefficient],
    ) -> [f64; 3] {
        let x = position[0];
        let y = position[1];
        let z = position[2];
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        let r5 = r2 * r2 * r;
        let r7 = r5 * r2;
        let mut c20 = 0.0;
        let mut c21 = 0.0;
        let mut s21 = 0.0;
        let mut c22 = 0.0;
        let mut s22 = 0.0;
        for coefficient in coefficients {
            match (coefficient.degree, coefficient.order) {
                (2, 0) => c20 = coefficient.c,
                (2, 1) => {
                    c21 = coefficient.c;
                    s21 = coefficient.s;
                }
                (2, 2) => {
                    c22 = coefficient.c;
                    s22 = coefficient.s;
                }
                _ => {}
            }
        }
        let sqrt5 = 5.0_f64.sqrt();
        let sqrt15 = 15.0_f64.sqrt();
        let a = 0.5 * sqrt5 * c20;
        let b = sqrt15;
        let d = 0.5 * sqrt15;
        let q = a * (2.0 * z * z - x * x - y * y)
            + b * z * (c21 * x + s21 * y)
            + d * (c22 * (x * x - y * y) + 2.0 * s22 * x * y);
        let dq = [
            -2.0 * a * x + b * c21 * z + 2.0 * d * (c22 * x + s22 * y),
            -2.0 * a * y + b * s21 * z + 2.0 * d * (-c22 * y + s22 * x),
            4.0 * a * z + b * (c21 * x + s21 * y),
        ];
        let scale = mu * radius * radius;
        [
            scale * (dq[0] / r5 - 5.0 * q * x / r7),
            scale * (dq[1] / r5 - 5.0 * q * y / r7),
            scale * (dq[2] / r5 - 5.0 * q * z / r7),
        ]
    }
}
