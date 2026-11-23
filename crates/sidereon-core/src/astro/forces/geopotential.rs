//! Spherical-harmonic geopotential perturbation force.
//!
//! The model evaluates fully normalized real coefficients in a body-fixed
//! frame, beginning at degree 2 so the central two-body term remains separate.
//! If a propagation context supplies an Earth-orientation provider, the input
//! GCRF position is rotated into ITRF and the resulting acceleration is rotated
//! back to GCRF. Non-zonal terms require that provider; zonal-only models can
//! be evaluated directly in the input frame.
//!
//! # Evaluation algorithm
//!
//! Every model, whether built from the embedded EGM96 table, from caller
//! coefficients or from an ICGEM file
//! ([`crate::astro::forces::IcgemGravityField`]), is evaluated by the method
//! of Holmes and Featherstone (2002), "A unified approach to the Clenshaw
//! summation and the recursive computation of very high degree and order
//! normalised associated Legendre functions", *Journal of Geodesy*
//! 76:279-299. For each order `m`, the functions `Pnm(t) / u^m` (with
//! `t = sin(latitude)` and `u = cos(latitude)`) are built by the forward
//! column recursion over degree from sectoral seeds scaled by `2^-930`, and
//! the sum over order is a Horner scheme in `u`. No intermediate underflows
//! at high degree near the poles, where `u^m` would, and nothing overflows
//! through degree 2700 (Holmes and Featherstone, Section 3). The gradient is
//! formed directly in Cartesian coordinates: with `dPnm/dt = k_nm Pn,m+1 / u`
//! and `u^m cos(m lambda) = Re((x + i y)^m) / r^m` every term is a polynomial
//! in `u`, so the acceleration has no `1/u` factor and holds on the polar
//! axis. Orekit's `HolmesFeatherstoneAttractionModel` uses the same recursion
//! and scaling. The acceleration
//! ([`SphericalHarmonicGravity::body_fixed_acceleration_km_s2`]) and the
//! potential ([`SphericalHarmonicGravity::body_fixed_potential_km2_s2`]) come
//! from one sum.
//!
//! A model holds its coefficient triangle and the position-independent
//! recursion factors of that one method through its maximum degree; clones
//! share the factors.

use crate::astro::constants::{MU_EARTH, RE_EARTH};
use crate::astro::error::PropagationError;
use crate::astro::forces::icgem::TimeVariableCoefficients;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use crate::astro::time::scales::tt_from_tdb_j2000_seconds;
use nalgebra::Vector3;
use std::sync::Arc;

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
/// fully normalized real convention for `Pnm(sin latitude)`. The values are
/// those of EGM96 (Lemoine et al., 1998, NASA/TP-1998-206861), a tide-free
/// model: the ICGEM distribution of EGM96 states `tide_system tide_free`.
pub const EGM96_DEGREE_ORDER_36: &str = include_str!("egm96_to_36.txt");

/// How a geopotential's `C20` treats the permanent tide, IERS Conventions
/// (2010) Section 1.1 and Section 6.2.2.
///
/// The degree-2 zonal tide-generating potential has a nonzero mean, the
/// permanent tide, of amplitude `H0 = -0.31460 m`. It deforms the Earth, and
/// the deformation changes `C20` by `A0 H0 k20` (Equation 6.14). A field's
/// tide system says which of these two parts its `C20` holds, which decides
/// what [`crate::astro::forces::SolidEarthTideGravity`] must remove from its
/// corrections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TideSystem {
    /// "Conventional tide free": `C20` holds neither the permanent
    /// deformation nor the permanent tide-generating potential. EGM96 and the
    /// NGA release of EGM2008 are tide-free.
    TideFree,
    /// "Zero tide": `C20` holds the permanent deformation `A0 H0 k20` but not
    /// the permanent tide-generating potential. The IERS Conventions (2010)
    /// Table 6.2 values are zero-tide.
    ZeroTide,
    /// "Mean tide": `C20` holds the permanent deformation and the permanent
    /// tide-generating potential `A0 H0` as well. That second part varies as
    /// `r^2` about the geocentre, not as the `r^-3` of a `C20` term, so it is
    /// right in `C20` only at the reference radius.
    MeanTide,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CoefficientPair {
    pub(crate) c: f64,
    pub(crate) s: f64,
}

impl CoefficientPair {
    pub(crate) const ZERO: Self = Self { c: 0.0, s: 0.0 };
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
    /// Tide system of the coefficients this selector builds: every selector
    /// reads the embedded EGM96 table, which is tide-free.
    pub fn tide_system(&self) -> TideSystem {
        TideSystem::TideFree
    }

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
            self.tide_system(),
        )
    }
}

/// Fully normalized spherical-harmonic gravity perturbation.
///
/// The force is perturbation-only: degree 0 and degree 1 rows are ignored, and
/// acceleration sums begin at degree 2.
///
/// A model read from an ICGEM file with time-variable coefficients evaluates
/// them at the epoch of each state it is given; see
/// [`SphericalHarmonicGravity::body_fixed_acceleration_at_km_s2`].
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalHarmonicGravity {
    mu_km3_s2: f64,
    reference_radius_km: f64,
    max_degree: u16,
    max_order: u16,
    coefficients: Vec<CoefficientPair>,
    has_non_zonal_terms: bool,
    tide_system: TideSystem,
    holmes_featherstone: Arc<HolmesFeatherstoneTables>,
    time_variable: Option<Arc<TimeVariableCoefficients>>,
}

/// Position-independent factors of the Holmes and Featherstone (2002)
/// recursion through one maximum degree.
#[derive(Debug, PartialEq)]
struct HolmesFeatherstoneTables {
    /// `[a_nm, b_nm, k_nm]` at [`coefficient_index`]`(n, m)`: the column
    /// recursion `Q_nm = a_nm t Q_n-1,m - b_nm Q_n-2,m` (Holmes and
    /// Featherstone 2002, Equations 11 and 12) and the derivative factor
    /// `dQ_nm/dt = k_nm Q_n,m+1`.
    recursion: Vec<[f64; 3]>,
    /// Sectoral seeds `Q_mm = Pmm / u^m`, scaled by `2^-930`, for
    /// `m = 0..=max_degree` (Equation 13).
    sectoral: Vec<f64>,
}

/// Binary exponent of the Holmes and Featherstone sectoral scaling. Orekit
/// uses the same value; `2^-930` is about `1e-280`, the factor of Holmes and
/// Featherstone (2002), Section 3.
const HOLMES_FEATHERSTONE_SCALING_EXPONENT: i32 = 930;

impl HolmesFeatherstoneTables {
    fn new(max_degree: u16) -> Self {
        let n_max = usize::from(max_degree);
        let mut recursion = vec![[0.0; 3]; triangular_len(max_degree)];
        for n in 0..=n_max {
            for m in 0..=n {
                let idx = n * (n + 1) / 2 + m;
                let mut entry = [0.0; 3];
                if n > m {
                    // Integer products stay below 2^53 through degree 65535,
                    // so each factor is one rounded quotient and one sqrt.
                    let nn = n as u64;
                    let mm = m as u64;
                    entry[0] = (((2 * nn - 1) * (2 * nn + 1)) as f64
                        / ((nn - mm) * (nn + mm)) as f64)
                        .sqrt();
                    if n > m + 1 {
                        entry[1] = (((2 * nn + 1) * (nn + mm - 1) * (nn - mm - 1)) as f64
                            / ((nn - mm) * (nn + mm) * (2 * nn - 3)) as f64)
                            .sqrt();
                    }
                    // Ratio of the normalization factors of orders m and
                    // m + 1; the order-zero factor carries no 2.
                    entry[2] = if m == 0 {
                        ((nn * (nn + 1)) as f64 / 2.0).sqrt()
                    } else {
                        (((nn - mm) * (nn + mm + 1)) as f64).sqrt()
                    };
                }
                recursion[idx] = entry;
            }
        }
        let mut sectoral = vec![0.0; n_max + 1];
        sectoral[0] = f64::powi(2.0, -HOLMES_FEATHERSTONE_SCALING_EXPONENT);
        if n_max >= 1 {
            sectoral[1] = 3.0_f64.sqrt() * sectoral[0];
        }
        for m in 2..=n_max {
            sectoral[m] = ((2 * m + 1) as f64 / (2 * m) as f64).sqrt() * sectoral[m - 1];
        }
        Self {
            recursion,
            sectoral,
        }
    }

    /// Fill `column[n]` with the scaled `Q_nm` for `n = m..=n_max`.
    fn fill_column(&self, m: usize, n_max: usize, t: f64, column: &mut [f64]) {
        column[m] = self.sectoral[m];
        if m < n_max {
            let idx = (m + 1) * (m + 2) / 2 + m;
            column[m + 1] = self.recursion[idx][0] * t * column[m];
            for n in (m + 2)..=n_max {
                let [a, b, _] = self.recursion[n * (n + 1) / 2 + m];
                column[n] = a * t * column[n - 1] - b * column[n - 2];
            }
        }
    }
}

/// Potential and acceleration of the degree 2 and higher terms.
struct HarmonicSum {
    potential_km2_s2: f64,
    acceleration_km_s2: [f64; 3],
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

    /// Build from caller-supplied fully normalized coefficients in the tide
    /// system `tide_system`.
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
        tide_system: TideSystem,
    ) -> Result<Self, PropagationError> {
        validate_finite_positive(mu_km3_s2, "mu_km3_s2")?;
        validate_finite_positive(reference_radius_km, "reference_radius_km")?;
        validate_degree_order(max_degree, max_order)?;

        let len = triangular_len(max_degree);
        let mut dense = vec![CoefficientPair::ZERO; len];
        let mut seen = vec![false; len];

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
        }

        Ok(Self::from_dense(
            mu_km3_s2,
            reference_radius_km,
            max_degree,
            max_order,
            dense,
            tide_system,
            None,
        ))
    }

    /// Build a model from a dense coefficient triangle through `max_degree`,
    /// with entries above `max_order` zero. The caller has validated every
    /// argument.
    pub(crate) fn from_dense(
        mu_km3_s2: f64,
        reference_radius_km: f64,
        max_degree: u16,
        max_order: u16,
        coefficients: Vec<CoefficientPair>,
        tide_system: TideSystem,
        time_variable: Option<TimeVariableCoefficients>,
    ) -> Self {
        debug_assert_eq!(coefficients.len(), triangular_len(max_degree));
        let static_non_zonal = (2..=max_degree).any(|degree| {
            (1..=degree.min(max_order)).any(|order| {
                let pair = coefficients[coefficient_index(degree, order)];
                pair.c != 0.0 || pair.s != 0.0
            })
        });
        let has_non_zonal_terms = static_non_zonal
            || time_variable
                .as_ref()
                .is_some_and(TimeVariableCoefficients::has_non_zonal_terms);
        Self {
            mu_km3_s2,
            reference_radius_km,
            max_degree,
            max_order,
            coefficients,
            has_non_zonal_terms,
            tide_system,
            holmes_featherstone: Arc::new(HolmesFeatherstoneTables::new(max_degree)),
            time_variable: time_variable.map(Arc::new),
        }
    }

    /// Build from an ASCII coefficient table in the tide system
    /// `tide_system`.
    ///
    /// Accepted data rows are either `n m Cnm Snm ...` or `gfc n m Cnm Snm ...`.
    /// Blank lines and lines beginning with `#` are ignored.
    pub fn from_normalized_ascii_table(
        mu_km3_s2: f64,
        reference_radius_km: f64,
        max_degree: u16,
        max_order: u16,
        table: &str,
        tide_system: TideSystem,
    ) -> Result<Self, PropagationError> {
        let coefficients = parse_ascii_coefficients(table)?;
        Self::from_normalized_coefficients(
            mu_km3_s2,
            reference_radius_km,
            max_degree,
            max_order,
            &coefficients,
            tide_system,
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

    /// Tide system of the coefficients, as the constructor was told.
    pub fn tide_system(&self) -> TideSystem {
        self.tide_system
    }

    /// Return the coefficient at `degree, order`, if it lies inside the model.
    ///
    /// For a time-variable model this is the static part only: zero for a
    /// coefficient the file gives by `gfct` records. Use
    /// [`Self::coefficient_at`] for its value at an epoch.
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

    /// Whether any coefficient varies with time.
    pub fn is_time_variable(&self) -> bool {
        self.time_variable.is_some()
    }

    /// Return the coefficient at `degree, order` at `epoch_tt_seconds` (TT
    /// seconds since J2000), or `None` if it lies outside the model.
    ///
    /// For a static model the epoch is validated and otherwise ignored.
    pub fn coefficient_at(
        &self,
        degree: u16,
        order: u16,
        epoch_tt_seconds: f64,
    ) -> Result<Option<SphericalHarmonicCoefficient>, PropagationError> {
        let low = self.time_variable_coefficients(epoch_tt_seconds)?;
        if degree > self.max_degree || order > degree || order > self.max_order {
            return Ok(None);
        }
        let idx = coefficient_index(degree, order);
        let pair = low.get(idx).copied().unwrap_or(self.coefficients[idx]);
        Ok(Some(SphericalHarmonicCoefficient {
            degree,
            order,
            c: pair.c,
            s: pair.s,
        }))
    }

    /// Coefficients through the highest time-variable degree evaluated at
    /// `epoch_tt_seconds`, empty for a static model.
    fn time_variable_coefficients(
        &self,
        epoch_tt_seconds: f64,
    ) -> Result<Vec<CoefficientPair>, PropagationError> {
        validate_finite(epoch_tt_seconds, "epoch_tt_seconds")?;
        match &self.time_variable {
            Some(time_variable) => time_variable
                .coefficients_at(&self.coefficients, epoch_tt_seconds)
                .map_err(|error| PropagationError::InvalidInput(error.to_string())),
            None => Ok(Vec::new()),
        }
    }

    /// Evaluate the perturbation acceleration in the body-fixed frame at
    /// `epoch_tt_seconds` (TT seconds since J2000), the epoch at which
    /// time-variable coefficients are evaluated.
    ///
    /// For a static model this equals
    /// [`Self::body_fixed_acceleration_km_s2`]. For a time-variable model an
    /// epoch outside the validity of a coefficient's `gfct` records is an
    /// [`PropagationError::InvalidInput`].
    pub fn body_fixed_acceleration_at_km_s2(
        &self,
        position_body_fixed_km: [f64; 3],
        epoch_tt_seconds: f64,
    ) -> Result<[f64; 3], PropagationError> {
        if self.time_variable.is_none() {
            validate_finite(epoch_tt_seconds, "epoch_tt_seconds")?;
            return self.body_fixed_acceleration_km_s2(position_body_fixed_km);
        }
        let low = self.time_variable_coefficients(epoch_tt_seconds)?;
        Ok(self
            .holmes_featherstone_sum(position_body_fixed_km, &low)?
            .acceleration_km_s2)
    }

    /// Perturbing potential of the degree 2 and higher terms in the
    /// body-fixed frame, km^2/s^2, positive with the convention
    /// `V = mu / r (1 + sum ...)` whose gradient is the acceleration.
    ///
    /// Evaluated by the Holmes and Featherstone (2002) method for every model.
    /// A time-variable model needs an epoch: use
    /// [`Self::body_fixed_potential_at_km2_s2`].
    pub fn body_fixed_potential_km2_s2(
        &self,
        position_body_fixed_km: [f64; 3],
    ) -> Result<f64, PropagationError> {
        self.require_static()?;
        Ok(self
            .holmes_featherstone_sum(position_body_fixed_km, &[])?
            .potential_km2_s2)
    }

    /// Perturbing potential at `epoch_tt_seconds` (TT seconds since J2000);
    /// see [`Self::body_fixed_potential_km2_s2`].
    pub fn body_fixed_potential_at_km2_s2(
        &self,
        position_body_fixed_km: [f64; 3],
        epoch_tt_seconds: f64,
    ) -> Result<f64, PropagationError> {
        let low = self.time_variable_coefficients(epoch_tt_seconds)?;
        Ok(self
            .holmes_featherstone_sum(position_body_fixed_km, &low)?
            .potential_km2_s2)
    }

    fn require_static(&self) -> Result<(), PropagationError> {
        if self.time_variable.is_some() {
            return Err(PropagationError::InvalidInput(
                "a time-variable spherical-harmonic model is evaluated at an epoch; \
                 use the _at form"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Evaluate the perturbation acceleration in the body-fixed frame.
    ///
    /// A time-variable model needs an epoch: use
    /// [`Self::body_fixed_acceleration_at_km_s2`].
    pub fn body_fixed_acceleration_km_s2(
        &self,
        position_body_fixed_km: [f64; 3],
    ) -> Result<[f64; 3], PropagationError> {
        self.require_static()?;
        Ok(self
            .holmes_featherstone_sum(position_body_fixed_km, &[])?
            .acceleration_km_s2)
    }

    /// Holmes and Featherstone (2002) sum with Cartesian gradient; see the
    /// module documentation. Coefficients at indices below `low.len()` are
    /// read from `low` instead of the static table.
    ///
    /// With `t = z/r`, `u = rho/r`, `q = a/r`, `Q_nm = Pnm/u^m` and
    /// `Phi_m = C cos(m lambda) + S sin(m lambda)`, each term
    /// `mu/r q^n u^m Q_nm Phi_m` is `mu a^n Q_nm(z/r) (C Re + S Im)(x+iy)^m /
    /// r^(n+m+1)`. Its Cartesian gradient is
    /// `mu/r^2 q^n [ -(x_j/r) ((n+m+1) Q + t Q') u^m Phi_m + e_z Q' u^m Phi_m
    /// + m Q u^(m-1) (e_x Phi'_m + e_y Psi'_m) ]` with
    /// `Phi'_m = C cos((m-1) lambda) + S sin((m-1) lambda)`,
    /// `Psi'_m = S cos((m-1) lambda) - C sin((m-1) lambda)` and
    /// `Q'_nm = k_nm Q_n,m+1`. All powers of `u` are non-negative, so the
    /// expression holds on the polar axis, where `lambda` is taken as zero.
    fn holmes_featherstone_sum(
        &self,
        position_body_fixed_km: [f64; 3],
        low: &[CoefficientPair],
    ) -> Result<HarmonicSum, PropagationError> {
        validate_vec3(position_body_fixed_km, "position_body_fixed_km")?;
        let [x, y, z] = position_body_fixed_km;
        let rho2 = x * x + y * y;
        let r2 = rho2 + z * z;
        if r2 == 0.0 {
            return Err(PropagationError::NumericalFailure(
                "zero position magnitude".to_string(),
            ));
        }
        if !r2.is_finite() {
            return Err(PropagationError::NumericalFailure(
                "position magnitude overflows".to_string(),
            ));
        }
        let r = r2.sqrt();
        let rho = rho2.sqrt();
        let t = z / r;
        let u = rho / r;
        let (cos_lambda, sin_lambda) = if rho > 0.0 {
            (x / rho, y / rho)
        } else {
            (1.0, 0.0)
        };

        let n_max = usize::from(self.max_degree);
        let m_max = usize::from(self.max_order);
        let tables = &*self.holmes_featherstone;
        let (cos_m, sin_m) = split_longitude_trig(m_max, cos_lambda, sin_lambda);
        let q_pow = split_powers(n_max, self.reference_radius_km / r);

        let mut column = vec![0.0; n_max + 1];
        let mut next_column = vec![0.0; n_max + 1];
        if m_max < n_max {
            tables.fill_column(m_max + 1, n_max, t, &mut next_column);
        }

        // Horner accumulators in u: potential, the radial-like sum shared by
        // all three axes, the z sum, and the x and y sums of order m >= 1.
        let mut h_potential = 0.0;
        let mut h_radial = 0.0;
        let mut h_z = 0.0;
        let mut h_x = 0.0;
        let mut h_y = 0.0;
        for m in (0..=m_max).rev() {
            tables.fill_column(m, n_max, t, &mut column);
            let mut sum_c = 0.0;
            let mut sum_s = 0.0;
            let mut radial_c = 0.0;
            let mut radial_s = 0.0;
            let mut dt_c = 0.0;
            let mut dt_s = 0.0;
            let order_term = (m + 1) as f64;
            for n in m.max(2)..=n_max {
                let idx = n * (n + 1) / 2 + m;
                let pair = if idx < low.len() {
                    low[idx]
                } else {
                    self.coefficients[idx]
                };
                if pair.c == 0.0 && pair.s == 0.0 {
                    continue;
                }
                let p = column[n] * q_pow[n];
                let dp = if n > m {
                    tables.recursion[idx][2] * next_column[n] * q_pow[n]
                } else {
                    0.0
                };
                let w = (n as f64 + order_term) * p + t * dp;
                sum_c += p * pair.c;
                sum_s += p * pair.s;
                radial_c += w * pair.c;
                radial_s += w * pair.s;
                dt_c += dp * pair.c;
                dt_s += dp * pair.s;
            }
            let (c_m, s_m) = (cos_m[m], sin_m[m]);
            h_potential = h_potential * u + (c_m * sum_c + s_m * sum_s);
            h_radial = h_radial * u + (c_m * radial_c + s_m * radial_s);
            h_z = h_z * u + (c_m * dt_c + s_m * dt_s);
            if m >= 1 {
                let (c_p, s_p) = (cos_m[m - 1], sin_m[m - 1]);
                let order = m as f64;
                h_x = h_x * u + order * (c_p * sum_c + s_p * sum_s);
                h_y = h_y * u + order * (c_p * sum_s - s_p * sum_c);
            }
            std::mem::swap(&mut column, &mut next_column);
        }

        let unscale = f64::powi(2.0, HOLMES_FEATHERSTONE_SCALING_EXPONENT);
        let mu_over_r = self.mu_km3_s2 / r;
        let mu_over_r2 = mu_over_r / r;
        let radial = h_radial * unscale;
        let potential_km2_s2 = mu_over_r * (h_potential * unscale);
        let acceleration_km_s2 = [
            mu_over_r2 * (h_x * unscale - x / r * radial),
            mu_over_r2 * (h_y * unscale - y / r * radial),
            mu_over_r2 * (h_z * unscale - z / r * radial),
        ];
        if !potential_km2_s2.is_finite() || acceleration_km_s2.iter().any(|a| !a.is_finite()) {
            return Err(PropagationError::NumericalFailure(format!(
                "spherical-harmonic sum through degree {} is not finite at radius {r} km \
                 (reference radius {} km)",
                self.max_degree, self.reference_radius_km
            )));
        }
        Ok(HarmonicSum {
            potential_km2_s2,
            acceleration_km_s2,
        })
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
                    .map(|orientation| ctx.record_orientation(orientation))
                    .map_err(|error| {
                        PropagationError::from_frame("body-fixed frame evaluation failed", error)
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
        let accel_body_fixed = if self.time_variable.is_some() {
            // ICGEM epochs are TT; the state epoch is TDB.
            let epoch_tt_seconds = tt_from_tdb_j2000_seconds(state.epoch_tdb_seconds);
            self.body_fixed_acceleration_at_km_s2(position_body_fixed_km, epoch_tt_seconds)?
        } else {
            self.body_fixed_acceleration_km_s2(position_body_fixed_km)?
        };
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

/// `cos(m lambda)` and `sin(m lambda)` for `m = 0..=max_order`, each formed
/// from two lower orders split near the middle so rounding grows with
/// `log2(m)` rather than `m`.
fn split_longitude_trig(max_order: usize, cos_1: f64, sin_1: f64) -> (Vec<f64>, Vec<f64>) {
    let mut cos_m = vec![0.0; max_order + 1];
    let mut sin_m = vec![0.0; max_order + 1];
    cos_m[0] = 1.0;
    if max_order >= 1 {
        cos_m[1] = cos_1;
        sin_m[1] = sin_1;
    }
    for m in 2..=max_order {
        let p = m / 2;
        let q = m - p;
        cos_m[m] = cos_m[p] * cos_m[q] - sin_m[p] * sin_m[q];
        sin_m[m] = sin_m[p] * cos_m[q] + cos_m[p] * sin_m[q];
    }
    (cos_m, sin_m)
}

/// `ratio^n` for `n = 0..=max_degree`, split like [`split_longitude_trig`].
fn split_powers(max_degree: usize, ratio: f64) -> Vec<f64> {
    let mut powers = vec![0.0; max_degree + 1];
    powers[0] = 1.0;
    if max_degree >= 1 {
        powers[1] = ratio;
    }
    for n in 2..=max_degree {
        let p = n / 2;
        powers[n] = powers[p] * powers[n - p];
    }
    powers
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
    validate_table_size(max_degree)?;
    if max_order > max_degree {
        return Err(PropagationError::InvalidInput(
            "max_order must not exceed max_degree".to_string(),
        ));
    }
    Ok(())
}

/// Refuse a degree whose recursion table, the largest per-coefficient
/// table, holds more bytes than this target can address (`isize::MAX`);
/// only a 32-bit target reaches the bound below degree 65536.
fn validate_table_size(max_degree: u16) -> Result<(), PropagationError> {
    let addressable = triangular_len(max_degree)
        .checked_mul(core::mem::size_of::<[f64; 3]>())
        .is_some_and(|bytes| isize::try_from(bytes).is_ok());
    if addressable {
        Ok(())
    } else {
        Err(PropagationError::InvalidInput(format!(
            "max_degree {max_degree} needs coefficient tables larger than this target can address"
        )))
    }
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

/// `k (k + 1) / 2`, halving the even factor first so that no intermediate
/// exceeds the result: for every `k <= 65536` it fits a 32-bit `usize`.
const fn triangle_number(k: usize) -> usize {
    if k.is_multiple_of(2) {
        (k / 2) * (k + 1)
    } else {
        k * k.div_ceil(2)
    }
}

pub(crate) fn triangular_len(max_degree: u16) -> usize {
    triangle_number(max_degree as usize + 1)
}

pub(crate) fn coefficient_index(degree: u16, order: u16) -> usize {
    triangle_number(degree as usize) + order as usize
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
    fn egm96_results_are_pinned() {
        // Holmes and Featherstone bits. Each component is within 1.3e-15
        // (8 units in the last place) of the 4.1.0 release, which summed the
        // same terms by a different recursion; the zonal-only model's bits
        // are unchanged.
        let cases: [(SphericalHarmonicGravity, [f64; 3], [u64; 3]); 6] = [
            (
                SphericalHarmonicGravity::egm96_degree_order_36().expect("EGM96"),
                [0.0, 0.0, 7000.0],
                [0x3e7625f4256e442f, 0xbe530b1ec735a2da, 0x3ef6dc4ac71d4fba],
            ),
            (
                SphericalHarmonicGravity::egm96_degree_order_36().expect("EGM96"),
                [6778.137, 0.0, 0.0],
                [0xbeea557b05203aab, 0xbe5bc6af19774773, 0x3e60a7bae13f0290],
            ),
            (
                SphericalHarmonicGravity::egm96_degree_order_36().expect("EGM96"),
                [3000.0, -6000.0, -1500.0],
                [0xbed135970c0fa20c, 0x3ee0a504f13268f3, 0x3edda3dfbe02071d],
            ),
            (
                SphericalHarmonicGravity::egm96_truncated(20, 12).expect("EGM96"),
                [3000.0, -6000.0, -1500.0],
                [0xbed129350c706049, 0x3ee0a8dc6e1378f7, 0x3eddab485b0c6140],
            ),
            (
                SphericalHarmonicGravity::earth_egm96_truncated(6, 0).expect("EGM96"),
                [0.0, 0.0, 7000.0],
                [0, 0, 0x3ef6e60251454b89],
            ),
            (
                SphericalHarmonicGravity::earth_egm96_truncated(6, 0).expect("EGM96"),
                [3000.0, -6000.0, -1500.0],
                [0xbed080caa762b83e, 0x3ee080caa762b83e, 0x3eddda192cb043ae],
            ),
        ];
        for (model, position, bits) in cases {
            let a = model
                .body_fixed_acceleration_km_s2(position)
                .expect("acceleration");
            assert_eq!(a.map(f64::to_bits), bits, "{position:?}");
        }
    }

    #[test]
    fn holmes_featherstone_tables_match_the_recursion_definitions() {
        let tables = HolmesFeatherstoneTables::new(6);
        // Holmes and Featherstone (2002) Equations 11 and 12 at n = 5, m = 2.
        let [a, b, k] = tables.recursion[coefficient_index(5, 2)];
        assert_close(a, (9.0_f64 * 11.0 / (3.0 * 7.0)).sqrt(), 1.0e-15);
        assert_close(
            b,
            (11.0_f64 * 6.0 * 2.0 / (3.0 * 7.0 * 7.0)).sqrt(),
            1.0e-15,
        );
        assert_close(k, (3.0_f64 * 8.0).sqrt(), 1.0e-15);
        // Order zero carries the sqrt(2) of the normalization.
        assert_close(
            tables.recursion[coefficient_index(3, 0)][2],
            6.0_f64.sqrt(),
            1.0e-15,
        );
        // Sectorals P_mm / u^m, unscaled: sqrt(3), sqrt(15)/2, sqrt(70)/4.
        let unscale = f64::powi(2.0, HOLMES_FEATHERSTONE_SCALING_EXPONENT);
        assert_close(tables.sectoral[1] * unscale, 3.0_f64.sqrt(), 1.0e-15);
        assert_close(tables.sectoral[2] * unscale, 15.0_f64.sqrt() / 2.0, 1.0e-15);
        assert_close(tables.sectoral[3] * unscale, 70.0_f64.sqrt() / 4.0, 1.0e-15);
    }

    #[test]
    fn holmes_featherstone_columns_match_low_degree_polynomials() {
        // Pnm(t) = u^m Q_nm(t) at t = 0.3125, u = sqrt(1 - t^2): the
        // expected values are the fully normalized low-degree polynomials of
        // Montenbruck and Gill section 3.2.4, evaluated independently.
        let t = 0.3125_f64;
        let u = 0.949_917_759_598_166_5_f64;
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
        let tables = HolmesFeatherstoneTables::new(3);
        let unscale = f64::powi(2.0, HOLMES_FEATHERSTONE_SCALING_EXPONENT);
        let mut column = [0.0; 4];
        // Largest measured 1.3 units in the last place, at degree 3.
        for (degree, order, expected_value) in expected {
            tables.fill_column(usize::from(order), 3, t, &mut column);
            let value = column[usize::from(degree)] * unscale * u.powi(i32::from(order));
            assert!(
                (value - expected_value).abs() <= 2.0 * f64::EPSILON * expected_value.abs(),
                "degree={degree} order={order} value={value} expected={expected_value}"
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
                TideSystem::TideFree,
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
            TideSystem::TideFree,
        )
        .expect("degree two gravity");
        // |a| is about 2.2e-5 km/s^2, whose unit in the last place is
        // 3.4e-21. Largest measured differences: 1.7e-21 off the axis and
        // 6.8e-21 (two units) on it, where the closed form and the sum
        // each round differently.
        for (position, tolerance) in [
            ([7033.0, -1221.0, 1329.0], 5.0e-21),
            ([0.0, 0.0, 7000.0], 1.0e-20),
            ([0.0, 0.0, -7000.0], 1.0e-20),
        ] {
            let actual = gravity
                .body_fixed_acceleration_km_s2(position)
                .expect("body acceleration");
            let expected = degree_two_closed_form(position, MU_EARTH, RE_EARTH, &coefficients);
            assert_close_vec(actual, expected, tolerance);
        }
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
            TideSystem::TideFree,
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
    fn permissive_provider_departure_is_reported_by_the_propagation() {
        // 2028-07-01 00:00 TDB is JD 2461953.5, 10408.5 days after J2000 and
        // past the UT1 table.
        let epoch = 10_408.5 * 86_400.0;
        let propagator_at = |epoch: f64| StatePropagator {
            initial: CartesianState::new(epoch, [7000.0, -1210.0, 1300.0], [1.0, 7.2, 0.5]),
            force_model: ForceModelKind::earth_phase_b(4, 4, None).expect("phase B"),
            integrator: IntegratorKind::Rk4,
            options: IntegratorOptions {
                initial_step: 30.0,
                ..IntegratorOptions::default()
            },
            drag: None,
            space_weather: None,
        };
        let propagator = propagator_at(epoch);

        let strict = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::new()));
        assert!(propagator
            .propagate_to_with_context(epoch + 120.0, &strict)
            .is_err());
        assert_eq!(strict.ut1_departure(), None);

        let permissive = PropagationContext::new().with_body_fixed_frame_provider(Arc::new(
            TdbEarthOrientationProvider::new()
                .with_validity(crate::astro::time::ValidityMode::Permissive),
        ));
        let result = propagator
            .propagate_to_with_context(epoch + 120.0, &permissive)
            .expect("permissive propagation");
        assert_eq!(
            result.ut1_degraded,
            Some(crate::astro::time::DegradeReason::AfterCoverage)
        );
        assert_eq!(permissive.ut1_departure(), result.ut1_degraded);
        let states = propagator
            .ephemeris_with_context(&[epoch + 60.0], &permissive)
            .expect("permissive ephemeris");
        assert_eq!(states.len(), 1);

        // A run that fails after a force model accepted UT1 outside the table
        // still passes the departure back to the context.
        let failing = StatePropagator {
            options: IntegratorOptions {
                initial_step: 30.0,
                max_steps: 1,
                ..IntegratorOptions::default()
            },
            ..propagator_at(epoch)
        };
        let fresh_failure = PropagationContext::new().with_body_fixed_frame_provider(Arc::new(
            TdbEarthOrientationProvider::new()
                .with_validity(crate::astro::time::ValidityMode::Permissive),
        ));
        assert!(matches!(
            failing.propagate_to_with_context(epoch + 120.0, &fresh_failure),
            Err(PropagationError::MaxStepsExceeded)
        ));
        assert_eq!(
            fresh_failure.ut1_departure(),
            Some(crate::astro::time::DegradeReason::AfterCoverage)
        );

        // A strict provider's refusal is typed.
        assert!(matches!(
            propagator.propagate_to_with_context(epoch + 120.0, &strict),
            Err(PropagationError::Ut1OutsideCoverage(
                crate::astro::time::DegradeReason::AfterCoverage
            ))
        ));

        // Inside the table a permissive provider reports nothing.
        let inside = propagator_at(0.0);
        let fresh = PropagationContext::new().with_body_fixed_frame_provider(Arc::new(
            TdbEarthOrientationProvider::new()
                .with_validity(crate::astro::time::ValidityMode::Permissive),
        ));
        let inside_result = inside
            .propagate_to_with_context(120.0, &fresh)
            .expect("in-table propagation");
        assert_eq!(inside_result.ut1_degraded, None);
        assert_eq!(fresh.ut1_departure(), None);
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

    #[test]
    fn triangle_indices_hold_in_a_32_bit_usize_through_degree_65535() {
        // 65536 * 65537 / 2 = 2_147_516_416 < 2^32, though the product is not.
        assert_eq!(triangular_len(u16::MAX), 2_147_516_416);
        assert_eq!(coefficient_index(u16::MAX, u16::MAX), 2_147_516_415);
        assert_eq!(coefficient_index(u16::MAX, 0), 2_147_450_880);
        for degree in [0_u16, 1, 2, 3, 359, 360, 2190, 65_534] {
            let n = usize::from(degree);
            assert_eq!(triangular_len(degree), (n + 1) * (n + 2) / 2);
            assert_eq!(coefficient_index(degree, degree), n * (n + 1) / 2 + n);
        }
    }

    #[test]
    fn degrees_beyond_the_address_space_are_typed_errors() {
        // 2_147_516_416 entries of 24 bytes: addressable only with 64 bits.
        let result = validate_table_size(u16::MAX);
        if cfg!(target_pointer_width = "64") {
            assert!(result.is_ok());
        } else {
            assert!(matches!(result, Err(PropagationError::InvalidInput(_))));
            assert!(matches!(
                SphericalHarmonicGravity::from_normalized_coefficients(
                    MU_EARTH,
                    RE_EARTH,
                    u16::MAX,
                    0,
                    &[],
                    TideSystem::TideFree,
                ),
                Err(PropagationError::InvalidInput(_))
            ));
        }
        assert!(validate_table_size(2190).is_ok());
    }
}
