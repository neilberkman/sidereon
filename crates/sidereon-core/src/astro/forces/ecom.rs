//! Empirical CODE orbit model (ECOM) solar radiation pressure for GNSS
//! satellites: ECOM2 (Arnold et al. 2015) and ECOM1 (Beutler et al. 1994;
//! Springer, Beutler and Rothacher 1999).
//!
//! # ECOM2 (Orekit 13.1.10 port)
//!
//! [`Ecom2SolarRadiationPressure`] ports Orekit 13.1.10's
//! `org.orekit.forces.radiation.ECOM2` (`double` and
//! `FieldSpacecraftState<Gradient>`) operation for operation in Hipparchus
//! 4.0.3 `FastMath`, `Vector3D`, `FieldVector3D`, `MathArrays.linearCombination`
//! and `Gradient` arithmetic, supporting general Fourier truncation `(n_d, n_b)`
//! with named constructors for Arnold et al.'s recommended sets `D2B1`
//! (`n_d = 1, n_b = 1`) and `D4B1` (`n_d = 2, n_b = 1`):
//!
//! ```text
//! a = nu * (D(du) e_D + Y0 e_Y + B(du) e_B)
//!
//! D(du) = D0 + sum_{i=1}^{n_d} (D_{2i,c} cos(2i du) + D_{2i,s} sin(2i du))
//! Y(du) = Y0
//! B(du) = B0 + sum_{i=1}^{n_b} (B_{2i-1,c} cos((2i - 1) du)
//!                               + B_{2i-1,s} sin((2i - 1) du))
//! ```
//!
//! The angular argument is `du = u - u_s` (Arnold et al. 2015, Sect. 4.2,
//! eq. 5), formed in Orekit's order from the satellite position `satPos` (m),
//! velocity `satVel` (m/s) and geocentric Sun position `sunPos` (m):
//! `Z = satPos x satVel`, `Y = normalize(Z x sunPos)`,
//! `X = normalize(Y x Z)`, `du = FastMath.atan2(satPos . Y, satPos . X)`.
//! The DYB axes are `e_D = normalize(sunPos - satPos)`,
//! `e_Y = normalize(e_D x satPos)` and `e_B = e_D x e_Y`, combined by
//! Hipparchus's three-term compensated linear combination
//! `new Vector3D(d_u, e_D, Y0, e_Y, b_u, e_B)`.
//!
//! Coefficient order follows Orekit's `ECOM2` driver order:
//! `B0`, `B1c..B(2*n_b-1)c`, `B1s..B(2*n_b-1)s`, `D0`, `D2c..D(2*n_d)c`,
//! `D2s..D(2*n_d)s`, `Y0` (`2 * (n_d + n_b) + 3` coefficients).
//!
//! # ECOM1
//!
//! [`Ecom1SolarRadiationPressure`] implements the nine-parameter ECOM1 model
//! (Beutler et al. 1994; Springer, Beutler and Rothacher 1999; Arnold et al.
//! 2015, eq. 3):
//!
//! ```text
//! a = nu * (D(u) e_D + Y(u) e_Y + B(u) e_B)
//!
//! D(u) = D0 + Dc cos(u) + Ds sin(u)
//! Y(u) = Y0 + Yc cos(u) + Ys sin(u)
//! B(u) = B0 + Bc cos(u) + Bs sin(u)
//! ```
//!
//! ECOM1 has no Orekit counterpart. It shares the Orekit-ported shadow
//! lighting ratio `nu` (with Earth-only linear conical lighting and gradients
//! taken directly in exponent-separated `1/km` arithmetic), and evaluates every
//! ECOM1 quantity in exponent-separated arithmetic on every input in both the
//! double and forward-mode field paths, rounding to binary64 once at the output
//! (so the double acceleration equals the field value bit for bit): the DYB
//! frame axes `e_D`, `e_Y = normalize(sunPos x satPos)` via the compensated
//! scaled cross product (zero iff `sunPos` and `satPos` are exactly collinear),
//! and `e_B = e_D x e_Y`; the argument-of-latitude harmonics
//! `sin(u) = y / hypot(x, y)` and `cos(u) = x / hypot(x, y)` directly from
//! `y = r . ahead` and `x = r . node` (with `node = unit(z x (r x v))` and
//! `ahead = normal x node`, never passing `u`, `sin(u)`, or `cos(u)` through
//! binary64 intermediates; the angle `u` reported by [`ecom_geometry`] and
//! [`ecom_geometry_with_shadow`] is computed from `libm::atan2` on the rescaled
//! pair); the Fourier sums; the three-term compensated linear combination; the
//! lighting scaling; and the unit conversion to km/s^2. The harmonic pair
//! `(sin u, cos u)` is evaluated only when at least one once-per-revolution
//! coefficient is nonzero (or when computing parameter partials).
//!
//! # Units and Sun distance
//!
//! Every coefficient is an SI acceleration in m/s^2, matching Orekit's `ECOM2`
//! convention (with no a priori model, no `1e-9` scale and no Sun-distance
//! scaling `(AU / |r_sun|)^2`, as neither Arnold et al. 2015 eq. 2 nor Orekit
//! scales the acceleration). Accelerations returned through [`ForceModel`] and
//! the explicit-body methods are in km/s^2; parameter partials are in km/s^2
//! per m/s^2; state partials are in 1/s^2 (position) and 1/s (velocity).
//!
//! Through [`ForceModel`], Sun and Moon positions come from the context's
//! [DE ephemeris](PropagationContext::de_ephemeris) or the default GCRS analytic ephemeris
//! ([`sun_moon_gcrs_at_tdb_seconds`](crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds)),
//! matching [`SolarRadiationPressure`](crate::astro::forces::SolarRadiationPressure).
//!
//! # Documented departures from Orekit 13.1.10
//!
//! - **Shadow lighting factor:** Orekit 13.1.10's `ECOM2.acceleration` applies
//!   no lighting ratio—its occulting bodies feed only event detectors, so it
//!   accelerates the satellite inside Earth's umbra despite its class
//!   documentation. Here the ECOM2 vector is multiplied by the lighting ratio
//!   `nu` (`Vector3D(ratio, v)` in `f64`, `FieldVector3D.scalarMultiply` in
//!   forward-mode `Gradient`). By default both [`Ecom2SolarRadiationPressure`]
//!   and [`Ecom1SolarRadiationPressure`] use the crate's ported Orekit
//!   `SolarRadiationPressure.getLightingRatio`
//!   ([`ShadowFunction::ApparentDisc`]) with the WGS84 oblate Earth
//!   ([`EarthShadowModel::Wgs84Oblate`]) and the Moon as a second occulting
//!   body, configurable via [`Ecom2SolarRadiationPressure::with_shadow_function`],
//!   [`Ecom2SolarRadiationPressure::with_earth_shadow_model`] and
//!   [`Ecom2SolarRadiationPressure::with_moon_eclipse`] (and the matching
//!   builders on [`Ecom1SolarRadiationPressure`]). In full sunlight
//!   (`nu == 1`), the acceleration and partials match raw Orekit bit for bit.
//!   Under [`ShadowFunction::ApparentDisc`], because Orekit's `maskingRatio`
//!   has no outer `[0, 1]` clamp, grazing overshoots near
//!   `ANGULAR_MARGIN = 1e-10` carry through to `nu` and the scaled
//!   acceleration (see [`crate::astro::events::eclipse`]). Where `nu == 0`
//!   (umbra), the acceleration and all partials are exactly zero without
//!   forming the DYB frame or angle argument (Orekit throws on degenerate
//!   umbra geometries; on the exact conical antumbra axis where only the
//!   lighting gradient is undefined, `acceleration` and `parameter_partials`
//!   return zero and `acceleration_partials` returns
//!   [`PropagationError::NumericalFailure`]). Under [`ShadowFunction::Conical`]
//!   (which has no Orekit counterpart), full sunlight (`nu == 1` with zero
//!   gradient) keeps the Orekit path and its bits; outside full sunlight
//!   (penumbra and antumbra), acceleration, state partials, and parameter
//!   partials are evaluated in exponent-separated arithmetic throughout from
//!   the km inputs and rounded once at the output, so the double acceleration
//!   equals the field value part bit for bit.
//! - **Undefined frame with zero coefficients and exact collinearity
//!   predicates:** Orekit throws `ZERO_NORM` when the Sun lies along the orbit
//!   normal (`Y = Z x sunPos == 0`), when position and velocity are parallel
//!   (`Z == 0`), or when `e_D` is parallel to `satPos` (`e_Y == 0`, e.g., at
//!   beta = 0 noon); on non-axis-aligned exact noon or exact Sun-along-normal
//!   inputs, Orekit's rounded `e_D x satPos` or `Z x sunPos` can instead be
//!   nonzero rounding noise and return a finite frame where the geometry is
//!   mathematically undefined. Here exact exponent-separated predicates
//!   (`sun x sat == 0` via error-free products and exact sums for `e_Y`; and
//!   `r x v == 0`, or `r . sun == 0 && v . sun == 0` with `r x v != 0`, for
//!   `Y = Z x sun` and `delta_u`) detect exact collinearity and orthogonality
//!   on both the double and field fast paths without altering any input where
//!   both predicates are false. Where only terms whose coefficients are all
//!   zero depend on an undefined vector, the remaining terms are returned
//!   (bit-identical to Orekit's three-term linear combination with those zero
//!   terms for ECOM2). Where a nonzero coefficient or a requested parameter
//!   partial needs the undefined vector, a typed
//!   [`PropagationError::NumericalFailure`] is returned.
//! - **Overflowing vector normalization and scaled fallback:** Hipparchus
//!   4.0.3's `Vector3D.normalize` and `FieldVector3D.normalize` return a zero
//!   vector when a finite vector's squared norm overflows to `+Inf`
//!   (`1 / +Inf = 0`). Where a vector's norm value is finite and nonzero,
//!   `normalize` returns Hipparchus's exact result (including non-finite
//!   gradient entries). Where the norm value is not finite while every
//!   component value is finite, it prescales by an exact power of two. In
//!   addition, where unscaled intermediate ECOM2 frame cross products,
//!   normalizations, `atan2` derivatives, Fourier sums, linear combinations,
//!   lighting scaling, or unit conversions produce a non-finite value or
//!   derivative, or a zero norm or denominator, on finite inputs while the
//!   mathematical result is finite and defined, ECOM2 falls back to
//!   exponent-separated arithmetic evaluated directly from the km inputs (with
//!   no km-to-m input rounding, converting only the outputs, forming
//!   `e_Y = normalize(sunPos x satPos)` via the compensated scaled cross
//!   product so exact collinearity always gives zero rather than `e_D` rounding
//!   noise, recomputing flushed frame vectors from the unflushed scaled inputs,
//!   taking conical lighting gradients directly in `1/km`, and scaling
//!   apparent-disc `1/m` gradients by `1000` in `ScaledF64`), leaving every
//!   input whose unscaled Orekit evaluation is finite (and nonzero where a
//!   direction is formed, and not caught by the exact collinearity/orthogonality
//!   predicates) bit-identical to Orekit. ECOM1 is evaluated in
//!   exponent-separated arithmetic throughout on every input and rounded to
//!   binary64 once at the output.
//! - **Exact-axis annular parameter partials:** Where field lighting evaluation
//!   fails on an exact-axis annular eclipse while the double lighting ratio is
//!   defined, parameter partials use the double lighting ratio value.
//! - **Non-finite field evaluation and value-only overflow:** Where
//!   forward-mode `Gradient` evaluation produces `NaN` or an infinity in any
//!   requested partial, a typed [`PropagationError::NumericalFailure`] is
//!   returned. Where the acceleration value itself overflows binary64 while the
//!   requested partials remain finite (for example, summing many near-`f64::MAX`
//!   coefficients), `acceleration` returns
//!   [`PropagationError::NumericalFailure`] while `parameter_partials` and
//!   `acceleration_partials` return `Ok` with those finite partials.
//!
//! # References
//!
//! - Beutler G., Brockmann E., Gurtner W., Hugentobler U., Mervart L.,
//!   Rothacher M., Verdun A. (1994) Extended orbit modeling techniques at the
//!   CODE processing center of the International GPS Service for Geodynamics
//!   (IGS): theory and initial results. Manuscripta Geodaetica 19(6):367-386.
//! - Springer T.A., Beutler G., Rothacher M. (1999) A new solar radiation
//!   pressure model for GPS satellites. GPS Solutions 2(3):50-62.
//! - Arnold D., Meindl M., Beutler G., Dach R., Schaer S., Lutz S., Prange L.,
//!   Sosnica K., Mervart L., Jaeggi A. (2015) CODE's new solar radiation
//!   pressure model for GNSS orbit determination. Journal of Geodesy
//!   89(8):775-791.
//! - Orekit 13.1.10, `org.orekit.forces.radiation.ECOM2` and
//!   `org.orekit.forces.radiation.SolarRadiationPressure` (Apache-2.0).

use crate::astro::attitude::gnss::DybAxes;
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::events::eclipse::{
    ad::{
        exact_collinear3, exact_dot3_is_zero, FieldVec3, Grad, ScaledF64, ScaledFieldVec3,
        ScaledGrad, ScaledVec3, Vec3,
    },
    conical_lighting_ratio_double_scaled, conical_lighting_ratio_field_scaled,
    earth_lighting_ratio_double, earth_lighting_ratio_field, earth_moon_lighting_ratio_double,
    earth_moon_lighting_ratio_field, fast_math, EarthShadowModel, ShadowFunction,
};
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_parameter_partials, finite_partials,
    validate_position, AccelerationPartials, ForceModel, ForceParameter,
};
use crate::astro::forces::third_body::sun_moon_positions_km;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, RowVector3, Vector3};

/// Number of ECOM2 D2B1 (`n_d = 1, n_b = 1`) coefficients.
pub const ECOM2_D2B1_COEFFICIENT_COUNT: usize = 7;

/// Number of ECOM2 D4B1 (`n_d = 2, n_b = 1`) coefficients.
pub const ECOM2_D4B1_COEFFICIENT_COUNT: usize = 9;

/// Number of ECOM1 coefficients.
pub const ECOM1_COEFFICIENT_COUNT: usize = 9;

/// ECOM2 D2B1 (`n_d = 1, n_b = 1`) coefficients in Orekit driver order, each
/// an SI acceleration in m/s^2 (Arnold et al. 2015, Table 4).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ecom2D2b1Coefficients {
    /// Constant term along `e_B`.
    pub b0: f64,
    /// `cos(du)` term along `e_B`.
    pub b1c: f64,
    /// `sin(du)` term along `e_B`.
    pub b1s: f64,
    /// Constant term along `e_D`.
    pub d0: f64,
    /// `cos(2 du)` term along `e_D`.
    pub d2c: f64,
    /// `sin(2 du)` term along `e_D`.
    pub d2s: f64,
    /// Constant term along `e_Y`.
    pub y0: f64,
}

impl Ecom2D2b1Coefficients {
    /// Short coefficient labels in Orekit order.
    pub const NAMES: [&'static str; ECOM2_D2B1_COEFFICIENT_COUNT] =
        ["B0", "B1c", "B1s", "D0", "D2c", "D2s", "Y0"];

    /// Estimable [`ForceModel::parameters`] names in Orekit order.
    pub const PARAMETER_NAMES: [&'static str; ECOM2_D2B1_COEFFICIENT_COUNT] = [
        "ecom2.b0",
        "ecom2.b1c",
        "ecom2.b1s",
        "ecom2.d0",
        "ecom2.d2c",
        "ecom2.d2s",
        "ecom2.y0",
    ];

    /// The coefficients in [`Self::NAMES`] (Orekit) order.
    pub fn to_array(&self) -> [f64; ECOM2_D2B1_COEFFICIENT_COUNT] {
        [
            self.b0, self.b1c, self.b1s, self.d0, self.d2c, self.d2s, self.y0,
        ]
    }

    /// Coefficients from an array in [`Self::NAMES`] (Orekit) order.
    pub fn from_array(values: [f64; ECOM2_D2B1_COEFFICIENT_COUNT]) -> Self {
        let [b0, b1c, b1s, d0, d2c, d2s, y0] = values;
        Self {
            b0,
            b1c,
            b1s,
            d0,
            d2c,
            d2s,
            y0,
        }
    }
}

/// ECOM2 D4B1 (`n_d = 2, n_b = 1`) coefficients in Orekit driver order, each
/// an SI acceleration in m/s^2 (Arnold et al. 2015, Table 4).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ecom2D4b1Coefficients {
    /// Constant term along `e_B`.
    pub b0: f64,
    /// `cos(du)` term along `e_B`.
    pub b1c: f64,
    /// `sin(du)` term along `e_B`.
    pub b1s: f64,
    /// Constant term along `e_D`.
    pub d0: f64,
    /// `cos(2 du)` term along `e_D`.
    pub d2c: f64,
    /// `cos(4 du)` term along `e_D`.
    pub d4c: f64,
    /// `sin(2 du)` term along `e_D`.
    pub d2s: f64,
    /// `sin(4 du)` term along `e_D`.
    pub d4s: f64,
    /// Constant term along `e_Y`.
    pub y0: f64,
}

impl Ecom2D4b1Coefficients {
    /// Short coefficient labels in Orekit order.
    pub const NAMES: [&'static str; ECOM2_D4B1_COEFFICIENT_COUNT] =
        ["B0", "B1c", "B1s", "D0", "D2c", "D4c", "D2s", "D4s", "Y0"];

    /// Estimable [`ForceModel::parameters`] names in Orekit order.
    pub const PARAMETER_NAMES: [&'static str; ECOM2_D4B1_COEFFICIENT_COUNT] = [
        "ecom2.b0",
        "ecom2.b1c",
        "ecom2.b1s",
        "ecom2.d0",
        "ecom2.d2c",
        "ecom2.d4c",
        "ecom2.d2s",
        "ecom2.d4s",
        "ecom2.y0",
    ];

    /// The coefficients in [`Self::NAMES`] (Orekit) order.
    pub fn to_array(&self) -> [f64; ECOM2_D4B1_COEFFICIENT_COUNT] {
        [
            self.b0, self.b1c, self.b1s, self.d0, self.d2c, self.d4c, self.d2s, self.d4s, self.y0,
        ]
    }

    /// Coefficients from an array in [`Self::NAMES`] (Orekit) order.
    pub fn from_array(values: [f64; ECOM2_D4B1_COEFFICIENT_COUNT]) -> Self {
        let [b0, b1c, b1s, d0, d2c, d4c, d2s, d4s, y0] = values;
        Self {
            b0,
            b1c,
            b1s,
            d0,
            d2c,
            d4c,
            d2s,
            d4s,
            y0,
        }
    }
}

/// ECOM1 nine-parameter coefficients (Beutler et al. 1994; Springer, Beutler
/// and Rothacher 1999; Arnold et al. 2015 eq. 3), each an SI acceleration in
/// m/s^2.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ecom1Coefficients {
    /// Constant term along `e_D`.
    pub d0: f64,
    /// `cos(u)` term along `e_D`.
    pub dc: f64,
    /// `sin(u)` term along `e_D`.
    pub ds: f64,
    /// Constant term along `e_Y`.
    pub y0: f64,
    /// `cos(u)` term along `e_Y`.
    pub yc: f64,
    /// `sin(u)` term along `e_Y`.
    pub ys: f64,
    /// Constant term along `e_B`.
    pub b0: f64,
    /// `cos(u)` term along `e_B`.
    pub bc: f64,
    /// `sin(u)` term along `e_B`.
    pub bs: f64,
}

impl Ecom1Coefficients {
    /// Short coefficient labels in array order.
    pub const NAMES: [&'static str; ECOM1_COEFFICIENT_COUNT] =
        ["D0", "Dc", "Ds", "Y0", "Yc", "Ys", "B0", "Bc", "Bs"];

    /// Estimable [`ForceModel::parameters`] names in array order.
    pub const PARAMETER_NAMES: [&'static str; ECOM1_COEFFICIENT_COUNT] = [
        "ecom1.d0", "ecom1.dc", "ecom1.ds", "ecom1.y0", "ecom1.yc", "ecom1.ys", "ecom1.b0",
        "ecom1.bc", "ecom1.bs",
    ];

    /// The coefficients in [`Self::NAMES`] order.
    pub fn to_array(&self) -> [f64; ECOM1_COEFFICIENT_COUNT] {
        [
            self.d0, self.dc, self.ds, self.y0, self.yc, self.ys, self.b0, self.bc, self.bs,
        ]
    }

    /// Coefficients from an array in [`Self::NAMES`] order.
    pub fn from_array(values: [f64; ECOM1_COEFFICIENT_COUNT]) -> Self {
        let [d0, dc, ds, y0, yc, ys, b0, bc, bs] = values;
        Self {
            d0,
            dc,
            ds,
            y0,
            yc,
            ys,
            b0,
            bc,
            bs,
        }
    }
}

/// The geometry both ECOM models evaluate at one satellite state and Sun (and
/// optional Moon) position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcomGeometry {
    /// The DYB axes built in Orekit's `ECOM2` order (`e_D`, `e_Y`, `e_B`).
    pub dyb: DybAxes,
    /// `du = u - u_s`, rad, in `(-pi, pi]` (Arnold et al. 2015, Sect. 4.2,
    /// eq. 5); `None` when `r x v = 0` or the Sun lies along the orbit normal.
    pub delta_u_rad: Option<f64>,
    /// Satellite argument of latitude `u`, rad, in `(-pi, pi]` (ECOM1);
    /// `None` when `r x v = 0` or the orbit lies in the frame's equatorial
    /// plane.
    pub argument_of_latitude_rad: Option<f64>,
    /// Apparent-disc lighting ratio `nu` from the ported Orekit
    /// `SolarRadiationPressure.getLightingRatio`. Nominally in `[0, 1]`,
    /// unclamped near `ANGULAR_MARGIN = 1e-10` (grazing overshoots carry
    /// through).
    pub lighting_ratio: f64,
}

/// Evaluate ECOM geometry using the apparent-disc lighting ratio for a WGS84
/// oblate Earth without lunar occultation.
///
/// `position_km`, `velocity_km_s` and `sun_position_km` are in one geocentric
/// inertial frame. Call [`ecom_geometry_with_shadow`] to include lunar
/// occultation or select [`EarthShadowModel::Spherical`].
///
/// Errors: [`PropagationError::InvalidInput`] for non-finite, zero, or
/// out-of-range inputs; [`PropagationError::NumericalFailure`] when the DYB
/// frame (`e_D` or `e_Y`) is undefined or overflows.
pub fn ecom_geometry(
    position_km: Vector3<f64>,
    velocity_km_s: Vector3<f64>,
    sun_position_km: Vector3<f64>,
) -> Result<EcomGeometry, PropagationError> {
    ecom_geometry_with_shadow(
        position_km,
        velocity_km_s,
        sun_position_km,
        None,
        EarthShadowModel::Wgs84Oblate,
    )
}

/// Evaluate ECOM geometry with an explicit [`EarthShadowModel`] and optional
/// Moon position (km) in the same geocentric inertial frame.
pub fn ecom_geometry_with_shadow(
    position_km: Vector3<f64>,
    velocity_km_s: Vector3<f64>,
    sun_position_km: Vector3<f64>,
    moon_position_km: Option<Vector3<f64>>,
    earth_shadow_model: EarthShadowModel,
) -> Result<EcomGeometry, PropagationError> {
    validate_finite_vec3("position_km", position_km)?;
    validate_finite_vec3("velocity_km_s", velocity_km_s)?;
    validate_finite_vec3("sun_position_km", sun_position_km)?;
    if let Some(moon) = moon_position_km {
        validate_finite_vec3("moon_position_km", moon)?;
    }

    let lighting_ratio = lighting_ratio_double(
        position_km,
        sun_position_km,
        moon_position_km,
        ShadowFunction::ApparentDisc,
        earth_shadow_model,
        moon_position_km.is_some(),
    )?;

    let [e_d, e_y, e_b] = dyb_frame_double_km(position_km, sun_position_km).ok_or_else(|| {
        PropagationError::NumericalFailure("ECOM DYB frame is undefined or overflows".to_string())
    })?;

    let delta_u_rad = orekit_delta_u_double_from_km(position_km, velocity_km_s, sun_position_km);
    let argument_of_latitude_rad = ecom1_u_double(as_array(position_km), as_array(velocity_km_s));

    Ok(EcomGeometry {
        dyb: DybAxes {
            d: [e_d.x, e_d.y, e_d.z],
            y: [e_y.x, e_y.y, e_y.z],
            b: [e_b.x, e_b.y, e_b.z],
        },
        delta_u_rad,
        argument_of_latitude_rad,
        lighting_ratio,
    })
}

/// ECOM2 empirical solar radiation pressure model (Arnold et al. 2015),
/// porting Orekit 13.1.10's `org.orekit.forces.radiation.ECOM2` with
/// configurable shadow scaling (defaulting to [`ShadowFunction::ApparentDisc`],
/// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled).
#[derive(Debug, Clone, PartialEq)]
pub struct Ecom2SolarRadiationPressure {
    n_d: usize,
    n_b: usize,
    coefficients: Vec<f64>,
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
}

impl Ecom2SolarRadiationPressure {
    /// Number of coefficients `2 * (n_d + n_b) + 3`, or `None` on overflow.
    pub fn coefficient_count(n_d: usize, n_b: usize) -> Option<usize> {
        n_d.checked_add(n_b)?.checked_mul(2)?.checked_add(3)
    }

    /// Estimable parameter names for truncation `(n_d, n_b)` in Orekit order:
    /// `ecom2.b0`, `ecom2.b1c..`, `ecom2.b1s..`, `ecom2.d0`, `ecom2.d2c..`,
    /// `ecom2.d2s..`, `ecom2.y0`.
    pub fn parameter_names(n_d: usize, n_b: usize) -> Result<Vec<String>, PropagationError> {
        let count = Self::coefficient_count(n_d, n_b).ok_or_else(|| {
            PropagationError::InvalidInput("ECOM2 truncation orders overflow".to_string())
        })?;
        let mut names = Vec::with_capacity(count);
        names.push("ecom2.b0".to_string());
        for i in 1..=n_b {
            names.push(format!("ecom2.b{}c", 2 * i - 1));
        }
        for i in 1..=n_b {
            names.push(format!("ecom2.b{}s", 2 * i - 1));
        }
        names.push("ecom2.d0".to_string());
        for i in 1..=n_d {
            names.push(format!("ecom2.d{}c", 2 * i));
        }
        for i in 1..=n_d {
            names.push(format!("ecom2.d{}s", 2 * i));
        }
        names.push("ecom2.y0".to_string());
        Ok(names)
    }

    /// Build an ECOM2 model with general truncation `(n_d, n_b)` and
    /// coefficients in Orekit order (`B0`, `Bcos`, `Bsin`, `D0`, `Dcos`,
    /// `Dsin`, `Y0`), each in m/s^2.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`],
    /// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled.
    pub fn new(n_d: usize, n_b: usize, coefficients: Vec<f64>) -> Result<Self, PropagationError> {
        let expected = Self::coefficient_count(n_d, n_b).ok_or_else(|| {
            PropagationError::InvalidInput("ECOM2 truncation orders overflow".to_string())
        })?;
        expect_parameter_count(&coefficients, expected)?;
        validate_finite_slice("ECOM2 coefficient", &coefficients)?;
        Ok(Self {
            n_d,
            n_b,
            coefficients,
            shadow_function: ShadowFunction::ApparentDisc,
            earth_shadow_model: EarthShadowModel::Wgs84Oblate,
            include_moon_eclipse: true,
        })
    }

    /// Build Arnold et al.'s recommended 7-parameter `D2B1` model
    /// (`n_d = 1, n_b = 1`).
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`],
    /// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled.
    pub fn d2b1(coefficients: Ecom2D2b1Coefficients) -> Result<Self, PropagationError> {
        Self::new(1, 1, coefficients.to_array().to_vec())
    }

    /// Build Arnold et al.'s recommended 9-parameter `D4B1` model
    /// (`n_d = 2, n_b = 1`).
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`],
    /// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled.
    pub fn d4b1(coefficients: Ecom2D4b1Coefficients) -> Result<Self, PropagationError> {
        Self::new(2, 1, coefficients.to_array().to_vec())
    }

    /// Select the shadow function used to compute the solar lighting ratio `nu`.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`]. Selecting
    /// [`ShadowFunction::Conical`] switches to the Earth-only linear conical
    /// lighting ratio (evaluated directly as
    /// `(rho - r_umbra) / (r_penumbra - r_umbra)` in penumbra and
    /// `rho / r_penumbra` in antumbra from the shared conical geometry) and
    /// disables lunar occultation ([`Self::moon_eclipse_enabled`] becomes
    /// `false`). Selecting [`ShadowFunction::ApparentDisc`] keeps the current
    /// [`Self::moon_eclipse_enabled`] setting.
    pub fn with_shadow_function(mut self, function: ShadowFunction) -> Self {
        self.shadow_function = function;
        if matches!(function, ShadowFunction::Conical) {
            self.include_moon_eclipse = false;
        }
        self
    }

    /// Shadow function used to compute the solar lighting ratio `nu`.
    pub fn shadow_function(&self) -> ShadowFunction {
        self.shadow_function
    }

    /// Select the Earth figure model used for shadow evaluation, in the
    /// apparent-disc occultation and in the conical shadow alike.
    ///
    /// Defaults to [`EarthShadowModel::Wgs84Oblate`].
    pub fn with_earth_shadow_model(mut self, model: EarthShadowModel) -> Self {
        self.earth_shadow_model = model;
        self
    }

    /// Earth figure model used for shadow evaluation.
    pub fn earth_shadow_model(&self) -> EarthShadowModel {
        self.earth_shadow_model
    }

    /// Enable or disable lunar occultation in the shadow factor.
    ///
    /// Enabled by default under [`ShadowFunction::ApparentDisc`]. Setting
    /// `false` disables lunar occultation while keeping the selected
    /// [`Self::shadow_function`] (so on the default model it evaluates
    /// Earth-only apparent-disc occultation). Setting `true` enables lunar
    /// occultation and, if [`ShadowFunction::Conical`] was selected, switches
    /// to [`ShadowFunction::ApparentDisc`].
    pub fn with_moon_eclipse(mut self, enabled: bool) -> Self {
        self.include_moon_eclipse = enabled;
        if enabled {
            self.shadow_function = ShadowFunction::ApparentDisc;
        }
        self
    }

    /// Whether lunar occultation is included in the lighting ratio.
    pub fn moon_eclipse_enabled(&self) -> bool {
        self.include_moon_eclipse
    }

    /// Fourier truncation rank along `e_D` (even harmonics `2, 4, ..., 2*n_d`).
    pub fn n_d(&self) -> usize {
        self.n_d
    }

    /// Fourier truncation rank along `e_B` (odd harmonics `1, 3, ..., 2*n_b - 1`).
    pub fn n_b(&self) -> usize {
        self.n_b
    }

    /// Coefficients in Orekit order, m/s^2.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Evaluate ECOM2 acceleration (km/s^2) with an explicit inertial Sun
    /// position (km), when lunar occultation is disabled (either
    /// [`ShadowFunction::ApparentDisc`] with [`Self::with_moon_eclipse`]`(false)`,
    /// or [`ShadowFunction::Conical`]).
    ///
    /// Returns [`PropagationError::InvalidInput`] when
    /// [`Self::moon_eclipse_enabled`] is `true` (the default); call
    /// [`Self::acceleration_with_sun_moon`] instead.
    pub fn acceleration_with_sun(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        if self.include_moon_eclipse {
            return Err(PropagationError::InvalidInput(
                "ECOM2 lunar eclipse is enabled; call acceleration_with_sun_moon".to_string(),
            ));
        }
        ecom2_acceleration_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            None,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            false,
        )
    }

    /// Evaluate ECOM2 acceleration (km/s^2) with explicit inertial Sun and
    /// Moon positions (km).
    ///
    /// Earth-only shadow functions ignore `moon_position_km`.
    pub fn acceleration_with_sun_moon(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
        moon_position_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        let moon = if self.include_moon_eclipse {
            validate_finite_vec3("moon_position_km", moon_position_km)?;
            Some(moon_position_km)
        } else {
            None
        };
        ecom2_acceleration_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            moon,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    /// Evaluate ECOM2 position (1/s^2) and velocity (1/s) partials with an
    /// explicit inertial Sun position (km), when lunar occultation is disabled.
    ///
    /// Returns [`PropagationError::InvalidInput`] when
    /// [`Self::moon_eclipse_enabled`] is `true` (the default); call
    /// [`Self::acceleration_partials_with_sun_moon`] instead.
    pub fn acceleration_partials_with_sun(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
    ) -> Result<AccelerationPartials, PropagationError> {
        if self.include_moon_eclipse {
            return Err(PropagationError::InvalidInput(
                "ECOM2 lunar eclipse is enabled; call acceleration_partials_with_sun_moon"
                    .to_string(),
            ));
        }
        ecom2_state_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            None,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            false,
        )
    }

    /// Evaluate ECOM2 position (1/s^2) and velocity (1/s) partials with
    /// explicit inertial Sun and Moon positions (km).
    pub fn acceleration_partials_with_sun_moon(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
        moon_position_km: Vector3<f64>,
    ) -> Result<AccelerationPartials, PropagationError> {
        let moon = if self.include_moon_eclipse {
            validate_finite_vec3("moon_position_km", moon_position_km)?;
            Some(moon_position_km)
        } else {
            None
        };
        ecom2_state_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            moon,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    /// Evaluate ECOM2 parameter partials (km/s^2 per m/s^2, in Orekit
    /// coefficient order) with an explicit inertial Sun position (km), when
    /// lunar occultation is disabled.
    ///
    /// Returns [`PropagationError::InvalidInput`] when
    /// [`Self::moon_eclipse_enabled`] is `true` (the default); call
    /// [`Self::parameter_partials_with_sun_moon`] instead.
    pub fn parameter_partials_with_sun(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        if self.include_moon_eclipse {
            return Err(PropagationError::InvalidInput(
                "ECOM2 lunar eclipse is enabled; call parameter_partials_with_sun_moon".to_string(),
            ));
        }
        ecom2_parameter_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            None,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            false,
        )
    }

    /// Evaluate ECOM2 parameter partials (km/s^2 per m/s^2, in Orekit
    /// coefficient order) with explicit inertial Sun and Moon positions (km).
    pub fn parameter_partials_with_sun_moon(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
        moon_position_km: Vector3<f64>,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        let moon = if self.include_moon_eclipse {
            validate_finite_vec3("moon_position_km", moon_position_km)?;
            Some(moon_position_km)
        } else {
            None
        };
        ecom2_parameter_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            moon,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn bodies_km(
        &self,
        epoch_tdb_seconds: f64,
        ctx: &PropagationContext,
    ) -> Result<(Vector3<f64>, Option<Vector3<f64>>), PropagationError> {
        let bodies = sun_moon_positions_km(epoch_tdb_seconds, ctx)?;
        let moon = self.include_moon_eclipse.then_some(bodies.moon_km);
        Ok((bodies.sun_km, moon))
    }

    fn validate_parameters(&self, parameters: &[f64]) -> Result<(), PropagationError> {
        expect_parameter_count(parameters, self.coefficients.len())?;
        validate_finite_slice("ECOM2 parameter", parameters)
    }
}

impl ForceModel for Ecom2SolarRadiationPressure {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom2_acceleration_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            self.n_d,
            self.n_b,
            &self.coefficients,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        let names = Self::parameter_names(self.n_d, self.n_b)
            .expect("validated truncation on construction");
        names
            .into_iter()
            .zip(&self.coefficients)
            .map(|(name, &value)| ForceParameter::new(name, value))
            .collect()
    }

    fn parameter_count(&self) -> usize {
        self.coefficients.len()
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.validate_parameters(parameters)?;
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom2_acceleration_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            self.n_d,
            self.n_b,
            parameters,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        self.validate_parameters(parameters)?;
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom2_state_partials_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            self.n_d,
            self.n_b,
            parameters,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        self.validate_parameters(parameters)?;
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom2_parameter_partials_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            self.n_d,
            self.n_b,
            parameters,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }
}

/// ECOM1 nine-parameter empirical solar radiation pressure model (Beutler et
/// al. 1994; Springer, Beutler and Rothacher 1999; Arnold et al. 2015 eq. 3),
/// with configurable shadow scaling (defaulting to
/// [`ShadowFunction::ApparentDisc`], [`EarthShadowModel::Wgs84Oblate`], and
/// lunar occultation enabled).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ecom1SolarRadiationPressure {
    coefficients: Ecom1Coefficients,
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
}

impl Ecom1SolarRadiationPressure {
    /// Build from nine ECOM1 coefficients in m/s^2.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`],
    /// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled.
    pub fn new(coefficients: Ecom1Coefficients) -> Result<Self, PropagationError> {
        validate_finite_slice("ECOM1 coefficient", &coefficients.to_array())?;
        Ok(Self {
            coefficients,
            shadow_function: ShadowFunction::ApparentDisc,
            earth_shadow_model: EarthShadowModel::Wgs84Oblate,
            include_moon_eclipse: true,
        })
    }

    /// Select the shadow function used to compute the solar lighting ratio `nu`.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`]. Selecting
    /// [`ShadowFunction::Conical`] switches to the Earth-only linear conical
    /// lighting ratio (evaluated directly as
    /// `(rho - r_umbra) / (r_penumbra - r_umbra)` in penumbra and
    /// `rho / r_penumbra` in antumbra from the shared conical geometry) and
    /// disables lunar occultation ([`Self::moon_eclipse_enabled`] becomes
    /// `false`). Selecting [`ShadowFunction::ApparentDisc`] keeps the current
    /// [`Self::moon_eclipse_enabled`] setting.
    pub fn with_shadow_function(mut self, function: ShadowFunction) -> Self {
        self.shadow_function = function;
        if matches!(function, ShadowFunction::Conical) {
            self.include_moon_eclipse = false;
        }
        self
    }

    /// Shadow function used to compute the solar lighting ratio `nu`.
    pub fn shadow_function(&self) -> ShadowFunction {
        self.shadow_function
    }

    /// Select the Earth figure model used for shadow evaluation, in the
    /// apparent-disc occultation and in the conical shadow alike.
    ///
    /// Defaults to [`EarthShadowModel::Wgs84Oblate`].
    pub fn with_earth_shadow_model(mut self, model: EarthShadowModel) -> Self {
        self.earth_shadow_model = model;
        self
    }

    /// Earth figure model used for shadow evaluation.
    pub fn earth_shadow_model(&self) -> EarthShadowModel {
        self.earth_shadow_model
    }

    /// Enable or disable lunar occultation in the shadow factor.
    ///
    /// Enabled by default under [`ShadowFunction::ApparentDisc`]. Setting
    /// `false` disables lunar occultation while keeping the selected
    /// [`Self::shadow_function`] (so on the default model it evaluates
    /// Earth-only apparent-disc occultation). Setting `true` enables lunar
    /// occultation and, if [`ShadowFunction::Conical`] was selected, switches
    /// to [`ShadowFunction::ApparentDisc`].
    pub fn with_moon_eclipse(mut self, enabled: bool) -> Self {
        self.include_moon_eclipse = enabled;
        if enabled {
            self.shadow_function = ShadowFunction::ApparentDisc;
        }
        self
    }

    /// Whether lunar occultation is included in the lighting ratio.
    pub fn moon_eclipse_enabled(&self) -> bool {
        self.include_moon_eclipse
    }

    /// Coefficients, m/s^2.
    pub fn coefficients(&self) -> Ecom1Coefficients {
        self.coefficients
    }

    /// Evaluate ECOM1 acceleration (km/s^2) with an explicit inertial Sun
    /// position (km), when lunar occultation is disabled (either
    /// [`ShadowFunction::ApparentDisc`] with [`Self::with_moon_eclipse`]`(false)`,
    /// or [`ShadowFunction::Conical`]).
    ///
    /// Returns [`PropagationError::InvalidInput`] when
    /// [`Self::moon_eclipse_enabled`] is `true` (the default); call
    /// [`Self::acceleration_with_sun_moon`] instead.
    pub fn acceleration_with_sun(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        if self.include_moon_eclipse {
            return Err(PropagationError::InvalidInput(
                "ECOM1 lunar eclipse is enabled; call acceleration_with_sun_moon".to_string(),
            ));
        }
        ecom1_acceleration_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            None,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            false,
        )
    }

    /// Evaluate ECOM1 acceleration (km/s^2) with explicit inertial Sun and
    /// Moon positions (km).
    ///
    /// Earth-only shadow functions ignore `moon_position_km`.
    pub fn acceleration_with_sun_moon(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
        moon_position_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        let moon = if self.include_moon_eclipse {
            validate_finite_vec3("moon_position_km", moon_position_km)?;
            Some(moon_position_km)
        } else {
            None
        };
        ecom1_acceleration_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            moon,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    /// Evaluate ECOM1 position (1/s^2) and velocity (1/s) partials with an
    /// explicit inertial Sun position (km), when lunar occultation is disabled.
    ///
    /// Returns [`PropagationError::InvalidInput`] when
    /// [`Self::moon_eclipse_enabled`] is `true` (the default); call
    /// [`Self::acceleration_partials_with_sun_moon`] instead.
    pub fn acceleration_partials_with_sun(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
    ) -> Result<AccelerationPartials, PropagationError> {
        if self.include_moon_eclipse {
            return Err(PropagationError::InvalidInput(
                "ECOM1 lunar eclipse is enabled; call acceleration_partials_with_sun_moon"
                    .to_string(),
            ));
        }
        ecom1_state_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            None,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            false,
        )
    }

    /// Evaluate ECOM1 position (1/s^2) and velocity (1/s) partials with
    /// explicit inertial Sun and Moon positions (km).
    pub fn acceleration_partials_with_sun_moon(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
        moon_position_km: Vector3<f64>,
    ) -> Result<AccelerationPartials, PropagationError> {
        let moon = if self.include_moon_eclipse {
            validate_finite_vec3("moon_position_km", moon_position_km)?;
            Some(moon_position_km)
        } else {
            None
        };
        ecom1_state_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            moon,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    /// Evaluate ECOM1 parameter partials (km/s^2 per m/s^2, in
    /// [`Ecom1Coefficients::NAMES`] order) with an explicit inertial Sun
    /// position (km), when lunar occultation is disabled.
    ///
    /// Returns [`PropagationError::InvalidInput`] when
    /// [`Self::moon_eclipse_enabled`] is `true` (the default); call
    /// [`Self::parameter_partials_with_sun_moon`] instead.
    pub fn parameter_partials_with_sun(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
    ) -> Result<[Vector3<f64>; ECOM1_COEFFICIENT_COUNT], PropagationError> {
        if self.include_moon_eclipse {
            return Err(PropagationError::InvalidInput(
                "ECOM1 lunar eclipse is enabled; call parameter_partials_with_sun_moon".to_string(),
            ));
        }
        ecom1_parameter_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            None,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            false,
        )
    }

    /// Evaluate ECOM1 parameter partials (km/s^2 per m/s^2, in
    /// [`Ecom1Coefficients::NAMES`] order) with explicit inertial Sun and Moon
    /// positions (km).
    pub fn parameter_partials_with_sun_moon(
        &self,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
        sun_position_km: Vector3<f64>,
        moon_position_km: Vector3<f64>,
    ) -> Result<[Vector3<f64>; ECOM1_COEFFICIENT_COUNT], PropagationError> {
        let moon = if self.include_moon_eclipse {
            validate_finite_vec3("moon_position_km", moon_position_km)?;
            Some(moon_position_km)
        } else {
            None
        };
        ecom1_parameter_partials_with_bodies(
            position_km,
            velocity_km_s,
            sun_position_km,
            moon,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn bodies_km(
        &self,
        epoch_tdb_seconds: f64,
        ctx: &PropagationContext,
    ) -> Result<(Vector3<f64>, Option<Vector3<f64>>), PropagationError> {
        let bodies = sun_moon_positions_km(epoch_tdb_seconds, ctx)?;
        let moon = self.include_moon_eclipse.then_some(bodies.moon_km);
        Ok((bodies.sun_km, moon))
    }

    fn parse_parameters(
        parameters: &[f64],
    ) -> Result<[f64; ECOM1_COEFFICIENT_COUNT], PropagationError> {
        expect_parameter_count(parameters, ECOM1_COEFFICIENT_COUNT)?;
        validate_finite_slice("ECOM1 parameter", parameters)?;
        let mut array = [0.0; ECOM1_COEFFICIENT_COUNT];
        array.copy_from_slice(parameters);
        Ok(array)
    }
}

impl ForceModel for Ecom1SolarRadiationPressure {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom1_acceleration_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            self.coefficients.to_array(),
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        Ecom1Coefficients::PARAMETER_NAMES
            .iter()
            .zip(self.coefficients.to_array())
            .map(|(&name, value)| ForceParameter::new(name, value))
            .collect()
    }

    fn parameter_count(&self) -> usize {
        ECOM1_COEFFICIENT_COUNT
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let coeffs = Self::parse_parameters(parameters)?;
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom1_acceleration_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            coeffs,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        let coeffs = Self::parse_parameters(parameters)?;
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom1_state_partials_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            coeffs,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
    }

    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        let coeffs = Self::parse_parameters(parameters)?;
        validate_position(state)?;
        let (sun_km, moon_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
        ecom1_parameter_partials_with_bodies(
            state.position_km,
            state.velocity_km_s,
            sun_km,
            moon_km,
            coeffs,
            self.shadow_function,
            self.earth_shadow_model,
            self.include_moon_eclipse,
        )
        .map(|arr| arr.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Lighting ratio helpers (apparent-disc Orekit port and linear conical shadow)
// ---------------------------------------------------------------------------

fn lighting_ratio_double(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<f64, PropagationError> {
    let (sat, sun) = (as_array(sat_pos_km), as_array(sun_pos_km));
    match (shadow_function, include_moon_eclipse) {
        (ShadowFunction::ApparentDisc, true) => {
            let moon = moon_pos_km.ok_or_else(|| {
                PropagationError::ForceModelFailure("eclipse: missing Moon position".to_string())
            })?;
            earth_moon_lighting_ratio_double(sat, sun, as_array(moon), earth_shadow_model)
                .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))
        }
        (ShadowFunction::ApparentDisc, false) => {
            earth_lighting_ratio_double(sat, sun, earth_shadow_model)
                .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))
        }
        (ShadowFunction::Conical, _) => {
            conical_lighting_ratio_double_scaled(sat, sun, earth_shadow_model)
                .map(|(nu, _)| nu.to_f64())
                .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))
        }
    }
}

fn lighting_ratio_double_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<(ScaledF64, bool), PropagationError> {
    if matches!(shadow_function, ShadowFunction::Conical) {
        let (ratio_s, is_sunlit) = conical_lighting_ratio_double_scaled(
            as_array(sat_pos_km),
            as_array(sun_pos_km),
            earth_shadow_model,
        )
        .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))?;
        if !ratio_s.is_finite() {
            return Err(PropagationError::NumericalFailure(
                "eclipse: lighting ratio is not finite".to_string(),
            ));
        }
        return Ok((ratio_s, is_sunlit));
    }
    let ratio = lighting_ratio_double(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    )?;
    if !ratio.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "eclipse: lighting ratio is not finite".to_string(),
        ));
    }
    Ok((ScaledF64::from_f64(ratio), ratio == 1.0))
}

fn lighting_ratio_field(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<Grad<3>, PropagationError> {
    let (sat, sun) = (as_array(sat_pos_km), as_array(sun_pos_km));
    let ratio = match (shadow_function, include_moon_eclipse) {
        (ShadowFunction::ApparentDisc, true) => {
            let moon = moon_pos_km.ok_or_else(|| {
                PropagationError::ForceModelFailure("eclipse: missing Moon position".to_string())
            })?;
            earth_moon_lighting_ratio_field(sat, sun, as_array(moon), earth_shadow_model)
                .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))?
        }
        (ShadowFunction::ApparentDisc, false) => {
            earth_lighting_ratio_field(sat, sun, earth_shadow_model)
                .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))?
        }
        (ShadowFunction::Conical, _) => {
            let (ratio_km, _) =
                conical_lighting_ratio_field_scaled(sat, sun, earth_shadow_model)
                    .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))?;
            if !ratio_km.is_all_finite() {
                return Err(PropagationError::NumericalFailure(
                    "eclipse: lighting ratio gradient is not finite".to_string(),
                ));
            }
            let m_per_km = ScaledF64::from_f64(M_PER_KM);
            Grad {
                value: ratio_km.val.to_f64(),
                grad: [
                    ratio_km.grad[0].div(m_per_km).to_f64(),
                    ratio_km.grad[1].div(m_per_km).to_f64(),
                    ratio_km.grad[2].div(m_per_km).to_f64(),
                ],
            }
        }
    };
    if !ratio.is_all_finite() {
        return Err(PropagationError::NumericalFailure(
            "eclipse: lighting ratio gradient is not finite".to_string(),
        ));
    }
    Ok(ratio)
}

fn lighting_ratio_field_conical_scaled_km(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    earth_shadow_model: EarthShadowModel,
) -> Result<(ScaledGrad<3>, bool), PropagationError> {
    let (ratio, is_sunlit) = conical_lighting_ratio_field_scaled(
        as_array(sat_pos_km),
        as_array(sun_pos_km),
        earth_shadow_model,
    )
    .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))?;
    if !ratio.is_all_finite() {
        return Err(PropagationError::NumericalFailure(
            "eclipse: lighting ratio gradient is not finite".to_string(),
        ));
    }
    Ok((ratio, is_sunlit))
}

#[inline]
fn is_sunlit_field(ratio: Grad<3>) -> bool {
    ratio.value == 1.0 && ratio.grad[0] == 0.0 && ratio.grad[1] == 0.0 && ratio.grad[2] == 0.0
}

enum ParameterLighting {
    Umbra,
    Lit { ratio: ScaledF64, is_sunlit: bool },
}

/// Lighting ratio value for parameter partials.
///
/// Uses the field lighting ratio when finite (keeping 0-ULP agreement with
/// Orekit's `FieldSpacecraftState` parameter derivatives), and falls back to
/// the double lighting ratio when field evaluation hits `NumericalFailure`
/// (such as an exact-axis annular eclipse where double evaluation is defined).
fn lighting_ratio_for_parameter_partials(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<ParameterLighting, PropagationError> {
    if matches!(shadow_function, ShadowFunction::Conical) {
        let (ratio_s, is_sunlit) = conical_lighting_ratio_double_scaled(
            as_array(sat_pos_km),
            as_array(sun_pos_km),
            earth_shadow_model,
        )
        .map_err(|e| PropagationError::InvalidInput(format!("eclipse: {e}")))?;
        if !ratio_s.is_finite() {
            return Err(PropagationError::NumericalFailure(
                "eclipse: lighting ratio is not finite".to_string(),
            ));
        }
        return if ratio_s.is_zero() {
            Ok(ParameterLighting::Umbra)
        } else {
            Ok(ParameterLighting::Lit {
                ratio: ratio_s,
                is_sunlit,
            })
        };
    }
    match lighting_ratio_field(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    ) {
        Ok(ratio3) => {
            if ratio3.value == 0.0 {
                Ok(ParameterLighting::Umbra)
            } else {
                Ok(ParameterLighting::Lit {
                    ratio: ScaledF64::from_f64(ratio3.value),
                    is_sunlit: is_sunlit_field(ratio3),
                })
            }
        }
        Err(PropagationError::NumericalFailure(_)) => {
            let ratio = lighting_ratio_double(
                sat_pos_km,
                sun_pos_km,
                moon_pos_km,
                shadow_function,
                earth_shadow_model,
                include_moon_eclipse,
            )?;
            if !ratio.is_finite() {
                return Err(PropagationError::NumericalFailure(
                    "eclipse: lighting ratio is not finite".to_string(),
                ));
            }
            if ratio == 0.0 {
                Ok(ParameterLighting::Umbra)
            } else {
                Ok(ParameterLighting::Lit {
                    ratio: ScaledF64::from_f64(ratio),
                    is_sunlit: ratio == 1.0,
                })
            }
        }
        Err(err) => Err(err),
    }
}

// ---------------------------------------------------------------------------
// ECOM2 double & Field<Gradient> core (Orekit 13.1.10 ECOM2.acceleration port)
// ---------------------------------------------------------------------------

#[inline]
fn dyb_e_d_double_fast(sat_pos: Vec3, sun_pos: Vec3) -> Option<Vec3> {
    let raw = sun_pos.sub(sat_pos);
    let n2 = raw.norm_sq();
    if n2 > 0.0 && n2.is_finite() {
        return raw.normalize().filter(|v| vec3_is_finite(*v));
    }
    None
}

#[inline]
fn dyb_e_y_double_fast(e_d: Vec3, sat_pos: Vec3) -> Option<Vec3> {
    let raw = e_d.cross(sat_pos);
    let n2 = raw.norm_sq();
    if n2 > 0.0 && n2.is_finite() {
        return raw.normalize().filter(|v| vec3_is_finite(*v));
    }
    None
}

#[inline]
fn scaled_e_y_double(sat_s: ScaledVec3, sun_s: ScaledVec3) -> Option<ScaledVec3> {
    sun_s.cross(sat_s).normalize_scaled().map(|(ey, _)| ey)
}

#[inline]
fn scaled_e_y_field<const N: usize>(
    sat_s: ScaledFieldVec3<N>,
    sun_s: ScaledFieldVec3<N>,
) -> Option<ScaledFieldVec3<N>> {
    sun_s.cross(sat_s).normalize()
}

#[inline]
fn is_collinear_exact(a: Vector3<f64>, b: Vector3<f64>) -> bool {
    exact_collinear3(as_array(a), as_array(b))
}

#[inline]
fn is_delta_u_undefined_exact(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
) -> bool {
    let r = as_array(sat_pos_km);
    let v = as_array(sat_vel_km_s);
    let s = as_array(sun_pos_km);
    exact_collinear3(r, v) || (exact_dot3_is_zero(r, s) && exact_dot3_is_zero(v, s))
}

#[inline]
fn dyb_frame_double_km(sat_pos_km: Vector3<f64>, sun_pos_km: Vector3<f64>) -> Option<[Vec3; 3]> {
    if !(sat_pos_km.iter().all(|c| c.is_finite()) && sun_pos_km.iter().all(|c| c.is_finite())) {
        return None;
    }
    if is_collinear_exact(sun_pos_km, sat_pos_km) {
        return None;
    }
    if let (Ok(sat_m), Ok(sun_m)) = (to_vec3_m(sat_pos_km), to_vec3_m(sun_pos_km)) {
        if let Some(e_d) = dyb_e_d_double_fast(sat_m, sun_m) {
            if let Some(e_y) = dyb_e_y_double_fast(e_d, sat_m) {
                let e_b = e_d.cross(e_y);
                if vec3_is_finite(e_b) {
                    return Some([e_d, e_y, e_b]);
                }
            }
        }
    }
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    let (e_d_s, _) = sun_s.sub(sat_s).normalize_scaled()?;
    let e_y_s = scaled_e_y_double(sat_s, sun_s)?;
    let e_b_s = e_d_s.cross(e_y_s);
    Some([e_d_s.to_vec3()?, e_y_s.to_vec3()?, e_b_s.to_vec3()?])
}

#[inline]
fn dyb_e_d_field_fast<const N: usize>(
    sat_pos: FieldVec3<N>,
    sun_pos: FieldVec3<N>,
) -> Option<FieldVec3<N>> {
    let raw = sun_pos.sub(sat_pos);
    let n2 = raw.norm_sq().value;
    if n2 > 0.0 && n2.is_finite() {
        return raw.normalize().filter(|v| field_vec3_is_all_finite(*v));
    }
    None
}

#[inline]
fn dyb_e_y_field_fast<const N: usize>(
    e_d: FieldVec3<N>,
    sat_pos: FieldVec3<N>,
) -> Option<FieldVec3<N>> {
    let raw = e_d.cross(sat_pos);
    let n2 = raw.norm_sq().value;
    if n2 > 0.0 && n2.is_finite() {
        return raw.normalize().filter(|v| field_vec3_is_all_finite(*v));
    }
    None
}

#[inline]
fn dyb_e_b_field_fast<const N: usize>(
    e_d: FieldVec3<N>,
    e_y: FieldVec3<N>,
) -> Option<FieldVec3<N>> {
    let eb = e_d.cross(e_y);
    field_vec3_is_all_finite(eb).then_some(eb)
}

#[inline]
fn scaled_atan2_double(y: ScaledF64, x: ScaledF64, atan2_fn: fn(f64, f64) -> f64) -> Option<f64> {
    if !(y.is_finite() && x.is_finite()) {
        return None;
    }
    let e_max = match (y.is_zero(), x.is_zero()) {
        (true, true) => return None,
        (true, false) => x.e,
        (false, true) => y.e,
        (false, false) => y.e.max(x.e),
    };
    let sy = if y.is_zero() {
        0.0
    } else {
        libm::ldexp(y.m, (y.e - e_max).max(-1100))
    };
    let sx = if x.is_zero() {
        0.0
    } else {
        libm::ldexp(x.m, (x.e - e_max).max(-1100))
    };
    let angle = atan2_fn(sy, sx);
    angle.is_finite().then_some(angle)
}

#[inline]
fn orekit_delta_u_double_fast(
    sat_pos: Vec3,
    sat_vel_km_s: Vector3<f64>,
    sun_pos: Vec3,
) -> Option<f64> {
    let sat_vel = to_vec3_m(sat_vel_km_s).ok()?;
    let z = sat_pos.cross(sat_vel);
    let y_raw = z.cross(sun_pos);
    let y_n2 = y_raw.norm_sq();
    if !(y_n2 > 0.0 && y_n2.is_finite()) {
        return None;
    }
    let y = y_raw.normalize().filter(|v| vec3_is_finite(*v))?;
    let x_raw = y.cross(z);
    let x_n2 = x_raw.norm_sq();
    if !(x_n2 > 0.0 && x_n2.is_finite()) {
        return None;
    }
    let x = x_raw.normalize().filter(|v| vec3_is_finite(*v))?;
    let py = sat_pos.dot(y);
    let px = sat_pos.dot(x);
    if py.is_finite() && px.is_finite() && (py != 0.0 || px != 0.0) {
        let du = fast_math::atan2(py, px);
        if du.is_finite() {
            return Some(du);
        }
    }
    None
}

#[inline]
fn orekit_delta_u_double_scaled(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    sat_s: ScaledVec3,
    vel_s: ScaledVec3,
    sun_s: ScaledVec3,
) -> Option<f64> {
    if is_delta_u_undefined_exact(sat_pos_km, sat_vel_km_s, sun_pos_km) {
        return None;
    }
    let z = sat_s.cross(vel_s);
    let (y_s, _) = z.cross(sun_s).normalize_scaled()?;
    let (x_s, _) = y_s.cross(z).normalize_scaled()?;
    scaled_atan2_double(sat_s.dot(y_s), sat_s.dot(x_s), fast_math::atan2)
}

#[inline]
fn orekit_delta_u_double_from_km(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
) -> Option<f64> {
    if !(sat_pos_km.iter().all(|c| c.is_finite())
        && sun_pos_km.iter().all(|c| c.is_finite())
        && sat_vel_km_s.iter().all(|c| c.is_finite()))
    {
        return None;
    }
    if is_delta_u_undefined_exact(sat_pos_km, sat_vel_km_s, sun_pos_km) {
        return None;
    }
    if let (Ok(sat_m), Ok(sun_m)) = (to_vec3_m(sat_pos_km), to_vec3_m(sun_pos_km)) {
        if let Some(du) = orekit_delta_u_double_fast(sat_m, sat_vel_km_s, sun_m) {
            return Some(du);
        }
    }
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let vel_s = ScaledVec3::from_vec3(Vec3::new(sat_vel_km_s.x, sat_vel_km_s.y, sat_vel_km_s.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    orekit_delta_u_double_scaled(sat_pos_km, sat_vel_km_s, sun_pos_km, sat_s, vel_s, sun_s)
}

#[inline]
fn orekit_delta_u_field_fast<const N: usize>(
    sat_pos: FieldVec3<N>,
    sat_vel_km_s: Vector3<f64>,
    sat_vel_is_var: bool,
    sun_pos: FieldVec3<N>,
) -> Option<Grad<N>> {
    let vel_ms = to_vec3_m(sat_vel_km_s).ok()?;
    let sat_vel = if sat_vel_is_var {
        FieldVec3::<N>::new(
            Grad::variable(3, vel_ms.x),
            Grad::variable(4, vel_ms.y),
            Grad::variable(5, vel_ms.z),
        )
    } else {
        FieldVec3::<N>::constant(vel_ms)
    };
    let z = sat_pos.cross(sat_vel);
    let y_raw = z.cross(sun_pos);
    let y_n2 = y_raw.norm_sq().value;
    if !(y_n2 > 0.0 && y_n2.is_finite()) {
        return None;
    }
    let y = y_raw.normalize().filter(|v| field_vec3_is_all_finite(*v))?;
    let x_raw = y.cross(z);
    let x_n2 = x_raw.norm_sq().value;
    if !(x_n2 > 0.0 && x_n2.is_finite()) {
        return None;
    }
    let x = x_raw.normalize().filter(|v| field_vec3_is_all_finite(*v))?;
    let py = sat_pos.dot(y);
    let px = sat_pos.dot(x);
    if py.is_all_finite() && px.is_all_finite() {
        return py.atan2_unscaled_with_fn(px, fast_math::atan2);
    }
    None
}

#[inline]
fn orekit_delta_u_field_scaled<const N: usize>(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    sat_s: ScaledFieldVec3<N>,
    vel_s: ScaledFieldVec3<N>,
    sun_s: ScaledFieldVec3<N>,
) -> Option<(f64, [ScaledF64; N])> {
    if is_delta_u_undefined_exact(sat_pos_km, sat_vel_km_s, sun_pos_km) {
        return None;
    }
    let z = sat_s.cross(vel_s);
    let y = z.cross(sun_s).normalize()?;
    let x = y.cross(z).normalize()?;
    sat_s.dot(y).atan2_scaled(sat_s.dot(x), fast_math::atan2)
}

#[inline]
fn ecom2_has_nonzero_periodic(n_d: usize, n_b: usize, params: &[f64]) -> bool {
    let b_periodic = n_b > 0 && params[1..=(2 * n_b)].iter().any(|&c| c != 0.0);
    let d_start = 2 * n_b + 2;
    let d_end = 2 * (n_b + n_d) + 1;
    let d_periodic = n_d > 0 && params[d_start..=d_end].iter().any(|&c| c != 0.0);
    b_periodic || d_periodic
}

#[inline]
fn ecom2_needs_y_or_b(n_d: usize, n_b: usize, params: &[f64]) -> bool {
    let y0 = params[2 * (n_d + n_b) + 2];
    y0 != 0.0 || params[0..=(2 * n_b)].iter().any(|&c| c != 0.0)
}

enum FieldAccel<const N: usize> {
    Fast(FieldVec3<N>),
    Scaled(ScaledFieldVec3<N>),
}

#[allow(clippy::too_many_arguments)]
fn ecom2_acceleration_double_scaled(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_s: ScaledF64,
    is_sunlit: bool,
    n_d: usize,
    n_b: usize,
    params: &[f64],
) -> Result<Vector3<f64>, PropagationError> {
    if !(sat_pos_km.iter().all(|c| c.is_finite())
        && sun_pos_km.iter().all(|c| c.is_finite())
        && sat_vel_km_s.iter().all(|c| c.is_finite())
        && ratio_s.is_finite())
    {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 inputs are not finite".to_string(),
        ));
    }
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let vel_s = ScaledVec3::from_vec3(Vec3::new(sat_vel_km_s.x, sat_vel_km_s.y, sat_vel_km_s.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    let (e_d, _) = sun_s.sub(sat_s).normalize_scaled().ok_or_else(|| {
        PropagationError::NumericalFailure("ECOM2 eD is undefined or overflows".to_string())
    })?;

    let need_y_or_b = ecom2_needs_y_or_b(n_d, n_b, params);
    let ey_opt = if is_collinear_exact(sun_pos_km, sat_pos_km) {
        None
    } else {
        scaled_e_y_double(sat_s, sun_s)
    };
    let (e_y, e_b) = match ey_opt {
        Some(ey) => (ey, e_d.cross(ey)),
        None if !need_y_or_b => (ScaledVec3::ZERO, ScaledVec3::ZERO),
        None => {
            return Err(PropagationError::NumericalFailure(
                "ECOM2 eY is undefined: eD is parallel to satellite position".to_string(),
            ));
        }
    };

    let need_periodic = ecom2_has_nonzero_periodic(n_d, n_b, params);
    let (b_u, d_u) = if n_d == 0 && n_b == 0 {
        (
            ScaledF64::from_f64(params[0]),
            ScaledF64::from_f64(params[1]),
        )
    } else if let Some(delta_u) =
        orekit_delta_u_double_scaled(sat_pos_km, sat_vel_km_s, sun_pos_km, sat_s, vel_s, sun_s)
    {
        let mut b_u = ScaledF64::from_f64(params[0]);
        for i in 1..=n_b {
            let (sin, cos) = fast_math::sin_cos(((2 * i - 1) as f64) * delta_u);
            b_u = b_u
                .add(ScaledF64::from_f64(cos).mul(ScaledF64::from_f64(params[i])))
                .add(ScaledF64::from_f64(sin).mul(ScaledF64::from_f64(params[i + n_b])));
        }
        let mut d_u = ScaledF64::from_f64(params[2 * n_b + 1]);
        for i in 1..=n_d {
            let (sin, cos) = fast_math::sin_cos(((2 * i) as f64) * delta_u);
            d_u = d_u
                .add(ScaledF64::from_f64(cos).mul(ScaledF64::from_f64(params[2 * n_b + 1 + i])))
                .add(
                    ScaledF64::from_f64(sin)
                        .mul(ScaledF64::from_f64(params[2 * n_b + 1 + i + n_d])),
                );
        }
        (b_u, d_u)
    } else if !need_periodic {
        (
            ScaledF64::from_f64(params[0]),
            ScaledF64::from_f64(params[2 * n_b + 1]),
        )
    } else {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 delta_u is undefined or overflows".to_string(),
        ));
    };

    let y0 = ScaledF64::from_f64(params[2 * (n_d + n_b) + 2]);
    let raw = ScaledVec3::lincomb3(d_u, e_d, y0, e_y, b_u, e_b);
    let scaled = if is_sunlit {
        raw
    } else {
        raw.scalar_mul(ratio_s)
    };
    let acc_km = scaled
        .scalar_div(ScaledF64::from_f64(M_PER_KM))
        .to_vec3()
        .ok_or_else(|| {
            PropagationError::NumericalFailure("ECOM2 acceleration is not finite".to_string())
        })?;
    finite_acceleration(Vector3::new(acc_km.x, acc_km.y, acc_km.z))
}

#[allow(clippy::too_many_arguments)]
fn ecom2_acceleration_with_bodies(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    n_d: usize,
    n_b: usize,
    params: &[f64],
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<Vector3<f64>, PropagationError> {
    validate_finite_vec3("position_km", sat_pos_km)?;
    validate_finite_vec3("sun_position_km", sun_pos_km)?;

    let (ratio_s, is_sunlit) = lighting_ratio_double_scaled(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    )?;
    if ratio_s.is_zero() {
        return Ok(Vector3::zeros());
    }

    let sat_vel_km_s = ecom_velocity(sat_vel_km_s, ecom2_has_nonzero_periodic(n_d, n_b, params))?;
    let need_y_or_b = ecom2_needs_y_or_b(n_d, n_b, params);
    let ey_collinear = is_collinear_exact(sun_pos_km, sat_pos_km);
    if ey_collinear && need_y_or_b {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 eY is undefined: eD is parallel to satellite position".to_string(),
        ));
    }
    let has_periodic_terms = n_d > 0 || n_b > 0;
    let need_periodic = ecom2_has_nonzero_periodic(n_d, n_b, params);
    let du_undefined =
        has_periodic_terms && is_delta_u_undefined_exact(sat_pos_km, sat_vel_km_s, sun_pos_km);
    if du_undefined && need_periodic {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 delta_u is undefined or overflows".to_string(),
        ));
    }

    let use_fast = !matches!(shadow_function, ShadowFunction::Conical) || is_sunlit;

    let fast_attempt = if use_fast {
        let ratio = ratio_s.to_f64();
        (|| -> Option<Vector3<f64>> {
            let sat_pos = to_vec3_m(sat_pos_km).ok()?;
            let sun_pos = to_vec3_m(sun_pos_km).ok()?;
            let e_d = dyb_e_d_double_fast(sat_pos, sun_pos)?;

            let (e_y, e_b) = if ey_collinear {
                (Vec3::ZERO, Vec3::ZERO)
            } else {
                match dyb_e_y_double_fast(e_d, sat_pos) {
                    Some(ey) => {
                        let eb = e_d.cross(ey);
                        if !vec3_is_finite(eb) {
                            return None;
                        }
                        (ey, eb)
                    }
                    None if !need_y_or_b => (Vec3::ZERO, Vec3::ZERO),
                    None => return None,
                }
            };

            let (b_u, d_u) = if !has_periodic_terms {
                (params[0], params[1])
            } else if du_undefined {
                (params[0], params[2 * n_b + 1])
            } else if let Some(delta_u) = orekit_delta_u_double_fast(sat_pos, sat_vel_km_s, sun_pos)
            {
                let mut b_u = params[0];
                for i in 1..=n_b {
                    let (sin, cos) = fast_math::sin_cos(((2 * i - 1) as f64) * delta_u);
                    b_u += params[i] * cos + params[i + n_b] * sin;
                }
                let mut d_u = params[2 * n_b + 1];
                for i in 1..=n_d {
                    let (sin, cos) = fast_math::sin_cos(((2 * i) as f64) * delta_u);
                    d_u += params[2 * n_b + 1 + i] * cos + params[2 * n_b + 1 + i + n_d] * sin;
                }
                (b_u, d_u)
            } else if !need_periodic {
                (params[0], params[2 * n_b + 1])
            } else {
                return None;
            };

            if !(b_u.is_finite() && d_u.is_finite()) {
                return None;
            }
            let y0 = params[2 * (n_d + n_b) + 2];
            let raw = Vec3::lincomb3(d_u, e_d, y0, e_y, b_u, e_b);
            let scaled = Vec3::from_scaled(ratio, raw);
            if !vec3_is_finite(scaled) {
                return None;
            }
            let acc = Vector3::new(
                scaled.x / M_PER_KM,
                scaled.y / M_PER_KM,
                scaled.z / M_PER_KM,
            );
            acc.iter().all(|c| c.is_finite()).then_some(acc)
        })()
    } else {
        None
    };

    if let Some(acc) = fast_attempt {
        return Ok(acc);
    }

    ecom2_acceleration_double_scaled(
        sat_pos_km,
        sat_vel_km_s,
        sun_pos_km,
        ratio_s,
        is_sunlit,
        n_d,
        n_b,
        params,
    )
}

#[allow(clippy::too_many_arguments)]
fn ecom2_acceleration_field_scaled<const N: usize>(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    sat_pos_is_var: bool,
    sat_vel_is_var: bool,
    ratio_s_km: ScaledGrad<N>,
    is_sunlit: bool,
    n_d: usize,
    n_b: usize,
    params: &[Grad<N>],
    params_f64: &[f64],
    require_full_frame: bool,
) -> Result<ScaledFieldVec3<N>, PropagationError> {
    if !(sat_pos_km.iter().all(|c| c.is_finite())
        && sun_pos_km.iter().all(|c| c.is_finite())
        && sat_vel_km_s.iter().all(|c| c.is_finite())
        && ratio_s_km.is_all_finite())
    {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 field inputs are not finite".to_string(),
        ));
    }
    let sat_s = if sat_pos_is_var {
        scaled_field_vec3_variable::<N>(sat_pos_km, 0)
    } else {
        scaled_field_vec3_constant::<N>(sat_pos_km)
    };
    let vel_s = if sat_vel_is_var {
        scaled_field_vec3_variable::<N>(sat_vel_km_s, 3)
    } else {
        scaled_field_vec3_constant::<N>(sat_vel_km_s)
    };
    let sun_s = scaled_field_vec3_constant::<N>(sun_pos_km);
    let e_d = sun_s.sub(sat_s).normalize().ok_or_else(|| {
        PropagationError::NumericalFailure("ECOM2 field eD is undefined or not finite".to_string())
    })?;

    let need_y_or_b = require_full_frame || ecom2_needs_y_or_b(n_d, n_b, params_f64);
    let ey_opt = if is_collinear_exact(sun_pos_km, sat_pos_km) {
        None
    } else {
        scaled_e_y_field(sat_s, sun_s)
    };
    let (e_y, e_b) = match ey_opt {
        Some(ey) => (ey, e_d.cross(ey)),
        None if !need_y_or_b => (ScaledFieldVec3::ZERO, ScaledFieldVec3::ZERO),
        None => {
            return Err(PropagationError::NumericalFailure(
                "ECOM2 field eY is undefined: eD is parallel to satellite position".to_string(),
            ));
        }
    };

    let params_s: Vec<ScaledGrad<N>> = params.iter().map(|&p| ScaledGrad::from_grad(p)).collect();
    let has_periodic_terms = n_d > 0 || n_b > 0;
    let need_periodic = (require_full_frame && has_periodic_terms)
        || ecom2_has_nonzero_periodic(n_d, n_b, params_f64);

    let (b_u, d_u) = if !has_periodic_terms {
        (params_s[0], params_s[1])
    } else {
        let du_opt =
            orekit_delta_u_field_scaled(sat_pos_km, sat_vel_km_s, sun_pos_km, sat_s, vel_s, sun_s);
        if let Some((du_val, du_grad)) = du_opt {
            let mut b_u = params_s[0];
            for i in 1..=n_b {
                let k = (2 * i - 1) as f64;
                let k_s = ScaledF64::from_f64(k);
                let mut k_grad = [ScaledF64::ZERO; N];
                for j in 0..N {
                    k_grad[j] = du_grad[j].mul(k_s);
                }
                let (sin, cos) =
                    ScaledGrad::sin_cos_from_scaled(k * du_val, k_grad, fast_math::sin_cos);
                b_u = b_u
                    .add(cos.mul(params_s[i]))
                    .add(sin.mul(params_s[i + n_b]));
            }
            let mut d_u = params_s[2 * n_b + 1];
            for i in 1..=n_d {
                let k = (2 * i) as f64;
                let k_s = ScaledF64::from_f64(k);
                let mut k_grad = [ScaledF64::ZERO; N];
                for j in 0..N {
                    k_grad[j] = du_grad[j].mul(k_s);
                }
                let (sin, cos) =
                    ScaledGrad::sin_cos_from_scaled(k * du_val, k_grad, fast_math::sin_cos);
                d_u = d_u
                    .add(cos.mul(params_s[2 * n_b + 1 + i]))
                    .add(sin.mul(params_s[2 * n_b + 1 + i + n_d]));
            }
            (b_u, d_u)
        } else if !need_periodic {
            (params_s[0], params_s[2 * n_b + 1])
        } else {
            return Err(PropagationError::NumericalFailure(
                "ECOM2 field delta_u is undefined or overflows".to_string(),
            ));
        }
    };

    let y0 = params_s[2 * (n_d + n_b) + 2];
    let raw = ScaledFieldVec3::lincomb3(d_u, e_d, y0, e_y, b_u, e_b);
    let accel_m_s = if is_sunlit {
        raw
    } else {
        raw.scalar_mul(ratio_s_km)
    };
    Ok(accel_m_s.scalar_div(ScaledF64::from_f64(M_PER_KM)))
}

#[allow(clippy::too_many_arguments)]
fn ecom2_acceleration_field<const N: usize>(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    sat_pos_is_var: bool,
    sat_vel_is_var: bool,
    ratio_fast: Grad<N>,
    ratio_s_km: ScaledGrad<N>,
    is_sunlit: bool,
    n_d: usize,
    n_b: usize,
    params: &[Grad<N>],
    params_f64: &[f64],
    require_full_frame: bool,
) -> Result<FieldAccel<N>, PropagationError> {
    let need_y_or_b = require_full_frame || ecom2_needs_y_or_b(n_d, n_b, params_f64);
    let ey_collinear = is_collinear_exact(sun_pos_km, sat_pos_km);
    if ey_collinear && need_y_or_b {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 field eY is undefined: eD is parallel to satellite position".to_string(),
        ));
    }
    let has_periodic_terms = n_d > 0 || n_b > 0;
    let need_periodic = (require_full_frame && has_periodic_terms)
        || ecom2_has_nonzero_periodic(n_d, n_b, params_f64);
    let du_undefined =
        has_periodic_terms && is_delta_u_undefined_exact(sat_pos_km, sat_vel_km_s, sun_pos_km);
    if du_undefined && need_periodic {
        return Err(PropagationError::NumericalFailure(
            "ECOM2 field delta_u is undefined or overflows".to_string(),
        ));
    }

    let fast_attempt = (|| -> Option<FieldVec3<N>> {
        let sat_m = to_vec3_m(sat_pos_km).ok()?;
        let sun_m = to_vec3_m(sun_pos_km).ok()?;
        let sat_pos = if sat_pos_is_var {
            FieldVec3::<N>::new(
                Grad::variable(0, sat_m.x),
                Grad::variable(1, sat_m.y),
                Grad::variable(2, sat_m.z),
            )
        } else {
            FieldVec3::<N>::constant(sat_m)
        };
        let sun_pos = FieldVec3::<N>::constant(sun_m);
        let e_d = dyb_e_d_field_fast(sat_pos, sun_pos)?;

        let (e_y, e_b) = if ey_collinear {
            (FieldVec3::ZERO, FieldVec3::ZERO)
        } else {
            match dyb_e_y_field_fast(e_d, sat_pos) {
                Some(ey) => (ey, dyb_e_b_field_fast(e_d, ey)?),
                None if !need_y_or_b => (FieldVec3::ZERO, FieldVec3::ZERO),
                None => return None,
            }
        };

        let (b_u, d_u) = if !has_periodic_terms {
            (params[0], params[1])
        } else if du_undefined {
            (params[0], params[2 * n_b + 1])
        } else {
            let delta_u_opt =
                orekit_delta_u_field_fast(sat_pos, sat_vel_km_s, sat_vel_is_var, sun_pos);

            if let Some(delta_u) = delta_u_opt {
                let mut b_u = params[0];
                for i in 1..=n_b {
                    let (sin, cos) = delta_u.mul_f64((2 * i - 1) as f64).sin_cos();
                    b_u = b_u.add(cos.mul(params[i])).add(sin.mul(params[i + n_b]));
                }
                let mut d_u = params[2 * n_b + 1];
                for i in 1..=n_d {
                    let (sin, cos) = delta_u.mul_f64((2 * i) as f64).sin_cos();
                    d_u = d_u
                        .add(cos.mul(params[2 * n_b + 1 + i]))
                        .add(sin.mul(params[2 * n_b + 1 + i + n_d]));
                }
                (b_u, d_u)
            } else if !need_periodic {
                (params[0], params[2 * n_b + 1])
            } else {
                return None;
            }
        };

        let y0 = params[2 * (n_d + n_b) + 2];
        let raw = FieldVec3::lincomb3(d_u, e_d, y0, e_y, b_u, e_b);
        let accel = if is_sunlit {
            raw
        } else {
            raw.scalar_mul(ratio_fast)
        };
        field_vec3_is_all_finite(accel).then_some(accel)
    })();

    if let Some(accel) = fast_attempt {
        return Ok(FieldAccel::Fast(accel));
    }

    ecom2_acceleration_field_scaled(
        sat_pos_km,
        sat_vel_km_s,
        sun_pos_km,
        sat_pos_is_var,
        sat_vel_is_var,
        ratio_s_km,
        is_sunlit,
        n_d,
        n_b,
        params,
        params_f64,
        require_full_frame,
    )
    .map(FieldAccel::Scaled)
}

#[allow(clippy::too_many_arguments)]
fn ecom2_state_partials_with_bodies(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    n_d: usize,
    n_b: usize,
    params: &[f64],
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<AccelerationPartials, PropagationError> {
    validate_finite_vec3("position_km", sat_pos_km)?;
    validate_finite_vec3("sun_position_km", sun_pos_km)?;

    if matches!(shadow_function, ShadowFunction::Conical) {
        let (ratio_km, is_sunlit) =
            lighting_ratio_field_conical_scaled_km(sat_pos_km, sun_pos_km, earth_shadow_model)?;
        if !is_sunlit && ratio_km.val.is_zero() {
            return Ok(AccelerationPartials::zeros());
        }
        let ratio_s_km = ScaledGrad {
            val: ratio_km.val,
            grad: [
                ratio_km.grad[0],
                ratio_km.grad[1],
                ratio_km.grad[2],
                ScaledF64::ZERO,
                ScaledF64::ZERO,
                ScaledF64::ZERO,
            ],
        };
        let sat_vel_km_s =
            ecom_velocity(sat_vel_km_s, ecom2_has_nonzero_periodic(n_d, n_b, params))?;
        let params_g: Vec<Grad<6>> = params.iter().map(|&c| Grad::constant(c)).collect();
        let partials = if is_sunlit {
            match ecom2_acceleration_field(
                sat_pos_km,
                sat_vel_km_s,
                sun_pos_km,
                true,
                true,
                Grad::<6>::constant(1.0),
                ratio_s_km,
                true,
                n_d,
                n_b,
                &params_g,
                params,
                false,
            )? {
                FieldAccel::Fast(accel) => extract_state_partials(accel),
                FieldAccel::Scaled(accel_s) => extract_state_partials_scaled(accel_s)?,
            }
        } else {
            let accel_s = ecom2_acceleration_field_scaled(
                sat_pos_km,
                sat_vel_km_s,
                sun_pos_km,
                true,
                true,
                ratio_s_km,
                false,
                n_d,
                n_b,
                &params_g,
                params,
                false,
            )?;
            extract_state_partials_scaled(accel_s)?
        };
        return finite_partials(partials);
    }

    let ratio3 = lighting_ratio_field(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    )?;
    if ratio3.is_zero() {
        return Ok(AccelerationPartials::zeros());
    }

    let ratio = Grad::<6>::from_pos3(ratio3);
    let m_per_km = ScaledF64::from_f64(M_PER_KM);
    let ratio_s_km = ScaledGrad {
        val: ScaledF64::from_f64(ratio3.value),
        grad: [
            ScaledF64::from_f64(ratio3.grad[0]).mul(m_per_km),
            ScaledF64::from_f64(ratio3.grad[1]).mul(m_per_km),
            ScaledF64::from_f64(ratio3.grad[2]).mul(m_per_km),
            ScaledF64::ZERO,
            ScaledF64::ZERO,
            ScaledF64::ZERO,
        ],
    };
    let sat_vel_km_s = ecom_velocity(sat_vel_km_s, ecom2_has_nonzero_periodic(n_d, n_b, params))?;
    let params_g: Vec<Grad<6>> = params.iter().map(|&c| Grad::constant(c)).collect();

    let partials = match ecom2_acceleration_field(
        sat_pos_km,
        sat_vel_km_s,
        sun_pos_km,
        true,
        true,
        ratio,
        ratio_s_km,
        is_sunlit_field(ratio3),
        n_d,
        n_b,
        &params_g,
        params,
        false,
    )? {
        FieldAccel::Fast(accel) => extract_state_partials(accel),
        FieldAccel::Scaled(accel_s) => extract_state_partials_scaled(accel_s)?,
    };

    finite_partials(partials)
}

#[allow(clippy::too_many_arguments)]
fn ecom2_parameter_partials_chunk<const CHUNK: usize>(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_s: ScaledF64,
    is_sunlit: bool,
    force_scaled: bool,
    n_d: usize,
    n_b: usize,
    params: &[f64],
    start: usize,
    len: usize,
    out: &mut Vec<Vector3<f64>>,
) -> Result<(), PropagationError> {
    let mut params_g = Vec::with_capacity(params.len());
    for (idx, &c) in params.iter().enumerate() {
        if idx >= start && idx < start + len {
            params_g.push(Grad::variable(idx - start, c));
        } else {
            params_g.push(Grad::constant(c));
        }
    }
    let ratio_s_chunk = ScaledGrad {
        val: ratio_s,
        grad: [ScaledF64::ZERO; CHUNK],
    };
    let field_res = if force_scaled {
        ecom2_acceleration_field_scaled(
            sat_pos_km,
            sat_vel_km_s,
            sun_pos_km,
            false,
            false,
            ratio_s_chunk,
            is_sunlit,
            n_d,
            n_b,
            &params_g,
            params,
            true,
        )
        .map(FieldAccel::Scaled)?
    } else {
        let ratio = Grad::<CHUNK>::constant(ratio_s.to_f64());
        ecom2_acceleration_field(
            sat_pos_km,
            sat_vel_km_s,
            sun_pos_km,
            false,
            false,
            ratio,
            ratio_s_chunk,
            is_sunlit,
            n_d,
            n_b,
            &params_g,
            params,
            true,
        )?
    };
    match field_res {
        FieldAccel::Fast(accel) => {
            for k in 0..len {
                out.push(Vector3::new(
                    accel.x.grad[k] / M_PER_KM,
                    accel.y.grad[k] / M_PER_KM,
                    accel.z.grad[k] / M_PER_KM,
                ));
            }
        }
        FieldAccel::Scaled(accel_s) => {
            for k in 0..len {
                out.push(extract_param_partial_scaled(accel_s, k)?);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ecom2_parameter_partials_with_bodies(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    n_d: usize,
    n_b: usize,
    params: &[f64],
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<Vec<Vector3<f64>>, PropagationError> {
    validate_finite_vec3("position_km", sat_pos_km)?;
    validate_finite_vec3("sun_position_km", sun_pos_km)?;

    let (ratio_s, sunlit) = match lighting_ratio_for_parameter_partials(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    )? {
        ParameterLighting::Umbra => return Ok(vec![Vector3::zeros(); params.len()]),
        ParameterLighting::Lit { ratio, is_sunlit } => (ratio, is_sunlit),
    };
    let sat_vel_km_s = ecom_velocity(sat_vel_km_s, n_d != 0 || n_b != 0)?;
    let force_scaled = matches!(shadow_function, ShadowFunction::Conical) && !sunlit;

    let mut out = Vec::with_capacity(params.len());
    match params.len() {
        3 => ecom2_parameter_partials_chunk::<3>(
            sat_pos_km,
            sat_vel_km_s,
            sun_pos_km,
            ratio_s,
            sunlit,
            force_scaled,
            n_d,
            n_b,
            params,
            0,
            3,
            &mut out,
        )?,
        7 => ecom2_parameter_partials_chunk::<7>(
            sat_pos_km,
            sat_vel_km_s,
            sun_pos_km,
            ratio_s,
            sunlit,
            force_scaled,
            n_d,
            n_b,
            params,
            0,
            7,
            &mut out,
        )?,
        9 => ecom2_parameter_partials_chunk::<9>(
            sat_pos_km,
            sat_vel_km_s,
            sun_pos_km,
            ratio_s,
            sunlit,
            force_scaled,
            n_d,
            n_b,
            params,
            0,
            9,
            &mut out,
        )?,
        total => {
            let mut start = 0;
            while start < total {
                let len = (total - start).min(8);
                ecom2_parameter_partials_chunk::<8>(
                    sat_pos_km,
                    sat_vel_km_s,
                    sun_pos_km,
                    ratio_s,
                    sunlit,
                    force_scaled,
                    n_d,
                    n_b,
                    params,
                    start,
                    len,
                    &mut out,
                )?;
                start += len;
            }
        }
    }
    finite_parameter_partials(out)
}

// ---------------------------------------------------------------------------
// ECOM1 double & Field<Gradient> core (exponent-separated arithmetic throughout)
// ---------------------------------------------------------------------------

#[inline]
fn ecom1_u_yx_double_scaled(
    pos_s: ScaledVec3,
    vel_s: ScaledVec3,
) -> Option<(ScaledF64, ScaledF64)> {
    let momentum_s = pos_s.cross(vel_s);
    let (normal_s, _) = momentum_s.normalize_scaled()?;
    let (node_s, _) = ScaledVec3 {
        x: momentum_s.y.negate(),
        y: momentum_s.x,
        z: ScaledF64::ZERO,
    }
    .normalize_scaled()?;
    let ahead_s = normal_s.cross(node_s);
    Some((pos_s.dot(ahead_s), pos_s.dot(node_s)))
}

#[inline]
fn ecom1_u_double_scaled(pos_s: ScaledVec3, vel_s: ScaledVec3) -> Option<f64> {
    let (y, x) = ecom1_u_yx_double_scaled(pos_s, vel_s)?;
    scaled_atan2_double(y, x, libm::atan2)
}

#[inline]
fn ecom1_sin_cos_double_scaled(
    pos_s: ScaledVec3,
    vel_s: ScaledVec3,
) -> Option<(ScaledF64, ScaledF64)> {
    let (y, x) = ecom1_u_yx_double_scaled(pos_s, vel_s)?;
    ScaledF64::sin_cos_from_yx(y, x)
}

/// ECOM1 argument of latitude `u`, evaluated in exponent-separated arithmetic
/// (`node = unit(z x (r x v))`, `ahead = normal x node`,
/// `u = atan2(r . ahead, r . node)`) for reporting by `ecom_geometry*`.
fn ecom1_u_double(position_km: [f64; 3], velocity_km_s: [f64; 3]) -> Option<f64> {
    if !(position_km.iter().all(|c| c.is_finite()) && velocity_km_s.iter().all(|c| c.is_finite())) {
        return None;
    }
    let pos_s = ScaledVec3::from_vec3(Vec3::new(position_km[0], position_km[1], position_km[2]));
    let vel_s = ScaledVec3::from_vec3(Vec3::new(
        velocity_km_s[0],
        velocity_km_s[1],
        velocity_km_s[2],
    ));
    ecom1_u_double_scaled(pos_s, vel_s)
}

/// Forward-mode `ScaledF64` evaluation of `(sin u, cos u) = (y / rho, x / rho)`
/// with `y = pos . ahead`, `x = pos . node`, and `rho = hypot(x, y)`, without
/// binary64 `u`, `sin`, or `cos` intermediates.
fn ecom1_sin_cos_field_scaled<const N: usize>(
    pos_s: ScaledFieldVec3<N>,
    vel_s: ScaledFieldVec3<N>,
) -> Option<(ScaledGrad<N>, ScaledGrad<N>)> {
    let momentum_s = pos_s.cross(vel_s);
    let normal_s = momentum_s.normalize()?;
    let node_s = ScaledFieldVec3 {
        x: momentum_s.y.negate(),
        y: momentum_s.x,
        z: ScaledGrad::ZERO,
    }
    .normalize()?;
    let ahead_s = normal_s.cross(node_s);
    pos_s.dot(ahead_s).sin_cos_from_yx(pos_s.dot(node_s))
}

#[inline]
fn ecom_velocity(velocity: Vector3<f64>, required: bool) -> Result<Vector3<f64>, PropagationError> {
    match validate_finite_vec3("velocity_km_s", velocity) {
        Ok(()) => Ok(velocity),
        Err(_) if !required => Ok(Vector3::zeros()),
        Err(error) => Err(error),
    }
}

fn ecom1_has_nonzero_periodic(c: [f64; ECOM1_COEFFICIENT_COUNT]) -> bool {
    c[1] != 0.0 || c[2] != 0.0 || c[4] != 0.0 || c[5] != 0.0 || c[7] != 0.0 || c[8] != 0.0
}

#[inline]
fn ecom1_needs_y_or_b(c: [f64; ECOM1_COEFFICIENT_COUNT]) -> bool {
    c[3] != 0.0 || c[4] != 0.0 || c[5] != 0.0 || c[6] != 0.0 || c[7] != 0.0 || c[8] != 0.0
}

#[inline]
fn scaled_field_vec3_constant<const N: usize>(v: Vector3<f64>) -> ScaledFieldVec3<N> {
    ScaledFieldVec3 {
        x: ScaledGrad {
            val: ScaledF64::from_f64(v.x),
            grad: [ScaledF64::ZERO; N],
        },
        y: ScaledGrad {
            val: ScaledF64::from_f64(v.y),
            grad: [ScaledF64::ZERO; N],
        },
        z: ScaledGrad {
            val: ScaledF64::from_f64(v.z),
            grad: [ScaledF64::ZERO; N],
        },
    }
}

#[inline]
fn scaled_field_vec3_variable<const N: usize>(
    v: Vector3<f64>,
    var_base: usize,
) -> ScaledFieldVec3<N> {
    let one = ScaledF64::from_f64(1.0);
    let mut gx = [ScaledF64::ZERO; N];
    let mut gy = [ScaledF64::ZERO; N];
    let mut gz = [ScaledF64::ZERO; N];
    gx[var_base] = one;
    gy[var_base + 1] = one;
    gz[var_base + 2] = one;
    ScaledFieldVec3 {
        x: ScaledGrad {
            val: ScaledF64::from_f64(v.x),
            grad: gx,
        },
        y: ScaledGrad {
            val: ScaledF64::from_f64(v.y),
            grad: gy,
        },
        z: ScaledGrad {
            val: ScaledF64::from_f64(v.z),
            grad: gz,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn ecom1_acceleration_with_bodies(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    c: [f64; ECOM1_COEFFICIENT_COUNT],
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<Vector3<f64>, PropagationError> {
    validate_finite_vec3("position_km", sat_pos_km)?;
    validate_finite_vec3("sun_position_km", sun_pos_km)?;

    let (ratio_s, is_sunlit) = lighting_ratio_double_scaled(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    )?;
    if ratio_s.is_zero() {
        return Ok(Vector3::zeros());
    }

    let sat_vel_km_s = ecom_velocity(sat_vel_km_s, ecom1_has_nonzero_periodic(c))?;
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let vel_s = ScaledVec3::from_vec3(Vec3::new(sat_vel_km_s.x, sat_vel_km_s.y, sat_vel_km_s.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    let (e_d, _) = sun_s.sub(sat_s).normalize_scaled().ok_or_else(|| {
        PropagationError::NumericalFailure("ECOM1 eD is undefined or overflows".to_string())
    })?;

    let need_y_or_b = ecom1_needs_y_or_b(c);
    let (e_y, e_b) = match scaled_e_y_double(sat_s, sun_s) {
        Some(ey) => (ey, e_d.cross(ey)),
        None if !need_y_or_b => (ScaledVec3::ZERO, ScaledVec3::ZERO),
        None => {
            return Err(PropagationError::NumericalFailure(
                "ECOM1 eY is undefined: eD is parallel to satellite position".to_string(),
            ));
        }
    };

    let c_s = c.map(ScaledF64::from_f64);
    let [d0, dc, ds, y0, yc, ys, b0, bc, bs] = c_s;
    let (d_u, y_u, b_u) = if ecom1_has_nonzero_periodic(c) {
        let (su, cu) = ecom1_sin_cos_double_scaled(sat_s, vel_s).ok_or_else(|| {
            PropagationError::NumericalFailure(
                "ECOM1 argument of latitude is undefined (equatorial or rectilinear orbit)"
                    .to_string(),
            )
        })?;
        (
            d0.add(cu.mul(dc)).add(su.mul(ds)),
            y0.add(cu.mul(yc)).add(su.mul(ys)),
            b0.add(cu.mul(bc)).add(su.mul(bs)),
        )
    } else {
        (d0, y0, b0)
    };

    let raw = ScaledVec3::lincomb3(d_u, e_d, y_u, e_y, b_u, e_b);
    let scaled = if is_sunlit {
        raw
    } else {
        raw.scalar_mul(ratio_s)
    };
    let acc_km = scaled
        .scalar_div(ScaledF64::from_f64(M_PER_KM))
        .to_vec3()
        .ok_or_else(|| {
            PropagationError::NumericalFailure("ECOM1 acceleration is not finite".to_string())
        })?;
    finite_acceleration(Vector3::new(acc_km.x, acc_km.y, acc_km.z))
}

#[allow(clippy::too_many_arguments)]
fn ecom1_acceleration_field<const N: usize>(
    sat_s: ScaledFieldVec3<N>,
    vel_s: ScaledFieldVec3<N>,
    sun_s: ScaledFieldVec3<N>,
    ratio: ScaledGrad<N>,
    is_sunlit: bool,
    c: [ScaledGrad<N>; ECOM1_COEFFICIENT_COUNT],
    c_f64: [f64; ECOM1_COEFFICIENT_COUNT],
    require_full_frame: bool,
) -> Result<ScaledFieldVec3<N>, PropagationError> {
    let e_d = sun_s.sub(sat_s).normalize().ok_or_else(|| {
        PropagationError::NumericalFailure("ECOM1 field eD is undefined or not finite".to_string())
    })?;

    let need_y_or_b = require_full_frame || ecom1_needs_y_or_b(c_f64);
    let (e_y, e_b) = match scaled_e_y_field(sat_s, sun_s) {
        Some(ey) => (ey, e_d.cross(ey)),
        None if !need_y_or_b => (ScaledFieldVec3::ZERO, ScaledFieldVec3::ZERO),
        None => {
            return Err(PropagationError::NumericalFailure(
                "ECOM1 field eY is undefined: eD is parallel to satellite position".to_string(),
            ));
        }
    };

    let [d0, dc, ds, y0, yc, ys, b0, bc, bs] = c;
    let need_u = require_full_frame || ecom1_has_nonzero_periodic(c_f64);
    let (d_u, y_u, b_u) = if need_u {
        let (su, cu) = ecom1_sin_cos_field_scaled(sat_s, vel_s).ok_or_else(|| {
            PropagationError::NumericalFailure(
                "ECOM1 field argument of latitude is undefined or overflows".to_string(),
            )
        })?;
        (
            d0.add(cu.mul(dc)).add(su.mul(ds)),
            y0.add(cu.mul(yc)).add(su.mul(ys)),
            b0.add(cu.mul(bc)).add(su.mul(bs)),
        )
    } else {
        (d0, y0, b0)
    };

    let raw = ScaledFieldVec3::lincomb3(d_u, e_d, y_u, e_y, b_u, e_b);
    let scaled = if is_sunlit {
        raw
    } else {
        raw.scalar_mul(ratio)
    };
    Ok(scaled.scalar_div(ScaledF64::from_f64(M_PER_KM)))
}

#[allow(clippy::too_many_arguments)]
fn ecom1_state_partials_with_bodies(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    c: [f64; ECOM1_COEFFICIENT_COUNT],
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<AccelerationPartials, PropagationError> {
    validate_finite_vec3("position_km", sat_pos_km)?;
    validate_finite_vec3("sun_position_km", sun_pos_km)?;

    let (ratio_s, is_sunlit) = if matches!(shadow_function, ShadowFunction::Conical) {
        let (ratio_km, is_sunlit) =
            lighting_ratio_field_conical_scaled_km(sat_pos_km, sun_pos_km, earth_shadow_model)?;
        if !is_sunlit && ratio_km.val.is_zero() {
            return Ok(AccelerationPartials::zeros());
        }
        (
            ScaledGrad {
                val: ratio_km.val,
                grad: [
                    ratio_km.grad[0],
                    ratio_km.grad[1],
                    ratio_km.grad[2],
                    ScaledF64::ZERO,
                    ScaledF64::ZERO,
                    ScaledF64::ZERO,
                ],
            },
            is_sunlit,
        )
    } else {
        let ratio3 = lighting_ratio_field(
            sat_pos_km,
            sun_pos_km,
            moon_pos_km,
            shadow_function,
            earth_shadow_model,
            include_moon_eclipse,
        )?;
        if ratio3.is_zero() {
            return Ok(AccelerationPartials::zeros());
        }
        let m_per_km = ScaledF64::from_f64(M_PER_KM);
        (
            ScaledGrad {
                val: ScaledF64::from_f64(ratio3.value),
                grad: [
                    ScaledF64::from_f64(ratio3.grad[0]).mul(m_per_km),
                    ScaledF64::from_f64(ratio3.grad[1]).mul(m_per_km),
                    ScaledF64::from_f64(ratio3.grad[2]).mul(m_per_km),
                    ScaledF64::ZERO,
                    ScaledF64::ZERO,
                    ScaledF64::ZERO,
                ],
            },
            is_sunlit_field(ratio3),
        )
    };

    let sat_vel_km_s = ecom_velocity(sat_vel_km_s, ecom1_has_nonzero_periodic(c))?;
    let sat_s = scaled_field_vec3_variable::<6>(sat_pos_km, 0);
    let vel_s = scaled_field_vec3_variable::<6>(sat_vel_km_s, 3);
    let sun_s = scaled_field_vec3_constant::<6>(sun_pos_km);
    let c_s = c.map(|v| ScaledGrad {
        val: ScaledF64::from_f64(v),
        grad: [ScaledF64::ZERO; 6],
    });

    let accel_s = ecom1_acceleration_field(sat_s, vel_s, sun_s, ratio_s, is_sunlit, c_s, c, false)?;
    let partials = extract_state_partials_scaled(accel_s)?;
    finite_partials(partials)
}

#[allow(clippy::too_many_arguments)]
fn ecom1_parameter_partials_with_bodies(
    sat_pos_km: Vector3<f64>,
    sat_vel_km_s: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    c: [f64; ECOM1_COEFFICIENT_COUNT],
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
) -> Result<[Vector3<f64>; ECOM1_COEFFICIENT_COUNT], PropagationError> {
    validate_finite_vec3("position_km", sat_pos_km)?;
    validate_finite_vec3("sun_position_km", sun_pos_km)?;

    let (ratio_val, sunlit) = match lighting_ratio_for_parameter_partials(
        sat_pos_km,
        sun_pos_km,
        moon_pos_km,
        shadow_function,
        earth_shadow_model,
        include_moon_eclipse,
    )? {
        ParameterLighting::Umbra => return Ok([Vector3::zeros(); ECOM1_COEFFICIENT_COUNT]),
        ParameterLighting::Lit { ratio, is_sunlit } => (ratio, is_sunlit),
    };

    let sat_vel_km_s = ecom_velocity(sat_vel_km_s, true)?;
    let sat_s = scaled_field_vec3_constant::<9>(sat_pos_km);
    let vel_s = scaled_field_vec3_constant::<9>(sat_vel_km_s);
    let sun_s = scaled_field_vec3_constant::<9>(sun_pos_km);
    let ratio_s = ScaledGrad {
        val: ratio_val,
        grad: [ScaledF64::ZERO; 9],
    };
    let one = ScaledF64::from_f64(1.0);
    let c_s = core::array::from_fn(|k| {
        let mut grad = [ScaledF64::ZERO; 9];
        grad[k] = one;
        ScaledGrad {
            val: ScaledF64::from_f64(c[k]),
            grad,
        }
    });

    let accel_s = ecom1_acceleration_field(sat_s, vel_s, sun_s, ratio_s, sunlit, c_s, c, true)?;
    let mut out = [Vector3::zeros(); ECOM1_COEFFICIENT_COUNT];
    for (k, slot) in out.iter_mut().enumerate() {
        let v = Vector3::new(
            accel_s.x.grad[k].to_f64(),
            accel_s.y.grad[k].to_f64(),
            accel_s.z.grad[k].to_f64(),
        );
        if !v.iter().all(|x| x.is_finite()) {
            return Err(PropagationError::NumericalFailure(
                "ECOM1 parameter partial is not finite".to_string(),
            ));
        }
        *slot = v;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared conversion and validation helpers
// ---------------------------------------------------------------------------

#[inline]
fn extract_state_partials(accel: FieldVec3<6>) -> AccelerationPartials {
    AccelerationPartials {
        wrt_position: Matrix3::from_rows(&[
            RowVector3::new(accel.x.grad[0], accel.x.grad[1], accel.x.grad[2]),
            RowVector3::new(accel.y.grad[0], accel.y.grad[1], accel.y.grad[2]),
            RowVector3::new(accel.z.grad[0], accel.z.grad[1], accel.z.grad[2]),
        ]),
        wrt_velocity: Matrix3::from_rows(&[
            RowVector3::new(accel.x.grad[3], accel.x.grad[4], accel.x.grad[5]),
            RowVector3::new(accel.y.grad[3], accel.y.grad[4], accel.y.grad[5]),
            RowVector3::new(accel.z.grad[3], accel.z.grad[4], accel.z.grad[5]),
        ]),
    }
}

fn extract_state_partials_scaled(
    accel: ScaledFieldVec3<6>,
) -> Result<AccelerationPartials, PropagationError> {
    let mut wrt_position = Matrix3::zeros();
    let mut wrt_velocity = Matrix3::zeros();
    let comps = [accel.x, accel.y, accel.z];
    for row in 0..3 {
        for col in 0..3 {
            let p = comps[row].grad[col].to_f64();
            let v = comps[row].grad[col + 3].to_f64();
            if !(p.is_finite() && v.is_finite()) {
                return Err(PropagationError::NumericalFailure(
                    "ECOM acceleration partial is not finite".to_string(),
                ));
            }
            wrt_position[(row, col)] = p;
            wrt_velocity[(row, col)] = v;
        }
    }
    Ok(AccelerationPartials {
        wrt_position,
        wrt_velocity,
    })
}

fn extract_param_partial_scaled<const N: usize>(
    accel: ScaledFieldVec3<N>,
    k: usize,
) -> Result<Vector3<f64>, PropagationError> {
    let v = Vector3::new(
        accel.x.grad[k].to_f64(),
        accel.y.grad[k].to_f64(),
        accel.z.grad[k].to_f64(),
    );
    if !v.iter().all(|x| x.is_finite()) {
        return Err(PropagationError::NumericalFailure(
            "ECOM parameter partial is not finite".to_string(),
        ));
    }
    Ok(v)
}

#[inline]
fn as_array(v: Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

#[inline]
fn vec3_is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[inline]
fn field_vec3_is_all_finite<const N: usize>(v: FieldVec3<N>) -> bool {
    v.x.is_all_finite() && v.y.is_all_finite() && v.z.is_all_finite()
}

fn to_vec3_m(v_km: Vector3<f64>) -> Result<Vec3, PropagationError> {
    let vm = Vec3::new(v_km.x * M_PER_KM, v_km.y * M_PER_KM, v_km.z * M_PER_KM);
    if !vec3_is_finite(vm) {
        return Err(PropagationError::NumericalFailure(
            "ECOM vector overflows in metre conversion".to_string(),
        ));
    }
    Ok(vm)
}

fn validate_finite_vec3(field: &'static str, v: Vector3<f64>) -> Result<(), PropagationError> {
    if !v.iter().all(|c| c.is_finite()) {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite"
        )));
    }
    Ok(())
}

fn validate_finite_slice(label: &'static str, values: &[f64]) -> Result<(), PropagationError> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PropagationError::InvalidInput(format!(
                "{label} at index {idx} must be finite"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
