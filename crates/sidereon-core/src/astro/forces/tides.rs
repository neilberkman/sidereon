//! Solid Earth tide and solid Earth pole tide geopotential perturbations.
//!
//! These forces implement the low-degree time-variable geopotential corrections
//! from IERS Conventions (2010), Chapter 6, for numerical propagation:
//!
//! * [`SolidEarthTideGravity`] applies Chapter 6, Section 6.2.1. Step 1 gives
//!   the frequency-independent anelastic Love-number corrections to fully
//!   normalized `Cnm`/`Snm` from the Sun and Moon: degree 2, degree 3, and the
//!   degree 4 corrections produced by degree-2 tides. Step 2 adds the
//!   frequency-dependent corrections to `C20`, `C21`/`S21` and `C22`/`S22`,
//!   Equations (6.8a) and (6.8b) summed over the constituents of Tables 6.5b,
//!   6.5a and 6.5c, with each argument `theta_f = m (theta_g + pi) - N . F`
//!   formed from the Greenwich mean sidereal time and the Delaunay arguments.
//!   Step 3 (Section 6.2.2) then takes out of `C20` the permanent tide that
//!   the geopotential already holds, by its [`TideSystem`]: nothing for a
//!   tide-free field such as EGM96, the permanent deformation `A0 H0 k20` of
//!   Equation (6.14) for a zero-tide field, and that plus the permanent
//!   tide-generating potential `A0 H0` for a mean-tide field.
//! * [`SolidEarthPoleTideGravity`] applies the Chapter 6 solid Earth pole tide
//!   correction to normalized `C21` and `S21`, using the polar motion stored in
//!   the propagation context's Earth-orientation provider.
//!
//! Both models are perturbation-only additive forces. They require a
//! body-fixed frame provider in [`PropagationContext`] because the coefficients
//! are evaluated in the terrestrial frame and then rotated back to GCRF.

use crate::astro::bodies::sun_moon::sun_moon_ecef_with_polar_motion;
use crate::astro::constants::astro::{GM_MOON_KM3_S2, GM_SUN_KM3_S2};
use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD};
use crate::astro::constants::units::{ARCSEC_TO_RAD, M_PER_KM};
use crate::astro::error::PropagationError;
use crate::astro::forces::geopotential::{
    SphericalHarmonicCoefficient, SphericalHarmonicGravity, TideSystem, EGM96_MU_KM3_S2,
    EGM96_REFERENCE_RADIUS_KM,
};
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::frames::nutation::iers_2010_solid_tide_arguments;
use crate::astro::frames::orientation::EarthOrientation;
use crate::astro::frames::transforms::{
    greenwich_mean_sidereal_time_radians, with_ut1_validity, PolarMotion,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use crate::astro::time::scales::TimeScales;
use crate::astro::time::ValidityMode;
use nalgebra::Vector3;
use std::f64::consts::PI;

const SOLID_TIDE_MAX_DEGREE: u16 = 4;
const SOLID_TIDE_MAX_ORDER: u16 = 3;
const POLE_TIDE_MAX_DEGREE: u16 = 2;
const POLE_TIDE_MAX_ORDER: u16 = 1;

/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Re k20`.
pub const SOLID_EARTH_TIDE_K20_REAL: f64 = 0.30190;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Im k20`.
pub const SOLID_EARTH_TIDE_K20_IMAG: f64 = 0.0;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `k20+`.
pub const SOLID_EARTH_TIDE_K20_PLUS: f64 = -0.00089;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Re k21`.
pub const SOLID_EARTH_TIDE_K21_REAL: f64 = 0.29830;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Im k21`.
pub const SOLID_EARTH_TIDE_K21_IMAG: f64 = -0.00144;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `k21+`.
pub const SOLID_EARTH_TIDE_K21_PLUS: f64 = -0.00080;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Re k22` = 0.30102.
pub const SOLID_EARTH_TIDE_K22_REAL: f64 = f64::from_bits(0x3fd3_43e9_63dc_486b);
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Im k22`.
pub const SOLID_EARTH_TIDE_K22_IMAG: f64 = -0.00130;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `k22+`.
pub const SOLID_EARTH_TIDE_K22_PLUS: f64 = -0.00057;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k30`.
pub const SOLID_EARTH_TIDE_K30_REAL: f64 = 0.093;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k31`.
pub const SOLID_EARTH_TIDE_K31_REAL: f64 = 0.093;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k32`.
pub const SOLID_EARTH_TIDE_K32_REAL: f64 = 0.093;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k33`.
pub const SOLID_EARTH_TIDE_K33_REAL: f64 = 0.094;

/// IERS Conventions (2010), Chapter 6, Equation (6.8c): `A0 = 1 / (Re sqrt(4 pi))`,
/// per metre of tide-generating potential amplitude.
pub const SOLID_EARTH_TIDE_A0_PER_M: f64 = 4.4228e-8;
/// IERS Conventions (2010), Chapter 6, Equation (6.14): amplitude `H0` of the
/// permanent (zero-frequency) part of the degree-2 zonal tide-generating
/// potential, metres.
pub const PERMANENT_TIDE_H0_M: f64 = -0.31460;

/// Chapter 6 solid Earth pole tide normalized-coefficient scale.
pub const SOLID_EARTH_POLE_TIDE_SCALE: f64 = -1.333e-9;
/// Chapter 6 solid Earth pole tide imaginary Love-number coupling.
pub const SOLID_EARTH_POLE_TIDE_IMAG_COUPLING: f64 = 0.0115;

#[derive(Debug, Clone, Copy, PartialEq)]
struct ComplexLoveNumber {
    real: f64,
    imag: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Degree2LoveNumber {
    primary: ComplexLoveNumber,
    plus: f64,
}

const DEGREE2_LOVE: [Degree2LoveNumber; 3] = [
    Degree2LoveNumber {
        primary: ComplexLoveNumber {
            real: SOLID_EARTH_TIDE_K20_REAL,
            imag: SOLID_EARTH_TIDE_K20_IMAG,
        },
        plus: SOLID_EARTH_TIDE_K20_PLUS,
    },
    Degree2LoveNumber {
        primary: ComplexLoveNumber {
            real: SOLID_EARTH_TIDE_K21_REAL,
            imag: SOLID_EARTH_TIDE_K21_IMAG,
        },
        plus: SOLID_EARTH_TIDE_K21_PLUS,
    },
    Degree2LoveNumber {
        primary: ComplexLoveNumber {
            real: SOLID_EARTH_TIDE_K22_REAL,
            imag: SOLID_EARTH_TIDE_K22_IMAG,
        },
        plus: SOLID_EARTH_TIDE_K22_PLUS,
    },
];

const DEGREE3_LOVE: [ComplexLoveNumber; 4] = [
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K30_REAL,
        imag: 0.0,
    },
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K31_REAL,
        imag: 0.0,
    },
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K32_REAL,
        imag: 0.0,
    },
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K33_REAL,
        imag: 0.0,
    },
];

/// Solid Earth tide geopotential perturbation force, IERS Conventions (2010)
/// Chapter 6, Section 6.2.1, Steps 1 and 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidEarthTideGravity {
    /// Earth gravitational parameter used by the correction potential, km^3/s^2.
    pub mu_earth_km3_s2: f64,
    /// Reference equatorial radius used by the correction potential, km.
    pub reference_radius_km: f64,
    /// Solar gravitational parameter, km^3/s^2.
    pub gm_sun_km3_s2: f64,
    /// Lunar gravitational parameter, km^3/s^2.
    pub gm_moon_km3_s2: f64,
    /// Whether the Step 2 frequency-dependent corrections of Tables 6.5a-c are
    /// added to the Step 1 corrections. `true` by default and from
    /// [`SolidEarthTideGravity::new`]; `false` gives Step 1 alone.
    pub frequency_dependent: bool,
    /// Tide system of the geopotential these corrections are added to; Step 3
    /// takes out of `C20` the permanent tide that geopotential already holds.
    /// [`TideSystem::TideFree`] by default and from
    /// [`SolidEarthTideGravity::new`], matching the embedded EGM96 field and
    /// the default zonal coefficients. A composite force refuses a value that
    /// differs from its zonal or spherical-harmonic field's.
    pub tide_system: TideSystem,
}

impl Default for SolidEarthTideGravity {
    fn default() -> Self {
        Self {
            mu_earth_km3_s2: EGM96_MU_KM3_S2,
            reference_radius_km: EGM96_REFERENCE_RADIUS_KM,
            gm_sun_km3_s2: GM_SUN_KM3_S2,
            gm_moon_km3_s2: GM_MOON_KM3_S2,
            frequency_dependent: true,
            tide_system: TideSystem::TideFree,
        }
    }
}

impl SolidEarthTideGravity {
    /// Build with explicit Earth, Sun, and Moon parameters, the Step 2
    /// corrections on, and a tide-free geopotential.
    pub fn new(
        mu_earth_km3_s2: f64,
        reference_radius_km: f64,
        gm_sun_km3_s2: f64,
        gm_moon_km3_s2: f64,
    ) -> Self {
        Self {
            mu_earth_km3_s2,
            reference_radius_km,
            gm_sun_km3_s2,
            gm_moon_km3_s2,
            frequency_dependent: true,
            tide_system: TideSystem::TideFree,
        }
    }

    /// Build for `geopotential`: its gravitational parameter, reference
    /// radius and tide system, the crate's Sun and Moon parameters, and the
    /// Step 2 corrections on.
    pub fn for_geopotential(geopotential: &SphericalHarmonicGravity) -> Self {
        Self {
            mu_earth_km3_s2: geopotential.mu_km3_s2(),
            reference_radius_km: geopotential.reference_radius_km(),
            gm_sun_km3_s2: GM_SUN_KM3_S2,
            gm_moon_km3_s2: GM_MOON_KM3_S2,
            frequency_dependent: true,
            tide_system: geopotential.tide_system(),
        }
    }

    /// Step 1 and Step 3 coefficient corrections from already body-fixed Sun
    /// and Moon positions. The Step 2 corrections depend on the epoch rather
    /// than on the bodies' positions and are not included; see
    /// [`SolidEarthTideGravity::coefficient_corrections_at_epoch`].
    pub fn coefficient_corrections_for_body_fixed_bodies(
        &self,
        sun_itrf_km: [f64; 3],
        moon_itrf_km: [f64; 3],
    ) -> Result<[SphericalHarmonicCoefficient; 10], PropagationError> {
        self.corrections(sun_itrf_km, moon_itrf_km, None)
    }

    /// Step 1 from the bodies, plus `step2` when given, then Step 3, in the
    /// order Orekit's `SolidTidesField` applies them.
    fn corrections(
        &self,
        sun_itrf_km: [f64; 3],
        moon_itrf_km: [f64; 3],
        step2: Option<[SphericalHarmonicCoefficient; 3]>,
    ) -> Result<[SphericalHarmonicCoefficient; 10], PropagationError> {
        validate_positive(self.mu_earth_km3_s2, "mu_earth_km3_s2")?;
        validate_positive(self.reference_radius_km, "reference_radius_km")?;
        validate_positive(self.gm_sun_km3_s2, "gm_sun_km3_s2")?;
        validate_positive(self.gm_moon_km3_s2, "gm_moon_km3_s2")?;

        let mut corrections = empty_solid_tide_coefficients();
        add_body_tide_coefficients(self, self.gm_sun_km3_s2, sun_itrf_km, &mut corrections)?;
        add_body_tide_coefficients(self, self.gm_moon_km3_s2, moon_itrf_km, &mut corrections)?;
        if let Some(step2) = step2 {
            for (index, correction) in step2.iter().enumerate() {
                corrections[index].c += correction.c;
                corrections[index].s += correction.s;
            }
        }
        corrections[0].c -= permanent_tide_c20(self.tide_system);
        if corrections
            .iter()
            .any(|coefficient| !coefficient.c.is_finite() || !coefficient.s.is_finite())
        {
            return Err(PropagationError::NumericalFailure(
                "solid Earth tide coefficient is not representable".to_string(),
            ));
        }
        Ok(corrections)
    }

    /// Coefficient corrections at an epoch using the context's body-fixed
    /// frame provider: Step 1, plus Step 2 when
    /// [`SolidEarthTideGravity::frequency_dependent`] is set, then Step 3.
    pub fn coefficient_corrections_at_epoch(
        &self,
        epoch_tdb_seconds: f64,
        ctx: &PropagationContext,
    ) -> Result<[SphericalHarmonicCoefficient; 10], PropagationError> {
        let orientation = orientation_at_state(ctx, epoch_tdb_seconds)?;
        self.corrections_for_orientation(&orientation)
    }

    fn corrections_for_orientation(
        &self,
        orientation: &EarthOrientation,
    ) -> Result<[SphericalHarmonicCoefficient; 10], PropagationError> {
        let bodies = sun_moon_itrf_km(orientation)?;
        let step2 = if self.frequency_dependent {
            Some(frequency_dependent_corrections_at(
                &orientation.time_scales(),
            )?)
        } else {
            None
        };
        self.corrections(bodies.sun_itrf_km, bodies.moon_itrf_km, step2)
    }
}

impl ForceModel for SolidEarthTideGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
        let position_itrf_km = orientation
            .gcrf_to_itrf_position_km(state.position_array())
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth tide body-fixed position rotation failed: {error}"
                ))
            })?;
        let corrections = self.corrections_for_orientation(&orientation)?;
        let tide = SphericalHarmonicGravity::from_normalized_coefficients(
            self.mu_earth_km3_s2,
            self.reference_radius_km,
            SOLID_TIDE_MAX_DEGREE,
            SOLID_TIDE_MAX_ORDER,
            &corrections,
            // A field of corrections holds no permanent tide of its own.
            TideSystem::TideFree,
        )?;
        let accel_itrf = tide.body_fixed_acceleration_km_s2(position_itrf_km)?;
        let accel_gcrf = orientation
            .itrf_to_gcrf_position_km(accel_itrf)
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth tide inertial acceleration rotation failed: {error}"
                ))
            })?;
        Ok(Vector3::new(accel_gcrf[0], accel_gcrf[1], accel_gcrf[2]))
    }
}

/// Solid Earth pole tide geopotential perturbation force.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidEarthPoleTideGravity {
    /// Earth gravitational parameter used by the correction potential, km^3/s^2.
    pub mu_earth_km3_s2: f64,
    /// Reference equatorial radius used by the correction potential, km.
    pub reference_radius_km: f64,
}

impl Default for SolidEarthPoleTideGravity {
    fn default() -> Self {
        Self {
            mu_earth_km3_s2: EGM96_MU_KM3_S2,
            reference_radius_km: EGM96_REFERENCE_RADIUS_KM,
        }
    }
}

impl SolidEarthPoleTideGravity {
    /// Build with explicit Earth gravity parameters.
    pub fn new(mu_earth_km3_s2: f64, reference_radius_km: f64) -> Self {
        Self {
            mu_earth_km3_s2,
            reference_radius_km,
        }
    }

    /// Coefficient correction from time scales and polar motion.
    pub fn coefficient_correction(
        &self,
        time_scales: TimeScales,
        polar_motion: PolarMotion,
    ) -> Result<SphericalHarmonicCoefficient, PropagationError> {
        validate_positive(self.mu_earth_km3_s2, "mu_earth_km3_s2")?;
        validate_positive(self.reference_radius_km, "reference_radius_km")?;
        if !polar_motion.xp_rad.is_finite() || !polar_motion.yp_rad.is_finite() {
            return Err(PropagationError::InvalidInput(
                "polar motion components must be finite".to_string(),
            ));
        }
        let (m1, m2) = wobble_arcsec(time_scales, polar_motion)?;
        Ok(SphericalHarmonicCoefficient {
            degree: 2,
            order: 1,
            c: SOLID_EARTH_POLE_TIDE_SCALE * (m1 + SOLID_EARTH_POLE_TIDE_IMAG_COUPLING * m2),
            s: SOLID_EARTH_POLE_TIDE_SCALE * (m2 - SOLID_EARTH_POLE_TIDE_IMAG_COUPLING * m1),
        })
    }
}

impl ForceModel for SolidEarthPoleTideGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
        let position_itrf_km = orientation
            .gcrf_to_itrf_position_km(state.position_array())
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth pole tide body-fixed position rotation failed: {error}"
                ))
            })?;
        let correction =
            self.coefficient_correction(orientation.time_scales(), orientation.polar_motion())?;
        let tide = SphericalHarmonicGravity::from_normalized_coefficients(
            self.mu_earth_km3_s2,
            self.reference_radius_km,
            POLE_TIDE_MAX_DEGREE,
            POLE_TIDE_MAX_ORDER,
            &[correction],
            // A field of corrections holds no permanent tide of its own.
            TideSystem::TideFree,
        )?;
        let accel_itrf = tide.body_fixed_acceleration_km_s2(position_itrf_km)?;
        let accel_gcrf = orientation
            .itrf_to_gcrf_position_km(accel_itrf)
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth pole tide inertial acceleration rotation failed: {error}"
                ))
            })?;
        Ok(Vector3::new(accel_gcrf[0], accel_gcrf[1], accel_gcrf[2]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SunMoonItrfKm {
    sun_itrf_km: [f64; 3],
    moon_itrf_km: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BodyGeometry {
    radius_ratio: f64,
    radius_ratio_fraction: f64,
    radius_ratio_exponent: i32,
    cos_m: [f64; 4],
    sin_m: [f64; 4],
    p2: [f64; 3],
    p3: [f64; 4],
}

fn orientation_at_state(
    ctx: &PropagationContext,
    epoch_tdb_seconds: f64,
) -> Result<EarthOrientation, PropagationError> {
    let provider = ctx.body_fixed_frame_provider().ok_or_else(|| {
        PropagationError::InvalidInput(
            "solid Earth tide geopotential forces require a body-fixed frame provider".to_string(),
        )
    })?;
    provider
        .orientation_at_tdb_seconds(epoch_tdb_seconds)
        .map(|orientation| ctx.record_orientation(orientation))
        .map_err(|error| {
            PropagationError::from_frame(
                "solid Earth tide body-fixed frame evaluation failed",
                error,
            )
        })
}

fn sun_moon_itrf_km(orientation: &EarthOrientation) -> Result<SunMoonItrfKm, PropagationError> {
    // The orientation's provider already applied the UT1 policy and the
    // context recorded any departure, so its time scales are accepted as is.
    let bodies = with_ut1_validity(&orientation.time_scales(), ValidityMode::Permissive, |ts| {
        sun_moon_ecef_with_polar_motion(ts, orientation.polar_motion())
    })
    .map(|validated| validated.value)
    .map_err(|error| PropagationError::ForceModelFailure(format!("Sun/Moon: {error}")))?;
    Ok(SunMoonItrfKm {
        sun_itrf_km: meters_to_km(bodies.sun),
        moon_itrf_km: meters_to_km(bodies.moon),
    })
}

fn meters_to_km(position_m: [f64; 3]) -> [f64; 3] {
    [
        position_m[0] / M_PER_KM,
        position_m[1] / M_PER_KM,
        position_m[2] / M_PER_KM,
    ]
}

fn empty_solid_tide_coefficients() -> [SphericalHarmonicCoefficient; 10] {
    [
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 3,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 4,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 4,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 4,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
    ]
}

/// Part of the Step 1 `C20` correction that a geopotential in `tide_system`
/// already holds, to be subtracted (Section 6.2.2).
///
/// The time average of the Step 1 `C20` correction is the permanent
/// deformation `A0 H0 k20` of Equation (6.14). A zero-tide `C20` holds it,
/// so Equation (6.13) subtracts it. A mean-tide `C20` also holds the permanent
/// part of the tide-generating potential, `A0 H0` (Chapter 1, Section 1.1).
/// That part is not the Earth's own field: the tide-generating potential of
/// an external body grows with distance from the geocentre as `r^2`, while a
/// `C20` term of the geopotential falls as `r^-3`, so folded into `C20` it is
/// right only at the reference radius and wrong everywhere a satellite flies.
/// It is subtracted as well, whether or not a third-body force supplies the
/// external potential with its own `r^2` dependence. A tide-free `C20` holds
/// neither part. The products are formed
/// as Orekit forms its IERS 2010 permanent tide, `(A0 * H0) * k20`.
fn permanent_tide_c20(tide_system: TideSystem) -> f64 {
    let potential = SOLID_EARTH_TIDE_A0_PER_M * PERMANENT_TIDE_H0_M;
    let deformation = potential * SOLID_EARTH_TIDE_K20_REAL;
    match tide_system {
        TideSystem::TideFree => 0.0,
        TideSystem::ZeroTide => deformation,
        TideSystem::MeanTide => deformation + potential,
    }
}

/// One tidal constituent of IERS Conventions (2010) Tables 6.5a-c.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FrequencyDependentTerm {
    /// Doodson number as printed without its comma, e.g. 165555 for K1. Only
    /// the transcription test reads it, against the multipliers.
    #[cfg_attr(not(test), allow(dead_code))]
    doodson_number: u32,
    /// Multipliers of the Doodson arguments (tau, s, h, p, N', ps). The first is
    /// the order `m` of the coefficient the term corrects.
    doodson: [i8; 6],
    /// Multipliers `N` of the Delaunay arguments (l, l', F, D, Omega), with the
    /// argument `theta_f = m (theta_g + pi) - N . F`.
    delaunay: [i8; 5],
    /// In-phase amplitude, units of 1e-12: `A0 Hf dkf_R` in Table 6.5b,
    /// `A1 dkf_R Hf` in Table 6.5a, `A2 dkf Hf` in Table 6.5c.
    in_phase: f64,
    /// Out-of-phase amplitude, units of 1e-12: `A0 Hf dkf_I` in Table 6.5b,
    /// `A1 dkf_I Hf` in Table 6.5a, and zero in Table 6.5c.
    out_of_phase: f64,
}

const fn term(
    doodson_number: u32,
    doodson: [i8; 6],
    delaunay: [i8; 5],
    in_phase: f64,
    out_of_phase: f64,
) -> FrequencyDependentTerm {
    FrequencyDependentTerm {
        doodson_number,
        doodson,
        delaunay,
        in_phase,
        out_of_phase,
    }
}

/// Unit of the amplitudes in Tables 6.5a-c.
const TABLE_6_5_UNIT: f64 = 1.0e-12;

/// IERS Conventions (2010) Table 6.5b: zonal (long-period) tides, `k20`.
#[rustfmt::skip]
const K20_TERMS: [FrequencyDependentTerm; 21] = [
    term( 55565, [ 0,  0,  0,  0,  1,  0], [ 0,  0,  0,  0,  1],   16.6,  -6.7),
    term( 55575, [ 0,  0,  0,  0,  2,  0], [ 0,  0,  0,  0,  2],   -0.1,   0.1),
    term( 56554, [ 0,  0,  1,  0,  0, -1], [ 0, -1,  0,  0,  0],   -1.2,   0.8), // Sa
    term( 57555, [ 0,  0,  2,  0,  0,  0], [ 0,  0, -2,  2, -2],   -5.5,   4.3), // Ssa
    term( 57565, [ 0,  0,  2,  0,  1,  0], [ 0,  0, -2,  2, -1],    0.1,  -0.1),
    term( 58554, [ 0,  0,  3,  0,  0, -1], [ 0, -1, -2,  2, -2],   -0.3,   0.2),
    term( 63655, [ 0,  1, -2,  1,  0,  0], [ 1,  0,  0, -2,  0],   -0.3,   0.7), // Msm
    term( 65445, [ 0,  1,  0, -1, -1,  0], [-1,  0,  0,  0, -1],    0.1,  -0.2),
    term( 65455, [ 0,  1,  0, -1,  0,  0], [-1,  0,  0,  0,  0],   -1.2,   3.7), // Mm
    term( 65465, [ 0,  1,  0, -1,  1,  0], [-1,  0,  0,  0,  1],    0.1,  -0.2),
    term( 65655, [ 0,  1,  0,  1,  0,  0], [ 1,  0, -2,  0, -2],    0.1,  -0.2),
    term( 73555, [ 0,  2, -2,  0,  0,  0], [ 0,  0,  0, -2,  0],    0.0,   0.6), // Msf
    term( 75355, [ 0,  2,  0, -2,  0,  0], [-2,  0,  0,  0,  0],    0.0,   0.3),
    term( 75555, [ 0,  2,  0,  0,  0,  0], [ 0,  0, -2,  0, -2],    0.6,   6.3), // Mf
    term( 75565, [ 0,  2,  0,  0,  1,  0], [ 0,  0, -2,  0, -1],    0.2,   2.6),
    term( 75575, [ 0,  2,  0,  0,  2,  0], [ 0,  0, -2,  0,  0],    0.0,   0.2),
    term( 83655, [ 0,  3, -2,  1,  0,  0], [ 1,  0, -2, -2, -2],    0.1,   0.2), // Mstm
    term( 85455, [ 0,  3,  0, -1,  0,  0], [-1,  0, -2,  0, -2],    0.4,   1.1), // Mtm
    term( 85465, [ 0,  3,  0, -1,  1,  0], [-1,  0, -2,  0, -1],    0.2,   0.5),
    term( 93555, [ 0,  4, -2,  0,  0,  0], [ 0,  0, -2, -2, -2],    0.1,   0.2), // Msqm
    term( 95355, [ 0,  4,  0, -2,  0,  0], [-2,  0, -2,  0, -2],    0.1,   0.1), // Mqm
];

/// IERS Conventions (2010) Table 6.5a: diurnal tides, `k21`.
#[rustfmt::skip]
const K21_TERMS: [FrequencyDependentTerm; 48] = [
    term(125755, [ 1, -3,  0,  2,  0,  0], [ 2,  0,  2,  0,  2],   -0.1,   0.0), // 2Q1
    term(127555, [ 1, -3,  2,  0,  0,  0], [ 0,  0,  2,  2,  2],   -0.1,   0.0), // sigma1
    term(135645, [ 1, -2,  0,  1, -1,  0], [ 1,  0,  2,  0,  1],   -0.1,   0.0),
    term(135655, [ 1, -2,  0,  1,  0,  0], [ 1,  0,  2,  0,  2],   -0.7,   0.1), // Q1
    term(137455, [ 1, -2,  2, -1,  0,  0], [-1,  0,  2,  2,  2],   -0.1,   0.0), // rho1
    term(145545, [ 1, -1,  0,  0, -1,  0], [ 0,  0,  2,  0,  1],   -1.3,   0.1),
    term(145555, [ 1, -1,  0,  0,  0,  0], [ 0,  0,  2,  0,  2],   -6.8,   0.6), // O1
    term(147555, [ 1, -1,  2,  0,  0,  0], [ 0,  0,  0,  2,  0],    0.1,   0.0), // tau1
    term(153655, [ 1,  0, -2,  1,  0,  0], [ 1,  0,  2, -2,  2],    0.1,   0.0), // Ntau1
    term(155445, [ 1,  0,  0, -1, -1,  0], [-1,  0,  2,  0,  1],    0.1,   0.0),
    term(155455, [ 1,  0,  0, -1,  0,  0], [-1,  0,  2,  0,  2],    0.4,   0.0), // Lk1
    term(155655, [ 1,  0,  0,  1,  0,  0], [ 1,  0,  0,  0,  0],    1.3,  -0.1), // No1
    term(155665, [ 1,  0,  0,  1,  1,  0], [ 1,  0,  0,  0,  1],    0.3,   0.0),
    term(157455, [ 1,  0,  2, -1,  0,  0], [-1,  0,  0,  2,  0],    0.3,   0.0), // chi1
    term(157465, [ 1,  0,  2, -1,  1,  0], [-1,  0,  0,  2,  1],    0.1,   0.0),
    term(162556, [ 1,  1, -3,  0,  0,  1], [ 0,  1,  2, -2,  2],   -1.9,   0.1), // pi1
    term(163545, [ 1,  1, -2,  0, -1,  0], [ 0,  0,  2, -2,  1],    0.5,   0.0),
    term(163555, [ 1,  1, -2,  0,  0,  0], [ 0,  0,  2, -2,  2],  -43.4,   2.9), // P1
    term(164554, [ 1,  1, -1,  0,  0, -1], [ 0, -1,  2, -2,  2],    0.6,   0.0),
    term(164556, [ 1,  1, -1,  0,  0,  1], [ 0,  1,  0,  0,  0],    1.6,  -0.1), // S1
    term(165345, [ 1,  1,  0, -2, -1,  0], [-2,  0,  2,  0,  1],    0.1,   0.0),
    term(165535, [ 1,  1,  0,  0, -2,  0], [ 0,  0,  0,  0, -2],    0.1,   0.0),
    term(165545, [ 1,  1,  0,  0, -1,  0], [ 0,  0,  0,  0, -1],   -8.8,   0.5),
    term(165555, [ 1,  1,  0,  0,  0,  0], [ 0,  0,  0,  0,  0],  470.9, -30.2), // K1
    term(165565, [ 1,  1,  0,  0,  1,  0], [ 0,  0,  0,  0,  1],   68.1,  -4.6),
    term(165575, [ 1,  1,  0,  0,  2,  0], [ 0,  0,  0,  0,  2],   -1.6,   0.1),
    term(166455, [ 1,  1,  1, -1,  0,  0], [-1,  0,  0,  1,  0],    0.1,   0.0),
    term(166544, [ 1,  1,  1,  0, -1, -1], [ 0, -1,  0,  0, -1],   -0.1,   0.0),
    term(166554, [ 1,  1,  1,  0,  0, -1], [ 0, -1,  0,  0,  0],  -20.6,  -0.3), // psi1
    term(166556, [ 1,  1,  1,  0,  0,  1], [ 0,  1, -2,  2, -2],    0.3,   0.0),
    term(166564, [ 1,  1,  1,  0,  1, -1], [ 0, -1,  0,  0,  1],   -0.3,   0.0),
    term(167355, [ 1,  1,  2, -2,  0,  0], [-2,  0,  0,  2,  0],   -0.2,   0.0),
    term(167365, [ 1,  1,  2, -2,  1,  0], [-2,  0,  0,  2,  1],   -0.1,   0.0),
    term(167555, [ 1,  1,  2,  0,  0,  0], [ 0,  0, -2,  2, -2],   -5.0,   0.3), // phi1
    term(167565, [ 1,  1,  2,  0,  1,  0], [ 0,  0, -2,  2, -1],    0.2,   0.0),
    term(168554, [ 1,  1,  3,  0,  0, -1], [ 0, -1, -2,  2, -2],   -0.2,   0.0),
    term(173655, [ 1,  2, -2,  1,  0,  0], [ 1,  0,  0, -2,  0],   -0.5,   0.0), // theta1
    term(173665, [ 1,  2, -2,  1,  1,  0], [ 1,  0,  0, -2,  1],   -0.1,   0.0),
    term(175445, [ 1,  2,  0, -1, -1,  0], [-1,  0,  0,  0, -1],    0.1,   0.0),
    term(175455, [ 1,  2,  0, -1,  0,  0], [-1,  0,  0,  0,  0],   -2.1,   0.1), // J1
    term(175465, [ 1,  2,  0, -1,  1,  0], [-1,  0,  0,  0,  1],   -0.4,   0.0),
    term(183555, [ 1,  3, -2,  0,  0,  0], [ 0,  0,  0, -2,  0],   -0.2,   0.0), // So1
    term(185355, [ 1,  3,  0, -2,  0,  0], [-2,  0,  0,  0,  0],   -0.1,   0.0),
    term(185555, [ 1,  3,  0,  0,  0,  0], [ 0,  0, -2,  0, -2],   -0.6,   0.0), // Oo1
    term(185565, [ 1,  3,  0,  0,  1,  0], [ 0,  0, -2,  0, -1],   -0.4,   0.0),
    term(185575, [ 1,  3,  0,  0,  2,  0], [ 0,  0, -2,  0,  0],   -0.1,   0.0),
    term(195455, [ 1,  4,  0, -1,  0,  0], [-1,  0, -2,  0, -2],   -0.1,   0.0), // nu1
    term(195465, [ 1,  4,  0, -1,  1,  0], [-1,  0, -2,  0, -1],   -0.1,   0.0),
];

/// IERS Conventions (2010) Table 6.5c: semidiurnal tides, `k22`; the corrections are
/// only to the real part, so the out-of-phase amplitudes are zero.
#[rustfmt::skip]
const K22_TERMS: [FrequencyDependentTerm; 2] = [
    term(245655, [ 2, -1,  0,  1,  0,  0], [ 1,  0,  2,  0,  2],   -0.3,   0.0), // N2
    term(255555, [ 2,  0,  0,  0,  0,  0], [ 0,  0,  2,  0,  2],   -1.2,   0.0), // M2
];

/// Step 2 corrections at the instant of `time_scales`, with `theta_g` the
/// Greenwich mean sidereal time and the Delaunay arguments evaluated in Julian
/// centuries of TT from J2000.0. The time scales are accepted with whatever UT1
/// they carry: the caller's orientation provider has already applied its UT1
/// policy to them.
fn frequency_dependent_corrections_at(
    time_scales: &TimeScales,
) -> Result<[SphericalHarmonicCoefficient; 3], PropagationError> {
    let arguments = frequency_dependent_arguments_at(time_scales)?;
    Ok(frequency_dependent_corrections(
        arguments[0],
        [
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
            arguments[5],
        ],
    ))
}

fn frequency_dependent_arguments_at(
    time_scales: &TimeScales,
) -> Result<[f64; 6], PropagationError> {
    let gmst_rad = with_ut1_validity(
        time_scales,
        ValidityMode::Permissive,
        greenwich_mean_sidereal_time_radians,
    )
    .map(|validated| validated.value)
    .map_err(|error| PropagationError::from_frame("solid Earth tide sidereal time", error))?;
    let t = ((time_scales.jd_whole - J2000_JD) + time_scales.tt_fraction) / DAYS_PER_JULIAN_CENTURY;
    let delaunay_rad = iers_2010_solid_tide_arguments(t).map_err(|error| {
        PropagationError::ForceModelFailure(format!(
            "solid Earth tide fundamental arguments: {error}"
        ))
    })?;
    Ok([
        gmst_rad + PI,
        delaunay_rad[0],
        delaunay_rad[1],
        delaunay_rad[2],
        delaunay_rad[3],
        delaunay_rad[4],
    ])
}

/// Step 2 corrections to normalized `C20`, `C21`/`S21` and `C22`/`S22`, in
/// that order, for `gamma_rad = theta_g + pi` and the Delaunay arguments
/// (l, l', F, D, Omega) in radians.
///
/// Equation (6.8a) gives `dC20 = sum(ip cos(theta_f) - op sin(theta_f))`.
/// Equation (6.8b) with `eta_1 = -i` gives
/// `dC21 = sum(ip sin(theta_f) + op cos(theta_f))` and
/// `dS21 = sum(ip cos(theta_f) - op sin(theta_f))`, and with `eta_2 = 1` gives
/// `dC22 = sum(ip cos(theta_f) - op sin(theta_f))` and
/// `dS22 = -sum(ip sin(theta_f) + op cos(theta_f))`.
fn frequency_dependent_corrections(
    gamma_rad: f64,
    delaunay_rad: [f64; 5],
) -> [SphericalHarmonicCoefficient; 3] {
    let mut corrections = [
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
    ];
    for (index, terms) in [&K20_TERMS[..], &K21_TERMS[..], &K22_TERMS[..]]
        .into_iter()
        .enumerate()
    {
        for term in terms {
            let (dc, ds) = frequency_dependent_term(term, gamma_rad, delaunay_rad);
            corrections[index].c += dc;
            corrections[index].s += ds;
        }
    }
    corrections
}

/// Contribution `(dC2m, dS2m)` of one constituent, `m` being its first Doodson
/// multiplier.
fn frequency_dependent_term(
    term: &FrequencyDependentTerm,
    gamma_rad: f64,
    delaunay_rad: [f64; 5],
) -> (f64, f64) {
    let order = term.doodson[0];
    let mut theta = f64::from(order) * gamma_rad;
    for (multiplier, argument) in term.delaunay.iter().zip(delaunay_rad) {
        theta -= f64::from(*multiplier) * argument;
    }
    let sin_theta = libm::sin(theta);
    let cos_theta = libm::cos(theta);
    let in_phase = term.in_phase * TABLE_6_5_UNIT;
    let out_of_phase = term.out_of_phase * TABLE_6_5_UNIT;
    match order {
        0 => (in_phase * cos_theta - out_of_phase * sin_theta, 0.0),
        1 => (
            in_phase * sin_theta + out_of_phase * cos_theta,
            in_phase * cos_theta - out_of_phase * sin_theta,
        ),
        _ => (
            in_phase * cos_theta - out_of_phase * sin_theta,
            -(in_phase * sin_theta + out_of_phase * cos_theta),
        ),
    }
}

fn add_body_tide_coefficients(
    model: &SolidEarthTideGravity,
    gm_body_km3_s2: f64,
    body_itrf_km: [f64; 3],
    corrections: &mut [SphericalHarmonicCoefficient; 10],
) -> Result<(), PropagationError> {
    let geometry = body_geometry(body_itrf_km, model.reference_radius_km)?;
    let gm_ratio = gm_body_km3_s2 / model.mu_earth_km3_s2;

    for order in 0..=2 {
        let love = DEGREE2_LOVE[order as usize];
        let radius_power = geometry.radius_ratio.powi(3);
        let gm_power = gm_ratio * radius_power;
        let base = gm_power * geometry.p2[order as usize] / 5.0;
        let ordinary_range = gm_ratio.is_normal()
            && geometry.radius_ratio.is_normal()
            && radius_power.is_normal()
            && gm_power.is_normal()
            && base.is_normal();
        let cosine = geometry.cos_m[order as usize];
        let sine = geometry.sin_m[order as usize];
        let (dc, ds) = if ordinary_range {
            coefficient_delta(base, love.primary, cosine, sine)
        } else {
            (
                scaled_body_product(
                    gm_body_km3_s2,
                    model.mu_earth_km3_s2,
                    &geometry,
                    3,
                    &[
                        geometry.p2[order as usize],
                        1.0 / 5.0,
                        love.primary.real * cosine + love.primary.imag * sine,
                    ],
                ),
                scaled_body_product(
                    gm_body_km3_s2,
                    model.mu_earth_km3_s2,
                    &geometry,
                    3,
                    &[
                        geometry.p2[order as usize],
                        1.0 / 5.0,
                        love.primary.real * sine - love.primary.imag * cosine,
                    ],
                ),
            )
        };
        let idx = order as usize;
        corrections[idx].c += dc;
        corrections[idx].s += ds;

        let dc4 = if ordinary_range {
            base * love.plus * cosine
        } else {
            scaled_body_product(
                gm_body_km3_s2,
                model.mu_earth_km3_s2,
                &geometry,
                3,
                &[geometry.p2[order as usize], 1.0 / 5.0, love.plus, cosine],
            )
        };
        let ds4 = if ordinary_range {
            base * love.plus * sine
        } else {
            scaled_body_product(
                gm_body_km3_s2,
                model.mu_earth_km3_s2,
                &geometry,
                3,
                &[geometry.p2[order as usize], 1.0 / 5.0, love.plus, sine],
            )
        };
        let idx4 = 7 + order as usize;
        corrections[idx4].c += dc4;
        corrections[idx4].s += ds4;
    }

    for order in 0..=3 {
        let love = DEGREE3_LOVE[order as usize];
        let radius_power = geometry.radius_ratio.powi(4);
        let gm_power = gm_ratio * radius_power;
        let base = gm_power * geometry.p3[order as usize] / 7.0;
        let ordinary_range = gm_ratio.is_normal()
            && geometry.radius_ratio.is_normal()
            && radius_power.is_normal()
            && gm_power.is_normal()
            && base.is_normal();
        let cosine = geometry.cos_m[order as usize];
        let sine = geometry.sin_m[order as usize];
        let (dc, ds) = if ordinary_range {
            coefficient_delta(base, love, cosine, sine)
        } else {
            (
                scaled_body_product(
                    gm_body_km3_s2,
                    model.mu_earth_km3_s2,
                    &geometry,
                    4,
                    &[
                        geometry.p3[order as usize],
                        1.0 / 7.0,
                        love.real * cosine + love.imag * sine,
                    ],
                ),
                scaled_body_product(
                    gm_body_km3_s2,
                    model.mu_earth_km3_s2,
                    &geometry,
                    4,
                    &[
                        geometry.p3[order as usize],
                        1.0 / 7.0,
                        love.real * sine - love.imag * cosine,
                    ],
                ),
            )
        };
        let idx = 3 + order as usize;
        corrections[idx].c += dc;
        corrections[idx].s += ds;
    }

    Ok(())
}

fn coefficient_delta(
    base: f64,
    love: ComplexLoveNumber,
    cos_m_lambda: f64,
    sin_m_lambda: f64,
) -> (f64, f64) {
    (
        base * (love.real * cos_m_lambda + love.imag * sin_m_lambda),
        base * (love.real * sin_m_lambda - love.imag * cos_m_lambda),
    )
}

fn body_geometry(
    position_km: [f64; 3],
    reference_radius_km: f64,
) -> Result<BodyGeometry, PropagationError> {
    validate_vec3(position_km, "body position")?;
    let scale = position_km
        .into_iter()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "zero tide-raising body position magnitude".to_string(),
        ));
    }
    let x = position_km[0] / scale;
    let y = position_km[1] / scale;
    let z = position_km[2] / scale;
    let rho_scaled = (x * x + y * y).sqrt();
    let radius_scaled = (rho_scaled * rho_scaled + z * z).sqrt();
    let quotient = reference_radius_km / scale;
    let staged_ratio = quotient / radius_scaled;
    let radius_ratio = if staged_ratio.is_finite() {
        staged_ratio
    } else {
        let scaled_radius = scale * radius_scaled;
        if scaled_radius.is_finite() {
            reference_radius_km / scaled_radius
        } else {
            staged_ratio
        }
    };
    let (reference_fraction, reference_exponent) = libm::frexp(reference_radius_km);
    let (scale_fraction, scale_exponent) = libm::frexp(scale);
    let (radius_fraction, radius_exponent) = libm::frexp(radius_scaled);
    let (radius_ratio_fraction, radius_ratio_adjustment) =
        libm::frexp((reference_fraction / scale_fraction) / radius_fraction);
    let radius_ratio_exponent =
        reference_exponent - scale_exponent - radius_exponent + radius_ratio_adjustment;
    let sin_lat = z / radius_scaled;
    let cos_lat = rho_scaled / radius_scaled;
    let (cos_m, sin_m) = longitude_trig(x, y, rho_scaled);
    let p2 = degree2_legendre(sin_lat, cos_lat);
    let p3 = degree3_legendre(sin_lat, cos_lat);
    Ok(BodyGeometry {
        radius_ratio,
        radius_ratio_fraction,
        radius_ratio_exponent,
        cos_m,
        sin_m,
        p2,
        p3,
    })
}

fn scaled_body_product(
    gm_body_km3_s2: f64,
    mu_earth_km3_s2: f64,
    geometry: &BodyGeometry,
    degree: i32,
    factors: &[f64],
) -> f64 {
    let sign = factors
        .iter()
        .filter(|factor| factor.is_sign_negative())
        .count();
    if factors.contains(&0.0) {
        return if sign % 2 == 0 { 0.0 } else { -0.0 };
    }

    let (gm_fraction, gm_exponent) = libm::frexp(gm_body_km3_s2);
    let (mu_fraction, mu_exponent) = libm::frexp(mu_earth_km3_s2);
    let mut fraction = (gm_fraction / mu_fraction) * geometry.radius_ratio_fraction.powi(degree);
    let mut exponent = gm_exponent - mu_exponent + degree * geometry.radius_ratio_exponent;
    for factor in factors {
        let (factor_fraction, factor_exponent) = libm::frexp(factor.abs());
        fraction *= factor_fraction;
        exponent += factor_exponent;
    }
    let (fraction, adjustment) = libm::frexp(fraction);
    let product = libm::scalbn(fraction, exponent + adjustment);
    if sign % 2 == 0 {
        product
    } else {
        -product
    }
}

fn longitude_trig(x: f64, y: f64, rho: f64) -> ([f64; 4], [f64; 4]) {
    let mut cos_m = [0.0_f64; 4];
    let mut sin_m = [0.0_f64; 4];
    cos_m[0] = 1.0;
    if rho == 0.0 {
        cos_m[1] = 1.0;
    } else {
        cos_m[1] = x / rho;
        sin_m[1] = y / rho;
    }
    for order in 2..=3 {
        cos_m[order] = cos_m[order - 1] * cos_m[1] - sin_m[order - 1] * sin_m[1];
        sin_m[order] = sin_m[order - 1] * cos_m[1] + cos_m[order - 1] * sin_m[1];
    }
    (cos_m, sin_m)
}

fn degree2_legendre(sin_lat: f64, cos_lat: f64) -> [f64; 3] {
    let sqrt5 = 5.0_f64.sqrt();
    let sqrt15 = 15.0_f64.sqrt();
    [
        0.5 * sqrt5 * (3.0 * sin_lat * sin_lat - 1.0),
        sqrt15 * sin_lat * cos_lat,
        0.5 * sqrt15 * cos_lat * cos_lat,
    ]
}

fn degree3_legendre(sin_lat: f64, cos_lat: f64) -> [f64; 4] {
    let sqrt7 = 7.0_f64.sqrt();
    let sqrt42 = 42.0_f64.sqrt();
    let sqrt70 = 70.0_f64.sqrt();
    let sqrt105 = 105.0_f64.sqrt();
    let sin2 = sin_lat * sin_lat;
    let cos2 = cos_lat * cos_lat;
    [
        0.5 * sqrt7 * (5.0 * sin_lat * sin2 - 3.0 * sin_lat),
        0.25 * sqrt42 * cos_lat * (5.0 * sin2 - 1.0),
        0.5 * sqrt105 * sin_lat * cos2,
        0.25 * sqrt70 * cos_lat * cos2,
    ]
}

fn wobble_arcsec(
    time_scales: TimeScales,
    polar_motion: PolarMotion,
) -> Result<(f64, f64), PropagationError> {
    let xp_arcsec = polar_motion.xp_rad / ARCSEC_TO_RAD;
    let yp_arcsec = polar_motion.yp_rad / ARCSEC_TO_RAD;
    let (mean_x_arcsec, mean_y_arcsec) = conventional_mean_pole_arcsec(time_scales)?;
    Ok((xp_arcsec - mean_x_arcsec, -(yp_arcsec - mean_y_arcsec)))
}

fn conventional_mean_pole_arcsec(time_scales: TimeScales) -> Result<(f64, f64), PropagationError> {
    if !time_scales.jd_tt.is_finite() {
        return Err(PropagationError::InvalidInput(
            "time_scales.jd_tt must be finite".to_string(),
        ));
    }
    let year = 2000.0 + (time_scales.jd_tt - J2000_JD) / 365.25;
    let dt = year - 2000.0;
    let (x_mas, y_mas) = if year <= 2010.0 {
        (
            55.974 + dt * (1.8243 + dt * (0.18413 + dt * 0.007024)),
            346.346 + dt * (1.7896 + dt * (-0.10729 - dt * 0.000908)),
        )
    } else {
        (23.513 + dt * 7.6141, 358.891 - dt * 0.6287)
    };
    Ok((x_mas / 1000.0, y_mas / 1000.0))
}

fn validate_positive(value: f64, field: &'static str) -> Result<(), PropagationError> {
    if !value.is_finite() {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite"
        )));
    }
    if value <= 0.0 {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be positive"
        )));
    }
    Ok(())
}

fn validate_vec3(values: [f64; 3], field: &'static str) -> Result<(), PropagationError> {
    for value in values {
        if !value.is_finite() {
            return Err(PropagationError::InvalidInput(format!(
                "{field} components must be finite"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::frames::orientation::TdbEarthOrientationProvider;
    use crate::astro::time::scales::TimeScales;
    use serde::Deserialize;
    use std::sync::Arc;

    const COEFFICIENT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tides/geopotential_tide_coefficients.json");

    #[derive(Debug, Deserialize)]
    struct CoefficientFixture {
        rows: Vec<FixtureRow>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureRow {
        id: String,
        source: String,
        sun_itrf_km: Option<[f64; 3]>,
        moon_itrf_km: Option<[f64; 3]>,
        earth_mu_km3_s2: Option<f64>,
        earth_radius_km: Option<f64>,
        gm_sun_km3_s2: Option<f64>,
        gm_moon_km3_s2: Option<f64>,
        jd_tt: Option<f64>,
        xp_arcsec: Option<f64>,
        yp_arcsec: Option<f64>,
        utc: Option<String>,
        max_abs_error: Option<f64>,
        coefficients: Vec<FixtureCoefficient>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCoefficient {
        degree: u16,
        order: u16,
        c: f64,
        s: f64,
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn coefficient(
        coefficients: &[SphericalHarmonicCoefficient],
        degree: u16,
        order: u16,
    ) -> &SphericalHarmonicCoefficient {
        coefficients
            .iter()
            .find(|coefficient| coefficient.degree == degree && coefficient.order == order)
            .expect("coefficient present")
    }

    fn fixture_row(id: &str) -> FixtureRow {
        let fixture: CoefficientFixture =
            serde_json::from_str(COEFFICIENT_FIXTURE).expect("parse coefficient fixture");
        fixture
            .rows
            .into_iter()
            .find(|row| row.id == id)
            .expect("fixture row present")
    }

    #[test]
    fn love_numbers_match_iers_table_6_3() {
        assert_eq!(SOLID_EARTH_TIDE_K20_REAL.to_bits(), 0.30190_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K20_IMAG.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            SOLID_EARTH_TIDE_K20_PLUS.to_bits(),
            (-0.00089_f64).to_bits()
        );
        assert_eq!(SOLID_EARTH_TIDE_K21_REAL.to_bits(), 0.29830_f64.to_bits());
        assert_eq!(
            SOLID_EARTH_TIDE_K21_IMAG.to_bits(),
            (-0.00144_f64).to_bits()
        );
        assert_eq!(
            SOLID_EARTH_TIDE_K21_PLUS.to_bits(),
            (-0.00080_f64).to_bits()
        );
        let expected_k22_real: f64 = "0.30102".parse().expect("published k22 decimal");
        assert_eq!(
            SOLID_EARTH_TIDE_K22_REAL.to_bits(),
            expected_k22_real.to_bits()
        );
        assert_eq!(
            SOLID_EARTH_TIDE_K22_IMAG.to_bits(),
            (-0.00130_f64).to_bits()
        );
        assert_eq!(
            SOLID_EARTH_TIDE_K22_PLUS.to_bits(),
            (-0.00057_f64).to_bits()
        );
        assert_eq!(SOLID_EARTH_TIDE_K30_REAL.to_bits(), 0.093_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K31_REAL.to_bits(), 0.093_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K32_REAL.to_bits(), 0.093_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K33_REAL.to_bits(), 0.094_f64.to_bits());
    }

    #[test]
    fn step1_axis_aligned_body_matches_iers_equation_6_6_fixture() {
        let row = fixture_row("solid_earth_step1_greenwich_equator_unit_body");
        assert!(row.source.contains("Eq. (6.6)"));
        let model = SolidEarthTideGravity::new(
            row.earth_mu_km3_s2.expect("earth mu"),
            row.earth_radius_km.expect("earth radius"),
            row.gm_sun_km3_s2.expect("sun gm"),
            row.gm_moon_km3_s2.expect("moon gm"),
        );
        let coefficients = model
            .coefficient_corrections_for_body_fixed_bodies(
                row.sun_itrf_km.expect("sun position"),
                row.moon_itrf_km.expect("moon position"),
            )
            .expect("coefficient corrections");

        for expected in row.coefficients {
            let actual = coefficient(&coefficients, expected.degree, expected.order);
            assert_close(actual.c, expected.c, 2.0e-17);
            assert_close(actual.s, expected.s, 2.0e-17);
        }
    }

    #[test]
    fn body_geometry_handles_extreme_finite_and_independent_normal_inputs() {
        let extreme = [f64::MAX, -f64::MAX, f64::MAX];
        let defaults = SolidEarthTideGravity::default();
        let geometry = body_geometry(extreme, defaults.reference_radius_km)
            .expect("finite extreme coordinates have representable ratios");
        assert!(geometry.radius_ratio.is_finite());
        assert!(geometry.p2.into_iter().all(f64::is_finite));
        assert!(geometry.p3.into_iter().all(f64::is_finite));

        let extreme_corrections = defaults
            .coefficient_corrections_for_body_fixed_bodies(extreme, [384_400.0, 0.0, 0.0])
            .expect("finite extreme body coordinates");
        assert!(extreme_corrections
            .iter()
            .all(|coefficient| coefficient.c.is_finite() && coefficient.s.is_finite()));

        let quotient_overflow = body_geometry([0.75, 0.75, 0.75], f64::MAX)
            .expect("representable radius ratio despite quotient overflow");
        assert!(quotient_overflow.radius_ratio.is_finite());

        let scaled_model = SolidEarthTideGravity::new(1.0e300, 1.0e150, 1.0, 1.0);
        let scaled = scaled_model
            .coefficient_corrections_for_body_fixed_bodies([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
            .expect("finite analytically representable extreme coefficients");
        let expected_c20 = 2.0e150 * 5.0_f64.sqrt() / 5.0 * SOLID_EARTH_TIDE_K20_REAL;
        let expected_c30 = 2.0e300 * 7.0_f64.sqrt() / 7.0 * SOLID_EARTH_TIDE_K30_REAL;
        assert_close(
            scaled[0].c,
            expected_c20,
            roundoff_gamma(32) * expected_c20.abs(),
        );
        assert_close(
            scaled[3].c,
            expected_c30,
            roundoff_gamma(32) * expected_c30.abs(),
        );

        let oracle: FieldOracle =
            serde_json::from_str(FIELD_ORACLE_FIXTURE).expect("parse SolidTidesField fixture");
        let epoch = oracle.epochs.first().expect("independent field epoch");
        let model = SolidEarthTideGravity::default();
        let step2 = frequency_dependent_corrections(
            epoch.gamma,
            [epoch.l, epoch.l_prime, epoch.f, epoch.d, epoch.omega],
        );
        let actual = model
            .corrections(epoch.sun_km, epoch.moon_km, Some(step2))
            .expect("normal independent body inputs");
        let tolerance = field_roundoff_bound(&model, epoch);
        for expected in &epoch.tide_free {
            let (c, s) = actual
                .iter()
                .find(|value| value.degree == expected.degree && value.order == expected.order)
                .map_or((0.0, 0.0), |value| (value.c, value.s));
            assert_close(c, expected.c, tolerance);
            assert_close(s, expected.s, tolerance);
        }
    }

    #[test]
    fn subnormal_radius_power_is_combined_with_large_gm_before_rounding() {
        let model = SolidEarthTideGravity::new(1.0, 1.0, 1.0e300, 1.0e300);
        let geometry = body_geometry([0.0, 0.0, 1.0e80], model.reference_radius_km)
            .expect("finite extreme body geometry");
        let gm_ratio = model.gm_sun_km3_s2 / model.mu_earth_km3_s2;
        let radius_power = geometry.radius_ratio.powi(4);
        let gm_power = gm_ratio * radius_power;
        let base = gm_power * geometry.p3[0] / 7.0;
        assert!(geometry.radius_ratio.is_normal());
        assert!(gm_ratio.is_normal());
        assert!(radius_power > 0.0 && !radius_power.is_normal());
        assert!(gm_power.is_normal());
        assert!(base.is_normal());
        let corrections = model
            .coefficient_corrections_for_body_fixed_bodies([0.0, 0.0, 1.0e80], [0.0, 0.0, 1.0e80])
            .expect("finite degree-two and degree-three corrections");

        let expected_c20 = 2.0e60 * 5.0_f64.sqrt() / 5.0 * SOLID_EARTH_TIDE_K20_REAL;
        let expected_c30 = 2.0e-20 * 7.0_f64.sqrt() / 7.0 * SOLID_EARTH_TIDE_K30_REAL;
        assert_close(
            corrections[0].c,
            expected_c20,
            roundoff_gamma(32) * expected_c20.abs(),
        );
        assert_close(
            corrections[3].c,
            expected_c30,
            roundoff_gamma(32) * expected_c30.abs(),
        );
    }

    #[test]
    fn pole_tide_coefficients_match_iers_chapter_6_equation() {
        let row = fixture_row("solid_earth_pole_tide_j2000_unit_wobble");
        assert!(row.source.contains("Table 7.7"));
        let time_scales = TimeScales {
            jd_whole: row.jd_tt.expect("jd_tt"),
            ut1_fraction: 0.0,
            tt_fraction: 0.0,
            tdb_fraction: 0.0,
            jd_ut1: row.jd_tt.expect("jd_tt"),
            jd_tt: row.jd_tt.expect("jd_tt"),
            jd_tdb: row.jd_tt.expect("jd_tt"),
            ut1_degraded: None,
        };
        let pole =
            PolarMotion::from_arcseconds(row.xp_arcsec.expect("xp"), row.yp_arcsec.expect("yp"))
                .expect("polar motion");
        let correction = SolidEarthPoleTideGravity::default()
            .coefficient_correction(time_scales, pole)
            .expect("pole tide coefficient");
        let expected = row.coefficients.first().expect("pole tide coefficient");

        assert_eq!(correction.degree, expected.degree);
        assert_eq!(correction.order, expected.order);
        assert_close(correction.c, expected.c, 1.0e-18);
        assert_close(correction.s, expected.s, 1.0e-18);
    }

    #[test]
    fn mean_pole_model_is_boundary_pinned_to_iers_table_7_7() {
        let at_2010 = TimeScales {
            jd_whole: J2000_JD + 10.0 * 365.25,
            ut1_fraction: 0.0,
            tt_fraction: 0.0,
            tdb_fraction: 0.0,
            jd_ut1: J2000_JD + 10.0 * 365.25,
            jd_tt: J2000_JD + 10.0 * 365.25,
            jd_tdb: J2000_JD + 10.0 * 365.25,
            ut1_degraded: None,
        };
        let after_2010 = TimeScales {
            jd_whole: J2000_JD + 11.0 * 365.25,
            ut1_fraction: 0.0,
            tt_fraction: 0.0,
            tdb_fraction: 0.0,
            jd_ut1: J2000_JD + 11.0 * 365.25,
            jd_tt: J2000_JD + 11.0 * 365.25,
            jd_tdb: J2000_JD + 11.0 * 365.25,
            ut1_degraded: None,
        };

        let (x_2010, y_2010) = conventional_mean_pole_arcsec(at_2010).expect("mean pole");
        let (x_2011, y_2011) = conventional_mean_pole_arcsec(after_2010).expect("mean pole");

        assert_close(x_2010, 0.099_654, 1.0e-15);
        assert_close(y_2010, 0.352_605, 1.0e-15);
        assert_close(x_2011, 0.107_268_1, 1.0e-15);
        assert_close(y_2011, 0.351_975_3, 1.0e-15);
    }

    #[test]
    fn step1_real_epoch_degree3_degree4_match_orekit_oracle() {
        let row = fixture_row("solid_earth_step1_orekit_2003_05_06_degree3_degree4");
        assert!(row.source.contains("Orekit"));
        assert_eq!(row.utc.as_deref(), Some("2003-05-06T13:43:32.125Z"));
        let scales =
            TimeScales::from_utc(2003, 5, 6, 13, 43, 32.125).expect("valid UTC time scales");
        let epoch_tdb_seconds = crate::astro::time::civil::j2000_seconds_from_split(
            scales.jd_whole,
            scales.tdb_fraction,
        );
        let ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()));
        let coefficients = SolidEarthTideGravity::default()
            .coefficient_corrections_at_epoch(epoch_tdb_seconds, &ctx)
            .expect("coefficient corrections");
        let tolerance = row.max_abs_error.expect("max_abs_error");

        for expected in row.coefficients {
            let actual = coefficient(&coefficients, expected.degree, expected.order);
            assert_close(actual.c, expected.c, tolerance);
            assert_close(actual.s, expected.s, tolerance);
        }
    }

    const STEP2_ORACLE_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tides/geopotential_tide_step2_orekit.json");

    #[derive(Debug, Deserialize)]
    struct Step2Oracle {
        epochs: Vec<Step2OracleEpoch>,
    }

    #[derive(Debug, Deserialize)]
    struct Step2OracleEpoch {
        jd_whole: f64,
        tt_fraction: f64,
        gamma: f64,
        l: f64,
        l_prime: f64,
        f: f64,
        d: f64,
        omega: f64,
        c20: f64,
        c21: f64,
        s21: f64,
        c22: f64,
        s22: f64,
    }

    fn step2_oracle() -> Vec<Step2OracleEpoch> {
        let oracle: Step2Oracle =
            serde_json::from_str(STEP2_ORACLE_FIXTURE).expect("parse Step 2 oracle fixture");
        assert_eq!(oracle.epochs.len(), 64);
        oracle.epochs
    }

    #[test]
    fn step2_uses_iers_equation_5_43_constant_pair_only_for_l_prime_and_d() {
        let arguments =
            iers_2010_solid_tide_arguments(0.0).expect("IERS 2010 tide arguments at J2000");
        let skyfield_arguments =
            crate::astro::frames::nutation::skyfield_fundamental_arguments(0.0)
                .expect("shared Skyfield arguments at J2000");
        let l_prime_arcseconds = 1_287_104.793_048_f64;
        let d_arcseconds = 1_072_260.703_692_f64;
        let l_prime_degrees = 357.529_109_18_f64;
        let d_degrees = 297.850_195_47_f64;
        let radians_per_degree = std::f64::consts::PI / 180.0;
        let expected_l_prime = l_prime_degrees * radians_per_degree;
        let expected_d = d_degrees * radians_per_degree;
        let arcsecond_path_magnitude =
            l_prime_arcseconds * ARCSEC_TO_RAD + d_arcseconds * ARCSEC_TO_RAD;
        let degree_path_magnitude =
            l_prime_degrees * radians_per_degree + d_degrees * radians_per_degree;
        let decimal_input_bound =
            0.5 * f64::EPSILON * (arcsecond_path_magnitude + degree_path_magnitude);
        let arcsecond_path_operation_bound = roundoff_gamma(2) * arcsecond_path_magnitude;
        let degree_path_operation_bound = roundoff_gamma(3) * degree_path_magnitude;
        let conversion_operation_bound =
            arcsecond_path_operation_bound + degree_path_operation_bound;
        let conversion_bound = decimal_input_bound + conversion_operation_bound;

        assert_close(arguments[1], expected_l_prime, conversion_bound);
        assert_close(arguments[3], expected_d, conversion_bound);
        assert_ne!(arguments[1].to_bits(), skyfield_arguments[1].to_bits());
        assert_ne!(arguments[3].to_bits(), skyfield_arguments[3].to_bits());
        for index in [0, 2, 4] {
            assert_eq!(
                arguments[index].to_bits(),
                skyfield_arguments[index].to_bits()
            );
        }
    }

    fn assert_step2_matches(
        actual: &[SphericalHarmonicCoefficient; 3],
        epoch: &Step2OracleEpoch,
        arguments: [f64; 6],
        argument_error: [f64; 6],
    ) -> f64 {
        assert_eq!(
            [
                (actual[0].degree, actual[0].order),
                (actual[1].degree, actual[1].order),
                (actual[2].degree, actual[2].order),
            ],
            [(2, 0), (2, 1), (2, 2)]
        );
        assert_eq!(actual[0].s, 0.0);
        let reference_arguments = [
            epoch.gamma,
            epoch.l,
            epoch.l_prime,
            epoch.f,
            epoch.d,
            epoch.omega,
        ];
        let mut maximum = 0.0_f64;
        for (index, terms) in [&K20_TERMS[..], &K21_TERMS[..], &K22_TERMS[..]]
            .into_iter()
            .enumerate()
        {
            let tolerance =
                step2_roundoff_bound(terms, arguments, reference_arguments, argument_error);
            let expected = [
                (epoch.c20, 0.0),
                (epoch.c21, epoch.s21),
                (epoch.c22, epoch.s22),
            ][index];
            for (computed, reference) in
                [(actual[index].c, expected.0), (actual[index].s, expected.1)]
            {
                let error = (computed - reference).abs();
                assert!(
                    error <= tolerance,
                    "order {index}: error {error:e}, bound {tolerance:e}"
                );
                maximum = maximum.max(error);
            }
        }
        maximum
    }

    fn roundoff_gamma(operations: usize) -> f64 {
        let accumulated = operations as f64 * (f64::EPSILON / 2.0);
        accumulated / (1.0 - accumulated)
    }

    fn angular_difference_bounds(left: f64, right: f64) -> (f64, f64) {
        let tau = 2.0 * PI;
        let turns = ((left - right).abs() / tau).ceil();
        let difference = (left - right).abs() % tau;
        let difference = difference.min(tau - difference);
        let reduction_error = roundoff_gamma(4) * (left.abs() + right.abs() + turns * tau);
        let lower = (difference - reduction_error).max(0.0);
        let lower = if lower == 0.0 {
            0.0
        } else {
            f64::from_bits(lower.to_bits() - 1)
        };
        let upper = difference + reduction_error;
        let upper = if upper == 0.0 {
            0.0
        } else {
            f64::from_bits(upper.to_bits() + 1)
        };
        (lower, upper)
    }

    #[test]
    fn circular_argument_guard_accounts_for_its_own_roundoff() {
        let full_turns = 1024.0 * 2.0 * PI;
        let (equivalent_lower, equivalent_upper) = angular_difference_bounds(0.0, full_turns);
        assert_eq!(equivalent_lower, 0.0);
        assert!(equivalent_upper > 0.0);
        let (shifted_lower, shifted_upper) = angular_difference_bounds(0.125, full_turns);
        assert!(shifted_lower > 0.12);
        assert!(shifted_upper < 0.13);
    }

    fn step2_roundoff_bound(
        terms: &[FrequencyDependentTerm],
        arguments: [f64; 6],
        reference_arguments: [f64; 6],
        argument_error: [f64; 6],
    ) -> f64 {
        terms
            .iter()
            .map(|term| {
                let multipliers = [
                    term.doodson[0],
                    term.delaunay[0],
                    term.delaunay[1],
                    term.delaunay[2],
                    term.delaunay[3],
                    term.delaunay[4],
                ];
                let mut phase_magnitude = 0.0;
                let mut phase_error = 0.0;
                for ((multiplier, argument), error) in
                    multipliers.into_iter().zip(arguments).zip(argument_error)
                {
                    phase_magnitude += f64::from(multiplier).abs() * argument.abs();
                    phase_error += f64::from(multiplier).abs() * error;
                }
                let reference_phase_magnitude = multipliers
                    .into_iter()
                    .zip(reference_arguments)
                    .map(|(multiplier, argument)| f64::from(multiplier).abs() * argument.abs())
                    .sum::<f64>();
                phase_magnitude = phase_magnitude.max(reference_phase_magnitude);
                let amplitude = (term.in_phase.abs() + term.out_of_phase.abs()) * TABLE_6_5_UNIT;
                amplitude
                    * (phase_error
                        + 2.0 * roundoff_gamma(11) * phase_magnitude
                        + 2.0 * roundoff_gamma(terms.len() + 8)
                        + 4.0 * f64::EPSILON)
            })
            .sum()
    }

    #[test]
    fn step2_tables_match_iers_tables_6_5() {
        assert_eq!(K20_TERMS.len(), 21);
        assert_eq!(K21_TERMS.len(), 48);
        assert_eq!(K22_TERMS.len(), 2);
        for (order, terms) in [
            (0, &K20_TERMS[..]),
            (1, &K21_TERMS[..]),
            (2, &K22_TERMS[..]),
        ] {
            for term in terms {
                let n = term.doodson.map(i32::from);
                // The Doodson number spells n1 and then n2..n6 each plus 5.
                let spelled = n[1..]
                    .iter()
                    .fold(n[0], |number, multiplier| number * 10 + multiplier + 5);
                assert_eq!(spelled as u32, term.doodson_number);
                assert_eq!(n[0], order, "{}", term.doodson_number);
                // With tau = theta_g + pi - s, s = F + Omega, h = s - D,
                // p = s - l, N' = -Omega and ps = h - l', the Doodson argument
                // equals m (theta_g + pi) - N . F for the table's Delaunay
                // multipliers N.
                let from_doodson = [
                    -n[3],
                    -n[5],
                    -n[0] + n[1] + n[2] + n[3] + n[5],
                    -n[2] - n[5],
                    -n[0] + n[1] + n[2] + n[3] - n[4] + n[5],
                ];
                assert_eq!(
                    from_doodson,
                    term.delaunay.map(|multiplier| -i32::from(multiplier)),
                    "{}",
                    term.doodson_number
                );
            }
        }
        assert!(K22_TERMS.iter().all(|term| term.out_of_phase == 0.0));
    }

    #[test]
    fn step2_k1_term_matches_the_worked_example_of_section_6_2_1() {
        // Section 6.2.1: for K1, theta_f = theta_g + pi and
        // dC21 = 470.9e-12 sin(theta_g + pi) - 30.2e-12 cos(theta_g + pi),
        // dS21 = 470.9e-12 cos(theta_g + pi) + 30.2e-12 sin(theta_g + pi).
        let k1 = K21_TERMS
            .iter()
            .find(|term| term.doodson_number == 165_555)
            .expect("K1 row");
        for gamma in [0.0, 0.7, 2.5, -4.0, 1234.5] {
            let (dc21, ds21) = frequency_dependent_term(k1, gamma, [3.1, -0.4, 17.0, 2.2, -9.9]);
            let expected_c = 470.9e-12 * libm::sin(gamma) - 30.2e-12 * libm::cos(gamma);
            let expected_s = 470.9e-12 * libm::cos(gamma) + 30.2e-12 * libm::sin(gamma);
            assert_close(dc21, expected_c, 1.0e-24);
            assert_close(ds21, expected_s, 1.0e-24);
        }
    }

    #[test]
    fn step2_matches_orekit_for_the_same_arguments() {
        // The fixture's arguments are fed in as Orekit formed them, so the two
        // sides differ only in trigonometric rounding and summation order,
        // for sums of terms up to 5e-10; the smallest table entry is 1e-13.
        let mut max_dev = 0.0_f64;
        for epoch in step2_oracle() {
            let arguments = [
                epoch.gamma,
                epoch.l,
                epoch.l_prime,
                epoch.f,
                epoch.d,
                epoch.omega,
            ];
            let actual = frequency_dependent_corrections(
                epoch.gamma,
                [epoch.l, epoch.l_prime, epoch.f, epoch.d, epoch.omega],
            );
            max_dev = max_dev.max(assert_step2_matches(&actual, &epoch, arguments, [0.0; 6]));
        }
        println!("Step 2, Orekit's arguments: max deviation {max_dev:.3e}");
    }

    #[test]
    fn step2_at_epoch_matches_orekit() {
        // The same instants through this crate's own sidereal time and
        // Delaunay arguments. Orekit took TAI as UT1, so UT1 = TT - 32.184 s.
        let mut max_dev = 0.0_f64;
        for epoch in step2_oracle() {
            let ut1_fraction = epoch.tt_fraction - 32.184 / 86_400.0;
            let time_scales = TimeScales {
                jd_whole: epoch.jd_whole,
                ut1_fraction,
                tt_fraction: epoch.tt_fraction,
                tdb_fraction: epoch.tt_fraction,
                jd_ut1: epoch.jd_whole + ut1_fraction,
                jd_tt: epoch.jd_whole + epoch.tt_fraction,
                jd_tdb: epoch.jd_whole + epoch.tt_fraction,
                ut1_degraded: None,
            };
            let arguments =
                frequency_dependent_arguments_at(&time_scales).expect("Step 2 arguments");
            let actual = frequency_dependent_corrections(
                arguments[0],
                [
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5],
                ],
            );
            let days = (epoch.jd_whole - J2000_JD).abs() + epoch.tt_fraction.abs();
            let centuries = days / DAYS_PER_JULIAN_CENTURY;
            assert!(centuries <= 1.0, "the polynomial bound covers one century");
            let delaunay_error =
                (roundoff_gamma(13) + roundoff_gamma(27)) * (10.0 + 8500.0 * centuries);
            let gmst_error =
                (roundoff_gamma(31) + roundoff_gamma(45)) * (20.0 + 2.0 * PI * 1.003 * days);
            let mut argument_error = [delaunay_error; 6];
            argument_error[0] = gmst_error;
            let reference_arguments = [
                epoch.gamma,
                epoch.l,
                epoch.l_prime,
                epoch.f,
                epoch.d,
                epoch.omega,
            ];
            for index in 0..6 {
                let producer_bound = argument_error[index];
                let (difference_lower, difference_upper) =
                    angular_difference_bounds(arguments[index], reference_arguments[index]);
                assert!(
                    difference_lower <= producer_bound,
                    "argument {index}: circular difference interval [{difference_lower:e}, {difference_upper:e}], producer bound {producer_bound:e}"
                );
                argument_error[index] = difference_upper;
            }
            max_dev = max_dev.max(assert_step2_matches(
                &actual,
                &epoch,
                arguments,
                argument_error,
            ));
        }
        println!("Step 2, this crate's arguments: max deviation {max_dev:.3e}");
    }

    #[test]
    fn step2_is_added_to_step1_at_epoch_unless_turned_off() {
        let scales =
            TimeScales::from_utc(2003, 5, 6, 13, 43, 32.125).expect("valid UTC time scales");
        let epoch_tdb_seconds = crate::astro::time::civil::j2000_seconds_from_split(
            scales.jd_whole,
            scales.tdb_fraction,
        );
        let ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()));
        let with_step2 = SolidEarthTideGravity::default();
        assert!(with_step2.frequency_dependent);
        let step1_only = SolidEarthTideGravity {
            frequency_dependent: false,
            ..with_step2
        };
        let full = with_step2
            .coefficient_corrections_at_epoch(epoch_tdb_seconds, &ctx)
            .expect("Step 1 and Step 2");
        let step1 = step1_only
            .coefficient_corrections_at_epoch(epoch_tdb_seconds, &ctx)
            .expect("Step 1");
        let orientation = orientation_at_state(&ctx, epoch_tdb_seconds).expect("orientation");
        let step2 = frequency_dependent_corrections_at(&orientation.time_scales()).expect("Step 2");

        for index in 0..3 {
            assert_eq!(full[index].c, step1[index].c + step2[index].c);
            assert_eq!(full[index].s, step1[index].s + step2[index].s);
        }
        assert_eq!(&full[3..], &step1[3..]);
        // The K1 term alone is 4.7e-10 in C21/S21.
        assert!((step2[1].c * step2[1].c + step2[1].s * step2[1].s).sqrt() > 1.0e-10);
    }

    const FIELD_ORACLE_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tides/geopotential_tide_orekit_field.json");

    #[derive(Debug, Deserialize)]
    struct FieldOracle {
        epochs: Vec<FieldOracleEpoch>,
    }

    #[derive(Debug, Deserialize)]
    struct FieldOracleEpoch {
        gamma: f64,
        l: f64,
        l_prime: f64,
        f: f64,
        d: f64,
        omega: f64,
        sun_km: [f64; 3],
        moon_km: [f64; 3],
        tide_free: Vec<FixtureCoefficient>,
        zero_tide: Vec<FixtureCoefficient>,
    }

    #[test]
    fn permanent_tide_matches_iers_equation_6_14() {
        // Equation (6.14): dC20_perm = A0 H0 k20 = (4.4228e-8)(-0.31460) k20.
        let a0_h0: f64 = 4.4228e-8 * -0.31460;
        assert_eq!(permanent_tide_c20(TideSystem::TideFree), 0.0);
        assert_eq!(
            permanent_tide_c20(TideSystem::ZeroTide).to_bits(),
            (a0_h0 * 0.30190).to_bits()
        );
        assert_close(
            permanent_tide_c20(TideSystem::ZeroTide),
            -4.2007e-9,
            1.0e-13,
        );
        // A mean-tide C20 also holds the permanent tide-generating potential.
        assert_eq!(
            permanent_tide_c20(TideSystem::MeanTide).to_bits(),
            (a0_h0 * 0.30190 + a0_h0).to_bits()
        );
        assert_close(
            permanent_tide_c20(TideSystem::MeanTide),
            -1.81148e-8,
            1.0e-13,
        );
    }

    #[test]
    fn step3_changes_only_c20_by_the_permanent_tide_of_the_tide_system() {
        let sun = [1.2e8, -8.0e7, 3.1e7];
        let moon = [2.9e5, 2.4e5, -1.1e5];
        let tide_free = SolidEarthTideGravity::default();
        let free = tide_free
            .coefficient_corrections_for_body_fixed_bodies(sun, moon)
            .expect("tide-free corrections");
        for (tide_system, removed) in [
            (TideSystem::ZeroTide, 4.4228e-8 * -0.31460 * 0.30190),
            (
                TideSystem::MeanTide,
                4.4228e-8 * -0.31460 * 0.30190 + 4.4228e-8 * -0.31460,
            ),
        ] {
            let model = SolidEarthTideGravity {
                tide_system,
                ..tide_free
            };
            let corrections = model
                .coefficient_corrections_for_body_fixed_bodies(sun, moon)
                .expect("corrections");
            assert_eq!(corrections[0].c, free[0].c - removed, "{tide_system:?}");
            assert_eq!(&corrections[1..], &free[1..], "{tide_system:?}");
        }
    }

    #[test]
    fn steps_1_to_3_match_orekit_solid_tides_field() {
        // Orekit's SolidTidesField was given the same Sun and Moon positions,
        // Earth and body constants, fundamental arguments and tide system, so
        // the two sides differ only in rounding: the Legendre functions (a
        // recursion there, closed forms here), the trigonometry and the order
        // of the sums.
        let oracle: FieldOracle =
            serde_json::from_str(FIELD_ORACLE_FIXTURE).expect("parse SolidTidesField fixture");
        assert_eq!(oracle.epochs.len(), 48);
        let mut max_dev = 0.0_f64;
        for epoch in &oracle.epochs {
            let step2 = frequency_dependent_corrections(
                epoch.gamma,
                [epoch.l, epoch.l_prime, epoch.f, epoch.d, epoch.omega],
            );
            for (tide_system, expected) in [
                (TideSystem::TideFree, &epoch.tide_free),
                (TideSystem::ZeroTide, &epoch.zero_tide),
            ] {
                let model = SolidEarthTideGravity {
                    tide_system,
                    ..SolidEarthTideGravity::default()
                };
                let actual = model
                    .corrections(epoch.sun_km, epoch.moon_km, Some(step2))
                    .expect("Steps 1 to 3");
                let tolerance = field_roundoff_bound(&model, epoch);
                assert_eq!(expected.len(), 12, "degrees 2 to 4, orders 0 to n");
                for row in expected.iter() {
                    let (c, s) = actual
                        .iter()
                        .find(|value| value.degree == row.degree && value.order == row.order)
                        .map_or((0.0, 0.0), |value| (value.c, value.s));
                    max_dev = max_dev.max((c - row.c).abs()).max((s - row.s).abs());
                    assert_close(c, row.c, tolerance);
                    assert_close(s, row.s, tolerance);
                }
            }
        }
        println!("max deviation from SolidTidesField: {max_dev:.3e}");
    }

    fn field_roundoff_bound(model: &SolidEarthTideGravity, epoch: &FieldOracleEpoch) -> f64 {
        let mut magnitude = permanent_tide_c20(model.tide_system).abs();
        for (position, gravitational_parameter) in [
            (epoch.sun_km, model.gm_sun_km3_s2),
            (epoch.moon_km, model.gm_moon_km3_s2),
        ] {
            let radius = position
                .into_iter()
                .map(|component| component * component)
                .sum::<f64>()
                .sqrt();
            let radius_ratio = model.reference_radius_km / radius;
            let mass_ratio = gravitational_parameter / model.mu_earth_km3_s2;
            magnitude += mass_ratio
                * (radius_ratio.powi(3) * 0.31 * 5.0_f64.sqrt() / 5.0
                    + radius_ratio.powi(4) * 0.094 * 7.0_f64.sqrt() / 7.0);
        }
        let arguments = [
            epoch.gamma,
            epoch.l,
            epoch.l_prime,
            epoch.f,
            epoch.d,
            epoch.omega,
        ];
        let step2_error = [&K20_TERMS[..], &K21_TERMS[..], &K22_TERMS[..]]
            .into_iter()
            .map(|terms| step2_roundoff_bound(terms, arguments, arguments, [0.0; 6]))
            .fold(0.0, f64::max);
        2.0 * roundoff_gamma(128) * magnitude + step2_error
    }

    #[test]
    fn for_geopotential_takes_the_field_constants_and_tide_system() {
        let coefficients = [SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: -0.484_169_48e-3,
            s: 0.0,
        }];
        let field = SphericalHarmonicGravity::from_normalized_coefficients(
            398_600.441_8,
            6_378.136_6,
            2,
            0,
            &coefficients,
            TideSystem::ZeroTide,
        )
        .expect("zero-tide field");
        let tide = SolidEarthTideGravity::for_geopotential(&field);
        assert_eq!(tide.mu_earth_km3_s2, 398_600.441_8);
        assert_eq!(tide.reference_radius_km, 6_378.136_6);
        assert_eq!(tide.tide_system, TideSystem::ZeroTide);
        assert!(tide.frequency_dependent);
        assert_eq!(tide.gm_sun_km3_s2, GM_SUN_KM3_S2);
        assert_eq!(tide.gm_moon_km3_s2, GM_MOON_KM3_S2);

        let egm96 = SphericalHarmonicGravity::egm96_truncated(4, 4).expect("EGM96");
        assert_eq!(egm96.tide_system(), TideSystem::TideFree);
        assert_eq!(
            SolidEarthTideGravity::for_geopotential(&egm96),
            SolidEarthTideGravity::default()
        );
    }

    #[test]
    fn tide_forces_require_body_fixed_frame_provider() {
        let state = CartesianState::new(0.0, [7000.0, 100.0, 30.0], [0.0, 7.5, 0.0]);
        let error = SolidEarthTideGravity::default()
            .acceleration(&state, &PropagationContext::default())
            .expect_err("missing provider fails");
        assert!(format!("{error}").contains("body-fixed frame provider"));

        let ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()));
        SolidEarthTideGravity::default()
            .acceleration(&state, &ctx)
            .expect("solid tide acceleration");
        SolidEarthPoleTideGravity::default()
            .acceleration(&state, &ctx)
            .expect("pole tide acceleration");
    }
}
