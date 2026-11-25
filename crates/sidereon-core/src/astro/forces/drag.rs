//! Atmospheric drag over NRLMSISE-00 density.
//!
//! `CartesianState::epoch_tdb_seconds` is absolute TDB seconds past J2000.
//! Earth orientation is obtained on UT1 from the [`EarthOrientationProvider`]
//! attached to [`PropagationContext`]. When the context supplies no provider,
//! drag uses [`TdbEarthOrientationProvider::new`] in
//! [`ValidityMode::Permissive`] so multi-year drag and decay evaluations past
//! the embedded UT1 table use the long-term `Delta T` curve while recording
//! the departure in [`PropagationContext`]; a caller-supplied provider keeps
//! its own [`ValidityMode`]. The body-fixed position for geodetic coordinates
//! and the Earth rotation vector `omega` come from that full IAU 2006/2000A
//! precession-nutation, UT1 apparent sidereal rotation, and polar-motion
//! orientation.
//!
//! NRLMSISE-00 calendar inputs (year, day of year, seconds of day) and
//! space-weather lookups are evaluated on UTC derived from the orientation's
//! [`TimeScales`]. Wind is neglected and the atmosphere co-rotates with Earth,
//! forming the GCRF atmosphere-relative velocity as `v - omega x r` with the
//! orientation's Earth rotation vector rotated into GCRF. Density caching is
//! intentionally left for a future optimization so each RHS evaluation is
//! reproducible from the instantaneous state.

use super::r#trait::{
    difference_quotient_partials, finite_acceleration, position_difference_quotients,
    AccelerationPartials,
};
use super::scaled::{wide_norm_squared, Wide};
use crate::astro::atmosphere::{self, NrlmsiseInput, MAX_ALTITUDE_KM};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::frames::orientation::{
    EarthOrientation, EarthOrientationProvider, TdbEarthOrientationProvider,
};
use crate::astro::frames::transforms::{
    itrs_to_geodetic_compute, mat3_vec3_mul, FrameTransformError,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::space_weather::{SpaceWeatherError, SpaceWeatherPolicy, SpaceWeatherTable};
use crate::astro::state::CartesianState;
use crate::astro::time::civil::{day_of_year_int, j2000_seconds, second_of_day};
use crate::astro::time::scales::TimeScales;
use crate::astro::time::ValidityMode;
use nalgebra::Vector3;
use std::sync::Arc;

/// Space-weather inputs to NRLMSISE-00 for drag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceWeather {
    /// Daily F10.7 from the previous day, solar flux units.
    pub f107: f64,
    /// 81-day centered average F10.7, solar flux units.
    pub f107a: f64,
    /// Daily magnetic Ap index.
    pub ap: f64,
}

impl Default for SpaceWeather {
    /// Reference quiet-Sun standard inputs from the atmosphere module.
    fn default() -> Self {
        Self {
            f107: atmosphere::DEFAULT_F107,
            f107a: atmosphere::DEFAULT_F107A,
            ap: atmosphere::DEFAULT_AP,
        }
    }
}

/// Where drag evaluations obtain space-weather values.
#[derive(Debug, Clone)]
pub enum SpaceWeatherSource {
    /// Constant values for every epoch.
    Fixed(SpaceWeather),
    /// Per-epoch values from a parsed CelesTrak table under the default
    /// (strict) [`SpaceWeatherPolicy`]: an epoch whose rows leave the daily Ap
    /// blank, such as the monthly predicted region, is refused.
    Table(Arc<SpaceWeatherTable>),
    /// Per-epoch values from a parsed CelesTrak table under an explicit
    /// policy, for example [`SpaceWeatherPolicy::lenient`] to accept the quiet
    /// default Ap in the monthly predicted region.
    TableWithPolicy(Arc<SpaceWeatherTable>, SpaceWeatherPolicy),
}

impl SpaceWeatherSource {
    /// Resolve space-weather inputs at one J2000-second epoch.
    pub fn at(&self, epoch_j2000_s: f64) -> Result<SpaceWeather, SpaceWeatherError> {
        match self {
            Self::Fixed(space_weather) => Ok(*space_weather),
            Self::Table(table) => table.space_weather_at(epoch_j2000_s),
            Self::TableWithPolicy(table, policy) => table
                .sample_at_with_policy(epoch_j2000_s, *policy)
                .map(|sample| sample.space_weather),
        }
    }
}

/// Atmospheric-drag force model using cannonball drag over NRLMSISE-00 density.
///
/// The stored factor is `B = C_D * A / m` in m^2/kg. Use
/// [`CompositeForceModel`](crate::astro::forces::CompositeForceModel) to layer
/// this force on a gravity model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragForce {
    bc_factor_m2_kg: f64,
    space_weather: SpaceWeather,
    cutoff_altitude_km: f64,
}

impl DragForce {
    /// Conventional reentry and decay altitude, km.
    pub const DEFAULT_REENTRY_ALTITUDE_KM: f64 = 100.0;

    /// Build from drag coefficient, cross-section area, mass, and cutoff.
    pub fn from_area_mass(
        cd: f64,
        area_m2: f64,
        mass_kg: f64,
        sw: SpaceWeather,
        cutoff_altitude_km: f64,
    ) -> Result<Self, PropagationError> {
        DragParameters::from_area_mass(cd, area_m2, mass_kg, sw, cutoff_altitude_km)
            .map(DragParameters::to_force)
    }

    /// Build directly from `B = C_D * A / m` in m^2/kg.
    pub fn from_bc_factor_m2_kg(
        bc_factor_m2_kg: f64,
        sw: SpaceWeather,
        cutoff_altitude_km: f64,
    ) -> Result<Self, PropagationError> {
        DragParameters::from_bc_factor_m2_kg(bc_factor_m2_kg, sw, cutoff_altitude_km)
            .map(DragParameters::to_force)
    }

    /// Build from reciprocal ballistic coefficient `BC = m / (C_D * A)`.
    pub fn from_ballistic_coefficient(
        bc_kg_m2: f64,
        sw: SpaceWeather,
        cutoff_altitude_km: f64,
    ) -> Result<Self, PropagationError> {
        DragParameters::from_ballistic_coefficient(bc_kg_m2, sw, cutoff_altitude_km)
            .map(DragParameters::to_force)
    }

    /// Build from drag coefficient, area, and mass with the default cutoff.
    pub fn from_area_mass_default_cutoff(
        cd: f64,
        area_m2: f64,
        mass_kg: f64,
        sw: SpaceWeather,
    ) -> Result<Self, PropagationError> {
        DragParameters::from_area_mass_default_cutoff(cd, area_m2, mass_kg, sw)
            .map(DragParameters::to_force)
    }

    /// Drag ballistic-coefficient factor `B = C_D * A / m`, m^2/kg.
    pub fn bc_factor_m2_kg(&self) -> f64 {
        self.bc_factor_m2_kg
    }

    /// Space-weather inputs used for density evaluation.
    pub fn space_weather(&self) -> SpaceWeather {
        self.space_weather
    }

    /// Density cutoff altitude, km.
    pub fn cutoff_altitude_km(&self) -> f64 {
        self.cutoff_altitude_km
    }
}

impl ForceModel for DragForce {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_drag_state(state)?;
        let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
        drag_acceleration(
            self.bc_factor_m2_kg,
            self.cutoff_altitude_km,
            state,
            &orientation,
            |_| Ok(self.space_weather),
        )
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        if vector_is_finite(&state.velocity_km_s) {
            difference_quotient_partials(self, parameters, state, ctx)
        } else {
            position_difference_quotients(self, parameters, state, ctx)
        }
    }
}

/// Atmospheric drag whose space weather is resolved per evaluation epoch.
#[derive(Debug, Clone)]
pub struct SourcedDragForce {
    bc_factor_m2_kg: f64,
    source: SpaceWeatherSource,
    cutoff_altitude_km: f64,
}

impl SourcedDragForce {
    /// Build from validated drag parameters and a dynamic source.
    ///
    /// The source supplies the per-epoch values; the fixed [`SpaceWeather`] inside
    /// `drag` is not consulted.
    pub fn new(drag: DragParameters, source: SpaceWeatherSource) -> Self {
        Self {
            bc_factor_m2_kg: drag.bc_factor_m2_kg,
            source,
            cutoff_altitude_km: drag.cutoff_altitude_km,
        }
    }

    /// Drag ballistic-coefficient factor `B = C_D * A / m`, m^2/kg.
    pub fn bc_factor_m2_kg(&self) -> f64 {
        self.bc_factor_m2_kg
    }

    /// Space-weather source used for density evaluation.
    pub fn source(&self) -> &SpaceWeatherSource {
        &self.source
    }

    /// Density cutoff altitude, km.
    pub fn cutoff_altitude_km(&self) -> f64 {
        self.cutoff_altitude_km
    }
}

impl ForceModel for SourcedDragForce {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_drag_state(state)?;
        let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
        drag_acceleration(
            self.bc_factor_m2_kg,
            self.cutoff_altitude_km,
            state,
            &orientation,
            |epoch| {
                self.source.at(epoch).map_err(|error| {
                    PropagationError::ForceModelFailure(format!("space weather: {error}"))
                })
            },
        )
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        if vector_is_finite(&state.velocity_km_s) {
            difference_quotient_partials(self, parameters, state, ctx)
        } else {
            position_difference_quotients(self, parameters, state, ctx)
        }
    }
}

/// Validated drag parameters that can be stored on propagator configs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragParameters {
    bc_factor_m2_kg: f64,
    space_weather: SpaceWeather,
    cutoff_altitude_km: f64,
}

impl DragParameters {
    /// Build from drag coefficient, cross-section area, mass, and cutoff.
    pub fn from_area_mass(
        cd: f64,
        area_m2: f64,
        mass_kg: f64,
        sw: SpaceWeather,
        cutoff_altitude_km: f64,
    ) -> Result<Self, PropagationError> {
        validate_finite_positive("cd", cd)?;
        validate_finite_positive("area_m2", area_m2)?;
        validate_finite_positive("mass_kg", mass_kg)?;
        let bc_factor_m2_kg = cd * area_m2 / mass_kg;
        Self::from_bc_factor_m2_kg(bc_factor_m2_kg, sw, cutoff_altitude_km)
    }

    /// Build directly from `B = C_D * A / m` in m^2/kg.
    pub fn from_bc_factor_m2_kg(
        bc_factor_m2_kg: f64,
        sw: SpaceWeather,
        cutoff_altitude_km: f64,
    ) -> Result<Self, PropagationError> {
        validate_finite_positive("bc_factor_m2_kg", bc_factor_m2_kg)?;
        validate_space_weather(sw)?;
        validate_cutoff(cutoff_altitude_km)?;
        Ok(Self {
            bc_factor_m2_kg,
            space_weather: sw,
            cutoff_altitude_km,
        })
    }

    /// Build from reciprocal ballistic coefficient `BC = m / (C_D * A)`.
    pub fn from_ballistic_coefficient(
        bc_kg_m2: f64,
        sw: SpaceWeather,
        cutoff_altitude_km: f64,
    ) -> Result<Self, PropagationError> {
        validate_finite_positive("bc_kg_m2", bc_kg_m2)?;
        Self::from_bc_factor_m2_kg(bc_kg_m2.recip(), sw, cutoff_altitude_km)
    }

    /// Build from drag coefficient, area, and mass with the default cutoff.
    pub fn from_area_mass_default_cutoff(
        cd: f64,
        area_m2: f64,
        mass_kg: f64,
        sw: SpaceWeather,
    ) -> Result<Self, PropagationError> {
        Self::from_area_mass(
            cd,
            area_m2,
            mass_kg,
            sw,
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
    }

    /// Convert to a force model without revalidating.
    pub fn to_force(self) -> DragForce {
        DragForce {
            bc_factor_m2_kg: self.bc_factor_m2_kg,
            space_weather: self.space_weather,
            cutoff_altitude_km: self.cutoff_altitude_km,
        }
    }

    /// Drag ballistic-coefficient factor `B = C_D * A / m`, m^2/kg.
    pub fn bc_factor_m2_kg(&self) -> f64 {
        self.bc_factor_m2_kg
    }

    /// Space-weather inputs used for density evaluation.
    pub fn space_weather(&self) -> SpaceWeather {
        self.space_weather
    }

    /// Density cutoff altitude, km.
    pub fn cutoff_altitude_km(&self) -> f64 {
        self.cutoff_altitude_km
    }
}

pub(crate) fn geodetic_altitude_km(
    state: &CartesianState,
    ctx: &PropagationContext,
) -> Result<f64, PropagationError> {
    validate_drag_state(state)?;
    let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
    Ok(geodetic_from_orientation(state, &orientation)?.alt_km)
}

pub(crate) fn map_frame_error(
    context: &'static str,
    error: FrameTransformError,
) -> PropagationError {
    PropagationError::NumericalFailure(format!("drag {context} failed: {error}"))
}

fn orientation_at_state(
    ctx: &PropagationContext,
    epoch_tdb_seconds: f64,
) -> Result<EarthOrientation, PropagationError> {
    let default_provider =
        TdbEarthOrientationProvider::new().with_validity(ValidityMode::Permissive);
    let provider: &dyn EarthOrientationProvider =
        ctx.body_fixed_frame_provider().unwrap_or(&default_provider);
    provider
        .orientation_at_tdb_seconds(epoch_tdb_seconds)
        .map(|orientation| ctx.record_orientation(orientation))
        .map_err(|error| {
            PropagationError::from_frame("drag body-fixed frame evaluation failed", error)
        })
}

fn drag_acceleration(
    bc_factor_m2_kg: f64,
    cutoff_altitude_km: f64,
    state: &CartesianState,
    orientation: &EarthOrientation,
    space_weather: impl FnOnce(f64) -> Result<SpaceWeather, PropagationError>,
) -> Result<Vector3<f64>, PropagationError> {
    let geodetic = match geodetic_from_orientation(state, orientation) {
        Ok(geodetic) => geodetic,
        Err(_)
            if !state.position_km.norm_squared().is_finite()
                && orientation
                    .gcrf_to_itrf_matrix()
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite()) =>
        {
            return Ok(Vector3::zeros());
        }
        Err(error) => return Err(error),
    };

    if geodetic.alt_km <= cutoff_altitude_km || geodetic.alt_km > MAX_ALTITUDE_KM {
        return Ok(Vector3::zeros());
    }

    if !vector_is_finite(&state.velocity_km_s) {
        return Err(PropagationError::InvalidInput(
            "velocity_km_s not finite".to_string(),
        ));
    }
    let calendar = calendar_from_time_scales(&orientation.time_scales())?;
    let space_weather = space_weather(calendar.space_weather_utc_j2000_s)?;
    validate_space_weather(space_weather)?;
    let input = NrlmsiseInput {
        year: calendar.year,
        doy: calendar.doy,
        sec: calendar.sec_of_day,
        alt: geodetic.alt_km,
        g_lat: geodetic.lat_deg,
        g_long: geodetic.lon_deg,
        lst: 0.0,
        f107a: space_weather.f107a,
        f107: space_weather.f107,
        ap: space_weather.ap,
        ap_array: None,
    };
    let density = atmosphere::nrlmsise00_with_lst(&input, None)
        .map_err(|error| {
            PropagationError::NumericalFailure(format!("drag atmosphere failed: {error}"))
        })?
        .density();
    if !density.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "drag density not finite".to_string(),
        ));
    }

    let v_rel_km_s = relative_velocity_km_s(state, orientation)?;
    let v_rel_m_s = v_rel_km_s * M_PER_KM;
    let speed_m_s = v_rel_m_s.norm();
    let accel_m_s2 = v_rel_m_s * (-0.5 * density * bc_factor_m2_kg * speed_m_s);
    let accel_km_s2 = accel_m_s2 / M_PER_KM;
    if vector_is_finite(&accel_km_s2) {
        return Ok(accel_km_s2);
    }

    // The velocity's norm and the cannonball scale are allowed to exceed
    // binary64 before the density and final units are applied.
    let meters_per_km = Wide::from(M_PER_KM);
    let velocity = <[f64; 3]>::from(v_rel_km_s).map(|value| Wide::from(value) * meters_per_km);
    let speed = wide_norm_squared(velocity).sqrt();
    let scale = Wide::from(-0.5) * Wide::from(density) * Wide::from(bc_factor_m2_kg) * speed;
    finite_acceleration(Vector3::from(
        velocity.map(|value| ((value * scale) / meters_per_km).value()),
    ))
}

fn validate_finite_positive(field: &'static str, value: f64) -> Result<(), PropagationError> {
    if !value.is_finite() {
        return Err(PropagationError::InvalidInput(format!(
            "{field} not finite"
        )));
    }
    if value <= 0.0 {
        return Err(PropagationError::InvalidInput(format!(
            "{field} not positive"
        )));
    }
    Ok(())
}

fn validate_finite_nonnegative(field: &'static str, value: f64) -> Result<(), PropagationError> {
    if !value.is_finite() {
        return Err(PropagationError::InvalidInput(format!(
            "{field} not finite"
        )));
    }
    if value < 0.0 {
        return Err(PropagationError::InvalidInput(format!("{field} negative")));
    }
    Ok(())
}

fn validate_space_weather(sw: SpaceWeather) -> Result<(), PropagationError> {
    validate_finite_nonnegative("f107", sw.f107)?;
    validate_finite_nonnegative("f107a", sw.f107a)?;
    validate_finite_nonnegative("ap", sw.ap)
}

fn validate_cutoff(cutoff_altitude_km: f64) -> Result<(), PropagationError> {
    if !cutoff_altitude_km.is_finite() {
        return Err(PropagationError::InvalidInput(
            "cutoff_altitude_km not finite".to_string(),
        ));
    }
    if !(0.0..=MAX_ALTITUDE_KM).contains(&cutoff_altitude_km) {
        return Err(PropagationError::InvalidInput(
            "cutoff_altitude_km out of domain".to_string(),
        ));
    }
    Ok(())
}

fn validate_drag_state(state: &CartesianState) -> Result<(), PropagationError> {
    if !state.epoch_tdb_seconds.is_finite() {
        return Err(PropagationError::InvalidInput(
            "epoch_tdb_seconds not finite".to_string(),
        ));
    }
    if !vector_is_finite(&state.position_km) {
        return Err(PropagationError::InvalidInput(
            "position_km not finite".to_string(),
        ));
    }
    if state.position_km.iter().all(|&value| value == 0.0) {
        return Err(PropagationError::NumericalFailure(
            "Zero position magnitude".to_string(),
        ));
    }
    Ok(())
}

fn vector_is_finite(v: &Vector3<f64>) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[derive(Debug, Clone, Copy)]
struct DragCalendar {
    year: i32,
    doy: i32,
    sec_of_day: f64,
    space_weather_utc_j2000_s: f64,
}

fn calendar_from_time_scales(ts: &TimeScales) -> Result<DragCalendar, PropagationError> {
    let (year, month, day, hour, minute, second) = ts.utc_calendar_fields().ok_or_else(|| {
        PropagationError::NumericalFailure("drag UTC calendar conversion failed".to_string())
    })?;
    let doy_i64 = day_of_year_int(year, month, day);
    let doy = i32::try_from(doy_i64).map_err(|_| {
        PropagationError::NumericalFailure("drag day of year out of i32 range".to_string())
    })?;
    let sec_of_day = second_of_day(hour, minute, second);
    let sw_second = if second >= 60.0 { 59.0 } else { second };
    let space_weather_utc_j2000_s = j2000_seconds(year, month, day, hour, minute, sw_second);
    if !(sec_of_day.is_finite() && space_weather_utc_j2000_s.is_finite()) {
        return Err(PropagationError::NumericalFailure(
            "drag UTC calendar seconds not finite".to_string(),
        ));
    }
    Ok(DragCalendar {
        year,
        doy,
        sec_of_day,
        space_weather_utc_j2000_s,
    })
}

#[derive(Debug, Clone, Copy)]
struct DragGeodetic {
    lat_deg: f64,
    lon_deg: f64,
    alt_km: f64,
}

fn geodetic_from_orientation(
    state: &CartesianState,
    orientation: &EarthOrientation,
) -> Result<DragGeodetic, PropagationError> {
    let r_itrf = orientation
        .gcrf_to_itrf_position_km(state.position_array())
        .map_err(|error| map_frame_error("gcrf_to_itrf", error))?;
    let (lat_deg, lon_deg, alt_km) = itrs_to_geodetic_compute(r_itrf[0], r_itrf[1], r_itrf[2])
        .map_err(|error| map_frame_error("geodetic", error))?;
    if !(lat_deg.is_finite() && lon_deg.is_finite() && alt_km.is_finite()) {
        return Err(PropagationError::NumericalFailure(
            "drag geodetic state not finite".to_string(),
        ));
    }
    Ok(DragGeodetic {
        lat_deg,
        lon_deg,
        alt_km,
    })
}

fn relative_velocity_km_s(
    state: &CartesianState,
    orientation: &EarthOrientation,
) -> Result<Vector3<f64>, PropagationError> {
    let omega_itrf = orientation.earth_rotation_vector_itrf_rad_s();
    let omega_gcrf_arr = mat3_vec3_mul(&orientation.itrf_to_gcrf_matrix(), &omega_itrf)
        .map_err(|error| map_frame_error("earth_rotation_vector", error))?;
    let omega_gcrf = Vector3::new(omega_gcrf_arr[0], omega_gcrf_arr[1], omega_gcrf_arr[2]);
    Ok(state.velocity_km_s - omega_gcrf.cross(&state.position_km))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::{earth::GM_EARTH_M3_S2, MU_EARTH, RE_EARTH};
    use crate::astro::elements::rv2coe;
    use crate::astro::frames::transforms::geodetic_to_itrs;
    use crate::astro::propagator::api::IntegratorOptions;
    use crate::astro::propagator::numerical::{ForceModelKind, IntegratorKind, StatePropagator};
    use std::f64::consts::TAU;

    const TEST_EPOCH_S: f64 = 646_315_200.25;
    const BC_FACTOR: f64 = 0.02;

    fn quiet_sw() -> SpaceWeather {
        SpaceWeather::default()
    }

    fn test_drag(bc_factor_m2_kg: f64) -> DragForce {
        DragForce::from_bc_factor_m2_kg(
            bc_factor_m2_kg,
            quiet_sw(),
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("valid drag")
    }

    fn test_drag_parameters(bc_factor_m2_kg: f64) -> DragParameters {
        DragParameters::from_bc_factor_m2_kg(
            bc_factor_m2_kg,
            quiet_sw(),
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("valid drag")
    }

    fn circular_state(epoch: f64, altitude_km: f64, inclination_rad: f64) -> CartesianState {
        let r = RE_EARTH + altitude_km;
        let v = (MU_EARTH / r).sqrt();
        CartesianState::new(
            epoch,
            [r, 0.0, 0.0],
            [
                0.0,
                v * libm::cos(inclination_rad),
                v * libm::sin(inclination_rad),
            ],
        )
    }

    fn circular_state_at_argument(
        epoch: f64,
        altitude_km: f64,
        inclination_rad: f64,
        argument_rad: f64,
    ) -> CartesianState {
        let r = RE_EARTH + altitude_km;
        let v = (MU_EARTH / r).sqrt();
        let cu = libm::cos(argument_rad);
        let su = libm::sin(argument_rad);
        let ci = libm::cos(inclination_rad);
        let si = libm::sin(inclination_rad);
        CartesianState::new(
            epoch,
            [r * cu, r * su * ci, r * su * si],
            [-v * su, v * cu * ci, v * cu * si],
        )
    }

    fn state_from_geodetic_alt(epoch: f64, altitude_km: f64) -> CartesianState {
        let (x_ecef, y_ecef, z_ecef) =
            geodetic_to_itrs(0.0, 0.0, altitude_km).expect("valid geodetic");
        let orientation = TdbEarthOrientationProvider::new()
            .orientation_at_tdb_seconds(epoch)
            .expect("valid orientation");
        let pos_eci = orientation
            .itrf_to_gcrf_position_km([x_ecef, y_ecef, z_ecef])
            .expect("valid ITRF to GCRF");
        let r = RE_EARTH + altitude_km;
        let v = (MU_EARTH / r).sqrt();
        CartesianState::new(epoch, pos_eci, [0.0, v, 0.0])
    }

    fn density_for_state(state: &CartesianState, sw: SpaceWeather) -> f64 {
        validate_drag_state(state).expect("valid drag state");
        let orientation =
            orientation_at_state(&PropagationContext::default(), state.epoch_tdb_seconds)
                .expect("valid orientation");
        let calendar =
            calendar_from_time_scales(&orientation.time_scales()).expect("valid calendar");
        let geodetic = geodetic_from_orientation(state, &orientation).expect("valid geodetic");
        let input = NrlmsiseInput {
            year: calendar.year,
            doy: calendar.doy,
            sec: calendar.sec_of_day,
            alt: geodetic.alt_km,
            g_lat: geodetic.lat_deg,
            g_long: geodetic.lon_deg,
            lst: 0.0,
            f107a: sw.f107a,
            f107: sw.f107,
            ap: sw.ap,
            ap_array: None,
        };
        atmosphere::nrlmsise00_with_lst(&input, None)
            .expect("valid atmosphere")
            .density()
    }

    fn hand_acceleration(state: &CartesianState, bc_factor_m2_kg: f64) -> Vector3<f64> {
        let orientation =
            orientation_at_state(&PropagationContext::default(), state.epoch_tdb_seconds)
                .expect("valid orientation");
        let rho = density_for_state(state, quiet_sw());
        let v_rel_m_s = relative_velocity_km_s(state, &orientation).expect("v_rel") * M_PER_KM;
        v_rel_m_s * (-0.5 * rho * bc_factor_m2_kg * v_rel_m_s.norm()) / M_PER_KM
    }

    fn specific_energy(state: &CartesianState) -> f64 {
        0.5 * state.velocity_km_s.norm_squared() - MU_EARTH / state.position_km.norm()
    }

    fn sma_km(state: &CartesianState) -> f64 {
        rv2coe(state.position_array(), state.velocity_array(), MU_EARTH)
            .expect("valid elements")
            .a
    }

    fn slope(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            numerator += (x - mean_x) * (y - mean_y);
            denominator += (x - mean_x) * (x - mean_x);
        }
        numerator / denominator
    }

    fn propagation_options() -> IntegratorOptions {
        IntegratorOptions {
            abs_tol: 1.0e-9,
            rel_tol: 1.0e-11,
            initial_step: 30.0,
            min_step: 1.0e-6,
            max_step: 120.0,
            max_steps: 200_000,
            dense_output: false,
        }
    }

    #[test]
    fn drag_accepts_later_epochs_with_finite_frame_and_density() {
        let ctx = PropagationContext::default();
        let epoch = j2000_seconds(3100, 1, 1, 0, 0, 0.0);
        let state = circular_state(epoch, 400.0, 0.1);
        let result = test_drag(0.02).acceleration(&state, &ctx).unwrap();
        assert!(result.norm() > 0.0);
    }

    #[test]
    fn drag_recovers_an_overflowing_relative_speed_squared() {
        let ctx = PropagationContext::default();
        let mut nominal = state_from_geodetic_alt(TEST_EPOCH_S, 400.0);
        let orientation = orientation_at_state(&ctx, nominal.epoch_tdb_seconds).unwrap();
        nominal.velocity_km_s.fill(0.0);
        let wind = -relative_velocity_km_s(&nominal, &orientation).unwrap();
        nominal.velocity_km_s = wind + Vector3::x();
        let mut large = nominal;
        large.velocity_km_s = wind + Vector3::x() * libm::ldexp(1.0, 520);
        let parameters = test_drag_parameters(libm::ldexp(1.0, -800));
        let drag = parameters.to_force();
        let source = SourcedDragForce::new(parameters, SpaceWeatherSource::Fixed(quiet_sw()));
        for model in [&drag as &dyn ForceModel, &source] {
            let reference = model.acceleration(&nominal, &ctx).unwrap();
            let actual = model.acceleration(&large, &ctx).unwrap();
            assert!(reference.x < 0.0);
            assert_eq!(actual.x.to_bits(), libm::ldexp(reference.x, 1040).to_bits());
        }
    }

    #[test]
    fn drag_force_matches_hand_computed_acceleration_0ulp() {
        let state = circular_state(TEST_EPOCH_S, 400.0, 51.6_f64.to_radians());
        let drag = test_drag(BC_FACTOR);
        let actual = drag
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");
        let expected = hand_acceleration(&state, BC_FACTOR);

        assert_eq!(actual.x.to_bits(), expected.x.to_bits());
        assert_eq!(actual.y.to_bits(), expected.y.to_bits());
        assert_eq!(actual.z.to_bits(), expected.z.to_bits());
    }

    #[test]
    fn drag_is_antiparallel_to_relative_velocity() {
        const DIRECTION_TOL: f64 = 1.0e-14;
        let state = circular_state(TEST_EPOCH_S, 380.0, 63.4_f64.to_radians());
        let drag = test_drag(BC_FACTOR);
        let ctx = PropagationContext::default();
        let accel = drag.acceleration(&state, &ctx).expect("drag acceleration");
        let orientation = orientation_at_state(&ctx, state.epoch_tdb_seconds).expect("orientation");
        let v_rel = relative_velocity_km_s(&state, &orientation).expect("v_rel");

        assert!(accel.dot(&v_rel) < 0.0);
        assert!(accel.cross(&v_rel).norm() <= accel.norm() * v_rel.norm() * DIRECTION_TOL);

        let high_density = density_for_state(&circular_state(TEST_EPOCH_S, 300.0, 0.0), quiet_sw());
        let low_density = density_for_state(&circular_state(TEST_EPOCH_S, 450.0, 0.0), quiet_sw());
        assert!(high_density > low_density);
    }

    #[test]
    fn drag_scales_linearly_with_bc() {
        const LINEAR_TOL: f64 = 1.0e-14;
        let state = circular_state(TEST_EPOCH_S, 400.0, 0.3);
        let accel = test_drag(BC_FACTOR)
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");
        let doubled = test_drag(2.0 * BC_FACTOR)
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");

        assert!((doubled - accel * 2.0).norm() <= accel.norm() * LINEAR_TOL);
    }

    #[test]
    fn rotating_atmosphere_reduces_drag_vs_inertial() {
        let state = circular_state(TEST_EPOCH_S, 400.0, 0.0);
        let drag = test_drag(BC_FACTOR);
        let rotating = drag
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");
        let rho = density_for_state(&state, quiet_sw());
        let v_eci_m_s = state.velocity_km_s * M_PER_KM;
        let inertial = v_eci_m_s * (-0.5 * rho * BC_FACTOR * v_eci_m_s.norm()) / M_PER_KM;

        assert!(rotating.norm() < inertial.norm());
    }

    #[test]
    fn drag_secularly_decreases_energy_and_sma() {
        const ENERGY_DROP_TOL: f64 = 1.0e-5;
        const SMA_DROP_TOL_KM: f64 = 1.0e-3;
        let initial = circular_state(TEST_EPOCH_S, 250.0, 70.0_f64.to_radians());
        let drag = test_drag_parameters(0.15);
        let propagator = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        )
        .with_options(propagation_options())
        .with_drag(drag);
        let epochs: Vec<f64> = (0..=12)
            .map(|i| initial.epoch_tdb_seconds + i as f64 * 600.0)
            .collect();
        let states = propagator.ephemeris(&epochs).expect("drag ephemeris");
        let elapsed: Vec<f64> = states
            .iter()
            .map(|state| state.epoch_tdb_seconds - initial.epoch_tdb_seconds)
            .collect();
        let energies: Vec<f64> = states.iter().map(specific_energy).collect();
        let smas: Vec<f64> = states.iter().map(sma_km).collect();

        assert!(energies[energies.len() - 1] < energies[0] - ENERGY_DROP_TOL);
        assert!(smas[smas.len() - 1] < smas[0] - SMA_DROP_TOL_KM);
        assert!(slope(&elapsed, &energies) < 0.0);
        assert!(slope(&elapsed, &smas) < 0.0);
    }

    #[test]
    fn no_drag_conserves_energy() {
        const ENERGY_TOL: f64 = 1.0e-8;
        const SMA_TOL_KM: f64 = 1.0e-5;
        let initial = circular_state(TEST_EPOCH_S, 250.0, 70.0_f64.to_radians());
        let propagator = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        )
        .with_options(IntegratorOptions {
            abs_tol: 1.0e-11,
            rel_tol: 1.0e-13,
            ..propagation_options()
        });
        let epochs: Vec<f64> = (0..=12)
            .map(|i| initial.epoch_tdb_seconds + i as f64 * 600.0)
            .collect();
        let states = propagator.ephemeris(&epochs).expect("no-drag ephemeris");
        let energies: Vec<f64> = states.iter().map(specific_energy).collect();
        let smas: Vec<f64> = states.iter().map(sma_km).collect();

        assert!((energies[energies.len() - 1] - energies[0]).abs() <= ENERGY_TOL);
        assert!((smas[smas.len() - 1] - smas[0]).abs() <= SMA_TOL_KM);
    }

    #[test]
    fn near_circular_leo_decay_rate_matches_kinghele_within_tolerance() {
        // King-Hele near-circular check, using B = 0.02 m^2/kg, altitude 360 km,
        // inclination 88 deg, quiet-Sun space weather, and a 24-sample
        // NRLMSISE-00 orbit-mean density. Observed SMA slope:
        // -5.039687362219701e-3 m/s. Expected averaged rate:
        // -5.012722827628129e-3 m/s. Relative difference: 5.38e-3.
        const AGREEMENT_TOL: f64 = 0.15;
        let altitude_km = 360.0;
        let inclination_rad = 88.0_f64.to_radians();
        let radius_km = RE_EARTH + altitude_km;
        let period_s = TAU * (radius_km.powi(3) / MU_EARTH).sqrt();
        let duration_s = 3.0 * period_s;
        let initial = circular_state(TEST_EPOCH_S, altitude_km, inclination_rad);
        let drag = test_drag_parameters(BC_FACTOR);

        let propagator = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        )
        .with_options(propagation_options())
        .with_drag(drag);
        let epochs: Vec<f64> = (0..=18)
            .map(|i| initial.epoch_tdb_seconds + duration_s * i as f64 / 18.0)
            .collect();
        let states = propagator.ephemeris(&epochs).expect("drag ephemeris");
        let elapsed: Vec<f64> = states
            .iter()
            .map(|state| state.epoch_tdb_seconds - initial.epoch_tdb_seconds)
            .collect();
        let sma_m: Vec<f64> = states
            .iter()
            .map(|state| sma_km(state) * M_PER_KM)
            .collect();
        let observed_rate_m_s = slope(&elapsed, &sma_m);

        let samples = 24;
        let mut density_sum = 0.0;
        for i in 0..samples {
            let fraction = i as f64 / samples as f64;
            let state = circular_state_at_argument(
                TEST_EPOCH_S + period_s * fraction,
                altitude_km,
                inclination_rad,
                TAU * fraction,
            );
            density_sum += density_for_state(&state, quiet_sw());
        }
        let mean_density = density_sum / samples as f64;
        let expected_rate_m_s =
            -BC_FACTOR * mean_density * (GM_EARTH_M3_S2 * radius_km * M_PER_KM).sqrt();

        let relative_error = ((observed_rate_m_s - expected_rate_m_s) / expected_rate_m_s).abs();
        assert!(
            relative_error <= AGREEMENT_TOL,
            "observed {observed_rate_m_s} m/s expected {expected_rate_m_s} m/s"
        );
    }

    #[test]
    fn drag_zero_above_ceiling_and_below_cutoff() {
        let at_ceiling = state_from_geodetic_alt(TEST_EPOCH_S, MAX_ALTITUDE_KM);
        let above_ceiling = state_from_geodetic_alt(TEST_EPOCH_S, MAX_ALTITUDE_KM + 1.0e-3);
        let accel_ceiling = test_drag(BC_FACTOR)
            .acceleration(&at_ceiling, &PropagationContext::default())
            .expect("ceiling evaluates");
        let accel_above = test_drag(BC_FACTOR)
            .acceleration(&above_ceiling, &PropagationContext::default())
            .expect("above ceiling zeroes");
        assert!(accel_ceiling.norm() > 0.0);
        assert_eq!(accel_above, Vector3::zeros());

        let cutoff_state = state_from_geodetic_alt(TEST_EPOCH_S, 100.0);
        let cutoff = geodetic_altitude_km(&cutoff_state, &PropagationContext::default())
            .expect("cutoff altitude");
        let drag =
            DragForce::from_bc_factor_m2_kg(BC_FACTOR, quiet_sw(), cutoff).expect("valid cutoff");
        let accel_cutoff = drag
            .acceleration(&cutoff_state, &PropagationContext::default())
            .expect("cutoff zeroes");
        assert_eq!(accel_cutoff, Vector3::zeros());

        let above_cutoff = state_from_geodetic_alt(TEST_EPOCH_S, 100.001);
        let accel_above_cutoff = drag
            .acceleration(&above_cutoff, &PropagationContext::default())
            .expect("above cutoff evaluates");
        assert!(accel_above_cutoff.norm() > 0.0);
    }

    #[test]
    fn constructors_reject_invalid_parameters() {
        let invalid_sw = SpaceWeather {
            f107: -1.0,
            ..SpaceWeather::default()
        };
        let cases = [
            DragParameters::from_area_mass(2.2, 1.0, -1.0, quiet_sw(), 100.0),
            DragParameters::from_bc_factor_m2_kg(-1.0, quiet_sw(), 100.0),
            DragParameters::from_bc_factor_m2_kg(BC_FACTOR, invalid_sw, 100.0),
            DragParameters::from_bc_factor_m2_kg(BC_FACTOR, quiet_sw(), -1.0),
        ];

        for case in cases {
            assert!(matches!(case, Err(PropagationError::InvalidInput(_))));
        }
    }

    #[test]
    fn zero_position_is_numerical_failure() {
        let drag = test_drag(BC_FACTOR);
        let state = CartesianState::new(TEST_EPOCH_S, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let error = drag
            .acceleration(&state, &PropagationContext::default())
            .expect_err("zero position fails");
        assert!(matches!(error, PropagationError::NumericalFailure(_)));

        let mapped = map_frame_error(
            "geodetic",
            FrameTransformError::InvalidInput {
                field: "itrs_position_km",
                reason: "components must be finite",
            },
        );
        match mapped {
            PropagationError::NumericalFailure(message) => {
                assert!(message.contains("drag geodetic"), "{message}");
            }
            other => panic!("expected numerical failure, got {other:?}"),
        }
    }

    #[test]
    fn drag_rotation_sign_matches_known_longitude() {
        const LON_TOL_DEG: f64 = 1.0e-12;
        let state = CartesianState::new(TEST_EPOCH_S, [RE_EARTH + 400.0, 0.0, 0.0], [0.0; 3]);
        let orientation =
            orientation_at_state(&PropagationContext::default(), state.epoch_tdb_seconds)
                .expect("valid orientation");
        let geodetic = geodetic_from_orientation(&state, &orientation).expect("valid geodetic");
        let r_itrf = crate::astro::frames::transforms::gcrs_to_itrs_compute(
            state.position_km.x,
            state.position_km.y,
            state.position_km.z,
            &orientation.time_scales(),
            false,
        )
        .expect("independent GCRS to ITRS");
        let (expected_lat, expected_lon, expected_alt) =
            itrs_to_geodetic_compute(r_itrf.0, r_itrf.1, r_itrf.2).expect("independent geodetic");

        assert!((geodetic.lat_deg - expected_lat).abs() <= LON_TOL_DEG);
        assert!((geodetic.lon_deg - expected_lon).abs() <= LON_TOL_DEG);
        assert!((geodetic.alt_km - expected_alt).abs() <= 1.0e-12);
    }

    #[test]
    fn drag_time_inputs_and_kinematics_match_independent_ut1_utc_and_gcrs_to_itrs() {
        use crate::astro::frames::transforms::{gcrs_to_itrs_compute, PolarMotion};
        use crate::astro::time::civil::j2000_seconds_from_split;
        use crate::astro::time::DegradeReason;

        // 2020-06-24 23:59:30.0 UTC lies ~69.184 s behind TDB, so its TDB date
        // is already 2020-06-25 00:00:39.184..., while UTC is still 2020-06-24
        // (day of year 176 in leap year 2020).
        let ts = TimeScales::from_utc(2020, 6, 24, 23, 59, 30.0).expect("valid UTC");
        let epoch_tdb_s = j2000_seconds_from_split(ts.jd_whole, ts.tdb_fraction);
        let state = circular_state(epoch_tdb_s, 400.0, 51.6_f64.to_radians());

        let ctx = PropagationContext::default();
        let orientation = orientation_at_state(&ctx, epoch_tdb_s).expect("orientation");
        let calendar = calendar_from_time_scales(&orientation.time_scales()).expect("calendar");
        assert_eq!(calendar.year, 2020);
        assert_eq!(calendar.doy, 176);
        assert!(
            (calendar.sec_of_day - (23.0 * 3600.0 + 59.0 * 60.0 + 30.0)).abs() < 1.0e-5,
            "sec_of_day {}",
            calendar.sec_of_day
        );

        // Independent GCRS -> ITRS evaluation directly from `TimeScales::from_utc`.
        let (x_itrs, y_itrs, z_itrs) = gcrs_to_itrs_compute(
            state.position_km.x,
            state.position_km.y,
            state.position_km.z,
            &ts,
            false,
        )
        .expect("independent GCRS->ITRS");
        let (lat_deg, lon_deg, alt_km) =
            itrs_to_geodetic_compute(x_itrs, y_itrs, z_itrs).expect("independent geodetic");
        let geodetic = geodetic_from_orientation(&state, &orientation).expect("drag geodetic");
        assert!((geodetic.lat_deg - lat_deg).abs() < 1.0e-10);
        assert!((geodetic.lon_deg - lon_deg).abs() < 1.0e-10);
        assert!((geodetic.alt_km - alt_km).abs() < 1.0e-10);

        // Independent relative velocity via `EarthOrientation::gcrf_to_itrf_state_km`
        // rotated back to GCRF: R^T (R v - omega_itrf x r_itrf) = v - omega_gcrf x r.
        let indep_orientation = EarthOrientation::from_time_scales(&ts).expect("indep orientation");
        let (_, v_itrf) = indep_orientation
            .gcrf_to_itrf_state_km(state.position_array(), state.velocity_array())
            .expect("GCRF->ITRF state");
        let v_rel_indep = mat3_vec3_mul(&indep_orientation.itrf_to_gcrf_matrix(), &v_itrf)
            .expect("ITRF->GCRF vel");
        let v_rel_drag = relative_velocity_km_s(&state, &orientation).expect("drag v_rel");
        let v_rel_diff = (v_rel_drag - Vector3::from(v_rel_indep)).norm();
        assert!(
            v_rel_diff < 1.0e-12,
            "relative velocity diff {v_rel_diff:e} km/s"
        );

        // Independent NRLMSISE-00 + drag acceleration.
        let sw = quiet_sw();
        let indep_density = atmosphere::nrlmsise00_with_lst(
            &NrlmsiseInput {
                year: 2020,
                doy: 176,
                sec: 23.0 * 3600.0 + 59.0 * 60.0 + 30.0,
                alt: alt_km,
                g_lat: lat_deg,
                g_long: lon_deg,
                lst: 0.0,
                f107a: sw.f107a,
                f107: sw.f107,
                ap: sw.ap,
                ap_array: None,
            },
            None,
        )
        .expect("indep density")
        .density();
        let v_m_s = Vector3::from(v_rel_indep) * M_PER_KM;
        let indep_accel = v_m_s * (-0.5 * indep_density * BC_FACTOR * v_m_s.norm()) / M_PER_KM;
        let actual = test_drag(BC_FACTOR)
            .acceleration(&state, &ctx)
            .expect("drag accel");
        let rel_diff = (actual - indep_accel).norm() / indep_accel.norm();
        assert!(
            rel_diff < 1.0e-9,
            "acceleration rel diff {rel_diff:e} ({actual:?} vs {indep_accel:?})"
        );

        // Provider with nonzero polar motion attached to PropagationContext is used.
        let pm = PolarMotion::from_arcseconds(0.35, 0.45).expect("polar motion");
        let pm_ctx = PropagationContext::new().with_body_fixed_frame_provider(Arc::new(
            TdbEarthOrientationProvider::with_polar_motion(pm),
        ));
        let pm_actual = test_drag(BC_FACTOR)
            .acceleration(&state, &pm_ctx)
            .expect("pm drag accel");
        assert_ne!(pm_actual, actual);
        let pm_indep =
            EarthOrientation::from_time_scales_with_polar_motion(&ts, pm).expect("pm orientation");
        let (pm_r_itrf, pm_v_itrf) = pm_indep
            .gcrf_to_itrf_state_km(state.position_array(), state.velocity_array())
            .expect("pm state");
        let pm_v_rel = mat3_vec3_mul(&pm_indep.itrf_to_gcrf_matrix(), &pm_v_itrf).expect("v_rel");
        let (pm_lat, pm_lon, pm_alt) =
            itrs_to_geodetic_compute(pm_r_itrf[0], pm_r_itrf[1], pm_r_itrf[2]).expect("pm geo");
        let pm_density = atmosphere::nrlmsise00_with_lst(
            &NrlmsiseInput {
                year: 2020,
                doy: 176,
                sec: 23.0 * 3600.0 + 59.0 * 60.0 + 30.0,
                alt: pm_alt,
                g_lat: pm_lat,
                g_long: pm_lon,
                lst: 0.0,
                f107a: sw.f107a,
                f107: sw.f107,
                ap: sw.ap,
                ap_array: None,
            },
            None,
        )
        .expect("pm density")
        .density();
        let pm_v_m_s = Vector3::from(pm_v_rel) * M_PER_KM;
        let pm_indep_accel =
            pm_v_m_s * (-0.5 * pm_density * BC_FACTOR * pm_v_m_s.norm()) / M_PER_KM;
        assert!(
            (pm_actual - pm_indep_accel).norm() / pm_indep_accel.norm() < 1.0e-9,
            "polar-motion accel mismatch"
        );

        // Outside UT1 table: default context uses permissive long-term Delta T
        // and records the departure; an explicit strict provider refuses.
        let future_ts = TimeScales::from_utc(2100, 1, 1, 0, 0, 0.0).expect("future UTC");
        let future_tdb = j2000_seconds_from_split(future_ts.jd_whole, future_ts.tdb_fraction);
        let future_state = circular_state(future_tdb, 400.0, 51.6_f64.to_radians());
        let default_ctx = PropagationContext::default();
        let future_accel = test_drag(BC_FACTOR)
            .acceleration(&future_state, &default_ctx)
            .expect("default context drag outside UT1 table");
        assert!(future_accel.norm() > 0.0);
        assert_eq!(
            default_ctx.ut1_departure(),
            Some(DegradeReason::AfterCoverage)
        );
        let strict_ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::new()));
        assert!(matches!(
            test_drag(BC_FACTOR).acceleration(&future_state, &strict_ctx),
            Err(PropagationError::Ut1OutsideCoverage(
                DegradeReason::AfterCoverage
            ))
        ));
    }

    #[test]
    fn drag_outside_embedded_ut1_table_uses_permissive_default_and_honors_strict_provider() {
        use crate::astro::time::DegradeReason;

        let drag = test_drag(BC_FACTOR);
        let drag_params = test_drag_parameters(BC_FACTOR);
        // 2028-01-01 00:00:00 TDB, 2027-07-04 00:00:00 TDB, 1973-01-01 00:00:00 TDB.
        let cases = [
            (883_569_600.0, DegradeReason::AfterCoverage),
            (867_931_200.0, DegradeReason::AfterCoverage),
            (-852_033_600.0, DegradeReason::BeforeCoverage),
        ];

        let strict_ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::new()));

        for (epoch_tdb, expected_reason) in cases {
            let state = circular_state(epoch_tdb, 400.0, 51.6_f64.to_radians());

            let default_ctx = PropagationContext::default();
            let accel = drag
                .acceleration(&state, &default_ctx)
                .expect("default context accepts out-of-table epoch");
            assert!(accel.norm() > 0.0 && vector_is_finite(&accel));
            assert_eq!(default_ctx.ut1_departure(), Some(expected_reason));

            let prop_result = StatePropagator::new(
                state.epoch_tdb_seconds,
                state.position_array(),
                state.velocity_array(),
                ForceModelKind::two_body(),
                IntegratorKind::Dp54,
            )
            .with_drag(drag_params)
            .propagate_to(epoch_tdb + 60.0)
            .expect("default drag propagation succeeds outside UT1 table");
            assert_eq!(prop_result.ut1_degraded, Some(expected_reason));

            let err = drag
                .acceleration(&state, &strict_ctx)
                .expect_err("strict caller provider must refuse out-of-table epoch");
            assert!(
                matches!(err, PropagationError::Ut1OutsideCoverage(reason) if reason == expected_reason),
                "epoch {epoch_tdb}: expected {expected_reason:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn sourced_drag_looks_up_space_weather_on_utc_across_midnight() {
        use crate::astro::space_weather::parse_csv;
        use crate::astro::time::civil::j2000_seconds_from_split;

        const MINI_CSV: &str = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n\
2020-06-23,2549,1,10,10,10,10,10,10,10,10,80,4,4,4,4,4,4,4,4,4,0.1,0,0,70.0,70.0,OBS,70.0,70.0,70.0,70.0\n\
2020-06-24,2549,2,10,10,10,10,10,10,10,10,80,4,4,4,4,4,4,4,4,4,0.1,0,0,220.0,220.0,OBS,80.0,80.0,80.0,80.0\n\
2020-06-25,2549,3,60,60,60,60,60,60,60,60,480,80,80,80,80,80,80,80,80,80,1.8,7,150,240.0,240.0,OBS,200.0,200.0,200.0,200.0\n";

        let table = Arc::new(parse_csv(MINI_CSV).expect("parse csv").value);
        // 2020-06-24 23:59:30 UTC -> TDB is 2020-06-25 00:00:39.184... (next day!).
        let ts = TimeScales::from_utc(2020, 6, 24, 23, 59, 30.0).expect("valid UTC");
        let epoch_tdb_s = j2000_seconds_from_split(ts.jd_whole, ts.tdb_fraction);
        let state = circular_state(epoch_tdb_s, 400.0, 51.6_f64.to_radians());

        let sourced = SourcedDragForce::new(
            test_drag_parameters(BC_FACTOR),
            SpaceWeatherSource::Table(table),
        );
        let ctx = PropagationContext::default();
        let actual = sourced.acceleration(&state, &ctx).expect("sourced drag");

        // On UTC day 2020-06-24: F10.7 = previous day (2020-06-23) = 70.0,
        // F10.7a = today (2020-06-24) = 80.0, Ap = today (2020-06-24) = 4.0.
        let utc_day_sw = SpaceWeather {
            f107: 70.0,
            f107a: 80.0,
            ap: 4.0,
        };
        let expected_utc = DragForce::from_bc_factor_m2_kg(
            BC_FACTOR,
            utc_day_sw,
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("utc drag")
        .acceleration(&state, &ctx)
        .expect("utc accel");
        // On TDB day 2020-06-25 it would have been F10.7=220, F10.7a=200, Ap=80.
        let tdb_day_sw = SpaceWeather {
            f107: 220.0,
            f107a: 200.0,
            ap: 80.0,
        };
        let wrong_tdb = DragForce::from_bc_factor_m2_kg(
            BC_FACTOR,
            tdb_day_sw,
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("tdb drag")
        .acceleration(&state, &ctx)
        .expect("tdb accel");

        assert_eq!(actual.x.to_bits(), expected_utc.x.to_bits());
        assert_eq!(actual.y.to_bits(), expected_utc.y.to_bits());
        assert_eq!(actual.z.to_bits(), expected_utc.z.to_bits());
        assert!(
            (actual - wrong_tdb).norm() > 0.5 * actual.norm(),
            "TDB-day space weather should differ strongly from UTC-day"
        );
    }

    #[test]
    fn drag_golden_case_bits() {
        // Golden case source: this module's drag path, using NRLMSISE-00 release
        // 20041227 density through atmosphere::nrlmsise00_with_lst. Inputs:
        // epoch 646315200.25 s TDB (2020-06-24 23:58:51.065721035 UTC), position
        // [6778.137, 0, 0] km GCRF, inclination 51.6 deg, B 0.02 m^2/kg,
        // F10.7/F10.7a/Ap 150/150/4. Intermediates: UTC year 2020, doy 176,
        // sec 86331.065721035004, geodetic lat 0.11286438798161325 deg,
        // lon 86.95568673113613 deg, alt 400.00008231857566 km,
        // density 1.970490020559495e-12 kg/m^3, v_rel [0, 4.269039280677301,
        // 6.009795934015082] km/s, acceleration [-0, -6.201169074140698e-10,
        // -8.729776944567759e-10] km/s^2.
        let state = circular_state(TEST_EPOCH_S, 400.0, 51.6_f64.to_radians());
        let accel = test_drag(BC_FACTOR)
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");

        assert_eq!(
            [accel.x.to_bits(), accel.y.to_bits(), accel.z.to_bits()],
            [
                9_223_372_036_854_775_808,
                13_692_436_670_303_356_892,
                13_694_882_189_777_148_921,
            ]
        );
    }
}
