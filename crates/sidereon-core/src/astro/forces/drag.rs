//! Atmospheric drag over NRLMSISE-00 density.
//!
//! `CartesianState::epoch_tdb_seconds` is interpreted here as continuous seconds
//! past the J2000 epoch. The drag model treats those seconds directly as the UT
//! epoch for NRLMSISE-00 calendar inputs and for the GMST-only ECI to ECEF
//! rotation. For modern satellite drag, the neglected TDB-UT1 and UTC-UT1
//! offsets are at most about 70 s: about 8e-4 day, 0.02 h of local-solar-time
//! phase, 2e-6 year of seasonal phase, and 5.1e-3 rad or 0.29 deg of longitude.
//! These are well below the empirical density uncertainty of NRLMSISE-00.
//!
//! Wind is neglected and the atmosphere is treated as co-rotating with Earth,
//! following the standard Vallado rotating-atmosphere drag convention. Density
//! caching is intentionally left for a future optimization so each RHS
//! evaluation is reproducible from the instantaneous state.

use crate::astro::atmosphere::{self, NrlmsiseInput, MAX_ALTITUDE_KM};
use crate::astro::constants::{
    earth::OMEGA_E_DOT_RAD_S,
    time::{DAYS_PER_JULIAN_YEAR, SECONDS_PER_DAY},
    units::M_PER_KM,
};
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::frames::transforms::{
    greenwich_mean_sidereal_time_radians_from_j2000_seconds, itrs_to_geodetic_compute,
    FrameTransformError,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::space_weather::{SpaceWeatherError, SpaceWeatherTable};
use crate::astro::state::CartesianState;
use crate::astro::time::civil::{civil_from_j2000_seconds, day_of_year_int, second_of_day};
use nalgebra::Vector3;
use std::sync::Arc;

const MAX_EPOCH_OFFSET_S: f64 = 1000.0 * DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY;

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
    /// Per-epoch values from a parsed CelesTrak table.
    Table(Arc<SpaceWeatherTable>),
}

impl SpaceWeatherSource {
    /// Resolve space-weather inputs at one J2000-second epoch.
    pub fn at(&self, epoch_j2000_s: f64) -> Result<SpaceWeather, SpaceWeatherError> {
        match self {
            Self::Fixed(space_weather) => Ok(*space_weather),
            Self::Table(table) => table.space_weather_at(epoch_j2000_s),
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
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        drag_acceleration(
            self.space_weather,
            self.bc_factor_m2_kg,
            self.cutoff_altitude_km,
            state,
        )
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
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let space_weather = self.source.at(state.epoch_tdb_seconds).map_err(|error| {
            PropagationError::ForceModelFailure(format!("space weather: {error}"))
        })?;
        drag_acceleration(
            space_weather,
            self.bc_factor_m2_kg,
            self.cutoff_altitude_km,
            state,
        )
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

pub(crate) fn geodetic_altitude_km(state: &CartesianState) -> Result<f64, PropagationError> {
    validate_drag_state(state)?;
    Ok(geodetic_from_validated_state(state)?.alt_km)
}

pub(crate) fn map_frame_error(
    context: &'static str,
    error: FrameTransformError,
) -> PropagationError {
    PropagationError::NumericalFailure(format!("drag {context} failed: {error}"))
}

fn drag_acceleration(
    space_weather: SpaceWeather,
    bc_factor_m2_kg: f64,
    cutoff_altitude_km: f64,
    state: &CartesianState,
) -> Result<Vector3<f64>, PropagationError> {
    validate_drag_state(state)?;
    let calendar = calendar_from_epoch(state.epoch_tdb_seconds);
    let geodetic = geodetic_from_validated_state(state)?;

    if geodetic.alt_km <= cutoff_altitude_km || geodetic.alt_km > MAX_ALTITUDE_KM {
        return Ok(Vector3::zeros());
    }

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

    let v_rel_km_s = relative_velocity_km_s(state);
    if !vector_is_finite(&v_rel_km_s) {
        return Err(PropagationError::NumericalFailure(
            "drag relative velocity not finite".to_string(),
        ));
    }
    let v_rel_m_s = v_rel_km_s * M_PER_KM;
    let speed_m_s = v_rel_m_s.norm();
    if !speed_m_s.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "drag relative speed not finite".to_string(),
        ));
    }

    let accel_m_s2 = v_rel_m_s * (-0.5 * density * bc_factor_m2_kg * speed_m_s);
    let accel_km_s2 = accel_m_s2 / M_PER_KM;
    if !vector_is_finite(&accel_km_s2) {
        return Err(PropagationError::NumericalFailure(
            "drag acceleration not finite".to_string(),
        ));
    }
    Ok(accel_km_s2)
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
    if state.epoch_tdb_seconds.abs() > MAX_EPOCH_OFFSET_S {
        return Err(PropagationError::InvalidInput(
            "epoch_tdb_seconds outside +/-1000 Julian years from J2000".to_string(),
        ));
    }
    if !vector_is_finite(&state.position_km) {
        return Err(PropagationError::InvalidInput(
            "position_km not finite".to_string(),
        ));
    }
    if !vector_is_finite(&state.velocity_km_s) {
        return Err(PropagationError::InvalidInput(
            "velocity_km_s not finite".to_string(),
        ));
    }
    if state.position_km.norm_squared() == 0.0 {
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
}

fn calendar_from_epoch(epoch_tdb_seconds: f64) -> DragCalendar {
    let sec_i = epoch_tdb_seconds.floor() as i64;
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(sec_i);
    let sec_of_day = second_of_day(hour as i32, minute as i32, second as f64)
        + (epoch_tdb_seconds - sec_i as f64);
    DragCalendar {
        year: year as i32,
        doy: day_of_year_int(year as i32, month as i32, day as i32) as i32,
        sec_of_day,
    }
}

#[derive(Debug, Clone, Copy)]
struct DragGeodetic {
    lat_deg: f64,
    lon_deg: f64,
    alt_km: f64,
}

fn geodetic_from_validated_state(state: &CartesianState) -> Result<DragGeodetic, PropagationError> {
    let theta = greenwich_mean_sidereal_time_radians_from_j2000_seconds(state.epoch_tdb_seconds)
        .map_err(|error| map_frame_error("gmst", error))?;
    let r_ecef = eci_to_ecef_gmst(state.position_km, theta);
    if !vector_is_finite(&r_ecef) {
        return Err(PropagationError::NumericalFailure(
            "drag ECEF position not finite".to_string(),
        ));
    }
    let (lat_deg, lon_deg, alt_km) = itrs_to_geodetic_compute(r_ecef.x, r_ecef.y, r_ecef.z)
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

fn eci_to_ecef_gmst(position_km: Vector3<f64>, theta: f64) -> Vector3<f64> {
    let c = libm::cos(theta);
    let s = libm::sin(theta);
    Vector3::new(
        c * position_km.x + s * position_km.y,
        -s * position_km.x + c * position_km.y,
        position_km.z,
    )
}

fn relative_velocity_km_s(state: &CartesianState) -> Vector3<f64> {
    let omega_cross_r = Vector3::new(
        -OMEGA_E_DOT_RAD_S * state.position_km.y,
        OMEGA_E_DOT_RAD_S * state.position_km.x,
        0.0,
    );
    state.velocity_km_s - omega_cross_r
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
        let theta =
            greenwich_mean_sidereal_time_radians_from_j2000_seconds(epoch).expect("valid gmst");
        let c = libm::cos(theta);
        let s = libm::sin(theta);
        let x_eci = c * x_ecef - s * y_ecef;
        let y_eci = s * x_ecef + c * y_ecef;
        let r = RE_EARTH + altitude_km;
        let v = (MU_EARTH / r).sqrt();
        CartesianState::new(epoch, [x_eci, y_eci, z_ecef], [0.0, v, 0.0])
    }

    fn density_for_state(state: &CartesianState, sw: SpaceWeather) -> f64 {
        validate_drag_state(state).expect("valid drag state");
        let calendar = calendar_from_epoch(state.epoch_tdb_seconds);
        let geodetic = geodetic_from_validated_state(state).expect("valid geodetic");
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
        let rho = density_for_state(state, quiet_sw());
        let v_rel_m_s = relative_velocity_km_s(state) * M_PER_KM;
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
        let accel = drag
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");
        let v_rel = relative_velocity_km_s(&state);

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
        // -5.041449755563205e-3 m/s. Expected averaged rate:
        // -5.013291857709087e-3 m/s. Relative difference: 5.62e-3.
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
        let cutoff = geodetic_altitude_km(&cutoff_state).expect("cutoff altitude");
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
        let theta = greenwich_mean_sidereal_time_radians_from_j2000_seconds(TEST_EPOCH_S)
            .expect("valid gmst");
        let state = CartesianState::new(TEST_EPOCH_S, [RE_EARTH + 400.0, 0.0, 0.0], [0.0; 3]);
        let geodetic = geodetic_from_validated_state(&state).expect("valid geodetic");
        let expected_lon = (-theta).to_degrees();
        let expected_lon = ((expected_lon + 180.0).rem_euclid(360.0)) - 180.0;

        assert!((geodetic.lon_deg - expected_lon).abs() <= LON_TOL_DEG);
    }

    #[test]
    fn drag_golden_case_bits() {
        // Golden case source: this module's drag path, using NRLMSISE-00 release
        // 20041227 density through atmosphere::nrlmsise00_with_lst. Inputs:
        // epoch 646315200.25 s, position [6778.137, 0, 0] km, inclination
        // 51.6 deg, B 0.02 m^2/kg, F10.7/F10.7a/Ap 150/150/4. Intermediates:
        // year 2020, doy 177, sec 0.25, GMST 4.775165029523421 rad,
        // geodetic lat 0 deg, lon 86.4031973298448 deg, alt 400 km,
        // density 1.9576468755557382e-12 kg/m^3, v_rel [0,
        // 4.269038333748105, 6.00979886918909] km/s, acceleration [-0,
        // -6.160751630773568e-10, -8.672884919136098e-10] km/s^2.
        let state = circular_state(TEST_EPOCH_S, 400.0, 51.6_f64.to_radians());
        let accel = test_drag(BC_FACTOR)
            .acceleration(&state, &PropagationContext::default())
            .expect("drag acceleration");

        assert_eq!(
            [accel.x.to_bits(), accel.y.to_bits(), accel.z.to_bits()],
            [
                9_223_372_036_854_775_808,
                13_692_397_580_950_677_421,
                13_694_827_167_186_369_312,
            ]
        );
    }
}
