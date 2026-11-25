use super::{AttitudeSample, AttitudeTrajectory};
use crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds;
use crate::astro::constants::units::M_PER_KM;
use crate::astro::de::DeEphemeris;
use crate::astro::frames::EarthOrientationProvider;
use crate::astro::propagator::analytic::{BrouwerLyddanePropagator, EcksteinHechlerPropagator};
use crate::astro::propagator::dense_output::DenseOutput;
use crate::astro::time::{
    j2000_seconds, j2000_seconds_from_split, timescale_offset_s, tt_from_tdb_j2000_seconds,
    TimeScale, TimeScales, GLONASST_MINUS_UTC_S,
};
use crate::ephemeris::{BroadcastEphemeris, Sp3};
use crate::observables::ObservableEphemerisSource;
use crate::orbit_scoring::{
    reference_velocity, validate_velocity_half_step, velocity_instants,
    velocity_instants_are_valid, OrbitScoreError, OrbitSource, DEFAULT_VELOCITY_HALF_STEP_S,
};
use crate::GnssSatelliteId;

/// Satellite state and Sun position for a borrowed orbit, on the absolute TDB
/// seconds past J2000 axis.
///
/// Samples contain GCRF vectors in metres and metres per second. Analytic
/// propagators must be defined in GCRF. Earth-fixed sources must supply
/// positions in a terrestrial frame aligned with the provider's ITRF. Their
/// positions are queried on the source's time axis, and their velocities are
/// differenced on that axis and transformed with the provider's rotation and
/// rotation rate.
///
/// A sample is unavailable when the orbit, Earth orientation or Sun ephemeris
/// cannot supply it. Some [`super::gnss_attitude`] laws query neighbouring
/// portions of the orbit as well as the requested epoch.
///
/// # Example
///
/// ```
/// use sidereon_core::astro::attitude::gnss::{
///     gnss_attitude, GnssAttitudeLaw, OrbitAttitudeTrajectory,
/// };
/// use sidereon_core::astro::frames::TdbEarthOrientationProvider;
/// use sidereon_core::astro::time::{tdb_from_tt_j2000_seconds, timescale_offset_s, TimeScale};
/// use sidereon_core::ephemeris::Sp3;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut sp3_text = String::from("\
/// # #aP2020  6 24  0  0  0.00000000      11 ORBIT IGS14 FIT  IGS
/// # ## 2111 259200.00000000   900.00000000 59024 0.0000000000000
/// # +    1     1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
/// # ");
/// # for i in 0..11 {
/// #     use std::fmt::Write;
/// #     writeln!(sp3_text, "*  2020  6 24 {:2} {:2}  0.00000000", i / 4, i % 4 * 15)?;
/// #     let offset = 100.0 * f64::from(i);
/// #     writeln!(sp3_text, "P  1{:14.6}{:14.6}{:14.6}{:14.6}",
/// #         15000.0 + offset, -20000.0 - offset, 5000.0 + offset, 0.0)?;
/// # }
/// # sp3_text.push_str("EOF\n");
/// # let sp3_bytes = sp3_text.as_bytes();
/// let product = Sp3::parse(sp3_bytes)?;
/// let satellite = "G01".parse()?;
/// let orientation = TdbEarthOrientationProvider::new();
/// let trajectory = OrbitAttitudeTrajectory::from_sp3(&product, satellite, &orientation);
/// let gpst = product.epochs_j2000_seconds()[0] + 450.0;
/// let tdb = tdb_from_tt_j2000_seconds(gpst + timescale_offset_s(TimeScale::Gpst, TimeScale::Tt)?);
/// let attitude = gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &trajectory, tdb)?;
/// assert!(attitude.yaw_rad.is_finite());
/// # Ok(())
/// # }
/// ```
pub struct OrbitAttitudeTrajectory<'a> {
    source: Source<'a>,
    de_ephemeris: Option<&'a DeEphemeris>,
    velocity_half_step_s: f64,
}

enum Source<'a> {
    EcksteinHechler(&'a EcksteinHechlerPropagator),
    BrouwerLyddane(&'a BrouwerLyddanePropagator),
    DenseOutput(&'a DenseOutput),
    EarthFixed {
        orbit: EarthFixedOrbit<'a>,
        satellite: GnssSatelliteId,
        orientation: &'a dyn EarthOrientationProvider,
    },
}

enum EarthFixedOrbit<'a> {
    Sp3(&'a Sp3),
    Broadcast(&'a BroadcastEphemeris),
    OrbitSource(&'a dyn OrbitSource),
}

impl EarthFixedOrbit<'_> {
    fn time_scale(&self) -> TimeScale {
        match self {
            Self::Sp3(product) => product.header.time_scale,
            Self::Broadcast(_) => TimeScale::Gpst,
            Self::OrbitSource(source) => source.time_scale(),
        }
    }

    fn position(&self, satellite: GnssSatelliteId, epoch: f64) -> Option<[f64; 3]> {
        match self {
            Self::Sp3(product) => Some(
                product
                    .position_at_j2000_seconds(satellite, epoch)
                    .ok()?
                    .position
                    .as_array(),
            ),
            Self::Broadcast(ephemeris) => Some(
                ephemeris
                    .observable_state_at_j2000_s(satellite, epoch)
                    .ok()?
                    .position_ecef_m,
            ),
            Self::OrbitSource(source) => Some(
                source
                    .state_at_j2000_s(satellite, epoch)
                    .ok()?
                    .position_ecef_m,
            ),
        }
    }
}

impl<'a> OrbitAttitudeTrajectory<'a> {
    fn new(source: Source<'a>) -> Self {
        Self {
            source,
            de_ephemeris: None,
            velocity_half_step_s: DEFAULT_VELOCITY_HALF_STEP_S,
        }
    }

    /// Borrow an Eckstein–Hechler propagator whose states are in GCRF.
    pub fn from_eckstein_hechler(propagator: &'a EcksteinHechlerPropagator) -> Self {
        Self::new(Source::EcksteinHechler(propagator))
    }

    /// Borrow a Brouwer–Lyddane propagator whose states are in GCRF.
    pub fn from_brouwer_lyddane(propagator: &'a BrouwerLyddanePropagator) -> Self {
        Self::new(Source::BrouwerLyddane(propagator))
    }

    /// Borrow DP5(4) dense output in GCRF. The accepted-step polynomials
    /// supply position and velocity continuously within the propagation arc.
    pub fn from_dense_output(output: &'a DenseOutput) -> Self {
        Self::new(Source::DenseOutput(output))
    }

    /// Borrow a parsed SP3 product and select one satellite.
    pub fn from_sp3(
        product: &'a Sp3,
        satellite: GnssSatelliteId,
        orientation: &'a dyn EarthOrientationProvider,
    ) -> Self {
        Self::from_earth_fixed(EarthFixedOrbit::Sp3(product), satellite, orientation)
    }

    /// Borrow a broadcast navigation set and select one satellite.
    pub fn from_broadcast(
        ephemeris: &'a BroadcastEphemeris,
        satellite: GnssSatelliteId,
        orientation: &'a dyn EarthOrientationProvider,
    ) -> Self {
        Self::from_earth_fixed(
            EarthFixedOrbit::Broadcast(ephemeris),
            satellite,
            orientation,
        )
    }

    /// Borrow any Earth-fixed orbit source and select one satellite. This
    /// accepts [`crate::orbit_scoring::PredictedSp3`],
    /// [`crate::orbit_scoring::BroadcastOrbit`] and
    /// [`crate::orbit_scoring::OrbitTrajectory`].
    pub fn from_orbit_source(
        source: &'a dyn OrbitSource,
        satellite: GnssSatelliteId,
        orientation: &'a dyn EarthOrientationProvider,
    ) -> Self {
        Self::from_earth_fixed(EarthFixedOrbit::OrbitSource(source), satellite, orientation)
    }

    fn from_earth_fixed(
        orbit: EarthFixedOrbit<'a>,
        satellite: GnssSatelliteId,
        orientation: &'a dyn EarthOrientationProvider,
    ) -> Self {
        Self::new(Source::EarthFixed {
            orbit,
            satellite,
            orientation,
        })
    }

    /// Use a JPL DE ephemeris for the Sun in place of
    /// [`sun_moon_gcrs_at_tdb_seconds`].
    #[must_use]
    pub fn with_de_ephemeris(mut self, ephemeris: &'a DeEphemeris) -> Self {
        self.de_ephemeris = Some(ephemeris);
        self
    }

    /// Set the velocity difference half step for Earth-fixed sources, in
    /// seconds of the source's time scale. The default is
    /// [`DEFAULT_VELOCITY_HALF_STEP_S`]. The difference is centred when both
    /// neighbours are available and one-sided when just one is available.
    /// For a node-only orbit, choose a half step that reaches adjacent nodes.
    ///
    /// Returns an error for a non-finite step or one below
    /// [`crate::orbit_scoring::MIN_VELOCITY_HALF_STEP_S`]. If the step rounds
    /// away at a query epoch, the sample is unavailable.
    pub fn with_velocity_half_step_s(mut self, half_step_s: f64) -> Result<Self, OrbitScoreError> {
        validate_velocity_half_step(half_step_s)?;
        self.velocity_half_step_s = half_step_s;
        Ok(self)
    }

    fn state_km(&self, tdb: f64) -> Option<([f64; 3], [f64; 3])> {
        let state = match &self.source {
            Source::EcksteinHechler(propagator) => propagator.propagate(tdb).ok()?,
            Source::BrouwerLyddane(propagator) => propagator.propagate(tdb).ok()?,
            Source::DenseOutput(output) => {
                let start = output.segments.first()?.t_start;
                let end = output.segments.last()?.y_end.epoch_tdb_seconds;
                if !(start.min(end)..=start.max(end)).contains(&tdb) {
                    return None;
                }
                output.eval(tdb).ok()?
            }
            Source::EarthFixed {
                orbit,
                satellite,
                orientation,
            } => {
                let orientation = orientation.orientation_at_tdb_seconds(tdb).ok()?;
                let t = source_seconds(tdb, orbit.time_scale(), &orientation.time_scales())?;
                let (minus, plus) = velocity_instants(t, self.velocity_half_step_s);
                if !velocity_instants_are_valid(t, (minus, plus)) {
                    return None;
                }
                let position_m = orbit.position(*satellite, t)?;
                let velocity_m_s = reference_velocity(
                    orbit.position(*satellite, plus),
                    orbit.position(*satellite, minus),
                    position_m,
                    t,
                    (minus, plus),
                )?;
                return orientation
                    .itrf_to_gcrf_state_km(
                        position_m.map(|x| x / M_PER_KM),
                        velocity_m_s.map(|x| x / M_PER_KM),
                    )
                    .ok();
            }
        };
        Some((state.position_km.into(), state.velocity_km_s.into()))
    }
}

impl AttitudeTrajectory for OrbitAttitudeTrajectory<'_> {
    fn sample(&self, t_s: f64) -> Option<AttitudeSample> {
        if !t_s.is_finite() {
            return None;
        }
        let (position_km, velocity_km_s) = self.state_km(t_s)?;
        let position_m = position_km.map(|x| x * M_PER_KM);
        let velocity_m_s = velocity_km_s.map(|x| x * M_PER_KM);
        if !position_m
            .into_iter()
            .chain(velocity_m_s)
            .all(f64::is_finite)
        {
            return None;
        }
        let sun = match self.de_ephemeris {
            Some(ephemeris) => ephemeris.sun_moon_positions_at_tdb_seconds(t_s).ok()?,
            None => sun_moon_gcrs_at_tdb_seconds(t_s).ok()?,
        };
        Some(AttitudeSample {
            position_m,
            velocity_m_s,
            sun_position_m: sun.sun,
        })
    }
}

fn source_seconds(tdb: f64, scale: TimeScale, scales: &TimeScales) -> Option<f64> {
    let t = match scale {
        TimeScale::Tdb => tdb,
        TimeScale::Tcg => j2000_seconds_from_split(scales.jd_whole, scales.tcg_fraction()),
        TimeScale::Tcb => j2000_seconds_from_split(scales.jd_whole, scales.tcb_fraction()),
        TimeScale::Utc | TimeScale::Glonasst => {
            let (year, month, day, hour, minute, second) = scales.utc_calendar_fields()?;
            let utc = j2000_seconds(year, month, day, hour, minute, second);
            if scale == TimeScale::Glonasst {
                utc + GLONASST_MINUS_UTC_S
            } else {
                utc
            }
        }
        _ => tt_from_tdb_j2000_seconds(tdb) + timescale_offset_s(TimeScale::Tt, scale).ok()?,
    };
    t.is_finite().then_some(t)
}
