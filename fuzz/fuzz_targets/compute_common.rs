#![allow(dead_code)]

use arbitrary::{Arbitrary, Unstructured};
use nalgebra::{DMatrix, DVector};

pub const MAX_VEC: usize = 8;
pub const MAX_MATRIX_DIM: usize = 4;
pub const MAX_EPOCHS: usize = 4;
pub const MAX_OBS: usize = 8;

pub trait Finite {
    fn push_finite_values(&self, out: &mut Vec<f64>);
}

pub fn fuzz_input<T>(data: &[u8]) -> Option<T>
where
    T: for<'a> Arbitrary<'a>,
{
    T::arbitrary(&mut Unstructured::new(data)).ok()
}

pub fn assert_success<T: Finite>(api: &str, value: T) {
    let mut values = Vec::new();
    value.push_finite_values(&mut values);
    for value in values {
        assert!(value.is_finite(), "ok-nonfinite {api}");
    }
}

pub fn assert_ok_finite_or_err<T, E>(api: &str, result: Result<T, E>)
where
    T: Finite,
{
    if let Ok(value) = result {
        assert_success(api, value);
    }
}

pub fn assert_option_finite<T>(api: &str, result: Option<T>)
where
    T: Finite,
{
    if let Some(value) = result {
        assert_success(api, value);
    }
}

pub fn assert_ok_or_err<T, E>(_api: &str, result: Result<T, E>) {
    let _ = result;
}

pub fn assert_all_finite(api: &str, values: impl IntoIterator<Item = f64>) {
    for value in values {
        assert!(value.is_finite(), "ok-nonfinite {api}");
    }
}

pub fn cap_vec<T>(mut values: Vec<T>, max: usize) -> Vec<T> {
    values.truncate(max);
    values
}

pub fn matrix_from_flat(values: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let rows = rows.clamp(1, MAX_MATRIX_DIM);
    let cols = cols.clamp(1, MAX_MATRIX_DIM);
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = values.get(r * cols + c).copied().unwrap_or(0.0);
        }
    }
    out
}

pub fn square_from_flat(values: &[f64], n: usize) -> Vec<Vec<f64>> {
    matrix_from_flat(values, n, n)
}

pub fn flat_square(values: &[f64], n: usize) -> Vec<f64> {
    let n = n.clamp(1, MAX_MATRIX_DIM);
    let mut out = vec![0.0; n * n];
    for (idx, value) in out.iter_mut().enumerate() {
        *value = values.get(idx).copied().unwrap_or(0.0);
    }
    out
}

pub fn bounded_abs_or_raw(value: f64, max_abs: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-max_abs, max_abs)
    } else {
        value
    }
}

pub fn bounded_positive_or_raw(value: f64, default: f64, max: f64) -> f64 {
    if value.is_finite() {
        value.abs().max(default).min(max)
    } else {
        value
    }
}

pub fn bounded_usize(value: u8, min: usize, max: usize) -> usize {
    min + usize::from(value) % (max - min + 1)
}

pub fn scale_index(value: u8, len: usize) -> usize {
    if len == 0 {
        0
    } else {
        usize::from(value) % len
    }
}

pub fn time_scale(value: u8) -> sidereon_core::astro::time::TimeScale {
    use sidereon_core::astro::time::TimeScale;
    match value % 7 {
        0 => TimeScale::Utc,
        1 => TimeScale::Tai,
        2 => TimeScale::Tt,
        3 => TimeScale::Tdb,
        4 => TimeScale::Gpst,
        5 => TimeScale::Gst,
        _ => TimeScale::Bdt,
    }
}

pub fn gnss_system(value: u8) -> sidereon_core::GnssSystem {
    use sidereon_core::GnssSystem;
    match value % 7 {
        0 => GnssSystem::Gps,
        1 => GnssSystem::Glonass,
        2 => GnssSystem::Galileo,
        3 => GnssSystem::BeiDou,
        4 => GnssSystem::Qzss,
        5 => GnssSystem::Navic,
        _ => GnssSystem::Sbas,
    }
}

pub fn sat_id(
    system: u8,
    prn: u8,
) -> Result<sidereon_core::GnssSatelliteId, sidereon_core::SatelliteIdError> {
    sidereon_core::GnssSatelliteId::new(gnss_system(system), prn % 64 + 1)
}

pub fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

impl Finite for f64 {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(*self);
    }
}

impl Finite for i64 {
    fn push_finite_values(&self, _out: &mut Vec<f64>) {}
}

impl Finite for usize {
    fn push_finite_values(&self, _out: &mut Vec<f64>) {}
}

impl<T: Finite> Finite for &T {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        (*self).push_finite_values(out);
    }
}

impl<T: Finite> Finite for Option<T> {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        if let Some(value) = self {
            value.push_finite_values(out);
        }
    }
}

impl<T: Finite> Finite for Vec<T> {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        for value in self {
            value.push_finite_values(out);
        }
    }
}

impl<const N: usize> Finite for [f64; N] {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend(self);
    }
}

impl<const R: usize, const C: usize> Finite for [[f64; C]; R] {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        for row in self {
            row.push_finite_values(out);
        }
    }
}

impl<A: Finite, B: Finite> Finite for (A, B) {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.0.push_finite_values(out);
        self.1.push_finite_values(out);
    }
}

impl<A: Finite, B: Finite, C: Finite> Finite for (A, B, C) {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.0.push_finite_values(out);
        self.1.push_finite_values(out);
        self.2.push_finite_values(out);
    }
}

impl Finite for DVector<f64> {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend(self.iter().copied());
    }
}

impl Finite for DMatrix<f64> {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend(self.iter().copied());
    }
}

impl Finite for sidereon_core::ItrfPositionM {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.as_array().push_finite_values(out);
    }
}

impl Finite for sidereon_core::ItrfVelocityMS {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.as_array().push_finite_values(out);
    }
}

impl Finite for sidereon_core::Wgs84Geodetic {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.lat_rad, self.lon_rad, self.height_m]);
    }
}

impl Finite for sidereon_core::astro::time::TimeScales {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.jd_whole,
            self.ut1_fraction,
            self.tt_fraction,
            self.tdb_fraction,
            self.jd_ut1,
            self.jd_tt,
            self.jd_tdb,
        ]);
    }
}

impl<T: Finite> Finite for sidereon_core::astro::time::Validated<T> {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.value.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::time::JulianDateSplit {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.jd_whole, self.fraction, self.to_jd()]);
    }
}

impl Finite for sidereon_core::astro::time::Duration {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.as_seconds());
    }
}

impl Finite for sidereon_core::astro::time::GnssWeekTow {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.tow_s);
    }
}

impl Finite for sidereon_core::astro::time::Instant {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.julian_date().push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::CartesianState {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.epoch_tdb_seconds);
        self.position_array().push_finite_values(out);
        self.velocity_array().push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::propagator::result::PropagationPoint {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.epoch_tdb_seconds);
        self.position_km.push_finite_values(out);
        self.velocity_km_s.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::propagator::result::PropagationResult {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.final_state.push_finite_values(out);
        self.points.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::covariance::Covariance6 {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.as_matrix().push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::math::least_squares::FdStep {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.sign_x0, self.h, self.dx]);
        self.x_perturbed.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::math::least_squares::LeastSquaresReport {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.x.push_finite_values(out);
        self.residual.push_finite_values(out);
        out.extend([self.cost, self.optimality_inf]);
        self.jacobian.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::sgp4::Prediction {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.position.push_finite_values(out);
        self.velocity.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::sgp4::JulianDate {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.0, self.1]);
    }
}

impl Finite for sidereon_core::orbit::CalendarEpoch {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.second);
    }
}

impl Finite for sidereon_core::orbit::Elements {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.epoch.push_finite_values(out);
        out.extend([
            self.a_m,
            self.e,
            self.i_rad,
            self.raan_rad,
            self.raan_rate_rad_s,
            self.raan_rate_j2_rad_s,
            self.arg_lat_rad,
            self.mean_motion_rad_s,
            self.h,
            self.k,
            self.arg_perigee_rad,
        ]);
    }
}

impl Finite for sidereon_core::orbit::FitStats {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.rms_m, self.max_m]);
    }
}

impl Finite for sidereon_core::orbit::ReducedOrbit {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.elements.push_finite_values(out);
        self.stats.push_finite_values(out);
    }
}

impl Finite for sidereon_core::orbit::DriftEntry {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.epoch.push_finite_values(out);
        out.push(self.error_m);
    }
}

impl Finite for sidereon_core::orbit::DriftReport {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.per_epoch.push_finite_values(out);
        out.extend([self.max_m, self.rms_m]);
        self.threshold_horizon.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::bodies::sun_moon::SunMoon {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.sun.push_finite_values(out);
        self.moon.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::events::CrossingEvent {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.time_seconds, self.value, self.threshold]);
    }
}

impl Finite for sidereon_core::astro::events::ExtremumEvent {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.time_seconds, self.value]);
    }
}

impl Finite for sidereon_core::astro::events::StateChangeEvent {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.time_seconds);
    }
}

impl Finite for sidereon_core::astro::passes::LookAngle {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.azimuth_deg, self.elevation_deg, self.range_km]);
    }
}

impl Finite for sidereon_core::astro::conjunction::EncounterFrame {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.x_hat.push_finite_values(out);
        self.y_hat.push_finite_values(out);
        self.z_hat.push_finite_values(out);
        self.relative_position_km.push_finite_values(out);
        self.relative_velocity_km_s.push_finite_values(out);
        out.extend([self.miss_km, self.relative_speed_km_s]);
    }
}

impl Finite for sidereon_core::astro::conjunction::CollisionPc {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.pc,
            self.miss_km,
            self.relative_speed_km_s,
            self.sigma_x_km,
            self.sigma_z_km,
        ]);
    }
}

impl Finite for sidereon_core::astro::tca::TcaCandidate {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.tca_time.push_finite_values(out);
        out.extend([self.tca_seconds_since_window_start, self.miss_distance_km]);
        self.relative_position_km.push_finite_values(out);
        self.relative_velocity_km_s.push_finite_values(out);
    }
}

impl Finite for sidereon_core::astro::tca::TcaConjunction {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.candidate.push_finite_values(out);
        self.collision_probability.push_finite_values(out);
    }
}

impl Finite for sidereon_core::positioning::Dop {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.gdop, self.pdop, self.hdop, self.vdop, self.tdop]);
    }
}

impl Finite for sidereon_core::positioning::LineOfSight {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.e_x, self.e_y, self.e_z]);
    }
}

impl Finite for sidereon_core::positioning::ReceiverSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.position.push_finite_values(out);
        self.geodetic.push_finite_values(out);
        out.push(self.rx_clock_s);
        for (_, clock) in &self.system_clocks_s {
            out.push(*clock);
        }
        self.dop.push_finite_values(out);
        out.extend(self.residuals_m.iter().copied());
    }
}

impl Finite for sidereon_core::precise_positioning::FloatResidual {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.code_m,
            self.phase_m,
            self.code_weight,
            self.phase_weight,
        ]);
    }
}

impl Finite for sidereon_core::precise_positioning::FloatSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.position_m.push_finite_values(out);
        out.extend(self.epoch_clocks_m.iter().copied());
        out.extend(self.ambiguities_m.values().copied());
        self.ztd_residual_m.push_finite_values(out);
        self.residuals_m.push_finite_values(out);
        out.extend([self.code_rms_m, self.phase_rms_m, self.weighted_rms_m]);
    }
}

impl Finite for sidereon_core::precise_positioning::FixedSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.position_m.push_finite_values(out);
        out.extend(self.epoch_clocks_m.iter().copied());
        out.extend(self.fixed_ambiguities_m.values().copied());
        self.ztd_residual_m.push_finite_values(out);
        self.float_solution.push_finite_values(out);
        self.residuals_m.push_finite_values(out);
        out.extend([
            self.code_rms_m,
            self.phase_rms_m,
            self.weighted_rms_m,
            self.integer.integer_ratio,
            self.integer.integer_best_score,
        ]);
        self.integer
            .integer_second_best_score
            .push_finite_values(out);
    }
}

impl Finite for sidereon_core::precise_positioning::ProtectionLevels {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.hpl_m, self.vpl_m]);
    }
}

impl Finite for sidereon_core::precise_positioning::RaimResult {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.test_statistic);
        self.threshold.push_finite_values(out);
        self.hpl_m.push_finite_values(out);
        self.vpl_m.push_finite_values(out);
        for stat in &self.satellite_statistics {
            out.extend([stat.code, stat.phase, stat.statistic]);
        }
    }
}

impl Finite for sidereon_core::precise_positioning::RaimFdeResult {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.solution.push_finite_values(out);
        self.raim.push_finite_values(out);
    }
}

impl Finite for sidereon_core::precise_positioning::RaimIdentification {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        for stat in &self.statistics {
            out.extend([stat.code, stat.phase, stat.statistic]);
        }
    }
}

impl Finite for sidereon_core::precise_positioning::KinematicEpochSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.position_m.push_finite_values(out);
        out.extend([self.clock_m, self.ztd_residual_m, self.innovation_rms_m]);
        out.extend(self.ambiguities_m.values().copied());
        self.position_covariance_m2.push_finite_values(out);
    }
}

impl Finite for sidereon_core::precise_positioning::KinematicUpdateSummary {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.innovation_rms_m);
    }
}

impl Finite for sidereon_core::precise_positioning::VelocitySolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.velocity_m_s.push_finite_values(out);
        out.extend([self.clock_drift_m_s, self.residual_rms_m_s]);
        self.robust_scale_m_s.push_finite_values(out);
    }
}

impl Finite for sidereon_core::precise_positioning::RangeRatePrediction {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.los_unit.push_finite_values(out);
        out.push(self.range_rate_m_s);
    }
}

impl Finite for sidereon_core::rtk_filter::FloatResidual {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.code_m,
            self.phase_m,
            self.code_sigma_m,
            self.phase_sigma_m,
            self.code_normalized,
            self.phase_normalized,
        ]);
    }
}

impl Finite for sidereon_core::rtk_filter::FloatBaselineSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.baseline_m.push_finite_values(out);
        out.extend(self.ambiguities_m.iter().map(|(_, v)| *v));
        out.extend(self.ambiguity_covariance_m.iter().copied());
        out.extend(self.ambiguity_covariance_inverse_m.iter().copied());
        self.residuals.push_finite_values(out);
        out.extend([self.code_rms_m, self.phase_rms_m, self.weighted_rms_m]);
    }
}

impl Finite for sidereon_core::rtk_filter::FixedBaselineSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.baseline_m.push_finite_values(out);
        out.extend(self.free_ambiguities_m.iter().map(|(_, v)| *v));
        out.extend(self.fixed_ambiguities_m.iter().map(|(_, v)| *v));
        self.residuals.push_finite_values(out);
        out.extend([self.code_rms_m, self.phase_rms_m, self.weighted_rms_m]);
    }
}

impl Finite for sidereon_core::rtk_filter::EpochUpdate {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.state.baseline_m.push_finite_values(out);
        out.extend(self.state.sd_ambiguities_m.iter().copied());
        out.extend(self.state.information.iter().copied());
        self.reported_baseline_m.push_finite_values(out);
        self.reported_sd_ambiguities_m.push_finite_values(out);
    }
}

impl Finite for sidereon_core::dgnss::AppliedCorrections {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend(self.corrected.iter().map(|obs| obs.pseudorange_m));
    }
}

impl Finite for sidereon_core::dgnss::PositionSolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.solution.push_finite_values(out);
        self.baseline_vector_m.push_finite_values(out);
        out.push(self.baseline_m);
    }
}

impl Finite for sidereon_core::ils::IlsResult {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.ratio, self.best_score]);
        self.second_best_score.push_finite_values(out);
        self.covariance.push_finite_values(out);
        self.covariance_inverse.push_finite_values(out);
    }
}

impl Finite for sidereon_core::atmosphere::MappingFactors {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.dry, self.wet]);
    }
}

impl Finite for sidereon_core::atmosphere::ZenithDelay {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.dry_m, self.wet_m]);
    }
}

impl Finite for sidereon_core::ephemeris::EccentricAnomaly {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.push(self.value);
    }
}

impl Finite for sidereon_core::ephemeris::OrbitState {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.a,
            self.n0,
            self.n,
            self.tk,
            self.mk,
            self.eccentric_anomaly,
            self.sin_e,
            self.cos_e,
            self.nu,
            self.phi,
            self.s2,
            self.c2,
            self.du,
            self.dr,
            self.di,
            self.u,
            self.r,
            self.i,
            self.xp,
            self.yp,
            self.omega_k,
            self.x_m,
            self.y_m,
            self.z_m,
        ]);
    }
}

impl Finite for sidereon_core::ephemeris::ClockOffset {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.dt_clock_poly_s,
            self.dt_rel_s,
            self.tgd_s,
            self.dt_clock_total_s,
        ]);
    }
}

impl Finite for sidereon_core::ephemeris::SatelliteState {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.orbit.push_finite_values(out);
        self.clock.push_finite_values(out);
    }
}

impl Finite for sidereon_core::signal::CorrelationResult {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([self.i, self.q, self.power]);
    }
}

impl Finite for sidereon_core::signal::AcquisitionResult {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.code_phase_chips,
            self.doppler_hz,
            self.peak_metric,
            self.metric,
            self.peak_power,
        ]);
    }
}

impl Finite for sidereon_core::observables::PredictedObservables {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        out.extend([
            self.geometric_range_m,
            self.range_rate_m_s,
            self.doppler_hz,
            self.elevation_deg,
            self.azimuth_deg,
            self.transmit_time_j2000_s,
        ]);
        self.sat_clock_s.push_finite_values(out);
        self.los_unit.push_finite_values(out);
        self.sat_pos_ecef_m.push_finite_values(out);
        self.sat_velocity_m_s.push_finite_values(out);
    }
}

impl Finite for sidereon_core::velocity::VelocitySolution {
    fn push_finite_values(&self, out: &mut Vec<f64>) {
        self.velocity_m_s.push_finite_values(out);
        out.extend([self.speed_m_s, self.clock_drift_s_s]);
        out.extend(self.residuals_m_s.iter().map(|(_, residual)| *residual));
    }
}
