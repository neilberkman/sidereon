//! # sidereon-core
//!
//! The complete Sidereon engine in one crate. It folds the numerical
//! astrodynamics core (orbit propagation, force models, frames, time, SGP4)
//! together with the GNSS domain layer (SP3, broadcast ephemeris, multi-GNSS
//! positioning, RTK/PPP, ionosphere/troposphere, DOP).
//!
//! - The propagation/astro layer is always present under the [`astro`] module.
//! - The GNSS layer lives behind the default-on `gnss` cargo feature, so a
//!   propagation-only consumer can build with `--no-default-features` (plus
//!   any astro features it wants) and never compile the IONEX/SP3 parsers.
//!
//! The GNSS façade is organized by user-facing tasks:
//!
//! - [`ephemeris`] - precise SP3 and broadcast ephemeris products,
//! - [`rinex`] - RINEX navigation/observation parsing and CRINEX decoding,
//! - [`antex`] - ANTEX receiver and satellite antenna calibration parsing,
//! - [`combinations`] - observable linear combinations such as ionosphere-free,
//! - [`observables`] - forward range, Doppler, and azimuth/elevation prediction,
//! - [`velocity`] - receiver velocity and clock-drift solve from range-rate data,
//! - [`positioning`] - single-point positioning and DOP diagnostics,
//! - [`dgnss`] - code-differential pseudorange correction and rover pairing,
//! - [`quality`] - pseudorange weighting, RAIM, and FDE integrity checks,
//! - [`observation_qc`] - RINEX observation completeness and signal rollups,
//! - [`signal`] - GPS C/A code generation, correlation, and acquisition,
//! - [`ppp_corrections`] - static-arc PPP correction precomputation,
//! - [`atmosphere`] - ionosphere and troposphere corrections,
//! - [`scenario`] - deterministic synthetic GNSS observation scenarios,
//! - [`orbit`] - compact reduced-orbit fitting/evaluation.
//!
//! Implementation modules (`sp3`, `rinex_nav`, `spp`, etc.) are crate-private.
//! This is a clean public surface rather than a compatibility shim around the
//! original implementation-shaped module layout.
//!
//! ## Units policy (internal representation)
//!
//! All quantities are stored and computed in **SI base units**, with the frame
//! and datum encoded in the type name (per the spec's frames-in-the-type-system
//! rule), never hidden behind a bare `position_m`:
//!
//! - **Length / position:** meters (`_m`). SP3 positions are ITRF/IGS-frame
//!   ECEF meters; SPP receiver positions are WGS84/ITRF-compatible ECEF meters.
//!   (The [`astro`] state layer works in kilometers; conversions happen
//!   explicitly at the boundary, never implicitly.)
//! - **Time / clock:** seconds (`_s`). Epochs are represented by the [`astro`]
//!   time family (`Instant`/`TimeScale`), always scale-tagged; there is no bare
//!   ambiguous epoch.
//! - **Velocity:** meters per second (`_m_s`).
//! - **Angles:** radians (`_rad`) internally. Degrees appear only at I/O edges
//!   and are named `_deg`.
//! - **Frequency:** hertz (`_hz`).
//!
//! Field and parameter names carry the unit suffix so the unit is visible at
//! every call site. Matrix/vector linear algebra uses `nalgebra`
//! (`DMatrix`/`DVector`) per the spec.

extern crate self as sidereon_core;

// ---------------------------------------------------------------------------
// Astro / propagation layer. Always present. The GNSS layer below depends on
// it via `crate::astro::*`.
// ---------------------------------------------------------------------------

mod validate;

#[cfg(all(test, sidereon_repo_tests))]
mod test_parity;

pub mod astro;
pub(crate) mod format;

// ---------------------------------------------------------------------------
// GNSS domain layer. Behind the default-on `gnss` feature so a propagation-only
// consumer can opt out. Additional product modules are added as each lands.
// ---------------------------------------------------------------------------

mod ambiguity; // shared RTK/PPP cycle-slip policy + wide-lane/narrow-lane prep
mod antenna; // shared ANTEX PCV/PCO zenith/azimuth interpolation kernels
pub mod antex; // ANTEX receiver/satellite antenna parser + PCO/PCV lookup
pub mod araim; // advanced RAIM multi-hypothesis protection levels
pub mod bias; // Bias-SINEX and DCB bias products
mod broadcast; // broadcast-ephemeris (GPS LNAV / Galileo I/NAV) orbit + clock
pub mod broadcast_comparison; // broadcast-vs-precise (SISRE orbit/clock) accuracy
pub mod carrier_phase; // carrier-phase combinations, cycle-slip detection, Hatch smoothing
pub mod clock_stability; // Allan-family receiver clock stability estimators
pub mod constants; // shared physical/time constants (used by astro + gnss)
pub mod constellation; // GNSS constellation identity catalog (CelesTrak/NAVCEN)
mod crinex; // Hatanaka (CRINEX) observation-file decoder
pub mod data; // sans-IO GNSS product filename and archive URL catalog
pub mod dop; // dilution-of-precision geometry (GDOP/PDOP/HDOP/VDOP/TDOP)
pub mod error_metrics; // covariance-derived CEP, radial, and ellipse metrics
pub mod exact_cache; // exact-product cache identity binding and atomic publication
pub mod frequencies; // canonical GNSS carrier-frequency table
mod glonass; // GLONASS PZ-90.11 state-vector RK4 propagation
pub mod has; // Galileo HAS MT1 correction payload decode/encode
mod ionex; // Klobuchar broadcast model + IONEX ionospheric maps
pub mod navigation; // navigation-message bit-level codecs (GPS LNAV)
pub mod nmea; // NMEA 0183 sentence parsing, stream grouping, and GGA writing
pub mod ntrip; // NTRIP client sans-I/O request, response, and stream handling
pub mod observables; // forward GNSS observable prediction
pub mod ppp_corrections; // static-arc PPP correction tables
pub mod precise_positioning; // static multi-epoch PPP float solve
mod reduced_orbit; // compact mean-element orbit approximation (fitted)
mod rinex_clock; // RINEX clock satellite-bias parsing and interpolation
mod rinex_common; // shared RINEX header concepts (time-system label mapping)
mod rinex_nav; // RINEX 3 navigation-message parsing (GPS/Galileo broadcast)
mod rinex_obs; // RINEX 3 observation parsing + single-frequency pseudoranges
mod rinex_qc; // RINEX observation/navigation lint and mechanical repair
pub mod rtcm; // RTCM 3 differential-GNSS stream decode/encode (MSM, station, ephemeris)
pub mod rtk; // RTK double-difference construction
pub mod sbas;
pub mod sbas_pl; // SBAS single-hypothesis protection levels
pub mod scenario; // scenario-driven synthetic GNSS observable generation
pub mod sidereal; // repeating-geometry residual filtering and period diagnostics
pub mod signal; // GPS C/A code, coherent correlation, and acquisition
pub mod source_localization; // ToA/TDOA source localization from arrival times
mod sp3; // SP3-c / SP3-d parser + arbitrary-epoch interpolation
mod spp; // single-point positioning (least-squares PVT)
pub mod ssr; // SSR correction store and corrected broadcast ephemeris source
pub mod staleness; // product-staleness graceful degradation for time-varying products
pub mod static_positioning; // multi-epoch static position fusion
mod static_reference_station; // RINEX reference-station static solve
mod tropo; // Saastamoinen zenith + Niell (NMF) mapping troposphere
pub mod velocity; // receiver velocity / clock-drift least-squares solve

mod error;
pub mod frame;
pub mod frame_catalog;
mod id;

pub mod atmosphere;
pub mod combinations;
pub mod dgnss;
pub mod ephemeris;
pub mod estimation; // Phase-2 estimation substrate: named operation-order recipes
pub mod geodesic; // WGS84 geodesic direct and inverse solvers
pub mod geodetic_time_series; // robust station velocity, trajectory, steps, and fields
pub mod geofence; // geodesic geofence containment and uncertainty gates
pub mod geoid; // geoid undulation grid + bilinear interpolation (orthometric heights)
pub mod geometry;
pub mod geometry_quality;
pub mod ils; // integer least squares ambiguity-resolution kernels
pub mod inertial; // ECEF strapdown INS frames, mechanization, and IMU error model
pub mod integrity; // shared protection and covariance primitives
pub mod observation_qc; // RINEX observation completeness and signal rollups
pub mod qc_obs {
    //! RINEX observation quality-control rollups.
    pub use crate::observation_qc::*;
}
pub mod orbit;
pub mod orbit_determination;
pub mod positioning;
pub mod prelude;
pub mod quality; // measurement weighting, RAIM, and FDE integrity checks
pub mod rinex;
pub mod rtk_filter; // sequential RTK baseline filter - serializable state ABI (kernel migration)
pub mod terrain;
pub mod terrain_store;
pub mod tides;
pub mod tolerances;

pub mod fusion; // GNSS/INS error-state prediction and EKF correction over the inertial surface

pub use crate::astro::frames::{
    EarthOrientation, EarthOrientationProvider, PolarMotionSample,
    PolarMotionSeriesEarthOrientationProvider, TdbEarthOrientationProvider,
};
pub use crate::error_metrics::{
    error_ellipse_from_enu_m2, horizontal_radius_at, metrics_from_ecef_covariance_m2,
    metrics_from_enu_covariance_m2, metrics_from_kinematic_solution,
    metrics_from_position_covariance, spherical_radius_at, vertical_radius_at, ErrorEllipse,
    ErrorMetricsError, PercentileRadius, PositionErrorMetrics,
};
pub use crate::estimation::{
    alpha_beta_apply_measurement, alpha_beta_filter_step, alpha_beta_predict,
    alpha_beta_steady_state_gains, cfar_ca_false_alarm_probability, cfar_ca_multiplier_from_pfa,
    cfar_ca_pfa_from_multiplier, cfar_ca_threshold, ewma_update, ewma_update_power_of_two,
    kalman_cv_steady_state_gains, mad_spread, nis_expected_value, nis_gate_test,
    nis_gate_threshold, nis_statistic, normalized_innovation, rts_smooth, smooth_track_rts,
    AlphaBetaGains, AlphaBetaState, AlphaBetaStep, PrimitiveError, ScalarKalmanGains,
    SmoothedTrack, SmoothedTrackEpoch, TrackCoordinateFrame, TrackError, TrackFilter,
    TrackFilterConfig, TrackGatedUpdate, TrackInnovation, TrackPrediction, TrackRtsEpoch,
    TrackRtsHistory, TrackRtsHistoryBuilder, TrackState, TrackUpdate, MAD_GAUSSIAN_CONSISTENCY,
};
pub use crate::quality::{
    reliability_araim, reliability_design, wtest_noncentrality, wtest_noncentrality_components,
    ObservationReliability, RangeReliabilityRow, ReliabilityOptions, ReliabilityReport,
    ReliabilitySummary, WtestNoncentralityComponents,
};
pub use araim::ProtectionModel;
pub use error::{Error, Result};
pub use frame::{
    geodetic_to_itrf, itrf_to_geodetic, FrameValueError, ItrfPositionM, ItrfVelocityMS,
    Wgs84Geodetic,
};
pub use frame_catalog::{
    catalog, catalog_entry, propagate_position, transform, transform_from_epoch, FrameCatalogError,
    HelmertParameters, HelmertRates, HelmertTransform, TerrestrialFrame, TerrestrialPositionM,
    TerrestrialState, TerrestrialVelocityMPerYear, TERRESTRIAL_FRAME_CATALOG,
};
pub use fusion::{
    loose_coupling_correction, smooth_fusion_rts, ukf_correct_closed_loop,
    validate_time_sync_gnss_order, validate_time_sync_imu_order, velocity_match_outage,
    velocity_match_outage_to_state, F64Bits, FusionFilterKind, FusionRtsEpoch, FusionRtsHistory,
    FusionRtsHistoryBuilder, FusionStateCodecError, FusionUpdate, GnssFixMeasurement,
    GnssFixStatus, GnssFixStatusWeighting, IggIiiMeasurementReweighting, InertialFilter,
    InertialFilterConfig, LooseCouplingConfig, NonHolonomicConstraintConfig,
    SerializableErrorStateLayout, SerializableFusionSnapshot, SerializableFusionState,
    SerializableGnssFixStatus, SerializableImuSample, SerializableImuSampleKind,
    SerializableInsFilterState, SerializableLooseMeasurement, SerializableNavState,
    SerializableRateEndpoint, SerializableSatelliteId, SerializableStationarityDetectorSample,
    SerializableStoredCheckpoint, SerializableStoredGnssMeasurement, SerializableStoredImuSample,
    SerializableTightCarrierPhaseObservation, SerializableTightFilterState,
    SerializableTightGnssEpoch, SerializableTightGnssObservation,
    SerializableTightRangeRateObservation, SerializableTimeSyncHistory,
    SerializableTimeSyncHistoryConfig, SmoothedFusionEpoch, SmoothedFusionTrajectory,
    StationarityDetectorSnapshotSample, StationaryDetectorConfig, StationaryUpdateConfig,
    TimeSyncHistoryConfig, TimeSyncHistoryStatus, TimeSyncUpdate, UkfUpdateOptions,
    UnscentedTransformOptions, VelocityMatchState, VelocityMatchedTrajectory,
    VelocityMatchingConfig, YangPredictionAdaptiveFactor, DEFAULT_TIME_SYNC_CHECKPOINT_CAPACITY,
    DEFAULT_TIME_SYNC_IMU_CAPACITY, FUSION_STATE_CODEC_VERSION,
};
pub use geodesic::{geodesic_direct, geodesic_inverse, GeodesicError};
pub use geofence::{
    containment, containment_probability, containment_probability_with_options, crossing,
    crossing_probability, crossing_probability_with_options, distance_to_boundary, CrossingEvent,
    CrossingKind, Fence, GeofenceError, GeofencePositionEstimate, PositionUncertainty,
    ProbabilityHysteresis, ProbabilityMethod, ProbabilityOptions, GEOFENCE_BOUNDARY_TOLERANCE_M,
    PLANAR_FAST_PATH_MAX_RADIUS_M,
};
pub use geoid::{
    egm96_undulations_deg, egm96_undulations_rad, ellipsoidal_height_m, geoid_undulation,
    geoid_undulations_deg, geoid_undulations_rad, orthometric_height_m, Egm2008GridSpacing,
    Egm2008RasterWindow, GeoidError, GeoidGrid, ProjVgridshiftArithmetic, ProjVgridshiftError,
};
pub use id::{GnssSatelliteId, GnssSystem, SatelliteIdError};
pub use inertial::{
    gauss_markov_bias_decay, gauss_markov_bias_variance_increment, gravity_ecef_mps2,
    mechanize_ecef, normal_gravity_mps2, rodrigues_delta_dcm, simulate_imu_samples,
    simulate_imu_samples_from_increments, true_imu_increment_between, AttitudeQuaternion,
    ConingCorrection, CorrectedImuIncrement, ImuBias, ImuCalibration, ImuErrorModel, ImuGrade,
    ImuRateRandomWalk, ImuSample, ImuSampleKind, ImuSimulationOptions, ImuSimulationOutput,
    ImuSimulator, ImuSpec, InertialError, MechanizationConfig, NavState, SimulatedImuSequence,
    StrapdownMechanizer, DEFAULT_IMU_SIM_SEED, WGS84_NORMAL_GRAVITY_EQUATOR_MPS2,
    WGS84_NORMAL_GRAVITY_POLE_MPS2, WGS84_SOMIGLIANA_K,
};
pub use observables::{
    emission_media_batch_at_j2000_s, observable_media_corrections, predict_batch_with_media,
    predict_batch_with_media_parallel, predict_ranges_with_media, predict_with_media,
    AppliedMediaCorrections, EmissionMediaBatch, EmissionMediaBatchOptions, EmissionMediaStatus,
    MediaPredictOptions, MediaPredictedObservables, MediaRangePrediction,
    ObservableIonosphereCorrection, ObservableMediaOptions, ObservableTroposphereCorrection,
};
pub use positioning::{RejectedSat, RejectionReason};
pub use sbas_pl::{
    sbas_protection_levels, AirborneModel, DegradationParams, ProtectionGeometry, ProtectionRow,
    SbasErrorModel, SbasKMultipliers, SbasPlError, SbasProtection, SbasSisError,
};
pub use sidereal::{
    orbit_repeat_lag, periodicity_strength, periodicity_strength_with_sample_interval,
    repeat_period, sidereal_filter, solar_day_period, SiderealFilterError, SiderealFilterOptions,
    SiderealFilterOutput, SiderealTemplateMethod, SIDEREAL_DAY_NANOS, SIDEREAL_DAY_SECONDS,
};
