//! # sidereon
//!
//! A thin, ergonomic API over the [`sidereon_core`] engine. It does not model
//! anything itself: every function here delegates to a `sidereon_core` reference
//! entry point and re-exports its result structs, so the numerical behavior is
//! identical to calling the core directly. The value it adds is a small, human
//! surface:
//!
//! - [`load_sp3`] parses a precise SP3 ephemeris product,
//! - [`parse_antex`] / [`load_antex`] parse ANTEX antenna calibration products,
//! - [`parse_rinex_nav`] / [`load_rinex_nav`] parse RINEX broadcast navigation
//!   products into a queryable broadcast ephemeris store,
//! - [`parse_rinex_obs`] / [`load_rinex_obs`] parse RINEX observation products,
//! - [`parse_rinex_clock`] / [`load_rinex_clock`] parse RINEX clock products,
//!   with lossy variants for best-effort recovery,
//! - [`decode_crinex`] / [`load_crinex`] expand Hatanaka-compressed
//!   observation files,
//! - [`solve_spp`] runs single-point positioning,
//! - [`solve_velocity`] solves receiver ECEF velocity and clock drift from
//!   range-rate or Doppler observations,
//! - [`solve_rtk_float_with`] / [`solve_rtk_fixed_with`] solve static RTK
//!   baselines from typed configs,
//! - [`solve_ppp_float_with`] / [`solve_ppp_fixed_with`] solve static PPP arcs
//!   from typed configs,
//! - GNSS utility modules such as [`frequencies`], [`combinations`],
//!   [`quality`], [`carrier_phase`], [`signal`], [`velocity`],
//!   [`broadcast_comparison`], [`constants`], [`navigation`], [`geometry`], and
//!   [`dgnss`] expose the core helper surface,
//! - [`astro`] exposes time/frame conversions, Sun/Moon positions, RF link
//!   budgets, eclipse events, conjunction/covariance utilities, CDM/OMM
//!   parsing, TCA screening, and orbit propagation,
//! - [`tle`], [`sgp4`], [`passes`], and [`tca`] remain root-level shortcuts for
//!   SGP4/TLE propagation, topocentric az/el/range over a ground station, and
//!   close-approach screening,
//! - one [`Error`] enum unifies product parsing/loading and every solve failure.
//!
//! The input, result, and ephemeris types each function takes and returns are
//! re-exported from `sidereon_core` under [`antex`], [`rinex`], [`astro`],
//! [`ephemeris`], [`positioning`], [`observables`], and the curated
//! [`rtk_filter`] / [`precise_positioning`] facades, so a consumer imports the
//! ergonomic API types from this one crate. Lower-level RTK/PPP internals remain
//! available under [`raw`] with their native core errors.
//!
//! ```
//! // Malformed input surfaces a single error type.
//! let parsed = sidereon::load_sp3(b"not a valid sp3 file");
//! assert!(matches!(parsed, Err(sidereon::Error::Sp3(_))));
//! ```
//!
//! Lower-level helpers re-exported through modules such as [`rinex`] keep their
//! native core error type. Map them explicitly at wrapper boundaries; there is
//! intentionally no blanket conversion into [`Error`].
//!
//! ```compile_fail
//! fn decode_from_reexported_core() -> sidereon::Result<()> {
//!     sidereon::rinex::decode_crinex("not a CRINEX file\n")?;
//!     Ok(())
//! }
//! ```
//!
//! Low-level RTK and PPP core modules live behind the explicit [`raw`] escape
//! hatch, not the ergonomic facades.
//!
//! ```
//! let _ = sidereon::raw::rtk_filter::RtkFilterScratch::new();
//! ```
//!
//! Hot-path RTK filter APIs are intentionally not part of the ergonomic
//! `sidereon::rtk_filter` facade.
//!
//! ```compile_fail
//! use sidereon::rtk_filter::{update_epoch_with_scratch, RtkFilterScratch};
//! ```
//!
//! PPP preparation and raw solver APIs are intentionally not part of the
//! ergonomic `sidereon::precise_positioning` facade.
//!
//! ```compile_fail
//! use sidereon::precise_positioning::{prepare_widelane_fixed_epochs, solve_float_epochs};
//! ```
//!
//! RTK config builders consume the config. Dropping the returned value must not
//! leave an unchanged copy available for solving.
//!
//! ```compile_fail
//! use sidereon::{
//!     rtk_filter::{FloatSolveOpts, MeasModel, StochasticModel},
//!     RtkFloatConfig,
//! };
//!
//! let epochs = Vec::new();
//! let ambiguity_ids = Vec::<String>::new();
//! let model = MeasModel {
//!     code_sigma_m: 0.3,
//!     phase_sigma_m: 0.003,
//!     sagnac: true,
//!     stochastic: StochasticModel::Rtklib,
//! };
//! let opts = FloatSolveOpts {
//!     position_tol_m: 1.0e-4,
//!     ambiguity_tol_m: 1.0e-4,
//!     max_iterations: 1,
//! };
//! let config = RtkFloatConfig::new(&epochs, [0.0; 3], &ambiguity_ids, &model, opts);
//! config.with_initial_baseline_m([1.0, 0.0, 0.0]);
//! let _ = sidereon::solve_rtk_float_with(config);
//! ```
//!
//! ```compile_fail
//! use std::collections::BTreeMap;
//!
//! use sidereon::{
//!     rtk_filter::{
//!         AmbiguityScale, AmbiguitySet, FixedSolveOpts, FloatSolveOpts, MeasModel,
//!         ResidualValidationOpts, StochasticModel, ValidatedFixedSolveOpts,
//!     },
//!     RtkFixedConfig,
//! };
//!
//! let epochs = Vec::new();
//! let ambiguity_ids = Vec::<String>::new();
//! let ambiguity_satellites = BTreeMap::new();
//! let wavelengths_m = BTreeMap::new();
//! let offsets_m = BTreeMap::new();
//! let float_only_systems = Vec::new();
//! let ambiguity_set = AmbiguitySet {
//!     ids: &ambiguity_ids,
//!     satellites: &ambiguity_satellites,
//!     scale: AmbiguityScale {
//!         wavelengths_m: &wavelengths_m,
//!         offsets_m: &offsets_m,
//!     },
//!     float_only_systems: &float_only_systems,
//! };
//! let model = MeasModel {
//!     code_sigma_m: 0.3,
//!     phase_sigma_m: 0.003,
//!     sagnac: true,
//!     stochastic: StochasticModel::Rtklib,
//! };
//! let opts = ValidatedFixedSolveOpts {
//!     float: FloatSolveOpts {
//!         position_tol_m: 1.0e-4,
//!         ambiguity_tol_m: 1.0e-4,
//!         max_iterations: 1,
//!     },
//!     fixed: FixedSolveOpts {
//!         position_tol_m: 1.0e-4,
//!         ambiguity_tol_m: 1.0e-4,
//!         max_iterations: 1,
//!         ratio_threshold: 3.0,
//!         partial_ambiguity_resolution: false,
//!         partial_min_ambiguities: 4,
//!     },
//!     residual: ResidualValidationOpts {
//!         threshold_sigma: None,
//!         max_exclusions: 0,
//!     },
//! };
//! let config = RtkFixedConfig::new(&epochs, [0.0; 3], ambiguity_set, &model, opts);
//! config.with_initial_baseline_m([1.0, 0.0, 0.0]);
//! let _ = sidereon::solve_rtk_fixed_with(config);
//! ```

use core::fmt;
use std::path::Path;

// Re-export core domain modules whose public helpers are intentionally part of
// the ergonomic crate surface.
pub use sidereon_core::{
    antex, astro, atmosphere, broadcast_comparison, carrier_phase, combinations, constants, dgnss,
    ephemeris, frequencies, geometry, navigation, observables, orbit, positioning, quality, rinex,
    rtcm, signal, velocity,
};
pub use sidereon_core::{
    geodetic_to_itrf, itrf_to_geodetic, FrameValueError, GnssSatelliteId, GnssSystem,
    ItrfPositionM, ItrfVelocityMS, SatelliteIdError, Wgs84Geodetic,
};

/// Stable RTK input, result, option, status, and error types used by the
/// ergonomic RTK solve wrappers.
pub mod rtk_filter {
    pub use sidereon_core::rtk_filter::{
        fix_wide_lane_rtk_arc, prepare_ionosphere_free_rtk_arc, solve_moving_baseline,
        solve_moving_baseline_epoch, solve_rtk_arc, solve_static_rtk_arc,
        solve_wide_lane_fixed_rtk_arc, AmbiguityScale, AmbiguitySearch, AmbiguitySet,
        CycleSlipOptions, CycleSlipPolicy, CycleSlipSplitArc, Epoch, FixedBaselineSolution,
        FixedSolveError, FixedSolveOpts, FloatBaselineSolution, FloatResidual, FloatSolveError,
        FloatSolveOpts, FloatSolveStatus, FullSetIntegerSummary, InnovationScreen,
        IntegerSearchMeta, IntegerStatus, IonosphereFreeBaselineError, MeasModel,
        MovingBaselineEpoch, MovingBaselineEpochSolution, MovingBaselineError, MovingBaselineOpts,
        MovingBaselineSequenceError, MovingBaselineStatus, PartialSearchMeta,
        ReceiverAntennaCalibration, ReceiverAntennaCorrections, ReceiverAntennaError,
        ResidualComponentKind, ResidualValidationMeta, ResidualValidationOpts,
        ResidualValidationOutlier, RtkArcConfig, RtkArcEpoch, RtkArcEpochSolution, RtkArcError,
        RtkArcObservation, RtkArcPreprocessing, RtkArcSolution, RtkDualCycleSlipConfig,
        RtkDualFrequencyArcEpoch, RtkDualFrequencyObservation,
        RtkDualFrequencySatelliteObservation, RtkIonosphereFreeArcConfig,
        RtkIonosphereFreeArcError, RtkIonosphereFreeArcSolution, RtkStaticArcConfig,
        RtkStaticArcError, RtkStaticArcSolution, RtkWideLaneArcConfig, RtkWideLaneArcError,
        RtkWideLaneArcSolution, RtkWideLaneFixedArcConfig, RtkWideLaneFixedArcError,
        RtkWideLaneFixedArcIntegerMethod, RtkWideLaneFixedArcMetadata, RtkWideLaneFixedArcSolution,
        RtkWideLaneFixedArcSolveConfig, RtkWideLaneFixedSequentialArcSolution,
        RtkWideLaneFixedStaticArcSolution, SatMeas, StochasticModel,
        ValidatedFixedBaselineSolution, ValidatedFixedSolveError, ValidatedFixedSolveOpts,
        WideLaneError, WideLaneOptions,
    };
}

/// Geoid undulation lookup and orthometric-height conversion: the
/// [`sidereon_core::geoid`] surface re-exported on the ergonomic crate.
pub mod geoid {
    pub use sidereon_core::geoid::{
        egm96_ellipsoidal_height_m, egm96_grid, egm96_orthometric_height_m, egm96_undulation,
        ellipsoidal_height_m, geoid_undulation, orthometric_height_m, GeoidError, GeoidGrid,
    };
}

/// Stable PPP input, result, option, status, and error types used by the
/// ergonomic PPP solve wrappers.
pub mod precise_positioning {
    pub use sidereon_core::precise_positioning::{
        solve_ppp_auto_init_fixed, solve_ppp_auto_init_fixed_with_strategy,
        solve_ppp_auto_init_float, solve_ppp_auto_init_float_with_strategy, AmbiguitySearch,
        FixedAmbiguityOptions, FixedIntegerMetadata, FixedSolution, FixedSolveConfig,
        FixedSolveError, FloatEpoch, FloatObservation, FloatResidual, FloatSolution,
        FloatSolveConfig, FloatSolveError, FloatSolveOptions, FloatState, FloatStatus,
        IntegerStatus, MeasurementWeights, MissingCorrection, NoEphemerisReason, PcvSample,
        PppAutoInitError, PppAutoInitOptions, PppAutoInitStrategy, PppCorrectionLookup,
        PppInitialGuess, RangeCorrections, ReceiverAntennaFrequency, ReceiverAntennaOptions,
        SatelliteClockCorrections, TroposphereOptions,
    };
}

/// Explicit escape hatch to the lower-level core RTK/PPP modules.
///
/// Items here keep their native `sidereon_core` error types and are outside the
/// one-error ergonomic wrapper surface.
pub mod raw {
    pub use sidereon_core::{precise_positioning, rtk_filter};
}

// Root-level propagation shortcuts retained for compatibility. The complete
// astrodynamics module tree is available as `sidereon::astro`.
pub use sidereon_core::astro::{passes, propagator, sgp4, state, tca, tle};

use sidereon_core::antex::{Antex, AntexError};
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::observables::ObservableEphemerisSource;
use sidereon_core::positioning::{
    EphemerisSource, ReceiverSolution, SolveInputs, SolvePolicy, SolvePolicyError,
};
use sidereon_core::precise_positioning::{
    FixedSolution, FixedSolveConfig, FixedSolveError as PppFixedSolveError, FloatEpoch,
    FloatSolution, FloatSolveConfig, FloatSolveError as PppFloatSolveError, FloatState,
};
use sidereon_core::rinex::clock::{RinexClock, RinexClockError};
use sidereon_core::rinex::nav::NavParseError;
use sidereon_core::rinex::observations::ObservationFile;
use sidereon_core::rtk_filter::{
    AmbiguitySet, Epoch, FloatBaselineSolution, FloatSolveError as RtkFloatSolveError,
    FloatSolveOpts, MeasModel, ReceiverAntennaCorrections, ValidatedFixedBaselineSolution,
    ValidatedFixedSolveError, ValidatedFixedSolveOpts,
};
use sidereon_core::velocity::{
    VelocityError, VelocityObservation, VelocitySolution, VelocitySolveOptions,
};

/// The one error type for the ergonomic API.
///
/// Each variant wraps the error of the `sidereon_core` reference entry point the
/// corresponding function delegates to. The SP3 variant wraps the core's own
/// [`sidereon_core::Error`] (which carries a human-readable parse message); the
/// solve variants wrap their technique-specific error verbatim so no diagnostic
/// detail is lost.
#[derive(Debug)]
pub enum Error {
    /// [`load_sp3`] failed to parse the SP3 product.
    Sp3(sidereon_core::Error),
    /// [`parse_antex`] or [`load_antex`] failed to parse the ANTEX product.
    Antex(AntexError),
    /// [`parse_rinex_nav`] or [`load_rinex_nav`] failed to parse the NAV product.
    RinexNav(NavParseError),
    /// [`parse_rinex_obs`] or [`load_rinex_obs`] failed to parse the OBS product.
    RinexObs(sidereon_core::Error),
    /// [`parse_rinex_clock`] or [`load_rinex_clock`] failed to parse the clock product.
    RinexClock(RinexClockError),
    /// [`decode_crinex`] or [`load_crinex`] failed to decode the CRINEX product.
    Crinex(sidereon_core::Error),
    /// A product file could not be read.
    Io(std::io::Error),
    /// [`solve_spp`] failed.
    Spp(SolvePolicyError),
    /// [`solve_velocity`] failed.
    Velocity(VelocityError),
    /// [`solve_rtk_float`] failed.
    RtkFloat(RtkFloatSolveError),
    /// [`solve_rtk_fixed`] failed.
    RtkFixed(ValidatedFixedSolveError),
    /// [`solve_ppp_float`] failed.
    PppFloat(PppFloatSolveError),
    /// [`solve_ppp_fixed`] failed.
    PppFixed(PppFixedSolveError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Sp3(e) => write!(f, "SP3 parse failed: {e}"),
            Error::Antex(e) => write!(f, "ANTEX parse failed: {e}"),
            Error::RinexNav(e) => write!(f, "RINEX NAV parse failed: {e}"),
            Error::RinexObs(e) => write!(f, "RINEX OBS parse failed: {e}"),
            Error::RinexClock(e) => write!(f, "RINEX clock parse failed: {e}"),
            Error::Crinex(e) => write!(f, "CRINEX decode failed: {e}"),
            Error::Io(e) => write!(f, "product file read failed: {e}"),
            Error::Spp(e) => write!(f, "{e}"),
            Error::Velocity(e) => write!(f, "velocity solve failed: {e}"),
            Error::RtkFloat(e) => write!(f, "{e}"),
            Error::RtkFixed(e) => write!(f, "{e}"),
            Error::PppFloat(e) => write!(f, "{e}"),
            Error::PppFixed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Sp3(e) => Some(e),
            Error::Antex(e) => Some(e),
            Error::RinexNav(e) => Some(e),
            Error::RinexObs(e) => Some(e),
            Error::RinexClock(e) => Some(e),
            Error::Crinex(e) => Some(e),
            Error::Io(e) => Some(e),
            Error::Spp(e) => Some(e),
            Error::Velocity(e) => Some(e),
            Error::RtkFloat(e) => Some(e),
            Error::RtkFixed(e) => Some(e),
            Error::PppFloat(e) => Some(e),
            Error::PppFixed(e) => Some(e),
        }
    }
}

impl From<AntexError> for Error {
    fn from(e: AntexError) -> Self {
        Error::Antex(e)
    }
}

impl From<NavParseError> for Error {
    fn from(e: NavParseError) -> Self {
        Error::RinexNav(e)
    }
}

impl From<RinexClockError> for Error {
    fn from(e: RinexClockError) -> Self {
        Error::RinexClock(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<SolvePolicyError> for Error {
    fn from(e: SolvePolicyError) -> Self {
        Error::Spp(e)
    }
}

impl From<VelocityError> for Error {
    fn from(e: VelocityError) -> Self {
        Error::Velocity(e)
    }
}

impl From<RtkFloatSolveError> for Error {
    fn from(e: RtkFloatSolveError) -> Self {
        Error::RtkFloat(e)
    }
}

impl From<ValidatedFixedSolveError> for Error {
    fn from(e: ValidatedFixedSolveError) -> Self {
        Error::RtkFixed(e)
    }
}

impl From<PppFloatSolveError> for Error {
    fn from(e: PppFloatSolveError) -> Self {
        Error::PppFloat(e)
    }
}

impl From<PppFixedSolveError> for Error {
    fn from(e: PppFixedSolveError) -> Self {
        Error::PppFixed(e)
    }
}

/// Result alias for the ergonomic API.
pub type Result<T> = core::result::Result<T, Error>;

/// Typed input bundle for a static multi-epoch float RTK baseline solve.
///
/// Coordinates are Earth-fixed ECEF/ITRF metres. `base_ecef_m` is the known base
/// receiver position, and `initial_baseline_m` is the rover-minus-base baseline
/// seed `[dx, dy, dz]` in metres. Epoch observations are already-normalized core
/// RTK epochs: code and carrier-phase observables are metres, satellite
/// positions are ECEF metres, and each ambiguity id must align with the double
/// differences built from `epochs`.
#[derive(Clone)]
pub struct RtkFloatConfig<'a> {
    /// Normalized double-difference epochs for the static RTK arc.
    pub epochs: &'a [Epoch],
    /// Known base-station ECEF/ITRF position, metres.
    pub base_ecef_m: [f64; 3],
    /// Ordered float ambiguity state ids, one per non-reference ambiguity.
    pub ambiguity_ids: &'a [String],
    /// Rover-minus-base ECEF/ITRF baseline seed, metres.
    pub initial_baseline_m: [f64; 3],
    /// Code/phase sigmas, stochastic model, and Sagnac switch.
    pub model: &'a MeasModel,
    /// Iteration and convergence controls, in metres and iterations.
    pub options: FloatSolveOpts,
    /// Receiver antenna PCO/PCV corrections; `None` means no receiver correction.
    pub receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
}

impl<'a> RtkFloatConfig<'a> {
    /// Build a float RTK config with a zero rover-minus-base initial baseline
    /// and no receiver antenna correction.
    #[must_use]
    pub fn new(
        epochs: &'a [Epoch],
        base_ecef_m: [f64; 3],
        ambiguity_ids: &'a [String],
        model: &'a MeasModel,
        options: FloatSolveOpts,
    ) -> Self {
        Self {
            epochs,
            base_ecef_m,
            ambiguity_ids,
            initial_baseline_m: [0.0; 3],
            model,
            options,
            receiver_antenna_corrections: None,
        }
    }

    /// Set the rover-minus-base ECEF/ITRF baseline seed, metres.
    #[must_use = "this builder consumes and returns the updated RTK float config"]
    pub fn with_initial_baseline_m(mut self, initial_baseline_m: [f64; 3]) -> Self {
        self.initial_baseline_m = initial_baseline_m;
        self
    }

    /// Set receiver antenna PCO/PCV corrections; `None` leaves them disabled.
    #[must_use = "this builder consumes and returns the updated RTK float config"]
    pub fn with_receiver_antenna_corrections(
        mut self,
        receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
    ) -> Self {
        self.receiver_antenna_corrections = receiver_antenna_corrections;
        self
    }
}

/// Typed input bundle for a static residual-validated fixed RTK baseline solve.
///
/// Coordinates are Earth-fixed ECEF/ITRF metres. `base_ecef_m` is the known base
/// receiver position, and `initial_baseline_m` is the rover-minus-base baseline
/// seed `[dx, dy, dz]` in metres. Ambiguity wavelengths and offsets live in
/// [`AmbiguitySet::scale`] and are metres; fixed integer decisions are carrier
/// cycles internally and converted back to metres in the returned solution.
#[derive(Clone)]
pub struct RtkFixedConfig<'a> {
    /// Normalized double-difference epochs for the static RTK arc.
    pub epochs: &'a [Epoch],
    /// Known base-station ECEF/ITRF position, metres.
    pub base_ecef_m: [f64; 3],
    /// Ordered ambiguity ids, satellite mapping, wavelength/offset scale, and
    /// constellations to leave float.
    pub initial_ambiguities: AmbiguitySet<'a>,
    /// Rover-minus-base ECEF/ITRF baseline seed, metres.
    pub initial_baseline_m: [f64; 3],
    /// Code/phase sigmas, stochastic model, and Sagnac switch.
    pub model: &'a MeasModel,
    /// Float solve, integer search, and residual-validation controls.
    pub options: ValidatedFixedSolveOpts,
    /// Receiver antenna PCO/PCV corrections; `None` means no receiver correction.
    pub receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
}

impl<'a> RtkFixedConfig<'a> {
    /// Build a fixed RTK config with a zero rover-minus-base initial baseline
    /// and no receiver antenna correction.
    #[must_use]
    pub fn new(
        epochs: &'a [Epoch],
        base_ecef_m: [f64; 3],
        initial_ambiguities: AmbiguitySet<'a>,
        model: &'a MeasModel,
        options: ValidatedFixedSolveOpts,
    ) -> Self {
        Self {
            epochs,
            base_ecef_m,
            initial_ambiguities,
            initial_baseline_m: [0.0; 3],
            model,
            options,
            receiver_antenna_corrections: None,
        }
    }

    /// Set the rover-minus-base ECEF/ITRF baseline seed, metres.
    #[must_use = "this builder consumes and returns the updated RTK fixed config"]
    pub fn with_initial_baseline_m(mut self, initial_baseline_m: [f64; 3]) -> Self {
        self.initial_baseline_m = initial_baseline_m;
        self
    }

    /// Set receiver antenna PCO/PCV corrections; `None` leaves them disabled.
    #[must_use = "this builder consumes and returns the updated RTK fixed config"]
    pub fn with_receiver_antenna_corrections(
        mut self,
        receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
    ) -> Self {
        self.receiver_antenna_corrections = receiver_antenna_corrections;
        self
    }
}

/// Typed input bundle for a static multi-epoch float PPP solve.
///
/// The ephemeris source supplies satellite ECEF/ITRF positions in metres and
/// clocks in seconds. `epochs` contain ionosphere-free code and carrier-phase
/// observations in metres. `initial_state.position_m` is ECEF/ITRF metres;
/// receiver clocks, ambiguities, and zenith tropospheric delay are represented
/// in metres, matching the core PPP state vector.
#[derive(Clone)]
pub struct PppFloatConfig<'a> {
    /// Observable ephemeris source, commonly an SP3 precise product.
    pub source: &'a dyn ObservableEphemerisSource,
    /// Static PPP epochs, ordered in time.
    pub epochs: &'a [FloatEpoch],
    /// Initial receiver position, per-epoch clocks, ambiguities, and ZTD.
    pub initial_state: FloatState,
    /// Measurement weights, corrections, troposphere, and iteration controls.
    pub solve: FloatSolveConfig,
}

impl<'a> PppFloatConfig<'a> {
    /// Build a float PPP config from the complete static-arc input bundle.
    pub fn new(
        source: &'a dyn ObservableEphemerisSource,
        epochs: &'a [FloatEpoch],
        initial_state: FloatState,
        solve: FloatSolveConfig,
    ) -> Self {
        Self {
            source,
            epochs,
            initial_state,
            solve,
        }
    }
}

/// Typed input bundle for a static integer-fixed PPP solve.
///
/// The ephemeris source and epochs must describe the same static arc used for
/// the supplied float solution. Coordinates are ECEF/ITRF metres; ambiguity
/// wavelengths and offsets inside `solve.ambiguity` are metres, and fixed
/// ambiguity decisions are reported in both carrier cycles and metres.
#[derive(Clone)]
pub struct PppFixedConfig<'a> {
    /// Observable ephemeris source, commonly an SP3 precise product.
    pub source: &'a dyn ObservableEphemerisSource,
    /// Static PPP epochs, ordered in time.
    pub epochs: &'a [FloatEpoch],
    /// Float PPP solution used as the integer ambiguity-search prior.
    pub float_solution: FloatSolution,
    /// Measurement weights, corrections, troposphere, and integer controls.
    pub solve: FixedSolveConfig,
}

impl<'a> PppFixedConfig<'a> {
    /// Build a fixed PPP config from the complete static-arc input bundle.
    pub fn new(
        source: &'a dyn ObservableEphemerisSource,
        epochs: &'a [FloatEpoch],
        float_solution: FloatSolution,
        solve: FixedSolveConfig,
    ) -> Self {
        Self {
            source,
            epochs,
            float_solution,
            solve,
        }
    }
}

/// Parse an SP3-c or SP3-d byte buffer into a precise-ephemeris product.
///
/// `bytes` is the full, already-decompressed file content. Delegates to
/// [`Sp3::parse`]; malformed input is returned as [`Error::Sp3`].
///
/// ```
/// // The parser rejects malformed input with `Error::Sp3`.
/// assert!(sidereon::load_sp3(b"garbage").is_err());
/// ```
pub fn load_sp3(bytes: &[u8]) -> Result<Sp3> {
    Sp3::parse(bytes).map_err(Error::Sp3)
}

/// Parse ANTEX text into receiver and satellite antenna calibrations.
///
/// Values are exposed in the core product's SI units: PCO/PCV are metres, with
/// azimuth and zenith grids in degrees. Malformed input is returned as
/// [`Error::Antex`].
pub fn parse_antex(text: &str) -> Result<Antex> {
    Antex::parse(text).map_err(Error::Antex)
}

/// Read and parse an ANTEX antenna calibration file.
///
/// Delegates to [`parse_antex`] after reading UTF-8 text from `path`.
pub fn load_antex(path: impl AsRef<Path>) -> Result<Antex> {
    let text = std::fs::read_to_string(path)?;
    parse_antex(&text)
}

/// Parse a RINEX NAV file into a queryable broadcast ephemeris store.
///
/// The store applies the core's default navigation usability policy and
/// implements [`EphemerisSource`], so it can feed [`solve_spp`].
pub fn parse_rinex_nav(text: &str) -> Result<BroadcastEphemeris> {
    BroadcastEphemeris::from_nav(text).map_err(Error::RinexNav)
}

/// Read and parse a RINEX NAV file into a queryable broadcast ephemeris store.
pub fn load_rinex_nav(path: impl AsRef<Path>) -> Result<BroadcastEphemeris> {
    let text = std::fs::read_to_string(path)?;
    parse_rinex_nav(&text)
}

/// Parse RINEX OBS text into a typed observation product.
pub fn parse_rinex_obs(text: &str) -> Result<ObservationFile> {
    ObservationFile::parse(text).map_err(Error::RinexObs)
}

/// Read and parse a RINEX OBS file.
pub fn load_rinex_obs(path: impl AsRef<Path>) -> Result<ObservationFile> {
    let text = std::fs::read_to_string(path)?;
    parse_rinex_obs(&text)
}

/// Strictly parse RINEX clock text into satellite clock-bias series.
pub fn parse_rinex_clock(text: &str) -> Result<RinexClock> {
    RinexClock::parse(text).map_err(Error::RinexClock)
}

/// Read and strictly parse a RINEX clock file.
pub fn load_rinex_clock(path: impl AsRef<Path>) -> Result<RinexClock> {
    let text = std::fs::read_to_string(path)?;
    parse_rinex_clock(&text)
}

/// Parse RINEX clock text while skipping malformed and non-`AS` rows.
pub fn parse_rinex_clock_lossy(text: &str) -> RinexClock {
    RinexClock::parse_lossy(text)
}

/// Read and lossily parse a RINEX clock file.
pub fn load_rinex_clock_lossy(path: impl AsRef<Path>) -> Result<RinexClock> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse_rinex_clock_lossy(&text))
}

/// Decode Compact RINEX (Hatanaka) OBS text into plain RINEX OBS text.
pub fn decode_crinex(text: &str) -> Result<String> {
    rinex::decode_crinex(text).map_err(Error::Crinex)
}

/// Read and decode a Compact RINEX (Hatanaka) OBS file.
pub fn load_crinex(path: impl AsRef<Path>) -> Result<String> {
    let text = std::fs::read_to_string(path)?;
    decode_crinex(&text)
}

/// Run single-point positioning under the public validation/orchestration
/// policy.
///
/// `eph` supplies satellite ECEF/ITRF positions in metres and satellite clocks
/// in seconds. `inputs.observations` are pseudoranges in metres, and
/// `inputs.t_rx_j2000_s` is the receive epoch in seconds since J2000 in the
/// ephemeris time scale. The SPP state uses `[x, y, z, clock]` with position in
/// ECEF/ITRF metres and receiver clock bias represented as metres internally;
/// the returned [`ReceiverSolution`] also exposes the receiver clock in seconds.
/// `with_geodetic` controls whether WGS84 latitude/longitude/height are
/// populated, and `policy` carries validation, masking, and correction behavior.
///
/// Delegates to [`sidereon_core::positioning::solve_with_policy`] and returns
/// its [`ReceiverSolution`], mapping any failure to [`Error::Spp`].
pub fn solve_spp(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Result<ReceiverSolution> {
    sidereon_core::positioning::solve_with_policy(eph, inputs, with_geodetic, policy)
        .map_err(Error::Spp)
}

/// Solve a batch of independent SPP epochs against a shared ephemeris, serially.
///
/// Each element of `epochs` is one receive instant's [`SolveInputs`]; element
/// `i` of the result is [`solve_spp`] applied to `epochs[i]` with the shared
/// `eph`, `with_geodetic`, and `policy`. The serial reference the parallel
/// [`solve_spp_batch`] is proven bit-identical against.
pub fn solve_spp_batch_serial(
    eph: &dyn EphemerisSource,
    epochs: &[SolveInputs],
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Vec<Result<ReceiverSolution>> {
    sidereon_core::positioning::solve_spp_batch_serial(eph, epochs, with_geodetic, policy)
        .into_iter()
        .map(|r| r.map_err(Error::Spp))
        .collect()
}

/// Solve a batch of independent SPP epochs against a shared ephemeris, fanning
/// the per-epoch solves across a rayon thread pool.
///
/// Each epoch is solved by the same single-epoch kernel as [`solve_spp`] and the
/// indexed parallel collect preserves order, so element `i` is byte-for-byte
/// identical to element `i` of [`solve_spp_batch_serial`]: epochs share only the
/// immutable `eph`/`policy`, with no cross-epoch state. The work is
/// embarrassingly parallel, so throughput scales with cores while every value
/// stays bit-exact. `eph` must be [`Sync`] to be shared across the pool; the
/// language bindings call this inside their GIL/scheduler release so the whole
/// fleet of fixes computes with no interpreter lock held.
pub fn solve_spp_batch(
    eph: &(dyn EphemerisSource + Sync),
    epochs: &[SolveInputs],
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Vec<Result<ReceiverSolution>> {
    sidereon_core::positioning::solve_spp_batch_parallel(eph, epochs, with_geodetic, policy)
        .into_iter()
        .map(|r| r.map_err(Error::Spp))
        .collect()
}

/// Solve receiver ECEF velocity and clock drift from one epoch of range-rate or
/// Doppler observations.
///
/// `source` supplies satellite ECEF/ITRF state at `t_rx_j2000_s`, expressed in
/// seconds since J2000. `observations` are either range rates in metres per
/// second or Doppler values in hertz, depending on `options.observable`; Doppler
/// rows use each observation's carrier frequency in hertz. `receiver_ecef_m` is
/// the known receiver ECEF/ITRF position in metres. The returned
/// [`VelocitySolution`] reports ECEF velocity in metres per second and receiver
/// clock drift in seconds per second.
///
/// Delegates to [`sidereon_core::velocity::solve`], returning its
/// [`VelocitySolution`] and mapping any failure to [`Error::Velocity`].
pub fn solve_velocity(
    source: &dyn ObservableEphemerisSource,
    observations: &[VelocityObservation],
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: VelocitySolveOptions,
) -> Result<VelocitySolution> {
    sidereon_core::velocity::solve(source, observations, receiver_ecef_m, t_rx_j2000_s, options)
        .map_err(Error::Velocity)
}

/// Solve a static multi-epoch float RTK baseline.
///
/// Prefer this typed-config form for new Rust callers. It delegates to the
/// lower-level positional [`solve_rtk_float`] function, preserving the exact
/// core solver path and error mapping.
///
/// ```
/// use sidereon::{
///     rtk_filter::{FloatSolveOpts, MeasModel, StochasticModel},
///     RtkFloatConfig,
/// };
///
/// let epochs = Vec::new();
/// let ambiguity_ids = Vec::<String>::new();
/// let model = MeasModel {
///     code_sigma_m: 0.3,
///     phase_sigma_m: 0.003,
///     sagnac: true,
///     stochastic: StochasticModel::Rtklib,
/// };
/// let opts = FloatSolveOpts {
///     position_tol_m: 1.0e-4,
///     ambiguity_tol_m: 1.0e-4,
///     max_iterations: 1,
/// };
/// let config = RtkFloatConfig::new(&epochs, [0.0; 3], &ambiguity_ids, &model, opts);
///
/// assert!(matches!(
///     sidereon::solve_rtk_float_with(config),
///     Err(sidereon::Error::RtkFloat(_))
/// ));
/// ```
pub fn solve_rtk_float_with(config: RtkFloatConfig<'_>) -> Result<FloatBaselineSolution> {
    solve_rtk_float(
        config.epochs,
        config.base_ecef_m,
        config.ambiguity_ids,
        config.initial_baseline_m,
        config.model,
        config.options,
        config.receiver_antenna_corrections,
    )
}

/// Lower-level positional form for a static multi-epoch float RTK baseline.
///
/// New Rust callers should prefer [`solve_rtk_float_with`] with
/// [`RtkFloatConfig`] so units and frame semantics are attached to named fields.
/// `base` is the known base receiver ECEF/ITRF position in metres.
/// `initial_baseline_m` is the rover-minus-base ECEF/ITRF baseline seed in
/// metres. `epochs` contain normalized double-difference code and carrier-phase
/// rows in metres; `ambiguity_ids` names the float ambiguity state columns.
/// Receiver antenna corrections are applied only when `Some(_)`; `None` is the
/// explicit no-correction case.
///
/// This function is kept for existing bindings and delegates to
/// [`sidereon_core::rtk_filter::solve_float_baseline`], returning its
/// [`FloatBaselineSolution`] and mapping any failure to [`Error::RtkFloat`].
pub fn solve_rtk_float(
    epochs: &[Epoch],
    base: [f64; 3],
    ambiguity_ids: &[String],
    initial_baseline_m: [f64; 3],
    model: &MeasModel,
    opts: FloatSolveOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<FloatBaselineSolution> {
    sidereon_core::rtk_filter::solve_float_baseline(
        epochs,
        base,
        ambiguity_ids,
        initial_baseline_m,
        model,
        opts,
        receiver_antenna_corrections,
    )
    .map_err(Error::RtkFloat)
}

/// Solve a static fixed RTK baseline with residual validation/FDE.
///
/// Prefer this typed-config form for new Rust callers. It delegates to the
/// lower-level positional [`solve_rtk_fixed`] function, preserving the exact
/// core solver path and error mapping.
pub fn solve_rtk_fixed_with(config: RtkFixedConfig<'_>) -> Result<ValidatedFixedBaselineSolution> {
    solve_rtk_fixed(
        config.epochs,
        config.base_ecef_m,
        config.initial_ambiguities,
        config.initial_baseline_m,
        config.model,
        config.options,
        config.receiver_antenna_corrections,
    )
}

/// Lower-level positional form for a static fixed RTK baseline with residual
/// validation/FDE.
///
/// New Rust callers should prefer [`solve_rtk_fixed_with`] with
/// [`RtkFixedConfig`] so units and frame semantics are attached to named fields.
/// `base` is the known base receiver ECEF/ITRF position in metres.
/// `initial_baseline_m` is the rover-minus-base ECEF/ITRF baseline seed in
/// metres. `initial_ambiguities` supplies ambiguity ids, satellite mapping,
/// wavelengths, and offsets for the integer search; wavelengths and offsets are
/// in metres and fixed ambiguity decisions are reported in carrier cycles and
/// metres. Receiver antenna corrections are applied only when `Some(_)`;
/// `None` is the explicit no-correction case.
///
/// This function is kept for existing bindings and delegates to
/// [`sidereon_core::rtk_filter::solve_fixed_baseline_validated`], returning its
/// [`ValidatedFixedBaselineSolution`] and mapping any failure to
/// [`Error::RtkFixed`].
pub fn solve_rtk_fixed(
    epochs: &[Epoch],
    base: [f64; 3],
    initial_ambiguities: AmbiguitySet,
    initial_baseline_m: [f64; 3],
    model: &MeasModel,
    opts: ValidatedFixedSolveOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<ValidatedFixedBaselineSolution> {
    sidereon_core::rtk_filter::solve_fixed_baseline_validated(
        epochs,
        base,
        initial_ambiguities,
        initial_baseline_m,
        model,
        opts,
        receiver_antenna_corrections,
    )
    .map_err(Error::RtkFixed)
}

/// Solve a static multi-epoch float PPP arc.
///
/// Prefer this typed-config form for new Rust callers. It delegates to the
/// lower-level positional [`solve_ppp_float`] function, preserving the exact core
/// solver path and error mapping.
pub fn solve_ppp_float_with(config: PppFloatConfig<'_>) -> Result<FloatSolution> {
    solve_ppp_float(
        config.source,
        config.epochs,
        config.initial_state,
        config.solve,
    )
}

/// Lower-level positional form for a static multi-epoch float PPP arc.
///
/// New Rust callers should prefer [`solve_ppp_float_with`] with
/// [`PppFloatConfig`] so units and frame semantics are attached to named fields.
/// `source` supplies satellite ECEF/ITRF positions in metres and clocks in
/// seconds. `epochs` contain ionosphere-free code and carrier phase in metres.
/// `initial_state.position_m` is ECEF/ITRF metres; receiver clocks, carrier
/// ambiguities, and zenith tropospheric delay are represented in metres.
///
/// This function is kept for existing bindings and delegates to
/// [`sidereon_core::precise_positioning::solve_float_epochs`], returning its
/// [`FloatSolution`] and mapping any failure to [`Error::PppFloat`].
pub fn solve_ppp_float(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    initial_state: FloatState,
    config: FloatSolveConfig,
) -> Result<FloatSolution> {
    sidereon_core::precise_positioning::solve_float_epochs(source, epochs, initial_state, config)
        .map_err(Error::PppFloat)
}

/// Search integer ambiguities from a float PPP solution and re-solve with them
/// held fixed.
///
/// Prefer this typed-config form for new Rust callers. It delegates to the
/// lower-level positional [`solve_ppp_fixed`] function, preserving the exact core
/// solver path and error mapping.
pub fn solve_ppp_fixed_with(config: PppFixedConfig<'_>) -> Result<FixedSolution> {
    solve_ppp_fixed(
        config.source,
        config.epochs,
        config.float_solution,
        config.solve,
    )
}

/// Lower-level positional form for static integer-fixed PPP.
///
/// New Rust callers should prefer [`solve_ppp_fixed_with`] with
/// [`PppFixedConfig`] so units and frame semantics are attached to named fields.
/// `source` and `epochs` must describe the same static ECEF/ITRF arc used to
/// produce `float_solution`. Ambiguity wavelengths and offsets in `config` are
/// metres; integer decisions in the returned solution are reported in carrier
/// cycles and converted metres.
///
/// This function is kept for existing bindings and delegates to
/// [`sidereon_core::precise_positioning::solve_fixed_from_float`], returning its
/// [`FixedSolution`] and mapping any failure to [`Error::PppFixed`].
pub fn solve_ppp_fixed(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    float_solution: FloatSolution,
    config: FixedSolveConfig,
) -> Result<FixedSolution> {
    sidereon_core::precise_positioning::solve_fixed_from_float(
        source,
        epochs,
        float_solution,
        config,
    )
    .map_err(Error::PppFixed)
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use sidereon_core::positioning::{Corrections, KlobucharCoeffs, Observation, SurfaceMet};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    // A tiny real SP3 product: five GPS satellites at coincident positions over
    // two epochs. It parses cleanly but is geometrically degenerate, so any SPP
    // solve against it fails. Reused from the core parity fixtures.
    const DEGENERATE_SP3: &[u8] =
        include_bytes!("../../sidereon-core/tests/fixtures/sp3/degenerate_coincident_5sat.sp3");
    const ANTEX_TEXT: &str =
        include_str!("../../sidereon-core/tests/fixtures/antex/igs20_wettzell_trim.atx");
    const RINEX_NAV_TEXT: &str =
        include_str!("../../sidereon-core/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx");
    const RINEX_OBS_TEXT: &str = include_str!(
        "../../sidereon-core/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
    );
    const RINEX_CLOCK_TEXT: &str =
        include_str!("../../sidereon-core/tests/fixtures/clk/synthetic_rinex_clock.clk");
    const CRINEX_TEXT: &str = include_str!(
        "../../sidereon-core/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx"
    );
    const STATIONS_TLE: &str =
        include_str!("../../sidereon-core/tests/fixtures/celestrak/stations.tle");

    fn fixture_path(parts: &[&str]) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../sidereon-core/tests/fixtures");
        for part in parts {
            path.push(part);
        }
        path
    }

    fn station_tle(name: &str) -> tca::TcaTle<'static> {
        let mut lines = STATIONS_TLE.lines();
        while let Some(object_name) = lines.next() {
            let Some(line1) = lines.next() else {
                break;
            };
            let Some(line2) = lines.next() else {
                break;
            };
            if object_name.trim() == name {
                return tca::TcaTle::new(line1, line2);
            }
        }
        panic!("missing station TLE {name}");
    }

    fn norm3(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn rtk_model() -> MeasModel {
        MeasModel {
            code_sigma_m: 0.3,
            phase_sigma_m: 0.003,
            sagnac: true,
            stochastic: rtk_filter::StochasticModel::Rtklib,
        }
    }

    fn rtk_float_options() -> FloatSolveOpts {
        FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 1,
        }
    }

    fn rtk_fixed_options() -> rtk_filter::ValidatedFixedSolveOpts {
        rtk_filter::ValidatedFixedSolveOpts {
            float: rtk_float_options(),
            fixed: rtk_filter::FixedSolveOpts {
                position_tol_m: 1.0e-4,
                ambiguity_tol_m: 1.0e-4,
                max_iterations: 1,
                ratio_threshold: 3.0,
                partial_ambiguity_resolution: false,
                partial_min_ambiguities: 4,
            },
            residual: rtk_filter::ResidualValidationOpts {
                threshold_sigma: None,
                max_exclusions: 0,
            },
        }
    }

    fn ppp_float_solve_config() -> FloatSolveConfig {
        FloatSolveConfig {
            weights: precise_positioning::MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: precise_positioning::TroposphereOptions::disabled(),
            corrections: precise_positioning::RangeCorrections::disabled(),
            opts: precise_positioning::FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            residual_screen: false,
        }
    }

    fn ppp_fixed_solve_config() -> FixedSolveConfig {
        let float = ppp_float_solve_config();
        FixedSolveConfig {
            weights: float.weights,
            tropo: float.tropo,
            corrections: float.corrections,
            opts: float.opts,
            ambiguity: precise_positioning::FixedAmbiguityOptions {
                wavelengths_m: BTreeMap::new(),
                offsets_m: BTreeMap::new(),
                ratio_threshold: 3.0,
            },
        }
    }

    fn empty_ppp_state() -> FloatState {
        FloatState {
            position_m: [0.0; 3],
            clocks_m: Vec::new(),
            ambiguities_m: BTreeMap::new(),
            ztd_m: 0.0,
        }
    }

    fn empty_ppp_float_solution() -> FloatSolution {
        FloatSolution {
            position_m: [0.0; 3],
            epoch_clocks_m: Vec::new(),
            ambiguities_m: BTreeMap::new(),
            ztd_residual_m: None,
            residuals_m: Vec::new(),
            used_sats: Vec::new(),
            iterations: 0,
            converged: false,
            status: precise_positioning::FloatStatus::MaxIterations,
            code_rms_m: 0.0,
            phase_rms_m: 0.0,
            weighted_rms_m: 0.0,
        }
    }

    #[test]
    fn load_sp3_parses_a_precise_product() {
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        assert_eq!(sp3.epoch_count(), 2);
        assert_eq!(sp3.satellites().len(), 5);
    }

    #[test]
    fn load_sp3_surfaces_parse_errors() {
        let err = load_sp3(b"not an sp3 file").unwrap_err();
        assert!(matches!(err, Error::Sp3(_)));
        assert!(err.to_string().contains("SP3 parse failed"));
        // The core parse message is preserved as the error source.
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn product_ingestion_wrappers_parse_fixture_products() {
        let antex = parse_antex(ANTEX_TEXT).expect("parse ANTEX fixture");
        assert!(!antex.antennas.is_empty());
        let loaded_antex =
            load_antex(fixture_path(&["antex", "igs20_wettzell_trim.atx"])).expect("load ANTEX");
        assert_eq!(loaded_antex.antennas.len(), antex.antennas.len());

        let nav = parse_rinex_nav(RINEX_NAV_TEXT).expect("parse RINEX NAV fixture");
        assert!(!nav.records().is_empty() || !nav.glonass_records().is_empty());
        let loaded_nav =
            load_rinex_nav(fixture_path(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]))
                .expect("load RINEX NAV");
        assert_eq!(loaded_nav.records().len(), nav.records().len());
        assert_eq!(
            loaded_nav.glonass_records().len(),
            nav.glonass_records().len()
        );

        let obs = parse_rinex_obs(RINEX_OBS_TEXT).expect("parse RINEX OBS fixture");
        assert!(!obs.epochs().is_empty());
        let loaded_obs = load_rinex_obs(fixture_path(&[
            "obs",
            "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx",
        ]))
        .expect("load RINEX OBS");
        assert_eq!(loaded_obs.epochs().len(), obs.epochs().len());

        let clock = parse_rinex_clock(RINEX_CLOCK_TEXT).expect("parse RINEX clock fixture");
        assert!(!clock.series_rows().is_empty());
        let loaded_clock = load_rinex_clock(fixture_path(&["clk", "synthetic_rinex_clock.clk"]))
            .expect("load RINEX clock");
        assert_eq!(loaded_clock.series_rows().len(), clock.series_rows().len());
        let lossy_clock = parse_rinex_clock_lossy("AS malformed\n");
        assert!(lossy_clock.series_rows().is_empty());
        let loaded_lossy_clock =
            load_rinex_clock_lossy(fixture_path(&["clk", "synthetic_rinex_clock.clk"]))
                .expect("load lossy RINEX clock");
        assert_eq!(
            loaded_lossy_clock.series_rows().len(),
            clock.series_rows().len()
        );

        let decoded = decode_crinex(CRINEX_TEXT).expect("decode CRINEX fixture");
        assert!(decoded.contains("RINEX VERSION / TYPE"));
        let loaded_decoded = load_crinex(fixture_path(&[
            "obs",
            "ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx",
        ]))
        .expect("load CRINEX");
        assert_eq!(loaded_decoded, decoded);
    }

    #[test]
    fn product_ingestion_wrappers_map_errors() {
        let err = match parse_rinex_nav("not a RINEX NAV file") {
            Ok(_) => panic!("invalid RINEX NAV unexpectedly parsed"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::RinexNav(_)));
        assert!(std::error::Error::source(&err).is_some());

        let err = match parse_rinex_nav(
            "     4.00           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n\
             XXX                                                         END OF HEADER\n\
             > EPH G01 LNAV\n",
        ) {
            Ok(_) => panic!("empty v4 EPH frame unexpectedly parsed"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::RinexNav(_)));

        let err = parse_rinex_obs("not a RINEX OBS file").unwrap_err();
        assert!(matches!(err, Error::RinexObs(_)));
        assert!(std::error::Error::source(&err).is_some());

        let err = parse_rinex_clock("AS malformed").unwrap_err();
        assert!(matches!(err, Error::RinexClock(_)));
        assert!(std::error::Error::source(&err).is_some());

        let err = decode_crinex("not a CRINEX file\n").unwrap_err();
        assert!(matches!(err, Error::Crinex(_)));
        let rendered = err.to_string();
        assert!(rendered.starts_with("CRINEX decode failed:"), "{rendered}");
        assert!(!rendered.contains("SP3 parse failed"), "{rendered}");
        assert!(std::error::Error::source(&err).is_some());

        let missing = fixture_path(&["missing.nope"]);
        let err = match load_rinex_nav(missing) {
            Ok(_) => panic!("missing RINEX NAV path unexpectedly loaded"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Io(_)));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn astro_umbrella_reexports_broader_astrodynamics_surface() {
        let budget = astro::rf::LinkBudget {
            eirp_dbw: 0.0,
            fspl_db: 165.0,
            receiver_gt_dbk: -12.0,
            other_losses_db: 3.0,
            required_cn0_dbhz: 35.0,
        };
        assert_eq!(
            astro::rf::link_margin(&budget)
                .expect("valid RF link budget")
                .to_bits(),
            13.599999999999994_f64.to_bits()
        );

        let ts =
            astro::time::TimeScales::from_utc(2000, 1, 1, 12, 0, 0.0).expect("valid UTC instant");
        assert!((ts.jd_tt - 2451545.0).abs() < 1.0e-3);

        let sun = [149_597_870.7, 0.0, 0.0];
        assert_eq!(
            astro::events::eclipse::status([-7000.0, 0.0, 0.0], sun)
                .expect("valid eclipse geometry"),
            astro::events::eclipse::EclipseStatus::Umbra
        );
        assert!(astro::covariance::symmetric(&[
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0]
        ]));

        assert!(core::mem::size_of::<astro::bodies::SunMoon>() > 0);
        assert!(core::mem::size_of::<astro::cdm::CdmKvn>() > 0);
        assert!(core::mem::size_of::<astro::conjunction::ConjunctionState>() > 0);
        assert!(core::mem::size_of::<astro::frames::transforms::TemeStateKm>() > 0);
        assert!(core::mem::size_of::<astro::omm::Omm>() > 0);
    }

    #[test]
    fn ground_site_sun_moon_helpers_reachable_through_facade() {
        use astro::bodies::{
            find_moon_elevation_crossings, find_moon_transits, moon_az_el, moon_illumination,
            sun_az_el, MoonElevationOptions,
        };
        use astro::frames::transforms::{itrs_to_topocentric, GeodeticStationKm};
        use astro::passes::UtcInstant;

        let station = GeodeticStationKm {
            latitude_deg: 51.4769,
            longitude_deg: 0.0,
            altitude_km: 0.046,
        };
        // Solar upper transit at Greenwich on 2024-06-20 (Skyfield de421):
        // az 180.0 deg, alt 61.96 deg.
        let noon = UtcInstant::from_utc(2024, 6, 20, 12, 1, 42, 0).expect("valid UTC");
        let sun = sun_az_el(&station, noon).expect("sun geometry");
        assert!((sun.elevation_deg - 61.96).abs() < 0.5);

        // Full moon of 2024-04-23 23:49 UTC: nearly fully lit.
        let full = UtcInstant::from_utc(2024, 4, 23, 23, 49, 0, 0).expect("valid UTC");
        let illum = moon_illumination(&station, full).expect("moon illumination");
        assert!(illum.illuminated_fraction > 0.95);
        let moon = moon_az_el(&station, full).expect("moon geometry");
        assert!((350_000.0..410_000.0).contains(&moon.range_km));

        let start = UtcInstant::from_utc(2024, 4, 23, 0, 0, 0, 0).expect("valid UTC");
        let end = UtcInstant::from_utc(2024, 4, 24, 0, 0, 0, 0).expect("valid UTC");
        assert_eq!(
            find_moon_elevation_crossings(&station, start, end, MoonElevationOptions::default())
                .expect("moon crossings")
                .len(),
            2
        );
        assert_eq!(
            find_moon_transits(&station, start, end, 300.0, 1.0)
                .expect("moon transits")
                .len(),
            2
        );

        // The shared Earth-fixed topocentric primitive is exposed too.
        let (_az, el, _range) =
            itrs_to_topocentric([0.0, 0.0, 7000.0], &station).expect("topocentric");
        assert!(el.is_finite());
    }

    #[test]
    fn tca_shortcut_screens_two_real_tles_over_one_day() {
        let primary_tle = station_tle("ISS (ZARYA)");
        let secondary_tle = station_tle("CSS (TIANHE)");
        let primary = sgp4::Satellite::from_tle(primary_tle.line1, primary_tle.line2)
            .expect("station TLE parses");
        let window = tca::TcaWindow::from_start_and_duration_seconds(primary.epoch_jd(), 86_400.0)
            .expect("valid one-day window");
        let options = tca::TcaFinderOptions {
            coarse_step_seconds: 120.0,
            time_tolerance_seconds: 1.0e-2,
        };

        let candidates =
            tca::find_tca_candidates_between_tles(primary_tle, secondary_tle, window, options)
                .expect("real TLE TCA search succeeds");
        assert!(!candidates.is_empty());

        let best = candidates
            .iter()
            .min_by(|a, b| a.miss_distance_km.total_cmp(&b.miss_distance_km))
            .expect("candidate set is nonempty");
        assert!(best.tca_seconds_since_window_start > 0.0);
        assert!(best.tca_seconds_since_window_start < 86_400.0);
        assert!(best.miss_distance_km.is_finite());
        assert!(best.miss_distance_km > 0.0);
        assert!((norm3(best.relative_position_km) - best.miss_distance_km).abs() < 1.0e-9);
        assert!(norm3(best.relative_velocity_km_s) > 0.0);

        let secondaries = [secondary_tle];
        let threshold_km = best.miss_distance_km + 1.0;
        let serial = tca::screen_tca_candidates_from_tle_catalog_serial(
            primary_tle,
            &secondaries,
            window,
            threshold_km,
            options,
        )
        .expect("serial real TLE screening succeeds");
        let parallel = tca::screen_tca_candidates_from_tle_catalog_parallel(
            primary_tle,
            &secondaries,
            window,
            threshold_km,
            options,
        )
        .expect("parallel real TLE screening succeeds");

        assert_eq!(serial, parallel);
        assert!(!serial.is_empty());
        assert!(serial.iter().all(|hit| hit.secondary_index == 0));
        assert!(serial
            .iter()
            .all(|hit| hit.candidate.miss_distance_km <= threshold_km));

        let pc_options = tca::TcaPcOptions::with_default_covariance(
            0.020,
            astro::conjunction::PcMethod::Alfano2005,
        );
        let conjunctions = tca::find_tca_conjunctions_between_tles(
            primary_tle,
            secondary_tle,
            window,
            options,
            pc_options,
        )
        .expect("real TLE TCA Pc search succeeds");
        assert_eq!(conjunctions.len(), candidates.len());
        assert!(conjunctions.iter().all(|conjunction| {
            conjunction.collision_probability.pc.is_finite()
                && (0.0..=1.0).contains(&conjunction.collision_probability.pc)
        }));
    }

    #[test]
    fn gnss_utility_modules_are_reexported() {
        assert_eq!(
            frequencies::frequency_hz(GnssSystem::Gps, frequencies::CarrierBand::L1),
            Some(constants::F_L1_HZ)
        );
        assert!(combinations::gamma(constants::F_L1_HZ, constants::F_L2_HZ)
            .expect("GPS L1/L2 gamma")
            .is_finite());
        assert_eq!(
            carrier_phase::geometry_free(100.0, 60.0)
                .expect("finite geometry-free combination")
                .to_bits(),
            40.0_f64.to_bits()
        );
        assert!(quality::pseudorange_variance(
            30.0,
            quality::PseudorangeVarianceOptions::default()
        )
        .expect("positive elevation variance")
        .is_finite());
        assert_eq!(
            signal::ca_code(1).expect("GPS PRN 1").len(),
            signal::CA_CODE_LENGTH
        );
        assert_eq!(
            velocity::doppler_to_range_rate(-1.0, constants::F_L1_HZ)
                .expect("valid Doppler conversion")
                .to_bits(),
            (constants::C_M_S / constants::F_L1_HZ).to_bits()
        );
        assert_eq!(navigation::lnav::PREAMBLE, 0b1000_1011);
        assert_eq!(dgnss::CodeObservation::new("G01", 1.0).satellite_id, "G01");
        assert!(core::mem::size_of::<geometry::VisibilityOptions>() > 0);
        assert!(core::mem::size_of::<broadcast_comparison::EpochInputs>() > 0);
    }

    #[test]
    fn solve_spp_delegates_to_the_core_solver_and_maps_errors() {
        // Two observations against a four-parameter solve is under-determined,
        // so the real core solver returns an error; the wrapper maps it into
        // `Error::Spp` rather than leaking the technique-specific type.
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        let sat = |prn| GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id");
        let inputs = SolveInputs {
            observations: vec![
                Observation {
                    satellite_id: sat(1),
                    pseudorange_m: 2.1e7,
                },
                Observation {
                    satellite_id: sat(2),
                    pseudorange_m: 2.1e7,
                },
            ],
            t_rx_j2000_s: 646_315_200.0,
            t_rx_second_of_day_s: 0.0,
            day_of_year: 176.0,
            initial_guess: [0.0, 0.0, 0.0, 0.0],
            corrections: Corrections::NONE,
            klobuchar: KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
            beidou_klobuchar: None,
            galileo_nequick: None,
            glonass_channels: std::collections::BTreeMap::new(),
            met: SurfaceMet {
                pressure_hpa: 1013.25,
                temperature_k: 288.15,
                relative_humidity: 0.5,
            },
            robust: None,
        };

        let result = solve_spp(&sp3, &inputs, false, SolvePolicy::default());
        assert!(matches!(result, Err(Error::Spp(_))), "got {result:?}");
        if let Err(err) = result {
            assert!(err.to_string().contains("SPP solve failed"));
            assert!(std::error::Error::source(&err).is_some());
        }
    }

    #[test]
    fn solve_velocity_delegates_to_core_solver_and_maps_errors() {
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        let result = solve_velocity(
            &sp3,
            &[],
            [0.0; 3],
            646_315_200.0,
            VelocitySolveOptions::default(),
        );

        assert!(
            matches!(
                result,
                Err(Error::Velocity(velocity::VelocityError::NoObservations))
            ),
            "got {result:?}"
        );
        if let Err(err) = result {
            assert!(err.to_string().contains("velocity solve failed"));
            assert!(std::error::Error::source(&err).is_some());
        }
    }

    #[test]
    fn solve_rtk_float_positional_wrapper_maps_errors() {
        let epochs = Vec::new();
        let ambiguity_ids = Vec::<String>::new();
        let model = rtk_model();

        let err = solve_rtk_float(
            &epochs,
            [0.0; 3],
            &ambiguity_ids,
            [0.0; 3],
            &model,
            rtk_float_options(),
            None,
        )
        .unwrap_err();

        assert!(matches!(err, Error::RtkFloat(_)));
        assert!(err.to_string().contains("RTK float"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn solve_rtk_fixed_positional_wrapper_maps_errors() {
        let epochs = Vec::new();
        let ambiguity_ids = Vec::<String>::new();
        let ambiguity_satellites = BTreeMap::new();
        let wavelengths_m = BTreeMap::new();
        let offsets_m = BTreeMap::new();
        let float_only_systems = Vec::new();
        let model = rtk_model();
        let ambiguity_set = AmbiguitySet {
            ids: &ambiguity_ids,
            satellites: &ambiguity_satellites,
            scale: rtk_filter::AmbiguityScale {
                wavelengths_m: &wavelengths_m,
                offsets_m: &offsets_m,
            },
            float_only_systems: &float_only_systems,
        };

        let err = solve_rtk_fixed(
            &epochs,
            [0.0; 3],
            ambiguity_set,
            [0.0; 3],
            &model,
            rtk_fixed_options(),
            None,
        )
        .unwrap_err();

        assert!(matches!(err, Error::RtkFixed(_)));
        assert!(err.to_string().contains("fixed RTK"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn solve_ppp_float_positional_wrapper_maps_errors_with_fixture_source() {
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        let epochs = Vec::new();

        let err = solve_ppp_float(&sp3, &epochs, empty_ppp_state(), ppp_float_solve_config())
            .unwrap_err();

        assert!(matches!(err, Error::PppFloat(_)));
        assert!(err.to_string().contains("PPP float"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn solve_ppp_fixed_positional_wrapper_maps_errors_with_fixture_source() {
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        let epochs = Vec::new();

        let err = solve_ppp_fixed(
            &sp3,
            &epochs,
            empty_ppp_float_solution(),
            ppp_fixed_solve_config(),
        )
        .unwrap_err();

        assert!(matches!(err, Error::PppFixed(_)));
        assert!(err.to_string().contains("PPP"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn solve_rtk_float_with_delegates_to_positional_solver() {
        let epochs = Vec::new();
        let ambiguity_ids = Vec::new();
        let model = rtk_model();
        let config = RtkFloatConfig {
            epochs: &epochs,
            base_ecef_m: [0.0; 3],
            ambiguity_ids: &ambiguity_ids,
            initial_baseline_m: [0.0; 3],
            model: &model,
            options: rtk_float_options(),
            receiver_antenna_corrections: None,
        };

        let typed = solve_rtk_float_with(config.clone()).unwrap_err();
        let positional = solve_rtk_float(
            config.epochs,
            config.base_ecef_m,
            config.ambiguity_ids,
            config.initial_baseline_m,
            config.model,
            config.options,
            config.receiver_antenna_corrections,
        )
        .unwrap_err();

        assert!(matches!(typed, Error::RtkFloat(_)));
        assert_eq!(typed.to_string(), positional.to_string());
    }

    #[test]
    fn solve_rtk_float_with_rejects_receiver_antenna_zero_base_geometry() {
        let base = [0.0; 3];
        let baseline = [1.0, 0.0, 0.0];
        let rover = [
            base[0] + baseline[0],
            base[1] + baseline[1],
            base[2] + baseline[2],
        ];
        let g01 = [15_000_000.0, 7_000_000.0, 21_000_000.0];
        let g02 = [-12_000_000.0, 18_000_000.0, 19_000_000.0];
        let range_m = |sat: [f64; 3], recv: [f64; 3]| {
            let dx = sat[0] - recv[0];
            let dy = sat[1] - recv[1];
            let dz = sat[2] - recv[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let mk = |sat: [f64; 3], id: &str| rtk_filter::SatMeas {
            sat: id.into(),
            sd_ambiguity_id: id.into(),
            base_code_m: range_m(sat, base),
            base_phase_m: range_m(sat, base),
            rover_code_m: range_m(sat, rover),
            rover_phase_m: range_m(sat, rover),
            base_tx_pos: sat,
            rover_tx_pos: sat,
            pos: sat,
        };
        let epochs = vec![rtk_filter::Epoch {
            references: vec![mk(g01, "G01")],
            nonref: vec![mk(g02, "G02")],
            velocity_mps: None,
            dt_s: 0.0,
        }];
        let ambiguity_ids = vec!["G02".to_string()];
        let model = rtk_model();
        let cal = rtk_filter::ReceiverAntennaCalibration {
            pco_neu_m: [0.0, 0.0, 0.0],
            noazi_pcv_m: vec![(0.0, 0.0)],
            azi_pcv_m: Vec::new(),
        };
        let corrections = ReceiverAntennaCorrections {
            base: cal.clone(),
            rover: cal,
        };
        let config =
            RtkFloatConfig::new(&epochs, base, &ambiguity_ids, &model, rtk_float_options())
                .with_initial_baseline_m(baseline)
                .with_receiver_antenna_corrections(Some(&corrections));

        let err = solve_rtk_float_with(config).unwrap_err();

        assert!(matches!(
            err,
            Error::RtkFloat(rtk_filter::FloatSolveError::ReceiverAntenna(
                rtk_filter::ReceiverAntennaError::InvalidGeometry
            ))
        ));
    }

    #[test]
    fn solve_rtk_fixed_with_delegates_to_positional_solver() {
        let epochs = Vec::new();
        let ambiguity_ids = Vec::new();
        let ambiguity_satellites = BTreeMap::new();
        let wavelengths_m = BTreeMap::new();
        let offsets_m = BTreeMap::new();
        let float_only_systems = Vec::new();
        let model = rtk_model();
        let ambiguity_set = AmbiguitySet {
            ids: &ambiguity_ids,
            satellites: &ambiguity_satellites,
            scale: rtk_filter::AmbiguityScale {
                wavelengths_m: &wavelengths_m,
                offsets_m: &offsets_m,
            },
            float_only_systems: &float_only_systems,
        };
        let config = RtkFixedConfig {
            epochs: &epochs,
            base_ecef_m: [0.0; 3],
            initial_ambiguities: ambiguity_set,
            initial_baseline_m: [0.0; 3],
            model: &model,
            options: rtk_fixed_options(),
            receiver_antenna_corrections: None,
        };

        let typed = solve_rtk_fixed_with(config.clone()).unwrap_err();
        let positional = solve_rtk_fixed(
            config.epochs,
            config.base_ecef_m,
            config.initial_ambiguities,
            config.initial_baseline_m,
            config.model,
            config.options,
            config.receiver_antenna_corrections,
        )
        .unwrap_err();

        assert!(matches!(typed, Error::RtkFixed(_)));
        assert_eq!(typed.to_string(), positional.to_string());
    }

    #[test]
    fn solve_ppp_float_with_delegates_to_positional_solver() {
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        let epochs = Vec::new();
        let config = PppFloatConfig {
            source: &sp3,
            epochs: &epochs,
            initial_state: empty_ppp_state(),
            solve: ppp_float_solve_config(),
        };

        let typed = solve_ppp_float_with(config.clone()).unwrap_err();
        let positional = solve_ppp_float(
            config.source,
            config.epochs,
            config.initial_state,
            config.solve,
        )
        .unwrap_err();

        assert!(matches!(typed, Error::PppFloat(_)));
        assert_eq!(typed.to_string(), positional.to_string());
    }

    #[test]
    fn solve_ppp_fixed_with_delegates_to_positional_solver() {
        let sp3 = load_sp3(DEGENERATE_SP3).expect("the fixture parses");
        let epochs = Vec::new();
        let config = PppFixedConfig {
            source: &sp3,
            epochs: &epochs,
            float_solution: empty_ppp_float_solution(),
            solve: ppp_fixed_solve_config(),
        };

        let typed = solve_ppp_fixed_with(config.clone()).unwrap_err();
        let positional = solve_ppp_fixed(
            config.source,
            config.epochs,
            config.float_solution,
            config.solve,
        )
        .unwrap_err();

        assert!(matches!(typed, Error::PppFixed(_)));
        assert_eq!(typed.to_string(), positional.to_string());
    }
}
