//! Single-point positioning (SPP).
//!
//! Recovers a receiver ECEF position and clock bias from a set of pseudoranges,
//! a satellite ephemeris source (a precise SP3 product or a broadcast navigation
//! message, via the [`EphemerisSource`] trait), and broadcast ionosphere /
//! Saastamoinen-Niell troposphere correction models. GPS L1 C/A, Galileo E1,
//! BeiDou B1I, and GLONASS G1 are supported; GPS, BeiDou, and GLONASS use
//! broadcast Klobuchar coefficients with carrier-frequency scaling, while
//! Galileo can use its broadcast NeQuick-G `ai0`/`ai1`/`ai2` coefficients when
//! supplied. GLONASS is FDMA, so its per-satellite carrier is resolved from the
//! broadcast/observation channel number ([`SolveInputs::glonass_channels`]) and
//! the Klobuchar L1 delay is scaled to it by `(f_L1 / f_k)^2`, matching
//! RTKLIB-demo5, which applies no per-satellite inter-frequency bias and carries
//! the single GLO-GPS offset on the per-system receiver clock. A satellite whose
//! carrier cannot be resolved is rejected when the ionosphere correction is
//! requested.
//!
//! The state vector is `[x_m, y_m, z_m, clk_0, clk_1, ...]`: three ECEF position
//! components (meters) followed by one receiver clock per distinct GNSS in the
//! solve, expressed as a length (meters). A single-system solve reduces to the
//! classic `[x_m, y_m, z_m, b_m]`; a multi-system solve adds an inter-system
//! bias parameter for each additional constellation. The seconds value
//! `rx_clock_s = clk_0 / c` (the reference system) and the per-system clocks are
//! reported only at the API boundary.
//!
//! The per-satellite predicted pseudorange is built in a pinned operation order:
//! a fixed-count transmit-time iteration (receive time minus geometric range
//! over `c`) locates the satellite ephemeris at transmission, an Earth-rotation
//! (Sagnac) closed-form rotation brings the satellite into the receive-time
//! frame, the geometric range and the line-of-sight azimuth/elevation follow,
//! then the ionosphere and troposphere delays are added to the predicted range
//! left-to-right. The residual the solver sees is `sqrt(w) * (P_meas - P_hat)`
//! with an elevation-based weight evaluated once at the frozen initial-guess
//! geometry.
//!
//! The geometric/clock/correction substrate and its 2-point finite-difference
//! Jacobian are arithmetic over the libm-bound model functions and are a
//! bit-exact (0-ULP) parity target against the reference recipe. The converged
//! position is produced by the trust-region least-squares solver in the
//! `sidereon-core` solver core, whose linear-algebra step is not bit-reproducible
//! across BLAS builds; the converged solution is therefore a sub-micron
//! solver-agreement result, not a 0-ULP claim.
//!
//! The bit-exact claim depends on the fused-multiply-add policy matching the
//! reference exactly. The substrate uses no contracted `a*b+c` anywhere the
//! reference computes the two roundings separately; the single deliberate
//! exception is the 3x3-by-vector rotation primitive, which uses `mul_add` to
//! reproduce the reference's rounding of that product. The certified target
//! pins `target-cpu`/features so the compiler neither introduces nor drops a
//! contraction; on a host that auto-contracts these expressions the last bit
//! can differ and the goldens are not expected to hold.

use crate::astro::angles::rad_to_deg_ref;
use crate::astro::math::least_squares::{
    self, solve_trf_with, LeastSquaresProblem, SolveOptions, Status, TrustRegionSolve,
};
use nalgebra::DVector;
use std::collections::BTreeMap;

mod config;
mod fallback;
mod source;
use crate::astro::math::robust::{huber_weight, mad_scale, RobustError};
pub use config::{
    DEFAULT_HUBER_K, DEFAULT_ROBUST_MAX_OUTER, DEFAULT_ROBUST_OUTER_TOL_M,
    DEFAULT_ROBUST_SCALE_FLOOR_M, ELEVATION_MASK_RAD, SIGMA0_M, TRANSMIT_TIME_ITERATIONS,
};
pub use fallback::{
    solve_broadcast, solve_with_fallback, BroadcastReason, FallbackError, FixSource,
    SourcedSolution,
};
pub use source::EphemerisSource;

pub use crate::constants::{C_M_S, F_L1_HZ, OMEGA_E_DOT_RAD_S};
use crate::dop::{dop, dop_multi, Dop, LineOfSight};
use crate::estimation::recipe::{
    EstimationRecipe, FrameRecipe, RangeRecipe, SagnacRecipe, SolverRecipe,
};
use crate::estimation::substrate::frames::{az_el_from_ecef, geodetic_from_ecef};
use crate::estimation::substrate::parameters::ParameterLayout;
use crate::estimation::substrate::range::{geometric_range, rotate_transmit_satellite};
use crate::frame::{ItrfPositionM, Wgs84Geodetic};
use crate::frequencies;
use crate::id::{GnssSatelliteId, GnssSystem};
pub use crate::ionex::GalileoNequickCoeffs;
use crate::ionex::{
    galileo_nequick_g_native_unchecked, klobuchar_native_unchecked, GalileoNequickEval,
    KlobucharParams,
};
use crate::quality::{
    validate_receiver_solution, SolutionValidationError, SolutionValidationOptions,
};
use crate::tropo::slant_components;
use crate::validate;

/// The single-frequency carrier (Hz) the ionosphere correction is reported on
/// for a constellation with one fixed single-frequency carrier, or `None` for a
/// system that has none (GLONASS, whose FDMA carrier is per-satellite). GPS L1
/// C/A and Galileo E1 are both at [`F_L1_HZ`]; BeiDou uses B1I. Klobuchar and
/// Galileo broadcast delays are reported on this carrier. GLONASS is resolved
/// per satellite by [`spp_iono_frequency_hz`] from its FDMA channel instead.
pub(crate) const fn carrier_frequency_hz(system: GnssSystem) -> Option<f64> {
    frequencies::default_spp_frequency_hz(system)
}

/// The carrier frequency (Hz) the broadcast ionosphere delay is scaled to for a
/// single satellite, or `None` if the satellite's system has no carrier the
/// model can resolve.
///
/// For the fixed-carrier systems (GPS L1, Galileo E1, BeiDou B1I) this is the
/// system carrier from [`carrier_frequency_hz`]. GLONASS is FDMA, so its carrier
/// is per-satellite: it is resolved from `glonass_channels` (the broadcast /
/// observation FDMA channel `k` keyed by GLONASS slot number) as the G1
/// frequency `1602.0 MHz + k * 562.5 kHz`. A GLONASS satellite whose channel is
/// not in the map, or whose channel is outside the valid FDMA range
/// `[-7, +6]` (the same domain the RINEX nav/obs parsers enforce via
/// [`crate::rinex_nav::valid_glonass_frequency_channel`]), has no resolvable
/// carrier and returns `None` -- `glonass_g1_frequency_hz` is a pure
/// `1602.0 MHz + k * 562.5 kHz` evaluation that would otherwise return a
/// bogus-but-positive carrier for an out-of-domain `k`. Mirroring RTKLIB-demo5,
/// the single GLO-GPS inter-system offset is carried by the existing per-system
/// receiver clock (see [`clock_systems`]) rather than a separate
/// inter-frequency-bias parameter, and the only GLONASS-specific term in the
/// measurement model is this per-satellite `(f_L1 / f_k)^2` ionosphere scaling.
pub(crate) fn spp_iono_frequency_hz(
    sat: GnssSatelliteId,
    glonass_channels: &BTreeMap<u8, i8>,
) -> Option<f64> {
    match sat.system {
        GnssSystem::Glonass => glonass_channels
            .get(&sat.prn)
            .copied()
            .filter(|&k| crate::rinex_nav::valid_glonass_frequency_channel(i32::from(k)))
            .map(frequencies::glonass_g1_frequency_hz),
        _ => carrier_frequency_hz(sat.system),
    }
}
use crate::constants::MEAN_EARTH_RADIUS_M;
const PI: f64 = std::f64::consts::PI;

// Agreement-track stopping thresholds for the independent SPP least-squares
// solver. These drive the solver to the true fixed point of the noise-free,
// by-construction-zero-residual problem so the converged position agrees with
// the reference solution to the documented sub-micron bound; they are the
// solver's own stopping thresholds, not a parity target's pinned scipy options.
/// Canonical light-time convergence tolerance (s). The canonical range recipe
/// ([`RangeRecipe::CanonicalLightTimeClosedFormSagnac`]) iterates the
/// transmit-epoch light-time loop until the signal travel time changes by less
/// than this between iterations, instead of the reference recipe's fixed
/// [`TRANSMIT_TIME_ITERATIONS`] truncation. `1e-13 s` is ~30 microns of range
/// (`tol * C_M_S`), far below the pseudorange noise floor; the loop is
/// quadratically convergent so it reaches this in ~3 iterations.
const CANONICAL_LIGHT_TIME_TOL_S: f64 = 1.0e-13;
/// Iteration cap for the canonical light-time loop, a safety bound the
/// quadratically convergent iteration never reaches in practice (it converges in
/// ~3 iterations); present so a pathological geometry cannot spin forever.
const CANONICAL_LIGHT_TIME_MAX_ITERS: usize = 10;
/// First-order optimality tolerance on `||J^T r||_inf`.
const SPP_SOLVER_GTOL: f64 = 1e-14;
/// Relative-cost-reduction tolerance.
const SPP_SOLVER_FTOL: f64 = 1e-15;
/// Relative-step tolerance.
const SPP_SOLVER_XTOL: f64 = 1e-14;
/// Maximum number of residual evaluations.
const SPP_SOLVER_MAX_NFEV: usize = 400;

/// A single GPS L1 pseudorange observation.
///
/// The input boundary of the pipeline is the pseudorange; raw observation
/// formation (RINEX decoding, code tracking) is out of scope. The receive epoch
/// and the time-of-day / day-of-year arguments are common to all observations
/// in one solve and are carried on [`SolveInputs`], not here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// The transmitting satellite.
    pub satellite_id: GnssSatelliteId,
    /// Measured pseudorange in meters.
    pub pseudorange_m: f64,
}

/// Why a satellite was excluded from the solve, in pinned priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// The SP3 product has no usable position or clock for the satellite at the
    /// transmit epoch.
    NoEphemeris,
    /// The satellite is below the elevation mask at the frozen geometry.
    LowElevation,
}

/// A rejected satellite paired with its rejection reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RejectedSat {
    /// The excluded satellite.
    pub satellite_id: GnssSatelliteId,
    /// The first matching rejection reason.
    pub reason: RejectionReason,
}

/// Models and convergence detail describing how a solution was produced.
#[derive(Debug, Clone, PartialEq)]
pub struct SolutionMetadata {
    /// Number of accepted solver iterations.
    pub iterations: usize,
    /// Whether the solver reached a convergence stopping criterion (as opposed
    /// to exhausting its evaluation budget).
    pub converged: bool,
    /// The solver's termination status.
    pub status: Status,
    /// Whether the ionosphere correction was applied.
    pub ionosphere_applied: bool,
    /// Whether the troposphere correction was applied.
    pub troposphere_applied: bool,
    /// Number of outer robust-reweighting iterations performed. `0` on the
    /// static path (`robust = None`); on the robust path this counts the
    /// reweighted resolves beyond the warm-start solve.
    pub outer_iterations: usize,
    /// The final MAD robust scale (m) of the last outer iteration, or `None` on
    /// the static path.
    pub final_robust_scale_m: Option<f64>,
    /// Number of satellites used in the final solve.
    pub used_count: usize,
    /// Distinct GNSS systems present in the final solve, in ascending order.
    pub systems: Vec<GnssSystem>,
    /// Degrees of freedom, `used_count - (3 + systems.len())`.
    pub redundancy: isize,
    /// Whether residual-based RAIM can test the final solve (`redundancy >= 1`).
    pub raim_checkable: bool,
}

/// A receiver position/clock solution with its geometry diagnostics.
#[derive(Debug, Clone)]
pub struct ReceiverSolution {
    /// Converged receiver position, ITRF/IGS ECEF meters.
    pub position: ItrfPositionM,
    /// The geodetic form of the position, if the conversion was requested.
    pub geodetic: Option<Wgs84Geodetic>,
    /// Receiver clock bias in seconds (`clk_0 / c`) for the reference GNSS - the
    /// first entry of `system_clocks_s`. For a single-system solve this is the
    /// only clock; for a multi-system solve the other systems' absolute clocks
    /// are in `system_clocks_s`.
    pub rx_clock_s: f64,
    /// The absolute receiver clock for each GNSS in the solve, in ascending
    /// system order, in seconds. One entry for a single-system solve; one per
    /// constellation for a multi-system solve. The first entry equals
    /// `rx_clock_s`; the inter-system bias for any other system is *its clock
    /// minus that reference* (these are absolute per-system clocks, not biases).
    pub system_clocks_s: Vec<(GnssSystem, f64)>,
    /// Dilution-of-precision scalars from the converged geometry. A
    /// single-system solve uses the 0-ULP four-state cofactor; a multi-system
    /// solve uses the general inverse with one clock column per constellation (a
    /// deterministic diagnostic, not a 0-ULP target). `None` only if the
    /// converged geometry is rank-deficient.
    pub dop: Option<Dop>,
    /// Per-constellation time (clock) DOP, one entry per GNSS in the solve, in
    /// the same ascending system order as `system_clocks_s`: the square root of
    /// that system's clock cofactor variance. The first entry's value equals
    /// `dop.tdop` (the reference clock). One entry for a single-system solve.
    /// Empty only when `dop` is `None` (rank-deficient geometry).
    ///
    /// This is exactly `dop.system_tdops`: the geometry layer reports the
    /// per-system TDOPs already GNSS-tagged in [`Dop::system_tdops`], so this is
    /// a direct copy and needs no re-tagging.
    pub system_tdops: Vec<(GnssSystem, f64)>,
    /// Post-fit residuals in meters, in `used_sats` order (unweighted
    /// `P_meas - P_hat`).
    pub residuals_m: Vec<f64>,
    /// The satellites that contributed to the solve, ascending id order.
    pub used_sats: Vec<GnssSatelliteId>,
    /// The excluded satellites, each with its reason.
    pub rejected_sats: Vec<RejectedSat>,
    /// Iteration / convergence / model metadata.
    pub metadata: SolutionMetadata,
}

impl ReceiverSolution {
    /// Root-mean-square of the post-fit pseudorange residuals over the used satellites (0.0 when empty).
    pub fn residual_rms_m(&self) -> f64 {
        residual_rms(&self.residuals_m)
    }
}

/// Which correction terms a solve applies, building up incrementally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Corrections {
    /// Apply the Klobuchar L1 ionosphere delay.
    pub ionosphere: bool,
    /// Apply the Saastamoinen/Niell troposphere delay.
    pub troposphere: bool,
}

impl Corrections {
    /// No atmospheric corrections (geometry + clock + Sagnac only).
    pub const NONE: Self = Self {
        ionosphere: false,
        troposphere: false,
    };
    /// Ionosphere only.
    pub const IONO: Self = Self {
        ionosphere: true,
        troposphere: false,
    };
    /// Ionosphere and troposphere.
    pub const IONO_TROPO: Self = Self {
        ionosphere: true,
        troposphere: true,
    };
}

/// Broadcast Klobuchar coefficients for the ionosphere term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KlobucharCoeffs {
    /// Cosine-amplitude polynomial coefficients (a0..a3).
    pub alpha: [f64; 4],
    /// Period polynomial coefficients (b0..b3).
    pub beta: [f64; 4],
}

/// Surface meteorology for the troposphere term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceMet {
    /// Total pressure (hPa).
    pub pressure_hpa: f64,
    /// Temperature (K).
    pub temperature_k: f64,
    /// Relative humidity, fraction in `[0, 1]`.
    pub relative_humidity: f64,
}

impl Default for SurfaceMet {
    /// Standard atmosphere: 1013.25 hPa, 288.15 K, 0.5 relative humidity.
    fn default() -> Self {
        Self {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        }
    }
}

/// Opt-in Huber/IRLS robust-reweighting configuration.
///
/// When a [`SolveInputs::robust`] is `Some(_)`, the solve runs an outer
/// iteratively-reweighted least-squares loop on top of the static elevation
/// weighting: a warm-start solve at the base elevation weights (bit-identical to
/// the static path), then re-solves that rebuild the weight vector each outer
/// iteration as `base_elevation_weight * huber(r_i / s)`, where `r_i` is the
/// current unweighted post-fit residual and `s` is a floored MAD scale. With
/// `robust = None` the solve is byte-identical to the static elevation-weighted
/// solve. `Default` matches the `DEFAULT_*` config constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustConfig {
    /// Huber tuning constant `k`; residuals scaled below this keep full weight.
    pub huber_k: f64,
    /// Floor (m) on the MAD scale, preventing a near-perfect fit from
    /// down-weighting every satellite.
    pub scale_floor_m: f64,
    /// Maximum total outer solves (the warm start plus reweighted resolves).
    pub max_outer: usize,
    /// Outer-loop position L2 step tolerance (m).
    pub outer_tol_m: f64,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            huber_k: DEFAULT_HUBER_K,
            scale_floor_m: DEFAULT_ROBUST_SCALE_FLOOR_M,
            max_outer: DEFAULT_ROBUST_MAX_OUTER,
            outer_tol_m: DEFAULT_ROBUST_OUTER_TOL_M,
        }
    }
}

/// Everything one SPP solve needs besides the SP3 product itself.
///
/// The receive epoch is carried as seconds-since-J2000 (`t_rx_j2000_s`), the
/// argument the transmit-time iteration differences against the geometric range
/// to land the satellite ephemeris at transmission, with no Julian-date
/// round-trip inside the loop. The Klobuchar diurnal argument
/// (`t_rx_second_of_day_s`) and the Niell seasonal argument (`day_of_year`) are
/// supplied directly so the correction kernels run in their bit-exact native
/// units.
#[derive(Debug, Clone)]
pub struct SolveInputs {
    /// The pseudorange observations (any order; the solve sorts them).
    pub observations: Vec<Observation>,
    /// Receive epoch, seconds since J2000 in the SP3 product's time scale.
    pub t_rx_j2000_s: f64,
    /// GPS second-of-day of the receive epoch (Klobuchar diurnal argument).
    pub t_rx_second_of_day_s: f64,
    /// Fractional day-of-year of the receive epoch (Niell seasonal argument).
    pub day_of_year: f64,
    /// Initial guess `[x_m, y_m, z_m, b_m]`.
    pub initial_guess: [f64; 4],
    /// The correction terms to apply.
    pub corrections: Corrections,
    /// Broadcast Klobuchar coefficients (used iff `corrections.ionosphere`).
    /// Applied to every system unless `beidou_klobuchar` overrides BeiDou.
    pub klobuchar: KlobucharCoeffs,
    /// Optional BeiDou-specific Klobuchar coefficients (the broadcast `BDSA`/
    /// `BDSB` set). When present, BeiDou satellites use these instead of
    /// [`klobuchar`](Self::klobuchar); both feed the same model, frequency-scaled
    /// to B1I. `None` falls back to `klobuchar` for BeiDou too.
    pub beidou_klobuchar: Option<KlobucharCoeffs>,
    /// Optional Galileo-specific NeQuick-G coefficients (the broadcast `GAL`
    /// `ai0`/`ai1`/`ai2` set). When present, Galileo satellites use these instead
    /// of the GPS Klobuchar coefficients. `None` preserves the historical
    /// Klobuchar fallback so existing zero-Galileo goldens stay bit-identical.
    pub galileo_nequick: Option<GalileoNequickCoeffs>,
    /// GLONASS FDMA channel numbers keyed by GLONASS slot (PRN), from the
    /// broadcast nav `freq_channel` field or the observation header's
    /// `GLONASS SLOT / FRQ #` records. Used only to resolve the per-satellite
    /// GLONASS carrier for the ionosphere `(f_L1 / f_k)^2` scaling; an empty map
    /// is correct for any solve with no GLONASS observation and leaves every
    /// other constellation bit-identical. A GLONASS observation with the
    /// ionosphere correction requested but no channel here is rejected with
    /// [`SppError::IonosphereUnsupported`].
    pub glonass_channels: BTreeMap<u8, i8>,
    /// Surface meteorology (used iff `corrections.troposphere`).
    pub met: SurfaceMet,
    /// Opt-in Huber/IRLS robust reweighting. `None` (the default behavior)
    /// runs the static elevation-weighted solve byte-identically; `Some(_)`
    /// adds the outer reweighting loop described on [`RobustConfig`].
    pub robust: Option<RobustConfig>,
}

/// Input-validation failure category for SPP public entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SppInputErrorKind {
    /// A floating-point input was NaN or infinite.
    NonFinite,
    /// A positive physical input was zero or negative.
    NotPositive,
    /// A non-negative physical input was negative.
    Negative,
    /// A finite numeric input was outside its accepted range.
    OutOfRange,
    /// A required input field was absent.
    Missing,
    /// A text field could not be parsed as a float.
    FloatParse,
    /// A text field could not be parsed as an integer.
    IntParse,
    /// A civil date field was out of range.
    InvalidCivilDate,
    /// A civil time field was out of range.
    InvalidCivilTime,
}

impl core::fmt::Display for SppInputErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let label = match self {
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::Negative => "negative",
            Self::OutOfRange => "out of range",
            Self::Missing => "missing",
            Self::FloatParse => "invalid float",
            Self::IntParse => "invalid integer",
            Self::InvalidCivilDate => "invalid civil date",
            Self::InvalidCivilTime => "invalid civil time",
        };
        f.write_str(label)
    }
}

impl From<&validate::FieldError> for SppInputErrorKind {
    fn from(error: &validate::FieldError) -> Self {
        match error {
            validate::FieldError::Missing { .. } => Self::Missing,
            validate::FieldError::NonFinite { .. } => Self::NonFinite,
            validate::FieldError::NotPositive { .. } => Self::NotPositive,
            validate::FieldError::Negative { .. } => Self::Negative,
            validate::FieldError::OutOfRange { .. } => Self::OutOfRange,
            validate::FieldError::FloatParse { .. } => Self::FloatParse,
            validate::FieldError::IntParse { .. } => Self::IntParse,
            validate::FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            validate::FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

/// Error from [`solve`].
#[derive(Debug, Clone)]
pub enum SppError {
    /// A public SPP input was malformed, non-finite, or outside its physical
    /// domain. Boundary validation rejects this before satellite selection or
    /// least-squares evaluation.
    InvalidInput {
        /// The invalid input field.
        field: &'static str,
        /// The validation failure category.
        kind: SppInputErrorKind,
    },
    /// Fewer usable satellites survived rejection than the solve has parameters
    /// (`3 + n_systems`: three position components plus one receiver clock per
    /// GNSS), so the solve is underdetermined.
    TooFewSatellites {
        /// The number of satellites that survived rejection.
        used: usize,
        /// The number of satellites required (`3 + n_systems`).
        required: usize,
    },
    /// The trust-region step hit a rank-deficient Jacobian (degenerate geometry).
    Singular(least_squares::SolveError),
    /// The same satellite appears in more than one observation. One pseudorange
    /// per satellite is required, so the input is rejected rather than silently
    /// picking one (which would make the result depend on observation order).
    DuplicateObservation {
        /// The satellite that was observed more than once.
        satellite: GnssSatelliteId,
    },
    /// A satellite that survived the frozen selection had no usable SP3
    /// position/clock at a transmit epoch reached during the solve. Returned
    /// instead of panicking; normally precluded by the selection step.
    EphemerisLost {
        /// The satellite whose ephemeris became unavailable during the solve.
        satellite: GnssSatelliteId,
    },
    /// The ionosphere correction was requested but an observed satellite has no
    /// resolvable carrier frequency, so the L1 Klobuchar delay cannot be scaled
    /// to it. GPS L1, Galileo E1, and BeiDou B1I have fixed carriers; a GLONASS
    /// satellite resolves its per-satellite FDMA carrier from
    /// [`SolveInputs::glonass_channels`], so a GLONASS observation whose channel
    /// is absent from that map -- or present but outside the valid FDMA range
    /// `[-7, +6]` -- (rather than GLONASS as a whole) is rejected here rather
    /// than corrected with an undefined or out-of-domain frequency.
    IonosphereUnsupported {
        /// The satellite the ionosphere model does not cover.
        satellite: GnssSatelliteId,
    },
}

impl core::fmt::Display for SppError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SppError::InvalidInput { field, kind } => {
                write!(f, "invalid SPP input {field}: {kind}")
            }
            SppError::TooFewSatellites { used, required } => write!(
                f,
                "only {used} usable satellites; need at least {required} \
                 (3 position + 1 clock per GNSS)"
            ),
            SppError::Singular(e) => write!(f, "degenerate geometry: {e}"),
            SppError::DuplicateObservation { satellite } => {
                write!(f, "satellite {satellite} observed more than once")
            }
            SppError::EphemerisLost { satellite } => {
                write!(f, "satellite {satellite} lost ephemeris during the solve")
            }
            SppError::IonosphereUnsupported { satellite } => write!(
                f,
                "ionosphere correction has no modeled carrier frequency for {satellite}"
            ),
        }
    }
}

impl std::error::Error for SppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SppError::Singular(error) => Some(error),
            _ => None,
        }
    }
}

impl From<least_squares::SolveError> for SppError {
    fn from(e: least_squares::SolveError) -> Self {
        SppError::Singular(e)
    }
}

/// Language-independent SPP solve policy used by the public API boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SolvePolicy {
    /// Business-level solution validation gates.
    pub validation: SolutionValidationOptions,
    /// Optional count of near-surface golden-spiral seeds for cold starts.
    pub coarse_search_seeds: Option<usize>,
}

/// Error from [`solve_with_policy`].
#[derive(Debug, Clone)]
pub enum SolvePolicyError {
    /// The underlying SPP solver failed.
    Solve(SppError),
    /// The solved receiver state failed a business-level validation gate.
    Validation(SolutionValidationError),
    /// Coarse search found no converged redundant candidate.
    NoCoarseSolution,
}

impl From<SppError> for SolvePolicyError {
    fn from(error: SppError) -> Self {
        Self::Solve(error)
    }
}

impl From<SolutionValidationError> for SolvePolicyError {
    fn from(error: SolutionValidationError) -> Self {
        Self::Validation(error)
    }
}

impl core::fmt::Display for SolvePolicyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Solve(error) => write!(f, "SPP solve failed: {error}"),
            Self::Validation(error) => write!(f, "SPP validation failed: {error}"),
            Self::NoCoarseSolution => write!(f, "coarse search found no converged SPP solution"),
        }
    }
}

impl std::error::Error for SolvePolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Solve(error) => Some(error),
            Self::Validation(error) => Some(error),
            Self::NoCoarseSolution => None,
        }
    }
}

/// The SPP measurement-model operation-order selections, resolved from a
/// strategy's [`EstimationRecipe`]: the transmit-time light-time range recipe,
/// the Sagnac rotation recipe, and the receiver-frame (geodetic / az-el) recipe.
///
/// Threading these into [`sat_model`] is what makes SPP consume its
/// `recipe.range` / `recipe.sagnac` / `recipe.frame` rather than hard-coding a
/// single op-order. [`Self::reference`] is the SPP Skyfield reference selection,
/// so the legacy entry points reproduce the current behavior bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SppModelRecipe {
    pub range: RangeRecipe,
    pub sagnac: SagnacRecipe,
    pub frame: FrameRecipe,
}

impl SppModelRecipe {
    /// The model selections carried by `recipe` (its range/sagnac/frame stages).
    pub(crate) const fn from_recipe(recipe: &EstimationRecipe) -> Self {
        Self {
            range: recipe.range,
            sagnac: recipe.sagnac,
            frame: recipe.frame,
        }
    }

    /// The SPP Skyfield reference model selections (the
    /// [`EstimationRecipe::spp`] range/sagnac/frame stages).
    pub(crate) const fn reference() -> Self {
        Self::from_recipe(&EstimationRecipe::spp())
    }
}

/// Per-satellite model used by the solve path: the Sagnac-rotated satellite
/// position, the topocentric az/el, and the predicted pseudorange.
///
/// In test builds the struct additionally carries the named intermediate
/// quantities (transmit time, satellite ECEF, Sagnac angle, geometric range,
/// ionosphere, troposphere) so the 0-ULP trace-replay parity test can assert
/// each one bit-for-bit against the reference recipe; the solve path never
/// reads them, so they are gated out of production builds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SatModel {
    pub sat_rot_ecef_m: [f64; 3],
    pub el_rad: f64,
    pub p_hat_m: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub az_rad: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub tau_s: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub t_tx_j2000_s: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub sat_ecef_m: [f64; 3],
    #[cfg(all(test, sidereon_repo_tests))]
    pub dt_sat_s: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub theta_rad: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub rho_m: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub iono_m: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub tropo_m: f64,
}

/// The broadcast ionosphere correction a satellite's system uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SppIonosphere {
    /// GPS/BeiDou Klobuchar alpha/beta model.
    Klobuchar(KlobucharCoeffs),
    /// Galileo NeQuick-G effective-ionisation coefficients.
    GalileoNequick(GalileoNequickCoeffs),
}

/// The ionosphere coefficients a satellite's system uses: Galileo prefers its
/// `galileo_nequick` (`GAL`) set when present; BeiDou prefers its
/// `beidou_klobuchar` (`BDSA`/`BDSB`) set when present; all missing
/// constellation-specific sets fall back to the shared GPS Klobuchar values to
/// preserve existing callers.
fn ionosphere_for(system: GnssSystem, inputs: &SolveInputs) -> SppIonosphere {
    match (system, inputs.galileo_nequick, inputs.beidou_klobuchar) {
        (GnssSystem::Galileo, Some(gal), _) => SppIonosphere::GalileoNequick(gal),
        (GnssSystem::BeiDou, _, Some(bds)) => SppIonosphere::Klobuchar(bds),
        _ => SppIonosphere::Klobuchar(inputs.klobuchar),
    }
}

/// Per-epoch inputs shared by every satellite's [`sat_model`] evaluation in a
/// solve: the ephemeris source plus the epoch and correction arguments that do
/// not vary between satellites. Bundling them lets [`sat_model`] take only the
/// per-satellite arguments (id, receiver state, measurement, system Klobuchar)
/// instead of a long positional parameter list.
pub(crate) struct SatModelEnv<'a> {
    pub eph: &'a dyn EphemerisSource,
    /// Receive epoch, seconds since J2000 in the SP3 product's time scale.
    pub t_rx_j2000_s: f64,
    /// GPS second-of-day of the receive epoch (Klobuchar diurnal argument).
    pub t_rx_second_of_day_s: f64,
    /// Fractional day-of-year of the receive epoch (Niell seasonal argument).
    pub day_of_year: f64,
    /// The correction terms to apply.
    pub corrections: Corrections,
    /// Surface meteorology (used iff `corrections.troposphere`).
    pub met: &'a SurfaceMet,
    /// GLONASS FDMA channel numbers keyed by slot (PRN), used to resolve the
    /// per-satellite GLONASS carrier for the ionosphere scaling.
    pub glonass_channels: &'a BTreeMap<u8, i8>,
    /// The range/sagnac/frame operation-order selections [`sat_model`] consumes,
    /// resolved from the strategy's recipe.
    pub model: SppModelRecipe,
}

/// Build the per-satellite predicted pseudorange in the SPP operation order
/// SELECTED BY THE RECIPE on [`SatModelEnv::model`], sharing the
/// parity-sensitive range and frame substrate with the other strategies.
///
/// The three model stages are read from the recipe rather than hard-coded:
/// - **range** (`env.model.range`): the transmit-time light-time iteration.
///   [`RangeRecipe::SppMeasuredPseudorangeFixedIter`] (the SPP reference) seeds
///   `tau` from the measured pseudorange and runs a fixed iteration count (no
///   convergence test). [`RangeRecipe::CanonicalLightTimeClosedFormSagnac`] (the
///   canonical strategy) seeds the same way but iterates the light-time loop to
///   convergence (the IERS-rigorous op-order). These are the two light-time
///   recipes the SPP measurement model implements; the observable
///   rounded-microsecond and RTK provided-transmit recipes are other strategies'
///   range models and never reach here.
/// - **sagnac** (`env.model.sagnac`): the closed-form Sagnac Z-rotation and the
///   pre/post-rotation geometric range route through
///   [`crate::estimation::substrate::range`] under the selected recipe.
/// - **frame** (`env.model.frame`): the receiver geodetic conversion and the
///   geodetic ENU azimuth/elevation route through
///   [`crate::estimation::substrate::frames`] under the selected recipe (the SPP
///   reference selects [`FrameRecipe::SppSkyfieldAuThreeIter`], the Skyfield AU
///   three-iteration solve).
///
/// The raw residual ([`residual_unweighted`], `P_meas - P_hat`) the trust-region
/// finite-difference solver differences carries no design rows of its own; the
/// substrate [`crate::estimation::substrate::rows`] `ResidualRow` assembly serves
/// the RTK/PPP normal-equation stacks.
///
/// Returns `None` if the ephemeris source has no usable position/clock for the
/// satellite at the transmit epoch.
pub(crate) fn sat_model(
    env: &SatModelEnv,
    sat: GnssSatelliteId,
    rx_ecef_m: [f64; 3],
    b_m: f64,
    p_meas_m: f64,
    ionosphere: SppIonosphere,
) -> Option<SatModel> {
    let sagnac = env.model.sagnac;
    let frame = env.model.frame;

    // Transmit-time light-time iteration, selected by the range recipe.
    let (sat_pos, dt_sat, tau) = match env.model.range {
        RangeRecipe::SppMeasuredPseudorangeFixedIter => {
            // Fixed iteration count, no inner convergence test; seed tau from the
            // measured pseudorange.
            let mut tau = p_meas_m / C_M_S;
            let mut t_tx = env.t_rx_j2000_s - tau;
            let mut sat_pos = [0.0f64; 3];
            let mut dt_sat = 0.0f64;
            for _ in 0..TRANSMIT_TIME_ITERATIONS {
                let (pos, clk) = env.eph.position_clock_at_j2000_s(sat, t_tx)?;
                sat_pos = pos;
                dt_sat = clk;
                // Pre-rotation geometric range through the shared substrate (the
                // closed-form recipe = plain `norm3(sub3(sat, recv))`).
                let rho0 = geometric_range(sagnac, sat_pos, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S);
                tau = rho0 / C_M_S;
                t_tx = env.t_rx_j2000_s - tau;
            }
            (sat_pos, dt_sat, tau)
        }
        RangeRecipe::CanonicalLightTimeClosedFormSagnac => {
            // Full iterative light-time (the IERS-rigorous op-order): iterate the
            // transmit epoch until the signal travel time stops changing, rather
            // than the reference recipe's fixed two-iteration truncation. Seeded,
            // like the reference, from the measured pseudorange; the iteration
            // converges to the geometric light-time fixed point
            // `t_tx = t_rx - rho(t_tx)/c` with the closed-form Sagnac range (never
            // a first-order scalar Sagnac). The satellite clock `dt_sat` returned
            // by the ephemeris already carries the relativistic periodic term
            // (the broadcast Keplerian evaluation applies `F*e*sqrt(A)*sin(E)`;
            // SP3 precise clocks include it, the SPP L3 no-op), so the canonical
            // relativistically-correct range consumes it directly with no
            // double-counting term.
            let mut tau = p_meas_m / C_M_S;
            let mut t_tx = env.t_rx_j2000_s - tau;
            let mut sat_pos = [0.0f64; 3];
            let mut dt_sat = 0.0f64;
            let mut prev_tau = f64::INFINITY;
            for _ in 0..CANONICAL_LIGHT_TIME_MAX_ITERS {
                let (pos, clk) = env.eph.position_clock_at_j2000_s(sat, t_tx)?;
                sat_pos = pos;
                dt_sat = clk;
                let rho0 = geometric_range(sagnac, sat_pos, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S);
                tau = rho0 / C_M_S;
                t_tx = env.t_rx_j2000_s - tau;
                if (tau - prev_tau).abs() <= CANONICAL_LIGHT_TIME_TOL_S {
                    break;
                }
                prev_tau = tau;
            }
            (sat_pos, dt_sat, tau)
        }
        RangeRecipe::ObservableRoundedMicrosecondFixedIter
        | RangeRecipe::RtkProvidedTxFirstOrderSagnac => unreachable!(
            "the SPP measurement model runs only the measured-pseudorange or canonical light-time recipe"
        ),
    };

    // Sagnac / Earth-rotation rotation over the flight time, selected by recipe.
    let sat_rot = rotate_transmit_satellite(sagnac, sat_pos, tau, OMEGA_E_DOT_RAD_S);

    // Geometric range (post-Sagnac) through the shared substrate.
    let rho = geometric_range(sagnac, sat_rot, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S);

    // Geometry for corrections: az/el from rx and the Sagnac-rotated satellite,
    // through the recipe-selected frame substrate.
    let g = az_el_from_ecef(frame, rx_ecef_m, sat_rot);

    let mut iono_m = 0.0;
    let mut tropo_m = 0.0;
    if env.corrections.ionosphere {
        // The SPP 0-ULP trace oracle pins this multiply-then-divide order, which
        // `rad_to_deg_ref` implements (`rad * 180 / PI`).
        let lat_deg = rad_to_deg_ref(g.geodetic.lat_rad);
        let lon_deg = rad_to_deg_ref(g.geodetic.lon_rad);
        let az_deg = rad_to_deg_ref(g.az_rad);
        let el_deg = rad_to_deg_ref(g.el_rad);
        // A used satellite always has a resolvable carrier here (the solve
        // rejects an ionosphere request for any satellite that does not, GLONASS
        // included via its FDMA channel), so the fallback is unreachable. The
        // GLONASS per-satellite carrier makes the Klobuchar delay scale by
        // `(f_L1 / f_k)^2` inside the kernel, exactly as RTKLIB-demo5 does.
        let freq_hz = spp_iono_frequency_hz(sat, env.glonass_channels).unwrap_or(F_L1_HZ);
        iono_m = match ionosphere {
            SppIonosphere::Klobuchar(klobuchar) => klobuchar_native_unchecked(
                &KlobucharParams {
                    alpha: klobuchar.alpha,
                    beta: klobuchar.beta,
                },
                lat_deg,
                lon_deg,
                az_deg,
                el_deg,
                env.t_rx_second_of_day_s,
                freq_hz,
            ),
            SppIonosphere::GalileoNequick(coeffs) => galileo_nequick_g_native_unchecked(
                &coeffs,
                GalileoNequickEval {
                    lat_deg,
                    lon_deg,
                    el_deg,
                    t_gal_s: env.t_rx_second_of_day_s,
                    day_of_year: env.day_of_year,
                    frequency_hz: freq_hz,
                },
            ),
        };
    }
    if env.corrections.troposphere {
        tropo_m = slant_components(
            g.el_rad,
            g.geodetic,
            env.met.pressure_hpa,
            env.met.temperature_k,
            env.met.relative_humidity,
            env.day_of_year,
        )
        .slant_m;
    }

    // Predicted pseudorange, left-to-right; c*dt_sat is a single multiply.
    let p_hat = rho + b_m - C_M_S * dt_sat + iono_m + tropo_m;

    Some(SatModel {
        sat_rot_ecef_m: sat_rot,
        el_rad: g.el_rad,
        p_hat_m: p_hat,
        #[cfg(all(test, sidereon_repo_tests))]
        az_rad: g.az_rad,
        #[cfg(all(test, sidereon_repo_tests))]
        tau_s: tau,
        // Bit-identical to the loop's final `t_tx = t_rx - tau` (same operands).
        #[cfg(all(test, sidereon_repo_tests))]
        t_tx_j2000_s: env.t_rx_j2000_s - tau,
        #[cfg(all(test, sidereon_repo_tests))]
        sat_ecef_m: sat_pos,
        #[cfg(all(test, sidereon_repo_tests))]
        dt_sat_s: dt_sat,
        #[cfg(all(test, sidereon_repo_tests))]
        theta_rad: OMEGA_E_DOT_RAD_S * tau,
        #[cfg(all(test, sidereon_repo_tests))]
        rho_m: rho,
        #[cfg(all(test, sidereon_repo_tests))]
        iono_m,
        #[cfg(all(test, sidereon_repo_tests))]
        tropo_m,
    })
}

/// The frozen-geometry selection: used satellites (ascending id), rejected
/// satellites with reason, and the per-used-sat weight from the elevation at
/// the initial-guess geometry.
pub(crate) struct Selection {
    pub used: Vec<GnssSatelliteId>,
    pub rejected: Vec<RejectedSat>,
    /// `weight` per used satellite, index-aligned to `used`.
    pub weights: Vec<f64>,
}

pub(crate) fn select_sats(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    model: SppModelRecipe,
) -> Selection {
    let rx0 = [
        inputs.initial_guess[0],
        inputs.initial_guess[1],
        inputs.initial_guess[2],
    ];
    let b0 = inputs.initial_guess[3];

    // Ascending satellite-id order, never observation order.
    let mut obs: Vec<&Observation> = inputs.observations.iter().collect();
    obs.sort_by_key(|o| o.satellite_id);

    let mut used = Vec::new();
    let mut rejected = Vec::new();
    let mut weights = Vec::new();

    let env = SatModelEnv {
        eph,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
        day_of_year: inputs.day_of_year,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &inputs.glonass_channels,
        model,
    };
    for ob in obs {
        let model = sat_model(
            &env,
            ob.satellite_id,
            rx0,
            b0,
            ob.pseudorange_m,
            ionosphere_for(ob.satellite_id.system, inputs),
        );
        let Some(model) = model else {
            rejected.push(RejectedSat {
                satellite_id: ob.satellite_id,
                reason: RejectionReason::NoEphemeris,
            });
            continue;
        };
        if model.el_rad < ELEVATION_MASK_RAD {
            rejected.push(RejectedSat {
                satellite_id: ob.satellite_id,
                reason: RejectionReason::LowElevation,
            });
            continue;
        }
        let sin_el = model.el_rad.sin();
        let weight = (sin_el * sin_el) / (SIGMA0_M * SIGMA0_M);
        used.push(ob.satellite_id);
        weights.push(weight);
    }

    Selection {
        used,
        rejected,
        weights,
    }
}

/// The distinct GNSS present in `used`, in ascending system order.
///
/// The receiver-clock part of the state has one entry per system, each the
/// *absolute* receiver clock for that system (not a bias); the first is the
/// reference clock and a system's inter-system bias is its clock minus that
/// reference. For a single-system solve this is one element and the state is the
/// classic `[x, y, z, b]`.
pub(crate) fn clock_systems(used: &[GnssSatelliteId]) -> Vec<GnssSystem> {
    let mut systems: Vec<GnssSystem> = used.iter().map(|s| s.system).collect();
    systems.sort_unstable();
    systems.dedup();
    systems
}

/// The unweighted residual vector `P_meas - P_hat` at state `x`, in `used` order.
///
/// The state is `[x, y, z, clk_0, clk_1, ...]` where `clk_i` is the absolute
/// receiver clock for the i-th system returned by [`clock_systems`] (in meters).
/// Each satellite's residual uses its own system's clock, so a multi-GNSS set is
/// solved with one absolute receiver clock per system (a system's inter-system
/// bias is its clock minus the reference `clk_0`). A single-system set reduces to
/// `[x, y, z, b]` and `clk_0 = x[3]`.
///
/// Returns `Err(satellite)` if a used satellite has no observation or no usable
/// ephemeris at `x` (the frozen used set is fixed, but a finite-difference probe
/// could in principle reach an epoch off the ephemeris coverage). The caller
/// turns that into an [`SppError`] rather than panicking.
pub(crate) fn residual_unweighted(
    eph: &dyn EphemerisSource,
    used: &[GnssSatelliteId],
    obs_by_id: &[(GnssSatelliteId, f64)],
    x: &[f64],
    inputs: &SolveInputs,
    model: SppModelRecipe,
) -> Result<Vec<f64>, GnssSatelliteId> {
    let rx = [x[0], x[1], x[2]];
    let systems = clock_systems(used);
    let env = SatModelEnv {
        eph,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
        day_of_year: inputs.day_of_year,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &inputs.glonass_channels,
        model,
    };
    let mut out = Vec::with_capacity(used.len());
    for &sat in used {
        let p_meas = obs_by_id
            .iter()
            .find(|(id, _)| *id == sat)
            .map(|(_, p)| *p)
            .ok_or(sat)?;
        // The clock for this satellite's system (index 0 = reference clock).
        let sys_idx = systems.iter().position(|s| *s == sat.system).unwrap_or(0);
        let b = x[3 + sys_idx];
        let m =
            sat_model(&env, sat, rx, b, p_meas, ionosphere_for(sat.system, inputs)).ok_or(sat)?;
        out.push(p_meas - m.p_hat_m);
    }
    Ok(out)
}

/// Run the SPP solve from synthesized/measured pseudoranges.
///
/// Uses the core trust-region weighted least-squares solver over the
/// `sqrt(w) * (P_meas - P_hat)` residual. The converged position/clock is a
/// sub-micron solver-agreement result (the linear-algebra step is not
/// bit-reproducible across BLAS builds), not a 0-ULP claim. The residual /
/// Jacobian substrate evaluated at recorded states is the 0-ULP target and is
/// exercised by the trace-replay parity test, not by this entry point.
///
/// This is the reference SPP entry point: it runs the legacy
/// [`SolverRecipe::NalgebraTrfLegacy`] trust-region factorization, so its
/// existing goldens are unchanged. [`solve_with_solver`] selects the owned
/// deterministic kernel.
pub fn solve(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
) -> Result<ReceiverSolution, SppError> {
    validate_solve_inputs(inputs)?;
    solve_inner(
        eph,
        inputs,
        with_geodetic,
        SppModelRecipe::reference(),
        TrustRegionSolve::NalgebraLu,
    )
}

/// SPP's trust-region stage recognizes the owned deterministic solver
/// ([`SolverRecipe::OwnedDeterministicTrf`]), which owns the dense subproblem
/// factorization with a fixed reduction order and its own frozen-bits golden;
/// every other recipe selects the legacy nalgebra LU path that [`solve`] uses.
/// The other [`SolverRecipe`] variants name other strategies' linear-solve
/// stages (RTK first-tie, PPP last-tie, host LAPACK) and are not SPP
/// trust-region solvers.
const fn trust_region_solve(solver: SolverRecipe) -> TrustRegionSolve {
    match solver {
        SolverRecipe::OwnedDeterministicTrf => TrustRegionSolve::OwnedGaussianFirstTie,
        _ => TrustRegionSolve::NalgebraLu,
    }
}

/// SPP solve with an explicit [`SolverRecipe`] for the trust-region stage.
///
/// Selecting [`SolverRecipe::NalgebraTrfLegacy`] is bit-identical to [`solve`].
/// [`SolverRecipe::OwnedDeterministicTrf`] swaps in the owned deterministic
/// Gaussian-elimination factorization for the dense trust-region subproblem (no
/// nalgebra LU, no black-box BLAS in that solve), pinned to its own frozen-bits
/// golden; all other model stages are unchanged. The owned kernel owns ONLY the
/// subproblem factorization: the normal-matrix / gradient / norm reductions that
/// build the subproblem still go through nalgebra's CPU-dispatched dense
/// algebra, so the cross-platform bit guarantee is scoped to the factorization
/// (the converged bits are this build's reproducible output, not a portable
/// constant).
pub fn solve_with_solver(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    solver: SolverRecipe,
) -> Result<ReceiverSolution, SppError> {
    validate_solve_inputs(inputs)?;
    solve_inner(
        eph,
        inputs,
        with_geodetic,
        SppModelRecipe::reference(),
        trust_region_solve(solver),
    )
}

fn solve_inner(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    model: SppModelRecipe,
    linear_solve: TrustRegionSolve,
) -> Result<ReceiverSolution, SppError> {
    // One pseudorange per satellite. Reject duplicates deterministically (by
    // the smallest repeated id) so the result can never depend on observation
    // order and the parameter-count check below (`sel.used.len() < n_params`,
    // where `n_params = 3 + n_clocks`) counts distinct satellites.
    let mut ids: Vec<GnssSatelliteId> =
        inputs.observations.iter().map(|o| o.satellite_id).collect();
    ids.sort_unstable();
    if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
        return Err(SppError::DuplicateObservation { satellite: w[0] });
    }

    // The broadcast Klobuchar delay is computed on L1 and scaled to each
    // satellite's carrier by `(f_L1 / f)^2`. GPS L1, Galileo E1, and BeiDou B1I
    // have fixed carriers; GLONASS is FDMA, so its carrier is resolved per
    // satellite from `glonass_channels`. A satellite whose carrier cannot be
    // resolved (a GLONASS observation with no channel in the map, or a channel
    // outside the valid `[-7, +6]` FDMA range) cannot be scaled, so reject an
    // ionosphere-corrected solve that includes it rather
    // than apply an undefined correction. This runs before selection so the
    // model is never evaluated for it (`select_sats` would otherwise call
    // `sat_model` with the correction for every observation).
    if inputs.corrections.ionosphere {
        if let Some(sat) = ids
            .iter()
            .find(|s| spp_iono_frequency_hz(**s, &inputs.glonass_channels).is_none())
        {
            return Err(SppError::IonosphereUnsupported { satellite: *sat });
        }
    }

    let sel = select_sats(eph, inputs, model);

    // One receiver-clock parameter per distinct GNSS (a reference clock plus an
    // inter-system bias for each additional system), so the state has
    // `3 + n_systems` parameters and needs at least that many usable satellites.
    // Floor the clock count at one: the minimum solve is the four-parameter
    // single-system form even when no satellite survives selection.
    let systems = clock_systems(&sel.used);
    let n_clocks = systems.len();
    // SPP's weighted-residual rows feed the trust-region solver, which owns the
    // normal-equation factorization (NormalRecipe::SppWeightedResidualFiniteDifference
    // via SolverRecipe::NalgebraTrfLegacy); only the parameter stack is named here.
    let n_params = ParameterLayout::spp(n_clocks.max(1)).dim();
    if sel.used.len() < n_params {
        return Err(SppError::TooFewSatellites {
            used: sel.used.len(),
            required: n_params,
        });
    }

    let obs_by_id: Vec<(GnssSatelliteId, f64)> = inputs
        .observations
        .iter()
        .map(|o| (o.satellite_id, o.pseudorange_m))
        .collect();

    let used = sel.used.clone();
    let inputs_ref = inputs.clone();
    let obs_ref = obs_by_id.clone();
    let eph_ref = eph;
    let n_used = used.len();

    // The least-squares solver's residual closure cannot return an error, so an
    // ephemeris loss during a probe is recorded here and surfaced as an
    // SppError after the solve (rather than panicking inside the closure).
    let lost = std::rc::Rc::new(std::cell::Cell::new(None::<GnssSatelliteId>));
    let lost_in = lost.clone();
    let residual = move |x: &DVector<f64>| -> DVector<f64> {
        match residual_unweighted(eph_ref, &used, &obs_ref, x.as_slice(), &inputs_ref, model) {
            Ok(r) => DVector::from_vec(r),
            Err(sat) => {
                lost_in.set(Some(sat));
                DVector::from_vec(vec![0.0; n_used])
            }
        }
    };

    // Extend the 4-element initial guess `[x, y, z, b_ref]` with a zero starting
    // value for each additional system's inter-system bias.
    let mut x0v = inputs.initial_guess.to_vec();
    x0v.extend(std::iter::repeat_n(0.0, n_clocks - 1));
    let x0 = DVector::from_vec(x0v);
    // Agreement-track stopping thresholds (see the SPP_SOLVER_* constants).
    let opts = SolveOptions {
        gtol: SPP_SOLVER_GTOL,
        ftol: SPP_SOLVER_FTOL,
        xtol: SPP_SOLVER_XTOL,
        max_nfev: SPP_SOLVER_MAX_NFEV,
    };

    // The static elevation weights (base weights), index-aligned to `sel.used`.
    let base_weights = DVector::from_row_slice(&sel.weights);

    // The warm-start solve uses the base elevation weights exactly. On the
    // static path (`robust == None`) this is the literal current sequence: a
    // single `with_weights(residual, x0, base_weights)` solve and nothing else,
    // so the byte output is unchanged. On the robust path it seeds the outer
    // loop.
    //
    // Check for an ephemeris loss recorded by the residual closure BEFORE
    // propagating a solver error: a lost satellite zeroes its residual row,
    // which can itself make the Jacobian singular, and EphemerisLost is the
    // more specific, actionable cause.
    let problem = LeastSquaresProblem::with_weights(&residual, x0, base_weights);
    let report_result = solve_trf_with(&problem, &opts, linear_solve);
    if let Some(satellite) = lost.get() {
        return Err(SppError::EphemerisLost { satellite });
    }
    let mut report = report_result?;

    let mut outer_iterations = 0usize;
    let mut final_robust_scale_m: Option<f64> = None;

    // Outer Huber/IRLS reweighting loop, ONLY on the robust path. Each iteration
    // recomputes the unweighted post-fit residuals at the current converged
    // state, derives a floored MAD scale, builds the effective weight vector
    // `base_elevation_weight * huber(r_i / s)` index-aligned to `sel.used`,
    // rebuilds the problem warm-started at the previous state, and re-solves. It
    // stops when the position step drops below `outer_tol_m` or the reweighted
    // solve budget left after the warm start is hit (recording
    // `converged = false` if the inner solve itself did not converge on the
    // final pass).
    if let Some(rc) = inputs.robust {
        for _ in 0..rc.max_outer.saturating_sub(1) {
            if lost.get().is_some() {
                break;
            }
            // Unweighted post-fit residuals at the current state, in used order.
            let post = match residual_unweighted(
                eph,
                &sel.used,
                &obs_by_id,
                report.x.as_slice(),
                inputs,
                model,
            ) {
                Ok(r) => r,
                Err(satellite) => return Err(SppError::EphemerisLost { satellite }),
            };
            let scale = mad_scale(&post, rc.scale_floor_m).map_err(map_robust_error)?;
            // Effective weight per used sat: base elevation weight times the
            // Huber multiplier of the scaled residual.
            let eff: Vec<f64> = post
                .iter()
                .zip(sel.weights.iter())
                .map(|(&r, &bw)| bw * huber_weight(r / scale, rc.huber_k))
                .collect();
            let eff_w = DVector::from_row_slice(&eff);
            let x_prev = report.x.clone();
            let problem = LeastSquaresProblem::with_weights(&residual, x_prev.clone(), eff_w);
            let next = solve_trf_with(&problem, &opts, linear_solve);
            if let Some(satellite) = lost.get() {
                return Err(SppError::EphemerisLost { satellite });
            }
            report = next?;
            outer_iterations += 1;
            final_robust_scale_m = Some(scale);
            // Position L2 step between successive outer solves.
            let dpos = ((report.x[0] - x_prev[0]).powi(2)
                + (report.x[1] - x_prev[1]).powi(2)
                + (report.x[2] - x_prev[2]).powi(2))
            .sqrt();
            if dpos < rc.outer_tol_m {
                break;
            }
        }
    }

    let xs = &report.x;
    let position = ItrfPositionM::new(xs[0], xs[1], xs[2]).expect("valid ITRF position");
    let rx_clock_s = xs[3] / C_M_S;
    // One receiver clock (seconds) per system, in the same order as the state's
    // clock parameters. The first equals `rx_clock_s` (the reference system).
    let system_clocks_s: Vec<(GnssSystem, f64)> = systems
        .iter()
        .enumerate()
        .map(|(i, &sys)| (sys, xs[3 + i] / C_M_S))
        .collect();
    let geodetic = if with_geodetic {
        Some(geodetic_from_ecef(model.frame, [xs[0], xs[1], xs[2]]))
    } else {
        None
    };

    // Post-fit unweighted residuals in used order.
    let residuals_m = residual_unweighted(eph, &sel.used, &obs_by_id, xs.as_slice(), inputs, model)
        .map_err(|satellite| SppError::EphemerisLost { satellite })?;

    // DOP from the converged geometry: line-of-sight unit vectors to the
    // Sagnac-rotated satellite positions, with the frozen weights. A
    // single-system solve uses the 0-ULP four-state cofactor inverse; a
    // multi-system solve uses the general (3 + n_systems) inverse with one clock
    // column per GNSS (a deterministic geometry diagnostic, not a 0-ULP target).
    // The receiver-clock argument does not affect the line of sight, so the
    // reference clock is passed for every satellite.
    let rx_ecef = [xs[0], xs[1], xs[2]];
    let geo = geodetic_from_ecef(model.frame, [xs[0], xs[1], xs[2]]);
    let mut los = Vec::with_capacity(sel.used.len());
    let mut clock_index = Vec::with_capacity(sel.used.len());
    let env = SatModelEnv {
        eph,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
        day_of_year: inputs.day_of_year,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &inputs.glonass_channels,
        model,
    };
    for &sat in &sel.used {
        let p_meas = obs_by_id
            .iter()
            .find(|(id, _)| *id == sat)
            .map(|(_, p)| *p)
            .ok_or(SppError::EphemerisLost { satellite: sat })?;
        let m = sat_model(
            &env,
            sat,
            rx_ecef,
            xs[3],
            p_meas,
            ionosphere_for(sat.system, inputs),
        )
        .ok_or(SppError::EphemerisLost { satellite: sat })?;
        let dx = m.sat_rot_ecef_m[0] - rx_ecef[0];
        let dy = m.sat_rot_ecef_m[1] - rx_ecef[1];
        let dz = m.sat_rot_ecef_m[2] - rx_ecef[2];
        let n = (dx * dx + dy * dy + dz * dz).sqrt();
        los.push(LineOfSight::new(dx / n, dy / n, dz / n));
        let idx = systems.iter().position(|s| *s == sat.system).unwrap_or(0);
        clock_index.push(idx);
    }
    // `systems` is the clock-column ordering: `clock_index[k] ==
    // systems.position(sat.system)`, so `systems[c]` owns clock column `c` (the
    // same ordering `system_clocks_s` uses). The multi-system path is handed
    // that mapping and returns `Dop::system_tdops` already GNSS-tagged; the
    // single-system 0-ULP `dop` carries no constellation identity, so tag its
    // lone clock here with the one system in the solve.
    let dop_result = if n_clocks == 1 {
        dop(&los, &sel.weights, geo).ok().map(|mut d| {
            d.system_tdops = vec![(systems[0], d.tdop)];
            d
        })
    } else {
        dop_multi(&los, &clock_index, &systems, n_clocks, &sel.weights, geo).ok()
    };
    // The solution's per-system TDOPs come straight from the now-tagged
    // `Dop::system_tdops`; empty when the converged geometry is rank-deficient.
    let system_tdops: Vec<(GnssSystem, f64)> = dop_result
        .as_ref()
        .map(|d| d.system_tdops.clone())
        .unwrap_or_default();

    let converged = matches!(
        report.status,
        Status::GradientTolerance | Status::CostTolerance | Status::StepTolerance
    );
    let metadata_used_count = sel.used.len();
    let metadata_redundancy = redundancy(&systems, metadata_used_count);

    Ok(ReceiverSolution {
        position,
        geodetic,
        rx_clock_s,
        system_clocks_s,
        dop: dop_result,
        system_tdops,
        residuals_m,
        used_sats: sel.used,
        rejected_sats: sel.rejected,
        metadata: SolutionMetadata {
            iterations: report.iterations,
            converged,
            status: report.status,
            ionosphere_applied: inputs.corrections.ionosphere,
            troposphere_applied: inputs.corrections.troposphere,
            outer_iterations,
            final_robust_scale_m,
            used_count: metadata_used_count,
            systems,
            redundancy: metadata_redundancy,
            raim_checkable: metadata_redundancy >= 1,
        },
    })
}

/// Run SPP under the public API's language-independent validation/orchestration
/// policy.
///
/// Thin compatibility wrapper over the runtime strategy selector
/// ([`crate::estimation::strategies::estimate`]): it drives the shared
/// per-technique implementation [`run`] under the SPP reference strategy, which
/// resolves to the SPP reference recipe. The reference strategy always yields an
/// SPP solution or an SPP error, so the result is bit-identical to the recipe
/// driving [`run`] directly.
pub fn solve_with_policy(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Result<ReceiverSolution, SolvePolicyError> {
    use crate::estimation::recipe::StrategyId;
    use crate::estimation::strategies::{
        estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput,
    };
    match estimate(
        EstimateInput::Spp {
            eph,
            inputs,
            with_geodetic,
            policy,
        },
        EstimateOptions::new(StrategyId::spp_reference()),
    ) {
        Ok(EstimateOutput::Spp(solution)) => Ok(*solution),
        Err(EstimateError::Spp(error)) => Err(error),
        Ok(_) | Err(_) => {
            unreachable!("the SPP reference strategy yields an SPP solution or an SPP error")
        }
    }
}

/// Solve a batch of independent SPP epochs against a shared ephemeris, serially.
///
/// Element `i` of the result is [`solve_with_policy`] applied to `epochs[i]`,
/// with the shared `eph`, `with_geodetic`, and `policy` (every epoch is one
/// receive instant's [`SolveInputs`]; the receiver's clock and position are
/// re-estimated per epoch, so the epochs are independent). The first solve error
/// for an epoch becomes that element's `Err`. This is the single-threaded
/// reference the parallel [`solve_spp_batch_parallel`] is proven bit-identical
/// against.
pub fn solve_spp_batch_serial(
    eph: &dyn EphemerisSource,
    epochs: &[SolveInputs],
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Vec<Result<ReceiverSolution, SolvePolicyError>> {
    epochs
        .iter()
        .map(|inputs| solve_with_policy(eph, inputs, with_geodetic, policy))
        .collect()
}

/// Solve a batch of independent SPP epochs against a shared ephemeris, fanning
/// the independent per-epoch solves across a rayon thread pool.
///
/// Each epoch is solved by the same serial [`solve_with_policy`] kernel and the
/// indexed parallel collect preserves input order, so element `i` is
/// byte-for-byte identical to element `i` of [`solve_spp_batch_serial`]: the
/// epochs share only the immutable `eph`/`policy`, there is no cross-epoch state
/// and no reduction, and a single solve is unchanged. The work is embarrassingly
/// parallel (epochs are independent), so throughput scales with cores while
/// every value stays bit-exact. `eph` must be [`Sync`] to be shared across the
/// pool.
pub fn solve_spp_batch_parallel(
    eph: &(dyn EphemerisSource + Sync),
    epochs: &[SolveInputs],
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Vec<Result<ReceiverSolution, SolvePolicyError>> {
    use rayon::prelude::*;
    epochs
        .par_iter()
        .map(|inputs| solve_with_policy(eph, inputs, with_geodetic, policy))
        .collect()
}

/// Drive SPP from a resolved [`EstimationRecipe`]: the shared per-technique
/// implementation that [`crate::estimation::strategies::estimate`] dispatches to.
/// The recipe's range/sagnac/frame stages select the SPP measurement-model
/// operation order ([`SppModelRecipe`], threaded into [`sat_model`]) and its
/// [`SolverRecipe`] selects the trust-region factorization; the public
/// validation/orchestration policy is applied here. For the SPP reference recipe
/// every selected order equals the value the legacy [`solve`] path hard-coded, so
/// this is bit-identical to it.
pub(crate) fn run(
    recipe: &EstimationRecipe,
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    policy: SolvePolicy,
) -> Result<ReceiverSolution, SolvePolicyError> {
    validate_solve_inputs(inputs)?;
    let model = SppModelRecipe::from_recipe(recipe);
    match policy.coarse_search_seeds {
        Some(seed_count) => solve_coarse(
            eph,
            inputs,
            with_geodetic,
            policy,
            seed_count,
            model,
            recipe.solver,
        ),
        None => solve_validated(
            eph,
            inputs,
            with_geodetic,
            policy.validation,
            model,
            recipe.solver,
        ),
    }
}

fn solve_validated(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    validation: SolutionValidationOptions,
    model: SppModelRecipe,
    solver: SolverRecipe,
) -> Result<ReceiverSolution, SolvePolicyError> {
    let solution = solve_inner(
        eph,
        inputs,
        with_geodetic,
        model,
        trust_region_solve(solver),
    )?;
    validate_receiver_solution(&solution, validation)?;
    Ok(solution)
}

fn solve_coarse(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    policy: SolvePolicy,
    seed_count: usize,
    model: SppModelRecipe,
    solver: SolverRecipe,
) -> Result<ReceiverSolution, SolvePolicyError> {
    let mut candidates = Vec::new();
    let mut last_error = SolvePolicyError::NoCoarseSolution;

    for seed in std::iter::once(inputs.initial_guess).chain(coarse_seeds(seed_count)) {
        let mut seeded = inputs.clone();
        seeded.initial_guess = seed;
        match solve_validated(
            eph,
            &seeded,
            with_geodetic,
            policy.validation,
            model,
            solver,
        ) {
            Ok(solution) => candidates.push(solution),
            Err(error) => last_error = error,
        }
    }

    select_coarse_candidate(&candidates)
        .cloned()
        .ok_or(last_error)
}

fn coarse_seeds(n: usize) -> Vec<[f64; 4]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = golden * i as f64;
            [
                MEAN_EARTH_RADIUS_M * r * theta.cos(),
                MEAN_EARTH_RADIUS_M * r * theta.sin(),
                MEAN_EARTH_RADIUS_M * z,
                0.0,
            ]
        })
        .collect()
}

fn select_coarse_candidate(candidates: &[ReceiverSolution]) -> Option<&ReceiverSolution> {
    candidates
        .iter()
        .filter(|solution| solution.metadata.converged && solution.metadata.redundancy >= 1)
        .min_by(|a, b| compare_coarse_candidates(a, b))
}

fn compare_coarse_candidates(a: &ReceiverSolution, b: &ReceiverSolution) -> core::cmp::Ordering {
    b.used_sats
        .len()
        .cmp(&a.used_sats.len())
        .then_with(|| residual_rms(&a.residuals_m).total_cmp(&residual_rms(&b.residuals_m)))
        .then_with(|| candidate_gdop(a).total_cmp(&candidate_gdop(b)))
}

fn candidate_gdop(solution: &ReceiverSolution) -> f64 {
    solution
        .dop
        .as_ref()
        .map(|dop| dop.gdop)
        .unwrap_or(f64::INFINITY)
}

/// Root-mean-square of post-fit pseudorange residuals (0.0 when empty).
///
/// Exposed so language bindings can delegate residual-RMS reporting to the core
/// rather than recomputing the formula. [Claude]
pub fn residual_rms(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let sum_sq = residuals.iter().map(|r| r * r).sum::<f64>();
    (sum_sq / residuals.len() as f64).sqrt()
}

fn redundancy(systems: &[GnssSystem], used_count: usize) -> isize {
    used_count as isize - (3 + systems.len() as isize)
}

fn validate_solve_inputs(inputs: &SolveInputs) -> Result<(), SppError> {
    validate::finite(inputs.t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    validate::second_of_day(inputs.t_rx_second_of_day_s, "t_rx_second_of_day_s")
        .map_err(map_input_error)?;
    validate::finite_in_range_exclusive_upper(inputs.day_of_year, 1.0, 367.0, "day_of_year")
        .map_err(map_input_error)?;
    validate::finite_slice(&inputs.initial_guess, "initial_guess").map_err(map_input_error)?;
    validate_klobuchar(&inputs.klobuchar, "klobuchar")?;
    if let Some(klobuchar) = &inputs.beidou_klobuchar {
        validate_klobuchar(klobuchar, "beidou_klobuchar")?;
    }
    if let Some(nequick) = &inputs.galileo_nequick {
        validate_galileo_nequick(nequick)?;
    }
    if inputs.corrections.troposphere {
        validate_met(&inputs.met)?;
    }
    validate_observations(&inputs.observations)?;
    if let Some(robust) = inputs.robust {
        if robust.max_outer == 0 {
            return Err(SppError::InvalidInput {
                field: "robust.max_outer",
                kind: SppInputErrorKind::NotPositive,
            });
        }
        validate::finite_positive(robust.huber_k, "robust.huber_k").map_err(map_input_error)?;
        validate::finite_positive(robust.scale_floor_m, "robust.scale_floor_m")
            .map_err(map_input_error)?;
        validate::finite_positive(robust.outer_tol_m, "robust.outer_tol_m")
            .map_err(map_input_error)?;
    }
    Ok(())
}

fn validate_klobuchar(coeffs: &KlobucharCoeffs, field: &'static str) -> Result<(), SppError> {
    validate::finite_slice(&coeffs.alpha, field).map_err(map_input_error)?;
    validate::finite_slice(&coeffs.beta, field).map_err(map_input_error)
}

fn validate_galileo_nequick(coeffs: &GalileoNequickCoeffs) -> Result<(), SppError> {
    validate::finite(coeffs.ai0, "galileo_nequick").map_err(map_input_error)?;
    validate::finite(coeffs.ai1, "galileo_nequick").map_err(map_input_error)?;
    validate::finite(coeffs.ai2, "galileo_nequick").map_err(map_input_error)?;
    Ok(())
}

fn validate_met(met: &SurfaceMet) -> Result<(), SppError> {
    validate::finite_positive(met.pressure_hpa, "met.pressure_hpa").map_err(map_input_error)?;
    validate::finite_positive(met.temperature_k, "met.temperature_k").map_err(map_input_error)?;
    validate::fraction(met.relative_humidity, "met.relative_humidity").map_err(map_input_error)?;
    Ok(())
}

fn validate_observations(observations: &[Observation]) -> Result<(), SppError> {
    for obs in observations {
        validate::finite_positive(obs.pseudorange_m, "observation.pseudorange_m")
            .map_err(map_input_error)?;
    }
    Ok(())
}

fn map_input_error(error: validate::FieldError) -> SppError {
    SppError::InvalidInput {
        field: error.field(),
        kind: SppInputErrorKind::from(&error),
    }
}

fn map_robust_error(error: RobustError) -> SppError {
    let field = match error.field() {
        "scale_floor" => "robust.scale_floor_m",
        "residuals" | "values" => "robust.residuals",
        other => other,
    };
    let kind = match error.reason() {
        "not finite" => SppInputErrorKind::NonFinite,
        "not positive" => SppInputErrorKind::NotPositive,
        "negative" => SppInputErrorKind::Negative,
        "out of range" => SppInputErrorKind::OutOfRange,
        _ => SppInputErrorKind::OutOfRange,
    };
    SppError::InvalidInput { field, kind }
}

/// The core km/deg geodetic recipe, for the boundary cross-check against the
/// meters-native helper.
#[cfg(all(test, sidereon_repo_tests))]
pub(crate) mod test_support {
    use super::*;

    pub fn geodetic_from_ecef_m_for_test(x_m: f64, y_m: f64, z_m: f64) -> Wgs84Geodetic {
        geodetic_from_ecef(FrameRecipe::SppSkyfieldAuThreeIter, [x_m, y_m, z_m])
    }

    pub fn sat_model_for_test(
        env: &SatModelEnv,
        sat: GnssSatelliteId,
        rx: [f64; 3],
        b_m: f64,
        p_meas: f64,
        klobuchar: &KlobucharCoeffs,
    ) -> Option<SatModel> {
        sat_model(
            env,
            sat,
            rx,
            b_m,
            p_meas,
            SppIonosphere::Klobuchar(*klobuchar),
        )
    }

    pub fn sat_model_with_ionosphere_for_test(
        env: &SatModelEnv,
        sat: GnssSatelliteId,
        rx: [f64; 3],
        b_m: f64,
        p_meas: f64,
        ionosphere: SppIonosphere,
    ) -> Option<SatModel> {
        sat_model(env, sat, rx, b_m, p_meas, ionosphere)
    }

    /// The core km/deg geodetic recipe (Skyfield AU-internal), returning the
    /// public `(lat_deg, lon_deg, alt_km)`, for the boundary cross-check.
    pub fn itrs_to_geodetic_core_km(x_km: f64, y_km: f64, z_km: f64) -> (f64, f64, f64) {
        crate::astro::frames::transforms::itrs_to_geodetic_compute(x_km, y_km, z_km)
            .expect("valid ITRS coordinates")
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
