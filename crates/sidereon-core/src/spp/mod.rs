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
//! The per-satellite predicted pseudorange is built in a pinned operation order,
//! as RTKLIB `pntpos` builds it: the transmission epoch is placed from the
//! measured pseudorange as RTKLIB `satposs` places it (`t_rx - P / c`, less the
//! satellite clock read there), the satellite ephemeris is read at that epoch,
//! the geometric range is RTKLIB `geodist` (the Euclidean range from the
//! unrotated position plus the first-order Sagnac term), the line-of-sight
//! azimuth/elevation follow, then the ionosphere and troposphere delays are added
//! to the predicted range left-to-right. The residual the solver sees is
//! `sqrt(w) * (P_meas - P_hat)` with the elevation weight `w = sin^2(el) / sigma0^2`.
//!
//! The satellite selection, the elevation mask and the weights are evaluated at
//! the current iterate, as RTKLIB `estpos` re-runs `rescode` at every iteration,
//! and the ionosphere and troposphere delays at every state the solve reaches.
//! An iterate with a new selection runs the trust-region solve over that
//! selection with those weights; an iterate whose selection is the last one
//! takes the step RTKLIB `estpos` takes, `dx = (H^T W H)^-1 H^T W v` over the
//! design `H = [-e, 1]` with that iterate's weights and residuals. The solve ends
//! with the first step below [`SELECTION_STEP_TOL_M`], RTKLIB's
//! `norm(dx) < 1E-4`, and keeps it, so the position is a fixed point of the
//! RTKLIB iteration; one that has not ended after [`MAX_SELECTION_PASSES`]
//! solves and steps fails with [`SppError::SelectionUnsettled`]. The reported
//! satellites, rejections and weights are those of the last selection, and the
//! residuals, DOP and covariance are evaluated for them at the reported position.
//! A receiver RTKLIB places at the geocentre sees every satellite at the zenith,
//! so a solve from an all-zero initial guess keeps every satellite with an
//! ephemeris on its first pass and masks them from the second.
//!
//! The geometric/clock/correction substrate and its 2-point finite-difference
//! Jacobian are arithmetic over the libm-bound model functions and are a
//! bit-exact (0-ULP) parity target against the reference recipe, replayed with the
//! geometric light-time model that recipe places its transmission epochs with. The
//! converged position is produced by the trust-region least-squares solver in the
//! `sidereon-core` solver core, whose linear-algebra step is not bit-reproducible
//! across BLAS builds, and the Gauss-Newton steps above; the converged solution is
//! therefore a sub-micron solver-agreement result, not a 0-ULP claim.
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
    self, singular_value_diagnostics, LeastSquaresProblem, SolveOptions, Status, TrustRegionSolve,
};
use crate::astro::math::linear::invert_symmetric_pd;
use crate::astro::math::portable;
use crate::geometry_quality::{classify, GeometryQuality, GeometryQualityThresholds};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

mod config;
mod fallback;
mod source;
use crate::astro::math::robust::{huber_weight, mad_scale, RobustError};
pub use config::{
    DEFAULT_HUBER_K, DEFAULT_ROBUST_MAX_OUTER, DEFAULT_ROBUST_OUTER_TOL_M,
    DEFAULT_ROBUST_SCALE_FLOOR_M, ELEVATION_MASK_RAD, MAX_SELECTION_PASSES, SELECTION_STEP_TOL_M,
    SIGMA0_M, TRANSMIT_TIME_ITERATIONS,
};
pub use fallback::{
    solve_broadcast, solve_with_fallback, BroadcastReason, FallbackError, FixSource,
    SourcedSolution,
};
use source::TransmitStateMemo;
pub use source::{ClockRelativity, EphemerisSource, PositionClock, PositionClockGroupDelay};
pub(crate) use source::{Ut1Tracked, Ut1TrackedSource};

pub use crate::constants::{C_M_S, F_L1_HZ, OMEGA_E_DOT_RAD_S};
use crate::dop::{dop, dop_multi, Dop, LineOfSight, PositionCovariance};
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
use crate::observables::ObservableEphemerisSource;
use crate::quality::{
    validate_receiver_solution, SolutionValidationError, SolutionValidationOptions,
};
use crate::sbas::SbasIonoGrid;
use crate::tropo::slant_components;
use crate::validate;
use crate::velocity::{
    self, VelocityError, VelocityObservable, VelocityObservation, VelocitySolution,
    VelocitySolveOptions,
};

/// The single-frequency carrier (Hz) the ionosphere correction is reported on
/// for a constellation with one fixed single-frequency carrier, or `None` for a
/// system that has none (GLONASS, whose FDMA carrier is per-satellite). GPS,
/// QZSS and SBAS L1 and Galileo E1 are all at [`F_L1_HZ`]; BeiDou uses B1I and
/// NavIC L5. Klobuchar and Galileo broadcast delays are reported on this
/// carrier. GLONASS is resolved per satellite by [`spp_iono_frequency_hz`] from
/// its FDMA channel instead.
pub(crate) const fn carrier_frequency_hz(system: GnssSystem) -> Option<f64> {
    match system {
        GnssSystem::Sbas => Some(F_L1_HZ),
        _ => frequencies::default_spp_frequency_hz(system),
    }
}

/// The carrier frequency (Hz) the broadcast ionosphere delay is scaled to for a
/// single satellite, or `None` if the satellite's system has no carrier the
/// model can resolve.
///
/// For the fixed-carrier systems (GPS, QZSS and SBAS L1, Galileo E1, BeiDou
/// B1I, NavIC L5) this is the system carrier from [`carrier_frequency_hz`]. GLONASS is FDMA, so its carrier
/// is per-satellite: it is resolved from `glonass_channels` (the broadcast /
/// observation FDMA channel `k` keyed by GLONASS slot number) as the G1
/// frequency `1602.0 MHz + k * 562.5 kHz`. A GLONASS satellite whose channel is
/// not in the map, or whose channel is outside the FDMA allocation
/// `[-7, +6]` ([`crate::rinex_nav::valid_glonass_frequency_channel`]; the RINEX
/// nav/obs readers keep a stated channel outside it), has no resolvable
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
use crate::constants::{MEAN_EARTH_RADIUS_M, WGS84_A_M, WGS84_F};
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

/// Why a satellite was excluded from the solve.
///
/// SPP selection tests a satellite in the order RTKLIB `rescode` does and
/// reports the first reason that applies: [`Self::NoEphemeris`], then
/// [`Self::LowElevation`], then [`Self::SbasIonoUncovered`], then
/// [`Self::IonosphereCarrierUnresolved`]. [`Self::SbasWithdrawn`] is not
/// reported by SPP selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// The SP3 product has no usable position or clock for the satellite at the
    /// transmit epoch.
    NoEphemeris,
    /// The satellite is below the elevation mask at the state the reported
    /// selection was made at.
    LowElevation,
    /// The bound augmentation source withdrew the satellite.
    SbasWithdrawn,
    /// The augmentation ionosphere grid does not cover the satellite line of sight.
    SbasIonoUncovered,
    /// The ionosphere correction was requested and the satellite has no
    /// resolvable carrier frequency, so the L1 delay cannot be scaled to it.
    /// GPS, QZSS, SBAS, Galileo, BeiDou and NavIC have fixed carriers and never
    /// land here. A GLONASS satellite does when it has no channel in
    /// [`SolveInputs::glonass_channels`], or when its channel is outside the
    /// `-7..=6` FDMA allocation (the `7` real IGS headers give the extended
    /// slot `R28`, say). The satellite is left out of the solve, as RTKLIB
    /// `rescode` leaves out a satellite whose `sat2freq` is zero, and the rest
    /// of the epoch is solved.
    IonosphereCarrierUnresolved,
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
    /// Number of accepted iterations: the trust-region iterations of every solve
    /// the selection passes and the robust reweighting ran, plus one for each
    /// RTKLIB least-squares step taken between them.
    pub iterations: usize,
    /// Whether the solve converged: it ended with [`Status::SelectionSettled`],
    /// not with a robust budget spent ([`Status::OuterBudgetExhausted`]) or a
    /// robust solve whose last trust-region solve spent its evaluations.
    pub converged: bool,
    /// How the solve ended: [`Status::SelectionSettled`] when its last
    /// least-squares step fell below [`SELECTION_STEP_TOL_M`] at a selection that
    /// held (and, on the robust path, its position and selection then settled);
    /// [`Status::OuterBudgetExhausted`] when the robust budget ran out first; the
    /// last trust-region solve's own status when that solve spent its evaluations.
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
    /// The first UT1 departure the ephemeris source accepted while producing
    /// a satellite state for this solve, under a permissive UT1 policy (for
    /// example an SSR source's centre-of-mass to antenna-phase-centre
    /// conversion outside the UT1 table). `None` when every state was
    /// produced inside UT1 coverage or did not read UT1.
    pub ut1_degraded: Option<crate::astro::time::DegradeReason>,
}

/// A receiver position/clock solution with its geometry diagnostics.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Receiver clock drift in seconds per second when a Doppler/range-rate
    /// velocity solve was run with this receiver position. Pseudorange-only
    /// solves leave this as `None`.
    pub rx_clock_drift_s_s: Option<f64>,
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
    /// Position covariance in square metres.
    ///
    /// `ecef_m2` is the ITRF/IGS ECEF covariance. `enu_m2` is the same block
    /// rotated into the local geodetic east-north-up frame at the solved
    /// receiver position.
    pub position_covariance: PositionCovariance,
    /// Post-fit residuals in meters, in `used_sats` order (unweighted
    /// `P_meas - P_hat`).
    pub residuals_m: Vec<f64>,
    /// The satellites that contributed to the solve, ascending id order.
    pub used_sats: Vec<GnssSatelliteId>,
    /// The excluded satellites, each with its reason.
    pub rejected_sats: Vec<RejectedSat>,
    /// Geometry observability and covariance-validation diagnostics for the
    /// converged design. Snapshot SPP has no propagated prior, so
    /// `ZeroRedundancy` marks unvalidated covariance bounds, `Weak` leaves large
    /// bounds unclamped, and `RankDeficient` is routed through [`SppError::Singular`]
    /// instead of returning a solution.
    pub geometry_quality: GeometryQuality,
    /// Iteration / convergence / model metadata.
    pub metadata: SolutionMetadata,
}

/// One Doppler row for an SPP-family receiver velocity solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopplerObservation {
    /// Satellite identifier.
    pub satellite_id: GnssSatelliteId,
    /// Doppler shift in hertz.
    pub doppler_hz: f64,
    /// Carrier frequency in hertz.
    pub carrier_hz: f64,
    /// Satellite clock drift in seconds per second.
    pub sat_clock_drift_s_s: f64,
}

/// Inputs for the SPP-family Doppler velocity solve.
#[derive(Debug, Clone, PartialEq)]
pub struct DopplerVelocityInputs {
    /// Doppler observations for one epoch.
    pub observations: Vec<DopplerObservation>,
    /// Receiver ECEF/ITRF position in metres.
    pub receiver_ecef_m: [f64; 3],
    /// Receive epoch, seconds since J2000.
    pub t_rx_j2000_s: f64,
    /// Apply fixed-point light-time correction in the geometry substrate.
    pub light_time: bool,
    /// Apply Earth-rotation Sagnac correction in the geometry substrate.
    pub sagnac: bool,
}

impl DopplerVelocityInputs {
    /// Build Doppler velocity inputs from a receiver position solution.
    pub fn from_receiver_solution(
        solution: &ReceiverSolution,
        observations: Vec<DopplerObservation>,
        t_rx_j2000_s: f64,
    ) -> Self {
        Self {
            observations,
            receiver_ecef_m: solution.position.as_array(),
            t_rx_j2000_s,
            light_time: true,
            sagnac: true,
        }
    }
}

/// Result from solving position and, when possible, Doppler velocity together.
#[derive(Debug, Clone)]
pub struct SppDopplerSolution {
    /// Receiver position solution. `rx_clock_drift_s_s` is populated when
    /// `velocity` is `Some`.
    pub receiver: ReceiverSolution,
    /// Solved ECEF velocity and clock drift. `None` when no Doppler rows were
    /// supplied or the velocity system was not solvable.
    pub velocity: Option<VelocitySolution>,
    /// Velocity-solve failure when Doppler rows were present but not solvable.
    pub velocity_error: Option<VelocityError>,
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
/// weighting: a warm start from the settled static solve (identical to the
/// static path), then re-solves that each take the selection at the current
/// state and weight it as `elevation_weight * huber(r_i / s)`, where `r_i` is the
/// unweighted residual at that state and `s` is a floored MAD scale. The loop
/// settles when the position moves less than `outer_tol_m` and the selection at
/// the new state is the one solved with; after `max_outer` total solves without
/// settling it ends with [`Status::OuterBudgetExhausted`] and has not converged. With
/// `robust = None` the solve is byte-identical to the static elevation-weighted
/// solve. `Default` matches the `DEFAULT_*` config constants.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
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
    /// Optional augmentation ionosphere grid.
    pub sbas_iono: Option<SbasIonoGrid>,
    /// GLONASS FDMA channel numbers keyed by GLONASS slot (PRN), from the
    /// broadcast nav `freq_channel` field or the observation header's
    /// `GLONASS SLOT / FRQ #` records. Used only to resolve the per-satellite
    /// GLONASS carrier for the ionosphere `(f_L1 / f_k)^2` scaling; an empty map
    /// is correct for any solve with no GLONASS observation and leaves every
    /// other constellation bit-identical. A GLONASS observation with the
    /// ionosphere correction requested but no channel here, or a channel outside
    /// the FDMA allocation, is excluded from the solve and reported with
    /// [`RejectionReason::IonosphereCarrierUnresolved`].
    pub glonass_channels: BTreeMap<u8, i8>,
    /// Surface meteorology (used iff `corrections.troposphere`).
    pub met: SurfaceMet,
    /// Opt-in Huber/IRLS robust reweighting. `None` (the default behavior)
    /// runs the static elevation-weighted solve byte-identically; `Some(_)`
    /// adds the outer reweighting loop described on [`RobustConfig`].
    pub robust: Option<RobustConfig>,
    /// Which code the pseudoranges are: single-frequency, which takes the broadcast
    /// group delay, or ionosphere-free, which takes none.
    pub pseudorange_code: PseudorangeCode,
}

/// Which code an SPP solve's pseudoranges are, which decides whether the broadcast
/// single-frequency group delay (TGD, BGD) applies. RTKLIB `pntpos` `prange` subtracts it
/// from a single-frequency pseudorange and applies none to the ionosphere-free
/// combination (`IONOOPT_IFLC`), whose clock the broadcast `satposs` clock already is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PseudorangeCode {
    /// A single-frequency code (L1 C/A, E1, B1I): the group delay applies.
    #[default]
    SingleFrequency,
    /// The ionosphere-free combination: no group delay applies.
    IonosphereFree,
}

impl Default for SolveInputs {
    fn default() -> Self {
        Self {
            observations: Vec::new(),
            t_rx_j2000_s: 0.0,
            t_rx_second_of_day_s: 0.0,
            day_of_year: 1.0,
            initial_guess: [0.0; 4],
            corrections: Corrections::NONE,
            klobuchar: KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
            beidou_klobuchar: None,
            galileo_nequick: None,
            sbas_iono: None,
            glonass_channels: BTreeMap::new(),
            met: SurfaceMet::default(),
            robust: None,
            pseudorange_code: PseudorangeCode::SingleFrequency,
        }
    }
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
    /// A selected satellite had no usable position/clock at a state reached
    /// during the solve. Returned instead of panicking; normally precluded by the
    /// selection step.
    EphemerisLost {
        /// The satellite whose ephemeris became unavailable during the solve.
        satellite: GnssSatelliteId,
    },
    /// The satellite selection did not settle: after
    /// [`MAX_SELECTION_PASSES`] passes, each trust-region solve over a new
    /// selection and each least-squares step counting as one, no step at a
    /// selection that held had fallen below [`SELECTION_STEP_TOL_M`]. RTKLIB `estpos` fails the epoch the same way
    /// after `MAXITR` iterations. A satellite that each solve moves back across
    /// the elevation mask never settles.
    SelectionUnsettled {
        /// The number of passes run.
        passes: usize,
    },
    /// The ephemeris source refused a satellite state because producing it
    /// reads UT1 outside the UT1 table under a strict UT1 policy. The solve
    /// fails rather than dropping that satellite.
    Ut1OutsideCoverage(crate::astro::time::DegradeReason),
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
            SppError::SelectionUnsettled { passes } => {
                write!(
                    f,
                    "the satellite selection did not settle in {passes} passes"
                )
            }
            SppError::Ut1OutsideCoverage(reason) => {
                write!(
                    f,
                    "the ephemeris source refused a satellite state: {reason}"
                )
            }
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
/// strategy's [`EstimationRecipe`]: the transmit-time range recipe, the Sagnac
/// recipe, and the receiver-frame (geodetic / az-el) recipe.
///
/// Threading these into [`sat_model`] is what makes SPP consume its
/// `recipe.range` / `recipe.sagnac` / `recipe.frame` rather than hard-coding a
/// single op-order. [`Self::reference`] is the SPP reference selection: the RTKLIB
/// transmit-time placement and range with the Skyfield geodetic frame.
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

    /// The SPP reference model selections (the [`EstimationRecipe::spp`]
    /// range/sagnac/frame stages).
    pub(crate) const fn reference() -> Self {
        Self::from_recipe(&EstimationRecipe::spp())
    }

    /// The geometric light-time model the external SPP references (the Python trace
    /// recipe, the Go fixture) were computed with: the transmission epoch iterated a
    /// fixed number of times from the receiver's time tag, and the closed-form Sagnac
    /// rotation. It misses the receiver clock offset. Only the repository's replay of
    /// those references selects it.
    #[cfg(feature = "test-replays")]
    pub(crate) const fn geometric_light_time_replay() -> Self {
        Self {
            range: RangeRecipe::SppMeasuredPseudorangeFixedIter,
            sagnac: SagnacRecipe::ClosedFormZRotation,
            frame: FrameRecipe::SppSkyfieldAuThreeIter,
        }
    }
}

/// Per-satellite model used by the solve path: the satellite position the range
/// is formed from (in the transmission-epoch frame under RTKLIB's first-order
/// Sagnac term, rotated into the reception-epoch frame under the closed-form
/// rotation), the topocentric az/el, and the predicted pseudorange.
///
/// The scenario simulator also reads the range, satellite-clock, ionosphere,
/// and troposphere intermediates to build its ground-truth term ledger. Test
/// builds additionally carry transmit-time and Sagnac details for the 0-ULP
/// trace-replay parity checks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SatModel {
    pub sat_rot_ecef_m: [f64; 3],
    pub el_rad: f64,
    pub p_hat_m: f64,
    pub dt_sat_s: f64,
    pub rho_m: f64,
    pub iono_m: f64,
    pub tropo_m: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub az_rad: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub tau_s: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub t_tx_j2000_s: f64,
    #[cfg(all(test, sidereon_repo_tests))]
    pub sat_ecef_m: [f64; 3],
    #[cfg(all(test, sidereon_repo_tests))]
    pub theta_rad: f64,
    /// Epoch the final satellite clock was evaluated at.
    #[cfg(all(test, sidereon_repo_tests))]
    pub clock_epoch_j2000_s: f64,
}

/// The broadcast ionosphere correction a satellite's system uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SppIonosphere<'a> {
    /// GPS/BeiDou Klobuchar alpha/beta model.
    Klobuchar(KlobucharCoeffs),
    /// Galileo NeQuick-G effective-ionisation coefficients.
    GalileoNequick(GalileoNequickCoeffs),
    /// Augmentation grid delay model.
    SbasGrid(&'a SbasIonoGrid),
}

/// The ionosphere coefficients a satellite's system uses: Galileo prefers its
/// `galileo_nequick` (`GAL`) set when present; BeiDou prefers its
/// `beidou_klobuchar` (`BDSA`/`BDSB`) set when present; all missing
/// constellation-specific sets fall back to the shared GPS Klobuchar values to
/// preserve existing callers.
pub(crate) fn ionosphere_for<'a>(system: GnssSystem, inputs: &'a SolveInputs) -> SppIonosphere<'a> {
    if let Some(grid) = inputs
        .sbas_iono
        .as_ref()
        .filter(|_| inputs.corrections.ionosphere)
    {
        return SppIonosphere::SbasGrid(grid);
    }
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
    /// Which code the pseudoranges are; the group delay applies to single-frequency code.
    pub pseudorange_code: PseudorangeCode,
    /// Pseudorange, metres, that places each listed satellite's transmission epoch, where it
    /// differs from the one the residual is formed from; `None`, or a satellite not listed,
    /// places from the residual's pseudorange. A differential rover's satellites are placed
    /// from its raw code while the residual uses the corrected code, as RTKLIB `rtkpos`
    /// calls `satposs` with the rover's own observations.
    pub placement_pseudoranges_m: Option<&'a BTreeMap<GnssSatelliteId, f64>>,
}

/// Whether RTKLIB `satazel` puts every satellite at the zenith for a receiver at
/// `rx_ecef_m`: it does where the RTKLIB `ecef2pos` height is at or below
/// `-RE_WGS84`, taking azimuth `0` and elevation `pi / 2`, so a solve started from
/// the all-zero initial guess keeps every satellite through the elevation mask on
/// its first pass.
///
/// That height reaches `-RE_WGS84` only at the geocentre itself. Where
/// `|z| < 1e-4 m` the `ecef2pos` loop does not run, `v` stays `RE_WGS84` and the
/// height is `sqrt(x^2 + y^2 + z^2) - RE_WGS84`, which rounds to `-RE_WGS84` only
/// for a distance below half an ulp of `RE_WGS84`, about `4.7e-10 m`. Where
/// `|z| >= 1e-4 m` near the geocentre the loop runs to `sin(lat) = +-1`,
/// `v = RE_WGS84 / sqrt(1 - e2)` and `|z| = |z_0| + v e2`, a height of about
/// `|z_0| - b`, near `-6_356_752 m`. Any position more than 1 m from the geocentre
/// is therefore outside, and within 1 m the `ecef2pos` arithmetic is run as
/// written.
fn rtklib_sees_every_satellite_overhead(rx_ecef_m: [f64; 3]) -> bool {
    let r2 = rx_ecef_m[0] * rx_ecef_m[0] + rx_ecef_m[1] * rx_ecef_m[1];
    if r2 + rx_ecef_m[2] * rx_ecef_m[2] > 1.0 {
        return false;
    }
    // RTKLIB `ecef2pos`, height only, in its operation order.
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let mut z = rx_ecef_m[2];
    let mut zk = 0.0_f64;
    let mut v = WGS84_A_M;
    while (z - zk).abs() >= 1.0e-4 {
        zk = z;
        let sinp = z / (r2 + z * z).sqrt();
        v = WGS84_A_M / (1.0 - e2 * sinp * sinp).sqrt();
        z = rx_ecef_m[2] + v * e2 * sinp;
    }
    let height_m = (r2 + z * z).sqrt() - v;
    // RTKLIB `satazel` computes the angles only where `pos[2] > -RE_WGS84`, which
    // a NaN height fails as well.
    height_m.is_nan() || height_m <= -WGS84_A_M
}

/// Build the per-satellite predicted pseudorange in the SPP operation order
/// SELECTED BY THE RECIPE on [`SatModelEnv::model`], sharing the
/// parity-sensitive range and frame substrate with the other strategies.
///
/// The three model stages are read from the recipe rather than hard-coded:
/// - **range** (`env.model.range`): the transmission epoch.
///   [`RangeRecipe::RtklibSatpossPseudorange`] (the SPP reference) places it from
///   the measured pseudorange as RTKLIB `satposs` does, with no iteration.
///   [`RangeRecipe::CanonicalLightTimeClosedFormSagnac`] (the canonical strategy)
///   iterates the geometric light time to convergence from the reception epoch
///   corrected by the state's receiver clock (the IERS-rigorous op-order).
///   [`RangeRecipe::SppMeasuredPseudorangeFixedIter`] is the geometric light time
///   from the receiver's time tag that the external references were computed
///   with, which only their replay selects. The observable rounded-microsecond and
///   RTK provided-transmit recipes are other strategies' range models and never
///   reach here.
/// - **sagnac** (`env.model.sagnac`): RTKLIB's first-order scalar Sagnac term (the
///   reference), or the closed-form Z-rotation and the pre/post-rotation geometric
///   range, route through [`crate::estimation::substrate::range`] under the
///   selected recipe.
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
    ionosphere: SppIonosphere<'_>,
) -> Option<SatModel> {
    let sagnac = env.model.sagnac;
    let frame = env.model.frame;

    // Transmission epoch, selected by the range recipe.
    // `_t_tx` is read only by the test-build trace fields below.
    let (sat_pos, dt_sat, tau, group_delay, t_state, _t_tx) = match env.model.range {
        RangeRecipe::RtklibSatpossPseudorange => {
            // RTKLIB `satposs`: the clock read at `t_rx - P / c` (`ephclk`) places the
            // transmission epoch `t_rx - P / c - dts`, and the state is read there. No
            // light-time iteration: the pseudorange carries the flight time and the
            // receiver clock offset, so the epoch does not depend on the receiver state
            // and every evaluation of a solve reads the same two epochs. RTKLIB reads a
            // zero pseudorange as none and places no satellite for it.
            let p_place_m = env
                .placement_pseudoranges_m
                .and_then(|placement| placement.get(&sat).copied())
                .unwrap_or(p_meas_m);
            if !p_place_m.is_finite() || p_place_m <= 0.0 {
                return None;
            }
            let clock_epoch =
                crate::observables::pseudorange_clock_epoch_j2000_s(env.t_rx_j2000_s, p_place_m);
            // RTKLIB selects each broadcast record by the observation epoch (`teph`), the
            // reception epoch, for the clock and for the state alike.
            let placement_clock_s = env
                .eph
                .try_transmit_epoch_clock_s(sat, clock_epoch, env.t_rx_j2000_s)
                .ok()
                .flatten()?
                .value;
            let t_tx = crate::observables::pseudorange_transmit_epoch_from_clock_j2000_s(
                env.t_rx_j2000_s,
                p_place_m,
                placement_clock_s,
            );
            let (pos, clk, gd) = env
                .eph
                .try_position_clock_group_delay_selected_at_j2000_s(sat, t_tx, env.t_rx_j2000_s)
                .ok()
                .flatten()?
                .value;
            // The flight time a closed-form rotation turns the satellite through, when a
            // recipe pairs one with this placement: the geometric range over `c`. RTKLIB
            // `geodist` rotates nothing and takes the first-order Sagnac term instead.
            let tau = geometric_range(SagnacRecipe::Off, pos, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S)
                / C_M_S;
            (pos, clk, tau, gd, t_tx, t_tx)
        }
        RangeRecipe::SppMeasuredPseudorangeFixedIter => {
            // Geometric light time from the receiver's time tag: fixed iteration count,
            // no inner convergence test; seed tau from the measured pseudorange. The
            // external SPP references were computed this way; it misses the receiver
            // clock offset.
            let mut tau = p_meas_m / C_M_S;
            let mut t_tx = env.t_rx_j2000_s - tau;
            let mut sat_pos = [0.0f64; 3];
            let mut dt_sat = 0.0f64;
            let mut group_delay = None;
            let mut t_state = t_tx;
            for _ in 0..TRANSMIT_TIME_ITERATIONS {
                let (pos, clk, gd) = env.eph.position_clock_group_delay_at_j2000_s(sat, t_tx)?;
                sat_pos = pos;
                dt_sat = clk;
                group_delay = gd;
                t_state = t_tx;
                // Pre-rotation geometric range through the shared substrate (the
                // closed-form recipe = plain `norm3(sub3(sat, recv))`).
                let rho0 = geometric_range(sagnac, sat_pos, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S);
                tau = rho0 / C_M_S;
                t_tx = env.t_rx_j2000_s - tau;
            }
            (sat_pos, dt_sat, tau, group_delay, t_state, t_tx)
        }
        RangeRecipe::CanonicalLightTimeClosedFormSagnac => {
            // Full iterative light-time (the IERS-rigorous op-order): iterate the
            // transmit epoch until the signal travel time stops changing, rather
            // than a fixed truncation. The reception epoch is the receiver's time tag
            // less the receiver clock offset of the state, `b / c`, so the fixed point
            // `t_tx = (t_rx - b / c) - rho(t_tx) / c` is the true transmission epoch;
            // the receiver's time tag alone would move each satellite by `v · b / c`.
            // Seeded, like the reference, from the measured pseudorange; the range is
            // the closed-form Sagnac range (never a first-order scalar Sagnac). The
            // satellite clock's relativistic periodic term is applied once, after the
            // iteration: a broadcast clock carries it (`F*e*sqrt(A)*sin(E)`), and a
            // precise product clock takes the `peph2pos` term the source returns.
            let t_rx_true = env.t_rx_j2000_s - b_m / C_M_S;
            let mut tau = p_meas_m / C_M_S;
            let mut t_tx = env.t_rx_j2000_s - tau;
            let mut sat_pos = [0.0f64; 3];
            let mut dt_sat = 0.0f64;
            let mut group_delay = None;
            let mut t_state = t_tx;
            let mut prev_tau = f64::INFINITY;
            for _ in 0..CANONICAL_LIGHT_TIME_MAX_ITERS {
                let (pos, clk, gd) = env.eph.position_clock_group_delay_at_j2000_s(sat, t_tx)?;
                sat_pos = pos;
                dt_sat = clk;
                group_delay = gd;
                t_state = t_tx;
                let rho0 = geometric_range(sagnac, sat_pos, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S);
                tau = rho0 / C_M_S;
                t_tx = t_rx_true - tau;
                if (tau - prev_tau).abs() <= CANONICAL_LIGHT_TIME_TOL_S {
                    break;
                }
                prev_tau = tau;
            }
            (sat_pos, dt_sat, tau, group_delay, t_state, t_tx)
        }
        RangeRecipe::ObservableRoundedMicrosecondFixedIter
        | RangeRecipe::RtkProvidedTxFirstOrderSagnac => unreachable!(
            "the SPP measurement model runs only the RTKLIB placement or a geometric light-time recipe"
        ),
    };

    // Single-frequency group delay. The source's clock is the one RTKLIB `satposs`
    // returns, without TGD or BGD; RTKLIB `pntpos` applies the delay to the
    // single-frequency pseudorange (`prange`: `P1 - TGD`). It is taken here from the
    // clock, for the record the clock came from, as the final transmit-time step returns
    // it: `(poly + rel) - TGD`, which is the broadcast `dt_clock_total_s` bit for bit.
    // Relativistic clock term for a product clock that leaves it to the user (SP3 and
    // RINEX CLK), as RTKLIB `peph2pos` applies it for positioning: `dts - 2 r·v / c²`,
    // at the epoch the clock came from. A broadcast, SSR- or SBAS-corrected clock carries
    // its own term and the source returns none. The source is handed the position it
    // returned at that epoch, so a precise source interpolates only the position 1 ms
    // later.
    let dt_sat = match env.eph.clock_relativity_for_state_s(sat, t_state, sat_pos) {
        ClockRelativity::NotApplicable => dt_sat,
        ClockRelativity::Term(relativity_s) => dt_sat + relativity_s,
        ClockRelativity::Unavailable => return None,
    };

    let group_delay = match env.pseudorange_code {
        PseudorangeCode::SingleFrequency => group_delay,
        PseudorangeCode::IonosphereFree => None,
    };
    let dt_sat = match group_delay {
        Some(group_delay_s) => dt_sat - group_delay_s,
        None => dt_sat,
    };

    // Sagnac / Earth-rotation correction, selected by recipe: the closed-form rotation
    // over the flight time turns the satellite into the reception-epoch frame, and
    // RTKLIB's first-order term leaves it in the transmission-epoch frame.
    let sat_rot = rotate_transmit_satellite(sagnac, sat_pos, tau, OMEGA_E_DOT_RAD_S);

    // Geometric range through the shared substrate: RTKLIB `geodist` under the
    // first-order term, the Euclidean range from the rotated position otherwise.
    let rho = geometric_range(sagnac, sat_rot, rx_ecef_m, OMEGA_E_DOT_RAD_S, C_M_S);

    // Geometry for corrections: az/el from rx and that satellite position, as RTKLIB
    // `satazel` takes it from the `geodist` line of sight, through the
    // recipe-selected frame substrate. A receiver RTKLIB places at or below the
    // geocentre sees every satellite overhead (`satazel`).
    let mut g = az_el_from_ecef(frame, rx_ecef_m, sat_rot);
    if rtklib_sees_every_satellite_overhead(rx_ecef_m) {
        g.az_rad = 0.0;
        g.el_rad = PI / 2.0;
    }

    let mut iono_m = 0.0;
    let mut tropo_m = 0.0;
    // RTKLIB evaluates no broadcast ionosphere delay for a receiver more than 1 km
    // below the ellipsoid or a satellite at or below its horizon (`ionmodel`), and
    // no augmentation-grid delay more than 100 m below it or at or below the
    // horizon (`sbsioncorr`); the delay there is zero. The troposphere model gates
    // the same way itself.
    let ionosphere_gated = match ionosphere {
        SppIonosphere::SbasGrid(_) => g.geodetic.height_m < -100.0 || g.el_rad <= 0.0,
        SppIonosphere::Klobuchar(_) | SppIonosphere::GalileoNequick(_) => {
            g.geodetic.height_m < -1.0e3 || g.el_rad <= 0.0
        }
    };
    if env.corrections.ionosphere && !ionosphere_gated {
        // The SPP 0-ULP trace oracle pins this multiply-then-divide order, which
        // `rad_to_deg_ref` implements (`rad * 180 / PI`).
        let lat_deg = rad_to_deg_ref(g.geodetic.lat_rad);
        let lon_deg = rad_to_deg_ref(g.geodetic.lon_rad);
        let az_deg = rad_to_deg_ref(g.az_rad);
        let el_deg = rad_to_deg_ref(g.el_rad);
        // A used satellite always has a resolvable carrier here: selection
        // excludes, from an ionosphere-corrected solve, any satellite that does
        // not, GLONASS included via its FDMA channel. Selection evaluates such a
        // satellite with the L1 fallback only to classify it and then discards
        // that model, so no solved measurement uses the fallback. The GLONASS
        // per-satellite carrier makes the Klobuchar delay scale by
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
            SppIonosphere::SbasGrid(grid) => {
                grid.slant_delay_m(g.geodetic, g.el_rad, g.az_rad, freq_hz)?
            }
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
        dt_sat_s: dt_sat,
        rho_m: rho,
        iono_m,
        tropo_m,
        #[cfg(all(test, sidereon_repo_tests))]
        az_rad: g.az_rad,
        #[cfg(all(test, sidereon_repo_tests))]
        tau_s: tau,
        // The RTKLIB placement's transmission epoch, or a light-time loop's final
        // `t_tx = t_rx - tau` (the geometric reference) or `t_rx - b / c - tau` (canonical).
        #[cfg(all(test, sidereon_repo_tests))]
        t_tx_j2000_s: _t_tx,
        #[cfg(all(test, sidereon_repo_tests))]
        sat_ecef_m: sat_pos,
        #[cfg(all(test, sidereon_repo_tests))]
        theta_rad: OMEGA_E_DOT_RAD_S * tau,
        #[cfg(all(test, sidereon_repo_tests))]
        clock_epoch_j2000_s: t_state,
    })
}

/// The satellite selection at one receiver state, as RTKLIB `rescode` makes it:
/// used satellites (ascending id), rejected satellites with reason, and for each
/// used satellite its elevation weight, line of sight and residual at that state.
pub(crate) struct Selection {
    pub used: Vec<GnssSatelliteId>,
    pub rejected: Vec<RejectedSat>,
    /// `weight` per used satellite, index-aligned to `used`: `sin^2(el) / sigma0^2`
    /// at the state the selection was made at.
    pub weights: Vec<f64>,
    /// Unit line of sight from the receiver to the satellite position the range
    /// is formed from, per used satellite, index-aligned to `used`.
    pub lines_of_sight: Vec<LineOfSight>,
    /// `P_meas - P_hat` per used satellite, index-aligned to `used`, each with the
    /// clock of its own system.
    pub residuals_m: Vec<f64>,
}

/// The clock a satellite's residual takes: SBAS ranges on the GPS clock.
pub(crate) const fn clock_system(system: GnssSystem) -> GnssSystem {
    match system {
        GnssSystem::Sbas => GnssSystem::Gps,
        system => system,
    }
}

/// The selection at the receiver state `rx_ecef_m` with the clock `clock_m` gives
/// each system, the satellites of `placement` placed from its pseudoranges
/// ([`SatModelEnv::placement_pseudoranges_m`]).
///
/// Reasons are tested in RTKLIB `rescode` order and the first that applies is
/// reported: no ephemeris, the elevation mask, augmentation-grid coverage, then
/// the carrier the ionosphere delay is scaled to.
pub(crate) fn select_at(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    model: SppModelRecipe,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
    rx_ecef_m: [f64; 3],
    clock_m: &dyn Fn(GnssSystem) -> f64,
) -> Selection {
    // Ascending satellite-id order, never observation order.
    let mut obs: Vec<&Observation> = inputs.observations.iter().collect();
    obs.sort_by_key(|o| o.satellite_id);

    let mut used = Vec::new();
    let mut rejected = Vec::new();
    let mut weights = Vec::new();
    let mut lines_of_sight = Vec::new();
    let mut residuals_m = Vec::new();

    let env = SatModelEnv {
        eph,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
        day_of_year: inputs.day_of_year,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &inputs.glonass_channels,
        model,
        pseudorange_code: inputs.pseudorange_code,
        placement_pseudoranges_m: placement,
    };
    for ob in obs {
        let sat = ob.satellite_id;
        let b = clock_m(clock_system(sat.system));
        let ionosphere = ionosphere_for(sat.system, inputs);
        let Some(model) = sat_model(&env, sat, rx_ecef_m, b, ob.pseudorange_m, ionosphere) else {
            // With an augmentation grid bound, a line of sight the grid does
            // not cover leaves no model either. The grid-free model tells an
            // ephemeris gap from that, and gives the elevation, which is
            // tested before coverage.
            let reason = match ionosphere {
                SppIonosphere::SbasGrid(_) => {
                    let grid_free = SppIonosphere::Klobuchar(KlobucharCoeffs {
                        alpha: [0.0; 4],
                        beta: [0.0; 4],
                    });
                    match sat_model(&env, sat, rx_ecef_m, b, ob.pseudorange_m, grid_free) {
                        None => RejectionReason::NoEphemeris,
                        Some(geometry) if geometry.el_rad < ELEVATION_MASK_RAD => {
                            RejectionReason::LowElevation
                        }
                        Some(_) => RejectionReason::SbasIonoUncovered,
                    }
                }
                _ => RejectionReason::NoEphemeris,
            };
            rejected.push(RejectedSat {
                satellite_id: sat,
                reason,
            });
            continue;
        };
        if model.el_rad < ELEVATION_MASK_RAD {
            rejected.push(RejectedSat {
                satellite_id: sat,
                reason: RejectionReason::LowElevation,
            });
            continue;
        }
        // The ionosphere delay is the one term of the model that reads the
        // carrier: it is computed on L1 and scaled to each satellite's carrier by
        // `(f_L1 / f)^2` (the group delay and every other term are taken as the
        // source gives them). GPS, QZSS and SBAS L1, Galileo E1, BeiDou B1I and
        // NavIC L5 are fixed carriers; GLONASS is FDMA, so its carrier is resolved
        // per satellite from `glonass_channels`. A satellite whose carrier cannot
        // be resolved (a GLONASS observation with no channel in the map, or a
        // channel outside the `-7..=6` FDMA allocation) cannot take the
        // correction, so with it applied it is excluded and reported, and the
        // other satellites are solved without it, as RTKLIB `rescode` skips a
        // satellite whose `sat2freq` is zero. Without the correction nothing in
        // the model reads the carrier and the satellite is used, where `rescode`
        // skips it whatever the ionosphere option. Its model above was evaluated
        // with the L1 fallback only to classify it, and is discarded.
        if inputs.corrections.ionosphere
            && spp_iono_frequency_hz(sat, &inputs.glonass_channels).is_none()
        {
            rejected.push(RejectedSat {
                satellite_id: sat,
                reason: RejectionReason::IonosphereCarrierUnresolved,
            });
            continue;
        }
        let sin_el = libm::sin(model.el_rad);
        let weight = (sin_el * sin_el) / (SIGMA0_M * SIGMA0_M);
        used.push(sat);
        weights.push(weight);
        lines_of_sight.push(line_of_sight(model.sat_rot_ecef_m, rx_ecef_m));
        residuals_m.push(ob.pseudorange_m - model.p_hat_m);
    }

    Selection {
        used,
        rejected,
        weights,
        lines_of_sight,
        residuals_m,
    }
}

/// The unit vector from the receiver to the satellite position a range is formed
/// from.
pub(crate) fn line_of_sight(sat_ecef_m: [f64; 3], rx_ecef_m: [f64; 3]) -> LineOfSight {
    let dx = sat_ecef_m[0] - rx_ecef_m[0];
    let dy = sat_ecef_m[1] - rx_ecef_m[1];
    let dz = sat_ecef_m[2] - rx_ecef_m[2];
    let n = (dx * dx + dy * dy + dz * dz).sqrt();
    LineOfSight::new(dx / n, dy / n, dz / n)
}

/// The distinct GNSS present in `used`, in ascending system order.
///
/// The receiver-clock part of the state has one entry per system, each the
/// *absolute* receiver clock for that system (not a bias); the first is the
/// reference clock and a system's inter-system bias is its clock minus that
/// reference. For a single-system solve this is one element and the state is the
/// classic `[x, y, z, b]`.
pub(crate) fn clock_systems(used: &[GnssSatelliteId]) -> Vec<GnssSystem> {
    let mut systems: Vec<GnssSystem> = used.iter().map(|s| clock_system(s.system)).collect();
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
/// ephemeris at `x` (the used set is fixed for one solve, but a finite-difference probe
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
    residual_unweighted_placed(eph, used, obs_by_id, x, inputs, model, None)
}

/// [`residual_unweighted`] with the satellites of `placement` placed from its
/// pseudoranges ([`SatModelEnv::placement_pseudoranges_m`]).
fn residual_unweighted_placed(
    eph: &dyn EphemerisSource,
    used: &[GnssSatelliteId],
    obs_by_id: &[(GnssSatelliteId, f64)],
    x: &[f64],
    inputs: &SolveInputs,
    model: SppModelRecipe,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
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
        pseudorange_code: inputs.pseudorange_code,
        placement_pseudoranges_m: placement,
    };
    let mut out = Vec::with_capacity(used.len());
    for &sat in used {
        let p_meas = obs_by_id
            .iter()
            .find(|(id, _)| *id == sat)
            .map(|(_, p)| *p)
            .ok_or(sat)?;
        // The clock for this satellite's system (index 0 = reference clock).
        let sys_idx = systems
            .iter()
            .position(|s| *s == clock_system(sat.system))
            .unwrap_or(0);
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

/// [`solve`] with the satellites of `placement` placed from its pseudoranges while the
/// residuals use the observations' ([`SatModelEnv::placement_pseudoranges_m`]): a
/// differential rover is placed from its raw code and solved from its corrected code.
pub(crate) fn solve_placed(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    placement: &BTreeMap<GnssSatelliteId, f64>,
    with_geodetic: bool,
) -> Result<ReceiverSolution, SppError> {
    validate_solve_inputs(inputs)?;
    solve_inner_placed(
        eph,
        inputs,
        with_geodetic,
        SppModelRecipe::reference(),
        TrustRegionSolve::NalgebraLu,
        Some(placement),
    )
}

/// Solve receiver ECEF velocity and clock drift from Doppler rows using SPP
/// position geometry.
///
/// With no pseudoranges to place the satellites by, each satellite is read from the
/// geometric light-time prediction at the reception epoch ([`velocity::solve`]), a
/// prediction from a known position. [`solve_with_doppler_velocity`] places them from
/// the epoch's pseudoranges instead.
pub fn solve_doppler_velocity(
    source: &dyn ObservableEphemerisSource,
    inputs: &DopplerVelocityInputs,
) -> Result<VelocitySolution, VelocityError> {
    let observations: Vec<_> = inputs
        .observations
        .iter()
        .map(|obs| VelocityObservation {
            satellite_id: obs.satellite_id,
            value: obs.doppler_hz,
            carrier_hz: obs.carrier_hz,
            sat_clock_drift_s_s: obs.sat_clock_drift_s_s,
        })
        .collect();
    velocity::solve(
        source,
        &observations,
        inputs.receiver_ecef_m,
        inputs.t_rx_j2000_s,
        VelocitySolveOptions {
            observable: VelocityObservable::Doppler,
            light_time: inputs.light_time,
            sagnac: inputs.sagnac,
        },
    )
}

/// Solve SPP position and attach a Doppler velocity/clock-drift estimate when
/// the Doppler rows are usable.
///
/// Each Doppler satellite is read at the transmission epoch of its pseudorange in
/// `inputs`, placed as RTKLIB `satposs` places it, the state the position solve used, and
/// the rows carry the rate of the first-order Sagnac term the code rows' ranges carry. A
/// Doppler satellite with no pseudorange has no transmission epoch and no row, as RTKLIB
/// places no satellite without one.
///
/// A pseudorange-only or underdetermined Doppler epoch still returns the
/// receiver position; in that case `receiver.rx_clock_drift_s_s` and `velocity`
/// are `None`, with the velocity failure retained in `velocity_error`.
pub fn solve_with_doppler_velocity<E>(
    eph: &E,
    inputs: &SolveInputs,
    doppler_observations: &[DopplerObservation],
    with_geodetic: bool,
) -> Result<SppDopplerSolution, SppError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    let mut receiver = solve(eph, inputs, with_geodetic)?;
    if doppler_observations.is_empty() {
        return Ok(SppDopplerSolution {
            receiver,
            velocity: None,
            velocity_error: None,
        });
    }

    let velocity_inputs = DopplerVelocityInputs::from_receiver_solution(
        &receiver,
        doppler_observations.to_vec(),
        inputs.t_rx_j2000_s,
    );
    // Each Doppler satellite is read at the transmission epoch its pseudorange places, the
    // state the position solve used, as RTKLIB `estvel` reads the `satposs` states.
    let pseudoranges_m: BTreeMap<GnssSatelliteId, f64> = inputs
        .observations
        .iter()
        .map(|obs| (obs.satellite_id, obs.pseudorange_m))
        .collect();
    let velocity_observations: Vec<_> = velocity_inputs
        .observations
        .iter()
        .map(|obs| VelocityObservation {
            satellite_id: obs.satellite_id,
            value: obs.doppler_hz,
            carrier_hz: obs.carrier_hz,
            sat_clock_drift_s_s: obs.sat_clock_drift_s_s,
        })
        .collect();
    let placed = velocity::solve_placed(
        eph,
        &velocity_observations,
        velocity_inputs.receiver_ecef_m,
        velocity_inputs.t_rx_j2000_s,
        VelocitySolveOptions {
            observable: VelocityObservable::Doppler,
            light_time: velocity_inputs.light_time,
            sagnac: velocity_inputs.sagnac,
        },
        &pseudoranges_m,
    );
    match placed {
        Ok(velocity) => {
            receiver.rx_clock_drift_s_s = Some(velocity.clock_drift_s_s);
            Ok(SppDopplerSolution {
                receiver,
                velocity: Some(velocity),
                velocity_error: None,
            })
        }
        Err(error) => Ok(SppDopplerSolution {
            receiver,
            velocity: None,
            velocity_error: Some(error),
        }),
    }
}

/// SPP's trust-region stage recognizes the owned deterministic solver
/// ([`SolverRecipe::OwnedDeterministicTrf`]), which owns the trust-region
/// assembly and dense subproblem factorization with a fixed reduction order and
/// its own frozen-bits golden;
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
/// golden; all other model stages are unchanged. The owned kernel uses
/// fixed-order scalar arithmetic for the complete trust-region assembly and
/// factorization (no nalgebra LU or black-box BLAS), so its converged bits are
/// portable across CPU targets.
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

/// [`solve_tracked`] reading `eph` through a [`Ut1TrackedSource`]: a UT1
/// refusal anywhere in the solve fails it with [`SppError::Ut1OutsideCoverage`]
/// (taking precedence over the error or the reduced solution the missing state
/// led to), and an accepted departure is reported in
/// [`SolutionMetadata::ut1_degraded`].
fn solve_inner(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    model: SppModelRecipe,
    linear_solve: TrustRegionSolve,
) -> Result<ReceiverSolution, SppError> {
    solve_inner_placed(eph, inputs, with_geodetic, model, linear_solve, None)
}

/// [`solve_inner`] with the satellites of `placement` placed from its pseudoranges
/// ([`SatModelEnv::placement_pseudoranges_m`]).
fn solve_inner_placed(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    model: SppModelRecipe,
    linear_solve: TrustRegionSolve,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
) -> Result<ReceiverSolution, SppError> {
    let tracked = Ut1TrackedSource::new(eph);
    let result = solve_tracked(
        &tracked,
        inputs,
        with_geodetic,
        model,
        linear_solve,
        placement,
    );
    if let Some(reason) = tracked.refusal() {
        return Err(SppError::Ut1OutsideCoverage(reason));
    }
    let mut solution = result?;
    solution.metadata.ut1_degraded = tracked.departure();
    Ok(solution)
}

/// The receiver state an SPP solve iterates on: the position and one absolute
/// clock per system the last pass solved for.
pub(crate) struct IterateState {
    pub(crate) rx_ecef_m: [f64; 3],
    clocks_m: Vec<(GnssSystem, f64)>,
    /// The clock a system with no estimate yet starts from: the reference clock of
    /// the last pass, or the initial guess's clock before the first. A system
    /// joins at zero inter-system bias, as RTKLIB starts its offsets at zero.
    reference_clock_m: f64,
}

impl IterateState {
    pub(crate) fn from_initial_guess(initial_guess: [f64; 4]) -> Self {
        Self {
            rx_ecef_m: [initial_guess[0], initial_guess[1], initial_guess[2]],
            clocks_m: Vec::new(),
            reference_clock_m: initial_guess[3],
        }
    }

    pub(crate) fn clock_m(&self, system: GnssSystem) -> f64 {
        self.clocks_m
            .iter()
            .find(|(s, _)| *s == system)
            .map_or(self.reference_clock_m, |&(_, clock)| clock)
    }

    /// The parameter vector `[x, y, z, clk_0, clk_1, ...]` for `systems`.
    pub(crate) fn parameters(&self, systems: &[GnssSystem]) -> DVector<f64> {
        let mut x = self.rx_ecef_m.to_vec();
        x.extend(systems.iter().map(|&system| self.clock_m(system)));
        DVector::from_vec(x)
    }

    pub(crate) fn update(&mut self, systems: &[GnssSystem], x: &DVector<f64>) {
        self.rx_ecef_m = [x[0], x[1], x[2]];
        self.clocks_m = systems
            .iter()
            .enumerate()
            .map(|(i, &system)| (system, x[3 + i]))
            .collect();
        self.reference_clock_m = x[3];
    }
}

/// The selection at `state`, refused with [`SppError::TooFewSatellites`] when it
/// keeps fewer satellites than the solve has parameters, as RTKLIB `estpos` stops
/// when `rescode` returns fewer rows than unknowns.
fn select_at_state(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    model: SppModelRecipe,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
    state: &IterateState,
) -> Result<(Selection, Vec<GnssSystem>), SppError> {
    let sel = select_at(eph, inputs, model, placement, state.rx_ecef_m, &|system| {
        state.clock_m(system)
    });
    // One receiver-clock parameter per distinct GNSS (a reference clock plus an
    // inter-system bias for each additional system), so the state has
    // `3 + n_systems` parameters and needs at least that many usable satellites.
    // Floor the clock count at one: the minimum solve is the four-parameter
    // single-system form even when no satellite survives selection.
    let systems = clock_systems(&sel.used);
    // SPP's weighted-residual rows feed the trust-region solver, which owns the
    // normal-equation factorization (NormalRecipe::SppWeightedResidualFiniteDifference
    // via SolverRecipe::NalgebraTrfLegacy); only the parameter stack is named here.
    let n_params = ParameterLayout::spp(systems.len().max(1)).dim();
    if sel.used.len() < n_params {
        return Err(SppError::TooFewSatellites {
            used: sel.used.len(),
            required: n_params,
        });
    }
    Ok((sel, systems))
}

/// The state column of each used satellite's clock: `3 +` the index of its clock
/// system in `systems`.
fn clock_columns(used: &[GnssSatelliteId], systems: &[GnssSystem]) -> Vec<usize> {
    used.iter()
        .map(|sat| {
            3 + systems
                .iter()
                .position(|s| *s == clock_system(sat.system))
                .unwrap_or(0)
        })
        .collect()
}

/// The weighted normal matrix `H^T W H` of a pseudorange design whose row `k` is
/// `[-e_k, 0, .., 1, .., 0]` with the one in state column `clock_columns[k]`, over
/// `n_params` state parameters. `None` when the rows do not form a design with at
/// least as many rows as parameters.
pub(crate) fn weighted_normal_matrix(
    los: &[LineOfSight],
    clock_columns: &[usize],
    n_params: usize,
    weights: &[f64],
) -> Option<Vec<Vec<f64>>> {
    if los.len() != clock_columns.len() || los.len() != weights.len() || n_params <= 3 {
        return None;
    }
    if los.len() < n_params {
        return None;
    }
    let mut normal = vec![vec![0.0_f64; n_params]; n_params];
    for k in 0..los.len() {
        let row = design_row(los[k], clock_columns[k], n_params)?;
        let weight = weights[k];
        for i in 0..n_params {
            for j in 0..n_params {
                normal[i][j] += row[i] * weight * row[j];
            }
        }
    }
    Some(normal)
}

/// One design row `[-e_x, -e_y, -e_z, 0, .., 1, .., 0]`, the one in state column
/// `clock_column`.
fn design_row(los: LineOfSight, clock_column: usize, n_params: usize) -> Option<Vec<f64>> {
    if clock_column < 3 || clock_column >= n_params {
        return None;
    }
    let mut row = vec![0.0_f64; n_params];
    row[0] = -los.e_x;
    row[1] = -los.e_y;
    row[2] = -los.e_z;
    row[clock_column] = 1.0;
    Some(row)
}

/// The step RTKLIB `estpos` takes from a state: the weighted least-squares
/// `dx = (H^T W H)^-1 H^T W v` over the design `H` of [`weighted_normal_matrix`]
/// (`[-e, 1]` in the satellite's clock column, the partials of the predicted
/// range that RTKLIB `rescode` forms), the residuals `v = P_meas - P_hat` and the
/// weights `W` at that state. `None` when the design is singular.
pub(crate) fn rtklib_step(
    los: &[LineOfSight],
    clock_columns: &[usize],
    n_params: usize,
    weights: &[f64],
    residuals_m: &[f64],
) -> Option<Vec<f64>> {
    if residuals_m.len() != los.len() {
        return None;
    }
    let normal = weighted_normal_matrix(los, clock_columns, n_params, weights)?;
    let q = invert_symmetric_pd(&normal)?;
    let mut rhs = vec![0.0_f64; n_params];
    for k in 0..los.len() {
        let row = design_row(los[k], clock_columns[k], n_params)?;
        for i in 0..n_params {
            rhs[i] += row[i] * weights[k] * residuals_m[k];
        }
    }
    let dx: Vec<f64> = q
        .iter()
        .map(|q_row| {
            let mut sum = 0.0_f64;
            for j in 0..n_params {
                sum += q_row[j] * rhs[j];
            }
            sum
        })
        .collect();
    dx.iter().all(|v| v.is_finite()).then_some(dx)
}

/// `norm(dx)` as RTKLIB `estpos` takes it, over its own parameters: the position,
/// the GPS receiver clock and one inter-system bias per other system, each the
/// system's clock less the GPS clock. `clock_groups` gives, for each receiver
/// (one for SPP, one per epoch for the static solve), the state column of its
/// first clock and its clock systems in column order; each absolute clock step
/// `d_s` is taken to RTKLIB's parameters as `d_GPS` and `d_s - d_GPS`. The step
/// is the same step in either parametrization, so the solve stops on the iterate
/// `estpos` stops on. Without a GPS clock RTKLIB holds its GPS clock parameter by
/// a pseudo-observation, and its step there is taken as zero.
pub(crate) fn rtklib_step_norm(dx: &[f64], clock_groups: &[(usize, &[GnssSystem])]) -> f64 {
    let mut norm2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    for &(offset, systems) in clock_groups {
        let gps = systems.iter().position(|&s| s == GnssSystem::Gps);
        let d_gps = gps.map_or(0.0, |i| dx[offset + i]);
        if gps.is_some() {
            norm2 += d_gps * d_gps;
        }
        for (i, _) in systems.iter().enumerate() {
            if Some(i) == gps {
                continue;
            }
            let bias = dx[offset + i] - d_gps;
            norm2 += bias * bias;
        }
    }
    norm2.sqrt()
}

/// Whether a positioning solve that ended with `status` converged: a
/// trust-region convergence criterion, or a settled selection.
pub(crate) const fn solve_converged(status: Status) -> bool {
    matches!(
        status,
        Status::GradientTolerance
            | Status::CostTolerance
            | Status::StepTolerance
            | Status::SelectionSettled
    )
}

/// How one trust-region pass over a selection ended.
pub(crate) enum PassEnd {
    /// The solve over the selection ran to its end.
    Solved(least_squares::LeastSquaresReport),
    /// A satellite's line of sight left the augmentation ionosphere grid at a
    /// state the solve evaluated, a finite-difference probe or a trial point
    /// perhaps. The pass ends at the last iterate the solve accepted, the given
    /// state; the next pass selects there, and when the selection there is the
    /// same, takes RTKLIB's step, which evaluates no probe, so the next selection
    /// finds the crossing where `rescode` would.
    CoverageLost(DVector<f64>),
}

/// Whether `sat`, which has no model at `rx_ecef_m`, has one there with the
/// augmentation grid set aside: its line of sight left the grid, not its
/// ephemeris.
pub(crate) fn lost_grid_coverage(
    env: &SatModelEnv,
    inputs: &SolveInputs,
    sat: GnssSatelliteId,
    rx_ecef_m: [f64; 3],
    b_m: f64,
    p_meas_m: f64,
) -> bool {
    if !matches!(
        ionosphere_for(sat.system, inputs),
        SppIonosphere::SbasGrid(_)
    ) {
        return false;
    }
    let grid_free = SppIonosphere::Klobuchar(KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    });
    sat_model(env, sat, rx_ecef_m, b_m, p_meas_m, grid_free).is_some()
}

/// The environment the SPP model reads for `inputs`.
pub(crate) fn model_env<'a>(
    eph: &'a dyn EphemerisSource,
    inputs: &'a SolveInputs,
    model: SppModelRecipe,
    placement: Option<&'a BTreeMap<GnssSatelliteId, f64>>,
) -> SatModelEnv<'a> {
    SatModelEnv {
        eph,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
        day_of_year: inputs.day_of_year,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &inputs.glonass_channels,
        model,
        pseudorange_code: inputs.pseudorange_code,
        placement_pseudoranges_m: placement,
    }
}

/// One trust-region solve over the satellites `used` with the fixed `weights`,
/// from `x0`. A used satellite whose line of sight leaves the augmentation grid
/// at a state the solve reaches ends the pass there; a used satellite whose
/// ephemeris is lost fails the solve.
#[allow(clippy::too_many_arguments)]
fn solve_selected(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    model: SppModelRecipe,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
    linear_solve: TrustRegionSolve,
    obs_by_id: &[(GnssSatelliteId, f64)],
    used: &[GnssSatelliteId],
    systems: &[GnssSystem],
    x0: DVector<f64>,
    weights: &[f64],
) -> Result<PassEnd, SppError> {
    // Agreement-track stopping thresholds (see the SPP_SOLVER_* constants).
    let opts = SolveOptions {
        gtol: SPP_SOLVER_GTOL,
        ftol: SPP_SOLVER_FTOL,
        xtol: SPP_SOLVER_XTOL,
        max_nfev: SPP_SOLVER_MAX_NFEV,
    };
    let n_used = used.len();
    // The least-squares solver's residual closure cannot return an error, so the
    // first satellite that has no model, and the state it had none at, are
    // recorded here, and the closure returns a non-finite residual, which stops
    // the solve at once.
    let lost = std::cell::RefCell::new(None::<(GnssSatelliteId, DVector<f64>)>);
    let residual = |x: &DVector<f64>| -> DVector<f64> {
        match residual_unweighted_placed(
            eph,
            used,
            obs_by_id,
            x.as_slice(),
            inputs,
            model,
            placement,
        ) {
            Ok(r) => DVector::from_vec(r),
            Err(sat) => {
                let mut lost = lost.borrow_mut();
                if lost.is_none() {
                    *lost = Some((sat, x.clone()));
                }
                DVector::from_element(n_used, f64::NAN)
            }
        }
    };
    let problem =
        LeastSquaresProblem::with_weights(&residual, x0, DVector::from_row_slice(weights));
    let mut last_accepted: Option<DVector<f64>> = None;
    let result = least_squares::solve_trf_observed(&problem, &opts, linear_solve, &mut |x| {
        last_accepted = Some(x.clone());
    });
    // The recorded loss is the cause of the error the non-finite residual raised.
    if let Some((satellite, x)) = lost.into_inner() {
        let p_meas = obs_by_id
            .iter()
            .find(|(id, _)| *id == satellite)
            .map(|(_, p)| *p)
            .ok_or(SppError::EphemerisLost { satellite })?;
        let column = 3 + systems
            .iter()
            .position(|s| *s == clock_system(satellite.system))
            .unwrap_or(0);
        let env = model_env(eph, inputs, model, placement);
        let rx = [x[0], x[1], x[2]];
        if lost_grid_coverage(&env, inputs, satellite, rx, x[column], p_meas) {
            if let Some(accepted) = last_accepted {
                return Ok(PassEnd::CoverageLost(accepted));
            }
        }
        return Err(SppError::EphemerisLost { satellite });
    }
    Ok(PassEnd::Solved(result?))
}

/// The weighted design `sqrt(W) H` of [`weighted_normal_matrix`], whose rank,
/// singular values and condition number are the geometry diagnostics of a
/// solution: the design RTKLIB `estpos` solves with, and the one the DOP and
/// covariance are formed from.
pub(crate) fn weighted_design(
    los: &[LineOfSight],
    clock_columns: &[usize],
    n_params: usize,
    weights: &[f64],
) -> Option<DMatrix<f64>> {
    if los.len() != clock_columns.len() || los.len() != weights.len() {
        return None;
    }
    let mut design = DMatrix::zeros(los.len(), n_params);
    for k in 0..los.len() {
        let row = design_row(los[k], clock_columns[k], n_params)?;
        let sqrt_weight = weights[k].sqrt();
        for (j, value) in row.iter().enumerate() {
            design[(k, j)] = sqrt_weight * value;
        }
    }
    Some(design)
}

/// Lines of sight and residuals, index-aligned to a set of used satellites.
type UsedGeometry = (Vec<LineOfSight>, Vec<f64>);

/// The lines of sight and residuals of the satellites `used` at `state`, for a
/// set the selection at `state` was not made for. `None` when a satellite's line
/// of sight has left the augmentation grid there.
fn evaluate_used(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    model: SppModelRecipe,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
    used: &[GnssSatelliteId],
    state: &IterateState,
) -> Result<Option<UsedGeometry>, SppError> {
    let env = model_env(eph, inputs, model, placement);
    let mut los = Vec::with_capacity(used.len());
    let mut residuals_m = Vec::with_capacity(used.len());
    for &sat in used {
        let p_meas = inputs
            .observations
            .iter()
            .find(|o| o.satellite_id == sat)
            .map(|o| o.pseudorange_m)
            .ok_or(SppError::EphemerisLost { satellite: sat })?;
        let b = state.clock_m(clock_system(sat.system));
        let ionosphere = ionosphere_for(sat.system, inputs);
        let Some(m) = sat_model(&env, sat, state.rx_ecef_m, b, p_meas, ionosphere) else {
            if lost_grid_coverage(&env, inputs, sat, state.rx_ecef_m, b, p_meas) {
                return Ok(None);
            }
            return Err(SppError::EphemerisLost { satellite: sat });
        };
        los.push(line_of_sight(m.sat_rot_ecef_m, state.rx_ecef_m));
        residuals_m.push(p_meas - m.p_hat_m);
    }
    Ok(Some((los, residuals_m)))
}

/// The satellite set a solution reports and the geometry it reports it with.
struct FinalSet {
    used: Vec<GnssSatelliteId>,
    rejected: Vec<RejectedSat>,
    weights: Vec<f64>,
    lines_of_sight: Vec<LineOfSight>,
    residuals_m: Vec<f64>,
}

fn solve_tracked(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    model: SppModelRecipe,
    linear_solve: TrustRegionSolve,
    placement: Option<&BTreeMap<GnssSatelliteId, f64>>,
) -> Result<ReceiverSolution, SppError> {
    // One pseudorange per satellite. Reject duplicates deterministically (by
    // the smallest repeated id) so the result can never depend on observation
    // order and the parameter-count check (`used < n_params`, where
    // `n_params = 3 + n_clocks`) counts distinct satellites.
    let mut ids: Vec<GnssSatelliteId> =
        inputs.observations.iter().map(|o| o.satellite_id).collect();
    ids.sort_unstable();
    if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
        return Err(SppError::DuplicateObservation { satellite: w[0] });
    }

    let memo = TransmitStateMemo::new(eph, inputs.observations.len());
    let eph: &dyn EphemerisSource = &memo;
    let obs_by_id: Vec<(GnssSatelliteId, f64)> = inputs
        .observations
        .iter()
        .map(|o| (o.satellite_id, o.pseudorange_m))
        .collect();

    // RTKLIB `estpos` iterates `rescode` and a least-squares step: at each iterate
    // `rescode` selects the satellites (ephemeris, elevation mask, ionosphere
    // coverage, carrier), evaluates the corrections and the elevation variances,
    // and `estpos` takes the weighted least-squares step over the design
    // `[-e, 1]`, stopping once the step is below 1e-4 m and failing after `MAXITR`
    // iterations. Here every iterate is selected and weighted the same way. An
    // iterate whose selection is new runs the trust-region solve over it to that
    // selection's optimum, which brings a far or cold start close; an iterate whose
    // selection is the last one takes RTKLIB's step. The solve stops after the
    // first step below 1e-4 m, which it keeps, as `estpos` does.
    let mut state = IterateState::from_initial_guess(inputs.initial_guess);
    let mut iterations = 0usize;
    let mut passes = 0usize;
    // The satellites of the last pass, when its selection can end the solve.
    let mut last_used: Option<Vec<GnssSatelliteId>> = None;
    let settled = loop {
        let (sel, systems) = select_at_state(eph, inputs, model, placement, &state)?;
        if last_used.as_ref() == Some(&sel.used) {
            // The selection holds: the step RTKLIB takes here, from this
            // iterate's residuals and weights.
            let columns = clock_columns(&sel.used, &systems);
            let dx = rtklib_step(
                &sel.lines_of_sight,
                &columns,
                3 + systems.len(),
                &sel.weights,
                &sel.residuals_m,
            )
            .ok_or(SppError::Singular(
                least_squares::SolveError::SingularJacobian,
            ))?;
            if passes == MAX_SELECTION_PASSES {
                return Err(SppError::SelectionUnsettled { passes });
            }
            let x = state.parameters(&systems) + DVector::from_row_slice(&dx);
            state.update(&systems, &x);
            passes += 1;
            iterations += 1;
            if rtklib_step_norm(&dx, &[(3, &systems)]) < SELECTION_STEP_TOL_M {
                // `estpos` ends here and reports the selection it stepped with.
                // The reported geometry is taken at the position reached: the
                // selection there when it is the same one, else that selection's
                // satellites evaluated there.
                let post = select_at(eph, inputs, model, placement, state.rx_ecef_m, &|system| {
                    state.clock_m(system)
                });
                if post.used == sel.used {
                    break post;
                }
                if let Some((lines_of_sight, residuals_m)) =
                    evaluate_used(eph, inputs, model, placement, &sel.used, &state)?
                {
                    break Selection {
                        lines_of_sight,
                        residuals_m,
                        ..sel
                    };
                }
                // A satellite left the augmentation grid with the last step: the
                // selection changed.
                last_used = None;
                continue;
            }
            last_used = Some(sel.used);
            continue;
        }
        if passes == MAX_SELECTION_PASSES {
            return Err(SppError::SelectionUnsettled { passes });
        }
        // A new selection: the trust-region solve over it, from this iterate. Its
        // pass cannot end the solve; the next iterate is selected again.
        let end = solve_selected(
            eph,
            inputs,
            model,
            placement,
            linear_solve,
            &obs_by_id,
            &sel.used,
            &systems,
            state.parameters(&systems),
            &sel.weights,
        )?;
        passes += 1;
        match end {
            PassEnd::Solved(report) => {
                iterations += report.iterations;
                state.update(&systems, &report.x);
                last_used = Some(sel.used);
            }
            PassEnd::CoverageLost(x) => {
                // The last accepted iterate: its selection, if the same, is stepped
                // from rather than solved again.
                state.update(&systems, &x);
                last_used = Some(sel.used);
            }
        }
    };

    let mut status = Status::SelectionSettled;
    let mut outer_iterations = 0usize;
    let mut final_robust_scale_m: Option<f64> = None;
    let mut final_set = FinalSet {
        used: settled.used,
        rejected: settled.rejected,
        weights: settled.weights,
        lines_of_sight: settled.lines_of_sight,
        residuals_m: settled.residuals_m,
    };

    // Outer Huber/IRLS reweighting loop, ONLY on the robust path, warm-started
    // from the settled solve above. Each iteration takes the selection at the
    // current state, derives a floored MAD scale from its residuals, builds the
    // effective weight vector `elevation_weight * huber(r_i / s)` index-aligned
    // to that selection, and re-solves from the current state. It settles when the
    // position step drops below `outer_tol_m` and the selection at the new state
    // is the one solved with; a solve whose budget runs out first ends with
    // [`Status::OuterBudgetExhausted`] and has not converged. The reported set is
    // the one the last solve used, at the state it reached, with its effective
    // weights.
    if let Some(rc) = inputs.robust {
        let (mut sel, mut systems) = select_at_state(eph, inputs, model, placement, &state)?;
        let mut settled = false;
        // How the last solve ended; a least-squares step ends at its own target.
        let mut last_inner = Status::SelectionSettled;
        // After a coverage loss the selection at the last accepted iterate, when it
        // is the one solved over, is stepped from rather than solved again.
        let mut step_next = false;
        for _ in 0..rc.max_outer.saturating_sub(1) {
            let scale = mad_scale(&sel.residuals_m, rc.scale_floor_m).map_err(map_robust_error)?;
            let eff: Vec<f64> = sel
                .residuals_m
                .iter()
                .zip(sel.weights.iter())
                .map(|(&r, &bw)| bw * huber_weight(r / scale, rc.huber_k))
                .collect();
            let prev_rx = state.rx_ecef_m;
            outer_iterations += 1;
            if step_next {
                step_next = false;
                let columns = clock_columns(&sel.used, &systems);
                let dx = rtklib_step(
                    &sel.lines_of_sight,
                    &columns,
                    3 + systems.len(),
                    &eff,
                    &sel.residuals_m,
                )
                .ok_or(SppError::Singular(
                    least_squares::SolveError::SingularJacobian,
                ))?;
                let x = state.parameters(&systems) + DVector::from_row_slice(&dx);
                state.update(&systems, &x);
                iterations += 1;
                last_inner = Status::SelectionSettled;
            } else {
                match solve_selected(
                    eph,
                    inputs,
                    model,
                    placement,
                    linear_solve,
                    &obs_by_id,
                    &sel.used,
                    &systems,
                    state.parameters(&systems),
                    &eff,
                )? {
                    PassEnd::Solved(report) => {
                        iterations += report.iterations;
                        last_inner = report.status;
                        state.update(&systems, &report.x);
                    }
                    PassEnd::CoverageLost(x) => {
                        // The last accepted iterate, reported with the selection and
                        // elevation weights there until a reweighted solve or step
                        // replaces them.
                        state.update(&systems, &x);
                        let (next, next_systems) =
                            select_at_state(eph, inputs, model, placement, &state)?;
                        step_next = next.used == sel.used;
                        final_set = FinalSet {
                            used: next.used.clone(),
                            rejected: next.rejected.clone(),
                            weights: next.weights.clone(),
                            lines_of_sight: next.lines_of_sight.clone(),
                            residuals_m: next.residuals_m.clone(),
                        };
                        final_robust_scale_m = None;
                        sel = next;
                        systems = next_systems;
                        continue;
                    }
                }
            }
            // Position L2 step between successive outer solves.
            let dx = state.rx_ecef_m[0] - prev_rx[0];
            let dy = state.rx_ecef_m[1] - prev_rx[1];
            let dz = state.rx_ecef_m[2] - prev_rx[2];
            let dpos = (dx * dx + dy * dy + dz * dz).sqrt();
            let (next, next_systems) = select_at_state(eph, inputs, model, placement, &state)?;
            let same_set = next.used == sel.used;
            let solved_geometry = if same_set {
                Some((next.lines_of_sight.clone(), next.residuals_m.clone()))
            } else {
                evaluate_used(eph, inputs, model, placement, &sel.used, &state)?
            };
            (final_set, final_robust_scale_m) = match solved_geometry {
                Some((lines_of_sight, residuals_m)) => (
                    FinalSet {
                        used: sel.used,
                        rejected: sel.rejected,
                        weights: eff,
                        lines_of_sight,
                        residuals_m,
                    },
                    Some(scale),
                ),
                // A satellite left the augmentation grid: report the selection and
                // elevation weights at the state reached.
                None => (
                    FinalSet {
                        used: next.used.clone(),
                        rejected: next.rejected.clone(),
                        weights: next.weights.clone(),
                        lines_of_sight: next.lines_of_sight.clone(),
                        residuals_m: next.residuals_m.clone(),
                    },
                    None,
                ),
            };
            sel = next;
            systems = next_systems;
            if dpos < rc.outer_tol_m && same_set {
                settled = true;
                break;
            }
        }
        status = if outer_iterations == 0 {
            Status::SelectionSettled
        } else if !solve_converged(last_inner) {
            last_inner
        } else if settled {
            Status::SelectionSettled
        } else {
            Status::OuterBudgetExhausted
        };
    }

    // The clock columns of the reported set: the systems it was solved with.
    let systems = clock_systems(&final_set.used);
    let n_clocks = systems.len();
    let xs = state.parameters(&systems);
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

    // DOP and covariance from the reported geometry: the line-of-sight unit
    // vectors to the satellite positions the ranges were formed from, with the
    // reported weights. A single-system solve uses the 0-ULP four-state cofactor
    // inverse; a multi-system solve uses the general (3 + n_systems) inverse with
    // one clock column per GNSS (a deterministic geometry diagnostic, not a 0-ULP
    // target).
    let geo = geodetic_from_ecef(model.frame, [xs[0], xs[1], xs[2]]);
    let columns = clock_columns(&final_set.used, &systems);
    let clock_index: Vec<usize> = columns.iter().map(|column| column - 3).collect();
    let los = &final_set.lines_of_sight;
    // `systems` is the clock-column ordering: `clock_index[k] ==
    // systems.position(clock system of sat k)`, so `systems[c]` owns clock column
    // `c` (the same ordering `system_clocks_s` uses). The multi-system path is
    // handed that mapping and returns `Dop::system_tdops` already GNSS-tagged; the
    // single-system 0-ULP `dop` carries no constellation identity, so tag its
    // lone clock here with the one system in the solve.
    let dop_result = if n_clocks == 1 {
        dop(los, &final_set.weights, geo).ok().map(|mut d| {
            d.system_tdops = vec![(systems[0], d.tdop)];
            d
        })
    } else {
        dop_multi(
            los,
            &clock_index,
            &systems,
            n_clocks,
            &final_set.weights,
            geo,
        )
        .ok()
    };
    let n_params = xs.len();
    // The geometry diagnostics of the reported design, `sqrt(W) [-e, 1]` at the
    // reported position with the reported weights.
    let jacobian = weighted_design(los, &columns, n_params, &final_set.weights).ok_or(
        SppError::Singular(least_squares::SolveError::SingularJacobian),
    )?;
    let jacobian_svd = portable::svd(&jacobian, false, false);
    let singular_values: Vec<f64> = jacobian_svd
        .singular_values
        .iter()
        .map(|value| value.0)
        .collect();
    let diagnostics =
        singular_value_diagnostics(&singular_values, jacobian.nrows(), jacobian.ncols());
    if diagnostics.rank < n_params || dop_result.is_none() {
        return Err(SppError::Singular(
            least_squares::SolveError::SingularJacobian,
        ));
    }
    let gdop = dop_result
        .as_ref()
        .expect("full-rank SPP geometry has DOP")
        .gdop;
    // The solution's per-system TDOPs come straight from the now-tagged
    // `Dop::system_tdops`; empty when the converged geometry is rank-deficient.
    let system_tdops: Vec<(GnssSystem, f64)> = dop_result
        .as_ref()
        .map(|d| d.system_tdops.clone())
        .unwrap_or_default();
    let position_covariance =
        spp_position_covariance(los, &columns, n_params, &final_set.weights, geo).ok_or(
            SppError::Singular(least_squares::SolveError::SingularJacobian),
        )?;

    let converged = solve_converged(status);
    let metadata_used_count = final_set.used.len();
    let metadata_redundancy = redundancy(&systems, metadata_used_count);
    let geometry_quality = classify(
        diagnostics.rank,
        n_params,
        metadata_redundancy as i32,
        diagnostics.condition_number,
        gdop,
        false,
        GeometryQualityThresholds::default(),
    );

    Ok(ReceiverSolution {
        position,
        geodetic,
        rx_clock_s,
        rx_clock_drift_s_s: None,
        system_clocks_s,
        dop: dop_result,
        system_tdops,
        position_covariance,
        residuals_m: final_set.residuals_m,
        used_sats: final_set.used,
        rejected_sats: final_set.rejected,
        geometry_quality,
        metadata: SolutionMetadata {
            iterations,
            converged,
            status,
            ionosphere_applied: inputs.corrections.ionosphere,
            troposphere_applied: inputs.corrections.troposphere,
            outer_iterations,
            final_robust_scale_m,
            used_count: metadata_used_count,
            systems,
            redundancy: metadata_redundancy,
            raim_checkable: metadata_redundancy >= 1,
            ut1_degraded: None,
        },
    })
}

fn spp_position_covariance(
    los: &[LineOfSight],
    clock_columns: &[usize],
    n_params: usize,
    weights: &[f64],
    receiver: Wgs84Geodetic,
) -> Option<PositionCovariance> {
    let normal = weighted_normal_matrix(los, clock_columns, n_params, weights)?;
    let q = invert_symmetric_pd(&normal)?;
    let ecef_m2 = [
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ];
    let enu_m2 = crate::dop::rotate_covariance_ecef_to_enu_m2(ecef_m2, receiver).ok()?;
    Some(PositionCovariance { ecef_m2, enu_m2 })
}

/// Run SPP under the public API's language-independent validation/orchestration
/// policy.
///
/// Thin compatibility wrapper over the runtime strategy selector
/// ([`crate::estimation::strategies::estimate`]): it drives the shared
/// per-technique implementation `run` under the SPP reference strategy, which
/// resolves to the SPP reference recipe. The reference strategy always yields an
/// SPP solution or an SPP error, so the result is bit-identical to the recipe
/// driving `run` directly.
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
/// When `parallel` is disabled, the parallel entry point is equivalent to this
/// serial iterator.
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
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    #[cfg(feature = "parallel")]
    let epochs = epochs.par_iter();
    #[cfg(not(feature = "parallel"))]
    let epochs = epochs.iter();
    epochs
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
            // A UT1 refusal is a property of the ephemeris source at this
            // epoch, not of the seed, so no other seed can avoid it.
            Err(error @ SolvePolicyError::Solve(SppError::Ut1OutsideCoverage(_))) => {
                return Err(error)
            }
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
                MEAN_EARTH_RADIUS_M * r * libm::cos(theta),
                MEAN_EARTH_RADIUS_M * r * libm::sin(theta),
                MEAN_EARTH_RADIUS_M * z,
                0.0,
            ]
        })
        .collect()
}

/// A seed's solution is a candidate when its solve converged, or when it is a
/// robust solve whose reweighting spent its budget: the reweighting starts only
/// from a settled solve, which converged, so the seed reached the solution's
/// basin, and the position is where the reweighting left it. Every seed shares the
/// robust configuration, so refusing these would leave a robust coarse search
/// with no candidate where the plain solve returns one.
fn coarse_candidate_ended_well(status: Status) -> bool {
    solve_converged(status) || status == Status::OuterBudgetExhausted
}

fn select_coarse_candidate(candidates: &[ReceiverSolution]) -> Option<&ReceiverSolution> {
    candidates
        .iter()
        .filter(|solution| {
            coarse_candidate_ended_well(solution.metadata.status)
                && solution.metadata.redundancy >= 1
        })
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
/// rather than recomputing the formula.
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

pub(crate) fn validate_solve_inputs(inputs: &SolveInputs) -> Result<(), SppError> {
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

    /// The selection at the position and clocks of `solution`, the one it reports.
    pub fn selection_at_solution_for_test(
        eph: &dyn EphemerisSource,
        inputs: &SolveInputs,
        solution: &ReceiverSolution,
        model: SppModelRecipe,
    ) -> Selection {
        let clocks: Vec<(GnssSystem, f64)> = solution
            .system_clocks_s
            .iter()
            .map(|&(system, clock_s)| (system, clock_s * C_M_S))
            .collect();
        let reference = clocks[0].1;
        select_at(
            eph,
            inputs,
            model,
            None,
            solution.position.as_array(),
            &|system| {
                clocks
                    .iter()
                    .find(|(s, _)| *s == system)
                    .map_or(reference, |&(_, clock)| clock)
            },
        )
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

    /// The model of the pseudorange a receiver at `rx` with clock `b_m` measures from `sat`:
    /// the fixed point `P = p_hat(P)`. The reference model places the transmission epoch
    /// from the pseudorange itself (`t_rx - P / c - dts`, RTKLIB `satposs`), so a synthetic
    /// observation is self-consistent only where the pseudorange that places the satellite
    /// is the one the model predicts. Each pass changes `P` by the range rate over `c`
    /// times the previous change, about `1e-5`, so a few passes reach it; a model whose
    /// transmission epoch does not depend on `P` returns after one.
    pub fn self_consistent_model_for_test(
        env: &SatModelEnv,
        sat: GnssSatelliteId,
        rx: [f64; 3],
        b_m: f64,
        klobuchar: &KlobucharCoeffs,
    ) -> Option<SatModel> {
        let ionosphere = SppIonosphere::Klobuchar(*klobuchar);
        let mut p_meas = 22_000_000.0 + b_m;
        let mut model = sat_model(env, sat, rx, b_m, p_meas, ionosphere)?;
        for _ in 0..8 {
            if model.p_hat_m.to_bits() == p_meas.to_bits() {
                break;
            }
            p_meas = model.p_hat_m;
            model = sat_model(env, sat, rx, b_m, p_meas, ionosphere)?;
        }
        assert!(
            (model.p_hat_m - p_meas).abs() < 1.0e-6,
            "{sat}: synthetic pseudorange did not settle: {} m from its own placement",
            model.p_hat_m - p_meas
        );
        Some(model)
    }

    /// [`solve`] with the measurement model `model` in place of the reference one.
    pub fn solve_with_model_for_test(
        eph: &dyn EphemerisSource,
        inputs: &SolveInputs,
        with_geodetic: bool,
        model: SppModelRecipe,
    ) -> Result<ReceiverSolution, SppError> {
        validate_solve_inputs(inputs)?;
        solve_inner(
            eph,
            inputs,
            with_geodetic,
            model,
            TrustRegionSolve::NalgebraLu,
        )
    }

    /// At one state, the SPP model, which places the transmission epoch from the pseudorange
    /// as RTKLIB `satposs` does and ranges it with `geodist`, differs from the geometric
    /// light-time model the reference recipe was computed with (`replay_env`) only through
    /// the transmission epoch:
    ///
    /// - its epoch is `satposs`'s, `(t_rx - P / c) - dts` with `dts` the source's clock at
    ///   `t_rx - P / c`, bit for bit, and its state and clock are read there;
    /// - its satellite position is the source's at that epoch, not rotated, and its range is
    ///   `geodist` of it, `|r_s - r_r| + ω (x_s y_r - y_s x_r) / c`, bit for bit;
    /// - the geometric model read its state at `t_rx - |r_s - r_r| / c`, so the RTKLIB epoch
    ///   is that one moved by `(|r_s - r_r| - P) / c - dts`: the receiver clock, the media
    ///   delays and the Sagnac range the geometric light time leaves out. The two epochs are
    ///   doubles near 6.5e8 s, whose step is 2^-23 s, and the geometric iteration's first
    ///   step carries a further `rdot · b / c²`; three steps bound both;
    /// - the range moves by the range rate times the epoch difference, within 0.5 mm: for a
    ///   GPS orbit RTKLIB's first-order Sagnac term and the closed-form rotation differ by
    ///   about 0.1 mm;
    /// - the clock moves by its drift over that time, far below a millimetre;
    /// - the line of sight turns by at most the Earth's rotation over the flight time
    ///   applied to the satellite (the replay rotates the satellite into the reception
    ///   frame, RTKLIB takes elevation and azimuth from the unrotated vector) plus the
    ///   satellite's motion over the epoch difference, about 6 µrad, and the media delays
    ///   move with it, within 5 mm down to a few degrees of elevation.
    // The state is the model's own argument list plus a label.
    #[allow(clippy::too_many_arguments)]
    pub fn assert_only_the_transmit_epoch_differs(
        source: &dyn EphemerisSource,
        replay_env: &SatModelEnv<'_>,
        sat: GnssSatelliteId,
        rx: [f64; 3],
        b: f64,
        p_meas: f64,
        klobuchar: &KlobucharCoeffs,
        label: &str,
    ) {
        let rtklib_env = SatModelEnv {
            model: SppModelRecipe::reference(),
            ..*replay_env
        };
        let g = sat_model_for_test(replay_env, sat, rx, b, p_meas, klobuchar)
            .expect("geometric light-time model");
        let r = sat_model_for_test(&rtklib_env, sat, rx, b, p_meas, klobuchar)
            .expect("RTKLIB placement model");

        let t_rx = replay_env.t_rx_j2000_s;
        let clock_epoch = t_rx - p_meas / C_M_S;
        let dts = EphemerisSource::try_transmit_epoch_clock_s(source, sat, clock_epoch, t_rx)
            .expect("no refusal")
            .expect("a clock at t_rx - P / c")
            .value;
        let t_tx = clock_epoch - dts;
        assert_eq!(
            r.t_tx_j2000_s.to_bits(),
            t_tx.to_bits(),
            "{label}: satposs epoch"
        );
        assert_eq!(
            r.clock_epoch_j2000_s.to_bits(),
            t_tx.to_bits(),
            "{label}: state epoch"
        );
        let pos = EphemerisSource::try_position_clock_group_delay_selected_at_j2000_s(
            source, sat, t_tx, t_rx,
        )
        .expect("no refusal")
        .expect("state at the transmission epoch")
        .value
        .0;
        assert_eq!(
            r.sat_ecef_m.map(f64::to_bits),
            pos.map(f64::to_bits),
            "{label}: position"
        );
        assert_eq!(
            r.sat_rot_ecef_m.map(f64::to_bits),
            pos.map(f64::to_bits),
            "{label}: not rotated"
        );
        let d = [pos[0] - rx[0], pos[1] - rx[1], pos[2] - rx[2]];
        let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let geodist = distance + OMEGA_E_DOT_RAD_S * (pos[0] * rx[1] - pos[1] * rx[0]) / C_M_S;
        assert_eq!(r.rho_m.to_bits(), geodist.to_bits(), "{label}: geodist");

        let g_d = [
            g.sat_ecef_m[0] - rx[0],
            g.sat_ecef_m[1] - rx[1],
            g.sat_ecef_m[2] - rx[2],
        ];
        let g_distance = (g_d[0] * g_d[0] + g_d[1] * g_d[1] + g_d[2] * g_d[2]).sqrt();
        let dt = r.clock_epoch_j2000_s - g.clock_epoch_j2000_s;
        let expected_dt = (g_distance - p_meas) / C_M_S - dts;
        let epoch_step = 2.0_f64.powi(-23);
        assert!(
            (dt - expected_dt).abs() <= 3.0 * epoch_step,
            "{label}: epochs {dt} s apart where the pseudorange puts them {expected_dt} s apart"
        );

        let selected_position = |t: f64| {
            EphemerisSource::try_position_clock_group_delay_selected_at_j2000_s(
                source, sat, t, t_rx,
            )
            .expect("no refusal")
            .expect("state near the transmission epoch")
            .value
            .0
        };
        let plus = selected_position(t_tx + 0.5);
        let minus = selected_position(t_tx - 0.5);
        let velocity = [plus[0] - minus[0], plus[1] - minus[1], plus[2] - minus[2]];
        let range_rate = (d[0] * velocity[0] + d[1] * velocity[1] + d[2] * velocity[2]) / distance;
        let moved = (r.rho_m - g.rho_m) - range_rate * dt;
        assert!(
            moved.abs() <= 5.0e-4,
            "{label}: range moved {} m beyond rdot·dt = {} m",
            moved,
            range_rate * dt
        );
        assert!(
            (C_M_S * (r.dt_sat_s - g.dt_sat_s)).abs() <= 1.0e-5,
            "{label}: clock"
        );
        let speed =
            (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                .sqrt();
        let tau = g_distance / C_M_S;
        let turn_rad = (OMEGA_E_DOT_RAD_S * tau * libm::hypot(pos[0], pos[1]) + speed * dt.abs())
            / distance
            * 1.01
            + 1.0e-9;
        assert!(
            (r.el_rad - g.el_rad).abs() <= turn_rad,
            "{label}: elevation moved {} rad beyond the {turn_rad} rad the line of sight turns",
            r.el_rad - g.el_rad
        );
        let az_diff = (r.az_rad - g.az_rad + std::f64::consts::PI)
            .rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        assert!(
            (az_diff * libm::cos(g.el_rad)).abs() <= turn_rad,
            "{label}: azimuth moved {az_diff} rad beyond the {turn_rad} rad the line of sight turns"
        );
        let media_tolerance_m = 5.0e-3;
        assert!(
            (r.iono_m - g.iono_m).abs() <= media_tolerance_m,
            "{label}: iono moved {} m",
            r.iono_m - g.iono_m
        );
        assert!(
            (r.tropo_m - g.tropo_m).abs() <= media_tolerance_m,
            "{label}: tropo moved {} m",
            r.tropo_m - g.tropo_m
        );
        let p_hat_moved = (r.p_hat_m - g.p_hat_m) - range_rate * dt;
        let media_moved = (r.iono_m - g.iono_m) + (r.tropo_m - g.tropo_m);
        assert!(
            (p_hat_moved - media_moved).abs() <= 6.0e-4,
            "{label}: p_hat moved {} m beyond rdot·dt and the media delays",
            p_hat_moved - media_moved
        );
    }

    pub fn sat_model_with_ionosphere_for_test(
        env: &SatModelEnv,
        sat: GnssSatelliteId,
        rx: [f64; 3],
        b_m: f64,
        p_meas: f64,
        ionosphere: SppIonosphere<'_>,
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
