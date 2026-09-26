//! Continuity attestation for precise-ephemeris sample series.
//!
//! A merged orbit product is assembled per `(epoch, satellite)` cell from
//! several analysis centers. That is exactly the operation that can splice two
//! physically inconsistent arcs together while every input remains individually
//! well-formed. This module attests against that: it takes an ordered sample
//! series and either attests that it is continuous or reports each violation
//! with the epochs, the interval, and the magnitude that exceeded its bound.
//!
//! # Two checks, two jobs
//!
//! A single displacement-per-interval gate cannot do this work alone, and the
//! reason is quantitative rather than stylistic. Measured on a real GFZ ultra
//! product (satellite G01, 576 epochs at 300 s), adjacent-epoch ECEF chord
//! distances run 827-956 km, so the implied chord speed is 2757-3187 m/s. A
//! defensible upper bound for the class sits near 6 km/s (see [`OrbitClass`]),
//! which leaves several hundred kilometres of displacement per epoch pair
//! underneath the bound. A 500 m splice - a serious defect - moves the implied
//! speed by under 2 m/s, roughly 0.05% of the observed chord speed. It is
//! invisible to a speed gate by four orders of magnitude.
//!
//! So the two checks are deliberately separated:
//!
//! - [`ContinuityCheck::SpeedBound`] is a *gross corruption* gate. Its bound is
//!   a true physical upper bound for the orbit class (see [`OrbitClass`]), so it
//!   cannot false-positive on real data; it catches a record from the wrong
//!   satellite, the wrong day, or a corrupt field. It is insensitive by
//!   construction and is not asked to be otherwise.
//! - [`ContinuityCheck::HoldOutResidual`] supplies the sensitivity. Each interior
//!   sample is held out, predicted from its neighbours through the same
//!   sliding-window Lagrange substrate the product's own interpolator uses
//!   ([`super::interp::interpolate_precise_state`]), and compared against the
//!   stored record. On a clean arc the residual is the interpolator's own
//!   error - centimetres for GNSS MEO at 5-15 minute spacing. At a splice it
//!   jumps to the magnitude of the splice, which is what localizes the offending
//!   epoch pair.
//!
//! Run both. The bound gate is nearly free and rules out nonsense; the residual
//! check is the one that finds a spliced arc.
//!
//! # Frame
//!
//! Samples are ITRF/IGS ECEF, matching [`PreciseEphemerisSample`]. The bound is
//! therefore an *earth-fixed* bound, and [`OrbitClass`] derives it as such. Using
//! an inertial orbital speed here would be a category error: for a prograde MEO
//! satellite the earth-fixed speed is materially lower than the inertial speed
//! (the measurement above versus an inertial 3874 m/s for GPS), and for a
//! geostationary satellite it is near zero.
//!
//! # Ordering is this module's responsibility
//!
//! [`check_continuity`] sorts internally. A caller-ordered sequence that is
//! trusted is the failure this module exists to prevent, so shuffled input and
//! sorted input produce the identical verdict - there is a test that pins
//! exactly that. One ordered structure ([`OrderedSeries`]) feeds every check, and
//! it is built so that a zero or negative interval is *unrepresentable* in the
//! comparison path rather than merely rejected: duplicate epochs are split out
//! into [`ContinuityDefect::DuplicateEpoch`] during construction, after which
//! adjacent pairs are strictly increasing by construction and the pair iterator
//! is the only way to reach a comparison.
//!
//! Duplicates are reported, never silently deduplicated: two records for one
//! epoch is a real defect of the data, and which of them is "the" sample is not
//! this module's call to make.
//!
//! # This reports; it does not refuse
//!
//! [`check_continuity`] returns a [`ContinuityReport`] whether or not the series
//! is continuous. A caller may legitimately want the product together with its
//! defects - refusing is the caller's decision, made by consulting
//! [`ContinuityReport::attested`]. The bounds themselves are physical and are
//! never inferred from the data being validated: a check that can be widened
//! until it passes is not a check.
//!
//! The options are refused, though, when a bound is not a finite non-negative
//! number ([`ContinuityOptionsError`]). No sample exceeds a NaN or infinite
//! bound, so such a check would find nothing and report the series attested
//! without having tested it.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::astro::constants::earth::OMEGA_E_DOT_RAD_S;
use crate::astro::constants::MU_EARTH;
use crate::constants::KM_TO_M;
use crate::id::GnssSatelliteId;
use crate::sp3::interp::{
    gather_sp3_precise_series, instant_to_j2000_seconds, interpolate_precise_state,
    nominal_positive_spacing, precise_node_j2000_seconds, precise_node_j2000_seconds_from_instant,
    select_position_nodes, selectable_reach_s, sp3_epoch_j2000_seconds, PreciseQuery,
    Sp3InterpolationOptions, NEVILLE_POINTS,
};
use crate::sp3::samples::PreciseEphemerisSample;
use crate::sp3::Sp3;
use crate::{Error, Result};

/// Inclusive evaluation window on the SP3 seconds-since-J2000 axis.
///
/// This identifies epochs the caller intends to evaluate. Continuity findings
/// outside the window can still influence it through the interpolation
/// neighbourhood, which is why queries also require a [`StencilExtent`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochWindow {
    from_j2000_s: f64,
    through_j2000_s: f64,
}

impl EpochWindow {
    /// Construct an inclusive evaluation window.
    ///
    /// Both endpoints must be finite and `from_j2000_s` must not be later than
    /// `through_j2000_s`.
    pub fn new(from_j2000_s: f64, through_j2000_s: f64) -> Result<Self> {
        if !from_j2000_s.is_finite() || !through_j2000_s.is_finite() {
            return Err(Error::InvalidInput(
                "SP3 continuity window endpoints must be finite".to_string(),
            ));
        }
        if from_j2000_s > through_j2000_s {
            return Err(Error::InvalidInput(
                "SP3 continuity window start must not follow its end".to_string(),
            ));
        }
        Ok(Self {
            from_j2000_s,
            through_j2000_s,
        })
    }

    /// Inclusive first evaluation epoch, seconds since J2000.
    pub fn from_j2000_s(self) -> f64 {
        self.from_j2000_s
    }

    /// Inclusive last evaluation epoch, seconds since J2000.
    pub fn through_j2000_s(self) -> f64 {
        self.through_j2000_s
    }
}

/// Time reach of the SP3 position interpolator's sliding node stencil.
///
/// Construct this with [`StencilExtent::for_sp3`]. The extent is derived from
/// the product interval and the same 11-node constant used by the position
/// interpolator, rather than accepted as a caller-supplied duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilExtent {
    before_s: f64,
    after_s: f64,
}

impl StencilExtent {
    /// Derive the interpolation reach for an SP3 product.
    ///
    /// The position interpolator selects up to 11 nodes from each satellite's
    /// own node series, not from the product's header grid, and it serves a
    /// query up to one nominal spacing outside that series or across a coverage
    /// gap. The farthest selected node can therefore sit a full window span
    /// plus one nominal spacing from the query, and the window span is measured
    /// from the satellite's actual nodes: a run tolerates gaps up to 1.5 times
    /// its nominal spacing, so eleven nodes at 0, 600, 1500, 2400, ..., 8700 s
    /// span 8,700 s although their nominal spacing is 600 s.
    ///
    /// The reach reported here is the largest such value over the satellites
    /// in the product, computed with the interpolator's own run and window
    /// rules, with a floor of eleven header intervals for products whose
    /// satellites carry fewer than two nodes. Understating the reach errs in
    /// the unsafe direction: a window-scoped verdict such as
    /// [`ContinuityReport::verdict_for_window`] would accept a window whose
    /// interpolation selects a node the defect sits on.
    ///
    /// A non-finite or non-positive declared interval is rejected. The interval
    /// is used here only as a length for the reach floor, not as a grid step,
    /// so it need not be a whole number of ticks.
    pub fn for_sp3(sp3: &Sp3) -> Result<Self> {
        let interval_s = sp3.header.epoch_interval_s;
        if !interval_s.is_finite() || interval_s <= 0.0 {
            return Err(Error::InvalidInput(
                "SP3 stencil extent requires a positive finite epoch interval".to_string(),
            ));
        }
        if !sp3
            .epochs_j2000_seconds()
            .first()
            .is_some_and(|epoch| epoch.is_finite())
        {
            return Err(Error::InvalidInput(
                "SP3 stencil extent requires at least one representable epoch".to_string(),
            ));
        }

        // One pass over the epochs, collecting each satellite's node series.
        // `epochs` is public and may be longer than the private node table, so
        // the table is consulted by lookup rather than by index.
        let mut series: BTreeMap<GnssSatelliteId, Vec<f64>> = BTreeMap::new();
        for (idx, epoch) in sp3.epochs.iter().enumerate() {
            let Some(nodes_at_epoch) = sp3.interp_raw.get(idx) else {
                break;
            };
            let Some(seconds) = sp3_epoch_j2000_seconds(sp3, idx, epoch) else {
                continue;
            };
            let seconds = precise_node_j2000_seconds(seconds);
            for sat in nodes_at_epoch.keys() {
                series.entry(*sat).or_default().push(seconds);
            }
        }

        let gap_threshold_factor = sp3.interpolation.gap_threshold_factor();
        let mut reach_s = NEVILLE_POINTS as f64 * interval_s;
        for nodes in series.values() {
            if let Some(reach) = selectable_reach_s(nodes, gap_threshold_factor) {
                reach_s = reach_s.max(reach);
            }
        }
        Ok(Self {
            before_s: reach_s,
            after_s: reach_s,
        })
    }

    /// Reach before an evaluated epoch, seconds.
    ///
    /// The widest selectable window span in the product plus one nominal
    /// spacing: the farthest a selected node can sit behind a query, at a run
    /// end or when a query one spacing past a run is still served.
    pub fn before_s(self) -> f64 {
        self.before_s
    }

    /// Reach after an evaluated epoch, seconds.
    ///
    /// The widest selectable window span in the product plus one nominal
    /// spacing: the farthest a selected node can sit ahead of a query, at a run
    /// start or when a query one spacing before a run is anchored to it.
    pub fn after_s(self) -> f64 {
        self.after_s
    }

    /// Time span that can contain a node selected for any query in `window`.
    ///
    /// Measured from the window bounds themselves. An earlier version snapped
    /// each bound to the header grid first; on the upper side that snap moved
    /// the bound earlier and let a selected node fall outside it.
    fn influence_bounds(self, window: EpochWindow) -> (f64, f64) {
        (
            window.from_j2000_s - self.before_s,
            window.through_j2000_s + self.after_s,
        )
    }
}

/// Each satellite's position nodes in a product, for asking exactly which of
/// them an evaluation window's interpolations select.
///
/// [`StencilExtent`] bounds that reach by the widest window the product can
/// select, which is safe but wide: near a run edge a query reaches ten spacings
/// back, so the bound reaches that far from every query in every window. This
/// answers exactly, by applying the position interpolator's own serving and
/// node-selection rule to each satellite's node series. Merge continuity
/// verdicts ([`super::combine::MergeContinuityReport::verdict_for_window`])
/// use it.
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolationNodes {
    series: BTreeMap<GnssSatelliteId, Vec<f64>>,
    gap_threshold_factor: f64,
}

impl InterpolationNodes {
    /// The position node series of every satellite in `sp3`, read under the
    /// product's own interpolation options, as its interpolator reads them.
    pub fn for_sp3(sp3: &Sp3) -> Self {
        let mut satellites: Vec<GnssSatelliteId> = sp3
            .interp_raw
            .iter()
            .flat_map(|nodes| nodes.keys().copied())
            .collect();
        satellites.sort_unstable();
        satellites.dedup();
        let series = satellites
            .into_iter()
            .map(|sat| (sat, gather_sp3_precise_series(sp3, sat).x))
            .collect();
        Self {
            series,
            gap_threshold_factor: sp3.interpolation.gap_threshold_factor(),
        }
    }

    /// Epochs of the nodes that some position query of `sat` in `window`
    /// selects, ascending and seconds since J2000; empty when no query in the
    /// window is served for it.
    ///
    /// The selection changes only where a query crosses a node, one nominal
    /// spacing either side of a node (the serving limits, the coverage-gap
    /// limits and the anchoring to a run beyond a gap), so the rule is
    /// evaluated at each such point inside the window, at the window's ends
    /// and between each consecutive pair of them, which covers every query.
    pub fn selected_nodes(&self, sat: GnssSatelliteId, window: EpochWindow) -> Vec<f64> {
        let Some(x) = self.series.get(&sat) else {
            return Vec::new();
        };
        let (from, through) = (window.from_j2000_s(), window.through_j2000_s());
        let mut points = vec![from, through];
        if let Some(nominal) = nominal_positive_spacing(x) {
            for &node in x {
                for point in [node - nominal, node, node + nominal] {
                    if point > from && point < through {
                        points.push(point);
                    }
                }
            }
        }
        points.sort_by(f64::total_cmp);
        points.dedup();
        let midpoints: Vec<f64> = points
            .windows(2)
            .map(|pair| pair[0] + (pair[1] - pair[0]) / 2.0)
            .collect();

        let mut selected: BTreeSet<usize> = BTreeSet::new();
        for query in points.into_iter().chain(midpoints) {
            let query = PreciseQuery::at(query).j2000_s();
            if let Ok(nodes) =
                select_position_nodes(x.len(), query, self.gap_threshold_factor, |i| x[i])
            {
                selected.extend(nodes);
            }
        }
        selected.into_iter().map(|index| x[index]).collect()
    }
}

/// Window-scoped continuity decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowContinuityDecision {
    /// No recorded finding can enter an interpolation stencil for the window.
    Accept,
    /// At least one recorded finding can enter an interpolation stencil.
    Refuse,
}

/// A window-scoped decision that retains both influencing and global findings.
///
/// `all_defects` remains available for logging findings the caller accepted
/// around. Merge reports additionally populate `influencing_splices` and
/// `all_splices` through their own verdict helper.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowContinuityVerdict<'a> {
    /// Accept or refuse the requested evaluation window.
    pub decision: WindowContinuityDecision,
    /// Defects whose time support intersects a stencil used by the window.
    pub influencing_defects: Vec<&'a ContinuityDefect>,
    /// Contributor-changing violations influencing the window.
    pub influencing_splices: Vec<&'a super::combine::MergeContinuityViolation>,
    /// Every defect in the underlying continuity report.
    pub all_defects: &'a [ContinuityDefect],
    /// Every contributor-changing violation in the merge report.
    pub all_splices: Vec<&'a super::combine::MergeContinuityViolation>,
}

impl WindowContinuityVerdict<'_> {
    /// Whether the requested window is accepted.
    pub fn accepted(&self) -> bool {
        self.decision == WindowContinuityDecision::Accept
    }
}

/// Orbit class supplying a physical earth-fixed displacement bound.
///
/// Each bound is `sqrt(mu / a_min) + omega_earth * r_max`, the inertial speed at
/// the class's tightest published semi-major axis plus the largest possible
/// earth-rotation transport term at its widest radius. That sum is a true upper
/// bound on earth-fixed speed for any geometry in the class, so the gate cannot
/// false-positive on physically real data. It is correspondingly loose - see the
/// module docs for why that is the correct trade for this check and where the
/// sensitivity actually comes from.
///
/// Bounds are constants of the orbit class. None of them is derived from the
/// series being validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitClass {
    /// GNSS MEO: GLONASS (a ~ 25 510 km) through Galileo (a ~ 29 600 km),
    /// covering GPS, BeiDou MEO, NavIC MEO, and QZSS's MEO-like arcs.
    MeoGnss,
    /// Geostationary and inclined-geosynchronous (a ~ 42 164 km), including
    /// BeiDou GEO/IGSO and QZSS.
    Geosynchronous,
    /// Low earth orbit from a ~ 6 678 km (300 km altitude) upward.
    Leo,
}

impl OrbitClass {
    /// Tightest published semi-major axis for the class, meters.
    const fn min_semi_major_axis_m(self) -> f64 {
        match self {
            Self::MeoGnss => 25_510_000.0,
            Self::Geosynchronous => 42_164_000.0,
            Self::Leo => 6_678_000.0,
        }
    }

    /// Widest radius for the class, meters, for the earth-rotation term.
    const fn max_radius_m(self) -> f64 {
        match self {
            Self::MeoGnss => 29_600_000.0,
            Self::Geosynchronous => 42_164_000.0,
            Self::Leo => 8_378_000.0,
        }
    }

    /// Physical earth-fixed speed bound for the class, meters per second.
    pub fn max_earth_fixed_speed_m_s(self) -> f64 {
        let mu_m3_s2 = MU_EARTH * KM_TO_M * KM_TO_M * KM_TO_M;
        (mu_m3_s2 / self.min_semi_major_axis_m()).sqrt() + OMEGA_E_DOT_RAD_S * self.max_radius_m()
    }
}

/// Which checks to run, and with what bounds.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ContinuityOptions {
    /// Earth-fixed speed bound for the adjacent-pair gate. `None` disables the
    /// gate.
    pub speed_bound: Option<SpeedBound>,
    /// Hold-out interpolation residual tolerance in meters. `None` disables the
    /// residual check.
    ///
    /// This is the sensitive check. A tolerance well above the interpolator's own
    /// error at the product's sampling (centimetres for GNSS MEO at 5-15 minutes)
    /// and well below the smallest splice worth reporting is the useful range;
    /// 1.0 m is a defensible default for a merged GNSS orbit product.
    pub residual_tolerance_m: Option<f64>,
    /// How the hold-out replay reads the retained node series.
    ///
    /// Nothing sets this from a product: [`check_continuity`] sees samples,
    /// not the product they came from, so it defaults to the default policy
    /// however the product was configured. A product read with a non-default
    /// gap threshold is checked under that threshold only if the caller
    /// copies `product.interpolation_options()` here; otherwise the replay
    /// splits runs at 1.5 nominal spacings while the interpolator in use
    /// does not, and the two disagree about which nodes a prediction uses.
    pub interpolation: Sp3InterpolationOptions,
}

/// Source of the adjacent-pair speed bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedBound {
    /// Derive the bound from the orbit class.
    OrbitClass(OrbitClass),
    /// An explicit caller-supplied earth-fixed bound, meters per second.
    ExplicitMaxSpeed(f64),
}

/// Why a [`ContinuityOptions`] bound is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContinuityOptionRejection {
    /// The bound is NaN or an infinity. No implied speed or residual exceeds
    /// it, so the check would find nothing and attest a series it never
    /// tested.
    NotFinite,
    /// The bound is below zero. A speed or a distance bound is a magnitude,
    /// and every sample would exceed a negative one.
    Negative,
}

impl fmt::Display for ContinuityOptionRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "it is not finite"),
            Self::Negative => write!(f, "it is negative"),
        }
    }
}

/// A [`ContinuityOptions`] bound refused: the field that held it, the value as
/// supplied, and why.
///
/// A bound must be a finite number at least zero. `None` is how a check is
/// disabled; a NaN or infinite bound is not.
#[derive(Debug, Clone, Copy)]
pub struct ContinuityOptionsError {
    /// The field that held the value: `"speed_bound"` for
    /// [`SpeedBound::ExplicitMaxSpeed`], or `"residual_tolerance_m"`.
    pub field: &'static str,
    /// The value as supplied.
    pub value: f64,
    /// Why it is refused.
    pub reason: ContinuityOptionRejection,
}

/// Values compare by their bits, so a refused NaN equals itself and the error
/// can sit in `Eq` error enums.
impl PartialEq for ContinuityOptionsError {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.value.to_bits() == other.value.to_bits()
            && self.reason == other.reason
    }
}

impl Eq for ContinuityOptionsError {}

impl fmt::Display for ContinuityOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "continuity {} {} is refused: {}",
            self.field, self.value, self.reason
        )
    }
}

impl std::error::Error for ContinuityOptionsError {}

/// The continuity bound rule: a finite number at least zero.
fn checked_bound(
    field: &'static str,
    value: f64,
) -> core::result::Result<(), ContinuityOptionsError> {
    let reason = if !value.is_finite() {
        ContinuityOptionRejection::NotFinite
    } else if value < 0.0 {
        ContinuityOptionRejection::Negative
    } else {
        return Ok(());
    };
    Err(ContinuityOptionsError {
        field,
        value,
        reason,
    })
}

impl SpeedBound {
    fn value_m_s(self) -> f64 {
        match self {
            Self::OrbitClass(class) => class.max_earth_fixed_speed_m_s(),
            Self::ExplicitMaxSpeed(bound) => bound,
        }
    }
}

impl ContinuityOptions {
    /// Build continuity settings from the optional speed and residual checks.
    ///
    /// `None` disables the corresponding check; assign `Some` bounds when the
    /// check is required. An explicit speed bound or a residual tolerance that
    /// is not a finite number at least zero is refused
    /// ([`ContinuityOptions::validate`]).
    pub fn new(
        speed_bound: Option<SpeedBound>,
        residual_tolerance_m: Option<f64>,
    ) -> core::result::Result<Self, ContinuityOptionsError> {
        let options = Self {
            speed_bound,
            residual_tolerance_m,
            interpolation: Sp3InterpolationOptions::DEFAULT,
        };
        options.validate()?;
        Ok(options)
    }

    /// Check every bound these options hold: an explicit speed bound and the
    /// residual tolerance must each be a finite number at least zero.
    ///
    /// The fields are public, so a value set after construction is checked
    /// again wherever the options are used: [`check_continuity`] and a merge
    /// with [`MergeOptions::verify_continuity`](crate::ephemeris::MergeOptions::verify_continuity)
    /// refuse options this refuses. A class-derived bound is always finite, and
    /// [`Sp3InterpolationOptions`] is validated when it is built.
    pub fn validate(&self) -> core::result::Result<(), ContinuityOptionsError> {
        if let Some(SpeedBound::ExplicitMaxSpeed(bound_m_s)) = self.speed_bound {
            checked_bound("speed_bound", bound_m_s)?;
        }
        if let Some(tolerance_m) = self.residual_tolerance_m {
            checked_bound("residual_tolerance_m", tolerance_m)?;
        }
        Ok(())
    }

    /// Both checks, with the class bound and a 1 m residual tolerance.
    pub fn for_orbit_class(class: OrbitClass) -> Self {
        Self {
            speed_bound: Some(SpeedBound::OrbitClass(class)),
            residual_tolerance_m: Some(1.0),
            interpolation: Sp3InterpolationOptions::default(),
        }
    }

    /// The same settings, with the hold-out replay reading node series under
    /// `interpolation`.
    #[must_use]
    pub fn with_interpolation_options(mut self, interpolation: Sp3InterpolationOptions) -> Self {
        self.interpolation = interpolation;
        self
    }
}

/// One continuity defect. Every variant names the satellite, and each locates
/// itself in time where it has an epoch.
#[derive(Debug, Clone, PartialEq)]
pub enum ContinuityDefect {
    /// Two or more samples share one epoch. Reported, never deduplicated: which
    /// record is authoritative is not this module's decision.
    DuplicateEpoch {
        /// The satellite.
        sat: GnssSatelliteId,
        /// The repeated epoch, seconds since J2000.
        epoch_j2000_s: f64,
        /// How many samples carried this epoch (>= 2).
        occurrences: usize,
    },
    /// A satellite carried a single usable sample, so no adjacent pair and no
    /// hold-out prediction exist. Not a pass.
    SingleSampleSeries {
        /// The satellite.
        sat: GnssSatelliteId,
    },
    /// A sample no check can use: its epoch names no finite J2000 second, or
    /// its position is not finite. It takes no part in any comparison, and its
    /// neighbours are compared across the hole it leaves, so it is reported
    /// rather than dropped. Not a pass, even when every sample of a satellite
    /// is unusable.
    UnusableSample {
        /// The satellite.
        sat: GnssSatelliteId,
        /// Index of the sample in the slice given to [`check_continuity`].
        sample_index: usize,
        /// The sample's epoch, seconds since J2000, when it names one.
        epoch_j2000_s: Option<f64>,
        /// Why no check can use it.
        reason: UnusableSampleReason,
    },
    /// An adjacent pair implies an earth-fixed speed above the physical bound.
    SpeedBound {
        /// The satellite.
        sat: GnssSatelliteId,
        /// Earlier epoch of the pair, seconds since J2000.
        from_j2000_s: f64,
        /// Later epoch of the pair, seconds since J2000.
        to_j2000_s: f64,
        /// Elapsed interval, seconds. Strictly positive by construction.
        interval_s: f64,
        /// 3D chord displacement over the interval, meters.
        displacement_m: f64,
        /// Implied earth-fixed chord speed, meters per second.
        implied_speed_m_s: f64,
        /// The bound it exceeded, meters per second.
        bound_m_s: f64,
    },
    /// A sample disagrees with the arc its neighbours describe. This is the
    /// splice detector: `preceding_j2000_s` and `epoch_j2000_s` bracket the
    /// offending pair, and `node_epochs_j2000_s` names every node the
    /// prediction was formed from.
    HoldOutResidual {
        /// The satellite.
        sat: GnssSatelliteId,
        /// Epoch of the held-out sample, seconds since J2000.
        epoch_j2000_s: f64,
        /// Epoch of the preceding sample, seconds since J2000 - the other side
        /// of the offending pair.
        preceding_j2000_s: f64,
        /// 3D distance between the stored record and the value predicted from
        /// its neighbours, meters.
        residual_m: f64,
        /// The tolerance it exceeded, meters.
        tolerance_m: f64,
        /// Epochs of the retained nodes the prediction used, seconds since
        /// J2000, ascending.
        ///
        /// The replay keeps every other sample, so these sit on twice the
        /// product spacing, and near a run end the window slides inward to the
        /// run's last eleven retained nodes, twenty product spacings long. A
        /// residual can therefore come from a node far from the bracketing
        /// pair, and these are the records it rests on besides the held-out
        /// sample itself. They are selected by the same rule the position
        /// interpolator applies.
        node_epochs_j2000_s: Vec<f64>,
    },
}

impl ContinuityDefect {
    /// The satellite this defect concerns.
    pub fn satellite(&self) -> GnssSatelliteId {
        match self {
            Self::DuplicateEpoch { sat, .. }
            | Self::SingleSampleSeries { sat }
            | Self::UnusableSample { sat, .. }
            | Self::SpeedBound { sat, .. }
            | Self::HoldOutResidual { sat, .. } => *sat,
        }
    }

    /// Whether this finding can enter any interpolation stencil used by the
    /// evaluation window.
    ///
    /// A hold-out residual is placed on its offending pair, the held-out sample
    /// and its predecessor. The nodes its prediction used stay in
    /// `node_epochs_j2000_s` for attribution, but a window whose stencil reaches
    /// only those nodes interpolates records the check did not find at fault,
    /// and is not refused for them.
    pub(super) fn influences(&self, window: EpochWindow, stencil: StencilExtent) -> bool {
        let (needed_from, needed_through) = stencil.influence_bounds(window);
        let support = match self {
            Self::DuplicateEpoch { epoch_j2000_s, .. } => Some((*epoch_j2000_s, *epoch_j2000_s)),
            Self::SingleSampleSeries { .. } => None,
            Self::UnusableSample { epoch_j2000_s, .. } => {
                epoch_j2000_s.map(|epoch_j2000_s| (epoch_j2000_s, epoch_j2000_s))
            }
            Self::SpeedBound {
                from_j2000_s,
                to_j2000_s,
                ..
            } => Some((from_j2000_s.min(*to_j2000_s), from_j2000_s.max(*to_j2000_s))),
            Self::HoldOutResidual {
                epoch_j2000_s,
                preceding_j2000_s,
                ..
            } => Some((
                epoch_j2000_s.min(*preceding_j2000_s),
                epoch_j2000_s.max(*preceding_j2000_s),
            )),
        };

        match support {
            Some((from, through)) => from <= needed_through && through >= needed_from,
            // A single-sample series, or a sample whose epoch names no second,
            // has no epoch. Querying the existing report must therefore treat
            // it conservatively rather than invent a location or hide an
            // unresolved input defect.
            None => true,
        }
    }
}

/// Why [`ContinuityDefect::UnusableSample`] took no part in a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnusableSampleReason {
    /// The epoch names no finite second on the J2000 axis.
    EpochNotPlaced,
    /// A position coordinate is NaN or infinite.
    NonFinitePosition,
}

/// Which check produced a defect, for callers filtering a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityCheck {
    /// Input well-formedness: duplicate epochs, single-sample series,
    /// unusable samples.
    Input,
    /// The physical earth-fixed speed gate.
    SpeedBound,
    /// The hold-out interpolation residual check.
    HoldOutResidual,
}

/// Result of a continuity check.
///
/// Absence of defects is the attestation; presence of defects is the structured
/// report. Both are the same type so a caller cannot accidentally handle only
/// one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContinuityReport {
    /// Every defect found, ordered by satellite then epoch.
    pub defects: Vec<ContinuityDefect>,
    /// Adjacent pairs the speed gate examined.
    pub pairs_checked: usize,
    /// Samples the hold-out residual check examined.
    pub residuals_checked: usize,
    /// Samples the residual check could not evaluate because the held-out
    /// neighbourhood was not interpolatable (a coverage gap, or too few
    /// neighbours). Reported rather than silently dropped: a caller must be able
    /// to tell "checked and clean" from "not checked".
    pub residuals_skipped: usize,
}

impl ContinuityReport {
    /// Whether the series is attested continuous: no defects of any class.
    pub fn attested(&self) -> bool {
        self.defects.is_empty()
    }

    /// Defects produced by one check.
    pub fn defects_from(&self, check: ContinuityCheck) -> impl Iterator<Item = &ContinuityDefect> {
        self.defects.iter().filter(move |defect| {
            let source = match defect {
                ContinuityDefect::DuplicateEpoch { .. }
                | ContinuityDefect::SingleSampleSeries { .. }
                | ContinuityDefect::UnusableSample { .. } => ContinuityCheck::Input,
                ContinuityDefect::SpeedBound { .. } => ContinuityCheck::SpeedBound,
                ContinuityDefect::HoldOutResidual { .. } => ContinuityCheck::HoldOutResidual,
            };
            source == check
        })
    }

    /// Findings that can influence evaluation in `window` through `stencil`.
    ///
    /// This filters the existing report. It does not rerun continuity checks or
    /// alter their defaults. A [`ContinuityDefect::SingleSampleSeries`] has no
    /// stored epoch, so it conservatively influences every window, as does a
    /// [`ContinuityDefect::UnusableSample`] whose epoch names no second.
    pub fn defects_influencing(
        &self,
        window: EpochWindow,
        stencil: StencilExtent,
    ) -> Vec<&ContinuityDefect> {
        self.defects
            .iter()
            .filter(|defect| defect.influences(window, stencil))
            .collect()
    }

    /// Decide whether recorded defects can influence an evaluation window.
    ///
    /// The full report remains available in [`WindowContinuityVerdict::all_defects`]
    /// whether the result accepts or refuses the window.
    pub fn verdict_for_window(
        &self,
        window: EpochWindow,
        stencil: StencilExtent,
    ) -> WindowContinuityVerdict<'_> {
        let influencing_defects = self.defects_influencing(window, stencil);
        let decision = if influencing_defects.is_empty() {
            WindowContinuityDecision::Accept
        } else {
            WindowContinuityDecision::Refuse
        };
        WindowContinuityVerdict {
            decision,
            influencing_defects,
            influencing_splices: Vec::new(),
            all_defects: &self.defects,
            all_splices: Vec::new(),
        }
    }
}

/// One satellite's samples, sorted by epoch with duplicates already extracted.
///
/// Construction is the only way to obtain one, and construction sorts. After it,
/// `x` is strictly increasing, so every adjacent pair has a strictly positive
/// interval *by construction* - a zero or negative interval is unrepresentable
/// in the comparison path rather than checked for at each use.
struct OrderedSeries {
    /// Node epochs, seconds since J2000, strictly increasing.
    x: Vec<f64>,
    /// Node positions in file-native kilometres, matching the interpolation
    /// substrate's fit units.
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    /// Node positions in SI meters, for displacement arithmetic.
    pos_m: Vec<[f64; 3]>,
}

impl OrderedSeries {
    /// Sort `samples` by epoch and split out duplicates as defects.
    ///
    /// A sample whose epoch names no finite J2000 second, or whose position is
    /// not finite, cannot be placed on the comparison path. It is withheld from
    /// every check and reported as [`ContinuityDefect::UnusableSample`], so a
    /// series left short by it is never attested silently.
    fn build(
        sat: GnssSatelliteId,
        samples: &[(usize, &PreciseEphemerisSample)],
        defects: &mut Vec<ContinuityDefect>,
    ) -> Option<Self> {
        let mut placed: Vec<(f64, [f64; 3])> = Vec::with_capacity(samples.len());
        for &(sample_index, sample) in samples {
            let node = instant_to_j2000_seconds(&sample.epoch)
                .filter(|seconds| seconds.is_finite())
                .and_then(|_| precise_node_j2000_seconds_from_instant(&sample.epoch))
                .filter(|node| node.is_finite());
            let unusable = |epoch_j2000_s, reason| ContinuityDefect::UnusableSample {
                sat,
                sample_index,
                epoch_j2000_s,
                reason,
            };
            let Some(node) = node else {
                defects.push(unusable(None, UnusableSampleReason::EpochNotPlaced));
                continue;
            };
            if !sample.position_ecef_m.iter().all(|c| c.is_finite()) {
                defects.push(unusable(
                    Some(node),
                    UnusableSampleReason::NonFinitePosition,
                ));
                continue;
            }
            placed.push((node, sample.position_ecef_m));
        }

        // The sort is this module's job, not the caller's: a shuffled input must
        // reach the checks in the identical order a sorted one does.
        placed.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut series = Self {
            x: Vec::with_capacity(placed.len()),
            kx: Vec::with_capacity(placed.len()),
            ky: Vec::with_capacity(placed.len()),
            kz: Vec::with_capacity(placed.len()),
            pos_m: Vec::with_capacity(placed.len()),
        };

        let mut index = 0usize;
        while index < placed.len() {
            let epoch = placed[index].0;
            let mut run = 1usize;
            while index + run < placed.len() && placed[index + run].0 == epoch {
                run += 1;
            }
            if run > 1 {
                // Duplicates are a defect of the data and are not resolved here.
                // Every record for the epoch is withheld from the comparison
                // path: silently keeping one would be picking an authoritative
                // sample, which is the caller's decision.
                defects.push(ContinuityDefect::DuplicateEpoch {
                    sat,
                    epoch_j2000_s: epoch,
                    occurrences: run,
                });
            } else {
                let (node, pos_m) = placed[index];
                series.x.push(node);
                series.kx.push(pos_m[0] / KM_TO_M);
                series.ky.push(pos_m[1] / KM_TO_M);
                series.kz.push(pos_m[2] / KM_TO_M);
                series.pos_m.push(pos_m);
            }
            index += run;
        }

        if series.x.is_empty() {
            return None;
        }
        Some(series)
    }

    fn len(&self) -> usize {
        self.x.len()
    }

    /// Every adjacent pair, in time order. The only path to a comparison, and
    /// every interval it yields is strictly positive.
    fn pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (1..self.x.len()).map(|index| (index - 1, index))
    }
}

/// Check an ordered ephemeris sample sequence for continuity.
///
/// Samples for any number of satellites may be supplied in any order; they are
/// grouped by satellite and sorted by epoch internally. The verdict is
/// independent of the order they arrive in.
///
/// This never refuses a series. Every finding lands in
/// [`ContinuityReport::defects`], and whether a product with defects is
/// acceptable is the caller's decision. It refuses options whose bounds
/// [`ContinuityOptions::validate`] refuses, before reading any sample: a NaN
/// or infinite bound would find nothing and attest the series untested.
pub fn check_continuity(
    samples: &[PreciseEphemerisSample],
    options: &ContinuityOptions,
) -> core::result::Result<ContinuityReport, ContinuityOptionsError> {
    options.validate()?;
    Ok(check_validated_continuity(samples, options))
}

/// [`check_continuity`] over options already validated.
pub(super) fn check_validated_continuity(
    samples: &[PreciseEphemerisSample],
    options: &ContinuityOptions,
) -> ContinuityReport {
    let mut by_sat: BTreeMap<GnssSatelliteId, Vec<(usize, &PreciseEphemerisSample)>> =
        BTreeMap::new();
    for (sample_index, sample) in samples.iter().enumerate() {
        by_sat
            .entry(sample.sat)
            .or_default()
            .push((sample_index, sample));
    }

    let mut report = ContinuityReport::default();
    for (sat, sat_samples) in by_sat {
        // Each satellite's defects are collected and ordered on their own before
        // joining the report, so the report reads as one timeline per satellite
        // rather than interleaving satellites or check passes.
        let mut sat_defects = Vec::new();
        let Some(series) = OrderedSeries::build(sat, &sat_samples, &mut sat_defects) else {
            report.defects.append(&mut sat_defects);
            continue;
        };
        if series.len() < 2 {
            sat_defects.push(ContinuityDefect::SingleSampleSeries { sat });
            report.defects.append(&mut sat_defects);
            continue;
        }

        if let Some(bound) = options.speed_bound {
            check_speed_bound(sat, &series, bound, &mut sat_defects, &mut report);
        }
        if let Some(tolerance_m) = options.residual_tolerance_m {
            check_hold_out_residual(
                sat,
                &series,
                tolerance_m,
                options.interpolation.gap_threshold_factor(),
                &mut sat_defects,
                &mut report,
            );
        }

        sat_defects.sort_by(|a, b| defect_sort_key(a).total_cmp(&defect_sort_key(b)));
        report.defects.append(&mut sat_defects);
    }
    report
}

fn check_speed_bound(
    sat: GnssSatelliteId,
    series: &OrderedSeries,
    bound: SpeedBound,
    defects: &mut Vec<ContinuityDefect>,
    report: &mut ContinuityReport,
) {
    let bound_m_s = bound.value_m_s();
    for (lo, hi) in series.pairs() {
        let interval_s = series.x[hi] - series.x[lo];
        let displacement_m = distance_m(series.pos_m[lo], series.pos_m[hi]);
        report.pairs_checked += 1;

        let implied_speed_m_s = displacement_m / interval_s;
        if implied_speed_m_s > bound_m_s {
            defects.push(ContinuityDefect::SpeedBound {
                sat,
                from_j2000_s: series.x[lo],
                to_j2000_s: series.x[hi],
                interval_s,
                displacement_m,
                implied_speed_m_s,
                bound_m_s,
            });
        }
    }
}

/// Hold out each interior sample and compare it against the arc its neighbours
/// describe.
///
/// The prediction runs through the same sliding-window Lagrange substrate the
/// product's own interpolator uses, so the residual on a clean arc is the
/// interpolator's own error rather than a second, differently-wrong model of the
/// orbit.
///
/// # Why the hold-out is by parity, not one node at a time
///
/// Deleting a single node from an otherwise uniform series does not leave a
/// series that is merely one sample shorter: it leaves one interval of twice the
/// nominal spacing, which the substrate correctly classifies as a *coverage gap*
/// and refuses to interpolate across. The evaluation then degrades to a
/// one-sided extrapolation, whose error grows with the polynomial degree and
/// reaches hundreds of kilometres at the arc's end - it would measure the
/// extrapolation, not the data.
///
/// Holding out every other sample instead keeps the retained series uniform (at
/// twice the spacing), so the substrate sees no gap and each held-out epoch is a
/// genuine interpolation bracketed by real neighbours. Two passes of opposite
/// parity cover every interior sample exactly once. This is the same decimation
/// hold-out the sample-source parity oracle uses.
///
/// Endpoints are never held out: with no neighbour on one side any evaluation
/// there is an extrapolation regardless of scheme.
fn check_hold_out_residual(
    sat: GnssSatelliteId,
    series: &OrderedSeries,
    tolerance_m: f64,
    gap_threshold_factor: f64,
    defects: &mut Vec<ContinuityDefect>,
    report: &mut ContinuityReport,
) {
    let interior = series.len().saturating_sub(2);
    if interior == 0 {
        // Only endpoints exist; nothing can be held out with a neighbour on both
        // sides. Counted as skipped so the caller can see the check did not run.
        report.residuals_skipped += series.len();
        return;
    }

    for parity in [0usize, 1usize] {
        // Retained nodes: every sample whose index shares this parity. Held-out
        // nodes are the interior samples of the opposite parity.
        let keep: Vec<usize> = (0..series.len())
            .filter(|index| index % 2 == parity)
            .collect();
        let held: Vec<usize> = (1..series.len() - 1)
            .filter(|index| index % 2 != parity)
            .collect();
        if held.is_empty() {
            continue;
        }
        if keep.len() < 2 {
            // Too few retained nodes to define any fit for this parity.
            report.residuals_skipped += held.len();
            continue;
        }

        let x: Vec<f64> = keep.iter().map(|&i| series.x[i]).collect();
        let kx: Vec<f64> = keep.iter().map(|&i| series.kx[i]).collect();
        let ky: Vec<f64> = keep.iter().map(|&i| series.ky[i]).collect();
        let kz: Vec<f64> = keep.iter().map(|&i| series.kz[i]).collect();

        for index in held {
            let query = series.x[index];
            match interpolate_precise_state(
                sat,
                &x,
                &kx,
                &ky,
                &kz,
                &[],
                query,
                gap_threshold_factor,
            ) {
                Ok(state) => {
                    report.residuals_checked += 1;
                    let predicted = [state.position.x_m, state.position.y_m, state.position.z_m];
                    let residual_m = distance_m(predicted, series.pos_m[index]);
                    if residual_m > tolerance_m {
                        defects.push(ContinuityDefect::HoldOutResidual {
                            sat,
                            epoch_j2000_s: query,
                            preceding_j2000_s: series.x[index - 1],
                            residual_m,
                            tolerance_m,
                            node_epochs_j2000_s: selected_node_epochs(
                                &x,
                                query,
                                gap_threshold_factor,
                            ),
                        });
                    }
                }
                Err(_) => {
                    // The retained neighbourhood is not interpolatable at this
                    // epoch - a real coverage gap in the product, not an artifact
                    // of the hold-out. Not a defect of the data, but not a pass
                    // either, so it is counted rather than dropped.
                    report.residuals_skipped += 1;
                }
            }
        }
    }
}

/// Epochs of the retained nodes the interpolator selected for `query`.
///
/// Called only after the interpolation succeeded; the selection is the
/// interpolator's own [`select_position_nodes`] over the same nodes, query and
/// policy.
fn selected_node_epochs(x: &[f64], query: f64, gap_threshold_factor: f64) -> Vec<f64> {
    // The interpolator selects with the query as `PreciseQuery` restates it.
    let query = PreciseQuery::at(query).j2000_s();
    select_position_nodes(x.len(), query, gap_threshold_factor, |i| x[i])
        .map(|window| x[window].to_vec())
        .unwrap_or_default()
}

/// Epoch a defect is anchored at, for ordering a report as a timeline.
fn defect_sort_key(defect: &ContinuityDefect) -> f64 {
    match defect {
        ContinuityDefect::DuplicateEpoch { epoch_j2000_s, .. } => *epoch_j2000_s,
        ContinuityDefect::SingleSampleSeries { .. } => f64::NEG_INFINITY,
        ContinuityDefect::UnusableSample { epoch_j2000_s, .. } => {
            epoch_j2000_s.unwrap_or(f64::NEG_INFINITY)
        }
        ContinuityDefect::SpeedBound { from_j2000_s, .. } => *from_j2000_s,
        ContinuityDefect::HoldOutResidual { epoch_j2000_s, .. } => *epoch_j2000_s,
    }
}

fn distance_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
