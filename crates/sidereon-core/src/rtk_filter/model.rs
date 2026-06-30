//! Measurement-model domain types and pure geometry for the RTK baseline
//! filter: the epoch/satellite observation structs, the single-difference
//! variance (stochastic) models, and the geometric-range / line-of-sight
//! primitives the double-difference row builders consume.

use crate::astro::math::vec3::{norm3, sub3};

use crate::constants::{C_M_S, OMEGA_E_DOT_RAD_S};
use crate::estimation::recipe::{FrameRecipe, SagnacRecipe};
use crate::id::GnssSystem;

/// Code vs carrier-phase double-difference row. The `Ord` is the covariance
/// block sort order and must keep `Code < Phase` (Elixir sorts the block key
/// `{epoch, kind, ref}` with the atoms `:code < :phase`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RowKind {
    #[default]
    Code,
    Phase,
}

/// One satellite's base+rover code/phase observations and per-receiver
/// transmit-time ECEF positions at an epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct SatMeas {
    pub sat: String,
    /// Single-difference ambiguity id (gap-segmentation already applied upstream).
    pub sd_ambiguity_id: String,
    pub base_code_m: f64,
    pub base_phase_m: f64,
    pub rover_code_m: f64,
    pub rover_phase_m: f64,
    /// Transmit-time satellite ECEF position (metres) for the base receiver.
    pub base_tx_pos: [f64; 3],
    /// Transmit-time satellite ECEF position (metres) for the rover receiver.
    pub rover_tx_pos: [f64; 3],
    /// Shared receive-time satellite ECEF position (metres). Used only for the
    /// elevation-dependent variance models: the Elixir reference computes
    /// elevation from `epoch.positions` (the shared map), NOT the per-receiver
    /// transmit-time maps. For synthetic fixtures this may equal the tx position.
    pub pos: [f64; 3],
}

/// One RTK epoch: the per-system double-difference reference satellites present
/// this epoch plus the non-reference satellites observed at both receivers.
/// Each non-reference satellite differences against the reference of its own
/// system (the first byte of the satellite id); a one-element `references` is
/// the historical single-system epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Epoch {
    pub references: Vec<SatMeas>,
    pub nonref: Vec<SatMeas>,
    /// Optional rover ECEF velocity in metres/second for the velocity-propagated
    /// predict branch. The value is an epoch input, not carried filter state.
    pub velocity_mps: Option<[f64; 3]>,
    /// Elapsed seconds since the previous filter epoch, used only when
    /// [`crate::rtk_filter::UpdateOpts::dynamics_model`] is
    /// [`crate::rtk_filter::DynamicsModel::VelocityPropagated`].
    pub dt_s: f64,
}

/// Constellation letter of a satellite (or single-difference ambiguity) id,
/// mirroring the Elixir `satellite_system/1` (`String.first`). Delegates to the
/// canonical [`crate::id::constellation_letter`] extractor.
pub(super) fn satellite_system(id: &str) -> &str {
    crate::id::constellation_letter(id)
}

/// The constellation of a satellite id as a typed [`GnssSystem`], for the
/// per-system double-difference reference grouping.
///
/// Returns `None` when the leading character is not a recognized RINEX/IGS
/// system letter. The double-difference grouping pairs each satellite with the
/// reference of its OWN system, so an id with an unrecognized constellation
/// never shares a reference with a valid satellite; that is exactly the
/// behavior of the previous letter-string grouping, where such an id grouped
/// under its raw leading byte and matched no valid reference. Boundary-facing
/// error/state strings keep using [`satellite_system`].
pub(super) fn system_of(id: &str) -> Option<GnssSystem> {
    GnssSystem::from_letter(id.chars().next()?)
}

/// The `:float_only_systems` boundary list (constellation letters) as a typed
/// set of [`GnssSystem`]. Entries that are not exactly one recognized system
/// letter are dropped: under the prior `s == satellite_system(sat)` string
/// test such an entry could never equal a satellite's single-letter
/// constellation, so dropping it is behavior-preserving for every valid GNSS
/// token. This is the single conversion of the boundary list the sequential
/// filter and the static fixed search both build on.
pub(super) fn float_only_set(float_only_systems: &[String]) -> Vec<GnssSystem> {
    float_only_systems
        .iter()
        .filter_map(|s| {
            let mut chars = s.chars();
            let letter = chars.next()?;
            // A multi-character entry never equalled a one-letter system label.
            if chars.next().is_some() {
                return None;
            }
            GnssSystem::from_letter(letter)
        })
        .collect()
}

/// True when a satellite's constellation is excluded from integer ambiguity
/// resolution by the caller's `:float_only_systems` (as the typed
/// [`float_only_set`]). A satellite whose leading byte is not a recognized
/// system letter ([`system_of`] is `None`) is never an AR target, matching the
/// prior letter-string membership test for every valid GNSS token. This is the
/// single definition of the float-only membership test the sequential filter
/// (`has_search_targets`, the search-target screen) and the static fixed search
/// (`float_only_ambiguity_ids`) share.
pub(super) fn is_float_only_system(sat: &str, float_only: &[GnssSystem]) -> bool {
    system_of(sat).is_some_and(|system| float_only.contains(&system))
}

/// Single-difference variance model, mirroring the Elixir `weights` options
/// (`stochastic_model` / `elevation_weighting`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticModel {
    /// `SD variance = 2σ²`; with `elevation_weighting`, σ is first scaled by
    /// `1/max(sin el, MIN_ELEVATION_SIN)`.
    Simple { elevation_weighting: bool },
    /// RTKLIB floor-plus-elevation: `SD variance = 2(σ² + σ²/sin²el)` with the
    /// same clamped `sin el`.
    Rtklib,
}

/// Floor on `sin(elevation)` in the variance models (Elixir `@min_elevation_sin`).
pub(crate) const MIN_ELEVATION_SIN: f64 = 0.05;

/// Measurement-model configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasModel {
    pub code_sigma_m: f64,
    pub phase_sigma_m: f64,
    pub sagnac: bool,
    pub stochastic: StochasticModel,
}

/// Sine of the satellite elevation seen from `base`.
///
/// PARITY: the local-up here is GEOCENTRIC (`base/|base|`), exactly as the
/// Elixir `local_up/1` - NOT the geodetic ellipsoid normal. The two differ by
/// up to ~0.19°, which shifts low-elevation variances; do not "fix" this to the
/// geodetic frame (`substrate::frames::geodetic_from_ecef` under the SPP recipe)
/// without changing the Elixir reference in lockstep.
pub(super) fn elevation_sin(base: [f64; 3], sat_pos: [f64; 3]) -> f64 {
    // Geocentric local-up (`base / |base|`, reciprocal-multiply), shared with the
    // PPP/RTK antenna code; see [`crate::frame::geocentric_up`] for the
    // geocentric-vs-geodetic distinction this parity note used to document inline.
    let up =
        crate::estimation::substrate::frames::local_up(FrameRecipe::GeocentricUpRtkReference, base);
    let d = sub3(sat_pos, base);
    let dn = norm3(d);
    if dn > 0.0 {
        // Elixir op order: normalize the LOS first, THEN dot with up
        // (`unit3(sub3(sat, base))` = `scale3(_, 1.0 / n)`, then `dot3`).
        let inv = 1.0 / dn;
        let los = [d[0] * inv, d[1] * inv, d[2] * inv];
        los[0] * up[0] + los[1] * up[1] + los[2] * up[2]
    } else {
        -1.0
    }
}

/// Single-difference variance for one receiver-satellite pair (Elixir
/// `single_difference_variance/4`).
pub(super) fn single_difference_variance(
    sigma_m: f64,
    stochastic: StochasticModel,
    base: [f64; 3],
    sat_pos: [f64; 3],
) -> f64 {
    match stochastic {
        StochasticModel::Simple {
            elevation_weighting: false,
        } => 2.0 * sigma_m * sigma_m,
        StochasticModel::Simple {
            elevation_weighting: true,
        } => {
            let sin_el = elevation_sin(base, sat_pos).max(MIN_ELEVATION_SIN);
            let scaled = sigma_m / sin_el;
            2.0 * scaled * scaled
        }
        StochasticModel::Rtklib => {
            let sin_el = elevation_sin(base, sat_pos).max(MIN_ELEVATION_SIN);
            2.0 * (sigma_m * sigma_m + sigma_m * sigma_m / (sin_el * sin_el))
        }
    }
}

pub(super) fn range_m(sat: [f64; 3], recv: [f64; 3]) -> f64 {
    norm3(sub3(sat, recv))
}

pub(super) fn geometric_range_m(sat: [f64; 3], recv: [f64; 3], sagnac: bool) -> f64 {
    let recipe = if sagnac {
        SagnacRecipe::RtklibFirstOrderScalar
    } else {
        SagnacRecipe::Off
    };
    crate::estimation::substrate::range::geometric_range(
        recipe,
        sat,
        recv,
        OMEGA_E_DOT_RAD_S,
        C_M_S,
    )
}

/// LOS unit vector `(recv - sat)/range` - the partial of range w.r.t. the
/// receiver position (Euclidean; Sagnac corrects the scalar range only).
pub(super) fn range_derivative(recv: [f64; 3], sat: [f64; 3]) -> [f64; 3] {
    // Elixir `scale3(sub3(receiver, sat_pos), 1.0 / rho)`: multiply by reciprocal,
    // NOT component-wise division (they round differently - 0-ULP requires this).
    let rho = range_m(sat, recv);
    let inv = 1.0 / rho;
    let d = sub3(recv, sat);
    [d[0] * inv, d[1] * inv, d[2] * inv]
}
