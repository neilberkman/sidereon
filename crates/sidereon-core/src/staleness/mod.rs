//! Product-staleness graceful degradation for time-varying GNSS products.
//!
//! Time-varying products (IONEX TEC maps, rapid/predicted SP3 orbit/clock files)
//! publish with latency and gaps, so the product for the exact requested epoch is
//! not always on hand. A direct lookup against a missing epoch is a hard failure,
//! which is brittle for real-time and operational use.
//!
//! This module sits on top of the [`Ionex`](crate::atmosphere::Ionex) and
//! [`Sp3`](crate::ephemeris::Sp3) parsers and adds a selection layer that
//! degrades gracefully instead of erroring: given a SET of available parsed
//! products and a requested epoch (or epoch range), it returns a usable handle,
//! falling back to the most-recent available product within a configurable
//! staleness cap. Every result carries [`StalenessMetadata`] describing which
//! source epoch was used, how stale it is, and the [`DegradationKind`], so a
//! degraded answer is never substituted silently. Only a request that exceeds the
//! staleness cap fails, with a typed [`SelectionError`].
//!
//! # Degradation paths
//!
//! - **Exact**: a product covering the requested epoch is present. The original
//!   product is returned untouched and the downstream evaluation is bit-for-bit
//!   identical to calling the parser/interpolator directly. Staleness is zero.
//! - **IONEX diurnal shift**: when no product covers the requested day, the
//!   most-recent prior day's grid is advanced by whole days onto the requested
//!   epoch ([`Ionex::with_map_epochs_shifted_days`](crate::atmosphere::Ionex)).
//!   TEC is approximately 24-hour periodic, so this is near-lossless for the
//!   boundary window. The grid values are unchanged; only the epoch axis moves.
//! - **SP3 nearest-prior**: when no product covers the requested epoch, the
//!   most-recent prior product is selected as-is, with the staleness measured
//!   from its last epoch.
//!
//! # Network
//!
//! This layer is pure and no-network: it selects among products the caller has
//! already parsed. Fetching the products is a per-binding concern.

use std::borrow::Cow;
use std::fmt;

use crate::astro::constants::time::{SECONDS_PER_DAY, SECONDS_PER_DAY_I64};
use crate::atmosphere::Ionex;
use crate::ephemeris::{EphemerisSource, Sp3, Sp3State};
use crate::frame::Wgs84Geodetic;
use crate::id::GnssSatelliteId;
use crate::ionex::ionex_slant_delay;

/// Default staleness cap, in whole days.
///
/// A request whose nearest usable product is older than this is rejected with
/// [`SelectionError::BeyondStalenessCap`]. Three days spans the typical
/// rapid/predicted product latency plus a weekend gap.
pub const DEFAULT_MAX_STALENESS_DAYS: u32 = 3;

/// How a selected product's source epoch relates to the requested epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationKind {
    /// A product covering the requested epoch was present; no degradation.
    Exact,
    /// No product covered the requested epoch; the most-recent prior product was
    /// used as-is (SP3 path).
    NearestPrior,
    /// No product covered the requested day; a prior day's IONEX grid was
    /// advanced by whole days onto the requested epoch (diurnal persistence).
    DiurnalShift,
}

impl DegradationKind {
    /// Whether this result used the exact present product (no degradation).
    pub fn is_exact(self) -> bool {
        matches!(self, DegradationKind::Exact)
    }
}

/// Structured description of the product staleness behind a selection result.
///
/// Attached to every [`IonexSelection`] / [`Sp3Selection`]; a degraded result is
/// never produced without it. Epoch fields are seconds since the J2000 epoch
/// (2000-01-01 12:00:00). `staleness_s` is `requested - source` and is never
/// negative. This is the public type the language bindings wrap (Python
/// dataclass, Elixir struct, C handle) and the broadcast-fallback path reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StalenessMetadata {
    /// Which degradation path produced the result.
    pub kind: DegradationKind,
    /// The requested epoch, J2000 seconds. For a range request this is the
    /// latest (most-stale) epoch of the range.
    pub requested_epoch_j2000_s: f64,
    /// The source product epoch the result is backed by, J2000 seconds. For
    /// [`DegradationKind::Exact`] this equals the requested epoch; for a diurnal
    /// shift it is the same time-of-day a whole number of days earlier; for
    /// nearest-prior it is the source product's last epoch.
    pub source_epoch_j2000_s: f64,
    /// Staleness `requested - source`, seconds. Zero for an exact result; never
    /// negative.
    pub staleness_s: f64,
    /// Staleness in days (`staleness_s / 86400`). For a diurnal shift this is the
    /// integer day offset applied.
    pub staleness_days: f64,
}

impl StalenessMetadata {
    /// Metadata for a present, exact result (zero staleness) at `epoch_j2000_s`.
    fn exact(epoch_j2000_s: f64) -> Self {
        Self {
            kind: DegradationKind::Exact,
            requested_epoch_j2000_s: epoch_j2000_s,
            source_epoch_j2000_s: epoch_j2000_s,
            staleness_s: 0.0,
            staleness_days: 0.0,
        }
    }
}

/// Configurable staleness cap for product selection.
///
/// A selection that would rely on a product older than `max_staleness_s` fails
/// with [`SelectionError::BeyondStalenessCap`] rather than returning data past
/// the cap. The [`Default`] is [`DEFAULT_MAX_STALENESS_DAYS`].
///
/// ```
/// use sidereon_core::staleness::StalenessPolicy;
/// let policy = StalenessPolicy::default();
/// assert_eq!(policy.max_staleness_s, 3.0 * 86_400.0);
/// assert_eq!(StalenessPolicy::days(1.0).max_staleness_s, 86_400.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StalenessPolicy {
    /// Maximum tolerated staleness, seconds.
    pub max_staleness_s: f64,
}

impl StalenessPolicy {
    /// A policy with a cap expressed in days.
    pub fn days(days: f64) -> Self {
        Self {
            max_staleness_s: days * SECONDS_PER_DAY,
        }
    }

    /// A policy with a cap expressed in seconds.
    pub fn seconds(seconds: f64) -> Self {
        Self {
            max_staleness_s: seconds,
        }
    }
}

impl Default for StalenessPolicy {
    fn default() -> Self {
        Self::days(f64::from(DEFAULT_MAX_STALENESS_DAYS))
    }
}

/// Error returned when no product can satisfy a request.
///
/// No degraded data is ever returned through this type: a successful selection
/// always carries [`StalenessMetadata`], and these variants are the only outcomes
/// where the layer declines to produce a result.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionError {
    /// The product set was empty.
    EmptyProductSet,
    /// The requested range was malformed (non-finite, or end before start).
    InvalidRange {
        /// Range start, J2000 seconds.
        start_epoch_j2000_s: f64,
        /// Range end, J2000 seconds.
        end_epoch_j2000_s: f64,
    },
    /// No product covers or precedes the requested epoch; only later products
    /// are available, so there is nothing to degrade to.
    NoPriorProduct {
        /// The requested epoch, J2000 seconds.
        requested_epoch_j2000_s: f64,
    },
    /// The most-recent usable product is older than the staleness cap.
    BeyondStalenessCap {
        /// The requested epoch, J2000 seconds.
        requested_epoch_j2000_s: f64,
        /// The source epoch that would have been used, J2000 seconds.
        source_epoch_j2000_s: f64,
        /// How stale that source is, seconds.
        staleness_s: f64,
        /// The cap that was exceeded, seconds.
        max_staleness_s: f64,
    },
    /// A product in the set was malformed (e.g. no epochs, or an epoch that
    /// cannot be projected onto the J2000 axis), or the only prior product
    /// cannot cover the requested range even after a whole-day diurnal shift.
    InvalidProduct(String),
    /// The staleness policy cap was non-finite or negative. A cap that is not a
    /// finite, non-negative number of seconds cannot bound degradation:
    /// comparisons such as `staleness_s > NaN` are always false, which would
    /// admit arbitrarily stale data without surfacing it. The selection layer
    /// rejects such a policy rather than masking the failure it exists to catch.
    InvalidPolicy {
        /// The rejected cap, seconds.
        max_staleness_s: f64,
    },
    /// An epoch computation overflowed the i64 J2000-second axis for an extreme
    /// requested range. No usable result can be produced without wrapping, so
    /// the request is declined rather than returning a wrapped epoch.
    Overflow {
        /// Which computation overflowed.
        context: &'static str,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionError::EmptyProductSet => write!(f, "product set is empty"),
            SelectionError::InvalidRange {
                start_epoch_j2000_s,
                end_epoch_j2000_s,
            } => write!(
                f,
                "invalid epoch range [{start_epoch_j2000_s}, {end_epoch_j2000_s}]"
            ),
            SelectionError::NoPriorProduct {
                requested_epoch_j2000_s,
            } => write!(
                f,
                "no product at or before requested epoch {requested_epoch_j2000_s} J2000 s"
            ),
            SelectionError::BeyondStalenessCap {
                requested_epoch_j2000_s,
                source_epoch_j2000_s,
                staleness_s,
                max_staleness_s,
            } => write!(
                f,
                "nearest product (epoch {source_epoch_j2000_s} J2000 s) is {staleness_s} s stale \
                 for requested epoch {requested_epoch_j2000_s} J2000 s, over the {max_staleness_s} s cap"
            ),
            SelectionError::InvalidProduct(msg) => write!(f, "invalid product in set: {msg}"),
            SelectionError::InvalidPolicy { max_staleness_s } => write!(
                f,
                "staleness cap {max_staleness_s} s is not a finite, non-negative number of seconds"
            ),
            SelectionError::Overflow { context } => {
                write!(f, "epoch arithmetic overflow: {context}")
            }
        }
    }
}

/// Reject a staleness cap that cannot bound degradation.
///
/// A cap must be a finite, non-negative number of seconds. A non-finite or
/// negative cap is the silent-masking hazard this layer exists to prevent
/// (`staleness_s > NaN` is always false), so it is a typed error, not a default.
fn validate_policy(policy: StalenessPolicy) -> Result<(), SelectionError> {
    if policy.max_staleness_s.is_finite() && policy.max_staleness_s >= 0.0 {
        Ok(())
    } else {
        Err(SelectionError::InvalidPolicy {
            max_staleness_s: policy.max_staleness_s,
        })
    }
}

impl std::error::Error for SelectionError {}

/// A selected IONEX product plus its staleness metadata.
///
/// Obtain one from [`select_ionex`] or [`select_ionex_over_range`]. The inner
/// product is either the present product (borrowed, byte-identical to the
/// caller's) or a diurnal-shifted copy; [`IonexSelection::ionex`] exposes it and
/// [`IonexSelection::slant_delay`] runs the standard slant-delay evaluation on
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct IonexSelection<'a> {
    ionex: Cow<'a, Ionex>,
    metadata: StalenessMetadata,
}

impl IonexSelection<'_> {
    /// The staleness metadata for this selection.
    pub fn metadata(&self) -> StalenessMetadata {
        self.metadata
    }

    /// The usable IONEX product: the present product for an exact result, or the
    /// diurnal-shifted copy for a degraded one.
    pub fn ionex(&self) -> &Ionex {
        self.ionex.as_ref()
    }

    /// Slant ionospheric group delay (positive meters) from the selected product.
    ///
    /// Delegates to [`ionex_slant_delay`](crate::atmosphere::ionex_slant_delay)
    /// on the inner product. For an exact selection this is the inner product
    /// untouched, so the result is bit-for-bit identical to calling
    /// `ionex_slant_delay` on the caller's product directly.
    pub fn slant_delay(
        &self,
        receiver: Wgs84Geodetic,
        elevation_rad: f64,
        azimuth_rad: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
    ) -> crate::Result<f64> {
        ionex_slant_delay(
            self.ionex.as_ref(),
            receiver,
            elevation_rad,
            azimuth_rad,
            epoch_j2000_s,
            frequency_hz,
        )
    }
}

/// A selected SP3 product plus its staleness metadata.
///
/// Obtain one from [`select_sp3`] or [`select_sp3_over_range`]. The product is
/// borrowed from the caller's set; the interpolation entry points and the
/// [`EphemerisSource`] impl delegate straight to it, so an exact selection is
/// bit-for-bit identical to interpolating the caller's product directly.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3Selection<'a> {
    sp3: &'a Sp3,
    metadata: StalenessMetadata,
}

impl Sp3Selection<'_> {
    /// The staleness metadata for this selection.
    pub fn metadata(&self) -> StalenessMetadata {
        self.metadata
    }

    /// The selected SP3 product.
    pub fn sp3(&self) -> &Sp3 {
        self.sp3
    }

    /// Interpolate `sat` at a J2000-second epoch on the selected product.
    ///
    /// Delegates to [`Sp3::position_at_j2000_seconds`](crate::ephemeris::Sp3),
    /// so an exact selection is bit-for-bit identical to calling it on the
    /// caller's product.
    pub fn position_at_j2000_seconds(
        &self,
        sat: GnssSatelliteId,
        query_j2000_s: f64,
    ) -> crate::Result<Sp3State> {
        self.sp3.position_at_j2000_seconds(sat, query_j2000_s)
    }
}

impl EphemerisSource for Sp3Selection<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.sp3.position_clock_at_j2000_s(sat, t_j2000_s)
    }
}

/// Select an IONEX product usable at `requested_epoch_j2000_s`, degrading to a
/// diurnal-shifted prior product within `policy` when the exact day is absent.
///
/// See [`select_ionex_over_range`]; this is the single-epoch case.
pub fn select_ionex(
    products: &[Ionex],
    requested_epoch_j2000_s: i64,
    policy: StalenessPolicy,
) -> Result<IonexSelection<'_>, SelectionError> {
    select_ionex_over_range(
        products,
        requested_epoch_j2000_s,
        requested_epoch_j2000_s,
        policy,
    )
}

/// Select an IONEX product usable across `[start, end]` (J2000 seconds).
///
/// Resolution order:
/// 1. If a product covers the whole range, it is returned unchanged
///    ([`DegradationKind::Exact`], zero staleness). When several products cover
///    the range the choice is deterministic: the one with the latest start epoch
///    (freshest), ties broken by the smallest last epoch (tightest span), then
///    by slice order.
/// 2. Otherwise the prior products (last epoch before `start`) are tried
///    freshest-first. Each is advanced by whole days so its grid lands on the
///    range end (the most-stale point), via diurnal persistence
///    ([`DegradationKind::DiurnalShift`]); the first whose shifted grid actually
///    covers the whole range and fits the cap is returned. Trying candidates in
///    order means a partial freshest product cannot mask an older, wider product
///    that does cover.
/// 3. If no prior product covers the range after shifting, or the freshest prior
///    already exceeds the staleness cap, or no prior product exists, a typed
///    [`SelectionError`] is returned.
pub fn select_ionex_over_range(
    products: &[Ionex],
    start_epoch_j2000_s: i64,
    end_epoch_j2000_s: i64,
    policy: StalenessPolicy,
) -> Result<IonexSelection<'_>, SelectionError> {
    validate_policy(policy)?;
    if products.is_empty() {
        return Err(SelectionError::EmptyProductSet);
    }
    if end_epoch_j2000_s < start_epoch_j2000_s {
        return Err(SelectionError::InvalidRange {
            start_epoch_j2000_s: start_epoch_j2000_s as f64,
            end_epoch_j2000_s: end_epoch_j2000_s as f64,
        });
    }

    // 1. Exact coverage of the whole range, with a deterministic tie-break:
    //    latest start (freshest), then smallest last epoch (tightest span).
    let mut exact: Option<(&Ionex, i64, i64)> = None;
    for product in products {
        let (lo, hi) = ionex_span(product)?;
        if lo <= start_epoch_j2000_s && end_epoch_j2000_s <= hi {
            let better = match exact {
                None => true,
                Some((_, best_lo, best_hi)) => lo > best_lo || (lo == best_lo && hi < best_hi),
            };
            if better {
                exact = Some((product, lo, hi));
            }
        }
    }
    if let Some((product, _, _)) = exact {
        return Ok(IonexSelection {
            ionex: Cow::Borrowed(product),
            metadata: StalenessMetadata::exact(end_epoch_j2000_s as f64),
        });
    }

    // 2. Diurnal-shift from a prior product (last epoch before the range start),
    //    tried freshest-first. A partial freshest product whose shifted grid does
    //    not cover the range must not mask an older, wider product that does, so
    //    the candidates are walked in order and the first that both fits the cap
    //    and covers the range wins.
    let mut priors: Vec<(&Ionex, i64, i64)> = products
        .iter()
        .filter_map(|product| match ionex_span(product) {
            Ok((lo, hi)) if hi < start_epoch_j2000_s => Some(Ok((product, lo, hi))),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<_, _>>()?;
    if priors.is_empty() {
        return Err(SelectionError::NoPriorProduct {
            requested_epoch_j2000_s: end_epoch_j2000_s as f64,
        });
    }
    // Freshest (largest last epoch) first; ties broken by the widest span
    // (smallest first epoch), which is the most likely to cover after shifting.
    // Staleness is then monotonically non-decreasing down the list.
    priors.sort_by(|a, b| b.2.cmp(&a.2).then(a.1.cmp(&b.1)));

    // Every arithmetic step is checked, and a candidate that cannot be evaluated
    // (or shifted) within the i64 axis is skipped rather than aborting the scan,
    // so a fresher non-representable candidate cannot mask an older usable one.
    // The terminal error reflects the first binding reason a usable result was
    // not produced: cap exceedance, then overflow, then no covering grid.
    let mut beyond_cap: Option<(i64, i64)> = None; // (source_epoch, staleness)
    let mut overflow_ctx: Option<&'static str> = None;
    for (product, lo, hi) in priors {
        // Whole-day shift that brings the source grid up onto the range end.
        // Ceil division avoids the `gap + 86399` term, which could overflow even
        // when the shifted epoch itself would fit.
        let Some(gap_s) = end_epoch_j2000_s.checked_sub(hi) else {
            overflow_ctx.get_or_insert("end - hi");
            continue;
        }; // > 0 by selection
        let days = gap_s / SECONDS_PER_DAY_I64 + i64::from(gap_s % SECONDS_PER_DAY_I64 != 0); // >= 1
        let Some(staleness_s) = days.checked_mul(SECONDS_PER_DAY_I64) else {
            overflow_ctx.get_or_insert("days * 86400");
            continue;
        };
        let Some(source_epoch_j2000_s) = end_epoch_j2000_s.checked_sub(staleness_s) else {
            overflow_ctx.get_or_insert("end - staleness");
            continue;
        };

        // Staleness is non-decreasing down the list, so once one candidate
        // exceeds the cap every remaining (older) candidate does too. Record the
        // freshest (least-stale) exceedance and stop.
        if staleness_s as f64 > policy.max_staleness_s {
            beyond_cap = Some((source_epoch_j2000_s, staleness_s));
            break;
        }

        // The shifted grid is the source span advanced by `staleness_s`; compute
        // its bounds with checked arithmetic so an unrepresentable shift skips to
        // the next candidate instead of failing the whole request.
        let (Some(shifted_lo), Some(shifted_hi)) =
            (lo.checked_add(staleness_s), hi.checked_add(staleness_s))
        else {
            overflow_ctx.get_or_insert("epoch + staleness");
            continue;
        };

        // The ceil shift lands the grid's last epoch at or past the range end,
        // but a partial product can still start after the range start once
        // shifted. Only a grid that actually covers the request is usable; if it
        // does not, fall through to the next (older, wider) candidate.
        if shifted_lo <= start_epoch_j2000_s && end_epoch_j2000_s <= shifted_hi {
            // Bounds are representable, so the full shift cannot overflow.
            let shifted = product
                .with_map_epochs_shifted_days(days)
                .map_err(|error| SelectionError::InvalidProduct(error.to_string()))?;
            return Ok(IonexSelection {
                ionex: Cow::Owned(shifted),
                metadata: StalenessMetadata {
                    kind: DegradationKind::DiurnalShift,
                    requested_epoch_j2000_s: end_epoch_j2000_s as f64,
                    source_epoch_j2000_s: source_epoch_j2000_s as f64,
                    staleness_s: staleness_s as f64,
                    staleness_days: days as f64,
                },
            });
        }
    }

    if let Some((source_epoch_j2000_s, staleness_s)) = beyond_cap {
        return Err(SelectionError::BeyondStalenessCap {
            requested_epoch_j2000_s: end_epoch_j2000_s as f64,
            source_epoch_j2000_s: source_epoch_j2000_s as f64,
            staleness_s: staleness_s as f64,
            max_staleness_s: policy.max_staleness_s,
        });
    }
    if let Some(context) = overflow_ctx {
        return Err(SelectionError::Overflow { context });
    }
    // Every prior product within the cap was too partial to cover the range once
    // shifted onto it.
    Err(SelectionError::InvalidProduct(format!(
        "no prior IONEX product covers requested range \
         [{start_epoch_j2000_s}, {end_epoch_j2000_s}] J2000 s after a whole-day diurnal shift"
    )))
}

/// Select an SP3 product usable at `requested_epoch_j2000_s`, degrading to the
/// most-recent prior product within `policy`.
///
/// See [`select_sp3_over_range`]; this is the single-epoch case.
pub fn select_sp3(
    products: &[Sp3],
    requested_epoch_j2000_s: f64,
    policy: StalenessPolicy,
) -> Result<Sp3Selection<'_>, SelectionError> {
    select_sp3_over_range(
        products,
        requested_epoch_j2000_s,
        requested_epoch_j2000_s,
        policy,
    )
}

/// Select an SP3 product usable across `[start, end]` (J2000 seconds).
///
/// Resolution order:
/// 1. If a product covers the whole range, it is returned unchanged
///    ([`DegradationKind::Exact`], zero staleness). When several products cover
///    the range the choice is deterministic: the one with the latest start epoch
///    (freshest), ties broken by the smallest last epoch (tightest span), then
///    by slice order.
/// 2. Otherwise the most-recent product that covers the range start but ends
///    before the range end is selected as-is ([`DegradationKind::NearestPrior`]),
///    with staleness measured from that last epoch to the range end (the
///    most-stale point). Requiring it to cover the start (`lo <= start`) keeps
///    out a product beginning after the range start, which could not serve the
///    start; a product entirely before the range qualifies trivially. This also
///    admits a product that covers the start but ends before the end — the
///    nearest-prior source for the worst-case end.
/// 3. If that staleness exceeds the cap, or no prior product exists, a typed
///    [`SelectionError`] is returned.
pub fn select_sp3_over_range(
    products: &[Sp3],
    start_epoch_j2000_s: f64,
    end_epoch_j2000_s: f64,
    policy: StalenessPolicy,
) -> Result<Sp3Selection<'_>, SelectionError> {
    validate_policy(policy)?;
    if products.is_empty() {
        return Err(SelectionError::EmptyProductSet);
    }
    if !start_epoch_j2000_s.is_finite()
        || !end_epoch_j2000_s.is_finite()
        || end_epoch_j2000_s < start_epoch_j2000_s
    {
        return Err(SelectionError::InvalidRange {
            start_epoch_j2000_s,
            end_epoch_j2000_s,
        });
    }

    // 1. Exact coverage of the whole range, with a deterministic tie-break:
    //    latest start (freshest), then smallest last epoch (tightest span).
    let mut exact: Option<(&Sp3, f64, f64)> = None;
    for product in products {
        let (lo, hi) = sp3_span(product)?;
        if lo <= start_epoch_j2000_s && end_epoch_j2000_s <= hi {
            let better = match exact {
                None => true,
                Some((_, best_lo, best_hi)) => lo > best_lo || (lo == best_lo && hi < best_hi),
            };
            if better {
                exact = Some((product, lo, hi));
            }
        }
    }
    if let Some((product, _, _)) = exact {
        return Ok(Sp3Selection {
            sp3: product,
            metadata: StalenessMetadata::exact(end_epoch_j2000_s),
        });
    }

    // 2. Most-recent product that covers the range start but ends before the
    //    range end: it is the nearest-prior source for the worst-case end. The
    //    `lo <= start` guard keeps out a product that begins after the range
    //    start (it cannot serve the start at all, so it is not a usable prior);
    //    a product entirely before the range satisfies it trivially.
    let mut best: Option<(&Sp3, f64)> = None;
    for product in products {
        let (lo, hi) = sp3_span(product)?;
        if lo <= start_epoch_j2000_s
            && hi < end_epoch_j2000_s
            && best.is_none_or(|(_, best_hi)| hi > best_hi)
        {
            best = Some((product, hi));
        }
    }
    let (product, hi) = best.ok_or(SelectionError::NoPriorProduct {
        requested_epoch_j2000_s: end_epoch_j2000_s,
    })?;

    let staleness_s = end_epoch_j2000_s - hi; // > 0 by selection
    if staleness_s > policy.max_staleness_s {
        return Err(SelectionError::BeyondStalenessCap {
            requested_epoch_j2000_s: end_epoch_j2000_s,
            source_epoch_j2000_s: hi,
            staleness_s,
            max_staleness_s: policy.max_staleness_s,
        });
    }

    Ok(Sp3Selection {
        sp3: product,
        metadata: StalenessMetadata {
            kind: DegradationKind::NearestPrior,
            requested_epoch_j2000_s: end_epoch_j2000_s,
            source_epoch_j2000_s: hi,
            staleness_s,
            staleness_days: staleness_s / SECONDS_PER_DAY,
        },
    })
}

/// The `[first, last]` IONEX map-epoch span in J2000 seconds.
fn ionex_span(product: &Ionex) -> Result<(i64, i64), SelectionError> {
    let epochs = product.map_epochs_s();
    let first = *epochs
        .first()
        .ok_or_else(|| SelectionError::InvalidProduct("IONEX product has no maps".into()))?;
    let last = *epochs.last().expect("non-empty epochs has a last element");
    Ok((first, last))
}

/// The `[first, last]` SP3 epoch span in J2000 seconds.
fn sp3_span(product: &Sp3) -> Result<(f64, f64), SelectionError> {
    let epochs = product.epochs_j2000_seconds();
    let first = *epochs
        .first()
        .ok_or_else(|| SelectionError::InvalidProduct("SP3 product has no epochs".into()))?;
    let last = *epochs.last().expect("non-empty epochs has a last element");
    Ok((first, last))
}

#[cfg(test)]
mod tests;
