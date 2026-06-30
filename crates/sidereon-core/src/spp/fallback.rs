//! First-class broadcast-ephemeris positioning and a precise-with-broadcast
//! fallback entry that carries source and staleness provenance.
//!
//! Precise products (SP3 orbit and clock) deliver the most accurate satellite
//! positions, but they publish with latency and require a network fetch, so the
//! product for the exact requested epoch is not always on hand. Broadcast
//! ephemeris, decoded from the navigation message a receiver already tracks, is
//! always available and needs no network, which is what enables real-time and
//! offline positioning. The accuracy gap between the two is bounded and well
//! characterized (see the accuracy-delta note below), so a system that prefers
//! precise when it is fresh and degrades to broadcast otherwise gets the best
//! available fix at every epoch without ever stalling.
//!
//! This module wires those two paths into the public surface:
//!
//! - [`solve_broadcast`] is the explicit broadcast-only SPP entry. A
//!   [`BroadcastEphemeris`] is an [`EphemerisSource`], so feeding it to the
//!   generic [`solve`](crate::positioning::solve) already works; this is the
//!   supported, named real-time/offline mode rather than that fact left implicit.
//!   The decode-to-source half of the pipeline is
//!   [`BroadcastRecord::from_lnav`](crate::ephemeris::BroadcastRecord::from_lnav),
//!   which turns decoded GPS LNAV subframes into a record a
//!   [`BroadcastEphemeris`] can hold, so the full chain is
//!   `lnav::decode -> BroadcastRecord::from_lnav -> BroadcastStore -> solve_broadcast`.
//! - [`solve_with_fallback`] is the unified entry: try the precise path through
//!   the product-staleness selection layer ([`select_sp3`]); if no precise
//!   product covers the epoch or the nearest one is beyond the staleness cap,
//!   fall back to the broadcast path. The result is a [`SourcedSolution`] whose
//!   [`FixSource`] names which source produced the fix (precise-exact,
//!   precise-degraded, or broadcast) and carries the [`StalenessMetadata`] /
//!   rejection reason, so a degraded or substituted answer is never silent.
//!
//! # Correctness
//!
//! When a precise product covers the requested epoch, the selection layer returns
//! the caller's product untouched and the fallback solve is bit-for-bit identical
//! to calling [`solve`](crate::positioning::solve) on that SP3 directly: the
//! broadcast path is purely additive and changes no precise-present output bit. A
//! solve failure on a product that covers the exact epoch is a genuine error,
//! surfaced as [`FallbackError::Precise`] rather than masked by silently
//! re-solving on broadcast. Broadcast is used when the staleness selection
//! declines outright, or when a stale-but-within-cap product is selected and then
//! cannot serve the epoch; in both cases the result's [`FixSource::Broadcast`]
//! records the reason ([`BroadcastReason`]), so the source is never substituted
//! silently.
//!
//! # Expected broadcast-vs-precise accuracy delta
//!
//! The broadcast and precise SPP solutions differ by the broadcast signal-in-space
//! range error (SISRE): the broadcast orbit and clock are a least-squares fit and
//! a polynomial extrapolation, where the precise product is a post-processed
//! estimate. For healthy GPS the broadcast orbit error is roughly 1-2 m RMS (3D),
//! dominated by the along-track and radial components, and the broadcast satellite
//! clock adds a comparable error (see [`crate::broadcast_comparison`], which
//! measures exactly this on a committed reference arc). A common per-epoch clock
//! offset absorbs into the estimated receiver clock, but the per-satellite orbit
//! error and clock scatter do not, and on an L1-only solve the broadcast clock
//! (which subtracts TGD for the single-frequency user) differs from the precise
//! ionosphere-free SP3 clock (no TGD) by a further per-satellite amount. Mapped
//! through the geometry, the *position* difference between a broadcast-only and a
//! precise SPP fix on the same pseudoranges is therefore at the ~10 m level at a
//! single epoch (not merely the orbit RMS). The reference-arc integration test
//! `broadcast_spp_fallback_arc` measures ~13 m and asserts agreement within a
//! labeled 20 m bound; that bound is the documented accuracy delta, not a
//! bit-exact claim (two orbit/clock sources legitimately differ at the meter
//! level).
//!
//! # Network
//!
//! This module is pure and no-network, like the rest of `sidereon-core`: it
//! selects among products the caller has already parsed and solves in memory.
//! Fetching SP3/clock products or collecting the navigation message is a
//! per-binding concern.

use crate::ephemeris::{BroadcastEphemeris, Sp3};
use crate::staleness::{select_sp3, SelectionError, StalenessMetadata, StalenessPolicy};

use super::{solve, EphemerisSource, ReceiverSolution, SolveInputs, SppError};

/// Which ephemeris source produced a [`SourcedSolution`], with its provenance.
///
/// A fallback solve never substitutes a source silently: this enum is always
/// present on the result and records both which source was used and how it
/// related to the requested epoch.
///
/// This is not [`PartialEq`] because its broadcast reason can carry an
/// [`SppError`], which is not comparable; classify with the `is_*` accessors or
/// match on the variant.
#[derive(Debug, Clone)]
pub enum FixSource {
    /// A precise SP3 product produced the fix. The carried [`StalenessMetadata`]
    /// distinguishes a precise-exact result
    /// ([`DegradationKind::Exact`](crate::staleness::DegradationKind::Exact),
    /// zero staleness) from a precise-degraded one
    /// ([`DegradationKind::NearestPrior`](crate::staleness::DegradationKind::NearestPrior),
    /// nonzero staleness) and reports the source epoch and staleness.
    Precise(StalenessMetadata),
    /// The broadcast ephemeris path produced the fix because the precise path was
    /// not used. The carried [`BroadcastReason`] explains why, so the substitution
    /// is always explicit.
    Broadcast(BroadcastReason),
}

/// Why [`solve_with_fallback`] produced a fix from broadcast ephemeris.
///
/// A broadcast fix is never substituted silently: the result records whether the
/// precise selection was declined outright, or a stale-but-within-cap precise
/// product was selected and then turned out unusable for the requested epoch.
#[derive(Debug, Clone)]
pub enum BroadcastReason {
    /// The precise product staleness selection declined: there was no precise
    /// product set, none covering or preceding the epoch, or the nearest product
    /// was beyond the staleness cap. The selection layer's [`SelectionError`] is
    /// the exact reason.
    PreciseUnavailable(SelectionError),
    /// A stale (within-cap) precise product was selected, but it could not produce
    /// a fix for the requested epoch -- typically its coverage does not reach the
    /// epoch (an SP3 nearest-prior product ends before it). This is the
    /// "precise unavailable for this epoch" condition the fallback exists for, so
    /// broadcast was used; the selected product's staleness and the precise solve
    /// error are carried so the degraded-then-fell-back path is explicit. A solve
    /// failure on a product that DOES cover the epoch is a genuine error and is
    /// returned as [`FallbackError::Precise`] instead, not turned into this.
    PreciseDegradedUnusable {
        /// Staleness of the degraded precise product that was tried.
        staleness: StalenessMetadata,
        /// The precise solve error that triggered the fallback.
        error: SppError,
    },
}

impl BroadcastReason {
    /// The precise selection's staleness for the degraded-then-fell-back case, or
    /// `None` when the precise selection was declined outright. This is the
    /// staleness of the precise product that was *not* used; the broadcast fix
    /// itself carries no precise staleness.
    pub fn attempted_staleness(&self) -> Option<StalenessMetadata> {
        match self {
            BroadcastReason::PreciseUnavailable(_) => None,
            BroadcastReason::PreciseDegradedUnusable { staleness, .. } => Some(*staleness),
        }
    }
}

impl FixSource {
    /// Whether a precise SP3 product produced the fix (exact or degraded).
    pub fn is_precise(&self) -> bool {
        matches!(self, FixSource::Precise(_))
    }

    /// Whether the broadcast path produced the fix.
    pub fn is_broadcast(&self) -> bool {
        matches!(self, FixSource::Broadcast(_))
    }

    /// Whether a precise product covering the exact epoch produced the fix (no
    /// degradation, zero staleness).
    pub fn is_precise_exact(&self) -> bool {
        matches!(self, FixSource::Precise(meta) if meta.kind.is_exact())
    }

    /// The staleness metadata of the source that produced the fix: the precise
    /// product's staleness for a precise fix, or `None` for a broadcast fix (the
    /// broadcast fix is not backed by a precise product). For the
    /// degraded-then-fell-back case, the staleness of the precise product that was
    /// *tried* is available via
    /// [`BroadcastReason::attempted_staleness`].
    pub fn staleness(&self) -> Option<StalenessMetadata> {
        match self {
            FixSource::Precise(meta) => Some(*meta),
            FixSource::Broadcast(_) => None,
        }
    }
}

/// A receiver solution paired with the provenance of the ephemeris that produced
/// it.
///
/// Returned by [`solve_with_fallback`]. The public language bindings wrap this as
/// the real-time positioning result so callers always see which source and how
/// stale the fix is.
#[derive(Debug, Clone)]
pub struct SourcedSolution {
    /// The solved receiver position/clock with its geometry diagnostics.
    pub solution: ReceiverSolution,
    /// Which ephemeris source produced the fix, with its staleness/rejection
    /// provenance.
    pub source: FixSource,
}

/// Error from [`solve_with_fallback`], tagged with which path failed.
#[derive(Debug, Clone)]
pub enum FallbackError {
    /// A usable precise product was selected but its SPP solve failed. The
    /// fallback does not silently re-solve on broadcast in this case, since the
    /// precise product was fresh enough to use; the underlying solve error is
    /// surfaced.
    Precise(SppError),
    /// The broadcast fallback path was taken (the precise selection was declined)
    /// and its SPP solve failed.
    Broadcast(SppError),
}

impl core::fmt::Display for FallbackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FallbackError::Precise(error) => write!(f, "precise SPP solve failed: {error}"),
            FallbackError::Broadcast(error) => {
                write!(f, "broadcast-fallback SPP solve failed: {error}")
            }
        }
    }
}

impl std::error::Error for FallbackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FallbackError::Precise(error) | FallbackError::Broadcast(error) => Some(error),
        }
    }
}

/// Solve a receiver position from broadcast ephemeris ALONE: the supported
/// real-time / offline single-point-positioning mode.
///
/// This is the explicit broadcast-only entry point. Broadcast ephemeris decoded
/// from the navigation message is always available and needs no network, so this
/// is the path a receiver uses when no precise product is on hand. It is a thin,
/// named wrapper over the generic [`solve`](crate::positioning::solve): a
/// [`BroadcastEphemeris`] is an [`EphemerisSource`], so the result is bit-for-bit
/// identical to calling `solve(&broadcast, inputs, with_geodetic)`. Taking the
/// concrete [`BroadcastEphemeris`] makes the broadcast-only contract explicit in
/// the type system rather than relying on the caller to pass the right source.
///
/// The store can come from a parsed RINEX navigation file
/// ([`BroadcastEphemeris::from_nav`](crate::ephemeris::BroadcastEphemeris::from_nav))
/// or from records decoded straight off the air via
/// [`BroadcastRecord::from_lnav`](crate::ephemeris::BroadcastRecord::from_lnav)
/// and [`BroadcastEphemeris::new`](crate::ephemeris::BroadcastEphemeris::new),
/// which closes the `lnav::decode -> broadcast source` half of the real-time
/// pipeline.
pub fn solve_broadcast(
    broadcast: &BroadcastEphemeris,
    inputs: &SolveInputs,
    with_geodetic: bool,
) -> Result<ReceiverSolution, SppError> {
    solve(broadcast, inputs, with_geodetic)
}

/// Solve a receiver position, preferring precise products and falling back to
/// broadcast ephemeris, reporting which source was used and how stale it is.
///
/// The precise path is tried first through the product-staleness selection layer
/// ([`select_sp3`]) at the receive epoch (`inputs.t_rx_j2000_s`):
///
/// - If a precise product covers the epoch ([`DegradationKind::Exact`]) it is
///   used. The solve is bit-for-bit identical to
///   [`solve`](crate::positioning::solve) on that SP3 (the selection layer borrows
///   the caller's product untouched), and the result is
///   [`FixSource::Precise`] with zero staleness. A solve failure here is a genuine
///   error (the data covers the epoch), returned as [`FallbackError::Precise`],
///   never masked by a silent broadcast re-solve.
/// - If a stale-but-within-cap precise product is selected
///   ([`DegradationKind::NearestPrior`]) and it actually produces a fix, the
///   result is [`FixSource::Precise`] carrying the nonzero
///   [`StalenessMetadata`]. If instead it cannot serve the requested epoch (its
///   coverage ends before it, so the solve fails on missing ephemeris), broadcast
///   produces the fix and the result is
///   [`FixSource::Broadcast`]`(`[`BroadcastReason::PreciseDegradedUnusable`]`)`,
///   carrying the tried product's staleness and the precise solve error. This is
///   the "precise unavailable for this epoch" condition the fallback exists for.
/// - If the precise selection is declined outright (no product set, none covering
///   or preceding the epoch, or the nearest beyond the staleness cap), broadcast
///   produces the fix and the result is
///   [`FixSource::Broadcast`]`(`[`BroadcastReason::PreciseUnavailable`]`)` carrying
///   the selection layer's [`SelectionError`].
///
/// A broadcast fix is therefore never substituted silently: its [`BroadcastReason`]
/// always records why precise was not used.
///
/// `policy` bounds how stale a precise product may be before broadcast is
/// preferred; a generous cap keeps precise in use across normal product latency,
/// a zero cap forces broadcast whenever no product covers the exact epoch.
pub fn solve_with_fallback(
    precise: &[Sp3],
    broadcast: &dyn EphemerisSource,
    inputs: &SolveInputs,
    policy: StalenessPolicy,
    with_geodetic: bool,
) -> Result<SourcedSolution, FallbackError> {
    match select_sp3(precise, inputs.t_rx_j2000_s, policy) {
        Ok(selection) => {
            let metadata = selection.metadata();
            match solve(&selection, inputs, with_geodetic) {
                Ok(solution) => Ok(SourcedSolution {
                    solution,
                    source: FixSource::Precise(metadata),
                }),
                Err(error) if metadata.kind.is_exact() => {
                    // The product covers the exact epoch, so a solve failure is a
                    // genuine error (geometry, inputs, or a real ephemeris gap),
                    // not staleness. Surface it rather than masking it on broadcast.
                    Err(FallbackError::Precise(error))
                }
                Err(error) => {
                    // A degraded (stale, within-cap) product was selected but could
                    // not produce a fix for the requested epoch. That is exactly the
                    // condition the fallback exists for, so use broadcast and record
                    // the degraded-then-fell-back provenance.
                    broadcast_fix(
                        broadcast,
                        inputs,
                        with_geodetic,
                        BroadcastReason::PreciseDegradedUnusable {
                            staleness: metadata,
                            error,
                        },
                    )
                }
            }
        }
        Err(precise_rejection) => broadcast_fix(
            broadcast,
            inputs,
            with_geodetic,
            BroadcastReason::PreciseUnavailable(precise_rejection),
        ),
    }
}

/// Solve on the broadcast source and tag the result with `reason`.
fn broadcast_fix(
    broadcast: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    reason: BroadcastReason,
) -> Result<SourcedSolution, FallbackError> {
    let solution = solve(broadcast, inputs, with_geodetic).map_err(FallbackError::Broadcast)?;
    Ok(SourcedSolution {
        solution,
        source: FixSource::Broadcast(reason),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staleness::DegradationKind;

    fn meta(kind: DegradationKind, staleness_s: f64) -> StalenessMetadata {
        StalenessMetadata {
            kind,
            requested_epoch_j2000_s: 1000.0,
            source_epoch_j2000_s: 1000.0 - staleness_s,
            staleness_s,
            staleness_days: staleness_s / 86_400.0,
        }
    }

    #[test]
    fn fix_source_precise_exact_classification() {
        let exact = FixSource::Precise(meta(DegradationKind::Exact, 0.0));
        assert!(exact.is_precise());
        assert!(exact.is_precise_exact());
        assert!(!exact.is_broadcast());
        assert_eq!(exact.staleness().map(|m| m.staleness_s), Some(0.0));
    }

    #[test]
    fn fix_source_precise_degraded_is_not_exact() {
        let degraded = FixSource::Precise(meta(DegradationKind::NearestPrior, 3600.0));
        assert!(degraded.is_precise());
        assert!(!degraded.is_precise_exact());
        assert_eq!(degraded.staleness().map(|m| m.staleness_s), Some(3600.0));
    }

    #[test]
    fn fix_source_broadcast_unavailable_has_no_staleness_and_carries_reason() {
        let broadcast = FixSource::Broadcast(BroadcastReason::PreciseUnavailable(
            SelectionError::EmptyProductSet,
        ));
        assert!(broadcast.is_broadcast());
        assert!(!broadcast.is_precise());
        assert!(!broadcast.is_precise_exact());
        assert_eq!(broadcast.staleness(), None);
        assert!(matches!(
            broadcast,
            FixSource::Broadcast(BroadcastReason::PreciseUnavailable(
                SelectionError::EmptyProductSet
            ))
        ));
    }

    #[test]
    fn broadcast_degraded_reason_exposes_attempted_staleness() {
        let staleness = meta(DegradationKind::NearestPrior, 7200.0);
        let reason = BroadcastReason::PreciseDegradedUnusable {
            staleness,
            error: SppError::TooFewSatellites {
                used: 0,
                required: 4,
            },
        };
        // The broadcast fix carries no precise staleness of its own, but the
        // reason exposes the staleness of the precise product that was tried.
        assert_eq!(
            reason.attempted_staleness().map(|m| m.staleness_s),
            Some(7200.0)
        );
        let unavailable = BroadcastReason::PreciseUnavailable(SelectionError::EmptyProductSet);
        assert_eq!(unavailable.attempted_staleness(), None);
        assert_eq!(FixSource::Broadcast(reason).staleness(), None);
    }
}
