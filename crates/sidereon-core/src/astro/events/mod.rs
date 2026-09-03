pub mod eclipse;
pub mod root;
pub mod r#trait;

pub use r#trait::{
    CrossingDirection, CrossingEvent, DiscreteEventPredicate, EventFinder, EventFinderError,
    ExtremumEvent, ExtremumKind, ScalarEventPredicate, StateChangeEvent,
};

#[derive(Debug, Clone)]
/// Event record stored in [`crate::astro::propagator::PropagationResult::events`].
///
/// The built-in RK4 and DP54 integrators currently return an empty event vector,
/// so no built-in propagation path constructs this record.
pub struct DetectedEvent {
    /// No built-in propagation path currently populates this field: RK4 and
    /// DP54 return an empty [`crate::astro::propagator::PropagationResult::events`]
    /// vector.
    pub epoch_tdb_seconds: f64,
    /// No built-in propagation path currently populates this field: RK4 and
    /// DP54 return an empty [`crate::astro::propagator::PropagationResult::events`]
    /// vector.
    pub name: String,
    // Additional fields as needed
}
