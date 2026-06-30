pub mod eclipse;
pub mod root;
pub mod r#trait;

pub use r#trait::{
    CrossingDirection, CrossingEvent, DiscreteEventPredicate, EventFinder, EventFinderError,
    ExtremumEvent, ExtremumKind, ScalarEventPredicate, StateChangeEvent,
};

#[derive(Debug, Clone)]
pub struct DetectedEvent {
    pub epoch_tdb_seconds: f64,
    pub name: String,
    // Additional fields as needed
}
