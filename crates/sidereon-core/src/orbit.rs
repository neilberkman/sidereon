//! Compact orbit approximations.
//!
//! Reduced orbits are fitted approximations for caching, transport, and quick
//! visibility math. They are not precise ephemeris products; use
//! [`crate::ephemeris::Sp3`] or [`crate::ephemeris::BroadcastEphemeris`] when
//! full-fidelity products are available.

pub use crate::reduced_orbit::{
    drift, drift_piecewise_reduced_orbit_source, drift_reduced_orbit_source, fit, fit_piecewise,
    fit_piecewise_reduced_orbit_source, fit_reduced_orbit_source, fit_with_model, piecewise_drift,
    piecewise_position, piecewise_position_velocity, position, position_velocity,
    select_piecewise_segment, CalendarEpoch, DriftEntry, DriftReport, EcefSample, Elements,
    FitStats, Frame, Model, PiecewiseOrbit, PiecewiseOrbitError, PiecewiseOrbitSourceFit,
    PiecewiseOrbitSourceFitOptions, PiecewiseSegment, ReducedOrbit, ReducedOrbitError,
    ReducedOrbitSource, ReducedOrbitSourceDrift, ReducedOrbitSourceDriftOptions,
    ReducedOrbitSourceError, ReducedOrbitSourceFit, ReducedOrbitSourceFitOptions,
    ReducedOrbitSourceSampling, MIN_SAMPLES,
};

/// Role-oriented alias for a fitted reduced-orbit model.
pub type ReducedOrbitModel = ReducedOrbit;

/// Error type returned by reduced-orbit fitting/evaluation.
pub type Error = ReducedOrbitError;
