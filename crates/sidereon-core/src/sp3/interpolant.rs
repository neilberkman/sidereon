//! Cached precise-ephemeris interpolant.

use std::collections::BTreeMap;

use crate::astro::time::model::{Instant, TimeScale};
use crate::id::GnssSatelliteId;
use crate::observables::{
    ObservableEphemerisSource, ObservableState, ObservableStateBatch, ObservablesError,
};
use crate::sp3::interp::{
    fit_clock_spline_arcs, gather_sp3_precise_series, instant_to_j2000_seconds,
    interpolate_precise_state, interpolate_precise_state_with_clock_arcs, ClockSplineArc,
    PreciseSatSeries, Sp3InterpolationOptions,
};
use crate::sp3::{
    PreciseEphemerisSample, PreciseEphemerisSamples, PreciseSamplesError, Sp3, Sp3State,
};
use crate::{Error, Result};

/// Error returned while building a [`PreciseEphemerisInterpolant`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreciseInterpolantError {
    /// A sample-backed build failed sample validation.
    Samples(PreciseSamplesError),
}

impl core::fmt::Display for PreciseInterpolantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Samples(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for PreciseInterpolantError {}

impl From<PreciseSamplesError> for PreciseInterpolantError {
    fn from(error: PreciseSamplesError) -> Self {
        Self::Samples(error)
    }
}

/// A reusable precise-ephemeris interpolant with cached per-satellite nodes.
///
/// The handle owns the same native-unit node vectors the scalar SP3 evaluator
/// gathers on each call. Evaluation still uses the shared precise interpolation
/// substrate, so changing from a parsed [`Sp3`] to this handle changes only when
/// nodes are gathered, not the arithmetic recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct PreciseEphemerisInterpolant {
    time_scale: TimeScale,
    pub(super) nodes: BTreeMap<GnssSatelliteId, FittedPreciseSatSeries>,
    pub(super) interpolation: Sp3InterpolationOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct FittedPreciseSatSeries {
    pub(super) series: PreciseSatSeries,
    pub(super) clock_arcs: Vec<ClockSplineArc>,
}

impl FittedPreciseSatSeries {
    pub(super) fn new(series: PreciseSatSeries) -> Self {
        let clock_arcs = fit_clock_spline_arcs(&series.clk);
        Self { series, clock_arcs }
    }
}

impl PreciseEphemerisInterpolant {
    /// Build a cached interpolant from a parsed SP3 product.
    ///
    /// Nodes are copied from the product's native SP3 kilometer and microsecond
    /// values, matching the scalar [`Sp3::position_at_j2000_seconds`] gather.
    /// The real-product decimation hold-out oracle pins the parsed-text versus
    /// sample-backed 3D difference at no more than `4.020965667248365e-8` m
    /// over interior held-out 5-minute records bracketed by 15-minute nodes.
    pub fn from_sp3(source: &Sp3) -> Self {
        let mut nodes = BTreeMap::new();
        for &sat in source.satellites() {
            let series = gather_sp3_precise_series(source, sat);
            if !series.x.is_empty() {
                nodes.insert(sat, FittedPreciseSatSeries::new(series));
            }
        }
        Self {
            time_scale: source.header.time_scale,
            nodes,
            interpolation: source.interpolation_options(),
        }
    }

    /// Build a cached interpolant from precise samples.
    ///
    /// This validates samples through [`PreciseEphemerisSamples::from_samples`]
    /// and then copies the prepared native-unit node series into this handle.
    /// The real-product decimation hold-out oracle pins the sample-backed versus
    /// parsed-text 3D difference at no more than `4.020965667248365e-8` m over
    /// interior held-out 5-minute records bracketed by 15-minute nodes.
    pub fn from_samples(
        samples: impl IntoIterator<Item = PreciseEphemerisSample>,
    ) -> core::result::Result<Self, PreciseInterpolantError> {
        let source = PreciseEphemerisSamples::from_samples(samples)?;
        Ok(Self::from_precise_ephemeris_samples(&source))
    }

    /// Build a cached interpolant from an existing sample-backed source.
    pub fn from_precise_ephemeris_samples(source: &PreciseEphemerisSamples) -> Self {
        Self {
            time_scale: source.time_scale(),
            nodes: source
                .node_series()
                .iter()
                .map(|(&sat, series)| (sat, FittedPreciseSatSeries::new(series.clone())))
                .collect(),
            interpolation: source.interpolation_options(),
        }
    }

    /// The time scale of the source epochs used to build this handle.
    pub fn time_scale(&self) -> TimeScale {
        self.time_scale
    }

    /// The interpretation policy this handle reads its node series with.
    ///
    /// Copied from the source product or samples at build time, and written
    /// into a store built from this handle.
    pub fn interpolation_options(&self) -> Sp3InterpolationOptions {
        self.interpolation
    }

    /// Return the handle with a different interpretation policy.
    #[must_use]
    pub fn with_interpolation_options(mut self, options: Sp3InterpolationOptions) -> Self {
        self.interpolation = options;
        self
    }

    pub(super) fn node_series(&self) -> &BTreeMap<GnssSatelliteId, FittedPreciseSatSeries> {
        &self.nodes
    }

    /// The satellites this handle can interpolate, in ascending order.
    pub fn satellites(&self) -> impl Iterator<Item = GnssSatelliteId> + '_ {
        self.nodes.keys().copied()
    }

    /// Interpolate the state of `sat` at an arbitrary J2000-second epoch.
    ///
    /// The error surface matches [`Sp3::position_at_j2000_seconds`]:
    /// [`Error::UnknownSatellite`] for a satellite with no nodes,
    /// [`Error::EpochOutOfRange`] for an out-of-coverage query, and
    /// [`Error::InvalidInput`] for a non-finite query.
    pub fn position_at_j2000_seconds(&self, sat: GnssSatelliteId, query: f64) -> Result<Sp3State> {
        static EMPTY_F64: [f64; 0] = [];
        static EMPTY_CLK: [(f64, f64, bool); 0] = [];
        match self.nodes.get(&sat) {
            Some(fitted) => interpolate_precise_state_with_clock_arcs(
                sat,
                &fitted.series.x,
                &fitted.series.kx,
                &fitted.series.ky,
                &fitted.series.kz,
                &fitted.clock_arcs,
                query,
                self.interpolation.gap_threshold_factor(),
            ),
            None => interpolate_precise_state(
                sat,
                &EMPTY_F64,
                &EMPTY_F64,
                &EMPTY_F64,
                &EMPTY_F64,
                &EMPTY_CLK,
                query,
                self.interpolation.gap_threshold_factor(),
            ),
        }
    }

    /// Interpolate the state of `sat` at an arbitrary [`Instant`].
    ///
    /// The query instant must use the same time scale as the source used to
    /// build this handle.
    pub fn position(&self, sat: GnssSatelliteId, epoch: Instant) -> Result<Sp3State> {
        if epoch.scale != self.time_scale {
            return Err(Error::InvalidInput(format!(
                "precise-interpolant query time scale {} does not match source time scale {}",
                epoch.scale.abbrev(),
                self.time_scale.abbrev()
            )));
        }
        let query = instant_to_j2000_seconds(&epoch).ok_or(Error::EpochOutOfRange)?;
        self.position_at_j2000_seconds(sat, query)
    }

    /// ECEF states for parallel satellite and epoch arrays.
    ///
    /// This is the same output contract as
    /// [`ObservableEphemerisSource::observable_states_at_j2000_s`].
    pub fn observable_states_at_j2000_s(
        &self,
        satellites: &[GnssSatelliteId],
        epochs_j2000_s: &[f64],
    ) -> core::result::Result<ObservableStateBatch, ObservablesError> {
        <Self as ObservableEphemerisSource>::observable_states_at_j2000_s(
            self,
            satellites,
            epochs_j2000_s,
        )
    }

    /// ECEF states for many satellites at one shared epoch.
    ///
    /// This is the same output contract as
    /// [`ObservableEphemerisSource::observable_states_at_shared_j2000_s`].
    pub fn observable_states_at_shared_j2000_s(
        &self,
        satellites: &[GnssSatelliteId],
        epoch_j2000_s: f64,
    ) -> ObservableStateBatch {
        <Self as ObservableEphemerisSource>::observable_states_at_shared_j2000_s(
            self,
            satellites,
            epoch_j2000_s,
        )
    }
}

impl ObservableEphemerisSource for PreciseEphemerisInterpolant {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> core::result::Result<ObservableState, ObservablesError> {
        let state = self
            .position_at_j2000_seconds(sat, t_j2000_s)
            .map_err(ObservablesError::Ephemeris)?;
        Ok(ObservableState {
            position_ecef_m: state.position.as_array(),
            clock_s: state.clock_s,
        })
    }
}
