//! General event-finding primitives over scalar and discrete predicates.
//!
//! The finder samples a predicate over a uniform time grid, brackets events
//! between neighboring samples, and refines event times to a requested
//! tolerance. Time is represented as J2000-relative seconds by convention, but
//! the machinery only requires a monotonic scalar time coordinate.

use crate::astro::events::root::{sign_change_bracketed, try_bisect_crossing_until, RootError};
use crate::validate;
use rayon::prelude::*;

const GOLDEN_RESPHI: f64 = 0.381_966_011_250_105_1;
const MAX_GOLDEN_ITERATIONS: usize = 128;
const MAX_EVENT_COARSE_SAMPLES: usize = 1_000_000;

/// A scalar function of time used by the general event finder.
pub trait ScalarEventPredicate {
    /// Return the scalar value at `time_seconds`.
    fn value_at(&self, time_seconds: f64) -> f64;
}

impl<F> ScalarEventPredicate for F
where
    F: Fn(f64) -> f64,
{
    fn value_at(&self, time_seconds: f64) -> f64 {
        self(time_seconds)
    }
}

/// A boolean state function of time used to find discrete state changes.
pub trait DiscreteEventPredicate {
    /// Return the discrete state at `time_seconds`.
    fn state_at(&self, time_seconds: f64) -> bool;
}

impl<F> DiscreteEventPredicate for F
where
    F: Fn(f64) -> bool,
{
    fn state_at(&self, time_seconds: f64) -> bool {
        self(time_seconds)
    }
}

/// Error while configuring or evaluating an [`EventFinder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EventFinderError {
    /// Invalid event-finder input.
    #[error("invalid event-finder input {field}: {reason}")]
    InvalidInput {
        /// Field or input source that failed validation.
        field: &'static str,
        /// Stable reason string.
        reason: &'static str,
    },
}

/// Direction of a threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingDirection {
    /// Scalar value crossed from below the threshold to at or above it.
    Rising,
    /// Scalar value crossed from at or above the threshold to below it.
    Falling,
}

/// Kind of scalar extremum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremumKind {
    /// Local maximum.
    Maximum,
    /// Local minimum.
    Minimum,
}

/// A refined scalar threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingEvent {
    /// Refined event time.
    pub time_seconds: f64,
    /// Scalar value at the refined event time.
    pub value: f64,
    /// Threshold that was crossed.
    pub threshold: f64,
    /// Crossing direction.
    pub direction: CrossingDirection,
}

/// A refined local scalar extremum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtremumEvent {
    /// Refined event time.
    pub time_seconds: f64,
    /// Scalar value at the refined event time.
    pub value: f64,
    /// Whether this extremum is a maximum or minimum.
    pub kind: ExtremumKind,
}

/// A refined boolean state transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateChangeEvent {
    /// Refined transition time.
    pub time_seconds: f64,
    /// State before the transition bracket.
    pub previous_state: bool,
    /// State after the transition bracket.
    pub next_state: bool,
}

/// Uniform-scan event finder over a finite time window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventFinder {
    start_seconds: f64,
    end_seconds: f64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
}

impl EventFinder {
    /// Construct an event finder over `[start_seconds, end_seconds]`.
    pub fn new(
        start_seconds: f64,
        end_seconds: f64,
        step_seconds: f64,
        time_tolerance_seconds: f64,
    ) -> Result<Self, EventFinderError> {
        let start_seconds =
            validate::finite(start_seconds, "start_seconds").map_err(map_event_input)?;
        let end_seconds = validate::finite(end_seconds, "end_seconds").map_err(map_event_input)?;
        validate::range_order(start_seconds, end_seconds, "end_seconds")
            .map_err(map_event_input)?;
        let step_seconds =
            validate::positive_step(step_seconds, "step_seconds").map_err(map_event_input)?;
        let time_tolerance_seconds =
            validate::positive_step(time_tolerance_seconds, "time_tolerance_seconds")
                .map_err(map_event_input)?;

        Ok(Self {
            start_seconds,
            end_seconds,
            step_seconds,
            time_tolerance_seconds,
        })
    }

    /// Start of the search window.
    pub fn start_seconds(self) -> f64 {
        self.start_seconds
    }

    /// End of the search window.
    pub fn end_seconds(self) -> f64 {
        self.end_seconds
    }

    /// Uniform coarse-scan step.
    pub fn step_seconds(self) -> f64 {
        self.step_seconds
    }

    /// Event-time refinement tolerance.
    pub fn time_tolerance_seconds(self) -> f64 {
        self.time_tolerance_seconds
    }

    /// Find refined threshold crossings of a scalar predicate.
    pub fn find_crossings<P>(
        self,
        predicate: P,
        threshold: f64,
    ) -> Result<Vec<CrossingEvent>, EventFinderError>
    where
        P: ScalarEventPredicate,
    {
        self.find_crossings_ref(&predicate, threshold)
    }

    /// Find threshold crossings for a batch of scalar predicates, serially.
    ///
    /// Result element `i` belongs to `predicates[i]`.
    pub fn find_crossings_batch_serial<P>(
        self,
        predicates: &[P],
        threshold: f64,
    ) -> Vec<Result<Vec<CrossingEvent>, EventFinderError>>
    where
        P: ScalarEventPredicate,
    {
        predicates
            .iter()
            .map(|predicate| self.find_crossings_ref(predicate, threshold))
            .collect()
    }

    /// Find threshold crossings for a batch of scalar predicates in parallel.
    ///
    /// Each predicate is evaluated by the same kernel as
    /// [`EventFinder::find_crossings`]. The indexed collect preserves input
    /// order, so result element `i` belongs to `predicates[i]`.
    pub fn find_crossings_batch_parallel<P>(
        self,
        predicates: &[P],
        threshold: f64,
    ) -> Vec<Result<Vec<CrossingEvent>, EventFinderError>>
    where
        P: ScalarEventPredicate + Sync,
    {
        predicates
            .par_iter()
            .map(|predicate| self.find_crossings_ref(predicate, threshold))
            .collect()
    }

    fn find_crossings_ref<P>(
        self,
        predicate: &P,
        threshold: f64,
    ) -> Result<Vec<CrossingEvent>, EventFinderError>
    where
        P: ScalarEventPredicate + ?Sized,
    {
        let threshold = validate::finite(threshold, "threshold").map_err(map_event_input)?;
        let samples = self.scalar_samples(predicate)?;
        let mut events = Vec::new();

        for (left_index, pair) in samples.windows(2).enumerate() {
            let a = pair[0];
            let b = pair[1];
            let value_a = a.value - threshold;
            let value_b = b.value - threshold;
            let Some(direction) =
                crossing_direction_for_sample_pair(&samples, left_index, threshold)
            else {
                continue;
            };
            let time_seconds = if value_a == 0.0 {
                let zero_run_start = zero_run_start(&samples, left_index, threshold);
                samples[zero_run_start].time_seconds
            } else if value_b == 0.0 {
                b.time_seconds
            } else {
                try_bisect_crossing_until(
                    a.time_seconds,
                    b.time_seconds,
                    |time| finite_predicate_value(predicate.value_at(time) - threshold),
                    midpoint_seconds,
                    |lo, hi| (hi - lo).abs() <= self.time_tolerance_seconds,
                )
                .map_err(map_root_error)?
            };
            if events.last().is_some_and(|event: &CrossingEvent| {
                event.time_seconds.to_bits() == time_seconds.to_bits()
            }) {
                continue;
            }
            let value = finite_predicate_value(predicate.value_at(time_seconds))?;

            events.push(CrossingEvent {
                time_seconds,
                value,
                threshold,
                direction,
            });
        }

        Ok(events)
    }

    /// Find refined local maxima and minima of a scalar predicate.
    pub fn find_extrema<P>(self, predicate: P) -> Result<Vec<ExtremumEvent>, EventFinderError>
    where
        P: ScalarEventPredicate,
    {
        self.find_extrema_ref(&predicate)
    }

    /// Find local extrema for a batch of scalar predicates, serially.
    ///
    /// Result element `i` belongs to `predicates[i]`.
    pub fn find_extrema_batch_serial<P>(
        self,
        predicates: &[P],
    ) -> Vec<Result<Vec<ExtremumEvent>, EventFinderError>>
    where
        P: ScalarEventPredicate,
    {
        predicates
            .iter()
            .map(|predicate| self.find_extrema_ref(predicate))
            .collect()
    }

    /// Find local extrema for a batch of scalar predicates in parallel.
    ///
    /// Each predicate is evaluated by the same kernel as
    /// [`EventFinder::find_extrema`]. The indexed collect preserves input order,
    /// so result element `i` belongs to `predicates[i]`.
    pub fn find_extrema_batch_parallel<P>(
        self,
        predicates: &[P],
    ) -> Vec<Result<Vec<ExtremumEvent>, EventFinderError>>
    where
        P: ScalarEventPredicate + Sync,
    {
        predicates
            .par_iter()
            .map(|predicate| self.find_extrema_ref(predicate))
            .collect()
    }

    fn find_extrema_ref<P>(self, predicate: &P) -> Result<Vec<ExtremumEvent>, EventFinderError>
    where
        P: ScalarEventPredicate + ?Sized,
    {
        let samples = self.extrema_samples(predicate)?;
        let mut events = Vec::new();

        let mut index = 1;
        while index + 1 < samples.len() {
            let run_start = index;
            let run_value = samples[run_start].value;
            let mut run_end = run_start;
            while run_end + 1 < samples.len() && samples[run_end + 1].value == run_value {
                run_end += 1;
            }

            if run_end + 1 >= samples.len() {
                break;
            }

            let prev = samples[run_start - 1];
            let next = samples[run_end + 1];
            let kind = if run_value > prev.value && run_value > next.value {
                Some(ExtremumKind::Maximum)
            } else if run_value < prev.value && run_value < next.value {
                Some(ExtremumKind::Minimum)
            } else {
                None
            };

            if let Some(kind) = kind {
                events.push(self.refine_extremum(
                    predicate,
                    kind,
                    prev.time_seconds,
                    next.time_seconds,
                )?);
            }

            index = run_end + 1;
        }

        Ok(events)
    }

    /// Find refined changes of a boolean state predicate.
    pub fn find_state_changes<P>(
        self,
        predicate: P,
    ) -> Result<Vec<StateChangeEvent>, EventFinderError>
    where
        P: DiscreteEventPredicate,
    {
        self.find_state_changes_ref(&predicate)
    }

    /// Find state changes for a batch of discrete predicates, serially.
    ///
    /// Result element `i` belongs to `predicates[i]`.
    pub fn find_state_changes_batch_serial<P>(
        self,
        predicates: &[P],
    ) -> Vec<Result<Vec<StateChangeEvent>, EventFinderError>>
    where
        P: DiscreteEventPredicate,
    {
        predicates
            .iter()
            .map(|predicate| self.find_state_changes_ref(predicate))
            .collect()
    }

    /// Find state changes for a batch of discrete predicates in parallel.
    ///
    /// Each predicate is evaluated by the same kernel as
    /// [`EventFinder::find_state_changes`]. The indexed collect preserves input
    /// order, so result element `i` belongs to `predicates[i]`.
    pub fn find_state_changes_batch_parallel<P>(
        self,
        predicates: &[P],
    ) -> Vec<Result<Vec<StateChangeEvent>, EventFinderError>>
    where
        P: DiscreteEventPredicate + Sync,
    {
        predicates
            .par_iter()
            .map(|predicate| self.find_state_changes_ref(predicate))
            .collect()
    }

    fn find_state_changes_ref<P>(
        self,
        predicate: &P,
    ) -> Result<Vec<StateChangeEvent>, EventFinderError>
    where
        P: DiscreteEventPredicate + ?Sized,
    {
        let samples = self.state_samples(predicate)?;
        let mut events = Vec::new();

        for pair in samples.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if a.state == b.state {
                continue;
            }

            let time_seconds =
                self.refine_state_change(predicate, a.time_seconds, b.time_seconds, a.state);
            if events.last().is_some_and(|event: &StateChangeEvent| {
                event.time_seconds.to_bits() == time_seconds.to_bits()
            }) {
                continue;
            }
            events.push(StateChangeEvent {
                time_seconds,
                previous_state: a.state,
                next_state: b.state,
            });
        }

        Ok(events)
    }

    fn scalar_samples<P>(self, predicate: &P) -> Result<Vec<ScalarSample>, EventFinderError>
    where
        P: ScalarEventPredicate + ?Sized,
    {
        let duration_seconds = self.end_seconds - self.start_seconds;
        let sample_iterations = self.coarse_sample_iterations()?;
        let mut samples = Vec::with_capacity(sample_iterations.saturating_add(1));
        let mut offset_seconds = 0.0;

        for _ in 0..sample_iterations {
            if offset_seconds >= duration_seconds {
                break;
            }
            let time_seconds = self.start_seconds + offset_seconds;
            if time_seconds >= self.end_seconds {
                break;
            }
            samples.push(ScalarSample {
                time_seconds,
                value: finite_predicate_value(predicate.value_at(time_seconds))?,
            });
            let next_offset_seconds = offset_seconds + self.step_seconds;
            if next_offset_seconds <= offset_seconds {
                return Err(non_advancing_sample_step_error());
            }
            offset_seconds = next_offset_seconds;
        }
        if offset_seconds < duration_seconds
            && self.start_seconds + offset_seconds < self.end_seconds
        {
            return Err(too_many_event_samples_error());
        }

        samples.push(ScalarSample {
            time_seconds: self.end_seconds,
            value: finite_predicate_value(predicate.value_at(self.end_seconds))?,
        });
        Ok(samples)
    }

    fn extrema_samples<P>(self, predicate: &P) -> Result<Vec<ScalarSample>, EventFinderError>
    where
        P: ScalarEventPredicate + ?Sized,
    {
        let mut samples = self.scalar_samples(predicate)?;
        if samples.len() == 2 {
            let midpoint = midpoint_seconds(samples[0].time_seconds, samples[1].time_seconds);
            if midpoint != samples[0].time_seconds && midpoint != samples[1].time_seconds {
                samples.insert(
                    1,
                    ScalarSample {
                        time_seconds: midpoint,
                        value: finite_predicate_value(predicate.value_at(midpoint))?,
                    },
                );
            }
        }
        Ok(samples)
    }

    fn state_samples<P>(self, predicate: &P) -> Result<Vec<StateSample>, EventFinderError>
    where
        P: DiscreteEventPredicate + ?Sized,
    {
        let duration_seconds = self.end_seconds - self.start_seconds;
        let sample_iterations = self.coarse_sample_iterations()?;
        let mut samples = Vec::with_capacity(sample_iterations.saturating_add(1));
        let mut offset_seconds = 0.0;

        for _ in 0..sample_iterations {
            if offset_seconds >= duration_seconds {
                break;
            }
            let time_seconds = self.start_seconds + offset_seconds;
            if time_seconds >= self.end_seconds {
                break;
            }
            samples.push(StateSample {
                time_seconds,
                state: predicate.state_at(time_seconds),
            });
            let next_offset_seconds = offset_seconds + self.step_seconds;
            if next_offset_seconds <= offset_seconds {
                return Err(non_advancing_sample_step_error());
            }
            offset_seconds = next_offset_seconds;
        }
        if offset_seconds < duration_seconds
            && self.start_seconds + offset_seconds < self.end_seconds
        {
            return Err(too_many_event_samples_error());
        }

        samples.push(StateSample {
            time_seconds: self.end_seconds,
            state: predicate.state_at(self.end_seconds),
        });
        Ok(samples)
    }

    fn coarse_sample_iterations(self) -> Result<usize, EventFinderError> {
        let duration_seconds = self.end_seconds - self.start_seconds;
        if duration_seconds <= 0.0 {
            return Ok(0);
        }
        if !duration_seconds.is_finite() {
            return Err(too_many_event_samples_error());
        }

        let coarse_samples = (duration_seconds / self.step_seconds).ceil();
        if !(coarse_samples.is_finite() && coarse_samples >= 1.0) {
            return Err(too_many_event_samples_error());
        }
        if coarse_samples > MAX_EVENT_COARSE_SAMPLES as f64 {
            return Err(too_many_event_samples_error());
        }

        Ok((coarse_samples as usize).saturating_add(1))
    }

    fn refine_extremum<P>(
        self,
        predicate: &P,
        kind: ExtremumKind,
        low: f64,
        high: f64,
    ) -> Result<ExtremumEvent, EventFinderError>
    where
        P: ScalarEventPredicate + ?Sized,
    {
        let mut lo = low;
        let mut hi = high;

        for _ in 0..MAX_GOLDEN_ITERATIONS {
            if (hi - lo).abs() <= self.time_tolerance_seconds {
                break;
            }
            let span = hi - lo;
            let left = lo + GOLDEN_RESPHI * span;
            let right = hi - GOLDEN_RESPHI * span;
            if !(left > lo && right < hi) {
                break;
            }

            let score_left =
                extremum_score(kind, finite_predicate_value(predicate.value_at(left))?);
            let score_right =
                extremum_score(kind, finite_predicate_value(predicate.value_at(right))?);

            let score_delta = (score_left - score_right).abs();
            let score_scale = score_left.abs().max(score_right.abs()).max(1.0);
            if score_delta <= 16.0 * f64::EPSILON * score_scale {
                lo = left;
                hi = right;
            } else if score_left > score_right {
                hi = right;
            } else {
                lo = left;
            }
        }

        let time_seconds = midpoint_seconds(lo, hi);
        let value = finite_predicate_value(predicate.value_at(time_seconds))?;
        Ok(ExtremumEvent {
            time_seconds,
            value,
            kind,
        })
    }

    fn refine_state_change<P>(self, predicate: &P, low: f64, high: f64, low_state: bool) -> f64
    where
        P: DiscreteEventPredicate + ?Sized,
    {
        let mut lo = low;
        let mut hi = high;

        while (hi - lo).abs() > self.time_tolerance_seconds {
            let mid = midpoint_seconds(lo, hi);
            if mid == lo || mid == hi {
                return mid;
            }
            let mid_state = predicate.state_at(mid);
            if exact_state_transition_midpoint(predicate, lo, hi, mid, low_state, mid_state) {
                return mid;
            }
            if mid_state == low_state {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        midpoint_seconds(lo, hi)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalarSample {
    time_seconds: f64,
    value: f64,
}

#[derive(Debug, Clone, Copy)]
struct StateSample {
    time_seconds: f64,
    state: bool,
}

fn midpoint_seconds(a: f64, b: f64) -> f64 {
    (a + b) * 0.5
}

fn map_root_error(error: RootError<EventFinderError>) -> EventFinderError {
    match error {
        RootError::InvalidInput { field, reason } => {
            EventFinderError::InvalidInput { field, reason }
        }
        RootError::Predicate(error) => error,
    }
}

fn crossing_direction_for_sample_pair(
    samples: &[ScalarSample],
    left_index: usize,
    threshold: f64,
) -> Option<CrossingDirection> {
    let value_a = samples[left_index].value - threshold;
    let value_b = samples[left_index + 1].value - threshold;

    if value_a == 0.0 || value_b == 0.0 {
        return exact_sample_crossing_direction(samples, left_index, threshold, value_a, value_b);
    }
    if !sign_change_bracketed(value_a, value_b).unwrap_or(false) {
        return None;
    }
    Some(crossing_direction_from_sides(value_a, value_b))
}

fn exact_sample_crossing_direction(
    samples: &[ScalarSample],
    left_index: usize,
    threshold: f64,
    value_a: f64,
    value_b: f64,
) -> Option<CrossingDirection> {
    if value_a == 0.0 && value_b == 0.0 {
        return None;
    }

    if value_b == 0.0 {
        let right_value = first_nonzero_value_from(samples, left_index + 2, threshold);
        return match right_value {
            Some(right) => crossing_direction_from_opposite_sides(value_a, right),
            None => Some(crossing_direction_from_sides(value_a, 0.0)),
        };
    }

    let zero_run_start = zero_run_start(samples, left_index, threshold);
    match last_nonzero_value_before(samples, zero_run_start, threshold) {
        Some(_) => None,
        None => Some(crossing_direction_from_sides(0.0, value_b)),
    }
}

fn zero_run_start(samples: &[ScalarSample], zero_index: usize, threshold: f64) -> usize {
    let mut index = zero_index;
    while index > 0 && samples[index - 1].value - threshold == 0.0 {
        index -= 1;
    }
    index
}

fn last_nonzero_value_before(
    samples: &[ScalarSample],
    end_index: usize,
    threshold: f64,
) -> Option<f64> {
    samples[..end_index]
        .iter()
        .rev()
        .map(|sample| sample.value - threshold)
        .find(|value| *value != 0.0)
}

fn first_nonzero_value_from(
    samples: &[ScalarSample],
    start_index: usize,
    threshold: f64,
) -> Option<f64> {
    samples
        .iter()
        .skip(start_index)
        .map(|sample| sample.value - threshold)
        .find(|value| *value != 0.0)
}

fn crossing_direction_from_opposite_sides(left: f64, right: f64) -> Option<CrossingDirection> {
    if left < 0.0 && right > 0.0 {
        Some(CrossingDirection::Rising)
    } else if left > 0.0 && right < 0.0 {
        Some(CrossingDirection::Falling)
    } else {
        None
    }
}

fn crossing_direction_from_sides(left: f64, right: f64) -> CrossingDirection {
    if left < 0.0 || (left == 0.0 && right > 0.0) {
        CrossingDirection::Rising
    } else {
        CrossingDirection::Falling
    }
}

fn exact_state_transition_midpoint<P>(
    predicate: &P,
    lo: f64,
    hi: f64,
    mid: f64,
    low_state: bool,
    mid_state: bool,
) -> bool
where
    P: DiscreteEventPredicate + ?Sized,
{
    if mid_state == low_state {
        predicate.state_at(adjacent_float_toward(mid, hi)) != low_state
    } else {
        predicate.state_at(adjacent_float_toward(mid, lo)) == low_state
    }
}

fn adjacent_float_toward(value: f64, target: f64) -> f64 {
    if target > value {
        next_float_up(value)
    } else {
        next_float_down(value)
    }
}

fn next_float_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        return value;
    }
    let bits = value.to_bits();
    if bits == 0x8000_0000_0000_0000 {
        f64::from_bits(1)
    } else if value >= 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn next_float_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        return value;
    }
    let bits = value.to_bits();
    if bits == 0 {
        f64::from_bits(0x8000_0000_0000_0001)
    } else if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn extremum_score(kind: ExtremumKind, value: f64) -> f64 {
    match kind {
        ExtremumKind::Maximum => value,
        ExtremumKind::Minimum => -value,
    }
}

fn finite_predicate_value(value: f64) -> Result<f64, EventFinderError> {
    validate::finite(value, "predicate").map_err(map_event_input)
}

fn too_many_event_samples_error() -> EventFinderError {
    EventFinderError::InvalidInput {
        field: "step_seconds",
        reason: "too many samples",
    }
}

fn non_advancing_sample_step_error() -> EventFinderError {
    EventFinderError::InvalidInput {
        field: "step_seconds",
        reason: "does not advance samples",
    }
}

fn map_event_input(error: validate::FieldError) -> EventFinderError {
    EventFinderError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    #[derive(Debug, Clone, Copy)]
    struct ShiftedSine {
        phase_seconds: f64,
    }

    impl ScalarEventPredicate for ShiftedSine {
        fn value_at(&self, time_seconds: f64) -> f64 {
            (time_seconds + self.phase_seconds).sin()
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StepState {
        transition_seconds: f64,
    }

    impl DiscreteEventPredicate for StepState {
        fn state_at(&self, time_seconds: f64) -> bool {
            time_seconds >= self.transition_seconds
        }
    }

    fn finder(start: f64, end: f64) -> EventFinder {
        EventFinder::new(start, end, 0.2, 1.0e-12).expect("valid finder")
    }

    #[test]
    fn scalar_samples_step_from_relative_offset_after_nonzero_start() {
        let start = 1_000_000_000.0;
        let step = 0.1;
        let end = start + 0.5;
        let samples = EventFinder::new(start, end, step, 1.0e-12)
            .expect("valid finder")
            .scalar_samples(&|time| time)
            .expect("finite samples");
        let expected_times = [
            start,
            start + step,
            start + 2.0 * step,
            start + 3.0 * step,
            start + 4.0 * step,
            end,
        ];

        assert_eq!(samples.len(), expected_times.len());
        for (index, (sample, expected_time)) in samples.iter().zip(expected_times).enumerate() {
            assert_eq!(
                sample.time_seconds.to_bits(),
                expected_time.to_bits(),
                "sample {index} time"
            );
            assert_eq!(
                sample.value.to_bits(),
                expected_time.to_bits(),
                "sample {index} value"
            );
        }
    }

    #[test]
    fn scalar_samples_preserve_repeated_addition_near_endpoint() {
        let samples = EventFinder::new(0.0, 1.0, 0.1, 1.0e-12)
            .expect("valid finder")
            .scalar_samples(&|time| time)
            .expect("finite samples");

        assert_eq!(samples.len(), 12);
        assert_eq!(
            samples[samples.len() - 2].time_seconds.to_bits(),
            0.999_999_999_999_999_9_f64.to_bits()
        );
        assert_eq!(
            samples
                .last()
                .expect("endpoint sample")
                .time_seconds
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn scalar_samples_reject_infeasible_grid_without_sampling() {
        let finder = EventFinder::new(0.0, 1.0, f64::MIN_POSITIVE, 1.0e-12).expect("valid finder");

        assert_invalid_field(
            finder.find_crossings(|_| 1.0, 0.0).unwrap_err(),
            "step_seconds",
            "too many samples",
        );
        assert_invalid_field(
            finder.find_extrema(|time| time).unwrap_err(),
            "step_seconds",
            "too many samples",
        );
    }

    #[test]
    fn state_changes_reject_infeasible_grid_without_sampling() {
        let finder = EventFinder::new(0.0, 1.0, f64::MIN_POSITIVE, 1.0e-12).expect("valid finder");
        let state_calls = Cell::new(0);

        assert_invalid_field(
            finder
                .find_state_changes(|time| {
                    state_calls.set(state_calls.get() + 1);
                    time >= 0.5
                })
                .unwrap_err(),
            "step_seconds",
            "too many samples",
        );
        assert_eq!(state_calls.get(), 0);

        let predicates: [fn(f64) -> bool; 3] =
            [|time| time >= 0.25, |time| time >= 0.5, |time| time >= 0.75];
        let serial = finder.find_state_changes_batch_serial(&predicates);
        let parallel = finder.find_state_changes_batch_parallel(&predicates);
        assert_eq!(serial, parallel);
        assert!(serial.iter().all(|result| {
            matches!(
                result,
                Err(EventFinderError::InvalidInput {
                    field: "step_seconds",
                    reason: "too many samples"
                })
            )
        }));
    }

    #[test]
    fn crossings_find_sine_zeroes_with_direction() {
        let events = finder(-0.4, TAU + 0.4)
            .find_crossings(f64::sin, 0.0)
            .expect("finite sine samples");

        assert_eq!(events.len(), 3);
        assert_close(events[0].time_seconds, 0.0, 1.0e-12);
        assert_eq!(events[0].direction, CrossingDirection::Rising);
        assert_close(events[0].value, 0.0, 1.0e-12);

        assert_close(events[1].time_seconds, PI, 1.0e-12);
        assert_eq!(events[1].direction, CrossingDirection::Falling);
        assert_close(events[1].value, 0.0, 1.0e-12);

        assert_close(events[2].time_seconds, TAU, 1.0e-12);
        assert_eq!(events[2].direction, CrossingDirection::Rising);
        assert_close(events[2].value, 0.0, 1.0e-12);
    }

    #[test]
    fn crossings_suppress_tangential_threshold_touch() {
        let tangent_from_above_events = EventFinder::new(0.0, 2.0, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time: f64| (time - 1.0) * (time - 1.0), 0.0)
            .expect("finite tangent samples");

        assert!(tangent_from_above_events.is_empty());

        let tangent_from_below_events = EventFinder::new(0.0, 2.0, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time: f64| -(time - 1.0) * (time - 1.0), 0.0)
            .expect("finite tangent samples");

        assert!(tangent_from_below_events.is_empty());

        let crossing_events = EventFinder::new(0.0, 2.0, 0.25, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time: f64| 0.25 - (time - 1.0) * (time - 1.0), 0.0)
            .expect("finite crossing samples");

        assert_eq!(crossing_events.len(), 2);
        assert_eq!(crossing_events[0].direction, CrossingDirection::Rising);
        assert_eq!(crossing_events[0].time_seconds.to_bits(), 0.5_f64.to_bits());
        assert_eq!(crossing_events[1].direction, CrossingDirection::Falling);
        assert_eq!(crossing_events[1].time_seconds.to_bits(), 1.5_f64.to_bits());
    }

    #[test]
    fn crossings_detect_opposite_side_threshold_plateaus() {
        let rising_events = plateau_crossings([-1.0, 0.0, 0.0, 1.0]);
        assert_eq!(rising_events.len(), 1);
        assert_eq!(rising_events[0].direction, CrossingDirection::Rising);
        assert_eq!(rising_events[0].time_seconds.to_bits(), 1.0_f64.to_bits());

        let falling_events = plateau_crossings([1.0, 0.0, 0.0, -1.0]);
        assert_eq!(falling_events.len(), 1);
        assert_eq!(falling_events[0].direction, CrossingDirection::Falling);
        assert_eq!(falling_events[0].time_seconds.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn crossings_emit_boundary_threshold_plateaus_at_start() {
        let rising_events = plateau_crossings([0.0, 0.0, 1.0]);
        assert_eq!(rising_events.len(), 1);
        assert_eq!(rising_events[0].direction, CrossingDirection::Rising);
        assert_eq!(rising_events[0].time_seconds.to_bits(), 0.0_f64.to_bits());

        let falling_events = plateau_crossings([0.0, 0.0, -1.0]);
        assert_eq!(falling_events.len(), 1);
        assert_eq!(falling_events[0].direction, CrossingDirection::Falling);
        assert_eq!(falling_events[0].time_seconds.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn crossings_suppress_same_side_threshold_plateaus() {
        assert!(plateau_crossings([-1.0, 0.0, 0.0, -1.0]).is_empty());
        assert!(plateau_crossings([1.0, 0.0, 0.0, 1.0]).is_empty());
    }

    fn plateau_crossings<const N: usize>(values: [f64; N]) -> Vec<CrossingEvent> {
        EventFinder::new(0.0, (N - 1) as f64, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_crossings(
                move |time: f64| {
                    let index = time.round() as usize;
                    assert!(index < N);
                    assert_eq!(time.to_bits(), (index as f64).to_bits());
                    values[index]
                },
                0.0,
            )
            .expect("finite plateau samples")
    }

    #[test]
    fn crossings_detect_exact_right_endpoint_once() {
        let final_endpoint_events = EventFinder::new(0.0, 1.0, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time| 1.0 - time, 0.0)
            .expect("finite endpoint samples");

        assert_eq!(final_endpoint_events.len(), 1);
        assert_eq!(
            final_endpoint_events[0].time_seconds.to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            final_endpoint_events[0].direction,
            CrossingDirection::Falling
        );

        let shared_endpoint_events = EventFinder::new(0.0, 2.0, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time| 1.0 - time, 0.0)
            .expect("finite endpoint samples");

        assert_eq!(shared_endpoint_events.len(), 1);
        assert_eq!(
            shared_endpoint_events[0].time_seconds.to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            shared_endpoint_events[0].direction,
            CrossingDirection::Falling
        );

        let interior_events = EventFinder::new(0.0, 1.0, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time| 0.5 - time, 0.0)
            .expect("finite interior samples");

        assert_eq!(interior_events.len(), 1);
        assert_close(interior_events[0].time_seconds, 0.5, 1.0e-12);
        assert_eq!(interior_events[0].direction, CrossingDirection::Falling);
    }

    #[test]
    fn crossings_detect_exact_start_endpoint_once() {
        let start = 12.0;
        let start_endpoint_events = EventFinder::new(start, start + 1.0, 0.25, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time| time - start, 0.0)
            .expect("finite endpoint samples");

        assert_eq!(start_endpoint_events.len(), 1);
        assert_eq!(
            start_endpoint_events[0].time_seconds.to_bits(),
            start.to_bits()
        );
        assert_eq!(
            start_endpoint_events[0].direction,
            CrossingDirection::Rising
        );

        let interior_events = EventFinder::new(start, start + 1.0, 0.5, 1.0e-12)
            .expect("valid finder")
            .find_crossings(|time| time - (start + 0.5), 0.0)
            .expect("finite endpoint samples");

        assert_eq!(interior_events.len(), 1);
        assert_eq!(
            interior_events[0].time_seconds.to_bits(),
            (start + 0.5_f64).to_bits()
        );
        assert_eq!(interior_events[0].direction, CrossingDirection::Rising);
    }

    #[test]
    fn extrema_find_sine_maximum_and_minimum() {
        let events = EventFinder::new(0.0, TAU, 0.2, 1.0e-8)
            .expect("valid finder")
            .find_extrema(f64::sin)
            .expect("finite sine samples");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, ExtremumKind::Maximum);
        assert_close(events[0].time_seconds, FRAC_PI_2, 5.0e-8);
        assert_close(events[0].value, 1.0, 1.0e-12);

        assert_eq!(events[1].kind, ExtremumKind::Minimum);
        assert_close(events[1].time_seconds, 3.0 * FRAC_PI_2, 5.0e-8);
        assert_close(events[1].value, -1.0, 1.0e-12);
    }

    #[test]
    fn extrema_detect_short_window_inside_single_coarse_step() {
        let events = EventFinder::new(0.0, 1.0, 10.0, 1.0e-12)
            .expect("valid finder")
            .find_extrema(|time: f64| -(time - 0.3) * (time - 0.3))
            .expect("finite parabola samples");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, ExtremumKind::Maximum);
        assert_close(events[0].time_seconds, 0.3, 1.0e-8);
        assert_close(events[0].value, 0.0, 1.0e-12);
    }

    #[test]
    fn extrema_deduplicate_flat_minimum_and_maximum() {
        let minima = sampled_extrema([2.0, 1.0, 1.0, 2.0]);
        assert_eq!(minima.len(), 1);
        assert_eq!(minima[0].kind, ExtremumKind::Minimum);
        assert!((1.0..=2.0).contains(&minima[0].time_seconds));
        assert_close(minima[0].value, 1.0, 1.0e-12);

        let maxima = sampled_extrema([1.0, 2.0, 2.0, 1.0]);
        assert_eq!(maxima.len(), 1);
        assert_eq!(maxima[0].kind, ExtremumKind::Maximum);
        assert!((1.0..=2.0).contains(&maxima[0].time_seconds));
        assert_close(maxima[0].value, 2.0, 1.0e-12);
    }

    #[test]
    fn extrema_keep_distinct_adjacent_minima() {
        let events = sampled_extrema([2.0, 1.0, 2.0, 1.0, 2.0]);
        let minima: Vec<_> = events
            .iter()
            .filter(|event| event.kind == ExtremumKind::Minimum)
            .collect();

        assert_eq!(minima.len(), 2);
        assert_close(minima[0].time_seconds, 1.0, 1.0e-8);
        assert_close(minima[1].time_seconds, 3.0, 1.0e-8);
    }

    #[test]
    fn state_changes_find_step_transition() {
        let events = EventFinder::new(0.0, 5.0, 1.0, 1.0e-9)
            .expect("valid finder")
            .find_state_changes(|time| time >= 2.5)
            .expect("state changes");

        assert_eq!(events.len(), 1);
        assert_close(events[0].time_seconds, 2.5, 1.0e-9);
        assert!(!events[0].previous_state);
        assert!(events[0].next_state);
    }

    #[test]
    fn state_change_refinement_returns_exact_midpoint_transition() {
        let events = EventFinder::new(0.0, 2.0, 2.0, 1.0e-12)
            .expect("valid finder")
            .find_state_changes(|time| time >= 1.0)
            .expect("state changes");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].time_seconds.to_bits(), 1.0_f64.to_bits());
        assert!(!events[0].previous_state);
        assert!(events[0].next_state);
    }

    #[test]
    fn state_changes_keep_sampling_inside_window() {
        let start = 12.0;
        let end = start + 1.0;
        let min_seen = Cell::new(f64::INFINITY);
        let max_seen = Cell::new(f64::NEG_INFINITY);
        let transition_seconds = start + 0.65;

        let events = EventFinder::new(start, end, 0.4, 1.0e-12)
            .expect("valid finder")
            .find_state_changes(|time| {
                min_seen.set(min_seen.get().min(time));
                max_seen.set(max_seen.get().max(time));
                time >= transition_seconds
            })
            .expect("state changes");

        assert_eq!(events.len(), 1);
        assert!((start..=end).contains(&events[0].time_seconds));
        assert_close(events[0].time_seconds, transition_seconds, 1.0e-12);
        assert!(!events[0].previous_state);
        assert!(events[0].next_state);
        assert!(min_seen.get() >= start);
        assert!(max_seen.get() <= end);
    }

    #[test]
    fn state_change_refinement_stops_when_midpoint_cannot_shrink_bracket() {
        let high = 1.0_f64;
        let low = f64::from_bits(high.to_bits() - 1);
        let finder = EventFinder::new(low, high, high - low, f64::MIN_POSITIVE)
            .expect("valid adjacent-float finder");
        let state_calls = Cell::new(0);

        let transition = finder.refine_state_change(
            &|time| {
                state_calls.set(state_calls.get() + 1);
                time >= high
            },
            low,
            high,
            false,
        );

        assert_eq!(transition.to_bits(), high.to_bits());
        assert_eq!(state_calls.get(), 0);
    }

    #[test]
    fn batch_parallel_matches_serial_in_input_order() {
        let wave_finder =
            EventFinder::new(-0.8, TAU + 0.8, 0.2, 1.0e-10).expect("valid wave finder");
        let waves = [
            ShiftedSine { phase_seconds: 0.0 },
            ShiftedSine {
                phase_seconds: 0.35,
            },
            ShiftedSine {
                phase_seconds: -0.45,
            },
            ShiftedSine { phase_seconds: 0.7 },
        ];

        let crossing_serial = wave_finder.find_crossings_batch_serial(&waves, 0.0);
        let crossing_parallel = wave_finder.find_crossings_batch_parallel(&waves, 0.0);
        assert_eq!(crossing_serial, crossing_parallel);
        assert_eq!(crossing_serial.len(), waves.len());
        assert!(crossing_serial
            .iter()
            .all(|events| events.as_ref().is_ok_and(|events| !events.is_empty())));

        let extrema_serial = wave_finder.find_extrema_batch_serial(&waves);
        let extrema_parallel = wave_finder.find_extrema_batch_parallel(&waves);
        assert_eq!(extrema_serial, extrema_parallel);
        assert_eq!(extrema_serial.len(), waves.len());
        assert!(extrema_serial
            .iter()
            .all(|events| events.as_ref().is_ok_and(|events| events.len() >= 2)));

        let state_finder = EventFinder::new(0.0, 5.0, 0.25, 1.0e-10).expect("valid state finder");
        let states = [
            StepState {
                transition_seconds: 0.6,
            },
            StepState {
                transition_seconds: 1.9,
            },
            StepState {
                transition_seconds: 3.4,
            },
            StepState {
                transition_seconds: 4.75,
            },
        ];
        let state_serial = state_finder.find_state_changes_batch_serial(&states);
        let state_parallel = state_finder.find_state_changes_batch_parallel(&states);
        assert_eq!(state_serial, state_parallel);
        assert_eq!(state_serial.len(), states.len());
        for (result, predicate) in state_serial.iter().zip(states.iter()) {
            let events = result.as_ref().expect("state changes");
            assert_eq!(events.len(), 1);
            assert_close(
                events[0].time_seconds,
                predicate.transition_seconds,
                1.0e-10,
            );
        }
    }

    #[test]
    fn invalid_window_and_steps_are_rejected() {
        assert_invalid_field(
            EventFinder::new(1.0, 0.0, 1.0, 1.0).unwrap_err(),
            "end_seconds",
            "out of range",
        );
        assert_invalid_field(
            EventFinder::new(0.0, 1.0, 0.0, 1.0).unwrap_err(),
            "step_seconds",
            "not positive",
        );
        assert_invalid_field(
            EventFinder::new(0.0, 1.0, 1.0, 0.0).unwrap_err(),
            "time_tolerance_seconds",
            "not positive",
        );
    }

    #[test]
    fn non_finite_scalar_inputs_are_rejected() {
        let finder = EventFinder::new(0.0, 1.0, 0.5, 1.0e-9).expect("valid finder");
        assert_invalid_field(
            finder.find_crossings(|time| time, f64::NAN).unwrap_err(),
            "threshold",
            "not finite",
        );
        assert_invalid_field(
            finder
                .find_extrema(|time| if time < 0.5 { time } else { f64::NAN })
                .unwrap_err(),
            "predicate",
            "not finite",
        );
    }

    #[test]
    fn crossings_reject_non_finite_midpoint_values() {
        let finder = EventFinder::new(0.0, 2.0, 2.0, 0.25).expect("valid finder");
        assert_invalid_field(
            finder
                .find_crossings(
                    |time| {
                        if time == 1.0 {
                            f64::NAN
                        } else {
                            time - 1.0
                        }
                    },
                    0.0,
                )
                .unwrap_err(),
            "predicate",
            "not finite",
        );

        let crossing = finder
            .find_crossings(|time| time - 1.0, 0.0)
            .expect("finite midpoint predicate should resolve normally");
        assert_eq!(crossing.len(), 1);
        assert_close(crossing[0].time_seconds, 1.0, 0.25);
    }

    fn assert_invalid_field(
        error: EventFinderError,
        expected_field: &'static str,
        expected_reason: &'static str,
    ) {
        let EventFinderError::InvalidInput { field, reason } = error;
        assert_eq!(field, expected_field);
        assert_eq!(reason, expected_reason);
    }

    fn sampled_extrema<const N: usize>(values: [f64; N]) -> Vec<ExtremumEvent> {
        assert!(N >= 2);
        EventFinder::new(0.0, (N - 1) as f64, 1.0, 1.0e-12)
            .expect("valid finder")
            .find_extrema(move |time| piecewise_linear_sample(&values, time))
            .expect("finite sample extrema")
    }

    fn piecewise_linear_sample<const N: usize>(values: &[f64; N], time: f64) -> f64 {
        if time <= 0.0 {
            return values[0];
        }

        let last_index = N - 1;
        let last_time = last_index as f64;
        if time >= last_time {
            return values[last_index];
        }

        let left_index = time.floor() as usize;
        let fraction = time - left_index as f64;
        values[left_index] + (values[left_index + 1] - values[left_index]) * fraction
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }
}
