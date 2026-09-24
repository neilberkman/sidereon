//! The exact epoch axis of an SP3 product and the grid its epochs lie on.
//!
//! An SP3 epoch record states its seconds to eight decimals, so every epoch a
//! file can hold is a whole number of 10-nanosecond ticks from the J2000
//! origin. Epochs are compared on that axis, never on floating seconds: two
//! epochs are one instant exactly when their ticks are equal, and a step
//! between epochs is an exact integer.
//!
//! [`product_grid`] is the one rule for which grid a product's epochs lie on;
//! [`Sp3::satellite_coverage`] reports it.

use super::write::epoch_tick;
use super::Sp3;

/// Ticks per second of the SP3 epoch axis (the `F11.8` seconds field).
pub(super) const TICKS_PER_SECOND: i128 = 100_000_000;

/// The seconds a whole number of ticks spans, for a step or an interval. The
/// count is below 2^53 for any interval a header field can state, so the
/// conversion to `f64` is exact and the division rounds once, giving the value
/// the eight-decimal field reads back as.
pub(super) fn interval_seconds(ticks: i128) -> f64 {
    ticks as f64 / TICKS_PER_SECOND as f64
}

/// The whole number of ticks an interval states, when it is positive and is
/// the `f64` nearest a whole number of ticks - the value an eight-decimal field
/// reads back as. `None` otherwise.
pub(super) fn interval_ticks(interval_s: f64) -> Option<i128> {
    if !interval_s.is_finite() || interval_s <= 0.0 {
        return None;
    }
    let scaled = (interval_s * TICKS_PER_SECOND as f64).round();
    if !(1.0..9_007_199_254_740_992.0).contains(&scaled) {
        return None;
    }
    let ticks = scaled as i128;
    (interval_seconds(ticks) == interval_s).then_some(ticks)
}

/// Each epoch of `sp3` on the tick axis, in file order; `None` for an epoch no
/// SP3 record states exactly.
pub(super) fn product_ticks(sp3: &Sp3) -> Vec<Option<i128>> {
    sp3.epochs
        .iter()
        .map(|epoch| epoch_tick(epoch, sp3.header.time_system))
        .collect()
}

/// The grid a product's epochs lie on, as [`Sp3::satellite_coverage`] reports
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3EpochGrid {
    /// The grid step, seconds. When every step between consecutive epochs is
    /// the same, it is that step. When the steps differ, it is the header's
    /// declared interval if every step is a whole multiple of it - the product
    /// skips epochs of its declared grid - and `None` otherwise. With fewer than
    /// two placed epochs it is the header interval, if that states a positive
    /// whole number of ticks. Also `None` when epochs are out of order.
    pub interval_s: Option<f64>,
    /// Whether the header's declared interval is the grid step.
    pub agrees_with_header: bool,
    /// Indices into [`Sp3::epochs`] of epochs that do not follow the placed
    /// epoch before them in time (equal or earlier), in file order.
    pub out_of_order: Vec<usize>,
    /// Indices into [`Sp3::epochs`] of epochs no SP3 record states exactly: an
    /// instant that is not a whole number of 10-nanosecond ticks.
    pub unplaced: Vec<usize>,
}

/// The grid facts [`product_grid`] finds, on the tick axis.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct GridFacts {
    /// Grid step in ticks; see [`Sp3EpochGrid::interval_s`].
    pub(super) step: Option<i128>,
    /// The header interval in ticks, when it states a positive whole number.
    pub(super) header: Option<i128>,
    pub(super) out_of_order: Vec<usize>,
    pub(super) unplaced: Vec<usize>,
}

impl GridFacts {
    pub(super) fn public(&self) -> Sp3EpochGrid {
        Sp3EpochGrid {
            interval_s: self.step.map(interval_seconds),
            agrees_with_header: self.step.is_some() && self.step == self.header,
            out_of_order: self.out_of_order.clone(),
            unplaced: self.unplaced.clone(),
        }
    }
}

/// The grid epochs at `ticks` (file order) lie on, given the header's declared
/// interval.
///
/// Equal steps are a uniform grid of that step whatever the header says, so a
/// wrong or zero header interval does not misplace a uniform product. Unequal
/// steps are a grid only when every step is a whole multiple of the header's
/// declared interval: the product then skips epochs of that grid. A step that
/// is not - one shorter than the declared interval, or one no declared
/// interval divides - leaves the product on no grid. Epochs out of order leave
/// it on no grid too.
pub(super) fn product_grid(ticks: &[Option<i128>], header_interval_s: f64) -> GridFacts {
    let header = interval_ticks(header_interval_s);
    let mut unplaced = Vec::new();
    let mut out_of_order = Vec::new();
    let mut steps: Vec<i128> = Vec::new();
    let mut previous: Option<i128> = None;
    for (index, tick) in ticks.iter().enumerate() {
        let Some(tick) = *tick else {
            unplaced.push(index);
            continue;
        };
        if let Some(before) = previous {
            if tick <= before {
                out_of_order.push(index);
                continue;
            }
            steps.push(tick - before);
        }
        previous = Some(tick);
    }

    let step = if !out_of_order.is_empty() {
        None
    } else if steps.is_empty() {
        header
    } else if steps.iter().all(|&step| step == steps[0]) {
        Some(steps[0])
    } else {
        header.filter(|&declared| steps.iter().all(|&step| step % declared == 0))
    };
    GridFacts {
        step,
        header,
        out_of_order,
        unplaced,
    }
}
