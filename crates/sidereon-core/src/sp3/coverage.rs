//! Per-satellite position and clock coverage of an SP3 product.
//!
//! The header satellite list and the epoch list describe the product as a
//! whole; neither says which satellite carries what at which epoch. A merged
//! product makes the difference plain: its epoch grid is the union of its
//! inputs, while each satellite holds only the cells the merge accepted, and a
//! clock can be absent where a position is present. [`Sp3::satellite_coverage`]
//! states, for each satellite, where its positions and its clocks are and
//! where they are not.
//!
//! Coverage is a statement about records, not about interpolation. A satellite
//! whose positions cover a span can still be refused an interpolation near a
//! span edge or inside a gap, and a continuity check with no findings says the
//! records present agree with one another, not that every epoch is covered.

use super::grid::{product_grid, product_ticks, Sp3EpochGrid};
use super::Sp3;
use crate::astro::time::model::Instant;
use crate::id::GnssSatelliteId;

/// A run of product epochs at which a satellite carries a channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3CoverageSpan {
    /// Index into [`Sp3::epochs`] of the first epoch of the run.
    pub first_index: usize,
    /// Index into [`Sp3::epochs`] of the last epoch of the run.
    pub last_index: usize,
    /// The first epoch of the run.
    pub first_epoch: Instant,
    /// The last epoch of the run.
    pub last_epoch: Instant,
}

impl Sp3CoverageSpan {
    /// Number of product epochs in the run, every one carrying the channel.
    pub fn epochs(&self) -> usize {
        self.last_index - self.first_index + 1
    }
}

/// A stretch of the product in which a satellite carries no value for a
/// channel, between two spans or at either end of the product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sp3CoverageGap {
    /// Index into [`Sp3::epochs`] of the last epoch before the gap that
    /// carries the channel. `None` when the gap opens the product.
    pub after_index: Option<usize>,
    /// Index into [`Sp3::epochs`] of the first epoch after the gap that
    /// carries the channel. `None` when the gap closes the product.
    pub before_index: Option<usize>,
    /// Product epochs inside the gap, none of which carries the channel. Zero
    /// when the gap is only a break in the product's own epoch list - a step
    /// longer than one grid step, an epoch out of order, or an epoch no record
    /// states exactly - with the channel carried on both sides.
    pub missing_epochs: usize,
}

/// Where a satellite carries one channel (positions or clocks) in a product.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3ChannelCoverage {
    /// Number of product epochs at which the channel is carried.
    pub epochs: usize,
    /// Maximal runs of carried epochs, in file order. A run ends at a product
    /// epoch that does not carry the channel and wherever
    /// [`Sp3::satellite_coverage`] says a step ends one.
    pub spans: Vec<Sp3CoverageSpan>,
    /// Every stretch without the channel, in file order: before the first
    /// span, between spans and after the last. A channel carried nowhere has
    /// one gap covering the whole product.
    pub gaps: Vec<Sp3CoverageGap>,
}

impl Sp3ChannelCoverage {
    /// The first epoch carrying the channel.
    pub fn first_epoch(&self) -> Option<Instant> {
        self.spans.first().map(|span| span.first_epoch)
    }

    /// The last epoch carrying the channel.
    pub fn last_epoch(&self) -> Option<Instant> {
        self.spans.last().map(|span| span.last_epoch)
    }

    /// Whether the channel is carried at every product epoch without a break.
    pub fn is_complete(&self) -> bool {
        self.epochs > 0 && self.gaps.is_empty()
    }
}

/// One satellite's position and clock coverage in a product.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3SatelliteCoverage {
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// Whether the header satellite list declares it.
    pub declared: bool,
    /// Epochs with a position record.
    pub positions: Sp3ChannelCoverage,
    /// Epochs with a clock: a position record's clock, or a clock-only record
    /// (missing-orbit sentinel with a valid clock).
    pub clocks: Sp3ChannelCoverage,
}

/// Position and clock coverage of every satellite in a product, with the grid
/// its epochs lie on.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3Coverage {
    /// The grid the product's epochs lie on, and whether the header's declared
    /// interval is its step.
    pub grid: Sp3EpochGrid,
    /// Each satellite's coverage, ascending by satellite.
    pub satellites: Vec<Sp3SatelliteCoverage>,
}

impl Sp3 {
    /// Position and clock coverage of every satellite, with the product's
    /// epoch grid.
    ///
    /// Every satellite the header declares is listed, including one with no
    /// record at all, and so is every satellite with a record that the header
    /// does not declare. Coverage is stated on the product's own epoch list:
    /// indices are into [`Sp3::epochs`]. Epochs are placed on the exact
    /// 10-nanosecond tick axis, and the grid is the one rule the merge also
    /// applies ([`Sp3EpochGrid`]): a uniform product's own step whatever its
    /// header says, or the header interval when the product skips epochs of
    /// it. A span ends at an epoch that does not carry the channel, at a step
    /// longer than one grid step (the product has no epoch there), at an epoch
    /// out of time order and on either side of an epoch no record states
    /// exactly; the grid reports those epochs. When the epochs lie on no grid,
    /// steps end no span.
    pub fn satellite_coverage(&self) -> Sp3Coverage {
        let epochs = self.epochs.len().min(self.states.len());
        let mut satellites: Vec<GnssSatelliteId> = self.header.satellites.clone();
        for index in 0..epochs {
            satellites.extend(self.states[index].keys().copied());
            if let Some(clock_records) = self.clock_records.get(index) {
                satellites.extend(clock_records.keys().copied());
            }
        }
        satellites.sort_unstable();
        satellites.dedup();

        let ticks = product_ticks(self);
        let facts = product_grid(&ticks[..epochs], self.header.epoch_interval_s);
        let step_breaks: Vec<bool> = (1..epochs)
            .map(|index| match (ticks[index - 1], ticks[index]) {
                (Some(before), Some(after)) => {
                    after <= before || facts.step.is_some_and(|step| after - before > step)
                }
                _ => true,
            })
            .collect();

        let satellites = satellites
            .into_iter()
            .map(|satellite| {
                let position_at: Vec<bool> = (0..epochs)
                    .map(|index| self.states[index].contains_key(&satellite))
                    .collect();
                let clock_at: Vec<bool> = (0..epochs)
                    .map(|index| {
                        self.states[index]
                            .get(&satellite)
                            .is_some_and(|state| state.clock_s.is_some())
                            || self
                                .clock_records
                                .get(index)
                                .is_some_and(|records| records.contains_key(&satellite))
                    })
                    .collect();
                Sp3SatelliteCoverage {
                    satellite,
                    declared: self.header.satellites.contains(&satellite),
                    positions: channel_coverage(&self.epochs, &position_at, &step_breaks),
                    clocks: channel_coverage(&self.epochs, &clock_at, &step_breaks),
                }
            })
            .collect();
        Sp3Coverage {
            grid: facts.public(),
            satellites,
        }
    }
}

/// Spans and gaps of one channel, from whether each product epoch carries it
/// and whether the step into each epoch (from index 1) ends a span.
fn channel_coverage(
    epochs: &[Instant],
    carried: &[bool],
    step_breaks: &[bool],
) -> Sp3ChannelCoverage {
    let mut spans: Vec<Sp3CoverageSpan> = Vec::new();
    let mut open: Option<usize> = None;
    for (index, &here) in carried.iter().enumerate() {
        let continues = open.is_some() && here && !step_breaks[index - 1];
        if let Some(first) = open {
            if !continues {
                spans.push(span(epochs, first, index - 1));
                open = None;
            }
        }
        if here && open.is_none() {
            open = Some(index);
        }
    }
    if let Some(first) = open {
        spans.push(span(epochs, first, carried.len() - 1));
    }

    let mut gaps: Vec<Sp3CoverageGap> = Vec::new();
    match (spans.first(), spans.last()) {
        (Some(first), Some(last)) => {
            if first.first_index > 0 {
                gaps.push(Sp3CoverageGap {
                    after_index: None,
                    before_index: Some(first.first_index),
                    missing_epochs: first.first_index,
                });
            }
            for pair in spans.windows(2) {
                gaps.push(Sp3CoverageGap {
                    after_index: Some(pair[0].last_index),
                    before_index: Some(pair[1].first_index),
                    missing_epochs: pair[1].first_index - pair[0].last_index - 1,
                });
            }
            if last.last_index + 1 < carried.len() {
                gaps.push(Sp3CoverageGap {
                    after_index: Some(last.last_index),
                    before_index: None,
                    missing_epochs: carried.len() - last.last_index - 1,
                });
            }
        }
        _ if !carried.is_empty() => gaps.push(Sp3CoverageGap {
            after_index: None,
            before_index: None,
            missing_epochs: carried.len(),
        }),
        _ => {}
    }

    Sp3ChannelCoverage {
        epochs: carried.iter().filter(|&&here| here).count(),
        spans,
        gaps,
    }
}

fn span(epochs: &[Instant], first_index: usize, last_index: usize) -> Sp3CoverageSpan {
    Sp3CoverageSpan {
        first_index,
        last_index,
        first_epoch: epochs[first_index],
        last_epoch: epochs[last_index],
    }
}
