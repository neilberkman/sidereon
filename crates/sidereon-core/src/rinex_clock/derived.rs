//! Views derived from a product's records, maintained record by record.
//!
//! The satellite series, the skipped-record list and the record notices are
//! kept consistent with the body entries. A full build sorts once; an edit
//! adds or removes one record's contribution in logarithmic time plus the cost
//! of shifting a vector, so appending records in time order is linear overall.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::astro::time::model::Instant;

use super::epoch::epoch_cmp;
use super::header::ClockLayout;
use super::record::{ClockRecord, ClockRecordReading, ClockRecordType};
use super::{ClockPoint, RinexClockNotice, RinexClockSkip};

/// One `AS` record's place in its satellite's timeline: its epoch and the
/// file-order key of the body entry it came from.
#[derive(Debug, Clone, PartialEq)]
struct Sample {
    epoch: Instant,
    order: u64,
}

/// Views derived from the records.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct Derived {
    /// Every `AS` record with an epoch, per satellite, ordered by epoch and
    /// then file order; records repeating an epoch are all here.
    samples: BTreeMap<String, Vec<Sample>>,
    /// The public series: per satellite, one point per epoch, taken from the
    /// last record in file order with that epoch.
    series: BTreeMap<String, Vec<ClockPoint>>,
    /// Records read from the source that are not in the series, by line.
    skipped: Vec<RinexClockSkip>,
    /// Source lines of records with surplus values, read at the other
    /// layout's columns, and read as whitespace-separated values.
    surplus_lines: Vec<usize>,
    other_layout_lines: Vec<usize>,
    whitespace_lines: Vec<usize>,
}

/// Builds [`Derived`] from records given one at a time in body order, keeping
/// only each record's contribution.
#[derive(Debug)]
pub(super) struct DerivedBuilder {
    declared: Option<ClockLayout>,
    points: BTreeMap<String, Vec<(ClockPoint, u64)>>,
    derived: Derived,
}

impl DerivedBuilder {
    pub(super) fn new(declared: Option<ClockLayout>) -> Self {
        Self {
            declared,
            points: BTreeMap::new(),
            derived: Derived::default(),
        }
    }

    /// Add one record with its body entry's order key.
    pub(super) fn add(&mut self, record: &ClockRecord, order: u64) {
        if let Some((key, point)) = sample_of(record) {
            self.points.entry(key).or_default().push((point, order));
        }
        self.derived.note_lines(record, self.declared, true);
    }

    pub(super) fn finish(self) -> Derived {
        let mut derived = self.derived;
        derived.skipped.sort_by_key(|skip| skip.line);
        derived.surplus_lines.sort_unstable();
        derived.other_layout_lines.sort_unstable();
        derived.whitespace_lines.sort_unstable();
        for (key, mut entries) in self.points {
            entries.sort_by(|(a, ao), (b, bo)| sample_cmp(&a.epoch, *ao, &b.epoch, *bo));
            let samples = entries
                .iter()
                .map(|(point, order)| Sample {
                    epoch: point.epoch,
                    order: *order,
                })
                .collect();
            let mut series = Vec::<ClockPoint>::with_capacity(entries.len());
            for (point, _) in entries {
                match series.last_mut() {
                    Some(previous) if same_epoch(&previous.epoch, &point.epoch) => {
                        *previous = point;
                    }
                    _ => series.push(point),
                }
            }
            derived.samples.insert(key.clone(), samples);
            derived.series.insert(key, series);
        }
        derived
    }
}

/// The series key and point of an `AS` record with an epoch. A record read
/// from the source is keyed by its canonical satellite identifier; a typed
/// record by the name it was built with.
fn sample_of(record: &ClockRecord) -> Option<(String, ClockPoint)> {
    let point = record.clock_point()?;
    let key = match (record.line, &record.satellite) {
        (Some(_), Some(satellite)) => satellite.clone(),
        _ => record.name.clone(),
    };
    Some((key, point))
}

fn sample_cmp(a_epoch: &Instant, a_order: u64, b_epoch: &Instant, b_order: u64) -> Ordering {
    epoch_cmp(a_epoch, b_epoch).then_with(|| a_order.cmp(&b_order))
}

fn same_epoch(a: &Instant, b: &Instant) -> bool {
    epoch_cmp(a, b) == Ordering::Equal
}

/// Which line-keyed findings a source record contributes to.
fn record_findings(record: &ClockRecord, declared: Option<ClockLayout>) -> [bool; 3] {
    let readings = [Some(record.reading), record.continuation_reading];
    let whitespace = readings.contains(&Some(ClockRecordReading::Whitespace));
    let other_layout = !whitespace
        && declared.is_some_and(|declared| {
            readings.contains(&Some(ClockRecordReading::Columns(declared.other())))
        });
    [!record.surplus.is_empty(), other_layout, whitespace]
}

fn insert_line(lines: &mut Vec<usize>, line: usize) {
    let at = lines.partition_point(|&existing| existing < line);
    lines.insert(at, line);
}

fn remove_line(lines: &mut Vec<usize>, line: usize) {
    if let Ok(at) = lines.binary_search(&line) {
        lines.remove(at);
    }
}

impl Derived {
    /// Add one record's contribution. `order` is its body entry's order key.
    pub(super) fn insert(
        &mut self,
        record: &ClockRecord,
        order: u64,
        declared: Option<ClockLayout>,
    ) {
        if let Some((key, point)) = sample_of(record) {
            let samples = self.samples.entry(key.clone()).or_default();
            let at = samples.partition_point(|sample| {
                sample_cmp(&sample.epoch, sample.order, &point.epoch, order) == Ordering::Less
            });
            samples.insert(
                at,
                Sample {
                    epoch: point.epoch,
                    order,
                },
            );
            let last_of_epoch = samples
                .get(at + 1)
                .is_none_or(|next| !same_epoch(&next.epoch, &point.epoch));
            if last_of_epoch {
                let series = self.series.entry(key).or_default();
                let index = series.partition_point(|existing| {
                    epoch_cmp(&existing.epoch, &point.epoch) == Ordering::Less
                });
                if series
                    .get(index)
                    .is_some_and(|existing| same_epoch(&existing.epoch, &point.epoch))
                {
                    series[index] = point;
                } else {
                    series.insert(index, point);
                }
            }
        }
        self.note_lines(record, declared, false);
    }

    /// The order key of the record whose point becomes the series sample when
    /// `record` is removed: an earlier record in file order with the same
    /// satellite and epoch, when `record` is the one the series shows.
    pub(super) fn removal_replacement(&self, record: &ClockRecord, order: u64) -> Option<u64> {
        let (key, point) = sample_of(record)?;
        let samples = self.samples.get(&key)?;
        let at = samples
            .binary_search_by(|sample| sample_cmp(&sample.epoch, sample.order, &point.epoch, order))
            .ok()?;
        if samples
            .get(at + 1)
            .is_some_and(|next| same_epoch(&next.epoch, &point.epoch))
        {
            return None;
        }
        let previous = samples.get(at.checked_sub(1)?)?;
        same_epoch(&previous.epoch, &point.epoch).then_some(previous.order)
    }

    /// Remove one record's contribution. `replacement` is the point of the
    /// record [`Derived::removal_replacement`] named, when it named one.
    pub(super) fn remove(
        &mut self,
        record: &ClockRecord,
        order: u64,
        declared: Option<ClockLayout>,
        replacement: Option<ClockPoint>,
    ) {
        if let Some((key, point)) = sample_of(record) {
            if let Some(samples) = self.samples.get_mut(&key) {
                if let Ok(at) = samples.binary_search_by(|sample| {
                    sample_cmp(&sample.epoch, sample.order, &point.epoch, order)
                }) {
                    let shown = samples
                        .get(at + 1)
                        .is_none_or(|next| !same_epoch(&next.epoch, &point.epoch));
                    samples.remove(at);
                    if samples.is_empty() {
                        self.samples.remove(&key);
                    }
                    if shown {
                        if let Some(series) = self.series.get_mut(&key) {
                            let index = series.partition_point(|existing| {
                                epoch_cmp(&existing.epoch, &point.epoch) == Ordering::Less
                            });
                            if series
                                .get(index)
                                .is_some_and(|existing| same_epoch(&existing.epoch, &point.epoch))
                            {
                                match replacement {
                                    Some(replacement) => series[index] = replacement,
                                    None => {
                                        series.remove(index);
                                    }
                                }
                            }
                            if series.is_empty() {
                                self.series.remove(&key);
                            }
                        }
                    }
                }
            }
        }
        let Some(line) = record.line else {
            return;
        };
        if record.record_type != ClockRecordType::As {
            if let Ok(at) = self.skipped.binary_search_by_key(&line, |skip| skip.line) {
                self.skipped.remove(at);
            }
        }
        let [surplus, other_layout, whitespace] = record_findings(record, declared);
        if surplus {
            remove_line(&mut self.surplus_lines, line);
        }
        if other_layout {
            remove_line(&mut self.other_layout_lines, line);
        }
        if whitespace {
            remove_line(&mut self.whitespace_lines, line);
        }
    }

    /// Replace every order key through `renumber`, which must preserve order.
    pub(super) fn renumber(&mut self, renumber: impl Fn(u64) -> u64) {
        for samples in self.samples.values_mut() {
            for sample in samples {
                sample.order = renumber(sample.order);
            }
        }
    }

    fn note_lines(&mut self, record: &ClockRecord, declared: Option<ClockLayout>, bulk: bool) {
        let Some(line) = record.line else {
            return;
        };
        let [surplus, other_layout, whitespace] = record_findings(record, declared);
        let add = |lines: &mut Vec<usize>| {
            if bulk {
                lines.push(line);
            } else {
                insert_line(lines, line);
            }
        };
        if surplus {
            add(&mut self.surplus_lines);
        }
        if other_layout {
            add(&mut self.other_layout_lines);
        }
        if whitespace {
            add(&mut self.whitespace_lines);
        }
        if record.record_type != ClockRecordType::As {
            let skip = RinexClockSkip::new(line, record.record_type.code());
            if bulk {
                self.skipped.push(skip);
            } else {
                let at = self
                    .skipped
                    .partition_point(|existing| existing.line < line);
                self.skipped.insert(at, skip);
            }
        }
    }

    pub(super) fn series(&self) -> &BTreeMap<String, Vec<ClockPoint>> {
        &self.series
    }

    pub(super) fn skipped(&self) -> &[RinexClockSkip] {
        &self.skipped
    }

    /// Notices for the record findings.
    pub(super) fn notices(&self) -> Vec<RinexClockNotice> {
        let notice = |lines: &[usize], make: fn(usize, usize) -> RinexClockNotice| {
            lines
                .first()
                .map(|&first_line| make(lines.len(), first_line))
        };
        [
            notice(&self.surplus_lines, |records, first_line| {
                RinexClockNotice::SurplusValues {
                    records,
                    first_line,
                }
            }),
            notice(&self.other_layout_lines, |records, first_line| {
                RinexClockNotice::OtherLayoutRecords {
                    records,
                    first_line,
                }
            }),
            notice(&self.whitespace_lines, |records, first_line| {
                RinexClockNotice::WhitespaceRecords {
                    records,
                    first_line,
                }
            }),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}
