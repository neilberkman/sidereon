//! The RINEX clock writer's policy for departures from what a product states,
//! and the departures it reports.

use std::fmt;

use crate::astro::time::model::Instant;

/// Whether the RINEX clock writer may emit one kind of departure from what a
/// product states, or refuses to write it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClockWriteLeniency {
    /// Refuse to write, naming what cannot be stated.
    #[default]
    Strict,
    /// Write, and report the departure as a [`ClockWriteDeparture`].
    Allow,
}

/// The policies the RINEX clock writer applies to departures from what a
/// product states.
///
/// The default allows none, so [`super::RinexClock::to_rinex_string`] writes
/// exactly what the product holds or refuses. A field set to
/// [`ClockWriteLeniency::Allow`] lets
/// [`super::RinexClock::to_rinex_string_with_policy`] emit that departure and
/// return each one it emitted, so a caller asking for an approximate file is
/// told exactly where it departs.
///
/// Values are never approximated under any policy: a value no 19-column field
/// states exactly is refused, because the file would read back as another
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ClockWritePolicy {
    /// Epochs that no microsecond text states exactly, written as the nearest
    /// microsecond text.
    pub nearest_microsecond_epochs: ClockWriteLeniency,
}

impl ClockWritePolicy {
    /// A policy that allows no departure, the same as
    /// [`ClockWritePolicy::default`] but usable in a constant.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            nearest_microsecond_epochs: ClockWriteLeniency::Strict,
        }
    }

    /// A policy that allows every departure.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            nearest_microsecond_epochs: ClockWriteLeniency::Allow,
        }
    }

    /// This policy with `nearest_microsecond_epochs`.
    #[must_use]
    pub const fn with_nearest_microsecond_epochs(
        mut self,
        nearest_microsecond_epochs: ClockWriteLeniency,
    ) -> Self {
        self.nearest_microsecond_epochs = nearest_microsecond_epochs;
        self
    }
}

/// A departure from what a product states that the writer emitted under a
/// [`ClockWritePolicy`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ClockWriteDeparture {
    /// An epoch no microsecond text states exactly, written as the nearest
    /// microsecond text.
    EpochAtNearestMicrosecond {
        /// Index of the record in [`super::RinexClock::records`] order.
        record: usize,
        /// Satellite or receiver name the record is written with.
        name: String,
        /// The epoch the product holds, when it has an instant.
        epoch: Option<Instant>,
        /// The epoch fields as written: year, month, day, hour, minute and
        /// seconds, separated by single blanks.
        written: String,
    },
}

impl fmt::Display for ClockWriteDeparture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpochAtNearestMicrosecond {
                record,
                name,
                written,
                ..
            } => write!(
                f,
                "record {record} ({name}) written at the nearest microsecond epoch {written}"
            ),
        }
    }
}
