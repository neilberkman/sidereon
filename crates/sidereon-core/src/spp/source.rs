//! Ephemeris-source abstraction for SPP.

use std::cell::RefCell;

use crate::id::GnssSatelliteId;
use crate::sp3::{MmapPreciseEphemerisInterpolant, PreciseEphemerisInterpolant, Sp3};

/// The relativistic clock term a positioning model applies to a source's satellite clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockRelativity {
    /// The clock needs no term here: it carries one already (broadcast, SSR- and
    /// SBAS-corrected clocks) or the source defines none.
    NotApplicable,
    /// Add this term, seconds, to the clock: `-2 r·v / c²` for a precise product clock,
    /// as RTKLIB `peph2pos` forms it.
    Term(f64),
    /// The clock needs the term and it cannot be formed: the position 1 ms later, or the
    /// state itself, is outside the product's coverage. RTKLIB `peph2pos` returns no state
    /// there, and the positioning models decline the satellite.
    Unavailable,
}

impl ClockRelativity {
    /// The term, seconds, when there is one to add.
    pub const fn term(self) -> Option<f64> {
        match self {
            Self::Term(term_s) => Some(term_s),
            Self::NotApplicable | Self::Unavailable => None,
        }
    }
}

/// A source of satellite position and clock at a transmit epoch.
///
/// The SPP pipeline is written against this trait rather than a concrete product
/// so it can run on either a precise SP3 ephemeris or a broadcast navigation
/// message. The contract is exactly what the transmit-time iteration needs: the
/// ECEF position (meters) and the satellite clock offset (seconds) at a given
/// J2000 second, or `None` if the source has no usable ephemeris for that
/// satellite at that instant.
pub trait EphemerisSource {
    /// ECEF position (m) and satellite clock offset (s) for `sat` at `t_j2000_s`.
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)>;

    /// Group delay, seconds, the single-frequency pseudorange model subtracts from the
    /// satellite clock of `sat` at `t_j2000_s`, or `None` for none.
    ///
    /// The clock [`Self::position_clock_at_j2000_s`] returns is the one RTKLIB
    /// `satposs` returns, without a group delay, which an ionosphere-free model uses
    /// as it is. RTKLIB `pntpos` applies the broadcast TGD or BGD to a single-frequency
    /// pseudorange (`prange`: `P1 - TGD`); the SPP model here subtracts this delay from
    /// the satellite clock instead, the same term on the other side of the equation.
    /// `None` by default: a precise clock carries no broadcast group delay.
    fn single_frequency_group_delay_s(
        &self,
        _sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Option<f64> {
        None
    }

    /// Relativistic clock term, seconds, the SPP model adds to the clock of
    /// [`Self::position_clock_at_j2000_s`] for `sat` at `t_j2000_s`, where that clock is a
    /// product clock that leaves it to the user.
    ///
    /// [`ClockRelativity::NotApplicable`] by default: the clock needs no term, because it
    /// carries one already (broadcast, SSR- and SBAS-corrected sources) or the source
    /// defines none. SP3 and precise-interpolant sources return `-2 r·v / c²` as RTKLIB
    /// `peph2pos` forms it, or [`ClockRelativity::Unavailable`] where it cannot be formed: the
    /// SP3 and RINEX CLK clock conventions leave the periodic relativistic term to the
    /// user, and `peph2pos` applies it for positioning. The clock itself stays the
    /// product's, so data access returns it as written.
    fn clock_relativity_s(&self, _sat: GnssSatelliteId, _t_j2000_s: f64) -> ClockRelativity {
        ClockRelativity::NotApplicable
    }

    /// [`Self::clock_relativity_s`] for a state this source has just returned:
    /// `position_m` is the position [`Self::position_clock_at_j2000_s`] (or
    /// [`Self::position_clock_group_delay_at_j2000_s`]) returned, with a clock, for `sat`
    /// at `t_j2000_s`. The answer is the one [`Self::clock_relativity_s`] gives there, bit
    /// for bit.
    ///
    /// The default calls [`Self::clock_relativity_s`]. A precise source overrides it to
    /// take the position it already interpolated and interpolate only the position 1 ms
    /// later, where [`Self::clock_relativity_s`] interpolates the state again.
    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        _position_m: [f64; 3],
    ) -> ClockRelativity {
        self.clock_relativity_s(sat, t_j2000_s)
    }

    /// [`Self::position_clock_at_j2000_s`] and [`Self::single_frequency_group_delay_s`]
    /// from one evaluation: the position, the clock and the group delay of the record
    /// they come from. The SPP transmit-time iteration calls this once per step.
    ///
    /// The default calls the two methods; a source that selects a record or forms a
    /// corrected state overrides it to do that once.
    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        let (position, clock) = self.position_clock_at_j2000_s(sat, t_j2000_s)?;
        Some((
            position,
            clock,
            self.single_frequency_group_delay_s(sat, t_j2000_s),
        ))
    }
}

impl EphemerisSource for Sp3 {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.position_at_j2000_seconds(sat, t_j2000_s).ok()?;
        let clk = state.clock_s?;
        Some((state.position.as_array(), clk))
    }

    /// The `peph2pos` relativistic term for the product clock this source returns.
    fn clock_relativity_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ClockRelativity {
        crate::sp3::peph2pos_clock_relativity(
            self.position_at_j2000_seconds(sat, t_j2000_s),
            || self.position_after_ephpos_step(sat, t_j2000_s),
        )
    }

    /// The `peph2pos` relativistic term for a state this source returned, from its
    /// position and the position 1 ms later.
    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        position_m: [f64; 3],
    ) -> ClockRelativity {
        crate::sp3::peph2pos_state_clock_relativity(position_m, || {
            self.position_after_ephpos_step(sat, t_j2000_s)
        })
    }
}

impl EphemerisSource for PreciseEphemerisInterpolant {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.position_at_j2000_seconds(sat, t_j2000_s).ok()?;
        let clk = state.clock_s?;
        Some((state.position.as_array(), clk))
    }

    /// The `peph2pos` relativistic term for the product clock this source returns.
    fn clock_relativity_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ClockRelativity {
        crate::sp3::peph2pos_clock_relativity(
            self.position_at_j2000_seconds(sat, t_j2000_s),
            || self.position_after_ephpos_step(sat, t_j2000_s),
        )
    }

    /// The `peph2pos` relativistic term for a state this source returned, from its
    /// position and the position 1 ms later.
    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        position_m: [f64; 3],
    ) -> ClockRelativity {
        crate::sp3::peph2pos_state_clock_relativity(position_m, || {
            self.position_after_ephpos_step(sat, t_j2000_s)
        })
    }
}

impl EphemerisSource for MmapPreciseEphemerisInterpolant<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.position_at_j2000_seconds(sat, t_j2000_s).ok()?;
        let clk = state.clock_s?;
        Some((state.position.as_array(), clk))
    }

    /// The `peph2pos` relativistic term for the product clock this source returns.
    fn clock_relativity_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ClockRelativity {
        crate::sp3::peph2pos_clock_relativity(
            self.position_at_j2000_seconds(sat, t_j2000_s),
            || self.position_after_ephpos_step(sat, t_j2000_s),
        )
    }

    /// The `peph2pos` relativistic term for a state this source returned, from its
    /// position and the position 1 ms later.
    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        position_m: [f64; 3],
    ) -> ClockRelativity {
        crate::sp3::peph2pos_state_clock_relativity(position_m, || {
            self.position_after_ephpos_step(sat, t_j2000_s)
        })
    }
}

/// Epochs remembered per satellite by [`TransmitStateMemo`]: the transmit-time
/// iteration's seed epoch, the epoch at the current receiver state, and one for each of
/// the three position probes of a finite-difference Jacobian, so the state at the
/// current epoch is still held when the receiver-clock probe asks for it again; one
/// more is spare.
const MEMO_EPOCHS_PER_SATELLITE: usize = 6;

/// Memo of the transmit-epoch states one SPP solve asks its source for.
///
/// A solve evaluates each satellite's model at many receiver states: in selection, at
/// every residual and finite-difference probe of the trust-region solve, and for the
/// final residuals and geometry. The transmit-time iteration's first step is at
/// `t_rx - P / c` every time, and a probe of a receiver clock leaves the transmit epoch
/// unchanged. The memo answers a repeated
/// [`EphemerisSource::position_clock_group_delay_at_j2000_s`] or
/// [`EphemerisSource::clock_relativity_for_state_s`] query, keyed by the satellite and
/// the bits of the epoch, with the answer the source gave before, which is the answer it
/// gives again: an ephemeris source's answers are functions of their arguments. Every
/// other query goes to the source.
pub(crate) struct TransmitStateMemo<'a> {
    source: &'a dyn EphemerisSource,
    satellites: RefCell<Vec<SatelliteMemo>>,
}

/// State of one satellite at one epoch, as the source returned it.
type MemoState = Option<([f64; 3], f64, Option<f64>)>;

#[derive(Clone, Copy)]
struct MemoEntry {
    t_bits: u64,
    state: Option<MemoState>,
    relativity: Option<ClockRelativity>,
}

struct SatelliteMemo {
    sat: GnssSatelliteId,
    /// Most recently used first.
    entries: [Option<MemoEntry>; MEMO_EPOCHS_PER_SATELLITE],
}

impl<'a> TransmitStateMemo<'a> {
    /// A memo over `source`, sized for `satellites` satellites.
    pub(crate) fn new(source: &'a dyn EphemerisSource, satellites: usize) -> Self {
        Self {
            source,
            satellites: RefCell::new(Vec::with_capacity(satellites)),
        }
    }

    /// Run `f` on the entry for `sat` at `t_j2000_s`, made empty if absent (replacing
    /// the least recently used), and mark it most recently used.
    fn with_entry<R>(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        f: impl FnOnce(&mut MemoEntry) -> R,
    ) -> R {
        let mut satellites = self.satellites.borrow_mut();
        let index = match satellites.iter().position(|memo| memo.sat == sat) {
            Some(index) => index,
            None => {
                satellites.push(SatelliteMemo {
                    sat,
                    entries: [None; MEMO_EPOCHS_PER_SATELLITE],
                });
                satellites.len() - 1
            }
        };
        let entries = &mut satellites[index].entries;
        let t_bits = t_j2000_s.to_bits();
        let slot = match entries
            .iter()
            .position(|entry| entry.is_some_and(|entry| entry.t_bits == t_bits))
        {
            Some(slot) => slot,
            None => {
                let slot = MEMO_EPOCHS_PER_SATELLITE - 1;
                entries[slot] = Some(MemoEntry {
                    t_bits,
                    state: None,
                    relativity: None,
                });
                slot
            }
        };
        entries[..=slot].rotate_right(1);
        let entry = entries[0]
            .as_mut()
            .expect("the entry was just placed first");
        f(entry)
    }
}

impl EphemerisSource for TransmitStateMemo<'_> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.source.position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        self.source.single_frequency_group_delay_s(sat, t_j2000_s)
    }

    fn clock_relativity_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ClockRelativity {
        self.source.clock_relativity_s(sat, t_j2000_s)
    }

    /// The source's term for its state at `t_j2000_s`; `position_m` is that state's
    /// position, so the epoch keys the term.
    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        position_m: [f64; 3],
    ) -> ClockRelativity {
        self.with_entry(sat, t_j2000_s, |entry| {
            *entry.relativity.get_or_insert_with(|| {
                self.source
                    .clock_relativity_for_state_s(sat, t_j2000_s, position_m)
            })
        })
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64, Option<f64>)> {
        self.with_entry(sat, t_j2000_s, |entry| {
            *entry.state.get_or_insert_with(|| {
                self.source
                    .position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
            })
        })
    }
}
