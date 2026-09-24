//! Ephemeris-source abstraction for SPP.

use std::cell::{Cell, RefCell};

use crate::astro::time::{DegradeReason, Validated};
use crate::id::GnssSatelliteId;
use crate::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
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

/// ECEF position (m) and satellite clock offset (s) at one transmit epoch.
pub type PositionClock = ([f64; 3], f64);

/// ECEF position (m), satellite clock offset (s) and single-frequency group delay (s,
/// `None` for none) at one transmit epoch, from one evaluation.
pub type PositionClockGroupDelay = ([f64; 3], f64, Option<f64>);

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
    /// they come from.
    ///
    /// The default calls the two methods; a source that selects a record or forms a
    /// corrected state overrides it to do that once. The solves read through
    /// [`Self::try_position_clock_group_delay_at_j2000_s`], so a source that overrides
    /// this read overrides that one too: a source that never refuses wraps this read
    /// in it, as the broadcast store and the SBAS-corrected sources do.
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

    /// [`Self::position_clock_at_j2000_s`] with the reason a source refused
    /// a state it could otherwise produce.
    ///
    /// `Ok(None)` means the source has no usable ephemeris, exactly as `None`
    /// from [`Self::position_clock_at_j2000_s`]. `Err` means the source
    /// refused the state under a policy, such as
    /// [`crate::Error::Ut1OutsideCoverage`] from an SSR source whose
    /// centre-of-mass to antenna-phase-centre conversion reads UT1 outside
    /// the table under [`crate::astro::time::ValidityMode::Strict`]; a solve
    /// reports that refusal instead of dropping the satellite. A state
    /// produced under a permissive policy carries its departure in
    /// [`Validated::degraded`].
    ///
    /// The default implementation wraps [`Self::position_clock_at_j2000_s`]
    /// and never refuses; sources with such a policy override it.
    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClock>>, crate::Error> {
        Ok(self
            .position_clock_at_j2000_s(sat, t_j2000_s)
            .map(Validated::ok))
    }

    /// [`Self::position_clock_group_delay_at_j2000_s`] with the reason a source refused
    /// a state it could otherwise produce, as [`Self::try_position_clock_at_j2000_s`]
    /// reports it. The SPP transmit-time iteration calls this once per step, and the
    /// solves read every state through it.
    ///
    /// The default calls [`Self::try_position_clock_at_j2000_s`] and
    /// [`Self::single_frequency_group_delay_s`], so a source that states its refusals
    /// through [`Self::try_position_clock_at_j2000_s`] keeps them here too. A source
    /// that never refuses and forms the state and delay in one evaluation overrides it
    /// to wrap [`Self::position_clock_group_delay_at_j2000_s`].
    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        let Some(state) = self.try_position_clock_at_j2000_s(sat, t_j2000_s)? else {
            return Ok(None);
        };
        let (position, clock) = state.value;
        Ok(Some(Validated {
            value: (
                position,
                clock,
                self.single_frequency_group_delay_s(sat, t_j2000_s),
            ),
            degraded: state.degraded,
        }))
    }

    /// [`Self::try_position_clock_group_delay_at_j2000_s`] at `t_j2000_s` of the record
    /// the source selects at `selection_j2000_s`.
    ///
    /// RTKLIB `satposs` selects each satellite's broadcast record by the observation epoch
    /// (`seleph(teph, ...)`, `teph` the reception epoch) and evaluates it at the
    /// transmission epoch; the positioning models pass the reception epoch here. A source
    /// that chooses among records by epoch, as a broadcast store does, chooses at
    /// `selection_j2000_s`, and the SSR- and SBAS-corrected sources choose their broadcast
    /// records there. A continuous product, such as SP3 or a precise interpolant, has no
    /// record to choose, and the default reads the state at `t_j2000_s`.
    fn try_position_clock_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        _selection_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        self.try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
    }

    /// Satellite clock offset, seconds, that places the transmission epoch of a
    /// pseudorange received at `t_rx`: RTKLIB `satposs` reads the clock with `ephclk` at
    /// `t_j2000_s = t_rx - P / c`, from the record selected at the reception epoch
    /// `selection_j2000_s`, and the transmission epoch is that epoch less the clock
    /// ([`crate::observables::pseudorange_transmit_epoch_j2000_s`]).
    ///
    /// `Ok(None)` where the source has no clock for `sat` at `t_j2000_s`, and `Err` for a
    /// refusal, as [`Self::try_position_clock_at_j2000_s`] reports them. The default is
    /// the clock [`Self::try_position_clock_group_delay_selected_at_j2000_s`] returns
    /// there: a precise product's clock as written, without the `peph2pos` relativistic
    /// term, which RTKLIB reads only for the state at the transmission epoch. A broadcast
    /// store returns its clock polynomial alone, as `ephclk` does (`eph2clk`, `geph2clk`),
    /// and the SSR- and SBAS-corrected sources return that of the broadcast store they
    /// correct, as `satposs` calls `ephclk` whatever the ephemeris option. RTKLIB reads
    /// the broadcast clock for a precise ephemeris too; a precise source here has no
    /// broadcast store, and its own clock is the one it has. The two differ by
    /// nanoseconds, which move the transmission epoch by as much and the satellite by
    /// micrometres.
    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<f64>>, crate::Error> {
        Ok(self
            .try_position_clock_group_delay_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s)?
            .map(|state| Validated {
                value: state.value.1,
                degraded: state.degraded,
            }))
    }

    /// The SSR corrections this source applies, which a solve asks why the source has no
    /// state for a satellite ([`crate::ssr::SsrCorrectionSource::correction_size_refusal`]).
    /// `None`, the default, for a source that applies no SSR corrections; the SSR-corrected
    /// sources return themselves, and a source that wraps another returns the inner
    /// source's.
    fn ssr_correction_source(&self) -> Option<&dyn crate::ssr::SsrCorrectionSource> {
        None
    }

    /// Variance (m²) of the satellite position and clock error of the state
    /// [`Self::try_position_clock_group_delay_selected_at_j2000_s`] returns for `sat` at
    /// `t_j2000_s` from the record selected at `selection_j2000_s`: the `var` RTKLIB
    /// `satposs` returns with each satellite, which `rescode` adds to the pseudorange
    /// variance a single-point solve weights the satellite by.
    ///
    /// A broadcast store returns RTKLIB `var_uraeph` of the selected record's accuracy
    /// (the URA, or the Galileo SISA), the GLONASS `ERREPH_GLO` variance, or the SBAS
    /// URA variance; an SSR-corrected source RTKLIB `var_urassr` of the satellite's SSR
    /// URA; an SBAS-corrected source the variance of the fast correction it applies.
    /// `0.0` by default: the source states no error. The SP3 and precise-interpolant
    /// sources keep the default. RTKLIB `peph2pos` forms its variance from the SP3
    /// records' standard deviations, which [`Sp3`] does not keep, and from the distance
    /// to the nearest node of its linear clock interpolation, where these sources
    /// interpolate the clock by spline.
    fn ephemeris_variance_m2(
        &self,
        _sat: GnssSatelliteId,
        _t_j2000_s: f64,
        _selection_j2000_s: f64,
    ) -> f64 {
        0.0
    }
}

/// A view of an ephemeris source that reads it through its fallible methods
/// ([`EphemerisSource::try_position_clock_at_j2000_s`],
/// [`ObservableEphemerisSource::try_observable_state_at_j2000_s`]) and
/// remembers what it saw: the first UT1 refusal, which a solve returns as its
/// error, and the first accepted UT1 departure, which a solve reports on its
/// result. A refused satellite still reads as unavailable inside the solve, so
/// the solve never uses a state that was not produced, and the caller never
/// receives a result that silently lacks it.
pub(crate) struct Ut1Tracked<'a, S: ?Sized> {
    inner: &'a S,
    refusal: Cell<Option<DegradeReason>>,
    departure: Cell<Option<DegradeReason>>,
}

/// [`Ut1Tracked`] over a type-erased [`EphemerisSource`].
pub(crate) type Ut1TrackedSource<'a> = Ut1Tracked<'a, dyn EphemerisSource + 'a>;

impl<'a, S: ?Sized> Ut1Tracked<'a, S> {
    pub(crate) fn new(inner: &'a S) -> Self {
        Self {
            inner,
            refusal: Cell::new(None),
            departure: Cell::new(None),
        }
    }

    /// The first UT1 refusal seen, if any.
    pub(crate) fn refusal(&self) -> Option<DegradeReason> {
        self.refusal.get()
    }

    /// The first accepted UT1 departure seen, if any.
    pub(crate) fn departure(&self) -> Option<DegradeReason> {
        self.departure.get()
    }

    fn note_departure(&self, degraded: Option<DegradeReason>) {
        if self.departure.get().is_none() {
            self.departure.set(degraded);
        }
    }

    fn note_error(&self, error: &crate::Error) {
        if let crate::Error::Ut1OutsideCoverage(reason) = error {
            if self.refusal.get().is_none() {
                self.refusal.set(Some(*reason));
            }
        }
    }

    fn note_observables_error(&self, error: &ObservablesError) {
        if let ObservablesError::Ephemeris(error) = error {
            self.note_error(error);
        }
    }
}

impl<S: ?Sized> Ut1Tracked<'_, S> {
    fn note<T>(&self, result: &Result<Option<Validated<T>>, crate::Error>) {
        match result {
            Ok(Some(state)) => self.note_departure(state.degraded),
            Ok(None) => {}
            Err(error) => self.note_error(error),
        }
    }
}

impl<S: EphemerisSource + ?Sized> EphemerisSource for Ut1Tracked<'_, S> {
    fn ssr_correction_source(&self) -> Option<&dyn crate::ssr::SsrCorrectionSource> {
        self.inner.ssr_correction_source()
    }

    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let result = self.inner.try_position_clock_at_j2000_s(sat, t_j2000_s);
        self.note(&result);
        result.ok().flatten().map(|state| state.value)
    }

    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClock>>, crate::Error> {
        let result = self.inner.try_position_clock_at_j2000_s(sat, t_j2000_s);
        self.note(&result);
        result
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        self.inner.single_frequency_group_delay_s(sat, t_j2000_s)
    }

    fn clock_relativity_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ClockRelativity {
        self.inner.clock_relativity_s(sat, t_j2000_s)
    }

    fn clock_relativity_for_state_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        position_m: [f64; 3],
    ) -> ClockRelativity {
        self.inner
            .clock_relativity_for_state_s(sat, t_j2000_s, position_m)
    }

    fn position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<PositionClockGroupDelay> {
        self.try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
            .ok()
            .flatten()
            .map(|state| state.value)
    }

    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        let result = self
            .inner
            .try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s);
        self.note(&result);
        result
    }

    fn try_position_clock_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        let result = self
            .inner
            .try_position_clock_group_delay_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s);
        self.note(&result);
        result
    }

    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<f64>>, crate::Error> {
        let result = self
            .inner
            .try_transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s);
        self.note(&result);
        result
    }

    fn ephemeris_variance_m2(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> f64 {
        self.inner
            .ephemeris_variance_m2(sat, t_j2000_s, selection_j2000_s)
    }
}

impl<S: ObservableEphemerisSource + ?Sized> ObservableEphemerisSource for Ut1Tracked<'_, S> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        self.try_observable_state_at_j2000_s(sat, t_j2000_s)
            .map(|state| state.value)
    }

    fn try_observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Validated<ObservableState>, ObservablesError> {
        let result = self.inner.try_observable_state_at_j2000_s(sat, t_j2000_s);
        match &result {
            Ok(state) => self.note_departure(state.degraded),
            Err(error) => self.note_observables_error(error),
        }
        result
    }

    fn ssr_corrections(&self) -> Option<&dyn crate::ssr::SsrCorrectionSource> {
        self.inner.ssr_corrections()
    }

    fn clock_includes_relativity(&self) -> bool {
        self.inner.clock_includes_relativity()
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        ObservableEphemerisSource::single_frequency_group_delay_s(self.inner, sat, t_j2000_s)
    }

    fn clock_relativity_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> ClockRelativity {
        ObservableEphemerisSource::clock_relativity_s(self.inner, sat, t_j2000_s)
    }

    fn observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<(ObservableState, Option<f64>), ObservablesError> {
        self.try_observable_state_group_delay_at_j2000_s(sat, t_j2000_s)
            .map(|state| state.value)
    }

    fn try_observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Validated<(ObservableState, Option<f64>)>, ObservablesError> {
        let result = self
            .inner
            .try_observable_state_group_delay_at_j2000_s(sat, t_j2000_s);
        match &result {
            Ok(state) => self.note_departure(state.degraded),
            Err(error) => self.note_observables_error(error),
        }
        result
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        let result = self.inner.velocity_at_j2000_s(sat, t_j2000_s);
        if let Some(Err(error)) = &result {
            self.note_observables_error(error);
        }
        result
    }

    fn try_observable_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Validated<Option<f64>>, ObservablesError> {
        let result =
            self.inner
                .try_observable_transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s);
        match &result {
            Ok(clock) => self.note_departure(clock.degraded),
            Err(error) => self.note_observables_error(error),
        }
        result
    }

    fn try_observable_state_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Validated<(ObservableState, Option<f64>)>, ObservablesError> {
        let result = self
            .inner
            .try_observable_state_group_delay_selected_at_j2000_s(
                sat,
                t_j2000_s,
                selection_j2000_s,
            );
        match &result {
            Ok(state) => self.note_departure(state.degraded),
            Err(error) => self.note_observables_error(error),
        }
        result
    }

    fn velocity_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        let result = self
            .inner
            .velocity_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s);
        if let Some(Err(error)) = &result {
            self.note_observables_error(error);
        }
        result
    }
}

impl<S: crate::positioning::RinexSppAssemblySource + ?Sized>
    crate::positioning::RinexSppAssemblySource for Ut1Tracked<'_, S>
{
    fn rinex_spp_broadcast_corrections(&self) -> crate::positioning::RinexSppBroadcastCorrections {
        self.inner.rinex_spp_broadcast_corrections()
    }

    fn rinex_spp_ionosphere_at(
        &self,
        t_j2000_s: f64,
        corrections: &mut crate::positioning::RinexSppBroadcastCorrections,
    ) {
        self.inner.rinex_spp_ionosphere_at(t_j2000_s, corrections);
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

/// Entries remembered per satellite by [`TransmitStateMemo`], each keyed by a query epoch
/// and the epoch the record is selected at. The RTKLIB placement asks for three per
/// satellite whatever the receiver state: the clock at `t_rx - P / c` and the state at the
/// transmission epoch, both from the record selected at `t_rx`, and the relativistic term
/// of that state. The geometric light-time recipes ask for the seed
/// epoch, the epoch at the current receiver state, and one for each of the three
/// position probes of a finite-difference Jacobian, so the state at the current epoch is
/// still held when the receiver-clock probe asks for it again; one more is spare.
const MEMO_EPOCHS_PER_SATELLITE: usize = 6;

/// Memo of the transmit-epoch states one SPP solve asks its source for.
///
/// A solve evaluates each satellite's model at many receiver states: in selection, at
/// every residual and finite-difference probe of the trust-region solve, and for the
/// final residuals and geometry. The RTKLIB placement reads the clock at `t_rx - P / c`
/// and the state at the transmission epoch it gives, both fixed by the pseudorange and
/// selected at `t_rx`, so every evaluation after the first asks the same epochs. The memo
/// answers a repeated [`EphemerisSource::position_clock_group_delay_at_j2000_s`],
/// [`EphemerisSource::try_position_clock_group_delay_selected_at_j2000_s`],
/// [`EphemerisSource::try_transmit_epoch_clock_s`] or
/// [`EphemerisSource::clock_relativity_for_state_s`] query, keyed by the satellite and
/// the bits of the query and selection epochs, with the answer the source gave before, which is the answer it
/// gives again: an ephemeris source's answers are functions of their arguments. Every
/// other query goes to the source.
pub(crate) struct TransmitStateMemo<'a> {
    source: &'a dyn EphemerisSource,
    satellites: RefCell<Vec<SatelliteMemo>>,
}

/// State of one satellite at one epoch, as the source returned it, with the UT1
/// departure it was produced under. A refusal is not held: it is asked for again, so
/// it reaches the caller as the error every time.
type MemoState = Option<Validated<PositionClockGroupDelay>>;

#[derive(Clone, Copy)]
struct MemoEntry {
    t_bits: u64,
    /// Bits of the epoch the source selected its record at: the query epoch itself for a
    /// read that selects nothing apart.
    selection_bits: u64,
    state: Option<MemoState>,
    transmit_epoch_clock: Option<Option<Validated<f64>>>,
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

    /// Run `f` on the entry for `sat` at `t_j2000_s` with the record selected at
    /// `selection_j2000_s`, made empty if absent (replacing the least recently used), and
    /// mark it most recently used.
    fn with_entry<R>(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
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
        let selection_bits = selection_j2000_s.to_bits();
        let slot = match entries.iter().position(|entry| {
            entry.is_some_and(|entry| {
                entry.t_bits == t_bits && entry.selection_bits == selection_bits
            })
        }) {
            Some(slot) => slot,
            None => {
                let slot = MEMO_EPOCHS_PER_SATELLITE - 1;
                entries[slot] = Some(MemoEntry {
                    t_bits,
                    selection_bits,
                    state: None,
                    transmit_epoch_clock: None,
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
    fn ssr_correction_source(&self) -> Option<&dyn crate::ssr::SsrCorrectionSource> {
        self.source.ssr_correction_source()
    }

    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.source.position_clock_at_j2000_s(sat, t_j2000_s)
    }

    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClock>>, crate::Error> {
        self.source.try_position_clock_at_j2000_s(sat, t_j2000_s)
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
        self.with_entry(sat, t_j2000_s, t_j2000_s, |entry| {
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
        self.try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
            .ok()
            .flatten()
            .map(|state| state.value)
    }

    /// The source's fallible read, remembered when it gives a state or none; a
    /// refusal is asked for again, so every caller receives it.
    fn try_position_clock_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        self.remembered_state(sat, t_j2000_s, t_j2000_s, || {
            self.source
                .try_position_clock_group_delay_at_j2000_s(sat, t_j2000_s)
        })
    }

    /// The source's selected read, remembered as [`Self::try_position_clock_group_delay_at_j2000_s`]
    /// remembers its read, keyed by both epochs.
    fn try_position_clock_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        self.remembered_state(sat, t_j2000_s, selection_j2000_s, || {
            self.source
                .try_position_clock_group_delay_selected_at_j2000_s(
                    sat,
                    t_j2000_s,
                    selection_j2000_s,
                )
        })
    }

    /// The source's clock, remembered when it gives one or none; a refusal is asked for
    /// again, so every caller receives it.
    fn try_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<Validated<f64>>, crate::Error> {
        if let Some(clock) = self.with_entry(sat, t_j2000_s, selection_j2000_s, |entry| {
            entry.transmit_epoch_clock
        }) {
            return Ok(clock);
        }
        let result = self
            .source
            .try_transmit_epoch_clock_s(sat, t_j2000_s, selection_j2000_s);
        if let Ok(clock) = &result {
            let clock = *clock;
            self.with_entry(sat, t_j2000_s, selection_j2000_s, |entry| {
                entry.transmit_epoch_clock = Some(clock)
            });
        }
        result
    }

    fn ephemeris_variance_m2(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> f64 {
        self.source
            .ephemeris_variance_m2(sat, t_j2000_s, selection_j2000_s)
    }
}

impl TransmitStateMemo<'_> {
    /// The state remembered for `sat` at `t_j2000_s` with the record selected at
    /// `selection_j2000_s`, or `read`'s answer, remembered when it gives a state or none.
    fn remembered_state(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
        read: impl FnOnce() -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error>,
    ) -> Result<Option<Validated<PositionClockGroupDelay>>, crate::Error> {
        if let Some(state) = self.with_entry(sat, t_j2000_s, selection_j2000_s, |entry| entry.state)
        {
            return Ok(state);
        }
        let result = read();
        if let Ok(state) = &result {
            let state = *state;
            self.with_entry(sat, t_j2000_s, selection_j2000_s, |entry| {
                entry.state = Some(state)
            });
        }
        result
    }
}
