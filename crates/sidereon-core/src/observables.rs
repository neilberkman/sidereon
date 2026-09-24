//! Forward GNSS observable prediction.
//!
//! This module owns the language-independent geometry behind Sidereon'
//! `Observables.predict`: transmit-time iteration, Sagnac rotation, line of
//! sight, range rate, Doppler, and topocentric azimuth/elevation. Ephemeris
//! parsing and interpolation stay with their existing SP3/broadcast products.
//!
//! Two transmit-time models live here.
//!
//! - Prediction has no pseudorange. [`predict`], [`predict_ranges`],
//!   [`transmit_time_satellite_state`], [`predict_transmit_geometry`] and
//!   [`transmit_epoch_j2000_s`] solve the geometric light time
//!   `t_tx = t_rx - |r_s(t_tx) - r_r| / c` by a fixed iteration, rotate the satellite
//!   into the reception-epoch frame over the flight time when Sagnac correction is on,
//!   and keep every bit of the flight time. `t_rx` is taken as the true reception time.
//! - Positioning has a pseudorange, whose time tag carries the receiver clock offset.
//!   SPP, the static solve, DGNSS, tight fusion and the PPP rows place it as RTKLIB
//!   `satposs` does ([`pseudorange_transmit_epoch_j2000_s`]):
//!   `t_tx = t_rx - P / c - dts(t_rx - P / c)`, with no iteration, and range the state
//!   there with `geodist` ([`pseudorange_transmit_geometry`]): the unrotated position and
//!   the first-order Sagnac term. The geometric light time from the receiver's time tag
//!   misses the receiver clock offset `dtr` and moves each satellite by `v · dtr` along
//!   its track, which is a range error of `rdot · dtr`.

use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use std::f64::consts::PI;

use crate::astro::time::civil;
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use crate::constants::{
    AZIMUTH_ZENITH_EPS, C_M_S, DEGREES_PER_CIRCLE, DEGREES_PER_SEMICIRCLE, F_L1_HZ, J2000_JD,
    KM_TO_M, MICROSECONDS_PER_SECOND, OBSERVABLE_TRANSMIT_TIME_ITERATIONS, OMEGA_E_DOT_RAD_S,
    SECONDS_PER_DAY,
};
use crate::ephemeris::BroadcastEphemeris;
use crate::estimation::recipe::SagnacRecipe;
use crate::frame::Wgs84Geodetic;
use crate::id::GnssSatelliteId;
use crate::ionex::{
    ionex_slant_delay, ionex_slant_delay_with_policy, ionosphere_delay, Ionex, IonexSlantPolicy,
    IonoModel,
};
use crate::sp3::Sp3;
use crate::spp::EphemerisSource;
use crate::tropo::{tropo_mapping, tropo_zenith, MappingModel, Met, TropoModel};
use crate::validate;
use crate::Error;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const FD_HALF_S: f64 = 0.5;

/// Satellite state required by the observable predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservableState {
    /// Satellite ECEF position in meters at the query epoch.
    pub position_ecef_m: [f64; 3],
    /// Satellite clock offset in seconds. SP3 clocks can be absent.
    pub clock_s: Option<f64>,
}

/// Position sentinel written for a failed element in [`ObservableStateBatch`].
///
/// The matching [`ObservableStateBatch::element_results`] entry carries the
/// exact scalar error. Consumers must check that result entry before using the
/// position or clock arrays.
pub const OBSERVABLE_STATE_MISSING_POSITION_ECEF_M: [f64; 3] = [f64::NAN; 3];

/// Per-element category for a batched satellite-state query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableStateElementStatus {
    /// The element contains a usable state.
    Valid,
    /// The source has no usable state for this satellite and epoch.
    Gap,
    /// The scalar evaluator returned an error that is not a gap.
    Error,
}

/// Contiguous output arrays for a batched satellite-state query.
///
/// Element `i` of `positions_ecef_m`, `clocks_s`, and `element_results` belongs
/// to input satellite `i`. When `element_results[i]` is `Ok(())`, the position
/// and clock entries are the exact [`ObservableState`] returned by the scalar
/// evaluator. When it is `Err`, `positions_ecef_m[i]` is
/// [`OBSERVABLE_STATE_MISSING_POSITION_ECEF_M`] and `clocks_s[i]` is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservableStateBatch {
    /// Satellite ECEF positions in meters, one entry per input element.
    pub positions_ecef_m: Vec<[f64; 3]>,
    /// Satellite clock offsets in seconds, one entry per input element.
    pub clocks_s: Vec<Option<f64>>,
    /// Per-element scalar result, preserving the exact scalar error on failure.
    pub element_results: Vec<Result<(), ObservablesError>>,
}

/// Per-satellite status for an emission-epoch state/media batch row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmissionMediaStatus {
    /// The row contains state, clock, ionosphere, and troposphere outputs.
    Valid,
    /// The ephemeris product has no usable state for this satellite and epoch.
    Gap,
    /// The row had a state, but its elevation was below the requested cutoff.
    BelowElevationCutoff,
    /// The scalar evaluator returned a non-gap error.
    Error,
}

/// Options for [`emission_media_batch_at_j2000_s`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EmissionMediaBatchOptions<'a> {
    /// Carrier frequency used for ionospheric group delay, hertz.
    pub carrier_hz: f64,
    /// Troposphere and ionosphere correction options.
    pub media: ObservableMediaOptions<'a>,
    /// Optional minimum topocentric elevation, radians.
    ///
    /// Rows below this cutoff keep the satellite state and clock, set both media
    /// delays to `None`, and receive [`EmissionMediaStatus::BelowElevationCutoff`].
    pub min_elevation_rad: Option<f64>,
}

impl Default for EmissionMediaBatchOptions<'_> {
    fn default() -> Self {
        Self {
            carrier_hz: F_L1_HZ,
            media: ObservableMediaOptions::default(),
            min_elevation_rad: None,
        }
    }
}

/// Contiguous per-satellite outputs for emission-epoch state and media lookup.
///
/// Element `i` of every vector belongs to input satellite `i`. Missing product
/// coverage is represented loudly: the status is [`EmissionMediaStatus::Gap`],
/// the error is retained, and every value field is `None`. No missing row is
/// written as zero. A valid zero can still appear inside `Some(0.0)` when the
/// selected scalar model genuinely returns zero or a correction is disabled.
#[derive(Debug, Clone, PartialEq)]
pub struct EmissionMediaBatch {
    /// Satellite ECEF positions in meters, one entry per input element.
    pub positions_ecef_m: Vec<Option<[f64; 3]>>,
    /// Satellite clock offsets in seconds, one entry per input element.
    pub clocks_s: Vec<Option<f64>>,
    /// Ionospheric slant group delays in meters, one entry per input element.
    pub ionosphere_slant_delays_m: Vec<Option<f64>>,
    /// Tropospheric slant delays in meters, one entry per input element.
    pub troposphere_delays_m: Vec<Option<f64>>,
    /// Per-element typed status.
    pub statuses: Vec<EmissionMediaStatus>,
    /// Per-element non-success error, if any.
    pub element_errors: Vec<Option<ObservablesError>>,
}

/// Receiver-dependent setup for repeated emission/media batch calls.
///
/// Construct this once for a fixed receiver and pass it to
/// [`emission_media_batch_at_j2000_s_with_receiver_context_into`] to avoid
/// repeating the ECEF-to-geodetic conversion and local-frame trigonometry in
/// every row of every call.
#[derive(Debug, Clone, PartialEq)]
pub struct EmissionMediaReceiverContext {
    topocentric: TopocentricReceiver,
}

impl EmissionMediaReceiverContext {
    /// Build receiver setup for emission/media batch evaluation.
    pub fn new(receiver_ecef_m: [f64; 3]) -> Result<Self, ObservablesError> {
        validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_input_error)?;
        Ok(Self {
            topocentric: topocentric_receiver(receiver_ecef_m)?,
        })
    }

    /// Receiver ECEF position in meters.
    pub fn receiver_ecef_m(&self) -> [f64; 3] {
        self.topocentric.ecef_m
    }

    /// Receiver geodetic position used by media models.
    pub fn receiver_geodetic(&self) -> Wgs84Geodetic {
        self.topocentric.geodetic
    }
}

/// An ephemeris product usable by [`predict`].
pub trait ObservableEphemerisSource {
    /// ECEF position and optional satellite clock at seconds since J2000.
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError>;

    /// [`Self::observable_state_at_j2000_s`] with the UT1 departure the source
    /// accepted to produce the state.
    ///
    /// A source whose state reads UT1 under a UT1 policy, such as an SSR source
    /// converting a centre-of-mass orbit to the antenna phase centre, returns
    /// `Err(`[`ObservablesError::Ephemeris`]`(`[`crate::Error::Ut1OutsideCoverage`]`))`
    /// for a refusal under [`crate::astro::time::ValidityMode::Strict`], and the
    /// state with the departure in [`crate::astro::time::Validated::degraded`]
    /// under [`crate::astro::time::ValidityMode::Permissive`]. The default
    /// implementation wraps [`Self::observable_state_at_j2000_s`] and reports no
    /// departure; sources with such a policy override it.
    fn try_observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<ObservableState>, ObservablesError> {
        self.observable_state_at_j2000_s(sat, t_j2000_s)
            .map(crate::astro::time::Validated::ok)
    }

    /// The SSR orbit and clock corrections this source applies, when it applies any.
    ///
    /// `None` by default, for sources that apply no SSR corrections. The SSR-corrected
    /// sources return themselves, so a precise positioning solve can check that an SSR
    /// bias belongs to the solution of the orbit and clock it uses at each observation's
    /// transmission time. A source that wraps an SSR-corrected source returns it here to
    /// keep those biases usable.
    fn ssr_corrections(&self) -> Option<&dyn crate::ssr::SsrCorrectionSource> {
        None
    }

    /// Whether the satellite clock this source returns already includes the satellite's
    /// relativistic clock correction, so a model must not add `-2 r·v / c²` to it again.
    ///
    /// `false` by default: the clock is treated as a product clock that leaves the
    /// periodic relativistic term to the user, as SP3 and RINEX CLK clocks do. A broadcast
    /// source returns `true`: its clock carries the broadcast
    /// `F·e·√A·sin E` term, as RTKLIB `eph2pos` includes it, and a GLONASS broadcast
    /// clock needs no term, as RTKLIB `geph2pos` adds none. An SSR-corrected source
    /// returns `true`: its clock carries `-2 r·v / c²` (RTKLIB `satpos_ssr`, HAS SIS ICD
    /// Eq. 24). A source that wraps another returns the wrapped source's answer.
    fn clock_includes_relativity(&self) -> bool {
        false
    }

    /// Group delay, seconds, a single-frequency pseudorange model subtracts from the
    /// satellite clock of `sat` at `t_j2000_s`, or `None` for none.
    ///
    /// The clock [`Self::observable_state_at_j2000_s`] returns is the one RTKLIB
    /// `satposs` returns, without the broadcast TGD or BGD; an ionosphere-free model,
    /// such as the PPP rows, uses it as it is. RTKLIB `pntpos` applies the group delay
    /// to a single-frequency pseudorange (`prange`: `P1 - TGD`). `None` by default: a
    /// precise clock carries no broadcast group delay. A broadcast source returns its
    /// record's delay (see [`crate::spp::EphemerisSource::single_frequency_group_delay_s`]),
    /// and a source that wraps another returns the wrapped source's answer.
    fn single_frequency_group_delay_s(
        &self,
        _sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Option<f64> {
        None
    }

    /// Relativistic clock term, seconds, that a positioning model adds to the clock of
    /// [`Self::observable_state_at_j2000_s`] for `sat` at `t_j2000_s`, where that clock is
    /// a product clock that leaves it to the user.
    ///
    /// `NotApplicable` by default: the clock needs no term here, because it carries one
    /// already (broadcast, SSR- and SBAS-corrected sources) or the source defines none.
    /// SP3 and precise-interpolant sources return `-2 r·v / c²` as RTKLIB `peph2pos` forms
    /// it for positioning, or `Unavailable` where the position 1 ms later cannot be
    /// interpolated. The state itself stays the product's, so data access returns the
    /// clock as written. The PPP rows do not call this; they add their own term under
    /// [`Self::clock_includes_relativity`].
    fn clock_relativity_s(
        &self,
        _sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> crate::spp::ClockRelativity {
        crate::spp::ClockRelativity::NotApplicable
    }

    /// [`Self::observable_state_at_j2000_s`] and [`Self::single_frequency_group_delay_s`]
    /// from one evaluation. The default calls the two methods; a source that forms a
    /// corrected state overrides it to do that once.
    ///
    /// The solves read through [`Self::try_observable_state_group_delay_at_j2000_s`], so
    /// a source that overrides this read overrides that one too: a source that never
    /// refuses wraps this read in it, as the SBAS-corrected sources do.
    fn observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<(ObservableState, Option<f64>), ObservablesError> {
        let state = self.observable_state_at_j2000_s(sat, t_j2000_s)?;
        Ok((state, self.single_frequency_group_delay_s(sat, t_j2000_s)))
    }

    /// [`Self::observable_state_group_delay_at_j2000_s`] with the UT1 departure the
    /// source accepted to produce the state, or its refusal, as
    /// [`Self::try_observable_state_at_j2000_s`] reports them. The solves read every state
    /// through it.
    ///
    /// The default calls [`Self::try_observable_state_at_j2000_s`] and
    /// [`Self::single_frequency_group_delay_s`], so a source that states its refusals and
    /// departures through [`Self::try_observable_state_at_j2000_s`] keeps them here too.
    /// A source that never refuses and forms the state and delay in one evaluation
    /// overrides it to wrap [`Self::observable_state_group_delay_at_j2000_s`].
    fn try_observable_state_group_delay_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<(ObservableState, Option<f64>)>, ObservablesError>
    {
        let state = self.try_observable_state_at_j2000_s(sat, t_j2000_s)?;
        Ok(crate::astro::time::Validated {
            value: (
                state.value,
                self.single_frequency_group_delay_s(sat, t_j2000_s),
            ),
            degraded: state.degraded,
        })
    }

    /// [`Self::try_observable_state_group_delay_at_j2000_s`] at `t_j2000_s` of the record
    /// the source selects at `selection_j2000_s`.
    ///
    /// RTKLIB `satposs` selects each satellite's broadcast record by the observation epoch
    /// (`seleph(teph, ...)`, `teph` the reception epoch) and evaluates it at the
    /// transmission epoch; the positioning models pass the reception epoch here (see
    /// [`crate::spp::EphemerisSource::try_position_clock_group_delay_selected_at_j2000_s`]).
    /// The default reads the state at `t_j2000_s`, as for a continuous product with no
    /// record to choose; a broadcast source and the SSR- and SBAS-corrected sources choose
    /// at `selection_j2000_s`, and a source that wraps another returns the wrapped
    /// source's answer.
    fn try_observable_state_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        _selection_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<(ObservableState, Option<f64>)>, ObservablesError>
    {
        self.try_observable_state_group_delay_at_j2000_s(sat, t_j2000_s)
    }

    /// [`Self::velocity_at_j2000_s`] of the record the source selects at
    /// `selection_j2000_s`, as [`Self::try_observable_state_group_delay_selected_at_j2000_s`]
    /// selects it. The default is [`Self::velocity_at_j2000_s`].
    fn velocity_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        _selection_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        self.velocity_at_j2000_s(sat, t_j2000_s)
    }

    /// Satellite clock offset, seconds, that places the transmission epoch of a
    /// pseudorange, with the UT1 departure the source accepted to produce it: RTKLIB
    /// `satposs` reads the clock with `ephclk` at `t_j2000_s = t_rx - P / c`, from the
    /// record selected at the reception epoch `selection_j2000_s`, and the transmission
    /// epoch is that epoch less the clock ([`pseudorange_transmit_epoch_j2000_s`]).
    ///
    /// The value is `None` where the source's state carries no clock, and the error is
    /// the one [`Self::try_observable_state_group_delay_selected_at_j2000_s`] returns. The
    /// default is the clock of that state: a precise product's clock as written, without
    /// the `peph2pos` relativistic term. A broadcast source returns its clock polynomial
    /// alone, as `ephclk` does (`eph2clk`, `geph2clk`), and the SSR- and SBAS-corrected
    /// sources return that of the broadcast store they correct, as `satposs` calls
    /// `ephclk` whatever the ephemeris option (see
    /// [`crate::spp::EphemerisSource::try_transmit_epoch_clock_s`]). A source that
    /// wraps another returns the wrapped source's answer.
    fn try_observable_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<Option<f64>>, ObservablesError> {
        self.try_observable_state_group_delay_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s)
            .map(|state| crate::astro::time::Validated {
                value: state.value.0.clock_s,
                degraded: state.degraded,
            })
    }

    /// Satellite ECEF velocity, metres per second, of the state this source returns at
    /// `t_j2000_s`, when the source defines one.
    ///
    /// `None` by default: the predictors then difference the positions this source returns
    /// half a second either side of `t_j2000_s`. A source whose state can change model
    /// within that second, such as one that switches between broadcast records or applies
    /// corrections from a reference epoch on, returns the velocity of the state it uses at
    /// `t_j2000_s` here, so the velocity never blends two models and never needs a state
    /// the source does not have.
    fn velocity_at_j2000_s(
        &self,
        _sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        None
    }

    /// ECEF states for parallel satellite and epoch arrays.
    ///
    /// `satellites[i]` is evaluated at `epochs_j2000_s[i]`. The output is
    /// index-aligned with the input and preserves the scalar result for every
    /// element. A length mismatch is the only batch-level error.
    fn observable_states_at_j2000_s(
        &self,
        satellites: &[GnssSatelliteId],
        epochs_j2000_s: &[f64],
    ) -> Result<ObservableStateBatch, ObservablesError> {
        if satellites.len() != epochs_j2000_s.len() {
            return Err(ObservablesError::InvalidInput {
                field: "epochs_j2000_s",
                kind: ObservablesInputErrorKind::OutOfRange,
            });
        }

        let mut batch = ObservableStateBatch::with_capacity(satellites.len());
        for (&sat, &epoch_j2000_s) in satellites.iter().zip(epochs_j2000_s.iter()) {
            batch.push_state_result(self.observable_state_at_j2000_s(sat, epoch_j2000_s));
        }
        Ok(batch)
    }

    /// ECEF states for many satellites at one shared epoch.
    ///
    /// The output is index-aligned with `satellites` and preserves the scalar
    /// result for every element.
    fn observable_states_at_shared_j2000_s(
        &self,
        satellites: &[GnssSatelliteId],
        epoch_j2000_s: f64,
    ) -> ObservableStateBatch {
        let mut batch = ObservableStateBatch::with_capacity(satellites.len());
        for &sat in satellites {
            batch.push_state_result(self.observable_state_at_j2000_s(sat, epoch_j2000_s));
        }
        batch
    }
}

impl ObservableEphemerisSource for Sp3 {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let state = self
            .position_at_j2000_seconds(sat, t_j2000_s)
            .map_err(ObservablesError::Ephemeris)?;
        Ok(ObservableState {
            position_ecef_m: state.position.as_array(),
            clock_s: state.clock_s,
        })
    }

    /// The `peph2pos` relativistic term for the SP3 clock this source returns.
    fn clock_relativity_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> crate::spp::ClockRelativity {
        crate::sp3::peph2pos_clock_relativity(
            self.position_at_j2000_seconds(sat, t_j2000_s),
            || self.position_after_ephpos_step(sat, t_j2000_s),
        )
    }
}

impl ObservableEphemerisSource for BroadcastEphemeris {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let Some((position_ecef_m, clock_s)) =
            EphemerisSource::position_clock_at_j2000_s(self, sat, t_j2000_s)
        else {
            return Err(ObservablesError::NoEphemeris);
        };
        Ok(ObservableState {
            position_ecef_m,
            clock_s: Some(clock_s),
        })
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        self.selected_record_velocity(sat, t_j2000_s).map(Ok)
    }

    /// True: the broadcast clock carries the broadcast relativistic term (GPS, Galileo,
    /// QZSS, BeiDou), and a GLONASS broadcast clock takes none.
    fn clock_includes_relativity(&self) -> bool {
        true
    }

    fn single_frequency_group_delay_s(&self, sat: GnssSatelliteId, t_j2000_s: f64) -> Option<f64> {
        BroadcastEphemeris::single_frequency_group_delay_s(self, sat, t_j2000_s)
    }

    /// [`BroadcastEphemeris::transmit_epoch_clock_s`]: the clock polynomial alone, as
    /// RTKLIB `ephclk` reads it, of the record selected at `selection_j2000_s`.
    fn try_observable_transmit_epoch_clock_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<Option<f64>>, ObservablesError> {
        BroadcastEphemeris::transmit_epoch_clock_s(self, sat, t_j2000_s, selection_j2000_s)
            .map(|clock_s| crate::astro::time::Validated::ok(Some(clock_s)))
            .ok_or(ObservablesError::NoEphemeris)
    }

    /// The state, `satposs` clock and group delay at `t_j2000_s` of the record selected at
    /// `selection_j2000_s`.
    fn try_observable_state_group_delay_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<crate::astro::time::Validated<(ObservableState, Option<f64>)>, ObservablesError>
    {
        let (position_ecef_m, clock_s, group_delay) =
            EphemerisSource::try_position_clock_group_delay_selected_at_j2000_s(
                self,
                sat,
                t_j2000_s,
                selection_j2000_s,
            )
            .map_err(ObservablesError::Ephemeris)?
            .ok_or(ObservablesError::NoEphemeris)?
            .value;
        Ok(crate::astro::time::Validated::ok((
            ObservableState {
                position_ecef_m,
                clock_s: Some(clock_s),
            },
            group_delay,
        )))
    }

    fn velocity_selected_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        self.selected_record_velocity_at(sat, t_j2000_s, selection_j2000_s)
            .map(Ok)
    }
}

/// Input-validation failure category for observable prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservablesInputErrorKind {
    /// A floating-point input was NaN or infinite.
    NonFinite,
    /// A positive physical input was zero or negative.
    NotPositive,
    /// A non-negative physical input was negative.
    Negative,
    /// A finite numeric input was outside its accepted range.
    OutOfRange,
    /// A required input field was absent.
    Missing,
    /// A text field could not be parsed as a float.
    FloatParse,
    /// A text field could not be parsed as an integer.
    IntParse,
    /// A civil date field was out of range.
    InvalidCivilDate,
    /// A civil time field was out of range.
    InvalidCivilTime,
}

impl core::fmt::Display for ObservablesInputErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let label = match self {
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::Negative => "negative",
            Self::OutOfRange => "out of range",
            Self::Missing => "missing",
            Self::FloatParse => "invalid float",
            Self::IntParse => "invalid integer",
            Self::InvalidCivilDate => "invalid civil date",
            Self::InvalidCivilTime => "invalid civil time",
        };
        f.write_str(label)
    }
}

impl From<&validate::FieldError> for ObservablesInputErrorKind {
    fn from(error: &validate::FieldError) -> Self {
        match error {
            validate::FieldError::Missing { .. } => Self::Missing,
            validate::FieldError::NonFinite { .. } => Self::NonFinite,
            validate::FieldError::NotPositive { .. } => Self::NotPositive,
            validate::FieldError::Negative { .. } => Self::Negative,
            validate::FieldError::OutOfRange { .. } => Self::OutOfRange,
            validate::FieldError::FloatParse { .. } => Self::FloatParse,
            validate::FieldError::IntParse { .. } => Self::IntParse,
            validate::FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            validate::FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

/// Error returned by the observable predictor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservablesError {
    /// A public predictor input or ephemeris-source state was malformed,
    /// non-finite, or outside its physical domain.
    InvalidInput {
        /// The invalid input field.
        field: &'static str,
        /// The validation failure category.
        kind: ObservablesInputErrorKind,
    },
    /// The ephemeris product has no usable record for the satellite/epoch.
    NoEphemeris,
    /// The underlying ephemeris product returned a structured crate error.
    Ephemeris(Error),
    /// The selected media model returned a structured crate error.
    Media(Error),
}

impl core::fmt::Display for ObservablesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, kind } => {
                write!(f, "invalid observable input {field}: {kind}")
            }
            Self::NoEphemeris => write!(f, "no ephemeris"),
            Self::Ephemeris(err) => write!(f, "{err}"),
            Self::Media(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ObservablesError {}

impl ObservableStateBatch {
    /// Build an empty batch with capacity for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions_ecef_m: Vec::with_capacity(capacity),
            clocks_s: Vec::with_capacity(capacity),
            element_results: Vec::with_capacity(capacity),
        }
    }

    /// Number of elements in the batch.
    pub fn len(&self) -> usize {
        self.element_results.len()
    }

    /// Whether the batch contains no elements.
    pub fn is_empty(&self) -> bool {
        self.element_results.is_empty()
    }

    /// Reconstruct element `index` as the scalar state result.
    ///
    /// Returns `None` when `index` is out of range.
    pub fn element(&self, index: usize) -> Option<Result<ObservableState, &ObservablesError>> {
        match self.element_results.get(index)? {
            Ok(()) => Some(Ok(ObservableState {
                position_ecef_m: self.positions_ecef_m[index],
                clock_s: self.clocks_s[index],
            })),
            Err(error) => Some(Err(error)),
        }
    }

    /// Status category for element `index`.
    ///
    /// Returns `None` when `index` is out of range.
    pub fn element_status(&self, index: usize) -> Option<ObservableStateElementStatus> {
        match self.element_results.get(index)? {
            Ok(()) => Some(ObservableStateElementStatus::Valid),
            Err(error) if is_observable_state_gap(error) => Some(ObservableStateElementStatus::Gap),
            Err(_) => Some(ObservableStateElementStatus::Error),
        }
    }

    fn push_state_result(&mut self, result: Result<ObservableState, ObservablesError>) {
        match result {
            Ok(state) => {
                self.positions_ecef_m.push(state.position_ecef_m);
                self.clocks_s.push(state.clock_s);
                self.element_results.push(Ok(()));
            }
            Err(error) => {
                self.positions_ecef_m
                    .push(OBSERVABLE_STATE_MISSING_POSITION_ECEF_M);
                self.clocks_s.push(None);
                self.element_results.push(Err(error));
            }
        }
    }
}

impl EmissionMediaBatch {
    /// Build an empty batch with capacity for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions_ecef_m: Vec::with_capacity(capacity),
            clocks_s: Vec::with_capacity(capacity),
            ionosphere_slant_delays_m: Vec::with_capacity(capacity),
            troposphere_delays_m: Vec::with_capacity(capacity),
            statuses: Vec::with_capacity(capacity),
            element_errors: Vec::with_capacity(capacity),
        }
    }

    /// Number of elements in the batch.
    pub fn len(&self) -> usize {
        self.statuses.len()
    }

    /// Whether the batch contains no elements.
    pub fn is_empty(&self) -> bool {
        self.statuses.is_empty()
    }

    /// Remove all rows while retaining allocated storage for reuse.
    pub fn clear(&mut self) {
        self.positions_ecef_m.clear();
        self.clocks_s.clear();
        self.ionosphere_slant_delays_m.clear();
        self.troposphere_delays_m.clear();
        self.statuses.clear();
        self.element_errors.clear();
    }

    /// Ensure each output vector can hold at least `capacity` elements.
    pub fn reserve(&mut self, capacity: usize) {
        reserve_at_least(&mut self.positions_ecef_m, capacity);
        reserve_at_least(&mut self.clocks_s, capacity);
        reserve_at_least(&mut self.ionosphere_slant_delays_m, capacity);
        reserve_at_least(&mut self.troposphere_delays_m, capacity);
        reserve_at_least(&mut self.statuses, capacity);
        reserve_at_least(&mut self.element_errors, capacity);
    }

    /// Status category for element `index`.
    ///
    /// Returns `None` when `index` is out of range.
    pub fn element_status(&self, index: usize) -> Option<EmissionMediaStatus> {
        self.statuses.get(index).copied()
    }

    fn push_valid(&mut self, state: ObservableState, media: AppliedMediaCorrections) {
        self.positions_ecef_m.push(Some(state.position_ecef_m));
        self.clocks_s.push(state.clock_s);
        self.ionosphere_slant_delays_m
            .push(Some(media.ionosphere_m));
        self.troposphere_delays_m.push(Some(media.troposphere_m));
        self.statuses.push(EmissionMediaStatus::Valid);
        self.element_errors.push(None);
    }

    fn push_gap(&mut self, error: ObservablesError) {
        self.positions_ecef_m.push(None);
        self.clocks_s.push(None);
        self.ionosphere_slant_delays_m.push(None);
        self.troposphere_delays_m.push(None);
        self.statuses.push(EmissionMediaStatus::Gap);
        self.element_errors.push(Some(error));
    }

    fn push_below_cutoff(&mut self, state: ObservableState) {
        self.positions_ecef_m.push(Some(state.position_ecef_m));
        self.clocks_s.push(state.clock_s);
        self.ionosphere_slant_delays_m.push(None);
        self.troposphere_delays_m.push(None);
        self.statuses
            .push(EmissionMediaStatus::BelowElevationCutoff);
        self.element_errors.push(None);
    }

    fn push_error(&mut self, state: Option<ObservableState>, error: ObservablesError) {
        self.positions_ecef_m
            .push(state.map(|state| state.position_ecef_m));
        self.clocks_s.push(state.and_then(|state| state.clock_s));
        self.ionosphere_slant_delays_m.push(None);
        self.troposphere_delays_m.push(None);
        self.statuses.push(EmissionMediaStatus::Error);
        self.element_errors.push(Some(error));
    }
}

fn reserve_at_least<T>(values: &mut Vec<T>, capacity: usize) {
    if values.capacity() < capacity {
        values.reserve(capacity - values.capacity());
    }
}

/// Whether a scalar observable-state error represents a data gap.
///
/// This is the same classification used by ephemeris grid sampling: missing
/// data, out-of-range precise interpolation, and unknown satellites are gaps;
/// malformed inputs and other source errors are not.
pub fn is_observable_state_gap(error: &ObservablesError) -> bool {
    matches!(
        error,
        ObservablesError::NoEphemeris
            | ObservablesError::Ephemeris(crate::Error::EpochOutOfRange)
            | ObservablesError::Ephemeris(crate::Error::UnknownSatellite(_))
    )
}

/// Options controlling observable prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PredictOptions {
    /// Carrier frequency used to scale Doppler, hertz.
    pub carrier_hz: f64,
    /// Apply fixed-point light-time / transmit-time correction.
    pub light_time: bool,
    /// Apply Earth-rotation Sagnac correction.
    pub sagnac: bool,
}

/// Options controlling transmit-time satellite-state evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransmitTimeOptions {
    /// Apply fixed-point light-time / transmit-time correction.
    pub light_time: bool,
    /// Apply Earth-rotation Sagnac correction to the returned position/velocity.
    pub sagnac: bool,
}

impl Default for TransmitTimeOptions {
    fn default() -> Self {
        Self {
            light_time: true,
            sagnac: true,
        }
    }
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        }
    }
}

/// Troposphere correction settings for a predicted tracking observable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservableTroposphereCorrection {
    /// Surface meteorology used by the Saastamoinen zenith delay.
    pub met: Met,
    /// Mapping function applied to the zenith dry and wet delays.
    pub mapping: MappingModel,
}

impl Default for ObservableTroposphereCorrection {
    fn default() -> Self {
        Self {
            met: Met::new_unchecked(1013.25, 288.15, 0.5),
            mapping: MappingModel::Niell,
        }
    }
}

/// Ionosphere correction model for a predicted tracking observable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObservableIonosphereCorrection<'a> {
    /// Broadcast ionosphere model evaluated on the requested carrier.
    Broadcast(IonoModel),
    /// Parsed IONEX vertical-TEC grid evaluated on the requested carrier.
    Ionex(&'a Ionex),
    /// Parsed IONEX vertical-TEC grid evaluated with explicit coverage,
    /// missing-node and mapping policies.
    IonexWithPolicy(&'a Ionex, IonexSlantPolicy),
}

/// Optional media corrections for one predicted tracking observable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct ObservableMediaOptions<'a> {
    /// Neutral-atmosphere slant delay to add to the range, if present.
    pub troposphere: Option<ObservableTroposphereCorrection>,
    /// Ionospheric group delay to add to the range, if present.
    pub ionosphere: Option<ObservableIonosphereCorrection<'a>>,
}

impl ObservableMediaOptions<'_> {
    fn is_disabled(self) -> bool {
        self.troposphere.is_none() && self.ionosphere.is_none()
    }

    fn needs_instant(self) -> bool {
        self.troposphere.is_some()
            || matches!(
                self.ionosphere,
                Some(ObservableIonosphereCorrection::Broadcast(_))
            )
    }

    fn needs_carrier(self) -> bool {
        self.ionosphere.is_some()
    }

    fn needs_ionex_epoch(self) -> bool {
        matches!(
            self.ionosphere,
            Some(
                ObservableIonosphereCorrection::Ionex(_)
                    | ObservableIonosphereCorrection::IonexWithPolicy(_, _)
            )
        )
    }
}

/// Prediction options plus optional media corrections.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct MediaPredictOptions<'a> {
    /// Geometry, light-time, Sagnac, and carrier options.
    pub prediction: PredictOptions,
    /// Troposphere and ionosphere correction options.
    pub media: ObservableMediaOptions<'a>,
}

/// Media delays applied to a predicted tracking observable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppliedMediaCorrections {
    /// Slant tropospheric delay in meters.
    pub troposphere_m: f64,
    /// Ionospheric group delay in meters on the requested carrier.
    pub ionosphere_m: f64,
    /// Sum of troposphere and ionosphere delays in meters.
    pub total_m: f64,
}

impl Default for AppliedMediaCorrections {
    fn default() -> Self {
        Self {
            troposphere_m: 0.0,
            ionosphere_m: 0.0,
            total_m: 0.0,
        }
    }
}

/// Predicted observables with an additional media-corrected range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaPredictedObservables {
    /// Geometry, range-rate, Doppler, clock, and sky position prediction.
    pub prediction: PredictedObservables,
    /// Range after adding the selected media delays, meters.
    pub range_m: f64,
    /// Media delays applied to `range_m`.
    pub media: AppliedMediaCorrections,
}

/// Satellite state at its signal transmit time for one receive epoch.
///
/// `transmit_position_ecef_m` is the ephemeris position evaluated at
/// `transmit_time_j2000_s`. From [`transmit_time_satellite_state`],
/// `position_ecef_m` is that position transported into the receive-time ECEF frame
/// when [`TransmitTimeOptions::sagnac`] is enabled, and `velocity_m_s` is the ECEF
/// velocity at transmit time with the same transport applied. From
/// [`pseudorange_transmit_satellite_state`] neither is transported, and the range
/// carries RTKLIB's first-order Sagnac term instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmitTimeSatelliteState {
    /// Offset of the transmit time from the receive time, `t_rx - t_tx`, seconds: the
    /// geometric flight time from [`transmit_time_satellite_state`], and the flight time
    /// with the receiver and satellite clock offsets from
    /// [`pseudorange_transmit_satellite_state`].
    pub signal_flight_time_s: f64,
    /// [`Self::signal_flight_time_s`] rounded to whole microseconds, for display. The
    /// transmit time is not rounded.
    pub transmit_offset_us: i64,
    /// Transmit time as seconds since J2000.
    pub transmit_time_j2000_s: f64,
    /// Satellite clock offset at transmit time, seconds.
    pub clock_s: Option<f64>,
    /// Ephemeris ECEF satellite position at transmit time, metres.
    pub transmit_position_ecef_m: [f64; 3],
    /// Sagnac-transported ECEF satellite position, metres.
    pub position_ecef_m: [f64; 3],
    /// Sagnac-transported ECEF satellite velocity, metres per second.
    pub velocity_m_s: [f64; 3],
    /// Geometric range after optional Sagnac transport, metres.
    pub geometric_range_m: f64,
    /// Receiver-to-satellite line-of-sight unit vector in ECEF.
    pub los_unit: [f64; 3],
}

/// Predicted GNSS observables at one receive epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictedObservables {
    /// Geometric range after optional Sagnac rotation, meters.
    pub geometric_range_m: f64,
    /// Range-rate LOS projection, meters per second.
    pub range_rate_m_s: f64,
    /// Doppler shift at `PredictOptions::carrier_hz`, hertz.
    pub doppler_hz: f64,
    /// Satellite clock offset at transmit time, seconds.
    pub sat_clock_s: Option<f64>,
    /// Topocentric elevation, degrees.
    pub elevation_deg: f64,
    /// Topocentric azimuth in `[0, 360)`, degrees.
    ///
    /// At (and arbitrarily near) the receiver's zenith the azimuth is
    /// geometrically undefined; it is defined here to be exactly `0.0` once the
    /// horizontal line-of-sight projection falls below
    /// [`crate::constants::AZIMUTH_ZENITH_EPS`], rather than returning rounding
    /// noise or erroring.
    pub azimuth_deg: f64,
    /// Signal flight time `t_rx - t_tx` rounded to whole microseconds, for display. The
    /// transmit time is `t_rx` less the unrounded flight time.
    pub transmit_offset_us: i64,
    /// Transmit time as seconds since J2000.
    pub transmit_time_j2000_s: f64,
    /// Receiver-to-satellite line-of-sight unit vector in ECEF.
    pub los_unit: [f64; 3],
    /// Sagnac-rotated satellite ECEF position in meters.
    pub sat_pos_ecef_m: [f64; 3],
    /// Sagnac-rotated satellite ECEF velocity in meters per second.
    pub sat_velocity_m_s: [f64; 3],
}

/// Convert split Julian date to seconds since J2000.
pub fn j2000_seconds_from_split(jd_whole: f64, jd_fraction: f64) -> Result<f64, ObservablesError> {
    validate::finite(jd_whole, "jd_whole").map_err(map_input_error)?;
    validate::finite(jd_fraction, "jd_fraction").map_err(map_input_error)?;
    validate::finite(
        civil::j2000_seconds_from_split(jd_whole, jd_fraction),
        "j2000_seconds",
    )
    .map_err(map_input_error)
}

/// Evaluate optional media range corrections at a supplied topocentric geometry.
///
/// This is the correction kernel used by [`predict_with_media`] and
/// [`predict_ranges_with_media`]. Delays are positive meters and are summed in
/// IERS TN36 range-sign convention: neutral-atmosphere slant delay first, then
/// ionospheric group delay on `carrier_hz`.
pub fn observable_media_corrections(
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    t_rx_j2000_s: f64,
    carrier_hz: f64,
    options: ObservableMediaOptions<'_>,
) -> Result<AppliedMediaCorrections, ObservablesError> {
    if options.is_disabled() {
        return Ok(AppliedMediaCorrections::default());
    }
    validate::finite(elevation_rad, "elevation_rad").map_err(map_input_error)?;
    validate::finite(azimuth_rad, "azimuth_rad").map_err(map_input_error)?;
    if options.needs_carrier() {
        validate::finite_positive(carrier_hz, "carrier_hz").map_err(map_input_error)?;
    }
    let epoch = if options.needs_instant() {
        Some(media_instant(t_rx_j2000_s)?)
    } else {
        None
    };
    let ionex_epoch_j2000_s = if options.needs_ionex_epoch() {
        Some(rounded_j2000_seconds(t_rx_j2000_s)?)
    } else {
        None
    };

    let troposphere_m = match options.troposphere {
        Some(troposphere) => {
            let epoch = epoch.expect("troposphere media requires an epoch");
            let zenith = tropo_zenith(TropoModel::Saastamoinen, receiver, troposphere.met)
                .map_err(map_media_error)?;
            let mapping = tropo_mapping(troposphere.mapping, elevation_rad, receiver, epoch)
                .map_err(map_media_error)?;
            let delay_m = zenith.dry_m * mapping.dry + zenith.wet_m * mapping.wet;
            validate::finite(delay_m, "media.troposphere_m").map_err(map_input_error)?;
            delay_m
        }
        None => 0.0,
    };

    let ionosphere_m = match options.ionosphere {
        Some(ObservableIonosphereCorrection::Broadcast(model)) => {
            let epoch = epoch.expect("broadcast ionosphere media requires an epoch");
            let delay_m = ionosphere_delay(
                receiver,
                elevation_rad,
                azimuth_rad,
                epoch,
                carrier_hz,
                &model,
            )
            .map_err(map_media_error)?;
            validate::finite(delay_m, "media.ionosphere_m").map_err(map_input_error)?;
            delay_m
        }
        Some(ObservableIonosphereCorrection::Ionex(ionex)) => {
            let ionex_epoch_j2000_s =
                ionex_epoch_j2000_s.expect("IONEX media requires an integer epoch");
            let delay_m = ionex_slant_delay(
                ionex,
                receiver,
                elevation_rad,
                azimuth_rad,
                ionex_epoch_j2000_s,
                carrier_hz,
            )
            .map_err(map_media_error)?;
            validate::finite(delay_m, "media.ionosphere_m").map_err(map_input_error)?;
            delay_m
        }
        Some(ObservableIonosphereCorrection::IonexWithPolicy(ionex, policy)) => {
            let ionex_epoch_j2000_s =
                ionex_epoch_j2000_s.expect("IONEX media requires an integer epoch");
            let delay_m = ionex_slant_delay_with_policy(
                ionex,
                receiver,
                elevation_rad,
                azimuth_rad,
                ionex_epoch_j2000_s,
                carrier_hz,
                policy,
            )
            .map_err(map_media_error)?
            .delay_m;
            validate::finite(delay_m, "media.ionosphere_m").map_err(map_input_error)?;
            delay_m
        }
        None => 0.0,
    };

    let total_m = troposphere_m + ionosphere_m;
    validate::finite(total_m, "media.total_m").map_err(map_input_error)?;
    Ok(AppliedMediaCorrections {
        troposphere_m,
        ionosphere_m,
        total_m,
    })
}

/// Evaluate ECEF states for parallel satellite and epoch arrays.
///
/// This delegates to [`ObservableEphemerisSource::observable_states_at_j2000_s`].
pub fn observable_states_at_j2000_s(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    epochs_j2000_s: &[f64],
) -> Result<ObservableStateBatch, ObservablesError> {
    source.observable_states_at_j2000_s(satellites, epochs_j2000_s)
}

/// Evaluate ECEF states for many satellites at one shared epoch.
///
/// This delegates to
/// [`ObservableEphemerisSource::observable_states_at_shared_j2000_s`].
pub fn observable_states_at_shared_j2000_s(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    epoch_j2000_s: f64,
) -> ObservableStateBatch {
    source.observable_states_at_shared_j2000_s(satellites, epoch_j2000_s)
}

/// Evaluate satellite emission-epoch states, clocks, and media delays in one call.
///
/// `satellites[i]` is evaluated at `emission_epochs_j2000_s[i]`. For each row,
/// the function first calls the same scalar
/// [`ObservableEphemerisSource::observable_state_at_j2000_s`] path used by the
/// public state APIs. When a state is available, it derives topocentric geometry
/// from `receiver_ecef_m` and delegates the ionosphere/troposphere work to
/// [`observable_media_corrections`]. The output vectors are index-aligned with
/// the inputs.
///
/// Missing ephemeris coverage is not zero-filled: it is returned as
/// [`EmissionMediaStatus::Gap`] with all value fields set to `None` and the
/// scalar gap error retained in [`EmissionMediaBatch::element_errors`]. A length
/// mismatch is the only shape-level error.
pub fn emission_media_batch_at_j2000_s(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    receiver_ecef_m: [f64; 3],
    options: EmissionMediaBatchOptions<'_>,
) -> Result<EmissionMediaBatch, ObservablesError> {
    validate_emission_media_batch_inputs(
        satellites,
        emission_epochs_j2000_s,
        receiver_ecef_m,
        options,
    )?;

    let mut batch = EmissionMediaBatch::with_capacity(satellites.len());
    emission_media_batch_at_j2000_s_unchecked(
        source,
        satellites,
        emission_epochs_j2000_s,
        receiver_ecef_m,
        options,
        &mut batch,
    );
    Ok(batch)
}

/// Evaluate emission-epoch states and media delays into caller-owned storage.
///
/// This is the allocation-reuse variant of [`emission_media_batch_at_j2000_s`].
/// On success, `output` contains exactly the rows for the supplied inputs. Its
/// vectors retain capacity across calls, so repeated calls with stable maximum
/// lengths can run without heap allocation after warm-up.
pub fn emission_media_batch_at_j2000_s_into(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    receiver_ecef_m: [f64; 3],
    options: EmissionMediaBatchOptions<'_>,
    output: &mut EmissionMediaBatch,
) -> Result<(), ObservablesError> {
    validate_emission_media_batch_inputs(
        satellites,
        emission_epochs_j2000_s,
        receiver_ecef_m,
        options,
    )?;
    output.clear();
    output.reserve(satellites.len());
    emission_media_batch_at_j2000_s_unchecked(
        source,
        satellites,
        emission_epochs_j2000_s,
        receiver_ecef_m,
        options,
        output,
    );
    Ok(())
}

/// Evaluate emission-epoch states and media delays using cached receiver setup.
///
/// This combines receiver setup reuse with caller-owned output reuse. On a warm
/// call with enough output capacity, this path performs no heap allocation.
pub fn emission_media_batch_at_j2000_s_with_receiver_context_into(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    receiver: &EmissionMediaReceiverContext,
    options: EmissionMediaBatchOptions<'_>,
    output: &mut EmissionMediaBatch,
) -> Result<(), ObservablesError> {
    validate_emission_media_batch_context_inputs(satellites, emission_epochs_j2000_s, options)?;
    output.clear();
    output.reserve(satellites.len());
    emission_media_batch_at_j2000_s_with_receiver_unchecked(
        source,
        satellites,
        emission_epochs_j2000_s,
        TopocentricReceiverSource::Cached(&receiver.topocentric),
        options,
        output,
    );
    Ok(())
}

fn validate_emission_media_batch_inputs(
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    receiver_ecef_m: [f64; 3],
    options: EmissionMediaBatchOptions<'_>,
) -> Result<(), ObservablesError> {
    validate_emission_media_batch_context_inputs(satellites, emission_epochs_j2000_s, options)?;
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_input_error)?;
    Ok(())
}

fn validate_emission_media_batch_context_inputs(
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    options: EmissionMediaBatchOptions<'_>,
) -> Result<(), ObservablesError> {
    if satellites.len() != emission_epochs_j2000_s.len() {
        return Err(ObservablesError::InvalidInput {
            field: "emission_epochs_j2000_s",
            kind: ObservablesInputErrorKind::OutOfRange,
        });
    }
    validate_emission_media_batch_options(options)?;
    Ok(())
}

fn emission_media_batch_at_j2000_s_unchecked(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    receiver_ecef_m: [f64; 3],
    options: EmissionMediaBatchOptions<'_>,
    batch: &mut EmissionMediaBatch,
) {
    match topocentric_receiver(receiver_ecef_m) {
        Ok(receiver) => emission_media_batch_at_j2000_s_with_receiver_unchecked(
            source,
            satellites,
            emission_epochs_j2000_s,
            TopocentricReceiverSource::Cached(&receiver),
            options,
            batch,
        ),
        Err(_) => emission_media_batch_at_j2000_s_with_receiver_unchecked(
            source,
            satellites,
            emission_epochs_j2000_s,
            TopocentricReceiverSource::Ecef(receiver_ecef_m),
            options,
            batch,
        ),
    }
}

fn emission_media_batch_at_j2000_s_with_receiver_unchecked(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    emission_epochs_j2000_s: &[f64],
    receiver: TopocentricReceiverSource<'_>,
    options: EmissionMediaBatchOptions<'_>,
    batch: &mut EmissionMediaBatch,
) {
    for (&sat, &emission_epoch_j2000_s) in satellites.iter().zip(emission_epochs_j2000_s.iter()) {
        let state = match source.observable_state_at_j2000_s(sat, emission_epoch_j2000_s) {
            Ok(state) => state,
            Err(error) if is_observable_state_gap(&error) => {
                batch.push_gap(error);
                continue;
            }
            Err(error) => {
                batch.push_error(None, error);
                continue;
            }
        };

        if let Err(error) = validate_observable_state(&state) {
            batch.push_error(Some(state), error);
            continue;
        }

        let receiver_ecef_m = receiver.ecef_m();
        let dx = state.position_ecef_m[0] - receiver_ecef_m[0];
        let dy = state.position_ecef_m[1] - receiver_ecef_m[1];
        let dz = state.position_ecef_m[2] - receiver_ecef_m[2];
        let line_of_sight_m = [dx, dy, dz];
        let range = match geometric_range_m(line_of_sight_m) {
            Ok(range) => range,
            Err(error) => {
                batch.push_error(Some(state), error);
                continue;
            }
        };
        let topocentric = match receiver.topocentric(line_of_sight_m, range) {
            Ok(topocentric) => topocentric,
            Err(error) => {
                batch.push_error(Some(state), error);
                continue;
            }
        };

        if options
            .min_elevation_rad
            .is_some_and(|cutoff| topocentric.elevation_rad < cutoff)
        {
            batch.push_below_cutoff(state);
            continue;
        }

        let media = observable_media_corrections(
            topocentric.receiver,
            topocentric.elevation_rad,
            topocentric.azimuth_rad,
            emission_epoch_j2000_s,
            options.carrier_hz,
            options.media,
        );
        match media {
            Ok(media) => batch.push_valid(state, media),
            Err(error) => batch.push_error(Some(state), error),
        }
    }
}

/// Light-time seed, seconds, for an observation whose pseudorange is `pseudorange_m`: the
/// pseudorange over the speed of light, or [`NOMINAL_SIGNAL_FLIGHT_TIME_S`] when the
/// pseudorange is not a positive finite distance. The seed only picks where the
/// geometric light-time iteration ([`transmit_epoch_j2000_s`],
/// [`predict_transmit_geometry`]) first queries the source.
pub fn flight_time_seed_s(pseudorange_m: f64) -> f64 {
    let tau = pseudorange_m / C_M_S;
    if tau.is_finite() && tau > 0.0 {
        tau
    } else {
        NOMINAL_SIGNAL_FLIGHT_TIME_S
    }
}

/// Nominal GNSS signal flight time, seconds, used to seed the light-time iteration where no
/// pseudorange is available: about the flight time from a MEO satellite at mid elevation.
pub const NOMINAL_SIGNAL_FLIGHT_TIME_S: f64 = 0.075;

/// Epoch, seconds since J2000, at which RTKLIB `satposs` reads the satellite clock that
/// places the transmission epoch of a pseudorange `pseudorange_m` received at
/// `t_rx_j2000_s`: `t_rx - P / c`, as `timeadd(obs.time, -pr / CLIGHT)` forms it.
///
/// `t_rx` is the receiver's own time tag, so `P / c` carries the receiver clock offset
/// with the flight time and the satellite clock offset against it.
pub fn pseudorange_clock_epoch_j2000_s(t_rx_j2000_s: f64, pseudorange_m: f64) -> f64 {
    t_rx_j2000_s - pseudorange_m / C_M_S
}

/// Transmission epoch, seconds since J2000, of a pseudorange `pseudorange_m` received at
/// `t_rx_j2000_s`, given the satellite clock offset `clock_s` read at
/// [`pseudorange_clock_epoch_j2000_s`]: that epoch less the clock, as RTKLIB `satposs`
/// forms it (`timeadd(time, -dt)`).
///
/// The pseudorange is `c (t_rx - t_tx) + c (dtr - dts)` plus the media delays, so this
/// is the true transmission epoch up to the media delays over `c`, whatever the receiver
/// clock offset `dtr`. No light-time iteration takes part.
pub fn pseudorange_transmit_epoch_from_clock_j2000_s(
    t_rx_j2000_s: f64,
    pseudorange_m: f64,
    clock_s: f64,
) -> f64 {
    pseudorange_clock_epoch_j2000_s(t_rx_j2000_s, pseudorange_m) - clock_s
}

/// Transmission epoch, seconds since J2000, of `sat`'s pseudorange `pseudorange_m`
/// received at `t_rx_j2000_s`, placed as RTKLIB `satposs` places it: the clock
/// [`ObservableEphemerisSource::try_observable_transmit_epoch_clock_s`] gives at
/// [`pseudorange_clock_epoch_j2000_s`] from the record selected at the reception epoch
/// (RTKLIB `seleph(teph, ...)`), taken from that epoch
/// ([`pseudorange_transmit_epoch_from_clock_j2000_s`]).
///
/// The positioning models place every pseudorange this way. `pseudorange_m` has to be
/// a positive finite distance: RTKLIB reads a zero pseudorange as none and places no
/// satellite for it. [`ObservablesError::NoEphemeris`] where the source has no clock
/// for `sat` there.
pub fn pseudorange_transmit_epoch_j2000_s(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_rx_j2000_s: f64,
    pseudorange_m: f64,
) -> Result<f64, ObservablesError> {
    validate_pseudorange_placement_inputs(t_rx_j2000_s, pseudorange_m)?;
    let clock_epoch_j2000_s = pseudorange_clock_epoch_j2000_s(t_rx_j2000_s, pseudorange_m);
    let clock_s = source
        .try_observable_transmit_epoch_clock_s(sat, clock_epoch_j2000_s, t_rx_j2000_s)?
        .value
        .ok_or(ObservablesError::NoEphemeris)?;
    validate::finite(clock_s, "transmit epoch clock_s").map_err(map_input_error)?;
    let t_tx = pseudorange_transmit_epoch_from_clock_j2000_s(t_rx_j2000_s, pseudorange_m, clock_s);
    validate::finite(t_tx, "transmit_time_j2000_s").map_err(map_input_error)?;
    Ok(t_tx)
}

/// Transmit-time geometry of `sat` at the transmission epoch `t_tx_j2000_s` of a
/// pseudorange received at `t_rx_j2000_s` by a receiver at `receiver_ecef_m`, as RTKLIB
/// `pntpos`, `ppp` and `rtkpos` form it from the `satposs` state with `geodist` and
/// `satazel`.
///
/// `t_tx_j2000_s` is the pseudorange's transmission epoch
/// ([`pseudorange_transmit_epoch_j2000_s`]). The satellite position is the source's at
/// that epoch, from the record selected at the reception epoch as RTKLIB `satposs`
/// selects it (`seleph(teph, ...)`), in the ECEF frame of that epoch, not rotated. The geometric range is
/// `geodist`: `|r_s - r_r|`, plus the Sagnac term `ω (x_s y_r - y_s x_r) / c` when
/// `sagnac` is set. The line of sight is `(r_s - r_r) / |r_s - r_r|`, and the elevation
/// and azimuth are its own. [`TransmitGeometry::signal_flight_time_s`] is
/// `t_rx - t_tx`, which holds the receiver and satellite clock offsets with the flight
/// time.
pub fn pseudorange_transmit_geometry(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    t_tx_j2000_s: f64,
    sagnac: bool,
) -> Result<TransmitGeometry, ObservablesError> {
    let placed = pseudorange_placed_state(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        t_tx_j2000_s,
        sagnac,
    )?;
    let topocentric = topocentric(receiver_ecef_m, placed.line_of_sight_m, placed.distance_m)?;
    Ok(TransmitGeometry {
        signal_flight_time_s: placed.offset_s,
        transmit_offset_us: microseconds_from_tau(placed.offset_s),
        transmit_time_j2000_s: t_tx_j2000_s,
        sat_clock_s: placed.state.clock_s,
        sat_pos_ecef_m: placed.state.position_ecef_m,
        geometric_range_m: placed.range_m,
        los_unit: placed.los_unit,
        elevation_deg: topocentric.elevation_deg,
        azimuth_deg: topocentric.azimuth_deg,
    })
}

/// Satellite state of `sat` at the transmission epoch `t_tx_j2000_s` of a pseudorange
/// received at `t_rx_j2000_s`, with the velocity, as [`pseudorange_transmit_geometry`]
/// forms the geometry: position and velocity in the ECEF frame of the transmission
/// epoch, not rotated (`position_ecef_m` equals `transmit_position_ecef_m`), and the
/// `geodist` range. The velocity is [`pseudorange_transmit_velocity_m_s`]'s.
pub fn pseudorange_transmit_satellite_state(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    t_tx_j2000_s: f64,
    sagnac: bool,
) -> Result<TransmitTimeSatelliteState, ObservablesError> {
    let placed = pseudorange_placed_state(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        t_tx_j2000_s,
        sagnac,
    )?;
    let velocity = satellite_velocity_selected(source, sat, t_tx_j2000_s, t_rx_j2000_s)?;
    Ok(TransmitTimeSatelliteState {
        signal_flight_time_s: placed.offset_s,
        transmit_offset_us: microseconds_from_tau(placed.offset_s),
        transmit_time_j2000_s: t_tx_j2000_s,
        clock_s: placed.state.clock_s,
        transmit_position_ecef_m: placed.state.position_ecef_m,
        position_ecef_m: placed.state.position_ecef_m,
        velocity_m_s: velocity,
        geometric_range_m: placed.range_m,
        los_unit: placed.los_unit,
    })
}

/// Satellite ECEF velocity, metres per second, at the transmission epoch `t_tx_j2000_s`
/// of a pseudorange received at `t_rx_j2000_s`, in the ECEF frame of that epoch, not
/// rotated, from the record selected at the reception epoch as RTKLIB `satposs` selects
/// it: the source's own velocity
/// ([`ObservableEphemerisSource::velocity_selected_at_j2000_s`]) or a difference of its
/// positions half a second either side.
pub fn pseudorange_transmit_velocity_m_s(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_rx_j2000_s: f64,
    t_tx_j2000_s: f64,
) -> Result<[f64; 3], ObservablesError> {
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    validate::finite(t_tx_j2000_s, "transmit_time_j2000_s").map_err(map_input_error)?;
    satellite_velocity_selected(source, sat, t_tx_j2000_s, t_rx_j2000_s)
}

/// The state and `geodist` geometry shared by [`pseudorange_transmit_geometry`] and
/// [`pseudorange_transmit_satellite_state`].
struct PseudorangePlacedState {
    offset_s: f64,
    state: ObservableState,
    line_of_sight_m: [f64; 3],
    distance_m: f64,
    range_m: f64,
    los_unit: [f64; 3],
}

fn pseudorange_placed_state(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    t_tx_j2000_s: f64,
    sagnac: bool,
) -> Result<PseudorangePlacedState, ObservablesError> {
    validate_transmit_time_inputs(receiver_ecef_m, t_rx_j2000_s)?;
    validate::finite(t_tx_j2000_s, "transmit_time_j2000_s").map_err(map_input_error)?;
    let state = validated_selected_state(source, sat, t_tx_j2000_s, t_rx_j2000_s)?;
    let position = state.position_ecef_m;
    let line_of_sight_m = [
        position[0] - receiver_ecef_m[0],
        position[1] - receiver_ecef_m[1],
        position[2] - receiver_ecef_m[2],
    ];
    let distance_m = geometric_range_m(line_of_sight_m)?;
    let los_unit = [
        line_of_sight_m[0] / distance_m,
        line_of_sight_m[1] / distance_m,
        line_of_sight_m[2] / distance_m,
    ];
    let range_m = if sagnac {
        crate::geometry::range::sagnac_range_first_order(
            position,
            receiver_ecef_m,
            OMEGA_E_DOT_RAD_S,
            C_M_S,
        )
    } else {
        distance_m
    };
    validate::finite(range_m, "geometric_range_m").map_err(map_input_error)?;
    Ok(PseudorangePlacedState {
        offset_s: t_rx_j2000_s - t_tx_j2000_s,
        state,
        line_of_sight_m,
        distance_m,
        range_m,
        los_unit,
    })
}

fn validate_pseudorange_placement_inputs(
    t_rx_j2000_s: f64,
    pseudorange_m: f64,
) -> Result<(), ObservablesError> {
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    validate::finite_positive(pseudorange_m, "pseudorange_m").map_err(map_input_error)?;
    Ok(())
}

/// Transmission epoch, seconds since J2000, of `sat`'s signal received at
/// `t_rx_j2000_s` by a static receiver at `receiver_ecef_m`, from the geometry alone.
///
/// This is the geometric light-time iteration of [`predict`] alone, started from
/// `initial_tau_s` ([`flight_time_seed_s`]) so its first ephemeris query is near the
/// transmission time. It evaluates the ephemeris only at the iterated transmission
/// epochs, so it succeeds wherever `source` can place the satellite there. It serves
/// prediction, where no pseudorange exists; `t_rx_j2000_s` is taken as the true
/// reception time. The positioning models place a measured pseudorange with
/// [`pseudorange_transmit_epoch_j2000_s`] instead, which carries the receiver clock
/// offset this iteration cannot see.
pub fn transmit_epoch_j2000_s(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: TransmitTimeOptions,
    initial_tau_s: f64,
) -> Result<f64, ObservablesError> {
    validate_transmit_time_inputs(receiver_ecef_m, t_rx_j2000_s)?;
    let predict_options = PredictOptions {
        carrier_hz: F_L1_HZ,
        light_time: options.light_time,
        sagnac: options.sagnac,
    };
    solve_transmit_time_from(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        predict_options,
        initial_tau_s,
        TransmitEpochRounding::Exact,
    )
    .map(|solved| solved.transmit_time_j2000_s)
}

/// Transmit-time geometry of one observation, without the satellite velocity.
///
/// Two functions form it. [`predict_transmit_geometry`] solves the geometric light time,
/// and every field equals the one of the same name in [`PredictedObservables`] for the
/// same light-time seed. [`pseudorange_transmit_geometry`] places a measured pseudorange
/// as RTKLIB `satposs` does and ranges it with `geodist`; there the satellite position
/// is not rotated and the range carries the first-order Sagnac term instead. Range rate
/// and Doppler are left out because they need the satellite velocity;
/// [`transmit_velocity_m_s`] gives it where a caller needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmitGeometry {
    /// Offset of the transmission epoch from the reception epoch, `t_rx - t_tx`, seconds:
    /// the geometric flight time from [`predict_transmit_geometry`], and the flight time
    /// with the receiver and satellite clock offsets from
    /// [`pseudorange_transmit_geometry`].
    pub signal_flight_time_s: f64,
    /// [`Self::signal_flight_time_s`] rounded to whole microseconds, for display. The
    /// transmission epoch is not rounded.
    pub transmit_offset_us: i64,
    /// Transmission epoch, seconds since J2000.
    pub transmit_time_j2000_s: f64,
    /// Satellite clock offset at the transmission epoch, seconds, when the source has one.
    pub sat_clock_s: Option<f64>,
    /// Satellite ECEF position at the transmission epoch, metres: rotated into the
    /// reception-epoch frame when Sagnac correction is on for
    /// [`predict_transmit_geometry`], and in the transmission-epoch frame, not rotated, for
    /// [`pseudorange_transmit_geometry`].
    pub sat_pos_ecef_m: [f64; 3],
    /// Geometric range, metres: from the rotated position for
    /// [`predict_transmit_geometry`], and RTKLIB `geodist` for
    /// [`pseudorange_transmit_geometry`].
    pub geometric_range_m: f64,
    /// Receiver-to-satellite unit vector.
    pub los_unit: [f64; 3],
    /// Elevation, degrees.
    pub elevation_deg: f64,
    /// Azimuth, degrees.
    pub azimuth_deg: f64,
}

/// Transmit-time geometry of `sat` for a static receiver, with the geometric light-time
/// iteration started from `initial_tau_s` ([`flight_time_seed_s`]).
///
/// Unlike [`predict`], it never evaluates the satellite velocity, so it needs `source`
/// only at the iterated transmission epochs. Like [`predict`] it serves prediction,
/// where no pseudorange exists; a measured pseudorange is placed with
/// [`pseudorange_transmit_geometry`].
pub fn predict_transmit_geometry(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
    initial_tau_s: f64,
) -> Result<TransmitGeometry, ObservablesError> {
    validate_predict_inputs(receiver_ecef_m, t_rx_j2000_s, options)?;
    let solved = solve_transmit_time_from(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
        initial_tau_s,
        TransmitEpochRounding::Exact,
    )?;
    let dx = solved.sat_rot_ecef_m[0] - receiver_ecef_m[0];
    let dy = solved.sat_rot_ecef_m[1] - receiver_ecef_m[1];
    let dz = solved.sat_rot_ecef_m[2] - receiver_ecef_m[2];
    let range = geometric_range_m([dx, dy, dz])?;
    let los = [dx / range, dy / range, dz / range];
    let topocentric = topocentric(receiver_ecef_m, [dx, dy, dz], range)?;
    Ok(TransmitGeometry {
        signal_flight_time_s: solved.tau_s,
        transmit_offset_us: solved.transmit_offset_us,
        transmit_time_j2000_s: solved.transmit_time_j2000_s,
        sat_clock_s: solved.state.clock_s,
        sat_pos_ecef_m: solved.sat_rot_ecef_m,
        geometric_range_m: range,
        los_unit: los,
        elevation_deg: topocentric.elevation_deg,
        azimuth_deg: topocentric.azimuth_deg,
    })
}

/// Satellite ECEF velocity, metres per second, at the transmission epoch of `geometry`,
/// rotated into the reception-epoch frame when `sagnac` is set, as [`predict`] forms it:
/// the source's own velocity ([`ObservableEphemerisSource::velocity_at_j2000_s`]) or a
/// difference of its positions half a second either side.
pub fn transmit_velocity_m_s(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    geometry: &TransmitGeometry,
    sagnac: bool,
) -> Result<[f64; 3], ObservablesError> {
    let velocity = satellite_velocity(source, sat, geometry.transmit_time_j2000_s)?;
    let velocity_rot = sagnac_rotate(velocity, geometry.signal_flight_time_s, sagnac);
    validate::finite_vec3(velocity_rot, "satellite velocity_m_s").map_err(map_input_error)?;
    Ok(velocity_rot)
}

/// Evaluate a satellite's transmit-time ECEF state for one static receiver.
///
/// This is the per-satellite primitive underneath observable prediction: it
/// iterates light time, evaluates the ephemeris at the satellite's transmit
/// epoch, applies the Sagnac/Earth-rotation transport if requested, and returns
/// the transported position, velocity, clock, range, and line of sight without
/// constructing Doppler or topocentric observables.
pub fn transmit_time_satellite_state(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: TransmitTimeOptions,
) -> Result<TransmitTimeSatelliteState, ObservablesError> {
    transmit_time_satellite_state_with(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
        TransmitEpochRounding::Exact,
    )
}

fn transmit_time_satellite_state_with(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: TransmitTimeOptions,
    rounding: TransmitEpochRounding,
) -> Result<TransmitTimeSatelliteState, ObservablesError> {
    validate_transmit_time_inputs(receiver_ecef_m, t_rx_j2000_s)?;
    let predict_options = PredictOptions {
        carrier_hz: F_L1_HZ,
        light_time: options.light_time,
        sagnac: options.sagnac,
    };
    let solved = solve_transmit_time(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        predict_options,
        rounding,
    )?;

    let dx = solved.sat_rot_ecef_m[0] - receiver_ecef_m[0];
    let dy = solved.sat_rot_ecef_m[1] - receiver_ecef_m[1];
    let dz = solved.sat_rot_ecef_m[2] - receiver_ecef_m[2];
    let range = geometric_range_m([dx, dy, dz])?;
    let los = [dx / range, dy / range, dz / range];

    let velocity = satellite_velocity(source, sat, solved.transmit_time_j2000_s)?;
    let velocity_rot = sagnac_rotate(velocity, solved.tau_s, options.sagnac);
    validate::finite_vec3(velocity_rot, "satellite velocity_m_s").map_err(map_input_error)?;

    Ok(TransmitTimeSatelliteState {
        signal_flight_time_s: solved.tau_s,
        transmit_offset_us: solved.transmit_offset_us,
        transmit_time_j2000_s: solved.transmit_time_j2000_s,
        clock_s: solved.state.clock_s,
        transmit_position_ecef_m: solved.state.position_ecef_m,
        position_ecef_m: solved.sat_rot_ecef_m,
        velocity_m_s: velocity_rot,
        geometric_range_m: range,
        los_unit: los,
    })
}

/// Predict observables for `sat` from a static ECEF receiver.
pub fn predict(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<PredictedObservables, ObservablesError> {
    let (prediction, _) = predict_core(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
        TransmitEpochRounding::Exact,
    )?;
    Ok(prediction)
}

/// Predict observables and add optional troposphere and ionosphere range delays.
///
/// The embedded [`PredictedObservables`] keeps the geometric range and range-rate
/// fields unchanged. The corrected one-way range is reported as
/// [`MediaPredictedObservables::range_m`]. IERS TN36 treats the neutral
/// atmosphere and ionospheric group delay as positive additions to a code range;
/// no media range-rate derivative is applied here.
pub fn predict_with_media(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: MediaPredictOptions<'_>,
) -> Result<MediaPredictedObservables, ObservablesError> {
    let (prediction, topocentric) = predict_core(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options.prediction,
        TransmitEpochRounding::Exact,
    )?;
    if options.media.is_disabled() {
        return Ok(MediaPredictedObservables {
            range_m: prediction.geometric_range_m,
            prediction,
            media: AppliedMediaCorrections::default(),
        });
    }
    let media = observable_media_corrections(
        topocentric.receiver,
        topocentric.elevation_rad,
        topocentric.azimuth_rad,
        t_rx_j2000_s,
        options.prediction.carrier_hz,
        options.media,
    )?;
    let range_m = prediction.geometric_range_m + media.total_m;
    validate::finite(range_m, "range_m").map_err(map_input_error)?;
    Ok(MediaPredictedObservables {
        prediction,
        range_m,
        media,
    })
}

fn predict_core(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
    rounding: TransmitEpochRounding,
) -> Result<(PredictedObservables, TopocentricGeometry), ObservablesError> {
    validate_predict_inputs(receiver_ecef_m, t_rx_j2000_s, options)?;
    let solved = solve_transmit_time(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
        rounding,
    )?;

    let dx = solved.sat_rot_ecef_m[0] - receiver_ecef_m[0];
    let dy = solved.sat_rot_ecef_m[1] - receiver_ecef_m[1];
    let dz = solved.sat_rot_ecef_m[2] - receiver_ecef_m[2];
    let range = geometric_range_m([dx, dy, dz])?;
    let los = [dx / range, dy / range, dz / range];

    let velocity = satellite_velocity(source, sat, solved.transmit_time_j2000_s)?;
    let velocity_rot = sagnac_rotate(velocity, solved.tau_s, options.sagnac);
    validate::finite_vec3(velocity_rot, "satellite velocity_m_s").map_err(map_input_error)?;
    let range_rate = los[0] * velocity_rot[0] + los[1] * velocity_rot[1] + los[2] * velocity_rot[2];
    validate::finite(range_rate, "range_rate_m_s").map_err(map_input_error)?;
    let doppler_hz = -range_rate * options.carrier_hz / C_M_S;
    validate::finite(doppler_hz, "doppler_hz").map_err(map_input_error)?;
    let topocentric = topocentric(receiver_ecef_m, [dx, dy, dz], range)?;

    Ok((
        PredictedObservables {
            geometric_range_m: range,
            range_rate_m_s: range_rate,
            doppler_hz,
            sat_clock_s: solved.state.clock_s,
            elevation_deg: topocentric.elevation_deg,
            azimuth_deg: topocentric.azimuth_deg,
            transmit_offset_us: solved.transmit_offset_us,
            transmit_time_j2000_s: solved.transmit_time_j2000_s,
            los_unit: los,
            sat_pos_ecef_m: solved.sat_rot_ecef_m,
            sat_velocity_m_s: velocity_rot,
        },
        topocentric,
    ))
}

/// One batch prediction request: the satellite to observe, the static receiver
/// ECEF position in meters, and the receive epoch in seconds since J2000.
///
/// Each entry is fully independent of the others; the receiver position and
/// epoch are per-request, so a single batch can mix many satellites, many
/// receivers, and many epochs in any combination.
pub type PredictRequest = (GnssSatelliteId, [f64; 3], f64);

/// Predict observables for many `(satellite, receiver, epoch)` requests, serially.
///
/// Element `i` of the result is exactly what [`predict`] returns for
/// `requests[i]` (including its `Err`), evaluated with the shared `options`.
/// This is the single-threaded reference that [`predict_batch_parallel`] is
/// proven bit-identical against; it lets a caller predict a whole fleet/arc in
/// one call instead of paying per-call dispatch overhead in a host-language loop.
pub fn predict_batch(
    source: &dyn ObservableEphemerisSource,
    requests: &[PredictRequest],
    options: PredictOptions,
) -> Vec<Result<PredictedObservables, ObservablesError>> {
    requests
        .iter()
        .map(|&(sat, receiver_ecef_m, t_rx_j2000_s)| {
            predict(source, sat, receiver_ecef_m, t_rx_j2000_s, options)
        })
        .collect()
}

/// Predict media-corrected observables for many requests, serially.
///
/// Element `i` is the result of [`predict_with_media`] for `requests[i]` with
/// the shared options.
pub fn predict_batch_with_media(
    source: &dyn ObservableEphemerisSource,
    requests: &[PredictRequest],
    options: MediaPredictOptions<'_>,
) -> Vec<Result<MediaPredictedObservables, ObservablesError>> {
    requests
        .iter()
        .map(|&(sat, receiver_ecef_m, t_rx_j2000_s)| {
            predict_with_media(source, sat, receiver_ecef_m, t_rx_j2000_s, options)
        })
        .collect()
}

/// Predict observables for many `(satellite, receiver, epoch)` requests, fanning
/// the independent requests across a rayon thread pool.
///
/// Each request is evaluated by the same scalar [`predict`] kernel and the
/// indexed parallel collect preserves input order, so element `i` is
/// byte-for-byte identical to element `i` of [`predict_batch`]: the requests
/// share no mutable state and a single `predict` is internally sequential, so
/// throughput scales with cores while every value stays bit-exact. The
/// `source` must be `Sync` because it is read concurrently from every worker.
pub fn predict_batch_parallel(
    source: &(dyn ObservableEphemerisSource + Sync),
    requests: &[PredictRequest],
    options: PredictOptions,
) -> Vec<Result<PredictedObservables, ObservablesError>> {
    #[cfg(feature = "parallel")]
    let requests = requests.par_iter();
    #[cfg(not(feature = "parallel"))]
    let requests = requests.iter();
    requests
        .map(|&(sat, receiver_ecef_m, t_rx_j2000_s)| {
            predict(source, sat, receiver_ecef_m, t_rx_j2000_s, options)
        })
        .collect()
}

/// Predict media-corrected observables for many requests in parallel.
///
/// Each worker evaluates the same scalar [`predict_with_media`] path and the
/// indexed parallel collect preserves request order.
pub fn predict_batch_with_media_parallel(
    source: &(dyn ObservableEphemerisSource + Sync),
    requests: &[PredictRequest],
    options: MediaPredictOptions<'_>,
) -> Vec<Result<MediaPredictedObservables, ObservablesError>> {
    #[cfg(feature = "parallel")]
    let requests = requests.par_iter();
    #[cfg(not(feature = "parallel"))]
    let requests = requests.iter();
    requests
        .map(|&(sat, receiver_ecef_m, t_rx_j2000_s)| {
            predict_with_media(source, sat, receiver_ecef_m, t_rx_j2000_s, options)
        })
        .collect()
}

/// One batch range-prediction request: the satellite, the static receiver ECEF
/// position in meters, and the receive epoch in seconds since J2000.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RangePredictionRequest {
    /// The satellite to range against.
    pub sat: GnssSatelliteId,
    /// Static receiver ECEF position, meters.
    pub receiver_ecef_m: [f64; 3],
    /// Receive epoch, seconds since J2000.
    pub t_rx_j2000_s: f64,
}

impl RangePredictionRequest {
    /// Build a range request from its satellite, receiver position, and epoch.
    #[must_use]
    pub const fn new(sat: GnssSatelliteId, receiver_ecef_m: [f64; 3], t_rx_j2000_s: f64) -> Self {
        Self {
            sat,
            receiver_ecef_m,
            t_rx_j2000_s,
        }
    }
}

/// The geometry-only result of one [`predict_ranges`] request.
///
/// A projection of [`transmit_time_satellite_state`]: the transmit-time geometry
/// a range-only consumer needs, without the Doppler / topocentric fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangePrediction {
    /// Geometric range after optional Sagnac transport, meters.
    pub geometric_range_m: f64,
    /// Satellite clock offset at transmit time, seconds (`None` if absent).
    pub sat_clock_s: Option<f64>,
    /// Transmit time as seconds since J2000.
    pub transmit_time_j2000_s: f64,
    /// Sagnac-transported satellite ECEF position, meters.
    pub sat_pos_ecef_m: [f64; 3],
}

/// Range-only prediction with an additional media-corrected range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaRangePrediction {
    /// Geometry-only range prediction.
    pub prediction: RangePrediction,
    /// Range after adding the selected media delays, meters.
    pub range_m: f64,
    /// Media delays applied to `range_m`.
    pub media: AppliedMediaCorrections,
}

/// Predict geometric ranges for many `(satellite, receiver, epoch)` requests in
/// one call, writing into a caller-provided `out` slice.
///
/// `out[i]` is filled from `requests[i]` by the range-only transmit-time kernel
/// `range_prediction_at_rx`: the same light-time iteration and Sagnac transport
/// as [`transmit_time_satellite_state`], projected to the range geometry. It is
/// therefore bit-identical to calling that predictor in a loop and reading its
/// geometry fields, and the whole batch is one native call over the array (no
/// per-request host-language dispatch).
///
/// Internally this is the vectorized hot path: it drops the finite-difference
/// **velocity** evaluation that [`transmit_time_satellite_state`] performs and
/// that a range consumer never uses, cutting the per-request ephemeris
/// evaluations by a third (from 6 to 4), and writes each result in a single pass
/// over `out`. The range values are unchanged to the bit, because the velocity
/// term never entered a [`RangePrediction`]; only the discarded work is removed.
/// `options.carrier_hz` is unused (ranges carry no Doppler);
/// `options.light_time` / `options.sagnac` are honored.
///
/// Errors:
/// - [`ObservablesError::InvalidInput`] with field `out` if `out.len()` differs
///   from `requests.len()`.
/// - The first request error (invalid input or missing ephemeris) aborts the
///   batch and is returned; `out` is then partially written.
pub fn predict_ranges(
    source: &dyn ObservableEphemerisSource,
    requests: &[RangePredictionRequest],
    options: PredictOptions,
    out: &mut [RangePrediction],
) -> Result<(), ObservablesError> {
    if out.len() != requests.len() {
        return Err(ObservablesError::InvalidInput {
            field: "out",
            kind: ObservablesInputErrorKind::OutOfRange,
        });
    }
    for (request, slot) in requests.iter().zip(out.iter_mut()) {
        *slot = range_prediction_at_rx(
            source,
            request.sat,
            request.receiver_ecef_m,
            request.t_rx_j2000_s,
            options,
        )?;
    }
    Ok(())
}

/// Predict media-corrected ranges for many requests.
///
/// `out[i].prediction` is the same geometry-only value produced by
/// [`predict_ranges`] for `requests[i]`. `out[i].range_m` adds the selected
/// troposphere and ionosphere delays.
pub fn predict_ranges_with_media(
    source: &dyn ObservableEphemerisSource,
    requests: &[RangePredictionRequest],
    options: MediaPredictOptions<'_>,
    out: &mut [MediaRangePrediction],
) -> Result<(), ObservablesError> {
    if out.len() != requests.len() {
        return Err(ObservablesError::InvalidInput {
            field: "out",
            kind: ObservablesInputErrorKind::OutOfRange,
        });
    }
    for (request, slot) in requests.iter().zip(out.iter_mut()) {
        if options.media.is_disabled() {
            let prediction = range_prediction_at_rx(
                source,
                request.sat,
                request.receiver_ecef_m,
                request.t_rx_j2000_s,
                options.prediction,
            )?;
            *slot = MediaRangePrediction {
                range_m: prediction.geometric_range_m,
                prediction,
                media: AppliedMediaCorrections::default(),
            };
            continue;
        }
        let (prediction, topocentric) = range_prediction_core(
            source,
            request.sat,
            request.receiver_ecef_m,
            request.t_rx_j2000_s,
            options.prediction,
        )?;
        let media = observable_media_corrections(
            topocentric.receiver,
            topocentric.elevation_rad,
            topocentric.azimuth_rad,
            request.t_rx_j2000_s,
            options.prediction.carrier_hz,
            options.media,
        )?;
        let range_m = prediction.geometric_range_m + media.total_m;
        validate::finite(range_m, "range_m").map_err(map_input_error)?;
        *slot = MediaRangePrediction {
            prediction,
            range_m,
            media,
        };
    }
    Ok(())
}

/// Range-only transmit-time kernel: iterate light time / Sagnac to the geometric
/// range at receive epoch `t_rx_j2000_s` and project just the [`RangePrediction`]
/// geometry.
///
/// This is [`transmit_time_satellite_state`] with the finite-difference velocity
/// (and its two extra ephemeris evaluations) removed, since a range prediction
/// never carries velocity. Every returned field is bit-identical to the
/// corresponding field of `transmit_time_satellite_state` for the same inputs.
fn range_prediction_at_rx(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<RangePrediction, ObservablesError> {
    let (prediction, _, _) =
        range_prediction_state(source, sat, receiver_ecef_m, t_rx_j2000_s, options)?;
    Ok(prediction)
}

fn range_prediction_core(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<(RangePrediction, TopocentricGeometry), ObservablesError> {
    let (prediction, line_of_sight_m, range) =
        range_prediction_state(source, sat, receiver_ecef_m, t_rx_j2000_s, options)?;
    let topocentric = topocentric(receiver_ecef_m, line_of_sight_m, range)?;
    Ok((prediction, topocentric))
}

fn range_prediction_state(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<(RangePrediction, [f64; 3], f64), ObservablesError> {
    validate_transmit_time_inputs(receiver_ecef_m, t_rx_j2000_s)?;
    let solved = solve_transmit_time(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
        TransmitEpochRounding::Exact,
    )?;
    let dx = solved.sat_rot_ecef_m[0] - receiver_ecef_m[0];
    let dy = solved.sat_rot_ecef_m[1] - receiver_ecef_m[1];
    let dz = solved.sat_rot_ecef_m[2] - receiver_ecef_m[2];
    let line_of_sight_m = [dx, dy, dz];
    let range = geometric_range_m([dx, dy, dz])?;
    Ok((
        RangePrediction {
            geometric_range_m: range,
            sat_clock_s: solved.state.clock_s,
            transmit_time_j2000_s: solved.transmit_time_j2000_s,
            sat_pos_ecef_m: solved.sat_rot_ecef_m,
        },
        line_of_sight_m,
        range,
    ))
}

#[derive(Debug, Clone, Copy)]
struct SolvedTransmitTime {
    tau_s: f64,
    transmit_offset_us: i64,
    transmit_time_j2000_s: f64,
    state: ObservableState,
    sat_rot_ecef_m: [f64; 3],
}

/// How the geometric light-time solution turns a flight time into a transmission epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransmitEpochRounding {
    /// `t_rx - tau`, every bit of the flight time kept.
    Exact,
    /// `t_rx - round(tau · 10⁶) / 10⁶`: the flight time rounded to whole microseconds,
    /// as the external reference fixtures were computed. Only the repository's replay of
    /// those fixtures selects it.
    #[cfg_attr(not(feature = "test-replays"), allow(dead_code))]
    WholeMicrosecond,
}

fn solve_transmit_time(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
    rounding: TransmitEpochRounding,
) -> Result<SolvedTransmitTime, ObservablesError> {
    solve_transmit_time_from(
        source,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
        0.0,
        rounding,
    )
}

/// The light-time iteration started from `initial_tau_s` instead of zero, so its first
/// ephemeris query is at `t_rx - initial_tau_s` rather than at the reception time.
fn solve_transmit_time_from(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
    initial_tau_s: f64,
    rounding: TransmitEpochRounding,
) -> Result<SolvedTransmitTime, ObservablesError> {
    if !options.light_time {
        let state = validated_state_at_j2000_s(source, sat, t_rx_j2000_s)?;
        let sat_rot = sagnac_rotate(state.position_ecef_m, 0.0, options.sagnac);
        validate::finite_vec3(sat_rot, "satellite position_ecef_m").map_err(map_input_error)?;
        return Ok(SolvedTransmitTime {
            tau_s: 0.0,
            transmit_offset_us: 0,
            transmit_time_j2000_s: t_rx_j2000_s,
            state,
            sat_rot_ecef_m: sat_rot,
        });
    }

    let mut tau = initial_tau_s;
    for iter in 0..OBSERVABLE_TRANSMIT_TIME_ITERATIONS {
        let t_tx = light_time_transmit_epoch_j2000_s(t_rx_j2000_s, tau, rounding);
        let state = validated_state_at_j2000_s(source, sat, t_tx)?;
        let sat_rot = sagnac_rotate(state.position_ecef_m, tau, options.sagnac);
        validate::finite_vec3(sat_rot, "satellite position_ecef_m").map_err(map_input_error)?;
        let dx = sat_rot[0] - receiver_ecef_m[0];
        let dy = sat_rot[1] - receiver_ecef_m[1];
        let dz = sat_rot[2] - receiver_ecef_m[2];
        let range = geometric_range_m([dx, dy, dz])?;
        let new_tau = range / C_M_S;

        if iter + 1 == OBSERVABLE_TRANSMIT_TIME_ITERATIONS {
            return finalize_transmit_time(
                source,
                sat,
                t_rx_j2000_s,
                new_tau,
                options.sagnac,
                rounding,
            );
        }

        tau = new_tau;
    }

    unreachable!("fixed transmit-time loop always returns on its last iteration")
}

fn finalize_transmit_time(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_rx_j2000_s: f64,
    tau: f64,
    sagnac: bool,
    rounding: TransmitEpochRounding,
) -> Result<SolvedTransmitTime, ObservablesError> {
    let t_tx = light_time_transmit_epoch_j2000_s(t_rx_j2000_s, tau, rounding);
    validate::finite(t_tx, "transmit_time_j2000_s").map_err(map_input_error)?;
    let state = validated_state_at_j2000_s(source, sat, t_tx)?;
    let sat_rot = sagnac_rotate(state.position_ecef_m, tau, sagnac);
    validate::finite_vec3(sat_rot, "satellite position_ecef_m").map_err(map_input_error)?;
    Ok(SolvedTransmitTime {
        tau_s: tau,
        transmit_offset_us: microseconds_from_tau(tau),
        transmit_time_j2000_s: t_tx,
        state,
        sat_rot_ecef_m: sat_rot,
    })
}

/// Transmission epoch of the geometric light-time solution for flight time `tau`.
fn light_time_transmit_epoch_j2000_s(
    t_rx_j2000_s: f64,
    tau: f64,
    rounding: TransmitEpochRounding,
) -> f64 {
    match rounding {
        TransmitEpochRounding::Exact => t_rx_j2000_s - tau,
        TransmitEpochRounding::WholeMicrosecond => {
            t_rx_j2000_s - microseconds_from_tau(tau) as f64 / MICROSECONDS_PER_SECOND
        }
    }
}

/// A flight time in whole microseconds, for the reported
/// [`PredictedObservables::transmit_offset_us`].
fn microseconds_from_tau(tau_s: f64) -> i64 {
    (tau_s * MICROSECONDS_PER_SECOND).round() as i64
}

/// The geometric light-time predictors with the transmission epoch rounded to whole
/// microseconds, `t_rx - round(tau · 10⁶) / 10⁶`, as the external reference fixtures
/// (the Python application and formula oracles) computed it.
///
/// The public predictors keep every bit of the flight time; these exist only under the
/// test-only `test-replays` feature, which the crate's own tests enable, so they can
/// replay those fixtures bit for bit and check that the public predictors differ from
/// them by the rounding of the transmission epoch alone.
#[cfg(feature = "test-replays")]
#[doc(hidden)]
pub mod rounded_microsecond_replay {
    use super::*;

    /// [`super::predict`] with the transmission epoch rounded to whole microseconds.
    pub fn predict(
        source: &dyn ObservableEphemerisSource,
        sat: GnssSatelliteId,
        receiver_ecef_m: [f64; 3],
        t_rx_j2000_s: f64,
        options: PredictOptions,
    ) -> Result<PredictedObservables, ObservablesError> {
        let (prediction, _) = predict_core(
            source,
            sat,
            receiver_ecef_m,
            t_rx_j2000_s,
            options,
            TransmitEpochRounding::WholeMicrosecond,
        )?;
        Ok(prediction)
    }

    /// [`super::transmit_time_satellite_state`] with the transmission epoch rounded to
    /// whole microseconds.
    pub fn transmit_time_satellite_state(
        source: &dyn ObservableEphemerisSource,
        sat: GnssSatelliteId,
        receiver_ecef_m: [f64; 3],
        t_rx_j2000_s: f64,
        options: TransmitTimeOptions,
    ) -> Result<TransmitTimeSatelliteState, ObservablesError> {
        transmit_time_satellite_state_with(
            source,
            sat,
            receiver_ecef_m,
            t_rx_j2000_s,
            options,
            TransmitEpochRounding::WholeMicrosecond,
        )
    }
}

fn satellite_velocity(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_tx_j2000_s: f64,
) -> Result<[f64; 3], ObservablesError> {
    if let Some(velocity) = source.velocity_at_j2000_s(sat, t_tx_j2000_s) {
        let velocity = velocity?;
        validate::finite_vec3(velocity, "satellite velocity_m_s").map_err(map_input_error)?;
        return Ok(velocity);
    }
    let plus = validated_state_at_j2000_s(source, sat, t_tx_j2000_s + FD_HALF_S)?;
    let minus = validated_state_at_j2000_s(source, sat, t_tx_j2000_s - FD_HALF_S)?;
    let denom = 2.0 * FD_HALF_S;
    let velocity = [
        (plus.position_ecef_m[0] - minus.position_ecef_m[0]) / denom,
        (plus.position_ecef_m[1] - minus.position_ecef_m[1]) / denom,
        (plus.position_ecef_m[2] - minus.position_ecef_m[2]) / denom,
    ];
    validate::finite_vec3(velocity, "satellite velocity_m_s").map_err(map_input_error)
}

fn validate_predict_inputs(
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<(), ObservablesError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_input_error)?;
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    validate::finite_positive(options.carrier_hz, "options.carrier_hz").map_err(map_input_error)?;
    Ok(())
}

fn validate_transmit_time_inputs(
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
) -> Result<(), ObservablesError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_input_error)?;
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    Ok(())
}

fn validate_emission_media_batch_options(
    options: EmissionMediaBatchOptions<'_>,
) -> Result<(), ObservablesError> {
    if options.media.needs_carrier() {
        validate::finite_positive(options.carrier_hz, "options.carrier_hz")
            .map_err(map_input_error)?;
    }
    if let Some(cutoff) = options.min_elevation_rad {
        validate::finite(cutoff, "options.min_elevation_rad").map_err(map_input_error)?;
        if !(0.0..=core::f64::consts::FRAC_PI_2).contains(&cutoff) {
            return Err(invalid_observable_input(
                "options.min_elevation_rad",
                ObservablesInputErrorKind::OutOfRange,
            ));
        }
    }
    Ok(())
}

/// The state at `t_j2000_s` of the record `source` selects at `selection_j2000_s`,
/// validated.
fn validated_selected_state(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
    selection_j2000_s: f64,
) -> Result<ObservableState, ObservablesError> {
    let state = source
        .try_observable_state_group_delay_selected_at_j2000_s(sat, t_j2000_s, selection_j2000_s)?
        .value
        .0;
    validate_observable_state(&state)?;
    Ok(state)
}

/// [`satellite_velocity`] of the record `source` selects at `selection_j2000_s`.
fn satellite_velocity_selected(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_tx_j2000_s: f64,
    selection_j2000_s: f64,
) -> Result<[f64; 3], ObservablesError> {
    if let Some(velocity) =
        source.velocity_selected_at_j2000_s(sat, t_tx_j2000_s, selection_j2000_s)
    {
        let velocity = velocity?;
        validate::finite_vec3(velocity, "satellite velocity_m_s").map_err(map_input_error)?;
        return Ok(velocity);
    }
    let plus = validated_selected_state(source, sat, t_tx_j2000_s + FD_HALF_S, selection_j2000_s)?;
    let minus = validated_selected_state(source, sat, t_tx_j2000_s - FD_HALF_S, selection_j2000_s)?;
    let denom = 2.0 * FD_HALF_S;
    let velocity = [
        (plus.position_ecef_m[0] - minus.position_ecef_m[0]) / denom,
        (plus.position_ecef_m[1] - minus.position_ecef_m[1]) / denom,
        (plus.position_ecef_m[2] - minus.position_ecef_m[2]) / denom,
    ];
    validate::finite_vec3(velocity, "satellite velocity_m_s").map_err(map_input_error)
}

fn validated_state_at_j2000_s(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
) -> Result<ObservableState, ObservablesError> {
    let state = source.observable_state_at_j2000_s(sat, t_j2000_s)?;
    validate_observable_state(&state)?;
    Ok(state)
}

fn validate_observable_state(state: &ObservableState) -> Result<(), ObservablesError> {
    validate::finite_vec3(state.position_ecef_m, "observable state position_ecef_m")
        .map_err(map_input_error)?;
    if let Some(clock_s) = state.clock_s {
        validate::finite(clock_s, "observable state clock_s").map_err(map_input_error)?;
    }
    Ok(())
}

fn geometric_range_m(delta_ecef_m: [f64; 3]) -> Result<f64, ObservablesError> {
    let range = (delta_ecef_m[0] * delta_ecef_m[0]
        + delta_ecef_m[1] * delta_ecef_m[1]
        + delta_ecef_m[2] * delta_ecef_m[2])
        .sqrt();
    validate::finite_positive(range, "geometric_range_m").map_err(map_input_error)
}

fn map_input_error(error: validate::FieldError) -> ObservablesError {
    ObservablesError::InvalidInput {
        field: error.field(),
        kind: ObservablesInputErrorKind::from(&error),
    }
}

fn invalid_observable_input(
    field: &'static str,
    kind: ObservablesInputErrorKind,
) -> ObservablesError {
    ObservablesError::InvalidInput { field, kind }
}

fn media_instant(t_rx_j2000_s: f64) -> Result<Instant, ObservablesError> {
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    let days = (t_rx_j2000_s / SECONDS_PER_DAY).floor();
    let seconds_into_day = t_rx_j2000_s - days * SECONDS_PER_DAY;
    let fraction = seconds_into_day / SECONDS_PER_DAY;
    let split = JulianDateSplit::new(J2000_JD + days, fraction).map_err(|_| {
        invalid_observable_input("t_rx_j2000_s", ObservablesInputErrorKind::OutOfRange)
    })?;
    Ok(Instant::from_julian_date(TimeScale::Gpst, split))
}

fn rounded_j2000_seconds(t_rx_j2000_s: f64) -> Result<i64, ObservablesError> {
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    let rounded = t_rx_j2000_s.round();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(invalid_observable_input(
            "t_rx_j2000_s",
            ObservablesInputErrorKind::OutOfRange,
        ));
    }
    Ok(rounded as i64)
}

fn map_media_error(error: Error) -> ObservablesError {
    match error {
        Error::InvalidInput(message) => map_media_invalid_input(&message),
        Error::IonexOutOfCoverage(_)
        | Error::IonexNodesNotAvailable(_)
        | Error::IonexSlantUnavailable(_) => ObservablesError::Media(error),
        _ => invalid_observable_input("media", ObservablesInputErrorKind::OutOfRange),
    }
}

fn map_media_invalid_input(message: &str) -> ObservablesError {
    let kind = if message.ends_with("not finite") {
        ObservablesInputErrorKind::NonFinite
    } else if message.ends_with("not positive") {
        ObservablesInputErrorKind::NotPositive
    } else if message.ends_with("negative") {
        ObservablesInputErrorKind::Negative
    } else {
        ObservablesInputErrorKind::OutOfRange
    };
    let field = if message.starts_with("elevation_rad ") {
        "media.elevation_rad"
    } else if message.starts_with("receiver.lat_rad ") {
        "media.receiver.lat_rad"
    } else if message.starts_with("receiver.lon_rad ") {
        "media.receiver.lon_rad"
    } else if message.starts_with("receiver.height_m ") {
        "media.receiver.height_m"
    } else if message.starts_with("pressure_hpa ") {
        "media.pressure_hpa"
    } else if message.starts_with("temperature_k ") {
        "media.temperature_k"
    } else if message.starts_with("relative_humidity ") {
        "media.relative_humidity"
    } else if message.starts_with("frequency_hz ") {
        "media.carrier_hz"
    } else if message.starts_with("azimuth_rad ") {
        "media.azimuth_rad"
    } else {
        "media"
    };
    invalid_observable_input(field, kind)
}

fn sagnac_rotate(pos: [f64; 3], tau_s: f64, apply: bool) -> [f64; 3] {
    let sagnac = if apply {
        SagnacRecipe::ClosedFormZRotation
    } else {
        SagnacRecipe::Off
    };
    crate::estimation::substrate::range::rotate_transmit_satellite(
        sagnac,
        pos,
        tau_s,
        OMEGA_E_DOT_RAD_S,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TopocentricReceiver {
    ecef_m: [f64; 3],
    geodetic: Wgs84Geodetic,
    sin_lat: f64,
    cos_lat: f64,
    sin_lon: f64,
    cos_lon: f64,
}

#[derive(Debug, Clone, Copy)]
enum TopocentricReceiverSource<'a> {
    Cached(&'a TopocentricReceiver),
    Ecef([f64; 3]),
}

impl TopocentricReceiverSource<'_> {
    fn ecef_m(self) -> [f64; 3] {
        match self {
            Self::Cached(receiver) => receiver.ecef_m,
            Self::Ecef(receiver_ecef_m) => receiver_ecef_m,
        }
    }

    fn topocentric(
        self,
        delta_ecef_m: [f64; 3],
        range_m: f64,
    ) -> Result<TopocentricGeometry, ObservablesError> {
        match self {
            Self::Cached(receiver) => topocentric_with_receiver(receiver, delta_ecef_m, range_m),
            Self::Ecef(receiver_ecef_m) => topocentric(receiver_ecef_m, delta_ecef_m, range_m),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TopocentricGeometry {
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    elevation_deg: f64,
    azimuth_deg: f64,
}

fn topocentric(
    receiver_ecef_m: [f64; 3],
    delta_ecef_m: [f64; 3],
    range_m: f64,
) -> Result<TopocentricGeometry, ObservablesError> {
    let receiver = topocentric_receiver(receiver_ecef_m)?;
    topocentric_with_receiver(&receiver, delta_ecef_m, range_m)
}

fn topocentric_receiver(
    receiver_ecef_m: [f64; 3],
) -> Result<TopocentricReceiver, ObservablesError> {
    let (lat_deg, lon_deg, height_km) = itrs_to_geodetic_compute(
        receiver_ecef_m[0] / KM_TO_M,
        receiver_ecef_m[1] / KM_TO_M,
        receiver_ecef_m[2] / KM_TO_M,
    )
    .map_err(|_| ObservablesError::InvalidInput {
        field: "receiver_ecef_m",
        kind: ObservablesInputErrorKind::OutOfRange,
    })?;
    // The application oracle pins this multiply-then-divide order.
    let lat = lat_deg * PI / DEGREES_PER_SEMICIRCLE;
    let lon = lon_deg * PI / DEGREES_PER_SEMICIRCLE;
    let receiver = Wgs84Geodetic::new(lat, lon, height_km * KM_TO_M).map_err(|_| {
        ObservablesError::InvalidInput {
            field: "receiver_ecef_m",
            kind: ObservablesInputErrorKind::OutOfRange,
        }
    })?;

    Ok(TopocentricReceiver {
        ecef_m: receiver_ecef_m,
        geodetic: receiver,
        sin_lat: libm::sin(lat),
        cos_lat: libm::cos(lat),
        sin_lon: libm::sin(lon),
        cos_lon: libm::cos(lon),
    })
}

fn topocentric_with_receiver(
    receiver: &TopocentricReceiver,
    delta_ecef_m: [f64; 3],
    range_m: f64,
) -> Result<TopocentricGeometry, ObservablesError> {
    let dx = delta_ecef_m[0];
    let dy = delta_ecef_m[1];
    let dz = delta_ecef_m[2];

    let e = -receiver.sin_lon * dx + receiver.cos_lon * dy;
    let n = -receiver.sin_lat * receiver.cos_lon * dx - receiver.sin_lat * receiver.sin_lon * dy
        + receiver.cos_lat * dz;
    let u = receiver.cos_lat * receiver.cos_lon * dx
        + receiver.cos_lat * receiver.sin_lon * dy
        + receiver.sin_lat * dz;

    // Near the zenith the horizontal projection (e, n) is pure rounding noise,
    // so azimuth is degenerate and defined to be 0.0 (RTKLIB satazel semantics).
    // Outside that threshold the multiply-then-divide order is pinned by the
    // application oracle.
    let horiz_sq = e * e + n * n;
    let (azimuth_rad, mut azimuth_deg) = if horiz_sq < AZIMUTH_ZENITH_EPS * range_m * range_m {
        (0.0, 0.0)
    } else {
        let raw_azimuth_rad = libm::atan2(e, n);
        (
            if raw_azimuth_rad < 0.0 {
                raw_azimuth_rad + 2.0 * PI
            } else {
                raw_azimuth_rad
            },
            raw_azimuth_rad * DEGREES_PER_SEMICIRCLE / PI,
        )
    };
    if azimuth_deg < 0.0 {
        azimuth_deg += DEGREES_PER_CIRCLE;
    }
    // range_m is an ECEF norm, so at an exact zenith u/range_m can round just
    // past 1.0 and make asin return NaN. Clamp to the valid asin domain; this
    // only touches the previously-NaN boundary and leaves in-range values bit
    // identical.
    let sin_elevation = (u / range_m).clamp(-1.0, 1.0);
    let elevation_rad = libm::asin(sin_elevation);
    let elevation_deg = elevation_rad * DEGREES_PER_SEMICIRCLE / PI;

    validate::finite(elevation_rad, "elevation_rad").map_err(map_input_error)?;
    validate::finite(elevation_deg, "elevation_deg").map_err(map_input_error)?;
    validate::finite(azimuth_rad, "azimuth_rad").map_err(map_input_error)?;
    validate::finite(azimuth_deg, "azimuth_deg").map_err(map_input_error)?;
    Ok(TopocentricGeometry {
        receiver: receiver.geodetic,
        elevation_rad,
        azimuth_rad,
        elevation_deg,
        azimuth_deg,
    })
}

#[cfg(test)]
mod public_api_tests {
    use super::*;
    use crate::{GnssSatelliteId, GnssSystem};

    #[derive(Debug, Clone, Copy)]
    struct StaticSource {
        state: ObservableState,
    }

    impl ObservableEphemerisSource for StaticSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(self.state)
        }
    }

    #[test]
    fn predict_ranges_matches_transmit_time_loop_bitwise() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let options = PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        };
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid satellite id");
        let requests = [
            RangePredictionRequest {
                sat: sat1,
                receiver_ecef_m: [4_027_894.0, 307_046.0, 4_919_474.0],
                t_rx_j2000_s: 646_272_000.0,
            },
            RangePredictionRequest {
                sat: sat2,
                receiver_ecef_m: [1_130_000.0, -4_830_000.0, 3_994_000.0],
                t_rx_j2000_s: 646_272_060.0,
            },
        ];
        let mut out = [RangePrediction {
            geometric_range_m: 0.0,
            sat_clock_s: None,
            transmit_time_j2000_s: 0.0,
            sat_pos_ecef_m: [0.0; 3],
        }; 2];
        predict_ranges(&source, &requests, options, &mut out).expect("batch range prediction");

        let tt_options = TransmitTimeOptions {
            light_time: options.light_time,
            sagnac: options.sagnac,
        };
        for (request, got) in requests.iter().zip(out.iter()) {
            let single = transmit_time_satellite_state(
                &source,
                request.sat,
                request.receiver_ecef_m,
                request.t_rx_j2000_s,
                tt_options,
            )
            .expect("single transmit-time state");
            assert_eq!(
                got.geometric_range_m.to_bits(),
                single.geometric_range_m.to_bits()
            );
            assert_eq!(
                got.transmit_time_j2000_s.to_bits(),
                single.transmit_time_j2000_s.to_bits()
            );
            assert_eq!(
                got.sat_clock_s.map(f64::to_bits),
                single.clock_s.map(f64::to_bits)
            );
            assert_eq!(
                got.sat_pos_ecef_m.map(f64::to_bits),
                single.position_ecef_m.map(f64::to_bits)
            );
        }
    }

    #[test]
    fn predict_ranges_batch_matches_scalar_calls_bitwise() {
        // Item 3: the vectorized batch kernel must be byte-identical to solving
        // each request in its own one-element call (no cross-request state).
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let options = PredictOptions::default();
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid satellite id");
        let requests = [
            RangePredictionRequest {
                sat: sat1,
                receiver_ecef_m: [4_027_894.0, 307_046.0, 4_919_474.0],
                t_rx_j2000_s: 646_272_000.0,
            },
            RangePredictionRequest {
                sat: sat2,
                receiver_ecef_m: [1_130_000.0, -4_830_000.0, 3_994_000.0],
                t_rx_j2000_s: 646_272_060.0,
            },
            RangePredictionRequest {
                sat: sat1,
                receiver_ecef_m: [-2_700_000.0, -4_290_000.0, 3_855_000.0],
                t_rx_j2000_s: 646_272_120.0,
            },
        ];
        let zero = RangePrediction {
            geometric_range_m: 0.0,
            sat_clock_s: None,
            transmit_time_j2000_s: 0.0,
            sat_pos_ecef_m: [0.0; 3],
        };

        let mut batch = [zero; 3];
        predict_ranges(&source, &requests, options, &mut batch).expect("batch ranges");

        for (i, request) in requests.iter().enumerate() {
            let mut single = [zero; 1];
            predict_ranges(&source, std::slice::from_ref(request), options, &mut single)
                .expect("single range");
            assert_eq!(
                batch[i].geometric_range_m.to_bits(),
                single[0].geometric_range_m.to_bits()
            );
            assert_eq!(
                batch[i].transmit_time_j2000_s.to_bits(),
                single[0].transmit_time_j2000_s.to_bits()
            );
            assert_eq!(
                batch[i].sat_clock_s.map(f64::to_bits),
                single[0].sat_clock_s.map(f64::to_bits)
            );
            assert_eq!(
                batch[i].sat_pos_ecef_m.map(f64::to_bits),
                single[0].sat_pos_ecef_m.map(f64::to_bits)
            );
        }
    }

    #[test]
    fn predict_ranges_rejects_length_mismatch() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: None,
            },
        };
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let requests = [RangePredictionRequest {
            sat,
            receiver_ecef_m: [4_027_894.0, 307_046.0, 4_919_474.0],
            t_rx_j2000_s: 646_272_000.0,
        }];
        let mut out: [RangePrediction; 0] = [];
        let err = predict_ranges(&source, &requests, PredictOptions::default(), &mut out)
            .expect_err("length mismatch must fail");
        match err {
            ObservablesError::InvalidInput { field, kind } => {
                assert_eq!(field, "out");
                assert_eq!(kind, ObservablesInputErrorKind::OutOfRange);
            }
            other => panic!("expected InvalidInput(out, OutOfRange), got {other:?}"),
        }
    }

    #[test]
    fn topocentric_elevation_is_ninety_at_non_equatorial_zenith() {
        // A ~45 deg receiver (non-equatorial) with the satellite placed exactly
        // along the receiver's recovered geodetic up, so the east/north
        // projection is zero and the asin argument u/range lands on the domain
        // boundary. For this receiver the rounding pushes u/range to just past
        // 1.0 (1.0000000000000002): without the clamp asin returns NaN and the
        // finite check turns it into an InvalidInput error. The clamp must yield
        // a finite ~90 deg elevation instead. This exact receiver was found by
        // scanning ECEF positions for the >1.0 rounding case.
        let rx = [
            4_509_179.095_483_66,
            275_556.225_682_215_9,
            4_487_348.408_865_919,
        ];
        let (lat_deg, lon_deg, _h) =
            itrs_to_geodetic_compute(rx[0] / KM_TO_M, rx[1] / KM_TO_M, rx[2] / KM_TO_M)
                .expect("receiver geodetic conversion");
        assert!(lat_deg.abs() > 40.0, "receiver must be non-equatorial");

        // Reconstruct the up unit vector exactly as `topocentric` does, then put
        // the satellite straight overhead along it.
        let lat = lat_deg * PI / DEGREES_PER_SEMICIRCLE;
        let lon = lon_deg * PI / DEGREES_PER_SEMICIRCLE;
        let (sl, cl) = (libm::sin(lat), libm::cos(lat));
        let (so, co) = (libm::sin(lon), libm::cos(lon));
        let up = [cl * co, cl * so, sl];

        let d = 20_000_000.0_f64;
        let delta = [up[0] * d, up[1] * d, up[2] * d];
        let range = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        // Guard the regression premise: this geometry really does overshoot the
        // asin domain (the bug this test pins).
        let u = cl * co * delta[0] + cl * so * delta[1] + sl * delta[2];
        assert!(
            u / range > 1.0,
            "test geometry must overshoot the asin domain"
        );

        let geometry = topocentric(rx, delta, range).expect("non-equatorial zenith must not error");
        assert!(geometry.elevation_deg.is_finite());
        assert!((geometry.elevation_deg - 90.0).abs() < 1e-9);
    }

    /// The geometric light time keeps every bit of the flight time: the transmission
    /// epoch is the reception epoch less the flight time, not less a whole number of
    /// microseconds.
    #[test]
    fn geometric_light_time_does_not_round_the_transmission_epoch() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let rx = [4_027_894.0, 307_046.0, 4_919_474.0];
        let t_rx = 646_272_000.0;
        let state =
            transmit_time_satellite_state(&source, sat, rx, t_rx, TransmitTimeOptions::default())
                .expect("state");
        assert_eq!(
            state.transmit_time_j2000_s.to_bits(),
            (t_rx - state.signal_flight_time_s).to_bits()
        );
        assert_eq!(
            state.transmit_offset_us,
            (state.signal_flight_time_s * 1.0e6).round() as i64
        );
        let prediction =
            predict(&source, sat, rx, t_rx, PredictOptions::default()).expect("prediction");
        assert_eq!(
            prediction.transmit_time_j2000_s.to_bits(),
            state.transmit_time_j2000_s.to_bits()
        );
    }

    /// A pseudorange's transmission epoch is RTKLIB `satposs`'s,
    /// `(t_rx - P / c) - dts(t_rx - P / c)`, and the state there is ranged with RTKLIB
    /// `geodist`: the unrotated position, the Euclidean range plus
    /// `ω (x_s y_r - y_s x_r) / c`, and the line of sight of the unrotated vector.
    #[test]
    fn pseudorange_placement_is_satposs_and_geodist() {
        let clock_s = 1.25e-6;
        let position = [20_200_000.0, 14_000_000.0, 21_700_000.0];
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: position,
                clock_s: Some(clock_s),
            },
        };
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let rx = [4_027_894.0, 307_046.0, 4_919_474.0];
        let t_rx = 646_272_000.0;
        let pseudorange_m = 21_345_678.912_345;

        let t_tx = pseudorange_transmit_epoch_j2000_s(&source, sat, t_rx, pseudorange_m)
            .expect("placed epoch");
        let satposs = (t_rx + (-pseudorange_m / C_M_S)) + (-clock_s);
        assert_eq!(t_tx.to_bits(), satposs.to_bits());
        assert_eq!(
            t_tx.to_bits(),
            pseudorange_transmit_epoch_from_clock_j2000_s(t_rx, pseudorange_m, clock_s).to_bits()
        );

        for sagnac in [true, false] {
            let geometry = pseudorange_transmit_geometry(&source, sat, rx, t_rx, t_tx, sagnac)
                .expect("placed geometry");
            let d = [
                position[0] - rx[0],
                position[1] - rx[1],
                position[2] - rx[2],
            ];
            let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let range = if sagnac {
                distance + OMEGA_E_DOT_RAD_S * (position[0] * rx[1] - position[1] * rx[0]) / C_M_S
            } else {
                distance
            };
            assert_eq!(geometry.geometric_range_m.to_bits(), range.to_bits());
            assert_eq!(
                geometry.sat_pos_ecef_m.map(f64::to_bits),
                position.map(f64::to_bits)
            );
            assert_eq!(
                geometry.los_unit.map(f64::to_bits),
                [d[0] / distance, d[1] / distance, d[2] / distance].map(f64::to_bits)
            );
            assert_eq!(geometry.transmit_time_j2000_s.to_bits(), t_tx.to_bits());
            assert_eq!(
                geometry.signal_flight_time_s.to_bits(),
                (t_rx - t_tx).to_bits()
            );
            assert_eq!(geometry.sat_clock_s, Some(clock_s));

            let state = pseudorange_transmit_satellite_state(&source, sat, rx, t_rx, t_tx, sagnac)
                .expect("placed state");
            assert_eq!(state.geometric_range_m.to_bits(), range.to_bits());
            assert_eq!(
                state.position_ecef_m.map(f64::to_bits),
                position.map(f64::to_bits)
            );
            assert_eq!(
                state.transmit_position_ecef_m.map(f64::to_bits),
                position.map(f64::to_bits)
            );
            assert_eq!(state.velocity_m_s, [0.0; 3]);
        }
    }

    /// A pseudorange that is not a positive finite distance places no satellite, as
    /// RTKLIB reads a zero pseudorange as none, and a source without a clock at
    /// `t_rx - P / c` cannot place one.
    #[test]
    fn pseudorange_placement_needs_a_pseudorange_and_a_clock() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let with_clock = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(0.0),
            },
        };
        for (pseudorange_m, kind) in [
            (0.0, ObservablesInputErrorKind::NotPositive),
            (-1.0, ObservablesInputErrorKind::NotPositive),
            (f64::NAN, ObservablesInputErrorKind::NonFinite),
        ] {
            match pseudorange_transmit_epoch_j2000_s(&with_clock, sat, 0.0, pseudorange_m) {
                Err(ObservablesError::InvalidInput { field, kind: got }) => {
                    assert_eq!(field, "pseudorange_m");
                    assert_eq!(got, kind);
                }
                other => panic!("{pseudorange_m}: expected an invalid pseudorange, got {other:?}"),
            }
        }
        let without_clock = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: None,
            },
        };
        assert!(matches!(
            pseudorange_transmit_epoch_j2000_s(&without_clock, sat, 0.0, 21_000_000.0),
            Err(ObservablesError::NoEphemeris)
        ));
    }

    #[test]
    fn transmit_time_state_matches_predict_substrate_with_no_light_time() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let rx = [4_027_894.0, 307_046.0, 4_919_474.0];
        let state = transmit_time_satellite_state(
            &source,
            sat,
            rx,
            646_272_000.0,
            TransmitTimeOptions {
                light_time: false,
                sagnac: true,
            },
        )
        .expect("state");
        let prediction = predict(
            &source,
            sat,
            rx,
            646_272_000.0,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: false,
                sagnac: true,
            },
        )
        .expect("prediction");

        assert_eq!(state.signal_flight_time_s.to_bits(), 0.0f64.to_bits());
        assert_eq!(state.transmit_offset_us, 0);
        assert_eq!(
            state.transmit_time_j2000_s.to_bits(),
            646_272_000.0f64.to_bits()
        );
        assert_eq!(state.clock_s.unwrap().to_bits(), 1.25e-6f64.to_bits());
        assert_eq!(
            state.transmit_position_ecef_m.map(f64::to_bits),
            source.state.position_ecef_m.map(f64::to_bits)
        );
        assert_eq!(
            state.position_ecef_m.map(f64::to_bits),
            prediction.sat_pos_ecef_m.map(f64::to_bits)
        );
        assert_eq!(
            state.velocity_m_s.map(f64::to_bits),
            prediction.sat_velocity_m_s.map(f64::to_bits)
        );
        assert_eq!(
            state.geometric_range_m.to_bits(),
            prediction.geometric_range_m.to_bits()
        );
        assert_eq!(
            state.los_unit.map(f64::to_bits),
            prediction.los_unit.map(f64::to_bits)
        );
    }
}

#[cfg(test)]
mod media_validation_tests {
    //! Provenance: IERS TN36 media corrections. The troposphere and ionosphere
    //! model functions are validated in their own modules; these tests prove the
    //! observable media wrapper reuses those functions exactly and keeps the
    //! default-off prediction path bit-identical.

    use super::*;
    use crate::astro::time::civil::split_julian_date_from_j2000_seconds;
    use crate::ionex::{IonexHeader, IonexMappingFunction, TecGridSamples};
    use crate::GnssSystem;

    const T_RX_J2000_S: f64 = 646_272_000.0;
    const T_RX_J2000_I64: i64 = 646_272_000;

    #[derive(Debug, Clone, Copy)]
    struct StaticSource {
        state: ObservableState,
    }

    impl ObservableEphemerisSource for StaticSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(self.state)
        }
    }

    fn epoch() -> Instant {
        let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(T_RX_J2000_I64);
        Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit::new(jd_whole, fraction).expect("valid media epoch"),
        )
    }

    fn receiver() -> Wgs84Geodetic {
        Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid receiver")
    }

    fn met() -> Met {
        Met::new(1013.25, 288.15, 0.5).expect("valid met")
    }

    fn klobuchar_model() -> IonoModel {
        IonoModel::Klobuchar(crate::ionex::KlobucharParams {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        })
    }

    fn ionex() -> Ionex {
        let map = vec![vec![Some(12.0); 3]; 3];
        Ionex::from_samples(TecGridSamples {
            map_epochs: vec![epoch()],
            lat_nodes_deg: vec![90.0, 0.0, -90.0],
            lon_nodes_deg: vec![-180.0, 0.0, 180.0],
            dlat_deg: -90.0,
            dlon_deg: 180.0,
            shell_height_km: 450.0,
            base_radius_km: 6371.0,
            exponent: 0,
            tec_maps: vec![map],
            rms_maps: Vec::new(),
            height_maps: Vec::new(),
            header: IonexHeader::new(IonexMappingFunction::CosZ),
        })
        .expect("valid IONEX samples")
    }

    fn direct_troposphere(elevation_rad: f64) -> f64 {
        let zenith =
            tropo_zenith(TropoModel::Saastamoinen, receiver(), met()).expect("zenith delay");
        let mapping = tropo_mapping(MappingModel::Niell, elevation_rad, receiver(), epoch())
            .expect("mapping");
        zenith.dry_m * mapping.dry + zenith.wet_m * mapping.wet
    }

    fn assert_bits_eq(label: &str, got: f64, expected: f64) {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "{label}: got {got:?}, expected {expected:?}"
        );
    }

    fn assert_prediction_bits_eq(got: &PredictedObservables, expected: &PredictedObservables) {
        assert_bits_eq(
            "geometric range",
            got.geometric_range_m,
            expected.geometric_range_m,
        );
        assert_bits_eq("range-rate", got.range_rate_m_s, expected.range_rate_m_s);
        assert_bits_eq("Doppler", got.doppler_hz, expected.doppler_hz);
        assert_eq!(
            got.sat_clock_s.map(f64::to_bits),
            expected.sat_clock_s.map(f64::to_bits)
        );
        assert_bits_eq("elevation", got.elevation_deg, expected.elevation_deg);
        assert_bits_eq("azimuth", got.azimuth_deg, expected.azimuth_deg);
        assert_eq!(got.transmit_offset_us, expected.transmit_offset_us);
        assert_bits_eq(
            "transmit time",
            got.transmit_time_j2000_s,
            expected.transmit_time_j2000_s,
        );
        for k in 0..3 {
            assert_bits_eq("los", got.los_unit[k], expected.los_unit[k]);
            assert_bits_eq(
                "satellite position",
                got.sat_pos_ecef_m[k],
                expected.sat_pos_ecef_m[k],
            );
            assert_bits_eq(
                "satellite velocity",
                got.sat_velocity_m_s[k],
                expected.sat_velocity_m_s[k],
            );
        }
    }

    fn assert_range_prediction_bits_eq(got: &RangePrediction, expected: &RangePrediction) {
        assert_bits_eq(
            "range geometric",
            got.geometric_range_m,
            expected.geometric_range_m,
        );
        assert_eq!(
            got.sat_clock_s.map(f64::to_bits),
            expected.sat_clock_s.map(f64::to_bits)
        );
        assert_bits_eq(
            "range transmit time",
            got.transmit_time_j2000_s,
            expected.transmit_time_j2000_s,
        );
        for k in 0..3 {
            assert_bits_eq(
                "range satellite position",
                got.sat_pos_ecef_m[k],
                expected.sat_pos_ecef_m[k],
            );
        }
    }

    #[test]
    fn media_corrections_match_direct_tropo_and_klobuchar_bits() {
        for elevation_deg in [5.0_f64, 15.0, 90.0] {
            let elevation_rad = elevation_deg * PI / DEGREES_PER_SEMICIRCLE;
            let azimuth_rad = 0.25;
            let options = ObservableMediaOptions {
                troposphere: Some(ObservableTroposphereCorrection {
                    met: met(),
                    mapping: MappingModel::Niell,
                }),
                ionosphere: Some(ObservableIonosphereCorrection::Broadcast(klobuchar_model())),
            };
            let got = observable_media_corrections(
                receiver(),
                elevation_rad,
                azimuth_rad,
                T_RX_J2000_S,
                F_L1_HZ,
                options,
            )
            .expect("media corrections");
            let expected_tropo = direct_troposphere(elevation_rad);
            let expected_iono = ionosphere_delay(
                receiver(),
                elevation_rad,
                azimuth_rad,
                epoch(),
                F_L1_HZ,
                &klobuchar_model(),
            )
            .expect("direct Klobuchar");

            assert_bits_eq("troposphere", got.troposphere_m, expected_tropo);
            assert_bits_eq("Klobuchar", got.ionosphere_m, expected_iono);
            assert_bits_eq("total", got.total_m, expected_tropo + expected_iono);
        }
    }

    #[test]
    fn media_corrections_match_direct_ionex_bits() {
        let ionex = ionex();
        for elevation_deg in [5.0_f64, 15.0, 90.0] {
            let elevation_rad = elevation_deg * PI / DEGREES_PER_SEMICIRCLE;
            let azimuth_rad = 1.0;
            let got = observable_media_corrections(
                receiver(),
                elevation_rad,
                azimuth_rad,
                T_RX_J2000_S,
                F_L1_HZ,
                ObservableMediaOptions {
                    troposphere: None,
                    ionosphere: Some(ObservableIonosphereCorrection::Ionex(&ionex)),
                },
            )
            .expect("IONEX media correction");
            let expected = ionex_slant_delay(
                &ionex,
                receiver(),
                elevation_rad,
                azimuth_rad,
                T_RX_J2000_I64,
                F_L1_HZ,
            )
            .expect("direct IONEX");

            assert_bits_eq("IONEX", got.ionosphere_m, expected);
            assert_bits_eq("IONEX total", got.total_m, expected);
        }
    }

    #[test]
    fn default_media_prediction_matches_predict_bits() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        let rx = [6_378_137.0, 0.0, 0.0];
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [26_378_137.0, 0.0, 0.0],
                clock_s: Some(0.0),
            },
        };
        let options = PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: false,
            sagnac: false,
        };
        let plain = predict(&source, sat, rx, T_RX_J2000_S, options).expect("plain predict");
        let media = predict_with_media(
            &source,
            sat,
            rx,
            T_RX_J2000_S,
            MediaPredictOptions {
                prediction: options,
                media: ObservableMediaOptions::default(),
            },
        )
        .expect("default media predict");

        assert_prediction_bits_eq(&media.prediction, &plain);
        assert_bits_eq("default range", media.range_m, plain.geometric_range_m);
        assert_eq!(media.media, AppliedMediaCorrections::default());
    }

    #[test]
    fn default_media_prediction_skips_media_epoch_for_large_epoch() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        let rx = [6_378_137.0, 0.0, 0.0];
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [26_378_137.0, 0.0, 0.0],
                clock_s: Some(0.0),
            },
        };
        let options = PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: false,
            sagnac: false,
        };
        let t_rx = 1.0e20;
        let plain = predict(&source, sat, rx, t_rx, options).expect("plain predict");
        let media = predict_with_media(
            &source,
            sat,
            rx,
            t_rx,
            MediaPredictOptions {
                prediction: options,
                media: ObservableMediaOptions::default(),
            },
        )
        .expect("default media predict");

        assert_prediction_bits_eq(&media.prediction, &plain);
        assert_bits_eq("default range", media.range_m, plain.geometric_range_m);
        assert_eq!(media.media, AppliedMediaCorrections::default());
    }

    #[test]
    fn below_troposphere_validity_returns_typed_error() {
        let err = observable_media_corrections(
            receiver(),
            2.0 * PI / DEGREES_PER_SEMICIRCLE,
            0.0,
            T_RX_J2000_S,
            F_L1_HZ,
            ObservableMediaOptions {
                troposphere: Some(ObservableTroposphereCorrection::default()),
                ionosphere: None,
            },
        )
        .expect_err("below mapping validity must fail");

        match err {
            ObservablesError::InvalidInput { field, kind } => {
                assert_eq!(field, "media.elevation_rad");
                assert_eq!(kind, ObservablesInputErrorKind::OutOfRange);
            }
            other => panic!("expected typed InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn range_media_prediction_adds_direct_troposphere_bits() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        let rx = [6_378_137.0, 0.0, 0.0];
        let elevation_rad = core::f64::consts::FRAC_PI_2;
        let range_m = 20_000_000.0;
        let delta = [range_m, 0.0, 0.0];
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [rx[0] + delta[0], rx[1] + delta[1], rx[2] + delta[2]],
                clock_s: Some(0.0),
            },
        };
        let options = MediaPredictOptions {
            prediction: PredictOptions {
                carrier_hz: f64::NAN,
                light_time: false,
                sagnac: false,
            },
            media: ObservableMediaOptions {
                troposphere: Some(ObservableTroposphereCorrection::default()),
                ionosphere: None,
            },
        };
        let request = [RangePredictionRequest {
            sat,
            receiver_ecef_m: rx,
            t_rx_j2000_s: T_RX_J2000_S,
        }];
        let zero_prediction = RangePrediction {
            geometric_range_m: 0.0,
            sat_clock_s: None,
            transmit_time_j2000_s: 0.0,
            sat_pos_ecef_m: [0.0; 3],
        };
        let mut out = [MediaRangePrediction {
            prediction: zero_prediction,
            range_m: 0.0,
            media: AppliedMediaCorrections::default(),
        }];
        predict_ranges_with_media(&source, &request, options, &mut out)
            .expect("range media prediction");
        let got = out[0];
        let expected = got.prediction.geometric_range_m + direct_troposphere(elevation_rad);
        assert_bits_eq("corrected range", got.range_m, expected);
    }

    #[test]
    fn default_range_media_prediction_matches_range_bits_with_unused_carrier() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        let rx = [6_378_137.0, 0.0, 0.0];
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [26_378_137.0, 0.0, 0.0],
                clock_s: Some(0.0),
            },
        };
        let options = PredictOptions {
            carrier_hz: f64::NAN,
            light_time: false,
            sagnac: false,
        };
        let request = [RangePredictionRequest {
            sat,
            receiver_ecef_m: rx,
            t_rx_j2000_s: T_RX_J2000_S,
        }];
        let zero_prediction = RangePrediction {
            geometric_range_m: 0.0,
            sat_clock_s: None,
            transmit_time_j2000_s: 0.0,
            sat_pos_ecef_m: [0.0; 3],
        };
        let mut plain = [zero_prediction];
        predict_ranges(&source, &request, options, &mut plain).expect("plain range");
        let mut media = [MediaRangePrediction {
            prediction: zero_prediction,
            range_m: 0.0,
            media: AppliedMediaCorrections::default(),
        }];
        predict_ranges_with_media(
            &source,
            &request,
            MediaPredictOptions {
                prediction: options,
                media: ObservableMediaOptions::default(),
            },
            &mut media,
        )
        .expect("default media range");

        assert_range_prediction_bits_eq(&media[0].prediction, &plain[0]);
        assert_bits_eq(
            "default range",
            media[0].range_m,
            plain[0].geometric_range_m,
        );
        assert_eq!(media[0].media, AppliedMediaCorrections::default());
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use crate::{GnssSatelliteId, GnssSystem};

    #[derive(Debug, Clone, Copy)]
    struct StaticSource {
        state: ObservableState,
    }

    impl ObservableEphemerisSource for StaticSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(self.state)
        }
    }

    fn sp3_fixture() -> Sp3 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
        Sp3::parse(&bytes).expect("parse SP3 fixture")
    }

    fn static_source(position_ecef_m: [f64; 3]) -> StaticSource {
        StaticSource {
            state: ObservableState {
                position_ecef_m,
                clock_s: Some(0.0),
            },
        }
    }

    fn no_light_time_options() -> PredictOptions {
        PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: false,
            sagnac: true,
        }
    }

    fn assert_invalid_observables_input(
        err: ObservablesError,
        field: &'static str,
        kind: ObservablesInputErrorKind,
    ) {
        match err {
            ObservablesError::InvalidInput {
                field: got_field,
                kind: got_kind,
            } => {
                assert_eq!(got_field, field);
                assert_eq!(got_kind, kind);
            }
            other => panic!("expected InvalidInput({field}, {kind:?}), got {other:?}"),
        }
    }

    #[test]
    fn split_julian_to_j2000_seconds_is_exact_at_noon() {
        let t = j2000_seconds_from_split(2_459_024.5, 0.5).expect("valid split Julian date");
        assert_eq!(t, 646_272_000.0);
    }

    #[test]
    fn split_julian_to_j2000_seconds_rejects_non_finite_parts() {
        for (jd_whole, jd_fraction, field) in [
            (f64::NAN, 0.5, "jd_whole"),
            (f64::INFINITY, 0.5, "jd_whole"),
            (2_459_024.5, f64::NAN, "jd_fraction"),
            (2_459_024.5, f64::NEG_INFINITY, "jd_fraction"),
        ] {
            let err = j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect_err("non-finite split Julian date part must fail");
            assert_invalid_observables_input(err, field, ObservablesInputErrorKind::NonFinite);
        }
    }

    /// The reference case of the external formula oracle, which rounded the transmission
    /// epoch to whole microseconds, replayed through that rounding bit for bit. The public
    /// predictor keeps every bit of the flight time and differs from the replay through
    /// the transmission epoch alone.
    #[test]
    fn sp3_predict_reference_case() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let rx = [3_512_900.0, 780_500.0, 5_248_700.0];
        let obs = rounded_microsecond_replay::predict(
            &sp3,
            sat,
            rx,
            646_272_000.0,
            PredictOptions::default(),
        )
        .expect("predict observables");
        let exact = predict(&sp3, sat, rx, 646_272_000.0, PredictOptions::default())
            .expect("predict observables");
        assert_eq!(exact.transmit_offset_us, obs.transmit_offset_us);
        let tau_s = 646_272_000.0 - exact.transmit_time_j2000_s;
        assert!(
            (tau_s - 69_288.0e-6).abs() <= 0.5e-6 + 2.0_f64.powi(-23),
            "flight time {tau_s} s"
        );
        let epoch_shift_s = exact.transmit_time_j2000_s - obs.transmit_time_j2000_s;
        let moved_m =
            (exact.geometric_range_m - obs.geometric_range_m) - obs.range_rate_m_s * epoch_shift_s;
        assert!(moved_m.abs() <= 1.0e-6, "range moved {moved_m} m");

        assert_eq!(obs.geometric_range_m.to_bits(), 0x4173cf438ba57358);
        assert_eq!(obs.range_rate_m_s.to_bits(), 0x402d7dd36f6b8980);
        assert_eq!(obs.doppler_hz.to_bits(), 0xc0535f534ba7c77d);
        assert_eq!(obs.sat_clock_s.unwrap().to_bits(), 0x3ef04d2d8279460c);
        assert_eq!(obs.elevation_deg.to_bits(), 0x4054590eed870f52);
        assert_eq!(obs.azimuth_deg.to_bits(), 0x40645ff5a090a131);
        assert_eq!(obs.transmit_offset_us, 69_288);
        assert_eq!(obs.transmit_time_j2000_s.to_bits(), 0x41c342a9fff72192);
        assert_eq!(
            obs.los_unit.map(f64::to_bits),
            [0x3fe4c70da9fa70dd, 0x3fc834429adb2bae, 0x3fe792a4f57fdcb1,]
        );
        assert_eq!(
            obs.sat_pos_ecef_m.map(f64::to_bits),
            [0x41703667d8c0eb8f, 0x4151f601b1d775f3, 0x4173992c0ec03dcd,]
        );
        assert_eq!(
            obs.sat_velocity_m_s.map(f64::to_bits),
            [0xc09c17d81e540ab6, 0x409a192982abbeb7, 0x40926013f2ae8000,]
        );
    }

    #[test]
    fn predict_rejects_invalid_entry_inputs() {
        let source = static_source([20_200_000.0, 14_000_000.0, 21_700_000.0]);
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");

        let err = predict(
            &source,
            sat,
            [f64::NAN, 0.0, 0.0],
            646_272_000.0,
            no_light_time_options(),
        )
        .expect_err("non-finite receiver position must fail");
        assert_invalid_observables_input(
            err,
            "receiver_ecef_m",
            ObservablesInputErrorKind::NonFinite,
        );

        let err = predict(
            &source,
            sat,
            [0.0, 0.0, 0.0],
            f64::INFINITY,
            no_light_time_options(),
        )
        .expect_err("non-finite receive time must fail");
        assert_invalid_observables_input(err, "t_rx_j2000_s", ObservablesInputErrorKind::NonFinite);

        let mut options = no_light_time_options();
        options.carrier_hz = 0.0;
        let err = predict(&source, sat, [0.0, 0.0, 0.0], 646_272_000.0, options)
            .expect_err("non-positive carrier must fail");
        assert_invalid_observables_input(
            err,
            "options.carrier_hz",
            ObservablesInputErrorKind::NotPositive,
        );
    }

    #[test]
    fn predict_rejects_invalid_source_state_and_zero_range() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");

        let source = static_source([f64::NAN, 14_000_000.0, 21_700_000.0]);
        let err = predict(
            &source,
            sat,
            [0.0, 0.0, 0.0],
            646_272_000.0,
            no_light_time_options(),
        )
        .expect_err("non-finite ephemeris position must fail");
        assert_invalid_observables_input(
            err,
            "observable state position_ecef_m",
            ObservablesInputErrorKind::NonFinite,
        );

        let source = static_source([1_000.0, 2_000.0, 3_000.0]);
        let err = predict(
            &source,
            sat,
            [1_000.0, 2_000.0, 3_000.0],
            646_272_000.0,
            no_light_time_options(),
        )
        .expect_err("zero geometric range must fail");
        assert_invalid_observables_input(
            err,
            "geometric_range_m",
            ObservablesInputErrorKind::NotPositive,
        );
    }

    #[test]
    fn topocentric_rejects_invalid_receiver_geodetic_conversion() {
        let err = topocentric([f64::MAX, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0)
            .expect_err("invalid receiver geodetic conversion must fail");

        assert_invalid_observables_input(
            err,
            "receiver_ecef_m",
            ObservablesInputErrorKind::OutOfRange,
        );
    }

    // WGS84 equatorial radius; a receiver here sits at lat=0, lon=0, height~=0,
    // so the geodetic up direction is +X and a satellite displaced along +X is
    // exactly overhead (degenerate azimuth geometry).
    const EQUATORIAL_RX_X_M: f64 = 6_378_137.0;

    #[test]
    fn topocentric_azimuth_is_zero_at_exact_zenith() {
        // Satellite displaced purely radially (+X) above an equatorial receiver:
        // east == north == 0, so azimuth is degenerate.
        let geometry = topocentric(
            [EQUATORIAL_RX_X_M, 0.0, 0.0],
            [20_000_000.0, 0.0, 0.0],
            20_000_000.0,
        )
        .expect("zenith topocentric must not error");
        assert_eq!(geometry.azimuth_deg, 0.0);
        assert!(geometry.azimuth_deg.is_finite());
        assert!((geometry.elevation_deg - 90.0).abs() < 1e-9);
    }

    #[test]
    fn topocentric_azimuth_is_zero_just_off_zenith() {
        // A ~1e-9 m horizontal nudge is pure rounding-scale noise at a 20,000 km
        // range, so azimuth stays pinned to 0.0 (RTKLIB satazel semantics).
        let geometry = topocentric(
            [EQUATORIAL_RX_X_M, 0.0, 0.0],
            [20_000_000.0, 1.0e-9, 1.0e-9],
            20_000_000.0,
        )
        .expect("near-zenith topocentric must not error");
        assert_eq!(geometry.azimuth_deg, 0.0);
        assert!(geometry.azimuth_deg.is_finite());
    }

    #[test]
    fn predict_azimuth_is_zero_at_exact_zenith() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [EQUATORIAL_RX_X_M + 20_000_000.0, 0.0, 0.0],
                clock_s: None,
            },
        };
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
        let obs = predict(
            &source,
            sat,
            [EQUATORIAL_RX_X_M, 0.0, 0.0],
            0.0,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: false,
                sagnac: false,
            },
        )
        .expect("zenith predict must not error");
        assert_eq!(obs.azimuth_deg, 0.0);
        assert!(obs.azimuth_deg.is_finite());
        assert!((obs.elevation_deg - 90.0).abs() < 1e-9);
    }

    fn batch_test_requests() -> Vec<PredictRequest> {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid satellite id");
        vec![
            (sat1, [4_027_894.0, 307_046.0, 4_919_474.0], 646_272_000.0),
            (sat2, [4_027_900.0, 307_050.0, 4_919_480.0], 646_272_030.0),
            (
                sat1,
                [1_130_000.0, -4_830_000.0, 3_994_000.0],
                646_272_060.0,
            ),
            (
                sat2,
                [-2_700_000.0, -4_290_000.0, 3_855_000.0],
                646_272_090.0,
            ),
        ]
    }

    fn assert_observables_bits_eq(a: &PredictedObservables, b: &PredictedObservables) {
        assert_eq!(a.geometric_range_m.to_bits(), b.geometric_range_m.to_bits());
        assert_eq!(a.range_rate_m_s.to_bits(), b.range_rate_m_s.to_bits());
        assert_eq!(a.doppler_hz.to_bits(), b.doppler_hz.to_bits());
        assert_eq!(a.elevation_deg.to_bits(), b.elevation_deg.to_bits());
        assert_eq!(a.azimuth_deg.to_bits(), b.azimuth_deg.to_bits());
        assert_eq!(a.transmit_offset_us, b.transmit_offset_us);
        assert_eq!(
            a.transmit_time_j2000_s.to_bits(),
            b.transmit_time_j2000_s.to_bits()
        );
        for k in 0..3 {
            assert_eq!(a.los_unit[k].to_bits(), b.los_unit[k].to_bits());
            assert_eq!(a.sat_pos_ecef_m[k].to_bits(), b.sat_pos_ecef_m[k].to_bits());
            assert_eq!(
                a.sat_velocity_m_s[k].to_bits(),
                b.sat_velocity_m_s[k].to_bits()
            );
        }
    }

    #[test]
    fn predict_batch_matches_scalar_loop_bitwise() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let options = PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        };
        let requests = batch_test_requests();
        let batch = predict_batch(&source, &requests, options);
        assert_eq!(batch.len(), requests.len());
        for (entry, &(sat, rx, t)) in batch.iter().zip(requests.iter()) {
            let scalar = predict(&source, sat, rx, t, options);
            match (entry, &scalar) {
                (Ok(b), Ok(s)) => assert_observables_bits_eq(b, s),
                (Err(_), Err(_)) => {}
                _ => panic!("batch and scalar predict disagree on Ok/Err"),
            }
        }
    }

    #[test]
    fn predict_batch_parallel_matches_serial_bitwise() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let options = PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        };
        let requests = batch_test_requests();
        let serial = predict_batch(&source, &requests, options);
        let parallel = predict_batch_parallel(&source, &requests, options);
        assert_eq!(serial.len(), parallel.len());
        for (i, (&(sat, receiver_ecef_m, t_rx_j2000_s), p)) in
            requests.iter().zip(parallel.iter()).enumerate()
        {
            let scalar = predict(&source, sat, receiver_ecef_m, t_rx_j2000_s, options);
            match (p, &scalar) {
                (Ok(batch), Ok(single)) => assert_observables_bits_eq(batch, single),
                (Err(_), Err(_)) => {}
                _ => panic!("parallel batch and scalar prediction {i} disagree on Ok/Err"),
            }
        }
    }
}
