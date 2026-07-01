//! Sample-backed precise-ephemeris source.
//!
//! The canonical precise-ephemeris intermediate representation is a set of
//! per-satellite ECEF position (+ optional clock) samples on a time axis. SP3
//! text is one serialization of that IR; [`super::Sp3`] is the parser. This
//! module builds the same interpolatable source directly from samples, with no
//! text in the loop, and drives the exact same interpolation substrate the
//! SP3-parsed source uses ([`super::interp::interpolate_precise_state`]).
//!
//! # Byte-identical parity with the SP3 path
//!
//! [`PreciseEphemerisSamples::from_samples`] gathers the same node vectors the
//! SP3 gather builds (floored J2000-second axis; file-native km position; native
//! microsecond clock) and feeds the shared interpolator. The byte-identity
//! contract is precise:
//!
//! - **Byte-identical** holds when the samples are the faithful image of the
//!   interpolation fit nodes, that is the round-trip case: samples obtained from
//!   [`Sp3::precise_ephemeris_samples`] on a parsed source, rebuilt via
//!   `from_samples`, interpolate to byte-identical states and predicted ranges.
//!   The SI value each sample carries is the exact `km * KM_TO_M` /
//!   `us * US_TO_S` image of a fit node, and the single reconstructing divide is
//!   its correctly-rounded inverse.
//! - **At the sample's own precision** otherwise. Samples carrying lower
//!   precision (for example reconstructed from 6-decimal SP3 *text*, which
//!   serializes only 6 fractional digits, or externally rounded) interpolate at
//!   that precision, not to the full-precision fit. This is why the contract is
//!   stated against the fit nodes and NOT against serialize-then-reparse: SP3
//!   text carries ~0.5 mm less than the SI sample, so a source rebuilt from
//!   re-serialized text can differ from the parsed source at that scale.
//!
//! One further numeric caveat is inherent even in the round-trip case and
//! documented rather than hidden: the SP3 interpolator fits in the file-native
//! units (km / microseconds), while a [`PreciseEphemerisSample`] carries SI
//! meters / seconds. The `km -> m` map (`km * 1000`) is not injective on
//! IEEE-754 doubles: distinct adjacent km floats can round to the same meters
//! value. So a sample whose meters came from a km node that shares its meters
//! image with an adjacent km float reconstructs to the correctly-rounded km,
//! which may differ from the original by <= 1 ULP (a few nanometres). For
//! samples whose SI values are the faithful image of the fit nodes (the common,
//! round-trip case) the reconstruction is exact and parity is byte-identical.

use std::collections::BTreeMap;

use crate::astro::time::model::{Instant, TimeScale};
use crate::constants::{KM_TO_M, US_TO_S};
use crate::id::GnssSatelliteId;
use crate::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
use crate::sp3::interp::{instant_to_j2000_seconds, interpolate_precise_state};
use crate::sp3::{Sp3, Sp3State};
use crate::{Error, Result};

/// One precise-ephemeris sample: a satellite's ECEF position (and optional
/// clock) at one epoch, in SI units.
///
/// This is the canonical serialization-independent IR element. `position_ecef_m`
/// is the ITRF/IGS ECEF position in meters; `clock_s` is the satellite clock
/// offset in seconds, `None` when the source carried no clock estimate.
///
/// `clock_event` mirrors the SP3 `E` clock-event flag: when `true` this epoch
/// marks a clock discontinuity, and the interpolator splits the clock arc there
/// (it never interpolates a clock across a reset). The common case carries no
/// event; use [`PreciseEphemerisSample::new`] for it and set the field directly
/// when reconstructing a flagged epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreciseEphemerisSample {
    /// The satellite this sample describes.
    pub sat: GnssSatelliteId,
    /// The sample epoch, tagged with its time scale.
    pub epoch: Instant,
    /// Satellite position in the ITRF/IGS ECEF frame, meters.
    pub position_ecef_m: [f64; 3],
    /// Satellite clock offset in seconds (`None` when no clock estimate exists).
    pub clock_s: Option<f64>,
    /// Whether this epoch carries the SP3 `E` clock-event flag: `true` splits
    /// the clock interpolation arc here (a clock reset takes effect at this
    /// epoch), matching [`super::Sp3Flags::clock_event`]. Defaults to `false`.
    pub clock_event: bool,
}

impl PreciseEphemerisSample {
    /// Build a sample with no clock-event flag (the common, no-reset case).
    ///
    /// `clock_event` defaults to `false`. For an epoch that carries an SP3 `E`
    /// clock reset, construct with this and then set `clock_event = true`.
    pub fn new(
        sat: GnssSatelliteId,
        epoch: Instant,
        position_ecef_m: [f64; 3],
        clock_s: Option<f64>,
    ) -> Self {
        Self {
            sat,
            epoch,
            position_ecef_m,
            clock_s,
            clock_event: false,
        }
    }
}

/// Validation failure building a [`PreciseEphemerisSamples`] source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreciseSamplesError {
    /// No samples were supplied.
    Empty,
    /// A satellite has only a single sample; interpolation needs at least two.
    SingleSampleSatellite(GnssSatelliteId),
    /// A satellite's sample epochs are not strictly increasing in time.
    NonMonotonicEpochs(GnssSatelliteId),
    /// Samples carry more than one time scale; a source is a single time axis.
    MixedTimeScales,
    /// A sample epoch cannot be expressed as seconds since J2000.
    EpochNotRepresentable(GnssSatelliteId),
    /// A sample position or clock value was not finite.
    NonFiniteSample(GnssSatelliteId),
}

impl core::fmt::Display for PreciseSamplesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "no precise-ephemeris samples supplied"),
            Self::SingleSampleSatellite(sat) => {
                write!(f, "satellite {sat} has a single sample; need at least two")
            }
            Self::NonMonotonicEpochs(sat) => {
                write!(
                    f,
                    "satellite {sat} sample epochs are not strictly increasing"
                )
            }
            Self::MixedTimeScales => write!(f, "samples carry more than one time scale"),
            Self::EpochNotRepresentable(sat) => {
                write!(
                    f,
                    "satellite {sat} sample epoch is not representable as J2000 seconds"
                )
            }
            Self::NonFiniteSample(sat) => write!(f, "satellite {sat} has a non-finite sample"),
        }
    }
}

impl std::error::Error for PreciseSamplesError {}

/// Per-satellite node series in the file-native fit units, ready for the shared
/// interpolation substrate: floored J2000-second axis, km position axes, and
/// native `(x, clock_us, clock_event)` clock nodes.
#[derive(Debug, Clone, PartialEq)]
struct SatSeries {
    x: Vec<f64>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    clk: Vec<(f64, f64, bool)>,
}

/// A precise-ephemeris source built from samples rather than parsed text.
///
/// Implements [`crate::observables::ObservableEphemerisSource`] and exposes the
/// same [`PreciseEphemerisSamples::position_at_j2000_seconds`] query as the
/// SP3-parsed source, sharing its interpolation substrate.
#[derive(Debug, Clone, PartialEq)]
pub struct PreciseEphemerisSamples {
    time_scale: TimeScale,
    nodes: BTreeMap<GnssSatelliteId, SatSeries>,
}

impl PreciseEphemerisSamples {
    /// Build a source from an iterator of samples.
    ///
    /// Samples are grouped by satellite, keeping their supplied order. Each
    /// satellite's series is validated to be strictly increasing in epoch and to
    /// carry at least two nodes. All samples must share one time scale. The node
    /// substrate is prepared exactly as the SP3 gather prepares it (floored
    /// J2000-second axis; native km position; native microsecond clock), so the
    /// interpolation is byte-identical to the SP3 path for samples that are the
    /// faithful image of the fit nodes (the round-trip case); samples carrying
    /// lower precision interpolate at that precision. See the module docs for the
    /// precise byte-identity contract and the SI-vs-native reconstruction caveat.
    pub fn from_samples(
        samples: impl IntoIterator<Item = PreciseEphemerisSample>,
    ) -> core::result::Result<Self, PreciseSamplesError> {
        let mut time_scale: Option<TimeScale> = None;
        let mut grouped: BTreeMap<GnssSatelliteId, SatSeries> = BTreeMap::new();

        for sample in samples {
            match time_scale {
                None => time_scale = Some(sample.epoch.scale),
                Some(scale) if scale == sample.epoch.scale => {}
                Some(_) => return Err(PreciseSamplesError::MixedTimeScales),
            }

            if !sample.position_ecef_m.iter().all(|c| c.is_finite())
                || sample.clock_s.is_some_and(|c| !c.is_finite())
            {
                return Err(PreciseSamplesError::NonFiniteSample(sample.sat));
            }

            // Node axis: floored to whole seconds, matching the SP3 gather (the
            // query, at evaluation time, is not floored).
            let xi = instant_to_j2000_seconds(&sample.epoch)
                .ok_or(PreciseSamplesError::EpochNotRepresentable(sample.sat))?
                .floor();

            // SI -> file-native fit units. The single divide is the correctly
            // rounded inverse of the SP3 parser's `km * KM_TO_M` / `us * US_TO_S`
            // (see the module docs for the non-injective boundary).
            let series = grouped.entry(sample.sat).or_insert_with(|| SatSeries {
                x: Vec::new(),
                kx: Vec::new(),
                ky: Vec::new(),
                kz: Vec::new(),
                clk: Vec::new(),
            });
            series.x.push(xi);
            series.kx.push(sample.position_ecef_m[0] / KM_TO_M);
            series.ky.push(sample.position_ecef_m[1] / KM_TO_M);
            series.kz.push(sample.position_ecef_m[2] / KM_TO_M);
            if let Some(clock_s) = sample.clock_s {
                // Carry the clock-event flag onto the node so the shared
                // interpolator splits the clock arc at an `E` reset exactly as
                // the SP3 path does (see `interp::interpolate_clock`).
                series.clk.push((xi, clock_s / US_TO_S, sample.clock_event));
            }
        }

        if grouped.is_empty() {
            return Err(PreciseSamplesError::Empty);
        }
        for (&sat, series) in &grouped {
            if series.x.len() < 2 {
                return Err(PreciseSamplesError::SingleSampleSatellite(sat));
            }
            if series.x.windows(2).any(|w| w[1] <= w[0]) {
                return Err(PreciseSamplesError::NonMonotonicEpochs(sat));
            }
        }

        Ok(Self {
            time_scale: time_scale.expect("non-empty group has a time scale"),
            nodes: grouped,
        })
    }

    /// The time scale every sample epoch is expressed in.
    pub fn time_scale(&self) -> TimeScale {
        self.time_scale
    }

    /// The satellites this source can interpolate, in ascending order.
    pub fn satellites(&self) -> impl Iterator<Item = GnssSatelliteId> + '_ {
        self.nodes.keys().copied()
    }

    /// Interpolate the state of `sat` at an arbitrary J2000-second epoch.
    ///
    /// Identical recipe, substrate, and error surface as
    /// [`Sp3::position_at_j2000_seconds`]: [`Error::UnknownSatellite`] for a
    /// satellite with no nodes, [`Error::EpochOutOfRange`] for an out-of-coverage
    /// query, [`Error::InvalidInput`] for a non-finite query.
    pub fn position_at_j2000_seconds(&self, sat: GnssSatelliteId, query: f64) -> Result<Sp3State> {
        // Drive the shared interpolator even for a missing satellite (empty node
        // slices) so the validation order matches the SP3 path exactly: the
        // interpolator validates the query (finite) BEFORE it reports
        // `UnknownSatellite` for an empty node set. A missing-satellite lookup
        // here must not shadow an `InvalidInput` for a non-finite query.
        static EMPTY_F64: [f64; 0] = [];
        static EMPTY_CLK: [(f64, f64, bool); 0] = [];
        match self.nodes.get(&sat) {
            Some(series) => interpolate_precise_state(
                sat,
                &series.x,
                &series.kx,
                &series.ky,
                &series.kz,
                &series.clk,
                query,
            ),
            None => interpolate_precise_state(
                sat, &EMPTY_F64, &EMPTY_F64, &EMPTY_F64, &EMPTY_F64, &EMPTY_CLK, query,
            ),
        }
    }

    /// Interpolate the state of `sat` at an arbitrary [`Instant`].
    ///
    /// The query instant must be tagged with the same time scale as the samples.
    pub fn position(&self, sat: GnssSatelliteId, epoch: Instant) -> Result<Sp3State> {
        if epoch.scale != self.time_scale {
            return Err(Error::InvalidInput(format!(
                "precise-sample query time scale {} does not match source time scale {}",
                epoch.scale.abbrev(),
                self.time_scale.abbrev()
            )));
        }
        let query = instant_to_j2000_seconds(&epoch).ok_or(Error::EpochOutOfRange)?;
        self.position_at_j2000_seconds(sat, query)
    }
}

impl ObservableEphemerisSource for PreciseEphemerisSamples {
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

impl Sp3 {
    /// Extract this product as the canonical precise-ephemeris samples, in SI
    /// units, one per real position record in ascending epoch order.
    ///
    /// Round-tripping `PreciseEphemerisSamples::from_samples(sp3.
    /// precise_ephemeris_samples())` rebuilds the same interpolatable source
    /// (byte-identical for samples whose meters are the faithful image of the fit
    /// km; see the module docs).
    pub fn precise_ephemeris_samples(&self) -> Vec<PreciseEphemerisSample> {
        let mut out = Vec::new();
        for (idx, &epoch) in self.epochs.iter().enumerate() {
            if let Ok(states) = self.states_at(idx) {
                for (&sat, state) in states {
                    out.push(PreciseEphemerisSample {
                        sat,
                        epoch,
                        position_ecef_m: state.position.as_array(),
                        clock_s: state.clock_s,
                        clock_event: state.flags.clock_event,
                    });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::time::model::{InstantRepr, JulianDateSplit};
    use crate::GnssSystem;

    const J2000_JD_WHOLE: f64 = 2_451_545.0;
    const SECONDS_PER_DAY: f64 = 86_400.0;

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    fn sample(
        scale: TimeScale,
        j2000_s: f64,
        prn: u8,
        pos: [f64; 3],
        clk: Option<f64>,
    ) -> PreciseEphemerisSample {
        let split =
            JulianDateSplit::new(J2000_JD_WHOLE, j2000_s / SECONDS_PER_DAY).expect("valid split");
        PreciseEphemerisSample::new(
            gps(prn),
            Instant {
                scale,
                repr: InstantRepr::JulianDate(split),
            },
            pos,
            clk,
        )
    }

    #[test]
    fn from_samples_rejects_empty() {
        let err = PreciseEphemerisSamples::from_samples(std::iter::empty())
            .expect_err("empty sample set must fail");
        assert_eq!(err, PreciseSamplesError::Empty);
    }

    #[test]
    fn from_samples_rejects_single_sample_satellite() {
        let samples = vec![sample(
            TimeScale::Gpst,
            0.0,
            21,
            [20_000_000.0, 14_000_000.0, 21_000_000.0],
            Some(1.0e-6),
        )];
        let err =
            PreciseEphemerisSamples::from_samples(samples).expect_err("single sample must fail");
        assert_eq!(err, PreciseSamplesError::SingleSampleSatellite(gps(21)));
    }

    #[test]
    fn from_samples_rejects_non_monotonic_epochs() {
        let samples = vec![
            sample(TimeScale::Gpst, 900.0, 21, [1.0e7, 2.0e7, 3.0e7], None),
            sample(TimeScale::Gpst, 900.0, 21, [1.0e7, 2.0e7, 3.0e7], None),
        ];
        let err = PreciseEphemerisSamples::from_samples(samples)
            .expect_err("repeated epoch must fail as non-monotonic");
        assert_eq!(err, PreciseSamplesError::NonMonotonicEpochs(gps(21)));

        let descending = vec![
            sample(TimeScale::Gpst, 1_800.0, 7, [1.0e7, 2.0e7, 3.0e7], None),
            sample(TimeScale::Gpst, 900.0, 7, [1.1e7, 2.1e7, 3.1e7], None),
        ];
        let err = PreciseEphemerisSamples::from_samples(descending)
            .expect_err("descending epochs must fail");
        assert_eq!(err, PreciseSamplesError::NonMonotonicEpochs(gps(7)));
    }

    #[test]
    fn from_samples_rejects_mixed_time_scales() {
        let samples = vec![
            sample(TimeScale::Gpst, 0.0, 21, [1.0e7, 2.0e7, 3.0e7], None),
            sample(TimeScale::Utc, 900.0, 21, [1.0e7, 2.0e7, 3.0e7], None),
        ];
        let err = PreciseEphemerisSamples::from_samples(samples)
            .expect_err("mixed time scales must fail");
        assert_eq!(err, PreciseSamplesError::MixedTimeScales);
    }

    #[test]
    fn from_samples_rejects_non_finite_sample() {
        let samples = vec![
            sample(TimeScale::Gpst, 0.0, 21, [f64::NAN, 2.0e7, 3.0e7], None),
            sample(TimeScale::Gpst, 900.0, 21, [1.0e7, 2.0e7, 3.0e7], None),
        ];
        let err = PreciseEphemerisSamples::from_samples(samples).expect_err("non-finite must fail");
        assert_eq!(err, PreciseSamplesError::NonFiniteSample(gps(21)));
    }

    #[test]
    fn from_samples_out_of_range_query_errors() {
        let samples = vec![
            sample(
                TimeScale::Gpst,
                0.0,
                21,
                [2.0e7, 1.4e7, 2.1e7],
                Some(1.0e-6),
            ),
            sample(
                TimeScale::Gpst,
                900.0,
                21,
                [2.0e7, 1.4e7, 2.1e7],
                Some(1.0e-6),
            ),
        ];
        let source = PreciseEphemerisSamples::from_samples(samples).expect("valid source");
        // A query far past the node span (many node spacings) is refused, exactly
        // like the SP3 path.
        let err = source
            .position_at_j2000_seconds(gps(21), 1_000_000.0)
            .expect_err("out-of-coverage query must fail");
        assert_eq!(err, Error::EpochOutOfRange);
    }

    #[test]
    fn unknown_sat_with_non_finite_query_is_invalid_input() {
        let samples = vec![
            sample(
                TimeScale::Gpst,
                0.0,
                21,
                [2.0e7, 1.4e7, 2.1e7],
                Some(1.0e-6),
            ),
            sample(
                TimeScale::Gpst,
                900.0,
                21,
                [2.0e7, 1.4e7, 2.1e7],
                Some(1.0e-6),
            ),
        ];
        let source = PreciseEphemerisSamples::from_samples(samples).expect("valid source");

        // The query is validated (finite) BEFORE the satellite-map lookup, so a
        // non-finite query on a missing satellite returns InvalidInput, matching
        // the SP3 path (not UnknownSatellite).
        let err = source
            .position_at_j2000_seconds(gps(7), f64::NAN)
            .expect_err("non-finite query on unknown sat must fail");
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );

        // A finite query on a missing satellite still reports UnknownSatellite.
        let err = source
            .position_at_j2000_seconds(gps(7), 0.0)
            .expect_err("finite query on unknown sat must fail");
        assert_eq!(err, Error::UnknownSatellite(gps(7)));
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod parity_tests {
    use super::*;
    use crate::observables::{
        predict, predict_ranges, PredictOptions, RangePrediction, RangePredictionRequest,
    };
    use crate::GnssSystem;

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    fn round_trip_safe_km(km: f64) -> bool {
        (km * KM_TO_M) / KM_TO_M == km
    }
    fn round_trip_safe_us(us: f64) -> bool {
        (us * US_TO_S) / US_TO_S == us
    }

    /// Author an SP3-c product from round-trip-safe km/us values, reusing a real
    /// fixture's header. The samples this parses to serialize losslessly, so the
    /// sample-backed source is byte-identical to this parsed source.
    fn authored_sp3() -> Sp3 {
        let header_src = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GAP_G01_20201760000_15M.sp3"
        );
        let gap = std::fs::read_to_string(header_src).expect("read header fixture");
        let epoch_start = gap.find("\n*  ").expect("first epoch line") + 1;
        let header = &gap[..epoch_start];

        // A gentle, non-collinear integer-km path (radius ~26000 km), integer-us
        // clock; every value is round-trip-safe, asserted below.
        let xs = [
            26_000.0, 25_990.0, 25_960.0, 25_910.0, 25_840.0, 25_750.0, 25_640.0, 25_510.0,
            25_360.0, 25_190.0, 25_000.0, 24_790.0,
        ];
        let ys = [
            1_000.0, 2_000.0, 2_990.0, 3_960.0, 4_910.0, 5_840.0, 6_750.0, 7_640.0, 8_510.0,
            9_360.0, 10_190.0, 11_000.0,
        ];
        let zs = [
            -3_000.0, -3_050.0, -3_120.0, -3_210.0, -3_320.0, -3_450.0, -3_600.0, -3_770.0,
            -3_960.0, -4_170.0, -4_400.0, -4_650.0,
        ];
        let cs = [
            100.0, 142.0, -313.0, 6_159.0, 1_234.0, -884.0, 401.0, 862.0, -606.0, 10.0, -369.0,
            3_654.0,
        ];

        let mut text = String::from(header);
        for i in 0..xs.len() {
            assert!(round_trip_safe_km(xs[i]), "xs[{i}] not round-trip-safe");
            assert!(round_trip_safe_km(ys[i]), "ys[{i}] not round-trip-safe");
            assert!(round_trip_safe_km(zs[i]), "zs[{i}] not round-trip-safe");
            assert!(round_trip_safe_us(cs[i]), "cs[{i}] not round-trip-safe");
            let total_min = i * 15;
            let hour = total_min / 60;
            let minute = total_min % 60;
            text.push_str(&format!("*  2020  6 24 {hour:2} {minute:2}  0.00000000\n"));
            text.push_str(&format!(
                "PG01{:14.6}{:14.6}{:14.6}{:14.6}\n",
                xs[i], ys[i], zs[i], cs[i]
            ));
        }
        text.push_str("EOF\n");
        Sp3::parse(text.as_bytes()).expect("parse authored SP3")
    }

    /// Author an SP3-c product whose epoch at index `event_idx` carries the `E`
    /// clock-event flag on PG01's position record, so the clock arc is split
    /// there. Values are round-trip-safe (asserted), so the parsed samples are
    /// the faithful image of the fit nodes.
    fn authored_sp3_with_clock_event(event_idx: usize) -> Sp3 {
        let header_src = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GAP_G01_20201760000_15M.sp3"
        );
        let gap = std::fs::read_to_string(header_src).expect("read header fixture");
        let epoch_start = gap.find("\n*  ").expect("first epoch line") + 1;
        let header = &gap[..epoch_start];

        let xs = [
            26_000.0, 25_990.0, 25_960.0, 25_910.0, 25_840.0, 25_750.0, 25_640.0, 25_510.0,
            25_360.0, 25_190.0, 25_000.0, 24_790.0,
        ];
        let ys = [
            1_000.0, 2_000.0, 2_990.0, 3_960.0, 4_910.0, 5_840.0, 6_750.0, 7_640.0, 8_510.0,
            9_360.0, 10_190.0, 11_000.0,
        ];
        let zs = [
            -3_000.0, -3_050.0, -3_120.0, -3_210.0, -3_320.0, -3_450.0, -3_600.0, -3_770.0,
            -3_960.0, -4_170.0, -4_400.0, -4_650.0,
        ];
        // A clock series with a hard reset at the event epoch, so the split
        // (vs a spline fit across it) actually moves the interpolated value.
        let cs = [
            100.0, 142.0, 180.0, 210.0, 235.0, 260.0, -7_500.0, -7_550.0, -7_680.0, -7_790.0,
            -7_880.0, -7_000.0,
        ];

        let mut text = String::from(header);
        for i in 0..xs.len() {
            assert!(round_trip_safe_km(xs[i]), "xs[{i}] not round-trip-safe");
            assert!(round_trip_safe_km(ys[i]), "ys[{i}] not round-trip-safe");
            assert!(round_trip_safe_km(zs[i]), "zs[{i}] not round-trip-safe");
            assert!(round_trip_safe_us(cs[i]), "cs[{i}] not round-trip-safe");
            let total_min = i * 15;
            let hour = total_min / 60;
            let minute = total_min % 60;
            text.push_str(&format!("*  2020  6 24 {hour:2} {minute:2}  0.00000000\n"));
            // Position record: PG01 + 4 fixed-width fields (through column 60).
            let mut record = format!(
                "PG01{:14.6}{:14.6}{:14.6}{:14.6}",
                xs[i], ys[i], zs[i], cs[i]
            );
            if i == event_idx {
                // The clock-event `E` flag lives at column 74 (see
                // `sp3::parse_flags`). Pad out to it, then place `E`.
                while record.len() < 74 {
                    record.push(' ');
                }
                record.push('E');
            }
            record.push('\n');
            text.push_str(&record);
        }
        text.push_str("EOF\n");
        let sp3 = Sp3::parse(text.as_bytes()).expect("parse authored SP3");
        // Confirm the flag actually parsed onto the intended epoch.
        let state = sp3.state(gps(1), event_idx).expect("event-epoch state");
        assert!(
            state.flags.clock_event,
            "authored E flag did not parse at epoch {event_idx}"
        );
        sp3
    }

    /// FIX 1 parity: an SP3 product with an `E` clock-event epoch, extracted to
    /// samples and rebuilt via `from_samples`, must interpolate byte-identical
    /// clocks across the reset. The clock arc split is only preserved if the
    /// per-sample `clock_event` flag survives the round trip.
    #[test]
    fn from_samples_preserves_clock_event_arc_split() {
        let event_idx = 6usize;
        let sp3 = authored_sp3_with_clock_event(event_idx);
        let extracted = sp3.precise_ephemeris_samples();
        // The flag must be carried on the extracted sample at the event epoch.
        assert!(
            extracted.iter().any(|s| s.sat == gps(1) && s.clock_event),
            "extracted samples dropped the clock-event flag"
        );
        let samples = PreciseEphemerisSamples::from_samples(extracted).expect("source");

        let epochs = sp3.epochs_j2000_seconds();
        assert!(epochs.len() > event_idx + 1);

        // A grid spanning the reset: nodes and interior midpoints on both sides.
        let mut queries = Vec::new();
        for w in epochs.windows(2) {
            queries.push(w[0]);
            queries.push(0.5 * (w[0] + w[1]));
        }
        queries.push(*epochs.last().unwrap());

        let mut saw_some_clock = false;
        for &q in &queries {
            let a = sp3.position_at_j2000_seconds(gps(1), q).expect("sp3 state");
            let b = samples
                .position_at_j2000_seconds(gps(1), q)
                .expect("samples state");
            assert_eq!(
                a.clock_s.map(f64::to_bits),
                b.clock_s.map(f64::to_bits),
                "clock bits differ at query {q} across the reset"
            );
            if a.clock_s.is_some() {
                saw_some_clock = true;
            }
        }
        assert!(saw_some_clock, "expected clock estimates across the grid");

        // Sanity: the split genuinely changes the clock near the reset. Fitting
        // one spline across the reset (ignoring the event) would give a very
        // different value; assert the arc-split clock stays near the local
        // sub-arc data rather than being pulled across the discontinuity.
        let reset_epoch = epochs[event_idx];
        let just_after = 0.5 * (epochs[event_idx] + epochs[event_idx + 1]);
        let clk_after = sp3
            .position_at_j2000_seconds(gps(1), just_after)
            .expect("state after reset")
            .clock_s
            .expect("clock after reset");
        // The post-reset sub-arc clocks are around -7500 to -8000 us
        // (-7.5e-3 to -8.0e-3 s); a spline crossing the reset would land far
        // from that. This confirms the split is in force on both paths.
        assert!(
            clk_after < -1.0e-3,
            "post-reset clock {clk_after:e} s is not on the post-reset sub-arc; \
             the arc split was not applied"
        );
        let _ = reset_epoch;
    }

    fn assert_state_bits_eq(a: &Sp3State, b: &Sp3State) {
        assert_eq!(
            a.position.as_array().map(f64::to_bits),
            b.position.as_array().map(f64::to_bits),
            "position bits differ"
        );
        assert_eq!(
            a.clock_s.map(f64::to_bits),
            b.clock_s.map(f64::to_bits),
            "clock bits differ"
        );
    }

    #[test]
    fn from_samples_is_byte_identical_to_parsed_sp3() {
        let sp3 = authored_sp3();
        let samples =
            PreciseEphemerisSamples::from_samples(sp3.precise_ephemeris_samples()).expect("source");

        let epochs = sp3.epochs_j2000_seconds();
        assert!(epochs.len() >= 4);

        // Query grid: nodes, interior midpoints, quarter points.
        let mut queries = Vec::new();
        for w in epochs.windows(2) {
            queries.push(w[0]);
            queries.push(0.5 * (w[0] + w[1]));
            queries.push(0.75 * w[0] + 0.25 * w[1]);
        }
        queries.push(*epochs.last().unwrap());

        // Interpolated-state parity.
        for &q in &queries {
            let a = sp3.position_at_j2000_seconds(gps(1), q).expect("sp3 state");
            let b = samples
                .position_at_j2000_seconds(gps(1), q)
                .expect("samples state");
            assert_state_bits_eq(&a, &b);
        }

        // Predicted-range parity via the batch hot path, over a receiver grid.
        let receivers = [
            [4_027_894.0, 307_046.0, 4_919_474.0],
            [1_130_000.0, -4_830_000.0, 3_994_000.0],
            [-2_700_000.0, -4_290_000.0, 3_855_000.0],
        ];
        let options = PredictOptions::default();
        for &q in &queries {
            for rx in receivers {
                let requests = [RangePredictionRequest {
                    sat: gps(1),
                    receiver_ecef_m: rx,
                    t_rx_j2000_s: q,
                }];
                let mut a = [RangePrediction {
                    geometric_range_m: 0.0,
                    sat_clock_s: None,
                    transmit_time_j2000_s: 0.0,
                    sat_pos_ecef_m: [0.0; 3],
                }; 1];
                let mut b = a;
                predict_ranges(&sp3, &requests, options, &mut a).expect("sp3 ranges");
                predict_ranges(&samples, &requests, options, &mut b).expect("sample ranges");
                assert_eq!(
                    a[0].geometric_range_m.to_bits(),
                    b[0].geometric_range_m.to_bits()
                );
                assert_eq!(
                    a[0].transmit_time_j2000_s.to_bits(),
                    b[0].transmit_time_j2000_s.to_bits()
                );
                assert_eq!(
                    a[0].sat_clock_s.map(f64::to_bits),
                    b[0].sat_clock_s.map(f64::to_bits)
                );
                assert_eq!(
                    a[0].sat_pos_ecef_m.map(f64::to_bits),
                    b[0].sat_pos_ecef_m.map(f64::to_bits)
                );

                // Full forward observables agree too.
                let oa = predict(&sp3, gps(1), rx, q, options).expect("sp3 predict");
                let ob = predict(&samples, gps(1), rx, q, options).expect("samples predict");
                assert_eq!(
                    oa.geometric_range_m.to_bits(),
                    ob.geometric_range_m.to_bits()
                );
                assert_eq!(oa.doppler_hz.to_bits(), ob.doppler_hz.to_bits());
            }
        }
    }

    #[test]
    fn from_samples_tracks_real_fixture_to_sub_micron() {
        // On a real product the km -> meters map is not injective (see module
        // docs), so a meters-carrying sample reconstructs to the correctly-rounded
        // km, within <= 1 ULP of the fit node. This bounds the resulting
        // divergence far below any physical threshold and confirms the vast
        // majority of grid points are still byte-identical.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).expect("read fixture");
        let sp3 = Sp3::parse(&bytes).expect("parse fixture");
        let samples =
            PreciseEphemerisSamples::from_samples(sp3.precise_ephemeris_samples()).expect("source");

        let epochs = sp3.epochs_j2000_seconds();
        let sats: Vec<_> = sp3.satellites().to_vec();
        let mut compared = 0u64;
        let mut byte_identical = 0u64;
        let mut max_abs_diff_m = 0.0f64;

        for &sat in sats.iter().take(20) {
            for w in epochs.windows(2) {
                for q in [w[0], 0.5 * (w[0] + w[1])] {
                    let (Ok(a), Ok(b)) = (
                        sp3.position_at_j2000_seconds(sat, q),
                        samples.position_at_j2000_seconds(sat, q),
                    ) else {
                        continue;
                    };
                    let pa = a.position.as_array();
                    let pb = b.position.as_array();
                    for k in 0..3 {
                        compared += 1;
                        if pa[k].to_bits() == pb[k].to_bits() {
                            byte_identical += 1;
                        }
                        max_abs_diff_m = max_abs_diff_m.max((pa[k] - pb[k]).abs());
                    }
                }
            }
        }

        assert!(compared > 0);
        assert!(
            max_abs_diff_m < 1.0e-6,
            "max divergence {max_abs_diff_m:e} m exceeds sub-micron bound"
        );
        assert!(
            byte_identical * 100 >= compared * 90,
            "expected the vast majority byte-identical, got {byte_identical}/{compared}"
        );
    }
}
