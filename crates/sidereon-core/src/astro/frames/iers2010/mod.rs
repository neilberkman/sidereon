//! IERS 2010 Earth orientation from a caller-supplied daily EOP series.
//!
//! The reference is Orekit 13.1.10's raw CIRF, TIRF and ITRF providers and
//! Hipparchus 4.0.3. The IAU 2006/2000A celestial pole and CIO series are IERS
//! tables 5.2a, 5.2b and 5.2d. Table 5.2e supplies the GMST polynomial used
//! with TT in the subdaily EOP model (tables 8.2 and 8.3). The accompanying
//! term order preserves the compiled Poisson series' accumulation order.
//! The Orekit and Hipparchus notices are in the crate's third-party notices.
//!
//! Raw evaluation is date-deterministic. Orekit's factory frames interpolate
//! and shift CIRF rotations, and interpolate tidal corrections on a separate
//! grid. Each cache slot anchors its grid to the epoch that initializes it.
//! Across the reference epochs the factory and raw orientations differ by at
//! most 1.36e-13 rad in rotation and 7.17e-12 rad/s in rotation rate.
//!
//! Strict validity requires the epoch to lie in the closed EOP interval.
//! Permissive evaluation uses Orekit's direct subdaily pole, UT1 and LOD
//! corrections with a zero daily baseline and zero celestial pole offsets.
//! An empty series uses that fallback at every epoch and reports
//! [`DegradeReason::BeforeCoverage`]. With one to three entries, interpolation uses the
//! available samples. At the origin and on the negative X axis, the CIRF
//! quaternion is the continuous limit of Orekit's formula. The CIP must lie
//! on the visible unit hemisphere. State transforms use exponent-separated
//! arithmetic if an intermediate overflows. TAI, TT, UT1 and TDB dates must
//! fit in the signed 64-bit whole seconds of Orekit's date representation.
//! The TDB entry point uses the crate's TT–TDB convention. UTC MJD with
//! seconds of day preserves precision across distant epochs and leap seconds.
//! Leap labels in a positive UTC step are measured as elapsed TAI seconds
//! from 23:59:59, including during the pre-1972 UTC drifts.

pub(crate) mod date;
pub(crate) mod rotation;
mod series;

use self::date::Offset;
use self::rotation::{Rotation, Transform};
use super::orientation::{invalid_input, EarthOrientation, EarthOrientationProvider};
use super::transforms::{FrameTransformError, PolarMotion};
use crate::astro::events::eclipse::fast_math::linear_combination_3;
use crate::astro::time::scales::{tdb_minus_tt_seconds, TimeScales};
use crate::astro::time::{DegradeReason, ValidityMode};

const DAY: f64 = 86_400.0;
const CENTURY: f64 = 36_525.0 * DAY;
const ARCSEC_TO_RAD: f64 = std::f64::consts::TAU / 1_296_000.0;
const TT_MINUS_TAI: Offset = Offset::millis(32_184);
const ERA_0: f64 = std::f64::consts::TAU * 0.7790572732640;
const ERA_1A: f64 = std::f64::consts::TAU / DAY;
const ERA_1B: f64 = ERA_1A * 0.00273781191135448;

fn earth_rotation_angle(tai: f64, scale_minus_tai: f64) -> f64 {
    let whole_days = (tai as i128 / 86_400 * 86_400) as f64;
    let remainder = (tai - whole_days) + scale_minus_tai;
    ERA_0 + ERA_1A * remainder + ERA_1B * (whole_days + remainder)
}

fn date_range(field: &'static str) -> FrameTransformError {
    invalid_input(field, "outside the representable date range")
}

/// Daily Earth orientation at midnight UTC, with IAU 2006/2000A corrections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iers2010EopSample {
    /// UTC Modified Julian Date.
    pub mjd_utc: i32,
    /// Pole x coordinate, arcseconds.
    pub xp_arcsec: f64,
    /// Pole y coordinate, arcseconds.
    pub yp_arcsec: f64,
    /// UT1 minus UTC, seconds.
    pub ut1_minus_utc_s: f64,
    /// Excess length of day, seconds.
    pub lod_s: f64,
    /// Celestial intermediate pole X correction, arcseconds.
    pub dx_arcsec: f64,
    /// Celestial intermediate pole Y correction, arcseconds.
    pub dy_arcsec: f64,
    /// Optional pole x and y rates in radians per second. Missing rates are
    /// differentiated from the neighboring daily pole coordinates.
    pub pole_rates_rad_s: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Sample {
    epoch: Offset,
    values: [f64; 6],
    pole_rates: Option<[f64; 2]>,
}

/// CIO-based GCRF to ITRF provider using IERS 2010 daily EOP and subdaily tides.
///
/// An input or derived date outside the signed 64-bit whole-second range
/// returns [`FrameTransformError::InvalidInput`].
#[derive(Debug, Clone, PartialEq)]
pub struct Iers2010EarthOrientationProvider {
    samples: Box<[Sample]>,
    validity: ValidityMode,
}

struct Evaluation {
    #[cfg(all(test, sidereon_repo_tests))]
    daily: [f64; 6],
    corrected: [f64; 6],
    #[cfg(all(test, sidereon_repo_tests))]
    xys: [f64; 3],
    corrected_xys: [f64; 3],
    era: f64,
    sp: f64,
    ut1_minus_tai: f64,
    degraded: Option<DegradeReason>,
}

impl Evaluation {
    fn transform(&self) -> Result<Transform, FrameTransformError> {
        let [x, y, _] = self.corrected_xys;
        if x.abs().max(y.abs()) > 1.0 || linear_combination_3(x, x, y, y, -1.0, 1.0) > 0.0 {
            return Err(invalid_input(
                "celestial pole",
                "X and Y must lie in the unit disc",
            ));
        }
        let rotation = Rotation::celestial(self.corrected_xys);
        let transform = Transform::new(rotation, self.era, self.sp, self.corrected);
        if transform
            .rotation
            .0
            .into_iter()
            .chain(transform.rate)
            .any(|v| !v.is_finite())
        {
            return Err(invalid_input(
                "orientation",
                "rotation and rate must be finite",
            ));
        }
        Ok(transform)
    }
}

impl Iers2010EarthOrientationProvider {
    /// Build a strict provider from finite daily EOP values with strictly
    /// increasing MJDs. An empty series can be queried in permissive mode.
    pub fn new(samples: Vec<Iers2010EopSample>) -> Result<Self, FrameTransformError> {
        if samples
            .windows(2)
            .any(|pair| pair[0].mjd_utc >= pair[1].mjd_utc)
        {
            return Err(invalid_input("samples", "MJDs must be strictly increasing"));
        }
        let mut samples: Vec<_> = samples
            .into_iter()
            .map(|sample| Sample {
                epoch: date::utc(sample.mjd_utc, Offset::seconds(0)),
                values: [
                    ARCSEC_TO_RAD * sample.xp_arcsec,
                    ARCSEC_TO_RAD * sample.yp_arcsec,
                    sample.ut1_minus_utc_s,
                    sample.lod_s,
                    ARCSEC_TO_RAD * sample.dx_arcsec,
                    ARCSEC_TO_RAD * sample.dy_arcsec,
                ],
                pole_rates: sample.pole_rates_rad_s,
            })
            .collect();
        if samples.iter().any(|sample| {
            sample
                .values
                .into_iter()
                .chain(sample.pole_rates.into_iter().flatten())
                .any(|v| !v.is_finite())
        }) {
            return Err(invalid_input(
                "samples",
                "EOP and supplied pole rates must be finite",
            ));
        }
        for index in 0..samples.len() {
            if samples[index].pole_rates.is_none() {
                let epoch = samples[index].epoch;
                let mut cubic = Hermite::<2>::default();
                for neighbor in neighbors(&samples, epoch, 4) {
                    let [x, y, ..] = neighbor.values;
                    cubic.add(neighbor.epoch.minus(epoch).as_seconds(), [x, y], None);
                }
                samples[index].pole_rates = Some(cubic.evaluate().1);
            }
        }
        Ok(Self {
            samples: samples.into_boxed_slice(),
            validity: ValidityMode::Strict,
        })
    }

    /// Select the validity policy for epochs outside the daily EOP series.
    pub const fn with_validity(mut self, validity: ValidityMode) -> Self {
        self.validity = validity;
        self
    }

    /// Evaluate at a UTC MJD and seconds within that UTC day.
    ///
    /// On a positive leap day, values starting at 86400 address the leap
    /// second or fractional step. This entry point preserves subsecond
    /// precision independently of the number of elapsed days since J2000.
    pub fn orientation_at_utc_mjd(
        &self,
        mjd_utc: i32,
        seconds_in_day: f64,
    ) -> Result<EarthOrientation, FrameTransformError> {
        if !(0.0..date::end_of_utc_day(mjd_utc)).contains(&seconds_in_day) {
            return Err(invalid_input("seconds_in_day", "outside the UTC day"));
        }
        let seconds = Offset::from_seconds(seconds_in_day)
            .ok_or_else(|| invalid_input("seconds_in_day", "outside the UTC day"))?;
        self.orientation(date::utc(mjd_utc, seconds))
    }

    fn orientation(&self, epoch: Offset) -> Result<EarthOrientation, FrameTransformError> {
        epoch.whole_seconds().ok_or_else(|| date_range("TAI"))?;
        let tt = epoch.plus(TT_MINUS_TAI);
        let (day, day_start) = tt.day_and_start().ok_or_else(|| date_range("TT"))?;
        let tdb_minus_tt = tdb_minus_tt_seconds(tt.as_seconds());
        Offset::from_seconds(tdb_minus_tt)
            .and_then(|offset| tt.checked_plus(offset))
            .and_then(Offset::whole_seconds)
            .ok_or_else(|| date_range("TDB"))?;
        let evaluation = self.evaluate(epoch)?;
        let transform = evaluation.transform()?;
        let tt_fraction = tt.minus(day_start).as_seconds() / DAY;
        let jd_whole = 2_451_545.0 + day as f64;
        let ut1_fraction = (epoch.minus(day_start).as_seconds() + evaluation.ut1_minus_tai) / DAY;
        let tdb_fraction = tt_fraction + tdb_minus_tt / DAY;
        let scales = TimeScales {
            jd_whole,
            ut1_fraction,
            tt_fraction,
            tdb_fraction,
            jd_ut1: jd_whole + ut1_fraction,
            jd_tt: jd_whole + tt_fraction,
            jd_tdb: jd_whole + tdb_fraction,
            ut1_degraded: evaluation.degraded,
        };
        let [xp, yp, ..] = evaluation.corrected;
        Ok(EarthOrientation::from_iers2010(
            scales,
            PolarMotion::from_radians(xp, yp)?,
            transform.rotation,
            transform.rotation.matrix(),
            transform.rate,
        ))
    }

    fn evaluate(&self, epoch: Offset) -> Result<Evaluation, FrameTransformError> {
        let degraded = match self.samples.first().zip(self.samples.last()) {
            Some((first, last)) if (first.epoch..=last.epoch).contains(&epoch) => None,
            Some((_, last)) if epoch > last.epoch => Some(DegradeReason::AfterCoverage),
            _ => Some(DegradeReason::BeforeCoverage),
        };
        if let (ValidityMode::Strict, Some(reason)) = (self.validity, degraded) {
            return Err(FrameTransformError::Ut1OutsideCoverage { reason });
        }

        let tai = epoch.as_seconds();
        let tt = epoch.plus(TT_MINUS_TAI).as_seconds();
        let (xys_plus_xy, tide) = series::evaluate(tt, tai);
        let daily = if degraded.is_some() {
            [0.0; 6]
        } else {
            self.interpolate(epoch)
        };
        let mut corrected = daily;
        if degraded.is_some() {
            corrected[..4].copy_from_slice(&tide);
        } else {
            for (value, correction) in corrected.iter_mut().zip(tide) {
                *value += correction;
            }
        }
        let [x, y, s_plus_xy] = xys_plus_xy;
        let corrected_x = x + corrected[4];
        let corrected_y = y + corrected[5];
        if corrected.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("EOP", "interpolated values must be finite"));
        }
        let tai_minus_utc = date::tai_minus_utc(epoch);
        let ut1_offset = Offset::from_seconds(corrected[2])
            .and_then(|offset| offset.checked_minus(tai_minus_utc))
            .ok_or_else(|| date_range("UT1"))?;
        epoch
            .checked_plus(ut1_offset)
            .and_then(Offset::whole_seconds)
            .ok_or_else(|| date_range("UT1"))?;
        let ut1_minus_tai = if corrected[2].abs() < i64::MAX as f64 {
            ut1_offset.as_seconds()
        } else {
            corrected[2] - tai_minus_utc.as_seconds()
        };
        Ok(Evaluation {
            #[cfg(all(test, sidereon_repo_tests))]
            daily,
            corrected,
            #[cfg(all(test, sidereon_repo_tests))]
            xys: [x, y, s_plus_xy - x * y / 2.0],
            corrected_xys: [
                corrected_x,
                corrected_y,
                s_plus_xy - corrected_x * corrected_y / 2.0,
            ],
            era: earth_rotation_angle(tai, ut1_minus_tai),
            sp: (-47e-6 * ARCSEC_TO_RAD) * (tt / CENTURY),
            ut1_minus_tai,
            degraded,
        })
    }

    fn interpolate(&self, epoch: Offset) -> [f64; 6] {
        let daily = neighbors(&self.samples, epoch, 4);
        let mut values = Hermite::<4>::default();
        let mut before_leap = true;
        for sample in daily {
            let [_, _, mut dut1, lod, dx, dy] = sample.values;
            if dut1 - daily[0].values[2] > 0.9 {
                dut1 -= 1.0;
                before_leap &= epoch < sample.epoch.minus(Offset::seconds(1));
            }
            values.add(
                sample.epoch.minus(epoch).as_seconds(),
                [dut1, lod, dx, dy],
                None,
            );
        }
        let [mut dut1, lod, dx, dy] = values.evaluate().0;
        if !before_leap {
            dut1 += 1.0;
        }

        let mut pole = Hermite::<2>::default();
        for sample in neighbors(&self.samples, epoch, 2) {
            let [x, y, ..] = sample.values;
            pole.add(
                sample.epoch.minus(epoch).as_seconds(),
                [x, y],
                sample.pole_rates,
            );
        }
        let [x, y] = pole.evaluate().0;
        [x, y, dut1, lod, dx, dy]
    }
}

impl EarthOrientationProvider for Iers2010EarthOrientationProvider {
    fn orientation_at_tdb_seconds(
        &self,
        epoch: f64,
    ) -> Result<EarthOrientation, FrameTransformError> {
        let tdb = Offset::from_seconds(epoch).ok_or_else(|| date_range("epoch_tdb_seconds"))?;
        tdb.whole_seconds()
            .ok_or_else(|| date_range("epoch_tdb_seconds"))?;
        let first = epoch - tdb_minus_tt_seconds(epoch);
        let tt = Offset::from_seconds(tdb_minus_tt_seconds(first))
            .and_then(|offset| tdb.checked_minus(offset))
            .ok_or_else(|| date_range("TT"))?;
        tt.whole_seconds().ok_or_else(|| date_range("TT"))?;
        let tai = tt.minus(TT_MINUS_TAI);
        // A rounded absolute TDB second can represent the endpoint even when
        // its inverse is just outside the closed interval.
        for endpoint in self.samples.first().into_iter().chain(self.samples.last()) {
            let tt = endpoint.epoch.plus(TT_MINUS_TAI).as_seconds();
            if epoch == tt + tdb_minus_tt_seconds(tt) {
                return self.orientation(endpoint.epoch);
            }
        }
        self.orientation(tai)
    }

    fn ut1_validity(&self) -> ValidityMode {
        self.validity
    }
}

fn neighbors(samples: &[Sample], date: Offset, count: usize) -> &[Sample] {
    let count = count.min(samples.len());
    let lower = samples
        .partition_point(|sample| sample.epoch <= date)
        .saturating_sub(1);
    let start = lower
        .saturating_sub((count - 1) / 2)
        .min(samples.len() - count);
    &samples[start..start + count]
}

/// Newton-form Hermite interpolation, with insertion and evaluation in the
/// order of Hipparchus HermiteInterpolator.
struct Hermite<const N: usize> {
    len: usize,
    x: [f64; 4],
    top: [[f64; N]; 4],
    bottom: [[f64; N]; 4],
}

impl<const N: usize> Default for Hermite<N> {
    fn default() -> Self {
        Self {
            len: 0,
            x: [0.0; 4],
            top: [[0.0; N]; 4],
            bottom: [[0.0; N]; 4],
        }
    }
}

impl<const N: usize> Hermite<N> {
    fn add(&mut self, x: f64, value: [f64; N], rate: Option<[f64; N]>) {
        for (i, values) in [Some(value), rate].into_iter().flatten().enumerate() {
            let n = self.len;
            self.bottom.copy_within(n - i..n, n - i + 1);
            self.bottom[n - i] = values;
            let mut bottom = values;
            for j in (0..n - i).rev() {
                let inv = 1.0 / (x - self.x[j]);
                for (current, previous) in bottom.iter_mut().zip(&mut self.bottom[j]) {
                    *previous = inv * (*current - *previous);
                    *current = *previous;
                }
            }
            self.top[n] = bottom;
            self.x[n] = x;
            self.len += 1;
        }
    }

    fn evaluate(&self) -> ([f64; N], [f64; N]) {
        let mut value = [0.0; N];
        let mut derivative = [0.0; N];
        let mut factor = 1.0;
        let mut derivative_factor = 0.0;
        for (&x, divided_difference) in self.x[..self.len].iter().zip(&self.top) {
            let delta = 0.0 - x;
            for (i, &difference) in divided_difference.iter().enumerate() {
                value[i] += difference * factor;
                derivative[i] += difference * derivative_factor;
            }
            derivative_factor *= delta;
            derivative_factor += factor;
            factor *= delta;
        }
        (value, derivative)
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
