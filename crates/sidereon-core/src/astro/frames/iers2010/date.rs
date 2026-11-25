use crate::astro::time::scales::TimeTables;

const ATTOS: i128 = 1_000_000_000_000_000_000;
const NANOS: i128 = 1_000_000_000;
const J2000_MJD: i32 = 51_544;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Offset(i128);

impl Offset {
    pub(super) const fn seconds(seconds: i64) -> Self {
        Self(seconds as i128 * ATTOS)
    }

    pub(super) const fn millis(ms: i64) -> Self {
        Self(ms as i128 * (ATTOS / 1000))
    }

    pub(super) fn from_seconds(seconds: f64) -> Option<Self> {
        if !seconds.is_finite() {
            return None;
        }
        // TimeOffset(double) rounds relative to the nearest whole second.
        let nearest = seconds.round_ties_even();
        let scaled = (seconds - nearest) * ATTOS as f64;
        let floor = scaled.floor();
        let atto = floor as i128 + i128::from(scaled - floor >= 0.5);
        (nearest as i128)
            .checked_mul(ATTOS)?
            .checked_add(atto)
            .map(Self)
    }

    pub(super) fn whole_seconds(self) -> Option<i64> {
        i64::try_from(self.0.div_euclid(ATTOS)).ok()
    }

    pub(super) fn day_and_start(self) -> Option<(i64, Self)> {
        let day = self.whole_seconds()?.div_euclid(86_400);
        // The midnight of a representable instant can precede i64::MIN.
        Some((day, Self(i128::from(day) * 86_400 * ATTOS)))
    }

    pub(super) fn as_seconds(self) -> f64 {
        let mut seconds = self.0.div_euclid(ATTOS);
        let mut sub = self.0.rem_euclid(ATTOS);
        if sub > ATTOS / 2 {
            seconds += 1;
            sub -= ATTOS;
        }
        seconds as f64 + sub as f64 / ATTOS as f64
    }

    pub(super) fn plus(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    pub(super) fn minus(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    pub(super) fn checked_plus(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    pub(super) fn checked_minus(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(Self)
    }

    fn drift(self, slope: i128, divisor: i128) -> Self {
        // Keep TimeOffset.divide's quotient and remainder order for negative
        // durations, where the final division may truncate toward zero.
        let product = self.0 * slope;
        let seconds = product.div_euclid(ATTOS);
        let attos = product.rem_euclid(ATTOS);
        let whole = seconds / divisor;
        let remainder = seconds % divisor;
        Self(
            whole * ATTOS
                + remainder * (ATTOS / divisor)
                + (attos + remainder * (ATTOS % divisor)) / divisor,
        )
    }

    fn tai_noon_relative(mjd: i32) -> Self {
        let days = i128::from(mjd) - i128::from(J2000_MJD);
        Self((days * 86_400 - 43_200) * ATTOS)
    }
}

#[derive(Clone, Copy)]
struct UtcSegment {
    start: i32,
    reference: i32,
    base: Offset,
    slope: i128,
}

impl UtcSegment {
    const fn historical(start: i32, reference: i32, base_100ns: i64, slope: i128) -> Self {
        Self {
            start,
            reference,
            base: Offset(base_100ns as i128 * 100_000_000_000),
            slope,
        }
    }

    fn utc_to_tai_offset(self, utc: Offset) -> Offset {
        self.base.plus(
            utc.minus(Offset::tai_noon_relative(self.reference))
                .drift(self.slope, NANOS),
        )
    }

    fn tai_to_utc(self, tai: Offset) -> Offset {
        let reference = Offset::tai_noon_relative(self.reference).plus(self.base);
        self.base
            .plus(tai.minus(reference).drift(self.slope, NANOS + self.slope))
    }
}

const PRE_1972: [UtcSegment; 13] = [
    UtcSegment::historical(37300, 37300, 14228180, 15),
    UtcSegment::historical(37512, 37300, 13728180, 15),
    UtcSegment::historical(37665, 37665, 18458580, 13),
    UtcSegment::historical(38334, 37665, 19458580, 13),
    UtcSegment::historical(38395, 38761, 32401300, 15),
    UtcSegment::historical(38486, 38761, 33401300, 15),
    UtcSegment::historical(38639, 38761, 34401300, 15),
    UtcSegment::historical(38761, 38761, 35401300, 15),
    UtcSegment::historical(38820, 38761, 36401300, 15),
    UtcSegment::historical(38942, 38761, 37401300, 15),
    UtcSegment::historical(39004, 38761, 38401300, 15),
    UtcSegment::historical(39126, 39126, 43131700, 30),
    UtcSegment::historical(39887, 39126, 42131700, 30),
];

fn segments() -> impl DoubleEndedIterator<Item = UtcSegment> {
    PRE_1972.into_iter().chain(
        TimeTables::embedded()
            .leap_seconds
            .iter()
            .map(|step| UtcSegment {
                start: step.mjd,
                reference: step.mjd,
                base: Offset::seconds(step.tai_utc as i64),
                slope: 0,
            }),
    )
}

pub(super) fn utc(mjd: i32, seconds_in_day: Offset) -> Offset {
    let (civil, shift) = if seconds_in_day >= Offset::seconds(86_400) {
        // During a positive step, extend 23:59:59 by elapsed TAI seconds.
        let last_second = Offset::seconds(86_399);
        (last_second, seconds_in_day.minus(last_second))
    } else {
        (seconds_in_day, Offset::seconds(0))
    };
    let epoch = Offset::tai_noon_relative(mjd).plus(civil);
    let tai = segments()
        .rev()
        .find(|segment| segment.start <= mjd)
        .map_or(epoch, |segment| {
            epoch.plus(segment.utc_to_tai_offset(epoch))
        });
    tai.plus(shift)
}

pub(super) fn end_of_utc_day(mjd: i32) -> f64 {
    let Some(next) = mjd.checked_add(1) else {
        return 86_400.0;
    };
    let midnight = Offset::tai_noon_relative(next);
    let mut offsets = segments().rev().filter(|segment| segment.start <= next);
    match offsets.next() {
        Some(current) if current.start == next => {
            let old = offsets.next().map_or(Offset::seconds(0), |previous| {
                previous.utc_to_tai_offset(midnight)
            });
            let leap = current
                .utc_to_tai_offset(midnight)
                .minus(old)
                .drift(NANOS, NANOS + current.slope);
            86_400.0 + leap.as_seconds()
        }
        _ => 86_400.0,
    }
}

pub(super) fn tai_minus_utc(epoch: Offset) -> Offset {
    let mut previous: Option<UtcSegment> = None;
    for current in segments() {
        let midnight = Offset::tai_noon_relative(current.start);
        let leap_start = previous.map_or(midnight, |segment| {
            midnight.plus(segment.utc_to_tai_offset(midnight))
        });
        if epoch < leap_start {
            break;
        }
        previous = Some(current);
    }
    previous.map_or(Offset::seconds(0), |segment| segment.tai_to_utc(epoch))
}
