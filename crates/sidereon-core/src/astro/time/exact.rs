//! Exact time arithmetic: epochs held without rounding, and values rounded once.
//!
//! A civil label states its second as a decimal, and a split Julian date or a
//! double of seconds holds a binary fraction. Neither form holds the other
//! exactly, so a conversion that rounds at each step (the clock-field sum, then
//! the division by the day) can land one unit in the last place away from the
//! value the label states, and a difference of two rounded epochs is not the
//! difference of the labels.
//!
//! [`ExactEpoch`] holds an epoch as whole seconds plus whole attoseconds, and
//! any digits below the attosecond as an exact remainder, the role RTKLIB's
//! `gtime_t` plays, so the difference of two epochs is exact and is rounded
//! once. The crate-internal [`ExactSeconds`] holds any decimal or
//! binary value exactly, as an integer over a power of two times a power of
//! ten, and rounds once, to the nearest `f64` with ties to even, when a result
//! leaves it.

use std::cmp::Ordering;
use std::sync::Arc;

use smallvec::SmallVec;

use super::civil::days_in_month;
use super::scales::julian_day_number;

/// Attoseconds in one second.
const ATTOSECONDS_PER_SECOND: u64 = 1_000_000_000_000_000_000;

/// Seconds in one day.
const SECONDS_PER_DAY: i64 = 86_400;

/// Integer Julian Day Number of the J2000 calendar day (2000-01-01).
const J2000_JULIAN_DAY_NUMBER: i64 = 2_451_545;

/// Seconds from civil midnight of the J2000 day to the J2000 epoch (noon).
const J2000_NOON_OFFSET_S: i64 = 43_200;

/// An epoch held exactly: whole seconds since J2000 (2000-01-01 12:00:00), the
/// fraction of a second as a whole number of attoseconds, and any remainder
/// below the attosecond as an exact decimal.
///
/// This is the role RTKLIB's `gtime_t` (whole seconds plus a fraction) plays.
/// The fraction is an integer here, so every decimal second a format states to
/// at most eighteen fractional digits is held on the attosecond grid, and the
/// few seconds that state finer digits keep them in the remainder: no decimal
/// second is rounded. The difference of two epochs
/// ([`ExactEpoch::seconds_since`]) is exact before it is rounded once to an
/// `f64`. The epoch is in the caller's own time scale; no leap second is
/// applied, the same no-leap contract as [`super::civil::j2000_seconds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExactEpoch {
    seconds: i64,
    attoseconds: u64,
    residue: Residue,
}

/// The part of an epoch below its nearest attosecond: `digits * 10^-places`
/// attoseconds, in `[-1/2, 1/2)`. `digits` carries no trailing zero, so equal
/// remainders are equal fields; zero is `(0, 0)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
struct Residue {
    digits: i64,
    places: u16,
}

impl Residue {
    const ZERO: Self = Self {
        digits: 0,
        places: 0,
    };

    fn is_zero(self) -> bool {
        self.digits == 0
    }

    /// The remainder in seconds, exactly.
    fn exact_seconds(self) -> ExactSeconds {
        ExactSeconds::from_decimal(i128::from(self.digits), 18 + u32::from(self.places))
    }
}

impl PartialOrd for ExactEpoch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactEpoch {
    /// Time order. The attosecond count is the nearest one to the epoch and
    /// the remainder lies in `[-1/2, 1/2)` attosecond, so a later count is a
    /// later epoch; equal counts are ordered by their remainders.
    fn cmp(&self, other: &Self) -> Ordering {
        (self.seconds, self.attoseconds)
            .cmp(&(other.seconds, other.attoseconds))
            .then_with(|| {
                if self.residue == other.residue {
                    Ordering::Equal
                } else {
                    self.residue
                        .exact_seconds()
                        .sub(&other.residue.exact_seconds())
                        .sign()
                }
            })
    }
}

impl ExactEpoch {
    /// Attoseconds in one second, the resolution of [`Self::attoseconds`].
    pub const ATTOSECONDS_PER_SECOND: u64 = ATTOSECONDS_PER_SECOND;

    /// The J2000 epoch, 2000-01-01 12:00:00.
    pub const J2000: Self = Self {
        seconds: 0,
        attoseconds: 0,
        residue: Residue::ZERO,
    };

    /// An epoch `seconds` whole seconds and `attoseconds` attoseconds after
    /// J2000. `None` when `attoseconds` is not below
    /// [`Self::ATTOSECONDS_PER_SECOND`].
    #[must_use]
    pub fn new(seconds: i64, attoseconds: u64) -> Option<Self> {
        (attoseconds < ATTOSECONDS_PER_SECOND).then_some(Self {
            seconds,
            attoseconds,
            residue: Residue::ZERO,
        })
    }

    /// The epoch stated by a finite number of seconds since J2000. The value
    /// is read as its shortest decimal, like the second field of a civil
    /// label.
    #[must_use]
    pub fn from_j2000_seconds(seconds: f64) -> Option<Self> {
        let (attoseconds, residue) = attoseconds_of_shortest_decimal(seconds)?;
        let mut epoch = Self::from_total_attoseconds(attoseconds)?;
        epoch.residue = residue;
        Some(epoch)
    }

    /// The exact binary value of a finite J2000-seconds `f64` as an epoch query.
    #[must_use]
    pub fn from_binary_j2000_seconds(seconds: f64) -> Option<ExactEpochQuery> {
        Some(ExactEpochQuery {
            epoch: Self::J2000,
            offset: Arc::new(ExactSeconds::from_f64(seconds)?),
        })
    }

    /// Begin an exact query at this epoch, with no offset.
    #[must_use]
    pub fn query(self) -> ExactEpochQuery {
        ExactEpochQuery {
            epoch: self,
            offset: Arc::new(ExactSeconds::from_integer(0)),
        }
    }

    /// `self` less `seconds`, with `seconds` read as its shortest decimal.
    /// Use for decimal values stated by a caller, not computed binary offsets;
    /// use [`ExactEpochQuery::checked_sub_binary_seconds`] for those.
    #[must_use]
    pub fn checked_sub_seconds(self, seconds: f64) -> Option<Self> {
        self.checked_offset_seconds(-seconds)
    }

    /// `self` plus `seconds`, with `seconds` read as its shortest decimal.
    /// Use for decimal values stated by a caller, not computed binary offsets;
    /// use [`ExactEpochQuery::checked_add_binary_seconds`] for those.
    #[must_use]
    pub fn checked_add_seconds(self, seconds: f64) -> Option<Self> {
        self.checked_offset_seconds(seconds)
    }

    fn checked_offset_seconds(self, offset_s: f64) -> Option<Self> {
        let (offset_attoseconds, offset_residue) = attoseconds_of_shortest_decimal(offset_s)?;
        let attoseconds = self.total_attoseconds().checked_add(offset_attoseconds)?;
        if self.residue.is_zero() || offset_residue.is_zero() {
            let residue = if self.residue.is_zero() {
                offset_residue
            } else {
                self.residue
            };
            let mut result = Self::from_total_attoseconds(attoseconds)?;
            result.residue = residue;
            return Some(result);
        }
        let places = self.residue.places.max(offset_residue.places);
        let residue_sum = if self.residue.places == offset_residue.places {
            i128::from(self.residue.digits).checked_add(i128::from(offset_residue.digits))?
        } else {
            let scale_residue = |residue: Residue| {
                i128::from(residue.digits)
                    .checked_mul(10_i128.checked_pow(u32::from(places - residue.places))?)
            };
            scale_residue(self.residue)?.checked_add(scale_residue(offset_residue)?)?
        };
        let (carry, residue_digits) = if places > 38 {
            (0, residue_sum)
        } else {
            let denominator = 10_i128.checked_pow(u32::from(places))?;
            let carry = residue_sum
                .checked_add(denominator / 2)?
                .div_euclid(denominator);
            (
                carry,
                residue_sum.checked_sub(carry.checked_mul(denominator)?)?,
            )
        };
        let total = attoseconds.checked_add(carry)?;
        let mut result = Self::from_total_attoseconds(total)?;
        let mut digits = i64::try_from(residue_digits).ok()?;
        let mut residue_places = places;
        while digits != 0 && residue_places > 0 && digits % 10 == 0 {
            digits /= 10;
            residue_places -= 1;
        }
        if digits == 0 {
            residue_places = 0;
        }
        result.residue = Residue {
            digits,
            places: residue_places,
        };
        Some(result)
    }

    /// The epoch a civil label states, in the label's own time scale.
    ///
    /// The second is read as the shortest decimal that reads back to the given
    /// `f64` (`0.1` is one tenth of a second), exactly as a label's seconds
    /// field states it, and every digit of it is kept: digits below the
    /// attosecond, which only a second below 0.1 written with more than
    /// eighteen fractional digits carries, are held in the remainder
    /// ([`Self::sub_attosecond`]). The date is proleptic Gregorian for every
    /// year. The clock fields are not range-checked: an hour of 24 is the next
    /// midnight, as in [`super::civil::j2000_seconds`]. `None` for a date that
    /// does not exist (a month outside 1 through 12, or a day outside the
    /// month), for a non-finite second, which states no epoch, and for an
    /// epoch whose whole seconds since J2000 do not fit an `i64` (beyond about
    /// 2.9e11 years), which this type cannot hold.
    #[must_use]
    pub fn from_civil(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> Option<Self> {
        if !(1..=12).contains(&month)
            || !(1..=days_in_month(i64::from(year), i64::from(month))).contains(&i64::from(day))
        {
            return None;
        }
        let (second, residue) = attoseconds_of_shortest_decimal(second)?;
        let days = julian_day_number(year, month, day) - J2000_JULIAN_DAY_NUMBER;
        let clock_seconds = i128::from(days) * i128::from(SECONDS_PER_DAY)
            - i128::from(J2000_NOON_OFFSET_S)
            + i128::from(hour) * 3_600
            + i128::from(minute) * 60;
        let total = clock_seconds
            .checked_mul(i128::from(ATTOSECONDS_PER_SECOND))?
            .checked_add(second)?;
        let mut epoch = Self::from_total_attoseconds(total)?;
        epoch.residue = residue;
        Some(epoch)
    }

    /// Whole seconds since J2000 of the nearest attosecond to the epoch,
    /// rounded toward negative infinity.
    #[must_use]
    pub fn whole_seconds(self) -> i64 {
        self.seconds
    }

    /// The fraction of a second after [`Self::whole_seconds`], in attoseconds:
    /// with it, the attosecond nearest to the epoch (halfway rounds up).
    #[must_use]
    pub fn attoseconds(self) -> u64 {
        self.attoseconds
    }

    /// The exact remainder of the epoch past [`Self::whole_seconds`] and
    /// [`Self::attoseconds`], as `(digits, places)`: `digits * 10^-places`
    /// attoseconds, in `[-1/2, 1/2)`. `(0, 0)` for an epoch on the attosecond
    /// grid.
    #[must_use]
    pub fn sub_attosecond(self) -> (i64, u16) {
        (self.residue.digits, self.residue.places)
    }

    /// Seconds from `earlier` to `self`: the `f64` nearest to the exact
    /// difference, ties to even. Two labels a whole number of seconds apart
    /// are exactly that many seconds apart.
    #[must_use]
    pub fn seconds_since(self, earlier: Self) -> f64 {
        if self.residue.is_zero() && earlier.residue.is_zero() {
            return nearest_ratio(
                self.total_attoseconds() - earlier.total_attoseconds(),
                u128::from(ATTOSECONDS_PER_SECOND),
            );
        }
        self.exact_seconds().sub(&earlier.exact_seconds()).to_f64()
    }

    /// Seconds since J2000: the `f64` nearest to the exact value, ties to even.
    #[must_use]
    pub fn j2000_seconds(self) -> f64 {
        self.seconds_since(Self::J2000)
    }

    /// Seconds within a positive whole-second period, rounded once from the exact
    /// epoch fraction. A value just below the period can round to the period itself.
    pub(crate) fn seconds_modulo(self, period_seconds: i64) -> Option<f64> {
        if period_seconds <= 0 {
            return None;
        }
        let whole = if self.attoseconds == 0 && self.residue.digits < 0 {
            (self.seconds.rem_euclid(period_seconds) - 1).rem_euclid(period_seconds)
        } else {
            self.seconds.rem_euclid(period_seconds)
        };
        let attoseconds = if self.attoseconds == 0 && self.residue.digits < 0 {
            i128::from(ATTOSECONDS_PER_SECOND)
        } else {
            i128::from(self.attoseconds)
        };
        let fraction = ExactSeconds::from_decimal(
            i128::from(whole) * i128::from(ATTOSECONDS_PER_SECOND) + attoseconds,
            18,
        );
        Some(
            if self.residue.is_zero() {
                fraction
            } else {
                fraction.add(&self.residue.exact_seconds())
            }
            .to_f64(),
        )
    }

    /// Split Julian date `(jd_whole, fraction)`: the civil midnight that opens
    /// the epoch's day (a `*.5` boundary) and the `f64` nearest to the exact
    /// fraction of the day after it, ties to even. The same pair
    /// [`super::civil::split_julian_date`] returns for a label within the day.
    #[must_use]
    pub fn split_julian_date(self) -> (f64, f64) {
        let per_day = i128::from(SECONDS_PER_DAY) * i128::from(ATTOSECONDS_PER_SECOND);
        let from_midnight = self.total_attoseconds()
            + i128::from(J2000_NOON_OFFSET_S) * i128::from(ATTOSECONDS_PER_SECOND);
        let mut day = from_midnight.div_euclid(per_day);
        let mut within = from_midnight.rem_euclid(per_day);
        if within == 0 && self.residue.digits < 0 {
            // Just before a midnight: the epoch belongs to the day that ends.
            day -= 1;
            within = per_day;
        }
        let jd_whole = (i128::from(J2000_JULIAN_DAY_NUMBER) + day) as f64 - 0.5;
        let fraction = if self.residue.is_zero() {
            nearest_ratio(within, per_day.unsigned_abs())
        } else {
            ExactSeconds::from_decimal(within, 18)
                .add(&self.residue.exact_seconds())
                .div_to_f64(SECONDS_PER_DAY as u64)
        };
        (jd_whole, fraction)
    }

    /// How the time from `earlier` to `self` compares with `seconds`,
    /// exactly, with `seconds` read as the shortest decimal that reads back to
    /// it, as a label's second is read: a 0.3 s threshold is three tenths of a
    /// second, not the double below them. An infinite `seconds` compares as
    /// infinity; `None` for NaN.
    pub(crate) fn compare_interval(self, earlier: Self, seconds: f64) -> Option<Ordering> {
        if seconds.is_nan() {
            return None;
        }
        let Some(threshold) = ExactSeconds::from_shortest_decimal(seconds) else {
            return Some(if seconds > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        };
        Some(
            self.exact_seconds()
                .sub(&earlier.exact_seconds())
                .sub(&threshold)
                .sign(),
        )
    }

    /// Whether `self` and `other` lie more than `seconds` apart, either way,
    /// compared exactly: `difference.abs() > seconds` on the exact difference
    /// and the decimal `seconds` states, so `false` for NaN and for positive
    /// infinity.
    pub(crate) fn interval_exceeds(self, other: Self, seconds: f64) -> bool {
        let (later, earlier) = if self >= other {
            (self, other)
        } else {
            (other, self)
        };
        later.compare_interval(earlier, seconds) == Some(Ordering::Greater)
    }

    /// The epoch as exact seconds since J2000.
    pub(crate) fn exact_seconds(self) -> ExactSeconds {
        if self.attoseconds == 0 && self.residue.is_zero() {
            return ExactSeconds::from_integer(i128::from(self.seconds));
        }
        let on_grid = ExactSeconds::from_decimal(self.total_attoseconds(), 18);
        if self.residue.is_zero() {
            on_grid
        } else {
            on_grid.add(&self.residue.exact_seconds())
        }
    }

    /// The epoch's attoseconds since J2000 on the grid, without the remainder.
    fn total_attoseconds(self) -> i128 {
        i128::from(self.seconds) * i128::from(ATTOSECONDS_PER_SECOND) + i128::from(self.attoseconds)
    }

    /// The epoch `total` attoseconds after J2000; `None` when its whole
    /// seconds do not fit an `i64`.
    fn from_total_attoseconds(total: i128) -> Option<Self> {
        let per_second = i128::from(ATTOSECONDS_PER_SECOND);
        Some(Self {
            seconds: i64::try_from(total.div_euclid(per_second)).ok()?,
            attoseconds: total.rem_euclid(per_second) as u64,
            residue: Residue::ZERO,
        })
    }
}

/// An exact epoch plus exact binary offsets computed by a caller.
///
/// Civil labels use [`ExactEpoch`]. Arithmetic results such as `pseudorange / c`
/// are binary `f64` values and must use this carrier rather than interpreting
/// those results as shortest-decimal labels.
#[derive(Debug, Clone)]
pub struct ExactEpochQuery {
    epoch: ExactEpoch,
    offset: Arc<ExactSeconds>,
}

impl PartialEq for ExactEpochQuery {
    fn eq(&self, other: &Self) -> bool {
        if self.epoch == other.epoch {
            if Arc::ptr_eq(&self.offset, &other.offset) {
                return true;
            }
            if self.offset.binary_places == other.offset.binary_places
                && self.offset.decimal_places == other.offset.decimal_places
            {
                return self.offset.same_value_at_same_denominator(&other.offset);
            }
            if let Some(equal) = self.offset.same_value_at_common_denominator(&other.offset) {
                return equal;
            }
        }
        self.equals_with_wide_arithmetic(other)
    }
}

impl Eq for ExactEpochQuery {}

impl ExactEpochQuery {
    pub(crate) fn shares_representation(&self, other: &Self) -> bool {
        self.epoch == other.epoch && Arc::ptr_eq(&self.offset, &other.offset)
    }

    #[cold]
    #[inline(never)]
    fn equals_with_wide_arithmetic(&self, other: &Self) -> bool {
        self.exact_seconds().sub(&other.exact_seconds()).sign() == Ordering::Equal
    }

    fn exact_seconds(&self) -> ExactSeconds {
        self.epoch.exact_seconds().add(&self.offset)
    }

    /// Add a finite binary `f64` offset exactly.
    #[must_use]
    pub fn checked_add_binary_seconds(mut self, seconds: f64) -> Option<Self> {
        self.offset = Arc::new(self.offset.add(&ExactSeconds::from_f64(seconds)?));
        Some(self)
    }

    /// Subtract a finite binary `f64` offset exactly.
    #[must_use]
    pub fn checked_sub_binary_seconds(mut self, seconds: f64) -> Option<Self> {
        self.offset = Arc::new(self.offset.sub(&ExactSeconds::from_f64(seconds)?));
        Some(self)
    }

    /// Seconds from `earlier` to this query, rounded once to `f64`.
    #[must_use]
    pub fn seconds_since(&self, earlier: ExactEpoch) -> f64 {
        if self.epoch == earlier {
            return self.offset.to_f64();
        }
        self.epoch
            .exact_seconds()
            .sub(&earlier.exact_seconds())
            .add(&self.offset)
            .to_f64()
    }

    /// Seconds from another query, including both exact offset expressions.
    #[must_use]
    pub fn seconds_since_query(&self, earlier: &Self) -> f64 {
        if self.epoch == earlier.epoch {
            return self.offset.sub(&earlier.offset).to_f64();
        }
        self.epoch
            .exact_seconds()
            .sub(&earlier.epoch.exact_seconds())
            .add(&self.offset)
            .sub(&earlier.offset)
            .to_f64()
    }

    /// Compare the exact elapsed interval from `earlier` with a shortest-decimal
    /// threshold. NaN has no ordering; infinities compare as infinite thresholds.
    pub(crate) fn compare_interval_query(&self, earlier: &Self, seconds: f64) -> Option<Ordering> {
        self.compare_interval_seconds(earlier.exact_seconds(), seconds)
    }

    pub(crate) fn compare_interval_binary_j2000_seconds(
        &self,
        earlier_j2000_s: f64,
        seconds: f64,
    ) -> Option<Ordering> {
        self.compare_interval_seconds(ExactSeconds::from_f64(earlier_j2000_s)?, seconds)
    }

    fn compare_interval_seconds(&self, earlier: ExactSeconds, seconds: f64) -> Option<Ordering> {
        if seconds.is_nan() {
            return None;
        }
        let Some(threshold) = ExactSeconds::from_shortest_decimal(seconds) else {
            return Some(if seconds > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        };
        Some(self.exact_seconds().sub(&earlier).sub(&threshold).sign())
    }

    /// Compare the exact absolute distances from this query to two other queries.
    pub(crate) fn compare_distance_to(&self, first: &Self, second: &Self) -> Ordering {
        let first_delta = self.exact_seconds().sub(&first.exact_seconds());
        let second_delta = self.exact_seconds().sub(&second.exact_seconds());
        let first_distance = if first_delta.sign() == Ordering::Less {
            first_delta.negated()
        } else {
            first_delta
        };
        let second_distance = if second_delta.sign() == Ordering::Less {
            second_delta.negated()
        } else {
            second_delta
        };
        first_distance.sub(&second_distance).sign()
    }

    pub(crate) fn exact_hash_words(&self) -> Vec<u64> {
        self.exact_seconds().canonical_hash_words()
    }

    /// J2000 seconds, rounded once to `f64`.
    #[must_use]
    pub fn j2000_seconds(&self) -> f64 {
        self.seconds_since(ExactEpoch::J2000)
    }

    /// The exact label component of the query.
    #[must_use]
    pub const fn epoch(&self) -> ExactEpoch {
        self.epoch
    }
}

/// The shortest decimal that reads back to `second`, as the signed count of
/// its nearest attosecond (halfway rounds toward positive infinity) and the
/// exact remainder past it. `None` for a non-finite value or one whose count
/// does not fit an `i128`.
fn attoseconds_of_shortest_decimal(second: f64) -> Option<(i128, Residue)> {
    if !second.is_finite() {
        return None;
    }
    let text = format!("{second}");
    let decimal = ShortestDecimal::parse(&text);
    let mut count: i128 = 0;
    for &digit in decimal.integer {
        count = count
            .checked_mul(10)?
            .checked_add(i128::from(digit - b'0'))?;
    }
    for index in 0..18 {
        let digit = decimal.fraction.get(index).map_or(0, |d| d - b'0');
        count = count.checked_mul(10)?.checked_add(i128::from(digit))?;
    }
    // The digits past the attosecond, a fraction `rest / 10^places` of one
    // attosecond. A shortest decimal has at most seventeen significant
    // digits, so `rest` fits a `u64`, and a fraction of at least one half
    // has its first digit in the first place, so `places` is then at most 17.
    let rest = decimal.fraction.get(18..).unwrap_or(&[]);
    let rest = &rest[..rest.len() - rest.iter().rev().take_while(|&&d| d == b'0').count()];
    let mut residue = Residue::ZERO;
    if !rest.is_empty() {
        let places = u16::try_from(rest.len()).ok()?;
        let mut digits: u64 = 0;
        for &digit in rest {
            digits = digits
                .checked_mul(10)?
                .checked_add(u64::from(digit - b'0'))?;
        }
        let whole = (places <= 18).then(|| 10_u64.pow(u32::from(places)));
        let at_least_half = whole.is_some_and(|whole| digits * 2 >= whole);
        let above_half = whole.is_some_and(|whole| digits * 2 > whole);
        // The value is `count + fraction` with the decimal's sign; keep the
        // count nearest to it and the remainder in [-1/2, 1/2).
        let (step, remainder) = match (decimal.negative, at_least_half, above_half) {
            (false, true, _) => (1, -((whole? - digits) as i64)),
            (false, false, _) => (0, digits as i64),
            (true, _, true) => (1, (whole? - digits) as i64),
            (true, _, false) => (0, -(digits as i64)),
        };
        count = count.checked_add(step)?;
        residue = Residue {
            digits: remainder,
            places,
        };
    }
    Some(if decimal.negative {
        (-count, residue)
    } else {
        (count, residue)
    })
}

/// The digit text of a decimal number: Rust's `Display` of a finite `f64`,
/// which is the shortest decimal that reads back to it and never uses an
/// exponent.
struct ShortestDecimal<'a> {
    negative: bool,
    integer: &'a [u8],
    fraction: &'a [u8],
}

impl<'a> ShortestDecimal<'a> {
    fn parse(text: &'a str) -> Self {
        let (negative, body) = match text.strip_prefix('-') {
            Some(body) => (true, body),
            None => (false, text),
        };
        let (integer, fraction) = body.split_once('.').unwrap_or((body, ""));
        Self {
            negative,
            integer: integer.as_bytes(),
            fraction: fraction.as_bytes(),
        }
    }
}

/// The `f64` nearest to `count / per_unit`, ties to even.
///
/// The odd part of `per_unit` (what is left after its factors of two) must be
/// below `2^71`: the dividend is shifted until its top bit is bit 126, so the
/// quotient then carries at least 56 significant bits, the 53 an `f64` keeps,
/// its rounding bit and bits below those into which a nonzero remainder is
/// folded, so the one rounding of the conversion sees it. Scaling back by a
/// power of two is exact.
pub(crate) fn nearest_ratio(count: i128, per_unit: u128) -> f64 {
    let magnitude = count.unsigned_abs();
    if magnitude == 0 {
        return 0.0;
    }
    let twos = per_unit.trailing_zeros();
    let odd = per_unit >> twos;
    debug_assert!(odd < 1 << 71, "odd part of the divisor too wide");
    let shift = magnitude.leading_zeros().saturating_sub(1);
    let scaled = magnitude << shift;
    let quotient = scaled / odd;
    let inexact = !scaled.is_multiple_of(odd);
    let value = (quotient | u128::from(inexact)) as f64 * power_of_two(-i64::from(shift + twos));
    if count < 0 {
        -value
    } else {
        value
    }
}

/// `2^exponent`, for `exponent` in `-1074..=1023`.
fn power_of_two(exponent: i64) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1_u64 << (exponent + 1074))
    }
}

/// A non-negative integer of any width: little-endian 64-bit limbs with no zero
/// limb at the top (zero has no limbs).
#[derive(Debug, Clone, Default, Eq)]
struct Natural(SmallVec<[u64; 4]>);

impl PartialEq for Natural {
    fn eq(&self, other: &Self) -> bool {
        let left = self.0.as_slice();
        let right = other.0.as_slice();
        if left.len() != right.len() {
            return false;
        }
        match left.len() {
            0 => true,
            1 => left[0] == right[0],
            2 => left[0] == right[0] && left[1] == right[1],
            _ => left == right,
        }
    }
}

impl Natural {
    fn from_u128(value: u128) -> Self {
        if value == 0 {
            return Self::default();
        }
        let mut limbs = SmallVec::new();
        limbs.push(value as u64);
        let upper = (value >> 64) as u64;
        if upper != 0 {
            limbs.push(upper);
        }
        Self(limbs)
    }

    fn to_u128(&self) -> Option<u128> {
        match self.0.as_slice() {
            [] => Some(0),
            [low] => Some(u128::from(*low)),
            [low, high] => Some(u128::from(*low) | (u128::from(*high) << 64)),
            _ => None,
        }
    }

    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn bit_len(&self) -> u64 {
        self.0.last().map_or(0, |top| {
            self.0.len() as u64 * 64 - u64::from(top.leading_zeros())
        })
    }

    fn mul_small(&mut self, factor: u64) {
        let mut carry = 0_u128;
        for limb in &mut self.0 {
            let product = u128::from(*limb) * u128::from(factor) + carry;
            *limb = product as u64;
            carry = product >> 64;
        }
        if carry != 0 {
            self.0.push(carry as u64);
        }
        self.trim();
    }

    fn add_small(&mut self, addend: u64) {
        let mut carry = addend;
        for limb in &mut self.0 {
            if carry == 0 {
                return;
            }
            let (sum, overflow) = limb.overflowing_add(carry);
            *limb = sum;
            carry = u64::from(overflow);
        }
        if carry != 0 {
            self.0.push(carry);
        }
    }

    fn mul_pow10(&mut self, mut exponent: u32) {
        const TEN_TO_THE_19: u64 = 10_000_000_000_000_000_000;
        while exponent >= 19 {
            self.mul_small(TEN_TO_THE_19);
            exponent -= 19;
        }
        if exponent > 0 {
            self.mul_small(10_u64.pow(exponent));
        }
    }

    fn shl(&self, bits: u64) -> Self {
        if self.is_zero() {
            return Self::default();
        }
        let limbs = (bits / 64) as usize;
        let within = (bits % 64) as u32;
        let mut out = SmallVec::new();
        for _ in 0..limbs {
            out.push(0);
        }
        if within == 0 {
            out.extend_from_slice(&self.0);
        } else {
            let mut carry = 0_u64;
            for &limb in &self.0 {
                out.push((limb << within) | carry);
                carry = limb >> (64 - within);
            }
            out.push(carry);
        }
        let mut natural = Self(out);
        natural.trim();
        natural
    }

    fn shr1(&mut self) {
        let mut carry = 0_u64;
        for limb in self.0.iter_mut().rev() {
            let low = *limb << 63;
            *limb = (*limb >> 1) | carry;
            carry = low;
        }
        self.trim();
    }

    fn add(&self, other: &Self) -> Self {
        let (long, short) = if self.0.len() >= other.0.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut out = SmallVec::new();
        let mut carry = false;
        for (index, &limb) in long.0.iter().enumerate() {
            let (sum, first) = limb.overflowing_add(short.0.get(index).copied().unwrap_or(0));
            let (sum, second) = sum.overflowing_add(u64::from(carry));
            out.push(sum);
            carry = first || second;
        }
        if carry {
            out.push(1);
        }
        Self(out)
    }

    /// `self -= other`, for `self >= other`.
    fn sub_assign(&mut self, other: &Self) {
        let mut borrow = false;
        for (index, limb) in self.0.iter_mut().enumerate() {
            let (difference, first) =
                limb.overflowing_sub(other.0.get(index).copied().unwrap_or(0));
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            *limb = difference;
            borrow = first || second;
        }
        debug_assert!(!borrow, "subtrahend larger than minuend");
        self.trim();
    }

    fn div_small(&mut self, divisor: u64) -> u64 {
        let divisor = u128::from(divisor);
        let mut remainder = 0_u128;
        for limb in self.0.iter_mut().rev() {
            let value = (remainder << 64) | u128::from(*limb);
            *limb = (value / divisor) as u64;
            remainder = value % divisor;
        }
        self.trim();
        remainder as u64
    }

    fn compare(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

/// The `f64` nearest to `numerator / denominator`, ties to even, with a
/// subnormal result rounded on the subnormal grid; `denominator` is nonzero.
fn nearest_quotient(numerator: &Natural, denominator: &Natural) -> f64 {
    if numerator.is_zero() {
        return 0.0;
    }
    // `exponent` is the floor of the base-two logarithm of the quotient.
    let mut exponent = numerator.bit_len() as i64 - denominator.bit_len() as i64;
    let below = if exponent >= 0 {
        numerator.compare(&denominator.shl(exponent as u64)) == Ordering::Less
    } else {
        numerator.shl(exponent.unsigned_abs()).compare(denominator) == Ordering::Less
    };
    if below {
        exponent -= 1;
    }
    if exponent > 1023 {
        return f64::INFINITY;
    }
    // The place of the last bit the result keeps: the 53rd significant bit,
    // or the subnormal floor.
    let unit = (exponent - 52).max(-1074);
    let (mut remainder, divisor) = if unit >= 0 {
        (numerator.clone(), denominator.shl(unit as u64))
    } else {
        (numerator.shl(unit.unsigned_abs()), denominator.clone())
    };
    // The quotient is below 2^53: long division from bit 53 down.
    let mut shifted = divisor.shl(53);
    let mut quotient = 0_u64;
    for bit in (0..=53).rev() {
        if remainder.compare(&shifted) != Ordering::Less {
            remainder.sub_assign(&shifted);
            quotient |= 1 << bit;
        }
        if bit > 0 {
            shifted.shr1();
        }
    }
    match remainder.shl(1).compare(&divisor) {
        Ordering::Greater => quotient += 1,
        Ordering::Equal if quotient & 1 == 1 => quotient += 1,
        _ => {}
    }
    // At most 2^53 and a power of two in range: the product is exact.
    quotient as f64 * power_of_two(unit)
}

/// An exact number of seconds (or of any unit):
/// `±magnitude / (2^binary_places * 10^decimal_places)`.
///
/// Every finite `f64` and every decimal is one of these, and sums,
/// differences and integer multiples of them are too; a result is rounded
/// once, to the nearest `f64` with ties to even, by [`Self::to_f64`] or
/// [`Self::div_to_f64`].
#[derive(Debug, Clone)]
pub(crate) struct ExactSeconds {
    negative: bool,
    magnitude: Natural,
    binary_places: u32,
    decimal_places: u32,
}

impl ExactSeconds {
    fn same_representation(&self, other: &Self) -> bool {
        self.negative == other.negative
            && self.magnitude == other.magnitude
            && self.binary_places == other.binary_places
            && self.decimal_places == other.decimal_places
    }

    fn same_value_at_same_denominator(&self, other: &Self) -> bool {
        if self.magnitude.is_zero() || other.magnitude.is_zero() {
            return self.magnitude.is_zero() && other.magnitude.is_zero();
        }
        self.same_representation(other)
    }

    fn aligned_magnitude_u128(&self, binary_places: u32, decimal_places: u32) -> Option<u128> {
        let magnitude = self.magnitude.to_u128()?;
        if magnitude == 0 {
            return Some(0);
        }
        let binary_shift = binary_places.checked_sub(self.binary_places)?;
        let shifted_magnitude = magnitude.checked_shl(binary_shift)?;
        if shifted_magnitude.checked_shr(binary_shift)? != magnitude {
            return None;
        }
        let decimal_shift = decimal_places.checked_sub(self.decimal_places)?;
        if decimal_shift == 0 {
            return Some(shifted_magnitude);
        }
        shifted_magnitude.checked_mul(10_u128.checked_pow(decimal_shift)?)
    }

    fn same_value_at_common_denominator(&self, other: &Self) -> Option<bool> {
        if self.magnitude.is_zero() || other.magnitude.is_zero() {
            return Some(self.magnitude.is_zero() && other.magnitude.is_zero());
        }
        if self.negative != other.negative {
            return Some(false);
        }
        if self.decimal_places == other.decimal_places {
            let self_magnitude = self.magnitude.to_u128()?;
            let other_magnitude = other.magnitude.to_u128()?;
            if self.binary_places == other.binary_places {
                return Some(self_magnitude == other_magnitude);
            }
            let (finer_magnitude, binary_shift, coarser_magnitude) =
                if self.binary_places > other.binary_places {
                    (
                        self_magnitude,
                        self.binary_places - other.binary_places,
                        other_magnitude,
                    )
                } else {
                    (
                        other_magnitude,
                        other.binary_places - self.binary_places,
                        self_magnitude,
                    )
                };
            if binary_shift >= u128::BITS {
                return Some(false);
            }
            let Some(reduced_magnitude) = finer_magnitude.checked_shr(binary_shift) else {
                return Some(false);
            };
            return Some(
                reduced_magnitude == coarser_magnitude
                    && reduced_magnitude.checked_shl(binary_shift) == Some(finer_magnitude),
            );
        }
        let binary_places = self.binary_places.max(other.binary_places);
        let decimal_places = self.decimal_places.max(other.decimal_places);
        let left = self.aligned_magnitude_u128(binary_places, decimal_places)?;
        let right = other.aligned_magnitude_u128(binary_places, decimal_places)?;
        Some(left == right)
    }

    /// A whole number.
    pub(crate) fn from_integer(value: i128) -> Self {
        Self::from_decimal(value, 0)
    }

    /// `count / 10^decimal_places`.
    pub(crate) fn from_decimal(count: i128, decimal_places: u32) -> Self {
        Self {
            negative: count < 0,
            magnitude: Natural::from_u128(count.unsigned_abs()),
            binary_places: 0,
            decimal_places,
        }
    }

    /// The exact binary value of a finite `f64`; `None` for a non-finite one.
    pub(crate) fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let stored = bits & ((1_u64 << 52) - 1);
        let (mantissa, exponent) = if biased == 0 {
            (stored, -1074)
        } else {
            (stored | (1_u64 << 52), biased - 1075)
        };
        let mantissa = Natural::from_u128(u128::from(mantissa));
        let (magnitude, binary_places) = if exponent >= 0 {
            (mantissa.shl(exponent as u64), 0)
        } else {
            (mantissa, exponent.unsigned_abs() as u32)
        };
        Some(Self {
            negative: value.is_sign_negative() && !magnitude.is_zero(),
            magnitude,
            binary_places,
            decimal_places: 0,
        })
    }

    /// The shortest decimal that reads back to a finite `f64` (`0.1` is one
    /// tenth), exactly; `None` for a non-finite value.
    pub(crate) fn from_shortest_decimal(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self::from_integer(0));
        }
        let text = format!("{value}");
        let decimal = ShortestDecimal::parse(&text);
        let mut magnitude = Natural::default();
        for &digit in decimal.integer.iter().chain(decimal.fraction) {
            magnitude.mul_small(10);
            magnitude.add_small(u64::from(digit - b'0'));
        }
        Some(Self {
            negative: decimal.negative && !magnitude.is_zero(),
            magnitude,
            binary_places: 0,
            decimal_places: decimal.fraction.len() as u32,
        })
    }

    /// The magnitude over the finer denominator `2^binary_places *
    /// 10^decimal_places`, which must be at least this value's own.
    fn magnitude_over(&self, binary_places: u32, decimal_places: u32) -> Natural {
        let mut magnitude = self
            .magnitude
            .shl(u64::from(binary_places - self.binary_places));
        magnitude.mul_pow10(decimal_places - self.decimal_places);
        magnitude
    }

    /// `self + other`, exactly.
    pub(crate) fn add(&self, other: &Self) -> Self {
        if self.magnitude.is_zero() {
            return other.clone();
        }
        if other.magnitude.is_zero() {
            return self.clone();
        }
        let binary_places = self.binary_places.max(other.binary_places);
        let decimal_places = self.decimal_places.max(other.decimal_places);
        if let (Some(left), Some(right)) = (
            self.aligned_magnitude_u128(binary_places, decimal_places),
            other.aligned_magnitude_u128(binary_places, decimal_places),
        ) {
            let (negative, magnitude) = if self.negative == other.negative {
                let Some(sum) = left.checked_add(right) else {
                    return self.add_wide(other, binary_places, decimal_places);
                };
                (self.negative, sum)
            } else if left < right {
                (other.negative, right - left)
            } else {
                (self.negative, left - right)
            };
            return Self {
                negative: negative && magnitude != 0,
                magnitude: Natural::from_u128(magnitude),
                binary_places,
                decimal_places,
            };
        }
        self.add_wide(other, binary_places, decimal_places)
    }

    fn add_wide(&self, other: &Self, binary_places: u32, decimal_places: u32) -> Self {
        let left = self.magnitude_over(binary_places, decimal_places);
        let right = other.magnitude_over(binary_places, decimal_places);
        let (negative, magnitude) = if self.negative == other.negative {
            (self.negative, left.add(&right))
        } else {
            match left.compare(&right) {
                Ordering::Less => {
                    let mut magnitude = right;
                    magnitude.sub_assign(&left);
                    (other.negative, magnitude)
                }
                _ => {
                    let mut magnitude = left;
                    magnitude.sub_assign(&right);
                    (self.negative, magnitude)
                }
            }
        };
        Self {
            negative: negative && !magnitude.is_zero(),
            magnitude,
            binary_places,
            decimal_places,
        }
    }

    /// The sign of the value: `Less` below zero, `Equal` at zero, `Greater`
    /// above it.
    pub(crate) fn sign(&self) -> Ordering {
        if self.magnitude.is_zero() {
            Ordering::Equal
        } else if self.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }

    fn canonical_hash_words(&self) -> Vec<u64> {
        let mut magnitude = self.magnitude.clone();
        if magnitude.is_zero() {
            return vec![0, 0, 0, 0];
        }
        let mut denominator_twos = u64::from(self.binary_places) + u64::from(self.decimal_places);
        let mut denominator_fives = u64::from(self.decimal_places);
        while denominator_twos > 0 && magnitude.0.first().is_some_and(|limb| limb & 1 == 0) {
            magnitude.div_small(2);
            denominator_twos -= 1;
        }
        while denominator_fives > 0 {
            let mut reduced = magnitude.clone();
            if reduced.div_small(5) != 0 {
                break;
            }
            magnitude = reduced;
            denominator_fives -= 1;
        }
        let mut words = Vec::with_capacity(magnitude.0.len() + 4);
        words.push(u64::from(self.negative && !magnitude.is_zero()));
        words.push(denominator_twos);
        words.push(denominator_fives);
        words.push(magnitude.0.len() as u64);
        words.extend_from_slice(&magnitude.0);
        words
    }

    /// `-self`.
    pub(crate) fn negated(&self) -> Self {
        Self {
            negative: !self.negative && !self.magnitude.is_zero(),
            ..self.clone()
        }
    }

    /// `self - other`, exactly.
    pub(crate) fn sub(&self, other: &Self) -> Self {
        self.add(&other.negated())
    }

    /// `self * factor`, exactly.
    pub(crate) fn mul_integer(&self, factor: i64) -> Self {
        let mut magnitude = self.magnitude.clone();
        magnitude.mul_small(factor.unsigned_abs());
        Self {
            negative: (self.negative != (factor < 0)) && !magnitude.is_zero(),
            magnitude,
            ..*self
        }
    }

    /// The `f64` nearest to the value, ties to even.
    pub(crate) fn to_f64(&self) -> f64 {
        self.div_to_f64(1)
    }

    /// The `f64` nearest to `self / divisor`, ties to even; `divisor` is
    /// nonzero.
    pub(crate) fn div_to_f64(&self, divisor: u64) -> f64 {
        let value = self.small_quotient(divisor).unwrap_or_else(|| {
            let mut denominator =
                Natural::from_u128(u128::from(divisor)).shl(u64::from(self.binary_places));
            denominator.mul_pow10(self.decimal_places);
            nearest_quotient(&self.magnitude, &denominator)
        });
        if self.negative {
            -value
        } else {
            value
        }
    }

    /// [`Self::div_to_f64`] of the magnitude through [`nearest_ratio`], when
    /// the magnitude and the denominator fit it.
    fn small_quotient(&self, divisor: u64) -> Option<f64> {
        let magnitude = self.magnitude.to_u128()?;
        let magnitude = i128::try_from(magnitude).ok()?;
        let denominator = u128::from(divisor)
            .checked_mul(10_u128.checked_pow(self.decimal_places)?)?
            .checked_mul(1_u128.checked_shl(self.binary_places)?)?;
        let odd = denominator >> denominator.trailing_zeros();
        (odd < 1 << 71).then(|| nearest_ratio(magnitude, denominator))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// `count / 10^places` as decimal text, which `str::parse` reads to the
    /// correctly rounded double: the independent reference for every
    /// rounding here.
    fn decimal_text(count: i128, places: u32) -> String {
        let digits = count.unsigned_abs().to_string();
        let places = places as usize;
        let padded = format!("{digits:0>width$}", width = places + 1);
        let (whole, fraction) = padded.split_at(padded.len() - places);
        let sign = if count < 0 { "-" } else { "" };
        if fraction.is_empty() {
            format!("{sign}{whole}")
        } else {
            format!("{sign}{whole}.{fraction}")
        }
    }

    fn power_of_ten(places: u32) -> Natural {
        let mut natural = Natural::from_u128(1);
        natural.mul_pow10(places);
        natural
    }

    #[test]
    fn decimals_round_as_str_parse_rounds_them() {
        // Random counts of up to 126 bits over 0 to 45 decimal places, read
        // through the narrow path where it applies and through the wide long
        // division always, against the correctly rounded `str::parse`.
        let mut state = 0x3c6e_f372_fe94_f82b_u64;
        for _ in 0..40_000 {
            let bits = xorshift(&mut state) % 126 + 1;
            let wide = (u128::from(xorshift(&mut state)) << 64) | u128::from(xorshift(&mut state));
            let magnitude = (wide >> (128 - bits)) as i128;
            let count = if xorshift(&mut state).is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            };
            let places = (xorshift(&mut state) % 46) as u32;
            let expected: f64 = decimal_text(count, places).parse().unwrap();
            let exact = ExactSeconds::from_decimal(count, places);
            assert_eq!(
                exact.to_f64().to_bits(),
                expected.to_bits(),
                "{count} / 10^{places}"
            );
            let wide_path = nearest_quotient(
                &Natural::from_u128(count.unsigned_abs()),
                &power_of_ten(places),
            );
            assert_eq!(
                wide_path.to_bits(),
                expected.abs().to_bits(),
                "{count} / 10^{places} (wide)"
            );
        }
    }

    #[test]
    fn subnormal_and_overflowing_quotients_round_on_their_grid() {
        for (count, places) in [
            (1_i128, 324_u32),
            (2, 324),
            (3, 324),
            (25, 325),
            (24_703_282_292_062_327, 340),
            (24_703_282_292_062_328, 340),
            (49_406_564_584_124_654, 340),
            (22_250_738_585_072_011, 324),
            (22_250_738_585_072_014, 324),
        ] {
            let text = decimal_text(count, places);
            let expected: f64 = text.parse().unwrap();
            assert_eq!(
                ExactSeconds::from_decimal(count, places).to_f64().to_bits(),
                expected.to_bits(),
                "{text}"
            );
        }
        let mut huge = Natural::from_u128(18);
        huge.mul_pow10(307);
        assert_eq!(
            nearest_quotient(&huge, &Natural::from_u128(1)),
            f64::INFINITY
        );
    }

    #[test]
    fn doubles_and_their_shortest_decimals_are_held_exactly() {
        let mut state = 0x510e_527f_ade6_82d1_u64;
        let mut values = vec![
            0.0,
            -0.0,
            0.1,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
            -f64::MAX,
            1.0e-30,
            59.999_999_999_999_99,
        ];
        for _ in 0..20_000 {
            values.push(f64::from_bits(xorshift(&mut state)));
        }
        for value in values.into_iter().filter(|value| value.is_finite()) {
            let binary = ExactSeconds::from_f64(value).unwrap();
            assert_eq!(binary.to_f64(), value, "{value:e}");
            let decimal = ExactSeconds::from_shortest_decimal(value).unwrap();
            assert_eq!(decimal.to_f64(), value, "{value:e}");
        }
        assert!(ExactSeconds::from_f64(f64::NAN).is_none());
        assert!(ExactSeconds::from_shortest_decimal(f64::INFINITY).is_none());
    }

    #[test]
    fn sums_and_products_are_exact_until_rounded() {
        // 0.1 + 0.2 held exactly is 0.3, one rounding; the f64 sum is not.
        let sum = ExactSeconds::from_shortest_decimal(0.1)
            .unwrap()
            .add(&ExactSeconds::from_shortest_decimal(0.2).unwrap());
        assert_eq!(sum.to_f64(), 0.3);
        assert_ne!(0.1_f64 + 0.2, 0.3);
        // A binary value plus a decimal one, and back.
        let third = ExactSeconds::from_f64(1.0 / 3.0).unwrap();
        let tenth = ExactSeconds::from_shortest_decimal(0.1).unwrap();
        assert_eq!(third.add(&tenth).sub(&tenth).to_f64(), 1.0 / 3.0);
        assert_eq!(tenth.sub(&tenth).to_f64().to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            tenth.mul_integer(-86_400).to_f64(),
            -8_640.0,
            "0.1 * 86400 held exactly"
        );
        assert_eq!(tenth.sign(), Ordering::Greater);
        assert_eq!(tenth.negated().sign(), Ordering::Less);
        assert_eq!(tenth.sub(&tenth).sign(), Ordering::Equal);
        // Exact differences of doubles within a factor of two agree with the
        // (then exact) f64 subtraction.
        let mut state = 0x9b05_688c_2b3e_6c1f_u64;
        for _ in 0..10_000 {
            let a = 1.0 + (xorshift(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
            let b = 1.0 + (xorshift(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
            let exact = ExactSeconds::from_f64(a)
                .unwrap()
                .sub(&ExactSeconds::from_f64(b).unwrap());
            assert_eq!(exact.to_f64(), a - b);
        }
    }

    #[test]
    fn checked_u128_sum_matches_independent_scaled_integer() {
        let left = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(5),
            binary_places: 1,
            decimal_places: 1,
        };
        let right = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(3),
            binary_places: 2,
            decimal_places: 1,
        };
        let sum = left.add(&right);
        let expected_scaled = 5_u128 * 2 + 3;
        assert_eq!(sum.magnitude.to_u128(), Some(expected_scaled));
        assert_eq!(sum.binary_places, 2);
        assert_eq!(sum.decimal_places, 1);
        assert_eq!(sum.to_f64(), 13.0 / 40.0);

        let negative = ExactSeconds {
            negative: true,
            magnitude: Natural::from_u128(3),
            binary_places: 3,
            decimal_places: 0,
        };
        let positive = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1),
            binary_places: 2,
            decimal_places: 0,
        };
        let difference = negative.add(&positive);
        assert!(difference.negative);
        assert_eq!(difference.magnitude.to_u128(), Some(1));
        assert_eq!(difference.binary_places, 3);
        assert_eq!(difference.to_f64(), -0.125);
    }

    #[test]
    fn checked_u128_overflow_falls_back_to_unbounded_limbs() {
        let negative_minimum = ExactSeconds::from_integer(i128::MIN);
        let sum = negative_minimum.add(&negative_minimum);
        assert!(sum.negative);
        assert_eq!(sum.magnitude.0.as_slice(), &[0, 0, 1]);
        assert_eq!(sum.binary_places, 0);
        assert_eq!(sum.decimal_places, 0);

        let largest = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(u128::MAX),
            binary_places: 0,
            decimal_places: 0,
        };
        let half = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1),
            binary_places: 1,
            decimal_places: 0,
        };
        let sum = largest.add(&half);
        assert_eq!(sum.magnitude.0.as_slice(), &[u64::MAX, u64::MAX, 1]);
        assert_eq!(sum.binary_places, 1);
        assert_eq!(sum.decimal_places, 0);
    }

    #[test]
    fn aligned_magnitude_u128_checks_shift_loss_and_skips_zero_decimal_scale() {
        let unit = ExactSeconds::from_integer(1);
        assert_eq!(unit.aligned_magnitude_u128(127, 0), Some(1_u128 << 127));
        assert_eq!(unit.aligned_magnitude_u128(128, 0), None);

        let largest = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(u128::MAX),
            binary_places: 0,
            decimal_places: 0,
        };
        assert_eq!(largest.aligned_magnitude_u128(0, 0), Some(u128::MAX));
        assert_eq!(largest.aligned_magnitude_u128(1, 0), None);

        assert_eq!(unit.aligned_magnitude_u128(0, 38), Some(10_u128.pow(38)));
        assert_eq!(unit.aligned_magnitude_u128(0, 39), None);
    }

    #[test]
    fn common_denominator_equality_reduces_binary_scale_without_losing_bits() {
        let numerator_at_finer_scale = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1_u128 << 127),
            binary_places: 127,
            decimal_places: 2,
        };
        let numerator_at_coarser_scale = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1),
            binary_places: 0,
            decimal_places: 2,
        };
        assert_eq!(
            numerator_at_finer_scale.same_value_at_common_denominator(&numerator_at_coarser_scale),
            Some(true)
        );
        assert_eq!(
            numerator_at_coarser_scale.same_value_at_common_denominator(&numerator_at_finer_scale),
            Some(true)
        );

        let exact_half_at_finer_scale = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(2),
            binary_places: 2,
            decimal_places: 0,
        };
        let exact_half_at_coarser_scale = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1),
            binary_places: 1,
            decimal_places: 0,
        };
        assert_eq!(
            exact_half_at_finer_scale
                .same_value_at_common_denominator(&exact_half_at_coarser_scale),
            Some(true)
        );
        assert_eq!(
            exact_half_at_coarser_scale
                .same_value_at_common_denominator(&exact_half_at_finer_scale),
            Some(true)
        );

        let numerator_with_discarded_bit = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(3),
            binary_places: 2,
            decimal_places: 0,
        };
        assert_eq!(
            numerator_with_discarded_bit
                .same_value_at_common_denominator(&exact_half_at_coarser_scale),
            Some(false)
        );
        assert_eq!(
            exact_half_at_coarser_scale
                .same_value_at_common_denominator(&numerator_with_discarded_bit),
            Some(false)
        );

        let largest_finer_numerator = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(u128::MAX),
            binary_places: 1,
            decimal_places: 0,
        };
        let largest_coarser_numerator = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1_u128 << 127),
            binary_places: 0,
            decimal_places: 0,
        };
        assert_eq!(
            largest_finer_numerator.same_value_at_common_denominator(&largest_coarser_numerator),
            Some(false)
        );
        assert_eq!(
            largest_coarser_numerator.same_value_at_common_denominator(&largest_finer_numerator),
            Some(false)
        );
    }

    #[test]
    fn common_denominator_equality_short_circuits_zero_and_sign() {
        let positive_zero = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(0),
            binary_places: 9,
            decimal_places: 3,
        };
        let negative_zero = ExactSeconds {
            negative: true,
            magnitude: Natural::from_u128(0),
            binary_places: 0,
            decimal_places: 0,
        };
        assert_eq!(
            positive_zero.same_value_at_common_denominator(&negative_zero),
            Some(true)
        );
        let nonzero = ExactSeconds::from_f64(0.125).unwrap();
        assert_eq!(
            positive_zero.same_value_at_common_denominator(&nonzero),
            Some(false)
        );
        assert_eq!(
            nonzero.same_value_at_common_denominator(&positive_zero),
            Some(false)
        );

        let negative_half_at_finer_scale = ExactSeconds {
            negative: true,
            magnitude: Natural::from_u128(2),
            binary_places: 2,
            decimal_places: 0,
        };
        let negative_half_at_coarser_scale = ExactSeconds {
            negative: true,
            magnitude: Natural::from_u128(1),
            binary_places: 1,
            decimal_places: 0,
        };
        assert_eq!(
            negative_half_at_finer_scale
                .same_value_at_common_denominator(&negative_half_at_coarser_scale),
            Some(true)
        );
        assert_eq!(
            negative_half_at_coarser_scale
                .same_value_at_common_denominator(&negative_half_at_finer_scale),
            Some(true)
        );

        let negative_half = ExactSeconds {
            negative: true,
            magnitude: Natural::from_u128(1),
            binary_places: 1,
            decimal_places: 0,
        };
        let positive_half_at_finer_scale = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(2),
            binary_places: 2,
            decimal_places: 0,
        };
        assert_eq!(
            negative_half.same_value_at_common_denominator(&positive_half_at_finer_scale),
            Some(false)
        );
    }

    #[test]
    fn common_denominator_equality_preserves_large_shift_and_wide_fallback() {
        let one_at_large_binary_scale = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1),
            binary_places: 128,
            decimal_places: 0,
        };
        let one = ExactSeconds::from_integer(1);
        assert_eq!(
            one_at_large_binary_scale.same_value_at_common_denominator(&one),
            Some(false)
        );
        assert_eq!(
            one.same_value_at_common_denominator(&one_at_large_binary_scale),
            Some(false)
        );

        let wide_integer = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1).shl(128),
            binary_places: 1,
            decimal_places: 0,
        };
        let wide_fraction = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1).shl(127),
            binary_places: 0,
            decimal_places: 0,
        };
        assert_eq!(
            wide_integer.same_value_at_common_denominator(&wide_fraction),
            None
        );
        let wide_integer_query = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(wide_integer),
        };
        let wide_fraction_query = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(wide_fraction),
        };
        assert_eq!(wide_integer_query, wide_fraction_query);
    }

    #[test]
    fn exact_query_equality_retains_cross_origin_and_decimal_semantics() {
        let at_one_second = ExactEpoch::new(1, 0).unwrap().query();
        let from_j2000 = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_f64(1.0).unwrap()),
        };
        assert_eq!(at_one_second, from_j2000);

        let decimal_tenth = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_shortest_decimal(0.1).unwrap()),
        };
        let binary_tenth = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_f64(0.1).unwrap()),
        };
        assert_ne!(decimal_tenth, binary_tenth);
    }

    #[test]
    fn cloned_exact_query_shares_immutable_offset_and_arithmetic_detaches() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExactEpochQuery>();

        let original = ExactEpoch::from_civil(2026, 9, 25, 12, 30, 0.125)
            .unwrap()
            .query();
        let original_clone = original.clone();
        assert!(Arc::ptr_eq(&original.offset, &original_clone.offset));
        assert_eq!(original, original_clone);

        let advanced = original_clone
            .clone()
            .checked_add_binary_seconds(0.25)
            .expect("finite binary increment");
        assert!(!Arc::ptr_eq(&original.offset, &advanced.offset));
        assert_eq!(original.j2000_seconds(), original_clone.j2000_seconds());
        assert_eq!(advanced.seconds_since_query(&original), 0.25);

        let retreated = original
            .clone()
            .checked_sub_binary_seconds(0.25)
            .expect("finite binary decrement");
        assert_eq!(retreated.seconds_since_query(&original), -0.25);
        assert_eq!(original, original_clone);

        let independently_built = ExactEpoch::from_civil(2026, 9, 25, 12, 30, 0.125)
            .unwrap()
            .query();
        assert!(!Arc::ptr_eq(&original.offset, &independently_built.offset));
        assert_eq!(original, independently_built);
        assert_eq!(format!("{original:?}"), format!("{original_clone:?}"));

        let same_offset_different_epoch = ExactEpochQuery {
            epoch: ExactEpoch::new(1, 0).unwrap(),
            offset: original.offset.clone(),
        };
        assert!(Arc::ptr_eq(
            &original.offset,
            &same_offset_different_epoch.offset
        ));
        assert_ne!(original, same_offset_different_epoch);
        assert!(original.shares_representation(&original_clone));
        assert!(!original.shares_representation(&independently_built));
        assert!(!original.shares_representation(&same_offset_different_epoch));
    }

    #[test]
    fn exact_query_equality_distinguishes_values_with_the_same_f64_epoch() {
        let whole = ExactEpoch::new(1_000_000_000, 0).unwrap().query();
        let advanced = whole
            .clone()
            .checked_add_binary_seconds(1.0e-9)
            .expect("finite binary increment");
        assert_eq!(
            whole.j2000_seconds().to_bits(),
            advanced.j2000_seconds().to_bits()
        );
        assert_ne!(whole, advanced);
    }

    #[test]
    fn shortest_decimal_signed_zero_avoids_decimal_parsing() {
        let positive_zero = ExactSeconds::from_shortest_decimal(0.0).unwrap();
        let negative_zero = ExactSeconds::from_shortest_decimal(-0.0).unwrap();
        assert!(positive_zero.magnitude.is_zero());
        assert!(negative_zero.magnitude.is_zero());
        assert!(!negative_zero.negative);
        assert_eq!(negative_zero.to_f64().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn natural_equality_matches_limbs_at_every_storage_width() {
        for limb_count in 0..=6 {
            let limbs: Vec<u64> = (0..limb_count).map(|index| index as u64 + 1).collect();
            let natural = Natural(limbs.iter().copied().collect());
            let equal = Natural(limbs.iter().copied().collect());
            assert_eq!(natural, equal);
            for changed_limb in 0..limb_count {
                let mut different = equal.clone();
                different.0[changed_limb] += 1;
                assert_ne!(natural, different);
                assert_ne!(different, natural);
            }
            let mut longer = equal.clone();
            longer.0.push(1);
            assert_ne!(natural, longer);
            assert_ne!(longer, natural);
        }
        let mut spilled_small = Natural(SmallVec::with_capacity(8));
        spilled_small.0.extend([u64::MAX, u64::MAX]);
        assert!(spilled_small.0.spilled());
        assert_eq!(spilled_small, Natural::from_u128(u128::MAX));
    }

    #[test]
    fn natural_inline_limbs_spill_and_preserve_carry_borrow_and_shift() {
        let mut four_limbs = Natural::default();
        for _ in 0..4 {
            four_limbs.0.push(u64::MAX);
        }
        assert!(!four_limbs.0.spilled());

        let one = Natural::from_u128(1);
        let carried = four_limbs.add(&one);
        assert!(carried.0.spilled());
        assert_eq!(carried.0.as_slice(), &[0, 0, 0, 0, 1]);

        let mut borrowed = carried.clone();
        borrowed.sub_assign(&one);
        assert_eq!(borrowed.0.as_slice(), &[u64::MAX; 4]);
        let mut zero = borrowed.clone();
        zero.sub_assign(&borrowed);
        assert!(zero.is_zero());
        assert_eq!(zero.to_u128(), Some(0));

        let carried_u128 = Natural::from_u128(u128::MAX).add(&one);
        assert_eq!(carried_u128.0.as_slice(), &[0, 0, 1]);
        assert_eq!(
            Natural::from_u128(u128::MAX).shl(1).0.as_slice(),
            &[u64::MAX - 1, u64::MAX, 1]
        );
        let mut shifted_back = Natural::from_u128(u128::MAX).shl(1);
        shifted_back.shr1();
        assert_eq!(shifted_back.0.as_slice(), &[u64::MAX, u64::MAX]);
        assert_eq!(one.shl(256).0.as_slice(), &[0, 0, 0, 0, 1]);
    }

    #[test]
    fn nearest_ratio_matches_the_decimal_reading() {
        let mut state = 0x1f83_d9ab_fb41_bd6b_u64;
        for _ in 0..40_000 {
            let bits = xorshift(&mut state) % 126 + 1;
            let wide = (u128::from(xorshift(&mut state)) << 64) | u128::from(xorshift(&mut state));
            let count = (wide >> (128 - bits)) as i128;
            let count = if xorshift(&mut state).is_multiple_of(2) {
                count
            } else {
                -count
            };
            let places = (xorshift(&mut state) % 20) as u32;
            let expected: f64 = decimal_text(count, places).parse().unwrap();
            assert_eq!(
                nearest_ratio(count, 10_u128.pow(places)).to_bits(),
                expected.to_bits(),
                "{count} / 10^{places}"
            );
        }
    }

    #[test]
    fn exact_epochs_difference_labels_exactly() {
        let at = |second: f64| ExactEpoch::from_civil(2026, 9, 23, 6, 30, second).unwrap();
        // Labels a tenth of a second apart are a tenth apart, rounded once;
        // the difference of their rounded J2000 doubles is not.
        assert_eq!(at(0.3).seconds_since(at(0.2)), 0.1);
        assert_ne!(at(0.3).j2000_seconds() - at(0.2).j2000_seconds(), 0.1);
        // Whole-second labels are whole seconds apart, across days.
        let a = ExactEpoch::from_civil(2026, 9, 23, 23, 59, 30.0).unwrap();
        let b = ExactEpoch::from_civil(2026, 9, 24, 0, 0, 0.0).unwrap();
        assert_eq!(b.seconds_since(a), 30.0);
        assert_eq!(a.seconds_since(b), -30.0);
        assert_eq!(ExactEpoch::J2000.j2000_seconds(), 0.0);
        assert_eq!(
            ExactEpoch::from_civil(2000, 1, 1, 12, 0, 0.0),
            Some(ExactEpoch::J2000)
        );
        // Every decimal label on the attosecond grid differences exactly:
        // compare with the decimal text of the difference.
        let mut state = 0xa54f_f53a_5f1d_36f1_u64;
        for _ in 0..20_000 {
            let digits = (xorshift(&mut state) % 16) as u32;
            let scale = 10_u64.pow(digits);
            let first = (xorshift(&mut state) % (60 * scale)) as f64 / scale as f64;
            let second = (xorshift(&mut state) % (60 * scale)) as f64 / scale as f64;
            let minutes = (xorshift(&mut state) % 1_000) as i32;
            let later = ExactEpoch::from_civil(2026, 9, 23, 0, minutes, first).unwrap();
            let earlier = ExactEpoch::from_civil(2026, 9, 23, 0, 0, second).unwrap();
            let difference = later.total_attoseconds() - earlier.total_attoseconds();
            let expected: f64 = decimal_text(difference, 18).parse().unwrap();
            assert_eq!(later.seconds_since(earlier).to_bits(), expected.to_bits());
            let expected_j2000: f64 = decimal_text(later.total_attoseconds(), 18).parse().unwrap();
            assert_eq!(later.j2000_seconds().to_bits(), expected_j2000.to_bits());
        }
    }

    #[test]
    fn equivalent_epoch_queries_have_identical_canonical_hash_words() {
        let decimal_half = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_decimal(5, 1)),
        };
        let binary_half = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_f64(0.5).unwrap()),
        };
        assert_eq!(decimal_half, binary_half);
        assert_eq!(
            decimal_half.exact_hash_words(),
            binary_half.exact_hash_words()
        );

        let decimal_tenth = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_shortest_decimal(0.1).unwrap()),
        };
        let binary_tenth = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_f64(0.1).unwrap()),
        };
        assert_eq!(
            decimal_tenth.j2000_seconds().to_bits(),
            binary_tenth.j2000_seconds().to_bits()
        );
        assert_ne!(decimal_tenth, binary_tenth);

        let same_representation = binary_half.clone();
        assert_eq!(binary_half, same_representation);

        let whole_second = ExactEpoch::new(1, 0).unwrap().query();
        let binary_offset = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(ExactSeconds::from_f64(1.0).unwrap()),
        };
        assert_eq!(whole_second, binary_offset);
        assert_eq!(
            whole_second.exact_hash_words(),
            binary_offset.exact_hash_words()
        );

        let first_offset = ExactSeconds::from_decimal(1, 1);
        let second_offset = ExactSeconds::from_decimal(2, 1);
        let first_query = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(first_offset),
        };
        let second_query = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(second_offset),
        };
        assert_ne!(first_query, second_query);

        let wide_integer = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1).shl(127),
            binary_places: 0,
            decimal_places: 0,
        };
        let wide_fraction = ExactSeconds {
            negative: false,
            magnitude: Natural::from_u128(1).shl(128),
            binary_places: 1,
            decimal_places: 0,
        };
        let wide_integer_query = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(wide_integer),
        };
        let wide_fraction_query = ExactEpochQuery {
            epoch: ExactEpoch::J2000,
            offset: Arc::new(wide_fraction),
        };
        assert_eq!(wide_integer_query, wide_fraction_query);
    }

    #[test]
    fn zero_addition_and_whole_second_epochs_keep_minimal_denominators() {
        let zero = ExactEpoch::J2000.exact_seconds();
        assert!(zero.magnitude.is_zero());
        assert_eq!(zero.binary_places, 0);
        assert_eq!(zero.decimal_places, 0);

        let binary_offset = ExactSeconds::from_f64(0.125).unwrap();
        let zero_then_binary = zero.add(&binary_offset);
        assert!(zero_then_binary.same_representation(&binary_offset));
        let binary_then_zero = binary_offset.add(&zero);
        assert!(binary_then_zero.same_representation(&binary_offset));

        let whole_second = ExactEpoch::new(646_272_000, 0).unwrap().exact_seconds();
        assert_eq!(whole_second.magnitude.to_u128(), Some(646_272_000));
        assert_eq!(whole_second.binary_places, 0);
        assert_eq!(whole_second.decimal_places, 0);
        assert_eq!(whole_second.to_f64(), 646_272_000.0);

        let attosecond_epoch = ExactEpoch::new(0, 1).unwrap().exact_seconds();
        assert_eq!(attosecond_epoch.decimal_places, 18);
        assert_eq!(attosecond_epoch.to_f64(), 1.0e-18);
    }

    #[test]
    fn identical_epoch_difference_shortcuts_match_general_exact_arithmetic() {
        let receive_epoch = ExactEpoch::new(646_272_000, 500_000_000_000_000_000).unwrap();
        let wide_offset = ExactSeconds::from_decimal(1_234_567_890_123_456_789, 30)
            .add(&ExactSeconds::from_f64(0.000_000_000_000_3).unwrap());
        let single_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(wide_offset.clone()),
        };
        let old_seconds_since = receive_epoch
            .exact_seconds()
            .sub(&receive_epoch.exact_seconds())
            .add(&wide_offset)
            .to_f64();
        assert_eq!(
            single_query.seconds_since(receive_epoch).to_bits(),
            old_seconds_since.to_bits()
        );

        let earlier_offset = wide_offset.add(&ExactSeconds::from_decimal(1, 40));
        let later_offset = earlier_offset.add(&ExactSeconds::from_decimal(1, 45));
        let earlier_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(earlier_offset),
        };
        let later_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(later_offset),
        };
        let old_query_difference = later_query
            .epoch
            .exact_seconds()
            .sub(&earlier_query.epoch.exact_seconds())
            .add(&later_query.offset)
            .sub(&earlier_query.offset)
            .to_f64();
        assert_eq!(
            later_query.seconds_since_query(&earlier_query).to_bits(),
            old_query_difference.to_bits()
        );

        let subnormal_offset = ExactSeconds::from_f64(f64::from_bits(1)).unwrap();
        let zero_offset = ExactSeconds::from_integer(0);
        let subnormal_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(subnormal_offset.clone()),
        };
        let zero_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(zero_offset.clone()),
        };
        let old_subnormal_difference = receive_epoch
            .exact_seconds()
            .sub(&receive_epoch.exact_seconds())
            .add(&subnormal_offset)
            .sub(&zero_offset)
            .to_f64();
        assert_eq!(
            subnormal_query.seconds_since_query(&zero_query).to_bits(),
            old_subnormal_difference.to_bits()
        );

        let tiny_offset = ExactSeconds::from_decimal(1, 30);
        let cancelled_offset = tiny_offset.add(&tiny_offset.negated());
        let tiny_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(tiny_offset.clone()),
        };
        let cancelled_query = ExactEpochQuery {
            epoch: receive_epoch,
            offset: Arc::new(cancelled_offset.clone()),
        };
        let old_cancelled_difference = receive_epoch
            .exact_seconds()
            .sub(&receive_epoch.exact_seconds())
            .add(&tiny_offset)
            .sub(&cancelled_offset)
            .to_f64();
        assert_eq!(
            tiny_query.seconds_since_query(&cancelled_query).to_bits(),
            old_cancelled_difference.to_bits()
        );
    }

    #[test]
    fn exact_epoch_offsets_keep_decimal_remainders() {
        let receive = ExactEpoch::from_j2000_seconds(646_272_000.000_000_1).unwrap();
        let transmit = receive
            .checked_sub_seconds(0.070_712_000_000_000_01)
            .unwrap();
        assert_eq!(transmit.seconds_since(receive), -0.070_712_000_000_000_01);
        assert_eq!(
            transmit.checked_add_seconds(0.070_712_000_000_000_01),
            Some(receive)
        );
        assert_eq!(
            ExactEpoch::from_j2000_seconds(10.0)
                .unwrap()
                .checked_sub_seconds(0.1),
            ExactEpoch::from_civil(2000, 1, 1, 12, 0, 9.9)
        );
        assert_eq!(
            ExactEpoch::from_j2000_seconds(0.0)
                .unwrap()
                .checked_sub_seconds(1.0e-30)
                .unwrap()
                .sub_attosecond(),
            (-1, 12)
        );
    }

    #[test]
    fn exact_epoch_offset_cancellation_is_canonical_and_hashes_equally() {
        use std::hash::{Hash, Hasher};

        let tiny = ExactEpoch::J2000.checked_add_seconds(1.0e-30).unwrap();
        let cancelled = tiny.checked_sub_seconds(1.0e-30).unwrap();
        assert_eq!(cancelled, ExactEpoch::J2000);
        assert_eq!(cancelled.sub_attosecond(), (0, 0));
        let hash = |epoch: ExactEpoch| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            epoch.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(cancelled), hash(ExactEpoch::J2000));

        let tiny = ExactEpoch::J2000.checked_add_seconds(1.0e-100).unwrap();
        assert_eq!(tiny.checked_add_seconds(0.0), Some(tiny));
        let doubled = tiny.checked_add_seconds(1.0e-100).unwrap();
        assert_eq!(doubled.seconds_since(ExactEpoch::J2000), 2.0e-100);
        assert_eq!(doubled.sub_attosecond(), (2, 82));

        let negative = ExactEpoch::J2000.checked_sub_seconds(1.0e-30).unwrap();
        assert_eq!((negative.whole_seconds(), negative.attoseconds()), (0, 0));
        assert_eq!(negative.sub_attosecond(), (-1, 12));

        let large_remainder = ExactEpoch {
            seconds: 0,
            attoseconds: 0,
            residue: Residue {
                digits: i64::MAX,
                places: 100,
            },
        };
        assert_eq!(large_remainder.checked_add_seconds(1.0e-30), None);
    }

    #[test]
    fn exact_epoch_week_modulo_keeps_negative_sub_attosecond_fraction() {
        let boundary = ExactEpoch::from_j2000_seconds(604_800.0).unwrap();
        let just_before = boundary.checked_sub_seconds(1.0e-30).unwrap();
        assert_eq!(just_before.sub_attosecond(), (-1, 12));
        assert_eq!(just_before.seconds_modulo(604_800), Some(604_800.0));
        assert_eq!(
            boundary
                .checked_add_seconds(1.0e-30)
                .unwrap()
                .seconds_modulo(604_800),
            Some(1.0e-30)
        );
    }

    #[test]
    fn exact_epochs_read_the_shortest_decimal_and_keep_every_digit() {
        let noon = |second: f64| ExactEpoch::from_civil(2000, 1, 1, 12, 0, second).unwrap();
        let epoch = noon(0.1);
        assert_eq!(
            (epoch.whole_seconds(), epoch.attoseconds()),
            (0, 100_000_000_000_000_000)
        );
        assert_eq!(epoch.sub_attosecond(), (0, 0));
        // Digits past the attosecond are kept: the nearest attosecond and the
        // exact remainder, in [-1/2, 1/2) attosecond.
        let half = noon(0.000_000_000_000_000_002_5);
        assert_eq!((half.attoseconds(), half.sub_attosecond()), (3, (-5, 1)));
        let below = noon(0.000_000_000_000_000_003_25);
        assert_eq!((below.attoseconds(), below.sub_attosecond()), (3, (25, 2)));
        let tiny = noon(1.0e-30);
        assert_eq!((tiny.attoseconds(), tiny.sub_attosecond()), (0, (1, 12)));
        assert_eq!(tiny.j2000_seconds(), 1.0e-30);
        assert_eq!(tiny.seconds_since(ExactEpoch::J2000), 1.0e-30);
        let negative_tiny = noon(-1.0e-30);
        assert_eq!(
            (
                negative_tiny.whole_seconds(),
                negative_tiny.attoseconds(),
                negative_tiny.sub_attosecond()
            ),
            (0, 0, (-1, 12))
        );
        assert_eq!(negative_tiny.j2000_seconds(), -1.0e-30);
        assert_eq!(tiny.seconds_since(negative_tiny), 2.0e-30);
        let smallest = noon(f64::from_bits(1));
        assert_eq!(smallest.j2000_seconds(), f64::from_bits(1));
        // Just past half an attosecond below zero: the count steps down.
        let negative_above_half = noon(-0.000_000_000_000_000_000_75);
        assert_eq!(
            (
                negative_above_half.whole_seconds(),
                negative_above_half.attoseconds(),
                negative_above_half.sub_attosecond()
            ),
            (-1, ATTOSECONDS_PER_SECOND - 1, (25, 2))
        );
        assert_eq!(negative_above_half.j2000_seconds(), -7.5e-19);
        // Time order across the remainder and the grid.
        let ordered = [
            negative_above_half,
            negative_tiny,
            ExactEpoch::J2000,
            tiny,
            ExactEpoch::new(0, 1).unwrap(),
            below,
            epoch,
        ];
        for pair in ordered.windows(2) {
            assert!(pair[0] < pair[1], "{pair:?}");
            assert!(pair[1].seconds_since(pair[0]) > 0.0, "{pair:?}");
        }
        // The split of an epoch just before midnight belongs to the day that
        // ends there.
        let before_midnight = ExactEpoch::from_civil(2026, 9, 24, 0, 0, -1.0e-30).unwrap();
        let (jd_whole, fraction) = before_midnight.split_julian_date();
        assert_eq!((jd_whole, fraction), (2_461_306.5, 1.0));
        let (jd_whole, fraction) = ExactEpoch::from_civil(2026, 9, 23, 0, 0, 1.0e-30)
            .unwrap()
            .split_julian_date();
        assert_eq!(
            (jd_whole, fraction),
            super::super::civil::split_julian_date(2026, 9, 23, 0, 0, 1.0e-30)
        );
        // A negative second counts back from the minute.
        let before = noon(-0.25);
        assert_eq!(
            (before.whole_seconds(), before.attoseconds()),
            (-1, 750_000_000_000_000_000)
        );
        // A date that does not exist is refused, and every year is counted
        // on the proleptic Gregorian calendar.
        assert!(ExactEpoch::from_civil(2026, 13, 1, 0, 0, 0.0).is_none());
        assert!(ExactEpoch::from_civil(2026, 0, 1, 0, 0, 0.0).is_none());
        assert!(ExactEpoch::from_civil(2026, 2, 29, 0, 0, 0.0).is_none());
        assert!(ExactEpoch::from_civil(2024, 2, 29, 0, 0, 0.0).is_some());
        assert!(ExactEpoch::from_civil(2024, 4, 31, 0, 0, 0.0).is_none());
        // -5000-01-01 12:00 is 2,556,697 days before J2000 (Julian Day
        // -105152), a count the truncating division put a day off.
        assert_eq!(
            ExactEpoch::from_civil(-5000, 1, 1, 12, 0, 0.0)
                .unwrap()
                .whole_seconds(),
            -2_556_697 * 86_400
        );
        // Only a non-finite second and whole seconds past an i64 are refused
        // otherwise.
        assert!(ExactEpoch::from_civil(2000, 1, 1, 12, 0, f64::NAN).is_none());
        assert!(ExactEpoch::from_civil(2000, 1, 1, 12, 0, 1.0e300).is_none());
        assert!(ExactEpoch::from_civil(2000, 1, 1, 12, 0, 1.0e18).is_some());
        assert_eq!(ExactEpoch::new(5, ATTOSECONDS_PER_SECOND), None);
        assert!(ExactEpoch::new(5, ATTOSECONDS_PER_SECOND - 1).is_some());
        assert!(epoch > ExactEpoch::J2000 && before < ExactEpoch::J2000);
    }

    #[test]
    fn exact_epochs_with_a_remainder_difference_as_their_decimals() {
        // Seconds with digits below the attosecond: the time from J2000 is the
        // second itself, bit for bit, since its shortest decimal reads back to
        // it.
        let mut state = 0x7137_449a_2ec0_4b8b_u64;
        for _ in 0..20_000 {
            let significand = (xorshift(&mut state) % 100_000_000_000_000_000) as i128;
            let exponent = (xorshift(&mut state) % 60) as u32 + 18;
            let count = if xorshift(&mut state).is_multiple_of(2) {
                significand
            } else {
                -significand
            };
            let text = decimal_text(count, exponent);
            let second: f64 = text.parse().unwrap();
            let epoch = ExactEpoch::from_civil(2000, 1, 1, 12, 0, second).unwrap();
            assert_eq!(epoch.j2000_seconds().to_bits(), second.to_bits(), "{text}");
            assert_eq!(
                Some(epoch.cmp(&ExactEpoch::J2000)),
                second.partial_cmp(&0.0),
                "{text}"
            );
        }
    }

    #[test]
    fn exact_epochs_compare_intervals_with_a_threshold_exactly() {
        let at = |second: f64| ExactEpoch::from_civil(2026, 9, 23, 6, 30, second).unwrap();
        // Exactly half a second apart: not more than 0.5 s either way.
        assert!(!at(0.6).interval_exceeds(at(0.1), 0.5));
        assert!(!at(0.1).interval_exceeds(at(0.6), 0.5));
        assert!(at(0.6).interval_exceeds(at(0.1), 0.499_999_999_999_999_9));
        // A threshold is read as the decimal it states: a gap of three tenths
        // does not exceed a 0.3 s threshold, though the double 0.3 is below
        // three tenths, and it exceeds 0.29.
        assert!(!at(0.5).interval_exceeds(at(0.2), 0.3));
        assert!(at(0.5).interval_exceeds(at(0.2), 0.29));
        assert_eq!(
            at(0.5).compare_interval(at(0.2), 0.3),
            Some(Ordering::Equal)
        );
        assert!(!at(0.5).interval_exceeds(at(0.2), f64::INFINITY));
        assert!(!at(0.5).interval_exceeds(at(0.2), f64::NAN));
    }

    #[test]
    fn exact_queries_compare_attosecond_intervals_and_distance_ties() {
        let at_ninety = ExactEpoch::new(90, 0).unwrap().query();
        let at_zero = ExactEpoch::J2000.query();
        let one_attosecond = ExactSeconds::from_decimal(1, 18);
        let before = ExactEpochQuery {
            epoch: ExactEpoch::new(90, 0).unwrap(),
            offset: Arc::new(one_attosecond.negated()),
        };
        let after = ExactEpochQuery {
            epoch: ExactEpoch::new(90, 0).unwrap(),
            offset: Arc::new(one_attosecond),
        };

        assert_eq!(
            at_ninety.compare_interval_query(&before, 2.0e-18),
            Some(Ordering::Less)
        );
        assert_eq!(
            after.compare_interval_query(&at_ninety, 1.0e-18),
            Some(Ordering::Equal)
        );
        assert_eq!(
            at_ninety.compare_distance_to(&before, &after),
            Ordering::Equal
        );
        assert_eq!(
            after.compare_interval_query(&at_zero, 90.0),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn binary_epoch_interval_comparison_matches_exact_query_comparison() {
        let query = ExactEpoch::new(604_800, 0)
            .unwrap()
            .query()
            .checked_add_binary_seconds(-1.0e-30)
            .unwrap();
        let node_seconds = [
            -1.0e12,
            -0.1,
            -f64::from_bits(1),
            0.0,
            f64::from_bits(1),
            0.1,
            604_800.0,
            1.0e12,
        ];
        let thresholds = [
            -f64::INFINITY,
            -90.0,
            -1.0e-30,
            -0.0,
            0.0,
            1.0e-30,
            90.0,
            f64::INFINITY,
            f64::NAN,
        ];

        for node_s in node_seconds {
            let node_query = ExactEpoch::from_binary_j2000_seconds(node_s).unwrap();
            for threshold in thresholds {
                assert_eq!(
                    query.compare_interval_binary_j2000_seconds(node_s, threshold),
                    query.compare_interval_query(&node_query, threshold),
                    "node={node_s:?}, threshold={threshold:?}"
                );
            }
        }

        assert_eq!(
            query.compare_interval_binary_j2000_seconds(f64::INFINITY, 0.0),
            None
        );
        assert_eq!(
            query.compare_interval_binary_j2000_seconds(f64::NEG_INFINITY, 0.0),
            None
        );
    }
}
