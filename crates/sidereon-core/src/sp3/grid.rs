//! The exact epoch axis of an SP3 product and the grid its epochs lie on.
//!
//! An SP3 epoch record states its seconds to eight decimals, so every epoch a
//! file can hold is a whole number of 10-nanosecond ticks from the J2000
//! origin. Epochs are compared on that axis, never on floating seconds: two
//! epochs are one instant exactly when their ticks are equal, and a step
//! between epochs is an exact integer.
//!
//! [`product_grid`] is the one rule for which grid a product's epochs lie on.
//! The merge applies it to each input and [`Sp3::satellite_coverage`] reports
//! it, so the two cannot disagree about a product's cadence.
//!
//! [`epoch_interval_ticks`] is the one rule for whether `f64` seconds name an
//! interval on that axis. The merge's target grid, its input identity and the
//! exact-product cadence check all apply it. Merge output adds SP3's strict
//! upper bound; general header inspection remains permissive.
//!
//! The reader and writer hold the line-2 interval as the value its `F14.8`
//! field states. That field has the same 10-nanosecond resolution, so every
//! positive value it holds is an interval by this rule. It is also narrower,
//! below 100000 s as SP3 requires, and it holds zero and negative values: the
//! reader keeps what a file states rather than refusing the file, and a header
//! whose value is not an interval declares no grid ([`product_grid`]).
//! The merge refuses a computed grid step at or above the field's strict
//! 100000-second limit, preserving actual input cadence instead of silently
//! densifying it. The writer retains its own field-width refusal for products
//! whose public header has been changed after merging.

use core::fmt;

use super::write::epoch_tick;
use super::Sp3;

/// Ticks per second of the SP3 epoch axis (the `F11.8` seconds field).
pub(super) const TICKS_PER_SECOND: i128 = 100_000_000;
/// Ticks per day on the SP3 epoch axis.
pub(super) const TICKS_PER_DAY: i128 = 86_400 * TICKS_PER_SECOND;

/// Tick counts below this convert to `f64` seconds exactly: the count itself
/// is an exact `f64`, so [`interval_seconds`] rounds once.
const EXACT_TICK_LIMIT: i128 = 1 << 53;

/// The seconds a whole number of ticks spans, for a step or an interval. The
/// count is below 2^53 for any interval a header field can state, so the
/// conversion to `f64` is exact and the division rounds once, giving the value
/// the eight-decimal field reads back as.
pub(super) fn interval_seconds(ticks: i128) -> f64 {
    ticks as f64 / TICKS_PER_SECOND as f64
}

/// Why a value given in seconds is not an SP3 epoch interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Sp3EpochIntervalRejection {
    /// The value is NaN or an infinity.
    NotFinite,
    /// The value is zero or negative. An epoch interval is a step forward in
    /// time.
    NotPositive,
    /// The value is not the seconds of any whole number of the 10-nanosecond
    /// ticks an SP3 epoch record resolves: no eight-decimal seconds text reads
    /// back as it, as `600.0000000001` or `1e-9` does not.
    NotWholeTicks,
    /// The value is too large for `f64` seconds to name one whole number of
    /// ticks. From 2^26 s (about 777 days) adjacent tick counts can read back as
    /// the same `f64`, and a value that more than one count reads back as is
    /// refused rather than resolved to either. A value within a tick of 2^53
    /// ticks or past it (about 1042 days) is beyond the range where ticks and
    /// seconds convert exactly.
    BeyondTickResolution,
    /// The positive interval is whole ticks but is at least the SP3-d
    /// specification's strict 100000-second upper limit.
    OutsideSpecificationRange,
}

impl fmt::Display for Sp3EpochIntervalRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "it is not finite"),
            Self::NotPositive => write!(f, "it is not positive"),
            Self::NotWholeTicks => write!(
                f,
                "it is not a whole number of the 10-nanosecond ticks an SP3 epoch states"
            ),
            Self::BeyondTickResolution => write!(
                f,
                "at this size f64 seconds do not name one whole number of 10-nanosecond ticks"
            ),
            Self::OutsideSpecificationRange => {
                write!(
                    f,
                    "it is outside the SP3 requirement that intervals be below 100000 s"
                )
            }
        }
    }
}

/// A value refused as an SP3 epoch interval: the field that held it, the value
/// as supplied, and why.
///
/// An SP3 epoch interval is a positive whole number of the 10-nanosecond ticks
/// an epoch record's `F11.8` seconds field resolves, given in seconds as the
/// `f64` that the interval's eight-decimal text reads back as. `450.5` and
/// `0.00000003` are intervals; `600.0000000001` is not, although it is within a
/// microsecond of a whole second.
#[derive(Debug, Clone, Copy)]
pub struct Sp3EpochIntervalError {
    /// The option or field that held the value, such as
    /// `"target_epoch_interval_s"`.
    pub field: &'static str,
    /// The value as supplied.
    pub value: f64,
    /// Why it is not an epoch interval.
    pub reason: Sp3EpochIntervalRejection,
}

/// Values compare by their bits, so a refused NaN equals itself and the error
/// can sit in `Eq` error enums.
impl PartialEq for Sp3EpochIntervalError {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.value.to_bits() == other.value.to_bits()
            && self.reason == other.reason
    }
}

impl Eq for Sp3EpochIntervalError {}

impl fmt::Display for Sp3EpochIntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} s is not an SP3 epoch interval: {}",
            self.field, self.value, self.reason
        )
    }
}

impl std::error::Error for Sp3EpochIntervalError {}

/// The whole number of ticks an epoch interval states, or why `interval_s`
/// states none.
///
/// `interval_s` states `ticks` when it is the `f64` nearest `ticks` / 10^8 s:
/// the value the interval's eight-decimal text reads back as, and the value
/// [`interval_seconds`] gives for `ticks`. It must state exactly one positive
/// count below [`EXACT_TICK_LIMIT`].
pub(super) fn epoch_interval_ticks(interval_s: f64) -> Result<i128, Sp3EpochIntervalRejection> {
    if !interval_s.is_finite() {
        return Err(Sp3EpochIntervalRejection::NotFinite);
    }
    if interval_s <= 0.0 {
        return Err(Sp3EpochIntervalRejection::NotPositive);
    }
    // Every count that reads back as `interval_s` lies within one of this
    // estimate. Such a count differs from `interval_s * 10^8` by at most half
    // an `f64` spacing of `interval_s`, times 10^8: below 2^27 s that is under
    // 0.75. Forming the product rounds by at most half a spacing of a value
    // below 2^53, 0.5, and rounding to a whole number moves it at most 0.5
    // more, so the estimate is under 1.75 from the count; both are whole
    // numbers, so they differ by at most one.
    let estimate = (interval_s * TICKS_PER_SECOND as f64).round();
    if estimate >= (EXACT_TICK_LIMIT - 1) as f64 {
        return Err(Sp3EpochIntervalRejection::BeyondTickResolution);
    }
    let estimate = estimate as i128;
    let mut stated = None;
    for ticks in (estimate - 1).max(1)..=estimate + 1 {
        if interval_seconds(ticks) == interval_s {
            if stated.is_some() {
                return Err(Sp3EpochIntervalRejection::BeyondTickResolution);
            }
            stated = Some(ticks);
        }
    }
    stated.ok_or(Sp3EpochIntervalRejection::NotWholeTicks)
}

/// [`epoch_interval_ticks`] with the refusal typed for `field`.
pub(super) fn checked_epoch_interval_ticks(
    field: &'static str,
    interval_s: f64,
) -> Result<i128, Sp3EpochIntervalError> {
    epoch_interval_ticks(interval_s).map_err(|reason| Sp3EpochIntervalError {
        field,
        value: interval_s,
        reason,
    })
}

/// Check a merge output interval against both the exact tick rule and SP3-d's
/// strict `0 < interval < 100000 s` contract.
pub(super) fn checked_merge_epoch_interval_ticks(
    field: &'static str,
    interval_s: f64,
) -> Result<i128, Sp3EpochIntervalError> {
    let ticks = checked_epoch_interval_ticks(field, interval_s)?;
    if ticks >= 100_000 * TICKS_PER_SECOND {
        return Err(Sp3EpochIntervalError {
            field,
            value: interval_s,
            reason: Sp3EpochIntervalRejection::OutsideSpecificationRange,
        });
    }
    Ok(ticks)
}

/// Check a computed merge output interval against SP3-d's strict upper bound.
pub(super) fn checked_merge_output_ticks(
    field: &'static str,
    ticks: i128,
) -> Result<i128, Sp3EpochIntervalError> {
    let interval_s = interval_seconds(ticks);
    if ticks <= 0 {
        return Err(Sp3EpochIntervalError {
            field,
            value: interval_s,
            reason: Sp3EpochIntervalRejection::NotPositive,
        });
    }
    if ticks >= 100_000 * TICKS_PER_SECOND {
        return Err(Sp3EpochIntervalError {
            field,
            value: interval_s,
            reason: Sp3EpochIntervalRejection::OutsideSpecificationRange,
        });
    }
    Ok(ticks)
}

/// Seconds since J2000 for a whole number of ticks since J2000, whole seconds
/// and remainder converted separately so a whole-second epoch is exact.
pub(super) fn tick_seconds(ticks: i128) -> f64 {
    ticks.div_euclid(TICKS_PER_SECOND) as f64
        + ticks.rem_euclid(TICKS_PER_SECOND) as f64 / TICKS_PER_SECOND as f64
}

/// The whole number of ticks an interval states by [`epoch_interval_ticks`],
/// or `None` when it states none.
pub(super) fn interval_ticks(interval_s: f64) -> Option<i128> {
    epoch_interval_ticks(interval_s).ok()
}

/// Each epoch of `sp3` on the tick axis, in file order; `None` for an epoch no
/// SP3 record states exactly.
pub(super) fn product_ticks(sp3: &Sp3) -> Vec<Option<i128>> {
    sp3.epochs
        .iter()
        .map(|epoch| epoch_tick(epoch, sp3.header.time_system))
        .collect()
}

/// The grid a product's epochs lie on, as [`Sp3::satellite_coverage`] reports
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3EpochGrid {
    /// The grid step, seconds. When every step between consecutive epochs is
    /// the same, it is that step. When the steps differ, it is the header's
    /// declared interval if every step is a whole multiple of it - the product
    /// skips epochs of its declared grid - and `None` otherwise. With fewer than
    /// two placed epochs it is the header interval, if that states a positive
    /// whole number of ticks. Also `None` when epochs are out of order.
    pub interval_s: Option<f64>,
    /// Whether the header's declared interval is the grid step.
    pub agrees_with_header: bool,
    /// Indices into [`Sp3::epochs`] of epochs that do not follow the placed
    /// epoch before them in time (equal or earlier), in file order.
    pub out_of_order: Vec<usize>,
    /// Indices into [`Sp3::epochs`] of epochs no SP3 record states exactly: an
    /// instant that is not a whole number of 10-nanosecond ticks.
    pub unplaced: Vec<usize>,
}

/// The grid facts [`product_grid`] finds, on the tick axis.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct GridFacts {
    /// Grid step in ticks; see [`Sp3EpochGrid::interval_s`].
    pub(super) step: Option<i128>,
    /// The header interval in ticks, when it states a positive whole number.
    pub(super) header: Option<i128>,
    pub(super) out_of_order: Vec<usize>,
    pub(super) unplaced: Vec<usize>,
}

impl GridFacts {
    pub(super) fn public(&self) -> Sp3EpochGrid {
        Sp3EpochGrid {
            interval_s: self.step.map(interval_seconds),
            agrees_with_header: self.step.is_some() && self.step == self.header,
            out_of_order: self.out_of_order.clone(),
            unplaced: self.unplaced.clone(),
        }
    }
}

/// The grid epochs at `ticks` (file order) lie on, given the header's declared
/// interval.
///
/// Equal steps are a uniform grid of that step whatever the header says, so a
/// wrong or zero header interval does not misplace a uniform product. Unequal
/// steps are a grid only when every step is a whole multiple of the header's
/// declared interval: the product then skips epochs of that grid. A step that
/// is not - one shorter than the declared interval, or one no declared
/// interval divides - leaves the product on no grid. Epochs out of order leave
/// it on no grid too.
pub(super) fn product_grid(ticks: &[Option<i128>], header_interval_s: f64) -> GridFacts {
    let header = interval_ticks(header_interval_s);
    let mut unplaced = Vec::new();
    let mut out_of_order = Vec::new();
    let mut steps: Vec<i128> = Vec::new();
    let mut previous: Option<i128> = None;
    for (index, tick) in ticks.iter().enumerate() {
        let Some(tick) = *tick else {
            unplaced.push(index);
            continue;
        };
        if let Some(before) = previous {
            if tick <= before {
                out_of_order.push(index);
                continue;
            }
            steps.push(tick - before);
        }
        previous = Some(tick);
    }

    let step = if !out_of_order.is_empty() {
        None
    } else if steps.is_empty() {
        header
    } else if steps.iter().all(|&step| step == steps[0]) {
        Some(steps[0])
    } else {
        header.filter(|&declared| steps.iter().all(|&step| step % declared == 0))
    };
    GridFacts {
        step,
        header,
        out_of_order,
        unplaced,
    }
}

/// Greatest common divisor of two non-negative tick counts.
pub(super) fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::{
        checked_epoch_interval_ticks, checked_merge_epoch_interval_ticks,
        checked_merge_output_ticks, epoch_interval_ticks, interval_ticks, Sp3EpochIntervalError,
        Sp3EpochIntervalRejection, TICKS_PER_SECOND,
    };

    /// The tick count an eight-decimal seconds text states, read from its
    /// digits, and the `f64` the text reads back as. The count comes from the
    /// text alone, never from the rule under test.
    fn text_interval(text: &str) -> (i128, f64) {
        let (whole, fraction) = text.split_once('.').expect("eight-decimal text");
        assert_eq!(fraction.len(), 8, "{text}");
        let ticks = format!("{whole}{fraction}").parse::<i128>().unwrap();
        (ticks, text.parse::<f64>().unwrap())
    }

    #[test]
    fn every_eight_decimal_interval_states_its_own_tick_count() {
        for text in [
            "0.00000001",
            "0.00000002",
            "0.00000003",
            "0.00000007",
            "0.10000000",
            "0.29999999",
            "1.00000000",
            "1.50000000",
            "30.00000000",
            "300.00000000",
            "450.50000000",
            "599.99999999",
            "600.00000001",
            "900.00000000",
            "86400.00000000",
            "99999.99999999",
        ] {
            let (ticks, value) = text_interval(text);
            assert_eq!(epoch_interval_ticks(value), Ok(ticks), "{text}");
            assert_eq!(interval_ticks(value), Some(ticks), "{text}");
        }
        // Every whole multiple of 10 ns from 1 to 10^5 ticks, and a spread of
        // larger counts, written as its eight-decimal text.
        let larger = (1..=100_000_i128)
            .chain((0..2_000_i128).map(|index| 1_000_003 * index * index + 7 * index + 100_001));
        for ticks in larger {
            let text = format!("{}.{:08}", ticks / 100_000_000, ticks % 100_000_000);
            let (expected, value) = text_interval(&text);
            assert_eq!(expected, ticks);
            assert_eq!(epoch_interval_ticks(value), Ok(ticks), "{text}");
        }
    }

    #[test]
    fn a_value_off_the_tick_grid_is_refused_however_near_a_whole_second() {
        for value in [
            600.0000000001,
            599.9999999999,
            1.0e-9,
            5.0e-9,
            1.5e-8,
            0.123456789,
            300.000000005,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            assert_eq!(
                epoch_interval_ticks(value),
                Err(Sp3EpochIntervalRejection::NotWholeTicks),
                "{value:e}"
            );
        }
    }

    #[test]
    fn zero_negative_and_non_finite_values_are_refused_by_reason() {
        for value in [0.0, -0.0, -1.0e-8, -300.0, f64::MIN] {
            assert_eq!(
                epoch_interval_ticks(value),
                Err(Sp3EpochIntervalRejection::NotPositive),
                "{value:e}"
            );
        }
        for value in [f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                epoch_interval_ticks(value),
                Err(Sp3EpochIntervalRejection::NotFinite),
                "{value:e}"
            );
        }
    }

    #[test]
    fn merge_output_interval_obeys_the_specification_boundary() {
        let below = "99999.99999999".parse::<f64>().unwrap();
        let below_ticks = 9_999_999_999_999_i128;
        assert_eq!(
            checked_merge_epoch_interval_ticks("target_epoch_interval_s", below),
            Ok(below_ticks)
        );
        assert_eq!(
            checked_merge_output_ticks("merged_epoch_interval_s", below_ticks),
            Ok(below_ticks)
        );

        for (field, value, ticks) in [
            (
                "target_epoch_interval_s",
                100_000.0,
                10_000_000_000_000_i128,
            ),
            (
                "target_epoch_interval_s",
                100_000.00000001,
                10_000_000_000_001_i128,
            ),
        ] {
            let error = checked_merge_epoch_interval_ticks(field, value).unwrap_err();
            assert_eq!(error.field, field);
            assert_eq!(error.value, value);
            assert_eq!(
                error.reason,
                Sp3EpochIntervalRejection::OutsideSpecificationRange
            );
            assert_eq!(
                checked_merge_output_ticks("merged_epoch_interval_s", ticks),
                Err(Sp3EpochIntervalError {
                    field: "merged_epoch_interval_s",
                    value,
                    reason: Sp3EpochIntervalRejection::OutsideSpecificationRange,
                })
            );
        }
        assert_eq!(100_000 * TICKS_PER_SECOND, 10_000_000_000_000_i128);
    }

    #[test]
    fn a_value_that_names_more_than_one_tick_count_is_refused() {
        // 2^26 s is exactly 6_710_886_400_000_000 ticks and no neighbouring
        // count reads back as it; 7e7 s is exactly 7e15 ticks, likewise.
        assert_eq!(
            epoch_interval_ticks(67_108_864.0),
            Ok(6_710_886_400_000_000)
        );
        assert_eq!(epoch_interval_ticks(7.0e7), Ok(7_000_000_000_000_000));
        // The next f64 above 2^26 s, 0x1.0000000000001p+26, lies within half a
        // spacing of both 6_710_886_400_000_001 and 6_710_886_400_000_002
        // ticks, so it names neither.
        let shared = f64::from_bits(0x4190_0000_0000_0001);
        assert_eq!(
            epoch_interval_ticks(shared),
            Err(Sp3EpochIntervalRejection::BeyondTickResolution)
        );
        // Within a tick of 2^53 ticks or past it, ticks and seconds no longer
        // convert exactly.
        for value in [1.0e8, 90071992.5474099, 1.0e300, f64::MAX] {
            assert_eq!(
                epoch_interval_ticks(value),
                Err(Sp3EpochIntervalRejection::BeyondTickResolution),
                "{value:e}"
            );
        }
        // 2^53 - 3 ticks is below the limit and states only itself.
        assert_eq!(
            epoch_interval_ticks(90071992.54740989),
            Ok(9_007_199_254_740_989)
        );
    }

    #[test]
    fn a_refusal_names_the_field_the_value_and_the_reason() {
        let error = checked_epoch_interval_ticks("target_epoch_interval_s", 600.0000000001)
            .expect_err("off the tick grid");
        assert_eq!(
            error,
            Sp3EpochIntervalError {
                field: "target_epoch_interval_s",
                value: 600.0000000001,
                reason: Sp3EpochIntervalRejection::NotWholeTicks,
            }
        );
        assert!(
            error
                .to_string()
                .starts_with("target_epoch_interval_s 600.0000000001 s"),
            "{error}"
        );
        // A refused NaN compares equal to itself, so the error is `Eq`.
        let nan = checked_epoch_interval_ticks("target_epoch_interval_s", f64::NAN).unwrap_err();
        let same = nan;
        assert_eq!(nan, same);
        assert_eq!(nan.reason, Sp3EpochIntervalRejection::NotFinite);
    }
}
