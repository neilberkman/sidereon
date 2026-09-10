//! Shared RINEX header concepts used across the RINEX family of readers.
//!
//! The observation, navigation, and clock readers all decode the same RINEX
//! header time-system field. Keeping the label mapping here (rather than
//! duplicating it per reader, or borrowing it from an unrelated parser module)
//! gives the family one canonical, format-faithful decode and serialize point.

use std::collections::BTreeMap;

use crate::astro::time::civil::j2000_seconds;
use crate::astro::time::model::TimeScale;
use crate::rinex_obs::ObsEpochTime;

/// Map a RINEX header time-system label onto the core [`TimeScale`].
///
/// Returns `None` for an empty or unrecognized label so each reader can apply
/// its own policy for the unknown case (the observation/clock readers default a
/// blank label to GPST, an explicit unknown label is rejected). The RINEX/SP3
/// `GLO` time system is defined as UTC (RINEX 3 spec), not the UTC+3h GLONASS
/// system time: [`TimeScale::Glonasst`] is a conversion-only variant and is
/// never produced by parsing a file's time-system label.
pub(crate) fn time_scale_label(label: &str) -> Option<TimeScale> {
    match label.trim() {
        "GPS" => Some(TimeScale::Gpst),
        "QZS" => Some(TimeScale::Qzsst),
        "GLO" => Some(TimeScale::Utc),
        "GAL" => Some(TimeScale::Gst),
        "BDT" => Some(TimeScale::Bdt),
        "UTC" => Some(TimeScale::Utc),
        "TAI" => Some(TimeScale::Tai),
        _ => None,
    }
}

/// The canonical RINEX header time-system label for a core [`TimeScale`].
///
/// The inverse of [`time_scale_label`] for the scales a RINEX header can name,
/// used by the serializers so a parsed product round-trips its time system.
/// `GLO` is intentionally not emitted: a `GLO` label decodes to [`TimeScale::Utc`]
/// (the two are indistinguishable after parsing), so `Utc` serializes as `UTC`.
pub(crate) fn time_scale_rinex_label(scale: TimeScale) -> Option<&'static str> {
    match scale {
        TimeScale::Gpst => Some("GPS"),
        TimeScale::Qzsst => Some("QZS"),
        TimeScale::Gst => Some("GAL"),
        TimeScale::Bdt => Some("BDT"),
        TimeScale::Utc => Some("UTC"),
        TimeScale::Tai => Some("TAI"),
        TimeScale::Tt | TimeScale::Tcg | TimeScale::Tdb | TimeScale::Tcb | TimeScale::Glonasst => {
            None
        }
    }
}

pub(crate) fn obs_epoch_seconds(epoch: ObsEpochTime) -> f64 {
    j2000_seconds(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    )
}

/// Columns and decimals of the observation `INTERVAL` header field (`F10.3`).
pub(crate) const OBS_INTERVAL_WIDTH: usize = 10;
pub(crate) const OBS_INTERVAL_DECIMALS: usize = 3;

/// Whether the `INTERVAL` header field can record this cadence exactly.
///
/// A cadence the field cannot carry must never be written: the line would
/// overrun its ten columns and the file would no longer read back as the same
/// product. Repair therefore declines to adopt such a cadence, the writer omits
/// it, and the parser rejects one it finds in a file. An inferred cadence is a
/// multiple of a millisecond, so only a very long one - epochs a year apart, say
/// - runs out of columns.
pub(crate) fn writable_obs_interval_s(interval_s: f64) -> bool {
    // A non-finite interval formats as `NaN` or `inf`, which the parser refuses
    // outright, so it is unwritable rather than merely imprecise. The shared
    // representability rule carries non-finite values through for the callers
    // that model them separately, so exclude them here.
    interval_s.is_finite()
        && crate::validate::representable_in_fixed_field(
            interval_s,
            Some(OBS_INTERVAL_WIDTH),
            OBS_INTERVAL_DECIMALS,
        )
}

/// Whether a declared RINEX observation interval can be used as a cadence.
///
/// `INTERVAL` is optional product metadata. RINEX permits zero to represent an
/// unknown header item; zero and other unusable values remain available to lint
/// and repair but must never be used for gap calculations.
pub(crate) fn usable_obs_interval_s(interval_s: f64) -> bool {
    interval_s.is_finite() && interval_s > 0.0
}

pub(crate) fn dominant_obs_interval_s(times: &[ObsEpochTime]) -> Option<f64> {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for pair in times.windows(2) {
        let delta_s = obs_epoch_seconds(pair[1]) - obs_epoch_seconds(pair[0]);
        if !delta_s.is_finite() || delta_s <= 0.0 {
            continue;
        }
        let delta_ms = (delta_s * 1000.0).round();
        if !delta_ms.is_finite() || delta_ms < 1.0 || delta_ms > i64::MAX as f64 {
            continue;
        }
        *counts.entry(delta_ms as i64).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(delta_ms, count)| (*count, -(*delta_ms)))
        .map(|(delta_ms, _)| delta_ms as f64 / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_labels_round_trip() {
        for label in ["GPS", "QZS", "GAL", "BDT", "UTC", "TAI"] {
            let scale = time_scale_label(label).expect("known label");
            assert_eq!(time_scale_rinex_label(scale), Some(label));
        }
    }

    #[test]
    fn glonass_label_is_utc() {
        assert_eq!(time_scale_label("GLO"), Some(TimeScale::Utc));
        // UTC serializes back as UTC (GLO and UTC are indistinguishable).
        assert_eq!(time_scale_rinex_label(TimeScale::Utc), Some("UTC"));
    }

    #[test]
    fn unsupported_labels_are_none() {
        for scale in [
            TimeScale::Tt,
            TimeScale::Tcg,
            TimeScale::Tdb,
            TimeScale::Tcb,
            TimeScale::Glonasst,
        ] {
            assert_eq!(time_scale_rinex_label(scale), None);
        }
    }

    #[test]
    fn unknown_label_is_none() {
        assert_eq!(time_scale_label("XYZ"), None);
        assert_eq!(time_scale_label(""), None);
    }

    #[test]
    fn usable_observation_interval_requires_positive_finite_seconds() {
        assert!(usable_obs_interval_s(30.0));
        for unusable in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!usable_obs_interval_s(unusable), "{unusable:?}");
        }
    }

    #[test]
    fn writable_observation_interval_is_bounded_by_the_header_field() {
        // F10.3: ten columns, three decimals.
        assert!(writable_obs_interval_s(30.0));
        assert!(writable_obs_interval_s(0.001));
        assert!(writable_obs_interval_s(999_999.999));
        for unwritable in [
            // Needs an eleventh column.
            1_000_000.0,
            // Epochs a year apart; the cadence repair would otherwise infer.
            31_536_000.0,
            // Carries precision the three decimals drop.
            0.000_4,
            1e-300,
            f64::NAN,
            f64::INFINITY,
        ] {
            assert!(!writable_obs_interval_s(unwritable), "{unwritable:?}");
        }
    }

    #[test]
    fn dominant_observation_interval_never_rounds_to_zero() {
        let epoch = |second| ObsEpochTime {
            year: 2026,
            month: 7,
            day: 24,
            hour: 0,
            minute: 0,
            second,
        };
        assert_eq!(dominant_obs_interval_s(&[epoch(0.0), epoch(0.0004)]), None);
        assert_eq!(
            dominant_obs_interval_s(&[epoch(0.0), epoch(0.0006)]),
            Some(0.001)
        );
    }
}
