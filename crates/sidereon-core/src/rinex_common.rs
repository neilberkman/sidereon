//! Shared RINEX header concepts used across the RINEX family of readers.
//!
//! The observation, navigation, and clock readers all decode the same RINEX
//! header time-system field. Keeping the label mapping here (rather than
//! duplicating it per reader, or borrowing it from an unrelated parser module)
//! gives the family one canonical, format-faithful decode and serialize point.

use crate::astro::time::model::TimeScale;

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
pub(crate) fn time_scale_rinex_label(scale: TimeScale) -> &'static str {
    match scale {
        TimeScale::Gpst => "GPS",
        TimeScale::Qzsst => "QZS",
        TimeScale::Gst => "GAL",
        TimeScale::Bdt => "BDT",
        TimeScale::Utc => "UTC",
        TimeScale::Tai => "TAI",
        // No RINEX header label maps to these scales; GPST is the RINEX default
        // time system and keeps a serialized header parseable.
        TimeScale::Tt | TimeScale::Tdb | TimeScale::Glonasst => "GPS",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_labels_round_trip() {
        for label in ["GPS", "QZS", "GAL", "BDT", "UTC", "TAI"] {
            let scale = time_scale_label(label).expect("known label");
            assert_eq!(time_scale_rinex_label(scale), label);
        }
    }

    #[test]
    fn glonass_label_is_utc() {
        assert_eq!(time_scale_label("GLO"), Some(TimeScale::Utc));
        // UTC serializes back as UTC (GLO and UTC are indistinguishable).
        assert_eq!(time_scale_rinex_label(TimeScale::Utc), "UTC");
    }

    #[test]
    fn unknown_label_is_none() {
        assert_eq!(time_scale_label("XYZ"), None);
        assert_eq!(time_scale_label(""), None);
    }
}
