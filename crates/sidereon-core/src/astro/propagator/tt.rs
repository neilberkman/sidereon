//! Shared TT/TDB epoch mapping for numerical and variational propagation.
//!
//! Numerical propagation integrates the equations of motion with Terrestrial
//! Time as the independent variable (IERS Conventions 2010, chapter 10),
//! carried as elapsed TT seconds since the initial epoch so each step length
//! is exact. Force models receive absolute TDB seconds past J2000: at a
//! segment endpoint its own TDB, and within a segment the starting endpoint's
//! TDB advanced by elapsed TT and by the change of TDB - TT, clamped to the
//! segment's TDB span so a body-fixed frame provider covering the endpoints
//! covers every evaluation.

use crate::astro::time::scales::{tdb_minus_tt_seconds, tt_from_tdb_j2000_seconds};

/// An epoch on both scales: TDB seconds past J2000 and TT seconds elapsed
/// since a propagation's initial epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TtAnchor {
    pub(crate) tdb: f64,
    pub(crate) elapsed: f64,
}

/// One integration segment between two anchored epochs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TtSegment {
    pub(crate) from: TtAnchor,
    pub(crate) to: TtAnchor,
}

impl TtSegment {
    pub(crate) fn start(&self) -> f64 {
        self.from.elapsed
    }

    pub(crate) fn target(&self) -> f64 {
        self.to.elapsed
    }
}

/// Mapping between elapsed TT seconds and absolute TDB seconds anchored at an
/// initial epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TtEpochMap {
    /// TDB seconds past J2000 of the initial epoch.
    initial_tdb: f64,
    /// TT seconds past J2000 of the initial epoch, used as the argument of
    /// TDB - TT only: an error of a few rounding units moves TDB - TT by less
    /// than 1e-16 s.
    tt_reference: f64,
    /// TDB - TT in seconds at `tt_reference`.
    initial_tdb_minus_tt: f64,
}

impl TtEpochMap {
    /// Anchor a mapping at `initial_tdb_j2000_s` and return the mapping with
    /// the starting anchor at zero elapsed TT.
    pub(crate) fn from_initial_tdb(initial_tdb_j2000_s: f64) -> (Self, TtAnchor) {
        let tt_reference = tt_from_tdb_j2000_seconds(initial_tdb_j2000_s);
        let initial_tdb_minus_tt = tdb_minus_tt_seconds(tt_reference);
        (
            Self {
                initial_tdb: initial_tdb_j2000_s,
                tt_reference,
                initial_tdb_minus_tt,
            },
            TtAnchor {
                tdb: initial_tdb_j2000_s,
                elapsed: 0.0,
            },
        )
    }

    /// Map a target TDB epoch past J2000 to its elapsed-TT anchor relative to
    /// this mapping's initial epoch.
    ///
    /// Forms elapsed TT as
    /// `(target - initial_tdb) - (tdb_minus_tt(tt_target) - tdb_minus_tt(tt_initial))`
    /// so no rounded absolute TT values are subtracted.
    pub(crate) fn anchor_for_tdb(&self, tdb_j2000_s: f64) -> TtAnchor {
        let tt_target = tt_from_tdb_j2000_seconds(tdb_j2000_s);
        TtAnchor {
            tdb: tdb_j2000_s,
            elapsed: (tdb_j2000_s - self.initial_tdb)
                - (tdb_minus_tt_seconds(tt_target) - self.initial_tdb_minus_tt),
        }
    }

    /// The change of TDB - TT from `from` to `to` TT seconds elapsed after the
    /// initial epoch.
    pub(crate) fn tdb_minus_tt_change(&self, from: f64, to: f64) -> f64 {
        tdb_minus_tt_seconds(self.tt_reference + to)
            - tdb_minus_tt_seconds(self.tt_reference + from)
    }

    /// The TDB epoch a force model receives `elapsed` TT seconds into the
    /// propagation, within `segment`.
    pub(crate) fn tdb_at(&self, segment: &TtSegment, elapsed: f64) -> f64 {
        let TtSegment { from, to } = *segment;
        if elapsed == to.elapsed {
            return to.tdb;
        }
        if elapsed == from.elapsed {
            return from.tdb;
        }
        let tdb =
            from.tdb + ((elapsed - from.elapsed) + self.tdb_minus_tt_change(from.elapsed, elapsed));
        tdb.max(from.tdb.min(to.tdb)).min(from.tdb.max(to.tdb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, SECONDS_PER_DAY};
    use crate::astro::propagator::VariationalEpoch;
    use crate::astro::time::civil::civil_fields_from_j2000_seconds;
    use crate::astro::time::model::TimeScale;
    use crate::astro::time::scales::TimeScales;

    /// Independent evaluation of the Astronomical Almanac TDB - TT series
    /// directly from its published terms.
    fn independent_tdb_minus_tt_s(tt_j2000_s: f64) -> f64 {
        let t = tt_j2000_s / (DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY);
        0.001657 * libm::sin(628.3076 * t + 6.2401)
            + 0.000022 * libm::sin(575.3385 * t + 4.2970)
            + 0.000014 * libm::sin(1256.6152 * t + 6.1969)
            + 0.000005 * libm::sin(606.9777 * t + 4.0212)
            + 0.000005 * libm::sin(52.9691 * t + 0.4444)
            + 0.000002 * libm::sin(21.3299 * t + 5.5431)
            + 0.000010 * t * libm::sin(628.3076 * t + 4.2490)
    }

    /// Independent elapsed TT between two TDB epochs past J2000, solving
    /// `dt_tt + (f(tt0 + dt_tt) - f(tt0)) = target - initial` by fixed-point
    /// iteration without subtracting rounded absolute TT values.
    fn independent_elapsed_tt(initial_tdb: f64, target_tdb: f64) -> f64 {
        let dt_tdb = target_tdb - initial_tdb;
        let mut tt0 = initial_tdb;
        for _ in 0..4 {
            tt0 = initial_tdb - independent_tdb_minus_tt_s(tt0);
        }
        let f0 = independent_tdb_minus_tt_s(tt0);
        let mut dt_tt = dt_tdb;
        for _ in 0..4 {
            let f1 = independent_tdb_minus_tt_s(tt0 + dt_tt);
            dt_tt = dt_tdb - (f1 - f0);
        }
        dt_tt
    }

    /// Independent elapsed TT via split-Julian-date `TimeScales`.
    fn time_scales_elapsed_tt(initial_tdb: f64, target_tdb: f64) -> f64 {
        let ts = |tdb: f64| {
            let (y, m, d, h, min, s) = civil_fields_from_j2000_seconds(tdb).expect("civil");
            TimeScales::from_scale(TimeScale::Tdb, y, m, d, h, min, s).expect("time scales")
        };
        let a = ts(initial_tdb);
        let b = ts(target_tdb);
        (b.jd_whole - a.jd_whole) * SECONDS_PER_DAY
            + (b.tt_fraction - a.tt_fraction) * SECONDS_PER_DAY
    }

    #[test]
    fn elapsed_tt_avoids_absolute_tt_rounding_cancellation() {
        let cases = [
            (600_019_700.0, 600_021_500.0),
            (646_315_200.0, 646_315_260.0),
            (831_902_400.0, 831_903_300.0),
            (831_902_400.0, 831_901_500.0),
            (0.0, 1_800.0),
            (600_019_700.0, 600_019_700.0),
        ];
        for (initial, target) in cases {
            let (map, start) = TtEpochMap::from_initial_tdb(initial);
            assert_eq!(start.tdb.to_bits(), initial.to_bits());
            assert_eq!(start.elapsed.to_bits(), 0.0_f64.to_bits());

            let anchor = map.anchor_for_tdb(target);
            let variational = VariationalEpoch::from_tdb_seconds(initial, target);
            assert_eq!(anchor.elapsed.to_bits(), variational.elapsed_tt_s.to_bits());
            assert_eq!(anchor.tdb.to_bits(), variational.tdb_j2000_s.to_bits());

            let expected = independent_elapsed_tt(initial, target);
            let err = (anchor.elapsed - expected).abs();
            assert!(
                err <= 1.0e-13,
                "initial={initial}, target={target}: got {:.15}, expected {:.15}, err={err:e}",
                anchor.elapsed,
                expected
            );

            let ts_elapsed = time_scales_elapsed_tt(initial, target);
            let ts_err = (anchor.elapsed - ts_elapsed).abs();
            assert!(
                ts_err <= 1.0e-10,
                "initial={initial}, target={target}: TimeScales diff {ts_err:e}"
            );
        }

        // The 60 s span at 646_315_200.0 must preserve the ~1.946e-8 s TT-TDB
        // difference rather than rounding to 60.0 s.
        let (map, _) = TtEpochMap::from_initial_tdb(646_315_200.0);
        let sixty = map.anchor_for_tdb(646_315_260.0).elapsed;
        assert_ne!(sixty, 60.0);
        assert!(
            ((sixty - 60.0) - 1.946e-8).abs() < 1.0e-10,
            "sixty={sixty:.15}"
        );
    }
}
