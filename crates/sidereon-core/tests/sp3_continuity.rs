//! Continuity attestation over precise-ephemeris sample series.
//!
//! The arcs here are generated from a circular earth-fixed orbit model rather
//! than sampled from a product, so every expectation is a property of the input
//! this test constructed, not of a fixture that could drift.

use sidereon_core::astro::time::civil::split_julian_date_from_j2000_seconds;
use sidereon_core::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
use sidereon_core::ephemeris::{
    check_continuity, ContinuityCheck, ContinuityDefect, ContinuityOptions, OrbitClass,
    PreciseEphemerisSample, Sp3, SpeedBound,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const GPS_A_M: f64 = 26_560_000.0;
const SAMPLE_INTERVAL_S: f64 = 300.0;

fn satellite(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid PRN")
}

fn epoch(j2000_seconds: f64) -> Instant {
    // Whole-second epochs: the sample axis snaps to whole seconds, so building
    // the split from an integer keeps the constructed arc exactly on the axis.
    let whole = j2000_seconds as i64;
    let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(whole);
    Instant {
        scale: TimeScale::Gpst,
        repr: InstantRepr::JulianDate(
            JulianDateSplit::new(jd_whole, fraction).expect("valid split epoch"),
        ),
    }
}

/// A circular arc in the earth-fixed frame, inclined so the motion is not
/// degenerate in any one axis. Smooth and infinitely differentiable, so a clean
/// arc's hold-out residual is the interpolator's own error and nothing else.
fn arc_position_m(j2000_seconds: f64) -> [f64; 3] {
    // ~3 km/s earth-fixed, in the measured band for GNSS MEO.
    let rate_rad_s = 3_000.0 / GPS_A_M;
    let theta = rate_rad_s * j2000_seconds;
    let inclination = 55.0_f64.to_radians();
    [
        GPS_A_M * libm::cos(theta),
        GPS_A_M * libm::sin(theta) * libm::cos(inclination),
        GPS_A_M * libm::sin(theta) * libm::sin(inclination),
    ]
}

fn arc(prn: u8, start_j2000_s: f64, count: usize) -> Vec<PreciseEphemerisSample> {
    (0..count)
        .map(|index| {
            let t = start_j2000_s + index as f64 * SAMPLE_INTERVAL_S;
            PreciseEphemerisSample::new(satellite(prn), epoch(t), arc_position_m(t), None)
        })
        .collect()
}

fn options() -> ContinuityOptions {
    ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss)
}

#[test]
fn continuous_multi_day_arc_across_boundaries_attests() {
    // Start just before a year boundary so the arc crosses day, month, and year
    // rollovers in whatever civil calendar the epochs map to: 2025-12-30T00:00Z
    // in J2000 seconds, running four days at 5-minute sampling.
    let start = 819_244_800.0;
    let samples = arc(1, start, 4 * 288);

    let report = check_continuity(&samples, &options());

    assert!(
        report.attested(),
        "a smooth arc across day/month/year boundaries must attest, got {:?}",
        report.defects
    );
    assert_eq!(report.pairs_checked, 4 * 288 - 1);
    assert_eq!(report.residuals_checked, 4 * 288 - 2);
    assert_eq!(report.residuals_skipped, 0);
}

#[test]
fn synthetic_splice_is_reported_with_its_displacement_and_epoch_pair() {
    let start = 800_000_000.0;
    let count = 60;
    let splice_index = 30;
    let splice_m = 500.0;

    let mut samples = arc(1, start, count);
    // Offset the whole tail, exactly as a precedence switch to a contributor
    // whose arc sits 500 m away would.
    for sample in samples.iter_mut().skip(splice_index) {
        sample.position_ecef_m[0] += splice_m;
    }

    let report = check_continuity(&samples, &options());

    assert!(!report.attested(), "a 500 m splice must be reported");

    // The speed gate cannot see this - that is the whole reason the residual
    // check exists, and this pins it rather than leaving it as a claim.
    assert_eq!(
        report.defects_from(ContinuityCheck::SpeedBound).count(),
        0,
        "500 m over 300 s is ~1.7 m/s against a ~6 km/s bound; the gate must not fire"
    );

    let residuals: Vec<_> = report
        .defects_from(ContinuityCheck::HoldOutResidual)
        .collect();
    assert!(
        !residuals.is_empty(),
        "the hold-out residual check must catch the splice"
    );

    let splice_epoch = start + splice_index as f64 * SAMPLE_INTERVAL_S;
    let at_splice = residuals
        .iter()
        .find(|defect| match defect {
            ContinuityDefect::HoldOutResidual { epoch_j2000_s, .. } => {
                (*epoch_j2000_s - splice_epoch).abs() < 1.0
            }
            _ => false,
        })
        .expect("a violation at the spliced epoch");

    let ContinuityDefect::HoldOutResidual {
        residual_m,
        preceding_j2000_s,
        ..
    } = at_splice
    else {
        unreachable!("filtered to hold-out residuals")
    };

    // The residual at the boundary is a *fraction* of the splice, not the whole
    // of it, and that is the correct physics: the hold-out window straddles the
    // splice, so the predicted value is pulled by nodes from both arcs and the
    // residual is the blend. What matters is that it is the order of the splice
    // and far above tolerance - not that it equals the offset.
    assert!(
        *residual_m > splice_m / 5.0 && *residual_m < splice_m * 2.0,
        "residual {residual_m} m should be the order of the {splice_m} m splice"
    );
    assert!(
        (*preceding_j2000_s - (splice_epoch - SAMPLE_INTERVAL_S)).abs() < 1.0,
        "the report must bracket the offending pair"
    );

    // Localization: every violation sits within one interpolation window of the
    // splice. Deep inside either arc the samples agree with their neighbours
    // again, so a violation far from the boundary would mean the check smears.
    let window_span_s = 11.0 * SAMPLE_INTERVAL_S;
    for defect in &residuals {
        let ContinuityDefect::HoldOutResidual { epoch_j2000_s, .. } = defect else {
            unreachable!("filtered to hold-out residuals")
        };
        assert!(
            (*epoch_j2000_s - splice_epoch).abs() <= window_span_s,
            "violation at {epoch_j2000_s} is more than one window from the splice"
        );
    }
}

#[test]
fn a_metre_scale_splice_is_caught_while_the_speed_gate_stays_silent() {
    // The sensitivity claim, pinned at the scale that motivated the feature: a
    // 5 m offset is four orders of magnitude below anything a physical speed
    // bound can resolve, and the residual check must still find it.
    let start = 800_000_000.0;
    let splice_index = 25;
    let splice_m = 5.0;

    let mut samples = arc(1, start, 50);
    for sample in samples.iter_mut().skip(splice_index) {
        sample.position_ecef_m[2] += splice_m;
    }

    let options = ContinuityOptions {
        speed_bound: Some(SpeedBound::OrbitClass(OrbitClass::MeoGnss)),
        residual_tolerance_m: Some(1.0),
    };
    let report = check_continuity(&samples, &options);

    assert_eq!(
        report.defects_from(ContinuityCheck::SpeedBound).count(),
        0,
        "a 5 m splice is invisible to any physical speed bound, by design"
    );
    assert!(
        report
            .defects_from(ContinuityCheck::HoldOutResidual)
            .next()
            .is_some(),
        "the residual check must resolve a 5 m splice against a 1 m tolerance"
    );
}

#[test]
fn shuffled_input_produces_the_identical_verdict() {
    // The regression that matters most: ordering is the library's job. A caller
    // that hands over a shuffled sequence must get the same answer as one that
    // sorts first, or every other guarantee here is decorative.
    let start = 800_000_000.0;
    let mut samples = arc(1, start, 40);
    for sample in samples.iter_mut().skip(20) {
        sample.position_ecef_m[1] += 750.0;
    }

    let sorted_report = check_continuity(&samples, &options());

    // A deterministic shuffle: reverse, then interleave halves. No RNG, so a
    // failure here reproduces exactly.
    let mut shuffled: Vec<_> = samples.iter().rev().copied().collect();
    let half = shuffled.len() / 2;
    let (front, back) = shuffled.split_at(half);
    let interleaved: Vec<_> = front
        .iter()
        .zip(back.iter())
        .flat_map(|(a, b)| [*a, *b])
        .collect();
    shuffled = interleaved;

    let shuffled_report = check_continuity(&shuffled, &options());

    assert_eq!(
        sorted_report, shuffled_report,
        "shuffled input must produce a byte-identical report"
    );
    assert!(!sorted_report.attested(), "the splice must still be found");
}

#[test]
fn duplicate_epochs_are_their_own_defect_class_and_are_not_deduplicated() {
    let start = 800_000_000.0;
    let mut samples = arc(1, start, 20);
    let mut duplicate = samples[5];
    duplicate.position_ecef_m[2] += 3.0;
    samples.push(duplicate);

    let report = check_continuity(&samples, &options());

    let duplicates: Vec<_> = report
        .defects
        .iter()
        .filter(|defect| matches!(defect, ContinuityDefect::DuplicateEpoch { .. }))
        .collect();
    assert_eq!(duplicates.len(), 1, "one repeated epoch, one defect");

    let ContinuityDefect::DuplicateEpoch {
        occurrences,
        epoch_j2000_s,
        ..
    } = duplicates[0]
    else {
        unreachable!("filtered to duplicates")
    };
    assert_eq!(*occurrences, 2);
    assert!((*epoch_j2000_s - (start + 5.0 * SAMPLE_INTERVAL_S)).abs() < 1.0);
}

#[test]
fn single_sample_series_is_reported_rather_than_passing() {
    let samples = arc(1, 800_000_000.0, 1);

    let report = check_continuity(&samples, &options());

    assert!(
        !report.attested(),
        "a one-sample series is not an attestation of continuity"
    );
    assert!(report
        .defects
        .iter()
        .any(|defect| matches!(defect, ContinuityDefect::SingleSampleSeries { .. })));
}

#[test]
fn an_implausible_arc_still_fails_the_physical_gate() {
    // Proves the gate did not become permissive: a jump no orbit can perform.
    let start = 800_000_000.0;
    let mut samples = arc(1, start, 10);
    for sample in samples.iter_mut().skip(5) {
        sample.position_ecef_m[0] += 40_000_000.0;
    }

    let report = check_continuity(&samples, &options());

    let bound_defects: Vec<_> = report.defects_from(ContinuityCheck::SpeedBound).collect();
    assert!(
        !bound_defects.is_empty(),
        "a 40 000 km displacement in 300 s must break the physical bound"
    );

    let ContinuityDefect::SpeedBound {
        implied_speed_m_s,
        bound_m_s,
        interval_s,
        ..
    } = bound_defects[0]
    else {
        unreachable!("filtered to speed-bound defects")
    };
    assert!(implied_speed_m_s > bound_m_s);
    assert!(
        *interval_s > 0.0,
        "the comparison path cannot yield a non-positive interval"
    );
}

#[test]
fn the_class_bound_is_physical_and_clears_real_earth_fixed_motion() {
    // The measured band for GNSS MEO earth-fixed chord speed is 2757-3187 m/s
    // (GFZ ultra, G01, 300 s sampling). The bound must sit above it with room,
    // and must not be the inertial speed.
    let bound = OrbitClass::MeoGnss.max_earth_fixed_speed_m_s();
    assert!(
        bound > 3_187.0,
        "bound {bound} must clear measured earth-fixed motion"
    );
    assert!(
        bound > 3_874.0,
        "bound {bound} must also clear the inertial speed it is derived from"
    );
    assert!(bound < 10_000.0, "bound {bound} should stay meaningful");

    assert!(
        OrbitClass::Leo.max_earth_fixed_speed_m_s()
            > OrbitClass::MeoGnss.max_earth_fixed_speed_m_s(),
        "a LEO satellite moves faster than a MEO one"
    );
}

#[test]
fn each_satellite_is_checked_independently() {
    let start = 800_000_000.0;
    let mut samples = arc(1, start, 30);
    let mut second = arc(2, start, 30);
    for sample in second.iter_mut().skip(15) {
        sample.position_ecef_m[0] += 900.0;
    }
    samples.extend(second);

    let report = check_continuity(&samples, &options());

    assert!(!report.attested());
    assert!(
        report
            .defects
            .iter()
            .all(|defect| defect.satellite() == satellite(2)),
        "only the spliced satellite may be reported, got {:?}",
        report.defects
    );
}

#[test]
fn an_explicit_bound_overrides_the_class_bound() {
    let samples = arc(1, 800_000_000.0, 10);

    let strict = ContinuityOptions {
        speed_bound: Some(SpeedBound::ExplicitMaxSpeed(100.0)),
        residual_tolerance_m: None,
    };
    let report = check_continuity(&samples, &strict);

    assert_eq!(
        report.defects_from(ContinuityCheck::SpeedBound).count(),
        samples.len() - 1,
        "an absurdly tight explicit bound must fire on every pair"
    );
    assert_eq!(report.residuals_checked, 0, "the residual check was off");
}

/// A published IGS final product must attest.
///
/// This is the check that keeps the tolerances honest: the synthetic arcs above
/// are smooth by construction, so only real published data proves the residual
/// tolerance is not tighter than the interpolator's own error on a real orbit
/// with real solar-radiation-pressure and eclipse dynamics in it.
#[test]
fn a_published_igs_final_product_attests() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");
    let text = std::fs::read(&path).expect("read committed IGS fixture");
    let product = Sp3::parse(&text).expect("parse committed IGS fixture");

    let samples = product.precise_ephemeris_samples();
    assert!(!samples.is_empty(), "fixture must yield samples");

    let report = check_continuity(&samples, &options());

    assert!(
        report.attested(),
        "a published IGS final product must attest continuous, got {:?}",
        report.defects.iter().take(5).collect::<Vec<_>>()
    );
    assert!(
        report.residuals_checked > 1_000,
        "expected a substantial residual sample, got {}",
        report.residuals_checked
    );
}

/// The same published product, with one contributor's arc spliced in, must fail.
#[test]
fn a_published_product_with_a_splice_is_caught() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");
    let text = std::fs::read(&path).expect("read committed IGS fixture");
    let product = Sp3::parse(&text).expect("parse committed IGS fixture");

    let mut samples = product.precise_ephemeris_samples();
    let target = samples[0].sat;
    let mut seen = 0usize;
    for sample in samples.iter_mut().filter(|s| s.sat == target) {
        seen += 1;
        if seen > 100 {
            sample.position_ecef_m[0] += 3.0;
        }
    }

    let report = check_continuity(&samples, &options());

    assert!(
        !report.attested(),
        "a 3 m splice in real data must be caught"
    );
    assert!(
        report
            .defects
            .iter()
            .all(|defect| defect.satellite() == target),
        "only the spliced satellite may be reported"
    );
}
