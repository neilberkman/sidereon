//! Window-scoped continuity over a real daily SP3 product and a writer-derived
//! consecutive day.
//!
//! Primary fixture provenance:
//! `COD0MGXFIN_20201770000_01D_05M_ORB.SP3`, public CODE MGEX final product,
//! 2020-06-25, retained from the repository's established IGS fixture set.
//! Canonical archive: `https://cddis.nasa.gov/archive/gnss/products/mgex/2111/`.
//! Committed bytes: SHA-256
//! `54b70fa009a840ecf8cec25fbd4d749c9aaef7c95bdf463484e115f74d802215`.
//! Verified with `shasum -a 256` on 2026-08-21.
//!
//! No consecutive daily pair is committed. Only the seam-injection test makes
//! a second day: it clones the real product, advances every public epoch and
//! line-2 day field by 86,400 seconds, serializes through [`Sp3::to_sp3_string`],
//! and reparses it. No derived file is treated as an external continuity oracle.

use sidereon_core::astro::time::civil::split_julian_date_add_seconds;
use sidereon_core::astro::time::model::{InstantRepr, JulianDateSplit};
use sidereon_core::ephemeris::{
    check_continuity, merge, ContinuityDefect, ContinuityOptions, ContinuityReport, EpochWindow,
    MergeCombine, MergeContinuityReport, MergeContinuityViolation, MergeOptions,
    MergePrecedenceScope, OrbitClass, Sp3, StencilExtent, WindowContinuityDecision,
};

const DAY_S: f64 = 86_400.0;

fn real_daily_product() -> Sp3 {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");
    let bytes = std::fs::read(path).expect("read committed CODE final product");
    Sp3::parse(&bytes).expect("parse committed CODE final product")
}

fn writer_derived_next_day(product: &Sp3) -> Sp3 {
    let mut shifted = product.clone();
    for epoch in &mut shifted.epochs {
        match &mut epoch.repr {
            InstantRepr::JulianDate(split) => {
                let (jd_whole, fraction) =
                    split_julian_date_add_seconds(split.jd_whole, split.fraction, DAY_S);
                *split = JulianDateSplit::new(jd_whole, fraction).expect("shifted split epoch");
            }
            InstantRepr::Nanos(nanos) => *nanos += 86_400_000_000_000_i128,
        }
    }
    shifted.header.seconds_of_week += DAY_S;
    if shifted.header.seconds_of_week >= 604_800.0 {
        shifted.header.seconds_of_week -= 604_800.0;
        shifted.header.gnss_week += 1;
    }
    shifted.header.mjd += 1;

    Sp3::parse(shifted.to_sp3_string().as_bytes()).expect("reparse writer-derived next day")
}

fn precedence_options() -> MergeOptions {
    let mut options = MergeOptions::default();
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = MergePrecedenceScope::Cell;
    options.min_agree = 1;
    options
}

#[test]
fn real_product_window_query_preserves_global_attestation() {
    let product = real_daily_product();
    let epochs = product.epochs_j2000_seconds();
    let report = check_continuity(
        &product.precise_ephemeris_samples(),
        &ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss),
    );
    assert!(report.attested(), "real product must attest globally");

    let window = EpochWindow::new(epochs[72], epochs[216]).expect("daytime window");
    let stencil = StencilExtent::for_sp3(&product).expect("product stencil");
    let verdict = report.verdict_for_window(window, stencil);

    assert_eq!(verdict.decision, WindowContinuityDecision::Accept);
    assert!(verdict.influencing_defects.is_empty());
    assert!(verdict.all_defects.is_empty());
}

#[test]
fn merged_daily_window_verdict_flips_when_the_stencil_reaches_the_seam() {
    let first = real_daily_product();
    let second = writer_derived_next_day(&first);
    let first_epochs = first.epochs_j2000_seconds();
    let seam = *first_epochs.last().expect("first-day terminal epoch");
    let sat = first.satellites()[0];

    let (merged, mut merge_report) =
        merge(&[first, second], &precedence_options()).expect("merge consecutive daily products");
    assert!(
        merged.epochs_j2000_seconds().first().copied().unwrap() < seam
            && merged.epochs_j2000_seconds().last().copied().unwrap() > seam,
        "merged product must cover both sides of the seam"
    );

    let defect = ContinuityDefect::SpeedBound {
        sat,
        from_j2000_s: seam,
        to_j2000_s: seam + merged.header.epoch_interval_s,
        interval_s: merged.header.epoch_interval_s,
        displacement_m: 3_000_000.0,
        implied_speed_m_s: 10_000.0,
        bound_m_s: 6_000.0,
    };
    merge_report.continuity = Some(MergeContinuityReport {
        report: ContinuityReport {
            defects: vec![defect.clone()],
            ..ContinuityReport::default()
        },
        violations: vec![MergeContinuityViolation {
            defect,
            from_sources: vec![0],
            to_sources: vec![1],
            crosses_contributors: true,
        }],
    });

    let stencil = StencilExtent::for_sp3(&merged).expect("merged-product stencil");
    assert_eq!(merged.header.epoch_interval_s, 300.0);
    // Eleven spacings, not five: ten for the one-sided stencil at a run edge
    // plus one for a query served just outside a run. Every satellite in this
    // product is sampled at the 300 s header interval.
    assert_eq!(stencil.before_s(), 3_300.0);
    assert_eq!(stencil.after_s(), 3_300.0);

    let inside_one_day =
        EpochWindow::new(seam - 18.0 * 3_600.0, seam - 6.0 * 3_600.0).expect("inside-day window");
    let inside_verdict = merge_report
        .continuity_verdict_for_window(inside_one_day, stencil)
        .expect("injected continuity report");
    assert_eq!(inside_verdict.decision, WindowContinuityDecision::Accept);
    assert!(inside_verdict.influencing_defects.is_empty());
    assert_eq!(inside_verdict.all_defects.len(), 1);
    assert_eq!(inside_verdict.all_splices.len(), 1);

    let straddles = EpochWindow::new(seam - 600.0, seam + 600.0).expect("straddling window");
    let straddling_verdict = merge_report
        .continuity_verdict_for_window(straddles, stencil)
        .expect("injected continuity report");
    assert_eq!(
        straddling_verdict.decision,
        WindowContinuityDecision::Refuse
    );
    assert_eq!(straddling_verdict.influencing_defects.len(), 1);
    assert_eq!(straddling_verdict.influencing_splices.len(), 1);

    let reaches_seam = EpochWindow::new(seam - 7_200.0, seam - stencil.after_s())
        .expect("half-width boundary window");
    assert_eq!(
        merge_report
            .continuity_verdict_for_window(reaches_seam, stencil)
            .expect("injected continuity report")
            .decision,
        WindowContinuityDecision::Refuse,
        "an inclusive window whose stencil reaches the seam must refuse"
    );

    let misses_seam = EpochWindow::new(seam - 7_200.0, seam - stencil.after_s() - 0.001)
        .expect("outside-stencil window");
    assert_eq!(
        merge_report
            .continuity_verdict_for_window(misses_seam, stencil)
            .expect("injected continuity report")
            .decision,
        WindowContinuityDecision::Accept,
        "moving one millisecond beyond the derived stencil must accept"
    );
}

#[test]
fn window_and_stencil_constructors_reject_invalid_axes() {
    assert!(EpochWindow::new(f64::NAN, 1.0).is_err());
    assert!(EpochWindow::new(2.0, 1.0).is_err());

    let mut product = real_daily_product();
    product.header.epoch_interval_s = 0.0;
    assert!(StencilExtent::for_sp3(&product).is_err());
}

/// A defect that only the run-edge stencil can reach must still refuse.
///
/// The interpolator centers its 11 nodes on the query only in the interior of
/// a run. For a query in the last interval it keeps the node count and slides
/// the window inward, so the earliest selected node sits up to ten intervals
/// back rather than five, and a query served one spacing past a run reaches
/// eleven. A reach derived from the centered stencil reported 1,500 s here, and
/// a defect 1,800 s before the window fell outside it while sitting inside the
/// nodes the interpolator actually selects. That is the unsafe direction: the
/// verdict said Accept for a window the defect moves.
///
/// Demonstrated on this product by perturbing a node 1,650 s before a query in
/// the final interval and watching the interpolated position move by 3.36 m.
#[test]
fn a_defect_only_the_edge_stencil_reaches_still_refuses() {
    let product = real_daily_product();
    let epochs = product.epochs_j2000_seconds();
    let interval_s = product.header.epoch_interval_s;
    assert_eq!(interval_s, 300.0);
    let n = epochs.len();
    let sat = product.satellites()[0];

    // Queries anywhere in the product's final interval.
    let final_interval = EpochWindow::new(epochs[n - 2], epochs[n - 1]).expect("final interval");
    let stencil = StencilExtent::for_sp3(&product).expect("product stencil");

    // Six intervals before the window start: outside the centered five, inside
    // the eleven nodes the interpolator selects at the run end (indices
    // n-11 .. n-1 for any query past epochs[n-2]).
    let reached_by_edge_stencil = epochs[n - 2] - 6.0 * interval_s;
    let report = ContinuityReport {
        defects: vec![ContinuityDefect::DuplicateEpoch {
            sat,
            epoch_j2000_s: reached_by_edge_stencil,
            occurrences: 2,
        }],
        ..ContinuityReport::default()
    };
    let verdict = report.verdict_for_window(final_interval, stencil);
    assert_eq!(
        verdict.decision,
        WindowContinuityDecision::Refuse,
        "a node the edge stencil selects must count as influencing"
    );
    assert_eq!(verdict.influencing_defects.len(), 1);

    // Twelve spacings before the window start is beyond even the widest
    // stencil (eleven: ten for the edge slide, one for a query served just
    // outside a run), so the corrected reach does not refuse everything.
    let beyond_any_stencil = epochs[n - 2] - 12.0 * interval_s;
    let report = ContinuityReport {
        defects: vec![ContinuityDefect::DuplicateEpoch {
            sat,
            epoch_j2000_s: beyond_any_stencil,
            occurrences: 2,
        }],
        ..ContinuityReport::default()
    };
    let verdict = report.verdict_for_window(final_interval, stencil);
    assert_eq!(verdict.decision, WindowContinuityDecision::Accept);
    assert!(verdict.influencing_defects.is_empty());
    assert_eq!(
        verdict.all_defects.len(),
        1,
        "the finding is still reported"
    );
}
