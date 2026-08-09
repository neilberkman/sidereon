//! Continuity as a merge post-condition: a splice must be reported *and*
//! attributed to the contributors on both sides of it.

use sidereon_core::ephemeris::{
    merge, ContinuityOptions, MergeCombine, MergeOptions, MergePrecedenceScope, OrbitClass, Sp3,
};

/// A multi-epoch single-satellite SP3 whose positions follow a smooth
/// earth-fixed arc, offset bodily by `offset_m` along X. Two sources built with
/// different offsets are each internally continuous but mutually inconsistent,
/// which is exactly the splice a precedence switch can create.
fn arc_source(start_epoch_index: usize, count: usize, offset_m: f64) -> Sp3 {
    const A_KM: f64 = 26_560.0;
    const STEP_S: f64 = 900.0;

    let mut records = String::new();
    for index in 0..count {
        let epoch_index = start_epoch_index + index;
        let seconds = epoch_index as f64 * STEP_S;
        let theta = 3.0 / A_KM * seconds;
        let x = A_KM * theta.cos() + offset_m / 1000.0;
        let y = A_KM * theta.sin() * 0.573_576;
        let z = A_KM * theta.sin() * 0.819_152;
        let minutes = (epoch_index * 15) % 60;
        let hours = (epoch_index * 15) / 60;
        records.push_str(&format!(
            "*  2020  6 25 {hours:2}{minutes:3}  0.00000000\n\
             PG01 {x:13.6} {y:13.6} {z:13.6}    100.000000\n"
        ));
    }

    let start_seconds = start_epoch_index as f64 * STEP_S;
    let hours = (start_epoch_index * 15) / 60;
    let minutes = (start_epoch_index * 15) % 60;
    let body = format!(
        "#cP2020  6 25 {hours:2}{minutes:3}  0.00000000     {count:3} ORBIT IGS14 FIT  TST\n\
         ## 2111 {:14.8}   900.00000000 59025 0.0000000000000\n\
         +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* TEST SP3-c FIXTURE\n\
         {records}EOF\n",
        432_000.0 + start_seconds
    );
    Sp3::parse(body.as_bytes()).expect("parse arc source")
}

fn options() -> MergeOptions {
    MergeOptions {
        combine: MergeCombine::Precedence,
        precedence_scope: MergePrecedenceScope::Cell,
        min_agree: 1,
        position_tolerance_m: 10_000.0,
        verify_continuity: Some(ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss)),
        ..MergeOptions::default()
    }
}

#[test]
fn a_switch_between_consistent_contributors_validates() {
    // Source 0 covers the first half, source 1 the second, and they lie on the
    // same arc. Precedence switches mid-arc, but nothing is spliced.
    let first = arc_source(0, 12, 0.0);
    let second = arc_source(12, 12, 0.0);

    let (merged, report) = merge(&[first, second], &options()).expect("merge");

    // Not a vacuous pass: the product must actually span both contributors.
    assert_eq!(
        merged.epochs_j2000_seconds().len(),
        24,
        "both halves merged"
    );

    let continuity = report.continuity.expect("post-condition requested");
    assert!(
        continuity.report.residuals_checked >= 20,
        "the residual check must have actually run, got {}",
        continuity.report.residuals_checked
    );
    assert!(
        continuity.attested(),
        "a switch between consistent contributors is not a defect, got {:?}",
        continuity.report.defects
    );
    assert!(continuity.violations.is_empty());
}

#[test]
fn a_switch_between_inconsistent_contributors_reports_and_names_both_sides() {
    // Same coverage split, but source 1's arc sits 400 m away. The merged
    // product is spliced at the handover.
    let first = arc_source(0, 12, 0.0);
    let second = arc_source(12, 12, 400.0);

    let (_merged, report) = merge(&[first, second], &options()).expect("merge");

    let continuity = report.continuity.expect("post-condition requested");
    assert!(
        !continuity.attested(),
        "a 400 m splice at the handover must be reported"
    );

    let splices: Vec<_> = continuity.splices().collect();
    assert!(
        !splices.is_empty(),
        "the violation must be attributed across a contributor change, got {:?}",
        continuity.violations
    );

    let splice = splices[0];
    assert_eq!(
        splice.from_sources,
        vec![0],
        "the earlier side came from source 0"
    );
    assert_eq!(
        splice.to_sources,
        vec![1],
        "the later side came from source 1"
    );
    assert!(splice.crosses_contributors);
}

#[test]
fn the_post_condition_reports_without_refusing_the_merge() {
    // The merge must still return the product: a caller may legitimately want
    // the product together with its defects.
    let first = arc_source(0, 12, 0.0);
    let second = arc_source(12, 12, 400.0);

    let (merged, report) = merge(&[first, second], &options()).expect("merge must not fail");

    assert!(!report.continuity.expect("requested").attested());
    assert!(
        !merged.to_sp3_string().is_empty(),
        "the product is returned alongside its defects"
    );
}

#[test]
fn the_post_condition_is_absent_and_the_product_unchanged_when_not_requested() {
    let sources = || vec![arc_source(0, 12, 0.0), arc_source(12, 12, 400.0)];

    let mut without_options = options();
    without_options.verify_continuity = None;

    let (with, with_report) = merge(&sources(), &options()).expect("merge");
    let (without, without_report) = merge(&sources(), &without_options).expect("merge");

    assert!(without_report.continuity.is_none());
    assert!(with_report.continuity.is_some());
    assert_eq!(
        with.to_sp3_string(),
        without.to_sp3_string(),
        "verifying continuity must not change one byte of the merged product"
    );
}
