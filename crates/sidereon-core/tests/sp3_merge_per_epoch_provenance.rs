//! Per-epoch merge provenance: which contributor supplied each cell, where
//! selection changed, and what each contributor covered.
//!
//! The SP3 sources here are built as text in the test so every expectation is a
//! property of an input this file states outright.

use sidereon_core::ephemeris::{
    merge, CellSelection, MergeCombine, MergeOptions, MergePrecedenceScope, OutlierRejectOptions,
    ProvenanceMode, Sp3, TransitionReason,
};

/// A two-epoch, single-satellite SP3. `positions_km` is `[epoch0, epoch1]`, each
/// the satellite's X coordinate; Y and Z are fixed so a caller controls exactly
/// one axis of disagreement.
fn source(positions_km: [f64; 2]) -> Sp3 {
    let body = format!(
        "#cP2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST\n\
         ## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n\
         +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* TEST SP3-c FIXTURE\n\
         *  2020  6 25  0  0  0.00000000\n\
         PG01 {:13.6} -20000.000000   5000.000000    100.000000\n\
         *  2020  6 25  0 15  0.00000000\n\
         PG01 {:13.6} -20000.000000   5000.000000    100.000000\n\
         EOF\n",
        positions_km[0], positions_km[1]
    );
    Sp3::parse(body.as_bytes()).expect("parse test sp3")
}

/// A source carrying only the second epoch, so a satellite's first epoch has one
/// contributor and its second has two.
fn late_source(position_km: f64) -> Sp3 {
    let body = format!(
        "#cP2020  6 25  0 15  0.00000000       1 ORBIT IGS14 FIT  TST\n\
         ## 2111 432900.00000000   900.00000000 59025 0.0000000000000\n\
         +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* TEST SP3-c FIXTURE\n\
         *  2020  6 25  0 15  0.00000000\n\
         PG01 {position_km:13.6} -20000.000000   5000.000000    100.000000\n\
         EOF\n"
    );
    Sp3::parse(body.as_bytes()).expect("parse test sp3")
}

fn precedence_options(mode: Option<ProvenanceMode>) -> MergeOptions {
    let mut options = MergeOptions::default();
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = MergePrecedenceScope::Cell;
    options.min_agree = 1;
    options.provenance = mode;
    options
}

#[test]
fn provenance_is_absent_unless_requested() {
    // The absence must be first-class: a caller has to be able to tell "not
    // requested" from "one contributor supplied everything".
    let (_merged, report) =
        merge(&[source([15_000.0, 15_100.0])], &precedence_options(None)).expect("merge");

    assert!(
        report.provenance.is_none(),
        "provenance must be None when it was never requested"
    );
}

#[test]
fn a_single_contributor_merge_records_it_for_every_epoch_with_no_mid_arc_transition() {
    let (_merged, report) = merge(
        &[source([15_000.0, 15_100.0])],
        &precedence_options(Some(ProvenanceMode::Full)),
    )
    .expect("merge");

    let provenance = report.provenance.expect("provenance requested");
    assert_eq!(provenance.cells.len(), 2, "two accepted cells");
    for cell in &provenance.cells {
        assert_eq!(
            cell.position.selected_source(),
            Some(0),
            "the only contributor supplies every cell"
        );
    }

    // The arc's opening entry is a transition from `None`; there must be no
    // further change.
    assert_eq!(provenance.transitions.len(), 1, "only the opening entry");
    assert_eq!(provenance.transitions[0].from_source, None);
    assert_eq!(provenance.transitions[0].to_source, Some(0));

    let coverage = &provenance.coverage[0];
    assert_eq!(coverage.cells_contributed, 2);
    assert_eq!(coverage.cells_selected, 2);
    assert_eq!(coverage.cells_absent, 0);
    assert!(coverage.first_epoch.is_some() && coverage.last_epoch.is_some());
}

#[test]
fn a_forced_precedence_switch_records_one_transition_naming_both_sides() {
    // Source 0 carries only the first epoch; source 1 carries both. Under cell
    // precedence the supplier must change at the second epoch.
    // Source 0 carries its first epoch only.
    let early = Sp3::parse(
        "#cP2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST\n\
         ## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n\
         +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* TEST SP3-c FIXTURE\n\
         *  2020  6 25  0  0  0.00000000\n\
         PG01  15000.000000 -20000.000000   5000.000000    100.000000\n\
         EOF\n"
            .as_bytes(),
    )
    .expect("parse trimmed source");

    let late = late_source(15_100.0);

    let (_merged, report) = merge(
        &[early, late],
        &precedence_options(Some(ProvenanceMode::Full)),
    )
    .expect("merge");

    let provenance = report.provenance.expect("provenance requested");
    assert_eq!(provenance.cells.len(), 2);
    assert_eq!(provenance.cells[0].position.selected_source(), Some(0));
    assert_eq!(provenance.cells[1].position.selected_source(), Some(1));

    // The opening entry plus exactly one real change.
    let changes: Vec<_> = provenance
        .transitions
        .iter()
        .filter(|transition| transition.from_source.is_some())
        .collect();
    assert_eq!(changes.len(), 1, "exactly one mid-arc transition");
    assert_eq!(changes[0].from_source, Some(0));
    assert_eq!(changes[0].to_source, Some(1));
    assert_eq!(
        changes[0].reason,
        TransitionReason::SoleAvailability,
        "source 0 stopped carrying the cell; that is availability, not preference"
    );

    // Coverage: source 0 supplied one of the two accepted cells and was absent
    // for the other.
    assert_eq!(provenance.coverage[0].cells_contributed, 1);
    assert_eq!(provenance.coverage[0].cells_absent, 1);
    assert_eq!(provenance.coverage[1].cells_contributed, 1);
    assert_eq!(provenance.coverage[1].cells_absent, 1);
}

#[test]
fn outlier_rejection_is_recorded_as_its_own_reason() {
    // Three sources: 0 and 1 agree at the second epoch, 2 agrees with everyone
    // at the first. Source 0 goes wild at the second epoch and is rejected from
    // the consensus by the guard, so the selection moves off it for a reason
    // that is not availability.
    let wild = source([15_000.0, 25_000.0]);
    let steady_a = source([15_000.0, 15_100.0]);
    let steady_b = source([15_000.0, 15_100.0]);

    let mut options = MergeOptions::default();
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = MergePrecedenceScope::Cell;
    options.min_agree = 2;
    options.position_tolerance_m = 1.0;
    options.outlier_reject = Some(OutlierRejectOptions {
        position_tolerance_m: 1.0,
        clock_tolerance_s: 1.0e-6,
    });
    options.provenance = Some(ProvenanceMode::Full);

    let (_merged, report) = merge(&[wild, steady_a, steady_b], &options).expect("merge");

    let provenance = report.provenance.expect("provenance requested");
    assert_eq!(provenance.cells[0].position.selected_source(), Some(0));
    assert_eq!(
        provenance.cells[1].position.selected_source(),
        Some(1),
        "the rejected preferred source must not supply the cell"
    );

    let changes: Vec<_> = provenance
        .transitions
        .iter()
        .filter(|transition| transition.from_source.is_some())
        .collect();
    assert_eq!(changes.len(), 1);
    assert_eq!(
        changes[0].reason,
        TransitionReason::OutlierRejection,
        "source 0 was still present but rejected; that is not availability or preference"
    );
    assert!(
        !report.position_outliers.is_empty(),
        "the rejection is also in the existing flag list"
    );
}

#[test]
fn a_combined_cell_names_no_single_supplier() {
    // Under the default mean rule the written value is a combination, so "which
    // contributor supplied this cell" has no answer and the record must say so
    // rather than nominate one.
    let a = source([15_000.0, 15_100.0]);
    let b = source([15_000.0, 15_100.0]);

    let mut options = MergeOptions::default();
    options.provenance = Some(ProvenanceMode::Full);
    let (_merged, report) = merge(&[a, b], &options).expect("merge");

    let provenance = report.provenance.expect("provenance requested");
    for cell in &provenance.cells {
        assert!(
            matches!(cell.position, CellSelection::Combined { .. }),
            "a mean-combined cell must be recorded as combined"
        );
        assert_eq!(
            cell.position.selected_source(),
            None,
            "a combined value has no single supplier"
        );
        assert_eq!(cell.position.members(), vec![0, 1], "both members recorded");
    }
    for coverage in &provenance.coverage {
        assert_eq!(
            coverage.cells_selected, 0,
            "no cell has a single supplier under a combining rule"
        );
        assert_eq!(coverage.cells_contributed, 2);
    }
}

#[test]
fn summary_and_full_modes_agree_on_every_transition_they_both_describe() {
    let early = source([15_000.0, 15_100.0]);
    let late = late_source(15_100.0);

    let (_full_merged, full_report) = merge(
        &[early.clone(), late.clone()],
        &precedence_options(Some(ProvenanceMode::Full)),
    )
    .expect("merge");
    let (_summary_merged, summary_report) = merge(
        &[early, late],
        &precedence_options(Some(ProvenanceMode::Summary)),
    )
    .expect("merge");

    let full = full_report.provenance.expect("full provenance");
    let summary = summary_report.provenance.expect("summary provenance");

    assert_eq!(
        full.transitions, summary.transitions,
        "the two modes must describe identical transitions"
    );
    assert_eq!(
        full.coverage, summary.coverage,
        "the two modes must describe identical coverage"
    );
    assert!(
        summary.cells.is_empty(),
        "summary mode carries no per-cell entries"
    );
    assert!(!full.cells.is_empty(), "full mode carries per-cell entries");
}

#[test]
fn the_merged_product_is_byte_identical_whether_or_not_provenance_is_enabled() {
    // The test that keeps the feature honest: provenance is an observation of
    // the merge, never an input to it.
    let sources = || vec![source([15_000.0, 15_100.0]), late_source(15_100.5)];

    let (without, report_without) =
        merge(&sources(), &precedence_options(None)).expect("merge without provenance");
    let (with, report_with) = merge(&sources(), &precedence_options(Some(ProvenanceMode::Full)))
        .expect("merge with provenance");

    assert_eq!(
        without.to_sp3_string(),
        with.to_sp3_string(),
        "enabling provenance must not change one byte of the merged product"
    );

    // And the rest of the audit trail is untouched too.
    assert_eq!(report_without.agreement, report_with.agreement);
    assert_eq!(report_without.single_source, report_with.single_source);
    assert_eq!(report_without.quarantined, report_with.quarantined);
    assert_eq!(
        report_without.position_outliers,
        report_with.position_outliers
    );
}
