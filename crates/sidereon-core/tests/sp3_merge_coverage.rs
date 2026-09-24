//! What a merge says about each satellite: which contributors a continuity
//! finding rests on, which windows it refuses, and where each satellite's
//! positions and clocks are and are not.
//!
//! The fixture is synthetic: six GPS satellites on circular trajectories, a
//! 300-second grid on 2020-06-25. Source A carries epochs 0-71, source B epochs
//! 60-83, and B can be displaced bodily along X.

use sidereon_core::ephemeris::{
    check_continuity, merge, CellSelection, ContinuityDefect, ContinuityOptions, EpochWindow,
    MergeCombine, MergeContinuityCellRole, MergeOptions, MergePrecedenceScope, OrbitClass, Sp3,
    Sp3CoverageGap, StencilExtent, WindowContinuityDecision,
};

const STEP_S: usize = 300;

/// Epoch record text for `index` steps after 2020-06-25 00:00.
fn epoch_fields(index: usize) -> String {
    format!(
        "2020  6 25 {:2}{:3}  0.00000000",
        index / 12,
        (index % 12) * 5
    )
}

/// What one satellite carries at one epoch of a built product.
#[derive(Clone, Copy, PartialEq)]
enum Record {
    Full,
    NoClock,
    ClockOnly,
    Absent,
}

/// A product over `indices` (steps of 300 s from 00:00), declaring 300 s, with
/// `offset_m` added to every X coordinate and `record` choosing what each
/// satellite carries at each epoch.
fn product(indices: &[usize], offset_m: f64, record: impl Fn(u8, usize) -> Record) -> Sp3 {
    let first = indices[0];
    let mut text = format!(
        "#cP{}     {:3} ORBIT IGS14 FIT  TST\n\
         ## 2111 {:14.8}   300.00000000 59025 {:.13}\n\
         +    6   G01G02G03G04G05G06  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* SYNTHETIC SP3 COVERAGE FIXTURE\n",
        epoch_fields(first),
        indices.len(),
        345_600.0 + (first * STEP_S) as f64,
        (first * STEP_S) as f64 / 86_400.0
    );
    for &index in indices {
        let seconds = (index * STEP_S) as f64;
        text.push_str(&format!("*  {}\n", epoch_fields(index)));
        for prn in 1..=6u8 {
            let angle = seconds * core::f64::consts::TAU / 43_200.0 + f64::from(prn) * 0.3;
            let x = 26_560.0 * libm::cos(angle) + offset_m / 1000.0;
            let y = 26_560.0 * libm::sin(angle) * 0.6;
            let z = 26_560.0 * libm::sin(angle) * 0.8;
            let clock = 10.0 + f64::from(prn);
            match record(prn, index) {
                Record::Full => {
                    text.push_str(&format!("PG{prn:02}{x:14.6}{y:14.6}{z:14.6}{clock:14.6}\n"))
                }
                Record::NoClock => text.push_str(&format!(
                    "PG{prn:02}{x:14.6}{y:14.6}{z:14.6}{:14.6}\n",
                    999_999.999_999
                )),
                Record::ClockOnly => text.push_str(&format!(
                    "PG{prn:02}{:14.6}{:14.6}{:14.6}{clock:14.6}\n",
                    0.0, 0.0, 0.0
                )),
                Record::Absent => {}
            }
        }
    }
    text.push_str("EOF\n");
    Sp3::parse(text.as_bytes()).expect("synthetic SP3")
}

fn source(first: usize, count: usize, offset_m: f64) -> Sp3 {
    let indices: Vec<usize> = (first..first + count).collect();
    product(&indices, offset_m, |_, _| Record::Full)
}

fn precedence(scope: MergePrecedenceScope) -> MergeOptions {
    let mut options = MergeOptions::default();
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = scope;
    options.min_agree = 1;
    options.position_tolerance_m = 5.0;
    options
}

/// A 0.8 m displacement of B inside the 5 m agreement tolerance: cell
/// precedence writes A through epoch 71 and B after it. The hold-out replay
/// keeps alternate nodes, so near the end of B's run the window for a sample
/// between two B records slides back over A's records. That violation rests on
/// A as well as B, and the report says so.
#[test]
fn a_hold_out_residual_is_attributed_to_every_node_its_prediction_used() {
    let sources = [source(0, 72, 0.0), source(60, 24, 0.8)];
    let mut options = precedence(MergePrecedenceScope::Cell);
    options.verify_continuity = Some(ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss));

    let (merged, report) = merge(&sources, &options).expect("merge");
    let start = merged.epochs_j2000_seconds()[0];
    let continuity = report.continuity.expect("verification requested");
    assert!(!continuity.attested(), "the 0.8 m handover is reported");

    let inside_b: Vec<_> = continuity
        .violations
        .iter()
        .filter(|violation| violation.from_sources == vec![1] && violation.to_sources == vec![1])
        .collect();
    assert!(
        !inside_b.is_empty(),
        "a violation bracketed by two B records, got {:?}",
        continuity.violations
    );
    for violation in inside_b {
        let ContinuityDefect::HoldOutResidual {
            epoch_j2000_s,
            node_epochs_j2000_s,
            ..
        } = &violation.defect
        else {
            panic!("only the hold-out check sees a 0.8 m splice: {violation:?}");
        };
        // The retained series is every other node, and the window holds eleven
        // of them.
        assert_eq!(node_epochs_j2000_s.len(), 11);
        assert!(node_epochs_j2000_s
            .windows(2)
            .all(|pair| pair[1] - pair[0] == 2.0 * STEP_S as f64));
        assert!(
            node_epochs_j2000_s[0] <= start + 71.0 * STEP_S as f64,
            "the window reaches back over A's records"
        );

        assert!(violation.crosses_contributors, "{violation:?}");
        assert_eq!(violation.sources, vec![0, 1]);
        assert_eq!(violation.cells.len(), 12);
        let held_out: Vec<_> = violation
            .cells
            .iter()
            .filter(|cell| cell.role == MergeContinuityCellRole::HeldOut)
            .collect();
        assert_eq!(held_out.len(), 1);
        assert_eq!(held_out[0].epoch_j2000_s, *epoch_j2000_s);
        assert_eq!(
            held_out[0].selection,
            Some(CellSelection::SingleSource { source: 1 })
        );
        // A's nodes were written from A, with B in their agreement cluster:
        // the selected source stays distinct from the other members.
        assert!(violation.cells.iter().any(|cell| {
            cell.role == MergeContinuityCellRole::InterpolationNode
                && cell.selection
                    == Some(CellSelection::Precedence {
                        source: 0,
                        members: vec![0, 1],
                    })
        }));
    }
}

/// With no displacement the same merge has no finding at all.
#[test]
fn the_undisplaced_merge_attests() {
    let sources = [source(0, 72, 0.0), source(60, 24, 0.0)];
    let mut options = precedence(MergePrecedenceScope::Cell);
    options.verify_continuity = Some(ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss));

    let (_, report) = merge(&sources, &options).expect("merge");
    let continuity = report.continuity.expect("verification requested");
    assert!(continuity.attested(), "{:?}", continuity.report.defects);
    assert!(continuity.violations.is_empty());
}

/// The window verdict on the 0.8 m merge refuses exactly the windows whose
/// interpolations use the handover between A and B or a held-out record at
/// fault. A window whose nodes are all A's records - every single-epoch window
/// through epoch 67, and the window 51-66 - interpolates records no finding
/// implicates and is accepted, although its conservative stencil bound
/// reaches the handover.
#[test]
fn merge_window_verdicts_refuse_only_windows_that_use_the_handover_or_a_record_at_fault() {
    let sources = [source(0, 72, 0.0), source(60, 24, 0.8)];
    let mut options = precedence(MergePrecedenceScope::Cell);
    options.verify_continuity = Some(ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss));
    let (merged, report) = merge(&sources, &options).expect("merge");
    let epochs = merged.epochs_j2000_seconds();
    assert_eq!(epochs.len(), 84);

    for (index, &epoch) in epochs.iter().enumerate() {
        let window = EpochWindow::new(epoch, epoch).expect("window");
        let verdict = report
            .continuity_verdict_for_window(window)
            .expect("verification requested");
        let expected = if index >= 68 {
            WindowContinuityDecision::Refuse
        } else {
            WindowContinuityDecision::Accept
        };
        assert_eq!(verdict.decision, expected, "single-epoch window at {index}");
    }

    let early = EpochWindow::new(epochs[51], epochs[66]).expect("window");
    assert_eq!(
        report
            .continuity_verdict_for_window(early)
            .expect("verification requested")
            .decision,
        WindowContinuityDecision::Accept
    );
    let reaching = EpochWindow::new(epochs[51], epochs[68]).expect("window");
    let verdict = report
        .continuity_verdict_for_window(reaching)
        .expect("verification requested");
    assert_eq!(verdict.decision, WindowContinuityDecision::Refuse);
    assert!(!verdict.influencing_splices.is_empty());

    // The same findings checked on the product alone place each on its
    // offending pair and bound a window's reach by the stencil extent: the
    // record pairs at 80-82 enter single-epoch windows from epoch 69, eleven
    // spacings back.
    let plain = check_continuity(
        &merged.precise_ephemeris_samples(),
        &ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss),
    );
    let stencil = StencilExtent::for_sp3(&merged).expect("stencil");
    assert_eq!(stencil.before_s(), 3_300.0);
    for (index, &epoch) in epochs.iter().enumerate() {
        let window = EpochWindow::new(epoch, epoch).expect("window");
        let expected = if index >= 69 {
            WindowContinuityDecision::Refuse
        } else {
            WindowContinuityDecision::Accept
        };
        assert_eq!(
            plain.verdict_for_window(window, stencil).decision,
            expected,
            "plain single-epoch window at {index}"
        );
    }
}

/// Coverage of a single product: a satellite missing from some epochs, one
/// carrying positions without clocks, one carrying only clocks, and a step in
/// the product's own epoch list longer than its declared interval.
#[test]
fn coverage_of_a_product_reports_spans_and_gaps_per_channel() {
    // Epoch 5 is not in the product at all.
    let indices = [0, 1, 2, 3, 4, 6, 7, 8];
    let sp3 = product(&indices, 0.0, |prn, index| match (prn, index) {
        (1, _) => Record::Full,
        (2, 2 | 3) => Record::Absent,
        (2, _) => Record::Full,
        (3, 7 | 8) => Record::NoClock,
        (3, _) => Record::Full,
        (4, 0 | 1) => Record::ClockOnly,
        (4, _) => Record::Full,
        (_, _) => Record::Absent,
    });
    let coverage = sp3.satellite_coverage();
    assert_eq!(coverage.grid.interval_s, Some(300.0));
    assert!(coverage.grid.agrees_with_header);
    assert!(coverage.grid.out_of_order.is_empty());
    assert!(coverage.grid.unplaced.is_empty());
    assert_eq!(coverage.satellites.len(), 6);
    let of = |prn: u8| {
        coverage
            .satellites
            .iter()
            .find(|satellite| satellite.satellite.prn == prn)
            .expect("listed")
    };

    // G01 carries everything, but the product itself skips epoch 5: the span
    // ends there and the gap between holds no product epoch.
    let g01 = of(1);
    assert_eq!(g01.positions.epochs, 8);
    assert_eq!(g01.positions.spans.len(), 2);
    assert_eq!(
        (
            g01.positions.spans[0].first_index,
            g01.positions.spans[0].last_index
        ),
        (0, 4)
    );
    assert_eq!(g01.positions.spans[0].epochs(), 5);
    assert_eq!(
        g01.positions.gaps,
        vec![Sp3CoverageGap {
            after_index: Some(4),
            before_index: Some(5),
            missing_epochs: 0,
        }]
    );
    assert!(!g01.positions.is_complete());
    assert_eq!(g01.clocks, g01.positions);

    // G02 is absent at product epochs 2 and 3.
    let g02 = of(2);
    assert_eq!(g02.positions.epochs, 6);
    assert_eq!(
        g02.positions.gaps,
        vec![
            Sp3CoverageGap {
                after_index: Some(1),
                before_index: Some(4),
                missing_epochs: 2,
            },
            Sp3CoverageGap {
                after_index: Some(4),
                before_index: Some(5),
                missing_epochs: 0,
            },
        ]
    );

    // G03 has positions throughout but no clock at the last two epochs.
    let g03 = of(3);
    assert_eq!(g03.positions.epochs, 8);
    assert_eq!(g03.clocks.epochs, 6);
    assert_eq!(
        g03.clocks.gaps.last(),
        Some(&Sp3CoverageGap {
            after_index: Some(5),
            before_index: None,
            missing_epochs: 2,
        })
    );

    // G04 carries clock-only records at the first two epochs.
    let g04 = of(4);
    assert_eq!(g04.positions.epochs, 6);
    assert_eq!(g04.clocks.epochs, 8);
    assert_eq!(
        g04.positions.gaps.first(),
        Some(&Sp3CoverageGap {
            after_index: None,
            before_index: Some(2),
            missing_epochs: 2,
        })
    );

    // G05 and G06 are declared and carry nothing.
    for prn in [5, 6] {
        let empty = of(prn);
        assert!(empty.declared);
        assert_eq!(empty.positions.epochs, 0);
        assert!(empty.positions.spans.is_empty());
        assert_eq!(
            empty.positions.gaps,
            vec![Sp3CoverageGap {
                after_index: None,
                before_index: None,
                missing_epochs: 8,
            }]
        );
        assert_eq!(empty.clocks, empty.positions);
    }
}

/// Coverage with a zero declared interval: a gapped product then lies on no
/// grid, and steps end no span; a uniform one lies on its own step.
#[test]
fn coverage_with_a_zero_interval() {
    let mut gapped = product(&[0, 1, 2, 4, 5], 0.0, |_, _| Record::Full);
    gapped.header.epoch_interval_s = 0.0;
    let coverage = gapped.satellite_coverage();
    assert_eq!(coverage.grid.interval_s, None);
    assert!(!coverage.grid.agrees_with_header);
    assert_eq!(coverage.satellites[0].positions.spans.len(), 1);
    assert!(coverage.satellites[0].positions.is_complete());

    let mut uniform = product(&[0, 1, 2, 3], 0.0, |_, _| Record::Full);
    uniform.header.epoch_interval_s = 0.0;
    let coverage = uniform.satellite_coverage();
    assert_eq!(coverage.grid.interval_s, Some(300.0));
    assert!(!coverage.grid.agrees_with_header);
}

/// Epochs out of time order are reported, lie on no grid, and end a span.
#[test]
fn coverage_reports_epochs_out_of_order() {
    let sp3 = product(&[0, 2, 1, 3], 0.0, |_, _| Record::Full);
    let coverage = sp3.satellite_coverage();
    assert_eq!(coverage.grid.out_of_order, vec![2]);
    assert_eq!(coverage.grid.interval_s, None);
    let g01 = &coverage.satellites[0].positions;
    assert_eq!(g01.spans.len(), 2);
    assert_eq!(
        g01.gaps,
        vec![Sp3CoverageGap {
            after_index: Some(1),
            before_index: Some(2),
            missing_epochs: 0,
        }]
    );
}
