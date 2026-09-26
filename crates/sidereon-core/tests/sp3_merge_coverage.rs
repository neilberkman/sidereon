//! What a merge says about each satellite: which contributors a continuity
//! finding rests on, where positions and clocks are and are not, and which
//! epochs and cells it did not write.
//!
//! The fixture is synthetic: six GPS satellites on circular trajectories, a
//! 300-second grid on 2020-06-25. Source A carries epochs 0-71, source B epochs
//! 60-83, and B can be displaced bodily along X.

use sidereon_core::ephemeris::{
    check_continuity, merge, CellSelection, ClockOmission, ClockOmissionReason, ContinuityDefect,
    ContinuityOptions, DroppedEpochReason, EpochWindow, MergeCombine, MergeContinuityCellRole,
    MergeOptions, MergePrecedenceScope, OrbitClass, Sp3, Sp3CoverageGap, StencilExtent,
    WindowContinuityDecision,
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

/// Cell precedence fills A's end from B: positions run to epoch 83, but B's
/// clocks there cannot be put on A's datum, which is observable only where the
/// two overlap and is never extrapolated. The coverage and the omissions say
/// so; the union grid alone does not.
#[test]
fn coverage_separates_positions_from_clocks_and_names_each_omitted_clock() {
    let sources = [source(0, 72, 0.0), source(60, 24, 0.0)];
    let (merged, report) = merge(&sources, &precedence(MergePrecedenceScope::Cell)).expect("merge");

    assert_eq!(merged.epochs.len(), 84);
    assert!(report.omitted_epochs.is_empty());
    assert!(report.arc_withheld.is_empty());

    let coverage = merged.satellite_coverage();
    assert_eq!(coverage.grid.interval_s, Some(300.0));
    assert!(coverage.grid.agrees_with_header);
    assert_eq!(coverage.satellites.len(), 6);
    for satellite in &coverage.satellites {
        assert!(satellite.declared);
        assert_eq!(satellite.positions.epochs, 84);
        assert!(satellite.positions.is_complete());

        assert_eq!(satellite.clocks.epochs, 72);
        assert_eq!(satellite.clocks.spans.len(), 1);
        assert_eq!(satellite.clocks.spans[0].first_index, 0);
        assert_eq!(satellite.clocks.spans[0].last_index, 71);
        assert_eq!(satellite.clocks.last_epoch(), Some(merged.epochs[71]));
        assert_eq!(
            satellite.clocks.gaps,
            vec![Sp3CoverageGap {
                after_index: Some(71),
                before_index: None,
                missing_epochs: 12,
            }]
        );
    }

    assert_eq!(report.clock_omissions.len(), 6 * 12);
    for omission in &report.clock_omissions {
        assert_eq!(omission.reason, ClockOmissionReason::DatumNotObservable);
        assert_eq!(omission.source, 1);
        assert!(!omission.cell_has_clock);
        let index = merged
            .epochs
            .iter()
            .position(|epoch| *epoch == omission.epoch)
            .expect("the cell's position was written");
        assert!(index >= 72, "{index}");
    }
}

/// Satellite-arc precedence makes A the owner of every arc, so B's epochs past
/// A's end hold no cell at all. They are not written as blocks of missing
/// records: the product ends where the data ends, and the report lists the
/// omitted epochs and every withheld cell.
#[test]
fn satellite_arc_precedence_omits_and_reports_empty_epochs() {
    let sources = [source(0, 72, 0.0), source(60, 24, 0.0)];
    let (merged, report) =
        merge(&sources, &precedence(MergePrecedenceScope::SatelliteArc)).expect("merge");

    assert_eq!(merged.epochs.len(), 72);
    assert_eq!(merged.header.num_epochs, 72);
    assert_eq!(merged.epochs[..], sources[0].epochs[..]);
    assert_eq!(report.omitted_epochs[..], sources[1].epochs[12..]);
    assert_eq!(report.arc_withheld.len(), 6 * 12);
    for flag in &report.arc_withheld {
        assert_eq!(flag.sources, vec![1]);
        assert!(report.omitted_epochs.contains(&flag.epoch));
    }
    assert_eq!(report.clock_omissions.len(), 6 * 12);
    for omission in &report.clock_omissions {
        assert_eq!(omission.reason, ClockOmissionReason::DatumNotObservable);
        assert_eq!(omission.source, 1);
        assert!(report.omitted_epochs.contains(&omission.epoch));
    }

    for satellite in merged.satellite_coverage().satellites {
        assert!(satellite.positions.is_complete());
        assert!(satellite.clocks.is_complete());
        assert_eq!(satellite.positions.epochs, 72);
    }

    // The product writes and reads back with the same 72 epochs.
    let text = merged.to_sp3_string().expect("write merged product");
    let reread = Sp3::parse(text.as_bytes()).expect("reread merged product");
    assert_eq!(reread.epochs, merged.epochs);
}

/// A product that skips an epoch of its grid - a real product with a missing
/// epoch, or a merge that omitted an empty one - is still on that grid, and
/// merges on it rather than being refused for a non-uniform cadence.
#[test]
fn a_product_that_skips_an_epoch_merges_on_its_grid() {
    let skipping = product(&[0, 1, 2, 4, 5], 0.0, |_, _| Record::Full);
    let (merged, report) =
        merge(&[skipping], &precedence(MergePrecedenceScope::Cell)).expect("merge");
    assert_eq!(merged.header.epoch_interval_s, 300.0);
    assert_eq!(merged.epochs.len(), 5);
    assert!(report.omitted_epochs.is_empty());
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
    )
    .expect("valid continuity options");
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

/// Listing B first makes it the owner of every arc, so A's epochs before B's
/// start hold no cell. The product starts at 05:00, and its header states that
/// start exactly: the MJD fraction is 5/24 of a day rounded once to the
/// thirteen decimals its field holds, so the product writes and reads back.
#[test]
fn a_reversed_satellite_arc_merge_starting_at_five_writes_and_reads_back() {
    let sources = [source(60, 24, 0.0), source(0, 72, 0.0)];
    let (merged, report) =
        merge(&sources, &precedence(MergePrecedenceScope::SatelliteArc)).expect("merge");
    assert_eq!(merged.epochs[..], sources[0].epochs[..]);
    assert_eq!(report.omitted_epochs[..], sources[1].epochs[..60]);
    assert_eq!(merged.header.mjd, 59025);
    assert_eq!(merged.header.mjd_fraction, 0.208_333_333_333_3);
    assert_eq!(merged.header.seconds_of_week, 363_600.0);

    let text = merged.to_sp3_string().expect("write the merged product");
    assert!(
        text.starts_with("#cP2020  6 25  5  0  0.00000000"),
        "{text}"
    );
    assert!(text.contains("\n## 2111 363600.00000000   300.00000000 59025 0.2083333333333\n"));
    let reread = Sp3::parse(text.as_bytes()).expect("reread");
    assert_eq!(reread.epochs, merged.epochs);
    assert_eq!(reread.header.mjd_fraction, merged.header.mjd_fraction);
}

/// A single product starting at 05:00 merges and writes.
#[test]
fn a_merge_starting_at_five_writes() {
    let (merged, _) = merge(
        &[source(60, 24, 0.0)],
        &precedence(MergePrecedenceScope::Cell),
    )
    .expect("merge");
    let text = merged.to_sp3_string().expect("write the merged product");
    let reread = Sp3::parse(text.as_bytes()).expect("reread");
    assert_eq!(reread.epochs, merged.epochs);
}

/// When the merge accepts no cell at all it returns a product with no epochs,
/// its header on the first union-grid epoch. SP3 sets no minimum epoch count,
/// so the product writes, states that start on line 1, and reads back.
#[test]
fn a_merge_that_accepts_no_cell_writes_an_epochless_product() {
    let indices: Vec<usize> = (60..64).collect();
    let near = product(&indices, 0.0, |_, _| Record::NoClock);
    let far = product(&indices, 1_000.0, |_, _| Record::NoClock);
    let mut options = MergeOptions::default();
    options.min_agree = 2;
    let (merged, report) = merge(&[near, far], &options).expect("merge");

    assert!(merged.epochs.is_empty());
    assert_eq!(report.omitted_epochs.len(), 4);
    assert_eq!(report.quarantined.len(), 4 * 6);
    let start = sources_start(&indices);
    assert_eq!(merged.declared_start_j2000_s(), Some(start));
    assert_eq!(merged.header.mjd_fraction, 0.208_333_333_333_3);

    let text = merged.to_sp3_string().expect("write an epoch-less product");
    assert!(
        text.starts_with("#cP2020  6 25  5  0  0.00000000       0 "),
        "{text}"
    );
    let reread = Sp3::parse(text.as_bytes()).expect("reread");
    assert!(reread.epochs.is_empty());
    assert_eq!(reread.declared_start_j2000_s(), Some(start));
    assert_eq!(reread.to_sp3_string().expect("rewrite"), text);
}

/// Seconds since J2000 of the first of `indices`.
fn sources_start(indices: &[usize]) -> f64 {
    source(indices[0], 1, 0.0).epochs_j2000_seconds()[0]
}

/// Two products on the same 600 s cadence, one offset by 300 s from the other:
/// the default grid is the common divisor of their steps and offsets, so every
/// epoch of both is merged and none is dropped.
#[test]
fn the_default_grid_holds_every_epoch_of_phase_offset_inputs() {
    let even: Vec<usize> = (0..24).step_by(2).collect();
    let odd: Vec<usize> = (1..24).step_by(2).collect();
    let sources = [
        product(&even, 0.0, |_, _| Record::Full),
        product(&odd, 0.0, |_, _| Record::Full),
    ];
    let (merged, report) = merge(&sources, &precedence(MergePrecedenceScope::Cell)).expect("merge");
    assert_eq!(merged.header.epoch_interval_s, 300.0);
    assert_eq!(merged.epochs.len(), 24);
    assert!(report.dropped_input_epochs.is_empty());
}

/// An explicit coarser target drops the input epochs off its grid and reports
/// every one.
#[test]
fn an_explicit_target_reports_every_input_epoch_it_drops() {
    let mut options = precedence(MergePrecedenceScope::Cell);
    options.target_epoch_interval_s = Some(900.0);
    let (merged, report) = merge(&[source(0, 12, 0.0)], &options).expect("merge");
    assert_eq!(merged.epochs.len(), 4);
    assert_eq!(report.dropped_input_epochs.len(), 8);
    for dropped in &report.dropped_input_epochs {
        assert_eq!(dropped.source, 0);
        assert_ne!(dropped.epoch_index % 3, 0);
        assert_eq!(dropped.reason, DroppedEpochReason::OffTargetGrid);
    }
}

/// Two inputs half a second apart are two epochs, never one: the merge keys
/// epochs on the exact 10-nanosecond tick an epoch record states, and the
/// default grid holds both.
#[test]
fn inputs_half_a_second_apart_are_separate_epochs() {
    let whole = source(0, 1, 0.0);
    let text = whole
        .to_sp3_string()
        .expect("write")
        .replace("25  0  0  0.00000000", "25  0  0  0.50000000");
    let half = Sp3::parse(text.as_bytes()).expect("parse the half-second product");
    let (merged, report) =
        merge(&[whole, half], &precedence(MergePrecedenceScope::Cell)).expect("merge");
    assert_eq!(merged.epochs.len(), 2);
    assert_eq!(merged.header.epoch_interval_s, 0.5);
    assert!(report.dropped_input_epochs.is_empty());
    let written = merged.to_sp3_string().expect("write");
    assert!(written.contains("*  2020  6 25  0  0  0.00000000\n"));
    assert!(written.contains("*  2020  6 25  0  0  0.50000000\n"));
}

/// Each source clock the merge does not write is reported, including one left
/// out of a cell that got its clock from other sources: a third source whose
/// datum cannot be estimated (four common clocks, five needed) takes no part in
/// the mean.
#[test]
fn each_source_clock_left_out_of_a_clocked_cell_is_reported() {
    let indices: Vec<usize> = (0..12).collect();
    let a = product(&indices, 0.0, |_, _| Record::Full);
    let b = product(&indices, 0.0, |_, _| Record::Full);
    let c = product(&indices, 0.0, |prn, _| {
        if prn <= 4 {
            Record::Full
        } else {
            Record::NoClock
        }
    });
    let (merged, report) = merge(&[a, b, c], &MergeOptions::default()).expect("merge");
    assert_eq!(merged.epochs.len(), 12);
    assert_eq!(report.clock_omissions.len(), 12 * 4);
    for omission in &report.clock_omissions {
        assert_eq!(
            omission,
            &ClockOmission {
                epoch: omission.epoch,
                satellite: omission.satellite,
                source: 2,
                reason: ClockOmissionReason::DatumNotObservable,
                cell_has_clock: true,
            }
        );
        assert!(omission.satellite.prn <= 4);
    }
}

/// A gapped product is on a grid only when every step is a whole multiple of
/// its declared interval. A 300 s step under a declared 600 s interval is off
/// that grid, and the merge refuses the product rather than guess its cadence.
/// A uniform product is on its own step whatever its header says.
#[test]
fn a_gapped_product_merges_only_on_its_declared_grid() {
    let mut off_grid = product(&[0, 1, 2, 4, 5], 0.0, |_, _| Record::Full);
    off_grid.header.epoch_interval_s = 600.0;
    let error = merge(&[off_grid.clone()], &precedence(MergePrecedenceScope::Cell))
        .expect_err("a step shorter than the declared interval");
    assert!(error.to_string().contains("lie on no grid"), "{error}");
    let coverage = off_grid.satellite_coverage();
    assert_eq!(coverage.grid.interval_s, None);
    assert!(!coverage.grid.agrees_with_header);

    let mut wrong_header = product(&[0, 1, 2, 3], 0.0, |_, _| Record::Full);
    wrong_header.header.epoch_interval_s = 900.0;
    let (merged, _) = merge(
        &[wrong_header.clone()],
        &precedence(MergePrecedenceScope::Cell),
    )
    .expect("merge");
    assert_eq!(merged.header.epoch_interval_s, 300.0);
    let coverage = wrong_header.satellite_coverage();
    assert_eq!(coverage.grid.interval_s, Some(300.0));
    assert!(!coverage.grid.agrees_with_header);
    assert!(coverage.satellites[0].positions.is_complete());
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

/// Epochs held as integer nanoseconds from the J2000 origin merge on the same
/// axis as parsed ones: the merged product of a counted copy equals the merged
/// product of the original.
#[test]
fn nanosecond_epochs_merge_on_the_parsed_axis() {
    use sidereon_core::astro::time::model::InstantRepr;
    let parsed = source(0, 12, 0.0);
    let mut counted = parsed.clone();
    for (epoch, seconds) in counted.epochs.iter_mut().zip(parsed.epochs_j2000_seconds()) {
        epoch.repr = InstantRepr::Nanos(seconds as i128 * 1_000_000_000);
    }
    let options = precedence(MergePrecedenceScope::Cell);
    let (from_parsed, _) = merge(&[parsed], &options).expect("merge parsed");
    let (from_counted, _) = merge(&[counted], &options).expect("merge counted");
    assert_eq!(
        from_counted.to_sp3_string().expect("write"),
        from_parsed.to_sp3_string().expect("write")
    );
}
