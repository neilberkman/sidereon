#![cfg(sidereon_repo_tests)]

use sidereon_core::ephemeris::{
    align_clock_reference, merge, MergeCombine, MergeOptions, Sp3, Sp3FrameReconciliationOptions,
};

const COD: &str = include_str!("fixtures/sp3/COD0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3");
const GFZ: &str = include_str!("fixtures/sp3/GFZ0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3");

fn assert_round_trip(product: &Sp3, label: &str) {
    let text = product
        .to_sp3_string()
        .unwrap_or_else(|error| panic!("{label} must write: {error}"));
    let reparsed = Sp3::parse(text.as_bytes()).expect("written SP3 reparses");
    assert_eq!(reparsed.epochs, product.epochs, "{label} epochs");
    let bits = |value: Option<f64>| value.map(f64::to_bits);
    for epoch in 0..product.epoch_count() {
        let expected_states = product.states_at(epoch).expect("states available");
        let actual_states = reparsed
            .states_at(epoch)
            .expect("reparsed states available");
        assert_eq!(
            actual_states.keys().collect::<Vec<_>>(),
            expected_states.keys().collect::<Vec<_>>()
        );
        for (sat, expected) in expected_states {
            let actual = &actual_states[sat];
            for (field, expected, actual) in [
                ("x", expected.position.x_m, actual.position.x_m),
                ("y", expected.position.y_m, actual.position.y_m),
                ("z", expected.position.z_m, actual.position.z_m),
            ] {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{label} {sat} {field} epoch {epoch}"
                );
            }
            assert_eq!(
                bits(actual.clock_s),
                bits(expected.clock_s),
                "{label} {sat} clock epoch {epoch}"
            );
            assert_eq!(
                actual.flags, expected.flags,
                "{label} {sat} flags epoch {epoch}"
            );
            assert_eq!(
                actual.velocity.map(|v| [
                    v.vx_m_s.to_bits(),
                    v.vy_m_s.to_bits(),
                    v.vz_m_s.to_bits()
                ]),
                expected.velocity.map(|v| [
                    v.vx_m_s.to_bits(),
                    v.vy_m_s.to_bits(),
                    v.vz_m_s.to_bits()
                ]),
                "{label} {sat} velocity epoch {epoch}"
            );
            assert_eq!(
                bits(actual.clock_rate_s_s),
                bits(expected.clock_rate_s_s),
                "{label} {sat} rate epoch {epoch}"
            );
        }

        let expected_clocks = product
            .clock_records_at(epoch)
            .expect("clock records available");
        let actual_clocks = reparsed
            .clock_records_at(epoch)
            .expect("reparsed clock records available");
        assert_eq!(
            actual_clocks.keys().collect::<Vec<_>>(),
            expected_clocks.keys().collect::<Vec<_>>()
        );
        for (sat, expected) in expected_clocks {
            let actual = &actual_clocks[sat];
            assert_eq!(
                actual.clock_s.to_bits(),
                expected.clock_s.to_bits(),
                "{label} {sat} clock-only clock epoch {epoch}"
            );
            assert_eq!(
                actual.clock_us.to_bits(),
                expected.clock_us.to_bits(),
                "{label} {sat} raw clock epoch {epoch}"
            );
            assert_eq!(
                actual.flags, expected.flags,
                "{label} {sat} clock-only flags epoch {epoch}"
            );
            assert_eq!(
                actual.velocity.map(|v| [
                    v.vx_m_s.to_bits(),
                    v.vy_m_s.to_bits(),
                    v.vz_m_s.to_bits()
                ]),
                expected.velocity.map(|v| [
                    v.vx_m_s.to_bits(),
                    v.vy_m_s.to_bits(),
                    v.vz_m_s.to_bits()
                ]),
                "{label} {sat} clock-only velocity epoch {epoch}"
            );
            assert_eq!(
                bits(actual.clock_rate_s_s),
                bits(expected.clock_rate_s_s),
                "{label} {sat} clock-only rate epoch {epoch}"
            );
            assert_eq!(
                bits(actual.clock_rate_raw),
                bits(expected.clock_rate_raw),
                "{label} {sat} raw clock rate epoch {epoch}"
            );
        }
    }
}

fn assert_midpoint_interpolation_round_trip(product: &Sp3, label: &str) {
    let text = product
        .to_sp3_string()
        .unwrap_or_else(|error| panic!("{label} must write: {error}"));
    let reparsed = Sp3::parse(text.as_bytes()).expect("written SP3 reparses");
    let epochs = product.epochs_j2000_seconds();
    assert!(epochs.len() > 11, "{label} needs an interpolation window");
    let start = epochs.len() / 2;
    let mut compared = false;
    'queries: for index in start..epochs.len() - 1 {
        let query = epochs[index] + (epochs[index + 1] - epochs[index]) * 0.5;
        for satellite in product.header.satellites.iter().copied() {
            let Ok(before) = product.position_at_j2000_seconds(satellite, query) else {
                continue;
            };
            let after = reparsed
                .position_at_j2000_seconds(satellite, query)
                .unwrap_or_else(|error| panic!("{label} reparsed interpolation: {error}"));
            assert_eq!(
                before.position.as_array().map(f64::to_bits),
                after.position.as_array().map(f64::to_bits),
                "{label} midpoint position for {satellite} at {query}"
            );
            assert_eq!(
                before.clock_s.map(f64::to_bits),
                after.clock_s.map(f64::to_bits),
                "{label} midpoint clock for {satellite} at {query}"
            );
            compared = true;
            break 'queries;
        }
    }
    assert!(compared, "{label} has no interpolable midpoint satellite");
}

fn small_product(clocks_us: [f64; 3]) -> String {
    small_product_with_velocity(clocks_us, false)
}

fn small_product_with_velocity(clocks_us: [f64; 3], velocity: bool) -> String {
    let data_type = if velocity { 'V' } else { 'P' };
    let mut text = format!(
        "#c{data_type}2020  6 24  0  0  0.00000000       2 ORBIT IGS14 FIT  TST\n\
         ## 2111 259200.00000000   900.00000000 59024 0.0000000000000\n\
         +    3   G01G02G03\n\
         ++         0  0  0\n\
         %c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n"
    );
    for minute in [0, 15] {
        text.push_str(&format!("*  2020  6 24  0 {minute:2}  0.00000000\n"));
        for (index, sat) in ["G01", "G02", "G03"].into_iter().enumerate() {
            let (x, y, z) = if sat == "G03" {
                (0.0, 0.0, 0.0)
            } else {
                (15_600.0 + index as f64, 20_180.0, 21_700.0)
            };
            text.push_str(&format!(
                "P{sat}{x:14.6}{y:14.6}{z:14.6}{:14.6}\n",
                clocks_us[index]
            ));
            if velocity && sat != "G03" {
                text.push_str(&format!(
                    "V{sat}{:14.6}{:14.6}{:14.6}{:14.6}\n",
                    12.345678, -23.456789, 34.567891, -123.456789
                ));
            }
        }
    }
    text.push_str("EOF\n");
    text
}

#[test]
fn parsed_column_grid_inputs_merge_and_align_to_writable_products() {
    // These are literal, frozen parsed products. They exercise real parsed
    // column values and the sub-ULP arithmetic path with two contributors.
    let reference = Sp3::parse(COD.as_bytes()).expect("COD fixture parses");
    let other = Sp3::parse(GFZ.as_bytes()).expect("GFZ fixture parses");

    // Both real input products must themselves pass the strict writer before
    // they enter the parse -> align/merge -> write -> parse invariant.
    assert_round_trip(&reference, "literal COD input");
    assert_round_trip(&other, "literal GFZ input");

    let (merged, _) = merge(
        &[reference.clone(), other.clone()],
        &MergeOptions::default(),
    )
    .expect("two-center merge succeeds");
    assert_round_trip(&merged, "two-center mean merge");
    assert_midpoint_interpolation_round_trip(&merged, "two-center mean merge");

    let aligned = align_clock_reference(&reference, &other, 3);
    assert_round_trip(&aligned, "clock-aligned product");
    assert_midpoint_interpolation_round_trip(&aligned, "clock-aligned product");
}

#[test]
fn helmert_reconciled_grid_inputs_merge_to_a_writable_product() {
    let cod_text = COD.replace("IGc20", "IGS14");
    let gfz_text = GFZ.replace("IGc20", "IGS20");
    let cod = Sp3::parse(cod_text.as_bytes()).expect("IGS14 COD input parses");
    let gfz = Sp3::parse(gfz_text.as_bytes()).expect("IGS20 GFZ input parses");
    assert_round_trip(&cod, "IGS14 COD input");
    assert_round_trip(&gfz, "IGS20 GFZ input");

    let mut options = MergeOptions::default();
    options.frame_reconciliation = Sp3FrameReconciliationOptions::helmert();
    let (merged, report) = merge(&[cod, gfz], &options).expect("Helmert merge succeeds");

    assert_eq!(report.frame_reconciliations.len(), 1);
    assert_round_trip(&merged, "Helmert-reconciled grid merge");
    assert_midpoint_interpolation_round_trip(&merged, "Helmert-reconciled grid merge");
}

#[test]
fn deterministic_column_grid_matrix_preserves_round_trip_for_all_merge_rules() {
    // Sign, magnitude, signed zero, and half-grid means (138.9831105 us and
    // -598.4102345 us). G03 is clock-only, which exercises its distinct raw
    // clock record cache. The common datum offset is zero: the three satellite
    // differences are +1, -1, and 0 column units.
    let source_a = Sp3::parse(small_product([-598.410235, 138.983111, -0.0]).as_bytes())
        .expect("small source A parses");
    let source_b = Sp3::parse(small_product([-598.410234, 138.983110, 0.0]).as_bytes())
        .expect("small source B parses");
    assert_round_trip(&source_a, "small source A");
    assert_round_trip(&source_b, "small source B");

    let aligned = align_clock_reference(&source_a, &source_b, 3);
    assert_round_trip(&aligned, "small aligned product");

    let velocity_reference =
        Sp3::parse(small_product_with_velocity([-598.410235, 138.983111, -0.0], true).as_bytes())
            .expect("velocity reference parses");
    let velocity_other =
        Sp3::parse(small_product_with_velocity([-598.410234, 138.983110, 0.0], true).as_bytes())
            .expect("velocity other parses");
    assert_round_trip(&velocity_reference, "velocity reference");
    assert_round_trip(&velocity_other, "velocity other");
    let velocity_aligned = align_clock_reference(&velocity_reference, &velocity_other, 3);
    assert_round_trip(&velocity_aligned, "velocity-aligned product");

    for combine in [
        MergeCombine::Mean,
        MergeCombine::Median,
        MergeCombine::Precedence,
    ] {
        let mut options = MergeOptions::default();
        options.combine = combine;
        options.clock_min_common = 3;
        let (merged, _) = merge(&[source_a.clone(), source_b.clone()], &options)
            .unwrap_or_else(|error| panic!("{combine:?} merge: {error}"));
        assert_round_trip(&merged, &format!("{combine:?} small merge"));
        assert!(
            merged.clock_record(merged.header.satellites[2], 0).is_ok(),
            "{combine:?} retains G03 as a clock-only record"
        );
    }

    // A three-source case separates mean, median, and precedence while the
    // per-epoch median datum offset remains zero for every source.
    let source_c = Sp3::parse(small_product([-598.410231, 138.983111, -0.000004]).as_bytes())
        .expect("small source C parses");
    let mut outputs = Vec::new();
    for combine in [
        MergeCombine::Mean,
        MergeCombine::Median,
        MergeCombine::Precedence,
    ] {
        let mut options = MergeOptions::default();
        options.combine = combine;
        options.clock_min_common = 3;
        let (merged, _) = merge(
            &[source_a.clone(), source_b.clone(), source_c.clone()],
            &options,
        )
        .unwrap_or_else(|error| panic!("{combine:?} three-source merge: {error}"));
        assert_round_trip(&merged, &format!("{combine:?} three-source merge"));
        outputs.push(merged);
    }
    let g01 = outputs[0].header.satellites[0];
    let mean = outputs[0].state(g01, 0).expect("mean G01").clock_s.unwrap();
    let median = outputs[1]
        .state(g01, 0)
        .expect("median G01")
        .clock_s
        .unwrap();
    let precedence = outputs[2]
        .state(g01, 0)
        .expect("precedence G01")
        .clock_s
        .unwrap();
    assert_ne!(mean.to_bits(), median.to_bits());
    assert_ne!(median.to_bits(), precedence.to_bits());
}
