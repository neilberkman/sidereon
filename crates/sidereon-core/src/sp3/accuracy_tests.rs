use super::*;
use crate::astro::time::ExactEpoch;

fn accuracy_fixture(p_codes: &str, v_codes: &str) -> String {
    accuracy_fixture_with_bases(p_codes, v_codes, 1.25, 1.025)
}

fn accuracy_fixture_with_bases(
    p_codes: &str,
    v_codes: &str,
    position_velocity_base: f64,
    clock_rate_base: f64,
) -> String {
    format!(
        "\
#dV2022  1  2  3  4  5.00000000       1 ORBIT IGS20 FIT  TST
## 2191 270245.00000000   300.00000000 59581 0.1281597222222
+    1   G05  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f {position_velocity_base:10.7} {clock_rate_base:12.9}  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* ACCURACY TEST FIXTURE
*  2022  1  2  3  4  5.00000000
PG05  10000.000000  20000.000000  30000.000000    -50.000000 {p_codes}
VG05  10000.000000 -20000.000000  30000.000000      1.000000 {v_codes}
EOF
"
    )
}

fn sat() -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, 5).expect("valid satellite id")
}

fn replace_p_record(product: String, replacement: &str) -> String {
    let record = product
        .lines()
        .find(|line| line.starts_with("PG05"))
        .expect("fixture has a position record");
    product.replacen(record, replacement, 1)
}

fn p_record(x_km: f64, y_km: f64, z_km: f64, clock_us: f64, codes: &str) -> String {
    format!("PG05{x_km:14.6}{y_km:14.6}{z_km:14.6}{clock_us:14.6} {codes}")
}

#[test]
fn raw_and_effective_record_accuracy_are_both_publicly_inspectable() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, -1, 99, 0);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, -1, -2, 999);
    let product = Sp3::parse(accuracy_fixture(&p_codes, &v_codes).as_bytes()).unwrap();

    let raw = product.record_accuracy_codes(sat(), 0).unwrap();
    let raw_position = raw.p.unwrap();
    assert_eq!(raw_position.axis_exponents, [Some(0), Some(-1), Some(99)]);
    assert_eq!(raw_position.clock_exponent, Some(0));
    assert_eq!(raw_position.position_velocity_base, Some(1.25));
    assert_eq!(raw_position.clock_rate_base, Some(1.025));
    let raw_velocity = raw.v.unwrap();
    assert_eq!(raw_velocity.axis_exponents, [Some(0), Some(-1), Some(-2)]);
    assert_eq!(raw_velocity.clock_exponent, Some(999));

    let effective = product.record_accuracy(sat(), 0).unwrap();
    let position_accuracy = effective.p.unwrap();
    assert_eq!(
        position_accuracy.position_sigma_m[0],
        Sp3AccuracyValue::Known(1.0e-3)
    );
    assert_eq!(
        position_accuracy.position_sigma_m[1],
        Sp3AccuracyValue::Known(1.25_f64.powi(-1) * 1.0e-3)
    );
    assert_eq!(
        position_accuracy.position_sigma_m[2],
        Sp3AccuracyValue::TooLarge
    );
    assert_eq!(
        position_accuracy.clock_sigma_m,
        Sp3AccuracyValue::Known(crate::constants::C_M_S * 1.0e-12)
    );
    let velocity_accuracy = effective.v.unwrap();
    assert_eq!(
        velocity_accuracy.velocity_sigma_m_s[0],
        Sp3AccuracyValue::Known(1.0e-7)
    );
    assert_eq!(
        velocity_accuracy.velocity_sigma_m_s[1],
        Sp3AccuracyValue::Known(1.25_f64.powi(-1) * 1.0e-7)
    );
    assert_eq!(
        velocity_accuracy.velocity_sigma_m_s[2],
        Sp3AccuracyValue::Known(1.25_f64.powi(-2) * 1.0e-7)
    );
    assert_eq!(
        velocity_accuracy.clock_rate_sigma_m_s,
        Sp3AccuracyValue::TooLarge
    );
    assert_eq!(
        position_accuracy.position_variance_m2()[0],
        Sp3AccuracyValue::Known((1.0e-3_f64) * (1.0e-3_f64))
    );

    let emitted = product.to_sp3_string().unwrap();
    let reparsed = Sp3::parse(emitted.as_bytes()).unwrap();
    assert_eq!(reparsed.record_accuracy_codes(sat(), 0).unwrap(), raw);
}

#[test]
fn blank_codes_remain_distinct_from_zero_exponents() {
    let p_codes = "              ";
    let v_codes = "              ";
    let product = Sp3::parse(accuracy_fixture(p_codes, v_codes).as_bytes()).unwrap();
    let raw = product.record_accuracy_codes(sat(), 0).unwrap();
    let raw_position = raw.p.unwrap();
    assert_eq!(raw_position.axis_exponents, [None; 3]);
    assert_eq!(raw_position.clock_exponent, None);
    let effective = product.record_accuracy(sat(), 0).unwrap();
    assert_eq!(
        effective.p.unwrap().position_sigma_m,
        [Sp3AccuracyValue::Unknown; 3]
    );
    assert_eq!(
        effective.v.unwrap().clock_rate_sigma_m_s,
        Sp3AccuracyValue::Unknown
    );
    let reparsed = Sp3::parse(product.to_sp3_string().unwrap().as_bytes()).unwrap();
    assert_eq!(reparsed.record_accuracy_codes(sat(), 0).unwrap(), raw);
}

#[test]
fn variance_accessor_keeps_unavailable_states_and_marks_unrepresentable_squares() {
    assert_eq!(
        Sp3AccuracyValue::Unknown.variance(),
        Sp3AccuracyValue::Unknown
    );
    assert_eq!(
        Sp3AccuracyValue::Known(f64::MAX).variance(),
        Sp3AccuracyValue::Overflow
    );
    assert_eq!(
        Sp3AccuracyValue::Known(0.0).variance(),
        Sp3AccuracyValue::Known(0.0)
    );
    assert_eq!(
        Sp3AccuracyValue::Known(f64::from_bits(1)).variance(),
        Sp3AccuracyValue::Overflow
    );
}

/// The reference rounds the exact rational `99^155 * binary64(299792458 * 1e-12)`.
/// The gamma bound allows two 155-factor products, scale/reference rounding and
/// bound evaluation, using twice the binary64 unit roundoff conservatively.
#[test]
fn clock_accuracy_scaling_avoids_an_overflowing_unscaled_power() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, 0, 0, 155);
    let v_codes = "              ";
    let product =
        Sp3::parse(accuracy_fixture_with_bases(&p_codes, v_codes, 1.25, 99.0).as_bytes()).unwrap();
    let actual = product
        .record_accuracy(sat(), 0)
        .unwrap()
        .p
        .unwrap()
        .clock_sigma_m;
    let reference = f64::from_bits(0x7f6c_c54b_d11e_5c6b);
    let Sp3AccuracyValue::Known(actual) = actual else {
        panic!("finite scaled clock deviation should remain known");
    };
    assert!(actual.is_finite());
    let operation_roundoff = (2.0 * 155.0 + 4.0) * f64::EPSILON;
    let error_bound = reference * operation_roundoff / (1.0 - operation_roundoff);
    assert!((actual - reference).abs() <= error_bound);
}

#[test]
fn scaled_negative_power_can_be_finite_when_the_reciprocal_overflows() {
    let base = f64::from_bits(1_u64 << 50);
    let unit_scale = crate::constants::C_M_S * 1.0e-12;
    let expected = f64::from_bits(unit_scale.to_bits() + (1024_u64 << 52));
    assert!(expected.is_finite());
    assert_eq!(
        decode_accuracy(Some(-1), 999, Some(base), unit_scale),
        Sp3AccuracyValue::Known(expected)
    );
}

/// Reciprocal, product and independent division contribute three roundings;
/// the gamma bound uses twice the unit roundoff for the bound evaluation too.
#[test]
fn scaled_negative_power_agrees_with_independent_division_within_roundoff() {
    let base = 5.0e-309;
    let unit_scale = crate::constants::C_M_S * 1.0e-12;
    let reference = unit_scale / base;
    let Sp3AccuracyValue::Known(actual) = decode_accuracy(Some(-1), 999, Some(base), unit_scale)
    else {
        panic!("finite scaled reciprocal should remain known");
    };
    let operation_roundoff = 3.0 * f64::EPSILON;
    let error_bound = reference * operation_roundoff / (1.0 - operation_roundoff);
    assert!((actual - reference).abs() <= error_bound);
}

#[test]
fn normalized_power_scaling_preserves_subnormal_final_clock_sigma() {
    let base = 2.0_f64.powi(520);
    let unit_scale = 1.0e-7;
    let expected = unit_scale * f64::from_bits(1_u64 << 34);
    assert!(expected > 0.0);
    assert_eq!(scaled_integer_power(base, -2, unit_scale), expected);
    assert_eq!(
        decode_accuracy(Some(-2), 99, Some(base), unit_scale),
        Sp3AccuracyValue::Known(expected)
    );
}

#[test]
fn precise_source_accuracy_uses_the_state_epoch_not_the_selection_epoch() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, 0, 0, 0);
    let product = Sp3::parse(accuracy_fixture(&p_codes, &p_codes).as_bytes()).unwrap();
    let state_epoch = ExactEpoch::from_j2000_seconds(product.epochs_j2000_seconds()[0])
        .unwrap()
        .query();
    let selection_epoch = state_epoch.clone().checked_add_binary_seconds(5.0).unwrap();
    let expected = product.accuracy_variance_at_epoch_query(sat(), &state_epoch);
    let wrong_epoch = product.accuracy_variance_at_epoch_query(sat(), &selection_epoch);
    assert_ne!(expected, wrong_epoch);
    assert_eq!(
        crate::spp::EphemerisSource::ephemeris_variance_at_epoch_query(
            &product,
            sat(),
            &state_epoch,
            &selection_epoch,
        ),
        expected
    );
}

#[test]
fn merge_consensus_does_not_inherit_one_contributors_accuracy() {
    let left_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, 0, 0, 0);
    let right_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 1, 1, 1);
    let mut left = Sp3::parse(accuracy_fixture(&left_codes, &left_codes).as_bytes()).unwrap();
    let mut right = Sp3::parse(accuracy_fixture(&right_codes, &right_codes).as_bytes()).unwrap();
    left.header.satellite_accuracy_codes[0] = 21;
    right.header.satellite_accuracy_codes[0] = 37;

    let (merged, _) = merge(&[left.clone(), right.clone()], &MergeOptions::default()).unwrap();
    assert_eq!(merged.header.satellite_accuracy_codes, vec![0]);

    let consensus_accuracy = merged.record_accuracy_codes(sat(), 0).unwrap().p.unwrap();
    assert_eq!(consensus_accuracy.axis_exponents, [None; 3]);
    assert_eq!(consensus_accuracy.clock_exponent, Some(0));
    assert_eq!(consensus_accuracy.position_velocity_base, None);
    assert_eq!(consensus_accuracy.clock_rate_base, Some(1.025));

    let options = MergeOptions {
        combine: MergeCombine::Precedence,
        ..MergeOptions::default()
    };
    let (precedence, _) = merge(&[left.clone(), right], &options).unwrap();
    assert_eq!(
        precedence
            .record_accuracy_codes(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .axis_exponents,
        [Some(0), Some(0), Some(0)]
    );

    let (single_source, _) = merge(&[left], &MergeOptions::default()).unwrap();
    assert_eq!(single_source.header.satellite_accuracy_codes, vec![21]);
    assert_eq!(
        single_source
            .record_accuracy_codes(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .axis_exponents,
        [Some(0), Some(0), Some(0)]
    );
}

#[test]
fn merge_takes_position_and_clock_accuracy_from_their_independent_contributors() {
    let position_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 2, 3, 4, 5);
    let clock_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 8, 9, 10, 11);
    let position_only = replace_p_record(
        accuracy_fixture(&position_codes, "              "),
        &p_record(
            10_000.0,
            20_000.0,
            30_000.0,
            999_999.999999,
            &position_codes,
        ),
    );
    let clock_only = replace_p_record(
        accuracy_fixture(&clock_codes, "              "),
        &p_record(0.0, 0.0, 0.0, -25.0, &clock_codes),
    );
    let mut position_source = Sp3::parse(position_only.as_bytes()).unwrap();
    let mut clock_source = Sp3::parse(clock_only.as_bytes()).unwrap();
    position_source.header.satellite_accuracy_codes[0] = 17;
    clock_source.header.satellite_accuracy_codes[0] = 9;
    let options = MergeOptions {
        clock_min_common: 1,
        ..MergeOptions::default()
    };

    let (merged, _) = merge(&[clock_source.clone(), position_source.clone()], &options).unwrap();
    assert_eq!(merged.header.satellite_accuracy_codes, vec![17]);
    let accuracy = merged.record_accuracy_codes(sat(), 0).unwrap().p.unwrap();
    assert_eq!(accuracy.axis_exponents, [Some(2), Some(3), Some(4)]);
    assert_eq!(accuracy.clock_exponent, Some(11));

    let (non_base_contributor, _) = merge(&[clock_source, position_source], &options).unwrap();
    assert_eq!(
        non_base_contributor.header.satellite_accuracy_codes,
        vec![17]
    );
}

#[test]
fn consensus_clock_does_not_replace_single_source_position_accuracy() {
    let position_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 2, 3, 4, 5);
    let other_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 8, 9, 10, 11);
    let position_and_clock = accuracy_fixture(&position_codes, "              ");
    let matching_clock_only = replace_p_record(
        accuracy_fixture(&other_codes, "              "),
        &p_record(0.0, 0.0, 0.0, -50.0, &other_codes),
    );
    let position_source = Sp3::parse(position_and_clock.as_bytes()).unwrap();
    let clock_source = Sp3::parse(matching_clock_only.as_bytes()).unwrap();
    let options = MergeOptions {
        clock_min_common: 1,
        ..MergeOptions::default()
    };

    let (merged, _) = merge(&[position_source, clock_source], &options).unwrap();
    let accuracy = merged.record_accuracy_codes(sat(), 0).unwrap().p.unwrap();
    assert_eq!(accuracy.axis_exponents, [Some(2), Some(3), Some(4)]);
    assert_eq!(accuracy.clock_exponent, None);
}

#[test]
fn changed_base_refuses_nontrivial_accuracy_reencoding() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, -1, 99, 0);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, 0, 0, 0);
    let mut product = Sp3::parse(accuracy_fixture(&p_codes, &v_codes).as_bytes()).unwrap();
    product.header.pos_vel_base = Some(1.5);

    assert!(matches!(
        product.to_sp3_string(),
        Err(Sp3WriteError::AccuracyNotRepresentable {
            component: "position y",
            exponent: Some(-1),
            ..
        })
    ));
}

#[test]
fn zero_exponent_survives_changed_valid_base() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, 99, 99, 0);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 0, 99, 99, 0);
    let mut product = Sp3::parse(accuracy_fixture(&p_codes, &v_codes).as_bytes()).unwrap();
    product.header.pos_vel_base = Some(1.5);

    let emitted = product.to_sp3_string().unwrap();
    let reparsed = Sp3::parse(emitted.as_bytes()).unwrap();
    assert_eq!(
        reparsed
            .record_accuracy_codes(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .axis_exponents,
        [Some(0), Some(99), Some(99)]
    );
    assert_eq!(
        reparsed
            .record_accuracy(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .position_sigma_m[0],
        Sp3AccuracyValue::Known(1.0e-3)
    );
}

#[test]
fn writer_exactly_reencodes_rationally_equal_cross_base_powers() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 99, 99, 0);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 99, 99, 0);
    let mut product =
        Sp3::parse(accuracy_fixture_with_bases(&p_codes, &v_codes, 4.0, 1.025).as_bytes()).unwrap();
    product.header.pos_vel_base = Some(2.0);

    let emitted = product.to_sp3_string().unwrap();
    let reparsed = Sp3::parse(emitted.as_bytes()).unwrap();
    assert_eq!(
        reparsed
            .record_accuracy_codes(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .axis_exponents[0],
        Some(2)
    );
    assert_eq!(
        reparsed
            .record_accuracy(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .position_sigma_m[0],
        Sp3AccuracyValue::Known(4.0e-3)
    );
}

#[test]
fn writer_refuses_rounded_but_not_exact_cross_base_power_matches() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 99, 99, 0);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 99, 99, 0);
    let mut product =
        Sp3::parse(accuracy_fixture_with_bases(&p_codes, &v_codes, 4.0, 1.025).as_bytes()).unwrap();
    product.header.pos_vel_base = Some(3.0);

    assert!(matches!(
        product.to_sp3_string(),
        Err(Sp3WriteError::AccuracyNotRepresentable {
            component: "position x",
            exponent: Some(1),
            ..
        })
    ));
}

#[test]
fn writer_exactly_reencodes_reciprocal_powers() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 99, 99, 0);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 1, 99, 99, 0);
    let mut product =
        Sp3::parse(accuracy_fixture_with_bases(&p_codes, &v_codes, 0.5, 1.025).as_bytes()).unwrap();
    product.header.pos_vel_base = Some(2.0);

    let reparsed = Sp3::parse(product.to_sp3_string().unwrap().as_bytes()).unwrap();
    assert_eq!(
        reparsed
            .record_accuracy_codes(sat(), 0)
            .unwrap()
            .p
            .unwrap()
            .axis_exponents[0],
        Some(-1)
    );
}

#[test]
fn writer_places_accuracy_codes_and_flags_in_their_sp3_columns() {
    let p_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 2, -1, 99, 5);
    let v_codes = format!("{:>2} {:>2} {:>2} {:>3}  ", 3, -2, 98, -9);
    let mut product = Sp3::parse(accuracy_fixture(&p_codes, &v_codes).as_bytes()).unwrap();
    product.states[0].get_mut(&sat()).unwrap().flags = Sp3Flags {
        clock_event: true,
        clock_predicted: true,
        maneuver: true,
        orbit_predicted: true,
    };

    let emitted = product.to_sp3_string().unwrap();
    let p_line = emitted
        .lines()
        .find(|line| line.starts_with("PG05"))
        .unwrap();
    let v_line = emitted
        .lines()
        .find(|line| line.starts_with("VG05"))
        .unwrap();

    for (line, axes, clock) in [
        (p_line, [" 2", "-1", "99"], "  5"),
        (v_line, [" 3", "-2", "98"], " -9"),
    ] {
        assert_eq!(&line[61..63], axes[0]);
        assert_eq!(&line[64..66], axes[1]);
        assert_eq!(&line[67..69], axes[2]);
        assert_eq!(&line[70..73], clock);
    }
    assert_eq!(&p_line[74..80], "EP  MP");
}
