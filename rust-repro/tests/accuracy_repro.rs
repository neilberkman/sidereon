use sidereon_core::sp3::{
    merge, MergeCombine, MergeOptions, MergePrecedenceScope, OutlierRejectOptions, Sp3,
    Sp3AccuracyValue, Sp3WriteError,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const A: &str = include_str!("../fixtures/A.sp3");
const B: &str = include_str!("../fixtures/B.sp3");
const B_SIGMA_X_M: f64 = 1.25_f64.powi(13) * 1.0e-3;

fn satellite(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn merge_options() -> MergeOptions {
    MergeOptions {
        position_tolerance_m: 5.0,
        clock_tolerance_s: 50.0e-9,
        min_agree: 1,
        clock_min_common: 1,
        combine: MergeCombine::Precedence,
        precedence_scope: MergePrecedenceScope::Cell,
        outlier_reject: Some(OutlierRejectOptions::new(5.0, 50.0e-9)),
        target_epoch_interval_s: Some(300.0),
        systems: Some([GnssSystem::Gps].into_iter().collect()),
        ..MergeOptions::default()
    }
}

fn assert_known_x_sigma(product: &Sp3, prn: u8, expected: f64) {
    let accuracy = product.record_accuracy(satellite(prn), 0).unwrap();
    assert_eq!(
        accuracy.p.unwrap().position_sigma_m[0],
        Sp3AccuracyValue::Known(expected)
    );
}

#[test]
fn published_305_facade_and_core_round_trip_each_synthetic_source() {
    let facade_a = sidereon::load_sp3(A.as_bytes()).unwrap();
    let facade_b = sidereon::load_sp3(B.as_bytes()).unwrap();
    assert_eq!(facade_a.epoch_count(), 2);
    assert_eq!(facade_b.epoch_count(), 2);
    assert!(facade_a.to_sp3_string().is_ok());
    assert!(facade_b.to_sp3_string().is_ok());
    assert!(facade_a.to_sp3_string_with_rounding().is_ok());
    assert!(facade_b.to_sp3_string_with_rounding().is_ok());

    let core_a = Sp3::parse(A.as_bytes()).unwrap();
    let core_b = Sp3::parse(B.as_bytes()).unwrap();
    assert_eq!(core_a.epoch_count(), 2);
    assert_eq!(core_b.epoch_count(), 2);
    assert_known_x_sigma(&core_b, 12, B_SIGMA_X_M);
    let p_codes = core_b
        .record_accuracy_codes(satellite(12), 0)
        .unwrap()
        .p
        .unwrap();
    assert_eq!(p_codes.axis_exponents[0], Some(13));
    assert_eq!(p_codes.position_velocity_base, Some(1.25));

    let (strict_text, _) = core_b.to_sp3_string_with_rounding().unwrap();
    let strict_reread = Sp3::parse(strict_text.as_bytes()).unwrap();
    assert_known_x_sigma(&strict_reread, 12, B_SIGMA_X_M);
}

#[test]
fn published_305_merge_order_changes_header_base_and_writer_result() {
    let a = Sp3::parse(A.as_bytes()).unwrap();
    let b = Sp3::parse(B.as_bytes()).unwrap();

    let (a_then_b, _) = merge(&[a.clone(), b.clone()], &merge_options()).unwrap();
    assert_eq!(a_then_b.epoch_count(), 2);
    assert_eq!(a_then_b.header.pos_vel_base, Some(0.0));
    assert_known_x_sigma(&a_then_b, 12, B_SIGMA_X_M);
    let merged_codes = a_then_b
        .record_accuracy_codes(satellite(12), 0)
        .unwrap()
        .p
        .unwrap();
    assert_eq!(merged_codes.axis_exponents[0], Some(13));
    assert_eq!(merged_codes.position_velocity_base, Some(1.25));
    assert!(matches!(
        a_then_b.to_sp3_string(),
        Err(Sp3WriteError::AccuracyNotRepresentable {
            component: "position x",
            exponent: Some(13),
            ..
        })
    ));
    assert!(matches!(
        a_then_b.to_sp3_string_with_rounding(),
        Err(Sp3WriteError::AccuracyNotRepresentable {
            component: "position x",
            exponent: Some(13),
            ..
        })
    ));

    let (b_then_a, _) = merge(&[b, a], &merge_options()).unwrap();
    assert_eq!(b_then_a.header.pos_vel_base, Some(1.25));
    let (text, report) = b_then_a.to_sp3_string_with_rounding().unwrap();
    assert_eq!(report.position_x_m, 0.0);
    let reread = Sp3::parse(text.as_bytes()).unwrap();
    assert_known_x_sigma(&reread, 12, B_SIGMA_X_M);
    let codes = reread
        .record_accuracy_codes(satellite(12), 0)
        .unwrap()
        .p
        .unwrap();
    assert_eq!(codes.axis_exponents[0], Some(13));
    assert_eq!(codes.position_velocity_base, Some(1.25));
}
