use sidereon_core::ephemeris::{
    merge, ClockOmissionReason, MergeCombine, MergeOptions, MergePrecedenceScope,
    OutlierRejectOptions, Sp3, Sp3AccuracyValue, Sp3WriteError,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const A: &str = include_str!("../fixtures/A.sp3");
const B: &str = include_str!("../fixtures/B.sp3");
const B_SIGMA_X_M: f64 = (1_220_703_125.0 / 67_108_864.0) * 1.0e-3;

fn satellite(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn merge_options() -> MergeOptions {
    let mut options = MergeOptions::default();
    options.position_tolerance_m = 5.0;
    options.clock_tolerance_s = 50.0e-9;
    options.min_agree = 1;
    options.clock_min_common = 1;
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = MergePrecedenceScope::Cell;
    options.outlier_reject = Some(OutlierRejectOptions::new(5.0, 50.0e-9));
    options.target_epoch_interval_s = Some(300.0);
    options.systems = Some([GnssSystem::Gps].into_iter().collect());
    options
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

    let a_codes = core_a
        .record_accuracy_codes(satellite(1), 0)
        .unwrap()
        .p
        .unwrap();
    assert_eq!(a_codes.axis_exponents, [None; 3]);
    assert_eq!(a_codes.clock_exponent, None);
    let a_accuracy = core_a.record_accuracy(satellite(1), 0).unwrap().p.unwrap();
    assert_eq!(a_accuracy.position_sigma_m, [Sp3AccuracyValue::Unknown; 3]);

    let a_strict = Sp3::parse(core_a.to_sp3_string().unwrap().as_bytes()).unwrap();
    let (a_rounded_text, a_report) = core_a.to_sp3_string_with_rounding().unwrap();
    let a_rounded = Sp3::parse(a_rounded_text.as_bytes()).unwrap();
    assert_eq!(a_report.position_x_m, 0.0);
    assert_eq!(a_strict.epoch_count(), 2);
    assert_eq!(a_rounded.epoch_count(), 2);
    assert_eq!(
        a_strict.record_accuracy_codes(satellite(1), 0).unwrap(),
        core_a.record_accuracy_codes(satellite(1), 0).unwrap()
    );
    assert_eq!(
        a_rounded.record_accuracy_codes(satellite(1), 0).unwrap(),
        core_a.record_accuracy_codes(satellite(1), 0).unwrap()
    );
    assert_eq!(
        a_strict
            .record_accuracy(satellite(1), 0)
            .unwrap()
            .p
            .unwrap()
            .position_sigma_m,
        [Sp3AccuracyValue::Unknown; 3]
    );
    assert_eq!(
        a_rounded
            .record_accuracy(satellite(1), 0)
            .unwrap()
            .p
            .unwrap()
            .position_sigma_m,
        [Sp3AccuracyValue::Unknown; 3]
    );

    let b_strict = Sp3::parse(core_b.to_sp3_string().unwrap().as_bytes()).unwrap();
    let (b_rounded_text, b_report) = core_b.to_sp3_string_with_rounding().unwrap();
    let b_rounded = Sp3::parse(b_rounded_text.as_bytes()).unwrap();
    assert_eq!(b_report.position_x_m, 0.0);
    assert_eq!(b_strict.epoch_count(), 2);
    assert_eq!(b_rounded.epoch_count(), 2);
    assert_known_x_sigma(&b_strict, 12, B_SIGMA_X_M);
    assert_known_x_sigma(&b_rounded, 12, B_SIGMA_X_M);
}

#[test]
fn published_305_merge_order_changes_header_base_and_writer_result() {
    let a = Sp3::parse(A.as_bytes()).unwrap();
    let b = Sp3::parse(B.as_bytes()).unwrap();

    let (a_then_b, report) = merge(&[a.clone(), b.clone()], &merge_options()).unwrap();
    assert_eq!(a_then_b.epoch_count(), 2);
    assert_eq!(a_then_b.header.pos_vel_base, Some(0.0));
    assert_eq!(report.clock_omissions.len(), 2);
    assert!(report.clock_omissions.iter().all(|omission| {
        omission.source == 1 && omission.reason == ClockOmissionReason::DatumNotObservable
    }));
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
    let strict_reread = Sp3::parse(b_then_a.to_sp3_string().unwrap().as_bytes()).unwrap();
    let (text, report) = b_then_a.to_sp3_string_with_rounding().unwrap();
    assert_eq!(report.position_x_m, 0.0);
    let rounded_reread = Sp3::parse(text.as_bytes()).unwrap();
    assert_eq!(strict_reread.epoch_count(), 2);
    assert_eq!(rounded_reread.epoch_count(), 2);
    assert_known_x_sigma(&strict_reread, 12, B_SIGMA_X_M);
    assert_known_x_sigma(&rounded_reread, 12, B_SIGMA_X_M);
    let codes = rounded_reread
        .record_accuracy_codes(satellite(12), 0)
        .unwrap()
        .p
        .unwrap();
    assert_eq!(codes.axis_exponents[0], Some(13));
    assert_eq!(codes.position_velocity_base, Some(1.25));
}
