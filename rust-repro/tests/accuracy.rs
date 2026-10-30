use sidereon::ephemeris::{Sp3AccuracyRoundingPolicy, Sp3DetailedRoundingReport};
use sidereon_core::ephemeris::{merge, MergeOptions, Sp3, Sp3AccuracyValue, Sp3WriteError};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const A: &str = include_str!("../fixtures/A.sp3");
const B: &str = include_str!("../fixtures/B.sp3");
const B_SIGMA: f64 = 0.018189894035458565;

fn sat(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).unwrap()
}

fn sigma(product: &Sp3, prn: u8) -> Sp3AccuracyValue {
    product
        .record_accuracy(sat(prn), 0)
        .unwrap()
        .p
        .unwrap()
        .position_sigma_m[0]
}

#[test]
fn repaired_original_merge_and_both_existing_writers() {
    let a = sidereon::load_sp3(A.as_bytes()).unwrap();
    let b = sidereon::load_sp3(B.as_bytes()).unwrap();
    for source in [&a, &b] {
        source.to_sp3_string().unwrap();
        source.to_sp3_string_with_rounding().unwrap();
    }
    let (merged, _) = merge(&[a, b], &MergeOptions::default()).unwrap();
    assert_eq!(merged.header.pos_vel_base, Some(1.25));
    assert_eq!(sigma(&merged, 12), Sp3AccuracyValue::Known(B_SIGMA));
    let strict = Sp3::parse(merged.to_sp3_string().unwrap().as_bytes()).unwrap();
    let (rounded, _) = merged.to_sp3_string_with_rounding().unwrap();
    let rounded = Sp3::parse(rounded.as_bytes()).unwrap();
    for result in [&strict, &rounded] {
        assert_eq!(sigma(result, 12), Sp3AccuracyValue::Known(B_SIGMA));
        assert_eq!(sigma(result, 1), Sp3AccuracyValue::Unknown);
    }
    let (_, report): (String, Sp3DetailedRoundingReport) = merged
        .to_sp3_string_with_accuracy_rounding(Sp3AccuracyRoundingPolicy::Conservative)
        .unwrap();
    assert_eq!(report.accuracy_sigmas.position_sigma_x_m, 0.0);
}

#[test]
fn constrained_bases_refuse_strict_and_report_conservative_conversion() {
    let constrained_a = A
        .replacen("%f  0.0000000", "%f  2.0000000", 1)
        .lines()
        .map(|line| {
            if line.starts_with("PG01") {
                format!("{line}  1 99 99 999  ")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let a = Sp3::parse(constrained_a.as_bytes()).unwrap();
    let b = Sp3::parse(B.as_bytes()).unwrap();
    let (merged, _) = merge(&[a, b], &MergeOptions::default()).unwrap();
    assert_eq!(merged.header.pos_vel_base, Some(2.0));
    assert!(matches!(
        merged.to_sp3_string(),
        Err(Sp3WriteError::AccuracyNotRepresentable { .. })
    ));
    assert!(matches!(
        merged.to_sp3_string_with_rounding(),
        Err(Sp3WriteError::AccuracyNotRepresentable { .. })
    ));
    let (text, report) = merged
        .to_sp3_string_with_accuracy_rounding(Sp3AccuracyRoundingPolicy::Conservative)
        .unwrap();
    assert_eq!(report.accuracy_sigmas.position_sigma_x_m, 0.032 - B_SIGMA);
    assert_eq!(report.accuracy_sigmas.position_sigma_y_m, 0.0);
    assert_eq!(sigma(&merged, 12), Sp3AccuracyValue::Known(B_SIGMA));
    let reparsed = Sp3::parse(text.as_bytes()).unwrap();
    assert_eq!(sigma(&reparsed, 12), Sp3AccuracyValue::Known(0.032));
    assert_eq!(sigma(&reparsed, 1), Sp3AccuracyValue::Known(0.002));
}
