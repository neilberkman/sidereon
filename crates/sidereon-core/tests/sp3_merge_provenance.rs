use std::collections::BTreeSet;

use sidereon_core::data::{
    distribution_location_for_identity, product, AnalysisCenter, ArchiveCompression,
    DistributionSource, ProductDate, ProductType,
};
use sidereon_core::ephemeris::{
    MergeCombine, MergeOptions, MergePrecedenceScope, OutlierRejectOptions, Sp3ArtifactIdentity,
    Sp3FrameLabelSet, Sp3FrameReconciliationOptions, Sp3MergeInputIdentity,
    Sp3MergeInputIdentityError,
};
use sidereon_core::GnssSystem;

fn identity(center: AnalysisCenter) -> sidereon_core::data::ProductIdentity {
    let issue = match center {
        AnalysisCenter::IgsUlt
        | AnalysisCenter::CodUlt
        | AnalysisCenter::EsaUlt
        | AnalysisCenter::GfzUlt
        | AnalysisCenter::WumNrt => Some("0000"),
        _ => None,
    };
    product(
        center,
        ProductType::Sp3,
        ProductDate::new(2026, 7, 16).unwrap(),
        None,
        issue,
    )
    .unwrap()
    .identity()
    .unwrap()
}

fn artifact(center: AnalysisCenter, byte: u8) -> Sp3ArtifactIdentity {
    let requested_identity = identity(center);
    let mut resolved_identity = requested_identity.clone();
    resolved_identity.format_version = Some("SP3-d".to_string());
    Sp3ArtifactIdentity {
        official_filename: requested_identity.official_filename.clone(),
        requested_identity,
        resolved_identity,
        distribution_source: DistributionSource::Direct,
        product_sha256: format!("{byte:02x}").repeat(32),
        product_byte_length: 12_345,
        archive_sha256: format!("{:02x}", byte.wrapping_add(1)).repeat(32),
        archive_byte_length: 6_789,
        compression: ArchiveCompression::Gzip,
    }
}

fn complete_policy(combine: MergeCombine) -> MergeOptions {
    MergeOptions {
        position_tolerance_m: 0.0,
        clock_tolerance_s: 2.5e-9,
        min_agree: 2,
        clock_min_common: 3,
        combine,
        precedence_scope: MergePrecedenceScope::SatelliteArc,
        outlier_reject: Some(OutlierRejectOptions {
            position_tolerance_m: 1.25,
            clock_tolerance_s: 7.5e-9,
        }),
        target_epoch_interval_s: Some(900.0),
        systems: Some(BTreeSet::from([GnssSystem::Gps, GnssSystem::Galileo])),
        frame_reconciliation: Sp3FrameReconciliationOptions {
            asserted_equivalent_label_sets: vec![
                Sp3FrameLabelSet::new(["IGS20", "ITRF2020"]),
                Sp3FrameLabelSet::new(["IGS14", "ITRF2014"]),
            ],
            helmert: true,
        },
    }
}

#[test]
fn public_v1_golden_vectors_are_literal_and_cross_surface_stable() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../golden/sp3-merge-input-v1.json")).unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let expected = &fixture["expected"];
    let first = artifact(AnalysisCenter::Esa, 0x11);
    let second = artifact(AnalysisCenter::Cod, 0x22);
    assert_eq!(
        fixture["artifacts"]["esa"]["official_filename"],
        first.official_filename
    );
    assert_eq!(
        fixture["artifacts"]["esa"]["product_sha256"],
        first.product_sha256
    );
    assert_eq!(
        fixture["artifacts"]["cod"]["official_filename"],
        second.official_filename
    );
    assert_eq!(
        fixture["artifacts"]["cod"]["product_sha256"],
        second.product_sha256
    );
    let mean = Sp3MergeInputIdentity::new(
        &[first.clone(), second.clone()],
        &complete_policy(MergeCombine::Mean),
    )
    .unwrap();
    let mean_reverse = Sp3MergeInputIdentity::new(
        &[second.clone(), first.clone()],
        &complete_policy(MergeCombine::Mean),
    )
    .unwrap();
    assert_eq!(mean.stable_id, expected["mean_esa_cod"].as_str().unwrap());
    assert_eq!(mean, mean_reverse);
    assert_eq!(
        mean.contributors[0].official_filename,
        second.official_filename
    );
    assert_eq!(
        mean.contributors[1].official_filename,
        first.official_filename
    );
    assert_eq!(mean.precedence_contributors, None);

    let median = Sp3MergeInputIdentity::new(
        &[first.clone(), second.clone()],
        &complete_policy(MergeCombine::Median),
    )
    .unwrap();
    let median_reverse = Sp3MergeInputIdentity::new(
        &[second.clone(), first.clone()],
        &complete_policy(MergeCombine::Median),
    )
    .unwrap();
    assert_eq!(
        median.stable_id,
        expected["median_esa_cod"].as_str().unwrap()
    );
    assert_eq!(median, median_reverse);

    let precedence = Sp3MergeInputIdentity::new(
        &[first.clone(), second.clone()],
        &complete_policy(MergeCombine::Precedence),
    )
    .unwrap();
    let precedence_reverse = Sp3MergeInputIdentity::new(
        &[second.clone(), first.clone()],
        &complete_policy(MergeCombine::Precedence),
    )
    .unwrap();
    assert_eq!(
        precedence.stable_id,
        expected["precedence_esa_cod"].as_str().unwrap()
    );
    assert_eq!(
        precedence_reverse.stable_id,
        expected["precedence_cod_esa"].as_str().unwrap()
    );
    assert_eq!(
        precedence
            .precedence_contributors
            .as_ref()
            .unwrap()
            .first()
            .unwrap()
            .official_filename,
        first.official_filename
    );

    let single = Sp3MergeInputIdentity::new(
        std::slice::from_ref(&first),
        &complete_policy(MergeCombine::Mean),
    )
    .unwrap();
    assert_eq!(
        single.stable_id,
        expected["single_mean_esa"].as_str().unwrap()
    );
    assert_ne!(single.stable_id, mean.stable_id);

    let mutations = &fixture["required_mutations"];
    let mut changed_bytes = first.clone();
    changed_bytes.product_sha256 = mutations["changed_product_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let changed_bytes = Sp3MergeInputIdentity::new(
        &[changed_bytes, second.clone()],
        &complete_policy(MergeCombine::Mean),
    )
    .unwrap();
    assert_ne!(changed_bytes.stable_id, mean.stable_id);

    let mut changed_resolved = first.clone();
    changed_resolved.resolved_identity.format_version = Some(
        mutations["changed_resolved_format_version"]
            .as_str()
            .unwrap()
            .to_string(),
    );
    let changed_resolved = Sp3MergeInputIdentity::new(
        &[changed_resolved, second.clone()],
        &complete_policy(MergeCombine::Mean),
    )
    .unwrap();
    assert_ne!(changed_resolved.stable_id, mean.stable_id);

    let mut changed_policy = complete_policy(MergeCombine::Mean);
    changed_policy.clock_tolerance_s = mutations["changed_clock_tolerance_s"].as_f64().unwrap();
    let changed_policy =
        Sp3MergeInputIdentity::new(&[first.clone(), second.clone()], &changed_policy).unwrap();
    assert_ne!(changed_policy.stable_id, mean.stable_id);

    let mut reordered_policy = complete_policy(MergeCombine::Mean);
    reordered_policy
        .frame_reconciliation
        .asserted_equivalent_label_sets
        .reverse();
    reordered_policy.systems = Some([GnssSystem::Galileo, GnssSystem::Gps].into_iter().collect());
    let reordered =
        Sp3MergeInputIdentity::new(&[second.clone(), first.clone()], &reordered_policy).unwrap();
    assert_eq!(reordered.stable_id, mean.stable_id);

    let mut malformed = first;
    malformed.product_sha256 = mutations["malformed_product_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[malformed, second], &complete_policy(MergeCombine::Mean)),
        Err(Sp3MergeInputIdentityError::InvalidContributor { .. })
    ));
}

#[test]
fn merge_input_identity_is_order_independent_and_verifiable() {
    let first = artifact(AnalysisCenter::Esa, 0x11);
    let second = artifact(AnalysisCenter::Cod, 0x22);
    let policy = MergeOptions::default();

    let forward = Sp3MergeInputIdentity::new(&[first.clone(), second.clone()], &policy).unwrap();
    let reverse = Sp3MergeInputIdentity::new(&[second.clone(), first.clone()], &policy).unwrap();

    assert_eq!(forward.stable_id, reverse.stable_id);
    assert_eq!(forward.contributors, reverse.contributors);
    assert!(forward.verify(&policy).unwrap());
    assert!(forward
        .verify_against(&[second.clone(), first.clone()], &policy)
        .unwrap());

    let mut corrupted_record = forward.clone();
    corrupted_record.contributors[0].product_sha256 = "ff".repeat(32);
    assert!(!corrupted_record.verify(&policy).unwrap());
    assert!(!corrupted_record
        .verify_against(&[second, first], &policy)
        .unwrap());
}

#[test]
fn precedence_order_is_bound_as_semantic_merge_policy() {
    let first = artifact(AnalysisCenter::Esa, 0x11);
    let second = artifact(AnalysisCenter::Cod, 0x22);
    let policy = MergeOptions {
        combine: MergeCombine::Precedence,
        ..MergeOptions::default()
    };

    let forward = Sp3MergeInputIdentity::new(&[first.clone(), second.clone()], &policy).unwrap();
    let reverse = Sp3MergeInputIdentity::new(&[second, first], &policy).unwrap();

    assert_ne!(forward.stable_id, reverse.stable_id);
    assert_eq!(forward.contributors, reverse.contributors);
    assert_ne!(
        forward.precedence_contributors,
        reverse.precedence_contributors
    );
    assert!(forward.verify(&policy).unwrap());
    assert!(reverse.verify(&policy).unwrap());
}

#[test]
fn artifact_or_policy_changes_change_the_stable_identity() {
    let first = artifact(AnalysisCenter::Esa, 0x11);
    let second = artifact(AnalysisCenter::Cod, 0x22);
    let original =
        Sp3MergeInputIdentity::new(&[first.clone(), second.clone()], &MergeOptions::default())
            .unwrap();

    let mut changed_artifact = second.clone();
    changed_artifact.product_sha256 = "33".repeat(32);
    let changed_artifact =
        Sp3MergeInputIdentity::new(&[first.clone(), changed_artifact], &MergeOptions::default())
            .unwrap();
    assert_ne!(original.stable_id, changed_artifact.stable_id);

    let changed_policy = MergeOptions {
        combine: MergeCombine::Median,
        ..MergeOptions::default()
    };
    let changed_policy = Sp3MergeInputIdentity::new(&[first, second], &changed_policy).unwrap();
    assert_ne!(original.stable_id, changed_policy.stable_id);
}

#[test]
fn policy_set_order_does_not_change_the_stable_identity() {
    let contributor = artifact(AnalysisCenter::Esa, 0x11);
    let first = MergeOptions {
        systems: Some(BTreeSet::from([GnssSystem::Galileo, GnssSystem::Gps])),
        frame_reconciliation: Sp3FrameReconciliationOptions {
            asserted_equivalent_label_sets: vec![
                Sp3FrameLabelSet::new(["IGS20", "ITRF2020"]),
                Sp3FrameLabelSet::new(["IGS14", "ITRF2014"]),
            ],
            helmert: false,
        },
        ..MergeOptions::default()
    };

    let second = MergeOptions {
        systems: Some(BTreeSet::from([GnssSystem::Gps, GnssSystem::Galileo])),
        frame_reconciliation: Sp3FrameReconciliationOptions {
            asserted_equivalent_label_sets: vec![
                Sp3FrameLabelSet::new(["ITRF2014", "IGS14"]),
                Sp3FrameLabelSet::new(["ITRF2020", "IGS20"]),
            ],
            helmert: false,
        },
        ..MergeOptions::default()
    };

    let first = Sp3MergeInputIdentity::new(std::slice::from_ref(&contributor), &first).unwrap();
    let second = Sp3MergeInputIdentity::new(&[contributor], &second).unwrap();
    assert_eq!(first.stable_id, second.stable_id);
}

#[test]
fn non_executable_merge_policies_fail_closed() {
    let contributor = artifact(AnalysisCenter::Esa, 0x11);
    let empty_systems = MergeOptions {
        systems: Some(BTreeSet::new()),
        ..MergeOptions::default()
    };
    assert!(matches!(
        Sp3MergeInputIdentity::new(std::slice::from_ref(&contributor), &empty_systems),
        Err(Sp3MergeInputIdentityError::InvalidPolicy("systems filter"))
    ));

    let fractional_interval = MergeOptions {
        target_epoch_interval_s: Some(1.5),
        ..MergeOptions::default()
    };
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[contributor], &fractional_interval),
        Err(Sp3MergeInputIdentityError::InvalidPolicy(
            "target epoch interval"
        ))
    ));

    let near_zero_interval = MergeOptions {
        target_epoch_interval_s: Some(1.0e-12),
        ..MergeOptions::default()
    };
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[artifact(AnalysisCenter::Esa, 0x22)], &near_zero_interval),
        Err(Sp3MergeInputIdentityError::InvalidPolicy(
            "target epoch interval"
        ))
    ));

    let incomplete_frame_set = MergeOptions {
        frame_reconciliation: Sp3FrameReconciliationOptions {
            asserted_equivalent_label_sets: vec![Sp3FrameLabelSet::new(["IGS20"])],
            helmert: false,
        },
        ..MergeOptions::default()
    };
    assert!(matches!(
        Sp3MergeInputIdentity::new(
            &[artifact(AnalysisCenter::Esa, 0x33)],
            &incomplete_frame_set
        ),
        Err(Sp3MergeInputIdentityError::InvalidPolicy(
            "asserted frame label set"
        ))
    ));
}

#[test]
fn negative_zero_tolerances_have_the_same_identity_as_positive_zero() {
    let contributor = artifact(AnalysisCenter::Esa, 0x11);
    let positive = MergeOptions {
        position_tolerance_m: 0.0,
        clock_tolerance_s: 0.0,
        outlier_reject: Some(OutlierRejectOptions {
            position_tolerance_m: 0.0,
            clock_tolerance_s: 0.0,
        }),
        ..MergeOptions::default()
    };
    let negative = MergeOptions {
        position_tolerance_m: -0.0,
        clock_tolerance_s: -0.0,
        outlier_reject: Some(OutlierRejectOptions {
            position_tolerance_m: -0.0,
            clock_tolerance_s: -0.0,
        }),
        ..MergeOptions::default()
    };

    let positive =
        Sp3MergeInputIdentity::new(std::slice::from_ref(&contributor), &positive).unwrap();
    let negative = Sp3MergeInputIdentity::new(&[contributor], &negative).unwrap();
    assert_eq!(positive.stable_id, negative.stable_id);
}

#[test]
fn incomplete_or_mismatched_provenance_fails_closed() {
    let mut contributor = artifact(AnalysisCenter::Esa, 0x11);
    contributor.archive_sha256 = "not-a-digest".to_string();
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[contributor], &MergeOptions::default()),
        Err(Sp3MergeInputIdentityError::InvalidContributor { .. })
    ));

    let mut contributor = artifact(AnalysisCenter::Esa, 0x11);
    contributor.resolved_identity = identity(AnalysisCenter::Cod);
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[contributor], &MergeOptions::default()),
        Err(Sp3MergeInputIdentityError::InvalidContributor { .. })
    ));

    let mut contributor = artifact(AnalysisCenter::Esa, 0x11);
    contributor.resolved_identity.format_version = None;
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[contributor], &MergeOptions::default()),
        Err(Sp3MergeInputIdentityError::InvalidContributor { .. })
    ));

    let mut contributor = artifact(AnalysisCenter::Esa, 0x11);
    contributor.requested_identity.format_version = Some("SP3-c".to_string());
    contributor.resolved_identity.format_version = Some("SP3-d".to_string());
    assert!(matches!(
        Sp3MergeInputIdentity::new(&[contributor], &MergeOptions::default()),
        Err(Sp3MergeInputIdentityError::InvalidContributor { .. })
    ));
}

#[test]
fn direct_location_retains_cataloged_overlap_cadence() {
    let alternate = product(
        AnalysisCenter::GfzUlt,
        ProductType::Sp3,
        ProductDate::new(2021, 5, 15).unwrap(),
        Some("05M"),
        Some("0000"),
    )
    .unwrap()
    .identity()
    .unwrap();
    assert_eq!(alternate.span, "02D");
    assert_eq!(alternate.sample, "05M");

    let location =
        distribution_location_for_identity(&alternate, DistributionSource::Direct).unwrap();
    assert!(location
        .original_url
        .as_deref()
        .unwrap()
        .ends_with(&location.archive_filename));
    assert!(location
        .archive_filename
        .starts_with(&alternate.official_filename));
}

/// The multi-center ultra consensus path is not limited to the ESA/GFZ pair:
/// the IGS combined ultra and the Wuhan MGEX near-real-time line participate
/// behind their catalog entries, and the merge-input identity names every
/// contributor's line distinctly.
#[test]
fn four_center_ultra_consensus_includes_igs_and_wum() {
    let contributors = [
        artifact(AnalysisCenter::EsaUlt, 0x31),
        artifact(AnalysisCenter::GfzUlt, 0x32),
        artifact(AnalysisCenter::IgsUlt, 0x33),
        artifact(AnalysisCenter::WumNrt, 0x34),
    ];
    let policy = complete_policy(MergeCombine::Median);
    let merge_input = Sp3MergeInputIdentity::new(&contributors, &policy).expect("merge input");

    assert_eq!(merge_input.contributors.len(), 4);
    let filenames: BTreeSet<&str> = merge_input
        .contributors
        .iter()
        .map(|contributor| contributor.official_filename.as_str())
        .collect();
    assert!(filenames.contains("IGS0OPSULT_20261970000_02D_15M_ORB.SP3"));
    assert!(filenames.contains("WUM0MGXNRT_20261970000_02D_05M_ORB.SP3"));
    assert!(filenames.contains("ESA0OPSULT_20261970000_02D_05M_ORB.SP3"));
    assert!(filenames.contains("GFZ0OPSULT_20261970000_02D_05M_ORB.SP3"));

    assert_eq!(merge_input.verify(&policy), Ok(true));

    // Contributor order never changes the stable consensus identity.
    let mut reversed = contributors.clone();
    reversed.reverse();
    let reversed_input = Sp3MergeInputIdentity::new(&reversed, &policy).expect("merge input");
    assert_eq!(merge_input, reversed_input);
}
