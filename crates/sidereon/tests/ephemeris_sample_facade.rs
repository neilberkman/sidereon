use sidereon::ephemeris::{
    sample, EphemerisSampleStatus, MergeOptions, MergePrecedenceScope,
    MmapPreciseEphemerisInterpolant, OutlierRejectOptions, PreciseEphemerisInterpolant, Sp3,
};

#[test]
fn facade_reexports_sp3_merge_controls_and_prediction_metadata() {
    let sp3 = Sp3::parse(DEGENERATE_SP3).expect("parse SP3 fixture");
    let summary = sp3.prediction_summary();
    assert!(summary.epochs.iter().all(|epoch| epoch.is_observed()));
    assert_eq!(summary.observed_through, sp3.epochs.last().copied());

    let mut options = MergeOptions::default();
    options.precedence_scope = MergePrecedenceScope::SatelliteArc;
    options.outlier_reject = Some(OutlierRejectOptions {
        position_tolerance_m: 0.5,
        clock_tolerance_s: 5.0e-9,
    });
    assert_eq!(options.precedence_scope, MergePrecedenceScope::SatelliteArc);
    assert!(options.outlier_reject.is_some());
}
use sidereon::{
    emission_media_batch_at_j2000_s, EmissionMediaBatchOptions, EmissionMediaStatus,
    GnssSatelliteId, GnssSystem,
};

const DEGENERATE_SP3: &[u8] =
    include_bytes!("../../sidereon-core/tests/fixtures/sp3/degenerate_coincident_5sat.sp3");

#[test]
fn facade_reexports_ephemeris_sampler() {
    let sp3 = Sp3::parse(DEGENERATE_SP3).expect("parse SP3 fixture");
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid G01");
    let epochs = sp3.epochs_j2000_seconds();

    let rows = sample(&sp3, &[sat], epochs[0], epochs[1], epochs[1] - epochs[0])
        .expect("sample through facade re-export");

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].sat, sat);
    assert_eq!(rows[0].status, EphemerisSampleStatus::Valid);
    assert!(rows[0].position_ecef_m.is_some());
    assert_eq!(rows[0].clock_s, Some(0.0));
}

#[test]
fn facade_reexports_emission_media_batch() {
    let sp3 = Sp3::parse(DEGENERATE_SP3).expect("parse SP3 fixture");
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid G01");
    let epoch = sp3.epochs_j2000_seconds()[0];

    let batch = emission_media_batch_at_j2000_s(
        &sp3,
        &[sat],
        &[epoch],
        [6_378_137.0, 0.0, 0.0],
        EmissionMediaBatchOptions::default(),
    )
    .expect("facade emission media batch");

    assert_eq!(batch.element_status(0), Some(EmissionMediaStatus::Valid));
    assert_eq!(batch.positions_ecef_m[0], Some([26_560_000.0, 0.0, 0.0]));
    assert_eq!(batch.clocks_s[0], Some(0.0));
    assert_eq!(batch.ionosphere_slant_delays_m[0], Some(0.0));
    assert_eq!(batch.troposphere_delays_m[0], Some(0.0));
}

#[test]
fn facade_reexports_precise_interpolant_store() {
    let sp3 = Sp3::parse(DEGENERATE_SP3).expect("parse SP3 fixture");
    let memory = PreciseEphemerisInterpolant::from_sp3(&sp3);
    let store_bytes = memory
        .to_mmap_store_bytes()
        .expect("build precise interpolant store");
    let mapped =
        MmapPreciseEphemerisInterpolant::from_bytes(&store_bytes).expect("open mapped store");
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid G01");
    let epoch = sp3.epochs_j2000_seconds()[0];

    let got = mapped
        .position_at_j2000_seconds(sat, epoch)
        .expect("mapped facade evaluation");
    let want = memory
        .position_at_j2000_seconds(sat, epoch)
        .expect("in-memory facade evaluation");

    assert_eq!(
        got.position.as_array().map(f64::to_bits),
        want.position.as_array().map(f64::to_bits)
    );
    assert_eq!(
        got.clock_s.map(f64::to_bits),
        want.clock_s.map(f64::to_bits)
    );
    assert_eq!(
        sidereon::precise_interpolant_store_checksum64(&store_bytes),
        mapped.checksum64()
    );
}
