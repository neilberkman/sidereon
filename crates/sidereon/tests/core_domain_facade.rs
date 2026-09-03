#[test]
fn core_domain_modules_are_reachable_through_facade() {
    assert_eq!(
        sidereon::constellation::gnss_sp3_id(sidereon::GnssSystem::Gps, 3),
        "G03"
    );

    assert!(sidereon::ppp_corrections::PppCorrections::default()
        .diagnostics
        .warnings
        .is_empty());

    let double_difference = sidereon::rtk::DoubleDifference {
        satellite_id: "G03".to_string(),
        reference_satellite_id: "G01".to_string(),
        ambiguity_id: "G03-G01".to_string(),
        code_m: 0.0,
        phase_m: 0.0,
    };
    assert_eq!(double_difference.reference_satellite_id, "G01");

    assert_eq!(
        sidereon::staleness::StalenessPolicy::default().max_staleness_s,
        3.0 * sidereon::constants::SECONDS_PER_DAY
    );

    assert_eq!(
        sidereon::tides::TideInputErrorKind::Missing.to_string(),
        "missing"
    );

    assert!(sidereon::ils::lambda_ils_search(&[1.2], &[vec![1.0]], 3.0).is_ok());

    assert_eq!(
        sidereon::terrain::DtedLookupOptions::default().interpolation,
        sidereon::terrain::DtedInterpolation::Bilinear
    );

    let phase = [0.0, 1.0, 2.0];
    let oadev = sidereon::clock_stability::overlapping_adev(
        sidereon::clock_stability::AllanSeries::PhaseSeconds(&phase),
        1.0,
        &[1],
    )
    .expect("facade clock stability");
    assert_eq!(oadev.deviation[0].to_bits(), 0.0_f64.to_bits());

    let _options = sidereon::clock_stability::PowerLawNoiseOptions::sampled_at_nyquist(1.0);
    assert_eq!(
        sidereon::clock_stability::allan_deviation_power_law_slope(
            sidereon::clock_stability::PowerLawNoiseType::WhiteFM
        )
        .to_bits(),
        (-0.5_f64).to_bits()
    );

    assert_eq!(
        sidereon::atmosphere::troposphere::NIELL_MIN_MAPPING_ELEVATION_RAD.to_bits(),
        0x3faa_cee9_f37b_ebd6
    );

    let core_fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sidereon-core/tests/fixtures");
    let dted_tile = core_fixture_dir
        .join("dted/tiles")
        .join("n36_w107_1arc_v3.dt2");
    let terrain_id = sidereon::terrain_store::TerrainTileId::new(36, -107);
    let entries = [sidereon::terrain_store::DtedTileListEntry::new(
        terrain_id, dted_tile,
    )];
    let store_bytes =
        sidereon::terrain_store::dted_tile_list_to_mmap_store(&entries).expect("facade list build");
    let store =
        sidereon::terrain_store::MmapTerrain::from_bytes(&store_bytes).expect("facade store read");
    assert_eq!(store.tile_count(), 1);
    assert_eq!(store.tile_ids(), &[terrain_id]);

    let ionex_bytes = std::fs::read(core_fixture_dir.join("ionex/synthetic_2map_7x7.20i"))
        .expect("IONEX fixture");
    let ionex = sidereon::atmosphere::Ionex::parse(&ionex_bytes).expect("facade IONEX parse");
    let frequency_hz = sidereon::frequencies::frequency_hz(
        sidereon::GnssSystem::Gps,
        sidereon::frequencies::CarrierBand::L1,
    )
    .expect("facade GPS L1 frequency");
    let request = sidereon::atmosphere::IonexSlantRequest::new(
        sidereon::Wgs84Geodetic::new(30.0_f64.to_radians(), 0.0_f64.to_radians(), 0.0)
            .expect("facade receiver"),
        45.0_f64.to_radians(),
        90.0_f64.to_radians(),
        ionex.map_epochs_s()[0],
        frequency_hz,
    );
    let mut batch = [f64::NAN];
    ionex
        .slant_delays_batch(&[request], &mut batch)
        .expect("facade IONEX batch");
    let scalar = sidereon::atmosphere::ionex_slant_delay(
        &ionex,
        request.receiver,
        request.elevation_rad,
        request.azimuth_rad,
        request.epoch_j2000_s,
        request.frequency_hz,
    )
    .expect("facade IONEX scalar");
    assert_eq!(batch[0].to_bits(), scalar.to_bits());

    let _shadow_model = sidereon::astro::events::eclipse::EarthShadowModel::Wgs84Oblate;
    assert_eq!(
        sidereon::astro::events::eclipse::WGS84_FLATTENING.to_bits(),
        0x3f6b_775a_84f3_e128
    );
}
