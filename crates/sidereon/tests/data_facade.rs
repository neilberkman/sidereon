use sidereon::data::{
    default_sample_for_date, dted_cache_relpath, hgt_to_dted, mgex_nav, mgex_sp3, ops_ultra_sp3,
    parse_skadi_tile_id, skadi_archive_url, station_obs_url, terrain_tile_index,
    ultra_sp3_locations, validate_exact_product_set, AnalysisCenter, ProductDate, ProductType,
};

#[test]
fn facade_reexports_data_catalog_derivation() {
    let product = ops_ultra_sp3(
        AnalysisCenter::IgsUlt,
        ProductDate::new(2024, 9, 3).expect("valid date"),
        None,
        Some("0600"),
    )
    .expect("ultra product");

    assert_eq!(
        product.canonical_filename().expect("filename"),
        "IGS0OPSULT_20242470600_02D_15M_ORB.SP3"
    );
    assert_eq!(
        product.archive_url().expect("url"),
        "https://igs.bkg.bund.de/root_ftp/IGS/products/2330/IGS0OPSULT_20242470600_02D_15M_ORB.SP3.gz"
    );
}

#[test]
fn facade_reexports_date_aware_product_defaults() {
    let legacy = ProductDate::new(2021, 5, 17).expect("valid legacy date");
    let current = ProductDate::new(2026, 7, 19).expect("valid current date");

    assert_eq!(
        default_sample_for_date(AnalysisCenter::Gfz, ProductType::Sp3, legacy),
        Ok("15M")
    );
    assert_eq!(
        default_sample_for_date(AnalysisCenter::Gfz, ProductType::Sp3, current),
        Ok("05M")
    );
}

#[test]
fn facade_reexports_exact_product_set_gate() {
    let first = mgex_sp3(
        AnalysisCenter::Cod,
        ProductDate::new(2026, 7, 12).expect("valid date"),
        None,
    )
    .expect("first product")
    .identity()
    .expect("first identity");
    let second = mgex_sp3(
        AnalysisCenter::Cod,
        ProductDate::new(2026, 7, 13).expect("valid date"),
        None,
    )
    .expect("second product")
    .identity()
    .expect("second identity");

    assert_eq!(
        validate_exact_product_set(&[first.clone(), second.clone()], &[second.clone(), first]),
        Ok(())
    );
    assert!(validate_exact_product_set(&[second], &[]).is_err());
}

#[test]
fn facade_reexports_ultra_sp3_fallback_locations() {
    let locations = ultra_sp3_locations(
        AnalysisCenter::EsaUlt,
        ProductDate::new(2026, 7, 13).expect("valid date"),
        "0000",
    )
    .expect("ultra locations");

    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].pattern, "primary_02D_05M");

    let code_locations = ultra_sp3_locations(
        AnalysisCenter::CodUlt,
        ProductDate::new(2026, 7, 14).expect("valid date"),
        "0000",
    )
    .expect("CODE ultra locations");
    assert_eq!(code_locations[0].pattern, "primary_01D_05M");
    assert_eq!(
        code_locations[0].url,
        "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20261950000_01D_05M_ORB.SP3"
    );
    assert_eq!(code_locations.len(), 1);
}

#[test]
fn facade_reexports_expanded_data_catalog_derivation() {
    let date = ProductDate::new(2020, 6, 25).expect("valid date");
    let nav = mgex_nav(AnalysisCenter::Igs, date, None).expect("nav product");

    assert_eq!(
        nav.canonical_filename().expect("filename"),
        "BRDC00WRD_R_20201770000_01D_MN.rnx"
    );
    assert_eq!(
        station_obs_url("WTZR00DEU", date, "30S").expect("url"),
        "https://igs.bkg.bund.de/root_ftp/IGS/obs/2020/177/WTZR00DEU_R_20201770000_01D_30S_MO.crx.gz"
    );
}

#[test]
fn facade_reexports_terrain_data_derivation_and_conversion() {
    assert_eq!(
        terrain_tile_index(36.75, -106.25).expect("tile index"),
        (36, -107)
    );
    assert_eq!(
        parse_skadi_tile_id("N36W107").expect("parse tile id"),
        (36, -107)
    );
    assert_eq!(
        skadi_archive_url(36, -107).expect("skadi URL"),
        "https://s3.amazonaws.com/elevation-tiles-prod/skadi/N36/N36W107.hgt.gz"
    );
    assert_eq!(
        dted_cache_relpath(36, -107).expect("DTED cache path"),
        "n30_w100/n36_w107_1arc_v3.dt2"
    );
    assert!(hgt_to_dted(36, -107, &[]).is_err());
}

#[test]
fn facade_reexports_publication_resilience_apis() {
    use sidereon::data::{
        newest_published_product, parse_archive_listing, predicted_ionex_line_candidates,
        publication_listing_urls, resolve_first_published,
    };

    let map_date = ProductDate::new(2026, 8, 5).expect("valid date");
    let candidates = predicted_ionex_line_candidates(map_date, None).expect("candidates");
    assert_eq!(candidates.len(), 2);

    let listing = "CODE/IONO/P2/2026/COD0OPSPRD_20262170000_01D_01H_GIM.INX.gz;1;\
2026-08-04T06:51:15Z;00";
    let objects = parse_archive_listing(listing).expect("recognized listing");
    assert_eq!(
        resolve_first_published(&candidates, &objects).expect("resolvable"),
        Some(1)
    );
    let newest = newest_published_product(
        sidereon::data::AnalysisCenter::CodPrd2,
        ProductType::Ionex,
        &objects,
    )
    .expect("supported line")
    .expect("published");
    assert_eq!(newest.date, map_date);

    assert_eq!(
        publication_listing_urls(
            sidereon::data::AnalysisCenter::WumNrt,
            ProductType::Sp3,
            ProductDate::new(2026, 8, 4).expect("valid date"),
        )
        .expect("listing URLs"),
        vec![
            "ftp://igs.gnsswhu.cn/pub/gps/products/mgex/2430/".to_string(),
            "ftp://igs.gnsswhu.cn/pub/gps/products/mgex/2429/".to_string(),
        ]
    );
}
