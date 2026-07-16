use std::process::Command;

use sidereon_core::data::{
    allowed_hosts, archive_url, canonical_filename, catalog, cddis_archive_url, dted_block_dir,
    dted_cache_relpath, dted_tile_filename, gim_date_candidates, latest_ops_ultra_sp3, mgex_clk,
    mgex_ionex, mgex_nav, mgex_sp3, no_open_mirrors, open_mirror_code, ops_ultra_sp3,
    parse_skadi_tile_id, predicted_ionex, product_convention, rapid_ionex, skadi_archive_url,
    skadi_band, skadi_source_entry, skadi_tile_id, space_weather_archive_url,
    space_weather_cache_relpath, space_weather_filename, space_weather_source_entry, station_obs,
    station_obs_filename, station_obs_protocol, station_obs_url, terrain_tile_index,
    ultra_sp3_locations, validate_exact_product_set, AnalysisCenter, ArchiveCompression,
    ArchiveProtocol, DataCatalogError, DistributionSource, ExactProductSetError, ProductCampaign,
    ProductDate, ProductDateTime, ProductFormat, ProductPublisher, ProductRequest, ProductType,
    SolutionClass, SpaceWeatherProduct, UltraIssue,
};

fn date(year: i32, month: u8, day: u8) -> ProductDate {
    ProductDate::new(year, month, day).expect("valid test date")
}

#[test]
fn final_sp3_urls_match_binding_catalog_examples() {
    let esa = mgex_sp3(AnalysisCenter::Esa, date(2020, 6, 24), None).expect("ESA SP3 product");
    assert_eq!(
        esa.canonical_filename().expect("filename"),
        "ESA0MGNFIN_20201760000_01D_05M_ORB.SP3"
    );
    assert_eq!(
        esa.archive_url().expect("url"),
        "https://navigation-office.esa.int/products/gnss-products/2111/ESA0MGNFIN_20201760000_01D_05M_ORB.SP3.gz"
    );

    let gfz = mgex_sp3(AnalysisCenter::Gfz, date(2020, 6, 24), None).expect("GFZ SP3 product");
    assert_eq!(
        gfz.canonical_filename().expect("filename"),
        "GFZ0OPSRAP_20201760000_01D_15M_ORB.SP3"
    );
    assert_eq!(
        gfz.archive_url().expect("url"),
        "https://isdc-data.gfz.de/gnss/products/rapid/w2111/GFZ0OPSRAP_20201760000_01D_15M_ORB.SP3.gz"
    );
}

#[test]
fn ionex_urls_match_binding_catalog_examples() {
    let esa = mgex_ionex(AnalysisCenter::Esa, date(2024, 6, 24), None).expect("ESA IONEX product");
    assert_eq!(
        esa.canonical_filename().expect("filename"),
        "ESA0OPSFIN_20241760000_01D_02H_GIM.INX"
    );
    assert_eq!(
        esa.archive_url().expect("url"),
        "https://navigation-office.esa.int/products/gnss-products/2320/ESA0OPSFIN_20241760000_01D_02H_GIM.INX.gz"
    );

    let rapid = rapid_ionex(date(2026, 6, 13), None).expect("rapid IONEX product");
    assert_eq!(
        rapid.canonical_filename().expect("filename"),
        "COD0OPSRAP_20261640000_01D_01H_GIM.INX"
    );
    assert_eq!(
        rapid.archive_url().expect("url"),
        "http://ftp.aiub.unibe.ch/CODE/COD0OPSRAP_20261640000_01D_01H_GIM.INX.gz"
    );
}

#[test]
fn product_identity_is_independent_of_distributor() {
    let product =
        mgex_sp3(AnalysisCenter::Esa, date(2020, 6, 24), None).expect("ESA final SP3 product");
    let identity = product.identity().expect("identity");

    assert_eq!(identity.family, ProductType::Sp3);
    assert_eq!(identity.publisher, ProductPublisher::Esa);
    assert_eq!(identity.solution, SolutionClass::Final);
    assert_eq!(identity.campaign, ProductCampaign::MultiGnss);
    assert_eq!(identity.format, ProductFormat::Sp3);
    assert_eq!(identity.version, 0);
    assert_eq!(identity.span, "01D");
    assert_eq!(identity.sample, "05M");
    assert_eq!(
        identity.official_filename,
        "ESA0MGNFIN_20201760000_01D_05M_ORB.SP3"
    );

    let direct = product
        .distribution_location(DistributionSource::Direct)
        .expect("direct location");
    let cddis = product
        .distribution_location(DistributionSource::NasaCddis)
        .expect("CDDIS location");
    assert_eq!(product.identity().expect("identity"), identity);
    assert_eq!(direct.source, DistributionSource::Direct);
    assert_eq!(cddis.source, DistributionSource::NasaCddis);
    assert_eq!(direct.compression, ArchiveCompression::Gzip);
    assert_eq!(cddis.compression, ArchiveCompression::Gzip);
    assert_eq!(
        cddis.original_url.as_deref(),
        Some(
            "https://cddis.nasa.gov/archive/gnss/products/2111/\
ESA0MGNFIN_20201760000_01D_05M_ORB.SP3.gz"
        )
    );
}

#[test]
fn cddis_ionex_path_uses_current_year_and_day_of_year_layout() {
    let product = rapid_ionex(date(2026, 6, 13), None).expect("CODE rapid IONEX");
    let identity = product.identity().expect("identity");
    assert_eq!(identity.publisher, ProductPublisher::Code);
    assert_eq!(identity.solution, SolutionClass::Rapid);
    assert_eq!(identity.campaign, ProductCampaign::Operational);
    assert_eq!(identity.format, ProductFormat::Ionex);
    assert_eq!(
        cddis_archive_url(&identity).expect("CDDIS URL"),
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2026/164/\
COD0OPSRAP_20261640000_01D_01H_GIM.INX.gz"
    );
}

#[test]
fn exact_request_and_cache_key_keep_source_selection_explicit() {
    let product = ops_ultra_sp3(AnalysisCenter::IgsUlt, date(2024, 9, 3), None, Some("0600"))
        .expect("IGS ultra product");
    let identity = product.identity().expect("identity");
    let request = ProductRequest::new(
        identity.clone(),
        vec![DistributionSource::NasaCddis, DistributionSource::Direct],
    )
    .expect("request");
    assert_eq!(request.identity, identity);
    assert_eq!(
        request.distributors,
        vec![DistributionSource::NasaCddis, DistributionSource::Direct]
    );
    assert_ne!(
        request
            .identity
            .cache_relpath(DistributionSource::NasaCddis)
            .expect("CDDIS cache path"),
        request
            .identity
            .cache_relpath(DistributionSource::Direct)
            .expect("direct cache path")
    );
    assert_eq!(
        ProductRequest::new(identity, vec![]),
        Err(DataCatalogError::NoDistributionSources)
    );
}

#[test]
fn exact_identity_key_is_stable_across_interfaces() {
    let identity = mgex_sp3(AnalysisCenter::Cod, date(2026, 7, 12), None)
        .expect("CODE product")
        .identity()
        .expect("identity");
    assert_eq!(identity.analysis_center, AnalysisCenter::Cod);
    assert_eq!(identity.format_version, None);
    assert_eq!(
        identity.key().expect("cache key"),
        "cod-final-a91258c21fa4860c34ce"
    );
}

#[test]
fn exact_product_set_requires_every_declared_identity_before_processing() {
    let first = mgex_sp3(AnalysisCenter::Cod, date(2026, 7, 12), None)
        .expect("first product")
        .identity()
        .expect("first identity");
    let second = mgex_sp3(AnalysisCenter::Cod, date(2026, 7, 13), None)
        .expect("second product")
        .identity()
        .expect("second identity");

    assert_eq!(
        validate_exact_product_set(
            &[first.clone(), second.clone()],
            &[second.clone(), first.clone()]
        ),
        Ok(())
    );
    assert_eq!(
        validate_exact_product_set(&[], &[]),
        Err(ExactProductSetError::EmptyExpected)
    );
    assert_eq!(
        validate_exact_product_set(
            &[first.clone(), second.clone()],
            std::slice::from_ref(&first),
        ),
        Err(ExactProductSetError::Mismatch {
            missing: vec![second],
            unexpected: vec![],
            duplicate_expected: vec![],
            duplicate_available: vec![],
        })
    );
}

#[test]
fn exact_product_set_rejects_duplicates_and_undeclared_products() {
    let first = mgex_sp3(AnalysisCenter::Cod, date(2026, 7, 12), None)
        .expect("first product")
        .identity()
        .expect("first identity");
    let second = mgex_sp3(AnalysisCenter::Cod, date(2026, 7, 13), None)
        .expect("second product")
        .identity()
        .expect("second identity");

    assert_eq!(
        validate_exact_product_set(
            &[first.clone(), first.clone()],
            &[first.clone(), second.clone(), second.clone()]
        ),
        Err(ExactProductSetError::Mismatch {
            missing: vec![],
            unexpected: vec![second.clone()],
            duplicate_expected: vec![first],
            duplicate_available: vec![second],
        })
    );
}

#[test]
fn exact_product_set_compares_prediction_metadata_not_only_filenames() {
    let predicted_one_day = predicted_ionex(AnalysisCenter::CodPrd1, date(2026, 7, 15), None)
        .expect("one-day prediction")
        .identity()
        .expect("one-day identity");
    let predicted_two_day = predicted_ionex(AnalysisCenter::CodPrd2, date(2026, 7, 14), None)
        .expect("two-day prediction")
        .identity()
        .expect("two-day identity");
    assert_eq!(
        predicted_one_day.official_filename,
        predicted_two_day.official_filename
    );

    assert_eq!(
        validate_exact_product_set(
            std::slice::from_ref(&predicted_one_day),
            std::slice::from_ref(&predicted_two_day),
        ),
        Err(ExactProductSetError::Mismatch {
            missing: vec![predicted_one_day],
            unexpected: vec![predicted_two_day],
            duplicate_expected: vec![],
            duplicate_available: vec![],
        })
    );
}

#[test]
fn caller_constructed_identity_must_agree_with_its_official_filename() {
    let product = mgex_sp3(AnalysisCenter::Cod, date(2026, 7, 12), None).expect("CODE SP3");
    let identity = product.identity().expect("identity");
    assert_eq!(identity.campaign, ProductCampaign::MultiGnssExperiment);
    assert_eq!(identity.validate(), Ok(()));

    let mut inconsistent = identity.clone();
    inconsistent.publisher = ProductPublisher::Esa;
    assert_eq!(
        inconsistent.validate(),
        Err(DataCatalogError::InconsistentProductIdentity {
            field: "official_filename",
        })
    );
    assert_eq!(
        cddis_archive_url(&inconsistent),
        Err(DataCatalogError::InconsistentProductIdentity {
            field: "official_filename",
        })
    );
    assert_eq!(
        ProductRequest::new(inconsistent.clone(), vec![DistributionSource::NasaCddis]),
        Err(DataCatalogError::InconsistentProductIdentity {
            field: "official_filename",
        })
    );
    assert_eq!(
        inconsistent.cache_relpath(DistributionSource::NasaCddis),
        Err(DataCatalogError::InconsistentProductIdentity {
            field: "official_filename",
        })
    );

    let mut unsafe_identity = identity;
    unsafe_identity.official_filename = "../escape.SP3".to_string();
    assert_eq!(
        unsafe_identity.cache_relpath(DistributionSource::NasaCddis),
        Err(DataCatalogError::InvalidOfficialFilename(
            "../escape.SP3".to_string()
        ))
    );
}

#[test]
fn broadcast_navigation_identity_validates_fields_not_encoded_in_filename() {
    let product = mgex_nav(AnalysisCenter::Igs, date(2026, 7, 12), None).expect("IGS NAV");
    let identity = product.identity().expect("identity");
    assert_eq!(identity.validate(), Ok(()));

    let mut inconsistent = identity;
    inconsistent.sample = "30S".to_string();
    assert_eq!(
        inconsistent.validate(),
        Err(DataCatalogError::InconsistentProductIdentity {
            field: "broadcast_navigation",
        })
    );
}

#[test]
fn local_sources_have_no_network_url_and_cddis_does_not_expand_family_scope() {
    let sp3 = mgex_sp3(AnalysisCenter::Cod, date(2020, 6, 24), None).expect("SP3");
    let local = sp3
        .distribution_location(DistributionSource::LocalFile)
        .expect("local location");
    assert_eq!(local.original_url, None);
    assert_eq!(local.compression, ArchiveCompression::None);

    let nav = mgex_nav(AnalysisCenter::Igs, date(2020, 6, 25), None).expect("NAV");
    assert_eq!(
        nav.distribution_location(DistributionSource::NasaCddis),
        Err(DataCatalogError::UnsupportedDistribution {
            source: DistributionSource::NasaCddis,
            product_type: ProductType::Nav,
        })
    );
}

#[test]
fn clock_and_broadcast_nav_urls_match_binding_catalog_examples() {
    let clk = mgex_clk(AnalysisCenter::Gfz, date(2020, 6, 24), None).expect("GFZ clock product");
    assert_eq!(
        clk.canonical_filename().expect("filename"),
        "GFZ0OPSRAP_20201760000_01D_30S_CLK.CLK"
    );
    assert_eq!(
        clk.archive_url().expect("url"),
        "https://isdc-data.gfz.de/gnss/products/rapid/w2111/GFZ0OPSRAP_20201760000_01D_30S_CLK.CLK.gz"
    );

    let nav =
        mgex_nav(AnalysisCenter::Igs, date(2020, 6, 25), None).expect("IGS broadcast nav product");
    assert_eq!(
        nav.canonical_filename().expect("filename"),
        "BRDC00WRD_R_20201770000_01D_MN.rnx"
    );
    assert_eq!(
        nav.archive_url().expect("url"),
        "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2020/177/BRDC00WRD_R_20201770000_01D_MN.rnx.gz"
    );
}

#[test]
fn station_observation_derivation_matches_binding_catalog_examples() {
    assert_eq!(
        station_obs_filename("ESBC00DNK", date(2020, 6, 25), "30S").expect("filename"),
        "ESBC00DNK_R_20201770000_01D_30S_MO.crx"
    );
    assert_eq!(
        station_obs_url("WTZR00DEU", date(2020, 6, 25), "30S").expect("url"),
        "https://igs.bkg.bund.de/root_ftp/IGS/obs/2020/177/WTZR00DEU_R_20201770000_01D_30S_MO.crx.gz"
    );
    assert_eq!(station_obs_protocol(), ArchiveProtocol::Https);

    let obs = station_obs("WTZR00DEU", date(2020, 6, 25), None).expect("station obs product");
    assert_eq!(
        obs.canonical_filename().expect("filename"),
        "WTZR00DEU_R_20201770000_01D_30S_MO.crx"
    );
    assert_eq!(
        obs.archive_url().expect("url"),
        "https://igs.bkg.bund.de/root_ftp/IGS/obs/2020/177/WTZR00DEU_R_20201770000_01D_30S_MO.crx.gz"
    );
}

#[test]
fn mirror_gating_matches_binding_catalog() {
    let err = product_convention(AnalysisCenter::Igs, ProductType::Ionex)
        .expect_err("IGS IONEX is mirror gated");
    assert_eq!(
        err,
        DataCatalogError::NoOpenMirror {
            center: "igs".to_string(),
            product_type: "ionex".to_string()
        }
    );

    assert!(no_open_mirrors()
        .iter()
        .any(|entry| entry.center == "grg" && entry.product_type == "sp3"));
    assert_eq!(
        open_mirror_code("grg_ult", "clk"),
        Err(DataCatalogError::NoOpenMirror {
            center: "grg_ult".to_string(),
            product_type: "clk".to_string()
        })
    );
    assert!(open_mirror_code("igs", "nav").is_ok());
}

#[test]
fn predicted_ionex_aliases_apply_the_existing_date_offset() {
    let prd1 = predicted_ionex(AnalysisCenter::CodPrd1, date(2026, 6, 14), None).expect("prd1");
    assert_eq!(
        prd1.canonical_filename().expect("filename"),
        "COD0OPSPRD_20261650000_01D_01H_GIM.INX"
    );

    let prd2 = predicted_ionex(AnalysisCenter::CodPrd2, date(2026, 6, 14), None).expect("prd2");
    assert_eq!(
        prd2.canonical_filename().expect("filename"),
        "COD0OPSPRD_20261660000_01D_01H_GIM.INX"
    );

    let same_file_prd1 =
        predicted_ionex(AnalysisCenter::CodPrd1, date(2026, 6, 15), None).expect("prd1");
    assert_eq!(
        same_file_prd1.canonical_filename().expect("filename"),
        prd2.canonical_filename().expect("filename")
    );
    assert_ne!(
        same_file_prd1.identity().expect("identity").key(),
        prd2.identity().expect("identity").key(),
        "prediction horizon must remain part of the normalized cache identity"
    );
    assert_ne!(
        same_file_prd1
            .identity()
            .expect("P1 identity")
            .cache_relpath(DistributionSource::Direct)
            .expect("P1 cache path"),
        prd2.identity()
            .expect("P2 identity")
            .cache_relpath(DistributionSource::Direct)
            .expect("P2 cache path"),
        "P1 and P2 must not collide even when their official filenames match"
    );
}

#[test]
fn predicted_ionex_direct_urls_use_exact_aiub_tier_and_identity_year() {
    let p1 = predicted_ionex(AnalysisCenter::CodPrd1, date(2026, 7, 15), None)
        .expect("P1 predicted IONEX");
    assert_eq!(
        p1.archive_url().expect("P1 direct URL"),
        "https://www.aiub.unibe.ch/download/CODE/IONO/P1/2026/\
COD0OPSPRD_20261960000_01D_01H_GIM.INX.gz"
    );

    let p2 = predicted_ionex(AnalysisCenter::CodPrd2, date(2026, 7, 15), None)
        .expect("P2 predicted IONEX");
    assert_eq!(
        p2.archive_url().expect("P2 direct URL"),
        "https://www.aiub.unibe.ch/download/CODE/IONO/P2/2026/\
COD0OPSPRD_20261970000_01D_01H_GIM.INX.gz"
    );

    let boundary = predicted_ionex(AnalysisCenter::CodPrd2, date(2026, 12, 31), None)
        .expect("year-boundary P2 predicted IONEX");
    assert_eq!(boundary.date, date(2027, 1, 1));
    assert_eq!(
        boundary.archive_url().expect("year-boundary P2 URL"),
        "https://www.aiub.unibe.ch/download/CODE/IONO/P2/2027/\
COD0OPSPRD_20270010000_01D_01H_GIM.INX.gz"
    );
}

#[test]
fn ultra_rapid_sp3_urls_match_binding_catalog_examples() {
    let igs = ops_ultra_sp3(AnalysisCenter::IgsUlt, date(2024, 9, 3), None, Some("0600"))
        .expect("IGS ultra SP3 product");
    assert_eq!(
        igs.canonical_filename().expect("filename"),
        "IGS0OPSULT_20242470600_02D_15M_ORB.SP3"
    );
    assert_eq!(
        igs.archive_url().expect("url"),
        "https://igs.bkg.bund.de/root_ftp/IGS/products/2330/IGS0OPSULT_20242470600_02D_15M_ORB.SP3.gz"
    );

    let esa = ops_ultra_sp3(AnalysisCenter::EsaUlt, date(2024, 9, 3), None, Some("0600"))
        .expect("ESA ultra SP3 product");
    assert_eq!(
        esa.archive_url().expect("url"),
        "https://navigation-office.esa.int/products/gnss-products/2330/ESA0OPSULT_20242470600_02D_05M_ORB.SP3.gz"
    );

    let cod = ops_ultra_sp3(AnalysisCenter::CodUlt, date(2026, 7, 14), None, None)
        .expect("CODE ultra SP3 product");
    assert_eq!(
        cod.canonical_filename().expect("filename"),
        "COD0OPSULT_20261950000_01D_05M_ORB.SP3"
    );
    assert_eq!(
        cod.archive_url().expect("url"),
        "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20261950000_01D_05M_ORB.SP3"
    );
}

#[test]
fn ultra_rapid_sp3_locations_include_current_patterns_and_fallbacks() {
    let esa = ultra_sp3_locations(AnalysisCenter::EsaUlt, date(2026, 7, 12), "1800")
        .expect("ESA candidates");
    assert_eq!(esa[0].pattern, "primary_02D_05M");
    assert!(esa[0].filename.ends_with("_02D_05M_ORB.SP3"));
    assert_eq!(esa[1].pattern, "alternate_02D_15M");

    let code = ultra_sp3_locations(AnalysisCenter::CodUlt, date(2026, 7, 14), "0000")
        .expect("CODE candidates");
    assert_eq!(code.len(), 3);
    assert_eq!(code[0].pattern, "primary_01D_05M");
    assert_eq!(code[0].filename, "COD0OPSULT_20261950000_01D_05M_ORB.SP3");
    assert_eq!(
        code[0].url,
        "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20261950000_01D_05M_ORB.SP3"
    );
    assert_eq!(code[1].pattern, "alternate_02D_05M");
    assert_eq!(code[1].filename, "COD0OPSULT_20261950000_02D_05M_ORB.SP3");
    assert_eq!(
        code[1].url,
        "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20261950000_02D_05M_ORB.SP3"
    );
    let alias = &code[2];
    assert_eq!(alias.pattern, "alias_latest");
    assert_eq!(alias.filename, "COD0OPSULT.SP3");
    assert_eq!(
        alias.url,
        "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT.SP3"
    );
    assert!(code.iter().all(|candidate| candidate
        .url
        .starts_with("https://www.aiub.unibe.ch/download/CODE/")));
    assert!(allowed_hosts().contains(&"www.aiub.unibe.ch"));

    let catalog_entry = catalog()
        .iter()
        .find(|entry| entry.center == AnalysisCenter::CodUlt)
        .expect("CODE ultra catalog entry");
    assert_eq!(catalog_entry.protocol, ArchiveProtocol::Https);
    assert_eq!(catalog_entry.host, "www.aiub.unibe.ch");
    assert_eq!(catalog_entry.root_url, "https://www.aiub.unibe.ch/download");

    let gfz_target = ProductDateTime::new(date(2026, 7, 12), 10, 0, 0).expect("target");
    let gfz_issues =
        sidereon_core::data::ultra_issue_candidates(AnalysisCenter::GfzUlt, gfz_target)
            .expect("GFZ issues");
    assert_eq!(gfz_issues[0].issue, "0900");
}

#[test]
#[ignore = "network test for AIUB's current CODE ultra-rapid object store"]
fn live_aiub_code_ultra_day_195_downloads_and_parses_as_sp3() {
    let candidate = ultra_sp3_locations(AnalysisCenter::CodUlt, date(2026, 7, 14), "0000")
        .expect("CODE candidates")
        .remove(0);
    assert_eq!(candidate.pattern, "primary_01D_05M");

    let response = Command::new("curl")
        .args([
            "--http1.1",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            &candidate.url,
        ])
        .output()
        .expect("run curl");
    assert!(
        response.status.success(),
        "curl failed for {}: {}",
        candidate.url,
        String::from_utf8_lossy(&response.stderr)
    );
    assert!(response.stdout.starts_with(b"#dP"));
    assert_eq!(response.stdout.len(), 1_473_962);

    let sp3 = sidereon_core::ephemeris::Sp3::parse(&response.stdout)
        .expect("downloaded CODE object parses as SP3");
    assert_eq!(sp3.epoch_count(), 289);
    assert_eq!(sp3.header.epoch_interval_s, 300.0);
}

#[test]
fn free_functions_derive_string_identical_names_and_urls() {
    let name = canonical_filename(
        AnalysisCenter::GfzUlt,
        ProductType::Sp3,
        date(2024, 9, 3),
        None,
        Some("1200"),
    )
    .expect("filename");
    assert_eq!(name, "GFZ0OPSULT_20242471200_02D_05M_ORB.SP3");

    let url = archive_url(
        AnalysisCenter::GfzUlt,
        ProductType::Sp3,
        date(2024, 9, 3),
        None,
        Some("1200"),
    )
    .expect("url");
    assert_eq!(
        url,
        "https://isdc-data.gfz.de/gnss/products/ultra/w2330/GFZ0OPSULT_20242471200_02D_05M_ORB.SP3.gz"
    );
}

#[test]
fn date_from_gps_week_day_can_drive_product_derivation() {
    let date = ProductDate::from_gps_week_day(2111, 3).expect("week/day date");
    assert_eq!(date, ProductDate::new(2020, 6, 24).expect("date"));

    let name = canonical_filename(AnalysisCenter::Esa, ProductType::Sp3, date, None, None)
        .expect("filename");
    assert_eq!(name, "ESA0MGNFIN_20201760000_01D_05M_ORB.SP3");
}

#[test]
fn pure_issue_and_ionex_candidate_selection_matches_bindings() {
    let target = ProductDateTime::new(date(2024, 9, 3), 13, 0, 0).expect("target");
    let available = [
        UltraIssue::new(date(2024, 9, 3), "0000").expect("issue"),
        UltraIssue::new(date(2024, 9, 3), "0600").expect("issue"),
    ];
    let selected = latest_ops_ultra_sp3(AnalysisCenter::GfzUlt, target, None, Some(&available))
        .expect("latest available product");
    assert_eq!(
        selected.canonical_filename().expect("filename"),
        "GFZ0OPSULT_20242470600_02D_05M_ORB.SP3"
    );

    let candidates =
        gim_date_candidates(AnalysisCenter::CodPrd1, date(2026, 6, 14), 1).expect("candidates");
    assert_eq!(candidates, vec![date(2026, 6, 14), date(2026, 6, 13)]);
}

#[test]
fn skadi_source_entry_and_host_allowlist_are_cataloged() {
    let source = skadi_source_entry();
    assert_eq!(source.protocol, ArchiveProtocol::Https);
    assert_eq!(source.host, "s3.amazonaws.com");
    assert_eq!(source.compression, ArchiveCompression::Gzip);
    assert_eq!(source.compression.as_str(), "gzip");
    assert_eq!(
        source.root_url,
        "https://s3.amazonaws.com/elevation-tiles-prod"
    );
    assert!(allowed_hosts().contains(&"s3.amazonaws.com"));
}

#[test]
fn celestrak_space_weather_catalog_entry_and_paths_are_stable() {
    let source = space_weather_source_entry();
    assert_eq!(source.protocol, ArchiveProtocol::Https);
    assert_eq!(source.host, "celestrak.org");
    assert_eq!(source.compression, ArchiveCompression::None);
    assert_eq!(source.compression.as_str(), "none");
    assert_eq!(source.root_url, "https://celestrak.org/SpaceData");
    assert!(allowed_hosts().contains(&"celestrak.org"));

    assert_eq!(SpaceWeatherProduct::All.code(), "sw_all");
    assert_eq!(
        SpaceWeatherProduct::from_code("sw_last5"),
        Some(SpaceWeatherProduct::Last5Years)
    );
    assert_eq!("sw_all".parse(), Ok(SpaceWeatherProduct::All));
    assert_eq!(
        "bad".parse::<SpaceWeatherProduct>(),
        Err(DataCatalogError::UnknownProductType("bad".to_string()))
    );
    assert_eq!(
        space_weather_filename(SpaceWeatherProduct::All),
        "SW-All.csv"
    );
    assert_eq!(
        space_weather_filename(SpaceWeatherProduct::Last5Years),
        "SW-Last5Years.csv"
    );
    assert_eq!(
        space_weather_archive_url(SpaceWeatherProduct::All),
        "https://celestrak.org/SpaceData/SW-All.csv"
    );
    assert_eq!(
        space_weather_archive_url(SpaceWeatherProduct::Last5Years),
        "https://celestrak.org/SpaceData/SW-Last5Years.csv"
    );
    assert_eq!(
        space_weather_cache_relpath(SpaceWeatherProduct::All),
        "space-weather/SW-All.csv"
    );
    assert_eq!(
        space_weather_cache_relpath(SpaceWeatherProduct::Last5Years),
        "space-weather/SW-Last5Years.csv"
    );
}

#[test]
fn skadi_tile_and_dted_derivation_match_known_tile_ids() {
    assert_eq!(skadi_tile_id(36, -107).expect("tile id"), "N36W107");
    assert_eq!(skadi_band(36).expect("band"), "N36");
    assert_eq!(
        skadi_archive_url(36, -107).expect("url"),
        "https://s3.amazonaws.com/elevation-tiles-prod/skadi/N36/N36W107.hgt.gz"
    );
    assert_eq!(
        dted_tile_filename(36, -107).expect("filename"),
        "n36_w107_1arc_v3.dt2"
    );
    assert_eq!(dted_block_dir(36, -107).expect("block"), "n30_w100");
    assert_eq!(
        dted_cache_relpath(36, -107).expect("relative path"),
        "n30_w100/n36_w107_1arc_v3.dt2"
    );

    assert_eq!(skadi_tile_id(-1, 10).expect("tile id"), "S01E010");
    assert_eq!(skadi_band(-1).expect("band"), "S01");
    assert_eq!(dted_block_dir(-1, 10).expect("block"), "s00_e010");
    assert_eq!(
        skadi_archive_url(-1, 10).expect("url"),
        "https://s3.amazonaws.com/elevation-tiles-prod/skadi/S01/S01E010.hgt.gz"
    );

    assert_eq!(dted_block_dir(32, -117).expect("block"), "n30_w110");
    assert_eq!(dted_block_dir(43, -112).expect("block"), "n40_w110");
    assert_eq!(dted_block_dir(20, -103).expect("block"), "n20_w100");
}

#[test]
fn southern_and_western_hemisphere_tiles_floor_to_sw_corner() {
    // A fractional coordinate names the tile at its floored (south-west) integer
    // corner. DTED block directories use the signed tile index hemisphere and
    // truncate the absolute index magnitude to a ten-degree bucket.

    // Southern latitude, western longitude.
    assert_eq!(
        terrain_tile_index(-32.83, -117.12).expect("index"),
        (-33, -118)
    );
    assert_eq!(skadi_tile_id(-33, -118).expect("tile id"), "S33W118");
    assert_eq!(skadi_band(-33).expect("band"), "S33");
    assert_eq!(
        skadi_archive_url(-33, -118).expect("url"),
        "https://s3.amazonaws.com/elevation-tiles-prod/skadi/S33/S33W118.hgt.gz"
    );
    assert_eq!(
        dted_tile_filename(-33, -118).expect("filename"),
        "s33_w118_1arc_v3.dt2"
    );
    assert_eq!(dted_block_dir(-33, -118).expect("block"), "s30_w110");
    assert_eq!(
        dted_cache_relpath(-33, -118).expect("relpath"),
        "s30_w110/s33_w118_1arc_v3.dt2"
    );

    // Southern latitude, eastern longitude.
    assert_eq!(terrain_tile_index(-33.92, 18.42).expect("index"), (-34, 18));
    assert_eq!(skadi_tile_id(-34, 18).expect("tile id"), "S34E018");
    assert_eq!(dted_block_dir(-34, 18).expect("block"), "s30_e010");
    assert_eq!(
        dted_cache_relpath(-34, 18).expect("relpath"),
        "s30_e010/s34_e018_1arc_v3.dt2"
    );

    // Just south and west of the origin: the floored corner is -1, not 0.
    assert_eq!(terrain_tile_index(-0.5, -0.5).expect("index"), (-1, -1));
    assert_eq!(skadi_tile_id(-1, -1).expect("tile id"), "S01W001");
    assert_eq!(dted_block_dir(-1, -1).expect("block"), "s00_w000");

    // Northern and eastern control: the same flooring rule, no sign flip.
    assert_eq!(terrain_tile_index(45.5, 10.5).expect("index"), (45, 10));
    assert_eq!(skadi_tile_id(45, 10).expect("tile id"), "N45E010");
    assert_eq!(dted_block_dir(45, 10).expect("block"), "n40_e010");
}

#[test]
fn dted_block_dir_truncates_signed_indices_to_observed_buckets() {
    assert_eq!(dted_block_dir(32, -110).expect("block"), "n30_w110");
    assert_eq!(dted_block_dir(32, -111).expect("block"), "n30_w110");
    assert_eq!(dted_block_dir(32, -1).expect("block"), "n30_w000");
    assert_eq!(dted_block_dir(32, -10).expect("block"), "n30_w010");
    assert_eq!(dted_block_dir(1, 1).expect("block"), "n00_e000");
    assert_eq!(dted_block_dir(-1, 1).expect("block"), "s00_e000");
    assert_ne!(
        dted_block_dir(1, 1).expect("north block"),
        dted_block_dir(-1, 1).expect("south block")
    );
}

#[test]
fn dted_block_dir_matches_observed_block_layout_fixture() {
    let mut checked = 0usize;
    for (line_index, line) in include_str!("fixtures/dted/observed_block_layout.txt")
        .lines()
        .enumerate()
    {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut fields = line.split_whitespace();
        let tile_stem = fields.next().expect("observed tile stem");
        let observed_block = fields.next().expect("observed block dir");
        assert!(
            fields.next().is_none(),
            "unexpected field in observed layout line {}",
            line_index + 1
        );

        let (lat_index, lon_index) = parse_observed_dted_tile_stem(tile_stem);
        let derived = dted_block_dir(lat_index, lon_index).expect("derived block dir");
        assert_eq!(
            derived,
            observed_block,
            "observed layout line {}",
            line_index + 1
        );
        checked += 1;
    }
    assert_eq!(checked, 888);
}

fn parse_observed_dted_tile_stem(stem: &str) -> (i32, i32) {
    let mut parts = stem.split('_');
    let lat = parts.next().expect("latitude token");
    let lon = parts.next().expect("longitude token");
    assert_eq!(parts.next(), Some("1arc"));
    assert_eq!(parts.next(), Some("v3"));
    assert_eq!(parts.next(), None);

    (
        parse_signed_index(lat, 'n', 's'),
        parse_signed_index(lon, 'e', 'w'),
    )
}

fn parse_signed_index(token: &str, positive: char, negative: char) -> i32 {
    let mut chars = token.chars();
    let hemi = chars.next().expect("hemisphere");
    let magnitude = chars.as_str().parse::<i32>().expect("index magnitude");
    match hemi {
        h if h == positive => magnitude,
        h if h == negative => -magnitude,
        _ => panic!("unexpected hemisphere token {token}"),
    }
}

#[test]
fn parse_skadi_tile_id_validates_format_and_range() {
    assert_eq!(
        parse_skadi_tile_id("N36W107").expect("parsed tile"),
        (36, -107)
    );
    assert_eq!(
        parse_skadi_tile_id("S01E010").expect("parsed tile"),
        (-1, 10)
    );
    assert_eq!(
        parse_skadi_tile_id("N90E000"),
        Err(DataCatalogError::InvalidTileIndex {
            lat_index: 90,
            lon_index: 0
        })
    );
    assert_eq!(
        parse_skadi_tile_id("S00E010"),
        Err(DataCatalogError::InvalidTileId("S00E010".to_string()))
    );
    assert_eq!(
        parse_skadi_tile_id("n36w107"),
        Err(DataCatalogError::InvalidTileId("n36w107".to_string()))
    );
}

#[test]
fn terrain_tile_index_matches_reader_grid_and_clamps_upper_edges() {
    assert_eq!(
        terrain_tile_index(36.75, -106.25).expect("tile index"),
        (36, -107)
    );
    assert_eq!(
        terrain_tile_index(-0.25, 10.9).expect("tile index"),
        (-1, 10)
    );
    assert_eq!(
        terrain_tile_index(90.0, 180.0).expect("upper edge tile index"),
        (89, 179)
    );
    assert_eq!(
        terrain_tile_index(-90.0, -180.0).expect("lower edge tile index"),
        (-90, -180)
    );
    assert_eq!(
        terrain_tile_index(f64::NAN, -106.5),
        Err(DataCatalogError::InvalidCoordinate {
            lat_deg_bits: f64::NAN.to_bits(),
            lon_deg_bits: (-106.5f64).to_bits()
        })
    );
    assert_eq!(
        skadi_tile_id(90, 0),
        Err(DataCatalogError::InvalidTileIndex {
            lat_index: 90,
            lon_index: 0
        })
    );
}
