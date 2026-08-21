//! Network-free nominal issue scheduling pinned to published IGS and analysis
//! center descriptions.
//!
//! The source URLs, access date, retrieval tool, exact digests, and deterministic
//! resolution of date-only publication statements are committed in
//! `tests/fixtures/data/nominal_issue_schedule_provenance.json`.

use sidereon_core::data::{
    next_issue_due, AnalysisCenter, DataCatalogError, NominalCoverageInterval, ProductDate,
    ProductDateTime, ProductType,
};

fn date(year: i32, month: u8, day: u8) -> ProductDate {
    ProductDate::new(year, month, day).expect("valid test date")
}

fn at(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> ProductDateTime {
    ProductDateTime::new(date(year, month, day), hour, minute, second).expect("valid test instant")
}

#[test]
fn nominal_schedule_provenance_pins_tools_sources_and_digests() {
    let fixture = include_str!("fixtures/data/nominal_issue_schedule_provenance.json");
    let provenance: serde_json::Value = serde_json::from_str(fixture).expect("provenance JSON");
    assert_eq!(provenance["schema_version"], 1);
    assert_eq!(provenance["recorded_at_utc"], "2026-08-21");
    assert_eq!(provenance["retrieval"]["tool"], "curl 8.7.1");
    let sources = provenance["sources"].as_array().expect("source array");
    assert_eq!(sources.len(), 6);
    assert!(sources.iter().all(|source| {
        source["url"]
            .as_str()
            .is_some_and(|url| url.starts_with("https://"))
            && source["sha256"]
                .as_str()
                .is_some_and(|digest| digest.len() == 64)
    }));
}

#[test]
fn ultra_due_boundary_names_the_observed_and_predicted_halves() {
    let before = next_issue_due(
        AnalysisCenter::IgsUlt,
        ProductType::Sp3,
        at(2026, 8, 4, 2, 59, 59),
    )
    .expect("IGS ultra schedule");
    assert_eq!(before.identity.date, date(2026, 8, 3));
    assert_eq!(before.identity.issue.as_deref(), Some("0000"));
    assert_eq!(before.due_at, at(2026, 8, 4, 3, 0, 0));
    assert_eq!(
        before.covers.observed,
        Some(NominalCoverageInterval {
            from: at(2026, 8, 3, 0, 0, 0),
            until: at(2026, 8, 4, 0, 0, 0),
        })
    );
    assert_eq!(
        before.covers.predicted,
        Some(NominalCoverageInterval {
            from: at(2026, 8, 4, 0, 0, 0),
            until: at(2026, 8, 5, 0, 0, 0),
        })
    );

    let after = next_issue_due(
        AnalysisCenter::IgsUlt,
        ProductType::Sp3,
        at(2026, 8, 4, 3, 0, 1),
    )
    .expect("next IGS ultra issue");
    assert_eq!(after.identity.date, date(2026, 8, 3));
    assert_eq!(after.identity.issue.as_deref(), Some("0600"));
    assert_eq!(after.due_at, at(2026, 8, 4, 9, 0, 0));

    let code = next_issue_due(
        AnalysisCenter::CodUlt,
        ProductType::Sp3,
        at(2026, 8, 4, 2, 49, 59),
    )
    .expect("CODE archived ultra schedule");
    assert_eq!(code.identity.issue.as_deref(), Some("0000"));
    assert!(code.covers.observed.is_some());
    assert!(
        code.covers.predicted.is_none(),
        "the cataloged dated CODE issue is its one-day archived product"
    );
}

#[test]
fn rapid_due_boundary_advances_across_the_utc_day() {
    let before = next_issue_due(
        AnalysisCenter::Gfz,
        ProductType::Clk,
        at(2026, 8, 4, 15, 44, 59),
    )
    .expect("GFZ rapid schedule");
    assert_eq!(before.identity.date, date(2026, 8, 3));
    assert_eq!(before.due_at, at(2026, 8, 4, 15, 45, 0));

    let after = next_issue_due(
        AnalysisCenter::Gfz,
        ProductType::Clk,
        at(2026, 8, 4, 15, 45, 1),
    )
    .expect("next GFZ rapid issue");
    assert_eq!(after.identity.date, date(2026, 8, 4));
    assert_eq!(after.due_at, at(2026, 8, 5, 15, 45, 0));
}

#[test]
fn final_due_boundary_advances_over_a_gps_week_rollover() {
    let issue_date = ProductDate::from_gps_week_day(2430, 6).expect("week 2430 Saturday");
    assert_eq!(issue_date, date(2026, 8, 8));

    let before = next_issue_due(
        AnalysisCenter::Igs,
        ProductType::Sp3,
        at(2026, 8, 21, 23, 59, 58),
    )
    .expect("IGS final schedule");
    assert_eq!(before.identity.date, issue_date);
    assert_eq!(before.due_at, at(2026, 8, 21, 23, 59, 59));

    let after = next_issue_due(
        AnalysisCenter::Igs,
        ProductType::Sp3,
        at(2026, 8, 22, 0, 0, 0),
    )
    .expect("next IGS final batch");
    assert_eq!(after.identity.date, date(2026, 8, 15));
    assert_eq!(after.identity.date.gps_week().unwrap(), 2431);
    assert_eq!(after.due_at, at(2026, 8, 28, 23, 59, 59));
}

#[test]
fn predicted_ionex_due_time_tracks_each_catalog_horizon() {
    let now = at(2026, 8, 3, 23, 59, 59);
    let p1 = next_issue_due(AnalysisCenter::CodPrd1, ProductType::Ionex, now)
        .expect("one-day prediction schedule");
    let p2 = next_issue_due(AnalysisCenter::CodPrd2, ProductType::Ionex, now)
        .expect("two-day prediction schedule");
    assert_eq!(p1.due_at, at(2026, 8, 4, 0, 0, 0));
    assert_eq!(p2.due_at, at(2026, 8, 4, 0, 0, 0));
    assert_eq!(p1.identity.date, date(2026, 8, 5));
    assert_eq!(p2.identity.date, date(2026, 8, 6));
    assert!(p1.covers.observed.is_none());
    assert!(p1.covers.predicted.is_some());
    assert!(p2.covers.observed.is_none());
    assert!(p2.covers.predicted.is_some());
}

#[test]
fn every_requested_catalog_line_has_a_network_free_due_query() {
    let now = at(2026, 8, 4, 7, 8, 0);
    let lines = [
        (AnalysisCenter::Igs, ProductType::Sp3),
        (AnalysisCenter::Esa, ProductType::Sp3),
        (AnalysisCenter::Esa, ProductType::Clk),
        (AnalysisCenter::Esa, ProductType::Ionex),
        (AnalysisCenter::Cod, ProductType::Sp3),
        (AnalysisCenter::Cod, ProductType::Clk),
        (AnalysisCenter::Cod, ProductType::Ionex),
        (AnalysisCenter::Gfz, ProductType::Sp3),
        (AnalysisCenter::Gfz, ProductType::Clk),
        (AnalysisCenter::IgsUlt, ProductType::Sp3),
        (AnalysisCenter::CodUlt, ProductType::Sp3),
        (AnalysisCenter::EsaUlt, ProductType::Sp3),
        (AnalysisCenter::GfzUlt, ProductType::Sp3),
        (AnalysisCenter::CodRap, ProductType::Ionex),
        (AnalysisCenter::CodPrd1, ProductType::Ionex),
        (AnalysisCenter::CodPrd2, ProductType::Ionex),
    ];
    for (center, product_type) in lines {
        let issue = next_issue_due(center, product_type, now)
            .unwrap_or_else(|error| panic!("{center}/{product_type}: {error}"));
        assert!(issue.due_at >= now, "{center}/{product_type}");
        issue.identity.validate().expect("catalog identity");
    }

    assert!(matches!(
        next_issue_due(AnalysisCenter::WumNrt, ProductType::Sp3, now),
        Err(DataCatalogError::UnsupportedNominalSchedule {
            center: AnalysisCenter::WumNrt,
            product_type: ProductType::Sp3,
        })
    ));
}
