//! Scoreboard validation tests.
//!
//! Provenance: fixture layout follows the public SP3-c examples already carried
//! in this repository. The synthetic arc is generated from the public two-body
//! propagator and public frame transform APIs to validate the complete harness
//! path without network access.

use serde_json::Value;
use sidereon_core::astro::frames::EarthOrientation;
use sidereon_core::astro::math::least_squares::SolveOptions;
use sidereon_core::astro::propagator::{
    ForceModelKind, IntegratorKind, IntegratorOptions, StatePropagator,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::civil::{
    civil_from_j2000_seconds, j2000_seconds, split_julian_date,
    split_julian_date_from_j2000_seconds, MJD_JD_OFFSET,
};
use sidereon_core::astro::time::gnss::{seconds_of_week_from_calendar, week_from_calendar};
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::constants::{J2000_JD, SECONDS_PER_DAY};
use sidereon_core::data::{AnalysisCenter, ProductDate, ProductDateTime, ProductType};
use sidereon_core::ephemeris::{ExactSp3ValidationError, Sp3};
use sidereon_core::{
    EarthOrientationProvider, GnssSatelliteId, GnssSystem, TdbEarthOrientationProvider,
};
use sidereon_scoreboard::{
    parse_product_date, publication_status, resolve_latest_available_rapid_sp3, run_with_fetcher,
    score_sp3_bytes, FetchOutcome, HttpsFetcher, ListingFetcher, ListingOutcome, ProductCandidate,
    ProductFetcher, PublicationStatusOutcome, ScoreOptions, ScoreboardError, ScoreboardStatus,
};

const SP3_POSITION_3D_QUANTIZATION_BOUND_M: f64 = 8.660_254_037_844_386e-4;

fn date(year: i32, month: u8, day: u8) -> ProductDate {
    ProductDate::new(year, month, day).expect("valid date")
}

#[test]
fn fixture_schema_is_exact_and_counts_add_up() {
    let bytes = include_bytes!("fixtures/minimal_sp3.sp3");
    let report = score_sp3_bytes(
        bytes,
        "fixture.sp3",
        date(2020, 6, 24),
        &ScoreOptions::default(),
    )
    .expect("fixture scores");
    let value = serde_json::to_value(&report).expect("report JSON");
    assert_keys(
        &value,
        &[
            "attempted_candidates",
            "date_utc",
            "notes",
            "per_constellation",
            "per_sat",
            "product",
            "sidereon_version",
            "status",
        ],
    );
    assert_eq!(value["status"], "scored");
    assert_keys(
        &value["product"],
        &["agency", "name", "parser_skipped_records"],
    );
    assert_keys(&value["per_sat"], &["bottom", "skipped", "top"]);
    let gps = &value["per_constellation"]["GPS"];
    assert_keys(
        gps,
        &[
            "fit_count",
            "median_rms_3d_m",
            "sat_count",
            "skipped",
            "worst_rms_3d_m",
        ],
    );
    let sat_count = gps["sat_count"].as_u64().unwrap();
    let fit_count = gps["fit_count"].as_u64().unwrap();
    let skipped = gps["skipped"].as_u64().unwrap();
    assert_eq!(fit_count + skipped, sat_count);
    assert_eq!(report.per_sat.skipped.len(), 1);
    assert_eq!(
        report
            .product
            .as_ref()
            .expect("scored report has product")
            .parser_skipped_records,
        0
    );
    let skipped = serde_json::to_value(&report.per_sat.skipped[0]).expect("skip row JSON");
    assert_keys(&skipped, &["constellation", "reason", "satellite"]);
}

#[test]
fn parser_skipped_records_are_visible() {
    let bytes = include_str!("fixtures/minimal_sp3.sp3")
        .replace("+    1   G01  0", "+    2   G01R28  0")
        .into_bytes();
    let report = score_sp3_bytes(
        &bytes,
        "fixture-with-unsupported.sp3",
        date(2020, 6, 24),
        &ScoreOptions::default(),
    )
    .expect("fixture scores");

    assert_eq!(
        report
            .product
            .as_ref()
            .expect("scored report has product")
            .parser_skipped_records,
        1
    );
    assert!(report
        .notes
        .iter()
        .any(|note| note.contains("product.parser_skipped_records")));
}

#[test]
fn mocked_fetch_resolves_without_network() {
    struct MockFetcher;

    impl ProductFetcher for MockFetcher {
        fn fetch(
            &self,
            candidate: &ProductCandidate,
        ) -> Result<FetchOutcome, sidereon_scoreboard::ScoreboardError> {
            if candidate.name.contains("20261950000") {
                Ok(FetchOutcome::Available(exact_candidate_sp3(candidate)))
            } else {
                Ok(FetchOutcome::NotPosted { http_status: None })
            }
        }
    }

    let resolution = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 1, &MockFetcher)
        .expect("previous day resolves");
    let resolved = resolution.resolved.expect("resolved product");
    assert!(resolved.candidate.name.contains("20261950000"));
    assert_eq!(resolved.bytes, exact_candidate_sp3(&resolved.candidate));
    assert!(resolution
        .attempted
        .iter()
        .any(|candidate| candidate.name == "IGS0OPSRAP_20261950000_01D_15M_ORB.SP3"));
}

#[test]
fn pretransition_candidate_dates_are_rejected_before_fetch() {
    struct MustNotFetch;

    impl ProductFetcher for MustNotFetch {
        fn fetch(&self, _candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            panic!("unsupported historical candidate must not be emitted")
        }
    }

    let error = resolve_latest_available_rapid_sp3(date(2020, 6, 25), 1, &MustNotFetch)
        .expect_err("historical rapid/ultra naming was not modeled");
    assert!(matches!(
        error,
        ScoreboardError::UnsupportedCandidateEra {
            gps_week: 2111,
            minimum_gps_week: 2238,
            ..
        }
    ));
}

#[test]
fn transition_week_target_skips_unsupported_lookback_dates() {
    struct Missing;

    impl ProductFetcher for Missing {
        fn fetch(&self, _candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            Ok(FetchOutcome::NotPosted {
                http_status: Some(404),
            })
        }
    }

    let transition_date = date(2022, 11, 27);
    let resolution = resolve_latest_available_rapid_sp3(transition_date, 4, &Missing)
        .expect("the first supported long-name date remains usable");

    assert!(resolution.resolved.is_none());
    assert_eq!(resolution.attempted.len(), 19);
    assert!(resolution
        .attempted
        .iter()
        .all(|candidate| candidate.date == transition_date));
}

#[test]
fn source_network_failure_falls_back_to_an_independent_archive() {
    struct SourceFallbackFetcher {
        bkg_attempts: std::cell::Cell<usize>,
    }

    impl ProductFetcher for SourceFallbackFetcher {
        fn fetch(
            &self,
            candidate: &ProductCandidate,
        ) -> Result<FetchOutcome, sidereon_scoreboard::ScoreboardError> {
            if candidate.source == "BKG IGS" {
                self.bkg_attempts.set(self.bkg_attempts.get() + 1);
                return Err(sidereon_scoreboard::ScoreboardError::Network {
                    archive_source: candidate.source.to_string(),
                    name: candidate.name.clone(),
                    url: candidate.url.clone(),
                    message: "simulated connection failure".to_string(),
                });
            }
            if candidate.source == "ESA" {
                return Ok(FetchOutcome::Available(exact_candidate_sp3(candidate)));
            }
            panic!("resolver should stop after the independent fallback succeeds");
        }
    }

    let fetcher = SourceFallbackFetcher {
        bkg_attempts: std::cell::Cell::new(0),
    };
    let resolution = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 4, &fetcher)
        .expect("one source outage does not abort the scoreboard");

    assert!(resolution.resolved.is_some());
    assert_eq!(fetcher.bkg_attempts.get(), 1);
    assert_eq!(resolution.attempted.len(), 2);
    assert_eq!(resolution.attempted[0].source, "BKG IGS");
    assert!(resolution.attempted_errors[0]
        .as_deref()
        .is_some_and(|error| error.contains("simulated connection failure")));
    assert_eq!(resolution.attempted[1].source, "ESA");
    assert!(resolution.attempted_errors[1].is_none());
}

#[test]
fn missing_candidate_urls_are_no_data_report() {
    struct MissingFetcher;

    impl ProductFetcher for MissingFetcher {
        fn fetch(
            &self,
            _candidate: &ProductCandidate,
        ) -> Result<FetchOutcome, sidereon_scoreboard::ScoreboardError> {
            Ok(FetchOutcome::NotPosted {
                http_status: Some(404),
            })
        }
    }

    let report = run_with_fetcher(date(2026, 7, 5), 0, &MissingFetcher).expect("no-data report");
    assert_eq!(report.status, ScoreboardStatus::NoData);
    assert!(report.product.is_none());
    assert_eq!(report.attempted_candidates.len(), 19);
    assert!(report.per_constellation.is_empty());
    assert!(report.per_sat.top.is_empty());
    assert!(report
        .attempted_candidates
        .iter()
        .any(|candidate| candidate.url.contains("/products/2426/")));
    assert!(report
        .attempted_candidates
        .iter()
        .all(|candidate| candidate.http_status == Some(404)));
    assert!(report
        .notes
        .iter()
        .any(|note| note.contains("attempted URL")));
    let attempted =
        serde_json::to_value(&report.attempted_candidates[0]).expect("attempted candidate JSON");
    assert_keys(
        &attempted,
        &[
            "cadence",
            "date_utc",
            "http_status",
            "name",
            "source",
            "url",
        ],
    );
}

#[test]
fn candidate_urls_use_product_dates_gps_week() {
    struct CaptureFetcher;

    impl ProductFetcher for CaptureFetcher {
        fn fetch(
            &self,
            candidate: &ProductCandidate,
        ) -> Result<FetchOutcome, sidereon_scoreboard::ScoreboardError> {
            if candidate.name == "IGS0OPSRAP_20261850000_01D_15M_ORB.SP3" {
                Ok(FetchOutcome::Available(exact_candidate_sp3(candidate)))
            } else {
                Ok(FetchOutcome::NotPosted { http_status: None })
            }
        }
    }

    let resolution = resolve_latest_available_rapid_sp3(date(2026, 7, 5), 1, &CaptureFetcher)
        .expect("previous GPS week resolves");
    let resolved = resolution.resolved.expect("resolved product");
    assert_eq!(
        resolved.candidate.url,
        "https://igs.bkg.bund.de/root_ftp/IGS/products/2425/IGS0OPSRAP_20261850000_01D_15M_ORB.SP3.gz"
    );
}

#[test]
fn ordinary_not_posted_candidate_falls_back_to_next_candidate() {
    struct AbsentThenValid {
        calls: std::cell::Cell<usize>,
    }

    impl ProductFetcher for AbsentThenValid {
        fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            match call {
                0 => Ok(FetchOutcome::NotPosted {
                    http_status: Some(404),
                }),
                1 => Ok(FetchOutcome::Available(exact_candidate_sp3(candidate))),
                _ => panic!("resolver continued after a valid candidate"),
            }
        }
    }

    let fetcher = AbsentThenValid {
        calls: std::cell::Cell::new(0),
    };
    let resolution = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &fetcher)
        .expect("ordinary absence permits the documented next candidate");

    assert_eq!(fetcher.calls.get(), 2);
    assert_eq!(resolution.attempted_http_statuses, vec![Some(404), None]);
    assert_eq!(
        resolution
            .resolved
            .expect("second candidate resolves")
            .candidate
            .name,
        "IGS0OPSULT_20261961800_02D_15M_ORB.SP3"
    );
}

#[test]
fn exact_product_integrity_failures_are_terminal() {
    for invalid in [
        InvalidCandidateBytes::Malformed,
        InvalidCandidateBytes::ParseInvalid,
        InvalidCandidateBytes::Cadence,
        InvalidCandidateBytes::Span,
        InvalidCandidateBytes::Start,
        InvalidCandidateBytes::AgencySubstitution,
    ] {
        let fetcher = InvalidThenPanicFetcher {
            invalid,
            calls: std::cell::Cell::new(0),
        };
        let error = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &fetcher)
            .expect_err("integrity failure must not try a later valid candidate");
        assert_eq!(fetcher.calls.get(), 1, "case {invalid:?}");
        match (invalid, error) {
            (
                InvalidCandidateBytes::Malformed | InvalidCandidateBytes::ParseInvalid,
                ScoreboardError::ExactSp3Integrity {
                    error: ExactSp3ValidationError::Parse(_),
                    ..
                },
            ) => {}
            (
                InvalidCandidateBytes::Cadence,
                ScoreboardError::ExactSp3Integrity {
                    error: ExactSp3ValidationError::CadenceMismatch { .. },
                    ..
                },
            ) => {}
            (
                InvalidCandidateBytes::Span,
                ScoreboardError::ExactSp3Integrity {
                    error: ExactSp3ValidationError::SpanMismatch { .. },
                    ..
                },
            ) => {}
            (
                InvalidCandidateBytes::Start,
                ScoreboardError::ExactSp3Integrity {
                    error: ExactSp3ValidationError::DeclaredStartMismatch { .. },
                    ..
                },
            ) => {}
            (
                InvalidCandidateBytes::AgencySubstitution,
                ScoreboardError::ExactSp3Integrity {
                    error: ExactSp3ValidationError::AgencyMismatch { expected, actual },
                    ..
                },
            ) => {
                assert_eq!(expected, "IGS");
                assert_eq!(actual, "ESOC");
            }
            (_, other) => panic!("unexpected error for {invalid:?}: {other:?}"),
        }
    }
}

#[test]
fn absence_followed_by_integrity_failure_returns_the_integrity_failure() {
    struct AbsentThenInvalid {
        calls: std::cell::Cell<usize>,
    }

    impl ProductFetcher for AbsentThenInvalid {
        fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            match call {
                0 => Ok(FetchOutcome::NotPosted {
                    http_status: Some(410),
                }),
                1 => Ok(FetchOutcome::Available(exact_candidate_sp3_custom(
                    candidate, 0, -1, None,
                ))),
                _ => panic!("resolver continued after an integrity failure"),
            }
        }
    }

    let fetcher = AbsentThenInvalid {
        calls: std::cell::Cell::new(0),
    };
    let error = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &fetcher)
        .expect_err("integrity failure after absence remains terminal");

    assert_eq!(fetcher.calls.get(), 2);
    assert!(matches!(
        error,
        ScoreboardError::ExactSp3Integrity {
            error: ExactSp3ValidationError::SpanMismatch { .. },
            ..
        }
    ));
}

#[test]
fn digest_failure_is_typed_and_terminal() {
    struct DigestFailure {
        calls: std::cell::Cell<usize>,
    }

    impl ProductFetcher for DigestFailure {
        fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call != 0 {
                panic!("resolver continued after a digest failure");
            }
            Err(ScoreboardError::DigestMismatch {
                archive_source: candidate.source.to_string(),
                name: candidate.name.clone(),
                expected: "expected-digest".to_string(),
                actual: "actual-digest".to_string(),
            })
        }
    }

    let fetcher = DigestFailure {
        calls: std::cell::Cell::new(0),
    };
    let error = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &fetcher)
        .expect_err("digest mismatch must remain terminal");

    assert_eq!(fetcher.calls.get(), 1);
    assert!(matches!(error, ScoreboardError::DigestMismatch { .. }));
}

#[test]
fn first_nonabsence_fetch_error_is_not_reported_as_no_data() {
    struct OneNetworkFailureThenAbsent;

    impl ProductFetcher for OneNetworkFailureThenAbsent {
        fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            if candidate.source == "BKG IGS" {
                Err(ScoreboardError::Network {
                    archive_source: candidate.source.to_string(),
                    name: candidate.name.clone(),
                    url: candidate.url.clone(),
                    message: "first transport failure".to_string(),
                })
            } else {
                Ok(FetchOutcome::NotPosted {
                    http_status: Some(404),
                })
            }
        }
    }

    let error =
        resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &OneNetworkFailureThenAbsent)
            .expect_err("a transport failure cannot collapse into publication absence");

    match error {
        ScoreboardError::Network { message, .. } => {
            assert_eq!(message, "first transport failure");
        }
        other => panic!("expected the first transport failure, got {other:?}"),
    }
}

#[test]
fn nonabsence_status_cannot_be_injected_as_not_posted() {
    struct InvalidAbsenceStatus {
        calls: std::cell::Cell<usize>,
    }

    impl ProductFetcher for InvalidAbsenceStatus {
        fn fetch(&self, _candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            self.calls.set(self.calls.get() + 1);
            Ok(FetchOutcome::NotPosted {
                http_status: Some(403),
            })
        }
    }

    let fetcher = InvalidAbsenceStatus {
        calls: std::cell::Cell::new(0),
    };
    let error = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &fetcher)
        .expect_err("authorization failure is not publication absence");

    assert_eq!(fetcher.calls.get(), 1);
    assert!(matches!(
        error,
        ScoreboardError::HttpStatus { status: 403, .. }
    ));
}

#[test]
fn fetcher_authorization_error_is_terminal_without_later_fetch() {
    struct AuthorizationFailure {
        calls: std::cell::Cell<usize>,
    }

    impl ProductFetcher for AuthorizationFailure {
        fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call != 0 {
                panic!("resolver continued after an authorization failure");
            }
            Err(ScoreboardError::HttpStatus {
                archive_source: candidate.source.to_string(),
                name: candidate.name.clone(),
                url: candidate.url.clone(),
                status: 403,
            })
        }
    }

    let fetcher = AuthorizationFailure {
        calls: std::cell::Cell::new(0),
    };
    let error = resolve_latest_available_rapid_sp3(date(2026, 7, 15), 0, &fetcher)
        .expect_err("authorization failure is terminal");

    assert_eq!(fetcher.calls.get(), 1);
    assert!(matches!(
        error,
        ScoreboardError::HttpStatus { status: 403, .. }
    ));
}

#[test]
#[ignore = "network test for current public SP3 archives"]
fn live_current_product_candidate_resolves() {
    let target = sidereon_scoreboard::utc_today().expect("UTC date");
    let resolution = resolve_latest_available_rapid_sp3(target, 4, &HttpsFetcher)
        .expect("live resolver does not fail");
    assert!(
        resolution.resolved.is_some(),
        "no posted product in {} attempts: {:#?}",
        resolution.attempted.len(),
        resolution
            .attempted
            .iter()
            .map(|candidate| &candidate.url)
            .collect::<Vec<_>>()
    );
}

#[test]
fn synthetic_state_arc_runs_full_path_to_near_zero_rms() {
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=8).map(|step| start + step * 60).collect();
    let initial = CartesianState::new(start as f64, [7078.0, -30.0, 820.0], [0.20, 7.35, 1.05]);
    let sp3 = synthetic_sp3(initial, &epochs);
    let mut options = ScoreOptions::default();
    options.fit_options.force_model = ForceModelKind::two_body();
    options.fit_options.integrator = IntegratorKind::Dp54;
    options.fit_options.integrator_options = IntegratorOptions {
        abs_tol: 1.0e-12,
        rel_tol: 1.0e-13,
        initial_step: 10.0,
        max_step: 60.0,
        ..IntegratorOptions::default()
    };
    options.fit_options.solver_options = SolveOptions {
        gtol: 1.0e-15,
        ftol: 1.0e-15,
        xtol: 1.0e-15,
        max_nfev: 1200,
    };

    let report = score_sp3_bytes(sp3.as_bytes(), "synthetic.sp3", date(2026, 6, 1), &options)
        .expect("synthetic arc scores");
    let parsed = Sp3::parse(sp3.as_bytes()).expect("synthetic SP3 parses");
    assert_eq!(parsed.precise_ephemeris_state_samples().len(), epochs.len());
    assert!(!report
        .notes
        .iter()
        .any(|note| note.contains("position-sample fitter")));
    let top = report.per_sat.top.first().expect("fit row");
    let top_json = serde_json::to_value(top).expect("top row JSON");
    assert_keys(&top_json, FIT_ROW_KEYS);
    let bottom_json = serde_json::to_value(report.per_sat.bottom.first().expect("bottom row"))
        .expect("bottom row JSON");
    assert_keys(&bottom_json, FIT_ROW_KEYS);
    assert!(
        top.rms_3d_m < SP3_POSITION_3D_QUANTIZATION_BOUND_M,
        "synthetic RMS was {:.17e} m",
        top.rms_3d_m
    );
    assert!(report.per_sat.skipped.is_empty());
}

#[test]
fn partial_velocity_arc_is_reported_as_skip() {
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=8).map(|step| start + step * 60).collect();
    let initial = CartesianState::new(start as f64, [7078.0, -30.0, 820.0], [0.20, 7.35, 1.05]);
    let sp3 = blank_first_velocity_record(&synthetic_sp3(initial, &epochs));
    let mut options = ScoreOptions::default();
    options.fit_options.force_model = ForceModelKind::two_body();

    let report = score_sp3_bytes(
        sp3.as_bytes(),
        "partial-velocity.sp3",
        date(2026, 6, 1),
        &options,
    )
    .expect("partial velocity arc scores");

    assert!(report.per_sat.top.is_empty());
    assert_eq!(report.per_sat.skipped.len(), 1);
    assert_eq!(
        report.per_sat.skipped[0].reason,
        "partial_velocity_samples:8/9"
    );
    let gps = report.per_constellation.get("GPS").expect("GPS report");
    assert_eq!(gps.sat_count, 1);
    assert_eq!(gps.fit_count, 0);
    assert_eq!(gps.skipped, 1);
}

#[test]
fn date_parser_rejects_extra_fields() {
    assert!(parse_product_date("2026-07-04-extra").is_err());
}

fn assert_keys(value: &Value, expected: &[&str]) {
    let object = value.as_object().expect("JSON object");
    let mut keys = object.keys().map(String::as_str).collect::<Vec<_>>();
    keys.sort_unstable();
    assert_eq!(keys, expected);
}

const FIT_ROW_KEYS: &[&str] = &[
    "along_rms_m",
    "constellation",
    "cross_rms_m",
    "low_sample_count",
    "n",
    "radial_rms_m",
    "rms_3d_m",
    "satellite",
];

#[derive(Debug, Clone, Copy)]
enum InvalidCandidateBytes {
    Malformed,
    ParseInvalid,
    Cadence,
    Span,
    Start,
    AgencySubstitution,
}

struct InvalidThenPanicFetcher {
    invalid: InvalidCandidateBytes,
    calls: std::cell::Cell<usize>,
}

impl ProductFetcher for InvalidThenPanicFetcher {
    fn fetch(&self, candidate: &ProductCandidate) -> Result<FetchOutcome, ScoreboardError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call != 0 {
            panic!("resolver continued after an integrity failure");
        }
        let bytes = match self.invalid {
            InvalidCandidateBytes::Malformed => b"<html>not an SP3 product</html>".to_vec(),
            InvalidCandidateBytes::ParseInvalid => b"#dP truncated header\nEOF\n".to_vec(),
            InvalidCandidateBytes::Cadence => {
                exact_candidate_sp3_custom(candidate, 0, 0, Some(60.0))
            }
            InvalidCandidateBytes::Span => exact_candidate_sp3_custom(candidate, 0, -1, None),
            InvalidCandidateBytes::Start => exact_candidate_sp3_custom(candidate, 300, 0, None),
            InvalidCandidateBytes::AgencySubstitution => {
                exact_candidate_sp3_custom_agency(candidate, 0, 0, None, Some("ESOC"))
            }
        };
        Ok(FetchOutcome::Available(bytes))
    }
}

fn exact_candidate_sp3(candidate: &ProductCandidate) -> Vec<u8> {
    exact_candidate_sp3_custom(candidate, 0, 0, None)
}

fn exact_candidate_sp3_custom(
    candidate: &ProductCandidate,
    start_offset_s: i64,
    count_adjustment: isize,
    header_cadence_override_s: Option<f64>,
) -> Vec<u8> {
    exact_candidate_sp3_custom_agency(
        candidate,
        start_offset_s,
        count_adjustment,
        header_cadence_override_s,
        None,
    )
}

fn exact_candidate_sp3_custom_agency(
    candidate: &ProductCandidate,
    start_offset_s: i64,
    count_adjustment: isize,
    header_cadence_override_s: Option<f64>,
    agency_override: Option<&str>,
) -> Vec<u8> {
    let fields = candidate.name.split('_').collect::<Vec<_>>();
    assert_eq!(fields.len(), 5, "candidate must use a long product name");
    let date_issue = fields[1];
    assert_eq!(date_issue.len(), 11);
    let hour = date_issue[7..9].parse::<i32>().expect("issue hour");
    let minute = date_issue[9..11].parse::<i32>().expect("issue minute");
    let span_s = duration_token_seconds(fields[2]);
    let cadence_s = duration_token_seconds(fields[3]);
    let epoch_count = isize::try_from(span_s / cadence_s)
        .expect("fixture epoch count")
        .checked_add(count_adjustment)
        .and_then(|count| usize::try_from(count).ok())
        .expect("positive fixture epoch count");
    let start = j2000_seconds(
        candidate.date.year,
        i32::from(candidate.date.month),
        i32::from(candidate.date.day),
        hour,
        minute,
        0.0,
    ) as i64
        + start_offset_s;
    let (year, month, day, start_hour, start_minute, start_second) =
        civil_from_j2000_seconds(start);
    let gps_week = week_from_calendar(TimeScale::Gpst, year, month, day)
        .expect("post-transition fixture GPS week");
    let seconds_of_week =
        seconds_of_week_from_calendar(year, month, day, start_hour, start_minute, start_second);
    let (jd_whole, mjd_fraction) = split_julian_date(
        i32::try_from(year).expect("fixture year"),
        i32::try_from(month).expect("fixture month"),
        i32::try_from(day).expect("fixture day"),
        i32::try_from(start_hour).expect("fixture hour"),
        i32::try_from(start_minute).expect("fixture minute"),
        start_second as f64,
    );
    let mjd = u32::try_from((jd_whole - MJD_JD_OFFSET) as i64).expect("fixture MJD");
    let expected_agency = match &candidate.name[..3] {
        "IGS" => "IGS",
        "ESA" => "ESOC",
        "GFZ" => "GFZ",
        other => panic!("unsupported fixture producer {other}"),
    };
    let agency = agency_override.unwrap_or(expected_agency);
    let mut text = format!(
        "#dP{} {epoch_count:>7} {:<5}{:>6}{:>4} {}\n",
        format_calendar(
            year,
            month,
            day,
            start_hour,
            start_minute,
            start_second as f64
        ),
        "ORBIT",
        "IGS20",
        "FIT",
        agency
    );
    text.push_str(&format!(
        "## {:>4} {:15.8} {:14.8} {:>5} {:.13}\n",
        gps_week,
        seconds_of_week,
        header_cadence_override_s.unwrap_or(cadence_s as f64),
        mjd,
        mjd_fraction
    ));
    text.push_str("+    1   G01");
    for _ in 1..17 {
        text.push_str("  0");
    }
    text.push('\n');
    for _ in 1..5 {
        text.push_str("+        ");
        for _ in 0..17 {
            text.push_str("  0");
        }
        text.push('\n');
    }
    for _ in 0..5 {
        text.push_str("++       ");
        for _ in 0..17 {
            text.push_str("  0");
        }
        text.push('\n');
    }
    text.push_str("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    text.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    text.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
    text.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
    text.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    text.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    for _ in 0..4 {
        text.push_str("/* SCOREBOARD EXACT PRODUCT TEST FIXTURE\n");
    }
    for index in 0..epoch_count {
        let epoch = start + i64::try_from(index).expect("epoch index") * cadence_s;
        let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(epoch);
        text.push_str(&format!(
            "*  {}\n",
            format_calendar(year, month, day, hour, minute, second as f64)
        ));
        text.push_str("PG01  15000.000000 -20000.000000   5000.000000    123.456789\n");
    }
    text.push_str("EOF\n");
    text.into_bytes()
}

fn duration_token_seconds(token: &str) -> i64 {
    let amount = token[..2].parse::<i64>().expect("duration amount");
    let unit = match token.as_bytes()[2] {
        b'M' => 60,
        b'H' => 3_600,
        b'D' => 86_400,
        other => panic!("unsupported fixture duration unit {other}"),
    };
    amount * unit
}

fn synthetic_sp3(initial: CartesianState, epochs_j2000_s: &[i64]) -> String {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite");
    let propagator = StatePropagator {
        initial,
        force_model: ForceModelKind::two_body(),
        integrator: IntegratorKind::Dp54,
        options: IntegratorOptions {
            abs_tol: 1.0e-12,
            rel_tol: 1.0e-13,
            initial_step: 10.0,
            max_step: 60.0,
            ..IntegratorOptions::default()
        },
        drag: None,
        space_weather: None,
    };
    let query_epochs = epochs_j2000_s
        .iter()
        .map(|&epoch| epoch as f64)
        .collect::<Vec<_>>();
    let states = propagator.ephemeris(&query_epochs).expect("truth arc");
    let provider = TdbEarthOrientationProvider::new();
    let mut out = String::new();
    out.push_str(&format!(
        "#cV{} {:>7} ORBIT IGS14 FIT  TST\n",
        format_calendar(2026, 6, 1, 0, 0, 0.0),
        epochs_j2000_s.len()
    ));
    out.push_str("## 2421  86400.00000000    60.00000000 61192 0.0000000000000\n");
    out.push_str("+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
    out.push_str("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
    out.push_str("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    out.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    out.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
    out.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
    out.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    out.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    for (state, &epoch) in states.iter().zip(epochs_j2000_s) {
        let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(epoch);
        out.push_str(&format!(
            "*  {}\n",
            format_calendar(year, month, day, hour, minute, second as f64)
        ));
        let instant = instant_at(TimeScale::Gpst, epoch);
        let seed = EarthOrientation::from_instant(instant).expect("seed orientation");
        let tdb_seconds = (seed.time_scales().jd_tdb - J2000_JD) * SECONDS_PER_DAY;
        let orientation = provider
            .orientation_at_tdb_seconds(tdb_seconds)
            .expect("orientation");
        let (position_itrf_km, velocity_itrf_km_s) = orientation
            .gcrf_to_itrf_state_km(state.position_array(), state.velocity_array())
            .expect("state transform");
        out.push_str(&format!(
            "P{sat}{:14.6}{:14.6}{:14.6}{:14.6}\n",
            position_itrf_km[0], position_itrf_km[1], position_itrf_km[2], 0.0
        ));
        out.push_str(&format!(
            "V{sat}{:14.6}{:14.6}{:14.6}{:14.6}\n",
            velocity_itrf_km_s[0] * 10_000.0,
            velocity_itrf_km_s[1] * 10_000.0,
            velocity_itrf_km_s[2] * 10_000.0,
            0.0
        ));
    }
    out.push_str("EOF\n");
    out
}

fn blank_first_velocity_record(sp3: &str) -> String {
    let mut replaced = false;
    let lines = sp3
        .lines()
        .map(|line| {
            if !replaced && line.starts_with("VG01") {
                replaced = true;
                format!("VG01{:14.6}{:14.6}{:14.6}{:14.6}", 0.0, 0.0, 0.0, 0.0)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>();
    format!("{}\n", lines.join("\n"))
}

fn instant_at(scale: TimeScale, epoch_j2000_s: i64) -> Instant {
    let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(epoch_j2000_s);
    Instant::from_julian_date(
        scale,
        JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date"),
    )
}

fn format_calendar(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    seconds: f64,
) -> String {
    format!("{year:4} {month:>2} {day:>2} {hour:>2} {minute:>2} {seconds:11.8}")
}

fn core_listing_fixture(name: &str) -> String {
    let path = format!(
        "{}/../sidereon-core/tests/fixtures/listings/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

/// Recorded 2026-08-04 GFZ ultra scenario: one bounded query answers the
/// newest published issue and its lag behind nominal without fetching any
/// product bytes.
#[test]
fn publication_status_answers_the_recorded_gfz_lag_scenario() {
    struct RecordedGfz {
        fetched: std::cell::RefCell<Vec<String>>,
    }
    impl ListingFetcher for RecordedGfz {
        fn fetch_listing(&self, url: &str) -> Result<ListingOutcome, ScoreboardError> {
            self.fetched.borrow_mut().push(url.to_string());
            assert_eq!(url, "https://isdc-data.gfz.de/gnss/products/ultra/w2430/");
            Ok(ListingOutcome::Available(core_listing_fixture(
                "gfz-ultra-w2430-20260804.html",
            )))
        }
    }

    let fetcher = RecordedGfz {
        fetched: std::cell::RefCell::new(Vec::new()),
    };
    let now = ProductDateTime::new(date(2026, 8, 4), 7, 8, 0).expect("query instant");
    let outcome = publication_status(AnalysisCenter::GfzUlt, ProductType::Sp3, now, &fetcher)
        .expect("supported line");

    match outcome {
        PublicationStatusOutcome::Published {
            product,
            listing_url,
            behind_nominal_minutes,
        } => {
            assert_eq!(product.date, date(2026, 8, 3));
            assert_eq!(product.issue, "0300");
            assert_eq!(product.filename, "GFZ0OPSULT_20262150300_02D_05M_ORB.SP3");
            assert_eq!(product.observed_at.as_deref(), Some("2026-08-04 08:20"));
            assert_eq!(
                listing_url,
                "https://isdc-data.gfz.de/gnss/products/ultra/w2430/"
            );
            assert_eq!(behind_nominal_minutes, 28 * 60 + 8);
        }
        other => panic!("expected Published, got {other:?}"),
    }
    assert_eq!(
        fetcher.fetched.borrow().len(),
        1,
        "one listing GET answered the query"
    );
}

/// Recorded 2026-08-04 BKG state: the current week's directory does not
/// exist (authoritative 404), and the bounded walk-back answers from the
/// previous week's directory.
#[test]
fn publication_status_walks_back_when_the_current_week_directory_is_absent() {
    struct RecordedBkg;
    impl ListingFetcher for RecordedBkg {
        fn fetch_listing(&self, url: &str) -> Result<ListingOutcome, ScoreboardError> {
            match url {
                "https://igs.bkg.bund.de/root_ftp/IGS/products/2430/" => {
                    Ok(ListingOutcome::NotPosted(404))
                }
                "https://igs.bkg.bund.de/root_ftp/IGS/products/2429/" => Ok(
                    ListingOutcome::Available(core_listing_fixture("bkg-igs-2429-20260804.html")),
                ),
                other => panic!("unexpected listing URL {other}"),
            }
        }
    }

    let now = ProductDateTime::new(date(2026, 8, 4), 7, 8, 0).expect("query instant");
    let outcome = publication_status(AnalysisCenter::IgsUlt, ProductType::Sp3, now, &RecordedBkg)
        .expect("supported line");
    match outcome {
        PublicationStatusOutcome::Published {
            product,
            listing_url,
            ..
        } => {
            assert_eq!(product.date, date(2026, 7, 28));
            assert_eq!(product.issue, "1800");
            assert_eq!(
                listing_url,
                "https://igs.bkg.bund.de/root_ftp/IGS/products/2429/"
            );
        }
        other => panic!("expected Published, got {other:?}"),
    }
}

/// A transport failure is `Unreachable`, never `NothingPublished`, and never
/// falls back to an older directory whose answer would masquerade as lag.
#[test]
fn publication_status_reports_transport_failure_as_unreachable() {
    struct Down {
        calls: std::cell::RefCell<usize>,
    }
    impl ListingFetcher for Down {
        fn fetch_listing(&self, url: &str) -> Result<ListingOutcome, ScoreboardError> {
            *self.calls.borrow_mut() += 1;
            Err(ScoreboardError::InvalidArgument(format!(
                "connection refused fetching {url}"
            )))
        }
    }

    let fetcher = Down {
        calls: std::cell::RefCell::new(0),
    };
    let now = ProductDateTime::new(date(2026, 8, 4), 7, 8, 0).expect("query instant");
    let outcome = publication_status(AnalysisCenter::GfzUlt, ProductType::Sp3, now, &fetcher)
        .expect("supported line");
    match outcome {
        PublicationStatusOutcome::Unreachable {
            listing_url,
            reason,
        } => {
            assert_eq!(
                listing_url,
                "https://isdc-data.gfz.de/gnss/products/ultra/w2430/"
            );
            assert!(reason.contains("connection refused"), "{reason}");
        }
        other => panic!("expected Unreachable, got {other:?}"),
    }
    assert_eq!(*fetcher.calls.borrow(), 1, "no walk-back past an unknown");
}

/// Every listing answering with no objects of the line is the reachable
/// "nothing published" answer, carrying the URLs that were consulted.
#[test]
fn publication_status_distinguishes_nothing_published_from_unreachable() {
    struct EmptyWeeks;
    impl ListingFetcher for EmptyWeeks {
        fn fetch_listing(&self, _url: &str) -> Result<ListingOutcome, ScoreboardError> {
            Ok(ListingOutcome::NotPosted(404))
        }
    }

    let now = ProductDateTime::new(date(2026, 8, 4), 7, 8, 0).expect("query instant");
    let outcome = publication_status(AnalysisCenter::IgsUlt, ProductType::Sp3, now, &EmptyWeeks)
        .expect("supported line");
    assert_eq!(
        outcome,
        PublicationStatusOutcome::NothingPublished {
            listing_urls: vec![
                "https://igs.bkg.bund.de/root_ftp/IGS/products/2430/".to_string(),
                "https://igs.bkg.bund.de/root_ftp/IGS/products/2429/".to_string(),
            ],
        }
    );
}

/// A reachable archive serving an unrecognizable body (an error page, a
/// format change) is `Unreachable`, never `NothingPublished`: the closed
/// dialect detection refuses to convert "I cannot read this" into "nothing
/// is published here".
#[test]
fn publication_status_treats_an_unrecognizable_listing_as_unreachable() {
    struct ErrorPage;
    impl ListingFetcher for ErrorPage {
        fn fetch_listing(&self, _url: &str) -> Result<ListingOutcome, ScoreboardError> {
            Ok(ListingOutcome::Available(
                "<html><body><h1>503 Service Unavailable</h1></body></html>".to_string(),
            ))
        }
    }

    let now = ProductDateTime::new(date(2026, 8, 4), 7, 8, 0).expect("query instant");
    let outcome = publication_status(AnalysisCenter::GfzUlt, ProductType::Sp3, now, &ErrorPage)
        .expect("supported line");
    match outcome {
        PublicationStatusOutcome::Unreachable { reason, .. } => {
            assert!(reason.contains("unrecognized archive listing"), "{reason}");
        }
        other => panic!("expected Unreachable, got {other:?}"),
    }
}
