use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_sidereon")
}

fn fixture(parts: &[&str]) -> PathBuf {
    let mut path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sidereon-core/tests/fixtures");
    for part in parts {
        path.push(part);
    }
    path
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("run sidereon binary")
}

fn temp_text_file(name: &str, text: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "sidereon-cli-test-{}-{nonce}-{name}",
        std::process::id()
    ));
    fs::write(&path, text).expect("write temp file");
    path
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout utf8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr utf8")
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn metrics_json_reports_expected_numeric_bounds() {
    let output = run(&["metrics", "--enu-cov", "4,0,0,0,9,0,0,0,16", "--json"]);
    assert!(
        output.status.success(),
        "status {:?}\nstderr:\n{}",
        output.status.code(),
        stderr(&output)
    );
    let json: Value = serde_json::from_slice(&output.stdout).expect("metrics JSON");
    assert_eq!(json["enu_covariance_m2"][0][0].as_f64(), Some(4.0));
    assert_eq!(json["enu_covariance_m2"][1][1].as_f64(), Some(9.0));
    assert_eq!(json["enu_covariance_m2"][2][2].as_f64(), Some(16.0));
    assert_eq!(json["sigma_e_m"].as_f64(), Some(2.0));
    assert_eq!(json["sigma_n_m"].as_f64(), Some(3.0));
    assert_eq!(json["sigma_u_m"].as_f64(), Some(4.0));
    assert_eq!(json["ellipse_orientation_deg"].as_f64(), Some(90.0));
    assert_close(
        json["cep_m"].as_f64().expect("CEP"),
        2.9263950341693947,
        1.0e-12,
        "CEP",
    );
    assert_close(
        json["r95_m"].as_f64().expect("R95"),
        6.366519799238128,
        1.0e-12,
        "R95",
    );
    assert_close(
        json["vertical_radius_m"].as_f64().expect("vertical radius"),
        7.839855938160215,
        1.0e-12,
        "vertical radius",
    );
}

#[test]
fn inspect_observation_fixture_reports_structure() {
    let obs = fixture(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
    let output = run(&["inspect", obs.to_str().expect("fixture path utf8")]);
    assert!(
        output.status.success(),
        "status {:?}\nstderr:\n{}",
        output.status.code(),
        stderr(&output)
    );
    let stdout = stdout(&output);
    assert!(stdout.contains("type: RINEX OBS"));
    assert!(stdout.contains("epochs: 2"));
    assert!(stdout.contains("satellites:"));
    assert!(stdout.contains("G05"));
}

#[test]
fn inspect_tle_fixture_is_not_misclassified_as_antex() {
    let tle = fixture(&["celestrak", "stations.tle"]);
    let output = run(&["inspect", tle.to_str().expect("fixture path utf8")]);
    assert!(
        output.status.success(),
        "status {:?}\nstderr:\n{}",
        output.status.code(),
        stderr(&output)
    );
    let stdout = stdout(&output);
    assert!(stdout.contains("type: TLE"), "{stdout}");
    assert!(!stdout.contains("type: ANTEX"), "{stdout}");
    assert!(stdout.contains("tle_pairs:"), "{stdout}");
}

#[test]
fn inspect_empty_and_garbage_text_are_unrecognized() {
    for (name, text) in [("empty.txt", ""), ("garbage.txt", "not a gnss file\n")] {
        let path = temp_text_file(name, text);
        let output = run(&["inspect", path.to_str().expect("temp path utf8")]);
        assert!(
            !output.status.success(),
            "{name} should not inspect successfully\nstdout:\n{}\nstderr:\n{}",
            stdout(&output),
            stderr(&output)
        );
        assert!(
            stderr(&output).contains("unrecognized file type"),
            "{name} stderr:\n{}",
            stderr(&output)
        );
        let _ = fs::remove_file(path);
    }
}

#[test]
fn inspect_nav_reports_compatible_time_bases_separately() {
    let nav = fixture(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
    let output = run(&["inspect", nav.to_str().expect("fixture path utf8")]);
    assert!(
        output.status.success(),
        "status {:?}\nstderr:\n{}",
        output.status.code(),
        stderr(&output)
    );
    let stdout = stdout(&output);
    assert!(stdout.contains("type: RINEX NAV"), "{stdout}");
    assert!(stdout.contains("native_week_s"), "{stdout}");
    if !stdout.contains("glonass_records: 0") {
        assert!(stdout.contains("glonass_j2000_s"), "{stdout}");
    }
    assert!(!stdout.contains("native_s"), "{stdout}");
}

#[test]
fn qc_json_includes_lint_counts_and_qc_report() {
    let obs = fixture(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx"]);
    let output = run(&[
        "qc",
        "--obs",
        obs.to_str().expect("fixture path utf8"),
        "--json",
    ]);
    assert!(
        output.status.success(),
        "status {:?}\nstderr:\n{}",
        output.status.code(),
        stderr(&output)
    );
    let json: Value = serde_json::from_slice(&output.stdout).expect("qc JSON");
    assert!(json["lint"]["counts"]["fatal"].as_u64().is_some());
    assert!(
        json["qc"]["total_epoch_records"]
            .as_u64()
            .expect("total epochs")
            >= 2
    );
}

#[test]
fn solve_json_reports_successful_epochs_and_metrics() {
    let obs = fixture(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
    let nav = fixture(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
    let sp3 = fixture(&["sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"]);
    let output = run(&[
        "solve",
        "--obs",
        obs.to_str().expect("fixture path utf8"),
        "--nav",
        nav.to_str().expect("fixture path utf8"),
        "--sp3",
        sp3.to_str().expect("fixture path utf8"),
        "--json",
    ]);
    assert!(
        output.status.success(),
        "status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout(&output),
        stderr(&output)
    );
    let json: Value = serde_json::from_slice(&output.stdout).expect("solve JSON");
    assert_eq!(json["summary"]["assembled_epochs"].as_u64(), Some(2));
    assert!(json["summary"]["solved_count"].as_u64().expect("solved") >= 1);

    let first = json["epochs"]
        .as_array()
        .expect("epochs array")
        .iter()
        .find(|epoch| epoch["solved"].as_bool() == Some(true))
        .expect("successful epoch");
    let lat = first["lat_deg"].as_f64().expect("lat");
    let lon = first["lon_deg"].as_f64().expect("lon");
    let height = first["height_m"].as_f64().expect("height");
    let cep = first["metrics"]["cep_m"].as_f64().expect("CEP");
    assert!((50.0..60.0).contains(&lat), "lat {lat}");
    assert!((5.0..15.0).contains(&lon), "lon {lon}");
    assert!((-100.0..500.0).contains(&height), "height {height}");
    assert!(cep.is_finite() && cep >= 0.0, "CEP {cep}");
}

#[test]
fn ppc_score_json_reports_score_and_provenance() {
    let truth = temp_text_file(
        "ppc-reference.csv",
        "GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n0,2325,0,0,0\n1,2325,0,0.00001,0\n",
    );
    let solution = temp_text_file(
        "ppc-solution.csv",
        "GPS TOW (s),Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n1,0,0.00001,0\n",
    );
    let output = run(&[
        "ppc-score",
        "--truth",
        truth.to_str().expect("truth path utf8"),
        "--solution",
        solution.to_str().expect("solution path utf8"),
        "--route",
        "contract-route",
        "--dataset-revision",
        "ppc-test-revision",
        "--dataset-sha256",
        "test-digest",
        "--git-commit",
        "deadbeef",
        "--json",
    ]);
    assert!(
        output.status.success(),
        "status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout(&output),
        stderr(&output)
    );
    let json: Value = serde_json::from_slice(&output.stdout).expect("PPC score JSON");
    assert_eq!(json["metadata"]["scorer_version"], "sidereon-ppc-v1");
    assert_eq!(
        json["metadata"]["sidereon_version"],
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(json["metadata"]["git_commit"], "deadbeef");
    assert_eq!(json["metadata"]["dataset_revision"], "ppc-test-revision");
    assert_eq!(json["metadata"]["dataset_sha256"], "test-digest");
    assert_eq!(json["metadata"]["threshold_m"].as_f64(), Some(0.5));
    assert_eq!(json["routes"][0]["route"], "contract-route");
    assert_eq!(json["routes"][0]["score_percent"].as_f64(), Some(100.0));
    assert_eq!(json["average_score_percent"].as_f64(), Some(100.0));

    let _ = fs::remove_file(truth);
    let _ = fs::remove_file(solution);
}

#[test]
fn ppc_score_rejects_mismatched_route_inputs() {
    let truth = temp_text_file(
        "ppc-reference-count.csv",
        "GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n0,2325,0,0,0\n",
    );
    let solution = temp_text_file(
        "ppc-solution-count.csv",
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n0,1,2,3\n",
    );
    let output = run(&[
        "ppc-score",
        "--truth",
        truth.to_str().expect("truth path utf8"),
        "--truth",
        truth.to_str().expect("truth path utf8"),
        "--solution",
        solution.to_str().expect("solution path utf8"),
    ]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--truth and --solution must have the same number of paths"),
        "stderr:\n{}",
        stderr(&output)
    );

    let _ = fs::remove_file(truth);
    let _ = fs::remove_file(solution);
}

#[test]
fn ppc_solve_help_describes_private_causal_adapter() {
    let output = run(&["ppc-solve", "--help"]);
    assert!(output.status.success(), "stderr:\n{}", stderr(&output));
    let stdout = stdout(&output);
    assert!(stdout.contains("private causal single-frequency PPC adapter"));
    assert!(stdout.contains("--max-base-age-s"));
    assert!(stdout.contains("--ambiguity-retirement-age-s"));
    assert!(stdout.contains("--max-ambiguity-columns"));
    assert!(stdout.contains("--solution-out"));
}

#[test]
fn ppc_solve_validates_scientific_options_before_loading_inputs() {
    let common = [
        "ppc-solve",
        "--base-obs",
        "missing-base.obs",
        "--rover-obs",
        "missing-rover.obs",
        "--nav",
        "missing.nav",
        "--truth",
        "missing-reference.csv",
        "--solution-out",
        "missing-solution.csv",
    ];
    for (option, expected) in [
        (
            "--max-base-age-s=-1",
            "--max-base-age-s must be finite and non-negative",
        ),
        ("--max-epochs=0", "--max-epochs must be positive"),
        (
            "--elevation-mask-deg=90",
            "--elevation-mask-deg must be finite and in [0, 90)",
        ),
        (
            "--hold-sigma-m=0",
            "--hold-sigma-m must be finite and positive",
        ),
        (
            "--process-noise-sigma-m=-1",
            "--process-noise-sigma-m must be finite and non-negative",
        ),
        (
            "--ambiguity-retirement-age-s=-1",
            "--ambiguity-retirement-age-s must be finite and non-negative",
        ),
        (
            "--max-ambiguity-columns=0",
            "--max-ambiguity-columns must be positive",
        ),
    ] {
        let mut args = common.to_vec();
        args.push(option);
        let output = run(&args);
        assert!(
            !output.status.success(),
            "option {option} unexpectedly passed"
        );
        assert!(
            stderr(&output).contains(expected),
            "option {option} stderr:\n{}",
            stderr(&output)
        );
    }
}

#[test]
fn ppc_solve_rejects_an_output_that_aliases_an_input() {
    let output = run(&[
        "ppc-solve",
        "--base-obs",
        "same.obs",
        "--rover-obs",
        "missing-rover.obs",
        "--nav",
        "missing.nav",
        "--truth",
        "missing-reference.csv",
        "--solution-out",
        "same.obs",
    ]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--solution-out must not alias --base-obs"),
        "stderr:\n{}",
        stderr(&output)
    );
}

#[cfg(unix)]
#[test]
fn ppc_solve_rejects_a_hard_link_to_an_input() {
    let input = temp_text_file("ppc-hard-link-input.obs", "immutable input\n");
    let output = temp_text_file("ppc-hard-link-output.obs", "placeholder\n");
    fs::remove_file(&output).expect("remove placeholder");
    fs::hard_link(&input, &output).expect("create input hard link");
    let result = run(&[
        "ppc-solve",
        "--base-obs",
        input.to_str().expect("input path utf8"),
        "--rover-obs",
        "missing-rover.obs",
        "--nav",
        "missing.nav",
        "--truth",
        "missing-reference.csv",
        "--solution-out",
        output.to_str().expect("output path utf8"),
    ]);
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("--solution-out must not alias --base-obs"),
        "stderr:\n{}",
        stderr(&result)
    );
    assert_eq!(
        fs::read_to_string(&input).expect("read input"),
        "immutable input\n"
    );
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(input);
}
