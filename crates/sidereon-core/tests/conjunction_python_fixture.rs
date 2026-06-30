#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps Area 6 conjunction and covariance reference
//! numbers as a JSON fixture for the Python binding's pytest.
//!
//! The fixture is computed only by public `sidereon-core` APIs:
//! collision-probability methods, encounter-frame construction, B-plane
//! covariance projection, PSD checks, and RTN->ECI covariance rotation. The
//! binding marshals into those same APIs, so the Python tests compare against
//! engine output, not hand-authored truth.

use std::path::PathBuf;

use sidereon_core::astro::conjunction::{
    collision_probability, encounter_frame, encounter_plane_covariance, ConjunctionState, PcMethod,
};
use sidereon_core::astro::covariance::{positive_semidefinite, rtn_to_eci, symmetric};

const OBJ1: ConjunctionState = ConjunctionState {
    position_km: [378.39559, 4305.721887, 5752.767554],
    velocity_km_s: [2.360800244, 5.580331936, -4.322349039],
    covariance_km2: [
        [44.5757544811362, 81.6751751052616, -67.8687662707124],
        [81.6751751052616, 158.453402956163, -128.616921644857],
        [-67.8687662707124, -128.616921644858, 105.490542562701],
    ],
};

const OBJ2: ConjunctionState = ConjunctionState {
    position_km: [374.5180598, 4307.560983, 5751.130418],
    velocity_km_s: [-5.388125081, -3.946827739, 3.322820358],
    covariance_km2: [
        [2.31067077720423, 1.69905293875632, -1.4170164577661],
        [1.69905293875632, 1.24957388457206, -1.04174164279599],
        [-1.4170164577661, -1.04174164279599, 0.869260558223714],
    ],
};

const HARD_BODY_RADIUS_KM: f64 = 0.020;
const RTN_POSITION_KM: [f64; 3] = [7000.123, 1234.5, -250.7];
const RTN_VELOCITY_KM_S: [f64; 3] = [1.2, 7.4, 0.3];
const COVARIANCE_RTN: [[f64; 3]; 3] = [[4.0, 0.5, 0.1], [0.5, 9.0, 0.2], [0.1, 0.2, 16.0]];

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn hex3(values: [f64; 3]) -> Vec<String> {
    values.iter().map(|&v| hex(v)).collect()
}

fn hex_mat2(values: &[[f64; 2]; 2]) -> Vec<Vec<String>> {
    values
        .iter()
        .map(|row| row.iter().map(|&v| hex(v)).collect())
        .collect()
}

fn hex_mat3(values: &[[f64; 3]; 3]) -> Vec<Vec<String>> {
    values.iter().map(|row| hex3(*row)).collect()
}

fn add_cov(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

#[test]
fn conjunction_reference_self_validates() {
    let frame = encounter_frame(
        OBJ1.position_km,
        OBJ1.velocity_km_s,
        OBJ2.position_km,
        OBJ2.velocity_km_s,
    )
    .expect("reference encounter frame");
    assert!(frame.miss_km.is_finite());
    assert!(frame.relative_speed_km_s > 0.0);

    for method in [
        PcMethod::FosterEqualArea,
        PcMethod::FosterNumerical,
        PcMethod::Alfano2005,
    ] {
        let pc =
            collision_probability(&OBJ1, &OBJ2, HARD_BODY_RADIUS_KM, method).expect("defined Pc");
        assert!(pc.pc.is_finite());
        assert!(pc.pc > 0.0);
    }

    let combined = add_cov(&OBJ1.covariance_km2, &OBJ2.covariance_km2);
    let projected = encounter_plane_covariance(&frame, &combined).expect("projected covariance");
    assert!(projected[0][0].is_finite());

    let eci = rtn_to_eci(&COVARIANCE_RTN, RTN_POSITION_KM, RTN_VELOCITY_KM_S)
        .expect("non-degenerate RTN frame");
    assert!(symmetric(&COVARIANCE_RTN));
    assert!(positive_semidefinite(&COVARIANCE_RTN));
    assert!(eci[0][0].is_finite());

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn pc_json(method: PcMethod) -> serde_json::Value {
    use serde_json::json;

    let pc = collision_probability(&OBJ1, &OBJ2, HARD_BODY_RADIUS_KM, method).expect("defined Pc");
    let name = match method {
        PcMethod::FosterEqualArea => "FOSTER_EQUAL_AREA",
        PcMethod::FosterNumerical => "FOSTER_NUMERICAL",
        PcMethod::Alfano2005 => "ALFANO_2005",
    };
    json!({
        "method": name,
        "pc_hex": hex(pc.pc),
        "miss_km_hex": hex(pc.miss_km),
        "relative_speed_km_s_hex": hex(pc.relative_speed_km_s),
        "sigma_x_km_hex": hex(pc.sigma_x_km),
        "sigma_z_km_hex": hex(pc.sigma_z_km),
    })
}

fn state_json(state: &ConjunctionState) -> serde_json::Value {
    use serde_json::json;

    json!({
        "position_km_hex": hex3(state.position_km),
        "velocity_km_s_hex": hex3(state.velocity_km_s),
        "covariance_km2_hex": hex_mat3(&state.covariance_km2),
    })
}

fn dump_fixture() {
    use serde_json::json;

    let frame = encounter_frame(
        OBJ1.position_km,
        OBJ1.velocity_km_s,
        OBJ2.position_km,
        OBJ2.velocity_km_s,
    )
    .expect("reference encounter frame");
    let combined = add_cov(&OBJ1.covariance_km2, &OBJ2.covariance_km2);
    let projected = encounter_plane_covariance(&frame, &combined).expect("projected covariance");
    let eci = rtn_to_eci(&COVARIANCE_RTN, RTN_POSITION_KM, RTN_VELOCITY_KM_S)
        .expect("non-degenerate RTN frame");

    let doc = json!({
        "source": "conjunction_reference_self_validates",
        "object1": state_json(&OBJ1),
        "object2": state_json(&OBJ2),
        "hard_body_radius_km_hex": hex(HARD_BODY_RADIUS_KM),
        "frame": {
            "x_hat_hex": hex3(frame.x_hat),
            "y_hat_hex": hex3(frame.y_hat),
            "z_hat_hex": hex3(frame.z_hat),
            "relative_position_km_hex": hex3(frame.relative_position_km),
            "relative_velocity_km_s_hex": hex3(frame.relative_velocity_km_s),
            "miss_km_hex": hex(frame.miss_km),
            "relative_speed_km_s_hex": hex(frame.relative_speed_km_s),
        },
        "collision_probability": [
            pc_json(PcMethod::FosterEqualArea),
            pc_json(PcMethod::FosterNumerical),
            pc_json(PcMethod::Alfano2005),
        ],
        "combined_covariance_km2_hex": hex_mat3(&combined),
        "encounter_plane_covariance_hex": hex_mat2(&projected),
        "rtn": {
            "position_km_hex": hex3(RTN_POSITION_KM),
            "velocity_km_s_hex": hex3(RTN_VELOCITY_KM_S),
            "covariance_rtn_hex": hex_mat3(&COVARIANCE_RTN),
            "covariance_eci_hex": hex_mat3(&eci),
            "symmetric": symmetric(&COVARIANCE_RTN),
            "positive_semidefinite": positive_semidefinite(&COVARIANCE_RTN),
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/conjunction.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped conjunction fixture to {out:?}");
}
