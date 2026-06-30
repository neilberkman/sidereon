#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::astro::time::model::TimeScale;
use sidereon_core::orbit::{
    drift, fit_piecewise, fit_with_model, piecewise_drift, piecewise_position, position,
    position_velocity, select_piecewise_segment, CalendarEpoch, EcefSample, Elements, Frame, Model,
};

const GOLDEN: &str = include_str!("fixtures/reduced_orbit_golden.json");

fn parse_hex_float(s: &str) -> f64 {
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
        (-1.0, rest)
    } else {
        (1.0, s)
    };
    let body = body
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("not a hex float (missing 0x): {s:?}"));
    let (mantissa, exponent) = body
        .split_once('p')
        .unwrap_or_else(|| panic!("not a hex float (missing p exponent): {s:?}"));
    let exponent: i32 = exponent
        .parse()
        .unwrap_or_else(|_| panic!("bad hex exponent in {s:?}"));
    let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut value = u64::from_str_radix(whole, 16)
        .unwrap_or_else(|_| panic!("bad integer hex digits in {s:?}")) as f64;
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        let digit = c
            .to_digit(16)
            .unwrap_or_else(|| panic!("bad hex frac digit {c:?} in {s:?}"));
        value += digit as f64 * scale;
        scale /= 16.0;
    }
    sign * value * 2.0_f64.powi(exponent)
}

fn hexf(v: &Value) -> f64 {
    parse_hex_float(v.as_str().expect("hex float string"))
}

fn doc() -> Value {
    serde_json::from_str(GOLDEN).expect("parse reduced_orbit_golden.json")
}

fn epoch(s: &str) -> CalendarEpoch {
    let (date, time) = s.split_once('T').expect("ISO epoch has T");
    let mut d = date.split('-');
    let mut t = time.split(':');
    CalendarEpoch::new(
        d.next().unwrap().parse().unwrap(),
        d.next().unwrap().parse().unwrap(),
        d.next().unwrap().parse().unwrap(),
        t.next().unwrap().parse().unwrap(),
        t.next().unwrap().parse().unwrap(),
        t.next().unwrap().parse().unwrap(),
    )
}

fn vec3(v: &Value) -> [f64; 3] {
    let arr = v.as_array().expect("vec3 array");
    [hexf(&arr[0]), hexf(&arr[1]), hexf(&arr[2])]
}

fn ecef_sample(v: &Value) -> EcefSample {
    let pos = vec3(&v["ecef_m"]);
    EcefSample::new(
        epoch(v["epoch"].as_str().expect("sample epoch")),
        pos[0],
        pos[1],
        pos[2],
    )
}

fn samples(case: &Value) -> Vec<EcefSample> {
    case["samples"]
        .as_array()
        .expect("samples array")
        .iter()
        .map(ecef_sample)
        .collect()
}

fn model_from_name(name: &str) -> Model {
    match name {
        "circular_secular" => Model::CircularSecular,
        "eccentric_secular" => Model::EccentricSecular,
        other => panic!("unknown reduced-orbit model {other:?}"),
    }
}

fn model_from_case(case: &Value) -> Model {
    model_from_name(case["model"].as_str().expect("case model"))
}

fn elements_from_map(map: &Value) -> Elements {
    let e = &map["elements"];
    Elements {
        model: model_from_name(map["model"].as_str().expect("map model")),
        epoch: epoch(map["epoch"].as_str().expect("map epoch")),
        a_m: hexf(&e["a_m"]),
        e: hexf(&e["e"]),
        i_rad: hexf(&e["i_rad"]),
        raan_rad: hexf(&e["raan_rad"]),
        raan_rate_rad_s: hexf(&e["raan_rate_rad_s"]),
        raan_rate_j2_rad_s: hexf(&e["raan_rate_j2_rad_s"]),
        arg_lat_rad: hexf(&e["arg_lat_rad"]),
        mean_motion_rad_s: hexf(&e["mean_motion_rad_s"]),
        h: e.get("h").map_or(0.0, hexf),
        k: e.get("k").map_or(0.0, hexf),
        arg_perigee_rad: e.get("arg_perigee_rad").map_or(0.0, hexf),
    }
}

fn assert_close(label: &str, got: f64, want: f64, tol: f64) {
    let delta = (got - want).abs();
    assert!(
        delta <= tol,
        "{label}: got {got:.17e}, want {want:.17e}, delta {delta:.3e}, tol {tol:.3e}"
    );
}

fn assert_vec3(label: &str, got: [f64; 3], want: &Value, tol: f64) {
    let want = vec3(want);
    for axis in 0..3 {
        assert_close(&format!("{label}[{axis}]"), got[axis], want[axis], tol);
    }
}

#[test]
fn fit_recovers_independent_astropy_scipy_oracle_elements() {
    let golden = doc();

    let circular = &golden["cases"]["circular"];
    let fit = fit_with_model(
        &samples(circular),
        TimeScale::Utc,
        model_from_case(circular),
    )
    .expect("circular fit");
    let expected = &circular["fit"]["map"]["elements"];
    assert_eq!(
        fit.stats.n_samples,
        circular["fit"]["stats"]["n_samples"].as_u64().unwrap() as usize
    );
    assert_close(
        "circular a_m",
        fit.elements.a_m,
        hexf(&expected["a_m"]),
        25.0,
    );
    assert_close(
        "circular i_rad",
        fit.elements.i_rad,
        hexf(&expected["i_rad"]),
        2.0e-6,
    );
    assert_close(
        "circular raan_rad",
        fit.elements.raan_rad,
        hexf(&expected["raan_rad"]),
        2.0e-6,
    );
    assert_close(
        "circular raan_rate_rad_s",
        fit.elements.raan_rate_rad_s,
        hexf(&expected["raan_rate_rad_s"]),
        5.0e-10,
    );
    assert_close(
        "circular arg_lat_rad",
        fit.elements.arg_lat_rad,
        hexf(&expected["arg_lat_rad"]),
        6.0e-6,
    );
    assert_close(
        "circular mean_motion_rad_s",
        fit.elements.mean_motion_rad_s,
        hexf(&expected["mean_motion_rad_s"]),
        5.0e-10,
    );
    assert!(fit.stats.rms_m < 25.0, "circular rms {}", fit.stats.rms_m);
    assert!(fit.stats.max_m < 50.0, "circular max {}", fit.stats.max_m);

    let eccentric = &golden["cases"]["eccentric"];
    let fit = fit_with_model(
        &samples(eccentric),
        TimeScale::Utc,
        model_from_case(eccentric),
    )
    .expect("eccentric fit");
    let expected = &eccentric["fit"]["map"]["elements"];
    assert_eq!(
        fit.stats.n_samples,
        eccentric["fit"]["stats"]["n_samples"].as_u64().unwrap() as usize
    );
    assert_close(
        "eccentric a_m",
        fit.elements.a_m,
        hexf(&expected["a_m"]),
        50.0,
    );
    assert_close("eccentric e", fit.elements.e, hexf(&expected["e"]), 2.0e-6);
    assert_close("eccentric h", fit.elements.h, hexf(&expected["h"]), 2.0e-6);
    assert_close("eccentric k", fit.elements.k, hexf(&expected["k"]), 2.0e-6);
    assert_close(
        "eccentric i_rad",
        fit.elements.i_rad,
        hexf(&expected["i_rad"]),
        2.0e-6,
    );
    assert_close(
        "eccentric raan_rad",
        fit.elements.raan_rad,
        hexf(&expected["raan_rad"]),
        6.0e-6,
    );
    assert_close(
        "eccentric raan_rate_rad_s",
        fit.elements.raan_rate_rad_s,
        hexf(&expected["raan_rate_rad_s"]),
        5.0e-10,
    );
    assert_close(
        "eccentric arg_lat_rad",
        fit.elements.arg_lat_rad,
        hexf(&expected["arg_lat_rad"]),
        6.0e-6,
    );
    assert_close(
        "eccentric mean_motion_rad_s",
        fit.elements.mean_motion_rad_s,
        hexf(&expected["mean_motion_rad_s"]),
        5.0e-10,
    );
    assert!(fit.stats.rms_m < 50.0, "eccentric rms {}", fit.stats.rms_m);
    assert!(fit.stats.max_m < 100.0, "eccentric max {}", fit.stats.max_m);
}

#[test]
fn position_velocity_and_drift_evaluate_oracle_model_maps() {
    let golden = doc();
    let cases = golden["cases"].as_object().expect("cases object");

    for (name, case) in cases {
        let elements = elements_from_map(&case["fit"]["map"]);

        for pos in case["positions"].as_array().expect("positions array") {
            let query = epoch(pos["epoch"].as_str().expect("position epoch"));
            let gcrs = position(&elements, query, TimeScale::Utc, Frame::Gcrs)
                .expect("valid reduced-orbit position");
            assert_vec3(
                &format!("{name} gcrs position"),
                gcrs,
                &pos["gcrs_m"],
                1.0e-5,
            );

            let ecef = position(&elements, query, TimeScale::Utc, Frame::Ecef)
                .expect("valid reduced-orbit position");
            assert_vec3(
                &format!("{name} ecef position"),
                ecef,
                &pos["ecef_m"],
                100.0,
            );
        }

        for vel in case["velocities"].as_array().expect("velocities array") {
            let query = epoch(vel["epoch"].as_str().expect("velocity epoch"));
            let (_pos, gcrs_vel) = position_velocity(&elements, query, TimeScale::Utc, Frame::Gcrs)
                .expect("valid reduced-orbit position/velocity");
            assert_vec3(
                &format!("{name} gcrs velocity"),
                gcrs_vel,
                &vel["gcrs_m_s"],
                1.0e-9,
            );
        }

        let report = drift(&elements, &samples(case), TimeScale::Utc, 1.0e9)
            .expect("valid reduced-orbit drift");
        let expected = &case["drift"];
        let expected_rows = expected["per_epoch"].as_array().expect("drift rows");
        assert_eq!(
            report.per_epoch.len(),
            expected_rows.len(),
            "{name} drift rows"
        );
        assert_close(
            &format!("{name} drift max_m"),
            report.max_m,
            hexf(&expected["max_m"]),
            100.0,
        );
        assert_close(
            &format!("{name} drift rms_m"),
            report.rms_m,
            hexf(&expected["rms_m"]),
            100.0,
        );
        for (got, want) in report.per_epoch.iter().zip(expected_rows) {
            assert_eq!(
                got.epoch,
                epoch(want["epoch"].as_str().expect("drift epoch")),
                "{name} drift epoch"
            );
            assert_close(
                &format!("{name} drift {}", want["epoch"].as_str().unwrap()),
                got.error_m,
                hexf(&want["error_m"]),
                100.0,
            );
        }
    }
}

#[test]
fn piecewise_fit_segment_selection_and_positions_match_oracle_fixture() {
    let golden = doc();
    let case = &golden["cases"]["eccentric"];
    let expected = &golden["piecewise"];
    let t0 = epoch(golden["epoch0"].as_str().expect("epoch0"));
    let t1 = epoch(
        case["samples"]
            .as_array()
            .expect("samples array")
            .last()
            .expect("last sample")["epoch"]
            .as_str()
            .expect("last sample epoch"),
    );
    let segment_s = hexf(&expected["segment_s"]).round() as i64;

    let fit = fit_piecewise(
        &samples(case),
        TimeScale::Utc,
        Model::EccentricSecular,
        t0,
        t1,
        segment_s,
    )
    .expect("piecewise fit");

    let empty_report =
        piecewise_drift(&fit, &[], TimeScale::Utc, 1.0).expect("empty truth drift report");
    assert!(empty_report.per_epoch.is_empty());
    assert_eq!(empty_report.max_m.to_bits(), 0.0_f64.to_bits());
    assert_eq!(empty_report.rms_m.to_bits(), 0.0_f64.to_bits());
    assert_eq!(empty_report.threshold_horizon, None);

    assert_eq!(
        fit.segments.len(),
        expected["segments"]
            .as_array()
            .expect("segments array")
            .len()
    );

    const SEGMENT_BITS: [[u64; 12]; 3] = [
        [
            0x4179546001cf1c9f,
            0x3f9581060cf08b5e,
            0x3feeb7be4eaab54f,
            0x3feccccd3babfd35,
            0xbe41fd6098ae7a67,
            0xbe40d328537e8dd5,
            0x3ffcccce64470cd1,
            0x3f231e25d07b3033,
            0x3f8bb4e2bfa3e51c,
            0x3f9072776cf93bfc,
            0x3fe34fa6acf0a6e5,
            0x3fee68d6203d802a,
        ],
        [
            0x4179545fd6f4d557,
            0x3f95810d5e43e60c,
            0x3feeb7c92d502ba5,
            0x3feccc4b60803ab9,
            0xbe4057a8f6e4d085,
            0xbe40d31ca87276a8,
            0x4006cd2738a46d06,
            0x3f231e2193dc2ddb,
            0x3f8bb4eb530bf6cc,
            0x3f90727d61b02a98,
            0x3ff186d713388be4,
            0x3fff422c7666cbef,
        ],
        [
            0x4179545ff03f890b,
            0x3f95810836f619a2,
            0x3feeb7c537b24e19,
            0x3feccbd40412fec2,
            0xbe416e04ab7ebc96,
            0xbe40d3212e9db367,
            0xc00310110a8a3eed,
            0x3f231e237d36bcd5,
            0x3f8bb4e61788c609,
            0x3f907278d8e118b6,
            0x3fdb0a8bfbfb9b77,
            0x3fe628fe419ca80d,
        ],
    ];

    for (index, seg) in fit.segments.iter().enumerate() {
        let e = &seg.orbit.elements;
        let got = [
            e.a_m.to_bits(),
            e.e.to_bits(),
            e.i_rad.to_bits(),
            e.raan_rad.to_bits(),
            e.raan_rate_rad_s.to_bits(),
            e.raan_rate_j2_rad_s.to_bits(),
            e.arg_lat_rad.to_bits(),
            e.mean_motion_rad_s.to_bits(),
            e.h.to_bits(),
            e.k.to_bits(),
            seg.orbit.stats.rms_m.to_bits(),
            seg.orbit.stats.max_m.to_bits(),
        ];
        assert_eq!(got, SEGMENT_BITS[index], "piecewise segment {index} bits");
        assert_eq!(seg.orbit.stats.n_samples, 9);
    }

    const POSITION_BITS: [[u64; 3]; 4] = [
        [0xc171bba826325350, 0xc1425cc7fe82ded1, 0x4171ccd202c5ad17],
        [0xc1715102ae7dda3b, 0xc172c5ff1e2b02d6, 0x4145a09e7e63f9c4],
        [0x41340d617d132139, 0xc172959829f423d3, 0xc171e72b84cc2cab],
        [0x416cdd54f4e8b818, 0xc153c12ec8611374, 0xc17487514877aa34],
    ];

    for (pos_index, pos) in expected["positions"]
        .as_array()
        .expect("positions array")
        .iter()
        .enumerate()
    {
        let query = epoch(pos["epoch"].as_str().expect("position epoch"));
        let got = piecewise_position(&fit, query, TimeScale::Utc, Frame::Gcrs)
            .expect("piecewise position");
        assert_eq!(
            [got[0].to_bits(), got[1].to_bits(), got[2].to_bits()],
            POSITION_BITS[pos_index],
            "piecewise position {pos_index} bits"
        );
        assert_vec3("piecewise gcrs position", got, &pos["gcrs_m"], 150.0);

        let seg = select_piecewise_segment(&fit, query).expect("selected segment");
        let seg_index = fit
            .segments
            .iter()
            .position(|candidate| core::ptr::eq(candidate, seg))
            .expect("selected segment belongs to model");
        assert_eq!(
            seg_index,
            pos["segment_index"].as_u64().expect("segment index") as usize
        );
    }
}
