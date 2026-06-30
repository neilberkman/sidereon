#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::{
    covariance::{self, Covariance6},
    frames::{nutation, precession, transforms},
    time::TimeScales,
};

#[derive(Debug, Arbitrary)]
struct Input {
    ts: [f64; 7],
    p: [f64; 3],
    v: [f64; 3],
    q: [f64; 3],
    mat3: [[f64; 3]; 3],
    mat6: [[f64; 6]; 6],
    stm6: [[f64; 6]; 6],
    diag6: [f64; 6],
    scalars: [f64; 8],
    compat: bool,
}

fn time_scales(raw: [f64; 7]) -> TimeScales {
    TimeScales {
        jd_whole: raw[0],
        ut1_fraction: raw[1],
        tt_fraction: raw[2],
        tdb_fraction: raw[3],
        jd_ut1: raw[4],
        jd_tt: raw[5],
        jd_tdb: raw[6],
    }
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let ts = time_scales(input.ts);

    assert_ok_finite_or_err(
        "frames::nutation::skyfield_fundamental_arguments",
        nutation::skyfield_fundamental_arguments(input.scalars[0]),
    );
    assert_ok_finite_or_err(
        "frames::nutation::skyfield_iau2000a_radians",
        nutation::skyfield_iau2000a_radians(input.scalars[1]),
    );
    assert_ok_finite_or_err(
        "frames::nutation::skyfield_mean_obliquity_radians",
        nutation::skyfield_mean_obliquity_radians(input.scalars[2]),
    );
    assert_ok_finite_or_err(
        "frames::nutation::build_skyfield_nutation_matrix",
        nutation::build_skyfield_nutation_matrix(
            input.scalars[0],
            input.scalars[1],
            input.scalars[2],
        ),
    );
    assert_ok_finite_or_err(
        "frames::nutation::skyfield_equation_of_the_equinoxes_complimentary_terms",
        nutation::skyfield_equation_of_the_equinoxes_complimentary_terms(input.scalars[3]),
    );
    assert_ok_finite_or_err(
        "frames::precession::compute_skyfield_precession_matrix",
        precession::compute_skyfield_precession_matrix(input.scalars[4]),
    );
    assert_success(
        "frames::precession::build_icrs_to_j2000",
        precession::build_icrs_to_j2000(),
    );

    assert_ok_finite_or_err(
        "transforms::greenwich_mean_sidereal_time_radians",
        transforms::greenwich_mean_sidereal_time_radians(&ts),
    );
    assert_ok_finite_or_err(
        "transforms::greenwich_apparent_sidereal_time_radians",
        transforms::greenwich_apparent_sidereal_time_radians(&ts),
    );
    assert_ok_finite_or_err(
        "transforms::mat3_vec3_mul",
        transforms::mat3_vec3_mul(&input.mat3, &input.p),
    );
    let state = transforms::TemeStateKm {
        position_km: input.p,
        velocity_km_s: input.v,
    };
    assert_ok_finite_or_err(
        "transforms::teme_to_gcrs_compute",
        transforms::teme_to_gcrs_compute(&state, &ts, input.compat),
    );
    assert_ok_finite_or_err(
        "transforms::gcrs_to_itrs_matrix",
        transforms::gcrs_to_itrs_matrix(&ts),
    );
    assert_ok_finite_or_err(
        "transforms::mean_of_date_to_itrs_matrix",
        transforms::mean_of_date_to_itrs_matrix(&ts),
    );
    assert_ok_finite_or_err(
        "transforms::gcrs_to_itrs_compute",
        transforms::gcrs_to_itrs_compute(input.p[0], input.p[1], input.p[2], &ts, input.compat),
    );
    assert_ok_finite_or_err(
        "transforms::itrs_to_gcrs_matrix",
        transforms::itrs_to_gcrs_matrix(&ts),
    );
    assert_ok_finite_or_err(
        "transforms::itrs_to_gcrs_compute",
        transforms::itrs_to_gcrs_compute(input.p[0], input.p[1], input.p[2], &ts),
    );
    assert_ok_finite_or_err(
        "transforms::itrs_to_geodetic_compute",
        transforms::itrs_to_geodetic_compute(input.p[0], input.p[1], input.p[2]),
    );
    assert_ok_finite_or_err(
        "transforms::geodetic_from_ecef_proj",
        transforms::geodetic_from_ecef_proj(input.p[0], input.p[1], input.p[2]),
    );
    assert_ok_finite_or_err(
        "transforms::geodetic_to_itrs",
        transforms::geodetic_to_itrs(input.scalars[5], input.scalars[6], input.scalars[7]),
    );
    let station = transforms::GeodeticStationKm {
        latitude_deg: input.scalars[5],
        longitude_deg: input.scalars[6],
        altitude_km: input.scalars[7],
    };
    assert_ok_finite_or_err(
        "transforms::gcrs_to_topocentric_compute",
        transforms::gcrs_to_topocentric_compute(input.q, &station, &ts, input.compat),
    );

    assert_ok_finite_or_err(
        "Covariance6::try_from_matrix",
        Covariance6::try_from_matrix(input.mat6),
    );
    assert_ok_finite_or_err(
        "Covariance6::from_diagonal",
        Covariance6::from_diagonal(input.diag6),
    );
    if let Ok(cov) = Covariance6::try_from_matrix(input.mat6) {
        assert_success(
            "Covariance6::position_covariance_km2",
            cov.position_covariance_km2(),
        );
        assert_ok_finite_or_err(
            "Covariance6::propagate_with_stm",
            cov.propagate_with_stm(&input.stm6),
        );
    }
    assert_ok_finite_or_err(
        "covariance::rtn_to_eci",
        covariance::rtn_to_eci(&input.mat3, input.p, input.v),
    );
    let _ = covariance::symmetric(&input.mat3);
    let _ = covariance::positive_semidefinite(&input.mat3);
});
