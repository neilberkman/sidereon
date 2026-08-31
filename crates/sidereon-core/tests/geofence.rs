//! Provenance: Fixtures are analytic WGS84 meridian, equator, and dateline
//! polygon cases derived from published geodesic geometry. Probability checks
//! use textbook Gaussian half-space formulas and fixed deterministic
//! quadrature. No geofence package source was read or referenced.

use sidereon_core::{
    containment, containment_probability, containment_probability_with_options, crossing,
    crossing_probability, distance_to_boundary, geodesic_direct, geodesic_inverse, CrossingKind,
    Fence, GeofenceError, GeofencePositionEstimate, PositionUncertainty, ProbabilityHysteresis,
    ProbabilityMethod, ProbabilityOptions, Wgs84Geodetic,
};

fn geo(lat_deg: f64, lon_deg: f64) -> Wgs84Geodetic {
    Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), 0.0).expect("valid geodetic")
}

fn east_from(lat_deg: f64, lon_deg: f64, distance_m: f64) -> Wgs84Geodetic {
    let (lat2, lon2, _) =
        geodesic_direct(lat_deg, lon_deg, 90.0, distance_m).expect("direct geodesic");
    geo(lat2, lon2)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x * core::f64::consts::FRAC_1_SQRT_2))
}

#[test]
fn dateline_long_edge_contains_the_short_geodesic_polygon() {
    let fence = Fence::new([
        geo(-10.0, 170.0),
        geo(-10.0, -170.0),
        geo(10.0, -170.0),
        geo(10.0, 170.0),
    ])
    .expect("fence");

    assert!(containment(geo(0.0, 180.0), &fence).expect("dateline point"));
    assert!(!containment(geo(0.0, 0.0), &fence).expect("greenwich point"));
    assert!(!fence.planar_fast_path_applies(geo(0.0, 179.0)));

    let signed = distance_to_boundary(geo(0.0, 179.0), &fence).expect("distance");
    let (expected, _, _) = geodesic_inverse(0.0, 179.0, 0.0, 170.0).expect("oracle");
    assert_close(signed, expected, 1.0);
}

#[test]
fn boundary_distance_matches_equatorial_edge_oracle() {
    let fence = Fence::new([
        geo(-1.0, -1.0),
        geo(-1.0, 1.0),
        geo(1.0, 1.0),
        geo(1.0, -1.0),
    ])
    .expect("fence");
    let point = geo(0.0, 1.01);
    let signed = distance_to_boundary(point, &fence).expect("distance");
    let (expected, _, _) = geodesic_inverse(0.0, 1.01, 0.0, 1.0).expect("oracle");
    assert_close(signed, -expected, 0.01);
}

#[test]
fn planar_fast_path_stays_inside_its_bound() {
    let fence = Fence::new([
        geo(44.999, -122.001),
        geo(44.999, -121.999),
        geo(45.001, -121.999),
        geo(45.001, -122.001),
    ])
    .expect("fence");
    let point = geo(45.0, -121.9995);
    assert!(fence.planar_fast_path_applies(point));
    let planar = fence
        .distance_to_boundary_planar_fast(point)
        .expect("planar distance")
        .expect("inside bound");
    let signed = distance_to_boundary(point, &fence).expect("distance");
    let (expected, _, _) = geodesic_inverse(45.0, -121.9995, 45.0, -121.999).expect("oracle");
    assert_close(signed, expected, 0.05);
    assert_close(planar, signed, 0.05);
}

#[test]
fn containment_probability_matches_circular_gaussian_half_space() {
    let fence = Fence::new([
        geo(-0.01, 0.0),
        geo(-0.01, 0.02),
        geo(0.01, 0.02),
        geo(0.01, 0.0),
    ])
    .expect("fence");
    let distance_m = 30.0;
    let sigma_m = 20.0;
    let point = east_from(0.0, 0.0, distance_m);
    let covariance = [
        [sigma_m * sigma_m, 0.0, 0.0],
        [0.0, sigma_m * sigma_m, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let probability = containment_probability(
        point,
        PositionUncertainty::EnuCovarianceM2(covariance),
        &fence,
    )
    .expect("probability");
    let (oracle_distance_m, _, _) =
        geodesic_inverse(0.0, point.lon_rad.to_degrees(), 0.0, 0.0).expect("oracle");
    assert_close(probability, normal_cdf(oracle_distance_m / sigma_m), 2.0e-9);

    let cep_m = sigma_m * libm::sqrt(2.0_f64 * libm::log(2.0));
    let from_cep = containment_probability(point, PositionUncertainty::CepRadiusM(cep_m), &fence)
        .expect("cep probability");
    assert_close(from_cep, probability, 1.0e-12);
}

#[test]
fn quadrature_agrees_with_half_space_approximation_inside_bound() {
    let fence = Fence::new([
        geo(-0.02, 0.0),
        geo(-0.02, 0.04),
        geo(0.02, 0.04),
        geo(0.02, 0.0),
    ])
    .expect("fence");
    let point = east_from(0.0, 0.0, 25.0);
    let covariance = [[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 0.0]];
    let approximation = containment_probability_with_options(
        point,
        PositionUncertainty::EnuCovarianceM2(covariance),
        &fence,
        ProbabilityOptions {
            method: ProbabilityMethod::BoundaryNormal,
        },
    )
    .expect("approximation");
    let quadrature = containment_probability_with_options(
        point,
        PositionUncertainty::EnuCovarianceM2(covariance),
        &fence,
        ProbabilityOptions {
            method: ProbabilityMethod::PlanarQuadrature,
        },
    )
    .expect("quadrature");
    assert_close(quadrature, approximation, 0.02);
}

#[test]
fn probabilistic_crossing_does_not_chatter_below_confidence() {
    let fence = Fence::new([
        geo(-0.01, 0.0),
        geo(-0.01, 0.02),
        geo(0.01, 0.02),
        geo(0.01, 0.0),
    ])
    .expect("fence");
    let covariance = [[900.0, 0.0, 0.0], [0.0, 900.0, 0.0], [0.0, 0.0, 0.0]];
    let samples: Vec<_> = [-5.0, 5.0, -4.0, 4.0]
        .into_iter()
        .map(|offset_m| GeofencePositionEstimate {
            position: east_from(0.0, 0.0, offset_m),
            uncertainty: PositionUncertainty::EnuCovarianceM2(covariance),
        })
        .collect();
    let hysteresis = ProbabilityHysteresis::new(0.8, 0.8).expect("hysteresis");
    let events = crossing_probability(&samples, &fence, hysteresis).expect("events");
    assert!(events.is_empty());
}

#[test]
fn hysteresis_rejects_non_confident_thresholds() {
    assert!(ProbabilityHysteresis::new(0.5, 0.8).is_err());
    assert!(ProbabilityHysteresis::new(0.8, 0.5).is_err());
    assert!(ProbabilityHysteresis::new(0.49, 0.9).is_err());
}

#[test]
fn probabilistic_crossing_waits_for_confident_initial_state() {
    let fence = Fence::new([
        geo(-0.01, 0.0),
        geo(-0.01, 0.02),
        geo(0.01, 0.02),
        geo(0.01, 0.0),
    ])
    .expect("fence");
    let covariance = [[900.0, 0.0, 0.0], [0.0, 900.0, 0.0], [0.0, 0.0, 0.0]];
    let samples: Vec<_> = [3.0, -30.0, 30.0]
        .into_iter()
        .map(|offset_m| GeofencePositionEstimate {
            position: east_from(0.0, 0.0, offset_m),
            uncertainty: PositionUncertainty::EnuCovarianceM2(covariance),
        })
        .collect();
    let hysteresis = ProbabilityHysteresis::new(0.8, 0.8).expect("hysteresis");
    let events = crossing_probability(&samples, &fence, hysteresis).expect("events");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].sample_index, 2);
    assert_eq!(events[0].kind, CrossingKind::Entered);
}

#[test]
fn planar_quadrature_boundary_edge_and_corner_probabilities() {
    let fence = Fence::new([
        geo(0.0, 0.0),
        geo(0.0, 0.02),
        geo(0.02, 0.02),
        geo(0.02, 0.0),
    ])
    .expect("fence");
    let sigma_m = 20.0;
    let covariance = [
        [sigma_m * sigma_m, 0.0, 0.0],
        [0.0, sigma_m * sigma_m, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let options = ProbabilityOptions {
        method: ProbabilityMethod::PlanarQuadrature,
    };
    let edge_probability = containment_probability_with_options(
        geo(0.01, 0.0),
        PositionUncertainty::EnuCovarianceM2(covariance),
        &fence,
        options,
    )
    .expect("edge probability");
    assert_close(edge_probability, 0.5, 0.03);

    let corner_probability = containment_probability_with_options(
        geo(0.0, 0.0),
        PositionUncertainty::EnuCovarianceM2(covariance),
        &fence,
        options,
    )
    .expect("corner probability");
    assert_close(corner_probability, 0.25, 0.03);
}

#[test]
fn zero_uncertainty_boundary_probability_matches_containment() {
    let fence = Fence::new([
        geo(0.0, 0.0),
        geo(0.0, 0.02),
        geo(0.02, 0.02),
        geo(0.02, 0.0),
    ])
    .expect("fence");
    let point = geo(0.0, 0.0);
    assert!(containment(point, &fence).expect("containment"));

    let default_probability =
        containment_probability(point, PositionUncertainty::CepRadiusM(0.0), &fence)
            .expect("default probability");
    assert_close(default_probability, 1.0, 0.0);

    let quadrature_probability = containment_probability_with_options(
        point,
        PositionUncertainty::CepRadiusM(0.0),
        &fence,
        ProbabilityOptions {
            method: ProbabilityMethod::PlanarQuadrature,
        },
    )
    .expect("quadrature probability");
    assert_close(quadrature_probability, 1.0, 0.0);
}

#[test]
fn dateline_equivalent_closing_vertex_is_removed() {
    let fence = Fence::new([
        geo(0.0, 180.0),
        geo(0.0, 179.9),
        geo(0.1, 179.9),
        geo(0.0, -180.0),
    ])
    .expect("fence");
    assert_eq!(fence.edge_count(), 3);
}

#[test]
fn ambiguous_hemisphere_fence_is_rejected() {
    let fence = Fence::new([
        geo(0.0, 0.0),
        geo(0.0, 90.0),
        geo(0.0, 180.0),
        geo(0.0, -90.0),
    ]);
    assert!(matches!(
        fence,
        Err(GeofenceError::InvalidInput {
            field: "vertices",
            reason: "interior orientation is ambiguous",
        })
    ));
}

#[test]
fn repeated_nonclosing_vertices_are_rejected() {
    let too_few_distinct =
        Fence::new([geo(0.0, 0.0), geo(0.0, 0.01), geo(0.0, 0.0), geo(0.0, 0.01)]);
    assert!(matches!(
        too_few_distinct,
        Err(GeofenceError::TooFewVertices)
    ));

    let repeated_with_three_distinct = Fence::new([
        geo(0.0, 0.0),
        geo(0.0, 0.01),
        geo(0.01, 0.01),
        geo(0.0, 0.01),
    ]);
    assert!(matches!(
        repeated_with_three_distinct,
        Err(GeofenceError::InvalidInput {
            field: "vertices",
            reason: "vertices must be distinct",
        })
    ));
}

#[test]
fn crossing_reports_boolean_entry_and_exit() {
    let fence = Fence::new([
        geo(-0.01, 0.0),
        geo(-0.01, 0.02),
        geo(0.01, 0.02),
        geo(0.01, 0.0),
    ])
    .expect("fence");
    let positions = [
        east_from(0.0, 0.0, -20.0),
        east_from(0.0, 0.0, 20.0),
        east_from(0.0, 0.0, -20.0),
    ];
    let events = crossing(&positions, &fence).expect("events");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].kind, CrossingKind::Entered);
    assert_eq!(events[1].kind, CrossingKind::Left);
}

#[test]
fn degenerate_covariance_at_corner_is_not_overconfident() {
    let fence = Fence::new([
        geo(0.0, 0.0),
        geo(0.0, 0.01),
        geo(0.01, 0.01),
        geo(0.01, 0.0),
    ])
    .expect("fence");
    let covariance = [[100.0, 100.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 0.0]];
    let probability = containment_probability(
        geo(0.0, 0.0),
        PositionUncertainty::EnuCovarianceM2(covariance),
        &fence,
    )
    .expect("probability");
    assert!(
        (0.45..=0.55).contains(&probability),
        "probability={probability}"
    );
}
