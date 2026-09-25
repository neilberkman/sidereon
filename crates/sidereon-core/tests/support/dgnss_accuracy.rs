use sidereon_core::constants::{C_M_S, OMEGA_E_DOT_RAD_S};
use sidereon_core::dgnss::{CodeObservation, PositionSolution};
use sidereon_core::ephemeris::Sp3;

#[allow(dead_code)]
#[path = "../../src/spp/interval_certificate.rs"]
mod interval_certificate;

use interval_certificate::Interval;

fn magnitude(value: Interval) -> f64 {
    value.lower().abs().max(value.upper().abs())
}

fn norm_upper(values: impl IntoIterator<Item = f64>) -> f64 {
    let squared = values.into_iter().fold(Interval::point(0.0), |sum, value| {
        sum.add(Interval::point(value).square())
    });
    Interval::new(squared.lower().max(0.0), squared.upper())
        .sqrt()
        .upper()
}

fn geometric_range(satellite: [f64; 3], receiver: [Interval; 3]) -> Interval {
    let offsets =
        std::array::from_fn::<_, 3, _>(|axis| Interval::point(satellite[axis]).sub(receiver[axis]));
    let distance = offsets[0]
        .square()
        .add(offsets[1].square())
        .add(offsets[2].square())
        .sqrt();
    let rotation = Interval::point(OMEGA_E_DOT_RAD_S)
        .mul(
            Interval::point(satellite[0])
                .mul(receiver[1])
                .sub(Interval::point(satellite[1]).mul(receiver[0])),
        )
        .div(Interval::point(C_M_S));
    distance.add(rotation)
}

fn design_row(satellite: [f64; 3], receiver: [Interval; 3]) -> [Interval; 4] {
    let offsets =
        std::array::from_fn::<_, 3, _>(|axis| receiver[axis].sub(Interval::point(satellite[axis])));
    let distance = offsets[0]
        .square()
        .add(offsets[1].square())
        .add(offsets[2].square())
        .sqrt();
    assert!(distance.lower() > 0.0);
    let rotation = Interval::point(OMEGA_E_DOT_RAD_S).div(Interval::point(C_M_S));
    [
        offsets[0]
            .div(distance)
            .sub(rotation.mul(Interval::point(satellite[1]))),
        offsets[1]
            .div(distance)
            .add(rotation.mul(Interval::point(satellite[0]))),
        offsets[2].div(distance),
        Interval::point(1.0),
    ]
}

fn inverse_normal(rows: &[[Interval; 4]]) -> [[Interval; 4]; 4] {
    let zero = Interval::point(0.0);
    let mut augmented = [[zero; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = rows
                .iter()
                .fold(zero, |sum, design| sum.add(design[row].mul(design[column])));
        }
        augmented[row][row + 4] = Interval::point(1.0);
    }
    for pivot in 0..4 {
        let divisor = augmented[pivot][pivot];
        assert!(!divisor.contains(0.0), "geometry inverse is not certified");
        for column in 0..8 {
            augmented[pivot][column] = augmented[pivot][column].div(divisor);
        }
        let pivot_row = augmented[pivot];
        for row in 0..4 {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for column in 0..8 {
                augmented[row][column] = augmented[row][column].sub(factor.mul(pivot_row[column]));
            }
        }
    }
    std::array::from_fn(|row| std::array::from_fn(|column| augmented[row][column + 4]))
}

#[test]
fn interval_geometry_inverse_encloses_the_orthogonal_design_inverse() {
    let rows = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ]
    .map(|row| row.map(Interval::point));
    let inverse = inverse_normal(&rows);
    for row in 0..4 {
        for column in 0..4 {
            assert!(inverse[row][column].contains(if row == column { 0.25 } else { 0.0 }));
        }
    }
}

#[test]
#[should_panic(expected = "geometry inverse is not certified")]
fn interval_geometry_inverse_refuses_a_rank_deficient_design() {
    inverse_normal(&[[Interval::point(1.0); 4]; 4]);
}

pub(super) fn assert_clean_solution(
    source: &Sp3,
    base: [f64; 3],
    rover: [f64; 3],
    base_observations: &[CodeObservation],
    rover_observations: &[CodeObservation],
    result: &PositionSolution,
) {
    let solution = &result.solution;
    let radius_m = 1.0;
    let position = solution.position.as_array();
    let receiver_region =
        rover.map(|value| Interval::point(value).add(Interval::new(-radius_m, radius_m)));
    for axis in 0..3 {
        assert!(
            receiver_region[axis].contains(position[axis]),
            "solution left the a priori box"
        );
    }
    let truth_clock_m = C_M_S * -2.0e-6 - C_M_S * 1.0e-6;
    let returned_clock_m = Interval::new(
        solution.rx_clock_s.next_down(),
        solution.rx_clock_s.next_up(),
    )
    .mul(Interval::point(C_M_S));
    assert_eq!(base_observations.len(), rover_observations.len());
    assert_eq!(solution.used_sats.len(), rover_observations.len());
    assert_eq!(solution.residuals_m.len(), rover_observations.len());

    let mut truth_errors = Vec::new();
    let mut evaluation_errors = Vec::new();
    let mut rows = Vec::new();
    let mut variance_min = f64::INFINITY;
    let mut variance_max = 0.0_f64;
    for (index, (base_observation, rover_observation)) in
        base_observations.iter().zip(rover_observations).enumerate()
    {
        assert_eq!(
            base_observation.satellite_id,
            rover_observation.satellite_id
        );
        let satellite = super::sat_from_token(&rover_observation.satellite_id);
        assert_eq!(solution.used_sats[index], satellite);
        let (base_satellite, base_clock, _) = super::exact_placed_state(
            source,
            satellite,
            super::T_RX_J2000_S,
            base_observation.pseudorange_m,
        );
        let base_model = geometric_range(base_satellite, base.map(Interval::point))
            .sub(Interval::point(C_M_S).mul(Interval::point(base_clock)));
        let correction = Interval::point(base_observation.pseudorange_m).sub(base_model);
        let corrected = Interval::point(rover_observation.pseudorange_m).sub(correction);
        let (rover_satellite, rover_clock, variance) = super::exact_placed_state(
            source,
            satellite,
            super::T_RX_J2000_S,
            rover_observation.pseudorange_m,
        );
        assert!(variance.is_finite() && variance >= 0.0);
        variance_min = variance_min.min(variance);
        variance_max = variance_max.max(variance);
        let satellite_clock_m = Interval::point(C_M_S).mul(Interval::point(rover_clock));
        let truth_model = geometric_range(rover_satellite, rover.map(Interval::point))
            .add(Interval::point(truth_clock_m))
            .sub(satellite_clock_m);
        truth_errors.push(magnitude(corrected.sub(truth_model)));
        let returned_model = geometric_range(rover_satellite, position.map(Interval::point))
            .add(returned_clock_m)
            .sub(satellite_clock_m);
        let returned_residual = corrected.sub(returned_model);
        assert!(
            returned_residual.contains(solution.residuals_m[index]),
            "{satellite}: reported residual is outside the independent evaluation enclosure"
        );
        evaluation_errors.push(returned_residual.width());
        rows.push(design_row(rover_satellite, receiver_region));
    }

    let constant_variance = Interval::point(0.3)
        .square()
        .add(Interval::point(5.0).square())
        .add(Interval::point(3.0).square());
    let phase_variance = Interval::point(0.003).square();
    let ratio_squared = Interval::point(300.0).square();
    let minimum_variance = Interval::point(variance_min)
        .add(constant_variance)
        .add(phase_variance.add(phase_variance).mul(ratio_squared));
    let maximum_variance = Interval::point(variance_max).add(constant_variance).add(
        phase_variance
            .add(phase_variance.div(Interval::point(1.0 / 16.0)))
            .mul(ratio_squared),
    );
    let optimal_residual_norm = Interval::point(norm_upper(truth_errors.iter().copied()))
        .mul(maximum_variance.div(minimum_variance).sqrt());
    let evaluation_norm = Interval::point(norm_upper(evaluation_errors));
    let reported_norm = norm_upper(solution.residuals_m.iter().copied());
    assert!(
        reported_norm <= optimal_residual_norm.add(evaluation_norm).upper(),
        "clean residual norm {reported_norm:e} exceeds the input-derived envelope {:e}",
        optimal_residual_norm.add(evaluation_norm).upper()
    );

    let ideal_residual_norm = optimal_residual_norm.add(evaluation_norm.mul(Interval::point(2.0)));
    let inverse = inverse_normal(&rows);
    for component in 0..4 {
        let mut bound = Interval::point(0.0);
        for (design, truth_error) in rows.iter().zip(&truth_errors) {
            let gain = (0..4).fold(Interval::point(0.0), |sum, column| {
                sum.add(inverse[component][column].mul(design[column]))
            });
            bound = bound.add(
                Interval::point(magnitude(gain))
                    .mul(Interval::point(*truth_error).add(ideal_residual_norm)),
            );
        }
        assert!(
            bound.upper() < radius_m,
            "geometry bound does not contract the input box"
        );
        let difference = if component < 3 {
            Interval::point(position[component]).sub(Interval::point(rover[component]))
        } else {
            returned_clock_m.sub(Interval::point(truth_clock_m))
        };
        assert!(
            magnitude(difference) <= bound.upper(),
            "component {component}: error enclosure {difference:?}, bound {:e} m",
            bound.upper()
        );
    }
    for axis in 0..3 {
        let baseline = Interval::point(position[axis]).sub(Interval::point(base[axis]));
        assert!(
            baseline.contains(result.baseline_vector_m[axis]),
            "baseline component {axis} is outside its arithmetic enclosure"
        );
    }
    let baseline = result.baseline_vector_m.map(Interval::point);
    let length = baseline[0]
        .square()
        .add(baseline[1].square())
        .add(baseline[2].square())
        .sqrt();
    assert!(
        length.contains(result.baseline_m),
        "baseline length is outside its arithmetic enclosure"
    );
}
