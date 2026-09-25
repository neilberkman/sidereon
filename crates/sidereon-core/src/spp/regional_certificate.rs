//! Input-derived bounds for the four-parameter weighted SPP iteration.
//!
//! The arithmetic enclosures require IEEE 754 binary64 operations in
//! round-to-nearest, ties-to-even mode, with no reassociation, fused-operation
//! contraction, or fast-math transformations, and gradual underflow (no
//! flush-to-zero or denormals-are-zero mode). Every intermediate used by the
//! certificate must remain finite. `f64` basic arithmetic and `f64::sqrt` must
//! provide correctly rounded results; Rust documents `sqrt` as the rounded
//! infinite-precision square root. Targets or compiler/runtime modes that do
//! not provide these guarantees must reject this certificate rather than
//! treating its bounds as valid.
//!
//! `gamma` uses `f64::EPSILON` (twice the binary64 unit roundoff), conservatively
//! bounding the relative error for its stated operation count. The additive
//! `operations * f64::MIN_POSITIVE` term in `arithmetic_error` deliberately
//! exceeds the gradual-underflow rounding allowance of at most half a minimum
//! subnormal per operation. These bounds assume no overflow in the operations
//! being enclosed.

use super::Selection;

/// Bounds for one satellite throughout the receiver/clock ball.
pub(super) struct SatelliteRegion {
    /// Euclidean norm bound for a design row's directional derivative.
    pub design_derivative: f64,
    /// Norm bound for the Sagnac and atmospheric range-gradient terms omitted
    /// from the geometric design row.
    pub correction_gradient: f64,
    /// Upper bound for the inverse measurement variance.
    pub weight_max: f64,
    /// Norm bound for the inverse-variance gradient.
    pub weight_gradient: f64,
    /// Euclidean error enclosure for the computed centre design row.
    pub centre_design_error: f64,
    /// Absolute error enclosure for the computed centre weight.
    pub centre_weight_error: f64,
    /// Absolute error enclosure for the computed centre residual, metres.
    pub centre_residual_error: f64,
}

/// A derivative certificate for `T(x)=x+(HᵀWH)⁻¹HᵀWv` over the supplied ball.
pub(super) struct RegionalCertificate {
    pub contraction: f64,
    pub gain_norm: f64,
    step_error: f64,
}

fn upper(value: f64) -> f64 {
    assert!(value.is_finite() && value >= 0.0);
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let result = f64::from_bits(value.to_bits() + 1);
    assert!(result.is_finite());
    result
}

fn add(left: f64, right: f64) -> f64 {
    upper(left + right)
}

fn multiply(left: f64, right: f64) -> f64 {
    upper(left * right)
}

fn divide(numerator: f64, denominator: f64) -> f64 {
    assert!(denominator > 0.0);
    upper(numerator / denominator)
}

fn root(value: f64) -> f64 {
    upper(value.sqrt())
}

fn lower_difference(left: f64, right: f64) -> f64 {
    assert!(left.is_finite() && right.is_finite() && left > right);
    let result = left - right;
    assert!(result > 0.0);
    let result = f64::from_bits(result.to_bits() - 1);
    assert!(result > 0.0);
    result
}

fn norm(values: impl IntoIterator<Item = f64>) -> f64 {
    root(values.into_iter().fold(0.0, |sum, value| {
        add(sum, multiply(value.abs(), value.abs()))
    }))
}

fn gamma(operations: usize) -> f64 {
    assert!(operations <= 1_000_000);
    let product = multiply(operations as f64, f64::EPSILON);
    divide(product, lower_difference(1.0, product))
}

fn arithmetic_error(operations: usize, magnitude: f64) -> f64 {
    add(
        multiply(gamma(operations), magnitude),
        multiply(operations as f64, f64::MIN_POSITIVE),
    )
}

fn inverse_norm_certificate(matrix: &[[f64; 4]; 4]) -> f64 {
    let rows: Vec<Vec<f64>> = matrix.iter().map(|row| row.to_vec()).collect();
    let inverse = crate::astro::math::linear::invert_symmetric_pd(&rows)
        .expect("full-rank reference normal matrix");
    let mut residual_entries = Vec::with_capacity(16);
    for row in 0..4 {
        for column in 0..4 {
            let mut product = 0.0;
            let mut magnitude = 0.0;
            for inner in 0..4 {
                product += matrix[row][inner] * inverse[inner][column];
                magnitude = add(
                    magnitude,
                    multiply(matrix[row][inner].abs(), inverse[inner][column].abs()),
                );
            }
            let identity = if row == column { 1.0 } else { 0.0 };
            let evaluation_error = arithmetic_error(9, add(identity, magnitude));
            residual_entries.push(add((identity - product).abs(), evaluation_error));
        }
    }
    let residual_norm = norm(residual_entries);
    assert!(residual_norm < 1.0, "uncertified reference inverse");
    divide(
        norm(inverse.iter().flatten().copied()),
        lower_difference(1.0, residual_norm),
    )
}

/// Certify the complete state-dependent weighted iteration, including changes
/// to its design matrix, weights and gain. All supplied per-satellite bounds
/// must hold throughout the same convex ball and fixed satellite selection.
/// The positive bound calculations round outward after every operation.
pub(super) fn certify(
    centre: &Selection,
    centre_step: [f64; 4],
    regions: &[SatelliteRegion],
    radius_m: f64,
) -> RegionalCertificate {
    assert!(radius_m.is_finite() && radius_m > 0.0);
    bounds(centre, centre_step, regions, radius_m)
}

/// Bound the error of a computed weighted step relative to the real-arithmetic
/// step at the same endpoint, including the supplied model-input enclosures.
pub(super) fn step_evaluation_error(
    endpoint: &Selection,
    step: [f64; 4],
    regions: &[SatelliteRegion],
) -> f64 {
    bounds(endpoint, step, regions, 0.0).step_error
}

/// The same contractive map is evaluated at both endpoints. Its two step
/// defects and their evaluation errors bound endpoint separation independently
/// of the observed separation. The oracle model-agreement gates are separate.
pub(super) fn endpoint_distance_bound(
    reference_step: [f64; 4],
    solution_step: [f64; 4],
    reference_evaluation_error: f64,
    solution_evaluation_error: f64,
    contraction: f64,
) -> f64 {
    assert!((0.0..0.5).contains(&contraction));
    divide(
        add(
            add(norm(reference_step), reference_evaluation_error),
            add(norm(solution_step), solution_evaluation_error),
        ),
        lower_difference(1.0, contraction),
    )
}

/// Enclose the four-component distance used to check membership of the fixed
/// receiver/clock ball. This check does not define the comparison tolerance.
pub(super) fn endpoint_distance(left: [f64; 4], right: [f64; 4]) -> f64 {
    norm(left.into_iter().zip(right).map(|(left, right)| {
        add(
            (left - right).abs(),
            arithmetic_error(1, add(left.abs(), right.abs())),
        )
    }))
}

fn bounds(
    centre: &Selection,
    centre_step: [f64; 4],
    regions: &[SatelliteRegion],
    radius_m: f64,
) -> RegionalCertificate {
    let count = centre.used.len();
    assert!((4..=64).contains(&count));
    assert_eq!(centre.lines_of_sight.len(), count);
    assert_eq!(centre.weights.len(), count);
    assert_eq!(centre.residuals_m.len(), count);
    assert_eq!(regions.len(), count);
    assert!(radius_m.is_finite() && radius_m >= 0.0);
    assert!(centre_step.iter().all(|value| value.is_finite()));

    let design: Vec<[f64; 4]> = centre
        .lines_of_sight
        .iter()
        .map(|los| [-los.e_x, -los.e_y, -los.e_z, 1.0])
        .collect();
    let mut normal = [[0.0; 4]; 4];
    let mut normal_magnitudes = [[0.0; 4]; 4];
    let mut right_hand = [0.0; 4];
    let mut right_hand_magnitudes = [0.0; 4];
    for (index, row) in design.iter().enumerate() {
        let weight = centre.weights[index];
        let residual = centre.residuals_m[index];
        assert!(weight.is_finite() && weight > 0.0 && residual.is_finite());
        assert!(row.iter().all(|value| value.is_finite()));
        for row_index in 0..4 {
            right_hand[row_index] += row[row_index] * weight * residual;
            right_hand_magnitudes[row_index] = add(
                right_hand_magnitudes[row_index],
                multiply(multiply(row[row_index].abs(), weight), residual.abs()),
            );
            for column_index in 0..4 {
                normal[row_index][column_index] += row[row_index] * weight * row[column_index];
                normal_magnitudes[row_index][column_index] = add(
                    normal_magnitudes[row_index][column_index],
                    multiply(
                        multiply(row[row_index].abs(), weight),
                        row[column_index].abs(),
                    ),
                );
            }
        }
    }
    let normal_formation_error = norm(
        normal_magnitudes
            .iter()
            .flatten()
            .map(|magnitude| arithmetic_error(3 * count, *magnitude)),
    );
    let mut normal_change = normal_formation_error;
    let mut row_norms = Vec::with_capacity(count);
    let mut weighted_design_changes = Vec::with_capacity(count);
    let mut weighted_design_columns = Vec::with_capacity(count);
    let mut residual_changes = Vec::with_capacity(count);

    for (index, region) in regions.iter().enumerate() {
        for value in [
            region.design_derivative,
            region.correction_gradient,
            region.weight_max,
            region.weight_gradient,
            region.centre_design_error,
            region.centre_weight_error,
            region.centre_residual_error,
        ] {
            assert!(value.is_finite() && value >= 0.0);
        }
        assert!(region.weight_max > 0.0);
        let row_norm = norm(design[index]);
        let design_change = add(
            region.centre_design_error,
            multiply(radius_m, region.design_derivative),
        );
        let weight_change = add(
            region.centre_weight_error,
            multiply(radius_m, region.weight_gradient),
        );
        let regional_row_norm = add(row_norm, design_change);
        normal_change = add(
            normal_change,
            add(
                multiply(
                    weight_change,
                    multiply(regional_row_norm, regional_row_norm),
                ),
                multiply(
                    multiply(centre.weights[index], design_change),
                    add(multiply(2.0, row_norm), design_change),
                ),
            ),
        );
        weighted_design_changes.push(add(
            multiply(weight_change, row_norm),
            multiply(region.weight_max, design_change),
        ));
        weighted_design_columns.push(multiply(region.weight_max, regional_row_norm));
        residual_changes.push(add(
            region.centre_residual_error,
            multiply(radius_m, add(regional_row_norm, region.correction_gradient)),
        ));
        row_norms.push(regional_row_norm);
    }

    let centre_inverse_norm = inverse_norm_certificate(&normal);
    let inverse_perturbation = multiply(centre_inverse_norm, normal_change);
    assert!(
        inverse_perturbation < 1.0,
        "normal matrix not certified throughout ball"
    );
    let inverse_norm = divide(
        centre_inverse_norm,
        lower_difference(1.0, inverse_perturbation),
    );
    let weighted_design_norm = norm(weighted_design_columns);
    let gain_norm = multiply(inverse_norm, weighted_design_norm);
    let centre_step_norm = norm(centre_step);
    let mut step_residual = [0.0; 4];
    for row in 0..4 {
        let mut product = 0.0;
        let mut magnitude = 0.0;
        for column in 0..4 {
            product += normal[row][column] * centre_step[column];
            magnitude = add(
                magnitude,
                multiply(normal[row][column].abs(), centre_step[column].abs()),
            );
        }
        step_residual[row] = add(
            (right_hand[row] - product).abs(),
            add(
                arithmetic_error(9, add(right_hand[row].abs(), magnitude)),
                arithmetic_error(3 * count, right_hand_magnitudes[row]),
            ),
        );
    }
    let step_error = multiply(
        inverse_norm,
        add(
            add(
                multiply(
                    norm(weighted_design_changes),
                    norm(centre.residuals_m.iter().copied()),
                ),
                multiply(weighted_design_norm, norm(residual_changes.iter().copied())),
            ),
            add(
                multiply(normal_change, centre_step_norm),
                norm(step_residual),
            ),
        ),
    );
    let step_max = add(centre_step_norm, step_error);
    let mut gain_variation = 0.0;
    for (index, region) in regions.iter().enumerate() {
        let orthogonal_residual = add(
            add(centre.residuals_m[index].abs(), residual_changes[index]),
            multiply(row_norms[index], step_max),
        );
        gain_variation = add(
            gain_variation,
            multiply(
                add(
                    multiply(region.design_derivative, region.weight_max),
                    multiply(row_norms[index], region.weight_gradient),
                ),
                orthogonal_residual,
            ),
        );
    }
    gain_variation = add(
        gain_variation,
        multiply(
            multiply(
                weighted_design_norm,
                norm(regions.iter().map(|region| region.design_derivative)),
            ),
            step_max,
        ),
    );
    let contraction = add(
        multiply(
            gain_norm,
            norm(regions.iter().map(|region| region.correction_gradient)),
        ),
        multiply(inverse_norm, gain_variation),
    );
    assert!(
        contraction < 0.5,
        "uncertified weighted iteration contraction: {contraction}"
    );
    RegionalCertificate {
        contraction,
        gain_norm,
        step_error,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        certify, endpoint_distance, endpoint_distance_bound, step_evaluation_error, SatelliteRegion,
    };
    use crate::dop::LineOfSight;
    use crate::id::{GnssSatelliteId, GnssSystem};
    use crate::spp::Selection;

    fn cardinal_selection() -> Selection {
        Selection {
            used: (1..=6)
                .map(|prn| GnssSatelliteId {
                    system: GnssSystem::Gps,
                    prn,
                })
                .collect(),
            rejected: Vec::new(),
            weights: vec![1.0; 6],
            variances_m2: vec![1.0; 6],
            lines_of_sight: vec![
                LineOfSight::new(1.0, 0.0, 0.0),
                LineOfSight::new(-1.0, 0.0, 0.0),
                LineOfSight::new(0.0, 1.0, 0.0),
                LineOfSight::new(0.0, -1.0, 0.0),
                LineOfSight::new(0.0, 0.0, 1.0),
                LineOfSight::new(0.0, 0.0, -1.0),
            ],
            residuals_m: vec![6.0, 8.0, 0.0, 4.0, 0.0, 6.0],
        }
    }

    fn exact_regions() -> Vec<SatelliteRegion> {
        (0..6)
            .map(|_| SatelliteRegion {
                design_derivative: 0.0,
                correction_gradient: 0.0,
                weight_max: 1.0,
                weight_gradient: 0.0,
                centre_design_error: 0.0,
                centre_weight_error: 0.0,
                centre_residual_error: 0.0,
            })
            .collect()
    }

    #[test]
    fn residual_enclosure_detects_a_perturbed_exact_step() {
        let selection = cardinal_selection();
        let regions = exact_regions();
        let exact_step = [1.0, 2.0, 3.0, 4.0];
        let perturbed_step = [1.0, 2.0, 3.0, 4.25];

        let error = step_evaluation_error(&selection, perturbed_step, &regions);

        assert!(error >= 0.25);
        assert!(step_evaluation_error(&selection, exact_step, &regions) < 0.25);
        let certificate = certify(&selection, exact_step, &regions, 1.0);
        assert!(certificate.gain_norm >= (5.0_f64 / 3.0).sqrt());
    }

    #[test]
    fn endpoint_distance_and_theorem_include_clock_component() {
        let membership_distance = endpoint_distance([0.0; 4], [3.0, 4.0, 0.0, 12.0]);
        let theorem_bound =
            endpoint_distance_bound([3.0, 0.0, 0.0, 4.0], [0.0, 12.0, 0.0, 0.0], 0.0, 0.0, 0.25);

        assert!(membership_distance >= 13.0);
        assert!(theorem_bound >= 68.0 / 3.0);
    }

    #[test]
    fn singular_centre_is_refused() {
        let mut selection = cardinal_selection();
        selection.lines_of_sight = vec![LineOfSight::new(1.0, 0.0, 0.0); 6];
        let refusal = std::panic::catch_unwind(|| {
            step_evaluation_error(&selection, [0.0; 4], &exact_regions())
        });

        assert!(refusal.is_err());
    }

    #[test]
    fn centre_uncertainty_survives_zero_radius_evaluation() {
        let selection = cardinal_selection();
        let exact_step = [1.0, 2.0, 3.0, 4.0];
        let exact_error = step_evaluation_error(&selection, exact_step, &exact_regions());
        let mut uncertain_regions = exact_regions();
        for region in &mut uncertain_regions {
            region.centre_residual_error = 0.5;
        }

        let uncertain_error = step_evaluation_error(&selection, exact_step, &uncertain_regions);

        assert!(uncertain_error > exact_error);
    }

    #[test]
    fn nonzero_region_bounds_cover_analytic_design_weight_and_correction_derivative() {
        let mut selection = cardinal_selection();
        for residual in &mut selection.residuals_m {
            *residual *= 1.0e-3;
        }
        let step = [1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3];
        let weight_slope = 0.02;
        let correction_slope = 0.01;
        let radius_m = 1.0e-3;
        let minimum_satellite_range_m = 10.0 - radius_m;
        let design_derivative_bound = 0.101;
        let weight_max = 1.000_021;
        assert!(design_derivative_bound > 1.0 / minimum_satellite_range_m);
        assert!(weight_max > 1.0 + weight_slope * radius_m);
        let regions: Vec<_> = (0..6)
            .map(|index| SatelliteRegion {
                design_derivative: design_derivative_bound,
                correction_gradient: if index == 0 { correction_slope } else { 0.0 },
                weight_max: if index == 0 { weight_max } else { 1.0 },
                weight_gradient: if index == 0 { weight_slope } else { 0.0 },
                centre_design_error: 0.0,
                centre_weight_error: 0.0,
                centre_residual_error: 0.0,
            })
            .collect();
        let certificate = certify(&selection, step, &regions, radius_m);

        let directional_design_derivatives = [
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
        ];
        let inverse_normal_diagonal = [0.5, 0.5, 0.5, 1.0 / 6.0];
        let mut design_residual_term = [0.0; 4];
        let mut weight_residual_term = [0.0; 4];
        let mut design_step_term = [0.0; 4];
        let mut correction_term = [0.0; 4];
        for index in 0..6 {
            let row = [
                -selection.lines_of_sight[index].e_x,
                -selection.lines_of_sight[index].e_y,
                -selection.lines_of_sight[index].e_z,
                1.0,
            ];
            let derivative = directional_design_derivatives[index];
            let orthogonal_residual = selection.residuals_m[index]
                - row
                    .iter()
                    .zip(step)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum::<f64>();
            let derivative_times_step = derivative
                .iter()
                .zip(step)
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>();
            let weight_derivative = if index == 0 { weight_slope } else { 0.0 };
            let correction_derivative = if index == 0 { correction_slope } else { 0.0 };
            for component in 0..4 {
                design_residual_term[component] += derivative[component] * orthogonal_residual;
                weight_residual_term[component] +=
                    row[component] * weight_derivative * orthogonal_residual;
                design_step_term[component] -= row[component] * derivative_times_step;
                correction_term[component] += row[component] * correction_derivative;
            }
        }
        let analytic_directional_derivative: [f64; 4] = std::array::from_fn(|component| {
            inverse_normal_diagonal[component]
                * (design_residual_term[component]
                    + weight_residual_term[component]
                    + design_step_term[component]
                    + correction_term[component])
        });
        let analytic_derivative_norm = analytic_directional_derivative
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();

        assert!(design_residual_term.iter().any(|value| *value != 0.0));
        assert!(weight_residual_term.iter().any(|value| *value != 0.0));
        assert!(correction_term.iter().any(|value| *value != 0.0));
        assert!(certificate.step_error > 0.0);
        assert!(analytic_derivative_norm <= certificate.contraction);
    }

    #[test]
    fn refuses_normal_perturbation_bound_that_does_not_preserve_rank() {
        let selection = cardinal_selection();
        let mut regions = exact_regions();
        for region in &mut regions {
            region.design_derivative = 10.0;
        }
        let refusal =
            std::panic::catch_unwind(|| certify(&selection, [1.0, 2.0, 3.0, 4.0], &regions, 0.1));

        let Err(payload) = refusal else {
            panic!("large design variation must refuse the normal-matrix certificate");
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(message.starts_with("normal matrix not certified throughout ball"));
    }

    #[test]
    fn refuses_correction_gradient_bound_that_exceeds_contraction_cap() {
        let selection = cardinal_selection();
        let mut regions = exact_regions();
        regions[0].correction_gradient = 0.2;
        let refusal = std::panic::catch_unwind(|| {
            certify(&selection, [1.0, 2.0, 3.0, 4.0], &regions, 1.0e-6)
        });

        let Err(payload) = refusal else {
            panic!("large correction gradient must refuse the contraction certificate");
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(message.starts_with("uncertified weighted iteration contraction:"));
    }
}
