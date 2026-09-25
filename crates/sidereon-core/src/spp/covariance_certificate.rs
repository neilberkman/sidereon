use super::interval_certificate::Interval;

/// Independently enclose `(HᵀWH)⁻¹` from interval design rows and weights.
/// The cofactor inverse is intentionally separate from the production solver.
pub(super) fn inverse_interval(
    h_rows: &[[Interval; 4]],
    weights: &[Interval],
) -> [[Interval; 4]; 4] {
    assert!(!h_rows.is_empty() && h_rows.len() == weights.len());
    assert!(weights.iter().all(|weight| weight.lower() > 0.0));

    let zero = Interval::point(0.0);
    let mut normal = [[zero; 4]; 4];
    for (row, weight) in h_rows.iter().zip(weights) {
        for row_index in 0..4 {
            for column_index in 0..4 {
                normal[row_index][column_index] = normal[row_index][column_index]
                    .add(weight.mul(row[row_index]).mul(row[column_index]));
            }
        }
    }

    let determinant_interval = determinant(&normal);
    assert!(
        !determinant_interval.contains(0.0),
        "normal determinant interval contains zero"
    );

    let mut cofactors = [[zero; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            let mut minor = [[zero; 3]; 3];
            let mut minor_row = 0;
            for source_row in 0..4 {
                if source_row == row {
                    continue;
                }
                let mut minor_column = 0;
                for source_column in 0..4 {
                    if source_column == column {
                        continue;
                    }
                    minor[minor_row][minor_column] = normal[source_row][source_column];
                    minor_column += 1;
                }
                minor_row += 1;
            }
            let cofactor = determinant(&minor);
            cofactors[row][column] = if (row + column) % 2 == 0 {
                cofactor
            } else {
                cofactor.mul(Interval::point(-1.0))
            };
        }
    }

    let mut inverse = [[zero; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            inverse[row][column] = cofactors[column][row].div(determinant_interval);
        }
    }
    inverse
}

/// Binary64 values that can round to this finite binary32 value under
/// round-to-nearest, ties-to-even. Closed endpoints conservatively include ties.
pub(super) fn f32_quantization_cell(value: f32) -> Interval {
    assert!(value.is_finite());
    if value == 0.0 {
        let half_min_subnormal = f32::from_bits(1) as f64 * 0.5;
        return if value.is_sign_negative() {
            Interval::new(-half_min_subnormal, 0.0)
        } else {
            Interval::new(0.0, half_min_subnormal)
        };
    }

    let center = f64::from(value);
    let previous = adjacent_f32(value, false);
    let next = adjacent_f32(value, true);
    let lower = match previous {
        Some(adjacent) => (f64::from(adjacent) + center) * 0.5,
        None => center - (f64::from(next.expect("finite f32 has a successor")) - center) * 0.5,
    };
    let upper = match next {
        Some(adjacent) => (center + f64::from(adjacent)) * 0.5,
        None => {
            center + (center - f64::from(previous.expect("finite f32 has a predecessor"))) * 0.5
        }
    };
    Interval::new(lower, upper)
}

fn adjacent_f32(value: f32, upward: bool) -> Option<f32> {
    let bits = value.to_bits();
    let next_bits = if value == 0.0 {
        if upward {
            1
        } else {
            0x8000_0001
        }
    } else if (value > 0.0) == upward {
        bits.checked_add(1)?
    } else {
        bits.checked_sub(1)?
    };
    let adjacent = f32::from_bits(next_bits);
    adjacent.is_finite().then_some(adjacent)
}

fn determinant<const SIZE: usize>(matrix: &[[Interval; SIZE]; SIZE]) -> Interval {
    fn accumulate<const SIZE: usize>(
        matrix: &[[Interval; SIZE]; SIZE],
        row: usize,
        used_columns: usize,
        product: Interval,
        sum: &mut Option<Interval>,
    ) {
        if row == SIZE {
            *sum = Some(match *sum {
                Some(current) => current.add(product),
                None => product,
            });
            return;
        }
        for column in 0..SIZE {
            let bit = 1_usize << column;
            if used_columns & bit != 0 {
                continue;
            }
            let inversions = (used_columns >> (column + 1)).count_ones();
            let term = product.mul(matrix[row][column]);
            let term = if inversions.is_multiple_of(2) {
                term
            } else {
                term.mul(Interval::point(-1.0))
            };
            accumulate(matrix, row + 1, used_columns | bit, term, sum);
        }
    }

    assert!(SIZE > 0 && SIZE <= 4);
    let mut sum = None;
    accumulate(matrix, 0, 0, Interval::point(1.0), &mut sum);
    sum.expect("nonempty determinant")
}

#[cfg(test)]
mod tests {
    use super::{f32_quantization_cell, inverse_interval};
    use crate::spp::interval_certificate::Interval;

    fn points(rows: &[[f64; 4]]) -> Vec<[Interval; 4]> {
        rows.iter().map(|row| row.map(Interval::point)).collect()
    }

    fn unit_weights(count: usize) -> Vec<Interval> {
        vec![Interval::point(1.0); count]
    }

    #[test]
    fn cardinal_design_has_expected_diagonal_inverse() {
        let rows = points(&[
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let inverse = inverse_interval(&rows, &unit_weights(rows.len()));

        for axis in 0..3 {
            assert!(inverse[axis][axis].contains(0.5));
        }
        assert!(inverse[3][3].contains(1.0 / 6.0));
        for row in 0..4 {
            for column in 0..4 {
                if row != column {
                    assert!(inverse[row][column].contains(0.0));
                }
            }
        }
    }

    #[test]
    fn off_diagonal_spd_design_matches_analytic_inverse() {
        let rows = points(&[
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let inverse = inverse_interval(&rows, &unit_weights(rows.len()));
        let expected = [
            [1.0, -1.0, 1.0, 0.0],
            [-1.0, 2.0, -2.0, 0.0],
            [1.0, -2.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        for row in 0..4 {
            for column in 0..4 {
                assert!(inverse[row][column].contains(expected[row][column]));
            }
        }
    }

    #[test]
    fn singular_design_and_nonpositive_weights_are_refused() {
        let singular = points(&[
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]);
        assert!(std::panic::catch_unwind(|| {
            inverse_interval(&singular, &unit_weights(singular.len()))
        })
        .is_err());

        let full_rank = points(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let weights = [Interval::new(0.0, 1.0); 4];
        assert!(std::panic::catch_unwind(|| inverse_interval(&full_rank, &weights)).is_err());
    }

    #[test]
    fn f32_quantization_cells_cover_zero_subnormal_and_normal_values() {
        let min_subnormal = f32::from_bits(1);
        let negative_min_subnormal = f32::from_bits(0x8000_0001);
        let cases = [
            (0.0_f32, 0.0, f64::from(min_subnormal) * 0.5),
            (-0.0_f32, -f64::from(min_subnormal) * 0.5, 0.0),
            (
                min_subnormal,
                f64::from(min_subnormal) * 0.5,
                f64::from(f32::from_bits(2)) * 0.5 + f64::from(min_subnormal) * 0.5,
            ),
            (
                negative_min_subnormal,
                (f64::from(f32::from_bits(0x8000_0002)) + f64::from(negative_min_subnormal)) * 0.5,
                -f64::from(min_subnormal) * 0.5,
            ),
            (
                1.5_f32,
                (1.5_f64 + f64::from(f32::from_bits(1.5_f32.to_bits() - 1))) * 0.5,
                (1.5_f64 + f64::from(f32::from_bits(1.5_f32.to_bits() + 1))) * 0.5,
            ),
        ];

        for (value, expected_lower, expected_upper) in cases {
            let cell = f32_quantization_cell(value);
            assert_eq!(cell.lower(), expected_lower);
            assert_eq!(cell.upper(), expected_upper);
            assert!(cell.contains(f64::from(value)));
        }

        let negative_zero = f32_quantization_cell(-0.0);
        let positive_zero = f32_quantization_cell(0.0);
        assert!(negative_zero.upper() == 0.0 && negative_zero.lower() < 0.0);
        assert!(positive_zero.lower() == 0.0 && positive_zero.upper() > 0.0);
    }
}
