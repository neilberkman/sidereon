//! Shared CCSDS Navigation Data Message 6x6 covariance-block primitives.

use crate::astro::covariance::{Covariance6, Covariance6Error, Mat6};
use crate::format::fmtnum::fmt_num;
use crate::validate::{self, FieldError};

use super::FieldMap;

/// CCSDS 6x6 state covariance lower-triangle KVN keys in row-major order.
pub(crate) const COVARIANCE6_KEYS: [&str; 21] = [
    "CX_X",
    "CY_X",
    "CY_Y",
    "CZ_X",
    "CZ_Y",
    "CZ_Z",
    "CX_DOT_X",
    "CX_DOT_Y",
    "CX_DOT_Z",
    "CX_DOT_X_DOT",
    "CY_DOT_X",
    "CY_DOT_Y",
    "CY_DOT_Z",
    "CY_DOT_X_DOT",
    "CY_DOT_Y_DOT",
    "CZ_DOT_X",
    "CZ_DOT_Y",
    "CZ_DOT_Z",
    "CZ_DOT_X_DOT",
    "CZ_DOT_Y_DOT",
    "CZ_DOT_Z_DOT",
];

/// Matrix positions matching [`COVARIANCE6_KEYS`] row-major lower-triangle order.
pub(crate) const COVARIANCE6_POSITIONS: [(usize, usize); 21] = [
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
];

/// Read a CCSDS 6x6 lower-triangle covariance block from KVN fields.
pub(crate) fn read_covariance6(map: &FieldMap) -> Result<Covariance6, FieldError> {
    let mut lower = [0.0_f64; 21];
    for (slot, key) in lower.iter_mut().zip(COVARIANCE6_KEYS) {
        let raw = map.get(key).ok_or(FieldError::Missing { field: key })?;
        *slot = validate::strict_f64(raw, key)?;
    }
    covariance6_from_lower_triangle(&lower)
}

/// Read the 21 lower-triangle values of a CCSDS 6x6 covariance block from KVN
/// fields exactly as stated, in [`COVARIANCE6_KEYS`] order. Each must be a
/// finite number; the matrix is not otherwise checked, since a matrix printed
/// to a few digits can fall short of positive semidefinite only through that
/// rounding.
pub(crate) fn read_lower_triangle6(map: &FieldMap) -> Result<[f64; 21], FieldError> {
    let mut lower = [0.0_f64; 21];
    for (slot, key) in lower.iter_mut().zip(COVARIANCE6_KEYS) {
        let raw = map.get(key).ok_or(FieldError::Missing { field: key })?;
        *slot = validate::strict_f64(raw, key)?;
    }
    Ok(lower)
}

/// The symmetric 6x6 matrix whose lower triangle is `lower`, in
/// [`COVARIANCE6_KEYS`] order.
pub(crate) fn mirror_lower_triangle6(lower: &[f64; 21]) -> Mat6 {
    let mut matrix: Mat6 = [[0.0_f64; 6]; 6];
    for ((row, col), value) in COVARIANCE6_POSITIONS.into_iter().zip(lower.iter().copied()) {
        matrix[row][col] = value;
        matrix[col][row] = value;
    }
    matrix
}

/// Build a validated [`Covariance6`] from its 21 lower-triangle values in
/// [`COVARIANCE6_KEYS`] order, mirroring each value across the diagonal.
pub(crate) fn covariance6_from_lower_triangle(
    lower: &[f64; 21],
) -> Result<Covariance6, FieldError> {
    let mut matrix: Mat6 = [[0.0_f64; 6]; 6];
    for ((row, col), value) in COVARIANCE6_POSITIONS.into_iter().zip(lower.iter().copied()) {
        matrix[row][col] = value;
        matrix[col][row] = value;
    }
    Covariance6::try_from_matrix(matrix).map_err(map_covariance6_error)
}

/// The lower-triangle values of a [`Covariance6`] in [`COVARIANCE6_KEYS`] order.
pub(crate) fn covariance6_lower_triangle(cov: &Covariance6) -> [f64; 21] {
    let matrix = cov.as_matrix();
    let mut lower = [0.0_f64; 21];
    for (slot, (row, col)) in lower.iter_mut().zip(COVARIANCE6_POSITIONS) {
        *slot = matrix[row][col];
    }
    lower
}

/// The unit CCSDS 502.0-B-3 tables 3-3, 4-3 and 8-7 give a 6x6 covariance
/// keyword: `km**2` for two position components, `km**2/s` for one position
/// and one velocity component, and `km**2/s**2` for two velocity components.
/// `None` for a keyword that is not one of [`COVARIANCE6_KEYS`].
pub(crate) fn covariance6_unit(key: &str) -> Option<&'static [&'static str]> {
    const POSITION: &[&str] = &["km**2"];
    const MIXED: &[&str] = &["km**2/s"];
    const VELOCITY: &[&str] = &["km**2/s**2"];
    let index = COVARIANCE6_KEYS
        .iter()
        .position(|candidate| *candidate == key)?;
    let (row, col) = COVARIANCE6_POSITIONS[index];
    Some(match (row >= 3, col >= 3) {
        (false, _) => POSITION,
        (true, false) => MIXED,
        (true, true) => VELOCITY,
    })
}

/// Write a CCSDS 6x6 lower-triangle covariance block as KVN lines.
pub(crate) fn write_covariance6(cov: &Covariance6) -> Vec<String> {
    covariance6_lower_triangle(cov)
        .into_iter()
        .zip(COVARIANCE6_KEYS)
        .map(|(value, key)| format!("{key} = {}", fmt_num(value)))
        .collect()
}

/// Map covariance validation failures into the shared field-error vocabulary.
///
/// Non-finite values map to [`FieldError::NonFinite`], asymmetric matrices map
/// to [`FieldError::OutOfRange`] because the symmetry relation is outside the
/// accepted tolerance, and non-PSD matrices map to [`FieldError::NotPositive`].
fn map_covariance6_error(error: Covariance6Error) -> FieldError {
    match error {
        Covariance6Error::NonFinite => FieldError::NonFinite {
            field: "covariance",
        },
        Covariance6Error::Asymmetric => FieldError::OutOfRange {
            field: "covariance",
            min: 0.0,
            max: 0.0,
            upper_inclusive: true,
        },
        Covariance6Error::NotPositiveSemidefinite => FieldError::NotPositive {
            field: "covariance",
        },
        Covariance6Error::NotFactorizable | Covariance6Error::InvalidInterpolationParameter => {
            FieldError::OutOfRange {
                field: "covariance",
                min: 0.0,
                max: 0.0,
                upper_inclusive: true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::ndm::tokenize;

    #[test]
    fn covariance6_round_trips_through_kvn_lines() {
        let covariance = Covariance6::from_diagonal([1.0, 2.0, 3.0, 4.0e-6, 5.0e-6, 6.0e-6])
            .expect("diagonal covariance");
        let lines = write_covariance6(&covariance);
        let map = FieldMap::from_pairs(tokenize(&lines.join("\n")));
        let recovered = read_covariance6(&map).expect("read covariance");

        assert_eq!(recovered.as_matrix(), covariance.as_matrix());
    }

    #[test]
    fn covariance6_units_follow_the_component_kinds() {
        assert_eq!(covariance6_unit("CZ_Y"), Some(&["km**2"][..]));
        assert_eq!(covariance6_unit("CY_DOT_Z"), Some(&["km**2/s"][..]));
        assert_eq!(covariance6_unit("CZ_DOT_Y_DOT"), Some(&["km**2/s**2"][..]));
        assert_eq!(covariance6_unit("CR_R"), None);
    }

    #[test]
    fn lower_triangle_round_trips_through_covariance6() {
        let covariance = Covariance6::from_diagonal([1.0, 2.0, 3.0, 4.0e-6, 5.0e-6, 6.0e-6])
            .expect("diagonal covariance");
        let lower = covariance6_lower_triangle(&covariance);
        let rebuilt = covariance6_from_lower_triangle(&lower).expect("rebuilt covariance");
        assert_eq!(rebuilt.as_matrix(), covariance.as_matrix());
    }

    #[test]
    fn missing_covariance_key_yields_field_error() {
        let map = FieldMap::parse("CX_X = 1\n");

        assert_eq!(
            read_covariance6(&map),
            Err(FieldError::Missing { field: "CY_X" })
        );
    }
}
