#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sha2::{Digest, Sha256};
use sidereon_core::astro::time::civil::j2000_seconds;
use sidereon_core::ephemeris::Sp3;
use sidereon_core::geometry::{
    dop_series, geometry_cofactor, visible, DopOptions, DopWeighting, LineOfSight,
    VisibilityOptions, Wgs84Geodetic,
};
use sidereon_core::observables::{transmit_time_satellite_state, TransmitTimeOptions};
use sidereon_core::positioning::{
    Corrections, EphemerisSource, KlobucharCoeffs, Observation, PseudorangeCode, SolveInputs,
    SurfaceMet,
};
use sidereon_core::static_positioning::{solve_static, StaticEpoch, StaticSolveOptions};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

const ORACLE: &str = include_str!("fixtures/sp3/rtklib_sp3_node_oracle.json");
const GEOMETRY_SP3: &str = "tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3";
const STATIC_SP3: &str = "tests/fixtures/sp3/trimmed_go_static.sp3";
const NAV_RINEX: &str = "tests/fixtures/nav/BRDM00DLR_S_20201760000_01D_MN.rnx";
const RTKLIB_REVISION: &str = "75a2e56275485b21a67bd35bc94bbeb8936e1a74";
const RTKLIB_SOURCE_URL: &str =
    "https://github.com/rtklibexplorer/RTKLIB/tree/75a2e56275485b21a67bd35bc94bbeb8936e1a74";
const NAV_SHA256: &str = "778e99a30b9fc3f2ea2844219f459535a9060645ec6accf89d83410154db5c53";
const TRIG_ABSOLUTE_ERROR: f64 = 2.0 * f64::EPSILON;
const RTKLIB_4X4_INVERSE_OPERATION_COUNT: usize = 163;

fn oracle() -> Value {
    serde_json::from_str(ORACLE).expect("parse RTKLIB SP3 oracle")
}

fn fixture_bytes(relative_path: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    std::fs::read(path).expect("read oracle input fixture")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn assert_fixture_provenance(document: &Value) {
    let provenance = &document["provenance"];
    assert_eq!(provenance["repository"], "rtklibexplorer/RTKLIB");
    assert_eq!(provenance["branch"], "demo5");
    assert_eq!(provenance["revision"], RTKLIB_REVISION);
    assert_eq!(provenance["source_url"], RTKLIB_SOURCE_URL);
    assert_eq!(
        provenance["generator"],
        "fixtures-generators/rtklib_sp3_oracle/generate.sh"
    );
    let source_functions: Vec<_> = provenance["source_functions"]
        .as_array()
        .expect("RTKLIB source function references")
        .iter()
        .map(|value| value.as_str().expect("source function reference"))
        .collect();
    assert_eq!(
        source_functions,
        vec![
            "preceph.c:peph2pos/pephpos",
            "rtkcmn.c:satazel/dops",
            "pntpos.c:pntpos"
        ]
    );

    let geometry_path = GEOMETRY_SP3.strip_prefix("tests/fixtures/").unwrap();
    let static_path = STATIC_SP3.strip_prefix("tests/fixtures/").unwrap();
    let navigation_path = NAV_RINEX.strip_prefix("tests/fixtures/nav/").unwrap();
    assert_eq!(provenance["geometry_fixture"], geometry_path);
    assert_eq!(provenance["static_fixture"], static_path);
    assert_eq!(provenance["navigation_file"], navigation_path);
    assert_eq!(
        provenance["navigation_source_url"],
        "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2020/176/BRDM00DLR_S_20201760000_01D_MN.rnx.gz"
    );

    let geometry_hash = sha256_hex(&fixture_bytes(GEOMETRY_SP3));
    let static_hash = sha256_hex(&fixture_bytes(STATIC_SP3));
    let navigation_hash = sha256_hex(&fixture_bytes(NAV_RINEX));
    assert_eq!(
        provenance["geometry_sha256"].as_str(),
        Some(geometry_hash.as_str())
    );
    assert_eq!(
        provenance["static_sha256"].as_str(),
        Some(static_hash.as_str())
    );
    assert_eq!(provenance["navigation_sha256"].as_str(), Some(NAV_SHA256));
    assert_eq!(
        provenance["navigation_sha256"].as_str(),
        Some(navigation_hash.as_str())
    );
}

fn sp3(relative_path: &str) -> Sp3 {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    Sp3::parse(&std::fs::read(path).expect("read SP3 fixture")).expect("parse SP3 fixture")
}

type RecordedSatelliteStates = BTreeMap<(GnssSatelliteId, u64), ([f64; 3], f64)>;

struct RecordingEphemerisSource<'a> {
    source: &'a Sp3,
    states: RefCell<RecordedSatelliteStates>,
}

impl RecordingEphemerisSource<'_> {
    fn new(source: &Sp3) -> RecordingEphemerisSource<'_> {
        RecordingEphemerisSource {
            source,
            states: RefCell::new(BTreeMap::new()),
        }
    }
}

impl EphemerisSource for RecordingEphemerisSource<'_> {
    fn position_clock_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        epoch_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state =
            EphemerisSource::position_clock_at_j2000_s(self.source, satellite, epoch_j2000_s)?;
        self.states
            .borrow_mut()
            .insert((satellite, epoch_j2000_s.to_bits()), state);
        Some(state)
    }

    fn clock_relativity_s(
        &self,
        satellite: GnssSatelliteId,
        epoch_j2000_s: f64,
    ) -> sidereon_core::positioning::ClockRelativity {
        EphemerisSource::clock_relativity_s(self.source, satellite, epoch_j2000_s)
    }

    fn clock_relativity_for_state_s(
        &self,
        satellite: GnssSatelliteId,
        epoch_j2000_s: f64,
        position_m: [f64; 3],
    ) -> sidereon_core::positioning::ClockRelativity {
        EphemerisSource::clock_relativity_for_state_s(
            self.source,
            satellite,
            epoch_j2000_s,
            position_m,
        )
    }

    fn try_transmit_epoch_clock_s(
        &self,
        satellite: GnssSatelliteId,
        epoch_j2000_s: f64,
        selection_j2000_s: f64,
    ) -> Result<Option<sidereon_core::astro::time::Validated<f64>>, sidereon_core::Error> {
        EphemerisSource::try_transmit_epoch_clock_s(
            self.source,
            satellite,
            epoch_j2000_s,
            selection_j2000_s,
        )
    }
}

fn static_fixture_epochs(baseline: &Value) -> [StaticEpoch; 2] {
    let satellites = baseline["input_satellites"]
        .as_array()
        .expect("RTKLIB input satellite identifiers");
    let pseudoranges = baseline["pseudoranges_m"]
        .as_array()
        .expect("RTKLIB input pseudoranges");
    assert_eq!(satellites.len(), 8);
    assert_eq!(pseudoranges.len(), satellites.len());
    let observations = satellites
        .iter()
        .zip(pseudoranges)
        .map(|(satellite, pseudorange)| Observation {
            satellite_id: satellite_id(satellite),
            pseudorange_m: pseudorange.as_f64().expect("pseudorange"),
        })
        .collect();
    let inputs = SolveInputs {
        observations,
        t_rx_j2000_s: 646_272_000.0,
        t_rx_second_of_day_s: 43_200.0,
        day_of_year: 176.5,
        initial_guess: [4.5e6, 0.5e6, 4.5e6, 0.0],
        corrections: Corrections::NONE,
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
        robust: None,
        pseudorange_code: PseudorangeCode::SingleFrequency,
    };
    let epoch = StaticEpoch::from_solve_inputs(inputs);
    [epoch.clone(), epoch]
}

fn satellite_id(value: &Value) -> GnssSatelliteId {
    value
        .as_str()
        .expect("satellite identifier")
        .parse()
        .expect("valid satellite identifier")
}

fn vector3(value: &Value) -> [f64; 3] {
    let components = value.as_array().expect("three vector components");
    assert_eq!(components.len(), 3);
    [
        components[0].as_f64().expect("x component"),
        components[1].as_f64().expect("y component"),
        components[2].as_f64().expect("z component"),
    ]
}

fn matrix4(value: &Value) -> [[f64; 4]; 4] {
    let entries = value.as_array().expect("4x4 matrix values");
    assert_eq!(entries.len(), 16);
    std::array::from_fn(|row_index| {
        std::array::from_fn(|column_index| {
            entries[row_index * 4 + column_index]
                .as_f64()
                .expect("matrix entry")
        })
    })
}

fn matrix_frobenius_norm(matrix: &[[f64; 4]; 4]) -> f64 {
    let norm = libm::sqrt(
        matrix
            .iter()
            .flatten()
            .map(|value| value * value)
            .sum::<f64>(),
    );
    norm * (1.0 + gamma_bound(40))
}

fn enu_rotation_orthogonality_bound(rotation: &[[f64; 3]; 3]) -> f64 {
    let mut defect_bound = 0.0;
    for row_index in 0..3 {
        for column_index in 0..3 {
            let mut product_sum = 0.0;
            let mut product_magnitude = 0.0;
            for axis_index in 0..3 {
                let product = rotation[axis_index][row_index] * rotation[axis_index][column_index];
                product_sum += product;
                product_magnitude += product.abs();
            }
            let identity_entry = if row_index == column_index { 1.0 } else { 0.0 };
            defect_bound += (identity_entry - product_sum).abs()
                + gamma_bound(6)
                    * (1.0 + gamma_bound(4))
                    * (identity_entry.abs() + product_magnitude);
        }
    }
    defect_bound * (1.0 + gamma_bound(16))
}

fn matrix_infinity_norm_upper(matrix: &[[f64; 4]; 4]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>() * (1.0 + gamma_bound(3)))
        .fold(0.0_f64, f64::max)
}

fn rtklib_scaled_lu_residual_ceiling(matrix: &[[f64; 4]; 4], inverse: &[[f64; 4]; 4]) -> f64 {
    let row_scales: Vec<f64> = matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).fold(0.0_f64, f64::max))
        .collect();
    let maximum_scale = row_scales.iter().copied().fold(0.0_f64, f64::max);
    let minimum_scale = row_scales.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(minimum_scale > 0.0, "scaled LU nonzero row scales");

    let scale_ratio = maximum_scale / minimum_scale * (1.0 + gamma_bound(1));
    let multiplier_bound = scale_ratio * (1.0 + gamma_bound(1)).powi(4);
    let growth_step = (1.0 + multiplier_bound) * (1.0 + gamma_bound(2));
    let growth_bound = growth_step.powi(3) * (1.0 + gamma_bound(3));
    let maximum_entry = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let lower_norm_bound = (1.0 + 3.0 * multiplier_bound) * (1.0 + gamma_bound(2));
    let upper_norm_bound = 4.0 * growth_bound * maximum_entry * (1.0 + gamma_bound(2));
    let lu_product_norm_bound = lower_norm_bound * upper_norm_bound * (1.0 + gamma_bound(1));
    let solve_rounding = gamma_bound(RTKLIB_4X4_INVERSE_OPERATION_COUNT)
        * lu_product_norm_bound
        * matrix_infinity_norm_upper(inverse);
    let residual_evaluation_rounding = gamma_bound(8)
        * (1.0 + gamma_bound(8))
        * (1.0 + matrix_infinity_norm_upper(matrix) * matrix_infinity_norm_upper(inverse));
    2.0 * (solve_rounding + residual_evaluation_rounding) * (1.0 + gamma_bound(3))
}

fn permutation_absolute_product_sum(
    matrix: &[[f64; 4]; 4],
    excluded_row: Option<usize>,
    excluded_column: Option<usize>,
) -> f64 {
    fn enumerate_products(
        matrix: &[[f64; 4]; 4],
        rows: &[usize],
        columns: &[usize],
        row_index: usize,
        used_columns: u8,
        product: f64,
        products: &mut Vec<f64>,
    ) {
        if row_index == rows.len() {
            products.push(product);
            return;
        }
        for (column_position, &column) in columns.iter().enumerate() {
            let column_bit = 1_u8 << column_position;
            if used_columns & column_bit == 0 {
                enumerate_products(
                    matrix,
                    rows,
                    columns,
                    row_index + 1,
                    used_columns | column_bit,
                    product * matrix[rows[row_index]][column].abs(),
                    products,
                );
            }
        }
    }

    let rows: Vec<usize> = (0..4).filter(|&row| Some(row) != excluded_row).collect();
    let columns: Vec<usize> = (0..4)
        .filter(|&column| Some(column) != excluded_column)
        .collect();
    let permutation_sum_operation_count = match rows.len() {
        3 => 17,
        4 => 95,
        _ => unreachable!("only three-by-three minors and four-by-four determinants"),
    };
    let mut products = Vec::with_capacity(permutation_sum_operation_count + 1);
    enumerate_products(matrix, &rows, &columns, 0, 0, 1.0, &mut products);
    let mut sum = 0.0;
    for product in products {
        sum += product;
    }
    sum * (1.0 + gamma_bound(permutation_sum_operation_count + 1))
}

fn cofactor_inverse_forward_error_bound(matrix: &[[f64; 4]; 4], inverse: &[[f64; 4]; 4]) -> f64 {
    let determinant = sidereon_core::astro::math::linear::det4_cofactor(matrix);
    let determinant_error = gamma_bound(45)
        * permutation_absolute_product_sum(matrix, None, None)
        * (1.0 + gamma_bound(1));
    let determinant_lower = (determinant.abs() - determinant_error) * (1.0 - gamma_bound(1));
    assert!(
        determinant_lower > 0.0,
        "cofactor determinant is separated from zero"
    );

    let mut inverse_entry_error_squares = 0.0;
    for row_index in 0..4 {
        for column_index in 0..4 {
            let numerator =
                sidereon_core::astro::math::linear::minor3_of_4(matrix, column_index, row_index)
                    .abs();
            let numerator_error = gamma_bound(14)
                * permutation_absolute_product_sum(matrix, Some(column_index), Some(row_index))
                * (1.0 + gamma_bound(1));
            let division_error = gamma_bound(1) * inverse[row_index][column_index].abs();
            let numerator_contribution =
                numerator_error / determinant_lower * (1.0 + gamma_bound(1));
            let determinant_contribution = (numerator + numerator_error) * determinant_error
                / determinant_lower.powi(2)
                * (1.0 + gamma_bound(3));
            let entry_error = (division_error + numerator_contribution + determinant_contribution)
                * (1.0 + gamma_bound(3));
            inverse_entry_error_squares += entry_error * entry_error;
        }
    }
    libm::sqrt(inverse_entry_error_squares * (1.0 + gamma_bound(31))) * (1.0 + gamma_bound(40))
}

fn transformed_normal_difference_bound(
    ecef_normal: &[[f64; 4]; 4],
    local_normal: &[[f64; 4]; 4],
    rotation: &[[f64; 3]; 3],
) -> f64 {
    let mut transform = [[0.0; 4]; 4];
    for row_index in 0..3 {
        for column_index in 0..3 {
            transform[row_index][column_index] = -rotation[row_index][column_index];
        }
    }
    transform[3][3] = 1.0;

    let mut intermediate = [[0.0; 4]; 4];
    let mut intermediate_error = [[0.0; 4]; 4];
    for row_index in 0..4 {
        for column_index in 0..4 {
            let mut sum = 0.0;
            let mut magnitude = 0.0;
            for inner_index in 0..4 {
                let product =
                    transform[row_index][inner_index] * ecef_normal[inner_index][column_index];
                sum += product;
                magnitude += product.abs();
            }
            intermediate[row_index][column_index] = sum;
            intermediate_error[row_index][column_index] =
                gamma_bound(8) * magnitude * (1.0 + gamma_bound(3));
        }
    }

    let mut total_error = 0.0;
    for row_index in 0..4 {
        for column_index in 0..4 {
            let mut transformed = 0.0;
            let mut magnitude = 0.0;
            let mut propagated_error = 0.0;
            for inner_index in 0..4 {
                let product =
                    intermediate[row_index][inner_index] * transform[column_index][inner_index];
                transformed += product;
                magnitude += product.abs();
                propagated_error += intermediate_error[row_index][inner_index]
                    * transform[column_index][inner_index].abs();
            }
            let element_error = (local_normal[row_index][column_index] - transformed).abs()
                * (1.0 + gamma_bound(1))
                + propagated_error * (1.0 + gamma_bound(3))
                + gamma_bound(8) * magnitude * (1.0 + gamma_bound(3));
            total_error += element_error;
        }
    }
    total_error * (1.0 + gamma_bound(19))
}

fn sidereon_cofactor_local_residual_ceiling(
    ecef_normal: &[[f64; 4]; 4],
    ecef_inverse: &[[f64; 4]; 4],
    local_normal: &[[f64; 4]; 4],
    local_inverse: &[[f64; 4]; 4],
    rotation: &[[f64; 3]; 3],
) -> f64 {
    let ecef_inverse_error = cofactor_inverse_forward_error_bound(ecef_normal, ecef_inverse);
    let ecef_normal_norm = matrix_frobenius_norm(ecef_normal);
    let ecef_inverse_norm = matrix_frobenius_norm(ecef_inverse) + ecef_inverse_error;
    let rotation_norm_squared = rotation
        .iter()
        .flatten()
        .map(|entry| entry * entry)
        .sum::<f64>()
        * (1.0 + gamma_bound(20));
    let transform_norm_squared = rotation_norm_squared.max(1.0);
    let orthogonality_defect = enu_rotation_orthogonality_bound(rotation);
    assert!(orthogonality_defect < 1.0, "ENU rotation near orthogonal");

    let inverse_residual = ecef_normal_norm * ecef_inverse_error;
    let inverse_rounding = transform_norm_squared * inverse_residual;
    let basis_orthogonality = orthogonality_defect
        + transform_norm_squared * ecef_normal_norm * ecef_inverse_norm * orthogonality_defect;
    let basis_normal = transformed_normal_difference_bound(ecef_normal, local_normal, rotation)
        * matrix_frobenius_norm(local_inverse);
    let covariance_rotation_rounding = gamma_bound(12)
        * (rotation_norm_squared + 1.0)
        * ecef_inverse_norm
        * matrix_frobenius_norm(local_normal);
    let residual_evaluation_rounding = gamma_bound(8)
        * (1.0 + gamma_bound(8))
        * (1.0 + matrix_frobenius_norm(local_normal) * matrix_frobenius_norm(local_inverse));
    (inverse_rounding
        + basis_orthogonality
        + basis_normal
        + covariance_rotation_rounding
        + residual_evaluation_rounding)
        * (1.0 + gamma_bound(10))
}

fn inverse_residual_error_bound(
    matrix: &[[f64; 4]; 4],
    inverse: &[[f64; 4]; 4],
    arithmetic_ceiling: f64,
) -> (f64, f64) {
    let mut residual_infinity_bound = 0.0_f64;
    for row_index in 0..4 {
        let mut residual_row_bound = 0.0_f64;
        for column_index in 0..4 {
            let mut product_sum = 0.0_f64;
            let mut product_magnitude = 0.0_f64;
            for inner_index in 0..4 {
                let product = matrix[row_index][inner_index] * inverse[inner_index][column_index];
                product_sum += product;
                product_magnitude +=
                    (matrix[row_index][inner_index] * inverse[inner_index][column_index]).abs();
            }
            product_magnitude *= 1.0 + gamma_bound(3);
            let identity_entry = if row_index == column_index { 1.0 } else { 0.0 };
            let residual = identity_entry - product_sum;
            let residual_rounding = gamma_bound(8)
                * (1.0 + gamma_bound(8))
                * (identity_entry.abs() + product_magnitude);
            residual_row_bound += residual.abs() + residual_rounding;
        }
        residual_infinity_bound = residual_infinity_bound.max(residual_row_bound);
    }
    let residual_frobenius_bound = 2.0 * residual_infinity_bound * (1.0 + gamma_bound(8));
    assert!(
        arithmetic_ceiling < 1.0,
        "algorithm-derived inverse ceiling"
    );
    assert!(
        residual_frobenius_bound <= arithmetic_ceiling,
        "inverse residual exceeds its algorithm-derived roundoff ceiling"
    );
    assert!(residual_frobenius_bound < 1.0, "certified inverse residual");
    let inverse_norm_bound = matrix_frobenius_norm(inverse);
    let inverse_error_bound = inverse_norm_bound * residual_frobenius_bound
        / (1.0 - residual_frobenius_bound)
        * (1.0 + gamma_bound(8));
    (inverse_norm_bound, inverse_error_bound)
}

fn normal_matrix_perturbation_bound(
    geometry: &Value,
    actual_rows: &[[f64; 4]],
    actual_satellites: &[GnssSatelliteId],
) -> f64 {
    assert_eq!(actual_rows.len(), actual_satellites.len());
    let oracle_rows = geometry["satellites"]
        .as_array()
        .expect("RTKLIB satellite rows");
    let mut rank_one_difference_bound = 0.0;
    let mut normal_rounding_bound = 0.0;

    for (actual_row, satellite) in actual_rows.iter().zip(actual_satellites) {
        let satellite_name = satellite.to_string();
        let oracle_row = oracle_rows
            .iter()
            .find(|row| row["id"].as_str() == Some(satellite_name.as_str()))
            .expect("RTKLIB row for DOP satellite");
        let azimuth = oracle_row["dop_azimuth_rad"]
            .as_f64()
            .expect("RTKLIB DOP azimuth");
        let elevation = oracle_row["dop_elevation_rad"]
            .as_f64()
            .expect("RTKLIB DOP elevation");
        let cosine_elevation = libm::cos(elevation);
        let oracle_design_row = [
            cosine_elevation * libm::sin(azimuth),
            cosine_elevation * libm::cos(azimuth),
            libm::sin(elevation),
            1.0,
        ];
        let row_difference = libm::sqrt(
            actual_row
                .iter()
                .zip(oracle_design_row)
                .map(|(actual, oracle)| (actual - oracle).powi(2))
                .sum::<f64>(),
        ) * (1.0 + gamma_bound(8));
        let actual_row_rounding = gamma_bound(5) * libm::sqrt(3.0) * (1.0 + gamma_bound(20));
        let oracle_row_rounding = gamma_bound(2) * libm::sqrt(2.0) + 6.0 * TRIG_ABSOLUTE_ERROR;
        let row_difference_bound = row_difference + actual_row_rounding + oracle_row_rounding;
        let actual_row_norm = libm::sqrt(actual_row.iter().map(|value| value * value).sum::<f64>())
            * (1.0 + gamma_bound(8))
            + actual_row_rounding;
        let oracle_row_norm = libm::sqrt(
            oracle_design_row
                .iter()
                .map(|value| value * value)
                .sum::<f64>(),
        ) * (1.0 + gamma_bound(8))
            + oracle_row_rounding;
        rank_one_difference_bound += row_difference_bound * (actual_row_norm + oracle_row_norm);
        normal_rounding_bound +=
            actual_row_norm * actual_row_norm + oracle_row_norm * oracle_row_norm;
    }

    (rank_one_difference_bound
        + 4.0 * gamma_bound(2 * actual_rows.len() + 4) * normal_rounding_bound)
        * (1.0 + gamma_bound(2 * actual_rows.len() + 24))
}

fn angular_error_deg(actual: f64, expected: f64) -> f64 {
    (actual - expected + 180.0).rem_euclid(360.0) - 180.0
}

fn angular_position_bound_deg(position_error_m: f64, range_m: f64, elevation_rad: f64) -> f64 {
    let direction_bound_rad = libm::asin((position_error_m / range_m).min(1.0));
    let maximum_absolute_elevation_rad =
        (elevation_rad.abs() + direction_bound_rad).min(std::f64::consts::FRAC_PI_2);
    direction_bound_rad
        / libm::cos(maximum_absolute_elevation_rad)
            .abs()
            .max(f64::MIN_POSITIVE)
        * 180.0
        / std::f64::consts::PI
        + 64.0 * f64::EPSILON * 360.0
}

fn position_error_bound_m(
    source: &Sp3,
    satellite: GnssSatelliteId,
    query_s: f64,
    time_radius_s: f64,
) -> f64 {
    let epochs = source.epochs_j2000_seconds();
    let mut nodes = Vec::new();
    for (epoch_index, epoch_s) in epochs.into_iter().enumerate() {
        if let Ok(state) = source.state(satellite, epoch_index) {
            let position_m = state.position.as_array();
            nodes.push((epoch_s, position_m.map(|component| component / 1000.0)));
        }
    }
    assert!(nodes.len() >= 11, "{satellite}: enough SP3 position nodes");

    let nominal_spacing_s = nodes
        .windows(2)
        .map(|pair| pair[1].0 - pair[0].0)
        .filter(|spacing| *spacing > 0.0)
        .fold(f64::INFINITY, f64::min);
    let gap_limit_s = nominal_spacing_s * source.interpolation_options().gap_threshold_factor();
    let mut pivot = 0;
    while pivot + 1 < nodes.len() && nodes[pivot + 1].0 < query_s {
        pivot += 1;
    }
    if pivot + 1 < nodes.len()
        && nodes[pivot + 1].0 - nodes[pivot].0 > gap_limit_s
        && query_s >= nodes[pivot + 1].0 - nominal_spacing_s
    {
        pivot += 1;
    }
    let mut run_start = pivot;
    while run_start > 0 && nodes[run_start].0 - nodes[run_start - 1].0 <= gap_limit_s {
        run_start -= 1;
    }
    let mut run_end = pivot + 1;
    while run_end < nodes.len() && nodes[run_end].0 - nodes[run_end - 1].0 <= gap_limit_s {
        run_end += 1;
    }
    let window_len = 11.min(run_end - run_start);
    let mut window_start = pivot.saturating_sub(5).max(run_start);
    if window_start + window_len > run_end {
        window_start = run_end - window_len;
    }
    let selected_nodes = &nodes[window_start..window_start + window_len];
    assert_eq!(
        selected_nodes.len(),
        11,
        "{satellite}: selected SP3 stencil"
    );

    let offsets_s: Vec<_> = selected_nodes
        .iter()
        .map(|(epoch_s, _)| epoch_s - query_s)
        .collect();
    let offset_rounding_radius_s = gamma_bound(2)
        * offsets_s
            .iter()
            .map(|offset_s| offset_s.abs())
            .fold(0.0, f64::max);
    let node_norms_km: Vec<_> = selected_nodes
        .iter()
        .map(|(_, position_km)| {
            let coordinate_l1_km =
                position_km[0].abs() + position_km[1].abs() + position_km[2].abs();
            coordinate_l1_km + gamma_bound(4) * coordinate_l1_km
        })
        .collect();
    let mut component_error_km = [[0.0; 11]; 3];
    let mut component_magnitude_km = [[0.0; 11]; 3];
    for (node_index, (_, position_km)) in selected_nodes.iter().enumerate() {
        let offset_s = offsets_s[node_index];
        let theta = sidereon_core::constants::OMEGA_E_DOT_RAD_S * offset_s;
        assert!(
            theta.abs() <= 0.5,
            "{satellite}: bounded earth-rotation angle"
        );
        let xy_magnitude_km = position_km[0].abs() + position_km[1].abs();
        let source_rounding_xy_km = gamma_bound(4) * xy_magnitude_km;
        let source_rounding_z_km = gamma_bound(4) * position_km[2].abs();
        let rotation_rounding_km = 2.0 * gamma_bound(3) * xy_magnitude_km
            + 4.0 * TRIG_ABSOLUTE_ERROR * xy_magnitude_km
            + gamma_bound(1) * node_norms_km[node_index] * theta.abs();
        component_magnitude_km[0][node_index] = xy_magnitude_km;
        component_magnitude_km[1][node_index] = xy_magnitude_km;
        component_magnitude_km[2][node_index] = position_km[2].abs();
        component_error_km[0][node_index] = source_rounding_xy_km + rotation_rounding_km;
        component_error_km[1][node_index] = source_rounding_xy_km + rotation_rounding_km;
        component_error_km[2][node_index] = source_rounding_z_km;
    }

    let mut component_errors_m = [0.0; 3];
    for (component_index, component_error_m) in component_errors_m.iter_mut().enumerate() {
        let (interpolated_magnitude_km, recurrence_error_km) = neville_roundoff_bound(
            &offsets_s,
            &component_magnitude_km[component_index],
            &component_error_km[component_index],
        );
        *component_error_m = recurrence_error_km * 1000.0
            + 2.0 * gamma_bound(1) * interpolated_magnitude_km * 1000.0;
    }
    let derivative_bound_km_s =
        lagrange_query_derivative_bound_m_s(&offsets_s, &node_norms_km, time_radius_s);
    let offset_derivative_bound_km_s = lagrange_node_offset_derivative_bound_km_s(
        &offsets_s,
        &node_norms_km,
        time_radius_s,
        offset_rounding_radius_s,
    );
    let position_bound_m = component_errors_m.iter().sum::<f64>()
        + 2.0
            * 1000.0
            * (derivative_bound_km_s * time_radius_s
                + offset_derivative_bound_km_s * offset_rounding_radius_s);
    position_bound_m * (1.0 + gamma_bound(32768))
}

fn neville_roundoff_bound(offsets_s: &[f64], magnitudes: &[f64], errors: &[f64]) -> (f64, f64) {
    let mut magnitude = magnitudes.to_vec();
    let mut error = errors.to_vec();
    for order in 1..magnitude.len() {
        for index in 0..magnitude.len() - order {
            let high_offset = offsets_s[index + order];
            let low_offset = offsets_s[index];
            let denominator = (high_offset - low_offset).abs();
            let high_scale = high_offset.abs() / denominator;
            let low_scale = low_offset.abs() / denominator;
            let numerator_magnitude =
                high_scale * magnitude[index] + low_scale * magnitude[index + 1];
            let denominator_relative_error =
                f64::EPSILON * (high_offset.abs() + low_offset.abs()) / denominator;
            assert!(denominator_relative_error < 1.0);
            let operation_error =
                gamma_bound(4) + denominator_relative_error / (1.0 - denominator_relative_error);
            error[index] = high_scale * error[index]
                + low_scale * error[index + 1]
                + 2.0 * numerator_magnitude * operation_error;
            magnitude[index] = numerator_magnitude;
        }
    }
    (magnitude[0], error[0])
}

fn lagrange_query_derivative_bound_m_s(
    offsets_s: &[f64],
    node_norms_km: &[f64],
    time_radius_s: f64,
) -> f64 {
    let mut derivative_bound_km_s = 0.0;
    for node_index in 0..offsets_s.len() {
        let denominator = offsets_s
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != node_index)
            .map(|(_, other_offset)| {
                (offsets_s[node_index] - other_offset).abs()
                    - f64::EPSILON * (offsets_s[node_index].abs() + other_offset.abs())
            })
            .product::<f64>();
        assert!(denominator > 0.0);
        let mut basis_bound = 1.0;
        for (other_index, other_offset) in offsets_s.iter().enumerate() {
            if other_index != node_index {
                basis_bound *= other_offset.abs() + time_radius_s;
            }
        }
        basis_bound /= denominator;
        let mut derivative_basis_bound = 0.0;
        for omitted_index in 0..offsets_s.len() {
            if omitted_index == node_index {
                continue;
            }
            let mut term = 1.0;
            for (other_index, other_offset) in offsets_s.iter().enumerate() {
                if other_index != node_index && other_index != omitted_index {
                    term *= other_offset.abs() + time_radius_s;
                }
            }
            derivative_basis_bound += term / denominator;
        }
        derivative_bound_km_s += node_norms_km[node_index]
            * (sidereon_core::constants::OMEGA_E_DOT_RAD_S * basis_bound + derivative_basis_bound);
    }
    derivative_bound_km_s
}

fn lagrange_node_offset_derivative_bound_km_s(
    offsets_s: &[f64],
    node_norms_km: &[f64],
    query_radius_s: f64,
    offset_radius_s: f64,
) -> f64 {
    let mut node_sensitivity_sum_km_s = 0.0;
    for node_index in 0..offsets_s.len() {
        let mut denominator_lower = 1.0;
        for (other_index, other_offset) in offsets_s.iter().enumerate() {
            if other_index != node_index {
                let separation = (offsets_s[node_index] - other_offset).abs();
                denominator_lower *= separation
                    - 2.0 * offset_radius_s
                    - f64::EPSILON * (offsets_s[node_index].abs() + other_offset.abs());
            }
        }
        assert!(denominator_lower > 0.0);
        let mut basis_upper = 1.0;
        for (other_index, other_offset) in offsets_s.iter().enumerate() {
            if other_index != node_index {
                basis_upper *= other_offset.abs() + query_radius_s + offset_radius_s;
            }
        }
        basis_upper /= denominator_lower;

        for varied_index in 0..offsets_s.len() {
            let mut derivative_basis_upper = 0.0;
            if varied_index == node_index {
                for omitted_index in 0..offsets_s.len() {
                    if omitted_index == node_index {
                        continue;
                    }
                    let mut term = 1.0;
                    for (other_index, _) in offsets_s.iter().enumerate() {
                        if other_index != node_index && other_index != omitted_index {
                            term *= (offsets_s[node_index] - offsets_s[other_index]).abs()
                                + 2.0 * offset_radius_s;
                        }
                    }
                    derivative_basis_upper += term;
                }
                derivative_basis_upper *= basis_upper / denominator_lower;
            } else {
                let mut numerator_derivative = 1.0;
                for (other_index, other_offset) in offsets_s.iter().enumerate() {
                    if other_index != node_index && other_index != varied_index {
                        numerator_derivative *=
                            other_offset.abs() + query_radius_s + offset_radius_s;
                    }
                }
                let first_term = numerator_derivative / denominator_lower;
                let mut denominator_derivative = 1.0;
                for (other_index, _) in offsets_s.iter().enumerate() {
                    if other_index != node_index && other_index != varied_index {
                        denominator_derivative *= (offsets_s[node_index] - offsets_s[other_index])
                            .abs()
                            + 2.0 * offset_radius_s;
                    }
                }
                let second_term = basis_upper * denominator_derivative / denominator_lower;
                derivative_basis_upper = first_term + second_term;
            }
            node_sensitivity_sum_km_s += node_norms_km[node_index] * derivative_basis_upper;
        }
        node_sensitivity_sum_km_s +=
            sidereon_core::constants::OMEGA_E_DOT_RAD_S * node_norms_km[node_index] * basis_upper;
    }
    node_sensitivity_sum_km_s
}

fn gamma_bound(operation_count: usize) -> f64 {
    let scaled_epsilon = operation_count as f64 * f64::EPSILON;
    scaled_epsilon / (1.0 - scaled_epsilon)
}

fn epoch_ulp_seconds(epoch_s: f64) -> f64 {
    f64::from_bits(epoch_s.to_bits() + 1) - epoch_s
}

fn dop_covariance_error_bound(
    source: &Sp3,
    geometry: &Value,
    epoch_s: f64,
    actual_satellites: &[GnssSatelliteId],
) -> ([f64; 5], [f64; 5], [f64; 5]) {
    let receiver_ecef_m = vector3(&geometry["dop_receiver_ecef_m"]);
    let satellites = geometry["satellites"]
        .as_array()
        .expect("RTKLIB satellite rows");
    for row in satellites {
        let satellite_position = vector3(&row["position_ecef_m"]);
        let range_m = libm::hypot(
            libm::hypot(
                satellite_position[0] - receiver_ecef_m[0],
                satellite_position[1] - receiver_ecef_m[1],
            ),
            satellite_position[2] - receiver_ecef_m[2],
        );
        let position_error_m = position_error_bound_m(
            source,
            satellite_id(&row["id"]),
            epoch_s,
            2.0 * epoch_ulp_seconds(epoch_s),
        );
        let elevation_rad = row["dop_elevation_rad"].as_f64().expect("RTKLIB elevation");
        let direction_error_rad = libm::asin((position_error_m / range_m).min(1.0));
        assert!(
            (elevation_rad - 5.0_f64.to_radians()).abs() > direction_error_rad,
            "{}: DOP visibility threshold is not separated from the position bound",
            row["id"]
        );
    }

    let oracle_normal = matrix4(&geometry["dop_normal_matrix_local"]);
    let oracle_covariance = matrix4(&geometry["dop_covariance_local"]);
    let oracle_ceiling = rtklib_scaled_lu_residual_ceiling(&oracle_normal, &oracle_covariance);
    let (oracle_inverse_norm, oracle_inverse_error) =
        inverse_residual_error_bound(&oracle_normal, &oracle_covariance, oracle_ceiling);

    let actual_geometry =
        sidereon_dop_cofactor(source, receiver_ecef_m, epoch_s, actual_satellites);
    let actual_ceiling = sidereon_cofactor_local_residual_ceiling(
        &actual_geometry.ecef_normal,
        &actual_geometry.ecef_inverse,
        &actual_geometry.local_normal,
        &actual_geometry.local_inverse,
        &actual_geometry.enu_rotation,
    );
    let (actual_inverse_norm, actual_inverse_error) = inverse_residual_error_bound(
        &actual_geometry.local_normal,
        &actual_geometry.local_inverse,
        actual_ceiling,
    );
    let normal_perturbation_bound = normal_matrix_perturbation_bound(
        geometry,
        &actual_geometry.local_design_rows,
        actual_satellites,
    );
    let inverse_norm_product =
        (actual_inverse_norm + actual_inverse_error) * (oracle_inverse_norm + oracle_inverse_error);
    let covariance_difference_bound = inverse_norm_product * normal_perturbation_bound
        + actual_inverse_error
        + oracle_inverse_error;
    let actual_variances = covariance_variances(&actual_geometry.cofactor);
    let oracle_variances = covariance_variances_from_state(&oracle_covariance);
    let rotation_norm_squared = actual_geometry
        .enu_rotation
        .iter()
        .flatten()
        .map(|entry| entry * entry)
        .sum::<f64>()
        * (1.0 + gamma_bound(20));
    let actual_rotation_rounding = (gamma_bound(12) * rotation_norm_squared
        + enu_rotation_orthogonality_bound(&actual_geometry.enu_rotation))
        * actual_inverse_norm;
    let variance_projection_ranks = [4.0, 3.0, 2.0, 1.0, 1.0];
    let mut variance_error_bounds = [0.0; 5];
    for index in 0..5 {
        let actual_rotation_factor = if (1..=3).contains(&index) {
            rotation_norm_squared
        } else {
            1.0
        };
        let actual_rounding = if index <= 3 {
            actual_rotation_rounding
        } else {
            0.0
        };
        let actual_rounding = actual_rounding
            + gamma_bound(4)
                * variance_projection_ranks[index]
                * actual_inverse_norm
                * actual_rotation_factor;
        let oracle_rounding = gamma_bound(4)
            * variance_projection_ranks[index]
            * matrix_frobenius_norm(&oracle_covariance);
        let variance_difference_bound =
            variance_projection_ranks[index] * actual_rotation_factor * covariance_difference_bound
                + actual_rounding
                + oracle_rounding;
        let exact_variance_lower = actual_variances[index].min(oracle_variances[index]);
        assert!(
            exact_variance_lower > 0.0,
            "DOP variance enclosure is positive"
        );
        variance_error_bounds[index] = variance_difference_bound
            / (2.0 * libm::sqrt(exact_variance_lower))
            * (1.0 + gamma_bound(32));
    }
    (actual_variances, variance_error_bounds, oracle_variances)
}

struct SidereonDopCofactor {
    cofactor: sidereon_core::geometry::GeometryCofactor,
    enu_rotation: [[f64; 3]; 3],
    ecef_normal: [[f64; 4]; 4],
    ecef_inverse: [[f64; 4]; 4],
    local_design_rows: Vec<[f64; 4]>,
    local_normal: [[f64; 4]; 4],
    local_inverse: [[f64; 4]; 4],
}

fn sidereon_dop_cofactor(
    source: &Sp3,
    receiver_ecef_m: [f64; 3],
    epoch_s: f64,
    satellites: &[GnssSatelliteId],
) -> SidereonDopCofactor {
    let mut line_of_sight = Vec::with_capacity(satellites.len());
    let mut local_design_rows = Vec::with_capacity(satellites.len());
    let mut transmit_options = TransmitTimeOptions::default();
    transmit_options.light_time = false;
    transmit_options.sagnac = false;
    for &satellite in satellites {
        let state = transmit_time_satellite_state(
            source,
            satellite,
            receiver_ecef_m,
            epoch_s,
            transmit_options,
        )
        .expect("unweighted DOP transmit geometry");
        let los = LineOfSight::new(state.los_unit[0], state.los_unit[1], state.los_unit[2]);
        line_of_sight.push(los);
    }
    let (latitude_deg, longitude_deg, _) =
        sidereon_core::astro::frames::transforms::itrs_to_geodetic_compute(
            receiver_ecef_m[0] / 1000.0,
            receiver_ecef_m[1] / 1000.0,
            receiver_ecef_m[2] / 1000.0,
        )
        .expect("DOP receiver geodetic coordinates");
    let degree_to_radian = std::f64::consts::PI / 180.0;
    let receiver = Wgs84Geodetic::new(
        latitude_deg * degree_to_radian,
        sidereon_core::astro::angles::normalize_geodetic_lon_rad(longitude_deg * degree_to_radian),
        0.0,
    )
    .expect("valid DOP receiver");
    let enu_rotation = sidereon_core::dop::ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad);
    let mut ecef_normal = [[0.0; 4]; 4];
    for row_index in 0..4 {
        for column_index in 0..4 {
            for los in &line_of_sight {
                let row_component = match row_index {
                    0 => -los.e_x,
                    1 => -los.e_y,
                    2 => -los.e_z,
                    _ => 1.0,
                };
                let column_component = match column_index {
                    0 => -los.e_x,
                    1 => -los.e_y,
                    2 => -los.e_z,
                    _ => 1.0,
                };
                ecef_normal[row_index][column_index] += row_component * column_component;
            }
        }
    }
    for los in &line_of_sight {
        let sidereon_ecef_row = [-los.e_x, -los.e_y, -los.e_z];
        local_design_rows.push([
            -(enu_rotation[0][0] * sidereon_ecef_row[0]
                + enu_rotation[0][1] * sidereon_ecef_row[1]
                + enu_rotation[0][2] * sidereon_ecef_row[2]),
            -(enu_rotation[1][0] * sidereon_ecef_row[0]
                + enu_rotation[1][1] * sidereon_ecef_row[1]
                + enu_rotation[1][2] * sidereon_ecef_row[2]),
            -(enu_rotation[2][0] * sidereon_ecef_row[0]
                + enu_rotation[2][1] * sidereon_ecef_row[1]
                + enu_rotation[2][2] * sidereon_ecef_row[2]),
            1.0,
        ]);
    }
    let weights = vec![1.0; line_of_sight.len()];
    let cofactor =
        geometry_cofactor(&line_of_sight, &weights, receiver).expect("Sidereon DOP cofactor");
    let mut local_normal = [[0.0; 4]; 4];
    for row_index in 0..4 {
        for column_index in 0..4 {
            let mut sum = 0.0;
            for design_row in &local_design_rows {
                sum += design_row[row_index] * design_row[column_index];
            }
            local_normal[row_index][column_index] = sum;
        }
    }
    let mut local_inverse = [[0.0; 4]; 4];
    for row_index in 0..3 {
        for column_index in 0..3 {
            local_inverse[row_index][column_index] = cofactor.position_enu[row_index][column_index];
        }
        let mut clock_cross = 0.0;
        for axis_index in 0..3 {
            clock_cross += enu_rotation[row_index][axis_index] * cofactor.state[axis_index][3];
        }
        local_inverse[row_index][3] = -clock_cross;
        local_inverse[3][row_index] = -clock_cross;
    }
    local_inverse[3][3] = cofactor.state[3][3];
    SidereonDopCofactor {
        ecef_normal,
        ecef_inverse: cofactor.state,
        cofactor,
        enu_rotation,
        local_design_rows,
        local_normal,
        local_inverse,
    }
}

fn covariance_variances(cofactor: &sidereon_core::geometry::GeometryCofactor) -> [f64; 5] {
    [
        cofactor.state[0][0] + cofactor.state[1][1] + cofactor.state[2][2] + cofactor.state[3][3],
        cofactor.position_enu[0][0] + cofactor.position_enu[1][1] + cofactor.position_enu[2][2],
        cofactor.position_enu[0][0] + cofactor.position_enu[1][1],
        cofactor.position_enu[2][2],
        cofactor.state[3][3],
    ]
}

fn covariance_variances_from_state(covariance: &[[f64; 4]; 4]) -> [f64; 5] {
    [
        covariance[0][0] + covariance[1][1] + covariance[2][2] + covariance[3][3],
        covariance[0][0] + covariance[1][1] + covariance[2][2],
        covariance[0][0] + covariance[1][1],
        covariance[2][2],
        covariance[3][3],
    ]
}

#[test]
fn rtklib_peph2pos_oracle_matches_sp3_node_positions() {
    let document = oracle();
    assert_fixture_provenance(&document);
    let source = sp3(GEOMETRY_SP3);
    let epoch = j2000_seconds(2020, 6, 24, 12, 0, 0.0);
    let oracle_satellites = document["geometry"]["satellites"]
        .as_array()
        .expect("RTKLIB satellite rows");
    assert!(!oracle_satellites.is_empty());

    for oracle_satellite in oracle_satellites {
        let satellite = satellite_id(&oracle_satellite["id"]);
        let expected = vector3(&oracle_satellite["position_ecef_m"]);
        let actual = source
            .position_at_j2000_seconds(satellite, epoch)
            .expect("interpolate a satellite at an SP3 node")
            .position
            .as_array();
        let error_m = ((actual[0] - expected[0]).powi(2)
            + (actual[1] - expected[1]).powi(2)
            + (actual[2] - expected[2]).powi(2))
        .sqrt();
        let position_error_bound_m =
            position_error_bound_m(&source, satellite, epoch, 2.0 * epoch_ulp_seconds(epoch));
        assert!(
            error_m <= position_error_bound_m,
            "{satellite}: RTKLIB node-position difference {error_m:e} m exceeds {position_error_bound_m:e} m"
        );
    }
}

#[test]
fn rtklib_satazel_oracle_matches_visibility_and_angles() {
    let document = oracle();
    assert_fixture_provenance(&document);
    let geometry = &document["geometry"];
    let source = sp3(GEOMETRY_SP3);
    let receiver = vector3(&geometry["receiver_ecef_m"]);
    let epoch = j2000_seconds(2020, 6, 24, 12, 0, 0.0);
    let mut visibility_options = VisibilityOptions::default();
    visibility_options.elevation_mask_deg = 10.0;
    visibility_options.systems = Some(BTreeSet::from([GnssSystem::Gps]));
    let visible_rows = visible(
        &source,
        source.satellites(),
        receiver,
        epoch,
        &visibility_options,
    )
    .expect("calculate visibility");
    for oracle_row in geometry["satellites"]
        .as_array()
        .expect("RTKLIB satellite rows")
    {
        let satellite_position = vector3(&oracle_row["position_ecef_m"]);
        let range_m = libm::hypot(
            libm::hypot(
                satellite_position[0] - receiver[0],
                satellite_position[1] - receiver[1],
            ),
            satellite_position[2] - receiver[2],
        );
        let position_error_m = position_error_bound_m(
            &source,
            satellite_id(&oracle_row["id"]),
            epoch,
            2.0 * epoch_ulp_seconds(epoch),
        );
        let direction_error_rad = libm::asin((position_error_m / range_m).min(1.0));
        let elevation_rad = oracle_row["application_elevation_rad"]
            .as_f64()
            .expect("RTKLIB elevation");
        assert!(
            (elevation_rad - 10.0_f64.to_radians()).abs() > direction_error_rad,
            "{}: application visibility threshold is not separated from the position bound",
            oracle_row["id"]
        );
    }
    let actual_ids: BTreeSet<_> = visible_rows
        .iter()
        .map(|row| row.satellite.to_string())
        .collect();
    let expected_ids: BTreeSet<_> = geometry["visible_gps_mask10"]
        .as_array()
        .expect("RTKLIB visible GPS list")
        .iter()
        .map(|value| value.as_str().expect("satellite identifier").to_owned())
        .collect();
    assert_eq!(actual_ids, expected_ids);

    for actual_row in visible_rows {
        let satellite_name = actual_row.satellite.to_string();
        let oracle_row = geometry["satellites"]
            .as_array()
            .expect("RTKLIB satellite rows")
            .iter()
            .find(|row| row["id"].as_str() == Some(satellite_name.as_str()))
            .expect("RTKLIB azimuth/elevation row");
        let satellite_position = vector3(&oracle_row["position_ecef_m"]);
        let range_m = ((satellite_position[0] - receiver[0]).powi(2)
            + (satellite_position[1] - receiver[1]).powi(2)
            + (satellite_position[2] - receiver[2]).powi(2))
        .sqrt();
        let elevation_rad = oracle_row["application_elevation_rad"]
            .as_f64()
            .expect("RTKLIB elevation");
        let position_error_bound_m = position_error_bound_m(
            &source,
            actual_row.satellite,
            epoch,
            2.0 * epoch_ulp_seconds(epoch),
        );
        let tolerance_deg =
            angular_position_bound_deg(position_error_bound_m, range_m, elevation_rad);
        let expected_elevation = oracle_row["application_elevation_deg"]
            .as_f64()
            .expect("RTKLIB elevation degrees");
        let expected_azimuth = oracle_row["application_azimuth_deg"]
            .as_f64()
            .expect("RTKLIB azimuth degrees");
        assert!(
            (actual_row.elevation_deg - expected_elevation).abs() <= tolerance_deg,
            "{satellite_name}: elevation difference {} deg exceeds {tolerance_deg} deg",
            (actual_row.elevation_deg - expected_elevation).abs()
        );
        assert!(
            angular_error_deg(actual_row.azimuth_deg, expected_azimuth).abs() <= tolerance_deg,
            "{satellite_name}: azimuth difference {} deg exceeds {tolerance_deg} deg",
            angular_error_deg(actual_row.azimuth_deg, expected_azimuth).abs()
        );
    }
}

#[test]
fn rtklib_dops_oracle_matches_the_first_sample() {
    let document = oracle();
    assert_fixture_provenance(&document);
    let geometry = &document["geometry"];
    let source = sp3(GEOMETRY_SP3);
    let receiver = vector3(&geometry["dop_receiver_ecef_m"]);
    let epoch = j2000_seconds(2020, 6, 24, 12, 0, 0.0);
    let mut dop_options = DopOptions::default();
    dop_options.visibility.elevation_mask_deg = 5.0;
    dop_options.visibility.systems = Some(BTreeSet::from([GnssSystem::Gps]));
    dop_options.weighting = DopWeighting::Unit;
    dop_options.light_time = false;
    let samples = dop_series(
        &source,
        source.satellites(),
        None,
        receiver,
        (epoch, epoch + 3_600.0),
        300,
        &dop_options,
    )
    .expect("calculate DOP series");
    let first = samples.first().expect("first DOP sample");
    assert_eq!(first.step_index, 0);

    let expected_satellites: BTreeSet<_> = geometry["satellites"]
        .as_array()
        .expect("RTKLIB satellite rows")
        .iter()
        .filter(|row| row["dop_visible_at_5_deg"] == true)
        .map(|row| satellite_id(&row["id"]))
        .collect();
    let actual_satellites: BTreeSet<_> = first.geometry.satellites.iter().copied().collect();
    assert_eq!(actual_satellites, expected_satellites);

    let expected_dop = &geometry["dops_mask5_deg"];
    let geometry_epoch_s = j2000_seconds(2020, 6, 24, 12, 0, 0.0);
    let geometry_source = sp3(GEOMETRY_SP3);
    let (actual_variances, dops_error_bound, oracle_variances) = dop_covariance_error_bound(
        &geometry_source,
        geometry,
        geometry_epoch_s,
        &first.geometry.satellites,
    );
    let actual_values = [
        first.geometry.dop.gdop,
        first.geometry.dop.pdop,
        first.geometry.dop.hdop,
        first.geometry.dop.vdop,
        first.geometry.dop.tdop,
    ];
    let fields = ["gdop", "pdop", "hdop", "vdop", "tdop"];
    let expected_values =
        fields.map(|field| expected_dop[field].as_f64().expect("RTKLIB DOP value"));
    for ((((field, actual), expected), tolerance), index) in fields
        .into_iter()
        .zip(actual_values)
        .zip(expected_values)
        .zip(dops_error_bound)
        .zip(0..5)
    {
        let actual_from_covariance = libm::sqrt(actual_variances[index]);
        let expected_from_covariance = libm::sqrt(oracle_variances[index]);
        let actual_scalar_rounding = gamma_bound(2) * (actual.abs() + actual_from_covariance.abs());
        let oracle_scalar_rounding =
            gamma_bound(2) * (expected.abs() + expected_from_covariance.abs());
        assert!(
            (actual - actual_from_covariance).abs() <= actual_scalar_rounding,
            "{field}: public DOP output does not match its certified covariance"
        );
        assert!(
            (expected - expected_from_covariance).abs() <= oracle_scalar_rounding,
            "{field}: RTKLIB DOP output does not match its recorded covariance"
        );
        let tolerance = tolerance + gamma_bound(2) * (actual.abs() + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{field}: Sidereon {actual:.17e}, RTKLIB {expected:.17e}, bound {tolerance:.3e}"
        );
    }
}

#[test]
fn rtklib_peph2pos_states_match_static_solver_transmit_queries() {
    let document = oracle();
    assert_fixture_provenance(&document);
    let baseline = &document["go_fixture_static"];
    let state_rows = baseline["sp3_tx_states"]
        .as_array()
        .expect("RTKLIB static SP3 transmit states");
    assert_eq!(state_rows.len(), 8);

    let source = sp3(STATIC_SP3);
    let recording_source = RecordingEphemerisSource::new(&source);
    let epochs = static_fixture_epochs(baseline);
    solve_static(&recording_source, &epochs, StaticSolveOptions::default())
        .expect("solve the duplicated static fixture epochs");
    let recorded_states = recording_source.states.into_inner();

    let receive_epoch_j2000_s: f64 = 646_272_000.0;
    let epoch_ulp_s = epoch_ulp_seconds(receive_epoch_j2000_s);
    let time_error_bound_s = 2.0 * epoch_ulp_s;
    let mut compared_satellites = BTreeSet::new();
    assert_eq!(recorded_states.len(), 8);

    for expected_state in state_rows {
        let satellite = satellite_id(&expected_state["id"]);
        let expected_epoch = receive_epoch_j2000_s
            + expected_state["transmit_epoch_offset_s"]
                .as_f64()
                .expect("RTKLIB transmit epoch offset");
        let satellite_states: Vec<_> = recorded_states
            .iter()
            .filter(|((recorded_satellite, _), _)| *recorded_satellite == satellite)
            .collect();
        assert_eq!(satellite_states.len(), 1, "{satellite}: unique query count");
        let recorded_state = satellite_states[0];
        let ((_, epoch_bits), (position_m, _)) = recorded_state;
        let actual_epoch = f64::from_bits(*epoch_bits);
        let time_error_s = (actual_epoch - expected_epoch).abs();
        assert!(
            time_error_s <= time_error_bound_s,
            "{satellite}: transmit-epoch difference {time_error_s:e} s exceeds {time_error_bound_s:e} s"
        );

        let actual_position = *position_m;
        let expected_position = vector3(&expected_state["position_ecef_m"]);
        let position_bound_m =
            position_error_bound_m(&source, satellite, expected_epoch, time_error_bound_s);
        let position_error_m = libm::sqrt(
            (actual_position[0] - expected_position[0]).powi(2)
                + (actual_position[1] - expected_position[1]).powi(2)
                + (actual_position[2] - expected_position[2]).powi(2),
        );
        assert!(
            position_error_m <= position_bound_m,
            "{satellite}: state difference {position_error_m:e} m exceeds {position_bound_m:e} m"
        );
        assert!(compared_satellites.insert(satellite));
    }
    assert_eq!(compared_satellites.len(), 8);
}

#[test]
fn rtklib_pntpos_result_is_retained_as_an_independent_baseline() {
    let document = oracle();
    assert_fixture_provenance(&document);
    let baseline = &document["go_fixture_static"];
    let epoch: Vec<_> = baseline["epoch_gps"]
        .as_array()
        .expect("RTKLIB positioning epoch")
        .iter()
        .map(|value| value.as_i64().expect("epoch field"))
        .collect();
    assert_eq!(epoch, [2020, 6, 24, 12, 0, 0]);
    assert_eq!(baseline["options"]["ephemeris"], "precise");
    assert_eq!(
        baseline["options"]["elevation_mask_deg"].as_i64(),
        Some(-90)
    );
    assert_eq!(baseline["options"]["iono"], "off");
    assert_eq!(baseline["options"]["tropo"], "off");
    assert_eq!(baseline["status"].as_i64(), Some(1));
    assert_eq!(
        vector3(&baseline["initial_position_ecef_m"]),
        [4_500_000.0, 500_000.0, 4_500_000.0]
    );
    assert!(vector3(&baseline["position_ecef_m"])
        .iter()
        .all(|value| value.is_finite()));
    assert!(baseline["clock_s"]
        .as_f64()
        .expect("RTKLIB receiver clock")
        .is_finite());
    let input_satellites: Vec<_> = baseline["input_satellites"]
        .as_array()
        .expect("RTKLIB input satellite identifiers")
        .iter()
        .map(|value| value.as_str().expect("satellite identifier"))
        .collect();
    assert_eq!(
        input_satellites,
        ["G08", "G10", "G16", "G18", "G20", "G21", "G26", "G27"]
    );
    let input_pseudoranges: Vec<_> = baseline["pseudoranges_m"]
        .as_array()
        .expect("RTKLIB input pseudoranges")
        .iter()
        .map(|value| value.as_f64().expect("pseudorange"))
        .collect();
    assert_eq!(
        input_pseudoranges,
        [
            23_825_519.844459895,
            22_717_690.10174763,
            20_478_653.376262885,
            21_768_335.23365917,
            21_248_327.738292538,
            20_808_709.800933376,
            21_126_481.58786735,
            21_341_367.541037586,
        ]
    );
    let used_satellites: Vec<_> = baseline["used_satellites"]
        .as_array()
        .expect("RTKLIB used satellites")
        .iter()
        .map(|value| value.as_str().expect("satellite identifier"))
        .collect();
    assert_eq!(used_satellites, input_satellites);
    assert_eq!(baseline["message"], "");
}
