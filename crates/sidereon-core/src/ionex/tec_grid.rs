//! Regular-grid TEC ionosphere delay variant.

use crate::astro::math::vec3::{
    dot3_fused_z_yx_ref as dot_three_fused, unit3_ref_unchecked as unit_vector,
};

use crate::constants::DEG_TO_RAD;
pub use crate::constants::MEAN_EARTH_RADIUS_M as EARTH_RADIUS_M;
use crate::frequencies::{self, CarrierBand};
use crate::validate;
use crate::GnssSystem;

pub const IONOSPHERE_HEIGHT_M: f64 = 450_000.0;
pub const IONOSPHERE_CONSTANT: f64 = 40.308193 * 1e16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TecGridEpoch {
    pub unix_nanos: i64,
    pub day_of_year: u16,
}

impl TecGridEpoch {
    pub fn new(unix_nanos: i64, day_of_year: u16) -> Self {
        Self {
            unix_nanos,
            day_of_year,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TecGridShellGeometry {
    pub earth_radius_m: f64,
    pub shell_height_m: f64,
}

impl TecGridShellGeometry {
    pub const fn new(earth_radius_m: f64, shell_height_m: f64) -> Self {
        Self {
            earth_radius_m,
            shell_height_m,
        }
    }

    pub const fn default_shell() -> Self {
        Self {
            earth_radius_m: EARTH_RADIUS_M,
            shell_height_m: IONOSPHERE_HEIGHT_M,
        }
    }

    pub fn shell_radius_m(self) -> f64 {
        self.earth_radius_m + self.shell_height_m
    }
}

impl Default for TecGridShellGeometry {
    fn default() -> Self {
        Self::default_shell()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TecGridEvalOptions {
    pub epoch: TecGridEpoch,
    pub min_elevation_rad: f64,
    pub nan_pierce_point_height_m: f64,
    pub frequency_hz: f64,
    pub shell_geometry: TecGridShellGeometry,
}

impl TecGridEvalOptions {
    pub fn l1(epoch: TecGridEpoch) -> Self {
        Self {
            epoch,
            min_elevation_rad: 5.0 * DEG_TO_RAD,
            nan_pierce_point_height_m: IONOSPHERE_HEIGHT_M,
            frequency_hz: frequencies::frequency_hz(GnssSystem::Gps, CarrierBand::L1)
                .expect("canonical GPS L1 carrier exists"),
            shell_geometry: TecGridShellGeometry::default(),
        }
    }

    pub fn with_shell_geometry(mut self, shell_geometry: TecGridShellGeometry) -> Self {
        self.nan_pierce_point_height_m = shell_geometry.shell_height_m;
        self.shell_geometry = shell_geometry;
        self
    }
}

#[derive(Clone, Debug)]
pub struct TecGrid {
    epochs_ns: Vec<f64>,
    latitudes_deg: Vec<f64>,
    longitudes_deg: Vec<f64>,
    values: Vec<f64>,
}

impl TecGrid {
    pub fn new(
        epochs_ns: Vec<f64>,
        latitudes_deg: Vec<f64>,
        longitudes_deg: Vec<f64>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        if epochs_ns.len() < 2 || latitudes_deg.len() < 2 || longitudes_deg.len() < 2 {
            return Err("TEC grid axes must each contain at least two entries".to_string());
        }
        if !strictly_increasing(&epochs_ns)
            || !strictly_increasing(&latitudes_deg)
            || !strictly_increasing(&longitudes_deg)
        {
            return Err("TEC grid axes must be strictly increasing".to_string());
        }
        let expected = epochs_ns
            .len()
            .checked_mul(latitudes_deg.len())
            .and_then(|v| v.checked_mul(longitudes_deg.len()))
            .ok_or_else(|| "TEC grid dimensions overflow".to_string())?;
        if values.len() != expected {
            return Err(format!(
                "TEC grid has {} values but expected {}",
                values.len(),
                expected
            ));
        }
        validate::finite_slice(&values, "TEC grid values").map_err(field_error_string)?;
        Ok(Self {
            epochs_ns,
            latitudes_deg,
            longitudes_deg,
            values,
        })
    }

    pub fn vtec_at_pierce_point(
        &self,
        epoch: TecGridEpoch,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> Result<f64, String> {
        let latitude_deg = if latitude_deg.abs() > 87.5 {
            clamp(latitude_deg, -87.5, 87.5)
        } else {
            latitude_deg
        };
        self.interpolate_vtec(epoch.unix_nanos as f64, latitude_deg, longitude_deg)
    }

    pub(crate) fn interpolate_vtec(
        &self,
        epoch_ns: f64,
        latitude_deg: f64,
        longitude_deg: f64,
    ) -> Result<f64, String> {
        let epoch_ns = finite_query_value(epoch_ns, "timestamp")?;
        let latitude_deg = finite_query_value(latitude_deg, "latitude")?;
        let longitude_deg = finite_query_value(longitude_deg, "longitude")?;
        let (epoch_i, epoch_y) = interval(&self.epochs_ns, epoch_ns, "timestamp")?;
        let (lat_i, lat_y) = interval(&self.latitudes_deg, latitude_deg, "latitude")?;
        let (lon_i, lon_y) = interval(&self.longitudes_deg, longitude_deg, "longitude")?;

        let indices = [epoch_i, lat_i, lon_i];
        let norm_distances = [epoch_y, lat_y, lon_y];
        let shift_norm_distances = [
            1.0 - norm_distances[0],
            1.0 - norm_distances[1],
            1.0 - norm_distances[2],
        ];
        let shift_indices = [indices[0] + 1, indices[1] + 1, indices[2] + 1];

        let mut value = 0.0;
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    let i0 = if a == 0 { indices[0] } else { shift_indices[0] };
                    let i1 = if b == 0 { indices[1] } else { shift_indices[1] };
                    let i2 = if c == 0 { indices[2] } else { shift_indices[2] };
                    let w0 = if a == 0 {
                        shift_norm_distances[0]
                    } else {
                        norm_distances[0]
                    };
                    let w1 = if b == 0 {
                        shift_norm_distances[1]
                    } else {
                        norm_distances[1]
                    };
                    let w2 = if c == 0 {
                        shift_norm_distances[2]
                    } else {
                        norm_distances[2]
                    };

                    let mut weight = 1.0;
                    weight *= w0;
                    weight *= w1;
                    weight *= w2;
                    let term = self.value_at(i0, i1, i2) * weight;
                    value += term;
                }
            }
        }
        Ok(value)
    }

    fn value_at(&self, epoch_i: usize, lat_i: usize, lon_i: usize) -> f64 {
        let n_lat = self.latitudes_deg.len();
        let n_lon = self.longitudes_deg.len();
        self.values[(epoch_i * n_lat + lat_i) * n_lon + lon_i]
    }
}

pub fn iono_delay_xyz<F>(
    grid: &TecGrid,
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
) -> Result<f64, String>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    validate_frequency(options.frequency_hz)?;

    let (_vtec, stec) = tec_xyz(grid, options, sat_xyz, receiver_xyz, ecef_to_lla)?;
    let delay_m = IONOSPHERE_CONSTANT * stec / (options.frequency_hz * options.frequency_hz);
    validate::finite(delay_m, "ionosphere_delay_m")
        .map_err(field_error_string)
        .map(|_| delay_m)
}

pub fn tec_xyz<F>(
    grid: &TecGrid,
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
) -> Result<(f64, f64), String>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let shell_radius_m = validate_tec_geometry_inputs(options, sat_xyz, receiver_xyz)?;
    let (_pp_xyz, pp_lonlatalt, mut elevation_rad) =
        pierce_point_with_shell_radius(sat_xyz, receiver_xyz, shell_radius_m, &ecef_to_lla);
    if elevation_rad < options.min_elevation_rad {
        elevation_rad = options.min_elevation_rad;
    }
    validate::finite(elevation_rad, "elevation_rad").map_err(field_error_string)?;

    let pp_lonlatalt = if pp_lonlatalt.iter().any(|v| v.is_nan()) {
        let receiver_lonlatalt = ecef_to_lla(receiver_xyz);
        [
            receiver_lonlatalt[0],
            receiver_lonlatalt[1],
            options.nan_pierce_point_height_m,
        ]
    } else {
        pp_lonlatalt
    };

    let vtec = grid.vtec_at_pierce_point(options.epoch, pp_lonlatalt[0], pp_lonlatalt[1])?;
    validate::finite(vtec, "vtec").map_err(field_error_string)?;
    let obliquity_arg =
        options.shell_geometry.earth_radius_m * elevation_rad.cos() / shell_radius_m;
    validate::finite(obliquity_arg, "obliquity_arg").map_err(field_error_string)?;
    let mapping_denominator = 1.0 - obliquity_arg * obliquity_arg;
    validate::finite_positive(mapping_denominator, "TEC mapping denominator")
        .map_err(field_error_string)?;
    let stec = vtec / mapping_denominator.sqrt();
    validate::finite(stec, "stec").map_err(field_error_string)?;
    Ok((vtec, stec))
}

pub fn pierce_point_with_shell_radius<F>(
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    shell_radius_m: f64,
    ecef_to_lla: F,
) -> ([f64; 3], [f64; 3], f64)
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let receiver_sat_vector = [
        sat_xyz[0] - receiver_xyz[0],
        sat_xyz[1] - receiver_xyz[1],
        sat_xyz[2] - receiver_xyz[2],
    ];

    let receiver_up = unit_vector(receiver_xyz);
    let sat_unit = unit_vector(&receiver_sat_vector);
    let elevation_rad = dot_three_fused(&sat_unit, &receiver_up).asin();

    let a = 1.0;
    let b = 2.0 * dot_three_fused(receiver_xyz, &sat_unit);
    let c = dot_three_fused(receiver_xyz, receiver_xyz) - shell_radius_m * shell_radius_m;
    let t = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);

    let pp_xyz = [
        receiver_xyz[0] + t * sat_unit[0],
        receiver_xyz[1] + t * sat_unit[1],
        receiver_xyz[2] + t * sat_unit[2],
    ];
    let pp_lonlatalt = ecef_to_lla(&pp_xyz);
    (pp_xyz, pp_lonlatalt, elevation_rad)
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn strictly_increasing(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[1] > w[0])
}

fn finite_query_value(value: f64, name: &'static str) -> Result<f64, String> {
    validate::finite(value, name).map_err(field_error_string)
}

fn field_error_string(error: validate::FieldError) -> String {
    format!("{} {}", error.field(), error.reason())
}

fn validate_frequency(frequency_hz: f64) -> Result<(), String> {
    validate::finite_positive(frequency_hz, "frequency_hz")
        .map(|_| ())
        .map_err(field_error_string)
}

fn validate_tec_geometry_inputs(
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
) -> Result<f64, String> {
    validate::finite_vec3(*sat_xyz, "satellite_xyz").map_err(field_error_string)?;
    validate::finite_vec3(*receiver_xyz, "receiver_xyz").map_err(field_error_string)?;
    validate::finite(options.min_elevation_rad, "min_elevation_rad").map_err(field_error_string)?;
    validate::finite(
        options.nan_pierce_point_height_m,
        "nan_pierce_point_height_m",
    )
    .map_err(field_error_string)?;
    validate::finite_positive(options.shell_geometry.earth_radius_m, "earth_radius_m")
        .map_err(field_error_string)?;
    validate::finite_nonneg(options.shell_geometry.shell_height_m, "shell_height_m")
        .map_err(field_error_string)?;

    let shell_radius_m = options.shell_geometry.shell_radius_m();
    validate::finite_positive(shell_radius_m, "shell_radius_m").map_err(field_error_string)?;

    let receiver_radius_m = dot_three_fused(receiver_xyz, receiver_xyz).sqrt();
    validate::finite_positive(receiver_radius_m, "receiver radius_m")
        .map_err(field_error_string)?;

    let line_of_sight_m = [
        sat_xyz[0] - receiver_xyz[0],
        sat_xyz[1] - receiver_xyz[1],
        sat_xyz[2] - receiver_xyz[2],
    ];
    validate::finite_vec3(line_of_sight_m, "line of sight_m").map_err(field_error_string)?;
    let line_of_sight_norm_m = dot_three_fused(&line_of_sight_m, &line_of_sight_m).sqrt();
    validate::finite_positive(line_of_sight_norm_m, "line of sight_m")
        .map_err(field_error_string)?;

    Ok(shell_radius_m)
}

fn interval(axis: &[f64], x: f64, name: &str) -> Result<(usize, f64), String> {
    if x < axis[0] || x > axis[axis.len() - 1] {
        return Err(format!("{name} {x} is out of TEC grid bounds"));
    }
    let upper = axis.partition_point(|v| *v <= x);
    let mut lower = upper.saturating_sub(1);
    if lower >= axis.len() - 1 {
        lower = axis.len() - 2;
    }
    let y = (x - axis[lower]) / (axis[lower + 1] - axis[lower]);
    Ok((lower, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_grid() -> TecGrid {
        TecGrid::new(
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![20.0, 30.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect("small TEC grid")
    }

    #[test]
    fn interpolate_vtec_rejects_non_finite_query_coordinates() {
        let grid = small_grid();
        let cases = [
            (f64::NAN, 5.0, 25.0, "timestamp"),
            (f64::INFINITY, 5.0, 25.0, "timestamp"),
            (5.0, f64::NAN, 25.0, "latitude"),
            (5.0, 5.0, f64::NAN, "longitude"),
        ];

        for (epoch_ns, latitude_deg, longitude_deg, field) in cases {
            let error = grid
                .interpolate_vtec(epoch_ns, latitude_deg, longitude_deg)
                .expect_err("non-finite TEC coordinate must be rejected");
            assert!(error.contains(field), "{error}");
            assert!(error.contains("not finite"), "{error}");
        }
    }

    #[test]
    fn interpolate_vtec_valid_query_still_interpolates() {
        let grid = small_grid();

        assert_eq!(
            grid.interpolate_vtec(0.0, 0.0, 20.0)
                .expect("lower corner")
                .to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(
            grid.interpolate_vtec(5.0, 5.0, 25.0)
                .expect("center point")
                .to_bits(),
            4.5f64.to_bits()
        );
    }

    #[test]
    fn tec_grid_rejects_nonfinite_values() {
        let error = TecGrid::new(
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![20.0, 30.0],
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect_err("nonfinite TEC grid cells must be rejected");

        assert!(error.contains("TEC grid values"), "{error}");
        assert!(error.contains("not finite"), "{error}");
    }

    #[test]
    fn tec_xyz_rejects_degenerate_geometry_without_nonfinite_success() {
        fn passthrough_lla(xyz: &[f64; 3]) -> [f64; 3] {
            [xyz[0], xyz[1], xyz[2]]
        }

        let grid = TecGrid::new(
            vec![0.0, 1.0],
            vec![-10.0, 10.0],
            vec![0.0, 20.0],
            vec![0.0; 8],
        )
        .expect("regular TEC grid");
        let mut options = TecGridEvalOptions::l1(TecGridEpoch::new(0, 0));
        options.min_elevation_rad = 0.0;
        options.nan_pierce_point_height_m = 0.0;

        let error = tec_xyz(
            &grid,
            options,
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            passthrough_lla,
        )
        .expect_err("zero receiver and satellite vectors must be rejected");

        assert!(error.contains("receiver radius_m"), "{error}");
        assert!(error.contains("not positive"), "{error}");
    }

    #[test]
    fn iono_delay_xyz_rejects_invalid_frequency() {
        fn passthrough_lla(_: &[f64; 3]) -> [f64; 3] {
            [25.0, 5.0, IONOSPHERE_HEIGHT_M]
        }

        let grid = small_grid();
        let sat_xyz = [2.0, 0.0, 0.0];
        let receiver_xyz = [1.0, 0.0, 0.0];
        for (frequency_hz, reason) in [(0.0, "not positive"), (f64::NAN, "not finite")] {
            let mut options = TecGridEvalOptions::l1(TecGridEpoch::new(0, 1));
            options.frequency_hz = frequency_hz;

            let error = iono_delay_xyz(&grid, options, &sat_xyz, &receiver_xyz, passthrough_lla)
                .expect_err("invalid TEC-grid frequency must be rejected");
            assert!(error.contains("frequency_hz"), "{error}");
            assert!(error.contains(reason), "{error}");
        }
    }
}
