//! Portable scoring and input readers for the PPC moving-RTK dataset.
//!
//! The module deliberately does not download or vendor dataset files. Callers
//! provide local reference and solution paths, together with provenance fields
//! that can be recorded in a CI artifact.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sidereon_core::astro::time::gnss::{seconds_of_week_from_calendar, week_from_calendar};
use sidereon_core::astro::time::{days_in_month, gps_utc_offset_s, split_julian_date, TimeScale};
use sidereon_core::constants::{GPS_EPOCH_TO_J2000_S, SECONDS_PER_WEEK};
use sidereon_core::inertial::ImuSample;

/// PPC's published three-dimensional error threshold, in metres.
pub const PPC_DEFAULT_THRESHOLD_M: f64 = 0.5;

/// Stable identifier for the Sidereon PPC scorer.
pub const PPC_SCORER_VERSION: &str = "sidereon-ppc-v1";

/// One reference-trajectory sample.
#[derive(Debug, Clone, PartialEq)]
pub struct PpcTruthSample {
    /// GPS week when present in the source CSV.
    pub gps_week: Option<u32>,
    /// GPS time of week, seconds.
    pub tow_s: f64,
    /// Reference antenna phase-centre ECEF coordinates, metres.
    pub ecef_m: [f64; 3],
}

/// One solution sample used by the causal scorer.
#[derive(Debug, Clone, PartialEq)]
pub struct PpcSolutionSample {
    /// GPS time of week, seconds.
    pub tow_s: f64,
    /// Solution ECEF coordinates, metres.
    pub ecef_m: [f64; 3],
}

/// One PPC IMU sample normalized to Sidereon's SI units.
#[derive(Debug, Clone, PartialEq)]
pub struct PpcImuSample {
    /// GPS week when present in the source CSV.
    pub gps_week: Option<u32>,
    /// GPS time of week, seconds.
    pub tow_s: f64,
    /// Specific force, metres per second squared.
    pub acceleration_mps2: [f64; 3],
    /// Angular rate, radians per second.
    pub angular_rate_rad_s: [f64; 3],
}

impl PpcImuSample {
    /// Convert this row into Sidereon's rate-form IMU sample.
    ///
    /// The source CSV uses GPS week/TOW and degrees/s for gyro rates; the
    /// returned sample uses seconds since J2000 and radians/s.
    pub fn to_core_sample(&self) -> ImuSample {
        let continuous_gps_s = self
            .gps_week
            .map(|week| f64::from(week) * SECONDS_PER_WEEK + self.tow_s)
            .unwrap_or(self.tow_s);
        ImuSample::rate(
            continuous_gps_s - GPS_EPOCH_TO_J2000_S,
            self.acceleration_mps2,
            self.angular_rate_rad_s,
        )
    }
}

/// A pair of files representing one PPC route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpcRouteInput {
    /// Stable route name, for example `nagoya-run1`.
    pub name: String,
    /// PPC `reference.csv` path.
    pub truth_path: PathBuf,
    /// RTKLIB `.pos` or supported solution CSV path.
    pub solution_path: PathBuf,
}

/// Distance-weighted result for one PPC route.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PpcRouteScore {
    /// Route name supplied by the caller.
    pub route: String,
    /// Percentage of traveled distance with 3D error at or below the threshold.
    pub score_percent: f64,
    /// Distance counted as successful, metres.
    pub good_distance_m: f64,
    /// Total truth-trajectory distance scored, metres.
    pub total_distance_m: f64,
    /// Number of truth intervals counted as successful.
    pub good_samples: usize,
    /// Number of truth intervals included in the denominator.
    pub total_samples: usize,
    /// Truth intervals for which no causal solution sample existed.
    pub missing_solution_samples: usize,
    /// Number of truth samples read from the reference file.
    pub truth_samples: usize,
    /// Number of usable solution samples read from the solution file.
    pub solution_samples: usize,
    /// Number of scored intervals with a finite solution error.
    pub finite_error_samples: usize,
    /// Median finite 3D error, metres.
    pub median_error_m: Option<f64>,
    /// RMS finite 3D error, metres.
    pub rms_error_m: Option<f64>,
    /// 95th-percentile finite 3D error, metres.
    pub p95_error_m: Option<f64>,
    /// Maximum finite 3D error, metres.
    pub max_error_m: Option<f64>,
}

/// Reproducibility metadata attached to a PPC report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PpcRunMetadata {
    /// Scorer implementation identifier.
    pub scorer_version: String,
    /// Sidereon scoring executable package version.
    pub sidereon_version: String,
    /// Source-control commit when supplied by the build or caller.
    pub git_commit: Option<String>,
    /// Dataset revision or archive identifier supplied by the caller.
    pub dataset_revision: Option<String>,
    /// SHA-256 digest of the dataset archive supplied by the caller.
    pub dataset_sha256: Option<String>,
    /// Error threshold used by the scorer, metres.
    pub threshold_m: f64,
}

/// Complete PPC scoring report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PpcScoreReport {
    /// Reproducibility metadata.
    pub metadata: PpcRunMetadata,
    /// Per-route scores in caller order.
    pub routes: Vec<PpcRouteScore>,
    /// Unweighted mean of the route percentages, as specified by PPC.
    pub average_score_percent: f64,
}

/// Errors raised by PPC readers and scoring.
#[derive(Debug, thiserror::Error)]
pub enum PpcError {
    /// A file could not be read.
    #[error("read {path}: {source}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// CSV syntax or decoding failed.
    #[error("parse CSV {path}: {source}")]
    Csv {
        /// Path that failed.
        path: String,
        /// Underlying CSV error.
        #[source]
        source: csv::Error,
    },
    /// A required CSV column was absent or incomplete.
    #[error("{path}: missing CSV column {column}")]
    MissingColumn {
        /// Path that failed.
        path: String,
        /// Required column name or complete column group.
        column: String,
    },
    /// A required field could not be converted to a finite value.
    #[error("{path}: invalid {field} at row {row}: {value}")]
    InvalidField {
        /// Path that failed.
        path: String,
        /// Field label.
        field: String,
        /// One-based source row number.
        row: usize,
        /// Original field text.
        value: String,
    },
    /// Usable timestamps were duplicate or regressed in source order.
    #[error("{path}: non-monotonic {kind} timestamp at row {row}: {current_s} after {previous_s}")]
    NonMonotonicTimestamp {
        /// Path that failed.
        path: String,
        /// Sample family.
        kind: &'static str,
        /// One-based source row number.
        row: usize,
        /// Previous normalized timestamp.
        previous_s: f64,
        /// Current normalized timestamp.
        current_s: f64,
    },
    /// A nominally single-route input crossed GPS week boundaries.
    #[error(
        "{path}: {kind} spans multiple GPS weeks at row {row}: week {current_week} after week {expected_week}"
    )]
    MultipleGpsWeeks {
        /// Path that failed.
        path: String,
        /// Sample family.
        kind: &'static str,
        /// One-based source row number.
        row: usize,
        /// GPS week established by the first usable row.
        expected_week: u32,
        /// GPS week on the rejected row.
        current_week: u32,
    },
    /// Truth and solution inputs declared different GPS weeks.
    #[error(
        "PPC GPS week mismatch: reference {truth_path} is week {truth_week}, solution {solution_path} is week {solution_week}"
    )]
    GpsWeekMismatch {
        /// Reference path.
        truth_path: String,
        /// Solution path.
        solution_path: String,
        /// GPS week declared by the reference.
        truth_week: u32,
        /// GPS week declared by the solution.
        solution_week: u32,
    },
    /// The scoring threshold was non-finite or not positive.
    #[error("invalid PPC threshold {threshold_m}: must be finite and positive")]
    InvalidThreshold {
        /// Rejected threshold, metres.
        threshold_m: f64,
    },
    /// A reader produced no usable samples.
    #[error("{path}: no usable {kind} samples")]
    NoSamples {
        /// Path that failed.
        path: String,
        /// Sample family.
        kind: &'static str,
    },
    /// The solution file did not look like a supported format.
    #[error("{path}: unsupported solution format")]
    UnsupportedSolutionFormat {
        /// Path that failed.
        path: String,
    },
}

/// Read PPC `reference.csv` into ECEF truth samples.
///
/// Reference rows are authoritative and therefore parsed strictly. Timestamps
/// must be strictly increasing in source order; the reader never sorts them.
/// PPC routes are single-week inputs, so a file that changes GPS week is
/// rejected instead of ambiguously reducing absolute epochs to TOW.
pub fn read_reference_csv(path: &Path) -> Result<Vec<PpcTruthSample>, PpcError> {
    let mut reader = csv_reader(path)?;
    let headers = csv_headers(&mut reader, path)?;
    let tow = required_header(&headers, path, "GPS TOW (s)")?;
    let week = find_header(&headers, "GPS Week");
    let lat = required_header(&headers, path, "Latitude (deg)")?;
    let lon = required_header(&headers, path, "Longitude (deg)")?;
    let height = required_header(&headers, path, "Ellipsoid Height (m)")?;
    let mut samples = Vec::new();
    let mut previous_tow = None;
    let mut route_week = None;

    for (row_index, record) in reader.records().enumerate() {
        let row = row_index + 2;
        let record = record.map_err(|source| csv_error(path, source))?;
        let tow_s = tow_field(path, row, record.get(tow).unwrap_or(""))?;
        ensure_monotonic(path, "reference", row, &mut previous_tow, tow_s)?;
        let lat_deg = finite_field(path, "Latitude (deg)", row, record.get(lat).unwrap_or(""))?;
        let lon_deg = finite_field(path, "Longitude (deg)", row, record.get(lon).unwrap_or(""))?;
        let height_m = finite_field(
            path,
            "Ellipsoid Height (m)",
            row,
            record.get(height).unwrap_or(""),
        )?;
        let gps_week = week
            .map(|index| u32_field(path, "GPS Week", row, record.get(index).unwrap_or("")))
            .transpose()?;
        ensure_single_week(path, "reference", row, &mut route_week, gps_week)?;
        let ecef_m = llh_to_ecef_checked(lat_deg, lon_deg, height_m).ok_or_else(|| {
            PpcError::InvalidField {
                path: path.display().to_string(),
                field: "reference geodetic coordinates".to_string(),
                row,
                value: format!("{lat_deg},{lon_deg},{height_m}"),
            }
        })?;
        samples.push(PpcTruthSample {
            gps_week,
            tow_s,
            ecef_m,
        });
    }

    if samples.is_empty() {
        return Err(PpcError::NoSamples {
            path: path.display().to_string(),
            kind: "reference",
        });
    }
    Ok(samples)
}

/// Read a PPC IMU CSV and convert angular rates from degrees/s to radians/s.
pub fn read_imu_csv(path: &Path) -> Result<Vec<PpcImuSample>, PpcError> {
    let mut reader = csv_reader(path)?;
    let headers = csv_headers(&mut reader, path)?;
    let tow = required_header(&headers, path, "GPS TOW (s)")?;
    let week = find_header(&headers, "GPS Week");
    let ax = required_header(&headers, path, "Acc X (m/s^2)")?;
    let ay = required_header(&headers, path, "Acc Y (m/s^2)")?;
    let az = required_header(&headers, path, "Acc Z (m/s^2)")?;
    let gx = required_header(&headers, path, "Ang Rate X (deg/s)")?;
    let gy = required_header(&headers, path, "Ang Rate Y (deg/s)")?;
    let gz = required_header(&headers, path, "Ang Rate Z (deg/s)")?;
    let mut samples = Vec::new();
    let mut previous_tow = None;

    for (row_index, record) in reader.records().enumerate() {
        let row = row_index + 2;
        let record = record.map_err(|source| csv_error(path, source))?;
        let tow_s = tow_field(path, row, record.get(tow).unwrap_or(""))?;
        ensure_monotonic(path, "IMU", row, &mut previous_tow, tow_s)?;
        let field = |index: usize, name: &str| {
            finite_field(path, name, row, record.get(index).unwrap_or(""))
        };
        let deg_to_rad = std::f64::consts::PI / 180.0;
        samples.push(PpcImuSample {
            gps_week: week
                .map(|index| u32_field(path, "GPS Week", row, record.get(index).unwrap_or("")))
                .transpose()?,
            tow_s,
            acceleration_mps2: [
                field(ax, "Acc X (m/s^2)")?,
                field(ay, "Acc Y (m/s^2)")?,
                field(az, "Acc Z (m/s^2)")?,
            ],
            angular_rate_rad_s: [
                field(gx, "Ang Rate X (deg/s)")? * deg_to_rad,
                field(gy, "Ang Rate Y (deg/s)")? * deg_to_rad,
                field(gz, "Ang Rate Z (deg/s)")? * deg_to_rad,
            ],
        });
    }

    if samples.is_empty() {
        return Err(PpcError::NoSamples {
            path: path.display().to_string(),
            kind: "IMU",
        });
    }
    Ok(samples)
}

/// Read an RTKLIB `.pos` file with LLH or ECEF coordinates.
///
/// Both week/TOW rows and calendar rows are accepted. Calendar rows use the
/// `% GPST` or `% UTC` header label; UTC epochs are shifted onto the GPST
/// time-of-week axis using Sidereon's leap-second table. Standard RTKLIB header
/// labels select LLH versus ECEF coordinates. Malformed/non-finite solution
/// rows are filtered, while usable rows must remain strictly ordered. Like the
/// PPC route scorer, this reader rejects inputs spanning more than one GPS week.
pub fn read_rtklib_pos(path: &Path) -> Result<Vec<PpcSolutionSample>, PpcError> {
    read_rtklib_pos_file(path).map(|file| file.samples)
}

fn read_rtklib_pos_file(path: &Path) -> Result<ParsedSolutionFile, PpcError> {
    let text = std::fs::read_to_string(path).map_err(|source| PpcError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut samples = Vec::new();
    let mut time_scale = None;
    let mut coordinates = None;
    let mut previous_tow = None;
    let mut route_week = None;

    for (line_index, line) in text.lines().enumerate() {
        let row = line_index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('%') {
            time_scale = pos_header_time_scale(trimmed).or(time_scale);
            coordinates = pos_header_coordinates(trimmed).or(coordinates);
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        let Some(parsed) = parse_pos_row(
            &fields,
            time_scale,
            coordinates.unwrap_or(PosCoordinates::Llh),
        ) else {
            continue;
        };
        ensure_single_week(
            path,
            "solution",
            row,
            &mut route_week,
            parsed.timestamp.gps_week,
        )?;
        ensure_monotonic(
            path,
            "solution",
            row,
            &mut previous_tow,
            parsed.timestamp.tow_s,
        )?;
        samples.push(parsed.sample);
    }

    if samples.is_empty() {
        return Err(PpcError::NoSamples {
            path: path.display().to_string(),
            kind: "solution",
        });
    }
    Ok(ParsedSolutionFile {
        samples,
        gps_week: route_week,
    })
}

/// Read either an RTKLIB `.pos` file or a supported solution CSV.
pub fn read_solution(path: &Path) -> Result<Vec<PpcSolutionSample>, PpcError> {
    read_solution_file(path).map(|file| file.samples)
}

fn read_solution_file(path: &Path) -> Result<ParsedSolutionFile, PpcError> {
    let bytes = std::fs::read(path).map_err(|source| PpcError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let text = String::from_utf8_lossy(&bytes);
    let first_data = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('%'));
    let Some(first_data) = first_data else {
        return Err(PpcError::UnsupportedSolutionFormat {
            path: path.display().to_string(),
        });
    };
    if first_data.contains(',') {
        return read_solution_csv_file(path);
    }
    read_rtklib_pos_file(path)
}

/// Score one route with causal zero-order-hold matching and distance weighting.
///
/// Inputs are expected to represent one GPS week and be in strictly increasing
/// time-of-week order, as produced by this module's readers. A truth interval is
/// classified using the newest solution at or before its endpoint; no
/// interpolation or future solution is used. Intervals without such a solution
/// stay in the denominator.
pub fn score_route(
    route: impl Into<String>,
    truth: &[PpcTruthSample],
    solution: &[PpcSolutionSample],
    threshold_m: f64,
) -> PpcRouteScore {
    let mut solution_index = 0usize;
    let mut last_solution = None;
    let mut good_distance_m = 0.0;
    let mut total_distance_m = 0.0;
    let mut good_samples = 0usize;
    let mut total_samples = 0usize;
    let mut missing_solution_samples = 0usize;
    let mut errors_m = Vec::new();

    for index in 1..truth.len() {
        let current = &truth[index];
        while solution_index < solution.len() && solution[solution_index].tow_s <= current.tow_s {
            last_solution = Some(&solution[solution_index]);
            solution_index += 1;
        }
        let distance_m = norm3(current.ecef_m, truth[index - 1].ecef_m);
        if !distance_m.is_finite() {
            continue;
        }
        total_distance_m += distance_m;
        total_samples += 1;
        let Some(solution_sample) = last_solution else {
            missing_solution_samples += 1;
            continue;
        };
        let error_m = norm3(solution_sample.ecef_m, current.ecef_m);
        if !error_m.is_finite() {
            continue;
        }
        errors_m.push(error_m);
        if error_m <= threshold_m {
            good_distance_m += distance_m;
            good_samples += 1;
        }
    }

    errors_m.sort_by(f64::total_cmp);
    let median_error_m = percentile_sorted(&errors_m, 0.5);
    let p95_error_m = percentile_sorted(&errors_m, 0.95);
    let rms_error_m = if errors_m.is_empty() {
        None
    } else {
        let direct = (errors_m.iter().map(|error| error * error).sum::<f64>()
            / errors_m.len() as f64)
            .sqrt();
        Some(if direct.is_finite() {
            direct
        } else {
            stable_rms(&errors_m)
        })
    };
    let max_error_m = errors_m.last().copied();
    let score_percent = if total_distance_m > 0.0 {
        100.0 * good_distance_m / total_distance_m
    } else {
        0.0
    };
    PpcRouteScore {
        route: route.into(),
        score_percent,
        good_distance_m,
        total_distance_m,
        good_samples,
        total_samples,
        missing_solution_samples,
        truth_samples: truth.len(),
        solution_samples: solution.len(),
        finite_error_samples: errors_m.len(),
        median_error_m,
        rms_error_m,
        p95_error_m,
        max_error_m,
    }
}

/// Read and score a list of routes in the supplied order.
pub fn score_routes(
    routes: &[PpcRouteInput],
    threshold_m: f64,
    mut metadata: PpcRunMetadata,
) -> Result<PpcScoreReport, PpcError> {
    if !threshold_m.is_finite() || threshold_m <= 0.0 {
        return Err(PpcError::InvalidThreshold { threshold_m });
    }
    metadata.threshold_m = threshold_m;
    let mut scores = Vec::with_capacity(routes.len());
    for route in routes {
        let truth = read_reference_csv(&route.truth_path)?;
        let solution = read_solution_file(&route.solution_path)?;
        let truth_week = truth.first().and_then(|sample| sample.gps_week);
        if let (Some(truth_week), Some(solution_week)) = (truth_week, solution.gps_week) {
            if truth_week != solution_week {
                return Err(PpcError::GpsWeekMismatch {
                    truth_path: route.truth_path.display().to_string(),
                    solution_path: route.solution_path.display().to_string(),
                    truth_week,
                    solution_week,
                });
            }
        }
        scores.push(score_route(
            &route.name,
            &truth,
            &solution.samples,
            threshold_m,
        ));
    }
    let average_score_percent = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|score| score.score_percent).sum::<f64>() / scores.len() as f64
    };
    Ok(PpcScoreReport {
        metadata,
        routes: scores,
        average_score_percent,
    })
}

#[derive(Debug)]
struct ParsedSolutionFile {
    samples: Vec<PpcSolutionSample>,
    gps_week: Option<u32>,
}

fn read_solution_csv_file(path: &Path) -> Result<ParsedSolutionFile, PpcError> {
    let mut reader = csv_reader(path)?;
    let headers = csv_headers(&mut reader, path)?;
    let timestamp =
        if let Some(index) = find_header_any(&headers, &["GPS TOW (s)", "tow_s", "time_s"]) {
            SolutionTimestamp::Tow {
                tow: index,
                week: find_header(&headers, "GPS Week"),
            }
        } else if let Some(index) = find_header_any(
            &headers,
            &["Time (UTC)", "UTC", "utc", "timestamp_utc", "time_utc"],
        ) {
            SolutionTimestamp::Utc(index)
        } else {
            return Err(PpcError::MissingColumn {
                path: path.display().to_string(),
                column: "GPS TOW (s), tow_s, or Time (UTC)".to_string(),
            });
        };

    let x = find_header_any(&headers, &["ECEF X (m)", "x_m", "x"]);
    let y = find_header_any(&headers, &["ECEF Y (m)", "y_m", "y"]);
    let z = find_header_any(&headers, &["ECEF Z (m)", "z_m", "z"]);
    let lat = find_header_any(&headers, &["Latitude (deg)", "lat_deg", "lat"]);
    let lon = find_header_any(&headers, &["Longitude (deg)", "lon_deg", "lon"]);
    let height = find_header_any(&headers, &["Ellipsoid Height (m)", "height_m", "height"]);
    let coordinates = match (x, y, z) {
        (Some(x), Some(y), Some(z)) => SolutionCoordinates::Ecef([x, y, z]),
        (None, None, None) => match (lat, lon, height) {
            (Some(lat), Some(lon), Some(height)) => SolutionCoordinates::Llh([lat, lon, height]),
            (None, None, None) => {
                return Err(PpcError::MissingColumn {
                    path: path.display().to_string(),
                    column: "ECEF X/Y/Z or latitude/longitude/height".to_string(),
                })
            }
            _ => {
                return Err(PpcError::MissingColumn {
                    path: path.display().to_string(),
                    column: "complete latitude/longitude/height triplet".to_string(),
                })
            }
        },
        _ => {
            return Err(PpcError::MissingColumn {
                path: path.display().to_string(),
                column: "complete ECEF X/Y/Z triplet".to_string(),
            })
        }
    };

    let mut samples = Vec::new();
    let mut previous_tow = None;
    let mut route_week = None;
    for (row_index, record) in reader.records().enumerate() {
        let row = row_index + 2;
        let record = record.map_err(|source| csv_error(path, source))?;
        let Some(parsed_timestamp) = timestamp.parse(&record) else {
            continue;
        };
        let Some(ecef_m) = coordinates.parse(&record) else {
            continue;
        };
        ensure_single_week(
            path,
            "solution",
            row,
            &mut route_week,
            parsed_timestamp.gps_week,
        )?;
        ensure_monotonic(
            path,
            "solution",
            row,
            &mut previous_tow,
            parsed_timestamp.tow_s,
        )?;
        samples.push(PpcSolutionSample {
            tow_s: parsed_timestamp.tow_s,
            ecef_m,
        });
    }
    if samples.is_empty() {
        return Err(PpcError::NoSamples {
            path: path.display().to_string(),
            kind: "solution",
        });
    }
    Ok(ParsedSolutionFile {
        samples,
        gps_week: route_week,
    })
}

#[derive(Debug, Clone, Copy)]
enum SolutionTimestamp {
    Tow { tow: usize, week: Option<usize> },
    Utc(usize),
}

impl SolutionTimestamp {
    fn parse(self, record: &csv::StringRecord) -> Option<NormalizedTimestamp> {
        match self {
            Self::Tow { tow, week } => Some(NormalizedTimestamp {
                gps_week: match week {
                    Some(index) => Some(record.get(index)?.trim().parse::<u32>().ok()?),
                    None => None,
                },
                tow_s: parse_tow(record.get(tow)?)?,
            }),
            Self::Utc(index) => parse_utc_timestamp(record.get(index)?),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SolutionCoordinates {
    Ecef([usize; 3]),
    Llh([usize; 3]),
}

impl SolutionCoordinates {
    fn parse(self, record: &csv::StringRecord) -> Option<[f64; 3]> {
        let indices = match self {
            Self::Ecef(indices) | Self::Llh(indices) => indices,
        };
        let values = [
            parse_finite(record.get(indices[0])?)?,
            parse_finite(record.get(indices[1])?)?,
            parse_finite(record.get(indices[2])?)?,
        ];
        match self {
            Self::Ecef(_) => valid_ecef(values).then_some(values),
            Self::Llh(_) => llh_to_ecef_checked(values[0], values[1], values[2]),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NormalizedTimestamp {
    gps_week: Option<u32>,
    tow_s: f64,
}

#[derive(Debug)]
struct ParsedSolutionRow {
    timestamp: NormalizedTimestamp,
    sample: PpcSolutionSample,
}

#[derive(Debug, Clone, Copy)]
enum PosTimeScale {
    Gpst,
    Utc,
}

#[derive(Debug, Clone, Copy)]
enum PosCoordinates {
    Llh,
    Ecef,
}

fn parse_pos_row(
    fields: &[&str],
    time_scale: Option<PosTimeScale>,
    coordinates: PosCoordinates,
) -> Option<ParsedSolutionRow> {
    if fields.len() < 5 {
        return None;
    }
    let timestamp = if let Ok(gps_week) = fields[0].parse::<u32>() {
        NormalizedTimestamp {
            gps_week: Some(gps_week),
            tow_s: parse_tow(fields[1])?,
        }
    } else {
        parse_calendar_timestamp(fields[0], fields[1], time_scale?)?
    };
    let values = [
        parse_finite(fields[2])?,
        parse_finite(fields[3])?,
        parse_finite(fields[4])?,
    ];
    let ecef_m = match coordinates {
        PosCoordinates::Llh => llh_to_ecef_checked(values[0], values[1], values[2])?,
        PosCoordinates::Ecef => valid_ecef(values).then_some(values)?,
    };
    Some(ParsedSolutionRow {
        timestamp,
        sample: PpcSolutionSample {
            tow_s: timestamp.tow_s,
            ecef_m,
        },
    })
}

fn pos_header_time_scale(line: &str) -> Option<PosTimeScale> {
    match line
        .trim_start_matches('%')
        .split_whitespace()
        .next()?
        .to_ascii_uppercase()
        .as_str()
    {
        "GPST" | "GPS" => Some(PosTimeScale::Gpst),
        "UTC" => Some(PosTimeScale::Utc),
        _ => None,
    }
}

fn pos_header_coordinates(line: &str) -> Option<PosCoordinates> {
    let lowercase = line.to_ascii_lowercase();
    if lowercase.contains("x-ecef")
        || (lowercase.contains("x/y/z-ecef") && lowercase.contains("wgs84"))
    {
        Some(PosCoordinates::Ecef)
    } else if lowercase.contains("latitude")
        && lowercase.contains("longitude")
        && lowercase.contains("height")
    {
        Some(PosCoordinates::Llh)
    } else {
        None
    }
}

fn parse_utc_timestamp(value: &str) -> Option<NormalizedTimestamp> {
    let value = value.trim();
    let value = value
        .strip_suffix('Z')
        .or_else(|| value.strip_suffix('z'))
        .unwrap_or(value)
        .trim();
    let value = value
        .strip_suffix("UTC")
        .or_else(|| value.strip_suffix("utc"))
        .unwrap_or(value)
        .trim();
    let (date, time) = value.split_once('T').or_else(|| value.split_once(' '))?;
    parse_calendar_timestamp(date.trim(), time.trim(), PosTimeScale::Utc)
}

fn parse_calendar_timestamp(
    date: &str,
    time: &str,
    scale: PosTimeScale,
) -> Option<NormalizedTimestamp> {
    let separator = if date.contains('/') { '/' } else { '-' };
    let mut date_fields = date.split(separator);
    let year = date_fields.next()?.parse::<i64>().ok()?;
    let month = date_fields.next()?.parse::<i64>().ok()?;
    let day = date_fields.next()?.parse::<i64>().ok()?;
    if date_fields.next().is_some() {
        return None;
    }
    let mut time_fields = time.split(':');
    let hour = time_fields.next()?.parse::<i64>().ok()?;
    let minute = time_fields.next()?.parse::<i64>().ok()?;
    let second = time_fields.next()?.parse::<f64>().ok()?;
    if time_fields.next().is_some()
        || !(1..=9_999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !second.is_finite()
        || !(0.0..60.0).contains(&second)
    {
        return None;
    }
    let mut gps_week = week_from_calendar(TimeScale::Gpst, year, month, day)?;
    let whole_second = second.trunc() as i64;
    let mut tow_s = seconds_of_week_from_calendar(year, month, day, hour, minute, whole_second)
        + second.fract();
    if matches!(scale, PosTimeScale::Utc) {
        let (jd_whole, fraction) = split_julian_date(
            year as i32,
            month as i32,
            day as i32,
            hour as i32,
            minute as i32,
            second,
        );
        tow_s += gps_utc_offset_s(jd_whole + fraction);
    }
    if tow_s >= SECONDS_PER_WEEK {
        tow_s -= SECONDS_PER_WEEK;
        gps_week = gps_week.checked_add(1)?;
    }
    Some(NormalizedTimestamp {
        gps_week: Some(gps_week),
        tow_s,
    })
}

fn csv_reader(path: &Path) -> Result<csv::Reader<std::fs::File>, PpcError> {
    csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|source| csv_error(path, source))
}

fn csv_headers(
    reader: &mut csv::Reader<std::fs::File>,
    path: &Path,
) -> Result<csv::StringRecord, PpcError> {
    reader
        .headers()
        .cloned()
        .map_err(|source| csv_error(path, source))
}

fn csv_error(path: &Path, source: csv::Error) -> PpcError {
    PpcError::Csv {
        path: path.display().to_string(),
        source,
    }
}

fn required_header(
    headers: &csv::StringRecord,
    path: &Path,
    name: &str,
) -> Result<usize, PpcError> {
    find_header(headers, name).ok_or_else(|| PpcError::MissingColumn {
        path: path.display().to_string(),
        column: name.to_string(),
    })
}

fn find_header(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    find_header_any(headers, &[name])
}

fn find_header_any(headers: &csv::StringRecord, names: &[&str]) -> Option<usize> {
    headers.iter().position(|header| {
        let normalized = header.trim().trim_start_matches('\u{feff}');
        names
            .iter()
            .any(|name| normalized.eq_ignore_ascii_case(name))
    })
}

fn finite_field(path: &Path, field: &str, row: usize, value: &str) -> Result<f64, PpcError> {
    parse_finite(value).ok_or_else(|| PpcError::InvalidField {
        path: path.display().to_string(),
        field: field.to_string(),
        row,
        value: value.to_string(),
    })
}

fn tow_field(path: &Path, row: usize, value: &str) -> Result<f64, PpcError> {
    parse_tow(value).ok_or_else(|| PpcError::InvalidField {
        path: path.display().to_string(),
        field: "GPS TOW (s)".to_string(),
        row,
        value: value.to_string(),
    })
}

fn u32_field(path: &Path, field: &str, row: usize, value: &str) -> Result<u32, PpcError> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|_| PpcError::InvalidField {
            path: path.display().to_string(),
            field: field.to_string(),
            row,
            value: value.to_string(),
        })
}

fn parse_finite(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn parse_tow(value: &str) -> Option<f64> {
    parse_finite(value).filter(|value| (0.0..SECONDS_PER_WEEK).contains(value))
}

fn ensure_monotonic(
    path: &Path,
    kind: &'static str,
    row: usize,
    previous: &mut Option<f64>,
    current: f64,
) -> Result<(), PpcError> {
    if let Some(previous_s) = *previous {
        if current <= previous_s {
            return Err(PpcError::NonMonotonicTimestamp {
                path: path.display().to_string(),
                kind,
                row,
                previous_s,
                current_s: current,
            });
        }
    }
    *previous = Some(current);
    Ok(())
}

fn ensure_single_week(
    path: &Path,
    kind: &'static str,
    row: usize,
    expected: &mut Option<u32>,
    current: Option<u32>,
) -> Result<(), PpcError> {
    let Some(current_week) = current else {
        return Ok(());
    };
    if let Some(expected_week) = *expected {
        if current_week != expected_week {
            return Err(PpcError::MultipleGpsWeeks {
                path: path.display().to_string(),
                kind,
                row,
                expected_week,
                current_week,
            });
        }
    } else {
        *expected = Some(current_week);
    }
    Ok(())
}

fn llh_to_ecef_checked(lat_deg: f64, lon_deg: f64, height_m: f64) -> Option<[f64; 3]> {
    if !(-90.0..=90.0).contains(&lat_deg) || !(-180.0..=180.0).contains(&lon_deg) {
        return None;
    }
    let ecef_m = llh_to_ecef(lat_deg, lon_deg, height_m);
    valid_ecef(ecef_m).then_some(ecef_m)
}

fn valid_ecef(ecef_m: [f64; 3]) -> bool {
    ecef_m.iter().all(|value| value.is_finite()) && norm3(ecef_m, [0.0; 3]).is_finite()
}

fn llh_to_ecef(lat_deg: f64, lon_deg: f64, height_m: f64) -> [f64; 3] {
    let a = 6_378_137.0;
    let flattening = 1.0 / 298.257_223_563;
    let e2 = flattening * (2.0 - flattening);
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    [
        (n + height_m) * cos_lat * lon.cos(),
        (n + height_m) * cos_lat * lon.sin(),
        (n * (1.0 - e2) + height_m) * sin_lat,
    ]
}

fn norm3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn stable_rms(values: &[f64]) -> f64 {
    let scale = values.iter().copied().fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let normalized_sum = values
        .iter()
        .map(|value| (value / scale).powi(2))
        .sum::<f64>();
    scale * (normalized_sum / values.len() as f64).sqrt()
}

fn percentile_sorted(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * probability).round() as usize;
    sorted.get(index).copied()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn temp_file(name: &str, text: &str) -> PathBuf {
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sidereon-ppc-test-{}-{nonce}-{name}",
            std::process::id()
        ));
        fs::write(&path, text).expect("write PPC test file");
        path
    }

    fn truth(tow_s: f64, x_m: f64) -> PpcTruthSample {
        PpcTruthSample {
            gps_week: None,
            tow_s,
            ecef_m: [x_m, 0.0, 0.0],
        }
    }

    fn solution(tow_s: f64, x_m: f64) -> PpcSolutionSample {
        PpcSolutionSample {
            tow_s,
            ecef_m: [x_m, 0.0, 0.0],
        }
    }

    #[test]
    fn score_uses_latest_solution_not_later_than_truth() {
        let truth = vec![truth(0.0, 0.0), truth(1.0, 1.0), truth(2.0, 2.0)];
        let solution = vec![solution(1.5, 2.0)];
        let score = score_route("test", &truth, &solution, PPC_DEFAULT_THRESHOLD_M);
        assert_eq!(score.good_samples, 1);
        assert_eq!(score.total_samples, 2);
        assert_eq!(score.missing_solution_samples, 1);
        assert_eq!(score.score_percent, 50.0);
    }

    #[test]
    fn score_holds_past_solution_without_interpolation() {
        let truth = vec![truth(0.0, 0.0), truth(1.0, 1.0), truth(2.0, 2.0)];
        let solution = vec![solution(0.0, 0.0), solution(2.0, 2.0)];
        let score = score_route("test", &truth, &solution, PPC_DEFAULT_THRESHOLD_M);
        assert_eq!(score.good_samples, 1);
        assert_eq!(score.missing_solution_samples, 0);
        assert_eq!(score.score_percent, 50.0);
    }

    #[test]
    fn score_is_distance_weighted_and_threshold_is_inclusive() {
        let truth = vec![truth(0.0, 0.0), truth(1.0, 1.0), truth(2.0, 4.0)];
        let solution = vec![solution(1.0, 1.5), solution(2.0, 4.6)];
        let score = score_route("test", &truth, &solution, PPC_DEFAULT_THRESHOLD_M);
        assert_eq!(score.good_samples, 1, "an exact 0.5 m error must pass");
        assert_eq!(score.total_samples, 2);
        assert_eq!(score.good_distance_m, 1.0);
        assert_eq!(score.total_distance_m, 4.0);
        assert_eq!(score.score_percent, 25.0);
    }

    #[test]
    fn reference_reader_rejects_duplicate_or_regressing_timestamps() {
        for (name, rows) in [
            ("duplicate.csv", "1,2325,35,136,40\n1,2325,35,136,40\n"),
            ("regressing.csv", "2,2325,35,136,40\n1,2325,35,136,40\n"),
        ] {
            let path = temp_file(
                name,
                &format!(
                    "GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n{rows}"
                ),
            );
            let error = read_reference_csv(&path).expect_err("timestamps must be ordered");
            assert!(matches!(error, PpcError::NonMonotonicTimestamp { .. }));
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn reference_reader_is_strict_for_truth_fields_and_csv_shape() {
        let invalid = temp_file(
            "invalid-truth.csv",
            "GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n1,not-a-week,35,136,40\n",
        );
        assert!(matches!(
            read_reference_csv(&invalid).expect_err("invalid week must fail"),
            PpcError::InvalidField { .. }
        ));
        let _ = fs::remove_file(invalid);

        let malformed = temp_file(
            "malformed-truth.csv",
            "GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n1,2325,35,136\n",
        );
        assert!(matches!(
            read_reference_csv(&malformed).expect_err("short CSV row must fail"),
            PpcError::Csv { .. }
        ));
        let _ = fs::remove_file(malformed);
    }

    #[test]
    fn solution_csv_filters_invalid_data_rows() {
        let path = temp_file(
            "filtered-solution.csv",
            "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n1,1,2,3\n2,NaN,2,3\nbad,1,2,3\n3,4,5,6\n",
        );
        let samples = read_solution(&path).expect("read valid solution rows");
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].tow_s, 1.0);
        assert_eq!(samples[1].tow_s, 3.0);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn solution_csv_requires_complete_coordinate_triplets() {
        let path = temp_file(
            "partial-ecef.csv",
            "GPS TOW (s),ECEF X (m),ECEF Y (m)\n1,1,2\n",
        );
        let error = read_solution(&path).expect_err("partial ECEF must not panic or parse");
        assert!(matches!(error, PpcError::MissingColumn { .. }));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn solution_csv_rejects_non_monotonic_usable_rows() {
        let path = temp_file(
            "regressing-solution.csv",
            "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n2,1,2,3\n1,4,5,6\n",
        );
        let error = read_solution(&path).expect_err("solution timestamps must be ordered");
        assert!(matches!(error, PpcError::NonMonotonicTimestamp { .. }));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rtklib_calendar_utc_is_normalized_to_gpst() {
        let gpst = temp_file(
            "gpst.pos",
            "%  GPST latitude(deg) longitude(deg) height(m)\ninvalid row\n2024/07/07 00:00:18.000 35.0 136.0 40.0\n",
        );
        let utc = temp_file(
            "utc.pos",
            "%  UTC latitude(deg) longitude(deg) height(m)\n2024/07/07 00:00:00.000 35.0 136.0 40.0\n",
        );
        let gpst_samples = read_rtklib_pos(&gpst).expect("read GPST calendar row");
        let utc_samples = read_rtklib_pos(&utc).expect("read UTC calendar row");
        assert_eq!(gpst_samples[0].tow_s, 18.0);
        assert_eq!(utc_samples[0].tow_s, gpst_samples[0].tow_s);
        assert_eq!(utc_samples[0].ecef_m, gpst_samples[0].ecef_m);
        let _ = fs::remove_file(gpst);
        let _ = fs::remove_file(utc);
    }

    #[test]
    fn rtklib_ecef_header_selects_ecef_coordinates() {
        let path = temp_file(
            "ecef.pos",
            "%  GPST              x-ecef(m)      y-ecef(m)      z-ecef(m)   Q\n2325 550380.000 1.0 2.0 3.0 1\n",
        );
        let samples = read_rtklib_pos(&path).expect("read RTKLIB ECEF row");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].tow_s, 550_380.0);
        assert_eq!(samples[0].ecef_m, [1.0, 2.0, 3.0]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rtklib_rejects_week_rollover_in_single_route() {
        let path = temp_file(
            "rollover.pos",
            "%  GPST latitude(deg) longitude(deg) height(m)\n2325 604799.000 35.0 136.0 40.0\n2326 0.000 35.0 136.0 40.0\n",
        );
        let error = read_rtklib_pos(&path).expect_err("PPC route must stay in one GPS week");
        assert!(matches!(error, PpcError::MultipleGpsWeeks { .. }));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn score_routes_rejects_declared_week_mismatch() {
        let truth_path = temp_file(
            "week-reference.csv",
            "GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n1,2325,35,136,40\n2,2325,35,136.00001,40\n",
        );
        let solution_path = temp_file(
            "week-solution.pos",
            "%  GPST latitude(deg) longitude(deg) height(m)\n2326 1.000 35.0 136.0 40.0\n",
        );
        let route = PpcRouteInput {
            name: "week-mismatch".to_string(),
            truth_path: truth_path.clone(),
            solution_path: solution_path.clone(),
        };
        let metadata = PpcRunMetadata {
            scorer_version: PPC_SCORER_VERSION.to_string(),
            sidereon_version: "test".to_string(),
            git_commit: None,
            dataset_revision: None,
            dataset_sha256: None,
            threshold_m: PPC_DEFAULT_THRESHOLD_M,
        };
        let error = score_routes(&[route], PPC_DEFAULT_THRESHOLD_M, metadata)
            .expect_err("declared GPS weeks must agree");
        assert!(matches!(error, PpcError::GpsWeekMismatch { .. }));
        let _ = fs::remove_file(truth_path);
        let _ = fs::remove_file(solution_path);
    }

    #[test]
    fn solution_csv_accepts_iso_utc_timestamp() {
        let path = temp_file(
            "utc-solution.csv",
            "Time (UTC),ECEF X (m),ECEF Y (m),ECEF Z (m)\n2024-07-07T00:00:00Z,1,2,3\n",
        );
        let samples = read_solution(&path).expect("read UTC solution CSV");
        assert_eq!(samples[0].tow_s, 18.0);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn solution_csv_filters_invalid_derived_coordinates() {
        let path = temp_file(
            "invalid-geodetic-solution.csv",
            "GPS TOW (s),Latitude (deg),Longitude (deg),Ellipsoid Height (m)\n1,100,136,40\n2,35,136,40\n",
        );
        let samples = read_solution(&path).expect("retain valid derived coordinate row");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].tow_s, 2.0);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn imu_reader_rejects_non_monotonic_timestamps() {
        let path = temp_file(
            "regressing-imu.csv",
            "GPS TOW (s),GPS Week,Acc X (m/s^2),Acc Y (m/s^2),Acc Z (m/s^2),Ang Rate X (deg/s),Ang Rate Y (deg/s),Ang Rate Z (deg/s)\n2,2325,1,2,3,4,5,6\n1,2325,1,2,3,4,5,6\n",
        );
        let error = read_imu_csv(&path).expect_err("IMU timestamps must be ordered");
        assert!(matches!(error, PpcError::NonMonotonicTimestamp { .. }));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn llh_conversion_matches_equatorial_radius() {
        let ecef = llh_to_ecef(0.0, 0.0, 0.0);
        assert!((ecef[0] - 6_378_137.0).abs() < 1e-6);
        assert_eq!(ecef[1], 0.0);
    }

    #[test]
    fn imu_conversion_uses_j2000_and_radians() {
        let sample = PpcImuSample {
            gps_week: Some(2325),
            tow_s: 550_380.0,
            acceleration_mps2: [1.0, 2.0, 3.0],
            angular_rate_rad_s: [std::f64::consts::PI / 180.0, 0.0, 0.0],
        };
        let core = sample.to_core_sample();
        assert_eq!(
            core.t_j2000_s,
            2325.0 * SECONDS_PER_WEEK + 550_380.0 - GPS_EPOCH_TO_J2000_S
        );
        assert!(matches!(
            core.kind,
            sidereon_core::inertial::ImuSampleKind::Rate { .. }
        ));
    }
}
