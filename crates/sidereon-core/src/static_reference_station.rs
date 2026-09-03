//! Static reference-station solve from paired rover/reference RINEX arcs.
//!
//! This module composes existing language-independent pieces: RINEX SPP epoch
//! assembly, code-DGNSS pseudorange corrections, stacked static SPP, and the RTK
//! RINEX arc builders/static carrier solver. It returns a station coordinate
//! with a covariance from the final normal equations, not from epoch scatter.

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::math::vec3;
use crate::dgnss::{apply_corrections, pseudorange_corrections, solve_position, CodeObservation};
use crate::dop::rotate_covariance_ecef_to_enu_m2;
use crate::frame::{itrf_to_geodetic, ItrfPositionM, Wgs84Geodetic};
use crate::observables::ObservableEphemerisSource;
use crate::positioning::{
    spp_inputs_from_rinex_obs, EphemerisSource, RinexSppAssemblySource, RinexSppEpochInputs,
    RinexSppOptions, StaticEpoch, StaticSolution, StaticSolveOptions,
};
use crate::rinex::observations::{ObsEpochTime, RinexObs};
use crate::rtk_filter::{
    build_rinex_rtk_arc, IntegerStatus, RtkRinexArcOptions, RtkStaticArcConfig,
    RtkStaticArcSolution,
};
use crate::spp::{Corrections, Observation};
use crate::validate;

/// High-level solve mode selected for the reported station coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticReferenceStationMode {
    /// Code-DGNSS corrected pseudoranges stacked through the static SPP solver.
    CodeDgnss,
    /// Carrier RTK float baseline added to the surveyed reference coordinate.
    CarrierFloat,
    /// Carrier RTK integer-fixed baseline added to the surveyed reference coordinate.
    CarrierFixed,
}

/// Fix label for the reported station coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticReferenceFixStatus {
    /// Code-DGNSS solution; no carrier integer fix was attempted for the selected result.
    CodeDgnss,
    /// Carrier RTK float baseline.
    CarrierFloat,
    /// Carrier RTK integer-fixed baseline.
    CarrierFixed,
}

impl From<StaticReferenceFixStatus> for crate::fusion::GnssFixStatus {
    fn from(value: StaticReferenceFixStatus) -> Self {
        match value {
            StaticReferenceFixStatus::CodeDgnss => Self::Single,
            StaticReferenceFixStatus::CarrierFloat => Self::Float,
            StaticReferenceFixStatus::CarrierFixed => Self::Fixed,
        }
    }
}

/// Attempt status for one mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticReferenceModeStatus {
    /// The mode was enabled and solved.
    Solved,
    /// The mode was enabled but failed.
    Failed,
}

/// Position covariance for the returned station coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticReferenceStationCovariance {
    /// ECEF position covariance in square metres.
    pub position_ecef_m2: [[f64; 3]; 3],
    /// Local ENU position covariance in square metres.
    pub position_enu_m2: [[f64; 3]; 3],
}

/// Per-epoch diagnostic rollup for a solved mode.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticReferenceEpochDiagnostic {
    /// Mode that produced this diagnostic row.
    pub mode: StaticReferenceStationMode,
    /// Epoch index in the assembled solve input for this mode.
    pub epoch_index: usize,
    /// Satellites used by the epoch, sorted by the underlying solver.
    pub used_satellites: Vec<String>,
    /// Number of rejected satellites for code-DGNSS epochs.
    pub rejected_satellite_count: usize,
    /// Code residual RMS for the epoch, metres, when code rows are available.
    pub code_residual_rms_m: Option<f64>,
    /// Carrier residual RMS for the epoch, metres, when carrier rows are available.
    pub phase_residual_rms_m: Option<f64>,
    /// Total unweighted residual RMS for the epoch, metres.
    pub residual_rms_m: Option<f64>,
}

/// Report for a mode attempted by the wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticReferenceModeReport {
    /// Mode attempted.
    pub mode: StaticReferenceStationMode,
    /// Whether the mode solved or failed.
    pub status: StaticReferenceModeStatus,
    /// Solved epoch count or zero on failure.
    pub used_epochs: usize,
    /// Number of raw RINEX epochs skipped by the mode builder.
    pub skipped_epochs: usize,
    /// Measurement rows used by the final solve, when known.
    pub used_measurements: usize,
    /// Failure detail, when the mode failed.
    pub error: Option<StaticReferenceModeError>,
}

/// Typed failure detail for one attempted static reference-station mode.
#[derive(Debug, Clone, PartialEq)]
pub enum StaticReferenceModeError {
    /// RINEX/SPP input assembly failed before mode-specific solving.
    RinexAssembly {
        /// Observation side being assembled.
        side: &'static str,
        /// Source error text.
        reason: String,
    },
    /// Code-DGNSS reference and rover assemblies had no epoch in common.
    NoMatchedCodeEpochs,
    /// Code-DGNSS correction or single-epoch solve failed.
    CodeDgnss {
        /// Source error text.
        reason: String,
    },
    /// Multi-epoch static code solve failed.
    StaticSolve {
        /// Source error text.
        reason: String,
    },
    /// Carrier RTK arc construction failed.
    CarrierArc {
        /// Source error text.
        reason: String,
    },
    /// Carrier static RTK solve failed.
    CarrierSolve {
        /// Source error text.
        reason: String,
    },
    /// Frame, coordinate, or covariance conversion failed.
    Frame {
        /// Conversion field.
        field: &'static str,
        /// Source error text.
        reason: String,
    },
    /// Corrected code observations could not be applied or converted.
    CorrectedObservation {
        /// Source error text.
        reason: String,
    },
    /// A corrected satellite identifier could not be parsed back to a typed ID.
    InvalidCorrectedSatelliteId {
        /// Invalid satellite identifier text.
        satellite_id: String,
    },
}

impl core::fmt::Display for StaticReferenceModeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RinexAssembly { side, reason } => {
                write!(f, "{side} RINEX assembly failed: {reason}")
            }
            Self::NoMatchedCodeEpochs => f.write_str("no matched epochs"),
            Self::CodeDgnss { reason } => {
                write!(f, "code-DGNSS failed: {reason}")
            }
            Self::StaticSolve { reason } => {
                write!(f, "static code solve failed: {reason}")
            }
            Self::CarrierArc { reason } => {
                write!(f, "carrier RTK arc failed: {reason}")
            }
            Self::CarrierSolve { reason } => {
                write!(f, "carrier RTK solve failed: {reason}")
            }
            Self::Frame { field, reason } => {
                write!(f, "{field} conversion failed: {reason}")
            }
            Self::CorrectedObservation { reason } => {
                write!(f, "corrected observation failed: {reason}")
            }
            Self::InvalidCorrectedSatelliteId { satellite_id } => {
                write!(f, "invalid corrected satellite id {satellite_id}")
            }
        }
    }
}

impl std::error::Error for StaticReferenceModeError {}

/// Code-DGNSS static solve detail.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticReferenceCodeSolution {
    /// Solved rover/reference-station coordinate.
    pub position: ItrfPositionM,
    /// Geodetic coordinate when requested.
    pub geodetic: Option<Wgs84Geodetic>,
    /// Position covariance from the code-DGNSS normal equations.
    pub covariance: StaticReferenceStationCovariance,
    /// Multi-epoch static solution when the code path stacked more than one epoch.
    pub static_solution: Option<StaticSolution>,
    /// Baseline vector, rover minus reference, metres.
    pub baseline_vector_m: [f64; 3],
    /// Baseline length, metres.
    pub baseline_m: f64,
    /// Per-epoch diagnostic rollups.
    pub diagnostics: Vec<StaticReferenceEpochDiagnostic>,
}

/// Carrier RTK static solve detail.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticReferenceCarrierSolution {
    /// Solved rover/reference-station coordinate.
    pub position: ItrfPositionM,
    /// Geodetic coordinate when requested.
    pub geodetic: Option<Wgs84Geodetic>,
    /// Position covariance from the selected carrier baseline normal equations.
    pub covariance: StaticReferenceStationCovariance,
    /// Selected baseline vector, rover minus reference, metres.
    pub baseline_vector_m: [f64; 3],
    /// Baseline length, metres.
    pub baseline_m: f64,
    /// Integer ambiguity status from the fixed RTK solve.
    pub integer_status: IntegerStatus,
    /// Integer ratio from the fixed RTK solve, when a search ran.
    pub integer_ratio: Option<f64>,
    /// Full carrier static arc solution.
    pub rtk_solution: RtkStaticArcSolution,
    /// Per-epoch diagnostic rollups from the selected float/fixed residuals.
    pub diagnostics: Vec<StaticReferenceEpochDiagnostic>,
}

/// Final station solution returned by the RINEX wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticReferenceStationSolution {
    /// Selected mode used for the reported coordinate.
    pub mode: StaticReferenceStationMode,
    /// Reported fix status for the selected coordinate.
    pub fix_status: StaticReferenceFixStatus,
    /// Solved rover/reference-station coordinate.
    pub position: ItrfPositionM,
    /// Geodetic coordinate when requested.
    pub geodetic: Option<Wgs84Geodetic>,
    /// Position covariance for the reported coordinate.
    pub covariance: StaticReferenceStationCovariance,
    /// Baseline vector, rover minus reference, metres.
    pub baseline_vector_m: [f64; 3],
    /// Baseline length, metres.
    pub baseline_m: f64,
    /// Code-DGNSS solution detail when the mode was enabled and solved.
    pub code_solution: Option<StaticReferenceCodeSolution>,
    /// Carrier RTK solution detail when the mode was enabled and solved.
    pub carrier_solution: Option<StaticReferenceCarrierSolution>,
    /// Per-mode attempt reports.
    pub mode_reports: Vec<StaticReferenceModeReport>,
    /// Diagnostics for the selected mode.
    pub diagnostics: Vec<StaticReferenceEpochDiagnostic>,
}

/// Carrier RTK options for the RINEX station wrapper.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StaticReferenceCarrierRinexOptions {
    /// RINEX-to-RTK arc extraction options.
    pub arc_options: RtkRinexArcOptions,
    /// Static RTK solve configuration. The wrapper overwrites its base position
    /// and ambiguity scale maps from the function inputs and RINEX arc.
    pub static_config: RtkStaticArcConfig,
}

impl StaticReferenceCarrierRinexOptions {
    /// Build carrier RINEX options from the arc extraction and static solve
    /// configurations.
    #[must_use]
    pub const fn new(arc_options: RtkRinexArcOptions, static_config: RtkStaticArcConfig) -> Self {
        Self {
            arc_options,
            static_config,
        }
    }
}

/// RINEX wrapper options. `None` disables that mode.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StaticReferenceStationRinexOptions {
    /// Code-DGNSS RINEX/SPP assembly options.
    pub code_options: Option<RinexSppOptions>,
    /// Carrier RTK RINEX/static options.
    pub carrier_options: Option<StaticReferenceCarrierRinexOptions>,
    /// Whether to include geodetic coordinates in the selected and nested results.
    pub with_geodetic: bool,
}

impl StaticReferenceStationRinexOptions {
    /// Build RINEX wrapper options from the optionally enabled modes.
    ///
    /// `None` disables the corresponding code or carrier mode.
    #[must_use]
    pub const fn new(
        code_options: Option<RinexSppOptions>,
        carrier_options: Option<StaticReferenceCarrierRinexOptions>,
        with_geodetic: bool,
    ) -> Self {
        Self {
            code_options,
            carrier_options,
            with_geodetic,
        }
    }

    /// Build options with both code-DGNSS and carrier RTK enabled.
    #[must_use]
    pub const fn code_and_carrier(
        code_options: RinexSppOptions,
        carrier_options: StaticReferenceCarrierRinexOptions,
        with_geodetic: bool,
    ) -> Self {
        Self {
            code_options: Some(code_options),
            carrier_options: Some(carrier_options),
            with_geodetic,
        }
    }

    /// Build options with only code-DGNSS enabled.
    #[must_use]
    pub const fn code_only(code_options: RinexSppOptions, with_geodetic: bool) -> Self {
        Self {
            code_options: Some(code_options),
            carrier_options: None,
            with_geodetic,
        }
    }

    /// Build options with only carrier RTK enabled.
    #[must_use]
    pub const fn carrier_only(
        carrier_options: StaticReferenceCarrierRinexOptions,
        with_geodetic: bool,
    ) -> Self {
        Self {
            code_options: None,
            carrier_options: Some(carrier_options),
            with_geodetic,
        }
    }
}

/// Error returned by [`solve_static_reference_station_rinex`].
#[derive(Debug, Clone, PartialEq)]
pub enum StaticReferenceStationError {
    /// Public input validation failed.
    InvalidInput {
        /// Invalid input field.
        field: &'static str,
        /// Validation reason.
        reason: &'static str,
    },
    /// Neither code-DGNSS nor carrier RTK mode was enabled.
    NoEnabledModes,
    /// Every enabled mode failed. Individual errors are carried in the reports.
    AllModesFailed {
        /// Reports for the failed modes.
        mode_reports: Vec<StaticReferenceModeReport>,
    },
}

impl core::fmt::Display for StaticReferenceStationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(
                    f,
                    "invalid static reference-station input {field}: {reason}"
                )
            }
            Self::NoEnabledModes => {
                f.write_str("static reference-station solve has no enabled modes")
            }
            Self::AllModesFailed { mode_reports } => {
                f.write_str("all static reference-station modes failed")?;
                for report in mode_reports {
                    if let Some(error) = &report.error {
                        write!(f, "; {}: {error}", mode_label(report.mode))?;
                    } else {
                        write!(f, "; {}: failed", mode_label(report.mode))?;
                    }
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for StaticReferenceStationError {}

/// Solve one rover/reference-station coordinate from paired reference and rover
/// RINEX observation files plus a known reference coordinate.
///
/// `reference_position_m` is the ECEF coordinate of the point represented by the
/// reference observations. The returned coordinate is the corresponding point
/// represented by the rover observations; caller-owned antenna marker/ARP
/// conversions should be applied consistently before and after this call.
pub fn solve_static_reference_station_rinex<S>(
    source: &S,
    reference_obs: &RinexObs,
    rover_obs: &RinexObs,
    reference_position_m: [f64; 3],
    options: &StaticReferenceStationRinexOptions,
) -> Result<StaticReferenceStationSolution, StaticReferenceStationError>
where
    S: EphemerisSource + ObservableEphemerisSource + RinexSppAssemblySource,
{
    validate::finite_vec3(reference_position_m, "reference_position_m")
        .map_err(static_reference_input_error)?;

    if options.code_options.is_none() && options.carrier_options.is_none() {
        return Err(StaticReferenceStationError::NoEnabledModes);
    }

    let mut reports = Vec::new();
    let code = match &options.code_options {
        Some(code_options) => match solve_code_dgnss_static(
            source,
            reference_obs,
            rover_obs,
            reference_position_m,
            code_options,
            options.with_geodetic,
        ) {
            Ok(solution) => {
                reports.push(StaticReferenceModeReport {
                    mode: StaticReferenceStationMode::CodeDgnss,
                    status: StaticReferenceModeStatus::Solved,
                    used_epochs: solution.diagnostics.len(),
                    skipped_epochs: 0,
                    used_measurements: solution
                        .diagnostics
                        .iter()
                        .map(|row| row.used_satellites.len())
                        .sum(),
                    error: None,
                });
                Some(solution)
            }
            Err(error) => {
                reports.push(failed_report(StaticReferenceStationMode::CodeDgnss, error));
                None
            }
        },
        None => None,
    };

    let carrier = match &options.carrier_options {
        Some(carrier_options) => match solve_carrier_static(
            source,
            reference_obs,
            rover_obs,
            reference_position_m,
            carrier_options,
            options.with_geodetic,
        ) {
            Ok((solution, skipped_epochs)) => {
                reports.push(StaticReferenceModeReport {
                    mode: solution_mode_from_carrier(&solution),
                    status: StaticReferenceModeStatus::Solved,
                    used_epochs: solution.diagnostics.len(),
                    skipped_epochs,
                    used_measurements: carrier_used_measurements(&solution),
                    error: None,
                });
                Some(solution)
            }
            Err(error) => {
                reports.push(failed_report(
                    StaticReferenceStationMode::CarrierFixed,
                    error,
                ));
                None
            }
        },
        None => None,
    };

    let selected = select_solution(reference_position_m, code, carrier, reports)?;
    Ok(selected)
}

fn solve_code_dgnss_static<S>(
    source: &S,
    reference_obs: &RinexObs,
    rover_obs: &RinexObs,
    reference_position_m: [f64; 3],
    code_options: &RinexSppOptions,
    with_geodetic: bool,
) -> Result<StaticReferenceCodeSolution, StaticReferenceModeError>
where
    S: EphemerisSource + ObservableEphemerisSource + RinexSppAssemblySource,
{
    let reference_epochs =
        spp_inputs_from_rinex_obs(reference_obs, source, code_options).map_err(|error| {
            StaticReferenceModeError::RinexAssembly {
                side: "reference",
                reason: error.to_string(),
            }
        })?;
    let rover_epochs =
        spp_inputs_from_rinex_obs(rover_obs, source, code_options).map_err(|error| {
            StaticReferenceModeError::RinexAssembly {
                side: "rover",
                reason: error.to_string(),
            }
        })?;
    let matched = matched_code_epochs(&reference_epochs, &rover_epochs);
    if matched.is_empty() {
        return Err(StaticReferenceModeError::NoMatchedCodeEpochs);
    }

    if matched.len() == 1 {
        return solve_single_code_epoch(source, reference_position_m, matched[0], with_geodetic);
    }

    let static_options = StaticSolveOptions::from_solve_inputs(&matched[0].1.inputs, with_geodetic);
    let mut static_epochs = Vec::with_capacity(matched.len());
    for (reference_epoch, rover_epoch) in matched {
        let corrected = corrected_rover_observations(
            source,
            reference_position_m,
            reference_epoch,
            rover_epoch,
        )?;
        let mut inputs = rover_epoch.inputs.clone();
        inputs.observations = corrected;
        inputs.corrections = Corrections::NONE;
        let mut static_epoch = StaticEpoch::from_solve_inputs(inputs);
        static_epoch.weights = Some(vec![0.5; static_epoch.measurements.len()]);
        static_epochs.push(static_epoch);
    }

    let static_solution = crate::static_positioning::solve_static_without_influence(
        source,
        &static_epochs,
        static_options,
    )
    .map_err(|error| StaticReferenceModeError::StaticSolve {
        reason: error.to_string(),
    })?;
    let position = static_solution.position;
    let covariance = StaticReferenceStationCovariance {
        position_ecef_m2: static_solution.covariance.position_ecef_m2,
        position_enu_m2: static_solution.covariance.position_enu_m2,
    };
    let baseline_vector_m = vec3::sub3(position.as_array(), reference_position_m);
    let baseline_m = vec3::norm3(baseline_vector_m);
    let diagnostics = code_static_diagnostics(&static_solution);

    Ok(StaticReferenceCodeSolution {
        position,
        geodetic: static_solution.geodetic,
        covariance,
        static_solution: Some(static_solution),
        baseline_vector_m,
        baseline_m,
        diagnostics,
    })
}

fn solve_single_code_epoch<S>(
    source: &S,
    reference_position_m: [f64; 3],
    matched: (&RinexSppEpochInputs, &RinexSppEpochInputs),
    with_geodetic: bool,
) -> Result<StaticReferenceCodeSolution, StaticReferenceModeError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    let (reference_epoch, rover_epoch) = matched;
    let reference_codes = code_observations(&reference_epoch.inputs.observations);
    let rover_codes = code_observations(&rover_epoch.inputs.observations);
    let solution = solve_position(
        source,
        reference_position_m,
        &reference_codes,
        &rover_codes,
        rover_epoch.inputs.clone(),
        with_geodetic,
    )
    .map_err(|error| StaticReferenceModeError::CodeDgnss {
        reason: error.to_string(),
    })?;
    let covariance = StaticReferenceStationCovariance {
        position_ecef_m2: solution.solution.position_covariance.ecef_m2,
        position_enu_m2: solution.solution.position_covariance.enu_m2,
    };
    let position = solution.solution.position;
    let diagnostics = vec![StaticReferenceEpochDiagnostic {
        mode: StaticReferenceStationMode::CodeDgnss,
        epoch_index: 0,
        used_satellites: solution
            .solution
            .used_sats
            .iter()
            .map(ToString::to_string)
            .collect(),
        rejected_satellite_count: solution.solution.rejected_sats.len(),
        code_residual_rms_m: Some(solution.solution.residual_rms_m()),
        phase_residual_rms_m: None,
        residual_rms_m: Some(solution.solution.residual_rms_m()),
    }];

    Ok(StaticReferenceCodeSolution {
        position,
        geodetic: solution.solution.geodetic,
        covariance,
        static_solution: None,
        baseline_vector_m: solution.baseline_vector_m,
        baseline_m: solution.baseline_m,
        diagnostics,
    })
}

fn solve_carrier_static<S>(
    source: &S,
    reference_obs: &RinexObs,
    rover_obs: &RinexObs,
    reference_position_m: [f64; 3],
    carrier_options: &StaticReferenceCarrierRinexOptions,
    with_geodetic: bool,
) -> Result<(StaticReferenceCarrierSolution, usize), StaticReferenceModeError>
where
    S: ObservableEphemerisSource,
{
    let arc = build_rinex_rtk_arc(
        source,
        reference_obs,
        rover_obs,
        &carrier_options.arc_options,
    )
    .map_err(|error| StaticReferenceModeError::CarrierArc {
        reason: error.to_string(),
    })?;
    let mut config = carrier_options.static_config.clone();
    config.arc.base_m = reference_position_m;
    config.arc.wavelengths_m = arc.wavelengths_m.clone();
    config.arc.offsets_m = arc.offsets_m.clone();
    let rtk_solution =
        crate::rtk_filter::solve_static_rtk_arc(&arc.epochs, &config).map_err(|error| {
            StaticReferenceModeError::CarrierSolve {
                reason: error.to_string(),
            }
        })?;
    let fixed = &rtk_solution.fixed_solution.fixed_solution;
    let (baseline_vector_m, covariance_ecef_m2, diagnostics) =
        if fixed.search.integer_status == IntegerStatus::Fixed {
            (
                fixed.baseline_m,
                fixed.baseline_covariance_m2,
                carrier_diagnostics(StaticReferenceStationMode::CarrierFixed, &fixed.residuals),
            )
        } else {
            (
                rtk_solution.float_solution.baseline_m,
                rtk_solution.float_solution.baseline_covariance_m2,
                carrier_diagnostics(
                    StaticReferenceStationMode::CarrierFloat,
                    &rtk_solution.float_solution.residuals,
                ),
            )
        };
    let position_m = vec3::add3(reference_position_m, baseline_vector_m);
    let position =
        ItrfPositionM::new(position_m[0], position_m[1], position_m[2]).map_err(|error| {
            StaticReferenceModeError::Frame {
                field: "position",
                reason: error.to_string(),
            }
        })?;
    let covariance = covariance_from_ecef(position, covariance_ecef_m2)?;
    let geodetic = if with_geodetic {
        Some(
            itrf_to_geodetic(position).map_err(|error| StaticReferenceModeError::Frame {
                field: "geodetic",
                reason: error.to_string(),
            })?,
        )
    } else {
        None
    };

    Ok((
        StaticReferenceCarrierSolution {
            position,
            geodetic,
            covariance,
            baseline_vector_m,
            baseline_m: vec3::norm3(baseline_vector_m),
            integer_status: fixed.search.integer_status,
            integer_ratio: fixed.search.integer_ratio,
            rtk_solution,
            diagnostics,
        },
        arc.skipped_epoch_count,
    ))
}

fn select_solution(
    reference_position_m: [f64; 3],
    code: Option<StaticReferenceCodeSolution>,
    carrier: Option<StaticReferenceCarrierSolution>,
    reports: Vec<StaticReferenceModeReport>,
) -> Result<StaticReferenceStationSolution, StaticReferenceStationError> {
    if code.is_none() && carrier.is_none() {
        return Err(StaticReferenceStationError::AllModesFailed {
            mode_reports: reports,
        });
    }

    let (mode, position, geodetic, covariance, baseline_vector_m, mut baseline_m, diagnostics) =
        match (carrier.as_ref(), code.as_ref()) {
            (Some(carrier), _)
                if solution_mode_from_carrier(carrier)
                    == StaticReferenceStationMode::CarrierFixed =>
            {
                (
                    StaticReferenceStationMode::CarrierFixed,
                    carrier.position,
                    carrier.geodetic,
                    carrier.covariance,
                    carrier.baseline_vector_m,
                    carrier.baseline_m,
                    carrier.diagnostics.clone(),
                )
            }
            (_, Some(code)) => (
                StaticReferenceStationMode::CodeDgnss,
                code.position,
                code.geodetic,
                code.covariance,
                code.baseline_vector_m,
                code.baseline_m,
                code.diagnostics.clone(),
            ),
            (Some(carrier), None) => (
                StaticReferenceStationMode::CarrierFloat,
                carrier.position,
                carrier.geodetic,
                carrier.covariance,
                carrier.baseline_vector_m,
                carrier.baseline_m,
                carrier.diagnostics.clone(),
            ),
            (None, None) => unreachable!("handled above"),
        };
    let baseline_vector_m = if baseline_vector_m.iter().all(|value| value.is_finite()) {
        baseline_vector_m
    } else {
        let fallback = vec3::sub3(position.as_array(), reference_position_m);
        baseline_m = vec3::norm3(fallback);
        fallback
    };

    Ok(StaticReferenceStationSolution {
        mode,
        fix_status: fix_status_from_mode(mode),
        position,
        geodetic,
        covariance,
        baseline_vector_m,
        baseline_m,
        code_solution: code,
        carrier_solution: carrier,
        mode_reports: reports,
        diagnostics,
    })
}

fn matched_code_epochs<'a>(
    reference_epochs: &'a [RinexSppEpochInputs],
    rover_epochs: &'a [RinexSppEpochInputs],
) -> Vec<(&'a RinexSppEpochInputs, &'a RinexSppEpochInputs)> {
    let rover_by_epoch = rover_epochs
        .iter()
        .map(|epoch| (epoch_key(epoch.epoch), epoch))
        .collect::<BTreeMap<_, _>>();
    reference_epochs
        .iter()
        .filter_map(|reference_epoch| {
            rover_by_epoch
                .get(&epoch_key(reference_epoch.epoch))
                .map(|&rover_epoch| (reference_epoch, rover_epoch))
        })
        .collect()
}

fn corrected_rover_observations<S>(
    source: &S,
    reference_position_m: [f64; 3],
    reference_epoch: &RinexSppEpochInputs,
    rover_epoch: &RinexSppEpochInputs,
) -> Result<Vec<Observation>, StaticReferenceModeError>
where
    S: ObservableEphemerisSource,
{
    let corrections = pseudorange_corrections(
        source,
        reference_position_m,
        &code_observations(&reference_epoch.inputs.observations),
        reference_epoch.inputs.t_rx_j2000_s,
    )
    .map_err(|error| StaticReferenceModeError::CodeDgnss {
        reason: error.to_string(),
    })?;
    let corrected = apply_corrections(
        &code_observations(&rover_epoch.inputs.observations),
        &corrections,
    )
    .map_err(|error| StaticReferenceModeError::CorrectedObservation {
        reason: error.to_string(),
    })?;
    corrected
        .corrected
        .into_iter()
        .map(|obs| {
            obs.satellite_id
                .parse()
                .map(|satellite_id| Observation {
                    satellite_id,
                    pseudorange_m: obs.pseudorange_m,
                })
                .map_err(|_| StaticReferenceModeError::InvalidCorrectedSatelliteId {
                    satellite_id: obs.satellite_id,
                })
        })
        .collect()
}

fn code_observations(observations: &[Observation]) -> Vec<CodeObservation> {
    observations
        .iter()
        .map(|obs| CodeObservation::new(obs.satellite_id.to_string(), obs.pseudorange_m))
        .collect()
}

fn code_static_diagnostics(solution: &StaticSolution) -> Vec<StaticReferenceEpochDiagnostic> {
    let mut residuals_by_epoch = BTreeMap::<usize, Vec<f64>>::new();
    for residual in &solution.residuals_m {
        residuals_by_epoch
            .entry(residual.epoch_index)
            .or_default()
            .push(residual.residual_m);
    }

    solution
        .used_sats
        .iter()
        .enumerate()
        .map(|(epoch_index, sats)| {
            let residuals = residuals_by_epoch
                .get(&epoch_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let rms = residuals_rms(residuals.iter().copied());
            StaticReferenceEpochDiagnostic {
                mode: StaticReferenceStationMode::CodeDgnss,
                epoch_index,
                used_satellites: sats.iter().map(ToString::to_string).collect(),
                rejected_satellite_count: solution
                    .rejected_sats
                    .get(epoch_index)
                    .map_or(0, Vec::len),
                code_residual_rms_m: rms,
                phase_residual_rms_m: None,
                residual_rms_m: rms,
            }
        })
        .collect()
}

fn carrier_diagnostics(
    mode: StaticReferenceStationMode,
    residuals: &[crate::rtk_filter::FloatResidual],
) -> Vec<StaticReferenceEpochDiagnostic> {
    #[derive(Default)]
    struct Accum {
        sats: BTreeSet<String>,
        code: Vec<f64>,
        phase: Vec<f64>,
    }

    let mut by_epoch = BTreeMap::<usize, Accum>::new();
    for residual in residuals {
        let entry = by_epoch.entry(residual.epoch_index).or_default();
        entry.sats.insert(residual.satellite_id.clone());
        entry.code.push(residual.code_m);
        entry.phase.push(residual.phase_m);
    }

    by_epoch
        .into_iter()
        .map(|(epoch_index, accum)| {
            let total = accum
                .code
                .iter()
                .chain(accum.phase.iter())
                .copied()
                .collect::<Vec<_>>();
            StaticReferenceEpochDiagnostic {
                mode,
                epoch_index,
                used_satellites: accum.sats.into_iter().collect(),
                rejected_satellite_count: 0,
                code_residual_rms_m: residuals_rms(accum.code),
                phase_residual_rms_m: residuals_rms(accum.phase),
                residual_rms_m: residuals_rms(total),
            }
        })
        .collect()
}

fn covariance_from_ecef(
    position: ItrfPositionM,
    position_ecef_m2: [[f64; 3]; 3],
) -> Result<StaticReferenceStationCovariance, StaticReferenceModeError> {
    let geodetic = itrf_to_geodetic(position).map_err(|error| StaticReferenceModeError::Frame {
        field: "geodetic",
        reason: error.to_string(),
    })?;
    let position_enu_m2 =
        rotate_covariance_ecef_to_enu_m2(position_ecef_m2, geodetic).map_err(|error| {
            StaticReferenceModeError::Frame {
                field: "covariance",
                reason: error.to_string(),
            }
        })?;
    Ok(StaticReferenceStationCovariance {
        position_ecef_m2,
        position_enu_m2,
    })
}

fn solution_mode_from_carrier(
    solution: &StaticReferenceCarrierSolution,
) -> StaticReferenceStationMode {
    if solution.integer_status == IntegerStatus::Fixed {
        StaticReferenceStationMode::CarrierFixed
    } else {
        StaticReferenceStationMode::CarrierFloat
    }
}

fn carrier_used_measurements(solution: &StaticReferenceCarrierSolution) -> usize {
    if solution.integer_status == IntegerStatus::Fixed {
        solution
            .rtk_solution
            .fixed_solution
            .fixed_solution
            .n_observations
    } else {
        solution.rtk_solution.float_solution.n_observations
    }
}

fn fix_status_from_mode(mode: StaticReferenceStationMode) -> StaticReferenceFixStatus {
    match mode {
        StaticReferenceStationMode::CodeDgnss => StaticReferenceFixStatus::CodeDgnss,
        StaticReferenceStationMode::CarrierFloat => StaticReferenceFixStatus::CarrierFloat,
        StaticReferenceStationMode::CarrierFixed => StaticReferenceFixStatus::CarrierFixed,
    }
}

fn mode_label(mode: StaticReferenceStationMode) -> &'static str {
    match mode {
        StaticReferenceStationMode::CodeDgnss => "code-DGNSS",
        StaticReferenceStationMode::CarrierFloat => "carrier-float",
        StaticReferenceStationMode::CarrierFixed => "carrier-fixed",
    }
}

fn residuals_rms(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value * value;
        count += 1;
    }
    (count > 0).then(|| (sum / count as f64).sqrt())
}

fn failed_report(
    mode: StaticReferenceStationMode,
    error: StaticReferenceModeError,
) -> StaticReferenceModeReport {
    StaticReferenceModeReport {
        mode,
        status: StaticReferenceModeStatus::Failed,
        used_epochs: 0,
        skipped_epochs: 0,
        used_measurements: 0,
        error: Some(error),
    }
}

fn epoch_key(epoch: ObsEpochTime) -> (i32, u8, u8, u8, u8, u64) {
    (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second.to_bits(),
    )
}

fn static_reference_input_error(error: validate::FieldError) -> StaticReferenceStationError {
    StaticReferenceStationError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}
