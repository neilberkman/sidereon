//! Moving-baseline RTK: both receivers move each epoch (RTKLIB "moving-base").
//!
//! Standard relative RTK positions a moving rover against a station whose ECEF
//! coordinates are FIXED for the whole arc. The double-difference cancellation,
//! the iterated float baseline least squares, and the LAMBDA integer fix are all
//! unchanged when the base also moves: the double difference removes the
//! receiver clocks and the per-satellite biases regardless of where the base is,
//! so the estimator still solves for the baseline vector `rover - base`. The
//! single change is that the base ECEF position is supplied per epoch (typically
//! the base receiver's own navigation fix) instead of being held constant.
//!
//! This module threads a per-epoch base position through the existing static
//! [`solve_float_baseline`] and [`solve_fixed_baseline`] solvers, solving each
//! epoch independently, and reports per epoch the baseline vector, its length,
//! and whether the integer ambiguities were fixed. A baseline observed between
//! two moving platforms can change every epoch, so each epoch is solved on its
//! own; an optional warm start carries the previous epoch's baseline forward as
//! the next epoch's linearization point for continuity.

use crate::astro::math::vec3::norm3;

use super::{
    solve_fixed_baseline, solve_float_baseline, AmbiguitySet, Epoch, FixedBaselineSolution,
    FixedSolveError, FixedSolveOpts, FloatBaselineSolution, FloatPrior, FloatSolveError,
    FloatSolveOpts, IntegerStatus, MeasModel, ReceiverAntennaCorrections,
};

/// One moving-baseline epoch: the base receiver's own ECEF position this epoch,
/// the double-difference observations, and the ambiguity set to resolve.
///
/// The observation [`Epoch`] and [`AmbiguitySet`] are exactly the inputs the
/// static fixed-base solvers take; only [`base_position_m`](Self::base_position_m)
/// is new, and it is what changes from epoch to epoch in moving-base mode.
#[derive(Clone, Copy)]
pub struct MovingBaselineEpoch<'a> {
    /// Base receiver ECEF position (metres) at this epoch. In moving-base mode
    /// this is the base's own navigation fix, refreshed every epoch rather than
    /// a surveyed constant.
    pub base_position_m: [f64; 3],
    /// Double-difference observations (references + non-references) this epoch.
    pub epoch: &'a Epoch,
    /// The ambiguity set for this epoch: ordered ids, the id -> rover-satellite
    /// map, the cycle/metre scaling, and the constellations held float-only.
    pub ambiguities: AmbiguitySet<'a>,
}

/// Integer ambiguity verdict for a moving-baseline epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovingBaselineStatus {
    /// LAMBDA fixed the integer ambiguities; the reported baseline is the
    /// integer-fixed solution.
    Fixed,
    /// The integers were not fixed (e.g. the LAMBDA ratio test failed); the
    /// reported baseline is the float solution.
    Float,
}

/// One solved moving-baseline epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingBaselineEpochSolution {
    /// Base receiver ECEF position (metres) used for this epoch.
    pub base_position_m: [f64; 3],
    /// Baseline vector `rover - base` (metres) in the ECEF frame. This is the
    /// integer-fixed baseline when [`status`](Self::status) is
    /// [`MovingBaselineStatus::Fixed`], otherwise the float baseline.
    pub baseline_m: [f64; 3],
    /// Euclidean baseline length (metres).
    pub baseline_length_m: f64,
    /// Whether the integer ambiguities were fixed this epoch.
    pub status: MovingBaselineStatus,
    /// The float baseline solution the epoch reduced through.
    pub float: FloatBaselineSolution,
    /// The integer-fixed solution. The fixed solver always runs and returns a
    /// solution; inspect [`status`](Self::status) (or
    /// `fixed.search.integer_status`) for the fix verdict. When the integers are
    /// not fixed this carries the float-equivalent baseline.
    pub fixed: FixedBaselineSolution,
}

/// Controls for a moving-baseline solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingBaselineOpts {
    /// Measurement model (code/phase sigma, Sagnac, stochastic weighting).
    pub model: MeasModel,
    /// Float baseline solver controls.
    pub float: FloatSolveOpts,
    /// Integer-fixed baseline solver controls.
    pub fixed: FixedSolveOpts,
    /// Initial baseline guess (metres) for the first epoch's linearization.
    pub initial_baseline_m: [f64; 3],
    /// Carry each epoch's solved baseline forward as the next epoch's
    /// linearization point (RTKLIB-style continuity). When false, every epoch
    /// linearizes from [`initial_baseline_m`](Self::initial_baseline_m).
    pub warm_start: bool,
}

/// Why a single moving-baseline epoch solve could not complete.
#[derive(Debug, Clone, PartialEq)]
pub enum MovingBaselineError {
    /// The float baseline prerequisite failed.
    Float(FloatSolveError),
    /// The integer-fixed solve failed.
    Fixed(FixedSolveError),
}

impl core::fmt::Display for MovingBaselineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float(error) => write!(f, "moving-baseline float solve failed: {error}"),
            Self::Fixed(error) => write!(f, "moving-baseline fixed solve failed: {error}"),
        }
    }
}

impl std::error::Error for MovingBaselineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Float(error) => Some(error),
            Self::Fixed(error) => Some(error),
        }
    }
}

/// A moving-baseline epoch solve failure tagged with the epoch index it occurred
/// at, returned by the sequence solver [`solve_moving_baseline`].
#[derive(Debug, Clone, PartialEq)]
pub struct MovingBaselineSequenceError {
    /// Zero-based index of the failing epoch in the input slice.
    pub epoch_index: usize,
    /// The underlying single-epoch failure.
    pub error: MovingBaselineError,
}

impl core::fmt::Display for MovingBaselineSequenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "moving-baseline solve failed at epoch {}: {}",
            self.epoch_index, self.error
        )
    }
}

impl std::error::Error for MovingBaselineSequenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Solve one moving-baseline epoch: run the float baseline solve against the
/// epoch's own base position, then the integer-fixed re-solve conditioned on the
/// float result, and report the baseline vector plus the integer-fix verdict.
///
/// `initial_baseline_m` is the linearization point for the float solve (the
/// approximate baseline). The double difference cancels the base position, so a
/// per-epoch base is the only difference from the fixed-base static solve.
pub fn solve_moving_baseline_epoch(
    input: MovingBaselineEpoch<'_>,
    initial_baseline_m: [f64; 3],
    model: &MeasModel,
    float_opts: FloatSolveOpts,
    fixed_opts: FixedSolveOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<MovingBaselineEpochSolution, MovingBaselineError> {
    let MovingBaselineEpoch {
        base_position_m,
        epoch,
        ambiguities,
    } = input;

    let float = solve_float_baseline(
        std::slice::from_ref(epoch),
        base_position_m,
        ambiguities.ids,
        initial_baseline_m,
        model,
        float_opts,
        receiver_antenna_corrections,
    )
    .map_err(MovingBaselineError::Float)?;

    let fixed = solve_fixed_baseline(
        std::slice::from_ref(epoch),
        base_position_m,
        ambiguities,
        FloatPrior {
            baseline_m: float.baseline_m,
            ambiguities_m: &float.ambiguities_m,
            covariance_m: &float.ambiguity_covariance_m,
        },
        model,
        fixed_opts,
        receiver_antenna_corrections,
    )
    .map_err(MovingBaselineError::Fixed)?;

    let status = match fixed.search.integer_status {
        IntegerStatus::Fixed => MovingBaselineStatus::Fixed,
        IntegerStatus::NotFixed => MovingBaselineStatus::Float,
    };
    let baseline_m = fixed.baseline_m;

    Ok(MovingBaselineEpochSolution {
        base_position_m,
        baseline_m,
        baseline_length_m: norm3(baseline_m),
        status,
        float,
        fixed,
    })
}

/// Solve a sequence of moving-baseline epochs, each against its own base
/// position. Epochs are solved independently; with
/// [`MovingBaselineOpts::warm_start`] enabled each solved baseline seeds the next
/// epoch's float linearization point for continuity. The first epoch always
/// starts from [`MovingBaselineOpts::initial_baseline_m`].
pub fn solve_moving_baseline(
    epochs: &[MovingBaselineEpoch<'_>],
    opts: MovingBaselineOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<Vec<MovingBaselineEpochSolution>, MovingBaselineSequenceError> {
    let mut solutions = Vec::with_capacity(epochs.len());
    let mut initial_baseline_m = opts.initial_baseline_m;
    for (epoch_index, &epoch) in epochs.iter().enumerate() {
        let solution = solve_moving_baseline_epoch(
            epoch,
            initial_baseline_m,
            &opts.model,
            opts.float,
            opts.fixed,
            receiver_antenna_corrections,
        )
        .map_err(|error| MovingBaselineSequenceError { epoch_index, error })?;
        if opts.warm_start {
            initial_baseline_m = solution.baseline_m;
        }
        solutions.push(solution);
    }
    Ok(solutions)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::constants::{C_M_S, F_L1_HZ};
    use crate::rtk_filter::{SatMeas, StochasticModel};

    /// Five well-spread satellites with fixed integer ambiguities, the same
    /// geometry the static fixed-baseline golden uses.
    const SATS: [(&str, [f64; 3], i64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 4),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 9),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -3),
    ];

    fn range_m(sat: [f64; 3], recv: [f64; 3]) -> f64 {
        let d = [sat[0] - recv[0], sat[1] - recv[1], sat[2] - recv[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// Build a perfect synthetic double-difference epoch for a base at
    /// `base` and a rover at `base + baseline`, with the fixed cycle biases.
    fn synth_epoch(base: [f64; 3], baseline: [f64; 3]) -> Epoch {
        let rover = [
            base[0] + baseline[0],
            base[1] + baseline[1],
            base[2] + baseline[2],
        ];
        let lambda = C_M_S / F_L1_HZ;
        let mk = |pos: [f64; 3], id: &str, cycles: i64| SatMeas {
            sat: id.into(),
            sd_ambiguity_id: id.into(),
            base_code_m: range_m(pos, base),
            base_phase_m: range_m(pos, base),
            rover_code_m: range_m(pos, rover),
            rover_phase_m: range_m(pos, rover) + cycles as f64 * lambda,
            base_tx_pos: pos,
            rover_tx_pos: pos,
            pos,
        };
        Epoch {
            references: vec![mk(SATS[0].1, SATS[0].0, SATS[0].2)],
            nonref: SATS[1..].iter().map(|&(id, p, c)| mk(p, id, c)).collect(),
            velocity_mps: None,
            dt_s: 0.0,
        }
    }

    fn model() -> MeasModel {
        MeasModel {
            code_sigma_m: 0.3,
            phase_sigma_m: 0.003,
            sagnac: false,
            stochastic: StochasticModel::Simple {
                elevation_weighting: false,
            },
        }
    }

    fn ambiguity_ids() -> Vec<String> {
        vec![
            "G02".to_string(),
            "G03".to_string(),
            "G04".to_string(),
            "G05".to_string(),
        ]
    }

    fn float_opts() -> FloatSolveOpts {
        FloatSolveOpts {
            position_tol_m: 1.0e-3,
            ambiguity_tol_m: 1.0e-6,
            max_iterations: 10,
        }
    }

    fn fixed_opts() -> FixedSolveOpts {
        FixedSolveOpts {
            position_tol_m: 1.0e-3,
            ambiguity_tol_m: 1.0e-6,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: false,
            partial_min_ambiguities: 4,
        }
    }

    fn scale_maps() -> (
        BTreeMap<String, f64>,
        BTreeMap<String, f64>,
        BTreeMap<String, String>,
    ) {
        let lambda = C_M_S / F_L1_HZ;
        let ids = ambiguity_ids();
        let wavelengths = ids.iter().map(|id| (id.clone(), lambda)).collect();
        let offsets = ids.iter().map(|id| (id.clone(), 0.0)).collect();
        let satellites = ids.iter().map(|id| (id.clone(), id.clone())).collect();
        (wavelengths, offsets, satellites)
    }

    #[test]
    fn recovers_baseline_per_epoch_as_base_moves() {
        // Base walks along a track; the baseline to the rover is constant.
        let bases = [
            [4_075_580.0, 931_854.0, 4_801_568.0],
            [4_075_585.0, 931_860.0, 4_801_572.0],
            [4_075_590.0, 931_867.0, 4_801_575.0],
        ];
        let truth = [1.2, -0.85, 0.91];
        let ids = ambiguity_ids();
        let (wavelengths, offsets, satellites) = scale_maps();
        let model = model();

        let epochs: Vec<Epoch> = bases.iter().map(|&b| synth_epoch(b, truth)).collect();
        let inputs: Vec<MovingBaselineEpoch<'_>> = epochs
            .iter()
            .zip(bases.iter())
            .map(|(epoch, &base_position_m)| MovingBaselineEpoch {
                base_position_m,
                epoch,
                ambiguities: AmbiguitySet {
                    ids: &ids,
                    satellites: &satellites,
                    scale: super::super::AmbiguityScale {
                        wavelengths_m: &wavelengths,
                        offsets_m: &offsets,
                    },
                    float_only_systems: &[],
                },
            })
            .collect();

        let opts = MovingBaselineOpts {
            model,
            float: float_opts(),
            fixed: fixed_opts(),
            initial_baseline_m: [-30.0, 25.0, -10.0],
            warm_start: true,
        };

        let solutions = solve_moving_baseline(&inputs, opts, None).expect("moving-baseline solve");
        assert_eq!(solutions.len(), 3);
        for (solution, &base) in solutions.iter().zip(bases.iter()) {
            assert_eq!(solution.status, MovingBaselineStatus::Fixed);
            assert_eq!(solution.base_position_m, base);
            for (k, (got, want)) in solution.baseline_m.iter().zip(truth.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= 1.0e-6,
                    "baseline component {k}: {got} vs {want}"
                );
            }
            let truth_len =
                (truth[0] * truth[0] + truth[1] * truth[1] + truth[2] * truth[2]).sqrt();
            assert!((solution.baseline_length_m - truth_len).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn moving_baseline_tracks_a_changing_baseline() {
        // Both platforms move and the baseline between them changes each epoch.
        let scenario = [
            ([4_075_580.0, 931_854.0, 4_801_568.0], [1.2, -0.85, 0.91]),
            ([4_075_585.0, 931_860.0, 4_801_572.0], [2.0, 0.5, -1.3]),
            ([4_075_590.0, 931_867.0, 4_801_575.0], [-0.7, 1.1, 0.4]),
        ];
        let ids = ambiguity_ids();
        let (wavelengths, offsets, satellites) = scale_maps();
        let model = model();

        let epochs: Vec<Epoch> = scenario
            .iter()
            .map(|&(base, baseline)| synth_epoch(base, baseline))
            .collect();
        let inputs: Vec<MovingBaselineEpoch<'_>> = epochs
            .iter()
            .zip(scenario.iter())
            .map(|(epoch, &(base_position_m, _))| MovingBaselineEpoch {
                base_position_m,
                epoch,
                ambiguities: AmbiguitySet {
                    ids: &ids,
                    satellites: &satellites,
                    scale: super::super::AmbiguityScale {
                        wavelengths_m: &wavelengths,
                        offsets_m: &offsets,
                    },
                    float_only_systems: &[],
                },
            })
            .collect();

        let opts = MovingBaselineOpts {
            model,
            float: float_opts(),
            fixed: fixed_opts(),
            initial_baseline_m: [0.0, 0.0, 0.0],
            warm_start: false,
        };

        let solutions = solve_moving_baseline(&inputs, opts, None).expect("moving-baseline solve");
        for (solution, &(_, baseline)) in solutions.iter().zip(scenario.iter()) {
            assert_eq!(solution.status, MovingBaselineStatus::Fixed);
            for (k, (got, want)) in solution.baseline_m.iter().zip(baseline.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= 1.0e-6,
                    "baseline component {k}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn single_epoch_solve_matches_sequence_first_epoch() {
        let base = [4_075_580.0, 931_854.0, 4_801_568.0];
        let truth = [1.2, -0.85, 0.91];
        let ids = ambiguity_ids();
        let (wavelengths, offsets, satellites) = scale_maps();
        let epoch = synth_epoch(base, truth);
        let input = MovingBaselineEpoch {
            base_position_m: base,
            epoch: &epoch,
            ambiguities: AmbiguitySet {
                ids: &ids,
                satellites: &satellites,
                scale: super::super::AmbiguityScale {
                    wavelengths_m: &wavelengths,
                    offsets_m: &offsets,
                },
                float_only_systems: &[],
            },
        };

        let solution = solve_moving_baseline_epoch(
            input,
            [-30.0, 25.0, -10.0],
            &model(),
            float_opts(),
            fixed_opts(),
            None,
        )
        .expect("single-epoch moving-baseline solve");
        assert_eq!(solution.status, MovingBaselineStatus::Fixed);
        for (got, want) in solution.baseline_m.iter().zip(truth.iter()) {
            assert!((got - want).abs() <= 1.0e-6);
        }
    }
}
