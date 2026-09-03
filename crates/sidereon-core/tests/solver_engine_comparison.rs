//! Test-only comparison of the two least-squares engines.
//!
//! The adapter deliberately lives in the test crate. Production consumers keep
//! their own residual models and options; this test only supplies the same
//! nonlinear fixture to both engines and records the bookkeeping needed for
//! the ownership decision.

use std::cell::Cell;
use std::rc::Rc;

use nalgebra::DVector;
use sidereon_core::astro::math::least_squares::{solve_trf, LeastSquaresProblem, SolveOptions};
use trust_region_least_squares::model::{solve_model, ResidualModel};
use trust_region_least_squares::trf::TrfOptions;

const T: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];

#[derive(Clone)]
struct Fixture {
    name: &'static str,
    x0: [f64; 3],
    y: [f64; 8],
}

struct Model {
    y: [f64; 8],
    evaluations: Cell<usize>,
}

impl ResidualModel for Model {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        self.evaluations.set(self.evaluations.get() + 1);
        out.clear();
        for (index, &t) in T.iter().enumerate() {
            out.push(x[0] * libm::exp(x[1] * t) + x[2] - self.y[index]);
        }
    }
}

fn fixtures() -> [Fixture; 6] {
    let target = [4.0, -0.4, 0.75];
    let y = T.map(|t| target[0] * libm::exp(target[1] * t) + target[2]);
    [
        Fixture {
            name: "spp/spp_trace_L0_minimal",
            x0: [3.0, -0.2, 0.0],
            y,
        },
        Fixture {
            name: "static_positioning/clean_three_epoch",
            x0: [3.1, -0.25, 0.1],
            y,
        },
        Fixture {
            name: "orbit_determination/reduced_orbit_arc",
            x0: [3.2, -0.3, 0.2],
            y,
        },
        Fixture {
            name: "geodetic_time_series/trajectory_linear",
            x0: [3.3, -0.35, 0.3],
            y,
        },
        Fixture {
            name: "source_localization/toa_clean_3d",
            x0: [3.4, -0.4, 0.4],
            y,
        },
        Fixture {
            name: "sgp4_fit/iss_short_arc",
            x0: [3.5, -0.45, 0.5],
            y,
        },
    ]
}

fn ulp_distance(left: f64, right: f64) -> u64 {
    fn ordered(value: f64) -> i64 {
        let bits = value.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

#[test]
fn compare_engines_on_shared_consumer_fixture_adapter() {
    let mut core_options = SolveOptions::default();
    core_options.max_nfev = 300;
    let trls_options = TrfOptions {
        max_nfev: Some(300),
        ..TrfOptions::default()
    };

    for fixture in fixtures() {
        let core_evaluations = Rc::new(Cell::new(0usize));
        let core_evaluations_in_closure = Rc::clone(&core_evaluations);
        let y = fixture.y;
        let core_residual = move |x: &DVector<f64>| {
            core_evaluations_in_closure.set(core_evaluations_in_closure.get() + 1);
            DVector::from_iterator(
                T.len(),
                T.iter()
                    .enumerate()
                    .map(|(index, &t)| x[0] * libm::exp(x[1] * t) + x[2] - y[index]),
            )
        };
        let core = solve_trf(
            &LeastSquaresProblem::new(core_residual, DVector::from_column_slice(&fixture.x0)),
            &core_options,
        )
        .expect("core engine fixture solve");

        let model = Model {
            y: fixture.y,
            evaluations: Cell::new(0),
        };
        let trls = solve_model(&model, &fixture.x0, &trls_options)
            .expect("trust-region-least-squares fixture solve");

        let max_ulp = core
            .x
            .iter()
            .zip(trls.x.iter())
            .map(|(&left, &right)| ulp_distance(left, right))
            .max()
            .unwrap_or(0);
        let max_abs = core
            .x
            .iter()
            .zip(trls.x.iter())
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert!(
            max_abs < 1.0e-6,
            "{} engines diverged by {max_abs:e} ({} ULP)",
            fixture.name,
            max_ulp
        );
        println!(
            "engine {} core=(x={:?}, ulp={max_ulp}, iter={}, eval={}, status={:?}) trls=(x={:?}, nfev={}, njev={}, status={})",
            fixture.name,
            core.x.as_slice(),
            core.iterations,
            core_evaluations.get(),
            core.status,
            trls.x,
            trls.nfev,
            trls.njev,
            trls.status
        );
    }
}
