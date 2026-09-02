//! Numerical astrodynamics engine for orbit propagation, force models, and
//! future flight-dynamics primitives.
//!
//! Current scope:
//!
//! - inertial Cartesian state representation
//! - two-body gravity and J2 perturbation
//! - fixed-step RK4
//! - adaptive Dormand-Prince 5(4) (`DP54`)
//! - propagation results with step statistics
//!
//! Planned future work includes dense output, event handling, richer propagation
//! contexts, additional force models, covariance propagation, estimation, and
//! maneuver support.

pub mod almanac;
pub mod angles;
pub mod anomaly;
pub mod apparent;
pub mod atmosphere;
pub mod bodies;
pub mod cdm;
pub mod conjunction;
pub mod constants;
pub mod covariance;
pub mod coverage;
pub mod data;
pub mod doppler;
pub mod elements;
pub mod equinoctial;
/// Propagation error types distinguishing invalid inputs from model or integration failures.
pub mod error;
/// Event detectors for geometry crossings and close approaches along propagated trajectories.
pub mod events;
/// Force models computing spacecraft acceleration from state and environment.
pub mod forces;
pub mod frames;
/// ODE integrators and step-control types for propagating dynamic states.
pub mod integrators;
pub mod iod;
pub mod lambert;
/// Numerical matrix, vector, polynomial, and robust-estimation helpers used by solvers.
pub mod math;
pub mod ndm;
pub mod observation;
pub mod oem;
pub mod omm;
pub mod opm;
pub mod passes;
/// Orbit propagation APIs; start with dynamics, integration options, and propagation result types.
pub mod propagator;
pub mod relative;
pub mod rf;
pub mod sgp4;
pub mod space_weather;
pub mod spk;
/// Cartesian state and derivative types exchanged by the propagator.
pub mod state;
pub mod tca;
pub mod tdm;
pub mod time;
pub mod tle;
pub mod tolerances;
pub mod xml;

pub use spk::{
    DafByteOrder, DafFileRecord, DafSpk, Spk, SpkError, SpkSegmentDescriptor, SpkState,
    SpkStateVector,
};

#[cfg(all(feature = "sgp4-debug-oracle", sgp4_oracle_built))]
#[doc(hidden)]
pub mod sgp4_cpp_oracle {
    //! Test-only oracle bridge to the Vallado C++ implementation.
    //! Compiled in only when the `sgp4-debug-oracle` feature is on and the
    //! development-only C++ oracle sources were found by the build script.
    //! Not part of the public API.

    use std::os::raw::{c_char, c_double, c_int};

    pub const CPP_DUMP_DOUBLE_COUNT: usize = 112;
    pub const CPP_DUMP_INT_COUNT: usize = 5;

    extern "C" {
        pub fn cpp_sgp4init_dump(
            satnum: *const c_char,
            opsmode: c_char,
            epoch_sgp4: c_double,
            bstar: c_double,
            ndot: c_double,
            nddot: c_double,
            ecco: c_double,
            argpo: c_double,
            inclo: c_double,
            mo: c_double,
            no_kozai: c_double,
            nodeo: c_double,
            epochyr: c_int,
            epochdays: c_double,
            jdsatepoch: c_double,
            jdsatepoch_frac: c_double,
            double_out: *mut c_double,
            int_out: *mut c_int,
        ) -> c_int;

        pub fn cpp_sgp4_step(
            satnum: *const c_char,
            opsmode: c_char,
            epoch_sgp4: c_double,
            bstar: c_double,
            ndot: c_double,
            nddot: c_double,
            ecco: c_double,
            argpo: c_double,
            inclo: c_double,
            mo: c_double,
            no_kozai: c_double,
            nodeo: c_double,
            epochyr: c_int,
            epochdays: c_double,
            jdsatepoch: c_double,
            jdsatepoch_frac: c_double,
            tsince: c_double,
            r_out: *mut c_double,
            v_out: *mut c_double,
        ) -> c_int;
    }

    /// Force-reference the C symbols so the linker pulls in the static lib.
    /// Without this, the rlib has no use of the symbols and the linker
    /// strips the entire archive when compiling integration tests.
    #[doc(hidden)]
    pub fn force_link_oracle() -> usize {
        let init_dump = cpp_sgp4init_dump as *const ();
        let step = cpp_sgp4_step as *const ();

        init_dump as usize ^ step as usize
    }
}

#[cfg(all(feature = "sgp4-debug-oracle", sgp4_oracle_built))]
pub use sgp4_cpp_oracle::cpp_sgp4_step;

pub use anomaly::{
    eccentric_to_mean, eccentric_to_true, mean_to_eccentric, mean_to_true, propagate_kepler,
    solve_kepler, true_to_eccentric, true_to_mean, AnomalyError, KeplerSolution,
};
pub use elements::{coe2rv, rv2coe, ClassicalElements, ElementsError, OrbitType};
pub use equinoctial::{
    coe2eq, coe2mee, eq2coe, eq2mee, eq2rv, mee2coe, mee2eq, mee2rv, rv2eq, rv2mee,
    EquinoctialElements, EquinoctialError, ModifiedEquinoctialElements, RetrogradeFactor,
};
pub use error::PropagationError;
pub use state::CartesianState;
pub use time::Time;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::forces::TwoBodyGravity;
    use crate::astro::integrators::{Integrator, DP54};
    use crate::astro::propagator::{api::IntegratorOptions, OrbitalDynamics, PropagationContext};
    use nalgebra::Vector3;

    #[test]
    fn test_two_body_dp54_precision() {
        let r_mag: f64 = 7000.0;
        let mu: f64 = 398600.4418;
        let v_mag: f64 = (mu / r_mag).sqrt();
        let initial_state = CartesianState {
            epoch_tdb_seconds: 0.0,
            position_km: Vector3::new(r_mag, 0.0, 0.0),
            velocity_km_s: Vector3::new(0.0, v_mag, 0.0),
        };

        let force = TwoBodyGravity::default();
        let dynamics = OrbitalDynamics {
            force_model: &force,
        };
        let integrator = DP54;
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions {
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            initial_step: 1.0,
            min_step: 1e-15,
            ..IntegratorOptions::default()
        };

        let period = 2.0 * std::f64::consts::PI * (r_mag.powi(3) / mu).sqrt();
        let result = integrator
            .propagate(initial_state, period, &dynamics, &ctx, &opts)
            .unwrap();

        let final_pos = result.final_state.position_km;
        let final_vel = result.final_state.velocity_km_s;

        // Oracle 1: Return to start precision (Sub-millimeter)
        assert!(
            (final_pos.x - r_mag).abs() < 1e-7,
            "Position X error too large: {}",
            (final_pos.x - r_mag).abs()
        );
        assert!(
            final_pos.y.abs() < 1e-7,
            "Position Y error too large: {}",
            final_pos.y.abs()
        );

        // Oracle 2: Energy conservation (Specific mechanical energy)
        let initial_energy = v_mag.powi(2) / 2.0 - mu / r_mag;
        let final_v_mag = final_vel.norm();
        let final_r_mag = final_pos.norm();
        let final_energy = final_v_mag.powi(2) / 2.0 - mu / final_r_mag;
        assert!(
            (final_energy - initial_energy).abs() < 1e-10,
            "Energy conservation failure: {}",
            (final_energy - initial_energy).abs()
        );
    }
}
