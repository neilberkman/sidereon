use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Epoch-tagged position and velocity vectors exchanged by the astrodynamics
/// APIs.
///
/// Numerical propagation uses ECI/GCRF vectors in kilometers and kilometers
/// per second and advances the epoch along the TDB seconds-from-J2000 axis.
/// The [`crate::astro::relative`] APIs reuse this container for chief-frame
/// relative vectors.
pub struct CartesianState {
    /// Propagation epoch in TDB seconds from J2000.
    ///
    /// Integrators validate this value as finite and advance it by each step.
    /// Relative transforms copy an input epoch, while
    /// [`crate::astro::relative::cw_propagate`] adds its `dt`.
    pub epoch_tdb_seconds: f64,
    /// Position components in kilometers, stored in x/y/z order.
    ///
    /// Numerical propagation and force/frame transforms use ECI/GCRF
    /// coordinates; [`crate::astro::relative`] uses this field for
    /// deputy-minus-chief components in the chief's rotating frame.
    pub position_km: Vector3<f64>,
    /// Velocity components in kilometers per second, stored in x/y/z order.
    ///
    /// Numerical propagation and force/frame transforms use ECI/GCRF
    /// coordinates; [`crate::astro::relative`] uses this field for relative
    /// velocity components in the chief's rotating frame.
    pub velocity_km_s: Vector3<f64>,
}

impl CartesianState {
    /// Construct a state from an epoch and position and velocity arrays.
    ///
    /// The arrays are copied into nalgebra vectors in x/y/z order and the
    /// epoch is copied unchanged. This constructor performs no finiteness or
    /// frame validation.
    pub fn new(epoch_tdb_seconds: f64, position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            epoch_tdb_seconds,
            position_km: Vector3::from_column_slice(&position),
            velocity_km_s: Vector3::from_column_slice(&velocity),
        }
    }

    /// Copy position components into an `[x, y, z]` array.
    ///
    /// Values are returned in the order accepted by [`Self::new`], without
    /// unit conversion or validation.
    pub fn position_array(&self) -> [f64; 3] {
        [self.position_km.x, self.position_km.y, self.position_km.z]
    }

    /// Copy velocity components into an `[x, y, z]` array.
    ///
    /// Values are returned in the order accepted by [`Self::new`], without
    /// unit conversion or validation.
    pub fn velocity_array(&self) -> [f64; 3] {
        [
            self.velocity_km_s.x,
            self.velocity_km_s.y,
            self.velocity_km_s.z,
        ]
    }
}

#[derive(Debug, Clone, Copy)]
/// Cartesian position and velocity rates supplied to a numerical integrator.
///
/// Dynamics models provide the two rates, RK4 and DP54 integrate both into a
/// [`CartesianState`], and DP54 retains seven derivatives for dense output.
pub struct StateDerivative {
    /// Position rate in kilometers per second.
    ///
    /// [`crate::astro::propagator::dynamics::OrbitalDynamics`] sets this to
    /// the state's velocity; integrators multiply it by elapsed seconds before
    /// adding it to position.
    pub dpos_km_s: Vector3<f64>,
    /// Velocity rate (acceleration) in kilometers per second squared.
    ///
    /// [`crate::astro::propagator::dynamics::OrbitalDynamics`] obtains this
    /// from the force model; integrators multiply it by elapsed seconds before
    /// adding it to velocity.
    pub dvel_km_s2: Vector3<f64>,
}

impl StateDerivative {
    /// Construct a derivative from position and velocity rates.
    ///
    /// The supplied vectors are stored unchanged as the two ODE components;
    /// this constructor performs no validation.
    pub fn new(dpos: Vector3<f64>, dvel: Vector3<f64>) -> Self {
        Self {
            dpos_km_s: dpos,
            dvel_km_s2: dvel,
        }
    }
}
