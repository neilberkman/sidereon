use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianState {
    pub epoch_tdb_seconds: f64,
    pub position_km: Vector3<f64>,
    pub velocity_km_s: Vector3<f64>,
}

impl CartesianState {
    pub fn new(epoch_tdb_seconds: f64, position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            epoch_tdb_seconds,
            position_km: Vector3::from_column_slice(&position),
            velocity_km_s: Vector3::from_column_slice(&velocity),
        }
    }

    pub fn position_array(&self) -> [f64; 3] {
        [self.position_km.x, self.position_km.y, self.position_km.z]
    }

    pub fn velocity_array(&self) -> [f64; 3] {
        [
            self.velocity_km_s.x,
            self.velocity_km_s.y,
            self.velocity_km_s.z,
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StateDerivative {
    pub dpos_km_s: Vector3<f64>,
    pub dvel_km_s2: Vector3<f64>,
}

impl StateDerivative {
    pub fn new(dpos: Vector3<f64>, dvel: Vector3<f64>) -> Self {
        Self {
            dpos_km_s: dpos,
            dvel_km_s2: dvel,
        }
    }
}
