use crate::astro::state::CartesianState;
use crate::astro::state::StateDerivative;
use nalgebra::Vector3;

/// Dense output segment for continuous interpolation.
/// Currently implements Shampine's 4th-order continuous extension for DP5(4).
#[derive(Debug, Clone)]
pub struct DenseSegment {
    pub t_start: f64,
    pub h: f64,
    pub y_start: CartesianState,
    pub y_end: CartesianState,
    pub ks: [StateDerivative; 7],
}

impl DenseSegment {
    /// Evaluate the interpolant at time t.
    /// Returns Err if t is out of range [t_start, t_start + h] (with a small epsilon).
    pub fn eval(&self, t: f64) -> Result<CartesianState, String> {
        // Bit-exact endpoint checks based on time
        if t == self.t_start {
            return Ok(self.y_start);
        }
        if t == self.t_start + self.h {
            return Ok(self.y_end);
        }

        let theta = if self.h.abs() < 1e-18 {
            0.0
        } else {
            (t - self.t_start) / self.h
        };

        // Allow for very small numerical overshoot at boundaries
        if !(-1e-12..=1.0 + 1e-12).contains(&theta) {
            let t_end = self.t_start + self.h;
            return Err(format!(
                "Time {} out of range [{}, {}] (theta={})",
                t, self.t_start, t_end, theta
            ));
        }

        let theta = theta.clamp(0.0, 1.0);

        // P[i, j] coefficients for theta^1, theta^2, theta^3, theta^4
        let p_col1 = [
            -8048581381.0 / 2820520608.0,
            0.0,
            131558114200.0 / 32700410799.0,
            -1754552775.0 / 470086768.0,
            127303824393.0 / 49829197408.0,
            -282668133.0 / 205662961.0,
            40617522.0 / 29380423.0,
        ];

        let p_col2 = [
            8663915743.0 / 2820520608.0,
            0.0,
            -68118460800.0 / 10900136933.0,
            14199869525.0 / 1410260304.0,
            -318862633887.0 / 49829197408.0,
            2019193451.0 / 616988883.0,
            -110615467.0 / 29380423.0,
        ];

        let p_col3 = [
            -12715105075.0 / 11282082432.0,
            0.0,
            87487479700.0 / 32700410799.0,
            -10690763975.0 / 1880347072.0,
            701980252875.0 / 199316789632.0,
            -1453857185.0 / 822651844.0,
            69997945.0 / 29380423.0,
        ];

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta4 = theta3 * theta;

        let mut dpos = Vector3::zeros();
        let mut dvel = Vector3::zeros();

        // k1 contribution (p_1,0 = 1)
        let b1 = theta + p_col1[0] * theta2 + p_col2[0] * theta3 + p_col3[0] * theta4;
        dpos += self.ks[0].dpos_km_s * b1;
        dvel += self.ks[0].dvel_km_s2 * b1;

        // k2..k7 contributions (p_i,0 = 0)
        for i in 1..7 {
            let bi = p_col1[i] * theta2 + p_col2[i] * theta3 + p_col3[i] * theta4;
            dpos += self.ks[i].dpos_km_s * bi;
            dvel += self.ks[i].dvel_km_s2 * bi;
        }

        Ok(CartesianState {
            epoch_tdb_seconds: t,
            position_km: self.y_start.position_km + dpos * self.h,
            velocity_km_s: self.y_start.velocity_km_s + dvel * self.h,
        })
    }

    pub fn from_dp54_stages(
        t_start: f64,
        h: f64,
        y_start: CartesianState,
        y_end: CartesianState,
        ks: &[StateDerivative; 7],
    ) -> Self {
        Self {
            t_start,
            h,
            y_start,
            y_end,
            ks: *ks,
        }
    }

    pub fn t_end(&self) -> f64 {
        self.t_start + self.h
    }
}

/// Collection of dense segments covering a full propagation range.
#[derive(Debug, Clone, Default)]
pub struct DenseOutput {
    pub segments: Vec<DenseSegment>,
}

impl DenseOutput {
    /// Evaluate the interpolated state at any time t within the covered range.
    pub fn eval(&self, t: f64) -> Result<CartesianState, String> {
        if self.segments.is_empty() {
            return Err("Dense output is empty".to_string());
        }

        let first = &self.segments[0];
        let last = &self.segments[self.segments.len() - 1];

        let t_start = first.t_start;
        let t_end = last.t_end();

        let (t_min, t_max) = if t_start <= t_end {
            (t_start, t_end)
        } else {
            (t_end, t_start)
        };

        if t < t_min - 1e-7 || t > t_max + 1e-7 {
            return Err(format!("Time {t} out of range [{t_min}, {t_max}]"));
        }

        // Binary search for the correct segment
        let forward = t_start <= t_end;
        let idx = if forward {
            self.segments
                .partition_point(|s| s.t_start <= t)
                .saturating_sub(1)
        } else {
            self.segments
                .partition_point(|s| s.t_start >= t)
                .saturating_sub(1)
        };

        self.segments[idx].eval(t)
    }
}
