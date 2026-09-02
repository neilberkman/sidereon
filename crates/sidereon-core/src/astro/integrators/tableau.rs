/// Coefficients for the Dormand-Prince embedded fifth- and fourth-order step
/// used by [`DP54`](crate::astro::integrators::DP54).
///
/// [`Default::default`] fills the seven stage-time fractions, lower-triangular
/// stage coefficients, and paired solution weights used by the adaptive step
/// calculation.
pub struct DP54Tableau {
    /// Dimensionless fractions of the signed step duration used for
    /// intermediate stage epochs by [`DP54`](crate::astro::integrators::DP54).
    ///
    /// [`Default::default`] sets the seven values to
    /// `[0, 1/5, 3/10, 4/5, 8/9, 1, 1]`; the step calculation multiplies an
    /// entry by `h` and adds it to the current epoch.
    pub c: [f64; 7],
    /// Lower-triangular stage-coupling coefficients used by
    /// [`DP54`](crate::astro::integrators::DP54).
    ///
    /// For stage `i` from 1 through 5, the step calculation uses the first `i`
    /// values of row `i` to combine previously computed position and velocity
    /// derivatives before multiplying by `h`. [`Default::default`] supplies
    /// rows of lengths 1 through 6 after the empty first row.
    pub a: Vec<Vec<f64>>,
    /// Weights for the fifth-order position and velocity increments computed
    /// by [`DP54`](crate::astro::integrators::DP54).
    ///
    /// The step calculation applies the first six values to the stage
    /// derivatives, multiplies the sums by `h`, and adds them to the input
    /// state. The default seventh value is zero because the FSAL derivative is
    /// reserved for the embedded error calculation.
    pub b5: [f64; 7],
    /// Weights for the embedded fourth-order position and velocity increments
    /// used by [`DP54`](crate::astro::integrators::DP54).
    ///
    /// The step calculation applies all seven values, including the derivative
    /// evaluated at the proposed endpoint, and subtracts the resulting
    /// increments from the fifth-order increments to form adaptive error
    /// estimates. [`Default::default`] sets the seventh value to `1/40`.
    pub b4: [f64; 7],
}

impl Default for DP54Tableau {
    fn default() -> Self {
        // Dormand-Prince 5(4) coefficients (DOPRI5)
        Self {
            c: [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0],
            a: vec![
                vec![],
                vec![1.0 / 5.0],
                vec![3.0 / 40.0, 9.0 / 40.0],
                vec![44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
                vec![
                    19372.0 / 6561.0,
                    -25360.0 / 2187.0,
                    64448.0 / 6561.0,
                    -212.0 / 729.0,
                ],
                vec![
                    9017.0 / 3168.0,
                    -355.0 / 33.0,
                    46732.0 / 5247.0,
                    49.0 / 176.0,
                    -5103.0 / 18656.0,
                ],
                vec![
                    35.0 / 384.0,
                    0.0,
                    500.0 / 1113.0,
                    125.0 / 192.0,
                    -2187.0 / 6784.0,
                    11.0 / 84.0,
                ],
            ],
            b5: [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ],
            b4: [
                5179.0 / 57600.0,
                0.0,
                7571.0 / 16695.0,
                393.0 / 640.0,
                -92097.0 / 339200.0,
                187.0 / 2100.0,
                1.0 / 40.0,
            ],
        }
    }
}
