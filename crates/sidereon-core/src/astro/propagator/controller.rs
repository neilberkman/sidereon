/// Adaptive step-size controller used by [`crate::astro::integrators::DP54`]
/// after each trial step. DP54 initializes it from its defaults, overrides
/// [`PIController::order`] to `5.0`, and separately bounds the returned step by
/// [`crate::astro::propagator::api::IntegratorOptions::max_step`].
pub struct PIController {
    /// Multiplier applied to the power-law factor before the scale bounds are
    /// applied in [`PIController::next_step`]. The default is `0.9`.
    pub safety_factor: f64,
    /// Lower bound for the scale factor in the ordinary error branch of
    /// [`PIController::next_step`]. The default is `0.33`; very small errors
    /// use [`PIController::max_scale`] directly instead.
    pub min_scale: f64,
    /// Upper bound for the scale factor in the ordinary error branch of
    /// [`PIController::next_step`], and the direct multiplier for errors at or
    /// below `1e-15`. The default is `6.0`.
    pub max_scale: f64,
    /// Value used in the denominator of the power exponent as `order + 1.0`.
    /// [`crate::astro::integrators::DP54`] overrides the default `8.0` with
    /// `5.0` for its fifth-order error estimate.
    pub order: f64,
}

impl Default for PIController {
    fn default() -> Self {
        Self {
            safety_factor: 0.9,
            min_scale: 0.33,
            max_scale: 6.0,
            order: 8.0,
        }
    }
}

impl PIController {
    /// Compute the next signed integration step from a normalized error.
    /// `current_h` is the trial step in seconds, while DP54 supplies `error` as
    /// the larger of its normalized position and velocity error ratios. For
    /// `error <= 1e-15`, this returns `current_h * max_scale`; otherwise it
    /// applies the classic Hairer/Wanner factor
    /// `safety_factor * pow(1.0 / error, 1.0 / (order + 1.0))`, clamps that
    /// factor to `[min_scale, max_scale]`, and multiplies by `current_h`.
    pub fn next_step(&self, current_h: f64, error: f64) -> f64 {
        if error <= 1e-15 {
            return current_h * self.max_scale;
        }

        // Classic Hairer/Wanner controller
        let factor = self.safety_factor * libm::pow(1.0 / error, 1.0 / (self.order + 1.0));
        current_h * factor.clamp(self.min_scale, self.max_scale)
    }
}
