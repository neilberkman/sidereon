pub struct PIController {
    pub safety_factor: f64,
    pub min_scale: f64,
    pub max_scale: f64,
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
    pub fn next_step(&self, current_h: f64, error: f64) -> f64 {
        if error <= 1e-15 {
            return current_h * self.max_scale;
        }

        // Classic Hairer/Wanner controller
        let factor = self.safety_factor * (1.0 / error).powf(1.0 / (self.order + 1.0));
        current_h * factor.clamp(self.min_scale, self.max_scale)
    }
}
