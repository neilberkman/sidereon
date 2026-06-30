//! Special functions with a deterministic, cross-platform implementation.
//!
//! The Gauss error function `erf` is provided by the pure-Rust `libm` crate
//! rather than the platform C math library. The system `libm` is not
//! bit-identical across platforms, so binding it via `extern "C"` would break
//! cross-platform 0-ULP determinism. `libm::erf` is a deterministic port that
//! produces the same bits on every platform.

/// Gauss error function, deterministic across platforms via the `libm` crate.
#[inline]
pub(crate) fn erf(x: f64) -> f64 {
    libm::erf(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_has_frozen_bits() {
        // Frozen-bits golden of the deterministic `libm::erf`. These bits are
        // identical on every platform; a tolerance check would have hidden the
        // 1-ULP drift the platform C `erf` introduced at erf(1.0).
        assert_eq!(erf(0.0).to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(erf(0.75).to_bits(), 0x3fe6_c1c9_759d_0e60);
        assert_eq!(erf(1.0).to_bits(), 0x3fea_f767_a741_088b);
        assert_eq!(erf(1.5).to_bits(), 0x3fee_ea55_5713_7ae0);
        assert_eq!(erf(2.0).to_bits(), 0x3fef_d9ae_1427_95e3);
        assert_eq!(erf(6.0).to_bits(), 0x3ff0_0000_0000_0000);
        // erf is odd: erf(-x) == -erf(x), exactly.
        assert_eq!(erf(-0.75).to_bits(), (-erf(0.75)).to_bits());
        assert_eq!(erf(-6.0).to_bits(), (-1.0_f64).to_bits());
    }
}
