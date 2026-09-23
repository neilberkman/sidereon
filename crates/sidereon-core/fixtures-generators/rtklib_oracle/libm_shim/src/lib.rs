//! The Rust `libm` crate's `sin`, `cos`, `atan2`, `sqrt` and `fabs` as C symbols
//! `sidereon_libm_*`, for the RTKLIB ephemeris oracle harness.

/// `libm::sin`.
#[no_mangle]
pub extern "C" fn sidereon_libm_sin(x: f64) -> f64 {
    libm::sin(x)
}

/// `libm::cos`.
#[no_mangle]
pub extern "C" fn sidereon_libm_cos(x: f64) -> f64 {
    libm::cos(x)
}

/// `libm::atan2`.
#[no_mangle]
pub extern "C" fn sidereon_libm_atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// `libm::sqrt`.
#[no_mangle]
pub extern "C" fn sidereon_libm_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// `libm::fabs`.
#[no_mangle]
pub extern "C" fn sidereon_libm_fabs(x: f64) -> f64 {
    libm::fabs(x)
}
