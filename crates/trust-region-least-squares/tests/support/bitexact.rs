//! Shared guard for the platform-pinned hex-bit replay fixtures.
//!
//! The golden-replay tests assert `f64::to_bits` equality against committed
//! SciPy 1.18.0 fixtures. Those bits only reproduce on the canonical reference
//! host (a non-AVX-512 x86_64 Linux box loading the wheel-bundled OpenBLAS),
//! because numpy's array `pow` dispatches a wider SVML kernel on AVX-512 that
//! differs by 1 ULP and cannot be disabled at runtime. On any other platform a
//! downstream `cargo test` would fail through no fault of the crate, so the
//! replays stay off by default and only run when the canonical environment is
//! signalled via `SIDEREON_BITEXACT`.
//!
//! This file is pulled in via `#[path = "support/bitexact.rs"] mod bitexact;`
//! rather than being its own test binary (it lives under `tests/support/`,
//! which Cargo does not compile as a test target).

#![allow(dead_code)]

/// True when the canonical bit-exact replay environment is signalled.
pub fn bitexact_enabled() -> bool {
    std::env::var("SIDEREON_BITEXACT").is_ok()
}

/// Top-of-test guard for the platform-pinned hex-bit replays. Returns `true`
/// (the test body should `return` immediately) and prints a one-line notice
/// when the bit-exact environment is not signalled, so a default `cargo test`
/// is green on every platform.
pub fn skip_platform_pinned_replay() -> bool {
    if bitexact_enabled() {
        false
    } else {
        eprintln!(
            "skipping platform-pinned bit-exact replay; set SIDEREON_BITEXACT=1 on a non-AVX-512 host"
        );
        true
    }
}
