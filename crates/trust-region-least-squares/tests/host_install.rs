//! Process-wide identity of the exact host numerics configuration.
//!
//! Bit-for-bit parity is only meaningful if every result in a process came from
//! the *same* pinned LAPACK/NumPy runtime, so the configuration can be
//! installed once and then read back. Installing the identical configuration
//! again is a no-op; installing a different one is rejected rather than
//! silently re-pointing part of the process at another runtime.
//!
//! This lives in its own test binary because the installed configuration is
//! process-global: Cargo runs each integration-test file as a separate process,
//! so the single test below owns the whole lifecycle.

use trust_region_least_squares::hostlapack::{LapackError, LapackSvd};

#[test]
fn identical_install_is_idempotent_and_conflicting_reconfiguration_fails() {
    let pinned = "/opt/pinned/scipy.libs/libscipy_openblas.so";
    let config = LapackSvd::with_path(pinned);

    assert!(
        LapackSvd::installed().is_none(),
        "nothing should be installed before the first install"
    );

    config.install().expect("first install");
    assert_eq!(LapackSvd::installed(), Some(&config));

    // Idempotent: the same handle, and an independently built equal handle.
    config.install().expect("identical reinstall");
    LapackSvd::with_path(pinned)
        .install()
        .expect("equal configuration reinstall");

    // A different LAPACK path is a conflict.
    let err = LapackSvd::with_path("/opt/other/scipy.libs/libscipy_openblas.so")
        .install()
        .expect_err("conflicting LAPACK path must be rejected");
    assert!(
        matches!(err, LapackError::ConflictingHostInstall { .. }),
        "{err}"
    );

    // So is refining the same LAPACK path with a NumPy BLAS path: that changes
    // which runtime the dot/matvec/power dispatch binds to.
    let err = LapackSvd::with_path(pinned)
        .with_numpy_blas_path("/opt/pinned/numpy.libs/libscipy_openblas.so")
        .install()
        .expect_err("conflicting NumPy BLAS path must be rejected");
    assert!(
        matches!(err, LapackError::ConflictingHostInstall { .. }),
        "{err}"
    );

    // A rejected install must not have disturbed the installed configuration.
    assert_eq!(LapackSvd::installed(), Some(&config));
}
