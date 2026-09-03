#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

/// Parse EMS and RTKLIB SBAS log lines into [`SbasLogBlock`] records.
///
/// EMS calendar components are converted to GPST, while RTKLIB lines provide
/// GPST week and seconds-of-week directly. Recognized records retain input
/// order and classify 29-byte bodies or 32-byte framed blocks; unrecognized
/// lines are skipped, while invalid recognized epochs or block lengths return
/// errors.
pub mod format;

/// Decode and encode SBAS body and CRC-framed wire blocks.
///
/// The decoder maps phase-A message IDs to typed [`SbasMessage`] records and
/// retains other six-bit IDs as raw unsupported payloads. [`SbasBlock`] keeps
/// the selected [`SbasWireForm`] so encoding can reproduce either form.
pub mod message;

/// Apply stored SBAS corrections through broadcast ephemeris source adapters.
///
/// The borrowed and owned adapters reject disabled source GEOs and withdrawn
/// satellites, combine fresh fast and GPS long-term corrections when
/// available, and otherwise follow [`SbasSolveMode`]. A selected GEO uses its
/// stored navigation state with the available fast clock correction.
pub mod source;

/// Store SBAS corrections and navigation state partitioned by source GEO.
///
/// [`SbasCorrectionStore::ingest`] retains masks, fast and long-term
/// corrections, GEO navigation, ionospheric grids, and withdrawal or disable
/// state. Freshness policies govern correction lookups; ionospheric delay uses
/// a 350-kilometer pierce-point shell with four-point or three-point
/// interpolation, and broadcast PRNs 120 through 158 map to SBAS slots 20
/// through 58.
pub mod store;

pub use format::{parse_ems_lines, parse_rtklib_lines, SbasLogBlock};
pub use message::{
    SbasBlock, SbasDoNotUse, SbasFastCorrections, SbasFastDegradation, SbasGeoAlmanac, SbasGeoNav,
    SbasIgpDelay, SbasIgpMask, SbasIntegrity, SbasIonoDelays, SbasLongTermCorrections,
    SbasLongTermHalf, SbasLongTermRecord, SbasMessage, SbasMessageType, SbasMixedCorrections,
    SbasMixedFastCorrections, SbasNetworkTime, SbasPrnMask, SbasUnsupported, SbasWireForm,
    SpareBits,
};
pub use source::{
    IssueAwareBroadcast, SbasCorrectedEphemeris, SbasCorrectedEphemerisOwned, SbasSolveMode,
};
pub use store::{
    give_variance_m2_for_givei, sat_to_sbas_prn, sbas_prn_to_sat, udre_variance_m2_for_udrei,
    SbasCorrectionStore, SbasFastCorrection, SbasGeoState, SbasIgp, SbasIonoGrid,
    SbasLongTermCorrection, SBAS_GIVE_VARIANCE_M2, SBAS_UDRE_VARIANCE_M2,
};
