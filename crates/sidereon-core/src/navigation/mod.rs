//! GNSS navigation-message synthesis and decoding.
//!
//! These are bit-level codecs for the data streams modulated onto the GNSS
//! signals (as opposed to [`crate::broadcast`], which evaluates an already
//! decoded navigation message into an orbit and clock). Currently this covers
//! the GPS L1 C/A legacy navigation (LNAV) message.

pub mod lnav;
