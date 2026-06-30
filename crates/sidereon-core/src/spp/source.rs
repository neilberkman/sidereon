//! Ephemeris-source abstraction for SPP.

use crate::id::GnssSatelliteId;
use crate::sp3::Sp3;

/// A source of satellite position and clock at a transmit epoch.
///
/// The SPP pipeline is written against this trait rather than a concrete product
/// so it can run on either a precise SP3 ephemeris or a broadcast navigation
/// message. The contract is exactly what the transmit-time iteration needs: the
/// ECEF position (meters) and the satellite clock offset (seconds) at a given
/// J2000 second, or `None` if the source has no usable ephemeris for that
/// satellite at that instant.
pub trait EphemerisSource {
    /// ECEF position (m) and satellite clock offset (s) for `sat` at `t_j2000_s`.
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)>;
}

impl EphemerisSource for Sp3 {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.position_at_j2000_seconds(sat, t_j2000_s).ok()?;
        let clk = state.clock_s?;
        Some((state.position.as_array(), clk))
    }
}
