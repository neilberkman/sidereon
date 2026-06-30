//! RTCM 3 stationary antenna reference point messages 1005 and 1006.
//!
//! Message 1005 (RTCM 10403.3 Table 3.5-9) gives the Earth-centred,
//! Earth-fixed (ECEF) coordinates of a reference station's antenna reference
//! point. Message 1006 (Table 3.5-10) is identical but appends the antenna
//! height above the marker. Both carry the ECEF components as 38-bit
//! two's-complement integers in units of 0.0001 m, and the height (1006) as an
//! unsigned 16-bit integer in the same unit.
//!
//! The coordinates are stored as their raw transmitted integers so the body
//! round-trips byte-for-byte; the [`StationCoordinates::x_m`] family converts to
//! meters.

use crate::error::{Error, Result};

use super::bits::{BitReader, BitWriter};

/// ECEF reference-point scale: each integer step is 0.0001 m.
const ECEF_SCALE_M: f64 = 0.0001;

/// A decoded message 1005 or 1006 antenna reference point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StationCoordinates {
    /// 1005 or 1006.
    pub message_number: u16,
    /// Reference station identifier (DF003).
    pub reference_station_id: u16,
    /// ITRF realization year (DF021, 6 bits).
    pub itrf_realization_year: u8,
    /// GPS service supported at this station (DF022).
    pub gps_indicator: bool,
    /// GLONASS service supported (DF023).
    pub glonass_indicator: bool,
    /// Galileo service supported (DF024).
    pub galileo_indicator: bool,
    /// Reference-station indicator (DF141): physical vs non-physical station.
    pub reference_station_indicator: bool,
    /// Antenna reference point ECEF X (DF025), raw integer of 0.0001 m steps.
    pub ecef_x: i64,
    /// Single receiver oscillator indicator (DF142).
    pub single_receiver_oscillator: bool,
    /// Reserved field DF001 (1 bit), preserved for exact round-trip.
    pub reserved: bool,
    /// Antenna reference point ECEF Y (DF026), raw integer of 0.0001 m steps.
    pub ecef_y: i64,
    /// Quarter-cycle indicator (DF364, 2 bits).
    pub quarter_cycle_indicator: u8,
    /// Antenna reference point ECEF Z (DF027), raw integer of 0.0001 m steps.
    pub ecef_z: i64,
    /// Antenna height above the marker (DF028), raw integer of 0.0001 m steps.
    /// Present only for message 1006.
    pub antenna_height: Option<u16>,
}

impl StationCoordinates {
    /// ECEF X in meters.
    pub fn x_m(&self) -> f64 {
        self.ecef_x as f64 * ECEF_SCALE_M
    }

    /// ECEF Y in meters.
    pub fn y_m(&self) -> f64 {
        self.ecef_y as f64 * ECEF_SCALE_M
    }

    /// ECEF Z in meters.
    pub fn z_m(&self) -> f64 {
        self.ecef_z as f64 * ECEF_SCALE_M
    }

    /// Antenna height in meters, if this is a 1006 message.
    pub fn antenna_height_m(&self) -> Option<f64> {
        self.antenna_height.map(|h| f64::from(h) * ECEF_SCALE_M)
    }

    /// Decode a 1005 / 1006 body (without the transport frame).
    pub fn decode(body: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1005 && message_number != 1006 {
            return Err(Error::Parse(format!(
                "message {message_number} is not station coordinates 1005/1006"
            )));
        }
        let reference_station_id = r.u(12)? as u16;
        let itrf_realization_year = r.u(6)? as u8;
        let gps_indicator = r.flag()?;
        let glonass_indicator = r.flag()?;
        let galileo_indicator = r.flag()?;
        let reference_station_indicator = r.flag()?;
        let ecef_x = r.i(38)?;
        let single_receiver_oscillator = r.flag()?;
        let reserved = r.flag()?;
        let ecef_y = r.i(38)?;
        let quarter_cycle_indicator = r.u(2)? as u8;
        let ecef_z = r.i(38)?;
        let antenna_height = if message_number == 1006 {
            Some(r.u(16)? as u16)
        } else {
            None
        };

        Ok(Self {
            message_number,
            reference_station_id,
            itrf_realization_year,
            gps_indicator,
            glonass_indicator,
            galileo_indicator,
            reference_station_indicator,
            ecef_x,
            single_receiver_oscillator,
            reserved,
            ecef_y,
            quarter_cycle_indicator,
            ecef_z,
            antenna_height,
        })
    }

    /// Encode this station coordinate message body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(u64::from(self.message_number), 12);
        w.push_u(u64::from(self.reference_station_id), 12);
        w.push_u(u64::from(self.itrf_realization_year), 6);
        w.push_flag(self.gps_indicator);
        w.push_flag(self.glonass_indicator);
        w.push_flag(self.galileo_indicator);
        w.push_flag(self.reference_station_indicator);
        w.push_i(self.ecef_x, 38);
        w.push_flag(self.single_receiver_oscillator);
        w.push_flag(self.reserved);
        w.push_i(self.ecef_y, 38);
        w.push_u(u64::from(self.quarter_cycle_indicator), 2);
        w.push_i(self.ecef_z, 38);
        if let Some(height) = self.antenna_height {
            w.push_u(u64::from(height), 16);
        }
        w.into_bytes()
    }
}
