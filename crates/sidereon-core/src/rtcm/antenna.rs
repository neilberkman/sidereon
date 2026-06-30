//! RTCM 3 antenna and receiver descriptor messages 1007, 1008, and 1033.
//!
//! These messages carry the human-readable equipment strings a receiver needs
//! to apply the correct antenna calibration (RTCM 10403.3 Tables 3.5-11,
//! 3.5-12, 3.5-31):
//!
//!   * **1007** - antenna descriptor and setup id.
//!   * **1008** - 1007 plus the antenna serial number.
//!   * **1033** - 1008 plus the receiver type, firmware version, and serial
//!     number.
//!
//! Each string is length-prefixed by an 8-bit character count followed by that
//! many 8-bit characters (DF030, DF033, DF228, DF230, DF232). The counts are
//! reconstructed from the string lengths on encode, so the body round-trips.

use crate::error::{Error, Result};

use super::bits::{BitReader, BitWriter};

/// A decoded antenna / receiver descriptor message (1007, 1008, or 1033).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AntennaDescriptor {
    /// 1007, 1008, or 1033.
    pub message_number: u16,
    /// Reference station identifier (DF003).
    pub reference_station_id: u16,
    /// Antenna descriptor string (DF030).
    pub antenna_descriptor: String,
    /// Antenna setup id (DF031).
    pub antenna_setup_id: u8,
    /// Antenna serial number (DF033). Present for 1008 and 1033.
    pub antenna_serial_number: Option<String>,
    /// Receiver type descriptor (DF228). Present for 1033.
    pub receiver_type: Option<String>,
    /// Receiver firmware version (DF230). Present for 1033.
    pub receiver_firmware_version: Option<String>,
    /// Receiver serial number (DF232). Present for 1033.
    pub receiver_serial_number: Option<String>,
}

impl AntennaDescriptor {
    /// Decode a 1007 / 1008 / 1033 body (without the transport frame).
    pub fn decode(body: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if !matches!(message_number, 1007 | 1008 | 1033) {
            return Err(Error::Parse(format!(
                "message {message_number} is not an antenna descriptor 1007/1008/1033"
            )));
        }
        let reference_station_id = r.u(12)? as u16;
        let antenna_descriptor = read_string(&mut r)?;
        let antenna_setup_id = r.u(8)? as u8;

        let mut descriptor = Self {
            message_number,
            reference_station_id,
            antenna_descriptor,
            antenna_setup_id,
            antenna_serial_number: None,
            receiver_type: None,
            receiver_firmware_version: None,
            receiver_serial_number: None,
        };

        if matches!(message_number, 1008 | 1033) {
            descriptor.antenna_serial_number = Some(read_string(&mut r)?);
        }
        if message_number == 1033 {
            descriptor.receiver_type = Some(read_string(&mut r)?);
            descriptor.receiver_firmware_version = Some(read_string(&mut r)?);
            descriptor.receiver_serial_number = Some(read_string(&mut r)?);
        }

        Ok(descriptor)
    }

    /// Encode this descriptor body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(u64::from(self.message_number), 12);
        w.push_u(u64::from(self.reference_station_id), 12);
        write_string(&mut w, &self.antenna_descriptor);
        w.push_u(u64::from(self.antenna_setup_id), 8);
        if matches!(self.message_number, 1008 | 1033) {
            write_string(&mut w, self.antenna_serial_number.as_deref().unwrap_or(""));
        }
        if self.message_number == 1033 {
            write_string(&mut w, self.receiver_type.as_deref().unwrap_or(""));
            write_string(
                &mut w,
                self.receiver_firmware_version.as_deref().unwrap_or(""),
            );
            write_string(&mut w, self.receiver_serial_number.as_deref().unwrap_or(""));
        }
        w.into_bytes()
    }
}

/// Read an 8-bit-counted run of 8-bit characters as a string.
fn read_string(r: &mut BitReader<'_>) -> Result<String> {
    let count = r.u(8)? as usize;
    let mut s = String::with_capacity(count);
    for _ in 0..count {
        s.push(r.u(8)? as u8 as char);
    }
    Ok(s)
}

/// Write a string as an 8-bit count followed by its 8-bit characters.
fn write_string(w: &mut BitWriter, s: &str) {
    let bytes = s.as_bytes();
    w.push_u(bytes.len() as u64, 8);
    for &b in bytes {
        w.push_u(u64::from(b), 8);
    }
}
