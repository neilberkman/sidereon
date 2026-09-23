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
//! many 8-bit characters (DF030, DF033, DF228, DF230, DF232). Each byte is read
//! as the character with that code point (`U+0000`..=`U+00FF`, ISO 8859-1), so
//! every byte value is kept, and written back as that one byte. The counts are
//! reconstructed from the string lengths on encode, so the body round-trips.

use crate::error::{Error, Result};

use super::bits::{BitReader, FieldWriter};
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};

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
    /// Every body bit after the last field, the zeros that align the body to a
    /// byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros: read under [`RtcmPolicy::Lenient`] and written back
    /// after the last field by `encode_with_policy` under that policy, so the
    /// body re-encodes byte for byte. Empty when the bits after the last field
    /// are fewer than eight zeros, for every body read under
    /// [`RtcmPolicy::Strict`], and for a message built by hand; `encode`
    /// refuses a nonempty value. A tail set by hand is zero-padded to the byte
    /// when written and reads back with that padding.
    pub trailing_bits: Vec<bool>,
}

impl AntennaDescriptor {
    /// Decode a 1007 / 1008 / 1033 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`]: bits after the last field other than the zero
    /// byte alignment are refused.
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if !matches!(message_number, 1007 | 1008 | 1033) {
            return Err(Error::Parse(format!(
                "message {message_number} is not an antenna descriptor 1007/1008/1033"
            ))
            .into());
        }
        let reference_station_id = r.u(12)? as u16;
        let antenna_descriptor = read_string(r)?;
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
            trailing_bits: Vec::new(),
        };

        if matches!(message_number, 1008 | 1033) {
            descriptor.antenna_serial_number = Some(read_string(r)?);
        }
        if message_number == 1033 {
            descriptor.receiver_type = Some(read_string(r)?);
            descriptor.receiver_firmware_version = Some(read_string(r)?);
            descriptor.receiver_serial_number = Some(read_string(r)?);
        }

        Ok(descriptor)
    }

    /// Encode this descriptor body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the field when `message_number` is not
    /// 1007, 1008 or 1033; when a string the message carries is absent (the
    /// antenna serial number of 1008 and 1033, the three receiver strings of
    /// 1033) or one it does not carry is present (it would be dropped); when
    /// the reference station ID is wider than 12 bits; or when a string holds
    /// a character above `U+00FF`, which no 8-bit character states, or more
    /// than the 255 characters its 8-bit count states.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported as an
    /// [`RtcmDeparture::TrailingBits`]; every other refusal of `encode` applies
    /// under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        let (serial, receiver) = match number {
            1007 => (false, false),
            1008 => (true, false),
            1033 => (true, true),
            _ => {
                return Err(Error::InvalidInput(format!(
                    "RTCM message number {number} is not an antenna descriptor 1007/1008/1033"
                )))
            }
        };
        let presence = [
            ("antenna serial number", &self.antenna_serial_number, serial),
            ("receiver type", &self.receiver_type, receiver),
            (
                "receiver firmware version",
                &self.receiver_firmware_version,
                receiver,
            ),
            (
                "receiver serial number",
                &self.receiver_serial_number,
                receiver,
            ),
        ];
        for (field, value, carried) in presence {
            match (value.is_some(), carried) {
                (true, false) => {
                    return Err(Error::InvalidInput(format!(
                        "RTCM {number} carries no {field}, and one is given"
                    )))
                }
                (false, true) => {
                    return Err(Error::InvalidInput(format!(
                        "RTCM {number} carries the {field}, and none is given"
                    )))
                }
                _ => {}
            }
        }
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        write_string(&mut w, "antenna descriptor", &self.antenna_descriptor)?;
        w.u("antenna setup ID", u64::from(self.antenna_setup_id), 8)?;
        for (field, value, _) in presence {
            if let Some(value) = value {
                write_string(&mut w, field, value)?;
            }
        }
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

/// Read an 8-bit-counted run of 8-bit characters as a string, each byte as the
/// character with that code point.
fn read_string(r: &mut BitReader<'_>) -> DecodeResult<String> {
    let count = r.u(8)? as usize;
    let mut s = String::with_capacity(count);
    for _ in 0..count {
        s.push(char::from(r.u(8)? as u8));
    }
    Ok(s)
}

/// Write a string as an 8-bit count followed by one 8-bit character per
/// character, refusing a character above `U+00FF` or more than 255 of them.
fn write_string(w: &mut FieldWriter, field: &str, s: &str) -> Result<()> {
    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        let byte = u8::try_from(u32::from(c)).map_err(|_| {
            Error::InvalidInput(format!(
                "RTCM {field} character {c:?} (U+{:04X}) is not an 8-bit character",
                u32::from(c)
            ))
        })?;
        bytes.push(byte);
    }
    w.u(
        format_args!("{field} character count"),
        bytes.len() as u64,
        8,
    )?;
    for byte in bytes {
        w.u(field, u64::from(byte), 8)?;
    }
    Ok(())
}

impl super::TrailingBits for AntennaDescriptor {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
