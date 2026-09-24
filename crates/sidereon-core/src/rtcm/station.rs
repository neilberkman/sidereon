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

use super::bits::{BitReader, FieldWriter};
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};
use super::{RtcmEncodeError, RtcmRecordKind};

/// ECEF reference-point scale: each integer step is 0.0001 m.
const ECEF_SCALE_M: f64 = 0.0001;

/// A decoded message 1005 or 1006 antenna reference point.
#[derive(Clone, Debug, PartialEq, Eq)]
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

    /// Decode a 1005 / 1006 body (without the transport frame) under
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
        if message_number != 1005 && message_number != 1006 {
            return Err(Error::Parse(format!(
                "message {message_number} is not station coordinates 1005/1006"
            ))
            .into());
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
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this station coordinate message body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming the field when `message_number` is not
    /// 1005 or 1006, when `antenna_height` is absent from a 1006 or present in
    /// a 1005 (the height would be left out of the body, or written where the
    /// decoder reads none), or when a value is wider than its field: the
    /// 12-bit station ID, the 6-bit ITRF year, the 2-bit quarter-cycle
    /// indicator, or an ECEF component outside the 38-bit two's-complement
    /// range.
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
        match (number, self.antenna_height) {
            (1005, None) | (1006, Some(_)) => {}
            (1005, Some(_)) | (1006, None) => {
                return Err(RtcmEncodeError::FieldPresence {
                    message_number: number,
                    record: RtcmRecordKind::StationCoordinates,
                    field: "antenna height",
                    carried: number == 1006,
                }
                .into())
            }
            _ => {
                return Err(RtcmEncodeError::MessageNumber {
                    message_number: number,
                    record: RtcmRecordKind::StationCoordinates,
                }
                .into())
            }
        }
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.u(
            "ITRF realization year",
            u64::from(self.itrf_realization_year),
            6,
        )?;
        w.flag(self.gps_indicator);
        w.flag(self.glonass_indicator);
        w.flag(self.galileo_indicator);
        w.flag(self.reference_station_indicator);
        w.i("ECEF X", self.ecef_x, 38)?;
        w.flag(self.single_receiver_oscillator);
        w.flag(self.reserved);
        w.i("ECEF Y", self.ecef_y, 38)?;
        w.u(
            "quarter-cycle indicator",
            u64::from(self.quarter_cycle_indicator),
            2,
        )?;
        w.i("ECEF Z", self.ecef_z, 38)?;
        if let Some(height) = self.antenna_height {
            w.u("antenna height", u64::from(height), 16)?;
        }
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

impl super::TrailingBits for StationCoordinates {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
