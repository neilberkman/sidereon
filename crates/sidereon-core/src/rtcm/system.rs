//! RTCM 3 message 1013 (system parameters) and message 1029 (Unicode text
//! string), RTCM 10403.3 Tables 3.5-16, 3.5-17 and 3.5-47.
//!
//! Every field is stored as its raw transmitted integer, so a decode followed
//! by an encode reproduces the body bit for bit.

use crate::error::{Error, Result};

use super::bits::{BitReader, FieldWriter};
use super::network::{check_count, read_counted, write_tail};
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};

/// One message announcement of a 1013 message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageAnnouncement {
    /// Announced message number (DF055, 12 bits).
    pub message_number: u16,
    /// Message sync flag (DF056): whether the message is sent synchronously.
    pub synchronous: bool,
    /// Message transmission interval (DF057, 16 bits), scale 0.1 s.
    pub interval: u16,
}

/// A decoded message 1013, system parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemParameters {
    /// Reference station ID (DF003, 12 bits).
    pub reference_station_id: u16,
    /// Modified Julian day number (DF051, 16 bits).
    pub mjd: u16,
    /// Seconds of day, UTC (DF052, 17 bits).
    pub seconds_of_day: u32,
    /// Number of message announcements the header states (DF053, 5 bits), as
    /// transmitted.
    pub announcement_count: u8,
    /// Leap seconds, GPS - UTC (DF054, 8 bits).
    pub leap_seconds: u8,
    /// The message announcements, in transmitted order.
    pub announcements: Vec<MessageAnnouncement>,
    /// Every body bit after the last record kept for the round trip: the bits
    /// of a record cut short under [`RtcmPolicy::Lenient`]
    /// ([`RtcmDeparture::RecordsShort`]), or bits after the records other than
    /// the zero byte alignment ([`RtcmDeparture::TrailingBits`]). Empty for
    /// every body read under [`RtcmPolicy::Strict`].
    pub trailing_bits: Vec<bool>,
}

impl SystemParameters {
    /// Decode a 1013 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body, &mut DecodeContext::new(RtcmPolicy::Strict)).map_err(Into::into)
    }

    /// Decode a 1013 body under `policy`, returning the departures read under
    /// [`RtcmPolicy::Lenient`].
    pub fn decode_with_policy(
        body: &[u8],
        policy: RtcmPolicy,
    ) -> Result<(Self, Vec<RtcmDeparture>)> {
        let mut ctx = DecodeContext::new(policy);
        let message = Self::decode_inner(body, &mut ctx)?;
        Ok((message, ctx.into_departures()))
    }

    pub(crate) fn decode_inner(body: &[u8], ctx: &mut DecodeContext) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1013 {
            return Err(Error::Parse(format!(
                "message {message_number} is not system parameters 1013"
            ))
            .into());
        }
        let reference_station_id = r.u(12)? as u16;
        let mjd = r.u(16)? as u16;
        let seconds_of_day = r.u(17)? as u32;
        let announcement_count = r.u(5)? as u8;
        let leap_seconds = r.u(8)? as u8;
        let (announcements, trailing_bits) =
            read_counted(&mut r, ctx, 1013, usize::from(announcement_count), |r| {
                Ok(MessageAnnouncement {
                    message_number: r.u(12)? as u16,
                    synchronous: r.flag()?,
                    interval: r.u(16)? as u16,
                })
            })?;
        Ok(Self {
            reference_station_id,
            mjd,
            seconds_of_day,
            announcement_count,
            leap_seconds,
            announcements,
            trailing_bits,
        })
    }

    /// Encode this body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming what the message cannot state: a header
    /// count other than the number of announcements, nonempty
    /// `trailing_bits`, or a value wider than its field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] a header
    /// count above the number of announcements and nonempty `trailing_bits`
    /// are written and reported; every other refusal of [`Self::encode`]
    /// applies under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let mut departures = Vec::new();
        let short = check_count(
            1013,
            usize::from(self.announcement_count),
            self.announcements.len(),
            policy,
            &mut departures,
        )?;
        let mut w = FieldWriter::new(1013);
        w.u("message number", 1013, 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.u("MJD", u64::from(self.mjd), 16)?;
        w.u("seconds of day", u64::from(self.seconds_of_day), 17)?;
        w.u("announcement count", u64::from(self.announcement_count), 5)?;
        w.u("leap seconds", u64::from(self.leap_seconds), 8)?;
        for (index, a) in self.announcements.iter().enumerate() {
            w.u(
                format_args!("announcement {index} message number"),
                u64::from(a.message_number),
                12,
            )?;
            w.flag(a.synchronous);
            w.u(
                format_args!("announcement {index} interval"),
                u64::from(a.interval),
                16,
            )?;
        }
        write_tail(&mut w, short, &self.trailing_bits, policy, &mut departures)?;
        Ok((w.into_bytes(), departures))
    }
}

/// A decoded message 1029, a Unicode text string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextMessage {
    /// Reference station ID (DF003, 12 bits).
    pub reference_station_id: u16,
    /// Modified Julian day number (DF051, 16 bits).
    pub mjd: u16,
    /// Seconds of day, UTC (DF052, 17 bits).
    pub seconds_of_day: u32,
    /// Number of characters to follow (DF138, 7 bits), as transmitted. It
    /// counts characters, not code units, and is kept apart from the text.
    pub character_count: u8,
    /// The UTF-8 code units (DF140), as transmitted; their number is the
    /// 8-bit DF139. See [`Self::text`].
    pub code_units: Vec<u8>,
    /// Every body bit after the last field kept for the round trip; see
    /// [`RtcmDeparture::TrailingBits`]. Empty for every body read under
    /// [`RtcmPolicy::Strict`] and for a message built by hand.
    pub trailing_bits: Vec<bool>,
}

impl TextMessage {
    /// The text, when the code units are UTF-8.
    pub fn text(&self) -> std::result::Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.code_units)
    }

    /// Decode a 1029 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`]. The text is the DF139 count of UTF-8 code
    /// units. RTKLIB `decode_type1029` reads the DF138 count of bytes instead,
    /// which cuts a text with multi-byte characters short.
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if message_number != 1029 {
            return Err(Error::Parse(format!(
                "message {message_number} is not a text message 1029"
            ))
            .into());
        }
        let reference_station_id = r.u(12)? as u16;
        let mjd = r.u(16)? as u16;
        let seconds_of_day = r.u(17)? as u32;
        let character_count = r.u(7)? as u8;
        let units = r.u(8)? as usize;
        let mut code_units = Vec::with_capacity(units);
        for _ in 0..units {
            code_units.push(r.u(8)? as u8);
        }
        Ok(Self {
            reference_station_id,
            mjd,
            seconds_of_day,
            character_count,
            code_units,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the field when a value is wider than its
    /// field (more than 255 code units, a character count above 127), or for
    /// nonempty `trailing_bits`.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let mut w = FieldWriter::new(1029);
        w.u("message number", 1029, 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.u("MJD", u64::from(self.mjd), 16)?;
        w.u("seconds of day", u64::from(self.seconds_of_day), 17)?;
        w.u("character count", u64::from(self.character_count), 7)?;
        w.u("code unit count", self.code_units.len() as u64, 8)?;
        for &unit in &self.code_units {
            w.u("code unit", u64::from(unit), 8)?;
        }
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

impl super::TrailingBits for TextMessage {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
