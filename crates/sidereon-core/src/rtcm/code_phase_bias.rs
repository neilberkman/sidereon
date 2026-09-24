//! RTCM 3 message 1230: GLONASS L1 and L2 code-phase biases.
//!
//! A reference station states, for each of the four GLONASS FDMA signals whose
//! bit is set in the signal mask (DF422), the bias between its code and phase
//! observations (RTCM 10403.3 Table 3.5-80): L1 C/A (DF423), L1 P (DF424), L2
//! C/A (DF425) and L2 P (DF426), each an int16 at 0.02 m.
//!
//! The biases are stored as their raw transmitted integers, in mask order, so
//! the body round-trips byte for byte; a signal whose mask bit is clear has no
//! value.

use crate::error::{Error, Result};

use super::bits::{BitReader, FieldWriter};
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};

/// A DF423..DF426 code-phase bias of `-2^15` is not valid, as RTKLIB
/// `decode_type1230` tests it.
pub const GLONASS_CODE_PHASE_BIAS_INVALID: i16 = i16::MIN;

/// Scale of DF423..DF426, metres per unit.
const BIAS_SCALE_M: f64 = 0.02;

/// A decoded message 1230, GLONASS L1 and L2 code-phase biases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlonassCodePhaseBiases {
    /// Reference station ID (DF003).
    pub reference_station_id: u16,
    /// GLONASS code-phase bias indicator (DF421): whether the station's code
    /// and phase observations are aligned (true) or not.
    pub aligned: bool,
    /// The three reserved bits after DF421 (DF001), kept for the round trip.
    pub reserved: u8,
    /// L1 C/A code-phase bias (DF423), present when mask bit 3 is set.
    pub l1_ca: Option<i16>,
    /// L1 P code-phase bias (DF424), present when mask bit 2 is set.
    pub l1_p: Option<i16>,
    /// L2 C/A code-phase bias (DF425), present when mask bit 1 is set.
    pub l2_ca: Option<i16>,
    /// L2 P code-phase bias (DF426), present when mask bit 0 is set.
    pub l2_p: Option<i16>,
    /// Every body bit after the last field, the zeros that align the body to a
    /// byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros: read under [`RtcmPolicy::Lenient`] and written back
    /// after the last field by `encode_with_policy` under that policy, so the
    /// body re-encodes byte for byte. Empty when the bits after the last field
    /// are fewer than eight zeros, for every body read under
    /// [`RtcmPolicy::Strict`], and for a message built by hand; `encode`
    /// refuses a nonempty value.
    pub trailing_bits: Vec<bool>,
}

impl GlonassCodePhaseBiases {
    /// The FDMA signal mask (DF422) the present biases state: bit 3 L1 C/A,
    /// bit 2 L1 P, bit 1 L2 C/A, bit 0 L2 P.
    pub fn signal_mask(&self) -> u8 {
        [self.l1_ca, self.l1_p, self.l2_ca, self.l2_p]
            .iter()
            .enumerate()
            .filter(|(_, bias)| bias.is_some())
            .fold(0, |mask, (index, _)| mask | (1 << (3 - index)))
    }

    /// The biases in metres, in the order L1 C/A, L1 P, L2 C/A, L2 P: `None`
    /// for a signal the message does not carry or whose value is
    /// [`GLONASS_CODE_PHASE_BIAS_INVALID`].
    pub fn biases_m(&self) -> [Option<f64>; 4] {
        [self.l1_ca, self.l1_p, self.l2_ca, self.l2_p].map(|bias| {
            bias.filter(|&raw| raw != GLONASS_CODE_PHASE_BIAS_INVALID)
                .map(|raw| f64::from(raw) * BIAS_SCALE_M)
        })
    }

    /// Decode a 1230 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`]: bits after the last field other than the zero
    /// byte alignment are refused.
    ///
    /// A mask with no bit set is a complete message with no bias. RTKLIB
    /// `decode_type1230` refuses such a body as short (it asks for more bits
    /// than the header and a first bias); RTCM 10403.3 gives it no bias
    /// fields, and it is read here.
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if message_number != 1230 {
            return Err(Error::Parse(format!(
                "message {message_number} is not GLONASS code-phase biases 1230"
            ))
            .into());
        }
        let reference_station_id = r.u(12)? as u16;
        let aligned = r.flag()?;
        let reserved = r.u(3)? as u8;
        let mask = r.u(4)? as u8;
        let mut biases = [None; 4];
        for (index, bias) in biases.iter_mut().enumerate() {
            if mask & (1 << (3 - index)) != 0 {
                *bias = Some(r.i(16)? as i16);
            }
        }
        let [l1_ca, l1_p, l2_ca, l2_p] = biases;
        Ok(Self {
            reference_station_id,
            aligned,
            reserved,
            l1_ca,
            l1_p,
            l2_ca,
            l2_p,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this message body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the field when a value is wider than its
    /// field: the 12-bit station ID or the 3-bit reserved field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported as an
    /// [`RtcmDeparture::TrailingBits`]; every other refusal of `encode` applies
    /// under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let mut w = FieldWriter::new(1230);
        w.u("message number", 1230, 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.flag(self.aligned);
        w.u("reserved", u64::from(self.reserved), 3)?;
        w.u("signal mask", u64::from(self.signal_mask()), 4)?;
        for (name, bias) in [
            ("L1 C/A code-phase bias", self.l1_ca),
            ("L1 P code-phase bias", self.l1_p),
            ("L2 C/A code-phase bias", self.l2_ca),
            ("L2 P code-phase bias", self.l2_p),
        ] {
            if let Some(bias) = bias {
                w.i(name, i64::from(bias), 16)?;
            }
        }
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

impl super::TrailingBits for GlonassCodePhaseBiases {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
