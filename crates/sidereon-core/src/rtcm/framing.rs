//! RTCM 3 transport framing: preamble sync, length, and CRC-24Q verification.
//!
//! Every RTCM 3 message is wrapped in the transport frame defined in Section 4:
//!
//! ```text
//! +----------+------------------+------------------+-------------------+
//! | 0xD3 (8) | reserved(6) len(10) | message body (len bytes) | CRC-24Q (24) |
//! +----------+------------------+------------------+-------------------+
//! ```
//!
//! The 8-bit preamble is `0xD3`, the six bits after it are reserved (zero), and
//! the next ten bits give the body length in bytes (0..=1023). The trailing
//! 24-bit CRC-24Q covers the preamble, the length word, and the body. The body
//! itself is the input to [`crate::rtcm::Message::decode`].
//!
//! The scanner ([`FrameScanner`]) is forgiving in the sans-I/O sense: it slides
//! over an arbitrary byte buffer, resynchronizes on the next `0xD3` whenever the
//! length runs past the buffer or the CRC fails, and yields only frames whose
//! CRC verifies. This is how a real receiver locks onto a noisy serial stream.

use crate::error::{Error, Result};

use super::crc::crc24q;

/// The RTCM 3 frame preamble byte.
pub const PREAMBLE: u8 = 0xD3;
/// Maximum message-body length, in bytes (the length field is 10 bits).
pub const MAX_BODY_LEN: usize = 0x3FF;
/// Overhead a frame adds around its body: 3 header bytes plus 3 CRC bytes.
pub const FRAME_OVERHEAD: usize = 6;

/// A single decoded frame: the borrowed message body plus the full frame size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedFrame<'a> {
    /// The message body (the bytes between the length word and the CRC).
    pub body: &'a [u8],
    /// Total length of the frame in bytes, including preamble, length, and CRC.
    pub frame_len: usize,
}

/// Wrap a message body in an RTCM 3 transport frame with a fresh CRC-24Q.
///
/// Returns [`Error::InvalidInput`] if the body exceeds [`MAX_BODY_LEN`].
pub fn encode_frame(body: &[u8]) -> Result<Vec<u8>> {
    if body.len() > MAX_BODY_LEN {
        return Err(Error::InvalidInput(format!(
            "RTCM body of {} bytes exceeds the 1023-byte frame limit",
            body.len()
        )));
    }
    let len = body.len() as u16;
    let mut out = Vec::with_capacity(body.len() + FRAME_OVERHEAD);
    out.push(PREAMBLE);
    // Six reserved bits (zero) followed by the high two bits of the length.
    out.push((len >> 8) as u8 & 0x03);
    out.push((len & 0xFF) as u8);
    out.extend_from_slice(body);
    let crc = crc24q(&out);
    out.push((crc >> 16) as u8);
    out.push((crc >> 8) as u8);
    out.push(crc as u8);
    Ok(out)
}

/// Decode the single frame that begins at the start of `bytes`.
///
/// Verifies the preamble and the CRC-24Q. Returns [`Error::Parse`] if the
/// preamble is missing, the buffer is shorter than the declared frame, or the
/// CRC does not match.
pub fn decode_frame(bytes: &[u8]) -> Result<DecodedFrame<'_>> {
    if bytes.len() < FRAME_OVERHEAD {
        return Err(Error::Parse(format!(
            "RTCM frame too short: {} bytes, need at least {FRAME_OVERHEAD}",
            bytes.len()
        )));
    }
    if bytes[0] != PREAMBLE {
        return Err(Error::Parse(format!(
            "RTCM frame missing 0xD3 preamble (found {:#04x})",
            bytes[0]
        )));
    }
    let len = ((usize::from(bytes[1] & 0x03)) << 8) | usize::from(bytes[2]);
    let total = 3 + len + 3;
    if bytes.len() < total {
        return Err(Error::Parse(format!(
            "RTCM frame truncated: declares {len}-byte body needing {total} bytes, have {}",
            bytes.len()
        )));
    }
    let crc_computed = crc24q(&bytes[..3 + len]);
    let crc_framed = (u32::from(bytes[3 + len]) << 16)
        | (u32::from(bytes[3 + len + 1]) << 8)
        | u32::from(bytes[3 + len + 2]);
    if crc_computed != crc_framed {
        return Err(Error::Parse(format!(
            "RTCM CRC-24Q mismatch: computed {crc_computed:#08x}, frame carries {crc_framed:#08x}"
        )));
    }
    Ok(DecodedFrame {
        body: &bytes[3..3 + len],
        frame_len: total,
    })
}

/// A forgiving iterator that yields every CRC-valid frame in a byte buffer.
///
/// Bytes that are not a valid frame start (a stray `0xD3`, a truncated tail, or
/// a body whose CRC fails) are skipped one at a time, exactly as a hardware
/// receiver resynchronizes on a serial stream.
pub struct FrameScanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FrameScanner<'a> {
    /// Begin scanning `bytes` from the start.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

impl<'a> Iterator for FrameScanner<'a> {
    type Item = DecodedFrame<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] != PREAMBLE {
                self.pos += 1;
                continue;
            }
            match decode_frame(&self.bytes[self.pos..]) {
                Ok(frame) => {
                    self.pos += frame.frame_len;
                    return Some(frame);
                }
                Err(_) => {
                    // Not a real frame here; slide past this 0xD3 and resync.
                    self.pos += 1;
                }
            }
        }
        None
    }
}
