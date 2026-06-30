//! Most-significant-bit-first bit reader and writer for RTCM 3 message bodies.
//!
//! RTCM 3 packs every data field as a run of bits, most-significant bit first,
//! with no byte alignment between fields (RTCM 10403.x, Section 3.2). Three field
//! encodings appear in the standard and all three are supported here:
//!
//!   * `uint(n)` - an unsigned integer ([`BitReader::u`] / [`BitWriter::push_u`]).
//!   * `int(n)`  - a two's-complement signed integer ([`BitReader::i`] /
//!     [`BitWriter::push_i`]).
//!   * `intS(n)` - a sign-and-magnitude signed integer where the most
//!     significant bit is the sign and the remaining `n - 1` bits the magnitude
//!     ([`BitReader::ism`] / [`BitWriter::push_ism`]). The GLONASS ephemeris
//!     (message 1020) uses this representation for its orbit terms.
//!
//! The writer pads the final byte with zero bits, which is exactly how RTCM
//! byte-aligns a message body before the CRC, so a decode followed by an encode
//! reproduces the original payload bytes.

use crate::error::{Error, Result};

/// A forgiving MSB-first reader over a borrowed RTCM message body.
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap a message body, positioned at its first bit.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    /// Bits not yet consumed.
    pub(crate) fn remaining_bits(&self) -> usize {
        self.bytes.len() * 8 - self.bit_pos
    }

    /// Read `n` bits (`n <= 64`) as an unsigned integer, MSB first.
    pub(crate) fn u(&mut self, n: usize) -> Result<u64> {
        debug_assert!(n <= 64);
        if self.remaining_bits() < n {
            return Err(Error::Parse(format!(
                "RTCM body truncated: need {n} more bits, {} remain",
                self.remaining_bits()
            )));
        }
        let mut acc: u64 = 0;
        for _ in 0..n {
            let byte = self.bytes[self.bit_pos / 8];
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            acc = (acc << 1) | u64::from(bit);
            self.bit_pos += 1;
        }
        Ok(acc)
    }

    /// Read a single bit as a boolean flag.
    pub(crate) fn flag(&mut self) -> Result<bool> {
        Ok(self.u(1)? != 0)
    }

    /// Read `n` bits (`n < 64`) as a two's-complement signed integer.
    pub(crate) fn i(&mut self, n: usize) -> Result<i64> {
        debug_assert!(n < 64);
        let raw = self.u(n)?;
        let sign_bit = 1u64 << (n - 1);
        if raw & sign_bit != 0 {
            Ok(raw as i64 - (1i64 << n))
        } else {
            Ok(raw as i64)
        }
    }

    /// Read `n` bits (`n < 64`) as a sign-and-magnitude signed integer: the
    /// leading bit is the sign (1 = negative), the remaining `n - 1` bits the
    /// magnitude.
    pub(crate) fn ism(&mut self, n: usize) -> Result<i64> {
        debug_assert!((1..64).contains(&n));
        let negative = self.flag()?;
        let magnitude = self.u(n - 1)? as i64;
        Ok(if negative { -magnitude } else { magnitude })
    }
}

/// An MSB-first bit accumulator that produces an RTCM message body.
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    nbits: usize,
}

impl BitWriter {
    /// A new empty writer.
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            nbits: 0,
        }
    }

    fn push_bit(&mut self, bit: u8) {
        let byte_index = self.nbits / 8;
        if byte_index >= self.bytes.len() {
            self.bytes.push(0);
        }
        if bit != 0 {
            self.bytes[byte_index] |= 1 << (7 - (self.nbits % 8));
        }
        self.nbits += 1;
    }

    /// Append `n` bits (`n <= 64`) of `value`, MSB first.
    pub(crate) fn push_u(&mut self, value: u64, n: usize) {
        debug_assert!(n <= 64);
        for k in (0..n).rev() {
            self.push_bit(((value >> k) & 1) as u8);
        }
    }

    /// Append a single flag bit.
    pub(crate) fn push_flag(&mut self, value: bool) {
        self.push_bit(u8::from(value));
    }

    /// Append `n` bits (`n < 64`) of `value` as a two's-complement signed field.
    pub(crate) fn push_i(&mut self, value: i64, n: usize) {
        debug_assert!(n < 64);
        let mask = (1u64 << n) - 1;
        self.push_u((value as u64) & mask, n);
    }

    /// Append `n` bits (`n < 64`) of `value` as a sign-and-magnitude field.
    pub(crate) fn push_ism(&mut self, value: i64, n: usize) {
        debug_assert!((1..64).contains(&n));
        self.push_flag(value < 0);
        self.push_u(value.unsigned_abs(), n - 1);
    }

    /// Consume the writer, returning the byte-aligned body (the final partial
    /// byte is zero-padded, matching RTCM's pre-CRC alignment).
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
