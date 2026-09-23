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
//!     ([`BitReader::ism_signed_zero`] / [`FieldWriter::ism`]). The GLONASS
//!     ephemeris (message 1020) uses this representation for its orbit terms.
//!
//! The writer pads the final byte with zero bits, which is exactly how RTCM
//! byte-aligns a message body before the CRC, so a decode followed by an encode
//! reproduces the original payload bytes. The RTCM encoders write through
//! [`FieldWriter`], which refuses a value wider than its field.

use core::fmt;

use crate::error::Error;

/// Internal typed error for exhausting an RTCM message-body bit reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutOfInput {
    requested_bits: usize,
    remaining_bits: usize,
}

impl OutOfInput {
    fn new(requested_bits: usize, remaining_bits: usize) -> Self {
        Self {
            requested_bits,
            remaining_bits,
        }
    }
}

impl fmt::Display for OutOfInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RTCM body truncated: need {} more bits, {} remain",
            self.requested_bits, self.remaining_bits
        )
    }
}

impl From<OutOfInput> for Error {
    fn from(error: OutOfInput) -> Self {
        Error::Parse(error.to_string())
    }
}

/// A forgiving MSB-first reader over a borrowed RTCM message body.
#[derive(Clone)]
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
    pub(crate) fn u(&mut self, n: usize) -> std::result::Result<u64, OutOfInput> {
        debug_assert!(n <= 64);
        if self.remaining_bits() < n {
            return Err(OutOfInput::new(n, self.remaining_bits()));
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
    pub(crate) fn flag(&mut self) -> std::result::Result<bool, OutOfInput> {
        Ok(self.u(1)? != 0)
    }

    /// Read `n` bits (`n < 64`) as a two's-complement signed integer.
    pub(crate) fn i(&mut self, n: usize) -> std::result::Result<i64, OutOfInput> {
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
    /// magnitude. The decoders read through [`Self::ism_signed_zero`], which
    /// also keeps a negative zero's sign.
    #[cfg(test)]
    pub(crate) fn ism(&mut self, n: usize) -> std::result::Result<i64, OutOfInput> {
        debug_assert!((1..64).contains(&n));
        let negative = self.flag()?;
        let magnitude = self.u(n - 1)? as i64;
        Ok(if negative { -magnitude } else { magnitude })
    }

    /// Read `n` bits (`n < 64`) as a sign-and-magnitude signed integer, and say
    /// whether it was spelled as negative zero (sign set, magnitude zero); the
    /// value of a negative zero is `0`.
    pub(crate) fn ism_signed_zero(
        &mut self,
        n: usize,
    ) -> std::result::Result<(i64, bool), OutOfInput> {
        debug_assert!((1..64).contains(&n));
        let negative = self.flag()?;
        let magnitude = self.u(n - 1)? as i64;
        Ok(if negative {
            (-magnitude, magnitude == 0)
        } else {
            (magnitude, false)
        })
    }

    /// Consume and return every bit not yet read.
    pub(crate) fn rest(&mut self) -> Vec<bool> {
        let mut bits = Vec::with_capacity(self.remaining_bits());
        while self.bit_pos < self.bytes.len() * 8 {
            let byte = self.bytes[self.bit_pos / 8];
            bits.push((byte >> (7 - (self.bit_pos % 8))) & 1 == 1);
            self.bit_pos += 1;
        }
        bits
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

    /// Append `n` bits (`n < 64`) of `value` as a sign-and-magnitude field. The
    /// encoders write through [`FieldWriter::ism`], which checks the width.
    #[cfg(test)]
    pub(crate) fn push_ism(&mut self, value: i64, n: usize) {
        debug_assert!((1..64).contains(&n));
        self.push_flag(value < 0);
        self.push_u(value.unsigned_abs(), n - 1);
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> usize {
        self.nbits
    }

    /// Consume the writer, returning the byte-aligned body (the final partial
    /// byte is zero-padded, matching RTCM's pre-CRC alignment).
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// An RTCM body writer that refuses a value its field cannot hold.
///
/// [`BitWriter`] keeps the low `n` bits of whatever it is given. The RTCM
/// encoders write through this type instead, so a value wider than its field
/// is refused with an [`Error::InvalidInput`] naming the message, the field,
/// the value and the field's range, rather than written as other bits.
pub(crate) struct FieldWriter {
    writer: BitWriter,
    message_number: u16,
}

impl FieldWriter {
    /// A writer for a body of message `message_number`, which it names in
    /// every refusal. The message number itself is not written.
    pub(crate) fn new(message_number: u16) -> Self {
        Self {
            writer: BitWriter::new(),
            message_number,
        }
    }

    fn refuse(&self, field: impl fmt::Display, value: i128, width: usize, range: &str) -> Error {
        Error::InvalidInput(format!(
            "RTCM {} {field} {value} does not fit its {width}-bit {range}",
            self.message_number
        ))
    }

    /// Write `value` as an unsigned `width`-bit field (`width <= 64`).
    pub(crate) fn u(
        &mut self,
        field: impl fmt::Display,
        value: u64,
        width: usize,
    ) -> Result<(), Error> {
        debug_assert!(width <= 64);
        if width < 64 && value >> width != 0 {
            let widest = (1u64 << width) - 1;
            return Err(self.refuse(
                field,
                i128::from(value),
                width,
                &format!("unsigned field (0..={widest})"),
            ));
        }
        self.writer.push_u(value, width);
        Ok(())
    }

    /// Write `value` as a two's-complement `width`-bit field (`width < 64`).
    pub(crate) fn i(
        &mut self,
        field: impl fmt::Display,
        value: i64,
        width: usize,
    ) -> Result<(), Error> {
        debug_assert!((1..64).contains(&width));
        let low = -(1i64 << (width - 1));
        let high = (1i64 << (width - 1)) - 1;
        if !(low..=high).contains(&value) {
            return Err(self.refuse(
                field,
                i128::from(value),
                width,
                &format!("two's-complement field ({low}..={high})"),
            ));
        }
        self.writer.push_i(value, width);
        Ok(())
    }

    /// Write `value` as a sign-and-magnitude `width`-bit field (`width < 64`).
    ///
    /// `negative_zero` writes zero with its sign bit set, the second spelling
    /// of zero the representation has; it is refused with a nonzero value,
    /// whose sign the value itself states.
    pub(crate) fn ism(
        &mut self,
        field: impl fmt::Display,
        value: i64,
        width: usize,
        negative_zero: bool,
    ) -> Result<(), Error> {
        debug_assert!((2..64).contains(&width));
        let widest = (1u64 << (width - 1)) - 1;
        if value.unsigned_abs() > widest {
            return Err(self.refuse(
                field,
                i128::from(value),
                width,
                &format!("sign-magnitude field (-{widest}..={widest})"),
            ));
        }
        if negative_zero && value != 0 {
            return Err(Error::InvalidInput(format!(
                "RTCM {} {field} is marked negative zero but holds {value}",
                self.message_number
            )));
        }
        self.writer.push_flag(value < 0 || negative_zero);
        self.writer.push_u(value.unsigned_abs(), width - 1);
        Ok(())
    }

    /// Write one flag bit.
    pub(crate) fn flag(&mut self, value: bool) {
        self.writer.push_flag(value);
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> usize {
        self.writer.bit_len()
    }

    /// The message number this writer names in its refusals.
    pub(crate) fn message_number(&self) -> u16 {
        self.message_number
    }

    /// Consume the writer, returning the zero-padded, byte-aligned body.
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.writer.into_bytes()
    }
}
