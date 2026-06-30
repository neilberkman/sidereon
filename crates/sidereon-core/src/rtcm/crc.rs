//! CRC-24Q, the cyclic redundancy check that guards every RTCM 3 frame.
//!
//! RTCM 3 (Section 4.2) appends a 24-bit CRC computed with the Qualcomm
//! "CRC-24Q" polynomial over the frame's preamble, length, and message body. The
//! generator is
//!
//! ```text
//! x^24 + x^23 + x^18 + x^17 + x^14 + x^11 + x^10 + x^7 + x^6 + x^5 + x^4 + x^3 + x + 1
//! ```
//!
//! which is `0x1864CFB`. For RTCM the register starts at zero, bits are fed
//! most-significant first, and there is no final inversion or reflection. The
//! same polynomial and bit mechanics, started from `0xB704CE` instead of zero,
//! is the "CRC-24/OPENPGP" definition whose published check value over the ASCII
//! string `123456789` is `0x21CF02`; the tests use that to anchor the algorithm.

/// The CRC-24Q generator polynomial (without the implicit `x^24` term).
const POLYNOMIAL: u32 = 0x0186_4CFB;
const TOP_BIT: u32 = 0x0100_0000;
const MASK_24: u32 = 0x00FF_FFFF;

/// Compute the CRC-24Q of `data` as RTCM specifies it (initial register zero).
pub(crate) fn crc24q(data: &[u8]) -> u32 {
    crc24q_with_init(0, data)
}

/// Compute the CRC-24Q of `data` from an arbitrary initial register value.
///
/// RTCM uses an initial value of zero ([`crc24q`]); the parameter exists so the
/// tests can reproduce the published "CRC-24/OPENPGP" check value, which uses
/// the same polynomial from an initial value of `0xB704CE`.
pub(crate) fn crc24q_with_init(init: u32, data: &[u8]) -> u32 {
    let mut crc: u32 = init & MASK_24;
    for &byte in data {
        crc ^= u32::from(byte) << 16;
        for _ in 0..8 {
            crc <<= 1;
            if crc & TOP_BIT != 0 {
                crc ^= POLYNOMIAL;
            }
        }
    }
    crc & MASK_24
}
