//! Galileo HAS MT1 correction payload parser.
//!
//! This module parses the recovered HAS Message Type 1 payload, starting at the
//! 32-bit MT1 header and ending at the byte-aligned MT1 body. E6 page recovery,
//! Reed-Solomon reconstruction, and outer transport framing are intentionally
//! outside this module.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::constants::{C_M_S, F_E1_HZ, F_E5A_HZ, F_L1_HZ, F_L2_HZ};
use crate::error::{Error, Result};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::rtcm::bits::{BitReader, BitWriter};

const HAS_ORBIT_RADIAL_SCALE_M: f64 = 0.0025;
const HAS_ORBIT_ALONG_CROSS_SCALE_M: f64 = 0.0080;
const HAS_CLOCK_SCALE_M: f64 = 0.0025;
const HAS_CODE_BIAS_SCALE_M: f64 = 0.02;
const HAS_PHASE_BIAS_SCALE_CYCLES: f64 = 0.01;

/// HAS orbit delta radial invalid / not available sentinel (-4096 = -(1 << 12)).
pub const HAS_ORBIT_RADIAL_INVALID: i16 = -(1 << 12);

/// HAS orbit delta along-track and cross-track invalid / not available sentinel (-2048 = -(1 << 11)).
pub const HAS_ORBIT_ALONG_CROSS_INVALID: i16 = -(1 << 11);

/// HAS clock delta data unavailable sentinel (-4096 = -(1 << 12)) per Galileo HAS SIS ICD Table 31 / Table 34.
pub const HAS_CLOCK_INVALID: i16 = -(1 << 12);

/// HAS clock delta satellite shall not be used sentinel (+4095 = (1 << 12) - 1) per Galileo HAS SIS ICD Table 31 / Table 34.
pub const HAS_CLOCK_DO_NOT_USE: i16 = (1 << 12) - 1;

/// HAS code-bias invalid / not available sentinel (-1024 = -(1 << 10)).
pub const HAS_CODE_BIAS_INVALID: i16 = -(1 << 10);

/// HAS phase-bias invalid / not available sentinel (-1024 = -(1 << 10)).
pub const HAS_PHASE_BIAS_INVALID: i16 = -(1 << 10);

/// HAS MT1 header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HasMt1Header {
    /// Time of hour in GST seconds.
    pub toh_s: u16,
    /// Mask content block present.
    pub mask: bool,
    /// Orbit correction content block present.
    pub orbit: bool,
    /// Clock full-set content block present.
    pub clock_full_set: bool,
    /// Clock subset content block present.
    pub clock_subset: bool,
    /// Code-bias content block present.
    pub code_bias: bool,
    /// Phase-bias content block present.
    pub phase_bias: bool,
    /// Reserved 4-bit field.
    pub reserved: u8,
    /// Mask identifier.
    pub mask_id: u8,
    /// IOD set identifier.
    pub iod_set_id: u8,
}

/// One GNSS mask inside an HAS MT1 mask block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HasGnssMask {
    /// Corrected constellation.
    pub system: GnssSystem,
    /// Corrected PRNs in HAS mask order.
    pub satellites: Vec<u8>,
    /// Corrected signal indices in HAS mask order.
    pub signals: Vec<u8>,
    /// Optional row-major satellite by signal cell mask.
    pub cell_mask: Option<Vec<bool>>,
    /// HAS navigation-message index.
    pub nav_message: u8,
}

/// HAS MT1 mask block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HasMaskBlock {
    /// GNSS-specific masks in transmitted order.
    pub systems: Vec<HasGnssMask>,
    /// Reserved bits after all GNSS masks per Galileo HAS SIS ICD Table 15.
    pub reserved: u8,
}

/// HAS MT1 orbit correction block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasOrbitBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Orbit records in mask order, with unavailable corrections represented as None.
    pub records: Vec<HasOrbitCorrection>,
}

/// HAS orbit correction for one satellite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasOrbitCorrection {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// Reference navigation issue.
    pub iode: u32,
    /// Delta radial, meters, additive per HAS ICD, or None if unavailable.
    pub radial_m: Option<f64>,
    /// Delta in-track, meters, additive per HAS ICD, or None if unavailable.
    pub along_m: Option<f64>,
    /// Delta cross-track, meters, additive per HAS ICD, or None if unavailable.
    pub cross_m: Option<f64>,
}

/// HAS per-GNSS clock metadata record per Galileo HAS SIS ICD Table 28 / Table 33.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HasClockSystem {
    /// GNSS constellation system identifier.
    pub system: GnssSystem,
    /// Delta clock multiplier index (0..=3), scaling HAS_CLOCK_SCALE_M by 1, 2, 3, or 4 per Galileo HAS SIS ICD Table 29.
    pub multiplier_index: u8,
}

/// HAS MT1 clock full-set or subset correction block per Galileo HAS SIS ICD Section 5.2.3 and Section 5.2.4.
#[derive(Clone, Debug, PartialEq)]
pub struct HasClockBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Per-system clock metadata records in transmitted GNSS order.
    pub systems: Vec<HasClockSystem>,
    /// Clock records grouped in transmitted systems order and within each system's mask satellite order, with unavailable corrections represented as None.
    pub records: Vec<HasClockCorrection>,
}

/// HAS clock correction for one satellite per Galileo HAS SIS ICD Section 5.2.3.2 and Section 5.2.4.1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasClockCorrection {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// Delta clock in meters, additive per HAS ICD, or None if unavailable or satellite shall not be used.
    pub correction_m: Option<f64>,
    /// Whether the satellite shall not be used (Galileo HAS SIS ICD Table 31 / Table 34 sentinel +4095).
    ///
    /// Valid states are:
    /// - `correction_m: Some(m), do_not_use: false`: an available correction.
    /// - `correction_m: None, do_not_use: false`: data unavailable (wire signed13 -4096).
    /// - `correction_m: None, do_not_use: true`: satellite shall not be used (wire signed13 +4095).
    ///
    /// `correction_m: Some(_), do_not_use: true` is contradictory: the encoder refuses it, and
    /// [`SsrCorrectionStore::ingest_has_mt1`](crate::ssr::SsrCorrectionStore::ingest_has_mt1)
    /// reads it as do-not-use and removes the satellite's stored clock correction.
    pub do_not_use: bool,
}

/// HAS MT1 code-bias block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasCodeBiasBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Code-bias records in mask cell order, with unavailable biases represented as None.
    pub records: Vec<HasCodeBias>,
}

/// HAS code bias for one satellite signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasCodeBias {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// HAS signal index.
    pub signal_id: u8,
    /// Bias in meters, or None if unavailable. Add to the pseudorange observation per HAS ICD.
    pub bias_m: Option<f64>,
}

/// HAS MT1 phase-bias block.
#[derive(Clone, Debug, PartialEq)]
pub struct HasPhaseBiasBlock {
    /// Validity interval index.
    pub validity_interval: u8,
    /// Phase-bias records in mask cell order, with unavailable biases represented as None.
    pub records: Vec<HasPhaseBias>,
}

/// HAS phase bias for one satellite signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasPhaseBias {
    /// Corrected satellite.
    pub sat: GnssSatelliteId,
    /// HAS signal index.
    pub signal_id: u8,
    /// Bias in cycles, or None if unavailable.
    pub bias_cycles: Option<f64>,
    /// Phase discontinuity indicator.
    pub discontinuity_indicator: u8,
}

/// Result of converting phase bias cycles to meters using carrier frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HasPhaseBiasConversion {
    /// Phase bias converted to meters with carrier frequency.
    Available {
        /// Phase bias converted to meters.
        bias_m: f64,
        /// Carrier frequency in hertz.
        frequency_hz: f64,
    },
    /// Transmitted as unavailable (sentinel).
    TransmittedUnavailable,
    /// Carrier frequency is unknown or unassigned for this satellite/signal.
    UnknownSignal,
    /// Numerical input is non-finite or invalid.
    InvalidInput,
}

impl HasPhaseBias {
    /// Get the conversion status and derived meters/frequency for this phase bias.
    pub fn conversion(&self) -> HasPhaseBiasConversion {
        let Some(cycles) = self.bias_cycles else {
            return HasPhaseBiasConversion::TransmittedUnavailable;
        };
        if !cycles.is_finite() {
            return HasPhaseBiasConversion::InvalidInput;
        }
        let Ok(frequency_hz) = has_signal_frequency_hz(self.sat.system, self.signal_id) else {
            return HasPhaseBiasConversion::UnknownSignal;
        };
        let wavelength_m = C_M_S / frequency_hz;
        let bias_m = cycles * wavelength_m;
        if !bias_m.is_finite() {
            return HasPhaseBiasConversion::InvalidInput;
        }
        HasPhaseBiasConversion::Available {
            bias_m,
            frequency_hz,
        }
    }

    /// Derived phase bias in meters, or None if unavailable, unknown signal, or invalid input.
    pub fn bias_m(&self) -> Option<f64> {
        match self.conversion() {
            HasPhaseBiasConversion::Available { bias_m, .. } => Some(bias_m),
            _ => None,
        }
    }
}

/// Decoded HAS MT1 payload.
#[derive(Clone, Debug, PartialEq)]
pub struct HasMt1Message {
    /// MT1 header.
    pub header: HasMt1Header,
    /// Mask block, when present.
    pub mask: Option<HasMaskBlock>,
    /// Orbit correction block, when present.
    pub orbit: Option<HasOrbitBlock>,
    /// Clock full-set correction block, when present.
    pub clock_full_set: Option<HasClockBlock>,
    /// Clock subset correction block, when present.
    pub clock_subset: Option<HasClockBlock>,
    /// Code-bias block, when present.
    pub code_bias: Option<HasCodeBiasBlock>,
    /// Phase-bias block, when present.
    pub phase_bias: Option<HasPhaseBiasBlock>,
    /// Remaining non-decoded padding or future-extension bits.
    pub padding_bits: Vec<bool>,
}

/// Immutable context carrying a validated HAS MT1 mask definition.
///
/// Per Galileo HAS SIS ICD Issue 1.0 May 2022 §§5.1, 5.2.1, 7.6, 7.6.1, and 7.7
/// (<https://www.gsc-europa.eu/sites/default/files/sites/all/files/Galileo-HAS-SIS-ICD_in_force.pdf>),
/// correction-bearing HAS MT1 messages that omit an inline mask block (e.g. high-rate
/// clock corrections) associate with a prior transmitted mask definition through matching
/// `mask_id` and `iod_set_id` identifiers.
///
/// # Provenance and Invariants
///
/// `HasMt1Context` has no public unchecked constructor and cannot be derived from
/// arbitrary caller-constructed structs. It is produced exclusively by successful
/// decoding of an HAS MT1 payload that contains an inline mask block (`header.mask = true`),
/// guaranteeing that the mask has passed all bit-level validation and structural consistency
/// checks. All fields are private and accessible only via read-only getters.
///
/// # Caller Responsibility
///
/// This codec provides raw sans-I/O decoding and encoding using explicit caller-selected
/// prior definitions. Callers are responsible for maintaining the correct prior definition,
/// tracking reference epochs, managing lifetime and validity intervals, enforcing IOD rules,
/// and handling potential identifier rollover (5-bit `mask_id` / `iod_set_id` modulo 32).
/// The codec does not implement an automatic stream cache, guessed TTLs, or temporal heuristics;
/// context is derived exclusively from successful complete inline decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HasMt1Context {
    mask_id: u8,
    iod_set_id: u8,
    mask: HasMaskBlock,
}

impl HasMt1Context {
    /// Mask identifier (5-bit, 0..=31).
    pub fn mask_id(&self) -> u8 {
        self.mask_id
    }

    /// IOD set identifier (5-bit, 0..=31).
    pub fn iod_set_id(&self) -> u8 {
        self.iod_set_id
    }

    /// Read-only reference to the validated mask block.
    pub fn mask(&self) -> &HasMaskBlock {
        &self.mask
    }
}

impl HasMt1Message {
    /// Decode one recovered HAS MT1 payload.
    pub fn decode(body: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(body);
        let header = read_header(&mut r)?;
        let mask = if header.mask {
            let m = read_mask_block(&mut r)?;
            validate_mask_block_structural(&m).map_err(|e| Error::Parse(e.to_string()))?;
            Some(m)
        } else {
            None
        };
        let Some(mask_ref) = mask.as_ref() else {
            if header.orbit
                || header.clock_full_set
                || header.clock_subset
                || header.code_bias
                || header.phase_bias
            {
                return Err(Error::Parse(
                    "HAS MT1 correction blocks require a mask in this stateless decoder"
                        .to_string(),
                ));
            }
            return Ok(Self {
                header,
                mask,
                orbit: None,
                clock_full_set: None,
                clock_subset: None,
                code_bias: None,
                phase_bias: None,
                padding_bits: read_padding_bits(&mut r)?,
            });
        };

        let orbit = if header.orbit {
            Some(read_orbit_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let clock_full_set = if header.clock_full_set {
            Some(read_clock_full_set_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let clock_subset = if header.clock_subset {
            Some(read_clock_subset_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let code_bias = if header.code_bias {
            Some(read_code_bias_block(&mut r, mask_ref)?)
        } else {
            None
        };
        let phase_bias = if header.phase_bias {
            Some(read_phase_bias_block(&mut r, mask_ref)?)
        } else {
            None
        };

        Ok(Self {
            header,
            mask,
            orbit,
            clock_full_set,
            clock_subset,
            code_bias,
            phase_bias,
            padding_bits: read_padding_bits(&mut r)?,
        })
    }

    /// Decode one recovered HAS MT1 payload using an optional prior mask context.
    ///
    /// Per Galileo HAS SIS ICD Issue 1.0 May 2022 §§5.1, 5.2.1, 7.6, 7.6.1, and 7.7,
    /// maskless correction-bearing messages associate with a prior mask definition via
    /// matching `(mask_id, iod_set_id)` pairs.
    ///
    /// # Behavior
    ///
    /// - Inline mask present (`header.mask = true`): The message is decoded using its own
    ///   wire definition. Any supplied prior `context` is ignored (it is neither required nor
    ///   does a mismatch cause refusal). Upon successful parsing of the entire message and its
    ///   padding, a new immutable [`HasMt1Context`] is returned as `Some(context)`. If any
    ///   trailing block or bit is malformed, decoding fails and no context is yielded.
    /// - Maskless message without corrections: Decodes with `mask = None` and yields `None`.
    /// - Maskless message with corrections (`header.mask = false`): Requires a matching
    ///   prior `context`. If `context` is `None`, decoding fails with a named error
    ///   (`"missing prior inline-mask context for HAS MT1 correction blocks"`).
    ///   If `context` is provided, its `(mask_id, iod_set_id)` must exactly match the message
    ///   header; any mismatch is rejected. The decoded message preserves `header.mask = false`
    ///   and `mask = None`, and yields `None` (maskless messages do not yield new contexts).
    ///
    /// # Caller Responsibility
    ///
    /// Caller is responsible for matching reference epochs, validity intervals, and IOD rules.
    /// No automatic TTL, LRU caching, or rollover inference is performed.
    pub fn decode_with_context(
        body: &[u8],
        context: Option<&HasMt1Context>,
    ) -> Result<(Self, Option<HasMt1Context>)> {
        let mut r = BitReader::new(body);
        let header = read_header(&mut r)?;
        let (mask, yielded_context) = if header.mask {
            let inline_mask = read_mask_block(&mut r)?;
            validate_mask_block_structural(&inline_mask)
                .map_err(|e| Error::Parse(e.to_string()))?;
            let ctx = HasMt1Context {
                mask_id: header.mask_id,
                iod_set_id: header.iod_set_id,
                mask: inline_mask.clone(),
            };
            (Some(inline_mask), Some(ctx))
        } else {
            (None, None)
        };

        let active_mask = match mask.as_ref() {
            Some(inline_mask) => inline_mask,
            None => {
                let has_corrections = header.orbit
                    || header.clock_full_set
                    || header.clock_subset
                    || header.code_bias
                    || header.phase_bias;
                if !has_corrections {
                    return Ok((
                        Self {
                            header,
                            mask: None,
                            orbit: None,
                            clock_full_set: None,
                            clock_subset: None,
                            code_bias: None,
                            phase_bias: None,
                            padding_bits: read_padding_bits(&mut r)?,
                        },
                        None,
                    ));
                }
                let Some(ctx) = context else {
                    return Err(Error::Parse(
                        "missing prior inline-mask context for HAS MT1 correction blocks"
                            .to_string(),
                    ));
                };
                if header.mask_id != ctx.mask_id || header.iod_set_id != ctx.iod_set_id {
                    return Err(Error::Parse(format!(
                        "HAS MT1 context ID mismatch: message has mask_id {} iod_set_id {}, context has mask_id {} iod_set_id {}",
                        header.mask_id, header.iod_set_id, ctx.mask_id, ctx.iod_set_id
                    )));
                }
                validate_mask_block_structural(ctx.mask())
                    .map_err(|e| Error::Parse(e.to_string()))?;
                ctx.mask()
            }
        };

        let orbit = if header.orbit {
            Some(read_orbit_block(&mut r, active_mask)?)
        } else {
            None
        };
        let clock_full_set = if header.clock_full_set {
            Some(read_clock_full_set_block(&mut r, active_mask)?)
        } else {
            None
        };
        let clock_subset = if header.clock_subset {
            Some(read_clock_subset_block(&mut r, active_mask)?)
        } else {
            None
        };
        let code_bias = if header.code_bias {
            Some(read_code_bias_block(&mut r, active_mask)?)
        } else {
            None
        };
        let phase_bias = if header.phase_bias {
            Some(read_phase_bias_block(&mut r, active_mask)?)
        } else {
            None
        };

        let padding_bits = read_padding_bits(&mut r)?;

        Ok((
            Self {
                header,
                mask,
                orbit,
                clock_full_set,
                clock_subset,
                code_bias,
                phase_bias,
                padding_bits,
            },
            yielded_context,
        ))
    }

    /// Encode this message back to an MT1 payload.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when the message cannot be written as the MT1
    /// payload it describes:
    ///
    /// - a header flag disagrees with the presence of its block, or a
    ///   correction block is present without an inline mask (maskless
    ///   correction messages are written by
    ///   [`HasMt1Message::encode_with_context`]);
    /// - a header field is wider than its wire field: TOH above 3599 s,
    ///   reserved above 15, mask ID or IOD set ID above 31;
    /// - the mask block cannot state its lists exactly: no systems or more than
    ///   15, a system HAS gives no GNSS ID or a system named twice, satellites
    ///   outside `1..=40` (HAS SIS ICD Table 19) or signals outside `0..=15`, a
    ///   satellite or signal list that is not strictly ascending (the masks
    ///   state sets in ascending order, and every correction block is written
    ///   in that order), a cell mask whose length is not satellites times
    ///   signals, a navigation message index above 7, or reserved bits above
    ///   63;
    /// - a correction block with no record for a mask satellite or cell (a
    ///   clock subset block holds records only for the satellites it selects),
    ///   two records for one, or a record the mask does not declare, or whose
    ///   validity interval index is reserved. Records may be held in any
    ///   order; each is written at its mask position;
    /// - a correction that is not finite or whose scaled value falls outside
    ///   the range its field leaves beside the unavailable and do-not-use
    ///   sentinels, an IODE or discontinuity indicator wider than its field,
    ///   clock metadata that is missing for a mask system, names one system
    ///   twice, names a system the mask does not, or uses a reserved
    ///   multiplier index, or a clock record that carries a correction while
    ///   marked do-not-use. Full-set metadata may be held in any order; each
    ///   entry is written at its system's mask position.
    ///
    /// Each of those would otherwise be written as a frame that decodes to a
    /// different message, or to none.
    ///
    /// # Padding
    ///
    /// `padding_bits` are written after the last block. When they do not end
    /// on a byte boundary the payload is zero-filled to the next one, so
    /// decoding the result returns those padding bits followed by the fill.
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.header.mask != self.mask.is_some() {
            return Err(Error::InvalidInput(
                "HAS header mask flag does not match mask presence".to_string(),
            ));
        }
        if self.header.orbit != self.orbit.is_some() {
            return Err(Error::InvalidInput(
                "HAS header orbit flag does not match orbit block presence".to_string(),
            ));
        }
        if self.header.clock_full_set != self.clock_full_set.is_some() {
            return Err(Error::InvalidInput(
                "HAS header clock full set flag does not match clock full set presence".to_string(),
            ));
        }
        if self.header.clock_subset != self.clock_subset.is_some() {
            return Err(Error::InvalidInput(
                "HAS header clock subset flag does not match clock subset presence".to_string(),
            ));
        }
        if self.header.code_bias != self.code_bias.is_some() {
            return Err(Error::InvalidInput(
                "HAS header code bias flag does not match code bias presence".to_string(),
            ));
        }
        if self.header.phase_bias != self.phase_bias.is_some() {
            return Err(Error::InvalidInput(
                "HAS header phase bias flag does not match phase bias presence".to_string(),
            ));
        }

        let Some(mask) = &self.mask else {
            if self.orbit.is_some()
                || self.clock_full_set.is_some()
                || self.clock_subset.is_some()
                || self.code_bias.is_some()
                || self.phase_bias.is_some()
            {
                return Err(Error::InvalidInput(
                    "HAS MT1 correction blocks require a mask".to_string(),
                ));
            }
            let mut w = BitWriter::new();
            write_header(&mut w, self.header)?;
            for &bit in &self.padding_bits {
                w.push_flag(bit);
            }
            return Ok(w.into_bytes());
        };

        validate_mask_block_structural(mask)?;
        let mut w = BitWriter::new();
        write_header(&mut w, self.header)?;
        write_mask_block(&mut w, mask)?;
        if let Some(orbit) = &self.orbit {
            write_orbit_block(&mut w, mask, orbit)?;
        }
        if let Some(clock) = &self.clock_full_set {
            write_clock_full_set_block(&mut w, mask, clock)?;
        }
        if let Some(clock) = &self.clock_subset {
            write_clock_subset_block(&mut w, mask, clock)?;
        }
        if let Some(code_bias) = &self.code_bias {
            write_code_bias_block(&mut w, mask, code_bias)?;
        }
        if let Some(phase_bias) = &self.phase_bias {
            write_phase_bias_block(&mut w, mask, phase_bias)?;
        }
        for &bit in &self.padding_bits {
            w.push_flag(bit);
        }
        Ok(w.into_bytes())
    }

    /// Encode this message back to an MT1 payload using an optional prior mask context.
    ///
    /// Per Galileo HAS SIS ICD Issue 1.0 May 2022 §§5.1, 5.2.1, 7.6, 7.6.1, and 7.7,
    /// maskless correction-bearing messages require a prior mask definition matching
    /// `(mask_id, iod_set_id)` to serialize satellite and cell-specific corrections.
    ///
    /// # Behavior
    ///
    /// - Inline mask present (`header.mask = true`): Encodes the message with its inline
    ///   mask block. Any supplied `context` is ignored, consistent with `decode_with_context`.
    /// - Maskless message without corrections: Encodes header and padding bits with no mask.
    /// - Maskless message with corrections (`header.mask = false`, `mask = None`):
    ///   Requires a matching prior `context`. If `context` is `None`, encoding fails with a
    ///   named error (`"missing prior inline-mask context for HAS MT1 correction blocks"`).
    ///   If `context` is provided, its `(mask_id, iod_set_id)` must exactly match the message
    ///   header; any mismatch is rejected. The resulting wire bytes omit the mask block,
    ///   preserving `header.mask = false` without injecting a cached mask onto the wire.
    ///
    /// # Caller Responsibility
    ///
    /// Caller is responsible for selecting the matching prior mask context and reference epoch.
    ///
    /// # Padding
    ///
    /// `padding_bits` are written after the last block. When they do not end
    /// on a byte boundary the payload is zero-filled to the next one, so
    /// decoding the result returns those padding bits followed by the fill.
    pub fn encode_with_context(&self, context: Option<&HasMt1Context>) -> Result<Vec<u8>> {
        if self.header.mask != self.mask.is_some() {
            return Err(Error::InvalidInput(
                "HAS header mask flag does not match mask presence".to_string(),
            ));
        }
        if self.header.orbit != self.orbit.is_some() {
            return Err(Error::InvalidInput(
                "HAS header orbit flag does not match orbit block presence".to_string(),
            ));
        }
        if self.header.clock_full_set != self.clock_full_set.is_some() {
            return Err(Error::InvalidInput(
                "HAS header clock full set flag does not match clock full set presence".to_string(),
            ));
        }
        if self.header.clock_subset != self.clock_subset.is_some() {
            return Err(Error::InvalidInput(
                "HAS header clock subset flag does not match clock subset presence".to_string(),
            ));
        }
        if self.header.code_bias != self.code_bias.is_some() {
            return Err(Error::InvalidInput(
                "HAS header code bias flag does not match code bias presence".to_string(),
            ));
        }
        if self.header.phase_bias != self.phase_bias.is_some() {
            return Err(Error::InvalidInput(
                "HAS header phase bias flag does not match phase bias presence".to_string(),
            ));
        }

        let mask: &HasMaskBlock = if let Some(inline_mask) = &self.mask {
            // When inline mask is present, message uses its own wire definition.
            // Any supplied prior context is ignored, consistent with decode_with_context.
            validate_mask_block_structural(inline_mask)?;
            inline_mask
        } else {
            let has_corrections = self.orbit.is_some()
                || self.clock_full_set.is_some()
                || self.clock_subset.is_some()
                || self.code_bias.is_some()
                || self.phase_bias.is_some();
            if !has_corrections {
                let mut w = BitWriter::new();
                write_header(&mut w, self.header)?;
                for &bit in &self.padding_bits {
                    w.push_flag(bit);
                }
                return Ok(w.into_bytes());
            }
            let Some(ctx) = context else {
                return Err(Error::InvalidInput(
                    "missing prior inline-mask context for HAS MT1 correction blocks".to_string(),
                ));
            };
            if self.header.mask_id != ctx.mask_id || self.header.iod_set_id != ctx.iod_set_id {
                return Err(Error::InvalidInput(format!(
                    "HAS MT1 context ID mismatch: message has mask_id {} iod_set_id {}, context has mask_id {} iod_set_id {}",
                    self.header.mask_id, self.header.iod_set_id, ctx.mask_id, ctx.iod_set_id
                )));
            }
            validate_mask_block_structural(ctx.mask())?;
            ctx.mask()
        };

        let mut w = BitWriter::new();
        write_header(&mut w, self.header)?;
        if self.header.mask {
            write_mask_block(&mut w, mask)?;
        }
        if let Some(orbit) = &self.orbit {
            write_orbit_block(&mut w, mask, orbit)?;
        }
        if let Some(clock) = &self.clock_full_set {
            write_clock_full_set_block(&mut w, mask, clock)?;
        }
        if let Some(clock) = &self.clock_subset {
            write_clock_subset_block(&mut w, mask, clock)?;
        }
        if let Some(code_bias) = &self.code_bias {
            write_code_bias_block(&mut w, mask, code_bias)?;
        }
        if let Some(phase_bias) = &self.phase_bias {
            write_phase_bias_block(&mut w, mask, phase_bias)?;
        }
        for &bit in &self.padding_bits {
            w.push_flag(bit);
        }
        Ok(w.into_bytes())
    }
}

/// Resolve an HAS MT1 TOH to a J2000 epoch using the GST reception time.
pub fn has_mt1_reference_j2000_s(
    reception_gst: crate::astro::time::model::GnssWeekTow,
    toh_s: u16,
) -> Result<f64> {
    if reception_gst.system != crate::astro::time::model::TimeScale::Gst {
        return Err(Error::InvalidInput(
            "HAS reception time must use GST".to_string(),
        ));
    }
    if !reception_gst.tow_s.is_finite()
        || reception_gst.tow_s < 0.0
        || reception_gst.tow_s >= crate::constants::SECONDS_PER_WEEK
    {
        return Err(Error::InvalidInput(format!(
            "HAS reception TOW {} is not a valid finite week-tow in [0, 604800)",
            reception_gst.tow_s
        )));
    }
    if toh_s > 3599 {
        return Err(Error::InvalidInput(format!(
            "HAS MT1 TOH {toh_s} s exceeds 0..=3599 s range (Table 13)"
        )));
    }
    let reception_s =
        f64::from(reception_gst.week) * crate::constants::SECONDS_PER_WEEK + reception_gst.tow_s;
    let hour = (reception_s / crate::constants::SECONDS_PER_HOUR).floor();
    let candidate = hour * crate::constants::SECONDS_PER_HOUR + f64::from(toh_s);
    let gst_s = if candidate <= reception_s {
        candidate
    } else {
        candidate - crate::constants::SECONDS_PER_HOUR
    };
    Ok(gst_s - crate::constants::GPS_EPOCH_TO_J2000_S)
}

/// HAS validity interval in seconds for a VI index.
pub const fn has_validity_interval_s(index: u8) -> Option<f64> {
    match index {
        0 => Some(5.0),
        1 => Some(10.0),
        2 => Some(15.0),
        3 => Some(20.0),
        4 => Some(30.0),
        5 => Some(60.0),
        6 => Some(90.0),
        7 => Some(120.0),
        8 => Some(180.0),
        9 => Some(240.0),
        10 => Some(300.0),
        11 => Some(600.0),
        12 => Some(900.0),
        13 => Some(1800.0),
        14 => Some(3600.0),
        _ => None,
    }
}

fn read_header(r: &mut BitReader<'_>) -> Result<HasMt1Header> {
    let toh_s = r.u(12)? as u16;
    if toh_s > 3599 {
        return Err(Error::Parse(format!(
            "HAS MT1 header TOH {toh_s} s exceeds 0..=3599 s range (Table 13)"
        )));
    }
    Ok(HasMt1Header {
        toh_s,
        mask: r.flag()?,
        orbit: r.flag()?,
        clock_full_set: r.flag()?,
        clock_subset: r.flag()?,
        code_bias: r.flag()?,
        phase_bias: r.flag()?,
        reserved: r.u(4)? as u8,
        mask_id: r.u(5)? as u8,
        iod_set_id: r.u(5)? as u8,
    })
}

fn write_header(w: &mut BitWriter, header: HasMt1Header) -> Result<()> {
    if header.toh_s > 3599 {
        return Err(Error::InvalidInput(format!(
            "HAS MT1 header TOH {} s exceeds 0..=3599 s range (Table 13)",
            header.toh_s
        )));
    }
    if header.reserved > 15 {
        return Err(Error::InvalidInput(format!(
            "HAS MT1 header reserved {} exceeds 4-bit maximum (15)",
            header.reserved
        )));
    }
    if header.mask_id > 31 {
        return Err(Error::InvalidInput(format!(
            "HAS MT1 header mask ID {} exceeds 5-bit maximum (31)",
            header.mask_id
        )));
    }
    if header.iod_set_id > 31 {
        return Err(Error::InvalidInput(format!(
            "HAS MT1 header IOD set ID {} exceeds 5-bit maximum (31)",
            header.iod_set_id
        )));
    }
    w.push_u(u64::from(header.toh_s), 12);
    w.push_flag(header.mask);
    w.push_flag(header.orbit);
    w.push_flag(header.clock_full_set);
    w.push_flag(header.clock_subset);
    w.push_flag(header.code_bias);
    w.push_flag(header.phase_bias);
    w.push_u(u64::from(header.reserved), 4);
    w.push_u(u64::from(header.mask_id), 5);
    w.push_u(u64::from(header.iod_set_id), 5);
    Ok(())
}

fn read_mask_block(r: &mut BitReader<'_>) -> Result<HasMaskBlock> {
    let nsys = r.u(4)? as usize;
    if nsys == 0 {
        return Err(Error::Parse(
            "HAS MT1 mask Nsys is reserved value 0".to_string(),
        ));
    }
    let mut systems = Vec::with_capacity(nsys);
    for _ in 0..nsys {
        let gnss_id = r.u(4)? as u8;
        let system = has_gnss_system(gnss_id)?;
        let sat_mask = r.u(40)?;
        let signal_mask = r.u(16)? as u16;
        let cell_mask_available = r.flag()?;
        let satellites = mask_indices(sat_mask, 40)
            .into_iter()
            .map(|idx| idx + 1)
            .collect::<Vec<_>>();
        let signals = mask_indices(u64::from(signal_mask), 16);
        let cell_mask = if cell_mask_available {
            let mut cells = Vec::with_capacity(satellites.len() * signals.len());
            for _ in 0..satellites.len() * signals.len() {
                cells.push(r.flag()?);
            }
            Some(cells)
        } else {
            None
        };
        let nav_message = r.u(3)? as u8;
        systems.push(HasGnssMask {
            system,
            satellites,
            signals,
            cell_mask,
            nav_message,
        });
    }
    let reserved = r.u(6)? as u8;
    Ok(HasMaskBlock { systems, reserved })
}

/// Satellite mask width: HAS satellites run `1..=40` (HAS SIS ICD Table 19).
const HAS_SATELLITE_MASK_BITS: u8 = 40;
/// Signal mask width: HAS signal indices run `0..=15`.
const HAS_SIGNAL_MASK_BITS: u8 = 16;

/// Refuse a mask block that the MT1 mask syntax (HAS SIS ICD Table 15) cannot
/// state exactly. Every mask is checked before it is written or used to write
/// or read correction blocks: a caller-built inline mask, an inline mask read
/// from the wire, and the mask of a prior context.
fn validate_mask_block_structural(mask: &HasMaskBlock) -> Result<()> {
    let refuse = |what: String| Err(Error::InvalidInput(format!("HAS MT1 mask {what}")));
    // Nsys is four bits and 0 is reserved, which the reader refuses.
    if !(1..=15).contains(&mask.systems.len()) {
        return refuse(format!(
            "holds {} systems; Nsys states 1..=15",
            mask.systems.len()
        ));
    }
    if mask.reserved > 63 {
        return refuse(format!(
            "reserved value {} exceeds 6-bit maximum (63)",
            mask.reserved
        ));
    }
    for (index, system) in mask.systems.iter().enumerate() {
        let name = system.system;
        if has_gnss_id(name).is_none() {
            return refuse(format!("names {name:?}, which has no HAS GNSS ID"));
        }
        if mask.systems[..index]
            .iter()
            .any(|earlier| earlier.system == name)
        {
            return refuse(format!("names {name:?} more than once"));
        }
        if system.nav_message > 7 {
            return refuse(format!(
                "{name:?} nav_message {} exceeds 3-bit maximum (7)",
                system.nav_message
            ));
        }
        if let Some(&prn) = system
            .satellites
            .iter()
            .find(|&&prn| !(1..=HAS_SATELLITE_MASK_BITS).contains(&prn))
        {
            return refuse(format!(
                "{name:?} satellite {prn} is outside 1..={HAS_SATELLITE_MASK_BITS}"
            ));
        }
        if !system.satellites.windows(2).all(|pair| pair[0] < pair[1]) {
            return refuse(format!(
                "{name:?} satellite list {:?} is not strictly ascending",
                system.satellites
            ));
        }
        if let Some(&signal) = system
            .signals
            .iter()
            .find(|&&signal| signal >= HAS_SIGNAL_MASK_BITS)
        {
            return refuse(format!(
                "{name:?} signal {signal} is outside 0..={}",
                HAS_SIGNAL_MASK_BITS - 1
            ));
        }
        if !system.signals.windows(2).all(|pair| pair[0] < pair[1]) {
            return refuse(format!(
                "{name:?} signal list {:?} is not strictly ascending",
                system.signals
            ));
        }
        if let Some(cells) = &system.cell_mask {
            let expected = system.satellites.len() * system.signals.len();
            if cells.len() != expected {
                return refuse(format!(
                    "{name:?} cell mask holds {} cells for {expected} satellite/signal pairs",
                    cells.len()
                ));
            }
        }
    }
    Ok(())
}

/// Arrange a correction block's records in mask order. Each record names its
/// satellite, and a bias record its signal too, so its mask position is fixed
/// whatever order the block holds the records in, and it is written there.
/// Refused by name: a mask entry with no record (an unavailable correction is
/// a record holding `None`, so an omitted one is ambiguous), a second record
/// for one entry, and a record for an entry the mask does not declare.
fn records_in_mask_order<'a, R, K>(
    block_name: &str,
    expected: &[K],
    records: &'a [R],
    key: impl Fn(&R) -> K,
    describe: impl Fn(K) -> String,
) -> Result<Vec<&'a R>>
where
    K: Copy + Eq + std::hash::Hash,
{
    let positions: std::collections::HashMap<K, usize> = expected
        .iter()
        .enumerate()
        .map(|(index, &entry)| (entry, index))
        .collect();
    let mut slots: Vec<Option<&'a R>> = vec![None; expected.len()];
    for record in records {
        let entry = key(record);
        let Some(slot) = positions
            .get(&entry)
            .and_then(|&index| slots.get_mut(index))
        else {
            return Err(Error::InvalidInput(format!(
                "HAS {block_name} records contain unexpected {}, which the mask does not declare",
                describe(entry)
            )));
        };
        if slot.replace(record).is_some() {
            return Err(Error::InvalidInput(format!(
                "HAS {block_name} records contain {} more than once",
                describe(entry)
            )));
        }
    }
    slots
        .into_iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (slot, &entry))| {
            slot.ok_or_else(|| {
                Error::InvalidInput(format!(
                    "HAS {block_name} records missing {} (mask entry {index})",
                    describe(entry)
                ))
            })
        })
        .collect()
}

fn describe_satellite(sat: GnssSatelliteId) -> String {
    format!("satellite {sat}")
}

fn describe_cell((sat, signal_id): (GnssSatelliteId, u8)) -> String {
    format!("satellite {sat} signal {signal_id}")
}

/// Mask cells as `(satellite, signal)` keys in mask order.
fn mask_cell_keys(mask: &HasMaskBlock) -> Result<Vec<(GnssSatelliteId, u8)>> {
    Ok(mask_cells(mask)?
        .into_iter()
        .map(|cell| (cell.sat, cell.signal_id))
        .collect())
}

fn write_mask_block(w: &mut BitWriter, mask: &HasMaskBlock) -> Result<()> {
    validate_mask_block_structural(mask)?;
    w.push_u(mask.systems.len() as u64, 4);
    for system in &mask.systems {
        let gnss_id = has_gnss_id(system.system).ok_or_else(|| {
            Error::InvalidInput(format!("unsupported HAS GNSS system {:?}", system.system))
        })?;
        w.push_u(u64::from(gnss_id), 4);
        let sat_mask = mask_from_indices(system.satellites.iter().map(|prn| prn - 1), 40);
        let signal_mask = mask_from_indices(system.signals.iter().copied(), 16);
        w.push_u(sat_mask, 40);
        w.push_u(signal_mask, 16);
        w.push_flag(system.cell_mask.is_some());
        if let Some(cells) = &system.cell_mask {
            for &cell in cells {
                w.push_flag(cell);
            }
        }
        w.push_u(u64::from(system.nav_message), 3);
    }
    w.push_u(u64::from(mask.reserved), 6);
    Ok(())
}

fn read_orbit_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasOrbitBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS orbit VI is reserved".to_string()));
    }
    let mut records = Vec::new();
    for sat in mask_satellites(mask)? {
        let iode = r.u(iode_bits(sat.system))? as u32;
        let radial = r.i(13)? as i16;
        let along = r.i(12)? as i16;
        let cross = r.i(12)? as i16;
        let radial_m = (radial != HAS_ORBIT_RADIAL_INVALID)
            .then_some(f64::from(radial) * HAS_ORBIT_RADIAL_SCALE_M);
        let along_m = (along != HAS_ORBIT_ALONG_CROSS_INVALID)
            .then_some(f64::from(along) * HAS_ORBIT_ALONG_CROSS_SCALE_M);
        let cross_m = (cross != HAS_ORBIT_ALONG_CROSS_INVALID)
            .then_some(f64::from(cross) * HAS_ORBIT_ALONG_CROSS_SCALE_M);
        records.push(HasOrbitCorrection {
            sat,
            iode,
            radial_m,
            along_m,
            cross_m,
        });
    }
    Ok(HasOrbitBlock {
        validity_interval,
        records,
    })
}

fn write_orbit_block(w: &mut BitWriter, mask: &HasMaskBlock, orbit: &HasOrbitBlock) -> Result<()> {
    validate_mask_block_structural(mask)?;
    if has_validity_interval_s(orbit.validity_interval).is_none() {
        return Err(Error::InvalidInput(format!(
            "HAS orbit VI {} is reserved",
            orbit.validity_interval
        )));
    }
    let records = records_in_mask_order(
        "orbit",
        &mask_satellites(mask)?,
        &orbit.records,
        |rec| rec.sat,
        describe_satellite,
    )?;
    w.push_u(u64::from(orbit.validity_interval), 4);
    for rec in records {
        let max_iode = match rec.sat.system {
            GnssSystem::Gps => 255,
            GnssSystem::Galileo => 1023,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unsupported GNSS system {:?} for orbit IODE",
                    rec.sat.system
                )))
            }
        };
        if rec.iode > max_iode {
            return Err(Error::InvalidInput(format!(
                "HAS orbit IODE {} for {} exceeds {}-bit maximum ({max_iode})",
                rec.iode,
                rec.sat,
                iode_bits(rec.sat.system)
            )));
        }
        w.push_u(u64::from(rec.iode), iode_bits(rec.sat.system));

        let raw_radial = if let Some(m) = rec.radial_m {
            if !m.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "HAS orbit radial correction for {} is not finite",
                    rec.sat
                )));
            }
            let raw = (m / HAS_ORBIT_RADIAL_SCALE_M).round() as i64;
            if !(-4095..=4095).contains(&raw) {
                return Err(Error::InvalidInput(format!(
                    "HAS orbit radial correction for {} ({m} m) out of range (raw {raw} not in [-4095, 4095])",
                    rec.sat
                )));
            }
            raw
        } else {
            i64::from(HAS_ORBIT_RADIAL_INVALID)
        };
        w.push_i(raw_radial, 13);

        let raw_along = if let Some(m) = rec.along_m {
            if !m.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "HAS orbit along correction for {} is not finite",
                    rec.sat
                )));
            }
            let raw = (m / HAS_ORBIT_ALONG_CROSS_SCALE_M).round() as i64;
            if !(-2047..=2047).contains(&raw) {
                return Err(Error::InvalidInput(format!(
                    "HAS orbit along correction for {} ({m} m) out of range (raw {raw} not in [-2047, 2047])",
                    rec.sat
                )));
            }
            raw
        } else {
            i64::from(HAS_ORBIT_ALONG_CROSS_INVALID)
        };
        w.push_i(raw_along, 12);

        let raw_cross = if let Some(m) = rec.cross_m {
            if !m.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "HAS orbit cross correction for {} is not finite",
                    rec.sat
                )));
            }
            let raw = (m / HAS_ORBIT_ALONG_CROSS_SCALE_M).round() as i64;
            if !(-2047..=2047).contains(&raw) {
                return Err(Error::InvalidInput(format!(
                    "HAS orbit cross correction for {} ({m} m) out of range (raw {raw} not in [-2047, 2047])",
                    rec.sat
                )));
            }
            raw
        } else {
            i64::from(HAS_ORBIT_ALONG_CROSS_INVALID)
        };
        w.push_i(raw_cross, 12);
    }
    Ok(())
}

/// Read HAS MT1 clock full-set block per Galileo HAS SIS ICD Section 5.2.3, Table 27.
fn read_clock_full_set_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasClockBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS clock VI is reserved".to_string()));
    }
    let mut systems = Vec::with_capacity(mask.systems.len());
    for system_mask in &mask.systems {
        let multiplier_index = r.u(2)? as u8;
        systems.push(HasClockSystem {
            system: system_mask.system,
            multiplier_index,
        });
    }
    let mut records = Vec::new();
    for (system_mask, sys_meta) in mask.systems.iter().zip(&systems) {
        let multiplier = dcm_multiplier(sys_meta.multiplier_index);
        for &prn in &system_mask.satellites {
            let dcc = r.i(13)? as i16;
            let (correction_m, do_not_use) = has_clock_value_m(dcc, multiplier);
            records.push(HasClockCorrection {
                sat: has_satellite(system_mask.system, prn)?,
                correction_m,
                do_not_use,
            });
        }
    }
    Ok(HasClockBlock {
        validity_interval,
        systems,
        records,
    })
}

/// Arrange full-set clock metadata in mask system order. Each entry names its
/// system, so it is written at that system's mask position whatever order
/// `systems` holds it in. Refused by name: a reserved multiplier index, a
/// system named twice, a system the mask does not declare, and a mask system
/// with no entry.
fn clock_full_set_metadata_in_mask_order<'a>(
    mask: &HasMaskBlock,
    systems: &'a [HasClockSystem],
) -> Result<Vec<&'a HasClockSystem>> {
    for meta in systems {
        if meta.multiplier_index > 3 {
            return Err(Error::InvalidInput(format!(
                "HAS clock multiplier index {} for {:?} is reserved (must be 0..=3)",
                meta.multiplier_index, meta.system
            )));
        }
    }
    let mut seen = Vec::new();
    for meta in systems {
        if seen.contains(&meta.system) {
            return Err(Error::InvalidInput(format!(
                "duplicate HAS clock metadata for {:?}",
                meta.system
            )));
        }
        seen.push(meta.system);
    }
    for meta in systems {
        if !mask.systems.iter().any(|m| m.system == meta.system) {
            return Err(Error::InvalidInput(format!(
                "extraneous HAS clock metadata for {:?} not declared in mask",
                meta.system
            )));
        }
    }
    mask.systems
        .iter()
        .map(|mask_sys| {
            systems
                .iter()
                .find(|meta| meta.system == mask_sys.system)
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "missing HAS clock metadata for {:?}",
                        mask_sys.system
                    ))
                })
        })
        .collect()
}

/// Write HAS MT1 clock full-set block per Galileo HAS SIS ICD Section 5.2.3, Table 27.
fn write_clock_full_set_block(
    w: &mut BitWriter,
    mask: &HasMaskBlock,
    clock: &HasClockBlock,
) -> Result<()> {
    validate_mask_block_structural(mask)?;
    if has_validity_interval_s(clock.validity_interval).is_none() {
        return Err(Error::InvalidInput(format!(
            "HAS clock VI {} is reserved",
            clock.validity_interval
        )));
    }
    let systems = clock_full_set_metadata_in_mask_order(mask, &clock.systems)?;
    for rec in &clock.records {
        if rec.correction_m.is_some() && rec.do_not_use {
            return Err(Error::InvalidInput(format!(
                "contradictory HAS clock correction for {}: correction is Some while do_not_use is true",
                rec.sat
            )));
        }
    }
    let records = records_in_mask_order(
        "clock",
        &mask_satellites(mask)?,
        &clock.records,
        |rec| rec.sat,
        describe_satellite,
    )?;
    w.push_u(u64::from(clock.validity_interval), 4);
    for meta in &systems {
        w.push_u(u64::from(meta.multiplier_index), 2);
    }
    let mut rec_iter = records.into_iter();
    for (system_mask, meta) in mask.systems.iter().zip(systems) {
        let multiplier = dcm_multiplier(meta.multiplier_index);
        let scale_m = HAS_CLOCK_SCALE_M * multiplier;
        let min_m = -4095.0 * scale_m;
        let max_m = 4094.0 * scale_m;
        for _ in 0..system_mask.satellites.len() {
            let Some(rec) = rec_iter.next() else {
                return Err(Error::InvalidInput(
                    "HAS clock records missing expected satellite".to_string(),
                ));
            };
            let raw = if rec.do_not_use {
                i64::from(HAS_CLOCK_DO_NOT_USE)
            } else if let Some(m) = rec.correction_m {
                if !m.is_finite() {
                    return Err(Error::InvalidInput(format!(
                        "HAS clock correction for {} is not finite",
                        rec.sat
                    )));
                }
                let raw = (m / scale_m).round() as i64;
                if !(-4095..=4094).contains(&raw) {
                    return Err(Error::InvalidInput(format!(
                        "HAS clock correction for {} ({m} m) out of range for {:?} multiplier index {} ({multiplier:.1}x): allowed range is [{min_m:.4}, {max_m:.4}] m",
                        rec.sat, meta.system, meta.multiplier_index
                    )));
                }
                raw
            } else {
                i64::from(HAS_CLOCK_INVALID)
            };
            w.push_i(raw, 13);
        }
    }
    Ok(())
}

/// Read HAS MT1 clock subset block per Galileo HAS SIS ICD Section 5.2.4, Table 32 and Table 33.
fn read_clock_subset_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasClockBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS clock subset VI is reserved".to_string()));
    }
    let nsys_sub = r.u(4)? as usize;
    let mut systems = Vec::with_capacity(nsys_sub);
    let mut records = Vec::new();
    for _ in 0..nsys_sub {
        let gnss_id = r.u(4)? as u8;
        let system = has_gnss_system(gnss_id)?;
        if systems.iter().any(|s: &HasClockSystem| s.system == system) {
            return Err(Error::Parse(format!(
                "duplicate HAS clock subset GNSS ID {gnss_id} ({system:?})"
            )));
        }
        let dcm_idx = r.u(2)? as u8;
        systems.push(HasClockSystem {
            system,
            multiplier_index: dcm_idx,
        });
        let multiplier = dcm_multiplier(dcm_idx);
        let Some(system_mask) = mask.systems.iter().find(|m| m.system == system) else {
            return Err(Error::Parse(format!(
                "HAS clock subset GNSS {:?} not in mask",
                system
            )));
        };
        let mut subset = Vec::with_capacity(system_mask.satellites.len());
        for _ in 0..system_mask.satellites.len() {
            subset.push(r.flag()?);
        }
        for (&present, &prn) in subset.iter().zip(&system_mask.satellites) {
            if !present {
                continue;
            }
            let dcc = r.i(13)? as i16;
            let (correction_m, do_not_use) = has_clock_value_m(dcc, multiplier);
            records.push(HasClockCorrection {
                sat: has_satellite(system, prn)?,
                correction_m,
                do_not_use,
            });
        }
    }
    Ok(HasClockBlock {
        validity_interval,
        systems,
        records,
    })
}

/// Write HAS MT1 clock subset block per Galileo HAS SIS ICD Section 5.2.4, Table 32 and Table 33.
fn write_clock_subset_block(
    w: &mut BitWriter,
    mask: &HasMaskBlock,
    clock: &HasClockBlock,
) -> Result<()> {
    validate_mask_block_structural(mask)?;
    if has_validity_interval_s(clock.validity_interval).is_none() {
        return Err(Error::InvalidInput(format!(
            "HAS clock subset VI {} is reserved",
            clock.validity_interval
        )));
    }
    if clock.systems.len() > 15 {
        return Err(Error::InvalidInput(format!(
            "HAS clock subset system count {} exceeds 4-bit limit (max 15)",
            clock.systems.len()
        )));
    }
    if clock.systems.is_empty() {
        if !clock.records.is_empty() {
            return Err(Error::InvalidInput(
                "HAS clock subset records present but systems list is empty".to_string(),
            ));
        }
        w.push_u(u64::from(clock.validity_interval), 4);
        w.push_u(0, 4);
        return Ok(());
    }

    let mut seen_systems = Vec::new();
    for meta in &clock.systems {
        if meta.multiplier_index > 3 {
            return Err(Error::InvalidInput(format!(
                "HAS clock subset multiplier index {} for {:?} is reserved (must be 0..=3)",
                meta.multiplier_index, meta.system
            )));
        }
        if seen_systems.contains(&meta.system) {
            return Err(Error::InvalidInput(format!(
                "duplicate HAS clock subset system {:?}",
                meta.system
            )));
        }
        seen_systems.push(meta.system);
        if !mask.systems.iter().any(|m| m.system == meta.system) {
            return Err(Error::InvalidInput(format!(
                "HAS clock subset system {:?} is not declared in mask",
                meta.system
            )));
        }
    }

    let mut seen_sats = Vec::new();
    for rec in &clock.records {
        if rec.correction_m.is_some() && rec.do_not_use {
            return Err(Error::InvalidInput(format!(
                "contradictory HAS clock subset correction for {}: correction is Some while do_not_use is true",
                rec.sat
            )));
        }
        if seen_sats.contains(&rec.sat) {
            return Err(Error::InvalidInput(format!(
                "duplicate HAS clock subset satellite {}",
                rec.sat
            )));
        }
        seen_sats.push(rec.sat);
        if !clock.systems.iter().any(|s| s.system == rec.sat.system) {
            return Err(Error::InvalidInput(format!(
                "HAS clock subset satellite {} has no corresponding system metadata",
                rec.sat
            )));
        }
    }

    // Each record names its satellite, so its place in the block is fixed by
    // its system's position in `systems` and the satellite's position in the
    // mask, whatever order `records` holds them in.
    for rec in &clock.records {
        let declared = mask
            .systems
            .iter()
            .find(|m| m.system == rec.sat.system)
            .is_some_and(|m| m.satellites.contains(&rec.sat.prn));
        if !declared {
            return Err(Error::InvalidInput(format!(
                "HAS clock subset satellite {} is not declared in mask",
                rec.sat
            )));
        }
    }

    w.push_u(u64::from(clock.validity_interval), 4);
    w.push_u(clock.systems.len() as u64, 4);

    for meta in &clock.systems {
        let Some(system_mask) = mask.systems.iter().find(|m| m.system == meta.system) else {
            return Err(Error::InvalidInput(format!(
                "HAS clock subset system {:?} is not declared in mask",
                meta.system
            )));
        };
        let gnss_id = has_gnss_id(meta.system).ok_or_else(|| {
            Error::InvalidInput(format!("unsupported HAS GNSS system {:?}", meta.system))
        })?;
        w.push_u(u64::from(gnss_id), 4);
        w.push_u(u64::from(meta.multiplier_index), 2);

        let multiplier = dcm_multiplier(meta.multiplier_index);
        let scale_m = HAS_CLOCK_SCALE_M * multiplier;
        let min_m = -4095.0 * scale_m;
        let max_m = 4094.0 * scale_m;

        let mut selected_records: Vec<&HasClockCorrection> =
            Vec::with_capacity(system_mask.satellites.len());
        for &prn in &system_mask.satellites {
            let sat = has_satellite(meta.system, prn)?;
            let record = clock.records.iter().find(|r| r.sat == sat);
            w.push_flag(record.is_some());
            selected_records.extend(record);
        }

        for rec in selected_records {
            let raw = if rec.do_not_use {
                i64::from(HAS_CLOCK_DO_NOT_USE)
            } else if let Some(m) = rec.correction_m {
                if !m.is_finite() {
                    return Err(Error::InvalidInput(format!(
                        "HAS clock subset correction for {} is not finite",
                        rec.sat
                    )));
                }
                let raw = (m / scale_m).round() as i64;
                if !(-4095..=4094).contains(&raw) {
                    return Err(Error::InvalidInput(format!(
                        "HAS clock subset correction for {} ({m} m) out of range for {:?} multiplier index {} ({multiplier:.1}x): allowed range is [{min_m:.4}, {max_m:.4}] m",
                        rec.sat, meta.system, meta.multiplier_index
                    )));
                }
                raw
            } else {
                i64::from(HAS_CLOCK_INVALID)
            };
            w.push_i(raw, 13);
        }
    }
    Ok(())
}

fn read_code_bias_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasCodeBiasBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS code-bias VI is reserved".to_string()));
    }
    let mut records = Vec::new();
    for cell in mask_cells(mask)? {
        let raw = r.i(11)? as i16;
        let bias_m =
            (raw != HAS_CODE_BIAS_INVALID).then_some(f64::from(raw) * HAS_CODE_BIAS_SCALE_M);
        records.push(HasCodeBias {
            sat: cell.sat,
            signal_id: cell.signal_id,
            bias_m,
        });
    }
    Ok(HasCodeBiasBlock {
        validity_interval,
        records,
    })
}

fn write_code_bias_block(
    w: &mut BitWriter,
    mask: &HasMaskBlock,
    code_bias: &HasCodeBiasBlock,
) -> Result<()> {
    validate_mask_block_structural(mask)?;
    if has_validity_interval_s(code_bias.validity_interval).is_none() {
        return Err(Error::InvalidInput(format!(
            "HAS code-bias VI {} is reserved",
            code_bias.validity_interval
        )));
    }
    let records = records_in_mask_order(
        "code-bias",
        &mask_cell_keys(mask)?,
        &code_bias.records,
        |rec| (rec.sat, rec.signal_id),
        describe_cell,
    )?;
    w.push_u(u64::from(code_bias.validity_interval), 4);
    for rec in records {
        let raw = if let Some(m) = rec.bias_m {
            if !m.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "HAS code bias for {} signal {} is not finite",
                    rec.sat, rec.signal_id
                )));
            }
            let raw = (m / HAS_CODE_BIAS_SCALE_M).round() as i64;
            if !(-1023..=1023).contains(&raw) {
                return Err(Error::InvalidInput(format!(
                    "HAS code bias for {} signal {} ({m} m) out of range (raw {raw} not in [-1023, 1023])",
                    rec.sat, rec.signal_id
                )));
            }
            raw
        } else {
            i64::from(HAS_CODE_BIAS_INVALID)
        };
        w.push_i(raw, 11);
    }
    Ok(())
}

fn read_phase_bias_block(r: &mut BitReader<'_>, mask: &HasMaskBlock) -> Result<HasPhaseBiasBlock> {
    let validity_interval = r.u(4)? as u8;
    if has_validity_interval_s(validity_interval).is_none() {
        return Err(Error::Parse("HAS phase-bias VI is reserved".to_string()));
    }
    let mut records = Vec::new();
    for cell in mask_cells(mask)? {
        let raw = r.i(11)? as i16;
        let discontinuity_indicator = r.u(2)? as u8;
        let bias_cycles = if raw == HAS_PHASE_BIAS_INVALID {
            None
        } else {
            Some(f64::from(raw) * HAS_PHASE_BIAS_SCALE_CYCLES)
        };
        records.push(HasPhaseBias {
            sat: cell.sat,
            signal_id: cell.signal_id,
            bias_cycles,
            discontinuity_indicator,
        });
    }
    Ok(HasPhaseBiasBlock {
        validity_interval,
        records,
    })
}

fn write_phase_bias_block(
    w: &mut BitWriter,
    mask: &HasMaskBlock,
    phase_bias: &HasPhaseBiasBlock,
) -> Result<()> {
    validate_mask_block_structural(mask)?;
    if has_validity_interval_s(phase_bias.validity_interval).is_none() {
        return Err(Error::InvalidInput(format!(
            "HAS phase-bias VI {} is reserved",
            phase_bias.validity_interval
        )));
    }
    let records = records_in_mask_order(
        "phase-bias",
        &mask_cell_keys(mask)?,
        &phase_bias.records,
        |rec| (rec.sat, rec.signal_id),
        describe_cell,
    )?;
    w.push_u(u64::from(phase_bias.validity_interval), 4);
    for rec in records {
        if rec.discontinuity_indicator > 3 {
            return Err(Error::InvalidInput(format!(
                "HAS phase bias discontinuity indicator {} for {} signal {} exceeds 2-bit maximum (3)",
                rec.discontinuity_indicator, rec.sat, rec.signal_id
            )));
        }
        let raw = if let Some(cycles) = rec.bias_cycles {
            if !cycles.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "HAS phase bias for {} signal {} is not finite",
                    rec.sat, rec.signal_id
                )));
            }
            let raw = (cycles / HAS_PHASE_BIAS_SCALE_CYCLES).round() as i64;
            if !(-1023..=1023).contains(&raw) {
                return Err(Error::InvalidInput(format!(
                    "HAS phase bias for {} signal {} ({cycles} cycles) out of range (raw {raw} not in [-1023, 1023])",
                    rec.sat, rec.signal_id
                )));
            }
            raw
        } else {
            i64::from(HAS_PHASE_BIAS_INVALID)
        };
        w.push_i(raw, 11);
        w.push_u(u64::from(rec.discontinuity_indicator), 2);
    }
    Ok(())
}

/// Delta clock multiplier (DCM) scaling factor for a 2-bit DCM index per Galileo HAS SIS ICD Table 29.
pub const fn dcm_multiplier(value: u8) -> f64 {
    match value {
        0 => 1.0,
        1 => 2.0,
        2 => 3.0,
        _ => 4.0,
    }
}

fn has_clock_value_m(raw: i16, multiplier: f64) -> (Option<f64>, bool) {
    if raw == HAS_CLOCK_INVALID {
        (None, false)
    } else if raw == HAS_CLOCK_DO_NOT_USE {
        (None, true)
    } else {
        (Some(f64::from(raw) * HAS_CLOCK_SCALE_M * multiplier), false)
    }
}

#[derive(Clone, Copy)]
struct HasCell {
    sat: GnssSatelliteId,
    signal_id: u8,
}

fn mask_satellites(mask: &HasMaskBlock) -> Result<Vec<GnssSatelliteId>> {
    let mut out = Vec::new();
    for system in &mask.systems {
        for &prn in &system.satellites {
            out.push(has_satellite(system.system, prn)?);
        }
    }
    Ok(out)
}

fn mask_cells(mask: &HasMaskBlock) -> Result<Vec<HasCell>> {
    let mut out = Vec::new();
    for system in &mask.systems {
        for (sat_idx, &prn) in system.satellites.iter().enumerate() {
            let sat = has_satellite(system.system, prn)?;
            for (sig_idx, &signal_id) in system.signals.iter().enumerate() {
                let present = system
                    .cell_mask
                    .as_ref()
                    .map(|cells| cells[sat_idx * system.signals.len() + sig_idx])
                    .unwrap_or(true);
                if present {
                    out.push(HasCell { sat, signal_id });
                }
            }
        }
    }
    Ok(out)
}

fn read_padding_bits(r: &mut BitReader<'_>) -> Result<Vec<bool>> {
    let mut padding = Vec::with_capacity(r.remaining_bits());
    while r.remaining_bits() > 0 {
        padding.push(r.flag()?);
    }
    Ok(padding)
}

fn has_gnss_system(gnss_id: u8) -> Result<GnssSystem> {
    match gnss_id {
        0 => Ok(GnssSystem::Gps),
        2 => Ok(GnssSystem::Galileo),
        _ => Err(Error::Parse(format!("unsupported HAS GNSS id {gnss_id}"))),
    }
}

/// The satellite a HAS mask entry names. HAS numbers satellites `1..=40` for
/// both GPS and Galileo (HAS SIS ICD Table 19), narrower than the shared
/// satellite-token range, so the bound is checked here rather than left to the
/// identifier constructor.
fn has_satellite(system: GnssSystem, prn: u8) -> Result<GnssSatelliteId> {
    if !(1..=HAS_SATELLITE_MASK_BITS).contains(&prn) {
        return Err(Error::Parse(format!(
            "invalid HAS satellite {system:?} {prn}: outside 1..={HAS_SATELLITE_MASK_BITS}"
        )));
    }
    GnssSatelliteId::new(system, prn)
        .map_err(|err| Error::Parse(format!("invalid HAS satellite {system:?} {prn}: {err}")))
}

fn has_gnss_id(system: GnssSystem) -> Option<u8> {
    match system {
        GnssSystem::Gps => Some(0),
        GnssSystem::Galileo => Some(2),
        _ => None,
    }
}

fn iode_bits(system: GnssSystem) -> usize {
    match system {
        GnssSystem::Galileo => 10,
        _ => 8,
    }
}

fn mask_indices(mask: u64, width: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for idx in 0..width {
        if ((mask >> (width - 1 - idx)) & 1) != 0 {
            out.push(idx as u8);
        }
    }
    out
}

fn mask_from_indices(indices: impl IntoIterator<Item = u8>, width: usize) -> u64 {
    let mut mask = 0_u64;
    for idx in indices {
        mask |= 1_u64 << (width - 1 - usize::from(idx));
    }
    mask
}

fn has_signal_frequency_hz(system: GnssSystem, signal_id: u8) -> Result<f64> {
    match (system, signal_id) {
        (GnssSystem::Gps, 0 | 3 | 4 | 5) => Ok(F_L1_HZ),
        (GnssSystem::Gps, 6..=9) => Ok(F_L2_HZ),
        (GnssSystem::Gps, 11..=13) => Ok(F_E5A_HZ),
        (GnssSystem::Galileo, 0..=2) => Ok(F_E1_HZ),
        (GnssSystem::Galileo, 3..=5) => Ok(F_E5A_HZ),
        (GnssSystem::Galileo, 6..=8) => Ok(1_207_140_000.0),
        (GnssSystem::Galileo, 9..=11) => Ok(1_191_795_000.0),
        (GnssSystem::Galileo, 12..=14) => Ok(1_278_750_000.0),
        _ => Err(Error::Parse(format!(
            "unsupported HAS signal {signal_id} for {system:?}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn mt1_mask_orbit_clock_and_bias_blocks_roundtrip() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 1234,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 7,
                iod_set_id: 9,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![3],
                    signals: vec![0, 9],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: 42,
                    radial_m: Some(1.25),
                    along_m: Some(-2.0),
                    cross_m: Some(3.0),
                }],
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    correction_m: Some(-0.75),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasCodeBias {
                        sat,
                        signal_id: 0,
                        bias_m: Some(0.24),
                    },
                    HasCodeBias {
                        sat,
                        signal_id: 9,
                        bias_m: Some(-0.46),
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![
                    HasPhaseBias {
                        sat,
                        signal_id: 0,
                        bias_cycles: Some(1.25),
                        discontinuity_indicator: 1,
                    },
                    HasPhaseBias {
                        sat,
                        signal_id: 9,
                        bias_cycles: Some(-2.5),
                        discontinuity_indicator: 2,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };

        let body = message.encode().expect("encode HAS MT1");
        let decoded = HasMt1Message::decode(&body).unwrap();
        // Header 32 + mask 74 + orbit 49 + clock full set 19 + code bias 26 +
        // phase bias 30 = 230 payload bits, so `into_bytes` zero-pads the final
        // byte with two bits that `decode` returns as trailing padding.
        assert_eq!(body.len(), 29);
        assert_eq!(decoded.padding_bits.len(), 2);
        assert!(
            decoded.padding_bits.iter().all(|bit| !bit),
            "trailing padding must be zero bits"
        );
        // The padding is the only difference: re-encoding is byte-identical.
        assert_eq!(decoded.encode().unwrap(), body);
        let decoded_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..decoded.clone()
        };
        assert_eq!(decoded_without_padding, message);
    }

    fn mask_only_message(systems: Vec<HasGnssMask>) -> HasMt1Message {
        HasMt1Message {
            header: HasMt1Header {
                toh_s: 1234,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 7,
                iod_set_id: 9,
            },
            mask: Some(HasMaskBlock {
                systems,
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        }
    }

    fn system_mask(system: GnssSystem, satellites: Vec<u8>, signals: Vec<u8>) -> HasGnssMask {
        HasGnssMask {
            system,
            satellites,
            signals,
            cell_mask: None,
            nav_message: 0,
        }
    }

    /// HAS SIS ICD Table 19 numbers GPS and Galileo satellites 1..=40, so G33..G40
    /// and E37..E40 - past the old per-constellation caps - encode and decode.
    #[test]
    fn mt1_mask_carries_satellites_up_to_forty() {
        let message = mask_only_message(vec![
            system_mask(GnssSystem::Gps, vec![1, 32, 33, 40], vec![0]),
            system_mask(GnssSystem::Galileo, vec![36, 37, 40], vec![0, 15]),
        ]);
        let body = message.encode().expect("a Table 19 mask encodes");
        let decoded = HasMt1Message::decode(&body).expect("and decodes");
        assert_eq!(decoded.mask, message.mask);
    }

    /// A mask list the MT1 masks cannot state is refused by name before any bit
    /// is written, instead of underflowing, shifting onto another bit, or
    /// relabelling the system.
    #[test]
    fn mt1_encode_refuses_masks_it_cannot_state() {
        for (systems, needle) in [
            (
                vec![system_mask(GnssSystem::Gps, vec![0], vec![0])],
                "outside 1..=40",
            ),
            (
                vec![system_mask(GnssSystem::Gps, vec![41], vec![0])],
                "outside 1..=40",
            ),
            (
                vec![system_mask(GnssSystem::Galileo, vec![1], vec![16])],
                "outside 0..=15",
            ),
            (
                vec![system_mask(GnssSystem::Gps, vec![5, 3], vec![0])],
                "not strictly ascending",
            ),
            (
                vec![system_mask(GnssSystem::Gps, vec![3, 3], vec![0])],
                "not strictly ascending",
            ),
            (
                vec![system_mask(GnssSystem::Gps, vec![3], vec![9, 0])],
                "not strictly ascending",
            ),
            (
                vec![system_mask(GnssSystem::Glonass, vec![3], vec![0])],
                "no HAS GNSS ID",
            ),
            (Vec::new(), "Nsys states 1..=15"),
        ] {
            let err = mask_only_message(systems)
                .encode()
                .expect_err("the mask cannot be stated");
            assert!(
                matches!(err, Error::InvalidInput(ref text) if text.contains(needle)),
                "expected {needle:?}, got {err}"
            );
        }

        let mut wrong_cells = system_mask(GnssSystem::Gps, vec![3, 4], vec![0, 9]);
        wrong_cells.cell_mask = Some(vec![true; 3]);
        let err = mask_only_message(vec![wrong_cells])
            .encode()
            .expect_err("three cells for four pairs");
        assert!(err.to_string().contains("cell mask holds 3 cells"), "{err}");
    }

    #[test]
    fn orbit_writer_refuses_omitted_satellite_by_name() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 0,
                records: vec![HasOrbitCorrection {
                    sat: sat1,
                    iode: 10,
                    radial_m: Some(0.05),
                    along_m: Some(0.08),
                    cross_m: Some(0.16),
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("G02"),
            "expected error naming omitted satellite G02, got: {err_msg}"
        );
        assert!(
            err_msg.contains("missing satellite"),
            "expected missing satellite error, got: {err_msg}"
        );
    }

    #[test]
    fn orbit_writer_places_reordered_records_at_mask_positions() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 0,
                records: vec![
                    HasOrbitCorrection {
                        sat: sat2,
                        iode: 20,
                        radial_m: Some(0.10),
                        along_m: Some(0.16),
                        cross_m: Some(0.24),
                    },
                    HasOrbitCorrection {
                        sat: sat1,
                        iode: 10,
                        radial_m: Some(0.05),
                        along_m: Some(0.08),
                        cross_m: Some(0.16),
                    },
                ],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // Each record names its satellite, so the reordered block is written
        // with G01 first, exactly as the same block in mask order.
        let body = message.encode().expect("reordered records encode");
        let mut in_mask_order = message.clone();
        in_mask_order.orbit.as_mut().unwrap().records.reverse();
        assert_eq!(in_mask_order.orbit.as_ref().unwrap().records[0].sat, sat1);
        assert_eq!(
            in_mask_order.encode().expect("mask-order records encode"),
            body
        );
        let decoded = HasMt1Message::decode(&body).unwrap();
        let decoded_records = &decoded.orbit.as_ref().unwrap().records;
        assert_eq!(decoded_records[0].sat, sat1);
        assert_eq!(decoded_records[0].iode, 10);
        assert_eq!(decoded_records[1].sat, sat2);
        assert_eq!(decoded_records[1].iode, 20);
    }

    /// Each record names its satellite, and a bias record its signal too, so a
    /// block holding its records in another order than the mask is written
    /// with every record at its mask position. A mask entry with no record, a
    /// second record for one entry and a record for an entry the mask does not
    /// declare are refused by name.
    #[test]
    fn writers_place_reordered_records_and_refuse_omitted_or_duplicate_ones() {
        let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let g02 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let cells = [(g01, 0_u8), (g01, 9), (g02, 0), (g02, 9)];
        let in_mask_order = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: true,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0, 9],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 2,
                records: [(g01, 10_u32), (g02, 20)]
                    .into_iter()
                    .map(|(sat, iode)| HasOrbitCorrection {
                        sat,
                        iode,
                        radial_m: Some(f64::from(iode) * HAS_ORBIT_RADIAL_SCALE_M),
                        along_m: None,
                        cross_m: Some(-f64::from(iode) * HAS_ORBIT_ALONG_CROSS_SCALE_M),
                    })
                    .collect(),
            }),
            clock_full_set: Some(HasClockBlock {
                validity_interval: 2,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 1,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: g01,
                        correction_m: Some(7.0 * HAS_CLOCK_SCALE_M * 2.0),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: g02,
                        correction_m: None,
                        do_not_use: true,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 2,
                records: cells
                    .iter()
                    .zip(1_i16..)
                    .map(|(&(sat, signal_id), raw)| HasCodeBias {
                        sat,
                        signal_id,
                        bias_m: Some(f64::from(raw) * HAS_CODE_BIAS_SCALE_M),
                    })
                    .collect(),
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 2,
                records: cells
                    .iter()
                    .zip(1_u8..)
                    .map(|(&(sat, signal_id), n)| HasPhaseBias {
                        sat,
                        signal_id,
                        bias_cycles: Some(-f64::from(n) * HAS_PHASE_BIAS_SCALE_CYCLES),
                        discontinuity_indicator: n % 4,
                    })
                    .collect(),
            }),
            padding_bits: Vec::new(),
        };
        let body = in_mask_order
            .encode()
            .expect("records in mask order encode");

        let mut reordered = in_mask_order.clone();
        reordered.orbit.as_mut().unwrap().records.reverse();
        reordered.clock_full_set.as_mut().unwrap().records.reverse();
        reordered.code_bias.as_mut().unwrap().records.reverse();
        reordered
            .phase_bias
            .as_mut()
            .unwrap()
            .records
            .rotate_left(1);
        assert_eq!(reordered.encode().expect("reordered records encode"), body);

        let decoded = HasMt1Message::decode(&body).unwrap();
        let orbit_sats: Vec<_> = decoded
            .orbit
            .as_ref()
            .unwrap()
            .records
            .iter()
            .map(|rec| rec.sat)
            .collect();
        assert_eq!(orbit_sats, vec![g01, g02]);
        let clock = decoded.clock_full_set.as_ref().unwrap();
        let clock_sats: Vec<_> = clock.records.iter().map(|rec| rec.sat).collect();
        assert_eq!(clock_sats, vec![g01, g02]);
        assert!(!clock.records[0].do_not_use && clock.records[1].do_not_use);
        let code_cells: Vec<_> = decoded
            .code_bias
            .as_ref()
            .unwrap()
            .records
            .iter()
            .map(|rec| (rec.sat, rec.signal_id))
            .collect();
        assert_eq!(code_cells, cells);
        let phase = &decoded.phase_bias.as_ref().unwrap().records;
        let phase_cells: Vec<_> = phase.iter().map(|rec| (rec.sat, rec.signal_id)).collect();
        assert_eq!(phase_cells, cells);
        let indicators: Vec<_> = phase
            .iter()
            .map(|rec| rec.discontinuity_indicator)
            .collect();
        assert_eq!(indicators, vec![1, 2, 3, 0]);

        let refusal = |edit: &dyn Fn(&mut HasMt1Message)| {
            let mut message = in_mask_order.clone();
            edit(&mut message);
            message
                .encode()
                .expect_err("the block cannot be written")
                .to_string()
        };
        for (err, needle) in [
            (
                refusal(&|m: &mut HasMt1Message| {
                    m.orbit.as_mut().unwrap().records.pop();
                }),
                "HAS orbit records missing satellite G02 (mask entry 1)",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    let records = &mut m.orbit.as_mut().unwrap().records;
                    records[1] = records[0];
                }),
                "HAS orbit records contain satellite G01 more than once",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    let records = &mut m.orbit.as_mut().unwrap().records;
                    records.push(HasOrbitCorrection {
                        sat: GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap(),
                        ..records[0]
                    });
                }),
                "HAS orbit records contain unexpected satellite G03, which the mask does not declare",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    m.clock_full_set.as_mut().unwrap().records.remove(0);
                }),
                "HAS clock records missing satellite G01 (mask entry 0)",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    let records = &mut m.clock_full_set.as_mut().unwrap().records;
                    records[0] = records[1];
                }),
                "HAS clock records contain satellite G02 more than once",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    m.code_bias.as_mut().unwrap().records.remove(2);
                }),
                "HAS code-bias records missing satellite G02 signal 0 (mask entry 2)",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    let records = &mut m.code_bias.as_mut().unwrap().records;
                    records[3] = records[0];
                }),
                "HAS code-bias records contain satellite G01 signal 0 more than once",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    m.phase_bias.as_mut().unwrap().records.pop();
                }),
                "HAS phase-bias records missing satellite G02 signal 9 (mask entry 3)",
            ),
            (
                refusal(&|m: &mut HasMt1Message| {
                    let records = &mut m.phase_bias.as_mut().unwrap().records;
                    records[2] = records[1];
                }),
                "HAS phase-bias records contain satellite G01 signal 9 more than once",
            ),
        ] {
            assert!(err.contains(needle), "expected {needle:?}, got {err}");
        }
    }

    #[test]
    fn orbit_writer_multi_satellite_round_trip() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Galileo, 2).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 100,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 3,
                iod_set_id: 4,
            },
            mask: Some(HasMaskBlock {
                systems: vec![
                    HasGnssMask {
                        system: GnssSystem::Gps,
                        satellites: vec![1],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                    HasGnssMask {
                        system: GnssSystem::Galileo,
                        satellites: vec![2],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                ],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 1,
                records: vec![
                    HasOrbitCorrection {
                        sat: sat1,
                        iode: 42,
                        radial_m: Some(1.25),
                        along_m: Some(-2.0),
                        cross_m: Some(3.0),
                    },
                    HasOrbitCorrection {
                        sat: sat2,
                        iode: 100,
                        radial_m: Some(-0.5),
                        along_m: Some(1.6),
                        cross_m: Some(-2.4),
                    },
                ],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let encoded = message.encode().unwrap();
        let decoded = HasMt1Message::decode(&encoded).unwrap();
        assert_eq!(decoded.orbit, message.orbit);
        let decoded_records = decoded.orbit.unwrap().records;
        assert_eq!(decoded_records.len(), 2);
        assert_eq!(decoded_records[0].sat, sat1);
        assert_eq!(decoded_records[0].radial_m, Some(1.25));
        assert_eq!(decoded_records[1].sat, sat2);
        assert_eq!(decoded_records[1].radial_m, Some(-0.5));
    }

    #[test]
    fn orbit_block_with_unavailable_and_genuine_zero_round_trip() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let sat3 = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 10,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2, 3],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: Some(HasOrbitBlock {
                validity_interval: 2,
                records: vec![
                    // sat1: unavailable correction
                    HasOrbitCorrection {
                        sat: sat1,
                        iode: 11,
                        radial_m: None,
                        along_m: None,
                        cross_m: None,
                    },
                    // sat2: present non-zero correction
                    HasOrbitCorrection {
                        sat: sat2,
                        iode: 22,
                        radial_m: Some(1.25),
                        along_m: Some(-2.0),
                        cross_m: Some(3.0),
                    },
                    // sat3: genuine zero correction
                    HasOrbitCorrection {
                        sat: sat3,
                        iode: 33,
                        radial_m: Some(0.0),
                        along_m: Some(0.0),
                        cross_m: Some(0.0),
                    },
                ],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let encoded = message.encode().unwrap();
        let decoded = HasMt1Message::decode(&encoded).unwrap();
        assert_eq!(decoded.orbit, message.orbit);
        // The encoder pads to a byte boundary, so the decoded message carries
        // the padding bits the constructed one does not. They must be zero.
        assert!(
            decoded.padding_bits.iter().all(|bit| !bit),
            "padding must be zero bits"
        );
        let decoded_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..decoded.clone()
        };
        assert_eq!(decoded_without_padding, message);

        let records = decoded.orbit.unwrap().records;
        assert_eq!(records.len(), 3);

        // sat1: unavailable
        assert_eq!(records[0].sat, sat1);
        assert_eq!(records[0].iode, 11);
        assert_eq!(records[0].radial_m, None);
        assert_eq!(records[0].along_m, None);
        assert_eq!(records[0].cross_m, None);

        // sat2: present
        assert_eq!(records[1].sat, sat2);
        assert_eq!(records[1].iode, 22);
        assert_eq!(records[1].radial_m, Some(1.25));
        assert_eq!(records[1].along_m, Some(-2.0));
        assert_eq!(records[1].cross_m, Some(3.0));

        // sat3: genuine zero, not confused with unavailable
        assert_eq!(records[2].sat, sat3);
        assert_eq!(records[2].iode, 33);
        assert_eq!(records[2].radial_m, Some(0.0));
        assert_eq!(records[2].along_m, Some(0.0));
        assert_eq!(records[2].cross_m, Some(0.0));
    }

    #[test]
    fn clock_and_bias_blocks_with_unavailable_and_genuine_zero_round_trip() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let sat3 = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 20,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 2,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2, 3],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 1,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        correction_m: None,
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        correction_m: Some(0.5),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat3,
                        correction_m: Some(0.0),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 1,
                records: vec![
                    HasCodeBias {
                        sat: sat1,
                        signal_id: 0,
                        bias_m: None,
                    },
                    HasCodeBias {
                        sat: sat2,
                        signal_id: 0,
                        bias_m: Some(0.24),
                    },
                    HasCodeBias {
                        sat: sat3,
                        signal_id: 0,
                        bias_m: Some(0.0),
                    },
                ],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 1,
                records: vec![
                    HasPhaseBias {
                        sat: sat1,
                        signal_id: 0,
                        bias_cycles: None,
                        discontinuity_indicator: 1,
                    },
                    HasPhaseBias {
                        sat: sat2,
                        signal_id: 0,
                        bias_cycles: Some(1.25),
                        discontinuity_indicator: 2,
                    },
                    HasPhaseBias {
                        sat: sat3,
                        signal_id: 0,
                        bias_cycles: Some(0.0),
                        discontinuity_indicator: 0,
                    },
                ],
            }),
            padding_bits: Vec::new(),
        };

        let encoded = message.encode().unwrap();
        let decoded = HasMt1Message::decode(&encoded).unwrap();
        // The encoder pads to a byte boundary, so the decoded message carries
        // the padding bits the constructed one does not. They must be zero.
        assert!(
            decoded.padding_bits.iter().all(|bit| !bit),
            "padding must be zero bits"
        );
        let decoded_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..decoded.clone()
        };
        assert_eq!(decoded_without_padding, message);

        let clock_records = decoded.clock_full_set.unwrap().records;
        assert_eq!(clock_records[0].correction_m, None);
        assert!(!clock_records[0].do_not_use);
        assert_eq!(clock_records[1].correction_m, Some(0.5));
        assert!(!clock_records[1].do_not_use);
        assert_eq!(clock_records[2].correction_m, Some(0.0));
        assert!(!clock_records[2].do_not_use);

        let code_records = decoded.code_bias.unwrap().records;
        assert_eq!(code_records[0].bias_m, None);
        assert_eq!(code_records[1].bias_m, Some(0.24));
        assert_eq!(code_records[2].bias_m, Some(0.0));

        let phase_records = decoded.phase_bias.unwrap().records;
        assert_eq!(phase_records[0].bias_cycles, None);
        assert_eq!(phase_records[0].bias_m(), None);
        assert_eq!(phase_records[0].discontinuity_indicator, 1);
        assert_eq!(phase_records[1].bias_cycles, Some(1.25));
        assert_eq!(phase_records[1].discontinuity_indicator, 2);
        assert_eq!(phase_records[2].bias_cycles, Some(0.0));
        assert_eq!(phase_records[2].bias_m(), Some(0.0));
        assert_eq!(phase_records[2].discontinuity_indicator, 0);
    }

    #[test]
    fn clock_writer_refuses_omitted_satellite_by_name() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    correction_m: Some(0.5),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("G02"),
            "expected error naming omitted satellite G02, got: {err_msg}"
        );
    }

    #[test]
    fn code_bias_writer_refuses_omitted_cell_by_name() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1],
                    signals: vec![0, 9],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 0,
                // Only signal 0, omitting signal 9
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(0.10),
                }],
            }),
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("G01") && err_msg.contains("9"),
            "expected error naming G01 signal 9, got: {err_msg}"
        );
    }

    #[test]
    fn toh_reference_uses_current_or_previous_gst_hour() {
        let reception = crate::astro::time::model::GnssWeekTow::new(
            crate::astro::time::model::TimeScale::Gst,
            2400,
            3605.0,
        )
        .unwrap();
        let same_hour = has_mt1_reference_j2000_s(reception, 5).unwrap();
        let expected_same = f64::from(reception.week) * crate::constants::SECONDS_PER_WEEK + 3605.0
            - crate::constants::GPS_EPOCH_TO_J2000_S;
        assert_eq!(same_hour.to_bits(), expected_same.to_bits());

        let previous_hour = has_mt1_reference_j2000_s(reception, 3599).unwrap();
        assert_eq!(previous_hour.to_bits(), (expected_same - 6.0).to_bits());
    }

    #[test]
    fn clock_full_set_coarse_multiplier_round_trip() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        // Multiplier index 2 corresponds to 3.0x scaling (scale factor 0.0075 m).
        // 15.0 m / 0.0075 m = 2000 raw.
        // -7.5 m / 0.0075 m = -1000 raw.
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 42,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 2,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 2,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat1,
                        correction_m: Some(15.0),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat2,
                        correction_m: Some(-7.5),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let encoded = message.encode().unwrap();
        let decoded = HasMt1Message::decode(&encoded).unwrap();
        let decoded_clock = decoded.clock_full_set.unwrap();
        assert_eq!(
            decoded_clock.systems,
            vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 2,
            }]
        );
        assert_eq!(decoded_clock.validity_interval, 2);
        assert_eq!(decoded_clock.records.len(), 2);
        assert_eq!(decoded_clock.records[0].sat, sat1);
        assert_eq!(decoded_clock.records[0].correction_m, Some(15.0));
        assert_eq!(decoded_clock.records[1].sat, sat2);
        assert_eq!(decoded_clock.records[1].correction_m, Some(-7.5));
    }

    #[test]
    fn clock_subset_coarse_multiplier_round_trip() {
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        // Multiplier index 1 corresponds to 2.0x scaling (scale factor 0.0050 m).
        // Subset only provides correction for sat2 (12.5 m / 0.0050 m = 2500 raw).
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 100,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 2,
                iod_set_id: 3,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 3,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 1,
                }],
                records: vec![HasClockCorrection {
                    sat: sat2,
                    correction_m: Some(12.5),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let encoded = message.encode().unwrap();
        let decoded = HasMt1Message::decode(&encoded).unwrap();
        let decoded_clock = decoded.clock_subset.unwrap();
        assert_eq!(
            decoded_clock.systems,
            vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 1,
            }]
        );
        assert_eq!(decoded_clock.validity_interval, 3);
        assert_eq!(decoded_clock.records.len(), 1);
        assert_eq!(decoded_clock.records[0].sat, sat2);
        assert_eq!(decoded_clock.records[0].correction_m, Some(12.5));
    }

    #[test]
    fn clock_full_set_writer_refuses_out_of_range_correction_by_name() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        // At multiplier index 0 (1.0x, scale 0.0025 m), valid raw range is -4095..=4094,
        // which allows [-10.2375, 10.2350] m. A correction of 15.0 m cannot fit.
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    correction_m: Some(15.0),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("G01"),
            "expected error naming satellite G01, got: {err_msg}"
        );
        assert!(
            err_msg.contains("multiplier index 0"),
            "expected error naming multiplier index 0, got: {err_msg}"
        );
        assert!(
            err_msg.contains("10.235"),
            "expected error stating allowed range bound, got: {err_msg}"
        );
    }

    #[test]
    fn clock_subset_writer_refuses_out_of_range_correction_by_name() {
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        // At multiplier index 1 (2.0x, scale 0.0050 m), valid raw range is -4095..=4094,
        // which allows [-20.4750, 20.4700] m. A correction of -25.0 m cannot fit.
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 1,
                }],
                records: vec![HasClockCorrection {
                    sat: sat2,
                    correction_m: Some(-25.0),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("G02"),
            "expected error naming satellite G02, got: {err_msg}"
        );
        assert!(
            err_msg.contains("multiplier index 1"),
            "expected error naming multiplier index 1, got: {err_msg}"
        );
        assert!(
            err_msg.contains("20.475"),
            "expected error stating allowed range bound, got: {err_msg}"
        );
    }

    #[test]
    fn clock_writer_refuses_reserved_multiplier_index() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let message = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 4,
                }],
                records: vec![HasClockCorrection {
                    sat,
                    correction_m: Some(0.0),
                    do_not_use: false,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("multiplier index 4 for Gps is reserved"),
            "expected reserved multiplier error naming index, system and refusal, got: {err_msg}"
        );
    }

    #[test]
    fn clock_full_set_independent_bits_multi_system_multipliers() {
        let sat_gps1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat_gal2 = GnssSatelliteId::new(GnssSystem::Galileo, 2).unwrap();

        // Build raw bits independently:
        // Mask: GPS PRN 1, Galileo PRN 2.
        // Full set: GPS multiplier x1 (wire index 0, scale 0.0025 m), raw 1200 -> 3.0 m.
        //           Galileo multiplier x3 (wire index 2, scale 0.0075 m), raw -800 -> -6.0 m.
        let mut w = BitWriter::new();
        // Header (32 bits):
        w.push_u(100, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(true); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(1, 5); // mask_id
        w.push_u(1, 5); // iod_set_id

        // Mask block (138 bits):
        w.push_u(2, 4); // Nsys = 2
                        // System 0: GPS
        w.push_u(0, 4); // gnss_id GPS = 0
        w.push_u(1u64 << 39, 40); // sat_mask (PRN 1)
        w.push_u(1u64 << 15, 16); // sig_mask (sig 0)
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
                        // System 1: Galileo
        w.push_u(2, 4); // gnss_id Galileo = 2
        w.push_u(1u64 << 38, 40); // sat_mask (PRN 2)
        w.push_u(1u64 << 15, 16); // sig_mask (sig 0)
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
        w.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock full-set block (34 bits):
        w.push_u(2, 4); // VI = 2
        w.push_u(0, 2); // DCM GPS = wire index 0 (x1)
        w.push_u(2, 2); // DCM Galileo = wire index 2 (x3)
        w.push_i(1200, 13); // DCC GPS PRN 1 = 1200
        w.push_i(-800, 13); // DCC Galileo PRN 2 = -800

        let payload_bits_len = 32 + 138 + 34; // 204 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding independently constructed full-set bits");
        let decoded_clock = decoded
            .clock_full_set
            .as_ref()
            .expect("clock_full_set present");
        assert_eq!(decoded_clock.validity_interval, 2);
        assert_eq!(
            decoded_clock.systems,
            vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 2,
                },
            ]
        );
        assert_eq!(decoded_clock.records.len(), 2);
        assert_eq!(decoded_clock.records[0].sat, sat_gps1);
        assert_eq!(decoded_clock.records[0].correction_m, Some(3.0));
        assert!(!decoded_clock.records[0].do_not_use);
        assert_eq!(decoded_clock.records[1].sat, sat_gal2);
        assert_eq!(decoded_clock.records[1].correction_m, Some(-6.0));
        assert!(!decoded_clock.records[1].do_not_use);

        // Re-encode and compare all original meaningful payload bits
        let re_encoded = decoded
            .encode()
            .expect("re-encoding decoded full-set message");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "meaningful payload bit mismatch at bit index {bit}"
            );
        }
    }

    #[test]
    fn clock_subset_independent_bits_multi_system_multipliers() {
        let sat_gal4 = GnssSatelliteId::new(GnssSystem::Galileo, 4).unwrap();
        let sat_gps1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();

        // Build raw bits independently:
        // Mask Block order: GPS (PRN 1, 2) then Galileo (PRN 3, 4).
        // Transmitted Subset order: Galileo first, then GPS (differs from Mask Block order).
        // Galileo: DCM wire index 2 (x3, scale 0.0075 m), SatMsub 0 1 (PRN 3 unselected, PRN 4 selected), raw 400 -> 3.0 m.
        // GPS: DCM wire index 0 (x1, scale 0.0025 m), SatMsub 1 0 (PRN 1 selected, PRN 2 unselected), raw -500 -> -1.25 m.
        let mut w = BitWriter::new();
        // Header (32 bits):
        w.push_u(200, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(false); // clock_full_set
        w.push_flag(true); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(2, 5); // mask_id
        w.push_u(3, 5); // iod_set_id

        // Mask block (138 bits):
        w.push_u(2, 4); // Nsys = 2
                        // System 0 in Mask: GPS (PRN 1, 2)
        w.push_u(0, 4); // gnss_id GPS = 0
        w.push_u((1u64 << 39) | (1u64 << 38), 40); // sat_mask (PRN 1, 2)
        w.push_u(1u64 << 15, 16); // sig_mask (sig 0)
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
                        // System 1 in Mask: Galileo (PRN 3, 4)
        w.push_u(2, 4); // gnss_id Galileo = 2
        w.push_u((1u64 << 37) | (1u64 << 36), 40); // sat_mask (PRN 3, 4)
        w.push_u(1u64 << 15, 16); // sig_mask (sig 0)
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
        w.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock subset block (50 bits):
        w.push_u(3, 4); // VI = 3
        w.push_u(2, 4); // Nsys_sub = 2
                        // Subset GNSS 0: Galileo (transmitted before GPS)
        w.push_u(2, 4); // gnss_id = 2 (Galileo)
        w.push_u(2, 2); // DCM wire index 2 (x3)
        w.push_flag(false); // PRN 3 unselected
        w.push_flag(true); // PRN 4 selected
        w.push_i(400, 13); // DCC for Galileo PRN 4: 400 * 0.0075 = 3.0 m
                           // Subset GNSS 1: GPS
        w.push_u(0, 4); // gnss_id = 0 (GPS)
        w.push_u(0, 2); // DCM wire index 0 (x1)
        w.push_flag(true); // PRN 1 selected
        w.push_flag(false); // PRN 2 unselected
        w.push_i(-500, 13); // DCC for GPS PRN 1: -500 * 0.0025 = -1.25 m

        let payload_bits_len = 32 + 138 + 50; // 220 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding independently constructed subset bits");
        let decoded_clock = decoded.clock_subset.as_ref().expect("clock_subset present");
        assert_eq!(decoded_clock.validity_interval, 3);
        // Metadata order must match transmitted order (Galileo first, then GPS):
        assert_eq!(
            decoded_clock.systems,
            vec![
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 2,
                },
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
            ]
        );
        // Satellite records association and order: Galileo PRN 4 first, then GPS PRN 1:
        assert_eq!(decoded_clock.records.len(), 2);
        assert_eq!(decoded_clock.records[0].sat, sat_gal4);
        assert_eq!(decoded_clock.records[0].correction_m, Some(3.0));
        assert!(!decoded_clock.records[0].do_not_use);
        assert_eq!(decoded_clock.records[1].sat, sat_gps1);
        assert_eq!(decoded_clock.records[1].correction_m, Some(-1.25));
        assert!(!decoded_clock.records[1].do_not_use);

        // Re-encode and compare all original meaningful payload bits
        let re_encoded = decoded
            .encode()
            .expect("re-encoding decoded subset message");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "meaningful payload bit mismatch at bit index {bit}"
            );
        }
    }

    #[test]
    fn clock_subset_zero_selected_satellites_metadata_survives() {
        let sat_gps1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();

        // Part A: One system with corrections, one system with zero selected satellites (independent bits)
        let mut w_a = BitWriter::new();
        // Header (32 bits):
        w_a.push_u(50, 12); // toh_s
        w_a.push_flag(true); // mask
        w_a.push_flag(false); // orbit
        w_a.push_flag(false); // clock_full_set
        w_a.push_flag(true); // clock_subset
        w_a.push_flag(false); // code_bias
        w_a.push_flag(false); // phase_bias
        w_a.push_u(0, 4); // reserved
        w_a.push_u(1, 5); // mask_id
        w_a.push_u(1, 5); // iod_set_id

        // Mask block (138 bits): Nsys = 2 (GPS PRN 1, 2; Galileo PRN 1, 2)
        w_a.push_u(2, 4);
        // System 0: GPS
        w_a.push_u(0, 4);
        w_a.push_u((1u64 << 39) | (1u64 << 38), 40);
        w_a.push_u(1u64 << 15, 16);
        w_a.push_flag(false);
        w_a.push_u(0, 3);
        // System 1: Galileo
        w_a.push_u(2, 4);
        w_a.push_u((1u64 << 39) | (1u64 << 38), 40);
        w_a.push_u(1u64 << 15, 16);
        w_a.push_flag(false);
        w_a.push_u(0, 3);
        w_a.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock subset block (37 bits):
        w_a.push_u(2, 4); // VI = 2
        w_a.push_u(2, 4); // Nsys_sub = 2
                          // System 0: GPS (multiplier 0 -> 1.0x, scale 0.0025m)
        w_a.push_u(0, 4); // gnss_id = 0
        w_a.push_u(0, 2); // DCM = 0
        w_a.push_flag(true); // PRN 1 selected
        w_a.push_flag(false); // PRN 2 unselected
        w_a.push_i(400, 13); // DCC for GPS PRN 1: 400 * 0.0025 = 1.0 m
                             // System 1: Galileo (multiplier 3 -> 4.0x, scale 0.0100m, zero selected satellites)
        w_a.push_u(2, 4); // gnss_id = 2
        w_a.push_u(3, 2); // DCM = 3
        w_a.push_flag(false); // PRN 1 unselected
        w_a.push_flag(false); // PRN 2 unselected

        let payload_bits_a = 32 + 138 + 37; // 207 bits
        let raw_a = w_a.into_bytes();

        let decoded_a = HasMt1Message::decode(&raw_a)
            .expect("decoding independently constructed subset bits with empty Galileo selection");
        let clock_a = decoded_a
            .clock_subset
            .as_ref()
            .expect("clock_subset present");
        assert_eq!(
            clock_a.systems,
            vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 3,
                },
            ]
        );
        assert_eq!(clock_a.records.len(), 1);
        assert_eq!(clock_a.records[0].sat, sat_gps1);
        assert_eq!(clock_a.records[0].correction_m, Some(1.0));
        assert!(!clock_a.records[0].do_not_use);

        let re_encoded_a = decoded_a
            .encode()
            .expect("re-encoding subset with empty Galileo selection");
        for bit in 0..payload_bits_a {
            let raw_bit = (raw_a[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded_a[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "Part A meaningful payload bit mismatch at bit index {bit}"
            );
        }

        // Part B: All-empty selections across all systems (independent bits)
        let mut w_b = BitWriter::new();
        // Header (32 bits):
        w_b.push_u(60, 12);
        w_b.push_flag(true);
        w_b.push_flag(false);
        w_b.push_flag(false);
        w_b.push_flag(true);
        w_b.push_flag(false);
        w_b.push_flag(false);
        w_b.push_u(0, 4);
        w_b.push_u(1, 5);
        w_b.push_u(1, 5);

        // Mask block (138 bits): GPS (PRN 1, 2), Galileo (PRN 1)
        w_b.push_u(2, 4);
        w_b.push_u(0, 4);
        w_b.push_u((1u64 << 39) | (1u64 << 38), 40);
        w_b.push_u(1u64 << 15, 16);
        w_b.push_flag(false);
        w_b.push_u(0, 3);
        w_b.push_u(2, 4);
        w_b.push_u(1u64 << 39, 40);
        w_b.push_u(1u64 << 15, 16);
        w_b.push_flag(false);
        w_b.push_u(0, 3);
        w_b.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock subset block (23 bits): all selected flags false
        w_b.push_u(1, 4); // VI = 1
        w_b.push_u(2, 4); // Nsys_sub = 2
        w_b.push_u(0, 4); // GPS
        w_b.push_u(1, 2); // DCM = 1
        w_b.push_flag(false); // PRN 1
        w_b.push_flag(false); // PRN 2
        w_b.push_u(2, 4); // Galileo
        w_b.push_u(2, 2); // DCM = 2
        w_b.push_flag(false); // PRN 1

        let payload_bits_b = 32 + 138 + 23; // 193 bits
        let raw_b = w_b.into_bytes();

        let decoded_b = HasMt1Message::decode(&raw_b)
            .expect("decoding independently constructed all-empty subset selection");
        let clock_b = decoded_b
            .clock_subset
            .as_ref()
            .expect("clock_subset present");
        assert_eq!(
            clock_b.systems,
            vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 1,
                },
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 2,
                },
            ]
        );
        assert!(clock_b.records.is_empty());

        let re_encoded_b = decoded_b
            .encode()
            .expect("re-encoding all-empty subset selection");
        for bit in 0..payload_bits_b {
            let raw_bit = (raw_b[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded_b[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "Part B meaningful payload bit mismatch at bit index {bit}"
            );
        }
    }

    #[test]
    fn clock_writer_refuses_out_of_range_per_system_scale_and_preserves_boundaries() {
        let sat_gps = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat_gal = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();

        // GPS at multiplier index 0 (1.0x, scale 0.0025 m): max valid correction is 4094 * 0.0025 = 10.2350 m.
        // Galileo at multiplier index 1 (2.0x, scale 0.0050 m): max valid correction is 4094 * 0.0050 = 20.4700 m.
        // 15.0 m fits Galileo (raw 3000) but exceeds GPS (raw 6000).
        let message_invalid = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![
                    HasGnssMask {
                        system: GnssSystem::Gps,
                        satellites: vec![1],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                    HasGnssMask {
                        system: GnssSystem::Galileo,
                        satellites: vec![1],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                ],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![
                    HasClockSystem {
                        system: GnssSystem::Gps,
                        multiplier_index: 0,
                    },
                    HasClockSystem {
                        system: GnssSystem::Galileo,
                        multiplier_index: 1,
                    },
                ],
                records: vec![
                    HasClockCorrection {
                        sat: sat_gps,
                        correction_m: Some(15.0),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat_gal,
                        correction_m: Some(15.0),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let err = message_invalid.encode().unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("G01"),
            "expected error naming satellite G01, got: {err_msg}"
        );
        assert!(
            err_msg.contains("Gps"),
            "expected error naming system Gps, got: {err_msg}"
        );
        assert!(
            err_msg.contains("multiplier index 0"),
            "expected error naming multiplier index 0, got: {err_msg}"
        );
        assert!(
            err_msg.contains("10.235"),
            "expected error stating allowed range bound, got: {err_msg}"
        );

        // Valid boundary test:
        // GPS at index 0 raw boundary: +4094 -> 10.2350 m, -4095 -> -10.2375 m.
        // Galileo at index 1 raw boundary: +4094 -> 20.4700 m, -4095 -> -20.4750 m.
        let message_boundary = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(HasMaskBlock {
                systems: vec![
                    HasGnssMask {
                        system: GnssSystem::Gps,
                        satellites: vec![1],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                    HasGnssMask {
                        system: GnssSystem::Galileo,
                        satellites: vec![1],
                        signals: vec![0],
                        cell_mask: None,
                        nav_message: 0,
                    },
                ],
                reserved: 0,
            }),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![
                    HasClockSystem {
                        system: GnssSystem::Gps,
                        multiplier_index: 0,
                    },
                    HasClockSystem {
                        system: GnssSystem::Galileo,
                        multiplier_index: 1,
                    },
                ],
                records: vec![
                    HasClockCorrection {
                        sat: sat_gps,
                        correction_m: Some(10.2350),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat_gal,
                        correction_m: Some(-20.4750),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        let encoded_boundary = message_boundary.encode().expect("encoding boundary values");
        let decoded_boundary =
            HasMt1Message::decode(&encoded_boundary).expect("decoding boundary values");
        let decoded_clock = decoded_boundary
            .clock_full_set
            .expect("clock_full_set present");
        assert_eq!(decoded_clock.records[0].correction_m, Some(10.2350));
        assert!(!decoded_clock.records[0].do_not_use);
        assert_eq!(decoded_clock.records[1].correction_m, Some(-20.4750));
        assert!(!decoded_clock.records[1].do_not_use);
    }

    #[test]
    fn clock_full_set_writer_metadata_validation_errors() {
        let sat_gps = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat_gal = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        let mask = HasMaskBlock {
            systems: vec![
                HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
                HasGnssMask {
                    system: GnssSystem::Galileo,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
            ],
            reserved: 0,
        };

        // Missing metadata: mask has [GPS, Galileo], clock has only [GPS]
        let mut msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask.clone()),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![
                    HasClockCorrection {
                        sat: sat_gps,
                        correction_m: Some(0.0),
                        do_not_use: false,
                    },
                    HasClockCorrection {
                        sat: sat_gal,
                        correction_m: Some(0.0),
                        do_not_use: false,
                    },
                ],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("missing HAS clock metadata for Galileo"),
            "got: {err}"
        );

        // Extraneous metadata: mask has only [GPS], clock has [GPS, Galileo]
        let mut single_mask = mask.clone();
        single_mask.systems.truncate(1);
        msg.mask = Some(single_mask);
        msg.clock_full_set = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 0,
                },
            ],
            records: vec![HasClockCorrection {
                sat: sat_gps,
                correction_m: Some(0.0),
                do_not_use: false,
            }],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("extraneous HAS clock metadata for Galileo"),
            "got: {err}"
        );

        // Duplicate metadata: clock has [GPS, GPS]
        msg.mask = Some(mask.clone());
        msg.clock_full_set = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
            ],
            records: vec![
                HasClockCorrection {
                    sat: sat_gps,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: sat_gal,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
            ],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("duplicate HAS clock metadata for Gps"),
            "got: {err}"
        );

        // Metadata out of mask order: mask has [GPS, Galileo], clock has
        // [Galileo, GPS]. Each entry names its system, so each multiplier
        // index is written at its system's mask position: the same bytes as
        // the block with metadata in mask order.
        let gps_meta = HasClockSystem {
            system: GnssSystem::Gps,
            multiplier_index: 1,
        };
        let gal_meta = HasClockSystem {
            system: GnssSystem::Galileo,
            multiplier_index: 2,
        };
        msg.clock_full_set = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![gal_meta, gps_meta],
            records: vec![
                HasClockCorrection {
                    sat: sat_gps,
                    correction_m: Some(10.0 * HAS_CLOCK_SCALE_M * 2.0),
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: sat_gal,
                    correction_m: Some(-10.0 * HAS_CLOCK_SCALE_M * 3.0),
                    do_not_use: false,
                },
            ],
        });
        let reordered = msg.encode().expect("metadata out of mask order encodes");
        msg.clock_full_set.as_mut().unwrap().systems = vec![gps_meta, gal_meta];
        assert_eq!(
            msg.encode().expect("metadata in mask order encodes"),
            reordered
        );
        let decoded = HasMt1Message::decode(&reordered).unwrap();
        assert_eq!(
            decoded.clock_full_set.as_ref().unwrap().systems,
            vec![gps_meta, gal_meta]
        );

        // Invalid-index metadata: multiplier index 4
        msg.clock_full_set = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 4,
                },
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 0,
                },
            ],
            records: vec![
                HasClockCorrection {
                    sat: sat_gps,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: sat_gal,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
            ],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("multiplier index 4 for Gps is reserved"),
            "got: {err}"
        );
    }

    #[test]
    fn clock_subset_writer_validation_errors() {
        let sat_gps1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat_gps2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let sat_gal1 = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        let mask = HasMaskBlock {
            systems: vec![
                HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
                HasGnssMask {
                    system: GnssSystem::Galileo,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
            ],
            reserved: 0,
        };

        let mut msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };

        // Empty systems with empty records is valid syntax, not an error: Table 32 gives
        // Nsys_sub a plain 4-bit range with no reservation at zero, so a subset block that
        // corrects nothing still encodes and round-trips losslessly.
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: Vec::new(),
            records: Vec::new(),
        });
        let enc = msg.encode().expect("Nsys_sub = 0 is valid syntax");
        let dec = HasMt1Message::decode(&enc).expect("Nsys_sub = 0 decodes");
        assert_eq!(dec.header, msg.header);
        assert_eq!(dec.mask, msg.mask);
        assert_eq!(
            dec.clock_subset, msg.clock_subset,
            "zero-subset block round-trips losslessly"
        );
        let sub = dec.clock_subset.as_ref().expect("subset block retained");
        assert_eq!(sub.validity_interval, 0);
        assert!(sub.systems.is_empty());
        assert!(sub.records.is_empty());

        // Records present without any system metadata is a genuine inconsistency: every
        // record must be attributable to a transmitted system entry.
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: Vec::new(),
            records: vec![HasClockCorrection {
                sat: sat_gps1,
                correction_m: Some(0.0),
                do_not_use: false,
            }],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("records present but systems list is empty"),
            "got: {err}"
        );

        // Duplicate system metadata
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 1,
                },
            ],
            records: Vec::new(),
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("duplicate HAS clock subset system Gps"),
            "got: {err}"
        );

        // System absent from mask
        let mut single_mask = msg.mask.clone().unwrap();
        single_mask.systems.truncate(1); // only GPS
        msg.mask = Some(single_mask);
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![HasClockSystem {
                system: GnssSystem::Galileo,
                multiplier_index: 0,
            }],
            records: Vec::new(),
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string().contains("Galileo is not declared in mask"),
            "got: {err}"
        );

        // Restore two-system mask
        msg.mask = Some(HasMaskBlock {
            systems: vec![
                HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
                HasGnssMask {
                    system: GnssSystem::Galileo,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
            ],
            reserved: 0,
        });

        // Invalid multiplier index
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 5,
            }],
            records: Vec::new(),
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("multiplier index 5 for Gps is reserved"),
            "got: {err}"
        );

        // Record lacking corresponding metadata
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 0,
            }],
            records: vec![HasClockCorrection {
                sat: sat_gal1,
                correction_m: Some(0.0),
                do_not_use: false,
            }],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("satellite E01 has no corresponding system metadata"),
            "got: {err}"
        );

        // Duplicate satellite record
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 0,
            }],
            records: vec![
                HasClockCorrection {
                    sat: sat_gps1,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: sat_gps1,
                    correction_m: Some(1.0),
                    do_not_use: false,
                },
            ],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("duplicate HAS clock subset satellite G01"),
            "got: {err}"
        );

        // Records out of mask order within a system are written at their mask
        // positions: the same bytes as the block in mask order.
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 0,
            }],
            records: vec![
                HasClockCorrection {
                    sat: sat_gps2,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: sat_gps1,
                    correction_m: Some(1.0),
                    do_not_use: false,
                },
            ],
        });
        let reordered = msg.encode().expect("reordered subset records encode");
        msg.clock_subset.as_mut().unwrap().records.reverse();
        assert_eq!(msg.encode().expect("mask-order subset encodes"), reordered);

        // Records out of the transmitted system order are written in the
        // order `systems` gives: the same bytes as the block in that order.
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
            ],
            records: vec![
                HasClockCorrection {
                    sat: sat_gps1,
                    correction_m: Some(0.0),
                    do_not_use: false,
                },
                HasClockCorrection {
                    sat: sat_gal1,
                    correction_m: Some(1.0),
                    do_not_use: false,
                },
            ],
        });
        let reordered = msg.encode().expect("records across systems encode");
        msg.clock_subset.as_mut().unwrap().records.reverse();
        assert_eq!(
            msg.encode().expect("transmitted-order subset encodes"),
            reordered
        );
        let decoded = HasMt1Message::decode(&reordered).unwrap();
        let subset = decoded.clock_subset.unwrap();
        assert_eq!(subset.records[0].sat, sat_gal1);
        assert_eq!(subset.records[1].sat, sat_gps1);

        // A satellite the mask does not declare is refused by name.
        msg.clock_subset = Some(HasClockBlock {
            validity_interval: 0,
            systems: vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 0,
            }],
            records: vec![HasClockCorrection {
                sat: GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap(),
                correction_m: Some(0.0),
                do_not_use: false,
            }],
        });
        let err = msg.encode().unwrap_err();
        assert!(
            err.to_string()
                .contains("satellite G03 is not declared in mask"),
            "got: {err}"
        );
    }

    #[test]
    fn clock_subset_reader_rejects_duplicate_and_unknown_gnss_ids() {
        // Construct raw bits for subset message with duplicate GNSS ID (GPS twice)
        let mut w_dup = BitWriter::new();
        w_dup.push_u(10, 12); // toh_s
        w_dup.push_flag(true); // mask
        w_dup.push_flag(false); // orbit
        w_dup.push_flag(false); // clock_full_set
        w_dup.push_flag(true); // clock_subset
        w_dup.push_flag(false); // code_bias
        w_dup.push_flag(false); // phase_bias
        w_dup.push_u(0, 4); // reserved
        w_dup.push_u(1, 5); // mask_id
        w_dup.push_u(1, 5); // iod_set_id

        // Mask with GPS
        w_dup.push_u(1, 4); // Nsys = 1
        w_dup.push_u(0, 4); // gnss_id = 0 (GPS)
        w_dup.push_u(1u64 << 39, 40); // sat_mask PRN 1
        w_dup.push_u(1u64 << 15, 16); // sig_mask
        w_dup.push_flag(false); // cell_mask = None
        w_dup.push_u(0, 3); // nav_message
        w_dup.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock subset: Nsys_sub = 2, both GPS (gnss_id = 0)
        w_dup.push_u(0, 4); // VI = 0
        w_dup.push_u(2, 4); // Nsys_sub = 2
        w_dup.push_u(0, 4); // gnss_id = 0
        w_dup.push_u(0, 2); // DCM = 0
        w_dup.push_flag(false); // sat 1 not selected
        w_dup.push_u(0, 4); // duplicate gnss_id = 0
        w_dup.push_u(0, 2); // DCM = 0
        w_dup.push_flag(false); // sat 1 not selected

        let raw_dup = w_dup.into_bytes();
        let err = HasMt1Message::decode(&raw_dup).unwrap_err();
        assert!(
            err.to_string()
                .contains("duplicate HAS clock subset GNSS ID 0"),
            "got: {err}"
        );

        // Construct raw bits for subset message with unknown GNSS ID (gnss_id = 5)
        let mut w_unk = BitWriter::new();
        w_unk.push_u(10, 12);
        w_unk.push_flag(true);
        w_unk.push_flag(false);
        w_unk.push_flag(false);
        w_unk.push_flag(true);
        w_unk.push_flag(false);
        w_unk.push_flag(false);
        w_unk.push_u(0, 4);
        w_unk.push_u(1, 5);
        w_unk.push_u(1, 5);

        // Mask with GPS
        w_unk.push_u(1, 4);
        w_unk.push_u(0, 4);
        w_unk.push_u(1u64 << 39, 40);
        w_unk.push_u(1u64 << 15, 16);
        w_unk.push_flag(false);
        w_unk.push_u(0, 3);
        w_unk.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock subset: Nsys_sub = 1, gnss_id = 5 (unknown)
        w_unk.push_u(0, 4); // VI
        w_unk.push_u(1, 4); // Nsys_sub = 1
        w_unk.push_u(5, 4); // gnss_id = 5 (unsupported)

        let raw_unk = w_unk.into_bytes();
        let err = HasMt1Message::decode(&raw_unk).unwrap_err();
        assert!(
            err.to_string().contains("unsupported HAS GNSS id 5"),
            "got: {err}"
        );
    }

    #[test]
    fn clock_full_set_independent_bits_empty_system_satellites_retains_dcm() {
        let sat_gps1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();

        // Construct raw bits:
        // Mask Block: Nsys = 2. GPS has PRN 1 (sat_mask 1<<39). Galileo has NO satellites (sat_mask = 0).
        // Clock full-set: GPS DCM = 0 (1.0x), Galileo DCM = 3 (4.0x).
        // GPS PRN 1 has DCC = 400 (1.0 m). Galileo has 0 satellites -> 0 DCC bits.
        // The Galileo 2-bit DCM multiplier still occupies its slot.
        let mut w = BitWriter::new();
        // Header (32 bits):
        w.push_u(10, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(true); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(1, 5); // mask_id
        w.push_u(1, 5); // iod_set_id

        // Mask block (138 bits): Nsys = 2
        w.push_u(2, 4);
        // System 0: GPS (PRN 1)
        w.push_u(0, 4); // GPS
        w.push_u(1u64 << 39, 40); // PRN 1
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
                        // System 1: Galileo (0 satellites)
        w.push_u(2, 4); // Galileo
        w.push_u(0, 40); // 0 satellites
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
        w.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock full-set block (21 bits):
        w.push_u(2, 4); // VI = 2
        w.push_u(0, 2); // GPS DCM index 0
        w.push_u(3, 2); // Galileo DCM index 3 (retained despite 0 satellites in mask)
        w.push_i(400, 13); // DCC GPS PRN 1: 400 * 0.0025 = 1.0 m

        let payload_bits_len = 32 + 138 + 21; // 191 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding full-set message with empty Galileo satellites");
        let decoded_clock = decoded
            .clock_full_set
            .as_ref()
            .expect("clock_full_set present");
        assert_eq!(decoded_clock.validity_interval, 2);
        assert_eq!(
            decoded_clock.systems,
            vec![
                HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                },
                HasClockSystem {
                    system: GnssSystem::Galileo,
                    multiplier_index: 3,
                },
            ]
        );
        assert_eq!(decoded_clock.records.len(), 1);
        assert_eq!(decoded_clock.records[0].sat, sat_gps1);
        assert_eq!(decoded_clock.records[0].correction_m, Some(1.0));
        assert!(!decoded_clock.records[0].do_not_use);

        let re_encoded = decoded
            .encode()
            .expect("re-encoding full-set message with empty Galileo satellites");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "meaningful payload bit mismatch at bit index {bit}"
            );
        }
    }

    #[test]
    fn clock_full_set_independent_bits_sentinel_distinction_and_round_trip() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let sat3 = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();

        // Independently construct raw bits with all three states per Galileo HAS SIS ICD Table 31:
        // - Sat 1: -4096 (data unavailable sentinel)
        // - Sat 2: +4095 (satellite shall not be used sentinel)
        // - Sat 3: -400 (ordinary numeric correction: -400 * 0.0025 m = -1.0 m)
        let mut w = BitWriter::new();
        // Header (32 bits):
        w.push_u(100, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(true); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(1, 5); // mask_id
        w.push_u(1, 5); // iod_set_id

        // Mask block (74 bits): GPS PRNs 1, 2, 3
        w.push_u(1, 4); // Nsys = 1
        w.push_u(0, 4); // GPS
        w.push_u((1u64 << 39) | (1u64 << 38) | (1u64 << 37), 40); // PRN 1, 2, 3
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
        w.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock full-set block (45 bits):
        w.push_u(2, 4); // VI = 2
        w.push_u(0, 2); // DCM = 0 (1.0x, scale 0.0025 m)
        w.push_i(-4096, 13); // DCC Sat 1: -4096 (unavailable)
        w.push_i(4095, 13); // DCC Sat 2: +4095 (do not use)
        w.push_i(-400, 13); // DCC Sat 3: -400 (-1.0 m)

        let payload_bits_len = 32 + 74 + 45; // 151 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding independently constructed full-set sentinel bits");
        let decoded_clock = decoded
            .clock_full_set
            .as_ref()
            .expect("clock_full_set present");
        assert_eq!(decoded_clock.validity_interval, 2);
        assert_eq!(decoded_clock.records.len(), 3);

        // Sat 1: data unavailable
        assert_eq!(decoded_clock.records[0].sat, sat1);
        assert_eq!(decoded_clock.records[0].correction_m, None);
        assert!(!decoded_clock.records[0].do_not_use);

        // Sat 2: satellite shall not be used
        assert_eq!(decoded_clock.records[1].sat, sat2);
        assert_eq!(decoded_clock.records[1].correction_m, None);
        assert!(decoded_clock.records[1].do_not_use);

        // Sat 3: available numeric correction
        assert_eq!(decoded_clock.records[2].sat, sat3);
        assert_eq!(decoded_clock.records[2].correction_m, Some(-1.0));
        assert!(!decoded_clock.records[2].do_not_use);

        // Re-encode and verify all 145 meaningful payload bits match identically
        let re_encoded = decoded
            .encode()
            .expect("re-encoding full-set sentinel message");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "meaningful payload bit mismatch at bit index {bit}"
            );
        }
    }

    #[test]
    fn clock_subset_independent_bits_sentinel_distinction_and_round_trip() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat2 = GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap();
        let sat3 = GnssSatelliteId::new(GnssSystem::Gps, 3).unwrap();

        // Independently construct raw bits for subset block per Galileo HAS SIS ICD Table 34:
        // - Sat 1: -4096 (data unavailable sentinel)
        // - Sat 2: +4095 (satellite shall not be used sentinel)
        // - Sat 3: +500 (ordinary numeric correction: +500 * 0.0050 m = 2.5 m at DCM = 1)
        let mut w = BitWriter::new();
        // Header (32 bits):
        w.push_u(150, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(false); // clock_full_set
        w.push_flag(true); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(1, 5); // mask_id
        w.push_u(1, 5); // iod_set_id

        // Mask block (74 bits): GPS PRNs 1, 2, 3
        w.push_u(1, 4); // Nsys = 1
        w.push_u(0, 4); // GPS
        w.push_u((1u64 << 39) | (1u64 << 38) | (1u64 << 37), 40); // PRN 1, 2, 3
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
        w.push_u(0, 6); // reserved (6 bits per Table 15)

        // Clock subset block (56 bits):
        w.push_u(2, 4); // VI = 2
        w.push_u(1, 4); // Nsys_sub = 1
        w.push_u(0, 4); // gnss_id = 0 (GPS)
        w.push_u(1, 2); // DCM = 1 (2.0x, scale 0.0050 m)
        w.push_flag(true); // PRN 1 selected
        w.push_flag(true); // PRN 2 selected
        w.push_flag(true); // PRN 3 selected
        w.push_i(-4096, 13); // DCC Sat 1: -4096 (unavailable)
        w.push_i(4095, 13); // DCC Sat 2: +4095 (do not use)
        w.push_i(500, 13); // DCC Sat 3: 500 (2.5 m)

        let payload_bits_len = 32 + 74 + 56; // 162 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding independently constructed subset sentinel bits");
        let decoded_clock = decoded.clock_subset.as_ref().expect("clock_subset present");
        assert_eq!(decoded_clock.validity_interval, 2);
        assert_eq!(decoded_clock.records.len(), 3);

        // Sat 1: data unavailable
        assert_eq!(decoded_clock.records[0].sat, sat1);
        assert_eq!(decoded_clock.records[0].correction_m, None);
        assert!(!decoded_clock.records[0].do_not_use);

        // Sat 2: satellite shall not be used
        assert_eq!(decoded_clock.records[1].sat, sat2);
        assert_eq!(decoded_clock.records[1].correction_m, None);
        assert!(decoded_clock.records[1].do_not_use);

        // Sat 3: available numeric correction
        assert_eq!(decoded_clock.records[2].sat, sat3);
        assert_eq!(decoded_clock.records[2].correction_m, Some(2.5));
        assert!(!decoded_clock.records[2].do_not_use);

        // Re-encode and verify all 156 meaningful payload bits match identically
        let re_encoded = decoded
            .encode()
            .expect("re-encoding subset sentinel message");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(
                raw_bit, re_bit,
                "meaningful payload bit mismatch at bit index {bit}"
            );
        }
    }

    #[test]
    fn clock_writer_refuses_contradictory_some_and_do_not_use() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let mask = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };

        // Full-set refusal
        let msg_full = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: true,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask.clone()),
            orbit: None,
            clock_full_set: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    correction_m: Some(1.0),
                    do_not_use: true,
                }],
            }),
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err_full = msg_full.encode().unwrap_err();
        let err_msg_full = err_full.to_string();
        assert!(
            err_msg_full.contains("contradictory"),
            "expected error mentioning contradictory, got: {err_msg_full}"
        );
        assert!(
            err_msg_full.contains("G01"),
            "expected error naming satellite G01, got: {err_msg_full}"
        );

        // Subset refusal
        let msg_sub = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 0,
                systems: vec![HasClockSystem {
                    system: GnssSystem::Gps,
                    multiplier_index: 0,
                }],
                records: vec![HasClockCorrection {
                    sat: sat1,
                    correction_m: Some(1.0),
                    do_not_use: true,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err_sub = msg_sub.encode().unwrap_err();
        let err_msg_sub = err_sub.to_string();
        assert!(
            err_msg_sub.contains("contradictory"),
            "expected error mentioning contradictory, got: {err_msg_sub}"
        );
        assert!(
            err_msg_sub.contains("G01"),
            "expected error naming satellite G01, got: {err_msg_sub}"
        );
    }

    #[test]
    fn mask_writer_refuses_overrange_reserved() {
        let mask = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 64,
        };
        let mut msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let err_64 = msg.encode().unwrap_err();
        assert!(
            err_64
                .to_string()
                .contains("mask reserved value 64 exceeds 6-bit maximum (63)"),
            "got: {err_64}"
        );

        msg.mask.as_mut().unwrap().reserved = 255;
        let err_255 = msg.encode().unwrap_err();
        assert!(
            err_255
                .to_string()
                .contains("mask reserved value 255 exceeds 6-bit maximum (63)"),
            "got: {err_255}"
        );

        msg.mask.as_mut().unwrap().reserved = 63;
        assert!(msg.encode().is_ok());
    }

    #[test]
    fn mask_truncated_reserved_block_fails_decode() {
        let mut w = BitWriter::new();
        // Header (32 bits)
        w.push_u(10, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(false); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // header reserved
        w.push_u(1, 5); // mask_id
        w.push_u(1, 5); // iod_set_id

        // Mask block: Nsys = 1 (4 bits) + GPS mask (64 bits) = 68 bits.
        // Total so far: 32 + 68 = 100 bits.
        w.push_u(1, 4); // Nsys = 1
        w.push_u(0, 4); // gnss_id GPS = 0
        w.push_u(1u64 << 39, 40); // sat_mask (PRN 1)
        w.push_u(1u64 << 15, 16); // sig_mask (sig 0)
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message

        // Push only 4 bits (needs 6 reserved bits per Table 15).
        // Total bits = 104 bits = exactly 13 bytes.
        w.push_u(0b1010, 4);

        let truncated_13_bytes = w.into_bytes();
        assert_eq!(truncated_13_bytes.len(), 13);
        assert_eq!(truncated_13_bytes.len() * 8, 104);

        // Buffer has 104 bits; reading header + GNSS mask consumes 100 bits.
        // Reading 6 reserved bits requires bit index 106, which exceeds the buffer.
        // Byte padding cannot satisfy it because the slice itself ends at 104 bits.
        let err = HasMt1Message::decode(&truncated_13_bytes).unwrap_err();
        assert_eq!(
            err,
            Error::Parse("RTCM body truncated: need 6 more bits, 4 remain".to_string())
        );
    }

    #[test]
    fn mask_reserved_and_tail_padding_roundtrip_preservation() {
        let msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 500,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 5,
                iod_set_id: 7,
            },
            mask: Some(HasMaskBlock {
                systems: vec![HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1, 5],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                }],
                reserved: 42, // Non-zero reserved bits (0b101010)
            }),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: vec![true, false, true, true, false, true],
        };

        let encoded = msg
            .encode()
            .expect("encoding with nonzero reserved and explicit tail padding");
        let decoded = HasMt1Message::decode(&encoded).expect("decoding message");

        assert_eq!(decoded.mask.as_ref().unwrap().reserved, 42);
        assert_eq!(
            decoded.padding_bits,
            vec![true, false, true, true, false, true]
        );
        assert_eq!(decoded, msg);

        let re_encoded = decoded.encode().expect("re-encoding message");
        assert_eq!(re_encoded, encoded);
    }

    #[test]
    fn mask_one_system_nonzero_reserved_followed_by_orbit() {
        let sat1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let mut w = BitWriter::new();
        // Header (32 bits)
        w.push_u(123, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(true); // orbit
        w.push_flag(false); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(1, 5); // mask_id
        w.push_u(2, 5); // iod_set_id

        // Mask block (74 bits): Nsys = 1, GPS PRN 1, reserved = 45
        w.push_u(1, 4); // Nsys = 1
        w.push_u(0, 4); // GPS
        w.push_u(1u64 << 39, 40); // PRN 1
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
        w.push_u(45, 6); // reserved = 45 (0b101101)

        // Orbit block (49 bits): VI = 2 (4 bits), IODE = 42 (8 bits for GPS),
        // radial = 10 (13 bits), along = -20 (12 bits), cross = 30 (12 bits)
        w.push_u(2, 4); // VI = 2
        w.push_u(42, 8); // IODE = 42
        w.push_i(10, 13); // radial: 10 * 0.0025 = 0.025 m
        w.push_i(-20, 12); // along: -20 * 0.008 = -0.16 m
        w.push_i(30, 12); // cross: 30 * 0.008 = 0.24 m

        let payload_bits_len = 32 + 74 + 49; // 155 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding one-system nonzero reserved + orbit");
        let mask = decoded.mask.as_ref().expect("mask present");
        assert_eq!(mask.reserved, 45);

        let orbit = decoded.orbit.as_ref().expect("orbit present");
        assert_eq!(orbit.validity_interval, 2);
        assert_eq!(orbit.records.len(), 1);
        assert_eq!(orbit.records[0].sat, sat1);
        assert_eq!(orbit.records[0].iode, 42);
        assert_eq!(orbit.records[0].radial_m, Some(0.025));
        assert_eq!(orbit.records[0].along_m, Some(-0.16));
        assert_eq!(orbit.records[0].cross_m, Some(0.24));

        let re_encoded = decoded.encode().expect("re-encoding one-system message");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(raw_bit, re_bit, "bit mismatch at {bit}");
        }
    }

    #[test]
    fn mask_two_system_nonzero_reserved_followed_by_clock() {
        let sat_gps1 = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let sat_gal2 = GnssSatelliteId::new(GnssSystem::Galileo, 2).unwrap();
        let mut w = BitWriter::new();
        // Header (32 bits)
        w.push_u(456, 12); // toh_s
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(true); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(false); // phase_bias
        w.push_u(0, 4); // reserved
        w.push_u(3, 5); // mask_id
        w.push_u(4, 5); // iod_set_id

        // Mask block (138 bits): Nsys = 2, GPS PRN 1, Galileo PRN 2, reserved = 26
        w.push_u(2, 4); // Nsys = 2
                        // System 0: GPS
        w.push_u(0, 4); // GPS
        w.push_u(1u64 << 39, 40); // PRN 1
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
                        // System 1: Galileo
        w.push_u(2, 4); // Galileo
        w.push_u(1u64 << 38, 40); // PRN 2
        w.push_u(1u64 << 15, 16); // sig 0
        w.push_flag(false); // cell_mask = None
        w.push_u(0, 3); // nav_message
                        // Reserved bits written ONCE after all GNSS masks per Table 15
        w.push_u(26, 6); // reserved = 26 (0b011010)

        // Clock full-set block (34 bits):
        w.push_u(3, 4); // VI = 3
        w.push_u(0, 2); // DCM GPS = wire index 0 (1.0x, scale 0.0025 m)
        w.push_u(1, 2); // DCM Galileo = wire index 1 (2.0x, scale 0.0050 m)
        w.push_i(800, 13); // DCC GPS PRN 1: 800 * 0.0025 = 2.0 m
        w.push_i(-600, 13); // DCC Galileo PRN 2: -600 * 0.0050 = -3.0 m

        let payload_bits_len = 32 + 138 + 34; // 204 bits
        let raw_bytes = w.into_bytes();

        let decoded = HasMt1Message::decode(&raw_bytes)
            .expect("decoding two-system nonzero reserved + clock");
        let mask = decoded.mask.as_ref().expect("mask present");
        assert_eq!(mask.reserved, 26);

        let clock = decoded
            .clock_full_set
            .as_ref()
            .expect("clock_full_set present");
        assert_eq!(clock.validity_interval, 3);
        assert_eq!(clock.records.len(), 2);
        assert_eq!(clock.records[0].sat, sat_gps1);
        assert_eq!(clock.records[0].correction_m, Some(2.0));
        assert_eq!(clock.records[1].sat, sat_gal2);
        assert_eq!(clock.records[1].correction_m, Some(-3.0));

        let re_encoded = decoded.encode().expect("re-encoding two-system message");
        for bit in 0..payload_bits_len {
            let raw_bit = (raw_bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            let re_bit = (re_encoded[bit / 8] >> (7 - (bit % 8))) & 1;
            assert_eq!(raw_bit, re_bit, "bit mismatch at {bit}");
        }
    }

    const REFERENCE_CORPUS_PACKET_1_HEX: &str =
        "8fcc82c6207ffd7bee008140dfffffffefbedbfffffe0befc8a79b41241000a38010fdafd010817000b\
         fa1e3cb009c00ecc03ef5703405faaff0fdd007f7c0b40000805e22e0501bf95fd7ff2500268350297\
         77a540f804413fce0a7fae26010fdffe094f40013048857ffc053eb363dce07dfca1dfebfc000b80fe\
         d07f7c1b4fe4c5701e07bdaa177fe002059fd6ff22cfdb840fe88580d40080487019dfd606234002fe\
         c0060afef82381a93800becffc0200097e77f8841fb8022009083fe8047fd8107e9c09002820ff480e\
         ff384003100e008078020057ff2107ee803bfb420fdf81881c03ffb5006029083fa3fe80261000e7f6\
         bfc420ff7016804040003fdbffe080003f91ff2107fb7fcffac1fff981a81183fff5ff9ff2083fc803\
         400e107fe7f6c0981dffc7f500f03a003017fc707ff81faa0794398bdf70e53c4ff37fbfaa0d039056\
         e8bb876604e148203b7715e21e5fbc752f35df3ac81c0a00bc310b61441604881508a088348ae04e16\
         0200d42ba1444287818c0cc33854f13d3b9e778e53c87a7f00dbbe97bef6a08020434f84e9bcd43a0d\
         017ffe7f9ff4f8c03c0d823fd7f43d57aafa80b829851055f17cbf96fdfebbb676ef6e0d8308630a5f\
         93e77d0fa5febfafeef7232cb61662ac26088903233e83aa757f200cc2e05707bf63dcfbbf2e0bc2a0\
         59041f1bcd79bfc20e83406902bf7be17c5f3700200400801ed3bc779fc80b428051021ef3c3f82f35\
         e9fb075ff30aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\
         aaaaaaaaaaaaaaa";

    fn reference_corpus_packet_1_bytes() -> Vec<u8> {
        (0..REFERENCE_CORPUS_PACKET_1_HEX.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&REFERENCE_CORPUS_PACKET_1_HEX[i..i + 2], 16)
                    .expect("valid hex byte")
            })
            .collect()
    }

    #[test]
    fn reference_corpus_regression_packet_1() {
        use sha2::{Digest, Sha256};

        // Provenance:
        // - Upstream repository: https://github.com/nlsfi/HASlib (commit d036ea9d4efec44a45f94b6412446ca6b0f609ef)
        // - Recording: Tests/TestRecordings.zip recording galileo_ssr000.sbf, packet sequence_index 1
        // - Raw packet size: 583 bytes (4664 bits)
        // - Upstream decoder entry points: SSR_HAS.__init__ and HAS_Storage.feedMessage
        // - License: Upstream HASlib is licensed under EUPL-1.2. This test fixture is a small broadcast
        //   message sample extracted from bundled recording data, not a copied implementation.
        assert_eq!(REFERENCE_CORPUS_PACKET_1_HEX.len(), 583 * 2);
        let raw_bytes = reference_corpus_packet_1_bytes();
        assert_eq!(raw_bytes.len(), 583);

        // Verify SHA-256 matches independent computation
        let computed_sha256 = format!("{:x}", Sha256::digest(&raw_bytes));
        assert_eq!(
            computed_sha256,
            "b340e3199763fadad60636a285e80d390c166755ddee11e70f55a99427c4d751"
        );

        let decoded =
            HasMt1Message::decode(&raw_bytes).expect("decoding reference corpus packet 1");

        // Complete header assertions
        assert_eq!(decoded.header.toh_s, 2300);
        assert!(decoded.header.mask);
        assert!(decoded.header.orbit);
        assert!(!decoded.header.clock_full_set);
        assert!(!decoded.header.clock_subset);
        assert!(decoded.header.code_bias);
        assert!(!decoded.header.phase_bias);
        assert_eq!(decoded.header.reserved, 0);
        assert_eq!(decoded.header.mask_id, 22);
        assert_eq!(decoded.header.iod_set_id, 6);

        // Ordered GNSS masks assertions
        let mask = decoded.mask.as_ref().expect("mask present");
        assert_eq!(mask.systems.len(), 2);
        assert_eq!(mask.reserved, 0);

        // GPS mask (index 0)
        let mask_gps = &mask.systems[0];
        assert_eq!(mask_gps.system, GnssSystem::Gps);
        assert_eq!(mask_gps.satellites.len(), 26);
        assert_eq!(mask_gps.satellites.first().copied(), Some(2));
        assert_eq!(mask_gps.satellites.last().copied(), Some(31));
        assert_eq!(
            mask_gps.satellites,
            vec![
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27,
                29, 30, 31
            ]
        );
        assert_eq!(mask_gps.signals, vec![0, 7, 9]);
        let cell_mask = mask_gps.cell_mask.as_ref().expect("GPS cell mask present");
        assert_eq!(cell_mask.len(), 26 * 3);
        let cell_bitmap: String = cell_mask
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        assert_eq!(
            cell_bitmap,
            "101111111111111111111111111111111101111101111101101101111111111111111111111111"
        );
        assert_eq!(mask_gps.nav_message, 0);

        // Galileo mask (index 1)
        let mask_gal = &mask.systems[1];
        assert_eq!(mask_gal.system, GnssSystem::Galileo);
        assert_eq!(mask_gal.satellites.len(), 23);
        assert_eq!(mask_gal.satellites.first().copied(), Some(1));
        assert_eq!(mask_gal.satellites.last().copied(), Some(36));
        assert_eq!(
            mask_gal.satellites,
            vec![
                1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 15, 19, 21, 24, 25, 26, 27, 30, 31, 33, 34, 36
            ]
        );
        assert_eq!(mask_gal.signals, vec![1, 4, 7, 13]);
        assert!(mask_gal.cell_mask.is_none());
        assert_eq!(mask_gal.nav_message, 0);

        // Orbit block assertions
        let orbit = decoded.orbit.as_ref().expect("orbit block present");
        assert_eq!(orbit.validity_interval, 10);
        assert_eq!(orbit.records.len(), 49); // 26 GPS + 23 Galileo

        // Representative GPS orbit: beginning (PRN 2) and end (PRN 31)
        let orbit_gps_first = &orbit.records[0];
        assert_eq!(
            orbit_gps_first.sat,
            GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap()
        );
        assert_eq!(orbit_gps_first.iode, 56);
        assert_eq!(orbit_gps_first.radial_m, Some(33.0 * 0.0025)); // 0.0825 m
        assert_eq!(orbit_gps_first.along_m, Some(-75.0 * 0.008)); // -0.6 m
        assert_eq!(orbit_gps_first.cross_m, Some(-96.0 * 0.008)); // -0.768 m

        let orbit_gps_last = &orbit.records[25];
        assert_eq!(
            orbit_gps_last.sat,
            GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap()
        );
        assert_eq!(orbit_gps_last.iode, 39);
        assert_eq!(orbit_gps_last.radial_m, Some(2.0 * 0.0025)); // 0.005 m
        assert_eq!(orbit_gps_last.along_m, Some(-77.0 * 0.008)); // -0.616 m
        assert_eq!(orbit_gps_last.cross_m, Some(-16.0 * 0.008)); // -0.128 m

        // Representative Galileo orbit: beginning (PRN 1) and end (PRN 36)
        let orbit_gal_first = &orbit.records[26];
        assert_eq!(
            orbit_gal_first.sat,
            GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap()
        );
        assert_eq!(orbit_gal_first.iode, 32);
        assert_eq!(orbit_gal_first.radial_m, Some(18.0 * 0.0025)); // 0.045 m
        assert_eq!(orbit_gal_first.along_m, Some(-50.0 * 0.008)); // -0.4 m
        assert_eq!(orbit_gal_first.cross_m, Some(-15.0 * 0.008)); // -0.12 m

        let orbit_gal_last = &orbit.records[48];
        assert_eq!(
            orbit_gal_last.sat,
            GnssSatelliteId::new(GnssSystem::Galileo, 36).unwrap()
        );
        assert_eq!(orbit_gal_last.iode, 31);
        assert_eq!(orbit_gal_last.radial_m, Some(-64.0 * 0.0025)); // -0.16 m
        assert_eq!(
            orbit_gal_last.along_m.unwrap().to_bits(),
            (-43.0 * 0.008_f64).to_bits()
        );
        assert_eq!(orbit_gal_last.cross_m, Some(60.0 * 0.008)); // 0.48 m

        // Code bias block assertions
        let code_bias = decoded.code_bias.as_ref().expect("code bias block present");
        assert_eq!(code_bias.validity_interval, 10);
        assert_eq!(code_bias.records.len(), 164); // 72 GPS + 92 Galileo

        // Representative GPS code biases
        let cb_gps_first = &code_bias.records[0];
        assert_eq!(
            cb_gps_first.sat,
            GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap()
        );
        assert_eq!(cb_gps_first.signal_id, 0);
        assert_eq!(
            cb_gps_first.bias_m.unwrap().to_bits(),
            (230.0 * 0.02_f64).to_bits()
        ); // 4.60 m

        let cb_gps_second = &code_bias.records[1];
        assert_eq!(
            cb_gps_second.sat,
            GnssSatelliteId::new(GnssSystem::Gps, 2).unwrap()
        );
        assert_eq!(cb_gps_second.signal_id, 9);
        assert_eq!(
            cb_gps_second.bias_m.unwrap().to_bits(),
            (379.0 * 0.02_f64).to_bits()
        ); // 7.58 m

        let cb_gps_last = &code_bias.records[71];
        assert_eq!(
            cb_gps_last.sat,
            GnssSatelliteId::new(GnssSystem::Gps, 31).unwrap()
        );
        assert_eq!(cb_gps_last.signal_id, 9);
        assert_eq!(
            cb_gps_last.bias_m.unwrap().to_bits(),
            (191.0 * 0.02_f64).to_bits()
        ); // 3.82 m

        // Representative Galileo code biases (including unavailable sentinel check for PRN 30)
        let sat_gal30 = GnssSatelliteId::new(GnssSystem::Galileo, 30).unwrap();
        let gal30_records: Vec<&HasCodeBias> = code_bias
            .records
            .iter()
            .filter(|rec| rec.sat == sat_gal30)
            .collect();
        assert_eq!(gal30_records.len(), 4);
        for rec in &gal30_records {
            assert!(
                rec.bias_m.is_none(),
                "Galileo PRN 30 signals are unavailable"
            );
        }

        let cb_gal_last = &code_bias.records[163];
        assert_eq!(
            cb_gal_last.sat,
            GnssSatelliteId::new(GnssSystem::Galileo, 36).unwrap()
        );
        assert_eq!(cb_gal_last.signal_id, 13);
        assert_eq!(
            cb_gal_last.bias_m.unwrap().to_bits(),
            (-104.0 * 0.02_f64).to_bits()
        ); // -2.08 m

        // The first corpus packet has no clock corrections
        assert!(decoded.clock_full_set.is_none());
        assert!(decoded.clock_subset.is_none());

        // Padding and re-encoding equality
        assert_eq!(decoded.padding_bits.len(), 353);
        assert_eq!(
            decoded.padding_bits[0..4],
            [false, true, false, true] // Alternating 0101... pattern
        );

        let re_encoded = decoded
            .encode()
            .expect("re-encoding reference corpus packet");
        assert_eq!(re_encoded, raw_bytes);
    }

    const REFERENCE_CORPUS_SEQUENCE_2_HEX: &str =
        "90d202c650f2f865fee61ac0f304f7f660cdf3a7fa410c0f4ff3fcd3fd3d07ed5771bf08167faf81c0\
         0e606bf82855804a02f018ffbc05e01e04dff67f1e037ff6809c085fed04dff1c091ffc8007f4008ff\
         bafac2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn reference_corpus_sequence_2_bytes() -> Vec<u8> {
        (0..REFERENCE_CORPUS_SEQUENCE_2_HEX.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&REFERENCE_CORPUS_SEQUENCE_2_HEX[i..i + 2], 16)
                    .expect("valid hex byte")
            })
            .collect()
    }

    #[test]
    fn reference_corpus_regression_packet_2_maskless_clock() {
        use sha2::{Digest, Sha256};

        // Provenance:
        // - Upstream repository: https://github.com/nlsfi/HASlib (commit d036ea9d4efec44a45f94b6412446ca6b0f609ef)
        // - Recording: Tests/TestRecordings.zip recording galileo_ssr000.sbf, packet sequence_index 2
        // - Raw packet size: 106 bytes (848 bits)
        // - Upstream decoder entry points: SSR_HAS.__init__ and HAS_Storage.feedMessage
        // - Expected fields generated by unmodified nlsfi/HASlib d036ea9d4efec44a45f94b6412446ca6b0f609ef
        // - License: Upstream HASlib is licensed under EUPL-1.2. This test fixture is a small broadcast
        //   message sample extracted from bundled recording data, not a copied implementation.
        assert_eq!(REFERENCE_CORPUS_SEQUENCE_2_HEX.len(), 106 * 2);
        let raw_bytes = reference_corpus_sequence_2_bytes();
        assert_eq!(raw_bytes.len(), 106);

        // Verify SHA-256 matches independent computation
        let computed_sha256 = format!("{:x}", Sha256::digest(&raw_bytes));
        assert_eq!(
            computed_sha256,
            "8e3b4da8885ffc0735fc004cbcd3498324e2e8dc135f9215317c60b5cbb35aaf"
        );

        // Precursor mask context MUST come from sidereon's successful parsing of the precursor bytes,
        // not constructed from oracle expected metadata.
        let p1_bytes = reference_corpus_packet_1_bytes();
        let (_p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None)
            .expect("decoding precursor packet 1 with decode_with_context");
        let p1_ctx = p1_ctx_opt.expect("precursor packet 1 must yield HasMt1Context");
        assert_eq!(p1_ctx.mask_id(), 22);
        assert_eq!(p1_ctx.iod_set_id(), 6);

        // Decode packet 2 using precursor context
        let (decoded, p2_ctx_opt) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx))
            .expect("decoding maskless sequence 2 with precursor context");
        assert!(
            p2_ctx_opt.is_none(),
            "maskless message yields no new context"
        );

        // Header assertions: maskless clockFull
        assert_eq!(decoded.header.toh_s, 2317);
        assert!(!decoded.header.mask);
        assert!(!decoded.header.orbit);
        assert!(decoded.header.clock_full_set);
        assert!(!decoded.header.clock_subset);
        assert!(!decoded.header.code_bias);
        assert!(!decoded.header.phase_bias);
        assert_eq!(decoded.header.reserved, 0);
        assert_eq!(decoded.header.mask_id, 22);
        assert_eq!(decoded.header.iod_set_id, 6);

        // The decoded maskless message preserves header.mask = false and mask = None
        assert!(decoded.mask.is_none());
        assert!(decoded.orbit.is_none());
        assert!(decoded.clock_subset.is_none());
        assert!(decoded.code_bias.is_none());
        assert!(decoded.phase_bias.is_none());

        // Clock full-set block assertions
        let clock = decoded
            .clock_full_set
            .as_ref()
            .expect("clock_full_set present");
        assert_eq!(clock.validity_interval, 5);
        assert_eq!(clock.systems.len(), 2);
        assert_eq!(clock.systems[0].system, GnssSystem::Gps);
        assert_eq!(clock.systems[0].multiplier_index, 0); // 1.0x (DCM multiplier)
        assert_eq!(clock.systems[1].system, GnssSystem::Galileo);
        assert_eq!(clock.systems[1].multiplier_index, 0); // 1.0x (DCM multiplier)

        assert_eq!(clock.records.len(), 49); // 26 GPS + 23 Galileo

        // Expected records produced through unmodified nlsfi/HASlib d036ea9d4efec44a45f94b6412446ca6b0f609ef
        // (system, prn, raw_dcc_opt, do_not_use)
        let expected_records: [(GnssSystem, u8, Option<i16>, bool); 49] = [
            // GPS (26 satellites)
            (GnssSystem::Gps, 2, Some(-417), false),
            (GnssSystem::Gps, 3, Some(407), false),
            (GnssSystem::Gps, 4, Some(-141), false),
            (GnssSystem::Gps, 5, Some(428), false),
            (GnssSystem::Gps, 6, Some(486), false),
            (GnssSystem::Gps, 7, Some(317), false),
            (GnssSystem::Gps, 8, Some(-77), false),
            (GnssSystem::Gps, 9, Some(205), false),
            (GnssSystem::Gps, 10, Some(-396), false),
            (GnssSystem::Gps, 11, Some(-23), false),
            (GnssSystem::Gps, 12, Some(134), false),
            (GnssSystem::Gps, 13, Some(244), false),
            (GnssSystem::Gps, 14, Some(-25), false),
            (GnssSystem::Gps, 16, Some(-204), false),
            (GnssSystem::Gps, 18, Some(-23), false),
            (GnssSystem::Gps, 19, Some(-761), false),
            (GnssSystem::Gps, 20, Some(-598), false),
            (GnssSystem::Gps, 21, Some(-570), false),
            (GnssSystem::Gps, 23, Some(-124), false),
            (GnssSystem::Gps, 24, Some(359), false),
            (GnssSystem::Gps, 25, Some(-161), false),
            (GnssSystem::Gps, 26, Some(112), false),
            (GnssSystem::Gps, 27, Some(115), false),
            (GnssSystem::Gps, 29, Some(107), false),
            (GnssSystem::Gps, 30, Some(-251), false),
            (GnssSystem::Gps, 31, Some(342), false),
            // Galileo (23 satellites)
            (GnssSystem::Galileo, 1, Some(37), false),
            (GnssSystem::Galileo, 2, Some(47), false),
            (GnssSystem::Galileo, 3, Some(49), false),
            (GnssSystem::Galileo, 4, Some(-17), false),
            (GnssSystem::Galileo, 5, Some(47), false),
            (GnssSystem::Galileo, 7, Some(30), false),
            (GnssSystem::Galileo, 8, Some(155), false),
            (GnssSystem::Galileo, 9, Some(-39), false),
            (GnssSystem::Galileo, 10, Some(-113), false),
            (GnssSystem::Galileo, 11, Some(55), false),
            (GnssSystem::Galileo, 12, Some(-19), false),
            (GnssSystem::Galileo, 15, Some(39), false),
            (GnssSystem::Galileo, 19, Some(66), false),
            (GnssSystem::Galileo, 21, Some(-19), false),
            (GnssSystem::Galileo, 24, Some(155), false),
            (GnssSystem::Galileo, 25, Some(-57), false),
            (GnssSystem::Galileo, 26, Some(72), false),
            (GnssSystem::Galileo, 27, Some(-4), false),
            (GnssSystem::Galileo, 30, None, false), // Status: NOT_AVAILABLE (wire sentinel -4096)
            (GnssSystem::Galileo, 31, Some(-48), false),
            (GnssSystem::Galileo, 33, Some(71), false),
            (GnssSystem::Galileo, 34, Some(-70), false),
            (GnssSystem::Galileo, 36, Some(-168), false),
        ];

        for (i, &(system, prn, raw_opt, dnu)) in expected_records.iter().enumerate() {
            let rec = &clock.records[i];
            let expected_sat = GnssSatelliteId::new(system, prn).unwrap();
            assert_eq!(rec.sat, expected_sat, "record {i} satellite mismatch");
            assert_eq!(rec.do_not_use, dnu, "record {i} do_not_use mismatch");
            match raw_opt {
                Some(raw) => {
                    let expected_m = f64::from(raw) * HAS_CLOCK_SCALE_M;
                    assert_eq!(
                        rec.correction_m,
                        Some(expected_m),
                        "record {i} value mismatch for {expected_sat}"
                    );
                    assert_eq!(
                        rec.correction_m.unwrap().to_bits(),
                        expected_m.to_bits(),
                        "record {i} bitwise mismatch for {expected_sat}"
                    );
                }
                None => {
                    assert!(
                        rec.correction_m.is_none(),
                        "record {i} for {expected_sat} should be unavailable"
                    );
                }
            }
        }

        // Padding bits: 171 bits with alternating 0101... pattern
        assert_eq!(decoded.padding_bits.len(), 171);
        assert_eq!(decoded.padding_bits[0..4], [false, true, false, true]);

        // Encode with context: exact reproduction of raw wire bytes
        let re_encoded = decoded
            .encode_with_context(Some(&p1_ctx))
            .expect("re-encoding maskless packet 2 with context");
        assert_eq!(re_encoded, raw_bytes);
    }

    #[test]
    fn stateless_methods_still_refuse_correction_bearing_maskless_input() {
        let raw_bytes = reference_corpus_sequence_2_bytes();

        // Stateless decode must refuse correction-bearing maskless input with named error
        let err = HasMt1Message::decode(&raw_bytes).unwrap_err();
        assert_eq!(
            err,
            Error::Parse(
                "HAS MT1 correction blocks require a mask in this stateless decoder".to_string()
            )
        );

        // Stateless encode on a decoded maskless message must refuse with named error
        let p1_bytes = reference_corpus_packet_1_bytes();
        let (_p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();
        let (decoded, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();
        let enc_err = decoded.encode().unwrap_err();
        assert_eq!(
            enc_err,
            Error::InvalidInput("HAS MT1 correction blocks require a mask".to_string())
        );
    }

    #[test]
    fn missing_prior_inline_mask_context_fails_by_name() {
        let raw_bytes = reference_corpus_sequence_2_bytes();

        // decode_with_context with None context fails by name
        let dec_err = HasMt1Message::decode_with_context(&raw_bytes, None).unwrap_err();
        assert_eq!(
            dec_err,
            Error::Parse(
                "missing prior inline-mask context for HAS MT1 correction blocks".to_string()
            )
        );

        // encode_with_context with None context fails by name
        let p1_bytes = reference_corpus_packet_1_bytes();
        let (_p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();
        let (decoded, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();
        let enc_err = decoded.encode_with_context(None).unwrap_err();
        assert_eq!(
            enc_err,
            Error::InvalidInput(
                "missing prior inline-mask context for HAS MT1 correction blocks".to_string()
            )
        );
    }

    #[test]
    fn context_id_mismatch_rejected_both_directions() {
        let raw_bytes = reference_corpus_sequence_2_bytes();

        let p1_bytes = reference_corpus_packet_1_bytes();
        let (p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();
        let (decoded, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();

        // 1. Same Mask ID (22), different IOD Set ID (7 != 6), derived via public codec
        let mut p1_diff_iod = p1_msg.clone();
        p1_diff_iod.header.iod_set_id = 7;
        let p1_diff_iod_wire = p1_diff_iod.encode().unwrap();
        let (_, ctx_diff_iod_opt) =
            HasMt1Message::decode_with_context(&p1_diff_iod_wire, None).unwrap();
        let ctx_diff_iod = ctx_diff_iod_opt.unwrap();

        let dec_err_iod =
            HasMt1Message::decode_with_context(&raw_bytes, Some(&ctx_diff_iod)).unwrap_err();
        assert!(
            dec_err_iod
                .to_string()
                .contains("HAS MT1 context ID mismatch"),
            "got: {dec_err_iod}"
        );
        let enc_err_iod = decoded
            .encode_with_context(Some(&ctx_diff_iod))
            .unwrap_err();
        assert!(
            enc_err_iod
                .to_string()
                .contains("HAS MT1 context ID mismatch"),
            "got: {enc_err_iod}"
        );

        // 2. Different Mask ID (23 != 22), same IOD Set ID (6), derived via public codec
        let mut p1_diff_mask = p1_msg.clone();
        p1_diff_mask.header.mask_id = 23;
        let p1_diff_mask_wire = p1_diff_mask.encode().unwrap();
        let (_, ctx_diff_mask_opt) =
            HasMt1Message::decode_with_context(&p1_diff_mask_wire, None).unwrap();
        let ctx_diff_mask = ctx_diff_mask_opt.unwrap();

        let dec_err_mask =
            HasMt1Message::decode_with_context(&raw_bytes, Some(&ctx_diff_mask)).unwrap_err();
        assert!(
            dec_err_mask
                .to_string()
                .contains("HAS MT1 context ID mismatch"),
            "got: {dec_err_mask}"
        );
        let enc_err_mask = decoded
            .encode_with_context(Some(&ctx_diff_mask))
            .unwrap_err();
        assert!(
            enc_err_mask
                .to_string()
                .contains("HAS MT1 context ID mismatch"),
            "got: {enc_err_mask}"
        );
    }

    #[test]
    fn inline_mask_packet_uses_own_wire_definition_and_yields_new_context() {
        let p1_bytes = reference_corpus_packet_1_bytes();
        let (p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();

        // Explicitly internal invariant test: supply an irrelevant prior context with mismatched IDs (99, 99)
        let mut irrelevant_ctx = p1_ctx.clone();
        irrelevant_ctx.mask_id = 99;
        irrelevant_ctx.iod_set_id = 99;

        // Decode with irrelevant context: must NOT be refused, must use own wire definition,
        // and must yield a new immutable context matching the wire packet IDs (22, 6)
        let (decoded, yielded_ctx_opt) =
            HasMt1Message::decode_with_context(&p1_bytes, Some(&irrelevant_ctx))
                .expect("inline-mask decode ignores irrelevant prior context");
        let yielded_ctx = yielded_ctx_opt.expect("inline-mask decode yields new context");
        assert_eq!(yielded_ctx.mask_id(), 22);
        assert_eq!(yielded_ctx.iod_set_id(), 6);
        assert_eq!(decoded.header.mask_id, 22);
        assert_eq!(decoded.header.iod_set_id, 6);

        // Encode with irrelevant context: must use own wire mask and match original bytes
        let re_encoded = p1_msg
            .encode_with_context(Some(&irrelevant_ctx))
            .expect("inline-mask encode ignores irrelevant prior context");
        assert_eq!(re_encoded, p1_bytes);
    }

    #[test]
    fn malformed_trailing_data_after_valid_inline_mask_cannot_yield_context() {
        let p1_bytes = reference_corpus_packet_1_bytes();
        // Truncate packet 1 after the mask block (e.g. 50 bytes into the 583-byte packet)
        let truncated = &p1_bytes[..50];
        let err = HasMt1Message::decode_with_context(truncated, None);
        assert!(err.is_err(), "truncated payload must fail decode");
    }

    #[test]
    fn independently_malformed_record_count_or_order_cannot_encode_under_context() {
        let raw_bytes = reference_corpus_sequence_2_bytes();

        let p1_bytes = reference_corpus_packet_1_bytes();
        let (_p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();
        let (decoded, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();

        // 1. Missing record (48 records instead of 49): pop final record so preceding records match mask order
        let mut msg_missing = decoded.clone();
        msg_missing.clock_full_set.as_mut().unwrap().records.pop();
        let err_missing = msg_missing.encode_with_context(Some(&p1_ctx)).unwrap_err();
        assert!(
            err_missing.to_string().contains("missing satellite"),
            "got: {err_missing}"
        );

        // 2. Out-of-order records (swap record 0 and 1) name their satellites,
        //    so they are written at their mask positions: the same bytes.
        let mut msg_swapped = decoded.clone();
        msg_swapped
            .clock_full_set
            .as_mut()
            .unwrap()
            .records
            .swap(0, 1);
        assert_eq!(
            msg_swapped.encode_with_context(Some(&p1_ctx)).unwrap(),
            decoded.encode_with_context(Some(&p1_ctx)).unwrap()
        );

        // 2b. A second record for one satellite in place of another is refused.
        let mut msg_duplicate = decoded.clone();
        let records = &mut msg_duplicate.clock_full_set.as_mut().unwrap().records;
        records[1] = records[0];
        let duplicated = records[0].sat;
        let err_duplicate = msg_duplicate
            .encode_with_context(Some(&p1_ctx))
            .unwrap_err();
        assert!(
            err_duplicate
                .to_string()
                .contains(&format!("satellite {duplicated} more than once")),
            "got: {err_duplicate}"
        );

        // 3. Unexpected record (add satellite G01, which is absent from packet 1's GPS mask)
        let mut msg_extra = decoded.clone();
        msg_extra
            .clock_full_set
            .as_mut()
            .unwrap()
            .records
            .push(HasClockCorrection {
                sat: GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap(),
                correction_m: Some(0.0),
                do_not_use: false,
            });
        let err_extra = msg_extra.encode_with_context(Some(&p1_ctx)).unwrap_err();
        assert!(
            err_extra.to_string().contains("unexpected satellite"),
            "got: {err_extra}"
        );
    }

    #[test]
    fn context_immutability_and_no_shared_cache() {
        let raw_bytes = reference_corpus_sequence_2_bytes();

        let p1_bytes = reference_corpus_packet_1_bytes();
        let (_p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();

        // Decode packet 2 with context
        let (mut decoded, _) =
            HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();

        // Mutate message aggressively
        decoded.clock_full_set = None;
        decoded.header.mask_id = 0;
        decoded.header.iod_set_id = 0;

        // Verify p1_ctx was not mutated
        assert_eq!(p1_ctx.mask_id(), 22);
        assert_eq!(p1_ctx.iod_set_id(), 6);
        assert_eq!(p1_ctx.mask().systems.len(), 2);

        // Decode another message with p1_ctx; verify p1_ctx remains unchanged
        let (decoded2, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();
        assert_eq!(p1_ctx.mask_id(), 22);
        assert_eq!(p1_ctx.iod_set_id(), 6);
        assert_eq!(decoded2.header.mask_id, 22);

        // Verify two contexts are completely independent (no shared cache).
        // Explicitly internal invariant test directly mutating private fields to verify value independence:
        let mut ctx2 = p1_ctx.clone();
        ctx2.mask_id = 10;
        ctx2.iod_set_id = 15;
        assert_eq!(p1_ctx.mask_id(), 22);
        assert_eq!(p1_ctx.iod_set_id(), 6);
        assert_eq!(ctx2.mask_id(), 10);
        assert_eq!(ctx2.iod_set_id(), 15);
    }

    #[test]
    fn caller_built_clone_modified_to_different_id_refused() {
        let raw_bytes = reference_corpus_sequence_2_bytes();

        let p1_bytes = reference_corpus_packet_1_bytes();
        let (p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();
        let (decoded, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();

        // 1. Caller modifies decoded message header mask_id, then checks encode rejects prior real context
        let mut msg_mod = decoded.clone();
        msg_mod.header.mask_id = 31;
        let enc_err = msg_mod.encode_with_context(Some(&p1_ctx)).unwrap_err();
        assert!(
            enc_err.to_string().contains("HAS MT1 context ID mismatch"),
            "got: {enc_err}"
        );

        // Create mismatched decode wire by editing public message header then stateless-encoding
        // with inline mask, obtaining a context with mask_id 31, and encoding maskless through actual codec
        let mut p1_mod = p1_msg.clone();
        p1_mod.header.mask_id = 31;
        let wire_p1_mod = p1_mod.encode().unwrap();
        let (_, ctx_mod_opt) = HasMt1Message::decode_with_context(&wire_p1_mod, None).unwrap();
        let ctx_mod = ctx_mod_opt.unwrap();

        let mismatched_wire = msg_mod.encode_with_context(Some(&ctx_mod)).unwrap();
        let dec_err =
            HasMt1Message::decode_with_context(&mismatched_wire, Some(&p1_ctx)).unwrap_err();
        assert!(
            dec_err.to_string().contains("HAS MT1 context ID mismatch"),
            "got: {dec_err}"
        );

        // 2. Repeat for iod_set_id mismatch in each direction
        let mut msg_mod_iod = decoded.clone();
        msg_mod_iod.header.iod_set_id = 15;
        let enc_err_iod = msg_mod_iod.encode_with_context(Some(&p1_ctx)).unwrap_err();
        assert!(
            enc_err_iod
                .to_string()
                .contains("HAS MT1 context ID mismatch"),
            "got: {enc_err_iod}"
        );

        let mut p1_mod_iod = p1_msg.clone();
        p1_mod_iod.header.iod_set_id = 15;
        let wire_p1_mod_iod = p1_mod_iod.encode().unwrap();
        let (_, ctx_mod_iod_opt) =
            HasMt1Message::decode_with_context(&wire_p1_mod_iod, None).unwrap();
        let ctx_mod_iod = ctx_mod_iod_opt.unwrap();

        let mismatched_wire_iod = msg_mod_iod.encode_with_context(Some(&ctx_mod_iod)).unwrap();
        let dec_err_iod =
            HasMt1Message::decode_with_context(&mismatched_wire_iod, Some(&p1_ctx)).unwrap_err();
        assert!(
            dec_err_iod
                .to_string()
                .contains("HAS MT1 context ID mismatch"),
            "got: {dec_err_iod}"
        );
    }

    #[test]
    fn malformed_mask_in_context_refused() {
        // Explicitly internal invariant test: HasMt1Context fields are private to the
        // module, so external callers cannot construct or mutate a malformed context directly.
        // This validates defense-in-depth against an internally mutated context struct.
        let raw_bytes = reference_corpus_sequence_2_bytes();

        let p1_bytes = reference_corpus_packet_1_bytes();
        let (_p1_msg, p1_ctx_opt) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();
        let p1_ctx = p1_ctx_opt.unwrap();
        let (decoded, _) = HasMt1Message::decode_with_context(&raw_bytes, Some(&p1_ctx)).unwrap();

        // Mutate context mask with an invalid cell mask length
        let mut malformed_ctx = p1_ctx.clone();
        malformed_ctx.mask.systems[0].cell_mask = Some(vec![true; 5]); // expected 26*3 = 78
        let dec_err =
            HasMt1Message::decode_with_context(&raw_bytes, Some(&malformed_ctx)).unwrap_err();
        assert!(
            dec_err
                .to_string()
                .contains("cell mask holds 5 cells for 78"),
            "got: {dec_err}"
        );
        let enc_err = decoded
            .encode_with_context(Some(&malformed_ctx))
            .unwrap_err();
        assert!(
            enc_err
                .to_string()
                .contains("cell mask holds 5 cells for 78"),
            "got: {enc_err}"
        );
    }

    #[test]
    fn malformed_public_inline_mask_caller_encoding_refuses_out_of_order_prns() {
        let p1_bytes = reference_corpus_packet_1_bytes();
        let (mut p1_msg, _) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();

        // Caller constructs public inline-mask message with non-strictly-ordered satellite PRNs
        p1_msg.mask.as_mut().unwrap().systems[0]
            .satellites
            .swap(0, 1);

        let enc_err = p1_msg.encode_with_context(None).unwrap_err();
        assert!(
            enc_err.to_string().contains("not strictly ascending"),
            "got: {enc_err}"
        );
    }

    #[test]
    fn malformed_public_inline_mask_caller_encoding_refuses_invalid_cell_length() {
        let p1_bytes = reference_corpus_packet_1_bytes();
        let (mut p1_msg, _) = HasMt1Message::decode_with_context(&p1_bytes, None).unwrap();

        // Caller constructs public inline-mask message with invalid cell_mask length (5 != 26*3 = 78)
        p1_msg.mask.as_mut().unwrap().systems[0].cell_mask = Some(vec![true; 5]);

        let enc_err = p1_msg.encode_with_context(None).unwrap_err();
        assert!(
            enc_err
                .to_string()
                .contains("cell mask holds 5 cells for 78"),
            "got: {enc_err}"
        );
    }

    #[test]
    fn test_orbit_component_sentinels_independent_raw_bits() {
        // Construct raw wire bits with Table 15 boundary Reserved6.
        let make_raw_orbit = |radial_raw: i16, along_raw: i16, cross_raw: i16| -> Vec<u8> {
            let mut w = BitWriter::new();
            // Header (32 bits): TOH=100, mask=1, orbit=1, others=0, reserved=0, mask_id=1, iod_set_id=1
            w.push_u(100, 12);
            w.push_flag(true); // mask
            w.push_flag(true); // orbit
            w.push_flag(false); // clock_full_set
            w.push_flag(false); // clock_subset
            w.push_flag(false); // code_bias
            w.push_flag(false); // phase_bias
            w.push_u(0, 4); // reserved4
            w.push_u(1, 5); // mask_id
            w.push_u(1, 5); // iod_set_id

            // Mask block (Table 15): Nsys=1 (4 bits), GPS (4 bits), PRN 1 (40 bits), Sig 0 (16 bits), CMAF=0 (1 bit), NM=0 (3 bits), Reserved=0 (6 bits)
            w.push_u(1, 4); // Nsys = 1
            w.push_u(0, 4); // GNSS ID = 0 (GPS)
            w.push_u(1 << 39, 40); // SatM: PRN 1
            w.push_u(1 << 15, 16); // SigM: Sig 0
            w.push_flag(false); // CMAF = 0
            w.push_u(0, 3); // NM = 0
            w.push_u(0, 6); // Reserved6 (Table 15 boundary!)

            // Orbit block (Table 24/25): VI=5 (4 bits), LIOD=10 (8 bits for GPS), radial (13b), along (12b), cross (12b)
            w.push_u(5, 4); // VI = 5 (60s)
            w.push_u(10, 8); // LIOD (GPS 8-bit)
            w.push_i(i64::from(radial_raw), 13);
            w.push_i(i64::from(along_raw), 12);
            w.push_i(i64::from(cross_raw), 12);

            w.into_bytes()
        };

        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();

        // 1. Radial sentinel (-4096), along and cross numeric
        let bytes_a = make_raw_orbit(HAS_ORBIT_RADIAL_INVALID, 10, 20);
        let msg_a = HasMt1Message::decode(&bytes_a).expect("decode orbit radial sentinel");
        let rec_a = &msg_a.orbit.as_ref().unwrap().records[0];
        assert_eq!(rec_a.sat, sat);
        assert_eq!(rec_a.radial_m, None);
        assert_eq!(rec_a.along_m, Some(10.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M));
        assert_eq!(rec_a.cross_m, Some(20.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M));

        // 2. Along sentinel (-2048), radial and cross numeric
        let bytes_b = make_raw_orbit(50, HAS_ORBIT_ALONG_CROSS_INVALID, 20);
        let msg_b = HasMt1Message::decode(&bytes_b).expect("decode orbit along sentinel");
        let rec_b = &msg_b.orbit.as_ref().unwrap().records[0];
        assert_eq!(rec_b.radial_m, Some(50.0 * HAS_ORBIT_RADIAL_SCALE_M));
        assert_eq!(rec_b.along_m, None);
        assert_eq!(rec_b.cross_m, Some(20.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M));

        // 3. Cross sentinel (-2048), radial and along numeric
        let bytes_c = make_raw_orbit(50, 10, HAS_ORBIT_ALONG_CROSS_INVALID);
        let msg_c = HasMt1Message::decode(&bytes_c).expect("decode orbit cross sentinel");
        let rec_c = &msg_c.orbit.as_ref().unwrap().records[0];
        assert_eq!(rec_c.radial_m, Some(50.0 * HAS_ORBIT_RADIAL_SCALE_M));
        assert_eq!(rec_c.along_m, Some(10.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M));
        assert_eq!(rec_c.cross_m, None);

        // 4. All three sentinels independently None
        let bytes_d = make_raw_orbit(
            HAS_ORBIT_RADIAL_INVALID,
            HAS_ORBIT_ALONG_CROSS_INVALID,
            HAS_ORBIT_ALONG_CROSS_INVALID,
        );
        let msg_d = HasMt1Message::decode(&bytes_d).expect("decode all orbit sentinels");
        let rec_d = &msg_d.orbit.as_ref().unwrap().records[0];
        assert_eq!(rec_d.radial_m, None);
        assert_eq!(rec_d.along_m, None);
        assert_eq!(rec_d.cross_m, None);
    }

    #[test]
    fn test_code_bias_sentinel_vs_zero_raw_bits() {
        let make_raw_code_bias = |raw_bias: i16| -> Vec<u8> {
            let mut w = BitWriter::new();
            // Header (32 bits): mask=1, code_bias=1
            w.push_u(200, 12);
            w.push_flag(true); // mask
            w.push_flag(false); // orbit
            w.push_flag(false); // clock_full_set
            w.push_flag(false); // clock_subset
            w.push_flag(true); // code_bias
            w.push_flag(false); // phase_bias
            w.push_u(0, 4); // reserved4
            w.push_u(1, 5); // mask_id
            w.push_u(1, 5); // iod_set_id

            // Mask block: Nsys=1, GPS, PRN 1, Sig 0, Reserved6=0
            w.push_u(1, 4);
            w.push_u(0, 4);
            w.push_u(1 << 39, 40);
            w.push_u(1 << 15, 16);
            w.push_flag(false);
            w.push_u(0, 3);
            w.push_u(0, 6); // Reserved6

            // Code bias: VI=5, raw bias 11 bits
            w.push_u(5, 4);
            w.push_i(i64::from(raw_bias), 11);

            w.into_bytes()
        };

        // Sentinel -1024 decodes to None
        let bytes_sentinel = make_raw_code_bias(HAS_CODE_BIAS_INVALID);
        let msg_sentinel =
            HasMt1Message::decode(&bytes_sentinel).expect("decode code bias sentinel");
        assert_eq!(
            msg_sentinel.code_bias.as_ref().unwrap().records[0].bias_m,
            None
        );

        // Zero decodes to Some(0.0)
        let bytes_zero = make_raw_code_bias(0);
        let msg_zero = HasMt1Message::decode(&bytes_zero).expect("decode code bias zero");
        assert_eq!(
            msg_zero.code_bias.as_ref().unwrap().records[0].bias_m,
            Some(0.0)
        );
    }

    #[test]
    fn test_phase_bias_sentinel_vs_zero_and_unknown_carrier_raw_bits() {
        let mut w = BitWriter::new();
        // Header (32 bits): mask=1, phase_bias=1
        w.push_u(300, 12);
        w.push_flag(true); // mask
        w.push_flag(false); // orbit
        w.push_flag(false); // clock_full_set
        w.push_flag(false); // clock_subset
        w.push_flag(false); // code_bias
        w.push_flag(true); // phase_bias
        w.push_u(0, 4); // reserved4
        w.push_u(1, 5); // mask_id
        w.push_u(1, 5); // iod_set_id

        // Mask block: GPS, PRN 1, Signals: 0 (L1) and 1 (unknown carrier!)
        w.push_u(1, 4);
        w.push_u(0, 4);
        w.push_u(1 << 39, 40);
        w.push_u((1 << 15) | (1 << 14), 16); // Sig 0 and Sig 1
        w.push_flag(false); // CMAF=false: 2 cells
        w.push_u(0, 3);
        w.push_u(0, 6); // Reserved6

        // Phase bias: VI=5 (4b)
        w.push_u(5, 4);
        // Cell 0 (GPS PRN 1, Sig 0 L1): sentinel -1024, PDI = 2
        w.push_i(i64::from(HAS_PHASE_BIAS_INVALID), 11);
        w.push_u(2, 2);
        // Cell 1 (GPS PRN 1, Sig 1 unknown carrier): raw = 25 (0.25 cycles), PDI = 3
        w.push_i(25, 11);
        w.push_u(3, 2);

        let bytes = w.into_bytes();
        let msg = HasMt1Message::decode(&bytes).expect("decode phase bias raw bits");
        let recs = &msg.phase_bias.as_ref().unwrap().records;
        assert_eq!(recs.len(), 2);

        // Cell 0: Transmitted unavailable, raw PDI preserved
        assert_eq!(recs[0].signal_id, 0);
        assert_eq!(recs[0].bias_cycles, None);
        assert_eq!(recs[0].bias_m(), None);
        assert_eq!(recs[0].discontinuity_indicator, 2);
        assert_eq!(
            recs[0].conversion(),
            HasPhaseBiasConversion::TransmittedUnavailable
        );

        // Cell 1: Unknown carrier, raw cycles (0.25) and PDI (3) preserved, conversion is UnknownSignal
        assert_eq!(recs[1].signal_id, 1);
        assert_eq!(recs[1].bias_cycles, Some(0.25));
        assert_eq!(recs[1].bias_m(), None);
        assert_eq!(recs[1].discontinuity_indicator, 3);
        assert_eq!(recs[1].conversion(), HasPhaseBiasConversion::UnknownSignal);

        // Zero case for known signal
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let zero_pb = HasPhaseBias {
            sat,
            signal_id: 0,
            bias_cycles: Some(0.0),
            discontinuity_indicator: 1,
        };
        assert_eq!(zero_pb.bias_cycles, Some(0.0));
        assert_eq!(zero_pb.bias_m(), Some(0.0));
        assert_eq!(
            zero_pb.conversion(),
            HasPhaseBiasConversion::Available {
                bias_m: 0.0,
                frequency_hz: F_L1_HZ,
            }
        );
    }

    #[test]
    fn test_phase_bias_finite_max_cycles_converts_without_intermediate_overflow() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let pb = HasPhaseBias {
            sat,
            signal_id: 0,
            bias_cycles: Some(f64::MAX),
            discontinuity_indicator: 0,
        };
        assert_eq!(pb.bias_cycles, Some(f64::MAX));
        let expected_wavelength = C_M_S / F_L1_HZ;
        let expected_bias_m = f64::MAX * expected_wavelength;
        assert!(expected_bias_m.is_finite());
        assert_eq!(
            pb.conversion(),
            HasPhaseBiasConversion::Available {
                bias_m: expected_bias_m,
                frequency_hz: F_L1_HZ,
            }
        );
        assert_eq!(pb.bias_m(), Some(expected_bias_m));

        // Codec encoding must still refuse out-of-wire-range cycles
        let mask = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        let pb_block = HasPhaseBiasBlock {
            validity_interval: 5,
            records: vec![pb],
        };
        let mut w = BitWriter::new();
        assert!(matches!(
            write_phase_bias_block(&mut w, &mask, &pb_block),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_legal_extrema_roundtrips() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 63, // max 6-bit reserved
        });

        // 1. Positive extrema:
        // Orbit radial: +4095 * 0.0025 = 10.2375 m
        // Orbit along/cross: +2047 * 0.0080 = 16.376 m
        // Code bias: +1023 * 0.02 = 20.46 m
        // Phase bias: +1023 * 0.01 = 10.23 cycles
        let max_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 3599, // max TOH
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 15,   // max 4-bit reserved
                mask_id: 31,    // max 5-bit
                iod_set_id: 31, // max 5-bit
            },
            mask: mask.clone(),
            orbit: Some(HasOrbitBlock {
                validity_interval: 14, // max VI (3600s)
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: 255, // max 8-bit GPS IODE
                    radial_m: Some(4095.0 * HAS_ORBIT_RADIAL_SCALE_M),
                    along_m: Some(2047.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M),
                    cross_m: Some(2047.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M),
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 14,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(1023.0 * HAS_CODE_BIAS_SCALE_M),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 14,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(1023.0 * HAS_PHASE_BIAS_SCALE_CYCLES),
                    discontinuity_indicator: 3, // max 2-bit PDI
                }],
            }),
            padding_bits: Vec::new(),
        };
        let encoded_max = max_msg.encode().expect("encode max extrema");
        let decoded_max = HasMt1Message::decode(&encoded_max).expect("decode max extrema");
        // Header 32 + mask 74 + orbit 49 + code bias 15 + phase bias 17 = 187
        // payload bits, so the encoder zero-pads the final byte with five bits.
        assert_eq!(encoded_max.len(), 24);
        assert_eq!(decoded_max.padding_bits.len(), 5);
        assert!(
            decoded_max.padding_bits.iter().all(|bit| !bit),
            "trailing padding must be zero bits"
        );
        assert_eq!(
            decoded_max.encode().expect("re-encode max extrema"),
            encoded_max
        );
        let decoded_max_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..decoded_max.clone()
        };
        assert_eq!(decoded_max_without_padding, max_msg);

        // 2. Negative extrema:
        // Orbit radial: -4095 * 0.0025 = -10.2375 m
        // Orbit along/cross: -2047 * 0.0080 = -16.376 m
        // Code bias: -1023 * 0.02 = -20.46 m
        // Phase bias: -1023 * 0.01 = -10.23 cycles
        let min_msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: true,
                clock_full_set: false,
                clock_subset: false,
                code_bias: true,
                phase_bias: true,
                reserved: 0,
                mask_id: 0,
                iod_set_id: 0,
            },
            mask,
            orbit: Some(HasOrbitBlock {
                validity_interval: 0,
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: 0,
                    radial_m: Some(-4095.0 * HAS_ORBIT_RADIAL_SCALE_M),
                    along_m: Some(-2047.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M),
                    cross_m: Some(-2047.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M),
                }],
            }),
            clock_full_set: None,
            clock_subset: None,
            code_bias: Some(HasCodeBiasBlock {
                validity_interval: 0,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(-1023.0 * HAS_CODE_BIAS_SCALE_M),
                }],
            }),
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 0,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(-1023.0 * HAS_PHASE_BIAS_SCALE_CYCLES),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };
        let encoded_min = min_msg.encode().expect("encode min extrema");
        let decoded_min = HasMt1Message::decode(&encoded_min).expect("decode min extrema");
        // Same block layout as the positive extrema: 187 payload bits, five
        // zero pad bits.
        assert_eq!(encoded_min.len(), 24);
        assert_eq!(decoded_min.padding_bits.len(), 5);
        assert!(
            decoded_min.padding_bits.iter().all(|bit| !bit),
            "trailing padding must be zero bits"
        );
        assert_eq!(
            decoded_min.encode().expect("re-encode min extrema"),
            encoded_min
        );
        let decoded_min_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..decoded_min.clone()
        };
        assert_eq!(decoded_min_without_padding, min_msg);
    }

    #[test]
    fn test_nonfinite_and_out_of_range_and_sentinel_collision_refusals() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let mask = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };

        // Helper to test orbit radial write refusal
        let test_orbit_radial = |val: f64| {
            let mut w = BitWriter::new();
            let block = HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: 1,
                    radial_m: Some(val),
                    along_m: Some(0.0),
                    cross_m: Some(0.0),
                }],
            };
            write_orbit_block(&mut w, &mask, &block).unwrap_err()
        };

        assert!(matches!(
            test_orbit_radial(f64::NAN),
            Error::InvalidInput(_)
        ));
        assert!(matches!(
            test_orbit_radial(f64::INFINITY),
            Error::InvalidInput(_)
        ));
        assert!(matches!(
            test_orbit_radial(f64::NEG_INFINITY),
            Error::InvalidInput(_)
        ));
        // Overflow (+4096 * 0.0025 = 10.24 m)
        assert!(matches!(
            test_orbit_radial(4096.0 * HAS_ORBIT_RADIAL_SCALE_M),
            Error::InvalidInput(_)
        ));
        // Collision onto reserved sentinel (-4096 * 0.0025 = -10.24 m)
        assert!(matches!(
            test_orbit_radial(-4096.0 * HAS_ORBIT_RADIAL_SCALE_M),
            Error::InvalidInput(_)
        ));

        // Orbit along/cross
        let test_orbit_along = |val: f64| {
            let mut w = BitWriter::new();
            let block = HasOrbitBlock {
                validity_interval: 5,
                records: vec![HasOrbitCorrection {
                    sat,
                    iode: 1,
                    radial_m: Some(0.0),
                    along_m: Some(val),
                    cross_m: Some(0.0),
                }],
            };
            write_orbit_block(&mut w, &mask, &block).unwrap_err()
        };
        assert!(matches!(test_orbit_along(f64::NAN), Error::InvalidInput(_)));
        assert!(matches!(
            test_orbit_along(2048.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M),
            Error::InvalidInput(_)
        ));
        assert!(matches!(
            test_orbit_along(-2048.0 * HAS_ORBIT_ALONG_CROSS_SCALE_M),
            Error::InvalidInput(_)
        ));

        // Code bias
        let test_code_bias = |val: f64| {
            let mut w = BitWriter::new();
            let block = HasCodeBiasBlock {
                validity_interval: 5,
                records: vec![HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: Some(val),
                }],
            };
            write_code_bias_block(&mut w, &mask, &block).unwrap_err()
        };
        assert!(matches!(test_code_bias(f64::NAN), Error::InvalidInput(_)));
        assert!(matches!(
            test_code_bias(1024.0 * HAS_CODE_BIAS_SCALE_M),
            Error::InvalidInput(_)
        ));
        assert!(matches!(
            test_code_bias(-1024.0 * HAS_CODE_BIAS_SCALE_M),
            Error::InvalidInput(_)
        ));

        // Phase bias
        let test_phase_bias = |val: f64| {
            let mut w = BitWriter::new();
            let block = HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(val),
                    discontinuity_indicator: 0,
                }],
            };
            write_phase_bias_block(&mut w, &mask, &block).unwrap_err()
        };
        assert!(matches!(test_phase_bias(f64::NAN), Error::InvalidInput(_)));
        assert!(matches!(
            test_phase_bias(1024.0 * HAS_PHASE_BIAS_SCALE_CYCLES),
            Error::InvalidInput(_)
        ));
        assert!(matches!(
            test_phase_bias(-1024.0 * HAS_PHASE_BIAS_SCALE_CYCLES),
            Error::InvalidInput(_)
        ));
    }

    #[test]
    fn test_iode_pdi_header_width_refusals() {
        let gps_sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let gal_sat = GnssSatelliteId::new(GnssSystem::Galileo, 1).unwrap();
        let mask_gps = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        let mask_gal = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Galileo,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };

        // 1. GPS IODE > 255
        let mut w = BitWriter::new();
        let gps_orbit = HasOrbitBlock {
            validity_interval: 5,
            records: vec![HasOrbitCorrection {
                sat: gps_sat,
                iode: 256,
                radial_m: Some(0.0),
                along_m: Some(0.0),
                cross_m: Some(0.0),
            }],
        };
        let err_gps_iode = write_orbit_block(&mut w, &mask_gps, &gps_orbit).unwrap_err();
        assert!(err_gps_iode.to_string().contains("exceeds 8-bit maximum"));

        // 2. Galileo IODE > 1023
        let mut w = BitWriter::new();
        let gal_orbit = HasOrbitBlock {
            validity_interval: 5,
            records: vec![HasOrbitCorrection {
                sat: gal_sat,
                iode: 1024,
                radial_m: Some(0.0),
                along_m: Some(0.0),
                cross_m: Some(0.0),
            }],
        };
        let err_gal_iode = write_orbit_block(&mut w, &mask_gal, &gal_orbit).unwrap_err();
        assert!(err_gal_iode.to_string().contains("exceeds 10-bit maximum"));

        // 3. Phase bias PDI > 3
        let mut w = BitWriter::new();
        let pb = HasPhaseBiasBlock {
            validity_interval: 5,
            records: vec![HasPhaseBias {
                sat: gps_sat,
                signal_id: 0,
                bias_cycles: Some(0.0),
                discontinuity_indicator: 4,
            }],
        };
        let err_pdi = write_phase_bias_block(&mut w, &mask_gps, &pb).unwrap_err();
        assert!(err_pdi.to_string().contains("exceeds 2-bit maximum"));

        // 4. Header field overflow
        let hdr_base = HasMt1Header {
            toh_s: 0,
            mask: true,
            orbit: false,
            clock_full_set: false,
            clock_subset: false,
            code_bias: false,
            phase_bias: false,
            reserved: 0,
            mask_id: 0,
            iod_set_id: 0,
        };
        let mut w = BitWriter::new();
        let mut hdr_bad_toh = hdr_base;
        hdr_bad_toh.toh_s = 3600;
        assert!(write_header(&mut w, hdr_bad_toh).is_err());

        let mut hdr_bad_res = hdr_base;
        hdr_bad_res.reserved = 16;
        assert!(write_header(&mut w, hdr_bad_res).is_err());

        let mut hdr_bad_mid = hdr_base;
        hdr_bad_mid.mask_id = 32;
        assert!(write_header(&mut w, hdr_bad_mid).is_err());

        let mut hdr_bad_iod = hdr_base;
        hdr_bad_iod.iod_set_id = 32;
        assert!(write_header(&mut w, hdr_bad_iod).is_err());

        // Decode rejection of TOH > 3599
        let mut w = BitWriter::new();
        w.push_u(3600, 12); // invalid TOH
        w.push_flag(false);
        w.push_flag(false);
        w.push_flag(false);
        w.push_flag(false);
        w.push_flag(false);
        w.push_flag(false);
        w.push_u(0, 4);
        w.push_u(0, 5);
        w.push_u(0, 5);
        let dec_err = HasMt1Message::decode(&w.into_bytes()).unwrap_err();
        assert!(dec_err.to_string().contains("exceeds 0..=3599 s range"));

        // 5. has_mt1_reference_j2000_s checks
        let gst = crate::astro::time::model::GnssWeekTow::new(
            crate::astro::time::model::TimeScale::Gst,
            1042,
            1000.0,
        )
        .unwrap();
        assert!(has_mt1_reference_j2000_s(gst, 3600).is_err());

        let gst_nan = crate::astro::time::model::GnssWeekTow {
            system: crate::astro::time::model::TimeScale::Gst,
            week: 1042,
            tow_s: f64::NAN,
        };
        assert!(has_mt1_reference_j2000_s(gst_nan, 100).is_err());

        let gst_neg = crate::astro::time::model::GnssWeekTow {
            system: crate::astro::time::model::TimeScale::Gst,
            week: 1042,
            tow_s: -0.1,
        };
        assert!(has_mt1_reference_j2000_s(gst_neg, 100).is_err());

        let gst_week_overflow = crate::astro::time::model::GnssWeekTow {
            system: crate::astro::time::model::TimeScale::Gst,
            week: 1042,
            tow_s: 604800.0,
        };
        assert!(has_mt1_reference_j2000_s(gst_week_overflow, 100).is_err());
    }

    #[test]
    fn test_malformed_masks_and_empty_legal_masks() {
        // 1. Empty systems
        let mask_empty_sys = HasMaskBlock {
            systems: vec![],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_empty_sys).is_err());

        // 2. Systems count > 15
        let sys_16 = (0..16)
            .map(|i| HasGnssMask {
                system: if i % 2 == 0 {
                    GnssSystem::Gps
                } else {
                    GnssSystem::Galileo
                },
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            })
            .collect();
        let mask_16_sys = HasMaskBlock {
            systems: sys_16,
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_16_sys).is_err());

        // 3. Reserved > 63
        let mask_bad_res = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 64,
        };
        assert!(validate_mask_block_structural(&mask_bad_res).is_err());

        // 4. Duplicate systems
        let mask_dup_sys = HasMaskBlock {
            systems: vec![
                HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![1],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
                HasGnssMask {
                    system: GnssSystem::Gps,
                    satellites: vec![2],
                    signals: vec![0],
                    cell_mask: None,
                    nav_message: 0,
                },
            ],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_dup_sys).is_err());

        // 5. Nav message > 7
        assert!(validate_mask_block_structural(&HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 8,
            }],
            reserved: 0,
        })
        .is_err());

        // 6. PRN 0 and PRN 41
        let mask_prn0 = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![0],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_prn0).is_err());

        let mask_prn41 = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![41],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_prn41).is_err());

        // 7. Unsorted or duplicate PRNs
        let mask_unsorted_prn = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![5, 3],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_unsorted_prn).is_err());

        let mask_dup_prn = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![3, 3],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_dup_prn).is_err());

        // 8. Signal > 15, unsorted signals, duplicate signals
        let mask_sig16 = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![16],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_sig16).is_err());

        let mask_unsorted_sig = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![4, 2],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_unsorted_sig).is_err());

        let mask_dup_sig = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![2, 2],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_dup_sig).is_err());

        // 9. Cell mask length mismatch
        let mask_bad_cells = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0, 3],
                cell_mask: Some(vec![true; 3]), // expected 1*2 = 2
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_bad_cells).is_err());

        // 10. Legal empty satellites list: encodes and decodes cleanly
        let mask_empty_sats = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_empty_sats).is_ok());
        let msg_empty_sats = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask_empty_sats.clone()),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let enc_empty_sats = msg_empty_sats.encode().expect("encode empty sats mask");
        let dec_empty_sats =
            HasMt1Message::decode(&enc_empty_sats).expect("decode empty sats mask");
        // Header 32 + mask 74 = 106 payload bits, so the encoder zero-pads the
        // final byte with six bits.
        assert_eq!(enc_empty_sats.len(), 14);
        assert_eq!(dec_empty_sats.padding_bits.len(), 6);
        assert!(
            dec_empty_sats.padding_bits.iter().all(|bit| !bit),
            "trailing padding must be zero bits"
        );
        assert_eq!(
            dec_empty_sats.encode().expect("re-encode empty sats mask"),
            enc_empty_sats
        );
        let dec_empty_sats_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..dec_empty_sats.clone()
        };
        assert_eq!(dec_empty_sats_without_padding, msg_empty_sats);

        // 11. Legal empty signals list: encodes and decodes cleanly
        let mask_empty_sigs = HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        };
        assert!(validate_mask_block_structural(&mask_empty_sigs).is_ok());
        let msg_empty_sigs = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: Some(mask_empty_sigs.clone()),
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let enc_empty_sigs = msg_empty_sigs.encode().expect("encode empty sigs mask");
        let dec_empty_sigs =
            HasMt1Message::decode(&enc_empty_sigs).expect("decode empty sigs mask");
        // Same 106 payload bits as the empty-satellites mask: six zero pad bits.
        assert_eq!(enc_empty_sigs.len(), 14);
        assert_eq!(dec_empty_sigs.padding_bits.len(), 6);
        assert!(
            dec_empty_sigs.padding_bits.iter().all(|bit| !bit),
            "trailing padding must be zero bits"
        );
        assert_eq!(
            dec_empty_sigs.encode().expect("re-encode empty sigs mask"),
            enc_empty_sigs
        );
        let dec_empty_sigs_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..dec_empty_sigs.clone()
        };
        assert_eq!(dec_empty_sigs_without_padding, msg_empty_sigs);
    }

    #[test]
    fn test_clock_subset_zero_systems_lossless_syntax() {
        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        // 1. Lossless roundtrip of Nsys_sub = 0
        let msg_sub0 = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask: mask.clone(),
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![],
                records: vec![],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        let enc = msg_sub0.encode().expect("encode clock subset Nsys_sub=0");
        let dec = HasMt1Message::decode(&enc).expect("decode clock subset Nsys_sub=0");
        // Header 32 + mask 74 + clock subset 8 = 114 payload bits, so the
        // encoder zero-pads the final byte with six bits.
        assert_eq!(enc.len(), 15);
        assert_eq!(dec.padding_bits.len(), 6);
        assert!(
            dec.padding_bits.iter().all(|bit| !bit),
            "trailing padding must be zero bits"
        );
        assert_eq!(
            dec.encode().expect("re-encode clock subset Nsys_sub=0"),
            enc
        );
        let dec_without_padding = HasMt1Message {
            padding_bits: Vec::new(),
            ..dec.clone()
        };
        assert_eq!(dec_without_padding, msg_sub0);
        let sub = dec.clock_subset.unwrap();
        assert_eq!(sub.validity_interval, 5);
        assert!(sub.systems.is_empty());
        assert!(sub.records.is_empty());

        // 2. Systems empty but records present -> Error::InvalidInput
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let msg_sub_inconsistent = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: true,
                code_bias: false,
                phase_bias: false,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask,
            orbit: None,
            clock_full_set: None,
            clock_subset: Some(HasClockBlock {
                validity_interval: 5,
                systems: vec![],
                records: vec![HasClockCorrection {
                    sat,
                    correction_m: Some(0.1),
                    do_not_use: false,
                }],
            }),
            code_bias: None,
            phase_bias: None,
            padding_bits: Vec::new(),
        };
        assert!(msg_sub_inconsistent.encode().is_err());
    }

    #[test]
    fn test_caller_edit_regression_phase_authority() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap();
        let mask = Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: vec![1],
                signals: vec![0],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        });

        let mut msg = HasMt1Message {
            header: HasMt1Header {
                toh_s: 0,
                mask: true,
                orbit: false,
                clock_full_set: false,
                clock_subset: false,
                code_bias: false,
                phase_bias: true,
                reserved: 0,
                mask_id: 1,
                iod_set_id: 1,
            },
            mask,
            orbit: None,
            clock_full_set: None,
            clock_subset: None,
            code_bias: None,
            phase_bias: Some(HasPhaseBiasBlock {
                validity_interval: 5,
                records: vec![HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: Some(1.0),
                    discontinuity_indicator: 0,
                }],
            }),
            padding_bits: Vec::new(),
        };

        // Step 1: Initial state derived from 1.0 cycle
        let expected_m1 = 1.0 * (C_M_S / F_L1_HZ);
        assert_eq!(
            msg.phase_bias.as_ref().unwrap().records[0].bias_m(),
            Some(expected_m1)
        );

        // Step 2: Caller mutates cycles in-place to 2.5 cycles
        msg.phase_bias.as_mut().unwrap().records[0].bias_cycles = Some(2.5);
        let expected_m2 = 2.5 * (C_M_S / F_L1_HZ);
        // Getter dynamically recomputes from cycle authority; old meters cannot remain!
        assert_eq!(
            msg.phase_bias.as_ref().unwrap().records[0].bias_m(),
            Some(expected_m2)
        );

        // Step 3: Wire encoding reflects changed cycles
        let enc = msg.encode().unwrap();
        let dec = HasMt1Message::decode(&enc).unwrap();
        let dec_rec = &dec.phase_bias.as_ref().unwrap().records[0];
        assert_eq!(dec_rec.bias_cycles, Some(2.5));
        assert_eq!(dec_rec.bias_m(), Some(expected_m2));

        // Step 4: SSR ingestion uses recomputed derived meters
        let reception = crate::astro::time::model::GnssWeekTow::new(
            crate::astro::time::model::TimeScale::Gst,
            1042,
            0.0,
        )
        .unwrap();
        let mut store = crate::ssr::SsrCorrectionStore::new();
        store.ingest_has_mt1(&dec, reception).unwrap();
        assert_eq!(store.phase_bias(sat, 0), Some(expected_m2));

        // Step 5: Caller mutates cycles to None (unavailable sentinel)
        msg.phase_bias.as_mut().unwrap().records[0].bias_cycles = None;
        assert_eq!(msg.phase_bias.as_ref().unwrap().records[0].bias_m(), None);
        assert_eq!(
            msg.phase_bias.as_ref().unwrap().records[0].conversion(),
            HasPhaseBiasConversion::TransmittedUnavailable
        );
        let enc_none = msg.encode().unwrap();
        let dec_none = HasMt1Message::decode(&enc_none).unwrap();
        assert_eq!(
            dec_none.phase_bias.as_ref().unwrap().records[0].bias_cycles,
            None
        );
        assert_eq!(
            dec_none.phase_bias.as_ref().unwrap().records[0].bias_m(),
            None
        );
    }
}
