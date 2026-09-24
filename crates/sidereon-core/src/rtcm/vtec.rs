//! SSR ionosphere VTEC spherical-harmonics messages: RTCM 1264 and the IGS SSR
//! message 4076 subtype 201.
//!
//! Both carry the same fields after their identification (RTCM 10403.3
//! DF472..DF478; IGS SSR v1.00 Tables 24-27, IDF035..IDF041): the SSR header
//! without a satellite count, a VTEC quality indicator, and one to four
//! ionospheric layers, each with its height, degree `N`, order `M`, cosine
//! coefficients `C_nm` for `m = 0..=M`, `n = m..=N` and sine coefficients
//! `S_nm` for `m = 1..=M`, `n = m..=N`, in that order. Every value is the raw
//! transmitted integer.

use crate::error::{Error, Result};

use super::bits::{BitReader, FieldWriter};
use super::ssr::IGS_SSR_MESSAGE_NUMBER;
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};

/// RTCM message number of the RTCM SSR VTEC message.
const RTCM_VTEC_MESSAGE_NUMBER: u16 = 1264;

/// IGS SSR message number (IDF002) of the VTEC message.
pub(crate) const IGS_SSR_VTEC_SUBTYPE: u8 = 201;

/// A decoded SSR ionosphere VTEC spherical-harmonics message: RTCM 1264, or
/// IGS SSR 4076 subtype 201.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrVtecMessage {
    /// 1264, or 4076 for the IGS SSR message.
    pub message_number: u16,
    /// IGS SSR version (IDF001, 3 bits) of a 4076 message; `None` for 1264.
    pub igs_ssr_version: Option<u8>,
    /// SSR epoch time, GPS seconds of week (DF385, IDF003, 20 bits).
    pub epoch_time_s: u32,
    /// SSR update interval index (DF391, IDF004, 4 bits).
    pub update_interval: u8,
    /// Multiple-message indicator (DF388, IDF005).
    pub multiple_message: bool,
    /// IOD SSR (DF413, IDF007, 4 bits).
    pub iod_ssr: u8,
    /// SSR provider ID (DF414, IDF008).
    pub provider_id: u16,
    /// SSR solution ID (DF415, IDF009, 4 bits).
    pub solution_id: u8,
    /// VTEC quality indicator (DF478, IDF041, 9 bits), scale 0.05 TECU.
    pub quality_indicator: u16,
    /// The ionospheric layers, one to four (the count is transmitted less one
    /// in DF472, IDF035).
    pub layers: Vec<SsrVtecLayer>,
    /// Every body bit after the last field, the zeros that align the body to a
    /// byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros: read under [`RtcmPolicy::Lenient`] and written back
    /// after the last field by `encode_with_policy` under that policy, so the
    /// body re-encodes byte for byte. Empty for every body read under
    /// [`RtcmPolicy::Strict`] and for a message built by hand; `encode`
    /// refuses a nonempty value.
    pub trailing_bits: Vec<bool>,
}

/// One ionospheric layer of an SSR VTEC message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsrVtecLayer {
    /// Height of the layer (DF473, IDF036, 8 bits), scale 10 km.
    pub height: u8,
    /// Spherical-harmonics degree `N`, `1..=16`; transmitted as `N - 1`
    /// (DF474, IDF037, 4 bits).
    pub degree: u8,
    /// Spherical-harmonics order `M`, `1..=16`; transmitted as `M - 1` (DF475,
    /// IDF038, 4 bits).
    pub order: u8,
    /// Cosine coefficients `C_nm` (DF476, IDF039, int16, scale 0.005 TECU) for
    /// `m = 0..=M`, `n = m..=N`, in that order.
    pub cosine: Vec<i16>,
    /// Sine coefficients `S_nm` (DF477, IDF040, int16, scale 0.005 TECU) for
    /// `m = 1..=M`, `n = m..=N`, in that order.
    pub sine: Vec<i16>,
}

impl SsrVtecLayer {
    /// The number of cosine and sine coefficients a layer of degree `degree`
    /// and order `order` carries: the terms of the sequence the formats state,
    /// `C_nm` for `m = 0..=M`, `n = m..=N` and `S_nm` for `m = 1..=M`,
    /// `n = m..=N`. For `M <= N` these are the counts of the closed formula of
    /// IDF039/IDF040, `(N+1)(N+2)/2 - (N-M)(N-M+1)/2` and that less `N + 1`; an
    /// order above the degree adds no term, as no `n` runs from `m` to `N`
    /// for `m > N`.
    pub fn coefficient_counts(degree: u8, order: u8) -> (usize, usize) {
        let n = usize::from(degree);
        let terms = |m: usize| (n + 1).saturating_sub(m);
        let cosine = (0..=usize::from(order)).map(terms).sum();
        let sine = (1..=usize::from(order)).map(terms).sum();
        (cosine, sine)
    }
}

/// Whether `message_number` is the RTCM SSR VTEC message.
pub(crate) fn is_rtcm_vtec(message_number: u16) -> bool {
    message_number == RTCM_VTEC_MESSAGE_NUMBER
}

impl SsrVtecMessage {
    /// Decode a 1264 or 4076 subtype 201 body (without the transport frame)
    /// under [`RtcmPolicy::Strict`]: bits after the last field other than the
    /// zero byte alignment are refused.
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        let igs_ssr_version = match message_number {
            RTCM_VTEC_MESSAGE_NUMBER => None,
            IGS_SSR_MESSAGE_NUMBER => {
                let version = r.u(3)? as u8;
                let subtype = r.u(8)? as u8;
                if subtype != IGS_SSR_VTEC_SUBTYPE {
                    return Err(Error::Parse(format!(
                        "IGS SSR message number {subtype} is not the VTEC message 201"
                    ))
                    .into());
                }
                Some(version)
            }
            _ => {
                return Err(Error::Parse(format!(
                    "message {message_number} is not an SSR VTEC message 1264/4076"
                ))
                .into())
            }
        };
        let epoch_time_s = r.u(20)? as u32;
        let update_interval = r.u(4)? as u8;
        let multiple_message = r.flag()?;
        let iod_ssr = r.u(4)? as u8;
        let provider_id = r.u(16)? as u16;
        let solution_id = r.u(4)? as u8;
        let quality_indicator = r.u(9)? as u16;
        let layer_count = r.u(2)? as usize + 1;
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let height = r.u(8)? as u8;
            let degree = r.u(4)? as u8 + 1;
            let order = r.u(4)? as u8 + 1;
            let (cosines, sines) = SsrVtecLayer::coefficient_counts(degree, order);
            let mut cosine = Vec::with_capacity(cosines);
            for _ in 0..cosines {
                cosine.push(r.i(16)? as i16);
            }
            let mut sine = Vec::with_capacity(sines);
            for _ in 0..sines {
                sine.push(r.i(16)? as i16);
            }
            layers.push(SsrVtecLayer {
                height,
                degree,
                order,
                cosine,
                sine,
            });
        }
        Ok(Self {
            message_number,
            igs_ssr_version,
            epoch_time_s,
            update_interval,
            multiple_message,
            iod_ssr,
            provider_id,
            solution_id,
            quality_indicator,
            layers,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this message body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming what the message cannot state: a message
    /// number other than 1264 and 4076, an IGS SSR version held for 1264 or
    /// missing for 4076, no layer or more than four, a degree or order outside
    /// `1..=16`, a coefficient list whose length differs from the count the
    /// degree and order give ([`SsrVtecLayer::coefficient_counts`]), or a value
    /// wider than its field.
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
        match (number, self.igs_ssr_version) {
            (RTCM_VTEC_MESSAGE_NUMBER, None) | (IGS_SSR_MESSAGE_NUMBER, Some(_)) => {}
            (RTCM_VTEC_MESSAGE_NUMBER, Some(_)) => {
                return Err(Error::InvalidInput(
                    "RTCM 1264 carries no IGS SSR version, and one is given".to_string(),
                ))
            }
            (IGS_SSR_MESSAGE_NUMBER, None) => {
                return Err(Error::InvalidInput(
                    "RTCM 4076 IGS SSR VTEC message carries an IGS SSR version, and none is \
                     given"
                        .to_string(),
                ))
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "RTCM message number {number} is not an SSR VTEC message 1264/4076"
                )))
            }
        }
        if !(1..=4).contains(&self.layers.len()) {
            return Err(Error::InvalidInput(format!(
                "RTCM {number} VTEC message holds {} layers; it carries one to four",
                self.layers.len()
            )));
        }
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        if let Some(version) = self.igs_ssr_version {
            w.u("IGS SSR version", u64::from(version), 3)?;
            w.u("IGS SSR message number", u64::from(IGS_SSR_VTEC_SUBTYPE), 8)?;
        }
        w.u("epoch time", u64::from(self.epoch_time_s), 20)?;
        w.u("update interval", u64::from(self.update_interval), 4)?;
        w.flag(self.multiple_message);
        w.u("IOD SSR", u64::from(self.iod_ssr), 4)?;
        w.u("provider ID", u64::from(self.provider_id), 16)?;
        w.u("solution ID", u64::from(self.solution_id), 4)?;
        w.u(
            "VTEC quality indicator",
            u64::from(self.quality_indicator),
            9,
        )?;
        w.u("number of layers", self.layers.len() as u64 - 1, 2)?;
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, value) in [("degree", layer.degree), ("order", layer.order)] {
                if !(1..=16).contains(&value) {
                    return Err(Error::InvalidInput(format!(
                        "RTCM {number} VTEC layer {index} {name} {value} is outside 1..=16"
                    )));
                }
            }
            let (cosines, sines) = SsrVtecLayer::coefficient_counts(layer.degree, layer.order);
            for (name, held, count) in [
                ("cosine", layer.cosine.len(), cosines),
                ("sine", layer.sine.len(), sines),
            ] {
                if held != count {
                    return Err(Error::InvalidInput(format!(
                        "RTCM {number} VTEC layer {index} holds {held} {name} coefficients; \
                         degree {} and order {} carry {count}",
                        layer.degree, layer.order
                    )));
                }
            }
            w.u(
                format_args!("layer {index} height"),
                u64::from(layer.height),
                8,
            )?;
            w.u(
                format_args!("layer {index} degree"),
                u64::from(layer.degree - 1),
                4,
            )?;
            w.u(
                format_args!("layer {index} order"),
                u64::from(layer.order - 1),
                4,
            )?;
            for &c in layer.cosine.iter().chain(&layer.sine) {
                w.i(format_args!("layer {index} coefficient"), i64::from(c), 16)?;
            }
        }
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

impl super::TrailingBits for SsrVtecMessage {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
