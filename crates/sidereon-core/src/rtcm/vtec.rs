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
use super::{
    decode_body, write_trailing, DecodeContext, DecodeResult, RtcmConversionError, RtcmDeparture,
    RtcmEncodeError, RtcmPolicy, RtcmRecordKind, VtecEvaluationProblem,
};

const VTEC_EARTH_RADIUS_M: f64 = 6_370_000.0;
const EARTH_ROTATION_RAD_S: f64 = 7.292_115_146_7e-5;
const VTEC_COEFFICIENT_SCALE_TECU: f64 = 0.005;

/// RTCM message number of the RTCM SSR VTEC message.
const RTCM_VTEC_MESSAGE_NUMBER: u16 = 1264;

/// IGS SSR message number (IDF002) of the VTEC message.
pub(crate) const IGS_SSR_VTEC_SUBTYPE: u8 = 201;

fn physical_result_out_of_range(field: &'static str) -> Error {
    RtcmConversionError::VtecEvaluation(VtecEvaluationProblem::PhysicalResultOutOfRange { field })
        .into()
}

fn ionospheric_delay_m(stec_tecu: f64, frequency_hz: f64) -> Option<f64> {
    if !stec_tecu.is_finite() || stec_tecu < 0.0 {
        return None;
    }
    if stec_tecu == 0.0 {
        return Some(0.0);
    }

    let (coefficient_fraction, coefficient_exponent) = libm::frexp(40.3e16);
    let (stec_fraction, stec_exponent) = libm::frexp(stec_tecu);
    let (frequency_fraction, frequency_exponent) = libm::frexp(frequency_hz);
    let fraction = coefficient_fraction * stec_fraction / (frequency_fraction * frequency_fraction);
    let exponent = coefficient_exponent + stec_exponent - 2 * frequency_exponent;
    let delay = libm::scalbn(fraction, exponent);
    (delay.is_finite() && delay > 0.0).then_some(delay)
}

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

/// The VTEC and mapped STEC contribution for one thin ionosphere layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrVtecLayerEvaluation {
    /// Geocentric latitude of the layer pierce point, radians.
    pub pierce_latitude_rad: f64,
    /// Geocentric longitude of the layer pierce point, radians in `[-pi, pi]`.
    pub pierce_longitude_rad: f64,
    /// Mean-sun-fixed longitude used by the harmonic model, radians in `[0, 2*pi)`.
    pub sun_fixed_longitude_rad: f64,
    /// Evaluated VTEC, TECU, after the specification's negative-value clamp.
    pub vtec_tecu: f64,
    /// Thin-shell mapping factor `1 / sin(elevation + central_angle)`.
    pub mapping_factor: f64,
    /// Layer slant TEC, TECU.
    pub stec_tecu: f64,
}

/// Evaluated IGS/RTCM VTEC model at one receiver-satellite geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct SsrVtecEvaluation {
    /// Per-layer geometry and TEC contributions, in message order.
    pub layers: Vec<SsrVtecLayerEvaluation>,
    /// Sum of layer slant TEC values, TECU.
    pub stec_tecu: f64,
    /// First-order ionospheric pseudorange delay at the requested frequency, m.
    pub pseudorange_delay_m: f64,
    /// First-order ionospheric carrier-phase advance at the requested frequency, m.
    pub phase_range_advance_m: f64,
}

impl SsrVtecLayer {
    /// The number of cosine and sine coefficients a layer of degree `degree`
    /// and order `order` carries: the terms of the sequence the formats state,
    /// `C_nm` for `m = 0..=M`, `n = m..=N` and `S_nm` for `m = 1..=M`,
    /// `n = m..=N`. IGS SSR v1.00 §4.5.1 requires `M <= N`; callers validating
    /// a message must enforce that relation before using these counts. Lenient
    /// codec handling of `M > N` uses this degree-limited sequence as an
    /// explicit nonconforming interpretation; for `M >= N + 2` it differs
    /// from the IGS coefficient-count formula.
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
    /// Evaluate this model using the IGS SSR v1.00 thin-shell spherical-harmonic
    /// definition. Satellite coordinates are ECEF at signal transmission and
    /// are rotated into the reception frame using the geometric light time.
    /// `gps_seconds_of_day` is the transmitted SSR computation epoch modulo one
    /// GPS day, not the later time at which a retained model is queried.
    /// Coefficients are converted from their transmitted integer scale of
    /// 0.005 TECU; Legendre functions use the fully normalized, no-Condon-Shortley
    /// convention. The reserved -32768 coefficient sentinel is refused rather
    /// than interpreted as a physical value. Negative layer VTEC is replaced
    /// by zero as required by IGS.
    pub fn evaluate(
        &self,
        receiver_ecef_m: [f64; 3],
        satellite_transmit_ecef_m: [f64; 3],
        gps_seconds_of_day: f64,
        frequency_hz: f64,
    ) -> Result<SsrVtecEvaluation> {
        if !gps_seconds_of_day.is_finite() || !(0.0..86_400.0).contains(&gps_seconds_of_day) {
            return Err(RtcmConversionError::VtecEvaluation(
                VtecEvaluationProblem::ComputationTime,
            )
            .into());
        }
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(
                RtcmConversionError::VtecEvaluation(VtecEvaluationProblem::Frequency).into(),
            );
        }
        if receiver_ecef_m
            .iter()
            .chain(&satellite_transmit_ecef_m)
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(RtcmConversionError::VtecEvaluation(
                VtecEvaluationProblem::NonFiniteCoordinates,
            )
            .into());
        }
        if self.layers.is_empty() {
            return Err(
                RtcmConversionError::VtecEvaluation(VtecEvaluationProblem::LayerCount {
                    layers: 0,
                })
                .into(),
            );
        }
        if !(1..=4).contains(&self.layers.len()) {
            return Err(
                RtcmConversionError::VtecEvaluation(VtecEvaluationProblem::LayerCount {
                    layers: self.layers.len(),
                })
                .into(),
            );
        }
        if !matches!(
            (self.message_number, self.igs_ssr_version),
            (RTCM_VTEC_MESSAGE_NUMBER, None) | (IGS_SSR_MESSAGE_NUMBER, Some(0..=7))
        ) {
            return Err(RtcmConversionError::VtecEvaluation(
                VtecEvaluationProblem::MessageIdentity {
                    message_number: self.message_number,
                },
            )
            .into());
        }

        let receiver_radius = norm(receiver_ecef_m);
        let range = norm(subtract(satellite_transmit_ecef_m, receiver_ecef_m));
        if !receiver_radius.is_finite()
            || !range.is_finite()
            || receiver_radius <= 0.0
            || range <= 0.0
        {
            return Err(RtcmConversionError::VtecEvaluation(
                VtecEvaluationProblem::InvalidGeometry,
            )
            .into());
        }
        let light_time_s = range / crate::constants::C_M_S;
        let earth_rotation = EARTH_ROTATION_RAD_S * light_time_s;
        let (sin_rotation, cos_rotation) = libm::sincos(earth_rotation);
        let satellite_ecef_m = [
            cos_rotation * satellite_transmit_ecef_m[0]
                + sin_rotation * satellite_transmit_ecef_m[1],
            -sin_rotation * satellite_transmit_ecef_m[0]
                + cos_rotation * satellite_transmit_ecef_m[1],
            satellite_transmit_ecef_m[2],
        ];
        let receiver_latitude = libm::asin((receiver_ecef_m[2] / receiver_radius).clamp(-1.0, 1.0));
        let receiver_longitude = libm::atan2(receiver_ecef_m[1], receiver_ecef_m[0]);
        let receive_frame_range = norm(subtract(satellite_ecef_m, receiver_ecef_m));
        if !receive_frame_range.is_finite() || receive_frame_range <= 0.0 {
            return Err(RtcmConversionError::VtecEvaluation(
                VtecEvaluationProblem::InvalidGeometry,
            )
            .into());
        }
        let line_of_sight = scale(
            subtract(satellite_ecef_m, receiver_ecef_m),
            1.0 / receive_frame_range,
        );
        let (sin_latitude, cos_latitude) = libm::sincos(receiver_latitude);
        let (sin_longitude, cos_longitude) = libm::sincos(receiver_longitude);
        let east = [-sin_longitude, cos_longitude, 0.0];
        let north = [
            -sin_latitude * cos_longitude,
            -sin_latitude * sin_longitude,
            cos_latitude,
        ];
        let up = [
            cos_latitude * cos_longitude,
            cos_latitude * sin_longitude,
            sin_latitude,
        ];
        let elevation = libm::asin(dot(line_of_sight, up).clamp(-1.0, 1.0));
        if elevation < 0.0 {
            return Err(
                RtcmConversionError::VtecEvaluation(VtecEvaluationProblem::BelowHorizon).into(),
            );
        }
        let azimuth = libm::atan2(dot(line_of_sight, east), dot(line_of_sight, north));

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut stec_tecu = 0.0;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if !(1..=16).contains(&layer.degree)
                || !(1..=16).contains(&layer.order)
                || layer.order > layer.degree
            {
                return Err(RtcmConversionError::VtecEvaluation(
                    VtecEvaluationProblem::LayerDegreeOrder {
                        layer_index,
                        degree: layer.degree,
                        order: layer.order,
                    },
                )
                .into());
            }
            let (cosine_count, sine_count) =
                SsrVtecLayer::coefficient_counts(layer.degree, layer.order);
            if layer.cosine.len() != cosine_count || layer.sine.len() != sine_count {
                return Err(RtcmConversionError::VtecEvaluation(
                    VtecEvaluationProblem::CoefficientCounts {
                        layer_index,
                        cosine_expected: cosine_count,
                        cosine_actual: layer.cosine.len(),
                        sine_expected: sine_count,
                        sine_actual: layer.sine.len(),
                    },
                )
                .into());
            }
            if layer
                .cosine
                .iter()
                .chain(&layer.sine)
                .any(|&coefficient| coefficient == i16::MIN)
            {
                return Err(RtcmConversionError::VtecEvaluation(
                    VtecEvaluationProblem::UnavailableCoefficient { layer_index },
                )
                .into());
            }
            let shell_radius = VTEC_EARTH_RADIUS_M + f64::from(layer.height) * 10_000.0;
            if shell_radius <= receiver_radius {
                return Err(RtcmConversionError::VtecEvaluation(
                    VtecEvaluationProblem::ShellNotAboveReceiver { layer_index },
                )
                .into());
            }
            let ratio = (receiver_radius / shell_radius * libm::cos(elevation)).clamp(-1.0, 1.0);
            let central_angle = core::f64::consts::FRAC_PI_2 - elevation - libm::asin(ratio);
            let (sin_central, cos_central) = libm::sincos(central_angle);
            let pierce = [
                cos_central * up[0]
                    + sin_central * (libm::cos(azimuth) * north[0] + libm::sin(azimuth) * east[0]),
                cos_central * up[1]
                    + sin_central * (libm::cos(azimuth) * north[1] + libm::sin(azimuth) * east[1]),
                cos_central * up[2]
                    + sin_central * (libm::cos(azimuth) * north[2] + libm::sin(azimuth) * east[2]),
            ];
            let pierce_latitude = libm::asin(pierce[2].clamp(-1.0, 1.0));
            let pierce_longitude = libm::atan2(pierce[1], pierce[0]);
            let sun_fixed_longitude = (pierce_longitude
                + (gps_seconds_of_day - 50_400.0) * core::f64::consts::PI / 43_200.0)
                .rem_euclid(2.0 * core::f64::consts::PI);
            let legendre = fully_normalized_legendre(
                pierce_latitude,
                usize::from(layer.degree),
                usize::from(layer.order).min(usize::from(layer.degree)),
            );
            let mut cosine_index = 0;
            let mut sine_index = 0;
            let mut vtec = 0.0;
            for order in 0..=usize::from(layer.order).min(usize::from(layer.degree)) {
                for degree in order..=usize::from(layer.degree) {
                    let phase = order as f64 * sun_fixed_longitude;
                    let cosine = f64::from(*layer.cosine.get(cosine_index).ok_or_else(|| {
                        Error::from(RtcmConversionError::VtecEvaluation(
                            VtecEvaluationProblem::MissingCoefficient {
                                layer_index,
                                field: "cosine",
                                index: cosine_index,
                            },
                        ))
                    })?) * VTEC_COEFFICIENT_SCALE_TECU;
                    cosine_index += 1;
                    vtec += cosine * libm::cos(phase) * legendre[degree][order];
                    if order > 0 {
                        let sine = f64::from(*layer.sine.get(sine_index).ok_or_else(|| {
                            Error::from(RtcmConversionError::VtecEvaluation(
                                VtecEvaluationProblem::MissingCoefficient {
                                    layer_index,
                                    field: "sine",
                                    index: sine_index,
                                },
                            ))
                        })?) * VTEC_COEFFICIENT_SCALE_TECU;
                        sine_index += 1;
                        vtec += sine * libm::sin(phase) * legendre[degree][order];
                    }
                }
            }
            if !vtec.is_finite() {
                return Err(physical_result_out_of_range("VTEC"));
            }
            let vtec_tecu = vtec.max(0.0);
            let mapping_denominator = libm::sin(elevation + central_angle);
            if mapping_denominator <= 0.0 {
                return Err(RtcmConversionError::VtecEvaluation(
                    VtecEvaluationProblem::InvalidMappingFactor { layer_index },
                )
                .into());
            }
            let mapping_factor = 1.0 / mapping_denominator;
            if !mapping_factor.is_finite() {
                return Err(physical_result_out_of_range("mapping factor"));
            }
            let layer_stec = vtec_tecu * mapping_factor;
            if !layer_stec.is_finite() {
                return Err(physical_result_out_of_range("layer STEC"));
            }
            stec_tecu += layer_stec;
            if !stec_tecu.is_finite() {
                return Err(physical_result_out_of_range("STEC"));
            }
            layers.push(SsrVtecLayerEvaluation {
                pierce_latitude_rad: pierce_latitude,
                pierce_longitude_rad: pierce_longitude,
                sun_fixed_longitude_rad: sun_fixed_longitude,
                vtec_tecu,
                mapping_factor,
                stec_tecu: layer_stec,
            });
        }
        let pseudorange_delay_m = ionospheric_delay_m(stec_tecu, frequency_hz)
            .ok_or_else(|| physical_result_out_of_range("pseudorange delay"))?;
        Ok(SsrVtecEvaluation {
            layers,
            stec_tecu,
            pseudorange_delay_m,
            phase_range_advance_m: -pseudorange_delay_m,
        })
    }

    /// Decode a 1264 or 4076 subtype 201 body (without the transport frame)
    /// under [`RtcmPolicy::Strict`]: bits after the last field other than the
    /// zero byte alignment are refused.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_with_policy(body, RtcmPolicy::Strict).map(|(message, _)| message)
    }

    /// Decode under `policy`, returning departures read under the lenient
    /// policy. For an order above degree, lenient decoding follows the
    /// transmitted coefficient sequence (`m` through `min(M, N)`); this is
    /// explicitly nonconforming and for `M >= N + 2` differs from the IGS
    /// count formula. Strict decoding refuses the layer.
    pub fn decode_with_policy(
        body: &[u8],
        policy: RtcmPolicy,
    ) -> Result<(Self, Vec<RtcmDeparture>)> {
        let mut ctx = DecodeContext::new(policy);
        let message = decode_body(body, &mut ctx, Self::read)?;
        Ok((message, ctx.into_departures()))
    }

    pub(crate) fn read(r: &mut BitReader<'_>, ctx: &mut DecodeContext) -> DecodeResult<Self> {
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
                .into());
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
        for layer_index in 0..layer_count {
            let height = r.u(8)? as u8;
            let degree = r.u(4)? as u8 + 1;
            let order = r.u(4)? as u8 + 1;
            if order > degree {
                ctx.depart(RtcmDeparture::OrderExceedsDegree {
                    message_number,
                    layer_index,
                    degree,
                    order,
                })?;
            }
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
    /// [`Error::RtcmEncode`] naming what the message cannot state: a message
    /// number other than 1264 and 4076, an IGS SSR version held for 1264 or
    /// missing for 4076, no layer or more than four, a degree or order outside
    /// `1..=16`, an order greater than degree under strict policy, a coefficient
    /// list whose length differs from the count the
    /// degree and order give ([`SsrVtecLayer::coefficient_counts`]), or a value
    /// wider than its field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`], an
    /// order above degree uses the degree-limited coefficient sequence and is
    /// reported as [`RtcmDeparture::OrderExceedsDegree`]; for `M >= N + 2`
    /// this interpretation differs from the IGS count formula. Nonempty
    /// `trailing_bits` are also written and reported as
    /// [`RtcmDeparture::TrailingBits`]. Other refusals apply under both
    /// policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        match (number, self.igs_ssr_version) {
            (RTCM_VTEC_MESSAGE_NUMBER, None) | (IGS_SSR_MESSAGE_NUMBER, Some(_)) => {}
            (RTCM_VTEC_MESSAGE_NUMBER, Some(_)) => {
                return Err(RtcmEncodeError::FieldPresence {
                    message_number: number,
                    record: RtcmRecordKind::SsrVtec {
                        message_number: number,
                    },
                    field: "IGS SSR version",
                    carried: false,
                }
                .into());
            }
            (IGS_SSR_MESSAGE_NUMBER, None) => {
                return Err(RtcmEncodeError::FieldPresence {
                    message_number: number,
                    record: RtcmRecordKind::SsrVtec {
                        message_number: number,
                    },
                    field: "IGS SSR version",
                    carried: true,
                }
                .into());
            }
            _ => {
                return Err(RtcmEncodeError::MessageNumber {
                    message_number: number,
                    record: RtcmRecordKind::SsrVtec {
                        message_number: number,
                    },
                }
                .into());
            }
        }
        if !(1..=4).contains(&self.layers.len()) {
            return Err(RtcmEncodeError::ValueOutOfRange {
                message_number: number,
                field: "VTEC layer count".to_string(),
                value: self.layers.len() as i128,
                minimum: 1,
                maximum: 4,
            }
            .into());
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
        let mut departures = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, value) in [("degree", layer.degree), ("order", layer.order)] {
                if !(1..=16).contains(&value) {
                    return Err(RtcmEncodeError::ValueOutOfRange {
                        message_number: number,
                        field: format!("VTEC layer {index} {name}"),
                        value: i128::from(value),
                        minimum: 1,
                        maximum: 16,
                    }
                    .into());
                }
            }
            if layer.order > layer.degree {
                let departure = RtcmDeparture::OrderExceedsDegree {
                    message_number: number,
                    layer_index: index,
                    degree: layer.degree,
                    order: layer.order,
                };
                if policy == RtcmPolicy::Strict {
                    return Err(RtcmEncodeError::StrictDeparture(departure).into());
                }
                departures.push(departure);
            }
            let (cosines, sines) = SsrVtecLayer::coefficient_counts(layer.degree, layer.order);
            for (name, held, count) in [
                ("cosine", layer.cosine.len(), cosines),
                ("sine", layer.sine.len(), sines),
            ] {
                if held != count {
                    return Err(RtcmEncodeError::CountMismatch {
                        message_number: number,
                        field: if name == "cosine" {
                            "VTEC cosine coefficient"
                        } else {
                            "VTEC sine coefficient"
                        },
                        expected: count,
                        actual: held,
                    }
                    .into());
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
        departures.extend(write_trailing(&mut w, &self.trailing_bits, policy)?);
        Ok((w.into_bytes(), departures))
    }
}

fn norm(vector: [f64; 3]) -> f64 {
    libm::sqrt(dot(vector, vector))
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn fully_normalized_legendre(latitude_rad: f64, degree: usize, order: usize) -> [[f64; 17]; 17] {
    let x = libm::sin(latitude_rad);
    let cos_latitude = libm::cos(latitude_rad);
    let mut values = [[0.0; 17]; 17];
    values[0][0] = 1.0;
    for m in 0..=order {
        if m > 0 {
            values[m][m] = (2 * m - 1) as f64 * cos_latitude * values[m - 1][m - 1];
        }
        if m < degree {
            values[m + 1][m] = (2 * m + 1) as f64 * x * values[m][m];
        }
        for n in (m + 2)..=degree {
            values[n][m] = ((2 * n - 1) as f64 * x * values[n - 1][m]
                - (n + m - 1) as f64 * values[n - 2][m])
                / (n - m) as f64;
        }
    }
    for n in 0..=degree {
        for m in 0..=n.min(order) {
            let mut factorial_ratio = 1.0;
            for k in (n - m + 1)..=(n + m) {
                factorial_ratio /= k as f64;
            }
            let multiplicity = if m == 0 { 1.0 } else { 2.0 };
            let normalization = libm::sqrt((2 * n + 1) as f64 * multiplicity * factorial_ratio);
            values[n][m] *= normalization;
        }
    }
    values
}

impl super::TrailingBits for SsrVtecMessage {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn packed(fields: &[(u64, usize)]) -> Vec<u8> {
        let bit_count: usize = fields.iter().map(|(_, width)| width).sum();
        let mut bytes = vec![0u8; bit_count.div_ceil(8)];
        let mut bit = 0;
        for &(value, width) in fields {
            for shift in (0..width).rev() {
                if value >> shift & 1 != 0 {
                    bytes[bit / 8] |= 1 << (7 - bit % 8);
                }
                bit += 1;
            }
        }
        bytes
    }

    fn twos(value: i16) -> u64 {
        u64::from(value as u16)
    }

    fn malformed_order_body(message_number: u16, order: u8) -> Vec<u8> {
        let mut fields = vec![(u64::from(message_number), 12)];
        if message_number == 4076 {
            fields.extend([(1, 3), (u64::from(IGS_SSR_VTEC_SUBTYPE), 8)]);
        }
        fields.extend([
            (50_400, 20),
            (0, 4),
            (0, 1),
            (3, 4),
            (0x1234, 16),
            (2, 4),
            (7, 9),
            (0, 2),
            (45, 8),
            (0, 4),
            (u64::from(order - 1), 4),
            (twos(100), 16),
            (twos(-200), 16),
            (twos(300), 16),
            (twos(400), 16),
        ]);
        packed(&fields)
    }

    fn constant_model(cosine: Vec<i16>) -> SsrVtecMessage {
        let (degree, order, sine) = if cosine.len() == 3 {
            (1, 1, vec![0])
        } else {
            (1, 0, Vec::new())
        };
        SsrVtecMessage {
            message_number: 4076,
            igs_ssr_version: Some(1),
            epoch_time_s: 50_400,
            update_interval: 0,
            multiple_message: false,
            iod_ssr: 0,
            provider_id: 256,
            solution_id: 0,
            quality_indicator: 1,
            layers: vec![SsrVtecLayer {
                height: 45,
                degree,
                order,
                cosine,
                sine,
            }],
            trailing_bits: Vec::new(),
        }
    }

    fn vtec_problem(result: Result<SsrVtecEvaluation>) -> VtecEvaluationProblem {
        match result {
            Err(Error::RtcmConversion(error)) => match *error {
                RtcmConversionError::VtecEvaluation(problem) => problem,
                other => panic!("expected a VTEC evaluation refusal, got {other:?}"),
            },
            other => panic!("expected a VTEC evaluation refusal, got {other:?}"),
        }
    }

    /// Independent closed-form check of IGS SSR v1.00 §4.5.1: at the north
    /// pole P00=1 and fully normalized P10=sqrt(3); IDF039 scales each raw
    /// coefficient by 0.005 TECU. Zenith gives a unit thin-shell mapping factor.
    #[test]
    fn vtec_matches_igs_v100_harmonic_scale_mapping_and_range_signs() {
        let model = constant_model(vec![100, 200, 0]);
        let evaluated = model
            .evaluate(
                [0.0, 0.0, VTEC_EARTH_RADIUS_M],
                [0.0, 0.0, 26_000_000.0],
                50_400.0,
                1.0e9,
            )
            .unwrap();
        let expected_vtec_tecu = 0.5 + libm::sqrt(3.0);
        assert!((evaluated.layers[0].vtec_tecu - expected_vtec_tecu).abs() < 1.0e-12);
        assert!((evaluated.layers[0].mapping_factor - 1.0).abs() < 1.0e-12);
        assert!((evaluated.stec_tecu - expected_vtec_tecu).abs() < 1.0e-12);
        let expected_delay_m = 40.3e16 * expected_vtec_tecu / 1.0e18;
        assert!((evaluated.pseudorange_delay_m - expected_delay_m).abs() < 1.0e-12);
        assert_eq!(
            evaluated.phase_range_advance_m,
            -evaluated.pseudorange_delay_m
        );
    }

    /// The IGS SSR v1.00 mean-sun-fixed longitude rotates with computation
    /// time. An equatorial off-zenith ray keeps P11=sqrt(3), while its
    /// independently computed shell intersection and Sagnac rotation test the
    /// m=1 cosine/sine sum and thin-shell mapping away from singular geometry.
    /// The arithmetic budget covers 192 operations across both geometry paths
    /// and assumes two unit roundoffs for each of at most 40 libm results.
    #[test]
    fn vtec_matches_igs_sun_fixed_longitude_harmonics() {
        let mut model = constant_model(vec![0, 0, 100]);
        model.layers[0].sine = vec![200];
        let receiver = [VTEC_EARTH_RADIUS_M, 0.0, 0.0];
        let transmit_elevation = core::f64::consts::FRAC_PI_6;
        let geometric_range: f64 = 20_000_000.0;
        let satellite_tx = [
            receiver[0] + geometric_range * libm::sin(transmit_elevation),
            geometric_range * libm::cos(transmit_elevation),
            0.0,
        ];
        let dx_tx = satellite_tx[0] - receiver[0];
        let dy_tx = satellite_tx[1];
        let geometric_range = libm::sqrt(dx_tx * dx_tx + dy_tx * dy_tx);
        let rotation = EARTH_ROTATION_RAD_S * geometric_range / crate::constants::C_M_S;
        let satellite_rx = [
            libm::cos(rotation) * satellite_tx[0] + libm::sin(rotation) * satellite_tx[1],
            -libm::sin(rotation) * satellite_tx[0] + libm::cos(rotation) * satellite_tx[1],
            satellite_tx[2],
        ];
        let dx = satellite_rx[0] - receiver[0];
        let dy = satellite_rx[1];
        let elevation = libm::atan2(dx, dy.abs());
        let azimuth = libm::atan2(dy, 0.0);
        let shell_radius = VTEC_EARTH_RADIUS_M + 450_000.0;
        let shell_ratio = VTEC_EARTH_RADIUS_M / shell_radius * libm::cos(elevation);
        let central_angle = core::f64::consts::FRAC_PI_2 - elevation - libm::asin(shell_ratio);
        let expected_pierce_longitude = central_angle * libm::sin(azimuth);
        let expected_mapping_factor = 1.0 / libm::sin(elevation + central_angle);
        let expected_phase = (expected_pierce_longitude + core::f64::consts::PI / 3.0)
            .rem_euclid(2.0 * core::f64::consts::PI);
        let evaluated = model
            .evaluate(receiver, satellite_tx, 64_800.0, 1.0e9)
            .unwrap();
        let unit_roundoff = 0.5 * f64::EPSILON;
        let gamma =
            |operations: f64| operations * unit_roundoff / (1.0 - operations * unit_roundoff);
        let geometry_condition = 1.0
            + norm(satellite_tx) / geometric_range
            + 1.0 / libm::cos(elevation).abs()
            + shell_ratio.abs() / libm::sqrt(1.0 - shell_ratio * shell_ratio);
        let geometry_angle_bound = gamma(192.0) * geometry_condition + 80.0 * unit_roundoff;
        let phase_bound =
            geometry_angle_bound + gamma(4.0) * (1.0 + expected_phase.abs()) + 4.0 * unit_roundoff;
        let phase_error = (evaluated.layers[0].sun_fixed_longitude_rad - expected_phase
            + core::f64::consts::PI)
            .rem_euclid(2.0 * core::f64::consts::PI)
            - core::f64::consts::PI;
        let expected_vtec =
            libm::sqrt(3.0) * (0.5 * libm::cos(expected_phase) + libm::sin(expected_phase));
        let coefficient_sum_tecu = (100.0_f64 * VTEC_COEFFICIENT_SCALE_TECU).abs()
            + (200.0_f64 * VTEC_COEFFICIENT_SCALE_TECU).abs();
        let vtec_bound = libm::sqrt(3.0) * coefficient_sum_tecu * phase_bound
            + gamma(24.0) * (1.0 + expected_vtec.abs());
        let mapping_angle = elevation + central_angle;
        let mapping_angle_bound =
            2.0 * geometry_angle_bound + gamma(3.0) * (1.0 + mapping_angle.abs());
        let mapping_bound = libm::cos(mapping_angle).abs() / libm::sin(mapping_angle).powi(2)
            * mapping_angle_bound
            + gamma(4.0) * expected_mapping_factor.abs();
        let expected_stec = expected_vtec * expected_mapping_factor;
        let stec_bound = expected_mapping_factor.abs() * vtec_bound
            + expected_vtec.abs() * mapping_bound
            + gamma(4.0) * expected_stec.abs();
        assert!(
            elevation > 0.4 && elevation < 0.7,
            "well-conditioned off-zenith ray"
        );
        assert!(
            evaluated.layers[0].pierce_latitude_rad.abs() <= geometry_angle_bound,
            "equatorial pierce latitude exceeded geometry bound {geometry_angle_bound}"
        );
        assert!(
            (evaluated.layers[0].pierce_longitude_rad - expected_pierce_longitude).abs()
                <= geometry_angle_bound,
            "pierce longitude error exceeded operation/conditioning bound {geometry_angle_bound}"
        );
        assert!(
            phase_error.abs() <= phase_bound,
            "sun-fixed longitude error exceeded propagated bound {phase_bound}"
        );
        assert!(
            (evaluated.layers[0].vtec_tecu - expected_vtec).abs() <= vtec_bound,
            "VTEC error exceeded propagated bound {vtec_bound}"
        );
        assert!(
            (evaluated.stec_tecu - expected_stec).abs() <= stec_bound,
            "STEC error exceeded propagated bound {stec_bound}"
        );
    }

    /// At a polar receiver, rotating the satellite around the Earth axis does
    /// not change its 30-degree elevation. The pierce latitude and shell
    /// mapping therefore have the direct spherical-trigonometry values below,
    /// independently of the implementation's ECEF basis calculation.
    #[test]
    fn vtec_uses_igs_thin_shell_geometry_off_zenith() {
        let model = constant_model(vec![100, 200, 0]);
        let elevation = core::f64::consts::FRAC_PI_6;
        let range_m: f64 = 20_000_000.0;
        let satellite = [
            range_m * libm::cos(elevation),
            0.0,
            VTEC_EARTH_RADIUS_M + range_m * libm::sin(elevation),
        ];
        let evaluated = model
            .evaluate([0.0, 0.0, VTEC_EARTH_RADIUS_M], satellite, 50_400.0, 1.0e9)
            .unwrap();
        let shell_radius_m = VTEC_EARTH_RADIUS_M + 450_000.0;
        let central_angle = core::f64::consts::FRAC_PI_2
            - elevation
            - libm::asin(VTEC_EARTH_RADIUS_M / shell_radius_m * libm::cos(elevation));
        let expected_mapping = 1.0 / libm::sin(elevation + central_angle);
        let pierce_latitude = core::f64::consts::FRAC_PI_2 - central_angle;
        let expected_vtec = 0.5 + libm::sqrt(3.0) * libm::sin(pierce_latitude);
        assert!((evaluated.layers[0].mapping_factor - expected_mapping).abs() < 1.0e-12);
        assert!((evaluated.layers[0].pierce_latitude_rad - pierce_latitude).abs() < 1.0e-12);
        assert!((evaluated.layers[0].vtec_tecu - expected_vtec).abs() < 1.0e-12);
        assert!((evaluated.stec_tecu - expected_vtec * expected_mapping).abs() < 1.0e-12);
    }

    /// IGS SSR v1.00 §4.5.1 requires negative layer VTEC to contribute zero,
    /// rather than a negative ionospheric delay.
    #[test]
    fn negative_vtec_is_clamped_per_igs_v100() {
        let model = constant_model(vec![-100, 0, 0]);
        let evaluated = model
            .evaluate(
                [0.0, 0.0, VTEC_EARTH_RADIUS_M],
                [0.0, 0.0, 26_000_000.0],
                50_400.0,
                1.0e-200,
            )
            .unwrap();
        assert_eq!(evaluated.layers[0].vtec_tecu, 0.0);
        assert_eq!(evaluated.stec_tecu, 0.0);
        assert_eq!(evaluated.pseudorange_delay_m, 0.0);
        assert_eq!(evaluated.phase_range_advance_m, 0.0);
    }

    #[test]
    fn vtec_delay_scales_frequency_without_intermediate_square_overflow() {
        let model = constant_model(vec![100, 200, 0]);
        let evaluated = model
            .evaluate(
                [0.0, 0.0, VTEC_EARTH_RADIUS_M],
                [0.0, 0.0, 26_000_000.0],
                50_400.0,
                1.0e160,
            )
            .expect("finite physical delay despite overflowing frequency square");
        let expected = (40.3e16 / 1.0e160) * (0.5 + libm::sqrt(3.0)) / 1.0e160;
        assert!(evaluated.pseudorange_delay_m.is_finite());
        assert!(evaluated.pseudorange_delay_m > 0.0);
        assert!((evaluated.pseudorange_delay_m - expected).abs() <= expected * 1.0e-14);
        assert_eq!(
            evaluated.phase_range_advance_m,
            -evaluated.pseudorange_delay_m
        );
    }

    #[test]
    fn vtec_delay_reports_only_unrepresentable_final_results() {
        let model = constant_model(vec![100, 200, 0]);
        let receiver = [0.0, 0.0, VTEC_EARTH_RADIUS_M];
        let satellite = [0.0, 0.0, 26_000_000.0];
        for frequency_hz in [1.0e-200, 1.0e200] {
            assert_eq!(
                vtec_problem(model.evaluate(receiver, satellite, 50_400.0, frequency_hz)),
                VtecEvaluationProblem::PhysicalResultOutOfRange {
                    field: "pseudorange delay"
                }
            );
        }
        assert_eq!(
            vtec_problem(model.evaluate(receiver, satellite, 50_400.0, 0.0)),
            VtecEvaluationProblem::Frequency
        );
    }

    #[test]
    fn vtec_invalid_harmonic_model_returns_typed_evaluation_problem() {
        let mut model = constant_model(vec![100, 200, 0]);
        model.layers[0].order = 2;
        assert_eq!(
            vtec_problem(model.evaluate(
                [0.0, 0.0, VTEC_EARTH_RADIUS_M],
                [0.0, 0.0, 26_000_000.0],
                50_400.0,
                1.0e9,
            )),
            VtecEvaluationProblem::LayerDegreeOrder {
                layer_index: 0,
                degree: 1,
                order: 2,
            }
        );
    }

    /// IDF039/IDF040 reserve -163.84 TECU (raw -32768) for unavailable or
    /// out-of-range coefficients; it is not a negative physical coefficient.
    #[test]
    fn unavailable_vtec_coefficient_is_not_evaluated_as_zero_tec() {
        let model = constant_model(vec![i16::MIN, 0, 0]);
        let result = model.evaluate(
            [0.0, 0.0, VTEC_EARTH_RADIUS_M],
            [0.0, 0.0, 26_000_000.0],
            50_400.0,
            1.0e9,
        );
        assert_eq!(
            vtec_problem(result),
            VtecEvaluationProblem::UnavailableCoefficient { layer_index: 0 }
        );
    }

    #[test]
    fn vtec_native_and_igs_valid_packed_vectors_obey_order_at_most_degree() {
        let native_fields = [
            (1264, 12),
            (50_400, 20),
            (0, 4),
            (0, 1),
            (3, 4),
            (0x1234, 16),
            (2, 4),
            (7, 9),
            (0, 2),
            (45, 8),
            (1, 4),
            (0, 4),
            (twos(100), 16),
            (twos(-200), 16),
            (twos(300), 16),
            (twos(400), 16),
            (twos(-500), 16),
            (twos(0), 16),
            (twos(50), 16),
        ];
        let native_body = packed(&native_fields);
        let native = SsrVtecMessage::decode(&native_body).expect("packed native VTEC");
        assert_eq!(native.message_number, 1264);
        assert_eq!(native.igs_ssr_version, None);
        assert_eq!(native.layers[0].degree, 2);
        assert_eq!(native.layers[0].order, 1);
        assert_eq!(native.layers[0].cosine, [100, -200, 300, 400, -500]);
        assert_eq!(native.layers[0].sine, [0, 50]);
        assert_eq!(native.encode().expect("encode native VTEC"), native_body);

        let igs_fields = [
            (4076, 12),
            (1, 3),
            (201, 8),
            (50_400, 20),
            (0, 4),
            (0, 1),
            (3, 4),
            (0x1234, 16),
            (2, 4),
            (7, 9),
            (0, 2),
            (45, 8),
            (1, 4),
            (0, 4),
            (twos(-1), 16),
            (twos(2), 16),
            (twos(0), 16),
            (twos(3), 16),
            (twos(-4), 16),
            (twos(5), 16),
            (twos(6), 16),
        ];
        let igs_body = packed(&igs_fields);
        let igs = SsrVtecMessage::decode(&igs_body).expect("packed IGS VTEC");
        assert_eq!(igs.message_number, 4076);
        assert_eq!(igs.igs_ssr_version, Some(1));
        assert_eq!(igs.layers[0].degree, 2);
        assert_eq!(igs.layers[0].order, 1);
        assert_eq!(igs.layers[0].cosine, [-1, 2, 0, 3, -4]);
        assert_eq!(igs.layers[0].sine, [5, 6]);
        assert_eq!(igs.encode().expect("encode IGS VTEC"), igs_body);
    }

    #[test]
    fn vtec_order_above_degree_policy_round_trips_degree_limited_wire_sequence() {
        for message_number in [1264, 4076] {
            for order in [2, 3] {
                let body = malformed_order_body(message_number, order);
                let departure = RtcmDeparture::OrderExceedsDegree {
                    message_number,
                    layer_index: 0,
                    degree: 1,
                    order,
                };

                assert!(SsrVtecMessage::decode(&body).is_err());
                assert!(crate::rtcm::Message::decode(&body).is_err());

                let (decoded, departures) =
                    SsrVtecMessage::decode_with_policy(&body, RtcmPolicy::Lenient)
                        .expect("lenient VTEC sequence interpretation");
                assert_eq!(departures, vec![departure.clone()]);
                assert_eq!(decoded.layers[0].degree, 1);
                assert_eq!(decoded.layers[0].order, order);
                assert_eq!(decoded.layers[0].cosine, [100, -200, 300]);
                assert_eq!(decoded.layers[0].sine, [400]);
                assert_eq!(
                    vtec_problem(decoded.evaluate(
                        [0.0, 0.0, VTEC_EARTH_RADIUS_M],
                        [0.0, 0.0, 26_000_000.0],
                        50_400.0,
                        1.0e9,
                    )),
                    VtecEvaluationProblem::LayerDegreeOrder {
                        layer_index: 0,
                        degree: 1,
                        order,
                    }
                );

                let (generic, generic_departures) =
                    crate::rtcm::Message::decode_with_policy(&body, RtcmPolicy::Lenient)
                        .expect("generic lenient VTEC decode");
                assert!(matches!(&generic, crate::rtcm::Message::SsrVtec(_)));
                assert_eq!(generic_departures, vec![departure.clone()]);
                let (generic_encoded, generic_encode_departures) = generic
                    .encode_with_policy(RtcmPolicy::Lenient)
                    .expect("generic lenient VTEC encode");
                assert_eq!(generic_encode_departures, vec![departure.clone()]);
                assert_eq!(generic_encoded, body);

                assert!(decoded.encode().is_err());
                let (encoded, encode_departures) = decoded
                    .encode_with_policy(RtcmPolicy::Lenient)
                    .expect("lenient VTEC encode");
                assert_eq!(encode_departures, vec![departure]);
                assert_eq!(encoded, body);
            }
        }
    }

    #[test]
    fn vtec_sagnac_rotation_matches_independent_geometry() {
        let model = constant_model(vec![100, 0, 0]);
        let receiver = [VTEC_EARTH_RADIUS_M, 0.0, 0.0];
        let satellite_tx = [16_370_000.0, 20_000_000.0, 0.0];
        let dx_tx = satellite_tx[0] - receiver[0];
        let dy_tx = satellite_tx[1] - receiver[1];
        let dz_tx = satellite_tx[2] - receiver[2];
        let geometric_range = libm::sqrt(dx_tx * dx_tx + dy_tx * dy_tx + dz_tx * dz_tx);
        let rotation = EARTH_ROTATION_RAD_S * geometric_range / crate::constants::C_M_S;
        let satellite_rx = [
            libm::cos(rotation) * satellite_tx[0] + libm::sin(rotation) * satellite_tx[1],
            -libm::sin(rotation) * satellite_tx[0] + libm::cos(rotation) * satellite_tx[1],
            satellite_tx[2],
        ];
        let dx = satellite_rx[0] - receiver[0];
        let dy = satellite_rx[1] - receiver[1];
        let range_rx = libm::sqrt(dx * dx + dy * dy);
        let elevation = libm::asin((dx / range_rx).clamp(-1.0, 1.0));
        let azimuth = libm::atan2(dy, 0.0);
        let shell_radius = VTEC_EARTH_RADIUS_M + 450_000.0;
        let central = core::f64::consts::FRAC_PI_2
            - elevation
            - libm::asin(VTEC_EARTH_RADIUS_M / shell_radius * libm::cos(elevation));
        let expected_pierce_longitude = central * libm::sin(azimuth);

        let evaluated = model
            .evaluate(receiver, satellite_tx, 50_400.0, 1.0e9)
            .expect("nonzero-Sagnac VTEC geometry");
        assert!(rotation > 1.0e-6);
        assert!(expected_pierce_longitude.abs() > 1.0e-3);
        assert!(
            (evaluated.layers[0].pierce_longitude_rad - expected_pierce_longitude).abs() < 1.0e-10
        );
    }
}
