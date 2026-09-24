//! RTCM 3 network RTK messages: the auxiliary-station data 1014, the GPS and
//! GLONASS correction differences 1015-1017 and 1037-1039, the network RTK
//! residuals 1030 and 1031, the physical reference station position 1032, and
//! the FKP gradients 1034 and 1035 (RTCM 10403.3 Section 3.5.6).
//!
//! Every field is stored as its raw transmitted integer, so a decode followed
//! by an encode reproduces the body bit for bit. Where a header counts the
//! records that follow, the count is kept as transmitted: a body that ends
//! before the records it counts is refused under [`RtcmPolicy::Strict`] and
//! read up to its last complete record under [`RtcmPolicy::Lenient`]
//! ([`RtcmDeparture::RecordsShort`]).

use crate::error::{Error, Result};

use super::bits::{BitReader, FieldWriter, OutOfInput};
use super::{
    decode_body, is_departing_tail, write_trailing, DecodeContext, DecodeError, DecodeResult,
    RtcmDeparture, RtcmEncodeError, RtcmPolicy, RtcmRecordKind,
};

/// Read up to `declared` records with `read`, then what follows them: under
/// the strict policy a short body is refused as truncated and bits after the
/// last record other than the zero byte alignment are refused; under the
/// lenient policy the complete records are read, the bits of a cut record or
/// of a departing tail are returned, and the departure is recorded.
pub(super) fn read_counted<T>(
    r: &mut BitReader<'_>,
    ctx: &mut DecodeContext,
    message_number: u16,
    declared: usize,
    mut read: impl FnMut(&mut BitReader<'_>) -> std::result::Result<T, OutOfInput>,
) -> DecodeResult<(Vec<T>, Vec<bool>)> {
    let mut records = Vec::with_capacity(declared);
    for index in 0..declared {
        let mut trial = r.clone();
        match read(&mut trial) {
            Ok(record) => {
                *r = trial;
                records.push(record);
            }
            Err(error) if ctx.policy() == RtcmPolicy::Strict => {
                return Err(DecodeError::OutOfInput(error));
            }
            Err(_) => {
                ctx.depart(RtcmDeparture::RecordsShort {
                    message_number,
                    declared,
                    read: index,
                })?;
                break;
            }
        }
    }
    let rest = r.rest();
    let tail = if records.len() < declared {
        rest
    } else if is_departing_tail(&rest) {
        ctx.depart(RtcmDeparture::TrailingBits {
            message_number,
            bits: rest.clone(),
        })?;
        rest
    } else {
        Vec::new()
    };
    Ok((records, tail))
}

/// Check a header record count against the records to be written: equal, or
/// under the lenient policy larger (a message read leniently from a short
/// body), which is reported. Returns whether the count is short.
pub(super) fn check_count(
    message_number: u16,
    declared: usize,
    records: usize,
    policy: RtcmPolicy,
    departures: &mut Vec<RtcmDeparture>,
) -> Result<bool> {
    if declared > records && policy == RtcmPolicy::Lenient {
        departures.push(RtcmDeparture::RecordsShort {
            message_number,
            declared,
            read: records,
        });
        return Ok(true);
    }
    if declared != records {
        return Err(RtcmEncodeError::CountMismatch {
            message_number,
            field: "header record",
            expected: declared,
            actual: records,
        }
        .into());
    }
    Ok(false)
}

/// Write the kept bits after the records: the cut record's bits after a short
/// count, which the count's departure reports, or the trailing bits.
pub(super) fn write_tail(
    w: &mut FieldWriter,
    short: bool,
    bits: &[bool],
    policy: RtcmPolicy,
    departures: &mut Vec<RtcmDeparture>,
) -> Result<()> {
    if short {
        for &bit in bits {
            w.flag(bit);
        }
    } else {
        departures.extend(write_trailing(w, bits, policy)?);
    }
    Ok(())
}

/// A decoded message 1014, network auxiliary station data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkAuxiliaryStation {
    /// Network ID (DF059, 8 bits).
    pub network_id: u8,
    /// Subnetwork ID (DF072, 4 bits).
    pub subnetwork_id: u8,
    /// Number of auxiliary stations transmitted (DF058, 5 bits).
    pub auxiliary_station_count: u8,
    /// Master reference station ID (DF060, 12 bits).
    pub master_station_id: u16,
    /// Auxiliary reference station ID (DF061, 12 bits).
    pub auxiliary_station_id: u16,
    /// Auxiliary-master delta latitude (DF062, int20), scale 25e-6 degree.
    pub delta_latitude: i32,
    /// Auxiliary-master delta longitude (DF063, int21), scale 25e-6 degree.
    pub delta_longitude: i32,
    /// Auxiliary-master delta height (DF064, int23), scale 1 mm.
    pub delta_height: i32,
    /// Every body bit after the last field, the zeros that align the body to a
    /// byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros; see [`RtcmDeparture::TrailingBits`]. Empty for every
    /// body read under [`RtcmPolicy::Strict`] and for a message built by hand.
    pub trailing_bits: Vec<bool>,
}

impl NetworkAuxiliaryStation {
    /// Decode a 1014 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if message_number != 1014 {
            return Err(Error::Parse(format!(
                "message {message_number} is not network auxiliary station data 1014"
            ))
            .into());
        }
        Ok(Self {
            network_id: r.u(8)? as u8,
            subnetwork_id: r.u(4)? as u8,
            auxiliary_station_count: r.u(5)? as u8,
            master_station_id: r.u(12)? as u16,
            auxiliary_station_id: r.u(12)? as u16,
            delta_latitude: r.i(20)? as i32,
            delta_longitude: r.i(21)? as i32,
            delta_height: r.i(23)? as i32,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming the field when a value is wider than its
    /// field, or for nonempty `trailing_bits`
    /// ([`RtcmDeparture::TrailingBits`]).
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let mut w = FieldWriter::new(1014);
        w.u("message number", 1014, 12)?;
        w.u("network ID", u64::from(self.network_id), 8)?;
        w.u("subnetwork ID", u64::from(self.subnetwork_id), 4)?;
        w.u(
            "auxiliary station count",
            u64::from(self.auxiliary_station_count),
            5,
        )?;
        w.u("master station ID", u64::from(self.master_station_id), 12)?;
        w.u(
            "auxiliary station ID",
            u64::from(self.auxiliary_station_id),
            12,
        )?;
        w.i("delta latitude", i64::from(self.delta_latitude), 20)?;
        w.i("delta longitude", i64::from(self.delta_longitude), 21)?;
        w.i("delta height", i64::from(self.delta_height), 23)?;
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

/// The parts a correction-difference message number carries.
#[derive(Clone, Copy)]
struct DifferenceLayout {
    glonass: bool,
    geometric: bool,
    ionospheric: bool,
}

impl DifferenceLayout {
    fn of(message_number: u16) -> Option<Self> {
        let (glonass, index) = match message_number {
            1015..=1017 => (false, message_number - 1015),
            1037..=1039 => (true, message_number - 1037),
            _ => return None,
        };
        Some(Self {
            glonass,
            geometric: index >= 1,
            ionospheric: index != 1,
        })
    }
}

/// A decoded network RTK correction-difference message: GPS 1015
/// (ionospheric), 1016 (geometric), 1017 (both), or GLONASS 1037, 1038, 1039.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkCorrectionDifferences {
    /// 1015..=1017 or 1037..=1039.
    pub message_number: u16,
    /// Network ID (DF059, 8 bits).
    pub network_id: u8,
    /// Subnetwork ID (DF072, 4 bits).
    pub subnetwork_id: u8,
    /// Epoch time: GPS time of week (DF065, 23 bits) or GLONASS network epoch
    /// time (DF233, 20 bits), scale 0.1 s.
    pub epoch_time: u32,
    /// Multiple message indicator (DF066).
    pub multiple_message: bool,
    /// Master reference station ID (DF060, 12 bits).
    pub master_station_id: u16,
    /// Auxiliary reference station ID (DF061, 12 bits).
    pub auxiliary_station_id: u16,
    /// Number of satellite records the header states (DF067, DF234, 4 bits),
    /// as transmitted.
    pub satellite_count: u8,
    /// The satellite records, in transmitted order.
    pub satellites: Vec<NetworkCorrectionDifference>,
    /// Every body bit after the last record kept for the round trip: the bits
    /// of a record cut short under [`RtcmPolicy::Lenient`]
    /// ([`RtcmDeparture::RecordsShort`]), or bits after the records other than
    /// the zero byte alignment ([`RtcmDeparture::TrailingBits`]). Empty for
    /// every body read under [`RtcmPolicy::Strict`].
    pub trailing_bits: Vec<bool>,
}

/// One satellite record of a correction-difference message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkCorrectionDifference {
    /// Satellite ID (DF068, DF038, 6 bits).
    pub satellite_id: u8,
    /// Ambiguity status flag (DF074, DF235, 2 bits).
    pub ambiguity_status: u8,
    /// Non-sync count (DF075, DF236, 3 bits).
    pub non_sync_count: u8,
    /// Geometric carrier-phase correction difference (DF070, DF238, int17),
    /// scale 0.5 mm; carried by 1016, 1017, 1038 and 1039.
    pub geometric: Option<i32>,
    /// Issue of data of the ephemeris the geometric difference is formed with
    /// (GPS IODE DF071, GLONASS IOD DF239, 8 bits); carried with the geometric
    /// difference.
    pub iod: Option<u8>,
    /// Ionospheric carrier-phase correction difference (DF069, DF237, int17),
    /// scale 0.5 mm; carried by 1015, 1017, 1037 and 1039.
    pub ionospheric: Option<i32>,
}

impl NetworkCorrectionDifferences {
    /// Decode a 1015..1017 or 1037..1039 body (without the transport frame)
    /// under [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body, &mut DecodeContext::new(RtcmPolicy::Strict)).map_err(Into::into)
    }

    /// Decode a body under `policy`, returning the departures read under
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
        let layout = DifferenceLayout::of(message_number).ok_or_else(|| {
            Error::Parse(format!(
                "message {message_number} is not a correction-difference message \
                 1015-1017/1037-1039"
            ))
        })?;
        let network_id = r.u(8)? as u8;
        let subnetwork_id = r.u(4)? as u8;
        let epoch_time = r.u(if layout.glonass { 20 } else { 23 })? as u32;
        let multiple_message = r.flag()?;
        let master_station_id = r.u(12)? as u16;
        let auxiliary_station_id = r.u(12)? as u16;
        let satellite_count = r.u(4)? as u8;
        let (satellites, trailing_bits) = read_counted(
            &mut r,
            ctx,
            message_number,
            usize::from(satellite_count),
            |r| {
                let satellite_id = r.u(6)? as u8;
                let ambiguity_status = r.u(2)? as u8;
                let non_sync_count = r.u(3)? as u8;
                let (geometric, iod) = if layout.geometric {
                    (Some(r.i(17)? as i32), Some(r.u(8)? as u8))
                } else {
                    (None, None)
                };
                let ionospheric = if layout.ionospheric {
                    Some(r.i(17)? as i32)
                } else {
                    None
                };
                Ok(NetworkCorrectionDifference {
                    satellite_id,
                    ambiguity_status,
                    non_sync_count,
                    geometric,
                    iod,
                    ionospheric,
                })
            },
        )?;
        Ok(Self {
            message_number,
            network_id,
            subnetwork_id,
            epoch_time,
            multiple_message,
            master_station_id,
            auxiliary_station_id,
            satellite_count,
            satellites,
            trailing_bits,
        })
    }

    /// Encode this body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming what the message cannot state: a message
    /// number outside 1015..=1017 and 1037..=1039, a difference or IOD the
    /// number carries held as `None` or one it does not carry held as
    /// `Some`, a header count other than the number of records, nonempty
    /// `trailing_bits`, or a value wider than its field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] a header
    /// count above the number of records and nonempty `trailing_bits` are
    /// written and reported; every other refusal of [`Self::encode`] applies
    /// under both policies.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        let layout = DifferenceLayout::of(number).ok_or(RtcmEncodeError::MessageNumber {
            message_number: number,
            record: RtcmRecordKind::Network {
                family: "network correction-difference message 1015-1017/1037-1039",
            },
        })?;
        for s in &self.satellites {
            for (what, present, carried) in [
                (
                    "geometric difference",
                    s.geometric.is_some(),
                    layout.geometric,
                ),
                ("IOD", s.iod.is_some(), layout.geometric),
                (
                    "ionospheric difference",
                    s.ionospheric.is_some(),
                    layout.ionospheric,
                ),
            ] {
                if present != carried {
                    return Err(RtcmEncodeError::SatelliteFieldPresence {
                        message_number: number,
                        record: RtcmRecordKind::Network {
                            family: "network correction-difference message",
                        },
                        satellite: s.satellite_id,
                        field: what,
                        carried,
                    }
                    .into());
                }
            }
        }
        let mut departures = Vec::new();
        let short = check_count(
            number,
            usize::from(self.satellite_count),
            self.satellites.len(),
            policy,
            &mut departures,
        )?;
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u("network ID", u64::from(self.network_id), 8)?;
        w.u("subnetwork ID", u64::from(self.subnetwork_id), 4)?;
        w.u(
            "epoch time",
            u64::from(self.epoch_time),
            if layout.glonass { 20 } else { 23 },
        )?;
        w.flag(self.multiple_message);
        w.u("master station ID", u64::from(self.master_station_id), 12)?;
        w.u(
            "auxiliary station ID",
            u64::from(self.auxiliary_station_id),
            12,
        )?;
        w.u("satellite count", u64::from(self.satellite_count), 4)?;
        for s in &self.satellites {
            let id = s.satellite_id;
            w.u("satellite ID", u64::from(id), 6)?;
            w.u(
                format_args!("satellite {id} ambiguity status"),
                u64::from(s.ambiguity_status),
                2,
            )?;
            w.u(
                format_args!("satellite {id} non-sync count"),
                u64::from(s.non_sync_count),
                3,
            )?;
            if let (Some(geometric), Some(iod)) = (s.geometric, s.iod) {
                w.i(
                    format_args!("satellite {id} geometric difference"),
                    i64::from(geometric),
                    17,
                )?;
                w.u(format_args!("satellite {id} IOD"), u64::from(iod), 8)?;
            }
            if let Some(ionospheric) = s.ionospheric {
                w.i(
                    format_args!("satellite {id} ionospheric difference"),
                    i64::from(ionospheric),
                    17,
                )?;
            }
        }
        write_tail(&mut w, short, &self.trailing_bits, policy, &mut departures)?;
        Ok((w.into_bytes(), departures))
    }
}

/// A decoded network RTK residual message: GPS 1030 or GLONASS 1031.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetworkResiduals {
    /// 1030 or 1031.
    pub message_number: u16,
    /// Residuals epoch time: GPS time of week (DF224, 20 bits) or GLONASS
    /// `tk` (DF225, 17 bits), scale 1 s.
    pub epoch_time: u32,
    /// Reference station ID (DF003, 12 bits).
    pub reference_station_id: u16,
    /// Number of reference stations used to derive the residual statistics
    /// (DF223, 7 bits).
    pub reference_station_count: u8,
    /// Number of satellite records the header states (DF006, DF035, 5 bits),
    /// as transmitted.
    pub satellite_count: u8,
    /// The satellite records, in transmitted order.
    pub satellites: Vec<NetworkResidual>,
    /// The kept bits after the last record; see
    /// [`NetworkCorrectionDifferences::trailing_bits`].
    pub trailing_bits: Vec<bool>,
}

/// One satellite record of a network RTK residual message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkResidual {
    /// Satellite ID (DF009, DF038, 6 bits).
    pub satellite_id: u8,
    /// Constant term of the non-dispersive standard deviation `s_oc` (DF218,
    /// 8 bits), scale 0.5 mm.
    pub s_oc: u8,
    /// Distance-dependent term of the non-dispersive standard deviation `s_od`
    /// (DF219, 9 bits), scale 0.01 ppm.
    pub s_od: u16,
    /// Height-dependent term of the non-dispersive standard deviation `s_oh`
    /// (DF220, 6 bits), scale 0.1 ppm.
    pub s_oh: u8,
    /// Constant term of the dispersive standard deviation `s_lc` (DF221,
    /// 10 bits), scale 0.5 mm.
    pub s_lc: u16,
    /// Distance-dependent term of the dispersive standard deviation `s_ld`
    /// (DF222, 10 bits), scale 0.01 ppm.
    pub s_ld: u16,
}

impl NetworkResiduals {
    /// Decode a 1030 or 1031 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body, &mut DecodeContext::new(RtcmPolicy::Strict)).map_err(Into::into)
    }

    /// Decode a body under `policy`, returning the departures read under
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
        let epoch_bits = match message_number {
            1030 => 20,
            1031 => 17,
            _ => {
                return Err(Error::Parse(format!(
                    "message {message_number} is not a network RTK residual message 1030/1031"
                ))
                .into());
            }
        };
        let epoch_time = r.u(epoch_bits)? as u32;
        let reference_station_id = r.u(12)? as u16;
        let reference_station_count = r.u(7)? as u8;
        let satellite_count = r.u(5)? as u8;
        let (satellites, trailing_bits) = read_counted(
            &mut r,
            ctx,
            message_number,
            usize::from(satellite_count),
            |r| {
                Ok(NetworkResidual {
                    satellite_id: r.u(6)? as u8,
                    s_oc: r.u(8)? as u8,
                    s_od: r.u(9)? as u16,
                    s_oh: r.u(6)? as u8,
                    s_lc: r.u(10)? as u16,
                    s_ld: r.u(10)? as u16,
                })
            },
        )?;
        Ok(Self {
            message_number,
            epoch_time,
            reference_station_id,
            reference_station_count,
            satellite_count,
            satellites,
            trailing_bits,
        })
    }

    /// Encode this body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming what the message cannot state: a message
    /// number other than 1030 and 1031, a header count other than the number
    /// of records, nonempty `trailing_bits`, or a value wider than its field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`; see
    /// [`NetworkCorrectionDifferences::encode_with_policy`].
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        let epoch_bits = match number {
            1030 => 20,
            1031 => 17,
            _ => {
                return Err(RtcmEncodeError::MessageNumber {
                    message_number: number,
                    record: RtcmRecordKind::Network {
                        family: "network RTK residual message 1030/1031",
                    },
                }
                .into());
            }
        };
        let mut departures = Vec::new();
        let short = check_count(
            number,
            usize::from(self.satellite_count),
            self.satellites.len(),
            policy,
            &mut departures,
        )?;
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u("epoch time", u64::from(self.epoch_time), epoch_bits)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.u(
            "reference station count",
            u64::from(self.reference_station_count),
            7,
        )?;
        w.u("satellite count", u64::from(self.satellite_count), 5)?;
        for s in &self.satellites {
            let id = s.satellite_id;
            w.u("satellite ID", u64::from(id), 6)?;
            w.u(format_args!("satellite {id} s_oc"), u64::from(s.s_oc), 8)?;
            w.u(format_args!("satellite {id} s_od"), u64::from(s.s_od), 9)?;
            w.u(format_args!("satellite {id} s_oh"), u64::from(s.s_oh), 6)?;
            w.u(format_args!("satellite {id} s_lc"), u64::from(s.s_lc), 10)?;
            w.u(format_args!("satellite {id} s_ld"), u64::from(s.s_ld), 10)?;
        }
        write_tail(&mut w, short, &self.trailing_bits, policy, &mut departures)?;
        Ok((w.into_bytes(), departures))
    }
}

/// A decoded message 1032, the physical reference station position of a
/// non-physical (virtual) reference station.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalReferenceStation {
    /// Non-physical reference station ID (DF003, 12 bits).
    pub non_physical_station_id: u16,
    /// Physical reference station ID (DF226, 12 bits).
    pub physical_station_id: u16,
    /// ITRF realization year (DF021, 6 bits).
    pub itrf_realization_year: u8,
    /// Physical reference station ARP ECEF X (DF025, int38), scale 0.0001 m.
    pub ecef_x: i64,
    /// Physical reference station ARP ECEF Y (DF026, int38), scale 0.0001 m.
    pub ecef_y: i64,
    /// Physical reference station ARP ECEF Z (DF027, int38), scale 0.0001 m.
    pub ecef_z: i64,
    /// Every body bit after the last field kept for the round trip; see
    /// [`NetworkAuxiliaryStation::trailing_bits`].
    pub trailing_bits: Vec<bool>,
}

impl PhysicalReferenceStation {
    /// Decode a 1032 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if message_number != 1032 {
            return Err(Error::Parse(format!(
                "message {message_number} is not a physical reference station position 1032"
            ))
            .into());
        }
        Ok(Self {
            non_physical_station_id: r.u(12)? as u16,
            physical_station_id: r.u(12)? as u16,
            itrf_realization_year: r.u(6)? as u8,
            ecef_x: r.i(38)?,
            ecef_y: r.i(38)?,
            ecef_z: r.i(38)?,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming the field when a value is wider than its
    /// field, or for nonempty `trailing_bits`.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let mut w = FieldWriter::new(1032);
        w.u("message number", 1032, 12)?;
        w.u(
            "non-physical station ID",
            u64::from(self.non_physical_station_id),
            12,
        )?;
        w.u(
            "physical station ID",
            u64::from(self.physical_station_id),
            12,
        )?;
        w.u(
            "ITRF realization year",
            u64::from(self.itrf_realization_year),
            6,
        )?;
        w.i("ECEF X", self.ecef_x, 38)?;
        w.i("ECEF Y", self.ecef_y, 38)?;
        w.i("ECEF Z", self.ecef_z, 38)?;
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

/// A decoded FKP gradient message: GPS 1034 or GLONASS 1035.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FkpGradients {
    /// 1034 or 1035.
    pub message_number: u16,
    /// Reference station ID (DF003, 12 bits).
    pub reference_station_id: u16,
    /// FKP epoch time: GPS time of week (DF240, 20 bits) or GLONASS time
    /// (DF241, 17 bits), scale 1 s.
    pub epoch_time: u32,
    /// Number of satellite records the header states (DF006, DF035, 5 bits),
    /// as transmitted.
    pub satellite_count: u8,
    /// The satellite records, in transmitted order.
    pub satellites: Vec<FkpGradient>,
    /// The kept bits after the last record; see
    /// [`NetworkCorrectionDifferences::trailing_bits`].
    pub trailing_bits: Vec<bool>,
}

/// One satellite record of an FKP gradient message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FkpGradient {
    /// Satellite ID (DF009, DF038, 6 bits).
    pub satellite_id: u8,
    /// Issue of data (GPS IODE DF071, GLONASS IOD DF392, 8 bits).
    pub iod: u8,
    /// Geometric gradient north `N0` (DF242, int12), scale 0.01 ppm.
    pub geometric_north: i16,
    /// Geometric gradient east `E0` (DF243, int12), scale 0.01 ppm.
    pub geometric_east: i16,
    /// Ionospheric gradient north `NI` (DF244, int14), scale 0.01 ppm.
    pub ionospheric_north: i16,
    /// Ionospheric gradient east `EI` (DF245, int14), scale 0.01 ppm.
    pub ionospheric_east: i16,
}

impl FkpGradients {
    /// Decode a 1034 or 1035 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body, &mut DecodeContext::new(RtcmPolicy::Strict)).map_err(Into::into)
    }

    /// Decode a body under `policy`, returning the departures read under
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
        let epoch_bits = match message_number {
            1034 => 20,
            1035 => 17,
            _ => {
                return Err(Error::Parse(format!(
                    "message {message_number} is not an FKP gradient message 1034/1035"
                ))
                .into());
            }
        };
        let reference_station_id = r.u(12)? as u16;
        let epoch_time = r.u(epoch_bits)? as u32;
        let satellite_count = r.u(5)? as u8;
        let (satellites, trailing_bits) = read_counted(
            &mut r,
            ctx,
            message_number,
            usize::from(satellite_count),
            |r| {
                Ok(FkpGradient {
                    satellite_id: r.u(6)? as u8,
                    iod: r.u(8)? as u8,
                    geometric_north: r.i(12)? as i16,
                    geometric_east: r.i(12)? as i16,
                    ionospheric_north: r.i(14)? as i16,
                    ionospheric_east: r.i(14)? as i16,
                })
            },
        )?;
        Ok(Self {
            message_number,
            reference_station_id,
            epoch_time,
            satellite_count,
            satellites,
            trailing_bits,
        })
    }

    /// Encode this body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    ///
    /// # Errors
    ///
    /// [`Error::RtcmEncode`] naming what the message cannot state: a message
    /// number other than 1034 and 1035, a header count other than the number
    /// of records, nonempty `trailing_bits`, or a value wider than its field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`; see
    /// [`NetworkCorrectionDifferences::encode_with_policy`].
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        let epoch_bits = match number {
            1034 => 20,
            1035 => 17,
            _ => {
                return Err(RtcmEncodeError::MessageNumber {
                    message_number: number,
                    record: RtcmRecordKind::Network {
                        family: "FKP gradient message 1034/1035",
                    },
                }
                .into());
            }
        };
        let mut departures = Vec::new();
        let short = check_count(
            number,
            usize::from(self.satellite_count),
            self.satellites.len(),
            policy,
            &mut departures,
        )?;
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u(
            "reference station ID",
            u64::from(self.reference_station_id),
            12,
        )?;
        w.u("epoch time", u64::from(self.epoch_time), epoch_bits)?;
        w.u("satellite count", u64::from(self.satellite_count), 5)?;
        for s in &self.satellites {
            let id = s.satellite_id;
            w.u("satellite ID", u64::from(id), 6)?;
            w.u(format_args!("satellite {id} IOD"), u64::from(s.iod), 8)?;
            w.i(
                format_args!("satellite {id} geometric gradient north"),
                i64::from(s.geometric_north),
                12,
            )?;
            w.i(
                format_args!("satellite {id} geometric gradient east"),
                i64::from(s.geometric_east),
                12,
            )?;
            w.i(
                format_args!("satellite {id} ionospheric gradient north"),
                i64::from(s.ionospheric_north),
                14,
            )?;
            w.i(
                format_args!("satellite {id} ionospheric gradient east"),
                i64::from(s.ionospheric_east),
                14,
            )?;
        }
        write_tail(&mut w, short, &self.trailing_bits, policy, &mut departures)?;
        Ok((w.into_bytes(), departures))
    }
}

impl super::TrailingBits for NetworkAuxiliaryStation {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}

impl super::TrailingBits for PhysicalReferenceStation {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
