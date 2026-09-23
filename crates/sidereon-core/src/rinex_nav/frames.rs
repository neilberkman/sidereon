//! RINEX 4 system time offset (`> STO`), Earth orientation (`> EOP`) and ionosphere
//! (`> ION`) frames.

use crate::format::columns::raw_field;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::ionex::GalileoNequickCoeffs;
use crate::validate;

use super::{
    check_extra_lines, parse_f64, parse_record_epoch, record_fields, Decoded, IonoCorrections,
    KlobucharAlphaBeta, Layout, NavEpoch, NavParseError,
};

/// A RINEX 4 system time offset frame (`> STO SV MSG`).
#[derive(Debug, Clone, PartialEq)]
pub struct SystemTimeOffset {
    /// The satellite that transmitted the parameters.
    pub satellite_id: GnssSatelliteId,
    /// The message token of the frame marker.
    pub message_token: String,
    /// Reference epoch of the offset polynomial, as stated.
    pub reference_epoch: NavEpoch,
    /// Time offset type (`GPUT`, `GAUT`, `GAGP`, ...), trimmed.
    pub offset_code: String,
    /// SBAS provider identifier, when stated.
    pub sbas_id: Option<String>,
    /// UTC identifier (`UTC(USNO)`, `UTCGAL`, ...), when stated.
    pub utc_id: Option<String>,
    /// Transmission time of message, seconds of week.
    pub transmission_time_sow: f64,
    /// `a0`, seconds.
    pub a0_s: f64,
    /// `a1`, seconds per second; `None` when blank.
    pub a1_s_s: Option<f64>,
    /// `a2`, seconds per second squared; `None` when blank.
    pub a2_s_s2: Option<f64>,
}

/// A RINEX 4 Earth orientation frame (`> EOP SV MSG`).
#[derive(Debug, Clone, PartialEq)]
pub struct EarthOrientation {
    /// The satellite that transmitted the parameters.
    pub satellite_id: GnssSatelliteId,
    /// The message token of the frame marker.
    pub message_token: String,
    /// Reference epoch of the parameters, as stated.
    pub reference_epoch: NavEpoch,
    /// Pole x (arcseconds), its rate (arcseconds/day) and its rate of change
    /// (arcseconds/day²); a blank rate term is `None`.
    pub xp: [Option<f64>; 3],
    /// Pole y (arcseconds), its rate (arcseconds/day) and its rate of change
    /// (arcseconds/day²); a blank rate term is `None`.
    pub yp: [Option<f64>; 3],
    /// Transmission time of message, seconds of week.
    pub transmission_time_sow: f64,
    /// UT1 - UTC (seconds), its rate (seconds/day) and its rate of change
    /// (seconds/day²); a blank rate term is `None`.
    pub dut1: [Option<f64>; 3],
}

/// The model of a RINEX 4 ionosphere frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IonosphereModel {
    /// Klobuchar coefficients: GPS and QZSS `LNAV` and `CNVX`, BeiDou `D1D2`, NavIC
    /// `LNAV`. The ninth value is the region code QZSS states, as read.
    Klobuchar {
        /// The alpha and beta coefficients.
        coefficients: KlobucharAlphaBeta,
        /// The ninth value, as stated.
        region_code: Option<f64>,
    },
    /// Galileo NeQuick-G coefficients (`IFNV`) and disturbance flags.
    NequickG {
        /// `ai0`, `ai1`, `ai2`.
        coefficients: GalileoNequickCoeffs,
        /// The disturbance flags, as stated.
        disturbance_flags: Option<f64>,
    },
    /// BeiDou global ionospheric model coefficients `alpha1..alpha9` (`CNVX`).
    Bdgim([f64; 9]),
}

/// A RINEX 4 ionosphere frame (`> ION SV MSG`).
#[derive(Debug, Clone, PartialEq)]
pub struct IonosphereFrame {
    /// The satellite that transmitted the parameters.
    pub satellite_id: GnssSatelliteId,
    /// The message token of the frame marker.
    pub message_token: String,
    /// Transmission epoch, as stated, in the system's time scale.
    pub transmission_epoch: NavEpoch,
    /// The frame's values in order, `None` for a blank field; trailing blanks are not
    /// listed.
    pub values: Vec<Option<f64>>,
    /// The model the values form.
    pub model: IonosphereModel,
}

impl IonosphereFrame {
    /// The transmission epoch in GPS time, seconds since J2000 (BDT is GPS time less 14
    /// s; Galileo, QZSS and NavIC time run on GPS time).
    pub fn transmission_gpst_j2000_s(&self) -> f64 {
        let offset = if self.satellite_id.system == GnssSystem::BeiDou {
            crate::constants::GPST_MINUS_BDT_S
        } else {
            0.0
        };
        self.transmission_epoch.j2000_s() + offset
    }

    /// Put this frame's coefficients in the set of its system.
    pub(crate) fn apply_to(&self, iono: &mut IonoCorrections) {
        match self.model {
            IonosphereModel::Klobuchar { coefficients, .. } => match self.satellite_id.system {
                GnssSystem::Gps => iono.gps = Some(coefficients),
                GnssSystem::Qzss => iono.qzss = Some(coefficients),
                GnssSystem::BeiDou => iono.beidou = Some(coefficients),
                GnssSystem::Navic => iono.navic = Some(coefficients),
                _ => {}
            },
            IonosphereModel::NequickG {
                coefficients,
                disturbance_flags,
            } => {
                iono.galileo = Some(coefficients);
                iono.galileo_disturbance_flags = disturbance_flags;
            }
            IonosphereModel::Bdgim(coefficients) => iono.beidou_bdgim = Some(coefficients),
        }
    }
}

/// Put `frames` in transmission order, frames transmitted at the same instant in file
/// order, the order [`ionosphere_in_effect`] reads them in.
pub(crate) fn sort_by_transmission(frames: &mut [IonosphereFrame]) {
    frames.sort_by(|a, b| {
        a.transmission_gpst_j2000_s()
            .total_cmp(&b.transmission_gpst_j2000_s())
    });
}

/// The broadcast ionosphere coefficients in effect at `t_gpst_j2000_s` (GPS time,
/// seconds past J2000): the header's sets, each replaced by the `> ION` frame of its
/// system and model transmitted latest at or before the epoch. With no epoch, every
/// frame counts, so each set is the latest transmitted. `frames` are in the order
/// [`sort_by_transmission`] puts them in. A set no header row and no frame transmitted
/// by the epoch states is `None`.
pub(crate) fn ionosphere_in_effect(
    header: IonoCorrections,
    frames: &[IonosphereFrame],
    t_gpst_j2000_s: Option<f64>,
) -> IonoCorrections {
    let mut iono = header;
    for frame in frames
        .iter()
        .take_while(|frame| t_gpst_j2000_s.is_none_or(|t| frame.transmission_gpst_j2000_s() <= t))
    {
        frame.apply_to(&mut iono);
    }
    iono
}

fn optional_text(line: &str, start: usize, end: usize) -> Option<String> {
    let value = raw_field(line, start, end).trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// The numeric fields of a frame body, by position, with a non-blank field that does
/// not read refused.
fn frame_values(
    body: &[&str],
    sat: &str,
    field: &'static str,
) -> Result<Vec<Option<f64>>, NavParseError> {
    let values = record_fields(body, Layout::V3);
    for (index, value) in values.iter().enumerate() {
        if value.is_none()
            && !super::raw_record_field(body, Layout::V3, index)
                .trim()
                .is_empty()
        {
            return Err(NavParseError::BadField {
                satellite: sat.to_string(),
                field,
            });
        }
    }
    Ok(values)
}

/// `> STO`: the reference epoch, offset type, SBAS and UTC identifiers, then the
/// transmission time and `a0`, `a1`, `a2`.
pub(crate) fn parse_sto_frame(
    body: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    message_token: &str,
) -> Result<Decoded<SystemTimeOffset>, NavParseError> {
    const FIELD: &str = "system time offset";
    if body.len() < 2 {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = || NavParseError::BadField {
        satellite: sat.to_string(),
        field: FIELD,
    };
    let mut departures = Vec::new();
    check_extra_lines(body, 2, sat, &mut departures);
    let reference_epoch = parse_record_epoch(
        body[0],
        Layout::V3,
        sat,
        FIELD,
        validate::CivilSecondPolicy::Continuous,
    )?;
    let offset_code = optional_text(body[0], 24, 42).ok_or_else(bad)?;
    let line = body[1];
    let optional = |start: usize, end: usize| -> Result<Option<f64>, NavParseError> {
        if raw_field(line, start, end).trim().is_empty() {
            return Ok(None);
        }
        parse_f64(line, start, end).map(Some).ok_or_else(bad)
    };
    Ok(Decoded {
        value: SystemTimeOffset {
            satellite_id,
            message_token: message_token.to_string(),
            reference_epoch,
            offset_code,
            sbas_id: optional_text(body[0], 43, 61),
            utc_id: optional_text(body[0], 62, 80),
            transmission_time_sow: optional(4, 23)?.ok_or_else(bad)?,
            a0_s: optional(23, 42)?.ok_or_else(bad)?,
            a1_s_s: optional(42, 61)?,
            a2_s_s2: optional(61, 80)?,
        },
        departures,
    })
}

/// `> EOP`: the reference epoch with pole x and its rates, pole y and its rates, then
/// the transmission time with UT1 - UTC and its rates.
pub(crate) fn parse_eop_frame(
    body: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    message_token: &str,
) -> Result<Decoded<EarthOrientation>, NavParseError> {
    const FIELD: &str = "earth orientation";
    if body.len() < 3 {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = || NavParseError::BadField {
        satellite: sat.to_string(),
        field: FIELD,
    };
    let mut departures = Vec::new();
    check_extra_lines(body, 3, sat, &mut departures);
    let reference_epoch = parse_record_epoch(
        body[0],
        Layout::V3,
        sat,
        FIELD,
        validate::CivilSecondPolicy::Continuous,
    )?;
    let values = frame_values(&body[..3], sat, FIELD)?;
    let get = |index: usize| values.get(index).copied().flatten();
    // data[] order: line 1 fields 0-2 (x), line 2 fields 3-6 (blank, y), line 3 fields
    // 7-10 (t_tm, dUT1).
    if get(0).is_none() || get(4).is_none() || get(8).is_none() {
        return Err(bad());
    }
    Ok(Decoded {
        value: EarthOrientation {
            satellite_id,
            message_token: message_token.to_string(),
            reference_epoch,
            xp: [get(0), get(1), get(2)],
            yp: [get(4), get(5), get(6)],
            transmission_time_sow: get(7).ok_or_else(bad)?,
            dut1: [get(8), get(9), get(10)],
        },
        departures,
    })
}

/// `> ION`: the transmission epoch and the model's values. `Ok(None)` for a system
/// and message this crate has no model for, which is kept as text only.
pub(crate) fn parse_ion_frame(
    body: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    message_token: &str,
) -> Result<Option<Decoded<IonosphereFrame>>, NavParseError> {
    const FIELD: &str = "ionospheric correction";
    enum Kind {
        Klobuchar,
        Nequick,
        Bdgim,
    }
    let kind = match (satellite_id.system, message_token) {
        (GnssSystem::Gps | GnssSystem::Qzss, "LNAV" | "CNVX")
        | (GnssSystem::BeiDou, "D1D2")
        | (GnssSystem::Navic, "LNAV") => Kind::Klobuchar,
        (GnssSystem::Galileo, "IFNV") => Kind::Nequick,
        (GnssSystem::BeiDou, "CNVX") => Kind::Bdgim,
        _ => return Ok(None),
    };
    // Lines the model's values fill: three on the first line, four on each next one.
    let (value_count, lines) = match kind {
        Kind::Klobuchar => (9, 3),
        Kind::Nequick => (4, 2),
        Kind::Bdgim => (9, 3),
    };
    if body.is_empty() {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = || NavParseError::BadField {
        satellite: sat.to_string(),
        field: FIELD,
    };
    let mut departures = Vec::new();
    check_extra_lines(body, lines, sat, &mut departures);
    let transmission_epoch = parse_record_epoch(
        body[0],
        Layout::V3,
        sat,
        FIELD,
        validate::CivilSecondPolicy::Continuous,
    )?;
    let used = &body[..lines.min(body.len())];
    let mut values = frame_values(used, sat, FIELD)?;
    values.truncate(value_count);
    while values.last().is_some_and(Option::is_none) {
        values.pop();
    }
    let need = |index: usize| values.get(index).copied().flatten().ok_or_else(bad);
    let model = match kind {
        Kind::Klobuchar => IonosphereModel::Klobuchar {
            coefficients: KlobucharAlphaBeta {
                alpha: [need(0)?, need(1)?, need(2)?, need(3)?],
                beta: [need(4)?, need(5)?, need(6)?, need(7)?],
            },
            region_code: values.get(8).copied().flatten(),
        },
        Kind::Nequick => IonosphereModel::NequickG {
            coefficients: GalileoNequickCoeffs {
                ai0: need(0)?,
                ai1: need(1)?,
                ai2: need(2)?,
            },
            disturbance_flags: values.get(3).copied().flatten(),
        },
        Kind::Bdgim => {
            let mut alpha = [0.0; 9];
            for (index, slot) in alpha.iter_mut().enumerate() {
                *slot = need(index)?;
            }
            IonosphereModel::Bdgim(alpha)
        }
    };
    Ok(Some(Decoded {
        value: IonosphereFrame {
            satellite_id,
            message_token: message_token.to_string(),
            transmission_epoch,
            values,
            model,
        },
        departures,
    }))
}
