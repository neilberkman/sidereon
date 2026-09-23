//! SBAS (geostationary) broadcast records.

use crate::constants::KM_TO_M;
use crate::id::GnssSatelliteId;
use crate::validate;

use super::{
    check_extra_lines, parse_record_epoch, raw_record_field, record_fields, stated_field, Decoded,
    Layout, NavEpoch, NavParseError, NavVersion,
};

/// One SBAS broadcast record: an ECEF state at a reference epoch, propagated with a
/// constant acceleration, and a first-order clock, as RTKLIB `seph2pos` evaluates it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbasRecord {
    /// The transmitting satellite (`S20` is PRN 120).
    pub satellite_id: GnssSatelliteId,
    /// Reference epoch `t0` in GPS time, as stated.
    pub epoch: NavEpoch,
    /// Clock bias `aGf0` (seconds).
    pub af0_s: f64,
    /// Relative frequency bias `aGf1` (seconds per second).
    pub af1_s_s: f64,
    /// Transmission time of message, seconds of GPS week, as stated; `None` when blank.
    pub message_frame_time_s: Option<f64>,
    /// ECEF position at `t0` (meters).
    pub pos_m: [f64; 3],
    /// ECEF velocity at `t0` (meters/second).
    pub vel_m_s: [f64; 3],
    /// ECEF acceleration (meters/second²).
    pub acc_m_s2: [f64; 3],
    /// Health (0 is healthy).
    pub health: f64,
    /// Accuracy code (URA, meters), as stated.
    pub ura_m: Option<f64>,
    /// Issue of data navigation (IODN), as stated.
    pub iodn: Option<f64>,
}

impl SbasRecord {
    /// The reference epoch `t0`, seconds since J2000 in GPS time.
    pub fn t0_j2000_s(&self) -> f64 {
        self.epoch.j2000_s()
    }

    /// Position (meters, ECEF) and clock bias (seconds) at `t_j2000_s` (GPS time), as
    /// RTKLIB `seph2pos` evaluates them with `t = time - t0`:
    /// `pos + vel·t + acc·t·t/2` and `af0 + af1·t`.
    pub fn position_clock_at_j2000_s(&self, t_j2000_s: f64) -> ([f64; 3], f64) {
        let t = t_j2000_s - self.t0_j2000_s();
        (self.position_at(t), self.af0_s + self.af1_s_s * t)
    }

    /// Position (meters, ECEF) `t` seconds after `t0`:
    /// `rs[i]=seph->pos[i]+seph->vel[i]*t+seph->acc[i]*t*t/2.0`.
    pub(crate) fn position_at(&self, t: f64) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.pos_m[i] + self.vel_m_s[i] * t + self.acc_m_s2[i] * t * t / 2.0;
        }
        out
    }
}

/// Decode an SBAS record from its four lines: the epoch, `aGf0`, `aGf1` and the
/// transmission time; then x, y and z each with its rate and acceleration (km, km/s,
/// km/s²) and health, accuracy code and IODN.
pub(crate) fn parse_sbas_block(
    lines: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    _version: NavVersion,
    layout: Layout,
) -> Result<Decoded<SbasRecord>, NavParseError> {
    if lines.len() < 4 {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = |what: &'static str| NavParseError::BadField {
        satellite: sat.to_string(),
        field: what,
    };
    let mut departures = Vec::new();
    check_extra_lines(lines, 4, sat, &mut departures);
    let epoch = parse_record_epoch(
        lines[0],
        layout,
        sat,
        "epoch",
        validate::CivilSecondPolicy::Continuous,
    )?;
    let data = record_fields(&lines[..4], layout);
    let g = |index: usize, what: &'static str| data[index].ok_or_else(|| bad(what));
    let km = |index: usize, what: &'static str| g(index, what).map(|x| x * KM_TO_M);
    let raw = |index: usize| raw_record_field(lines, layout, index);
    Ok(Decoded {
        value: SbasRecord {
            satellite_id,
            epoch,
            af0_s: g(0, "af0")?,
            af1_s_s: g(1, "af1")?,
            message_frame_time_s: stated_field(raw(2), "transmission time", sat, &mut departures),
            pos_m: [km(3, "x")?, km(7, "y")?, km(11, "z")?],
            vel_m_s: [km(4, "vx")?, km(8, "vy")?, km(12, "vz")?],
            acc_m_s2: [km(5, "ax")?, km(9, "ay")?, km(13, "az")?],
            health: g(6, "health")?,
            ura_m: stated_field(raw(10), "accuracy code", sat, &mut departures),
            iodn: stated_field(raw(14), "iodn", sat, &mut departures),
        },
        departures,
    })
}
