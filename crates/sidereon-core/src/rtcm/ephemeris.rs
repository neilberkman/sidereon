//! RTCM 3 broadcast ephemeris messages 1019 (GPS), 1020 (GLONASS),
//! 1042 (BeiDou), 1044 (QZSS), and 1045/1046 (Galileo).
//!
//! Message 1019 (RTCM 10403.3 Table 3.5-21) carries one complete set of GPS
//! LNAV ephemeris and clock parameters; message 1020 (Table 3.5-22) carries one
//! GLONASS satellite's ephemeris. The GLONASS message stores its orbit terms in
//! sign-and-magnitude form (the leading bit is the sign), which the bit reader's
//! [`super::bits::BitReader::ism`] handles.
//!
//! Every field is stored as its raw transmitted integer (the `DFxxx` quantity),
//! preserving the integer-vs-sign-magnitude distinction exactly, so the body
//! round-trips byte-for-byte. The standard per-field scale factors are noted in
//! the struct docs; applying them yields the engineering-unit ephemeris that
//! [`crate::broadcast`] consumes.

use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::broadcast::{ClockPolynomial, KeplerianElements};
use crate::constants::SECONDS_PER_HOUR;
use crate::error::{Error, Result};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::rinex_nav::{
    gps_fit_interval_from_flag, gps_ura_index_to_meters, BroadcastGroupDelays, BroadcastIssue,
    BroadcastRecord, NavMessage,
};

use super::bits::{BitReader, BitWriter};
use super::DecodeResult;

const SEMICIRCLE_TO_RAD: f64 = core::f64::consts::PI;
const GALILEO_WEEK_OFFSET_TO_GPS: u32 = 1024;

fn scaled_i(value: impl Into<i64>, exponent: i32) -> f64 {
    (value.into() as f64) * 2.0_f64.powi(exponent)
}

fn scaled_u(value: u64, exponent: i32) -> f64 {
    (value as f64) * 2.0_f64.powi(exponent)
}

fn scaled_semicircle(value: impl Into<i64>, exponent: i32) -> f64 {
    scaled_i(value, exponent) * SEMICIRCLE_TO_RAD
}

fn gnss_week_tow(
    system: TimeScale,
    week: u32,
    tow_s: f64,
    field: &'static str,
) -> Result<GnssWeekTow> {
    GnssWeekTow::new(system, week, tow_s)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| Error::InvalidInput(format!("RTCM broadcast {field} is not representable")))
}

fn galileo_sisa_m(index: u8) -> f64 {
    match index {
        0..=49 => f64::from(index) * 0.01,
        50..=74 => 0.50 + f64::from(index - 50) * 0.02,
        75..=99 => 1.00 + f64::from(index - 75) * 0.04,
        100..=125 => 2.00 + f64::from(index - 100) * 0.16,
        _ => 8192.0,
    }
}

fn ura_or_wide(index: u8) -> f64 {
    gps_ura_index_to_meters(i64::from(index)).unwrap_or(8192.0)
}

fn raw_health(healthy: bool) -> f64 {
    if healthy {
        0.0
    } else {
        1.0
    }
}

/// A decoded GPS broadcast ephemeris (message 1019).
///
/// Angular quantities are in semicircles (scale noted per field), harmonic
/// correction terms in radians, distances in meters, and clock terms in
/// seconds, each recovered by multiplying the raw integer by its scale factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpsEphemeris {
    /// GPS satellite PRN (DF009).
    pub satellite_id: u8,
    /// GPS week number (DF076, 10 bits).
    pub week_number: u16,
    /// SV accuracy / URA index (DF077, 4 bits).
    pub sv_accuracy: u8,
    /// Code on L2 (DF078, 2 bits).
    pub code_on_l2: u8,
    /// Rate of inclination angle IDOT (DF079, int14, scale 2^-43 semicircles/s).
    pub idot: i32,
    /// Issue of data, ephemeris (DF071, 8 bits).
    pub iode: u8,
    /// Clock data reference time t_oc (DF081, uint16, scale 2^4 s).
    pub t_oc: u16,
    /// Clock drift rate a_f2 (DF082, int8, scale 2^-55 s/s^2).
    pub a_f2: i16,
    /// Clock drift a_f1 (DF083, int16, scale 2^-43 s/s).
    pub a_f1: i32,
    /// Clock bias a_f0 (DF084, int22, scale 2^-31 s).
    pub a_f0: i32,
    /// Issue of data, clock (DF085, 10 bits).
    pub iodc: u16,
    /// Orbit-radius sine correction C_rs (DF086, int16, scale 2^-5 m).
    pub c_rs: i32,
    /// Mean-motion difference dn (DF087, int16, scale 2^-43 semicircles/s).
    pub delta_n: i32,
    /// Mean anomaly at reference time M_0 (DF088, int32, scale 2^-31 semicircles).
    pub m0: i64,
    /// Latitude-argument cosine correction C_uc (DF089, int16, scale 2^-29 rad).
    pub c_uc: i32,
    /// Eccentricity e (DF090, uint32, scale 2^-33).
    pub eccentricity: u64,
    /// Latitude-argument sine correction C_us (DF091, int16, scale 2^-29 rad).
    pub c_us: i32,
    /// Square root of the semi-major axis sqrt(A) (DF092, uint32, scale 2^-19).
    pub sqrt_a: u64,
    /// Ephemeris reference time t_oe (DF093, uint16, scale 2^4 s).
    pub t_oe: u16,
    /// Inclination cosine correction C_ic (DF094, int16, scale 2^-29 rad).
    pub c_ic: i32,
    /// Longitude of ascending node Omega_0 (DF095, int32, scale 2^-31 semicircles).
    pub omega0: i64,
    /// Inclination sine correction C_is (DF096, int16, scale 2^-29 rad).
    pub c_is: i32,
    /// Inclination at reference time i_0 (DF097, int32, scale 2^-31 semicircles).
    pub i0: i64,
    /// Orbit-radius cosine correction C_rc (DF098, int16, scale 2^-5 m).
    pub c_rc: i32,
    /// Argument of perigee omega (DF099, int32, scale 2^-31 semicircles).
    pub omega: i64,
    /// Rate of right ascension Omega-dot (DF100, int24, scale 2^-43 semicircles/s).
    pub omega_dot: i32,
    /// Group delay differential t_GD (DF101, int8, scale 2^-31 s).
    pub t_gd: i16,
    /// SV health (DF102, 6 bits).
    pub sv_health: u8,
    /// L2 P-data flag (DF103).
    pub l2_p_data_flag: bool,
    /// Fit-interval flag (DF137).
    pub fit_interval: bool,
}

impl GpsEphemeris {
    /// The satellite identifier for this ephemeris.
    pub fn satellite(&self) -> Result<GnssSatelliteId> {
        GnssSatelliteId::new(GnssSystem::Gps, self.satellite_id)
            .map_err(|e| Error::Parse(format!("invalid GPS PRN in 1019: {e}")))
    }

    /// Decode a message 1019 body (without the transport frame).
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1019 {
            return Err(Error::Parse(format!(
                "message {message_number} is not GPS ephemeris 1019"
            ))
            .into());
        }
        Ok(Self {
            satellite_id: r.u(6)? as u8,
            week_number: r.u(10)? as u16,
            sv_accuracy: r.u(4)? as u8,
            code_on_l2: r.u(2)? as u8,
            idot: r.i(14)? as i32,
            iode: r.u(8)? as u8,
            t_oc: r.u(16)? as u16,
            a_f2: r.i(8)? as i16,
            a_f1: r.i(16)? as i32,
            a_f0: r.i(22)? as i32,
            iodc: r.u(10)? as u16,
            c_rs: r.i(16)? as i32,
            delta_n: r.i(16)? as i32,
            m0: r.i(32)?,
            c_uc: r.i(16)? as i32,
            eccentricity: r.u(32)?,
            c_us: r.i(16)? as i32,
            sqrt_a: r.u(32)?,
            t_oe: r.u(16)? as u16,
            c_ic: r.i(16)? as i32,
            omega0: r.i(32)?,
            c_is: r.i(16)? as i32,
            i0: r.i(32)?,
            c_rc: r.i(16)? as i32,
            omega: r.i(32)?,
            omega_dot: r.i(24)? as i32,
            t_gd: r.i(8)? as i16,
            sv_health: r.u(6)? as u8,
            l2_p_data_flag: r.flag()?,
            fit_interval: r.flag()?,
        })
    }

    /// Encode this GPS ephemeris body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(1019, 12);
        w.push_u(u64::from(self.satellite_id), 6);
        w.push_u(u64::from(self.week_number), 10);
        w.push_u(u64::from(self.sv_accuracy), 4);
        w.push_u(u64::from(self.code_on_l2), 2);
        w.push_i(i64::from(self.idot), 14);
        w.push_u(u64::from(self.iode), 8);
        w.push_u(u64::from(self.t_oc), 16);
        w.push_i(i64::from(self.a_f2), 8);
        w.push_i(i64::from(self.a_f1), 16);
        w.push_i(i64::from(self.a_f0), 22);
        w.push_u(u64::from(self.iodc), 10);
        w.push_i(i64::from(self.c_rs), 16);
        w.push_i(i64::from(self.delta_n), 16);
        w.push_i(self.m0, 32);
        w.push_i(i64::from(self.c_uc), 16);
        w.push_u(self.eccentricity, 32);
        w.push_i(i64::from(self.c_us), 16);
        w.push_u(self.sqrt_a, 32);
        w.push_u(u64::from(self.t_oe), 16);
        w.push_i(i64::from(self.c_ic), 16);
        w.push_i(self.omega0, 32);
        w.push_i(i64::from(self.c_is), 16);
        w.push_i(self.i0, 32);
        w.push_i(i64::from(self.c_rc), 16);
        w.push_i(self.omega, 32);
        w.push_i(i64::from(self.omega_dot), 24);
        w.push_i(i64::from(self.t_gd), 8);
        w.push_u(u64::from(self.sv_health), 6);
        w.push_flag(self.l2_p_data_flag);
        w.push_flag(self.fit_interval);
        w.into_bytes()
    }

    /// Convert this decoded RTCM ephemeris to the broadcast record consumed by
    /// the solver. `full_week` is the caller-unrolled GPS week and must agree
    /// with the 10-bit RTCM week residue.
    pub fn to_broadcast_record(&self, full_week: u32) -> Result<BroadcastRecord> {
        if full_week % 1024 != u32::from(self.week_number) {
            return Err(Error::InvalidInput(format!(
                "GPS full week {full_week} disagrees with 10-bit RTCM week {}",
                self.week_number
            )));
        }
        let satellite_id = self.satellite()?;
        let toe_sow = f64::from(self.t_oe) * 16.0;
        let toc_sow = f64::from(self.t_oc) * 16.0;
        let toe = gnss_week_tow(TimeScale::Gpst, full_week, toe_sow, "GPS toe")?;
        let toc = gnss_week_tow(TimeScale::Gpst, full_week, toc_sow, "GPS toc")?;
        let fit_interval_s = gps_fit_interval_from_flag(
            i64::from(u8::from(self.fit_interval)),
            i64::from(self.iode),
            i64::from(self.iodc),
        )
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(BroadcastRecord {
            satellite_id,
            message: NavMessage::GpsLnav,
            issue_of_data: BroadcastIssue {
                issue: u32::from(self.iode),
                message: NavMessage::GpsLnav,
            },
            week: full_week,
            toe,
            toc,
            elements: KeplerianElements {
                sqrt_a: scaled_u(self.sqrt_a, -19),
                e: scaled_u(self.eccentricity, -33),
                m0: scaled_semicircle(self.m0, -31),
                delta_n: scaled_semicircle(self.delta_n, -43),
                omega0: scaled_semicircle(self.omega0, -31),
                i0: scaled_semicircle(self.i0, -31),
                omega: scaled_semicircle(self.omega, -31),
                omega_dot: scaled_semicircle(self.omega_dot, -43),
                idot: scaled_semicircle(self.idot, -43),
                cuc: scaled_i(self.c_uc, -29),
                cus: scaled_i(self.c_us, -29),
                crc: scaled_i(self.c_rc, -5),
                crs: scaled_i(self.c_rs, -5),
                cic: scaled_i(self.c_ic, -29),
                cis: scaled_i(self.c_is, -29),
                toe_sow,
            },
            clock: ClockPolynomial {
                af0: scaled_i(self.a_f0, -31),
                af1: scaled_i(self.a_f1, -43),
                af2: scaled_i(self.a_f2, -55),
                toc_sow,
            },
            group_delays: BroadcastGroupDelays::gps_lnav(scaled_i(self.t_gd, -31)),
            cnav: None,
            sv_health: f64::from(self.sv_health),
            sv_accuracy_m: ura_or_wide(self.sv_accuracy),
            fit_interval_s: Some(fit_interval_s),
        })
    }
}

/// A decoded Galileo F/NAV broadcast ephemeris (message 1045).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GalileoFnavEphemeris {
    /// Galileo SVID, decoded from the six-bit satellite field. [`Self::satellite`]
    /// validates it against the Galileo PRN range before creating a
    /// [`GnssSatelliteId`].
    pub satellite_id: u8,
    /// Galileo GST week number, as transmitted in the twelve-bit field.
    /// [`Self::to_broadcast_record`] adds the 1024-week Galileo-to-GPS epoch
    /// offset when it builds the GST-tagged reference times.
    pub week_number: u16,
    /// Ten-bit Galileo navigation-data issue copied to `BroadcastIssue.issue`.
    pub iod_nav: u16,
    /// Eight-bit Galileo SISA index; [`Self::to_broadcast_record`] converts it
    /// with the Galileo SISA table into meters.
    pub sisa: u8,
    /// Signed fourteen-bit inclination rate, scaled as 2^-43 semicircles/s in
    /// the wire message and converted to radians/s for broadcast evaluation.
    pub idot: i32,
    /// Clock reference count from the fourteen-bit field; each count is 60 s
    /// when [`Self::to_broadcast_record`] computes `toc_sow`.
    pub t_oc: u16,
    /// Signed six-bit clock drift rate, with a 2^-59 scale to s/s^2.
    pub a_f2: i16,
    /// Signed twenty-one-bit clock drift, with a 2^-46 scale to s/s.
    pub a_f1: i32,
    /// Signed thirty-one-bit clock bias, with a 2^-34 scale to seconds.
    pub a_f0: i64,
    /// Signed sixteen-bit orbit-radius sine correction, with a 2^-5 scale to
    /// meters.
    pub c_rs: i32,
    /// Signed sixteen-bit mean-motion difference, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub delta_n: i32,
    /// Signed thirty-two-bit mean anomaly, scaled as 2^-31 semicircles before
    /// conversion to radians.
    pub m0: i64,
    /// Signed sixteen-bit latitude-argument cosine correction, with a 2^-29
    /// scale to radians.
    pub c_uc: i32,
    /// Unsigned thirty-two-bit eccentricity, scaled by 2^-33 for the
    /// dimensionless broadcast element.
    pub eccentricity: u64,
    /// Signed sixteen-bit latitude-argument sine correction, with a 2^-29
    /// scale to radians.
    pub c_us: i32,
    /// Unsigned thirty-two-bit square-root semi-major axis, scaled by 2^-19
    /// to the square-root-meter value used by the broadcast evaluator.
    pub sqrt_a: u64,
    /// Ephemeris reference count from the fourteen-bit field; each count is 60 s
    /// when [`Self::to_broadcast_record`] computes `toe_sow`.
    pub t_oe: u16,
    /// Signed sixteen-bit inclination cosine correction, with a 2^-29 scale to
    /// radians.
    pub c_ic: i32,
    /// Signed thirty-two-bit ascending-node longitude, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub omega0: i64,
    /// Signed sixteen-bit inclination sine correction, with a 2^-29 scale to
    /// radians.
    pub c_is: i32,
    /// Signed thirty-two-bit reference inclination, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub i0: i64,
    /// Signed sixteen-bit orbit-radius cosine correction, with a 2^-5 scale to
    /// meters.
    pub c_rc: i32,
    /// Signed thirty-two-bit argument of perigee, scaled as 2^-31 semicircles
    /// before conversion to radians.
    pub omega: i64,
    /// Signed twenty-four-bit right-ascension rate, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub omega_dot: i32,
    /// Signed ten-bit E5a/E1 group-delay term, scaled by 2^-32 to seconds in
    /// the broadcast record.
    pub bgd_e5a_e1: i16,
    /// Two-bit E5a signal-health value. The broadcast conversion treats zero as
    /// healthy only when [`Self::e5a_data_validity`] is false.
    pub e5a_signal_health: u8,
    /// E5a data-validity flag; false is the value used by the broadcast health
    /// predicate for valid data.
    pub e5a_data_validity: bool,
    /// Seven-bit reserved tail retained by [`Self::encode`] for exact body
    /// round trips; it is not used in broadcast conversion.
    pub reserved: u8,
}

impl GalileoFnavEphemeris {
    /// Validate the decoded SVID and return its Galileo [`GnssSatelliteId`].
    pub fn satellite(&self) -> Result<GnssSatelliteId> {
        GnssSatelliteId::new(GnssSystem::Galileo, self.satellite_id)
            .map_err(|e| Error::Parse(format!("invalid Galileo SVID in 1045: {e}")))
    }

    /// Decode an unframed RTCM 1045 body.
    ///
    /// The leading twelve-bit message number must be 1045; a different number
    /// is a parse error, and a body that ends before the fields are complete is
    /// reported as an input error.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1045 {
            return Err(Error::Parse(format!(
                "message {message_number} is not Galileo F/NAV ephemeris 1045"
            ))
            .into());
        }
        Ok(Self {
            satellite_id: r.u(6)? as u8,
            week_number: r.u(12)? as u16,
            iod_nav: r.u(10)? as u16,
            sisa: r.u(8)? as u8,
            idot: r.i(14)? as i32,
            t_oc: r.u(14)? as u16,
            a_f2: r.i(6)? as i16,
            a_f1: r.i(21)? as i32,
            a_f0: r.i(31)?,
            c_rs: r.i(16)? as i32,
            delta_n: r.i(16)? as i32,
            m0: r.i(32)?,
            c_uc: r.i(16)? as i32,
            eccentricity: r.u(32)?,
            c_us: r.i(16)? as i32,
            sqrt_a: r.u(32)?,
            t_oe: r.u(14)? as u16,
            c_ic: r.i(16)? as i32,
            omega0: r.i(32)?,
            c_is: r.i(16)? as i32,
            i0: r.i(32)?,
            c_rc: r.i(16)? as i32,
            omega: r.i(32)?,
            omega_dot: r.i(24)? as i32,
            bgd_e5a_e1: r.i(10)? as i16,
            e5a_signal_health: r.u(2)? as u8,
            e5a_data_validity: r.flag()?,
            reserved: r.u(7)? as u8,
        })
    }

    /// Encode the raw fields as an unframed RTCM 1045 body in decoder order.
    /// The result preserves every field, including the reserved tail, so a
    /// decode followed by an encode retains the 62-byte body used by the
    /// round-trip test.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(1045, 12);
        w.push_u(u64::from(self.satellite_id), 6);
        w.push_u(u64::from(self.week_number), 12);
        w.push_u(u64::from(self.iod_nav), 10);
        w.push_u(u64::from(self.sisa), 8);
        w.push_i(i64::from(self.idot), 14);
        w.push_u(u64::from(self.t_oc), 14);
        w.push_i(i64::from(self.a_f2), 6);
        w.push_i(i64::from(self.a_f1), 21);
        w.push_i(self.a_f0, 31);
        w.push_i(i64::from(self.c_rs), 16);
        w.push_i(i64::from(self.delta_n), 16);
        w.push_i(self.m0, 32);
        w.push_i(i64::from(self.c_uc), 16);
        w.push_u(self.eccentricity, 32);
        w.push_i(i64::from(self.c_us), 16);
        w.push_u(self.sqrt_a, 32);
        w.push_u(u64::from(self.t_oe), 14);
        w.push_i(i64::from(self.c_ic), 16);
        w.push_i(self.omega0, 32);
        w.push_i(i64::from(self.c_is), 16);
        w.push_i(self.i0, 32);
        w.push_i(i64::from(self.c_rc), 16);
        w.push_i(self.omega, 32);
        w.push_i(i64::from(self.omega_dot), 24);
        w.push_i(i64::from(self.bgd_e5a_e1), 10);
        w.push_u(u64::from(self.e5a_signal_health), 2);
        w.push_flag(self.e5a_data_validity);
        w.push_u(u64::from(self.reserved), 7);
        w.into_bytes()
    }

    /// Convert this raw F/NAV message into the Galileo broadcast record used by
    /// the orbital evaluator. Reference counts become seconds of GST week,
    /// orbital and clock integers receive their broadcast scale factors, and
    /// `iod_nav`, SISA, group delay, and health are copied into their canonical
    /// record fields; an invalid SVID or overflowing aligned week is rejected.
    pub fn to_broadcast_record(&self) -> Result<BroadcastRecord> {
        galileo_to_record(
            self.satellite()?,
            u32::from(self.week_number),
            self.iod_nav,
            self.sisa,
            self.idot,
            self.t_oc,
            self.a_f2,
            self.a_f1,
            self.a_f0,
            self.c_rs,
            self.delta_n,
            self.m0,
            self.c_uc,
            self.eccentricity,
            self.c_us,
            self.sqrt_a,
            self.t_oe,
            self.c_ic,
            self.omega0,
            self.c_is,
            self.i0,
            self.c_rc,
            self.omega,
            self.omega_dot,
            self.bgd_e5a_e1,
            0,
            raw_health(self.e5a_signal_health == 0 && !self.e5a_data_validity),
            NavMessage::GalileoFnav,
        )
    }
}

/// A decoded Galileo I/NAV broadcast ephemeris (message 1046).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GalileoInavEphemeris {
    /// Galileo SVID, decoded from the six-bit satellite field. [`Self::satellite`]
    /// validates it against the Galileo PRN range before creating a
    /// [`GnssSatelliteId`].
    pub satellite_id: u8,
    /// Galileo GST week number, as transmitted in the twelve-bit field.
    /// [`Self::to_broadcast_record`] adds the 1024-week Galileo-to-GPS epoch
    /// offset when it builds the GST-tagged reference times.
    pub week_number: u16,
    /// Ten-bit Galileo navigation-data issue copied to `BroadcastIssue.issue`.
    pub iod_nav: u16,
    /// Eight-bit Galileo SISA index; [`Self::to_broadcast_record`] converts it
    /// with the Galileo SISA table into meters.
    pub sisa_index: u8,
    /// Signed fourteen-bit inclination rate, scaled as 2^-43 semicircles/s in
    /// the wire message and converted to radians/s for broadcast evaluation.
    pub idot: i32,
    /// Clock reference count from the fourteen-bit field; each count is 60 s
    /// when [`Self::to_broadcast_record`] computes `toc_sow`.
    pub t_oc: u16,
    /// Signed six-bit clock drift rate, with a 2^-59 scale to s/s^2.
    pub a_f2: i16,
    /// Signed twenty-one-bit clock drift, with a 2^-46 scale to s/s.
    pub a_f1: i32,
    /// Signed thirty-one-bit clock bias, with a 2^-34 scale to seconds.
    pub a_f0: i64,
    /// Signed sixteen-bit orbit-radius sine correction, with a 2^-5 scale to
    /// meters.
    pub c_rs: i32,
    /// Signed sixteen-bit mean-motion difference, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub delta_n: i32,
    /// Signed thirty-two-bit mean anomaly, scaled as 2^-31 semicircles before
    /// conversion to radians.
    pub m0: i64,
    /// Signed sixteen-bit latitude-argument cosine correction, with a 2^-29
    /// scale to radians.
    pub c_uc: i32,
    /// Unsigned thirty-two-bit eccentricity, scaled by 2^-33 for the
    /// dimensionless broadcast element.
    pub eccentricity: u64,
    /// Signed sixteen-bit latitude-argument sine correction, with a 2^-29
    /// scale to radians.
    pub c_us: i32,
    /// Unsigned thirty-two-bit square-root semi-major axis, scaled by 2^-19
    /// to the square-root-meter value used by the broadcast evaluator.
    pub sqrt_a: u64,
    /// Ephemeris reference count from the fourteen-bit field; each count is 60 s
    /// when [`Self::to_broadcast_record`] computes `toe_sow`.
    pub t_oe: u16,
    /// Signed sixteen-bit inclination cosine correction, with a 2^-29 scale to
    /// radians.
    pub c_ic: i32,
    /// Signed thirty-two-bit ascending-node longitude, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub omega0: i64,
    /// Signed sixteen-bit inclination sine correction, with a 2^-29 scale to
    /// radians.
    pub c_is: i32,
    /// Signed thirty-two-bit reference inclination, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub i0: i64,
    /// Signed sixteen-bit orbit-radius cosine correction, with a 2^-5 scale to
    /// meters.
    pub c_rc: i32,
    /// Signed thirty-two-bit argument of perigee, scaled as 2^-31 semicircles
    /// before conversion to radians.
    pub omega: i64,
    /// Signed twenty-four-bit right-ascension rate, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub omega_dot: i32,
    /// Signed ten-bit E5a/E1 group-delay term, scaled by 2^-32 to seconds in
    /// the broadcast record.
    pub bgd_e5a_e1: i16,
    /// Signed ten-bit E5b/E1 group-delay term, scaled by 2^-32 to seconds in
    /// the broadcast record.
    pub bgd_e5b_e1: i16,
    /// Two-bit E5b signal-health value; it must be zero, with both validity
    /// flags false and E1b health zero, for the broadcast record to be healthy.
    pub e5b_signal_health: u8,
    /// E5b data-validity flag; false is required by the combined healthy-data
    /// predicate in broadcast conversion.
    pub e5b_data_validity: bool,
    /// Two-bit E1b signal-health value; it must be zero, with both validity
    /// flags false and E5b health zero, for the broadcast record to be healthy.
    pub e1b_signal_health: u8,
    /// E1b data-validity flag; false is required by the combined healthy-data
    /// predicate in broadcast conversion.
    pub e1b_data_validity: bool,
    /// Two-bit reserved tail retained by [`Self::encode`] for exact body round
    /// trips; it is not used in broadcast conversion.
    pub reserved: u8,
}

impl GalileoInavEphemeris {
    /// Validate the decoded SVID and return its Galileo [`GnssSatelliteId`].
    pub fn satellite(&self) -> Result<GnssSatelliteId> {
        GnssSatelliteId::new(GnssSystem::Galileo, self.satellite_id)
            .map_err(|e| Error::Parse(format!("invalid Galileo SVID in 1046: {e}")))
    }

    /// Decode an unframed RTCM 1046 body.
    ///
    /// The leading twelve-bit message number must be 1046; a different number
    /// is a parse error, and a body that ends before the fields are complete is
    /// reported as an input error.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1046 {
            return Err(Error::Parse(format!(
                "message {message_number} is not Galileo I/NAV ephemeris 1046"
            ))
            .into());
        }
        Ok(Self {
            satellite_id: r.u(6)? as u8,
            week_number: r.u(12)? as u16,
            iod_nav: r.u(10)? as u16,
            sisa_index: r.u(8)? as u8,
            idot: r.i(14)? as i32,
            t_oc: r.u(14)? as u16,
            a_f2: r.i(6)? as i16,
            a_f1: r.i(21)? as i32,
            a_f0: r.i(31)?,
            c_rs: r.i(16)? as i32,
            delta_n: r.i(16)? as i32,
            m0: r.i(32)?,
            c_uc: r.i(16)? as i32,
            eccentricity: r.u(32)?,
            c_us: r.i(16)? as i32,
            sqrt_a: r.u(32)?,
            t_oe: r.u(14)? as u16,
            c_ic: r.i(16)? as i32,
            omega0: r.i(32)?,
            c_is: r.i(16)? as i32,
            i0: r.i(32)?,
            c_rc: r.i(16)? as i32,
            omega: r.i(32)?,
            omega_dot: r.i(24)? as i32,
            bgd_e5a_e1: r.i(10)? as i16,
            bgd_e5b_e1: r.i(10)? as i16,
            e5b_signal_health: r.u(2)? as u8,
            e5b_data_validity: r.flag()?,
            e1b_signal_health: r.u(2)? as u8,
            e1b_data_validity: r.flag()?,
            reserved: r.u(2)? as u8,
        })
    }

    /// Encode the raw fields as an unframed RTCM 1046 body in decoder order.
    /// The result preserves every field, including the reserved tail, so a
    /// decode followed by an encode retains the 63-byte body used by the
    /// round-trip test.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(1046, 12);
        w.push_u(u64::from(self.satellite_id), 6);
        w.push_u(u64::from(self.week_number), 12);
        w.push_u(u64::from(self.iod_nav), 10);
        w.push_u(u64::from(self.sisa_index), 8);
        w.push_i(i64::from(self.idot), 14);
        w.push_u(u64::from(self.t_oc), 14);
        w.push_i(i64::from(self.a_f2), 6);
        w.push_i(i64::from(self.a_f1), 21);
        w.push_i(self.a_f0, 31);
        w.push_i(i64::from(self.c_rs), 16);
        w.push_i(i64::from(self.delta_n), 16);
        w.push_i(self.m0, 32);
        w.push_i(i64::from(self.c_uc), 16);
        w.push_u(self.eccentricity, 32);
        w.push_i(i64::from(self.c_us), 16);
        w.push_u(self.sqrt_a, 32);
        w.push_u(u64::from(self.t_oe), 14);
        w.push_i(i64::from(self.c_ic), 16);
        w.push_i(self.omega0, 32);
        w.push_i(i64::from(self.c_is), 16);
        w.push_i(self.i0, 32);
        w.push_i(i64::from(self.c_rc), 16);
        w.push_i(self.omega, 32);
        w.push_i(i64::from(self.omega_dot), 24);
        w.push_i(i64::from(self.bgd_e5a_e1), 10);
        w.push_i(i64::from(self.bgd_e5b_e1), 10);
        w.push_u(u64::from(self.e5b_signal_health), 2);
        w.push_flag(self.e5b_data_validity);
        w.push_u(u64::from(self.e1b_signal_health), 2);
        w.push_flag(self.e1b_data_validity);
        w.push_u(u64::from(self.reserved), 2);
        w.into_bytes()
    }

    /// Convert this raw I/NAV message into the Galileo broadcast record used by
    /// the orbital evaluator. Reference counts become seconds of GST week,
    /// orbital and clock integers receive their broadcast scale factors, and
    /// both group delays plus the combined signal-health state are retained;
    /// an invalid SVID or overflowing aligned week is rejected.
    pub fn to_broadcast_record(&self) -> Result<BroadcastRecord> {
        galileo_to_record(
            self.satellite()?,
            u32::from(self.week_number),
            self.iod_nav,
            self.sisa_index,
            self.idot,
            self.t_oc,
            self.a_f2,
            self.a_f1,
            self.a_f0,
            self.c_rs,
            self.delta_n,
            self.m0,
            self.c_uc,
            self.eccentricity,
            self.c_us,
            self.sqrt_a,
            self.t_oe,
            self.c_ic,
            self.omega0,
            self.c_is,
            self.i0,
            self.c_rc,
            self.omega,
            self.omega_dot,
            self.bgd_e5a_e1,
            self.bgd_e5b_e1,
            raw_health(
                self.e1b_signal_health == 0
                    && !self.e1b_data_validity
                    && self.e5b_signal_health == 0
                    && !self.e5b_data_validity,
            ),
            NavMessage::GalileoInav,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn galileo_to_record(
    satellite_id: GnssSatelliteId,
    week: u32,
    iod_nav: u16,
    sisa: u8,
    idot: i32,
    t_oc: u16,
    a_f2: i16,
    a_f1: i32,
    a_f0: i64,
    c_rs: i32,
    delta_n: i32,
    m0: i64,
    c_uc: i32,
    eccentricity: u64,
    c_us: i32,
    sqrt_a: u64,
    t_oe: u16,
    c_ic: i32,
    omega0: i64,
    c_is: i32,
    i0: i64,
    c_rc: i32,
    omega: i64,
    omega_dot: i32,
    bgd_e5a_e1: i16,
    bgd_e5b_e1: i16,
    sv_health: f64,
    message: NavMessage,
) -> Result<BroadcastRecord> {
    let toe_sow = f64::from(t_oe) * 60.0;
    let toc_sow = f64::from(t_oc) * 60.0;
    let gps_aligned_week = week
        .checked_add(GALILEO_WEEK_OFFSET_TO_GPS)
        .ok_or_else(|| Error::InvalidInput("RTCM Galileo week overflows GPST axis".to_string()))?;
    let toe = gnss_week_tow(TimeScale::Gst, gps_aligned_week, toe_sow, "Galileo toe")?;
    let toc = gnss_week_tow(TimeScale::Gst, gps_aligned_week, toc_sow, "Galileo toc")?;
    Ok(BroadcastRecord {
        satellite_id,
        message,
        issue_of_data: BroadcastIssue {
            issue: u32::from(iod_nav),
            message,
        },
        week: gps_aligned_week,
        toe,
        toc,
        elements: KeplerianElements {
            sqrt_a: scaled_u(sqrt_a, -19),
            e: scaled_u(eccentricity, -33),
            m0: scaled_semicircle(m0, -31),
            delta_n: scaled_semicircle(delta_n, -43),
            omega0: scaled_semicircle(omega0, -31),
            i0: scaled_semicircle(i0, -31),
            omega: scaled_semicircle(omega, -31),
            omega_dot: scaled_semicircle(omega_dot, -43),
            idot: scaled_semicircle(idot, -43),
            cuc: scaled_i(c_uc, -29),
            cus: scaled_i(c_us, -29),
            crc: scaled_i(c_rc, -5),
            crs: scaled_i(c_rs, -5),
            cic: scaled_i(c_ic, -29),
            cis: scaled_i(c_is, -29),
            toe_sow,
        },
        clock: ClockPolynomial {
            af0: scaled_i(a_f0, -34),
            af1: scaled_i(a_f1, -46),
            af2: scaled_i(a_f2, -59),
            toc_sow,
        },
        group_delays: BroadcastGroupDelays::galileo(
            scaled_i(bgd_e5a_e1, -32),
            scaled_i(bgd_e5b_e1, -32),
        ),
        cnav: None,
        sv_health,
        sv_accuracy_m: galileo_sisa_m(sisa),
        fit_interval_s: None,
    })
}

/// A decoded BeiDou broadcast ephemeris (message 1042).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeidouEphemeris {
    /// BeiDou satellite number, decoded from the six-bit satellite field.
    /// [`Self::satellite`] validates it against the BeiDou PRN range before
    /// creating a [`GnssSatelliteId`].
    pub satellite_id: u8,
    /// Native BeiDou BDT week from the thirteen-bit field; the conversion uses
    /// it unchanged for the record week and both BDT reference times.
    pub week_number: u16,
    /// Four-bit BeiDou signal-in-space accuracy index, mapped to meters by the
    /// GPS URA table used in [`Self::to_broadcast_record`].
    pub sv_urai: u8,
    /// Signed fourteen-bit inclination rate, scaled as 2^-43 semicircles/s and
    /// converted to radians/s for broadcast evaluation.
    pub idot: i32,
    /// Five-bit ephemeris-data issue copied to `BroadcastIssue.issue`; the
    /// message is D1 or D2 according to the satellite's GEO classification.
    pub aode: u8,
    /// Clock reference count from the seventeen-bit field; each count is 8 s
    /// when [`Self::to_broadcast_record`] computes `toc_sow`.
    pub t_oc: u32,
    /// Signed eleven-bit clock drift rate, with a 2^-66 scale to s/s^2.
    pub a_f2: i16,
    /// Signed twenty-two-bit clock drift, with a 2^-50 scale to s/s.
    pub a_f1: i32,
    /// Signed twenty-four-bit clock bias, with a 2^-33 scale to seconds.
    pub a_f0: i32,
    /// Five-bit clock-data issue retained for raw round trips; broadcast issue
    /// metadata comes from [`Self::aode`] instead.
    pub aodc: u8,
    /// Signed eighteen-bit orbit-radius sine correction, with a 2^-6 scale to
    /// meters.
    pub c_rs: i32,
    /// Signed sixteen-bit mean-motion difference, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub delta_n: i32,
    /// Signed thirty-two-bit mean anomaly, scaled as 2^-31 semicircles before
    /// conversion to radians.
    pub m0: i64,
    /// Signed eighteen-bit latitude-argument cosine correction, with a 2^-31
    /// scale to radians.
    pub c_uc: i32,
    /// Unsigned thirty-two-bit eccentricity, scaled by 2^-33 for the
    /// dimensionless broadcast element.
    pub eccentricity: u64,
    /// Signed eighteen-bit latitude-argument sine correction, with a 2^-31
    /// scale to radians.
    pub c_us: i32,
    /// Unsigned thirty-two-bit square-root semi-major axis, scaled by 2^-19
    /// to the square-root-meter value used by the broadcast evaluator.
    pub sqrt_a: u64,
    /// Ephemeris reference count from the seventeen-bit field; each count is 8 s
    /// when [`Self::to_broadcast_record`] computes `toe_sow`.
    pub t_oe: u32,
    /// Signed eighteen-bit inclination cosine correction, with a 2^-31 scale to
    /// radians.
    pub c_ic: i32,
    /// Signed thirty-two-bit ascending-node longitude, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub omega0: i64,
    /// Signed eighteen-bit inclination sine correction, with a 2^-31 scale to
    /// radians.
    pub c_is: i32,
    /// Signed thirty-two-bit reference inclination, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub i0: i64,
    /// Signed eighteen-bit orbit-radius cosine correction, with a 2^-6 scale to
    /// meters.
    pub c_rc: i32,
    /// Signed thirty-two-bit argument of perigee, scaled as 2^-31 semicircles
    /// before conversion to radians.
    pub omega: i64,
    /// Signed twenty-four-bit right-ascension rate, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub omega_dot: i32,
    /// Signed ten-bit TGD1 value, multiplied by 1e-10 to produce seconds in the
    /// BeiDou group-delay record.
    pub t_gd1: i16,
    /// Signed ten-bit TGD2 value, multiplied by 1e-10 to produce seconds in the
    /// BeiDou group-delay record.
    pub t_gd2: i16,
    /// BeiDou health flag; false becomes `0.0` and true becomes `1.0` in the
    /// broadcast record.
    pub sv_health: bool,
}

impl BeidouEphemeris {
    /// Validate the decoded satellite number and return its BeiDou
    /// [`GnssSatelliteId`].
    pub fn satellite(&self) -> Result<GnssSatelliteId> {
        GnssSatelliteId::new(GnssSystem::BeiDou, self.satellite_id)
            .map_err(|e| Error::Parse(format!("invalid BeiDou satellite ID in 1042: {e}")))
    }

    /// Decode an unframed RTCM 1042 body.
    ///
    /// The leading twelve-bit message number must be 1042; a different number
    /// is a parse error, and a body that ends before the fields are complete is
    /// reported as an input error.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1042 {
            return Err(Error::Parse(format!(
                "message {message_number} is not BeiDou ephemeris 1042"
            ))
            .into());
        }
        Ok(Self {
            satellite_id: r.u(6)? as u8,
            week_number: r.u(13)? as u16,
            sv_urai: r.u(4)? as u8,
            idot: r.i(14)? as i32,
            aode: r.u(5)? as u8,
            t_oc: r.u(17)? as u32,
            a_f2: r.i(11)? as i16,
            a_f1: r.i(22)? as i32,
            a_f0: r.i(24)? as i32,
            aodc: r.u(5)? as u8,
            c_rs: r.i(18)? as i32,
            delta_n: r.i(16)? as i32,
            m0: r.i(32)?,
            c_uc: r.i(18)? as i32,
            eccentricity: r.u(32)?,
            c_us: r.i(18)? as i32,
            sqrt_a: r.u(32)?,
            t_oe: r.u(17)? as u32,
            c_ic: r.i(18)? as i32,
            omega0: r.i(32)?,
            c_is: r.i(18)? as i32,
            i0: r.i(32)?,
            c_rc: r.i(18)? as i32,
            omega: r.i(32)?,
            omega_dot: r.i(24)? as i32,
            t_gd1: r.i(10)? as i16,
            t_gd2: r.i(10)? as i16,
            sv_health: r.flag()?,
        })
    }

    /// Encode the raw fields as an unframed RTCM 1042 body in decoder order.
    /// The result preserves every field, including both delay terms, so a
    /// decode followed by an encode retains the 64-byte body used by the
    /// round-trip test.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(1042, 12);
        w.push_u(u64::from(self.satellite_id), 6);
        w.push_u(u64::from(self.week_number), 13);
        w.push_u(u64::from(self.sv_urai), 4);
        w.push_i(i64::from(self.idot), 14);
        w.push_u(u64::from(self.aode), 5);
        w.push_u(u64::from(self.t_oc), 17);
        w.push_i(i64::from(self.a_f2), 11);
        w.push_i(i64::from(self.a_f1), 22);
        w.push_i(i64::from(self.a_f0), 24);
        w.push_u(u64::from(self.aodc), 5);
        w.push_i(i64::from(self.c_rs), 18);
        w.push_i(i64::from(self.delta_n), 16);
        w.push_i(self.m0, 32);
        w.push_i(i64::from(self.c_uc), 18);
        w.push_u(self.eccentricity, 32);
        w.push_i(i64::from(self.c_us), 18);
        w.push_u(self.sqrt_a, 32);
        w.push_u(u64::from(self.t_oe), 17);
        w.push_i(i64::from(self.c_ic), 18);
        w.push_i(self.omega0, 32);
        w.push_i(i64::from(self.c_is), 18);
        w.push_i(self.i0, 32);
        w.push_i(i64::from(self.c_rc), 18);
        w.push_i(self.omega, 32);
        w.push_i(i64::from(self.omega_dot), 24);
        w.push_i(i64::from(self.t_gd1), 10);
        w.push_i(i64::from(self.t_gd2), 10);
        w.push_flag(self.sv_health);
        w.into_bytes()
    }

    /// Convert this raw message into the BDT-tagged BeiDou broadcast record
    /// used by the orbital evaluator. The satellite selects the D1 or D2
    /// message tag, integer fields receive their broadcast scales, and both
    /// TGD terms are retained; an invalid satellite or unrepresentable time is
    /// rejected.
    pub fn to_broadcast_record(&self) -> Result<BroadcastRecord> {
        let satellite_id = self.satellite()?;
        let week = u32::from(self.week_number);
        let toe_sow = f64::from(self.t_oe) * 8.0;
        let toc_sow = f64::from(self.t_oc) * 8.0;
        let toe = gnss_week_tow(TimeScale::Bdt, week, toe_sow, "BeiDou toe")?;
        let toc = gnss_week_tow(TimeScale::Bdt, week, toc_sow, "BeiDou toc")?;
        let message = if crate::rinex_nav::is_beidou_geo(satellite_id) {
            NavMessage::BeidouD2
        } else {
            NavMessage::BeidouD1
        };
        Ok(BroadcastRecord {
            satellite_id,
            message,
            issue_of_data: BroadcastIssue {
                issue: u32::from(self.aode),
                message,
            },
            week,
            toe,
            toc,
            elements: KeplerianElements {
                sqrt_a: scaled_u(self.sqrt_a, -19),
                e: scaled_u(self.eccentricity, -33),
                m0: scaled_semicircle(self.m0, -31),
                delta_n: scaled_semicircle(self.delta_n, -43),
                omega0: scaled_semicircle(self.omega0, -31),
                i0: scaled_semicircle(self.i0, -31),
                omega: scaled_semicircle(self.omega, -31),
                omega_dot: scaled_semicircle(self.omega_dot, -43),
                idot: scaled_semicircle(self.idot, -43),
                cuc: scaled_i(self.c_uc, -31),
                cus: scaled_i(self.c_us, -31),
                crc: scaled_i(self.c_rc, -6),
                crs: scaled_i(self.c_rs, -6),
                cic: scaled_i(self.c_ic, -31),
                cis: scaled_i(self.c_is, -31),
                toe_sow,
            },
            clock: ClockPolynomial {
                af0: scaled_i(self.a_f0, -33),
                af1: scaled_i(self.a_f1, -50),
                af2: scaled_i(self.a_f2, -66),
                toc_sow,
            },
            group_delays: BroadcastGroupDelays::beidou(
                f64::from(self.t_gd1) * 1.0e-10,
                f64::from(self.t_gd2) * 1.0e-10,
            ),
            cnav: None,
            sv_health: f64::from(u8::from(self.sv_health)),
            sv_accuracy_m: ura_or_wide(self.sv_urai),
            fit_interval_s: None,
        })
    }
}

/// A decoded QZSS broadcast ephemeris (message 1044).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QzssEphemeris {
    /// QZSS satellite number, decoded from the four-bit satellite field.
    /// [`Self::satellite`] validates it against the QZSS PRN range before
    /// creating a [`GnssSatelliteId`].
    pub satellite_id: u8,
    /// Clock reference count from the sixteen-bit field; each count is 16 s
    /// when [`Self::to_broadcast_record`] computes `toc_sow`.
    pub t_oc: u16,
    /// Signed eight-bit clock drift rate, with a 2^-55 scale to s/s^2.
    pub a_f2: i16,
    /// Signed sixteen-bit clock drift, with a 2^-43 scale to s/s.
    pub a_f1: i32,
    /// Signed twenty-two-bit clock bias, with a 2^-31 scale to seconds.
    pub a_f0: i32,
    /// Eight-bit ephemeris-data issue copied to `BroadcastIssue.issue`.
    pub iode: u8,
    /// Signed sixteen-bit orbit-radius sine correction, with a 2^-5 scale to
    /// meters.
    pub c_rs: i32,
    /// Signed sixteen-bit mean-motion difference, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub delta_n: i32,
    /// Signed thirty-two-bit mean anomaly, scaled as 2^-31 semicircles before
    /// conversion to radians.
    pub m0: i64,
    /// Signed sixteen-bit latitude-argument cosine correction, with a 2^-29
    /// scale to radians.
    pub c_uc: i32,
    /// Unsigned thirty-two-bit eccentricity, scaled by 2^-33 for the
    /// dimensionless broadcast element.
    pub eccentricity: u64,
    /// Signed sixteen-bit latitude-argument sine correction, with a 2^-29
    /// scale to radians.
    pub c_us: i32,
    /// Unsigned thirty-two-bit square-root semi-major axis, scaled by 2^-19
    /// to the square-root-meter value used by the broadcast evaluator.
    pub sqrt_a: u64,
    /// Ephemeris reference count from the sixteen-bit field; each count is 16 s
    /// when [`Self::to_broadcast_record`] computes `toe_sow`.
    pub t_oe: u16,
    /// Signed sixteen-bit inclination cosine correction, with a 2^-29 scale to
    /// radians.
    pub c_ic: i32,
    /// Signed thirty-two-bit ascending-node longitude, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub omega0: i64,
    /// Signed sixteen-bit inclination sine correction, with a 2^-29 scale to
    /// radians.
    pub c_is: i32,
    /// Signed thirty-two-bit reference inclination, scaled as 2^-31
    /// semicircles before conversion to radians.
    pub i0: i64,
    /// Signed sixteen-bit orbit-radius cosine correction, with a 2^-5 scale to
    /// meters.
    pub c_rc: i32,
    /// Signed thirty-two-bit argument of perigee, scaled as 2^-31 semicircles
    /// before conversion to radians.
    pub omega: i64,
    /// Signed twenty-four-bit right-ascension rate, scaled as 2^-43
    /// semicircles/s before conversion to radians/s.
    pub omega_dot: i32,
    /// Signed fourteen-bit inclination rate, scaled as 2^-43 semicircles/s and
    /// converted to radians/s for broadcast evaluation.
    pub idot: i32,
    /// Two-bit L2 code value retained by [`Self::encode`]; the broadcast record
    /// does not consume this raw signal indicator.
    pub codes_on_l2: u8,
    /// Ten-bit GPS week residue checked against the caller-supplied `full_week`
    /// by [`Self::to_broadcast_record`].
    pub week_number: u16,
    /// Four-bit GPS URA index, mapped to meters by the GPS URA table.
    pub ura: u8,
    /// Six-bit satellite-health word copied numerically to the broadcast
    /// record, where zero is the healthy convention.
    pub sv_health: u8,
    /// Signed eight-bit GPS-style group-delay value, scaled by 2^-31 to
    /// seconds.
    pub t_gd: i16,
    /// Ten-bit clock-data issue retained for exact encoding; broadcast issue
    /// metadata comes from [`Self::iode`] instead.
    pub iodc: u16,
    /// Fit-interval flag: false becomes two hours and true becomes six hours in
    /// `BroadcastRecord.fit_interval_s`.
    pub fit_interval: bool,
}

impl QzssEphemeris {
    /// Validate the decoded satellite number and return its QZSS
    /// [`GnssSatelliteId`].
    pub fn satellite(&self) -> Result<GnssSatelliteId> {
        GnssSatelliteId::new(GnssSystem::Qzss, self.satellite_id)
            .map_err(|e| Error::Parse(format!("invalid QZSS satellite ID in 1044: {e}")))
    }

    /// Decode an unframed RTCM 1044 body.
    ///
    /// The leading twelve-bit message number must be 1044; a different number
    /// is a parse error, and a body that ends before the fields are complete is
    /// reported as an input error.
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1044 {
            return Err(Error::Parse(format!(
                "message {message_number} is not QZSS ephemeris 1044"
            ))
            .into());
        }
        Ok(Self {
            satellite_id: r.u(4)? as u8,
            t_oc: r.u(16)? as u16,
            a_f2: r.i(8)? as i16,
            a_f1: r.i(16)? as i32,
            a_f0: r.i(22)? as i32,
            iode: r.u(8)? as u8,
            c_rs: r.i(16)? as i32,
            delta_n: r.i(16)? as i32,
            m0: r.i(32)?,
            c_uc: r.i(16)? as i32,
            eccentricity: r.u(32)?,
            c_us: r.i(16)? as i32,
            sqrt_a: r.u(32)?,
            t_oe: r.u(16)? as u16,
            c_ic: r.i(16)? as i32,
            omega0: r.i(32)?,
            c_is: r.i(16)? as i32,
            i0: r.i(32)?,
            c_rc: r.i(16)? as i32,
            omega: r.i(32)?,
            omega_dot: r.i(24)? as i32,
            idot: r.i(14)? as i32,
            codes_on_l2: r.u(2)? as u8,
            week_number: r.u(10)? as u16,
            ura: r.u(4)? as u8,
            sv_health: r.u(6)? as u8,
            t_gd: r.i(8)? as i16,
            iodc: r.u(10)? as u16,
            fit_interval: r.flag()?,
        })
    }

    /// Encode the raw fields as an unframed RTCM 1044 body in decoder order.
    /// The result preserves every field, including the clock-data issue, so a
    /// decode followed by an encode retains the 61-byte body used by the
    /// round-trip test.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(1044, 12);
        w.push_u(u64::from(self.satellite_id), 4);
        w.push_u(u64::from(self.t_oc), 16);
        w.push_i(i64::from(self.a_f2), 8);
        w.push_i(i64::from(self.a_f1), 16);
        w.push_i(i64::from(self.a_f0), 22);
        w.push_u(u64::from(self.iode), 8);
        w.push_i(i64::from(self.c_rs), 16);
        w.push_i(i64::from(self.delta_n), 16);
        w.push_i(self.m0, 32);
        w.push_i(i64::from(self.c_uc), 16);
        w.push_u(self.eccentricity, 32);
        w.push_i(i64::from(self.c_us), 16);
        w.push_u(self.sqrt_a, 32);
        w.push_u(u64::from(self.t_oe), 16);
        w.push_i(i64::from(self.c_ic), 16);
        w.push_i(self.omega0, 32);
        w.push_i(i64::from(self.c_is), 16);
        w.push_i(self.i0, 32);
        w.push_i(i64::from(self.c_rc), 16);
        w.push_i(self.omega, 32);
        w.push_i(i64::from(self.omega_dot), 24);
        w.push_i(i64::from(self.idot), 14);
        w.push_u(u64::from(self.codes_on_l2), 2);
        w.push_u(u64::from(self.week_number), 10);
        w.push_u(u64::from(self.ura), 4);
        w.push_u(u64::from(self.sv_health), 6);
        w.push_i(i64::from(self.t_gd), 8);
        w.push_u(u64::from(self.iodc), 10);
        w.push_flag(self.fit_interval);
        w.into_bytes()
    }

    /// Convert this raw message into the GPST-tagged QZSS L/NAV broadcast record
    /// used by the orbital evaluator. `full_week` must have the same ten-bit
    /// residue as [`Self::week_number`]; reference counts and scale factors are
    /// applied before the record is returned, and mismatched weeks, invalid
    /// satellite IDs, or unrepresentable times are rejected.
    pub fn to_broadcast_record(&self, full_week: u32) -> Result<BroadcastRecord> {
        if full_week % 1024 != u32::from(self.week_number) {
            return Err(Error::InvalidInput(format!(
                "QZSS full week {full_week} disagrees with 10-bit RTCM week {}",
                self.week_number
            )));
        }
        let satellite_id = self.satellite()?;
        let toe_sow = f64::from(self.t_oe) * 16.0;
        let toc_sow = f64::from(self.t_oc) * 16.0;
        let toe = gnss_week_tow(TimeScale::Gpst, full_week, toe_sow, "QZSS toe")?;
        let toc = gnss_week_tow(TimeScale::Gpst, full_week, toc_sow, "QZSS toc")?;
        Ok(BroadcastRecord {
            satellite_id,
            message: NavMessage::QzssLnav,
            issue_of_data: BroadcastIssue {
                issue: u32::from(self.iode),
                message: NavMessage::QzssLnav,
            },
            week: full_week,
            toe,
            toc,
            elements: KeplerianElements {
                sqrt_a: scaled_u(self.sqrt_a, -19),
                e: scaled_u(self.eccentricity, -33),
                m0: scaled_semicircle(self.m0, -31),
                delta_n: scaled_semicircle(self.delta_n, -43),
                omega0: scaled_semicircle(self.omega0, -31),
                i0: scaled_semicircle(self.i0, -31),
                omega: scaled_semicircle(self.omega, -31),
                omega_dot: scaled_semicircle(self.omega_dot, -43),
                idot: scaled_semicircle(self.idot, -43),
                cuc: scaled_i(self.c_uc, -29),
                cus: scaled_i(self.c_us, -29),
                crc: scaled_i(self.c_rc, -5),
                crs: scaled_i(self.c_rs, -5),
                cic: scaled_i(self.c_ic, -29),
                cis: scaled_i(self.c_is, -29),
                toe_sow,
            },
            clock: ClockPolynomial {
                af0: scaled_i(self.a_f0, -31),
                af1: scaled_i(self.a_f1, -43),
                af2: scaled_i(self.a_f2, -55),
                toc_sow,
            },
            group_delays: BroadcastGroupDelays::gps_lnav(scaled_i(self.t_gd, -31)),
            cnav: None,
            sv_health: f64::from(self.sv_health),
            sv_accuracy_m: ura_or_wide(self.ura),
            fit_interval_s: Some(if self.fit_interval {
                6.0 * SECONDS_PER_HOUR
            } else {
                2.0 * SECONDS_PER_HOUR
            }),
        })
    }
}

/// A decoded GLONASS broadcast ephemeris (message 1020).
///
/// The orbit position / velocity / acceleration terms use sign-and-magnitude
/// integers (DF111..DF119). Every field below is the raw transmitted integer;
/// the noted scale factors recover km, km/s, km/s^2, and seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlonassEphemeris {
    /// GLONASS satellite slot number (DF038, 6 bits).
    pub satellite_id: u8,
    /// Frequency channel number (DF040, 5 bits; the wire value is k + 7).
    pub frequency_channel: u8,
    /// Almanac health C_n (DF104).
    pub almanac_health: bool,
    /// Almanac health availability (DF105).
    pub almanac_health_availability: bool,
    /// P1 flag (DF106, 2 bits).
    pub p1: u8,
    /// Frame time t_k (DF107, 12 bits).
    pub t_k: u16,
    /// MSB of the B_n health word (DF108).
    pub b_n_msb: bool,
    /// P2 flag (DF109).
    pub p2: bool,
    /// Ephemeris reference time t_b (DF110, 7 bits).
    pub t_b: u8,
    /// X-velocity (DF111, sign-magnitude 24-bit, scale 2^-20 km/s).
    pub xn_dot: i32,
    /// X-position (DF112, sign-magnitude 27-bit, scale 2^-11 km).
    pub xn: i32,
    /// X-acceleration (DF113, sign-magnitude 5-bit, scale 2^-30 km/s^2).
    pub xn_dot_dot: i8,
    /// Y-velocity (DF114, sign-magnitude 24-bit, scale 2^-20 km/s).
    pub yn_dot: i32,
    /// Y-position (DF115, sign-magnitude 27-bit, scale 2^-11 km).
    pub yn: i32,
    /// Y-acceleration (DF116, sign-magnitude 5-bit, scale 2^-30 km/s^2).
    pub yn_dot_dot: i8,
    /// Z-velocity (DF117, sign-magnitude 24-bit, scale 2^-20 km/s).
    pub zn_dot: i32,
    /// Z-position (DF118, sign-magnitude 27-bit, scale 2^-11 km).
    pub zn: i32,
    /// Z-acceleration (DF119, sign-magnitude 5-bit, scale 2^-30 km/s^2).
    pub zn_dot_dot: i8,
    /// P3 flag (DF120).
    pub p3: bool,
    /// Relative carrier-frequency offset gamma_n (DF121, sign-magnitude 11-bit,
    /// scale 2^-40).
    pub gamma_n: i16,
    /// GLONASS-M P flag (DF122, 2 bits).
    pub m_p: u8,
    /// Third-string l_n health flag (DF123).
    pub m_l_n_third: bool,
    /// Clock bias tau_n (DF124, sign-magnitude 22-bit, scale 2^-30 s).
    pub tau_n: i32,
    /// Inter-frequency bias delta_tau_n (DF125, sign-magnitude 5-bit, scale
    /// 2^-30 s).
    pub delta_tau_n: i8,
    /// Age of operation E_n (DF126, 5 bits, days).
    pub e_n: u8,
    /// GLONASS-M P4 flag (DF127).
    pub m_p4: bool,
    /// GLONASS-M F_t accuracy index (DF128, 4 bits).
    pub m_f_t: u8,
    /// GLONASS-M N_t calendar day number (DF129, 11 bits).
    pub m_n_t: u16,
    /// GLONASS-M M satellite type (DF130, 2 bits).
    pub m_m: u8,
    /// Additional data availability (DF131).
    pub additional_data_available: bool,
    /// N_A almanac reference day (DF132, 11 bits).
    pub n_a: u16,
    /// System time scale offset tau_c (DF133, sign-magnitude 32-bit, scale
    /// 2^-31 s).
    pub tau_c: i64,
    /// GLONASS-M N_4 four-year interval number (DF134, 5 bits).
    pub m_n4: u8,
    /// GLONASS-M tau_GPS offset to GPS time (DF135, sign-magnitude 22-bit, scale
    /// 2^-30 s).
    pub m_tau_gps: i32,
    /// Fifth-string l_n health flag (DF136).
    pub m_l_n_fifth: bool,
    /// Reserved field DF001 (7 bits), preserved for exact round-trip.
    pub reserved: u8,
}

impl GlonassEphemeris {
    /// The satellite identifier for this ephemeris.
    pub fn satellite(&self) -> Result<GnssSatelliteId> {
        GnssSatelliteId::new(GnssSystem::Glonass, self.satellite_id)
            .map_err(|e| Error::Parse(format!("invalid GLONASS slot in 1020: {e}")))
    }

    /// Decode a message 1020 body (without the transport frame).
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_inner(body).map_err(Into::into)
    }

    pub(crate) fn decode_inner(body: &[u8]) -> DecodeResult<Self> {
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1020 {
            return Err(Error::Parse(format!(
                "message {message_number} is not GLONASS ephemeris 1020"
            ))
            .into());
        }
        Ok(Self {
            satellite_id: r.u(6)? as u8,
            frequency_channel: r.u(5)? as u8,
            almanac_health: r.flag()?,
            almanac_health_availability: r.flag()?,
            p1: r.u(2)? as u8,
            t_k: r.u(12)? as u16,
            b_n_msb: r.flag()?,
            p2: r.flag()?,
            t_b: r.u(7)? as u8,
            xn_dot: r.ism(24)? as i32,
            xn: r.ism(27)? as i32,
            xn_dot_dot: r.ism(5)? as i8,
            yn_dot: r.ism(24)? as i32,
            yn: r.ism(27)? as i32,
            yn_dot_dot: r.ism(5)? as i8,
            zn_dot: r.ism(24)? as i32,
            zn: r.ism(27)? as i32,
            zn_dot_dot: r.ism(5)? as i8,
            p3: r.flag()?,
            gamma_n: r.ism(11)? as i16,
            m_p: r.u(2)? as u8,
            m_l_n_third: r.flag()?,
            tau_n: r.ism(22)? as i32,
            delta_tau_n: r.ism(5)? as i8,
            e_n: r.u(5)? as u8,
            m_p4: r.flag()?,
            m_f_t: r.u(4)? as u8,
            m_n_t: r.u(11)? as u16,
            m_m: r.u(2)? as u8,
            additional_data_available: r.flag()?,
            n_a: r.u(11)? as u16,
            tau_c: r.ism(32)?,
            m_n4: r.u(5)? as u8,
            m_tau_gps: r.ism(22)? as i32,
            m_l_n_fifth: r.flag()?,
            reserved: r.u(7)? as u8,
        })
    }

    /// Encode this GLONASS ephemeris body (without the transport frame).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_u(1020, 12);
        w.push_u(u64::from(self.satellite_id), 6);
        w.push_u(u64::from(self.frequency_channel), 5);
        w.push_flag(self.almanac_health);
        w.push_flag(self.almanac_health_availability);
        w.push_u(u64::from(self.p1), 2);
        w.push_u(u64::from(self.t_k), 12);
        w.push_flag(self.b_n_msb);
        w.push_flag(self.p2);
        w.push_u(u64::from(self.t_b), 7);
        w.push_ism(i64::from(self.xn_dot), 24);
        w.push_ism(i64::from(self.xn), 27);
        w.push_ism(i64::from(self.xn_dot_dot), 5);
        w.push_ism(i64::from(self.yn_dot), 24);
        w.push_ism(i64::from(self.yn), 27);
        w.push_ism(i64::from(self.yn_dot_dot), 5);
        w.push_ism(i64::from(self.zn_dot), 24);
        w.push_ism(i64::from(self.zn), 27);
        w.push_ism(i64::from(self.zn_dot_dot), 5);
        w.push_flag(self.p3);
        w.push_ism(i64::from(self.gamma_n), 11);
        w.push_u(u64::from(self.m_p), 2);
        w.push_flag(self.m_l_n_third);
        w.push_ism(i64::from(self.tau_n), 22);
        w.push_ism(i64::from(self.delta_tau_n), 5);
        w.push_u(u64::from(self.e_n), 5);
        w.push_flag(self.m_p4);
        w.push_u(u64::from(self.m_f_t), 4);
        w.push_u(u64::from(self.m_n_t), 11);
        w.push_u(u64::from(self.m_m), 2);
        w.push_flag(self.additional_data_available);
        w.push_u(u64::from(self.n_a), 11);
        w.push_ism(self.tau_c, 32);
        w.push_u(u64::from(self.m_n4), 5);
        w.push_ism(i64::from(self.m_tau_gps), 22);
        w.push_flag(self.m_l_n_fifth);
        w.push_u(u64::from(self.reserved), 7);
        w.into_bytes()
    }
}
