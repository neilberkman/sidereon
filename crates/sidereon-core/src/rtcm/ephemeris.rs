//! RTCM 3 broadcast ephemeris messages 1019 (GPS) and 1020 (GLONASS).
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

use crate::error::{Error, Result};
use crate::id::{GnssSatelliteId, GnssSystem};

use super::bits::{BitReader, BitWriter};

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
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1019 {
            return Err(Error::Parse(format!(
                "message {message_number} is not GPS ephemeris 1019"
            )));
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
        let mut r = BitReader::new(body);
        let message_number = r.u(12)? as u16;
        if message_number != 1020 {
            return Err(Error::Parse(format!(
                "message {message_number} is not GLONASS ephemeris 1020"
            )));
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
