//! RINEX 3.x and 4.xx navigation-message parsing (GPS LNAV, Galileo I/NAV and
//! F/NAV, BeiDou D1/D2).
//!
//! Version 4 wraps each record in a `> EPH|STO|EOP|ION SVNN MSG` frame marker but
//! keeps the same fixed-column broadcast-orbit layout, so the two versions share
//! the block parser; only the record grouping differs. CNAV-family messages
//! (CNAV/CNV1/CNV2/CNV3) reorder the orbit roster and are recognized but not
//! parsed.
//!
//! Reads broadcast ephemeris records out of a RINEX navigation file into the
//! typed [`BroadcastRecord`]s the [`crate::broadcast`] evaluator consumes. This
//! is deterministic byte-to-record parsing of a fixed-column text format, not a
//! float recipe: there is no 0-ULP claim here, and a small in-house parser is
//! used in preference to a heavyweight RINEX dependency (the published `rinex`
//! crate pulls ~90 transitive crates, including computational-geometry stacks,
//! for what is a fixed-width text read).
//!
//! Scope: the GPS, Galileo, and BeiDou Keplerian record layouts (eight lines:
//! the SV/epoch/clock line plus seven broadcast-orbit lines), plus the GLONASS
//! four-line state-vector layout (parsed by [`parse_glonass`] and evaluated by
//! the [`crate::glonass`] RK4 propagator, not the Keplerian path). Other
//! constellations' records (SBAS, QZSS) are recognized as record boundaries and
//! skipped, so a mixed file parses without error but yields only the supported
//! systems.

mod store;
pub use store::BroadcastStore;

mod write;
pub use write::encode_nav;

use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::astro::time::{civil, gnss};
use crate::broadcast::{ClockPolynomial, ConstellationConstants, KeplerianElements};
use crate::constants::KM_TO_M;
use crate::format::columns::{field, raw_field};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::ionex::GalileoNequickCoeffs;
use crate::validate::{self, FieldError};

/// Parse a fixed-column RINEX broadcast-orbit numeric field, accepting Fortran
/// `D`/`d` exponents. `None` for a missing, blank, or malformed field. The field
/// label matches the lenient numeric reader the RINEX family shares, so the
/// accepted/rejected forms are identical across the readers.
fn parse_f64(line: &str, start: usize, end: usize) -> Option<f64> {
    let value = crate::format::columns::fortran_f64(line, start, end, "numeric field")?;
    // The fixed-width `D19.12` serializer field cannot hold a three-digit
    // exponent, so a value outside that range is not representable in this format.
    // Treat it as absent (the lenient `None` the readers already use for a
    // malformed field) so the parse/encode domains agree: a required field then
    // surfaces as a parse error, an optional one as absent. Real broadcast values
    // have small exponents and are unaffected.
    write::d19_12_representable(value).then_some(value)
}

/// Fallback half-window (seconds, either side of `toe`) for a record that does
/// not broadcast a fit interval (Galileo, BeiDou). A coarse validity guard - a
/// stale or wrong-week product is off by at least a week, so this rejects it as
/// "no ephemeris" rather than silently extrapolating. GPS records carry an
/// explicit curve-fit interval (see [`BroadcastRecord::fit_interval_s`]) and use
/// half of that instead.
pub(crate) const MAX_EPHEMERIS_AGE_S: f64 = 4.0 * 3600.0;

/// GLONASS broadcast records are valid +/-15 minutes around their reference
/// epoch (the nominal half-hour upload cadence), so a query farther than this
/// reports no ephemeris rather than extrapolating the RK4 integration.
pub(crate) const GLONASS_MAX_AGE_S: f64 = 15.0 * 60.0;
const GPS_NOMINAL_FIT_INTERVAL_S: f64 = 4.0 * 3600.0;
const GPS_LEGACY_EXTENDED_FIT_INTERVAL_S: f64 = 8.0 * 3600.0;
const GLONASS_FREQ_CHANNEL_MIN: i32 = -7;
const GLONASS_FREQ_CHANNEL_MAX: i32 = 6;

pub(crate) fn valid_glonass_frequency_channel(channel: i32) -> bool {
    (GLONASS_FREQ_CHANNEL_MIN..=GLONASS_FREQ_CHANNEL_MAX).contains(&channel)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RinexVersion {
    major: u8,
    minor: u8,
}

impl RinexVersion {
    fn gps_fit_interval_uses_legacy_flag(self) -> bool {
        self.major == 3 && self.minor <= 2
    }
}

/// Which broadcast navigation message a record carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMessage {
    /// GPS legacy navigation message.
    GpsLnav,
    /// Galileo integrity navigation message (E1/E5b dual, E1 single-frequency).
    GalileoInav,
    /// Galileo F/NAV message (E5a).
    GalileoFnav,
    /// BeiDou D1 message (MEO/IGSO satellites).
    BeidouD1,
    /// BeiDou D2 message (geostationary satellites).
    BeidouD2,
}

/// A broadcast group-delay term carried by a RINEX NAV record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastGroupDelayTerm {
    /// GPS LNAV TGD.
    GpsTgd,
    /// Galileo BGD E5a/E1.
    GalileoBgdE5aE1,
    /// Galileo BGD E5b/E1.
    GalileoBgdE5bE1,
    /// BeiDou TGD1.
    BeidouTgd1,
    /// BeiDou TGD2.
    BeidouTgd2,
}

/// Per-signal broadcast group delays preserved from one NAV record.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BroadcastGroupDelays {
    /// GPS LNAV TGD, seconds.
    pub gps_tgd_s: Option<f64>,
    /// Galileo BGD E5a/E1, seconds.
    pub galileo_bgd_e5a_e1_s: Option<f64>,
    /// Galileo BGD E5b/E1, seconds.
    pub galileo_bgd_e5b_e1_s: Option<f64>,
    /// BeiDou TGD1, seconds.
    pub beidou_tgd1_s: Option<f64>,
    /// BeiDou TGD2, seconds.
    pub beidou_tgd2_s: Option<f64>,
}

impl BroadcastGroupDelays {
    /// Build the GPS LNAV delay set.
    pub const fn gps_lnav(tgd_s: f64) -> Self {
        Self {
            gps_tgd_s: Some(tgd_s),
            galileo_bgd_e5a_e1_s: None,
            galileo_bgd_e5b_e1_s: None,
            beidou_tgd1_s: None,
            beidou_tgd2_s: None,
        }
    }

    /// Build the Galileo delay set.
    pub const fn galileo(bgd_e5a_e1_s: f64, bgd_e5b_e1_s: f64) -> Self {
        Self {
            gps_tgd_s: None,
            galileo_bgd_e5a_e1_s: Some(bgd_e5a_e1_s),
            galileo_bgd_e5b_e1_s: Some(bgd_e5b_e1_s),
            beidou_tgd1_s: None,
            beidou_tgd2_s: None,
        }
    }

    /// Build the BeiDou delay set.
    pub const fn beidou(tgd1_s: f64, tgd2_s: f64) -> Self {
        Self {
            gps_tgd_s: None,
            galileo_bgd_e5a_e1_s: None,
            galileo_bgd_e5b_e1_s: None,
            beidou_tgd1_s: Some(tgd1_s),
            beidou_tgd2_s: Some(tgd2_s),
        }
    }

    /// Select a specific group-delay term.
    pub const fn get(&self, term: BroadcastGroupDelayTerm) -> Option<f64> {
        match term {
            BroadcastGroupDelayTerm::GpsTgd => self.gps_tgd_s,
            BroadcastGroupDelayTerm::GalileoBgdE5aE1 => self.galileo_bgd_e5a_e1_s,
            BroadcastGroupDelayTerm::GalileoBgdE5bE1 => self.galileo_bgd_e5b_e1_s,
            BroadcastGroupDelayTerm::BeidouTgd1 => self.beidou_tgd1_s,
            BroadcastGroupDelayTerm::BeidouTgd2 => self.beidou_tgd2_s,
        }
    }

    /// The delay term historically used for broadcast-clock evaluation.
    ///
    /// BeiDou has no signal choice at this store level, so it keeps the previous
    /// TGD1 behavior. Callers that know their signal should use [`Self::get`].
    pub const fn for_message(self, system: GnssSystem, message: NavMessage) -> Option<f64> {
        match (system, message) {
            (GnssSystem::Gps, NavMessage::GpsLnav) => self.get(BroadcastGroupDelayTerm::GpsTgd),
            (GnssSystem::Galileo, NavMessage::GalileoFnav) => {
                self.get(BroadcastGroupDelayTerm::GalileoBgdE5aE1)
            }
            (GnssSystem::Galileo, NavMessage::GalileoInav) => {
                self.get(BroadcastGroupDelayTerm::GalileoBgdE5bE1)
            }
            (GnssSystem::BeiDou, NavMessage::BeidouD1 | NavMessage::BeidouD2) => {
                self.get(BroadcastGroupDelayTerm::BeidouTgd1)
            }
            _ => None,
        }
    }
}

/// Whether a BeiDou PRN is a geostationary satellite (BDS-2 C01-C05, BDS-3
/// C59-C61), which take the geostationary orbit-evaluation branch.
pub fn is_beidou_geo(sat: GnssSatelliteId) -> bool {
    sat.system == GnssSystem::BeiDou && (sat.prn <= 5 || (59..=61).contains(&sat.prn))
}

/// A Klobuchar-8 broadcast ionosphere coefficient set (the eight alpha/beta
/// values transmitted by GPS and BeiDou; the same model serves both, evaluated
/// per carrier - see [`crate::ionex::klobuchar_native`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KlobucharAlphaBeta {
    /// Cosine-amplitude polynomial coefficients (a0..a3).
    pub alpha: [f64; 4],
    /// Period polynomial coefficients (b0..b3).
    pub beta: [f64; 4],
}

/// Broadcast ionosphere-correction coefficients from a RINEX header's
/// `IONOSPHERIC CORR` lines or RINEX 4 body `> ION` frames.
///
/// Captures the Klobuchar-8 sets used by GPS (`GPSA`/`GPSB`) and BeiDou
/// (`BDSA`/`BDSB`), plus Galileo's three NeQuick-G effective-ionisation
/// coefficients (`GAL`). QZSS and NavIC Klobuchar sets are not retained.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IonoCorrections {
    /// GPS broadcast Klobuchar coefficients (`GPSA`/`GPSB`), if present.
    pub gps: Option<KlobucharAlphaBeta>,
    /// BeiDou broadcast Klobuchar coefficients (`BDSA`/`BDSB`), if present.
    pub beidou: Option<KlobucharAlphaBeta>,
    /// Galileo broadcast NeQuick-G coefficients (`GAL`), if present.
    pub galileo: Option<GalileoNequickCoeffs>,
}

/// One parsed GLONASS broadcast record: a PZ-90.11 ECEF state vector and the
/// clock terms, evaluated by the crate's GLONASS RK4 propagator (GLONASS is not
/// Keplerian, so it does not use [`BroadcastRecord`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlonassRecord {
    /// The transmitting satellite.
    pub satellite_id: GnssSatelliteId,
    /// Reference epoch as seconds past J2000 in **UTC** (leap-second-independent;
    /// the store adds the GPS−UTC offset to compare with the GPST-aligned query).
    pub toe_utc_j2000_s: f64,
    /// PZ-90.11 ECEF position at the reference epoch (meters).
    pub pos_m: [f64; 3],
    /// PZ-90.11 ECEF velocity at the reference epoch (meters/second).
    pub vel_m_s: [f64; 3],
    /// Lunisolar acceleration at the reference epoch (meters/second^2).
    pub acc_m_s2: [f64; 3],
    /// Clock bias broadcast field (−TauN, seconds).
    pub clk_bias: f64,
    /// Relative frequency offset (+GammaN, dimensionless).
    pub gamma_n: f64,
    /// Satellite health (0 is healthy).
    pub sv_health: f64,
    /// FDMA frequency-channel number.
    pub freq_channel: i32,
}

/// A GLONASS record skipped by [`parse_glonass_lenient`] because its slot is not
/// representable as a [`GnssSatelliteId`] (an extended slot beyond the engine's
/// PRN cap, e.g. `R28` in real BKG/IGS products).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedGlonass {
    /// The 3-character satellite token as it appeared in the file (`R28`).
    pub token: String,
}

/// The result of a lenient GLONASS parse: the representable records plus the
/// slot tokens that were skipped.
///
/// Mirrors the partial-success reporting used elsewhere for unrepresentable
/// input (`RinexObs::skipped_records`, [`crate::constellation::Catalog`]): a
/// dropped record carries its identity rather than vanishing silently, so a
/// caller can surface how many / which slots were skipped.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GlonassParse {
    /// Records for representable slots, in file order.
    pub records: Vec<GlonassRecord>,
    /// Slots that could not be represented and were skipped, in file order.
    pub skipped: Vec<SkippedGlonass>,
}

/// One parsed broadcast navigation record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastRecord {
    /// The transmitting satellite.
    pub satellite_id: GnssSatelliteId,
    /// The navigation message the record carries.
    pub message: NavMessage,
    /// Native broadcast week number (from the broadcast record).
    pub week: u32,
    /// Scale-tagged ephemeris reference time (`toe`).
    pub toe: GnssWeekTow,
    /// Scale-tagged clock reference time (`toc`).
    pub toc: GnssWeekTow,
    /// Keplerian orbital elements (`toe_sow` is seconds of week).
    pub elements: KeplerianElements,
    /// Clock polynomial (`toc_sow` is the record's own epoch, seconds of week).
    pub clock: ClockPolynomial,
    /// Broadcast group-delay terms carried by this message.
    pub group_delays: BroadcastGroupDelays,
    /// Satellite health word (0 is healthy for the GPS/Galileo nominal case).
    pub sv_health: f64,
    /// Signal-in-space accuracy: GPS URA (m) / Galileo SISA (m).
    pub sv_accuracy_m: f64,
    /// GPS curve-fit interval in seconds, centered on `toe` (IS-GPS-200): the
    /// record is valid for `toe ± fit_interval_s / 2`. `None` for Galileo and
    /// BeiDou, which do not broadcast a fit interval in the RINEX record; those
    /// fall back to the crate's nominal four-hour age bound.
    pub fit_interval_s: Option<f64>,
}

impl BroadcastRecord {
    /// Native time scale used by this record's `toe`/`toc`.
    pub const fn time_scale(&self) -> TimeScale {
        self.toe.system
    }

    /// The per-constellation constants this record evaluates with.
    pub const fn constants(&self) -> ConstellationConstants {
        match self.satellite_id.system {
            GnssSystem::Galileo => ConstellationConstants::GALILEO,
            GnssSystem::BeiDou => ConstellationConstants::BEIDOU,
            // GPS (and any other Keplerian system) use the GPS constants.
            _ => ConstellationConstants::GPS,
        }
    }

    /// Group delay used by the broadcast-clock evaluator for this message.
    pub fn broadcast_clock_group_delay_s(&self) -> f64 {
        self.group_delays
            .for_message(self.satellite_id.system, self.message)
            .unwrap_or(0.0)
    }

    /// Build a GPS LNAV record from decoded navigation-message subframes.
    ///
    /// This closes the `lnav::decode -> broadcast source` half of the real-time
    /// pipeline: feed [`crate::navigation::lnav::decode`]'s output here, collect
    /// the records into a [`BroadcastStore`], and solve with
    /// [`solve_broadcast`](crate::positioning::solve_broadcast). The conversion
    /// matches the RINEX navigation parser's record exactly except for the inputs
    /// only the air interface carries:
    ///
    /// - The decoded angular elements are in semicircles (and semicircles/second)
    ///   as transmitted by GPS LNAV; they are scaled to the radians the
    ///   [`crate::broadcast`] evaluator expects (the harmonic `cuc..cis` terms are
    ///   already radians and `crc`/`crs` meters, so they pass through unchanged).
    /// - The 10-bit transmitted week number is ambiguous across the GPS
    ///   1024-week rollover, so the full (unrolled) week is taken from
    ///   `full_week` rather than inferred from the message. The caller-supplied
    ///   `full_week` must agree with the decoded 10-bit week
    ///   (`full_week % 1024 == decoded.week_number`); a disagreement means the
    ///   caller is unrolling against the wrong rollover epoch and is rejected with
    ///   [`LnavRecordError::WeekMismatch`] rather than silently dating the
    ///   ephemeris to the wrong GPS week.
    /// - The fit interval is derived from the fit-interval flag together with
    ///   IODE/IODC per IS-GPS-200N 20.3.3.4.3.1 and Table 20-XII (the table the
    ///   older revisions numbered 20-XI): `flag = 0` is the nominal 4-hour curve
    ///   fit; `flag = 1` is an extended fit whose length is set by IODE/IODC
    ///   (short-term extended `IODE < 240` is 6 hours; long-term extended
    ///   `IODE` in `240..=255` is 8/14/26 hours by IODC range). Reserved IODC
    ///   combinations are rejected with [`LnavRecordError::FitIntervalUnsupported`].
    /// - The 4-bit URA index maps to its IS-GPS-200N 20.3.3.3.1.3 meters value;
    ///   index 15 (no accuracy prediction / not to be used) carries no usable
    ///   bound and is rejected with [`LnavRecordError::NoUraPrediction`].
    ///
    /// LNAV is the GPS L1 C/A message, so a non-GPS `satellite_id` is rejected.
    pub fn from_lnav(
        decoded: &crate::navigation::lnav::LnavDecoded,
        satellite_id: GnssSatelliteId,
        full_week: u32,
    ) -> Result<Self, LnavRecordError> {
        if satellite_id.system != GnssSystem::Gps {
            return Err(LnavRecordError::NotGps(satellite_id));
        }

        // The unrolled `full_week` must reduce to the decoded 10-bit week
        // (IS-GPS-200N 20.3.3.3.1.1). A mismatch means the caller unrolled
        // against the wrong rollover epoch; trusting `full_week` would date the
        // ephemeris to the wrong GPS week, so reject it.
        if i64::from(full_week % 1024) != decoded.week_number {
            return Err(LnavRecordError::WeekMismatch {
                full_week,
                decoded_week: decoded.week_number,
            });
        }

        let sv_accuracy_m = gps_ura_index_to_meters(decoded.ura_index)
            .ok_or(LnavRecordError::NoUraPrediction(decoded.ura_index))?;
        let fit_interval_s =
            gps_fit_interval_from_flag(decoded.fit_interval_flag, decoded.iode, decoded.iodc)?;

        // GPS LNAV transmits the angular ephemeris elements in semicircles and
        // semicircles/second; the Keplerian evaluator works in radians.
        const SEMICIRCLE_TO_RAD: f64 = core::f64::consts::PI;

        let elements = KeplerianElements {
            sqrt_a: decoded.sqrt_a,
            e: decoded.eccentricity,
            m0: decoded.m0 * SEMICIRCLE_TO_RAD,
            delta_n: decoded.delta_n * SEMICIRCLE_TO_RAD,
            omega0: decoded.omega0 * SEMICIRCLE_TO_RAD,
            i0: decoded.i0 * SEMICIRCLE_TO_RAD,
            omega: decoded.omega * SEMICIRCLE_TO_RAD,
            omega_dot: decoded.omega_dot * SEMICIRCLE_TO_RAD,
            idot: decoded.idot * SEMICIRCLE_TO_RAD,
            cuc: decoded.cuc,
            cus: decoded.cus,
            crc: decoded.crc,
            crs: decoded.crs,
            cic: decoded.cic,
            cis: decoded.cis,
            toe_sow: decoded.toe as f64,
        };
        let clock = ClockPolynomial {
            af0: decoded.af0,
            af1: decoded.af1,
            af2: decoded.af2,
            toc_sow: decoded.toc as f64,
        };

        let toe = GnssWeekTow::new(TimeScale::Gpst, full_week, elements.toe_sow)
            .and_then(GnssWeekTow::normalized)
            .map_err(|_| LnavRecordError::InvalidEpoch("toe"))?;
        let toc = GnssWeekTow::new(TimeScale::Gpst, full_week, clock.toc_sow)
            .and_then(GnssWeekTow::normalized)
            .map_err(|_| LnavRecordError::InvalidEpoch("toc"))?;

        Ok(BroadcastRecord {
            satellite_id,
            message: NavMessage::GpsLnav,
            week: full_week,
            toe,
            toc,
            elements,
            clock,
            group_delays: BroadcastGroupDelays::gps_lnav(decoded.tgd),
            sv_health: decoded.sv_health as f64,
            sv_accuracy_m,
            fit_interval_s: Some(fit_interval_s),
        })
    }
}

/// The nominal GPS user range accuracy (URA) value in meters for a 4-bit URA
/// index N (IS-GPS-200N Section 20.3.3.3.1.3). Each value is the upper bound of
/// the URA band the index represents. Index 15 carries no accuracy prediction
/// (the SV is not to be used for safe navigation) and has no usable meters
/// bound, so it returns `None` rather than a fabricated finite value.
fn gps_ura_index_to_meters(index: i64) -> Option<f64> {
    let meters = match index {
        0 => 2.4,
        1 => 3.4,
        2 => 4.85,
        3 => 6.85,
        4 => 9.65,
        5 => 13.65,
        6 => 24.0,
        7 => 48.0,
        8 => 96.0,
        9 => 192.0,
        10 => 384.0,
        11 => 768.0,
        12 => 1536.0,
        13 => 3072.0,
        14 => 6144.0,
        // 15 = no accuracy prediction / not to be used; anything outside the
        // 4-bit range cannot occur from a decoded message either.
        _ => return None,
    };
    Some(meters)
}

const GPS_FIT_INTERVAL_6H_S: f64 = 6.0 * 3600.0;
const GPS_FIT_INTERVAL_8H_S: f64 = 8.0 * 3600.0;
const GPS_FIT_INTERVAL_14H_S: f64 = 14.0 * 3600.0;
const GPS_FIT_INTERVAL_26H_S: f64 = 26.0 * 3600.0;

/// Curve-fit interval (seconds) for a GPS LNAV record from its fit-interval flag
/// plus IODE/IODC, per IS-GPS-200N 20.3.3.4.3.1, 6.2.3, and Table 20-XII (the
/// table older revisions numbered 20-XI).
///
/// `flag = 0` is the nominal 4-hour fit. `flag = 1` is an extended fit: IODE
/// selects short-term extended operations (`IODE < 240`, a 6-hour fit) from
/// long-term extended operations (`IODE` in `240..=255`), and for the long-term
/// case the IODC range selects 8, 14, or 26 hours. Reserved IODC values and any
/// other flag/IODE/IODC combination are rejected.
fn gps_fit_interval_from_flag(
    fit_interval_flag: i64,
    iode: i64,
    iodc: i64,
) -> Result<f64, LnavRecordError> {
    let unsupported = || LnavRecordError::FitIntervalUnsupported {
        fit_interval_flag,
        iode,
        iodc,
    };
    match fit_interval_flag {
        0 => Ok(GPS_NOMINAL_FIT_INTERVAL_S),
        1 => {
            if (0..240).contains(&iode) {
                // Short-term extended operations (Table 20-XII, 2-14 day row).
                // IODE is an 8-bit unsigned field, so a negative value is not a
                // real decode and falls through to the unsupported error.
                Ok(GPS_FIT_INTERVAL_6H_S)
            } else if (240..=255).contains(&iode) {
                // Long-term extended operations: IODC selects the fit length.
                match iodc {
                    240..=247 => Ok(GPS_FIT_INTERVAL_8H_S),
                    248..=255 | 496 => Ok(GPS_FIT_INTERVAL_14H_S),
                    497..=503 | 1021..=1023 => Ok(GPS_FIT_INTERVAL_26H_S),
                    _ => Err(unsupported()),
                }
            } else {
                Err(unsupported())
            }
        }
        _ => Err(unsupported()),
    }
}

/// Failure building a [`BroadcastRecord`] from decoded LNAV subframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LnavRecordError {
    /// LNAV is the GPS L1 C/A message; the satellite is not a GPS satellite.
    NotGps(GnssSatelliteId),
    /// A derived week/time-of-week value was not representable.
    InvalidEpoch(&'static str),
    /// The caller-supplied `full_week` does not reduce to the decoded 10-bit week
    /// (`full_week % 1024 != decoded_week`), so it unrolls to the wrong GPS week.
    WeekMismatch {
        /// The caller-supplied unrolled week.
        full_week: u32,
        /// The 10-bit week decoded from the message.
        decoded_week: i64,
    },
    /// URA index 15 (or an out-of-range index) carries no accuracy prediction.
    NoUraPrediction(i64),
    /// The fit-interval flag / IODE / IODC combination is reserved or otherwise
    /// not a defined IS-GPS-200N Table 20-XII curve-fit interval.
    FitIntervalUnsupported {
        /// The 1-bit fit-interval flag from the message.
        fit_interval_flag: i64,
        /// The decoded IODE.
        iode: i64,
        /// The decoded IODC.
        iodc: i64,
    },
}

impl core::fmt::Display for LnavRecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LnavRecordError::NotGps(sat) => {
                write!(f, "LNAV is a GPS message; {sat} is not a GPS satellite")
            }
            LnavRecordError::InvalidEpoch(field) => {
                write!(f, "derived {field} week/TOW is not representable")
            }
            LnavRecordError::WeekMismatch {
                full_week,
                decoded_week,
            } => write!(
                f,
                "full_week {full_week} (week % 1024 = {}) disagrees with decoded 10-bit week {decoded_week}",
                full_week % 1024
            ),
            LnavRecordError::NoUraPrediction(index) => {
                write!(f, "URA index {index} carries no accuracy prediction")
            }
            LnavRecordError::FitIntervalUnsupported {
                fit_interval_flag,
                iode,
                iodc,
            } => write!(
                f,
                "fit interval flag {fit_interval_flag} with IODE {iode} / IODC {iodc} is not a defined curve-fit interval"
            ),
        }
    }
}

impl std::error::Error for LnavRecordError {}

fn broadcast_time_scale(system: GnssSystem) -> TimeScale {
    match system {
        GnssSystem::Galileo => TimeScale::Gst,
        GnssSystem::BeiDou => TimeScale::Bdt,
        _ => TimeScale::Gpst,
    }
}

/// Why a RINEX NAV file could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavParseError {
    /// The header did not declare a supported RINEX 3 navigation file.
    UnsupportedHeader(String),
    /// No `END OF HEADER` line was found.
    MissingHeaderEnd,
    /// A record was shorter than its message layout requires.
    TruncatedRecord(String),
    /// A required numeric field was missing or unparseable.
    BadField {
        /// The satellite whose record holds the bad field.
        satellite: String,
        /// Which field failed.
        field: &'static str,
    },
    /// A required header numeric field was malformed, non-finite, or out of range.
    BadHeaderField {
        /// Which header field failed.
        field: &'static str,
    },
}

impl core::fmt::Display for NavParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NavParseError::UnsupportedHeader(s) => write!(f, "unsupported RINEX NAV header: {s}"),
            NavParseError::MissingHeaderEnd => write!(f, "no END OF HEADER line"),
            NavParseError::TruncatedRecord(s) => write!(f, "truncated navigation record for {s}"),
            NavParseError::BadField { satellite, field } => {
                write!(f, "bad/missing {field} field in record for {satellite}")
            }
            NavParseError::BadHeaderField { field } => {
                write!(f, "bad/missing {field} field in navigation header")
            }
        }
    }
}

impl std::error::Error for NavParseError {}

/// Parse a RINEX 3.x or 4.xx navigation file into the supported (GPS, Galileo,
/// BeiDou) Keplerian records.
///
/// Records of other constellations (GLONASS state-vector, SBAS) are skipped, as
/// are version-4 CNAV-family messages (CNAV/CNV1/CNV2/CNV3): their broadcast-orbit
/// roster reorders the fixed columns (`t_op` for `toe`, `wn_op` for `week`, extra
/// `adot`/`deltaN0Dot` terms), so they are recognized but not parsed rather than
/// fed wrong values. The records are returned in file order; selection (by epoch,
/// health, message type) is the caller's job.
pub fn parse_nav(text: &str) -> Result<Vec<BroadcastRecord>, NavParseError> {
    let mut lines = text.lines();
    let version = verify_and_skip_header(&mut lines)?;
    if version.major >= 4 {
        parse_nav_v4(lines, version)
    } else {
        parse_nav_v3(lines, version)
    }
}

/// Version-3 body: a record starts at a line whose first three columns are a
/// system letter followed by two digits; continuation lines are column-indented.
fn parse_nav_v3<'a, I>(
    lines: I,
    version: RinexVersion,
) -> Result<Vec<BroadcastRecord>, NavParseError>
where
    I: Iterator<Item = &'a str>,
{
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for line in lines {
        if is_record_start(line) {
            blocks.push(vec![line]);
        } else if let Some(last) = blocks.last_mut() {
            last.push(line);
        }
    }

    let mut records = Vec::new();
    for block in &blocks {
        let letter = block[0].as_bytes()[0] as char;
        match GnssSystem::from_letter(letter) {
            Some(GnssSystem::Gps) | Some(GnssSystem::Galileo) | Some(GnssSystem::BeiDou) => {
                records.push(parse_keplerian_block(block, None, version)?);
            }
            // Recognized boundary, unsupported model (GLONASS state-vector, SBAS): skip.
            _ => {}
        }
    }
    Ok(records)
}

/// Version-4 body: each record is introduced by a `> EPH|STO|EOP|ION SVNN MSG`
/// frame marker. Only `EPH` frames carrying a supported Keplerian message are
/// parsed; the broadcast-orbit lines that follow the marker have the same
/// fixed-column layout as version 3, so they reuse [`parse_keplerian_block`].
/// The message type is taken from the marker token (so I/NAV vs F/NAV and D1 vs
/// D2 are explicit, not inferred) after the marker SV and message family are
/// cross-checked against the body line. STO/EOP/ION frames, other
/// constellations, and CNAV-family messages are skipped.
fn parse_nav_v4<'a, I>(
    lines: I,
    version: RinexVersion,
) -> Result<Vec<BroadcastRecord>, NavParseError>
where
    I: Iterator<Item = &'a str>,
{
    // Group by marker line: each frame is its marker plus the body lines up to
    // the next marker.
    let frames = v4_frames(lines);
    let mut records = Vec::new();
    for (marker, body) in &frames {
        let Some((frame_type, sv, msg_token)) = parse_v4_marker(marker) else {
            continue;
        };
        if frame_type != "EPH" {
            continue; // STO/EOP/ION carry no ephemeris.
        }
        let letter = sv.as_bytes().first().copied().map_or(' ', char::from);
        let supported = matches!(
            GnssSystem::from_letter(letter),
            Some(GnssSystem::Gps) | Some(GnssSystem::Galileo) | Some(GnssSystem::BeiDou)
        );
        if !supported {
            continue; // GLONASS/SBAS/QZSS/NavIC: not a supported Keplerian system.
        }
        // Only messages whose orbit roster matches the version-3 column layout
        // are parsed; CNAV-family (and anything unrecognized) is skipped.
        if let Some(message) = nav_message_from_v4_token(msg_token) {
            validate_v4_ephemeris_marker(sv, message, body)?;
            records.push(parse_keplerian_block(body, Some(message), version)?);
        }
    }
    Ok(records)
}

fn v4_frames<'a, I>(lines: I) -> Vec<(&'a str, Vec<&'a str>)>
where
    I: Iterator<Item = &'a str>,
{
    let mut frames: Vec<(&str, Vec<&str>)> = Vec::new();
    for line in lines {
        if is_v4_frame_marker(line) {
            frames.push((line, Vec::new()));
        } else if let Some((_, body)) = frames.last_mut() {
            body.push(line);
        }
    }
    frames
}

/// Whether a version-4 line is a frame marker (`> ...`).
fn is_v4_frame_marker(line: &str) -> bool {
    line.starts_with("> ")
}

/// Split a version-4 frame marker `> EPH G01 LNAV` into (frame type, SV, message
/// token), or `None` if it is malformed. Mirrors the RINEX-4 marker layout:
/// `>` then the 4-column frame class, the SV, and the message-type token.
fn parse_v4_marker(line: &str) -> Option<(&str, &str, &str)> {
    let rest = line.strip_prefix('>')?;
    let mut fields = rest.split_whitespace();
    let frame_type = fields.next()?;
    let sv = fields.next()?;
    let msg_token = fields.next()?;
    Some((frame_type, sv, msg_token))
}

/// Map a version-4 EPH message token to the [`NavMessage`] for the supported
/// Keplerian messages, or `None` for a message whose orbit layout does not match
/// the version-3 columns (CNAV-family) or is otherwise unsupported here.
fn nav_message_from_v4_token(token: &str) -> Option<NavMessage> {
    match token {
        "LNAV" => Some(NavMessage::GpsLnav),
        "INAV" => Some(NavMessage::GalileoInav),
        "FNAV" => Some(NavMessage::GalileoFnav),
        "D1" => Some(NavMessage::BeidouD1),
        "D2" => Some(NavMessage::BeidouD2),
        _ => None,
    }
}

fn validate_v4_ephemeris_marker(
    marker_sv: &str,
    message: NavMessage,
    body: &[&str],
) -> Result<(), NavParseError> {
    let Some(body_sv) = body
        .first()
        .and_then(|line| line.get(0..3))
        .map(str::trim)
        .filter(|sv| !sv.is_empty())
    else {
        return Ok(());
    };

    if marker_sv != body_sv {
        return Err(NavParseError::BadField {
            satellite: marker_sv.to_string(),
            field: "frame marker",
        });
    }

    let system = body_sv
        .as_bytes()
        .first()
        .and_then(|b| GnssSystem::from_letter(*b as char))
        .ok_or_else(|| NavParseError::BadField {
            satellite: body_sv.to_string(),
            field: "system",
        })?;
    if !nav_message_matches_system(message, system) {
        return Err(NavParseError::BadField {
            satellite: body_sv.to_string(),
            field: "message",
        });
    }

    Ok(())
}

fn nav_message_matches_system(message: NavMessage, system: GnssSystem) -> bool {
    matches!(
        (message, system),
        (NavMessage::GpsLnav, GnssSystem::Gps)
            | (
                NavMessage::GalileoInav | NavMessage::GalileoFnav,
                GnssSystem::Galileo,
            )
            | (
                NavMessage::BeidouD1 | NavMessage::BeidouD2,
                GnssSystem::BeiDou,
            )
    )
}

/// Parse the broadcast ionosphere coefficients from a RINEX header's
/// `IONOSPHERIC CORR` lines or RINEX 4 body `> ION` frames (GPS
/// `GPSA`/`GPSB`, BeiDou `BDSA`/`BDSB`, and Galileo `GAL`).
///
/// A complete header label pair or body frame yields the coefficient set; a
/// missing label or frame yields `None` for that system. Parsing is
/// deterministic text, not a 0-ULP target.
pub fn parse_iono_corrections(text: &str) -> Result<IonoCorrections, NavParseError> {
    parse_iono_corrections_checked(text)
}

fn parse_iono_corrections_checked(text: &str) -> Result<IonoCorrections, NavParseError> {
    // The IONOSPHERIC CORR line is `A4,1X,4(D12.4)`: a 4-char label, a space,
    // then up to four coefficients in 12-wide columns.
    //
    // GPS/BeiDou are Klobuchar models with four coefficients per row
    // (alpha0..alpha3 / beta0..beta3); all four columns are required and a
    // truncated row is a malformed header, not a tolerable short line.
    let klobuchar_row = |line: &str| -> Result<[f64; 4], NavParseError> {
        Ok([
            strict_header_f64(line, 5, 17, "ionospheric correction")?,
            strict_header_f64(line, 17, 29, "ionospheric correction")?,
            strict_header_f64(line, 29, 41, "ionospheric correction")?,
            strict_header_f64(line, 41, 53, "ionospheric correction")?,
        ])
    };
    // Galileo is NeQuick-G with three coefficients (ai0,ai1,ai2). The fourth
    // column is the disturbance flag, which real/merged headers frequently leave
    // blank; only the three coefficients are read, so the row parses whether or
    // not that flag is present.
    let nequick_row = |line: &str| -> Result<[f64; 3], NavParseError> {
        Ok([
            strict_header_f64(line, 5, 17, "ionospheric correction")?,
            strict_header_f64(line, 17, 29, "ionospheric correction")?,
            strict_header_f64(line, 29, 41, "ionospheric correction")?,
        ])
    };
    let (mut gpsa, mut gpsb, mut bdsa, mut bdsb, mut gal) = (None, None, None, None, None);
    for line in text.lines() {
        if line.contains("END OF HEADER") {
            break;
        }
        if !line.contains("IONOSPHERIC CORR") {
            continue;
        }
        match line.get(0..4).map(str::trim) {
            Some("GPSA") => gpsa = Some(klobuchar_row(line)?),
            Some("GPSB") => gpsb = Some(klobuchar_row(line)?),
            Some("BDSA") => bdsa = Some(klobuchar_row(line)?),
            Some("BDSB") => bdsb = Some(klobuchar_row(line)?),
            Some("GAL") => {
                let row = nequick_row(line)?;
                gal = Some(GalileoNequickCoeffs {
                    ai0: row[0],
                    ai1: row[1],
                    ai2: row[2],
                });
            }
            _ => {}
        }
    }
    let pair = |a: Option<[f64; 4]>, b: Option<[f64; 4]>| match (a, b) {
        (Some(alpha), Some(beta)) => Some(KlobucharAlphaBeta { alpha, beta }),
        _ => None,
    };
    let mut iono = IonoCorrections {
        gps: pair(gpsa, gpsb),
        beidou: pair(bdsa, bdsb),
        galileo: gal,
    };
    parse_v4_body_iono_corrections(text, &mut iono)?;
    Ok(iono)
}

fn parse_v4_body_iono_corrections(
    text: &str,
    iono: &mut IonoCorrections,
) -> Result<(), NavParseError> {
    let mut lines = text.lines();
    for line in lines.by_ref() {
        if line.contains("END OF HEADER") {
            break;
        }
    }

    for (marker, body) in v4_frames(lines) {
        let Some((frame_type, sv, _msg_token)) = parse_v4_marker(marker) else {
            continue;
        };
        if frame_type != "ION" {
            continue;
        }
        let values = parse_v4_iono_values(sv, &body)?;
        match sv
            .as_bytes()
            .first()
            .and_then(|b| GnssSystem::from_letter(*b as char))
        {
            Some(GnssSystem::Gps) => {
                iono.gps = Some(KlobucharAlphaBeta {
                    alpha: iono_values_4(&values, 0, sv)?,
                    beta: iono_values_4(&values, 4, sv)?,
                });
            }
            Some(GnssSystem::BeiDou) => {
                iono.beidou = Some(KlobucharAlphaBeta {
                    alpha: iono_values_4(&values, 0, sv)?,
                    beta: iono_values_4(&values, 4, sv)?,
                });
            }
            Some(GnssSystem::Galileo) => {
                let coeffs = iono_values_3(&values, 0, sv)?;
                iono.galileo = Some(GalileoNequickCoeffs {
                    ai0: coeffs[0],
                    ai1: coeffs[1],
                    ai2: coeffs[2],
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_v4_iono_values(sv: &str, body: &[&str]) -> Result<Vec<f64>, NavParseError> {
    if body.is_empty() {
        return Err(NavParseError::BadField {
            satellite: sv.to_string(),
            field: "ionospheric correction",
        });
    }

    let mut values = Vec::new();
    for (idx, line) in body.iter().enumerate() {
        let ranges: &[(usize, usize)] = if idx == 0 {
            &[(23, 42), (42, 61), (61, 80)]
        } else {
            &[(4, 23), (23, 42), (42, 61), (61, 80)]
        };
        for &(start, end) in ranges {
            let raw = raw_field(line, start, end);
            if raw.trim().is_empty() {
                continue;
            }
            values.push(
                validate::strict_f64(raw, "ionospheric correction")
                    .map_err(|error| map_record_field_error(error, sv))?,
            );
        }
    }
    Ok(values)
}

fn iono_values_4(values: &[f64], start: usize, sv: &str) -> Result<[f64; 4], NavParseError> {
    let Some(slice) = values.get(start..start + 4) else {
        return Err(NavParseError::BadField {
            satellite: sv.to_string(),
            field: "ionospheric correction",
        });
    };
    Ok([slice[0], slice[1], slice[2], slice[3]])
}

fn iono_values_3(values: &[f64], start: usize, sv: &str) -> Result<[f64; 3], NavParseError> {
    let Some(slice) = values.get(start..start + 3) else {
        return Err(NavParseError::BadField {
            satellite: sv.to_string(),
            field: "ionospheric correction",
        });
    };
    Ok([slice[0], slice[1], slice[2]])
}

/// The leap-second count (GPS − UTC) from the header's `LEAP SECONDS` line, used
/// to map a GLONASS (UTC) reference epoch onto the GPST-aligned query time. The
/// value is the first field; `None` if the line is absent.
pub fn parse_leap_seconds(text: &str) -> Result<Option<f64>, NavParseError> {
    parse_leap_seconds_checked(text)
}

fn parse_leap_seconds_checked(text: &str) -> Result<Option<f64>, NavParseError> {
    for line in text.lines() {
        if line.contains("END OF HEADER") {
            break;
        }
        if line.contains("LEAP SECONDS") {
            return strict_header_integer_f64(line, 0, 6, "leap seconds").map(Some);
        }
    }
    Ok(None)
}

/// Seconds from the J2000 epoch (2000-01-01 12:00) to a UTC calendar instant,
/// via the canonical no-leap civil conversion. Bit-identical to the previous
/// day-count arithmetic (the Julian Day Number is offset-equal to the Hinnant
/// day count, and the whole-second clock fields sum exactly in `f64`).
fn j2000_seconds_utc(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> f64 {
    civil::j2000_seconds(y as i32, mo as i32, d as i32, h as i32, mi as i32, s as f64)
}

/// Parse the GLONASS epoch line (`Rnn YYYY MM DD HH MM SS`) to a UTC second past
/// J2000.
fn parse_glonass_epoch(l0: &str, sat: &str) -> Result<f64, NavParseError> {
    let year = strict_record_int::<i64>(l0, 4, 8, "epoch", sat)?;
    let month = strict_record_int::<i64>(l0, 9, 11, "epoch", sat)?;
    let day = strict_record_int::<i64>(l0, 12, 14, "epoch", sat)?;
    let hour = strict_record_int::<i64>(l0, 15, 17, "epoch", sat)?;
    let minute = strict_record_int::<i64>(l0, 18, 20, "epoch", sat)?;
    let second = strict_record_int::<i64>(l0, 21, 23, "epoch", sat)?;
    let civil = validate::civil_datetime_with_second_policy(
        year,
        month,
        day,
        hour,
        minute,
        second as f64,
        validate::CivilSecondPolicy::UtcLike,
    )
    .map_err(|_| NavParseError::BadField {
        satellite: sat.to_string(),
        field: "epoch",
    })?;
    Ok(j2000_seconds_utc(
        civil.year,
        i64::from(civil.month),
        i64::from(civil.day),
        i64::from(civil.hour),
        i64::from(civil.minute),
        civil.second as i64,
    ))
}

/// Parse a 4-line RINEX 3 GLONASS record block into a [`GlonassRecord`]
/// (km/(km/s)/(km/s^2) state converted to SI). A missing or unparseable field is
/// a [`NavParseError`], not a silently dropped record.
fn parse_glonass_block(block: &[&str]) -> Result<GlonassRecord, NavParseError> {
    let l0 = block[0];
    let sat = l0.get(0..3).unwrap_or("").trim().to_string();
    if block.len() < 4 {
        return Err(NavParseError::TruncatedRecord(sat));
    }
    let bad = |what: &'static str| NavParseError::BadField {
        satellite: sat.clone(),
        field: what,
    };
    let satellite_id: GnssSatelliteId = sat.parse().map_err(|_| bad("prn"))?;
    let toe_utc_j2000_s = parse_glonass_epoch(l0, &sat)?;
    let clk_bias = parse_f64(l0, 23, 42).ok_or_else(|| bad("clock bias"))?;
    let gamma_n = parse_f64(l0, 42, 61).ok_or_else(|| bad("gamma_n"))?;
    let o1 = orbit_row(block[1]);
    let o2 = orbit_row(block[2]);
    let o3 = orbit_row(block[3]);
    let km = |v: Option<f64>, what: &'static str| v.map(|x| x * KM_TO_M).ok_or_else(|| bad(what));
    let g = |v: Option<f64>, what: &'static str| v.ok_or_else(|| bad(what));
    Ok(GlonassRecord {
        satellite_id,
        toe_utc_j2000_s,
        pos_m: [km(o1[0], "x")?, km(o2[0], "y")?, km(o3[0], "z")?],
        vel_m_s: [km(o1[1], "vx")?, km(o2[1], "vy")?, km(o3[1], "vz")?],
        acc_m_s2: [km(o1[2], "ax")?, km(o2[2], "ay")?, km(o3[2], "az")?],
        clk_bias,
        gamma_n,
        sv_health: g(o1[3], "health")?,
        freq_channel: glonass_frequency_channel(g(o2[3], "frequency channel")?, &sat)?,
    })
}

/// Parse all GLONASS (`R`) records from a RINEX 3.x navigation file, in file
/// order; selection is the caller's job. A malformed *supported* record is a
/// [`NavParseError`] rather than a silently dropped one, but a record for a slot
/// the engine cannot represent (an extended GLONASS slot beyond the PRN cap, e.g.
/// `R28` in real BKG/IGS products) is skipped rather than rejecting the whole
/// file - the same treatment unsupported constellations get in
/// [`parse_nav_v3`]. (Version-4 GLONASS frames are not yet parsed.)
pub fn parse_glonass(text: &str) -> Result<Vec<GlonassRecord>, NavParseError> {
    Ok(parse_glonass_lenient(text)?.records)
}

/// Like [`parse_glonass`], but also returns the slots that were skipped because
/// they are not representable as a [`GnssSatelliteId`] (an extended slot beyond
/// the PRN cap, e.g. `R28`).
///
/// [`parse_glonass`] drops that list silently; use this when a caller needs to
/// surface how many / which records were skipped, consistent with the
/// lenient-skip reporting elsewhere in the crate. A malformed *representable*
/// record is still a [`NavParseError`], not a skip.
pub fn parse_glonass_lenient(text: &str) -> Result<GlonassParse, NavParseError> {
    let mut lines = text.lines();
    verify_and_skip_header(&mut lines)?;
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for line in lines {
        if is_record_start(line) {
            blocks.push(vec![line]);
        } else if let Some(last) = blocks.last_mut() {
            last.push(line);
        }
    }
    let mut out = GlonassParse::default();
    for block in blocks.iter().filter(|b| b[0].starts_with('R')) {
        // A GLONASS slot beyond the engine's PRN cap is not representable as a
        // `GnssSatelliteId`. Skip such a record (one out-of-range slot must not
        // discard every other satellite's ephemeris) instead of erroring, but
        // record its identity so it is not lost silently; a representable slot
        // with a malformed numeric field still errors.
        let sat = block[0].get(0..3).unwrap_or("").trim();
        if sat.parse::<GnssSatelliteId>().is_err() {
            out.skipped.push(SkippedGlonass {
                token: sat.to_string(),
            });
            continue;
        }
        out.records.push(parse_glonass_block(block)?);
    }
    Ok(out)
}

/// Skip the header, returning the RINEX version. Major versions 3 and 4 share
/// the fixed-column orbit layout; version 4 wraps each record in a frame marker
/// line (see [`parse_v4_marker`]), which is why `parse_nav` dispatches on it.
fn verify_and_skip_header<'a, I>(lines: &mut I) -> Result<RinexVersion, NavParseError>
where
    I: Iterator<Item = &'a str>,
{
    let mut version_seen: Option<RinexVersion> = None;
    for line in lines.by_ref() {
        if line.contains("RINEX VERSION / TYPE") {
            // Column 0-8 holds the version; column 20 the file type ('N' = NAV).
            let version = line.get(0..9).unwrap_or("").trim();
            let detected = parse_rinex_version(version);
            let is_nav = line.get(20..21) == Some("N");
            match (detected, is_nav) {
                (Some(v), true) => version_seen = Some(v),
                _ => {
                    return Err(NavParseError::UnsupportedHeader(
                        line.trim_end().to_string(),
                    ))
                }
            }
        }
        if line.contains("END OF HEADER") {
            return version_seen.ok_or_else(|| {
                NavParseError::UnsupportedHeader("no RINEX VERSION / TYPE".to_string())
            });
        }
    }
    Err(NavParseError::MissingHeaderEnd)
}

fn parse_rinex_version(version: &str) -> Option<RinexVersion> {
    let (major, minor) = version.split_once('.')?;
    let major = major.trim().parse::<u8>().ok()?;
    if !matches!(major, 3 | 4) {
        return None;
    }
    let minor_digits = minor
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    if minor_digits.is_empty() {
        return None;
    }
    let minor = minor_digits.parse::<u8>().ok()?;
    Some(RinexVersion { major, minor })
}

fn is_record_start(line: &str) -> bool {
    let b = line.as_bytes();
    b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1].is_ascii_digit() && b[2].is_ascii_digit()
}

/// The four broadcast-orbit values of a continuation line (columns 4/23/42/61).
fn orbit_row(line: &str) -> [Option<f64>; 4] {
    [
        parse_f64(line, 4, 23),
        parse_f64(line, 23, 42),
        parse_f64(line, 42, 61),
        parse_f64(line, 61, 80),
    ]
}

#[derive(Debug, Clone, Copy)]
struct ClockReferenceEpoch {
    week: u32,
    sow: f64,
}

fn parse_keplerian_block(
    block: &[&str],
    message_override: Option<NavMessage>,
    version: RinexVersion,
) -> Result<BroadcastRecord, NavParseError> {
    let l0 = block.first().copied().unwrap_or("");
    let sat = l0.get(0..3).unwrap_or("").trim().to_string();
    if block.len() < 8 {
        return Err(NavParseError::TruncatedRecord(sat));
    }
    let bad = |what: &'static str| NavParseError::BadField {
        satellite: sat.clone(),
        field: what,
    };

    let letter = l0
        .as_bytes()
        .first()
        .copied()
        .map(|b| b as char)
        .ok_or_else(|| bad("system"))?;
    let system = GnssSystem::from_letter(letter).ok_or_else(|| bad("system"))?;
    let satellite_id: GnssSatelliteId = sat.parse().map_err(|_| bad("prn"))?;

    // Clock line: epoch (-> toc) and the af0/af1/af2 polynomial.
    let time_scale = broadcast_time_scale(system);
    let toc_epoch = parse_toc(l0, &sat, time_scale)?;
    let toc_sow = toc_epoch.sow;
    let af0 = parse_f64(l0, 23, 42).ok_or_else(|| bad("af0"))?;
    let af1 = parse_f64(l0, 42, 61).ok_or_else(|| bad("af1"))?;
    let af2 = parse_f64(l0, 61, 80).ok_or_else(|| bad("af2"))?;

    let o1 = orbit_row(block[1]);
    let o2 = orbit_row(block[2]);
    let o3 = orbit_row(block[3]);
    let o4 = orbit_row(block[4]);
    let o5 = orbit_row(block[5]);
    let o6 = orbit_row(block[6]);

    let g = |v: Option<f64>, what: &'static str| v.ok_or_else(|| bad(what));

    let elements = KeplerianElements {
        crs: g(o1[1], "crs")?,
        delta_n: g(o1[2], "deltaN")?,
        m0: g(o1[3], "m0")?,
        cuc: g(o2[0], "cuc")?,
        e: g(o2[1], "e")?,
        cus: g(o2[2], "cus")?,
        sqrt_a: g(o2[3], "sqrtA")?,
        toe_sow: g(o3[0], "toe")?,
        cic: g(o3[1], "cic")?,
        omega0: g(o3[2], "omega0")?,
        cis: g(o3[3], "cis")?,
        i0: g(o4[0], "i0")?,
        crc: g(o4[1], "crc")?,
        omega: g(o4[2], "omega")?,
        omega_dot: g(o4[3], "omegaDot")?,
        idot: g(o5[0], "idot")?,
    };
    let clock = ClockPolynomial {
        af0,
        af1,
        af2,
        toc_sow,
    };

    let week = finite_integral_u32(g(o5[2], "week")?, "week", &sat)?;
    let toe = GnssWeekTow::new(time_scale, week, elements.toe_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("toe"))?;
    let toc = GnssWeekTow::new(time_scale, toc_epoch.week, clock.toc_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("toc"))?;
    let message = if let Some(message) = message_override {
        message
    } else {
        match system {
            GnssSystem::Galileo => galileo_message(g(o5[1], "data sources")?, &sat)?,
            GnssSystem::BeiDou => {
                if is_beidou_geo(satellite_id) {
                    NavMessage::BeidouD2
                } else {
                    NavMessage::BeidouD1
                }
            }
            _ => NavMessage::GpsLnav,
        }
    };

    let sv_accuracy_m = g(o6[0], "accuracy")?;
    let sv_health = g(o6[1], "health")?;
    let group_delays = match system {
        GnssSystem::Gps => BroadcastGroupDelays::gps_lnav(g(o6[2], "gps tgd")?),
        // RINEX Galileo ORBIT-6 carries BGD E5a/E1 in field 3 and BGD E5b/E1 in
        // field 4; both are part of the message representation regardless of
        // which one a clock consumer later selects.
        GnssSystem::Galileo => {
            BroadcastGroupDelays::galileo(g(o6[2], "bgd e5a/e1")?, g(o6[3], "bgd e5b/e1")?)
        }
        GnssSystem::BeiDou => {
            BroadcastGroupDelays::beidou(g(o6[2], "beidou tgd1")?, g(o6[3], "beidou tgd2")?)
        }
        _ => BroadcastGroupDelays::default(),
    };

    // Only GPS LNAV broadcasts a curve-fit interval (ORBIT-7 field 2); Galileo
    // and BeiDou leave that column blank or spare, so they carry no fit interval.
    let fit_interval_s = match system {
        GnssSystem::Gps => {
            Some(gps_fit_interval_s(block[7], version).map_err(|()| bad("fit interval"))?)
        }
        _ => None,
    };

    Ok(BroadcastRecord {
        satellite_id,
        message,
        week,
        toe,
        toc,
        elements,
        clock,
        group_delays,
        sv_health,
        sv_accuracy_m,
        fit_interval_s,
    })
}

/// The GPS curve-fit interval in seconds from the ORBIT-7 fit-interval field.
/// RINEX 3.03+ and 4.xx record this field in hours. Legacy RINEX 3.02 and older
/// files may carry the broadcast 0/1 fit-interval flag instead, where 1 means
/// more than four hours rather than one hour. Per IS-GPS-200 the decoded value
/// is the total interval centered on `toe`; a zero or absent field denotes the
/// nominal four hours.
///
/// A blank/absent field is the legitimate nominal case (some products omit it);
/// a present but non-numeric field is a malformed record, reported as `Err` so
/// the caller can raise the same `BadField` error as for other numeric fields
/// rather than silently substituting four hours.
fn gps_fit_interval_s(orbit7: &str, version: RinexVersion) -> Result<f64, ()> {
    let value = match field(orbit7, 23, 42) {
        None => 0.0,
        Some(_) => parse_f64(orbit7, 23, 42).ok_or(())?,
    };
    if value == 0.0 {
        Ok(GPS_NOMINAL_FIT_INTERVAL_S)
    } else if version.gps_fit_interval_uses_legacy_flag() && value == 1.0 {
        Ok(GPS_LEGACY_EXTENDED_FIT_INTERVAL_S)
    } else {
        Ok(value * 3600.0)
    }
}

/// Classify a Galileo record from its data-source word (orbit-5 field 1): source
/// bit 1 is F/NAV, source bits 0/2 are I/NAV. Bits 8/9 describe the clock-pair
/// frequency and do not determine the navigation message type.
fn galileo_message(data_sources: f64, sat: &str) -> Result<NavMessage, NavParseError> {
    let word = finite_integral_u32(data_sources, "data sources", sat)?;
    if word & 0b010 != 0 {
        Ok(NavMessage::GalileoFnav)
    } else if word & 0b101 != 0 {
        Ok(NavMessage::GalileoInav)
    } else {
        // No source bit set: default to I/NAV (the operational E1 message).
        Ok(NavMessage::GalileoInav)
    }
}

fn finite_integral_u32(value: f64, field: &'static str, sat: &str) -> Result<u32, NavParseError> {
    validate::finite(value, field).map_err(|error| map_record_field_error(error, sat))?;
    if value < 0.0 || value > f64::from(u32::MAX) || value.trunc() != value {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field,
        });
    }
    Ok(value as u32)
}

fn glonass_frequency_channel(value: f64, sat: &str) -> Result<i32, NavParseError> {
    const FIELD: &str = "frequency channel";
    validate::finite(value, FIELD).map_err(|error| map_record_field_error(error, sat))?;
    let channel = value as i32;
    if value.trunc() != value || !valid_glonass_frequency_channel(channel) {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field: FIELD,
        });
    }
    Ok(channel)
}

fn strict_header_f64(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<f64, NavParseError> {
    validate::strict_f64(raw_field(line, start, end), field).map_err(map_header_field_error)
}

fn strict_header_integer_f64(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<f64, NavParseError> {
    let value = strict_header_f64(line, start, end, field)?;
    if value.trunc() != value {
        return Err(NavParseError::BadHeaderField { field });
    }
    Ok(value)
}

fn strict_record_int<T>(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
    satellite: &str,
) -> Result<T, NavParseError>
where
    T: core::str::FromStr,
{
    validate::strict_int::<T>(raw_field(line, start, end), field)
        .map_err(|error| map_record_field_error(error, satellite))
}

fn map_record_field_error(error: FieldError, satellite: &str) -> NavParseError {
    NavParseError::BadField {
        satellite: satellite.to_string(),
        field: error.field(),
    }
}

fn map_header_field_error(error: FieldError) -> NavParseError {
    NavParseError::BadHeaderField {
        field: error.field(),
    }
}

/// Parse the clock reference epoch from the SV/epoch line into week and seconds
/// of week in the record's broadcast time scale.
fn parse_toc(
    l0: &str,
    sat: &str,
    time_scale: TimeScale,
) -> Result<ClockReferenceEpoch, NavParseError> {
    let year = strict_record_int::<i64>(l0, 4, 8, "toc epoch", sat)?;
    let month = strict_record_int::<i64>(l0, 9, 11, "toc epoch", sat)?;
    let day = strict_record_int::<i64>(l0, 12, 14, "toc epoch", sat)?;
    let hour = strict_record_int::<i64>(l0, 15, 17, "toc epoch", sat)?;
    let minute = strict_record_int::<i64>(l0, 18, 20, "toc epoch", sat)?;
    let second = strict_record_int::<i64>(l0, 21, 23, "toc epoch", sat)?;
    let civil = validate::civil_datetime_with_second_policy(
        year,
        month,
        day,
        hour,
        minute,
        second as f64,
        validate::CivilSecondPolicy::Continuous,
    )
    .map_err(|_| NavParseError::BadField {
        satellite: sat.to_string(),
        field: "toc epoch",
    })?;
    let month = i64::from(civil.month);
    let day = i64::from(civil.day);
    let week = gnss::week_from_calendar(time_scale, civil.year, month, day).ok_or_else(|| {
        NavParseError::BadField {
            satellite: sat.to_string(),
            field: "toc epoch",
        }
    })?;
    let sow = gnss::seconds_of_week_from_calendar(
        civil.year,
        month,
        day,
        i64::from(civil.hour),
        i64::from(civil.minute),
        civil.second as i64,
    );
    Ok(ClockReferenceEpoch { week, sow })
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
