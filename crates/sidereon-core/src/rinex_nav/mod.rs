//! RINEX 2.xx, 3.xx and 4.xx navigation-message reading and writing.
//!
//! Records read: GPS LNAV, CNAV and CNAV-2; QZSS LNAV, CNAV and CNAV-2; Galileo
//! I/NAV and F/NAV; BeiDou D1/D2; NavIC LNAV (all Keplerian, [`BroadcastRecord`]);
//! GLONASS FDMA ([`GlonassRecord`]); SBAS ([`SbasRecord`]); and the RINEX 4 system
//! time offset, Earth orientation and ionosphere frames. BeiDou CNAV-1/2/3 and NavIC
//! L1 frames are recognized and reported as not decoded, and [`parse_nav_file`]
//! keeps their text.
//!
//! Version 4 wraps each record in a `> EPH|STO|EOP|ION SVNN MSG` frame marker and
//! keeps the version 3 fixed-column layout for the legacy messages; GPS/QZSS CNAV
//! and CNAV-2 use their own roster. Version 2 files hold one system each (`N` GPS,
//! `G` GLONASS, `H` SBAS), with a two-digit PRN, a two-digit year and three
//! columns of indentation; QZSS and Galileo version 2 extensions carry a system
//! letter as version 3 does.
//!
//! This is deterministic byte-to-record parsing of a fixed-column text format, not
//! a float recipe: there is no 0-ULP claim here, and a small in-house parser is
//! used in preference to a heavyweight RINEX dependency (the published `rinex`
//! crate pulls ~90 transitive crates, including computational-geometry stacks,
//! for what is a fixed-width text read).
//!
//! [`parse_nav_file`] reads a whole file into a [`NavFile`] that keeps every
//! header record and every block, decoded or not, with its text, and
//! [`encode_nav_file`] writes it back: a file read and written unchanged comes
//! back byte for byte. The strict readers ([`parse_nav`], [`parse_glonass`],
//! [`parse_sbas`]) refuse the first block of their kind that departs from the
//! format; the lenient readers keep every readable record and report each
//! departure and each block they could not read with its line number and reason.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod store;
pub(crate) use store::ura_variance_m2;
pub use store::{BroadcastStore, NavMessagePreference};

mod write;
pub use write::{encode_nav, encode_nav_file, NavWriteError};

mod header;
pub use header::{HeaderIonoRow, NavHeader, TimeSystemCorrection};

mod body;
pub use body::{NavEntry, NavEntryKind, NavFile, NavItem, UndecodedBlock, UndecodedReason};

mod frames;
pub use frames::{EarthOrientation, IonosphereFrame, IonosphereModel, SystemTimeOffset};

mod geo;
pub use geo::SbasRecord;

use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::astro::time::{civil, gnss};
use crate::broadcast::{ClockPolynomial, ConstellationConstants, KeplerianElements};
use crate::constants::{KM_TO_M, SECONDS_PER_HOUR, SECONDS_PER_WEEK};
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

/// Largest distance, seconds, between a query and the `toe` of the Keplerian
/// record selected for it, as RTKLIB `seleph` bounds it (`rtklib.h`): GPS, QZSS
/// and NavIC `MAXDTOE + 1` = 7201 s, Galileo `MAXDTOE_GAL` = 14400 s, BeiDou
/// `MAXDTOE_CMP + 1` = 21601 s. The broadcast fit interval does not enter the
/// selection; it stays on the record as metadata.
pub(crate) fn keplerian_max_dtoe_s(system: GnssSystem) -> f64 {
    match system {
        GnssSystem::Galileo => 14_400.0,
        GnssSystem::BeiDou => 21_601.0,
        _ => 7_201.0,
    }
}

/// Largest distance, seconds, between a query and the reference epoch of the
/// GLONASS record selected for it: RTKLIB `MAXDTOE_GLO`.
pub(crate) const GLONASS_MAX_AGE_S: f64 = 1800.0;
/// Largest distance, seconds, between a query and the reference epoch of the SBAS
/// record selected for it: RTKLIB `MAXDTOE_SBS`.
pub(crate) const SBAS_MAX_AGE_S: f64 = 360.0;
/// Step of RTKLIB `ephpos` (`tt = 1E-3`), seconds: a broadcast record's velocity is the
/// difference of its positions at an epoch and this long after, both evaluated from the
/// record's reduced time (time of week, or time from the reference epoch) as RTKLIB adds
/// the step to its exact `gtime_t`.
pub(crate) const EPHPOS_STEP_S: f64 = 1.0e-3;

/// Seconds of GPS week at the J2000 epoch: `GPS_EPOCH_TO_J2000_S` is a whole number
/// of seconds, 1042 weeks and 561600 s.
pub(crate) const J2000_GPS_SECONDS_OF_WEEK: f64 = 561_600.0;
const _: () = assert!(
    crate::constants::GPS_EPOCH_TO_J2000_S - 1042.0 * SECONDS_PER_WEEK == J2000_GPS_SECONDS_OF_WEEK
);

/// A query epoch as a Keplerian broadcast record of `sat` reads it: seconds since J2000
/// in the record's own time scale (GPS time for GPS, Galileo, QZSS and NavIC, BDT for
/// BeiDou), seconds of week in that scale, and whether `sat` takes the BeiDou
/// geostationary orbit branch. `None` for a system without a Keplerian broadcast model
/// here, or a non-finite epoch.
///
/// The seconds of week are formed as RTKLIB's `gtime_t` holds an instant, whole seconds
/// and a fraction apart: the whole-second part of `t_j2000_s` is placed in its week with
/// integer-valued arithmetic, which is exact, and the fraction is added once at the end.
/// The result is `t_j2000_s`'s seconds of week rounded once, and exact whenever that
/// value is representable, which holds for every `|t_j2000_s| >= 2^20` s (every epoch
/// after mid-January 2000 or before mid-December 1999): there the fraction is a multiple
/// of 2^-32 s, which doubles below one week (2^-33 s spacing) resolve. Adding
/// `GPS_EPOCH_TO_J2000_S` to the J2000 seconds first would instead round to the 2.4e-7 s
/// spacing of doubles near 1.4e9 s and drop the last bit of a fractional epoch. The BDT
/// seconds since J2000, `t_j2000_s - 14`, are likewise rounded once. NavIC week and
/// seconds of week are read on the GPS week, as RTKLIB reads the RINEX `IRN week`.
pub(crate) fn query_native_time(sat: GnssSatelliteId, t_j2000_s: f64) -> Option<(f64, f64, bool)> {
    if !t_j2000_s.is_finite() {
        return None;
    }
    let whole_s = t_j2000_s.floor();
    let fraction_s = t_j2000_s - whole_s;
    let gps_whole_sow_s = (whole_s.rem_euclid(SECONDS_PER_WEEK) + J2000_GPS_SECONDS_OF_WEEK)
        .rem_euclid(SECONDS_PER_WEEK);
    let seconds_of_week = |whole_sow_s: f64| {
        let sow_s = whole_sow_s + fraction_s;
        // A fraction within half an ulp of the next whole second can round the sum up to
        // the week's end, which is the next week's start.
        if sow_s >= SECONDS_PER_WEEK {
            sow_s - SECONDS_PER_WEEK
        } else {
            sow_s
        }
    };
    match sat.system {
        GnssSystem::Gps | GnssSystem::Galileo | GnssSystem::Qzss | GnssSystem::Navic => {
            Some((t_j2000_s, seconds_of_week(gps_whole_sow_s), false))
        }
        // BDT runs GPST - 14 s; its week epoch is a whole number of GPS weeks later.
        GnssSystem::BeiDou => Some((
            t_j2000_s - crate::constants::GPST_MINUS_BDT_S,
            seconds_of_week(
                (gps_whole_sow_s - crate::constants::GPST_MINUS_BDT_S).rem_euclid(SECONDS_PER_WEEK),
            ),
            is_beidou_geo(sat),
        )),
        _ => None,
    }
}

/// A week/TOW pair in its own scale as continuous seconds since J2000 in that scale:
/// the whole weeks and the epoch offsets are whole seconds and are summed first, so a
/// fractional TOW keeps every bit.
pub(crate) fn week_tow_native_j2000_s(week_tow: GnssWeekTow) -> f64 {
    let epoch_offset_s = match week_tow.system {
        TimeScale::Bdt => crate::constants::BDS_EPOCH_MINUS_GPS_EPOCH_S,
        _ => 0.0,
    };
    (f64::from(week_tow.week) * SECONDS_PER_WEEK + epoch_offset_s
        - crate::constants::GPS_EPOCH_TO_J2000_S)
        + week_tow.tow_s
}

/// A record's `toe` in seconds since J2000 in the record's own time scale, the scale
/// [`query_native_time`] returns.
pub(crate) fn toe_native_j2000_s(record: &BroadcastRecord) -> f64 {
    week_tow_native_j2000_s(record.toe)
}

/// A record's reduced time `tk` one [`EPHPOS_STEP_S`] later, rounded as RTKLIB forms it.
/// `ephpos` adds the step to its `gtime_t` with `timeadd`, which adds it to the fraction
/// of a second and carries any whole second, and `eph2pos`/`geph2pos` then form
/// `timediff(time, toe)`, the whole seconds plus that fraction. A broadcast reference
/// epoch is a whole second, so `tk`'s whole and fractional parts are the instant's own,
/// and adding the step to the fraction first rounds where RTKLIB rounds.
pub(crate) fn ephpos_stepped_tk(tk_s: f64) -> f64 {
    let whole_s = tk_s.floor();
    let fraction_s = (tk_s - whole_s) + EPHPOS_STEP_S;
    let carry_s = fraction_s.floor();
    (whole_s + carry_s) + (fraction_s - carry_s)
}
const GPS_NOMINAL_FIT_INTERVAL_S: f64 = 4.0 * SECONDS_PER_HOUR;
/// Fit interval for legacy RINEX 3.00–3.02 GPS records when the fit-interval flag
/// is 1 (extended fit). RINEX 3.02 Table A6 explicitly defines flag 1 as 6 hours
/// (21600.0 s). The old constant was an inherited defect now independently
/// verified against primary text.
const GPS_LEGACY_EXTENDED_FIT_INTERVAL_S: f64 = 6.0 * SECONDS_PER_HOUR;
/// QZSS fit interval for fit flag 0, two hours (IS-QZSS-PNT), as RTKLIB reads it.
const QZSS_SHORT_FIT_INTERVAL_S: f64 = 2.0 * SECONDS_PER_HOUR;
/// QZSS fit interval for fit flag 1, "more than two hours" in IS-QZSS-PNT, which RTKLIB
/// `decode_eph` reads as four hours.
const QZSS_LONG_FIT_INTERVAL_S: f64 = 4.0 * SECONDS_PER_HOUR;
const GLONASS_FREQ_CHANNEL_MIN: i32 = -7;
const GLONASS_FREQ_CHANNEL_MAX: i32 = 6;
/// GLONASS G1 and G2 FDMA base frequencies (Hz), RTKLIB `FREQ1_GLO` and `FREQ2_GLO`.
const GLONASS_G1_BASE_HZ: f64 = 1.602e9;
const GLONASS_G2_BASE_HZ: f64 = 1.246e9;
/// The RINEX 3.05/4.00 value of a GLONASS L1/L2 group delay difference that is not
/// known, `.999999999999E+09`.
const GLONASS_DELAY_UNKNOWN_S: f64 = 0.999_999_999_999e9;
/// The RINEX 3.05/4.00 value of GLONASS status or health flags that are not known,
/// `999999999999`.
const GLONASS_FLAGS_UNKNOWN: f64 = 999_999_999_999.0;

/// Whether a GLONASS frequency channel lies in the `-7..=6` FDMA allocation,
/// the channels a carrier frequency is resolved for. The readers keep a stated
/// channel outside it; this is the check their consumers apply.
pub(crate) fn valid_glonass_frequency_channel(channel: i32) -> bool {
    (GLONASS_FREQ_CHANNEL_MIN..=GLONASS_FREQ_CHANNEL_MAX).contains(&channel)
}

/// A RINEX version number as the `RINEX VERSION / TYPE` record states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NavVersion {
    /// Major version, 2, 3 or 4.
    pub major: u8,
    /// Minor version, the digits after the point (`3.05` is minor 5).
    pub minor: u8,
}

impl NavVersion {
    /// Build a version.
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    fn gps_fit_interval_uses_legacy_flag(self) -> bool {
        self.major == 3 && self.minor <= 2
    }

    /// Whether a GLONASS record carries the fourth broadcast-orbit line (status flags,
    /// L1/L2 group delay difference, URAI and health flags), added in RINEX 3.05.
    pub(crate) fn glonass_has_fourth_orbit_line(self) -> bool {
        self.major >= 4 || (self.major == 3 && self.minor >= 5)
    }
}

impl core::fmt::Display for NavVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{:02}", self.major, self.minor)
    }
}

/// Which broadcast navigation message a record carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMessage {
    /// GPS legacy navigation message.
    GpsLnav,
    /// GPS CNAV message (L2C/L5), RINEX 4 token `CNAV`.
    GpsCnav,
    /// GPS CNAV-2 message (L1C), RINEX 4 token `CNV2`.
    GpsCnav2,
    /// QZSS legacy navigation message.
    QzssLnav,
    /// QZSS CNAV message, RINEX 4 token `CNAV`.
    QzssCnav,
    /// QZSS CNAV-2 message, RINEX 4 token `CNV2`.
    QzssCnav2,
    /// Galileo integrity navigation message (E1/E5b dual, E1 single-frequency).
    GalileoInav,
    /// Galileo F/NAV message (E5a).
    GalileoFnav,
    /// A Galileo record whose data-source word names neither I/NAV nor F/NAV alone:
    /// no source bit, or both the I/NAV bits (0 or 2) and the F/NAV bit (1). The
    /// word is kept on the record ([`BroadcastRecord::galileo_data_sources`]).
    GalileoUnclassified,
    /// BeiDou D1 message (MEO/IGSO satellites).
    BeidouD1,
    /// BeiDou D2 message (geostationary satellites).
    BeidouD2,
    /// NavIC (IRNSS) legacy navigation message, RINEX 4 token `LNAV`.
    NavicLnav,
}

impl NavMessage {
    /// Whether this is a GPS/QZSS CNAV-family message.
    pub const fn is_cnav_family(self) -> bool {
        matches!(
            self,
            Self::GpsCnav | Self::GpsCnav2 | Self::QzssCnav | Self::QzssCnav2
        )
    }
}

/// Broadcast issue-of-data plus the navigation message identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastIssue {
    /// The native issue value: IODE for GPS and QZSS, IODnav for Galileo, AODE for
    /// BeiDou, IODEC for NavIC.
    pub issue: u32,
    /// The navigation message carrying the issue value.
    pub message: NavMessage,
}

/// NavIC S-band carrier, Hz (RTKLIB `FREQs`).
const NAVIC_S_HZ: f64 = 2.492028e9;
/// NavIC L5 carrier, Hz (RTKLIB `FREQL5`).
const NAVIC_L5_HZ: f64 = 1.17645e9;
/// The factor on the NavIC TGD for the L5 single-frequency user, `(f_S / f_L5)²`,
/// formed as RTKLIB `SQR(FREQs/FREQL5)` forms it.
const NAVIC_L5_TGD_FACTOR: f64 = (NAVIC_S_HZ / NAVIC_L5_HZ) * (NAVIC_S_HZ / NAVIC_L5_HZ);

/// A broadcast group-delay term carried by a RINEX NAV record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastGroupDelayTerm {
    /// GPS/QZSS LNAV TGD (also the NavIC TGD).
    GpsTgd,
    /// Galileo BGD E5a/E1.
    GalileoBgdE5aE1,
    /// Galileo BGD E5b/E1.
    GalileoBgdE5bE1,
    /// BeiDou TGD1.
    BeidouTgd1,
    /// BeiDou TGD2.
    BeidouTgd2,
    /// GPS/QZSS CNAV ISC for L1 C/A.
    CnavIscL1Ca,
    /// GPS/QZSS CNAV ISC for L2C.
    CnavIscL2C,
    /// GPS/QZSS CNAV ISC for L5 I5.
    CnavIscL5I5,
    /// GPS/QZSS CNAV ISC for L5 Q5.
    CnavIscL5Q5,
    /// GPS/QZSS CNAV-2 ISC for L1C data.
    CnavIscL1Cd,
    /// GPS/QZSS CNAV-2 ISC for L1C pilot.
    CnavIscL1Cp,
}

/// A GPS/QZSS signal a CNAV-family group-delay correction applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnavSignal {
    /// L1 C/A.
    L1Ca,
    /// L2C.
    L2C,
    /// L5 I5.
    L5I5,
    /// L5 Q5.
    L5Q5,
    /// L1C pilot.
    L1Cp,
    /// L1C data.
    L1Cd,
}

/// Per-signal broadcast group delays preserved from one NAV record.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BroadcastGroupDelays {
    /// GPS/QZSS LNAV TGD (and the NavIC TGD), seconds.
    pub gps_tgd_s: Option<f64>,
    /// Galileo BGD E5a/E1, seconds.
    pub galileo_bgd_e5a_e1_s: Option<f64>,
    /// Galileo BGD E5b/E1, seconds.
    pub galileo_bgd_e5b_e1_s: Option<f64>,
    /// BeiDou TGD1, seconds.
    pub beidou_tgd1_s: Option<f64>,
    /// BeiDou TGD2, seconds.
    pub beidou_tgd2_s: Option<f64>,
    /// GPS/QZSS CNAV ISC for L1 C/A, seconds.
    pub cnav_isc_l1ca_s: Option<f64>,
    /// GPS/QZSS CNAV ISC for L2C, seconds.
    pub cnav_isc_l2c_s: Option<f64>,
    /// GPS/QZSS CNAV ISC for L5 I5, seconds.
    pub cnav_isc_l5i5_s: Option<f64>,
    /// GPS/QZSS CNAV ISC for L5 Q5, seconds.
    pub cnav_isc_l5q5_s: Option<f64>,
    /// GPS/QZSS CNAV-2 ISC for L1C data, seconds.
    pub cnav_isc_l1cd_s: Option<f64>,
    /// GPS/QZSS CNAV-2 ISC for L1C pilot, seconds.
    pub cnav_isc_l1cp_s: Option<f64>,
}

impl BroadcastGroupDelays {
    /// Build a GPS/QZSS LNAV delay set with an optional TGD.
    pub const fn gps_lnav_opt(tgd_s: Option<f64>) -> Self {
        Self {
            gps_tgd_s: tgd_s,
            galileo_bgd_e5a_e1_s: None,
            galileo_bgd_e5b_e1_s: None,
            beidou_tgd1_s: None,
            beidou_tgd2_s: None,
            cnav_isc_l1ca_s: None,
            cnav_isc_l2c_s: None,
            cnav_isc_l5i5_s: None,
            cnav_isc_l5q5_s: None,
            cnav_isc_l1cd_s: None,
            cnav_isc_l1cp_s: None,
        }
    }

    /// Build the GPS LNAV delay set.
    pub const fn gps_lnav(tgd_s: f64) -> Self {
        Self::gps_lnav_opt(Some(tgd_s))
    }

    /// Build a Galileo delay set with optional BGD terms.
    pub const fn galileo_opt(bgd_e5a_e1_s: Option<f64>, bgd_e5b_e1_s: Option<f64>) -> Self {
        Self {
            gps_tgd_s: None,
            galileo_bgd_e5a_e1_s: bgd_e5a_e1_s,
            galileo_bgd_e5b_e1_s: bgd_e5b_e1_s,
            beidou_tgd1_s: None,
            beidou_tgd2_s: None,
            cnav_isc_l1ca_s: None,
            cnav_isc_l2c_s: None,
            cnav_isc_l5i5_s: None,
            cnav_isc_l5q5_s: None,
            cnav_isc_l1cd_s: None,
            cnav_isc_l1cp_s: None,
        }
    }

    /// Build the Galileo delay set.
    pub const fn galileo(bgd_e5a_e1_s: f64, bgd_e5b_e1_s: f64) -> Self {
        Self::galileo_opt(Some(bgd_e5a_e1_s), Some(bgd_e5b_e1_s))
    }

    /// Build a BeiDou delay set with optional TGD terms.
    pub const fn beidou_opt(tgd1_s: Option<f64>, tgd2_s: Option<f64>) -> Self {
        Self {
            gps_tgd_s: None,
            galileo_bgd_e5a_e1_s: None,
            galileo_bgd_e5b_e1_s: None,
            beidou_tgd1_s: tgd1_s,
            beidou_tgd2_s: tgd2_s,
            cnav_isc_l1ca_s: None,
            cnav_isc_l2c_s: None,
            cnav_isc_l5i5_s: None,
            cnav_isc_l5q5_s: None,
            cnav_isc_l1cd_s: None,
            cnav_isc_l1cp_s: None,
        }
    }

    /// Build the BeiDou delay set.
    pub const fn beidou(tgd1_s: f64, tgd2_s: f64) -> Self {
        Self::beidou_opt(Some(tgd1_s), Some(tgd2_s))
    }

    /// Build a GPS/QZSS CNAV-family delay set.
    pub const fn cnav(
        tgd_s: Option<f64>,
        isc_l1ca_s: Option<f64>,
        isc_l2c_s: Option<f64>,
        isc_l5i5_s: Option<f64>,
        isc_l5q5_s: Option<f64>,
        isc_l1cd_s: Option<f64>,
        isc_l1cp_s: Option<f64>,
    ) -> Self {
        Self {
            gps_tgd_s: tgd_s,
            galileo_bgd_e5a_e1_s: None,
            galileo_bgd_e5b_e1_s: None,
            beidou_tgd1_s: None,
            beidou_tgd2_s: None,
            cnav_isc_l1ca_s: isc_l1ca_s,
            cnav_isc_l2c_s: isc_l2c_s,
            cnav_isc_l5i5_s: isc_l5i5_s,
            cnav_isc_l5q5_s: isc_l5q5_s,
            cnav_isc_l1cd_s: isc_l1cd_s,
            cnav_isc_l1cp_s: isc_l1cp_s,
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
            BroadcastGroupDelayTerm::CnavIscL1Ca => self.cnav_isc_l1ca_s,
            BroadcastGroupDelayTerm::CnavIscL2C => self.cnav_isc_l2c_s,
            BroadcastGroupDelayTerm::CnavIscL5I5 => self.cnav_isc_l5i5_s,
            BroadcastGroupDelayTerm::CnavIscL5Q5 => self.cnav_isc_l5q5_s,
            BroadcastGroupDelayTerm::CnavIscL1Cd => self.cnav_isc_l1cd_s,
            BroadcastGroupDelayTerm::CnavIscL1Cp => self.cnav_isc_l1cp_s,
        }
    }

    /// The total CNAV single-frequency clock adjustment (TGD - ISC), seconds.
    ///
    /// Callers subtract this from the satellite clock offset by passing it as the
    /// `tgd_s` argument to the broadcast evaluator. Returns `None` when TGD or
    /// the selected ISC is unavailable.
    pub fn cnav_single_frequency_correction_s(&self, signal: CnavSignal) -> Option<f64> {
        let isc = match signal {
            CnavSignal::L1Ca => self.cnav_isc_l1ca_s,
            CnavSignal::L2C => self.cnav_isc_l2c_s,
            CnavSignal::L5I5 => self.cnav_isc_l5i5_s,
            CnavSignal::L5Q5 => self.cnav_isc_l5q5_s,
            CnavSignal::L1Cp => self.cnav_isc_l1cp_s,
            CnavSignal::L1Cd => self.cnav_isc_l1cd_s,
        }?;
        Some(self.gps_tgd_s? - isc)
    }

    /// The delay term historically used for broadcast-clock evaluation.
    ///
    /// BeiDou has no signal choice at this store level, so it keeps the previous
    /// TGD1 behavior. CNAV-family clock evaluation keeps the record-level
    /// default of treating a missing TGD or L1 C/A ISC as zero. A Galileo record whose
    /// message is unclassified uses the BGD E5b/E1, as RTKLIB's default Galileo
    /// selection does. NavIC LNAV is the L5 single-frequency user's term: the IRNSS SPS
    /// ICD 1.1 section 6.2.1.5 states `(Δt_SV)L5 = Δt_SV - γ·TGD` with
    /// `γ = (f_S / f_L5)²`, which RTKLIB `prange` applies as `SQR(FREQs/FREQL5)·TGD`.
    /// Callers that know their signal should use [`Self::get`] or
    /// [`Self::cnav_single_frequency_correction_s`].
    pub const fn for_message(self, system: GnssSystem, message: NavMessage) -> Option<f64> {
        match (system, message) {
            (GnssSystem::Gps, NavMessage::GpsLnav) | (GnssSystem::Qzss, NavMessage::QzssLnav) => {
                self.get(BroadcastGroupDelayTerm::GpsTgd)
            }
            (GnssSystem::Navic, NavMessage::NavicLnav) => match self.gps_tgd_s {
                Some(tgd) => Some(NAVIC_L5_TGD_FACTOR * tgd),
                None => None,
            },
            (GnssSystem::Galileo, NavMessage::GalileoFnav) => {
                self.get(BroadcastGroupDelayTerm::GalileoBgdE5aE1)
            }
            (GnssSystem::Galileo, NavMessage::GalileoInav | NavMessage::GalileoUnclassified) => {
                self.get(BroadcastGroupDelayTerm::GalileoBgdE5bE1)
            }
            (GnssSystem::BeiDou, NavMessage::BeidouD1 | NavMessage::BeidouD2) => {
                self.get(BroadcastGroupDelayTerm::BeidouTgd1)
            }
            (
                GnssSystem::Gps | GnssSystem::Qzss,
                NavMessage::GpsCnav
                | NavMessage::GpsCnav2
                | NavMessage::QzssCnav
                | NavMessage::QzssCnav2,
            ) => match (self.gps_tgd_s, self.cnav_isc_l1ca_s) {
                (Some(tgd), Some(isc)) => Some(tgd - isc),
                (Some(tgd), None) => Some(tgd),
                (None, Some(isc)) => Some(-isc),
                (None, None) => Some(0.0),
            },
            _ => None,
        }
    }
}

/// CNAV/CNAV-2 parameters that have no legacy counterpart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CnavParameters {
    /// Semi-major axis rate ADOT (m/s), ORBIT-1 field 1.
    pub adot_m_s: f64,
    /// Rate of the mean-motion difference (rad/s^2), ORBIT-5 field 2.
    pub delta_n0_dot_rad_s2: f64,
    /// CEI data-sequence propagation epoch: WNop week plus top seconds of week.
    pub top: GnssWeekTow,
    /// URA_ED index, [-16, 15].
    pub ura_ed_index: i8,
    /// URA_NED0 index, [-16, 15].
    pub ura_ned0_index: i8,
    /// URA_NED1 index, [0, 7].
    pub ura_ned1_index: u8,
    /// URA_NED2 index, [0, 7].
    pub ura_ned2_index: u8,
    /// Transmission time of message t_tm, seconds of week.
    pub transmission_time_sow: f64,
    /// Optional decimal-coded flag bits.
    pub flags: Option<u32>,
}

/// Nominal URA meters for a CNAV ED/NED0 index.
///
/// Returns `None` for the no-prediction indices 15 and -16.
pub fn cnav_ura_nominal_m(index: i8) -> Option<f64> {
    match index {
        -16 | 15 => None,
        1 => Some(2.8),
        3 => Some(5.7),
        5 => Some(11.3),
        -15..=6 => Some(libm::pow(2.0_f64, 1.0 + f64::from(index) / 2.0)),
        7..=14 => Some(2.0_f64.powi(i32::from(index) - 2)),
        _ => None,
    }
}

/// Time-dependent CNAV URA_NED bound in meters at GPST `t`.
pub fn cnav_ura_ned_m(params: &CnavParameters, t: GnssWeekTow) -> Option<f64> {
    let ned0 = cnav_ura_nominal_m(params.ura_ned0_index)?;
    let ned1 = 2.0_f64.powi(-(14 + i32::from(params.ura_ned1_index)));
    let ned2 = 2.0_f64.powi(-(28 + i32::from(params.ura_ned2_index)));
    let dt_op = (f64::from(t.week) - f64::from(params.top.week)) * SECONDS_PER_WEEK
        + (t.tow_s - params.top.tow_s);
    let linear = ned0 + ned1 * dt_op;
    if dt_op <= 93_600.0 {
        Some(linear)
    } else {
        Some(linear + ned2 * (dt_op - 93_600.0) * (dt_op - 93_600.0))
    }
}

/// Whether a BeiDou PRN is a geostationary satellite, which takes the
/// geostationary orbit-evaluation branch and the D2 message.
///
/// The BDS ICD assigns PRN 1-5 and 59-63 to GEO satellites, and RTKLIB
/// `eph2pos` tests `prn<=5||prn>=59` over its 1..=63 BeiDou range. PRN 64 and
/// above are spellable satellite tokens but no GEO assignment covers them, so
/// they take the MEO/IGSO branch.
pub fn is_beidou_geo(sat: GnssSatelliteId) -> bool {
    sat.system == GnssSystem::BeiDou && ((1..=5).contains(&sat.prn) || (59..=63).contains(&sat.prn))
}

/// A Klobuchar-8 broadcast ionosphere coefficient set (the eight alpha/beta
/// values transmitted by GPS, QZSS, BeiDou and NavIC; the same model serves each,
/// evaluated per carrier - see [`crate::ionex::klobuchar_native`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KlobucharAlphaBeta {
    /// Cosine-amplitude polynomial coefficients (a0..a3).
    pub alpha: [f64; 4],
    /// Period polynomial coefficients (b0..b3).
    pub beta: [f64; 4],
}

/// Broadcast ionosphere-correction coefficients from a RINEX header's
/// `IONOSPHERIC CORR` (or version 2 `ION ALPHA`/`ION BETA`) records or RINEX 4
/// body `> ION` frames.
///
/// Captures the Klobuchar-8 sets of GPS (`GPSA`/`GPSB`, `LNAV`), QZSS
/// (`QZSA`/`QZSB`, `LNAV`), BeiDou (`BDSA`/`BDSB`, `D1D2`) and NavIC
/// (`IRNA`/`IRNB`, `LNAV`), Galileo's three NeQuick-G effective-ionisation
/// coefficients and disturbance flags (`GAL`, `IFNV`), and the nine BeiDou BDGIM
/// coefficients of a RINEX 4 `CNVX` frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IonoCorrections {
    /// GPS broadcast Klobuchar coefficients (`GPSA`/`GPSB`), if present.
    pub gps: Option<KlobucharAlphaBeta>,
    /// BeiDou broadcast Klobuchar coefficients (`BDSA`/`BDSB`), if present.
    pub beidou: Option<KlobucharAlphaBeta>,
    /// Galileo broadcast NeQuick-G coefficients (`GAL`), if present.
    pub galileo: Option<GalileoNequickCoeffs>,
    /// Galileo ionospheric disturbance flags (the fourth `GAL` value), as stated.
    pub galileo_disturbance_flags: Option<f64>,
    /// QZSS broadcast Klobuchar coefficients (`QZSA`/`QZSB`), if present.
    pub qzss: Option<KlobucharAlphaBeta>,
    /// NavIC broadcast Klobuchar coefficients (`IRNA`/`IRNB`), if present.
    pub navic: Option<KlobucharAlphaBeta>,
    /// BeiDou global ionospheric model (BDGIM) coefficients `alpha1..alpha9` from a
    /// RINEX 4 `CNVX` ionosphere frame, if present.
    pub beidou_bdgim: Option<[f64; 9]>,
}

/// One parsed GLONASS broadcast record: a PZ-90.11 ECEF state vector and the
/// clock terms, evaluated by the crate's GLONASS RK4 propagator (GLONASS is not
/// Keplerian, so it does not use [`BroadcastRecord`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlonassRecord {
    /// The transmitting satellite.
    pub satellite_id: GnssSatelliteId,
    /// Reference epoch `tb` as seconds past J2000 in **UTC**: the record's epoch
    /// rounded to the 15-minute grid, as RTKLIB `decode_geph` rounds it (`tb` is a
    /// count of 15-minute intervals, GLONASS ICD). [`Self::toe_gpst_j2000_s`] places it
    /// on the GPS timeline.
    pub toe_utc_j2000_s: f64,
    /// The epoch as the record states it, seconds past J2000 in UTC.
    pub epoch_utc_j2000_s: f64,
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
    /// Satellite health, BROADCAST ORBIT-1 field 4 (0 is healthy).
    pub sv_health: f64,
    /// FDMA frequency-channel number. A stated value above 128 is a receiver's
    /// unsigned spelling of a negative channel and is read as that value less 256,
    /// as RTKLIB `decode_geph` reads it.
    pub freq_channel: i32,
    /// The frequency channel field (BROADCAST ORBIT-2 field 4) as stated, before a value
    /// above 128 is folded into [`Self::freq_channel`]. The writer states it again while
    /// it folds to `freq_channel`.
    pub stated_freq_channel: i32,
    /// Message frame time (line 0 field 3) as stated: seconds of the UTC week in
    /// RINEX 3 and 4, seconds of the UTC day in RINEX 2. `None` when blank.
    pub message_frame_time_s: Option<f64>,
    /// Age of operational information `E_n` (days), BROADCAST ORBIT-3 field 4.
    pub age_days: Option<f64>,
    /// Status flags (RINEX 3.05/4.00 BROADCAST ORBIT-4 field 1) as stated.
    pub status_flags: Option<f64>,
    /// L1/L2 group delay difference field `ΔτN` (seconds, BROADCAST ORBIT-4 field 2)
    /// as stated, including the `.999999999999E+09` value for an unknown delay; see
    /// [`Self::l1_l2_group_delay_s`].
    pub l1_l2_group_delay_field_s: Option<f64>,
    /// Raw accuracy index URAI `F_T` (BROADCAST ORBIT-4 field 3) as stated.
    pub urai: Option<f64>,
    /// Health flags (BROADCAST ORBIT-4 field 4) as stated.
    pub health_flags: Option<f64>,
}

impl GlonassRecord {
    /// The reference epoch in GPS time, seconds since J2000: `tb` in UTC plus GPS - UTC
    /// from the leap-second table at that UTC instant, as RTKLIB `utc2gpst` converts it.
    /// The header `LEAP SECONDS` count does not enter it.
    pub fn toe_gpst_j2000_s(&self) -> f64 {
        self.toe_utc_j2000_s + gps_minus_utc_at_utc_j2000_s(self.toe_utc_j2000_s)
    }

    /// The L1/L2 group delay difference `ΔτN` in seconds, or `None` when the record has
    /// no fourth orbit line, leaves the field blank, or states it as not known
    /// (`.999999999999E+09`).
    pub fn l1_l2_group_delay_s(&self) -> Option<f64> {
        self.l1_l2_group_delay_field_s
            .filter(|value| *value != GLONASS_DELAY_UNKNOWN_S)
    }

    /// The status flags as a word: the stated value converted to the nearest integer, as
    /// RINEX 3.05 section 6.9 reads a bitwise field. `None` when blank, not known
    /// (`999999999999`), or not a non-negative value that fits `u32`.
    pub fn status_flags_word(&self) -> Option<u32> {
        glonass_flags_word(self.status_flags)
    }

    /// The health flags as a word, read as [`Self::status_flags_word`] reads the status
    /// flags.
    pub fn health_flags_word(&self) -> Option<u32> {
        glonass_flags_word(self.health_flags)
    }

    /// Whether the record reports the satellite healthy, by the RINEX 3.05 Table A10
    /// (4.01 Table A15) health fields:
    ///
    /// - health, the MSB of `Bn`, is 0 (1 is unhealthy);
    /// - where the health flags are stated with `AC` (bit 1) set, the almanac health `C`
    ///   (bit 0) is 1 (healthy); `C` is ignored when `AC` is 0;
    /// - where the health flags are stated, `l(3)` (bit 2, the health bit of string 3)
    ///   is 0. The tables give `l(3)` for GLONASS-M/K only, valid when the status flags'
    ///   type indicator `M` (bits 7-8) is `01`, so a record whose status flags state
    ///   another type does not use it; where the status flags are not stated, `l(3)` is
    ///   used, as RTKLIB demo5 uses it.
    ///
    /// RTKLIB demo5 forms `svh = Bn | flags << 1` and excludes a satellite with
    /// `(svh & 9) != 0 || (svh & 6) == 4`, which is the same rule without the type
    /// condition on `l(3)`.
    pub fn is_healthy(&self) -> bool {
        if self.sv_health != 0.0 {
            return false;
        }
        let Some(flags) = self.health_flags_word() else {
            return true;
        };
        let almanac_reported = flags & 0b010 != 0;
        let almanac_healthy = flags & 0b001 != 0;
        if almanac_reported && !almanac_healthy {
            return false;
        }
        let string3_unhealthy = flags & 0b100 != 0;
        let l3_valid = self
            .status_flags_word()
            .is_none_or(|status| (status >> 7) & 0b11 == 0b01);
        !(string3_unhealthy && l3_valid)
    }

    /// Satellite clock bias, seconds, at satellite clock time `t_sv_gpst_j2000_s` (GPS
    /// time scale, seconds past J2000): RTKLIB `geph2clk`, the GLONASS counterpart of
    /// [`crate::ephemeris::satellite_clock_bias_s`]. The time from the reference epoch
    /// is refined twice to remove the satellite clock from a time read on that clock;
    /// subtracting the returned bias from the satellite clock time gives system time,
    /// the epoch the orbit and its clock are evaluated at.
    pub fn clock_bias_s(&self, t_sv_gpst_j2000_s: f64) -> f64 {
        crate::glonass::clock_offset_s(
            self.clk_bias,
            self.gamma_n,
            t_sv_gpst_j2000_s - self.toe_gpst_j2000_s(),
        )
    }

    /// Single-frequency (G1) group delay, seconds, as RTKLIB `prange` applies the
    /// broadcast `ΔτN` to a G1 pseudorange: `-ΔτN / (γ - 1)` with
    /// `γ = (f_G1 / f_G2)²`, subtracted from the satellite clock as the other systems'
    /// TGD is. `None` where [`Self::l1_l2_group_delay_s`] is `None`.
    pub fn single_frequency_group_delay_s(&self) -> Option<f64> {
        let dtaun = self.l1_l2_group_delay_s()?;
        let ratio = GLONASS_G1_BASE_HZ / GLONASS_G2_BASE_HZ;
        let gamma = ratio * ratio;
        Some(-dtaun / (gamma - 1.0))
    }
}

fn glonass_flags_word(value: Option<f64>) -> Option<u32> {
    let value = value?;
    if value == GLONASS_FLAGS_UNKNOWN || !value.is_finite() {
        return None;
    }
    let word = value.round();
    if word < 0.0 || word > f64::from(u32::MAX) {
        return None;
    }
    Some(word as u32)
}

/// GPS - UTC, seconds, at a UTC instant given as seconds past J2000, from the crate's
/// leap-second table.
pub(crate) fn gps_minus_utc_at_utc_j2000_s(utc_j2000_s: f64) -> f64 {
    crate::astro::time::scales::gps_utc_offset_s(2_451_545.0 + utc_j2000_s / 86_400.0)
}

/// A GLONASS record skipped by [`parse_glonass_lenient`] because its slot token
/// is not representable as a [`GnssSatelliteId`].
///
/// The satellite-token range is `R01`..`R99`, which covers the extended slots
/// real BKG/IGS products carry, so what lands here is a token outside that
/// syntax altogether - `R00`, or a malformed slot field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedGlonass {
    /// The 3-character satellite token as it appeared in the file (`R00`).
    pub token: String,
    /// 1-based line number of the record's first line.
    pub line: usize,
}

/// The result of a lenient GLONASS parse: the readable records, the slot tokens
/// that were skipped, the records that could not be read, and the departures read
/// through.
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
    /// Records of representable slots that could not be read, with the reason.
    pub invalid: Vec<SkippedNavBlock>,
    /// Records kept although they depart from the format, with the departure.
    pub departures: Vec<NavDiagnostic>,
}

/// Fields of a legacy broadcast record that the orbit and clock models do not read,
/// as the record states them. `None` for a blank field or one the source does not
/// carry. The writer restates each one in the column it was read from.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StatedNavFields {
    /// BROADCAST ORBIT-5 field 2: GPS/QZSS codes on L2, the Galileo data-source word
    /// (see [`BroadcastRecord::galileo_data_sources`]), spare for BeiDou and NavIC.
    pub orbit5_field2: Option<f64>,
    /// BROADCAST ORBIT-5 field 4: GPS/QZSS L2 P data flag, spare elsewhere.
    pub orbit5_field4: Option<f64>,
    /// BROADCAST ORBIT-6 field 4: GPS/QZSS IODC, NavIC spare. Galileo's BGD E5b/E1 and
    /// BeiDou's TGD2 in this column are group delays and are held in
    /// [`BroadcastRecord::group_delays`].
    pub orbit6_field4: Option<f64>,
    /// BROADCAST ORBIT-7 field 1: transmission time of message, seconds of week, as
    /// stated (including a `.9999E+09` "not known" value).
    pub transmission_time_sow: Option<f64>,
    /// BROADCAST ORBIT-7 field 2: the GPS fit interval, the QZSS fit flag, the BeiDou
    /// AODC, spare for Galileo and NavIC, as stated.
    pub orbit7_field2: Option<f64>,
    /// BROADCAST ORBIT-7 field 3 (spare).
    pub orbit7_field3: Option<f64>,
    /// BROADCAST ORBIT-7 field 4 (spare).
    pub orbit7_field4: Option<f64>,
}

/// One parsed broadcast navigation record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastRecord {
    /// The transmitting satellite.
    pub satellite_id: GnssSatelliteId,
    /// The navigation message the record carries.
    pub message: NavMessage,
    /// Broadcast issue-of-data for issue-matched correction products. `None` for the
    /// GPS/QZSS CNAV family, whose RINEX 4 record carries no issue of data.
    pub issue_of_data: Option<BroadcastIssue>,
    /// Native broadcast week number as the record states it.
    pub week: u32,
    /// Scale-tagged ephemeris reference time (`toe`). Its week is the stated week
    /// moved by a whole week where that places `toe` within half a week of `toc`, as
    /// RTKLIB `adjweek` does; [`Self::week`] keeps the stated week.
    pub toe: GnssWeekTow,
    /// Scale-tagged clock reference time (`toc`).
    pub toc: GnssWeekTow,
    /// Keplerian orbital elements (`toe_sow` is seconds of week).
    pub elements: KeplerianElements,
    /// Clock polynomial (`toc_sow` is the record's own epoch, seconds of week).
    pub clock: ClockPolynomial,
    /// Broadcast group-delay terms carried by this message.
    pub group_delays: BroadcastGroupDelays,
    /// CNAV/CNAV-2 extension, present only for GPS/QZSS CNAV-family records.
    pub cnav: Option<CnavParameters>,
    /// Satellite health word (0 is healthy for the GPS/Galileo nominal case).
    pub sv_health: f64,
    /// Signal-in-space accuracy (m): the stated GPS/QZSS/NavIC URA, Galileo SISA or
    /// BeiDou URA, or the nominal value of the CNAV URA_ED index. `None` for the CNAV
    /// indices 15 and -16, which carry no accuracy prediction.
    pub sv_accuracy_m: Option<f64>,
    /// Curve-fit interval in seconds, centered on `toe`, as the record states it:
    /// GPS ORBIT-7 field 2 (hours in RINEX 2 and 3.03+, the 0/1 flag of RINEX
    /// 3.00-3.02), the QZSS fit flag read as RTKLIB reads it (0 two hours, 1 four
    /// hours). `None` when absent or unknown, and for Galileo, BeiDou, NavIC and the
    /// CNAV family, whose records state none. Broadcast record selection does not use
    /// it (see [`BroadcastStore`]).
    pub fit_interval_s: Option<f64>,
    /// Fields the record states that the orbit and clock models do not read.
    pub stated: StatedNavFields,
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
            // GPS, QZSS and NavIC use the GPS constants, as RTKLIB `eph2pos` does.
            _ => ConstellationConstants::GPS,
        }
    }

    /// Single-frequency group delay of this message, seconds: GPS, QZSS and NavIC LNAV
    /// TGD, Galileo I/NAV BGD E5b/E1, Galileo F/NAV BGD E5a/E1, BeiDou TGD1, CNAV TGD
    /// less ISC L1C/A. [`crate::broadcast::satellite_state`] subtracts it in
    /// `dt_clock_total_s`; the store's clock leaves it out, as RTKLIB `satposs` does, and
    /// returns it separately through `single_frequency_group_delay_s` for the
    /// single-frequency pseudorange model, as RTKLIB `pntpos` applies it.
    pub fn broadcast_clock_group_delay_s(&self) -> f64 {
        self.group_delays
            .for_message(self.satellite_id.system, self.message)
            .unwrap_or(0.0)
    }

    /// GPS/QZSS IODC (BROADCAST ORBIT-6 field 4), as stated.
    pub fn iodc(&self) -> Option<f64> {
        if matches!(self.satellite_id.system, GnssSystem::Gps | GnssSystem::Qzss) {
            self.stated.orbit6_field4
        } else {
            None
        }
    }

    /// GPS/QZSS codes on L2 (BROADCAST ORBIT-5 field 2), as stated.
    pub fn l2_codes(&self) -> Option<f64> {
        if matches!(self.satellite_id.system, GnssSystem::Gps | GnssSystem::Qzss) {
            self.stated.orbit5_field2
        } else {
            None
        }
    }

    /// GPS/QZSS L2 P data flag (BROADCAST ORBIT-5 field 4), as stated.
    pub fn l2p_data_flag(&self) -> Option<f64> {
        if matches!(self.satellite_id.system, GnssSystem::Gps | GnssSystem::Qzss) {
            self.stated.orbit5_field4
        } else {
            None
        }
    }

    /// The Galileo data-source word (BROADCAST ORBIT-5 field 2): bit 0 I/NAV E1-B, bit 1
    /// F/NAV E5a-I, bit 2 I/NAV E5b-I, bit 8 clock for E5a/E1, bit 9 clock for E5b/E1.
    /// The stated value is converted to the nearest integer, as RINEX 3.05 section 6.9
    /// reads a bitwise field. `None` for another system, a blank field, or a value
    /// outside `u32`.
    pub fn galileo_data_sources(&self) -> Option<u32> {
        if self.satellite_id.system != GnssSystem::Galileo {
            return None;
        }
        let value = self.stated.orbit5_field2?.round();
        if !(0.0..=f64::from(u32::MAX)).contains(&value) {
            None
        } else {
            Some(value as u32)
        }
    }

    /// BeiDou AODC (BROADCAST ORBIT-7 field 2), as stated.
    pub fn beidou_aodc(&self) -> Option<f64> {
        if self.satellite_id.system == GnssSystem::BeiDou {
            self.stated.orbit7_field2
        } else {
            None
        }
    }

    /// Transmission time of message, seconds of week: the CNAV `t_tm`, or the legacy
    /// BROADCAST ORBIT-7 field 1 as stated.
    pub fn transmission_time_sow(&self) -> Option<f64> {
        match self.cnav {
            Some(cnav) => Some(cnav.transmission_time_sow),
            None => self.stated.transmission_time_sow,
        }
    }

    /// Build a GPS LNAV record from decoded navigation-message subframes.
    ///
    /// This closes the `lnav::decode -> broadcast source` half of the real-time
    /// pipeline: feed [`crate::navigation::lnav::decode`]'s output here, collect
    /// the records into a `BroadcastStore`, and solve with
    /// [`solve_broadcast`](crate::positioning::solve_broadcast). The conversion
    /// matches the RINEX navigation parser's record exactly except for the inputs
    /// only the air interface carries:
    ///
    /// - The decoded angular elements are in semicircles (and semicircles/second)
    ///   as transmitted by GPS LNAV; they are scaled to the radians the
    ///   `crate::broadcast` evaluator expects (the harmonic `cuc..cis` terms are
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
    /// - The IODC is kept in [`StatedNavFields::orbit6_field4`] as a RINEX record
    ///   states it; the fit interval is written in hours.
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
            issue_of_data: Some(BroadcastIssue {
                issue: decoded.iode as u32,
                message: NavMessage::GpsLnav,
            }),
            week: full_week,
            toe,
            toc,
            elements,
            clock,
            group_delays: BroadcastGroupDelays::gps_lnav(decoded.tgd),
            cnav: None,
            sv_health: decoded.sv_health as f64,
            sv_accuracy_m: Some(sv_accuracy_m),
            fit_interval_s: Some(fit_interval_s),
            stated: StatedNavFields {
                orbit6_field4: Some(decoded.iodc as f64),
                orbit7_field2: Some(fit_interval_s / SECONDS_PER_HOUR),
                ..StatedNavFields::default()
            },
        })
    }
}

/// The nominal GPS user range accuracy (URA) value in meters for a 4-bit URA
/// index N (IS-GPS-200N Section 20.3.3.3.1.3). Each value is the upper bound of
/// the URA band the index represents. Index 15 carries no accuracy prediction
/// (the SV is not to be used for safe navigation) and has no usable meters
/// bound, so it returns `None` rather than a fabricated finite value.
pub(crate) fn gps_ura_index_to_meters(index: i64) -> Option<f64> {
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

const GPS_FIT_INTERVAL_6H_S: f64 = 6.0 * SECONDS_PER_HOUR;
const GPS_FIT_INTERVAL_8H_S: f64 = 8.0 * SECONDS_PER_HOUR;
const GPS_FIT_INTERVAL_14H_S: f64 = 14.0 * SECONDS_PER_HOUR;
const GPS_FIT_INTERVAL_26H_S: f64 = 26.0 * SECONDS_PER_HOUR;

/// Curve-fit interval (seconds) for a GPS LNAV record from its fit-interval flag
/// plus IODE/IODC, per IS-GPS-200N 20.3.3.4.3.1, 6.2.3, and Table 20-XII (the
/// table older revisions numbered 20-XI).
///
/// `flag = 0` is the nominal 4-hour fit. `flag = 1` is an extended fit: IODE
/// selects short-term extended operations (`IODE < 240`, a 6-hour fit) from
/// long-term extended operations (`IODE` in `240..=255`), and for the long-term
/// case the IODC range selects 8, 14, or 26 hours. Reserved IODC values and any
/// other flag/IODE/IODC combination are rejected.
pub(crate) fn gps_fit_interval_from_flag(
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

/// Why a RINEX NAV file, or one block of it, could not be read as the format lays it
/// out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavParseError {
    /// The header did not declare a supported RINEX navigation file.
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
    /// A non-blank body line that belongs to no record.
    UnexpectedLine {
        /// 1-based line number.
        line: usize,
    },
    /// A record followed by non-blank lines beyond its layout.
    ExtraRecordLines {
        /// The satellite whose record the lines follow.
        satellite: String,
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
            NavParseError::UnexpectedLine { line } => {
                write!(f, "line {line} belongs to no navigation record")
            }
            NavParseError::ExtraRecordLines { satellite } => write!(
                f,
                "navigation record for {satellite} is followed by lines beyond its layout"
            ),
        }
    }
}

impl std::error::Error for NavParseError {}

/// Diagnostic for a navigation block that lenient parsing could not decode or validate.
/// The lenient parsers retain the block's satellite token, its first line number, and the
/// [`NavParseError`] display text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedNavBlock {
    /// Satellite token associated with the failed block: v2/v3 trim the record's
    /// satellite field, while v4 copies the SV token from the frame marker.
    pub satellite: String,
    /// Display text of the [`NavParseError`] returned while parsing or validating the block.
    pub message: String,
    /// 1-based line number of the block's first line (the frame marker in RINEX 4).
    pub line: usize,
}

/// A departure from the format that a lenient reader read through: the record or
/// header value it concerns is kept, and the strict reader refuses it with
/// [`Self::error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavDiagnostic {
    /// 1-based line number of the record (or header line) concerned.
    pub line: usize,
    /// Satellite token of the record concerned; empty for a header line.
    pub satellite: String,
    /// The departure, as the strict reader reports it.
    pub error: NavParseError,
}

/// What a block that [`parse_nav_lenient`] read but does not return in
/// [`NavParse::records`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtherNavBlockKind {
    /// A GLONASS record, read by [`parse_glonass`].
    Glonass,
    /// An SBAS record, read by [`parse_sbas`].
    Sbas,
    /// A RINEX 4 system time offset frame (`> STO`), read by [`parse_nav_file`].
    SystemTimeOffset,
    /// A RINEX 4 Earth orientation frame (`> EOP`), read by [`parse_nav_file`].
    EarthOrientation,
    /// A RINEX 4 ionosphere frame (`> ION`), read by [`parse_iono_corrections`] and
    /// [`parse_nav_file`].
    Ionosphere,
    /// A message this crate recognizes and does not decode (BeiDou CNAV-1/2/3, NavIC
    /// L1); [`parse_nav_file`] keeps its text.
    NotDecoded,
}

/// A block that [`parse_nav_lenient`] read but does not return as a
/// [`BroadcastRecord`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtherNavBlock {
    /// 1-based line number of the block's first line.
    pub line: usize,
    /// Satellite token of the block.
    pub satellite: String,
    /// The RINEX 4 message token, where the block has one.
    pub message_token: Option<String>,
    /// What the block holds.
    pub kind: OtherNavBlockKind,
}

/// Result of lenient RINEX navigation parsing.
/// It keeps the successfully parsed Keplerian records, diagnostics for the blocks that
/// failed, the departures read through, and every other block of the file, so a caller
/// can tell a clean parse from one that left records out.
#[derive(Debug, Clone, PartialEq)]
pub struct NavParse {
    /// Successfully parsed Keplerian records, in input order. A malformed block is
    /// omitted here and reported in [`Self::skipped`].
    pub records: Vec<BroadcastRecord>,
    /// Diagnostics for Keplerian blocks, and blocks of no known system, whose parsing
    /// or marker validation returned a [`NavParseError`], and for body lines that
    /// belong to no record.
    pub skipped: Vec<SkippedNavBlock>,
    /// Departures from the format in the Keplerian records kept in [`Self::records`],
    /// and in the header.
    pub departures: Vec<NavDiagnostic>,
    /// Every other block of the file: GLONASS, SBAS and RINEX 4 non-ephemeris frames,
    /// and messages not decoded.
    pub other: Vec<OtherNavBlock>,
}

/// Parse a RINEX 2.xx, 3.xx or 4.xx navigation file into its Keplerian records:
/// GPS, QZSS, Galileo, BeiDou and NavIC.
///
/// A malformed Keplerian block, a departure from the format in one, or a non-blank
/// line that belongs to no record is an error; GLONASS, SBAS, RINEX 4 non-ephemeris
/// frames and messages not decoded are left to the other readers. The records are
/// returned in file order; selection by epoch, health, and message type is the
/// caller's job.
pub fn parse_nav(text: &str) -> Result<Vec<BroadcastRecord>, NavParseError> {
    let file = parse_nav_file(text)?;
    if let Some(error) = file.first_error(body::StrictScope::Keplerian) {
        return Err(error);
    }
    Ok(file.keplerian_records().collect())
}

/// Parse the Keplerian records, keeping every readable one.
///
/// Header failures remain fatal because no record boundaries are trustworthy
/// before the file type and version are known; a malformed optional header record
/// is reported in [`NavParse::departures`] instead. A malformed block is reported
/// in [`NavParse::skipped`] with its line number and reason.
pub fn parse_nav_lenient(text: &str) -> Result<NavParse, NavParseError> {
    Ok(parse_nav_file(text)?.nav_parse())
}

/// Parse a whole RINEX navigation file, keeping every header record and every
/// block with its text: decoded records, frames, messages this crate does not
/// decode, blocks that could not be read, and lines that belong to no record.
///
/// Header failures (no supported `RINEX VERSION / TYPE`, no `END OF HEADER`) are
/// errors. Everything after the header is read leniently: each entry carries the
/// departures read through, and a block that could not be read is kept as
/// [`NavItem::Undecoded`] with the reason. [`encode_nav_file`] writes the file back
/// byte for byte.
pub fn parse_nav_file(text: &str) -> Result<NavFile, NavParseError> {
    body::read_nav_file(text)
}

/// Parse the broadcast ionosphere coefficients from a RINEX header's
/// `IONOSPHERIC CORR` (version 2 `ION ALPHA`/`ION BETA`) records and the RINEX 4
/// body `> ION` frames: each header set is replaced by the frame of its system and model
/// transmitted latest, the rule [`BroadcastStore::iono_corrections`] applies, and
/// [`BroadcastStore::iono_corrections_at`] applies at an epoch.
///
/// A complete header label pair or body frame yields the coefficient set; a
/// missing label or frame yields `None` for that system. A malformed header row or
/// body frame is an error.
pub fn parse_iono_corrections(text: &str) -> Result<IonoCorrections, NavParseError> {
    let lines: Vec<&str> = text.lines().collect();
    let read = header::scan_header(&lines);
    if let Some(issue) = read
        .issues
        .iter()
        .find(|issue| issue.field == header::HeaderField::Ionosphere)
    {
        return Err(issue.error.clone());
    }
    let mut frames = body::body_ionosphere_frames(&lines[read.body_start..])?;
    frames::sort_by_transmission(&mut frames);
    Ok(frames::ionosphere_in_effect(
        read.header.iono,
        &frames,
        None,
    ))
}

/// The leap-second count (GPS − UTC) from the header's `LEAP SECONDS` record; `None`
/// if the record is absent. A malformed record is an error. The header is read up to
/// `END OF HEADER` (or to the end of the text when there is none).
///
/// The value is metadata: [`GlonassRecord::toe_gpst_j2000_s`] converts GLONASS epochs
/// with the leap-second table at each epoch, as RTKLIB does, and does not read it.
pub fn parse_leap_seconds(text: &str) -> Result<Option<f64>, NavParseError> {
    let lines: Vec<&str> = text.lines().collect();
    let read = header::scan_header(&lines);
    if let Some(issue) = read
        .issues
        .iter()
        .find(|issue| issue.field == header::HeaderField::LeapSeconds)
    {
        return Err(issue.error.clone());
    }
    Ok(read
        .header
        .leap_seconds
        .as_ref()
        .map(|leap| leap.current as f64))
}

/// Parse all GLONASS (`R`) records from a RINEX 2 (`G` file), 3 or 4 navigation file,
/// in file order; selection is the caller's job. A malformed record of a
/// representable slot, or a departure from the format in one, is a
/// [`NavParseError`], but a record whose slot token is not a representable satellite
/// id (`R00`, or a malformed slot field) is skipped rather than rejecting the whole
/// file.
pub fn parse_glonass(text: &str) -> Result<Vec<GlonassRecord>, NavParseError> {
    let file = parse_nav_file(text)?;
    if let Some(error) = file.first_error(body::StrictScope::Glonass) {
        return Err(error);
    }
    Ok(file.glonass_parse().records)
}

/// Like [`parse_glonass`], but keeps every readable record and reports the rest:
/// slots whose token is not representable as a [`GnssSatelliteId`] (`R00`, or a
/// malformed slot field) in [`GlonassParse::skipped`], records that could not be read
/// in [`GlonassParse::invalid`], and records kept through a departure from the format
/// in [`GlonassParse::departures`], each with its line number.
pub fn parse_glonass_lenient(text: &str) -> Result<GlonassParse, NavParseError> {
    Ok(parse_nav_file(text)?.glonass_parse())
}

/// Parse all SBAS (`S`) records from a RINEX 2 (`H` file), 3 or 4 navigation file, in
/// file order. A malformed SBAS record, or a departure from the format in one, is an
/// error.
pub fn parse_sbas(text: &str) -> Result<Vec<SbasRecord>, NavParseError> {
    let file = parse_nav_file(text)?;
    if let Some(error) = file.first_error(body::StrictScope::Sbas) {
        return Err(error);
    }
    Ok(file
        .entries
        .iter()
        .filter_map(|entry| match &entry.item {
            NavItem::Sbas(record) => Some(*record),
            _ => None,
        })
        .collect())
}

fn nav_block_satellite(block: &[&str]) -> String {
    block
        .first()
        .and_then(|line| line.get(0..3))
        .unwrap_or("")
        .trim()
        .to_string()
}

pub(crate) enum V4MarkerHeader<'a> {
    Eph { sv: &'a str, msg_token: &'a str },
    RecognizedNonEph,
}

pub(crate) fn parse_v4_eph_marker<'a>(
    marker: &'a str,
    body: &[&str],
) -> Result<V4MarkerHeader<'a>, NavParseError> {
    let rest = marker.strip_prefix('>').unwrap_or(marker).trim();
    let body_sat = nav_block_satellite(body);
    if rest.is_empty() {
        return Err(NavParseError::BadField {
            satellite: body_sat,
            field: "frame marker",
        });
    }
    let mut fields = rest.split_whitespace().peekable();
    let Some(frame_type) = fields.next() else {
        return Err(NavParseError::BadField {
            satellite: body_sat,
            field: "frame marker",
        });
    };
    if matches!(frame_type, "ION" | "STO" | "EOP") {
        return Ok(V4MarkerHeader::RecognizedNonEph);
    }
    if frame_type != "EPH" {
        return Err(NavParseError::BadField {
            satellite: body_sat,
            field: "frame marker",
        });
    }
    let Some(first_sv) = fields.next() else {
        return Err(NavParseError::BadField {
            satellite: body_sat,
            field: "prn",
        });
    };
    let sv = if first_sv.len() == 1 {
        let first_char = first_sv.chars().next().unwrap_or(' ');
        if !first_char.is_ascii_alphabetic() {
            return Err(NavParseError::BadField {
                satellite: first_sv.to_string(),
                field: "system",
            });
        }
        let Some(&prn_token) = fields.peek() else {
            return Err(NavParseError::BadField {
                satellite: first_sv.to_string(),
                field: "prn",
            });
        };
        if !prn_token.bytes().all(|b| b.is_ascii_digit()) {
            return Err(NavParseError::BadField {
                satellite: first_sv.to_string(),
                field: "prn",
            });
        }
        let _ = fields.next();
        let start = first_sv.as_ptr() as usize - marker.as_ptr() as usize;
        let end = prn_token.as_ptr() as usize - marker.as_ptr() as usize + prn_token.len();
        let combined = &marker[start..end];
        if !(1..=2).contains(&prn_token.len()) {
            return Err(NavParseError::BadField {
                satellite: combined.to_string(),
                field: "prn",
            });
        }
        combined
    } else {
        let first_char = first_sv.chars().next().unwrap_or(' ');
        if !first_char.is_ascii_alphabetic() {
            return Err(NavParseError::BadField {
                satellite: first_sv.to_string(),
                field: "system",
            });
        }
        let prn_part = &first_sv[first_char.len_utf8()..];
        if !prn_part.bytes().all(|b| b.is_ascii_digit()) || !(1..=2).contains(&prn_part.len()) {
            return Err(NavParseError::BadField {
                satellite: first_sv.to_string(),
                field: "prn",
            });
        }
        first_sv
    };

    let Some(msg_token) = fields.next() else {
        return Err(NavParseError::BadField {
            satellite: sv.to_string(),
            field: "message",
        });
    };

    if fields.next().is_some() {
        return Err(NavParseError::BadField {
            satellite: sv.to_string(),
            field: "frame marker",
        });
    }

    Ok(V4MarkerHeader::Eph { sv, msg_token })
}

/// Split a version-4 frame marker `> EPH G01 LNAV` into (frame type, SV, message
/// token), or `None` if it is malformed. Mirrors the RINEX-4 marker layout:
/// `>` then the 4-column frame class, the SV, and the message-type token.
pub(crate) fn parse_v4_marker(line: &str) -> Option<(&str, &str, &str)> {
    let rest = line.strip_prefix('>')?;
    let mut fields = rest.split_whitespace();
    let frame_type = fields.next()?;
    let first_sv = fields.next()?;
    // Non-padded satellite identifiers (e.g. "G 1") have whitespace between constellation
    // and PRN, producing separate whitespace tokens that must be recombined.
    let (sv, msg_token) = if first_sv.len() == 1
        && first_sv
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
    {
        let mut peek = fields.clone();
        let prn_token = peek.next()?;
        if prn_token.bytes().all(|b| b.is_ascii_digit()) && (1..=2).contains(&prn_token.len()) {
            let _ = fields.next();
            let msg_token = fields.next()?;
            let start = first_sv.as_ptr() as usize - line.as_ptr() as usize;
            let end = prn_token.as_ptr() as usize - line.as_ptr() as usize + prn_token.len();
            (&line[start..end], msg_token)
        } else {
            let msg_token = fields.next()?;
            (first_sv, msg_token)
        }
    } else {
        let msg_token = fields.next()?;
        (first_sv, msg_token)
    };
    Some((frame_type, sv, msg_token))
}

/// Map a version-4 EPH message token to the [`NavMessage`] for the decoded
/// Keplerian messages. `CNV2` is system-overloaded: GPS/QZSS CNV2 is decoded
/// as CNAV-2, while BeiDou CNV2 is a different roster and is not decoded.
pub(crate) fn nav_message_from_v4_token(token: &str, system: GnssSystem) -> Option<NavMessage> {
    match (token, system) {
        ("LNAV", GnssSystem::Gps) => Some(NavMessage::GpsLnav),
        ("CNAV", GnssSystem::Gps) => Some(NavMessage::GpsCnav),
        ("CNV2", GnssSystem::Gps) => Some(NavMessage::GpsCnav2),
        ("LNAV", GnssSystem::Qzss) => Some(NavMessage::QzssLnav),
        ("CNAV", GnssSystem::Qzss) => Some(NavMessage::QzssCnav),
        ("CNV2", GnssSystem::Qzss) => Some(NavMessage::QzssCnav2),
        ("INAV", GnssSystem::Galileo) => Some(NavMessage::GalileoInav),
        ("FNAV", GnssSystem::Galileo) => Some(NavMessage::GalileoFnav),
        ("D1", GnssSystem::BeiDou) => Some(NavMessage::BeidouD1),
        ("D2", GnssSystem::BeiDou) => Some(NavMessage::BeidouD2),
        ("LNAV", GnssSystem::Navic) => Some(NavMessage::NavicLnav),
        _ => None,
    }
}

pub(crate) fn satellites_match(marker_sv: &str, body_sv: &str) -> bool {
    match (
        marker_sv.parse::<GnssSatelliteId>(),
        body_sv.parse::<GnssSatelliteId>(),
    ) {
        (Ok(marker), Ok(body)) => marker == body,
        _ => marker_sv == body_sv,
    }
}

pub(crate) fn validate_v4_ephemeris_marker(
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

    if !satellites_match(marker_sv, body_sv) {
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
            | (NavMessage::GpsCnav | NavMessage::GpsCnav2, GnssSystem::Gps)
            | (NavMessage::QzssLnav, GnssSystem::Qzss)
            | (
                NavMessage::QzssCnav | NavMessage::QzssCnav2,
                GnssSystem::Qzss,
            )
            | (
                NavMessage::GalileoInav | NavMessage::GalileoFnav,
                GnssSystem::Galileo,
            )
            | (
                NavMessage::BeidouD1 | NavMessage::BeidouD2,
                GnssSystem::BeiDou,
            )
            | (NavMessage::NavicLnav, GnssSystem::Navic)
    )
}

/// Parse a RINEX version field (`F9.2`, columns 1-9), majors 2, 3 and 4.
fn parse_rinex_version(version: &str) -> Option<NavVersion> {
    let (major, minor) = version.split_once('.')?;
    let major = major.trim().parse::<u8>().ok()?;
    if !matches!(major, 2..=4) {
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
    Some(NavVersion { major, minor })
}

/// Whether a line starts a RINEX 3 (or lettered RINEX 2) record: a system letter, then
/// a PRN of one or two digits in columns 2-3, the one-digit form written `G 1` with a
/// space in column 2, as RTKLIB's `satid2no` reads it.
fn is_record_start(line: &str) -> bool {
    let Some(token) = line.as_bytes().get(..3) else {
        return false;
    };
    token[0].is_ascii_alphabetic()
        && (token[1].is_ascii_digit() || token[1] == b' ')
        && token[2].is_ascii_digit()
}

/// Whether a line starts a RINEX 2 record with a numeric `I2` PRN: a PRN in columns
/// 1-2, a blank column 3, and a two-digit year in columns 4-5.
fn is_v2_numeric_record_start(line: &str) -> bool {
    let bytes = line.as_bytes();
    bytes.len() >= 5
        && (bytes[0] == b' ' || bytes[0].is_ascii_digit())
        && bytes[1].is_ascii_digit()
        && bytes[2] == b' '
        && (bytes[3] == b' ' || bytes[3].is_ascii_digit())
        && bytes[4].is_ascii_digit()
}

#[cfg(test)]
mod panic_regression_tests {
    use super::{is_record_start, is_v2_numeric_record_start};

    #[test]
    fn malformed_utf8_replacement_does_not_panic_record_probe() {
        assert!(!is_record_start("G\u{FFFD}"));
        assert!(!is_v2_numeric_record_start("1\u{FFFD}"));
    }
}

/// Column layout of a record's lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// RINEX 3 and 4, and lettered RINEX 2 extensions: a four-column prefix, the epoch
    /// in columns 5-23, fields of 19 columns from column 24 on the first line and from
    /// column 5 on the others.
    V3,
    /// RINEX 2 with a numeric PRN: a three-column prefix, the epoch in columns 4-22
    /// with a two-digit year and `F5.1` seconds, fields from column 23 on the first
    /// line and from column 4 on the others.
    V2,
    /// Lettered RINEX 2 extension records: the version 3 columns with the epoch read
    /// as six numbers, as RTKLIB `str2time` reads it, a two-digit year allowed.
    V2Lettered,
}

impl Layout {
    const fn first_line_fields(self) -> [(usize, usize); 3] {
        match self {
            Layout::V2 => [(22, 41), (41, 60), (60, 79)],
            Layout::V3 | Layout::V2Lettered => [(23, 42), (42, 61), (61, 80)],
        }
    }

    const fn orbit_fields(self) -> [(usize, usize); 4] {
        match self {
            Layout::V2 => [(3, 22), (22, 41), (41, 60), (60, 79)],
            Layout::V3 | Layout::V2Lettered => [(4, 23), (23, 42), (42, 61), (61, 80)],
        }
    }
}

/// A calendar epoch as a navigation record or frame states it, in the time scale of its
/// system (GPS, Galileo, QZSS and NavIC time; BDT; UTC for GLONASS).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavEpoch {
    /// Year (four digits; a two-digit RINEX 2 year is read with the 1980 pivot).
    pub year: i32,
    /// Month, 1-12.
    pub month: u8,
    /// Day of month.
    pub day: u8,
    /// Hour.
    pub hour: u8,
    /// Minute.
    pub minute: u8,
    /// Seconds, with any fraction the record states.
    pub second: f64,
}

impl NavEpoch {
    /// Seconds since J2000 in the epoch's own scale.
    pub fn j2000_s(&self) -> f64 {
        civil::j2000_seconds(
            self.year,
            i32::from(self.month),
            i32::from(self.day),
            i32::from(self.hour),
            i32::from(self.minute),
            self.second,
        )
    }

    /// The calendar epoch of `j2000_s` seconds since J2000 (whole days and seconds of
    /// day computed exactly, the fraction kept).
    pub(crate) fn from_j2000_s(j2000_s: f64) -> Self {
        // J2000 is 2000-01-01 12:00, so midnight-based seconds are `j2000_s + 43200`.
        let from_midnight = j2000_s + 43_200.0;
        let days = (from_midnight / 86_400.0).floor();
        let second_of_day = from_midnight - days * 86_400.0;
        let jdn = 2_451_545_i64 + days as i64;
        let (year, month, day) = civil::civil_from_julian_day_number(jdn);
        let whole = second_of_day.floor();
        let fraction = second_of_day - whole;
        let whole = whole as i64;
        Self {
            year: year as i32,
            month: month as u8,
            day: day as u8,
            hour: (whole / 3600) as u8,
            minute: ((whole % 3600) / 60) as u8,
            second: (whole % 60) as f64 + fraction,
        }
    }
}

/// Read the epoch of a record's first line in `layout`.
fn parse_record_epoch(
    l0: &str,
    layout: Layout,
    sat: &str,
    field_name: &'static str,
    policy: validate::CivilSecondPolicy,
) -> Result<NavEpoch, NavParseError> {
    let bad = || NavParseError::BadField {
        satellite: sat.to_string(),
        field: field_name,
    };
    let (year, month, day, hour, minute, second) = match layout {
        Layout::V3 => (
            strict_record_int::<i64>(l0, 4, 8, field_name, sat)?,
            strict_record_int::<i64>(l0, 9, 11, field_name, sat)?,
            strict_record_int::<i64>(l0, 12, 14, field_name, sat)?,
            strict_record_int::<i64>(l0, 15, 17, field_name, sat)?,
            strict_record_int::<i64>(l0, 18, 20, field_name, sat)?,
            strict_record_int::<i64>(l0, 21, 23, field_name, sat)? as f64,
        ),
        Layout::V2 => (
            two_digit_year(strict_record_int::<i64>(l0, 3, 5, field_name, sat)?),
            strict_record_int::<i64>(l0, 6, 8, field_name, sat)?,
            strict_record_int::<i64>(l0, 9, 11, field_name, sat)?,
            strict_record_int::<i64>(l0, 12, 14, field_name, sat)?,
            strict_record_int::<i64>(l0, 15, 17, field_name, sat)?,
            validate::strict_f64(raw_field(l0, 17, 22), field_name)
                .map_err(|error| map_record_field_error(error, sat))?,
        ),
        Layout::V2Lettered => {
            let tokens: Vec<&str> = raw_field(l0, 4, 23).split_whitespace().collect();
            if tokens.len() != 6 {
                return Err(bad());
            }
            let int = |token: &str| token.parse::<i64>().map_err(|_| bad());
            let year = int(tokens[0])?;
            (
                if year < 100 {
                    two_digit_year(year)
                } else {
                    year
                },
                int(tokens[1])?,
                int(tokens[2])?,
                int(tokens[3])?,
                int(tokens[4])?,
                validate::strict_f64(tokens[5], field_name)
                    .map_err(|error| map_record_field_error(error, sat))?,
            )
        }
    };
    let civil =
        validate::civil_datetime_with_second_policy(year, month, day, hour, minute, second, policy)
            .map_err(|_| bad())?;
    Ok(NavEpoch {
        year: i32::try_from(civil.year).map_err(|_| bad())?,
        month: civil.month as u8,
        day: civil.day as u8,
        hour: civil.hour as u8,
        minute: civil.minute as u8,
        second: civil.second,
    })
}

/// A two-digit RINEX 2 year with RTKLIB `str2time`'s pivot: below 80 is 20xx.
fn two_digit_year(year: i64) -> i64 {
    if year < 80 {
        year + 2000
    } else {
        year + 1900
    }
}

/// The clock reference epoch of a record's first line as week and seconds of week in
/// the record's broadcast time scale.
fn epoch_week_tow(
    epoch: &NavEpoch,
    time_scale: TimeScale,
    sat: &str,
    field_name: &'static str,
) -> Result<(u32, f64), NavParseError> {
    let bad = || NavParseError::BadField {
        satellite: sat.to_string(),
        field: field_name,
    };
    let year = i64::from(epoch.year);
    let month = i64::from(epoch.month);
    let day = i64::from(epoch.day);
    let week = gnss::week_from_calendar(time_scale, year, month, day).ok_or_else(bad)?;
    let whole_second = epoch.second.floor();
    let sow = gnss::seconds_of_week_from_calendar(
        year,
        month,
        day,
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        whole_second as i64,
    )
    .ok_or_else(bad)?;
    Ok((week, sow + (epoch.second - whole_second)))
}

/// The numeric fields of a record's lines in RTKLIB's `data[]` order: three from the
/// first line, then four from each following line, `None` for a blank or unreadable
/// field.
fn record_fields(lines: &[&str], layout: Layout) -> Vec<Option<f64>> {
    let mut data = Vec::with_capacity(3 + 4 * lines.len().saturating_sub(1));
    if let Some(l0) = lines.first() {
        for (start, end) in layout.first_line_fields() {
            data.push(parse_f64(l0, start, end));
        }
    }
    for line in lines.iter().skip(1) {
        for (start, end) in layout.orbit_fields() {
            data.push(parse_f64(line, start, end));
        }
    }
    data
}

/// The raw text of field `index` in RTKLIB's `data[]` order.
fn raw_record_field<'a>(lines: &[&'a str], layout: Layout, index: usize) -> &'a str {
    if index < 3 {
        let (start, end) = layout.first_line_fields()[index];
        return lines.first().map_or("", |line| raw_field(line, start, end));
    }
    let line_index = 1 + (index - 3) / 4;
    let (start, end) = layout.orbit_fields()[(index - 3) % 4];
    lines
        .get(line_index)
        .map_or("", |line| raw_field(line, start, end))
}

/// A record decoded from its lines, with the departures from the format read through.
pub(crate) struct Decoded<T> {
    pub(crate) value: T,
    pub(crate) departures: Vec<NavParseError>,
}

/// Read an optional field the models do not use: blank is `None`; a value that does not
/// read, or cannot be written back in the `D19.12` field, is `None` with a departure.
fn stated_field(
    raw: &str,
    field: &'static str,
    sat: &str,
    departures: &mut Vec<NavParseError>,
) -> Option<f64> {
    if raw.trim().is_empty() {
        return None;
    }
    match validate::strict_f64(raw, field) {
        Ok(value) if write::d19_12_representable(value) => Some(value),
        _ => {
            departures.push(NavParseError::BadField {
                satellite: sat.to_string(),
                field,
            });
            None
        }
    }
}

/// Lines past `expected` in a block: a non-blank one is a departure.
fn check_extra_lines(
    lines: &[&str],
    expected: usize,
    sat: &str,
    departures: &mut Vec<NavParseError>,
) {
    if lines
        .iter()
        .skip(expected)
        .any(|line| !line.trim().is_empty())
    {
        departures.push(NavParseError::ExtraRecordLines {
            satellite: sat.to_string(),
        });
    }
}

/// Move `toe`'s week so it lies within half a week of `toc`, as RTKLIB `adjweek`
/// places `toe` relative to `toc`.
fn adjust_week_to(week: u32, tow_s: f64, toc: GnssWeekTow) -> u32 {
    let dt = (f64::from(week) - f64::from(toc.week)) * SECONDS_PER_WEEK + (tow_s - toc.tow_s);
    if dt < -crate::constants::HALF_WEEK_S {
        week.saturating_add(1)
    } else if dt > crate::constants::HALF_WEEK_S {
        week.saturating_sub(1)
    } else {
        week
    }
}

/// Decode a legacy Keplerian record (GPS/QZSS LNAV, Galileo I/NAV and F/NAV, BeiDou
/// D1/D2, NavIC LNAV) from its eight lines.
pub(crate) fn parse_keplerian_block(
    lines: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    message_override: Option<NavMessage>,
    version: NavVersion,
    layout: Layout,
) -> Result<Decoded<BroadcastRecord>, NavParseError> {
    if lines.len() < 8 {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = |what: &'static str| NavParseError::BadField {
        satellite: sat.to_string(),
        field: what,
    };
    let mut departures = Vec::new();
    check_extra_lines(lines, 8, sat, &mut departures);
    let system = satellite_id.system;
    let l0 = lines[0];

    // Clock line: epoch (-> toc) and the af0/af1/af2 polynomial.
    let time_scale = broadcast_time_scale(system);
    let epoch = parse_record_epoch(
        l0,
        layout,
        sat,
        "toc epoch",
        validate::CivilSecondPolicy::Continuous,
    )?;
    let (toc_week, toc_sow) = epoch_week_tow(&epoch, time_scale, sat, "toc epoch")?;
    let data = record_fields(&lines[..8], layout);
    let g = |index: usize, what: &'static str| data[index].ok_or_else(|| bad(what));

    let elements = KeplerianElements {
        crs: g(4, "crs")?,
        delta_n: g(5, "deltaN")?,
        m0: g(6, "m0")?,
        cuc: g(7, "cuc")?,
        e: g(8, "e")?,
        cus: g(9, "cus")?,
        sqrt_a: g(10, "sqrtA")?,
        toe_sow: g(11, "toe")?,
        cic: g(12, "cic")?,
        omega0: g(13, "omega0")?,
        cis: g(14, "cis")?,
        i0: g(15, "i0")?,
        crc: g(16, "crc")?,
        omega: g(17, "omega")?,
        omega_dot: g(18, "omegaDot")?,
        idot: g(19, "idot")?,
    };
    let clock = ClockPolynomial {
        af0: g(0, "af0")?,
        af1: g(1, "af1")?,
        af2: g(2, "af2")?,
        toc_sow,
    };

    let week = finite_integral_u32(g(21, "week")?, "week", sat)?;
    let toc = GnssWeekTow::new(time_scale, toc_week, clock.toc_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("toc"))?;
    let toe_unadjusted = GnssWeekTow::new(time_scale, week, elements.toe_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("toe"))?;
    let toe = GnssWeekTow {
        week: adjust_week_to(toe_unadjusted.week, toe_unadjusted.tow_s, toc),
        ..toe_unadjusted
    };

    let raw = |index: usize| raw_record_field(lines, layout, index);
    let mut stated = StatedNavFields {
        orbit5_field2: stated_field(raw(20), "orbit-5 field 2", sat, &mut departures),
        orbit5_field4: stated_field(raw(22), "orbit-5 field 4", sat, &mut departures),
        orbit6_field4: None,
        transmission_time_sow: stated_field(raw(27), "transmission time", sat, &mut departures),
        orbit7_field2: None,
        orbit7_field3: stated_field(raw(29), "orbit-7 field 3", sat, &mut departures),
        orbit7_field4: stated_field(raw(30), "orbit-7 field 4", sat, &mut departures),
    };

    let message = match message_override {
        Some(message) => message,
        None => match system {
            GnssSystem::Galileo => {
                let word = stated.orbit5_field2.ok_or_else(|| bad("data sources"))?;
                let (message, forbidden) = galileo_message(word, sat)?;
                if forbidden {
                    departures.push(bad("data sources"));
                }
                message
            }
            GnssSystem::BeiDou => {
                if is_beidou_geo(satellite_id) {
                    NavMessage::BeidouD2
                } else {
                    NavMessage::BeidouD1
                }
            }
            GnssSystem::Qzss => NavMessage::QzssLnav,
            GnssSystem::Navic => NavMessage::NavicLnav,
            _ => NavMessage::GpsLnav,
        },
    };
    let issue_of_data = Some(BroadcastIssue {
        issue: finite_integral_u32(g(3, "issue of data")?, "issue of data", sat)?,
        message,
    });

    // A blank or unreadable accuracy is not known: the record keeps its orbit and clock,
    // and the departure is reported.
    let sv_accuracy_m = data[23];
    if sv_accuracy_m.is_none() {
        departures.push(bad("accuracy"));
    }
    let sv_health = g(24, "health")?;
    let group_delays = match system {
        GnssSystem::Gps => {
            BroadcastGroupDelays::gps_lnav_opt(optional_keplerian_delay(raw(25), "gps tgd", sat)?)
        }
        GnssSystem::Qzss => {
            BroadcastGroupDelays::gps_lnav_opt(optional_keplerian_delay(raw(25), "qzss tgd", sat)?)
        }
        GnssSystem::Navic => {
            BroadcastGroupDelays::gps_lnav_opt(optional_keplerian_delay(raw(25), "navic tgd", sat)?)
        }
        // RINEX Galileo ORBIT-6 carries BGD E5a/E1 in field 3 and BGD E5b/E1 in
        // field 4; both are part of the message representation regardless of
        // which one a clock consumer later selects.
        GnssSystem::Galileo => BroadcastGroupDelays::galileo_opt(
            optional_keplerian_delay(raw(25), "bgd e5a/e1", sat)?,
            optional_keplerian_delay(raw(26), "bgd e5b/e1", sat)?,
        ),
        GnssSystem::BeiDou => BroadcastGroupDelays::beidou_opt(
            optional_keplerian_delay(raw(25), "beidou tgd1", sat)?,
            optional_keplerian_delay(raw(26), "beidou tgd2", sat)?,
        ),
        _ => BroadcastGroupDelays::default(),
    };
    if matches!(
        system,
        GnssSystem::Gps | GnssSystem::Qzss | GnssSystem::Navic
    ) {
        stated.orbit6_field4 = stated_field(raw(26), "iodc", sat, &mut departures);
    }

    // ORBIT-7 field 2: the GPS fit interval, the QZSS fit flag, the BeiDou AODC.
    let fit_interval_s = match system {
        GnssSystem::Gps => {
            let (start, end) = layout.orbit_fields()[1];
            stated.orbit7_field2 = parse_f64(lines[7], start, end);
            // An unreadable or negative fit field states no fit interval; the record
            // keeps its orbit and clock, and the departure is reported.
            gps_fit_interval_s(lines[7], layout, version).unwrap_or_else(|()| {
                departures.push(bad("fit interval"));
                None
            })
        }
        GnssSystem::Qzss => {
            stated.orbit7_field2 = stated_field(raw(28), "fit interval", sat, &mut departures);
            stated.orbit7_field2.map(qzss_fit_interval_s)
        }
        _ => {
            stated.orbit7_field2 = stated_field(raw(28), "orbit-7 field 2", sat, &mut departures);
            None
        }
    };

    Ok(Decoded {
        value: BroadcastRecord {
            satellite_id,
            message,
            issue_of_data,
            week,
            toe,
            toc,
            elements,
            clock,
            group_delays,
            cnav: None,
            sv_health,
            sv_accuracy_m,
            fit_interval_s,
            stated,
        },
        departures,
    })
}

/// The QZSS fit interval of a fit flag, as RTKLIB `decode_eph` reads it: 0 is two
/// hours, anything else four.
pub(crate) fn qzss_fit_interval_s(flag: f64) -> f64 {
    if flag == 0.0 {
        QZSS_SHORT_FIT_INTERVAL_S
    } else {
        QZSS_LONG_FIT_INTERVAL_S
    }
}

/// Decode a GPS/QZSS CNAV (9 lines) or CNAV-2 (10 lines) RINEX 4 record.
pub(crate) fn parse_cnav_block(
    block: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    message: NavMessage,
) -> Result<Decoded<BroadcastRecord>, NavParseError> {
    let is_cnav2 = matches!(message, NavMessage::GpsCnav2 | NavMessage::QzssCnav2);
    let required_lines = if is_cnav2 { 10 } else { 9 };
    if block.len() < required_lines {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = |what: &'static str| NavParseError::BadField {
        satellite: sat.to_string(),
        field: what,
    };
    let mut departures = Vec::new();
    check_extra_lines(block, required_lines, sat, &mut departures);
    let l0 = block[0];
    let epoch = parse_record_epoch(
        l0,
        Layout::V3,
        sat,
        "toc epoch",
        validate::CivilSecondPolicy::Continuous,
    )?;
    let (toc_week, toc_sow) = epoch_week_tow(&epoch, TimeScale::Gpst, sat, "toc epoch")?;
    let af0 = parse_f64(l0, 23, 42).ok_or_else(|| bad("af0"))?;
    let af1 = parse_f64(l0, 42, 61).ok_or_else(|| bad("af1"))?;
    let af2 = parse_f64(l0, 61, 80).ok_or_else(|| bad("af2"))?;

    let o1 = orbit_row(block[1]);
    let o2 = orbit_row(block[2]);
    let o3 = orbit_row(block[3]);
    let o4 = orbit_row(block[4]);
    let o5 = orbit_row(block[5]);
    let o6 = orbit_row(block[6]);
    let o8 = orbit_row(block[8]);
    let o9 = if is_cnav2 {
        Some(orbit_row(block[9]))
    } else {
        None
    };
    let cnav2_fields = o9.unwrap_or([None; 4]);

    let g = |v: Option<f64>, what: &'static str| v.ok_or_else(|| bad(what));
    let elements = KeplerianElements {
        crs: g(o1[1], "crs")?,
        delta_n: g(o1[2], "deltaN0")?,
        m0: g(o1[3], "m0")?,
        cuc: g(o2[0], "cuc")?,
        e: g(o2[1], "e")?,
        cus: g(o2[2], "cus")?,
        sqrt_a: g(o2[3], "sqrtA0")?,
        toe_sow: toc_sow,
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

    let week = toc_week;
    let toe = GnssWeekTow::new(TimeScale::Gpst, week, elements.toe_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("toe"))?;
    let toc = GnssWeekTow::new(TimeScale::Gpst, week, clock.toc_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("toc"))?;
    let wn_op = finite_integral_u32(
        g(if is_cnav2 { cnav2_fields[1] } else { o8[1] }, "wn_op")?,
        "wn_op",
        sat,
    )?;
    let top_sow = g(o3[0], "top")?;
    let top = GnssWeekTow::new(TimeScale::Gpst, wn_op, top_sow)
        .and_then(GnssWeekTow::normalized)
        .map_err(|_| bad("top"))?;
    let ura_ed_index = finite_integral_i8(g(o6[0], "ura_ed")?, "ura_ed", -16, 15, sat)?;
    let ura_ned0_index = finite_integral_i8(g(o5[2], "ura_ned0")?, "ura_ned0", -16, 15, sat)?;
    let ura_ned1_index = finite_integral_u8(g(o5[3], "ura_ned1")?, "ura_ned1", 0, 7, sat)?;
    let ura_ned2_index = finite_integral_u8(g(o6[3], "ura_ned2")?, "ura_ned2", 0, 7, sat)?;
    let health_max = if is_cnav2 { 1 } else { 7 };
    let sv_health = f64::from(finite_integral_u8(
        g(o6[1], "health")?,
        "health",
        0,
        health_max,
        sat,
    )?);
    let transmission_time_sow = g(if is_cnav2 { cnav2_fields[0] } else { o8[0] }, "t_tm")?;
    let flags = optional_integral_u32(
        if is_cnav2 {
            raw_orbit_field(block[9], 2)
        } else {
            raw_orbit_field(block[8], 2)
        },
        "flags",
        sat,
    )?;

    let tgd = optional_cnav_delay(raw_orbit_field(block[6], 2), "tgd", sat)?;
    let isc_l1ca = optional_cnav_delay(raw_orbit_field(block[7], 0), "isc_l1ca", sat)?;
    let isc_l2c = optional_cnav_delay(raw_orbit_field(block[7], 1), "isc_l2c", sat)?;
    let isc_l5i5 = optional_cnav_delay(raw_orbit_field(block[7], 2), "isc_l5i5", sat)?;
    let isc_l5q5 = optional_cnav_delay(raw_orbit_field(block[7], 3), "isc_l5q5", sat)?;
    let (isc_l1cd, isc_l1cp) = if is_cnav2 {
        (
            optional_cnav_delay(raw_orbit_field(block[8], 0), "isc_l1cd", sat)?,
            optional_cnav_delay(raw_orbit_field(block[8], 1), "isc_l1cp", sat)?,
        )
    } else {
        (None, None)
    };

    let cnav = CnavParameters {
        adot_m_s: g(o1[0], "adot")?,
        delta_n0_dot_rad_s2: g(o5[1], "deltaN0Dot")?,
        top,
        ura_ed_index,
        ura_ned0_index,
        ura_ned1_index,
        ura_ned2_index,
        transmission_time_sow,
        flags,
    };

    Ok(Decoded {
        value: BroadcastRecord {
            satellite_id,
            message,
            // The RINEX 4 CNAV and CNAV-2 records carry no issue of data.
            issue_of_data: None,
            week,
            toe,
            toc,
            elements,
            clock,
            group_delays: BroadcastGroupDelays::cnav(
                tgd, isc_l1ca, isc_l2c, isc_l5i5, isc_l5q5, isc_l1cd, isc_l1cp,
            ),
            cnav: Some(cnav),
            sv_health,
            sv_accuracy_m: cnav_ura_nominal_m(ura_ed_index),
            // The RINEX 4 CNAV and CNAV-2 records state no fit interval.
            fit_interval_s: None,
            stated: StatedNavFields::default(),
        },
        departures,
    })
}

/// The GPS curve-fit interval in seconds from the ORBIT-7 fit-interval field.
///
/// RINEX 2 and RINEX 3.03 Table A6 specify ORBIT-7 field 2 in hours, RINEX 2.11
/// with zero for an unknown interval. Per RINEX 3.03 Section 6.6, unknown or
/// unmodeled fields are blank. A blank or absent field indicates an unknown or
/// unprovided fit interval and decodes as `Ok(None)`.
///
/// Numeric zero (`0.0`) follows the missing-value convention for unpopulated
/// fields (rather than representing a physical zero-length interval or an
/// unsourced 4-hour broadcast interval) and decodes as `Ok(None)`. Explicit
/// positive values decode as hours (`value * SECONDS_PER_HOUR`).
///
/// For RINEX 3.00-3.02 the field is read as the flag `0 = 4 hours`, `1 = 6 hours`
/// (extended fit): `0.0` decodes as nominal 4 hours
/// ([`GPS_NOMINAL_FIT_INTERVAL_S`]), and `1.0` decodes as extended fit
/// ([`GPS_LEGACY_EXTENDED_FIT_INTERVAL_S`]).
///
/// A present but non-numeric, negative, or unrepresentable field is an error (`Err(())`).
fn gps_fit_interval_s(
    orbit7: &str,
    layout: Layout,
    version: NavVersion,
) -> Result<Option<f64>, ()> {
    let (start, end) = layout.orbit_fields()[1];
    if field(orbit7, start, end).is_none() {
        return Ok(None);
    }
    let value = parse_f64(orbit7, start, end).ok_or(())?;
    if value < 0.0 {
        return Err(());
    }
    Ok(gps_fit_interval_from_value(value, version))
}

/// The GPS fit interval a non-negative ORBIT-7 field 2 value states in `version`
/// (see [`gps_fit_interval_s`]).
pub(crate) fn gps_fit_interval_from_value(value: f64, version: NavVersion) -> Option<f64> {
    if value == 0.0 {
        if version.gps_fit_interval_uses_legacy_flag() {
            Some(GPS_NOMINAL_FIT_INTERVAL_S)
        } else {
            None
        }
    } else if version.gps_fit_interval_uses_legacy_flag() && value == 1.0 {
        Some(GPS_LEGACY_EXTENDED_FIT_INTERVAL_S)
    } else {
        Some(value * SECONDS_PER_HOUR)
    }
}

/// Classify a Galileo record from its data-source word (orbit-5 field 2), converted
/// to the nearest integer as RINEX 3.05 section 6.9 reads a bitwise field, by RINEX 3.05
/// Table A8:
///
/// - the source bits decide where they name one message: bit 1 (F/NAV E5a-I) is F/NAV,
///   bits 0 and 2 (I/NAV E1-B, E5b-I) are I/NAV;
/// - otherwise (no source bit, or I/NAV and F/NAV bits together) the clock bits decide:
///   bit 9 (clock for E5b,E1) is I/NAV, bit 8 (clock for E5a,E1) is F/NAV;
/// - otherwise the word names no message and the record is
///   [`NavMessage::GalileoUnclassified`], which is used as RTKLIB's default Galileo
///   selection (`sel = 0`) uses a record: with the BGD E5b/E1.
///
/// The second value is whether the word has a pattern the table forbids: bits 0-2 all
/// set, or bits 8 and 9 both set.
fn galileo_message(data_sources: f64, sat: &str) -> Result<(NavMessage, bool), NavParseError> {
    let bad = || NavParseError::BadField {
        satellite: sat.to_string(),
        field: "data sources",
    };
    if !data_sources.is_finite() {
        return Err(bad());
    }
    let rounded = data_sources.round();
    if rounded < 0.0 || rounded > f64::from(u32::MAX) {
        return Err(bad());
    }
    let word = rounded as u32;
    let fnav_source = word & 0b010 != 0;
    let inav_source = word & 0b101 != 0;
    let e5a_clock = word & (1 << 8) != 0;
    let e5b_clock = word & (1 << 9) != 0;
    let forbidden = word & 0b111 == 0b111 || (e5a_clock && e5b_clock);
    let message = match (inav_source, fnav_source, e5b_clock, e5a_clock) {
        (true, false, _, _) => NavMessage::GalileoInav,
        (false, true, _, _) => NavMessage::GalileoFnav,
        (_, _, true, false) => NavMessage::GalileoInav,
        (_, _, false, true) => NavMessage::GalileoFnav,
        _ => NavMessage::GalileoUnclassified,
    };
    Ok((message, forbidden))
}

/// Decode a GLONASS FDMA record: four lines, or five from RINEX 3.05 on.
pub(crate) fn parse_glonass_block(
    lines: &[&str],
    satellite_id: GnssSatelliteId,
    sat: &str,
    version: NavVersion,
    layout: Layout,
) -> Result<Decoded<GlonassRecord>, NavParseError> {
    if lines.len() < 4 {
        return Err(NavParseError::TruncatedRecord(sat.to_string()));
    }
    let bad = |what: &'static str| NavParseError::BadField {
        satellite: sat.to_string(),
        field: what,
    };
    let mut departures = Vec::new();
    let has_fourth_orbit_line = version.glonass_has_fourth_orbit_line();
    let expected = if has_fourth_orbit_line { 5 } else { 4 };
    if lines.len() < expected {
        // A RINEX 3.05+ record without its fourth orbit line: the state vector and
        // clock are all there; the 3.05 fields read as absent.
        departures.push(NavParseError::TruncatedRecord(sat.to_string()));
    }
    check_extra_lines(lines, expected, sat, &mut departures);
    let epoch = parse_record_epoch(
        lines[0],
        layout,
        sat,
        "epoch",
        validate::CivilSecondPolicy::UtcLike,
    )?;
    let epoch_utc_j2000_s = epoch.j2000_s();
    if !epoch_utc_j2000_s.is_finite() {
        return Err(bad("epoch"));
    }
    // RTKLIB `decode_geph`: toc=gpst2time(week,floor((tow+450.0)/900.0)*900). The week
    // starts and J2000 (12:00) both lie on the 15-minute grid.
    let toe_utc_j2000_s = ((epoch_utc_j2000_s + 450.0) / 900.0).floor() * 900.0;
    let data = record_fields(&lines[..expected.min(lines.len())], layout);
    let get = |index: usize| data.get(index).copied().flatten();
    let raw = |index: usize| raw_record_field(lines, layout, index);
    let km =
        |index: usize, what: &'static str| get(index).map(|x| x * KM_TO_M).ok_or_else(|| bad(what));
    let g = |index: usize, what: &'static str| get(index).ok_or_else(|| bad(what));

    let clk_bias = g(0, "clock bias")?;
    let gamma_n = g(1, "gamma_n")?;
    let message_frame_time_s = stated_field(raw(2), "message frame time", sat, &mut departures);
    let pos_m = [km(3, "x")?, km(7, "y")?, km(11, "z")?];
    let vel_m_s = [km(4, "vx")?, km(8, "vy")?, km(12, "vz")?];
    let acc_m_s2 = [km(5, "ax")?, km(9, "ay")?, km(13, "az")?];
    let sv_health = g(6, "health")?;
    let (freq_channel, stated_freq_channel) =
        glonass_frequency_channel(g(10, "frequency channel")?, sat)?;
    let age_days = stated_field(raw(14), "age of operation", sat, &mut departures);
    let (status_flags, l1_l2_group_delay_field_s, urai, health_flags) =
        if has_fourth_orbit_line && lines.len() >= 5 {
            (
                stated_field(raw(15), "status flags", sat, &mut departures),
                stated_field(raw(16), "l1/l2 group delay", sat, &mut departures),
                stated_field(raw(17), "urai", sat, &mut departures),
                stated_field(raw(18), "health flags", sat, &mut departures),
            )
        } else {
            (None, None, None, None)
        };

    Ok(Decoded {
        value: GlonassRecord {
            satellite_id,
            toe_utc_j2000_s,
            epoch_utc_j2000_s,
            pos_m,
            vel_m_s,
            acc_m_s2,
            clk_bias,
            gamma_n,
            sv_health,
            freq_channel,
            stated_freq_channel,
            message_frame_time_s,
            age_days,
            status_flags,
            l1_l2_group_delay_field_s,
            urai,
            health_flags,
        },
        departures,
    })
}

/// The four broadcast-orbit values of a version 3/4 continuation line (columns
/// 4/23/42/61).
fn orbit_row(line: &str) -> [Option<f64>; 4] {
    [
        parse_f64(line, 4, 23),
        parse_f64(line, 23, 42),
        parse_f64(line, 42, 61),
        parse_f64(line, 61, 80),
    ]
}

fn raw_orbit_field(line: &str, field_index: usize) -> &str {
    const RANGES: [(usize, usize); 4] = [(4, 23), (23, 42), (42, 61), (61, 80)];
    let (start, end) = RANGES[field_index];
    raw_field(line, start, end)
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

fn finite_integral_i8(
    value: f64,
    field: &'static str,
    min: i8,
    max: i8,
    sat: &str,
) -> Result<i8, NavParseError> {
    validate::finite(value, field).map_err(|error| map_record_field_error(error, sat))?;
    if value < f64::from(min) || value > f64::from(max) || value.trunc() != value {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field,
        });
    }
    Ok(value as i8)
}

fn finite_integral_u8(
    value: f64,
    field: &'static str,
    min: u8,
    max: u8,
    sat: &str,
) -> Result<u8, NavParseError> {
    validate::finite(value, field).map_err(|error| map_record_field_error(error, sat))?;
    if value < f64::from(min) || value > f64::from(max) || value.trunc() != value {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field,
        });
    }
    Ok(value as u8)
}

fn optional_integral_u32(
    raw: &str,
    field: &'static str,
    sat: &str,
) -> Result<Option<u32>, NavParseError> {
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let value =
        validate::strict_f64(raw, field).map_err(|error| map_record_field_error(error, sat))?;
    finite_integral_u32(value, field, sat).map(Some)
}

fn optional_cnav_delay(
    raw: &str,
    field: &'static str,
    sat: &str,
) -> Result<Option<f64>, NavParseError> {
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let value =
        validate::strict_f64(raw, field).map_err(|error| map_record_field_error(error, sat))?;
    if !write::d19_12_representable(value) {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field,
        });
    }
    let mut rendered = String::new();
    write::push_d19_12(&mut rendered, value);
    let mut sentinel = String::new();
    write::push_d19_12(&mut sentinel, -4096.0 * 2.0_f64.powi(-35));
    if rendered == sentinel {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn optional_keplerian_delay(
    raw: &str,
    field: &'static str,
    sat: &str,
) -> Result<Option<f64>, NavParseError> {
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let value =
        validate::strict_f64(raw, field).map_err(|error| map_record_field_error(error, sat))?;
    if !write::d19_12_representable(value) {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field,
        });
    }
    Ok(Some(value))
}

/// Read the GLONASS frequency channel field as the integer it states, with a value
/// above 128 read as that value less 256, as RTKLIB `decode_geph` reads it ("some
/// receiver output >128 for minus frequency number").
///
/// The field must hold a whole number that fits [`GlonassRecord::freq_channel`];
/// it is not held to the `-7..=6` FDMA allocation. RTKLIB keeps a record whose
/// channel is outside that allocation, reporting it only in its trace, and real
/// products carry such values for extended slots, so refusing the record would
/// lose a broadcast ephemeris the file states plainly. Consumers that need a
/// carrier check the allocation themselves with
/// [`valid_glonass_frequency_channel`].
fn glonass_frequency_channel(value: f64, sat: &str) -> Result<(i32, i32), NavParseError> {
    const FIELD: &str = "frequency channel";
    validate::finite(value, FIELD).map_err(|error| map_record_field_error(error, sat))?;
    if value.trunc() != value || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        return Err(NavParseError::BadField {
            satellite: sat.to_string(),
            field: FIELD,
        });
    }
    let channel = value as i32;
    Ok((fold_glonass_channel(channel), channel))
}

/// A stated GLONASS channel as RTKLIB `decode_geph` reads it: above 128, less 256.
pub(crate) fn fold_glonass_channel(channel: i32) -> i32 {
    if channel > 128 {
        channel - 256
    } else {
        channel
    }
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

#[cfg(all(test, sidereon_repo_tests))]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
