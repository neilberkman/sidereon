//! RINEX and CRINEX parsing.
//!
//! Parsing is separated from ephemeris evaluation: navigation records can be
//! parsed through [`nav`] or loaded into an [`crate::ephemeris::BroadcastEphemeris`],
//! while observation files are parsed through [`observations`].

/// RINEX clock parsing and satellite clock-bias interpolation.
pub mod clock {
    pub use crate::rinex_clock::{
        civil_to_clock_instant, civil_to_gps_seconds, ClockEpoch, ClockPoint, RinexClock,
        RinexClockError,
    };
}

/// Hatanaka/CRINEX observation-file decoding and encoding.
pub mod crinex {
    pub use crate::crinex::{
        decode, decode_to, encode_crinex, encode_stream, parse_stream, CrinexVersion, EpochRecord,
        ObsEpoch, ObsStream, SatRecord,
    };
}

/// RINEX navigation-message parsing.
pub mod nav {
    pub use crate::ionex::GalileoNequickCoeffs;
    pub use crate::rinex_nav::{
        encode_nav, parse_glonass, parse_glonass_lenient, parse_iono_corrections,
        parse_leap_seconds, parse_nav, BroadcastGroupDelayTerm, BroadcastGroupDelays,
        BroadcastRecord, GlonassParse, GlonassRecord, IonoCorrections, KlobucharAlphaBeta,
        NavMessage, NavParseError, SkippedGlonass,
    };

    /// Parse a RINEX NAV text into an evaluated broadcast ephemeris store.
    pub type BroadcastEphemeris = crate::rinex_nav::BroadcastStore;
}

/// RINEX observation parsing and pseudorange extraction.
pub mod observations {
    pub use crate::rinex_obs::{
        band_frequency_hz, carrier_phase_rows, observation_frequency_hz, observation_values,
        pseudoranges, CarrierPhaseRow, ObsEpoch, ObsEpochTime, ObsHeader, ObsPhaseShift,
        ObsScaleFactor, ObsValue, ObservationFilter, ObservationKind, ObservationValueRow,
        RinexObs, SignalPolicy,
    };

    /// Role-oriented alias for a parsed RINEX observation file.
    pub type ObservationFile = RinexObs;
}

pub use clock::RinexClock;
pub use crinex::{decode as decode_crinex, decode_to as decode_crinex_to, encode_crinex};
pub use nav::{parse_glonass, parse_iono_corrections, parse_leap_seconds, parse_nav};
pub use observations::{pseudoranges, ObservationFile};
