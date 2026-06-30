//! Canonical physical and unit constants shared by the core crates.
//!
//! Values with different standards or parity sources are intentionally named
//! distinctly in [`models`] or [`astro`] instead of being silently unified.

/// Unit-conversion constants.
pub mod units {
    /// Meters per kilometer.
    pub const M_PER_KM: f64 = 1_000.0;
    /// Millimeters per meter.
    pub const MM_PER_M: f64 = 1_000.0;
    /// Kilometers to meters.
    pub const KM_TO_M: f64 = M_PER_KM;
    /// Microseconds to seconds.
    pub const US_TO_S: f64 = 1.0e-6;
    /// Microseconds per second.
    pub const MICROSECONDS_PER_SECOND_I64: i64 = 1_000_000;
    /// Microseconds per second, floating point. Used where a microsecond count
    /// is divided or multiplied as an `f64`; kept distinct from [`US_TO_S`] so
    /// the divide-by-`1_000_000.0` operation order is preserved bit-for-bit.
    pub const MICROSECONDS_PER_SECOND: f64 = 1_000_000.0;
    /// Nanoseconds to seconds.
    pub const NS_TO_S: f64 = 1.0e-9;
    /// Degrees to radians.
    pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
    /// Radians to degrees.
    pub const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
    /// Degrees in one semicircle.
    pub const DEGREES_PER_SEMICIRCLE: f64 = 180.0;
    /// Degrees in one full circle.
    pub const DEGREES_PER_CIRCLE: f64 = 360.0;
    /// Arcseconds to radians.
    #[allow(clippy::excessive_precision)]
    pub const ARCSEC_TO_RAD: f64 = 4.848_136_811_095_359_935_899_141e-6;
}

/// Time-scale and Julian-date constants.
pub mod time {
    use super::units::MICROSECONDS_PER_SECOND_I64;

    /// Seconds per civil day.
    pub const SECONDS_PER_DAY: f64 = 86_400.0;
    /// Seconds per civil day, integer form for calendar arithmetic.
    pub const SECONDS_PER_DAY_I64: i64 = 86_400;
    /// Microseconds per civil day.
    pub const MICROSECONDS_PER_DAY_I64: i64 = SECONDS_PER_DAY_I64 * MICROSECONDS_PER_SECOND_I64;
    /// Seconds per GNSS week.
    pub const SECONDS_PER_WEEK: f64 = 604_800.0;
    /// Julian Date of the J2000 epoch.
    pub const J2000_JD: f64 = 2_451_545.0;
    /// Days per Julian century.
    pub const DAYS_PER_JULIAN_CENTURY: f64 = 36_525.0;
    /// TT - TAI, seconds.
    pub const TT_MINUS_TAI_S: f64 = 32.184;
    /// GPS/Galileo/QZSS system time minus TAI, seconds.
    pub const GPST_MINUS_TAI_S: f64 = 19.0;
    /// BeiDou time minus TAI, seconds.
    pub const BDT_MINUS_TAI_S: f64 = 33.0;
}

/// WGS84 Earth constants.
pub mod earth {
    use super::units::M_PER_KM;

    /// WGS84 geocentric gravitational constant (km^3/s^2).
    pub const GM_EARTH_KM3_S2: f64 = 398_600.441_8;
    /// WGS84 geocentric gravitational constant (m^3/s^2).
    pub const GM_EARTH_M3_S2: f64 = GM_EARTH_KM3_S2 * 1.0e9;
    /// WGS84 Earth equatorial radius (km).
    pub const WGS84_A_KM: f64 = 6_378.137;
    /// WGS84 Earth equatorial radius (m).
    pub const WGS84_A_M: f64 = WGS84_A_KM * M_PER_KM;
    /// Mean Earth radius used by spherical shell propagation and conical
    /// shadow (eclipse) models (km).
    pub const MEAN_EARTH_RADIUS_KM: f64 = 6_371.0;
    /// Mean Earth radius used by spherical shell propagation and conical
    /// shadow (eclipse) models (m).
    pub const MEAN_EARTH_RADIUS_M: f64 = MEAN_EARTH_RADIUS_KM * M_PER_KM;
    /// WGS84 flattening.
    pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
    /// WGS84 first eccentricity squared.
    pub const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
    /// WGS84 Earth rotation rate used by GNSS Sagnac/transport terms (rad/s).
    pub const OMEGA_E_DOT_RAD_S: f64 = 7.292_115_146_7e-5;
    /// Earth's J2 coefficient used by the core force models.
    pub const J2_EARTH: f64 = 1.082_626_68e-3;
}

/// Universal physical constants.
pub mod physics {
    /// Speed of light in vacuum (m/s), exact by SI definition (and the value
    /// IS-GPS-200 fixes for GNSS ranging).
    pub const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
}

/// Astronomical constants.
pub mod astro {
    /// Astronomical unit in kilometers, matching Skyfield's AU convention.
    pub const AU_KM: f64 = 149_597_870.700;
    /// Astronomical unit in meters, matching [`AU_KM`].
    pub const AU_M: f64 = AU_KM * 1_000.0;
    /// AU value used by the Montenbruck-Gill analytic Sun/Moon series.
    pub const MONTENBRUCK_AU_M: f64 = 149_597_870_691.0;
    /// Solar photosphere radius (km), used by conical eclipse-shadow geometry.
    pub const SOLAR_RADIUS_KM: f64 = 696_340.0;
}

/// Model-specific constants that intentionally differ from WGS84.
pub mod models {
    pub mod pz90 {
        /// Geocentric gravitational constant (m^3/s^2), PZ-90.11.
        pub const GM_M3_S2: f64 = 3.986_004_4e14;
        /// Second zonal harmonic of the geopotential, PZ-90.11.
        pub const J2: f64 = 1.082_625_7e-3;
        /// Earth rotation rate (rad/s), PZ-90.11.
        pub const OMEGA_E_RAD_S: f64 = 7.292_115e-5;
        /// Earth equatorial radius (m), PZ-90.11.
        pub const A_M: f64 = 6_378_136.0;
    }

    pub mod iers {
        /// Earth radius used by the Dehant solid Earth tide formulation (m).
        pub const SOLID_TIDE_EARTH_RADIUS_M: f64 = 6_378_136.6;
    }

    pub mod proj {
        /// PROJ-pinned WGS84 semi-major axis (m).
        pub const WGS84_A_M: f64 = f64::from_bits(0x4158_54a6_4000_0000);
        /// PROJ-pinned WGS84 semi-minor axis (m).
        pub const WGS84_B_M: f64 = f64::from_bits(0x4158_3fc4_141c_97d0);
        /// PROJ-pinned first eccentricity squared.
        pub const WGS84_ES: f64 = f64::from_bits(0x3f7b_6b90_f1fe_94f0);
        /// PROJ-pinned second eccentricity squared.
        pub const WGS84_E2S: f64 = f64::from_bits(0x3f7b_9adf_e197_dcd1);
        /// PROJ-pinned radian-to-degree multiplier.
        pub const RAD_TO_DEG: f64 = f64::from_bits(0x404c_a5dc_1a63_c1f8);
        /// PROJ-pinned half-pi value.
        pub const HALF_PI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
    }

    pub mod broadcast {
        /// GPS broadcast gravitational constant (m^3/s^2), IS-GPS-200.
        pub const GPS_GM_M3_S2: f64 = 3.986_005_0e14;
        /// GPS/Galileo broadcast Earth rotation rate (rad/s).
        pub const GPS_GALILEO_OMEGA_E_RAD_S: f64 = 7.292_115_146_7e-5;
        /// GPS broadcast relativistic clock constant (s/sqrt(m)).
        pub const GPS_DTR_F: f64 = -0.000_000_000_444_280_763_339_306;
        /// Galileo broadcast gravitational constant (m^3/s^2).
        pub const GALILEO_GM_M3_S2: f64 = 3.986_004_418e14;
        /// Galileo/BeiDou broadcast relativistic clock constant (s/sqrt(m)).
        pub const GALILEO_BEIDOU_DTR_F: f64 = -0.000_000_000_444_280_730_904_397_75;
        /// BeiDou broadcast Earth rotation rate (rad/s).
        pub const BEIDOU_OMEGA_E_RAD_S: f64 = 7.292_115e-5;
    }
}

/// Canonical Earth gravity-model constants: the (mu, Re, J2) triple in km units,
/// used together by the two-body + J2 force model. Geodetic code instead uses the
/// WGS84 ellipsoid parameters (earth::WGS84_A_KM with WGS84_E2/WGS84_F); RE_EARTH
/// and WGS84_A_KM share a numeric value but name distinct concepts (gravity
/// reference radius vs ellipsoid semi-major axis).
pub const MU_EARTH: f64 = earth::GM_EARTH_KM3_S2;
/// Earth reference equatorial radius for the gravity model (km).
pub const RE_EARTH: f64 = earth::WGS84_A_KM;
/// Earth J2 zonal harmonic for the gravity model.
pub const J2_EARTH: f64 = earth::J2_EARTH;
