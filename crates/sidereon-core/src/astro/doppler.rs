//! Doppler and range-rate computations for satellite-ground links.
//!
//! Positive range rate means the satellite is receding from the station.
//! Positive Doppler ratio means the transmitter is approaching the station.
//! The satellite ECEF velocity uses `v_ecef = R*v_gcrs - omega x r`, the
//! standard rotating-frame transport. This is consistent with core's
//! oracle-validated GNSS range-rate model
//! `precise_positioning::velocity::predict_range_rate_m_s`. It deliberately
//! differs from the legacy orbis `shift/4`, which had the transport sign
//! inverted and was never oracle-gated.

use crate::astro::constants::{models::pz90::OMEGA_E_RAD_S, units::M_PER_KM};
use crate::astro::frames::transforms::{
    gcrs_to_itrs_matrix, geodetic_to_itrs, mat3_vec3_mul, FrameTransformError,
};
use crate::astro::time::scales::TimeScales;
use crate::constants::C_M_S;

/// Speed of light in km/s.
const C_KM_S: f64 = C_M_S / M_PER_KM;

/// Earth rotation rate in rad/s (PZ-90 `OMEGA_E_RAD_S`) used in the standard ECEF transport term.
const OMEGA_EARTH: f64 = OMEGA_E_RAD_S;

/// Range-rate and Doppler shift result for a carrier frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopplerShift {
    /// Range rate in km/s; positive means receding from the station.
    pub range_rate_km_s: f64,
    /// Doppler shift in Hz; positive means a frequency increase.
    pub doppler_hz: f64,
    /// Dimensionless Doppler ratio; positive means approaching the station.
    pub doppler_ratio: f64,
}

/// Error while computing Doppler shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DopplerError {
    /// A frame transformation failed.
    #[error("doppler frame transform failed: {0}")]
    FrameTransform(#[from] FrameTransformError),
}

/// Compute range rate and dimensionless Doppler ratio from a GCRS state.
///
/// Position is in km, velocity is in km/s, station latitude/longitude are in
/// degrees, and station altitude is in km. Positive range rate means receding;
/// positive Doppler ratio means approaching.
pub fn range_rate_and_ratio(
    gcrs_position_km: [f64; 3],
    gcrs_velocity_km_s: [f64; 3],
    station_lat_deg: f64,
    station_lon_deg: f64,
    station_alt_km: f64,
    ts: &TimeScales,
) -> Result<(f64, f64), DopplerError> {
    let [sat_x, sat_y, sat_z] = gcrs_position_km;
    let [sat_vx, sat_vy, sat_vz] = gcrs_velocity_km_s;

    let r_mat = gcrs_to_itrs_matrix(ts)?;

    let pos_gcrs = [sat_x, sat_y, sat_z];
    let pos_itrs = mat3_vec3_mul(&r_mat, &pos_gcrs)?;

    let vel_gcrs = [sat_vx, sat_vy, sat_vz];
    let vel_itrs_rot = mat3_vec3_mul(&r_mat, &vel_gcrs)?;

    // [Claude] Standard rotating-frame transport: v_ecef = R*v_gcrs - omega x r_ecef.
    // Since omega x r = [-OMEGA*y, OMEGA*x, 0], subtracting it contributes [+OMEGA*y, -OMEGA*x, 0].
    let transport_x = OMEGA_EARTH * pos_itrs[1];
    let transport_y = -OMEGA_EARTH * pos_itrs[0];
    let transport_z = 0.0;

    let vel_itrs = [
        vel_itrs_rot[0] + transport_x,
        vel_itrs_rot[1] + transport_y,
        vel_itrs_rot[2] + transport_z,
    ];

    let (stn_x, stn_y, stn_z) = geodetic_to_itrs(station_lat_deg, station_lon_deg, station_alt_km)?;

    let range_vec = [
        pos_itrs[0] - stn_x,
        pos_itrs[1] - stn_y,
        pos_itrs[2] - stn_z,
    ];

    let range_mag =
        (range_vec[0] * range_vec[0] + range_vec[1] * range_vec[1] + range_vec[2] * range_vec[2])
            .sqrt();

    let range_unit = [
        range_vec[0] / range_mag,
        range_vec[1] / range_mag,
        range_vec[2] / range_mag,
    ];

    let range_rate =
        range_unit[0] * vel_itrs[0] + range_unit[1] * vel_itrs[1] + range_unit[2] * vel_itrs[2];

    let doppler_ratio = -range_rate / C_KM_S;

    Ok((range_rate, doppler_ratio))
}

/// Compute range rate, Doppler ratio, and carrier Doppler shift.
///
/// `frequency_hz` is multiplied by the Doppler ratio exactly as the Orbis
/// wrapper did.
pub fn doppler_shift(
    gcrs_position_km: [f64; 3],
    gcrs_velocity_km_s: [f64; 3],
    station_lat_deg: f64,
    station_lon_deg: f64,
    station_alt_km: f64,
    ts: &TimeScales,
    frequency_hz: f64,
) -> Result<DopplerShift, DopplerError> {
    let (range_rate_km_s, doppler_ratio) = range_rate_and_ratio(
        gcrs_position_km,
        gcrs_velocity_km_s,
        station_lat_deg,
        station_lon_deg,
        station_alt_km,
        ts,
    )?;

    Ok(DopplerShift {
        range_rate_km_s,
        doppler_hz: doppler_ratio * frequency_hz,
        doppler_ratio,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]

    use super::*;
    use crate::astro::frames::transforms::{gcrs_to_itrs_matrix, geodetic_to_itrs, mat3_vec3_mul};

    const GCRS_POSITION_KM: [f64; 3] = [
        3700.211211203995390,
        2015.912218120605530,
        5309.513078070447591,
    ];
    const GCRS_VELOCITY_KM_S: [f64; 3] =
        [-3.398428894395407, 6.869656830559572, -0.239850181126689];
    const STATION_LAT_DEG: f64 = 40.0;
    const STATION_LON_DEG: f64 = -74.0;
    const STATION_ALT_KM: f64 = 0.0;
    const FREQUENCY_HZ: f64 = 437.0e6;

    fn fixed_time_scales() -> TimeScales {
        TimeScales::from_utc(2018, 7, 4, 0, 0, 0.0).expect("valid UTC instant")
    }

    #[test]
    fn numerical_derivative_oracle() {
        let stn =
            geodetic_to_itrs(STATION_LAT_DEG, STATION_LON_DEG, STATION_ALT_KM).expect("station");

        let range_at = |t_off: f64| {
            let ts =
                TimeScales::from_utc(2018, 7, 4, 0, 0, 30.0 + t_off).expect("valid UTC instant");
            let r = gcrs_to_itrs_matrix(&ts).expect("valid frame transform");
            let pos_gcrs = [
                GCRS_POSITION_KM[0] + GCRS_VELOCITY_KM_S[0] * t_off,
                GCRS_POSITION_KM[1] + GCRS_VELOCITY_KM_S[1] * t_off,
                GCRS_POSITION_KM[2] + GCRS_VELOCITY_KM_S[2] * t_off,
            ];
            let p = mat3_vec3_mul(&r, &pos_gcrs).expect("matrix-vector multiply");
            let d = [p[0] - stn.0, p[1] - stn.1, p[2] - stn.2];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };

        let h = 0.01;
        let numerical = (range_at(h) - range_at(-h)) / (2.0 * h);

        let ts = TimeScales::from_utc(2018, 7, 4, 0, 0, 30.0).expect("valid UTC instant");
        let (analytic_range_rate, _) = range_rate_and_ratio(
            GCRS_POSITION_KM,
            GCRS_VELOCITY_KM_S,
            STATION_LAT_DEG,
            STATION_LON_DEG,
            STATION_ALT_KM,
            &ts,
        )
        .expect("valid Doppler computation");

        // The legacy orbis +omega x r sign would fail this by roughly 4.5e-3 km/s.
        assert!((analytic_range_rate - numerical).abs() < 1.0e-6);
    }

    #[test]
    fn consistent_with_oracle_gated_gnss_range_rate() {
        use crate::precise_positioning::velocity::{
            predict_range_rate_m_s, ReceiverVelocityState, VelocityObservation,
        };
        use crate::{GnssSatelliteId, GnssSystem};

        let ts = TimeScales::from_utc(2018, 7, 4, 0, 0, 30.0).expect("valid UTC instant");
        let r = gcrs_to_itrs_matrix(&ts).expect("valid frame transform");
        let pos_ecef = mat3_vec3_mul(&r, &GCRS_POSITION_KM).expect("matrix-vector multiply");
        let vel_rot = mat3_vec3_mul(&r, &GCRS_VELOCITY_KM_S).expect("matrix-vector multiply");
        let vel_ecef = [
            vel_rot[0] + OMEGA_EARTH * pos_ecef[1],
            vel_rot[1] - OMEGA_EARTH * pos_ecef[0],
            vel_rot[2],
        ];
        let stn =
            geodetic_to_itrs(STATION_LAT_DEG, STATION_LON_DEG, STATION_ALT_KM).expect("station");

        let obs = VelocityObservation {
            sat: GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id"),
            satellite_position_m: pos_ecef,
            satellite_velocity_m_s: vel_ecef,
            measured_range_rate_m_s: 0.0,
            sigma_m_s: 1.0,
            satellite_clock_drift_m_s: 0.0,
        };
        let receiver = ReceiverVelocityState {
            position_m: [stn.0, stn.1, stn.2],
            velocity_m_s: [0.0; 3],
            clock_drift_m_s: 0.0,
        };
        let pred = predict_range_rate_m_s(&obs, receiver).expect("nonzero line of sight");

        let (rr, _) = range_rate_and_ratio(
            GCRS_POSITION_KM,
            GCRS_VELOCITY_KM_S,
            STATION_LAT_DEG,
            STATION_LON_DEG,
            STATION_ALT_KM,
            &ts,
        )
        .expect("valid Doppler computation");

        assert!((rr - pred.range_rate_m_s).abs() < 1.0e-12);
    }

    #[test]
    fn frozen_bits_regression() {
        let ts = fixed_time_scales();

        let (range_rate_km_s, doppler_ratio) = range_rate_and_ratio(
            GCRS_POSITION_KM,
            GCRS_VELOCITY_KM_S,
            STATION_LAT_DEG,
            STATION_LON_DEG,
            STATION_ALT_KM,
            &ts,
        )
        .expect("valid Doppler computation");
        let shift = doppler_shift(
            GCRS_POSITION_KM,
            GCRS_VELOCITY_KM_S,
            STATION_LAT_DEG,
            STATION_LON_DEG,
            STATION_ALT_KM,
            &ts,
            FREQUENCY_HZ,
        )
        .expect("valid Doppler shift");

        // [Claude] Pinned to the corrected (-omega x r) physics validated by numerical_derivative_oracle.
        let expected_range_rate_km_s: f64 = 2.11937962917790934e-1;
        let expected_doppler_ratio: f64 = -7.06948948388124429e-7;
        let expected_doppler_hz: f64 = -3.08936690445610395e2;

        assert_eq!(
            range_rate_km_s.to_bits(),
            expected_range_rate_km_s.to_bits()
        );
        assert_eq!(doppler_ratio.to_bits(), expected_doppler_ratio.to_bits());
        assert_eq!(shift.range_rate_km_s.to_bits(), range_rate_km_s.to_bits());
        assert_eq!(shift.doppler_ratio.to_bits(), doppler_ratio.to_bits());
        assert_eq!(shift.doppler_hz.to_bits(), expected_doppler_hz.to_bits());
    }

    #[test]
    fn sign_and_physics_sanity() {
        let ts = fixed_time_scales();

        let (range_rate_km_s, doppler_ratio) = range_rate_and_ratio(
            GCRS_POSITION_KM,
            GCRS_VELOCITY_KM_S,
            STATION_LAT_DEG,
            STATION_LON_DEG,
            STATION_ALT_KM,
            &ts,
        )
        .expect("valid Doppler computation");
        let shift = doppler_shift(
            GCRS_POSITION_KM,
            GCRS_VELOCITY_KM_S,
            STATION_LAT_DEG,
            STATION_LON_DEG,
            STATION_ALT_KM,
            &ts,
            FREQUENCY_HZ,
        )
        .expect("valid Doppler shift");

        assert!(range_rate_km_s.abs() <= 8.0);
        assert!(shift.doppler_hz.abs() <= 12_000.0);
        assert_eq!(
            doppler_ratio.to_bits(),
            (-range_rate_km_s / C_KM_S).to_bits()
        );
        assert_eq!(
            shift.doppler_hz.to_bits(),
            (doppler_ratio * FREQUENCY_HZ).to_bits()
        );
    }
}
