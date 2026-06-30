//! Conical Earth-shadow (eclipse) geometry.
//!
//! Determines whether a satellite is in full sunlight, penumbra, or umbra
//! using a conical shadow model: the penumbra and umbra cones cast by Earth
//! given the Sun's position, and the satellite's perpendicular distance from
//! the Earth-Sun shadow axis. This is the authoritative implementation; the
//! Elixir binding is a thin marshaling layer over it.

use crate::astro::constants::{astro::SOLAR_RADIUS_KM, earth::MEAN_EARTH_RADIUS_KM};
use crate::astro::math::vec3;

/// Illumination state of a satellite relative to Earth's conical shadow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EclipseStatus {
    /// Full sunlight.
    Sunlit,
    /// Partial shadow between the umbra and penumbra cones.
    Penumbra,
    /// Full shadow inside the umbra cone.
    Umbra,
}

/// Error while computing conical eclipse geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EclipseError {
    #[error("invalid eclipse input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

/// Shadow fraction in `[0.0, 1.0]`: `0.0` is full sunlight, `1.0` is full
/// umbra, intermediate values are penumbra.
///
/// `sat_pos` is the satellite GCRS position (km); `sun_pos` is the vector from
/// Earth center to the Sun (km).
pub fn shadow_fraction(sat_pos: [f64; 3], sun_pos: [f64; 3]) -> Result<f64, EclipseError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;

    let d_sun = vec3::norm3(sun_pos);

    // Unit vector from Earth to Sun. The division form (not reciprocal-multiply
    // via `vec3::unit3`) is required to stay bit-exact with the reference.
    let sun_u = [sun_pos[0] / d_sun, sun_pos[1] / d_sun, sun_pos[2] / d_sun];

    // Project the satellite onto the Earth-Sun line. Positive points toward the
    // Sun; a satellite on the sunlit side cannot be in shadow.
    let proj = vec3::dot3(sat_pos, sun_u);
    if proj >= 0.0 {
        return Ok(0.0);
    }

    // Perpendicular distance from the satellite to the shadow axis.
    let perp = vec3::sub3(sat_pos, vec3::scale3(sun_u, proj));
    let rho = vec3::norm3(perp);

    // Distance behind Earth along the shadow axis (positive).
    let dist_behind = -proj;

    // Umbra cone (converging): sin(alpha) = (R_sun - R_earth) / d_sun.
    let alpha_umbra = ((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / d_sun).asin();
    // Penumbra cone (diverging): sin(alpha) = (R_sun + R_earth) / d_sun.
    let alpha_penumbra = ((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / d_sun).asin();

    // Cone radii at the satellite's distance behind Earth.
    let r_umbra = MEAN_EARTH_RADIUS_KM - dist_behind * alpha_umbra.tan();
    let r_penumbra = MEAN_EARTH_RADIUS_KM + dist_behind * alpha_penumbra.tan();

    if rho >= r_penumbra {
        // Beyond the penumbra cone: full sunlight.
        Ok(0.0)
    } else if r_umbra > 0.0 && rho <= r_umbra {
        // Inside the umbra cone: full shadow.
        Ok(1.0)
    } else if r_umbra > 0.0 {
        // Penumbra region: linear interpolation between the cone radii.
        Ok((r_penumbra - rho) / (r_penumbra - r_umbra))
    } else {
        // Past the umbra tip (antumbra): penumbra still applies, fraction
        // decreasing from the axis.
        Ok(0.0_f64.max((r_penumbra - rho) / r_penumbra))
    }
}

/// Classify the satellite's eclipse status from its [`shadow_fraction`].
pub fn status(sat_pos: [f64; 3], sun_pos: [f64; 3]) -> Result<EclipseStatus, EclipseError> {
    let fraction = shadow_fraction(sat_pos, sun_pos)?;
    if fraction >= 1.0 {
        Ok(EclipseStatus::Umbra)
    } else if fraction > 0.0 {
        Ok(EclipseStatus::Penumbra)
    } else {
        Ok(EclipseStatus::Sunlit)
    }
}

fn validate_nonzero_vec3(v: [f64; 3], field: &'static str) -> Result<(), EclipseError> {
    if !v.iter().all(|value| value.is_finite()) {
        return Err(invalid_eclipse_input(field, "not finite"));
    }
    let norm = vec3::norm3(v);
    if norm == 0.0 {
        return Err(invalid_eclipse_input(field, "zero vector"));
    }
    if !norm.is_finite() {
        return Err(invalid_eclipse_input(field, "out of range"));
    }
    Ok(())
}

fn invalid_eclipse_input(field: &'static str, reason: &'static str) -> EclipseError {
    EclipseError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use crate::astro::events::{CrossingDirection, EventFinder};

    use super::*;

    // Approximate Sun-Earth distance, ~1 AU in km, matching the reference
    // oracle inputs.
    const AU_KM: f64 = 149_597_870.7;

    #[test]
    fn shadow_fraction_matches_reference_bits() {
        // Frozen bits captured from the reference (Elixir) conical-shadow
        // implementation. Cross-language 0-ULP equality.
        let sun = [AU_KM, 0.0, 0.0];

        assert_eq!(
            shadow_fraction([7000.0, 0.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([0.0, 7000.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([-7000.0, 0.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([-7000.0, 6370.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0x3fe0_a32f_08e7_fb1f
        );
        assert_eq!(
            shadow_fraction([-7000.0, 6410.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            shadow_fraction([-7000.0, 6330.0, 0.0], sun)
                .expect("valid eclipse geometry")
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn status_classifies_cones() {
        let sun = [AU_KM, 0.0, 0.0];

        assert_eq!(
            status([7000.0, 0.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Sunlit
        );
        assert_eq!(
            status([0.0, 7000.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Sunlit
        );
        assert_eq!(
            status([-7000.0, 0.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Umbra
        );
        assert_eq!(
            status([-7000.0, 6370.0, 0.0], sun).expect("valid eclipse geometry"),
            EclipseStatus::Penumbra
        );
    }

    #[test]
    fn shadow_fraction_increases_toward_axis() {
        let sun = [AU_KM, 0.0, 0.0];
        let outside = shadow_fraction([-7000.0, 6410.0, 0.0], sun).expect("valid eclipse geometry");
        let penumbra =
            shadow_fraction([-7000.0, 6370.0, 0.0], sun).expect("valid eclipse geometry");
        let umbra = shadow_fraction([-7000.0, 6330.0, 0.0], sun).expect("valid eclipse geometry");
        let center = shadow_fraction([-7000.0, 0.0, 0.0], sun).expect("valid eclipse geometry");

        assert_eq!(outside, 0.0);
        assert!(penumbra > 0.0 && penumbra < 1.0);
        assert_eq!(umbra, 1.0);
        assert_eq!(center, 1.0);
    }

    #[test]
    fn event_finder_refines_eclipse_fraction_crossings() {
        let sun = [AU_KM, 0.0, 0.0];
        let x_km = -7000.0;
        let start_y_km = -7000.0;
        let end_y_km = 7000.0;
        let duration_seconds = 1200.0;
        let threshold = 0.5;
        let y_at = |time_seconds: f64| {
            start_y_km + (end_y_km - start_y_km) * time_seconds / duration_seconds
        };
        let fraction_at = |time_seconds: f64| {
            shadow_fraction([x_km, y_at(time_seconds), 0.0], sun)
                .expect("valid synthetic eclipse geometry")
        };

        let events = EventFinder::new(0.0, duration_seconds, 60.0, 1.0e-7)
            .expect("valid eclipse finder")
            .find_crossings(fraction_at, threshold)
            .expect("finite eclipse predicate");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].direction, CrossingDirection::Rising);
        assert_eq!(events[1].direction, CrossingDirection::Falling);
        assert_eq!(events[0].threshold, threshold);
        assert_eq!(events[1].threshold, threshold);

        let half_shadow_radius = shadow_radius_for_fraction(threshold, -x_km);
        let expected_enter =
            time_for_y(-half_shadow_radius, start_y_km, end_y_km, duration_seconds);
        let expected_exit = time_for_y(half_shadow_radius, start_y_km, end_y_km, duration_seconds);

        assert_close(events[0].time_seconds, expected_enter, 1.0e-6);
        assert_close(events[1].time_seconds, expected_exit, 1.0e-6);
        assert_close(events[0].value, threshold, 1.0e-7);
        assert_close(events[1].value, threshold, 1.0e-7);
        assert!(fraction_at(0.0) < threshold);
        assert!(fraction_at(duration_seconds * 0.5) > threshold);
        assert!(fraction_at(duration_seconds) < threshold);
    }

    fn shadow_radius_for_fraction(fraction: f64, dist_behind_km: f64) -> f64 {
        let alpha_umbra = ((SOLAR_RADIUS_KM - MEAN_EARTH_RADIUS_KM) / AU_KM).asin();
        let alpha_penumbra = ((SOLAR_RADIUS_KM + MEAN_EARTH_RADIUS_KM) / AU_KM).asin();
        let r_umbra = MEAN_EARTH_RADIUS_KM - dist_behind_km * alpha_umbra.tan();
        let r_penumbra = MEAN_EARTH_RADIUS_KM + dist_behind_km * alpha_penumbra.tan();
        r_penumbra - fraction * (r_penumbra - r_umbra)
    }

    fn time_for_y(y_km: f64, start_y_km: f64, end_y_km: f64, duration_seconds: f64) -> f64 {
        (y_km - start_y_km) * duration_seconds / (end_y_km - start_y_km)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn eclipse_helpers_reject_invalid_vectors() {
        assert_invalid_eclipse_field(
            shadow_fraction([0.0, 0.0, 0.0], [AU_KM, 0.0, 0.0]).unwrap_err(),
            "sat_pos",
            "zero vector",
        );
        assert_invalid_eclipse_field(
            status([7000.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]).unwrap_err(),
            "sun_pos",
            "not finite",
        );
    }

    fn assert_invalid_eclipse_field(
        error: EclipseError,
        expected: &'static str,
        expected_reason: &'static str,
    ) {
        let EclipseError::InvalidInput { field, reason } = error;
        assert_eq!(field, expected);
        assert_eq!(reason, expected_reason);
    }
}
