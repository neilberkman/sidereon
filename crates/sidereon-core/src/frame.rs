//! Frame-tagged position types.
//!
//! Every position value encodes its reference frame and datum in the **type
//! name**, never a bare `position_m` that hides the frame. SP3 products give
//! satellite states
//! in the ITRF/IGS-realization ECEF frame, in meters; that fact is carried by
//! [`ItrfPositionM`] so a consumer cannot accidentally mix it with, say, a
//! GCRS/TEME state from the core crate (which is in kilometers).

use core::fmt;

use crate::astro::math::vec3::{cross3, unit3};

/// Error returned when constructing frame-tagged values from invalid inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameValueError {
    /// A frame value component was non-finite or outside its documented domain.
    #[error("invalid frame value {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

const fn invalid_input(field: &'static str, reason: &'static str) -> FrameValueError {
    FrameValueError::InvalidInput { field, reason }
}

/// Geocentric local vertical at a receiver ECEF position: the position vector
/// normalized (the spherical radial direction, out from the geocenter).
///
/// PARITY / FRAME: this is the GEOCENTRIC up (`position / |position|`), **not**
/// the geodetic ellipsoid normal. The two differ by up to ~0.19 deg, which
/// shifts low-elevation variance weighting and antenna projections; the GNSS
/// baseline/PPP code paths were captured against the geocentric definition
/// (Elixir `local_up/1`), so this is deliberately the geocentric variant. A
/// geodetic ENU rotation (built from geodetic latitude/longitude) is a separate
/// construction and stays with its caller. Normalization is reciprocal-multiply
/// (`scale3(v, 1/n)` via [`unit3`]) to preserve the callers' 0-ULP goldens; a
/// zero-length position degenerates to `+Z`.
pub fn geocentric_up(position_ecef_m: [f64; 3]) -> [f64; 3] {
    unit3(position_ecef_m).unwrap_or([0.0, 0.0, 1.0])
}

/// East unit of the geocentric local frame for a given local `up`:
/// `normalize(Z x up)`, degenerating to `+X` when `up` is parallel to `Z`.
pub fn geocentric_east(up: [f64; 3]) -> [f64; 3] {
    unit3(cross3([0.0, 0.0, 1.0], up)).unwrap_or([1.0, 0.0, 0.0])
}

/// Geocentric local North-East-Up basis at a receiver ECEF position, returned as
/// `(north, east, up)`.
///
/// `up` is [`geocentric_up`], `east` is [`geocentric_east`], and
/// `north = normalize(up x east)`. This is the single shared geocentric basis
/// builder: the PPP correction tables, the static PPP solver, and the RTK
/// baseline-filter antenna model all consume it (each previously kept a
/// byte-identical copy). See [`geocentric_up`] for the geocentric-vs-geodetic
/// distinction.
pub fn geocentric_neu_basis(position_ecef_m: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let up = geocentric_up(position_ecef_m);
    let east = geocentric_east(up);
    let north = unit3(cross3(up, east)).unwrap_or([0.0, 0.0, 1.0]);
    (north, east, up)
}

/// A position in the ITRF / IGS-realization Earth-Centered-Earth-Fixed frame,
/// expressed in **meters**.
///
/// SP3 ephemerides are published in an IGS realization of the ITRF (e.g.
/// `IGS14`, `IGS20`); the exact realization string is carried on the SP3
/// header ([`crate::sp3::Sp3Header::coordinate_system`]), while this type fixes
/// the *kind* of frame (ITRF/IGS ECEF) and the *unit* (meters) in the type
/// system. There is intentionally no implicit conversion to the core crate's
/// kilometer states; conversion happens explicitly at an API boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItrfPositionM {
    /// ECEF X coordinate in meters.
    pub x_m: f64,
    /// ECEF Y coordinate in meters.
    pub y_m: f64,
    /// ECEF Z coordinate in meters.
    pub z_m: f64,
}

impl ItrfPositionM {
    /// Construct an ITRF ECEF position from meter components.
    pub const fn new(x_m: f64, y_m: f64, z_m: f64) -> Result<Self, FrameValueError> {
        if !x_m.is_finite() {
            return Err(invalid_input("x_m", "must be finite"));
        }
        if !y_m.is_finite() {
            return Err(invalid_input("y_m", "must be finite"));
        }
        if !z_m.is_finite() {
            return Err(invalid_input("z_m", "must be finite"));
        }
        Ok(Self { x_m, y_m, z_m })
    }

    /// The components as a `[x, y, z]` meter array.
    pub const fn as_array(self) -> [f64; 3] {
        [self.x_m, self.y_m, self.z_m]
    }
}

impl fmt::Display for ItrfPositionM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ITRF[{:.3}, {:.3}, {:.3}] m",
            self.x_m, self.y_m, self.z_m
        )
    }
}

/// A geodetic (ellipsoidal) position on the WGS84 datum.
///
/// Latitude and longitude are geodetic angles in **radians**; height is the
/// ellipsoidal height in **meters**. This is the receiver-position type the
/// atmospheric (ionosphere/troposphere) models take: they need the geodetic
/// latitude/longitude of the antenna and, for the troposphere, the height.
///
/// The frame and units are fixed in the type so a caller cannot mix this with
/// the ECEF [`ItrfPositionM`] or pass degrees where radians are expected.
/// Longitude is positive east; latitude is positive north.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wgs84Geodetic {
    /// Geodetic latitude in radians, positive north, range `[-pi/2, pi/2]`.
    pub lat_rad: f64,
    /// Geodetic longitude in radians, positive east, range `[-pi, pi]`.
    pub lon_rad: f64,
    /// Ellipsoidal height above the WGS84 ellipsoid in meters.
    pub height_m: f64,
}

impl Wgs84Geodetic {
    /// Construct a WGS84 geodetic position from radians and meters.
    pub const fn new(lat_rad: f64, lon_rad: f64, height_m: f64) -> Result<Self, FrameValueError> {
        if !lat_rad.is_finite() {
            return Err(invalid_input("lat_rad", "must be finite"));
        }
        if !lon_rad.is_finite() {
            return Err(invalid_input("lon_rad", "must be finite"));
        }
        if !height_m.is_finite() {
            return Err(invalid_input("height_m", "must be finite"));
        }
        if lat_rad < -core::f64::consts::FRAC_PI_2 || lat_rad > core::f64::consts::FRAC_PI_2 {
            return Err(invalid_input("lat_rad", "must be in [-pi/2, pi/2]"));
        }
        if lon_rad < -core::f64::consts::PI || lon_rad > core::f64::consts::PI {
            return Err(invalid_input("lon_rad", "must be in [-pi, pi]"));
        }
        Ok(Self {
            lat_rad,
            lon_rad,
            height_m,
        })
    }
}

impl fmt::Display for Wgs84Geodetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WGS84[lat {:.6} rad, lon {:.6} rad, h {:.3} m]",
            self.lat_rad, self.lon_rad, self.height_m
        )
    }
}

/// Convert a WGS84 geodetic position to an ITRF/ECEF position.
///
/// Free-function wrapper over the crate's single validated forward converter
/// [`crate::astro::frames::transforms::geodetic_to_itrs`] (the same WGS84 math
/// the positioning paths use); this only bridges the typed frame values and
/// their units (the frame types carry radians + meters, the internal converter
/// works in degrees + kilometers). No conversion math is reimplemented here.
pub fn geodetic_to_itrf(
    geodetic: Wgs84Geodetic,
) -> Result<ItrfPositionM, crate::astro::frames::transforms::FrameTransformError> {
    let (x_km, y_km, z_km) = crate::astro::frames::transforms::geodetic_to_itrs(
        geodetic.lat_rad.to_degrees(),
        geodetic.lon_rad.to_degrees(),
        geodetic.height_m / 1000.0,
    )?;
    ItrfPositionM::new(x_km * 1000.0, y_km * 1000.0, z_km * 1000.0)
        .map_err(frame_value_to_transform)
}

/// Convert an ITRF/ECEF position to a WGS84 geodetic position.
///
/// Free-function wrapper over the crate's single validated inverse converter
/// [`crate::astro::frames::transforms::itrs_to_geodetic_compute`]; like
/// [`geodetic_to_itrf`] it only bridges the typed frame values and their units.
/// No conversion math is reimplemented here.
pub fn itrf_to_geodetic(
    position: ItrfPositionM,
) -> Result<Wgs84Geodetic, crate::astro::frames::transforms::FrameTransformError> {
    let (lat_deg, lon_deg, alt_km) = crate::astro::frames::transforms::itrs_to_geodetic_compute(
        position.x_m / 1000.0,
        position.y_m / 1000.0,
        position.z_m / 1000.0,
    )?;
    Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), alt_km * 1000.0)
        .map_err(frame_value_to_transform)
}

fn frame_value_to_transform(
    error: FrameValueError,
) -> crate::astro::frames::transforms::FrameTransformError {
    match error {
        FrameValueError::InvalidInput { field, reason } => {
            crate::astro::frames::transforms::FrameTransformError::InvalidInput { field, reason }
        }
    }
}

/// A velocity in the ITRF / IGS-realization ECEF frame, in **meters per
/// second**.
///
/// Present only when the SP3 file is a velocity product (`#?V...` header /
/// `V`-records). The frame/unit are fixed in the type for the same reason as
/// [`ItrfPositionM`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItrfVelocityMS {
    /// ECEF X velocity in meters per second.
    pub vx_m_s: f64,
    /// ECEF Y velocity in meters per second.
    pub vy_m_s: f64,
    /// ECEF Z velocity in meters per second.
    pub vz_m_s: f64,
}

impl ItrfVelocityMS {
    /// Construct an ITRF ECEF velocity from meter-per-second components.
    pub const fn new(vx_m_s: f64, vy_m_s: f64, vz_m_s: f64) -> Result<Self, FrameValueError> {
        if !vx_m_s.is_finite() {
            return Err(invalid_input("vx_m_s", "must be finite"));
        }
        if !vy_m_s.is_finite() {
            return Err(invalid_input("vy_m_s", "must be finite"));
        }
        if !vz_m_s.is_finite() {
            return Err(invalid_input("vz_m_s", "must be finite"));
        }
        Ok(Self {
            vx_m_s,
            vy_m_s,
            vz_m_s,
        })
    }

    /// The components as a `[vx, vy, vz]` m/s array.
    pub const fn as_array(self) -> [f64; 3] {
        [self.vx_m_s, self.vy_m_s, self.vz_m_s]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::math::vec3::{norm3, scale3};

    fn bits(v: [f64; 3]) -> [u64; 3] {
        [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
    }

    /// The explicit clamp-and-match recipe the PPP/RTK modules previously
    /// copy-pasted. The shared builder must reproduce it bit-for-bit.
    fn reference_neu(position_ecef_m: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let up = match norm3(position_ecef_m) {
            n if n > 0.0 => scale3(position_ecef_m, 1.0 / n),
            _ => [0.0, 0.0, 1.0],
        };
        let cross = cross3([0.0, 0.0, 1.0], up);
        let east = if cross == [0.0, 0.0, 0.0] {
            [1.0, 0.0, 0.0]
        } else {
            match norm3(cross) {
                n if n > 0.0 => scale3(cross, 1.0 / n),
                _ => [1.0, 0.0, 0.0],
            }
        };
        let north = match norm3(cross3(up, east)) {
            n if n > 0.0 => scale3(cross3(up, east), 1.0 / n),
            _ => [0.0, 0.0, 1.0],
        };
        (north, east, up)
    }

    fn assert_close3(actual: [f64; 3], expected: [f64; 3], tol: f64) {
        for (a, e) in actual.into_iter().zip(expected) {
            assert!(
                (a - e).abs() <= tol,
                "component {a:?} differs from {e:?} by more than {tol}"
            );
        }
    }

    #[test]
    fn constructors_reject_invalid_frame_values() {
        assert_eq!(
            ItrfPositionM::new(f64::NAN, 0.0, 0.0),
            Err(FrameValueError::InvalidInput {
                field: "x_m",
                reason: "must be finite"
            })
        );
        assert_eq!(
            ItrfVelocityMS::new(0.0, f64::INFINITY, 0.0),
            Err(FrameValueError::InvalidInput {
                field: "vy_m_s",
                reason: "must be finite"
            })
        );
        assert_eq!(
            Wgs84Geodetic::new(core::f64::consts::FRAC_PI_2 + 1.0e-12, 0.0, 0.0),
            Err(FrameValueError::InvalidInput {
                field: "lat_rad",
                reason: "must be in [-pi/2, pi/2]"
            })
        );
        assert_eq!(
            Wgs84Geodetic::new(0.0, -core::f64::consts::PI - 1.0e-12, 0.0),
            Err(FrameValueError::InvalidInput {
                field: "lon_rad",
                reason: "must be in [-pi, pi]"
            })
        );
        assert_eq!(
            Wgs84Geodetic::new(0.0, 0.0, f64::NAN),
            Err(FrameValueError::InvalidInput {
                field: "height_m",
                reason: "must be finite"
            })
        );
    }

    #[test]
    fn constructors_preserve_valid_frame_values() {
        let position = ItrfPositionM::new(1.0, -2.0, 3.5).expect("valid ITRF position");
        assert_eq!(position.as_array(), [1.0, -2.0, 3.5]);

        let velocity = ItrfVelocityMS::new(0.1, -0.2, 0.3).expect("valid ITRF velocity");
        assert_eq!(velocity.as_array(), [0.1, -0.2, 0.3]);

        let geodetic =
            Wgs84Geodetic::new(-0.5, core::f64::consts::PI, -12.0).expect("valid geodetic");
        assert_eq!(geodetic.lat_rad.to_bits(), (-0.5_f64).to_bits());
        assert_eq!(geodetic.lon_rad.to_bits(), core::f64::consts::PI.to_bits());
        assert_eq!(geodetic.height_m.to_bits(), (-12.0_f64).to_bits());

        let west_antimeridian =
            Wgs84Geodetic::new(0.25, -core::f64::consts::PI, 42.0).expect("valid geodetic");
        assert_eq!(
            west_antimeridian.lon_rad.to_bits(),
            (-core::f64::consts::PI).to_bits()
        );
    }

    #[test]
    fn geocentric_basis_matches_reference_recipe_to_the_bit() {
        // A representative mid-latitude ECEF receiver position (metres).
        let position = [4_027_894.0, 307_045.0, 4_919_474.0];
        let (north, east, up) = geocentric_neu_basis(position);
        let (rn, re, ru) = reference_neu(position);
        assert_eq!(bits(north), bits(rn));
        assert_eq!(bits(east), bits(re));
        assert_eq!(bits(up), bits(ru));
        // up = normalized position; east and north are orthogonal unit vectors.
        assert_eq!(bits(up), bits(geocentric_up(position)));
        assert_eq!(bits(east), bits(geocentric_east(up)));
    }

    #[test]
    fn geocentric_basis_is_right_handed_enu_and_points_north() {
        let (north, east, up) = geocentric_neu_basis([6_378_137.0, 0.0, 0.0]);
        assert_close3(up, [1.0, 0.0, 0.0], 1.0e-15);
        assert_close3(east, [0.0, 1.0, 0.0], 1.0e-15);
        assert_close3(north, [0.0, 0.0, 1.0], 1.0e-15);

        let (north, east, up) = geocentric_neu_basis([4_027_894.0, 307_045.0, 4_919_474.0]);
        assert_close3(cross3(up, east), north, 1.0e-15);
        assert_close3(cross3(east, north), up, 1.0e-15);
    }

    #[test]
    fn geodetic_to_itrf_hits_equatorial_prime_meridian_reference() {
        // Sea level on the equator at the prime meridian sits on the +X axis at
        // the WGS84 equatorial radius.
        let origin = Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid geodetic");
        let ecef = geodetic_to_itrf(origin).expect("valid conversion");
        assert!((ecef.x_m - 6_378_137.0).abs() <= 1.0e-3, "x = {}", ecef.x_m);
        assert!(ecef.y_m.abs() <= 1.0e-6, "y = {}", ecef.y_m);
        assert!(ecef.z_m.abs() <= 1.0e-6, "z = {}", ecef.z_m);
    }

    #[test]
    fn geodetic_itrf_round_trip_is_tight() {
        // A real station (roughly Philadelphia) at 100 m ellipsoidal height.
        let lat_rad = 39.95_f64.to_radians();
        let lon_rad = (-75.16_f64).to_radians();
        let station = Wgs84Geodetic::new(lat_rad, lon_rad, 100.0).expect("valid geodetic");

        let ecef = geodetic_to_itrf(station).expect("valid conversion");
        let back = itrf_to_geodetic(ecef).expect("valid conversion");

        // The forward converter is the closed-form WGS84 map; the inverse is
        // Skyfield's 3-iteration reduction, so the round-trip closes to ~mm, not
        // to the bit.
        assert!((back.lat_rad - station.lat_rad).abs() <= 1.0e-9);
        assert!((back.lon_rad - station.lon_rad).abs() <= 1.0e-12);
        assert!((back.height_m - station.height_m).abs() <= 1.0e-6);
    }

    #[test]
    fn free_converters_agree_with_internal_converters_bit_for_bit() {
        use crate::astro::frames::transforms::{geodetic_to_itrs, itrs_to_geodetic_compute};

        // Forward: the public typed converter must equal the internal degree/km
        // converter with only the documented unit bridging applied.
        let lat_rad = 39.95_f64.to_radians();
        let lon_rad = (-75.16_f64).to_radians();
        let station = Wgs84Geodetic::new(lat_rad, lon_rad, 100.0).expect("valid geodetic");

        let (x_km, y_km, z_km) =
            geodetic_to_itrs(lat_rad.to_degrees(), lon_rad.to_degrees(), 100.0 / 1000.0)
                .expect("internal forward converter");
        let ecef = geodetic_to_itrf(station).expect("public forward converter");
        assert_eq!(ecef.x_m.to_bits(), (x_km * 1000.0).to_bits());
        assert_eq!(ecef.y_m.to_bits(), (y_km * 1000.0).to_bits());
        assert_eq!(ecef.z_m.to_bits(), (z_km * 1000.0).to_bits());

        // Inverse: same bit-for-bit agreement against the internal converter.
        let position = ItrfPositionM::new(1_113_194.0, -4_383_212.0, 4_077_985.0)
            .expect("valid ITRF position");
        let (lat_deg, lon_deg, alt_km) = itrs_to_geodetic_compute(
            position.x_m / 1000.0,
            position.y_m / 1000.0,
            position.z_m / 1000.0,
        )
        .expect("internal inverse converter");
        let geodetic = itrf_to_geodetic(position).expect("public inverse converter");
        assert_eq!(geodetic.lat_rad.to_bits(), lat_deg.to_radians().to_bits());
        assert_eq!(geodetic.lon_rad.to_bits(), lon_deg.to_radians().to_bits());
        assert_eq!(geodetic.height_m.to_bits(), (alt_km * 1000.0).to_bits());
    }

    #[test]
    fn geocentric_basis_handles_degenerate_positions() {
        // Zero-length position degenerates to +Z up, +X east.
        let (north, east, up) = geocentric_neu_basis([0.0, 0.0, 0.0]);
        assert_eq!(up, [0.0, 0.0, 1.0]);
        assert_eq!(east, [1.0, 0.0, 0.0]);
        // north = normalize(up x east) = [0,0,1] x [1,0,0] = [0,1,0].
        assert_eq!(north, [0.0, 1.0, 0.0]);
        // A purely polar position (up parallel to +Z) keeps east at +X.
        assert_eq!(geocentric_east([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]);
    }
}
