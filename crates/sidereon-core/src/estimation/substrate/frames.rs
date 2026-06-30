//! Local-frame substrate: recipe-selected receiver frames.
//!
//! Two distinct receiver-frame constructions live here, each keyed by
//! [`FrameRecipe`] so a strategy selects the floating-point operation order it
//! needs by enum value instead of owning a private copy of the helper:
//!
//! - The GEOCENTRIC local frame ([`local_up`] / [`local_neu_basis`],
//!   position-normalized up, `Z x up` east, `up x east` north). That single
//!   op-order is pinned in [`crate::frame`]. RTK selects it as
//!   [`FrameRecipe::GeocentricUpRtkReference`] and PPP as
//!   [`FrameRecipe::GeodeticNeuCrossProduct`]; today both resolve to the same
//!   geocentric construction, byte-for-byte, and are kept as distinct named
//!   variants only so the canonical frame (P6) can diverge without disturbing
//!   the reference goldens.
//! - The SPP Skyfield GEODETIC frame ([`geodetic_from_ecef`] /
//!   [`az_el_from_ecef`]), selected by [`FrameRecipe::SppSkyfieldAuThreeIter`]:
//!   the ECEF->geodetic conversion replicating the core
//!   `itrs_to_geodetic_compute` AU-internal three-iteration latitude solve, and
//!   the geodetic ENU topocentric azimuth/elevation built from that
//!   latitude/longitude. This is the SPP measurement model's frame, moved here
//!   from `spp/mod.rs` so the substrate owns it instead of panicking on the SPP
//!   recipe.
//! - The CANONICAL WGS84 GEODETIC frame ([`geodetic_from_ecef`] /
//!   [`az_el_from_ecef`]), selected by [`FrameRecipe::CanonicalWgs84`]: the same
//!   geodetic ENU topocentric construction, but with the ECEF->geodetic step
//!   computed directly in meters on the WGS84 ellipsoid via the core
//!   [`crate::astro::frames::transforms::geodetic_from_ecef_proj`] (a
//!   closed-form Bowring-style solve, bit-pinned to PROJ in the core), rather
//!   than the Skyfield AU-scaled three-iteration latitude loop. Canonical
//!   deliberately uses ONE consistent meters-native WGS84/ITRF basis under IERS
//!   conventions instead of a reference-specific AU-scaled path; the two share
//!   the WGS84 ellipsoid, so the geodetic latitude/longitude agree to well below
//!   a microarcsecond and the topocentric az/el to far below the elevation mask
//!   resolution.
//!
//! The geocentric basis and the geodetic ENU rotation are DIFFERENT
//! constructions (geocentric up vs the geodetic ellipsoid normal differ by up to
//! ~0.19 deg), so each frame recipe routes to exactly the helper its reference
//! was captured against; a recipe never reaches the helper for the other
//! construction.

use crate::astro::frames::transforms::geodetic_from_ecef_proj;

use crate::constants::{AU_KM, KM_TO_M, WGS84_A_KM, WGS84_E2};
use crate::estimation::recipe::FrameRecipe;
use crate::frame::{geocentric_neu_basis, geocentric_up, Wgs84Geodetic};

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;

/// Geocentric local up (`position / |position|`) at an ECEF position, selected
/// by frame recipe. Both geocentric recipes share this op-order; a zero-length
/// position degenerates to `+Z`. The non-geocentric recipes build their frames
/// elsewhere (SPP via [`geodetic_from_ecef`], DOP/canonical in their own
/// callers) and never reach this geocentric helper.
#[inline]
pub(crate) fn local_up(frame: FrameRecipe, position_ecef_m: [f64; 3]) -> [f64; 3] {
    match frame {
        FrameRecipe::GeocentricUpRtkReference | FrameRecipe::GeodeticNeuCrossProduct => {
            geocentric_up(position_ecef_m)
        }
        _ => unreachable!("geocentric local up is selected only by the RTK/PPP geocentric recipes"),
    }
}

/// Geocentric local North-East-Up basis `(north, east, up)` at an ECEF position,
/// selected by frame recipe. Both geocentric recipes share this op-order; the
/// non-geocentric recipes never reach this helper.
#[inline]
pub(crate) fn local_neu_basis(
    frame: FrameRecipe,
    position_ecef_m: [f64; 3],
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    match frame {
        FrameRecipe::GeocentricUpRtkReference | FrameRecipe::GeodeticNeuCrossProduct => {
            geocentric_neu_basis(position_ecef_m)
        }
        _ => unreachable!(
            "geocentric local NEU basis is selected only by the RTK/PPP geocentric recipes"
        ),
    }
}

/// Receiver geodetic position (geodetic latitude/longitude in radians, height in
/// meters) from an ECEF position in meters, selected by frame recipe.
///
/// [`FrameRecipe::SppSkyfieldAuThreeIter`] replicates the core
/// `itrs_to_geodetic_compute` operation tree (Skyfield's AU-internal
/// three-iteration latitude solve), taking meters in and radians+meters out so
/// no degree boundary appears inside the SPP loop. It is bit-exact (0 ULP)
/// against that core km/deg function at the boundary (the SPP geodetic
/// cross-check). The other frame recipes build their frames elsewhere and never
/// reach this Skyfield geodetic helper.
pub(crate) fn geodetic_from_ecef(frame: FrameRecipe, ecef_m: [f64; 3]) -> Wgs84Geodetic {
    match frame {
        FrameRecipe::SppSkyfieldAuThreeIter => skyfield_au_geodetic(ecef_m),
        FrameRecipe::CanonicalWgs84 => canonical_wgs84_geodetic(ecef_m),
        _ => unreachable!(
            "the geodetic receiver frame is selected only by the SPP Skyfield and canonical recipes"
        ),
    }
}

/// The SPP Skyfield AU-internal three-iteration ECEF->geodetic conversion.
fn skyfield_au_geodetic(ecef_m: [f64; 3]) -> Wgs84Geodetic {
    let x = ecef_m[0] / KM_TO_M;
    let y = ecef_m[1] / KM_TO_M;
    let z = ecef_m[2] / KM_TO_M;

    let x_au = x / AU_KM;
    let y_au = y / AU_KM;
    let z_au = z / AU_KM;

    let a_au = WGS84_A_KM / AU_KM;
    let r_xy = (x_au * x_au + y_au * y_au).sqrt();

    let lon_raw = y_au.atan2(x_au);
    let mut lon_shifted = (lon_raw - PI) % TAU;
    if lon_shifted < 0.0 {
        lon_shifted += TAU;
    }
    let lon = lon_shifted - PI;

    let mut lat = z_au.atan2(r_xy);
    let mut a_c = 0.0;
    let mut hyp = 0.0;
    for _ in 0..3 {
        let sin_lat = lat.sin();
        let e2_sin_lat = WGS84_E2 * sin_lat;
        a_c = a_au / (1.0 - e2_sin_lat * sin_lat).sqrt();
        hyp = z_au + a_c * e2_sin_lat;
        lat = hyp.atan2(r_xy);
    }

    let height_au = (hyp * hyp + r_xy * r_xy).sqrt() - a_c;
    let height_m = height_au * AU_KM * KM_TO_M;

    Wgs84Geodetic {
        lat_rad: lat,
        lon_rad: lon,
        height_m,
    }
}

/// The canonical meters-native WGS84 ECEF->geodetic conversion: the core
/// [`geodetic_from_ecef_proj`] closed-form solve (bit-pinned to PROJ in the
/// `sidereon-core` crate) on the WGS84 ellipsoid, taking meters in and
/// radians+meters out. Unlike [`skyfield_au_geodetic`] it never scales through
/// AU and never runs a fixed-count latitude loop; it is the single consistent
/// IERS-conventions geodetic basis the canonical strategy uses.
fn canonical_wgs84_geodetic(ecef_m: [f64; 3]) -> Wgs84Geodetic {
    // `geodetic_from_ecef_proj` returns `[lon_deg, lat_deg, alt_m]`.
    let [lon_deg, lat_deg, alt_m] =
        geodetic_from_ecef_proj(ecef_m[0], ecef_m[1], ecef_m[2]).expect("valid ECEF coordinates");
    Wgs84Geodetic {
        lat_rad: lat_deg.to_radians(),
        lon_rad: lon_deg.to_radians(),
        height_m: alt_m,
    }
}

/// Azimuth / elevation of a satellite from the receiver (ECEF meters in, radians
/// out) plus the receiver geodetic recomputed from the receiver ECEF position.
pub(crate) struct AzEl {
    pub geodetic: Wgs84Geodetic,
    pub az_rad: f64,
    pub el_rad: f64,
}

/// Topocentric azimuth/elevation of a satellite seen from the receiver, selected
/// by frame recipe.
///
/// [`FrameRecipe::SppSkyfieldAuThreeIter`] builds the geodetic ENU topocentric
/// rotation from the receiver geodetic latitude/longitude (itself the Skyfield
/// AU geodetic of [`geodetic_from_ecef`]); this is the SPP measurement model's
/// az/el. The geodetic ENU rotation is a separate construction from the shared
/// geocentric NEU basis (see the module note), so it stays keyed to the SPP
/// recipe and the other recipes never reach this helper.
pub(crate) fn az_el_from_ecef(
    frame: FrameRecipe,
    rx_ecef_m: [f64; 3],
    sat_ecef_m: [f64; 3],
) -> AzEl {
    let geo = match frame {
        FrameRecipe::SppSkyfieldAuThreeIter => skyfield_au_geodetic(rx_ecef_m),
        FrameRecipe::CanonicalWgs84 => canonical_wgs84_geodetic(rx_ecef_m),
        _ => unreachable!(
            "the geodetic ENU azimuth/elevation is selected only by the SPP Skyfield and canonical recipes"
        ),
    };
    geodetic_enu_az_el(geo, rx_ecef_m, sat_ecef_m)
}

/// The geodetic ENU topocentric azimuth/elevation of a satellite seen from a
/// receiver, given the receiver's geodetic latitude/longitude. The
/// ECEF->geodetic step that produces `geo` is the recipe-specific choice (the
/// Skyfield AU solve or the canonical WGS84 solve); this topocentric rotation is
/// shared by both, so the only frame divergence is the geodetic basis, not the
/// az/el math. The receiver-satellite delta and the basis trig are independent
/// (no shared rounding), so factoring `geo` out is bit-identical to computing it
/// inline.
fn geodetic_enu_az_el(geo: Wgs84Geodetic, rx_ecef_m: [f64; 3], sat_ecef_m: [f64; 3]) -> AzEl {
    let dx = sat_ecef_m[0] - rx_ecef_m[0];
    let dy = sat_ecef_m[1] - rx_ecef_m[1];
    let dz = sat_ecef_m[2] - rx_ecef_m[2];

    let sin_lat = geo.lat_rad.sin();
    let cos_lat = geo.lat_rad.cos();
    let sin_lon = geo.lon_rad.sin();
    let cos_lon = geo.lon_rad.cos();

    let e = -sin_lon * dx + cos_lon * dy;
    let n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
    let u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;

    let rng = (e * e + n * n + u * u).sqrt();
    let el = (u / rng).asin();
    let mut az = e.atan2(n);
    if az < 0.0 {
        az += TAU;
    }

    AzEl {
        geodetic: geo,
        az_rad: az,
        el_rad: el,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POSITION: [f64; 3] = [4_027_894.0, 307_045.0, 4_919_474.0];

    fn bits3(v: [f64; 3]) -> [u64; 3] {
        [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
    }

    #[test]
    fn geocentric_recipes_match_frame_helper_bits() {
        for frame in [
            FrameRecipe::GeocentricUpRtkReference,
            FrameRecipe::GeodeticNeuCrossProduct,
        ] {
            assert_eq!(
                bits3(local_up(frame, POSITION)),
                bits3(geocentric_up(POSITION))
            );
            let (n, e, u) = local_neu_basis(frame, POSITION);
            let (rn, re, ru) = geocentric_neu_basis(POSITION);
            assert_eq!(bits3(n), bits3(rn));
            assert_eq!(bits3(e), bits3(re));
            assert_eq!(bits3(u), bits3(ru));
        }
    }

    #[test]
    fn spp_geodetic_recipe_matches_skyfield_au_solve_bits() {
        // The SPP frame recipe routes to the Skyfield AU three-iteration geodetic
        // solve; selecting it is bit-identical to the underlying construction.
        let g = geodetic_from_ecef(FrameRecipe::SppSkyfieldAuThreeIter, POSITION);
        let want = skyfield_au_geodetic(POSITION);
        assert_eq!(g.lat_rad.to_bits(), want.lat_rad.to_bits());
        assert_eq!(g.lon_rad.to_bits(), want.lon_rad.to_bits());
        assert_eq!(g.height_m.to_bits(), want.height_m.to_bits());
    }

    #[test]
    fn spp_az_el_recipe_matches_geodetic_enu_bits() {
        let sat = [15_600_000.0, -20_400_000.0, 9_800_000.0];
        let got = az_el_from_ecef(FrameRecipe::SppSkyfieldAuThreeIter, POSITION, sat);
        let want = geodetic_enu_az_el(skyfield_au_geodetic(POSITION), POSITION, sat);
        assert_eq!(got.az_rad.to_bits(), want.az_rad.to_bits());
        assert_eq!(got.el_rad.to_bits(), want.el_rad.to_bits());
        assert_eq!(
            got.geodetic.lat_rad.to_bits(),
            want.geodetic.lat_rad.to_bits()
        );
    }

    #[test]
    fn canonical_geodetic_recipe_matches_proj_solve_bits() {
        // The canonical frame recipe routes to the meters-native WGS84 proj
        // solve; selecting it is bit-identical to that underlying construction.
        let g = geodetic_from_ecef(FrameRecipe::CanonicalWgs84, POSITION);
        let want = canonical_wgs84_geodetic(POSITION);
        assert_eq!(g.lat_rad.to_bits(), want.lat_rad.to_bits());
        assert_eq!(g.lon_rad.to_bits(), want.lon_rad.to_bits());
        assert_eq!(g.height_m.to_bits(), want.height_m.to_bits());
    }

    #[test]
    fn canonical_az_el_recipe_matches_geodetic_enu_bits() {
        let sat = [15_600_000.0, -20_400_000.0, 9_800_000.0];
        let got = az_el_from_ecef(FrameRecipe::CanonicalWgs84, POSITION, sat);
        let want = geodetic_enu_az_el(canonical_wgs84_geodetic(POSITION), POSITION, sat);
        assert_eq!(got.az_rad.to_bits(), want.az_rad.to_bits());
        assert_eq!(got.el_rad.to_bits(), want.el_rad.to_bits());
        assert_eq!(
            got.geodetic.lat_rad.to_bits(),
            want.geodetic.lat_rad.to_bits()
        );
    }

    #[test]
    fn canonical_and_skyfield_geodetic_agree_to_sub_milliarcsecond() {
        // Same WGS84 ellipsoid (identical a / e2 constants) via two op-orders:
        // the Skyfield AU-scaled three-iteration latitude loop and the canonical
        // meters-native PROJ closed-form solve. The geodetic basis clusters to
        // ~6.4e-11 rad in latitude (~13 microarcseconds, ~0.4 mm on the ground)
        // and negligibly in longitude/height, far below the elevation-mask
        // resolution. This sub-milliarcsecond geodetic agreement is what bounds
        // the canonical-vs-SPP measurement-model divergence: the frame feeds only
        // the atmospheric-correction az/el geometry.
        let skyfield = skyfield_au_geodetic(POSITION);
        let canonical = canonical_wgs84_geodetic(POSITION);
        // 1 milliarcsecond in radians (~4.85e-9 rad).
        let mas_rad = std::f64::consts::PI / 180.0 / 3_600.0 / 1.0e3;
        assert!((skyfield.lat_rad - canonical.lat_rad).abs() < mas_rad);
        assert!((skyfield.lon_rad - canonical.lon_rad).abs() < mas_rad);
        assert!((skyfield.height_m - canonical.height_m).abs() < 1.0e-6);
    }
}
