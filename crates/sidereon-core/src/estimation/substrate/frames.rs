//! Local-frame substrate: recipe-selected receiver frames.
//!
//! Two distinct receiver-frame constructions live here, each keyed by
//! [`FrameRecipe`] so a strategy selects the floating-point operation order it
//! needs by enum value instead of owning a private copy of the helper:
//!
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
//! RTK and PPP take their elevations, azimuths and receiver local frames from
//! the SPP geodetic frame too, through [`satazel`] and [`geodetic_neu_basis`],
//! as RTKLIB takes them from `satazel` and `xyz2enu`. No receiver model uses the
//! geocentric vertical (`position / |position|`), which leans up to ~0.19 deg
//! from the ellipsoid normal; [`crate::frame::geocentric_up`] remains for the
//! caller-selected geocentric DOP convention.

use crate::astro::frames::transforms::geodetic_from_ecef_proj;

use crate::constants::{AU_KM, KM_TO_M, WGS84_A_KM, WGS84_E2};
use crate::estimation::recipe::FrameRecipe;
use crate::frame::Wgs84Geodetic;

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;

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

    let lon_raw = libm::atan2(y_au, x_au);
    let mut lon_shifted = (lon_raw - PI) % TAU;
    if lon_shifted < 0.0 {
        lon_shifted += TAU;
    }
    let lon = lon_shifted - PI;

    let mut lat = libm::atan2(z_au, r_xy);
    let mut a_c = 0.0;
    let mut hyp = 0.0;
    for _ in 0..3 {
        let sin_lat = libm::sin(lat);
        let e2_sin_lat = WGS84_E2 * sin_lat;
        a_c = a_au / (1.0 - e2_sin_lat * sin_lat).sqrt();
        hyp = z_au + a_c * e2_sin_lat;
        lat = libm::atan2(hyp, r_xy);
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

#[derive(Clone, Copy)]
struct ReceiverFrame {
    recipe: FrameRecipe,
    receiver_bits: [u64; 3],
    geodetic: Wgs84Geodetic,
    sin_lat: f64,
    cos_lat: f64,
    sin_lon: f64,
    cos_lon: f64,
}

impl ReceiverFrame {
    fn new(recipe: FrameRecipe, rx_ecef_m: [f64; 3], geodetic: Wgs84Geodetic) -> Self {
        let sin_lat = libm::sin(geodetic.lat_rad);
        let cos_lat = libm::cos(geodetic.lat_rad);
        let sin_lon = libm::sin(geodetic.lon_rad);
        let cos_lon = libm::cos(geodetic.lon_rad);
        Self {
            recipe,
            receiver_bits: rx_ecef_m.map(f64::to_bits),
            geodetic,
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
        }
    }

    fn matches(&self, recipe: FrameRecipe, rx_ecef_m: [f64; 3]) -> bool {
        self.recipe == recipe && self.receiver_bits == rx_ecef_m.map(f64::to_bits)
    }
}

pub(crate) struct ReceiverFrameMemo {
    frame: std::cell::OnceCell<ReceiverFrame>,
}

impl ReceiverFrameMemo {
    pub(crate) fn new() -> Self {
        Self {
            frame: std::cell::OnceCell::new(),
        }
    }

    pub(crate) fn az_el(
        &self,
        recipe: FrameRecipe,
        rx_ecef_m: [f64; 3],
        sat_ecef_m: [f64; 3],
    ) -> AzEl {
        if let Some(receiver_frame) = self.frame.get() {
            if receiver_frame.matches(recipe, rx_ecef_m) {
                let delta = receiver_delta(rx_ecef_m, sat_ecef_m);
                return geodetic_enu_az_el(receiver_frame, delta);
            }
        }

        let geodetic = geodetic_for_recipe(recipe, rx_ecef_m);
        let delta = receiver_delta(rx_ecef_m, sat_ecef_m);
        let uncached_frame = ReceiverFrame::new(recipe, rx_ecef_m, geodetic);
        let _ = self.frame.set(uncached_frame);
        geodetic_enu_az_el(&uncached_frame, delta)
    }
}

/// Topocentric azimuth/elevation of a satellite seen from the receiver, selected
/// by frame recipe.
///
/// [`FrameRecipe::SppSkyfieldAuThreeIter`] builds the geodetic ENU topocentric
/// rotation from the receiver geodetic latitude/longitude (itself the Skyfield
/// AU geodetic of [`geodetic_from_ecef`]); this is the SPP measurement model's
/// az/el, and through [`satazel`] the RTK one. [`FrameRecipe::CanonicalWgs84`]
/// builds it from the canonical WGS84 geodetic.
pub(crate) fn az_el_from_ecef(
    frame: FrameRecipe,
    rx_ecef_m: [f64; 3],
    sat_ecef_m: [f64; 3],
) -> AzEl {
    let geodetic = geodetic_for_recipe(frame, rx_ecef_m);
    let delta = receiver_delta(rx_ecef_m, sat_ecef_m);
    let receiver_frame = ReceiverFrame::new(frame, rx_ecef_m, geodetic);
    geodetic_enu_az_el(&receiver_frame, delta)
}

fn geodetic_for_recipe(frame: FrameRecipe, rx_ecef_m: [f64; 3]) -> Wgs84Geodetic {
    match frame {
        FrameRecipe::SppSkyfieldAuThreeIter => skyfield_au_geodetic(rx_ecef_m),
        FrameRecipe::CanonicalWgs84 => canonical_wgs84_geodetic(rx_ecef_m),
        _ => unreachable!(
            "the geodetic ENU azimuth/elevation is selected only by the SPP Skyfield and canonical recipes"
        ),
    }
}

fn receiver_delta(rx_ecef_m: [f64; 3], sat_ecef_m: [f64; 3]) -> [f64; 3] {
    [
        sat_ecef_m[0] - rx_ecef_m[0],
        sat_ecef_m[1] - rx_ecef_m[1],
        sat_ecef_m[2] - rx_ecef_m[2],
    ]
}

/// RTKLIB `satazel`: the azimuth and elevation (radians) of the line of sight
/// from `rx_ecef_m` to `sat_ecef_m` in the geodetic (ellipsoid-normal) ENU frame
/// of the receiver, through the geodetic conversion the SPP selection uses. A
/// receiver RTKLIB `ecef2pos` places at or below `-RE_WGS84` in height, which is
/// only the geocentre, sees every satellite at the zenith (azimuth 0, elevation
/// `pi / 2`), as `satazel` has it. The RTK elevation mask, reference choice,
/// measurement variances and receiver-antenna model all take elevation from
/// here, as RTKLIB takes it from `satazel` for `selsat`, `ddres`, `varerr` and
/// `antmodel`.
pub(crate) fn satazel(rx_ecef_m: [f64; 3], sat_ecef_m: [f64; 3]) -> (f64, f64) {
    if crate::spp::rtklib_sees_every_satellite_overhead(rx_ecef_m) {
        return (0.0, core::f64::consts::FRAC_PI_2);
    }
    let geo = skyfield_au_geodetic(rx_ecef_m);
    let dx = sat_ecef_m[0] - rx_ecef_m[0];
    let dy = sat_ecef_m[1] - rx_ecef_m[1];
    let dz = sat_ecef_m[2] - rx_ecef_m[2];
    let range = libm::sqrt(dx * dx + dy * dy + dz * dz);
    let dx = dx / range;
    let dy = dy / range;
    let dz = dz / range;

    let sin_lat = libm::sin(geo.lat_rad);
    let cos_lat = libm::cos(geo.lat_rad);
    let sin_lon = libm::sin(geo.lon_rad);
    let cos_lon = libm::cos(geo.lon_rad);
    let east = -sin_lon * dx + cos_lon * dy;
    let north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
    let up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
    let horizontal_squared = east * east + north * north;
    let azimuth = if horizontal_squared < 1.0e-12 {
        0.0
    } else {
        let azimuth = libm::atan2(east, north);
        if azimuth < 0.0 {
            azimuth + TAU
        } else {
            azimuth
        }
    };
    (azimuth, libm::asin(up.clamp(-1.0, 1.0)))
}

/// The geodetic (ellipsoid-normal) local North-East-Up basis at a receiver,
/// returned as `(north, east, up)` unit vectors in ECEF, from the latitude and
/// longitude of the geodetic conversion [`satazel`] uses: the rows of RTKLIB
/// `xyz2enu` at the `ecef2pos` position.
pub(crate) fn geodetic_neu_basis(rx_ecef_m: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let geo = skyfield_au_geodetic(rx_ecef_m);
    let sin_lat = libm::sin(geo.lat_rad);
    let cos_lat = libm::cos(geo.lat_rad);
    let sin_lon = libm::sin(geo.lon_rad);
    let cos_lon = libm::cos(geo.lon_rad);
    (
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    )
}

/// The geodetic ENU topocentric azimuth/elevation of a satellite seen from a
/// receiver, given the receiver's geodetic latitude/longitude. The
/// ECEF->geodetic step that produces `geo` is the recipe-specific choice (the
/// Skyfield AU solve or the canonical WGS84 solve); this topocentric rotation is
/// shared by both, so the only frame divergence is the geodetic basis, not the
/// az/el math. The receiver-satellite delta and the basis trig are independent
/// (no shared rounding), so factoring `geo` out is bit-identical to computing it
/// inline.
fn geodetic_enu_az_el(receiver_frame: &ReceiverFrame, delta: [f64; 3]) -> AzEl {
    let sin_lat = receiver_frame.sin_lat;
    let cos_lat = receiver_frame.cos_lat;
    let sin_lon = receiver_frame.sin_lon;
    let cos_lon = receiver_frame.cos_lon;
    let dx = delta[0];
    let dy = delta[1];
    let dz = delta[2];

    let e = -sin_lon * dx + cos_lon * dy;
    let n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
    let u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;

    let rng = (e * e + n * n + u * u).sqrt();
    let el = libm::asin(u / rng);
    let mut az = libm::atan2(e, n);
    if az < 0.0 {
        az += TAU;
    }

    AzEl {
        geodetic: receiver_frame.geodetic,
        az_rad: az,
        el_rad: el,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_2;

    const POSITION: [f64; 3] = [4_027_894.0, 307_045.0, 4_919_474.0];

    #[test]
    fn satazel_and_the_neu_basis_share_the_geodetic_vertical() {
        let (north, east, up) = geodetic_neu_basis(POSITION);
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for (a, b) in [(north, east), (east, up), (up, north)] {
            assert!(dot(a, b).abs() < 1e-15);
        }
        for v in [north, east, up] {
            assert!((dot(v, v) - 1.0).abs() < 1e-15);
        }
        // A satellite straight up the basis's vertical is at the `satazel` zenith,
        // and one along its north is on the horizon at azimuth 0.
        let along = |v: [f64; 3]| std::array::from_fn(|i| POSITION[i] + 2.0e7 * v[i]);
        let (_az, el) = satazel(POSITION, along(up));
        assert!((el - FRAC_PI_2).abs() < 1e-7, "{el}");
        let (az, el) = satazel(POSITION, along(north));
        assert!(el.abs() < 1e-12, "{el}");
        assert!(az.abs() < 1e-12 || (az - TAU).abs() < 1e-12, "{az}");
        // The geodetic vertical leans from the geocentric one by about 0.19 deg at
        // this 50 deg latitude.
        let geocentric = crate::frame::geocentric_up(POSITION);
        let lean_deg = libm::acos(dot(up, geocentric)) * 180.0 / PI;
        assert!(lean_deg > 0.18 && lean_deg < 0.2, "{lean_deg}");
    }

    #[test]
    fn satazel_puts_every_satellite_at_the_zenith_for_a_receiver_at_the_geocentre() {
        assert_eq!(satazel([0.0; 3], [2.0e7, 1.0e7, -5.0e6]), (0.0, FRAC_PI_2));
    }

    #[test]
    fn satazel_matches_rtklib_local_angles_and_degenerate_azimuth() {
        let (azimuth, elevation) = satazel([10.0, 0.0, 0.0], [10.0, 10.0, 0.0]);
        assert!((azimuth - FRAC_PI_2).abs() < 1.0e-15, "{azimuth}");
        assert!(elevation.abs() < 1.0e-15, "{elevation}");

        let (azimuth, elevation) = satazel(
            POSITION,
            [
                POSITION[0] + 2.0e7 * geodetic_neu_basis(POSITION).2[0],
                POSITION[1] + 2.0e7 * geodetic_neu_basis(POSITION).2[1],
                POSITION[2] + 2.0e7 * geodetic_neu_basis(POSITION).2[2],
            ],
        );
        assert_eq!(azimuth, 0.0);
        assert!((elevation - FRAC_PI_2).abs() < 1.0e-7, "{elevation}");
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
        let geodetic = skyfield_au_geodetic(POSITION);
        let want = geodetic_enu_az_el(
            &ReceiverFrame::new(FrameRecipe::SppSkyfieldAuThreeIter, POSITION, geodetic),
            receiver_delta(POSITION, sat),
        );
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
        let geodetic = canonical_wgs84_geodetic(POSITION);
        let want = geodetic_enu_az_el(
            &ReceiverFrame::new(FrameRecipe::CanonicalWgs84, POSITION, geodetic),
            receiver_delta(POSITION, sat),
        );
        assert_eq!(got.az_rad.to_bits(), want.az_rad.to_bits());
        assert_eq!(got.el_rad.to_bits(), want.el_rad.to_bits());
        assert_eq!(
            got.geodetic.lat_rad.to_bits(),
            want.geodetic.lat_rad.to_bits()
        );
    }

    #[test]
    fn receiver_frame_memo_matches_wrapper_bits_for_both_recipes_and_edge_positions() {
        let positions = [
            POSITION,
            [WGS84_A_KM * KM_TO_M, 0.0, 0.0],
            [0.0, WGS84_A_KM * KM_TO_M, 0.0],
            [0.0, 0.0, 6_356_752.314_245],
            [-3_912_961.0, -3_656_402.0, -3_178_234.0],
        ];
        let recipes = [
            FrameRecipe::SppSkyfieldAuThreeIter,
            FrameRecipe::CanonicalWgs84,
        ];
        let satellites = [
            [15_600_000.0, -20_400_000.0, 9_800_000.0],
            [-20_100_000.0, 13_800_000.0, -7_300_000.0],
        ];

        for recipe in recipes {
            for receiver in positions {
                let memo = ReceiverFrameMemo::new();
                for satellite in satellites {
                    let expected = az_el_from_ecef(recipe, receiver, satellite);
                    let got = memo.az_el(recipe, receiver, satellite);
                    assert_eq!(got.az_rad.to_bits(), expected.az_rad.to_bits());
                    assert_eq!(got.el_rad.to_bits(), expected.el_rad.to_bits());
                    assert_eq!(
                        got.geodetic.lat_rad.to_bits(),
                        expected.geodetic.lat_rad.to_bits()
                    );
                    assert_eq!(
                        got.geodetic.lon_rad.to_bits(),
                        expected.geodetic.lon_rad.to_bits()
                    );
                    assert_eq!(
                        got.geodetic.height_m.to_bits(),
                        expected.geodetic.height_m.to_bits()
                    );
                }
            }
        }

        let memo = ReceiverFrameMemo::new();
        for (recipe, receiver) in [
            (FrameRecipe::SppSkyfieldAuThreeIter, POSITION),
            (FrameRecipe::CanonicalWgs84, POSITION),
            (
                FrameRecipe::SppSkyfieldAuThreeIter,
                [POSITION[0] + 0.25, POSITION[1], POSITION[2]],
            ),
        ] {
            let satellite = [15_600_000.0, -20_400_000.0, 9_800_000.0];
            let expected = az_el_from_ecef(recipe, receiver, satellite);
            let got = memo.az_el(recipe, receiver, satellite);
            assert_eq!(got.az_rad.to_bits(), expected.az_rad.to_bits());
            assert_eq!(got.el_rad.to_bits(), expected.el_rad.to_bits());
            assert_eq!(
                got.geodetic.lat_rad.to_bits(),
                expected.geodetic.lat_rad.to_bits()
            );
            assert_eq!(
                got.geodetic.lon_rad.to_bits(),
                expected.geodetic.lon_rad.to_bits()
            );
            assert_eq!(
                got.geodetic.height_m.to_bits(),
                expected.geodetic.height_m.to_bits()
            );
        }
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
