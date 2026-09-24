use sidereon_core::constants::F_L1_HZ;
use sidereon_core::frame::Wgs84Geodetic;
use sidereon_core::sbas::{SbasIgp, SbasIonoGrid};

fn igp(lat_deg: f64, lon_deg: f64, vertical_delay_m: f64) -> SbasIgp {
    SbasIgp {
        lat_deg,
        lon_deg,
        vertical_delay_m,
        give_variance_m2: None,
        t0_j2000_s: 0.0,
    }
}

#[test]
fn four_point_stencil_interpolates_at_zenith_ipp() {
    let grid = SbasIonoGrid::new(
        vec![
            igp(-5.0, -5.0, 1.0),
            igp(-5.0, 5.0, 2.0),
            igp(5.0, -5.0, 3.0),
            igp(5.0, 5.0, 4.0),
        ],
        0,
    );
    let receiver = Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid receiver");
    let delay = grid
        .slant_delay_m(receiver, core::f64::consts::FRAC_PI_2, 0.0, F_L1_HZ)
        .expect("covered IPP");
    assert!((delay - 2.5).abs() < 1.0e-12);
}

#[test]
fn three_point_stencil_uses_plane_through_active_points() {
    let grid = SbasIonoGrid::new(
        vec![
            igp(-5.0, -5.0, 1.0),
            igp(-5.0, 5.0, 3.0),
            igp(5.0, -5.0, 5.0),
        ],
        0,
    );
    let receiver = Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid receiver");
    let delay = grid
        .slant_delay_m(receiver, core::f64::consts::FRAC_PI_2, 0.0, F_L1_HZ)
        .expect("covered IPP");
    assert!((delay - 4.0).abs() < 1.0e-12);
}

#[test]
fn near_pole_cell_interpolates() {
    let grid = SbasIonoGrid::new(
        vec![
            igp(75.0, -5.0, 2.0),
            igp(75.0, 5.0, 4.0),
            igp(85.0, -5.0, 6.0),
            igp(85.0, 5.0, 8.0),
        ],
        0,
    );
    let receiver = Wgs84Geodetic::new(80_f64.to_radians(), 0.0, 0.0).expect("valid receiver");
    let delay = grid
        .slant_delay_m(receiver, core::f64::consts::FRAC_PI_2, 0.0, F_L1_HZ)
        .expect("covered IPP");
    assert!((delay - 5.0).abs() < 1.0e-12);
}

#[test]
fn under_determined_cell_returns_none() {
    let grid = SbasIonoGrid::new(vec![igp(-5.0, -5.0, 1.0), igp(-5.0, 5.0, 2.0)], 0);
    let receiver = Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid receiver");
    assert!(grid
        .slant_delay_m(receiver, core::f64::consts::FRAC_PI_2, 0.0, F_L1_HZ)
        .is_none());
}
