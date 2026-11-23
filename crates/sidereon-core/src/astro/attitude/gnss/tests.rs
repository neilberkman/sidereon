use super::*;
use crate::astro::math::vec3::{dot3, norm3};
use std::f64::consts::{FRAC_PI_2, SQRT_2, TAU};

const GM: f64 = 3.986_004_418e14;
const AU: f64 = 1.495_978_707e11;

/// A circular orbit of radius `a` in a plane whose normal makes Sun elevation
/// `beta` with a Sun fixed along +X; `u0` is the argument from noon at t = 0.
#[derive(Clone, Copy)]
struct Circular {
    a: f64,
    beta: f64,
    u0: f64,
}

impl Circular {
    fn n(&self) -> f64 {
        (GM / (self.a * self.a * self.a)).sqrt()
    }

    /// Noon direction p = +X projected in plane, q = n x p, normal n.
    fn frame(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (sb, cb) = (libm::sin(self.beta), libm::cos(self.beta));
        // Sun = cos(beta) p + sin(beta) n with Sun = +X.
        let p = [cb, 0.0, -sb];
        let n = [sb, 0.0, cb];
        let q = cross3(n, p);
        (p, q, n)
    }

    fn sample(&self, t: f64) -> AttitudeSample {
        let (p, q, _) = self.frame();
        let u = self.u0 + self.n() * t;
        let (su, cu) = (libm::sin(u), libm::cos(u));
        let v = self.a * self.n();
        AttitudeSample {
            position_m: [
                self.a * (cu * p[0] + su * q[0]),
                self.a * (cu * p[1] + su * q[1]),
                self.a * (cu * p[2] + su * q[2]),
            ],
            velocity_m_s: [
                v * (-su * p[0] + cu * q[0]),
                v * (-su * p[1] + cu * q[1]),
                v * (-su * p[2] + cu * q[2]),
            ],
            sun_position_m: [AU, 0.0, 0.0],
        }
    }

    fn trajectory(self) -> impl Fn(f64) -> Option<AttitudeSample> {
        move |t| Some(self.sample(t))
    }

    /// Time after t = 0 at which the argument from noon is `u`.
    fn time_at(&self, u: f64) -> f64 {
        (u - self.u0) / self.n()
    }
}

fn deg(x: f64) -> f64 {
    x.to_radians()
}

fn assert_orthonormal(x: [f64; 3], y: [f64; 3], z: [f64; 3]) {
    for v in [x, y, z] {
        assert!((norm3(v) - 1.0).abs() < 1e-14);
    }
    assert!(dot3(x, y).abs() < 1e-14);
    assert!(dot3(y, z).abs() < 1e-14);
    assert!(dot3(z, x).abs() < 1e-14);
    let c = cross3(x, y);
    for i in 0..3 {
        assert!((c[i] - z[i]).abs() < 1e-14, "right-handed");
    }
}

const GPS_A: f64 = 26_559_700.0;
const MEO_A: f64 = 27_906_100.0;

#[test]
fn nominal_yaw_matches_kouba_closed_form_and_geometric_axes() {
    for beta in [-60.0, -10.0, -1.0, 0.5, 15.0, 70.0] {
        for u in [-170.0, -91.0, -45.0, 10.0, 89.0, 135.0, 179.0] {
            let orbit = Circular {
                a: GPS_A,
                beta: deg(beta),
                u0: deg(u),
            };
            let s = orbit.sample(0.0);
            let nominal =
                nominal_yaw(s.position_m, s.velocity_m_s, s.sun_position_m).expect("nominal yaw");
            // Kouba (2009) nominal yaw with mu from midnight, mu = u - pi.
            let mu = deg(u) - PI;
            let expected = libm::atan2(-libm::tan(deg(beta)), libm::sin(mu));
            assert!(
                (wrap_pi(nominal.yaw_rad - expected)).abs() < 1e-12,
                "beta {beta} u {u}: {} vs {expected}",
                nominal.yaw_rad
            );
            assert!((nominal.beta_rad - deg(beta)).abs() < 1e-12);
            assert_orthonormal(nominal.body_x, nominal.body_y, nominal.body_z);
            // Y along sun x r, Z toward the Earth, Sun in the +X hemisphere.
            let y = unit3(cross3(s.sun_position_m, s.position_m)).unwrap();
            for i in 0..3 {
                assert!((nominal.body_y[i] - y[i]).abs() < 1e-12);
            }
            assert!(dot3(nominal.body_x, s.sun_position_m) > 0.0);
        }
    }
}

#[test]
fn nominal_yaw_rejects_invalid_vectors() {
    let s = Circular {
        a: GPS_A,
        beta: 0.3,
        u0: 0.2,
    }
    .sample(0.0);
    assert_eq!(
        nominal_yaw([f64::NAN, 0.0, 0.0], s.velocity_m_s, s.sun_position_m),
        Err(GnssAttitudeError::NonFinite {
            field: "position_m"
        })
    );
    assert_eq!(
        nominal_yaw(s.position_m, s.velocity_m_s, [0.0; 3]),
        Err(GnssAttitudeError::ZeroLength {
            field: "sun_position_m"
        })
    );
    assert_eq!(
        nominal_yaw(s.position_m, s.position_m, s.sun_position_m),
        Err(GnssAttitudeError::ZeroLength {
            field: "position_m x velocity_m_s"
        })
    );
    // Sun along the orbit normal.
    let h = cross3(s.position_m, s.velocity_m_s);
    assert_eq!(
        nominal_yaw(s.position_m, s.velocity_m_s, h),
        Err(GnssAttitudeError::DegenerateGeometry { t_s: None })
    );
}

#[test]
fn dyb_axes_are_sun_fixed_and_orthonormal() {
    let orbit = Circular {
        a: GPS_A,
        beta: deg(20.0),
        u0: deg(40.0),
    };
    let s = orbit.sample(0.0);
    let dyb = dyb_axes(s.position_m, s.sun_position_m).expect("DYB");
    assert_orthonormal(dyb.d, dyb.y, dyb.b);
    let d = unit3(sub3(s.sun_position_m, s.position_m)).unwrap();
    let nominal = nominal_yaw(s.position_m, s.velocity_m_s, s.sun_position_m).unwrap();
    for i in 0..3 {
        assert!((dyb.d[i] - d[i]).abs() < 1e-15);
        // Y is the nominal solar panel axis.
        assert!((dyb.y[i] - nominal.body_y[i]).abs() < 1e-12);
    }
    assert_eq!(
        dyb_axes([GPS_A, 0.0, 0.0], [AU, 0.0, 0.0]),
        Err(GnssAttitudeError::DegenerateGeometry { t_s: None })
    );
}

#[test]
fn trajectory_errors_are_typed() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let none = |_t: f64| -> Option<AttitudeSample> { None };
    assert_eq!(
        gnss_attitude(&law, &none, 5.0),
        Err(GnssAttitudeError::TrajectoryUnavailable { t_s: 5.0 })
    );
    let orbit = Circular {
        a: GPS_A,
        beta: deg(30.0),
        u0: 0.0,
    };
    assert_eq!(
        gnss_attitude(&law, &orbit.trajectory(), f64::INFINITY),
        Err(GnssAttitudeError::NonFinite { field: "t_s" })
    );
    let nan = |t: f64| {
        let mut s = orbit.sample(t);
        s.velocity_m_s[1] = f64::NAN;
        Some(s)
    };
    assert_eq!(
        gnss_attitude(&law, &nan, 0.0),
        Err(GnssAttitudeError::NonFinite {
            field: "velocity_m_s"
        })
    );
    // A trajectory that covers only the requested instant cannot supply the
    // turn start of a small-beta noon turn.
    let small = Circular {
        a: GPS_A,
        beta: deg(0.01),
        u0: deg(-0.1),
    };
    let only_now = |t: f64| (t == 0.0).then(|| small.sample(t));
    assert!(matches!(
        gnss_attitude(&law, &only_now, 0.0),
        Err(GnssAttitudeError::TrajectoryUnavailable { .. })
    ));
}

#[test]
fn law_parameters_are_validated() {
    let orbit = Circular {
        a: GPS_A,
        beta: deg(30.0),
        u0: 0.0,
    };
    let bad = GnssAttitudeLaw::GpsIIR {
        yaw_rate_rad_s: -0.1,
    };
    assert_eq!(
        gnss_attitude(&bad, &orbit.trajectory(), 0.0),
        Err(GnssAttitudeError::InvalidParameter {
            field: "yaw_rate_rad_s",
            value: -0.1
        })
    );
    let bad = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        period_s: f64::NAN,
        ..GALILEO_FOC_TURN
    });
    assert!(matches!(
        gnss_attitude(&bad, &orbit.trajectory(), 0.0),
        Err(GnssAttitudeError::InvalidParameter {
            field: "period_s",
            ..
        })
    ));
}

#[test]
fn gps_iia_defaults_need_a_prn_in_the_table() {
    assert_eq!(
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, None),
        Err(GnssAttitudeError::NoDefaultYawRate { prn: None })
    );
    assert_eq!(
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(33)),
        Err(GnssAttitudeError::NoDefaultYawRate { prn: Some(33) })
    );
    assert_eq!(gps_iia_default_yaw_rate_rad_s(0), None);
    assert_eq!(
        gps_iia_default_yaw_rate_rad_s(11),
        Some(0.2_f64.to_radians())
    );
    assert!(matches!(
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(1)),
        Ok(GnssAttitudeLaw::GpsIIA { .. })
    ));
}

#[test]
fn every_block_has_a_law_and_high_beta_is_nominal_except_geo() {
    let orbit = Circular {
        a: MEO_A,
        beta: deg(35.0),
        u0: deg(-3.0),
    };
    for block in [
        GnssBlock::GpsIIR,
        GnssBlock::GpsIIRM,
        GnssBlock::GpsIIF,
        GnssBlock::GpsIII,
        GnssBlock::GlonassM,
        GnssBlock::GalileoIov,
        GnssBlock::GalileoFoc,
        GnssBlock::Beidou2Igso,
        GnssBlock::Beidou2Meo,
        GnssBlock::Beidou3Igso,
        GnssBlock::Beidou3Meo { manufacturer: None },
        GnssBlock::Beidou3Meo {
            manufacturer: Some(BeidouManufacturer::Cast),
        },
        GnssBlock::Beidou3Meo {
            manufacturer: Some(BeidouManufacturer::Secm),
        },
    ] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::Nominal, "{block}");
        assert_eq!(attitude.yaw_rad, attitude.nominal_yaw_rad);
        assert_orthonormal(attitude.body_x, attitude.body_y, attitude.body_z);
    }
    for block in [GnssBlock::Beidou2Geo, GnssBlock::Beidou3Geo] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::OrbitNormal);
        assert_eq!(attitude.yaw_rad, 0.0);
        // Orbit normal: Y along -n, X along track.
        let s = orbit.sample(0.0);
        let n = unit3(cross3(s.position_m, s.velocity_m_s)).unwrap();
        for i in 0..3 {
            assert!((attitude.body_y[i] + n[i]).abs() < 1e-15);
        }
        assert!(dot3(attitude.body_x, s.velocity_m_s) > 0.0);
    }
}

/// Galileo satellite metadata 3.1.2 and Wang et al. (2018):
/// `psi = 90 s + (psi_0 - 90 s) cos(2 pi t / T)` from the crossing of the
/// start angle, `psi_0` the nominal yaw there and `s` its sign. On a circular
/// orbit with a fixed Sun the crossing is at a known time, so the law can be
/// written out independently. The Galileo start angle is the colinearity
/// angle, `cos(epsilon) = cos(beta) cos(in-plane angle)`; the BeiDou one is
/// the in-plane orbit angle.
#[test]
fn cosine_turns_follow_the_published_formula() {
    for (block, start_deg, period, colinearity) in [
        (GnssBlock::GalileoFoc, 10.0, 5656.0, true),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            6.0,
            3090.0,
            false,
        ),
    ] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        for (beta, noon) in [(1.5, true), (-0.4, true), (2.5, false), (-0.02, false)] {
            let centre = if noon { 0.0 } else { PI };
            let start = if colinearity {
                libm::acos(libm::cos(deg(start_deg)) / libm::cos(deg(beta)))
            } else {
                deg(start_deg)
            };
            let orbit = Circular {
                a: MEO_A,
                beta: deg(beta),
                u0: centre - deg(start_deg) - deg(2.0),
            };
            let t_start = orbit.time_at(centre - start);
            let psi0 = libm::atan2(-libm::tan(deg(beta)), libm::sin(centre - start - PI));
            let sign = if psi0 >= 0.0 { 1.0 } else { -1.0 };
            for k in 1..40 {
                let t = t_start + period * 0.5 * f64::from(k) / 40.0;
                let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
                let expected = FRAC_PI_2_TEST * sign
                    + (psi0 - FRAC_PI_2_TEST * sign) * libm::cos(TAU_TEST * (t - t_start) / period);
                assert_eq!(
                    attitude.mode,
                    if noon {
                        AttitudeMode::NoonTurn
                    } else {
                        AttitudeMode::MidnightTurn
                    },
                    "{block} beta {beta} k {k}"
                );
                // The start instant is found on the trajectory to 1e-9 s;
                // the yaw rate is at most pi / period rad/s.
                assert!(
                    wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
                    "{block} beta {beta} k {k}: {} vs {expected}",
                    attitude.yaw_rad
                );
                assert_orthonormal(attitude.body_x, attitude.body_y, attitude.body_z);
            }
            // Before the start angle and after half a period: nominal.
            for t in [t_start - 30.0, t_start + 0.5 * period + 30.0] {
                let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
                assert_eq!(attitude.mode, AttitudeMode::Nominal, "{block} beta {beta}");
            }
        }
        // Above the beta limit the turn does not start.
        let orbit = Circular {
            a: MEO_A,
            beta: deg(4.5),
            u0: -deg(start_deg) - deg(1.0),
        };
        let attitude = gnss_attitude(&law, &orbit.trajectory(), orbit.time_at(0.0)).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::Nominal);
    }
}

const FRAC_PI_2_TEST: f64 = std::f64::consts::FRAC_PI_2;
const TAU_TEST: f64 = std::f64::consts::TAU;

/// CSNO (2019) as quoted by the BeiDou-3 yaw studies: below `|beta| = 3 deg`
/// the SECM satellites steer with `beta` replaced by `+-3 deg`.
#[test]
fn secm_holds_beta_at_three_degrees() {
    let law = GnssAttitudeLaw::for_block(
        GnssBlock::Beidou3Meo {
            manufacturer: Some(BeidouManufacturer::Secm),
        },
        None,
    )
    .unwrap();
    for beta in [-2.9, -0.5, 0.0, 0.7, 2.99] {
        for u in [-179.0, -20.0, 0.5, 45.0, 178.0] {
            let orbit = Circular {
                a: MEO_A,
                beta: deg(beta),
                u0: deg(u),
            };
            let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0).unwrap();
            let held = if beta >= 0.0 { deg(3.0) } else { -deg(3.0) };
            let expected = libm::atan2(-libm::tan(held), libm::sin(deg(u) - PI));
            assert_eq!(attitude.mode, AttitudeMode::ModifiedYawSteering);
            assert!((wrap_pi(attitude.yaw_rad - expected)).abs() < 1e-12);
        }
    }
    let orbit = Circular {
        a: MEO_A,
        beta: deg(3.2),
        u0: deg(1.0),
    };
    let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0).unwrap();
    assert_eq!(attitude.mode, AttitudeMode::Nominal);
}

#[test]
fn unknown_beidou3_manufacturer_is_an_error_only_where_it_matters() {
    let law =
        GnssAttitudeLaw::for_block(GnssBlock::Beidou3Meo { manufacturer: None }, None).unwrap();
    let high = Circular {
        a: MEO_A,
        beta: deg(8.0),
        u0: deg(2.0),
    };
    assert_eq!(
        gnss_attitude(&law, &high.trajectory(), 0.0).unwrap().mode,
        AttitudeMode::Nominal
    );
    let low = Circular {
        a: MEO_A,
        beta: deg(1.0),
        u0: deg(60.0),
    };
    assert_eq!(
        gnss_attitude(&law, &low.trajectory(), 0.0),
        Err(GnssAttitudeError::BeidouManufacturerRequired { t_s: 0.0 })
    );
}

#[test]
fn attitude_reports_beta_orbit_angle_and_dyb() {
    let orbit = Circular {
        a: GPS_A,
        beta: deg(25.0),
        u0: deg(30.0),
    };
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIF, None).unwrap();
    let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0).unwrap();
    assert!((attitude.beta_rad - deg(25.0)).abs() < 1e-12);
    assert!((attitude.orbit_angle_rad - deg(30.0 - 180.0)).abs() < 1e-12);
    let dyb = attitude.dyb.expect("DYB");
    assert_orthonormal(dyb.d, dyb.y, dyb.b);
}

const GALILEO_A: f64 = 29_599_800.0;
const GLONASS_A: f64 = 25_508_200.0;

/// A circular orbit whose Sun elevation drifts linearly in time,
/// `beta(t) = beta0 + rate t`, with a Sun fixed along +X; `u0` is the argument
/// from noon at t = 0. The plane turns about the noon direction, so the
/// velocity in the plane is the inertial velocity up to the plane's rotation
/// (about 1e-4 of the orbital rate here), which the attitude laws do not use.
#[derive(Clone, Copy)]
struct Drifting {
    a: f64,
    beta0: f64,
    rate: f64,
    u0: f64,
}

impl Drifting {
    fn sample(&self, t: f64) -> AttitudeSample {
        Circular {
            a: self.a,
            beta: self.beta0 + self.rate * t,
            u0: self.u0,
        }
        .sample(t)
    }

    fn period(&self) -> f64 {
        TAU_TEST / (GM / (self.a * self.a * self.a)).sqrt()
    }
}

/// Galileo satellite metadata 3.1.2: the FOC turn starts when the colinearity
/// angle `epsilon = acos(r . y)`, folded into [0, 90 deg], with
/// `y = n x (n x s)`, falls below 10 deg. With unit `r`, `n` and `s`,
/// `r . y = -r . s`, so `cos(epsilon) = |r . s| = cos(beta) cos(in-plane
/// angle)`: at beta = 4 deg the turn starts 9.156 deg before noon, not 10.
#[test]
fn galileo_foc_turn_starts_at_the_colinearity_angle() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GalileoFoc, None).unwrap();
    for (beta_deg, noon) in [(4.0, true), (-4.0, false), (3.0, true), (-1.0, false)] {
        let beta = deg(beta_deg);
        let centre = if noon { 0.0 } else { PI };
        let start_in_plane = libm::acos(libm::cos(deg(10.0)) / libm::cos(beta));
        let orbit = Circular {
            a: GALILEO_A,
            beta,
            u0: centre - deg(12.0),
        };
        // Inside 10 deg in plane, outside 10 deg in colinearity: nominal.
        let before = 0.5 * (start_in_plane + deg(10.0));
        let t = orbit.time_at(centre - before);
        let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
        assert_eq!(
            attitude.mode,
            AttitudeMode::Nominal,
            "beta {beta_deg}: colinearity still above 10 deg"
        );
        let t_start = orbit.time_at(centre - start_in_plane);
        let psi0 = libm::atan2(-libm::tan(beta), libm::sin(centre - start_in_plane - PI));
        let sign = if psi0 >= 0.0 { 1.0 } else { -1.0 };
        for k in 1..40 {
            let t = t_start + 5656.0 * 0.5 * f64::from(k) / 40.0;
            let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
            let expected = FRAC_PI_2_TEST * sign
                + (psi0 - FRAC_PI_2_TEST * sign) * libm::cos(TAU_TEST * (t - t_start) / 5656.0);
            assert_ne!(
                attitude.mode,
                AttitudeMode::Nominal,
                "beta {beta_deg} k {k}"
            );
            assert!(
                wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
                "beta {beta_deg} k {k}: {} vs {expected}",
                attitude.yaw_rad
            );
        }
        // One millisecond either side of the start.
        let just_before = gnss_attitude(&law, &orbit.trajectory(), t_start - 1e-3).unwrap();
        let just_after = gnss_attitude(&law, &orbit.trajectory(), t_start + 1e-3).unwrap();
        assert_eq!(just_before.mode, AttitudeMode::Nominal, "beta {beta_deg}");
        assert_ne!(just_after.mode, AttitudeMode::Nominal, "beta {beta_deg}");
    }
}

/// BeiDou-2 switches at the orbit quadrature: from yaw steering to orbit
/// normal when `|beta|` there is at most 4.1 deg while `|beta|` decreases, and
/// back when it is above 3.9 deg while `|beta|` increases. A drift of 0.05 deg
/// per orbit leaves `|beta|` within 0.1 deg of both limits for a whole orbit,
/// so only the value at the last quadrature decides.
#[test]
fn beidou2_switch_is_decided_at_the_quadrature_for_slow_beta_drift() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::Beidou2Meo, None).unwrap();
    let probe = Drifting {
        a: MEO_A,
        beta0: 0.0,
        rate: 0.0,
        u0: 0.0,
    };
    let per_orbit = deg(0.05) / probe.period();
    // Evaluated at orbit angle 100 deg from midnight (argument from noon
    // -80 deg), 10 deg after the quadrature.
    for (beta_now, rate, expected) in [
        // |beta| increasing, 3.9286 deg at the quadrature: yaw steering.
        (3.93, per_orbit, AttitudeMode::Nominal),
        // |beta| decreasing, 4.0614 deg at the quadrature: orbit normal.
        (4.06, -per_orbit, AttitudeMode::OrbitNormal),
        // Negative beta, |beta| increasing past 3.9 at the quadrature.
        (-3.93, -per_orbit, AttitudeMode::Nominal),
        // Negative beta, |beta| decreasing below 4.1 at the quadrature.
        (-4.06, per_orbit, AttitudeMode::OrbitNormal),
    ] {
        let orbit = Drifting {
            a: MEO_A,
            beta0: deg(beta_now),
            rate,
            u0: deg(-80.0),
        };
        let trajectory = move |t: f64| Some(orbit.sample(t));
        let attitude = gnss_attitude(&law, &trajectory, 0.0).unwrap();
        assert_eq!(attitude.mode, expected, "beta {beta_now} rate {rate:e}");
    }
}

/// The GLONASS-M noon-turn end yaw (Dilssner et al. 2011) solves
/// `yaw_end = |phi(beta, d) - phi(beta, -d)| / 2` with `d = mu_rate yaw_end /
/// R`. `yaw_end = 0` solves it trivially; the turn uses the positive root.
#[test]
fn glonass_noon_turn_uses_the_positive_end_yaw_root() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GlonassM, None).unwrap();
    let rate = GLONASS_M_DEFAULT_YAW_RATE_RAD_S;
    for beta_deg in [2.0, -2.0, 2.03, 1.5] {
        let beta = deg(beta_deg);
        let orbit = Circular {
            a: GLONASS_A,
            beta,
            u0: deg(-1.0),
        };
        let n = orbit.n();
        let half_swing = |y: f64| {
            let d = n * y / rate;
            0.5 * (libm::atan2(-libm::tan(beta), libm::sin(d))
                - libm::atan2(-libm::tan(beta), -libm::sin(d)))
            .abs()
        };
        // y - half_swing(y) is negative just above 0 and positive at pi.
        let (mut lo, mut hi) = (1e-12, PI);
        assert!(lo - half_swing(lo) < 0.0 && hi - half_swing(hi) > 0.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid - half_swing(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let half_span = n * 0.5 * (lo + hi) / rate;
        let t_noon = orbit.time_at(0.0);
        let t_start = t_noon - half_span / n;
        let phi_start = libm::atan2(-libm::tan(beta), libm::sin(half_span));
        for dt in [1.0, 0.25 * (t_noon - t_start), t_noon - t_start - 1.0] {
            let t = t_start + dt;
            let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
            assert_eq!(
                attitude.mode,
                AttitudeMode::NoonTurn,
                "beta {beta_deg} dt {dt}"
            );
            let expected = phi_start - rate.copysign(beta) * dt;
            assert!(
                wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
                "beta {beta_deg} dt {dt}: {} vs {expected}",
                attitude.yaw_rad
            );
        }
    }
}

/// GLONASS-M crosses the shadow at the maximum rate and holds the end yaw once
/// reached (Dilssner et al. 2011). At beta = 0 the end yaw is 0.
#[test]
fn glonass_shadow_holds_the_end_yaw_at_beta_zero() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GlonassM, None).unwrap();
    let orbit = Circular {
        a: GLONASS_A,
        beta: 0.0,
        u0: PI - deg(20.0),
    };
    let t_midnight = orbit.time_at(PI);
    for dt in [-500.0, 0.0, 600.0, 1000.0, 1500.0] {
        let attitude = gnss_attitude(&law, &orbit.trajectory(), t_midnight + dt).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::Shadow, "dt {dt}");
        assert!(
            attitude.yaw_rad.abs() < 1e-15,
            "dt {dt}: yaw {} past the end yaw 0",
            attitude.yaw_rad
        );
    }
}

/// beta = 0 exactly: every law gives an attitude at every instant around noon
/// and midnight, including the turn centres where the Sun is along the radial
/// direction.
#[test]
fn every_law_is_defined_at_beta_zero() {
    for (block, prn, a) in [
        (GnssBlock::GpsIIA, Some(1), GPS_A),
        (GnssBlock::GpsIIR, None, GPS_A),
        (GnssBlock::GpsIIF, None, GPS_A),
        (GnssBlock::GpsIII, None, GPS_A),
        (GnssBlock::GlonassM, None, GLONASS_A),
        (GnssBlock::GalileoIov, None, GALILEO_A),
        (GnssBlock::GalileoFoc, None, GALILEO_A),
        (GnssBlock::Beidou2Meo, None, MEO_A),
        (GnssBlock::Beidou2Igso, None, 42_164_200.0),
        (GnssBlock::Beidou3Igso, None, 42_164_200.0),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            None,
            MEO_A,
        ),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Secm),
            },
            None,
            MEO_A,
        ),
    ] {
        let law = GnssAttitudeLaw::for_block(block, prn).unwrap();
        let orbit = Circular {
            a,
            beta: 0.0,
            u0: -deg(30.0),
        };
        for centre in [0.0, PI] {
            let t_centre = orbit.time_at(centre);
            for k in -8..=8 {
                let t = t_centre + 0.25 * f64::from(k);
                let attitude = gnss_attitude(&law, &orbit.trajectory(), t)
                    .unwrap_or_else(|e| panic!("{block} centre {centre} k {k}: {e}"));
                assert_finite(&attitude);
            }
        }
        // The satellite exactly on the Sun line at t = 0: u0 = 0 puts the
        // position along +X, and the Sun at +X (noon) or -X (midnight).
        let on_line = Circular {
            a,
            beta: 0.0,
            u0: 0.0,
        };
        for sun_sign in [1.0, -1.0] {
            let trajectory = move |t: f64| {
                let mut s = on_line.sample(t);
                s.sun_position_m[0] *= sun_sign;
                Some(s)
            };
            let s = trajectory(0.0).unwrap();
            assert_eq!(cross3(s.sun_position_m, s.position_m), [0.0; 3]);
            for t in [0.0, 1e-6, 1.0, 100.0, 1000.0] {
                let attitude = gnss_attitude(&law, &trajectory, t)
                    .unwrap_or_else(|e| panic!("{block} Sun sign {sun_sign} t {t}: {e}"));
                assert_finite(&attitude);
            }
        }
    }
    // GPS IIR at beta = 0 turns from the noon instant at the maximum rate, so
    // the yaw is -R dt (beta = +0, the turn starts at yaw 0).
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let orbit = Circular {
        a: GPS_A,
        beta: 0.0,
        u0: -deg(30.0),
    };
    let t_noon = orbit.time_at(0.0);
    for dt in [10.0, 450.0, 800.0] {
        let attitude = gnss_attitude(&law, &orbit.trajectory(), t_noon + dt).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "dt {dt}");
        let expected = -GPS_IIR_DEFAULT_YAW_RATE_RAD_S * dt;
        assert!(
            wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
            "dt {dt}: {} vs {expected}",
            attitude.yaw_rad
        );
    }
}

fn assert_finite(attitude: &GnssAttitude) {
    let scalars = [
        attitude.yaw_rad,
        attitude.nominal_yaw_rad,
        attitude.beta_rad,
        attitude.orbit_angle_rad,
    ];
    assert!(scalars.iter().all(|x| x.is_finite()), "{attitude:?}");
    for axis in [attitude.body_x, attitude.body_y, attitude.body_z] {
        assert!(axis.iter().all(|x| x.is_finite()), "{attitude:?}");
    }
    if let Some(dyb) = attitude.dyb {
        for axis in [dyb.d, dyb.y, dyb.b] {
            assert!(axis.iter().all(|x| x.is_finite()), "{attitude:?}");
        }
    }
}

/// `|beta|` exactly at the shadow limit: the shadow has zero length (for GPS
/// IIF the shadow-crossing rate `(phi_end - phi_start) / (end - start)` is
/// 0 / 0), and the limit of every shadow law is nominal yaw steering.
#[test]
fn grazing_shadow_is_nominal_yaw_steering() {
    let orbit = Circular {
        a: GPS_A,
        beta: deg(13.25),
        u0: PI - deg(5.0),
    };
    let t_midnight = orbit.time_at(PI);
    for dt in [0.0, 100.0, 1000.0] {
        let t = t_midnight + dt;
        let beta = gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &orbit.trajectory(), t)
            .unwrap()
            .beta_rad;
        // The shadow limit whose in-plane half span at this beta is exactly 0.
        let mut limit = PI - beta;
        let mut found = false;
        for _ in 0..256 {
            let ratio = libm::cos(limit - PI) / libm::cos(beta);
            if ratio == 1.0 {
                found = true;
                break;
            }
            limit = if ratio > 1.0 {
                f64::from_bits(limit.to_bits() - 1)
            } else {
                f64::from_bits(limit.to_bits() + 1)
            };
        }
        assert!(found, "dt {dt}: no limit with a zero half span");
        for law in [
            GnssAttitudeLaw::GpsIIF {
                yaw_rate_rad_s: GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
                yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: limit,
            },
            GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: gps_iia_default_yaw_rate_rad_s(1).unwrap(),
                yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: limit,
            },
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
                shadow_limit_rad: limit,
            },
        ] {
            let attitude = gnss_attitude(&law, &orbit.trajectory(), t)
                .unwrap_or_else(|e| panic!("{law:?} dt {dt}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::Nominal, "{law:?} dt {dt}");
        }
    }
}

/// `acos` arguments formed from rounded values can leave [-1, 1]: with a tiny
/// beta the Sun-satellite cosine rounds to 1 while `cos(beta)` is below 1, so
/// `cos(angle) / cos(beta)` exceeds 1.
#[test]
fn orbit_plane_angles_clamp_their_acos_arguments() {
    let beta = 3.0e-8;
    assert!(libm::cos(beta) < 1.0);
    let context = geometry::Context {
        t_s: 0.0,
        position: [-GPS_A, 0.0, 0.0],
        sun: [AU, 0.0, 0.0],
        e_r: [-1.0, 0.0, 0.0],
        e_n: [0.0, 0.0, 1.0],
        mu_rate: 1.46e-4,
        svb_cos: -1.0,
        beta,
    };
    let angle = context.in_orbit_plane_absolute_angle(0.5 * beta);
    assert!(angle.is_finite() && angle == 0.0, "{angle}");
}

/// Finite inputs whose products overflow are `NonFinite` naming the product,
/// never NaN results, wrong values or another error.
#[test]
fn overflowing_products_are_non_finite_errors() {
    assert_eq!(
        dyb_axes([1e308, 1e308, 0.0], [1e308, 0.0, 0.0]),
        Err(GnssAttitudeError::NonFinite {
            field: "position_m"
        })
    );
    assert_eq!(
        dyb_axes([1e154, 0.0, 0.0], [-1e154, 1e150, 0.0]),
        Err(GnssAttitudeError::NonFinite {
            field: "sun_position_m - position_m"
        })
    );
    // |s| |r x v| = 1.4e161: the cross product sun x (r x v) has an
    // infinite norm.
    let position = [GPS_A, 0.0, 0.0];
    let velocity = [0.0, 3874.0, 0.0];
    let sun = [1e150, 0.0, 1e150];
    let expected = Err(GnssAttitudeError::NonFinite {
        field: "sun_position_m x (position_m x velocity_m_s)",
    });
    assert_eq!(nominal_yaw(position, velocity, sun).map(|_| ()), expected);
    let trajectory = move |_t: f64| {
        Some(AttitudeSample {
            position_m: position,
            velocity_m_s: velocity,
            sun_position_m: sun,
        })
    };
    for block in [
        GnssBlock::GpsIIR,
        GnssBlock::GalileoFoc,
        GnssBlock::Beidou2Meo,
    ] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        assert_eq!(gnss_attitude(&law, &trajectory, 0.0).map(|_| ()), expected);
    }
    // A GPS orbit and Sun scaled to |r| = 1e50 m and |s| = 1e55 m: beta, the
    // orbit angle and the orbital rate are unchanged, and so is every
    // attitude, also where the turn's end search forms the Sun's direction
    // across the radial one (|r|^2 |s| = 1e155 would overflow when squared;
    // it is formed from unit vectors).
    let orbit = Circular {
        a: GPS_A,
        beta: deg(0.5),
        u0: -deg(0.5),
    };
    let scaled = move |t: f64| {
        let s = orbit.sample(t);
        let k = 1e50 / GPS_A;
        Some(AttitudeSample {
            position_m: s.position_m.map(|x| x * k),
            velocity_m_s: s.velocity_m_s.map(|x| x * k),
            sun_position_m: s.sun_position_m.map(|x| x * (1e55 / AU)),
        })
    };
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    for t in [0.0, 300.0, 600.0] {
        let big = gnss_attitude(&law, &scaled, t).unwrap_or_else(|e| panic!("t {t}: {e}"));
        let plain = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
        assert_eq!(big.mode, plain.mode, "t {t}");
        assert!(
            wrap_pi(big.yaw_rad - plain.yaw_rad).abs() < 1e-12,
            "t {t}: {} vs {}",
            big.yaw_rad,
            plain.yaw_rad
        );
    }
    assert_eq!(
        gnss_attitude(&law, &orbit.trajectory(), 300.0)
            .unwrap()
            .mode,
        AttitudeMode::NoonTurn
    );
}

/// A satellite on a straight line through `position` with constant
/// `velocity`, the Sun fixed at `sun`.
fn straight_line(
    position: [f64; 3],
    velocity: [f64; 3],
    sun: [f64; 3],
) -> impl Fn(f64) -> Option<AttitudeSample> {
    move |t| {
        Some(AttitudeSample {
            position_m: std::array::from_fn(|k| position[k] + velocity[k] * t),
            velocity_m_s: velocity,
            sun_position_m: sun,
        })
    }
}

/// A circular orbit of radius `a` at orbital rate `n` (any `n`; the
/// velocity is `a n`), in the plane of the Sun along +X at 1 m (beta = 0),
/// with noon at `t_noon`; the argument from noon is `n (t - t_noon)`, exact
/// to rounding for `t` near `t_noon` however large.
fn sun_plane_circle(a: f64, n: f64, t_noon: f64) -> impl Fn(f64) -> Option<AttitudeSample> {
    move |t| {
        // Each product is finite where `t - t_noon` is not.
        let u = n * t - n * t_noon;
        let (su, cu) = (libm::sin(u), libm::cos(u));
        Some(AttitudeSample {
            position_m: [a * cu, a * su, 0.0],
            velocity_m_s: [-a * n * su, a * n * cu, 0.0],
            sun_position_m: [1.0, 0.0, 0.0],
        })
    }
}

/// The cosine phase is the elapsed fraction of the period times 2 pi: a period
/// near `f64::MAX` with half of it elapsed has a finite phase, where forming
/// `2 pi x elapsed` first overflows. A circular orbit of period 1.6e308 s
/// (radius 1e150 m) with noon at 0.7e308 s, the turn from 1.4 rad before it.
/// A straight line past the Earth is no orbit (no conic of eccentricity below
/// 0.9 holds it): [`GnssAttitudeError::OutsideValidatedDomain`].
#[test]
fn cosine_turn_phase_is_finite_for_periods_near_the_largest_float() {
    let period = 1.5e308;
    let start = 1.4;
    let law = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        beta_limit_rad: 0.5,
        beta_limit_inclusive: true,
        start_angle: CosineTurnStartAngle::OrbitAngle,
        start_angle_rad: start,
        period_s: period,
    });
    let n = TAU / 1.6e308;
    let t_s = 0.7e308;
    let trajectory = sun_plane_circle(1.0e150, n, t_s);
    let elapsed = start / n;
    assert!(elapsed <= 0.5 * period);
    let attitude = gnss_attitude(&law, &trajectory, t_s).expect("cosine turn");
    assert_eq!(attitude.mode, AttitudeMode::NoonTurn);
    // beta = 0, so the start yaw is 0 (of either sign) and the yaw is
    // +-(pi / 2) (1 - cos(2 pi elapsed / period)).
    let expected = FRAC_PI_2 * (1.0 - libm::cos(elapsed / period * TAU));
    assert!(
        (attitude.yaw_rad.abs() - expected).abs() < 1e-9,
        "{} vs +-{expected}",
        attitude.yaw_rad
    );
    assert_orthonormal(attitude.body_x, attitude.body_y, attitude.body_z);
    let line = straight_line(
        [1.0e148, -7.0e148, 0.0],
        [0.0, 1.0e-159, 0.0],
        [1.0, 0.0, 0.0],
    );
    assert!(outside_domain(&gnss_attitude(&law, &line, t_s)));
}

/// A GLONASS-M noon turn needs the nominal yaw half swing over the turn to
/// equal the maximum-rate swing. With the maximum yaw rate below the orbital
/// rate the smallest such turn can reach past the quadrature into the night
/// half orbit, where the law has no definition: those instants are
/// `InvalidParameter`. Where the turn ends before the quadrature, it is the
/// smallest positive root, so the yaw meets the nominal yaw at the turn end.
#[test]
fn glonass_noon_turn_with_a_yaw_rate_below_the_orbital_rate() {
    let n = Circular {
        a: GLONASS_A,
        beta: 0.0,
        u0: 0.0,
    }
    .n();
    // k = 10 at beta = 10 deg: the maximum-rate swing over a half span d is
    // d / 10, the nominal half swing at the quadrature only 80 deg, so the
    // root lies beyond the quadrature.
    let rate = n / 10.0;
    let law = GnssAttitudeLaw::GlonassM {
        yaw_rate_rad_s: rate,
        shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
    };
    let orbit = Circular {
        a: GLONASS_A,
        beta: deg(10.0),
        u0: -deg(30.0),
    };
    for u in [-30.0, 0.0, 60.0, 100.0] {
        assert_eq!(
            gnss_attitude(&law, &orbit.trajectory(), orbit.time_at(deg(u))),
            Err(GnssAttitudeError::InvalidParameter {
                field: "yaw_rate_rad_s",
                value: rate
            }),
            "u {u}"
        );
    }
    // Midnight lies beyond the noon turn: the shadow law applies there.
    let midnight = gnss_attitude(&law, &orbit.trajectory(), orbit.time_at(PI)).expect("midnight");
    assert_eq!(midnight.mode, AttitudeMode::Shadow);

    // k = 2 at beta = 60 deg: the root has d below the quadrature (the half
    // swing there is 30 deg, beyond the 45 deg / 2 the rate allows).
    let k = 2.0;
    let beta = deg(60.0);
    let law = GnssAttitudeLaw::GlonassM {
        yaw_rate_rad_s: n / k,
        shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
    };
    let half_swing = |d: f64| {
        0.5 * (libm::atan2(-libm::tan(beta), libm::sin(d))
            - libm::atan2(-libm::tan(beta), -libm::sin(d)))
        .abs()
    };
    // The smallest positive root of y = half_swing(k y), on (0, pi / (2 k)].
    let (mut lo, mut hi) = (1e-6, FRAC_PI_2 / k);
    assert!(lo - half_swing(k * lo) < 0.0 && hi - half_swing(k * hi) > 0.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid - half_swing(k * mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let d = k * hi;
    let orbit = Circular {
        a: GLONASS_A,
        beta,
        u0: -d - deg(1.0),
    };
    let trajectory = orbit.trajectory();
    let t_end = orbit.time_at(d);
    let inside = gnss_attitude(&law, &trajectory, t_end - 1e-3).expect("turn end");
    assert_eq!(inside.mode, AttitudeMode::NoonTurn);
    assert!(
        wrap_pi(inside.yaw_rad - inside.nominal_yaw_rad).abs() < 1e-6,
        "{} vs nominal {}",
        inside.yaw_rad,
        inside.nominal_yaw_rad
    );
    let after = gnss_attitude(&law, &trajectory, t_end + 1e-3).expect("after the turn");
    assert_eq!(after.mode, AttitudeMode::Nominal);
}

/// A shadow crossing of positive in-plane half span whose start and end round
/// to the same instant (here 2e-4 s long at t = 1e13 s, where one unit in the
/// last place is 2e-3 s) is the grazing limit: nominal yaw steering, not a
/// 0 / 0 crossing rate.
#[test]
fn shadow_shorter_than_the_time_resolution_is_nominal_yaw_steering() {
    let t0 = 1.0e13;
    let orbit = Circular {
        a: GPS_A,
        beta: deg(13.25),
        u0: PI,
    };
    let trajectory = move |t: f64| Some(orbit.sample(t - t0));
    let beta = gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &trajectory, t0)
        .unwrap()
        .beta_rad;
    // The shadow limit with the largest cos(limit - pi) / cos(beta) below 1.
    let ratio = |limit: f64| libm::cos(limit - PI) / libm::cos(beta);
    let mut limit = PI - beta;
    for _ in 0..1024 {
        if ratio(limit) < 1.0 {
            break;
        }
        limit = f64::from_bits(limit.to_bits() - 1);
    }
    let r = ratio(limit);
    assert!(r < 1.0 && 1.0 - r < 1e-15, "ratio {r}");
    let half_span = libm::acos(r);
    let n = orbit.n();
    assert!(half_span > 0.0 && half_span / n < 4.0e-4, "{half_span}");
    for law in [
        GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s: GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
            yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
            shadow_limit_rad: limit,
        },
        GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s: gps_iia_default_yaw_rate_rad_s(1).unwrap(),
            yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
            shadow_limit_rad: limit,
        },
        GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
            shadow_limit_rad: limit,
        },
    ] {
        let attitude =
            gnss_attitude(&law, &trajectory, t0).unwrap_or_else(|e| panic!("{law:?}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::Nominal, "{law:?}");
    }
}

/// The orbital rate and the Sun-satellite cosine rate divide by `|r|^2`: a
/// position whose square underflows is `ZeroLength`, never a subnormal
/// divisor whose quotients overflow into a NaN rate that flips the approach
/// direction. With `|r|^2` normal and `|v|^2` finite both quotients stay below
/// `|v| / |r| < 9e307`.
#[test]
fn sun_satellite_cosine_rate_is_guarded() {
    for law in [
        GnssAttitudeLaw::OrbitNormal,
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap(),
    ] {
        // |r|^2 = 1e-320 is subnormal; (r . v) / |r|^2 would be 1e310.
        let tiny = straight_line([1e-160, 0.0, 0.0], [1e150, 1e140, 0.0], [1e20, 0.0, 0.0]);
        assert_eq!(
            gnss_attitude(&law, &tiny, 0.0).map(|_| ()),
            Err(GnssAttitudeError::ZeroLength {
                field: "position_m"
            })
        );
    }
    // The smallest normal |r|^2 with the largest |v|: (r . v) / |r|^2 is
    // 8.7e307, finite.
    let extreme = straight_line(
        [1.5e-154, 0.0, 0.0],
        [1.3e154, 1e140, 0.0],
        [1e20, 0.0, 0.0],
    );
    let attitude =
        gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &extreme, 0.0).expect("orbit normal");
    assert_eq!(attitude.mode, AttitudeMode::OrbitNormal);
    assert!(attitude.beta_rad.is_finite() && attitude.orbit_angle_rad.is_finite());
    assert_orthonormal(attitude.body_x, attitude.body_y, attitude.body_z);
}

/// A cosine turn is classified (noon or midnight) and its start condition
/// tested at the turn start, not at the evaluated instant.
#[test]
fn cosine_turn_is_classified_at_its_start() {
    // A turn lasting half an orbit from 80 deg before noon is still the noon
    // turn at 91 deg after noon.
    let orbit = Circular {
        a: MEO_A,
        beta: deg(1.0),
        u0: -deg(85.0),
    };
    let period = TAU / orbit.n();
    let law = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        beta_limit_rad: deg(3.0),
        beta_limit_inclusive: true,
        start_angle: CosineTurnStartAngle::OrbitAngle,
        start_angle_rad: deg(80.0),
        period_s: period,
    });
    let trajectory = orbit.trajectory();
    let t_start = orbit.time_at(-deg(80.0));
    let start = orbit.sample(t_start);
    let psi0 = nominal_yaw(start.position_m, start.velocity_m_s, start.sun_position_m)
        .unwrap()
        .yaw_rad;
    let cosine_yaw = |psi0: f64, elapsed: f64, period: f64| {
        let quarter = FRAC_PI_2.copysign(psi0);
        quarter + (psi0 - quarter) * libm::cos(elapsed / period * TAU)
    };
    for u in [-10.0, 89.0, 91.0, 99.0] {
        let t = orbit.time_at(deg(u));
        let attitude = gnss_attitude(&law, &trajectory, t).expect("turn");
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "u {u}");
        let expected = cosine_yaw(psi0, t - t_start, period);
        assert!(
            wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
            "u {u}: {} vs {expected}",
            attitude.yaw_rad
        );
    }

    // A colinearity start of 10 deg crossed at |beta| = 9.9 deg (below the
    // 9.95 deg limit) while |beta| reaches 10.05 deg later in the turn, where
    // the colinearity angle can no longer reach 10 deg. The orbit lies in the
    // XY plane; the Sun rises out of it at a constant rate, 0.049 rad over
    // the orbit.
    let a = MEO_A;
    let n = (GM / (a * a * a)).sqrt();
    let start_angle = deg(10.0);
    let beta_start = deg(9.9);
    let u_start = -libm::acos(libm::cos(start_angle) / libm::cos(beta_start));
    let t_s = 2500.0;
    // The crossing is at t = 0: u(0) = u_start, beta(0) = beta_start.
    let beta_rate = (deg(10.05) - beta_start) / t_s;
    let rising = move |t: f64| {
        let u = u_start + n * t;
        let b = beta_start + beta_rate * t;
        let (su, cu) = (libm::sin(u), libm::cos(u));
        Some(AttitudeSample {
            position_m: [a * cu, a * su, 0.0],
            velocity_m_s: [-a * n * su, a * n * cu, 0.0],
            sun_position_m: [AU * libm::cos(b), 0.0, AU * libm::sin(b)],
        })
    };
    let turn_period = 5656.0;
    let law = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        beta_limit_rad: deg(9.95),
        beta_limit_inclusive: false,
        start_angle: CosineTurnStartAngle::Colinearity,
        start_angle_rad: start_angle,
        period_s: turn_period,
    });
    let attitude = gnss_attitude(&law, &rising, t_s).expect("turn");
    assert!(attitude.beta_rad > deg(10.0), "{}", attitude.beta_rad);
    assert_eq!(attitude.mode, AttitudeMode::NoonTurn);
    let start = rising(0.0).unwrap();
    let psi0 = nominal_yaw(start.position_m, start.velocity_m_s, start.sun_position_m)
        .unwrap()
        .yaw_rad;
    let expected = cosine_yaw(psi0, t_s, turn_period);
    assert!(
        wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
        "{} vs {expected}",
        attitude.yaw_rad
    );
}

/// Every instant of a GPS or GLONASS-M turn finds the same turn, solved on
/// the trajectory, so the yaw is a function of the turn and not of the
/// requested instant: here with beta drifting by 2e-7 rad/s (the Sun's
/// largest rate), so that a start
/// estimated from each instant's beta and orbit angle would move between
/// instants. Inside the turn the yaw changes at the turn rate: the yaw
/// `phi_start + phi_dot (t - t_start)` at two instants of one turn differs by
/// `phi_dot (t2 - t1)` to the rounding of two yaw angles up to pi, of the
/// products and of `t - t_start` at t of order 1e4 s (each a few 1e-16 rad,
/// times the yaw rate for the time), under 1e-14 rad. A query's result does not
/// depend on the queries before it.
#[test]
fn turns_have_one_start_for_every_instant() {
    let iia_rate = gps_iia_default_yaw_rate_rad_s(1).unwrap();
    let cases = [
        (GnssBlock::GpsIIA, GPS_A, true, Some(-iia_rate)),
        (GnssBlock::GpsIIA, GPS_A, false, Some(iia_rate)),
        (
            GnssBlock::GpsIIR,
            GPS_A,
            true,
            Some(-GPS_IIR_DEFAULT_YAW_RATE_RAD_S),
        ),
        (
            GnssBlock::GpsIIR,
            GPS_A,
            false,
            Some(GPS_IIR_DEFAULT_YAW_RATE_RAD_S),
        ),
        (
            GnssBlock::GpsIIF,
            GPS_A,
            true,
            Some(-GPS_IIF_DEFAULT_YAW_RATE_RAD_S),
        ),
        // The IIF shadow rate links the nominal yaws at entry and exit.
        (GnssBlock::GpsIIF, GPS_A, false, None),
        (
            GnssBlock::GlonassM,
            GLONASS_A,
            true,
            Some(-GLONASS_M_DEFAULT_YAW_RATE_RAD_S),
        ),
        (
            GnssBlock::GlonassM,
            GLONASS_A,
            false,
            Some(GLONASS_M_DEFAULT_YAW_RATE_RAD_S),
        ),
    ];
    for (block, a, noon, rate) in cases {
        let law = GnssAttitudeLaw::for_block(block, Some(1)).unwrap();
        let orbit = Drifting {
            a,
            beta0: deg(1.0),
            rate: 2.0e-7,
            u0: -deg(30.0),
        };
        let trajectory = move |t: f64| Some(orbit.sample(t));
        let n = TAU_TEST / orbit.period();
        let t_centre = (if noon { 0.0 } else { PI } + deg(30.0)) / n;
        let event_at = |t: f64| {
            let ctx = geometry::Context::at(&trajectory, t).unwrap();
            laws::turn_event(&law, &ctx, &trajectory)
                .unwrap()
                .unwrap_or_else(|| panic!("{block} noon {noon}: no turn at t {t}"))
        };
        let turn = event_at(t_centre);
        assert!(
            turn.start_s < t_centre && t_centre < turn.end_s,
            "{block} noon {noon}"
        );
        // Two instants early in the turn's zone: inside the turn, and before
        // the GLONASS-M shadow yaw reaches its held end yaw.
        let span = turn.end_s - turn.start_s;
        let (t1, t2) = (turn.start_s + 0.05 * span, turn.start_s + 0.15 * span);
        for t in [t1, t2] {
            let other = event_at(t);
            assert_eq!(
                (other.start_s.to_bits(), other.end_s.to_bits()),
                (turn.start_s.to_bits(), turn.end_s.to_bits()),
                "{block} noon {noon} t {t}"
            );
        }
        let y1 = gnss_attitude(&law, &trajectory, t1).unwrap();
        let y2 = gnss_attitude(&law, &trajectory, t2).unwrap();
        assert_ne!(y1.mode, AttitudeMode::Nominal, "{block} noon {noon}");
        assert_eq!(y1.mode, y2.mode, "{block} noon {noon}");
        if let Some(rate) = rate {
            let change = wrap_pi(y2.yaw_rad - y1.yaw_rad);
            let expected = rate * (t2 - t1);
            assert!(
                (change - expected).abs() < 1.0e-14,
                "{block} noon {noon}: yaw change {change} vs {expected}"
            );
        }
        // The same query after others, in another order, gives the same bits.
        for t in [t2, turn.end_s + 500.0, turn.start_s - 500.0, t_centre] {
            gnss_attitude(&law, &trajectory, t).unwrap();
        }
        assert_eq!(gnss_attitude(&law, &trajectory, t1).unwrap(), y1);
    }
}

/// GPS IIA shadow crossings (Bar-Sever 1996; Kouba 2009; Kouba's `eclips.f`,
/// December 2017), written out here for a circular orbit with a fixed Sun,
/// where the nominal yaw `atan2(-tan(beta), sin(mu))` is known in closed form
/// and continuous for beta of one sign: from the nominal yaw at shadow entry
/// the yaw turns at `SIGN(R, b)` through the whole shadow (more than a full
/// turn at the fastest IIA rates and low beta); at the exit
/// `D = psi_n(t_e) - psi(t_e)`, taken in [-180, 180] deg, and the yaw turns at
/// `SIGN(R, D)` until it equals the nominal yaw, at most 1800 s after the
/// exit. The gap `|D| + sign(D) (psi_n(t) - psi_n(t_e)) - R (t - t_e)` closes
/// after the exit; its zero, bisected here, ends the turn. PRNs at rates from
/// 0.0815 to 0.2 deg/s, the published bias of +0.5 deg and a bias of -0.5 deg
/// (the sign PRN 23 carried until 2013 in `eclips.f`).
#[test]
fn gps_iia_shadow_crossing_and_recovery_follow_the_bias_and_the_shorter_way() {
    let mut reversed_recoveries = 0;
    let mut long_shadows = 0;
    for prn in [2_u8, 9, 11, 12, 13, 14, 15, 17, 21, 1] {
        let rate = gps_iia_default_yaw_rate_rad_s(prn).unwrap();
        for bias_deg in [0.5, -0.5] {
            let bias = deg(bias_deg);
            let law = GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: rate,
                yaw_bias_rad: bias,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            };
            for beta_deg in [0.5, 0.01, 2.0, -2.5, -0.3, 8.0] {
                let beta = deg(beta_deg);
                let orbit = Circular {
                    a: GPS_A,
                    beta,
                    u0: PI - deg(40.0),
                };
                let n = orbit.n();
                let t_midnight = orbit.time_at(PI);
                // Shadow: cos(beta) cos(u) < cos(limit), |u - pi| < y_shadow.
                let y_shadow = libm::acos(-libm::cos(GPS_SHADOW_LIMIT_RAD) / libm::cos(beta));
                let (t_in, t_out) = (t_midnight - y_shadow / n, t_midnight + y_shadow / n);
                if (t_out - t_in) * rate > TAU {
                    long_shadows += 1;
                }
                let t_b = libm::tan(beta);
                let nominal = |t: f64| libm::atan2(-t_b, libm::sin(n * (t - t_midnight)));
                let psi0 = nominal(t_in);
                let shadow_rate = if bias < 0.0 { -rate } else { rate };
                let in_shadow = |t: f64| psi0 + shadow_rate * (t - t_in);
                let x = nominal(t_out) - in_shadow(t_out);
                let d = x - TAU * libm::round(x / TAU);
                let recovery_rate = rate.copysign(d);
                if recovery_rate * shadow_rate < 0.0 {
                    reversed_recoveries += 1;
                }
                let recovery = |t: f64| in_shadow(t_out) + recovery_rate * (t - t_out);
                let gap = |t: f64| {
                    d.abs() + d.signum() * (nominal(t) - nominal(t_out)) - rate * (t - t_out)
                };
                let limit = t_out + 1800.0;
                let end = if gap(limit) > 0.0 {
                    limit
                } else {
                    let (mut lo, mut hi) = (t_out, limit);
                    while hi - lo > 1e-10 {
                        let mid = 0.5 * (lo + hi);
                        if gap(mid) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    0.5 * (lo + hi)
                };
                let trajectory = orbit.trajectory();
                // A grid every 10 s, skipping the grid points within 1e-7 s
                // of the shadow entry or the turn end (where the two
                // evaluations may differ by their rounding), and probes
                // 1e-5 s either side of the entry, the exit and the end, all
                // checked.
                let mut t = t_in - 100.0;
                let mut checks: Vec<f64> = Vec::new();
                while t < end + 200.0 {
                    if (t - t_in).abs() > 1e-7 && (t - end).abs() > 1e-7 {
                        checks.push(t);
                    }
                    t += 10.0;
                }
                checks.extend([
                    t_in - 1e-5,
                    t_in + 1e-5,
                    t_out - 1e-5,
                    t_out + 1e-5,
                    end - 1e-5,
                    end + 1e-5,
                ]);
                for t in checks {
                    let where_ = format!(
                        "PRN {prn} bias {bias_deg} beta {beta_deg} t - shadow entry {} (shadow {} s, recovery {} s at {recovery_rate:e})",
                        t - t_in,
                        t_out - t_in,
                        end - t_out
                    );
                    let attitude = gnss_attitude(&law, &trajectory, t)
                        .unwrap_or_else(|e| panic!("{where_}: {e}"));
                    let expected = if t > t_in && t <= t_out {
                        Some(in_shadow(t))
                    } else if t > t_out && (t < end || (end == limit && t <= limit)) {
                        Some(recovery(t))
                    } else {
                        None
                    };
                    match expected {
                        Some(yaw) => {
                            assert_eq!(attitude.mode, AttitudeMode::Shadow, "{where_}");
                            let error = wrap_pi(attitude.yaw_rad - yaw).abs();
                            assert!(error < 1e-9, "{where_}: yaw error {error:e}");
                        }
                        None => {
                            assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                        }
                    }
                }
            }
        }
    }
    assert!(
        long_shadows >= 20,
        "{long_shadows} shadows longer than a full turn"
    );
    assert!(
        reversed_recoveries >= 10,
        "{reversed_recoveries} recoveries reverse"
    );
}

/// GPS IIF shadow crossings with the Sun close to the orbit normal and a
/// shadow limit just above 90 deg: the nominal yaws at shadow entry and exit
/// are a few 1e-3 rad apart, so the crossing rate linking them is of order
/// 1e-7 rad/s, and the crossing ends at the shadow exit, where the yaw meets
/// the nominal yaw by construction (Dilssner 2010).
#[test]
fn gps_iif_slow_shadow_crossings_end_at_the_shadow_exit() {
    for (beta_deg, limit_deg) in [(89.9, 90.05), (89.99, 90.005), (60.0, 90.5)] {
        let (beta, limit) = (deg(beta_deg), deg(limit_deg));
        let law = GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s: GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
            yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
            shadow_limit_rad: limit,
        };
        let orbit = Circular {
            a: GPS_A,
            beta,
            u0: PI - deg(100.0),
        };
        let n = orbit.n();
        let t_midnight = orbit.time_at(PI);
        let y_shadow = libm::acos(-libm::cos(limit) / libm::cos(beta));
        let (t_in, t_out) = (t_midnight - y_shadow / n, t_midnight + y_shadow / n);
        let t_b = libm::tan(beta);
        let nominal = |t: f64| libm::atan2(-t_b, libm::sin(n * (t - t_midnight)));
        let (psi_in, psi_out) = (nominal(t_in), nominal(t_out));
        let rate = (psi_out - psi_in) / (t_out - t_in);
        let trajectory = orbit.trajectory();
        for k in 0..=40 {
            let t = t_in + (t_out - t_in) * (0.0025 + 0.995 * f64::from(k) / 40.0);
            let where_ = format!("beta {beta_deg} limit {limit_deg} t - entry {}", t - t_in);
            let attitude =
                gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::Shadow, "{where_}");
            let expected = psi_in + rate * (t - t_in);
            let error = wrap_pi(attitude.yaw_rad - expected).abs();
            assert!(error < 1e-9, "{where_}: yaw error {error:e}");
        }
        for t in [t_out + 1.0, t_out + 600.0, t_in - 1.0] {
            let attitude = gnss_attitude(&law, &trajectory, t).unwrap();
            assert_eq!(
                attitude.mode,
                AttitudeMode::Nominal,
                "beta {beta_deg} t {t}"
            );
        }
    }
    // Beta and the shadow limit a few units in the last place from 90 deg:
    // the nominal yaws at entry and exit are -pi / 2 to the last place or one
    // unit apart, never equal with a shadow present (the shadow's in-plane
    // half span has `sin` at least `sqrt(2 / j)` with `cos(beta)` about
    // `j x 2.2e-16`), and every instant is a finite shadow or nominal yaw.
    for kb in 1..=6_u32 {
        for kl in 1..=6_u32 {
            let beta = FRAC_PI_2 - f64::from(kb) * f64::EPSILON;
            let law = GnssAttitudeLaw::GpsIIF {
                yaw_rate_rad_s: GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
                yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: FRAC_PI_2 + f64::from(kl) * f64::EPSILON,
            };
            let orbit = Circular {
                a: GPS_A,
                beta,
                u0: PI,
            };
            let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0)
                .unwrap_or_else(|e| panic!("kb {kb} kl {kl}: {e}"));
            assert!(
                matches!(attitude.mode, AttitudeMode::Shadow | AttitudeMode::Nominal),
                "kb {kb} kl {kl}"
            );
            assert!(
                (attitude.yaw_rad + FRAC_PI_2).abs() <= 2.5e-16,
                "kb {kb} kl {kl}: {}",
                attitude.yaw_rad
            );
        }
    }
}

/// A GPS noon turn with `|beta|` so small that the zone where the nominal yaw
/// rate exceeds the maximum rate is narrower than the resolution of `t` at
/// noon (one unit in the last place of 1e12 s is 1.2e-4 s): it still turns at
/// the maximum rate from noon, as at `beta = 0`. Kouba (2009): the zone is
/// `|x| < x_s` with `sin^2(x_s)` about `tan|beta| mu_rate / R`, and the
/// nominal yaw at its start, `atan2(-tan(beta), sin(x_s))`, tends to 0 with
/// beta (not to the `-pi / 2` of noon itself); so the yaw 100 s after noon is
/// `-R x 100 s` to within `sqrt(tan|beta| R / mu_rate)` (5e-7 rad at
/// beta = 1e-14), the start offset `x_s / mu_rate` times R (5e-7 rad) and R
/// times the resolution of t (2e-6 rad at 3e12 s).
#[test]
fn gps_noon_turns_narrower_than_the_time_resolution_turn_from_noon() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let rate = GPS_IIR_DEFAULT_YAW_RATE_RAD_S;
    for t_noon in [1.0e12, 3.0e12] {
        for beta in [0.0, 3.0e-16, 1.0e-15, 1.0e-14] {
            let n = Circular {
                a: GPS_A,
                beta,
                u0: 0.0,
            }
            .n();
            let orbit = Circular {
                a: GPS_A,
                beta,
                u0: -(n * t_noon),
            };
            let trajectory = orbit.trajectory();
            for dt in [100.0, 500.0] {
                let where_ = format!("t {t_noon:e} beta {beta:e} dt {dt}");
                let attitude = gnss_attitude(&law, &trajectory, t_noon + dt)
                    .unwrap_or_else(|e| panic!("{where_}: {e}"));
                assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
                let error = wrap_pi(attitude.yaw_rad + rate * dt).abs();
                assert!(error < 1e-5, "{where_}: yaw error {error:e}");
            }
        }
    }
}

/// A custom GPS IIR maximum yaw rate so low that the noon turn's zone ends
/// 9 deg before the quadrature: the turn continues past the quadrature, into
/// the midnight half orbit, until 1800 s after its zone, the yaw not having
/// met the nominal yaw. Kouba (2009): the zone is where
/// `tan(beta) cos(x) > (sin^2(x) + tan^2(beta)) R / mu_rate`, which ends at
/// `cos(x) = (-T + sqrt(T^2 + 4 k^2 (1 + T^2))) / (2 k)`, `T = tan(beta)`,
/// `k = R / mu_rate`.
#[test]
fn gps_slow_custom_turns_continue_past_the_quadrature() {
    let rate = 1.0e-5;
    let law = GnssAttitudeLaw::GpsIIR {
        yaw_rate_rad_s: rate,
    };
    let beta = deg(60.0);
    let orbit = Circular {
        a: GPS_A,
        beta,
        u0: -deg(120.0),
    };
    let n = orbit.n();
    let t_b = libm::tan(beta);
    let k = rate / n;
    let cos_x = (-t_b + libm::sqrt(t_b * t_b + 4.0 * k * k * (1.0 + t_b * t_b))) / (2.0 * k);
    let x_s = libm::acos(cos_x);
    let t_noon = orbit.time_at(0.0);
    let (t_start, t_end) = (t_noon - x_s / n, t_noon + x_s / n);
    let t_quadrature = orbit.time_at(FRAC_PI_2);
    assert!(t_end < t_quadrature && t_quadrature < t_end + 1800.0);
    let psi0 = libm::atan2(-t_b, libm::sin(x_s));
    let trajectory = orbit.trajectory();
    for t in [
        t_quadrature - 300.0,
        t_quadrature + 150.0,
        t_quadrature + 600.0,
        t_end + 1800.0 - 1.0,
    ] {
        let expected = psi0 - rate * (t - t_start);
        // The nominal yaw after noon, atan2(-T, sin(mu)) with sin(mu) =
        // -sin(u), is still below the linear yaw: not yet met.
        let nominal = libm::atan2(-t_b, -libm::sin(n * (t - t_noon)));
        assert!(
            expected - nominal > 0.1,
            "t - quadrature {}",
            t - t_quadrature
        );
        let attitude = gnss_attitude(&law, &trajectory, t).unwrap();
        assert_eq!(
            attitude.mode,
            AttitudeMode::NoonTurn,
            "t - quadrature {}",
            t - t_quadrature
        );
        let error = wrap_pi(attitude.yaw_rad - expected).abs();
        assert!(
            error < 1e-9,
            "t - quadrature {}: {error:e}",
            t - t_quadrature
        );
    }
    let after = gnss_attitude(&law, &trajectory, t_end + 1800.0 + 1.0).unwrap();
    assert_eq!(after.mode, AttitudeMode::Nominal);
}

/// A custom GLONASS-M maximum yaw rate below the orbital rate (here
/// `mu_rate / R = 1.125`) with beta drifting from 9.3 deg a quarter orbit
/// before noon to 10.7 deg a quarter orbit after (0.049 rad over the orbit):
/// the Dilssner et al. (2011) noon zone, `|x| < half_span(beta)`, 90 deg at
/// beta 10 deg, reaches past the quadrature before noon (where beta is
/// smaller and the half span near 90.8 deg) and ends near 89.2 deg after it.
/// Instants within that turn are `InvalidParameter`, on either side of the
/// quadrature; instants of the same pass outside it follow the law.
#[test]
fn glonass_m_turns_reaching_the_quadrature_are_refused_within_the_turn_only() {
    let n = (GM / (GLONASS_A * GLONASS_A * GLONASS_A)).sqrt();
    // The half span is 90 deg at beta 10 deg: g(90 deg) = 90 deg - beta.
    let k = 90.0 / 80.0;
    let rate = n / k;
    let law = GnssAttitudeLaw::GlonassM {
        yaw_rate_rad_s: rate,
        shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
    };
    let quarter = FRAC_PI_2 / n;
    let orbit = Drifting {
        a: GLONASS_A,
        beta0: deg(10.0),
        rate: deg(0.7) / quarter,
        u0: 0.0,
    };
    let trajectory = move |t: f64| Some(orbit.sample(t));
    // Dilssner half span at one beta: k y with y the positive root of
    // y = |phi(d) - phi(-d)| / 2, d = k y, on (0, pi / k].
    let half_span = |beta: f64| {
        let swing = |y: f64| {
            let d = k * y;
            0.5 * (libm::atan2(-libm::tan(beta), libm::sin(d))
                - libm::atan2(-libm::tan(beta), -libm::sin(d)))
            .abs()
        };
        let (mut lo, mut hi) = (1e-12, PI / k);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid - swing(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        k * 0.5 * (lo + hi)
    };
    let beta_at_u = |u_deg: f64| deg(10.0) + deg(0.7) * u_deg / 90.0;
    // (argument from noon, inside the noon turn)
    for (u_deg, inside) in [
        (-45.0, true),
        (60.0, true),
        (-90.5, true),
        (89.0, true),
        (-91.5, false),
        (89.6, false),
        (120.0, false),
        (-130.0, false),
    ] {
        let in_zone = deg(u_deg).abs() < half_span(beta_at_u(u_deg));
        assert_eq!(in_zone, inside, "u {u_deg}");
        let t = deg(u_deg) / n;
        let result = gnss_attitude(&law, &trajectory, t);
        if inside {
            assert!(
                matches!(
                    result,
                    Err(GnssAttitudeError::InvalidParameter {
                        field: "yaw_rate_rad_s",
                        ..
                    })
                ),
                "u {u_deg}: {result:?}"
            );
        } else {
            let attitude = result.unwrap_or_else(|e| panic!("u {u_deg}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::Nominal, "u {u_deg}");
        }
    }
}

/// A Keplerian ellipse in the plane of [`Circular`] (the Sun fixed along +X at
/// elevation `beta(t) = beta + beta_rate t` above the plane, the plane turned
/// about the noon direction as in [`Drifting`]), with the argument of perigee
/// `omega` measured from noon in the direction of motion and the mean anomaly
/// `m0` at t = 0. The velocity is the in-plane Keplerian velocity.
#[derive(Clone, Copy)]
struct Kepler {
    a: f64,
    e: f64,
    beta: f64,
    beta_rate: f64,
    omega: f64,
    m0: f64,
}

impl Kepler {
    fn n(&self) -> f64 {
        (GM / (self.a * self.a * self.a)).sqrt()
    }

    fn period(&self) -> f64 {
        TAU / self.n()
    }

    /// Eccentric anomaly from the mean anomaly, by Newton's method from
    /// `M + e sin M` until the step no longer changes it.
    fn eccentric_anomaly(&self, m: f64) -> f64 {
        let mut ea = m + self.e * libm::sin(m);
        for _ in 0..100 {
            let step = (ea - self.e * libm::sin(ea) - m) / (1.0 - self.e * libm::cos(ea));
            ea -= step;
            if step.abs() <= 1e-16 * ea.abs().max(1.0) {
                break;
            }
        }
        ea
    }

    /// True anomaly at `t`.
    fn true_anomaly(&self, t: f64) -> f64 {
        let ea = self.eccentric_anomaly(self.m0 + self.n() * t);
        2.0 * libm::atan2(
            (1.0 + self.e).sqrt() * libm::sin(0.5 * ea),
            (1.0 - self.e).sqrt() * libm::cos(0.5 * ea),
        )
    }

    /// Argument from noon at `t`, in (-pi, pi].
    fn argument_from_noon(&self, t: f64) -> f64 {
        wrap_pi(self.omega + self.true_anomaly(t))
    }

    fn sample(&self, t: f64) -> AttitudeSample {
        let (p, q, _) = Circular {
            a: self.a,
            beta: self.beta + self.beta_rate * t,
            u0: 0.0,
        }
        .frame();
        let nu = self.true_anomaly(t);
        let u = self.omega + nu;
        let semi_latus = self.a * (1.0 - self.e * self.e);
        let r = semi_latus / (1.0 + self.e * libm::cos(nu));
        let h = (GM / semi_latus).sqrt();
        let (v_r, v_t) = (
            h * self.e * libm::sin(nu),
            h * (1.0 + self.e * libm::cos(nu)),
        );
        let (su, cu) = (libm::sin(u), libm::cos(u));
        let e_r: [f64; 3] = std::array::from_fn(|k| cu * p[k] + su * q[k]);
        let e_t: [f64; 3] = std::array::from_fn(|k| -su * p[k] + cu * q[k]);
        AttitudeSample {
            position_m: e_r.map(|x| r * x),
            velocity_m_s: std::array::from_fn(|k| v_r * e_r[k] + v_t * e_t[k]),
            sun_position_m: [AU, 0.0, 0.0],
        }
    }

    fn trajectory(self) -> impl Fn(f64) -> Option<AttitudeSample> {
        move |t| Some(self.sample(t))
    }

    /// The first instant at or after `t_after` at which the argument from
    /// noon is `u`.
    fn time_at(&self, u: f64, t_after: f64) -> f64 {
        let nu = u - self.omega;
        let ea = 2.0
            * libm::atan2(
                (1.0 - self.e).sqrt() * libm::sin(0.5 * nu),
                (1.0 + self.e).sqrt() * libm::cos(0.5 * nu),
            );
        let m = ea - self.e * libm::sin(ea);
        let t = (m - self.m0) / self.n();
        let period = self.period();
        t + period * libm::ceil((t_after - t) / period)
    }
}

/// A rotation matrix about the unit vector along `axis` by `angle` (Rodrigues).
fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let k = unit3(axis).unwrap();
    let (s, c) = (libm::sin(angle), libm::cos(angle));
    let v = 1.0 - c;
    [
        [
            c + k[0] * k[0] * v,
            k[0] * k[1] * v - k[2] * s,
            k[0] * k[2] * v + k[1] * s,
        ],
        [
            k[1] * k[0] * v + k[2] * s,
            c + k[1] * k[1] * v,
            k[1] * k[2] * v - k[0] * s,
        ],
        [
            k[2] * k[0] * v - k[1] * s,
            k[2] * k[1] * v + k[0] * s,
            c + k[2] * k[2] * v,
        ],
    ]
}

fn rotate(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// The same geometry turned rigidly into a general orientation, so every
/// component of every vector is non-zero and rounded: beta, the orbit angle
/// and the yaw are unchanged up to rounding.
fn tilted(
    trajectory: impl Fn(f64) -> Option<AttitudeSample>,
) -> impl Fn(f64) -> Option<AttitudeSample> {
    let m = rotation([0.3, -0.5, 0.8], 1.1);
    move |t| {
        trajectory(t).map(|s| AttitudeSample {
            position_m: rotate(&m, s.position_m),
            velocity_m_s: rotate(&m, s.velocity_m_s),
            sun_position_m: rotate(&m, s.sun_position_m),
        })
    }
}

fn assert_valid(attitude: &GnssAttitude, where_: &str) {
    assert_finite(attitude);
    assert!(
        -PI < attitude.yaw_rad && attitude.yaw_rad <= PI,
        "{where_}: yaw {}",
        attitude.yaw_rad
    );
    for v in [attitude.body_x, attitude.body_y, attitude.body_z] {
        assert!((norm3(v) - 1.0).abs() < 1e-13, "{where_}: {attitude:?}");
    }
    assert!(
        dot3(attitude.body_x, attitude.body_y).abs() < 1e-13,
        "{where_}"
    );
}

fn yaw_rate_of(law: &GnssAttitudeLaw) -> f64 {
    match *law {
        GnssAttitudeLaw::GpsIIA { yaw_rate_rad_s, .. }
        | GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s }
        | GnssAttitudeLaw::GpsIIF { yaw_rate_rad_s, .. }
        | GnssAttitudeLaw::GlonassM { yaw_rate_rad_s, .. } => yaw_rate_rad_s,
        _ => unreachable!(),
    }
}

/// Kouba (2009): a GPS noon turn (and the IIR midnight turn) exists while
/// `|beta| < beta_0 = atan(mu_rate / R)`; it starts at the nominal yaw where
/// the nominal yaw rate first exceeds R and turns at R the way the nominal yaw
/// moves. Inside the zone the nominal yaw moves faster than R, so at the
/// zone's end it is at or ahead of the linear yaw by the integral of the
/// excess rate, and the turn ends when the linear yaw catches it. Just inside
/// `beta_0` the zone lasts milliseconds and the excess is a small fraction of
/// R: the yaw stays within 1e-6 rad of nominal at every instant, and the turn
/// is over within a second of the centre.
#[test]
fn gps_turns_just_inside_the_nominal_rate_limit_stay_on_the_nominal_yaw() {
    let laws = [
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap(),
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIF, None).unwrap(),
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(1)).unwrap(),
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(11)).unwrap(),
    ];
    for law in laws {
        let rate = yaw_rate_of(&law);
        let iir = matches!(law, GnssAttitudeLaw::GpsIIR { .. });
        let n = Circular {
            a: GPS_A,
            beta: 0.0,
            u0: 0.0,
        }
        .n();
        let beta0 = libm::atan(n / rate);
        let centres: &[f64] = if iir { &[0.0, PI] } else { &[0.0] };
        for &centre in centres {
            for sign in [1.0, -1.0] {
                for eps in [1e-13, 1e-12, 1e-11, 1e-10, 1e-9] {
                    let orbit = Circular {
                        a: GPS_A,
                        beta: sign * (beta0 - eps),
                        u0: centre - deg(5.0),
                    };
                    let trajectory = orbit.trajectory();
                    let t_centre = orbit.time_at(centre);
                    let mut k = -300.0;
                    while k <= 2400.0 {
                        let t = t_centre + k;
                        let where_ = format!(
                            "{law:?} centre {centre} beta beta_0 - {eps:e} (sign {sign}) t - centre {k}"
                        );
                        let attitude = gnss_attitude(&law, &trajectory, t)
                            .unwrap_or_else(|e| panic!("{where_}: {e}"));
                        let off = wrap_pi(attitude.yaw_rad - attitude.nominal_yaw_rad).abs();
                        assert!(
                            off < 1e-6,
                            "{where_}: {:?} {off:e} from nominal",
                            attitude.mode
                        );
                        if k.abs() >= 1.0 {
                            assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                        }
                        k += 7.0;
                    }
                }
            }
        }
    }
}

/// Kouba (2009): with yaw bias `b` a GPS noon turn runs at `-sign(beta) R`
/// except for beta between 0 and `b`, where it runs the other way. Both
/// limits `beta -> 0+` and `beta -> 0-` give `sign(b) R`, so a noon turn at
/// beta = 0 turns at `sign(b) R` (GPS IIA `+R`, GPS IIF `-R`). Checked by the
/// yaw change between two instants inside the turn, and at beta = 0 by the
/// yaw itself: the turn starts at noon at yaw 0.
#[test]
fn gps_noon_turn_direction_through_beta_zero_follows_the_yaw_bias() {
    let iia = GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(1)).unwrap();
    let iif = GnssAttitudeLaw::for_block(GnssBlock::GpsIIF, None).unwrap();
    let iir = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    // A GPS IIA with the bias of the other sign (PRN 23 in `eclips.f` until
    // 2013): the reversal moves to beta in [-0.5 deg, 0).
    let iia_negative = GnssAttitudeLaw::GpsIIA {
        yaw_rate_rad_s: gps_iia_default_yaw_rate_rad_s(23).unwrap(),
        yaw_bias_rad: -GPS_IIA_DEFAULT_YAW_BIAS_RAD,
        shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
    };
    let cases: [(GnssAttitudeLaw, &[(f64, f64)]); 4] = [
        (
            iia,
            &[
                (-1e-9, 1.0),
                (0.0, 1.0),
                (1e-9, 1.0),
                (deg(0.25), 1.0),
                (deg(0.45), 1.0),
                (deg(0.55), -1.0),
                (deg(-0.3), 1.0),
            ],
        ),
        (
            iif,
            &[
                (1e-9, -1.0),
                (0.0, -1.0),
                (-1e-9, -1.0),
                (deg(-0.3), -1.0),
                (deg(-0.65), -1.0),
                (deg(-0.75), 1.0),
                (deg(0.3), -1.0),
            ],
        ),
        (iir, &[(1e-9, -1.0), (-1e-9, 1.0), (deg(0.3), -1.0)]),
        (
            iia_negative,
            &[
                (deg(-0.3), -1.0),
                (-1e-9, -1.0),
                (0.0, -1.0),
                (1e-9, -1.0),
                (deg(0.3), -1.0),
                (deg(-0.6), 1.0),
            ],
        ),
    ];
    for (law, betas) in cases {
        let rate = yaw_rate_of(&law);
        for &(beta, direction) in betas {
            let orbit = Circular {
                a: GPS_A,
                beta,
                u0: -deg(10.0),
            };
            let trajectory = orbit.trajectory();
            let t_noon = orbit.time_at(0.0);
            let where_ = format!("{law:?} beta {beta:e}");
            let y1 = gnss_attitude(&law, &trajectory, t_noon + 100.0).unwrap();
            let y2 = gnss_attitude(&law, &trajectory, t_noon + 200.0).unwrap();
            assert_eq!(y1.mode, AttitudeMode::NoonTurn, "{where_}");
            assert_eq!(y2.mode, AttitudeMode::NoonTurn, "{where_}");
            let change = wrap_pi(y2.yaw_rad - y1.yaw_rad);
            assert!(
                (change - direction * rate * 100.0).abs() < 1e-9,
                "{where_}: yaw change {change} vs {}",
                direction * rate * 100.0
            );
            if beta == 0.0 {
                let y = gnss_attitude(&law, &trajectory, t_noon + 500.0).unwrap();
                let expected = direction * rate * 500.0;
                assert!(
                    wrap_pi(y.yaw_rad - expected).abs() < 1e-9,
                    "{where_}: yaw {} vs {expected}",
                    y.yaw_rad
                );
            }
        }
    }
}

/// Every finite positive maximum yaw rate is valid. Very fast rates make the
/// GPS noon and midnight turns shorter than the time resolution, so the yaw
/// is nominal around them; very slow rates make the zone span nearly half an
/// orbit (Kouba 2009: `T cos(x) = (sin^2(x) + T^2) R / mu_rate` gives `x`
/// near `pi / 2`), so the turn starts near the quadrature at the nominal yaw
/// there, `atan2(-tan(beta), 1)`, and barely moves. GPS IIA rates above 1e5
/// times the orbital rate are outside the validated domain; below that the
/// shadow crossing's yaw is the linear yaw reduced exactly modulo 2 pi.
#[test]
fn extreme_yaw_rates_give_the_limit_attitude_or_a_typed_error() {
    let at_zero = Circular {
        a: GPS_A,
        beta: 0.0,
        u0: -deg(30.0),
    };
    let t_noon = at_zero.time_at(0.0);
    for rate in [1e306, 1e300, 1e10] {
        let law = GnssAttitudeLaw::GpsIIR {
            yaw_rate_rad_s: rate,
        };
        for dt in [-100.0, -1.0, 0.0, 1.0, 100.0] {
            let where_ = format!("IIR rate {rate:e} t - noon {dt}");
            let attitude = gnss_attitude(&law, &at_zero.trajectory(), t_noon + dt)
                .unwrap_or_else(|e| panic!("{where_}: {e}"));
            assert_valid(&attitude, &where_);
            if dt != 0.0 {
                assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
            }
        }
    }
    let ten = Circular {
        a: GPS_A,
        beta: deg(10.0),
        u0: -deg(30.0),
    };
    let t_noon = ten.time_at(0.0);
    for rate in [1e-160, 1e-315, f64::MIN_POSITIVE, 5e-324] {
        let law = GnssAttitudeLaw::GpsIIR {
            yaw_rate_rad_s: rate,
        };
        let where_ = format!("IIR rate {rate:e}");
        let attitude = gnss_attitude(&law, &ten.trajectory(), t_noon)
            .unwrap_or_else(|e| panic!("{where_}: {e}"));
        assert_valid(&attitude, &where_);
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
        let expected = libm::atan2(-libm::tan(deg(10.0)), 1.0);
        assert!(
            (attitude.yaw_rad - expected).abs() < 1e-12,
            "{where_}: yaw {} vs {expected}",
            attitude.yaw_rad
        );
    }
    let shadow = Circular {
        a: GPS_A,
        beta: deg(1.0),
        u0: PI - deg(30.0),
    };
    let t_midnight = shadow.time_at(PI);
    let iia = |rate: f64| GnssAttitudeLaw::GpsIIA {
        yaw_rate_rad_s: rate,
        yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
        shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
    };
    for rate in [1e306, 1e10] {
        let result = gnss_attitude(&iia(rate), &shadow.trajectory(), t_midnight);
        assert!(outside_domain(&result), "IIA {rate:e}: {result:?}");
    }
    // 9e4 times the orbital rate (13 rad/s) for some 1500 s into the shadow:
    // the linear yaw is of order 2e4 rad, whose rounding alone is 4e-12 rad.
    // Two instants 2^-30 s apart differ by R times their exact difference
    // modulo 2 pi.
    let rate = 9e4 * shadow.n();
    let t1 = t_midnight;
    let t2 = t1 + libm::ldexp(1.0, -30);
    let d = t2 - t1;
    let y1 = gnss_attitude(&iia(rate), &shadow.trajectory(), t1).unwrap();
    let y2 = gnss_attitude(&iia(rate), &shadow.trajectory(), t2).unwrap();
    assert_eq!(y1.mode, AttitudeMode::Shadow);
    assert_valid(&y1, "IIA 13 rad/s t1");
    assert_valid(&y2, "IIA 13 rad/s t2");
    let error = wrap_pi(y2.yaw_rad - y1.yaw_rad - rate * d);
    assert!(error.abs() < 1e-12, "yaw difference off by {error:e}");
    for rate in [1e306, 1e-300] {
        let law = GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s: rate,
            shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
        };
        for (orbit, t) in [(at_zero, t_noon), (shadow, t_midnight)] {
            match gnss_attitude(&law, &orbit.trajectory(), t) {
                Ok(attitude) => assert_valid(&attitude, &format!("GLONASS-M {rate:e}")),
                Err(e) => assert!(
                    matches!(
                        e,
                        GnssAttitudeError::InvalidParameter {
                            field: "yaw_rate_rad_s",
                            ..
                        }
                    ),
                    "GLONASS-M {rate:e}: {e}"
                ),
            }
        }
    }
}

/// The cosine turn's direction is the sign of the nominal yaw at its start
/// (Galileo satellite metadata 3.1.2), which at beta = 0 is the sign of
/// rounding. On an orbit in a general orientation with the Sun in its plane,
/// every instant of one turn must use the same start, and so the same
/// direction: `|psi| = (pi / 2) (1 - cos(2 pi t / T))` throughout, of one sign.
#[test]
fn cosine_turn_direction_is_one_per_turn_on_tilted_orbits_at_beta_zero() {
    for (block, a, start_deg) in [
        (GnssBlock::GalileoFoc, GALILEO_A, 10.0),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            MEO_A,
            6.0,
        ),
        (GnssBlock::Beidou3Igso, 42_164_200.0, 6.0),
    ] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        let GnssAttitudeLaw::CosineTurn(turn) = law else {
            unreachable!()
        };
        for centre in [0.0, PI] {
            let orbit = Circular {
                a,
                beta: 0.0,
                u0: centre - deg(start_deg + 3.0),
            };
            let trajectory = tilted(orbit.trajectory());
            let t_start = orbit.time_at(centre - deg(start_deg));
            let mut signs = std::collections::BTreeSet::new();
            let mut k = 1.0;
            while k < 0.5 * turn.period_s {
                let t = t_start + k;
                let where_ = format!("{block} centre {centre} t - start {k}");
                let attitude =
                    gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
                assert_ne!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                // The nominal yaw at the start is 0 before noon and pi before
                // midnight (of either sign at beta = 0).
                let c = libm::cos(TAU * k / turn.period_s);
                let expected = if centre == 0.0 {
                    FRAC_PI_2 * (1.0 - c)
                } else {
                    FRAC_PI_2 * (1.0 + c)
                };
                assert!(
                    (attitude.yaw_rad.abs() - expected).abs() < 1e-9,
                    "{where_}: yaw {} vs +-{expected}",
                    attitude.yaw_rad
                );
                if attitude.yaw_rad.abs() > 1e-6 {
                    signs.insert(attitude.yaw_rad > 0.0);
                }
                k += 1.0;
            }
            assert_eq!(
                signs.len(),
                1,
                "{block} centre {centre}: both directions in one turn"
            );
        }
    }
}

/// Kouba (2009): the in-plane half span `x_s` of the zone where the nominal
/// yaw rate exceeds `R`, the root in `[0, pi / 2]` of
/// `T cos(x) - (sin^2(x) + T^2) R / mu_rate` (positive at 0 when the zone
/// exists, negative at `pi / 2`), by bisection.
fn kouba_half_span(beta: f64, mu_rate: f64, rate: f64) -> f64 {
    let (t, k) = (libm::tan(beta).abs(), rate / mu_rate);
    let f = |x: f64| t * libm::cos(x) - (libm::sin(x).powi(2) + t * t) * k;
    let (mut lo, mut hi) = (0.0_f64, FRAC_PI_2);
    assert!(f(lo) > 0.0 && f(hi) < 0.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The first instant at or after `t_from` at which `angle` falls to `start`
/// or below, by a 1 s scan and bisection to 1e-10 s.
fn first_fall(angle: impl Fn(f64) -> f64, start: f64, t_from: f64, t_to: f64) -> Option<f64> {
    let mut t = t_from;
    if angle(t) <= start {
        return Some(t);
    }
    while t < t_to {
        let next = t + 1.0;
        if angle(next) <= start {
            let (mut lo, mut hi) = (t, next);
            while hi - lo > 1e-10 {
                let mid = 0.5 * lo + 0.5 * hi;
                if angle(mid) > start {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some(hi);
        }
        t = next;
    }
    None
}

/// The published cosine turns (Galileo satellite metadata 3.1.2, Wang et al.
/// 2018) on eccentric orbits: the turn starts at the first instant on the
/// approach to noon or midnight at which the start angle (colinearity
/// `acos(cos(beta) cos(x))` or the in-plane angle `x`) falls to its value, here
/// from the Keplerian argument from noon, and lasts half the period. Orbits of
/// eccentricity 0.02 (GNSS) and 0.3, 0.5 and 0.7, with perigee at noon,
/// quadrature, midnight and the other quadrature.
#[test]
fn cosine_turns_follow_the_published_formula_on_eccentric_orbits() {
    for (block, a) in [
        (GnssBlock::GalileoFoc, GALILEO_A),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            MEO_A,
        ),
        (GnssBlock::Beidou3Igso, 42_164_200.0),
    ] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        let GnssAttitudeLaw::CosineTurn(turn) = law else {
            unreachable!()
        };
        let colinearity = turn.start_angle == CosineTurnStartAngle::Colinearity;
        for e in [0.02, 0.3, 0.5, 0.7] {
            for omega_deg in [0.0, 90.0, 180.0, 270.0] {
                for beta_deg in [1.0, -2.5] {
                    let orbit = Kepler {
                        a,
                        e,
                        beta: deg(beta_deg),
                        beta_rate: 0.0,
                        omega: deg(omega_deg),
                        m0: 0.3,
                    };
                    let trajectory = orbit.trajectory();
                    for centre in [0.0, PI] {
                        let t_centre = orbit.time_at(centre, 0.0);
                        let angle = |t: f64| {
                            let u = orbit.argument_from_noon(t);
                            let x = if centre == 0.0 { -u } else { wrap_pi(PI - u) };
                            if colinearity {
                                libm::acos(libm::cos(deg(beta_deg)) * libm::cos(x)).copysign(x)
                            } else {
                                x
                            }
                        };
                        let t_start = first_fall(
                            angle,
                            turn.start_angle_rad,
                            orbit.time_at(centre - deg(60.0), t_centre - orbit.period()),
                            t_centre,
                        )
                        .expect("a start");
                        let u_start = orbit.argument_from_noon(t_start);
                        let psi0 = libm::atan2(-libm::tan(deg(beta_deg)), libm::sin(u_start - PI));
                        let quarter = FRAC_PI_2.copysign(psi0);
                        let half = 0.5 * turn.period_s;
                        for f in [0.001, 0.1, 0.5, 0.9, 0.999] {
                            let t = t_start + f * half;
                            let where_ = format!(
                                "{block} e {e} omega {omega_deg} beta {beta_deg} centre {centre} f {f}"
                            );
                            let attitude = gnss_attitude(&law, &trajectory, t)
                                .unwrap_or_else(|e| panic!("{where_}: {e}"));
                            assert_ne!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                            let expected = quarter
                                + (psi0 - quarter) * libm::cos(TAU * (t - t_start) / turn.period_s);
                            assert!(
                                wrap_pi(attitude.yaw_rad - expected).abs() < 1e-8,
                                "{where_}: yaw {} vs {expected}",
                                attitude.yaw_rad
                            );
                        }
                        for t in [t_start - 1.0, t_start + half + 1.0] {
                            let attitude = gnss_attitude(&law, &trajectory, t).unwrap();
                            assert_eq!(
                                attitude.mode,
                                AttitudeMode::Nominal,
                                "{block} e {e} omega {omega_deg} beta {beta_deg} centre {centre} t - start {}",
                                t - t_start
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Galileo satellite metadata 3.1.2: the turn starts when the colinearity
/// angle falls below the start value with `|beta|` below the limit, and then
/// lasts half the period whatever the angle does. Here the start falls
/// 0.13 deg of orbit before noon at beta 9.9992 deg, below a limit of
/// 9.9995 deg, and beta rises at 1.05e-6 rad/s (0.049 rad over the orbit)
/// past 10 deg before noon, so the colinearity angle, never below `|beta|`,
/// is back above the start value at noon: the turn that started at its first
/// fall, found here on a scan of `acos(cos(beta) cos(x))` of the closed form,
/// is still running.
#[test]
fn cosine_turn_continues_when_the_start_angle_rises_again() {
    let a = MEO_A;
    let n = (GM / (a * a * a)).sqrt();
    let start_angle = deg(10.0);
    let beta_start = deg(9.9992);
    let u_start = -libm::acos(libm::cos(start_angle) / libm::cos(beta_start));
    let beta_rate = 1.05e-6;
    let beta = move |t: f64| beta_start + beta_rate * t;
    let rising = move |t: f64| {
        let u = u_start + n * t;
        let b = beta(t);
        let (su, cu) = (libm::sin(u), libm::cos(u));
        Some(AttitudeSample {
            position_m: [a * cu, a * su, 0.0],
            velocity_m_s: [-a * n * su, a * n * cu, 0.0],
            sun_position_m: [AU * libm::cos(b), 0.0, AU * libm::sin(b)],
        })
    };
    let turn_period = 5656.0;
    let law = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        beta_limit_rad: deg(9.9995),
        beta_limit_inclusive: false,
        start_angle: CosineTurnStartAngle::Colinearity,
        start_angle_rad: start_angle,
        period_s: turn_period,
    });
    // The colinearity angle of the closed form, signed positive before noon.
    let colinearity = |t: f64| {
        let x = -(u_start + n * t);
        libm::acos(libm::cos(beta(t)) * libm::cos(x)).copysign(x)
    };
    let t_start = first_fall(colinearity, start_angle, -2000.0, 1000.0).expect("a fall");
    assert!(beta(t_start) < deg(9.9995), "{}", beta(t_start));
    assert!(beta(-u_start / n) > start_angle);
    let psi0 = libm::atan2(
        -libm::tan(beta(t_start)),
        libm::sin(u_start + n * t_start - PI),
    );
    let quarter = FRAC_PI_2.copysign(psi0);
    for t in [t_start + 1.0, 50.0, 100.0, 150.0, 1000.0, t_start + 2800.0] {
        let attitude = gnss_attitude(&law, &rising, t).unwrap_or_else(|e| panic!("t {t}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "t {t}");
        let expected = quarter + (psi0 - quarter) * libm::cos(TAU * (t - t_start) / turn_period);
        assert!(
            wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
            "t {t}: {} vs {expected}",
            attitude.yaw_rad
        );
    }
    let just_before_noon = 0.9 * (-u_start / n);
    assert!(
        colinearity(just_before_noon) > start_angle,
        "{}",
        colinearity(just_before_noon)
    );
    let before = gnss_attitude(&law, &rising, t_start - 1.0).unwrap();
    assert_eq!(before.mode, AttitudeMode::Nominal);
}

/// BeiDou-2 IGSO and MEO switch at the orbit quadrature, orbit angle 90 deg
/// from midnight (Orekit `BeidouMeo`). On an eccentric orbit the quadrature
/// instant is where the satellite actually is at that orbit angle; here
/// `|beta|` decreases through 4.1 deg so slowly that only the value at the
/// last quadrature decides, set a hair below and a hair above 4.1 deg there.
/// The quadrature is found from the Keplerian argument from noon.
#[test]
fn beidou2_switch_uses_the_quadrature_instant_on_eccentric_orbits() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::Beidou2Igso, None).unwrap();
    let rate = -1e-11;
    for e in [0.01, 0.1] {
        for omega_deg in [0.0, 120.0, 250.0] {
            // Evaluated at orbit angles from midnight after the quadrature.
            for mu_deg in [100.0, 179.0, 300.0, -0.3, 0.3, 80.0] {
                for (offset, expected) in [
                    (-1e-9, AttitudeMode::OrbitNormal),
                    (1e-9, AttitudeMode::Nominal),
                ] {
                    let probe = Kepler {
                        a: 42_164_200.0,
                        e,
                        beta: 0.0,
                        beta_rate: 0.0,
                        omega: deg(omega_deg),
                        m0: 0.0,
                    };
                    let t_eval = probe.time_at(deg(mu_deg) - PI, 2.0 * probe.period());
                    let t_quadrature = probe.time_at(-FRAC_PI_2, t_eval - probe.period());
                    assert!(t_quadrature <= t_eval);
                    let beta_q = BEIDOU2_YS_TO_ON_BETA_RAD + offset;
                    let orbit = Kepler {
                        beta: beta_q - rate * t_quadrature,
                        beta_rate: rate,
                        ..probe
                    };
                    let attitude = gnss_attitude(&law, &orbit.trajectory(), t_eval)
                        .unwrap_or_else(|e| panic!("{e}"));
                    assert_eq!(
                        attitude.mode, expected,
                        "e {e} omega {omega_deg} orbit angle {mu_deg} beta at the quadrature 4.1 deg {offset:+e}"
                    );
                }
            }
        }
    }
}

/// A cosine turn near the largest time: a circular orbit of radius 1e150 m
/// whose turn window (1.4 rad before noon) is at 4e307 s and noon at
/// 1.02e308 s, the turn period 1.79e308 s, evaluated at 1e308 and
/// 1.01e308 s.
#[test]
fn cosine_turn_is_found_near_the_largest_time() {
    let period = 1.79e308;
    let start = 1.4;
    let law = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        beta_limit_rad: 0.5,
        beta_limit_inclusive: true,
        start_angle: CosineTurnStartAngle::OrbitAngle,
        start_angle_rad: start,
        period_s: period,
    });
    let (t_start, t_noon) = (4.0e307, 1.02e308);
    let n = start / (t_noon - t_start);
    assert!(n.is_normal());
    let trajectory = sun_plane_circle(1.0e150, n, t_noon);
    for t_s in [1.0e308, 1.01e308] {
        let elapsed = t_s - t_start;
        assert!(elapsed <= 0.5 * period);
        let attitude =
            gnss_attitude(&law, &trajectory, t_s).unwrap_or_else(|e| panic!("t {t_s:e}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "t {t_s:e}");
        let expected = FRAC_PI_2 * (1.0 - libm::cos(elapsed / period * TAU));
        assert!(
            (attitude.yaw_rad.abs() - expected).abs() < 1e-9,
            "t {t_s:e}: {} vs +-{expected}",
            attitude.yaw_rad
        );
    }
}

/// The nominal X axis at the end of a GPS turn's zone is the direction of the
/// Sun's component across the radial direction. With the satellite within
/// 1e-200 rad of the Sun line that component is tiny but its direction is
/// defined; its squared norm underflows, which must not make the geometry
/// degenerate. Here beta = 0, the Sun lies in the orbit plane at azimuth
/// `eps` (so the satellite is on the Sun line at u = eps, which no instant
/// hits exactly), the zone is the noon instant and the turn runs at `-R` from
/// yaw 0.
#[test]
fn gps_turn_end_tolerates_a_sun_component_whose_square_underflows() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let rate = GPS_IIR_DEFAULT_YAW_RATE_RAD_S;
    let orbit = Circular {
        a: GPS_A,
        beta: 0.0,
        u0: 0.0,
    };
    let n = orbit.n();
    for eps in [1e-200, 3e-201, 7e-199, 1.3e-199, 2.9e-190, 5e-185, 1.1e-196] {
        let sun = [AU * libm::cos(eps), AU * libm::sin(eps), 0.0];
        let trajectory = move |t: f64| {
            let mut s = orbit.sample(t);
            s.sun_position_m = sun;
            Some(s)
        };
        let t_noon = eps / n;
        for dt in [100.0, 600.0] {
            let attitude = gnss_attitude(&law, &trajectory, t_noon + dt)
                .unwrap_or_else(|e| panic!("eps {eps:e} dt {dt}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "eps {eps:e} dt {dt}");
            assert!(
                wrap_pi(attitude.yaw_rad + rate * dt).abs() < 1e-9,
                "eps {eps:e} dt {dt}: yaw {}",
                attitude.yaw_rad
            );
        }
    }
}

/// A GPS turn ends where the linear yaw meets the nominal yaw (Kouba 2009).
/// At maximum yaw rates of 5000 and 1e5 rad/s the yaw moves by more than pi
/// in a millisecond, and the turn of a beta = 1e-9 noon pass meets the nominal
/// yaw within a millisecond of its zone: here the zone and the meeting from
/// the closed form (Kouba's zone by [`kouba_half_span`], the gap between the
/// nominal yaw and the linear yaw), with
/// the turn at the linear yaw before the meeting and nominal from 1e-6 s
/// after it.
#[test]
fn gps_turns_at_very_fast_custom_rates_end_at_once() {
    for rate in [5000.0, 1.0e5] {
        let law = GnssAttitudeLaw::GpsIIR {
            yaw_rate_rad_s: rate,
        };
        let beta = 1e-9;
        let orbit = Circular {
            a: GPS_A,
            beta,
            u0: -deg(10.0),
        };
        let n = orbit.n();
        let t_noon = orbit.time_at(0.0);
        let t_b = libm::tan(beta);
        let x_s = kouba_half_span(beta, n, rate);
        let (t_start, t_end) = (t_noon - x_s / n, t_noon + x_s / n);
        let nominal = |t: f64| libm::atan2(-t_b, libm::sin(n * (t - t_noon) + PI));
        let psi0 = nominal(t_start);
        let linear = |t: f64| psi0 - rate * (t - t_start);
        let gap0 = nominal(t_end) - linear(t_end);
        // The turn runs at -R: the gap is linear yaw minus nominal yaw.
        let gap0 = kouba_trailing_gap(-gap0);
        let gap = |t: f64| gap0 - (nominal(t) - nominal(t_end)) - rate * (t - t_end);
        let (mut lo, mut hi) = (t_end, t_end + 1e-2);
        assert!(gap(hi) < 0.0);
        while hi - lo > 1e-12 {
            let mid = 0.5 * (lo + hi);
            if gap(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let meeting = hi;
        assert!(meeting - t_end < 1e-3, "{}", meeting - t_end);
        for dt in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 10.0, 1000.0] {
            let t = meeting + dt;
            let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
            assert_eq!(
                attitude.mode,
                AttitudeMode::Nominal,
                "rate {rate} t - meeting {dt}: yaw {} nominal {}",
                attitude.yaw_rad,
                attitude.nominal_yaw_rad
            );
        }
        let inside = t_end + 0.5 * (meeting - t_end);
        let attitude = gnss_attitude(&law, &orbit.trajectory(), inside).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "rate {rate}");
        assert!(
            wrap_pi(attitude.yaw_rad - linear(inside)).abs() < 1e-6,
            "rate {rate}: {} vs {}",
            attitude.yaw_rad,
            linear(inside)
        );
    }
}

/// The meeting of a GPS noon turn with the nominal yaw on a fast custom orbit
/// (radius 4500 km, period 3000 s) where the 1800 s after the zone reach past
/// the opposite midnight, around which the nominal yaw moves faster than
/// these turns, and on to the next noon's zone: GPS IIF and IIA at beta above
/// the shadow limit or with a shadow limit just below 180 deg (no shadow turn
/// takes over at midnight) and maximum yaw rates of 0.07 to 2.5 times the
/// orbital rate. Against the meeting found here on a 0.01 s scan of the gap
/// between the Kouba nominal yaw `atan2(-tan(beta), sin(mu))` (continuous for
/// beta of one sign) and the linear yaw.
#[test]
fn gps_turn_meeting_is_found_across_the_opposite_centre_on_fast_orbits() {
    let a = 4_500_000.0;
    let mut across_midnight = 0;
    // A shadow limit just below 180 deg leaves no shadow at these betas, so no
    // shadow turn takes over at midnight where, at low beta, the nominal yaw
    // swings by nearly pi within seconds.
    let no_shadow = PI - 1e-6;
    let mut cases = vec![
        (GnssBlock::GpsIIF, 20.0, GPS_SHADOW_LIMIT_RAD),
        (GnssBlock::GpsIIF, -30.0, GPS_SHADOW_LIMIT_RAD),
        (GnssBlock::GpsIIA, 15.0, GPS_SHADOW_LIMIT_RAD),
        (GnssBlock::GpsIIA, -40.0, GPS_SHADOW_LIMIT_RAD),
        (GnssBlock::GpsIIF, 60.0, GPS_SHADOW_LIMIT_RAD),
        (GnssBlock::GpsIIA, -60.0, GPS_SHADOW_LIMIT_RAD),
    ];
    for beta_deg in [0.3, 0.8, 1.2, 2.0, 3.0, 5.0] {
        cases.push((GnssBlock::GpsIIF, beta_deg, no_shadow));
        cases.push((GnssBlock::GpsIIA, -beta_deg, no_shadow));
    }
    for (block, beta_deg, limit) in cases {
        for rate_factor in [0.07, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.5, 2.5] {
            let beta = deg(beta_deg);
            let probe = Circular {
                a,
                beta,
                u0: -deg(60.0),
            };
            let n = probe.n();
            let rate = n * rate_factor;
            let t_b = libm::tan(beta);
            if beta.abs() >= libm::atan(n / rate) {
                continue;
            }
            let law = match block {
                GnssBlock::GpsIIF => GnssAttitudeLaw::GpsIIF {
                    yaw_rate_rad_s: rate,
                    yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
                    shadow_limit_rad: limit,
                },
                _ => GnssAttitudeLaw::GpsIIA {
                    yaw_rate_rad_s: rate,
                    yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                    shadow_limit_rad: limit,
                },
            };
            let trajectory = probe.trajectory();
            let t_noon = probe.time_at(0.0);
            let x_s = kouba_half_span(beta, n, rate);
            let (t_start, t_end) = (t_noon - x_s / n, t_noon + x_s / n);
            let nominal = |t: f64| libm::atan2(-t_b, libm::sin(n * (t - t_noon) + PI));
            let psi0 = nominal(t_start);
            let phi_dot = -rate.copysign(beta);
            let linear = |t: f64| psi0 + phi_dot * (t - t_start);
            let direction = phi_dot.signum();
            let gap0 = direction * (nominal(t_end) - linear(t_end));
            let gap0 = kouba_trailing_gap(gap0);
            let gap =
                |t: f64| gap0 + direction * (nominal(t) - nominal(t_end)) - rate * (t - t_end);
            let mut reach = f64::INFINITY;
            let mut t = t_end;
            while t < t_end + 1800.0 {
                if gap(t + 0.01) <= 0.0 {
                    let (mut lo, mut hi) = (t, t + 0.01);
                    while hi - lo > 1e-9 {
                        let mid = 0.5 * (lo + hi);
                        if gap(mid) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    reach = hi;
                    break;
                }
                t += 0.01;
            }
            let end = reach.min(t_end + 1800.0);
            if end > probe.time_at(PI) {
                across_midnight += 1;
            }
            // Up to the next noon turn's zone: a grid every 61 s, skipping
            // the grid points within 1e-7 s of the turn end (where the two
            // evaluations may differ by their rounding), and probes 1e-5 s
            // either side of the end, all checked.
            let next_start = t_start + TAU / n;
            let in_turn = |t: f64| {
                let limit = t_end + 1800.0;
                t < end || (end == limit && t <= limit)
            };
            let mut checks = vec![end - 1e-5, end + 1e-5];
            let mut t = t_start + 0.5;
            while t < t_end + 2000.0 {
                if (t - end).abs() > 1e-7 {
                    checks.push(t);
                }
                t += 61.0;
            }
            for t in checks.into_iter().filter(|&t| t < next_start) {
                let where_ = format!(
                    "{block} beta {beta_deg} rate {rate_factor} mu_rate: t - zone end {} (meeting {})",
                    t - t_end,
                    end - t_end
                );
                let attitude =
                    gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
                if in_turn(t) {
                    assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
                    assert!(
                        wrap_pi(attitude.yaw_rad - linear(t)).abs() < 1e-9,
                        "{where_}: yaw {} vs {}",
                        attitude.yaw_rad,
                        linear(t)
                    );
                } else {
                    assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                }
            }
        }
    }
    assert!(
        across_midnight >= 6,
        "{across_midnight} turns run past midnight"
    );
}

/// A turn boundary at t = 0 (here the noon of a circular pass with u = 0 at
/// t = 0) is bisected in a bounded number of steps, not by halving down
/// through the subnormal range: a few hundred trajectory reads per call.
#[test]
fn turn_boundaries_at_time_zero_take_few_trajectory_reads() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let orbit = Circular {
        a: GPS_A,
        beta: deg(0.5),
        u0: 0.0,
    };
    let reads = std::cell::Cell::new(0_u32);
    let trajectory = |t: f64| {
        reads.set(reads.get() + 1);
        Some(orbit.sample(t))
    };
    for t in [0.0, 50.0, -50.0] {
        reads.set(0);
        gnss_attitude(&law, &trajectory, t).unwrap();
        assert!(reads.get() < 600, "t {t}: {} trajectory reads", reads.get());
    }
}

/// The published law of each block at Sun elevations at and next to zero,
/// written out for a circular orbit with a fixed Sun: `Some((mode, yaw))`, or
/// `None` for instants not checked. `u` is the argument from noon and `n` the
/// orbital rate. GPS (Kouba 2009, `eclips.f`): noon turns from Kouba's zone
/// edge at `-sign(beta) R`, at `sign(b) R` for beta between 0 and the bias
/// and at beta = 0; the IIR midnight turn at `sign(beta) R`; the IIA shadow at
/// `sign(b) R`; the IIF shadow at the rate linking the nominal yaws at entry
/// and exit. GLONASS-M (Dilssner et al. 2011): the noon turn at `-sign(beta)
/// R` over the half span `k y`, `y = |phi(k y) - phi(-k y)| / 2`, and the
/// shadow at `sign(beta) R` held at the exit's nominal yaw. Galileo IOV: the
/// auxiliary Sun vector with `G = sign(-sin(beta))`. GPS III: Montenbruck et
/// al. (2026) eqs. (11)-(18) ([`gps_iii_paper_yaw`]). Galileo FOC and BeiDou-3
/// CAST: the cosine turn from the start angle, signed by the start yaw.
/// BeiDou-3 SECM: beta held at `+-3 deg`. BeiDou-2 and GEO: orbit normal. Signs
/// of zero follow IEEE `copysign`, with `+0` counted positive.
fn small_beta_law(
    block: GnssBlock,
    a: f64,
    beta: f64,
    u: f64,
    dt_centre: f64,
) -> Option<(AttitudeMode, f64)> {
    let n = (GM / (a * a * a)).sqrt();
    let t_b = libm::tan(beta);
    let nominal = |u: f64| libm::atan2(-t_b, libm::sin(u - PI));
    let noon = libm::cos(u) > 0.0;
    // A Kouba turn around the centre at argument `centre_u` with zone half
    // span `x_s`, start yaw `psi0` and rate `r`: nominal before its start,
    // the linear yaw until it meets the nominal yaw ([`follow_gap`] from the
    // zone end, onward or trailing), at most 1800 s after the zone, nominal
    // after. `dt_centre` is the time from the centre, positive after it.
    let kouba_turn = |centre_u: f64,
                      x_s: f64,
                      psi0: f64,
                      r: f64,
                      onward: bool,
                      mode: AttitudeMode|
     -> Option<(AttitudeMode, f64)> {
        let elapsed = dt_centre + x_s / n;
        if elapsed <= 0.0 {
            return Some((AttitudeMode::Nominal, nominal(u)));
        }
        let at = |el: f64| nominal(centre_u + n * el - x_s);
        let zone_end = 2.0 * x_s / n;
        let d = r.signum();
        // Just past the zone end, where at beta = 0 the nominal yaw has
        // flipped by pi.
        let after = zone_end + 1e-9;
        let ahead = d * (at(after) - (psi0 + r * after));
        let gap0 = if onward {
            ahead.rem_euclid(TAU)
        } else {
            kouba_trailing_gap(ahead)
        };
        let limit = zone_end + 1800.0;
        let end = follow_gap(&at, after, limit, gap0, d, r.abs());
        // Within 1e-6 s of the meeting the two evaluations may differ.
        if (elapsed - end).abs() < 1e-6 {
            return None;
        }
        if elapsed < end || (end == limit && elapsed <= limit) {
            Some((mode, psi0 + r * elapsed))
        } else {
            Some((AttitudeMode::Nominal, nominal(u)))
        }
    };
    // Time from the centre of this half orbit, positive after it.
    let gps = |rate: f64, bias: f64, iir: bool| -> Option<(AttitudeMode, f64)> {
        if noon {
            let x_s = if beta == 0.0 {
                0.0
            } else {
                kouba_half_span(beta, n, rate)
            };
            let psi0 = libm::atan2(-t_b, libm::sin(x_s));
            let in_bias =
                bias != 0.0 && (beta == 0.0 || (beta * bias > 0.0 && beta.abs() <= bias.abs()));
            let r = if in_bias {
                rate.copysign(bias)
            } else {
                -rate.copysign(beta)
            };
            let onward = in_bias && beta != 0.0;
            kouba_turn(0.0, x_s, psi0, r, onward, AttitudeMode::NoonTurn)
        } else if iir {
            let x_s = if beta == 0.0 {
                0.0
            } else {
                kouba_half_span(beta, n, rate)
            };
            let psi0 = libm::atan2(-t_b, -libm::sin(x_s));
            kouba_turn(
                PI,
                x_s,
                psi0,
                rate.copysign(beta),
                false,
                AttitudeMode::MidnightTurn,
            )
        } else {
            let h = libm::acos(-libm::cos(GPS_SHADOW_LIMIT_RAD) / libm::cos(beta));
            let elapsed = dt_centre + h / n;
            if !(elapsed > 0.0 && elapsed < 2.0 * h / n) {
                return None;
            }
            let psi_in = libm::atan2(-t_b, -libm::sin(h));
            let r = if bias == GPS_IIF_DEFAULT_YAW_BIAS_RAD {
                (libm::atan2(-t_b, libm::sin(h)) - psi_in) / (2.0 * h / n)
            } else if bias < 0.0 {
                -rate
            } else {
                rate
            };
            Some((AttitudeMode::Shadow, psi_in + r * elapsed))
        }
    };
    let cosine = |start_deg: f64, period: f64, colinearity: bool| -> Option<(AttitudeMode, f64)> {
        let start = if colinearity {
            libm::acos(libm::cos(deg(start_deg)) / libm::cos(beta))
        } else {
            deg(start_deg)
        };
        let elapsed = dt_centre + start / n;
        if !(elapsed > 0.0 && elapsed < 0.5 * period) {
            return None;
        }
        let centre = if noon { 0.0 } else { PI };
        let psi0 = nominal(centre - start);
        let quarter = if psi0 >= 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
        let mode = if noon {
            AttitudeMode::NoonTurn
        } else {
            AttitudeMode::MidnightTurn
        };
        Some((
            mode,
            quarter + (psi0 - quarter) * libm::cos(TAU * elapsed / period),
        ))
    };
    match block {
        GnssBlock::GpsIIA => gps(
            gps_iia_default_yaw_rate_rad_s(1).unwrap(),
            GPS_IIA_DEFAULT_YAW_BIAS_RAD,
            false,
        ),
        GnssBlock::GpsIIR => gps(GPS_IIR_DEFAULT_YAW_RATE_RAD_S, 0.0, true),
        GnssBlock::GpsIIF => gps(
            GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
            GPS_IIF_DEFAULT_YAW_BIAS_RAD,
            false,
        ),
        GnssBlock::GlonassM => {
            let rate = GLONASS_M_DEFAULT_YAW_RATE_RAD_S;
            let k = n / rate;
            if noon {
                let swing = |y: f64| {
                    let d = k * y;
                    0.5 * (libm::atan2(-t_b, libm::sin(d)) - libm::atan2(-t_b, -libm::sin(d))).abs()
                };
                let (mut lo, mut hi) = (1e-12, PI.min(PI / k));
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if mid - swing(mid) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let h = k * 0.5 * (lo + hi);
                let elapsed = dt_centre + h / n;
                (elapsed > 0.0 && elapsed < 2.0 * h / n).then(|| {
                    (
                        AttitudeMode::NoonTurn,
                        libm::atan2(-t_b, libm::sin(h)) - rate.copysign(beta) * elapsed,
                    )
                })
            } else {
                let h = libm::acos(-libm::cos(GLONASS_M_SHADOW_LIMIT_RAD) / libm::cos(beta));
                let elapsed = dt_centre + h / n;
                if !(elapsed > 0.0 && elapsed < 2.0 * h / n) {
                    return None;
                }
                let (psi_in, psi_out) = (
                    libm::atan2(-t_b, -libm::sin(h)),
                    libm::atan2(-t_b, libm::sin(h)),
                );
                let linear = psi_in + rate.copysign(beta) * elapsed;
                let yaw = if beta.is_sign_negative() {
                    linear.max(psi_out)
                } else {
                    linear.min(psi_out)
                };
                Some((AttitudeMode::Shadow, yaw))
            }
        }
        GnssBlock::GalileoIov => {
            let s_x = -libm::sin(u) * libm::cos(beta);
            let s_y = -libm::sin(beta);
            let (sin_x, sin_y) = (libm::sin(deg(15.0)), libm::sin(deg(2.0)));
            if s_x.abs() >= sin_x {
                return Some((AttitudeMode::Nominal, nominal(u)));
            }
            let gamma = 1.0_f64.copysign(s_y);
            let s_hy = 0.5 * (sin_y * gamma + s_y)
                + 0.5 * (sin_y * gamma - s_y) * libm::cos(PI * s_x.abs() / sin_x);
            let mode = if noon {
                AttitudeMode::NoonTurn
            } else {
                AttitudeMode::MidnightTurn
            };
            Some((mode, libm::atan2(s_hy, s_x)))
        }
        GnssBlock::GpsIII => {
            // Orbit angle from midnight.
            let mu = u + PI;
            let (in_region, yaw) = gps_iii_paper_yaw(beta, mu, -libm::sin(beta));
            let mode = match (in_region, noon) {
                (false, _) => AttitudeMode::Nominal,
                (true, true) => AttitudeMode::NoonTurn,
                (true, false) => AttitudeMode::MidnightTurn,
            };
            Some((mode, yaw))
        }
        GnssBlock::GalileoFoc => cosine(10.0, 5656.0, true),
        GnssBlock::Beidou3Igso => cosine(6.0, 5740.0, false),
        GnssBlock::Beidou3Meo {
            manufacturer: Some(BeidouManufacturer::Cast),
        } => cosine(6.0, 3090.0, false),
        GnssBlock::Beidou3Meo {
            manufacturer: Some(BeidouManufacturer::Secm),
        } => {
            let held = if beta.is_sign_negative() {
                -deg(3.0)
            } else {
                deg(3.0)
            };
            Some((
                AttitudeMode::ModifiedYawSteering,
                libm::atan2(-libm::tan(held), libm::sin(u - PI)),
            ))
        }
        GnssBlock::Beidou2Igso
        | GnssBlock::Beidou2Meo
        | GnssBlock::Beidou2Geo
        | GnssBlock::Beidou3Geo => Some((AttitudeMode::OrbitNormal, 0.0)),
        _ => None,
    }
}

/// beta = 0 exactly (an orbit in the XY plane with the Sun along +X, so the
/// Sun's component along the orbit normal is exactly zero) and beta = +-1e-9
/// on an orbit turned into a general orientation: every block's yaw and turn
/// direction against [`small_beta_law`], around noon and midnight. Each law's
/// two sides of beta = 0 are told apart by the sign of beta, and laws whose
/// direction is the same on both sides (the GPS IIA and IIF noon turns, the
/// GPS IIA shadow) turn that way at beta = 0 too.
#[test]
fn every_law_has_its_published_yaw_and_direction_at_and_next_to_beta_zero() {
    let blocks = [
        (GnssBlock::GpsIIA, GPS_A),
        (GnssBlock::GpsIIR, GPS_A),
        (GnssBlock::GpsIIF, GPS_A),
        (GnssBlock::GpsIII, GPS_A),
        (GnssBlock::GlonassM, GLONASS_A),
        (GnssBlock::GalileoIov, GALILEO_A),
        (GnssBlock::GalileoFoc, GALILEO_A),
        (GnssBlock::Beidou2Meo, MEO_A),
        (GnssBlock::Beidou2Igso, 42_164_200.0),
        (GnssBlock::Beidou2Geo, 42_164_200.0),
        (GnssBlock::Beidou3Igso, 42_164_200.0),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            },
            MEO_A,
        ),
        (
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Secm),
            },
            MEO_A,
        ),
    ];
    let mut checked = 0;
    for (block, a) in blocks {
        let law = GnssAttitudeLaw::for_block(block, Some(1)).unwrap();
        for (beta, tilt) in [
            (0.0, false),
            (1e-9, true),
            (-1e-9, true),
            (1e-9, false),
            (-1e-9, false),
        ] {
            let orbit = Circular {
                a,
                beta,
                u0: -deg(30.0),
            };
            let plain = orbit.trajectory();
            let turned = tilted(orbit.trajectory());
            let trajectory: &dyn Fn(f64) -> Option<AttitudeSample> =
                if tilt { &turned } else { &plain };
            let n = orbit.n();
            for centre in [0.0, PI] {
                let t_centre = orbit.time_at(centre);
                for dt in [
                    -1500.0, -600.0, -300.0, -100.0, -1.0, -1e-3, 1e-3, 1.0, 100.0, 300.0, 600.0,
                    880.0, 905.0, 1500.0, 1640.0, 1700.0,
                ] {
                    let t = t_centre + dt;
                    let u = centre + n * dt;
                    let where_ =
                        format!("{block} beta {beta:e} tilted {tilt} centre {centre} dt {dt}");
                    let Some((mode, yaw)) = small_beta_law(block, a, beta, u, dt) else {
                        continue;
                    };
                    let attitude = gnss_attitude(&law, trajectory, t)
                        .unwrap_or_else(|e| panic!("{where_}: {e}"));
                    assert_eq!(
                        attitude.mode, mode,
                        "{where_}: yaw {} vs {yaw}",
                        attitude.yaw_rad
                    );
                    let error = wrap_pi(attitude.yaw_rad - yaw).abs();
                    assert!(error < 1e-8, "{where_}: yaw {} vs {yaw}", attitude.yaw_rad);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 600, "{checked} instants checked");
}

/// The instant in `[lo, hi]` where `inside` changes, by bisection to 1e-10 s.
fn change(inside: impl Fn(f64) -> bool, mut lo: f64, mut hi: f64) -> f64 {
    let at_lo = inside(lo);
    assert_ne!(at_lo, inside(hi));
    while hi - lo > 1e-10 {
        let mid = 0.5 * lo + 0.5 * hi;
        if inside(mid) == at_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * lo + 0.5 * hi
}

/// GPS IIA, IIR, IIF and GLONASS-M turns on eccentric orbits with a fixed Sun,
/// the zones solved here on the Keplerian closed form: Kouba's nominal-rate
/// zone `|mu_rate T cos(x) / (sin^2(x) + T^2)| > R` with the instantaneous
/// rate `mu_rate = h / r^2`, the shadow `cos(beta) cos(u) < cos(limit)`, the
/// Dilssner span at the instantaneous `k = mu_rate / R`; the yaw from the
/// nominal yaw at the zone start at the law's rate, the GPS turns until the
/// gap to the continuous nominal yaw closes (GPS IIA after the shadow the
/// shorter way, Bar-Sever 1996), the GPS IIF shadow at the rate linking its
/// nominal yaws, the GLONASS-M shadow held at the exit's nominal yaw.
/// Eccentricities 0.02 (GNSS) and 0.1 and 0.3 (0.25 for GLONASS-M), perigee
/// at noon, a quadrature and midnight, some orbits turned into a general
/// orientation.
#[test]
fn gps_and_glonass_turns_on_eccentric_orbits() {
    let iia = gps_iia_default_yaw_rate_rad_s(11).unwrap();
    let laws = [
        (GnssBlock::GpsIIA, GPS_A, iia),
        (GnssBlock::GpsIIR, GPS_A, GPS_IIR_DEFAULT_YAW_RATE_RAD_S),
        (GnssBlock::GpsIIF, GPS_A, GPS_IIF_DEFAULT_YAW_RATE_RAD_S),
        (
            GnssBlock::GlonassM,
            GLONASS_A,
            GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
        ),
    ];
    let mut turns = 0;
    for (block, a, rate) in laws {
        let law = GnssAttitudeLaw::for_block(block, Some(11)).unwrap();
        for (e, omega_deg, tilt) in [
            (0.02, 0.0, false),
            (0.02, 90.0, true),
            (0.02, 200.0, false),
            (0.1, 270.0, true),
            (0.3, 0.0, false),
            (0.3, 180.0, true),
        ] {
            // The GLONASS-M noon bound below 30 deg of beta is 0.25.
            let e = if block == GnssBlock::GlonassM {
                f64::min(e, 0.25)
            } else {
                e
            };
            for beta_deg in [0.6, -1.0, 8.0] {
                let beta = deg(beta_deg);
                let orbit = Kepler {
                    a,
                    e,
                    beta,
                    beta_rate: 0.0,
                    omega: deg(omega_deg),
                    m0: 0.0,
                };
                let plain = orbit.trajectory();
                let turned = tilted(orbit.trajectory());
                let trajectory: &dyn Fn(f64) -> Option<AttitudeSample> =
                    if tilt { &turned } else { &plain };
                let t_b = libm::tan(beta);
                let u_of = |t: f64| orbit.argument_from_noon(t);
                let nominal = |t: f64| libm::atan2(-t_b, libm::sin(u_of(t) - PI));
                let mu_rate = |t: f64| {
                    let s = orbit.sample(t);
                    norm3(cross3(s.position_m, s.velocity_m_s)) / dot3(s.position_m, s.position_m)
                };
                for noon in [true, false] {
                    let centre = orbit.time_at(if noon { 0.0 } else { PI }, orbit.period());
                    let x_of = |t: f64| {
                        if noon {
                            -u_of(t)
                        } else {
                            wrap_pi(PI - u_of(t))
                        }
                    };
                    let glonass = block == GnssBlock::GlonassM;
                    let shadow_limit = match block {
                        GnssBlock::GlonassM => GLONASS_M_SHADOW_LIMIT_RAD,
                        GnssBlock::GpsIIR => f64::NAN,
                        _ => GPS_SHADOW_LIMIT_RAD,
                    };
                    let shadow = !noon && !shadow_limit.is_nan();
                    let glonass_span = |t: f64| {
                        let k = mu_rate(t) / rate;
                        let swing = |y: f64| {
                            let d = k * y;
                            0.5 * (libm::atan2(-t_b, libm::sin(d))
                                - libm::atan2(-t_b, -libm::sin(d)))
                            .abs()
                        };
                        let (mut lo, mut hi) = (1e-12, PI.min(PI / k));
                        for _ in 0..200 {
                            let mid = 0.5 * (lo + hi);
                            if mid - swing(mid) < 0.0 {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        k * 0.5 * (lo + hi)
                    };
                    // Positive exactly inside the zone.
                    let zone_margin = |t: f64| -> f64 {
                        let x = x_of(t);
                        if shadow {
                            libm::cos(shadow_limit) - libm::cos(beta) * libm::cos(u_of(t))
                        } else if glonass {
                            if t_b.abs() < mu_rate(t) / rate {
                                glonass_span(t) - x.abs()
                            } else {
                                -1.0
                            }
                        } else {
                            let s = libm::sin(x);
                            mu_rate(t) * t_b.abs() * libm::cos(x) / (s * s + t_b * t_b) - rate
                        }
                    };
                    let in_zone = |t: f64| zone_margin(t) > 0.0;
                    // The zone anywhere in the half orbit between the
                    // quadratures (on an eccentric orbit it need not hold at
                    // the centre): a 5 s scan with the centre, one run of
                    // instants inside, its edges bisected. Where no scan
                    // point is inside, the scan point of largest margin is
                    // zoomed in on (a scan of twice the step around it at a
                    // twentieth of it, eleven times, down to 1e-13 s): a zone
                    // narrower than the step is found there, or there is none
                    // and the law must give no turn there.
                    let centre_u = if noon { 0.0 } else { PI };
                    let (t_lo, t_hi) = (
                        orbit.time_at(centre_u - FRAC_PI_2, centre - orbit.period()),
                        orbit.time_at(centre_u + FRAC_PI_2, centre),
                    );
                    let step = 5.0;
                    let mut scan: Vec<f64> = Vec::new();
                    let mut t = t_lo;
                    while t < t_hi {
                        scan.push(t);
                        t += step;
                    }
                    scan.push(centre);
                    scan.sort_by(f64::total_cmp);
                    let inside: Vec<usize> =
                        (0..scan.len()).filter(|&i| in_zone(scan[i])).collect();
                    let (start, zone_end) = match (inside.first(), inside.last()) {
                        (Some(&first), Some(&last)) => {
                            assert_eq!(last - first + 1, inside.len(), "one run of the zone");
                            (
                                change(in_zone, scan[first] - step, scan[first]),
                                change(in_zone, scan[last], scan[last] + step),
                            )
                        }
                        _ => {
                            let mut best = scan
                                .iter()
                                .copied()
                                .max_by(|a, b| zone_margin(*a).total_cmp(&zone_margin(*b)))
                                .unwrap();
                            let mut width = step;
                            for _ in 0..11 {
                                let (lo, fine) = (best - width, width / 20.0);
                                for k in 0..=40 {
                                    let t = lo + fine * f64::from(k);
                                    if zone_margin(t) > zone_margin(best) {
                                        best = t;
                                    }
                                }
                                width = fine;
                            }
                            if !in_zone(best) {
                                let where_ = format!(
                                    "{block} e {e} omega {omega_deg} beta {beta_deg} noon {noon}"
                                );
                                let attitude = gnss_attitude(&law, trajectory, best)
                                    .unwrap_or_else(|e| panic!("{where_}: {e}"));
                                assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                                continue;
                            }
                            (
                                change(in_zone, best - step, best),
                                change(in_zone, best, best + step),
                            )
                        }
                    };
                    let psi0 = nominal(start);
                    let bias = match block {
                        GnssBlock::GpsIIA => GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                        GnssBlock::GpsIIF => GPS_IIF_DEFAULT_YAW_BIAS_RAD,
                        _ => 0.0,
                    };
                    let (turn_rate, mode) = if noon {
                        let in_bias = bias != 0.0 && beta * bias > 0.0 && beta.abs() <= bias.abs();
                        (
                            if in_bias {
                                rate.copysign(bias)
                            } else {
                                -rate.copysign(beta)
                            },
                            AttitudeMode::NoonTurn,
                        )
                    } else {
                        match block {
                            GnssBlock::GpsIIR => (rate.copysign(beta), AttitudeMode::MidnightTurn),
                            GnssBlock::GpsIIF => (
                                (nominal(zone_end) - psi0) / (zone_end - start),
                                AttitudeMode::Shadow,
                            ),
                            GnssBlock::GlonassM => (rate.copysign(beta), AttitudeMode::Shadow),
                            _ => (rate, AttitudeMode::Shadow),
                        }
                    };
                    let linear = |t: f64| psi0 + turn_rate * (t - start);
                    let exit = linear(zone_end);
                    // After the zone: the rate, and the gap to the nominal yaw.
                    let (after, gap0) = if glonass || (block == GnssBlock::GpsIIF && !noon) {
                        (turn_rate, 0.0)
                    } else if block == GnssBlock::GpsIIA && !noon {
                        let x = nominal(zone_end) - exit;
                        let d = x - TAU * libm::round(x / TAU);
                        (rate.copysign(d), d.abs())
                    } else {
                        let g = turn_rate.signum() * (nominal(zone_end) - exit);
                        let reversed = noon && turn_rate.signum() == beta.signum();
                        let g = if reversed {
                            g.rem_euclid(TAU)
                        } else {
                            kouba_trailing_gap(g)
                        };
                        (turn_rate, g)
                    };
                    let gap = |t: f64| {
                        gap0 + after.signum() * (nominal(t) - nominal(zone_end))
                            - after.abs() * (t - zone_end)
                    };
                    let reach = if gap0 == 0.0 {
                        zone_end
                    } else {
                        let mut t = zone_end;
                        let limit = zone_end + 1800.0;
                        loop {
                            if t >= limit {
                                break limit;
                            }
                            let next = (t + 1.0).min(limit);
                            if gap(next) <= 0.0 {
                                break change(|x| gap(x) > 0.0, t, next);
                            }
                            t = next;
                        }
                    };
                    let hold = nominal(zone_end);
                    let expected = |t: f64| -> f64 {
                        if t <= zone_end {
                            let y = linear(t);
                            if glonass && !noon {
                                if turn_rate > 0.0 {
                                    y.min(hold)
                                } else {
                                    y.max(hold)
                                }
                            } else {
                                y
                            }
                        } else {
                            exit + after * (t - zone_end)
                        }
                    };
                    turns += 1;
                    let span = zone_end - start;
                    let mut checks = vec![
                        start + 1e-5,
                        start + 0.3 * span,
                        start + 0.7 * span,
                        zone_end - 1e-5,
                    ];
                    if reach > zone_end + 1e-2 {
                        checks.extend([zone_end + 0.5 * (reach - zone_end), reach - 1e-5]);
                    }
                    for t in checks {
                        let where_ = format!(
                            "{block} e {e} omega {omega_deg} tilted {tilt} beta {beta_deg} noon {noon} t - zone start {} (zone {span} s, meeting {} s after)",
                            t - start,
                            reach - zone_end
                        );
                        let attitude = gnss_attitude(&law, trajectory, t)
                            .unwrap_or_else(|e| panic!("{where_}: {e}"));
                        assert_eq!(attitude.mode, mode, "{where_}");
                        let error = wrap_pi(attitude.yaw_rad - expected(t)).abs();
                        assert!(
                            error < 1e-9,
                            "{where_}: yaw {} vs {}",
                            attitude.yaw_rad,
                            expected(t)
                        );
                    }
                    for t in [start - 1e-5, reach + 1e-5] {
                        let attitude = gnss_attitude(&law, trajectory, t).unwrap();
                        assert_eq!(
                            attitude.mode,
                            AttitudeMode::Nominal,
                            "{block} e {e} omega {omega_deg} beta {beta_deg} noon {noon} t - zone start {}",
                            t - start
                        );
                    }
                }
            }
        }
    }
    assert!(turns > 60, "{turns} turns");
}

/// Galileo IOV and GPS III on eccentric orbits: the auxiliary-Sun-vector law
/// of the Galileo satellite metadata (3.1.1) from the Keplerian argument from
/// noon, `S_x = -sin(eta) cos(beta)`, `S_y = -sin(beta)`, in the region
/// `|S_x| < sin(beta_x)`, `|S_y| < sin(beta_y)`, `G = sign(S_y)` (beta
/// fixed), with the metadata's `beta_x = 15 deg`, `beta_y = 2 deg` for Galileo
/// IOV and Montenbruck et al. (2026)'s `15 deg` and `5.8 deg` for GPS III, on
/// orbits turned into a general orientation.
#[test]
fn galileo_iov_law_on_eccentric_orbits() {
    for (block, a, beta_y_deg, betas) in [
        (
            GnssBlock::GalileoIov,
            GALILEO_A,
            2.0,
            [1.5, -0.4, 1e-9, -1e-9, 2.5, 1.9],
        ),
        (
            GnssBlock::GpsIII,
            GPS_A,
            5.8,
            [5.7, -3.3, 1e-9, -1e-9, 6.0, -0.4],
        ),
    ] {
        let law = GnssAttitudeLaw::for_block(block, None).unwrap();
        let (sin_x, sin_y) = (libm::sin(deg(15.0)), libm::sin(deg(beta_y_deg)));
        for (e, omega_deg) in [(0.02, 30.0), (0.3, 180.0), (0.5, 90.0)] {
            for beta_deg in betas {
                let beta = deg(beta_deg);
                let orbit = Kepler {
                    a,
                    e,
                    beta,
                    beta_rate: 0.0,
                    omega: deg(omega_deg),
                    m0: 1.0,
                };
                let trajectory = tilted(orbit.trajectory());
                let mut t = 0.0;
                let mut in_region = 0;
                while t < orbit.period() {
                    let u = orbit.argument_from_noon(t);
                    let s_x = -libm::sin(u) * libm::cos(beta);
                    let s_y = -libm::sin(beta);
                    let where_ = format!("{block} e {e} omega {omega_deg} beta {beta_deg} t {t}");
                    let attitude = gnss_attitude(&law, &trajectory, t)
                        .unwrap_or_else(|e| panic!("{where_}: {e}"));
                    if s_x.abs() < sin_x && s_y.abs() < sin_y {
                        in_region += 1;
                        let gamma = 1.0_f64.copysign(s_y);
                        let s_hy = 0.5 * (sin_y * gamma + s_y)
                            + 0.5 * (sin_y * gamma - s_y) * libm::cos(PI * s_x.abs() / sin_x);
                        let yaw = libm::atan2(s_hy, s_x);
                        assert_ne!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                        assert!(
                            wrap_pi(attitude.yaw_rad - yaw).abs() < 1e-9,
                            "{where_}: {} vs {yaw}",
                            attitude.yaw_rad
                        );
                    } else {
                        assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                    }
                    t += 97.0;
                }
                if beta_deg.abs() < beta_y_deg {
                    assert!(in_region > 5, "{block} e {e} beta {beta_deg}: {in_region}");
                } else {
                    assert_eq!(in_region, 0, "{block} e {e} beta {beta_deg}");
                }
            }
        }
    }
}

/// Every finite positive maximum yaw rate, from the smallest subnormal to the
/// largest float, for every GPS law and GLONASS-M at beta from 0 to 60 deg
/// around noon and midnight: a valid attitude (finite, yaw in `(-pi, pi]`,
/// orthonormal axes), or a typed error for a product the law cannot form
/// (`NonFinite` for the rate times the time in a turn, `InvalidParameter` for
/// a GLONASS-M rate whose ratio to the orbital rate overflows or whose noon
/// turn reaches the quadrature, `OutsideValidatedDomain` for a GPS rate below
/// 1.8e-3 times the orbital rate at beta = 0, where Kouba's zone would depend
/// on beta below its rounding, and for a GPS IIA rate above 1e5 times the
/// orbital rate). Where the rate is so fast that every turn but
/// the GPS IIA shadow lasts less than the resolution of time, the attitude a
/// second from the centres is nominal.
#[test]
fn every_yaw_rate_gives_a_valid_attitude_or_a_typed_error() {
    let rates = [
        5e-324,
        1e-310,
        f64::MIN_POSITIVE,
        1e-300,
        1e-160,
        1e-10,
        1e3,
        1e10,
        1e150,
        1e300,
        1e306,
        f64::MAX,
    ];
    let mut valid = 0;
    for rate in rates {
        let laws = [
            GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: rate,
                yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            },
            GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: rate,
                yaw_bias_rad: -GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            },
            GnssAttitudeLaw::GpsIIR {
                yaw_rate_rad_s: rate,
            },
            GnssAttitudeLaw::GpsIIF {
                yaw_rate_rad_s: rate,
                yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            },
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: rate,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
        ];
        for law in laws {
            for beta_deg in [0.0, 1e-7, -1e-7, 0.5, -8.0, 60.0] {
                let orbit = Circular {
                    a: GPS_A,
                    beta: deg(beta_deg),
                    u0: -deg(30.0),
                };
                for centre in [0.0, PI] {
                    let t_centre = orbit.time_at(centre);
                    for dt in [-1000.0, -1.0, 0.0, 1.0, 1000.0] {
                        let where_ = format!("{law:?} beta {beta_deg} centre {centre} dt {dt}");
                        match gnss_attitude(&law, &orbit.trajectory(), t_centre + dt) {
                            Ok(attitude) => {
                                assert_valid(&attitude, &where_);
                                valid += 1;
                                let shadow_turn = centre == PI
                                    && matches!(
                                        law,
                                        GnssAttitudeLaw::GpsIIA { .. }
                                            | GnssAttitudeLaw::GpsIIF { .. }
                                            | GnssAttitudeLaw::GlonassM { .. }
                                    )
                                    && beta_deg.abs() < 13.0;
                                if rate >= 1e10 && dt.abs() >= 1.0 && !shadow_turn {
                                    assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                                }
                            }
                            Err(GnssAttitudeError::NonFinite { field }) => assert!(
                                field.starts_with("yaw_rate_rad_s x")
                                    || field.starts_with("t_s - turn start"),
                                "{where_}: NonFinite {field}"
                            ),
                            Err(GnssAttitudeError::InvalidParameter {
                                field: "yaw_rate_rad_s",
                                ..
                            }) => assert!(
                                matches!(law, GnssAttitudeLaw::GlonassM { .. }),
                                "{where_}: InvalidParameter"
                            ),
                            Err(GnssAttitudeError::OutsideValidatedDomain { reason, .. }) => {
                                let slow_at_zero = beta_deg == 0.0
                                    && rate < 1.8e-3 * orbit.n()
                                    && reason.starts_with("beta within the rounding of zero");
                                let fast_iia = matches!(law, GnssAttitudeLaw::GpsIIA { .. })
                                    && rate > 1e5 * orbit.n()
                                    && reason.starts_with("GPS IIA maximum yaw rate");
                                assert!(slow_at_zero || fast_iia, "{where_}: {reason}");
                            }
                            Err(e) => panic!("{where_}: {e}"),
                        }
                    }
                }
            }
        }
    }
    assert!(valid > 1000, "{valid} valid attitudes");
}

/// Galileo satellite metadata 3.1.1 with custom limits `beta_x = 10 deg` and
/// `beta_y = 12 deg` at beta = 11 deg, above `beta_x`: the region
/// `|S_x| < sin(beta_x)`, here `|x| < asin(sin(10 deg) / cos(11 deg))` in plane
/// around midnight, is an arc, not one instant, and at its edges the yaw is
/// the nominal yaw.
#[test]
fn galileo_iov_region_with_custom_limits_is_an_arc() {
    let (beta_x, beta_y) = (deg(10.0), deg(12.0));
    let law = GnssAttitudeLaw::GalileoIov {
        beta_x_rad: beta_x,
        beta_y_rad: beta_y,
    };
    let beta = deg(11.0);
    let orbit = Circular {
        a: GALILEO_A,
        beta,
        u0: PI - deg(20.0),
    };
    let n = orbit.n();
    let t_midnight = orbit.time_at(PI);
    let x_edge = libm::asin(libm::sin(beta_x) / libm::cos(beta));
    let (sin_x, sin_y) = (libm::sin(beta_x), libm::sin(beta_y));
    for f in [-0.999, -0.5, 0.0, 0.5, 0.999] {
        let x = f * x_edge;
        let t = t_midnight + x / n;
        let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::MidnightTurn, "f {f}");
        let s_x = -libm::sin(PI + x) * libm::cos(beta);
        let s_y = -libm::sin(beta);
        let gamma = -1.0;
        let s_hy = 0.5 * (sin_y * gamma + s_y)
            + 0.5 * (sin_y * gamma - s_y) * libm::cos(PI * s_x.abs() / sin_x);
        let yaw = libm::atan2(s_hy, s_x);
        assert!(
            wrap_pi(attitude.yaw_rad - yaw).abs() < 1e-9,
            "f {f}: {} vs {yaw}",
            attitude.yaw_rad
        );
        if f.abs() > 0.99 {
            assert!(
                wrap_pi(attitude.yaw_rad - attitude.nominal_yaw_rad).abs() < 1e-3,
                "f {f}: {} vs nominal {}",
                attitude.yaw_rad,
                attitude.nominal_yaw_rad
            );
        }
    }
    for f in [-1.001, 1.001] {
        let t = t_midnight + f * x_edge / n;
        let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
        assert_eq!(attitude.mode, AttitudeMode::Nominal, "f {f}");
    }
}

/// Galileo satellite metadata 3.1.1 with the published limits at beta =
/// 1.9 deg: the region `|S_x| = cos(beta) |sin(x)| < sin(15 deg)` reaches
/// `x = asin(sin(15 deg) / cos(1.9 deg)) = 15.0083 deg` in plane on both sides
/// of noon and of midnight alike, so 14.95 deg from either centre is inside
/// and 15.05 deg outside.
#[test]
fn galileo_iov_region_is_the_same_around_noon_and_midnight() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GalileoIov, None).unwrap();
    let beta = deg(1.9);
    let orbit = Circular {
        a: GALILEO_A,
        beta,
        u0: -deg(30.0),
    };
    let edge = libm::asin(libm::sin(deg(15.0)) / libm::cos(beta));
    assert!(edge > deg(15.0) && edge < deg(15.01), "{edge}");
    for centre in [0.0, PI] {
        for (x_deg, inside) in [
            (14.95, true),
            (-14.95, true),
            (15.05, false),
            (-15.05, false),
        ] {
            let t = orbit.time_at(centre - deg(x_deg));
            let attitude = gnss_attitude(&law, &orbit.trajectory(), t).unwrap();
            let where_ = format!("centre {centre} x {x_deg}");
            if inside {
                assert_ne!(attitude.mode, AttitudeMode::Nominal, "{where_}");
            } else {
                assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
            }
        }
    }
}

/// Montenbruck et al. (2026), Section 3, in the paper's notation: the Sun unit
/// vector in the orbital frame (x along track, y along the orbit normal as
/// eq. 11 signs it, z toward the Earth), `s = (cos(beta) sin(mu), -sin(beta),
/// cos(beta) cos(mu))` with `mu` the orbit angle from midnight (eq. 11), and
/// the collinearity region `|s_x| < gamma_x = sin(15 deg)`,
/// `|s_y| < gamma_y = sin(5.8 deg)`. Inside it `g = cos(pi |s_x| / gamma_x)`
/// (eq. 17), `s_y* = 0.5 (1 + g) sign(s_y,0) gamma_y + 0.5 (1 - g) s_y`
/// (eq. 18) with `s_y,0` the `s_y` where the region began, and the yaw is
/// `atan2(s_y*, s_x)` (eq. 13); outside it the nominal yaw `atan2(s_y, s_x)`
/// (eq. 12). Returns whether `s` is in the region, and the yaw. The sign of a
/// zero `s_y,0` follows its IEEE sign.
fn gps_iii_paper_yaw(beta: f64, mu: f64, s_y0: f64) -> (bool, f64) {
    let (gamma_x, gamma_y) = (libm::sin(deg(15.0)), libm::sin(deg(5.8)));
    let s_x = libm::cos(beta) * libm::sin(mu);
    let s_y = -libm::sin(beta);
    if s_x.abs() < gamma_x && s_y.abs() < gamma_y {
        let g = libm::cos(PI * s_x.abs() / gamma_x);
        let s_y_star = 0.5 * (1.0 + g) * s_y0.signum() * gamma_y + 0.5 * (1.0 - g) * s_y;
        (true, libm::atan2(s_y_star, s_x))
    } else {
        (false, libm::atan2(s_y, s_x))
    }
}

/// GPS III, from the IGS name `BLOCK IIIA` or the constellation block types
/// `III` and `IIIA`, has the Galileo IOV law with `beta_x = 15 deg` and
/// `beta_y = 5.8 deg` (Montenbruck et al. 2026, Section 3 and Summary).
#[test]
fn gps_iii_has_the_iov_law_with_the_limits_of_montenbruck_et_al_2026() {
    let expected = GnssAttitudeLaw::GalileoIov {
        beta_x_rad: GPS_III_BETA_X_RAD,
        beta_y_rad: GPS_III_BETA_Y_RAD,
    };
    assert!((GPS_III_BETA_X_RAD - 15.0_f64.to_radians()).abs() <= f64::EPSILON);
    assert!((GPS_III_BETA_Y_RAD - 5.8_f64.to_radians()).abs() <= f64::EPSILON / 8.0);
    for block in [
        GnssBlock::from_igs_name("BLOCK IIIA").unwrap(),
        GnssBlock::from_constellation_block_type(crate::GnssSystem::Gps, "III", None).unwrap(),
        GnssBlock::from_constellation_block_type(crate::GnssSystem::Gps, "IIIA", None).unwrap(),
    ] {
        assert_eq!(block, GnssBlock::GpsIII);
        assert_eq!(block.system(), crate::GnssSystem::Gps);
        assert_eq!(GnssAttitudeLaw::for_block(block, None), Ok(expected));
        assert_eq!(GnssAttitudeLaw::for_block(block, Some(7)), Ok(expected));
    }
}

/// GPS III on circular orbits with a fixed Sun at the beta angles of the
/// paper's Table 1 and Fig. 6 (-5.7, -3.3 and -0.4 deg), their opposites, next
/// to the 5.8 deg limit on both sides, and beyond it: every 15 s of an orbit
/// the mode and yaw against [`gps_iii_paper_yaw`]; the turn windows, which for
/// a fixed beta reach `asin(sin(15 deg) / cos(beta))` of orbit angle either
/// side of noon and midnight, checked 1 ms either side of each edge, where the
/// yaw is the nominal yaw (`g = -1` gives `s_y* = s_y`); and the yaw rate at
/// noon and midnight, where `s_x = 0`, `g = 1` and `s_y* = sign(s_y,0)
/// gamma_y`, so differentiating eq. (13) gives `-ds_x/dt / s_y*`, of size
/// `mu_rate cos(beta) / gamma_y` (eq. 14 without the `cos(beta)`).
#[test]
fn gps_iii_turn_windows_and_yaw_follow_montenbruck_et_al_2026() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIII, None).unwrap();
    let (gamma_x, gamma_y) = (libm::sin(deg(15.0)), libm::sin(deg(5.8)));
    let mut in_region = 0;
    for beta_deg in [
        -5.7, -3.3, -0.4, 0.4, 3.3, 5.7, 5.79, -5.79, 5.81, -5.81, 8.0, -35.0,
    ] {
        let beta = deg(beta_deg);
        let orbit = Circular {
            a: GPS_A,
            beta,
            u0: -deg(60.0),
        };
        let trajectory = tilted(orbit.trajectory());
        let n = orbit.n();
        let s_y0 = -libm::sin(beta);
        let at = |t: f64| {
            gnss_attitude(&law, &trajectory, t)
                .unwrap_or_else(|e| panic!("beta {beta_deg} t {t}: {e}"))
        };
        let mut t = 0.0;
        while t < TAU / n {
            let where_ = format!("beta {beta_deg} t {t}");
            let mu = orbit.u0 + n * t + PI;
            let (inside, yaw) = gps_iii_paper_yaw(beta, mu, s_y0);
            let attitude = at(t);
            let mode = match (inside, libm::cos(mu) < 0.0) {
                (false, _) => AttitudeMode::Nominal,
                (true, true) => AttitudeMode::NoonTurn,
                (true, false) => AttitudeMode::MidnightTurn,
            };
            assert_eq!(attitude.mode, mode, "{where_}");
            assert!(
                wrap_pi(attitude.yaw_rad - yaw).abs() < 1e-9,
                "{where_}: {} vs {yaw}",
                attitude.yaw_rad
            );
            if !inside {
                // Eq. (1) and eq. (12) are the same nominal yaw.
                let eq1 = libm::atan2(-libm::tan(beta), libm::sin(mu));
                assert!(wrap_pi(yaw - eq1).abs() < 1e-15, "{where_}");
            }
            in_region += usize::from(inside);
            t += 15.0;
        }
        if beta.abs() >= deg(5.8) {
            continue;
        }
        let half = libm::asin(gamma_x / libm::cos(beta));
        for (u_centre, noon) in [(0.0, true), (PI, false)] {
            let t_centre = orbit.time_at(u_centre);
            let turn = if noon {
                AttitudeMode::NoonTurn
            } else {
                AttitudeMode::MidnightTurn
            };
            for edge in [t_centre - half / n, t_centre + half / n] {
                let entry = edge < t_centre;
                for dt in [-1e-3, 1e-3] {
                    let where_ = format!("beta {beta_deg} noon {noon} edge {edge} dt {dt}");
                    let attitude = at(edge + dt);
                    let inside = (dt > 0.0) == entry;
                    if inside {
                        assert_eq!(attitude.mode, turn, "{where_}");
                        assert!(
                            wrap_pi(attitude.yaw_rad - attitude.nominal_yaw_rad).abs() < 1e-9,
                            "{where_}: {} vs nominal {}",
                            attitude.yaw_rad,
                            attitude.nominal_yaw_rad
                        );
                    } else {
                        assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                    }
                }
            }
            // ds_x/dt = cos(beta) cos(mu) mu_rate, -1 or +1 times
            // cos(beta) mu_rate at noon (mu = pi) and midnight (mu = 0).
            let ds_x = if noon { -1.0 } else { 1.0 } * libm::cos(beta) * n;
            let expected = -ds_x / (s_y0.signum() * gamma_y);
            let h = 1.0;
            let rate = wrap_pi(at(t_centre + h).yaw_rad - at(t_centre - h).yaw_rad) / (2.0 * h);
            assert!(
                (rate - expected).abs() < 1e-5 * expected.abs(),
                "beta {beta_deg} noon {noon}: yaw rate {rate} vs {expected}"
            );
            // About 0.08 deg/s, the peak rate of the paper's Fig. 6.
            assert!(
                (rate.abs().to_degrees() - 0.0827).abs() < 1e-3,
                "{}",
                rate.to_degrees()
            );
        }
    }
    assert!(in_region > 1000, "{in_region} samples in the region");
}

/// GPS III with beta changing sign inside the collinearity region (beta drifts
/// at 1e-6 rad/s through zero at noon, on an orbit turned into a general
/// orientation): the auxiliary Sun vector keeps `sign(s_y,0)` from where the
/// region began (Montenbruck et al. 2026, eq. 18), found here by bisection on
/// the region condition, through the whole turn, also after `s_y` has changed
/// sign.
#[test]
fn gps_iii_turn_direction_is_set_where_the_region_begins() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIII, None).unwrap();
    let gamma_x = libm::sin(deg(15.0));
    let n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    let t_noon = deg(60.0) / n;
    for beta_rate in [1e-6, -1e-6] {
        let orbit = Kepler {
            a: GPS_A,
            e: 0.0,
            beta: -beta_rate * t_noon,
            beta_rate,
            omega: 0.0,
            m0: -deg(60.0),
        };
        let trajectory = tilted(orbit.trajectory());
        let beta_at = |t: f64| orbit.beta + orbit.beta_rate * t;
        let mu_at = |t: f64| orbit.argument_from_noon(t) + PI;
        let inside = |t: f64| libm::cos(beta_at(t)) * libm::sin(mu_at(t)).abs() < gamma_x;
        let entry = change(inside, t_noon - 3000.0, t_noon);
        let exit = change(inside, t_noon, t_noon + 3000.0);
        let s_y0 = -libm::sin(beta_at(entry));
        assert_eq!(s_y0.signum(), beta_rate.signum());
        let mut opposite = 0;
        let mut t = entry + 0.5;
        while t < exit {
            let where_ = format!("beta rate {beta_rate} t {t}");
            let (in_region, yaw) = gps_iii_paper_yaw(beta_at(t), mu_at(t), s_y0);
            assert!(in_region, "{where_}");
            let attitude =
                gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
            assert!(
                wrap_pi(attitude.yaw_rad - yaw).abs() < 1e-9,
                "{where_}: {} vs {yaw}",
                attitude.yaw_rad
            );
            if (-libm::sin(beta_at(t))).signum() != s_y0.signum() {
                opposite += 1;
            }
            t += 20.0;
        }
        assert!(opposite > 50, "{opposite} samples after s_y changed sign");
    }
}

/// GPS IIA after the shadow (Bar-Sever 1996; Kouba's `eclips.f`): from the
/// exit the yaw turns at `SIGN(R, D)`, `D` the nominal yaw minus the yaw there
/// wrapped to 180 deg, until it equals the nominal yaw. The nominal yaw moves
/// at under 1e-3 rad/s after these shadows and the yaw at `R = 0.2 deg/s`
/// (PRN 11), so with `|D|` below 1e-9 rad the two meet within 1e-6 s of the
/// exit and the attitude is nominal from then on. Beta is swept through the
/// value near 9.6 deg at which `D` passes through zero, on circular orbits
/// whose midnight is at t = 1e8 s, where one unit in the last place of t is
/// 1.5e-8 s and the nominal yaw moves by up to 1.5e-11 rad within it: `D` from
/// the closed form (shadow `|u - pi| < acos(-cos(limit) / cos(beta))`, yaw at
/// `+R` from the nominal yaw at the entry), its zero bisected in beta.
#[test]
fn gps_iia_shadow_exits_on_the_nominal_yaw_end_the_turn_at_the_exit() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(11)).unwrap();
    let rate = gps_iia_default_yaw_rate_rad_s(11).unwrap();
    let n = Circular {
        a: GPS_A,
        beta: 0.0,
        u0: 0.0,
    }
    .n();
    let t_midnight = 1.0e8;
    let half_span = |beta: f64| libm::acos(-libm::cos(GPS_SHADOW_LIMIT_RAD) / libm::cos(beta));
    let exit_gap = |beta: f64| {
        let (y, tb) = (half_span(beta), libm::tan(beta));
        let psi_in = libm::atan2(-tb, libm::sin(-y));
        let psi_out = libm::atan2(-tb, libm::sin(y));
        let x = psi_out - (psi_in + rate * 2.0 * y / n);
        x - TAU * libm::round(x / TAU)
    };
    let (mut lo, mut hi) = (deg(8.0), deg(8.0));
    while hi < deg(11.0) {
        lo = hi;
        hi = lo + deg(0.01);
        let (a, b) = (exit_gap(lo), exit_gap(hi));
        if a * b <= 0.0 && (a - b).abs() < 1.0 {
            break;
        }
    }
    assert!(exit_gap(lo) * exit_gap(hi) <= 0.0, "no zero of D");
    let slope = (exit_gap(hi) - exit_gap(lo)) / (hi - lo);
    while hi - lo > 1e-15 {
        let mid = 0.5 * (lo + hi);
        if (exit_gap(mid) > 0.0) == (exit_gap(lo) > 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let beta_zero = 0.5 * (lo + hi);
    let step = 2.0e-13 / slope.abs();
    let mut checked = 0;
    for k in -1500..=1500 {
        let beta = beta_zero + f64::from(k) * step;
        assert!(exit_gap(beta).abs() < 1e-9);
        let orbit = Circular {
            a: GPS_A,
            beta,
            u0: PI - n * t_midnight,
        };
        let trajectory = orbit.trajectory();
        let t_out = t_midnight + half_span(beta) / n;
        let inside = gnss_attitude(&law, &trajectory, t_out - 1.0).unwrap();
        assert_eq!(inside.mode, AttitudeMode::Shadow, "beta {beta:e}");
        for dt in [1.0, 30.0] {
            let attitude = gnss_attitude(&law, &trajectory, t_out + dt).unwrap();
            assert_eq!(
                attitude.mode,
                AttitudeMode::Nominal,
                "beta {beta:e} (D {:e}) t - exit {dt}",
                exit_gap(beta)
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 6002);
}

/// The noon turn of a GPS law on a circular orbit of orbital rate `n` with a
/// fixed Sun (Kouba 2009; `eclips.f`), written out: the zone
/// `|x| < kouba_half_span`, the yaw from the nominal yaw at the zone start at
/// `phi_dot`, then on until the gap to the nominal yaw
/// `psi = atan2(-tan(beta), sin(mu))` (continuous for beta of one sign) closes:
/// `g(t) = g0 + d (psi(t) - psi(t_e)) - R (t - t_e)`, `d = sign(phi_dot)`,
/// with `g0` the gap at the zone end taken in `[0, 2 pi)` for a turn reversed
/// by the bias and otherwise wrapped to `[-pi / 2, 3 pi / 2)` and clamped at
/// 0. Its first zero, on a 0.01 s scan bisected to 1e-9 s, ends the turn, at
/// most 1800 s after the zone.
#[derive(Clone, Copy, Debug)]
struct ClosedNoonTurn {
    start: f64,
    zone_end: f64,
    end: f64,
    psi0: f64,
    phi_dot: f64,
}

impl ClosedNoonTurn {
    fn new(n: f64, beta: f64, t_noon: f64, phi_dot: f64, reversed: bool) -> Self {
        let rate = phi_dot.abs();
        let t_b = libm::tan(beta);
        let x_s = kouba_half_span(beta, n, rate);
        let (start, zone_end) = (t_noon - x_s / n, t_noon + x_s / n);
        let nominal = |t: f64| libm::atan2(-t_b, libm::sin(n * (t - t_noon) + PI));
        let psi0 = nominal(start);
        let d = phi_dot.signum();
        let ahead = d * (nominal(zone_end) - (psi0 + phi_dot * (zone_end - start)));
        let g0 = if reversed {
            ahead.rem_euclid(TAU)
        } else {
            kouba_trailing_gap(ahead)
        };
        let gap = |t: f64| g0 + d * (nominal(t) - nominal(zone_end)) - rate * (t - zone_end);
        let limit = zone_end + 1800.0;
        let mut end = limit;
        let mut t = zone_end;
        while t < limit {
            let next = (t + 0.01).min(limit);
            if gap(next) <= 0.0 {
                let (mut lo, mut hi) = (t, next);
                while hi - lo > 1e-9 {
                    let mid = 0.5 * (lo + hi);
                    if gap(mid) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                end = hi;
                break;
            }
            t = next;
        }
        Self {
            start,
            zone_end,
            end,
            psi0,
            phi_dot,
        }
    }

    fn yaw(&self, t: f64) -> f64 {
        self.psi0 + self.phi_dot * (t - self.start)
    }

    /// Whether `t` is in the turn: from the start to the meeting, at most
    /// 1800 s after the zone (the margin's last instant still in the turn).
    fn contains(&self, t: f64) -> bool {
        let limit = self.zone_end + 1800.0;
        self.start <= t && (t < self.end || (self.end == limit && t <= limit))
    }
}

/// A GPS noon turn reversed by the yaw bias turns the way the nominal yaw
/// moves around midnight, `+sign(beta)` (Kouba 2009), so on a fast orbit
/// (radius 4500 km, period 3004.5 s) whose midnight falls within the 1800 s
/// after the noon zone, the gap to the nominal yaw closes on the way to
/// midnight, can reach zero just before the nominal yaw starts to move faster
/// than the turn, and opens again inside that zone: the turn ends at that
/// zero. GPS IIA with a custom bias of 0.5 rad (so beta from 10 to 25 deg is
/// within it), no shadow (a shadow limit just below 180 deg), maximum yaw
/// rates of 1.2 to 2 times the orbital rate, and at beta from 21 to 25 deg the
/// rates at which the gap at the midnight zone's start is 3e-3 rad below
/// zero, so that it is negative for only 36 to 42 s; against
/// [`ClosedNoonTurn`].
#[test]
fn gps_reversed_noon_turns_end_where_they_meet_the_nominal_yaw_before_midnight() {
    let a = 4_500_000.0;
    let mut just_before_midnight_zone = 0;
    for (beta_deg, rate_factor) in [
        (10.0, 1.2),
        (10.0, 1.55),
        (10.0, 2.0),
        (20.0, 1.2),
        (20.0, 1.55),
        (20.0, 2.0),
        (25.0, 1.55),
        (25.0, 2.0),
        (21.0, 1.589_628),
        (21.5, 1.603_825),
        (22.0, 1.618_537),
        (22.5, 1.633_854),
        (24.5, 1.704_101),
        (25.0, 1.725_24),
    ] {
        {
            let beta = deg(beta_deg);
            let orbit = Circular {
                a,
                beta,
                u0: -deg(60.0),
            };
            let n = orbit.n();
            let rate = rate_factor * n;
            let law = GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: rate,
                yaw_bias_rad: 0.5,
                shadow_limit_rad: PI - 1e-6,
            };
            let t_noon = orbit.time_at(0.0);
            let turn = ClosedNoonTurn::new(n, beta, t_noon, rate, true);
            let t_midnight = orbit.time_at(PI);
            // Kouba's zone around midnight at this rate, where the nominal yaw
            // moves at +sign(beta) faster than the turn.
            let zone_start = t_midnight - kouba_half_span(beta, n, rate) / n;
            if turn.end < zone_start && turn.end > zone_start - 30.0 {
                just_before_midnight_zone += 1;
            }
            let trajectory = orbit.trajectory();
            let mut checks = vec![turn.end - 1e-5, turn.end + 1e-5];
            let mut t = turn.start + 0.5;
            while t < turn.zone_end + 1900.0 {
                checks.push(t);
                t += 7.0;
            }
            for t in checks {
                let where_ = format!(
                    "beta {beta_deg} rate {rate_factor} n: t - noon {} (turn {:?})",
                    t - t_noon,
                    turn
                );
                let attitude =
                    gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
                if turn.contains(t) {
                    assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
                    let error = wrap_pi(attitude.yaw_rad - turn.yaw(t)).abs();
                    assert!(error < 1e-9, "{where_}: yaw error {error:e}");
                } else {
                    assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
                }
            }
        }
    }
    assert!(
        just_before_midnight_zone >= 4,
        "{just_before_midnight_zone} turns end within 30 s before the midnight zone"
    );
}

/// The Galileo IOV law's region (here GPS III, Montenbruck et al. 2026) begins
/// where `|beta|` falls below 5.8 deg, 10 s before the requested instant; the
/// region's `|S_x| < sin(15 deg)` holds from about an hour earlier until 10 s
/// after it. The beginning is solved from the trajectory before the requested
/// instant: the trajectory covers only the instants from one search step
/// (`pi / 8` of orbit angle) before a full turn of orbit angle before the
/// requested instant (where the law takes the orbit's eccentricity and the
/// motion of its plane) up to the requested instant. The yaw is eqs. (13), (17) and (18) with `G` the sign of
/// `s_y` where `|beta|` crossed 5.8 deg.
#[test]
fn galileo_iov_region_beginning_is_solved_before_the_requested_instant() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIII, None).unwrap();
    let n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    let rho = 1e-6;
    let beta_at = |t: f64| deg(5.8) - rho * (t - 10.0);
    let u_exit = libm::asin(libm::sin(deg(15.0)) / libm::cos(beta_at(30.0)));
    let m0 = u_exit - 30.0 * n;
    let orbit = Kepler {
        a: GPS_A,
        e: 0.0,
        beta: beta_at(0.0),
        beta_rate: -rho,
        omega: 0.0,
        m0,
    };
    let t_s = 20.0;
    let t_first = t_s - (TAU + FRAC_PI_8_TEST) / n * 1.01;
    let trajectory = move |t: f64| (t_first..=t_s).contains(&t).then(|| orbit.sample(t));
    let attitude = gnss_attitude(&law, &trajectory, t_s).unwrap_or_else(|e| panic!("{e}"));
    let mu = orbit.argument_from_noon(t_s) + PI;
    let (inside, yaw) = gps_iii_paper_yaw(beta_at(t_s), mu, -libm::sin(beta_at(10.0)));
    assert!(inside);
    assert_eq!(attitude.mode, AttitudeMode::NoonTurn);
    assert!(
        wrap_pi(attitude.yaw_rad - yaw).abs() < 1e-9,
        "{} vs {yaw}",
        attitude.yaw_rad
    );
}

const FRAC_PI_8_TEST: f64 = std::f64::consts::FRAC_PI_8;

/// A GPS IIR noon turn on a circular orbit of period 80 s (radius 500 km, the
/// orbital rate set, not Keplerian) at beta = 0.5 deg: Kouba's zone around
/// the noon at t = 10 s spans about 11 s, and the next noon is 80 s later.
/// Inside the zone the yaw is linear from the nominal yaw at its start.
#[test]
fn gps_turns_on_an_orbit_shorter_than_two_minutes_use_their_own_noon() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let rate = GPS_IIR_DEFAULT_YAW_RATE_RAD_S;
    let (a, n, beta) = (500_000.0, TAU / 80.0, deg(0.5));
    let (p, q, _) = Circular { a, beta, u0: 0.0 }.frame();
    let trajectory = move |t: f64| {
        let u = n * (t - 10.0);
        let (su, cu) = (libm::sin(u), libm::cos(u));
        Some(AttitudeSample {
            position_m: std::array::from_fn(|k| a * (cu * p[k] + su * q[k])),
            velocity_m_s: std::array::from_fn(|k| a * n * (-su * p[k] + cu * q[k])),
            sun_position_m: [AU, 0.0, 0.0],
        })
    };
    let x_s = kouba_half_span(beta, n, rate);
    let start = 10.0 - x_s / n;
    assert!(start > 4.0 && 10.0 + x_s / n < 16.0, "zone {x_s}");
    let psi0 = libm::atan2(-libm::tan(beta), libm::sin(x_s));
    for t in [6.0, 10.0, 10.5, 12.0, 15.0] {
        let attitude = gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("t {t}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "t {t}");
        let expected = psi0 - rate * (t - start);
        assert!(
            wrap_pi(attitude.yaw_rad - expected).abs() < 1e-9,
            "t {t}: {} vs {expected}",
            attitude.yaw_rad
        );
    }
}

/// BeiDou-2 IGSO (Orekit `BeidouMeo`): the mode is set at the last orbit
/// quadrature, orbit normal there if `|beta|` was at most 3.9 deg while it
/// rises or at most 4.1 deg while it falls. On orbits of eccentricity 0.15 and
/// 0.3, near perigee (where the orbital rate is up to 1.95 times the mean
/// motion), with beta drifting 0.8 deg per orbit so that `|beta|` now and at
/// the quadrature lie on either side of a limit; the quadrature from the
/// Keplerian closed form (argument from noon -90 deg).
#[test]
fn beidou2_switch_reads_the_last_orbit_on_eccentric_orbits() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::Beidou2Igso, None).unwrap();
    let mut decided_by_the_quadrature = 0;
    for e in [0.15, 0.3] {
        for omega_deg in [-100.0, 80.0] {
            for (beta_deg, drift_deg) in [(4.55, 0.8), (-4.55, -0.8), (3.4, -0.8), (-3.4, 0.8)] {
                let probe = Kepler {
                    a: 42_164_200.0,
                    e,
                    beta: 0.0,
                    beta_rate: 0.0,
                    omega: deg(omega_deg),
                    m0: 0.0,
                };
                let period = probe.period();
                let orbit = Kepler {
                    beta: deg(beta_deg),
                    beta_rate: deg(drift_deg) / period,
                    ..probe
                };
                for t in [0.0, 0.02 * period, -0.02 * period] {
                    let beta = orbit.beta + orbit.beta_rate * t;
                    let t_q = orbit.time_at(-FRAC_PI_2, t - period);
                    assert!(t_q <= t && t - t_q < period);
                    let beta_q = orbit.beta + orbit.beta_rate * t_q;
                    let falling = beta * orbit.beta_rate < 0.0;
                    let limit = if falling {
                        BEIDOU2_YS_TO_ON_BETA_RAD
                    } else {
                        BEIDOU2_ON_TO_YS_BETA_RAD
                    };
                    let expected = if beta_q.abs() <= limit {
                        AttitudeMode::OrbitNormal
                    } else {
                        AttitudeMode::Nominal
                    };
                    if (beta.abs() <= limit) != (beta_q.abs() <= limit) {
                        decided_by_the_quadrature += 1;
                    }
                    let attitude = gnss_attitude(&law, &orbit.trajectory(), t)
                        .unwrap_or_else(|err| panic!("{err}"));
                    assert_eq!(
                        attitude.mode, expected,
                        "e {e} omega {omega_deg} beta {beta_deg} drift {drift_deg} t {t}: beta {} at the quadrature {}",
                        beta.to_degrees(),
                        beta_q.to_degrees()
                    );
                }
            }
        }
    }
    assert!(
        decided_by_the_quadrature >= 16,
        "{decided_by_the_quadrature}"
    );
}

/// On an orbit of eccentricity 0.3 with perigee 90 deg after noon the orbital
/// rate grows through noon, so Kouba's nominal yaw rate
/// `mu_rate T cos(x) / (sin^2(x) + T^2)` peaks after noon and exceeds the
/// maximum yaw rate there although it does not at noon: the zone, and the
/// turn, lie wholly after noon. GPS IIR at a custom rate equal to the mean
/// motion with beta = 50 deg, and GPS IIF at its published rate with beta
/// 1.0002 times the noon limit `atan(mu_rate / R)`. Zone from the Keplerian
/// closed form, scanned between the quadratures and its edges bisected; the
/// yaw from the nominal yaw at its start at `-sign(beta) R`.
#[test]
fn gps_noon_zones_away_from_noon_on_eccentric_orbits_are_turns() {
    let n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    let probe = Kepler {
        a: GPS_A,
        e: 0.3,
        beta: 0.0,
        beta_rate: 0.0,
        omega: deg(90.0),
        m0: 0.0,
    };
    let mu_rate = |t: f64| {
        let s = probe.sample(t);
        norm3(cross3(s.position_m, s.velocity_m_s)) / dot3(s.position_m, s.position_m)
    };
    let t_noon = probe.time_at(0.0, 0.0);
    let iif_rate = GPS_IIF_DEFAULT_YAW_RATE_RAD_S;
    let iif_beta = 1.0002 * libm::atan(mu_rate(t_noon) / iif_rate);
    for (law, rate, beta) in [
        (GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s: n }, n, deg(50.0)),
        (
            GnssAttitudeLaw::for_block(GnssBlock::GpsIIF, None).unwrap(),
            iif_rate,
            iif_beta,
        ),
    ] {
        let orbit = Kepler { beta, ..probe };
        let t_b = libm::tan(beta);
        let in_zone = |t: f64| {
            let x = -orbit.argument_from_noon(t);
            let s = libm::sin(x);
            mu_rate(t) * t_b * libm::cos(x) / (s * s + t_b * t_b) > rate
        };
        assert!(!in_zone(t_noon), "{law:?}: zone at noon");
        let (t_lo, t_hi) = (
            orbit.time_at(-FRAC_PI_2, t_noon - orbit.period()),
            orbit.time_at(FRAC_PI_2, t_noon),
        );
        let mut inside = Vec::new();
        let mut t = t_lo;
        while t < t_hi {
            if in_zone(t) {
                inside.push(t);
            }
            t += 1.0;
        }
        let (first, last) = (inside[0], *inside.last().unwrap());
        assert!(first > t_noon, "{law:?}: zone from {first}");
        let start = change(in_zone, first - 1.0, first);
        let zone_end = change(in_zone, last, last + 1.0);
        let psi0 = libm::atan2(-t_b, libm::sin(orbit.argument_from_noon(start) - PI));
        let trajectory = orbit.trajectory();
        for f in [0.01, 0.3, 0.6, 0.99] {
            let t = start + f * (zone_end - start);
            let where_ = format!("{law:?} beta {beta}: zone {start}..{zone_end} t {t}");
            let attitude =
                gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
            let expected = psi0 - rate * (t - start);
            let error = wrap_pi(attitude.yaw_rad - expected).abs();
            assert!(error < 1e-8, "{where_}: yaw error {error:e}");
        }
        let before = gnss_attitude(&law, &trajectory, start - 1e-3).unwrap();
        assert_eq!(before.mode, AttitudeMode::Nominal, "{law:?}");
    }
}

/// A GPS IIF noon turn at a custom maximum yaw rate of 0.07 times the orbital
/// rate with a custom bias of 40 deg (beta = 30 deg within it, so reversed),
/// on a fast orbit (radius 4500 km, period 3004.5 s): the turn has not met
/// the nominal yaw 1800 s after its zone, past the third quadrature after its
/// noon and into the next noon half orbit, until the next noon turn starts.
/// Every instant from the turn start to past the next turn's start against
/// [`ClosedNoonTurn`], the later start ruling where both apply.
#[test]
fn gps_turns_lasting_into_the_next_half_orbit_of_their_kind_continue() {
    let a = 4_500_000.0;
    let beta = deg(30.0);
    let orbit = Circular {
        a,
        beta,
        u0: -deg(60.0),
    };
    let n = orbit.n();
    let rate = 0.07 * n;
    let law = GnssAttitudeLaw::GpsIIF {
        yaw_rate_rad_s: rate,
        yaw_bias_rad: deg(40.0),
        shadow_limit_rad: PI - 1e-6,
    };
    let period = TAU / n;
    let t_noon = orbit.time_at(0.0);
    let first = ClosedNoonTurn::new(n, beta, t_noon, rate, true);
    let second = ClosedNoonTurn::new(n, beta, t_noon + period, rate, true);
    let third_quadrature = orbit.time_at(-FRAC_PI_2) + period;
    assert!(
        first.end > third_quadrature && first.end > second.start,
        "{first:?} {second:?}"
    );
    let trajectory = orbit.trajectory();
    let mut checks = vec![t_noon + 2300.0, second.start - 1e-3, second.start + 1e-3];
    let mut t = first.start + 0.5;
    while t < second.start + 600.0 {
        checks.push(t);
        t += 13.0;
    }
    let mut past_quadrature = 0;
    for t in checks {
        let turn = if second.start <= t { second } else { first };
        let where_ = format!("t - noon {} ({turn:?})", t - t_noon);
        let attitude =
            gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
        if turn.contains(t) {
            assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_}");
            let error = wrap_pi(attitude.yaw_rad - turn.yaw(t)).abs();
            assert!(error < 1e-9, "{where_}: yaw error {error:e}");
            if t > third_quadrature && turn.start == first.start {
                past_quadrature += 1;
            }
        } else {
            assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_}");
        }
    }
    assert!(past_quadrature >= 5, "{past_quadrature}");
}

/// The geometry the papers define, formed from a sample without the laws'
/// code: `beta = asin(s . n)`, the in-plane argument from noon `u` (from the
/// Sun's projection on the orbit plane, in the direction of motion), the
/// orbital rate `|r x v| / |r|^2` and the nominal yaw
/// `atan2(-tan(beta), sin(mu))`, `mu = u + pi` (Kouba 2009).
#[derive(Clone, Copy, Debug)]
struct PaperGeometry {
    beta: f64,
    from_noon: f64,
    mu_rate: f64,
    nominal: f64,
}

fn paper_geometry(s: &AttitudeSample) -> PaperGeometry {
    let unit = |v: [f64; 3]| {
        let k = 1.0 / norm3(v);
        [v[0] * k, v[1] * k, v[2] * k]
    };
    let h = cross3(s.position_m, s.velocity_m_s);
    let n = unit(h);
    let sun = unit(s.sun_position_m);
    let e_r = unit(s.position_m);
    let sn = dot3(sun, n);
    let beta = libm::asin(sn);
    let plane: [f64; 3] = std::array::from_fn(|k| sun[k] - sn * n[k]);
    let from_noon = libm::atan2(dot3(cross3(plane, e_r), n), dot3(plane, e_r));
    PaperGeometry {
        beta,
        from_noon,
        mu_rate: norm3(h) / dot3(s.position_m, s.position_m),
        nominal: libm::atan2(-libm::tan(beta), libm::sin(from_noon + PI)),
    }
}

/// Kouba's (2009) nominal yaw rate `mu_rate T cos(x) / (sin^2(x) + T^2)`,
/// `T = |tan(beta)|`, `x` the in-plane angle to noon, over the limit `rate`,
/// minus one: positive inside the noon zone.
fn kouba_noon_excess(g: &PaperGeometry, rate: f64) -> f64 {
    let tb = libm::tan(g.beta).abs();
    let s = libm::sin(g.from_noon);
    g.mu_rate * tb * libm::cos(g.from_noon) / (s * s + tb * tb) / rate - 1.0
}

/// The gap to the nominal yaw at the end of a zone where the nominal yaw ran
/// ahead of a turn at the maximum rate (Kouba 2009): `ahead` is the nominal
/// minus the linear yaw, measured the way the turn goes, which the excess
/// rate keeps within `[0, pi)` modulo 2 pi; a value in `[3 pi / 2, 2 pi)` is a
/// nominal yaw a rounding behind the turn, which has then met it.
fn kouba_trailing_gap(ahead: f64) -> f64 {
    let wrapped = ahead.rem_euclid(TAU);
    if wrapped >= 1.5 * PI {
        0.0
    } else {
        wrapped
    }
}

/// The first instant after `from` at which
/// `gap0 + d (psi(t) - psi(from)) - rate (t - from)` reaches zero, with `psi`
/// the nominal yaw followed continuously in steps of at most 0.05 s, refined
/// where it moves by more than 0.2 rad in a step, and the crossing bisected
/// to 1e-10 s; `limit` when it does not before then.
fn follow_gap(
    nominal: &dyn Fn(f64) -> f64,
    from: f64,
    limit: f64,
    gap0: f64,
    d: f64,
    rate: f64,
) -> f64 {
    fn advance(
        nominal: &dyn Fn(f64) -> f64,
        a: f64,
        b: f64,
        state: &mut (f64, f64),
        gap: &dyn Fn(f64, f64) -> f64,
        depth: u32,
    ) -> Option<f64> {
        let (psi_a, unwrapped_a) = *state;
        let psi_b = nominal(b);
        let delta = wrap_pi(psi_b - psi_a);
        if delta.abs() > 0.2 && depth < 40 && b - a > 1e-12 {
            let pieces = 8;
            let mut lo = a;
            for k in 1..=pieces {
                let hi = if k == pieces {
                    b
                } else {
                    a + (b - a) * f64::from(k) / f64::from(pieces)
                };
                if let Some(met) = advance(nominal, lo, hi, state, gap, depth + 1) {
                    return Some(met);
                }
                lo = hi;
            }
            return None;
        }
        let unwrapped_b = unwrapped_a + delta;
        if gap(unwrapped_b, b) <= 0.0 {
            let (mut lo, mut hi) = (a, b);
            while hi - lo > 1e-10 {
                let mid = 0.5 * (lo + hi);
                let u = unwrapped_a + wrap_pi(nominal(mid) - psi_a);
                if gap(u, mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some(hi);
        }
        *state = (psi_b, unwrapped_b);
        None
    }
    if gap0 <= 0.0 {
        return from;
    }
    let gap = |unwrapped: f64, t: f64| gap0 + d * unwrapped - rate * (t - from);
    let mut state = (nominal(from), 0.0);
    let mut t = from;
    while t < limit {
        let next = (t + 0.05).min(limit);
        if let Some(met) = advance(nominal, t, next, &mut state, &gap, 0) {
            return met;
        }
        t = next;
    }
    limit
}

/// A GPS noon turn solved on a trajectory by fine scans, as Kouba (2009) and
/// `eclips.f` define it: the zone from the first to the last instant at which
/// Kouba's nominal yaw rate exceeds R ([`kouba_noon_excess`] on a scan of
/// `[t_noon - span, t_noon + span]` at `scan_step`, edges bisected; `None`
/// where there is none or it reaches the ends of the scan), the turn
/// from the nominal yaw at the zone start at `-sign(beta) R`, or at
/// `sign(b) R` onward for beta of the yaw bias's sign within it, until the
/// gap to the continuous nominal yaw closes ([`follow_gap`]), at most 1800 s
/// after the zone.
#[derive(Clone, Copy, Debug)]
struct ReferenceTurn {
    start: f64,
    zone_end: f64,
    end: f64,
    psi0: f64,
    phi_dot: f64,
}

impl ReferenceTurn {
    fn gps_noon(
        sample: &dyn Fn(f64) -> AttitudeSample,
        t_noon: f64,
        span: f64,
        scan_step: f64,
        rate: f64,
        bias: f64,
    ) -> Option<Self> {
        let inside = |t: f64| kouba_noon_excess(&paper_geometry(&sample(t)), rate) > 0.0;
        let (mut first, mut last) = (None, None);
        let points = libm::ceil(2.0 * span / scan_step) as i64;
        for k in 0..=points {
            let t = t_noon - span + k as f64 * scan_step;
            if inside(t) {
                first.get_or_insert(t);
                last = Some(t);
            }
        }
        let (first, last) = (first?, last?);
        if first == t_noon - span || last >= t_noon + span - scan_step {
            // A zone reaching the ends of the scan: not one this reference
            // follows.
            return None;
        }
        let start = change(inside, first - scan_step, first);
        let zone_end = change(inside, last, last + scan_step);
        let at_start = paper_geometry(&sample(start));
        let (psi0, beta) = (at_start.nominal, at_start.beta);
        let reversed =
            bias != 0.0 && (beta == 0.0 || (beta * bias > 0.0 && beta.abs() <= bias.abs()));
        let phi_dot = if reversed {
            rate.copysign(bias)
        } else {
            -rate.copysign(beta)
        };
        let d = phi_dot.signum();
        let nominal = |t: f64| paper_geometry(&sample(t)).nominal;
        let ahead = d * (nominal(zone_end) - (psi0 + phi_dot * (zone_end - start)));
        let gap0 = if reversed && beta != 0.0 {
            ahead.rem_euclid(TAU)
        } else {
            kouba_trailing_gap(ahead)
        };
        let end = follow_gap(&nominal, zone_end, zone_end + 1800.0, gap0, d, rate);
        Some(Self {
            start,
            zone_end,
            end,
            psi0,
            phi_dot,
        })
    }

    fn yaw(&self, t: f64) -> f64 {
        self.psi0 + self.phi_dot * (t - self.start)
    }

    fn contains(&self, t: f64) -> bool {
        let limit = self.zone_end + 1800.0;
        self.start <= t && (t < self.end || (self.end == limit && t <= limit))
    }

    /// Instants to check: a grid over the turn and around it, and probes
    /// 1e-5 s either side of the start, the zone end and the meeting, none
    /// within 1e-7 s of a boundary.
    fn checks(&self, grid: f64) -> Vec<f64> {
        let mut checks = Vec::new();
        let boundaries = [self.start, self.zone_end, self.end];
        for b in boundaries {
            checks.extend([b - 1e-5, b + 1e-5]);
        }
        let mut t = self.start - 3.0 * grid;
        while t < self.end + 3.0 * grid {
            if boundaries.iter().all(|b| (t - b).abs() > 1e-7) {
                checks.push(t);
            }
            t += grid;
        }
        let zone = self.zone_end - self.start;
        for f in [0.02, 0.3, 0.5, 0.7, 0.98] {
            checks.push(self.start + f * zone);
        }
        checks
    }
}

/// Asserts the law's attitude at each instant against a reference noon turn:
/// the turn's mode and linear yaw within it, nominal outside.
fn assert_noon_turn(
    law: &GnssAttitudeLaw,
    trajectory: &dyn Fn(f64) -> Option<AttitudeSample>,
    turn: &ReferenceTurn,
    checks: &[f64],
    where_: &str,
) {
    for &t in checks {
        let here = format!("{where_}: t - start {} ({turn:?})", t - turn.start);
        let attitude = gnss_attitude(law, trajectory, t).unwrap_or_else(|e| panic!("{here}: {e}"));
        if turn.contains(t) {
            assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{here}");
            let error = wrap_pi(attitude.yaw_rad - turn.yaw(t)).abs();
            assert!(
                error < 1e-9,
                "{here}: yaw {} vs {}",
                attitude.yaw_rad,
                turn.yaw(t)
            );
        } else {
            assert_eq!(attitude.mode, AttitudeMode::Nominal, "{here}");
        }
    }
}

/// Beta passes through zero once a year per orbit plane, in the eclipse
/// season, at about 2e-7 rad/s (the Sun's motion), so on some day it changes
/// sign inside or next to a GPS noon zone. Kouba's nominal yaw rate holds
/// beta constant through each instant, so it falls to zero at the instant
/// beta does, also where the nominal yaw keeps turning through the Sun line:
/// the zone runs from the first instant Kouba's rate exceeds R to the last,
/// the turn's direction and start yaw come from the beta at its start, and the
/// turn runs until it meets the continuous nominal yaw ([`ReferenceTurn`]).
/// GPS IIR, IIF and IIA (bias +0.5 deg, 0.12 deg/s) on a circular orbit with
/// `beta = 2e-7 rad/s (t - t0)`, `t0` at noon and from 0.02 s to 200 s either
/// side of it, with noon at t = 0, at 1000 s and at -350 s.
#[test]
fn gps_noon_turns_through_beta_zero_follow_the_whole_zone() {
    let n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    let iia_rate = 0.12_f64.to_radians();
    let laws = [
        (
            GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap(),
            GPS_IIR_DEFAULT_YAW_RATE_RAD_S,
            0.0,
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::GpsIIF, None).unwrap(),
            GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
            GPS_IIF_DEFAULT_YAW_BIAS_RAD,
        ),
        (
            GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: iia_rate,
                yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            },
            iia_rate,
            GPS_IIA_DEFAULT_YAW_BIAS_RAD,
        ),
    ];
    let beta_rate = 2e-7;
    let mut checked = 0;
    for (law, rate, bias) in laws {
        for t_noon in [0.0, 1000.0, -350.0] {
            for offset in [
                0.0, 0.02, -0.02, 0.15, -0.15, 0.4, -0.4, 3.0, -3.0, 60.0, -60.0, 200.0, -200.0,
            ] {
                let t0 = t_noon + offset;
                let orbit = Drifting {
                    a: GPS_A,
                    beta0: -beta_rate * t0,
                    rate: beta_rate,
                    u0: -n * t_noon,
                };
                let sample = move |t: f64| orbit.sample(t);
                let trajectory = move |t: f64| Some(orbit.sample(t));
                let turn = ReferenceTurn::gps_noon(&sample, t_noon, 40.0, 1e-3, rate, bias)
                    .unwrap_or_else(|| panic!("{law:?} offset {offset}: no zone"));
                let where_ = format!("{law:?} noon {t_noon} beta zero at noon {offset:+}");
                let checks = turn.checks(37.0);
                checked += checks.len();
                assert_noon_turn(&law, &trajectory, &turn, &checks, &where_);
            }
        }
    }
    assert!(checked > 4000, "{checked}");
}

/// The time at which a Keplerian orbit's argument from noon is 0, moved to
/// `t_noon` by its mean anomaly at t = 0.
fn kepler_with_noon_at(orbit: Kepler, t_noon: f64) -> Kepler {
    let nu = -orbit.omega;
    let ea = 2.0
        * libm::atan2(
            (1.0 - orbit.e).sqrt() * libm::sin(0.5 * nu),
            (1.0 + orbit.e).sqrt() * libm::cos(0.5 * nu),
        );
    let m_noon = ea - orbit.e * libm::sin(ea);
    Kepler {
        m0: m_noon - orbit.n() * t_noon,
        ..orbit
    }
}

/// The search for a zone that does not hold at its centre works in time
/// itself, so the half orbit may lie on either side of t = 0 or across it in
/// any position. Three zones off noon (a GPS IIR noon zone at a custom rate of
/// the mean motion on an orbit of eccentricity 0.3, a published GPS IIF zone
/// on the same orbit just past its existence at noon, and a GLONASS-M zone
/// at a custom rate on an orbit of eccentricity 0.1), each with noon moved so
/// that t = 0 lies before the zone, at its edges and inside it in several
/// places, and across the half orbit; the zone from a fine scan of the papers'
/// conditions, edges bisected, and the law's mode at instants inside and just
/// outside.
#[test]
fn zones_off_noon_are_found_wherever_zero_lies_in_the_half_orbit() {
    let gps_n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    // GLONASS-M at beta 20 deg on an orbit of eccentricity 0.1 with perigee
    // a quarter turn after noon, at the rate that puts noon 0.2 % short of
    // the span's condition `tan|beta| < mu_rate / R`: the orbital rate rises
    // through noon, so the span holds from about 75 s to 440 s after it.
    let glonass_orbit = Kepler {
        a: GLONASS_A,
        e: 0.1,
        beta: deg(20.0),
        beta_rate: 0.0,
        omega: deg(90.0),
        m0: 0.0,
    };
    let noon_rate = glonass_orbit.n() / libm::pow(1.0 - 0.1 * 0.1, 1.5);
    let glonass_rate = noon_rate * 1.002 / libm::tan(deg(20.0));
    let cases = [
        (
            GnssAttitudeLaw::GpsIIR {
                yaw_rate_rad_s: gps_n,
            },
            gps_n,
            Kepler {
                a: GPS_A,
                e: 0.3,
                beta: deg(50.0),
                beta_rate: 0.0,
                omega: deg(90.0),
                m0: 0.0,
            },
            false,
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::GpsIIF, None).unwrap(),
            GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
            Kepler {
                a: GPS_A,
                e: 0.3,
                beta: deg(5.002_721),
                beta_rate: 0.0,
                omega: deg(90.0),
                m0: 0.0,
            },
            false,
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: glonass_rate,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            glonass_rate,
            glonass_orbit,
            true,
        ),
    ];
    for (law, rate, probe, glonass) in cases {
        let inside_at = |orbit: &Kepler, t: f64| {
            let g = paper_geometry(&orbit.sample(t));
            if glonass {
                // Dilssner et al. (2011): inside while the turn at R over
                // [-d, d] does not reach the nominal yaw's half swing
                // |phi(d) - phi(-d)| / 2, which for d in [0, pi] is
                // atan(sin(d) / |tan(beta)|).
                let k = g.mu_rate / rate;
                let d = g.from_noon.abs();
                let tb = libm::tan(g.beta).abs();
                if d == 0.0 {
                    tb < k
                } else {
                    k * libm::atan2(libm::sin(d), tb) > d
                }
            } else {
                kouba_noon_excess(&g, rate) > 0.0
            }
        };
        // The zone relative to noon, from noon at t = 0.
        let base = kepler_with_noon_at(probe, 0.0);
        let quarter = 0.25 * base.period();
        let (mut first, mut last) = (None, None);
        let step = 0.01;
        let mut t = -quarter;
        while t < quarter {
            if inside_at(&base, t) {
                first.get_or_insert(t);
                last = Some(t);
            }
            t += step;
        }
        let (first, last) = (first.unwrap(), last.unwrap());
        assert!(!inside_at(&base, 0.0), "{law:?}: zone at noon");
        let start = change(|t| inside_at(&base, t), first - step, first);
        let end = change(|t| inside_at(&base, t), last, last + step);
        let width = end - start;
        let mut noons = vec![
            -200.0,
            -2.0,
            -0.05,
            0.0,
            0.05,
            2.0,
            200.0,
            0.4 * quarter,
            -1.7 * quarter,
        ];
        for f in [-0.5, -0.01, 0.0, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0, 1.01, 1.5] {
            noons.push(-(start + f * width));
        }
        for t_noon in noons {
            let orbit = kepler_with_noon_at(probe, t_noon);
            let trajectory = orbit.trajectory();
            let (zone_start, zone_end) = (t_noon + start, t_noon + end);
            let mut probes = vec![zone_start - 1e-3 * width];
            for f in [0.01, 0.25, 0.5, 0.75, 0.99] {
                probes.push(zone_start + f * width);
            }
            for t in probes {
                let where_ = format!("{law:?} noon {t_noon} zone {zone_start}..{zone_end} t {t}");
                let attitude =
                    gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{where_}: {e}"));
                let expected = if t >= zone_start && t <= zone_end {
                    AttitudeMode::NoonTurn
                } else {
                    AttitudeMode::Nominal
                };
                assert_eq!(attitude.mode, expected, "{where_}");
            }
        }
    }
}

/// Every instant lies in a noon or a midnight half orbit, also at the last
/// representable argument from noon before a quadrature (where the difference
/// from the other quadrature rounds to pi), at the quadratures, the centres
/// and their neighbours: GPS IIA, IIR, IIF and GLONASS-M at beta = 30 deg
/// (nominal yaw steering), BeiDou-2 MEO inside the 3.9 to 4.1 deg switch band,
/// and the Galileo IOV law with custom limits of 80 deg and 20 deg at beta
/// just below 20 deg (inside its region at the quadratures) all give an
/// attitude.
#[test]
fn every_instant_lies_in_a_half_orbit() {
    let targets = [
        FRAC_PI_2.next_down(),
        FRAC_PI_2,
        FRAC_PI_2.next_up(),
        -FRAC_PI_2.next_down(),
        -FRAC_PI_2,
        -FRAC_PI_2.next_up(),
        PI,
        PI.next_down(),
        -PI.next_down(),
        0.0,
        1e-300,
        -1e-300,
    ];
    let mut hits = 0;
    for target in targets {
        // An orbit whose argument from noon, as the laws form it at t = 0,
        // is the target itself.
        let mut found = None;
        let mut u0 = target;
        for _ in 0..16 {
            let orbit = Circular {
                a: GPS_A,
                beta: deg(30.0),
                u0,
            };
            let at = Context::at(&orbit.trajectory(), 0.0).unwrap();
            let u = at.argument_from_noon();
            if u == target {
                found = Some(u0);
                break;
            }
            u0 = if u < target {
                u0.next_up()
            } else {
                u0.next_down()
            };
        }
        let Some(u0) = found else {
            continue;
        };
        hits += 1;
        for (block, beta) in [
            (GnssBlock::GpsIIR, 30.0),
            (GnssBlock::GpsIIF, 30.0),
            (GnssBlock::GlonassM, 30.0),
            (GnssBlock::Beidou2Meo, 4.0),
        ] {
            let law = GnssAttitudeLaw::for_block(block, None).unwrap();
            let orbit = Circular {
                a: if block == GnssBlock::Beidou2Meo {
                    MEO_A
                } else {
                    GPS_A
                },
                beta: deg(beta),
                u0,
            };
            let attitude = gnss_attitude(&law, &orbit.trajectory(), 0.0)
                .unwrap_or_else(|e| panic!("{block} argument {target:e}: {e}"));
            if beta == 30.0 {
                assert_eq!(attitude.mode, AttitudeMode::Nominal, "{block} {target:e}");
            }
        }
        let iia = GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(1)).unwrap();
        let orbit = Circular {
            a: GPS_A,
            beta: deg(30.0),
            u0,
        };
        let attitude = gnss_attitude(&iia, &orbit.trajectory(), 0.0)
            .unwrap_or_else(|e| panic!("GPS IIA argument {target:e}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::Nominal);
        // The region holds wherever |beta| < 20 deg (cos(beta) < sin(80 deg)),
        // here since beta fell through 20 deg half an orbit ago.
        let iov = GnssAttitudeLaw::GalileoIov {
            beta_x_rad: deg(80.0),
            beta_y_rad: deg(20.0),
        };
        let half = PI / (GM / (GALILEO_A * GALILEO_A * GALILEO_A)).sqrt();
        let orbit = Drifting {
            a: GALILEO_A,
            beta0: deg(19.95),
            rate: -deg(0.1) / half,
            u0,
        };
        let attitude = gnss_attitude(&iov, &move |t: f64| Some(orbit.sample(t)), 0.0)
            .unwrap_or_else(|e| panic!("Galileo IOV argument {target:e}: {e}"));
        assert_ne!(attitude.mode, AttitudeMode::Nominal);
    }
    assert!(hits >= 8, "{hits} arguments reached");
}

fn outside_domain(result: &Result<GnssAttitude, GnssAttitudeError>) -> bool {
    matches!(
        result,
        Err(GnssAttitudeError::OutsideValidatedDomain { .. })
    )
}

/// The searches are shown correct on orbits up to an eccentricity: 0.9 for
/// the stepping to quadratures and centres, 0.3 for the GPS turn zones, and
/// for the GLONASS-M noon zone 0.25 with `|beta|` below 30 deg, 0.2 below
/// 60 deg and 0.1 above. Beyond, a law refuses with
/// [`GnssAttitudeError::OutsideValidatedDomain`] instead of giving an
/// attitude: GPS IIR at a custom rate on an orbit of eccentricity 0.5 whose
/// noon half orbit holds two zones, GLONASS-M in its noon half orbit at
/// eccentricity 0.2 with beta 85 deg, at 0.35 with beta 30 deg and at the
/// published rate at 0.3 with beta 5 deg (where its noon zone can hold), and
/// the other searching laws at eccentricity 0.95 and
/// 0.98 (where a step from apogee turns by more than a full turn). Within the
/// bounds the same laws give attitudes.
#[test]
fn orbits_beyond_the_validated_eccentricity_are_refused() {
    let kepler = |a: f64, e: f64, omega_deg: f64, beta_deg: f64| Kepler {
        a,
        e,
        beta: deg(beta_deg),
        beta_rate: 0.0,
        omega: deg(omega_deg),
        m0: 0.0,
    };
    let refused = [
        (
            GnssAttitudeLaw::GpsIIR {
                yaw_rate_rad_s: 1.161_306_92e-5,
            },
            kepler(GPS_A, 0.5, 175.0, 80.0),
        ),
        (
            GnssAttitudeLaw::GpsIIR {
                yaw_rate_rad_s: 9.584_184e-6,
            },
            kepler(GPS_A, 0.5, 170.0, 80.0),
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: 9.651_111_69e-6,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            kepler(GLONASS_A, 0.2, 175.0, 85.0),
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: 1.564_187_09e-4,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            kepler(GLONASS_A, 0.35, 160.0, 30.0),
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            kepler(GLONASS_A, 0.3, 160.0, 5.0),
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap(),
            kepler(GPS_A, 0.98, 0.0, 10.0),
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::GalileoFoc, None).unwrap(),
            kepler(GALILEO_A, 0.98, 0.0, 1.0),
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::Beidou2Meo, None).unwrap(),
            kepler(MEO_A, 0.95, 30.0, 4.0),
        ),
        (
            GnssAttitudeLaw::for_block(
                GnssBlock::Beidou3Meo {
                    manufacturer: Some(BeidouManufacturer::Cast),
                },
                None,
            )
            .unwrap(),
            kepler(MEO_A, 0.95, 30.0, 1.0),
        ),
    ];
    for (law, orbit) in refused {
        let trajectory = orbit.trajectory();
        // Instants of the noon half orbit (the GLONASS-M bounds are the noon
        // zone's) and two of the midnight one.
        for u_deg in [-80.0, -30.0, 0.0, 20.0, 85.0, 120.0, -150.0] {
            let t = orbit.time_at(deg(u_deg), 0.0);
            if matches!(law, GnssAttitudeLaw::GlonassM { .. }) && u_deg.abs() > 90.0 {
                continue;
            }
            let result = gnss_attitude(&law, &trajectory, t);
            assert!(
                outside_domain(&result),
                "{law:?} e {} t {t}: {result:?}",
                orbit.e
            );
        }
    }
    // The Galileo IOV law searches only inside its region.
    let iov = GnssAttitudeLaw::for_block(GnssBlock::GalileoIov, None).unwrap();
    let orbit = kepler(GALILEO_A, 0.95, 40.0, 1.0);
    let t_noon = orbit.time_at(0.0, 0.0);
    assert!(outside_domain(&gnss_attitude(
        &iov,
        &orbit.trajectory(),
        t_noon
    )));
    let accepted = [
        (
            GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap(),
            kepler(GPS_A, 0.3, 175.0, 1.0),
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            kepler(GLONASS_A, 0.1, 175.0, 85.0),
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            kepler(GLONASS_A, 0.25, 160.0, 29.0),
        ),
        (
            GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            kepler(GLONASS_A, 0.2, 160.0, 55.0),
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::GalileoFoc, None).unwrap(),
            kepler(GALILEO_A, 0.9, 0.0, 1.0),
        ),
        (
            GnssAttitudeLaw::for_block(GnssBlock::Beidou2Meo, None).unwrap(),
            kepler(MEO_A, 0.9, 30.0, 4.0),
        ),
    ];
    for (law, orbit) in accepted {
        let trajectory = orbit.trajectory();
        for f in [0.0, 0.13, 0.31, 0.5, 0.77] {
            let t = f * orbit.period();
            gnss_attitude(&law, &trajectory, t)
                .unwrap_or_else(|e| panic!("{law:?} e {} t {t}: {e}", orbit.e));
        }
    }
}

/// A GPS noon turn whose gap to the nominal yaw is followed across beta's
/// change of sign after its zone, in its own half orbit, on a fast eccentric
/// orbit: GPS IIF at twice the mean motion (above the orbital rate where beta
/// changes sign), no shadow, radius 4500 km, eccentricity 0.25 with perigee
/// at the quadrature after noon, beta 8e-5 rad at noon and falling at
/// 2e-7 rad/s, through zero 400 s after noon; against [`ReferenceTurn`].
#[test]
fn gps_turn_gap_is_followed_across_a_change_of_beta_sign() {
    let probe = Kepler {
        a: 4_500_000.0,
        e: 0.25,
        beta: 0.0,
        beta_rate: 0.0,
        omega: deg(90.0),
        m0: 0.0,
    };
    let n = probe.n();
    let t_noon = probe.time_at(0.0, 0.0);
    let t_change = t_noon + 400.0;
    let beta_rate = -2e-7;
    let orbit = Kepler {
        beta: -beta_rate * t_change,
        beta_rate,
        ..probe
    };
    let rate = 2.0 * n;
    assert!(rate > orbit.orbital_rate(t_change));
    assert!(t_change < probe.time_at(FRAC_PI_2, t_noon));
    let law = GnssAttitudeLaw::GpsIIF {
        yaw_rate_rad_s: rate,
        yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
        shadow_limit_rad: PI - 1e-6,
    };
    let sample = move |t: f64| orbit.sample(t);
    let span = 0.24 * probe.period();
    let turn = ReferenceTurn::gps_noon(
        &sample,
        t_noon,
        span,
        0.01,
        rate,
        GPS_IIF_DEFAULT_YAW_BIAS_RAD,
    )
    .unwrap();
    assert!(
        turn.zone_end < t_change && turn.end > t_change && turn.end < turn.zone_end + 1800.0,
        "{turn:?}"
    );
    let trajectory = orbit.trajectory();
    let mut checks = turn.checks(23.0);
    checks.push(turn.zone_end + 1795.0);
    assert_noon_turn(&law, &trajectory, &turn, &checks, "IIF across beta zero");
}

/// The Galileo IOV region beginning is the last change of the region
/// condition before the requested instant. With custom limits of 88.5 deg and
/// 20 deg and beta rising through zero and then steeply (from -1 deg at the
/// quadrature before midnight to 1.6 deg at the one after,
/// `beta = -1 deg + 2.6 deg (t / T)^30`, 0.045 rad over the half orbit), the
/// condition holds around midnight with beta negative, fails from about
/// 0.8 deg before the next quadrature (`cos(beta) |sin(eta)|` above
/// `sin(88.5 deg)` while beta is below 1.5 deg) and holds again about
/// 0.25 deg before it, where beta has passed 1.5 deg: the region the
/// satellite is in began at that last change, with beta positive, so
/// `G = -1`. The yaw against the metadata formula with the beginning from a
/// backward scan of the condition.
#[test]
fn galileo_iov_region_begins_at_the_last_change_before_the_instant() {
    let n = (GM / (GALILEO_A * GALILEO_A * GALILEO_A)).sqrt();
    let half = PI / n;
    let beta_at = move |t: f64| deg(-1.0) + deg(2.6) * libm::pow(t / half, 30.0);
    let eta_at = move |t: f64| FRAC_PI_2 + n * t;
    let (beta_x, beta_y) = (deg(88.5), deg(20.0));
    let law = GnssAttitudeLaw::GalileoIov {
        beta_x_rad: beta_x,
        beta_y_rad: beta_y,
    };
    let in_region = |t: f64| {
        libm::sin(eta_at(t)).abs() * libm::cos(beta_at(t)) < libm::sin(beta_x)
            && libm::sin(beta_at(t)).abs() < libm::sin(beta_y)
    };
    let trajectory = move |t: f64| {
        Some(
            Circular {
                a: GALILEO_A,
                beta: beta_at(t),
                u0: eta_at(t),
            }
            .sample(0.0),
        )
    };
    assert!(in_region(0.5 * half) && !in_region(half * (1.0 - 0.5 / 180.0)));
    let mut checked = 0;
    for f in [-0.2, -0.1, 0.3] {
        let t_s = half * (1.0 + f / 180.0);
        assert!(in_region(t_s), "{t_s}");
        let mut t = t_s;
        while in_region(t - 0.1) {
            t -= 0.1;
        }
        let entry = change(in_region, t - 0.1, t);
        assert!(
            beta_at(entry) > 0.0 && entry > 0.99 * half,
            "entry {entry} beta {}",
            beta_at(entry)
        );
        let gamma = 1.0_f64.copysign(-libm::sin(beta_at(entry)));
        let s_x = -libm::sin(eta_at(t_s)) * libm::cos(beta_at(t_s));
        let s_y = -libm::sin(beta_at(t_s));
        let sy = libm::sin(beta_y);
        let s_hy = 0.5 * (sy * gamma + s_y)
            + 0.5 * (sy * gamma - s_y) * libm::cos(PI * s_x.abs() / libm::sin(beta_x));
        let expected = libm::atan2(s_hy, s_x);
        let attitude = gnss_attitude(&law, &trajectory, t_s).unwrap_or_else(|e| panic!("{e}"));
        let error = wrap_pi(attitude.yaw_rad - expected).abs();
        assert!(
            error < 1e-9,
            "t {t_s}: yaw {} vs {expected}",
            attitude.yaw_rad
        );
        checked += 1;
    }
    assert_eq!(checked, 3);
}

/// A circular orbit whose argument from noon is `u(t)`: the state of
/// [`Circular`] at that argument.
fn circular_with(
    a: f64,
    beta: f64,
    u: impl Fn(f64) -> f64,
) -> impl Fn(f64) -> Option<AttitudeSample> {
    move |t| Some(Circular { a, beta, u0: u(t) }.sample(0.0))
}

/// Near 1e18 s the representable instants are 128 s apart, more than the
/// 45 s step of the Galileo FOC colinearity scan: the scan then visits every
/// representable instant, and the turn starts at the first one at which the
/// colinearity angle `acos(cos(beta) cos(u))` is at most 10 deg (Galileo
/// satellite metadata 3.1.2), with the published cosine yaw from there.
#[test]
fn colinearity_scan_finer_than_the_time_resolution_visits_every_instant() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GalileoFoc, None).unwrap();
    let n = (GM / (GALILEO_A * GALILEO_A * GALILEO_A)).sqrt();
    let t_ref: f64 = 1.0e18;
    assert_eq!(t_ref.next_up() - t_ref, 128.0);
    let beta = deg(2.0);
    // The first instant past the in-plane angle of -10 deg is at -9.95 deg,
    // where the colinearity angle is still 10.15 deg: the scan takes one
    // step, to the next instant.
    let noon = 128.0 * 30.0 + deg(9.95) / n;
    let u_at = move |t: f64| n * ((t - t_ref) - noon);
    let trajectory = circular_with(GALILEO_A, beta, u_at);
    let colinearity = |t: f64| libm::acos(libm::cos(beta) * libm::cos(u_at(t)));
    let mut k = 0.0;
    while colinearity(t_ref + k) > deg(10.0) || u_at(t_ref + k) < -deg(10.0) {
        k += 128.0;
        assert!(k < noon, "no start before noon");
    }
    let start = t_ref + k;
    assert!(u_at(start) < 0.0 && u_at(start - 128.0) > -deg(10.0));
    let psi0 = libm::atan2(-libm::tan(beta), libm::sin(u_at(start) + PI));
    let quarter = FRAC_PI_2.copysign(psi0);
    for j in 0..12 {
        let t = start + 128.0 * f64::from(j);
        let attitude = gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            attitude.mode,
            AttitudeMode::NoonTurn,
            "t - start {}",
            t - start
        );
        let expected = quarter + (psi0 - quarter) * libm::cos((t - start) / 5656.0 * TAU);
        let error = wrap_pi(attitude.yaw_rad - expected).abs();
        assert!(
            error < 1e-9,
            "t - start {}: yaw {} vs {expected}",
            t - start,
            attitude.yaw_rad
        );
    }
    let before = gnss_attitude(&law, &trajectory, start - 128.0).unwrap();
    assert_eq!(before.mode, AttitudeMode::Nominal);
}

/// Near 1.5e19 s the representable instants are 2048 s apart, more than the
/// 704 s a Galileo orbit takes over 5 deg: where the in-plane angle crosses
/// the start value just after one of them, the first representable instant
/// past the crossing is past noon, and the colinearity angle cannot be
/// followed to it. With beta 8 deg above the 5 deg start value there is no
/// turn at all, which the time resolution does not let the law tell from one:
/// [`GnssAttitudeError::OutsideValidatedDomain`], at that instant and after it.
#[test]
fn cosine_turn_window_coarser_than_the_time_resolution_is_refused() {
    let law = GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
        beta_limit_rad: deg(10.0),
        beta_limit_inclusive: false,
        start_angle: CosineTurnStartAngle::Colinearity,
        start_angle_rad: deg(5.0),
        period_s: 5656.0,
    });
    let n = (GM / (GALILEO_A * GALILEO_A * GALILEO_A)).sqrt();
    let t_ref: f64 = 1.5e19;
    assert_eq!(t_ref.next_up() - t_ref, 2048.0);
    let u_at = move |t: f64| -deg(5.0) + n * ((t - t_ref) - 1.0);
    let trajectory = circular_with(GALILEO_A, deg(8.0), u_at);
    assert!(u_at(t_ref + 2048.0) > 0.0);
    for t in [t_ref + 2048.0, t_ref + 4096.0] {
        let result = gnss_attitude(&law, &trajectory, t);
        assert!(outside_domain(&result), "t - ref {}: {result:?}", t - t_ref);
    }
}

/// Near 2e19 s the representable instants are 4096 s apart, so no instant
/// lies within the 1800 s after a GPS turn's zone except its end: the turn is
/// under way at its zone end. Near 8e18 s they are 1024 s apart: the last
/// instant within the 1800 s is the zone end plus 1024 s (the sum rounds to
/// plus 2048 s), so a turn at a slow custom rate that has not met the nominal
/// yaw holds there and is nominal at plus 2048 s.
#[test]
fn gps_turn_margin_is_exact_at_coarse_time_resolution() {
    let n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    let t_ref: f64 = 2.0e19;
    assert_eq!(t_ref.next_up() - t_ref, 4096.0);
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let trajectory = circular_with(GPS_A, deg(1.0), move |t| n * ((t - t_ref) - 0.3));
    let at = Context::at(&trajectory, t_ref).unwrap();
    let turn = laws::turn_event(&law, &at, &trajectory).unwrap().unwrap();
    let attitude = gnss_attitude(&law, &trajectory, turn.end_s).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(attitude.mode, AttitudeMode::NoonTurn);

    let t_ref: f64 = 8.0e18;
    assert_eq!(t_ref.next_up() - t_ref, 1024.0);
    let law = GnssAttitudeLaw::GpsIIR {
        yaw_rate_rad_s: 3e-5,
    };
    let trajectory = circular_with(GPS_A, deg(20.0), move |t| n * ((t - t_ref) - 0.3));
    let at = Context::at(&trajectory, t_ref).unwrap();
    let turn = laws::turn_event(&law, &at, &trajectory).unwrap().unwrap();
    let end = turn.end_s;
    assert!(end + 1800.0 == end + 2048.0);
    let held = gnss_attitude(&law, &trajectory, end + 1024.0).unwrap();
    assert_eq!(held.mode, AttitudeMode::NoonTurn);
    let after = gnss_attitude(&law, &trajectory, end + 2048.0).unwrap();
    assert_eq!(after.mode, AttitudeMode::Nominal);
}

/// Beta's own rate moves the nominal yaw too, `-sin(mu) sec^2(beta)
/// beta_rate / (sin^2(mu) + tan^2(beta))`, which Kouba's rate leaves out, so
/// the gap to the nominal yaw can fall and rise again between two instants
/// where Kouba's rate says it only falls. GPS IIA and IIF turns at custom
/// rates of 0.05 to 0.6 times the mean motion, without shadow, on a fast
/// circular orbit (radius 4500 km) with beta drifting at up to 1.6e-5 rad/s
/// (0.048 rad over the orbit) without changing sign; against [`ReferenceTurn`] at instants clear of the
/// previous and the next orbit's turns.
#[test]
fn gps_turn_meets_the_nominal_yaw_moved_by_beta_rate() {
    let a = 4_500_000.0;
    let n = (GM / (a * a * a)).sqrt();
    let period = TAU / n;
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for (beta_deg, beta_rate) in [
        (3.0, 3e-6),
        (3.0, -3e-6),
        (10.0, 1e-5),
        (10.0, -1e-5),
        (20.0, 1.6e-5),
        (20.0, -1.6e-5),
        (-20.0, 1.6e-5),
        (40.0, -1.6e-5),
    ] {
        for factor in [0.05, 0.1, 0.3, 0.6] {
            for iia in [false, true] {
                let rate = factor * n;
                let bias = if iia { 0.6 } else { -0.6 };
                let law = if iia {
                    GnssAttitudeLaw::GpsIIA {
                        yaw_rate_rad_s: rate,
                        yaw_bias_rad: bias,
                        shadow_limit_rad: PI - 1e-6,
                    }
                } else {
                    GnssAttitudeLaw::GpsIIF {
                        yaw_rate_rad_s: rate,
                        yaw_bias_rad: bias,
                        shadow_limit_rad: PI - 1e-6,
                    }
                };
                let t_noon = 2000.0;
                let orbit = Drifting {
                    a,
                    beta0: deg(beta_deg) - beta_rate * t_noon,
                    rate: beta_rate,
                    u0: -n * t_noon,
                };
                let sample = move |t: f64| orbit.sample(t);
                let Some(turn) =
                    ReferenceTurn::gps_noon(&sample, t_noon, 0.24 * period, 0.05, rate, bias)
                else {
                    continue;
                };
                // The previous orbit's turn may still run before this one
                // starts, and the next one starts about a period later.
                let previous_end = turn.end - period;
                let next_noon_zone = turn.start + period - 60.0;
                let trajectory = move |t: f64| Some(orbit.sample(t));
                for t in turn.checks(11.0) {
                    if t >= next_noon_zone || (t < turn.start && t <= previous_end + 60.0) {
                        continue;
                    }
                    let attitude = gnss_attitude(&law, &trajectory, t).unwrap();
                    let in_turn = turn.contains(t);
                    let agrees = if in_turn {
                        attitude.mode == AttitudeMode::NoonTurn
                            && wrap_pi(attitude.yaw_rad - turn.yaw(t)).abs() < 1e-9
                    } else {
                        attitude.mode == AttitudeMode::Nominal
                    };
                    if !agrees {
                        mismatches.push(format!(
                            "beta {beta_deg} rate {beta_rate} factor {factor} iia {iia} t - start {} reference {in_turn} law {:?} ({turn:?})",
                            t - turn.start,
                            attitude.mode
                        ));
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {checked}:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert!(checked > 2000, "{checked}");
}

/// Whether `values` rises (not strictly) to one maximum and falls after it,
/// to a relative tolerance for the rounding of the logarithms.
fn one_maximum(values: &[f64]) -> bool {
    let mut falling = false;
    for w in values.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() {
            continue;
        }
        let tol = 1e-12 * w[0].abs().max(1.0);
        if w[1] > w[0] + tol {
            if falling {
                return false;
            }
        } else if w[1] < w[0] - tol {
            falling = true;
        }
    }
    true
}

/// Kouba's nominal-rate margin (`glonass == false`) or the GLONASS-M noon
/// margin, up to the constant `-ln(R)`, at `points` in-plane angles of the
/// half orbit around noon (`noon == true`) or midnight of a Keplerian orbit
/// with eccentricity `e` and perigee `omega` from noon, with
/// `beta = beta_c + change (M - M_c) / pi` (`M` the mean anomaly, `M_c` the
/// centre's): one run per part of one sign of beta, with the largest `|beta|`
/// over the half orbit.
fn half_orbit_margins(
    e: f64,
    omega: f64,
    beta_c: f64,
    change: f64,
    noon: bool,
    glonass: bool,
    points: u32,
) -> (Vec<Vec<f64>>, f64) {
    let mean = |nu: f64| {
        let ea = 2.0
            * libm::atan2(
                (1.0 - e).sqrt() * libm::sin(0.5 * nu),
                (1.0 + e).sqrt() * libm::cos(0.5 * nu),
            );
        ea - e * libm::sin(ea)
    };
    let centre_u = if noon { 0.0 } else { PI };
    let m_c = mean(wrap_pi(centre_u - omega));
    let mut parts: Vec<Vec<f64>> = vec![Vec::new()];
    let (mut last_sign, mut largest) = (0.0, 0.0_f64);
    for i in 1..points {
        // In-plane angle to the centre, from the entry quadrature.
        let x = FRAC_PI_2 - PI * f64::from(i) / f64::from(points);
        let u = if noon { -x } else { PI - x };
        let nu = wrap_pi(u - omega);
        let beta = beta_c + change * wrap_pi(mean(nu) - m_c) / PI;
        largest = largest.max(beta.abs());
        let sign = beta.signum();
        if !glonass && last_sign != 0.0 && sign != last_sign {
            parts.push(Vec::new());
        }
        last_sign = sign;
        let tb = libm::tan(beta).abs();
        let rate = libm::log((1.0 + e * libm::cos(nu)).powi(2));
        let s = libm::sin(x);
        let value = if glonass {
            let d = x.abs();
            if d == 0.0 {
                rate - libm::log(tb)
            } else {
                rate + libm::log(libm::atan2(libm::sin(d), tb)) - libm::log(d)
            }
        } else {
            rate + libm::log(tb) + libm::log(libm::cos(x)) - libm::log(s * s + tb * tb)
        };
        parts.last_mut().unwrap().push(value);
    }
    (parts, largest)
}

/// The bounds the laws check: on Keplerian orbits with beta moving at a
/// constant rate, Kouba's nominal yaw rate rises to one maximum over each
/// half orbit (and over each part of one sign of beta) and falls after it on
/// every orbit of eccentricity up to the GPS bound 0.3 and 3e-4 past it (the
/// bound's allowance of 2e-4 and more), and the GLONASS-M noon margin on
/// every orbit of eccentricity up to 0.25 with `|beta|` below 30 deg, 0.2
/// below 60 deg and 0.1 above, each to 3e-4 past it, for every argument of
/// perigee and beta of a grid (both signs, 0 and 1e-9, steps of 1.5 deg to
/// 88.5 deg) with beta fixed and moving by 0.025 rad per half period either
/// way; at the corners of the GLONASS-M bounds (eccentricity at each bound,
/// 2e-4 and 3e-4 past it, the largest `|beta|` over the half orbit 1e-6 rad,
/// 0.25 deg and 1 deg below 30 deg and 60 deg) for every argument of perigee
/// in steps of 1 deg. The maximum yaw rate only shifts both margins
/// (`ln(mu_rate / R)`), so it does not enter. Beyond the bounds: Kouba's rate
/// has two maxima at eccentricity 0.35 for some perigees, and the GLONASS-M
/// margin at 0.2 with beta 85 deg, at 0.3 with beta fixed at 34.95 deg
/// (perigee 174 deg from noon) and at 0.3 with beta at 2.5 deg moving by
/// 0.1 rad per orbit. Each margin is sampled at 500 in-plane angles of the
/// half orbit (2000 for the cases beyond), arguments of perigee 12 deg apart
/// outside the corners, with the orbital rate
/// `n (1 + e cos(nu))^2 / (1 - e^2)^(3/2)`.
#[test]
fn turn_zone_margins_have_one_maximum_per_half_orbit_within_the_bounds() {
    let mut betas = vec![0.0, 1e-9, -1e-9];
    for k in 1..=59 {
        let b = deg(1.5 * f64::from(k));
        betas.extend([b, -b]);
    }
    let changes = [0.0, 0.025, -0.025];
    let mut checked = 0;
    for e in [0.0, 0.1, 0.2, 0.25, 0.3, 0.3003] {
        for k in 0..30 {
            let omega = deg(12.0 * f64::from(k));
            for &beta in &betas {
                for change in changes {
                    for noon in [true, false] {
                        let (m, largest) =
                            half_orbit_margins(e, omega, beta, change, noon, false, 500);
                        if largest >= FRAC_PI_2 {
                            continue;
                        }
                        assert!(
                            m.iter().all(|part| one_maximum(part)),
                            "Kouba e {e} omega {} beta {} change {change} noon {noon}",
                            12 * k,
                            beta.to_degrees()
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    for e in [0.0, 0.1, 0.1003, 0.2, 0.2003, 0.25, 0.2503] {
        for k in 0..30 {
            let omega = deg(12.0 * f64::from(k));
            for &beta in &betas {
                // The largest |beta| is at least the centre's.
                if (e > 0.1003 && beta.abs() >= deg(60.0))
                    || (e > 0.2003 && beta.abs() >= deg(30.0))
                {
                    continue;
                }
                for change in changes {
                    let (m, largest) = half_orbit_margins(e, omega, beta, change, true, true, 500);
                    if e <= glonass_noon_bound(largest) + 3e-4 && largest < FRAC_PI_2 {
                        assert!(
                            m.iter().all(|part| one_maximum(part)),
                            "GLONASS-M e {e} omega {} beta {} change {change}",
                            12 * k,
                            beta.to_degrees()
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    let mut corners = 0;
    for (bound, limit) in [(0.25, deg(30.0)), (0.2, deg(60.0))] {
        for e in [bound, bound + 2e-4, bound + 3e-4] {
            for change in changes {
                for k in 0..360 {
                    let omega = deg(f64::from(k));
                    // With beta positive over the half orbit its largest
                    // value is beta at the centre plus a shift set by the
                    // change and the orbit.
                    let shift =
                        half_orbit_margins(e, omega, limit, change, true, true, 500).1 - limit;
                    for below in [1e-6, deg(0.25), deg(1.0)] {
                        let target = limit - below;
                        let (m, largest) =
                            half_orbit_margins(e, omega, target - shift, change, true, true, 500);
                        assert!(
                            largest < limit && largest > target - 1e-9,
                            "corner e {e} omega {k} target {target}: largest {largest}"
                        );
                        assert!(
                            m.iter().all(|part| one_maximum(part)),
                            "GLONASS-M corner e {e} omega {k} largest {} change {change}",
                            largest.to_degrees()
                        );
                        corners += 1;
                    }
                }
            }
        }
    }
    assert_eq!(corners, 2 * 3 * 3 * 360 * 3);
    assert!(checked > 80_000, "{checked}");
    let bimodal = |m: &(Vec<Vec<f64>>, f64)| !m.0.iter().all(|part| one_maximum(part));
    let two_kouba = (0..72).any(|k| {
        bimodal(&half_orbit_margins(
            0.35,
            deg(5.0 * f64::from(k)),
            deg(80.0),
            0.0,
            true,
            false,
            2000,
        ))
    });
    assert!(two_kouba, "Kouba's rate at eccentricity 0.35");
    let two_glonass = (0..72).any(|k| {
        bimodal(&half_orbit_margins(
            0.2,
            deg(5.0 * f64::from(k)),
            deg(85.0),
            0.0,
            true,
            true,
            2000,
        ))
    });
    assert!(two_glonass, "GLONASS-M at eccentricity 0.2, beta 85 deg");
    assert!(bimodal(&half_orbit_margins(
        0.3,
        deg(174.0),
        deg(34.95),
        0.0,
        true,
        true,
        2000
    )));
    let moving = (0..180).any(|k| {
        bimodal(&half_orbit_margins(
            0.3,
            deg(2.0 * f64::from(k)),
            deg(2.5),
            0.05,
            true,
            true,
            2000,
        ))
    });
    assert!(moving, "GLONASS-M at eccentricity 0.3, beta 2.5 deg moving");
}

/// At beta = 0 the nominal yaw is constant between its flips by pi at noon
/// and midnight, and a GPS IIA noon turn bridges the noon flip from the noon
/// instant at `sign(b) R = +R` from yaw 0, so it meets the nominal yaw pi / R
/// after noon. With the PRN 11 rate (0.2 deg/s) that is 900 s; on an orbit of
/// period 1980 s it comes 90 s before midnight and before the shadow entry
/// (Sun-satellite angle 166.75 deg, 917.125 s after noon), so the yaw is
/// nominal from 900 s to the shadow entry. The same with beta 5e-16 of either
/// sign, within the rounding of zero, where the zone lasts microseconds; and
/// on an orbit of eccentricity 0.25 with perigee a quarter turn after noon,
/// where the meeting comes 129 s (PRN 11) and 124 s (PRN 12, 0.199 deg/s)
/// before midnight.
#[test]
fn gps_turn_meeting_before_the_next_centre_at_beta_zero() {
    let check = |law: &GnssAttitudeLaw,
                 trajectory: &dyn Fn(f64) -> Option<AttitudeSample>,
                 t_noon: f64,
                 t_shadow: f64,
                 rate: f64,
                 tolerance: f64,
                 where_: &str| {
        let meeting = t_noon + PI / rate;
        assert!(meeting + 6.0 < t_shadow, "{where_}");
        for dt in [-450.0, -5.0, -0.5] {
            let t = meeting + dt;
            let attitude =
                gnss_attitude(law, trajectory, t).unwrap_or_else(|e| panic!("{where_} {t}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::NoonTurn, "{where_} t {t}");
            let expected = rate * (t - t_noon);
            assert!(
                wrap_pi(attitude.yaw_rad - expected).abs() < tolerance,
                "{where_} t {t}: yaw {} vs {expected}",
                attitude.yaw_rad
            );
        }
        for t in [
            meeting + 0.5,
            meeting + 5.0,
            0.5 * (meeting + t_shadow),
            t_shadow - 0.2,
        ] {
            let attitude =
                gnss_attitude(law, trajectory, t).unwrap_or_else(|e| panic!("{where_} {t}: {e}"));
            assert_eq!(attitude.mode, AttitudeMode::Nominal, "{where_} t {t}");
        }
    };
    let prn11 = GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(11)).unwrap();
    let r11 = gps_iia_default_yaw_rate_rad_s(11).unwrap();
    for beta in [0.0, 5e-16, -5e-16] {
        let orbit = Circular {
            a: 3_408_025.3,
            beta,
            u0: 0.0,
        };
        let t_shadow = orbit.time_at(GPS_SHADOW_LIMIT_RAD);
        let tolerance = if beta == 0.0 { 1e-12 } else { 1e-6 };
        check(
            &prn11,
            &orbit.trajectory(),
            0.0,
            t_shadow,
            r11,
            tolerance,
            &format!("circular beta {beta}"),
        );
    }
    let prn12 = GnssAttitudeLaw::for_block(GnssBlock::GpsIIA, Some(12)).unwrap();
    let r12 = gps_iia_default_yaw_rate_rad_s(12).unwrap();
    let orbit = Kepler {
        a: 4_500_000.0,
        e: 0.25,
        beta: 0.0,
        beta_rate: 0.0,
        omega: deg(90.0),
        m0: 0.0,
    };
    let t_noon = orbit.time_at(0.0, 0.0);
    let t_shadow = orbit.time_at(GPS_SHADOW_LIMIT_RAD, t_noon);
    for (law, rate) in [(prn11, r11), (prn12, r12)] {
        check(
            &law,
            &orbit.trajectory(),
            t_noon,
            t_shadow,
            rate,
            1e-9,
            &format!("eccentric rate {rate}"),
        );
    }
}

/// On an orbit of eccentricity 0.3 the GLONASS-M noon margin has two maxima
/// on the noon half orbit from `|beta|` of 34.9 deg (argument of perigee
/// 174 deg from noon): at beta 34.99 deg, with perigee 175 deg and 174 deg
/// from noon and maximum yaw rates that put the margin's two zones 293 s and
/// 300 s apart, the noon half orbit lies outside the validated domain, both
/// inside the first zone and in the valley between them.
#[test]
fn glonass_m_noon_half_orbits_with_two_zones_are_outside_the_domain() {
    for (omega_deg, rate, dt) in [
        (175.0, 1.036_476_247_2e-4, -13_300.0),
        (174.0, 1.025_164_560_8e-4, -13_200.0),
    ] {
        let orbit = Kepler {
            a: GLONASS_A,
            e: 0.3,
            beta: deg(34.99),
            beta_rate: 0.0,
            omega: deg(omega_deg),
            m0: 0.0,
        };
        let law = GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s: rate,
            shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
        };
        let t_noon = orbit.time_at(0.0, 0.0);
        let t = t_noon + dt;
        assert!(libm::cos(orbit.argument_from_noon(t)) > 0.0);
        let result = gnss_attitude(&law, &orbit.trajectory(), t);
        assert!(outside_domain(&result), "omega {omega_deg}: {result:?}");
    }
}

/// The eccentricity is that of the conic through three positions about the
/// orbit normal at the instant, which a turning orbit plane biases by about
/// 0.04 times the square of the angle it turns over half an orbit: with beta
/// changing at 5e-5 rad/s (2.2 rad over a GPS orbit), through zero 4000 s
/// after t = 0 so that a turn could hold around the instants checked, an
/// orbit of eccentricity 0.32 fits a conic below 0.3, and such a plane lies
/// outside the validated domain for its motion. With beta changing at the Sun's rate
/// (2e-7 rad/s) an orbit of eccentricity 0.3 fits conics up to 0.3000008,
/// within the bound's allowance for that bias, and is inside it. On an orbit
/// of eccentricity 0.9 with beta changing by 0.5 rad over the orbit, an
/// instant just after perigee, whose half turn before lasts 1.9 % of the
/// period, is outside it too.
#[test]
fn orbit_plane_turning_beyond_the_bound_is_outside_the_domain() {
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let fast = Kepler {
        a: GPS_A,
        e: 0.32,
        beta: -0.2,
        beta_rate: 5e-5,
        omega: deg(60.0),
        m0: 0.0,
    };
    for t in [4500.0, 6000.0] {
        let result = gnss_attitude(&law, &fast.trajectory(), t);
        assert!(
            matches!(
                &result,
                Err(GnssAttitudeError::OutsideValidatedDomain { reason, .. })
                    if reason.contains("turns by more than 0.05 rad")
            ),
            "t {t}: {result:?}"
        );
    }
    let slow = Kepler {
        a: GPS_A,
        e: 0.3,
        beta: 0.1,
        beta_rate: 2e-7,
        omega: 0.5,
        m0: 0.0,
    };
    for t in [0.0, 1000.0, 20_000.0, 30_000.0] {
        gnss_attitude(&law, &slow.trajectory(), t).unwrap_or_else(|e| panic!("t {t}: {e}"));
    }
    // On an orbit of eccentricity 0.9 the half turn around perigee lasts
    // 1.9 % of the period: beta changing by 0.5 rad over the orbit changes by
    // 0.009 rad over that half turn. The bound is on the full turn, which
    // lasts the period wherever the instant lies.
    let eccentric = Kepler {
        a: GALILEO_A,
        e: 0.9,
        beta: deg(1.0),
        beta_rate: 0.0,
        omega: 0.0,
        m0: 0.0,
    };
    let eccentric = Kepler {
        beta_rate: 0.5 / eccentric.period(),
        ..eccentric
    };
    let galileo = GnssAttitudeLaw::for_block(GnssBlock::GalileoFoc, None).unwrap();
    let after_perigee = eccentric.time_at(FRAC_PI_2, 0.0);
    let half_turn =
        after_perigee - eccentric.time_at(-FRAC_PI_2, after_perigee - eccentric.period());
    assert!(half_turn < 0.02 * eccentric.period(), "{half_turn}");
    for t in [after_perigee, after_perigee + 0.5 * half_turn] {
        let result = gnss_attitude(&galileo, &eccentric.trajectory(), t);
        assert!(outside_domain(&result), "t {t}: {result:?}");
    }
}

/// A trajectory whose beta is that of [`Kepler`] at `beta(t)` instead of a
/// fixed rate.
fn kepler_with_beta(
    orbit: Kepler,
    beta: impl Fn(f64) -> f64,
) -> impl Fn(f64) -> Option<AttitudeSample> {
    move |t| {
        Some(
            Kepler {
                beta: beta(t),
                beta_rate: 0.0,
                ..orbit
            }
            .sample(t),
        )
    }
}

/// Beta passes through zero at most once within a half orbit for the Sun's
/// motion, and the laws refuse half orbits where their samples of beta show
/// otherwise: a GPS noon half orbit where beta is positive at the
/// quadratures and at noon but negative over a sixth of the half orbit
/// before noon; a GLONASS-M noon half orbit of eccentricity 0.25 whose beta
/// is 29 deg at the quadratures and at noon but reaches 31 deg in between, at
/// a maximum yaw rate (3e-4 rad/s) at which a noon zone could hold there;
/// and a Galileo IOV region whose beta leaves 2 deg and returns between the
/// region's start and noon.
#[test]
fn beta_beyond_the_sampled_assumptions_is_outside_the_domain() {
    let fast = Kepler {
        a: 4_500_000.0,
        e: 0.0,
        beta: 0.0,
        beta_rate: 0.0,
        omega: 0.0,
        m0: 0.0,
    };
    let quarter = 0.25 * fast.period();
    let dip = move |t: f64| {
        let x = (t + 0.5 * quarter) / (0.2 * quarter);
        2e-3 - 4e-3 * libm::exp(-x * x)
    };
    assert!(dip(-quarter) > 0.0 && dip(0.0) > 0.0 && dip(quarter) > 0.0);
    assert!(dip(-0.5 * quarter) < 0.0);
    let trajectory = kepler_with_beta(fast, dip);
    let law = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    for t in [-0.8 * quarter, -0.1 * quarter, 0.3 * quarter] {
        let result = gnss_attitude(&law, &trajectory, t);
        assert!(outside_domain(&result), "GPS t {t}: {result:?}");
    }

    let orbit = Kepler {
        a: GLONASS_A,
        e: 0.25,
        beta: 0.0,
        beta_rate: 0.0,
        omega: deg(180.0),
        m0: 0.0,
    };
    let t_noon = orbit.time_at(0.0, 0.0);
    let start = orbit.time_at(-FRAC_PI_2, t_noon - orbit.period());
    let end = orbit.time_at(FRAC_PI_2, t_noon);
    let bump_at = start + 0.25 * (end - start);
    let width = 0.07 * (end - start);
    let bump = move |t: f64| {
        let x = (t - bump_at) / width;
        deg(29.0) + deg(2.0) * libm::exp(-x * x)
    };
    let law = GnssAttitudeLaw::GlonassM {
        yaw_rate_rad_s: 3e-4,
        shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
    };
    let trajectory = kepler_with_beta(orbit, bump);
    for t in [t_noon - 1000.0, t_noon, 0.5 * (t_noon + end)] {
        let result = gnss_attitude(&law, &trajectory, t);
        assert!(outside_domain(&result), "GLONASS-M t {t}: {result:?}");
    }

    let orbit = Kepler {
        a: GALILEO_A,
        e: 0.0,
        beta: 0.0,
        beta_rate: 0.0,
        omega: 0.0,
        m0: 0.0,
    };
    let n = orbit.n();
    // The region starts 15 deg before noon; beta is 1.03 deg there and
    // 1.07 deg at noon and rises to 2.5 deg 7 deg before noon, above 2 deg
    // over 5 deg of orbit.
    let leave = move |t: f64| {
        let x = (t + deg(7.0) / n) / (deg(4.0) / n);
        deg(1.0) + deg(1.5) * libm::exp(-x * x)
    };
    let law = GnssAttitudeLaw::for_block(GnssBlock::GalileoIov, None).unwrap();
    let trajectory = kepler_with_beta(orbit, leave);
    for u_deg in [-1.0, 3.0, 10.0] {
        let t = deg(u_deg) / n;
        let result = gnss_attitude(&law, &trajectory, t);
        assert!(outside_domain(&result), "Galileo IOV u {u_deg}: {result:?}");
    }
}

/// Within the rounding of zero beta's sign is not taken, and a maximum yaw
/// rate so slow that Kouba's zone at such a beta reaches far from the centre
/// gives a zone that depends on beta below its resolution (beta formed as
/// `pi / 2` minus an angle rounds to 0 within about 1e-16 rad of zero): GPS
/// IIR at 1e-22 rad/s on a GPS orbit whose beta passes through zero at noon
/// at 6e-20 rad/s is outside the validated domain in its noon half orbit.
#[test]
fn slow_turns_at_beta_within_its_rounding_are_outside_the_domain() {
    let n = (GM / (GPS_A * GPS_A * GPS_A)).sqrt();
    let trajectory = move |t: f64| {
        Some(
            Circular {
                a: GPS_A,
                beta: 6e-20 * t,
                u0: n * t,
            }
            .sample(0.0),
        )
    };
    let law = GnssAttitudeLaw::GpsIIR {
        yaw_rate_rad_s: 1e-22,
    };
    for t in [-5000.0, -100.0, 0.0, 2500.0, 5000.0] {
        let result = gnss_attitude(&law, &trajectory, t);
        assert!(outside_domain(&result), "t {t}: {result:?}");
    }
}

/// In a half orbit whose beta changes sign, the maximum yaw rate must be at
/// least the orbital rate where beta changes sign. GPS IIR at 1e-3 times the
/// orbital rate on a circular GPS orbit whose beta rises from -0.006 rad at
/// noon through zero 45 deg after it (0.048 rad over the orbit) is outside
/// the validated domain in that noon half orbit: Kouba's zones on the two
/// sides of beta's change of sign lie degrees apart there (from 87 deg before
/// noon to 41 deg after it, and from 53 deg to 78 deg after it), also at
/// 65 deg after noon, between them. At the orbital rate itself and at the
/// published rate the same instants have attitudes.
#[test]
fn slow_turns_where_beta_changes_sign_are_outside_the_domain() {
    let probe = Kepler {
        a: GPS_A,
        e: 0.0,
        beta: -0.006,
        beta_rate: 0.0,
        omega: 0.0,
        m0: 0.0,
    };
    let n = probe.n();
    let orbit = Kepler {
        beta_rate: 0.048 / probe.period(),
        ..probe
    };
    let trajectory = orbit.trajectory();
    let instants = [-60.0, 0.0, 30.0, 50.0, 65.0].map(|u_deg: f64| deg(u_deg) / n);
    let slow = GnssAttitudeLaw::GpsIIR {
        yaw_rate_rad_s: 1e-3 * n,
    };
    for t in instants {
        let result = gnss_attitude(&slow, &trajectory, t);
        assert!(
            matches!(
                &result,
                Err(GnssAttitudeError::OutsideValidatedDomain { reason, .. })
                    if reason.contains("below the orbital rate where beta changes sign")
            ),
            "t {t}: {result:?}"
        );
    }
    for law in [
        GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s: n },
        GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap(),
    ] {
        for t in instants {
            gnss_attitude(&law, &trajectory, t).unwrap_or_else(|e| panic!("{law:?} t {t}: {e}"));
        }
    }
}

/// With the maximum yaw rate `R` at least the orbital rate `n` where beta
/// changes sign (the laws' rule for joining Kouba's zones of the two signs
/// into one turn), the two zones lie next to the centre. On a circular orbit
/// with `beta = s (x_c - x)`, `x` the in-plane angle to noon and
/// `s = 0.05 / (2 pi)` (0.05 rad over an orbit, the laws' bound), both sides
/// of the change at `x_c` hold a zone only for `|x_c|` below `s / (4 k)`,
/// `k = R / n` (0.002 rad at `k = 1`); the zones then lie within
/// `(1 + sqrt(2)) / 2 s / k` of noon (0.0096 rad) and at most
/// `(2 - sqrt(2)) / 2 s / k` apart (0.0023 rad), which they near as the far
/// zone shrinks to a point. From Kouba's rate on a scan of 240001 in-plane
/// angles over `[-0.012, 0.012]` rad, at `k` of 1, 1.2 and 2.
#[test]
fn zones_either_side_of_a_beta_sign_change_lie_next_to_the_centre() {
    let s = 0.05 / TAU;
    let runs = |k: f64, x_c: f64| {
        let points: u32 = 240_001;
        let mut runs: Vec<(f64, f64)> = Vec::new();
        let mut open: Option<(f64, f64)> = None;
        for j in 0..points {
            let x = -0.012 + 0.024 * f64::from(j) / f64::from(points - 1);
            let t = libm::tan(s * (x_c - x)).abs();
            let (sn, c) = (libm::sin(x), libm::cos(x));
            let inside = t > 0.0 && t * c / (sn * sn + t * t) > k;
            match (inside, open) {
                (true, None) => open = Some((x, x)),
                (true, Some((a, _))) => open = Some((a, x)),
                (false, Some(run)) => {
                    runs.push(run);
                    open = None;
                }
                (false, None) => {}
            }
        }
        runs.extend(open);
        runs
    };
    let (mut widest, mut gap_seen) = (0.0_f64, 0.0_f64);
    for k in [1.0, 1.2, 2.0] {
        let quarter = s / (4.0 * k);
        for f in [0.05, 0.1, 0.2, 0.249, 0.2499, -0.2, -0.2499] {
            let zones = runs(k, f * 4.0 * quarter);
            assert_eq!(zones.len(), 2, "k {k} x_c {f} of s / k: {zones:?}");
            let extent = zones
                .iter()
                .fold(0.0_f64, |m, &(a, b)| m.max(a.abs()).max(b.abs()));
            let gap = zones[1].0 - zones[0].1;
            assert!(
                extent < 0.5 * (1.0 + SQRT_2) * s / k,
                "k {k} f {f}: {extent}"
            );
            assert!(gap < 0.5 * (2.0 - SQRT_2) * s / k, "k {k} f {f}: {gap}");
            (widest, gap_seen) = (widest.max(extent), gap_seen.max(gap));
        }
        // Past s / (4 k) only the side holding noon has a zone.
        for f in [0.26, -0.26, 0.5] {
            assert_eq!(runs(k, f * 4.0 * quarter).len(), 1, "k {k} f {f}");
        }
    }
    assert!(widest < 0.0097 && widest > 0.0095, "{widest}");
    assert!(gap_seen < 0.0024 && gap_seen > 0.0022, "{gap_seen}");
}

/// The domain checks read the trajectory over the orbit before the instant,
/// and run only where the law could be in a turn: Galileo FOC at beta 30 deg
/// (the turn needs `|beta|` below 4.1 deg at its start) on a trajectory that
/// begins 3000 s before the instant reads it only over the 2828 s a turn
/// lasts and gives the nominal attitude; GPS IIR at the published rate on an
/// orbit of eccentricity 0.35 (above the bound of 0.3) at beta 60 deg, where
/// Kouba's zone needs `|beta|` below about 5.3 deg at the orbit's largest
/// orbital rate, gives the nominal attitude over the whole orbit, and the
/// same orbit at beta 3 deg is outside the validated domain. GLONASS-M at the
/// published rate on an orbit of eccentricity 0.3 (above the noon bound of
/// 0.25) at beta 25 deg, where its noon zone needs `|beta|` below about
/// 8 deg, gives the nominal attitude too.
#[test]
fn domain_checks_run_only_where_a_turn_could_exist() {
    let galileo = Kepler {
        a: GALILEO_A,
        e: 0.0,
        beta: deg(30.0),
        beta_rate: 0.0,
        omega: 0.0,
        m0: 0.0,
    };
    let arc_start = -3000.0;
    let arc = move |t: f64| (t >= arc_start).then(|| galileo.sample(t));
    let foc = GnssAttitudeLaw::for_block(GnssBlock::GalileoFoc, None).unwrap();
    for t in [0.0, 1000.0, 20_000.0] {
        let attitude = gnss_attitude(&foc, &arc, t).unwrap_or_else(|e| panic!("FOC t {t}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::Nominal, "FOC t {t}");
    }
    let iir = GnssAttitudeLaw::for_block(GnssBlock::GpsIIR, None).unwrap();
    let eccentric = |beta_deg: f64| Kepler {
        a: GPS_A,
        e: 0.35,
        beta: deg(beta_deg),
        beta_rate: 0.0,
        omega: deg(40.0),
        m0: 0.0,
    };
    let high = eccentric(60.0);
    for k in 0..16 {
        let t = high.period() * f64::from(k) / 16.0;
        let attitude =
            gnss_attitude(&iir, &high.trajectory(), t).unwrap_or_else(|e| panic!("IIR t {t}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::Nominal, "IIR t {t}");
    }
    let low = eccentric(3.0);
    let t_noon = low.time_at(0.0, 0.0);
    let result = gnss_attitude(&iir, &low.trajectory(), t_noon);
    assert!(outside_domain(&result), "IIR at beta 3 deg: {result:?}");
    let glonass = GnssAttitudeLaw::for_block(GnssBlock::GlonassM, None).unwrap();
    let orbit = Kepler {
        a: GLONASS_A,
        e: 0.3,
        beta: deg(25.0),
        beta_rate: 0.0,
        omega: deg(160.0),
        m0: 0.0,
    };
    for u_deg in [-80.0, -30.0, 0.0, 20.0, 85.0, 150.0] {
        let t = orbit.time_at(deg(u_deg), 0.0);
        let attitude = gnss_attitude(&glonass, &orbit.trajectory(), t)
            .unwrap_or_else(|e| panic!("GLONASS-M u {u_deg}: {e}"));
        assert_eq!(attitude.mode, AttitudeMode::Nominal, "GLONASS-M u {u_deg}");
    }
}

/// Point `i` of dimension `dim` of the Halton sequence (the radical inverse
/// of `i` in the `dim`-th prime base): the first points of every dimension
/// spread evenly over `[0, 1)`, and the dimensions do not repeat each other.
fn halton(i: u32, dim: usize) -> f64 {
    const PRIMES: [u32; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let base = PRIMES[dim];
    let (mut f, mut r, mut k) = (1.0, 0.0, i);
    while k > 0 {
        f /= f64::from(base);
        r += f * f64::from(k % base);
        k /= base;
    }
    r
}

/// The adjacent numbers around the change of `inside` in `[lo, hi]`,
/// bisected until no number lies between them.
fn change_pair(inside: impl Fn(f64) -> bool, mut lo: f64, mut hi: f64) -> (f64, f64) {
    let at_lo = inside(lo);
    assert_ne!(at_lo, inside(hi), "{lo} {hi}");
    loop {
        let mid = 0.5 * lo + 0.5 * hi;
        if mid <= lo || mid >= hi {
            return (lo, hi);
        }
        if inside(mid) == at_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

/// The first instant of `[lo, hi]` with the value `inside` has at `hi`.
fn change_to_the_last_bit(inside: impl Fn(f64) -> bool, lo: f64, hi: f64) -> f64 {
    change_pair(inside, lo, hi).1
}

impl Kepler {
    /// Beta at `t`.
    fn beta_at(&self, t: f64) -> f64 {
        self.beta + self.beta_rate * t
    }

    /// The orbital rate `h / r^2 = n (1 + e cos(nu))^2 / (1 - e^2)^(3/2)`.
    fn orbital_rate(&self, t: f64) -> f64 {
        let nu = self.true_anomaly(t);
        let q = 1.0 + self.e * libm::cos(nu);
        self.n() * q * q / libm::pow(1.0 - self.e * self.e, 1.5)
    }

    /// The nominal yaw `atan2(-tan(beta), sin(mu))`, `mu = u + pi`.
    fn nominal_yaw(&self, t: f64) -> f64 {
        libm::atan2(
            -libm::tan(self.beta_at(t)),
            -libm::sin(self.argument_from_noon(t)),
        )
    }
}

/// A GPS or GLONASS-M law for [`gps_reference`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum ReferenceBlock {
    Iia,
    Iir,
    Iif,
    Glonass,
}

#[derive(Clone, Copy, Debug)]
struct ReferenceParameters {
    block: ReferenceBlock,
    rate: f64,
    bias: f64,
    limit: f64,
}

impl ReferenceParameters {
    fn law(&self) -> GnssAttitudeLaw {
        match self.block {
            ReferenceBlock::Iia => GnssAttitudeLaw::GpsIIA {
                yaw_rate_rad_s: self.rate,
                yaw_bias_rad: self.bias,
                shadow_limit_rad: self.limit,
            },
            ReferenceBlock::Iir => GnssAttitudeLaw::GpsIIR {
                yaw_rate_rad_s: self.rate,
            },
            ReferenceBlock::Iif => GnssAttitudeLaw::GpsIIF {
                yaw_rate_rad_s: self.rate,
                yaw_bias_rad: self.bias,
                shadow_limit_rad: self.limit,
            },
            ReferenceBlock::Glonass => GnssAttitudeLaw::GlonassM {
                yaw_rate_rad_s: self.rate,
                shadow_limit_rad: self.limit,
            },
        }
    }

    /// Whether the zone of the noon (`noon == true`) or midnight half orbit
    /// is the shadow.
    fn shadow(&self, noon: bool) -> bool {
        !noon && self.block != ReferenceBlock::Iir
    }
}

/// A half orbit of [`Kepler`] from its closed form: quadrature to quadrature
/// around noon (argument from noon in `[-pi / 2, pi / 2)`) or midnight.
#[derive(Clone, Copy, Debug)]
struct ReferenceHalf {
    noon: bool,
    start: f64,
    centre: f64,
    end: f64,
}

impl ReferenceHalf {
    fn containing(orbit: &Kepler, t: f64) -> Self {
        let u = orbit.argument_from_noon(t);
        let noon = (-FRAC_PI_2..FRAC_PI_2).contains(&u);
        let (entry, centre, exit) = if noon {
            (-FRAC_PI_2, 0.0, FRAC_PI_2)
        } else {
            (FRAC_PI_2, PI, -FRAC_PI_2)
        };
        let start = orbit.time_at(entry, t - orbit.period());
        let start = if start > t {
            start - orbit.period()
        } else {
            start
        };
        let centre = orbit.time_at(centre, start);
        let end = orbit.time_at(exit, centre);
        Self {
            noon,
            start,
            centre,
            end,
        }
    }

    fn previous(&self, orbit: &Kepler) -> Self {
        Self::containing(orbit, self.start - 1e-3)
    }
}

/// Where a zone lies in its half orbit, from a scan of its margin.
#[derive(Clone, Copy, Debug)]
enum ReferenceSpan {
    None,
    /// Only the centre instant (Kouba's zone at beta = 0).
    AtCentre,
    Zone {
        start: f64,
        end: f64,
        from_start: bool,
        to_end: bool,
    },
}

/// The GPS IIA, IIR, IIF and GLONASS-M laws on a [`Kepler`] orbit evaluated
/// from its closed form (beta, the argument from noon, the orbital rate
/// `h / r^2` and the nominal yaw), independently of the law's searches: each
/// half orbit's zone from a scan of 2000 points and its centre, its edges
/// bisected to the last bit and a zone narrower than the scan zoomed in on
/// from the point of largest margin (Kouba 2009's nominal-rate zone, split
/// at beta's change of sign and run from the first instant inside to the
/// last; the shadow `cos(beta) cos(u) < cos(limit)`; the Dilssner et al.
/// 2011 span `k g(d) > d`, `k = mu_rate / R`, `g(d) = atan2(sin(d),
/// tan|beta|)`); the turn from the nominal yaw at the zone start at the
/// law's rate, after the zone until the gap to the continuous nominal yaw
/// closes (followed in steps of at most 1 s, finer where the nominal yaw
/// moves by more than 0.1 rad or the gap is small), at most 1800 s; the
/// latest turn started by the instant.
struct GpsReference {
    orbit: Kepler,
    p: ReferenceParameters,
    turns: std::cell::RefCell<std::collections::HashMap<u64, Option<SolvedTurn>>>,
}

#[derive(Clone, Copy, Debug)]
struct SolvedTurn {
    start: f64,
    zone_end: f64,
    /// The meeting with the nominal yaw (GPS), or the zone end.
    end: f64,
    /// The gap has not closed within the 1800 s: the turn holds at the end.
    held: bool,
    psi0: f64,
    rate: f64,
    after_rate: f64,
    mode: AttitudeMode,
    /// GLONASS-M: a noon zone holding at an end of its half orbit.
    refused: bool,
    /// GLONASS-M shadow: the exit's nominal yaw, held once reached.
    hold: Option<f64>,
}

/// The reference result at one instant: a yaw comes with the magnitude of
/// its rate of change in time (the turn's rate, 0 where a yaw is held) and
/// the turn's start instant, which set how far the uncertainty of that
/// instant moves it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ReferenceAttitude {
    Nominal,
    Yaw(AttitudeMode, f64, f64, f64),
    Refused,
}

impl GpsReference {
    fn new(orbit: Kepler, p: ReferenceParameters) -> Self {
        Self {
            orbit,
            p,
            turns: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// The in-plane angle to the centre, positive before it.
    fn to_centre(&self, noon: bool, t: f64) -> f64 {
        let u = self.orbit.argument_from_noon(t);
        if noon {
            -u
        } else {
            wrap_pi(PI - u)
        }
    }

    fn kouba_margin(&self, noon: bool, t: f64) -> f64 {
        let x = self.to_centre(noon, t);
        let tb = libm::tan(self.orbit.beta_at(t)).abs();
        let (s, c) = (libm::sin(x), libm::cos(x));
        let mu = self.orbit.orbital_rate(t);
        if c <= 0.0 {
            f64::NEG_INFINITY
        } else if s == 0.0 {
            libm::log(mu) - libm::log(tb) - libm::log(self.p.rate)
        } else if tb == 0.0 {
            f64::NEG_INFINITY
        } else {
            libm::log(mu * tb * c / (s * s + tb * tb)) - libm::log(self.p.rate)
        }
    }

    fn glonass_noon_margin(&self, t: f64) -> f64 {
        let k = self.orbit.orbital_rate(t) / self.p.rate;
        let d = self.to_centre(true, t).abs();
        let tb = libm::tan(self.orbit.beta_at(t)).abs();
        if d == 0.0 {
            libm::log(k) - libm::log(tb)
        } else {
            libm::log(k * libm::atan2(libm::sin(d), tb) / d)
        }
    }

    fn margin(&self, noon: bool, t: f64) -> f64 {
        if self.p.shadow(noon) {
            libm::cos(self.p.limit)
                - libm::cos(self.orbit.beta_at(t)) * libm::cos(self.orbit.argument_from_noon(t))
        } else if self.p.block == ReferenceBlock::Glonass {
            self.glonass_noon_margin(t)
        } else {
            self.kouba_margin(noon, t)
        }
    }

    fn zone(&self, half: &ReferenceHalf) -> ReferenceSpan {
        let noon = half.noon;
        let m = |t: f64| self.margin(noon, t);
        let inside = |t: f64| m(t) > 0.0;
        let points = 2000;
        let mut scan: Vec<f64> = (0..=points)
            .map(|k| half.start + (half.end - half.start) * f64::from(k) / f64::from(points))
            .collect();
        scan.push(half.centre);
        scan.sort_by(f64::total_cmp);
        let last = scan.len() - 1;
        let hits: Vec<usize> = (0..scan.len()).filter(|&i| inside(scan[i])).collect();
        let (first, final_) = match (hits.first(), hits.last()) {
            (Some(&i), Some(&j)) => (i, j),
            _ => {
                // Zoom in on the scan point of largest margin.
                let mut best = scan
                    .iter()
                    .copied()
                    .max_by(|a, b| m(*a).total_cmp(&m(*b)))
                    .unwrap();
                let mut width = (half.end - half.start) / f64::from(points);
                for _ in 0..14 {
                    let fine = width / 20.0;
                    for k in 0..=40 {
                        let t = (best - width + fine * f64::from(k)).clamp(half.start, half.end);
                        if m(t) > m(best) {
                            best = t;
                        }
                    }
                    width = fine;
                }
                if inside(best) {
                    let lo = (best - 2.0 * width * 20.0).max(half.start);
                    let hi = (best + 2.0 * width * 20.0).min(half.end);
                    let start = if inside(lo) {
                        lo
                    } else {
                        change_to_the_last_bit(inside, lo, best)
                    };
                    let end = if inside(hi) {
                        hi
                    } else {
                        change_pair(inside, best, hi).0
                    };
                    return ReferenceSpan::Zone {
                        start,
                        end,
                        from_start: false,
                        to_end: false,
                    };
                }
                let kouba = !self.p.shadow(noon) && self.p.block != ReferenceBlock::Glonass;
                let c = half.centre;
                if kouba
                    && libm::tan(self.orbit.beta_at(c)).abs()
                        < self.orbit.orbital_rate(c) / self.p.rate
                {
                    return ReferenceSpan::AtCentre;
                }
                return ReferenceSpan::None;
            }
        };
        if self.p.shadow(noon) || self.p.block == ReferenceBlock::Glonass {
            assert_eq!(final_ - first + 1, hits.len(), "one run of the zone");
        }
        let start = if first == 0 {
            half.start
        } else {
            change_to_the_last_bit(inside, scan[first - 1], scan[first])
        };
        let end = if final_ == last {
            half.end
        } else {
            change_pair(inside, scan[final_], scan[final_ + 1]).0
        };
        ReferenceSpan::Zone {
            start,
            end,
            from_start: first == 0,
            to_end: final_ == last,
        }
    }

    /// The turn of a half orbit, or none.
    fn turn(&self, half: &ReferenceHalf) -> Option<SolvedTurn> {
        let key = half.start.to_bits();
        if let Some(turn) = self.turns.borrow().get(&key) {
            return *turn;
        }
        let turn = self.solve_turn(half);
        self.turns.borrow_mut().insert(key, turn);
        turn
    }

    fn solve_turn(&self, half: &ReferenceHalf) -> Option<SolvedTurn> {
        let o = &self.orbit;
        let p = &self.p;
        let noon = half.noon;
        let (start, zone_end, at_centre) = match self.zone(half) {
            ReferenceSpan::None => return None,
            ReferenceSpan::AtCentre => (half.centre, half.centre, true),
            ReferenceSpan::Zone {
                start,
                end,
                from_start,
                to_end,
            } => {
                if p.block == ReferenceBlock::Glonass && noon && (from_start || to_end) {
                    return Some(SolvedTurn {
                        start,
                        zone_end: end,
                        end,
                        held: false,
                        psi0: 0.0,
                        rate: 0.0,
                        after_rate: 0.0,
                        mode: AttitudeMode::NoonTurn,
                        refused: true,
                        hold: None,
                    });
                }
                let kouba = !p.shadow(noon) && p.block != ReferenceBlock::Glonass;
                if kouba && o.beta_at(start) == 0.0 && o.beta_at(end) == 0.0 && end - start < 1e-6 {
                    // Kouba's zone at beta = 0 is the centre: the instants
                    // where the argument rounds to it.
                    (start, end, true)
                } else if end > start {
                    (start, end, false)
                } else if p.shadow(noon) || p.block == ReferenceBlock::Glonass {
                    // One instant of a grazing shadow or span: no turn.
                    return None;
                } else {
                    // One instant of Kouba's zone: turned from the centre.
                    (half.centre, half.centre, true)
                }
            }
        };
        // The nominal yaw just inside the zone's ends (at beta = 0, around
        // the flip at the centre, its limits).
        let psi0 = if at_centre {
            if noon {
                0.0
            } else {
                PI
            }
        } else {
            o.nominal_yaw(start)
        };
        let psi_end = if at_centre {
            o.nominal_yaw(zone_end + 1e-9)
        } else {
            o.nominal_yaw(zone_end)
        };
        let beta = o.beta_at(start);
        let sign_of = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
        let r = p.rate;
        if p.block == ReferenceBlock::Glonass {
            let (rate, mode, hold) = if noon {
                (-r * sign_of(beta), AttitudeMode::NoonTurn, None)
            } else {
                (r * sign_of(beta), AttitudeMode::Shadow, Some(psi_end))
            };
            return Some(SolvedTurn {
                start,
                zone_end,
                end: zone_end,
                held: true,
                psi0,
                rate,
                after_rate: 0.0,
                mode,
                refused: false,
                hold,
            });
        }
        if p.block == ReferenceBlock::Iif && !noon {
            return Some(SolvedTurn {
                start,
                zone_end,
                end: zone_end,
                held: false,
                psi0,
                rate: (psi_end - psi0) / (zone_end - start),
                after_rate: 0.0,
                mode: AttitudeMode::Shadow,
                refused: false,
                hold: None,
            });
        }
        // GPS turns that continue until they meet the nominal yaw.
        enum Approach {
            Trailing,
            Onward,
            Shorter,
        }
        let (rate, mode, approach) = if noon {
            let reversed = p.bias != 0.0
                && (beta == 0.0 || (beta * p.bias > 0.0 && beta.abs() <= p.bias.abs()));
            if reversed {
                let approach = if beta == 0.0 {
                    Approach::Trailing
                } else {
                    Approach::Onward
                };
                (r * sign_of(p.bias), AttitudeMode::NoonTurn, approach)
            } else {
                (
                    -r * sign_of(beta),
                    AttitudeMode::NoonTurn,
                    Approach::Trailing,
                )
            }
        } else if p.block == ReferenceBlock::Iir {
            (
                r * sign_of(beta),
                AttitudeMode::MidnightTurn,
                Approach::Trailing,
            )
        } else {
            (r * sign_of(p.bias), AttitudeMode::Shadow, Approach::Shorter)
        };
        let exit = psi0 + rate * (zone_end - start);
        let (after_rate, gap0) = match approach {
            Approach::Trailing => {
                let a = sign_of(rate) * (psi_end - exit);
                let a = a - TAU * libm::floor((a + FRAC_PI_2) / TAU);
                (rate, a.max(0.0))
            }
            Approach::Onward => {
                let a = sign_of(rate) * (psi_end - exit);
                (rate, a - TAU * libm::floor(a / TAU))
            }
            Approach::Shorter => {
                let dfr = wrap_pi(psi_end - exit);
                (r * sign_of(dfr), dfr.abs())
            }
        };
        let limit = zone_end + 1800.0;
        let (end, held) = if gap0 == 0.0 {
            (zone_end, false)
        } else {
            let from = if at_centre { zone_end + 1e-9 } else { zone_end };
            match reference_meeting(
                &|t| o.nominal_yaw(t),
                from,
                limit,
                gap0,
                sign_of(after_rate),
                after_rate.abs(),
            ) {
                Some(met) => (met, false),
                None => (limit, true),
            }
        };
        Some(SolvedTurn {
            start,
            zone_end,
            end,
            held,
            psi0,
            rate,
            after_rate,
            mode,
            refused: false,
            hold: None,
        })
    }

    fn attitude(&self, t: f64) -> ReferenceAttitude {
        let o = &self.orbit;
        let here = ReferenceHalf::containing(o, t);
        if self.p.block == ReferenceBlock::Glonass {
            if !here.noon && self.glonass_noon_margin(t) > 0.0 {
                return ReferenceAttitude::Refused;
            }
            return match self.turn(&here) {
                Some(turn) if turn.refused => {
                    if turn.start <= t && t <= turn.zone_end {
                        ReferenceAttitude::Refused
                    } else {
                        ReferenceAttitude::Nominal
                    }
                }
                Some(turn) if turn.start <= t && t <= turn.zone_end => {
                    let linear = turn.psi0 + turn.rate * (t - turn.start);
                    let yaw = match turn.hold {
                        Some(h) if turn.rate > 0.0 => linear.min(h),
                        Some(h) => linear.max(h),
                        None => linear,
                    };
                    let slope = if yaw == linear { turn.rate.abs() } else { 0.0 };
                    ReferenceAttitude::Yaw(turn.mode, yaw, slope, turn.start)
                }
                _ => ReferenceAttitude::Nominal,
            };
        }
        let mut half = here;
        for back in 0..64 {
            if back > 0 && t - half.end > 1800.0 {
                return ReferenceAttitude::Nominal;
            }
            if let Some(turn) = self.turn(&half).filter(|turn| turn.start <= t) {
                if self.p.block == ReferenceBlock::Iif && turn.mode == AttitudeMode::Shadow {
                    return if t < turn.zone_end {
                        ReferenceAttitude::Yaw(
                            turn.mode,
                            turn.psi0 + turn.rate * (t - turn.start),
                            turn.rate.abs(),
                            turn.start,
                        )
                    } else {
                        ReferenceAttitude::Nominal
                    };
                }
                if t <= turn.zone_end {
                    if t == turn.zone_end && turn.end == turn.zone_end && !turn.held {
                        return ReferenceAttitude::Nominal;
                    }
                    return ReferenceAttitude::Yaw(
                        turn.mode,
                        turn.psi0 + turn.rate * (t - turn.start),
                        turn.rate.abs(),
                        turn.start,
                    );
                }
                let running = t < turn.end || (turn.held && t <= turn.end);
                return if running {
                    let exit = turn.psi0 + turn.rate * (turn.zone_end - turn.start);
                    ReferenceAttitude::Yaw(
                        turn.mode,
                        exit + turn.after_rate * (t - turn.zone_end),
                        turn.rate.abs().max(turn.after_rate.abs()),
                        turn.start,
                    )
                } else {
                    ReferenceAttitude::Nominal
                };
            }
            half = half.previous(o);
        }
        panic!("no half orbit without a turn");
    }

    /// The instants at which the turns of the half orbits overlapping
    /// `[from, to]`, and of the earlier ones that end less than 1800 s before
    /// `from`, start, end their zones and meet the nominal yaw, with the
    /// half orbits' starts and centres.
    fn boundaries(&self, from: f64, to: f64) -> Vec<f64> {
        let mut out = Vec::new();
        for half in sweep_halves(&self.orbit, from, to) {
            if let Some(turn) = self.turn(&half) {
                out.extend([turn.start, turn.zone_end, turn.end]);
            }
            out.extend([half.start, half.centre]);
        }
        out
    }
}

/// The first instant after `from` at which
/// `gap0 + d (psi(t) - psi(from)) - speed (t - from)` is no longer positive,
/// `psi` the nominal yaw followed continuously, or `None` before `limit`:
/// steps of at most 1 s, quartered while the nominal yaw moves by more than
/// 0.1 rad over one (down to one unit in the last place) or while the gap at
/// either end is within twice its largest change over the step of zero (down
/// to 1e-4 s), the crossing bisected to the last bit.
fn reference_meeting(
    nominal: &dyn Fn(f64) -> f64,
    from: f64,
    limit: f64,
    gap0: f64,
    d: f64,
    speed: f64,
) -> Option<f64> {
    let gap = |unwrapped: f64, t: f64| gap0 + d * unwrapped - speed * (t - from);
    let (mut t, mut psi, mut unwrapped) = (from, nominal(from), 0.0);
    let mut step = 1.0_f64;
    while t < limit {
        let next = (t + step).min(limit).max(t.next_up());
        let psi_next = nominal(next);
        let moved = wrap_pi(psi_next - psi);
        let g_next = gap(unwrapped + moved, next);
        let g_here = gap(unwrapped, t);
        let reach = moved.abs() + speed * (next - t);
        let finer = next > t.next_up();
        if finer && moved.abs() > 0.1 {
            step *= 0.25;
            continue;
        }
        if finer && step > 1e-4 && g_next > 0.0 && g_here.min(g_next) < 2.0 * reach {
            step *= 0.25;
            continue;
        }
        if g_next <= 0.0 {
            let (base, base_psi) = (unwrapped, psi);
            let closed = |x: f64| gap(base + wrap_pi(nominal(x) - base_psi), x) <= 0.0;
            return Some(change_to_the_last_bit(closed, t, next));
        }
        t = next;
        psi = psi_next;
        unwrapped += moved;
        step = (step * 2.0).min(1.0);
    }
    None
}

/// Largest change of beta per half period drawn by the sweeps: just inside
/// the laws' bound of 0.05 rad over the orbit before an instant.
const SWEEP_BETA_CHANGE: f64 = 0.0249;

/// The GLONASS-M noon-zone eccentricity bound for the largest `|beta|` over
/// a noon half orbit: 0.25 below 30 deg, 0.2 below 60 deg and 0.1 above.
fn glonass_noon_bound(largest_beta: f64) -> f64 {
    if largest_beta < deg(30.0) {
        0.25
    } else if largest_beta < deg(60.0) {
        0.2
    } else {
        0.1
    }
}

/// The largest orbital rate of a [`Kepler`] orbit, at perigee:
/// `n (1 + e)^2 / (1 - e^2)^(3/2)`.
fn peak_rate(orbit: &Kepler) -> f64 {
    let e = orbit.e;
    orbit.n() * (1.0 + e) * (1.0 + e) / libm::pow(1.0 - e * e, 1.5)
}

/// One case of the GPS and GLONASS-M sweeps: a [`Kepler`] orbit with its
/// noon (`noon == true`) or midnight at `t_centre` and beta `beta_c` there,
/// moving by `beta_change` per half period, the law's parameters, and whether
/// the trajectory is turned into a general orientation.
#[derive(Clone, Copy, Debug)]
struct SweepCase {
    block: ReferenceBlock,
    a: f64,
    e: f64,
    omega: f64,
    beta_c: f64,
    beta_change: f64,
    rate: f64,
    bias: f64,
    limit: f64,
    noon: bool,
    t_centre: f64,
    tilt: bool,
}

impl SweepCase {
    fn orbit(&self) -> Kepler {
        let probe = Kepler {
            a: self.a,
            e: self.e,
            beta: 0.0,
            beta_rate: 0.0,
            omega: self.omega,
            m0: 0.0,
        };
        let beta_rate = self.beta_change * probe.n() / PI;
        let centre_u = if self.noon { 0.0 } else { PI };
        let nu = centre_u - self.omega;
        let e = self.e;
        let ea = 2.0
            * libm::atan2(
                (1.0 - e).sqrt() * libm::sin(0.5 * nu),
                (1.0 + e).sqrt() * libm::cos(0.5 * nu),
            );
        let m_centre = ea - e * libm::sin(ea);
        Kepler {
            beta: self.beta_c - beta_rate * self.t_centre,
            beta_rate,
            m0: m_centre - probe.n() * self.t_centre,
            ..probe
        }
    }

    /// The half orbit around the centre, the one before it, and the range
    /// of the grid of instants checked (from the start of the half orbit
    /// before to 1800 s after the end of the centre's).
    fn span(&self, orbit: &Kepler) -> (ReferenceHalf, ReferenceHalf, f64, f64) {
        let half = ReferenceHalf::containing(orbit, self.t_centre);
        let before = half.previous(orbit);
        (half, before, before.start, half.end + 1800.0)
    }
}

/// The half orbits overlapping `[from, to]` and the earlier ones that end
/// less than 1800 s before `from` (whose turns can reach it), the latest
/// first: the half orbits whose turn boundaries the sweeps probe.
fn sweep_halves(orbit: &Kepler, from: f64, to: f64) -> Vec<ReferenceHalf> {
    let mut half = ReferenceHalf::containing(orbit, to);
    let mut out = Vec::new();
    while half.end >= from - 1800.0 {
        out.push(half);
        half = half.previous(orbit);
    }
    out
}

/// Beta at the centre whose largest `|beta|` over the noon half orbits of
/// [`sweep_halves`] is `largest` (beta moving at a constant rate, so it peaks
/// at an end of them), of sign `sign`; where beta changes sign over them,
/// `|beta|` stays below `largest` or below its change over them.
fn beta_for_largest(case: &SweepCase, largest: f64, sign: f64) -> f64 {
    let orbit = SweepCase {
        beta_c: 0.0,
        ..*case
    }
    .orbit();
    let (_, _, from, to) = case.span(&orbit);
    let noon_halves: Vec<ReferenceHalf> = sweep_halves(&orbit, from, to)
        .into_iter()
        .filter(|h| h.noon)
        .collect();
    let lo = noon_halves
        .iter()
        .map(|h| h.start)
        .fold(f64::INFINITY, f64::min);
    let hi = noon_halves
        .iter()
        .map(|h| h.end)
        .fold(f64::NEG_INFINITY, f64::max);
    if orbit.beta_rate == 0.0 || noon_halves.is_empty() {
        return sign * largest;
    }
    let growing = if sign * orbit.beta_rate > 0.0 {
        hi - case.t_centre
    } else {
        case.t_centre - lo
    };
    sign * (largest - orbit.beta_rate.abs() * growing.max(0.0))
}

/// The law against [`GpsReference`] for one case, at instants around its
/// centre: a grid of 17 over the half orbit before the centre's, the centre's
/// and the 1800 s after, and probes 1e-4 s either side of every turn
/// boundary of the half orbits of [`sweep_halves`] over that range (the next
/// half orbit's too), none within 1e-5 s of one. Returns the number of
/// instants checked, the number the law refused as outside the validated
/// domain, and a description of each disagreement.
///
/// A yaw agrees within 1e-9 rad plus the reference yaw's rate of change
/// times the uncertainty of the turn's start instant (a held yaw has none):
/// `4e-15` times the larger magnitude of the instant and the start (a few
/// units in their last place) and 1e-10 s, the rounding of a zone's
/// condition (about 1e-15) over its rate of change. The
/// law may refuse with [`GnssAttitudeError::OutsideValidatedDomain`] only
/// where the case lies outside the domain by the closed form: an
/// eccentricity above 0.3 (GPS) or above the GLONASS-M noon bound for the
/// largest `|beta|` of the instant's noon half orbit; beta changing sign
/// within one orbit and 1800 s of the instants checked with a maximum yaw
/// rate below the orbit's largest orbital rate (GPS); and a GPS IIA rate
/// above 1e5 times the orbital rate at the instant.
fn run_sweep_case(case: &SweepCase, label: &str) -> (usize, usize, Vec<String>) {
    let orbit = case.orbit();
    let p = ReferenceParameters {
        block: case.block,
        rate: case.rate,
        bias: case.bias,
        limit: case.limit,
    };
    let law = p.law();
    let reference = GpsReference::new(orbit, p);
    let plain = orbit.trajectory();
    let turned = tilted(orbit.trajectory());
    let trajectory: &dyn Fn(f64) -> Option<AttitudeSample> =
        if case.tilt { &turned } else { &plain };
    let (_, _, from, to) = case.span(&orbit);
    let mut instants: Vec<f64> = (0..=16)
        .map(|k| from + (to - from) * f64::from(k) / 16.0)
        .collect();
    let boundaries = reference.boundaries(from, to);
    for &b in &boundaries {
        instants.extend([b - 1e-4, b + 1e-4]);
    }
    let glonass = case.block == ReferenceBlock::Glonass;
    let period = orbit.period();
    let sign_change = orbit.beta_rate != 0.0 && {
        let t_zero = -orbit.beta / orbit.beta_rate;
        from - 1800.0 - period <= t_zero && t_zero <= to + 1800.0 + period
    };
    let slow_at_sign_change = !glonass && sign_change && case.rate < peak_rate(&orbit) * 1.000_001;
    let eccentricity_bound = |t: f64| {
        if !glonass {
            return 0.3;
        }
        let half = ReferenceHalf::containing(&orbit, t);
        if half.noon {
            glonass_noon_bound(
                orbit
                    .beta_at(half.start)
                    .abs()
                    .max(orbit.beta_at(half.end).abs()),
            )
        } else {
            0.9
        }
    };
    let allowed = |t: f64, reason: &str| {
        (reason.starts_with("orbit eccentricity above") && case.e > eccentricity_bound(t))
            || (slow_at_sign_change
                && reason.contains("below the orbital rate where beta changes sign"))
            || (case.block == ReferenceBlock::Iia
                && reason.starts_with("GPS IIA maximum yaw rate")
                && case.rate > 1e5 * orbit.orbital_rate(t) * (1.0 - 1e-9))
    };
    let (mut checked, mut outside) = (0, 0);
    let mut failures = Vec::new();
    for t in instants {
        let near = boundaries
            .iter()
            .any(|&b| (t - b).abs() < 1e-5_f64.max(1e-12 * t.abs()));
        if near {
            continue;
        }
        let expected = reference.attitude(t);
        let result = gnss_attitude(&law, trajectory, t);
        let agrees = match (&expected, &result) {
            (ReferenceAttitude::Nominal, Ok(att)) => att.mode == AttitudeMode::Nominal,
            (ReferenceAttitude::Yaw(mode, yaw, slope, start), Ok(att)) => {
                let tolerance = 1e-9 + slope * (4e-15 * t.abs().max(start.abs()).max(1.0) + 1e-10);
                att.mode == *mode && wrap_pi(att.yaw_rad - yaw).abs() <= tolerance
            }
            (
                ReferenceAttitude::Refused,
                Err(GnssAttitudeError::InvalidParameter {
                    field: "yaw_rate_rad_s",
                    ..
                }),
            ) => true,
            (_, Err(GnssAttitudeError::OutsideValidatedDomain { reason, .. })) => {
                allowed(t, reason)
            }
            _ => false,
        };
        checked += 1;
        outside += usize::from(outside_domain(&result));
        if !agrees {
            failures.push(format!(
                "{label} {:?} a {} e {:.5} omega {:.1} beta_c {:e} change {:.4} rate/n {:e} bias {:.3} limit {:.2} noon {} tilt {} t - centre {:.6}: reference {expected:?} law {:?}",
                case.block,
                case.a,
                case.e,
                case.omega.to_degrees(),
                case.beta_c,
                case.beta_change,
                case.rate / orbit.n(),
                case.bias.to_degrees(),
                case.limit.to_degrees(),
                case.noon,
                case.tilt,
                t - case.t_centre,
                result.map(|a| (a.mode, a.yaw_rad))
            ));
        }
    }
    (checked, outside, failures)
}

/// Sample `i` of a quasi-random sweep of the validated domain for `block`
/// ([`run_sweep_case`]). The sample draws a GPS (or GLONASS) orbit or a fast
/// one (radius 4500 km); the eccentricity (GPS: up to 0.3, exactly 0.3, 0, or
/// from 0.3 to 0.3003, past the bound's allowance of 2e-4; GLONASS-M: a tier
/// of the noon bound, 0.25 with the largest `|beta|` over the noon half
/// orbits checked below 30 deg, 0.2 below 60 deg or 0.1 up to 89.9 deg, and
/// an eccentricity up to the tier's bound, exactly at it, or from it to
/// 3e-4 above); the argument of perigee from noon; beta (exactly 0 for every
/// fifth sample; GPS: at the centre, of either sign up to 89.9 deg less six
/// times its change per half period; GLONASS-M: its largest magnitude over
/// the noon half orbits checked within the tier, 1e-9 rad below the tier's
/// limit for every third sample; both concentrated toward zero otherwise);
/// beta's rate (0 for every fourth sample, otherwise up to
/// [`SWEEP_BETA_CHANGE`] per half period, either way); the maximum yaw rate
/// over the mean motion (log-uniform from 1e-9, or 1e-2 at beta = 0, to 1e12,
/// or to 2e5 for GPS IIA, beyond its cap of 1e5 times the orbital rate,
/// which is 1.95e5 times the mean motion at perigee at eccentricity 0.3, and
/// the extremes 5e-324 rad/s, 1e-300 rad/s, 1e300 and the largest float); the
/// yaw bias (published, within 2.25 deg, within 89 deg, or 1e6 rad) and
/// shadow limit (published or from 90.01 to 179.99 deg); and the time of the
/// centre. A third of the orbits with beta away from zero are turned into a
/// general orientation.
fn gps_sweep_sample(block: ReferenceBlock, i: u32) -> (usize, usize, Vec<String>) {
    let h = |dim: usize| halton(i + 1, dim);
    let published = GnssAttitudeLaw::for_block(
        match block {
            ReferenceBlock::Iia => GnssBlock::GpsIIA,
            ReferenceBlock::Iir => GnssBlock::GpsIIR,
            ReferenceBlock::Iif => GnssBlock::GpsIIF,
            ReferenceBlock::Glonass => GnssBlock::GlonassM,
        },
        Some(11),
    )
    .unwrap();
    let (pub_rate, pub_bias, pub_limit) = match published {
        GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        }
        | GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        } => (yaw_rate_rad_s, yaw_bias_rad, shadow_limit_rad),
        GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s } => (yaw_rate_rad_s, 0.0, PI - 0.1),
        GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s,
            shadow_limit_rad,
        } => (yaw_rate_rad_s, 0.0, shadow_limit_rad),
        _ => unreachable!(),
    };
    let glonass = block == ReferenceBlock::Glonass;
    let a = if h(7) < 1.0 / 3.0 {
        4_500_000.0
    } else if glonass {
        GLONASS_A
    } else {
        GPS_A
    };
    let n = (GM / (a * a * a)).sqrt();
    let zero_beta = i.is_multiple_of(5);
    let beta_draw = h(2);
    let sign = if (beta_draw * 1000.0).fract() < 0.5 {
        1.0
    } else {
        -1.0
    };
    let beta_change = if zero_beta || i.is_multiple_of(4) {
        0.0
    } else {
        SWEEP_BETA_CHANGE * (2.0 * h(3) - 1.0)
    };
    // The eccentricity, and for GLONASS-M the largest |beta| of the tier.
    let (e, largest_max) = if glonass {
        let tier = h(0);
        let (e_max, beta_max) = if tier < 0.4 {
            (0.25, deg(30.0))
        } else if tier < 0.7 {
            (0.2, deg(60.0))
        } else {
            (0.1, deg(89.9) - 6.0 * beta_change.abs())
        };
        let e = match i % 7 {
            1 => e_max,
            4 => e_max + 3e-4 * (tier * 13.0).fract(),
            _ => e_max * (tier * 10.0).fract(),
        };
        (e, beta_max)
    } else {
        let e = match i % 7 {
            1 => 0.3,
            2 => 0.0,
            4 => 0.3 + 3e-4 * (h(0) * 13.0).fract(),
            _ => 0.3 * h(0),
        };
        (e, deg(89.9) - 6.0 * beta_change.abs())
    };
    let omega = TAU * h(1);
    // The maximum yaw rate over the mean motion.
    let rate = match i % 23 {
        3 => pub_rate,
        5 if block != ReferenceBlock::Iia => f64::MAX,
        7 if block != ReferenceBlock::Iia => 1e300 * n,
        11 if !zero_beta => 5e-324,
        13 if !zero_beta => 1e-300,
        _ => {
            let (lo, hi) = match (block, zero_beta) {
                (ReferenceBlock::Iia, true) => (-2.0, 5.301),
                (ReferenceBlock::Iia, false) => (-9.0, 5.301),
                (_, true) => (-2.0, 12.0),
                (_, false) => (-9.0, 12.0),
            };
            n * libm::pow(10.0, lo + (hi - lo) * h(4))
        }
    };
    // The yaw bias: published, within 2.25 deg, within 89 deg, or 1e6 rad of
    // either sign; the shadow limit: published or anywhere in (90, 180) deg.
    let draw = h(5);
    let bias = if matches!(block, ReferenceBlock::Iir | ReferenceBlock::Glonass) {
        0.0
    } else if draw < 1.0 / 3.0 {
        pub_bias
    } else if draw < 2.0 / 3.0 {
        deg(4.5 * (3.0 * draw - 1.0) - 2.25)
    } else if draw < 5.0 / 6.0 {
        deg(178.0 * (6.0 * draw - 4.0) - 89.0)
    } else if (draw * 1000.0).fract() < 0.5 {
        1e6
    } else {
        -1e6
    };
    let limit = if h(6) < 0.5 {
        pub_limit
    } else {
        deg(90.01 + 89.98 * (h(6) * 2.0).fract())
    };
    let noon = (h(8) * 10.0).fract() < 0.5;
    let t_centre = 3.0e4 * (2.0 * h(8) - 1.0);
    let mut case = SweepCase {
        block,
        a,
        e,
        omega,
        beta_c: 0.0,
        beta_change,
        rate,
        bias,
        limit,
        noon,
        t_centre,
        tilt: false,
    };
    if !zero_beta {
        case.beta_c = if glonass {
            let largest = if i.is_multiple_of(3) {
                largest_max - 1e-9
            } else {
                largest_max * libm::pow(beta_draw, 3.0)
            };
            beta_for_largest(&case, largest, sign)
        } else {
            // Beta stays below 89.9 deg over the half orbits checked.
            sign * largest_max * libm::pow(beta_draw, 3.0)
        };
    }
    case.tilt = !zero_beta && case.beta_c.abs() > 1e-6 && (h(9) * 3.0) < 1.0;
    run_sweep_case(&case, &format!("sample {i}"))
}

fn run_gps_sweep(block: ReferenceBlock, samples: u32) {
    let (mut checked, mut outside) = (0, 0);
    let mut failures = Vec::new();
    for i in 0..samples {
        let (c, o, f) = gps_sweep_sample(block, i);
        checked += c;
        outside += o;
        failures.extend(f);
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} instants disagree:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(checked > 20 * samples as usize, "{checked}");
    assert!(outside * 10 < checked, "{outside} of {checked} refused");
}

/// GPS IIA over its validated domain against [`GpsReference`]
/// ([`gps_sweep_sample`]): eccentricity up to 0.3 and past its allowance (to
/// 0.3003), any argument of perigee, beta of either sign and exactly 0, beta
/// moving by up to 0.0249 rad per half period, maximum yaw rates from
/// 1e-300 rad/s to 2e5 times the mean motion (1e-2 at beta = 0), past the cap
/// of 1e5 times the orbital rate, yaw biases up to 89 deg and of 1e6 rad, and
/// shadow limits from 90.01 to 179.99 deg.
#[test]
fn gps_iia_follows_the_reference_over_its_validated_domain() {
    run_gps_sweep(ReferenceBlock::Iia, 300);
}

/// GPS IIR over its validated domain ([`gps_sweep_sample`]), with maximum yaw
/// rates from 1e-300 rad/s to the largest float.
#[test]
fn gps_iir_follows_the_reference_over_its_validated_domain() {
    run_gps_sweep(ReferenceBlock::Iir, 300);
}

/// GPS IIF over its validated domain ([`gps_sweep_sample`]), with maximum yaw
/// rates from 1e-300 rad/s to the largest float, yaw biases up to 89 deg and
/// of 1e6 rad, and shadow limits from 90.01 to 179.99 deg.
#[test]
fn gps_iif_follows_the_reference_over_its_validated_domain() {
    run_gps_sweep(ReferenceBlock::Iif, 300);
}

/// GLONASS-M over its validated domain ([`gps_sweep_sample`]): eccentricity up
/// to 0.25 with `|beta|` below 30 deg, 0.2 below 60 deg and 0.1 above (to
/// 3e-4 past each), up to 1e-9 rad below each tier's beta limit, with
/// maximum yaw rates from 1e-300 rad/s to the largest float (the noon turn
/// refused where it reaches the quadrature) and shadow limits from 90.01 to
/// 179.99 deg.
#[test]
fn glonass_m_follows_the_reference_over_its_validated_domain() {
    run_gps_sweep(ReferenceBlock::Glonass, 450);
}

/// GLONASS-M at the corners of its noon bounds against [`GpsReference`]
/// ([`run_sweep_case`]): eccentricity 0.25 and 0.2502 with the largest
/// `|beta|` over the noon half orbits checked 1e-9 rad below 30 deg, and 0.2
/// and 0.2002 with it 1e-9 rad below 60 deg, the argument of perigee from
/// noon every 8 deg, beta fixed and moving by 0.0249 rad per half period
/// either way, around a noon, with maximum yaw rates of 0.8 and 0.95 times
/// the orbital rate at noon over `tan|beta|` there (so that the noon zone
/// holds, with half spans from about 15 deg to 60 deg).
#[test]
fn glonass_m_follows_the_reference_at_the_corners_of_its_noon_bounds() {
    let (mut checked, mut outside) = (0, 0);
    let mut failures = Vec::new();
    for (bound, beta_limit) in [(0.25, deg(30.0)), (0.2, deg(60.0))] {
        for e in [bound, bound + 2e-4] {
            for k in 0..45 {
                for change in [0.0, SWEEP_BETA_CHANGE, -SWEEP_BETA_CHANGE] {
                    let mut case = SweepCase {
                        block: ReferenceBlock::Glonass,
                        a: GLONASS_A,
                        e,
                        omega: deg(8.0 * f64::from(k)),
                        beta_c: 0.0,
                        beta_change: change,
                        rate: 0.0,
                        bias: 0.0,
                        limit: GLONASS_M_SHADOW_LIMIT_RAD,
                        noon: true,
                        t_centre: 1234.5,
                        tilt: k % 3 == 1,
                    };
                    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                    case.beta_c = beta_for_largest(&case, beta_limit - 1e-9, sign);
                    let orbit = case.orbit();
                    let f = if (k / 2) % 2 == 0 { 0.8 } else { 0.95 };
                    case.rate =
                        f * orbit.orbital_rate(case.t_centre) / libm::tan(case.beta_c.abs());
                    let (c, o, more) = run_sweep_case(&case, &format!("corner {k}"));
                    checked += c;
                    outside += o;
                    failures.extend(more);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} instants disagree:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(checked > 10_000, "{checked}");
    // Only the cases 2e-4 past each bound can be refused.
    assert!(outside * 2 < checked, "{outside} of {checked} refused");
}

/// The Galileo IOV law (Galileo IOV and GPS III) on a [`Kepler`] orbit from
/// its closed form: the region `|S_x| < sin(beta_x)`, `|S_y| < sin(beta_y)`
/// with `S_x = -sin(u) cos(beta)`, `S_y = -sin(beta)`; its beginning, the
/// last change of the region condition before the instant, from a backward
/// scan in steps of a 4000th of the period and bisected to the last bit;
/// `G = sign(S_y)` there and the metadata's yaw (`Some(None)` outside the
/// region, `None` for a region that began more than 3.5 orbits back).
fn iov_reference(
    orbit: &Kepler,
    beta_x: f64,
    beta_y: f64,
    t: f64,
) -> Option<Option<(AttitudeMode, f64)>> {
    let (sin_x, sin_y) = (libm::sin(beta_x), libm::sin(beta_y));
    let s_x = |t: f64| -libm::sin(orbit.argument_from_noon(t)) * libm::cos(orbit.beta_at(t));
    let s_y = |t: f64| -libm::sin(orbit.beta_at(t));
    let in_region = |t: f64| s_x(t).abs() < sin_x && s_y(t).abs() < sin_y;
    if !in_region(t) {
        return Some(None);
    }
    let step = orbit.period() / 4000.0;
    let mut lo = t;
    while in_region(lo - step) {
        lo -= step;
        if t - lo > 3.5 * orbit.period() {
            // The law looks back four orbits: no beginning to compare.
            return None;
        }
    }
    let entry = change_to_the_last_bit(in_region, lo - step, lo);
    let gamma = 1.0_f64.copysign(s_y(entry));
    let (x, y) = (s_x(t), s_y(t));
    let s_hy =
        0.5 * (sin_y * gamma + y) + 0.5 * (sin_y * gamma - y) * libm::cos(PI * x.abs() / sin_x);
    let mode = if libm::cos(orbit.argument_from_noon(t)) > 0.0 {
        AttitudeMode::NoonTurn
    } else {
        AttitudeMode::MidnightTurn
    };
    Some(Some((mode, libm::atan2(s_hy, x))))
}

/// The Galileo IOV law (Galileo IOV and GPS III) over its validated domain
/// against [`iov_reference`]: a quasi-random sweep of eccentricity up to 0.9,
/// the argument of perigee from noon, beta (exactly 0 for every fifth
/// sample, otherwise of either sign up to 1.5 times `beta_y`), beta's rate
/// (0 for every fourth sample, otherwise up to [`SWEEP_BETA_CHANGE`] per half
/// period, within the laws' 0.05 rad over the orbit before an instant), the
/// published limits of both blocks and custom
/// ones (`beta_x` 3 to 85 deg, `beta_y` 0.5 to 20 deg), at 48 instants over
/// an orbit and 1e-4 s either side of each region boundary, none within
/// 1e-5 s of one.
#[test]
fn galileo_iov_law_follows_the_reference_over_its_validated_domain() {
    let mut checked = 0;
    let mut failures = Vec::new();
    for i in 0..300_u32 {
        let h = |dim: usize| halton(i + 1, dim);
        let (beta_x, beta_y, a) = match i % 3 {
            0 => (GALILEO_IOV_BETA_X_RAD, GALILEO_IOV_BETA_Y_RAD, GALILEO_A),
            1 => (GPS_III_BETA_X_RAD, GPS_III_BETA_Y_RAD, GPS_A),
            _ => (deg(3.0 + 82.0 * h(5)), deg(0.5 + 19.5 * h(6)), GPS_A),
        };
        let e = if i % 7 == 1 { 0.9 } else { 0.9 * h(0) };
        let omega = TAU * h(1);
        let sign = if (h(2) * 1000.0).fract() < 0.5 {
            1.0
        } else {
            -1.0
        };
        let beta_c = if i.is_multiple_of(5) {
            0.0
        } else {
            sign * 1.5 * beta_y * libm::pow(h(2), 2.0)
        };
        let probe = Kepler {
            a,
            e,
            beta: 0.0,
            beta_rate: 0.0,
            omega,
            m0: TAU * h(7),
        };
        let beta_rate = if i.is_multiple_of(4) || i.is_multiple_of(5) {
            0.0
        } else {
            SWEEP_BETA_CHANGE * (2.0 * h(3) - 1.0) * probe.n() / PI
        };
        let t0 = 3.0e4 * (2.0 * h(4) - 1.0);
        let orbit = Kepler {
            beta: beta_c - beta_rate * t0,
            beta_rate,
            ..probe
        };
        let law = GnssAttitudeLaw::GalileoIov {
            beta_x_rad: beta_x,
            beta_y_rad: beta_y,
        };
        let tilt = beta_c.abs() > 1e-6 && i % 3 != 2;
        let plain = orbit.trajectory();
        let turned = tilted(orbit.trajectory());
        let trajectory: &dyn Fn(f64) -> Option<AttitudeSample> =
            if tilt { &turned } else { &plain };
        let period = orbit.period();
        let (sin_x, sin_y) = (libm::sin(beta_x), libm::sin(beta_y));
        let in_region = |t: f64| {
            (libm::sin(orbit.argument_from_noon(t)) * libm::cos(orbit.beta_at(t))).abs() < sin_x
                && libm::sin(orbit.beta_at(t)).abs() < sin_y
        };
        // Region boundaries over the orbit from t0, on a scan of 4000 steps.
        let mut boundaries = Vec::new();
        let steps = 4000;
        let mut previous = in_region(t0);
        for k in 1..=steps {
            let t = t0 + period * f64::from(k) / f64::from(steps);
            let now = in_region(t);
            if now != previous {
                boundaries.push(change_to_the_last_bit(
                    in_region,
                    t - period / f64::from(steps),
                    t,
                ));
            }
            previous = now;
        }
        let mut instants: Vec<f64> = (0..48)
            .map(|k| t0 + period * (f64::from(k) + h(8)) / 48.0)
            .collect();
        for &b in &boundaries {
            instants.extend([b - 1e-4, b + 1e-4]);
        }
        for t in instants {
            if boundaries.iter().any(|&b| (t - b).abs() < 1e-5) {
                continue;
            }
            let Some(expected) = iov_reference(&orbit, beta_x, beta_y, t) else {
                continue;
            };
            let result = gnss_attitude(&law, trajectory, t);
            let agrees = match (expected, &result) {
                (None, Ok(att)) => att.mode == AttitudeMode::Nominal,
                (Some((mode, yaw)), Ok(att)) => {
                    att.mode == mode && wrap_pi(att.yaw_rad - yaw).abs() < 1e-9
                }
                _ => false,
            };
            checked += 1;
            if !agrees {
                failures.push(format!(
                    "sample {i} limits {:.2} {:.2} e {e:.3} omega {:.1} beta_c {beta_c:e} rate {beta_rate:e} tilt {tilt} t - t0 {}: reference {expected:?} law {:?}",
                    beta_x.to_degrees(),
                    beta_y.to_degrees(),
                    omega.to_degrees(),
                    t - t0,
                    result.map(|a| (a.mode, a.yaw_rad))
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} instants disagree:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(checked > 10_000, "{checked}");
}

/// A cosine turn (Galileo FOC, BeiDou-3 CAST and IGSO) on a [`Kepler`] orbit
/// from its closed form: for each noon and midnight, the latest first, its
/// start window (the in-plane angle at the start value before the centre),
/// its start (the window for the orbit angle; for the colinearity
/// `acos(cos(beta) cos(x))` its first fall to the start value after the
/// window and before the centre, on a scan of 4000 points with the scan's
/// lowest point zoomed in on, bisected to the last bit), and the turn if
/// `|beta|` at the start passes the limit; the latest such turn started by
/// the instant, while half the period has not elapsed (Galileo satellite
/// metadata 3.1.2, Wang et al. 2018).
struct CosineReference {
    orbit: Kepler,
    turn: CosineTurnParameters,
    starts: std::cell::RefCell<std::collections::HashMap<u64, Option<f64>>>,
}

impl CosineReference {
    fn new(orbit: Kepler, turn: CosineTurnParameters) -> Self {
        Self {
            orbit,
            turn,
            starts: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// The passes (centre instant, noon) whose windows lie within
    /// `[from, to]`, the latest first, with their windows.
    fn passes(&self, from: f64, to: f64) -> Vec<(f64, bool, f64)> {
        let o = &self.orbit;
        let period = o.period();
        let s = self.turn.start_angle_rad;
        let mut passes = Vec::new();
        for (noon, centre_u) in [(true, 0.0), (false, PI)] {
            let mut c = o.time_at(centre_u, from - period);
            while c < to + period {
                let window = o.time_at(centre_u - s, c - period);
                if (from..=to).contains(&window) {
                    passes.push((c, noon, window));
                }
                c = o.time_at(centre_u, c + 0.5 * period);
            }
        }
        passes.sort_by(|a, b| b.0.total_cmp(&a.0));
        passes
    }

    /// The start of the pass at centre `c`, if the start angle falls to its
    /// value before the centre.
    fn start(&self, c: f64, noon: bool, window: f64) -> Option<f64> {
        let key = c.to_bits();
        if let Some(start) = self.starts.borrow().get(&key) {
            return *start;
        }
        let o = &self.orbit;
        let s = self.turn.start_angle_rad;
        let x_of = |t: f64| {
            let u = o.argument_from_noon(t);
            if noon {
                -u
            } else {
                wrap_pi(PI - u)
            }
        };
        let start = match self.turn.start_angle {
            CosineTurnStartAngle::OrbitAngle => Some(window),
            CosineTurnStartAngle::Colinearity => {
                let angle = |t: f64| libm::acos(libm::cos(o.beta_at(t)) * libm::cos(x_of(t)));
                let above = |t: f64| angle(t) > s;
                let points = 4000;
                let scan: Vec<f64> = (0..=points)
                    .map(|k| window + (c - window) * f64::from(k) / f64::from(points))
                    .filter(|&t| x_of(t) > 0.0)
                    .collect();
                if !above(window) {
                    Some(window)
                } else if let Some(k) = scan.iter().position(|&t| !above(t)) {
                    Some(change_to_the_last_bit(above, scan[k - 1], scan[k]))
                } else {
                    let mut best = scan
                        .iter()
                        .copied()
                        .min_by(|a, b| angle(*a).total_cmp(&angle(*b)))
                        .unwrap();
                    let last = *scan.last().unwrap();
                    let mut width = (c - window) / f64::from(points);
                    for _ in 0..14 {
                        let fine = width / 20.0;
                        for k in 0..=40 {
                            let x = (best - width + fine * f64::from(k)).clamp(window, last);
                            if angle(x) < angle(best) {
                                best = x;
                            }
                        }
                        width = fine;
                    }
                    (!above(best)).then(|| change_to_the_last_bit(above, window, best))
                }
            }
        };
        self.starts.borrow_mut().insert(key, start);
        start
    }

    fn eligible(&self, start: f64) -> bool {
        let b = self.orbit.beta_at(start).abs();
        if self.turn.beta_limit_inclusive {
            b <= self.turn.beta_limit_rad
        } else {
            b < self.turn.beta_limit_rad
        }
    }

    fn attitude(&self, t: f64) -> Option<(AttitudeMode, f64)> {
        let half_turn = 0.5 * self.turn.period_s;
        let period = self.orbit.period();
        for (c, noon, window) in self.passes(t - half_turn - 2.0 * period, t) {
            if let Some(start) = self.start(c, noon, window).filter(|&start| start <= t) {
                if self.eligible(start) {
                    let elapsed = t - start;
                    if elapsed > half_turn {
                        return None;
                    }
                    let psi0 = self.orbit.nominal_yaw(start);
                    let quarter = if psi0 >= 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
                    let mode = if noon {
                        AttitudeMode::NoonTurn
                    } else {
                        AttitudeMode::MidnightTurn
                    };
                    return Some((
                        mode,
                        quarter + (psi0 - quarter) * libm::cos(TAU * elapsed / self.turn.period_s),
                    ));
                }
            }
            if t - window > half_turn {
                return None;
            }
        }
        None
    }

    /// The instants at which a turn starts or its half period ends over
    /// `[from, to]`.
    fn boundaries(&self, from: f64, to: f64) -> Vec<f64> {
        let half_turn = 0.5 * self.turn.period_s;
        let mut out = Vec::new();
        for (c, noon, window) in self.passes(from - half_turn, to) {
            if let Some(start) = self.start(c, noon, window) {
                if self.eligible(start) {
                    out.extend([start, start + half_turn]);
                }
            }
        }
        out
    }
}

/// The cosine turns over their validated domain against
/// [`CosineReference`]: a quasi-random sweep of eccentricity up to 0.9, the
/// argument of perigee from noon, beta (exactly 0 for every fifth sample,
/// otherwise of either sign up to 1.5 times the limit), beta's rate (0 for
/// every fourth sample, otherwise up to 0.0249 rad per half period), the
/// published Galileo FOC, BeiDou-3 CAST MEO and IGSO turns and custom ones
/// (either start angle from 2 to 60 deg, limits from 0.5 to 15 deg, periods
/// from 0.05 to 0.6 orbital periods), at 48 instants over an orbit and
/// 1e-4 s either side of each turn's start and end, none within 1e-5 s of
/// one.
#[test]
fn cosine_turns_follow_the_reference_over_their_validated_domain() {
    let mut checked = 0;
    let mut failures = Vec::new();
    for i in 0..300_u32 {
        let h = |dim: usize| halton(i + 1, dim);
        let probe_a = |a: f64| TAU / (GM / (a * a * a)).sqrt();
        let (turn, a) = match i % 4 {
            0 => (GALILEO_FOC_TURN, GALILEO_A),
            1 => (BEIDOU3_CAST_MEO_TURN, MEO_A),
            2 => (BEIDOU3_CAST_IGSO_TURN, 42_164_000.0),
            _ => {
                let a = if h(9) < 0.5 { GALILEO_A } else { 8_000_000.0 };
                (
                    CosineTurnParameters {
                        beta_limit_rad: deg(0.5 + 14.5 * h(5)),
                        beta_limit_inclusive: (h(5) * 100.0).fract() < 0.5,
                        start_angle: if (h(6) * 100.0).fract() < 0.5 {
                            CosineTurnStartAngle::Colinearity
                        } else {
                            CosineTurnStartAngle::OrbitAngle
                        },
                        start_angle_rad: deg(2.0 + 58.0 * h(6)),
                        period_s: probe_a(a) * (0.05 + 0.55 * h(7)),
                    },
                    a,
                )
            }
        };
        let law = GnssAttitudeLaw::CosineTurn(turn);
        let e = if i % 7 == 1 { 0.9 } else { 0.9 * h(0) };
        let omega = TAU * h(1);
        let sign = if (h(2) * 1000.0).fract() < 0.5 {
            1.0
        } else {
            -1.0
        };
        let beta_c = if i.is_multiple_of(5) {
            0.0
        } else {
            sign * 1.5 * turn.beta_limit_rad * h(2)
        };
        let probe = Kepler {
            a,
            e,
            beta: 0.0,
            beta_rate: 0.0,
            omega,
            m0: TAU * h(8),
        };
        let beta_rate = if i.is_multiple_of(4) || i.is_multiple_of(5) {
            0.0
        } else {
            SWEEP_BETA_CHANGE * (2.0 * h(3) - 1.0) * probe.n() / PI
        };
        let t0 = 3.0e4 * (2.0 * h(4) - 1.0);
        let orbit = Kepler {
            beta: beta_c - beta_rate * t0,
            beta_rate,
            ..probe
        };
        let tilt = beta_c.abs() > 1e-6 && i % 3 != 2;
        let plain = orbit.trajectory();
        let turned = tilted(orbit.trajectory());
        let trajectory: &dyn Fn(f64) -> Option<AttitudeSample> =
            if tilt { &turned } else { &plain };
        let period = orbit.period();
        let mut instants: Vec<f64> = (0..48)
            .map(|k| t0 + period * (f64::from(k) + h(8)) / 48.0)
            .collect();
        let reference = CosineReference::new(orbit, turn);
        let boundaries = reference.boundaries(t0, t0 + period);
        for &b in &boundaries {
            instants.extend([b - 1e-4, b + 1e-4]);
        }
        for t in instants {
            if boundaries.iter().any(|&b| (t - b).abs() < 1e-5) {
                continue;
            }
            let expected = reference.attitude(t);
            let result = gnss_attitude(&law, trajectory, t);
            let agrees = match (expected, &result) {
                (None, Ok(att)) => att.mode == AttitudeMode::Nominal,
                (Some((mode, yaw)), Ok(att)) => {
                    att.mode == mode && wrap_pi(att.yaw_rad - yaw).abs() < 1e-9
                }
                _ => false,
            };
            checked += 1;
            if !agrees {
                failures.push(format!(
                    "sample {i} {turn:?} a {a} e {e:.3} omega {:.1} beta_c {beta_c:e} rate {beta_rate:e} tilt {tilt} t - t0 {}: reference {expected:?} law {:?}",
                    omega.to_degrees(),
                    t - t0,
                    result.map(|a| (a.mode, a.yaw_rad))
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} instants disagree:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(checked > 10_000, "{checked}");
}

/// BeiDou-2 IGSO and MEO (Orekit `BeidouMeo`, with the instants solved on
/// the trajectory) over the validated domain: orbit normal or yaw steering
/// from beta at the instant, its change over the last orbit (one period back
/// on [`Kepler`]) and beta at the last quadrature (argument from noon
/// `-pi / 2`), both from the closed form. A quasi-random sweep of
/// eccentricity up to 0.9, the argument of perigee from noon, beta of either
/// sign around the switch limits (2.5 to 5.5 deg, and 0), beta's rate (1e-9
/// to 0.0249 rad per half period, rising and falling), the published limits
/// and custom ones, at 40 instants over an orbit, none within 1e-9 rad of a
/// limit.
#[test]
fn beidou2_switch_follows_the_reference_over_its_validated_domain() {
    let mut checked = 0;
    let mut failures = Vec::new();
    for i in 0..300_u32 {
        let h = |dim: usize| halton(i + 1, dim);
        let (ys_to_on, on_to_ys) = if i % 3 == 2 {
            (deg(1.0 + 9.0 * h(5)), deg(1.0 + 9.0 * h(6)))
        } else {
            (BEIDOU2_YS_TO_ON_BETA_RAD, BEIDOU2_ON_TO_YS_BETA_RAD)
        };
        let law = GnssAttitudeLaw::BeidouOrbitNormalSwitch {
            ys_to_on_beta_rad: ys_to_on,
            on_to_ys_beta_rad: on_to_ys,
        };
        let a = if i.is_multiple_of(2) {
            MEO_A
        } else {
            42_164_000.0
        };
        let e = if i % 7 == 1 { 0.9 } else { 0.9 * h(0) };
        let omega = TAU * h(1);
        let sign = if (h(2) * 1000.0).fract() < 0.5 {
            1.0
        } else {
            -1.0
        };
        let band = if i % 3 == 2 {
            ys_to_on.max(on_to_ys) + deg(1.0)
        } else {
            deg(5.5)
        };
        let beta_c = if i.is_multiple_of(11) {
            0.0
        } else {
            sign * band * (0.45 + 0.55 * h(2))
        };
        let probe = Kepler {
            a,
            e,
            beta: 0.0,
            beta_rate: 0.0,
            omega,
            m0: TAU * h(7),
        };
        // Beta always moves (with beta fixed the switch would follow the sign
        // of a rounded change), from 1e-9 to 0.0249 rad per half period.
        let change = SWEEP_BETA_CHANGE * libm::pow(10.0, -7.4 * h(3));
        let beta_rate = sign * if i.is_multiple_of(2) { change } else { -change } * probe.n() / PI;
        let t0 = 3.0e4 * (2.0 * h(4) - 1.0);
        let orbit = Kepler {
            beta: beta_c - beta_rate * t0,
            beta_rate,
            ..probe
        };
        let tilt = i % 3 != 1;
        let plain = orbit.trajectory();
        let turned = tilted(orbit.trajectory());
        let trajectory: &dyn Fn(f64) -> Option<AttitudeSample> =
            if tilt { &turned } else { &plain };
        let period = orbit.period();
        for k in 0..40 {
            let t = t0 + period * (f64::from(k) + h(8)) / 40.0;
            let beta = orbit.beta_at(t);
            let variation = beta - orbit.beta_at(t - period);
            let quadrature = {
                let q = orbit.time_at(-FRAC_PI_2, t - period);
                if q > t {
                    q - period
                } else {
                    q
                }
            };
            let beta_q = orbit.beta_at(quadrature);
            let lower = ys_to_on.min(on_to_ys);
            let upper = ys_to_on.max(on_to_ys);
            let decreasing = beta * variation <= 0.0;
            let edges = [
                lower - variation.abs(),
                upper + variation.abs(),
                ys_to_on,
                on_to_ys,
            ];
            if edges
                .iter()
                .any(|&x| (beta.abs() - x).abs() < 1e-9 || (beta_q.abs() - x).abs() < 1e-9)
            {
                continue;
            }
            let orbit_normal = if beta.abs() <= lower - variation.abs() {
                true
            } else if beta.abs() > upper + variation.abs() || (decreasing && beta.abs() >= ys_to_on)
            {
                false
            } else if !decreasing && beta.abs() <= on_to_ys {
                true
            } else if decreasing {
                beta_q.abs() <= ys_to_on
            } else {
                beta_q.abs() <= on_to_ys
            };
            let result = gnss_attitude(&law, trajectory, t);
            let agrees = match &result {
                Ok(att) if orbit_normal => att.mode == AttitudeMode::OrbitNormal,
                Ok(att) => att.mode == AttitudeMode::Nominal,
                Err(_) => false,
            };
            checked += 1;
            if !agrees {
                failures.push(format!(
                    "sample {i} limits {:.3} {:.3} a {a} e {e:.3} omega {:.1} beta {:.4} variation {variation:e} beta_q {:.4} tilt {tilt}: reference orbit normal {orbit_normal} law {:?}",
                    ys_to_on.to_degrees(),
                    on_to_ys.to_degrees(),
                    omega.to_degrees(),
                    beta.to_degrees(),
                    beta_q.to_degrees(),
                    result.map(|a| a.mode)
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} instants disagree:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(checked > 10_000, "{checked}");
}
