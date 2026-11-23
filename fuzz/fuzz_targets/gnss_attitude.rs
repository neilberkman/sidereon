#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::attitude::gnss::{
    dyb_axes, gnss_attitude, nominal_yaw, AttitudeSample, BeidouOrbitClass, CosineTurnParameters,
    CosineTurnStartAngle, GnssAttitudeLaw, GnssBlock,
};
use sidereon_core::GnssSystem;

#[derive(Debug, Arbitrary)]
struct Input {
    block_name: String,
    block_type: String,
    system: u8,
    orbit: u8,
    prn: Option<u8>,
    law: u8,
    parameters: [f64; 4],
    position: [f64; 3],
    velocity: [f64; 3],
    sun: [f64; 3],
    sun_rate: [f64; 3],
    t_s: f64,
}

// Block parsing, nominal yaw, DYB axes and every attitude law over an arbitrary
// straight-line trajectory: finite results or typed errors, never a panic or a
// hang.
fuzz_target!(|input: Input| {
    let _ = input.block_name.parse::<GnssBlock>();
    let system = match input.system % 4 {
        0 => GnssSystem::Gps,
        1 => GnssSystem::Glonass,
        2 => GnssSystem::Galileo,
        _ => GnssSystem::BeiDou,
    };
    let orbit = match input.orbit % 4 {
        0 => None,
        1 => Some(BeidouOrbitClass::Geo),
        2 => Some(BeidouOrbitClass::Igso),
        _ => Some(BeidouOrbitClass::Meo),
    };
    let parsed = GnssBlock::from_constellation_block_type(system, &input.block_type, orbit);

    if let Ok(nominal) = nominal_yaw(input.position, input.velocity, input.sun) {
        assert!(nominal.yaw_rad.is_finite() && nominal.beta_rad.is_finite());
        for axis in [nominal.body_x, nominal.body_y, nominal.body_z] {
            assert!(axis.iter().all(|x| x.is_finite()), "{nominal:?}");
        }
    }
    if let Ok(dyb) = dyb_axes(input.position, input.sun) {
        for axis in [dyb.d, dyb.y, dyb.b] {
            assert!(axis.iter().all(|x| x.is_finite()), "{dyb:?}");
        }
    }

    let [a, b, c, d] = input.parameters;
    let law = match input.law % 10 {
        0 => parsed
            .ok()
            .and_then(|block| GnssAttitudeLaw::for_block(block, input.prn).ok()),
        1 => Some(GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s: a,
            yaw_bias_rad: b,
            shadow_limit_rad: c,
        }),
        2 => Some(GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s: a }),
        3 => Some(GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s: a,
            yaw_bias_rad: b,
            shadow_limit_rad: c,
        }),
        4 => Some(GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s: a,
            shadow_limit_rad: b,
        }),
        5 => Some(GnssAttitudeLaw::GalileoIov {
            beta_x_rad: a,
            beta_y_rad: b,
        }),
        6 => Some(GnssAttitudeLaw::CosineTurn(CosineTurnParameters {
            beta_limit_rad: a,
            beta_limit_inclusive: d >= 0.0,
            start_angle: if d.is_sign_negative() {
                CosineTurnStartAngle::OrbitAngle
            } else {
                CosineTurnStartAngle::Colinearity
            },
            start_angle_rad: b,
            period_s: c,
        })),
        7 => Some(GnssAttitudeLaw::BeidouOrbitNormalSwitch {
            ys_to_on_beta_rad: a,
            on_to_ys_beta_rad: b,
        }),
        8 => Some(GnssAttitudeLaw::FixedBetaYawSteering { beta0_rad: a }),
        _ => Some(GnssAttitudeLaw::OrbitNormal),
    };
    let Some(law) = law else { return };

    let trajectory = |t: f64| {
        Some(AttitudeSample {
            position_m: std::array::from_fn(|k| input.position[k] + input.velocity[k] * t),
            velocity_m_s: input.velocity,
            sun_position_m: std::array::from_fn(|k| input.sun[k] + input.sun_rate[k] * t),
        })
    };
    if let Ok(attitude) = gnss_attitude(&law, &trajectory, input.t_s) {
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
});
