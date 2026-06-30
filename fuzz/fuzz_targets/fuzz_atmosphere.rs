#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::time::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::{
    atmosphere::{
        self, IonoModel, KlobucharParams, MappingModel, Met, TecGrid, TecGridEpoch,
        TecGridEvalOptions, TropoModel, ZwdEpoch, ZwdProfile, ZwdSlantOptions,
    },
    ephemeris::{
        satellite_clock_offset_s, satellite_position_ecef, satellite_state, ClockPolynomial,
        ConstellationConstants, KeplerianElements,
    },
    precise_positioning::{self, DualFrequencyObservation, TecConfig, TecEpoch, TecObservation},
    Wgs84Geodetic,
};

#[derive(Debug, Arbitrary)]
struct Input {
    receiver: [f64; 3],
    sat: [f64; 3],
    params: [f64; 16],
    grid_values: [f64; 8],
    nanos: i64,
    doy: u16,
    geo: bool,
}

fn geodetic(input: &Input) -> Result<Wgs84Geodetic, sidereon_core::FrameValueError> {
    Wgs84Geodetic::new(input.receiver[0], input.receiver[1], input.receiver[2])
}

fn instant(input: &Input) -> Option<Instant> {
    JulianDateSplit::new(input.params[0], input.params[1])
        .ok()
        .map(|jd| Instant::from_julian_date(TimeScale::Gpst, jd))
}

fn elements(input: &Input) -> KeplerianElements {
    KeplerianElements {
        sqrt_a: input.params[0],
        e: input.params[1],
        m0: input.params[2],
        delta_n: input.params[3],
        omega0: input.params[4],
        i0: input.params[5],
        omega: input.params[6],
        omega_dot: input.params[7],
        idot: input.params[8],
        cuc: input.params[9],
        cus: input.params[10],
        crc: input.params[11],
        crs: input.params[12],
        cic: input.params[13],
        cis: input.params[14],
        toe_sow: input.params[15],
    }
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let receiver = geodetic(&input);
    assert_ok_finite_or_err("Wgs84Geodetic::new", receiver);
    assert_ok_finite_or_err(
        "JulianDateSplit::new",
        JulianDateSplit::new(input.params[0], input.params[1]),
    );
    let epoch = instant(&input);
    let klob = KlobucharParams {
        alpha: [
            input.params[0],
            input.params[1],
            input.params[2],
            input.params[3],
        ],
        beta: [
            input.params[4],
            input.params[5],
            input.params[6],
            input.params[7],
        ],
    };
    let model = IonoModel::Klobuchar(klob);

    if let (Ok(receiver), Some(epoch)) = (receiver, epoch) {
        assert_ok_finite_or_err(
            "atmosphere::ionosphere_delay",
            atmosphere::ionosphere_delay(
                receiver,
                input.params[8],
                input.params[9],
                epoch,
                input.params[10],
                &model,
            ),
        );
        assert_ok_finite_or_err(
            "atmosphere::klobuchar",
            atmosphere::klobuchar(
                &klob,
                receiver,
                input.params[8],
                input.params[9],
                epoch,
                input.params[10],
            ),
        );
    }
    assert_ok_finite_or_err(
        "atmosphere::klobuchar_native",
        atmosphere::klobuchar_native(
            &klob,
            input.receiver[0],
            input.receiver[1],
            input.params[8],
            input.params[9],
            input.params[11],
            input.params[10],
        ),
    );

    let grid = TecGrid::new(
        vec![0.0, 1.0],
        vec![-10.0, 10.0],
        vec![0.0, 20.0],
        input.grid_values.to_vec(),
    );
    if let Ok(grid) = grid {
        let opts = TecGridEvalOptions {
            epoch: TecGridEpoch::new(input.nanos, input.doy),
            min_elevation_rad: input.params[12],
            nan_pierce_point_height_m: input.params[13],
            frequency_hz: input.params[10],
            ..TecGridEvalOptions::l1(TecGridEpoch::new(input.nanos, input.doy))
        };
        assert_ok_finite_or_err(
            "TecGrid::vtec_at_pierce_point",
            grid.vtec_at_pierce_point(opts.epoch, input.params[0], input.params[1]),
        );
        assert_ok_finite_or_err(
            "atmosphere::regular_tec_grid_delay_xyz",
            atmosphere::regular_tec_grid_delay_xyz(
                &grid,
                opts,
                &input.sat,
                &input.receiver,
                |xyz| [xyz[0], xyz[1], xyz[2]],
            ),
        );
        assert_ok_finite_or_err(
            "atmosphere::regular_tec_xyz",
            atmosphere::regular_tec_xyz(&grid, opts, &input.sat, &input.receiver, |xyz| {
                [xyz[0], xyz[1], xyz[2]]
            }),
        );
    }

    let met = Met::new(input.params[2], input.params[3], input.params[4]);
    assert_ok_or_err("Met::new", met.as_ref());
    if let Ok(standard) = Met::standard(input.receiver[2], input.params[4]) {
        assert_success("Met::standard", standard.pressure_hpa);
    }
    if let (Ok(receiver), Ok(met), Some(epoch)) = (geodetic(&input), met, epoch) {
        assert_ok_finite_or_err(
            "atmosphere::tropo_zenith",
            atmosphere::tropo_zenith(TropoModel::Saastamoinen, receiver, met),
        );
        assert_ok_finite_or_err(
            "atmosphere::tropo_zenith_zwd",
            atmosphere::tropo_zenith(
                TropoModel::ZwdAltitudeScaled(ZwdProfile::default()),
                receiver,
                met,
            ),
        );
        assert_ok_finite_or_err(
            "atmosphere::tropo_mapping",
            atmosphere::tropo_mapping(MappingModel::Niell, input.params[5], receiver, epoch),
        );
        assert_ok_finite_or_err(
            "atmosphere::tropo_slant",
            atmosphere::tropo_slant(input.params[5], receiver, met, epoch),
        );
    }
    assert_ok_finite_or_err(
        "atmosphere::zwd_zenith_wet_delay",
        atmosphere::zwd_zenith_wet_delay(ZwdProfile::default(), input.receiver[2]),
    );
    let zwd_epoch = ZwdEpoch::new(input.nanos, input.doy);
    assert_ok_or_err("ZwdEpoch::new", zwd_epoch.as_ref());
    if let Ok(zwd_epoch) = zwd_epoch {
        let zwd_opts = ZwdSlantOptions::new(zwd_epoch, ZwdProfile::default());
        assert_ok_or_err("ZwdSlantOptions::new", zwd_opts.as_ref());
        if let Ok(zwd_opts) = zwd_opts {
            assert_ok_finite_or_err(
                "atmosphere::tropo_zwd_delay_xyz",
                atmosphere::tropo_zwd_delay_xyz(zwd_opts, &input.sat, &input.receiver, |xyz| {
                    [xyz[0], xyz[1], xyz[2]]
                }),
            );
        }
    }

    let dual = DualFrequencyObservation {
        satellite_id: "G01".to_string(),
        ambiguity_id: "G01#0".to_string(),
        p1_m: input.params[0],
        p2_m: input.params[1],
        phi1_cyc: input.params[2],
        phi2_cyc: input.params[3],
        f1_hz: input.params[4],
        f2_hz: input.params[5],
        lli1: None,
        lli2: None,
    };
    assert_ok_finite_or_err(
        "precise_positioning::code_geometry_free_m",
        precise_positioning::code_geometry_free_m(&dual),
    );
    assert_ok_finite_or_err(
        "precise_positioning::phase_geometry_free_m",
        precise_positioning::phase_geometry_free_m(&dual),
    );
    assert_ok_finite_or_err(
        "precise_positioning::slant_tec_from_code_geometry_free_m",
        precise_positioning::slant_tec_from_code_geometry_free_m(
            input.params[0],
            input.params[4],
            input.params[5],
        ),
    );
    assert_ok_finite_or_err(
        "precise_positioning::slant_tec_from_phase_geometry_free_m",
        precise_positioning::slant_tec_from_phase_geometry_free_m(
            input.params[1],
            input.params[4],
            input.params[5],
        ),
    );
    if let Ok(code) = precise_positioning::estimate_code_slant_tec(&dual) {
        assert_all_finite(
            "precise_positioning::estimate_code_slant_tec",
            [code.code_geometry_free_m, code.slant_tec_tecu],
        );
    }
    if let Ok(phase) = precise_positioning::estimate_phase_slant_tec(&dual) {
        assert_all_finite(
            "precise_positioning::estimate_phase_slant_tec",
            [phase.phase_geometry_free_m, phase.slant_tec_tecu],
        );
    }
    let tec_config = TecConfig {
        shell_height_m: input.params[6],
        earth_radius_m: input.params[7],
    };
    assert_ok_finite_or_err(
        "precise_positioning::thin_shell_mapping_function",
        precise_positioning::thin_shell_mapping_function(input.params[8], tec_config),
    );
    assert_ok_finite_or_err(
        "precise_positioning::vertical_tec_from_slant_tec",
        precise_positioning::vertical_tec_from_slant_tec(
            input.params[9],
            input.params[8],
            tec_config,
        ),
    );
    assert_ok_finite_or_err(
        "precise_positioning::ionospheric_pierce_point",
        precise_positioning::ionospheric_pierce_point(
            input.receiver[0],
            input.receiver[1],
            input.params[8],
            input.params[9],
            tec_config,
        )
        .map(|pp| [pp.latitude_rad, pp.longitude_rad]),
    );
    assert_ok_or_err(
        "precise_positioning::estimate_tec",
        precise_positioning::estimate_tec(
            &[TecEpoch {
                time_s: input.params[10],
                receiver_latitude_rad: input.receiver[0],
                receiver_longitude_rad: input.receiver[1],
                observations: vec![TecObservation {
                    observation: dual,
                    elevation_rad: input.params[8],
                    azimuth_rad: input.params[9],
                }],
            }],
            tec_config,
        ),
    );

    let elements = elements(&input);
    let clock = ClockPolynomial {
        af0: input.params[0],
        af1: input.params[1],
        af2: input.params[2],
        toc_sow: input.params[3],
    };
    let consts = ConstellationConstants::GPS;
    assert_ok_finite_or_err(
        "ephemeris::eccentric_anomaly",
        sidereon_core::ephemeris::eccentric_anomaly(input.params[13], input.params[1]),
    );
    let orbit = satellite_position_ecef(&elements, &consts, input.params[11], input.geo);
    assert_ok_finite_or_err("ephemeris::satellite_position_ecef", orbit.as_ref());
    if let Ok(orbit) = orbit {
        assert_ok_finite_or_err(
            "ephemeris::satellite_clock_offset_s",
            satellite_clock_offset_s(
                &clock,
                &consts,
                &elements,
                orbit.sin_e,
                input.params[11],
                input.params[14],
            ),
        );
    }
    assert_ok_finite_or_err(
        "ephemeris::satellite_state",
        satellite_state(
            &elements,
            &clock,
            &consts,
            input.params[11],
            input.params[14],
            input.geo,
        ),
    );
});
