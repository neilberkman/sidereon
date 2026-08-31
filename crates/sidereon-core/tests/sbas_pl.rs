//! SBAS protection-level validation provenance.
//!
//! Tier 1 uses RTCA DO-229D Appendix A UDRE/GIVE tables, Appendix A tropo and
//! obliquity forms, Appendix J airborne multipath, and DO-229 protection-level
//! K constants. Tier 2 uses Walter and Enge, "Weighted RAIM for Precision
//! Approach", ION GPS-95, as the public weighted-projection formulation, with
//! fixed hand-worked normal-matrix outputs for the reference geometry below.

use sidereon_core::astro::time::model::{GnssWeekTow, TimeScale};
use sidereon_core::dop::line_of_sight_from_az_el_deg;
use sidereon_core::integrity::normal_q_inv;
use sidereon_core::sbas::{
    give_variance_m2_for_givei, sbas_prn_to_sat, udre_variance_m2_for_udrei, SbasCorrectionStore,
    SbasIgp, SbasIgpDelay, SbasIgpMask, SbasIonoDelays, SbasIonoGrid, SbasMessage, SpareBits,
};
use sidereon_core::sbas_pl::{
    sbas_obliquity_factor, sbas_protection_levels, sigma_air_multipath_m, sigma_flt_m_for_udrei,
    sigma_tropo_m, AirborneModel, DegradationParams, ProtectionGeometry, ProtectionRow,
    SbasErrorModel, SbasKMultipliers, SbasSisError,
};
use sidereon_core::{GnssSatelliteId, GnssSystem, Wgs84Geodetic};

const UDRE_BITS: [u64; 14] = [
    0x3faa_9fbe_76c8_b439,
    0x3fb7_a786_c226_809d,
    0x3fc2_7bb2_fec5_6d5d,
    0x3fd2_1cac_0831_26e9,
    0x3fdd_f06f_6944_6738,
    0x3fea_9ba5_e353_f7cf,
    0x3ff4_c985_f06f_6944,
    0x3ffd_ef34_d6a1_61e5,
    0x4004_5f3b_645a_1cac,
    0x400a_9ba5_e353_f7cf,
    0x4014_c985_f06f_6944,
    0x4034_c978_d4fd_f3b6,
    0x406c_deea_4a8c_154d,
    0x40a0_3d63_d70a_3d71,
];

const GIVE_BITS: [u64; 15] = [
    0x3f81_3404_ea4a_8c15,
    0x3fa1_0cb2_95e9_e1b1,
    0x3fb3_2ca5_7a78_6c22,
    0x3fc1_096b_b98c_7e28,
    0x3fca_9c77_9a6b_50b1,
    0x3fd3_295e_9e1b_089a,
    0x3fda_147a_e147_ae14,
    0x3fe1_07c8_4b5d_cc64,
    0x3fe5_8d4f_df3b_645a,
    0x3fea_9ba5_e353_f7cf,
    0x3ff3_288c_e703_afb8,
    0x3ffd_ef34_d6a1_61e5,
    0x400a_9ba5_e353_f7cf,
    0x4034_c978_d4fd_f3b6,
    0x4067_62a4_a8c1_54ca,
];

#[test]
fn udre_and_give_tables_are_frozen_bits() {
    for (udrei, &bits) in UDRE_BITS.iter().enumerate() {
        assert_eq!(
            udre_variance_m2_for_udrei(udrei as u8)
                .expect("valid UDREI")
                .to_bits(),
            bits
        );
    }
    assert!(udre_variance_m2_for_udrei(14).is_none());
    assert!(udre_variance_m2_for_udrei(15).is_none());

    for (givei, &bits) in GIVE_BITS.iter().enumerate() {
        assert_eq!(
            give_variance_m2_for_givei(givei as u8)
                .expect("valid GIVEI")
                .to_bits(),
            bits
        );
    }
    assert!(give_variance_m2_for_givei(15).is_none());
}

#[test]
fn closed_form_error_terms_match_do229_oracles() {
    let airborne = AirborneModel::aad_a();
    let cases = [
        (
            5.0_f64,
            1.226_153_329_897_585_7,
            3.039_178_452_443_691,
            0.451_461_249_647_695_74,
            0.577_422_947_182_963_2,
        ),
        (
            10.0_f64,
            0.669_874_063_202_278_6,
            2.789_270_370_828_692,
            0.324_976_103_820_864_45,
            0.484_983_987_420_810_9,
        ),
        (
            90.0_f64,
            0.12,
            1.0,
            0.130_065_407_196_165_94,
            0.382_775_404_315_774_3,
        ),
    ];
    for (el_deg, tropo_m, fpp, multipath_m, air_m) in cases {
        let el = el_deg.to_radians();
        assert_abs_diff(sigma_tropo_m(el).expect("tropo"), tropo_m, 2.0e-15);
        assert_abs_diff(sbas_obliquity_factor(el).expect("mapping"), fpp, 2.0e-15);
        assert_abs_diff(
            sigma_air_multipath_m(el).expect("multipath"),
            multipath_m,
            2.0e-15,
        );
        assert_abs_diff(airborne.sigma_air_m(el).expect("air"), air_m, 2.0e-15);
    }

    let degradation = DegradationParams::none();
    assert_eq!(
        sigma_flt_m_for_udrei(0, &degradation)
            .expect("sigma flt")
            .to_bits(),
        udre_variance_m2_for_udrei(0).unwrap().sqrt().to_bits()
    );
}

#[test]
fn store_populates_give_variance_from_givei() {
    let geo = sbas_prn_to_sat(120).expect("valid SBAS source");
    let epoch = GnssWeekTow::new(TimeScale::Gpst, 2400, 0.0).expect("valid epoch");
    let mut store = SbasCorrectionStore::new();
    let mut mask = [false; 201];
    mask[0] = true;
    store
        .ingest(
            &SbasMessage::IgpMask(SbasIgpMask {
                preamble: 0x53,
                band_number: 0,
                iodi: 0,
                mask,
                reserved: SpareBits::new(),
            }),
            geo,
            epoch,
        )
        .expect("ingest IGP mask");
    let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
    entries[0] = SbasIgpDelay {
        vertical_delay: 16,
        givei: 3,
    };
    store
        .ingest(
            &SbasMessage::IonoDelays(SbasIonoDelays {
                preamble: 0x53,
                band_number: 0,
                block_id: 0,
                iodi: 0,
                entries,
                reserved: SpareBits::new(),
            }),
            geo,
            epoch,
        )
        .expect("ingest iono");

    let grid = store.iono_grid(geo).expect("grid");
    assert_eq!(grid.igps().len(), 1);
    assert_eq!(
        grid.igps()[0].give_variance_m2.map(f64::to_bits),
        Some(give_variance_m2_for_givei(3).unwrap().to_bits())
    );
}

#[test]
fn variance_at_ipp_uses_bilinear_weights() {
    let grid = SbasIonoGrid::new(
        vec![
            igp(-5.0, -5.0, 1.0),
            igp(-5.0, 5.0, 2.0),
            igp(5.0, -5.0, 3.0),
            igp(5.0, 5.0, 4.0),
        ],
        0,
    );
    assert_abs_diff(grid.variance_at_ipp(0.0, 0.0).expect("variance"), 2.5, 0.0);
    assert_abs_diff(
        grid.variance_at_ipp(-2.5, 2.5).expect("variance"),
        2.25,
        0.0,
    );
}

#[test]
fn k_multipliers_are_pinned_to_mops_constants() {
    assert_eq!(
        SbasKMultipliers::PRECISION_APPROACH.k_h.to_bits(),
        6.0_f64.to_bits()
    );
    assert_eq!(
        SbasKMultipliers::PRECISION_APPROACH.k_v.to_bits(),
        5.33_f64.to_bits()
    );
    assert_eq!(
        SbasKMultipliers::EN_ROUTE_NPA.k_h.to_bits(),
        6.18_f64.to_bits()
    );
    assert_eq!(
        SbasKMultipliers::EN_ROUTE_NPA.k_v.to_bits(),
        5.33_f64.to_bits()
    );
    assert_eq!(
        normal_q_inv(5.0e-8).unwrap().to_bits(),
        0x4015_4e90_b4db_5fad
    );
    assert_abs_diff(normal_q_inv(5.0e-8).unwrap(), 5.33, 0.003_28);
    let rayleigh = libm::sqrt(-2.0 * libm::log(5.0e-9_f64));
    assert_abs_diff(rayleigh, 6.182_851_756_998_919, 1.0e-12);
    assert_abs_diff(rayleigh, SbasKMultipliers::EN_ROUTE_NPA.k_h, 0.003);
}

#[test]
fn d_major_reduces_to_larger_axis_when_cross_term_is_zero() {
    let geometry = geometry_from_az_el(&[(90.0, 60.0), (270.0, 60.0), (0.0, 30.0), (180.0, 30.0)]);
    let model = model_from_sigmas(&[1.0, 1.0, 2.0, 2.0]);
    let pl = sbas_protection_levels(&geometry, &model, SbasKMultipliers::PRECISION_APPROACH)
        .expect("protection");
    assert_abs_diff(pl.d_en_m2, 0.0, 1.0e-12);
    assert!(pl.d_north_m > pl.d_east_m);
    assert_abs_diff(pl.d_major_m, pl.d_north_m, 1.0e-12);
}

#[test]
fn protection_levels_match_fixed_weighted_geometry() {
    let geometry = geometry_from_az_el(&[
        (15.0, 15.0),
        (80.0, 70.0),
        (155.0, 25.0),
        (230.0, 55.0),
        (310.0, 35.0),
    ]);
    let model = model_from_sigmas(&[2.0, 1.0, 1.5, 1.2, 1.8]);
    let pl = sbas_protection_levels(&geometry, &model, SbasKMultipliers::PRECISION_APPROACH)
        .expect("protection");

    assert_abs_diff(pl.d_east_m, 1.500_300_322_250_245, 2.0e-12);
    assert_abs_diff(pl.d_north_m, 1.214_969_186_420_138, 2.0e-12);
    assert_abs_diff(pl.sigma_u_m, 2.563_615_538_395_546, 2.0e-12);
    assert_abs_diff(pl.d_en_m2, 0.159_258_839_573_006_66, 2.0e-12);
    assert_abs_diff(pl.d_major_m, 1.510_748_501_734_169, 2.0e-12);
    assert_abs_diff(pl.hpl_m, 9.064_491_010_405_014, 2.0e-12);
    assert_abs_diff(pl.vpl_m, 13.664_070_819_648_263, 2.0e-12);
}

fn geometry_from_az_el(az_el_deg: &[(f64, f64)]) -> ProtectionGeometry {
    let receiver = receiver();
    let rows = az_el_deg
        .iter()
        .enumerate()
        .map(|(idx, &(az_deg, el_deg))| ProtectionRow {
            id: gps((idx + 1) as u8),
            line_of_sight: line_of_sight_from_az_el_deg(az_deg, el_deg, receiver)
                .expect("valid az/el"),
            system: GnssSystem::Gps,
            elevation_rad: el_deg.to_radians(),
        })
        .collect();
    ProtectionGeometry {
        rows,
        receiver,
        clock_systems: vec![GnssSystem::Gps],
    }
}

fn model_from_sigmas(sigmas_m: &[f64]) -> SbasErrorModel {
    let rows = sigmas_m
        .iter()
        .enumerate()
        .map(|(idx, &sigma_flt_m)| SbasSisError {
            id: gps((idx + 1) as u8),
            sigma_flt_m,
            sigma_uire_m: 0.0,
            sigma_air_m: 0.0,
            sigma_tropo_m: 0.0,
        })
        .collect();
    SbasErrorModel::new(rows)
}

fn igp(lat_deg: f64, lon_deg: f64, give_variance_m2: f64) -> SbasIgp {
    SbasIgp {
        lat_deg,
        lon_deg,
        vertical_delay_m: 0.0,
        give_variance_m2: Some(give_variance_m2),
    }
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn receiver() -> Wgs84Geodetic {
    Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid receiver")
}

fn assert_abs_diff(left: f64, right: f64, tol: f64) {
    assert!(
        (left - right).abs() <= tol,
        "left={left:.17e} right={right:.17e} diff={:.17e} tol={tol:.17e}",
        (left - right).abs()
    );
}
