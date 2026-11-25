use std::sync::Arc;

use nalgebra::Vector3;
use serde_json::Value;
use sidereon_core::astro::de::{DeBody, DeEphemeris, DeError};
use sidereon_core::astro::error::PropagationError;
use sidereon_core::astro::forces::{
    EarthRadiationPressure, Ecom1Coefficients, Ecom1SolarRadiationPressure,
    Ecom2SolarRadiationPressure, ForceModel, SolarRadiationPressure, SolidEarthTideGravity,
    ThirdBodyGravity,
};
use sidereon_core::astro::frames::{EarthOrientationProvider, TdbEarthOrientationProvider};
use sidereon_core::astro::propagator::PropagationContext;
use sidereon_core::astro::state::CartesianState;

const RECORDS: &[u8] = include_bytes!("fixtures/bodies/de440/lnxp1990.440");
const ORACLE: &str = include_str!("fixtures/bodies/de440/de440_orekit.json");

fn bits(hex: &Value) -> u64 {
    u64::from_str_radix(hex.as_str().expect("binary64 hex"), 16).expect("hex")
}

fn fp(hex: &Value) -> f64 {
    f64::from_bits(bits(hex))
}

#[test]
fn de440_constants_and_sun_moon_states_match_orekit() {
    let oracle: Value = serde_json::from_str(ORACLE).expect("oracle");
    let de = DeEphemeris::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bodies/de440/lnxp1990.440"),
    )
    .expect("DE440 ephemeris");
    assert_eq!(oracle["orekit"], "13.1.10");
    assert_eq!(oracle["hipparchus"], "4.0.3");
    assert_eq!(de.version(), oracle["version"].as_u64().unwrap() as u32);
    assert_eq!(
        de.record_size(),
        oracle["record_size"].as_u64().unwrap() as usize
    );
    assert_eq!(
        RECORDS.len() / de.record_size() - 2,
        oracle["data_records"].as_u64().unwrap() as usize
    );
    let (first, last) = de.coverage_tdb_seconds();
    let coverage = &oracle["coverage_tdb_s"];
    assert_eq!(
        [first.to_bits(), last.to_bits()],
        [bits(&coverage[0]), bits(&coverage[1])]
    );
    assert_eq!(de.astronomical_unit_m().to_bits(), bits(&oracle["au_m"]));
    assert_eq!(de.earth_moon_mass_ratio().to_bits(), bits(&oracle["emrat"]));
    let constants = oracle["constants"].as_object().expect("constants");
    assert_eq!(de.constants().len(), constants.len());
    for (name, expected) in constants {
        assert_eq!(de.constants()[name].to_bits(), bits(expected), "{name}");
    }
    let gm_bodies = [
        ("SUN", DeBody::Sun),
        ("MERCURY", DeBody::Mercury),
        ("VENUS", DeBody::Venus),
        ("EARTH_MOON", DeBody::EarthMoon),
        ("EARTH", DeBody::Earth),
        ("MOON", DeBody::Moon),
        ("MARS", DeBody::Mars),
        ("JUPITER", DeBody::Jupiter),
        ("SATURN", DeBody::Saturn),
        ("URANUS", DeBody::Uranus),
        ("NEPTUNE", DeBody::Neptune),
        ("PLUTO", DeBody::Pluto),
        ("SOLAR_SYSTEM_BARYCENTER", DeBody::SolarSystem),
    ];
    for (name, body) in gm_bodies {
        assert_eq!(
            de.gm_m3_s2(body).expect(name).to_bits(),
            bits(&oracle["gm_m3_s2"][name]),
            "{name}"
        );
    }
    let rows = oracle["states"].as_array().expect("states");
    assert_eq!(fp(&rows[0][0]), first);
    assert_eq!(fp(&rows.last().unwrap()[0]), last);
    for row in rows {
        let epoch = fp(&row[0]);
        let state = de
            .sun_moon_at_tdb_seconds(epoch)
            .unwrap_or_else(|e| panic!("{epoch}: {e}"));
        let position = de.sun_moon_positions_at_tdb_seconds(epoch).unwrap();
        let vectors = [
            position.sun,
            state.sun.position_m,
            state.sun.velocity_m_s,
            position.moon,
            state.moon.position_m,
            state.moon.velocity_m_s,
        ];
        for (vector, actual) in vectors.iter().enumerate() {
            for (axis, value) in actual.iter().enumerate() {
                let reference = bits(&row[vector + 1][axis]);
                assert_eq!(
                    value.to_bits(),
                    reference,
                    "epoch {epoch}, vector {vector}, axis {axis}"
                );
            }
        }
    }
    eprintln!(
        "{} constants, {} GMs, {} epochs and {} state components at 0 ULP",
        constants.len(),
        gm_bodies.len(),
        rows.len(),
        rows.len() * 18
    );
    for epoch in [
        first - 0.000_125,
        last + 0.000_125,
        0.0,
        f64::MAX,
        -f64::MAX,
    ] {
        assert!(matches!(
            de.sun_moon_at_tdb_seconds(epoch),
            Err(DeError::OutsideCoverage { epoch_tdb_seconds }) if epoch_tdb_seconds == epoch
        ));
    }
    assert!(matches!(
        de.sun_moon_at_tdb_seconds(f64::NAN),
        Err(DeError::NonFiniteEpoch)
    ));
    assert!(matches!(
        DeEphemeris::from_bytes(&RECORDS[..RECORDS.len() - 1]),
        Err(DeError::InvalidFile(_))
    ));
}

#[test]
fn force_context_uses_de440_for_radiation_and_gravity() {
    let de = DeEphemeris::from_bytes(RECORDS).expect("DE440");
    let epoch = de.coverage_tdb_seconds().0 + 10_000.25;
    let bodies = de.sun_moon_at_tdb_seconds(epoch).expect("covered");
    let sun = Vector3::from(bodies.sun.position_m.map(|m| m / 1000.0));
    let moon = Vector3::from(bodies.moon.position_m.map(|m| m / 1000.0));
    let orientation_provider = Arc::new(TdbEarthOrientationProvider::default());
    let ctx = PropagationContext::new()
        .with_body_fixed_frame_provider(orientation_provider.clone())
        .with_de_ephemeris(Arc::new(de));
    let state = CartesianState::new(epoch, [15_000.0, 17_000.0, -8_000.0], [-3.0, 2.0, 1.0]);

    let srp = SolarRadiationPressure::new(1.2, 0.02)
        .unwrap()
        .with_moon_eclipse(true);
    let albedo = EarthRadiationPressure::new(1.2, 0.02)
        .unwrap()
        .with_moon_eclipse(true);
    let ecom1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::from_array([1.0e-9; 9]))
        .unwrap()
        .with_moon_eclipse(true);
    let ecom2 = Ecom2SolarRadiationPressure::new(0, 0, vec![1.0e-9; 3])
        .unwrap()
        .with_moon_eclipse(true);
    assert_eq!(
        srp.acceleration(&state, &ctx).unwrap(),
        srp.acceleration_with_sun_moon(state.position_km, sun, moon)
            .unwrap()
    );
    let de_only = PropagationContext::new()
        .with_de_ephemeris(Arc::new(DeEphemeris::from_bytes(RECORDS).unwrap()));
    assert_eq!(
        albedo.acceleration(&state, &de_only).unwrap(),
        albedo
            .acceleration_with_sun_moon(&state, sun, moon)
            .unwrap()
    );
    assert_eq!(
        ecom1.acceleration(&state, &ctx).unwrap(),
        ecom1
            .acceleration_with_sun_moon(state.position_km, state.velocity_km_s, sun, moon)
            .unwrap()
    );
    assert_eq!(
        ecom2.acceleration(&state, &ctx).unwrap(),
        ecom2
            .acceleration_with_sun_moon(state.position_km, state.velocity_km_s, sun, moon)
            .unwrap()
    );

    let mut tide = SolidEarthTideGravity::new(398_600.441_5, 6378.137, 1.327_124_4e11, 4902.8);
    tide.frequency_dependent = false;
    let orientation = orientation_provider
        .orientation_at_tdb_seconds(epoch)
        .unwrap();
    let rotate = |position: Vector3<f64>| {
        orientation
            .gcrf_to_itrf_position_km(position.into())
            .unwrap()
    };
    assert_eq!(
        tide.coefficient_corrections_at_epoch(epoch, &ctx).unwrap(),
        tide.coefficient_corrections_for_body_fixed_bodies(rotate(sun), rotate(moon))
            .unwrap(),
    );

    let third_body = ThirdBodyGravity::default();
    assert_ne!(
        third_body.acceleration(&state, &ctx).unwrap(),
        third_body
            .acceleration(&state, &PropagationContext::default())
            .unwrap()
    );
    let gap = CartesianState {
        epoch_tdb_seconds: 0.0,
        ..state
    };
    for force in [
        &srp as &dyn ForceModel,
        &albedo,
        &ecom1,
        &ecom2,
        &third_body,
    ] {
        assert!(
            matches!(force.acceleration(&gap, &ctx), Err(PropagationError::ForceModelFailure(message)) if message.contains("does not cover"))
        );
        assert!(force
            .acceleration(&gap, &PropagationContext::default())
            .is_ok());
    }
}

fn set_first_granule_x(data: &mut [u8], record_size: usize, slot: usize, coefficients: &[f64]) {
    let layout = 2696 + 12 * slot;
    let offset = u32::from_le_bytes(data[layout..layout + 4].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(data[layout + 4..layout + 8].try_into().unwrap()) as usize;
    assert!(coefficients.len() <= count);
    let first = 2 * record_size + 8 * (offset - 1);
    let (words, _) = data[first..first + 8 * count].as_chunks_mut::<8>();
    for (index, word) in words.iter_mut().enumerate() {
        let value = coefficients.get(index).copied().unwrap_or(0.0);
        word.copy_from_slice(&value.to_le_bytes());
    }
}

#[test]
fn de_recovers_finite_geocentric_state_after_barycentric_overflow() {
    let mut data = RECORDS.to_vec();
    let record_size = DeEphemeris::from_bytes(RECORDS).unwrap().record_size();
    let large = f64::MAX / 16.0;
    let position = 1.215_058_439_582_919_3e-199;
    let velocity = 7.031_588_192_030_782e-205;
    for (barycentric, lunar, expected_position, expected_velocity) in [
        ([large, 0.0], [1.0e-200, 0.0], position, 0.0),
        ([0.0, large], [0.0, 1.0e-200], -position, velocity),
    ] {
        set_first_granule_x(&mut data, record_size, 2, &barycentric);
        set_first_granule_x(&mut data, record_size, 10, &barycentric);
        set_first_granule_x(&mut data, record_size, 9, &lunar);
        let de = DeEphemeris::from_bytes(&data).unwrap();
        let epoch = de.coverage_tdb_seconds().0;
        let result = de.sun_moon_at_tdb_seconds(epoch).unwrap();
        assert_eq!(result.sun.position_m[0], expected_position);
        assert_eq!(result.sun.velocity_m_s[0], expected_velocity);
        assert_eq!(
            de.sun_moon_positions_at_tdb_seconds(epoch).unwrap().sun[0],
            expected_position
        );
    }
}

// At t = -1, Tₖ = (-1)ᵏ and the first three derivatives are 1, -4, 9.
// Sun and EMB granules last 16 days. Moon granules last 4 days.
// The columns give the coefficient and its rounded position and velocity
// contributions for Sun or EMB at that epoch, before the body's sign.
const SMALL_TERMS: [(f64, f64, f64); 3] = [
    (1.0e-200, 1.0e-197, 1.446_759_259_259_259_1e-203),
    (1.0, 1000.0, 0.001_446_759_259_259_259_2),
    (1.5e8, 1.5e11, 217_013.888_888_888_88),
];

#[test]
fn de_preserves_small_terms_after_one_barycentric_body_overflows() {
    let mut data = RECORDS.to_vec();
    let original = DeEphemeris::from_bytes(RECORDS).unwrap();
    let size = original.record_size();
    let epoch = original.coverage_tdb_seconds().0;
    let large = f64::MAX / 16.0;
    let lunar = 2.0_f64.powi(1010);
    let lunar_offset = lunar / (1.0 + original.earth_moon_mass_ratio());

    for (large_slot, small_slot, sign) in [(10, 2, -1.0), (2, 10, 1.0)] {
        for (small, expected, _) in SMALL_TERMS {
            set_first_granule_x(&mut data, size, large_slot, &[large, large, lunar_offset]);
            set_first_granule_x(&mut data, size, small_slot, &[small]);
            set_first_granule_x(&mut data, size, 9, &[sign * lunar]);
            let de = DeEphemeris::from_bytes(&data).unwrap();
            let result = de.sun_moon_positions_at_tdb_seconds(epoch).unwrap();
            assert_eq!(result.sun[0].to_bits(), (sign * expected).to_bits());
        }
    }
}

#[test]
fn de_preserves_small_velocity_after_one_barycentric_body_overflows() {
    let mut data = RECORDS.to_vec();
    let original = DeEphemeris::from_bytes(RECORDS).unwrap();
    let size = original.record_size();
    let epoch = original.coverage_tdb_seconds().0;
    let large = 2.0_f64.powi(1019);
    let lunar = 2.0_f64.powi(1010);
    let lunar_offset = lunar / (1.0 + original.earth_moon_mass_ratio());

    for (large_slot, small_slot, sign) in [(10, 2, -1.0), (2, 10, 1.0)] {
        for (small, _, expected) in SMALL_TERMS {
            set_first_granule_x(
                &mut data,
                size,
                large_slot,
                &[0.75 * large, large, 0.25 * large, lunar_offset],
            );
            set_first_granule_x(&mut data, size, small_slot, &[0.0, small]);
            set_first_granule_x(&mut data, size, 9, &[0.0, 0.0, 0.0, sign * lunar / 4.0]);
            let de = DeEphemeris::from_bytes(&data).unwrap();
            let result = de.sun_moon_at_tdb_seconds(epoch).unwrap();
            assert_eq!(
                result.sun.velocity_m_s[0].to_bits(),
                (sign * expected).to_bits()
            );
        }
    }
}

#[test]
fn de_preserves_small_barycentric_terms_after_lunar_overflow() {
    let mut data = RECORDS.to_vec();
    let original = DeEphemeris::from_bytes(RECORDS).unwrap();
    let size = original.record_size();
    let epoch = original.coverage_tdb_seconds().0;
    let large = f64::MAX / 16.0;
    let lunar = 2.0_f64.powi(1010);
    let lunar_offset = lunar / (1.0 + original.earth_moon_mass_ratio());
    set_first_granule_x(&mut data, size, 9, &[large, large, lunar]);
    for (small, expected, _) in SMALL_TERMS {
        for (sun, emb) in [(small, lunar_offset), (-lunar_offset, -small)] {
            set_first_granule_x(&mut data, size, 10, &[sun]);
            set_first_granule_x(&mut data, size, 2, &[emb]);
            let de = DeEphemeris::from_bytes(&data).unwrap();
            let result = de.sun_moon_positions_at_tdb_seconds(epoch).unwrap();
            assert_eq!(result.sun[0].to_bits(), expected.to_bits());
            assert_eq!(result.moon[0].to_bits(), (1000.0 * lunar).to_bits());
        }
    }
}

#[test]
fn de_preserves_small_barycentric_velocity_after_lunar_overflow() {
    let mut data = RECORDS.to_vec();
    let original = DeEphemeris::from_bytes(RECORDS).unwrap();
    let size = original.record_size();
    let epoch = original.coverage_tdb_seconds().0;
    let large = 2.0_f64.powi(1019);
    let lunar = 2.0_f64.powi(1010);
    let lunar_offset = lunar / (1.0 + original.earth_moon_mass_ratio());
    set_first_granule_x(
        &mut data,
        size,
        9,
        &[0.75 * large, large, 0.25 * large, lunar / 4.0],
    );
    for (small, _, expected) in SMALL_TERMS {
        let small_body = [0.0, small, 0.0, 0.0];
        let offset_body = [0.0, 0.0, 0.0, lunar_offset];
        for (sun, emb) in [
            (small_body, offset_body),
            (offset_body.map(|v| -v), small_body.map(|v| -v)),
        ] {
            set_first_granule_x(&mut data, size, 10, &sun);
            set_first_granule_x(&mut data, size, 2, &emb);
            let de = DeEphemeris::from_bytes(&data).unwrap();
            let result = de.sun_moon_at_tdb_seconds(epoch).unwrap();
            assert_eq!(result.sun.velocity_m_s[0].to_bits(), expected.to_bits());
            assert_eq!(result.moon.velocity_m_s[0].to_bits(), 0x7eaa_aaaa_aaaa_aaaa);
        }
    }
}

#[test]
fn de_preserves_small_lunar_coefficients_after_internal_overflow() {
    let mut data = RECORDS.to_vec();
    let original = DeEphemeris::from_bytes(RECORDS).unwrap();
    let size = original.record_size();
    let epoch = original.coverage_tdb_seconds().0;
    set_first_granule_x(&mut data, size, 10, &[]);
    set_first_granule_x(&mut data, size, 2, &[]);

    let large = f64::MAX / 16.0;
    set_first_granule_x(&mut data, size, 9, &[large, large, 1.0e-200]);
    let de = DeEphemeris::from_bytes(&data).unwrap();
    let position = de.sun_moon_positions_at_tdb_seconds(epoch).unwrap();
    assert_eq!(position.moon[0].to_bits(), 0x1707_eb9a_a8cf_1dde);
    assert_eq!(position.sun[0].to_bits(), 0x16a2_99f1_905a_9e45);

    let large = 2.0_f64.powi(1019);
    set_first_granule_x(
        &mut data,
        size,
        9,
        &[0.75 * large, large, 0.25 * large, 1.0e-200],
    );
    let de = DeEphemeris::from_bytes(&data).unwrap();
    let state = de.sun_moon_at_tdb_seconds(epoch).unwrap();
    assert_eq!(state.moon.velocity_m_s[0].to_bits(), 0x1624_6979_bc6c_790f);
    assert_eq!(state.sun.velocity_m_s[0].to_bits(), 0x15bf_bf0c_e8b5_f638);
}

#[test]
fn de_reports_undefined_and_overflowing_components() {
    let mut data = RECORDS.to_vec();
    let original = DeEphemeris::from_bytes(RECORDS).unwrap();
    let size = original.record_size();
    let epoch = original.coverage_tdb_seconds().0;
    for coefficient in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX / 16.0] {
        set_first_granule_x(&mut data, size, 10, &[coefficient]);
        let de = DeEphemeris::from_bytes(&data).unwrap();
        assert!(matches!(
            de.sun_moon_positions_at_tdb_seconds(epoch),
            Err(DeError::NonFiniteState)
        ));
    }
}

#[test]
fn de_uses_the_current_final_granule_start_across_a_gap() {
    let size = DeEphemeris::from_bytes(RECORDS).unwrap().record_size();
    let record = |index: usize| &RECORDS[(index + 2) * size..(index + 3) * size];
    let mut header = RECORDS[..2 * size].to_vec();
    let sun_layout = 2696 + 12 * 10;
    header[sun_layout + 4..sun_layout + 8].copy_from_slice(&22_u32.to_le_bytes());
    header[sun_layout + 8..sun_layout + 12].copy_from_slice(&1_u32.to_le_bytes());

    let de = DeEphemeris::from_bytes(&[&header, record(0), record(34)].concat()).unwrap();
    let current = DeEphemeris::from_bytes(&[&header, record(34)].concat()).unwrap();
    let epoch = 4_732_603_200.0;
    let expected = current.sun_moon_at_tdb_seconds(epoch).unwrap();
    assert_eq!(de.sun_moon_at_tdb_seconds(epoch).unwrap(), expected);
    assert!(matches!(
        de.sun_moon_at_tdb_seconds(epoch - 1.0),
        Err(DeError::OutsideCoverage { .. })
    ));
}

#[test]
fn de_uses_the_preceding_final_granule_endpoint_when_contiguous() {
    let size = DeEphemeris::from_bytes(RECORDS).unwrap().record_size();
    let mut data = RECORDS[..2 * size].to_vec();
    let sun_layout = 2696 + 12 * 10;
    data[sun_layout + 4..sun_layout + 8].copy_from_slice(&22_u32.to_le_bytes());
    data[sun_layout + 8..sun_layout + 12].copy_from_slice(&1_u32.to_le_bytes());

    let mut endpoint = data.clone();
    endpoint.extend_from_slice(&RECORDS[36 * size..37 * size]);
    set_first_granule_x(&mut endpoint, size, 10, &[1.0]);
    let expected = DeEphemeris::from_bytes(&endpoint).unwrap();

    data.extend_from_slice(&RECORDS[35 * size..]);
    set_first_granule_x(&mut data, size, 10, &[0.0, 1.0]);
    let de = DeEphemeris::from_bytes(&data).unwrap();
    let epoch = 4_732_603_200.0;
    let sun_x = |ephemeris: DeEphemeris| {
        ephemeris
            .sun_moon_positions_at_tdb_seconds(epoch)
            .unwrap()
            .sun[0]
            .to_bits()
    };
    assert_eq!(sun_x(de), sun_x(expected));
}

#[test]
fn de_accepts_big_endian_records() {
    let little = DeEphemeris::from_bytes(RECORDS).unwrap();
    let mut data = RECORDS.to_vec();
    for offset in [2652, 2660, 2668, 2680, 2688] {
        data[offset..offset + 8].reverse();
    }
    for offset in std::iter::once(2676).chain((2696..2860).step_by(4)) {
        data[offset..offset + 4].reverse();
    }
    for word in data[little.record_size()..].as_chunks_mut::<8>().0 {
        word.reverse();
    }
    let big = DeEphemeris::from_bytes(&data).unwrap();
    let epoch = little.coverage_tdb_seconds().0 + 10_000.25;
    assert_eq!(big.version(), little.version());
    assert_eq!(big.constants(), little.constants());
    assert_eq!(
        big.sun_moon_at_tdb_seconds(epoch).unwrap(),
        little.sun_moon_at_tdb_seconds(epoch).unwrap()
    );
}
