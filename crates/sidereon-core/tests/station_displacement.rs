use serde_json::Value;
use sidereon_core::astro::bodies::sun_moon_ecef_with_polar_motion;
use sidereon_core::astro::frames::transforms::PolarMotion;
use sidereon_core::astro::time::TimeScales;
use sidereon_core::tides::{
    ocean_tide_loading, parse_ocean_loading_blq_block, parse_ocean_loading_blq_blocks,
    solid_earth_pole_tide, solid_earth_tide, station_displacement_ecef_m,
    station_displacement_ecef_m_batch, BlqParseErrorKind, OceanLoadingBlq,
    StationDisplacementEpoch, StationDisplacementOptions, StationDisplacementPosition, TideError,
    TideInputErrorKind, NUM_OCEAN_CONSTITUENTS,
};
use sidereon_core::{geodetic_to_itrf, Wgs84Geodetic};

const ZIM2_LAT_DEG: f64 = 46.8771;
const ZIM2_LON_DEG: f64 = 7.4650;
const ZIM2_HEIGHT_M: f64 = 956.425;
const ZIM2_XP_ARCSEC: f64 = 0.169_051;
const ZIM2_YP_ARCSEC: f64 = 0.411_760;

// Real ZIM2 public BLQ block from the Onsala Space Observatory ocean tide
// loading provider, site lon/lat 7.4650/46.8771 and ellipsoidal height
// 956.425 m. Column order is the standard Bos-Scherneck/HARDISP BLQ order.
const ZIM2_BLQ_BLOCK: &str = r#"
$$ Station: ZIM2, Zimmerwald
$$ Source: Onsala Space Observatory ocean tide loading provider, ZIM2 public BLQ
$$ Ocean model: GOT4.7, long-period tides from FES99
$$ Column order: M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa
ZIM2
 0.00693 0.00228 0.00148 0.00061 0.00220 0.00094 0.00070 0.00001 0.00047 0.00025 0.00019
 0.00272 0.00076 0.00061 0.00020 0.00036 0.00025 0.00011 0.00005 0.00004 0.00001 0.00002
 0.00061 0.00026 0.00010 0.00009 0.00025 0.00002 0.00008 0.00003 0.00002 0.00000 0.00001
-72.3 -44.2 -90.8 -44.1 -62.9 -94.5 -64.3 171.0 3.4 3.6 1.1
 84.3 115.4 63.3 113.7 98.6 20.7 94.2 -44.5 -170.0 -162.7 -177.8
-29.3 1.7 -44.0 -4.2 44.2 -39.1 43.7 170.1 -93.3 -118.3 -176.4
"#;

const EXPECTED_ZIM2_BLQ: OceanLoadingBlq = OceanLoadingBlq {
    amplitude_m: [
        [
            0.00693, 0.00228, 0.00148, 0.00061, 0.00220, 0.00094, 0.00070, 0.00001, 0.00047,
            0.00025, 0.00019,
        ],
        [
            0.00272, 0.00076, 0.00061, 0.00020, 0.00036, 0.00025, 0.00011, 0.00005, 0.00004,
            0.00001, 0.00002,
        ],
        [
            0.00061, 0.00026, 0.00010, 0.00009, 0.00025, 0.00002, 0.00008, 0.00003, 0.00002,
            0.00000, 0.00001,
        ],
    ],
    phase_deg: [
        [
            -72.3, -44.2, -90.8, -44.1, -62.9, -94.5, -64.3, 171.0, 3.4, 3.6, 1.1,
        ],
        [
            84.3, 115.4, 63.3, 113.7, 98.6, 20.7, 94.2, -44.5, -170.0, -162.7, -177.8,
        ],
        [
            -29.3, 1.7, -44.0, -4.2, 44.2, -39.1, 43.7, 170.1, -93.3, -118.3, -176.4,
        ],
    ],
};

fn zim2_geodetic() -> Wgs84Geodetic {
    Wgs84Geodetic::new(
        ZIM2_LAT_DEG.to_radians(),
        ZIM2_LON_DEG.to_radians(),
        ZIM2_HEIGHT_M,
    )
    .expect("valid ZIM2 geodetic position")
}

fn norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn vec3(value: &Value) -> [f64; 3] {
    let values = value["values"].as_array().expect("values array");
    [
        values[0].as_f64().expect("x"),
        values[1].as_f64().expect("y"),
        values[2].as_f64().expect("z"),
    ]
}

#[test]
fn blq_parser_round_trips_real_public_zim2_block() {
    let block = parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK).expect("parse ZIM2 BLQ block");
    assert_eq!(block.station, "ZIM2");
    assert_eq!(block.coefficients, EXPECTED_ZIM2_BLQ);

    let encoded = block.to_blq_block();
    let reparsed = parse_ocean_loading_blq_block(&encoded).expect("reparse encoded BLQ block");
    assert_eq!(reparsed, block);
}

#[test]
fn blq_parser_rejects_out_of_table_constituent() {
    let bad = ZIM2_BLQ_BLOCK.replace("Ssa", "M4");
    let err = parse_ocean_loading_blq_block(&bad).expect_err("M4 is not in the ARG2 BLQ table");
    match err {
        TideError::BlqParse {
            kind: BlqParseErrorKind::UnsupportedConstituent { constituent },
            ..
        } => assert_eq!(constituent, "M4"),
        other => panic!("unexpected BLQ error {other:?}"),
    }
}

#[test]
fn blq_parser_applies_file_level_header_across_station_blocks() {
    let blocks = parse_ocean_loading_blq_blocks(
        r#"
$$ Column order: S2 M2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa
AAA
 1 2 0 0 0 0 0 0 0 0 0
 3 4 0 0 0 0 0 0 0 0 0
 5 6 0 0 0 0 0 0 0 0 0
 7 8 0 0 0 0 0 0 0 0 0
 9 10 0 0 0 0 0 0 0 0 0
 11 12 0 0 0 0 0 0 0 0 0
BBB
 13 14 0 0 0 0 0 0 0 0 0
 15 16 0 0 0 0 0 0 0 0 0
 17 18 0 0 0 0 0 0 0 0 0
 19 20 0 0 0 0 0 0 0 0 0
 21 22 0 0 0 0 0 0 0 0 0
 23 24 0 0 0 0 0 0 0 0 0
"#,
    )
    .expect("parse two BLQ blocks");

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].station, "AAA");
    assert_eq!(blocks[1].station, "BBB");
    assert_eq!(blocks[0].coefficients.amplitude_m[0][0], 2.0);
    assert_eq!(blocks[0].coefficients.amplitude_m[0][1], 1.0);
    assert_eq!(blocks[1].coefficients.amplitude_m[0][0], 14.0);
    assert_eq!(blocks[1].coefficients.amplitude_m[0][1], 13.0);
}

#[test]
fn station_displacement_entry_sums_components_and_batch_matches_scalar() {
    let block = parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK).expect("parse BLQ");
    let geodetic = zim2_geodetic();
    let receiver = geodetic_to_itrf(geodetic)
        .expect("geodetic to ITRF")
        .as_array();
    let epoch = StationDisplacementEpoch::from_utc(2026, 5, 13, 12, 30, 0.0)
        .with_polar_motion_arcsec(ZIM2_XP_ARCSEC, ZIM2_YP_ARCSEC);
    let mut options = StationDisplacementOptions::default();
    options.solid_earth_tide = true;
    options.pole_tide = true;
    options.ocean_loading = Some(&block.coefficients);

    let got =
        station_displacement_ecef_m(StationDisplacementPosition::from(geodetic), epoch, options)
            .expect("station displacement");

    let ts = TimeScales::from_utc(2026, 5, 13, 12, 30, 0.0).expect("time scales");
    let polar = PolarMotion::from_arcseconds(ZIM2_XP_ARCSEC, ZIM2_YP_ARCSEC).expect("polar motion");
    let sun_moon = sun_moon_ecef_with_polar_motion(&ts, polar).expect("Sun/Moon");
    let solid = solid_earth_tide(&receiver, 2026, 5, 13, 12.5, &sun_moon.sun, &sun_moon.moon)
        .expect("solid tide");
    let pole = solid_earth_pole_tide(&receiver, 2026, 5, 13, 12.5, ZIM2_XP_ARCSEC, ZIM2_YP_ARCSEC)
        .expect("pole tide");
    let ocean =
        ocean_tide_loading(&receiver, 2026, 5, 13, 12.5, &block.coefficients).expect("ocean");

    assert_eq!(got.solid_earth_tide_ecef_m, Some(solid));
    assert_eq!(got.pole_tide_ecef_m, Some(pole));
    assert_eq!(got.ocean_loading_ecef_m, Some(ocean));
    assert_eq!(
        got.ecef_m,
        [
            solid[0] + pole[0] + ocean[0],
            solid[1] + pole[1] + ocean[1],
            solid[2] + pole[2] + ocean[2],
        ]
    );

    let batch = station_displacement_ecef_m_batch(
        StationDisplacementPosition::from(geodetic),
        &[epoch],
        options,
    );
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].as_ref(), Ok(&got));
}

#[test]
fn station_displacement_requires_polar_motion_when_pole_tide_enabled() {
    let geodetic = zim2_geodetic();
    let mut options = StationDisplacementOptions::default();
    options.solid_earth_tide = false;
    options.pole_tide = true;
    options.ocean_loading = None;
    let err = station_displacement_ecef_m(
        StationDisplacementPosition::from(geodetic),
        StationDisplacementEpoch::from_utc(2026, 5, 13, 12, 0, 0.0),
        options,
    )
    .expect_err("pole tide requires polar motion");
    assert_eq!(
        err,
        TideError::MissingInput {
            field: "polar motion"
        }
    );
}

#[test]
fn station_displacement_validates_epoch_without_solid_tide() {
    let block = parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK).expect("parse BLQ");
    let geodetic = zim2_geodetic();
    let mut options = StationDisplacementOptions::default();
    options.solid_earth_tide = false;
    options.pole_tide = false;
    options.ocean_loading = Some(&block.coefficients);

    let err = station_displacement_ecef_m(
        StationDisplacementPosition::from(geodetic),
        StationDisplacementEpoch {
            year: 2026,
            month: 5,
            day: 13,
            hour: 20,
            minute: 99,
            second: 0.0,
            polar_motion: None,
        },
        options,
    )
    .expect_err("invalid minute must be rejected");
    assert_eq!(
        err,
        TideError::InvalidInput {
            field: "civil datetime",
            kind: TideInputErrorKind::InvalidCivilTime,
        }
    );

    let ocean_only = station_displacement_ecef_m(
        StationDisplacementPosition::from(geodetic),
        StationDisplacementEpoch::from_utc(2026, 5, 13, 12, 0, 0.0)
            .with_polar_motion_arcsec(f64::NAN, f64::NAN),
        options,
    )
    .expect("unused polar motion is ignored when solid and pole tide are disabled");
    assert!(ocean_only.ocean_loading_ecef_m.is_some());
}

#[test]
fn station_displacement_magnitude_bounds_hold_on_daily_grid() {
    let geodetic = zim2_geodetic();
    for hour in (0..24).step_by(3) {
        let epoch = StationDisplacementEpoch::from_utc(2026, 5, 13, hour, 0, 0.0);
        let displacement =
            station_displacement_ecef_m(StationDisplacementPosition::from(geodetic), epoch, {
                let mut options = StationDisplacementOptions::default();
                options.solid_earth_tide = true;
                options.pole_tide = false;
                options.ocean_loading = None;
                options
            })
            .expect("solid tide displacement");
        assert!(
            norm(displacement.ecef_m) < 0.5,
            "solid Earth tide magnitude is outside the expected decimetre band"
        );
    }

    let block = parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK).expect("parse BLQ");
    let receiver = geodetic_to_itrf(geodetic)
        .expect("geodetic to ITRF")
        .as_array();
    for hour in (0..24).step_by(3) {
        let ocean = ocean_tide_loading(&receiver, 2026, 5, 13, hour as f64, &block.coefficients)
            .expect("ocean loading");
        assert!(
            norm(ocean) < 0.02,
            "ZIM2 ocean loading magnitude is outside the expected centimetre band"
        );
    }
}

#[test]
fn s2_only_ocean_loading_repeats_on_solar_day() {
    let receiver = geodetic_to_itrf(zim2_geodetic())
        .expect("geodetic to ITRF")
        .as_array();
    let mut blq = OceanLoadingBlq {
        amplitude_m: [[0.0; NUM_OCEAN_CONSTITUENTS]; 3],
        phase_deg: [[0.0; NUM_OCEAN_CONSTITUENTS]; 3],
    };
    blq.amplitude_m[0][1] = 0.01;

    let d0 = ocean_tide_loading(&receiver, 2026, 5, 13, 0.0, &blq).expect("S2 at day 1");
    let d1 = ocean_tide_loading(&receiver, 2026, 5, 14, 0.0, &blq).expect("S2 at day 2");
    assert!(
        norm([d0[0] - d1[0], d0[1] - d1[1], d0[2] - d1[2]]) < 1.0e-6,
        "S2-only ocean loading should repeat after a solar day"
    );
}

#[test]
fn solid_earth_tide_matches_iers_dehant_reference_rows() {
    let doc: Value = serde_json::from_str(include_str!("fixtures/tides/tides_dehant_golden.json"))
        .expect("parse IERS DEHANT fixture");
    let cases = doc["cases"].as_array().expect("cases array");

    for case in cases {
        let id = case["id"].as_str().expect("case id");
        assert!(
            case["source"].as_str().is_some_and(|source| {
                source.contains("IERS Conventions") && source.contains("DEHANTTIDEINEL")
            }),
            "{id} must cite its source row"
        );
        if id == "case_4_2017_01_15" {
            continue;
        }

        let inputs = &case["inputs"];
        let xsta = vec3(&inputs["xsta_m"]);
        let xsun = vec3(&inputs["xsun_m"]);
        let xmon = vec3(&inputs["xmon_m"]);
        let year = inputs["date_utc"]["year"].as_i64().expect("year") as i32;
        let month = inputs["date_utc"]["month"].as_i64().expect("month") as i32;
        let day = inputs["date_utc"]["day"].as_i64().expect("day") as i32;
        let fhr = inputs["fhr_hours"]["value"].as_f64().expect("fhr");
        let expected = vec3(&case["expected"]["dxtide_m"]);

        let got = solid_earth_tide(&xsta, year, month, day, fhr, &xsun, &xmon).expect("solid tide");
        for i in 0..3 {
            assert!(
                (got[i] - expected[i]).abs() < 1.0e-9,
                "{id} component {i}: got {:.18e}, expected {:.18e}",
                got[i],
                expected[i]
            );
        }
    }
}
