use serde_json::Value;
use sidereon_core::astro::bodies::sun_moon_ecef_with_polar_motion;
use sidereon_core::astro::frames::transforms::PolarMotion;
use sidereon_core::astro::time::TimeScales;
use sidereon_core::tides::{
    ocean_tide_loading, parse_ocean_loading_blq_block, parse_ocean_loading_blq_blocks,
    solid_earth_pole_tide, solid_earth_tide, station_displacement_ecef_m,
    station_displacement_ecef_m_batch, write_ocean_loading_blq_blocks, BlqParseErrorKind,
    BlqWriteErrorKind, OceanLoadingBlq, OceanLoadingBlqBlock, OceanLoadingBlqComment,
    OceanLoadingBlqCommentPlacement, OceanTideConstituent, StationDisplacementEpoch,
    StationDisplacementOptions, StationDisplacementPosition, TideError, TideInputErrorKind,
    NUM_OCEAN_CONSTITUENTS,
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

    let encoded = block.to_blq_block().expect("write ZIM2 BLQ block");
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

// Station block for Onsala as published in the ocean tide loading provider's
// BLQ example page (barre.oso.chalmers.se/loading/example_blq.html), with the
// page's non-breaking spaces written as spaces. The station name starts in the
// third column and the column order is declared once in the file header.
const ONSALA_PROVIDER_BLQ: &str = "\
$$ Ocean loading displacement
$$
$$ Calculated using olfg/olmpp of H.-G. Scherneck
$$
$$ COLUMN ORDER:  M2  S2  N2  K2  K1  O1  P1  Q1  MF  MM SSA
$$
$$ ROW ORDER:
$$ AMPLITUDES (m)
$$   RADIAL
$$   TANGENTL    EW
$$   TANGENTL    NS
$$ PHASES (degrees)
$$   RADIAL
$$   TANGENTL    EW
$$   TANGENTL    NS
$$
$$ Displacement is defined positive in upwards, South and West direction.
$$ The phase lag is relative to Greenwich and lags positive. The
$$ Gutenberg-Bullen Green's function is used. In the ocean tide model the
$$ deficit of tidal water mass has been corrected by subtracting a uniform
$$ layer of water with a certain phase lag globally.
$$
$$
$$ Complete <model name> : No interpolation of ocean model was necessary
$$ <model name>_PP       : Ocean model has near the station been interpolated
$$
$$ Ocean tide model: GOT00.2, long period tides from FES99
$$
$$
  Onsala
$$ GOT00.2_PP ID: Aug  16, 2001 13:35
$$ Computed by OLMPP by H G Scherneck, Onsala Space Observatory, 2001
$$ Onsala,                              RADI TANG lon/lat:   11.9264   57.3958
  .00366 .00123 .00089 .00032 .00223 .00115 .00071 .00009 .00091 .00048 .00042
  .00149 .00035 .00040 .00009 .00046 .00043 .00015 .00009 .00013 .00006 .00007
  .00069 .00027 .00020 .00004 .00029 .00014 .00009 .00004 .00003 .00002 .00001
   -62.3  -51.3  -94.8  -39.7  -57.7 -110.6  -60.3 -164.6    9.9    5.8    2.1
    87.0  114.0   57.2  126.4  102.3   35.4   97.0   -6.8 -166.3 -169.8 -177.7
   109.9  152.4   86.4  149.1   50.7  -59.4   47.7  173.6  -27.8   -1.5    7.3
";

/// Station lookup as RTKLIB `readblq` performs it: lines starting `$$` or
/// shorter than two bytes are skipped, the name is the first
/// whitespace-delimited token from the third column (`buff+2`, at most 16
/// characters), compared case-insensitively; the next six non-`$$` lines with
/// eleven numbers are the coefficient rows, in the standard column order.
fn rtklib_style_readblq(text: &str, station: &str) -> Option<[[f64; NUM_OCEAN_CONSTITUENTS]; 6]> {
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.starts_with("$$") || line.len() < 2 {
            continue;
        }
        let Some(name) = line
            .get(2..)
            .and_then(|rest| rest.split_whitespace().next())
        else {
            continue;
        };
        let name = name
            .chars()
            .take(16)
            .collect::<String>()
            .to_ascii_uppercase();
        if name != station.to_ascii_uppercase() {
            continue;
        }
        let mut rows = [[0.0; NUM_OCEAN_CONSTITUENTS]; 6];
        let mut count = 0;
        for line in lines.by_ref() {
            if line.starts_with("$$") {
                continue;
            }
            let values = line
                .split_whitespace()
                .map_while(|token| token.parse::<f64>().ok())
                .collect::<Vec<_>>();
            if values.len() < NUM_OCEAN_CONSTITUENTS {
                continue;
            }
            rows[count].copy_from_slice(&values[..NUM_OCEAN_CONSTITUENTS]);
            count += 1;
            if count == 6 {
                return Some(rows);
            }
        }
        return None;
    }
    None
}

fn rows_of(blq: &OceanLoadingBlq) -> [[f64; NUM_OCEAN_CONSTITUENTS]; 6] {
    [
        blq.amplitude_m[0],
        blq.amplitude_m[1],
        blq.amplitude_m[2],
        blq.phase_deg[0],
        blq.phase_deg[1],
        blq.phase_deg[2],
    ]
}

#[test]
fn blq_writer_puts_the_station_where_the_provider_and_rtklib_read_it() {
    let block = parse_ocean_loading_blq_block(ONSALA_PROVIDER_BLQ).expect("parse provider block");
    assert_eq!(block.station, "Onsala");
    assert_eq!(block.coefficients.amplitude_m[0][0], 0.00366);
    assert_eq!(block.coefficients.phase_deg[2][10], 7.3);
    // The provider example is itself readable by the station-field rule.
    assert_eq!(
        rtklib_style_readblq(ONSALA_PROVIDER_BLQ, "ONSALA"),
        Some(rows_of(&block.coefficients))
    );

    let encoded = block.to_blq_block().expect("write provider block");
    assert!(encoded.contains("\n  Onsala\n"), "{encoded}");
    assert_eq!(
        rtklib_style_readblq(&encoded, "ONSALA"),
        Some(rows_of(&block.coefficients))
    );
    assert_eq!(
        parse_ocean_loading_blq_block(&encoded).expect("reparse written block"),
        block
    );

    let zim2 = parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK).expect("parse ZIM2 BLQ block");
    let encoded = zim2.to_blq_block().expect("write ZIM2 BLQ block");
    assert_eq!(
        rtklib_style_readblq(&encoded, "zim2"),
        Some(rows_of(&EXPECTED_ZIM2_BLQ))
    );
    // A station written in column 1 is read as its name without the first
    // two characters, which is why the writer indents it.
    assert_eq!(
        rtklib_style_readblq(&encoded.replace("\n  ZIM2\n", "\nZIM2\n"), "ZIM2"),
        None
    );
}

#[test]
fn blq_parser_retains_block_comments_in_order_and_the_writer_restates_them() {
    let block = parse_ocean_loading_blq_block(ONSALA_PROVIDER_BLQ).expect("parse provider block");
    let before_station = block
        .comments
        .iter()
        .filter(|comment| comment.placement == OceanLoadingBlqCommentPlacement::BeforeStation)
        .map(|comment| comment.line.as_str())
        .collect::<Vec<_>>();
    let before_rows = block
        .comments
        .iter()
        .filter(|comment| comment.placement == OceanLoadingBlqCommentPlacement::BeforeRow(0))
        .map(|comment| comment.line.as_str())
        .collect::<Vec<_>>();
    assert_eq!(before_station.len(), 29);
    assert_eq!(before_station[0], "$$ Ocean loading displacement");
    assert_eq!(
        before_station[4],
        "$$ COLUMN ORDER:  M2  S2  N2  K2  K1  O1  P1  Q1  MF  MM SSA"
    );
    assert_eq!(
        before_station[26],
        "$$ Ocean tide model: GOT00.2, long period tides from FES99"
    );
    assert_eq!(
        before_rows,
        [
            "$$ GOT00.2_PP ID: Aug  16, 2001 13:35",
            "$$ Computed by OLMPP by H G Scherneck, Onsala Space Observatory, 2001",
            "$$ Onsala,                              RADI TANG lon/lat:   11.9264   57.3958",
        ]
    );
    assert_eq!(block.comments.len(), 32);

    // Every retained line appears in the written block, in the same order.
    let encoded = block.to_blq_block().expect("write provider block");
    let written_comments = encoded
        .lines()
        .filter(|line| line.starts_with("$$"))
        .collect::<Vec<_>>();
    let retained = block
        .comments
        .iter()
        .map(|comment| comment.line.as_str())
        .collect::<Vec<_>>();
    assert_eq!(written_comments, retained);

    // Comments between rows and after the last block keep their places.
    let text = "\
$$ header
AAA
$$ before row 0
 1 0 0 0 0 0 0 0 0 0 0
 2 0 0 0 0 0 0 0 0 0 0
# between rows 2 and 3
 3 0 0 0 0 0 0 0 0 0 0
 4 0 0 0 0 0 0 0 0 0 0
 5 0 0 0 0 0 0 0 0 0 0
 6 0 0 0 0 0 0 0 0 0 0
$$ between blocks
BBB
 7 0 0 0 0 0 0 0 0 0 0
 8 0 0 0 0 0 0 0 0 0 0
 9 0 0 0 0 0 0 0 0 0 0
 10 0 0 0 0 0 0 0 0 0 0
 11 0 0 0 0 0 0 0 0 0 0
 12 0 0 0 0 0 0 0 0 0 0
! trailing
";
    let blocks = parse_ocean_loading_blq_blocks(text).expect("parse commented blocks");
    let comment = |placement, line: &str| OceanLoadingBlqComment {
        placement,
        line: line.to_string(),
    };
    assert_eq!(
        blocks[0].comments,
        vec![
            comment(OceanLoadingBlqCommentPlacement::BeforeStation, "$$ header"),
            comment(
                OceanLoadingBlqCommentPlacement::BeforeRow(0),
                "$$ before row 0"
            ),
            comment(
                OceanLoadingBlqCommentPlacement::BeforeRow(2),
                "# between rows 2 and 3"
            ),
        ]
    );
    assert_eq!(
        blocks[1].comments,
        vec![
            comment(
                OceanLoadingBlqCommentPlacement::BeforeStation,
                "$$ between blocks"
            ),
            comment(OceanLoadingBlqCommentPlacement::AfterRows, "! trailing"),
        ]
    );
    let written = write_ocean_loading_blq_blocks(&blocks).expect("write commented blocks");
    assert_eq!(
        parse_ocean_loading_blq_blocks(&written).expect("reparse commented blocks"),
        blocks
    );
}

#[test]
fn blq_multi_block_writer_carries_a_file_header_like_the_parser() {
    let text = "\
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
";
    let blocks = parse_ocean_loading_blq_blocks(text).expect("parse two blocks");
    assert_eq!(blocks[1].coefficients.amplitude_m[0][0], 14.0);
    assert!(blocks[1].comments.is_empty());

    let written = write_ocean_loading_blq_blocks(&blocks).expect("write two blocks");
    // Both blocks are written in the declared S2-first order.
    assert!(
        written.contains("\n  BBB\n               13               14 "),
        "{written}"
    );
    assert_eq!(
        parse_ocean_loading_blq_blocks(&written).expect("reparse two blocks"),
        blocks
    );

    // Written alone, BBB has no header and uses the standard order.
    let alone = blocks[1].to_blq_block().expect("write BBB alone");
    assert_eq!(
        parse_ocean_loading_blq_block(&alone).expect("reparse BBB alone"),
        blocks[1]
    );
}

fn blq_parse_error(text: &str) -> (usize, BlqParseErrorKind) {
    match parse_ocean_loading_blq_blocks(text).expect_err("BLQ text must be refused") {
        TideError::BlqParse { line, kind } => (line, kind),
        other => panic!("unexpected BLQ error {other:?}"),
    }
}

const SIX_ROWS: &str = "\
 1 2 0 0 0 0 0 0 0 0 0
 3 4 0 0 0 0 0 0 0 0 0
 5 6 0 0 0 0 0 0 0 0 0
 7 8 0 0 0 0 0 0 0 0 0
 9 10 0 0 0 0 0 0 0 0 0
 11 12 0 0 0 0 0 0 0 0 0
";

#[test]
fn blq_column_order_headers_follow_a_grammar_and_refuse_what_they_cannot_state() {
    // A declared header whose labels are not constituents is refused, not
    // treated as absent.
    assert_eq!(
        blq_parse_error(&format!("$$ Column order: nonsense\nAAA\n{SIX_ROWS}")),
        (
            1,
            BlqParseErrorKind::UnsupportedConstituent {
                constituent: "NONSENSE".to_string(),
            }
        )
    );
    // Eleven known labels followed by an extra one is not filtered down.
    assert_eq!(
        blq_parse_error(&format!(
            "$$ COLUMN ORDER: M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA EXTRA\nAAA\n{SIX_ROWS}"
        )),
        (
            1,
            BlqParseErrorKind::UnsupportedConstituent {
                constituent: "EXTRA".to_string(),
            }
        )
    );
    assert_eq!(
        blq_parse_error(&format!(
            "$$ COLUMN ORDER: M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA M2\nAAA\n{SIX_ROWS}"
        )),
        (
            1,
            BlqParseErrorKind::DuplicateConstituent {
                constituent: "M2".to_string(),
            }
        )
    );
    assert_eq!(
        blq_parse_error(&format!("$$ Column order: M2 S2 N2\nAAA\n{SIX_ROWS}")),
        (
            1,
            BlqParseErrorKind::WrongColumnCount {
                expected: NUM_OCEAN_CONSTITUENTS,
                found: 3,
            }
        )
    );
    // A bare label list is a header too.
    assert_eq!(
        blq_parse_error(&format!(
            "$$ M2 S2 N2 K2 K1 O1 P1 Q1 MF MM M4\nAAA\n{SIX_ROWS}"
        )),
        (
            1,
            BlqParseErrorKind::UnsupportedConstituent {
                constituent: "M4".to_string(),
            }
        )
    );

    // Prose that mentions a label is a comment, not a header.
    let prose = format!(
        "$$ These columns were computed for ZIM2\n$$ K1 and O1 are diurnal\nAAA\n{SIX_ROWS}"
    );
    let block = parse_ocean_loading_blq_block(&prose).expect("prose comments are comments");
    assert_eq!(block.coefficients.amplitude_m[0][0], 1.0);
    assert_eq!(block.coefficients.amplitude_m[0][1], 2.0);
    assert_eq!(block.comments.len(), 2);

    // A header after the station line and before the rows applies to the
    // whole block.
    let after_station =
        format!("AAA\n$$ Column order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA\n{SIX_ROWS}");
    let block = parse_ocean_loading_blq_block(&after_station).expect("header before rows");
    assert_eq!(block.coefficients.amplitude_m[0][0], 2.0);
    assert_eq!(block.coefficients.phase_deg[2][0], 12.0);

    // A comment whose labels are exactly the eleven constituents is a header
    // even with prose before the list, as the earlier parser read it.
    let labelled =
        format!("$$ Constituent order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA\nAAA\n{SIX_ROWS}");
    let block = parse_ocean_loading_blq_block(&labelled).expect("labelled header");
    assert_eq!(block.coefficients.amplitude_m[0][0], 2.0);
    assert_eq!(block.coefficients.amplitude_m[0][1], 1.0);
    let written = block.to_blq_block().expect("write labelled block");
    assert!(
        written.contains("\n  AAA\n                1                2 "),
        "{written}"
    );
    assert_eq!(
        parse_ocean_loading_blq_block(&written).expect("reparse labelled block"),
        block
    );
    // A misspelled order declaration is refused by name, not ignored: SA is
    // not the Ssa constituent, with or without a trailing full stop.
    for declaration in [
        "$$ Constituent order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SA",
        "$$ Constituent order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SA.",
    ] {
        assert_eq!(
            blq_parse_error(&format!("{declaration}\nAAA\n{SIX_ROWS}")),
            (
                1,
                BlqParseErrorKind::UnsupportedConstituent {
                    constituent: "SA".to_string(),
                }
            ),
            "{declaration}"
        );
    }
    assert_eq!(
        blq_parse_error(&format!("$$ Constituent order: S2 M2 N2\nAAA\n{SIX_ROWS}")),
        (
            1,
            BlqParseErrorKind::WrongColumnCount {
                expected: NUM_OCEAN_CONSTITUENTS,
                found: 3,
            }
        )
    );
    // Prose that lists the eleven constituents without declaring an order is
    // a comment: it neither permutes the rows nor resets a declared order.
    let listed = [
        "$$ Column order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA",
        "$$ diurnal K1 O1 P1 Q1, semidiurnal M2 S2 N2 K2, long-period MF MM SSA",
        "$$ The constituents are M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA",
        "AAA",
    ]
    .join("\n");
    let block = parse_ocean_loading_blq_block(&format!("{listed}\n{SIX_ROWS}"))
        .expect("prose listing constituents");
    assert_eq!(block.coefficients.amplitude_m[0][0], 2.0);
    assert_eq!(block.coefficients.amplitude_m[0][1], 1.0);
    assert_eq!(block.comments.len(), 3);

    // An order of things that are not constituents is prose.
    let stations =
        parse_ocean_loading_blq_block(&format!("$$ Station order: ZIM2 GOL2\nAAA\n{SIX_ROWS}"))
            .expect("station order is a comment");
    assert_eq!(stations.coefficients.amplitude_m[0][0], 1.0);
    assert_eq!(stations.comments.len(), 1);

    // Prose with one or two labels stays a comment.
    for prose in [
        "$$ M2 dominates at this site",
        "$$ Order of K1 and O1 phases",
    ] {
        let block = parse_ocean_loading_blq_block(&format!("{prose}\nAAA\n{SIX_ROWS}"))
            .expect("prose comment");
        assert_eq!(block.coefficients.amplitude_m[0][0], 1.0, "{prose}");
    }

    // A header between coefficient rows sets the order of the rows after it,
    // as the parser's rule for headers states.
    let (first, rest) = SIX_ROWS.split_at(SIX_ROWS.find("\n 5 6").expect("third row") + 1);
    let mid_block =
        format!("AAA\n{first}$$ Column order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA\n{rest}");
    let block = parse_ocean_loading_blq_block(&mid_block).expect("mid-block header");
    // Rows 0 and 1 in the standard order, rows 2 to 5 with S2 first.
    assert_eq!(block.coefficients.amplitude_m[0][..2], [1.0, 2.0]);
    assert_eq!(block.coefficients.amplitude_m[1][..2], [3.0, 4.0]);
    assert_eq!(block.coefficients.amplitude_m[2][..2], [6.0, 5.0]);
    assert_eq!(block.coefficients.phase_deg[0][..2], [8.0, 7.0]);
    assert_eq!(block.coefficients.phase_deg[2][..2], [12.0, 11.0]);
    assert_eq!(
        block.comments,
        vec![OceanLoadingBlqComment {
            placement: OceanLoadingBlqCommentPlacement::BeforeRow(2),
            line: "$$ Column order: S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA".to_string(),
        }]
    );
    let written = block.to_blq_block().expect("write mid-block header");
    assert_eq!(
        parse_ocean_loading_blq_block(&written).expect("reparse mid-block header"),
        block
    );
}

fn blq_write_error(block: &OceanLoadingBlqBlock) -> BlqWriteErrorKind {
    match block.to_blq_block().expect_err("block must be refused") {
        TideError::BlqWrite { block: 0, kind } => kind,
        other => panic!("unexpected BLQ write error {other:?}"),
    }
}

#[test]
fn blq_writer_refuses_what_the_parser_would_not_read_back() {
    let valid = parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK).expect("parse ZIM2 BLQ block");
    let with_station = |station: &str| {
        let mut block = valid.clone();
        block.station = station.to_string();
        blq_write_error(&block)
    };
    assert_eq!(with_station("A\nB"), BlqWriteErrorKind::StationLineBreak);
    assert_eq!(with_station(""), BlqWriteErrorKind::EmptyStation);
    assert_eq!(
        with_station(" ZIM2"),
        BlqWriteErrorKind::StationSurroundingWhitespace
    );
    assert_eq!(
        with_station("$$ZIM2"),
        BlqWriteErrorKind::StationReadsAsComment
    );
    assert_eq!(
        with_station("#ZIM2"),
        BlqWriteErrorKind::StationReadsAsComment
    );
    assert_eq!(
        with_station("-12.5"),
        BlqWriteErrorKind::StationReadsAsCoefficientRow
    );
    assert_eq!(
        with_station("M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA"),
        BlqWriteErrorKind::StationReadsAsHeader
    );

    let mut nan = valid.clone();
    nan.coefficients.amplitude_m[0][0] = f64::NAN;
    assert_eq!(
        blq_write_error(&nan),
        BlqWriteErrorKind::NonFiniteCoefficient {
            row: 0,
            constituent: OceanTideConstituent::M2,
        }
    );
    let mut infinite = valid.clone();
    infinite.coefficients.phase_deg[2][5] = f64::INFINITY;
    assert_eq!(
        blq_write_error(&infinite),
        BlqWriteErrorKind::NonFiniteCoefficient {
            row: 5,
            constituent: OceanTideConstituent::O1,
        }
    );

    let with_comment = |placement, line: &str| {
        let mut block = valid.clone();
        block.comments.push(OceanLoadingBlqComment {
            placement,
            line: line.to_string(),
        });
        block
    };
    let last = valid.comments.len();
    assert_eq!(
        blq_write_error(&with_comment(
            OceanLoadingBlqCommentPlacement::BeforeStation,
            "$$ a\nb"
        )),
        BlqWriteErrorKind::CommentLineBreak { index: last }
    );
    assert_eq!(
        blq_write_error(&with_comment(
            OceanLoadingBlqCommentPlacement::BeforeStation,
            "no marker"
        )),
        BlqWriteErrorKind::NotACommentLine { index: last }
    );
    assert_eq!(
        blq_write_error(&with_comment(
            OceanLoadingBlqCommentPlacement::BeforeStation,
            "   "
        )),
        BlqWriteErrorKind::NotACommentLine { index: last }
    );
    assert_eq!(
        blq_write_error(&with_comment(
            OceanLoadingBlqCommentPlacement::BeforeRow(6),
            "$$ after the rows"
        )),
        BlqWriteErrorKind::CommentPlacementOutOfRange { index: last }
    );
    assert_eq!(
        blq_write_error(&with_comment(
            OceanLoadingBlqCommentPlacement::BeforeStation,
            "$$ Column order: nonsense"
        )),
        BlqWriteErrorKind::InvalidHeader {
            index: last,
            kind: BlqParseErrorKind::UnsupportedConstituent {
                constituent: "NONSENSE".to_string(),
            },
        }
    );
    // Comments not grouped by placement in file order would be read back
    // regrouped.
    let mut out_of_order = with_comment(OceanLoadingBlqCommentPlacement::BeforeRow(3), "$$ a");
    out_of_order.comments.push(OceanLoadingBlqComment {
        placement: OceanLoadingBlqCommentPlacement::BeforeRow(1),
        line: "$$ b".to_string(),
    });
    assert_eq!(
        blq_write_error(&out_of_order),
        BlqWriteErrorKind::CommentsOutOfPlacementOrder { index: last + 1 }
    );

    // A comment after the rows is the last block's alone: before another
    // block the parser reads it as that block's.
    let trailing = with_comment(OceanLoadingBlqCommentPlacement::AfterRows, "$$ trailing");
    let written = trailing
        .to_blq_block()
        .expect("trailing comment on the only block");
    assert_eq!(
        parse_ocean_loading_blq_block(&written).expect("reparse trailing comment"),
        trailing
    );
    assert_eq!(
        write_ocean_loading_blq_blocks(&[trailing.clone(), valid.clone()])
            .expect_err("trailing comment before another block"),
        TideError::BlqWrite {
            block: 0,
            kind: BlqWriteErrorKind::AfterRowsBeforeAnotherBlock { index: last },
        }
    );
    let pair = [valid.clone(), trailing];
    let written = write_ocean_loading_blq_blocks(&pair).expect("trailing comment on last block");
    assert_eq!(
        parse_ocean_loading_blq_blocks(&written).expect("reparse pair"),
        pair
    );

    // A retained comment between rows, and an uncommented header before the
    // station, are written and read back unchanged.
    let mut block = with_comment(
        OceanLoadingBlqCommentPlacement::BeforeRow(3),
        "$$ phases follow",
    );
    block.comments.insert(
        0,
        OceanLoadingBlqComment {
            placement: OceanLoadingBlqCommentPlacement::BeforeStation,
            line: "M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA".to_string(),
        },
    );
    let written = block.to_blq_block().expect("write commented block");
    assert_eq!(
        parse_ocean_loading_blq_block(&written).expect("reparse commented block"),
        block
    );
}
