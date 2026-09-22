//! Terrain store fixture provenance: DTED tiles under
//! `tests/fixtures/dted/tiles` and points in `tests/fixtures/dted/dted_points.json`
//! are existing repository fixtures generated from the public DTED
//! UHL/DSI/ACC/data-record layout. The HGT void test uses the synthetic
//! `tests/fixtures/dted/hgt/n36_w107_reference.hgt` fixture already committed
//! for the SRTM1-to-DTED converter; its one void sample (-32768, at HGT row
//! 2366 column 2345) converts to the DTED null and reads as an unknown
//! elevation. Legacy store regression cases compare
//! terrain heights by `f64::to_bits()` against committed fixture values, and
//! source-post checks use the public Skadi SRTM1 excerpt in
//! `skadi_n36w107_5x5_posts.json`.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use sidereon_core::data::hgt_to_dted;
use sidereon_core::geoid::egm96_undulation;
use sidereon_core::terrain::{
    DtedHorizontalDatum, DtedInterpolation, DtedLookupOptions, DtedTerrain,
};
use sidereon_core::terrain_store::{
    dted_tile_list_to_mmap_store, dted_tree_to_mmap_store, terrain_store_checksum64,
    DtedTileListEntry, Egm96FifteenMinuteGeoid, MmapTerrain, OrthometricHeightM, TerrainDatumError,
    TerrainGeoidModel, TerrainStoreError, TerrainTileId, VerticalDatum, TERRAIN_STORE_NULL_POSTING,
};

const MULTI_TILE_STORE_CHECKSUM64: u64 = 0xff51_4a67_6a94_d479;
const SRTM1_POSTINGS_PER_AXIS: usize = 3601;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dted")
        .join(name)
}

fn temp_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{}-{nonce}", std::process::id()))
}

fn f64_from_hex(input: &str) -> f64 {
    let trimmed = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
        .expect("hex string has 0x prefix");
    let bits = u64::from_str_radix(trimmed, 16).expect("valid f64 bits");
    f64::from_bits(bits)
}

#[derive(Clone, Copy)]
struct TerrainFixtureCase {
    longitude_deg: f64,
    latitude_deg: f64,
    nearest_m: f64,
    bilinear_m: f64,
}

impl TerrainFixtureCase {
    fn expected_m(self, interpolation: DtedInterpolation) -> f64 {
        match interpolation {
            DtedInterpolation::Bilinear => self.bilinear_m,
            DtedInterpolation::NearestPosting => self.nearest_m,
        }
    }
}

struct NamedTerrainFixtureCase {
    case_id: String,
    case: TerrainFixtureCase,
}

fn terrain_fixture_cases() -> Vec<NamedTerrainFixtureCase> {
    let json: Value =
        serde_json::from_slice(&fs::read(fixture_path("dted_points.json")).expect("fixture json"))
            .expect("parse fixture json");
    json["multi_tile_cases"]
        .as_array()
        .expect("multi tile cases")
        .iter()
        .map(|case| NamedTerrainFixtureCase {
            case_id: case["case_id"].as_str().expect("case_id").to_string(),
            case: TerrainFixtureCase {
                longitude_deg: f64_from_hex(
                    case["longitude_bits"].as_str().expect("longitude bits"),
                ),
                latitude_deg: f64_from_hex(case["latitude_bits"].as_str().expect("latitude bits")),
                nearest_m: f64_from_hex(case["nearest_bits"].as_str().expect("nearest bits")),
                bilinear_m: f64_from_hex(case["bilinear_bits"].as_str().expect("bilinear bits")),
            },
        })
        .collect()
}

fn multi_tile_points() -> Vec<(f64, f64)> {
    terrain_fixture_cases()
        .iter()
        .map(|named| (named.case.longitude_deg, named.case.latitude_deg))
        .collect()
}

fn skadi_excerpt_posts_m() -> Vec<Vec<i16>> {
    let json: Value = serde_json::from_slice(
        &fs::read(fixture_path("skadi_n36w107_5x5_posts.json")).expect("Skadi excerpt json"),
    )
    .expect("parse Skadi excerpt json");
    assert_eq!(json["schema"], "skadi-srtm1-post-excerpt-v1");
    json["posts_m"]
        .as_array()
        .expect("posts_m rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("posts_m columns")
                .iter()
                .map(|value| {
                    i16::try_from(value.as_i64().expect("Skadi post integer"))
                        .expect("Skadi post fits i16")
                })
                .collect()
        })
        .collect()
}

fn hgt_bytes_from_skadi_excerpt(posts_lat_lon: &[Vec<i16>]) -> Vec<u8> {
    let lat_count = posts_lat_lon.len();
    let lon_count = posts_lat_lon
        .first()
        .expect("at least one latitude row")
        .len();
    assert!(lat_count >= 2);
    assert!(lon_count >= 2);
    assert!(posts_lat_lon.iter().all(|row| row.len() == lon_count));

    let lat_step = (SRTM1_POSTINGS_PER_AXIS - 1) / (lat_count - 1);
    let lon_step = (SRTM1_POSTINGS_PER_AXIS - 1) / (lon_count - 1);
    let mut hgt = vec![0u8; SRTM1_POSTINGS_PER_AXIS * SRTM1_POSTINGS_PER_AXIS * 2];

    for (lat_index, row) in posts_lat_lon.iter().enumerate() {
        let hgt_row = SRTM1_POSTINGS_PER_AXIS - 1 - lat_index * lat_step;
        for (lon_index, value) in row.iter().enumerate() {
            let hgt_col = lon_index * lon_step;
            let offset = 2 * (hgt_row * SRTM1_POSTINGS_PER_AXIS + hgt_col);
            hgt[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
        }
    }

    hgt
}

fn assert_height_results_match(
    got: &[sidereon_core::Result<f64>],
    want: &[sidereon_core::Result<f64>],
    context: &str,
) {
    assert_eq!(got.len(), want.len(), "{context} result length");
    for (idx, (got, want)) in got.iter().zip(want).enumerate() {
        match (got, want) {
            (Ok(got), Ok(want)) => assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{context} index {idx} height bits"
            ),
            (Err(got), Err(want)) => assert_eq!(got, want, "{context} index {idx} error"),
            (got, want) => panic!("{context} index {idx} mismatch: {got:?} != {want:?}"),
        }
    }
}

fn naive_primary_fixture_height(
    longitude_deg: f64,
    latitude_deg: f64,
    interpolation: DtedInterpolation,
) -> f64 {
    fn posting(lon_index: usize, lat_index: usize) -> f64 {
        (-20 + 7 * lon_index as i32 - 5 * lat_index as i32 + (lon_index * lat_index) as i32) as f64
    }

    fn round_ties_even(value: f64) -> usize {
        let lo = value.floor();
        let fraction = value - lo;
        if fraction < 0.5 || (fraction == 0.5 && (lo as usize).is_multiple_of(2)) {
            lo as usize
        } else {
            lo as usize + 1
        }
    }

    let lon_index = (longitude_deg - -107.0) * 4.0;
    let lat_index = (latitude_deg - 36.0) * 4.0;
    if interpolation == DtedInterpolation::NearestPosting {
        return posting(round_ties_even(lon_index), round_ties_even(lat_index));
    }

    let lon_lo = lon_index.floor() as usize;
    let lat_lo = lat_index.floor() as usize;
    let fx = lon_index - lon_lo as f64;
    let fy = lat_index - lat_lo as f64;
    let mut height_m = 0.0;
    for (di, wx) in [(0usize, 1.0 - fx), (1usize, fx)] {
        for (dj, wy) in [(0usize, 1.0 - fy), (1usize, fy)] {
            let weight = wx * wy;
            if weight != 0.0 {
                height_m += weight * posting(lon_lo + di, lat_lo + dj);
            }
        }
    }
    height_m
}

#[test]
fn dyadic_fixture_lookups_are_bit_identical_to_naive_scaling() {
    let root = fixture_path("tiles");
    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let mut dted = DtedTerrain::new(&root);
    let offsets = [(1_u32, 17_u32), (33, 64), (65, 129), (127, 191), (200, 255)];

    for interpolation in [
        DtedInterpolation::Bilinear,
        DtedInterpolation::NearestPosting,
    ] {
        let mut options = DtedLookupOptions::default();
        options.interpolation = interpolation;
        for (lon_numerator, lat_numerator) in offsets {
            let longitude_deg = -107.0 + f64::from(lon_numerator) / 256.0;
            let latitude_deg = 36.0 + f64::from(lat_numerator) / 256.0;
            let naive = naive_primary_fixture_height(longitude_deg, latitude_deg, interpolation);
            let dted_height = dted
                .height_m_with_options(longitude_deg, latitude_deg, options)
                .expect("DTED dyadic fixture height");
            let mmap_height = mmap
                .height_m_with_options(longitude_deg, latitude_deg, options)
                .expect("mmap dyadic fixture height");

            assert_eq!(dted_height.to_bits(), naive.to_bits());
            assert_eq!(mmap_height.to_bits(), naive.to_bits());
        }
    }
}

#[test]
fn mmap_store_matches_dted_reader_over_multi_tile_fixture() {
    let root = fixture_path("tiles");
    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let mut dted = DtedTerrain::new(&root);
    let points = multi_tile_points();

    assert_eq!(mmap.vertical_datum(), VerticalDatum::Egm96MslOrthometric);
    assert_eq!(mmap.tile_index().len(), 2);
    assert_eq!(mmap.tile_count(), 2);
    assert_eq!(
        mmap.tile_ids(),
        &[TerrainTileId::new(36, -107), TerrainTileId::new(36, -106)]
    );
    for tile in mmap.tile_index() {
        assert_eq!(tile.vertical_datum, VerticalDatum::Egm96MslOrthometric);
        assert_eq!(tile.data_offset as usize % 4096, 0);
    }

    for interpolation in [
        DtedInterpolation::Bilinear,
        DtedInterpolation::NearestPosting,
    ] {
        let mut options = DtedLookupOptions::default();
        options.interpolation = interpolation;
        let got = mmap.height_batch(&points, options);
        let want = dted.height_batch(&points, options);
        assert_height_results_match(&got, &want, &format!("{interpolation:?} batch"));

        for &(longitude_deg, latitude_deg) in &points {
            let got = mmap
                .height_m_with_options(longitude_deg, latitude_deg, options)
                .expect("mmap scalar height");
            let want = DtedTerrain::new(&root)
                .height_m_with_options(longitude_deg, latitude_deg, options)
                .expect("DTED scalar height");
            assert_eq!(got.to_bits(), want.to_bits());

            let typed = mmap
                .orthometric_height_m_with_options(longitude_deg, latitude_deg, options)
                .expect("typed orthometric height");
            assert_eq!(typed.metres().to_bits(), want.to_bits());
        }
    }
}

#[test]
fn mmap_store_matches_committed_multi_tile_fixture_bits() {
    let root = fixture_path("tiles");
    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let cases = terrain_fixture_cases();
    let points = cases
        .iter()
        .map(|named| (named.case.longitude_deg, named.case.latitude_deg))
        .collect::<Vec<_>>();

    assert_eq!(mmap.vertical_datum(), VerticalDatum::Egm96MslOrthometric);
    assert_eq!(mmap.tile_index().len(), 2);
    assert_eq!(mmap.tile_count(), 2);
    assert_eq!(
        mmap.tile_ids(),
        &[TerrainTileId::new(36, -107), TerrainTileId::new(36, -106)]
    );
    for tile in mmap.tile_index() {
        assert_eq!(tile.vertical_datum, VerticalDatum::Egm96MslOrthometric);
        assert_eq!(tile.data_offset as usize % 4096, 0);
    }

    for interpolation in [
        DtedInterpolation::Bilinear,
        DtedInterpolation::NearestPosting,
    ] {
        let mut options = DtedLookupOptions::default();
        options.interpolation = interpolation;
        let got = mmap.height_batch(&points, options);
        assert_eq!(got.len(), cases.len(), "{interpolation:?} batch length");

        for (named, got) in cases.iter().zip(got) {
            let expected = named.case.expected_m(interpolation);
            let got = got.expect("mmap batch height");
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{} {interpolation:?} batch height bits",
                named.case_id
            );

            let scalar = mmap
                .height_m_with_options(named.case.longitude_deg, named.case.latitude_deg, options)
                .expect("mmap scalar height");
            assert_eq!(
                scalar.to_bits(),
                expected.to_bits(),
                "{} {interpolation:?} scalar height bits",
                named.case_id
            );

            let typed = mmap
                .orthometric_height_m_with_options(
                    named.case.longitude_deg,
                    named.case.latitude_deg,
                    options,
                )
                .expect("typed orthometric height");
            assert_eq!(
                typed.metres().to_bits(),
                expected.to_bits(),
                "{} {interpolation:?} typed height bits",
                named.case_id
            );
        }
    }
}

#[test]
fn mmap_store_nearest_posting_matches_real_skadi_source_posts() {
    let posts_m = skadi_excerpt_posts_m();
    let root = temp_path("terrain-store-skadi-excerpt");
    fs::create_dir_all(&root).expect("create temp DTED root");
    let hgt = hgt_bytes_from_skadi_excerpt(&posts_m);
    fs::write(
        root.join("n36_w107_1arc_v3.dt2"),
        hgt_to_dted(36, -107, &hgt).expect("convert Skadi excerpt HGT"),
    )
    .expect("write Skadi excerpt DTED tile");

    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let mut options = DtedLookupOptions::default();
    options.interpolation = DtedInterpolation::NearestPosting;

    for (lon_index, lat_index) in [(0usize, 0usize), (0, 4), (3, 0), (3, 4), (2, 3), (4, 4)] {
        let longitude_deg = -107.0 + lon_index as f64 / 4.0;
        let latitude_deg = 36.0 + lat_index as f64 / 4.0;
        let height_m = mmap
            .height_m_with_options(longitude_deg, latitude_deg, options)
            .expect("mmap source post height");
        let source_height_m = f64::from(posts_m[lat_index][lon_index]);

        assert_eq!(
            height_m.to_bits(),
            source_height_m.to_bits(),
            "Skadi source post lon_index={lon_index} lat_index={lat_index}"
        );
    }

    fs::remove_dir_all(root).expect("remove temp DTED root");
}

#[test]
fn hgt_void_posting_reads_as_unknown_elevation_on_every_path() {
    let hgt = fs::read(fixture_path("hgt/n36_w107_reference.hgt")).expect("read HGT fixture");
    let dt2 = hgt_to_dted(36, -107, &hgt).expect("convert HGT fixture");
    let root = temp_path("terrain-store-hgt-void");
    fs::create_dir_all(&root).expect("create temp DTED root");
    fs::write(root.join("n36_w107_1arc_v3.dt2"), dt2).expect("write converted DTED tile");

    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let mut dted = DtedTerrain::new(&root);
    let mut options = DtedLookupOptions::default();
    options.interpolation = DtedInterpolation::NearestPosting;
    let latitude_deg = 36.0 + 1234.0 / 3600.0;
    let longitude_deg = -107.0 + 2345.0 / 3600.0;
    let unknown = sidereon_core::Error::UnknownTerrainElevation {
        lat_index: 36,
        lon_index: -107,
        latitude_posting: 1234,
        longitude_posting: 2345,
    };

    assert_eq!(
        dted.height_m_with_options(longitude_deg, latitude_deg, options),
        Err(unknown.clone())
    );
    assert_eq!(
        dted.height_batch(&[(longitude_deg, latitude_deg)], options),
        vec![Err(unknown.clone())]
    );
    assert_eq!(
        mmap.height_m_with_options(longitude_deg, latitude_deg, options),
        Err(unknown.clone())
    );
    assert_eq!(
        mmap.orthometric_height_m_with_options(longitude_deg, latitude_deg, options),
        Err(unknown.clone())
    );
    assert_eq!(
        mmap.orthometric_height_batch(&[(longitude_deg, latitude_deg)], options),
        vec![Err(unknown.clone())]
    );
    assert_eq!(
        mmap.height_batch(&[(longitude_deg, latitude_deg)], options),
        vec![Err(unknown)]
    );

    fs::remove_dir_all(root).expect("remove temp DTED root");
}

#[test]
fn absent_mmap_tile_returns_typed_error_not_zero() {
    let root = fixture_path("tiles");
    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let missing_lon = -104.5;
    let missing_lat = 36.5;

    let err = mmap
        .height_m(missing_lon, missing_lat)
        .expect_err("missing tile must not return zero");
    assert_eq!(
        err,
        sidereon_core::Error::MissingTerrainTile {
            lat_index: 36,
            lon_index: -105
        }
    );

    let typed_err = mmap
        .orthometric_height_m(missing_lon, missing_lat)
        .expect_err("typed missing tile must not return zero");
    assert_eq!(typed_err, err);

    let batch = mmap.height_batch(&[(missing_lon, missing_lat)], DtedLookupOptions::default());
    assert_eq!(batch, vec![Err(err.clone())]);

    let typed_batch =
        mmap.orthometric_height_batch(&[(missing_lon, missing_lat)], DtedLookupOptions::default());
    assert_eq!(typed_batch, vec![Err(err)]);
}

#[test]
fn dted_tree_conversion_is_byte_stable() {
    let root = fixture_path("tiles");
    let first = dted_tree_to_mmap_store(&root).expect("first conversion");
    let second = dted_tree_to_mmap_store(&root).expect("second conversion");
    assert_eq!(first, second);
    assert_eq!(
        terrain_store_checksum64(&first),
        MULTI_TILE_STORE_CHECKSUM64
    );
    assert_eq!(
        terrain_store_checksum64(&first),
        terrain_store_checksum64(&second)
    );

    let parsed = MmapTerrain::from_bytes(&first).expect("parse store");
    let reserialized = parsed.to_bytes();
    assert_eq!(
        terrain_store_checksum64(&first),
        terrain_store_checksum64(&reserialized)
    );
    assert_eq!(first, reserialized);
}

#[test]
fn dted_tile_list_conversion_matches_directory_bytes() {
    let root = fixture_path("tiles");
    let directory_bytes = dted_tree_to_mmap_store(&root).expect("directory conversion");
    let entries = [
        DtedTileListEntry::from_indices(36, -107, root.join("n36_w107_1arc_v3.dt2")),
        DtedTileListEntry::from_indices(36, -106, root.join("n36_w106_1arc_v3.dt2")),
    ];
    let list_bytes = dted_tile_list_to_mmap_store(&entries).expect("list conversion");

    assert_eq!(list_bytes, directory_bytes);
    assert_eq!(
        terrain_store_checksum64(&list_bytes),
        MULTI_TILE_STORE_CHECKSUM64
    );
    assert_eq!(
        terrain_store_checksum64(&list_bytes),
        terrain_store_checksum64(&directory_bytes)
    );
}

#[test]
fn dted_tile_list_rejects_wrong_tile_id() {
    let root = fixture_path("tiles");
    let entries = [DtedTileListEntry::from_indices(
        35,
        -107,
        root.join("n36_w107_1arc_v3.dt2"),
    )];
    let err = dted_tile_list_to_mmap_store(&entries).expect_err("wrong id must fail");
    assert!(matches!(
        err,
        sidereon_core::terrain_store::TerrainStoreError::TileIdMismatch { .. }
    ));
}

#[cfg(unix)]
#[test]
fn dted_tree_conversion_follows_symlinked_files_and_directories() {
    use std::os::unix::fs::symlink;

    let root = fixture_path("tiles");
    let real = dted_tree_to_mmap_store(&root).expect("real tree conversion");

    let file_root = temp_path("terrain-store-symlinked-files");
    fs::create_dir_all(&file_root).expect("create symlink file root");
    for tile_name in ["n36_w107_1arc_v3.dt2", "n36_w106_1arc_v3.dt2"] {
        symlink(root.join(tile_name), file_root.join(tile_name)).expect("create tile symlink");
    }
    let file_linked = dted_tree_to_mmap_store(&file_root).expect("symlinked file conversion");
    assert_eq!(file_linked, real);
    assert_eq!(
        terrain_store_checksum64(&file_linked),
        MULTI_TILE_STORE_CHECKSUM64
    );

    let alias_root = temp_path("terrain-store-symlinked-alias-files");
    fs::create_dir_all(&alias_root).expect("create alias symlink file root");
    symlink(
        root.join("n36_w107_1arc_v3.dt2"),
        alias_root.join("west_alias"),
    )
    .expect("create west alias symlink");
    symlink(
        root.join("n36_w106_1arc_v3.dt2"),
        alias_root.join("east_alias"),
    )
    .expect("create east alias symlink");
    let alias_linked = dted_tree_to_mmap_store(&alias_root).expect("alias symlink conversion");
    assert_eq!(alias_linked, real);
    assert_eq!(
        terrain_store_checksum64(&alias_linked),
        MULTI_TILE_STORE_CHECKSUM64
    );

    let dir_root = temp_path("terrain-store-symlinked-dir");
    fs::create_dir_all(&dir_root).expect("create symlink directory root");
    symlink(&root, dir_root.join("linked_tiles")).expect("create directory symlink");
    let dir_linked = dted_tree_to_mmap_store(&dir_root).expect("symlinked directory conversion");
    assert_eq!(dir_linked, real);
    assert_eq!(
        terrain_store_checksum64(&dir_linked),
        MULTI_TILE_STORE_CHECKSUM64
    );

    fs::remove_dir_all(file_root).expect("remove symlink file root");
    fs::remove_dir_all(alias_root).expect("remove alias symlink file root");
    fs::remove_dir_all(dir_root).expect("remove symlink directory root");
}

#[test]
fn orthometric_to_ellipsoidal_uses_pinned_egm96_one_degree_grid() {
    let orthometric = OrthometricHeightM::new(123.5);
    let latitude_deg = 37.0;
    let longitude_deg = -122.0;
    let got = orthometric
        .to_ellipsoidal_height_deg(
            latitude_deg,
            longitude_deg,
            TerrainGeoidModel::Egm96OneDegree,
        )
        .expect("convert terrain height");
    let expected = orthometric.metres()
        + egm96_undulation(latitude_deg.to_radians(), longitude_deg.to_radians());
    assert_eq!(got.metres().to_bits(), expected.to_bits());
}

#[test]
fn missing_egm96_fifteen_minute_grid_returns_typed_error() {
    let root = temp_path("missing-egm96-dac");
    fs::create_dir_all(&root).expect("create temp dir");
    let missing_path = root.join("WW15MGH.DAC");
    let err = Egm96FifteenMinuteGeoid::from_ww15mgh_dac_path(&missing_path)
        .expect_err("missing DAC must error");
    match err {
        TerrainDatumError::MissingEgm96Dac { path, remediation } => {
            assert_eq!(path, missing_path);
            assert!(remediation.contains("WW15MGH.DAC"));
            assert!(remediation.contains("from_ww15mgh_dac_bytes"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
    fs::remove_dir_all(root).expect("remove temp dir");
}

#[test]
fn store_file_round_trips_through_path_reader() {
    let root = fixture_path("tiles");
    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");
    let store_path = temp_path("terrain-store-file").with_extension("bin");
    fs::write(&store_path, &bytes).expect("write terrain store");

    let mmap = MmapTerrain::from_path(&store_path).expect("read terrain store");
    assert_eq!(mmap.as_bytes(), bytes.as_slice());
    assert_eq!(mmap.to_bytes(), bytes);

    fs::remove_file(store_path).expect("remove temp store");
}

/// Posting of the committed n36_w107 fixture formula, used to fill stores
/// written byte by byte below.
fn primary_fixture_posting(lon_index: usize, lat_index: usize) -> i16 {
    (-20 + 7 * lon_index as i32 - 5 * lat_index as i32 + (lon_index * lat_index) as i32) as i16
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A one-tile TMMAP001 store for tile (36,-107) with 5x5 postings, written
/// field by field from the container layout rather than by the converter:
/// 64-byte header, one 80-byte index record at byte 64, zero padding to the
/// 4096-byte data offset, then the little-endian `i16` payload.
fn hand_built_one_tile_store() -> Vec<u8> {
    let mut payload = Vec::new();
    for lon_index in 0..5 {
        for lat_index in 0..5 {
            payload.extend_from_slice(&primary_fixture_posting(lon_index, lat_index).to_le_bytes());
        }
    }
    let mut store = vec![0u8; 4096];
    store[0..8].copy_from_slice(b"TMMAP001");
    store[8..10].copy_from_slice(&1u16.to_le_bytes());
    store[10] = 1;
    store[12..16].copy_from_slice(&1u32.to_le_bytes());
    store[16..24].copy_from_slice(&64u64.to_le_bytes());
    store[24..32].copy_from_slice(&4096u64.to_le_bytes());
    store[32..40].copy_from_slice(&(4096 + payload.len() as u64).to_le_bytes());
    let record = 64;
    store[record..record + 4].copy_from_slice(&36i32.to_le_bytes());
    store[record + 4..record + 8].copy_from_slice(&(-107i32).to_le_bytes());
    store[record + 8..record + 12].copy_from_slice(&5u32.to_le_bytes());
    store[record + 12..record + 16].copy_from_slice(&5u32.to_le_bytes());
    store[record + 16..record + 24].copy_from_slice(&4096u64.to_le_bytes());
    store[record + 24..record + 32].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    store[record + 32..record + 40].copy_from_slice(&fnv1a64(&payload).to_le_bytes());
    store[record + 40..record + 48].copy_from_slice(&36.0f64.to_le_bytes());
    store[record + 48..record + 56].copy_from_slice(&(-107.0f64).to_le_bytes());
    store[record + 56..record + 64].copy_from_slice(&37.0f64.to_le_bytes());
    store[record + 64..record + 72].copy_from_slice(&(-106.0f64).to_le_bytes());
    store[record + 72] = 1;
    store.extend_from_slice(&payload);
    store
}

#[test]
fn hand_built_store_parses_and_matches_the_converter() {
    let store = hand_built_one_tile_store();
    let mut mmap = MmapTerrain::from_bytes(&store).expect("parse hand-built store");
    let mut options = DtedLookupOptions::default();
    options.interpolation = DtedInterpolation::NearestPosting;
    // Posting (2, 2): -20 + 14 - 10 + 4.
    assert_eq!(mmap.height_m_with_options(-106.5, 36.5, options), Ok(-12.0));
    let converted = dted_tile_list_to_mmap_store(&[DtedTileListEntry::from_indices(
        36,
        -107,
        fixture_path("tiles").join("n36_w107_1arc_v3.dt2"),
    )])
    .expect("convert fixture tile");
    assert_eq!(converted, store);
}

#[test]
fn index_bounds_that_disagree_with_the_tile_id_are_refused_at_parse() {
    // Absolute bytes 112 and 128 are the first record's min_longitude_deg and
    // max_longitude_deg. Finite but absurd bounds kept the tile selected for
    // (-106.5, 36.5) and fed an offset near 1e300 to the cell arithmetic.
    // The payload checksum does not cover index metadata.
    let mut store = hand_built_one_tile_store();
    store[112..120].copy_from_slice(&(-1e300f64).to_le_bytes());
    store[128..136].copy_from_slice(&1e300f64.to_le_bytes());
    assert_eq!(
        MmapTerrain::from_bytes(&store).expect_err("absurd bounds must be refused"),
        TerrainStoreError::TileBoundsMismatch {
            lat_index: 36,
            lon_index: -107,
            field: "min_longitude_deg",
        }
    );

    let above = |value: f64| f64::from_bits(value.to_bits() + 1);
    let below = |value: f64| f64::from_bits(value.to_bits() - 1);
    for (offset, field, value) in [
        (104, "min_latitude_deg", above(36.0)),
        (112, "min_longitude_deg", below(-107.0)),
        (120, "max_latitude_deg", below(37.0)),
        (128, "max_longitude_deg", above(-106.0)),
        (120, "max_latitude_deg", 36.0),
        (128, "max_longitude_deg", 1e300),
    ] {
        let mut store = hand_built_one_tile_store();
        store[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert_eq!(
            MmapTerrain::from_bytes(&store).expect_err("bound off the tile edge"),
            TerrainStoreError::TileBoundsMismatch {
                lat_index: 36,
                lon_index: -107,
                field,
            },
            "{field} = {value:e}"
        );
    }

    for (lat_index, lon_index) in [(90, -107), (-91, -107), (36, 180), (36, -181)] {
        let mut store = hand_built_one_tile_store();
        store[64..68].copy_from_slice(&i32::to_le_bytes(lat_index));
        store[68..72].copy_from_slice(&i32::to_le_bytes(lon_index));
        assert_eq!(
            MmapTerrain::from_bytes(&store).expect_err("tile id outside the domain"),
            TerrainStoreError::TileIdOutOfRange {
                lat_index,
                lon_index,
            }
        );
    }
}

#[test]
fn store_wire_lengths_with_high_bits_are_refused_not_truncated() {
    // Adding 2^32 or 2^63 to a length leaves its low 32 bits unchanged; the
    // parser must see the whole value on every target.
    for (offset, high_bit) in [
        (32usize, 32u32),
        (32, 63),
        (16, 32),
        (24, 32),
        (88, 32),
        (80, 32),
    ] {
        let mut store = hand_built_one_tile_store();
        let value = u64::from_le_bytes(store[offset..offset + 8].try_into().expect("u64 field"));
        store[offset..offset + 8].copy_from_slice(&(value + (1u64 << high_bit)).to_le_bytes());
        assert!(
            matches!(
                MmapTerrain::from_bytes(&store),
                Err(TerrainStoreError::Parse { .. })
            ),
            "field at byte {offset} with bit {high_bit} set"
        );
    }
}

#[test]
fn dted_null_postings_are_carried_into_the_store_as_unknown_elevations() {
    // The committed 5x5 fixture with posting (lon 2, lat 3) set to the DTED
    // null bit pattern and that profile's byte-sum checksum recomputed.
    let mut tile = fs::read(fixture_path("tiles/n36_w107_1arc_v3.dt2")).expect("read fixture");
    let block_len = 12 + 2 * 5;
    let block = 3428 + 2 * block_len;
    tile[block + 8 + 2 * 3..block + 8 + 2 * 3 + 2].copy_from_slice(&[0xFF, 0xFF]);
    let sum = tile[block..block + block_len - 4]
        .iter()
        .fold(0i32, |acc, b| acc + i32::from(*b));
    tile[block + block_len - 4..block + block_len].copy_from_slice(&sum.to_be_bytes());

    let root = temp_path("terrain-store-null-posting");
    fs::create_dir_all(&root).expect("create temp DTED root");
    fs::write(root.join("n36_w107_1arc_v3.dt2"), &tile).expect("write null-posting tile");
    let bytes = dted_tree_to_mmap_store(&root).expect("convert DTED tree");

    // Payload offset 4096; posting (lon 2, lat 3) is element 2*5 + 3.
    let stored = 4096 + 2 * (2 * 5 + 3);
    assert_eq!(
        i16::from_le_bytes([bytes[stored], bytes[stored + 1]]),
        TERRAIN_STORE_NULL_POSTING
    );

    let mut mmap = MmapTerrain::from_bytes(&bytes).expect("parse terrain store");
    let mut dted = DtedTerrain::new(&root);
    let unknown = sidereon_core::Error::UnknownTerrainElevation {
        lat_index: 36,
        lon_index: -107,
        latitude_posting: 3,
        longitude_posting: 2,
    };
    let mut nearest = DtedLookupOptions::default();
    nearest.interpolation = DtedInterpolation::NearestPosting;
    let bilinear = DtedLookupOptions::default();
    for (longitude_deg, latitude_deg, options, want) in [
        (-106.5, 36.75, nearest, Err(unknown.clone())),
        (-106.5, 36.75, bilinear, Err(unknown.clone())),
        (-106.375, 36.625, bilinear, Err(unknown.clone())),
        (-106.625, 36.875, bilinear, Err(unknown.clone())),
        (-106.25, 36.75, bilinear, Ok(-5.0)),
        (-106.25, 36.75, nearest, Ok(-5.0)),
    ] {
        let got = mmap.height_m_with_options(longitude_deg, latitude_deg, options);
        assert_eq!(got, want, "mapped ({longitude_deg}, {latitude_deg})");
        assert_eq!(
            dted.height_m_with_options(longitude_deg, latitude_deg, options),
            want,
            "raw ({longitude_deg}, {latitude_deg})"
        );
        assert_eq!(
            mmap.orthometric_height_batch(&[(longitude_deg, latitude_deg)], options),
            vec![want.clone().map(OrthometricHeightM::new)]
        );
    }
    assert_eq!(
        mmap.ellipsoidal_height_m(-106.375, 36.625),
        Err(TerrainDatumError::Terrain(unknown))
    );

    let reserialized = MmapTerrain::from_bytes(&bytes)
        .expect("parse terrain store")
        .to_bytes();
    assert_eq!(reserialized, bytes);

    fs::remove_dir_all(root).expect("remove temp DTED root");
}

/// A copy of both committed fixture tiles in which the eastern tile's western
/// edge posting at 36.5 N (profile 0, posting 2) is the DTED null.
fn fixture_pair_with_null_edge_posting(name: &str, with_west: bool) -> PathBuf {
    let root = temp_path(name);
    fs::create_dir_all(&root).expect("create temp DTED root");
    let mut east = fs::read(fixture_path("tiles/n36_w106_1arc_v3.dt2")).expect("read east tile");
    let block_len = 12 + 2 * 5;
    let block = 3428;
    east[block + 8 + 2 * 2..block + 8 + 2 * 2 + 2].copy_from_slice(&[0xFF, 0xFF]);
    let sum = east[block..block + block_len - 4]
        .iter()
        .fold(0i32, |acc, b| acc + i32::from(*b));
    east[block + block_len - 4..block + block_len].copy_from_slice(&sum.to_be_bytes());
    fs::write(root.join("n36_w106_1arc_v3.dt2"), east).expect("write east tile");
    if with_west {
        fs::copy(
            fixture_path("tiles/n36_w107_1arc_v3.dt2"),
            root.join("n36_w107_1arc_v3.dt2"),
        )
        .expect("copy west tile");
    }
    root
}

#[test]
fn a_null_edge_posting_defers_to_the_neighbouring_tile_on_that_edge() {
    let mut nearest = DtedLookupOptions::default();
    nearest.interpolation = DtedInterpolation::NearestPosting;
    let bilinear = DtedLookupOptions::default();
    let unknown = Err(sidereon_core::Error::UnknownTerrainElevation {
        lat_index: 36,
        lon_index: -106,
        latitude_posting: 2,
        longitude_posting: 0,
    });
    // The eastern tile (36,-106) is the first candidate on -106 and the only
    // one just east of it. Its edge posting (0, 2) at (-106, 36.5) is null;
    // the western tile's posting (4, 2) at exactly that point is
    // -20 + 7*4 - 5*2 + 4*2 = 6, and (4, 3) is 5.
    let just_east = -106.0 + 1e-9;
    let cases = [
        ((-106.0, 36.5), nearest, Ok(6.0)),
        // Its nearest posting is the same null edge posting.
        ((just_east, 36.5), nearest, Ok(6.0)),
        ((-106.0, 36.5), bilinear, Ok(6.0)),
        ((-106.0, 36.625), bilinear, Ok(5.5)),
        // Inside the eastern tile the null has nonzero bilinear weight.
        ((just_east, 36.5), bilinear, unknown.clone()),
    ];

    let root = fixture_pair_with_null_edge_posting("terrain-edge-null-pair", true);
    let store = dted_tree_to_mmap_store(&root).expect("convert pair");
    let mmap = MmapTerrain::from_bytes(&store).expect("parse pair store");
    for ((longitude_deg, latitude_deg), options, want) in cases.clone() {
        assert_eq!(
            DtedTerrain::new(&root).height_m_with_options(longitude_deg, latitude_deg, options),
            want,
            "raw ({longitude_deg}, {latitude_deg}) {options:?}"
        );
        assert_eq!(
            mmap.orthometric_height_m_with_options(longitude_deg, latitude_deg, options)
                .map(OrthometricHeightM::metres),
            want,
            "mapped ({longitude_deg}, {latitude_deg}) {options:?}"
        );
    }
    // Batches answer the same, including after a query that made the
    // eastern tile current.
    for (options, points, tail) in [
        (
            bilinear,
            [(-105.5, 36.5), (-106.0, 36.5), (-106.0, 36.625)],
            [Ok(6.0), Ok(5.5)],
        ),
        (
            nearest,
            [(-105.5, 36.5), (-106.0, 36.5), (just_east, 36.5)],
            [Ok(6.0), Ok(6.0)],
        ),
    ] {
        let first = DtedTerrain::new(&root).height_m_with_options(-105.5, 36.5, options);
        assert!(first.is_ok());
        let want = vec![first, tail[0].clone(), tail[1].clone()];
        assert_eq!(
            DtedTerrain::new(&root).height_batch(&points, options),
            want,
            "raw batch {options:?}"
        );
        assert_eq!(
            MmapTerrain::from_bytes(&store)
                .expect("parse pair store")
                .height_batch(&points, options),
            want,
            "mapped batch {options:?}"
        );
    }
    fs::remove_dir_all(root).expect("remove temp DTED root");

    // Without the neighbour the null stays unknown.
    let root = fixture_pair_with_null_edge_posting("terrain-edge-null-alone", false);
    let store = dted_tree_to_mmap_store(&root).expect("convert east tile");
    let mmap = MmapTerrain::from_bytes(&store).expect("parse east store");
    for ((longitude_deg, latitude_deg), options, _) in cases.clone() {
        assert_eq!(
            DtedTerrain::new(&root).height_m_with_options(longitude_deg, latitude_deg, options),
            unknown
        );
        assert_eq!(
            mmap.orthometric_height_m_with_options(longitude_deg, latitude_deg, options)
                .map(OrthometricHeightM::metres),
            unknown
        );
    }
    fs::remove_dir_all(root).expect("remove temp DTED root");

    // A neighbour consulted only because the first tile gave an unknown
    // elevation leaves that unknown standing when it cannot be read or states
    // another datum.
    let mut wgs72_west = fs::read(fixture_path("tiles/n36_w107_1arc_v3.dt2")).expect("read west");
    wgs72_west[224..229].copy_from_slice(b"WGS72");
    for (label, west) in [
        ("corrupt", b"not a DTED tile".to_vec()),
        ("wgs72", wgs72_west),
    ] {
        let root = fixture_pair_with_null_edge_posting(&format!("terrain-edge-{label}"), false);
        fs::write(root.join("n36_w107_1arc_v3.dt2"), west).expect("write west tile");
        for ((longitude_deg, latitude_deg), options, _) in cases.clone() {
            assert_eq!(
                DtedTerrain::new(&root).height_m_with_options(longitude_deg, latitude_deg, options),
                unknown,
                "{label} ({longitude_deg}, {latitude_deg}) {options:?}"
            );
        }
        fs::remove_dir_all(root).expect("remove temp DTED root");
    }
}

#[test]
fn store_conversion_refuses_a_tile_on_another_horizontal_datum() {
    let root = temp_path("terrain-store-wgs72");
    fs::create_dir_all(&root).expect("create temp DTED root");
    let mut tile = fs::read(fixture_path("tiles/n36_w107_1arc_v3.dt2")).expect("read fixture");
    // DSI character 145, after the 80-byte UHL.
    tile[224..229].copy_from_slice(b"WGS72");
    let path = root.join("n36_w107_1arc_v3.dt2");
    fs::write(&path, tile).expect("write WGS72 tile");

    assert_eq!(
        dted_tree_to_mmap_store(&root).expect_err("WGS72 tile must not be stored as WGS84"),
        TerrainStoreError::NonWgs84Tile {
            path: path.clone(),
            datum: DtedHorizontalDatum::Wgs72,
        }
    );
    assert_eq!(
        dted_tile_list_to_mmap_store(&[DtedTileListEntry::from_indices(36, -107, &path)])
            .expect_err("WGS72 tile must not be stored as WGS84"),
        TerrainStoreError::NonWgs84Tile {
            path,
            datum: DtedHorizontalDatum::Wgs72,
        }
    );
    fs::remove_dir_all(root).expect("remove temp DTED root");
}

#[test]
fn a_corner_query_tries_every_candidate_after_an_unreadable_neighbour() {
    // Candidates at the corner (-106, 37), in order: (37,-106), (36,-106),
    // (37,-107), (36,-107). The first holds a null at its corner posting, the
    // second is corrupt, and the third holds the height: its posting (4, 0)
    // is -20 + 7*4 = 8.
    let root = temp_path("terrain-corner-candidates");
    fs::create_dir_all(&root).expect("create temp DTED root");
    let north_latitude = |mut tile: Vec<u8>| {
        tile[12..20].copy_from_slice(b"0370000N");
        tile
    };
    let mut first =
        north_latitude(fs::read(fixture_path("tiles/n36_w106_1arc_v3.dt2")).expect("read tile"));
    let block_len = 12 + 2 * 5;
    let block = 3428;
    first[block + 8..block + 10].copy_from_slice(&[0xFF, 0xFF]);
    let sum = first[block..block + block_len - 4]
        .iter()
        .fold(0i32, |acc, b| acc + i32::from(*b));
    first[block + block_len - 4..block + block_len].copy_from_slice(&sum.to_be_bytes());
    fs::write(root.join("n37_w106_1arc_v3.dt2"), first).expect("write first tile");
    fs::write(root.join("n36_w106_1arc_v3.dt2"), b"not a DTED tile").expect("write corrupt tile");
    let third =
        north_latitude(fs::read(fixture_path("tiles/n36_w107_1arc_v3.dt2")).expect("read tile"));
    fs::write(root.join("n37_w107_1arc_v3.dt2"), third).expect("write third tile");

    let mut nearest = DtedLookupOptions::default();
    nearest.interpolation = DtedInterpolation::NearestPosting;
    for options in [DtedLookupOptions::default(), nearest] {
        assert_eq!(
            DtedTerrain::new(&root).height_m_with_options(-106.0, 37.0, options),
            Ok(8.0),
            "{options:?}"
        );
        assert_eq!(
            DtedTerrain::new(&root).height_batch(&[(-105.5, 37.5), (-106.0, 37.0)], options)[1],
            Ok(8.0),
            "batch {options:?}"
        );
    }

    // Without the third tile the fourth is absent too, and the first tile's
    // unknown elevation stands rather than the corrupt tile's error.
    fs::remove_file(root.join("n37_w107_1arc_v3.dt2")).expect("remove third tile");
    assert_eq!(
        DtedTerrain::new(&root).height_m(-106.0, 37.0),
        Err(sidereon_core::Error::UnknownTerrainElevation {
            lat_index: 37,
            lon_index: -106,
            latitude_posting: 0,
            longitude_posting: 0,
        })
    );
    fs::remove_dir_all(root).expect("remove temp DTED root");
}
