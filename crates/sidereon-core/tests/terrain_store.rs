//! Terrain store fixture provenance: DTED tiles under
//! `tests/fixtures/dted/tiles` and points in `tests/fixtures/dted/dted_points.json`
//! are existing repository fixtures generated from the public DTED
//! UHL/DSI/ACC/data-record layout. The HGT void test uses the synthetic
//! `tests/fixtures/dted/hgt/n36_w107_reference.hgt` fixture already committed
//! for the SRTM1-to-DTED converter. Legacy store regression cases compare
//! terrain heights by `f64::to_bits()` against committed fixture values, and
//! source-post checks use the public Skadi SRTM1 excerpt in
//! `skadi_n36w107_5x5_posts.json`.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use sidereon_core::data::hgt_to_dted;
use sidereon_core::geoid::egm96_undulation;
use sidereon_core::terrain::{DtedInterpolation, DtedLookupOptions, DtedTerrain};
use sidereon_core::terrain_store::{
    dted_tile_list_to_mmap_store, dted_tree_to_mmap_store, terrain_store_checksum64,
    DtedTileListEntry, Egm96FifteenMinuteGeoid, MmapTerrain, OrthometricHeightM, TerrainDatumError,
    TerrainGeoidModel, TerrainTileId, VerticalDatum,
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
fn mmap_store_returns_typed_zero_for_hgt_void_posting() {
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

    let got = mmap
        .height_m_with_options(longitude_deg, latitude_deg, options)
        .expect("mmap void height");
    let want = dted
        .height_m_with_options(longitude_deg, latitude_deg, options)
        .expect("DTED void height");
    let typed = mmap
        .orthometric_height_m_with_options(longitude_deg, latitude_deg, options)
        .expect("typed orthometric void height");
    let typed_batch = mmap.orthometric_height_batch(&[(longitude_deg, latitude_deg)], options);

    assert_eq!(got.to_bits(), want.to_bits());
    assert_eq!(typed.metres().to_bits(), want.to_bits());
    assert_eq!(got.to_bits(), 0.0f64.to_bits());
    assert_eq!(
        typed_batch[0]
            .as_ref()
            .map(|height| height.metres().to_bits()),
        Ok(0.0f64.to_bits())
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
