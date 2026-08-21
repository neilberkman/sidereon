use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use sidereon_core::data::{dted_cache_relpath, hgt_to_dted, HgtConversionError};
use sidereon_core::terrain::{DtedInterpolation, DtedLookupOptions, DtedTerrain};

const POSTINGS: usize = 3601;
const HGT_LEN: usize = POSTINGS * POSTINGS * 2;
const DTED_LEN: usize = 25_981_042;
const LAT_INDEX: i32 = 36;
const LON_INDEX: i32 = -107;
const REFERENCE_DT2_SHA256: &str =
    "1aef121ba4cadf1180efb74eabf6118d2df7b290957739ef99abf50d9a0f8304";

// Fixture provenance: the HGT payload in these tests is generated in memory
// from the closed-form `synthetic_hgt_sample` function below. No external
// terrain payload is copied. Selected postings pin positive, negative, minimum
// negative, endpoint, and void cases through the public SRTM1 conversion path.
// The committed `tests/fixtures/dted/hgt/n36_w107_reference.hgt` fixture is a
// synthetic SRTM1-sized HGT tile: all postings are zero except six selected
// big-endian i16 samples (1000, 8848, -321, -32768, -100, 42). The mostly-zero
// content keeps the source-control payload compressible while still pinning the
// exact bytes produced by the public conversion path.

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dted")
        .join(name)
}

fn temp_root(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{}-{nonce}", std::process::id()))
}

fn synthetic_hgt_sample(row: usize, col: usize) -> i16 {
    match (row, col) {
        (2366, 2345) => i16::MIN,
        (3500, 200) => -415,
        (1600, 3000) => -1,
        (0, 3600) => 8848,
        _ => (((row as i32 * 37 + col as i32 * 19) % 5000) - 1000) as i16,
    }
}

fn expected_posting(lat_posting: usize, lon_posting: usize) -> i16 {
    let sample = synthetic_hgt_sample(POSTINGS - 1 - lat_posting, lon_posting);
    if sample == i16::MIN {
        0
    } else {
        sample
    }
}

fn generated_hgt() -> Vec<u8> {
    let mut hgt = vec![0u8; HGT_LEN];
    for row in 0..POSTINGS {
        for col in 0..POSTINGS {
            let start = 2 * (row * POSTINGS + col);
            hgt[start..start + 2].copy_from_slice(&synthetic_hgt_sample(row, col).to_be_bytes());
        }
    }
    hgt
}

#[test]
fn hgt_to_dted_round_trips_selected_postings_through_reader() {
    let hgt = generated_hgt();
    let dt2 = hgt_to_dted(LAT_INDEX, LON_INDEX, &hgt).expect("convert HGT to DTED");
    assert_eq!(dt2.len(), DTED_LEN);

    let dt2_again = hgt_to_dted(LAT_INDEX, LON_INDEX, &hgt).expect("convert HGT to DTED again");
    assert!(
        dt2 == dt2_again,
        "same HGT input and tile indices must produce identical DTED bytes"
    );

    let root = temp_root("hgt-to-dted-roundtrip");
    let relpath = dted_cache_relpath(LAT_INDEX, LON_INDEX).expect("DTED cache path");
    let tile_path = root.join(relpath);
    fs::create_dir_all(tile_path.parent().expect("tile parent")).expect("create DTED block dir");
    fs::write(&tile_path, &dt2).expect("write converted DTED tile");

    let mut terrain = DtedTerrain::new(&root);
    let nearest = DtedLookupOptions {
        interpolation: DtedInterpolation::NearestPosting,
    };

    for (lat_posting, lon_posting) in [(0, 0), (100, 200), (1234, 2345), (2000, 3000), (3600, 3600)]
    {
        let lat = f64::from(LAT_INDEX) + lat_posting as f64 / 3600.0;
        let lon = f64::from(LON_INDEX) + lon_posting as f64 / 3600.0;
        let got = terrain
            .height_m_with_options(lon, lat, nearest)
            .expect("read converted DTED posting");
        let expected = f64::from(expected_posting(lat_posting, lon_posting));
        assert_eq!(
            got, expected,
            "posting lat_index={lat_posting} lon_index={lon_posting}"
        );
    }

    fs::remove_dir_all(root).expect("remove temp DTED root");
}

#[test]
fn hgt_to_dted_matches_committed_byte_stability_digest() {
    let hgt = fs::read(fixture_path("hgt/n36_w107_reference.hgt"))
        .expect("read committed HGT reference fixture");
    let dt2 = hgt_to_dted(LAT_INDEX, LON_INDEX, &hgt).expect("convert reference HGT to DTED");

    assert_eq!(dt2.len(), DTED_LEN);
    assert_eq!(sha256_hex(&dt2), REFERENCE_DT2_SHA256);
}

#[test]
fn hgt_to_dted_rejects_bad_length_and_invalid_tile_index() {
    assert_eq!(
        hgt_to_dted(LAT_INDEX, LON_INDEX, &[]),
        Err(HgtConversionError::BadLength {
            expected: HGT_LEN,
            got: 0
        })
    );
    assert_eq!(
        hgt_to_dted(90, LON_INDEX, &[]),
        Err(HgtConversionError::InvalidTileIndex {
            lat_index: 90,
            lon_index: LON_INDEX
        })
    );
}

#[test]
fn missing_dted_tile_reads_as_sea_level() {
    let root = temp_root("dted-empty-root");
    let mut terrain = DtedTerrain::new(&root);
    assert_eq!(
        terrain
            .height_m(f64::from(LON_INDEX) + 0.5, f64::from(LAT_INDEX) + 0.5)
            .expect("missing tile uses terrain sea-level fallback"),
        0.0
    );
}

#[test]
fn local_sha256_helper_matches_reference_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

fn sha256_hex(data: &[u8]) -> String {
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];

    let (blocks, remainder) = data.as_chunks::<64>();
    for chunk in blocks {
        compress_sha256_block(chunk, &mut state);
    }

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut final_blocks = [[0u8; 64]; 2];
    final_blocks[0][..remainder.len()].copy_from_slice(remainder);
    final_blocks[0][remainder.len()] = 0x80;

    let block_count = if remainder.len() > 55 {
        final_blocks[1][56..64].copy_from_slice(&bit_len.to_be_bytes());
        2
    } else {
        final_blocks[0][56..64].copy_from_slice(&bit_len.to_be_bytes());
        1
    };

    for block in final_blocks.iter().take(block_count) {
        compress_sha256_block(block, &mut state);
    }

    use std::fmt::Write as _;

    let mut hex = String::with_capacity(64);
    for word in state {
        write!(&mut hex, "{word:08x}").expect("write SHA-256 hex");
    }
    hex
}

fn compress_sha256_block(block: &[u8], state: &mut [u32; 8]) {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut w = [0u32; 64];
    for (word, bytes) in w.iter_mut().take(16).zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }

    let mut i = 16usize;
    while i < 64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
        i += 1;
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for (&round_k, &round_w) in K.iter().zip(&w) {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(round_k)
            .wrapping_add(round_w);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}
