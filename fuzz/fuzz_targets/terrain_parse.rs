#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::terrain::DtedTile;

fuzz_target!(|data: &[u8]| {
    let mut path = std::env::temp_dir();
    path.push(format!("sidereon-dted-fuzz-{}.dt2", std::process::id()));
    let _ = std::fs::write(&path, data);
    let _ = DtedTile::from_path(&path);
    let _ = std::fs::remove_file(path);
});
