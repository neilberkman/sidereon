#![no_main]

use libfuzzer_sys::fuzz_target;

fn labeled_header_line(prefix: &str, label: &str) -> String {
    format!("{prefix:<60}{label}")
}

fn v3_obs_types_line() -> String {
    let mut bytes = vec![b' '; 80];
    bytes[0] = b'G';
    bytes[5] = b'1';
    bytes[7..10].copy_from_slice(b"C1C");
    bytes[60..79].copy_from_slice(b"SYS / # / OBS TYPES");
    String::from_utf8(bytes).expect("static ASCII header line")
}

fn v3_header() -> String {
    [
        labeled_header_line("3.0", "CRINEX VERS   / TYPE"),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     3.04           OBSERVATION DATA    G                   ",
            "RINEX VERSION / TYPE",
        ),
        v3_obs_types_line(),
        labeled_header_line("", "END OF HEADER"),
    ]
    .join("\n")
}

fuzz_target!(|data: &[u8]| {
    let body = String::from_utf8_lossy(data);
    let input = format!("{}\n{}", v3_header(), body);
    let _ = sidereon_core::rinex::decode_crinex(&input);
});
