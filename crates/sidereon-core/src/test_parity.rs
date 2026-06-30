pub(crate) fn f64_from_hex(input: &str) -> Result<f64, std::num::ParseIntError> {
    let hex = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
        .unwrap_or(input);
    u64::from_str_radix(hex, 16).map(f64::from_bits)
}
