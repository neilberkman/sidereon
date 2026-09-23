//! Format-faithful numeric formatting helpers.

/// Fixed-decimal formatting copied from the TLE encoder.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn fixed_decimals(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

/// Shortest round-tripping decimal copied from the OMM/CDM encoders.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn fmt_num(value: f64) -> String {
    format!("{value}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_decimals_matches_requested_precision() {
        assert_eq!(fixed_decimals(1.23456, 2), "1.23");
    }

    #[test]
    fn fmt_num_round_trips_value() {
        let value = 1.0 / 3.0;
        let encoded = fmt_num(value);
        let decoded = encoded.parse::<f64>().unwrap();
        assert_eq!(decoded.to_bits(), value.to_bits());
    }
}
