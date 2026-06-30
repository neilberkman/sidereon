//! IONEX serialization - the inverse of the grid parser ([`Ionex::parse`]).
//!
//! Pure and deterministic: the same [`Ionex`] always produces byte-identical
//! text, and no I/O is performed. A parse -> encode -> parse pipeline round-trips
//! the canonical IR (node axes, geometry, exponent, map epochs, and every TEC /
//! RMS value), so re-reading the output yields an equal product.
//!
//! The grid is reconstructed from the canonical IR, not echoed from the source
//! bytes: the latitude/longitude axis bounds come from the node arrays, the
//! scaled-integer TEC field is recovered as `round(value / 10^EXPONENT)`, and the
//! map epoch is rendered back to the IONEX civil `year month day hour minute
//! second` record. Records the reader does not consume (the auxiliary block, the
//! free header descriptors) are not emitted; the output is the minimal IONEX that
//! re-parses to the same product.

use core::fmt::Write as _;

use super::grid::Ionex;
use super::j2000_seconds_from_instant;
use crate::astro::time::civil::civil_from_j2000_seconds;
use crate::astro::time::model::Instant;

/// TEC/RMS scaled-integer fields per data line (IONEX standard layout).
const VALUES_PER_LINE: usize = 16;
/// First byte column of the 20-character record-label field.
const LABEL_COLUMN: usize = 60;

impl Ionex {
    /// Serialize this product to standard IONEX text.
    ///
    /// Pure and deterministic. See this module's docs for the round-trip
    /// guarantee: re-parsing the result yields an equal [`Ionex`].
    pub fn to_ionex_string(&self) -> String {
        let mut out = String::new();
        self.write_header(&mut out);

        let scale = 10f64.powi(self.exponent());
        for (index, epoch) in self.map_epochs().iter().enumerate() {
            let map_number = index + 1;
            write_labeled(&mut out, &format!("{map_number:6}"), "START OF TEC MAP");
            write_epoch(&mut out, *epoch);
            self.write_map(&mut out, &self.tec_maps()[index], scale);
            write_labeled(&mut out, &format!("{map_number:6}"), "END OF TEC MAP");
        }
        for (index, grid) in self.rms_maps().iter().enumerate() {
            let map_number = index + 1;
            write_labeled(&mut out, &format!("{map_number:6}"), "START OF RMS MAP");
            if let Some(epoch) = self.map_epochs().get(index) {
                write_epoch(&mut out, *epoch);
            }
            self.write_map(&mut out, grid, scale);
            write_labeled(&mut out, &format!("{map_number:6}"), "END OF RMS MAP");
        }
        out
    }

    fn write_header(&self, out: &mut String) {
        let lat1 = self.lat_nodes_deg().first().copied().unwrap_or(0.0);
        let lat2 = self.lat_nodes_deg().last().copied().unwrap_or(0.0);
        let lon1 = self.lon_nodes_deg().first().copied().unwrap_or(0.0);
        let lon2 = self.lon_nodes_deg().last().copied().unwrap_or(0.0);

        write_labeled(out, "     1.0            I", "IONEX VERSION / TYPE");
        write_labeled(
            out,
            &format!("{lat1:8.1}{lat2:8.1}{:8.1}", self.dlat_deg()),
            "LAT1 / LAT2 / DLAT",
        );
        write_labeled(
            out,
            &format!("{lon1:8.1}{lon2:8.1}{:8.1}", self.dlon_deg()),
            "LON1 / LON2 / DLON",
        );
        let height = self.shell_height_km();
        write_labeled(
            out,
            &format!("{height:8.1}{height:8.1}{:8.1}", 0.0),
            "HGT1 / HGT2 / DHGT",
        );
        write_labeled(
            out,
            &format!("{:8.1}", self.base_radius_km()),
            "BASE RADIUS",
        );
        write_labeled(out, &format!("{:6}", self.exponent()), "EXPONENT");
        write_labeled(out, "", "END OF HEADER");
    }

    /// Emit one map's latitude bands. Each band is a `LAT/LON1/LON2/DLON/H`
    /// record (the reader keys only on the label) followed by the band's scaled
    /// integer fields, the inverse of the parser's band accumulation.
    fn write_map(&self, out: &mut String, grid: &[Vec<f64>], scale: f64) {
        let lon1 = self.lon_nodes_deg().first().copied().unwrap_or(0.0);
        let lon2 = self.lon_nodes_deg().last().copied().unwrap_or(0.0);
        let height = self.shell_height_km();
        for (lat_index, band) in grid.iter().enumerate() {
            let lat = self.lat_nodes_deg().get(lat_index).copied().unwrap_or(0.0);
            write_labeled(
                out,
                &format!(
                    "{lat:8.1}{lon1:8.1}{lon2:8.1}{:8.1}{height:8.1}",
                    self.dlon_deg()
                ),
                "LAT/LON1/LON2/DLON/H",
            );
            for chunk in band.chunks(VALUES_PER_LINE) {
                for value in chunk {
                    let scaled = (value / scale).round() as i64;
                    // The reader splits TEC/RMS fields on whitespace, so two
                    // adjacent values that each fill the I5 column would merge
                    // into one token. Emit a guaranteed leading space: values
                    // within I5 keep their familiar right-justified five-column
                    // form (byte-identical to a plain `{:5}` for any value up to
                    // four characters), and wider values stay separated.
                    let _ = write!(out, " {scaled:4}");
                }
                out.push('\n');
            }
        }
    }
}

/// Write `data` left-justified into the 0..60 field, the record label at column
/// 60, and a newline - the column layout the parser's `label_of` / `data_of`
/// read back.
fn write_labeled(out: &mut String, data: &str, label: &str) {
    let _ = writeln!(out, "{data:<LABEL_COLUMN$}{label}");
}

/// Emit an `EPOCH OF CURRENT MAP` record from a map epoch, the inverse of the
/// parser's `parse_epoch_j2000_s`.
fn write_epoch(out: &mut String, epoch: Instant) {
    let seconds =
        j2000_seconds_from_instant(epoch).expect("IONEX map epoch is convertible to J2000 seconds");
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(seconds);
    write_labeled(
        out,
        &format!("{year:6}{month:6}{day:6}{hour:6}{minute:6}{second:6}"),
        "EPOCH OF CURRENT MAP",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ionex::ionex_epoch_from_j2000_seconds;

    #[test]
    fn civil_from_j2000_seconds_inverts_the_parser_epoch() {
        // 2020-06-25 00:00:00 UTC is exactly 646_315_200 J2000 seconds (the value
        // pinned by the parser's `parse_epoch_accepts_valid_civil_datetime` test).
        assert_eq!(
            civil_from_j2000_seconds(646_315_200),
            (2020, 6, 25, 0, 0, 0)
        );
    }

    #[test]
    fn civil_from_j2000_seconds_round_trips_through_the_instant_epoch() {
        // Cover a time-of-day, a day rollover, and a pre-J2000 (negative) second
        // count: each must reproduce the civil parts the parser would have read.
        for seconds in [
            646_315_200_i64,
            646_315_200 + 7_323,  // +02:02:03
            646_315_200 + 86_400, // next day, midnight
            -43_200,              // J2000 origin: 2000-01-01 00:00:00
            -43_200 - 1,          // one second earlier: 1999-12-31 23:59:59
        ] {
            let epoch = ionex_epoch_from_j2000_seconds(seconds);
            let recovered = j2000_seconds_from_instant(epoch).expect("J2000 seconds");
            assert_eq!(recovered, seconds, "instant epoch round-trips its seconds");

            let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(seconds);
            assert!((1..=12).contains(&month), "month in range: {month}");
            assert!((1..=31).contains(&day), "day in range: {day}");
            assert!((0..24).contains(&hour), "hour in range: {hour}");
            assert!((0..60).contains(&minute), "minute in range: {minute}");
            assert!((0..60).contains(&second), "second in range: {second}");
            assert!(year >= 1999, "year plausible: {year}");
        }
    }
}
