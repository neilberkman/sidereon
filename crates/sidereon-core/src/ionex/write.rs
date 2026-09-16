//! IONEX serialization - the inverse of the grid parser ([`Ionex::parse`]).
//!
//! Pure and deterministic: the same [`Ionex`] always produces byte-identical
//! text, and no I/O is performed. A parse -> encode -> parse pipeline round-trips
//! the canonical IR (node axes, geometry, exponent, header records, map epochs,
//! and every TEC, RMS and height value), so re-reading the output yields an
//! equal product.
//!
//! The grid is reconstructed from the canonical IR, not echoed from the source
//! bytes: the latitude/longitude axis bounds come from the node arrays, the
//! scaled-integer field is recovered as `round(value / 10^EXPONENT)` with `9999`
//! for a non-available value, and the map epochs are rendered back to the IONEX
//! civil `year month day hour minute second` record. The records a product's
//! maps determine, `EPOCH OF FIRST MAP`, `EPOCH OF LAST MAP`,
//! `# OF MAPS IN FILE` and `MAP DIMENSION`, are written from the maps. Records
//! the reader does not keep, such as an auxiliary data block, are not emitted.

use core::fmt::Write as _;

use super::grid::{
    Grid, Ionex, BAND, BASE_RADIUS, COMMENT, DESCRIPTION, ELEVATION_CUTOFF, END_OF_FILE,
    END_OF_HEADER, END_OF_HEIGHT_MAP, END_OF_RMS_MAP, END_OF_TEC_MAP, EPOCH_OF_CURRENT_MAP,
    EPOCH_OF_FIRST_MAP, EPOCH_OF_LAST_MAP, EXPONENT, HGT_AXIS, INTERVAL, LAT_AXIS, LON_AXIS,
    MAPPING_FUNCTION, MAPS_IN_FILE, MAP_DIMENSION, NON_AVAILABLE, OBSERVABLES_USED,
    PGM_RUN_BY_DATE, SATELLITES, START_OF_HEIGHT_MAP, START_OF_RMS_MAP, START_OF_TEC_MAP, STATIONS,
    VERSION_TYPE,
};
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

        let scale = libm::pow(10.0, self.exponent() as f64);
        for (index, epoch) in self.map_epochs().iter().enumerate() {
            let map_number = index + 1;
            write_labeled(&mut out, &format!("{map_number:6}"), START_OF_TEC_MAP);
            write_labeled(&mut out, &epoch_data(*epoch), EPOCH_OF_CURRENT_MAP);
            self.write_map(&mut out, &self.tec_maps()[index], scale);
            write_labeled(&mut out, &format!("{map_number:6}"), END_OF_TEC_MAP);
        }
        for (maps, start, end) in [
            (self.rms_maps(), START_OF_RMS_MAP, END_OF_RMS_MAP),
            (self.height_maps(), START_OF_HEIGHT_MAP, END_OF_HEIGHT_MAP),
        ] {
            for (index, grid) in maps.iter().enumerate() {
                let map_number = index + 1;
                write_labeled(&mut out, &format!("{map_number:6}"), start);
                if let Some(epoch) = self.map_epochs().get(index) {
                    write_labeled(&mut out, &epoch_data(*epoch), EPOCH_OF_CURRENT_MAP);
                }
                self.write_map(&mut out, grid, scale);
                write_labeled(&mut out, &format!("{map_number:6}"), end);
            }
        }
        write_labeled(&mut out, "", END_OF_FILE);
        out
    }

    fn write_header(&self, out: &mut String) {
        let header = self.header();
        let lat1 = self.lat_nodes_deg().first().copied().unwrap_or(0.0);
        let lat2 = self.lat_nodes_deg().last().copied().unwrap_or(0.0);
        let lon1 = self.lon_nodes_deg().first().copied().unwrap_or(0.0);
        let lon2 = self.lon_nodes_deg().last().copied().unwrap_or(0.0);

        write_labeled(
            out,
            &format!(
                "{:8.1}{:12}{:<20}{}",
                header.version, "", "IONOSPHERE MAPS", header.satellite_system
            ),
            VERSION_TYPE,
        );
        write_labeled(
            out,
            &format!("{:<20}{:<20}{}", header.program, header.run_by, header.date),
            PGM_RUN_BY_DATE,
        );
        for description in &header.descriptions {
            write_labeled(out, description, DESCRIPTION);
        }
        if let (Some(first), Some(last)) = (self.map_epochs().first(), self.map_epochs().last()) {
            write_labeled(out, &epoch_data(*first), EPOCH_OF_FIRST_MAP);
            write_labeled(out, &epoch_data(*last), EPOCH_OF_LAST_MAP);
        }
        write_labeled(out, &format!("{:6}", header.interval_s), INTERVAL);
        write_labeled(out, &format!("{:6}", self.map_epochs().len()), MAPS_IN_FILE);
        let mapping_function = header
            .mapping_function
            .as_ref()
            .map_or(String::new(), |function| format!("  {}", function.code()));
        write_labeled(out, &mapping_function, MAPPING_FUNCTION);
        write_labeled(
            out,
            &format!("{:8.1}", header.elevation_cutoff_deg),
            ELEVATION_CUTOFF,
        );
        write_labeled(out, &header.observables_used, OBSERVABLES_USED);
        if let Some(count) = header.station_count {
            write_labeled(out, &format!("{count:6}"), STATIONS);
        }
        if let Some(count) = header.satellite_count {
            write_labeled(out, &format!("{count:6}"), SATELLITES);
        }
        write_labeled(out, &format!("{:8.1}", self.base_radius_km()), BASE_RADIUS);
        write_labeled(out, &format!("{:6}", 2), MAP_DIMENSION);
        let height = self.shell_height_km();
        write_labeled(
            out,
            &format!("{height:8.1}{height:8.1}{:8.1}", 0.0),
            HGT_AXIS,
        );
        write_labeled(
            out,
            &format!("{lat1:8.1}{lat2:8.1}{:8.1}", self.dlat_deg()),
            LAT_AXIS,
        );
        write_labeled(
            out,
            &format!("{lon1:8.1}{lon2:8.1}{:8.1}", self.dlon_deg()),
            LON_AXIS,
        );
        write_labeled(out, &format!("{:6}", self.exponent()), EXPONENT);
        for comment in &header.comments {
            write_labeled(out, comment, COMMENT);
        }
        write_labeled(out, "", END_OF_HEADER);
    }

    /// Emit one map's latitude bands. Each band is a `LAT/LON1/LON2/DLON/H`
    /// record followed by the band's scaled integer fields, with `9999` for a
    /// non-available value.
    fn write_map(&self, out: &mut String, grid: &Grid, scale: f64) {
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
                BAND,
            );
            for chunk in band.chunks(VALUES_PER_LINE) {
                for value in chunk {
                    let scaled = match value {
                        Some(value) => (value / scale).round() as i64,
                        None => NON_AVAILABLE,
                    };
                    // Emit a guaranteed leading space: values within I5 keep
                    // their right-justified five-column form, and wider values
                    // stay separated.
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

/// The `6I6` data of an epoch record, the inverse of the parser's epoch read.
// invariant: serializable IONEX epochs are whole, representable J2000 seconds.
#[allow(clippy::expect_used)]
fn epoch_data(epoch: Instant) -> String {
    let seconds =
        j2000_seconds_from_instant(epoch).expect("IONEX map epoch is convertible to J2000 seconds");
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(seconds);
    format!("{year:6}{month:6}{day:6}{hour:6}{minute:6}{second:6}")
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
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
