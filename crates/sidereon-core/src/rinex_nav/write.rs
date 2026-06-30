//! RINEX navigation serialization - the inverse of [`super::parse_nav`].
//!
//! Pure and deterministic: a given record set always produces byte-identical
//! text and no I/O is performed. The output is a minimal, well-formed RINEX 3
//! navigation file carrying the eight-line Keplerian broadcast-orbit block for
//! each supported (GPS, Galileo, BeiDou) record, so re-parsing it with
//! [`super::parse_nav`] reconstructs the same [`BroadcastRecord`]s.
//!
//! Round-trip scope. The canonical IR is the parsed [`BroadcastRecord`] set, not
//! the original bytes: fields the record does not retain (a Galileo data-source
//! word, IODE/IODC, transmission time) are written as canonical values that
//! re-decode to the same record - e.g. the data-source word is emitted as the
//! one bit pattern that classifies the stored [`super::NavMessage`]. Numeric
//! fields use the RINEX `D19.12` width, the same 13-significant-figure grid the
//! files carry, so a value read from a real file re-encodes to the same `f64`.

use core::fmt::Write as _;

use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::time::civil::civil_from_julian_day_number;
use crate::astro::time::gnss::week_epoch_julian_day_number;
use crate::astro::time::model::TimeScale;
use crate::astro::time::scales::julian_day_number;
use crate::id::GnssSystem;

use super::{BroadcastRecord, NavMessage};

/// Serialize broadcast navigation records to standard RINEX 3 navigation text.
///
/// The inverse of [`super::parse_nav`]: re-parsing the output yields the same
/// records. See the module documentation for the round-trip scope.
pub fn encode_nav(records: &[BroadcastRecord]) -> String {
    let mut out = String::with_capacity(64 + records.len() * 8 * 81);
    write_header(&mut out);
    for record in records {
        write_record(&mut out, record);
    }
    out
}

fn write_header(out: &mut String) {
    // Column 0-8 holds the version; column 20 the file type ('N' = NAV). Minor 4
    // keeps the parser off the legacy 3.02 fit-interval-flag path.
    let _ = writeln!(
        out,
        "{:<20}{:<40}{:<20}",
        "     3.04", "N: GNSS NAV DATA    M (MIXED)", "RINEX VERSION / TYPE"
    );
    let _ = writeln!(out, "{:<60}{:<20}", "", "END OF HEADER");
}

fn write_record(out: &mut String, record: &BroadcastRecord) {
    let sat = record.satellite_id;
    let system = sat.system;

    // SV / epoch / clock line: the toc civil epoch then the af0/af1/af2 clock
    // polynomial. The civil epoch is reconstructed from the record's `toc`, which
    // the parser recomputes back into the same week / seconds-of-week.
    let (year, month, day, hour, minute, second) = clock_epoch_civil(record);
    let sat_token = sat.to_string();
    let _ = write!(
        out,
        "{sat_token:<3} {year:04} {month:02} {day:02} {hour:02} {minute:02} {second:02}",
    );
    push_d19_12(out, record.clock.af0);
    push_d19_12(out, record.clock.af1);
    push_d19_12(out, record.clock.af2);
    out.push('\n');

    let e = &record.elements;
    // ORBIT-1 .. ORBIT-7, matching the fixed-column reader's field positions.
    write_orbit(out, [0.0, e.crs, e.delta_n, e.m0]);
    write_orbit(out, [e.cuc, e.e, e.cus, e.sqrt_a]);
    write_orbit(out, [e.toe_sow, e.cic, e.omega0, e.cis]);
    write_orbit(out, [e.i0, e.crc, e.omega, e.omega_dot]);
    write_orbit(
        out,
        [
            e.idot,
            data_source_word(system, record.message),
            f64::from(record.week),
            0.0,
        ],
    );
    write_orbit(
        out,
        [
            record.sv_accuracy_m,
            record.sv_health,
            group_delay_field(record, 0),
            group_delay_field(record, 1),
        ],
    );
    write_orbit(out, [0.0, fit_interval_hours(record), 0.0, 0.0]);
}

/// Reconstruct the civil toc epoch (integer seconds) from a record's `toc`
/// week / seconds-of-week, in the record's broadcast time scale.
fn clock_epoch_civil(record: &BroadcastRecord) -> (i64, i64, i64, i64, i64, i64) {
    let week = i64::from(record.toc.week);
    let sow = record.toc.tow_s.round() as i64;
    let base_jdn = week_epoch_jdn(record.toc.system);
    let total_jdn = base_jdn + week * 7 + sow.div_euclid(SECONDS_PER_DAY_I64);
    let tod = sow.rem_euclid(SECONDS_PER_DAY_I64);
    let (year, month, day) = civil_from_julian_day_number(total_jdn);
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;
    (year, month, day, hour, minute, second)
}

/// Julian Day Number of the start of a constellation's week numbering, the
/// inverse-side companion of [`super::gnss::week_from_calendar`]. Any scale that
/// never reaches a Keplerian broadcast record falls back to the GPS epoch.
fn week_epoch_jdn(scale: TimeScale) -> i64 {
    week_epoch_julian_day_number(scale).unwrap_or_else(|| julian_day_number(1980, 1, 6))
}

/// The Galileo data-source word that classifies `message`. GPS/BeiDou ignore this
/// column (their message is fixed or PRN-derived), so a zero word is written.
fn data_source_word(system: GnssSystem, message: NavMessage) -> f64 {
    if system != GnssSystem::Galileo {
        return 0.0;
    }
    match message {
        // Bit 1 selects F/NAV; bit 0 selects I/NAV (see `galileo_message`).
        NavMessage::GalileoFnav => 2.0,
        _ => 1.0,
    }
}

/// The two ORBIT-6 group-delay columns for a record (index 0 = field 3, index 1
/// = field 4), per the constellation's RINEX layout.
fn group_delay_field(record: &BroadcastRecord, index: usize) -> f64 {
    use super::BroadcastGroupDelayTerm as T;
    let gd = &record.group_delays;
    let term = match (record.satellite_id.system, index) {
        (GnssSystem::Gps, 0) => Some(T::GpsTgd),
        (GnssSystem::Galileo, 0) => Some(T::GalileoBgdE5aE1),
        (GnssSystem::Galileo, 1) => Some(T::GalileoBgdE5bE1),
        (GnssSystem::BeiDou, 0) => Some(T::BeidouTgd1),
        (GnssSystem::BeiDou, 1) => Some(T::BeidouTgd2),
        _ => None,
    };
    term.and_then(|t| gd.get(t)).unwrap_or(0.0)
}

/// The ORBIT-7 fit-interval column in hours. Only GPS broadcasts it; the value
/// re-decodes through [`super::gps_fit_interval_s`] (hours -> seconds) on a
/// non-legacy header.
fn fit_interval_hours(record: &BroadcastRecord) -> f64 {
    match (record.satellite_id.system, record.fit_interval_s) {
        (GnssSystem::Gps, Some(seconds)) => seconds / 3600.0,
        _ => 0.0,
    }
}

fn write_orbit(out: &mut String, values: [f64; 4]) {
    out.push_str("    ");
    for value in values {
        push_d19_12(out, value);
    }
    out.push('\n');
}

/// Append a value in RINEX `D19.12` fixed-width form: a leading sign or space,
/// one mantissa digit, twelve fraction digits, and a signed two-digit exponent
/// (e.g. ` 1.000000000000e+00`, `-2.907656250000e+02`), always 19 columns.
fn push_d19_12(out: &mut String, value: f64) {
    let negative = value.is_sign_negative() && value != 0.0;
    let magnitude = value.abs();
    let base = format!("{magnitude:.12e}");
    let (mantissa, _) = base.split_once('e').expect("scientific form has 'e'");
    let exponent = d19_12_exponent(value);
    let sign = if negative { '-' } else { ' ' };
    let _ = write!(out, "{sign}{mantissa}e{exponent:+03}");
}

/// The base-10 exponent [`push_d19_12`] emits for `value` (the rounded
/// `{:.12e}` exponent). Factored out so the parser shares the exact predicate.
fn d19_12_exponent(value: f64) -> i32 {
    let base = format!("{:.12e}", value.abs());
    let (_, exponent) = base.split_once('e').expect("scientific form has 'e'");
    exponent.parse().expect("scientific exponent parses")
}

/// Whether `value` fits the RINEX `D19.12` fixed field. The field reserves a
/// two-digit exponent (`e+NN`); a value whose rounded base-10 exponent needs
/// three digits would widen the field to 20 columns and shift every later
/// fixed column on reparse. Such a value cannot be represented in this format,
/// so the parser rejects it to keep the parse/encode domains aligned. Real
/// broadcast values have small exponents and are always representable.
pub(super) fn d19_12_representable(value: f64) -> bool {
    d19_12_exponent(value).unsigned_abs() <= 99
}
