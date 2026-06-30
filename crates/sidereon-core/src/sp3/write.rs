//! SP3 serialization - the inverse of the parser ([`super::Sp3::parse`]).
//!
//! Pure and deterministic: the same [`Sp3`] always produces byte-identical text.
//! No I/O. A read -> (merge) -> write pipeline round-trips: re-parsing the output
//! yields the same epochs, satellites, positions, and clocks to SP3 format
//! precision (mm / sub-ns). Header fields are derived from the product, never
//! hardcoded; parsed per-satellite accuracy codes are preserved, while the
//! `%f`/`%i` base descriptors are emitted as standard defaults.
//!
//! A satellite absent at an epoch is written as the SP3 missing-orbit sentinel
//! (`0.0 0.0 0.0`, bad clock), never a fabricated position - so a quarantined
//! `(sat, epoch)` cell from [`super::merge`] re-reads as missing, not zero. For
//! velocity products the matching `V` record is still emitted, using the SP3
//! missing-velocity vector and bad clock-rate sentinel when needed.

use core::fmt::Write as _;

use crate::astro::time::civil::civil_from_julian_day_number as civil_from_jdn;
use crate::constants::{KM_TO_M, SECONDS_PER_DAY, US_TO_S};

use super::{
    Sp3, Sp3DataType, Sp3Flags, Sp3TimeSystem, Sp3Version, BAD_CLOCK_US, CLOCK_RATE_TO_S_PER_S,
    DM_S_TO_M_S, MISSING_POSITION_KM, MISSING_VELOCITY_DM_S,
};

/// Maximum SP3 satellite-id slots per `+` / `++` header line.
const SATS_PER_LINE: usize = 17;
/// SP3-c fixes five `+`/`++` lines (85 slots); SP3-d may use more.
const MIN_PLUS_LINES: usize = 5;
const SP3_TIME_TICKS_PER_SECOND: i64 = 100_000_000;
const SP3_TIME_TICKS_PER_MINUTE: i64 = 60 * SP3_TIME_TICKS_PER_SECOND;
const SP3_TIME_TICKS_PER_HOUR: i64 = 60 * SP3_TIME_TICKS_PER_MINUTE;
const SP3_TIME_TICKS_PER_DAY: i64 = 24 * SP3_TIME_TICKS_PER_HOUR;

impl Sp3 {
    /// Serialize this product to standard SP3 text (the format named by its
    /// header version, `c` or `d`).
    ///
    /// Pure and deterministic. See this module's docs for the round-trip
    /// and missing-satellite guarantees.
    pub fn to_sp3_string(&self) -> String {
        let mut out =
            String::with_capacity(self.epochs.len() * (self.header.satellites.len() + 4) * 61);
        self.write_header(&mut out);
        self.write_records(&mut out);
        out.push_str("EOF\n");
        out
    }

    fn write_header(&self, out: &mut String) {
        let h = &self.header;
        let version = match h.version {
            Sp3Version::A => 'a',
            Sp3Version::B => 'b',
            Sp3Version::C => 'c',
            Sp3Version::D => 'd',
        };
        let dtype = match h.data_type {
            Sp3DataType::Position => 'P',
            Sp3DataType::Velocity => 'V',
        };

        // Line 1: version/type, first-epoch calendar (cosmetic - the parser reads
        // epochs from the `*` lines), epoch count, data descriptor, coordinate
        // system, orbit type, agency. Columns match the parser's field offsets.
        let (y, mo, d, hh, mi, ss) = self
            .epochs
            .first()
            .map(|epoch| julian_to_civil(epoch, h.time_system))
            .unwrap_or((2000, 1, 1, 0, 0, 0.0));
        let dt = format_calendar(y, mo, d, hh, mi, ss);
        let _ = writeln!(
            out,
            "#{version}{dtype}{dt} {n:>7} {data:<5}{coord:>6}{orbit:>4} {agency}",
            n = self.epochs.len(),
            data = "ORBIT",
            coord = h.coordinate_system,
            orbit = h.orbit_type,
            agency = h.agency,
        );

        // Line 2 (`##`): GPS week, seconds-of-week, epoch interval, MJD, MJD frac.
        let _ = writeln!(
            out,
            "## {wk:>4} {sow:15.8} {interval:14.8} {mjd:>5} {frac:.13}",
            wk = h.gnss_week,
            sow = h.seconds_of_week,
            interval = h.epoch_interval_s,
            mjd = h.mjd,
            frac = h.mjd_fraction,
        );

        // `+` satellite-id lines and `++` accuracy-exponent lines.
        let sats = &h.satellites;
        let n_lines = MIN_PLUS_LINES.max(sats.len().div_ceil(SATS_PER_LINE));
        for line in 0..n_lines {
            // `+` line: first carries the count in columns 3-5; all start ids at 9.
            if line == 0 {
                let _ = write!(out, "+  {:>3}   ", sats.len());
            } else {
                out.push_str("+        ");
            }
            for slot in 0..SATS_PER_LINE {
                match sats.get(line * SATS_PER_LINE + slot) {
                    Some(sat) => {
                        let _ = write!(out, "{sat}");
                    }
                    None => out.push_str("  0"),
                }
            }
            out.push('\n');
        }
        for line in 0..n_lines {
            out.push_str("++       ");
            for slot in 0..SATS_PER_LINE {
                let idx = line * SATS_PER_LINE + slot;
                let code = if idx < sats.len() {
                    h.satellite_accuracy_codes.get(idx).copied().unwrap_or(0)
                } else {
                    0
                };
                let _ = write!(out, "{code:>3}");
            }
            out.push('\n');
        }

        // `%c` descriptors - the first carries the time system at columns 9-11
        // (the only `%c` content the parser reads). `%f`/`%i` are standard
        // base/accuracy descriptors the parser skips.
        let tsys = h.time_system.label();
        let _ = writeln!(
            out,
            "%c M  cc {tsys} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
        );
        out.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        out.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
        out.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
        out.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        out.push_str("%i    0    0    0    0      0      0      0      0         0\n");

        // Provenance comments (e.g. merge derivation) are preserved when present.
        for comment in &self.comments {
            let _ = writeln!(out, "/* {comment}");
        }
    }

    fn write_records(&self, out: &mut String) {
        let with_velocity = matches!(self.header.data_type, Sp3DataType::Velocity);
        for (idx, epoch) in self.epochs.iter().enumerate() {
            let (y, mo, d, hh, mi, ss) = julian_to_civil(epoch, self.header.time_system);
            let _ = writeln!(out, "*  {}", format_calendar(y, mo, d, hh, mi, ss));

            let states = &self.states[idx];
            // Every header satellite gets a record at every epoch; an absent one
            // is the missing-orbit sentinel (so a quarantined cell is "missing",
            // never a fabricated zero position). Velocity products also get the
            // paired V record, using the missing-velocity sentinel as needed.
            for sat in &self.header.satellites {
                if let Some(state) = states.get(sat) {
                    let p = state.position;
                    let clk = clock_field_us(state.clock_s);
                    let _ = write!(
                        out,
                        "P{sat}{:14.6}{:14.6}{:14.6}{clk:14.6}",
                        p.x_m / KM_TO_M,
                        p.y_m / KM_TO_M,
                        p.z_m / KM_TO_M,
                    );
                    write_record_flags(out, state.flags);
                    out.push('\n');
                    if with_velocity {
                        write_velocity_record(out, sat, state.velocity, state.clock_rate_s_s);
                    }
                } else {
                    let _ = writeln!(
                        out,
                        "P{sat}{:14.6}{:14.6}{:14.6}{:14.6}",
                        MISSING_POSITION_KM, MISSING_POSITION_KM, MISSING_POSITION_KM, BAD_CLOCK_US
                    );
                    if with_velocity {
                        write_velocity_record(out, sat, None, None);
                    }
                }
            }
        }
    }
}

/// SP3 clock column (microseconds): the bad-clock sentinel when absent.
fn clock_field_us(clock_s: Option<f64>) -> f64 {
    match clock_s {
        Some(s) => s / US_TO_S,
        None => BAD_CLOCK_US,
    }
}

/// SP3 clock-rate column: the bad-clock sentinel when absent.
fn clock_rate_field(rate_s_s: Option<f64>) -> f64 {
    match rate_s_s {
        Some(r) => r / CLOCK_RATE_TO_S_PER_S,
        None => BAD_CLOCK_US,
    }
}

fn write_velocity_record(
    out: &mut String,
    sat: &crate::id::GnssSatelliteId,
    velocity: Option<crate::frame::ItrfVelocityMS>,
    clock_rate_s_s: Option<f64>,
) {
    let ((vx, vy, vz), rate) = match velocity {
        Some(v) => (
            (
                v.vx_m_s / DM_S_TO_M_S,
                v.vy_m_s / DM_S_TO_M_S,
                v.vz_m_s / DM_S_TO_M_S,
            ),
            clock_rate_field(clock_rate_s_s),
        ),
        None => (
            (
                MISSING_VELOCITY_DM_S,
                MISSING_VELOCITY_DM_S,
                MISSING_VELOCITY_DM_S,
            ),
            BAD_CLOCK_US,
        ),
    };
    let _ = writeln!(out, "V{sat}{vx:14.6}{vy:14.6}{vz:14.6}{rate:14.6}");
}

fn write_record_flags(out: &mut String, flags: Sp3Flags) {
    let last_col = if flags.orbit_predicted {
        Some(79)
    } else if flags.maneuver {
        Some(78)
    } else if flags.clock_predicted {
        Some(75)
    } else if flags.clock_event {
        Some(74)
    } else {
        None
    };
    let Some(last_col) = last_col else {
        return;
    };

    for col in 60..=last_col {
        out.push(match col {
            74 if flags.clock_event => 'E',
            75 if flags.clock_predicted => 'P',
            78 if flags.maneuver => 'M',
            79 if flags.orbit_predicted => 'P',
            _ => ' ',
        });
    }
}

/// `YYYY MM DD HH MM SS.SSSSSSSS` in the SP3 epoch-line / line-1 layout.
fn format_calendar(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    seconds: f64,
) -> String {
    format!("{year:4} {month:>2} {day:>2} {hour:>2} {minute:>2} {seconds:11.8}")
}

/// Inverse of `super::civil_to_julian_split`: a Julian-date Instant back to the
/// civil `(year, month, day, hour, minute, seconds)` it was built from.
fn julian_to_civil(
    epoch: &crate::astro::time::model::Instant,
    time_system: Sp3TimeSystem,
) -> (i64, i64, i64, i64, i64, f64) {
    let Some(split) = epoch.julian_date() else {
        return (2000, 1, 1, 0, 0, 0.0);
    };
    let ticks =
        (split.fraction * SECONDS_PER_DAY * SP3_TIME_TICKS_PER_SECOND as f64).round() as i64;

    // `jd_whole` is the `*.5` midnight boundary (jdn - 0.5); recover the integer
    // Julian Day Number, then the Fliegel-Van Flandern inverse for the calendar.
    let mut jdn = (split.jd_whole + 0.5).round() as i64;
    if is_utc_like(time_system)
        && (SP3_TIME_TICKS_PER_DAY..SP3_TIME_TICKS_PER_DAY + SP3_TIME_TICKS_PER_SECOND)
            .contains(&ticks)
    {
        let (year, month, day) = civil_from_jdn(jdn);
        let seconds =
            60.0 + (ticks - SP3_TIME_TICKS_PER_DAY) as f64 / SP3_TIME_TICKS_PER_SECOND as f64;
        return (year, month, day, 23, 59, seconds);
    }

    jdn += ticks.div_euclid(SP3_TIME_TICKS_PER_DAY);
    let ticks = ticks.rem_euclid(SP3_TIME_TICKS_PER_DAY);
    let (year, month, day) = civil_from_jdn(jdn);

    let hour = ticks / SP3_TIME_TICKS_PER_HOUR;
    let rem = ticks % SP3_TIME_TICKS_PER_HOUR;
    let minute = rem / SP3_TIME_TICKS_PER_MINUTE;
    let seconds = (rem % SP3_TIME_TICKS_PER_MINUTE) as f64 / SP3_TIME_TICKS_PER_SECOND as f64;
    (year, month, day, hour, minute, seconds)
}

fn is_utc_like(time_system: Sp3TimeSystem) -> bool {
    matches!(time_system, Sp3TimeSystem::Glonass | Sp3TimeSystem::Utc)
}
