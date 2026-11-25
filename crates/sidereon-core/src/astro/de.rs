//! JPL Development Ephemeris (DE) binary files.
//!
//! Two fixed-size header records contain the layout and constants. Subsequent
//! records contain Chebyshev coefficients, indexed by their own Julian-date
//! bounds. The reader accepts either byte order and the INPOP extensions to
//! the DE layout. It follows Orekit 13.1.10's `JPLEphemeridesLoader` and
//! `PosVelChebyshev`, including coefficient scaling before interpolation
//! and the Earth–Moon barycentre translation from ICRF to GCRF.
//!
//! Epochs are TDB coordinate seconds from JD 2451545.0 in TDB. File coverage
//! includes both endpoints. A date outside the available records, including
//! a gap between records, returns [`DeError::OutsideCoverage`]. At interior
//! granule starts the new granule applies, as in Orekit with a cold cache.
//! A warm cache can select the preceding endpoint because it estimates steps
//! in elapsed TAI across uniform TDB granules. At each body's final granule
//! start, the preceding granule applies at its endpoint (`t = +1`) if it
//! ends at that start. Across a record gap the new granule applies at
//! `t = -1`. Orekit can extrapolate the preceding granule across the gap or
//! fail to supply a state there. Later dates use the final granule.
//!
//! If a component is non-finite in Orekit order, the reader sums exact
//! products of the stored coefficients and binary64 Chebyshev values.
//! Unit conversion, velocity scaling, and the geocentric translation use
//! the same binary64 factors with exact arithmetic. The reader rounds
//! once per recovered component and reports [`DeError::NonFiniteState`]
//! for undefined results or overflow.
//!
//! Copyright 2002–2025 CS GROUP for the Orekit algorithms. See the crate's
//! third-party notices and Apache License, Version 2.0.

use std::{collections::BTreeMap, path::Path};

use crate::astro::bodies::sun_moon::SunMoon;
use crate::astro::constants::time::J2000_JD;
use crate::astro::events::eclipse::fast_math::sin_cos;
use crate::astro::frames::iers2010::date::Offset;
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use thiserror::Error;

const DAY: u32 = 86_400;
const TYPE_OFFSET: usize = 2840;
const SIZE_OFFSET: usize = 2856;
const POINTERS: usize = 2696;
const LIBRATION: usize = 2844;
const EMB: usize = 2;
const MOON: usize = 9;
const SUN: usize = 10;

/// Failure to open, parse or evaluate a DE binary ephemeris.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DeError {
    /// The file could not be read.
    #[error("Unable to read DE ephemeris: {0}")]
    Io(#[from] std::io::Error),
    /// The file has an invalid header, record layout or time interval.
    #[error("Invalid DE ephemeris: {0}")]
    InvalidFile(&'static str),
    /// The epoch is not finite.
    #[error("Non-finite DE epoch")]
    NonFiniteEpoch,
    /// No record covers the requested TDB epoch.
    #[error("DE ephemeris does not cover TDB epoch {epoch_tdb_seconds} s from J2000")]
    OutsideCoverage {
        /// Requested coordinate seconds from J2000 in TDB.
        epoch_tdb_seconds: f64,
    },
    /// The selected coefficients do not produce a finite state.
    #[error("Non-finite DE state")]
    NonFiniteState,
}

/// A body with a gravitational mass parameter in the DE header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeBody {
    /// Sun.
    Sun,
    /// Mercury.
    Mercury,
    /// Venus.
    Venus,
    /// Earth–Moon barycentre.
    EarthMoon,
    /// Earth.
    Earth,
    /// Moon.
    Moon,
    /// Mars system.
    Mars,
    /// Jupiter system.
    Jupiter,
    /// Saturn system.
    Saturn,
    /// Uranus system.
    Uranus,
    /// Neptune system.
    Neptune,
    /// Pluto system.
    Pluto,
    /// Sum over the solar system.
    SolarSystem,
}

/// Geocentric position and velocity in GCRF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeState {
    /// Cartesian position in metres.
    pub position_m: [f64; 3],
    /// Cartesian velocity in metres per second.
    pub velocity_m_s: [f64; 3],
}

/// Simultaneous geocentric Sun and Moon states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeSunMoon {
    /// Sun state.
    pub sun: DeState,
    /// Moon state.
    pub moon: DeState,
}

#[derive(Debug, Clone, Copy)]
enum Endian {
    Big,
    Little,
}

impl Endian {
    fn u32(self, bytes: &[u8], offset: usize) -> u32 {
        let bytes = bytes[offset..offset + 4].try_into().unwrap();
        match self {
            Self::Big => u32::from_be_bytes(bytes),
            Self::Little => u32::from_le_bytes(bytes),
        }
    }

    fn f64(self, bytes: &[u8], offset: usize) -> f64 {
        let bytes = bytes[offset..offset + 8].try_into().unwrap();
        match self {
            Self::Big => f64::from_be_bytes(bytes),
            Self::Little => f64::from_le_bytes(bytes),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    first: usize,
    coefficients: usize,
    granules: u32,
}

impl Layout {
    fn new(endian: Endian, bytes: &[u8], offset: usize) -> Self {
        Self {
            first: endian.u32(bytes, offset) as usize,
            coefficients: endian.u32(bytes, offset + 4) as usize,
            granules: endian.u32(bytes, offset + 8),
        }
    }

    fn words(self, components: usize) -> Option<usize> {
        self.coefficients
            .checked_mul(self.granules as usize)?
            .checked_mul(components)
    }
}

#[derive(Debug, Clone, Copy)]
struct Record {
    start: Offset,
    end: Offset,
}

/// An immutable JPL DE binary ephemeris.
#[derive(Debug)]
pub struct DeEphemeris {
    bytes: Vec<u8>,
    endian: Endian,
    version: u32,
    record_size: usize,
    constants: BTreeMap<String, f64>,
    coverage: Record,
    records: Vec<Record>,
    layouts: [Layout; 11],
    span_days: Offset,
    time: FileTime,
    components: usize,
    position_unit: f64,
}

impl DeEphemeris {
    /// Open and parse a DE binary file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DeError> {
        Self::parse(std::fs::read(path)?)
    }

    /// Parse a DE binary file from caller-supplied bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeError> {
        Self::parse(bytes.to_vec())
    }

    fn parse(bytes: Vec<u8>) -> Result<Self, DeError> {
        if bytes.len() < SIZE_OFFSET + 4 {
            return Err(DeError::InvalidFile("incomplete first header record"));
        }
        let endian = if Endian::Big.u32(&bytes, TYPE_OFFSET) > 1 << 15 {
            Endian::Little
        } else {
            Endian::Big
        };
        let version = endian.u32(&bytes, TYPE_OFFSET);
        let inpop = version == 100;
        let layouts =
            core::array::from_fn(|slot| Layout::new(endian, &bytes, POINTERS + 12 * slot));
        let nutation = Layout::new(endian, &bytes, POINTERS + 12 * 11);
        let libration = Layout::new(endian, &bytes, LIBRATION);
        let record_size = if inpop {
            (endian.u32(&bytes, SIZE_OFFSET) as usize).checked_mul(8)
        } else {
            layouts
                .iter()
                .try_fold(2_usize, |total, layout| total.checked_add(layout.words(3)?))
                .and_then(|total| total.checked_add(nutation.words(2)?))
                .and_then(|total| total.checked_add(libration.words(3)?))
                .and_then(|total| total.checked_mul(8))
        }
        .filter(|size| *size >= SIZE_OFFSET + 4 && bytes.len() / size >= 3)
        .ok_or(DeError::InvalidFile("invalid record size"))?;
        if !bytes.len().is_multiple_of(record_size) {
            return Err(DeError::InvalidFile("incomplete data record"));
        }

        let values = &bytes[record_size..2 * record_size];
        let mut constants = BTreeMap::new();
        for slot in 0..400 {
            let offset = 252 + 6 * slot;
            let name = std::str::from_utf8(&bytes[offset..offset + 6])
                .map_err(|_| DeError::InvalidFile("invalid constant name"))?
                .trim_matches(|ch| ch <= ' ');
            if name.is_empty() {
                break;
            }
            if 8 * (slot + 1) > values.len() {
                return Err(DeError::InvalidFile("incomplete constant values"));
            }
            let value = endian.f64(values, 8 * slot);
            constants.insert(name.to_owned(), value);
        }
        let au = endian.f64(&bytes, 2680);
        let emrat = endian.f64(&bytes, 2688);
        let loaded_au = *constants.entry("AU".into()).or_insert(au);
        let loaded_emrat = *constants.entry("EMRAT".into()).or_insert(emrat);
        if !(1.4e8..=1.6e8).contains(&au)
            || !(80.0..=82.0).contains(&emrat)
            || !((1000.0 * loaded_au - 1000.0 * au).abs() < 10.0
                && (loaded_emrat - emrat).abs() < 1.0e-5)
        {
            return Err(DeError::InvalidFile("inconsistent AU or EMRAT"));
        }

        let components = if inpop
            && constants
                .get("FORMAT")
                .is_some_and(|value| value.is_finite() && libm::remainder(*value, 10.0) as i32 != 1)
        {
            6
        } else {
            3
        };
        let position_unit = if inpop
            && constants
                .get("UNITE")
                .is_some_and(|v| v.is_finite() && *v as i32 == 0)
        {
            loaded_au * 1000.0
        } else {
            1000.0
        };
        let time = if inpop && constants.get("TIMESC").is_some_and(|v| *v as i32 == 1) {
            FileTime::Tcb
        } else {
            FileTime::Tdb
        };
        for layout in [layouts[EMB], layouts[MOON], layouts[SUN]] {
            if layout.first < 3
                || layout.coefficients == 0
                || layout.granules == 0
                || layout
                    .words(components)
                    .and_then(|words| (layout.first - 1).checked_add(words))
                    .is_none_or(|end| end > record_size / 8)
            {
                return Err(DeError::InvalidFile(
                    "invalid Sun, Moon or barycentre layout",
                ));
            }
        }

        let span = endian.f64(&bytes, 2668);
        if !(0.0 < span && span < 100.0) {
            return Err(DeError::InvalidFile("invalid record span"));
        }
        let span_days = Offset::from_seconds(span).unwrap();
        if [layouts[EMB], layouts[MOON], layouts[SUN]]
            .iter()
            .any(|layout| span_days.divided_by(layout.granules) == Offset::seconds(0))
        {
            return Err(DeError::InvalidFile("zero granule duration"));
        }
        let read_range = |record: &[u8], offset| -> Result<Record, DeError> {
            Ok(Record {
                start: time.tai_at_julian_date(endian.f64(record, offset))?,
                end: time.tai_at_julian_date(endian.f64(record, offset + 8))?,
            })
        };
        let coverage = read_range(&bytes, 2652)?;
        let mut records: Vec<Record> = Vec::with_capacity(bytes.len() / record_size - 2);
        for record in bytes.chunks_exact(record_size).skip(2) {
            let range = read_range(record, 0)?;
            if range.start >= range.end
                || range.start < coverage.start
                || range.end > coverage.end
                || records.last().is_some_and(|last| range.start < last.end)
            {
                return Err(DeError::InvalidFile("invalid data record bounds"));
            }
            records.push(range);
        }
        Ok(Self {
            bytes,
            endian,
            version,
            record_size,
            constants,
            coverage,
            records,
            layouts,
            span_days,
            time,
            components,
            position_unit,
        })
    }

    /// DE ephemeris number, such as 440 for DE440.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Size of each header or data record, in bytes.
    pub fn record_size(&self) -> usize {
        self.record_size
    }

    /// Inclusive bounds declared by the file, in TDB seconds from J2000.
    pub fn coverage_tdb_seconds(&self) -> (f64, f64) {
        (
            FileTime::Tdb.coordinate(self.coverage.start).as_seconds(),
            FileTime::Tdb.coordinate(self.coverage.end).as_seconds(),
        )
    }

    /// Header constants in the units stored in the file.
    pub fn constants(&self) -> &BTreeMap<String, f64> {
        &self.constants
    }

    /// Astronomical unit in metres.
    pub fn astronomical_unit_m(&self) -> f64 {
        1000.0 * self.constants["AU"]
    }

    /// Ratio of Earth mass to Moon mass.
    pub fn earth_moon_mass_ratio(&self) -> f64 {
        self.constants["EMRAT"]
    }

    /// Gravitational mass parameter in m³/s², if present.
    pub fn gm_m3_s2(&self, body: DeBody) -> Option<f64> {
        use DeBody::*;
        let names = match body {
            Sun => ["GMS", "GM_Sun"],
            Mercury => ["GM1", "GM_Mer"],
            Venus => ["GM2", "GM_Ven"],
            EarthMoon => ["GMB", "GM_EMB"],
            Mars => ["GM4", "GM_Mar"],
            Jupiter => ["GM5", "GM_Jup"],
            Saturn => ["GM6", "GM_Sat"],
            Uranus => ["GM7", "GM_Ura"],
            Neptune => ["GM8", "GM_Nep"],
            Pluto => ["GM9", "GM_Plu"],
            Earth => return Some(self.earth_moon_mass_ratio() * self.gm_m3_s2(Moon)?),
            Moon => return Some(self.gm_m3_s2(EarthMoon)? / (1.0 + self.earth_moon_mass_ratio())),
            SolarSystem => {
                let mut sum = self.gm_m3_s2(Sun)?;
                for body in [
                    Mercury, Venus, EarthMoon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto,
                ] {
                    sum += self.gm_m3_s2(body)?;
                }
                return sum.is_finite().then_some(sum);
            }
        };
        let raw = names.iter().find_map(|name| self.constants.get(*name))?;
        let au = self.astronomical_unit_m();
        let day = f64::from(DAY);
        let gm = raw * au * au * au / (day * day);
        if gm.is_finite() {
            return Some(gm);
        }
        let gm = ((raw / day) * (au / day)) * au * au;
        gm.is_finite().then_some(gm)
    }

    /// Evaluate geocentric GCRF positions in metres at a TDB epoch.
    pub fn sun_moon_positions_at_tdb_seconds(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<SunMoon, DeError> {
        let bodies = self.evaluate::<false>(epoch_tdb_seconds)?;
        Ok(SunMoon {
            sun: bodies.sun.position_m,
            moon: bodies.moon.position_m,
        })
    }

    /// Evaluate geocentric GCRF positions and velocities at a TDB epoch.
    pub fn sun_moon_at_tdb_seconds(&self, epoch_tdb_seconds: f64) -> Result<DeSunMoon, DeError> {
        self.evaluate::<true>(epoch_tdb_seconds)
    }

    fn evaluate<const VELOCITIES: bool>(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<DeSunMoon, DeError> {
        if !epoch_tdb_seconds.is_finite() {
            return Err(DeError::NonFiniteEpoch);
        }
        let outside = || DeError::OutsideCoverage { epoch_tdb_seconds };
        // Header Julian days fit in a signed 32-bit integer.
        if epoch_tdb_seconds.abs() > 2.0e14 {
            return Err(outside());
        }
        let coordinate = Offset::from_seconds(epoch_tdb_seconds).ok_or_else(outside)?;
        let epoch = FileTime::Tdb.to_tai(coordinate);
        let index = self.records.partition_point(|record| record.start <= epoch);
        let index = index.checked_sub(1).ok_or_else(outside)?;
        let range = self.records[index];
        if epoch > range.end {
            return Err(outside());
        }
        let start_coordinate = self.time.coordinate(range.start);
        let epoch_coordinate = self.time.coordinate(epoch);
        let granule = |slot| self.granule(index, start_coordinate, epoch, epoch_coordinate, slot);
        let moon = granule(MOON);
        let emb = granule(EMB);
        let sun = granule(SUN);
        let moon_fraction = 1.0 / (1.0 + self.earth_moon_mass_ratio());

        let states = [
            sun.evaluate::<VELOCITIES>(),
            emb.evaluate::<VELOCITIES>(),
            moon.evaluate::<VELOCITIES>(),
        ];
        let bodies = compose_sun_moon(states, moon_fraction);
        if bodies.is_finite() {
            Ok(bodies)
        } else {
            recover_sun_moon(bodies, [&sun, &emb, &moon], moon_fraction)
        }
    }

    fn granule(
        &self,
        mut record_index: usize,
        start_coordinate: Offset,
        epoch: Offset,
        epoch_coordinate: Offset,
        slot: usize,
    ) -> Granule<'_> {
        let start = self.records[record_index].start;
        let layout = self.layouts[slot];
        let duration = self
            .span_days
            .divided_by(layout.granules)
            .multiplied_by(DAY);
        let mut low = 0;
        let mut high = layout.granules;
        let granule_start = |index| {
            if index == 0 {
                start
            } else {
                self.time
                    .to_tai(start_coordinate.plus(duration.multiplied_by(index)))
            }
        };
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if granule_start(middle) <= epoch {
                low = middle;
            } else {
                high = middle;
            }
        }
        let selected_start = granule_start(low);
        let final_start = record_index + 1 == self.records.len()
            && low + 1 == layout.granules
            && epoch == selected_start;
        let duration = duration.as_seconds();
        let t = if final_start
            && (low > 0 || (record_index > 0 && self.records[record_index - 1].end == start))
        {
            if low > 0 {
                low -= 1;
            } else {
                record_index -= 1;
                low = layout.granules - 1;
            }
            1.0
        } else {
            let selected_coordinate = if low == 0 {
                start_coordinate
            } else {
                self.time.coordinate(selected_start)
            };
            let elapsed = epoch_coordinate.minus(selected_coordinate).as_seconds();
            (2.0 * elapsed - duration) / duration
        };
        let offset = (record_index + 2) * self.record_size;
        Granule {
            endian: self.endian,
            bytes: &self.bytes[offset..offset + self.record_size],
            position_unit: self.position_unit,
            first: layout.first - 1 + self.components * low as usize * layout.coefficients,
            coefficients: layout.coefficients,
            duration,
            t,
        }
    }
}

struct Granule<'a> {
    endian: Endian,
    bytes: &'a [u8],
    position_unit: f64,
    first: usize,
    coefficients: usize,
    duration: f64,
    t: f64,
}

impl Granule<'_> {
    #[inline]
    fn coefficient(&self, axis: usize, k: usize) -> f64 {
        self.endian
            .f64(self.bytes, 8 * (self.first + axis * self.coefficients + k))
    }

    fn evaluate<const VELOCITIES: bool>(&self) -> DeState {
        let coefficient = |axis, k| self.position_unit * self.coefficient(axis, k);
        let twice_t = 2.0 * self.t;

        let mut position_m = core::array::from_fn(|axis| coefficient(axis, 0));
        let mut velocity = [0.0; 3];
        let (mut p_previous, mut p) = (1.0, self.t);
        let (mut q_previous, mut q) = (0.0, 1.0);
        for k in 1..self.coefficients {
            for axis in 0..3 {
                let ck = coefficient(axis, k);
                position_m[axis] += ck * p;
                if VELOCITIES {
                    velocity[axis] += ck * q;
                }
            }
            (p_previous, p) = (p, twice_t * p - p_previous);
            if VELOCITIES {
                (q_previous, q) = (q, twice_t * q + 2.0 * p_previous - q_previous);
            }
        }
        let velocity_scale = 2.0 / self.duration;
        DeState {
            position_m,
            velocity_m_s: velocity.map(|value| value * velocity_scale),
        }
    }

    fn evaluate_exact(&self, axis: usize, velocity: bool) -> Option<BigInt> {
        let mut sum = BigInt::ZERO;
        let twice_t = 2.0 * self.t;
        let (mut p_previous, mut p) = (1.0, self.t);
        let (mut q_previous, mut q) = (0.0, 1.0);
        if !velocity {
            let (coefficient, shift) = binary64_integer(self.coefficient(axis, 0))?;
            sum = BigInt::from(coefficient) << (shift + BINARY64_SHIFT);
        }
        for k in 1..self.coefficients {
            let (coefficient, coefficient_shift) = binary64_integer(self.coefficient(axis, k))?;
            let (basis, basis_shift) = binary64_integer(if velocity { q } else { p })?;
            sum += BigInt::from(i128::from(coefficient) * i128::from(basis))
                << (coefficient_shift + basis_shift);
            (p_previous, p) = (p, twice_t * p - p_previous);
            if velocity {
                (q_previous, q) = (q, twice_t * q + 2.0 * p_previous - q_previous);
            }
        }
        let metres = scale_exact(sum, self.position_unit)?;
        if velocity {
            scale_exact(metres, 2.0 / self.duration)
        } else {
            Some(metres)
        }
    }
}

impl DeSunMoon {
    fn is_finite(self) -> bool {
        [
            self.sun.position_m,
            self.sun.velocity_m_s,
            self.moon.position_m,
            self.moon.velocity_m_s,
        ]
        .iter()
        .flatten()
        .all(|v| v.is_finite())
    }
}

fn compose_sun_moon([sun, emb, moon]: [DeState; 3], moon_fraction: f64) -> DeSunMoon {
    let geocentric = |sun: [f64; 3], emb: [f64; 3], moon: [f64; 3]| {
        core::array::from_fn(|axis| {
            let moon_offset = moon_fraction * moon[axis];
            // GCRF → EMB and EMB → ICRF are pure translations.
            // Compose them, invert, then translate the Sun.
            -(-moon_offset + emb[axis]) + sun[axis]
        })
    };
    let sun = DeState {
        position_m: geocentric(sun.position_m, emb.position_m, moon.position_m),
        velocity_m_s: geocentric(sun.velocity_m_s, emb.velocity_m_s, moon.velocity_m_s),
    };
    DeSunMoon { sun, moon }
}

#[cold]
#[inline(never)]
fn recover_sun_moon(
    mut bodies: DeSunMoon,
    [sun, emb, moon]: [&Granule<'_>; 3],
    moon_fraction: f64,
) -> Result<DeSunMoon, DeError> {
    for (velocity, sun_values, moon_values) in [
        (
            false,
            &mut bodies.sun.position_m,
            &mut bodies.moon.position_m,
        ),
        (
            true,
            &mut bodies.sun.velocity_m_s,
            &mut bodies.moon.velocity_m_s,
        ),
    ] {
        // Coefficient, Chebyshev value, metres per unit, and (for velocity) 2 / duration.
        let shift = BINARY64_SHIFT * if velocity { 4 } else { 3 };
        for (axis, (sun_value, moon_value)) in sun_values.iter_mut().zip(moon_values).enumerate() {
            if sun_value.is_finite() && moon_value.is_finite() {
                continue;
            }
            let lunar = moon
                .evaluate_exact(axis, velocity)
                .ok_or(DeError::NonFiniteState)?;
            if !moon_value.is_finite() {
                *moon_value = round_binary64(&lunar, shift)?;
            }
            if !sun_value.is_finite() {
                let barycentric = |body: &Granule<'_>| {
                    body.evaluate_exact(axis, velocity)
                        .ok_or(DeError::NonFiniteState)
                };
                let difference = (barycentric(sun)? - barycentric(emb)?) << BINARY64_SHIFT;
                let lunar_offset =
                    scale_exact(lunar, moon_fraction).ok_or(DeError::NonFiniteState)?;
                *sun_value = round_binary64(&(difference + lunar_offset), shift + BINARY64_SHIFT)?;
            }
        }
    }
    Ok(bodies)
}

// Every finite binary64 value is an integer multiple of 2^-1074. Keep the
// integer as a signed significand and a nonnegative shift. Each product of
// n binary64 factors then has a common quantum of 2^(-1074 * n).
const BINARY64_SHIFT: usize = 1074;

fn binary64_integer(value: f64) -> Option<(i64, usize)> {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as usize;
    if exponent == 0x7ff {
        return None;
    }
    let fraction = (bits & ((1 << 52) - 1)) as i64;
    let significand = fraction | (i64::from(exponent != 0) << 52);
    let signed = if bits >> 63 == 0 {
        significand
    } else {
        -significand
    };
    Some((signed, exponent.saturating_sub(1)))
}

fn scale_exact(value: BigInt, factor: f64) -> Option<BigInt> {
    let (significand, shift) = binary64_integer(factor)?;
    Some((value * significand) << shift)
}

fn round_binary64(value: &BigInt, shift: usize) -> Result<f64, DeError> {
    let magnitude = value.magnitude();
    // Retain 53 significant bits for normals, or round to the subnormal grid.
    let discarded = (magnitude.bits() as usize)
        .saturating_sub(53)
        .max(shift - BINARY64_SHIFT);
    let mut significand = (magnitude >> discarded).to_u64().unwrap();
    let halfway_bit = (discarded - 1) as u64;
    if magnitude.bit(halfway_bit)
        && (significand & 1 != 0
            || magnitude
                .trailing_zeros()
                .is_some_and(|zeros| zeros < halfway_bit))
    {
        significand += 1;
    }
    let rounded = libm::ldexp(significand as f64, discarded as i32 - shift as i32);
    if !rounded.is_finite() {
        return Err(DeError::NonFiniteState);
    }
    Ok(if value.sign() == Sign::Minus {
        -rounded
    } else {
        rounded
    })
}

#[derive(Debug, Clone, Copy)]
enum FileTime {
    Tdb,
    Tcb,
}

impl FileTime {
    fn offset_from_tai(self, tai: Offset) -> Offset {
        let tt_minus_tai = Offset::millis(32_184);
        let days = tai.plus(tt_minus_tai).as_seconds() / f64::from(DAY);
        let radians = |x: f64| {
            let high = f64::from_bits(x.to_bits() & !((1 << 30) - 1));
            let low = x - high;
            let a = 0.01745329052209854;
            let b = 1.997844754509471e-9;
            low * b + low * a + high * b + high * a
        };
        let (sin, cos) = sin_cos(radians(357.53) + radians(0.9856003) * days);
        let periodic = Offset::from_seconds(sin * (0.001658 + (2.0 * 0.000014) * cos)).unwrap();
        let offset = tt_minus_tai.plus(periodic);
        match self {
            Self::Tdb => offset,
            Self::Tcb => {
                let elapsed = tai.minus(Offset::seconds(-725_803_200)).as_seconds();
                offset.plus(Offset::from_seconds(1.550519768e-8 * elapsed).unwrap())
            }
        }
    }

    fn to_tai(self, coordinate: Offset) -> Offset {
        let mut tai = coordinate;
        for _ in 0..8 {
            let next = coordinate.minus(self.offset_from_tai(tai));
            if next == tai {
                break;
            }
            tai = next;
        }
        tai
    }

    fn coordinate(self, tai: Offset) -> Offset {
        tai.plus(self.offset_from_tai(tai))
    }

    fn tai_at_julian_date(self, jd: f64) -> Result<Offset, DeError> {
        if !jd.is_finite() || jd.abs() >= f64::from(i32::MAX) {
            return Err(DeError::InvalidFile("invalid Julian date"));
        }
        let mut day = jd.floor();
        let mut seconds = (jd + 0.5 - day) * f64::from(DAY);
        if seconds >= f64::from(DAY) {
            day += 1.0;
            seconds -= f64::from(DAY);
        }
        let noon = Offset::seconds((day - J2000_JD) as i64 * i64::from(DAY));
        let label = noon
            .plus(Offset::from_seconds(seconds).unwrap())
            .minus(Offset::seconds(43_200));
        Ok(self.to_tai(label))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_rounding_at_binary64_limits() {
        let shift = 2 * BINARY64_SHIFT;
        let half_subnormal = BigInt::from(1) << (BINARY64_SHIFT - 1);
        let max = scale_exact(BigInt::from(1) << BINARY64_SHIFT, f64::MAX).unwrap();
        let half_ulp = BigInt::from(1) << (shift + 970);

        for (value, bits) in [
            (&half_subnormal - 1, 0),
            (half_subnormal.clone(), 0),
            (&half_subnormal + 1, 1),
            (-&half_subnormal, (-0.0_f64).to_bits()),
            (&half_subnormal * 3, 2),
            (&max + &half_ulp - 1, f64::MAX.to_bits()),
        ] {
            assert_eq!(round_binary64(&value, shift).unwrap().to_bits(), bits);
        }
        assert!(matches!(
            round_binary64(&(max + half_ulp), shift),
            Err(DeError::NonFiniteState)
        ));
    }
}
