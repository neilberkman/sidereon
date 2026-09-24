//! RTCM 3 coordinate transformation messages 1021-1027 (RTCM 10403.3 Section
//! 3.5.12): the Helmert / abridged Molodenski (1021) and Molodenski-Badekas
//! (1022) transformation parameters, the ellipsoidal (1023) and plane (1024)
//! residual grids, and the projection parameters (1025, 1026, 1027).
//!
//! Every field is stored as its raw transmitted integer, so a decode followed
//! by an encode reproduces the body bit for bit. The source and target names of
//! 1021 and 1022 are counted runs of 8-bit characters, each byte read as the
//! character with that code point (`U+0000`..=`U+00FF`), so every byte value is
//! kept.

use crate::error::{Error, Result};

use super::bits::{BitReader, FieldWriter};
use super::{decode_body, write_trailing, DecodeContext, DecodeResult, RtcmDeparture, RtcmPolicy};

/// The rotation point of a Molodenski-Badekas transformation (1022).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotationPoint {
    /// X coordinate `XP` (DF163, int35), scale 1 mm.
    pub x: i64,
    /// Y coordinate `YP` (DF164, int35), scale 1 mm.
    pub y: i64,
    /// Z coordinate `ZP` (DF165, int35), scale 1 mm.
    pub z: i64,
}

/// A decoded Helmert / abridged Molodenski (1021) or Molodenski-Badekas
/// (1022) transformation message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelmertTransformation {
    /// 1021 or 1022.
    pub message_number: u16,
    /// Source name (DF144), at most 31 characters (count DF143, 5 bits).
    pub source_name: String,
    /// Target name (DF146), at most 31 characters (count DF145, 5 bits).
    pub target_name: String,
    /// System identification number (DF147, 8 bits).
    pub system_id: u8,
    /// Utilized transformation message indicator (DF148, 10 bits).
    pub utilized_messages: u16,
    /// Plate number (DF149, 5 bits).
    pub plate_number: u8,
    /// Computation indicator (DF150, 4 bits).
    pub computation_indicator: u8,
    /// Height indicator (DF151, 2 bits).
    pub height_indicator: u8,
    /// Latitude of the origin of the area of validity (DF152, int19), scale
    /// 2 arc seconds.
    pub validity_latitude: i32,
    /// Longitude of the origin of the area of validity (DF153, int20), scale
    /// 2 arc seconds.
    pub validity_longitude: i32,
    /// North/south extension of the area of validity (DF154, 14 bits), scale
    /// 2 arc seconds.
    pub validity_extension_latitude: u16,
    /// East/west extension of the area of validity (DF155, 14 bits), scale
    /// 2 arc seconds.
    pub validity_extension_longitude: u16,
    /// Translation in X (DF156, int23), scale 1 mm.
    pub dx: i32,
    /// Translation in Y (DF157, int23), scale 1 mm.
    pub dy: i32,
    /// Translation in Z (DF158, int23), scale 1 mm.
    pub dz: i32,
    /// Rotation around the X axis (DF159, int32), scale 2e-5 arc seconds.
    pub r1: i32,
    /// Rotation around the Y axis (DF160, int32), scale 2e-5 arc seconds.
    pub r2: i32,
    /// Rotation around the Z axis (DF161, int32), scale 2e-5 arc seconds.
    pub r3: i32,
    /// Scale correction (DF162, int25).
    pub ds: i32,
    /// Rotation point (DF163..DF165), carried by 1022 and `None` in 1021.
    pub rotation_point: Option<RotationPoint>,
    /// Semi-major axis of the source ellipsoid, as transmitted (DF166, 24
    /// bits), scale 1 mm.
    pub add_as: u32,
    /// Semi-minor axis of the source ellipsoid, as transmitted (DF167, 25
    /// bits), scale 1 mm.
    pub add_bs: u32,
    /// Semi-major axis of the target ellipsoid, as transmitted (DF168, 24
    /// bits), scale 1 mm.
    pub add_at: u32,
    /// Semi-minor axis of the target ellipsoid, as transmitted (DF169, 25
    /// bits), scale 1 mm.
    pub add_bt: u32,
    /// Horizontal Helmert/Molodenski quality indicator (DF214, 3 bits).
    pub horizontal_quality: u8,
    /// Vertical Helmert/Molodenski quality indicator (DF215, 3 bits).
    pub vertical_quality: u8,
    /// Every body bit after the last field, the zeros that align the body to a
    /// byte included, kept whenever those bits are anything other than fewer
    /// than eight zeros; see [`RtcmDeparture::TrailingBits`]. Empty for every
    /// body read under [`RtcmPolicy::Strict`] and for a message built by hand.
    pub trailing_bits: Vec<bool>,
}

/// Read a counted run of 8-bit characters, each byte as the character with
/// that code point.
fn read_name(r: &mut BitReader<'_>, count_bits: usize) -> DecodeResult<String> {
    let count = r.u(count_bits)? as usize;
    let mut name = String::with_capacity(count);
    for _ in 0..count {
        name.push(char::from(r.u(8)? as u8));
    }
    Ok(name)
}

/// Write a counted run of 8-bit characters, refusing a character above
/// `U+00FF` and more characters than the count holds.
fn write_name(w: &mut FieldWriter, field: &str, name: &str, count_bits: usize) -> Result<()> {
    let mut bytes = Vec::with_capacity(name.len());
    for c in name.chars() {
        let byte = u8::try_from(u32::from(c)).map_err(|_| {
            Error::InvalidInput(format!(
                "RTCM {field} character {c:?} (U+{:04X}) is not an 8-bit character",
                u32::from(c)
            ))
        })?;
        bytes.push(byte);
    }
    w.u(
        format_args!("{field} character count"),
        bytes.len() as u64,
        count_bits,
    )?;
    for byte in bytes {
        w.u(field, u64::from(byte), 8)?;
    }
    Ok(())
}

impl HelmertTransformation {
    /// Decode a 1021 or 1022 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if !matches!(message_number, 1021 | 1022) {
            return Err(Error::Parse(format!(
                "message {message_number} is not a Helmert transformation message 1021/1022"
            ))
            .into());
        }
        let source_name = read_name(r, 5)?;
        let target_name = read_name(r, 5)?;
        let system_id = r.u(8)? as u8;
        let utilized_messages = r.u(10)? as u16;
        let plate_number = r.u(5)? as u8;
        let computation_indicator = r.u(4)? as u8;
        let height_indicator = r.u(2)? as u8;
        let validity_latitude = r.i(19)? as i32;
        let validity_longitude = r.i(20)? as i32;
        let validity_extension_latitude = r.u(14)? as u16;
        let validity_extension_longitude = r.u(14)? as u16;
        let dx = r.i(23)? as i32;
        let dy = r.i(23)? as i32;
        let dz = r.i(23)? as i32;
        let r1 = r.i(32)? as i32;
        let r2 = r.i(32)? as i32;
        let r3 = r.i(32)? as i32;
        let ds = r.i(25)? as i32;
        let rotation_point = if message_number == 1022 {
            Some(RotationPoint {
                x: r.i(35)?,
                y: r.i(35)?,
                z: r.i(35)?,
            })
        } else {
            None
        };
        Ok(Self {
            message_number,
            source_name,
            target_name,
            system_id,
            utilized_messages,
            plate_number,
            computation_indicator,
            height_indicator,
            validity_latitude,
            validity_longitude,
            validity_extension_latitude,
            validity_extension_longitude,
            dx,
            dy,
            dz,
            r1,
            r2,
            r3,
            ds,
            rotation_point,
            add_as: r.u(24)? as u32,
            add_bs: r.u(25)? as u32,
            add_at: r.u(24)? as u32,
            add_bt: r.u(25)? as u32,
            horizontal_quality: r.u(3)? as u8,
            vertical_quality: r.u(3)? as u8,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming what the message cannot state: a message
    /// number other than 1021 and 1022, a rotation point absent from a 1022 or
    /// present in a 1021, a name with a character above `U+00FF` or more than
    /// 31 characters, nonempty `trailing_bits`, or a value wider than its
    /// field.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        match (number, self.rotation_point.is_some()) {
            (1021, false) | (1022, true) => {}
            (1021, true) => {
                return Err(Error::InvalidInput(
                    "RTCM 1021 carries no rotation point; a rotation point is written as 1022"
                        .to_string(),
                ))
            }
            (1022, false) => {
                return Err(Error::InvalidInput(
                    "RTCM 1022 carries a rotation point, and none is given".to_string(),
                ))
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "RTCM message number {number} is not a Helmert transformation message \
                     1021/1022"
                )))
            }
        }
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        write_name(&mut w, "source name", &self.source_name, 5)?;
        write_name(&mut w, "target name", &self.target_name, 5)?;
        w.u("system identification number", u64::from(self.system_id), 8)?;
        w.u(
            "utilized transformation message indicator",
            u64::from(self.utilized_messages),
            10,
        )?;
        w.u("plate number", u64::from(self.plate_number), 5)?;
        w.u(
            "computation indicator",
            u64::from(self.computation_indicator),
            4,
        )?;
        w.u("height indicator", u64::from(self.height_indicator), 2)?;
        w.i("validity latitude", i64::from(self.validity_latitude), 19)?;
        w.i("validity longitude", i64::from(self.validity_longitude), 20)?;
        w.u(
            "validity extension latitude",
            u64::from(self.validity_extension_latitude),
            14,
        )?;
        w.u(
            "validity extension longitude",
            u64::from(self.validity_extension_longitude),
            14,
        )?;
        w.i("dX", i64::from(self.dx), 23)?;
        w.i("dY", i64::from(self.dy), 23)?;
        w.i("dZ", i64::from(self.dz), 23)?;
        w.i("R1", i64::from(self.r1), 32)?;
        w.i("R2", i64::from(self.r2), 32)?;
        w.i("R3", i64::from(self.r3), 32)?;
        w.i("dS", i64::from(self.ds), 25)?;
        if let Some(point) = self.rotation_point {
            w.i("XP", point.x, 35)?;
            w.i("YP", point.y, 35)?;
            w.i("ZP", point.z, 35)?;
        }
        w.u("add aS", u64::from(self.add_as), 24)?;
        w.u("add bS", u64::from(self.add_bs), 25)?;
        w.u("add aT", u64::from(self.add_at), 24)?;
        w.u("add bT", u64::from(self.add_bt), 25)?;
        w.u(
            "horizontal quality indicator",
            u64::from(self.horizontal_quality),
            3,
        )?;
        w.u(
            "vertical quality indicator",
            u64::from(self.vertical_quality),
            3,
        )?;
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

/// Number of grid points a residual grid message carries.
pub const RESIDUAL_GRID_POINTS: usize = 16;

/// One grid point of a residual grid message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridResidual {
    /// Latitude residual `δφi` (DF199, int9, scale 3e-5 arc seconds) in 1023;
    /// northing residual `δNi` (DF209, int9, scale 1 mm) in 1024.
    pub horizontal_1: i16,
    /// Longitude residual `δλi` (DF200, int9, scale 3e-5 arc seconds) in
    /// 1023; easting residual `δEi` (DF210, int9, scale 1 mm) in 1024.
    pub horizontal_2: i16,
    /// Height residual `δhi` (DF201 in 1023, DF211 in 1024, int9), scale 1 mm.
    pub height: i16,
}

/// A decoded residual grid message: ellipsoidal (1023) or plane (1024).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidualGrid {
    /// 1023 or 1024.
    pub message_number: u16,
    /// System identification number (DF147, 8 bits).
    pub system_id: u8,
    /// Horizontal shift indicator (DF190).
    pub horizontal_shift: bool,
    /// Vertical shift indicator (DF191).
    pub vertical_shift: bool,
    /// Origin of the grids: latitude `φ0` (DF192, int21, scale 0.5 arc
    /// seconds) in 1023; northing `N0` (DF202, int25, scale 1 cm) in 1024.
    pub origin_1: i32,
    /// Origin of the grids: longitude `λ0` (DF193, int22, scale 0.5 arc
    /// seconds) in 1023; easting `E0` (DF203, 26 bits unsigned, scale 1 cm)
    /// in 1024.
    pub origin_2: i32,
    /// Grid area extension north/south: `Δφ` (DF194, 12 bits, scale 0.5 arc
    /// seconds) in 1023; `ΔN` (DF204, 12 bits, scale 1 cm) in 1024.
    pub extension_1: u16,
    /// Grid area extension east/west: `Δλ` (DF195, 12 bits, scale 0.5 arc
    /// seconds) in 1023; `ΔE` (DF205, 12 bits, scale 1 cm) in 1024.
    pub extension_2: u16,
    /// Mean offset: `Δφ` (DF196, int8, scale 0.001 arc seconds) in 1023;
    /// `ΔN` (DF206, int10, scale 1 cm) in 1024.
    pub mean_offset_1: i16,
    /// Mean offset: `Δλ` (DF197, int8, scale 0.001 arc seconds) in 1023;
    /// `ΔE` (DF207, int10, scale 1 cm) in 1024.
    pub mean_offset_2: i16,
    /// Mean height offset (DF198 in 1023, DF208 in 1024, int15), scale 1 cm.
    pub mean_height_offset: i16,
    /// The residuals of the 16 grid points, in transmitted order.
    pub residuals: [GridResidual; RESIDUAL_GRID_POINTS],
    /// Horizontal interpolation method indicator (DF212, 2 bits).
    pub horizontal_interpolation: u8,
    /// Vertical interpolation method indicator (DF213, 2 bits).
    pub vertical_interpolation: u8,
    /// Horizontal grid quality indicator (DF216, 3 bits).
    pub horizontal_quality: u8,
    /// Vertical grid quality indicator (DF217, 3 bits).
    pub vertical_quality: u8,
    /// Modified Julian day number (DF051, 16 bits).
    pub mjd: u16,
    /// Every body bit after the last field kept for the round trip; see
    /// [`HelmertTransformation::trailing_bits`].
    pub trailing_bits: Vec<bool>,
}

/// Widths of the fields that differ between 1023 and 1024.
#[derive(Clone, Copy)]
struct GridLayout {
    origin_1: usize,
    origin_2: usize,
    origin_2_signed: bool,
    mean_offset: usize,
}

impl GridLayout {
    fn of(message_number: u16) -> Option<Self> {
        match message_number {
            1023 => Some(Self {
                origin_1: 21,
                origin_2: 22,
                origin_2_signed: true,
                mean_offset: 8,
            }),
            1024 => Some(Self {
                origin_1: 25,
                origin_2: 26,
                origin_2_signed: false,
                mean_offset: 10,
            }),
            _ => None,
        }
    }
}

impl ResidualGrid {
    /// Decode a 1023 or 1024 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        let layout = GridLayout::of(message_number).ok_or_else(|| {
            Error::Parse(format!(
                "message {message_number} is not a residual grid message 1023/1024"
            ))
        })?;
        let system_id = r.u(8)? as u8;
        let horizontal_shift = r.flag()?;
        let vertical_shift = r.flag()?;
        let origin_1 = r.i(layout.origin_1)? as i32;
        let origin_2 = if layout.origin_2_signed {
            r.i(layout.origin_2)? as i32
        } else {
            r.u(layout.origin_2)? as i32
        };
        let extension_1 = r.u(12)? as u16;
        let extension_2 = r.u(12)? as u16;
        let mean_offset_1 = r.i(layout.mean_offset)? as i16;
        let mean_offset_2 = r.i(layout.mean_offset)? as i16;
        let mean_height_offset = r.i(15)? as i16;
        let mut residuals = [GridResidual::default(); RESIDUAL_GRID_POINTS];
        for residual in &mut residuals {
            *residual = GridResidual {
                horizontal_1: r.i(9)? as i16,
                horizontal_2: r.i(9)? as i16,
                height: r.i(9)? as i16,
            };
        }
        Ok(Self {
            message_number,
            system_id,
            horizontal_shift,
            vertical_shift,
            origin_1,
            origin_2,
            extension_1,
            extension_2,
            mean_offset_1,
            mean_offset_2,
            mean_height_offset,
            residuals,
            horizontal_interpolation: r.u(2)? as u8,
            vertical_interpolation: r.u(2)? as u8,
            horizontal_quality: r.u(3)? as u8,
            vertical_quality: r.u(3)? as u8,
            mjd: r.u(16)? as u16,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming what the message cannot state: a message
    /// number other than 1023 and 1024, nonempty `trailing_bits`, or a value
    /// wider than its field (for 1024 a negative easting, which DF203 does not
    /// hold).
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number;
        let layout = GridLayout::of(number).ok_or_else(|| {
            Error::InvalidInput(format!(
                "RTCM message number {number} is not a residual grid message 1023/1024"
            ))
        })?;
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u("system identification number", u64::from(self.system_id), 8)?;
        w.flag(self.horizontal_shift);
        w.flag(self.vertical_shift);
        w.i("grid origin 1", i64::from(self.origin_1), layout.origin_1)?;
        if layout.origin_2_signed {
            w.i("grid origin 2", i64::from(self.origin_2), layout.origin_2)?;
        } else {
            let easting = u64::try_from(self.origin_2).map_err(|_| {
                Error::InvalidInput(format!(
                    "RTCM {number} grid origin 2 {} is negative; DF203 is unsigned",
                    self.origin_2
                ))
            })?;
            w.u("grid origin 2", easting, layout.origin_2)?;
        }
        w.u("grid extension 1", u64::from(self.extension_1), 12)?;
        w.u("grid extension 2", u64::from(self.extension_2), 12)?;
        w.i(
            "mean offset 1",
            i64::from(self.mean_offset_1),
            layout.mean_offset,
        )?;
        w.i(
            "mean offset 2",
            i64::from(self.mean_offset_2),
            layout.mean_offset,
        )?;
        w.i("mean height offset", i64::from(self.mean_height_offset), 15)?;
        for (index, residual) in self.residuals.iter().enumerate() {
            w.i(
                format_args!("grid point {index} residual 1"),
                i64::from(residual.horizontal_1),
                9,
            )?;
            w.i(
                format_args!("grid point {index} residual 2"),
                i64::from(residual.horizontal_2),
                9,
            )?;
            w.i(
                format_args!("grid point {index} height residual"),
                i64::from(residual.height),
                9,
            )?;
        }
        w.u(
            "horizontal interpolation indicator",
            u64::from(self.horizontal_interpolation),
            2,
        )?;
        w.u(
            "vertical interpolation indicator",
            u64::from(self.vertical_interpolation),
            2,
        )?;
        w.u(
            "horizontal grid quality indicator",
            u64::from(self.horizontal_quality),
            3,
        )?;
        w.u(
            "vertical grid quality indicator",
            u64::from(self.vertical_quality),
            3,
        )?;
        w.u("MJD", u64::from(self.mjd), 16)?;
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

/// The parameters of a projection message, by its message number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionParameters {
    /// 1025: projection types other than LCC2SP and OM.
    NaturalOrigin {
        /// Latitude of natural origin `LaNO` (DF171, int34), scale 1.1e-8
        /// arc seconds.
        latitude: i64,
        /// Longitude of natural origin `LoNO` (DF172, int35), scale 1.1e-8
        /// arc seconds.
        longitude: i64,
        /// Scale factor at natural origin, as transmitted (DF173, 30 bits).
        add_scale: u32,
        /// False easting `FE` (DF174, 36 bits), scale 1 mm.
        false_easting: u64,
        /// False northing `FN` (DF175, int35), scale 1 mm.
        false_northing: i64,
    },
    /// 1026: Lambert conic conformal with two standard parallels.
    LambertConicConformal {
        /// Latitude of false origin `LaFO` (DF176, int34).
        latitude: i64,
        /// Longitude of false origin `LoFO` (DF177, int35).
        longitude: i64,
        /// Latitude of standard parallel 1 `LaSP1` (DF178, int34).
        standard_parallel_1: i64,
        /// Latitude of standard parallel 2 `LaSP2` (DF179, int34).
        standard_parallel_2: i64,
        /// Easting at false origin `EFO` (DF180, 36 bits), scale 1 mm.
        false_easting: u64,
        /// Northing at false origin `NFO` (DF181, int35), scale 1 mm.
        false_northing: i64,
    },
    /// 1027: oblique Mercator.
    ObliqueMercator {
        /// Rectification flag (DF182).
        rectification: bool,
        /// Latitude of the projection centre `LaPC` (DF183, int34).
        latitude: i64,
        /// Longitude of the projection centre `LoPC` (DF184, int35).
        longitude: i64,
        /// Azimuth of the initial line `AzIL` (DF185, 35 bits).
        azimuth: u64,
        /// Angle from the rectified to the skew grid, as transmitted (DF186,
        /// int26).
        rectified_to_skew: i32,
        /// Scale factor on the initial line, as transmitted (DF187, 30 bits).
        add_scale: u32,
        /// Easting at the projection centre `EPC` (DF188, 36 bits), scale 1 mm.
        easting: u64,
        /// Northing at the projection centre `NPC` (DF189, int35), scale 1 mm.
        northing: i64,
    },
}

/// A decoded projection parameter message: 1025, 1026 or 1027.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// System identification number (DF147, 8 bits).
    pub system_id: u8,
    /// Projection type (DF170, 6 bits).
    pub projection_type: u8,
    /// The projection parameters; their variant names the message number.
    pub parameters: ProjectionParameters,
    /// Every body bit after the last field kept for the round trip; see
    /// [`HelmertTransformation::trailing_bits`].
    pub trailing_bits: Vec<bool>,
}

impl Projection {
    /// The message number the parameters name: 1025, 1026 or 1027.
    pub fn message_number(&self) -> u16 {
        match self.parameters {
            ProjectionParameters::NaturalOrigin { .. } => 1025,
            ProjectionParameters::LambertConicConformal { .. } => 1026,
            ProjectionParameters::ObliqueMercator { .. } => 1027,
        }
    }

    /// Decode a 1025, 1026 or 1027 body (without the transport frame) under
    /// [`RtcmPolicy::Strict`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        decode_body(body, &mut DecodeContext::new(RtcmPolicy::Strict), |r, _| {
            Self::read(r)
        })
        .map_err(Into::into)
    }

    pub(crate) fn read(r: &mut BitReader<'_>) -> DecodeResult<Self> {
        let message_number = r.u(12)? as u16;
        if !(1025..=1027).contains(&message_number) {
            return Err(Error::Parse(format!(
                "message {message_number} is not a projection message 1025-1027"
            ))
            .into());
        }
        let system_id = r.u(8)? as u8;
        let projection_type = r.u(6)? as u8;
        let parameters = match message_number {
            1025 => ProjectionParameters::NaturalOrigin {
                latitude: r.i(34)?,
                longitude: r.i(35)?,
                add_scale: r.u(30)? as u32,
                false_easting: r.u(36)?,
                false_northing: r.i(35)?,
            },
            1026 => ProjectionParameters::LambertConicConformal {
                latitude: r.i(34)?,
                longitude: r.i(35)?,
                standard_parallel_1: r.i(34)?,
                standard_parallel_2: r.i(34)?,
                false_easting: r.u(36)?,
                false_northing: r.i(35)?,
            },
            _ => ProjectionParameters::ObliqueMercator {
                rectification: r.flag()?,
                latitude: r.i(34)?,
                longitude: r.i(35)?,
                azimuth: r.u(35)?,
                rectified_to_skew: r.i(26)? as i32,
                add_scale: r.u(30)? as u32,
                easting: r.u(36)?,
                northing: r.i(35)?,
            },
        };
        Ok(Self {
            system_id,
            projection_type,
            parameters,
            trailing_bits: Vec::new(),
        })
    }

    /// Encode this body (without the transport frame).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the field when a value is wider than its
    /// field, or for nonempty `trailing_bits`.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_policy(RtcmPolicy::Strict)
            .map(|(body, _)| body)
    }

    /// Encode this body under `policy`. Under [`RtcmPolicy::Lenient`] nonempty
    /// `trailing_bits` are written after the last field and reported.
    pub fn encode_with_policy(&self, policy: RtcmPolicy) -> Result<(Vec<u8>, Vec<RtcmDeparture>)> {
        let number = self.message_number();
        let mut w = FieldWriter::new(number);
        w.u("message number", u64::from(number), 12)?;
        w.u("system identification number", u64::from(self.system_id), 8)?;
        w.u("projection type", u64::from(self.projection_type), 6)?;
        match self.parameters {
            ProjectionParameters::NaturalOrigin {
                latitude,
                longitude,
                add_scale,
                false_easting,
                false_northing,
            } => {
                w.i("LaNO", latitude, 34)?;
                w.i("LoNO", longitude, 35)?;
                w.u("add SNO", u64::from(add_scale), 30)?;
                w.u("FE", false_easting, 36)?;
                w.i("FN", false_northing, 35)?;
            }
            ProjectionParameters::LambertConicConformal {
                latitude,
                longitude,
                standard_parallel_1,
                standard_parallel_2,
                false_easting,
                false_northing,
            } => {
                w.i("LaFO", latitude, 34)?;
                w.i("LoFO", longitude, 35)?;
                w.i("LaSP1", standard_parallel_1, 34)?;
                w.i("LaSP2", standard_parallel_2, 34)?;
                w.u("EFO", false_easting, 36)?;
                w.i("NFO", false_northing, 35)?;
            }
            ProjectionParameters::ObliqueMercator {
                rectification,
                latitude,
                longitude,
                azimuth,
                rectified_to_skew,
                add_scale,
                easting,
                northing,
            } => {
                w.flag(rectification);
                w.i("LaPC", latitude, 34)?;
                w.i("LoPC", longitude, 35)?;
                w.u("AzIL", azimuth, 35)?;
                w.i("Diff ARSG", i64::from(rectified_to_skew), 26)?;
                w.u("add SIL", u64::from(add_scale), 30)?;
                w.u("EPC", easting, 36)?;
                w.i("NPC", northing, 35)?;
            }
        }
        let departures = write_trailing(&mut w, &self.trailing_bits, policy)?;
        Ok((w.into_bytes(), departures))
    }
}

impl super::TrailingBits for HelmertTransformation {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}

impl super::TrailingBits for ResidualGrid {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}

impl super::TrailingBits for Projection {
    fn trailing_bits_mut(&mut self) -> &mut Vec<bool> {
        &mut self.trailing_bits
    }
}
