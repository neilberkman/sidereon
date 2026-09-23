//! JPL SPK/DAF binary-kernel parsing.
//!
//! This module starts with the DAF container layer used by SPK kernels. It can
//! read segment descriptors from an in-memory byte slice without requiring any
//! file I/O.
//!
//! ```
//! use sidereon_core::astro::spk::Spk;
//!
//! const RECORD_BYTES: usize = 1024;
//! const START_ADDRESS: usize = 513;
//! const END_ADDRESS: usize = 524;
//!
//! fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
//!     bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
//! }
//!
//! fn put_f64(bytes: &mut [u8], address: usize, value: f64) {
//!     let offset = (address - 1) * 8;
//!     bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
//! }
//!
//! fn put_ascii(bytes: &mut [u8], offset: usize, len: usize, text: &str) {
//!     bytes[offset..offset + len].fill(b' ');
//!     bytes[offset..offset + text.len()].copy_from_slice(text.as_bytes());
//! }
//!
//! fn put_summary(bytes: &mut [u8], offset: usize) {
//!     bytes[offset..offset + 8].copy_from_slice(&0.0f64.to_le_bytes());
//!     bytes[offset + 8..offset + 16].copy_from_slice(&10.0f64.to_le_bytes());
//!     for (index, value) in [10, 0, 1, 3, START_ADDRESS as i32, END_ADDRESS as i32]
//!         .into_iter()
//!         .enumerate()
//!     {
//!         put_i32(bytes, offset + 16 + index * 4, value);
//!     }
//! }
//!
//! let mut bytes = vec![0u8; END_ADDRESS * 8];
//! bytes[0..8].copy_from_slice(b"DAF/SPK ");
//! put_i32(&mut bytes, 8, 2);
//! put_i32(&mut bytes, 12, 6);
//! put_ascii(&mut bytes, 16, 60, "DOC SPK");
//! put_i32(&mut bytes, 76, 3);
//! put_i32(&mut bytes, 80, 3);
//! put_i32(&mut bytes, 84, (END_ADDRESS + 1) as i32);
//! bytes[88..96].copy_from_slice(b"LTL-IEEE");
//!
//! let summary_record = RECORD_BYTES * 2;
//! bytes[summary_record..summary_record + 8].copy_from_slice(&0.0f64.to_le_bytes());
//! bytes[summary_record + 8..summary_record + 16].copy_from_slice(&0.0f64.to_le_bytes());
//! bytes[summary_record + 16..summary_record + 24].copy_from_slice(&1.0f64.to_le_bytes());
//! put_summary(&mut bytes, summary_record + 24);
//! put_ascii(&mut bytes, RECORD_BYTES * 3, 40, "BODY 10 TO SSB");
//!
//! put_f64(&mut bytes, START_ADDRESS, 5.0);
//! put_f64(&mut bytes, START_ADDRESS + 1, 5.0);
//! put_f64(&mut bytes, START_ADDRESS + 2, 100.0);
//! put_f64(&mut bytes, START_ADDRESS + 3, 10.0);
//! put_f64(&mut bytes, START_ADDRESS + 4, 1.0);
//! put_f64(&mut bytes, START_ADDRESS + 5, 1.0);
//! put_f64(&mut bytes, START_ADDRESS + 6, 0.0);
//! put_f64(&mut bytes, START_ADDRESS + 7, 0.1);
//! put_f64(&mut bytes, START_ADDRESS + 8, 0.0);
//! put_f64(&mut bytes, START_ADDRESS + 9, 10.0);
//! put_f64(&mut bytes, START_ADDRESS + 10, 8.0);
//! put_f64(&mut bytes, START_ADDRESS + 11, 1.0);
//!
//! let spk = Spk::from_bytes(&bytes)?;
//! let state = spk.spk_state(10, 0, 5.0)?;
//! assert_eq!(state.target, 10);
//! assert_eq!(state.center, 0);
//! assert_eq!(state.frame, 1);
//! assert_eq!(state.position_km, [100.0, 10.0, 1.0]);
//! assert_eq!(state.velocity_km_s, [1.0, 0.0, 0.1]);
//! # Ok::<(), sidereon_core::astro::spk::SpkError>(())
//! ```

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

const DAF_RECORD_BYTES: usize = 1024;
const DAF_ID_BYTES: usize = 8;
const DAF_INTERNAL_NAME_BYTES: usize = 60;
const DAF_BINARY_FORMAT_OFFSET: usize = 88;
const DAF_BINARY_FORMAT_BYTES: usize = 8;
const DAF_FILE_RECORD_BYTES: usize = DAF_RECORD_BYTES;
const SUMMARY_CONTROL_WORDS: usize = 3;
const SPK_ND: i32 = 2;
const SPK_NI: i32 = 6;
const SPK_TYPE_2: i32 = 2;
const SPK_TYPE_3: i32 = 3;
const SPK_TYPE_21: i32 = 21;
/// Largest difference-table dimension (`MAXDIM`) a type-21 record may declare.
/// Matches CSPICE `MAXTRM` from `spk21.inc`; records above this are rejected.
const SPK_TYPE_21_MAX_TABLE_DIM: usize = 25;
/// Number of record epochs covered by each directory epoch in a type-21 segment.
const SPK_TYPE_21_DIRECTORY_STRIDE: usize = 100;

/// Endianness declared by the DAF binary-format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DafByteOrder {
    /// Little-endian IEEE floating-point and integer words (`LTL-IEEE`).
    LittleEndian,
    /// Big-endian IEEE floating-point and integer words (`BIG-IEEE`).
    BigEndian,
}

impl DafByteOrder {
    fn read_i32(self, bytes: &[u8], offset: usize, field: &'static str) -> Result<i32, SpkError> {
        let data = bytes.get(offset..offset + 4).ok_or(SpkError::Truncated {
            context: field,
            needed: offset + 4,
            actual: bytes.len(),
        })?;
        let mut word = [0u8; 4];
        word.copy_from_slice(data);
        Ok(match self {
            DafByteOrder::LittleEndian => i32::from_le_bytes(word),
            DafByteOrder::BigEndian => i32::from_be_bytes(word),
        })
    }

    fn read_f64(self, bytes: &[u8], offset: usize, field: &'static str) -> Result<f64, SpkError> {
        let data = bytes.get(offset..offset + 8).ok_or(SpkError::Truncated {
            context: field,
            needed: offset + 8,
            actual: bytes.len(),
        })?;
        let mut word = [0u8; 8];
        word.copy_from_slice(data);
        Ok(match self {
            DafByteOrder::LittleEndian => f64::from_le_bytes(word),
            DafByteOrder::BigEndian => f64::from_be_bytes(word),
        })
    }
}

/// Parsed metadata from the first DAF record.
#[derive(Debug, Clone, PartialEq)]
pub struct DafFileRecord {
    /// The eight-byte DAF identification word, such as `DAF/SPK`.
    pub id_word: String,
    /// The DAF array type from the identification word, such as `SPK`.
    pub file_type: String,
    /// Number of double-precision components in each array summary.
    pub double_components: i32,
    /// Number of integer components in each array summary.
    pub integer_components: i32,
    /// DAF internal file name with trailing padding removed.
    pub internal_name: String,
    /// One-based record number of the first summary record.
    pub forward_record: i32,
    /// One-based record number of the final summary record.
    pub backward_record: i32,
    /// First free DAF address.
    pub free_address: i32,
    /// Declared byte order for numeric DAF words.
    pub byte_order: DafByteOrder,
    /// Raw eight-byte binary-format identifier with trailing padding removed.
    pub binary_format: String,
}

/// Descriptor for one SPK segment advertised by the DAF summary records.
#[derive(Debug, Clone, PartialEq)]
pub struct SpkSegmentDescriptor {
    /// Segment name from the paired DAF name record.
    pub name: String,
    /// Coverage start ET/TDB seconds past J2000.
    pub start_et: f64,
    /// Coverage stop ET/TDB seconds past J2000.
    pub stop_et: f64,
    /// NAIF target body identifier.
    pub target: i32,
    /// NAIF center body identifier.
    pub center: i32,
    /// NAIF reference-frame identifier.
    pub frame: i32,
    /// SPK segment data type.
    pub data_type: i32,
    /// One-based DAF address of the first segment data word.
    pub start_address: i32,
    /// One-based DAF address of the last segment data word.
    pub end_address: i32,
}

/// A parsed DAF/SPK directory containing the file record and segment list.
#[derive(Debug, Clone, PartialEq)]
pub struct DafSpk {
    /// File-level DAF metadata.
    pub file_record: DafFileRecord,
    /// SPK segment descriptors in summary-record order.
    pub segments: Vec<SpkSegmentDescriptor>,
}

/// In-memory SPK kernel with parsed segment descriptors.
#[derive(Debug, Clone, PartialEq)]
pub struct Spk {
    bytes: Vec<u8>,
    directory: DafSpk,
}

/// Position and velocity evaluated from an SPK segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpkStateVector {
    /// Position vector in kilometers.
    pub position_km: [f64; 3],
    /// Velocity vector in kilometers per second.
    pub velocity_km_s: [f64; 3],
}

/// State returned by an SPK body-to-center query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpkState {
    /// NAIF target body identifier for the returned relative state.
    pub target: i32,
    /// NAIF center body identifier for the returned relative state.
    pub center: i32,
    /// Position of the target relative to the requested center, in kilometers.
    pub position_km: [f64; 3],
    /// Velocity of the target relative to the requested center, in kilometers per second.
    ///
    /// Every supported segment type yields it. Types 3 and 21 store velocity;
    /// for type 2 it is the time derivative of the position Chebyshev
    /// expansion, as CSPICE `SPKE02` computes it.
    pub velocity_km_s: [f64; 3],
    /// NAIF reference-frame identifier of `position_km` and `velocity_km_s`:
    /// the frame requested with `spk_state_in_frame`, or for `spk_state` the
    /// frame of the first segment the query evaluated. A `target == center`
    /// query evaluates no segment and reports the requested frame, or 0.
    pub frame: i32,
}

/// Error returned while reading an SPK/DAF byte slice.
#[derive(Debug, Clone, PartialEq)]
pub enum SpkError {
    /// File loading failed while reading an SPK kernel from disk.
    Io {
        /// Path passed to [`Spk::load`].
        path: String,
        /// Display string from the underlying I/O error.
        message: String,
    },
    /// The input ended before a required DAF field could be read.
    Truncated {
        /// Name of the field or record being read.
        context: &'static str,
        /// Minimum number of bytes needed to read the field.
        needed: usize,
        /// Number of bytes available in the input.
        actual: usize,
    },
    /// The file record did not identify a DAF/SPK kernel.
    UnsupportedDafId {
        /// Identification word found in the file record.
        id_word: String,
    },
    /// The file record named a binary format this parser does not implement.
    UnsupportedBinaryFormat {
        /// Binary-format identifier found in the file record.
        binary_format: String,
    },
    /// The DAF summary shape was not the SPK shape `ND=2, NI=6`.
    UnsupportedSummaryShape {
        /// Number of double components declared by the file record.
        nd: i32,
        /// Number of integer components declared by the file record.
        ni: i32,
    },
    /// A numeric DAF field was present but outside the supported range.
    InvalidField {
        /// Name of the invalid DAF field.
        field: &'static str,
        /// Value decoded from the field.
        value: i32,
    },
    /// A floating-point SPK field was present but outside the supported range.
    InvalidDoubleField {
        /// Name of the invalid SPK field.
        field: &'static str,
        /// Value decoded from the field.
        value: f64,
    },
    /// The requested ET is not covered by the segment.
    OutOfCoverage {
        /// Requested ET/TDB seconds past J2000.
        et: f64,
        /// Segment coverage start ET/TDB seconds past J2000.
        start_et: f64,
        /// Segment coverage stop ET/TDB seconds past J2000.
        stop_et: f64,
    },
    /// The segment data type is not supported by the requested evaluator.
    UnsupportedSegmentType {
        /// SPK segment type expected by the evaluator.
        expected: i32,
        /// SPK segment type found in the descriptor.
        actual: i32,
    },
    /// The segment addresses or directory do not describe a valid SPK layout.
    InvalidSegmentLayout {
        /// Name of the malformed segment component.
        context: &'static str,
    },
    /// No segment in the kernel names the requested NAIF body id.
    UnknownBody {
        /// NAIF body identifier that was not present in any segment.
        body: i32,
    },
    /// The kernels name both bodies but no segments connect them at any time.
    NoSegmentPath {
        /// Requested target NAIF body identifier.
        target: i32,
        /// Requested center NAIF body identifier.
        center: i32,
    },
    /// Segments connect the two bodies, but the chains of highest-priority
    /// segments covering the requested ET do not meet (CSPICE reports this as
    /// insufficient ephemeris data).
    CoverageGap {
        /// Requested target NAIF body identifier.
        target: i32,
        /// Requested center NAIF body identifier.
        center: i32,
        /// Requested ET/TDB seconds past J2000.
        et: f64,
    },
    /// A public state query reached an SPK segment type it cannot evaluate.
    UnsupportedStateSegmentType {
        /// SPK segment type found in the descriptor.
        data_type: i32,
    },
    /// A query needed a rotation between two frames, at least one of which
    /// is not one of the NAIF built-in inertial frames 1-21 whose constant
    /// mutual rotations CSPICE `IRFROT` defines. CSPICE would call `FRMCHG`
    /// with frame kernels and orientation data, which this reader does not
    /// hold.
    NonInertialFrameRotation {
        /// Frame the state is expressed in.
        from: i32,
        /// Frame the state had to be rotated into.
        to: i32,
    },
}

impl fmt::Display for SpkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpkError::Io { path, message } => {
                write!(f, "failed to read SPK kernel {path}: {message}")
            }
            SpkError::Truncated {
                context,
                needed,
                actual,
            } => write!(
                f,
                "truncated SPK/DAF input while reading {context}: need {needed} bytes, have {actual}"
            ),
            SpkError::UnsupportedDafId { id_word } => {
                write!(f, "unsupported DAF identification word {id_word:?}")
            }
            SpkError::UnsupportedBinaryFormat { binary_format } => {
                write!(f, "unsupported DAF binary format {binary_format:?}")
            }
            SpkError::UnsupportedSummaryShape { nd, ni } => {
                write!(f, "unsupported SPK summary shape ND={nd}, NI={ni}")
            }
            SpkError::InvalidField { field, value } => {
                write!(f, "invalid SPK/DAF field {field}: {value}")
            }
            SpkError::InvalidDoubleField { field, value } => {
                write!(f, "invalid SPK field {field}: {value}")
            }
            SpkError::OutOfCoverage {
                et,
                start_et,
                stop_et,
            } => write!(
                f,
                "ET {et} is outside SPK segment coverage [{start_et}, {stop_et}]"
            ),
            SpkError::UnsupportedSegmentType { expected, actual } => write!(
                f,
                "unsupported SPK segment type {actual}; expected type {expected}"
            ),
            SpkError::InvalidSegmentLayout { context } => {
                write!(f, "invalid SPK segment layout: {context}")
            }
            SpkError::UnknownBody { body } => {
                write!(f, "unknown SPK body {body}")
            }
            SpkError::NoSegmentPath { target, center } => {
                write!(f, "no SPK segment path from target {target} to center {center}")
            }
            SpkError::CoverageGap { target, center, et } => write!(
                f,
                "no SPK segment path from target {target} to center {center} covers ET {et}"
            ),
            SpkError::UnsupportedStateSegmentType { data_type } => {
                write!(f, "unsupported SPK state segment type {data_type}")
            }
            SpkError::NonInertialFrameRotation { from, to } => write!(
                f,
                "no constant rotation from SPK frame {from} to frame {to}: only NAIF inertial frames 1-{NAIF_INERTIAL_FRAMES} are rotated"
            ),
        }
    }
}

impl std::error::Error for SpkError {}

impl Spk {
    /// Parse an in-memory SPK kernel from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SpkError> {
        Self::from_vec(bytes.to_vec())
    }

    /// Read and parse an SPK kernel from a filesystem path.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, SpkError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| SpkError::Io {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
        Self::from_vec(bytes)
    }

    /// Return parsed file-record metadata.
    pub fn file_record(&self) -> &DafFileRecord {
        &self.directory.file_record
    }

    /// Return parsed segment descriptors in DAF summary order.
    pub fn segments(&self) -> &[SpkSegmentDescriptor] {
        &self.directory.segments
    }

    /// Query the state of `target` relative to `center` at ET/TDB seconds past J2000.
    ///
    /// This is the one-kernel case of [`SpkKernels::spk_state`]; segment
    /// selection and chain composition follow CSPICE `SPKSFS` and `SPKGEO`
    /// exactly, as described there.
    pub fn spk_state(&self, target: i32, center: i32, et: f64) -> Result<SpkState, SpkError> {
        resolve_spk_state(core::slice::from_ref(self), target, center, et, None)
    }

    /// Query the state of `target` relative to `center` in the NAIF frame
    /// `frame`. This is the one-kernel case of [`SpkKernels::spk_state_in_frame`].
    pub fn spk_state_in_frame(
        &self,
        target: i32,
        center: i32,
        et: f64,
        frame: i32,
    ) -> Result<SpkState, SpkError> {
        resolve_spk_state(core::slice::from_ref(self), target, center, et, Some(frame))
    }

    fn from_vec(bytes: Vec<u8>) -> Result<Self, SpkError> {
        let directory = parse_daf_spk(&bytes)?;
        Ok(Self { bytes, directory })
    }
}

/// SPK kernels in load order, queried with the NAIF segment-priority rules.
///
/// The SPK Required Reading gives the precedence: when segments for the same
/// body overlap in time, a segment in a later-loaded file takes precedence over
/// every segment in an earlier-loaded file, and within one file a segment later
/// in the file takes precedence over an earlier one. [`SpkKernels::spk_state`]
/// applies that order across every kernel held here, as CSPICE does across the
/// files loaded with `FURNSH`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpkKernels {
    kernels: Vec<Spk>,
}

impl SpkKernels {
    /// Create an empty kernel set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parsed kernel. It takes precedence over every kernel added before it.
    pub fn push(&mut self, kernel: Spk) {
        self.kernels.push(kernel);
    }

    /// Parse and add a kernel from bytes. It takes precedence over every kernel
    /// added before it.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), SpkError> {
        self.kernels.push(Spk::from_bytes(bytes)?);
        Ok(())
    }

    /// Read, parse and add a kernel file. It takes precedence over every kernel
    /// added before it.
    pub fn load(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), SpkError> {
        self.kernels.push(Spk::load(path)?);
        Ok(())
    }

    /// Return the kernels in load order (lowest precedence first).
    pub fn kernels(&self) -> &[Spk] {
        &self.kernels
    }

    /// Number of kernels held.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether no kernel is held.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Query the state of `target` relative to `center` at ET/TDB seconds past J2000.
    ///
    /// Segment selection follows CSPICE `SPKSFS`: for a body `B`, the one
    /// segment used is the highest-priority segment whose target is `B` and
    /// whose descriptor covers `et` inclusively, searching the last-loaded
    /// kernel first and, within a kernel, the last segment first. A segment is
    /// never used in the reverse direction, and a lower-priority segment is
    /// never substituted when the selected one cannot be evaluated.
    ///
    /// Chain composition follows CSPICE `SPKGEO`. The target chain starts at
    /// `target` and follows selected segments to their centers, keeping each
    /// leg in its own segment's frame, until it reaches `center`, the solar
    /// system barycenter (0), or a body with no covering segment. The chain
    /// holds at most 20 bodies; once it is full, every further leg is folded
    /// into the last entry, whose running sum is rotated into the new leg's
    /// frame before the leg is added.
    ///
    /// The observer chain starts at `center` and follows selected segments
    /// until it reaches a body in the target chain, or ends at the barycenter
    /// or at a body with no covering segment; if it has not met the target
    /// chain by then, the bodies are not joined at `et`. Its legs are summed
    /// as it goes: the running sum is rotated into the frame of each new leg
    /// whose frame differs, then the leg is added.
    ///
    /// The target-chain legs up to the common body are then summed the same
    /// way, leg by leg in chain order, each running sum rotated into the next
    /// leg's frame when the frames differ. The state is that sum minus the
    /// observer-chain sum, rotated into one frame first when their frames
    /// differ, and finally rotated into the output frame.
    ///
    /// `target == center` returns the zero state without consulting any
    /// segment, as `SPKGEO` does; its `frame` is 0.
    ///
    /// Every rotation is the constant `IRFROT` rotation between NAIF inertial
    /// frames (see [`inertial_frame_rotation`]). CSPICE takes the
    /// output frame from its caller; this method returns the state in the frame
    /// of the first segment evaluated (the target's own segment when it has
    /// one, otherwise the observer's first segment), so a kernel whose
    /// segments share one frame is returned in that frame without any
    /// rotation. Use [`SpkKernels::spk_state_in_frame`] to name the frame. A
    /// rotation that involves a frame outside NAIF inertial frames 1-21 returns
    /// [`SpkError::NonInertialFrameRotation`]; legs that share a frame are
    /// added without rotation whatever the frame.
    pub fn spk_state(&self, target: i32, center: i32, et: f64) -> Result<SpkState, SpkError> {
        resolve_spk_state(&self.kernels, target, center, et, None)
    }

    /// Query the state of `target` relative to `center` in the NAIF frame
    /// `frame`, as CSPICE `SPKGEO` does with the reference frame `REF`: the
    /// composed state is rotated into `frame` with the constant `IRFROT`
    /// rotation when it is expressed in another frame. Segment selection and
    /// chain composition are those of [`SpkKernels::spk_state`].
    pub fn spk_state_in_frame(
        &self,
        target: i32,
        center: i32,
        et: f64,
        frame: i32,
    ) -> Result<SpkState, SpkError> {
        resolve_spk_state(&self.kernels, target, center, et, Some(frame))
    }
}

impl FromIterator<Spk> for SpkKernels {
    fn from_iter<I: IntoIterator<Item = Spk>>(iter: I) -> Self {
        Self {
            kernels: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<Spk>> for SpkKernels {
    fn from(kernels: Vec<Spk>) -> Self {
        Self { kernels }
    }
}

/// Target-chain length `CHLEN` of CSPICE `SPKGEO`: the chain holds at most
/// this many bodies, and further legs are folded into its last entry.
const SPK_CHAIN_LENGTH: usize = 20;
/// NAIF id of the solar system barycenter, where a chain ends.
const NAIF_SSB: i32 = 0;

/// One segment of one kernel.
#[derive(Clone, Copy)]
struct SegmentRef<'a> {
    kernel: &'a Spk,
    segment: &'a SpkSegmentDescriptor,
}

/// Every segment in precedence order: last-loaded kernel first, and within a
/// kernel the last segment first.
fn segments_by_priority(kernels: &[Spk]) -> impl Iterator<Item = SegmentRef<'_>> {
    kernels.iter().rev().flat_map(|kernel| {
        kernel
            .directory
            .segments
            .iter()
            .rev()
            .map(move |segment| SegmentRef { kernel, segment })
    })
}

/// CSPICE `SPKSFS`: the highest-priority segment for `body` covering `et`.
fn select_segment(kernels: &[Spk], body: i32, et: f64) -> Option<SegmentRef<'_>> {
    segments_by_priority(kernels).find(|candidate| {
        candidate.segment.target == body
            && et >= candidate.segment.start_et
            && et <= candidate.segment.stop_et
    })
}

/// A state vector in `SPKGEO` layout: position then velocity.
type State6 = [f64; 6];

/// One selected and evaluated segment: CSPICE `SPKPVN` output.
struct EvaluatedLeg {
    state: State6,
    frame: i32,
    center: i32,
}

fn evaluate_leg(leg: SegmentRef<'_>, et: f64) -> Result<EvaluatedLeg, SpkError> {
    let state = evaluate_segment_state(leg.kernel, leg.segment, et)?;
    Ok(EvaluatedLeg {
        state: [
            state.position_km[0],
            state.position_km[1],
            state.position_km[2],
            state.velocity_km_s[0],
            state.velocity_km_s[1],
            state.velocity_km_s[2],
        ],
        frame: leg.segment.frame,
        center: leg.segment.center,
    })
}

/// `SPKGEO`'s `ISINRT`: both frames are among the built-in inertial frames.
fn both_inertial(a: i32, b: i32) -> bool {
    is_inertial_frame(a) && is_inertial_frame(b)
}

fn is_inertial_frame(frame: i32) -> bool {
    frame > 0 && frame <= NAIF_INERTIAL_FRAMES as i32
}

/// Rotate a state from frame `from` into frame `to` with `IRFROT` and two
/// `MXV` products, as `SPKGEO` does, refusing a non-inertial pair.
fn rotate_state(state: &State6, from: i32, to: i32) -> Result<State6, SpkError> {
    if !both_inertial(from, to) {
        return Err(SpkError::NonInertialFrameRotation { from, to });
    }
    let rotation = irfrot(from, to);
    let position = mxv(&rotation, [state[0], state[1], state[2]]);
    let velocity = mxv(&rotation, [state[3], state[4], state[5]]);
    Ok([
        position[0],
        position[1],
        position[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ])
}

fn add6(lhs: &State6, rhs: &State6) -> State6 {
    let mut out = [0.0; 6];
    for (index, value) in out.iter_mut().enumerate() {
        *value = lhs[index] + rhs[index];
    }
    out
}

fn sub6(lhs: &State6, rhs: &State6) -> State6 {
    let mut out = [0.0; 6];
    for (index, value) in out.iter_mut().enumerate() {
        *value = lhs[index] - rhs[index];
    }
    out
}

/// Total segment count: a chain longer than this has revisited a body.
fn segment_count(kernels: &[Spk]) -> usize {
    kernels
        .iter()
        .map(|kernel| kernel.directory.segments.len())
        .sum()
}

/// CSPICE `SPKGEO`, statement for statement, without allocating. `refid` is
/// the requested output frame; `None` takes the frame of the first segment
/// evaluated.
///
/// `SPKGEO` loops without end when the chain of selected segments revisits a
/// body before reaching the observer or the barycenter; such a chain is
/// reported here as the error for a chain that does not join.
fn resolve_spk_state(
    kernels: &[Spk],
    target: i32,
    center: i32,
    et: f64,
    refid: Option<i32>,
) -> Result<SpkState, SpkError> {
    if !et.is_finite() {
        return Err(SpkError::InvalidDoubleField {
            field: "ET",
            value: et,
        });
    }
    if target == center {
        return Ok(SpkState {
            target,
            center,
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
            frame: refid.unwrap_or(0),
        });
    }
    let unjoined = || missing_chain_error(kernels, target, center, et);
    let max_legs = segment_count(kernels);

    // Target chain. Index k holds CTARG(k + 1), TFRAME(k + 1) and STARG(k + 1):
    // STARG(1) is zero and STARG(k + 1), k >= 1, is the state of CTARG(k)
    // relative to CTARG(k + 1) in TFRAME(k + 1).
    let mut ctarg = [0i32; SPK_CHAIN_LENGTH];
    let mut tframe = [0i32; SPK_CHAIN_LENGTH];
    let mut starg = [[0.0f64; 6]; SPK_CHAIN_LENGTH];
    ctarg[0] = target;
    let mut count = 1usize;
    while count < SPK_CHAIN_LENGTH && ctarg[count - 1] != center && ctarg[count - 1] != NAIF_SSB {
        let Some(segment) = select_segment(kernels, ctarg[count - 1], et) else {
            break;
        };
        let leg = evaluate_leg(segment, et)?;
        if ctarg[..count].contains(&leg.center) {
            return Err(unjoined());
        }
        ctarg[count] = leg.center;
        tframe[count] = leg.frame;
        starg[count] = leg.state;
        count += 1;
    }
    if count > 1 {
        tframe[0] = tframe[1];
    }
    let default_frame = (count > 1).then_some(tframe[1]);

    if count == SPK_CHAIN_LENGTH {
        // Fold every further leg into the last entry.
        let last = SPK_CHAIN_LENGTH - 1;
        let mut folded = 0usize;
        while ctarg[last] != NAIF_SSB && ctarg[last] != center {
            let Some(segment) = select_segment(kernels, ctarg[last], et) else {
                break;
            };
            let leg = evaluate_leg(segment, et)?;
            let vtemp = if tframe[last] == leg.frame {
                starg[last]
            } else {
                rotate_state(&starg[last], tframe[last], leg.frame)?
            };
            starg[last] = add6(&vtemp, &leg.state);
            tframe[last] = leg.frame;
            ctarg[last] = leg.center;
            folded += 1;
            if folded > max_legs {
                return Err(unjoined());
            }
        }
    }
    let nct = count;

    // Observer chain.
    let mut cobs = center;
    let mut sobs = [0.0f64; 6];
    let mut ctpos = (ctarg[nct - 1] == cobs).then_some(nct - 1);
    let mut cframe = tframe[nct - 1];
    let mut observer_frame: Option<i32> = None;
    let mut legs = 0usize;
    while ctpos.is_none() && cobs != NAIF_SSB {
        let Some(segment) = select_segment(kernels, cobs, et) else {
            break;
        };
        let leg = evaluate_leg(segment, et)?;
        if legs == 0 {
            sobs = leg.state;
        }
        if observer_frame.is_none() {
            observer_frame = Some(leg.frame);
            cframe = leg.frame;
        }
        if cframe == leg.frame {
            if legs > 0 {
                sobs = add6(&sobs, &leg.state);
            }
        } else {
            let vtemp = rotate_state(&sobs, cframe, leg.frame)?;
            sobs = add6(&vtemp, &leg.state);
            cframe = leg.frame;
        }
        legs += 1;
        cobs = leg.center;
        ctpos = ctarg[..nct].iter().position(|&body| body == cobs);
        if legs > max_legs {
            return Err(unjoined());
        }
    }
    let Some(ctpos) = ctpos else {
        return Err(unjoined());
    };
    let refid = refid.or(default_frame).or(observer_frame).unwrap_or(cframe);

    if ctpos == 0 {
        tframe[0] = cframe;
    }
    for index in 1..ctpos {
        let next = if tframe[index] == tframe[index + 1] {
            add6(&starg[index], &starg[index + 1])
        } else {
            let stemp = rotate_state(&starg[index], tframe[index], tframe[index + 1])?;
            add6(&stemp, &starg[index + 1])
        };
        starg[index + 1] = next;
    }

    let mut state = if tframe[ctpos] == cframe {
        sub6(&starg[ctpos], &sobs)
    } else if tframe[ctpos] == refid {
        let stemp = rotate_state(&sobs, cframe, refid)?;
        cframe = refid;
        sub6(&starg[ctpos], &stemp)
    } else {
        let stemp = rotate_state(&starg[ctpos], tframe[ctpos], cframe)?;
        sub6(&stemp, &sobs)
    };
    if cframe != refid {
        state = rotate_state(&state, cframe, refid)?;
    }

    Ok(SpkState {
        target,
        center,
        position_km: [state[0], state[1], state[2]],
        velocity_km_s: [state[3], state[4], state[5]],
        frame: refid,
    })
}

/// Name the reason no chain joined `target` and `center` at `et`.
fn missing_chain_error(kernels: &[Spk], target: i32, center: i32, et: f64) -> SpkError {
    if !body_is_known(kernels, target) {
        return SpkError::UnknownBody { body: target };
    }
    if !body_is_known(kernels, center) {
        return SpkError::UnknownBody { body: center };
    }
    if !has_segment_path(kernels, target, center) {
        return SpkError::NoSegmentPath { target, center };
    }
    SpkError::CoverageGap { target, center, et }
}

fn body_is_known(kernels: &[Spk], body: i32) -> bool {
    segments_by_priority(kernels)
        .any(|candidate| candidate.segment.target == body || candidate.segment.center == body)
}

/// Whether any segments, at any time and in either direction, connect the two bodies.
fn has_segment_path(kernels: &[Spk], target: i32, center: i32) -> bool {
    let mut visited = vec![target];
    let mut cursor = 0;
    while cursor < visited.len() {
        let body = visited[cursor];
        cursor += 1;
        if body == center {
            return true;
        }
        for candidate in segments_by_priority(kernels) {
            let next = if candidate.segment.target == body {
                candidate.segment.center
            } else if candidate.segment.center == body {
                candidate.segment.target
            } else {
                continue;
            };
            if !visited.contains(&next) {
                visited.push(next);
            }
        }
    }
    false
}

/// Number of NAIF built-in inertial frames (CSPICE `NINERT`), ids 1 to 21.
pub const NAIF_INERTIAL_FRAMES: usize = 21;

/// CSPICE `CHGIRF` frame table: name, base frame, and the rotation from the
/// base frame as `angle axis` pairs, angles in arcseconds, applied right to
/// left. The strings are copied from `chgirf.f` (SPICELIB 4.5.0) and read
/// with the same parser, so each angle has the value CSPICE computes.
const IRF_DEFINITIONS: [(&str, &str, &str); NAIF_INERTIAL_FRAMES] = [
    ("J2000", "J2000", "0.0  1"),
    (
        "B1950",
        "J2000",
        "1152.84248596724 3  -1002.26108439117  2  1153.04066200330  3",
    ),
    ("FK4", "B1950", "0.525  3"),
    ("DE-118", "B1950", "0.53155  3"),
    ("DE-96", "B1950", "0.4107  3"),
    ("DE-102", "B1950", "0.1359  3"),
    ("DE-108", "B1950", "0.4775  3"),
    ("DE-111", "B1950", "0.5880  3"),
    ("DE-114", "B1950", "0.5529  3"),
    ("DE-122", "B1950", "0.5316  3"),
    ("DE-125", "B1950", "0.5754  3"),
    ("DE-130", "B1950", "0.5247  3"),
    ("GALACTIC", "FK4", "1177200.0  3  225360.0  1  1016100.0  3"),
    ("DE-200", "J2000", "0.0  3"),
    ("DE-202", "J2000", "0.0  3"),
    (
        "MARSIAU",
        "J2000",
        "324000.0D0 3 133610.4D0 2 -152348.4D0 3",
    ),
    ("ECLIPJ2000", "J2000", "84381.448 1"),
    ("ECLIPB1950", "B1950", "84404.836 1"),
    (
        "DE-140",
        "J2000",
        "1152.71013777252 3  -1002.25042010533  2  1153.75719544491  3",
    ),
    (
        "DE-142",
        "J2000",
        "1152.72061453864 3  -1002.25052830351  2  1153.74663857521  3",
    ),
    (
        "DE-143",
        "J2000",
        "1153.03919093833, 3, -1002.24822382286, 2, 1153.42900222357, 3",
    ),
];

/// 3x3 matrix, `m[i][j]` is the Fortran element `M(i + 1, j + 1)`.
type Mat3 = [[f64; 3]; 3];

/// Name of NAIF built-in inertial frame `frame` (1-21), as `IRFNAM` gives it.
pub fn inertial_frame_name(frame: i32) -> Option<&'static str> {
    is_inertial_frame(frame).then(|| IRF_DEFINITIONS[(frame - 1) as usize].0)
}

/// CSPICE `IRFROT`: the constant rotation taking a vector expressed in NAIF
/// inertial frame `from` to the same vector expressed in frame `to`, both
/// within 1-21. The matrix is built as `CHGIRF` builds it, so each element
/// has the value CSPICE computes with the same `sin`/`cos`.
pub fn inertial_frame_rotation(from: i32, to: i32) -> Result<[[f64; 3]; 3], SpkError> {
    if !both_inertial(from, to) {
        return Err(SpkError::NonInertialFrameRotation { from, to });
    }
    Ok(irfrot(from, to))
}

fn irfrot(from: i32, to: i32) -> Mat3 {
    if from == to {
        return rotate(0.0, 1);
    }
    let trans = irf_root_rotations();
    mxmt(&trans[(to - 1) as usize], &trans[(from - 1) as usize])
}

/// `CHGIRF`'s `TRANS`: the rotation from J2000 to each frame.
fn irf_root_rotations() -> &'static [Mat3; NAIF_INERTIAL_FRAMES] {
    static TRANS: std::sync::OnceLock<[Mat3; NAIF_INERTIAL_FRAMES]> = std::sync::OnceLock::new();
    TRANS.get_or_init(|| {
        let mut trans = [[[0.0; 3]; 3]; NAIF_INERTIAL_FRAMES];
        for (index, &(_, base, definition)) in IRF_DEFINITIONS.iter().enumerate() {
            let mut matrix = rotate(0.0, 1);
            let words: Vec<&str> = definition.split_whitespace().collect();
            let mut word = words.len();
            while word >= 2 {
                let axis = nparsd(words[word - 1]) as i32;
                let angle = nparsd(words[word - 2]);
                matrix = rotmat(&matrix, arcseconds_to_radians(angle), axis);
                word -= 2;
            }
            let base_index = IRF_DEFINITIONS[..=index]
                .iter()
                .position(|(name, _, _)| *name == base)
                .expect("each CHGIRF base frame precedes the frames defined on it");
            let base_matrix = if base_index == index {
                matrix
            } else {
                trans[base_index]
            };
            trans[index] = mxm(&matrix, &base_matrix);
        }
        trans
    })
}

/// `CONVRT(ANGLE, 'ARCSECONDS', 'RADIANS')`: `(x * (1/3600)) / DPR()`.
fn arcseconds_to_radians(arcseconds: f64) -> f64 {
    const DEGREES_PER_ARCSECOND: f64 = 1.0 / 3600.0;
    let degrees_per_radian = 180.0 / libm::acos(-1.0);
    (arcseconds * DEGREES_PER_ARCSECOND) / degrees_per_radian
}

/// SPICELIB `ROTATE`.
fn rotate(angle: f64, axis: i32) -> Mat3 {
    let s = libm::sin(angle);
    let c = libm::cos(angle);
    let (i1, i2, i3) = rotation_indices(axis);
    let mut m = [[0.0; 3]; 3];
    m[i1][i1] = 1.0;
    m[i2][i1] = 0.0;
    m[i3][i1] = 0.0;
    m[i1][i2] = 0.0;
    m[i2][i2] = c;
    m[i3][i2] = -s;
    m[i1][i3] = 0.0;
    m[i2][i3] = s;
    m[i3][i3] = c;
    m
}

/// SPICELIB `ROTMAT`: `[angle]_axis * m1`.
#[allow(clippy::needless_range_loop)] // transcribes the SPICELIB index loops
fn rotmat(m1: &Mat3, angle: f64, axis: i32) -> Mat3 {
    let s = libm::sin(angle);
    let c = libm::cos(angle);
    let (i1, i2, i3) = rotation_indices(axis);
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        out[i1][i] = m1[i1][i];
        out[i2][i] = c * m1[i2][i] + s * m1[i3][i];
        out[i3][i] = -(s * m1[i2][i]) + c * m1[i3][i];
    }
    out
}

/// `ROTATE`/`ROTMAT` index selection: `INDEXS = (3, 1, 2, 3, 1)`, zero-based.
fn rotation_indices(axis: i32) -> (usize, usize, usize) {
    const INDEXS: [usize; 5] = [2, 0, 1, 2, 0];
    let temp = axis.rem_euclid(3) as usize;
    (INDEXS[temp], INDEXS[temp + 1], INDEXS[temp + 2])
}

/// SPICELIB `MXM`: `m1 * m2`.
#[allow(clippy::needless_range_loop)] // transcribes the SPICELIB index loops
fn mxm(m1: &Mat3, m2: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m1[i][0] * m2[0][j] + m1[i][1] * m2[1][j] + m1[i][2] * m2[2][j];
        }
    }
    out
}

/// SPICELIB `MXMT`: `m1 * transpose(m2)`.
#[allow(clippy::needless_range_loop)] // transcribes the SPICELIB index loops
fn mxmt(m1: &Mat3, m2: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m1[i][0] * m2[j][0] + m1[i][1] * m2[j][1] + m1[i][2] * m2[j][2];
        }
    }
    out
}

/// SPICELIB `MXV`.
fn mxv(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, value) in out.iter_mut().enumerate() {
        *value = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
    }
    out
}

/// SPICELIB `NPARSD` for the numerals in [`IRF_DEFINITIONS`]: an optional
/// sign, digits with at most one decimal point, and an optional `D`/`E`
/// exponent, with blanks and commas ignored. The value is accumulated as
/// `NPARSD` accumulates it (integer part, then `DECVAL / DIVISR`, then the
/// power of ten), which is not always the correctly rounded decimal.
fn nparsd(text: &str) -> f64 {
    const BASE: f64 = 10.0;
    const LOOKUP: [f64; 11] = [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0,
    ];
    // NPARSD's INTBND: the largest power of ten below which adding one is
    // exact, divided by ten.
    let mut intbnd = BASE;
    let mut next = intbnd + 1.0;
    while intbnd != next {
        intbnd *= BASE;
        next = intbnd + 1.0;
    }
    intbnd /= BASE;

    let mut intval = 0.0;
    let mut decval = 0.0;
    let mut expval = 0.0;
    let mut divisr = 1.0;
    let mut factor = 1.0;
    let mut ecount = 0.0;
    let mut dpsign = [1.0, 1.0];
    let mut signdx = 0usize;
    let mut doint = true;
    let mut dodec = false;
    let mut doexp = false;
    let mut zeroi = false;
    let mut roundi = true;
    let mut roundd = true;
    for ch in text.chars() {
        match ch {
            '0'..='9' => {
                let digit = f64::from(ch as u8 - b'0');
                if doint {
                    if intval < intbnd {
                        intval = intval * BASE + digit;
                    } else {
                        ecount += 1.0;
                        factor /= BASE;
                        if roundi {
                            roundi = false;
                            if digit > 0.5 * BASE {
                                intval += 1.0;
                            }
                        }
                    }
                } else if dodec {
                    if zeroi {
                        if decval < intbnd {
                            decval = decval * BASE + digit;
                            ecount -= 1.0;
                        } else if roundd {
                            roundd = false;
                            if digit >= 0.5 * BASE {
                                decval += 1.0;
                            }
                        }
                    } else if divisr < intbnd {
                        decval = decval * BASE + digit;
                        divisr *= BASE;
                    }
                } else if doexp {
                    expval = expval * BASE + dpsign[1] * digit;
                }
            }
            '.' => {
                doint = false;
                dodec = true;
                zeroi = intval == 0.0;
            }
            'D' | 'd' | 'E' | 'e' => {
                doint = false;
                dodec = false;
                doexp = true;
                signdx = 1;
            }
            '+' => dpsign[signdx] = 1.0,
            '-' => dpsign[signdx] = -1.0,
            _ => {}
        }
    }

    let mut value = intval + (decval / divisr) * factor;
    expval += ecount;
    if expval < 0.0 {
        while expval < -BASE {
            value /= LOOKUP[10];
            expval += BASE;
        }
        value /= LOOKUP[(-expval) as usize];
    } else if expval > 0.0 {
        while expval > BASE {
            value *= LOOKUP[10];
            expval -= BASE;
        }
        value *= LOOKUP[expval as usize];
    }
    dpsign[0] * value
}

fn evaluate_segment_state(
    kernel: &Spk,
    segment: &SpkSegmentDescriptor,
    et: f64,
) -> Result<SpkState, SpkError> {
    let byte_order = kernel.directory.file_record.byte_order;
    let state = match segment.data_type {
        SPK_TYPE_2 => evaluate_type2_state(&kernel.bytes, byte_order, segment, et)?,
        SPK_TYPE_3 => evaluate_type3_state(&kernel.bytes, byte_order, segment, et)?,
        SPK_TYPE_21 => evaluate_type21_state(&kernel.bytes, byte_order, segment, et)?,
        data_type => return Err(SpkError::UnsupportedStateSegmentType { data_type }),
    };
    Ok(SpkState {
        target: segment.target,
        center: segment.center,
        position_km: state.position_km,
        velocity_km_s: state.velocity_km_s,
        frame: segment.frame,
    })
}

/// Query target state relative to center from a loaded SPK kernel.
pub fn spk_state(spk: &Spk, target: i32, center: i32, et: f64) -> Result<SpkState, SpkError> {
    spk.spk_state(target, center, et)
}

/// Parse the DAF/SPK directory from an in-memory kernel byte slice.
pub fn parse_daf_spk(bytes: &[u8]) -> Result<DafSpk, SpkError> {
    let file_record = parse_file_record(bytes)?;
    if file_record.double_components != SPK_ND || file_record.integer_components != SPK_NI {
        return Err(SpkError::UnsupportedSummaryShape {
            nd: file_record.double_components,
            ni: file_record.integer_components,
        });
    }

    let mut segments = Vec::new();
    let summary_words = summary_word_count(
        file_record.double_components,
        file_record.integer_components,
    )?;
    let summary_bytes = summary_words * 8;
    let name_bytes = summary_words * 8;

    let mut record = file_record.forward_record;
    validate_summary_record_pointer(record, "FWARD")?;
    let mut visited_records = Vec::new();
    while record != 0 {
        if visited_records.contains(&record) {
            return Err(SpkError::InvalidField {
                field: "summary record chain",
                value: record,
            });
        }
        visited_records.push(record);

        let summary_offset = record_offset(record, bytes.len(), "summary record")?;
        let name_record = paired_name_record(record, "name record")?;
        let name_offset = record_offset(name_record, bytes.len(), "name record")?;

        let next = read_summary_control_i32(
            file_record.byte_order,
            bytes,
            summary_offset,
            0,
            "next summary record",
        )?;
        validate_summary_record_pointer(next, "next summary record")?;
        let count = read_summary_control_i32(
            file_record.byte_order,
            bytes,
            summary_offset,
            2,
            "summary count",
        )?;
        if count < 0 {
            return Err(SpkError::InvalidField {
                field: "summary count",
                value: count,
            });
        }

        let count = usize::try_from(count).map_err(|_| SpkError::InvalidField {
            field: "summary count",
            value: count,
        })?;
        let summaries_start = summary_offset + SUMMARY_CONTROL_WORDS * 8;
        let summaries_end = summaries_start + count * summary_bytes;
        if summaries_end > summary_offset + DAF_RECORD_BYTES {
            return Err(SpkError::Truncated {
                context: "summary record entries",
                needed: summaries_end,
                actual: summary_offset + DAF_RECORD_BYTES,
            });
        }
        let names_end = name_offset + count * name_bytes;
        if names_end > name_offset + DAF_RECORD_BYTES {
            return Err(SpkError::Truncated {
                context: "name record entries",
                needed: names_end,
                actual: name_offset + DAF_RECORD_BYTES,
            });
        }

        for index in 0..count {
            let entry_offset = summaries_start + index * summary_bytes;
            let name_start = name_offset + index * name_bytes;
            let name = trim_ascii(&bytes[name_start..name_start + name_bytes]);

            let start_et =
                file_record
                    .byte_order
                    .read_f64(bytes, entry_offset, "segment start ET")?;
            let stop_et =
                file_record
                    .byte_order
                    .read_f64(bytes, entry_offset + 8, "segment stop ET")?;
            let ints_offset = entry_offset + 16;
            let target = file_record
                .byte_order
                .read_i32(bytes, ints_offset, "segment target")?;
            let center =
                file_record
                    .byte_order
                    .read_i32(bytes, ints_offset + 4, "segment center")?;
            let frame = file_record
                .byte_order
                .read_i32(bytes, ints_offset + 8, "segment frame")?;
            let data_type =
                file_record
                    .byte_order
                    .read_i32(bytes, ints_offset + 12, "segment data type")?;
            let start_address = file_record.byte_order.read_i32(
                bytes,
                ints_offset + 16,
                "segment start address",
            )?;
            let end_address =
                file_record
                    .byte_order
                    .read_i32(bytes, ints_offset + 20, "segment end address")?;

            segments.push(SpkSegmentDescriptor {
                name,
                start_et,
                stop_et,
                target,
                center,
                frame,
                data_type,
                start_address,
                end_address,
            });
        }

        record = next;
    }

    Ok(DafSpk {
        file_record,
        segments,
    })
}

/// Evaluate a type-2 SPK segment and return position in kilometers.
///
/// This is the position half of [`evaluate_type2_state`].
pub fn evaluate_type2_position(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
    et: f64,
) -> Result<[f64; 3], SpkError> {
    evaluate_type2_state(bytes, byte_order, segment, et).map(|state| state.position_km)
}

/// Evaluate a type-2 SPK segment and return position and velocity.
///
/// Type 2 stores Chebyshev coefficients for position only. As in CSPICE
/// `SPKR02`/`SPKE02`, the record is selected from the segment directory and
/// each component is evaluated with the `CHBINT` recurrence, which returns the
/// position and its derivative with respect to ET; the derivative is the
/// velocity in km/s.
pub fn evaluate_type2_state(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
    et: f64,
) -> Result<SpkStateVector, SpkError> {
    if segment.data_type != SPK_TYPE_2 {
        return Err(SpkError::UnsupportedSegmentType {
            expected: SPK_TYPE_2,
            actual: segment.data_type,
        });
    }

    let directory = read_type2_directory(bytes, byte_order, segment)?;
    let record_start = chebyshev_record_start(segment, &directory, et, "type-2 record address")?;
    let (mid, radius) = read_chebyshev_record_scale(
        bytes,
        byte_order,
        record_start,
        "type-2 record midpoint",
        "type-2 record radius",
    )?;

    let coeff_count = (directory.rsize - 2) / 3;
    let mut components = [(0.0, 0.0); 3];
    for (axis, component) in components.iter_mut().enumerate() {
        let coefficients = DafCoefficients {
            bytes,
            byte_order,
            start: record_start + 2 + axis * coeff_count,
            count: coeff_count,
        };
        *component = chbint(&coefficients, mid, radius, et)?;
    }

    Ok(SpkStateVector {
        position_km: [components[0].0, components[1].0, components[2].0],
        velocity_km_s: [components[0].1, components[1].1, components[2].1],
    })
}

/// Evaluate a type-3 SPK segment and return position and velocity.
///
/// As in CSPICE `SPKR03`/`SPKE03`, the record is selected from the segment
/// directory and each of the six components is evaluated with the `CHBVAL`
/// recurrence.
pub fn evaluate_type3_state(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
    et: f64,
) -> Result<SpkStateVector, SpkError> {
    if segment.data_type != SPK_TYPE_3 {
        return Err(SpkError::UnsupportedSegmentType {
            expected: SPK_TYPE_3,
            actual: segment.data_type,
        });
    }

    let directory = read_type3_directory(bytes, byte_order, segment)?;
    let record_start = chebyshev_record_start(segment, &directory, et, "type-3 record address")?;
    let (mid, radius) = read_chebyshev_record_scale(
        bytes,
        byte_order,
        record_start,
        "type-3 record midpoint",
        "type-3 record radius",
    )?;

    let coeff_count = (directory.rsize - 2) / 6;
    let mut components = [0.0; 6];
    for (index, component) in components.iter_mut().enumerate() {
        let coefficients = DafCoefficients {
            bytes,
            byte_order,
            start: record_start + 2 + index * coeff_count,
            count: coeff_count,
        };
        *component = chbval(&coefficients, mid, radius, et)?;
    }

    Ok(SpkStateVector {
        position_km: [components[0], components[1], components[2]],
        velocity_km_s: [components[3], components[4], components[5]],
    })
}

/// Evaluate a type-21 SPK segment (Extended Modified Difference Arrays) and
/// return position and velocity in kilometers and km/s.
///
/// Type 21 generalizes type 1: each logical record ("difference line") carries a
/// reference epoch, a stepsize vector, a reference state, and per-component
/// modified divided difference arrays whose table dimension `MAXDIM` is stored
/// in the segment rather than fixed at 15. Records are selected by the segment's
/// trailing epoch list (and directory), then evaluated with the Krogh MDA
/// recurrence. This mirrors CSPICE `SPKR21`/`SPKE21`.
pub fn evaluate_type21_state(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
    et: f64,
) -> Result<SpkStateVector, SpkError> {
    if segment.data_type != SPK_TYPE_21 {
        return Err(SpkError::UnsupportedSegmentType {
            expected: SPK_TYPE_21,
            actual: segment.data_type,
        });
    }

    let directory = read_type21_directory(bytes, byte_order, segment)?;
    let record_address = type21_record_address(bytes, byte_order, segment, &directory, et)?;

    // Assemble the SPKE21-style record: index 0 holds MAXDIM, indices
    // 1..=dlsize hold the difference line, matching CSPICE addressing exactly.
    let dlsize = 4 * directory.maxdim + 11;
    let mut record = [0.0f64; 4 * SPK_TYPE_21_MAX_TABLE_DIM + 12];
    record[0] = directory.maxdim as f64;
    for (offset, slot) in record[1..=dlsize].iter_mut().enumerate() {
        *slot = read_daf_f64(
            bytes,
            byte_order,
            record_address + offset,
            "type-21 difference line",
        )?;
    }

    evaluate_type21_record(&record, directory.maxdim, et)
}

#[derive(Debug, Clone, Copy)]
struct Type21Directory {
    /// One-based DAF address of the first record (difference line) word.
    begin_address: usize,
    /// One-based DAF address of the segment's final word.
    end_address: usize,
    /// Difference-table dimension per Cartesian component (`MAXDIM`).
    maxdim: usize,
    /// Number of logical records / epochs in the segment.
    record_count: usize,
    /// Number of directory epochs (`record_count / 100`).
    directory_count: usize,
}

fn read_type21_directory(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
) -> Result<Type21Directory, SpkError> {
    let begin = segment_start_address(segment)?;
    let end = segment_end_address(segment)?;
    if end < begin + 1 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "segment is shorter than the type-21 trailer",
        });
    }

    // The last two segment words are MAXDIM (end - 1) and the record count N (end).
    let maxdim = read_daf_f64(bytes, byte_order, end - 1, "type-21 MAXDIM")?;
    let record_count = read_daf_f64(bytes, byte_order, end, "type-21 record count")?;
    let maxdim = f64_to_usize(maxdim, "type-21 MAXDIM")?;
    let record_count = f64_to_usize(record_count, "type-21 record count")?;

    if maxdim == 0 || maxdim > SPK_TYPE_21_MAX_TABLE_DIM {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-21 difference table dimension is out of range",
        });
    }
    if record_count == 0 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-21 segment has zero records",
        });
    }

    let dlsize = 4 * maxdim + 11;
    let directory_count = record_count / SPK_TYPE_21_DIRECTORY_STRIDE;

    // Layout: N difference lines, then N epochs, then the directory epochs, then
    // the two trailer words (MAXDIM, N).
    let required_words = record_count
        .checked_mul(dlsize)
        .and_then(|words| words.checked_add(record_count))
        .and_then(|words| words.checked_add(directory_count))
        .and_then(|words| words.checked_add(2))
        .ok_or(SpkError::InvalidSegmentLayout {
            context: "type-21 segment word count overflow",
        })?;
    let segment_words = end - begin + 1;
    if required_words > segment_words {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-21 records exceed segment address range",
        });
    }

    Ok(Type21Directory {
        begin_address: begin,
        end_address: end,
        maxdim,
        record_count,
        directory_count,
    })
}

/// Resolve the one-based DAF address of the difference line covering `et`.
fn type21_record_address(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
    directory: &Type21Directory,
    et: f64,
) -> Result<usize, SpkError> {
    if !et.is_finite() {
        return Err(SpkError::InvalidDoubleField {
            field: "type-21 ET",
            value: et,
        });
    }
    if et < segment.start_et || et > segment.stop_et {
        return Err(SpkError::OutOfCoverage {
            et,
            start_et: segment.start_et,
            stop_et: segment.stop_et,
        });
    }

    // The epoch list ends just before the directory epochs and the two trailer
    // words, anchored at the segment end exactly as CSPICE SPKR21 computes it.
    let first_epoch_address = directory
        .end_address
        .checked_sub(directory.directory_count + 2 + directory.record_count)
        .ok_or(SpkError::InvalidSegmentLayout {
            context: "type-21 epoch list underflows segment",
        })?
        + 1;

    // Find the first record whose epoch is >= et (CSPICE LSTLTD + 1), clamped to
    // the final record. Reading the full epoch list is equivalent to the
    // directory-guided search and selects the identical record.
    let mut earlier = 0usize;
    for index in 0..directory.record_count {
        let epoch = read_daf_f64(
            bytes,
            byte_order,
            first_epoch_address + index,
            "type-21 epoch",
        )?;
        if epoch < et {
            earlier += 1;
        } else {
            break;
        }
    }
    let record_index = earlier.min(directory.record_count - 1);

    let dlsize = 4 * directory.maxdim + 11;
    checked_address_add(
        directory.begin_address,
        record_index
            .checked_mul(dlsize)
            .ok_or(SpkError::InvalidSegmentLayout {
                context: "type-21 record offset overflow",
            })?,
        "type-21 record address",
    )
}

/// Evaluate one assembled type-21 record at `et`. `record[0]` is `MAXDIM`;
/// `record[1..]` is the difference line. Ported line-for-line from CSPICE
/// `SPKE21` (Krogh's modified divided difference recurrence).
fn evaluate_type21_record(
    record: &[f64],
    maxdim: usize,
    et: f64,
) -> Result<SpkStateVector, SpkError> {
    const DIM: usize = SPK_TYPE_21_MAX_TABLE_DIM;
    let mut fc = [0.0f64; DIM];
    fc[0] = 1.0;
    let mut wc = [0.0f64; DIM - 1];
    let mut w = [0.0f64; DIM + 2];
    let mut g = [0.0f64; DIM];

    let tl = record[1];
    g[..maxdim].copy_from_slice(&record[2..2 + maxdim]);

    let refpos = [record[maxdim + 2], record[maxdim + 4], record[maxdim + 6]];
    let refvel = [record[maxdim + 3], record[maxdim + 5], record[maxdim + 7]];

    // Modified divided difference arrays, DT(MAXDIM, 3), stored column-major.
    let mut dt = [[0.0f64; 3]; DIM];
    for component in 0..3 {
        let base = (component + 1) * maxdim + 8;
        for (row, dt_row) in dt.iter_mut().enumerate().take(maxdim) {
            dt_row[component] = record[base + row];
        }
    }

    // KQMAX1 and the per-axis KQ orders are file-controlled. Validate them as
    // integers within the difference-table bounds before they index FC/WC/W/DT,
    // so a malformed kernel returns a typed error instead of panicking. SPICE
    // invariant: KQMAX1 = max(KQ) + 1, so 1 <= KQMAX1 <= MAXDIM+1 and each
    // 0 <= KQ < KQMAX1.
    let kqmax1_u = f64_to_usize(record[4 * maxdim + 8], "type-21 KQMAX1")?;
    if kqmax1_u < 1 || kqmax1_u > maxdim + 1 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-21 KQMAX1 is out of range",
        });
    }
    let kq_u = [
        f64_to_usize(record[4 * maxdim + 9], "type-21 KQ(1)")?,
        f64_to_usize(record[4 * maxdim + 10], "type-21 KQ(2)")?,
        f64_to_usize(record[4 * maxdim + 11], "type-21 KQ(3)")?,
    ];
    for &kqq in &kq_u {
        if kqq >= kqmax1_u {
            return Err(SpkError::InvalidSegmentLayout {
                context: "type-21 KQ order exceeds KQMAX1",
            });
        }
    }
    let kqmax1 = kqmax1_u as i64;
    let kq = [kq_u[0] as i64, kq_u[1] as i64, kq_u[2] as i64];

    let delta = et - tl;
    let mut tp = delta;
    let mq2 = kqmax1 - 2;
    let mut ks = kqmax1 - 1;

    // Build the stepsize-derived coefficients FC and WC.
    let mut j = 1i64;
    while j <= mq2 {
        let gj = g[(j - 1) as usize];
        if gj == 0.0 {
            return Err(SpkError::InvalidDoubleField {
                field: "type-21 stepsize vector",
                value: 0.0,
            });
        }
        fc[j as usize] = tp / gj;
        wc[(j - 1) as usize] = delta / gj;
        tp = delta + gj;
        j += 1;
    }

    // Seed the W array with reciprocals.
    let mut j = 1i64;
    while j <= kqmax1 {
        w[(j - 1) as usize] = 1.0 / j as f64;
        j += 1;
    }

    // Compute the W(K) terms used for position interpolation.
    let mut jx = 0i64;
    let mut ks1 = ks - 1;
    while ks >= 2 {
        jx += 1;
        let mut j = 1i64;
        while j <= jx {
            let term = fc[j as usize] * w[(j + ks1 - 1) as usize]
                - wc[(j - 1) as usize] * w[(j + ks - 1) as usize];
            w[(j + ks - 1) as usize] = term;
            j += 1;
        }
        ks = ks1;
        ks1 -= 1;
    }

    let mut state = [0.0f64; 6];
    for component in 0..3 {
        let kqq = kq[component];
        let mut sum = 0.0;
        let mut j = kqq;
        while j >= 1 {
            sum += dt[(j - 1) as usize][component] * w[(j + ks - 1) as usize];
            j -= 1;
        }
        state[component] = refpos[component] + delta * (refvel[component] + delta * sum);
    }

    // Recompute the W(K) terms for velocity interpolation.
    let mut j = 1i64;
    while j <= jx {
        let term = fc[j as usize] * w[(j + ks1 - 1) as usize]
            - wc[(j - 1) as usize] * w[(j + ks - 1) as usize];
        w[(j + ks - 1) as usize] = term;
        j += 1;
    }
    ks -= 1;

    for component in 0..3 {
        let kqq = kq[component];
        let mut sum = 0.0;
        let mut j = kqq;
        while j >= 1 {
            sum += dt[(j - 1) as usize][component] * w[(j + ks - 1) as usize];
            j -= 1;
        }
        state[component + 3] = refvel[component] + delta * sum;
    }

    Ok(SpkStateVector {
        position_km: [state[0], state[1], state[2]],
        velocity_km_s: [state[3], state[4], state[5]],
    })
}

fn parse_file_record(bytes: &[u8]) -> Result<DafFileRecord, SpkError> {
    if bytes.len() < DAF_FILE_RECORD_BYTES {
        return Err(SpkError::Truncated {
            context: "DAF file record",
            needed: DAF_FILE_RECORD_BYTES,
            actual: bytes.len(),
        });
    }

    let id_word = trim_ascii(&bytes[0..DAF_ID_BYTES]);
    let file_type = id_word
        .split_once('/')
        .map(|(_, file_type)| file_type.trim().to_string())
        .unwrap_or_default();
    if id_word != "DAF/SPK" {
        return Err(SpkError::UnsupportedDafId { id_word });
    }

    let binary_format = trim_ascii(
        &bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + DAF_BINARY_FORMAT_BYTES],
    );
    let byte_order = match binary_format.as_str() {
        "LTL-IEEE" => DafByteOrder::LittleEndian,
        "BIG-IEEE" => DafByteOrder::BigEndian,
        _ => {
            return Err(SpkError::UnsupportedBinaryFormat { binary_format });
        }
    };

    let double_components = byte_order.read_i32(bytes, 8, "ND")?;
    let integer_components = byte_order.read_i32(bytes, 12, "NI")?;
    let internal_name = trim_ascii(&bytes[16..16 + DAF_INTERNAL_NAME_BYTES]);
    let forward_record = byte_order.read_i32(bytes, 76, "FWARD")?;
    let backward_record = byte_order.read_i32(bytes, 80, "BWARD")?;
    let free_address = byte_order.read_i32(bytes, 84, "FREE")?;

    if forward_record < 0 {
        return Err(SpkError::InvalidField {
            field: "FWARD",
            value: forward_record,
        });
    }
    if backward_record < 0 {
        return Err(SpkError::InvalidField {
            field: "BWARD",
            value: backward_record,
        });
    }

    Ok(DafFileRecord {
        id_word,
        file_type,
        double_components,
        integer_components,
        internal_name,
        forward_record,
        backward_record,
        free_address,
        byte_order,
        binary_format,
    })
}

#[derive(Debug, Clone, Copy)]
struct ChebyshevDirectory {
    init: f64,
    intlen: f64,
    rsize: usize,
    n: usize,
}

fn read_type2_directory(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
) -> Result<ChebyshevDirectory, SpkError> {
    let start = segment_start_address(segment)?;
    let end = segment_end_address(segment)?;
    if end < start + 3 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "segment is shorter than the type-2 directory",
        });
    }

    let init = read_daf_f64(bytes, byte_order, end - 3, "type-2 INIT")?;
    let intlen = read_daf_f64(bytes, byte_order, end - 2, "type-2 INTLEN")?;
    let rsize = read_daf_f64(bytes, byte_order, end - 1, "type-2 RSIZE")?;
    let n = read_daf_f64(bytes, byte_order, end, "type-2 N")?;

    if !init.is_finite() {
        return Err(SpkError::InvalidDoubleField {
            field: "type-2 INIT",
            value: init,
        });
    }
    if !intlen.is_finite() || intlen <= 0.0 {
        return Err(SpkError::InvalidDoubleField {
            field: "type-2 INTLEN",
            value: intlen,
        });
    }

    let rsize = f64_to_usize(rsize, "type-2 RSIZE")?;
    let n = f64_to_usize(n, "type-2 N")?;
    if n == 0 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-2 segment has zero records",
        });
    }
    if rsize < 5 || (rsize - 2) % 3 != 0 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-2 RSIZE does not match three position components",
        });
    }

    let segment_words = end - start + 1;
    let required_words = n
        .checked_mul(rsize)
        .and_then(|words| words.checked_add(4))
        .ok_or(SpkError::InvalidSegmentLayout {
            context: "type-2 segment word count overflow",
        })?;
    if required_words > segment_words {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-2 records exceed segment address range",
        });
    }

    Ok(ChebyshevDirectory {
        init,
        intlen,
        rsize,
        n,
    })
}

fn read_type3_directory(
    bytes: &[u8],
    byte_order: DafByteOrder,
    segment: &SpkSegmentDescriptor,
) -> Result<ChebyshevDirectory, SpkError> {
    let start = segment_start_address(segment)?;
    let end = segment_end_address(segment)?;
    if end < start + 3 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "segment is shorter than the type-3 directory",
        });
    }

    let init = read_daf_f64(bytes, byte_order, end - 3, "type-3 INIT")?;
    let intlen = read_daf_f64(bytes, byte_order, end - 2, "type-3 INTLEN")?;
    let rsize = read_daf_f64(bytes, byte_order, end - 1, "type-3 RSIZE")?;
    let n = read_daf_f64(bytes, byte_order, end, "type-3 N")?;

    if !init.is_finite() {
        return Err(SpkError::InvalidDoubleField {
            field: "type-3 INIT",
            value: init,
        });
    }
    if !intlen.is_finite() || intlen <= 0.0 {
        return Err(SpkError::InvalidDoubleField {
            field: "type-3 INTLEN",
            value: intlen,
        });
    }

    let rsize = f64_to_usize(rsize, "type-3 RSIZE")?;
    let n = f64_to_usize(n, "type-3 N")?;
    if n == 0 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-3 segment has zero records",
        });
    }
    if rsize < 8 || (rsize - 2) % 6 != 0 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-3 RSIZE does not match six state components",
        });
    }

    let segment_words = end - start + 1;
    let required_words = n
        .checked_mul(rsize)
        .and_then(|words| words.checked_add(4))
        .ok_or(SpkError::InvalidSegmentLayout {
            context: "type-3 segment word count overflow",
        })?;
    if required_words > segment_words {
        return Err(SpkError::InvalidSegmentLayout {
            context: "type-3 records exceed segment address range",
        });
    }

    Ok(ChebyshevDirectory {
        init,
        intlen,
        rsize,
        n,
    })
}

/// Select the record for `et` as CSPICE `SPKR02`/`SPKR03` do:
/// `RECNO = INT((ET - INIT) / INTLEN) + 1`, then `RECNO = MIN(RECNO, NREC)`.
///
/// The segment descriptor's coverage is the gate, as in `SPKSFS`. Within it,
/// an epoch past the last directory interval uses the last record, and an
/// epoch less than one interval before `INIT` truncates to the first record.
/// A record number below one has no record in the segment and is refused.
fn chebyshev_record_index(
    segment: &SpkSegmentDescriptor,
    init: f64,
    intlen: f64,
    n: usize,
    et: f64,
) -> Result<usize, SpkError> {
    if et < segment.start_et || et > segment.stop_et {
        return Err(SpkError::OutOfCoverage {
            et,
            start_et: segment.start_et,
            stop_et: segment.stop_et,
        });
    }

    let truncated = ((et - init) / intlen).trunc();
    if !truncated.is_finite() || truncated < 0.0 {
        return Err(SpkError::OutOfCoverage {
            et,
            start_et: init,
            stop_et: init + intlen * n as f64,
        });
    }
    if truncated >= n as f64 {
        Ok(n - 1)
    } else {
        Ok(truncated as usize)
    }
}

fn chebyshev_record_start(
    segment: &SpkSegmentDescriptor,
    directory: &ChebyshevDirectory,
    et: f64,
    context: &'static str,
) -> Result<usize, SpkError> {
    let record_index =
        chebyshev_record_index(segment, directory.init, directory.intlen, directory.n, et)?;
    checked_address_add(
        segment_start_address(segment)?,
        record_index
            .checked_mul(directory.rsize)
            .ok_or(SpkError::InvalidSegmentLayout {
                context: "record offset overflow",
            })?,
        context,
    )
}

fn read_chebyshev_record_scale(
    bytes: &[u8],
    byte_order: DafByteOrder,
    record_start: usize,
    mid_field: &'static str,
    radius_field: &'static str,
) -> Result<(f64, f64), SpkError> {
    let mid = read_daf_f64(bytes, byte_order, record_start, mid_field)?;
    let radius = read_daf_f64(bytes, byte_order, record_start + 1, radius_field)?;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(SpkError::InvalidDoubleField {
            field: radius_field,
            value: radius,
        });
    }
    Ok((mid, radius))
}

/// Chebyshev coefficients `CP(1..=len)` read by index, so the recurrences
/// read them straight from the kernel bytes without copying.
trait ChebyshevCoefficients {
    /// Number of coefficients, `DEGP + 1`; at least one.
    fn len(&self) -> usize;
    /// Coefficient `CP(index + 1)`.
    fn get(&self, index: usize) -> Result<f64, SpkError>;
}

impl ChebyshevCoefficients for [f64] {
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    fn get(&self, index: usize) -> Result<f64, SpkError> {
        Ok(self[index])
    }
}

/// Consecutive coefficients of one component in a DAF record.
struct DafCoefficients<'a> {
    bytes: &'a [u8],
    byte_order: DafByteOrder,
    start: usize,
    count: usize,
}

impl ChebyshevCoefficients for DafCoefficients<'_> {
    fn len(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<f64, SpkError> {
        read_daf_f64(
            self.bytes,
            self.byte_order,
            self.start + index,
            "Chebyshev coefficient",
        )
    }
}

/// CSPICE `CHBVAL`: value of the Chebyshev expansion `cp` at `x`, with the
/// argument mapped to `s = (x - mid) / radius` and summed by the Clenshaw
/// recurrence in the order CSPICE uses. `cp` holds at least one coefficient.
fn chbval<C: ChebyshevCoefficients + ?Sized>(
    cp: &C,
    mid: f64,
    radius: f64,
    x: f64,
) -> Result<f64, SpkError> {
    let s = (x - mid) / radius;
    let s2 = 2.0 * s;
    let mut w = [0.0f64; 3];
    for index in (1..cp.len()).rev() {
        w[2] = w[1];
        w[1] = w[0];
        w[0] = cp.get(index)? + (s2 * w[1] - w[2]);
    }
    Ok((s * w[0] - w[1]) + cp.get(0)?)
}

/// CSPICE `CHBINT`: value and derivative with respect to `x` of the Chebyshev
/// expansion `cp`, with the argument mapped to `s = (x - mid) / radius`. The
/// derivative with respect to `s` is divided by `radius`, as CSPICE does.
/// `cp` holds at least one coefficient.
fn chbint<C: ChebyshevCoefficients + ?Sized>(
    cp: &C,
    mid: f64,
    radius: f64,
    x: f64,
) -> Result<(f64, f64), SpkError> {
    let s = (x - mid) / radius;
    let s2 = 2.0 * s;
    let mut w = [0.0f64; 3];
    let mut dw = [0.0f64; 3];
    for index in (1..cp.len()).rev() {
        w[2] = w[1];
        w[1] = w[0];
        w[0] = cp.get(index)? + (s2 * w[1] - w[2]);

        dw[2] = dw[1];
        dw[1] = dw[0];
        dw[0] = w[1] * 2.0 + dw[1] * s2 - dw[2];
    }
    let value = cp.get(0)? + (s * w[0] - w[1]);
    let derivative = (w[0] + s * dw[0] - dw[1]) / radius;
    Ok((value, derivative))
}

fn f64_to_usize(value: f64, field: &'static str) -> Result<usize, SpkError> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > usize::MAX as f64 {
        return Err(SpkError::InvalidDoubleField { field, value });
    }
    Ok(value as usize)
}

fn segment_start_address(segment: &SpkSegmentDescriptor) -> Result<usize, SpkError> {
    segment_address(segment.start_address, "segment start address")
}

fn segment_end_address(segment: &SpkSegmentDescriptor) -> Result<usize, SpkError> {
    let start = segment_start_address(segment)?;
    let end = segment_address(segment.end_address, "segment end address")?;
    if end < start {
        return Err(SpkError::InvalidSegmentLayout {
            context: "segment end address precedes start address",
        });
    }
    Ok(end)
}

fn segment_address(address: i32, field: &'static str) -> Result<usize, SpkError> {
    if address <= 0 {
        return Err(SpkError::InvalidField {
            field,
            value: address,
        });
    }
    usize::try_from(address).map_err(|_| SpkError::InvalidField {
        field,
        value: address,
    })
}

fn checked_address_add(
    address: usize,
    offset_words: usize,
    context: &'static str,
) -> Result<usize, SpkError> {
    address
        .checked_add(offset_words)
        .ok_or(SpkError::InvalidSegmentLayout { context })
}

fn read_daf_f64(
    bytes: &[u8],
    byte_order: DafByteOrder,
    address: usize,
    field: &'static str,
) -> Result<f64, SpkError> {
    if address == 0 {
        return Err(SpkError::InvalidSegmentLayout {
            context: "DAF addresses are one-based",
        });
    }
    let offset = (address - 1)
        .checked_mul(8)
        .ok_or(SpkError::InvalidSegmentLayout {
            context: "DAF address byte offset overflow",
        })?;
    byte_order.read_f64(bytes, offset, field)
}

fn read_summary_control_i32(
    byte_order: DafByteOrder,
    bytes: &[u8],
    summary_offset: usize,
    word_index: usize,
    field: &'static str,
) -> Result<i32, SpkError> {
    let value = byte_order.read_f64(bytes, summary_offset + word_index * 8, field)?;
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i32::MIN as f64
        || value > i32::MAX as f64
    {
        return Err(SpkError::InvalidField { field, value: 0 });
    }
    Ok(value as i32)
}

fn summary_word_count(nd: i32, ni: i32) -> Result<usize, SpkError> {
    if nd < 0 {
        return Err(SpkError::InvalidField {
            field: "ND",
            value: nd,
        });
    }
    if ni < 0 {
        return Err(SpkError::InvalidField {
            field: "NI",
            value: ni,
        });
    }

    let nd = usize::try_from(nd).map_err(|_| SpkError::InvalidField {
        field: "ND",
        value: nd,
    })?;
    let ni = usize::try_from(ni).map_err(|_| SpkError::InvalidField {
        field: "NI",
        value: ni,
    })?;
    Ok(nd + ni.div_ceil(2))
}

fn validate_summary_record_pointer(record: i32, field: &'static str) -> Result<(), SpkError> {
    paired_name_record(record, field).map(|_| ())
}

fn paired_name_record(record: i32, field: &'static str) -> Result<i32, SpkError> {
    record.checked_add(1).ok_or(SpkError::InvalidField {
        field,
        value: record,
    })
}

fn record_offset(record: i32, len: usize, context: &'static str) -> Result<usize, SpkError> {
    if record <= 0 {
        return Err(SpkError::InvalidField {
            field: context,
            value: record,
        });
    }
    let record = usize::try_from(record).map_err(|_| SpkError::InvalidField {
        field: context,
        value: record,
    })?;
    let offset = (record - 1)
        .checked_mul(DAF_RECORD_BYTES)
        .ok_or(SpkError::InvalidField {
            field: context,
            value: i32::MAX,
        })?;
    let needed = offset + DAF_RECORD_BYTES;
    if needed > len {
        return Err(SpkError::Truncated {
            context,
            needed,
            actual: len,
        });
    }
    Ok(offset)
}

fn trim_ascii(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .rposition(|byte| *byte != b' ' && *byte != 0)
        .map(|index| index + 1)
        .unwrap_or(0);
    let trimmed = &bytes[..end];
    String::from_utf8_lossy(trimmed).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_little_endian_daf_summaries() {
        let bytes = build_daf(DafByteOrder::LittleEndian);

        let parsed = parse_daf_spk(&bytes).unwrap();

        assert_eq!(parsed.file_record.id_word, "DAF/SPK");
        assert_eq!(parsed.file_record.file_type, "SPK");
        assert_eq!(parsed.file_record.double_components, 2);
        assert_eq!(parsed.file_record.integer_components, 6);
        assert_eq!(parsed.file_record.internal_name, "SYNTHETIC SPK");
        assert_eq!(parsed.file_record.forward_record, 3);
        assert_eq!(parsed.file_record.backward_record, 3);
        assert_eq!(parsed.file_record.free_address, 901);
        assert_eq!(parsed.file_record.byte_order, DafByteOrder::LittleEndian);
        assert_eq!(parsed.file_record.binary_format, "LTL-IEEE");
        assert_eq!(parsed.segments, expected_segments());
    }

    #[test]
    fn decodes_big_endian_daf_summaries() {
        let bytes = build_daf(DafByteOrder::BigEndian);

        let parsed = parse_daf_spk(&bytes).unwrap();

        assert_eq!(parsed.file_record.byte_order, DafByteOrder::BigEndian);
        assert_eq!(parsed.file_record.binary_format, "BIG-IEEE");
        assert_eq!(parsed.segments, expected_segments());
    }

    #[test]
    fn decodes_linear_daf_summary_record_chain() {
        let bytes = build_chained_daf(DafByteOrder::LittleEndian, 0.0);

        let parsed = parse_daf_spk(&bytes).unwrap();

        assert_eq!(parsed.file_record.forward_record, 3);
        assert_eq!(parsed.file_record.backward_record, 5);
        assert_eq!(parsed.segments, expected_segments());
    }

    #[test]
    fn cyclic_daf_summary_record_chain_returns_typed_error() {
        let bytes = build_chained_daf(DafByteOrder::LittleEndian, 3.0);

        let err = parse_daf_spk(&bytes).unwrap_err();

        assert_eq!(
            err,
            SpkError::InvalidField {
                field: "summary record chain",
                value: 3,
            }
        );
    }

    #[test]
    fn max_forward_summary_record_returns_typed_error() {
        let byte_order = DafByteOrder::LittleEndian;
        let mut bytes = build_daf(byte_order);
        write_i32(byte_order, &mut bytes, 76, i32::MAX);

        let err = parse_daf_spk(&bytes).unwrap_err();

        assert_eq!(
            err,
            SpkError::InvalidField {
                field: "FWARD",
                value: i32::MAX,
            }
        );
    }

    #[test]
    fn max_next_summary_record_returns_typed_error() {
        let byte_order = DafByteOrder::LittleEndian;
        let mut bytes = build_daf(byte_order);
        write_f64(
            byte_order,
            &mut bytes,
            DAF_RECORD_BYTES * 2,
            f64::from(i32::MAX),
        );

        let err = parse_daf_spk(&bytes).unwrap_err();

        assert_eq!(
            err,
            SpkError::InvalidField {
                field: "next summary record",
                value: i32::MAX,
            }
        );
    }

    #[test]
    fn truncated_header_returns_typed_error() {
        let err = parse_daf_spk(&[0u8; 16]).unwrap_err();

        assert_eq!(
            err,
            SpkError::Truncated {
                context: "DAF file record",
                needed: 1024,
                actual: 16,
            }
        );
    }

    #[test]
    fn evaluates_type2_position_records_and_boundaries() {
        let (bytes, segment) = build_type2_segment(DafByteOrder::LittleEndian);

        assert_position_close(
            evaluate_type2_position(&bytes, DafByteOrder::LittleEndian, &segment, 0.0).unwrap(),
            [2.0, -5.5, 7.0],
        );
        assert_position_close(
            evaluate_type2_position(&bytes, DafByteOrder::LittleEndian, &segment, 5.0).unwrap(),
            [-2.0, -3.0, 7.0],
        );
        assert_position_close(
            evaluate_type2_position(&bytes, DafByteOrder::LittleEndian, &segment, 10.0).unwrap(),
            [11.0, 19.0, -9.0],
        );
        assert_position_close(
            evaluate_type2_position(&bytes, DafByteOrder::LittleEndian, &segment, 20.0).unwrap(),
            [9.0, 23.0, -1.0],
        );
    }

    #[test]
    fn evaluates_big_endian_type2_position() {
        let (bytes, segment) = build_type2_segment(DafByteOrder::BigEndian);

        assert_position_close(
            evaluate_type2_position(&bytes, DafByteOrder::BigEndian, &segment, 15.0).unwrap(),
            [10.0, 19.0, -1.0],
        );
    }

    #[test]
    fn type2_out_of_coverage_returns_typed_error() {
        let (bytes, segment) = build_type2_segment(DafByteOrder::LittleEndian);

        let err = evaluate_type2_position(&bytes, DafByteOrder::LittleEndian, &segment, -0.001)
            .unwrap_err();

        assert_eq!(
            err,
            SpkError::OutOfCoverage {
                et: -0.001,
                start_et: 0.0,
                stop_et: 20.0,
            }
        );
    }

    #[test]
    fn evaluates_type3_state_records_and_boundaries() {
        let (bytes, segment) = build_type3_segment(DafByteOrder::LittleEndian);

        assert_state_close(
            evaluate_type3_state(&bytes, DafByteOrder::LittleEndian, &segment, 5.0).unwrap(),
            [1.0, 3.0, 5.0],
            [0.1, -0.3, 1.0],
        );
        assert_state_close(
            evaluate_type3_state(&bytes, DafByteOrder::LittleEndian, &segment, 10.0).unwrap(),
            [9.0, 22.0, -8.0],
            [0.0, 5.0, 3.75],
        );
        assert_state_close(
            evaluate_type3_state(&bytes, DafByteOrder::LittleEndian, &segment, 20.0).unwrap(),
            [11.0, 18.0, -2.0],
            [2.0, -1.0, 4.25],
        );
    }

    #[test]
    fn type3_out_of_coverage_returns_typed_error() {
        let (bytes, segment) = build_type3_segment(DafByteOrder::LittleEndian);

        let err =
            evaluate_type3_state(&bytes, DafByteOrder::LittleEndian, &segment, 20.001).unwrap_err();

        assert_eq!(
            err,
            SpkError::OutOfCoverage {
                et: 20.001,
                start_et: 0.0,
                stop_et: 20.0,
            }
        );
    }

    #[test]
    fn evaluates_type21_synthetic_records_and_boundaries() {
        let (bytes, segment) = build_type21_segment(DafByteOrder::LittleEndian);

        // Difference table is zero, so each record reduces to linear motion
        // refpos + delta * refvel about its own reference epoch. This makes the
        // selected-record/position/velocity result exactly checkable.
        assert_state_close(
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 0.0).unwrap(),
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
        );
        assert_state_close(
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 5.0).unwrap(),
            [105.0, 210.0, 315.0],
            [1.0, 2.0, 3.0],
        );
        // et == first epoch still resolves to the first record.
        assert_state_close(
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 10.0).unwrap(),
            [110.0, 220.0, 330.0],
            [1.0, 2.0, 3.0],
        );
        // Past the first epoch we cross into the second record.
        assert_state_close(
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 15.0).unwrap(),
            [1050.0, 2100.0, 3150.0],
            [10.0, 20.0, 30.0],
        );
        assert_state_close(
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 20.0).unwrap(),
            [1100.0, 2200.0, 3300.0],
            [10.0, 20.0, 30.0],
        );
    }

    #[test]
    fn evaluates_big_endian_type21_state() {
        let (bytes, segment) = build_type21_segment(DafByteOrder::BigEndian);

        assert_state_close(
            evaluate_type21_state(&bytes, DafByteOrder::BigEndian, &segment, 5.0).unwrap(),
            [105.0, 210.0, 315.0],
            [1.0, 2.0, 3.0],
        );
    }

    #[test]
    fn type21_out_of_coverage_returns_typed_error() {
        let (bytes, segment) = build_type21_segment(DafByteOrder::LittleEndian);

        let err = evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 20.001)
            .unwrap_err();

        assert_eq!(
            err,
            SpkError::OutOfCoverage {
                et: 20.001,
                start_et: 0.0,
                stop_et: 20.0,
            }
        );
    }

    #[test]
    fn type21_nonfinite_et_returns_typed_error() {
        let (bytes, segment) = build_type21_segment(DafByteOrder::LittleEndian);

        for et in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, et)
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    SpkError::InvalidDoubleField {
                        field: "type-21 ET",
                        ..
                    }
                ),
                "non-finite et {et} should be rejected, got {err:?}"
            );
        }
    }

    #[test]
    fn type21_malformed_order_words_return_typed_error() {
        // KQMAX1 lives at one-based address `start + 4*maxdim + 8`; for record 0
        // (start_address 1, maxdim 3) that is address 20. A hostile out-of-range
        // value must be rejected, not panic on an out-of-bounds array index.
        let maxdim = 3usize;
        let kqmax1_addr = 1 + 4 * maxdim + 8;
        let kq1_addr = 1 + 4 * maxdim + 9;

        // KQMAX1 far larger than MAXDIM+1.
        let (mut bytes, segment) = build_type21_segment(DafByteOrder::LittleEndian);
        write_f64_address(DafByteOrder::LittleEndian, &mut bytes, kqmax1_addr, 99.0);
        let err =
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 5.0).unwrap_err();
        assert!(
            matches!(err, SpkError::InvalidSegmentLayout { .. }),
            "oversized KQMAX1 should be rejected, got {err:?}"
        );

        // KQ order >= KQMAX1 (here KQMAX1 = 2, KQ(1) = 5).
        let (mut bytes, segment) = build_type21_segment(DafByteOrder::LittleEndian);
        write_f64_address(DafByteOrder::LittleEndian, &mut bytes, kq1_addr, 5.0);
        let err =
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 5.0).unwrap_err();
        assert!(
            matches!(err, SpkError::InvalidSegmentLayout { .. }),
            "out-of-range KQ should be rejected, got {err:?}"
        );

        // Non-integer KQMAX1 must also be rejected (f64_to_usize).
        let (mut bytes, segment) = build_type21_segment(DafByteOrder::LittleEndian);
        write_f64_address(
            DafByteOrder::LittleEndian,
            &mut bytes,
            kqmax1_addr,
            f64::NAN,
        );
        let err =
            evaluate_type21_state(&bytes, DafByteOrder::LittleEndian, &segment, 5.0).unwrap_err();
        assert!(
            matches!(
                err,
                SpkError::InvalidDoubleField { .. } | SpkError::InvalidSegmentLayout { .. }
            ),
            "non-integer KQMAX1 should be rejected, got {err:?}"
        );
    }

    /// Bit-exact parity against CSPICE on a real type-21 (Extended Modified
    /// Difference Array) kernel: a 433 Eros SPK fetched from JPL Horizons. The
    /// reference vectors are CSPICE `spkgeo(20000433, et, "J2000", 10)` output.
    /// See `tests/fixtures/spk/gen_eros_type21.py`. Agreement is within one ULP
    /// of the ~1e8 km / ~20 km/s magnitudes (observed max position residual
    /// 7.5e-9 km, velocity 4.4e-16 km/s).
    #[test]
    fn real_type21_kernel_matches_cspice_reference() {
        const KERNEL: &[u8] = include_bytes!("../../tests/fixtures/spk/horizons_eros_type21.bsp");

        // (et seconds past J2000 TDB, [x, y, z, vx, vy, vz]) km, km/s.
        const REFERENCE: &[(f64, [f64; 6])] = &[
            (
                757339200.0,
                [
                    198083634.33689928,
                    56306354.00566181,
                    67761020.0290685,
                    -14.136880898003753,
                    18.729945253375007,
                    8.080580941541488,
                ],
            ),
            (
                757655424.0,
                [
                    193484166.17007136,
                    62190955.161292516,
                    70271250.61392583,
                    -14.95317475425177,
                    18.482891309963502,
                    7.792787505001918,
                ],
            ),
            (
                760501440.0,
                [
                    140599517.39824444,
                    110142414.48840125,
                    87942357.2561364,
                    -22.110500498220798,
                    14.728648269072185,
                    4.367688325339683,
                ],
            ),
            (
                765244800.0,
                [
                    14324682.473833444,
                    151855494.96957216,
                    88809564.6055465,
                    -29.543141519840074,
                    1.384349579197926,
                    -4.552338928064369,
                ],
            ),
            (
                767879989.4592,
                [
                    -62463976.26374265,
                    142278295.29334122,
                    69496198.60194506,
                    -27.83899206786184,
                    -8.728214407471189,
                    -9.98623557339431,
                ],
            ),
            (
                773150400.0,
                [
                    -170058326.9714746,
                    51931259.239147335,
                    -1243746.8623651236,
                    -10.894277182103927,
                    -23.170448141059893,
                    -15.1244115495765,
                ],
            ),
            (
                778420810.5408,
                [
                    -175392526.76953015,
                    -73836504.48334283,
                    -73616410.40702733,
                    7.685028379798697,
                    -22.45642797493469,
                    -11.361635361252944,
                ],
            ),
            (
                781056000.0,
                [
                    -146671840.75331673,
                    -128352102.87455015,
                    -99379387.41255096,
                    13.724782826077085,
                    -18.71213758130164,
                    -8.14425619120024,
                ],
            ),
            (
                785799360.0,
                [
                    -65781054.32577276,
                    -197470134.64271438,
                    -124005727.09542452,
                    19.398793883808853,
                    -10.268325580683918,
                    -2.324610823885138,
                ],
            ),
            (
                788645376.0,
                [
                    -8859755.122267516,
                    -219270459.75263178,
                    -126097226.6160046,
                    20.34548822205983,
                    -5.074720269573395,
                    0.7953511794172388,
                ],
            ),
            (
                788961600.0,
                [
                    -2423286.488811064,
                    -220785626.12491044,
                    -125794359.14041424,
                    20.360009383792537,
                    -4.508637229520069,
                    1.1193915696949732,
                ],
            ),
        ];

        let spk = Spk::from_bytes(KERNEL).unwrap();

        let segments = spk.segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].data_type, SPK_TYPE_21);
        assert_eq!(segments[0].target, 20000433);
        assert_eq!(segments[0].center, 10);

        let mut max_position_error = 0.0f64;
        let mut max_velocity_error = 0.0f64;
        for &(et, expected) in REFERENCE {
            let state = spk.spk_state(20000433, 10, et).unwrap();
            let velocity = state.velocity_km_s;
            for axis in 0..3 {
                max_position_error =
                    max_position_error.max((state.position_km[axis] - expected[axis]).abs());
                max_velocity_error =
                    max_velocity_error.max((velocity[axis] - expected[axis + 3]).abs());
            }
        }

        // ~1-ULP gates at these magnitudes (|pos| ~2.2e8 km -> 1 ULP ~4.9e-8 km;
        // |vel| ~20 km/s -> 1 ULP ~4.4e-15 km/s). Measured agreement is sub-ULP
        // (~7.5e-9 km, ~4.4e-16 km/s); these gates assert the bit-exact claim
        // rather than a loose tolerance, and will catch any real regression.
        assert!(
            max_position_error < 5e-8,
            "type-21 position drift {max_position_error:e} km exceeds CSPICE parity gate"
        );
        assert!(
            max_velocity_error < 1e-14,
            "type-21 velocity drift {max_velocity_error:e} km/s exceeds CSPICE parity gate"
        );
    }

    #[test]
    fn spk_state_returns_direct_segment_state() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk.spk_state(301, 3, 5.0).unwrap();

        assert_eq!(spk.file_record().internal_name, "QUERY SPK");
        assert_eq!(spk.segments().len(), 4);
        assert_eq!(state.target, 301);
        assert_eq!(state.center, 3);
        assert_query_state_close(state, [100.0, 200.0, 300.0], [1.0, 2.0, 3.0], 1);
    }

    #[test]
    fn spk_state_chains_through_common_center() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk_state(&spk, 399, 3, 5.0).unwrap();

        assert_eq!(state.target, 399);
        assert_eq!(state.center, 3);
        assert_query_state_close(state, [950.0, -5.0, 5.0], [9.5, -0.25, 0.5], 1);
    }

    #[test]
    fn spk_state_prefers_later_overlapping_segments() {
        let bytes = build_priority_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let direct = spk.spk_state(301, 3, 5.0).unwrap();
        let chained = spk.spk_state(399, 3, 5.0).unwrap();

        assert_eq!(spk.segments().len(), 5);
        assert_query_state_close(direct, [900.0, 800.0, 700.0], [9.0, 8.0, 7.0], 1);
        assert_query_state_close(chained, [1950.0, 5.0, 25.0], [19.5, 0.25, 1.5], 1);
    }

    #[test]
    fn spk_state_prefers_later_type2_segment() {
        let bytes = build_position_only_priority_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk.spk_state(301, 3, 5.0).unwrap();

        // The later type-2 segment has one coefficient per axis, so its
        // Chebyshev derivative, and the velocity SPKE02 returns, is zero.
        assert_eq!(spk.segments().len(), 2);
        assert_query_state_close(state, [700.0, 800.0, 900.0], [0.0; 3], 1);
    }

    #[test]
    fn spk_state_errors_on_later_unsupported_segment() {
        let bytes = build_unsupported_priority_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let err = spk.spk_state(301, 3, 5.0).unwrap_err();
        let supported = spk.spk_state(302, 3, 5.0).unwrap();

        assert_eq!(err, SpkError::UnsupportedStateSegmentType { data_type: 99 });
        assert_query_state_close(supported, [400.0, 500.0, 600.0], [4.0, 5.0, 6.0], 1);
    }

    #[test]
    fn spk_state_prefers_later_segment_when_center_changes() {
        let bytes = build_changed_center_priority_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk.spk_state(301, 3, 5.0).unwrap();

        assert_eq!(spk.segments().len(), 3);
        assert_query_state_close(state, [1000.0, 80.0, 12.0], [100.0, 8.0, 1.2], 1);
    }

    #[test]
    fn spk_state_never_reverses_a_segment_to_break_a_priority_cycle() {
        let bytes = build_reversed_priority_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        // Body 301 has one segment (301 wrt 20). Body 20's highest-priority
        // segment is 20 wrt 301, so the SPKGEO target chain cycles
        // 301, 20, 301, ... without reaching 3, and body 3 has no segment of
        // its own. CSPICE reports insufficient data; it never evaluates the
        // 20-wrt-301 segment backwards.
        let err = spk.spk_state(301, 3, 5.0).unwrap_err();

        assert_eq!(spk.segments().len(), 3);
        assert_eq!(
            err,
            SpkError::CoverageGap {
                target: 301,
                center: 3,
                et: 5.0,
            }
        );
    }

    #[test]
    fn spk_state_chains_through_the_later_type2_segment() {
        let bytes = build_velocity_retention_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        // The later 800-wrt-20 segment is type 2; it takes precedence over the
        // earlier type-3 segment, and its velocity is its Chebyshev derivative.
        let state = spk.spk_state(800, 3, 5.0).unwrap();

        assert_query_state_close(state, [920.0, 3.0, 4.0], [2.0, 0.3, 0.4], 1);
    }

    /// `SPKGEO` for a target chain `target -> body -> observer` whose legs are
    /// in frames 1 and 2, with no observer leg: the first leg is rotated into
    /// frame 2 and added to the second, and the sum is rotated into the
    /// output frame 1.
    fn two_frame_chain(first: [f64; 6], second: [f64; 6]) -> [f64; 6] {
        let r12 = inertial_frame_rotation(1, 2).unwrap();
        let r21 = inertial_frame_rotation(2, 1).unwrap();
        let rotated = rotate6(&r12, &first);
        let sum = add6(&rotated, &second);
        let state = sub6(&sum, &[0.0; 6]);
        rotate6(&r21, &state)
    }

    fn rotate6(m: &[[f64; 3]; 3], v: &[f64; 6]) -> [f64; 6] {
        let p = mxv(m, [v[0], v[1], v[2]]);
        let w = mxv(m, [v[3], v[4], v[5]]);
        [p[0], p[1], p[2], w[0], w[1], w[2]]
    }

    fn assert_state6(actual: SpkState, expected: [f64; 6], frame: i32) {
        assert_eq!(actual.position_km, [expected[0], expected[1], expected[2]]);
        assert_eq!(
            actual.velocity_km_s,
            [expected[3], expected[4], expected[5]]
        );
        assert_eq!(actual.frame, frame);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn spk_state_rotates_legs_in_other_inertial_frames_instead_of_falling_back() {
        let bytes = build_frame_mismatch_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        // Body 700's highest-priority segment is 700 wrt 10 (frame 1), and
        // 10 wrt 3 is in frame 2 (B1950). SPKGEO rotates the first leg into
        // B1950 with IRFROT, adds the second, and rotates the sum into the
        // output frame; the earlier 700-wrt-20 chain is never used.
        let state = spk.spk_state(700, 3, 5.0).unwrap();
        let expected = two_frame_chain(
            [1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [900.0, 0.0, 0.0, 90.0, 0.0, 0.0],
        );
        assert_state6(state, expected, 1);

        // To within rounding, the result is the first leg plus the second leg
        // rotated from B1950 to J2000.
        let r21 = inertial_frame_rotation(2, 1).unwrap();
        let second = mxv(&r21, [900.0, 0.0, 0.0]);
        for axis in 0..3 {
            let approx = [1.0, 0.0, 0.0][axis] + second[axis];
            assert!((state.position_km[axis] - approx).abs() < 1e-9);
        }

        // The same query in B1950 is the rotated sum before the final rotation.
        let in_b1950 = spk.spk_state_in_frame(700, 3, 5.0, 2).unwrap();
        let r12 = inertial_frame_rotation(1, 2).unwrap();
        let sum = add6(
            &rotate6(&r12, &[1.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
            &[900.0, 0.0, 0.0, 90.0, 0.0, 0.0],
        );
        assert_state6(in_b1950, sub6(&sum, &[0.0; 6]), 2);
    }

    #[test]
    fn spk_state_rotates_every_chain_with_mixed_inertial_frames() {
        let bytes = build_frame_mismatch_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk.spk_state(701, 3, 5.0).unwrap();
        let expected = two_frame_chain(
            [701.0, 0.0, 0.0, 70.1, 0.0, 0.0],
            [30.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        );
        assert_state6(state, expected, 1);
    }

    #[test]
    fn spk_state_refuses_a_rotation_involving_a_non_inertial_frame() {
        let mut body_fixed = TestSegment::type3(3, 0, [50.0, 5.0, -5.0], [0.5, 0.25, -0.5]);
        body_fixed.frame = 10013;
        let spk = Spk::from_bytes(&build_test_spk(
            "NON INERTIAL SPK",
            &[
                body_fixed,
                TestSegment::type3(399, 3, [1000.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
            ],
        ))
        .unwrap();

        assert_eq!(
            spk.spk_state(399, 0, 5.0).unwrap_err(),
            SpkError::NonInertialFrameRotation { from: 1, to: 10013 }
        );
        // A leg in a non-inertial frame is used as is when nothing needs rotating.
        let direct = spk.spk_state(3, 0, 5.0).unwrap();
        assert_eq!(direct.position_km, [50.0, 5.0, -5.0]);
        assert_eq!(direct.frame, 10013);
        // Naming an unknown output frame is refused only when a rotation is needed.
        assert_eq!(
            spk.spk_state_in_frame(399, 3, 5.0, 22).unwrap_err(),
            SpkError::NonInertialFrameRotation { from: 1, to: 22 }
        );
        assert_eq!(
            inertial_frame_rotation(1, 0).unwrap_err(),
            SpkError::NonInertialFrameRotation { from: 1, to: 0 }
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn eclipj2000_rotation_is_the_j2000_obliquity_about_x() {
        // CHGIRF defines ECLIPJ2000 from J2000 as [84381.448"] about X.
        let eps = (84381.448 * (1.0 / 3600.0)) / (180.0 / libm::acos(-1.0));
        let (s, c) = (libm::sin(eps), libm::cos(eps));
        let to_ecliptic = inertial_frame_rotation(1, 17).unwrap();
        let expected = [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (to_ecliptic[i][j] - expected[i][j]).abs() <= 1e-16,
                    "[{i}][{j}] {} vs {}",
                    to_ecliptic[i][j],
                    expected[i][j]
                );
            }
        }
        // A vector on the J2000 pole has ecliptic latitude 90 - obliquity.
        let pole = mxv(&to_ecliptic, [0.0, 0.0, 1.0]);
        assert!((pole[1] - s).abs() <= 1e-16 && (pole[2] - c).abs() <= 1e-16);
        assert_eq!(inertial_frame_name(17), Some("ECLIPJ2000"));
        assert_eq!(inertial_frame_name(22), None);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn inertial_frame_rotations_are_orthonormal_and_invert_each_other() {
        for (a, b) in [(3, 2), (2, 3), (1, 3), (13, 1), (16, 18), (21, 4)] {
            let forward = inertial_frame_rotation(a, b).unwrap();
            let backward = inertial_frame_rotation(b, a).unwrap();
            let product = mxm(&forward, &backward);
            let gram = mxmt(&forward, &forward);
            for i in 0..3 {
                for j in 0..3 {
                    let identity = if i == j { 1.0 } else { 0.0 };
                    assert!((product[i][j] - identity).abs() < 1e-15, "{a}->{b}->{a}");
                    assert!((gram[i][j] - identity).abs() < 1e-15, "{a}->{b}");
                }
            }
        }
        // FK4 is B1950 turned 0.525" about Z (Fricke's equinox offset).
        let fk4_from_b1950 = inertial_frame_rotation(2, 3).unwrap();
        let angle = (0.525 * (1.0 / 3600.0)) / (180.0 / libm::acos(-1.0));
        assert!((fk4_from_b1950[0][1] - libm::sin(angle)).abs() < 1e-15);
        assert!((fk4_from_b1950[2][2] - 1.0).abs() < 1e-15);
        // DE-200 and DE-202 are J2000 turned by zero.
        assert_eq!(
            inertial_frame_rotation(1, 14).unwrap(),
            inertial_frame_rotation(1, 1).unwrap()
        );
    }

    #[test]
    fn nparsd_reads_the_chgirf_numerals_as_spicelib_does() {
        assert_eq!(nparsd("0.525"), 525.0 / 1000.0);
        assert_eq!(nparsd("84381.448"), 84381.0 + 448.0 / 1000.0);
        assert_eq!(
            nparsd("-1002.26108439117"),
            -(1002.0 + 26108439117.0 / 100000000000.0)
        );
        assert_eq!(nparsd("324000.0D0"), 324000.0);
        assert_eq!(nparsd("-152348.4D0"), -(152348.0 + 4.0 / 10.0));
        assert_eq!(nparsd("3,"), 3.0);
        assert_eq!(nparsd("0.0"), 0.0);
    }

    #[test]
    fn spk_state_returns_derivative_velocity_for_type2_segment() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk.spk_state(302, 3, 5.0).unwrap();

        assert_query_state_close(state, [7.0, 8.0, 9.0], [0.0; 3], 1);
    }

    #[test]
    fn spk_state_known_self_query_returns_zero_state() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let state = spk.spk_state(301, 301, 5.0).unwrap();

        assert_query_state_close(state, [0.0; 3], [0.0; 3], 0);
    }

    #[test]
    fn spk_state_self_query_is_zero_without_consulting_segments() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        // SPKGEO returns the zero state for TARG == OBS before any segment
        // search: a body's state relative to itself needs no ephemeris data.
        let unknown = spk.spk_state(999, 999, 5.0).unwrap();
        let uncovered = spk.spk_state(301, 301, 20.0).unwrap();

        assert_query_state_close(unknown, [0.0; 3], [0.0; 3], 0);
        assert_query_state_close(uncovered, [0.0; 3], [0.0; 3], 0);
    }

    #[test]
    fn spk_state_unknown_target_returns_typed_error() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let err = spk.spk_state(999, 0, 5.0).unwrap_err();

        assert_eq!(err, SpkError::UnknownBody { body: 999 });
    }

    #[test]
    fn spk_state_out_of_coverage_returns_typed_error() {
        let bytes = build_query_spk();
        let spk = Spk::from_bytes(&bytes).unwrap();

        let err = spk.spk_state(301, 3, 20.0).unwrap_err();

        assert_eq!(
            err,
            SpkError::CoverageGap {
                target: 301,
                center: 3,
                et: 20.0,
            }
        );
    }

    #[test]
    fn std_load_reads_spk_file() {
        let bytes = build_query_spk();
        let mut path = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!(
            "sidereon-spk-load-{}-{nonce}.bsp",
            std::process::id()
        ));

        std::fs::write(&path, &bytes).unwrap();

        let spk = Spk::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let state = spk.spk_state(301, 3, 5.0).unwrap();
        assert_query_state_close(state, [100.0, 200.0, 300.0], [1.0, 2.0, 3.0], 1);

        let err = Spk::load(&path).unwrap_err();
        match err {
            SpkError::Io {
                path: err_path,
                message,
            } => {
                assert_eq!(err_path, path.display().to_string());
                assert!(!message.is_empty());
            }
            other => panic!("expected IO error from missing SPK path, got {other:?}"),
        }
    }

    #[test]
    fn spk_state_uses_target_segment_not_later_reverse_segment() {
        let bytes = build_test_spk(
            "REVERSE PRIORITY SPK",
            &[
                TestSegment::type3(3, 0, [50.0, 5.0, -5.0], [0.5, 0.25, -0.5]),
                TestSegment::type3(399, 3, [1000.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
                TestSegment::type3(301, 3, [100.0, 200.0, 300.0], [1.0, 2.0, 3.0]),
                TestSegment::type3(301, 399, [7.0, 8.0, 9.0], [0.7, 0.8, 0.9]),
            ],
        );
        let spk = Spk::from_bytes(&bytes).unwrap();

        // Body 399's only segment with target 399 is 399 wrt 3. The later
        // 301-wrt-399 segment names 399 as its center and is not used to leave
        // 399: the chain is 399 -> 3 -> 0.
        let earth = spk.spk_state(399, 0, 5.0).unwrap();
        assert_eq!(earth.position_km, [1050.0, 5.0, -5.0]);
        assert_eq!(earth.velocity_km_s, [10.5, 0.25, -0.5]);

        // Observer 301's highest-priority segment is 301 wrt 399, which reaches
        // the target chain at its first body, so the state is 0 - (301 wrt 399).
        let reversed = spk.spk_state(399, 301, 5.0).unwrap();
        assert_eq!(reversed.position_km, [-7.0, -8.0, -9.0]);
        assert_eq!(reversed.velocity_km_s, [-0.7, -0.8, -0.9]);
        assert_eq!(reversed.frame, 1);
    }

    #[test]
    fn spk_kernels_later_loaded_kernel_takes_precedence() {
        let base = Spk::from_bytes(&build_test_spk(
            "BASE SPK",
            &[
                TestSegment::type3(3, 0, [50.0, 5.0, -5.0], [0.5, 0.25, -0.5]),
                TestSegment::type3(399, 3, [1000.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
            ],
        ))
        .unwrap();
        let update = Spk::from_bytes(&build_test_spk(
            "UPDATE SPK",
            &[TestSegment::type3(399, 3, [2000.0, 0.0, 0.0], [20.0, 0.0, 0.0]).covering(0.0, 4.0)],
        ))
        .unwrap();

        let mut kernels = SpkKernels::new();
        kernels.push(base.clone());
        kernels.push(update.clone());
        assert_eq!(kernels.len(), 2);

        // Inside the later kernel's coverage its segment wins; outside it the
        // earlier kernel's segment is used.
        assert_eq!(
            kernels.spk_state(399, 3, 2.0).unwrap().position_km,
            [2000.0, 0.0, 0.0]
        );
        assert_eq!(
            kernels.spk_state(399, 3, 5.0).unwrap().position_km,
            [1000.0, 0.0, 0.0]
        );
        // A chain mixes kernels: 399 -> 3 from the update, 3 -> 0 from the base.
        assert_eq!(
            kernels.spk_state(399, 0, 4.0).unwrap().position_km,
            [2050.0, 5.0, -5.0]
        );

        // Reversing the load order reverses the precedence.
        let reversed: SpkKernels = vec![update, base].into_iter().collect();
        assert_eq!(
            reversed.spk_state(399, 3, 2.0).unwrap().position_km,
            [1000.0, 0.0, 0.0]
        );
    }

    #[test]
    fn spk_kernels_unsupported_highest_priority_segment_is_an_error() {
        let base = Spk::from_bytes(&build_test_spk(
            "BASE SPK",
            &[TestSegment::type3(301, 3, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])],
        ))
        .unwrap();
        let update = Spk::from_bytes(&build_test_spk(
            "UNSUPPORTED UPDATE SPK",
            &[TestSegment::type3(301, 3, [9.0, 9.0, 9.0], [0.9, 0.9, 0.9]).with_data_type(99)],
        ))
        .unwrap();
        let kernels = SpkKernels::from(vec![base, update]);

        assert_eq!(
            kernels.spk_state(301, 3, 5.0).unwrap_err(),
            SpkError::UnsupportedStateSegmentType { data_type: 99 }
        );
    }

    #[test]
    fn spk_kernels_empty_set_reports_unknown_body() {
        let kernels = SpkKernels::new();
        assert!(kernels.is_empty());
        assert_eq!(
            kernels.spk_state(399, 0, 0.0).unwrap_err(),
            SpkError::UnknownBody { body: 399 }
        );
    }

    #[test]
    fn chebyshev_recurrences_match_hand_computed_values() {
        // Degree-3 expansion 0.5 T0 - 1.25 T1 + 2 T2 + 0.75 T3 at s = 0.25.
        // Every intermediate value is a short dyadic fraction, so the
        // recurrences are exact and the expected values are exact.
        let cp = [0.5, -1.25, 2.0, 0.75];
        assert_eq!(chbval(&cp[..], 0.0, 4.0, 1.0).unwrap(), -2.078125);
        let (value, derivative) = chbint(&cp[..], 0.0, 4.0, 1.0).unwrap();
        assert_eq!(value, -2.078125);
        // d/ds = -0.9375, divided by the radius 4.
        assert_eq!(derivative, -0.234375);

        // One coefficient: the value is the coefficient and the derivative zero.
        assert_eq!(chbval(&[7.0][..], 5.0, 5.0, 3.0).unwrap(), 7.0);
        assert_eq!(chbint(&[7.0][..], 5.0, 5.0, 3.0).unwrap(), (7.0, 0.0));
    }

    #[test]
    fn type2_state_velocity_is_the_chebyshev_derivative() {
        let (bytes, segment) = build_type2_segment(DafByteOrder::LittleEndian);

        // Record 1 has midpoint 5 and radius 5, so ET 7.5 is s = 0.5.
        // x = 1 + 2 T1 + 3 T2: value 0.5, dx/dt = (2 + 12 s) / 5 = 1.6.
        // y = -4 + 0.5 T1 - T2: value -3.25, dy/dt = (0.5 - 4 s) / 5 = -0.3.
        // z = 7: value 7, derivative 0.
        let state =
            evaluate_type2_state(&bytes, DafByteOrder::LittleEndian, &segment, 7.5).unwrap();
        assert_eq!(state.position_km, [0.5, -3.25, 7.0]);
        assert_eq!(state.velocity_km_s, [1.6, -0.3, 0.0]);
        assert_eq!(
            evaluate_type2_position(&bytes, DafByteOrder::LittleEndian, &segment, 7.5).unwrap(),
            state.position_km
        );
    }

    #[test]
    fn chebyshev_record_selection_truncates_and_clamps_like_spkr02() {
        let (bytes, mut segment) = build_type2_segment(DafByteOrder::LittleEndian);
        // The directory covers [0, 20] in two 10-second records; widen the
        // descriptor on both sides.
        segment.start_et = -15.0;
        segment.stop_et = 25.0;

        // Past the last interval: MIN(RECNO, NREC) selects the last record
        // (midpoint 15, radius 5, s = 1.4), x = 10 - T1.
        let late =
            evaluate_type2_state(&bytes, DafByteOrder::LittleEndian, &segment, 22.0).unwrap();
        assert_eq!(late.position_km[0], 10.0 + (-1.4 - 0.0));

        // Less than one interval before INIT: INT(-0.3) = 0 selects record 1
        // (midpoint 5, radius 5, s = -1.6), x = 1 + 2 T1 + 3 T2.
        let early =
            evaluate_type2_state(&bytes, DafByteOrder::LittleEndian, &segment, -3.0).unwrap();
        let s: f64 = -1.6;
        let expected = chbval(&[1.0, 2.0, 3.0][..], 5.0, 5.0, -3.0).unwrap();
        assert_eq!(early.position_km[0], expected);
        assert!((expected - (1.0 + 2.0 * s + 3.0 * (2.0 * s * s - 1.0))).abs() < 1e-12);

        // A full interval or more before INIT gives RECNO < 1: no record.
        assert_eq!(
            evaluate_type2_state(&bytes, DafByteOrder::LittleEndian, &segment, -12.0).unwrap_err(),
            SpkError::OutOfCoverage {
                et: -12.0,
                start_et: 0.0,
                stop_et: 20.0,
            }
        );
    }

    fn expected_segments() -> Vec<SpkSegmentDescriptor> {
        vec![
            SpkSegmentDescriptor {
                name: "MERCURY BARYCENTER".to_string(),
                start_et: -100.0,
                stop_et: 100.0,
                target: 1,
                center: 0,
                frame: 1,
                data_type: 2,
                start_address: 513,
                end_address: 700,
            },
            SpkSegmentDescriptor {
                name: "EARTH".to_string(),
                start_et: 200.0,
                stop_et: 400.0,
                target: 399,
                center: 3,
                frame: 17,
                data_type: 3,
                start_address: 701,
                end_address: 900,
            },
        ]
    }

    fn build_daf(byte_order: DafByteOrder) -> Vec<u8> {
        let mut bytes = vec![0u8; DAF_RECORD_BYTES * 4];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(&mut bytes, 16, DAF_INTERNAL_NAME_BYTES, "SYNTHETIC SPK");
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, 901);
        match byte_order {
            DafByteOrder::LittleEndian => bytes
                [DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8]
                .copy_from_slice(b"LTL-IEEE"),
            DafByteOrder::BigEndian => bytes
                [DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8]
                .copy_from_slice(b"BIG-IEEE"),
        }

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 2.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            -100.0,
            100.0,
            [1, 0, 1, 2, 513, 700],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            200.0,
            400.0,
            [399, 3, 17, 3, 701, 900],
        );
        write_ascii(&mut bytes, name_offset, 40, "MERCURY BARYCENTER");
        write_ascii(&mut bytes, name_offset + 40, 40, "EARTH");

        bytes
    }

    fn build_chained_daf(byte_order: DafByteOrder, final_next_record: f64) -> Vec<u8> {
        let mut bytes = vec![0u8; DAF_RECORD_BYTES * 6];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(&mut bytes, 16, DAF_INTERNAL_NAME_BYTES, "CHAINED SPK");
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 5);
        write_i32(byte_order, &mut bytes, 84, 901);
        match byte_order {
            DafByteOrder::LittleEndian => bytes
                [DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8]
                .copy_from_slice(b"LTL-IEEE"),
            DafByteOrder::BigEndian => bytes
                [DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8]
                .copy_from_slice(b"BIG-IEEE"),
        }

        let first_summary_offset = DAF_RECORD_BYTES * 2;
        let first_name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, first_summary_offset, 5.0);
        write_f64(byte_order, &mut bytes, first_summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, first_summary_offset + 16, 1.0);
        write_summary(
            byte_order,
            &mut bytes,
            first_summary_offset + 24,
            -100.0,
            100.0,
            [1, 0, 1, 2, 513, 700],
        );
        write_ascii(&mut bytes, first_name_offset, 40, "MERCURY BARYCENTER");

        let second_summary_offset = DAF_RECORD_BYTES * 4;
        let second_name_offset = DAF_RECORD_BYTES * 5;
        write_f64(
            byte_order,
            &mut bytes,
            second_summary_offset,
            final_next_record,
        );
        write_f64(byte_order, &mut bytes, second_summary_offset + 8, 3.0);
        write_f64(byte_order, &mut bytes, second_summary_offset + 16, 1.0);
        write_summary(
            byte_order,
            &mut bytes,
            second_summary_offset + 24,
            200.0,
            400.0,
            [399, 3, 17, 3, 701, 900],
        );
        write_ascii(&mut bytes, second_name_offset, 40, "EARTH");

        bytes
    }

    fn build_query_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let direct_start = 513usize;
        let target_to_ssb_start = 525usize;
        let center_to_ssb_start = 537usize;
        let type2_start = 549usize;
        let type2_end = type2_start + 8;
        let mut bytes = vec![0u8; type2_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(&mut bytes, 16, DAF_INTERNAL_NAME_BYTES, "QUERY SPK");
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (type2_end + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 4.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [301, 3, 1, 3, direct_start as i32, direct_start as i32 + 11],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [
                399,
                0,
                1,
                3,
                target_to_ssb_start as i32,
                target_to_ssb_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                3,
                0,
                1,
                3,
                center_to_ssb_start as i32,
                center_to_ssb_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 144,
            0.0,
            10.0,
            [302, 3, 1, 2, type2_start as i32, type2_end as i32],
        );

        write_ascii(&mut bytes, name_offset, 40, "BODY 301 TO CENTER 3");
        write_ascii(&mut bytes, name_offset + 40, 40, "BODY 399 TO SSB");
        write_ascii(&mut bytes, name_offset + 80, 40, "CENTER 3 TO SSB");
        write_ascii(
            &mut bytes,
            name_offset + 120,
            40,
            "TYPE2 BODY 302 TO CENTER 3",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            direct_start,
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            target_to_ssb_start,
            [1000.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            center_to_ssb_start,
            [50.0, 5.0, -5.0],
            [0.5, 0.25, -0.5],
        );
        write_type2_constant_segment(byte_order, &mut bytes, type2_start, [7.0, 8.0, 9.0]);

        bytes
    }

    fn build_priority_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let direct_early_start = 513usize;
        let direct_priority_start = 525usize;
        let target_early_start = 537usize;
        let center_start = 549usize;
        let target_priority_start = 561usize;
        let target_priority_end = target_priority_start + 11;
        let mut bytes = vec![0u8; target_priority_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(&mut bytes, 16, DAF_INTERNAL_NAME_BYTES, "PRIORITY SPK");
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (target_priority_end + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 5.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                3,
                direct_early_start as i32,
                direct_early_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                3,
                direct_priority_start as i32,
                direct_priority_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                399,
                0,
                1,
                3,
                target_early_start as i32,
                target_early_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 144,
            0.0,
            10.0,
            [3, 0, 1, 3, center_start as i32, center_start as i32 + 11],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 184,
            0.0,
            10.0,
            [
                399,
                0,
                1,
                3,
                target_priority_start as i32,
                target_priority_start as i32 + 11,
            ],
        );

        write_ascii(&mut bytes, name_offset, 40, "EARLY BODY 301 TO CENTER 3");
        write_ascii(
            &mut bytes,
            name_offset + 40,
            40,
            "PRIORITY BODY 301 TO CENTER 3",
        );
        write_ascii(&mut bytes, name_offset + 80, 40, "EARLY BODY 399 TO SSB");
        write_ascii(&mut bytes, name_offset + 120, 40, "CENTER 3 TO SSB");
        write_ascii(
            &mut bytes,
            name_offset + 160,
            40,
            "PRIORITY BODY 399 TO SSB",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            direct_early_start,
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            direct_priority_start,
            [900.0, 800.0, 700.0],
            [9.0, 8.0, 7.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            target_early_start,
            [1000.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            center_start,
            [50.0, 5.0, -5.0],
            [0.5, 0.25, -0.5],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            target_priority_start,
            [2000.0, 10.0, 20.0],
            [20.0, 0.5, 1.0],
        );

        bytes
    }

    fn build_position_only_priority_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let direct_early_start = 513usize;
        let direct_priority_start = 525usize;
        let direct_priority_end = direct_priority_start + 8;
        let mut bytes = vec![0u8; direct_priority_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(
            &mut bytes,
            16,
            DAF_INTERNAL_NAME_BYTES,
            "POSITION ONLY PRIORITY SPK",
        );
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (direct_priority_end + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 2.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                3,
                direct_early_start as i32,
                direct_early_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                2,
                direct_priority_start as i32,
                direct_priority_end as i32,
            ],
        );

        write_ascii(
            &mut bytes,
            name_offset,
            40,
            "EARLY TYPE3 BODY 301 TO CENTER 3",
        );
        write_ascii(
            &mut bytes,
            name_offset + 40,
            40,
            "LATE TYPE2 BODY 301 TO CENTER 3",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            direct_early_start,
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
        );
        write_type2_constant_segment(
            byte_order,
            &mut bytes,
            direct_priority_start,
            [700.0, 800.0, 900.0],
        );

        bytes
    }

    fn build_unsupported_priority_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let direct_early_start = 513usize;
        let unsupported_start = 525usize;
        let supported_start = 537usize;
        let supported_end = supported_start + 11;
        let mut bytes = vec![0u8; supported_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(
            &mut bytes,
            16,
            DAF_INTERNAL_NAME_BYTES,
            "UNSUPPORTED PRIORITY SPK",
        );
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (supported_end + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 3.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                3,
                direct_early_start as i32,
                direct_early_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                99,
                unsupported_start as i32,
                unsupported_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                302,
                3,
                1,
                3,
                supported_start as i32,
                supported_start as i32 + 11,
            ],
        );

        write_ascii(
            &mut bytes,
            name_offset,
            40,
            "EARLY TYPE3 BODY 301 TO CENTER 3",
        );
        write_ascii(
            &mut bytes,
            name_offset + 40,
            40,
            "LATE UNSUPPORTED BODY 301 TO 3",
        );
        write_ascii(
            &mut bytes,
            name_offset + 80,
            40,
            "SUPPORTED BODY 302 TO CENTER 3",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            direct_early_start,
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            unsupported_start,
            [900.0, 900.0, 900.0],
            [9.0, 9.0, 9.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            supported_start,
            [400.0, 500.0, 600.0],
            [4.0, 5.0, 6.0],
        );

        bytes
    }

    fn build_changed_center_priority_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let direct_early_start = 513usize;
        let center_start = 525usize;
        let target_priority_start = 537usize;
        let target_priority_end = target_priority_start + 11;
        let mut bytes = vec![0u8; target_priority_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(
            &mut bytes,
            16,
            DAF_INTERNAL_NAME_BYTES,
            "CHANGED CENTER PRIORITY SPK",
        );
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (target_priority_end + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 3.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [
                301,
                3,
                1,
                3,
                direct_early_start as i32,
                direct_early_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [20, 3, 1, 3, center_start as i32, center_start as i32 + 11],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                301,
                20,
                1,
                3,
                target_priority_start as i32,
                target_priority_start as i32 + 11,
            ],
        );

        write_ascii(&mut bytes, name_offset, 40, "EARLY BODY 301 TO CENTER 3");
        write_ascii(&mut bytes, name_offset + 40, 40, "CENTER 20 TO 3");
        write_ascii(
            &mut bytes,
            name_offset + 80,
            40,
            "PRIORITY BODY 301 TO CENTER 20",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            direct_early_start,
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            center_start,
            [100.0, 0.0, 5.0],
            [10.0, 0.0, 0.5],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            target_priority_start,
            [900.0, 80.0, 7.0],
            [90.0, 8.0, 0.7],
        );

        bytes
    }

    fn build_reversed_priority_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let forward_start = 513usize;
        let center_start = 525usize;
        let reversed_priority_start = 537usize;
        let reversed_priority_end = reversed_priority_start + 11;
        let mut bytes = vec![0u8; reversed_priority_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(
            &mut bytes,
            16,
            DAF_INTERNAL_NAME_BYTES,
            "REVERSED PRIORITY SPK",
        );
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(
            byte_order,
            &mut bytes,
            84,
            (reversed_priority_end + 1) as i32,
        );
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 3.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [
                301,
                20,
                1,
                3,
                forward_start as i32,
                forward_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [20, 3, 1, 3, center_start as i32, center_start as i32 + 11],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                20,
                301,
                1,
                3,
                reversed_priority_start as i32,
                reversed_priority_start as i32 + 11,
            ],
        );

        write_ascii(&mut bytes, name_offset, 40, "EARLY BODY 301 TO CENTER 20");
        write_ascii(&mut bytes, name_offset + 40, 40, "CENTER 20 TO 3");
        write_ascii(&mut bytes, name_offset + 80, 40, "PRIORITY BODY 20 TO 301");

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            forward_start,
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            center_start,
            [100.0, 0.0, 5.0],
            [10.0, 0.0, 0.5],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            reversed_priority_start,
            [900.0, 80.0, 7.0],
            [90.0, 8.0, 0.7],
        );

        bytes
    }

    fn build_velocity_retention_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let center_start = 513usize;
        let velocity_target_start = 525usize;
        let position_target_start = 537usize;
        let position_target_end = position_target_start + 8;
        let mut bytes = vec![0u8; position_target_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(
            &mut bytes,
            16,
            DAF_INTERNAL_NAME_BYTES,
            "VELOCITY RETENTION SPK",
        );
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (position_target_end + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 3.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [20, 3, 1, 3, center_start as i32, center_start as i32 + 11],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [
                800,
                20,
                1,
                3,
                velocity_target_start as i32,
                velocity_target_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                800,
                20,
                1,
                2,
                position_target_start as i32,
                position_target_end as i32,
            ],
        );

        write_ascii(&mut bytes, name_offset, 40, "CENTER 20 TO 3");
        write_ascii(
            &mut bytes,
            name_offset + 40,
            40,
            "VELOCITY TARGET 800 TO 20",
        );
        write_ascii(
            &mut bytes,
            name_offset + 80,
            40,
            "POSITION TARGET 800 TO 20",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            center_start,
            [20.0, 3.0, 4.0],
            [2.0, 0.3, 0.4],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            velocity_target_start,
            [100.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        );
        write_type2_constant_segment(
            byte_order,
            &mut bytes,
            position_target_start,
            [900.0, 0.0, 0.0],
        );

        bytes
    }

    fn build_frame_mismatch_spk() -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let good_center_start = 513usize;
        let good_target_start = 525usize;
        let bad_center_start = 537usize;
        let bad_target_start = 549usize;
        let unresolvable_center_start = 561usize;
        let unresolvable_target_start = 573usize;
        let unresolvable_target_end = unresolvable_target_start + 11;
        let mut bytes = vec![0u8; unresolvable_target_end * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(
            &mut bytes,
            16,
            DAF_INTERNAL_NAME_BYTES,
            "FRAME MISMATCH SPK",
        );
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(
            byte_order,
            &mut bytes,
            84,
            (unresolvable_target_end + 1) as i32,
        );
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 16, 6.0);

        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 24,
            0.0,
            10.0,
            [
                20,
                3,
                1,
                3,
                good_center_start as i32,
                good_center_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 64,
            0.0,
            10.0,
            [
                700,
                20,
                1,
                3,
                good_target_start as i32,
                good_target_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 104,
            0.0,
            10.0,
            [
                10,
                3,
                2,
                3,
                bad_center_start as i32,
                bad_center_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 144,
            0.0,
            10.0,
            [
                700,
                10,
                1,
                3,
                bad_target_start as i32,
                bad_target_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 184,
            0.0,
            10.0,
            [
                30,
                3,
                2,
                3,
                unresolvable_center_start as i32,
                unresolvable_center_start as i32 + 11,
            ],
        );
        write_summary(
            byte_order,
            &mut bytes,
            summary_offset + 224,
            0.0,
            10.0,
            [
                701,
                30,
                1,
                3,
                unresolvable_target_start as i32,
                unresolvable_target_start as i32 + 11,
            ],
        );

        write_ascii(&mut bytes, name_offset, 40, "GOOD CENTER 20 TO 3");
        write_ascii(&mut bytes, name_offset + 40, 40, "GOOD TARGET 700 TO 20");
        write_ascii(&mut bytes, name_offset + 80, 40, "BAD CENTER 10 TO 3");
        write_ascii(&mut bytes, name_offset + 120, 40, "BAD TARGET 700 TO 10");
        write_ascii(&mut bytes, name_offset + 160, 40, "BAD CENTER 30 TO 3");
        write_ascii(
            &mut bytes,
            name_offset + 200,
            40,
            "UNRESOLVABLE TARGET 701 TO 30",
        );

        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            good_center_start,
            [20.0, 3.0, 4.0],
            [2.0, 0.3, 0.4],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            good_target_start,
            [100.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            bad_center_start,
            [900.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            bad_target_start,
            [1.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            unresolvable_center_start,
            [30.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        );
        write_type3_constant_segment(
            byte_order,
            &mut bytes,
            unresolvable_target_start,
            [701.0, 0.0, 0.0],
            [70.1, 0.0, 0.0],
        );

        bytes
    }

    fn build_type2_segment(byte_order: DafByteOrder) -> (Vec<u8>, SpkSegmentDescriptor) {
        let rsize = 11usize;
        let n = 2usize;
        let end_address = rsize * n + 4;
        let mut bytes = vec![0u8; end_address * 8];

        write_type2_record(
            byte_order,
            &mut bytes,
            1,
            5.0,
            5.0,
            [[1.0, 2.0, 3.0], [-4.0, 0.5, -1.0], [7.0, 0.0, 0.0]],
        );
        write_type2_record(
            byte_order,
            &mut bytes,
            12,
            15.0,
            5.0,
            [[10.0, -1.0, 0.0], [20.0, 2.0, 1.0], [-3.0, 4.0, -2.0]],
        );
        write_f64_address(byte_order, &mut bytes, end_address - 3, 0.0);
        write_f64_address(byte_order, &mut bytes, end_address - 2, 10.0);
        write_f64_address(byte_order, &mut bytes, end_address - 1, rsize as f64);
        write_f64_address(byte_order, &mut bytes, end_address, n as f64);

        (
            bytes,
            SpkSegmentDescriptor {
                name: "TYPE 2 TEST".to_string(),
                start_et: 0.0,
                stop_et: 20.0,
                target: 301,
                center: 3,
                frame: 1,
                data_type: 2,
                start_address: 1,
                end_address: end_address as i32,
            },
        )
    }

    fn build_type3_segment(byte_order: DafByteOrder) -> (Vec<u8>, SpkSegmentDescriptor) {
        let rsize = 14usize;
        let n = 2usize;
        let end_address = rsize * n + 4;
        let mut bytes = vec![0u8; end_address * 8];

        write_type3_record(
            byte_order,
            &mut bytes,
            1,
            5.0,
            5.0,
            [
                [1.0, 2.0],
                [3.0, -4.0],
                [5.0, 0.5],
                [0.1, 0.2],
                [-0.3, 0.0],
                [1.0, -1.0],
            ],
        );
        write_type3_record(
            byte_order,
            &mut bytes,
            15,
            15.0,
            5.0,
            [
                [10.0, 1.0],
                [20.0, -2.0],
                [-5.0, 3.0],
                [1.0, 1.0],
                [2.0, -3.0],
                [4.0, 0.25],
            ],
        );
        write_f64_address(byte_order, &mut bytes, end_address - 3, 0.0);
        write_f64_address(byte_order, &mut bytes, end_address - 2, 10.0);
        write_f64_address(byte_order, &mut bytes, end_address - 1, rsize as f64);
        write_f64_address(byte_order, &mut bytes, end_address, n as f64);

        (
            bytes,
            SpkSegmentDescriptor {
                name: "TYPE 3 TEST".to_string(),
                start_et: 0.0,
                stop_et: 20.0,
                target: 499,
                center: 4,
                frame: 1,
                data_type: 3,
                start_address: 1,
                end_address: end_address as i32,
            },
        )
    }

    fn build_type21_segment(byte_order: DafByteOrder) -> (Vec<u8>, SpkSegmentDescriptor) {
        let maxdim = 3usize;
        let n = 2usize;
        let dlsize = 4 * maxdim + 11;
        // N difference lines, then N epochs, then 0 directory epochs, then the
        // two trailer words MAXDIM and N.
        let end_address = n * dlsize + n + 2;
        let mut bytes = vec![0u8; end_address * 8];

        // Record 0 (one-based start address 1): reference epoch 0.
        write_type21_record(
            byte_order,
            &mut bytes,
            1,
            maxdim,
            0.0,
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
        );
        // Record 1: reference epoch 10.
        write_type21_record(
            byte_order,
            &mut bytes,
            1 + dlsize,
            maxdim,
            10.0,
            [1000.0, 2000.0, 3000.0],
            [10.0, 20.0, 30.0],
        );

        // Per-record epochs follow the difference lines.
        let first_epoch = n * dlsize + 1;
        write_f64_address(byte_order, &mut bytes, first_epoch, 10.0);
        write_f64_address(byte_order, &mut bytes, first_epoch + 1, 20.0);

        // Trailer: MAXDIM then N.
        write_f64_address(byte_order, &mut bytes, end_address - 1, maxdim as f64);
        write_f64_address(byte_order, &mut bytes, end_address, n as f64);

        (
            bytes,
            SpkSegmentDescriptor {
                name: "TYPE 21 TEST".to_string(),
                start_et: 0.0,
                stop_et: 20.0,
                target: 2000001,
                center: 10,
                frame: 1,
                data_type: 21,
                start_address: 1,
                end_address: end_address as i32,
            },
        )
    }

    /// Write one type-21 difference line with a zero difference table, reducing
    /// evaluation to linear motion `refpos + delta * refvel`.
    fn write_type21_record(
        byte_order: DafByteOrder,
        bytes: &mut [u8],
        start_address: usize,
        maxdim: usize,
        tl: f64,
        refpos: [f64; 3],
        refvel: [f64; 3],
    ) {
        // The difference line maps to CSPICE RECORD(2..1+DLSIZE); start_address
        // is the one-based address of its first word (the reference epoch TL).
        write_f64_address(byte_order, bytes, start_address, tl);
        for offset in 0..maxdim {
            // Stepsize vector G: nonzero to avoid a divide-by-zero guard trip.
            write_f64_address(byte_order, bytes, start_address + 1 + offset, 1.0);
        }
        // Reference position/velocity, interleaved x, x', y, y', z, z'.
        write_f64_address(byte_order, bytes, start_address + maxdim + 1, refpos[0]);
        write_f64_address(byte_order, bytes, start_address + maxdim + 2, refvel[0]);
        write_f64_address(byte_order, bytes, start_address + maxdim + 3, refpos[1]);
        write_f64_address(byte_order, bytes, start_address + maxdim + 4, refvel[1]);
        write_f64_address(byte_order, bytes, start_address + maxdim + 5, refpos[2]);
        write_f64_address(byte_order, bytes, start_address + maxdim + 6, refvel[2]);
        // Modified divided difference arrays (MAXDIM * 3) left zero.
        // KQMAX1 then the per-component integration order array KQ.
        write_f64_address(byte_order, bytes, start_address + 4 * maxdim + 7, 2.0);
        write_f64_address(byte_order, bytes, start_address + 4 * maxdim + 8, 1.0);
        write_f64_address(byte_order, bytes, start_address + 4 * maxdim + 9, 1.0);
        write_f64_address(byte_order, bytes, start_address + 4 * maxdim + 10, 1.0);
    }

    fn write_type3_constant_segment(
        byte_order: DafByteOrder,
        bytes: &mut [u8],
        start_address: usize,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
    ) {
        let rsize = 8usize;
        let end_address = start_address + rsize + 3;
        write_f64_address(byte_order, bytes, start_address, 5.0);
        write_f64_address(byte_order, bytes, start_address + 1, 5.0);
        write_f64_address(byte_order, bytes, start_address + 2, position_km[0]);
        write_f64_address(byte_order, bytes, start_address + 3, position_km[1]);
        write_f64_address(byte_order, bytes, start_address + 4, position_km[2]);
        write_f64_address(byte_order, bytes, start_address + 5, velocity_km_s[0]);
        write_f64_address(byte_order, bytes, start_address + 6, velocity_km_s[1]);
        write_f64_address(byte_order, bytes, start_address + 7, velocity_km_s[2]);
        write_f64_address(byte_order, bytes, end_address - 3, 0.0);
        write_f64_address(byte_order, bytes, end_address - 2, 10.0);
        write_f64_address(byte_order, bytes, end_address - 1, rsize as f64);
        write_f64_address(byte_order, bytes, end_address, 1.0);
    }

    fn write_type2_constant_segment(
        byte_order: DafByteOrder,
        bytes: &mut [u8],
        start_address: usize,
        position_km: [f64; 3],
    ) {
        let rsize = 5usize;
        let end_address = start_address + rsize + 3;
        write_f64_address(byte_order, bytes, start_address, 5.0);
        write_f64_address(byte_order, bytes, start_address + 1, 5.0);
        write_f64_address(byte_order, bytes, start_address + 2, position_km[0]);
        write_f64_address(byte_order, bytes, start_address + 3, position_km[1]);
        write_f64_address(byte_order, bytes, start_address + 4, position_km[2]);
        write_f64_address(byte_order, bytes, end_address - 3, 0.0);
        write_f64_address(byte_order, bytes, end_address - 2, 10.0);
        write_f64_address(byte_order, bytes, end_address - 1, rsize as f64);
        write_f64_address(byte_order, bytes, end_address, 1.0);
    }

    fn write_type3_record(
        byte_order: DafByteOrder,
        bytes: &mut [u8],
        start_address: usize,
        mid: f64,
        radius: f64,
        coeffs: [[f64; 2]; 6],
    ) {
        write_f64_address(byte_order, bytes, start_address, mid);
        write_f64_address(byte_order, bytes, start_address + 1, radius);
        let mut address = start_address + 2;
        for component in coeffs {
            for coeff in component {
                write_f64_address(byte_order, bytes, address, coeff);
                address += 1;
            }
        }
    }

    fn write_type2_record(
        byte_order: DafByteOrder,
        bytes: &mut [u8],
        start_address: usize,
        mid: f64,
        radius: f64,
        coeffs: [[f64; 3]; 3],
    ) {
        write_f64_address(byte_order, bytes, start_address, mid);
        write_f64_address(byte_order, bytes, start_address + 1, radius);
        let mut address = start_address + 2;
        for component in coeffs {
            for coeff in component {
                write_f64_address(byte_order, bytes, address, coeff);
                address += 1;
            }
        }
    }

    fn write_f64_address(byte_order: DafByteOrder, bytes: &mut [u8], address: usize, value: f64) {
        write_f64(byte_order, bytes, (address - 1) * 8, value);
    }

    fn assert_position_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-12,
                "axis {axis}: actual {:?}, expected {:?}",
                actual,
                expected
            );
        }
    }

    fn assert_state_close(
        actual: SpkStateVector,
        expected_position: [f64; 3],
        expected_velocity: [f64; 3],
    ) {
        assert_position_close(actual.position_km, expected_position);
        assert_position_close(actual.velocity_km_s, expected_velocity);
    }

    fn assert_query_state_close(
        actual: SpkState,
        expected_position: [f64; 3],
        expected_velocity: [f64; 3],
        expected_frame: i32,
    ) {
        assert_position_close(actual.position_km, expected_position);
        assert_position_close(actual.velocity_km_s, expected_velocity);
        assert_eq!(actual.frame, expected_frame);
    }

    /// A synthetic single-record segment for [`build_test_spk`].
    struct TestSegment {
        target: i32,
        center: i32,
        frame: i32,
        data_type: i32,
        start_et: f64,
        stop_et: f64,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
    }

    impl TestSegment {
        /// A constant type-3 segment covering [0, 10] in frame 1.
        fn type3(target: i32, center: i32, position_km: [f64; 3], velocity_km_s: [f64; 3]) -> Self {
            Self {
                target,
                center,
                frame: 1,
                data_type: SPK_TYPE_3,
                start_et: 0.0,
                stop_et: 10.0,
                position_km,
                velocity_km_s,
            }
        }

        fn covering(mut self, start_et: f64, stop_et: f64) -> Self {
            self.start_et = start_et;
            self.stop_et = stop_et;
            self
        }

        /// Keep the type-3 data words but declare another data type.
        fn with_data_type(mut self, data_type: i32) -> Self {
            self.data_type = data_type;
            self
        }
    }

    /// Build a little-endian SPK with one summary record, one name record, and
    /// one constant type-3 data block per segment, in the order given.
    fn build_test_spk(name: &str, segments: &[TestSegment]) -> Vec<u8> {
        let byte_order = DafByteOrder::LittleEndian;
        let words_per_segment = 12usize;
        let first_address = 513usize;
        let end_address = first_address + segments.len() * words_per_segment - 1;
        let mut bytes = vec![0u8; end_address.max(first_address) * 8];

        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        write_i32(byte_order, &mut bytes, 8, 2);
        write_i32(byte_order, &mut bytes, 12, 6);
        write_ascii(&mut bytes, 16, DAF_INTERNAL_NAME_BYTES, name);
        write_i32(byte_order, &mut bytes, 76, 3);
        write_i32(byte_order, &mut bytes, 80, 3);
        write_i32(byte_order, &mut bytes, 84, (end_address + 1) as i32);
        bytes[DAF_BINARY_FORMAT_OFFSET..DAF_BINARY_FORMAT_OFFSET + 8].copy_from_slice(b"LTL-IEEE");

        let summary_offset = DAF_RECORD_BYTES * 2;
        let name_offset = DAF_RECORD_BYTES * 3;
        write_f64(byte_order, &mut bytes, summary_offset, 0.0);
        write_f64(byte_order, &mut bytes, summary_offset + 8, 0.0);
        write_f64(
            byte_order,
            &mut bytes,
            summary_offset + 16,
            segments.len() as f64,
        );

        for (index, segment) in segments.iter().enumerate() {
            let start = first_address + index * words_per_segment;
            write_summary(
                byte_order,
                &mut bytes,
                summary_offset + 24 + index * 40,
                segment.start_et,
                segment.stop_et,
                [
                    segment.target,
                    segment.center,
                    segment.frame,
                    segment.data_type,
                    start as i32,
                    (start + words_per_segment - 1) as i32,
                ],
            );
            write_ascii(
                &mut bytes,
                name_offset + index * 40,
                40,
                &format!("BODY {} TO {}", segment.target, segment.center),
            );
            write_type3_constant_segment(
                byte_order,
                &mut bytes,
                start,
                segment.position_km,
                segment.velocity_km_s,
            );
        }

        bytes
    }

    fn write_summary(
        byte_order: DafByteOrder,
        bytes: &mut [u8],
        offset: usize,
        start_et: f64,
        stop_et: f64,
        ints: [i32; 6],
    ) {
        write_f64(byte_order, bytes, offset, start_et);
        write_f64(byte_order, bytes, offset + 8, stop_et);
        for (index, value) in ints.into_iter().enumerate() {
            write_i32(byte_order, bytes, offset + 16 + index * 4, value);
        }
    }

    fn write_ascii(bytes: &mut [u8], offset: usize, len: usize, text: &str) {
        bytes[offset..offset + len].fill(b' ');
        bytes[offset..offset + text.len()].copy_from_slice(text.as_bytes());
    }

    fn write_i32(byte_order: DafByteOrder, bytes: &mut [u8], offset: usize, value: i32) {
        let word = match byte_order {
            DafByteOrder::LittleEndian => value.to_le_bytes(),
            DafByteOrder::BigEndian => value.to_be_bytes(),
        };
        bytes[offset..offset + 4].copy_from_slice(&word);
    }

    fn write_f64(byte_order: DafByteOrder, bytes: &mut [u8], offset: usize, value: f64) {
        let word = match byte_order {
            DafByteOrder::LittleEndian => value.to_le_bytes(),
            DafByteOrder::BigEndian => value.to_be_bytes(),
        };
        bytes[offset..offset + 8].copy_from_slice(&word);
    }
}
