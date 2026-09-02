//! Memory-mappable precise-ephemeris interpolant store.
//!
//! The store is the offline form of [`PreciseEphemerisInterpolant`]: a fixed
//! header, a sorted satellite index, and one aligned payload per satellite.
//! Payloads carry SP3-native position nodes and fitted clock spline coefficients
//! so opening the store validates bytes and builds only lightweight indexes. It
//! never refits clock splines at open or during evaluation.

use crate::artifact_bytes::{ArtifactBytes, DigestProvenance};
use std::collections::BTreeMap;
use std::fs;
use std::mem;
use std::path::{Path, PathBuf};

use crate::astro::time::model::{Instant, TimeScale};
use crate::constants::{KM_TO_M, OMEGA_E_DOT_RAD_S, US_TO_S};
use crate::frame::ItrfPositionM;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{
    ObservableEphemerisSource, ObservableState, ObservableStateBatch, ObservablesError,
};
use crate::sp3::interp::{instant_to_j2000_seconds, neville, NEVILLE_POINTS};
use crate::sp3::{PreciseEphemerisInterpolant, Sp3, Sp3State};
use crate::{validate, Error, Result};

const STORE_MAGIC: &[u8; 8] = b"PEMAP001";
const STORE_VERSION: u16 = 1;
const STORE_ALIGNMENT: usize = 4096;
const STORE_HEADER_LEN: usize = 64;
const SAT_INDEX_RECORD_LEN: usize = 96;
const CLOCK_NODE_RECORD_LEN: usize = 24;
const CLOCK_ARC_RECORD_LEN: usize = 64;

const HEADER_VERSION_OFFSET: usize = 8;
const HEADER_TIME_SCALE_OFFSET: usize = 10;
const HEADER_SAT_COUNT_OFFSET: usize = 12;
const HEADER_INDEX_OFFSET_OFFSET: usize = 16;
const HEADER_DATA_OFFSET_OFFSET: usize = 24;
const HEADER_TOTAL_LEN_OFFSET: usize = 32;
const HEADER_CHECKSUM_OFFSET: usize = 40;

const SAT_SYSTEM_OFFSET: usize = 0;
const SAT_PRN_OFFSET: usize = 1;
const SAT_POS_COUNT_OFFSET: usize = 4;
const SAT_CLOCK_NODE_COUNT_OFFSET: usize = 8;
const SAT_CLOCK_ARC_COUNT_OFFSET: usize = 12;
const SAT_POS_X_OFFSET_OFFSET: usize = 16;
const SAT_POS_KX_OFFSET_OFFSET: usize = 24;
const SAT_POS_KY_OFFSET_OFFSET: usize = 32;
const SAT_POS_KZ_OFFSET_OFFSET: usize = 40;
const SAT_CLOCK_NODE_OFFSET_OFFSET: usize = 48;
const SAT_CLOCK_ARC_OFFSET_OFFSET: usize = 56;
const SAT_DATA_OFFSET_OFFSET: usize = 64;
const SAT_DATA_LEN_OFFSET: usize = 72;
const SAT_CHECKSUM_OFFSET: usize = 80;

const CLOCK_NODE_X_OFFSET: usize = 0;
const CLOCK_NODE_US_OFFSET: usize = 8;
const CLOCK_NODE_EVENT_OFFSET: usize = 16;

const CLOCK_ARC_NODE_COUNT_OFFSET: usize = 0;
const CLOCK_ARC_COEFF_COUNT_OFFSET: usize = 4;
const CLOCK_ARC_X_OFFSET_OFFSET: usize = 8;
const CLOCK_ARC_C0_OFFSET_OFFSET: usize = 16;
const CLOCK_ARC_C1_OFFSET_OFFSET: usize = 24;
const CLOCK_ARC_C2_OFFSET_OFFSET: usize = 32;
const CLOCK_ARC_C3_OFFSET_OFFSET: usize = 40;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Errors from precise-interpolant store conversion, serialization, and open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreciseInterpolantStoreError {
    /// File I/O failed.
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// I/O error text.
        message: String,
    },
    /// Store bytes could not be parsed.
    Parse {
        /// Human-readable parse reason.
        reason: String,
    },
    /// The store version is not supported.
    UnsupportedVersion {
        /// Version tag found in the store header.
        version: u16,
    },
    /// The time-scale tag is not supported.
    UnsupportedTimeScale {
        /// Time-scale tag found in the store header.
        tag: u8,
    },
    /// The satellite-system tag is not supported.
    UnsupportedSatelliteSystem {
        /// Satellite-system tag found in an index record.
        tag: u8,
    },
    /// A satellite appears more than once in the index.
    DuplicateSatellite {
        /// Duplicated satellite id.
        sat: GnssSatelliteId,
    },
    /// The file-level checksum did not match the bytes opened.
    Checksum {
        /// Checksum stored in the header.
        expected: u64,
        /// Checksum computed from the byte span.
        found: u64,
    },
    /// A satellite payload checksum did not match its index record.
    SatelliteChecksum {
        /// Satellite whose payload failed verification.
        sat: GnssSatelliteId,
        /// Checksum stored in the satellite index record.
        expected: u64,
        /// Checksum computed from the satellite payload.
        found: u64,
    },
    /// An attested checksum did not match the checksum declared by the header.
    AttestedChecksumMismatch {
        /// Checksum asserted by the caller.
        claimed: u64,
        /// Checksum declared by the store header.
        declared: u64,
    },
}

impl core::fmt::Display for PreciseInterpolantStoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "{} failed: {message}", path.display()),
            Self::Parse { reason } => write!(f, "precise interpolant store parse error: {reason}"),
            Self::UnsupportedVersion { version } => {
                write!(
                    f,
                    "precise interpolant store version {version} is not supported"
                )
            }
            Self::UnsupportedTimeScale { tag } => {
                write!(
                    f,
                    "precise interpolant store time-scale tag {tag} is not supported"
                )
            }
            Self::UnsupportedSatelliteSystem { tag } => {
                write!(
                    f,
                    "precise interpolant store satellite-system tag {tag} is not supported"
                )
            }
            Self::DuplicateSatellite { sat } => {
                write!(f, "duplicate precise interpolant satellite {sat}")
            }
            Self::Checksum { expected, found } => write!(
                f,
                "precise interpolant store checksum expected {expected:#x} but found {found:#x}"
            ),
            Self::SatelliteChecksum {
                sat,
                expected,
                found,
            } => write!(
                f,
                "precise interpolant satellite {sat} checksum expected {expected:#x} but found {found:#x}"
            ),
            Self::AttestedChecksumMismatch { claimed, declared } => write!(
                f,
                "attested precise interpolant checksum {claimed:#x} does not match header checksum {declared:#x}"
            ),
        }
    }
}

impl std::error::Error for PreciseInterpolantStoreError {}

#[derive(Debug, Clone)]
enum F64Array<'a> {
    Borrowed(&'a [f64]),
    Offset { offset: usize, count: usize },
}

impl F64Array<'_> {
    #[cfg(feature = "mmap")]
    /// Promote an offset-backed array to `'static`.
    ///
    /// `Offset` carries no reference, so the value is independent of the byte
    /// lifetime. `Borrowed` is not promotable and returns `None`. This is what
    /// lets a memory-mapped reader own its bytes without becoming a
    /// self-referential struct or reaching for `unsafe`.
    fn into_static(self) -> Option<F64Array<'static>> {
        match self {
            Self::Borrowed(_) => None,
            Self::Offset { offset, count } => Some(F64Array::Offset { offset, count }),
        }
    }

    const fn len(&self) -> usize {
        match self {
            Self::Borrowed(values) => values.len(),
            Self::Offset { count, .. } => *count,
        }
    }

    fn get(&self, bytes: &[u8], idx: usize) -> f64 {
        match self {
            Self::Borrowed(values) => values[idx],
            Self::Offset { offset, .. } => mapped_f64(bytes, *offset, idx),
        }
    }
}

#[derive(Debug, Clone)]
struct MmapClockArc<'a> {
    x: F64Array<'a>,
    c0: F64Array<'a>,
    c1: F64Array<'a>,
    c2: F64Array<'a>,
    c3: F64Array<'a>,
}

impl MmapClockArc<'_> {
    #[cfg(feature = "mmap")]
    fn into_static(self) -> Option<MmapClockArc<'static>> {
        Some(MmapClockArc {
            x: self.x.into_static()?,
            c0: self.c0.into_static()?,
            c1: self.c1.into_static()?,
            c2: self.c2.into_static()?,
            c3: self.c3.into_static()?,
        })
    }

    fn node_count(&self) -> usize {
        self.x.len()
    }

    fn coeff_count(&self) -> usize {
        self.c0.len()
    }
}

#[derive(Debug, Clone)]
struct MmapSeries<'a> {
    pos_count: usize,
    clock_node_count: usize,
    pos_x: F64Array<'a>,
    pos_kx: F64Array<'a>,
    pos_ky: F64Array<'a>,
    pos_kz: F64Array<'a>,
    clock_arcs: Vec<MmapClockArc<'a>>,
}

impl MmapSeries<'_> {
    #[cfg(feature = "mmap")]
    fn into_static(self) -> Option<MmapSeries<'static>> {
        Some(MmapSeries {
            pos_count: self.pos_count,
            clock_node_count: self.clock_node_count,
            pos_x: self.pos_x.into_static()?,
            pos_kx: self.pos_kx.into_static()?,
            pos_ky: self.pos_ky.into_static()?,
            pos_kz: self.pos_kz.into_static()?,
            clock_arcs: self
                .clock_arcs
                .into_iter()
                .map(MmapClockArc::into_static)
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

#[derive(Debug)]
struct ParsedStore<'a> {
    time_scale: TimeScale,
    satellites: Vec<GnssSatelliteId>,
    series: BTreeMap<GnssSatelliteId, MmapSeries<'a>>,
}

#[derive(Clone, Copy)]
enum ArrayBacking<'a> {
    Borrowed(&'a [u8]),
    Offset,
}

#[derive(Clone, Copy)]
enum ChecksumValidation {
    Verified,
    Attested(u64),
}

impl ChecksumValidation {
    const fn digest_provenance(self) -> DigestProvenance {
        match self {
            Self::Verified => DigestProvenance::Verified,
            Self::Attested(_) => DigestProvenance::Attested,
        }
    }

    const fn attested_checksum64(self) -> Option<u64> {
        match self {
            Self::Verified => None,
            Self::Attested(checksum64) => Some(checksum64),
        }
    }

    const fn verifies_payloads(self) -> bool {
        matches!(self, Self::Verified)
    }
}

/// Evaluation-only precise-ephemeris interpolant backed by store bytes.
///
/// [`Self::from_path`] reads the artifact and validates its checksum on open.
/// [`Self::from_bytes`] accepts caller-managed bytes, including an mmap slice
/// from an application-owned mapping. Both paths parse only fixed metadata;
/// clock spline coefficients are consumed from the artifact as written.
pub struct MmapPreciseEphemerisInterpolant<'a> {
    bytes: ArtifactBytes<'a>,
    time_scale: TimeScale,
    satellites: Vec<GnssSatelliteId>,
    series: BTreeMap<GnssSatelliteId, MmapSeries<'a>>,
    digest_provenance: DigestProvenance,
    attested_checksum64: Option<u64>,
}

impl core::fmt::Debug for MmapPreciseEphemerisInterpolant<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MmapPreciseEphemerisInterpolant")
            .field("byte_len", &self.bytes.as_slice().len())
            .field("time_scale", &self.time_scale)
            .field("satellites", &self.satellites)
            .field("digest_provenance", &self.digest_provenance)
            .field("attested_checksum64", &self.attested_checksum64)
            .finish_non_exhaustive()
    }
}

impl MmapPreciseEphemerisInterpolant<'static> {
    /// Parse an owned precise-interpolant store byte vector.
    pub fn from_vec(bytes: Vec<u8>) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        Self::from_backing(
            ArtifactBytes::Owned(bytes),
            ArrayBacking::Offset,
            ChecksumValidation::Verified,
        )
    }

    /// Parse owned precise-interpolant store bytes using a caller-attested
    /// checksum.
    ///
    /// The byte-based counterpart of [`Self::from_path_attested`], with the
    /// same contract, including the O(8) cross-check of the claim against the
    /// header's declared checksum, which fails closed with
    /// [`PreciseInterpolantStoreError::AttestedChecksumMismatch`].
    pub fn from_vec_attested(
        bytes: Vec<u8>,
        claimed_checksum64: u64,
    ) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        Self::from_backing(
            ArtifactBytes::Owned(bytes),
            ArrayBacking::Offset,
            ChecksumValidation::Attested(claimed_checksum64),
        )
    }

    /// Open and parse a precise-interpolant store file.
    ///
    /// With the `mmap` feature the file is memory-mapped read-only and this
    /// reader owns the mapping; without it the file is read into memory. The
    /// entry point is the same either way, so enabling the feature speeds up
    /// every existing caller rather than asking anyone to migrate.
    ///
    /// The mapped parse uses offset-backed arrays rather than borrowed ones, so
    /// the reader owns its bytes without becoming a self-referential struct.
    pub fn from_path(
        path: impl AsRef<Path>,
    ) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        let path = path.as_ref();

        #[cfg(feature = "mmap")]
        {
            let bytes = crate::artifact_bytes::map_file_read_only(path).map_err(|err| {
                PreciseInterpolantStoreError::Io {
                    path: path.to_path_buf(),
                    message: err.to_string(),
                }
            })?;
            Self::from_mapped_backing(bytes, ChecksumValidation::Verified)
        }

        #[cfg(not(feature = "mmap"))]
        {
            let bytes = fs::read(path).map_err(|err| PreciseInterpolantStoreError::Io {
                path: path.to_path_buf(),
                message: err.to_string(),
            })?;
            Self::from_vec(bytes)
        }
    }

    /// Open a precise-interpolant store using a caller-attested checksum.
    ///
    /// This performs the same header, index, dimension, length, and payload
    /// layout validation as [`Self::from_path`] without recomputing file-level
    /// or per-satellite checksums. The claim must match the checksum declared
    /// by the header. A mismatch is an error and never falls back to hashing.
    /// With the `mmap` feature the file is memory-mapped read-only; without it
    /// the file is read into memory.
    pub fn from_path_attested(
        path: impl AsRef<Path>,
        claimed_checksum64: u64,
    ) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        let path = path.as_ref();

        #[cfg(feature = "mmap")]
        {
            let bytes = crate::artifact_bytes::map_file_read_only(path).map_err(|err| {
                PreciseInterpolantStoreError::Io {
                    path: path.to_path_buf(),
                    message: err.to_string(),
                }
            })?;
            Self::from_mapped_backing(bytes, ChecksumValidation::Attested(claimed_checksum64))
        }

        #[cfg(not(feature = "mmap"))]
        {
            let bytes = fs::read(path).map_err(|err| PreciseInterpolantStoreError::Io {
                path: path.to_path_buf(),
                message: err.to_string(),
            })?;
            Self::from_backing(
                ArtifactBytes::Owned(bytes),
                ArrayBacking::Offset,
                ChecksumValidation::Attested(claimed_checksum64),
            )
        }
    }

    #[cfg(feature = "mmap")]
    fn from_mapped_backing(
        bytes: ArtifactBytes<'static>,
        checksum_validation: ChecksumValidation,
    ) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        let parsed = parse_store(bytes.as_slice(), ArrayBacking::Offset, checksum_validation)?;
        // invariant: offset-backed parsing owns its backing bytes, so every
        // parsed series can be promoted to the store's static lifetime.
        #[allow(clippy::expect_used)]
        let series = parsed
            .series
            .into_iter()
            .map(|(sat, series)| series.into_static().map(|series| (sat, series)))
            .collect::<Option<BTreeMap<_, _>>>()
            .expect("offset-backed parse yields no borrows");
        Ok(Self {
            bytes,
            time_scale: parsed.time_scale,
            satellites: parsed.satellites,
            series,
            digest_provenance: checksum_validation.digest_provenance(),
            attested_checksum64: checksum_validation.attested_checksum64(),
        })
    }
}

impl<'a> MmapPreciseEphemerisInterpolant<'a> {
    /// Parse a borrowed precise-interpolant store byte span.
    ///
    /// The reader keeps the byte span in place. F64 payload arrays are borrowed
    /// directly from the span, so callers that pass an mmap-backed slice get a
    /// zero-copy reader.
    pub fn from_bytes(bytes: &'a [u8]) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        Self::from_backing(
            ArtifactBytes::Borrowed(bytes),
            ArrayBacking::Borrowed(bytes),
            ChecksumValidation::Verified,
        )
    }

    fn from_backing(
        bytes: ArtifactBytes<'a>,
        backing: ArrayBacking<'a>,
        checksum_validation: ChecksumValidation,
    ) -> core::result::Result<Self, PreciseInterpolantStoreError> {
        let parsed = parse_store(bytes.as_slice(), backing, checksum_validation)?;
        Ok(Self {
            bytes,
            time_scale: parsed.time_scale,
            satellites: parsed.satellites,
            series: parsed.series,
            digest_provenance: checksum_validation.digest_provenance(),
            attested_checksum64: checksum_validation.attested_checksum64(),
        })
    }

    /// Borrow the artifact bytes backing this reader.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Whether this reader is backed by a memory map rather than a copy in
    /// process memory.
    #[must_use]
    pub fn is_memory_mapped(&self) -> bool {
        self.bytes.is_memory_mapped()
    }

    /// Return who computed the checksum carried by this reader.
    #[must_use]
    pub const fn digest_provenance(&self) -> DigestProvenance {
        self.digest_provenance
    }

    /// Return the file-level checksum carried by this reader.
    ///
    /// Verified readers compute the checksum on demand. Attested readers return
    /// the caller's claim without hashing the byte span.
    #[must_use]
    // invariant: Attested readers are constructed only with a claimed checksum.
    #[allow(clippy::expect_used)]
    pub fn checksum64(&self) -> u64 {
        match self.digest_provenance {
            DigestProvenance::Verified => precise_interpolant_store_checksum64(self.bytes.as_ref()),
            DigestProvenance::Attested => self
                .attested_checksum64
                .expect("attested precise interpolant reader carries a checksum"),
        }
    }

    /// Re-verify the file-level and per-satellite payload checksums.
    ///
    /// Successful verification changes the digest provenance to
    /// [`DigestProvenance::Verified`].
    pub fn verify(&mut self) -> core::result::Result<(), PreciseInterpolantStoreError> {
        parse_store(
            self.bytes.as_ref(),
            ArrayBacking::Offset,
            ChecksumValidation::Verified,
        )?;
        self.digest_provenance = DigestProvenance::Verified;
        self.attested_checksum64 = None;
        Ok(())
    }

    /// The time scale of the stored epoch axis.
    #[must_use]
    pub const fn time_scale(&self) -> TimeScale {
        self.time_scale
    }

    /// The satellites present in the mapped artifact, in ascending order.
    #[must_use]
    pub fn satellites(&self) -> &[GnssSatelliteId] {
        &self.satellites
    }

    /// Interpolate the state of `sat` at an arbitrary J2000-second epoch.
    pub fn position_at_j2000_seconds(&self, sat: GnssSatelliteId, query: f64) -> Result<Sp3State> {
        let query = validate::finite(query, "query_j2000_s").map_err(map_query_input)?;
        let Some(series) = self.series.get(&sat) else {
            return Err(Error::UnknownSatellite(sat));
        };
        interpolate_mapped_state(self.bytes.as_ref(), series, query)
    }

    /// Interpolate the state of `sat` at an arbitrary [`Instant`].
    ///
    /// The query instant must use the same time scale as the source artifact.
    pub fn position(&self, sat: GnssSatelliteId, epoch: Instant) -> Result<Sp3State> {
        if epoch.scale != self.time_scale {
            return Err(Error::InvalidInput(format!(
                "mapped precise-interpolant query time scale {} does not match source time scale {}",
                epoch.scale.abbrev(),
                self.time_scale.abbrev()
            )));
        }
        let query = instant_to_j2000_seconds(&epoch).ok_or(Error::EpochOutOfRange)?;
        self.position_at_j2000_seconds(sat, query)
    }

    /// ECEF states for parallel satellite and epoch arrays.
    pub fn observable_states_at_j2000_s(
        &self,
        satellites: &[GnssSatelliteId],
        epochs_j2000_s: &[f64],
    ) -> core::result::Result<ObservableStateBatch, ObservablesError> {
        <Self as ObservableEphemerisSource>::observable_states_at_j2000_s(
            self,
            satellites,
            epochs_j2000_s,
        )
    }

    /// ECEF states for many satellites at one shared epoch.
    pub fn observable_states_at_shared_j2000_s(
        &self,
        satellites: &[GnssSatelliteId],
        epoch_j2000_s: f64,
    ) -> ObservableStateBatch {
        <Self as ObservableEphemerisSource>::observable_states_at_shared_j2000_s(
            self,
            satellites,
            epoch_j2000_s,
        )
    }
}

impl ObservableEphemerisSource for MmapPreciseEphemerisInterpolant<'_> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> core::result::Result<ObservableState, ObservablesError> {
        let state = self
            .position_at_j2000_seconds(sat, t_j2000_s)
            .map_err(ObservablesError::Ephemeris)?;
        Ok(ObservableState {
            position_ecef_m: state.position.as_array(),
            clock_s: state.clock_s,
        })
    }
}

impl PreciseEphemerisInterpolant {
    /// Serialize this fitted interpolant into canonical memory-mappable bytes.
    ///
    /// The output is deterministic for a deterministic source: satellites are
    /// sorted, offsets are fixed by the versioned layout, padding is zero-filled,
    /// and checksums are written after all payload bytes are finalized.
    pub fn to_mmap_store_bytes(
        &self,
    ) -> core::result::Result<Vec<u8>, PreciseInterpolantStoreError> {
        build_store(self)
    }

    /// Serialize this fitted interpolant into a store file.
    pub fn write_mmap_store(
        &self,
        output_path: impl AsRef<Path>,
    ) -> core::result::Result<(), PreciseInterpolantStoreError> {
        let bytes = self.to_mmap_store_bytes()?;
        let output_path = output_path.as_ref();
        fs::write(output_path, &bytes).map_err(|err| PreciseInterpolantStoreError::Io {
            path: output_path.to_path_buf(),
            message: err.to_string(),
        })
    }
}

impl Sp3 {
    /// Build the fitted precise-ephemeris interpolant artifact for this product.
    pub fn precise_interpolant_store_bytes(
        &self,
    ) -> core::result::Result<Vec<u8>, PreciseInterpolantStoreError> {
        PreciseEphemerisInterpolant::from_sp3(self).to_mmap_store_bytes()
    }

    /// Build and write the fitted precise-ephemeris interpolant artifact for
    /// this product.
    pub fn write_precise_interpolant_store(
        &self,
        output_path: impl AsRef<Path>,
    ) -> core::result::Result<(), PreciseInterpolantStoreError> {
        PreciseEphemerisInterpolant::from_sp3(self).write_mmap_store(output_path)
    }
}

/// Return the FNV-1a checksum for precise-interpolant store bytes.
///
/// The header checksum field is treated as zero during calculation. This is the
/// same value stored in the header of canonical artifacts.
#[must_use]
pub fn precise_interpolant_store_checksum64(bytes: &[u8]) -> u64 {
    artifact_checksum64(bytes)
}

// invariant: layouts are derived from the same validated source used to build the store.
#[allow(clippy::expect_used)]
fn build_store(
    source: &PreciseEphemerisInterpolant,
) -> core::result::Result<Vec<u8>, PreciseInterpolantStoreError> {
    let sat_count = source.node_series().len();
    let index_end = STORE_HEADER_LEN
        .checked_add(
            sat_count
                .checked_mul(SAT_INDEX_RECORD_LEN)
                .ok_or_else(|| parse_error("satellite index length overflows usize"))?,
        )
        .ok_or_else(|| parse_error("satellite index end overflows usize"))?;
    let data_offset = align_up(index_end, STORE_ALIGNMENT)?;

    let mut layouts = Vec::with_capacity(sat_count);
    let mut cursor = data_offset;
    for (&sat, fitted) in source.node_series() {
        cursor = align_up(cursor, STORE_ALIGNMENT)?;
        let data_offset = cursor;
        let series = &fitted.series;
        let pos_count = series.x.len();
        let clock_node_count = series.clk.len();
        let clock_arc_count = fitted.clock_arcs.len();

        let pos_x_offset = cursor;
        cursor = add_len(cursor, pos_count, 8)?;
        let pos_kx_offset = cursor;
        cursor = add_len(cursor, pos_count, 8)?;
        let pos_ky_offset = cursor;
        cursor = add_len(cursor, pos_count, 8)?;
        let pos_kz_offset = cursor;
        cursor = add_len(cursor, pos_count, 8)?;
        let clock_node_offset = cursor;
        cursor = add_len(cursor, clock_node_count, CLOCK_NODE_RECORD_LEN)?;
        let clock_arc_offset = cursor;
        cursor = add_len(cursor, clock_arc_count, CLOCK_ARC_RECORD_LEN)?;

        let mut arcs = Vec::with_capacity(clock_arc_count);
        for arc in &fitted.clock_arcs {
            let node_count = arc.x.len();
            let coeff_count = arc.c0.len();
            if arc.c1.len() != coeff_count
                || arc.c2.len() != coeff_count
                || arc.c3.len() != coeff_count
                || coeff_count != node_count.saturating_sub(1)
            {
                return Err(parse_error("clock arc coefficient shape is inconsistent"));
            }

            let x_offset = cursor;
            cursor = add_len(cursor, node_count, 8)?;
            let c0_offset = cursor;
            cursor = add_len(cursor, coeff_count, 8)?;
            let c1_offset = cursor;
            cursor = add_len(cursor, coeff_count, 8)?;
            let c2_offset = cursor;
            cursor = add_len(cursor, coeff_count, 8)?;
            let c3_offset = cursor;
            cursor = add_len(cursor, coeff_count, 8)?;
            arcs.push(PendingClockArcLayout {
                node_count,
                coeff_count,
                x_offset,
                c0_offset,
                c1_offset,
                c2_offset,
                c3_offset,
            });
        }

        layouts.push(PendingSatLayout {
            sat,
            data_offset,
            data_len: cursor - data_offset,
            pos_x_offset,
            pos_kx_offset,
            pos_ky_offset,
            pos_kz_offset,
            clock_node_offset,
            clock_arc_offset,
            arcs,
        });
    }

    let mut out = vec![0u8; cursor];
    out[..STORE_MAGIC.len()].copy_from_slice(STORE_MAGIC);
    write_u16(&mut out, HEADER_VERSION_OFFSET, STORE_VERSION);
    out[HEADER_TIME_SCALE_OFFSET] = time_scale_tag(source.time_scale());
    write_u32(
        &mut out,
        HEADER_SAT_COUNT_OFFSET,
        u32::try_from(sat_count).map_err(|_| parse_error("satellite count exceeds u32"))?,
    );
    write_u64(
        &mut out,
        HEADER_INDEX_OFFSET_OFFSET,
        STORE_HEADER_LEN as u64,
    );
    write_u64(&mut out, HEADER_DATA_OFFSET_OFFSET, data_offset as u64);
    write_u64(&mut out, HEADER_TOTAL_LEN_OFFSET, cursor as u64);

    for (idx, layout) in layouts.iter().enumerate() {
        let fitted = source
            .node_series()
            .get(&layout.sat)
            .expect("layout satellite came from source");
        let series = &fitted.series;
        let record_offset = STORE_HEADER_LEN + idx * SAT_INDEX_RECORD_LEN;
        let record = &mut out[record_offset..record_offset + SAT_INDEX_RECORD_LEN];
        record[SAT_SYSTEM_OFFSET] = layout.sat.system.letter() as u8;
        record[SAT_PRN_OFFSET] = layout.sat.prn;
        write_u32(
            record,
            SAT_POS_COUNT_OFFSET,
            u32::try_from(series.x.len()).map_err(|_| parse_error("position count exceeds u32"))?,
        );
        write_u32(
            record,
            SAT_CLOCK_NODE_COUNT_OFFSET,
            u32::try_from(series.clk.len())
                .map_err(|_| parse_error("clock node count exceeds u32"))?,
        );
        write_u32(
            record,
            SAT_CLOCK_ARC_COUNT_OFFSET,
            u32::try_from(fitted.clock_arcs.len())
                .map_err(|_| parse_error("clock arc count exceeds u32"))?,
        );
        write_u64(record, SAT_POS_X_OFFSET_OFFSET, layout.pos_x_offset as u64);
        write_u64(
            record,
            SAT_POS_KX_OFFSET_OFFSET,
            layout.pos_kx_offset as u64,
        );
        write_u64(
            record,
            SAT_POS_KY_OFFSET_OFFSET,
            layout.pos_ky_offset as u64,
        );
        write_u64(
            record,
            SAT_POS_KZ_OFFSET_OFFSET,
            layout.pos_kz_offset as u64,
        );
        write_u64(
            record,
            SAT_CLOCK_NODE_OFFSET_OFFSET,
            layout.clock_node_offset as u64,
        );
        write_u64(
            record,
            SAT_CLOCK_ARC_OFFSET_OFFSET,
            layout.clock_arc_offset as u64,
        );
        write_u64(record, SAT_DATA_OFFSET_OFFSET, layout.data_offset as u64);
        write_u64(record, SAT_DATA_LEN_OFFSET, layout.data_len as u64);

        write_f64_slice(&mut out, layout.pos_x_offset, &series.x);
        write_f64_slice(&mut out, layout.pos_kx_offset, &series.kx);
        write_f64_slice(&mut out, layout.pos_ky_offset, &series.ky);
        write_f64_slice(&mut out, layout.pos_kz_offset, &series.kz);
        for (node_idx, &(x, clock_us, event)) in series.clk.iter().enumerate() {
            let node_offset = layout.clock_node_offset + node_idx * CLOCK_NODE_RECORD_LEN;
            let node = &mut out[node_offset..node_offset + CLOCK_NODE_RECORD_LEN];
            write_f64(node, CLOCK_NODE_X_OFFSET, x);
            write_f64(node, CLOCK_NODE_US_OFFSET, clock_us);
            node[CLOCK_NODE_EVENT_OFFSET] = u8::from(event);
        }

        for (arc_idx, arc_layout) in layout.arcs.iter().enumerate() {
            let arc = &fitted.clock_arcs[arc_idx];
            let arc_offset = layout.clock_arc_offset + arc_idx * CLOCK_ARC_RECORD_LEN;
            let record = &mut out[arc_offset..arc_offset + CLOCK_ARC_RECORD_LEN];
            write_u32(
                record,
                CLOCK_ARC_NODE_COUNT_OFFSET,
                u32::try_from(arc_layout.node_count)
                    .map_err(|_| parse_error("clock arc node count exceeds u32"))?,
            );
            write_u32(
                record,
                CLOCK_ARC_COEFF_COUNT_OFFSET,
                u32::try_from(arc_layout.coeff_count)
                    .map_err(|_| parse_error("clock arc coefficient count exceeds u32"))?,
            );
            write_u64(
                record,
                CLOCK_ARC_X_OFFSET_OFFSET,
                arc_layout.x_offset as u64,
            );
            write_u64(
                record,
                CLOCK_ARC_C0_OFFSET_OFFSET,
                arc_layout.c0_offset as u64,
            );
            write_u64(
                record,
                CLOCK_ARC_C1_OFFSET_OFFSET,
                arc_layout.c1_offset as u64,
            );
            write_u64(
                record,
                CLOCK_ARC_C2_OFFSET_OFFSET,
                arc_layout.c2_offset as u64,
            );
            write_u64(
                record,
                CLOCK_ARC_C3_OFFSET_OFFSET,
                arc_layout.c3_offset as u64,
            );
            write_f64_slice(&mut out, arc_layout.x_offset, &arc.x);
            write_f64_slice(&mut out, arc_layout.c0_offset, &arc.c0);
            write_f64_slice(&mut out, arc_layout.c1_offset, &arc.c1);
            write_f64_slice(&mut out, arc_layout.c2_offset, &arc.c2);
            write_f64_slice(&mut out, arc_layout.c3_offset, &arc.c3);
        }

        let sat_checksum = fnv1a64(&out[layout.data_offset..layout.data_offset + layout.data_len]);
        let record = &mut out[record_offset..record_offset + SAT_INDEX_RECORD_LEN];
        write_u64(record, SAT_CHECKSUM_OFFSET, sat_checksum);
    }

    let checksum = artifact_checksum64(&out);
    write_u64(&mut out, HEADER_CHECKSUM_OFFSET, checksum);
    Ok(out)
}

#[derive(Debug)]
struct PendingSatLayout {
    sat: GnssSatelliteId,
    data_offset: usize,
    data_len: usize,
    pos_x_offset: usize,
    pos_kx_offset: usize,
    pos_ky_offset: usize,
    pos_kz_offset: usize,
    clock_node_offset: usize,
    clock_arc_offset: usize,
    arcs: Vec<PendingClockArcLayout>,
}

#[derive(Debug)]
struct PendingClockArcLayout {
    node_count: usize,
    coeff_count: usize,
    x_offset: usize,
    c0_offset: usize,
    c1_offset: usize,
    c2_offset: usize,
    c3_offset: usize,
}

fn parse_store<'a>(
    bytes: &[u8],
    backing: ArrayBacking<'a>,
    checksum_validation: ChecksumValidation,
) -> core::result::Result<ParsedStore<'a>, PreciseInterpolantStoreError> {
    if bytes.len() < STORE_HEADER_LEN {
        return Err(parse_error(format!(
            "store has {} bytes but needs at least {STORE_HEADER_LEN}",
            bytes.len()
        )));
    }
    if &bytes[..STORE_MAGIC.len()] != STORE_MAGIC {
        return Err(parse_error("missing precise interpolant store magic"));
    }
    let version = read_u16(bytes, HEADER_VERSION_OFFSET)?;
    if version != STORE_VERSION {
        return Err(PreciseInterpolantStoreError::UnsupportedVersion { version });
    }

    let expected_checksum = read_u64(bytes, HEADER_CHECKSUM_OFFSET)?;
    match checksum_validation {
        ChecksumValidation::Verified => {
            let found_checksum = artifact_checksum64(bytes);
            if expected_checksum != found_checksum {
                return Err(PreciseInterpolantStoreError::Checksum {
                    expected: expected_checksum,
                    found: found_checksum,
                });
            }
        }
        ChecksumValidation::Attested(claimed) if claimed != expected_checksum => {
            return Err(PreciseInterpolantStoreError::AttestedChecksumMismatch {
                claimed,
                declared: expected_checksum,
            });
        }
        ChecksumValidation::Attested(_) => {}
    }

    ensure_zero(bytes, 11, 12, "header reserved byte")?;
    ensure_zero(bytes, 48, STORE_HEADER_LEN, "header reserved bytes")?;
    let time_scale = time_scale_from_tag(bytes[HEADER_TIME_SCALE_OFFSET])?;
    let sat_count = read_u32(bytes, HEADER_SAT_COUNT_OFFSET)? as usize;
    let index_offset = read_u64(bytes, HEADER_INDEX_OFFSET_OFFSET)? as usize;
    let data_offset = read_u64(bytes, HEADER_DATA_OFFSET_OFFSET)? as usize;
    let total_len = read_u64(bytes, HEADER_TOTAL_LEN_OFFSET)? as usize;
    if total_len != bytes.len() {
        return Err(parse_error(format!(
            "header total length {total_len} does not match {}",
            bytes.len()
        )));
    }
    if index_offset != STORE_HEADER_LEN {
        return Err(parse_error(format!(
            "index offset must be {STORE_HEADER_LEN}, got {index_offset}"
        )));
    }

    let index_len = sat_count
        .checked_mul(SAT_INDEX_RECORD_LEN)
        .ok_or_else(|| parse_error("satellite index length overflows usize"))?;
    let index_end = index_offset
        .checked_add(index_len)
        .ok_or_else(|| parse_error("satellite index end overflows usize"))?;
    if index_end > bytes.len() {
        return Err(parse_error("satellite index extends past store length"));
    }
    let expected_data_offset = align_up(index_end, STORE_ALIGNMENT)?;
    if data_offset != expected_data_offset {
        return Err(parse_error(format!(
            "data offset must be {expected_data_offset}, got {data_offset}"
        )));
    }
    ensure_zero(bytes, index_end, data_offset, "index padding")?;

    let mut satellites = Vec::with_capacity(sat_count);
    let mut series = BTreeMap::new();
    let mut previous = None;
    let mut expected_next = data_offset;

    for idx in 0..sat_count {
        let record_offset = index_offset + idx * SAT_INDEX_RECORD_LEN;
        let record = &bytes[record_offset..record_offset + SAT_INDEX_RECORD_LEN];
        let sat = read_satellite(record)?;
        if previous.is_some_and(|prev| sat <= prev) {
            return Err(parse_error(
                "satellite index records are not strictly sorted",
            ));
        }
        previous = Some(sat);

        ensure_zero(record, 2, 4, "satellite index reserved bytes")?;
        ensure_zero(
            record,
            88,
            SAT_INDEX_RECORD_LEN,
            "satellite index reserved bytes",
        )?;

        let pos_count = read_u32(record, SAT_POS_COUNT_OFFSET)? as usize;
        let clock_node_count = read_u32(record, SAT_CLOCK_NODE_COUNT_OFFSET)? as usize;
        let clock_arc_count = read_u32(record, SAT_CLOCK_ARC_COUNT_OFFSET)? as usize;
        if pos_count < 2 {
            return Err(parse_error(format!(
                "satellite {sat} has invalid position node count {pos_count}"
            )));
        }

        let pos_x_offset = read_u64(record, SAT_POS_X_OFFSET_OFFSET)? as usize;
        let pos_kx_offset = read_u64(record, SAT_POS_KX_OFFSET_OFFSET)? as usize;
        let pos_ky_offset = read_u64(record, SAT_POS_KY_OFFSET_OFFSET)? as usize;
        let pos_kz_offset = read_u64(record, SAT_POS_KZ_OFFSET_OFFSET)? as usize;
        let clock_node_offset = read_u64(record, SAT_CLOCK_NODE_OFFSET_OFFSET)? as usize;
        let clock_arc_offset = read_u64(record, SAT_CLOCK_ARC_OFFSET_OFFSET)? as usize;
        let sat_data_offset = read_u64(record, SAT_DATA_OFFSET_OFFSET)? as usize;
        let sat_data_len = read_u64(record, SAT_DATA_LEN_OFFSET)? as usize;
        let expected_sat_data_offset = align_up(expected_next, STORE_ALIGNMENT)?;
        ensure_zero(
            bytes,
            expected_next,
            expected_sat_data_offset,
            "satellite padding",
        )?;
        if sat_data_offset != expected_sat_data_offset {
            return Err(parse_error(format!(
                "satellite {sat} data offset must be {expected_sat_data_offset}, got {sat_data_offset}"
            )));
        }
        let sat_data_end = sat_data_offset
            .checked_add(sat_data_len)
            .ok_or_else(|| parse_error(format!("satellite {sat} data end overflows usize")))?;
        if sat_data_end > bytes.len() {
            return Err(parse_error(format!(
                "satellite {sat} data extends past store length"
            )));
        }

        let sat_checksum = read_u64(record, SAT_CHECKSUM_OFFSET)?;
        if checksum_validation.verifies_payloads() {
            let found_sat_checksum = fnv1a64(&bytes[sat_data_offset..sat_data_end]);
            if sat_checksum != found_sat_checksum {
                return Err(PreciseInterpolantStoreError::SatelliteChecksum {
                    sat,
                    expected: sat_checksum,
                    found: found_sat_checksum,
                });
            }
        }

        let mut cursor = sat_data_offset;
        require_offset(sat, "position x", pos_x_offset, cursor)?;
        let pos_x = parse_f64_array(bytes, pos_x_offset, pos_count, sat, "position x", backing)?;
        validate_strictly_increasing_f64_array(bytes, &pos_x, sat, "position x")?;
        cursor = add_len(cursor, pos_count, 8)?;
        require_offset(sat, "position kx", pos_kx_offset, cursor)?;
        let pos_kx = parse_f64_array(bytes, pos_kx_offset, pos_count, sat, "position kx", backing)?;
        cursor = add_len(cursor, pos_count, 8)?;
        require_offset(sat, "position ky", pos_ky_offset, cursor)?;
        let pos_ky = parse_f64_array(bytes, pos_ky_offset, pos_count, sat, "position ky", backing)?;
        cursor = add_len(cursor, pos_count, 8)?;
        require_offset(sat, "position kz", pos_kz_offset, cursor)?;
        let pos_kz = parse_f64_array(bytes, pos_kz_offset, pos_count, sat, "position kz", backing)?;
        cursor = add_len(cursor, pos_count, 8)?;

        require_offset(sat, "clock nodes", clock_node_offset, cursor)?;
        for node_idx in 0..clock_node_count {
            let node_offset = clock_node_offset + node_idx * CLOCK_NODE_RECORD_LEN;
            let node = bytes
                .get(node_offset..node_offset + CLOCK_NODE_RECORD_LEN)
                .ok_or_else(|| parse_error(format!("satellite {sat} clock node out of bounds")))?;
            let x = read_f64(node, CLOCK_NODE_X_OFFSET)?;
            let clock_us = read_f64(node, CLOCK_NODE_US_OFFSET)?;
            if !x.is_finite() || !clock_us.is_finite() {
                return Err(parse_error(format!(
                    "satellite {sat} clock node {node_idx} is not finite"
                )));
            }
            match node[CLOCK_NODE_EVENT_OFFSET] {
                0 | 1 => {}
                tag => {
                    return Err(parse_error(format!(
                        "satellite {sat} clock node {node_idx} has invalid event tag {tag}"
                    )));
                }
            }
            ensure_zero(
                node,
                CLOCK_NODE_EVENT_OFFSET + 1,
                CLOCK_NODE_RECORD_LEN,
                "clock node reserved bytes",
            )?;
        }
        cursor = add_len(cursor, clock_node_count, CLOCK_NODE_RECORD_LEN)?;

        require_offset(sat, "clock arc index", clock_arc_offset, cursor)?;
        let clock_arc_index_end = add_len(cursor, clock_arc_count, CLOCK_ARC_RECORD_LEN)?;
        let mut arc_cursor = clock_arc_index_end;
        let mut arcs = Vec::with_capacity(clock_arc_count);
        for arc_idx in 0..clock_arc_count {
            let arc_offset = clock_arc_offset + arc_idx * CLOCK_ARC_RECORD_LEN;
            let arc_record = &bytes[arc_offset..arc_offset + CLOCK_ARC_RECORD_LEN];
            let node_count = read_u32(arc_record, CLOCK_ARC_NODE_COUNT_OFFSET)? as usize;
            let coeff_count = read_u32(arc_record, CLOCK_ARC_COEFF_COUNT_OFFSET)? as usize;
            if node_count == 0 {
                return Err(parse_error(format!(
                    "satellite {sat} clock arc {arc_idx} is empty"
                )));
            }
            if coeff_count != node_count.saturating_sub(1) {
                return Err(parse_error(format!(
                    "satellite {sat} clock arc {arc_idx} coefficient count {coeff_count} does not match node count {node_count}"
                )));
            }
            let x_offset = read_u64(arc_record, CLOCK_ARC_X_OFFSET_OFFSET)? as usize;
            let c0_offset = read_u64(arc_record, CLOCK_ARC_C0_OFFSET_OFFSET)? as usize;
            let c1_offset = read_u64(arc_record, CLOCK_ARC_C1_OFFSET_OFFSET)? as usize;
            let c2_offset = read_u64(arc_record, CLOCK_ARC_C2_OFFSET_OFFSET)? as usize;
            let c3_offset = read_u64(arc_record, CLOCK_ARC_C3_OFFSET_OFFSET)? as usize;
            ensure_zero(
                arc_record,
                CLOCK_ARC_C3_OFFSET_OFFSET + 8,
                CLOCK_ARC_RECORD_LEN,
                "clock arc reserved bytes",
            )?;

            require_offset(sat, "clock arc x", x_offset, arc_cursor)?;
            let x = parse_f64_array(bytes, x_offset, node_count, sat, "clock arc x", backing)?;
            validate_strictly_increasing_f64_array(bytes, &x, sat, "clock arc x")?;
            arc_cursor = add_len(arc_cursor, node_count, 8)?;
            require_offset(sat, "clock arc c0", c0_offset, arc_cursor)?;
            let c0 = parse_f64_array(bytes, c0_offset, coeff_count, sat, "clock arc c0", backing)?;
            arc_cursor = add_len(arc_cursor, coeff_count, 8)?;
            require_offset(sat, "clock arc c1", c1_offset, arc_cursor)?;
            let c1 = parse_f64_array(bytes, c1_offset, coeff_count, sat, "clock arc c1", backing)?;
            arc_cursor = add_len(arc_cursor, coeff_count, 8)?;
            require_offset(sat, "clock arc c2", c2_offset, arc_cursor)?;
            let c2 = parse_f64_array(bytes, c2_offset, coeff_count, sat, "clock arc c2", backing)?;
            arc_cursor = add_len(arc_cursor, coeff_count, 8)?;
            require_offset(sat, "clock arc c3", c3_offset, arc_cursor)?;
            let c3 = parse_f64_array(bytes, c3_offset, coeff_count, sat, "clock arc c3", backing)?;
            arc_cursor = add_len(arc_cursor, coeff_count, 8)?;

            arcs.push(MmapClockArc { x, c0, c1, c2, c3 });
        }

        if sat_data_end != arc_cursor {
            return Err(parse_error(format!(
                "satellite {sat} data length must be {}, got {sat_data_len}",
                arc_cursor - sat_data_offset
            )));
        }

        let inserted = series.insert(
            sat,
            MmapSeries {
                pos_count,
                clock_node_count,
                pos_x,
                pos_kx,
                pos_ky,
                pos_kz,
                clock_arcs: arcs,
            },
        );
        if inserted.is_some() {
            return Err(PreciseInterpolantStoreError::DuplicateSatellite { sat });
        }
        satellites.push(sat);
        expected_next = sat_data_end;
    }

    if expected_next != bytes.len() {
        return Err(parse_error(format!(
            "store has trailing bytes: expected length {expected_next}, got {}",
            bytes.len()
        )));
    }

    Ok(ParsedStore {
        time_scale,
        satellites,
        series,
    })
}

// invariant: validated mapped payloads contain finite ITRF coordinates.
#[allow(clippy::expect_used)]
fn interpolate_mapped_state(bytes: &[u8], series: &MmapSeries, query: f64) -> Result<Sp3State> {
    if series.pos_count < 2 {
        return Err(Error::EpochOutOfRange);
    }

    let nominal = nominal_positive_spacing(bytes, series).ok_or(Error::EpochOutOfRange)?;
    let first = series.pos_x.get(bytes, 0);
    let last = series.pos_x.get(bytes, series.pos_count - 1);
    if query < first - nominal || query > last + nominal {
        return Err(Error::EpochOutOfRange);
    }

    let gap_thresh = 1.5 * nominal;
    let mut bi = 0usize;
    while bi + 1 < series.pos_count && series.pos_x.get(bytes, bi + 1) <= query {
        bi += 1;
    }
    if bi + 1 < series.pos_count {
        let lo = series.pos_x.get(bytes, bi);
        let hi = series.pos_x.get(bytes, bi + 1);
        if hi - lo > gap_thresh && query > lo + nominal && query < hi - nominal {
            return Err(Error::EpochOutOfRange);
        }
    }

    let (x_m, y_m, z_m) = interpolate_mapped_position_neville(bytes, series, query);
    let clock_s = interpolate_mapped_clock(bytes, series, query);
    Ok(Sp3State {
        position: ItrfPositionM::new(x_m, y_m, z_m).expect("valid ITRF position"),
        clock_s,
        velocity: None,
        clock_rate_s_s: None,
        flags: crate::sp3::Sp3Flags::default(),
    })
}

fn interpolate_mapped_position_neville(
    bytes: &[u8],
    series: &MmapSeries,
    query: f64,
) -> (f64, f64, f64) {
    let n = series.pos_count;
    let nominal = nominal_positive_spacing(bytes, series).unwrap_or(1.0);
    let gap_thresh = 1.5 * nominal;

    let mut pivot = 0usize;
    while pivot + 1 < n && series.pos_x.get(bytes, pivot + 1) <= query {
        pivot += 1;
    }
    if pivot + 1 < n {
        let x_pivot = series.pos_x.get(bytes, pivot);
        let x_next = series.pos_x.get(bytes, pivot + 1);
        if (x_next - x_pivot) > gap_thresh && query >= x_next - nominal {
            pivot += 1;
        }
    }

    let mut run_lo = pivot;
    while run_lo > 0
        && (series.pos_x.get(bytes, run_lo) - series.pos_x.get(bytes, run_lo - 1)) <= gap_thresh
    {
        run_lo -= 1;
    }
    let mut run_hi = pivot + 1;
    while run_hi < n
        && (series.pos_x.get(bytes, run_hi) - series.pos_x.get(bytes, run_hi - 1)) <= gap_thresh
    {
        run_hi += 1;
    }
    let run_len = run_hi - run_lo;

    let win = NEVILLE_POINTS.min(run_len);
    let half = (NEVILLE_POINTS / 2) as isize;
    let mut start = pivot as isize - half;
    if start < run_lo as isize {
        start = run_lo as isize;
    }
    if start + win as isize > run_hi as isize {
        start = run_hi as isize - win as isize;
    }
    let start = start as usize;

    let mut t = [0.0f64; NEVILLE_POINTS];
    let mut px = [0.0f64; NEVILLE_POINTS];
    let mut py = [0.0f64; NEVILLE_POINTS];
    let mut pz = [0.0f64; NEVILLE_POINTS];
    for j in 0..win {
        let k = start + j;
        let tj = series.pos_x.get(bytes, k) - query;
        let kx = series.pos_kx.get(bytes, k);
        let ky = series.pos_ky.get(bytes, k);
        let kz = series.pos_kz.get(bytes, k);
        let theta = OMEGA_E_DOT_RAD_S * tj;
        let s = libm::sin(theta);
        let c = libm::cos(theta);
        t[j] = tj;
        px[j] = c * kx - s * ky;
        py[j] = s * kx + c * ky;
        pz[j] = kz;
    }

    let x_km = neville(&t[..win], &px[..win]);
    let y_km = neville(&t[..win], &py[..win]);
    let z_km = neville(&t[..win], &pz[..win]);
    (x_km * KM_TO_M, y_km * KM_TO_M, z_km * KM_TO_M)
}

fn interpolate_mapped_clock(bytes: &[u8], series: &MmapSeries, query: f64) -> Option<f64> {
    if series.clock_node_count < 2 {
        return None;
    }
    let mut chosen = None;
    for (idx, arc) in series.clock_arcs.iter().enumerate() {
        if mapped_arc_contains_query(bytes, arc, query) {
            chosen = Some(idx);
            break;
        }
    }
    let arc = match chosen {
        Some(idx) => &series.clock_arcs[idx],
        None => nearest_mapped_clock_arc(bytes, &series.clock_arcs, query)?,
    };
    if arc.node_count() < 2 {
        return None;
    }
    Some(evaluate_mapped_ppoly(bytes, arc, query) * US_TO_S)
}

fn mapped_arc_contains_query(bytes: &[u8], arc: &MmapClockArc, query: f64) -> bool {
    let node_count = arc.node_count();
    if node_count == 0 {
        return false;
    }
    let lo = arc.x.get(bytes, 0);
    let hi = arc.x.get(bytes, node_count - 1);
    query >= lo && query <= hi
}

fn nearest_mapped_clock_arc<'a, 'b>(
    bytes: &[u8],
    arcs: &'a [MmapClockArc<'b>],
    query: f64,
) -> Option<&'a MmapClockArc<'b>> {
    arcs.iter()
        .filter(|arc| arc.node_count() >= 2)
        .min_by(|arc1, arc2| {
            let d1 = mapped_span_distance(bytes, arc1, query);
            let d2 = mapped_span_distance(bytes, arc2, query);
            d1.partial_cmp(&d2).unwrap_or(core::cmp::Ordering::Equal)
        })
}

fn mapped_span_distance(bytes: &[u8], arc: &MmapClockArc, query: f64) -> f64 {
    let lo = arc.x.get(bytes, 0);
    let hi = arc.x.get(bytes, arc.node_count() - 1);
    if query < lo {
        lo - query
    } else if query > hi {
        query - hi
    } else {
        0.0
    }
}

fn evaluate_mapped_ppoly(bytes: &[u8], arc: &MmapClockArc, query: f64) -> f64 {
    let n = arc.node_count();
    let last = n - 2;
    let interval = if query.is_nan() {
        return f64::NAN;
    } else if query < arc.x.get(bytes, 0) {
        0
    } else if query >= arc.x.get(bytes, n - 1) {
        last
    } else {
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if arc.x.get(bytes, mid) <= query {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };

    debug_assert!(interval < arc.coeff_count());
    let s = query - arc.x.get(bytes, interval);
    let mut res = 0.0;
    let mut z = 1.0;
    res += arc.c3.get(bytes, interval) * z;
    z *= s;
    res += arc.c2.get(bytes, interval) * z;
    z *= s;
    res += arc.c1.get(bytes, interval) * z;
    z *= s;
    res += arc.c0.get(bytes, interval) * z;
    res
}

fn nominal_positive_spacing(bytes: &[u8], series: &MmapSeries) -> Option<f64> {
    let mut nominal = f64::INFINITY;
    for idx in 0..series.pos_count - 1 {
        let d = series.pos_x.get(bytes, idx + 1) - series.pos_x.get(bytes, idx);
        if d > 0.0 {
            nominal = nominal.min(d);
        }
    }
    if nominal.is_finite() {
        Some(nominal)
    } else {
        None
    }
}

fn map_query_input(error: validate::FieldError) -> Error {
    Error::InvalidInput(format!("{} {}", error.field(), error.reason()))
}

fn read_satellite(
    record: &[u8],
) -> core::result::Result<GnssSatelliteId, PreciseInterpolantStoreError> {
    let system_tag = record[SAT_SYSTEM_OFFSET];
    let system = GnssSystem::from_letter(char::from(system_tag))
        .ok_or(PreciseInterpolantStoreError::UnsupportedSatelliteSystem { tag: system_tag })?;
    let prn = record[SAT_PRN_OFFSET];
    GnssSatelliteId::new(system, prn).map_err(|err| parse_error(err.to_string()))
}

fn time_scale_tag(scale: TimeScale) -> u8 {
    match scale {
        TimeScale::Utc => 1,
        TimeScale::Tai => 2,
        TimeScale::Tt => 3,
        TimeScale::Tcg => 4,
        TimeScale::Tdb => 5,
        TimeScale::Tcb => 6,
        TimeScale::Gpst => 7,
        TimeScale::Gst => 8,
        TimeScale::Bdt => 9,
        TimeScale::Glonasst => 10,
        TimeScale::Qzsst => 11,
    }
}

fn time_scale_from_tag(tag: u8) -> core::result::Result<TimeScale, PreciseInterpolantStoreError> {
    match tag {
        1 => Ok(TimeScale::Utc),
        2 => Ok(TimeScale::Tai),
        3 => Ok(TimeScale::Tt),
        4 => Ok(TimeScale::Tcg),
        5 => Ok(TimeScale::Tdb),
        6 => Ok(TimeScale::Tcb),
        7 => Ok(TimeScale::Gpst),
        8 => Ok(TimeScale::Gst),
        9 => Ok(TimeScale::Bdt),
        10 => Ok(TimeScale::Glonasst),
        11 => Ok(TimeScale::Qzsst),
        other => Err(PreciseInterpolantStoreError::UnsupportedTimeScale { tag: other }),
    }
}

fn require_offset(
    sat: GnssSatelliteId,
    field: &str,
    got: usize,
    expected: usize,
) -> core::result::Result<(), PreciseInterpolantStoreError> {
    if got == expected {
        Ok(())
    } else {
        Err(parse_error(format!(
            "satellite {sat} {field} offset must be {expected}, got {got}"
        )))
    }
}

fn parse_f64_array<'a>(
    bytes: &[u8],
    offset: usize,
    count: usize,
    sat: GnssSatelliteId,
    field: &str,
    backing: ArrayBacking<'a>,
) -> core::result::Result<F64Array<'a>, PreciseInterpolantStoreError> {
    checked_range(bytes, offset, count, 8)?;
    let array = match backing {
        ArrayBacking::Borrowed(borrowed_bytes) => {
            F64Array::Borrowed(borrow_f64_slice(borrowed_bytes, offset, count, sat, field)?)
        }
        ArrayBacking::Offset => F64Array::Offset { offset, count },
    };
    for idx in 0..count {
        let value = array.get(bytes, idx);
        if !value.is_finite() {
            return Err(parse_error(format!(
                "satellite {sat} {field} value {idx} is not finite"
            )));
        }
    }
    Ok(array)
}

fn validate_strictly_increasing_f64_array(
    bytes: &[u8],
    values: &F64Array<'_>,
    sat: GnssSatelliteId,
    field: &str,
) -> core::result::Result<(), PreciseInterpolantStoreError> {
    for idx in 0..values.len().saturating_sub(1) {
        if values.get(bytes, idx + 1) <= values.get(bytes, idx) {
            return Err(parse_error(format!(
                "satellite {sat} {field} values are not strictly increasing"
            )));
        }
    }
    Ok(())
}

fn borrow_f64_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    count: usize,
    sat: GnssSatelliteId,
    field: &str,
) -> core::result::Result<&'a [f64], PreciseInterpolantStoreError> {
    let len = count
        .checked_mul(8)
        .ok_or_else(|| parse_error("byte range length overflows usize"))?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| parse_error("byte range end overflows usize"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| parse_error("byte range extends past store length"))?;
    if !cfg!(target_endian = "little") {
        return Err(parse_error(
            "zero-copy precise interpolant f64 arrays require a little-endian target",
        ));
    }
    if !(slice.as_ptr() as usize).is_multiple_of(mem::align_of::<f64>()) {
        return Err(parse_error(format!(
            "satellite {sat} {field} bytes are not aligned for zero-copy f64 access"
        )));
    }
    // SAFETY: f64 accepts every bit pattern, the byte range length was checked
    // above, and callers only reach this after the address-alignment check.
    let (prefix, values, suffix) = unsafe { slice.align_to::<f64>() };
    if !prefix.is_empty() || !suffix.is_empty() || values.len() != count {
        return Err(parse_error(format!(
            "satellite {sat} {field} bytes cannot be borrowed as f64 values"
        )));
    }
    Ok(values)
}

fn checked_range(
    bytes: &[u8],
    offset: usize,
    count: usize,
    item_len: usize,
) -> core::result::Result<(), PreciseInterpolantStoreError> {
    let len = count
        .checked_mul(item_len)
        .ok_or_else(|| parse_error("byte range length overflows usize"))?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| parse_error("byte range end overflows usize"))?;
    if end > bytes.len() {
        return Err(parse_error("byte range extends past store length"));
    }
    Ok(())
}

fn add_len(
    cursor: usize,
    count: usize,
    item_len: usize,
) -> core::result::Result<usize, PreciseInterpolantStoreError> {
    let len = count
        .checked_mul(item_len)
        .ok_or_else(|| parse_error("byte count overflows usize"))?;
    cursor
        .checked_add(len)
        .ok_or_else(|| parse_error("byte cursor overflows usize"))
}

fn align_up(
    value: usize,
    alignment: usize,
) -> core::result::Result<usize, PreciseInterpolantStoreError> {
    let rem = value % alignment;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - rem)
            .ok_or_else(|| parse_error("aligned offset overflows usize"))
    }
}

fn ensure_zero(
    bytes: &[u8],
    start: usize,
    end: usize,
    context: &str,
) -> core::result::Result<(), PreciseInterpolantStoreError> {
    if start > end || end > bytes.len() {
        return Err(parse_error(format!("{context} range is out of bounds")));
    }
    if bytes[start..end].iter().any(|&byte| byte != 0) {
        return Err(parse_error(format!("{context} must be zero-filled")));
    }
    Ok(())
}

fn parse_error(reason: impl Into<String>) -> PreciseInterpolantStoreError {
    PreciseInterpolantStoreError::Parse {
        reason: reason.into(),
    }
}

fn artifact_checksum64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for (idx, byte) in bytes.iter().enumerate() {
        let value = if (HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 8).contains(&idx) {
            0
        } else {
            *byte
        };
        hash = (hash ^ u64::from(value)).wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

// invariant: callers pass ranges validated by the mapped-store parser.
#[allow(clippy::expect_used)]
fn mapped_f64(bytes: &[u8], offset: usize, idx: usize) -> f64 {
    let start = offset + idx * 8;
    f64::from_le_bytes(
        bytes[start..start + 8]
            .try_into()
            .expect("validated f64 range"),
    )
}

fn read_u16(
    bytes: &[u8],
    offset: usize,
) -> core::result::Result<u16, PreciseInterpolantStoreError> {
    Ok(u16::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u32(
    bytes: &[u8],
    offset: usize,
) -> core::result::Result<u32, PreciseInterpolantStoreError> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u64(
    bytes: &[u8],
    offset: usize,
) -> core::result::Result<u64, PreciseInterpolantStoreError> {
    Ok(u64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_f64(
    bytes: &[u8],
    offset: usize,
) -> core::result::Result<f64, PreciseInterpolantStoreError> {
    Ok(f64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
) -> core::result::Result<[u8; N], PreciseInterpolantStoreError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| parse_error("numeric field offset overflows usize"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| parse_error("numeric field extends past record"))?;
    slice
        .try_into()
        .map_err(|_| parse_error("numeric field has wrong length"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(bytes: &mut [u8], offset: usize, value: f64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64_slice(bytes: &mut [u8], offset: usize, values: &[f64]) {
    for (idx, value) in values.iter().enumerate() {
        write_f64(bytes, offset + idx * 8, *value);
    }
}
