//! Backing storage for readers over large binary artifacts.
//!
//! The terrain store and the precise-interpolant store are both read by
//! indexing into a byte span: construction parses only the header, datum tag,
//! and tile/segment index, and every lookup addresses payload by offset. Neither
//! reader holds a reference into its own bytes, so the bytes can be owned,
//! borrowed, or memory-mapped without any of them being a self-referential
//! struct - the span is derived on demand from whichever backing is present.
//!
//! That is what makes the mapped variant safe to add: there is no interior
//! pointer to keep valid, no drop-order invariant hiding in field declaration
//! order, and no `unsafe` at any interface boundary.
//!
//! # Why mapping matters more than the copy
//!
//! Avoiding one `memcpy` is the smaller half. A memory map is demand-paged, so a
//! reader that queries a geographically local region faults in the handful of
//! pages covering those tiles and never touches the rest of the file. A
//! constructor that copies - however the bytes arrive - forfeits that and pays
//! for the whole artifact on every open. For a 30+ GB terrain store that is the
//! difference between a working process and one that cannot start.

use std::borrow::Cow;

/// Where a reader's bytes live.
///
/// `Mapped` is only available with the `mmap` feature, so the default build
/// carries no additional dependency and targets where mapping is meaningless
/// (wasm) simply do not enable it.
#[derive(Debug, Clone)]
pub enum ArtifactBytes<'a> {
    /// A span the caller owns and keeps alive.
    Borrowed(&'a [u8]),
    /// A vector this reader owns.
    Owned(Vec<u8>),
    /// A read-only memory map this reader owns.
    ///
    /// Shared behind an `Arc` so the reader stays cheap to clone; a map is a
    /// kernel-level resource and duplicating it per clone would be wasteful.
    #[cfg(feature = "mmap")]
    Mapped(std::sync::Arc<memmap2::Mmap>),
}

impl<'a> ArtifactBytes<'a> {
    /// Borrow the artifact bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes.as_slice(),
            #[cfg(feature = "mmap")]
            Self::Mapped(map) => &map[..],
        }
    }

    /// Whether these bytes are a memory map rather than a copy in process
    /// memory.
    ///
    /// Exposed so a caller - or a test - can assert that a path-based open
    /// actually mapped the file instead of reading it. A change that quietly
    /// relocated the copy would otherwise be indistinguishable from a fix.
    #[must_use]
    pub fn is_memory_mapped(&self) -> bool {
        #[cfg(feature = "mmap")]
        {
            matches!(self, Self::Mapped(_))
        }
        #[cfg(not(feature = "mmap"))]
        {
            false
        }
    }
}

impl AsRef<[u8]> for ArtifactBytes<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<'a> From<Cow<'a, [u8]>> for ArtifactBytes<'a> {
    fn from(bytes: Cow<'a, [u8]>) -> Self {
        match bytes {
            Cow::Borrowed(bytes) => Self::Borrowed(bytes),
            Cow::Owned(bytes) => Self::Owned(bytes),
        }
    }
}

/// Map a file read-only.
///
/// These artifacts are content-addressed and are mounted read-only where they
/// are deployed, so the map is never opened for writing.
///
/// # Safety of the underlying map
///
/// `memmap2::Mmap::map` is unsafe because the mapped file can be modified by
/// another process, which would change bytes under the reader. The contract
/// here is the same one the format already relies on: an artifact is
/// content-addressed and immutable once published. A caller that maps a file
/// somebody else is concurrently rewriting has a corrupt read either way; the
/// map does not introduce that hazard, it inherits it.
#[cfg(feature = "mmap")]
pub fn map_file_read_only(path: &std::path::Path) -> std::io::Result<ArtifactBytes<'static>> {
    let file = std::fs::File::open(path)?;
    // SAFETY: opened read-only, and the artifact is immutable once published -
    // see the note above.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    Ok(ArtifactBytes::Mapped(std::sync::Arc::new(map)))
}
