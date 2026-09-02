//! Stable, secret-free identity for the exact artifacts and policy of an SP3 merge.

use std::collections::HashSet;
use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::data::{ArchiveCompression, DistributionSource, ProductIdentity, ProductType};
use crate::tolerances::WHOLE_SECOND_EPS_S;

use super::{MergeCombine, MergeOptions, MergePrecedenceScope};

/// Version of the canonical merged-SP3 input identity encoding.
pub const SP3_MERGE_INPUT_SCHEMA_VERSION: u8 = 1;

/// Prefix carried by every public merged-SP3 input identity.
pub const SP3_MERGE_INPUT_ID_PREFIX: &str = "sidereon-sp3-merge-input-v1:";

/// Reproducible identity of one verified artifact supplied to an SP3 merge.
///
/// Retrieval time, cache status, URLs, HTTP metadata, failures, credentials,
/// and local paths do not belong here. They are observational acquisition
/// facts and deliberately do not affect the stable merge-input identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sp3ArtifactIdentity {
    /// Exact identity requested from the selected distributor.
    pub requested_identity: ProductIdentity,
    /// Identity resolved by parsing and validating the acquired bytes.
    pub resolved_identity: ProductIdentity,
    /// Explicit distributor that supplied the artifact.
    pub distribution_source: DistributionSource,
    /// Official decompressed product filename.
    pub official_filename: String,
    /// SHA-256 of the validated, decompressed product bytes.
    pub product_sha256: String,
    /// Length of the validated, decompressed product bytes.
    pub product_byte_length: u64,
    /// SHA-256 of the exact distributor archive bytes.
    pub archive_sha256: String,
    /// Length of the exact distributor archive bytes.
    pub archive_byte_length: u64,
    /// Compression applied to the distributor archive.
    pub compression: ArchiveCompression,
}

/// Canonical identity of a complete SP3 merge input set and merge policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sp3MergeInputIdentity {
    /// Canonical encoding version.
    pub schema_version: u8,
    /// Contributors in canonical order, independent of caller enumeration.
    pub contributors: Vec<Sp3ArtifactIdentity>,
    /// Ordered contributors when precedence combination makes order semantic.
    /// `None` for mean and median combination.
    pub precedence_contributors: Option<Vec<Sp3ArtifactIdentity>>,
    /// Versioned SHA-256 identity of the contributors and complete merge policy.
    pub stable_id: String,
}

/// Failure to build a complete merged-SP3 input identity.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Sp3MergeInputIdentityError {
    /// At least one verified contributor is required.
    #[error("merged-SP3 input identity requires at least one contributor")]
    EmptyContributors,
    /// A contributor record is incomplete or internally inconsistent.
    #[error("invalid merged-SP3 contributor {index}: {reason}")]
    InvalidContributor {
        /// Index in the caller-provided contributor list.
        index: usize,
        /// Stable failure description.
        reason: &'static str,
    },
    /// One resolved product identity appeared more than once.
    #[error("duplicate resolved product identity at contributor {index}")]
    DuplicateContributor {
        /// Index of the duplicate in the caller-provided list.
        index: usize,
    },
    /// Merge controls cannot be represented as a valid executable policy.
    #[error("invalid merged-SP3 policy: {0}")]
    InvalidPolicy(&'static str),
}

impl Sp3MergeInputIdentity {
    /// Validate, canonically order, and bind all contributors and merge controls.
    // invariant: formatting a digest into a String cannot fail.
    #[allow(clippy::expect_used)]
    pub fn new(
        contributors: &[Sp3ArtifactIdentity],
        policy: &MergeOptions,
    ) -> Result<Self, Sp3MergeInputIdentityError> {
        if contributors.is_empty() {
            return Err(Sp3MergeInputIdentityError::EmptyContributors);
        }

        validate_policy(policy)?;
        let mut seen = HashSet::new();
        let mut canonical = Vec::with_capacity(contributors.len());
        for (index, contributor) in contributors.iter().enumerate() {
            let bytes = canonical_contributor_bytes(contributor, index)?;
            let resolved = contributor
                .resolved_identity
                .canonical_bytes()
                .map_err(|_| invalid_contributor(index, "resolved identity"))?;
            if !seen.insert(resolved) {
                return Err(Sp3MergeInputIdentityError::DuplicateContributor { index });
            }
            canonical.push((bytes, contributor.clone()));
        }
        let (precedence, precedence_contributors) = if policy.combine == MergeCombine::Precedence {
            (
                canonical
                    .iter()
                    .map(|(encoded, _)| encoded.clone())
                    .collect(),
                Some(
                    canonical
                        .iter()
                        .map(|(_, contributor)| contributor.clone())
                        .collect(),
                ),
            )
        } else {
            (Vec::new(), None)
        };
        canonical.sort_by(|left, right| left.0.cmp(&right.0));

        let mut bytes = Vec::new();
        put_field(&mut bytes, b"sidereon.sp3.merge-input");
        bytes.push(SP3_MERGE_INPUT_SCHEMA_VERSION);
        put_u64(&mut bytes, canonical.len() as u64);
        for (encoded, _) in &canonical {
            put_field(&mut bytes, encoded);
        }
        put_field(&mut bytes, &canonical_policy_bytes(policy, &precedence));

        let digest = Sha256::digest(bytes);
        let mut encoded = String::with_capacity(SP3_MERGE_INPUT_ID_PREFIX.len() + 64);
        encoded.push_str(SP3_MERGE_INPUT_ID_PREFIX);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }

        Ok(Self {
            schema_version: SP3_MERGE_INPUT_SCHEMA_VERSION,
            contributors: canonical.into_iter().map(|(_, value)| value).collect(),
            precedence_contributors,
            stable_id: encoded,
        })
    }

    /// Recompute and verify this persisted identity and its merge controls.
    pub fn verify(&self, policy: &MergeOptions) -> Result<bool, Sp3MergeInputIdentityError> {
        let contributors = self
            .precedence_contributors
            .as_deref()
            .unwrap_or(&self.contributors);
        self.verify_against(contributors, policy)
    }

    /// Recompute and compare against separately persisted contributor records
    /// and merge controls.
    pub fn verify_against(
        &self,
        contributors: &[Sp3ArtifactIdentity],
        policy: &MergeOptions,
    ) -> Result<bool, Sp3MergeInputIdentityError> {
        let rebuilt = Self::new(contributors, policy)?;
        Ok(self.schema_version == rebuilt.schema_version
            && self.contributors == rebuilt.contributors
            && self.precedence_contributors == rebuilt.precedence_contributors
            && self.stable_id == rebuilt.stable_id)
    }
}

fn canonical_contributor_bytes(
    contributor: &Sp3ArtifactIdentity,
    index: usize,
) -> Result<Vec<u8>, Sp3MergeInputIdentityError> {
    contributor
        .requested_identity
        .validate()
        .map_err(|_| invalid_contributor(index, "requested identity"))?;
    contributor
        .resolved_identity
        .validate()
        .map_err(|_| invalid_contributor(index, "resolved identity"))?;
    if contributor.requested_identity.family != ProductType::Sp3
        || contributor.resolved_identity.family != ProductType::Sp3
    {
        return Err(invalid_contributor(index, "product family is not SP3"));
    }
    if contributor
        .resolved_identity
        .format_version
        .as_deref()
        .is_none_or(|version| version.trim().is_empty())
    {
        return Err(invalid_contributor(index, "resolved format version"));
    }
    if contributor.requested_identity.format_version.is_some()
        && contributor.requested_identity.format_version
            != contributor.resolved_identity.format_version
    {
        return Err(invalid_contributor(
            index,
            "resolved format version does not match requested version",
        ));
    }

    let mut requested_without_revision = contributor.requested_identity.clone();
    requested_without_revision.format_version = None;
    let mut resolved_without_revision = contributor.resolved_identity.clone();
    resolved_without_revision.format_version = None;
    if requested_without_revision != resolved_without_revision {
        return Err(invalid_contributor(
            index,
            "resolved identity does not match requested identity",
        ));
    }
    if contributor.official_filename != contributor.requested_identity.official_filename
        || contributor.official_filename != contributor.resolved_identity.official_filename
    {
        return Err(invalid_contributor(index, "official filename"));
    }
    if !valid_sha256(&contributor.product_sha256) {
        return Err(invalid_contributor(index, "product SHA-256"));
    }
    if !valid_sha256(&contributor.archive_sha256) {
        return Err(invalid_contributor(index, "archive SHA-256"));
    }
    if contributor.product_byte_length == 0 {
        return Err(invalid_contributor(index, "product byte length"));
    }
    if contributor.archive_byte_length == 0 {
        return Err(invalid_contributor(index, "archive byte length"));
    }

    let mut bytes = Vec::new();
    put_field(
        &mut bytes,
        &contributor
            .requested_identity
            .canonical_bytes()
            .map_err(|_| invalid_contributor(index, "requested identity"))?,
    );
    put_field(
        &mut bytes,
        &contributor
            .resolved_identity
            .canonical_bytes()
            .map_err(|_| invalid_contributor(index, "resolved identity"))?,
    );
    put_field(
        &mut bytes,
        contributor.distribution_source.code().as_bytes(),
    );
    put_field(&mut bytes, contributor.official_filename.as_bytes());
    put_field(&mut bytes, contributor.product_sha256.as_bytes());
    put_u64(&mut bytes, contributor.product_byte_length);
    put_field(&mut bytes, contributor.archive_sha256.as_bytes());
    put_u64(&mut bytes, contributor.archive_byte_length);
    put_field(&mut bytes, contributor.compression.as_str().as_bytes());
    Ok(bytes)
}

fn validate_policy(policy: &MergeOptions) -> Result<(), Sp3MergeInputIdentityError> {
    if !finite_nonnegative(policy.position_tolerance_m) {
        return Err(Sp3MergeInputIdentityError::InvalidPolicy(
            "position tolerance",
        ));
    }
    if !finite_nonnegative(policy.clock_tolerance_s) {
        return Err(Sp3MergeInputIdentityError::InvalidPolicy("clock tolerance"));
    }
    if policy.min_agree == 0 {
        return Err(Sp3MergeInputIdentityError::InvalidPolicy(
            "minimum agreement",
        ));
    }
    if policy.clock_min_common == 0 {
        return Err(Sp3MergeInputIdentityError::InvalidPolicy(
            "minimum common clocks",
        ));
    }
    if let Some(value) = policy.outlier_reject {
        if !finite_nonnegative(value.position_tolerance_m)
            || !finite_nonnegative(value.clock_tolerance_s)
        {
            return Err(Sp3MergeInputIdentityError::InvalidPolicy(
                "outlier rejection",
            ));
        }
    }
    if let Some(value) = policy.target_epoch_interval_s {
        if !value.is_finite()
            || (value - value.round()).abs() > WHOLE_SECOND_EPS_S
            || value.round() < 1.0
        {
            return Err(Sp3MergeInputIdentityError::InvalidPolicy(
                "target epoch interval",
            ));
        }
    }
    if policy
        .systems
        .as_ref()
        .is_some_and(|systems| systems.is_empty())
    {
        return Err(Sp3MergeInputIdentityError::InvalidPolicy("systems filter"));
    }
    for labels in &policy.frame_reconciliation.asserted_equivalent_label_sets {
        if labels.labels.len() < 2 || labels.labels.iter().any(|label| label.trim().is_empty()) {
            return Err(Sp3MergeInputIdentityError::InvalidPolicy(
                "asserted frame label set",
            ));
        }
    }
    Ok(())
}

fn canonical_policy_bytes(policy: &MergeOptions, precedence: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u64(
        &mut bytes,
        canonical_nonnegative_f64_bits(policy.position_tolerance_m),
    );
    put_u64(
        &mut bytes,
        canonical_nonnegative_f64_bits(policy.clock_tolerance_s),
    );
    put_u64(&mut bytes, policy.min_agree as u64);
    put_u64(&mut bytes, policy.clock_min_common as u64);
    bytes.push(match policy.combine {
        MergeCombine::Mean => 0,
        MergeCombine::Median => 1,
        MergeCombine::Precedence => 2,
    });
    bytes.push(match policy.precedence_scope {
        MergePrecedenceScope::Cell => 0,
        MergePrecedenceScope::SatelliteArc => 1,
    });
    match policy.outlier_reject {
        Some(value) => {
            bytes.push(1);
            put_u64(
                &mut bytes,
                canonical_nonnegative_f64_bits(value.position_tolerance_m),
            );
            put_u64(
                &mut bytes,
                canonical_nonnegative_f64_bits(value.clock_tolerance_s),
            );
        }
        None => bytes.push(0),
    }
    match policy.target_epoch_interval_s {
        Some(value) => {
            bytes.push(1);
            put_u64(&mut bytes, value.to_bits());
        }
        None => bytes.push(0),
    }

    match &policy.systems {
        Some(systems) => {
            bytes.push(1);
            put_u64(&mut bytes, systems.len() as u64);
            for system in systems {
                bytes.push(system.letter() as u8);
            }
        }
        None => bytes.push(0),
    }

    let mut label_sets: Vec<Vec<u8>> = policy
        .frame_reconciliation
        .asserted_equivalent_label_sets
        .iter()
        .map(|set| {
            let mut encoded = Vec::new();
            put_u64(&mut encoded, set.labels.len() as u64);
            for label in &set.labels {
                put_field(&mut encoded, label.as_bytes());
            }
            encoded
        })
        .collect();
    label_sets.sort();
    put_u64(&mut bytes, label_sets.len() as u64);
    for labels in label_sets {
        put_field(&mut bytes, &labels);
    }
    bytes.push(u8::from(policy.frame_reconciliation.helmert));
    put_u64(&mut bytes, precedence.len() as u64);
    for contributor in precedence {
        put_field(&mut bytes, contributor);
    }
    bytes
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn canonical_nonnegative_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_contributor(index: usize, reason: &'static str) -> Sp3MergeInputIdentityError {
    Sp3MergeInputIdentityError::InvalidContributor { index, reason }
}

fn put_field(output: &mut Vec<u8>, value: &[u8]) {
    put_u64(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_be_bytes());
}
