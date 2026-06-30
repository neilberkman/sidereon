//! Dilution of precision (DOP) from a satellite geometry.
//!
//! Dilution of precision summarises how the receiver-to-satellite geometry maps
//! measurement noise into solution uncertainty. From a design (geometry) matrix
//! `H` whose rows are the unit line-of-sight vectors plus a clock column, and a
//! diagonal weight matrix `W`, the cofactor matrix is
//!
//! ```text
//! Q = (H^T W H)^-1
//! ```
//!
//! a 4x4 symmetric matrix ordered `[x, y, z, clock]` (the position block in
//! ECEF metres, the clock state in the same length unit as the ranges). The DOP
//! scalars are square roots of sums of diagonal cofactor entries. The
//! horizontal/vertical split is taken after rotating the 3x3 position block into
//! a local east-north-up (ENU) frame at the receiver's geodetic
//! latitude/longitude.
//!
//! # ENU convention
//!
//! HDOP, VDOP, and PDOP are convention-dependent: they split the position
//! cofactor along a local "up" axis, and two definitions of "up" appear in the
//! literature. The default everywhere ([`dop`], [`geometry_cofactor`], and the
//! covariance/ellipse helpers) is the **geodetic-ellipsoid-normal** ENU built
//! from the receiver's geodetic latitude/longitude (the GNSS standard, matching
//! RTKLIB's `xyz2enu`). The alternative is a **geocentric-radial** ENU whose up
//! is the spherical radial direction `position / |position|`. The two "up"
//! axes differ by up to ~0.19 degrees (the deflection of the ellipsoid normal
//! from the radial), which moves the horizontal/vertical split by on the order
//! of `1e-3` relative. [`EnuConvention`] and the `*_with_convention` entry
//! points let a caller select the geocentric-radial variant; the default
//! helpers keep the geodetic-normal convention and its 0-ULP goldens. See
//! [`crate::frame::geocentric_up`] for the geocentric-vs-geodetic distinction.
//!
//! # Reproducibility
//!
//! The normal matrix `H^T W H` is accumulated by a plain left-to-right sum over
//! the satellites, and the 4x4 inverse is an explicit cofactor (adjugate /
//! determinant) expansion with a fixed term order rather than a LAPACK
//! factorisation. That keeps the whole computation libm/arithmetic-bound and
//! independently reproducible to the bit (it does not depend on a BLAS backend),
//! unlike a general dense inverse routed through LAPACK. The ENU rotation uses
//! `sin`/`cos` and the final scalars use `sqrt`; there is no fused multiply-add.
//!
//! # Failure mode
//!
//! A geometry with fewer than four independent line-of-sight directions, or one
//! whose normal matrix is singular or ill-conditioned, has no finite DOP. Such
//! geometries are reported as [`DopError::Singular`] rather than returning a
//! NaN-flagged or clamped result. The predicate is deterministic: the
//! determinant is exactly zero, or one of the variance diagonals that a DOP
//! scalar takes the square root of is negative or non-finite.

use crate::astro::math::linear::{
    invert_4x4_cofactor, invert_symmetric_pd, normal_matrix_4_weighted_column_outer, LinearError,
};
use crate::astro::math::mat3::{inline_rxr, inline_tr};

use crate::frame::Wgs84Geodetic;
use crate::id::GnssSystem;
use crate::validate;

const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
const LOS_UNIT_TOLERANCE: f64 = 1.0e-3;
/// Minimum ECEF radius (metres) for which the geocentric-radial "up" direction
/// is well defined. A receiver at or within this distance of the geocenter has
/// no meaningful radial axis, so [`EnuConvention::GeocentricRadial`] rejects it.
const GEOCENTRIC_MIN_RADIUS_M: f64 = 1.0;

/// A line-of-sight unit vector from the receiver toward a satellite, in ECEF.
///
/// The corresponding design-matrix row is `[-e_x, -e_y, -e_z, 1]`: the partial
/// derivative of the predicted range with respect to the receiver position is
/// `-e`, and the clock column is one (range increases one-for-one with the
/// receiver clock bias expressed as a length).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineOfSight {
    /// ECEF X component of the unit line-of-sight vector.
    pub e_x: f64,
    /// ECEF Y component of the unit line-of-sight vector.
    pub e_y: f64,
    /// ECEF Z component of the unit line-of-sight vector.
    pub e_z: f64,
}

impl LineOfSight {
    /// Construct a line-of-sight unit vector from ECEF components.
    pub const fn new(e_x: f64, e_y: f64, e_z: f64) -> Self {
        Self { e_x, e_y, e_z }
    }

    /// The design-matrix row `[-e_x, -e_y, -e_z, 1]` for this direction.
    fn design_row(self) -> [f64; 4] {
        [-self.e_x, -self.e_y, -self.e_z, 1.0]
    }
}

/// Construct an ECEF line-of-sight unit vector from topocentric azimuth and
/// elevation in degrees.
///
/// Azimuth is clockwise from geodetic north, elevation is positive above the
/// local horizon, and the receiver latitude/longitude define the local ENU
/// frame. The returned vector points from receiver to satellite in ECEF.
pub fn line_of_sight_from_az_el_deg(
    azimuth_deg: f64,
    elevation_deg: f64,
    receiver: Wgs84Geodetic,
) -> Result<LineOfSight, DopError> {
    validate_az_el_receiver(azimuth_deg, elevation_deg, receiver)?;
    let az = azimuth_deg * DEG_TO_RAD;
    let el = elevation_deg * DEG_TO_RAD;
    let cos_el = el.cos();
    let east = cos_el * az.sin();
    let north = cos_el * az.cos();
    let up = el.sin();

    let r = ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad);
    let e_x = r[0][0] * east + r[1][0] * north + r[2][0] * up;
    let e_y = r[0][1] * east + r[1][1] * north + r[2][1] * up;
    let e_z = r[0][2] * east + r[1][2] * north + r[2][2] * up;
    let los = LineOfSight::new(e_x, e_y, e_z);
    validate_los(core::slice::from_ref(&los))?;
    Ok(los)
}

/// The dilution-of-precision scalars for a geometry.
///
/// Each is dimensionless: the standard deviation of the solution component is
/// the corresponding DOP times the (range) measurement standard deviation. The
/// position split is in the local ENU frame at the receiver.
///
/// Produced by [`dop`] for a single receiver-clock state and by the positioning
/// pipeline's multi-clock geometry path for a multi-system state; the field
/// meanings below cover both.
#[derive(Debug, Clone, PartialEq)]
pub struct Dop {
    /// Geometric DOP: the square root of the trace of the cofactor matrix over
    /// every state - the three position coordinates and every clock (one for a
    /// single-system solve, one per constellation for a multi-system solve).
    pub gdop: f64,
    /// Position DOP: `sqrt(qE + qN + qU)` over the ENU position block.
    pub pdop: f64,
    /// Horizontal DOP: `sqrt(qE + qN)`.
    pub hdop: f64,
    /// Vertical DOP: `sqrt(qU)`.
    pub vdop: f64,
    /// Time (clock) DOP: the square root of the reference clock's cofactor
    /// variance (`Q[3][3]`). With several clocks this is the first (reference)
    /// system's clock; the others enter `gdop` through the trace.
    pub tdop: f64,
    /// Per-system time DOP: one entry per receiver-clock column, `(system,
    /// sqrt(Q[3+i][3+i]))` for the constellation that owns clock column `i`.
    /// Entry `0` is the reference clock and its value always equals
    /// [`tdop`](Self::tdop).
    ///
    /// The multi-GNSS geometry path (`dop_multi`) is given the system that owns
    /// each clock column and tags every entry, so this is exactly the tagged
    /// `Vec<(GnssSystem, f64)>` shape the positioning layer's `system_tdops`
    /// uses: a consumer can pair a `Dop` with an SPP solution without
    /// re-tagging. The system-agnostic single-clock [`dop`] has no
    /// constellation context, so it leaves this empty - read
    /// [`tdop`](Self::tdop) for its lone clock.
    pub system_tdops: Vec<(GnssSystem, f64)>,
}

/// Cofactor matrices from a single-clock GNSS design matrix.
///
/// `state` is `(H^T W H)^-1` for rows `[-e_x, -e_y, -e_z, 1]`, ordered
/// `[x, y, z, clock]`. The two position blocks are the top-left 3x3 block in ECEF
/// coordinates and the same block rotated into the local ENU frame at the
/// receiver.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryCofactor {
    /// Full 4x4 state cofactor matrix, ordered `[x, y, z, clock]`.
    pub state: [[f64; 4]; 4],
    /// Position cofactor block in ECEF coordinates.
    pub position_ecef: [[f64; 3]; 3],
    /// Position cofactor block in local ENU coordinates.
    pub position_enu: [[f64; 3]; 3],
}

/// Position covariance from a GNSS design matrix.
///
/// The matrices are in square metres when the supplied variance scale is in
/// square metres. Use a variance scale of `1.0` to get the raw cofactor block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionCovariance {
    /// Position covariance in ECEF coordinates, square metres.
    pub ecef_m2: [[f64; 3]; 3],
    /// Position covariance in local ENU coordinates, square metres.
    pub enu_m2: [[f64; 3]; 3],
}

/// Horizontal confidence ellipse from an ENU position covariance.
///
/// `azimuth_rad` is the semi-major-axis direction measured counter-clockwise from
/// local east toward local north. `confidence` is the probability used for the
/// two-degree-of-freedom chi-square scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizontalErrorEllipse {
    /// Requested confidence probability in `(0, 1)`.
    pub confidence: f64,
    /// Two-dimensional chi-square scale applied to the covariance eigenvalues.
    pub chi_square_scale: f64,
    /// Semi-major axis length, metres.
    pub semi_major_m: f64,
    /// Semi-minor axis length, metres.
    pub semi_minor_m: f64,
    /// Semi-major-axis azimuth in radians, from east toward north.
    pub azimuth_rad: f64,
}

/// A confidence ellipse from an arbitrary 2x2 covariance block.
///
/// Domain-neutral companion to [`HorizontalErrorEllipse`]: the axes carry
/// whatever unit the covariance is in (square that unit), and `orientation_rad`
/// is the semi-major-axis direction measured from the first axis toward the
/// second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorEllipse2 {
    /// Requested confidence probability in `(0, 1)`.
    pub confidence: f64,
    /// Two-degree-of-freedom chi-square scale `-2 ln(1 - confidence)` applied to
    /// the covariance eigenvalues.
    pub chi_square_scale: f64,
    /// Semi-major axis length (same unit as the square root of the covariance).
    pub semi_major: f64,
    /// Semi-minor axis length.
    pub semi_minor: f64,
    /// Semi-major-axis orientation in radians, from the first (row/col 0) axis
    /// toward the second (row/col 1) axis.
    pub orientation_rad: f64,
}

/// Why a geometry has no finite DOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DopError {
    /// A boundary input was malformed before DOP could be evaluated.
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// Fewer line-of-sight directions than estimated parameters were supplied
    /// (four for a single clock, `3 + n_clocks` for several), so the normal
    /// matrix cannot be full rank.
    TooFewSatellites,
    /// The normal matrix `H^T W H` is singular or ill-conditioned: its
    /// determinant is zero, or a variance diagonal is negative or non-finite.
    Singular,
}

impl core::fmt::Display for DopError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DopError::InvalidInput { field, reason } => {
                write!(f, "invalid DOP input {field}: {reason}")
            }
            DopError::TooFewSatellites => {
                write!(
                    f,
                    "fewer satellites than parameters: geometry is rank-deficient"
                )
            }
            DopError::Singular => {
                write!(f, "singular or ill-conditioned geometry: no finite DOP")
            }
        }
    }
}

impl std::error::Error for DopError {}

/// Which local east-north-up frame the position cofactor is rotated into before
/// the horizontal/vertical DOP split.
///
/// The two conventions differ only in the definition of local "up" and so in
/// the HDOP/VDOP partition; GDOP, PDOP, and TDOP are unaffected by the choice
/// (PDOP is the trace of the position block, which is rotation-invariant). See
/// the module-level "ENU convention" section for the ~0.19 degree difference
/// between the axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EnuConvention {
    /// Geodetic-ellipsoid-normal ENU built from the receiver's geodetic
    /// latitude/longitude. The GNSS-standard default (RTKLIB `xyz2enu`).
    #[default]
    GeodeticNormal,
    /// Geocentric-radial ENU whose up is the spherical radial direction
    /// `position / |position|` (see [`crate::frame::geocentric_neu_basis`]).
    GeocentricRadial,
}

/// ECEF -> ENU rotation rows `[east; north; up]` for the requested convention.
///
/// `GeodeticNormal` returns exactly [`ecef_to_enu_rotation`] (so the default
/// DOP path is byte-for-byte unchanged); `GeocentricRadial` builds the rows from
/// [`crate::frame::geocentric_neu_basis`] at the receiver's ECEF position.
fn enu_rotation(
    receiver: Wgs84Geodetic,
    convention: EnuConvention,
) -> Result<[[f64; 3]; 3], DopError> {
    match convention {
        EnuConvention::GeodeticNormal => {
            Ok(ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad))
        }
        EnuConvention::GeocentricRadial => {
            let ecef = crate::frame::geodetic_to_itrf(receiver)
                .map_err(|_| invalid_input("receiver", "geocentric basis unavailable"))?;
            let p = ecef.as_array();
            // Geocentric "up" is position / |position|, which is undefined at the
            // geocenter. `geocentric_neu_basis` would silently fall back to +Z
            // there and return an arbitrary frame; reject a zero/near-zero radius
            // instead of accepting that fabricated orientation.
            // `p` comes from a validated-finite ItrfPositionM, so `radius` is
            // finite and non-negative; a plain threshold comparison suffices.
            let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if radius <= GEOCENTRIC_MIN_RADIUS_M {
                return Err(invalid_input(
                    "receiver",
                    "geocentric up undefined at zero radius",
                ));
            }
            let (north, east, up) = crate::frame::geocentric_neu_basis(p);
            Ok([east, north, up])
        }
    }
}

/// ECEF -> ENU rotation matrix at geodetic latitude/longitude (radians).
fn ecef_to_enu_rotation(lat_rad: f64, lon_rad: f64) -> [[f64; 3]; 3] {
    let sphi = lat_rad.sin();
    let cphi = lat_rad.cos();
    let slam = lon_rad.sin();
    let clam = lon_rad.cos();
    [
        [-slam, clam, 0.0],
        [-sphi * clam, -sphi * slam, cphi],
        [cphi * clam, cphi * slam, sphi],
    ]
}

/// Rotate the 3x3 position cofactor block: `Q_enu = R Q_xyz R^T`, formed as
/// `(R Q) R^T`. Reads the top-left 3x3 block of the 4x4 cofactor matrix and
/// defers to [`rotate3`].
fn rotate_pos_block(q: &[[f64; 4]; 4], r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let qpos = [
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ];
    rotate3(&qpos, r)
}

/// Compute the DOP scalars from line-of-sight directions, diagonal weights, and
/// the receiver geodetic position.
///
/// `los` and `weights` must have the same length, which must be at least four;
/// `weights` are the non-negative diagonal of `W`. Returns
/// [`DopError::TooFewSatellites`] for fewer than four directions and
/// [`DopError::Singular`] for a singular or
/// ill-conditioned geometry (see the module docs for the exact predicate).
pub fn dop(los: &[LineOfSight], weights: &[f64], receiver: Wgs84Geodetic) -> Result<Dop, DopError> {
    dop_with_convention(los, weights, receiver, EnuConvention::GeodeticNormal)
}

/// [`dop`] with an explicit [`EnuConvention`] for the horizontal/vertical split.
///
/// [`EnuConvention::GeodeticNormal`] is the default [`dop`] path (0-ULP
/// goldens); [`EnuConvention::GeocentricRadial`] rotates the position block into
/// the geocentric-radial ENU instead, changing only HDOP/VDOP (by ~`1e-3`
/// relative; see the module "ENU convention" section). GDOP/PDOP/TDOP are
/// identical between conventions.
pub fn dop_with_convention(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    convention: EnuConvention,
) -> Result<Dop, DopError> {
    validate_dop_inputs(los, weights, receiver)?;
    if los.len() < 4 {
        return Err(DopError::TooFewSatellites);
    }

    let rows: Vec<[f64; 4]> = los.iter().map(|l| l.design_row()).collect();
    let a = normal_matrix_4_weighted_column_outer(&rows, weights).map_err(map_linear_error)?;
    let q = invert_4x4_cofactor(&a).ok_or(DopError::Singular)?;

    let r = enu_rotation(receiver, convention)?;
    let enu = rotate_pos_block(&q, &r);

    let qe = enu[0][0];
    let qn = enu[1][1];
    let qu = enu[2][2];
    let qt = q[3][3];

    // The DOP scalars take the square root of cofactor variances. A
    // well-posed geometry yields a positive-definite Q with strictly positive
    // variance diagonals; a rank-deficient or ill-conditioned geometry can
    // leave a tiny nonzero determinant (so `inv4` succeeds) yet produce a
    // negative or non-finite variance. Reject that here rather than returning a
    // NaN-flagged DOP. The same deterministic predicate is applied by the
    // reference recipe so both agree on the success/failure boundary.
    let gdop_arg = q[0][0] + q[1][1] + q[2][2] + q[3][3];
    let pdop_arg = qe + qn + qu;
    let hdop_arg = qe + qn;
    let vdop_arg = qu;
    let tdop_arg = qt;
    for arg in [gdop_arg, pdop_arg, hdop_arg, vdop_arg, tdop_arg] {
        // `!(arg >= 0.0)` (not `arg < 0.0`) so a NaN variance is also rejected.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let nonpositive_or_nan = !(arg >= 0.0);
        if nonpositive_or_nan || !arg.is_finite() {
            return Err(DopError::Singular);
        }
    }

    Ok(Dop {
        gdop: gdop_arg.sqrt(),
        pdop: pdop_arg.sqrt(),
        hdop: hdop_arg.sqrt(),
        vdop: vdop_arg.sqrt(),
        tdop: tdop_arg.sqrt(),
        // A single shared clock has exactly one clock column, but this entry
        // point carries no constellation identity, so the per-system vector is
        // empty (read `tdop` for the lone clock). The multi-system `dop_multi`
        // path, which does know each column's system, returns it tagged.
        system_tdops: Vec::new(),
    })
}

/// Compute the single-clock geometry cofactor matrix from line-of-sight rows.
///
/// This exposes the same cofactor matrix that [`dop`] uses internally:
/// `Q = (H^T W H)^-1`, with `H` rows `[-e_x, -e_y, -e_z, 1]`. The inverse is the
/// deterministic 4x4 cofactor expansion used by the 0-ULP DOP parity tests.
pub fn geometry_cofactor(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: Wgs84Geodetic,
) -> Result<GeometryCofactor, DopError> {
    geometry_cofactor_with_convention(los, weights, receiver, EnuConvention::GeodeticNormal)
}

/// [`geometry_cofactor`] with an explicit [`EnuConvention`] for the `position_enu`
/// block.
///
/// `position_ecef` and `state` are convention-independent; only `position_enu`
/// changes. [`EnuConvention::GeodeticNormal`] is the default
/// [`geometry_cofactor`] path.
pub fn geometry_cofactor_with_convention(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    convention: EnuConvention,
) -> Result<GeometryCofactor, DopError> {
    validate_dop_inputs(los, weights, receiver)?;
    if los.len() < 4 {
        return Err(DopError::TooFewSatellites);
    }

    let rows: Vec<[f64; 4]> = los.iter().map(|l| l.design_row()).collect();
    let a = normal_matrix_4_weighted_column_outer(&rows, weights).map_err(map_linear_error)?;
    let q = invert_4x4_cofactor(&a).ok_or(DopError::Singular)?;
    validate_cofactor_variances(&q)?;

    let r = enu_rotation(receiver, convention)?;
    let enu = rotate_pos_block(&q, &r);
    validate_matrix3(&enu, "position_enu")?;
    Ok(GeometryCofactor {
        state: q,
        position_ecef: position_block(&q),
        position_enu: enu,
    })
}

/// Position covariance from a single-clock GNSS design matrix.
///
/// `range_variance_scale_m2` multiplies the raw cofactor. For unit weights with a
/// common pseudorange standard deviation, pass `sigma_m * sigma_m`. If `weights`
/// already carry inverse variances, pass `1.0`.
pub fn position_covariance_from_geometry_m2(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    range_variance_scale_m2: f64,
) -> Result<PositionCovariance, DopError> {
    validate_variance_scale(range_variance_scale_m2)?;
    let cofactor = geometry_cofactor(los, weights, receiver)?;
    Ok(PositionCovariance {
        ecef_m2: scale_matrix3(cofactor.position_ecef, range_variance_scale_m2),
        enu_m2: scale_matrix3(cofactor.position_enu, range_variance_scale_m2),
    })
}

/// Horizontal confidence ellipse from a local ENU covariance matrix.
///
/// The 2D horizontal covariance is the east/north block of `covariance_enu_m2`.
/// The scale is the two-degree-of-freedom chi-square quantile
/// `-2 ln(1 - confidence)`.
pub fn horizontal_error_ellipse(
    covariance_enu_m2: [[f64; 3]; 3],
    confidence: f64,
) -> Result<HorizontalErrorEllipse, DopError> {
    validate_matrix3(&covariance_enu_m2, "covariance_enu_m2")?;
    let en_block = [
        [covariance_enu_m2[0][0], covariance_enu_m2[0][1]],
        [covariance_enu_m2[1][0], covariance_enu_m2[1][1]],
    ];
    let ellipse = error_ellipse_2x2(en_block, confidence).map_err(|err| match err {
        // Re-label only the covariance field so the GNSS wrapper reports its own
        // argument name; a confidence error keeps its field.
        DopError::InvalidInput {
            field: "covariance",
            reason,
        } => invalid_input("covariance_enu_m2", reason),
        other => other,
    })?;
    Ok(HorizontalErrorEllipse {
        confidence: ellipse.confidence,
        chi_square_scale: ellipse.chi_square_scale,
        semi_major_m: ellipse.semi_major,
        semi_minor_m: ellipse.semi_minor,
        azimuth_rad: ellipse.orientation_rad,
    })
}

/// Confidence ellipse from an arbitrary 2x2 covariance block.
///
/// Domain-neutral primitive: the semi-axes are scaled by the two-degree-of-
/// freedom chi-square quantile `-2 ln(1 - confidence)` applied to the
/// eigenvalues of the symmetrized block. [`horizontal_error_ellipse`] is the
/// GNSS-facing wrapper that feeds it the local east/north block, so both share
/// this eigensolve (and its goldens). The eigenvalues come from the closed-form
/// 2x2 symmetric solution `lambda = center +/- sqrt(half_delta^2 + b^2)`.
pub fn error_ellipse_2x2(
    covariance: [[f64; 2]; 2],
    confidence: f64,
) -> Result<ErrorEllipse2, DopError> {
    for row in &covariance {
        validate::finite_slice(row, "covariance").map_err(map_validation_error)?;
    }
    validate_confidence(confidence)?;

    let a = covariance[0][0];
    let b = 0.5 * (covariance[0][1] + covariance[1][0]);
    let c = covariance[1][1];
    let half_delta = 0.5 * (a - c);
    let center = 0.5 * (a + c);
    let root = (half_delta * half_delta + b * b).sqrt();
    let lambda_major = center + root;
    let lambda_minor = center - root;
    if !lambda_major.is_finite() || !lambda_minor.is_finite() || lambda_minor < -1.0e-12 {
        return Err(invalid_input("covariance", "not positive semidefinite"));
    }

    let chi_square_scale = -2.0 * (1.0 - confidence).ln();
    let semi_major = (lambda_major.max(0.0) * chi_square_scale).sqrt();
    let semi_minor = (lambda_minor.max(0.0) * chi_square_scale).sqrt();
    let orientation_rad = if root == 0.0 {
        0.0
    } else {
        0.5 * (2.0 * b).atan2(a - c)
    };
    Ok(ErrorEllipse2 {
        confidence,
        chi_square_scale,
        semi_major,
        semi_minor,
        orientation_rad,
    })
}

/// Horizontal confidence ellipse directly from line-of-sight rows and weights.
pub fn error_ellipse_from_geometry(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    range_variance_scale_m2: f64,
    confidence: f64,
) -> Result<HorizontalErrorEllipse, DopError> {
    let covariance =
        position_covariance_from_geometry_m2(los, weights, receiver, range_variance_scale_m2)?;
    horizontal_error_ellipse(covariance.enu_m2, confidence)
}

fn validate_dop_inputs(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: Wgs84Geodetic,
) -> Result<(), DopError> {
    if los.len() != weights.len() {
        return Err(invalid_input("weights", "length must match los"));
    }
    validate_los(los)?;
    validate_weights(weights)?;
    validate_receiver(receiver)
}

fn validate_los(los: &[LineOfSight]) -> Result<(), DopError> {
    for line in los {
        if !(line.e_x.is_finite() && line.e_y.is_finite() && line.e_z.is_finite()) {
            return Err(invalid_input("los", "not finite"));
        }
        let norm = (line.e_x * line.e_x + line.e_y * line.e_y + line.e_z * line.e_z).sqrt();
        if !norm.is_finite() {
            return Err(invalid_input("los", "not finite"));
        }
        if (norm - 1.0).abs() > LOS_UNIT_TOLERANCE {
            return Err(invalid_input("los", "not unit length"));
        }
    }
    Ok(())
}

fn validate_cofactor_variances(q: &[[f64; 4]; 4]) -> Result<(), DopError> {
    for row in q {
        validate::finite_slice(row, "cofactor").map_err(map_validation_error)?;
    }
    for (idx, row) in q.iter().enumerate() {
        let variance = row[idx];
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let negative_or_nan = !(variance >= 0.0);
        if negative_or_nan || !variance.is_finite() {
            return Err(DopError::Singular);
        }
    }
    Ok(())
}

fn validate_variance_scale(value: f64) -> Result<(), DopError> {
    if !value.is_finite() {
        return Err(invalid_input("range_variance_scale_m2", "not finite"));
    }
    if value < 0.0 {
        return Err(invalid_input("range_variance_scale_m2", "negative"));
    }
    Ok(())
}

fn validate_confidence(value: f64) -> Result<(), DopError> {
    if !value.is_finite() {
        return Err(invalid_input("confidence", "not finite"));
    }
    if !(0.0..1.0).contains(&value) {
        return Err(invalid_input("confidence", "out of range"));
    }
    Ok(())
}

fn validate_matrix3(matrix: &[[f64; 3]; 3], field: &'static str) -> Result<(), DopError> {
    for row in matrix {
        validate::finite_slice(row, field).map_err(map_validation_error)?;
    }
    Ok(())
}

fn position_block(q: &[[f64; 4]; 4]) -> [[f64; 3]; 3] {
    [
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ]
}

fn scale_matrix3(matrix: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    [
        [
            matrix[0][0] * scale,
            matrix[0][1] * scale,
            matrix[0][2] * scale,
        ],
        [
            matrix[1][0] * scale,
            matrix[1][1] * scale,
            matrix[1][2] * scale,
        ],
        [
            matrix[2][0] * scale,
            matrix[2][1] * scale,
            matrix[2][2] * scale,
        ],
    ]
}

fn validate_weights(weights: &[f64]) -> Result<(), DopError> {
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(invalid_input("weights", "not finite"));
    }
    if weights.iter().any(|&weight| weight < 0.0) {
        return Err(invalid_input("weights", "negative"));
    }
    Ok(())
}

fn validate_receiver(receiver: Wgs84Geodetic) -> Result<(), DopError> {
    if !(receiver.lat_rad.is_finite()
        && receiver.lon_rad.is_finite()
        && receiver.height_m.is_finite())
    {
        return Err(invalid_input("receiver", "not finite"));
    }
    if !(-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2).contains(&receiver.lat_rad) {
        return Err(invalid_input("receiver.lat_rad", "out of range"));
    }
    if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&receiver.lon_rad) {
        return Err(invalid_input("receiver.lon_rad", "out of range"));
    }
    Ok(())
}

fn validate_az_el_receiver(
    azimuth_deg: f64,
    elevation_deg: f64,
    receiver: Wgs84Geodetic,
) -> Result<(), DopError> {
    if !azimuth_deg.is_finite() {
        return Err(invalid_input("azimuth_deg", "not finite"));
    }
    if !elevation_deg.is_finite() {
        return Err(invalid_input("elevation_deg", "not finite"));
    }
    if !(-90.0..=90.0).contains(&elevation_deg) {
        return Err(invalid_input("elevation_deg", "out of range"));
    }
    validate_receiver(receiver)
}

fn invalid_input(field: &'static str, reason: &'static str) -> DopError {
    DopError::InvalidInput { field, reason }
}

fn map_linear_error(error: LinearError) -> DopError {
    match error {
        LinearError::InvalidInput { field, reason } => invalid_input(field, reason),
    }
}

fn map_validation_error(error: validate::FieldError) -> DopError {
    invalid_input(error.field(), error.reason())
}

// --- multi-system DOP (general (3 + n_systems) x (3 + n_systems)) -----------

/// `R Q R^T` for a 3x3 position cofactor block, formed as `(R Q) R^T`. Both
/// products use [`inline_rxr`]'s fixed left-to-right inner-sum order, so the
/// result is bit-identical to the explicit double loop this replaced.
fn rotate3(q: &[[f64; 3]; 3], r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    inline_rxr(&inline_rxr(r, q), &inline_tr(r))
}

/// Multi-system dilution of precision: like [`dop`] but with one receiver-clock
/// column per GNSS rather than a single shared clock.
///
/// `clock_index[i]` is the clock column (`0..n_clocks`) for satellite `i` - its
/// system's index in the solve's clock ordering. `systems[c]` is the GNSS that
/// owns clock column `c`, so `systems.len() == n_clocks` and the column ordering
/// matches the caller's clock ordering. The design row is therefore
/// `[-e_x, -e_y, -e_z, <one-hot over the n_clocks clock columns>]` and the
/// cofactor matrix is `(3 + n_clocks) x (3 + n_clocks)`. `PDOP`/`HDOP`/`VDOP` are
/// the position block (ENU-rotated, unambiguous); `GDOP` is the square root of
/// the full trace (all clocks); `TDOP` is the reference-system clock
/// (`Q[3][3]`). The per-system TDOPs (`sqrt(Q[3+c][3+c])` for every clock,
/// tagged with `systems[c]`) are returned in [`Dop::system_tdops`], in
/// clock-column order. Returns [`DopError::Singular`] for a rank-deficient
/// geometry.
///
/// This path uses a general symmetric inverse (see [`invert_symmetric_pd`]) and
/// is a deterministic geometry diagnostic, not a 0-ULP parity target; the
/// single-system [`dop`] retains the 0-ULP cofactor inverse.
///
/// Crate-internal: every `clock_index` must be in `0..n_clocks` (the solver
/// constructs them from the used satellites' system ordering, so this always
/// holds). It is not part of the public API because the index convention is
/// meaningless without the solver's clock ordering.
pub(crate) fn dop_multi(
    los: &[LineOfSight],
    clock_index: &[usize],
    systems: &[GnssSystem],
    n_clocks: usize,
    weights: &[f64],
    receiver: Wgs84Geodetic,
) -> Result<Dop, DopError> {
    validate_dop_multi_inputs(los, clock_index, systems, n_clocks, weights, receiver)?;
    let p = 3 + n_clocks;
    if los.len() < p {
        return Err(DopError::TooFewSatellites);
    }

    let mut a = vec![vec![0.0_f64; p]; p];
    for k in 0..los.len() {
        let mut row = vec![0.0_f64; p];
        row[0] = -los[k].e_x;
        row[1] = -los[k].e_y;
        row[2] = -los[k].e_z;
        row[3 + clock_index[k]] = 1.0;
        let w = weights[k];
        #[allow(clippy::needless_range_loop)]
        for i in 0..p {
            for j in 0..p {
                a[i][j] += row[i] * w * row[j];
            }
        }
    }
    let q = invert_symmetric_pd(&a).ok_or(DopError::Singular)?;

    let r = ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad);
    let qpos = [
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ];
    let enu = rotate3(&qpos, &r);

    let qe = enu[0][0];
    let qn = enu[1][1];
    let qu = enu[2][2];
    let qt = q[3][3];
    let trace: f64 = (0..p).map(|i| q[i][i]).sum();
    // Per-clock-column variances `Q[3+i][3+i]`, in clock-column order. Column 0
    // is the reference clock, so its variance is exactly `qt`.
    let system_tdop_args: Vec<f64> = (0..n_clocks).map(|i| q[3 + i][3 + i]).collect();

    let gdop_arg = trace;
    let pdop_arg = qe + qn + qu;
    let hdop_arg = qe + qn;
    let vdop_arg = qu;
    let tdop_arg = qt;
    // Every variance a DOP scalar takes the square root of - including each
    // per-system clock variance - must be finite and non-negative; a
    // rank-deficient geometry can leave a negative or non-finite diagonal even
    // when the full trace stays positive, so reject those here.
    for &arg in [gdop_arg, pdop_arg, hdop_arg, vdop_arg, tdop_arg]
        .iter()
        .chain(system_tdop_args.iter())
    {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let nonpositive_or_nan = !(arg >= 0.0);
        if nonpositive_or_nan || !arg.is_finite() {
            return Err(DopError::Singular);
        }
    }

    Ok(Dop {
        gdop: gdop_arg.sqrt(),
        pdop: pdop_arg.sqrt(),
        hdop: hdop_arg.sqrt(),
        vdop: vdop_arg.sqrt(),
        tdop: tdop_arg.sqrt(),
        system_tdops: system_tdop_args
            .iter()
            .enumerate()
            .map(|(i, &v)| (systems[i], v.sqrt()))
            .collect(),
    })
}

fn validate_dop_multi_inputs(
    los: &[LineOfSight],
    clock_index: &[usize],
    systems: &[GnssSystem],
    n_clocks: usize,
    weights: &[f64],
    receiver: Wgs84Geodetic,
) -> Result<(), DopError> {
    if los.len() != weights.len() {
        return Err(invalid_input("weights", "length must match los"));
    }
    if los.len() != clock_index.len() {
        return Err(invalid_input("clock_index", "length must match los"));
    }
    if n_clocks == 0 {
        return Err(invalid_input("n_clocks", "not positive"));
    }
    if systems.len() != n_clocks {
        return Err(invalid_input("systems", "length must match n_clocks"));
    }
    if clock_index.iter().any(|&idx| idx >= n_clocks) {
        return Err(invalid_input("clock_index", "out of range"));
    }
    validate_los(los)?;
    validate_weights(weights)?;
    validate_receiver(receiver)
}

#[cfg(all(test, sidereon_repo_tests))]
pub(crate) mod test_support {
    //! Internal accessors so the parity test can assert 0 ULP on the
    //! intermediates (normal matrix, cofactor matrix, ENU block) as well as the
    //! final scalars, without making them part of the public API.
    use super::*;

    pub(crate) fn normal_matrix_for(los: &[LineOfSight], weights: &[f64]) -> [[f64; 4]; 4] {
        let rows: Vec<[f64; 4]> = los.iter().map(|l| l.design_row()).collect();
        normal_matrix_4_weighted_column_outer(&rows, weights).expect("valid DOP test inputs")
    }

    pub(crate) fn det4_for(a: &[[f64; 4]; 4]) -> f64 {
        crate::astro::math::linear::det4_cofactor(a)
    }

    pub(crate) fn inv4_for(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
        invert_4x4_cofactor(a)
    }

    pub(crate) fn enu_block_for(q: &[[f64; 4]; 4], lat_rad: f64, lon_rad: f64) -> [[f64; 3]; 3] {
        let r = ecef_to_enu_rotation(lat_rad, lon_rad);
        rotate_pos_block(q, &r)
    }
}

#[cfg(test)]
mod public_api_tests {
    use super::*;

    fn receiver() -> Wgs84Geodetic {
        Wgs84Geodetic::new(45.0_f64.to_radians(), -75.0_f64.to_radians(), 100.0)
            .expect("valid receiver")
    }

    fn sample_geometry() -> (Vec<LineOfSight>, Vec<f64>, Wgs84Geodetic) {
        let rx = receiver();
        let az_el = [
            (5.0, 25.0),
            (80.0, 35.0),
            (155.0, 55.0),
            (235.0, 40.0),
            (310.0, 65.0),
        ];
        let los = az_el
            .into_iter()
            .map(|(az, el)| line_of_sight_from_az_el_deg(az, el, rx).expect("valid LOS"))
            .collect::<Vec<_>>();
        let weights = vec![1.0, 0.8, 1.4, 0.9, 1.1];
        (los, weights, rx)
    }

    #[test]
    fn geometry_cofactor_exposes_the_dop_position_block() {
        let (los, weights, rx) = sample_geometry();
        let d = dop(&los, &weights, rx).expect("DOP");
        let q = geometry_cofactor(&los, &weights, rx).expect("cofactor");

        let qe = q.position_enu[0][0];
        let qn = q.position_enu[1][1];
        let qu = q.position_enu[2][2];
        assert_eq!(d.pdop.to_bits(), (qe + qn + qu).sqrt().to_bits());
        assert_eq!(d.hdop.to_bits(), (qe + qn).sqrt().to_bits());
        assert_eq!(d.vdop.to_bits(), qu.sqrt().to_bits());
        assert_eq!(d.tdop.to_bits(), q.state[3][3].sqrt().to_bits());
    }

    #[test]
    fn position_covariance_scales_the_raw_cofactor() {
        let (los, weights, rx) = sample_geometry();
        let q = geometry_cofactor(&los, &weights, rx).expect("cofactor");
        let cov =
            position_covariance_from_geometry_m2(&los, &weights, rx, 4.0).expect("covariance");
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    cov.ecef_m2[i][j].to_bits(),
                    (q.position_ecef[i][j] * 4.0).to_bits()
                );
                assert_eq!(
                    cov.enu_m2[i][j].to_bits(),
                    (q.position_enu[i][j] * 4.0).to_bits()
                );
            }
        }
    }

    #[test]
    fn horizontal_error_ellipse_uses_chi_square_two_dof_scale() {
        let covariance = [[9.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 16.0]];
        let ellipse = horizontal_error_ellipse(covariance, 0.95).expect("ellipse");
        let expected_scale = -2.0 * (1.0_f64 - 0.95).ln();
        assert_eq!(ellipse.chi_square_scale.to_bits(), expected_scale.to_bits());
        assert!(ellipse.semi_major_m >= ellipse.semi_minor_m);
        assert!(ellipse.semi_minor_m > 0.0);
        assert!(ellipse.azimuth_rad.is_finite());
    }

    #[test]
    fn error_ellipse_2x2_matches_numpy_eigh() {
        // numpy.linalg.eigh of [[9, 2], [2, 4]] with the chi2(2) 0.95 scale.
        let ellipse = error_ellipse_2x2([[9.0, 2.0], [2.0, 4.0]], 0.95).expect("ellipse");
        assert!((ellipse.chi_square_scale - 5.99146454710798).abs() < 1e-12);
        assert!((ellipse.semi_major - 7.6240780089041085).abs() < 1e-12);
        assert!((ellipse.semi_minor - 4.445500379771495).abs() < 1e-12);
        assert!((ellipse.orientation_rad - 0.3373704711117763).abs() < 1e-12);
    }

    #[test]
    fn horizontal_error_ellipse_delegates_to_2x2_primitive() {
        // The GNSS wrapper must be byte-identical to the 2x2 primitive on the
        // east/north block.
        let cov3 = [[9.0, 2.0, 1.0], [2.0, 4.0, -3.0], [1.0, -3.0, 16.0]];
        let wrapper = horizontal_error_ellipse(cov3, 0.95).expect("wrapper");
        let primitive =
            error_ellipse_2x2([[cov3[0][0], cov3[0][1]], [cov3[1][0], cov3[1][1]]], 0.95)
                .expect("primitive");
        assert_eq!(
            wrapper.semi_major_m.to_bits(),
            primitive.semi_major.to_bits()
        );
        assert_eq!(
            wrapper.semi_minor_m.to_bits(),
            primitive.semi_minor.to_bits()
        );
        assert_eq!(
            wrapper.azimuth_rad.to_bits(),
            primitive.orientation_rad.to_bits()
        );
    }

    #[test]
    fn geocentric_convention_changes_only_horizontal_vertical_split() {
        let (los, weights, rx) = sample_geometry();
        let geodetic = dop(&los, &weights, rx).expect("geodetic DOP");
        let geocentric = dop_with_convention(&los, &weights, rx, EnuConvention::GeocentricRadial)
            .expect("geocentric DOP");

        // GDOP and TDOP read the unrotated state cofactor, so they are
        // bit-identical across conventions.
        assert_eq!(geodetic.gdop.to_bits(), geocentric.gdop.to_bits());
        assert_eq!(geodetic.tdop.to_bits(), geocentric.tdop.to_bits());

        // PDOP is the rotation-invariant position trace: equal to roundoff.
        assert!((geodetic.pdop - geocentric.pdop).abs() < 1e-9 * geodetic.pdop);

        // The H/V split moves with the ~0.19 deg up-axis deflection: different,
        // but on the order of 1e-3 relative.
        let hdop_rel = (geodetic.hdop - geocentric.hdop).abs() / geodetic.hdop;
        assert!(hdop_rel > 0.0, "convention must change HDOP");
        assert!(
            hdop_rel < 1e-2,
            "HDOP shift {hdop_rel} larger than expected"
        );
        assert_ne!(geodetic.vdop.to_bits(), geocentric.vdop.to_bits());
    }

    #[test]
    fn geocentric_convention_rejects_zero_radius_receiver() {
        // A receiver one equatorial radius below the equator/prime-meridian sits
        // at the geocenter (ECEF ~ origin), where geocentric "up" is undefined.
        // The geocentric-radial convention must reject it rather than silently
        // fall back to a +Z frame.
        let geocenter = Wgs84Geodetic::new(0.0, 0.0, -crate::astro::constants::earth::WGS84_A_M)
            .expect("valid geodetic receiver");
        let (los, weights, _) = sample_geometry();
        let err = dop_with_convention(&los, &weights, geocenter, EnuConvention::GeocentricRadial)
            .expect_err("zero-radius geocentric up must be rejected");
        assert!(matches!(
            err,
            DopError::InvalidInput {
                field: "receiver",
                ..
            }
        ));

        // The geodetic-normal convention does not use the radial axis, so the
        // same receiver is accepted there.
        assert!(
            dop_with_convention(&los, &weights, geocenter, EnuConvention::GeodeticNormal).is_ok()
        );
    }

    #[test]
    fn default_dop_equals_explicit_geodetic_convention_bit_for_bit() {
        let (los, weights, rx) = sample_geometry();
        let default = dop(&los, &weights, rx).expect("default");
        let explicit =
            dop_with_convention(&los, &weights, rx, EnuConvention::GeodeticNormal).expect("geo");
        assert_eq!(default.hdop.to_bits(), explicit.hdop.to_bits());
        assert_eq!(default.vdop.to_bits(), explicit.vdop.to_bits());
        assert_eq!(default.pdop.to_bits(), explicit.pdop.to_bits());
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
