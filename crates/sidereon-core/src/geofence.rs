//! Geodesic geofence containment with position uncertainty.
//!
//! Fences are polygons on WGS84. Vertices use geodetic latitude and longitude,
//! while ellipsoidal height is ignored. Boundary distances are metres, positive
//! inside the fence and negative outside it.

use crate::astro::math::special::erf;
use crate::dop::{self, PositionCovariance};
use crate::error_metrics::{self, PercentileRadius};
use crate::frame::Wgs84Geodetic;
use crate::geodesic::{geodesic_direct, geodesic_inverse, GeodesicError};

const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / core::f64::consts::PI;
const TWO_PI: f64 = 2.0 * core::f64::consts::PI;
const EDGE_MINIMIZE_ITERS: usize = 64;
const PSD_REL_TOL: f64 = 1.0e-12;
const RAY_EPS: f64 = 1.0e-12;
const RAY_PROBE: f64 = 1.0e-7;
const DEDUP_EPS: f64 = 1.0e-9;
const ORIENTATION_PROBE_MAX_M: f64 = 1.0;
const ORIENTATION_PROBE_FRACTION: f64 = 1.0e-3;
const ORIENTATION_CENTROID_NORM_TOL: f64 = 1.0e-12;

/// Maximum anchor-to-vertex, anchor-to-query, and edge length for the planar path.
///
/// When this 50 km bound is met, geodesic polar coordinates around the fence
/// anchor are used as a local plane. Outside the bound every public distance and
/// containment call uses the geodesic path.
pub const PLANAR_FAST_PATH_MAX_RADIUS_M: f64 = 50_000.0;

/// Boundary tolerance used when classifying a point on an edge, in metres.
pub const GEOFENCE_BOUNDARY_TOLERANCE_M: f64 = 1.0e-4;

#[allow(clippy::excessive_precision)]
const GL64_POSITIVE_NODES: [f64; 32] = [
    2.43502926634244325e-02,
    7.29931217877990424e-02,
    1.21462819296120544e-01,
    1.69644420423992831e-01,
    2.17423643740007083e-01,
    2.64687162208767424e-01,
    3.11322871990210970e-01,
    3.57220158337668126e-01,
    4.02270157963991570e-01,
    4.46366017253464087e-01,
    4.89403145707052956e-01,
    5.31279464019894565e-01,
    5.71895646202634000e-01,
    6.11155355172393278e-01,
    6.48965471254657311e-01,
    6.85236313054233270e-01,
    7.19881850171610771e-01,
    7.52819907260531940e-01,
    7.83972358943341385e-01,
    8.13265315122797539e-01,
    8.40629296252580316e-01,
    8.65999398154092770e-01,
    8.89315445995114140e-01,
    9.10522137078502825e-01,
    9.29569172131939570e-01,
    9.46411374858402765e-01,
    9.61008799652053658e-01,
    9.73326827789910975e-01,
    9.83336253884625977e-01,
    9.91013371476744287e-01,
    9.96340116771955220e-01,
    9.99305041735772170e-01,
];

#[allow(clippy::excessive_precision)]
const GL64_POSITIVE_WEIGHTS: [f64; 32] = [
    4.86909570091397237e-02,
    4.85754674415033935e-02,
    4.83447622348029404e-02,
    4.79993885964583034e-02,
    4.75401657148303222e-02,
    4.69681828162099788e-02,
    4.62847965813143539e-02,
    4.54916279274180727e-02,
    4.45905581637566079e-02,
    4.35837245293234157e-02,
    4.24735151236535352e-02,
    4.12625632426234581e-02,
    3.99537411327204536e-02,
    3.85501531786155635e-02,
    3.70551285402399844e-02,
    3.54722132568822401e-02,
    3.38051618371418630e-02,
    3.20579283548514254e-02,
    3.02346570724024884e-02,
    2.83396726142594486e-02,
    2.63774697150548978e-02,
    2.43527025687111306e-02,
    2.22701738083829967e-02,
    2.01348231535300216e-02,
    1.79517157756973571e-02,
    1.57260304760250269e-02,
    1.34630478967191179e-02,
    1.11681394601309738e-02,
    8.84675982636339009e-03,
    6.50445796897848854e-03,
    4.14703326056443250e-03,
    1.78328072169469699e-03,
];

/// Error returned by geofence construction and evaluation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum GeofenceError {
    /// Fewer than three distinct polygon vertices were supplied.
    #[error("geofence needs at least three vertices")]
    TooFewVertices,
    /// A fence input was malformed.
    #[error("invalid geofence input {field}: {reason}")]
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// Geodesic direct or inverse evaluation failed.
    #[error(transparent)]
    Geodesic(#[from] GeodesicError),
    /// ECEF covariance rotation failed.
    #[error("covariance rotation failed")]
    Dop(#[from] dop::DopError),
    /// Covariance or percentile radius validation failed.
    #[error("uncertainty validation failed")]
    ErrorMetrics(error_metrics::ErrorMetricsError),
}

/// Geodesic polygon fence on WGS84.
#[derive(Debug, Clone, PartialEq)]
pub struct Fence {
    vertices: Vec<Wgs84Geodetic>,
    edges: Vec<GeodesicEdge>,
    interior_winding_sign: f64,
    planar: Option<PlanarCache>,
}

impl Fence {
    /// Construct a fence from WGS84 geodetic vertices.
    ///
    /// The polygon edge between adjacent vertices is the shortest WGS84
    /// geodesic. A closing vertex equal to the first vertex is accepted and
    /// removed. Height is ignored by containment and distance calculations.
    pub fn new<I>(vertices: I) -> Result<Self, GeofenceError>
    where
        I: IntoIterator<Item = Wgs84Geodetic>,
    {
        let mut vertices: Vec<Wgs84Geodetic> = vertices.into_iter().collect();
        if vertices.len() >= 2
            && same_horizontal_position(vertices[0], vertices[vertices.len() - 1])?
        {
            vertices.pop();
        }
        if vertices.len() < 3 {
            return Err(GeofenceError::TooFewVertices);
        }
        validate_distinct_vertices(&vertices)?;

        let mut edges = Vec::with_capacity(vertices.len());
        for idx in 0..vertices.len() {
            let start = vertices[idx];
            let end = vertices[(idx + 1) % vertices.len()];
            let (length_m, azimuth_deg, _) = inverse_points(start, end)?;
            if length_m == 0.0 {
                return Err(invalid_input("vertices", "adjacent vertices must differ"));
            }
            edges.push(GeodesicEdge {
                start,
                length_m,
                azimuth_deg,
            });
        }

        let interior_winding_sign = interior_winding_sign(&vertices, &edges)?;
        let planar = build_planar_cache(&vertices, &edges)?;
        Ok(Self {
            vertices,
            edges,
            interior_winding_sign,
            planar,
        })
    }

    /// Fence vertices in their open polygon form.
    pub fn vertices(&self) -> &[Wgs84Geodetic] {
        &self.vertices
    }

    /// Number of polygon edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns true when the small-region planar path can evaluate this point.
    pub fn planar_fast_path_applies(&self, position: Wgs84Geodetic) -> bool {
        self.planar
            .as_ref()
            .and_then(|cache| project_from(cache.anchor, position).ok())
            .is_some_and(|projected| norm2(projected) <= PLANAR_FAST_PATH_MAX_RADIUS_M)
    }

    /// Boolean containment using geodesic polygon edges.
    ///
    /// Points within [`GEOFENCE_BOUNDARY_TOLERANCE_M`] of the boundary are
    /// classified as contained.
    pub fn contains(&self, position: Wgs84Geodetic) -> Result<bool, GeofenceError> {
        Ok(self.distance_to_boundary(position)? >= -GEOFENCE_BOUNDARY_TOLERANCE_M)
    }

    /// Signed distance to the polygon boundary, in metres.
    ///
    /// Positive values are inside, negative values are outside, and values near
    /// zero are on the boundary under [`GEOFENCE_BOUNDARY_TOLERANCE_M`].
    pub fn distance_to_boundary(&self, position: Wgs84Geodetic) -> Result<f64, GeofenceError> {
        Ok(self.boundary_distance_geodesic(position)?.signed_distance_m)
    }

    /// Fast signed boundary distance for small regions, in metres.
    ///
    /// Returns `None` when the fence or query point is outside
    /// [`PLANAR_FAST_PATH_MAX_RADIUS_M`]. This is a local-plane approximation.
    /// Use [`Self::distance_to_boundary`] when exact geodesic distance is
    /// required.
    pub fn distance_to_boundary_planar_fast(
        &self,
        position: Wgs84Geodetic,
    ) -> Result<Option<f64>, GeofenceError> {
        if let Some(cache) = &self.planar {
            if let Ok(point) = project_from(cache.anchor, position) {
                if norm2(point) <= PLANAR_FAST_PATH_MAX_RADIUS_M {
                    return Ok(Some(
                        boundary_distance_planar(&cache.vertices_m, point).signed_distance_m,
                    ));
                }
            }
        }
        Ok(None)
    }

    fn boundary_distance(
        &self,
        position: Wgs84Geodetic,
    ) -> Result<BoundaryDistance, GeofenceError> {
        self.boundary_distance_geodesic(position)
    }

    fn boundary_distance_geodesic(
        &self,
        position: Wgs84Geodetic,
    ) -> Result<BoundaryDistance, GeofenceError> {
        let mut nearest = None;
        for (edge_index, edge) in self.edges.iter().enumerate() {
            let candidate = nearest_on_edge(position, *edge, edge_index)?;
            if nearest
                .as_ref()
                .is_none_or(|closest: &NearestBoundary| candidate.distance_m < closest.distance_m)
            {
                nearest = Some(candidate);
            }
        }
        let nearest = nearest.ok_or(GeofenceError::TooFewVertices)?;
        if nearest.distance_m <= GEOFENCE_BOUNDARY_TOLERANCE_M {
            return Ok(BoundaryDistance {
                signed_distance_m: 0.0,
                normal_en: tangent_normal(self.edges[nearest.edge_index].azimuth_deg),
            });
        }

        let inside = contains_by_winding(position, &self.vertices, self.interior_winding_sign)?;
        let sign = if inside { 1.0 } else { -1.0 };
        let normal_en =
            normal_from_position_to_boundary(position, nearest.point, nearest.distance_m)
                .unwrap_or_else(|_| tangent_normal(self.edges[nearest.edge_index].azimuth_deg));
        Ok(BoundaryDistance {
            signed_distance_m: sign * nearest.distance_m,
            normal_en,
        })
    }
}

/// Position uncertainty accepted by probabilistic geofencing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionUncertainty {
    /// Local east-north-up covariance in square metres.
    EnuCovarianceM2([[f64; 3]; 3]),
    /// ECEF covariance in square metres, rotated at the supplied position.
    EcefCovarianceM2([[f64; 3]; 3]),
    /// DOP position covariance carrying both ECEF and ENU forms.
    PositionCovariance(PositionCovariance),
    /// Horizontal percentile radius, converted to an isotropic EN covariance.
    HorizontalRadius(PercentileRadius),
    /// Circular error probable radius in metres.
    CepRadiusM(f64),
}

impl From<PositionCovariance> for PositionUncertainty {
    fn from(value: PositionCovariance) -> Self {
        Self::PositionCovariance(value)
    }
}

impl From<PercentileRadius> for PositionUncertainty {
    fn from(value: PercentileRadius) -> Self {
        Self::HorizontalRadius(value)
    }
}

/// Probability integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityMethod {
    /// Closed-form Gaussian probability from signed distance and normal variance.
    BoundaryNormal,
    /// Fixed angular quadrature over the local planarized fence.
    PlanarQuadrature,
}

/// Options for [`containment_probability_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProbabilityOptions {
    /// Probability integration method.
    pub method: ProbabilityMethod,
}

impl Default for ProbabilityOptions {
    fn default() -> Self {
        Self {
            method: ProbabilityMethod::BoundaryNormal,
        }
    }
}

/// Position estimate used by probabilistic crossing detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeofencePositionEstimate {
    /// Estimated WGS84 geodetic position.
    pub position: Wgs84Geodetic,
    /// Position uncertainty associated with the estimate.
    pub uncertainty: PositionUncertainty,
}

/// Hysteresis confidence for probabilistic crossing detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityHysteresis {
    /// Required inside probability before emitting an entered event.
    pub enter_confidence: f64,
    /// Required outside probability before emitting a left event.
    pub leave_confidence: f64,
}

impl ProbabilityHysteresis {
    /// Construct hysteresis thresholds.
    ///
    /// Each confidence must be greater than 0.5 and less than 1.0. Values in
    /// that range create a gap between the enter and leave conditions.
    pub fn new(enter_confidence: f64, leave_confidence: f64) -> Result<Self, GeofenceError> {
        validate_confidence("enter_confidence", enter_confidence)?;
        validate_confidence("leave_confidence", leave_confidence)?;
        Ok(Self {
            enter_confidence,
            leave_confidence,
        })
    }
}

impl Default for ProbabilityHysteresis {
    fn default() -> Self {
        Self {
            enter_confidence: 0.95,
            leave_confidence: 0.95,
        }
    }
}

/// Crossing event kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingKind {
    /// The sequence entered the fence.
    Entered,
    /// The sequence left the fence.
    Left,
}

/// Crossing event reported for a sequence sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingEvent {
    /// Index of the sample that first satisfied the crossing condition.
    pub sample_index: usize,
    /// Direction of the crossing.
    pub kind: CrossingKind,
    /// Inside probability at the sample.
    pub inside_probability: f64,
}

/// Boolean containment for one position and fence.
pub fn containment(position: Wgs84Geodetic, fence: &Fence) -> Result<bool, GeofenceError> {
    fence.contains(position)
}

/// Signed boundary distance for one position and fence, in metres.
pub fn distance_to_boundary(position: Wgs84Geodetic, fence: &Fence) -> Result<f64, GeofenceError> {
    fence.distance_to_boundary(position)
}

/// Containment probability from position uncertainty.
///
/// This uses [`ProbabilityMethod::BoundaryNormal`]. It evaluates the signed
/// boundary distance and projects the horizontal covariance onto the nearest
/// boundary normal. The resulting half-space probability is exact for a locally
/// straight boundary and a Gaussian estimate in the local EN plane.
pub fn containment_probability(
    position: Wgs84Geodetic,
    uncertainty: PositionUncertainty,
    fence: &Fence,
) -> Result<f64, GeofenceError> {
    containment_probability_with_options(
        position,
        uncertainty,
        fence,
        ProbabilityOptions::default(),
    )
}

/// Containment probability with an explicit integration method.
pub fn containment_probability_with_options(
    position: Wgs84Geodetic,
    uncertainty: PositionUncertainty,
    fence: &Fence,
    options: ProbabilityOptions,
) -> Result<f64, GeofenceError> {
    let boundary = fence.boundary_distance(position)?;
    let covariance = uncertainty_to_enu_m2(uncertainty, position)?;
    match options.method {
        ProbabilityMethod::BoundaryNormal => {
            probability_boundary_normal_or_quadrature(fence, position, boundary, covariance)
        }
        ProbabilityMethod::PlanarQuadrature => {
            probability_planar_quadrature(fence, position, boundary, covariance)
        }
    }
}

/// Boolean crossing detection over a position sequence.
pub fn crossing(
    positions: &[Wgs84Geodetic],
    fence: &Fence,
) -> Result<Vec<CrossingEvent>, GeofenceError> {
    let mut events = Vec::new();
    let Some((&first, rest)) = positions.split_first() else {
        return Ok(events);
    };
    let mut inside = fence.contains(first)?;
    for (offset, &position) in rest.iter().enumerate() {
        let next_inside = fence.contains(position)?;
        if next_inside != inside {
            events.push(CrossingEvent {
                sample_index: offset + 1,
                kind: if next_inside {
                    CrossingKind::Entered
                } else {
                    CrossingKind::Left
                },
                inside_probability: if next_inside { 1.0 } else { 0.0 },
            });
            inside = next_inside;
        }
    }
    Ok(events)
}

/// Probabilistic crossing detection with probability hysteresis.
pub fn crossing_probability(
    samples: &[GeofencePositionEstimate],
    fence: &Fence,
    hysteresis: ProbabilityHysteresis,
) -> Result<Vec<CrossingEvent>, GeofenceError> {
    crossing_probability_with_options(samples, fence, hysteresis, ProbabilityOptions::default())
}

/// Probabilistic crossing detection with explicit probability options.
pub fn crossing_probability_with_options(
    samples: &[GeofencePositionEstimate],
    fence: &Fence,
    hysteresis: ProbabilityHysteresis,
    options: ProbabilityOptions,
) -> Result<Vec<CrossingEvent>, GeofenceError> {
    validate_confidence("enter_confidence", hysteresis.enter_confidence)?;
    validate_confidence("leave_confidence", hysteresis.leave_confidence)?;
    let mut events = Vec::new();
    let Some((first, rest)) = samples.split_first() else {
        return Ok(events);
    };
    let first_probability =
        containment_probability_with_options(first.position, first.uncertainty, fence, options)?;
    let outside_threshold = 1.0 - hysteresis.leave_confidence;
    let mut state = HysteresisState::from_probability(
        first_probability,
        hysteresis.enter_confidence,
        outside_threshold,
    );

    for (offset, sample) in rest.iter().enumerate() {
        let probability = containment_probability_with_options(
            sample.position,
            sample.uncertainty,
            fence,
            options,
        )?;
        match state {
            HysteresisState::Unknown => {
                state = HysteresisState::from_probability(
                    probability,
                    hysteresis.enter_confidence,
                    outside_threshold,
                );
            }
            HysteresisState::Outside if probability >= hysteresis.enter_confidence => {
                state = HysteresisState::Inside;
                events.push(CrossingEvent {
                    sample_index: offset + 1,
                    kind: CrossingKind::Entered,
                    inside_probability: probability,
                });
            }
            HysteresisState::Inside if probability <= outside_threshold => {
                state = HysteresisState::Outside;
                events.push(CrossingEvent {
                    sample_index: offset + 1,
                    kind: CrossingKind::Left,
                    inside_probability: probability,
                });
            }
            _ => {}
        }
    }
    Ok(events)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GeodesicEdge {
    start: Wgs84Geodetic,
    length_m: f64,
    azimuth_deg: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct PlanarCache {
    anchor: Wgs84Geodetic,
    vertices_m: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Copy)]
struct BoundaryDistance {
    signed_distance_m: f64,
    normal_en: [f64; 2],
}

#[derive(Debug, Clone, Copy)]
struct NearestBoundary {
    distance_m: f64,
    point: Wgs84Geodetic,
    edge_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct HorizontalEigen {
    major_m2: f64,
    minor_m2: f64,
    major_axis: [f64; 2],
    minor_axis: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HysteresisState {
    Unknown,
    Inside,
    Outside,
}

impl HysteresisState {
    fn from_probability(probability: f64, enter_confidence: f64, outside_threshold: f64) -> Self {
        if probability >= enter_confidence {
            Self::Inside
        } else if probability <= outside_threshold {
            Self::Outside
        } else {
            Self::Unknown
        }
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> GeofenceError {
    GeofenceError::InvalidInput { field, reason }
}

fn validate_probability(field: &'static str, probability: f64) -> Result<(), GeofenceError> {
    if probability.is_finite() && probability > 0.0 && probability < 1.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be in (0, 1)"))
    }
}

fn validate_confidence(field: &'static str, probability: f64) -> Result<(), GeofenceError> {
    if probability.is_finite() && probability > 0.5 && probability < 1.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be in (0.5, 1)"))
    }
}

fn same_horizontal_position(a: Wgs84Geodetic, b: Wgs84Geodetic) -> Result<bool, GeofenceError> {
    if a.lat_rad.to_bits() == b.lat_rad.to_bits() && a.lon_rad.to_bits() == b.lon_rad.to_bits() {
        return Ok(true);
    }
    let (distance_m, _, _) = inverse_points(a, b)?;
    Ok(distance_m <= GEOFENCE_BOUNDARY_TOLERANCE_M)
}

fn validate_distinct_vertices(vertices: &[Wgs84Geodetic]) -> Result<(), GeofenceError> {
    let mut distinct_count = 0;
    'vertex: for (idx, &vertex) in vertices.iter().enumerate() {
        for &previous in &vertices[..idx] {
            if same_horizontal_position(vertex, previous)? {
                continue 'vertex;
            }
        }
        distinct_count += 1;
    }
    if distinct_count < 3 {
        return Err(GeofenceError::TooFewVertices);
    }
    if distinct_count != vertices.len() {
        return Err(invalid_input("vertices", "vertices must be distinct"));
    }
    Ok(())
}

fn interior_winding_sign(
    vertices: &[Wgs84Geodetic],
    edges: &[GeodesicEdge],
) -> Result<f64, GeofenceError> {
    let mut xyz = [0.0_f64; 3];
    for &vertex in vertices {
        let cos_lat = libm::cos(vertex.lat_rad);
        xyz[0] += cos_lat * libm::cos(vertex.lon_rad);
        xyz[1] += cos_lat * libm::sin(vertex.lon_rad);
        xyz[2] += libm::sin(vertex.lat_rad);
    }
    let norm = (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt();
    if norm > ORIENTATION_CENTROID_NORM_TOL {
        let lat = libm::asin(xyz[2] / norm);
        let lon = libm::atan2(xyz[1], xyz[0]);
        let probe = Wgs84Geodetic::new(lat, lon, 0.0)
            .map_err(|_| invalid_input("vertices", "centroid is outside geodetic range"))?;
        let sum = winding_sum(probe, vertices)?;
        if sum.abs() > core::f64::consts::PI {
            return Ok(sum.signum());
        }
    }

    for edge in edges {
        if let Some(sign) = edge_orientation_probe(vertices, *edge)? {
            return Ok(sign);
        }
    }
    Err(invalid_input(
        "vertices",
        "interior orientation is ambiguous",
    ))
}

fn edge_orientation_probe(
    vertices: &[Wgs84Geodetic],
    edge: GeodesicEdge,
) -> Result<Option<f64>, GeofenceError> {
    let midpoint = direct_point(edge.start, edge.azimuth_deg, 0.5 * edge.length_m)?;
    let offset_m = (edge.length_m * ORIENTATION_PROBE_FRACTION).clamp(
        GEOFENCE_BOUNDARY_TOLERANCE_M * 10.0,
        ORIENTATION_PROBE_MAX_M,
    );
    let left = direct_point(midpoint, edge.azimuth_deg - 90.0, offset_m)?;
    let right = direct_point(midpoint, edge.azimuth_deg + 90.0, offset_m)?;
    let left_sum = winding_sum(left, vertices)?;
    let right_sum = winding_sum(right, vertices)?;
    let left_inside = left_sum.abs() > core::f64::consts::PI;
    let right_inside = right_sum.abs() > core::f64::consts::PI;
    match (left_inside, right_inside) {
        (true, false) => Ok(Some(left_sum.signum())),
        (false, true) => Ok(Some(right_sum.signum())),
        (true, true) => Err(invalid_input(
            "vertices",
            "interior orientation is ambiguous",
        )),
        (false, false) => Ok(None),
    }
}

fn inverse_points(a: Wgs84Geodetic, b: Wgs84Geodetic) -> Result<(f64, f64, f64), GeofenceError> {
    Ok(geodesic_inverse(
        a.lat_rad * RAD_TO_DEG,
        a.lon_rad * RAD_TO_DEG,
        b.lat_rad * RAD_TO_DEG,
        b.lon_rad * RAD_TO_DEG,
    )?)
}

fn direct_point(
    start: Wgs84Geodetic,
    azimuth_deg: f64,
    distance_m: f64,
) -> Result<Wgs84Geodetic, GeofenceError> {
    let (lat_deg, lon_deg, _) = geodesic_direct(
        start.lat_rad * RAD_TO_DEG,
        start.lon_rad * RAD_TO_DEG,
        azimuth_deg,
        distance_m,
    )?;
    Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
        .map_err(|_| invalid_input("geodesic_direct", "returned invalid geodetic position"))
}

fn build_planar_cache(
    vertices: &[Wgs84Geodetic],
    edges: &[GeodesicEdge],
) -> Result<Option<PlanarCache>, GeofenceError> {
    if edges
        .iter()
        .any(|edge| edge.length_m > PLANAR_FAST_PATH_MAX_RADIUS_M)
    {
        return Ok(None);
    }
    let anchor = vertices[0];
    let mut vertices_m = Vec::with_capacity(vertices.len());
    for &vertex in vertices {
        let projected = project_from(anchor, vertex)?;
        if norm2(projected) > PLANAR_FAST_PATH_MAX_RADIUS_M {
            return Ok(None);
        }
        vertices_m.push(projected);
    }
    Ok(Some(PlanarCache { anchor, vertices_m }))
}

fn project_from(origin: Wgs84Geodetic, point: Wgs84Geodetic) -> Result<[f64; 2], GeofenceError> {
    let (distance_m, azimuth_deg, _) = inverse_points(origin, point)?;
    if distance_m == 0.0 {
        return Ok([0.0, 0.0]);
    }
    let azimuth_rad = azimuth_deg * DEG_TO_RAD;
    Ok([
        distance_m * libm::sin(azimuth_rad),
        distance_m * libm::cos(azimuth_rad),
    ])
}

fn boundary_distance_planar(vertices_m: &[[f64; 2]], point: [f64; 2]) -> BoundaryDistance {
    let mut closest_distance = f64::INFINITY;
    let mut closest_normal = [1.0, 0.0];
    for idx in 0..vertices_m.len() {
        let a = vertices_m[idx];
        let b = vertices_m[(idx + 1) % vertices_m.len()];
        let (distance, normal) = distance_to_segment_planar(point, a, b);
        if distance < closest_distance {
            closest_distance = distance;
            closest_normal = normal;
        }
    }
    if closest_distance <= GEOFENCE_BOUNDARY_TOLERANCE_M {
        return BoundaryDistance {
            signed_distance_m: 0.0,
            normal_en: closest_normal,
        };
    }
    let inside = point_in_polygon_planar(vertices_m, point);
    BoundaryDistance {
        signed_distance_m: if inside {
            closest_distance
        } else {
            -closest_distance
        },
        normal_en: closest_normal,
    }
}

fn distance_to_segment_planar(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, [f64; 2]) {
    let ab = sub2(b, a);
    let ap = sub2(point, a);
    let denom = dot2(ab, ab);
    let t = if denom == 0.0 {
        0.0
    } else {
        (dot2(ap, ab) / denom).clamp(0.0, 1.0)
    };
    let nearest = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let delta = sub2(point, nearest);
    let distance = norm2(delta);
    if distance > 0.0 {
        (distance, [delta[0] / distance, delta[1] / distance])
    } else {
        let tangent_norm = norm2(ab);
        if tangent_norm == 0.0 {
            (0.0, [1.0, 0.0])
        } else {
            (0.0, [-ab[1] / tangent_norm, ab[0] / tangent_norm])
        }
    }
}

fn nearest_on_edge(
    position: Wgs84Geodetic,
    edge: GeodesicEdge,
    edge_index: usize,
) -> Result<NearestBoundary, GeofenceError> {
    let mut closest = nearest_at(edge, position, 0.0, edge_index)?;
    let end = nearest_at(edge, position, edge.length_m, edge_index)?;
    if end.distance_m < closest.distance_m {
        closest = end;
    }

    let mut lo = 0.0;
    let mut hi = edge.length_m;
    for _ in 0..EDGE_MINIMIZE_ITERS {
        let third = (hi - lo) / 3.0;
        let left = lo + third;
        let right = hi - third;
        let left_distance = distance_at(edge, position, left)?;
        let right_distance = distance_at(edge, position, right)?;
        if left_distance < right_distance {
            hi = right;
        } else {
            lo = left;
        }
    }
    let mid = 0.5 * (lo + hi);
    let interior = nearest_at(edge, position, mid, edge_index)?;
    if interior.distance_m < closest.distance_m {
        closest = interior;
    }
    Ok(closest)
}

fn nearest_at(
    edge: GeodesicEdge,
    position: Wgs84Geodetic,
    along_m: f64,
    edge_index: usize,
) -> Result<NearestBoundary, GeofenceError> {
    let point = direct_point(edge.start, edge.azimuth_deg, along_m)?;
    let (distance_m, _, _) = inverse_points(position, point)?;
    Ok(NearestBoundary {
        distance_m,
        point,
        edge_index,
    })
}

fn distance_at(
    edge: GeodesicEdge,
    position: Wgs84Geodetic,
    along_m: f64,
) -> Result<f64, GeofenceError> {
    let point = direct_point(edge.start, edge.azimuth_deg, along_m)?;
    let (distance_m, _, _) = inverse_points(position, point)?;
    Ok(distance_m)
}

fn normal_from_position_to_boundary(
    position: Wgs84Geodetic,
    boundary: Wgs84Geodetic,
    distance_m: f64,
) -> Result<[f64; 2], GeofenceError> {
    if distance_m == 0.0 {
        return Ok([1.0, 0.0]);
    }
    let (_, azimuth_deg, _) = inverse_points(position, boundary)?;
    let azimuth_rad = azimuth_deg * DEG_TO_RAD;
    Ok([libm::sin(azimuth_rad), libm::cos(azimuth_rad)])
}

fn tangent_normal(azimuth_deg: f64) -> [f64; 2] {
    let azimuth_rad = azimuth_deg * DEG_TO_RAD;
    let tangent = [libm::sin(azimuth_rad), libm::cos(azimuth_rad)];
    [-tangent[1], tangent[0]]
}

fn contains_by_winding(
    position: Wgs84Geodetic,
    vertices: &[Wgs84Geodetic],
    interior_sign: f64,
) -> Result<bool, GeofenceError> {
    let sum = winding_sum(position, vertices)?;
    if sum.abs() <= core::f64::consts::PI {
        return Ok(false);
    }
    if interior_sign != 0.0 {
        Ok(sum.signum() == interior_sign)
    } else {
        Ok(false)
    }
}

fn winding_sum(position: Wgs84Geodetic, vertices: &[Wgs84Geodetic]) -> Result<f64, GeofenceError> {
    let mut sum = 0.0;
    for idx in 0..vertices.len() {
        let a = vertices[idx];
        let b = vertices[(idx + 1) % vertices.len()];
        let (distance_a, azimuth_a_deg, _) = inverse_points(position, a)?;
        let (distance_b, azimuth_b_deg, _) = inverse_points(position, b)?;
        if distance_a <= GEOFENCE_BOUNDARY_TOLERANCE_M
            || distance_b <= GEOFENCE_BOUNDARY_TOLERANCE_M
        {
            return Ok(TWO_PI);
        }
        sum += wrap_pi((azimuth_b_deg - azimuth_a_deg) * DEG_TO_RAD);
    }
    Ok(sum)
}

fn point_in_polygon_planar(vertices: &[[f64; 2]], point: [f64; 2]) -> bool {
    let mut inside = false;
    let mut prev = vertices[vertices.len() - 1];
    for &curr in vertices {
        let crosses = (curr[1] > point[1]) != (prev[1] > point[1]);
        if crosses {
            let x_at_y = curr[0] + (point[1] - curr[1]) * (prev[0] - curr[0]) / (prev[1] - curr[1]);
            if point[0] < x_at_y {
                inside = !inside;
            }
        }
        prev = curr;
    }
    inside
}

fn uncertainty_to_enu_m2(
    uncertainty: PositionUncertainty,
    position: Wgs84Geodetic,
) -> Result<[[f64; 3]; 3], GeofenceError> {
    let covariance = match uncertainty {
        PositionUncertainty::EnuCovarianceM2(covariance) => covariance,
        PositionUncertainty::EcefCovarianceM2(covariance) => {
            dop::rotate_covariance_ecef_to_enu_m2(covariance, position)?
        }
        PositionUncertainty::PositionCovariance(covariance) => covariance.enu_m2,
        PositionUncertainty::HorizontalRadius(radius) => covariance_from_horizontal_radius(radius)?,
        PositionUncertainty::CepRadiusM(radius_m) => {
            covariance_from_horizontal_radius(PercentileRadius {
                probability: 0.5,
                radius_m,
                approx_m: radius_m,
                approx_valid: true,
            })?
        }
    };
    error_metrics::metrics_from_enu_covariance_m2(covariance)
        .map_err(GeofenceError::ErrorMetrics)?;
    Ok(covariance)
}

fn covariance_from_horizontal_radius(
    radius: PercentileRadius,
) -> Result<[[f64; 3]; 3], GeofenceError> {
    validate_probability("radius.probability", radius.probability)?;
    if !radius.radius_m.is_finite() || radius.radius_m < 0.0 {
        return Err(invalid_input(
            "radius.radius_m",
            "must be finite non-negative",
        ));
    }
    let sigma2 = if radius.radius_m == 0.0 {
        0.0
    } else {
        let scale = -2.0 * libm::log(1.0 - radius.probability);
        radius.radius_m * radius.radius_m / scale
    };
    Ok([[sigma2, 0.0, 0.0], [0.0, sigma2, 0.0], [0.0, 0.0, 0.0]])
}

fn probability_boundary_normal_or_quadrature(
    fence: &Fence,
    position: Wgs84Geodetic,
    boundary: BoundaryDistance,
    covariance: [[f64; 3]; 3],
) -> Result<f64, GeofenceError> {
    let trace = covariance[0][0].max(0.0) + covariance[1][1].max(0.0);
    let variance = projected_variance(covariance, boundary.normal_en).max(0.0);
    if trace > 0.0 && variance <= trace * PSD_REL_TOL {
        return probability_planar_quadrature(fence, position, boundary, covariance);
    }
    if variance == 0.0 {
        if boundary.signed_distance_m == 0.0 {
            return Ok(1.0);
        }
        return Ok(if boundary.signed_distance_m > 0.0 {
            1.0
        } else {
            0.0
        });
    }
    Ok(normal_cdf(boundary.signed_distance_m / variance.sqrt()).clamp(0.0, 1.0))
}

fn probability_planar_quadrature(
    fence: &Fence,
    position: Wgs84Geodetic,
    boundary: BoundaryDistance,
    covariance: [[f64; 3]; 3],
) -> Result<f64, GeofenceError> {
    let vertices = local_vertices(fence, position)?;
    let eigen = horizontal_eigen(covariance)?;
    let scale = eigen.major_m2.abs().max(1.0);
    if eigen.major_m2 <= scale * PSD_REL_TOL {
        if boundary.signed_distance_m == 0.0 {
            return Ok(1.0);
        }
        return Ok(if boundary.signed_distance_m > 0.0 {
            1.0
        } else {
            0.0
        });
    }
    let origin_inside = boundary.signed_distance_m > 0.0;
    let origin_on_boundary = boundary.signed_distance_m == 0.0;
    if eigen.minor_m2 <= eigen.major_m2 * PSD_REL_TOL {
        return Ok(rank_one_probability(
            &vertices,
            origin_inside,
            origin_on_boundary,
            scale2(eigen.major_axis, eigen.major_m2.sqrt()),
        ));
    }

    let major_sigma = eigen.major_m2.sqrt();
    let minor_sigma = eigen.minor_m2.sqrt();
    let integral = integrate_gl64(0.0, TWO_PI, |theta| {
        let unit = [libm::cos(theta), libm::sin(theta)];
        let direction = add2(
            scale2(eigen.major_axis, major_sigma * unit[0]),
            scale2(eigen.minor_axis, minor_sigma * unit[1]),
        );
        radial_mass_along_direction(&vertices, origin_inside, origin_on_boundary, direction)
    });
    Ok((integral / TWO_PI).clamp(0.0, 1.0))
}

fn local_vertices(fence: &Fence, position: Wgs84Geodetic) -> Result<Vec<[f64; 2]>, GeofenceError> {
    fence
        .vertices
        .iter()
        .map(|&vertex| project_from(position, vertex))
        .collect()
}

fn projected_variance(covariance: [[f64; 3]; 3], normal_en: [f64; 2]) -> f64 {
    let e = normal_en[0];
    let n = normal_en[1];
    e * e * covariance[0][0] + 2.0 * e * n * covariance[0][1] + n * n * covariance[1][1]
}

fn horizontal_eigen(covariance: [[f64; 3]; 3]) -> Result<HorizontalEigen, GeofenceError> {
    let a = covariance[0][0];
    let b = 0.5 * (covariance[0][1] + covariance[1][0]);
    let c = covariance[1][1];
    let center = 0.5 * (a + c);
    let half_delta = 0.5 * (a - c);
    let root = (half_delta * half_delta + b * b).sqrt();
    let major = center + root;
    let minor = center - root;
    let scale = major.abs().max(minor.abs()).max(1.0);
    if !major.is_finite() || !minor.is_finite() || minor < -scale * PSD_REL_TOL {
        return Err(GeofenceError::ErrorMetrics(
            error_metrics::ErrorMetricsError::NotPositiveSemidefinite,
        ));
    }
    let angle = if root == 0.0 {
        0.0
    } else {
        0.5 * libm::atan2(2.0 * b, a - c)
    };
    let major_axis = [libm::cos(angle), libm::sin(angle)];
    let minor_axis = [-libm::sin(angle), libm::cos(angle)];
    Ok(HorizontalEigen {
        major_m2: major.max(0.0),
        minor_m2: minor.max(0.0),
        major_axis,
        minor_axis,
    })
}

fn radial_mass_along_direction(
    vertices: &[[f64; 2]],
    origin_inside: bool,
    origin_on_boundary: bool,
    direction: [f64; 2],
) -> f64 {
    if norm2(direction) == 0.0 {
        return if origin_on_boundary || origin_inside {
            1.0
        } else {
            0.0
        };
    }
    let mut hits = ray_intersections(vertices, direction);
    sort_dedup(&mut hits);
    let mut inside = if origin_on_boundary {
        point_in_polygon_planar(vertices, scale2(direction, RAY_PROBE))
    } else {
        origin_inside
    };
    let mut previous = 0.0_f64;
    let mut mass = 0.0_f64;
    for hit in hits {
        if inside {
            mass += libm::exp(-0.5 * previous * previous) - libm::exp(-0.5 * hit * hit);
        }
        inside = !inside;
        previous = hit;
    }
    if inside {
        mass += libm::exp(-0.5 * previous * previous);
    }
    mass.clamp(0.0, 1.0)
}

fn ray_intersections(vertices: &[[f64; 2]], direction: [f64; 2]) -> Vec<f64> {
    let mut hits = Vec::new();
    for idx in 0..vertices.len() {
        let p = vertices[idx];
        let s = sub2(vertices[(idx + 1) % vertices.len()], p);
        let denom = cross2(direction, s);
        if denom.abs() <= RAY_EPS {
            continue;
        }
        let ray_t = cross2(p, s) / denom;
        let segment_t = cross2(p, direction) / denom;
        if ray_t > RAY_EPS && (-RAY_EPS..=1.0 + RAY_EPS).contains(&segment_t) {
            hits.push(ray_t);
        }
    }
    hits
}

fn rank_one_probability(
    vertices: &[[f64; 2]],
    origin_inside: bool,
    origin_on_boundary: bool,
    direction: [f64; 2],
) -> f64 {
    if norm2(direction) == 0.0 {
        return if origin_on_boundary || origin_inside {
            1.0
        } else {
            0.0
        };
    }
    let mut hits = line_intersections(vertices, direction);
    sort_dedup(&mut hits);
    if hits.is_empty() {
        if origin_on_boundary {
            let plus_inside = point_in_polygon_planar(vertices, scale2(direction, RAY_PROBE));
            let minus_inside = point_in_polygon_planar(vertices, scale2(direction, -RAY_PROBE));
            return match (minus_inside, plus_inside) {
                (true, true) => 1.0,
                (false, false) => 0.0,
                _ => 0.5,
            };
        }
        return if origin_inside { 1.0 } else { 0.0 };
    }
    let mut probability = 0.0;
    for idx in 0..=hits.len() {
        let lo = if idx == 0 {
            f64::NEG_INFINITY
        } else {
            hits[idx - 1]
        };
        let hi = if idx == hits.len() {
            f64::INFINITY
        } else {
            hits[idx]
        };
        let probe = if lo.is_infinite() {
            hi - 1.0
        } else if hi.is_infinite() {
            lo + 1.0
        } else {
            0.5 * (lo + hi)
        };
        let point = scale2(direction, probe);
        if point_in_polygon_planar(vertices, point) {
            probability += normal_interval_probability(lo, hi);
        }
    }
    probability.clamp(0.0, 1.0)
}

fn line_intersections(vertices: &[[f64; 2]], direction: [f64; 2]) -> Vec<f64> {
    let mut hits = Vec::new();
    for idx in 0..vertices.len() {
        let p = vertices[idx];
        let s = sub2(vertices[(idx + 1) % vertices.len()], p);
        let denom = cross2(direction, s);
        if denom.abs() <= RAY_EPS {
            continue;
        }
        let line_t = cross2(p, s) / denom;
        let segment_t = cross2(p, direction) / denom;
        if (-RAY_EPS..=1.0 + RAY_EPS).contains(&segment_t) {
            hits.push(line_t);
        }
    }
    hits
}

fn normal_interval_probability(lo: f64, hi: f64) -> f64 {
    let cdf_hi = if hi.is_infinite() && hi.is_sign_positive() {
        1.0
    } else {
        normal_cdf(hi)
    };
    let cdf_lo = if lo.is_infinite() && lo.is_sign_negative() {
        0.0
    } else {
        normal_cdf(lo)
    };
    cdf_hi - cdf_lo
}

fn integrate_gl64<F>(a: f64, b: f64, mut f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0;
    for idx in 0..GL64_POSITIVE_NODES.len() {
        let node = GL64_POSITIVE_NODES[idx];
        let weight = GL64_POSITIVE_WEIGHTS[idx];
        sum += weight * (f(center - half * node) + f(center + half * node));
    }
    half * sum
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * core::f64::consts::FRAC_1_SQRT_2))
}

fn sort_dedup(values: &mut Vec<f64>) {
    values.sort_by(f64::total_cmp);
    let mut deduped = Vec::with_capacity(values.len());
    for &value in values.iter() {
        if deduped
            .last()
            .is_none_or(|last: &f64| (value - *last).abs() > DEDUP_EPS)
        {
            deduped.push(value);
        }
    }
    *values = deduped;
}

fn wrap_pi(angle: f64) -> f64 {
    let wrapped = (angle + core::f64::consts::PI).rem_euclid(TWO_PI) - core::f64::consts::PI;
    if wrapped == -core::f64::consts::PI {
        core::f64::consts::PI
    } else {
        wrapped
    }
}

fn add2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale2(a: [f64; 2], scale: f64) -> [f64; 2] {
    [a[0] * scale, a[1] * scale]
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn norm2(a: [f64; 2]) -> f64 {
    dot2(a, a).sqrt()
}
