//! Sample-backed IONEX vertical-TEC source.
//!
//! The canonical IONEX intermediate representation is a set of vertical-TEC
//! maps on a strictly increasing epoch axis, with latitude and longitude nodes
//! in the order their signed steps give, shell geometry, optional RMS
//! and height maps, and the descriptive header records. A node whose value is
//! not available is `None`. IONEX text is one serialization of that IR;
//! [`super::Ionex`] is the parser. This module builds the same evaluatable
//! product directly from samples, with no text in the loop, and drives the
//! exact same slant-delay evaluator the parsed path uses.
//!
//! # Byte-identical parity with the parser path
//!
//! [`Ionex::from_samples`] accepts the same field values the parser stores:
//! map epochs, TEC/RMS grids in TECU, height grids in kilometers, node axes,
//! signed steps, shell geometry, `EXPONENT`, and the header records.
//! [`Ionex::tec_grid_samples`] clones those fields out of a parsed or
//! sample-built product. Therefore `Ionex::from_samples(ionex.tec_grid_samples())`
//! rebuilds an equal product byte-for-byte in every stored float and epoch,
//! except [`Ionex::skipped_records`], which a sample-built product has none of.
//! Unlike SP3 samples, there is no SI reconstruction boundary here: VTEC is TECU
//! on both sides. Serializing through text is exact too: the writer refuses a
//! value it cannot write exactly rather than rounding it.

use super::grid::{Grid, Ionex, IonexParts};
use super::header::IonexHeader;
use super::{utc_j2000_second, utc_map_epoch, IonexEpochError};
use crate::astro::time::model::Instant;

const IONEX_AXIS_DEG_LIMIT: f64 = 360.0;

/// One vertical-TEC sample at one grid node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TecSample {
    /// Map epoch, in any time scale that names an exact whole UTC second (see
    /// [`IonexEpochError`]); samples naming the same UTC second are one map,
    /// held as that UTC instant.
    pub epoch: Instant,
    /// Latitude node in degrees.
    pub lat_deg: f64,
    /// Longitude node in degrees.
    pub lon_deg: f64,
    /// Vertical TEC in TECU; `None` where the value is not available.
    pub vtec_tecu: Option<f64>,
    /// RMS value in TECU; `None` where the node has no RMS value.
    pub rms_tecu: Option<f64>,
    /// The value this node's IONEX height map holds, in kilometers, which is
    /// added to `HGT1` to give the single-layer height there; `None` where the
    /// node has no height value. It is an offset from `HGT1`, not a shell
    /// height: IONEX 1's example 1 gives every height as `0` with `HGT1` at
    /// 400 km.
    pub height_offset_km: Option<f64>,
}

/// Whole-grid IONEX vertical-TEC samples.
#[derive(Debug, Clone, PartialEq)]
pub struct TecGridSamples {
    /// Map epochs as instants, strictly increasing, each naming an exact whole
    /// UTC second. IONEX epochs are UT, so an epoch in another time scale is
    /// carried onto UTC exactly (GPST 2017-01-01 00:00:18 is map epoch
    /// 2017-01-01 00:00:00 UTC) and the product holds the UTC instant; an
    /// epoch with no exact whole UTC second is refused with the cause.
    pub map_epochs: Vec<Instant>,
    /// Latitude node values in degrees, monotonic in the direction `dlat_deg`
    /// gives.
    pub lat_nodes_deg: Vec<f64>,
    /// Longitude node values in degrees, monotonic in the direction `dlon_deg`
    /// gives.
    pub lon_nodes_deg: Vec<f64>,
    /// Signed latitude step in degrees.
    pub dlat_deg: f64,
    /// Signed longitude step in degrees.
    pub dlon_deg: f64,
    /// Single-layer shell height in kilometers.
    pub shell_height_km: f64,
    /// Mean earth radius used by the geometry, in kilometers.
    pub base_radius_km: f64,
    /// The IONEX `EXPONENT` header field.
    pub exponent: i32,
    /// Per-map vertical-TEC grids, indexed `[map][i_lat][i_lon]` (TECU); `None`
    /// where the value is not available.
    pub tec_maps: Vec<Vec<Vec<Option<f64>>>>,
    /// Per-map RMS grids, indexed `[map][i_lat][i_lon]` (TECU); empty where the
    /// product declares no RMS map, `None` where a node has no RMS value. This
    /// field is the authority on whether the maps exist: a stack whose every
    /// node is `None` is retained, and says the maps are declared with no value
    /// anywhere.
    pub rms_maps: Vec<Vec<Vec<Option<f64>>>>,
    /// Per-map height grids, indexed `[map][i_lat][i_lon]` (km); empty if
    /// absent, `None` where a node has no height value.
    pub height_maps: Vec<Vec<Vec<Option<f64>>>>,
    /// Descriptive header records.
    pub header: IonexHeader,
}

/// Validation failure building an IONEX sample source.
#[derive(Debug, Clone, PartialEq)]
pub enum TecSamplesError {
    /// No TEC samples were supplied.
    Empty,
    /// A latitude or longitude axis has fewer than two nodes.
    TooFewNodes(usize),
    /// Latitude nodes are not strictly monotonic in the direction `dlat_deg`
    /// gives.
    NonMonotonicLat,
    /// Longitude nodes are not strictly monotonic in the direction `dlon_deg`
    /// gives.
    NonMonotonicLon,
    /// Map epochs are not strictly increasing.
    NonMonotonicEpochs,
    /// A map epoch names no exact whole UTC second.
    ///
    /// The IONEX epoch axis is an axis of whole UT seconds, so an epoch is
    /// taken only where it states one exactly: an integer-nanosecond instant
    /// whose UTC reading is a whole number of seconds within `i64`, or a split
    /// Julian date whose two parts either sum to a whole second of its scale
    /// or encode one as a day boundary that names a second and an integer
    /// residual within the day, carried onto UTC exactly. A fractional epoch is
    /// refused here rather than accepted as the nearest whole second; the
    /// cause says why.
    EpochNotRepresentable(IonexEpochError),
    /// Grid dimensions do not match the epoch or node axes.
    ShapeMismatch,
    /// RMS map count or node coverage does not match the TEC maps.
    RmsCountMismatch,
    /// Height map count or node coverage does not match the TEC maps.
    HeightCountMismatch,
    /// A supplied float was NaN or infinite.
    NonFiniteValue,
    /// A signed grid step is zero, so it names no direction for its axis.
    NonPositiveStep,
    /// An axis coordinate or step falls outside `[-360, 360]` degrees.
    AxisOutOfRange(f64),
}

impl core::fmt::Display for TecSamplesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "no IONEX TEC samples supplied"),
            Self::TooFewNodes(count) => {
                write!(f, "IONEX grid axis has {count} nodes; need at least two")
            }
            Self::NonMonotonicLat => write!(
                f,
                "IONEX latitude nodes must be strictly monotonic in the direction of DLAT"
            ),
            Self::NonMonotonicLon => write!(
                f,
                "IONEX longitude nodes must be strictly monotonic in the direction of DLON"
            ),
            Self::NonMonotonicEpochs => {
                write!(f, "IONEX map epochs must be strictly increasing")
            }
            Self::EpochNotRepresentable(cause) => write!(f, "{cause}"),
            Self::ShapeMismatch => {
                write!(f, "IONEX TEC grid dimensions do not match the axes")
            }
            Self::RmsCountMismatch => write!(f, "IONEX RMS maps do not match TEC maps"),
            Self::HeightCountMismatch => write!(f, "IONEX height maps do not match TEC maps"),
            Self::NonFiniteValue => write!(f, "IONEX sample value is not finite"),
            Self::NonPositiveStep => write!(f, "IONEX grid step is zero"),
            Self::AxisOutOfRange(value) => {
                write!(f, "IONEX axis value {value} is outside [-360, 360] degrees")
            }
        }
    }
}

impl std::error::Error for TecSamplesError {}

impl Ionex {
    /// Build an IONEX product directly from whole-grid samples.
    ///
    /// The RMS and height map stacks are stored as given, as they are for a
    /// parsed product: a stack with no value at any node stays present and is
    /// distinct from an empty one, which says the product declares no map of
    /// that kind.
    ///
    /// Map epochs are held as UTC, the time system of an IONEX epoch record.
    /// A UTC epoch is stored as given; an epoch in another time scale is
    /// stored as the UTC instant of the same whole second, carried over
    /// exactly, and one with no exact whole UTC second is refused with
    /// [`TecSamplesError::EpochNotRepresentable`] and its cause.
    pub fn from_samples(
        mut samples: TecGridSamples,
    ) -> core::result::Result<Self, TecSamplesError> {
        validate_grid_samples(&samples)?;
        for epoch in &mut samples.map_epochs {
            *epoch = utc_map_epoch(*epoch).map_err(TecSamplesError::EpochNotRepresentable)?;
        }
        Self::from_parts(IonexParts {
            header: samples.header,
            lat_nodes_deg: samples.lat_nodes_deg,
            lon_nodes_deg: samples.lon_nodes_deg,
            dlat_deg: samples.dlat_deg,
            dlon_deg: samples.dlon_deg,
            shell_height_km: samples.shell_height_km,
            base_radius_km: samples.base_radius_km,
            exponent: samples.exponent,
            map_epochs: samples.map_epochs,
            tec_maps: samples.tec_maps,
            rms_maps: samples.rms_maps,
            height_maps: samples.height_maps,
            skipped_records: 0,
        })
        .map_err(|_| {
            // Public sample validation mirrors Ionex::from_parts. This fallback is
            // only for future private invariants that TecSamplesError cannot yet
            // classify more precisely.
            TecSamplesError::ShapeMismatch
        })
    }

    /// Build an IONEX product from a flat stream of node samples.
    ///
    /// Every node of the grid the samples span must appear exactly once. A
    /// product has RMS maps when any sample has an RMS value, and height maps
    /// when any has a height value; a node without one is `None` there.
    ///
    /// Flat node samples carry no map-presence field, so this input model
    /// cannot state RMS (or height) maps that are declared yet hold no value at
    /// any node: samples whose `rms_tecu` is `None` everywhere give a product
    /// with no RMS map. Build such a product through [`TecGridSamples`], whose
    /// `rms_maps` stack states the maps explicitly and is retained.
    // invariant: epoch and axis membership are validated before these lookups.
    #[allow(clippy::expect_used)]
    pub fn from_node_samples(
        samples: impl IntoIterator<Item = TecSample>,
        shell_height_km: f64,
        base_radius_km: f64,
        exponent: i32,
        header: IonexHeader,
    ) -> core::result::Result<Self, TecSamplesError> {
        let samples: Vec<TecSample> = samples.into_iter().collect();
        if samples.is_empty() {
            return Err(TecSamplesError::Empty);
        }
        for sample in &samples {
            validate_axis_value(sample.lat_deg)?;
            validate_axis_value(sample.lon_deg)?;
            for value in [sample.vtec_tecu, sample.rms_tecu, sample.height_offset_km]
                .into_iter()
                .flatten()
            {
                validate_finite(value)?;
            }
            utc_j2000_second(sample.epoch).map_err(TecSamplesError::EpochNotRepresentable)?;
        }

        let mut map_epochs = Vec::new();
        let mut lat_nodes_deg = Vec::new();
        let mut lon_nodes_deg = Vec::new();
        for sample in &samples {
            let epoch_s =
                utc_j2000_second(sample.epoch).map_err(TecSamplesError::EpochNotRepresentable)?;
            if !map_epochs
                .iter()
                .any(|&epoch| utc_j2000_second(epoch) == Ok(epoch_s))
            {
                map_epochs.push(sample.epoch);
            }
            push_unique_bits(&mut lat_nodes_deg, sample.lat_deg);
            push_unique_bits(&mut lon_nodes_deg, sample.lon_deg);
        }

        map_epochs.sort_by_key(|epoch| {
            utc_j2000_second(*epoch).expect("sample epochs were already validated")
        });
        lat_nodes_deg.sort_by(|a, b| b.total_cmp(a));
        lon_nodes_deg.sort_by(f64::total_cmp);

        if lat_nodes_deg.len() < 2 {
            return Err(TecSamplesError::TooFewNodes(lat_nodes_deg.len()));
        }
        if lon_nodes_deg.len() < 2 {
            return Err(TecSamplesError::TooFewNodes(lon_nodes_deg.len()));
        }

        let nmap = map_epochs.len();
        let nlat = lat_nodes_deg.len();
        let nlon = lon_nodes_deg.len();
        let empty_maps = |present: bool| -> Vec<Grid> {
            if present {
                vec![vec![vec![None; nlon]; nlat]; nmap]
            } else {
                Vec::new()
            }
        };
        let mut tec_maps = empty_maps(true);
        let mut rms_maps = empty_maps(samples.iter().any(|sample| sample.rms_tecu.is_some()));
        let mut height_maps = empty_maps(
            samples
                .iter()
                .any(|sample| sample.height_offset_km.is_some()),
        );
        let mut seen = vec![false; nmap * nlat * nlon];

        for sample in samples {
            let map_index = map_epochs
                .iter()
                .position(|&epoch| utc_j2000_second(epoch) == utc_j2000_second(sample.epoch))
                .expect("sample epoch exists in the map axis");
            let lat_index = find_bits(&lat_nodes_deg, sample.lat_deg)
                .expect("sample latitude exists in the latitude axis");
            let lon_index = find_bits(&lon_nodes_deg, sample.lon_deg)
                .expect("sample longitude exists in the longitude axis");
            let flat_index = (map_index * nlat + lat_index) * nlon + lon_index;
            if seen[flat_index] {
                return Err(TecSamplesError::ShapeMismatch);
            }
            seen[flat_index] = true;
            tec_maps[map_index][lat_index][lon_index] = sample.vtec_tecu;
            if let Some(map) = rms_maps.get_mut(map_index) {
                map[lat_index][lon_index] = sample.rms_tecu;
            }
            if let Some(map) = height_maps.get_mut(map_index) {
                map[lat_index][lon_index] = sample.height_offset_km;
            }
        }
        if seen.iter().any(|&value| !value) {
            return Err(TecSamplesError::ShapeMismatch);
        }

        Self::from_samples(TecGridSamples {
            map_epochs,
            dlat_deg: lat_nodes_deg[1] - lat_nodes_deg[0],
            dlon_deg: lon_nodes_deg[1] - lon_nodes_deg[0],
            lat_nodes_deg,
            lon_nodes_deg,
            shell_height_km,
            base_radius_km,
            exponent,
            tec_maps,
            rms_maps,
            height_maps,
            header,
        })
    }

    /// Extract this product as whole-grid IONEX samples.
    pub fn tec_grid_samples(&self) -> TecGridSamples {
        TecGridSamples {
            map_epochs: self.map_epochs().to_vec(),
            lat_nodes_deg: self.lat_nodes_deg().to_vec(),
            lon_nodes_deg: self.lon_nodes_deg().to_vec(),
            dlat_deg: self.dlat_deg(),
            dlon_deg: self.dlon_deg(),
            shell_height_km: self.shell_height_km(),
            base_radius_km: self.base_radius_km(),
            exponent: self.exponent(),
            tec_maps: self.tec_maps().to_vec(),
            rms_maps: self.rms_maps().to_vec(),
            height_maps: self.height_maps().to_vec(),
            header: self.header().clone(),
        }
    }

    /// Extract this product as one sample per grid node.
    pub fn tec_samples(&self) -> Vec<TecSample> {
        let nmap = self.map_epochs().len();
        let nlat = self.lat_nodes_deg().len();
        let nlon = self.lon_nodes_deg().len();
        let mut out = Vec::with_capacity(nmap * nlat * nlon);
        for (map_index, &epoch) in self.map_epochs().iter().enumerate() {
            for (lat_index, &lat_deg) in self.lat_nodes_deg().iter().enumerate() {
                for (lon_index, &lon_deg) in self.lon_nodes_deg().iter().enumerate() {
                    let node = |maps: &[Grid]| {
                        maps.get(map_index)
                            .and_then(|map| map[lat_index][lon_index])
                    };
                    out.push(TecSample {
                        epoch,
                        lat_deg,
                        lon_deg,
                        vtec_tecu: node(self.tec_maps()),
                        rms_tecu: node(self.rms_maps()),
                        height_offset_km: node(self.height_maps()),
                    });
                }
            }
        }
        out
    }
}

fn validate_grid_samples(samples: &TecGridSamples) -> core::result::Result<(), TecSamplesError> {
    if samples.map_epochs.is_empty() || samples.tec_maps.is_empty() {
        return Err(TecSamplesError::Empty);
    }
    validate_axis(
        &samples.lat_nodes_deg,
        samples.dlat_deg,
        TecSamplesError::NonMonotonicLat,
    )?;
    validate_axis(
        &samples.lon_nodes_deg,
        samples.dlon_deg,
        TecSamplesError::NonMonotonicLon,
    )?;
    validate_finite(samples.dlat_deg)?;
    validate_finite(samples.dlon_deg)?;
    validate_axis_value(samples.dlat_deg)?;
    validate_axis_value(samples.dlon_deg)?;
    // A step carries the direction its axis runs in, either way; only a zero
    // step names no direction. A step whose sign contradicts its nodes is
    // refused as a node-order failure above.
    if samples.dlat_deg == 0.0 || samples.dlon_deg == 0.0 {
        return Err(TecSamplesError::NonPositiveStep);
    }
    validate_finite(samples.shell_height_km)?;
    validate_finite(samples.base_radius_km)?;
    validate_finite(samples.header.version)?;
    validate_finite(samples.header.elevation_cutoff_deg)?;
    validate_epochs(&samples.map_epochs)?;

    if samples.tec_maps.len() != samples.map_epochs.len() {
        return Err(TecSamplesError::ShapeMismatch);
    }
    let dimensions = (
        samples.map_epochs.len(),
        samples.lat_nodes_deg.len(),
        samples.lon_nodes_deg.len(),
    );
    validate_maps(
        &samples.tec_maps,
        dimensions,
        TecSamplesError::ShapeMismatch,
    )?;
    for (maps, error) in [
        (&samples.rms_maps, TecSamplesError::RmsCountMismatch),
        (&samples.height_maps, TecSamplesError::HeightCountMismatch),
    ] {
        if !maps.is_empty() {
            validate_maps(maps, dimensions, error)?;
        }
    }
    Ok(())
}

fn validate_axis(
    nodes: &[f64],
    step: f64,
    error: TecSamplesError,
) -> core::result::Result<(), TecSamplesError> {
    if nodes.len() < 2 {
        return Err(TecSamplesError::TooFewNodes(nodes.len()));
    }
    for &node in nodes {
        validate_axis_value(node)?;
    }
    let ordered = if step > 0.0 {
        nodes.windows(2).all(|w| w[1] > w[0])
    } else if step < 0.0 {
        nodes.windows(2).all(|w| w[1] < w[0])
    } else {
        // A zero or non-finite step names no direction to check the nodes
        // against; the step checks refuse it by name.
        true
    };
    if !ordered {
        return Err(error);
    }
    Ok(())
}

fn validate_epochs(map_epochs: &[Instant]) -> core::result::Result<(), TecSamplesError> {
    let mut previous_s = None;
    for &epoch in map_epochs {
        let seconds = utc_j2000_second(epoch).map_err(TecSamplesError::EpochNotRepresentable)?;
        if previous_s.is_some_and(|previous| seconds <= previous) {
            return Err(TecSamplesError::NonMonotonicEpochs);
        }
        previous_s = Some(seconds);
    }
    Ok(())
}

fn validate_maps(
    maps: &[Grid],
    (expected_maps, expected_lat, expected_lon): (usize, usize, usize),
    dimension_error: TecSamplesError,
) -> core::result::Result<(), TecSamplesError> {
    if maps.len() != expected_maps {
        return Err(dimension_error);
    }
    for map in maps {
        if map.len() != expected_lat {
            return Err(dimension_error);
        }
        for row in map {
            if row.len() != expected_lon {
                return Err(dimension_error);
            }
            for value in row.iter().flatten() {
                validate_finite(*value)?;
            }
        }
    }
    Ok(())
}

fn validate_axis_value(value: f64) -> core::result::Result<(), TecSamplesError> {
    validate_finite(value)?;
    if !(-IONEX_AXIS_DEG_LIMIT..=IONEX_AXIS_DEG_LIMIT).contains(&value) {
        return Err(TecSamplesError::AxisOutOfRange(value));
    }
    Ok(())
}

fn validate_finite(value: f64) -> core::result::Result<(), TecSamplesError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TecSamplesError::NonFiniteValue)
    }
}

fn push_unique_bits(values: &mut Vec<f64>, value: f64) {
    if !values
        .iter()
        .any(|&existing| same_axis_node(existing, value))
    {
        values.push(value);
    }
}

fn find_bits(values: &[f64], value: f64) -> Option<usize> {
    values
        .iter()
        .position(|&existing| same_axis_node(existing, value))
}

fn same_axis_node(a: f64, b: f64) -> bool {
    a == b || a.to_bits() == b.to_bits()
}
