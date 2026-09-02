//! Sample-backed IONEX vertical-TEC source.
//!
//! The canonical IONEX intermediate representation is a set of vertical-TEC
//! maps on a strictly increasing epoch axis, with descending latitude nodes,
//! ascending longitude nodes, signed grid steps, shell geometry, and optional
//! RMS maps. IONEX text is one serialization of that IR; [`super::Ionex`] is
//! the parser. This module builds the same evaluatable product directly from
//! samples, with no text in the loop, and drives the exact same slant-delay
//! evaluator the parsed path uses.
//!
//! # Byte-identical parity with the parser path
//!
//! [`Ionex::from_samples`] accepts the same field values the parser stores:
//! map epochs, TEC/RMS grids in TECU, node axes, signed steps, shell geometry,
//! and `EXPONENT`. [`Ionex::tec_grid_samples`] clones those fields out of a
//! parsed or sample-built product. Therefore
//! `Ionex::from_samples(ionex.tec_grid_samples())` rebuilds an equal product
//! byte-for-byte in every stored float and epoch. Unlike SP3 samples, there is
//! no SI reconstruction boundary here: VTEC is TECU on both sides. The only
//! lossy boundary remains serialize-through-text, where the existing writer
//! recovers scaled integer fields with `round(value / 10^EXPONENT)`.

use super::grid::{Ionex, IonexParts};
use super::j2000_seconds_from_instant;
use crate::astro::time::model::{Instant, InstantRepr};

const IONEX_AXIS_DEG_LIMIT: f64 = 360.0;
const NANOS_PER_SECOND: i128 = 1_000_000_000;

/// One vertical-TEC sample at one grid node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TecSample {
    /// Map epoch.
    pub epoch: Instant,
    /// Latitude node in degrees.
    pub lat_deg: f64,
    /// Longitude node in degrees.
    pub lon_deg: f64,
    /// Vertical TEC in TECU.
    pub vtec_tecu: f64,
    /// Optional RMS value in TECU.
    pub rms_tecu: Option<f64>,
}

/// Whole-grid IONEX vertical-TEC samples.
#[derive(Debug, Clone, PartialEq)]
pub struct TecGridSamples {
    /// Map epochs as instants, strictly increasing.
    pub map_epochs: Vec<Instant>,
    /// Latitude node values in degrees, descending.
    pub lat_nodes_deg: Vec<f64>,
    /// Longitude node values in degrees, ascending.
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
    /// Per-map vertical-TEC grids, indexed `[map][i_lat][i_lon]` (TECU).
    pub tec_maps: Vec<Vec<Vec<f64>>>,
    /// Per-map RMS grids, indexed `[map][i_lat][i_lon]` (TECU); empty if absent.
    pub rms_maps: Vec<Vec<Vec<f64>>>,
}

/// Validation failure building an IONEX sample source.
#[derive(Debug, Clone, PartialEq)]
pub enum TecSamplesError {
    /// No TEC samples were supplied.
    Empty,
    /// A latitude or longitude axis has fewer than two nodes.
    TooFewNodes(usize),
    /// Latitude nodes are not strictly descending.
    NonMonotonicLat,
    /// Longitude nodes are not strictly ascending.
    NonMonotonicLon,
    /// Map epochs are not strictly increasing.
    NonMonotonicEpochs,
    /// A map epoch cannot be expressed as an exact integer J2000 second.
    EpochNotRepresentable,
    /// Grid dimensions do not match the epoch or node axes.
    ShapeMismatch,
    /// RMS map count or node coverage does not match the TEC maps.
    RmsCountMismatch,
    /// A supplied float was NaN or infinite.
    NonFiniteValue,
    /// A signed grid step is zero or has the wrong sign for the node ordering.
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
            Self::NonMonotonicLat => {
                write!(f, "IONEX latitude nodes must be strictly descending")
            }
            Self::NonMonotonicLon => {
                write!(f, "IONEX longitude nodes must be strictly ascending")
            }
            Self::NonMonotonicEpochs => {
                write!(f, "IONEX map epochs must be strictly increasing")
            }
            Self::EpochNotRepresentable => {
                write!(f, "IONEX map epoch is not an exact integer J2000 second")
            }
            Self::ShapeMismatch => {
                write!(f, "IONEX TEC grid dimensions do not match the axes")
            }
            Self::RmsCountMismatch => write!(f, "IONEX RMS maps do not match TEC maps"),
            Self::NonFiniteValue => write!(f, "IONEX sample value is not finite"),
            Self::NonPositiveStep => write!(f, "IONEX grid step is not valid"),
            Self::AxisOutOfRange(value) => {
                write!(f, "IONEX axis value {value} is outside [-360, 360] degrees")
            }
        }
    }
}

impl std::error::Error for TecSamplesError {}

impl Ionex {
    /// Build an IONEX product directly from whole-grid samples.
    // invariant: epoch and axis membership are validated before these lookups.
    #[allow(clippy::expect_used)]
    pub fn from_samples(samples: TecGridSamples) -> core::result::Result<Self, TecSamplesError> {
        validate_grid_samples(&samples)?;
        Self::from_parts(IonexParts {
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
    // invariant: epoch and axis membership are validated before these lookups.
    #[allow(clippy::expect_used)]
    pub fn from_node_samples(
        samples: impl IntoIterator<Item = TecSample>,
        shell_height_km: f64,
        base_radius_km: f64,
        exponent: i32,
    ) -> core::result::Result<Self, TecSamplesError> {
        let samples: Vec<TecSample> = samples.into_iter().collect();
        if samples.is_empty() {
            return Err(TecSamplesError::Empty);
        }
        for sample in &samples {
            validate_axis_value(sample.lat_deg)?;
            validate_axis_value(sample.lon_deg)?;
            validate_finite(sample.vtec_tecu)?;
            if let Some(rms) = sample.rms_tecu {
                validate_finite(rms)?;
            }
            exact_j2000_second(sample.epoch).ok_or(TecSamplesError::EpochNotRepresentable)?;
        }

        let mut map_epochs = Vec::new();
        let mut lat_nodes_deg = Vec::new();
        let mut lon_nodes_deg = Vec::new();
        for sample in &samples {
            let epoch_s =
                exact_j2000_second(sample.epoch).ok_or(TecSamplesError::EpochNotRepresentable)?;
            if !map_epochs
                .iter()
                .any(|&epoch| exact_j2000_second(epoch) == Some(epoch_s))
            {
                map_epochs.push(sample.epoch);
            }
            push_unique_bits(&mut lat_nodes_deg, sample.lat_deg);
            push_unique_bits(&mut lon_nodes_deg, sample.lon_deg);
        }

        map_epochs.sort_by_key(|epoch| {
            exact_j2000_second(*epoch).expect("sample epochs were already validated")
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
        let has_rms = samples.iter().any(|sample| sample.rms_tecu.is_some());
        let mut tec_maps = vec![vec![vec![f64::NAN; nlon]; nlat]; nmap];
        let mut rms_maps = if has_rms {
            vec![vec![vec![f64::NAN; nlon]; nlat]; nmap]
        } else {
            Vec::new()
        };
        let mut seen = vec![false; nmap * nlat * nlon];

        for sample in samples {
            let map_index = map_epochs
                .iter()
                .position(|&epoch| exact_j2000_second(epoch) == exact_j2000_second(sample.epoch))
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
            if has_rms {
                let rms = sample.rms_tecu.ok_or(TecSamplesError::RmsCountMismatch)?;
                rms_maps[map_index][lat_index][lon_index] = rms;
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
        }
    }

    /// Extract this product as one sample per grid node.
    pub fn tec_samples(&self) -> Vec<TecSample> {
        let nmap = self.map_epochs().len();
        let nlat = self.lat_nodes_deg().len();
        let nlon = self.lon_nodes_deg().len();
        let mut out = Vec::with_capacity(nmap * nlat * nlon);
        let has_rms = !self.rms_maps().is_empty();
        for (map_index, &epoch) in self.map_epochs().iter().enumerate() {
            for (lat_index, &lat_deg) in self.lat_nodes_deg().iter().enumerate() {
                for (lon_index, &lon_deg) in self.lon_nodes_deg().iter().enumerate() {
                    out.push(TecSample {
                        epoch,
                        lat_deg,
                        lon_deg,
                        vtec_tecu: self.tec_maps()[map_index][lat_index][lon_index],
                        rms_tecu: if has_rms {
                            Some(self.rms_maps()[map_index][lat_index][lon_index])
                        } else {
                            None
                        },
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
    validate_axis(&samples.lat_nodes_deg, true)?;
    validate_axis(&samples.lon_nodes_deg, false)?;
    validate_finite(samples.dlat_deg)?;
    validate_finite(samples.dlon_deg)?;
    validate_axis_value(samples.dlat_deg)?;
    validate_axis_value(samples.dlon_deg)?;
    if samples.dlat_deg >= 0.0 || samples.dlon_deg <= 0.0 {
        return Err(TecSamplesError::NonPositiveStep);
    }
    validate_finite(samples.shell_height_km)?;
    validate_finite(samples.base_radius_km)?;
    validate_epochs(&samples.map_epochs)?;

    if samples.tec_maps.len() != samples.map_epochs.len() {
        return Err(TecSamplesError::ShapeMismatch);
    }
    validate_maps(
        &samples.tec_maps,
        samples.map_epochs.len(),
        samples.lat_nodes_deg.len(),
        samples.lon_nodes_deg.len(),
        TecSamplesError::ShapeMismatch,
    )?;
    if !samples.rms_maps.is_empty() {
        if samples.rms_maps.len() != samples.map_epochs.len() {
            return Err(TecSamplesError::RmsCountMismatch);
        }
        validate_maps(
            &samples.rms_maps,
            samples.map_epochs.len(),
            samples.lat_nodes_deg.len(),
            samples.lon_nodes_deg.len(),
            TecSamplesError::RmsCountMismatch,
        )?;
    }
    Ok(())
}

fn validate_axis(nodes: &[f64], descending: bool) -> core::result::Result<(), TecSamplesError> {
    if nodes.len() < 2 {
        return Err(TecSamplesError::TooFewNodes(nodes.len()));
    }
    for &node in nodes {
        validate_axis_value(node)?;
    }
    if descending {
        if nodes.windows(2).any(|w| w[1] >= w[0]) {
            return Err(TecSamplesError::NonMonotonicLat);
        }
    } else if nodes.windows(2).any(|w| w[1] <= w[0]) {
        return Err(TecSamplesError::NonMonotonicLon);
    }
    Ok(())
}

fn validate_epochs(map_epochs: &[Instant]) -> core::result::Result<(), TecSamplesError> {
    let mut previous_s = None;
    for &epoch in map_epochs {
        let seconds = exact_j2000_second(epoch).ok_or(TecSamplesError::EpochNotRepresentable)?;
        if previous_s.is_some_and(|previous| seconds <= previous) {
            return Err(TecSamplesError::NonMonotonicEpochs);
        }
        previous_s = Some(seconds);
    }
    Ok(())
}

fn validate_maps(
    maps: &[Vec<Vec<f64>>],
    expected_maps: usize,
    expected_lat: usize,
    expected_lon: usize,
    dimension_error: TecSamplesError,
) -> core::result::Result<(), TecSamplesError> {
    if maps.len() != expected_maps {
        return Err(dimension_error.clone());
    }
    for map in maps {
        if map.len() != expected_lat {
            return Err(dimension_error.clone());
        }
        for row in map {
            if row.len() != expected_lon {
                return Err(dimension_error.clone());
            }
            for &value in row {
                validate_finite(value)?;
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

fn exact_j2000_second(epoch: Instant) -> Option<i64> {
    match epoch.repr {
        InstantRepr::Nanos(nanos) => {
            if nanos % NANOS_PER_SECOND != 0 {
                return None;
            }
            let seconds = nanos / NANOS_PER_SECOND;
            i64::try_from(seconds).ok()
        }
        InstantRepr::JulianDate(_) => j2000_seconds_from_instant(epoch),
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
