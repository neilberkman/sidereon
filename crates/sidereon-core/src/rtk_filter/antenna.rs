//! Receiver-antenna calibration and local-frame geometry for the RTK baseline
//! filter: NEU basis construction, line-of-sight zenith/azimuth, and PCO/PCV
//! projection used by the double-difference row builders.

use crate::astro::angles::rad_to_deg_ref;
use crate::astro::math::vec3::{add3, dot3, norm3, scale3, sub3, unit3};

use crate::antenna::{
    azimuth_bracket, blend_azimuth, interpolate_zenith_sorted, normalize_azimuth,
};

/// Selected receiver-antenna calibration for one frequency.
///
/// `noazi_pcv_m` entries are `(zenith_deg, value_m)`. `azi_pcv_m` entries are
/// `(azimuth_deg, zenith_deg, value_m)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiverAntennaCalibration {
    pub pco_neu_m: [f64; 3],
    pub noazi_pcv_m: Vec<(f64, f64)>,
    pub azi_pcv_m: Vec<(f64, f64, f64)>,
}

/// Base and rover receiver-antenna calibrations used by the DD row builder.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiverAntennaCorrections {
    pub base: ReceiverAntennaCalibration,
    pub rover: ReceiverAntennaCalibration,
}

/// Why a provided RTK receiver-antenna calibration could not be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverAntennaError {
    /// The calibration did not provide a PCV value for the current line of sight.
    MissingPcv,
    /// Satellite/receiver geometry could not form a valid line-of-sight vector.
    InvalidGeometry,
}

impl core::fmt::Display for ReceiverAntennaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingPcv => {
                write!(f, "RTK receiver antenna PCV calibration is missing")
            }
            Self::InvalidGeometry => {
                write!(f, "RTK receiver antenna geometry is invalid")
            }
        }
    }
}

impl std::error::Error for ReceiverAntennaError {}

fn los_projection(
    pco: [f64; 3],
    north_unit: [f64; 3],
    east_unit: [f64; 3],
    up_unit: [f64; 3],
    los: [f64; 3],
) -> f64 {
    let pco_ecef = add3(scale3(north_unit, pco[0]), scale3(east_unit, pco[1]));
    let pco_ecef = add3(pco_ecef, scale3(up_unit, pco[2]));
    dot3(pco_ecef, los)
}

fn los_zenith_azimuth_deg(
    los: [f64; 3],
    up: [f64; 3],
    north: [f64; 3],
    east: [f64; 3],
) -> (f64, f64) {
    let elevation_sin = dot3(los, up);
    let elevation_sin = (-1.0_f64).max(1.0_f64.min(elevation_sin));
    let zenith_deg = rad_to_deg_ref(elevation_sin.acos());

    let azimuth_rad = dot3(los, east).atan2(dot3(los, north));
    let azimuth_deg = rad_to_deg_ref(azimuth_rad);
    let azimuth_deg = if azimuth_deg < 0.0 {
        azimuth_deg + 360.0
    } else {
        azimuth_deg
    };

    (zenith_deg, azimuth_deg)
}

fn sorted_zenith_samples(samples: &[(f64, f64)], out: &mut Vec<(f64, f64)>) {
    out.clear();
    out.extend_from_slice(samples);
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
}

fn interpolate_samples(
    samples: &[(f64, f64)],
    zenith_deg: f64,
    scratch: &mut Vec<(f64, f64)>,
) -> Option<f64> {
    sorted_zenith_samples(samples, scratch);
    interpolate_zenith_sorted(scratch, zenith_deg)
}

#[derive(Debug, Default)]
pub(super) struct ReceiverAntennaScratch {
    zenith_samples: Vec<(f64, f64)>,
    azimuths: Vec<f64>,
    low_samples: Vec<(f64, f64)>,
    high_samples: Vec<(f64, f64)>,
}

fn interpolate_azimuth_pcv(
    samples: &[(f64, f64, f64)],
    azimuth_deg: f64,
    zenith_deg: f64,
    scratch: &mut ReceiverAntennaScratch,
) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }

    scratch.azimuths.clear();
    for &(az, _, _) in samples {
        if !scratch.azimuths.contains(&az) {
            scratch.azimuths.push(az);
        }
    }
    scratch
        .azimuths
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let azimuth = normalize_azimuth(azimuth_deg);
    let (low_deg, high_deg) = azimuth_bracket(&scratch.azimuths, azimuth);

    scratch.low_samples.clear();
    scratch.high_samples.clear();
    for &(az, zen, value) in samples {
        if az == low_deg {
            scratch.low_samples.push((zen, value));
        }
        if az == high_deg {
            scratch.high_samples.push((zen, value));
        }
    }

    let low_value = interpolate_samples(
        &scratch.low_samples,
        zenith_deg,
        &mut scratch.zenith_samples,
    )?;
    let high_value = interpolate_samples(
        &scratch.high_samples,
        zenith_deg,
        &mut scratch.zenith_samples,
    )?;

    Some(blend_azimuth(
        low_deg, high_deg, azimuth, low_value, high_value,
    ))
}

pub(super) fn pcv_m(
    calibration: &ReceiverAntennaCalibration,
    zenith_deg: f64,
    azimuth_deg: f64,
    scratch: &mut ReceiverAntennaScratch,
) -> Result<f64, ReceiverAntennaError> {
    if calibration.azi_pcv_m.is_empty() {
        interpolate_samples(
            &calibration.noazi_pcv_m,
            zenith_deg,
            &mut scratch.zenith_samples,
        )
        .ok_or(ReceiverAntennaError::MissingPcv)
    } else {
        interpolate_azimuth_pcv(&calibration.azi_pcv_m, azimuth_deg, zenith_deg, scratch)
            .ok_or(ReceiverAntennaError::MissingPcv)
    }
}

pub(super) fn receiver_antenna_correction(
    sat_pos: [f64; 3],
    receiver_pos: [f64; 3],
    calibration: &ReceiverAntennaCalibration,
    scratch: &mut ReceiverAntennaScratch,
) -> Result<f64, ReceiverAntennaError> {
    let receiver_norm = norm3(receiver_pos);
    if !receiver_norm.is_finite() || receiver_norm <= 0.0 {
        return Err(ReceiverAntennaError::InvalidGeometry);
    }
    let Some(los) = unit3(sub3(sat_pos, receiver_pos)) else {
        return Err(ReceiverAntennaError::InvalidGeometry);
    };
    let (north, east, up) = crate::estimation::substrate::frames::local_neu_basis(
        crate::estimation::recipe::FrameRecipe::GeocentricUpRtkReference,
        receiver_pos,
    );
    let pco_projection = los_projection(calibration.pco_neu_m, north, east, up, los);
    let (zenith_deg, azimuth_deg) = los_zenith_azimuth_deg(los, up, north, east);
    let pcv = pcv_m(calibration, zenith_deg, azimuth_deg, scratch)?;
    Ok(pco_projection + pcv)
}

/// The four positions a double-difference receiver-antenna correction needs: the
/// target and reference satellites (`sat_pos`/`ref_pos`) seen from the two
/// receiver stations (`base_pos`/`rover_pos`). Carried as one struct so the DD
/// row builders pass the geometry without a six-position argument list.
#[derive(Clone, Copy)]
pub(super) struct DoubleDifferenceAntennaGeometry {
    pub(super) sat_pos: [f64; 3],
    pub(super) ref_pos: [f64; 3],
    pub(super) base_pos: [f64; 3],
    pub(super) rover_pos: [f64; 3],
}

pub(super) fn double_difference_receiver_antenna_correction(
    geom: DoubleDifferenceAntennaGeometry,
    corrections: Option<&ReceiverAntennaCorrections>,
    scratch: &mut ReceiverAntennaScratch,
) -> Result<f64, ReceiverAntennaError> {
    let Some(corrections) = corrections else {
        return Ok(0.0);
    };

    let rover_sat_corr =
        receiver_antenna_correction(geom.sat_pos, geom.rover_pos, &corrections.rover, scratch)?;
    let base_sat_corr =
        receiver_antenna_correction(geom.sat_pos, geom.base_pos, &corrections.base, scratch)?;
    let rover_ref_corr =
        receiver_antenna_correction(geom.ref_pos, geom.rover_pos, &corrections.rover, scratch)?;
    let base_ref_corr =
        receiver_antenna_correction(geom.ref_pos, geom.base_pos, &corrections.base, scratch)?;

    Ok(rover_sat_corr - base_sat_corr - rover_ref_corr + base_ref_corr)
}
