use std::cell::RefCell;

use crate::astro::almanac::{
    angular_separation_rad, apparent_km, body_ecliptic, event_finder, instant_at_offset_seconds,
    latch_scalar, latched_or_finder, norm_checked, offset_instant, validate_scan_controls,
    AlmanacError, EclipseEvent, EclipseKind, EphemerisSource, MoonPhaseKind, NAIF_MOON, NAIF_SUN,
    PHASE_STEP_MAX_SECONDS,
};
use crate::astro::constants::{
    astro::{MOON_RADIUS_KM, SOLAR_RADIUS_KM},
    earth::MEAN_EARTH_RADIUS_KM,
    time::SECONDS_PER_HOUR,
};
use crate::astro::events::ExtremumKind;
use crate::astro::math::vec3::{dot3, norm3, scale3, sub3, unit3};
use crate::astro::passes::UtcInstant;

const ECLIPSE_PAD_SECONDS: f64 = 6.0 * SECONDS_PER_HOUR;
const ECLIPSE_EXTREMUM_STEP_SECONDS: f64 = 120.0;
const SOLAR_NODE_LIMIT_DEG: f64 = 1.58;
const LUNAR_NODE_LIMIT_DEG: f64 = 1.05;
const LUNAR_SHADOW_ENLARGEMENT: f64 = 1.02;
const GRAZING_MAGNITUDE_BAND: f64 = 0.01;
const GRAZING_GAMMA_BAND: f64 = 0.01;

/// Find lunar and solar eclipses whose maximum lies in the UTC window.
pub fn lunar_solar_eclipses(
    source: EphemerisSource<'_>,
    start: UtcInstant,
    end: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<EclipseEvent>, AlmanacError> {
    validate_scan_controls(step_seconds, time_tolerance_seconds, PHASE_STEP_MAX_SECONDS)?;

    let padded_start = offset_instant(start, -ECLIPSE_PAD_SECONDS);
    let padded_end = offset_instant(end, ECLIPSE_PAD_SECONDS);
    let phases = crate::astro::almanac::moon_phases(
        source,
        padded_start,
        padded_end,
        step_seconds,
        time_tolerance_seconds,
    )?;

    let mut events = Vec::new();
    for phase in phases {
        match phase.kind {
            MoonPhaseKind::New => {
                let beta = body_ecliptic(source, NAIF_MOON, phase.time)?.latitude_deg;
                if beta.abs() >= SOLAR_NODE_LIMIT_DEG {
                    continue;
                }
                if let Some(time_maximum) = find_minimum_time(
                    phase.time,
                    (padded_start, padded_end),
                    time_tolerance_seconds,
                    |time| Ok(solar_geometry(source, time)?.gamma),
                )? {
                    if time_maximum < start || time_maximum > end {
                        continue;
                    }
                    if let Some(event) = classify_solar(source, time_maximum)? {
                        events.push(event);
                    }
                }
            }
            MoonPhaseKind::Full => {
                let beta = body_ecliptic(source, NAIF_MOON, phase.time)?.latitude_deg;
                if beta.abs() >= LUNAR_NODE_LIMIT_DEG {
                    continue;
                }
                if let Some(time_maximum) = find_minimum_time(
                    phase.time,
                    (padded_start, padded_end),
                    time_tolerance_seconds,
                    |time| lunar_sigma(source, time),
                )? {
                    if time_maximum < start || time_maximum > end {
                        continue;
                    }
                    if let Some(event) = classify_lunar(source, time_maximum)? {
                        events.push(event);
                    }
                }
            }
            MoonPhaseKind::FirstQuarter | MoonPhaseKind::LastQuarter => {}
        }
    }
    events.sort_by_key(|event| event.time_maximum);
    Ok(events)
}

fn find_minimum_time<F>(
    center: UtcInstant,
    bounds: (UtcInstant, UtcInstant),
    time_tolerance_seconds: f64,
    scalar_fn: F,
) -> Result<Option<UtcInstant>, AlmanacError>
where
    F: Fn(UtcInstant) -> Result<f64, AlmanacError>,
{
    // Clamp the extremum sub-window to the range the moon-phase scan already
    // proved in-coverage. A phase within ECLIPSE_PAD_SECONDS of an SPK segment
    // edge would otherwise push a sample past the kernel coverage and raise a
    // spurious error for an eclipse whose maximum is itself in coverage.
    let window_start = offset_instant(center, -ECLIPSE_PAD_SECONDS);
    let window_end = offset_instant(center, ECLIPSE_PAD_SECONDS);
    let start = if window_start > bounds.0 {
        window_start
    } else {
        bounds.0
    };
    let end = if window_end < bounds.1 {
        window_end
    } else {
        bounds.1
    };
    let finder = event_finder(
        start,
        end,
        ECLIPSE_EXTREMUM_STEP_SECONDS,
        time_tolerance_seconds,
    )?;
    let latch = RefCell::new(None);
    let extrema = finder
        .find_extrema(|offset_seconds| {
            latch_scalar(&latch, || {
                let time = instant_at_offset_seconds(start, offset_seconds);
                scalar_fn(time)
            })
        })
        .map_err(|error| latched_or_finder(error, &latch))?;

    let minimum = extrema
        .into_iter()
        .filter(|event| event.kind == ExtremumKind::Minimum)
        .min_by(|a, b| a.value.total_cmp(&b.value));
    Ok(minimum.map(|event| instant_at_offset_seconds(start, event.time_seconds)))
}

fn lunar_sigma(source: EphemerisSource<'_>, time: UtcInstant) -> Result<f64, AlmanacError> {
    let moon = apparent_km(source, NAIF_MOON, time)?;
    let sun = apparent_km(source, NAIF_SUN, time)?;
    angular_separation_rad(moon, scale3(sun, -1.0))
}

fn classify_lunar(
    source: EphemerisSource<'_>,
    time_maximum: UtcInstant,
) -> Result<Option<EclipseEvent>, AlmanacError> {
    let moon = apparent_km(source, NAIF_MOON, time_maximum)?;
    let sun = apparent_km(source, NAIF_SUN, time_maximum)?;
    let moon_distance = norm_checked(moon, "moon")?;
    let sun_distance = norm_checked(sun, "sun")?;
    let sigma = angular_separation_rad(moon, scale3(sun, -1.0))?;

    let pi_m = libm::asin(MEAN_EARTH_RADIUS_KM / moon_distance);
    let pi_s = libm::asin(MEAN_EARTH_RADIUS_KM / sun_distance);
    let s_sun = libm::asin(SOLAR_RADIUS_KM / sun_distance);
    let s_moon = libm::asin(MOON_RADIUS_KM / moon_distance);
    let rho_u = LUNAR_SHADOW_ENLARGEMENT * (pi_m + pi_s - s_sun);
    let rho_p = LUNAR_SHADOW_ENLARGEMENT * (pi_m + pi_s + s_sun);

    let (kind, magnitude) = if sigma + s_moon <= rho_u {
        (
            EclipseKind::LunarTotal,
            (rho_u + s_moon - sigma) / (2.0 * s_moon),
        )
    } else if sigma - s_moon < rho_u {
        (
            EclipseKind::LunarPartial,
            (rho_u + s_moon - sigma) / (2.0 * s_moon),
        )
    } else if sigma - s_moon < rho_p {
        (
            EclipseKind::LunarPenumbral,
            (rho_p + s_moon - sigma) / (2.0 * s_moon),
        )
    } else {
        return Ok(None);
    };

    let moon_latitude_deg = body_ecliptic(source, NAIF_MOON, time_maximum)?.latitude_deg;
    let boundary = [
        (sigma + s_moon - rho_u).abs(),
        (sigma - s_moon - rho_u).abs(),
        (sigma - s_moon - rho_p).abs(),
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
        / (2.0 * s_moon);
    Ok(Some(EclipseEvent {
        time_maximum,
        kind,
        magnitude,
        moon_latitude_deg,
        gamma: 0.0,
        uncertain: boundary < GRAZING_MAGNITUDE_BAND,
    }))
}

#[derive(Debug, Clone, Copy)]
struct SolarGeometry {
    moon: [f64; 3],
    sun: [f64; 3],
    axis: [f64; 3],
    p: [f64; 3],
    gamma: f64,
    l1: f64,
    disk_reaches_earth: bool,
}

fn solar_geometry(
    source: EphemerisSource<'_>,
    time: UtcInstant,
) -> Result<SolarGeometry, AlmanacError> {
    let moon = apparent_km(source, NAIF_MOON, time)?;
    let sun = apparent_km(source, NAIF_SUN, time)?;
    let axis = unit3(sub3(moon, sun)).ok_or(AlmanacError::InvalidInput {
        field: "shadow_axis",
        reason: "degenerate",
    })?;
    let d = -dot3(moon, axis);
    let p = sub3(moon, scale3(axis, dot3(moon, axis)));
    let b = norm3(p);
    if !d.is_finite() || !b.is_finite() {
        return Err(AlmanacError::InvalidInput {
            field: "shadow_axis",
            reason: "must be finite",
        });
    }
    let gamma = b / MEAN_EARTH_RADIUS_KM;
    let moon_sun_distance = norm_checked(sub3(moon, sun), "moon_sun")?;
    let tan_f1 = (SOLAR_RADIUS_KM + MOON_RADIUS_KM) / moon_sun_distance;
    let l1 = (MOON_RADIUS_KM + d * tan_f1) / MEAN_EARTH_RADIUS_KM;
    Ok(SolarGeometry {
        moon,
        sun,
        axis,
        p,
        gamma,
        l1,
        disk_reaches_earth: gamma < 1.0 + l1,
    })
}

fn classify_solar(
    source: EphemerisSource<'_>,
    time_maximum: UtcInstant,
) -> Result<Option<EclipseEvent>, AlmanacError> {
    let geometry = solar_geometry(source, time_maximum)?;
    if !geometry.disk_reaches_earth {
        return Ok(None);
    }

    let Some(surface) = greatest_eclipse_surface_point(geometry)? else {
        return Ok(None);
    };
    let disk = solar_disk_geometry(geometry.moon, geometry.sun, surface.point)?;
    let magnitude = ((disk.s_sun + disk.s_moon - disk.sep) / (2.0 * disk.s_sun)).max(0.0);
    let mut kind = if disk.sep >= disk.s_sun + disk.s_moon {
        EclipseKind::SolarPartial
    } else if disk.s_moon >= disk.s_sun && disk.sep <= disk.s_moon - disk.s_sun {
        EclipseKind::SolarTotal
    } else if disk.s_moon < disk.s_sun && disk.sep <= disk.s_sun - disk.s_moon {
        EclipseKind::SolarAnnular
    } else {
        EclipseKind::SolarPartial
    };

    let mut uncertain = (geometry.gamma - (1.0 + geometry.l1)).abs() < GRAZING_GAMMA_BAND;
    let limb_boundary = (disk.sep - (disk.s_sun + disk.s_moon))
        .abs()
        .min((disk.sep - (disk.s_sun - disk.s_moon).abs()).abs());
    if limb_boundary / (2.0 * disk.s_sun) < GRAZING_MAGNITUDE_BAND {
        uncertain = true;
    }
    if let (Some(near), Some(far)) = (surface.near, surface.far) {
        let near_disk = solar_disk_geometry(geometry.moon, geometry.sun, near)?;
        let far_disk = solar_disk_geometry(geometry.moon, geometry.sun, far)?;
        if (near_disk.s_moon - near_disk.s_sun).signum()
            != (far_disk.s_moon - far_disk.s_sun).signum()
        {
            kind = EclipseKind::SolarHybrid;
            uncertain = true;
        }
    }

    let moon_latitude_deg = body_ecliptic(source, NAIF_MOON, time_maximum)?.latitude_deg;
    Ok(Some(EclipseEvent {
        time_maximum,
        kind,
        magnitude,
        moon_latitude_deg,
        gamma: geometry.gamma,
        uncertain,
    }))
}

#[derive(Debug, Clone, Copy)]
struct SurfacePoint {
    point: [f64; 3],
    near: Option<[f64; 3]>,
    far: Option<[f64; 3]>,
}

fn greatest_eclipse_surface_point(
    geometry: SolarGeometry,
) -> Result<Option<SurfacePoint>, AlmanacError> {
    let mk = dot3(geometry.moon, geometry.axis);
    let discriminant =
        mk * mk - (dot3(geometry.moon, geometry.moon) - MEAN_EARTH_RADIUS_KM.powi(2));
    if geometry.gamma < 1.0 && discriminant >= 0.0 {
        let root = discriminant.sqrt();
        let t_near = -mk - root;
        let t_far = -mk + root;
        let near = [
            geometry.moon[0] + t_near * geometry.axis[0],
            geometry.moon[1] + t_near * geometry.axis[1],
            geometry.moon[2] + t_near * geometry.axis[2],
        ];
        let far = [
            geometry.moon[0] + t_far * geometry.axis[0],
            geometry.moon[1] + t_far * geometry.axis[1],
            geometry.moon[2] + t_far * geometry.axis[2],
        ];
        Ok(Some(SurfacePoint {
            point: near,
            near: Some(near),
            far: Some(far),
        }))
    } else {
        let p_norm = norm_checked(geometry.p, "shadow_axis_offset")?;
        Ok(Some(SurfacePoint {
            point: scale3(geometry.p, MEAN_EARTH_RADIUS_KM / p_norm),
            near: None,
            far: None,
        }))
    }
}

#[derive(Debug, Clone, Copy)]
struct SolarDiskGeometry {
    s_sun: f64,
    s_moon: f64,
    sep: f64,
}

fn solar_disk_geometry(
    moon: [f64; 3],
    sun: [f64; 3],
    surface: [f64; 3],
) -> Result<SolarDiskGeometry, AlmanacError> {
    let sun_topo = sub3(sun, surface);
    let moon_topo = sub3(moon, surface);
    let sun_distance = norm_checked(sun_topo, "sun_topocentric")?;
    let moon_distance = norm_checked(moon_topo, "moon_topocentric")?;
    let s_sun = libm::asin(SOLAR_RADIUS_KM / sun_distance);
    let s_moon = libm::asin(MOON_RADIUS_KM / moon_distance);
    let sep = angular_separation_rad(sun_topo, moon_topo)?;
    Ok(SolarDiskGeometry { s_sun, s_moon, sep })
}
