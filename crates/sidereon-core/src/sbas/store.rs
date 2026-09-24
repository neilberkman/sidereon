use std::collections::{BTreeMap, BTreeSet};

use crate::astro::time::model::GnssWeekTow;
use crate::constants::{F_L1_HZ, GPS_EPOCH_TO_J2000_S, MEAN_EARTH_RADIUS_KM, SECONDS_PER_DAY};
use crate::error::{Error, Result};
use crate::frame::Wgs84Geodetic;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::ionex::pierce_point;
use crate::staleness::StalenessPolicy;
use crate::tolerances::SBAS_IGP_COORD_EPS_DEG;

use super::message::{
    SbasGeoNav, SbasIgpMask, SbasIntegrity, SbasIonoDelays, SbasLongTermHalf, SbasLongTermRecord,
    SbasMessage, SbasMixedCorrections,
};

const FAST_PRC_SCALE_M: f64 = 0.125;
const IONO_DELAY_SCALE_M: f64 = 0.125;
/// The all-ones nine-bit IGP vertical delay, which DO-229 defines as "do not
/// use".
const IGP_DELAY_DO_NOT_USE: u16 = 511;
/// The GIVEI value DO-229 defines as "not monitored".
const GIVEI_NOT_MONITORED: u8 = 15;
const LONG_POS_SCALE_M: f64 = 0.125;
/// RTKLIB `decode_longcorr1` scales the long-term velocity deltas by `P2_11`.
const LONG_RATE_SCALE_M_S: f64 = 1.0 / 2048.0;
const LONG_AF0_SCALE_S: f64 = 1.0 / 2_147_483_648.0;
const LONG_AF1_SCALE_S_S: f64 = 1.0 / 549_755_813_888.0;
const GEO_XY_POS_SCALE_M: f64 = 0.08;
const GEO_Z_POS_SCALE_M: f64 = 0.4;
const GEO_XY_RATE_SCALE_M_S: f64 = 0.000625;
const GEO_Z_RATE_SCALE_M_S: f64 = 0.004;
const GEO_XY_ACCEL_SCALE_M_S2: f64 = 0.0000125;
const GEO_Z_ACCEL_SCALE_M_S2: f64 = 0.0000625;
const GEO_AF0_SCALE_S: f64 = 1.0 / 2_147_483_648.0;
const GEO_AF1_SCALE_S_S: f64 = 1.0 / 1_099_511_627_776.0;
const GEO_DISABLED_TIMEOUT_S: f64 = 60.0;
const SBAS_SHELL_HEIGHT_KM: f64 = 350.0;

/// DO-229 UDRE variance table, in square meters, for UDREI values 0 through 13.
pub const SBAS_UDRE_VARIANCE_M2: [f64; 14] = [
    0.0520, 0.0924, 0.1444, 0.2830, 0.4678, 0.8315, 1.2992, 1.8709, 2.5465, 3.3260, 5.1968,
    20.7870, 230.9661, 2078.695,
];

/// RTKLIB `sbsioncorr` factor (per second) on a grid point's GIVE variance times its
/// age.
const RTKLIB_IONO_VARIANCE_AGE_FACTOR_PER_S: f64 = 9e-8;

/// DO-229 GIVE variance table, in square meters, for GIVEI values 0 through 14.
pub const SBAS_GIVE_VARIANCE_M2: [f64; 15] = [
    0.0084, 0.0333, 0.0749, 0.1331, 0.2079, 0.2994, 0.4075, 0.5322, 0.6735, 0.8315, 1.1974, 1.8709,
    3.3260, 20.7870, 187.0826,
];

/// Return the DO-229 UDRE variance for one UDREI value.
pub fn udre_variance_m2_for_udrei(udrei: u8) -> Option<f64> {
    SBAS_UDRE_VARIANCE_M2.get(usize::from(udrei)).copied()
}

/// Return the DO-229 GIVE variance for one GIVEI value.
pub fn give_variance_m2_for_givei(givei: u8) -> Option<f64> {
    SBAS_GIVE_VARIANCE_M2.get(usize::from(givei)).copied()
}

#[derive(Clone, Debug, PartialEq)]
/// Fast SBAS range correction retained for one monitored satellite.
///
/// [`SbasCorrectionStore::ingest`] derives the rate term from consecutive
/// corrections, and the corrected ephemeris uses both terms to extrapolate the
/// range correction into the satellite clock.
pub struct SbasFastCorrection {
    /// Current fast range correction in meters, after the signed message value
    /// is multiplied by the 0.125-meter scale factor.
    pub prc_m: f64,
    /// Rate derived from consecutive same-issue range corrections, in meters per second.
    pub rrc_m_s: f64,
    /// Current four-bit fast-correction integrity index; values at or above 14 withdraw the satellite.
    pub udrei: u8,
    /// Correction application epoch, including system latency, in seconds since J2000.
    pub t_of_j2000_s: f64,
    /// Fast-correction issue used for rate derivation and integrity matching.
    pub iodf: u8,
}

impl SbasFastCorrection {
    /// DO-229 UDRE variance, in square meters, for this fast correction.
    pub fn udre_variance_m2(&self) -> Option<f64> {
        udre_variance_m2_for_udrei(self.udrei)
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Long-term SBAS correction retained for a GPS satellite.
///
/// The store uses `iode` to select the broadcast state and advances the
/// position and clock deltas from [`SbasLongTermCorrection::t0_j2000_s`].
pub struct SbasLongTermCorrection {
    /// Broadcast ephemeris issue selected before applying this correction.
    pub iode: u8,
    /// ECEF position delta in meters at [`SbasLongTermCorrection::t0_j2000_s`],
    /// from the signed record coordinates multiplied by 0.125.
    pub delta_ecef_m: [f64; 3],
    /// ECEF position-delta rate in meters per second, from signed record rates
    /// multiplied by 2^-11.
    pub delta_ecef_rate_m_s: [f64; 3],
    /// Clock offset delta at the reference epoch, in seconds, from the signed
    /// record value multiplied by 1/2^31.
    pub delta_af0_s: f64,
    /// Clock-drift delta, in seconds per second, from the signed record value
    /// multiplied by 1/2^39.
    pub delta_af1_s_s: f64,
    /// Reference epoch for the position and clock deltas, in seconds since J2000.
    pub t0_j2000_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
/// One accepted SBAS ionospheric grid point.
///
/// [`SbasCorrectionStore::ingest`] obtains the coordinates from the DO-229
/// band table. Entries whose vertical delay is 511 or whose GIVEI is 15 are
/// not grid points; they are listed by [`SbasIonoGrid::unavailable_igps`].
pub struct SbasIgp {
    /// Grid latitude in degrees from the DO-229 band table, matched with the
    /// IGP coordinate tolerance during interpolation.
    pub lat_deg: f64,
    /// Grid longitude in degrees; lookup normalizes it to the [-180, 180] interval.
    pub lon_deg: f64,
    /// Vertical ionospheric delay in meters, from the message value multiplied
    /// by the 0.125-meter scale factor; interpolation uses this value.
    pub vertical_delay_m: f64,
    /// DO-229 GIVE variance in square meters, or None when no usable variance was supplied.
    pub give_variance_m2: Option<f64>,
    /// Epoch, seconds since J2000, of the message that gave the point its delay and
    /// GIVEI, which RTKLIB `decode_sbstype26` keeps as the point's `t0`.
    pub t0_j2000_s: f64,
}

#[derive(Clone, Debug, PartialEq, Default)]
/// IGP records associated with one SBAS ionospheric delay issue.
///
/// Ingestion replaces an existing coordinate match or appends a new point, so
/// the returned slice preserves the grid's update order.
pub struct SbasIonoGrid {
    igps: Vec<SbasIgp>,
    unavailable: Vec<SbasUnavailableIgp>,
    /// IODI associated with the stored IGP records.
    pub iodi: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why the latest ionospheric delay entry for a grid point makes it unusable.
pub enum SbasIgpUnavailableReason {
    /// The nine-bit vertical delay is 511, which DO-229 defines as "do not
    /// use". RTKLIB `decode_sbstype26` reads it as a zero delay instead.
    DoNotUse,
    /// The GIVEI is 15, which DO-229 defines as "not monitored".
    NotMonitored,
}

#[derive(Clone, Debug, PartialEq)]
/// A grid point whose latest delay entry makes it unusable, with the entry's
/// raw values.
pub struct SbasUnavailableIgp {
    /// Grid latitude in degrees from the DO-229 band table.
    pub lat_deg: f64,
    /// Grid longitude in degrees from the DO-229 band table.
    pub lon_deg: f64,
    /// The nine-bit vertical delay as broadcast.
    pub vertical_delay: u16,
    /// The four-bit GIVEI as broadcast.
    pub givei: u8,
    /// Why the point is unusable. A delay of 511 is reported as
    /// [`SbasIgpUnavailableReason::DoNotUse`] whatever its GIVEI.
    pub reason: SbasIgpUnavailableReason,
}

impl SbasIonoGrid {
    /// Construct an ionospheric grid from IGP records and the IODI value,
    /// with no unavailable points.
    pub fn new(igps: Vec<SbasIgp>, iodi: u8) -> Self {
        Self {
            igps,
            unavailable: Vec::new(),
            iodi,
        }
    }

    /// Ionospheric grid points in storage order.
    pub fn igps(&self) -> &[SbasIgp] {
        &self.igps
    }

    /// Grid points whose latest delay entry makes them unusable, in update
    /// order. Interpolation never uses them; a later usable entry for the same
    /// point moves it back to [`SbasIonoGrid::igps`].
    pub fn unavailable_igps(&self) -> &[SbasUnavailableIgp] {
        &self.unavailable
    }

    /// Interpolate the L1 slant ionospheric delay at a receiver look direction.
    pub fn slant_delay_m(
        &self,
        receiver: Wgs84Geodetic,
        elevation_rad: f64,
        azimuth_rad: f64,
        frequency_hz: f64,
    ) -> Option<f64> {
        if self.igps.is_empty() || !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return None;
        }
        let geom = pierce_point(
            receiver.lat_rad,
            receiver.lon_rad,
            azimuth_rad,
            elevation_rad,
            MEAN_EARTH_RADIUS_KM,
            SBAS_SHELL_HEIGHT_KM,
        );
        let vertical_delay_m =
            self.vertical_delay_at_ipp(geom.phi_ipp_deg, normalize_lon(geom.lambda_ipp_deg))?;
        let mapping = 1.0 / (1.0 - geom.s * geom.s).sqrt();
        let frequency_scale = (F_L1_HZ / frequency_hz) * (F_L1_HZ / frequency_hz);
        Some(vertical_delay_m * mapping * frequency_scale)
    }

    /// Variance (m²) of [`Self::slant_delay_m`] at a receiver look direction at
    /// `t_j2000_s`, as RTKLIB `sbsioncorr` states it: each covering point's GIVE
    /// variance times `9e-8 |t - t0|`, its age from [`SbasIgp::t0_j2000_s`] in seconds,
    /// interpolated with the delay's weights, mapped to the line of sight by the square
    /// of the obliquity factor, and scaled to `frequency_hz` by the square of the delay's
    /// `(f_L1 / f)²` scaling, as RTKLIB `rescode` scales it. A point without a GIVE
    /// variance contributes none, as RTKLIB `varicorr` gives none for a GIVEI it does
    /// not tabulate. `None` where the grid gives no delay.
    pub fn slant_variance_m2(
        &self,
        receiver: Wgs84Geodetic,
        elevation_rad: f64,
        azimuth_rad: f64,
        frequency_hz: f64,
        t_j2000_s: f64,
    ) -> Option<f64> {
        if self.igps.is_empty() || !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return None;
        }
        let geom = pierce_point(
            receiver.lat_rad,
            receiver.lon_rad,
            azimuth_rad,
            elevation_rad,
            MEAN_EARTH_RADIUS_KM,
            SBAS_SHELL_HEIGHT_KM,
        );
        let vertical_variance_m2 = self.interpolate_at_ipp(
            geom.phi_ipp_deg,
            normalize_lon(geom.lambda_ipp_deg),
            |point| {
                let give_variance_m2 = point.give_variance_m2.unwrap_or(0.0);
                give_variance_m2
                    * RTKLIB_IONO_VARIANCE_AGE_FACTOR_PER_S
                    * (t_j2000_s - point.t0_j2000_s).abs()
            },
        )?;
        let mapping = 1.0 / (1.0 - geom.s * geom.s).sqrt();
        let frequency_scale = (F_L1_HZ / frequency_hz) * (F_L1_HZ / frequency_hz);
        Some(vertical_variance_m2 * (mapping * mapping) * (frequency_scale * frequency_scale))
    }

    /// `value` of the covering points interpolated at an ionospheric pierce point with
    /// the weights [`Self::slant_delay_m`] interpolates the delay with: bilinear over
    /// four points, the plane through three.
    fn interpolate_at_ipp(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        value: impl Fn(&SbasIgp) -> f64,
    ) -> Option<f64> {
        let mut lats: Vec<f64> = self.igps.iter().map(|p| p.lat_deg).collect();
        lats.sort_by(f64_total_cmp);
        lats.dedup_by(|a, b| (*a - *b).abs() < SBAS_IGP_COORD_EPS_DEG);
        let mut lons: Vec<f64> = self.igps.iter().map(|p| normalize_lon(p.lon_deg)).collect();
        lons.sort_by(f64_total_cmp);
        lons.dedup_by(|a, b| (*a - *b).abs() < SBAS_IGP_COORD_EPS_DEG);
        let (lat0, lat1) = bracket_pair(&lats, lat_deg)?;
        let (lon0, lon1) = bracket_pair(&lons, lon_deg)?;
        if (lat1 - lat0).abs() < f64::EPSILON || (lon1 - lon0).abs() < f64::EPSILON {
            return None;
        }
        let corners = [
            self.find_igp(lat0, lon0),
            self.find_igp(lat0, lon1),
            self.find_igp(lat1, lon0),
            self.find_igp(lat1, lon1),
        ];
        let active: Vec<SbasIgp> = corners.into_iter().flatten().collect();
        match active.len() {
            4 => {
                let q = (lat_deg - lat0) / (lat1 - lat0);
                let p = (lon_deg - lon0) / (lon1 - lon0);
                let v00 = value(active_point(&active, lat0, lon0)?);
                let v01 = value(active_point(&active, lat0, lon1)?);
                let v10 = value(active_point(&active, lat1, lon0)?);
                let v11 = value(active_point(&active, lat1, lon1)?);
                Some(
                    (1.0 - p) * (1.0 - q) * v00
                        + p * (1.0 - q) * v01
                        + (1.0 - p) * q * v10
                        + p * q * v11,
                )
            }
            3 => plane_interpolate_value(&active, lat_deg, lon_deg, |point| Some(value(point))),
            _ => None,
        }
    }

    /// Interpolate the vertical GIVE variance at an ionospheric pierce point.
    pub fn variance_at_ipp(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        let variance_m2 = self
            .delay_variance_at_ipp(lat_deg, lon_deg)?
            .give_variance_m2?;
        (variance_m2.is_finite() && variance_m2 >= 0.0).then_some(variance_m2)
    }

    fn vertical_delay_at_ipp(&self, lat_deg: f64, lon_deg: f64) -> Option<f64> {
        self.delay_variance_at_ipp(lat_deg, lon_deg)
            .map(|value| value.vertical_delay_m)
    }

    fn delay_variance_at_ipp(&self, lat_deg: f64, lon_deg: f64) -> Option<IgpInterpolation> {
        let mut lats: Vec<f64> = self.igps.iter().map(|p| p.lat_deg).collect();
        lats.sort_by(f64_total_cmp);
        lats.dedup_by(|a, b| (*a - *b).abs() < SBAS_IGP_COORD_EPS_DEG);
        let mut lons: Vec<f64> = self.igps.iter().map(|p| normalize_lon(p.lon_deg)).collect();
        lons.sort_by(f64_total_cmp);
        lons.dedup_by(|a, b| (*a - *b).abs() < SBAS_IGP_COORD_EPS_DEG);
        let (lat0, lat1) = bracket_pair(&lats, lat_deg)?;
        let (lon0, lon1) = bracket_pair(&lons, lon_deg)?;
        if (lat1 - lat0).abs() < f64::EPSILON || (lon1 - lon0).abs() < f64::EPSILON {
            return None;
        }

        let corners = [
            self.find_igp(lat0, lon0),
            self.find_igp(lat0, lon1),
            self.find_igp(lat1, lon0),
            self.find_igp(lat1, lon1),
        ];
        let active: Vec<SbasIgp> = corners.into_iter().flatten().collect();
        match active.len() {
            4 => {
                let q = (lat_deg - lat0) / (lat1 - lat0);
                let p = (lon_deg - lon0) / (lon1 - lon0);
                let e00 = active_point(&active, lat0, lon0)?.vertical_delay_m;
                let e01 = active_point(&active, lat0, lon1)?.vertical_delay_m;
                let e10 = active_point(&active, lat1, lon0)?.vertical_delay_m;
                let e11 = active_point(&active, lat1, lon1)?.vertical_delay_m;
                let vertical_delay_m = (1.0 - p) * (1.0 - q) * e00
                    + p * (1.0 - q) * e01
                    + (1.0 - p) * q * e10
                    + p * q * e11;
                let give_variance_m2 = bilinear_give_variance_m2(&active, lat0, lat1, lon0, lon1)
                    .map(|(v00, v01, v10, v11)| {
                        (1.0 - p) * (1.0 - q) * v00
                            + p * (1.0 - q) * v01
                            + (1.0 - p) * q * v10
                            + p * q * v11
                    });
                Some(IgpInterpolation {
                    vertical_delay_m,
                    give_variance_m2,
                })
            }
            3 => Some(IgpInterpolation {
                vertical_delay_m: plane_interpolate(&active, lat_deg, lon_deg)?,
                give_variance_m2: plane_interpolate_give_variance(&active, lat_deg, lon_deg),
            }),
            _ => None,
        }
    }

    fn find_igp(&self, lat_deg: f64, lon_deg: f64) -> Option<SbasIgp> {
        self.igps
            .iter()
            .find(|p| {
                (p.lat_deg - lat_deg).abs() < SBAS_IGP_COORD_EPS_DEG
                    && (normalize_lon(p.lon_deg) - lon_deg).abs() < SBAS_IGP_COORD_EPS_DEG
            })
            .cloned()
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Propagated-state parameters decoded from an SBAS GEO navigation message.
///
/// The state is expressed in ECEF meters and seconds relative to its J2000
/// reference epoch; [`SbasGeoState::state_at`] applies the stored rates.
pub struct SbasGeoState {
    /// GEO ECEF position at [`SbasGeoState::t0_j2000_s`], in meters, after X/Y
    /// message values are scaled by 0.08 and Z by 0.4.
    pub position_ecef_m: [f64; 3],
    /// GEO ECEF velocity at the reference epoch, in meters per second, after
    /// X/Y message rates are scaled by 0.000625 and Z by 0.004.
    pub velocity_ecef_m_s: [f64; 3],
    /// GEO ECEF acceleration, in meters per second squared, after X/Y message
    /// accelerations are scaled by 0.0000125 and Z by 0.0000625.
    pub acceleration_ecef_m_s2: [f64; 3],
    /// GEO clock offset at the reference epoch, in seconds, after the message
    /// value is scaled by 1/2^31.
    pub clock_offset_s: f64,
    /// GEO clock drift, in seconds per second, after the message value is
    /// scaled by 1/2^40.
    pub clock_drift_s_s: f64,
    /// Reference epoch for position and clock propagation, in seconds since J2000.
    pub t0_j2000_s: f64,
    /// URA index from the message, which RTKLIB `decode_sbstype9` keeps as the GEO
    /// ephemeris `sva`.
    pub ura_index: u8,
}

impl SbasGeoState {
    /// Propagate the ECEF position and clock to a J2000-seconds epoch.
    ///
    /// Position uses the stored velocity and half the stored acceleration times
    /// elapsed time squared; clock uses offset plus drift times elapsed time.
    pub fn state_at(&self, t_j2000_s: f64) -> ([f64; 3], f64) {
        self.state_after(t_j2000_s - self.t0_j2000_s)
    }

    /// Propagate the ECEF position and clock `dt` seconds from the reference epoch.
    pub(crate) fn state_after(&self, dt: f64) -> ([f64; 3], f64) {
        let dt2 = dt * dt;
        let position = [
            self.position_ecef_m[0]
                + self.velocity_ecef_m_s[0] * dt
                + 0.5 * self.acceleration_ecef_m_s2[0] * dt2,
            self.position_ecef_m[1]
                + self.velocity_ecef_m_s[1] * dt
                + 0.5 * self.acceleration_ecef_m_s2[1] * dt2,
            self.position_ecef_m[2]
                + self.velocity_ecef_m_s[2] * dt
                + 0.5 * self.acceleration_ecef_m_s2[2] * dt2,
        ];
        let clock = self.clock_offset_s + self.clock_drift_s_s * dt;
        (position, clock)
    }
}

#[derive(Debug)]
/// In-memory SBAS correction state partitioned by source GEO.
///
/// The store retains the latest masks, corrections, navigation, ionospheric
/// data, withdrawal state, and policies used by the corrected-source lookups.
pub struct SbasCorrectionStore {
    partitions: BTreeMap<GnssSatelliteId, GeoPartition>,
    policy: StalenessPolicy,
    allow_partial: bool,
}

impl Default for SbasCorrectionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SbasCorrectionStore {
    /// Create an empty correction store.
    ///
    /// The initial freshness cap is 360 seconds and partial corrections are
    /// disabled until [`SbasCorrectionStore::allow_partial`] is enabled.
    pub fn new() -> Self {
        Self {
            partitions: BTreeMap::new(),
            policy: StalenessPolicy::seconds(360.0),
            allow_partial: false,
        }
    }

    /// Ingest one decoded [`SbasMessage`] for a source GEO at a GNSS epoch.
    ///
    /// A non-SBAS `geo` returns [`Error::InvalidInput`]. A message the wire
    /// form cannot carry as held returns [`Error::SbasEncode`] and changes
    /// nothing: ingest applies [`SbasMessage::validate`], the validation the
    /// encoder applies, so a hand-built message cannot address a mask block
    /// no message type names, supply a long-term half with a record count its
    /// velocity code does not carry, or give a non-velocity record a rate,
    /// clock drift or time of day that no message could deliver. The decoder
    /// only produces messages that pass. Supported messages update their
    /// corresponding masks, corrections, navigation, ionosphere, disable
    /// interval, or withdrawal state; unsupported variants are ignored.
    pub fn ingest(
        &mut self,
        message: &SbasMessage,
        geo: GnssSatelliteId,
        epoch: GnssWeekTow,
    ) -> Result<()> {
        if geo.system != GnssSystem::Sbas {
            return Err(Error::InvalidInput(
                "SBAS source GEO must be an SBAS id".to_string(),
            ));
        }
        message.validate()?;
        let epoch_j2000_s = epoch_to_j2000_s(epoch);
        let partition = self.partitions.entry(geo).or_default();
        partition.last_update_j2000_s = epoch_j2000_s;
        match message {
            SbasMessage::DoNotUse(_) => {
                partition.disabled_until_j2000_s = Some(epoch_j2000_s + GEO_DISABLED_TIMEOUT_S);
            }
            SbasMessage::PrnMask(mask) => {
                partition.active_iodp = Some(mask.iodp);
                partition
                    .masks
                    .insert(mask.iodp, resolve_prn_mask(&mask.mask));
            }
            SbasMessage::FastCorrections(fast) => {
                ingest_fast(
                    partition,
                    fast.message_type,
                    fast.iodf,
                    fast.iodp,
                    &fast.prc,
                    &fast.udrei,
                    epoch_j2000_s,
                );
            }
            SbasMessage::Integrity(integrity) => {
                ingest_integrity(partition, integrity);
            }
            SbasMessage::FastDegradation(degradation) => {
                if Some(degradation.iodp) == partition.active_iodp {
                    partition.system_latency_s = f64::from(degradation.system_latency_s);
                    ingest_degradation_factors(partition, degradation.iodp, &degradation.ai);
                }
            }
            SbasMessage::GeoNav(geo_nav) => {
                partition.geo_nav = Some(Timed {
                    value: geo_state_from_message(geo_nav, epoch),
                    epoch_j2000_s,
                });
            }
            SbasMessage::IgpMask(mask) => {
                partition
                    .igp_masks
                    .insert((mask.band_number, mask.iodi), mask.clone());
            }
            SbasMessage::IonoDelays(delays) => {
                ingest_iono(partition, delays, epoch_j2000_s);
            }
            SbasMessage::MixedCorrections(mixed) => {
                ingest_mixed(partition, mixed, epoch, epoch_j2000_s);
            }
            SbasMessage::LongTermCorrections(long) => {
                for half in &long.halves {
                    ingest_long_half(partition, half, epoch, epoch_j2000_s);
                }
            }
            SbasMessage::NetworkTime(_)
            | SbasMessage::GeoAlmanac(_)
            | SbasMessage::Unsupported(_) => {}
        }
        Ok(())
    }

    /// List source GEOs eligible for corrected-ephemeris selection at an epoch.
    ///
    /// A GEO must be enabled, have an active PRN mask, and contain fast
    /// corrections, an ionospheric grid, or GEO navigation. Results are ordered
    /// by newest partition update first.
    pub fn ready_geos(&self, t_j2000_s: f64) -> Vec<GnssSatelliteId> {
        let mut geos: Vec<(GnssSatelliteId, f64)> = self
            .partitions
            .iter()
            .filter(|(_, p)| !p.is_disabled(t_j2000_s))
            .filter(|(_, p)| p.active_iodp.is_some())
            .filter(|(_, p)| p.iono_grid.is_some() || p.geo_nav.is_some() || !p.fast.is_empty())
            .map(|(&geo, p)| (geo, p.last_update_j2000_s))
            .collect();
        geos.sort_by(|a, b| f64_total_cmp(&b.1, &a.1));
        geos.into_iter().map(|(geo, _)| geo).collect()
    }

    /// Return the latest stored fast correction for a source GEO and satellite.
    ///
    /// `None` means that the partition or satellite has no fast correction; this
    /// public getter does not apply the internal freshness check.
    pub fn fast(&self, geo: GnssSatelliteId, sat: GnssSatelliteId) -> Option<&SbasFastCorrection> {
        self.partitions.get(&geo)?.fast.get(&sat).map(|t| &t.value)
    }

    /// Count the corrections a source GEO addressed to active PRN-mask bits that
    /// name no satellite held here, per 1-based PRN mask number.
    ///
    /// Such a bit - a future-GNSS or unassigned mask number - keeps its place
    /// among the active bits, so the corrections after it still reach their own
    /// satellites. The corrections addressed to it are applied to no satellite
    /// and are counted here instead of being dropped silently. `None` means the
    /// GEO has no partition.
    pub fn unassigned_mask_corrections(&self, geo: GnssSatelliteId) -> Option<&BTreeMap<u8, u64>> {
        self.partitions
            .get(&geo)
            .map(|partition| &partition.unassigned_corrections)
    }

    /// Return the latest stored long-term correction for a source GEO and satellite.
    ///
    /// `None` means that no correction is stored for the pair; this public
    /// getter does not apply the internal freshness check.
    pub fn long_term(
        &self,
        geo: GnssSatelliteId,
        sat: GnssSatelliteId,
    ) -> Option<&SbasLongTermCorrection> {
        self.partitions
            .get(&geo)?
            .long_term
            .get(&sat)
            .map(|t| &t.value)
    }

    /// Return the stored ionospheric grid unless the GEO is currently disabled.
    ///
    /// The disable check evaluates the partition's latest ingested epoch, so a
    /// grid is hidden immediately after a `DoNotUse` message.
    pub fn iono_grid(&self, geo: GnssSatelliteId) -> Option<&SbasIonoGrid> {
        let partition = self.partitions.get(&geo)?;
        if partition.is_disabled(partition.last_update_j2000_s) {
            return None;
        }
        partition.iono_grid.as_ref().map(|t| &t.value)
    }

    /// Return the latest stored GEO navigation state for a source GEO.
    ///
    /// `None` means that no GEO navigation message has been ingested; this
    /// public getter does not apply freshness filtering.
    pub fn geo_nav(&self, geo: GnssSatelliteId) -> Option<&SbasGeoState> {
        self.partitions
            .get(&geo)?
            .geo_nav
            .as_ref()
            .map(|t| &t.value)
    }

    /// Replace the freshness policy used by internal correction lookups.
    ///
    /// A source is accepted when the absolute query/source epoch difference is
    /// no greater than the policy's `max_staleness_s` value.
    pub fn with_policy(mut self, policy: StalenessPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set whether corrected ephemeris may use only one available correction.
    ///
    /// When enabled, the corrected source may use a fast-only or long-term-only
    /// correction; the default is disabled.
    pub fn allow_partial(mut self, yes: bool) -> Self {
        self.allow_partial = yes;
        self
    }

    pub(crate) fn fresh_fast(
        &self,
        geo: GnssSatelliteId,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<&SbasFastCorrection> {
        let p = self.partitions.get(&geo)?;
        let timed = p.fast.get(&sat)?;
        self.fresh(timed.epoch_j2000_s, t_j2000_s)
            .then_some(&timed.value)
    }

    /// Variance (m²) of the fresh fast correction for `sat` at `t_j2000_s`, as RTKLIB
    /// `sbsfastcorr` states it: `varfcorr` of the UDREI (the DO-229 UDRE variance) plus
    /// the degradation `degfcorr(ai) t² / 2`, with `t` the time since the correction's
    /// message plus the system latency and `ai` the message type 7 degradation indicator
    /// of the satellite's mask slot. Without a type 7 indicator `ai` is 0, which
    /// `degfcorr` reads as the largest factor, 0.0058 m/s², as RTKLIB does. `None`
    /// without a fresh fast correction.
    pub(crate) fn fast_correction_variance_m2(
        &self,
        geo: GnssSatelliteId,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<f64> {
        let fast = self.fresh_fast(geo, sat, t_j2000_s)?;
        let p = self.partitions.get(&geo)?;
        let message_epoch_j2000_s = p
            .fast_message_epoch_j2000_s
            .get(&sat)
            .copied()
            .unwrap_or(fast.t_of_j2000_s - p.system_latency_s);
        let t = t_j2000_s - message_epoch_j2000_s + p.system_latency_s;
        let ai = p.fast_degradation_ai.get(&sat).copied().unwrap_or(0);
        let udre_variance_m2 = udre_variance_m2_for_udrei(fast.udrei).unwrap_or(0.0);
        Some(udre_variance_m2 + fast_degradation_factor(ai) * t * t / 2.0)
    }

    pub(crate) fn fresh_long_term(
        &self,
        geo: GnssSatelliteId,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<&SbasLongTermCorrection> {
        let p = self.partitions.get(&geo)?;
        let timed = p.long_term.get(&sat)?;
        self.fresh(timed.epoch_j2000_s, t_j2000_s)
            .then_some(&timed.value)
    }

    pub(crate) fn fresh_iono_grid(
        &self,
        geo: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<&SbasIonoGrid> {
        let p = self.partitions.get(&geo)?;
        let timed = p.iono_grid.as_ref()?;
        (!p.is_disabled(t_j2000_s) && self.fresh(timed.epoch_j2000_s, t_j2000_s))
            .then_some(&timed.value)
    }

    pub(crate) fn fresh_geo_nav(
        &self,
        geo: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<&SbasGeoState> {
        let timed = self.partitions.get(&geo)?.geo_nav.as_ref()?;
        self.fresh(timed.epoch_j2000_s, t_j2000_s)
            .then_some(&timed.value)
    }

    pub(crate) fn is_disabled(&self, geo: GnssSatelliteId, t_j2000_s: f64) -> bool {
        self.partitions
            .get(&geo)
            .is_some_and(|p| p.is_disabled(t_j2000_s))
    }

    pub(crate) fn is_withdrawn(&self, geo: GnssSatelliteId, sat: GnssSatelliteId) -> bool {
        self.partitions
            .get(&geo)
            .is_some_and(|p| p.withdrawn.contains(&sat))
    }

    pub(crate) fn allow_partial_corrections(&self) -> bool {
        self.allow_partial
    }

    #[cfg(test)]
    pub(crate) fn insert_long_term_for_test(
        &mut self,
        geo: GnssSatelliteId,
        sat: GnssSatelliteId,
        correction: SbasLongTermCorrection,
        epoch_j2000_s: f64,
    ) {
        self.partitions.entry(geo).or_default().long_term.insert(
            sat,
            Timed {
                value: correction,
                epoch_j2000_s,
            },
        );
    }

    fn fresh(&self, source_j2000_s: f64, t_j2000_s: f64) -> bool {
        (t_j2000_s - source_j2000_s).abs() <= self.policy.max_staleness_s
    }
}

#[derive(Debug, Default)]
struct GeoPartition {
    active_iodp: Option<u8>,
    masks: BTreeMap<u8, Vec<MaskSlot>>,
    /// Corrections addressed to a mask slot that names no satellite held here,
    /// counted per 1-based PRN mask number.
    unassigned_corrections: BTreeMap<u8, u64>,
    fast: BTreeMap<GnssSatelliteId, Timed<SbasFastCorrection>>,
    previous_fast: BTreeMap<GnssSatelliteId, SbasFastCorrection>,
    long_term: BTreeMap<GnssSatelliteId, Timed<SbasLongTermCorrection>>,
    igp_masks: BTreeMap<(u8, u8), SbasIgpMask>,
    iono_grid: Option<Timed<SbasIonoGrid>>,
    geo_nav: Option<Timed<SbasGeoState>>,
    withdrawn: BTreeSet<GnssSatelliteId>,
    /// Epoch of the message each satellite's current fast correction came in.
    fast_message_epoch_j2000_s: BTreeMap<GnssSatelliteId, f64>,
    /// Message type 7 fast-correction degradation indicator of each satellite.
    fast_degradation_ai: BTreeMap<GnssSatelliteId, u8>,
    system_latency_s: f64,
    disabled_until_j2000_s: Option<f64>,
    last_update_j2000_s: f64,
}

impl GeoPartition {
    fn is_disabled(&self, t_j2000_s: f64) -> bool {
        self.disabled_until_j2000_s
            .is_some_and(|until| t_j2000_s <= until)
    }
}

#[derive(Debug)]
struct Timed<T> {
    value: T,
    epoch_j2000_s: f64,
}

fn ingest_fast(
    partition: &mut GeoPartition,
    message_type: u8,
    iodf: u8,
    iodp: u8,
    prc: &[i16],
    udrei: &[u8],
    epoch_j2000_s: f64,
) {
    if Some(iodp) != partition.active_iodp {
        return;
    }
    // `ingest` validates the message type, 2 through 5, before this point.
    let start = usize::from(message_type.saturating_sub(2)) * 13;
    for (i, (&prc_raw, &udrei)) in prc.iter().zip(udrei.iter()).enumerate() {
        let Some(sat) = monitored_sat(partition, iodp, start + i) else {
            continue;
        };
        if udrei >= 14 {
            partition.withdrawn.insert(sat);
            continue;
        }
        let t_of_j2000_s = epoch_j2000_s + partition.system_latency_s;
        let prc_m = f64::from(prc_raw) * FAST_PRC_SCALE_M;
        let rrc_m_s = partition
            .previous_fast
            .get(&sat)
            .filter(|prev| prev.iodf == iodf)
            .and_then(|prev| {
                let dt = t_of_j2000_s - prev.t_of_j2000_s;
                (dt > 0.0).then_some((prc_m - prev.prc_m) / dt)
            })
            .unwrap_or(0.0);
        let correction = SbasFastCorrection {
            prc_m,
            rrc_m_s,
            udrei,
            t_of_j2000_s,
            iodf,
        };
        partition.previous_fast.insert(sat, correction.clone());
        partition
            .fast_message_epoch_j2000_s
            .insert(sat, epoch_j2000_s);
        partition.fast.insert(
            sat,
            Timed {
                value: correction,
                epoch_j2000_s: t_of_j2000_s,
            },
        );
        partition.withdrawn.remove(&sat);
    }
}

/// RTKLIB `degfcorr` fast-correction degradation factors (m/s²) of the type 7 indicators
/// 1 through 15.
const FAST_DEGRADATION_FACTORS_M_S2: [f64; 16] = [
    0.00000, 0.00005, 0.00009, 0.00012, 0.00015, 0.00020, 0.00030, 0.00045, 0.00060, 0.00090,
    0.00150, 0.00210, 0.00270, 0.00330, 0.00460, 0.00580,
];

/// RTKLIB `degfcorr`: the factor of indicator `ai` 1 through 15, and 0.0058 m/s² for
/// indicator 0, which is also the value of a satellite no type 7 message has reached.
fn fast_degradation_factor(ai: u8) -> f64 {
    if (1..=15).contains(&ai) {
        FAST_DEGRADATION_FACTORS_M_S2[usize::from(ai)]
    } else {
        0.0058
    }
}

/// Record the message type 7 degradation indicator of each satellite in the PRN mask of
/// issue `iodp`, by mask slot, as RTKLIB `decode_sbstype7` records them.
fn ingest_degradation_factors(partition: &mut GeoPartition, iodp: u8, ai: &[u8]) {
    let Some(mask) = partition.masks.get(&iodp) else {
        return;
    };
    for (slot, &indicator) in mask.iter().zip(ai) {
        if let MaskSlot::Satellite(sat) = *slot {
            partition.fast_degradation_ai.insert(sat, indicator);
        }
    }
}

fn ingest_integrity(partition: &mut GeoPartition, integrity: &SbasIntegrity) {
    let Some(iodp) = partition.active_iodp else {
        return;
    };
    for (idx, &udrei) in integrity.udrei.iter().enumerate() {
        let Some(sat) = monitored_sat(partition, iodp, idx) else {
            continue;
        };
        let block = idx / 13;
        let Some(fast) = partition.fast.get_mut(&sat) else {
            continue;
        };
        if integrity.iodf[block] != fast.value.iodf {
            continue;
        }
        fast.value.udrei = udrei;
        if udrei >= 14 {
            partition.withdrawn.insert(sat);
        }
    }
}

fn ingest_mixed(
    partition: &mut GeoPartition,
    mixed: &SbasMixedCorrections,
    epoch: GnssWeekTow,
    epoch_j2000_s: f64,
) {
    ingest_fast(
        partition,
        2 + mixed.fast.block_id,
        mixed.fast.iodf,
        mixed.fast.iodp,
        &mixed.fast.prc,
        &mixed.fast.udrei,
        epoch_j2000_s,
    );
    ingest_long_half(partition, &mixed.long_term, epoch, epoch_j2000_s);
}

fn ingest_long_half(
    partition: &mut GeoPartition,
    half: &SbasLongTermHalf,
    epoch: GnssWeekTow,
    epoch_j2000_s: f64,
) {
    if Some(half.iodp) != partition.active_iodp {
        return;
    }
    for record in &half.records {
        // Mask index 0 is the fill value of an unused record slot, not the
        // first monitored satellite.
        let Some(zero_based_index) = record.monitored_index.checked_sub(1) else {
            continue;
        };
        let Some(sat) = monitored_sat(partition, half.iodp, usize::from(zero_based_index)) else {
            continue;
        };
        if sat.system != GnssSystem::Gps {
            continue;
        }
        let t0_j2000_s = record
            .time_of_day_s
            .map(|tod| lift_time_of_day(epoch, f64::from(tod) * 16.0))
            .unwrap_or(epoch_j2000_s);
        let correction = long_record_to_correction(record, t0_j2000_s);
        partition.long_term.insert(
            sat,
            Timed {
                value: correction,
                epoch_j2000_s,
            },
        );
    }
}

fn ingest_iono(partition: &mut GeoPartition, delays: &SbasIonoDelays, epoch_j2000_s: f64) {
    let Some(mask) = partition.igp_masks.get(&(delays.band_number, delays.iodi)) else {
        return;
    };
    let active_positions: Vec<usize> = mask
        .mask
        .iter()
        .enumerate()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect();
    let start = usize::from(delays.block_id) * delays.entries.len();
    let (mut igps, mut unavailable) = partition
        .iono_grid
        .as_ref()
        .filter(|grid| grid.value.iodi == delays.iodi)
        .map(|grid| (grid.value.igps.clone(), grid.value.unavailable.clone()))
        .unwrap_or_default();
    for (slot, entry) in delays.entries.iter().enumerate() {
        let Some(&position) = active_positions.get(start + slot) else {
            continue;
        };
        let Some((lat_deg, lon_deg)) = igp_location(delays.band_number, position) else {
            continue;
        };
        let same_point = |point_lat_deg: f64, point_lon_deg: f64| {
            (point_lat_deg - lat_deg).abs() < SBAS_IGP_COORD_EPS_DEG
                && (normalize_lon(point_lon_deg) - normalize_lon(lon_deg)).abs()
                    < SBAS_IGP_COORD_EPS_DEG
        };
        unavailable.retain(|p| !same_point(p.lat_deg, p.lon_deg));
        let reason = if entry.vertical_delay == IGP_DELAY_DO_NOT_USE {
            Some(SbasIgpUnavailableReason::DoNotUse)
        } else if entry.givei == GIVEI_NOT_MONITORED {
            Some(SbasIgpUnavailableReason::NotMonitored)
        } else {
            None
        };
        if let Some(reason) = reason {
            // An earlier delay for this point no longer holds. For GIVEI 15
            // RTKLIB `decode_sbstype26` agrees, setting the GIVE to 0, which
            // `searchigp` skips.
            igps.retain(|p| !same_point(p.lat_deg, p.lon_deg));
            unavailable.push(SbasUnavailableIgp {
                lat_deg,
                lon_deg,
                vertical_delay: entry.vertical_delay,
                givei: entry.givei,
                reason,
            });
            continue;
        }
        let point = SbasIgp {
            lat_deg,
            lon_deg,
            vertical_delay_m: f64::from(entry.vertical_delay) * IONO_DELAY_SCALE_M,
            give_variance_m2: give_variance_m2_for_givei(entry.givei),
            t0_j2000_s: epoch_j2000_s,
        };
        if let Some(existing) = igps.iter_mut().find(|p| same_point(p.lat_deg, p.lon_deg)) {
            *existing = point;
        } else {
            igps.push(point);
        }
    }
    partition.iono_grid = Some(Timed {
        value: SbasIonoGrid {
            igps,
            unavailable,
            iodi: delays.iodi,
        },
        epoch_j2000_s,
    });
}

fn long_record_to_correction(
    record: &SbasLongTermRecord,
    t0_j2000_s: f64,
) -> SbasLongTermCorrection {
    SbasLongTermCorrection {
        iode: record.iode,
        delta_ecef_m: [
            f64::from(record.delta_x) * LONG_POS_SCALE_M,
            f64::from(record.delta_y) * LONG_POS_SCALE_M,
            f64::from(record.delta_z) * LONG_POS_SCALE_M,
        ],
        delta_ecef_rate_m_s: [
            f64::from(record.delta_x_rate) * LONG_RATE_SCALE_M_S,
            f64::from(record.delta_y_rate) * LONG_RATE_SCALE_M_S,
            f64::from(record.delta_z_rate) * LONG_RATE_SCALE_M_S,
        ],
        delta_af0_s: f64::from(record.delta_a_f0) * LONG_AF0_SCALE_S,
        delta_af1_s_s: f64::from(record.delta_a_f1) * LONG_AF1_SCALE_S_S,
        t0_j2000_s,
    }
}

fn geo_state_from_message(message: &SbasGeoNav, epoch: GnssWeekTow) -> SbasGeoState {
    SbasGeoState {
        position_ecef_m: [
            f64::from(message.x_m) * GEO_XY_POS_SCALE_M,
            f64::from(message.y_m) * GEO_XY_POS_SCALE_M,
            f64::from(message.z_m) * GEO_Z_POS_SCALE_M,
        ],
        velocity_ecef_m_s: [
            f64::from(message.x_rate_m_s) * GEO_XY_RATE_SCALE_M_S,
            f64::from(message.y_rate_m_s) * GEO_XY_RATE_SCALE_M_S,
            f64::from(message.z_rate_m_s) * GEO_Z_RATE_SCALE_M_S,
        ],
        acceleration_ecef_m_s2: [
            f64::from(message.x_accel_m_s2) * GEO_XY_ACCEL_SCALE_M_S2,
            f64::from(message.y_accel_m_s2) * GEO_XY_ACCEL_SCALE_M_S2,
            f64::from(message.z_accel_m_s2) * GEO_Z_ACCEL_SCALE_M_S2,
        ],
        clock_offset_s: f64::from(message.a_gf0_s) * GEO_AF0_SCALE_S,
        clock_drift_s_s: f64::from(message.a_gf1_s_s) * GEO_AF1_SCALE_S_S,
        t0_j2000_s: lift_time_of_day(epoch, f64::from(message.time_of_day_s) * 16.0),
        ura_index: message.ura,
    }
}

/// One active bit of a PRN mask, in set-bit order.
///
/// Corrections address the monitored satellites by their position among the
/// active bits, so every active bit keeps its place whether or not it names a
/// satellite held here. Dropping an unassigned bit would move every later
/// correction onto the next satellite in the mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskSlot {
    /// The bit names this satellite.
    Satellite(GnssSatelliteId),
    /// The bit names no satellite held here; the value is its 1-based PRN mask
    /// number.
    Unassigned(u8),
}

/// The satellite a correction at `zero_based_index` of the `iodp` mask
/// addresses. `None` when the index is past the mask's active bits or names an
/// unassigned mask number, in which case the correction is counted against
/// that mask number and applied to no satellite.
fn monitored_sat(
    partition: &mut GeoPartition,
    iodp: u8,
    zero_based_index: usize,
) -> Option<GnssSatelliteId> {
    let slot = *partition.masks.get(&iodp)?.get(zero_based_index)?;
    match slot {
        MaskSlot::Satellite(sat) => Some(sat),
        MaskSlot::Unassigned(mask_number) => {
            *partition
                .unassigned_corrections
                .entry(mask_number)
                .or_insert(0) += 1;
            None
        }
    }
}

fn resolve_prn_mask(mask: &[bool; 210]) -> Vec<MaskSlot> {
    mask.iter()
        .enumerate()
        .filter(|(_, active)| **active)
        .map(|(position, _)| match mask_position_to_sat(position) {
            Some(sat) => MaskSlot::Satellite(sat),
            // 210 positions, so the 1-based mask number always fits a u8.
            None => MaskSlot::Unassigned(u8::try_from(position + 1).unwrap_or(u8::MAX)),
        })
        .collect()
}

/// Map a zero-based PRN-mask position to a satellite id.
///
/// The mask is a fixed 210-position table whose assignments are RTCA DO-229's
/// PRN mask table. With `n` the 1-based mask number, `position + 1`:
///
/// - `n` 1..=37 are GPS PRN `n` (positions 0..=36);
/// - `n` 38..=61 are GLONASS slot `n - 37`, slots 1..=24 (positions 37..=60);
/// - `n` 62..=119 are future GNSS, unassigned;
/// - `n` 120..=158 are SBAS broadcast PRN `n` (positions 119..=157);
/// - `n` 159..=210 are unassigned.
///
/// Those bounds belong to the mask definition, not to the shared
/// satellite-token range, so no extended GLONASS slot such as `R28` has a mask
/// position. RTKLIB `decode_sbstype1` agrees only for `n` 1..=61 and
/// 120..=138: it reads 139..=182 as reserved, and 183..=202 as QZSS L1S and
/// QZSS PRNs from the QZSS interface specification. That QZSS assignment is not
/// DO-229's and is not read here, so those positions stay unassigned.
fn mask_position_to_sat(position: usize) -> Option<GnssSatelliteId> {
    let mask_number = position + 1;
    match mask_number {
        1..=37 => GnssSatelliteId::new(GnssSystem::Gps, u8::try_from(mask_number).ok()?).ok(),
        38..=61 => {
            GnssSatelliteId::new(GnssSystem::Glonass, u8::try_from(mask_number - 37).ok()?).ok()
        }
        120..=158 => sbas_prn_to_sat(u16::try_from(mask_number).ok()?),
        _ => None,
    }
}

/// The SBAS broadcast PRN window, and the stored slot window it maps onto.
///
/// These two are the same 39 satellites written two ways: the broadcast PRN as
/// it appears in the SBAS signal and in the WAAS/EGNOS message definitions, and
/// the `S20`..`S58` slot form the IGS product formats use. The pairing, not the
/// shared `S01`..`S99` token range, is what bounds these converters.
const SBAS_BROADCAST_PRN_RANGE: core::ops::RangeInclusive<u16> = 120..=158;
/// The stored slot form of [`SBAS_BROADCAST_PRN_RANGE`].
const SBAS_SLOT_RANGE: core::ops::RangeInclusive<u16> = 20..=58;
/// The fixed offset between the two windows.
const SBAS_PRN_MINUS_SLOT: u16 = 100;

/// Convert an SBAS broadcast PRN to the library's slot-form satellite id.
///
/// Broadcast PRNs 120 through 158 map to SBAS slots 20 through 58; values
/// outside that interval return `None`.
pub fn sbas_prn_to_sat(broadcast_prn: u16) -> Option<GnssSatelliteId> {
    if !SBAS_BROADCAST_PRN_RANGE.contains(&broadcast_prn) {
        return None;
    }
    let slot = broadcast_prn.checked_sub(SBAS_PRN_MINUS_SLOT)?;
    GnssSatelliteId::new(GnssSystem::Sbas, u8::try_from(slot).ok()?).ok()
}

/// Convert an SBAS slot-form satellite id to its broadcast PRN.
///
/// Only the SBAS slots that a broadcast PRN exists for convert: slot 20 through
/// 58, giving PRN 120 through 158. Every other input returns `None` - an id from
/// another constellation, an SBAS slot outside the window (`S01`, `S19`, `S59`,
/// `S99` are all spellable satellite tokens but name no broadcast PRN), and a
/// `prn` that bypassed [`GnssSatelliteId::new`] through the public fields (`0`,
/// `100`, `255`). The shared satellite-token range is `1..=99` for every
/// constellation, so this check is the only thing standing between a widened id
/// and an invented PRN such as 101 or 355.
pub fn sat_to_sbas_prn(sat: GnssSatelliteId) -> Option<u16> {
    if sat.system != GnssSystem::Sbas {
        return None;
    }
    let slot = u16::from(sat.prn);
    if !SBAS_SLOT_RANGE.contains(&slot) {
        return None;
    }
    slot.checked_add(SBAS_PRN_MINUS_SLOT)
}

fn epoch_to_j2000_s(epoch: GnssWeekTow) -> f64 {
    f64::from(epoch.week) * crate::constants::SECONDS_PER_WEEK + epoch.tow_s - GPS_EPOCH_TO_J2000_S
}

/// Place a message time of day in time, as RTKLIB `decode_sbstype9` and
/// `decode_longcorr1` place it. RTKLIB `readmsgs` rounds the message time of
/// week to a whole second, `(int)(tow+0.5)`; the decoders form
/// `t = tod - tow % 86400`, move `t` into (-43200, 43200] by a day, and add it
/// to the rounded time of week.
/// The day is decided on that rounded second, and the result is the start of
/// the chosen day plus the time of day, exactly. A time of day exactly half a
/// day from the rounded epoch falls half a day after it.
fn lift_time_of_day(epoch: GnssWeekTow, time_of_day_s: f64) -> f64 {
    let half_day = SECONDS_PER_DAY / 2.0;
    let rounded_tow_s = (epoch.tow_s + 0.5).floor();
    let second_of_day_s = rounded_tow_s.rem_euclid(SECONDS_PER_DAY);
    let mut offset = time_of_day_s - second_of_day_s;
    if offset <= -half_day {
        offset += SECONDS_PER_DAY;
    } else if offset > half_day {
        offset -= SECONDS_PER_DAY;
    }
    f64::from(epoch.week) * crate::constants::SECONDS_PER_WEEK + rounded_tow_s + offset
        - GPS_EPOCH_TO_J2000_S
}

#[derive(Clone, Copy)]
struct IgpBandSegment {
    fixed_deg: i16,
    variable_deg: &'static [i16],
    first_bit: usize,
    last_bit: usize,
}

const IGP_X1: [i16; 28] = [
    -75, -65, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
    40, 45, 50, 55, 65, 75, 85,
];
const IGP_X2: [i16; 23] = [
    -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    55,
];
const IGP_X3: [i16; 27] = [
    -75, -65, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
    40, 45, 50, 55, 65, 75,
];
const IGP_X4: [i16; 28] = [
    -85, -75, -65, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30,
    35, 40, 45, 50, 55, 65, 75,
];
const IGP_X5: [i16; 72] = [
    -180, -175, -170, -165, -160, -155, -150, -145, -140, -135, -130, -125, -120, -115, -110, -105,
    -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10,
    -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
    110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
];
const IGP_X6: [i16; 36] = [
    -180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20,
    -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
];
const IGP_X7: [i16; 12] = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150];
const IGP_X8: [i16; 12] = [-170, -140, -110, -80, -50, -20, 10, 40, 70, 100, 130, 160];

const IGP_BANDS_LOW: [[IgpBandSegment; 8]; 9] = [
    [
        IgpBandSegment {
            fixed_deg: -180,
            variable_deg: &IGP_X1,
            first_bit: 1,
            last_bit: 28,
        },
        IgpBandSegment {
            fixed_deg: -175,
            variable_deg: &IGP_X2,
            first_bit: 29,
            last_bit: 51,
        },
        IgpBandSegment {
            fixed_deg: -170,
            variable_deg: &IGP_X3,
            first_bit: 52,
            last_bit: 78,
        },
        IgpBandSegment {
            fixed_deg: -165,
            variable_deg: &IGP_X2,
            first_bit: 79,
            last_bit: 101,
        },
        IgpBandSegment {
            fixed_deg: -160,
            variable_deg: &IGP_X3,
            first_bit: 102,
            last_bit: 128,
        },
        IgpBandSegment {
            fixed_deg: -155,
            variable_deg: &IGP_X2,
            first_bit: 129,
            last_bit: 151,
        },
        IgpBandSegment {
            fixed_deg: -150,
            variable_deg: &IGP_X3,
            first_bit: 152,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: -145,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: -140,
            variable_deg: &IGP_X4,
            first_bit: 1,
            last_bit: 28,
        },
        IgpBandSegment {
            fixed_deg: -135,
            variable_deg: &IGP_X2,
            first_bit: 29,
            last_bit: 51,
        },
        IgpBandSegment {
            fixed_deg: -130,
            variable_deg: &IGP_X3,
            first_bit: 52,
            last_bit: 78,
        },
        IgpBandSegment {
            fixed_deg: -125,
            variable_deg: &IGP_X2,
            first_bit: 79,
            last_bit: 101,
        },
        IgpBandSegment {
            fixed_deg: -120,
            variable_deg: &IGP_X3,
            first_bit: 102,
            last_bit: 128,
        },
        IgpBandSegment {
            fixed_deg: -115,
            variable_deg: &IGP_X2,
            first_bit: 129,
            last_bit: 151,
        },
        IgpBandSegment {
            fixed_deg: -110,
            variable_deg: &IGP_X3,
            first_bit: 152,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: -105,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: -100,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: -95,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: -90,
            variable_deg: &IGP_X1,
            first_bit: 51,
            last_bit: 78,
        },
        IgpBandSegment {
            fixed_deg: -85,
            variable_deg: &IGP_X2,
            first_bit: 79,
            last_bit: 101,
        },
        IgpBandSegment {
            fixed_deg: -80,
            variable_deg: &IGP_X3,
            first_bit: 102,
            last_bit: 128,
        },
        IgpBandSegment {
            fixed_deg: -75,
            variable_deg: &IGP_X2,
            first_bit: 129,
            last_bit: 151,
        },
        IgpBandSegment {
            fixed_deg: -70,
            variable_deg: &IGP_X3,
            first_bit: 152,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: -65,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: -60,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: -55,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: -50,
            variable_deg: &IGP_X4,
            first_bit: 51,
            last_bit: 78,
        },
        IgpBandSegment {
            fixed_deg: -45,
            variable_deg: &IGP_X2,
            first_bit: 79,
            last_bit: 101,
        },
        IgpBandSegment {
            fixed_deg: -40,
            variable_deg: &IGP_X3,
            first_bit: 102,
            last_bit: 128,
        },
        IgpBandSegment {
            fixed_deg: -35,
            variable_deg: &IGP_X2,
            first_bit: 129,
            last_bit: 151,
        },
        IgpBandSegment {
            fixed_deg: -30,
            variable_deg: &IGP_X3,
            first_bit: 152,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: -25,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: -20,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: -15,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: -10,
            variable_deg: &IGP_X3,
            first_bit: 51,
            last_bit: 77,
        },
        IgpBandSegment {
            fixed_deg: -5,
            variable_deg: &IGP_X2,
            first_bit: 78,
            last_bit: 100,
        },
        IgpBandSegment {
            fixed_deg: 0,
            variable_deg: &IGP_X1,
            first_bit: 101,
            last_bit: 128,
        },
        IgpBandSegment {
            fixed_deg: 5,
            variable_deg: &IGP_X2,
            first_bit: 129,
            last_bit: 151,
        },
        IgpBandSegment {
            fixed_deg: 10,
            variable_deg: &IGP_X3,
            first_bit: 152,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: 15,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: 20,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: 25,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: 30,
            variable_deg: &IGP_X3,
            first_bit: 51,
            last_bit: 77,
        },
        IgpBandSegment {
            fixed_deg: 35,
            variable_deg: &IGP_X2,
            first_bit: 78,
            last_bit: 100,
        },
        IgpBandSegment {
            fixed_deg: 40,
            variable_deg: &IGP_X4,
            first_bit: 101,
            last_bit: 128,
        },
        IgpBandSegment {
            fixed_deg: 45,
            variable_deg: &IGP_X2,
            first_bit: 129,
            last_bit: 151,
        },
        IgpBandSegment {
            fixed_deg: 50,
            variable_deg: &IGP_X3,
            first_bit: 152,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: 55,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: 60,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: 65,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: 70,
            variable_deg: &IGP_X3,
            first_bit: 51,
            last_bit: 77,
        },
        IgpBandSegment {
            fixed_deg: 75,
            variable_deg: &IGP_X2,
            first_bit: 78,
            last_bit: 100,
        },
        IgpBandSegment {
            fixed_deg: 80,
            variable_deg: &IGP_X3,
            first_bit: 101,
            last_bit: 127,
        },
        IgpBandSegment {
            fixed_deg: 85,
            variable_deg: &IGP_X2,
            first_bit: 128,
            last_bit: 150,
        },
        IgpBandSegment {
            fixed_deg: 90,
            variable_deg: &IGP_X1,
            first_bit: 151,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: 95,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: 100,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: 105,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: 110,
            variable_deg: &IGP_X3,
            first_bit: 51,
            last_bit: 77,
        },
        IgpBandSegment {
            fixed_deg: 115,
            variable_deg: &IGP_X2,
            first_bit: 78,
            last_bit: 100,
        },
        IgpBandSegment {
            fixed_deg: 120,
            variable_deg: &IGP_X3,
            first_bit: 101,
            last_bit: 127,
        },
        IgpBandSegment {
            fixed_deg: 125,
            variable_deg: &IGP_X2,
            first_bit: 128,
            last_bit: 150,
        },
        IgpBandSegment {
            fixed_deg: 130,
            variable_deg: &IGP_X4,
            first_bit: 151,
            last_bit: 178,
        },
        IgpBandSegment {
            fixed_deg: 135,
            variable_deg: &IGP_X2,
            first_bit: 179,
            last_bit: 201,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: 140,
            variable_deg: &IGP_X3,
            first_bit: 1,
            last_bit: 27,
        },
        IgpBandSegment {
            fixed_deg: 145,
            variable_deg: &IGP_X2,
            first_bit: 28,
            last_bit: 50,
        },
        IgpBandSegment {
            fixed_deg: 150,
            variable_deg: &IGP_X3,
            first_bit: 51,
            last_bit: 77,
        },
        IgpBandSegment {
            fixed_deg: 155,
            variable_deg: &IGP_X2,
            first_bit: 78,
            last_bit: 100,
        },
        IgpBandSegment {
            fixed_deg: 160,
            variable_deg: &IGP_X3,
            first_bit: 101,
            last_bit: 127,
        },
        IgpBandSegment {
            fixed_deg: 165,
            variable_deg: &IGP_X2,
            first_bit: 128,
            last_bit: 150,
        },
        IgpBandSegment {
            fixed_deg: 170,
            variable_deg: &IGP_X3,
            first_bit: 151,
            last_bit: 177,
        },
        IgpBandSegment {
            fixed_deg: 175,
            variable_deg: &IGP_X2,
            first_bit: 178,
            last_bit: 200,
        },
    ],
];

const IGP_BANDS_POLAR: [[IgpBandSegment; 5]; 2] = [
    [
        IgpBandSegment {
            fixed_deg: 60,
            variable_deg: &IGP_X5,
            first_bit: 1,
            last_bit: 72,
        },
        IgpBandSegment {
            fixed_deg: 65,
            variable_deg: &IGP_X6,
            first_bit: 73,
            last_bit: 108,
        },
        IgpBandSegment {
            fixed_deg: 70,
            variable_deg: &IGP_X6,
            first_bit: 109,
            last_bit: 144,
        },
        IgpBandSegment {
            fixed_deg: 75,
            variable_deg: &IGP_X6,
            first_bit: 145,
            last_bit: 180,
        },
        IgpBandSegment {
            fixed_deg: 85,
            variable_deg: &IGP_X7,
            first_bit: 181,
            last_bit: 192,
        },
    ],
    [
        IgpBandSegment {
            fixed_deg: -60,
            variable_deg: &IGP_X5,
            first_bit: 1,
            last_bit: 72,
        },
        IgpBandSegment {
            fixed_deg: -65,
            variable_deg: &IGP_X6,
            first_bit: 73,
            last_bit: 108,
        },
        IgpBandSegment {
            fixed_deg: -70,
            variable_deg: &IGP_X6,
            first_bit: 109,
            last_bit: 144,
        },
        IgpBandSegment {
            fixed_deg: -75,
            variable_deg: &IGP_X6,
            first_bit: 145,
            last_bit: 180,
        },
        IgpBandSegment {
            fixed_deg: -85,
            variable_deg: &IGP_X8,
            first_bit: 181,
            last_bit: 192,
        },
    ],
];

fn igp_location(band_number: u8, position: usize) -> Option<(f64, f64)> {
    let one_based = position + 1;
    if band_number <= 8 {
        let segment = IGP_BANDS_LOW[usize::from(band_number)]
            .iter()
            .find(|segment| (segment.first_bit..=segment.last_bit).contains(&one_based))?;
        let lat = segment.variable_deg[one_based - segment.first_bit];
        return Some((f64::from(lat), f64::from(segment.fixed_deg)));
    }
    if (9..=10).contains(&band_number) {
        let segment = IGP_BANDS_POLAR[usize::from(band_number - 9)]
            .iter()
            .find(|segment| (segment.first_bit..=segment.last_bit).contains(&one_based))?;
        let lon = segment.variable_deg[one_based - segment.first_bit];
        return Some((f64::from(segment.fixed_deg), f64::from(lon)));
    }
    None
}

fn bracket_pair(values: &[f64], value: f64) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    values.windows(2).find_map(|pair| {
        let lo = pair[0];
        let hi = pair[1];
        (value >= lo && value <= hi).then_some((lo, hi))
    })
}

fn active_point(points: &[SbasIgp], lat_deg: f64, lon_deg: f64) -> Option<&SbasIgp> {
    points.iter().find(|p| {
        (p.lat_deg - lat_deg).abs() < SBAS_IGP_COORD_EPS_DEG
            && (normalize_lon(p.lon_deg) - lon_deg).abs() < SBAS_IGP_COORD_EPS_DEG
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct IgpInterpolation {
    vertical_delay_m: f64,
    give_variance_m2: Option<f64>,
}

fn bilinear_give_variance_m2(
    points: &[SbasIgp],
    lat0: f64,
    lat1: f64,
    lon0: f64,
    lon1: f64,
) -> Option<(f64, f64, f64, f64)> {
    let v00 = active_point(points, lat0, lon0)?.give_variance_m2?;
    let v01 = active_point(points, lat0, lon1)?.give_variance_m2?;
    let v10 = active_point(points, lat1, lon0)?.give_variance_m2?;
    let v11 = active_point(points, lat1, lon1)?.give_variance_m2?;
    Some((v00, v01, v10, v11))
}

fn plane_interpolate(points: &[SbasIgp], lat_deg: f64, lon_deg: f64) -> Option<f64> {
    plane_interpolate_value(points, lat_deg, lon_deg, |point| {
        Some(point.vertical_delay_m)
    })
}

fn plane_interpolate_give_variance(points: &[SbasIgp], lat_deg: f64, lon_deg: f64) -> Option<f64> {
    let variance_m2 =
        plane_interpolate_value(points, lat_deg, lon_deg, |point| point.give_variance_m2)?;
    (variance_m2.is_finite() && variance_m2 >= 0.0).then_some(variance_m2)
}

fn plane_interpolate_value(
    points: &[SbasIgp],
    lat_deg: f64,
    lon_deg: f64,
    value: impl Fn(&SbasIgp) -> Option<f64>,
) -> Option<f64> {
    let [p0, p1, p2] = points else {
        return None;
    };
    let x0 = p0.lon_deg;
    let y0 = p0.lat_deg;
    let z0 = value(p0)?;
    let x1 = p1.lon_deg;
    let y1 = p1.lat_deg;
    let z1 = value(p1)?;
    let x2 = p2.lon_deg;
    let y2 = p2.lat_deg;
    let z2 = value(p2)?;
    let det = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1);
    if det.abs() < 1.0e-12 {
        return None;
    }
    let a = (z0 * (y1 - y2) + z1 * (y2 - y0) + z2 * (y0 - y1)) / det;
    let b = (x0 * (z1 - z2) + x1 * (z2 - z0) + x2 * (z0 - z1)) / det;
    let c = (x0 * (y1 * z2 - y2 * z1) + x1 * (y2 * z0 - y0 * z2) + x2 * (y0 * z1 - y1 * z0)) / det;
    Some(a * lon_deg + b * lat_deg + c)
}

fn normalize_lon(mut lon_deg: f64) -> f64 {
    while lon_deg < -180.0 {
        lon_deg += 360.0;
    }
    while lon_deg > 180.0 {
        lon_deg -= 360.0;
    }
    lon_deg
}

fn f64_total_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.total_cmp(b)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::astro::time::model::TimeScale;
    use crate::sbas::message::{
        SbasFastCorrections, SbasGeoNav, SbasIgpDelay, SbasIgpMask, SbasIonoDelays,
        SbasLongTermCorrections, SbasMixedFastCorrections, SbasPrnMask, SpareBits,
    };

    fn epoch(tow_s: f64) -> GnssWeekTow {
        GnssWeekTow::new(TimeScale::Gpst, 2400, tow_s).expect("valid epoch")
    }

    fn geo() -> GnssSatelliteId {
        sbas_prn_to_sat(120).expect("valid SBAS PRN")
    }

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN")
    }

    fn mask_message() -> SbasMessage {
        let mut mask = [false; 210];
        mask[0] = true;
        mask[1] = true;
        SbasMessage::PrnMask(SbasPrnMask {
            preamble: 0x53,
            iodp: 1,
            mask,
            reserved: SpareBits::new(),
        })
    }

    #[test]
    fn sbas_prn_helpers_map_broadcast_to_slot_form() {
        assert_eq!(sbas_prn_to_sat(120), Some(geo()));
        assert_eq!(sat_to_sbas_prn(geo()), Some(120));
        assert_eq!(sbas_prn_to_sat(119), None);
    }

    /// Every broadcast PRN in the window round-trips to its slot and back, and
    /// the two values sitting just outside either end do not.
    #[test]
    fn sbas_prn_window_round_trips_and_stops_at_its_edges() {
        for broadcast_prn in 120..=158u16 {
            let sat = sbas_prn_to_sat(broadcast_prn)
                .unwrap_or_else(|| panic!("PRN {broadcast_prn} is inside the SBAS window"));
            assert_eq!(sat.system, GnssSystem::Sbas);
            assert_eq!(u16::from(sat.prn) + 100, broadcast_prn);
            assert_eq!(sat_to_sbas_prn(sat), Some(broadcast_prn));
        }
        for outside in [0u16, 1, 100, 119, 159, 160, 255, 355, u16::MAX] {
            assert_eq!(sbas_prn_to_sat(outside), None, "{outside}");
        }
    }

    /// The shared satellite-token range is 1..=99 for every constellation, so
    /// `S01`, `S19`, `S59` and `S99` are all now constructible identifiers. None
    /// of them names a broadcast PRN, and `sat_to_sbas_prn` must say so rather
    /// than return 101, 119, 159 or 199.
    #[test]
    fn sat_to_sbas_prn_rejects_slots_outside_the_broadcast_window() {
        for slot in [1u8, 2, 19, 59, 60, 99] {
            let sat = GnssSatelliteId::new(GnssSystem::Sbas, slot)
                .expect("the shared token range accepts S01..S99");
            assert_eq!(sat_to_sbas_prn(sat), None, "S{slot:02}");
        }
        for slot in 20..=58u8 {
            let sat = GnssSatelliteId::new(GnssSystem::Sbas, slot).expect("valid SBAS slot");
            assert_eq!(sat_to_sbas_prn(sat), Some(u16::from(slot) + 100));
        }
    }

    /// Other constellations never convert, however the id was built.
    #[test]
    fn sat_to_sbas_prn_rejects_other_constellations() {
        for system in [
            GnssSystem::Gps,
            GnssSystem::Glonass,
            GnssSystem::Galileo,
            GnssSystem::BeiDou,
            GnssSystem::Qzss,
            GnssSystem::Navic,
        ] {
            for prn in [1u8, 20, 30, 58, 99] {
                let sat = GnssSatelliteId::new(system, prn).expect("valid satellite token");
                assert_eq!(sat_to_sbas_prn(sat), None, "{system:?} {prn}");
            }
        }
    }

    /// `system` and `prn` are public and the derived `Deserialize` writes them
    /// straight through, so an id that never passed the constructor can reach
    /// this converter. Slot 0 would otherwise alias PRN 100, slot 100 would
    /// alias PRN 200 and slot 255 would alias PRN 355 - none of which exists.
    #[test]
    fn sat_to_sbas_prn_rejects_constructor_bypassing_struct_literals() {
        for prn in [0u8, 100, 255] {
            let bypass = GnssSatelliteId {
                system: GnssSystem::Sbas,
                prn,
            };
            assert!(
                GnssSatelliteId::new(GnssSystem::Sbas, prn).is_err(),
                "prn {prn} must not be constructible through new()"
            );
            assert_eq!(sat_to_sbas_prn(bypass), None, "prn {prn}");
        }
        // The same bypass on another constellation is refused by the system
        // check before the slot check ever runs.
        assert_eq!(
            sat_to_sbas_prn(GnssSatelliteId {
                system: GnssSystem::Gps,
                prn: 255,
            }),
            None
        );
    }

    /// The PRN mask layout is DO-229's: mask numbers 1..=37 are GPS, 38..=61
    /// GLONASS slot `n - 37` (R01..R24), 62..=119 future GNSS, 120..=158 SBAS.
    /// Position is `n - 1`. RTKLIB `decode_sbstype1` agrees only for 1..=61 and
    /// 120..=138; it reads 139..=182 as reserved and 183..=202 as QZSS. Widening
    /// the shared token range adds no position, so no extended slot such as
    /// `R28` has one, and position 63 is future GNSS, not `R27`.
    #[test]
    fn prn_mask_layout_follows_do_229() {
        let glonass =
            |slot| GnssSatelliteId::new(GnssSystem::Glonass, slot).expect("valid GLONASS slot");
        assert_eq!(mask_position_to_sat(0), Some(gps(1)));
        assert_eq!(mask_position_to_sat(31), Some(gps(32)));
        assert_eq!(mask_position_to_sat(32), Some(gps(33)));
        assert_eq!(mask_position_to_sat(36), Some(gps(37)));
        assert_eq!(mask_position_to_sat(37), Some(glonass(1)));
        assert_eq!(mask_position_to_sat(60), Some(glonass(24)));
        for position in [61, 62, 63, 64, 100, 118] {
            assert_eq!(mask_position_to_sat(position), None, "position {position}");
        }
        assert_eq!(mask_position_to_sat(119), Some(geo()));
        assert_eq!(
            mask_position_to_sat(157),
            Some(GnssSatelliteId::new(GnssSystem::Sbas, 58).expect("valid SBAS slot"))
        );
        for position in [158, 182, 192, 201, 209] {
            assert_eq!(mask_position_to_sat(position), None, "position {position}");
        }
    }

    /// An active mask bit that names no satellite keeps its place among the
    /// active bits. Corrections address satellites by that order, so dropping
    /// the bit would hand the correction after it to the wrong satellite.
    #[test]
    fn unassigned_mask_bits_keep_later_corrections_on_their_own_satellites() {
        let mut mask = [false; 210];
        mask[0] = true; // G01
        mask[70] = true; // mask number 71, future GNSS
        mask[119] = true; // SBAS PRN 120
        let mut store = SbasCorrectionStore::new();
        store
            .ingest(
                &SbasMessage::PrnMask(SbasPrnMask {
                    preamble: 0x53,
                    iodp: 1,
                    mask,
                    reserved: SpareBits::new(),
                }),
                geo(),
                epoch(10.0),
            )
            .unwrap();
        let mut prc = [0i16; 13];
        prc[0] = 8;
        prc[1] = 16;
        prc[2] = 24;
        store
            .ingest(
                &SbasMessage::FastCorrections(SbasFastCorrections {
                    preamble: 0x53,
                    message_type: 2,
                    iodf: 1,
                    iodp: 1,
                    prc,
                    udrei: [0u8; 13],
                    reserved: SpareBits::new(),
                }),
                geo(),
                epoch(20.0),
            )
            .unwrap();

        let g01 = store.fast(geo(), gps(1)).expect("G01 correction");
        assert_eq!(g01.prc_m.to_bits(), 1.0_f64.to_bits());
        let s20 = store.fast(geo(), geo()).expect("S20 correction");
        assert_eq!(
            s20.prc_m.to_bits(),
            3.0_f64.to_bits(),
            "the third active bit's correction reaches the third bit's satellite"
        );
        assert_eq!(
            store
                .unassigned_mask_corrections(geo())
                .and_then(|counts| counts.get(&71).copied()),
            Some(1),
            "the correction for mask number 71 is counted, not applied"
        );
    }

    /// A long-term record with mask index 0 is the fill of an unused record
    /// slot; it must not overwrite the first monitored satellite's correction.
    #[test]
    fn long_term_record_with_mask_index_zero_is_fill() {
        let mut store = SbasCorrectionStore::new();
        store.ingest(&mask_message(), geo(), epoch(10.0)).unwrap();
        let record = |monitored_index, delta_x| SbasLongTermRecord {
            monitored_index,
            iode: 7,
            delta_x,
            delta_y: 0,
            delta_z: 0,
            delta_x_rate: 0,
            delta_y_rate: 0,
            delta_z_rate: 0,
            delta_a_f0: 0,
            delta_a_f1: 0,
            time_of_day_s: None,
        };
        let half = |records| SbasLongTermHalf {
            velocity_code: false,
            iodp: 1,
            records,
            reserved: SpareBits(vec![(0, 1)]),
        };
        store
            .ingest(
                &SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                    preamble: 0x53,
                    halves: [
                        half(vec![record(1, 8), record(0, 0)]),
                        half(vec![record(0, 0), record(0, 0)]),
                    ],
                }),
                geo(),
                epoch(20.0),
            )
            .unwrap();
        let g01 = store
            .long_term(geo(), gps(1))
            .expect("G01 long-term correction");
        assert_eq!(g01.delta_ecef_m[0].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn fast_corrections_scale_and_derive_rrc_for_same_issue() {
        let mut store = SbasCorrectionStore::new();
        store.ingest(&mask_message(), geo(), epoch(10.0)).unwrap();
        let mut prc = [0i16; 13];
        let udrei = [0u8; 13];
        prc[0] = 8;
        let first = SbasMessage::FastCorrections(SbasFastCorrections {
            preamble: 0x53,
            message_type: 2,
            iodf: 1,
            iodp: 1,
            prc,
            udrei,
            reserved: SpareBits::new(),
        });
        store.ingest(&first, geo(), epoch(20.0)).unwrap();
        prc[0] = 16;
        let second = SbasMessage::FastCorrections(SbasFastCorrections {
            preamble: 0x9A,
            message_type: 2,
            iodf: 1,
            iodp: 1,
            prc,
            udrei,
            reserved: SpareBits::new(),
        });
        store.ingest(&second, geo(), epoch(28.0)).unwrap();
        let fast = store.fast(geo(), gps(1)).expect("fast correction");
        assert_eq!(fast.prc_m.to_bits(), 2.0_f64.to_bits());
        assert!((fast.rrc_m_s - 0.125).abs() < 1.0e-12);
    }

    #[test]
    fn udrei_withdraws_satellite() {
        let mut store = SbasCorrectionStore::new();
        store.ingest(&mask_message(), geo(), epoch(10.0)).unwrap();
        let prc = [0i16; 13];
        let mut udrei = [0u8; 13];
        udrei[0] = 15;
        let msg = SbasMessage::FastCorrections(SbasFastCorrections {
            preamble: 0x53,
            message_type: 2,
            iodf: 1,
            iodp: 1,
            prc,
            udrei,
            reserved: SpareBits::new(),
        });
        store.ingest(&msg, geo(), epoch(20.0)).unwrap();
        assert!(store.is_withdrawn(geo(), gps(1)));
    }

    #[test]
    fn igp_givei_15_is_not_added_to_grid() {
        let mut store = SbasCorrectionStore::new();
        let mut mask = [false; 201];
        mask[0] = true;
        mask[1] = true;
        let igp_mask = SbasMessage::IgpMask(SbasIgpMask {
            preamble: 0x53,
            band_number: 0,
            iodi: 2,
            mask,
            reserved: SpareBits(vec![(0, 4), (0, 1)]),
        });
        store.ingest(&igp_mask, geo(), epoch(0.0)).unwrap();
        let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
        entries[0] = SbasIgpDelay {
            vertical_delay: 8,
            givei: 0,
        };
        entries[1] = SbasIgpDelay {
            vertical_delay: 16,
            givei: 15,
        };
        let delays = SbasMessage::IonoDelays(SbasIonoDelays {
            preamble: 0x53,
            band_number: 0,
            block_id: 0,
            iodi: 2,
            entries,
            reserved: SpareBits(vec![(0, 7)]),
        });
        store.ingest(&delays, geo(), epoch(1.0)).unwrap();
        assert_eq!(store.iono_grid(geo()).unwrap().igps().len(), 1);
    }

    #[test]
    fn igp_locations_match_do229_fixture() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            band: u8,
            position: usize,
            lat_deg: Option<f64>,
            lon_deg: Option<f64>,
        }

        let fixtures: Vec<Fixture> =
            serde_json::from_str(include_str!("../../tests/fixtures/sbas_igp_locations.json"))
                .expect("valid IGP fixture");
        for fixture in fixtures {
            let got = igp_location(fixture.band, fixture.position);
            match (fixture.lat_deg, fixture.lon_deg) {
                (Some(lat), Some(lon)) => {
                    assert_eq!(got, Some((lat, lon)));
                }
                (None, None) => {
                    assert_eq!(got, None);
                }
                _ => panic!("fixture must contain both coordinates or neither"),
            }
        }
    }

    #[test]
    fn iono_grid_interpolates_four_points() {
        let grid = SbasIonoGrid::new(
            vec![
                SbasIgp {
                    lat_deg: 0.0,
                    lon_deg: 0.0,
                    vertical_delay_m: 1.0,
                    give_variance_m2: None,
                    t0_j2000_s: 0.0,
                },
                SbasIgp {
                    lat_deg: 0.0,
                    lon_deg: 5.0,
                    vertical_delay_m: 2.0,
                    give_variance_m2: None,
                    t0_j2000_s: 0.0,
                },
                SbasIgp {
                    lat_deg: 5.0,
                    lon_deg: 0.0,
                    vertical_delay_m: 3.0,
                    give_variance_m2: None,
                    t0_j2000_s: 0.0,
                },
                SbasIgp {
                    lat_deg: 5.0,
                    lon_deg: 5.0,
                    vertical_delay_m: 4.0,
                    give_variance_m2: None,
                    t0_j2000_s: 0.0,
                },
            ],
            0,
        );
        let vertical = grid.vertical_delay_at_ipp(2.5, 2.5).unwrap();
        assert_eq!(vertical.to_bits(), 2.5_f64.to_bits());
    }

    /// RTKLIB `sbsioncorr`: each covering point's GIVE variance times `9e-8 |t - t0|`,
    /// interpolated with the delay's weights and mapped by the square of the obliquity
    /// factor. At a zenith look the obliquity factor is 1, and the pierce point is the
    /// receiver's, the centre of the cell, where each point weighs a quarter.
    #[test]
    fn iono_grid_variance_grows_with_each_point_age_as_rtklib_states_it() {
        let point = |lat_deg: f64, lon_deg: f64, givei: u8, t0_j2000_s: f64| SbasIgp {
            lat_deg,
            lon_deg,
            vertical_delay_m: 1.0,
            give_variance_m2: give_variance_m2_for_givei(givei),
            t0_j2000_s,
        };
        let grid = SbasIonoGrid::new(
            vec![
                point(0.0, 0.0, 3, 1000.0),
                point(0.0, 5.0, 5, 1100.0),
                point(5.0, 0.0, 7, 1200.0),
                point(5.0, 5.0, 9, 1300.0),
            ],
            0,
        );
        let receiver = Wgs84Geodetic::new(2.5_f64.to_radians(), 2.5_f64.to_radians(), 0.0).unwrap();
        let t_j2000_s = 1400.0;
        let got = grid
            .slant_variance_m2(
                receiver,
                core::f64::consts::FRAC_PI_2,
                0.0,
                F_L1_HZ,
                t_j2000_s,
            )
            .expect("the grid covers the pierce point");
        let expected = [(3, 1000.0), (5, 1100.0), (7, 1200.0), (9, 1300.0)]
            .iter()
            .map(|&(givei, t0_j2000_s)| {
                0.25 * give_variance_m2_for_givei(givei).unwrap() * 9e-8 * (t_j2000_s - t0_j2000_s)
            })
            .sum::<f64>();
        assert!(
            (got - expected).abs() <= 1e-12 * expected,
            "variance {got} m², RTKLIB {expected} m²"
        );
        // Points just given their delays state no variance yet.
        let fresh = SbasIonoGrid::new(
            vec![
                point(0.0, 0.0, 3, t_j2000_s),
                point(0.0, 5.0, 5, t_j2000_s),
                point(5.0, 0.0, 7, t_j2000_s),
                point(5.0, 5.0, 9, t_j2000_s),
            ],
            0,
        )
        .slant_variance_m2(
            receiver,
            core::f64::consts::FRAC_PI_2,
            0.0,
            F_L1_HZ,
            t_j2000_s,
        )
        .expect("covered");
        assert_eq!(fresh, 0.0);
    }

    /// RTKLIB `decode_longcorr1` scales the velocity deltas by `P2_11` and the
    /// clock drift by `P2_39`, and places the 13-bit time of day in 16 s units
    /// on the message day.
    #[test]
    fn velocity_code_long_term_record_uses_rtklib_scales() {
        let mut store = SbasCorrectionStore::new();
        store.ingest(&mask_message(), geo(), epoch(10.0)).unwrap();
        let record = SbasLongTermRecord {
            monitored_index: 1,
            iode: 7,
            delta_x: 8,
            delta_y: -8,
            delta_z: 0,
            delta_x_rate: 1,
            delta_y_rate: -128,
            delta_z_rate: 127,
            delta_a_f0: 1,
            delta_a_f1: -1,
            time_of_day_s: Some(2),
        };
        let half = SbasLongTermHalf {
            velocity_code: true,
            iodp: 1,
            records: vec![record],
            reserved: SpareBits::new(),
        };
        store
            .ingest(
                &SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                    preamble: 0x53,
                    halves: [half.clone(), half],
                }),
                geo(),
                epoch(40.0),
            )
            .unwrap();
        let long = store.long_term(geo(), gps(1)).expect("G01 long-term");
        let p2_11 = 1.0 / 2048.0;
        assert_eq!(
            long.delta_ecef_rate_m_s.map(f64::to_bits),
            [p2_11, -128.0 * p2_11, 127.0 * p2_11].map(f64::to_bits)
        );
        assert_eq!(long.delta_ecef_m, [1.0, -1.0, 0.0]);
        assert_eq!(
            long.delta_af0_s.to_bits(),
            (1.0 / 2_147_483_648.0_f64).to_bits()
        );
        assert_eq!(
            long.delta_af1_s_s.to_bits(),
            (-1.0 / 549_755_813_888.0_f64).to_bits()
        );
        assert_eq!(long.t0_j2000_s, epoch_to_j2000_s(epoch(32.0)));
    }

    /// RTKLIB moves `t = tod - tow % 86400` into (-43200, 43200]: a time of
    /// day exactly half a day before the epoch is placed half a day after it.
    #[test]
    fn geo_time_of_day_half_a_day_away_falls_after_the_epoch() {
        let mut store = SbasCorrectionStore::new();
        let nav = |time_of_day_s| {
            SbasMessage::GeoNav(SbasGeoNav {
                preamble: 0x9A,
                time_of_day_s,
                ura: 0,
                x_m: 0,
                y_m: 0,
                z_m: 0,
                x_rate_m_s: 0,
                y_rate_m_s: 0,
                z_rate_m_s: 0,
                x_accel_m_s2: 0,
                y_accel_m_s2: 0,
                z_accel_m_s2: 0,
                a_gf0_s: 0,
                a_gf1_s_s: 0,
                reserved: SpareBits(vec![(0, 8)]),
            })
        };
        // Noon of the week's first day; time of day 0 is midnight before and
        // after, both half a day away.
        let noon = epoch(43_200.0);
        store.ingest(&nav(0), geo(), noon).unwrap();
        let t0 = store.geo_nav(geo()).expect("GEO nav").t0_j2000_s;
        assert_eq!(t0, epoch_to_j2000_s(noon) + 43_200.0);

        // 16 s units: 2700 * 16 = 43200, noon itself.
        store.ingest(&nav(2700), geo(), noon).unwrap();
        let t0 = store.geo_nav(geo()).expect("GEO nav").t0_j2000_s;
        assert_eq!(t0, epoch_to_j2000_s(noon));

        // RTKLIB rounds the time of week first: 11:59:59.6 is decided as
        // noon, so time of day 0 falls at the next midnight, exactly, where
        // the unrounded epoch would have kept it at the midnight before.
        let almost_noon = epoch(43_199.6);
        store.ingest(&nav(0), geo(), almost_noon).unwrap();
        let t0 = store.geo_nav(geo()).expect("GEO nav").t0_j2000_s;
        assert_eq!(t0, epoch_to_j2000_s(epoch(86_400.0)));

        // 23:59:44 read one second after midnight is the day before.
        let after_midnight = epoch(86_401.0);
        store.ingest(&nav(5399), geo(), after_midnight).unwrap();
        let t0 = store.geo_nav(geo()).expect("GEO nav").t0_j2000_s;
        assert_eq!(t0, epoch_to_j2000_s(after_midnight) - 17.0);
    }

    #[test]
    fn igp_givei_15_withdraws_an_earlier_delay_for_the_point() {
        let mut store = SbasCorrectionStore::new();
        let mut mask = [false; 201];
        mask[0] = true;
        mask[1] = true;
        store
            .ingest(
                &SbasMessage::IgpMask(SbasIgpMask {
                    preamble: 0x53,
                    band_number: 0,
                    iodi: 2,
                    mask,
                    reserved: SpareBits(vec![(0, 4), (0, 1)]),
                }),
                geo(),
                epoch(0.0),
            )
            .unwrap();
        let delays = |givei: u8| {
            let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
            entries[0] = SbasIgpDelay {
                vertical_delay: 8,
                givei: 0,
            };
            entries[1] = SbasIgpDelay {
                vertical_delay: 16,
                givei,
            };
            SbasMessage::IonoDelays(SbasIonoDelays {
                preamble: 0x53,
                band_number: 0,
                block_id: 0,
                iodi: 2,
                entries,
                reserved: SpareBits(vec![(0, 7)]),
            })
        };
        store.ingest(&delays(3), geo(), epoch(1.0)).unwrap();
        assert_eq!(store.iono_grid(geo()).unwrap().igps().len(), 2);
        store.ingest(&delays(15), geo(), epoch(2.0)).unwrap();
        let grid = store.iono_grid(geo()).unwrap();
        assert_eq!(grid.igps().len(), 1);
        assert_eq!(grid.igps()[0].vertical_delay_m, 1.0);
    }

    #[test]
    fn corrections_addressing_no_mask_block_are_refused() {
        let mut store = SbasCorrectionStore::new();
        store.ingest(&mask_message(), geo(), epoch(10.0)).unwrap();
        for message_type in [0, 1, 6, 63] {
            let fast = SbasMessage::FastCorrections(SbasFastCorrections {
                preamble: 0x53,
                message_type,
                iodf: 1,
                iodp: 1,
                prc: [8; 13],
                udrei: [0; 13],
                reserved: SpareBits::new(),
            });
            assert!(matches!(
                store.ingest(&fast, geo(), epoch(20.0)),
                Err(Error::SbasEncode(_))
            ));
        }
        assert!(store.fast(geo(), gps(1)).is_none());

        let mixed = SbasMessage::MixedCorrections(SbasMixedCorrections {
            preamble: 0x53,
            fast: SbasMixedFastCorrections {
                iodf: 1,
                iodp: 1,
                block_id: 4,
                prc: [8; 6],
                udrei: [0; 6],
                reserved: SpareBits(vec![(0, 4)]),
            },
            long_term: SbasLongTermHalf {
                velocity_code: false,
                iodp: 1,
                records: Vec::new(),
                reserved: SpareBits(vec![(0, 1)]),
            },
        });
        assert!(matches!(
            store.ingest(&mixed, geo(), epoch(20.0)),
            Err(Error::SbasEncode(_))
        ));
        assert!(store.fast(geo(), gps(1)).is_none());
    }

    /// DO-229 defines a vertical delay of 511 as "do not use": the point is
    /// not stored as 63.875 m, an earlier delay for it is withdrawn, the raw
    /// entry is kept with its reason, and a later usable entry restores it.
    #[test]
    fn igp_vertical_delay_511_is_do_not_use() {
        let mut store = SbasCorrectionStore::new();
        let mut mask = [false; 201];
        mask[0] = true;
        mask[1] = true;
        store
            .ingest(
                &SbasMessage::IgpMask(SbasIgpMask {
                    preamble: 0x53,
                    band_number: 0,
                    iodi: 2,
                    mask,
                    reserved: SpareBits(vec![(0, 4), (0, 1)]),
                }),
                geo(),
                epoch(0.0),
            )
            .unwrap();
        let delays = |second: SbasIgpDelay| {
            let mut entries: [SbasIgpDelay; 15] = core::array::from_fn(|_| SbasIgpDelay::default());
            entries[0] = SbasIgpDelay {
                vertical_delay: 8,
                givei: 0,
            };
            entries[1] = second;
            SbasMessage::IonoDelays(SbasIonoDelays {
                preamble: 0x53,
                band_number: 0,
                block_id: 0,
                iodi: 2,
                entries,
                reserved: SpareBits(vec![(0, 7)]),
            })
        };
        let usable = SbasIgpDelay {
            vertical_delay: 16,
            givei: 3,
        };
        store
            .ingest(&delays(usable.clone()), geo(), epoch(1.0))
            .unwrap();
        assert_eq!(store.iono_grid(geo()).unwrap().igps().len(), 2);

        for (entry, reason) in [
            (
                SbasIgpDelay {
                    vertical_delay: 511,
                    givei: 3,
                },
                SbasIgpUnavailableReason::DoNotUse,
            ),
            (
                SbasIgpDelay {
                    vertical_delay: 511,
                    givei: 15,
                },
                SbasIgpUnavailableReason::DoNotUse,
            ),
            (
                SbasIgpDelay {
                    vertical_delay: 16,
                    givei: 15,
                },
                SbasIgpUnavailableReason::NotMonitored,
            ),
        ] {
            store
                .ingest(&delays(usable.clone()), geo(), epoch(1.0))
                .unwrap();
            store
                .ingest(&delays(entry.clone()), geo(), epoch(2.0))
                .unwrap();
            let grid = store.iono_grid(geo()).unwrap();
            assert_eq!(grid.igps().len(), 1);
            assert_eq!(grid.igps()[0].vertical_delay_m, 1.0);
            assert!(grid.igps().iter().all(|p| p.vertical_delay_m != 63.875));
            let unavailable = grid.unavailable_igps();
            assert_eq!(unavailable.len(), 1);
            assert_eq!(unavailable[0].vertical_delay, entry.vertical_delay);
            assert_eq!(unavailable[0].givei, entry.givei);
            assert_eq!(unavailable[0].reason, reason);
        }

        store.ingest(&delays(usable), geo(), epoch(3.0)).unwrap();
        let grid = store.iono_grid(geo()).unwrap();
        assert_eq!(grid.igps().len(), 2);
        assert!(grid.unavailable_igps().is_empty());
    }

    /// Ingest applies the encoder's validation, so a hand-built long-term
    /// half the wire cannot carry changes nothing: no rate, clock drift or
    /// time of day from a record without the velocity code, no invented
    /// reference time for a velocity record without one, and no record count
    /// the velocity code does not carry.
    #[test]
    fn long_term_halves_the_wire_cannot_carry_are_refused() {
        let record = |monitored_index| SbasLongTermRecord {
            monitored_index,
            iode: 7,
            delta_x: 8,
            delta_y: 0,
            delta_z: 0,
            delta_x_rate: 0,
            delta_y_rate: 0,
            delta_z_rate: 0,
            delta_a_f0: 0,
            delta_a_f1: 0,
            time_of_day_s: None,
        };
        let half = |velocity_code, records| SbasLongTermHalf {
            velocity_code,
            iodp: 1,
            records,
            reserved: if velocity_code {
                SpareBits::new()
            } else {
                SpareBits(vec![(0, 1)])
            },
        };
        let good = || half(false, vec![record(0), record(0)]);

        let mut with_rate = record(1);
        with_rate.delta_x_rate = 4;
        let mut with_drift = record(1);
        with_drift.delta_a_f1 = 1;
        let mut with_time = record(1);
        with_time.time_of_day_s = Some(2);
        let refused = [
            half(false, vec![with_rate, record(0)]),
            half(false, vec![with_drift, record(0)]),
            half(false, vec![with_time, record(0)]),
            half(true, vec![record(1)]),
            half(false, vec![record(1)]),
            half(false, vec![record(1), record(0), record(0)]),
            half(true, Vec::new()),
        ];
        for bad in refused {
            let mut store = SbasCorrectionStore::new();
            store.ingest(&mask_message(), geo(), epoch(10.0)).unwrap();
            let message = SbasMessage::LongTermCorrections(SbasLongTermCorrections {
                preamble: 0x53,
                halves: [bad.clone(), good()],
            });
            assert!(
                matches!(
                    store.ingest(&message, geo(), epoch(20.0)),
                    Err(Error::SbasEncode(_))
                ),
                "expected {bad:?} to be refused"
            );
            assert!(store.long_term(geo(), gps(1)).is_none());
        }
    }
}
