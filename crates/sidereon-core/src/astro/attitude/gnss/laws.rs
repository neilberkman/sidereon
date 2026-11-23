//! The block attitude laws.

use super::geometry::{acos_clamped, beta_at, compute_phi, sample_at, wrap_pi, Context};
use super::search::{
    angle_to_centre, exceeds, golden_section, latest_within, passed, HalfOrbit, Search,
};
use super::{
    AttitudeMode, AttitudeTrajectory, CosineTurnParameters, CosineTurnStartAngle,
    GnssAttitudeError, GnssAttitudeLaw,
};
use crate::astro::math::vec3::{cross3, dot3, norm3, scale3};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Longest time after its zone ends that a GPS turn may continue while the
/// yaw catches up with the nominal yaw (Kouba's `eclips.f` `HALFHR` and Orekit
/// `END_MARGIN`, 1800 s).
const GPS_TURN_END_MARGIN_S: f64 = 1800.0;

/// Outcome of a law at one instant.
pub(super) enum Outcome {
    /// Nominal yaw steering.
    Nominal,
    /// A yaw angle from a turn, shadow, orbit-normal or modified law.
    Yaw { mode: AttitudeMode, yaw_rad: f64 },
}

pub(super) fn evaluate<T: AttitudeTrajectory + ?Sized>(
    law: &GnssAttitudeLaw,
    ctx: &Context,
    trajectory: &T,
) -> Result<Outcome, GnssAttitudeError> {
    match *law {
        GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        } => gps_linear(
            GpsKind::Iia {
                yaw_bias_rad,
                shadow_limit_rad,
            },
            yaw_rate_rad_s,
            ctx,
            trajectory,
        ),
        GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s } => {
            gps_linear(GpsKind::Iir, yaw_rate_rad_s, ctx, trajectory)
        }
        GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        } => gps_linear(
            GpsKind::Iif {
                yaw_bias_rad,
                shadow_limit_rad,
            },
            yaw_rate_rad_s,
            ctx,
            trajectory,
        ),
        GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s,
            shadow_limit_rad,
        } => glonass_m(yaw_rate_rad_s, shadow_limit_rad, ctx, trajectory),
        GnssAttitudeLaw::GalileoIov {
            beta_x_rad,
            beta_y_rad,
        } => galileo_iov(beta_x_rad, beta_y_rad, ctx, trajectory),
        GnssAttitudeLaw::CosineTurn(turn) => cosine_turn(&turn, ctx, trajectory),
        GnssAttitudeLaw::BeidouOrbitNormalSwitch {
            ys_to_on_beta_rad,
            on_to_ys_beta_rad,
        } => beidou_orbit_normal_switch(ys_to_on_beta_rad, on_to_ys_beta_rad, ctx, trajectory),
        GnssAttitudeLaw::FixedBetaYawSteering { beta0_rad } => Ok(fixed_beta(beta0_rad, ctx)),
        GnssAttitudeLaw::Beidou3MeoUnknownManufacturer {
            cast,
            secm_beta0_rad,
        } => {
            let cast = cosine_turn(&cast, ctx, trajectory)?;
            let secm = fixed_beta(secm_beta0_rad, ctx);
            match (cast, secm) {
                (Outcome::Nominal, Outcome::Nominal) => Ok(Outcome::Nominal),
                _ => Err(GnssAttitudeError::BeidouManufacturerRequired { t_s: ctx.t_s }),
            }
        }
        GnssAttitudeLaw::OrbitNormal => Ok(orbit_normal()),
    }
}

fn orbit_normal() -> Outcome {
    Outcome::Yaw {
        mode: AttitudeMode::OrbitNormal,
        yaw_rad: 0.0,
    }
}

/// The part of a half orbit where a GPS or GLONASS-M law leaves nominal yaw
/// steering, as the papers define it.
#[derive(Clone, Copy, Debug)]
enum Zone {
    /// Kouba (2009): the nominal yaw rate
    /// `mu_rate tan(beta) cos(mu) / (sin^2(mu) + tan^2(beta))` exceeds the
    /// maximum yaw rate `yaw_rate` in magnitude. After a GPS turn's zone the
    /// same condition with the turn's speed marks where the nominal yaw moves
    /// faster than the turn ([`after_zone`]).
    NominalRateAbove { yaw_rate: f64 },
    /// The Sun-satellite angle exceeds the shadow limit (the Earth's shadow).
    Shadow { limit: f64 },
    /// Dilssner et al. (2011): the in-plane angle to noon is within the half
    /// span over which a turn at the maximum rate links the nominal yaws on
    /// either side of noon (see [`glonass_noon_yaw_end`]).
    GlonassNoon { yaw_rate: f64 },
}

impl Zone {
    /// A margin positive exactly inside the zone at the instant of `ctx`, in
    /// the half orbit around noon (`noon == true`) or midnight, and larger
    /// where the zone's condition holds by more: the logarithm of Kouba's
    /// nominal yaw rate over the limit rate, `cos(limit)` minus the cosine of
    /// the Sun-satellite angle, or the logarithm of the GLONASS-M ratio of
    /// [`glonass_noon_log_ratio`].
    fn margin(self, ctx: &Context, noon: bool) -> Result<f64, GnssAttitudeError> {
        Ok(match self {
            Zone::NominalRateAbove { yaw_rate } => kouba_rate_log_ratio(ctx, noon, yaw_rate),
            Zone::Shadow { limit } => libm::cos(limit) - ctx.svb_cos,
            Zone::GlonassNoon { yaw_rate } => glonass_noon_log_ratio(ctx, yaw_rate)?,
        })
    }

    /// Whether the instant of `ctx` lies inside the zone of the half orbit
    /// around noon (`noon == true`) or midnight.
    fn contains(self, ctx: &Context, noon: bool) -> Result<bool, GnssAttitudeError> {
        Ok(self.margin(ctx, noon)? > 0.0)
    }

    /// Whether a nominal-rate zone holds at the exact centre at the geometry
    /// of `centre` (an instant next to it): Kouba's rate there,
    /// `mu_rate / tan|beta|`, exceeds the limit rate, that is
    /// `|beta| < atan(mu_rate / R)`.
    fn holds_at_centre(self, centre: &Context) -> bool {
        match self {
            Zone::NominalRateAbove { yaw_rate } => {
                centre.beta.abs() < libm::atan(centre.mu_rate / yaw_rate)
            }
            Zone::Shadow { .. } | Zone::GlonassNoon { .. } => false,
        }
    }

    /// The in-plane angle from the centre at which the zone's condition
    /// changes, at the beta and orbital rate of `ctx`: Kouba's nominal-rate
    /// half span, the shadow's in-plane half span
    /// `acos(cos(limit - pi) / cos(beta))` (Orekit's
    /// `inOrbitPlaneAbsoluteAngle`), or the Dilssner half span. The start and
    /// end yaws of a turn are the nominal yaws there, also where the zone is
    /// narrower than the resolution of the trajectory or of `t`.
    fn edge_angle(self, ctx: &Context) -> Result<f64, GnssAttitudeError> {
        Ok(match self {
            Zone::NominalRateAbove { yaw_rate } => {
                nominal_rate_half_span(ctx.beta, ctx.mu_rate, yaw_rate)
            }
            Zone::Shadow { limit } => ctx.in_orbit_plane_absolute_angle(limit - PI),
            Zone::GlonassNoon { yaw_rate } => {
                let k = glonass_rate_ratio(ctx, yaw_rate)?;
                k * glonass_noon_yaw_end(ctx.beta, k)
            }
        })
    }
}

/// The sign of Kouba's nominal yaw rate in the half orbit around noon
/// (`noon == true`), `-sign(beta)`, or midnight, `+sign(beta)`; 0 at beta = 0.
fn kouba_rate_sign(beta: f64, noon: bool) -> f64 {
    let moving = if noon { -beta } else { beta };
    if moving == 0.0 {
        0.0
    } else {
        moving.signum()
    }
}

/// `ln(|psi_dot| / speed)` with Kouba's (2009) nominal yaw rate at the instant
/// of `ctx`, `|psi_dot| = mu_rate T cos(x) / (sin^2(x) + T^2)`,
/// `T = |tan(beta)|` and `x` the in-plane angle to noon (`noon == true`) or
/// midnight; at the exact centre (`sin(x) = 0`) `mu_rate / T`, `+inf` at
/// beta = 0, and elsewhere `-inf` at beta = 0 and outside the half orbit
/// (`cos(x) <= 0`). As a sum of logarithms no product or quotient overflows or
/// underflows; a sum of squares that underflows gives `+inf`, the limit.
fn kouba_rate_log_ratio(ctx: &Context, noon: bool, speed: f64) -> f64 {
    let x = angle_to_centre(ctx, noon);
    let t = libm::tan(ctx.beta).abs();
    let c = libm::cos(x);
    if c <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let s = libm::sin(x);
    if s == 0.0 {
        return libm::log(ctx.mu_rate) - libm::log(t) - libm::log(speed);
    }
    if t == 0.0 {
        return f64::NEG_INFINITY;
    }
    libm::log(t) + libm::log(c) - libm::log(s * s + t * t) + libm::log(ctx.mu_rate)
        - libm::log(speed)
}

/// The nominal yaw's half swing over the in-plane angles `[-d, d]` around
/// noon, `g(d) = |phi(beta, d) - phi(beta, -d)| / 2` for `d` in `[0, pi]`:
/// with `T = |tan(beta)|`, `phi(beta, +-d) = -+(pi / 2 - atan(sin(d) / T))`
/// up to the sign of beta, so `g(d) = atan2(sin(d), T)`, formed without the
/// cancellation of two angles near `-pi / 2` (which a few 1e-16 rad from noon
/// leaves only rounding).
fn glonass_half_swing(beta: f64, d: f64) -> f64 {
    libm::atan2(libm::sin(d), libm::tan(beta).abs())
}

/// `ln(k g(d) / d)` for the GLONASS-M noon zone (Dilssner et al. 2011), with
/// `k = mu_rate / R`, `d` the in-plane angle to noon in magnitude and `g` the
/// nominal yaw's half swing ([`glonass_half_swing`]). `g(d) / d` falls from
/// `1 / tan|beta|` at `d = 0` (`g` is concave up to `pi / 2` and falls after
/// it), so this is positive exactly for `d` below the half span `k y`,
/// `y = g(k y)` (see [`glonass_noon_yaw_end`]), and at `d = 0` exactly for
/// `|beta| < atan(k)`.
fn glonass_noon_log_ratio(ctx: &Context, yaw_rate: f64) -> Result<f64, GnssAttitudeError> {
    let k = glonass_rate_ratio(ctx, yaw_rate)?;
    let d = angle_to_centre(ctx, true).abs();
    Ok(if d == 0.0 {
        libm::log(k) - libm::log(libm::tan(ctx.beta).abs())
    } else {
        libm::log(k) + libm::log(glonass_half_swing(ctx.beta, d)) - libm::log(d)
    })
}

/// `mu_rate / R` for the GLONASS-M noon turn; `InvalidParameter` for a
/// maximum yaw rate so small that it overflows.
fn glonass_rate_ratio(ctx: &Context, yaw_rate: f64) -> Result<f64, GnssAttitudeError> {
    let k = ctx.mu_rate / yaw_rate;
    if k.is_finite() {
        Ok(k)
    } else {
        Err(GnssAttitudeError::InvalidParameter {
            field: "yaw_rate_rad_s",
            value: yaw_rate,
        })
    }
}

/// In-plane half span of the zone where the nominal yaw rate exceeds the
/// maximum rate `R` (Kouba 2009) at Sun elevation `beta`: the `x` in
/// `[0, pi / 2]` with `T cos(x) = (sin^2(x) + T^2) k`, `T = |tan(beta)|`,
/// `k = R / mu_rate`, `T k < 1`. Squared, `sigma = sin^2(x)` solves
/// `k^2 sigma^2 + T^2 (1 + 2 k^2) sigma + T^2 (k^2 T^2 - 1) = 0`; its positive
/// root, written without cancellation and divided through by `T^2`, is
/// `2 (1 - k T) (1 + k T) / (1 + 2 k^2 + sqrt(1 + 4 k^2 + 4 (k / T)^2))`,
/// about `T / k` for small T, 0 at `beta = 0`, and 1 as `k` tends to 0. Every
/// term of the denominator is positive, so a ratio that overflows makes the
/// half span 0 (the true value is then below the smallest normal number) and
/// a ratio that underflows makes it `pi / 2`; `k T >= 1` has no zone.
fn nominal_rate_half_span(beta: f64, mu_rate: f64, yaw_rate: f64) -> f64 {
    let t = libm::tan(beta).abs();
    let k = yaw_rate / mu_rate;
    let kt = k * t;
    // `kt` is NaN only for an infinite `k` at `t = 0`.
    if t == 0.0 || kt.is_nan() || kt >= 1.0 {
        return 0.0;
    }
    let k_over_t = k / t;
    let sigma = 2.0 * (1.0 - kt) * (1.0 + kt)
        / (1.0 + 2.0 * k * k + libm::sqrt(1.0 + 4.0 * k * k + 4.0 * k_over_t * k_over_t));
    libm::asin(libm::sqrt(sigma.clamp(0.0, 1.0)))
}

/// Where a zone lies in its half orbit.
#[derive(Clone, Copy, Debug)]
enum Span {
    /// The zone holds at no instant of the half orbit the search resolves
    /// (and a nominal-rate zone not at the exact centre either).
    None,
    /// A nominal-rate zone that holds at the exact centre but at no
    /// representable instant: narrower than the resolution of `t`.
    AtCentre,
    /// The zone's first and last instants, both inside the half orbit.
    Within(f64, f64),
    /// The zone holds at the half orbit's first instant and ends at this one.
    FromStart(f64),
    /// The zone starts at this instant and holds at the half orbit's last.
    ToEnd(f64),
    /// The zone holds at both ends of the half orbit.
    Whole,
}

/// `|beta|` at or below which its sign is not taken: beta is formed as
/// `pi / 2` minus an angle near `pi / 2`, with a rounding of a few 1e-16 rad,
/// so below this its sign can be the rounding's.
const BETA_SIGN_FLOOR: f64 = 4.0 * f64::EPSILON;

/// The sign of beta, 0 within [`BETA_SIGN_FLOOR`] of zero.
fn beta_sign(beta: f64) -> f64 {
    if beta > BETA_SIGN_FLOOR {
        1.0
    } else if beta < -BETA_SIGN_FLOOR {
        -1.0
    } else {
        0.0
    }
}

/// A part of a half orbit on which beta keeps one sign: its first and last
/// instants and beta's sign there (0 where beta stays within
/// [`BETA_SIGN_FLOOR`] of zero).
#[derive(Clone, Copy, Debug)]
struct BetaPart {
    lo: f64,
    hi: f64,
    sign: f64,
}

/// Intervals of the time grid on which [`beta_samples`] takes beta over a
/// half orbit.
const BETA_SAMPLE_INTERVALS: u32 = 16;

/// Beta over a half orbit, in time order: at its start, its centre, its last
/// instant and the instants dividing `[start, last]` into
/// [`BETA_SAMPLE_INTERVALS`] equal intervals.
fn beta_samples<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
) -> Result<[(f64, f64); BETA_SAMPLE_INTERVALS as usize + 2], GnssAttitudeError> {
    let last = half.last();
    let mut samples = [(0.0, 0.0); BETA_SAMPLE_INTERVALS as usize + 2];
    for (k, slot) in (0..=BETA_SAMPLE_INTERVALS).zip(samples.iter_mut()) {
        let t = if k == BETA_SAMPLE_INTERVALS {
            last
        } else {
            half.start + (last - half.start) * (f64::from(k) / f64::from(BETA_SAMPLE_INTERVALS))
        };
        *slot = (t, search.at(t)?.beta);
    }
    samples[BETA_SAMPLE_INTERVALS as usize + 1] = (half.centre, search.at(half.centre)?.beta);
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(samples)
}

/// The parts of a half orbit on which beta keeps one sign, in order: the
/// whole half orbit, or the two parts either side of beta's change of sign,
/// the last instant with the sign beta has at the start and the first
/// without it, bisected between the two adjacent samples of
/// [`beta_samples`] whose betas have opposite signs. Beta changes
/// monotonically within a half orbit (the trajectory contract), so it changes
/// sign at most once there; samples whose signs change more than once are
/// [`GnssAttitudeError::OutsideValidatedDomain`].
///
/// Where beta changes sign, the turn speed `speed` must be at least the
/// orbital rate at the change (at both instants bracketing it), otherwise
/// the half orbit is outside the validated domain. This is a rule of the
/// laws, not of the papers: a zone found on both parts is taken as one turn
/// ([`zone_span`]), and at such speeds, with beta moving by at most 0.05 rad
/// over an orbit, both parts hold a zone only where beta changes sign within
/// 0.002 rad of orbit angle of the centre, and the two zones then lie within
/// 0.0097 rad of it and less than 0.0024 rad apart (on a circular orbit).
fn beta_parts<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
    speed: f64,
) -> Result<[Option<BetaPart>; 2], GnssAttitudeError> {
    let last = half.last();
    let mut first_sign = 0.0;
    let mut before: Option<(f64, f64)> = None;
    let mut change = None;
    for (t, beta) in beta_samples(search, half)? {
        let sign = beta_sign(beta);
        if sign == 0.0 {
            continue;
        }
        match before {
            None => first_sign = sign,
            Some((previous, previous_sign)) if previous_sign != sign => {
                if change.is_some() {
                    return Err(
                        search.outside("beta changes sign more than once within a half orbit")
                    );
                }
                change = Some((previous, t, previous_sign));
            }
            Some(_) => {}
        }
        before = Some((t, sign));
    }
    let Some((lo, hi, sign)) = change else {
        return Ok([
            Some(BetaPart {
                lo: half.start,
                hi: last,
                sign: first_sign,
            }),
            None,
        ]);
    };
    let keeps = |x: f64| -> Result<bool, GnssAttitudeError> { Ok(search.at(x)?.beta * sign > 0.0) };
    let (p, q) = search.transition(&keeps, lo, hi)?;
    let rate_there = search.at(p)?.mu_rate.max(search.at(q)?.mu_rate);
    if speed < rate_there {
        return Err(search.outside(
            "the maximum yaw rate is below the orbital rate where beta changes sign within the half orbit",
        ));
    }
    Ok([
        Some(BetaPart {
            lo: half.start,
            hi: p,
            sign,
        }),
        Some(BetaPart {
            lo: q,
            hi: last,
            sign: -sign,
        }),
    ])
}

/// Largest in-plane half span, from the centre, of Kouba's zone at `|beta|`
/// of twice [`BETA_SIGN_FLOOR`] for which a half orbit whose beta stays
/// within the floor is searched: within the floor beta's computed value is
/// rounding (it rounds to 0 within about 1e-16 rad of zero), so the zone
/// there depends on beta below its resolution, by no more than this angle of
/// orbit. It holds for maximum yaw rates above about 1.8e-3 times the
/// orbital rate.
const UNSIGNED_BETA_SPAN_MAX: f64 = 1e-6;

/// [`GnssAttitudeError::OutsideValidatedDomain`] for a part of a half orbit
/// where beta's sign is not taken and Kouba's zone at the turn speed `speed`
/// could reach beyond [`UNSIGNED_BETA_SPAN_MAX`] from the centre.
fn check_unsigned_beta_zone<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
    speed: f64,
) -> Result<(), GnssAttitudeError> {
    let mu_rate = search.at(half.centre)?.mu_rate;
    if nominal_rate_half_span(2.0 * BETA_SIGN_FLOOR, mu_rate, speed) <= UNSIGNED_BETA_SPAN_MAX {
        Ok(())
    } else {
        Err(search.outside(
            "beta within the rounding of zero and a maximum yaw rate below 1.8e-3 times the orbital rate",
        ))
    }
}

/// The first and last instants of `[lo, hi]` inside a zone with `margin`,
/// for a margin that rises to one maximum on `[lo, hi]` and falls after it:
/// an instant inside from [`Search::find_inside`] (from `first`), and the
/// changes of the margin's sign between it and the ends, each bisected in a
/// checked bracket. `None` where the search finds no instant inside.
fn zone_on<T, M>(
    search: &Search<'_, T>,
    lo: f64,
    hi: f64,
    first: f64,
    margin: &M,
) -> Result<Option<(f64, f64)>, GnssAttitudeError>
where
    T: AttitudeTrajectory + ?Sized,
    M: Fn(&Context) -> Result<f64, GnssAttitudeError>,
{
    let Some(probe) = search.find_inside(lo, hi, first, margin)? else {
        return Ok(None);
    };
    let inside = |t: f64| -> Result<bool, GnssAttitudeError> { Ok(margin(&search.at(t)?)? > 0.0) };
    let start = if inside(lo)? {
        lo
    } else {
        search.transition(&inside, lo, probe)?.1
    };
    let end = if inside(hi)? {
        hi
    } else {
        search.transition(&inside, probe, hi)?.0
    };
    Ok(Some((start, end)))
}

/// Largest orbit eccentricity on which the GPS turn zones are found: Kouba's
/// nominal yaw rate has one maximum per half orbit (and per part of one sign
/// of beta) on every orbit up to it and 3e-4 past it, from a scan of
/// eccentricity, argument of perigee, beta and its rate; it has two from
/// about 0.32.
const KOUBA_MAX_ECCENTRICITY: f64 = 0.3;

/// Largest orbit eccentricity on which the laws search the trajectory: every
/// step of the searches lasts less than an orbital period there (see the
/// search module).
const STEPPING_MAX_ECCENTRICITY: f64 = 0.9;

/// The GLONASS-M noon zone's bounds: the largest orbit eccentricity on
/// which it is found where the largest `|beta|` over the noon half orbit
/// ([`beta_samples`]) is below each limit, and
/// [`GLONASS_NOON_MAX_ECCENTRICITY`] for every beta. The Dilssner margin has
/// one maximum per half orbit on each and 3e-4 past it, from a scan of
/// eccentricity, argument of perigee (every degree at the corners of the
/// bounds), beta and its rate; with beta fixed it has two from `|beta|` of
/// 80 deg at eccentricity 0.15, 63.5 deg at 0.2 and 34.9 deg at 0.3, and at
/// 0.3 two from `|beta|` of 2.5 deg with beta moving by 0.1 rad per orbit.
const GLONASS_NOON_BOUNDS: [(f64, f64); 2] =
    [(30.0 * (PI / 180.0), 0.25), (60.0 * (PI / 180.0), 0.2)];
const GLONASS_NOON_MAX_ECCENTRICITY: f64 = 0.1;

/// Largest change of beta, and largest angle the orbit plane turns, over the
/// turn of orbit angle before an instant (one orbital period), on which the
/// searches are shown correct. The Sun's motion changes beta by at most about
/// 0.02 rad over a geostationary orbit.
const PLANE_MOTION_MAX: f64 = 0.05;

/// Allowance on the eccentricity bounds for the conic fit, which a plane
/// turning by up to [`PLANE_MOTION_MAX`] over the half turn it spans biases
/// by up to about 1e-4 (about 0.04 times the square of the turn).
const ECCENTRICITY_ALLOWANCE: f64 = 2e-4;

/// [`GnssAttitudeError::OutsideValidatedDomain`] with `reason` for an
/// eccentricity above `bound` (and its allowance).
fn check_eccentricity<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    eccentricity: f64,
    bound: f64,
    reason: &'static str,
) -> Result<(), GnssAttitudeError> {
    if eccentricity <= bound + ECCENTRICITY_ALLOWANCE {
        Ok(())
    } else {
        Err(search.outside(reason))
    }
}

/// The orbit's eccentricity ([`Search::orbit_shape`]), checked against
/// `bound` with `reason`, after checking that beta and the orbit plane turn
/// by at most [`PLANE_MOTION_MAX`] over the turn before the instant.
fn checked_eccentricity<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    ctx: &Context,
    bound: f64,
    reason: &'static str,
) -> Result<f64, GnssAttitudeError> {
    let shape = search.orbit_shape(ctx)?;
    if !(shape.beta_change <= PLANE_MOTION_MAX && shape.plane_turn <= PLANE_MOTION_MAX) {
        return Err(search.outside(
            "beta or the orbit plane turns by more than 0.05 rad over the orbit before the instant",
        ));
    }
    check_eccentricity(search, shape.eccentricity, bound, reason)?;
    Ok(shape.eccentricity)
}

/// Margin on the orbit's largest orbital rate in [`could_hold`]: twice the
/// largest rate of the conic through three of the satellite's positions
/// ([`Search::peak_orbital_rate`]), for an orbit that is not exactly a conic.
const PEAK_RATE_MARGIN: f64 = 2.0;

/// Allowance on the shadow's beta limit in [`could_hold`] for the rounding of
/// the Sun-satellite angle.
const SHADOW_BETA_ALLOWANCE: f64 = 1e-9;

/// The least `|beta|` over a span on which beta changes monotonically
/// between consecutive values of `betas` (taken in time order, either way,
/// at instants that divide the span where the trajectory contract has beta
/// monotone): 0 where two consecutive values are not of one sign, otherwise
/// the least magnitude.
fn least_abs_beta(betas: &[f64]) -> f64 {
    let mut least = f64::INFINITY;
    for (k, &beta) in betas.iter().enumerate() {
        if k > 0 && betas[k - 1] * beta <= 0.0 {
            return 0.0;
        }
        least = least.min(beta.abs());
    }
    least
}

/// The least `|beta|` over a half orbit: [`least_abs_beta`] of beta at its
/// start, centre and last instant.
fn half_orbit_least_beta<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
) -> Result<f64, GnssAttitudeError> {
    Ok(least_abs_beta(&[
        search.at(half.start)?.beta,
        search.at(half.centre)?.beta,
        search.at(half.last())?.beta,
    ]))
}

/// The orbit's largest orbital rate for [`could_hold`]: that of the conic
/// through the satellite's positions at the quadratures and centre of `half`
/// ([`Search::orbit_peak`]), infinite where the conic's eccentricity is above
/// [`STEPPING_MAX_ECCENTRICITY`] (and its allowance), where the steps that
/// found the half orbit may have turned by a full turn.
fn orbit_peak_rate<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    ctx: &Context,
    half: &HalfOrbit,
) -> Result<f64, GnssAttitudeError> {
    let (eccentricity, peak) = search.orbit_peak(ctx, half)?;
    Ok(
        if eccentricity <= STEPPING_MAX_ECCENTRICITY + ECCENTRICITY_ALLOWANCE {
            peak
        } else {
            f64::INFINITY
        },
    )
}

/// Whether `zone` could hold at an instant whose `|beta|` is at least
/// `least_beta`, on an orbit whose orbital rate is at most the conic's
/// largest `peak_rate` with [`PEAK_RATE_MARGIN`] (always, where that rate is
/// infinite: [`orbit_peak_rate`]). Kouba's nominal yaw rate is at most
/// `mu_rate / tan|beta|` (its value at the centre), and the GLONASS-M ratio
/// `k g(d) / d` at most `k / tan|beta|`, so neither zone holds where
/// `tan|beta|` is at least `mu_rate / R`; the shadow holds only where
/// `|beta|` is below `pi` minus its limit (with [`SHADOW_BETA_ALLOWANCE`]).
fn could_hold(zone: Zone, least_beta: f64, peak_rate: f64) -> bool {
    if !peak_rate.is_finite() {
        return true;
    }
    match zone {
        Zone::NominalRateAbove { yaw_rate } | Zone::GlonassNoon { yaw_rate } => {
            libm::tan(least_beta) < PEAK_RATE_MARGIN * peak_rate / yaw_rate
        }
        Zone::Shadow { limit } => least_beta < PI - limit + SHADOW_BETA_ALLOWANCE,
    }
}

/// The zone of a half orbit, evaluated over the whole half orbit, not only at
/// its centre (on an eccentric orbit the orbital rate changes through the
/// centre, so Kouba's rate peaks off it): from its first instant to its last.
///
/// Kouba's rate holds beta constant through each instant (Kouba 2009), so
/// where beta passes through zero it falls to zero at that instant and turns
/// the other way after it. A nominal-rate zone is therefore found on each
/// part of the half orbit on which beta keeps one sign ([`beta_parts`]), and
/// by rule (the papers do not say how a zone continues through beta's change
/// of sign) the zone runs from the first part's first instant inside to the
/// last part's last; the rule applies only where the maximum yaw rate is at
/// least the orbital rate at the change ([`beta_parts`]). On each part (and
/// on the whole half orbit for the shadow
/// and GLONASS-M zones) the zone is [`zone_on`], from the centre. Where the
/// zone's margin rises to one maximum over a part and falls after it, the
/// zone is one interval there and the search finds an instant in it whenever
/// it holds at one the search resolves. That holds on circular orbits and on
/// every orbit of eccentricity up to 0.3 for Kouba's rate, and on the orbits
/// of [`GLONASS_NOON_BOUNDS`] for the GLONASS-M noon zone, with beta and the
/// orbit plane turning by up to [`PLANE_MOTION_MAX`] over an orbit, each
/// checked by the laws; the shadow's margin peaks at midnight. A part where
/// beta's sign is not taken is searched only where Kouba's zone there stays
/// within [`UNSIGNED_BETA_SPAN_MAX`] of the centre.
fn zone_span<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
    zone: Zone,
) -> Result<Span, GnssAttitudeError> {
    let last = half.last();
    let parts = match zone {
        Zone::NominalRateAbove { yaw_rate } => beta_parts(search, half, yaw_rate)?,
        Zone::Shadow { .. } | Zone::GlonassNoon { .. } => [
            Some(BetaPart {
                lo: half.start,
                hi: last,
                sign: 0.0,
            }),
            None,
        ],
    };
    let margin = |ctx: &Context| zone.margin(ctx, half.noon);
    let mut found: Option<(f64, f64)> = None;
    for part in parts.into_iter().flatten() {
        if let Zone::NominalRateAbove { yaw_rate } = zone {
            if part.sign == 0.0 {
                check_unsigned_beta_zone(search, half, yaw_rate)?;
            }
        }
        if let Some((a, b)) = zone_on(search, part.lo, part.hi, half.centre, &margin)? {
            found = Some(found.map_or((a, b), |(first, _)| (first, b)));
        }
    }
    let Some((start, end)) = found else {
        return Ok(if zone.holds_at_centre(&search.at(half.centre)?) {
            Span::AtCentre
        } else {
            Span::None
        });
    };
    Ok(match (start == half.start, end == last) {
        (false, false) => Span::Within(start, end),
        (true, false) => Span::FromStart(end),
        (false, true) => Span::ToEnd(start),
        (true, true) => Span::Whole,
    })
}

/// One GPS or GLONASS-M turn: the instants it starts and its zone ends,
/// solved on the trajectory, and the geometry there.
#[derive(Clone, Copy, Debug)]
pub(super) struct TurnEvent {
    pub(super) start_s: f64,
    pub(super) end_s: f64,
    start: Context,
    end: Context,
    /// In-plane angles to the centre, positive before it, at which the start
    /// and end yaws are formed: the zone's edge angle ([`Zone::edge_angle`])
    /// at the start and end geometry, on the side of the centre the start and
    /// end lie on (a zone off the centre can start after it).
    start_angle: f64,
    end_angle: f64,
}

/// `sin(mu)`, with `mu` the orbit angle from midnight, at in-plane angle `x`
/// before noon (`noon == true`) is `sin(x)`, before midnight `-sin(x)`: the
/// factor on `x`.
fn centre_side(noon: bool) -> f64 {
    if noon {
        1.0
    } else {
        -1.0
    }
}

impl TurnEvent {
    /// The turn whose zone runs from `start_s` to `end_s` in the half orbit
    /// around noon (`noon == true`) or midnight.
    fn new<T: AttitudeTrajectory + ?Sized>(
        search: &Search<'_, T>,
        zone: Zone,
        noon: bool,
        start_s: f64,
        end_s: f64,
    ) -> Result<Self, GnssAttitudeError> {
        let (start, end) = (search.at(start_s)?, search.at(end_s)?);
        let side = |ctx: &Context, at_centre: f64| {
            let angle = angle_to_centre(ctx, noon);
            if angle == 0.0 {
                at_centre
            } else {
                angle.signum()
            }
        };
        Ok(Self {
            start_s,
            end_s,
            start,
            end,
            start_angle: side(&start, 1.0) * zone.edge_angle(&start)?,
            end_angle: side(&end, -1.0) * zone.edge_angle(&end)?,
        })
    }

    /// A nominal-rate zone too narrow to hold more than one instant: a turn
    /// starting and ending at the centre instant, with its start and end
    /// yaws formed at the zone's half span before and after the centre (at
    /// beta = 0 the nominal yaw flips by pi at the centre, which the turn at
    /// the maximum rate bridges; just above, the zone is narrower than the
    /// time resolution).
    fn at_centre<T: AttitudeTrajectory + ?Sized>(
        search: &Search<'_, T>,
        zone: Zone,
        half: &HalfOrbit,
    ) -> Result<Self, GnssAttitudeError> {
        let centre = search.at(half.centre)?;
        let edge = zone.edge_angle(&centre)?;
        Ok(Self {
            start_s: half.centre,
            end_s: half.centre,
            start: centre,
            end: centre,
            start_angle: edge,
            end_angle: -edge,
        })
    }

    /// Nominal yaw at the turn start, `atan2(-tan(beta), sin(mu))` with the
    /// start's beta at its in-plane angle.
    fn start_yaw(&self, noon: bool) -> f64 {
        compute_phi(self.start.beta, centre_side(noon) * self.start_angle)
    }

    /// Nominal yaw at the end of the turn's zone.
    fn end_yaw(&self, noon: bool) -> f64 {
        compute_phi(self.end.beta, centre_side(noon) * self.end_angle)
    }

    /// Sun elevation at the turn start. The turn direction, the GPS reverse
    /// noon-turn condition and the start yaw all take it, so none changes
    /// within a turn, also where beta changes sign during it.
    fn beta(&self) -> f64 {
        self.start.beta
    }
}

#[derive(Clone, Copy)]
enum GpsKind {
    Iia {
        yaw_bias_rad: f64,
        shadow_limit_rad: f64,
    },
    Iir,
    Iif {
        yaw_bias_rad: f64,
        shadow_limit_rad: f64,
    },
}

/// The zone of a GPS law on the noon (`noon == true`) or midnight half orbit.
fn gps_zone(kind: GpsKind, yaw_rate: f64, noon: bool) -> Zone {
    match kind {
        GpsKind::Iia {
            shadow_limit_rad, ..
        }
        | GpsKind::Iif {
            shadow_limit_rad, ..
        } if !noon => Zone::Shadow {
            limit: shadow_limit_rad,
        },
        _ => Zone::NominalRateAbove { yaw_rate },
    }
}

/// The zone of the GLONASS-M law on the noon or midnight half orbit.
fn glonass_zone(yaw_rate: f64, shadow_limit: f64, noon: bool) -> Zone {
    if noon {
        Zone::GlonassNoon { yaw_rate }
    } else {
        Zone::Shadow {
            limit: shadow_limit,
        }
    }
}

/// The GPS or GLONASS-M turn of the half orbit holding `ctx`, for checking
/// that every instant of a half orbit finds the same turn.
#[cfg(test)]
pub(super) fn turn_event<T: AttitudeTrajectory + ?Sized>(
    law: &GnssAttitudeLaw,
    ctx: &Context,
    trajectory: &T,
) -> Result<Option<TurnEvent>, GnssAttitudeError> {
    let search = Search::new(trajectory, ctx.t_s);
    let half = HalfOrbit::containing(&search, ctx)?;
    let noon = half.noon;
    let zone = match *law {
        GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        } => gps_zone(
            GpsKind::Iia {
                yaw_bias_rad,
                shadow_limit_rad,
            },
            yaw_rate_rad_s,
            noon,
        ),
        GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s } => gps_zone(GpsKind::Iir, yaw_rate_rad_s, noon),
        GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        } => gps_zone(
            GpsKind::Iif {
                yaw_bias_rad,
                shadow_limit_rad,
            },
            yaw_rate_rad_s,
            noon,
        ),
        GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s,
            shadow_limit_rad,
        } => glonass_zone(yaw_rate_rad_s, shadow_limit_rad, noon),
        _ => return Ok(None),
    };
    gps_turn_of(&search, &half, zone)
}

/// The turn of a half orbit (GPS; GLONASS-M in tests): its zone over the
/// whole half orbit ([`zone_span`]). A zone holding at an end of the half
/// orbit (a custom maximum yaw rate so slow that the zone reaches the
/// quadrature to rounding) starts or ends there. A zone holding at one
/// instant only is a nominal-rate zone narrower than the time resolution,
/// turned from the centre, or a grazing shadow, which is no turn.
fn gps_turn_of<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
    zone: Zone,
) -> Result<Option<TurnEvent>, GnssAttitudeError> {
    let event =
        |start: f64, end: f64| TurnEvent::new(search, zone, half.noon, start, end).map(Some);
    let at_centre = || TurnEvent::at_centre(search, zone, half).map(Some);
    match zone_span(search, half, zone)? {
        Span::None => Ok(None),
        Span::AtCentre => at_centre(),
        Span::Within(start, end) if end > start => event(start, end),
        Span::Within(..) => match zone {
            Zone::NominalRateAbove { .. } => at_centre(),
            _ => Ok(None),
        },
        Span::FromStart(end) => event(half.start, end),
        Span::ToEnd(start) => event(start, half.last()),
        Span::Whole => event(half.start, half.last()),
    }
}

/// Largest GPS IIA maximum yaw rate, over the orbital rate at the requested
/// instant, on which the law is shown correct. The IIA shadow crossing turns
/// at that rate for its whole length (tens of minutes), so its yaw is set
/// only to the rate times the uncertainty of the shadow entry instant, a few
/// units in the last place of `t` or the rounding of the Sun-satellite angle
/// over its rate of change where that is longer; the other laws' turns at
/// fast rates end at once.
const GPS_IIA_MAX_RATE_RATIO: f64 = 1e5;

/// Most half orbits the GPS turn search goes back over, and the search for
/// the end of a turn forward over.
const GPS_HALF_ORBITS: u32 = 4096;

/// GPS IIA, IIR and IIF: linear yaw at a constant rate from the nominal yaw at
/// the turn start until the yaw reaches the nominal yaw again (Kouba 2009;
/// Dilssner 2010; Kouba's `eclips.f`, December 2017), with the noon and
/// midnight turns started where the nominal yaw rate first exceeds the maximum
/// yaw rate and the shadow crossing at the shadow entry. A turn that has not
/// reached the nominal yaw 1800 s after its zone ends returns to nominal yaw
/// steering there.
///
/// The turn at an instant is the latest one started by then, if it is still
/// under way. Each half orbit (quadrature to quadrature) holds at most one
/// zone, inside it, so the turns start in the order of their half orbits: the
/// search takes the half orbit holding the instant, and then the ones before
/// it, until one has a turn started by the instant. A turn ends at most 1800 s
/// after its zone, so before the end of its half orbit plus 1800 s: the search
/// stops at a half orbit that ends more than 1800 s before the instant (a turn
/// can last past the next half orbit and into the one after on a fast orbit
/// with a slow custom rate).
///
/// A half orbit whose zone could not hold anywhere in it ([`could_hold`], at
/// the least `|beta|` of the half orbit and the orbit's largest orbital
/// rate) has no turn and is passed over without a search; the validated
/// domain ([`check_gps_domain`]) is checked at the first half orbit whose
/// zone could hold, so an instant no turn can reach is nominal yaw steering
/// whatever the orbit's eccentricity and plane motion.
fn gps_linear<T: AttitudeTrajectory + ?Sized>(
    kind: GpsKind,
    yaw_rate: f64,
    ctx: &Context,
    trajectory: &T,
) -> Result<Outcome, GnssAttitudeError> {
    let search = Search::new(trajectory, ctx.t_s);
    let t = ctx.t_s;
    let mut half = HalfOrbit::containing(&search, ctx)?;
    let peak_rate = orbit_peak_rate(&search, ctx, &half)?;
    let mut checked = false;
    for back in 0..GPS_HALF_ORBITS {
        if back > 0 && exceeds(t, half.end, GPS_TURN_END_MARGIN_S) {
            return Ok(Outcome::Nominal);
        }
        let zone = gps_zone(kind, yaw_rate, half.noon);
        if could_hold(zone, half_orbit_least_beta(&search, &half)?, peak_rate) {
            if !checked {
                check_gps_domain(kind, yaw_rate, ctx, &search)?;
                checked = true;
            }
            if let Some(turn) = gps_turn_of(&search, &half, zone)?.filter(|turn| turn.start_s <= t)
            {
                return Ok(
                    gps_turn(kind, yaw_rate, &half, &turn, t, &search)?.unwrap_or(Outcome::Nominal)
                );
            }
        }
        half = half.previous(&search)?;
    }
    Err(search.not_found())
}

/// The validated domain of the GPS laws: the GPS IIA maximum yaw rate at most
/// [`GPS_IIA_MAX_RATE_RATIO`] times the orbital rate at the requested
/// instant, and the eccentricity and plane motion of [`checked_eccentricity`]
/// with the bound [`KOUBA_MAX_ECCENTRICITY`].
fn check_gps_domain<T: AttitudeTrajectory + ?Sized>(
    kind: GpsKind,
    yaw_rate: f64,
    ctx: &Context,
    search: &Search<'_, T>,
) -> Result<(), GnssAttitudeError> {
    if matches!(kind, GpsKind::Iia { .. }) && yaw_rate > GPS_IIA_MAX_RATE_RATIO * ctx.mu_rate {
        return Err(search.outside(
            "GPS IIA maximum yaw rate above 1e5 times the orbital rate, where the shadow crossing's yaw is set only to the rate times the uncertainty of the shadow entry",
        ));
    }
    checked_eccentricity(
        search,
        ctx,
        KOUBA_MAX_ECCENTRICITY,
        "orbit eccentricity above 0.3, the bound for the GPS turn zones",
    )?;
    Ok(())
}

/// The rate of a GPS noon turn (Kouba's `eclips.f`, December 2017):
/// `-sign(beta) R`, the way the nominal yaw moves through noon, except that a
/// yaw bias `b` turns it the other way, at `sign(b) R`, for beta of the sign
/// of `b` with `|beta| <= |b|` (GPS IIA with `b = +0.5 deg` for beta in
/// `(0, 0.5 deg]`, GPS IIF with `b = -0.7 deg` for beta in `[-0.7 deg, 0)`).
/// Both sides of beta = 0 then turn at `sign(b) R`, and so does beta = 0
/// itself (`eclips.f` gives `-R` at that one point, from Fortran's
/// `SIGN(R, 0.0)`, where its noon half span is `0 x infinity`). Returns the
/// rate and whether it runs against the nominal yaw's motion.
fn gps_noon_rate(kind: GpsKind, yaw_rate: f64, beta: f64) -> (f64, bool) {
    let bias = match kind {
        GpsKind::Iia { yaw_bias_rad, .. } | GpsKind::Iif { yaw_bias_rad, .. } => yaw_bias_rad,
        GpsKind::Iir => 0.0,
    };
    if bias != 0.0 && (beta == 0.0 || (beta * bias > 0.0 && beta.abs() <= bias.abs())) {
        (yaw_rate.copysign(bias), beta != 0.0)
    } else {
        (-yaw_rate.copysign(beta), false)
    }
}

/// The yaw of a GPS turn at `t_s`, at or after its start; `None` once the turn
/// has ended. `half` is the turn's half orbit.
///
/// Within the zone the yaw is linear from the start yaw: noon turns at the
/// rate of [`gps_noon_rate`], the GPS IIR midnight turn at `sign(beta) R`, the
/// GPS IIA shadow crossing at `sign(b) R` (the yaw bias sets the direction in
/// the shadow; Bar-Sever 1996, Kouba 2009, `eclips.f`; `b = 0` turns at `+R`
/// as `eclips.f` does), and the GPS IIF shadow crossing at the rate that
/// reaches the nominal yaw of the shadow exit at the exit, where it ends
/// (Dilssner 2010). After the zone the yaw continues at the same rate until it
/// meets the nominal yaw, except after the GPS IIA shadow, where the yaw
/// turns at `R` toward the nominal yaw the shorter way from the exit
/// (Bar-Sever 1996: `D = psi_n(t_e) - psi(t_e)` wrapped to 180 deg, rate
/// `SIGN(R, D)`, until the yaw equals the nominal yaw; `eclips.f` takes the
/// shorter way as well).
fn gps_turn<T: AttitudeTrajectory + ?Sized>(
    kind: GpsKind,
    yaw_rate: f64,
    half: &HalfOrbit,
    turn: &TurnEvent,
    t_s: f64,
    search: &Search<'_, T>,
) -> Result<Option<Outcome>, GnssAttitudeError> {
    let noon = half.noon;
    if exceeds(t_s, turn.end_s, GPS_TURN_END_MARGIN_S) {
        return Ok(None);
    }
    let beta = turn.beta();
    let phi_start = turn.start_yaw(noon);
    let (rate, mode, approach) = if noon {
        let (rate, reversed) = gps_noon_rate(kind, yaw_rate, beta);
        let approach = if reversed {
            Approach::Onward
        } else {
            Approach::Trailing
        };
        (rate, AttitudeMode::NoonTurn, approach)
    } else {
        match kind {
            GpsKind::Iia { yaw_bias_rad, .. } => {
                let rate = if yaw_bias_rad < 0.0 {
                    -yaw_rate
                } else {
                    yaw_rate
                };
                (rate, AttitudeMode::Shadow, Approach::Shorter)
            }
            GpsKind::Iir => (
                yaw_rate.copysign(beta),
                AttitudeMode::MidnightTurn,
                Approach::Trailing,
            ),
            GpsKind::Iif { .. } => {
                // Ends at the shadow exit, at the nominal yaw there.
                if t_s >= turn.end_s {
                    return Ok(None);
                }
                let rate = (turn.end_yaw(noon) - phi_start) / (turn.end_s - turn.start_s);
                return Ok(Some(Outcome::Yaw {
                    mode: AttitudeMode::Shadow,
                    yaw_rad: linear_yaw(phi_start, rate, t_s, turn.start_s)?,
                }));
            }
        }
    };
    if t_s < turn.end_s {
        return Ok(Some(Outcome::Yaw {
            mode,
            yaw_rad: linear_yaw(phi_start, rate, t_s, turn.start_s)?,
        }));
    }
    let exit_yaw = linear_yaw(phi_start, rate, turn.end_s, turn.start_s)?;
    match after_zone(search, half, turn, exit_yaw, rate, approach, t_s)? {
        Some(after_rate) => Ok(Some(Outcome::Yaw {
            mode,
            yaw_rad: linear_yaw(exit_yaw, after_rate, t_s, turn.end_s)?,
        })),
        None => Ok(None),
    }
}

/// `x` in `[-pi, pi]` congruent to it modulo 2 pi: `sin` and `cos` reduce
/// their argument by the exact multiple of `pi / 2`, so this holds to a few
/// units in the last place for every finite `x`.
fn reduce_angle(x: f64) -> f64 {
    if x.abs() <= PI {
        x
    } else {
        libm::atan2(libm::sin(x), libm::cos(x))
    }
}

/// The linear yaw `phi + rate (t - t0)` in `(-pi, pi]`, exact up to a few
/// roundings for every finite product: `t - t0` is split into two numbers
/// that sum to it exactly, each product with `rate` into two more (`fma`), and
/// each term reduced modulo 2 pi with [`reduce_angle`]; a plain product would
/// carry its own rounding, of order 1e-16 of it, into the angle, and reducing
/// it with a rounded 2 pi more. A difference or product that overflows is
/// `NonFinite`.
fn linear_yaw(phi: f64, rate: f64, t: f64, t0: f64) -> Result<f64, GnssAttitudeError> {
    let hi = t - t0;
    if !hi.is_finite() {
        return Err(GnssAttitudeError::NonFinite {
            field: "t_s - turn start",
        });
    }
    let back = hi - t;
    let lo = (t - (hi - back)) + (-t0 - back);
    let p1 = rate * hi;
    if !p1.is_finite() {
        return Err(GnssAttitudeError::NonFinite {
            field: "yaw_rate_rad_s x (t_s - turn start)",
        });
    }
    let e1 = libm::fma(rate, hi, -p1);
    let p2 = rate * lo;
    let e2 = libm::fma(rate, lo, -p2);
    Ok(wrap_pi(
        reduce_angle(phi)
            + reduce_angle(p1)
            + reduce_angle(e1)
            + (reduce_angle(p2) + reduce_angle(e2)),
    ))
}

/// How a GPS turn approaches the nominal yaw after its zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Approach {
    /// The turn ran the way the nominal yaw moves through a zone where the
    /// nominal yaw moves faster than the turn (Kouba 2009): at the zone's end
    /// the nominal yaw is at or ahead of the linear yaw by less than pi, so a
    /// gap computed below zero there is rounding and the turn has met it.
    Trailing,
    /// The turn continues in its direction until it meets the nominal yaw (a
    /// noon turn reversed by the yaw bias): the gap is taken in `[0, 2 pi)`.
    Onward,
    /// The yaw turns toward the nominal yaw the shorter way from the zone's
    /// end (GPS IIA after the shadow): the gap is in `[0, pi]`.
    Shorter,
}

/// The gap from the linear yaw to the nominal yaw at one instant after a
/// turn's zone, measured the way the yaw turns, and the nominal yaw there.
#[derive(Clone, Copy)]
struct Gap {
    t: f64,
    gap: f64,
    nominal: f64,
}

/// Whether a GPS turn is still under way at `t` (at or after its zone end,
/// within 1800 s of it): its rate after the zone if so, `None` once it has met
/// the nominal yaw (Kouba 2009; Bar-Sever 1996), the first instant after the
/// zone at which the gap from the linear yaw to the nominal yaw, measured the
/// way the yaw turns after the zone, is zero. `exit_yaw` is the linear yaw at
/// the zone end and `rate` the turn's rate within the zone; `own` is the
/// turn's half orbit. The gap at the zone end is the turn's end yaw, the
/// nominal yaw at the zone's edge angle, minus the exit yaw; later gaps add
/// the change of the nominal yaw from that end yaw. The edge lies within one
/// unit in the last place of `t` after the zone end, and where the zone is
/// only the centre instant (beta near 0) the satellite is on the Sun line to
/// rounding, so the end yaw is the nominal yaw's limit just after it, where
/// the nominal yaw formed from the geometry at that instant may still be on
/// the other side of its flip by pi.
///
/// The gap changes at the rate the nominal yaw moves the way the turn goes,
/// minus the turn's speed. Kouba's nominal yaw rate (beta held constant
/// through each instant) exceeds the turn's speed only inside its zone at
/// that speed ([`Zone::NominalRateAbove`]), at most one on each part of a
/// half orbit on which beta keeps one sign; the gap rises there where Kouba's
/// rate goes the turn's way, and falls everywhere else. The search visits,
/// half orbit by half orbit from the turn's own, the edges of those zones on
/// the parts where Kouba's rate goes the turn's way or where beta's sign is
/// not taken ([`after_zone_stops`]), the change of beta's sign
/// ([`beta_parts`]) and the ends of the half orbits, up to the last instant
/// within 1800 s of the zone end. From one visited instant to the next the
/// gap adds the change of the nominal yaw, which with beta of one sign stays
/// within a half plane of yaw, so within half a turn; only a flip by pi
/// through a centre where beta is zero to rounding has its direction in
/// doubt. The change is therefore taken in `[-pi / 2, 3 pi / 2)` measured the
/// way Kouba's rate goes at the part's sign of beta, and where beta's sign is
/// not taken, the way the turn goes: the law takes the flip at a turn's own
/// centre at beta = 0 as the one the turn bridges, and every instant from a
/// later such centre on lies in that centre's own turn or shadow. The gap meets
/// zero in the interval up to a visited instant where its value there is not
/// positive or, since beta's own rate also moves the nominal yaw (most near a
/// centre where beta passes through zero, where Kouba's rate leaves it out),
/// where the lowest gap a golden-section search over the interval finds is
/// not positive: the interval's gap may then fall and rise again once. The
/// meeting is the change of "gap positive" up to that instant, bisected with
/// the gap measured from the interval's start, whose own gap is the positive
/// one the walk carried to it (not formed again from the geometry there), so
/// the bracket's ends are checked by [`Search::transition`]. Every instant of
/// the turn takes the same visited instants and brackets: the meeting depends
/// only on the turn, and the search stops at the first interval reaching `t`.
fn after_zone<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    own: &HalfOrbit,
    turn: &TurnEvent,
    exit_yaw: f64,
    rate: f64,
    approach: Approach,
    t: f64,
) -> Result<Option<f64>, GnssAttitudeError> {
    let end = turn.end_s;
    let limit = latest_within(end, GPS_TURN_END_MARGIN_S);
    let nominal_end = turn.end_yaw(own.noon);
    let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
    let (after_rate, gap) = match approach {
        Approach::Trailing => {
            let ahead = sign(rate) * (nominal_end - exit_yaw);
            let gap = (ahead + FRAC_PI_2).rem_euclid(TAU) - FRAC_PI_2;
            (rate, gap.max(0.0))
        }
        Approach::Onward => (
            rate,
            (sign(rate) * (nominal_end - exit_yaw)).rem_euclid(TAU),
        ),
        Approach::Shorter => {
            let difference = wrap_pi(nominal_end - exit_yaw);
            (rate.abs().copysign(difference), difference.abs())
        }
    };
    if gap == 0.0 {
        // Met at the zone end; `t` is at or after it.
        return Ok(None);
    }
    if t <= end {
        // The zone end itself, with the gap still open.
        return Ok(Some(after_rate));
    }
    let direction = sign(after_rate);
    let speed = after_rate.abs();
    // The gap at `x` from the one at `from`, the nominal yaw's change taken
    // in [-pi / 2, 3 pi / 2) measured the way `reference` goes.
    let gap_from = |from: &Gap, x: f64, reference: f64| -> Result<Gap, GnssAttitudeError> {
        let (_, nominal) = nominal_yaw_at(search.trajectory(), x)?;
        let measured = reference * wrap_pi(nominal - from.nominal);
        let along = if measured < -FRAC_PI_2 {
            measured + TAU
        } else {
            measured
        };
        Ok(Gap {
            t: x,
            gap: from.gap + direction * reference * along - speed * (x - from.t),
            nominal,
        })
    };
    let mut from = Gap {
        t: end,
        gap,
        nominal: nominal_end,
    };
    let mut half = *own;
    for index in 0..GPS_HALF_ORBITS {
        // A turn that ran the way the nominal yaw moves through a zone where
        // that is faster than the turn (Kouba's noon turns, the IIR midnight
        // turn) has that zone as its own half orbit's: it ended at the zone
        // end, so the parts of its half orbit up to there have no other.
        let settled = (index == 0 && approach == Approach::Trailing).then_some(end);
        let (stops, count) = after_zone_stops(search, &half, direction, speed, settled)?;
        for &(stop, reference) in &stops[..count] {
            let at = stop.min(limit);
            if at <= from.t {
                continue;
            }
            let interval = |from: &Gap, x: f64| gap_from(from, x, reference);
            let next = interval(&from, at)?;
            if let Some(met) = meeting(search, &from, &next, &interval)? {
                return Ok((t < met).then_some(after_rate));
            }
            if at >= t {
                return Ok(Some(after_rate));
            }
            from = next;
        }
        half = half.next(search)?;
    }
    Err(search.not_found())
}

/// The instants [`after_zone`] visits in a half orbit, in order (the first
/// `count` of the array), each with the direction in which the nominal yaw's
/// change up to it from the visited instant before is measured: beta's change
/// of sign, the edges of Kouba's zone at the turn's speed `speed` on each part
/// of one sign of beta where Kouba's rate goes the turn's way `direction` and
/// on a part where beta's sign is not taken (except on parts that start at
/// or before `settled`), and the end of the half orbit. A zone narrower than
/// the time resolution at the centre is the centre and the instant before it.
/// The direction is Kouba's at the part's sign of beta, or `direction` where
/// beta's sign is not taken.
fn after_zone_stops<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    half: &HalfOrbit,
    direction: f64,
    speed: f64,
    settled: Option<f64>,
) -> Result<([(f64, f64); 7], usize), GnssAttitudeError> {
    let noon = half.noon;
    let kouba = Zone::NominalRateAbove { yaw_rate: speed };
    let margin = |ctx: &Context| kouba.margin(ctx, noon);
    let mut stops = [(f64::NAN, 0.0); 7];
    let mut count = 0;
    let mut push = |t: f64, reference: f64| {
        stops[count] = (t, reference);
        count += 1;
    };
    let parts = beta_parts(search, half, speed)?;
    let two = parts[1].is_some();
    let mut reference = direction;
    for part in parts.into_iter().flatten() {
        let unsigned = part.sign == 0.0;
        reference = if unsigned {
            direction
        } else {
            kouba_rate_sign(part.sign, noon)
        };
        if two && part.lo != half.start {
            push(part.lo, reference);
        }
        let open = settled.is_none_or(|end| part.lo > end);
        if open && (unsigned || reference == direction) {
            if unsigned {
                check_unsigned_beta_zone(search, half, speed)?;
            }
            match zone_on(search, part.lo, part.hi, half.centre, &margin)? {
                Some((a, b)) => {
                    push(a, reference);
                    push(b, reference);
                }
                None if part.lo <= half.centre
                    && half.centre <= part.hi
                    && kouba.holds_at_centre(&search.at(half.centre)?) =>
                {
                    push(half.centre.next_down(), reference);
                    push(half.centre, reference);
                }
                None => {}
            }
        }
        if two && part.hi != half.last() {
            push(part.hi, reference);
        }
    }
    push(half.end, reference);
    stops[..count].sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok((stops, count))
}

/// The first instant of `(from.t, next.t]` at which the gap is no longer
/// positive (the gap at `from` is), or `None` (see [`after_zone`]).
fn meeting<T, G>(
    search: &Search<'_, T>,
    from: &Gap,
    next: &Gap,
    gap_from: &G,
) -> Result<Option<f64>, GnssAttitudeError>
where
    T: AttitudeTrajectory + ?Sized,
    G: Fn(&Gap, f64) -> Result<Gap, GnssAttitudeError>,
{
    let anchor = *from;
    let gap = |x: f64| -> Result<f64, GnssAttitudeError> {
        if x == anchor.t {
            return Ok(anchor.gap);
        }
        Ok(gap_from(&anchor, x)?.gap)
    };
    let closed = if next.gap <= 0.0 {
        next.t
    } else {
        let lowered = |x: f64| -> Result<f64, GnssAttitudeError> { Ok(-gap(x)?) };
        let (x, low) = golden_section(
            anchor.t,
            next.t,
            0.5 * anchor.t + 0.5 * next.t,
            &lowered,
            &|v: f64| v >= 0.0,
        )?;
        if low < 0.0 {
            return Ok(None);
        }
        x
    };
    let open = |x: f64| -> Result<bool, GnssAttitudeError> { Ok(gap(x)? > 0.0) };
    Ok(Some(search.transition(&open, anchor.t, closed)?.1))
}

/// The nominal yaw at `t` from the yaw-steering geometry, and the geometry:
/// the nominal X axis is the direction of the Sun's component across the
/// radial direction, `e_r x (s x e_r)` with unit vectors. With the Sun exactly
/// on the radial line that component is zero at that one instant; its limit
/// just after is the component of `e_r x (s x v)`, the first-order term of
/// `e_r x (s x e_r(t))`. The direction is taken after scaling the vector by
/// its largest component, so a component whose square underflows (the
/// satellite within 1e-150 rad of the Sun line) still has one.
fn nominal_yaw_at<T: AttitudeTrajectory + ?Sized>(
    trajectory: &T,
    t: f64,
) -> Result<(Context, f64), GnssAttitudeError> {
    let sample = sample_at(trajectory, t)?;
    let ctx = Context::from_sample(t, &sample)?;
    let unit = |v: [f64; 3]| scale3(v, 1.0 / norm3(v));
    let sun = unit(sample.sun_position_m);
    let mut across = cross3(ctx.e_r, cross3(sun, ctx.e_r));
    if across == [0.0; 3] {
        across = cross3(ctx.e_r, cross3(sun, unit(sample.velocity_m_s)));
    }
    let largest = across.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
    if largest == 0.0 {
        return Err(GnssAttitudeError::DegenerateGeometry { t_s: Some(t) });
    }
    let x = scale3(across, 1.0 / largest);
    let along = cross3(ctx.e_n, ctx.e_r);
    // X . n = -sin(psi), X . along = cos(psi); the scale cancels.
    let yaw = libm::atan2(-dot3(x, ctx.e_n), dot3(x, along));
    Ok((ctx, yaw))
}

/// The yaw at the end of a GLONASS-M noon turn at the maximum rate R
/// (Dilssner et al. 2011): the smallest positive root of
/// `f(y) = y - g(k y)`, with `g(d) = |phi(beta, d) - phi(beta, -d)| / 2` the
/// yaw half swing the nominal law makes over a turn of in-plane half span `d`
/// ([`glonass_half_swing`]) and `k = mu_rate / R` (finite and positive).
///
/// `y = 0` is a trivial root. For `|beta| < atan(k)` (the noon-turn
/// condition), `f(y) / y` tends to `1 - k / tan|beta| < 0` as `y -> 0+`. The
/// half swing `g` is increasing and concave on `d` in `[0, pi / 2]` and
/// decreasing on `[pi / 2, pi]` to `g(pi) = 0`, so on `y` in
/// `(0, min(pi, pi / k)]` `f` is convex then increasing, negative up to its
/// one root `y*` and positive after it (`f > 0` at the upper end). Bisection on
/// that interval with the open lower end treated as negative converges to
/// `y*` to the last bit, for any `k`. (Orekit brackets on `[0, pi]` from
/// 75 deg and stops at the trivial root when its first step lands on 0, which
/// happens when `y*` is below `75 deg - 1 rad`; for `k > 1` the interval
/// `(0, pi]` also holds further roots, with `d` past `pi`.)
fn glonass_noon_yaw_end(beta: f64, k: f64) -> f64 {
    let f = |y: f64| y - glonass_half_swing(beta, k * y);
    let (mut lo, mut hi) = (0.0_f64, PI.min(PI / k));
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let value = f(mid);
        if value < 0.0 {
            lo = mid;
        } else if value > 0.0 {
            hi = mid;
        } else {
            return mid;
        }
    }
    hi
}

/// GLONASS-M (Dilssner et al. 2011): a noon turn at the maximum yaw rate over
/// the span where the nominal yaw needs more, and a shadow crossing at the
/// maximum rate, held at the nominal yaw of the shadow exit once reached. Both
/// start where the satellite enters the span or the shadow, solved on the
/// trajectory over the half orbit holding the instant ([`zone_span`]).
///
/// The noon turn is formulated on the Sun side of the orbit. With a maximum
/// yaw rate below the orbital rate its half span can reach the quadrature, so
/// the turn would run on into the night half orbit, where the law does not
/// define it: the instants within such a turn, on either side of the
/// quadrature, are [`GnssAttitudeError::InvalidParameter`] for
/// `yaw_rate_rad_s`, and the law still applies outside it. On the night side
/// those are the instants where the noon span's condition holds (it can only
/// past the quadrature for a zone reaching it); on the Sun side the instants
/// of a zone that holds at the half orbit's start or end. Physical GLONASS-M
/// rates are some 30 times the orbital rate, with half spans of a few
/// degrees.
///
/// Where the half orbit's zone could not hold anywhere in it ([`could_hold`],
/// at its least `|beta|` and, at noon, the orbit's largest orbital rate) the
/// attitude is nominal yaw steering without a search or the validated
/// domain's checks.
fn glonass_m<T: AttitudeTrajectory + ?Sized>(
    yaw_rate: f64,
    shadow_limit: f64,
    ctx: &Context,
    trajectory: &T,
) -> Result<Outcome, GnssAttitudeError> {
    let search = Search::new(trajectory, ctx.t_s);
    let t = ctx.t_s;
    let refused = || GnssAttitudeError::InvalidParameter {
        field: "yaw_rate_rad_s",
        value: yaw_rate,
    };
    let checked = || {
        checked_eccentricity(
            &search,
            ctx,
            STEPPING_MAX_ECCENTRICITY,
            "orbit eccentricity above 0.9, the bound for the trajectory searches",
        )
    };
    if !HalfOrbit::is_noon(ctx) && (Zone::GlonassNoon { yaw_rate }).contains(ctx, true)? {
        checked()?;
        return Err(refused());
    }
    let half = HalfOrbit::containing(&search, ctx)?;
    let noon = half.noon;
    let zone = glonass_zone(yaw_rate, shadow_limit, noon);
    let peak_rate = orbit_peak_rate(&search, ctx, &half)?;
    if !could_hold(zone, half_orbit_least_beta(&search, &half)?, peak_rate) {
        return Ok(Outcome::Nominal);
    }
    let eccentricity = checked()?;
    if noon {
        let largest_beta = beta_samples(&search, &half)?
            .iter()
            .fold(0.0_f64, |m, &(_, beta)| m.max(beta.abs()));
        let bound = GLONASS_NOON_BOUNDS
            .iter()
            .find(|&&(beta, _)| largest_beta < beta)
            .map_or(GLONASS_NOON_MAX_ECCENTRICITY, |&(_, e)| e);
        check_eccentricity(
            &search,
            eccentricity,
            bound,
            "orbit eccentricity above the bound for the GLONASS-M noon zone: 0.25 for |beta| below 30 deg, 0.2 below 60 deg, 0.1 above",
        )?;
    }
    let (start_s, end_s) = match zone_span(&search, &half, zone)? {
        Span::Within(start, end) if end > start => (start, end),
        // No zone, or one instant of a grazing shadow or span.
        Span::None | Span::AtCentre | Span::Within(..) => return Ok(Outcome::Nominal),
        Span::FromStart(end) if noon => {
            return if t <= end {
                Err(refused())
            } else {
                Ok(Outcome::Nominal)
            };
        }
        Span::ToEnd(start) if noon => {
            return if t >= start {
                Err(refused())
            } else {
                Ok(Outcome::Nominal)
            };
        }
        Span::Whole if noon => return Err(refused()),
        // A shadow (Sun-satellite angle beyond 90 deg) never holds at a
        // quadrature.
        Span::FromStart(_) | Span::ToEnd(_) | Span::Whole => return Err(search.not_found()),
    };
    if t < start_s || t > end_s {
        return Ok(Outcome::Nominal);
    }
    let turn = TurnEvent::new(&search, zone, noon, start_s, end_s)?;
    let beta = turn.beta();
    let phi_start = turn.start_yaw(noon);
    let dt_start = t - turn.start_s;
    if noon {
        let phi_dot = -yaw_rate.copysign(beta);
        Ok(Outcome::Yaw {
            mode: AttitudeMode::NoonTurn,
            yaw_rad: phi_start + phi_dot * dt_start,
        })
    } else {
        // Turn at the maximum rate toward the end yaw and hold it once
        // reached. The start and end yaws are on the same side of the turn
        // direction for every beta, including +-0 (start -+pi, end -+0).
        let phi_end = turn.end_yaw(noon);
        let phi_dot = yaw_rate.copysign(beta);
        let linear_phi = phi_start + phi_dot * dt_start;
        let yaw_rad = if phi_dot > 0.0 {
            linear_phi.min(phi_end)
        } else {
            linear_phi.max(phi_end)
        };
        Ok(Outcome::Yaw {
            mode: AttitudeMode::Shadow,
            yaw_rad,
        })
    }
}

/// Galileo IOV (Galileo satellite metadata 3.1.1). With the Sun unit vector in
/// the orbital frame (X along track, Y across track, Z toward the Earth's
/// centre) `S = (-sin(eta) cos(beta), -sin(beta), -cos(eta) cos(beta))`, `eta`
/// the in-plane angle from noon, the auxiliary Sun vector replaces `S_y` by
/// `S_Hy = (sin(beta_y) G + S_y) / 2 + (sin(beta_y) G - S_y) / 2 cos(pi |S_x| /
/// sin(beta_x))` in the region `|S_x| < sin(beta_x)`, `|S_y| < sin(beta_y)`,
/// with `G` the sign of `S_y` at the beginning of the region, and the yaw (IGS
/// convention, Section 3.2) is `atan2(S_Hy, S_x)`. The region is the same
/// around noon and midnight, and at its edge `|S_x| = sin(beta_x)` the yaw is
/// the nominal yaw. (Orekit's `Galileo` instead bounds it by the in-plane
/// angle, `beta_x` at noon and `acos(cos(beta_x) / cos(beta))` at midnight.)
///
/// GPS III uses the same law with `beta_y = 5.8 deg` (Montenbruck et al. 2026):
/// their Sun vector (eq. 11) `s = (cos(beta) sin(mu), -sin(beta), cos(beta)
/// cos(mu))`, `mu` from midnight, is `S` above; their eqs. (17), (18) and (13)
/// are `S_Hy` and the yaw above with `gamma_x = sin(beta_x)`,
/// `gamma_y = sin(beta_y)` and `G = sign(s_y,0)`; their region is
/// `|s_x| < gamma_x`, `|s_y| < gamma_y`.
fn galileo_iov<T: AttitudeTrajectory + ?Sized>(
    beta_x: f64,
    beta_y: f64,
    ctx: &Context,
    trajectory: &T,
) -> Result<Outcome, GnssAttitudeError> {
    let search = Search::new(trajectory, ctx.t_s);
    let (sin_x, sin_y) = (libm::sin(beta_x), libm::sin(beta_y));
    let in_region = |c: &Context| iov_sun_x(c).abs() < sin_x && libm::sin(c.beta).abs() < sin_y;
    if !in_region(ctx) {
        return Ok(Outcome::Nominal);
    }
    checked_eccentricity(
        &search,
        ctx,
        STEPPING_MAX_ECCENTRICITY,
        "orbit eccentricity above 0.9, the bound for the trajectory searches",
    )?;
    let entry = iov_region_entry(&search, &in_region, ctx)?;
    let gamma = 1.0_f64.copysign(-libm::sin(search.at(entry)?.beta));
    let s_x = iov_sun_x(ctx);
    let s_y = -libm::sin(ctx.beta);
    let s_hy = 0.5 * (sin_y * gamma + s_y)
        + 0.5 * (sin_y * gamma - s_y) * libm::cos(PI * s_x.abs() / sin_x);
    Ok(Outcome::Yaw {
        mode: if ctx.sun_side() {
            AttitudeMode::NoonTurn
        } else {
            AttitudeMode::MidnightTurn
        },
        yaw_rad: libm::atan2(s_hy, s_x),
    })
}

/// The along-track component of the Sun unit vector, `-sin(eta) cos(beta)`.
fn iov_sun_x(ctx: &Context) -> f64 {
    -libm::sin(ctx.argument_from_noon()) * libm::cos(ctx.beta)
}

/// Most sides of half orbits (a quadrature to a centre or a centre to a
/// quadrature) searched back for an instant outside the Galileo IOV region.
const IOV_REGION_SIDES: u32 = 16;

/// The beginning of the Galileo IOV auxiliary region that holds at `ctx`: the
/// last change of the region condition before `ctx`. The search goes back
/// side by side of the half orbits (the quadratures and centres divide them),
/// from `ctx`. On a side `|S_x| = cos(beta) |sin(eta)|` rises to at most one
/// maximum ([`golden_section`] finds it): `|sin(eta)|` is monotone there and
/// beta's change can only turn it back near the quadrature, where `|S_x|`
/// nears `sin(beta_x)` only for `beta_x + |beta|` near 90 deg. That maximum
/// splits the side into parts on which `|S_x|` is monotone and the
/// condition `|S_x| < sin(beta_x)`, `|sin(beta)| < sin(beta_y)` changes from
/// false to true at most once going forward (`|beta|` has at most one
/// minimum there), checked on a grid ([`iov_entry_bracket`]). The latest
/// part, from `ctx` back, at whose start the condition does not hold holds
/// the beginning, bisected there. Up to 16
/// sides (a region spans a quadrature only where `|S_x| = cos(beta)` there
/// is below `sin(beta_x)`, `beta_x + beta_y` above 90 deg).
fn iov_region_entry<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    in_region: &dyn Fn(&Context) -> bool,
    ctx: &Context,
) -> Result<f64, GnssAttitudeError> {
    let in_region_at =
        |t: f64| -> Result<bool, GnssAttitudeError> { Ok(in_region(&search.at(t)?)) };
    let sun_x = |t: f64| -> Result<f64, GnssAttitudeError> { Ok(iov_sun_x(&search.at(t)?).abs()) };
    // The last quadrature or centre passed: the side holding `ctx` starts there.
    let mut theta = [0.0, FRAC_PI_2, PI, -FRAC_PI_2]
        .into_iter()
        .find(|&theta| passed(ctx, theta) && !passed(ctx, wrap_pi(theta + FRAC_PI_2)))
        .ok_or_else(|| search.not_found())?;
    let (mut reference, mut hi) = (*ctx, ctx.t_s);
    for _ in 0..IOV_REGION_SIDES {
        let lo = search
            .crossing(&reference, theta, false)?
            .ok_or_else(|| search.not_found())?;
        let (peak, _) = golden_section(lo, hi, 0.5 * lo + 0.5 * hi, &sun_x, &|_| false)?;
        for (a, b) in [(peak, hi), (lo, peak)] {
            if a < b && !in_region_at(a)? {
                let (out, inside) = iov_entry_bracket(search, &in_region_at, a, b)?;
                return Ok(search.transition(&in_region_at, out, inside)?.1);
            }
        }
        (reference, hi, theta) = (search.at(lo)?, lo, wrap_pi(theta - FRAC_PI_2));
    }
    Err(search.not_found())
}

/// Intervals of the time grid on which the Galileo IOV region condition is
/// sampled over a part of a side before its beginning there is bisected.
const IOV_SAMPLE_INTERVALS: u32 = 32;

/// The last instant outside the Galileo IOV region and the first inside it
/// after that on a grid of [`IOV_SAMPLE_INTERVALS`] intervals over `[a, b]`
/// (`a` outside), the bracket of the region's beginning there. The condition
/// changes from outside to inside at most once on such a part (see
/// [`iov_region_entry`]); samples inside followed by one outside are
/// [`GnssAttitudeError::OutsideValidatedDomain`].
fn iov_entry_bracket<T, F>(
    search: &Search<'_, T>,
    in_region_at: &F,
    a: f64,
    b: f64,
) -> Result<(f64, f64), GnssAttitudeError>
where
    T: AttitudeTrajectory + ?Sized,
    F: Fn(f64) -> Result<bool, GnssAttitudeError>,
{
    let mut out = a;
    let mut inside = None;
    for k in 1..=IOV_SAMPLE_INTERVALS {
        let t = if k == IOV_SAMPLE_INTERVALS {
            b
        } else {
            a + (b - a) * (f64::from(k) / f64::from(IOV_SAMPLE_INTERVALS))
        };
        if in_region_at(t)? {
            inside.get_or_insert(t);
        } else if inside.is_some() {
            return Err(search.outside(
                "the Galileo IOV region condition changes more than once between a quadrature or centre and the peak of |S_x|",
            ));
        } else {
            out = t;
        }
    }
    Ok((out, inside.unwrap_or(b)))
}

/// The start angle of a cosine turn at one instant, signed like
/// [`angle_to_centre`]: positive before noon or midnight. It decreases
/// monotonically through the half orbit around the centre, so its crossing of
/// the start value is unique there.
fn turn_angle(ctx: &Context, noon: bool, kind: CosineTurnStartAngle) -> f64 {
    let in_plane = angle_to_centre(ctx, noon);
    match kind {
        CosineTurnStartAngle::OrbitAngle => in_plane,
        CosineTurnStartAngle::Colinearity => {
            // Galileo satellite metadata 3.1.2: epsilon = acos(r . y) folded
            // into [0, 90 deg], y = n x (n x s) = (n . s) n - s, so
            // r . y = -r . s. Measured from the chosen centre it is the angle
            // between r and s (noon) or -s (midnight).
            let cos_epsilon = if noon { ctx.svb_cos } else { -ctx.svb_cos };
            acos_clamped(cos_epsilon).copysign(in_plane)
        }
    }
}

/// Most noon and midnight passes before the requested instant whose cosine
/// turns are checked.
const COSINE_TURN_PASSES: u32 = 16;

/// Points per start angle of in-plane motion on which the colinearity angle
/// is scanned for its fall to the start value.
const COLINEARITY_SCAN_POINTS: f64 = 32.0;

/// The cosine turn of Galileo FOC (Galileo satellite metadata 3.1.2) and the
/// BeiDou-3 CAST satellites (Wang et al. 2018).
///
/// A turn starts on the approach to noon or midnight at the first instant at
/// which the start angle falls to its value, if `|beta|` is within the limit
/// there (the metadata's switch-over: the angle below 10 deg now, above it at
/// the previous epoch, and beta below 4.1 deg), and lasts half the period
/// whatever the geometry does meanwhile. The in-plane orbit angle (BeiDou)
/// falls through the value once per approach. The colinearity angle (Galileo)
/// is never below the in-plane angle, so it can first reach the value once the
/// in-plane angle has; from that instant it is scanned to the centre
/// ([`colinearity_fall`]; at the centre the signed angle jumps from
/// `+|beta|` to `-|beta|`, which is no fall), and its first fall is bisected
/// to the last bit, also where it dips below the value only between two scan
/// points.
///
/// The turn at an instant is the latest one started by then whose start
/// passed the beta condition; it is a turn if half the period has not yet
/// elapsed. Each start is solved on the trajectory from the in-plane angle's
/// crossing of the start value, located the same way from every instant of the
/// pass, so every instant of a turn uses the same start, start yaw and
/// direction.
///
/// Where `|beta|` fails the beta condition over the whole last half period
/// ([`cosine_turn_could_start`]) no turn can be under way, and the attitude
/// is nominal yaw steering without a search or the validated domain's
/// checks.
fn cosine_turn<T: AttitudeTrajectory + ?Sized>(
    turn: &CosineTurnParameters,
    ctx: &Context,
    trajectory: &T,
) -> Result<Outcome, GnssAttitudeError> {
    let search = Search::new(trajectory, ctx.t_s);
    let half_period = 0.5 * turn.period_s;
    let eligible = |abs_beta: f64| {
        if turn.beta_limit_inclusive {
            abs_beta <= turn.beta_limit_rad
        } else {
            abs_beta < turn.beta_limit_rad
        }
    };
    if !cosine_turn_could_start(&search, ctx, half_period, &eligible)? {
        return Ok(Outcome::Nominal);
    }
    checked_eccentricity(
        &search,
        ctx,
        STEPPING_MAX_ECCENTRICITY,
        "orbit eccentricity above 0.9, the bound for the trajectory searches",
    )?;
    let started = |noon: bool, start: f64| -> Result<Option<Outcome>, GnssAttitudeError> {
        let start_ctx = search.at(start)?;
        if !eligible(start_ctx.beta.abs()) {
            return Ok(None);
        }
        let elapsed = ctx.t_s - start;
        Ok(Some(if elapsed <= half_period {
            cosine_yaw(turn, noon, &start_ctx, elapsed)
        } else {
            Outcome::Nominal
        }))
    };
    // The centre ahead, when this instant is within its in-plane start window.
    let ahead_noon = ctx.argument_from_noon() < 0.0;
    let mut reference = *ctx;
    if angle_to_centre(ctx, ahead_noon) <= turn.start_angle_rad {
        if let Some(found) = cosine_start(&search, turn, ahead_noon, ctx)? {
            if let Some(start) = found.start.filter(|&start| start <= ctx.t_s) {
                if let Some(outcome) = started(ahead_noon, start)? {
                    return Ok(outcome);
                }
            }
            reference = search.at(found.window)?;
        }
    }
    // The centres before this instant, the most recent first, each window
    // searched back from the one after it (half a turn on) or from this
    // instant.
    let mut noon = !ahead_noon;
    for _ in 0..COSINE_TURN_PASSES {
        let Some(found) = cosine_start(&search, turn, noon, &reference)? else {
            return Ok(Outcome::Nominal);
        };
        if let Some(start) = found.start {
            if let Some(outcome) = started(noon, start)? {
                return Ok(outcome);
            }
        }
        // Every earlier start precedes this window.
        if ctx.t_s - found.window > half_period {
            return Ok(Outcome::Nominal);
        }
        reference = search.at(found.window)?;
        noon = !noon;
    }
    Err(search.not_found())
}

/// Most quadratures and centres [`cosine_turn_could_start`] steps back over.
const COSINE_SKIP_SIDES: u32 = 8;

/// Whether a cosine turn could be under way at `ctx`: whether `|beta|`
/// passes the turn's beta condition `eligible` anywhere in the last half
/// period, `[t - period / 2, t]`, where every start of a turn still under way
/// lies. Beta changes monotonically within each half orbit (the trajectory
/// contract), so its least magnitude there ([`least_abs_beta`]) is that of
/// its values at `t`, at the quadratures and centres back to
/// `t - period / 2` ([`Search::crossing_since`], which reads the trajectory
/// at no earlier instant) and at `t - period / 2`. True, so that the full
/// search decides, where more than [`COSINE_SKIP_SIDES`] quadratures and
/// centres lie in that span or its start is not finite.
fn cosine_turn_could_start<T, E>(
    search: &Search<'_, T>,
    ctx: &Context,
    half_period: f64,
    eligible: &E,
) -> Result<bool, GnssAttitudeError>
where
    T: AttitudeTrajectory + ?Sized,
    E: Fn(f64) -> bool,
{
    let earliest = ctx.t_s - half_period;
    if !earliest.is_finite() {
        return Ok(true);
    }
    let mut betas = [0.0; COSINE_SKIP_SIDES as usize + 2];
    betas[0] = ctx.beta;
    let mut count = 1;
    let mut theta = [0.0, FRAC_PI_2, PI, -FRAC_PI_2]
        .into_iter()
        .find(|&theta| passed(ctx, theta) && !passed(ctx, wrap_pi(theta + FRAC_PI_2)))
        .ok_or_else(|| search.not_found())?;
    let mut from = *ctx;
    for _ in 0..COSINE_SKIP_SIDES {
        let Some(crossed) = search.crossing_since(&from, theta, earliest)? else {
            betas[count] = search.at(earliest)?.beta;
            count += 1;
            return Ok(eligible(least_abs_beta(&betas[..count])));
        };
        from = search.at(crossed)?;
        betas[count] = from.beta;
        count += 1;
        theta = wrap_pi(theta - FRAC_PI_2);
    }
    Ok(true)
}

/// The start window of a cosine turn and its start, if the start angle falls
/// to its value before the centre.
struct CosineStart {
    /// The instant the in-plane angle to the centre falls to the start value.
    window: f64,
    /// The turn start: the window itself for the orbit angle, the first fall
    /// of the colinearity angle after it.
    start: Option<f64>,
}

/// The start of the cosine turn of the last noon (`noon == true`) or
/// midnight whose start window `reference` has passed (by at most half a
/// turn); `None` when the trajectory's representable time ends before that
/// window.
fn cosine_start<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    turn: &CosineTurnParameters,
    noon: bool,
    reference: &Context,
) -> Result<Option<CosineStart>, GnssAttitudeError> {
    let centre = if noon { 0.0 } else { PI };
    let Some(window) = search.crossing(reference, centre - turn.start_angle_rad, false)? else {
        return Ok(None);
    };
    let start = match turn.start_angle {
        CosineTurnStartAngle::OrbitAngle => Some(window),
        CosineTurnStartAngle::Colinearity => colinearity_fall(search, turn, noon, window)?,
    };
    Ok(Some(CosineStart { window, start }))
}

/// Most scan points of the colinearity angle per approach: at 32 points per
/// start angle of orbit angle at the orbital rate where each step starts, an
/// orbit within the validated eccentricity reaches the centre in a few more
/// than 32 of them (the rate changes by under 3 % within a step).
const COLINEARITY_SCAN_LIMIT: u32 = 64 * 32;

/// The first instant from `window` (the in-plane angle's crossing of the start
/// value) at which the colinearity angle falls to the start value before the
/// centre, on a scan of `start / (32 mu_rate)` per step at the orbital rate
/// where each step starts (or the next representable instant, where that step
/// is below the resolution of `t`, so that the scan then visits every
/// instant); `None` when it stays above the value up to the centre. With
/// beta moving by at most [`PLANE_MOTION_MAX`] over a turn the angle
/// falls to one least value on the way to the centre and rises after it
/// (`cos(epsilon) = cos(beta) cos(x)` peaks where
/// `tan(x) = tan(beta) beta_rate / mu_rate`, once), so a dip below the start
/// value between two scan points, possible only with `|beta|` within a
/// scan step of the start value, holds that least value: where no scan point
/// falls, a golden-section search over the scanned span finds it. A window
/// at or past the centre (a time resolution coarser than the start window)
/// and a scan that has not reached the centre within its points are
/// [`GnssAttitudeError::OutsideValidatedDomain`].
fn colinearity_fall<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    turn: &CosineTurnParameters,
    noon: bool,
    window: f64,
) -> Result<Option<f64>, GnssAttitudeError> {
    let start_value = turn.start_angle_rad;
    let kind = CosineTurnStartAngle::Colinearity;
    let above = |t: f64| -> Result<bool, GnssAttitudeError> {
        Ok(turn_angle(&search.at(t)?, noon, kind) > start_value)
    };
    let first = search.at(window)?;
    if angle_to_centre(&first, noon) <= 0.0 {
        return Err(search.outside(
            "the time resolution at the instant is coarser than the cosine turn's start window",
        ));
    }
    if turn_angle(&first, noon, kind) <= start_value {
        return Ok(Some(window));
    }
    let (mut previous, mut rate) = (window, first.mu_rate);
    for _ in 0..COLINEARITY_SCAN_LIMIT {
        let stepped = previous + start_value / (COLINEARITY_SCAN_POINTS * rate);
        let t = if stepped > previous {
            stepped
        } else {
            previous.next_up()
        };
        if !t.is_finite() {
            return Err(search.not_found());
        }
        let at = search.at(t)?;
        if angle_to_centre(&at, noon) <= 0.0 {
            // Past the centre: the angle's last value before it, or its least
            // value on the way there.
            let before_centre = |x: f64| -> Result<bool, GnssAttitudeError> {
                Ok(angle_to_centre(&search.at(x)?, noon) > 0.0)
            };
            let (last, _) = search.transition(&before_centre, previous, t)?;
            if !above(last)? {
                return Ok(Some(search.transition(&above, previous, last)?.1));
            }
            let lowered = |x: f64| -> Result<f64, GnssAttitudeError> {
                Ok(-turn_angle(&search.at(x)?, noon, kind))
            };
            let (low_at, low) = golden_section(
                window,
                last,
                0.5 * window + 0.5 * last,
                &lowered,
                &|v: f64| -v <= start_value,
            )?;
            if -low > start_value {
                return Ok(None);
            }
            return Ok(Some(search.transition(&above, window, low_at)?.1));
        }
        if turn_angle(&at, noon, kind) <= start_value {
            return Ok(Some(search.transition(&above, previous, t)?.1));
        }
        (previous, rate) = (t, at.mu_rate);
    }
    Err(search.outside("the colinearity scan did not reach the centre within its points"))
}

/// The cosine-turn yaw `90 deg s + (psi_0 - 90 deg s) cos(2 pi elapsed / T)`,
/// `psi_0` the nominal yaw at the start and `s` its sign.
fn cosine_yaw(turn: &CosineTurnParameters, noon: bool, start: &Context, elapsed: f64) -> Outcome {
    let psi_start = start.geometric_nominal_yaw();
    let quarter = if psi_start >= 0.0 {
        FRAC_PI_2
    } else {
        -FRAC_PI_2
    };
    Outcome::Yaw {
        mode: if noon {
            AttitudeMode::NoonTurn
        } else {
            AttitudeMode::MidnightTurn
        },
        yaw_rad: quarter + (psi_start - quarter) * libm::cos((elapsed / turn.period_s) * TAU),
    }
}

/// BeiDou-3 SECM (CSNO 2019): nominal yaw steering with beta held at
/// `+-beta_0` while `|beta| < beta_0`.
fn fixed_beta(beta0: f64, ctx: &Context) -> Outcome {
    if ctx.beta.abs() >= beta0 {
        return Outcome::Nominal;
    }
    let held = if ctx.beta >= 0.0 { beta0 } else { -beta0 };
    // sin(mu) with mu from midnight is -sin(u) with u from noon.
    let sin_mu = -libm::sin(ctx.argument_from_noon());
    Outcome::Yaw {
        mode: AttitudeMode::ModifiedYawSteering,
        yaw_rad: libm::atan2(-libm::tan(held), sin_mu),
    }
}

/// The last orbit quadrature at or before `ctx`, orbit angle 90 deg from
/// midnight (argument from noon `-pi / 2`), solved on the trajectory: within
/// half a turn after it, searched back to it; otherwise back to the other
/// quadrature first.
fn last_quadrature<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    ctx: &Context,
) -> Result<f64, GnssAttitudeError> {
    let found = |instant: Option<f64>| instant.ok_or_else(|| search.not_found());
    if passed(ctx, -FRAC_PI_2) {
        return found(search.crossing(ctx, -FRAC_PI_2, false)?);
    }
    let other = found(search.crossing(ctx, FRAC_PI_2, false)?)?;
    found(search.crossing(&search.at(other)?, -FRAC_PI_2, false)?)
}

/// The instant one orbit before `ctx`: the last one before it at the same
/// argument from noon, a full turn of orbit angle back, solved on the
/// trajectory as two half turns.
fn one_orbit_before<T: AttitudeTrajectory + ?Sized>(
    search: &Search<'_, T>,
    ctx: &Context,
) -> Result<f64, GnssAttitudeError> {
    let found = |instant: Option<f64>| instant.ok_or_else(|| search.not_found());
    let u = ctx.argument_from_noon();
    let half_turn = found(search.crossing(ctx, wrap_pi(u + PI), false)?)?;
    found(search.crossing(&search.at(half_turn)?, u, false)?)
}

/// BeiDou-2 IGSO and MEO as Orekit's `BeidouMeo`: orbit normal while `|beta|`
/// is below the switch limit for a whole orbit, yaw steering while above, and
/// otherwise the mode the satellite switched to at the last quadrature, the
/// instant the orbit angle from midnight was 90 deg. The change of beta over
/// the last orbit is taken from the instant a full turn of orbit angle back
/// (Orekit takes `2 pi / mu_rate` at the requested instant, which on an
/// eccentric orbit is not the orbit's period). Where `|beta|` is above both
/// limits by more than that change, the satellite is in yaw steering whatever
/// the quadrature gives, and only the orbit's eccentricity is checked (the
/// instant one orbit back is found by the steps of the searches); the plane's
/// motion is not.
fn beidou_orbit_normal_switch<T: AttitudeTrajectory + ?Sized>(
    ys_to_on: f64,
    on_to_ys: f64,
    ctx: &Context,
    trajectory: &T,
) -> Result<Outcome, GnssAttitudeError> {
    let search = Search::new(trajectory, ctx.t_s);
    let beta = ctx.beta;
    let beta_variation = beta - beta_at(trajectory, one_orbit_before(&search, ctx)?)?;

    // |beta| over the last orbit lay within |beta_variation| of its value
    // now. Below both limits by more than that, the satellite has been
    // eligible for orbit normal at every quadrature whichever way |beta|
    // moves; above both, for yaw steering. (Orekit's `BeidouMeo` takes
    // `ys_to_on - |db|` and `on_to_ys + |db|`; with `ys_to_on > on_to_ys`
    // those overlap when |db| < (ys_to_on - on_to_ys) / 2 and skip the
    // quadrature test.)
    let lower = ys_to_on.min(on_to_ys);
    let upper = ys_to_on.max(on_to_ys);
    if beta.abs() > upper + beta_variation.abs() {
        check_eccentricity(
            &search,
            search.orbit_shape(ctx)?.eccentricity,
            STEPPING_MAX_ECCENTRICITY,
            "orbit eccentricity above 0.9, the bound for the trajectory searches",
        )?;
        return Ok(Outcome::Nominal);
    }
    checked_eccentricity(
        &search,
        ctx,
        STEPPING_MAX_ECCENTRICITY,
        "orbit eccentricity above 0.9, the bound for the trajectory searches",
    )?;
    if beta.abs() <= lower - beta_variation.abs() {
        return Ok(orbit_normal());
    }
    let abs_beta_decreasing = beta * beta_variation <= 0.0;
    if abs_beta_decreasing {
        if beta.abs() >= ys_to_on {
            return Ok(Outcome::Nominal);
        }
    } else if beta.abs() <= on_to_ys {
        return Ok(orbit_normal());
    }
    let beta_quadrature = beta_at(trajectory, last_quadrature(&search, ctx)?)?;
    let switched = if abs_beta_decreasing {
        beta_quadrature.abs() <= ys_to_on
    } else {
        beta_quadrature.abs() <= on_to_ys
    };
    Ok(if switched {
        orbit_normal()
    } else {
        Outcome::Nominal
    })
}
