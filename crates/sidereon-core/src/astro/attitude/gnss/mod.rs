//! GNSS satellite attitude: nominal yaw steering, the block-specific noon and
//! midnight turn and shadow-crossing laws, orbit-normal mode, and the
//! Sun-fixed DYB frame.
//!
//! [`OrbitAttitudeTrajectory`] supplies GCRF states and the Sun along an orbit
//! on the TDB seconds past J2000 axis.
//!
//! # Conventions
//!
//! All vectors are in one geocentric inertial frame chosen by the caller; the
//! velocity must be the inertial velocity. Body axes follow the IGS convention
//! (Montenbruck et al. 2015, Section 2): `+Z` points to the Earth's centre,
//! `+Y` is the solar panel axis and, under nominal yaw steering, is along
//! `sun x r`, and `+X` completes the right-handed triad so the Sun stays in
//! the `+X` hemisphere.
//!
//! The yaw angle `psi` is the rotation about `+Z` defined by
//! `X . n = -sin(psi)` and `Y . n = -cos(psi)`, with `n` the unit orbit normal
//! along `r x v`. Under nominal yaw steering `psi = atan2(-tan(beta), sin(mu))`
//! (Kouba 2009), where `beta` is the Sun's elevation above the orbit
//! plane and `mu` the orbit angle from midnight in the direction of motion.
//! Orbit-normal mode is `psi = 0`. Yaw angles are reported in `(-pi, pi]`.
//!
//! The DYB frame (Springer, Beutler and Rothacher 1999; Arnold et al. 2015) is
//! Sun-fixed and does not depend on the attitude law: `D` points from the
//! satellite to the Sun, `Y = unit(D x e_r)` (the nominal solar panel axis)
//! and `B = D x Y`.
//!
//! # Laws
//!
//! [`GnssAttitudeLaw::for_block`] gives the published law of each
//! [`GnssBlock`] with its published parameters; every parameter can also be set
//! explicitly.
//!
//! - GPS IIA, IIR (and IIR-M) and IIF: Kouba (2009) and Kouba's `eclips.f`
//!   (December 2017). Noon turns run at the maximum yaw rate `R` the way the
//!   nominal yaw moves, `-sign(beta) R`, except that a yaw bias `b` (GPS IIA
//!   `+0.5 deg`, GPS IIF `-0.7 deg`) turns them at `sign(b) R` for beta of the
//!   sign of `b` with `|beta| <= |b|`, and at beta = 0, the limit from both
//!   sides. The IIR midnight turn runs at `sign(beta) R`. The IIA shadow
//!   crossing runs at `sign(b) R` (the bias sets the direction in the shadow;
//!   Bar-Sever 1996), and after the exit the yaw turns at `R` toward the
//!   nominal yaw the shorter way (Bar-Sever 1996: the difference at the exit
//!   wrapped to 180 deg sets the direction; `eclips.f` takes the shorter way
//!   too). The IIF shadow crossing runs at the constant rate that links the
//!   nominal yaws at shadow entry and exit (Dilssner 2010). A turn starts at
//!   the geometric crossing (see Turn timing) and ends where the yaw meets the
//!   nominal yaw, at most 1800 s after its zone (`eclips.f`). Orekit's
//!   `GPSBlockIIA` keeps turning at `+R` after the shadow and in it whatever the
//!   bias, and reverses noon turns only for `0 < beta < b`; its `GPSBlockIIR`
//!   and `GPSBlockIIF` laws within a turn are the ones above.
//! - GPS III: its manufacturer has published no attitude law. Montenbruck et
//!   al. (2026) represent the JPL GPS III attitude by the Galileo IOV law
//!   below with a larger collinearity region, `|S_x| < sin(15 deg)` (about
//!   15 deg of orbit angle either side of noon and midnight) and
//!   `|S_y| < sin(5.8 deg)`, and recommend it as the transitional standard for
//!   orbit determination and PPP with GPS III; [`GnssAttitudeLaw::for_block`]
//!   gives it with [`GPS_III_BETA_X_RAD`] and [`GPS_III_BETA_Y_RAD`]. The model
//!   has no yaw bias: the turn direction is the sign of `S_y` where the region
//!   begins (their eq. 18). Orekit 13.1.10's `SatelliteType.BLOCK_IIIA` builds
//!   its `GPSBlockIIF` provider instead.
//! - GLONASS-M: noon turn at the maximum yaw rate over the span Dilssner et al.
//!   (2011) give, and the midnight shadow crossing at the maximum rate held
//!   once the nominal yaw of the shadow exit is reached.
//! - Galileo IOV: the auxiliary-Sun-vector law of the European GNSS Service
//!   Centre satellite metadata (Section 3.1.1, `beta_x = 15 deg`,
//!   `beta_y = 2 deg`; GPS III above with `beta_y = 5.8 deg`) in its region `|S_x| < sin(beta_x)`,
//!   `|S_y| < sin(beta_y)` of the Sun's along-track and cross-track components,
//!   the same around noon and midnight, with the sign `G` taken at the
//!   region's beginning. Orekit's `Galileo` bounds the region instead by the
//!   in-plane angle, `beta_x` at noon and `acos(cos(beta_x) / cos(beta))` at
//!   midnight, so it starts the law later and ends it earlier, with a step to
//!   the nominal yaw at its edges.
//! - Galileo FOC: the European GNSS Service Centre satellite metadata
//!   Section 3.1.2 law, `psi = 90 deg * s + (psi_0 - 90 deg * s) *
//!   cos(2 pi t / 5656 s)` from the switch-over, the first instant the
//!   colinearity angle is below 10 deg (it was above at the epoch before) with
//!   beta below 4.1 deg, `s` the sign of the yaw `psi_0` there. The metadata
//!   writes the beta condition without an absolute value; it is taken as
//!   `|beta| < 4.1 deg`, as the turn is symmetric in beta.
//! - BeiDou-2 GEO and BeiDou-3 GEO: orbit-normal mode. BeiDou-2 IGSO and MEO:
//!   yaw steering, switched to orbit-normal mode when `|beta|` falls below
//!   4.1 deg and back above 3.9 deg at the orbit quadrature (orbit angle 90 deg
//!   from midnight), as Orekit's `BeidouIGSO` and `BeidouMeo`; the quadrature
//!   instant and the instant one orbit back (a full turn of orbit angle), from
//!   which the change of beta over the last orbit is taken, are solved on the
//!   trajectory (Orekit extrapolates back to the quadrature at the requested
//!   instant's orbital rate and takes the orbit as `2 pi / mu_rate` there).
//! - BeiDou-3 IGSO and CAST MEO: the Wang et al. (2018) law, the same cosine law
//!   with a 3090 s (MEO) or 5740 s (IGSO) period from orbit angle 6 deg before
//!   noon or midnight with `|beta| <= 3 deg`.
//! - BeiDou-3 SECM MEO: nominal yaw steering with `beta` replaced by
//!   `+-3 deg`, by the sign of `beta`, while `|beta| < 3 deg` (CSNO 2019).
//!
//! Orekit 13.1.10 attaches its Galileo law to every BeiDou-3 block and to
//! Galileo FOC; the laws above are the ones the satellite operators published
//! for those blocks, so they differ from Orekit inside the small-beta windows.
//!
//! Every finite positive maximum yaw rate is accepted (the validated domain
//! below bounds the GPS IIA rate, and the GPS rates in a half orbit where
//! beta changes sign or is zero to rounding). A turn's yaw is the linear yaw reduced
//! modulo 2 pi exactly up to a few roundings, also where the rate times the
//! time in the turn is far beyond 2 pi; where that product overflows the
//! result is [`GnssAttitudeError::NonFinite`] naming it.
//!
//! # Turn timing
//!
//! Turn laws depend on when a turn started. The caller supplies the trajectory
//! as an [`AttitudeTrajectory`] (a function from time to state and Sun
//! position) and the laws read it at the turn start, around the turn end, one
//! orbit earlier and at the last quadrature, as Orekit's
//! `GNSSAttitudeProvider` reads its `PVCoordinatesProvider`. Nothing is cached
//! between calls: each result depends only on the law, the trajectory and the
//! instant.
//!
//! Each GPS and GLONASS-M turn has one start instant, the geometric crossing
//! the papers define, solved on the trajectory: where the nominal yaw rate
//! `mu_rate tan(beta) cos(mu) / (sin^2(mu) + tan^2(beta))` first exceeds the
//! maximum yaw rate for the GPS noon turns and the GPS IIR midnight turn
//! (Kouba 2009), where the in-plane angle to noon comes within the Dilssner et
//! al. (2011) half span for the GLONASS-M noon turn, and where the
//! Sun-satellite angle exceeds the shadow limit for the GPS IIA, IIF and
//! GLONASS-M shadow crossings.
//!
//! The quadratures (the in-plane angle 90 deg from noon) divide the orbit into
//! half orbits around noon and midnight. Each holds at most one zone, inside
//! it, so the turns follow the half orbits. The laws solve each quadrature and
//! centre on the trajectory: they step toward it by at most `pi / 8` of orbit
//! angle at the orbital rate where each step starts (less than an orbital
//! period on orbits of eccentricity up to 0.9) and bisect the step in which
//! the in-plane angle passes it, deciding which side of the angle an instant
//! lies on in exact arithmetic. A zone is looked for over its whole half
//! orbit, each condition taken with the instant's own orbital rate and beta
//! (on an eccentric orbit the nominal yaw rate peaks off the centre): from
//! the ends and the centre and, where none of them is inside it, by a
//! golden-section search in time for the condition's largest margin, which
//! works the same wherever t = 0 lies; its start and end are then bisected
//! between that instant and the half orbit's ends. This finds every zone
//! whose margin has one maximum per half orbit, which holds on the validated
//! domain below (GPS: eccentricity up to 0.3; GLONASS-M noon turns: up to
//! 0.25, 0.2 or 0.1 by `|beta|`). Kouba's nominal yaw rate holds beta
//! constant through each instant, so where beta passes through zero (in the
//! eclipse season, once a year per orbit plane) it falls to zero at that
//! instant and turns the other way after it. The GPS zones are found on each
//! part of the half orbit on which beta keeps one sign (its change of sign
//! bisected), and by rule a zone found on both parts is one turn, from the
//! first instant inside to the last, with the turn's direction and start yaw
//! taken from the beta at its start. The papers do not say how a zone
//! continues through beta's change of sign. The rule applies only where the
//! maximum yaw rate is at least the orbital rate at the change: both parts
//! then hold a zone only where beta changes sign within 0.002 rad of orbit
//! angle of the centre, and the two zones lie within 0.0097 rad of it and
//! less than 0.0024 rad apart (on a circular orbit, with beta moving by up
//! to 0.05 rad over an orbit). A slower rate is outside the validated domain
//! in that half orbit. Every bisection checks that the
//! ends of its bracket differ and takes its points from one fixed binary tree
//! of the representable numbers, so the same change bracketed from different
//! instants gives the same instant to the last bit, also where rounding makes
//! a condition flicker: every instant of a turn uses the same start, start yaw
//! (the nominal yaw at the zone's edge) and beta, and the yaw is a function of
//! the turn, not of the requested instant.
//!
//! The GPS turns run at their rate through their whole zone, also through a
//! shadow longer than one full turn at that rate (GPS IIA at 0.2 deg/s and low
//! beta), and after it until the yaw meets the nominal yaw, at most 1800 s
//! after the zone ends. The gap between the linear and the nominal yaw falls
//! everywhere except where Kouba's nominal yaw rate goes the turn's way faster
//! than the turn, within at most one zone per half orbit, where it rises; the
//! meeting is its first zero, bisected in the first interval between the
//! edges of those zones, the changes of beta's sign and the ends of the half
//! orbits whose later end has no positive gap, or whose lowest gap a
//! golden-section search over the interval finds below zero (beta's own rate
//! also moves the nominal yaw, most near a centre where beta passes through
//! zero, which Kouba's rate leaves out). Between two of those instants beta
//! keeps one sign, so the nominal yaw stays within half a turn and the gap is
//! followed without a multiple of 2 pi in doubt. The 1800 s are taken
//! exactly: the last instant within them is the last representable one at
//! most 1800 s after the zone, also where instants are more than 1800 s apart
//! (beyond 1.6e19 s). The turn at an instant
//! is the latest one started by then, searched back half orbit by half orbit
//! while a turn of that half orbit could still be under way: a GPS turn may
//! continue past the quadrature (a custom maximum yaw rate far below the
//! published ones, or a fast custom orbit), into the next half orbit of its
//! kind. The IIF shadow crossing ends at the exit. The GLONASS-M turns end
//! with the zone, and a GLONASS-M noon turn whose zone reaches the quadrature
//! (a custom maximum yaw rate below the orbital rate) is
//! [`GnssAttitudeError::InvalidParameter`] at every instant within it, on
//! either side of the quadrature.
//!
//! Orekit estimates the turn span instead from each requested instant's
//! geometry (the in-plane angle to the turn centre and the half span at that
//! instant's beta, divided by that instant's orbital rate), with Kouba's
//! small-angle half span `sqrt(beta_0 |beta| - beta^2)` for the GPS noon
//! turns. Its start therefore moves within a turn with the Sun's in-plane
//! motion and with beta, and it forms the start yaw with the beta of the
//! requested instant. On circular GPS and GLONASS orbits with the Sun moving
//! at its mean rate, its start lies up to 3.9 s from the geometric start
//! (Kouba's small-angle form alone starts up to 0.85 s later), its start yaw
//! (with the GPS IIF shadow rate drift) differs by up to 6.7e-3 rad, and its
//! in-turn yaw differs by up to 9.0e-3 rad.
//!
//! A Galileo FOC or BeiDou-3 CAST turn starts on the approach to noon or
//! midnight: the in-plane orbit angle's crossing of the start value is solved
//! like a quadrature, and is the BeiDou-3 CAST start; the Galileo FOC
//! colinearity angle, never below the in-plane angle, is scanned from there to
//! the centre at steps of 1/32 of the start value of orbit angle at the
//! orbital rate where each step starts (every representable instant where
//! that step is finer than the time resolution, as near 1e18 s), and its
//! first fall to 10 deg is bisected to the last bit; where no scan point is
//! below the value, a golden-section search over the scanned span finds the
//! angle's least value (beta moving slowly, the angle falls to one least
//! value and rises after it), so a fall between two scan points (`|beta|`
//! within a scan step of the start value) is found too. The turn at an instant
//! is the latest one started by then whose start met the beta condition. The
//! beginning of the Galileo IOV law's region (Galileo IOV and GPS III) is the
//! last change of the region condition before the requested instant: the
//! search goes back side by side of the half orbits (quadrature to centre and
//! centre to quadrature), splits each side where `|S_x|` peaks (beta's change
//! can turn `|S_x|` back near a quadrature), and bisects the latest part at
//! whose start the condition does not hold, from the trajectory up to the
//! requested instant only.
//!
//! # Validated domain
//!
//! The searches above are shown correct on the domain below by sweeps of it
//! against independent evaluations of each law from the closed form of
//! Keplerian orbits whose beta changes at a constant rate (some of them
//! turned into a general orientation), and by scans showing that each GPS
//! and GLONASS-M zone's margin rises to one maximum per half orbit there.
//! The sweeps reach each limit the laws check: eccentricities to 3e-4 past
//! each bound (past the bound's allowance below), beta changing by up to
//! 0.0498 rad over an orbit, the GLONASS-M noon tiers to 1e-9 rad below
//! their limits of `|beta|`, and the GPS IIA maximum yaw rate past its bound;
//! beta takes either sign and exactly 0, and `|beta|` reaches 89.9 deg less
//! six times its change over a half orbit. Where a turn could reach the
//! requested instant the laws check the domain, and outside it they return
//! [`GnssAttitudeError::OutsideValidatedDomain`] naming the limit, never an
//! attitude.
//!
//! The laws assume that beta changes monotonically within each half orbit
//! (part of the trajectory contract, see [`AttitudeTrajectory`]). They detect
//! a trajectory that breaks this only where it shows at the instants they
//! sample.
//!
//! Where no turn can reach the requested instant, the attitude is nominal yaw
//! steering, whatever the orbit's eccentricity and plane motion, and the
//! domain is not checked:
//!
//! - GPS and GLONASS-M: in each half orbit whose turn could reach the instant
//!   (for GPS the instant's and the earlier ones that end less than 1800 s
//!   before it, for GLONASS-M the instant's), the least `|beta|` (from its
//!   values at the half orbit's quadratures and centre) is at least the
//!   largest beta at which the zone can hold: `atan(mu_max / R)` for Kouba's
//!   zone and the GLONASS-M noon zone (whose margins are at most
//!   `ln(mu_rate / (R tan|beta|))`), with `mu_max` twice the largest orbital
//!   rate of the conic through the satellite's positions at the quadratures
//!   and centre of the instant's half orbit at the instant's angular
//!   momentum, and 180 deg less the shadow limit (and 1e-9 rad) for a shadow.
//!   That conic's eccentricity must be at most 0.9 (and the allowance below),
//!   where the steps that found the half orbits turn by less than an orbit.
//! - Galileo FOC and the BeiDou-3 CAST and IGSO cosine turns (and custom
//!   ones): `|beta|` fails the turn's beta condition over the whole last half
//!   period, where every start of a turn still under way lies (from its
//!   values at the instant, at the quadratures and centres back to half a
//!   period before it, and there; the trajectory is read no earlier). The
//!   steps that find those quadratures and centres turn by less than an
//!   orbit where the half period is shorter than the orbital period (as for
//!   the published turns) or the eccentricity is below about 0.97.
//! - BeiDou-2 IGSO and MEO: `|beta|` is above both switch limits by more than
//!   its change over the last orbit, and the eccentricity is at most 0.9 (the
//!   instant one orbit back is found by steps); the plane's motion is not
//!   checked.
//! - Galileo IOV and GPS III: the instant lies outside the region.
//!
//! Elsewhere the law takes the orbit's eccentricity as that of the conic
//! through the satellite's positions at the requested instant and about a
//! quarter and half a turn of orbit angle before it (so for any
//! gravitational parameter and scale), and requires:
//!
//! - beta, and the direction of the orbit normal, to change by at most
//!   0.05 rad over the turn of orbit angle before the instant, one orbital
//!   period (the Sun's motion changes beta by at most about 0.02 rad over a
//!   geostationary orbit);
//! - an eccentricity of at most 0.9, and less for some laws below. The bounds
//!   carry an allowance of 2e-4 for the conic fit, which a turning orbit plane
//!   biases by up to about 0.04 times the square of its turn.
//!
//! A search that runs out of steps, and a Galileo FOC start window narrower
//! than the time resolution at the instant (near 1.5e19 s), are refused too.
//! By law:
//!
//! - GPS IIA, IIR and IIF: eccentricity at most 0.3, any argument of perigee,
//!   beta of either sign and 0, any yaw bias and shadow limit. Beta changes
//!   sign at most once within a half orbit (it is monotone there); the laws
//!   sample it at the half orbit's start, centre and end and at the 15
//!   instants dividing it into 16 equal intervals, and refuse samples whose
//!   sign changes more than once. Where beta changes sign within a half
//!   orbit, the maximum yaw rate must be at least the orbital rate at the
//!   change (the rule of Turn timing; the published rates are 9.7 to 24
//!   times the mean motion of a GPS orbit). Otherwise every finite positive maximum yaw rate for GPS
//!   IIR and IIF; for GPS IIA at most 1e5 times the orbital rate at the
//!   requested instant (its shadow crossing turns at that rate for its whole
//!   length, so its yaw is set only to the rate times the uncertainty of the
//!   shadow entry instant: a few units in the last place of the time
//!   argument, or the rounding of the Sun-satellite angle over its rate of
//!   change where that is longer). In a half orbit where beta stays within 4
//!   machine epsilons of zero at those samples (its computed value is then
//!   rounding), the maximum yaw rate must be at least 1.8e-3 times the
//!   orbital rate at the centre, so that Kouba's zone at such a beta lies
//!   within 1e-6 rad of orbit angle of the centre.
//! - GLONASS-M: in a noon half orbit an eccentricity of at most 0.25 where
//!   the largest `|beta|` of the samples above is below 30 deg, 0.2 below
//!   60 deg and 0.1 above (the Dilssner margin has two maxima from `|beta|`
//!   of 34.9 deg at 0.3 with beta fixed, and from 2.5 deg at 0.3 with beta
//!   moving by 0.1 rad per orbit), elsewhere 0.9; every finite
//!   positive maximum yaw rate (a noon turn reaching the quadrature is
//!   [`GnssAttitudeError::InvalidParameter`] as above).
//! - Galileo IOV and GPS III: eccentricity at most 0.9, any acute `beta_x`
//!   and `beta_y`. The region condition changes from outside to inside at
//!   most once between a quadrature or centre and the peak of `|S_x|`; the law
//!   samples it on 32 equal intervals there before bisecting its beginning
//!   and refuses samples that leave the region again.
//! - Galileo FOC, BeiDou-3 CAST and IGSO cosine turns (and custom cosine
//!   turns with any acute start angle and beta limit and any positive period)
//!   and BeiDou-2 IGSO and MEO: eccentricity at most 0.9.
//!
//! # References
//!
//! - Kouba J. (2009) A simplified yaw-attitude model for eclipsing GPS
//!   satellites. GPS Solutions 13:1-12.
//! - Montenbruck O., Schmid R., Mercier F., Steigenberger P., Noll C.,
//!   Fatkulin R., Kogure S., Ganeshan A.S. (2015) GNSS satellite geometry and
//!   attitude models. Advances in Space Research 56(6):1015-1029.
//! - Dilssner F. (2010) GPS IIF-1 satellite: antenna phase center and attitude
//!   modeling. InsideGNSS 5(6):59-64.
//! - Dilssner F., Springer T., Gienger G., Dow J. (2011) The GLONASS-M
//!   satellite yaw-attitude model. Advances in Space Research 47(1):160-171.
//! - Wang C., Guo J., Zhao Q., Liu J. (2018) Yaw attitude modeling for BeiDou
//!   I06 and BeiDou-3 satellites. GPS Solutions 22:117.
//! - CSNO (2019) Definitions and descriptions of BDS/GNSS satellite parameters
//!   for high precision applications.
//! - European GNSS Service Centre, Galileo satellite metadata, Section 3.1.
//! - Montenbruck O., Duan B., Steigenberger P., Hugentobler U. (2026) Improved
//!   attitude modeling for GPS III satellites in the eclipse season. Advances
//!   in Space Research 77(4):4901-4908, doi:10.1016/j.asr.2025.12.111.
//! - Bar-Sever Y.E. (1996) A new model for GPS yaw attitude. Journal of
//!   Geodesy 70:714-723.
//! - Kouba J., `eclips.f` (December 2017), distributed by the IGS Analysis
//!   Center Coordinator.
//! - Springer T.A., Beutler G., Rothacher M. (1999) A new solar radiation
//!   pressure model for GPS satellites. GPS Solutions 2(3):50-62.
//! - Arnold D. et al. (2015) CODE's new solar radiation pressure model for
//!   GNSS orbit determination. Journal of Geodesy 89:775-791.
//! - Orekit 13.1.10, package `org.orekit.gnss.attitude` (Apache-2.0).

mod block;
mod geometry;
mod laws;
mod orbit;
mod search;

pub use block::{BeidouManufacturer, BeidouOrbitClass, GnssBlock, GnssBlockError};
pub use orbit::OrbitAttitudeTrajectory;

use crate::astro::math::vec3::{cross3, neg3, norm3, sub3, unit3};
use geometry::{wrap_pi, Context};
use std::f64::consts::PI;

/// Satellite state and Sun position at one instant, in one geocentric
/// inertial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudeSample {
    /// Satellite position, metres.
    pub position_m: [f64; 3],
    /// Satellite inertial velocity, metres per second.
    pub velocity_m_s: [f64; 3],
    /// Geocentric Sun position, metres.
    pub sun_position_m: [f64; 3],
}

/// A caller-supplied trajectory: satellite state and Sun position as a
/// function of time.
///
/// `t_s` is in seconds on any uniform time scale; the laws only take
/// differences of it. Return `None` for an instant the trajectory cannot
/// supply; the attitude call then fails with
/// [`GnssAttitudeError::TrajectoryUnavailable`]. Every closure
/// `Fn(f64) -> Option<AttitudeSample>` is a trajectory.
///
/// The laws assume that the Sun's elevation above the orbit plane, beta,
/// changes monotonically within each half orbit (from one quadrature, where
/// the orbit angle is 90 deg from noon, to the next), as a satellite orbit's
/// beta does: the Sun's motion and the plane's precession move it by up to
/// about 1 deg a day, and only near its yearly extremes (`|beta|` of 15 deg
/// or more for GNSS orbits), where that motion stops, can the short-period
/// motion of the orbit plane (below about 1e-4 rad) turn it back. A
/// trajectory whose beta turns back within a half orbit lies outside the
/// validated domain (see the module documentation); the laws detect that
/// only where it shows at the instants they sample.
///
/// Besides the requested instant, the laws read the trajectory: for GPS and
/// GLONASS-M, over the half orbit holding the requested instant (quadrature to
/// quadrature around a noon or midnight, and up to one search step of
/// `pi / 8` of orbit angle beyond each end), for GPS also over the earlier
/// half orbits that end less than 1800 s before the requested instant and,
/// after a turn's zone, over the half orbits up to 1800 s past it; for Galileo
/// FOC and BeiDou-3, from the requested instant back over the last half
/// period of the turn and, where a turn could be under way, over the start
/// windows of the passes whose turns could still be running; for the Galileo
/// IOV law (Galileo IOV and GPS III), from the requested instant back to one
/// search step before the quadrature before the noon or midnight (earlier
/// quadratures where the region spans one), and at no instant after the
/// requested one; and for BeiDou-2 IGSO and MEO, back over one orbit (a full
/// turn of orbit angle) and to the last quadrature. Where a turn could reach
/// the requested instant (see the module's validated domain), a law also
/// reads the trajectory back over a full turn of orbit angle and up to one
/// search step more, for the orbit's eccentricity and the motion of its
/// plane.
pub trait AttitudeTrajectory {
    /// The state at `t_s`, or `None` when the trajectory does not cover it.
    fn sample(&self, t_s: f64) -> Option<AttitudeSample>;
}

impl<F> AttitudeTrajectory for F
where
    F: Fn(f64) -> Option<AttitudeSample> + ?Sized,
{
    fn sample(&self, t_s: f64) -> Option<AttitudeSample> {
        self(t_s)
    }
}

/// Why an attitude could not be computed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum GnssAttitudeError {
    /// An input value is NaN or infinite, or a norm or product the law forms
    /// from finite inputs overflows (`field` then names that product).
    #[error("{field} is not finite")]
    NonFinite {
        /// The offending input.
        field: &'static str,
    },
    /// An input vector has zero length, position and velocity are parallel so
    /// no orbit plane is defined, or a product the law normalises underflows
    /// (`field` then names that product).
    #[error("{field} has zero length")]
    ZeroLength {
        /// The offending input.
        field: &'static str,
    },
    /// The trajectory returned no state at an instant the law needs.
    #[error("trajectory has no state at t = {t_s} s")]
    TrajectoryUnavailable {
        /// The instant requested from the trajectory.
        t_s: f64,
    },
    /// A law parameter is outside its domain.
    #[error("{field} = {value} is outside its domain")]
    InvalidParameter {
        /// The parameter.
        field: &'static str,
        /// Its value.
        value: f64,
    },
    /// GPS IIA yaw rates are per satellite and there is no published default
    /// for this PRN (or no PRN was given).
    #[error("no default GPS IIA yaw rate for PRN {prn:?}")]
    NoDefaultYawRate {
        /// The PRN given, if any.
        prn: Option<u8>,
    },
    /// The Sun lies along the orbit normal, so yaw steering is undefined, or
    /// (for [`dyb_axes`]) exactly along the radial direction.
    #[error("degenerate Sun geometry at t = {t_s:?} s")]
    DegenerateGeometry {
        /// The instant, when the input came from a trajectory.
        t_s: Option<f64>,
    },
    /// A BeiDou-3 MEO satellite whose manufacturer is unknown is in a
    /// small-beta window where the CAST and SECM laws give different yaw.
    #[error("BeiDou-3 MEO yaw at t = {t_s} s depends on the manufacturer")]
    BeidouManufacturerRequired {
        /// The requested instant.
        t_s: f64,
    },
    /// A turn boundary could not be found: the trajectory does not reach the
    /// noon, midnight or quadrature a law needs within its representable time,
    /// a bracket a search formed holds no change of its condition (which the
    /// laws' assumptions on the orbit exclude), the beginning of a Galileo IOV
    /// law region lies more than eight half orbits back (as where the region
    /// holds everywhere), or a cosine turn's period is so long that more than
    /// sixteen passes before the instant could still be in their turns.
    #[error("no end found for the yaw turn at t = {t_s} s")]
    TurnEndNotFound {
        /// The requested instant.
        t_s: f64,
    },
    /// The trajectory, the law's parameters or the instant lie outside the
    /// domain on which the law is shown to be correct (see the module's
    /// validated domain): the orbit's eccentricity is above the law's bound,
    /// beta or the orbit plane turns by more than 0.05 rad over an orbit,
    /// beta changes sign more than once within a half orbit, the Galileo IOV
    /// region condition changes more than once where it should change once,
    /// a GPS IIA maximum yaw rate is above its bound, a GPS maximum yaw rate
    /// is below the orbital rate where beta changes sign within a half orbit
    /// or too slow for a half orbit whose beta is zero to rounding, a search
    /// ran out of steps, or the time resolution at the instant is coarser
    /// than an interval the law resolves. `reason` says which.
    #[error("outside the validated domain at t = {t_s} s: {reason}")]
    OutsideValidatedDomain {
        /// The requested instant.
        t_s: f64,
        /// The limit exceeded.
        reason: &'static str,
    },
}

/// Which part of the attitude law produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AttitudeMode {
    /// Nominal yaw steering.
    Nominal,
    /// A turn around orbit noon.
    NoonTurn,
    /// A turn around orbit midnight outside a shadow-crossing law.
    MidnightTurn,
    /// The Earth-shadow crossing law (GPS IIA and IIF, GLONASS-M), including
    /// the GPS IIA post-shadow recovery.
    Shadow,
    /// Orbit-normal mode.
    OrbitNormal,
    /// Yaw steering with a modified Sun elevation (BeiDou-3 SECM).
    ModifiedYawSteering,
}

/// The angle whose crossing of [`CosineTurnParameters::start_angle_rad`]
/// starts a cosine turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CosineTurnStartAngle {
    /// The colinearity angle of the Galileo satellite metadata (Section
    /// 3.1.2): the angle between the satellite direction and the Sun
    /// direction (at noon) or the anti-Sun direction (at midnight),
    /// `acos(|r . s|)` with unit `r` and `s`. Its cosine is
    /// `cos(beta) cos(in-plane angle)`.
    Colinearity,
    /// The in-plane orbit angle from noon or midnight (Wang et al. 2018).
    OrbitAngle,
}

/// The cosine turn law shared by Galileo FOC and the BeiDou-3 CAST
/// satellites.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosineTurnParameters {
    /// The turn starts only if `|beta|` is below this at its start.
    pub beta_limit_rad: f64,
    /// Whether `|beta|` equal to the limit also starts a turn.
    pub beta_limit_inclusive: bool,
    /// Which angle to noon or midnight starts the turn.
    pub start_angle: CosineTurnStartAngle,
    /// Value of that angle, before noon or midnight, at which the turn
    /// starts.
    pub start_angle_rad: f64,
    /// Period of the cosine; the turn lasts half of it.
    pub period_s: f64,
}

/// GPS IIA default yaw bias, 0.5 deg (Kouba 2009; Orekit
/// `GPSBlockIIA.DEFAULT_YAW_BIAS`).
pub const GPS_IIA_DEFAULT_YAW_BIAS_RAD: f64 = 0.5 * PI / 180.0;
/// GPS IIR maximum yaw rate, 0.20 deg/s (Kouba 2009; Orekit
/// `GPSBlockIIR.DEFAULT_YAW_RATE`).
pub const GPS_IIR_DEFAULT_YAW_RATE_RAD_S: f64 = 0.2 * PI / 180.0;
/// GPS IIF maximum yaw rate, 0.11 deg/s (Orekit `GPSBlockIIF.DEFAULT_YAW_RATE`,
/// from Kouba's `eclips.f`).
pub const GPS_IIF_DEFAULT_YAW_RATE_RAD_S: f64 = 0.11 * PI / 180.0;
/// GPS IIF yaw bias, -0.7 deg (Kouba's `eclips.f`, December 2017; Orekit
/// `GPSBlockIIF.DEFAULT_YAW_BIAS`).
pub const GPS_IIF_DEFAULT_YAW_BIAS_RAD: f64 = -0.7 * PI / 180.0;
/// Sun-satellite angle beyond which GPS IIA and IIF are in the Earth's
/// shadow, 180 - 13.25 deg (Kouba's `eclips.f`; Orekit `NIGHT_TURN_LIMIT` of
/// `GPSBlockIIA` and `GPSBlockIIF`).
pub const GPS_SHADOW_LIMIT_RAD: f64 = (180.0 - 13.25) * PI / 180.0;
/// GLONASS-M maximum yaw rate, 0.25 deg/s (Dilssner et al. 2011; Orekit
/// `Glonass.DEFAULT_YAW_RATE`).
pub const GLONASS_M_DEFAULT_YAW_RATE_RAD_S: f64 = 0.25 * PI / 180.0;
/// Sun-satellite angle beyond which GLONASS-M is in the Earth's shadow,
/// 180 - 14.2 deg (Orekit `Glonass` `NIGHT_TURN_LIMIT`).
pub const GLONASS_M_SHADOW_LIMIT_RAD: f64 = (180.0 - 14.20) * PI / 180.0;
/// Galileo IOV `beta_x`, 15 deg (Galileo satellite metadata 3.1.1).
pub const GALILEO_IOV_BETA_X_RAD: f64 = 15.0 * PI / 180.0;
/// Galileo IOV `beta_y`, 2 deg (Galileo satellite metadata 3.1.1).
pub const GALILEO_IOV_BETA_Y_RAD: f64 = 2.0 * PI / 180.0;
/// GPS III `beta_x`, 15 deg: the Galileo IOV law's region
/// `|S_x| < sin(15 deg)` (Montenbruck et al. 2026, Section 3,
/// `gamma_x = sin(15 deg)`).
pub const GPS_III_BETA_X_RAD: f64 = 15.0 * PI / 180.0;
/// GPS III `beta_y`, 5.8 deg: the Galileo IOV law's region
/// `|S_y| < sin(5.8 deg)`, and `|S_y|` of the auxiliary Sun vector at noon and
/// midnight (Montenbruck et al. 2026, Section 3, `gamma_y = sin(5.8 deg)`; the
/// paper's first statement of the limits writes it `gamma_z`).
pub const GPS_III_BETA_Y_RAD: f64 = 5.8 * PI / 180.0;
/// Galileo FOC turn law (Galileo satellite metadata 3.1.2): `|beta| < 4.1 deg`,
/// colinearity angle 10 deg, period 5656 s.
pub const GALILEO_FOC_TURN: CosineTurnParameters = CosineTurnParameters {
    beta_limit_rad: 4.1 * PI / 180.0,
    beta_limit_inclusive: false,
    start_angle: CosineTurnStartAngle::Colinearity,
    start_angle_rad: 10.0 * PI / 180.0,
    period_s: 5656.0,
};
/// BeiDou-3 CAST MEO turn law (Wang et al. 2018): `|beta| <= 3 deg`, orbit
/// angle 6 deg, period 3090 s.
pub const BEIDOU3_CAST_MEO_TURN: CosineTurnParameters = CosineTurnParameters {
    beta_limit_rad: 3.0 * PI / 180.0,
    beta_limit_inclusive: true,
    start_angle: CosineTurnStartAngle::OrbitAngle,
    start_angle_rad: 6.0 * PI / 180.0,
    period_s: 3090.0,
};
/// BeiDou-3 IGSO turn law (Wang et al. 2018): `|beta| <= 3 deg`, orbit angle
/// 6 deg, period 5740 s.
pub const BEIDOU3_CAST_IGSO_TURN: CosineTurnParameters = CosineTurnParameters {
    beta_limit_rad: 3.0 * PI / 180.0,
    beta_limit_inclusive: true,
    start_angle: CosineTurnStartAngle::OrbitAngle,
    start_angle_rad: 6.0 * PI / 180.0,
    period_s: 5740.0,
};
/// BeiDou-3 SECM fixed Sun elevation `beta_0`, 3 deg (CSNO 2019).
pub const BEIDOU3_SECM_BETA0_RAD: f64 = 3.0 * PI / 180.0;
/// BeiDou-2 `|beta|` below which yaw steering switches to orbit normal, 4.1 deg.
pub const BEIDOU2_YS_TO_ON_BETA_RAD: f64 = 4.1 * PI / 180.0;
/// BeiDou-2 `|beta|` above which orbit normal switches back to yaw steering,
/// 3.9 deg.
pub const BEIDOU2_ON_TO_YS_BETA_RAD: f64 = 3.9 * PI / 180.0;

/// GPS IIA maximum yaw rates by PRN from Kouba's `eclips.f` (May 2017), in
/// deg/s, as Orekit's `GPSBlockIIA` carries them; index 0 is PRN 1.
const GPS_IIA_YAW_RATES_DEG_S: [f64; 32] = [
    0.1211, 0.1339, 0.1230, 0.1233, 0.1180, 0.1266, 0.1269, 0.1033, 0.1278, 0.0978, 0.2000, 0.1990,
    0.2000, 0.0815, 0.1303, 0.0838, 0.1401, 0.1069, 0.0980, 0.1030, 0.1366, 0.1025, 0.1140, 0.1089,
    0.1001, 0.1227, 0.1194, 0.1260, 0.1228, 0.1165, 0.0969, 0.1140,
];

/// The GPS IIA maximum yaw rate Kouba's `eclips.f` (May 2017) gives for a PRN,
/// in rad/s. The table is by PRN as the PRNs were assigned in 2017.
#[must_use]
pub fn gps_iia_default_yaw_rate_rad_s(prn: u8) -> Option<f64> {
    let index = usize::from(prn).checked_sub(1)?;
    GPS_IIA_YAW_RATES_DEG_S
        .get(index)
        .map(|deg| deg.to_radians())
}

/// A satellite attitude law with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GnssAttitudeLaw {
    /// GPS IIA (Kouba 2009).
    GpsIIA {
        /// Maximum yaw rate, rad/s.
        yaw_rate_rad_s: f64,
        /// Yaw bias, rad. Its sign sets the direction of the shadow crossing
        /// (`+R` for a bias of 0), and a noon turn with beta of its sign and
        /// `|beta| <= |yaw_bias|` (or beta = 0) turns at `sign(yaw_bias) R`.
        yaw_bias_rad: f64,
        /// Sun-satellite angle beyond which the satellite is in shadow.
        shadow_limit_rad: f64,
    },
    /// GPS IIR and IIR-M (Kouba 2009).
    GpsIIR {
        /// Maximum yaw rate, rad/s.
        yaw_rate_rad_s: f64,
    },
    /// GPS IIF (Kouba 2009; Dilssner 2010).
    GpsIIF {
        /// Maximum yaw rate, rad/s.
        yaw_rate_rad_s: f64,
        /// Yaw bias, rad. A noon turn with beta of its sign and
        /// `|beta| <= |yaw_bias|` (or beta = 0) turns at `sign(yaw_bias) R`.
        yaw_bias_rad: f64,
        /// Sun-satellite angle beyond which the satellite is in shadow.
        shadow_limit_rad: f64,
    },
    /// GLONASS-M (Dilssner et al. 2011).
    GlonassM {
        /// Maximum yaw rate, rad/s.
        yaw_rate_rad_s: f64,
        /// Sun-satellite angle beyond which the satellite is in shadow.
        shadow_limit_rad: f64,
    },
    /// Galileo IOV auxiliary-Sun-vector law (Galileo satellite metadata
    /// 3.1.1), also the GPS III law of Montenbruck et al. (2026) with
    /// [`GPS_III_BETA_X_RAD`] and [`GPS_III_BETA_Y_RAD`].
    GalileoIov {
        /// `beta_x`: the law applies while the Sun's along-track component
        /// `|S_x|` is below `sin(beta_x)`, rad.
        beta_x_rad: f64,
        /// `beta_y`: the law applies while `|beta|` is below it, rad; the
        /// auxiliary Sun vector's `|S_y|` is `sin(beta_y)` at noon and
        /// midnight.
        beta_y_rad: f64,
    },
    /// The cosine turn of Galileo FOC and the BeiDou-3 CAST satellites.
    CosineTurn(CosineTurnParameters),
    /// BeiDou-2 IGSO and MEO yaw steering with orbit-normal mode at small
    /// beta.
    BeidouOrbitNormalSwitch {
        /// `|beta|` below which yaw steering switches to orbit normal.
        ys_to_on_beta_rad: f64,
        /// `|beta|` above which orbit normal switches to yaw steering.
        on_to_ys_beta_rad: f64,
    },
    /// BeiDou-3 SECM yaw steering with `beta` held at `+-beta_0` below it.
    FixedBetaYawSteering {
        /// `beta_0`, rad.
        beta0_rad: f64,
    },
    /// A BeiDou-3 MEO satellite of unknown manufacturer: the result where the
    /// CAST and SECM laws agree, otherwise
    /// [`GnssAttitudeError::BeidouManufacturerRequired`].
    Beidou3MeoUnknownManufacturer {
        /// The CAST turn law.
        cast: CosineTurnParameters,
        /// The SECM `beta_0`, rad.
        secm_beta0_rad: f64,
    },
    /// Orbit-normal mode at all times.
    OrbitNormal,
}

impl GnssAttitudeLaw {
    /// The published law of `block` with its published parameters.
    ///
    /// `prn` selects the per-satellite GPS IIA yaw rate; the other blocks
    /// ignore it. A GPS IIA without a PRN, or with one outside the `eclips.f`
    /// table, is [`GnssAttitudeError::NoDefaultYawRate`]; build
    /// [`GnssAttitudeLaw::GpsIIA`] with the rate instead.
    pub fn for_block(block: GnssBlock, prn: Option<u8>) -> Result<Self, GnssAttitudeError> {
        Ok(match block {
            GnssBlock::GpsIIA => Self::GpsIIA {
                yaw_rate_rad_s: prn
                    .and_then(gps_iia_default_yaw_rate_rad_s)
                    .ok_or(GnssAttitudeError::NoDefaultYawRate { prn })?,
                yaw_bias_rad: GPS_IIA_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            },
            GnssBlock::GpsIIR | GnssBlock::GpsIIRM => Self::GpsIIR {
                yaw_rate_rad_s: GPS_IIR_DEFAULT_YAW_RATE_RAD_S,
            },
            GnssBlock::GpsIIF => Self::GpsIIF {
                yaw_rate_rad_s: GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
                yaw_bias_rad: GPS_IIF_DEFAULT_YAW_BIAS_RAD,
                shadow_limit_rad: GPS_SHADOW_LIMIT_RAD,
            },
            GnssBlock::GpsIII => Self::GalileoIov {
                beta_x_rad: GPS_III_BETA_X_RAD,
                beta_y_rad: GPS_III_BETA_Y_RAD,
            },
            GnssBlock::GlonassM => Self::GlonassM {
                yaw_rate_rad_s: GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
                shadow_limit_rad: GLONASS_M_SHADOW_LIMIT_RAD,
            },
            GnssBlock::GalileoIov => Self::GalileoIov {
                beta_x_rad: GALILEO_IOV_BETA_X_RAD,
                beta_y_rad: GALILEO_IOV_BETA_Y_RAD,
            },
            GnssBlock::GalileoFoc => Self::CosineTurn(GALILEO_FOC_TURN),
            GnssBlock::Beidou2Geo | GnssBlock::Beidou3Geo => Self::OrbitNormal,
            GnssBlock::Beidou2Igso | GnssBlock::Beidou2Meo => Self::BeidouOrbitNormalSwitch {
                ys_to_on_beta_rad: BEIDOU2_YS_TO_ON_BETA_RAD,
                on_to_ys_beta_rad: BEIDOU2_ON_TO_YS_BETA_RAD,
            },
            GnssBlock::Beidou3Igso => Self::CosineTurn(BEIDOU3_CAST_IGSO_TURN),
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Cast),
            } => Self::CosineTurn(BEIDOU3_CAST_MEO_TURN),
            GnssBlock::Beidou3Meo {
                manufacturer: Some(BeidouManufacturer::Secm),
            } => Self::FixedBetaYawSteering {
                beta0_rad: BEIDOU3_SECM_BETA0_RAD,
            },
            GnssBlock::Beidou3Meo { manufacturer: None } => Self::Beidou3MeoUnknownManufacturer {
                cast: BEIDOU3_CAST_MEO_TURN,
                secm_beta0_rad: BEIDOU3_SECM_BETA0_RAD,
            },
        })
    }

    fn validate(&self) -> Result<(), GnssAttitudeError> {
        fn positive(field: &'static str, value: f64) -> Result<(), GnssAttitudeError> {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(GnssAttitudeError::InvalidParameter { field, value })
            }
        }
        fn finite(field: &'static str, value: f64) -> Result<(), GnssAttitudeError> {
            if value.is_finite() {
                Ok(())
            } else {
                Err(GnssAttitudeError::InvalidParameter { field, value })
            }
        }
        fn acute(field: &'static str, value: f64) -> Result<(), GnssAttitudeError> {
            if value.is_finite() && value > 0.0 && value < PI / 2.0 {
                Ok(())
            } else {
                Err(GnssAttitudeError::InvalidParameter { field, value })
            }
        }
        fn shadow(field: &'static str, value: f64) -> Result<(), GnssAttitudeError> {
            if value.is_finite() && value > PI / 2.0 && value < PI {
                Ok(())
            } else {
                Err(GnssAttitudeError::InvalidParameter { field, value })
            }
        }
        fn cosine(turn: &CosineTurnParameters) -> Result<(), GnssAttitudeError> {
            acute("beta_limit_rad", turn.beta_limit_rad)?;
            acute("start_angle_rad", turn.start_angle_rad)?;
            positive("period_s", turn.period_s)
        }
        match self {
            Self::GpsIIA {
                yaw_rate_rad_s,
                yaw_bias_rad,
                shadow_limit_rad,
            }
            | Self::GpsIIF {
                yaw_rate_rad_s,
                yaw_bias_rad,
                shadow_limit_rad,
            } => {
                positive("yaw_rate_rad_s", *yaw_rate_rad_s)?;
                finite("yaw_bias_rad", *yaw_bias_rad)?;
                shadow("shadow_limit_rad", *shadow_limit_rad)
            }
            Self::GpsIIR { yaw_rate_rad_s } => positive("yaw_rate_rad_s", *yaw_rate_rad_s),
            Self::GlonassM {
                yaw_rate_rad_s,
                shadow_limit_rad,
            } => {
                positive("yaw_rate_rad_s", *yaw_rate_rad_s)?;
                shadow("shadow_limit_rad", *shadow_limit_rad)
            }
            Self::GalileoIov {
                beta_x_rad,
                beta_y_rad,
            } => {
                acute("beta_x_rad", *beta_x_rad)?;
                acute("beta_y_rad", *beta_y_rad)
            }
            Self::CosineTurn(turn) => cosine(turn),
            Self::BeidouOrbitNormalSwitch {
                ys_to_on_beta_rad,
                on_to_ys_beta_rad,
            } => {
                acute("ys_to_on_beta_rad", *ys_to_on_beta_rad)?;
                acute("on_to_ys_beta_rad", *on_to_ys_beta_rad)
            }
            Self::FixedBetaYawSteering { beta0_rad } => acute("beta0_rad", *beta0_rad),
            Self::Beidou3MeoUnknownManufacturer {
                cast,
                secm_beta0_rad,
            } => {
                cosine(cast)?;
                acute("secm_beta0_rad", *secm_beta0_rad)
            }
            Self::OrbitNormal => Ok(()),
        }
    }
}

/// The Sun-fixed DYB axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DybAxes {
    /// Unit vector from the satellite to the Sun.
    pub d: [f64; 3],
    /// `unit(D x e_r)`, the nominal solar panel axis.
    pub y: [f64; 3],
    /// `D x Y`.
    pub b: [f64; 3],
}

/// The attitude of a satellite at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssAttitude {
    /// The part of the law that produced the attitude.
    pub mode: AttitudeMode,
    /// Yaw angle in `(-pi, pi]` (see the module conventions).
    pub yaw_rad: f64,
    /// Nominal yaw-steering angle at the same instant, in `(-pi, pi]`.
    pub nominal_yaw_rad: f64,
    /// Sun elevation above the orbit plane, positive on the `r x v` side.
    pub beta_rad: f64,
    /// Orbit angle from midnight in the direction of motion, in `(-pi, pi]`.
    pub orbit_angle_rad: f64,
    /// Unit body `+X` axis in the input frame.
    pub body_x: [f64; 3],
    /// Unit body `+Y` axis in the input frame.
    pub body_y: [f64; 3],
    /// Unit body `+Z` axis in the input frame (toward the Earth's centre).
    pub body_z: [f64; 3],
    /// The DYB axes; `None` only when the Sun is exactly along the radial
    /// direction, where `Y` is undefined.
    pub dyb: Option<DybAxes>,
}

/// Nominal yaw steering at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NominalYaw {
    /// Yaw angle in `(-pi, pi]`.
    pub yaw_rad: f64,
    /// Sun elevation above the orbit plane.
    pub beta_rad: f64,
    /// Unit body `+X` axis.
    pub body_x: [f64; 3],
    /// Unit body `+Y` axis.
    pub body_y: [f64; 3],
    /// Unit body `+Z` axis.
    pub body_z: [f64; 3],
}

/// Nominal yaw steering for a satellite at `position_m` with inertial velocity
/// `velocity_m_s` and the Sun at `sun_position_m`, all in one geocentric
/// frame.
///
/// The arithmetic is RTKLIB's `sat_yaw`/`yaw_angle` (nominal branch), which the
/// PPP wind-up correction also uses: `beta` from the Sun and orbit normal, the
/// orbit angle from `acos` of the satellite direction and `sun x normal`, and
/// `yaw = atan2(-tan(beta), sin(mu)) + pi` with RTKLIB's `mu`. At
/// `|beta|, |mu| < 1e-12` RTKLIB's yaw is `pi`. RTKLIB's axes are the IGS axes
/// turned by 180 deg about `Z` (its X axis points away from the Sun), which
/// changes wind-up only by a constant half cycle; this function negates them
/// (exactly) and reports `wrap(yaw_RTKLIB - pi)`, so the result is in the IGS
/// convention of this module.
///
/// Non-finite inputs, zero-length vectors, parallel position and velocity and
/// inputs whose products overflow (`NonFinite` naming the product) are typed
/// errors; a Sun along the orbit normal is
/// [`GnssAttitudeError::DegenerateGeometry`] with no instant.
pub fn nominal_yaw(
    position_m: [f64; 3],
    velocity_m_s: [f64; 3],
    sun_position_m: [f64; 3],
) -> Result<NominalYaw, GnssAttitudeError> {
    geometry::validate_sample(&AttitudeSample {
        position_m,
        velocity_m_s,
        sun_position_m,
    })?;
    let rtklib = geometry::rtklib_nominal_yaw(position_m, velocity_m_s, sun_position_m)
        .ok_or(GnssAttitudeError::DegenerateGeometry { t_s: None })?;
    Ok(NominalYaw {
        yaw_rad: wrap_pi(rtklib.yaw_rad - PI),
        beta_rad: rtklib.beta_rad,
        body_x: neg3(rtklib.x),
        body_y: neg3(rtklib.y),
        body_z: neg3(rtklib.e_r),
    })
}

/// RTKLIB nominal yaw axes `(x, y)` for the PPP corrections, with the
/// arithmetic and its order unchanged and in RTKLIB's orientation (the IGS
/// axes turned by 180 deg about `Z`). `None` where a unit vector is undefined.
pub(crate) fn nominal_yaw_axes_rtklib(
    position: [f64; 3],
    velocity: [f64; 3],
    sun: [f64; 3],
) -> Option<([f64; 3], [f64; 3])> {
    geometry::rtklib_nominal_yaw(position, velocity, sun).map(|yaw| (yaw.x, yaw.y))
}

/// The Sun-fixed DYB axes for a satellite at `position_m` and the Sun at
/// `sun_position_m` (one geocentric frame).
///
/// Non-finite inputs, inputs whose norms or difference overflow, and
/// zero-length vectors are typed errors; a Sun exactly along the radial
/// direction is [`GnssAttitudeError::DegenerateGeometry`] with no instant.
pub fn dyb_axes(
    position_m: [f64; 3],
    sun_position_m: [f64; 3],
) -> Result<DybAxes, GnssAttitudeError> {
    let relative: [f64; 3] = std::array::from_fn(|k| sun_position_m[k] - position_m[k]);
    for (field, vector) in [
        ("position_m", position_m),
        ("sun_position_m", sun_position_m),
        ("sun_position_m - position_m", relative),
    ] {
        if !vector.iter().all(|c| c.is_finite()) || !norm3(vector).is_finite() {
            return Err(GnssAttitudeError::NonFinite { field });
        }
        if norm3(vector) == 0.0 {
            return Err(GnssAttitudeError::ZeroLength { field });
        }
    }
    dyb_from(position_m, sun_position_m).ok_or(GnssAttitudeError::DegenerateGeometry { t_s: None })
}

fn dyb_from(position: [f64; 3], sun: [f64; 3]) -> Option<DybAxes> {
    let e_r = unit3(position)?;
    let d = unit3(sub3(sun, position))?;
    let y = unit3(cross3(d, e_r))?;
    Some(DybAxes {
        d,
        y,
        b: cross3(d, y),
    })
}

/// The attitude of a satellite at `t_s` under `law`, reading the satellite
/// state and Sun position from `trajectory`.
///
/// Errors: invalid law parameters ([`GnssAttitudeError::InvalidParameter`]);
/// a trajectory instant the law needs that the trajectory cannot supply
/// ([`GnssAttitudeError::TrajectoryUnavailable`]); a non-finite or degenerate
/// state from the trajectory; the Sun along the orbit normal; an unresolved
/// BeiDou-3 MEO manufacturer where it matters.
pub fn gnss_attitude<T>(
    law: &GnssAttitudeLaw,
    trajectory: &T,
    t_s: f64,
) -> Result<GnssAttitude, GnssAttitudeError>
where
    T: AttitudeTrajectory + ?Sized,
{
    law.validate()?;
    let sample = geometry::sample_at(trajectory, t_s)?;
    let context = Context::from_sample(t_s, &sample)?;
    let nominal = geometry::rtklib_nominal_yaw(
        sample.position_m,
        sample.velocity_m_s,
        sample.sun_position_m,
    )
    .ok_or(GnssAttitudeError::DegenerateGeometry { t_s: Some(t_s) })?;
    // RTKLIB's nominal axes turned by 180 deg about Z into the IGS convention.
    let nominal_yaw_rad = wrap_pi(nominal.yaw_rad - PI);

    let (mode, yaw_rad, body_x, body_y, body_z) = match laws::evaluate(law, &context, trajectory)? {
        laws::Outcome::Nominal => (
            AttitudeMode::Nominal,
            nominal_yaw_rad,
            neg3(nominal.x),
            neg3(nominal.y),
            neg3(nominal.e_r),
        ),
        laws::Outcome::Yaw { mode, yaw_rad } => {
            let (x, y, z) = context.axes_for_yaw(yaw_rad);
            (mode, wrap_pi(yaw_rad), x, y, z)
        }
    };

    Ok(GnssAttitude {
        mode,
        yaw_rad,
        nominal_yaw_rad,
        beta_rad: context.beta,
        orbit_angle_rad: wrap_pi(context.argument_from_noon() + PI),
        body_x,
        body_y,
        body_z,
        dyb: dyb_from(sample.position_m, sample.sun_position_m),
    })
}

#[cfg(test)]
mod tests;
