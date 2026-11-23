//! Instants solved on the caller's trajectory.
//!
//! Every instant a law uses is where a condition changes, and is found inside
//! a bracket: two instants at which the condition differs, checked before the
//! bracket is bisected, so a bracket without a change is an error and never an
//! answer. The bisection resolves the change to two adjacent representable
//! numbers, taking as its next point the number in the bracket whose position
//! in the order of the representable numbers has the most trailing zero bits.
//! Those points form one fixed binary tree, so brackets of the same change
//! that differ in their ends reach the same cell of the tree before they come
//! near the change, and from there evaluate the same points: the result does
//! not depend on the bracket, also where rounding makes the condition flicker
//! within a few units in the last place of its change.
//!
//! Brackets come from the orbit's own structure. The in-plane argument from
//! noon increases along the orbit, so the instant it passes a value is the one
//! change of "passed" within a bracket that spans less than half a turn of it;
//! such brackets are found by stepping along the trajectory toward the value
//! by at most `pi / 8` of orbit angle per step at the orbital rate where the
//! step starts. "Passed" compares the argument with the value in exact
//! arithmetic (the difference as two numbers that sum to it, against pi as
//! two), so that no argument falls between the half turns that start at
//! `-pi / 2` and `pi / 2`. On a Keplerian orbit of eccentricity `e` the
//! orbital rate is at least `(1 - e)^2 / (1 - e^2)^(3/2)` times the mean
//! motion, so a step lasts at most `pi / 8` over that of mean anomaly, less
//! than an orbital period for `e` below about 0.97: no step turns by a full
//! turn, and a step that appears to move back has turned by more than half a
//! turn and is halved. Where a turn could reach the requested instant, the
//! laws take the orbit's eccentricity ([`Search::orbit_shape`]) and refuse
//! one above 0.9. The steps they take before that (to the quadratures and
//! centre of the instant's half orbit, and back over a cosine turn's last half
//! period) only decide that no turn can reach the instant: for GPS and
//! GLONASS-M only where the conic through the half orbit's quadratures and
//! centre, which lie on the orbit at those angles whatever the steps did,
//! has an eccentricity of at most 0.9. The quadratures, the
//! instants the argument from noon passes
//! `-pi / 2` and `pi / 2`, divide the orbit into half orbits around noon and
//! midnight, each holding one centre.
//!
//! Maxima are searched for in time itself ([`golden_section`]): each point is
//! placed at a fixed fraction of its bracket's length in seconds, so the
//! bracket shrinks the same way wherever zero lies in it, and after a fixed
//! number of points it is below one unit in the last place of the larger end
//! of the first bracket.

use super::geometry::{vector_angle, wrap_pi, Context};
use super::{AttitudeTrajectory, GnssAttitudeError};
use crate::astro::math::vec3::{cross3, dot3, norm3};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_8, PI};

/// pi minus [`PI`], so that `PI + PI_REMAINDER` is pi to about 1e-32.
const PI_REMAINDER: f64 = 1.224_646_799_147_353_2e-16;

/// Signed in-plane angle to noon (`noon == true`) or midnight, positive before
/// it in the direction of motion, in `(-pi, pi]`. Between the two antipodes of
/// that centre (one lap) it decreases from `pi` to `-pi`.
pub(super) fn angle_to_centre(ctx: &Context, noon: bool) -> f64 {
    let u = ctx.argument_from_noon();
    if noon {
        -u
    } else {
        wrap_pi(PI - u)
    }
}

/// `a - b` as two numbers whose sum is exactly `a - b`, the first the rounded
/// difference (Knuth's two-sum).
pub(super) fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let hi = a - b;
    let back = hi - a;
    (hi, (a - (hi - back)) + (-b - back))
}

/// The exact difference `hi + lo` (a pair from [`two_diff`] of numbers
/// within `[-pi, pi]`) compared with pi, taken as `PI + PI_REMAINDER`. With
/// `|lo|` at most half a unit in the last place of `hi`, `hi` decides unless
/// it is [`PI`].
fn compare_with_pi(hi: f64, lo: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if hi < PI {
        Ordering::Less
    } else if hi > PI {
        Ordering::Greater
    } else {
        lo.partial_cmp(&PI_REMAINDER).unwrap_or(Ordering::Equal)
    }
}

/// Whether the satellite has passed the argument from noon `theta` on its way
/// round: the argument from noon lies in `[theta, theta + pi)` modulo 2 pi,
/// decided on the exact difference of the two angles against pi itself (a
/// rounded difference can land on [`PI`] from below it). Every argument has
/// passed `-pi / 2` or `pi / 2` (only `pi / 2` itself has passed both), so
/// every instant lies in a noon or a midnight half orbit. Within a bracket
/// that spans less than half a turn and holds `theta`, this changes once, at
/// `theta`.
pub(super) fn passed(ctx: &Context, theta: f64) -> bool {
    passed_angle(ctx.argument_from_noon(), theta)
}

/// [`passed`] for the argument from noon `u`.
fn passed_angle(u: f64, theta: f64) -> bool {
    use std::cmp::Ordering;
    // Both angles lie in [-pi, pi], so the difference lies in (-2 pi, 2 pi):
    // passed for a difference in [0, pi) or in (-2 pi, -pi).
    let (hi, lo) = two_diff(u, theta);
    if hi >= 0.0 {
        compare_with_pi(hi, lo) == Ordering::Less
    } else {
        compare_with_pi(-hi, -lo) == Ordering::Greater
    }
}

/// Whether `x - y > c` exactly, for finite `x`, `y` and `c`: the time from
/// `y` to `x` exceeds `c` also where `x` and `y` are too large for `c` to be
/// added to `y` without rounding.
pub(super) fn exceeds(x: f64, y: f64, c: f64) -> bool {
    let (hi, lo) = two_diff(x, y);
    hi > c || (hi == c && lo > 0.0)
}

/// The latest representable instant at most `c` after `t` (exactly): `t + c`
/// rounded down where rounding took it beyond.
pub(super) fn latest_within(t: f64, c: f64) -> f64 {
    let sum = t + c;
    if exceeds(sum, t, c) {
        sum.next_down()
    } else {
        sum
    }
}

/// Position of `x` in the order of the representable numbers, with both zeros
/// at 0: adjacent numbers have adjacent keys.
fn key(x: f64) -> i64 {
    // The bit pattern without the sign is below 2^63.
    let magnitude = i64::try_from(x.to_bits() & !(1_u64 << 63)).unwrap_or(i64::MAX);
    if x.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// The number at a key.
fn from_key(k: i64) -> f64 {
    let magnitude = f64::from_bits(k.unsigned_abs());
    if k < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// The key strictly between `a` and `b` (`a + 1 < b`) with the most trailing
/// zero bits: 0 when the interval holds it, otherwise the number of the form
/// `m 2^h` in it with the largest `h`, which is unique.
fn coarsest_between(a: i64, b: i64) -> i64 {
    if a < 0 && b > 0 {
        return 0;
    }
    if b <= 0 {
        return -coarsest_between(-b, -a);
    }
    let (lo, hi) = (a + 1, b - 1);
    if lo == hi {
        return lo;
    }
    // The highest bit where the candidates differ: `hi` has it set and `lo`
    // clear, so `hi` with the bits below cleared lies in `[lo, hi]`; only `lo`
    // itself can have more trailing zeros.
    let h = 63 - (lo ^ hi).leading_zeros();
    if lo.trailing_zeros() > h {
        lo
    } else {
        (hi >> h) << h
    }
}

/// Most steps of the search for a bracket of an argument-from-noon crossing.
const CROSSING_SEARCH_STEPS: u32 = 4096;

/// Fraction of the larger side at which the search for a maximum places its
/// next point, `(3 - sqrt(5)) / 2` (golden-section search).
const GOLDEN_SECTION: f64 = 0.381_966_011_250_105_1;

/// Most points of the search for a maximum. Its bracket shrinks to about
/// 0.618 of its length per point, so after 128 points one that began at most
/// twice the larger magnitude of its ends long is below `2^-60` of that
/// magnitude, under one unit in its last place.
const MAXIMUM_SEARCH_STEPS: u32 = 128;

/// The best point of a golden-section search for the largest value of
/// `value` on `[lo, hi]`, and its value. The ends are evaluated first; then
/// each point lies in the larger side of the bracket, at [`GOLDEN_SECTION`]
/// of that side's length in time from the best point so far (`first` before
/// any, when inside), so the bracket shrinks in proportion to its length in
/// seconds wherever zero lies in it. The search stops at the first point
/// where `done` holds (that point is returned), after
/// [`MAXIMUM_SEARCH_STEPS`] points, or when no number lies strictly inside
/// the larger side. A NaN value counts as the lowest.
///
/// Where `value` rises to one maximum on `[lo, hi]` and falls after it, the
/// bracket always holds that maximum: of two points with different values
/// the lower lies on the far side of the maximum from the higher, and of two
/// with equal values the maximum lies between them unless `value` is
/// constant to its last bit from one to the other, at least a fifth of a
/// bracket that holds the maximum apart: a margin formed from an orbit is
/// that flat only over spans so short that it changes by a few roundings
/// across the whole bracket.
pub(super) fn golden_section<V, D>(
    lo: f64,
    hi: f64,
    first: f64,
    value: &V,
    done: &D,
) -> Result<(f64, f64), GnssAttitudeError>
where
    V: Fn(f64) -> Result<f64, GnssAttitudeError>,
    D: Fn(f64) -> bool,
{
    let eval = |t: f64| -> Result<f64, GnssAttitudeError> {
        let v = value(t)?;
        Ok(if v.is_nan() { f64::NEG_INFINITY } else { v })
    };
    let mut best = (lo, eval(lo)?);
    if done(best.1) || lo >= hi {
        return Ok(best);
    }
    let at_hi = eval(hi)?;
    if done(at_hi) {
        return Ok((hi, at_hi));
    }
    if at_hi > best.1 {
        best = (hi, at_hi);
    }
    let (mut a, mut b) = (lo, hi);
    let mut x = if a < first && first < b {
        first
    } else {
        0.5 * a + 0.5 * b
    };
    if !(a < x && x < b) {
        return Ok(best);
    }
    let mut fx = eval(x)?;
    if done(fx) {
        return Ok((x, fx));
    }
    for _ in 0..MAXIMUM_SEARCH_STEPS {
        // Half lengths, which do not overflow for finite ends.
        let (left, right) = (0.5 * x - 0.5 * a, 0.5 * b - 0.5 * x);
        let y = if right >= left {
            x + (2.0 * GOLDEN_SECTION) * right
        } else {
            x - (2.0 * GOLDEN_SECTION) * left
        };
        if !(a < y && y < b) || y == x {
            break;
        }
        let fy = eval(y)?;
        if done(fy) {
            return Ok((y, fy));
        }
        if fy > fx {
            if y > x {
                a = x;
            } else {
                b = x;
            }
            (x, fx) = (y, fy);
        } else if y > x {
            b = y;
        } else {
            a = y;
        }
    }
    Ok(if fx > best.1 { (x, fx) } else { best })
}

/// Smallest determinant of the fit of [`conic_eccentricity`] that fixes the
/// conic: about 2 for the angles a quarter and half a turn apart it is
/// given.
const CONIC_FIT_DETERMINANT: f64 = 0.25;

/// The conic about the Earth's centre through the positions of `c0`, `c1`
/// and `c2`: with in-plane angles `t_i` from `c0`'s position about its orbit
/// normal, `|r_0| / |r_i| = a + b cos(t_i) + c sin(t_i)` (a conic with a
/// focus at the origin), solved for `a`, `b`, `c`. Returns `a` and
/// `sqrt(b^2 + c^2)`, so the eccentricity is their quotient and the least
/// radius `|r_0|` over their sum; `None` where the three angles are too close
/// together to fix the conic or the fit is not an ellipse's.
fn conic_fit(c0: &Context, c1: &Context, c2: &Context) -> Option<(f64, f64)> {
    let angle = |c: &Context| libm::atan2(dot3(cross3(c0.e_r, c.e_r), c0.e_n), dot3(c0.e_r, c.e_r));
    let r0 = norm3(c0.position);
    let rows = [
        [1.0, 1.0, 0.0, 1.0],
        [
            1.0,
            libm::cos(angle(c1)),
            libm::sin(angle(c1)),
            r0 / norm3(c1.position),
        ],
        [
            1.0,
            libm::cos(angle(c2)),
            libm::sin(angle(c2)),
            r0 / norm3(c2.position),
        ],
    ];
    let det = |k: [usize; 3]| {
        let m = |i: usize, j: usize| rows[i][k[j]];
        m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
            - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
            + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
    };
    let d = det([0, 1, 2]);
    // Also a NaN determinant.
    if d.abs() < CONIC_FIT_DETERMINANT || d.is_nan() {
        return None;
    }
    let (a, b, c) = (det([3, 1, 2]) / d, det([0, 3, 2]) / d, det([0, 1, 3]) / d);
    let spread = libm::hypot(b, c);
    (a > 0.0 && (spread / a).is_finite()).then_some((a, spread))
}

/// The eccentricity of the conic of [`conic_fit`] through the positions of
/// `c0`, `c1` and `c2`; infinite where the fit fails.
fn conic_eccentricity(c0: &Context, c1: &Context, c2: &Context) -> f64 {
    conic_fit(c0, c1, c2).map_or(f64::INFINITY, |(a, spread)| spread / a)
}

/// The orbit's eccentricity and the motion of its plane over the turn of
/// orbit angle before an instant ([`Search::orbit_shape`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct OrbitShape {
    /// The eccentricity of the conic through the three positions
    /// ([`conic_eccentricity`]).
    pub(super) eccentricity: f64,
    /// The largest change of beta from the instant to the positions a
    /// quarter, a half, three quarters and a full turn before it.
    pub(super) beta_change: f64,
    /// The largest angle between the orbit normal at the instant and at those
    /// positions.
    pub(super) plane_turn: f64,
}

/// The caller's trajectory and the requested instant (the instant reported by
/// errors).
pub(super) struct Search<'a, T: AttitudeTrajectory + ?Sized> {
    trajectory: &'a T,
    t_query: f64,
}

impl<T: AttitudeTrajectory + ?Sized> Clone for Search<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: AttitudeTrajectory + ?Sized> Copy for Search<'_, T> {}

impl<'a, T: AttitudeTrajectory + ?Sized> Search<'a, T> {
    pub(super) fn new(trajectory: &'a T, t_query: f64) -> Self {
        Self {
            trajectory,
            t_query,
        }
    }

    pub(super) fn trajectory(&self) -> &'a T {
        self.trajectory
    }

    /// The geometry at `t`.
    pub(super) fn at(&self, t: f64) -> Result<Context, GnssAttitudeError> {
        Context::at(self.trajectory, t)
    }

    /// A boundary the law needs could not be found.
    pub(super) fn not_found(&self) -> GnssAttitudeError {
        GnssAttitudeError::TurnEndNotFound { t_s: self.t_query }
    }

    /// The input lies outside the domain on which the law's search is shown
    /// to be correct.
    pub(super) fn outside(&self, reason: &'static str) -> GnssAttitudeError {
        GnssAttitudeError::OutsideValidatedDomain {
            t_s: self.t_query,
            reason,
        }
    }

    /// Adjacent numbers `(a, b)`, `lo <= a < b <= hi`, with `f(a) == f(lo)`
    /// and `f(b) == f(hi)`, for a bracket whose ends differ: checked here, and
    /// [`GnssAttitudeError::TurnEndNotFound`] where they do not. Every bracket
    /// whose ends lie outside the units in the last place where `f` flickers
    /// around one change gives the same numbers (see the module notes).
    pub(super) fn transition<F>(
        &self,
        f: &F,
        lo: f64,
        hi: f64,
    ) -> Result<(f64, f64), GnssAttitudeError>
    where
        F: Fn(f64) -> Result<bool, GnssAttitudeError>,
    {
        if lo >= hi || lo.is_nan() || hi.is_nan() {
            return Err(self.not_found());
        }
        let at_lo = f(lo)?;
        if f(hi)? == at_lo {
            return Err(self.not_found());
        }
        let (mut a, mut b) = (key(lo), key(hi));
        while a + 1 < b {
            let c = coarsest_between(a, b);
            if f(from_key(c))? == at_lo {
                a = c;
            } else {
                b = c;
            }
        }
        Ok((from_key(a), from_key(b)))
    }

    /// The first instant at which the satellite has passed the argument from
    /// noon `theta` (see [`passed`]), searched from `from` forward
    /// (`forward == true`; `from` is at most half a turn before `theta`) or
    /// backward (`from` has passed `theta` by at most half a turn, the
    /// instant then being the last crossing at or before `from`). `None` when
    /// the trajectory's representable time ends first.
    ///
    /// The search steps toward `theta` by at most `pi / 8` of orbit angle at
    /// the orbital rate where each step starts, doubling while the angle moves
    /// the way it should and halving where a step lands more than half a turn
    /// on (the angle then appears to move back). The first step whose ends
    /// differ in "passed" brackets the crossing; the side of `from` itself is
    /// taken from the direction, so `from` may lie half a turn from `theta`.
    /// A search that has not bracketed the crossing after 4096 steps is
    /// [`GnssAttitudeError::OutsideValidatedDomain`]: on an orbit the angle
    /// turns by a sizeable fraction of `pi / 8` per step, so this is no
    /// orbit.
    pub(super) fn crossing(
        &self,
        from: &Context,
        theta: f64,
        forward: bool,
    ) -> Result<Option<f64>, GnssAttitudeError> {
        let Some((near, far)) = self.step_across(from, theta, forward, None)? else {
            return Ok(None);
        };
        let (lo, hi) = if forward {
            (near, far.t_s)
        } else {
            (far.t_s, near)
        };
        let crossed =
            |x: f64| -> Result<bool, GnssAttitudeError> { Ok(passed(&self.at(x)?, theta)) };
        Ok(Some(self.transition(&crossed, lo, hi)?.1))
    }

    /// The last crossing of `theta` at or before `from` ([`Search::crossing`]
    /// backward), if it lies at or after `earliest`: the search steps back no
    /// further than `earliest` (its last step ends there), so the trajectory
    /// is read at no instant before it. `None` where `[earliest, from]` holds
    /// no crossing.
    pub(super) fn crossing_since(
        &self,
        from: &Context,
        theta: f64,
        earliest: f64,
    ) -> Result<Option<f64>, GnssAttitudeError> {
        let Some((near, far)) = self.step_across(from, theta, false, Some(earliest))? else {
            return Ok(None);
        };
        let crossed =
            |x: f64| -> Result<bool, GnssAttitudeError> { Ok(passed(&self.at(x)?, theta)) };
        Ok(Some(self.transition(&crossed, far.t_s, near)?.1))
    }

    /// The step of the search of [`Search::crossing`] across `theta`: the
    /// instant it starts from and the geometry where it ends. A backward
    /// search with a `floor` ends its last step at the floor and is `None`
    /// where no step up to it crosses.
    fn step_across(
        &self,
        from: &Context,
        theta: f64,
        forward: bool,
        floor: Option<f64>,
    ) -> Result<Option<(f64, Context)>, GnssAttitudeError> {
        let (mut t, mut u) = (from.t_s, from.argument_from_noon());
        let mut was_passed = !forward;
        let mut step = FRAC_PI_8 / from.mu_rate;
        for _ in 0..CROSSING_SEARCH_STEPS {
            let mut next = if forward { t + step } else { t - step };
            let mut at_floor = false;
            if let Some(floor) = floor.filter(|_| !forward) {
                if next <= floor {
                    if t <= floor {
                        return Ok(None);
                    }
                    (next, at_floor) = (floor, true);
                }
            }
            if !next.is_finite() {
                return Ok(None);
            }
            if next == t {
                step *= 2.0;
                continue;
            }
            let ctx = self.at(next)?;
            let next_u = ctx.argument_from_noon();
            let moved = if forward {
                wrap_pi(next_u - u)
            } else {
                wrap_pi(u - next_u)
            };
            if moved < 0.0 {
                step *= 0.5;
                continue;
            }
            let next_passed = passed(&ctx, theta);
            if next_passed != was_passed {
                return Ok(Some((t, ctx)));
            }
            if at_floor {
                return Ok(None);
            }
            (t, u, was_passed) = (next, next_u, next_passed);
            step = (2.0 * step).min(FRAC_PI_8 / ctx.mu_rate);
        }
        Err(self
            .outside("the in-plane angle did not reach the value searched for within 4096 steps"))
    }

    /// The eccentricity of the orbit, as the conic through the satellite's
    /// positions at `ctx` and about a quarter and half a turn of in-plane
    /// angle before it ([`conic_eccentricity`]): the step of the search of
    /// [`Search::crossing`] back across each angle, halved eight times
    /// (so within `pi / 256` of it on an orbit). Only the trajectory's
    /// positions enter, so it is the eccentricity of a Keplerian orbit
    /// whatever its gravitational parameter and scale; on an orbit far beyond
    /// the validated eccentricity, where a step can turn by a full turn, the
    /// positions still lie on the orbit. A plane that turns by an angle
    /// `delta` over that half turn biases the fit by up to about
    /// `0.04 delta^2`. With it, the changes of beta and of the orbit normal
    /// from the positions a quarter, a half, three quarters and a full turn
    /// before `ctx` (the full turn lasts one orbital period wherever `ctx`
    /// lies, where half a turn around perigee can last a small part of it).
    pub(super) fn orbit_shape(&self, ctx: &Context) -> Result<OrbitShape, GnssAttitudeError> {
        let u = ctx.argument_from_noon();
        let quarter = self.near_crossing(ctx, wrap_pi(u - FRAC_PI_2))?;
        let half = self.near_crossing(&quarter, wrap_pi(u - PI))?;
        let three_quarters = self.near_crossing(&half, wrap_pi(u + FRAC_PI_2))?;
        let full = self.near_crossing(&three_quarters, u)?;
        let earlier = [quarter, half, three_quarters, full];
        Ok(OrbitShape {
            eccentricity: conic_eccentricity(ctx, &quarter, &half),
            beta_change: earlier
                .iter()
                .fold(0.0_f64, |m, c| m.max((c.beta - ctx.beta).abs())),
            plane_turn: earlier
                .iter()
                .fold(0.0_f64, |m, c| m.max(vector_angle(ctx.e_n, c.e_n))),
        })
    }

    /// The geometry near the last crossing of `theta` before `from` (which
    /// has passed it by at most half a turn): the step back across it,
    /// halved eight times.
    fn near_crossing(&self, from: &Context, theta: f64) -> Result<Context, GnssAttitudeError> {
        let (near, mut far) = self.step_across(from, theta, false, None)?.ok_or_else(|| {
            self.outside("the trajectory does not turn by half a turn before the instant")
        })?;
        let mut later = near;
        for _ in 0..8 {
            let mid = 0.5 * far.t_s + 0.5 * later;
            if mid <= far.t_s || mid >= later {
                break;
            }
            let ctx = self.at(mid)?;
            if passed(&ctx, theta) {
                later = mid;
            } else {
                far = ctx;
            }
        }
        Ok(far)
    }

    /// The eccentricity and the largest orbital rate of the conic
    /// ([`conic_fit`]) through the satellite's positions at the start,
    /// centre and end of `half` (a quarter turn apart; on the orbit at those
    /// angles also where a step of the search turned by a full turn), the
    /// rate at the angular momentum of `ctx`: `h / r_min^2`, with
    /// `h = mu_rate |r|^2` at `ctx` (constant along a Keplerian orbit) and
    /// `r_min` the conic's least radius. Both infinite where the fit fails.
    pub(super) fn orbit_peak(
        &self,
        ctx: &Context,
        half: &HalfOrbit,
    ) -> Result<(f64, f64), GnssAttitudeError> {
        let start = self.at(half.start)?;
        let fit = conic_fit(&start, &self.at(half.centre)?, &self.at(half.end)?);
        let Some((a, spread)) = fit else {
            return Ok((f64::INFINITY, f64::INFINITY));
        };
        let ratio = norm3(ctx.position) * (a + spread) / norm3(start.position);
        let peak = ctx.mu_rate * ratio * ratio;
        Ok((
            spread / a,
            if peak.is_finite() {
                peak
            } else {
                f64::INFINITY
            },
        ))
    }

    /// An instant of `[lo, hi]` with `margin > 0`, or `None` when the
    /// golden-section search for the largest margin ([`golden_section`]),
    /// from the ends and then from `first` (taken first, since a zone usually
    /// holds it), finds none. Where the margin rises to one maximum on
    /// `[lo, hi]` and falls after it, `None` means the margin is positive at
    /// no instant the search can resolve, below one unit in the last place of
    /// the larger end of `[lo, hi]`.
    pub(super) fn find_inside<M>(
        &self,
        lo: f64,
        hi: f64,
        first: f64,
        margin: &M,
    ) -> Result<Option<f64>, GnssAttitudeError>
    where
        M: Fn(&Context) -> Result<f64, GnssAttitudeError>,
    {
        let value = |t: f64| margin(&self.at(t)?);
        let (t, best) = golden_section(lo, hi, first, &value, &|v: f64| v > 0.0)?;
        Ok((best > 0.0).then_some(t))
    }
}

/// The argument from noon at which a noon (`noon == true`) or midnight half
/// orbit begins, `-pi / 2` or `pi / 2`.
fn half_orbit_entry(noon: bool) -> f64 {
    if noon {
        -FRAC_PI_2
    } else {
        FRAC_PI_2
    }
}

/// The argument from noon of noon (0) or midnight (`pi`).
fn centre_argument(noon: bool) -> f64 {
    if noon {
        0.0
    } else {
        PI
    }
}

/// One half orbit: from the quadrature before noon (`noon == true`) or
/// midnight to the quadrature after it, solved on the trajectory. The
/// instants from `start` up to (not including) `end` have passed the entry
/// quadrature and not the exit one; `centre` is the first instant past noon or
/// midnight.
#[derive(Clone, Copy, Debug)]
pub(super) struct HalfOrbit {
    pub(super) noon: bool,
    pub(super) start: f64,
    pub(super) centre: f64,
    pub(super) end: f64,
}

impl HalfOrbit {
    /// Whether the instant of `ctx` lies in a noon half orbit: past the
    /// argument from noon `-pi / 2` and not past `pi / 2`.
    pub(super) fn is_noon(ctx: &Context) -> bool {
        passed(ctx, -FRAC_PI_2) && !passed(ctx, FRAC_PI_2)
    }

    /// The half orbit holding the instant of `ctx`.
    pub(super) fn containing<T: AttitudeTrajectory + ?Sized>(
        search: &Search<'_, T>,
        ctx: &Context,
    ) -> Result<Self, GnssAttitudeError> {
        let noon = Self::is_noon(ctx);
        let centre_theta = centre_argument(noon);
        let found = |instant: Option<f64>| instant.ok_or_else(|| search.not_found());
        Ok(Self {
            noon,
            start: found(search.crossing(ctx, half_orbit_entry(noon), false)?)?,
            centre: found(search.crossing(ctx, centre_theta, !passed(ctx, centre_theta))?)?,
            end: found(search.crossing(ctx, half_orbit_entry(!noon), true)?)?,
        })
    }

    /// The half orbit before this one: its centre and start searched back
    /// from this one's start, each a quarter turn from where its search
    /// begins.
    pub(super) fn previous<T: AttitudeTrajectory + ?Sized>(
        &self,
        search: &Search<'_, T>,
    ) -> Result<Self, GnssAttitudeError> {
        let noon = !self.noon;
        let found = |instant: Option<f64>| instant.ok_or_else(|| search.not_found());
        let centre =
            found(search.crossing(&search.at(self.start)?, centre_argument(noon), false)?)?;
        let start = found(search.crossing(&search.at(centre)?, half_orbit_entry(noon), false)?)?;
        Ok(Self {
            noon,
            start,
            centre,
            end: self.start,
        })
    }

    /// The half orbit after this one.
    pub(super) fn next<T: AttitudeTrajectory + ?Sized>(
        &self,
        search: &Search<'_, T>,
    ) -> Result<Self, GnssAttitudeError> {
        let noon = !self.noon;
        let found = |instant: Option<f64>| instant.ok_or_else(|| search.not_found());
        let centre = found(search.crossing(&search.at(self.end)?, centre_argument(noon), true)?)?;
        let end = found(search.crossing(&search.at(centre)?, half_orbit_entry(!noon), true)?)?;
        Ok(Self {
            noon,
            start: self.end,
            centre,
            end,
        })
    }

    /// The last instant of the half orbit.
    pub(super) fn last(&self) -> f64 {
        self.end.next_down()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarsest_keys_have_the_most_trailing_zeros() {
        assert_eq!(coarsest_between(-5, 7), 0);
        assert_eq!(coarsest_between(0, 2), 1);
        assert_eq!(coarsest_between(7, 13), 8);
        assert_eq!(coarsest_between(8, 13), 12);
        assert_eq!(coarsest_between(-13, -7), -8);
        assert_eq!(coarsest_between(-13, 0), -8);
        for (a, b) in [(3_i64, 100_i64), (64, 65 + 64), (1 << 40, (1 << 41) + 3)] {
            let c = coarsest_between(a, b);
            assert!(a < c && c < b);
            for other in a + 1..b.min(a + 10_000) {
                assert!(
                    other.trailing_zeros() <= c.trailing_zeros(),
                    "{a} {b} {other}"
                );
            }
        }
        assert_eq!(from_key(key(-0.0)), 0.0);
        assert_eq!(
            key(f64::MIN_POSITIVE) - key(f64::MIN_POSITIVE.next_down()),
            1
        );
        assert_eq!(key(-1.5).abs(), key(1.5));
        assert_eq!(from_key(key(-2.75)), -2.75);
    }

    /// A bracket whose ends do not differ is an error, not an answer.
    #[test]
    fn brackets_without_a_change_are_errors() {
        let trajectory = |_: f64| None;
        let search = Search::new(&trajectory, 7.0);
        let f = |t: f64| -> Result<bool, GnssAttitudeError> { Ok(t > 5.0 && t < 6.0) };
        assert_eq!(
            search.transition(&f, 0.0, 10.0),
            Err(GnssAttitudeError::TurnEndNotFound { t_s: 7.0 })
        );
        assert_eq!(
            search.transition(&f, 3.0, 3.0),
            Err(GnssAttitudeError::TurnEndNotFound { t_s: 7.0 })
        );
        let (a, b) = search.transition(&f, 0.0, 5.5).unwrap();
        assert_eq!((a, b), (5.0, 5.0f64.next_up()));
    }

    /// A condition that flickers within a few units in the last place of its
    /// change resolves to the same adjacent numbers from every bracket whose
    /// ends lie outside the flicker, here ten brackets each around changes
    /// near 1234.5 (a number with many trailing zero bits), 1234.567891 and 0,
    /// also across zero and many binades, each in at most 66 evaluations.
    #[test]
    fn transitions_do_not_depend_on_the_bracket() {
        let trajectory = |_: f64| None;
        let search = Search::new(&trajectory, 0.0);
        for centre in [1234.5_f64, 1_234.567_891, 0.0] {
            let base = key(centre);
            // Flicker: true at and after the change, except at a few keys just
            // after it, and true at a few keys just before.
            let f = |t: f64| -> Result<bool, GnssAttitudeError> {
                let k = i128::from(key(t)) - i128::from(base);
                Ok(match k {
                    -3 | -1 => true,
                    1 | 2 | 4 => false,
                    _ => k >= 0,
                })
            };
            let mut results = std::collections::BTreeSet::new();
            for (lo, hi) in [
                (centre - 600.0, centre + 1.0),
                (centre - 1e-9, centre + 1e-9),
                (centre - 3.0, centre + 700.0),
                (centre - 1.0e5, centre + 0.25),
                (-1.0e300, centre + 1.0e300),
                (centre - 0.5, centre + 1e-6),
                (centre - 59.9, centre + 0.1),
                (centre - 60.1, centre + 59.0),
                (centre - 1e-3, centre + 1e3),
                (centre - 17.0, centre + 17.0),
            ] {
                let calls = std::cell::Cell::new(0_u32);
                let counted = |t: f64| {
                    calls.set(calls.get() + 1);
                    f(t)
                };
                let (a, b) = search.transition(&counted, lo, hi).unwrap();
                assert!(calls.get() <= 66, "{lo} {hi}: {} evaluations", calls.get());
                assert_eq!(key(b) - key(a), 1);
                results.insert((a.to_bits(), b.to_bits()));
            }
            assert_eq!(results.len(), 1, "centre {centre}: {results:?}");
        }
    }

    /// The search for a maximum places its points in time, so it finds a
    /// zone (a positive value) of a function that rises to one maximum and
    /// falls after it wherever zero lies in the bracket: peaks from -3000 s
    /// to 5000 s, zones from 1 ms to 200 s wide, brackets across zero in
    /// several positions and on one side of it, the first point 200 s before
    /// zero (where the value is below its value at zero, as in a half orbit
    /// with noon at -200 s), also with the value constant to its last bit for
    /// `|t|` below 1e-9 (an orbit's arithmetic does not resolve such times),
    /// and the same zones moved by any time. A function that is nowhere
    /// positive gives its best value, not a positive one.
    #[test]
    fn maxima_are_found_on_either_side_of_zero() {
        let brackets = [
            (-15_020.2, 6_518.3),
            (-1.0e4, 1.0e4),
            (-0.5, 1.0e4),
            (-1.0e4, 0.5),
            (-1e-300, 2.0e4),
            (-6_000.0, 1e-300),
            (3.0, 9_000.0),
            (-9_000.0, -3.0),
        ];
        let mut found = 0;
        for peak in [
            -3000.0, -1000.0, -199.0, -1.5, -1e-3, 0.0, 1e-3, 2.0, 185.6, 999.0, 1000.0, 5000.0,
        ] {
            for half_width in [1e-3, 0.4, 1.0, 891.6] {
                for flat_below in [0.0, 1e-9] {
                    for shift in [0.0, 7.25, -1.0e5] {
                        let value = |t: f64| -> Result<f64, GnssAttitudeError> {
                            let t = t - shift;
                            let t = if t.abs() < flat_below { 0.0 } else { t };
                            // Steeper on the left, a long tail on the right.
                            let x = t - peak;
                            let fall = if x < 0.0 { 3.0 * x * x } else { x * x };
                            Ok(half_width * half_width - fall)
                        };
                        for (lo, hi) in brackets {
                            let (lo, hi) = (lo + shift, hi + shift);
                            if !(lo < peak + shift && peak + shift < hi) {
                                continue;
                            }
                            let first = -200.0 + shift;
                            let (t, v) =
                                golden_section(lo, hi, first, &value, &|v: f64| v > 0.0).unwrap();
                            assert!(
                                v > 0.0 && value(t).unwrap() > 0.0,
                                "peak {peak} width {half_width} flat {flat_below} shift {shift} bracket {lo}..{hi}: best {t} {v}"
                            );
                            found += 1;
                            let below = |t: f64| -> Result<f64, GnssAttitudeError> {
                                Ok(value(t)? - 2.0 * half_width * half_width)
                            };
                            let (t, v) =
                                golden_section(lo, hi, first, &below, &|v: f64| v > 0.0).unwrap();
                            assert!(v <= 0.0 && (t - peak - shift).abs() < 1e-3 * (hi - lo));
                        }
                    }
                }
            }
        }
        assert!(found > 500, "{found}");
    }

    /// "Passed" is decided on the exact difference: the last number below
    /// `pi / 2` has passed `-pi / 2` (its rounded difference is [`PI`]) and
    /// not `pi / 2`, `pi / 2` has passed both, and every argument has passed
    /// one of the two quadratures; [`PI`] (below pi) has passed 0.
    #[test]
    fn passed_is_exact_at_the_quadratures() {
        let f = FRAC_PI_2;
        assert!(passed_angle(f.next_down(), -f));
        assert!(!passed_angle(f.next_down(), f));
        assert!(passed_angle(f, -f) && passed_angle(f, f));
        assert!(!passed_angle(f.next_up(), -f) && passed_angle(f.next_up(), f));
        assert!(passed_angle(-f, -f) && !passed_angle(-f.next_up(), -f));
        assert!(passed_angle(PI, 0.0) && passed_angle(PI, PI));
        assert!(!passed_angle(-PI, 0.0));
        let mut u = -PI;
        let mut count = 0;
        while u <= PI {
            for v in [u, u.next_up(), u.next_down()] {
                assert!(passed_angle(v, -f) || passed_angle(v, f), "{v}");
            }
            u += 1e-3;
            count += 1;
        }
        assert!(count > 6000);
        for v in [
            f.next_down(),
            f,
            f.next_up(),
            -f.next_down(),
            -f,
            -f.next_up(),
            PI,
            -PI,
        ] {
            assert!(passed_angle(v, -f) || passed_angle(v, f), "{v}");
        }
    }

    /// Time margins are taken exactly: the last instant within 1800 s of
    /// 8e18 (instants 1024 s apart) is 8e18 + 1024, the sum rounding to
    /// + 2048; near 2e19 (4096 s apart) no instant after it is within.
    #[test]
    fn time_margins_are_exact() {
        assert_eq!(latest_within(8.0e18, 1800.0), 8.0e18 + 1024.0);
        assert_eq!(latest_within(2.0e19, 1800.0), 2.0e19);
        assert_eq!(latest_within(100.0, 1800.0), 1900.0);
        assert!(exceeds(8.0e18 + 2048.0, 8.0e18, 1800.0));
        assert!(!exceeds(8.0e18 + 1024.0, 8.0e18, 1800.0));
        assert!(!exceeds(1900.0, 100.0, 1800.0));
        assert!(exceeds(1900.0_f64.next_up(), 100.0, 1800.0));
        assert!(exceeds(1.0e300, -1.0e300, 1800.0));
    }
}
