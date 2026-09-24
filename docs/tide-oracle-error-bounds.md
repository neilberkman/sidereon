# Tide oracle error bounds

The Chapter 6 comparisons use independently generated Orekit 13.1.8
coefficients. Their bounds depend on the input arguments, constituent
amplitudes, and floating-point operation counts, rather than observed residuals.

Write `u = 2^-53` and `gamma(n) = n u / (1 - n u)`. The two-sided formulas
sum the operation bounds for each implementation; a factor of two is used only
where both sides have the same operation budget.

For Step 2, forming a constituent phase takes six multiplications and five
additions. Its absolute error is bounded by `gamma(11)` times the sum of
the absolute products. Sine and cosine are Lipschitz with constant one.
Multiplying this phase error by the sum of the absolute in-phase and
out-of-phase amplitudes bounds its effect on each coefficient. The bound
also includes supplied argument errors, amplitude conversion, products,
addition, and summation across all constituents. The `4 epsilon` trig
allowance is conditional on treating each sine/cosine result in both
implementations as having absolute error no greater than `2u`. The pinned
Rust dependency is `libm = 0.2.16`; its source comments describe `sin` and
`cos` as “nearly rounded,” but provide no numerical maximum. See the [pinned
`sin` source](https://docs.rs/libm/0.2.16/src/libm/math/sin.rs.html) and
[`cos` source](https://docs.rs/libm/0.2.16/src/libm/math/cos.rs.html). The
Orekit oracle and the IERS Fortran oracle also use their respective runtime
trigonometric implementations, for which these bounds do not establish a
numeric error guarantee. Thus `2u` per result is an explicit working
assumption, not a consequence guaranteed by the cited library documentation;
the resulting tolerances are conditional numerical envelopes, not formal
portable upper bounds. The oracle comparisons exercise trig in the full
calculation but do not isolate trig error against an independent
high-precision reference.

When each program derives its own arguments, the fixture is restricted to
one Julian century around J2000. The absolute coefficient sum of any
Delaunay polynomial is bounded by `10 + 8500 |t|` radians on that interval:
the largest linear coefficient is 1,739,527,262.8478 arcseconds per century,
and the remaining terms fit within the constant allowance. For a Delaunay
argument, the Rust path has 13 rounded operations: three to form `t`, eight
for four Horner multiply-add stages, one signed remainder, and one radians
conversion. In Orekit 13.1.8, the fixture makes two `AbsoluteDate.shiftedBy`
calls; each date-offset TwoSum path performs eight rounded operations. The
date difference and century conversion add three, and the degree-four
polynomial adds eight, for `gamma(27)` total. Thus the Delaunay producer bound
uses `gamma(13) + gamma(27)`. The GMST Rust path counts 31 rounded operations
through ERA reduction, the degree-five precession polynomial, hour-to-radian
conversion, and adding pi. Orekit's GMST path counts 45: 16 for the two date
shifts, two `durationFrom` operations (two rounded operations each), eight for
ERA, three for the second century conversion, 12 for its degree-five
polynomial, and two final additions (GMST and pi). Its producer bound is
`gamma(31) + gamma(45)`. The GMST bound uses
`20 + 2 pi 1.003 |days|`, including unreduced Earth rotation, the precession
polynomial, and time conversion. The fixture sets TDB equal to TT, so that
comparison introduces no TT/TDB model offset.

The epoch test compares angles modulo `2 pi`, not as raw radians. Orekit's
fixture preserves the unwrapped Doodson arguments, while this crate's
fundamental-argument routine reduces each Delaunay angle modulo 1,296,000
arcseconds and GMST is returned in `[0, 2 pi)`. Each table multiplier is an
integer, so these representations give the same exact sine/cosine phase. The
test encloses each measured circular difference in an outward-rounded interval
that includes the comparison's own reduction error. Its lower endpoint must not
exceed the producer's operation-derived bound above. Requiring the upper endpoint
to meet that bound would incorrectly charge comparison roundoff to the argument
producer. The coefficient comparison then propagates the interval's upper
endpoint; the independent producer guard prevents an arbitrary phase discrepancy
from inflating that tolerance. The phase-formation
roundoff bound uses the larger of the Rust reduced-angle and Orekit unwrapped-angle
magnitudes. This is necessary because bounding only the reduced Rust phase
underestimates Orekit's six products and five additions on unwrapped arguments;
that omission caused the earlier order-zero coefficient bound to be too small.
The reduction check adds a four-operation roundoff allowance based on the two
supplied angles and number of turns; it cannot make a non-equivalent phase pass
by accepting arbitrary raw argument differences.

For Steps 1 through 3, the absolute coefficient magnitude is bounded from
the body masses and distances. Fully normalized degree-two and degree-three
Legendre functions are bounded by `sqrt(5)` and `sqrt(7)`. The sums of the
absolute real and imaginary Love numbers are below 0.31 and 0.094;
degree-four induced terms have smaller Love numbers than the degree-two
envelope. The permanent C20 contribution is included before applying a
128-operation budget per program. This envelope covers radius and latitude
evaluation, the degree-three harmonic recurrence, normalization, Love-number
products, both bodies, and permanent-tide subtraction. Step 2 contributes
its separately computed phase and summation bound. Absolute magnitudes
preserve a useful bound when individual coefficients cancel near zero.

## DEHANTTIDEINEL displacement comparison

The displacement fixture is the output of the independent IERS
`DEHANTTIDEINEL.F` program; the test does not use Rust output or observed
residuals to set its tolerance. Its maximum component error allowance is
computed from each case's input Sun and Moon vectors. The same absolute
allowance is applied to each ECEF component, including components whose true
displacement is close to zero.

For each body let `q2 = mass_ratio * RE * (RE/r)^3` and
`q3 = q2 * RE/r`, exactly the scale factors in Step 1. With normalized
direction cosines of magnitude at most one, the degree-two radial and
station-directed factors are bounded by `|x2| <= 3|l2| < 0.26` and
`|p2| <= 3|h2/2-l2| + |h2/2| < 1.4`. With `H3 = 0.292` and `L3 = 0.015`,
the degree-three factors satisfy `|x3| <= (3L3/2) * 4 < 0.1` and
`|p3| <= (5/2)|H3-3L3| + (3/2)|L3-H3| < 1.1`. The six diurnal,
semidiurnal, and latitude-dependent Step 1 corrections are bounded from their
explicit Love-number multipliers, normalized station/body coordinates, and
two-body sums. Together these contributions are enclosed by
`8(q2_sun + q2_moon + q3_sun + q3_moon)` metres per component. The factor 8 is
a deliberately outward-rounded envelope over those coefficient and
correction bounds, not a fitted multiplier.

For Step 2, the sum of absolute radial and transverse DATDI amplitudes in the
31 diurnal rows plus five long-period rows is below 0.03 m. The projection to
an ECEF component and summation operations are enclosed with 0.1 m, leaving
more than a factor-three allowance over that coefficient sum. Thus the operation-
magnitude envelope for a case is
`M = 8(q2_sun + q2_moon + q3_sun + q3_moon) + 0.1 m`.

The fixture dates are 1958–2040, so the routine's unreduced Step 2 angle
polynomials and integer multipliers keep every constituent phase below
2,000,000 degrees in absolute value. Each implementation's angle evaluation
is bounded by `gamma(64)` times this phase magnitude; the two-sided phase
allowance is `2 gamma(64) (2,000,000 degrees in radians)`. An additional
`8u` radians is conditional on the same `2u` absolute-error-per-result
assumption for the four sine/cosine evaluations (two functions in each
implementation). Since sine and cosine are 1-Lipschitz, multiplying this
phase allowance by the 0.1 m
Step 2 amplitude envelope bounds phase-induced displacement error. Steps 1
and 3 do not use these time-dependent phases. The remaining equation and summation
arithmetic is budgeted at 512 rounded operations per implementation and
bounded by `2 gamma(512) M`. The test therefore accepts a component only when
its absolute residual is at most
`2 gamma(512) M + 0.1 m (2 gamma(64) (2,000,000 degrees in radians) + 8u)`.

The diagnostic reports absolute residuals, the largest derived bound, and the
largest residual-to-bound ratio. It intentionally reports no “ULP” value:
dividing by `epsilon * |expected|` is not a representable-spacing metric, and
is ill-conditioned around zero. This comparison continues to use the
independent Fortran fixture as its sole expected-value source.

## Chapter 7 station-tide variants

PPP's default station tide uses `StationTideConstants::Conventions`, which
corrects three entries in the distributed Chapter 7 `DEHANTTIDEINEL.F`
diurnal table to match the Conventions text and its stated equations. Historical
Elixir and RTKLIB routine outputs use the uncorrected distributed `DATDI`
constants (`StationTideConstants::IersRoutine`). Their exact PPP fixture pins
are therefore checked with that explicit variant; the production default is
not changed and those expected values are not rebaselined. The documented
maximum displacement difference between variants is 0.18 mm.
