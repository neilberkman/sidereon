# RTK numerical evidence bounds

This note records the analytic error budgets used by the ZWD PROJ comparison and
the phase-wind-up transmit-time replay in the core tests. They bound numerical
sensitivity; they are not tolerances selected from observed residuals.

## ZWD elevation and delay tolerance

`fixtures-generators/generate_tropo_zwd.py` constructs the requested direction
using geodetic north/east but normalized receiver ECEF (geocentric up). PROJ
and the production API use the geodetic ellipsoid normal (geodetic up), so the
requested generator elevation is not treated as the actual elevation. The test
recomputes actual elevation from the fixture's ECEF difference using the
geodetic ENU basis and `atan2(up·d, hypot(east·d, north·d))`. It also checks
that the geocentric-versus-geodetic elevation discrepancy does not exceed the
angular separation of the two up vectors, apart from a `1e-12 rad` elementary-
function allowance. Every actual case has range at least 20 Mm and elevation
strictly between 4.8° and 84.2°; these checks establish the bound's domain.

The test's `atan2(U, hypot(E,N))` and production's `asin(unit(d)·up)` express
the same signed latitude of the line-of-sight unit vector in the local frame.
Elevation is 1-Lipschitz with respect to angular displacement of that unit
vector; it does not acquire a `1/cos(E)` factor when computed from all three
coordinates. For a perturbation of a nonzero vector by norm at most `delta`,
the direction angle is at most `2*asin(min(1, delta/r))`, where `r` is the
smaller endpoint norm. This avoids applying the ill-conditioned scalar `asin`
derivative to only the vertical component.

For the Niell mapping fraction, write `s = sin(E)`,
`q = s + b/(s+c)`, `D = s + a/q`, and `F = 1 + a/(1+b/(1+c))`, so `M = F/D`.
For the positive coefficients used here, `D >= s` and
`|dD/ds| <= 1 + a/s² + a/b`. Over all supported coefficient values,
`a_h <= 0.00315`, `a_h/b_h < 24.4`, `F_h < 1.004`,
`a_w <= 0.020`, `b_w >= 0.0052`, `a_w/b_w < 3.9`, and `F_w < 1.020`.
At `E >= 4.8°`, differentiation of `M` gives
`|dM_h/dE| < 3,800 rad⁻¹` and `|dM_w/dE| < 1,200 rad⁻¹`.
The fixtures' profiles and altitude interval give `ZHD < 3 m` and `ZWD < 0.3 m`,
so the slant-delay sensitivity is below
`3*3,800 + 0.3*1,200 = 11,760 m/rad`, rounded upward to `12,000 m/rad`.

The geometric computation uses round-to-nearest binary64, with unit roundoff
`u = EPSILON/2`; for `n` elementary rounded operations,
`gamma_n = n*u/(1-n*u)`. The fixtures have at most 60 Mm as the sum of
absolute ECEF input coordinates. `gamma_16 * 60 Mm < 1.1e-7 m` is the
coordinate-operation allowance for one ENU evaluation. Applying it to both
paths, their componentwise difference is at most `2*1.1e-7 m`, and the ENU
vector difference is below `2*sqrt(3)*1.1e-7 m`. With checked range at least
20 Mm, the two unit directions differ by at most
`2*(2*sqrt(3)*1.1e-7)/(20e6)` in chord length. The angular coordinate-roundoff
bound is `2*asin(2*sqrt(3)*1.1e-7/20e6)`, approximately `3.81e-14 rad`. The
generator up-vector mismatch is not included in this roundoff term: the test
evaluates the actual geodetic elevation of each generated ECEF vector and
checks the up-vector separation independently.

The `1e-12 rad` elevation allowance assumes the PROJ and Rust binary64
elementary functions used here (`sin`, `cos`, `hypot`, `atan2`, `asin`) are
within four ULP for these finite arguments. Their accumulated contribution,
plus normalization arithmetic is covered by the explicit `1e-12 rad`
allowance. The four-ULP premise is not guaranteed by IEEE-754 or for arbitrary
math libraries. For at most 100 rounded atmospheric operations on
intermediates below 100 m, the deliberately conservative accumulated budget
`100*gamma_100*100 m` is below `1.12e-10 m`. The `1.2e-10 m` allowance covers
this arithmetic budget and elementary-function roundoff under the stated
four-ULP assumption. Combining it with the derived `12,000 m/rad` delay
sensitivity and full `1e-12 rad` elevation allowance gives less than
`1.212e-8 m`; the checked tolerance `1.5e-8 m` rounds upward.
These are explicit numerical assumptions and operation-count bounds, not
bounds guaranteed for every math library.

## Wind-up transmit-time replay

The test compares the exact and microsecond-rounded transmit predictions at their
respective epochs. For each, it constructs the line-of-sight normal and the
satellite and receiver dipoles projected into the wind-up plane. No velocity
finite difference or fitted residual enters this bound.

For any nonzero vectors `x` and `y`, let `c = min(||x||, ||y||)`. Normalization
satisfies
`||x/||x|| - y/||y|||| <= 2||x-y||/c`; converting the unit-vector chord to an
angle gives the conservative bound `2*asin(min(1, ||x-y||/c))`. The minimum
endpoint dipole norm in `c` explicitly accounts for projection conditioning. The
phase-change bound adds the satellite-dipole and receiver-dipole angular bounds,
and twice the LOS-normal angular bound to account for the moving tangent-plane
frame. The test reproduces the production normalized-dot, `acos`, and signed
cross-product calculation for endpoint phases, then checks both branch margins.

The phase calculation's `acos` input is a normalized dot product formed from
three multiplications and two additions in `dot3`; each norm uses the same five
operations followed by `sqrt`, and production divides the dot by each norm in
sequence. The test asserts finite endpoint vectors, nonzero component products
and squared components are normal, and the resulting norms are normal. Dot
reductions may cancel to zero or a subnormal value; their absolute error bound
uses the sum of absolute products, not relative error in the reduced result.
For these inputs, Cauchy-Schwarz gives
`sum(abs(x_i*y_i))/(||x||||y||) <= 1`. The dot's absolute error normalized by
the exact norm product is therefore at most `gamma_5`. Each computed norm has
relative error at most the larger of
`sqrt(1+gamma_5)*(1+u)-1` and `1-sqrt(1-gamma_5)*(1-u)`, assuming IEEE-754
correctly-rounded square root. Including both sequential division roundings,
let `q=(1-e_norm)^2*(1-u)^2`; a conservative absolute cosine error is
`gamma_5/q + 1/q - 1`. Clamping to `[-1,1]` cannot increase this error.

For exact `a,b` in `[-1,1]`, the global inequality
`|acos(a)-acos(b)| <= 2*asin(sqrt(|a-b|/2))` follows from
`|cos(x)-cos(y)| >= 1-cos(|x-y|)`. It remains valid at both singular
derivative endpoints and requires no fitted phase margin. The test adds this
Hölder bound for each dipole endpoint. The shared angle-upper-bound helper
rounds each `asin` argument one representable value toward `+∞`, assumes
`libm::asin` is within four ULP on `[0, π/2]`, adds `4*EPSILON` radians, then
rounds both the adjusted result and its doubled angle toward `+∞` with
`nextafter`. This covers the `asin` inside the Hölder bound separately from
`4*EPSILON*PI` for each `libm::acos` evaluation. The `acos` term assumes the
pinned binary64 `libm` implementation is within four ULP on these fixture
inputs. Neither math-function accuracy assumption is guaranteed by IEEE-754
or Rust's API. The two endpoint budgets are included both in the wind-up
movement allowance and in the principal-branch guard. The signed phase uses
`dot(los,cross(satellite,receiver))` to select its sign. Its rounding error is
bounded from the three two-product cross reductions (`gamma_3`) and the final
three-product, two-addition dot reduction (`gamma_5`). When the computed
orientation is no larger than this absolute error bound, either sign may be
possible; the test then adds twice that endpoint's computed absolute phase
plus its `acos` error bound. This handles sign ambiguity near zero without
assuming a minimum phase or fitting a tolerance. The orientation bound assumes
finite inputs with no overflow. Each cross reduction includes three least-
subnormal absolute-error allowances, and the final dot includes five, so
cancellation to subnormal or zero orientation is covered as well. Conversion
to metres adds `gamma_3` times the sum of endpoint phase magnitudes times the
absolute metres-per-cycle coefficient divided by `2π`; this bounds the two
endpoint phase-to-cycle divisions, their multiplications by the common
coefficient, and the measured-difference subtraction. The coefficient is
computed once and held common to both endpoints. The
geodetic-versus-geocentric receiver-frame comparison uses the same endpoint
and conversion budgets and separately guards both frame phases against ±π.
These operation-derived terms replace arbitrary epsilon padding.
