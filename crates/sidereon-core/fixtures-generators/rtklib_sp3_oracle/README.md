# RTKLIB SP3 downstream oracle

This generator records an independent RTKLIB demo5 result for the SP3-derived
visibility and DOP pins, plus a `pntpos` result using the static-positioning
fixture's pseudoranges and broadcast navigation. It calls RTKLIB
`peph2pos`/`pephpos`, `satazel`, `dops`, `readrnx`, and `pntpos` directly; it
does not invoke Sidereon or read Sidereon's numerical
outputs. The source revision is pinned to
`rtklibexplorer/RTKLIB` demo5 commit
`75a2e56275485b21a67bd35bc94bbeb8936e1a74`. The generated JSON records that
revision, the upstream repository and commit URL, the generator path, the
navigation source URL, and SHA-256 digests of the SP3 and navigation inputs.

The geometry run uses the source fixture and application receiver/epoch in
`sidereon_gnss_application_golden.json`, plus the DOP receiver from
`spp_trace_L2_tropo.json`; it emits RTKLIB satellite positions, both receivers'
azimuth/elevation, 10-degree visibility, and five RTKLIB DOP values at a
5-degree mask. The static run uses the 8 pseudoranges and initial receiver
position in `go_fixture_parity.rs`, with GPS L1C, precise SP3 ephemerides,
explicit ionosphere/troposphere modes passed to `pntpos`, and an elevation
mask of -90 degrees so RTKLIB applies no elevation cutoff while iterating from
the Go fixture's initial receiver position. A zero-degree mask would reject
below-horizon satellites relative to that initial guess before the position
has converged. Its position is a reference result for review; it does not claim bit
equality with the two-epoch Sidereon static solve, which uses a different solve
formulation.

From the Sidereon repository root on the validation host:

```sh
RTKLIB_SRC=/path/to/rtklib/demo5/src \
RTKLIB_IONO_MODE=off \
RTKLIB_TROPO_MODE=off \
  bash crates/sidereon-core/fixtures-generators/rtklib_sp3_oracle/generate.sh
```

The generator uses the committed, uncompressed day-176 RINEX broadcast
navigation fixture. Its source URL and hash are recorded in the generated JSON.
The script requires explicit modes (`off` or `brdc` for ionosphere;
`off` or `saas` for troposphere), refuses a checkout at any revision other than the
pinned commit or any modified, staged, or untracked files under RTKLIB `src`, and writes
`crates/sidereon-core/tests/fixtures/sp3/rtklib_sp3_node_oracle.json`. Review
that generated oracle against the changed pins before updating any golden; the
generator never reads or copies their values.

RTKLIB `satposs` requires a broadcast ephemeris clock to estimate transmit time
before it can use precise SP3 states; SP3 alone cannot support `pntpos`. The
generator loads the RINEX navigation with `readrnx` and records its name, source
URL, and SHA-256 in the JSON. The `go_fixture_static` object is the RTKLIB `pntpos` result recorded as an
independent baseline. It is not compared with the current Sidereon static solve,
whose model and solve formulation differ. The integration test checks the
oracle's input names and SHA-256 digests against the fixture bytes, verifies the
explicit options and inputs, and requires RTKLIB to report every input satellite
as used with an empty message. It checks that the recorded result is finite but
does not apply a fitted tolerance to cross-model output.

The separate `sp3_tx_states` rows close the SP3 state-reference comparison for
the actual static inputs. For each of the eight GPS pseudoranges, RTKLIB reads
the raw SP3 product clock at `t_rx - P/c`, forms `t_tx` by subtracting that
clock, then emits the `peph2pos` ECEF state at `t_tx`. Since `peph2pos` applies
the relativistic correction to the product clock, the generator removes that
term when it reconstructs the raw SP3 clock for transmit-time placement. The Rust
test runs `solve_static` on the same eight observations duplicated at the same
receive epoch, with `StaticSolveOptions::default()`, and records unique
satellite/query-epoch states rather than callback counts. It compares those
positions without using the solved receiver position or residuals. Its tolerance
is an operation-derived bound for the two interpolation implementations, not an
SP3 coordinate-resolution allowance. The separate `pntpos` result remains a
baseline from a different model and is not treated as the `solve_static`
expectation.

For each comparison, the consumer reads the satellite's recorded SP3 nodes via
the public product API and applies the same contiguous-run and 11-node window
selection as `pephpos`/Sidereon's Neville interpolator. The roundoff bound uses
the actual node-minus-query offsets and coordinate magnitudes. With `n=11`, each
axis evaluates `n(n-1)/2 = 55` Neville cells per implementation; each cell has
two products, a numerator subtraction, a denominator subtraction, and a
division. The recurrence propagates an absolute magnitude and error enclosure:
for offsets `a,b`, `A=|a|/|a-b|`, `B=|b|/|a-b|`, and magnitude `M' = A M₁ + B
M₂`. Its local error uses `γ(4)` for numerator and quotient arithmetic plus
`d/(1-d)` for denominator rounding, where `d = ε(|a|+|b|)/|a-b|`; both
independent implementations' local budgets are added. SP3 km-to-m conversion
and the public API's m-to-km round trip are included in the node input budget;
the final km-to-m conversion is included for both outputs.

The node input budget also includes the per-node Earth-rotation correction. It
uses the observed `|x|+|y|` magnitudes, `γ(3)` for the rotation products and
sum/difference, and `4 × 2ε` for the two implementations' sine/cosine errors.
The angle-product rounding is bounded by `γ(1)|θ|` times the node norm. These
bounds assume binary64 round-to-nearest arithmetic without overflow/underflow,
correctly rounded decimal-to-binary input conversion, and absolute sine/cosine
errors no greater than `2ε` in the C and Rust math libraries for these angles.
Here `ε` is `f64::EPSILON` and `γ(k)=kε/(1-kε)`, deliberately using twice the
usual unit roundoff for conservative arithmetic budgets. The test does not fit
any budget to observed residuals. A final `γ(32768)` factor encloses rounding
while evaluating the positive recurrence and derivative bounds: the bound
calculation is capped by 10 operations per term in its 11³ node-offset
derivative loops, 6 per term in the 11³ query-derivative loops, 24 per Neville
cell across three axes, and the surrounding reductions. The selected angles are
asserted to stay within `|θ| ≤ 0.5` rad for the stated math-library assumption.

The two-ULP absolute-J2000 epoch allowance is propagated using an analytic
Lagrange time-derivative bound over that interval. For each basis polynomial,
the consumer bounds its magnitude and derivative from the selected offsets,
using `|offset| + time_radius` in each numerator factor and the actual node
separations in the denominator; it adds the Earth-rotation derivative
`OMEGA_E_DOT` times the node norm. Independent node-offset rounding is bounded
through the derivatives of the Lagrange basis with respect to each node offset;
its denominator factors use `|offsetᵢ-offsetⱼ| - 2r - ε(|offsetᵢ|+|offsetⱼ|)`,
where `r=γ(2) max(|offset|)`; the last term bounds subtraction rounding.
The resulting 3D sensitivity bounds times their
respective radii are added to the interpolation roundoff bound for static and
geometry state comparisons. The common epoch radius is two ULPs of the absolute
J2000 epoch. Node norms use the coordinate L1 norm as an upper bound.

Azimuth/elevation comparison propagates the resulting position bound through
the line-of-sight angle, including azimuth conditioning at the oracle elevation.
Its `64ε × 360°` additive term covers the degree conversion and wrapped-angle
arithmetic; threshold classification separately asserts a margin exceeding the
position-induced direction bound.
DOP comparison records the 4×4 `HᵀH` and inverse covariance from a duplicate
of RTKLIB `dops`'s construction, alongside the five DOP outputs. The generator
capture repeats the same fixed-order design-row construction, `matmul`, and
`matinv` calls as `dops` on the same azimuth/elevation rows. The pinned source
has no randomized or stateful arithmetic in those paths. The Rust consumer
forms its normal matrix from `TransmitTimeSatelliteState::los_unit` with light
time and Sagnac disabled, matching the state and LOS route used by
`dop_series`, and calls the same public cofactor kernel to obtain its inverse
candidate. RTKLIB's local rows are
`[cos(el) sin(az), cos(el) cos(az), sin(el), 1]`; Sidereon's public design
rows use the opposite spatial sign, `[-LOS, 1]`. The comparison explicitly
puts Sidereon's local rows and covariance into RTKLIB's coordinates with
`D = diag(-1, -1, -1, 1)`: rows use `D h`, and covariance uses `D Q D`.
Position variances are unchanged and position-clock cross terms change sign.
Thus the enclosed row difference is only from input geometry and arithmetic,
not a basis change. The row differences are enclosed with forward-error bounds
for the three-term ECEF-to-ENU products, RTKLIB row trigonometry and products,
and the Euclidean row-difference reduction.

For each candidate inverse `X` and recorded/computed matrix `A`, the consumer
encloses `R = I - A X` entrywise. Each four-term dot product and identity
subtraction has a `γ(8)` absolute error envelope based on an upward-bounded sum
of absolute products; the four residual magnitudes per row use a separate
positive reduction bound. The measured residual must be below both 1 and a
path-specific ceiling, itself required below 1.

The RTKLIB ceiling follows the pinned default `ludcmp`/`lubksb` path (`LAPACK`
is not enabled). For its original row scales `sᵢ=maxⱼ|Aᵢⱼ|`, set
`Rₛ=max(sᵢ)/min(sᵢ)`. The scaled pivot scores use rounded reciprocals and
products, so the multiplier bound is `μ=Rₛ(1+γ(1))⁵`. Each Schur update is
bounded by `(1+Rₛ)(1+γ(2))` with `Rₛ` replaced by `μ`; over three pivots this
gives `G=((1+μ)(1+γ(2)))³`. With `M=maxᵢⱼ|Aᵢⱼ|`, the factors obey
`||L||∞≤1+3μ` and `||U||∞≤4GM`; the ceiling propagates the solve roundoff as
`γ(163)||L||∞||U||∞||X||∞`, then adds the separately enclosed residual
evaluation error and converts to Frobenius norm. The operation count 163 is
51 operations in `ludcmp` (28 multiply/subtract updates, 10 pivot-score
multiplications, 4 row-scale reciprocals, 3 pivot reciprocals and 6 lower-factor
multiplications) plus four `lubksb` solves at at most 28 operations each (12
forward substitutions, 12 backward substitutions and 4 divisions).

For Sidereon's cofactor inverse, the consumer bounds the actual intermediates
instead of applying the RTKLIB bound. `P₄` is the sum of the 24 absolute
four-factor permutation products of the exact ECEF normal matrix. Enumerating
these products takes 72 rounded multiplications after excluding each exact
initial multiplication by one, and summing them takes 23 rounded additions,
for 95 operations. Including the final inflation multiplication gives the
envelope `γ(96)P₄`.
The determinant kernel error is `γ(45)P₄`. For each inverse numerator, `P₃`
is the sum of the six absolute three-factor products in that minor. The
products take 12 rounded multiplications after excluding each exact initial
multiplication by one; the sum takes five rounded additions, for 17 operations.
Including the final inflation multiplication gives the envelope `γ(18)P₃`.
The numerator kernel error is `γ(14)P₃`. It subtracts the determinant error from the
computed determinant to obtain a positive lower bound `dₗ`. Each inverse-entry
error is bounded by the division-rounding term plus
`e_num/dₗ + (|num|+e_num)e_det/dₗ²`. Those bounds are reduced to a Frobenius
error before propagation through the ENU conversion. The local residual
ceiling additionally encloses the measured difference between the local
normal matrix and `T Aₑ Tᵀ`, the rotation defect `||RᵀR-I||F`, covariance
rotation arithmetic and residual evaluation; `T=diag(-R,1)` and `Aₑ` is the
same ECEF normal matrix supplied to Sidereon's cofactor inverse. This keeps
both inverse ceilings tied to their actual algorithms and input magnitudes,
not a covariance or DOP output residual.

Since `||R||₂≤||R||F`, an accepted bound `η<1` certifies
`||A⁻¹-X||F≤||X||Fη/(1-η)`. The consumer forms an input-level normal matrix
perturbation bound from the enclosed row differences. For rows `h` and `g`, it uses
`||hhᵀ - ggᵀ||F ≤ ||h-g||₂ (||h||₂ + ||g||₂)`, then adds the
`γ(2N+4)` matrix-product/summation envelope for `N` rows and the dimension-4
entry-to-Frobenius reduction factor 4. The outer positive reductions use
`γ(2N+24)`: at most `2N` additions for the two per-row bound sums and 24
operations for the row-norm, multiplication and final-bound reductions. No
covariance or DOP difference enters this perturbation bound.

With candidate inverse norm bounds `U_a`, `U_g` and normal-matrix perturbation
bound `Δ_A`, the covariance difference is bounded by
`U_a U_g Δ_A` plus both inverse-residual certificates. The variance projection
rank factors are 4, 3, 2, 1, and 1 for GDOP, PDOP, HDOP, VDOP, and TDOP.
Sidereon's ECEF-to-ENU covariance rotation receives a separate
`γ(12) ||R||F² ||X||F` arithmetic allowance. The computed `RᵀR-I` residual,
enclosed entrywise by its three-term dot-product error, bounds the trace change
from finite-precision loss of orthogonality, including GDOP's ECEF-to-ENU
comparison. Diagonal reductions and square roots receive their own
operation-derived bounds. At most 32 arithmetic operations enclose each final
variance-to-DOP bound calculation. A positive lower variance converts the
independently propagated variance bound to a DOP bound through the square-root
difference identity.
