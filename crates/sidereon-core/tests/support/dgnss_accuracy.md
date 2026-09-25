# DGNSS clean-fixture accuracy certificate

The common-mode regression uses a surveyed base, a specified rover position,
receiver clock offsets of +1 and -2 microseconds, and exact transmit queries.
Its reference receiver-clock difference in metres is the binary64 result of
`C_M_S * -2.0e-6 - C_M_S * 1.0e-6`, evaluated in that order. This is an explicitly
specified comparison point, not a claim that the decimal clock difference is
exactly representable. All residual bounds below use this binary64 comparison
point, not an enclosure of the exact decimal clock difference.
It does not use previously observed solver output as an accuracy reference.
The legacy scalar-correction tests retain their original scalar model.

## Independent residual enclosure

The test treats the satellite positions and corrected clocks returned for the
fixed raw-pseudorange transmit epochs as its ephemeris inputs. Ephemeris accuracy
and the exact-query contract have separate reference and callback tests. This
certificate checks the DGNSS correction and positioning arithmetic, not an
independent orbit solution.

Outward-rounded intervals independently evaluate Euclidean range, first-order
Sagnac range, base correction subtraction, corrected rover pseudorange and the
rover clock/range model. Every arithmetic node encloses both its real result and
its binary64 rounding. The interval primitive's documented IEEE 754 assumptions
apply. The returned clock is expanded to adjacent binary64 seconds before
multiplication by the speed of light, enclosing the internal clock that was
rounded by the public seconds conversion.

Let E_i bound the reference residual at the specified rover and clock difference.
These bounds come from the input observations and independent interval model.
They also include any remaining closure error after synthetic pseudorange
iteration; no assumed convergence rate or fixed-point truncation allowance is
needed. Let d_i be the width of the independently evaluated residual enclosure
at the returned solution. The reported residual must lie in that enclosure.

For GPS single-frequency code with ionosphere and troposphere corrections off,
the RTKLIB-default variance is

    v_i = ephemeris_i + 0.3^2 + 5^2 + 3^2
          + 300^2 * (0.003^2 + 0.003^2 / sin(elevation)).

The error model clamps elevation to at least five degrees. Using
1/16 < sin(5 degrees) <= sin(elevation) <= 1 gives conservative input-derived
global bounds v_min and v_max. Any weighted least-squares minimizer with those
weights has unweighted residual norm no larger than

    R = sqrt(v_max / v_min) * ||E||_2,

because the specified rover is an admissible comparison point. The test requires
the reported residual norm to be no worse than R + ||d||_2. This is a numerical
quality assertion, not a claim that a solver stopping-step threshold proves
optimality. It implies a real residual norm bound R + 2*||d||_2 after accounting
for the reporting/evaluation enclosure.

## Geometry-to-position bound

Before using a local certificate, the returned position must lie in an a priori
box extending one metre in each coordinate from the specified rover. Throughout
this convex box, interval derivatives enclose the full range Jacobian, including
the Sagnac derivative and a unit receiver-clock column. Transmit states do not
move with receiver position: they were placed from the raw observations.

The mean-value theorem gives F(x) - F(x_truth) = A*(x - x_truth), with each row
of A enclosed by its interval derivative. An independent interval Gauss-Jordan
inverse encloses (A^T A)^-1 and refuses any pivot interval containing zero.
Multiplication by A^T encloses the left inverse. Component j is therefore bounded
by

    sum_i sup(abs(((A^T A)^-1 A^T)[j,i])) * (E_i + R + 2*||d||_2).

The test checks each position coordinate and the receiver clock in metres against
this bound, and requires every bound to be smaller than the initial one-metre
box radius. No observed position discrepancy sets the box or a tolerance.

Position, clock, baseline vector, baseline length and residuals also retain
bit-for-bit repeated-run checks. Those checks establish determinism only; the
independent residual and geometry enclosures establish numerical accuracy.
Independent interval subtraction and Euclidean norm additionally check the
reported baseline vector and length against the certified receiver position.
