# SP3 accuracy rounding

SP3 stores record accuracy as signed exponent codes interpreted by the `%f`
position/velocity and clock/rate bases. `Sp3` retains each code with the bases
that originally defined it, so merging products with different bases can keep
the source standard deviations even when one shared output base cannot state
every code exactly.

`Sp3::to_sp3_string()` remains strict. It refuses when an accuracy code cannot
be stated under the output header bases without changing its decoded meaning.
`Sp3::to_sp3_string_with_rounding()` still rounds only numeric record values to
their SP3 columns and does not convert accuracy codes.

Callers who explicitly accept conservative accuracy conversion can use
`Sp3::to_sp3_string_with_accuracy_rounding(Sp3AccuracyRoundingPolicy::Conservative)`.
For each finite known standard deviation, the writer chooses the smallest legal
finite SP3 value at least as large as the source value. It reports the maximum
absolute change per field in SI units through
`Sp3DetailedRoundingReport::accuracy_sigmas`. This is a change to the stated
standard deviation, not an absolute error bound. A source value above the
finite legal range is refused instead of replaced by an SP3 sentinel or rounded
down.

Blank codes and the 99/999 too-large sentinels remain unchanged. Unknown,
invalid-base, and overflow states are not converted to a different meaning; if
the selected output bases would change such a state, writing is refused. The
source `Sp3` is never modified. Header values and epochs are not rounded by this
API, and all normal structure, field-width, finite-value, and record-sentinel
checks still apply.

The Rust facade exposes the policy and report at
`sidereon::ephemeris::{Sp3AccuracyRoundingPolicy, Sp3AccuracyRoundingReport,
Sp3DetailedRoundingReport}`.
