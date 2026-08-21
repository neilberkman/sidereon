# Window-scoped SP3 continuity

`check_continuity` and the optional continuity post-condition on `merge` remain
product-wide reporters. They retain every defect and keep their existing
defaults and return values. A bounded evaluator can now ask those existing
reports which findings can enter an interpolation neighbourhood for a requested
inclusive epoch window.

## Decision

`EpochWindow` is expressed in seconds since J2000 on the SP3 product's own time
scale. `StencilExtent::for_sp3` derives five intervals of reach before and after
an evaluation epoch. Five is not a policy value supplied by the caller. It comes
from the same `NEVILLE_POINTS = 11` constant used by the degree-10 sliding-window
Lagrange position interpolator. The interval comes from the product header and
must be positive and finite. The product's first epoch anchors the nominal grid.
For non-node query times, filtering follows the interpolator by selecting the
last grid node at or before each window bound before applying the five-interval
reach.

A defect influences the window when its recorded epoch or epoch pair intersects
the evaluation window expanded by that derived reach. The bounds are inclusive,
so a finding exactly at the outermost stencil node influences the verdict. The
existing `SingleSampleSeries` finding has no epoch. It therefore influences every
window conservatively because a query over an existing report cannot assign it a
location.

`ContinuityReport::verdict_for_window` and
`MergeContinuityReport::verdict_for_window` return `Accept` only when no recorded
defect influences the expanded window. Otherwise they return `Refuse`. A merge
verdict also reports contributor-changing violations as splices. Both verdicts
retain the complete defect list, and merge verdicts retain the complete splice
list, so accepted findings outside the requested reach remain available for
logging and archive review. `MergeReport` preserves `None` when its continuity
post-condition was not requested.

## Contract boundary

The verdict answers whether a finding already recorded by the existing checker
can affect an epoch in the requested window through the nominal interpolation
stencil. It does not rerun continuity checks, prove that untested data is clean,
validate the caller's actual evaluation epochs, or guarantee interpolation is
possible across a coverage gap. It does not alter merge output, continuity
tolerances, interpolation arithmetic, or any default.

The CLI form is:

```text
sidereon inspect PRODUCT.SP3 --window FROM THROUGH
```

`FROM` and `THROUGH` are product-scale J2000 seconds. The command prints the
product-wide continuity summary and the window decision with the derived reach.

## Validation fixture

The integration test uses the committed public CODE MGEX final product
`COD0MGXFIN_20201770000_01D_05M_ORB.SP3`, SHA-256
`54b70fa009a840ecf8cec25fbd4d749c9aaef7c95bdf463484e115f74d802215`,
from the repository's established IGS fixture set and the canonical CDDIS MGEX
week 2111 archive. The digest was verified with `shasum -a 256` on 2026-08-21.

No consecutive daily pair is committed. The seam test constructs only its
second day by advancing the real product by 86,400 seconds, serializing with
Sidereon's SP3 writer, and reparsing. It injects a structured seam finding into
the merge report. This derived day is test input for window selection, not an
external continuity oracle. The untouched real product remains the primary
continuity fixture and attests under the existing checker.
