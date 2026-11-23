# ICGEM gravity field fixtures

Read by `tests/icgem.rs` and by the unit tests in
`src/astro/forces/icgem.rs`. The `.gfc` files are written by
`fixtures-generators/icgem_orekit_oracle/make_fields.py`; `icgem_orekit.json`
is printed by `fixtures-generators/icgem_orekit_oracle/generate.sh`.

| File | Content | Source and licence |
| --- | --- | --- |
| `egm96_d36.gfc` | EGM96 through degree and order 36, fully normalized, formal sigmas, `tide_free` | Lemoine et al. (1998), *The Development of the Joint NASA GSFC and NIMA Geopotential Model EGM96*, NASA/TP-1998-206861. A work of NASA and NIMA, in the public domain. Same values as the crate's embedded `egm96_to_36.txt`. |
| `unnormalized_d8.gfc` | EGM96 through degree 8 multiplied by `N_nm` and written unnormalized to 16 digits, no `tide_system` line | Derived from EGM96 as above. |
| `synthetic_icgem2.gfc` | Format `icgem2.0`, degree 6: EGM96 static values; synthetic `gfct`, `trnd`, `acos` and `asin` records over validity intervals, including degree 1 | EGM96 values public domain; the time-variable values are synthetic and describe no published model. |
| `synthetic_icgem1.gfc` | Format `icgem1.0`, degree 5: EGM96 static values; synthetic `gfct` (epoch `20050101`), `trnd`, `dot`, `acos` and `asin` records | As above. |
| `icgem_orekit.json` | Orekit 13.1.10 / Hipparchus 4.0.3 accelerations, potentials and time-variable coefficients for the files above and for a synthetic degree 2190 field | Output of the generator; the orekit-data commit is recorded in the file. |

The degree 2190 field is not stored: the generator and the test both write it
from the same integer formula (SplitMix64 mantissas scaled near `1e-5 / n^2`,
with EGM96's `C20`).
