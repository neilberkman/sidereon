# RTKLIB SPP selection and RAIM FDE oracles

`generate.sh` writes `crates/sidereon-core/tests/fixtures/rtk/rtklib_spp_selection_oracle.json` by running the pinned RTKLIB `pntpos` implementation over the checked-in ESBC and WTZR observation fixtures. Each run records four initial receiver positions for each observation epoch, including RTKLIB's solution, covariance, used satellites, and satellite-state inputs. At each returned solution, the helper reevaluates unmodified RTKLIB `rescode` to record aligned residuals, design rows, and effective variances at that exact receiver state; it does not reuse the pre-update `ssat.resp` values.

Each used GPS satellite state retains the human-readable `transmit_epoch` produced by `time2epoch` and `%.17g`. It also records `transmit_j2000_whole_s`, the exact integer part of the post-`timeadd` RTKLIB `gtime_t` relative to J2000, and `transmit_fraction_bits`, the exact binary64 bits of `gtime_t.sec` as 16 hexadecimal digits. The latter pair reconstructs the C query instant without combining the whole second and fraction into `ep[5]` or converting that rounded decimal label back into an epoch. The C helper requires binary64 `double` for this encoding.

Each successful positioning result also records `lsq_weighted_design_columns` and `lsq_covariance`. These are copied around the unchanged RTKLIB `lsq` call used by `estpos`: columns are the weighted design matrix passed to `lsq`, and the covariance is its returned binary64 matrix, both at the final accepted position iteration. The wrapper delegates once to RTKLIB and only copies values; it verifies that the six position covariance entries round to the returned `sol.qr` floats. `lsq_receiver_state` and `lsq_step` record the state immediately before and the increment returned by that same call. `lsq_geodetic_rad_m` is RTKLIB `ecef2pos` applied to that pre-update ECEF state. The wrapper passively initializes tracked state from the starting position and zero clocks, then mirrors each successful `estpos` update in parameter order. Runtime bitwise checks require the replayed final position and receiver clock to match `sol.rr` and `sol.dtr[0]`, and the exported final state-plus-step to match the replay. This uses the unmodified included `pntpos.c`; it does not directly capture its local state. `estvel` uses four parameters and does not replace the captured positioning matrix (`estpos` uses `NX=9` in this build).

Successful cases also include `geodetic_rad_m`, the latitude, longitude, and height returned by RTKLIB `ecef2pos` for the actual `sol.rr` ECEF result. This is a C reference candidate for the center-evaluator checks; it is not a position oracle generated from the Rust result.

## Consumer checks

The Rust consumer reconstructs the exact C transmit instants and checks C and native satellite states against independent GPS LNAV interval evaluations. It independently encloses model rows at both endpoints and the final C least-squares prestate. Both binary64 covariance matrices must lie inside their respective independent inverse enclosures; the six recorded binary32 entries must be the exact rounding of the captured C covariance entries. This does not assert that every value in an interval enclosure rounds to the same binary32 value.

Endpoint comparison uses a fixed one-metre receiver/clock ball and a bound on the complete state-dependent weighted iteration, including changing design rows and weights. Input-derived model and arithmetic enclosures determine the comparison bound; the observed endpoint separation is checked against that bound but does not determine it. The consumer refuses cases outside the supported GPS, atmosphere and fixed-selection certificate domains.

## Regenerate

Set `RTKLIB_SRC` to the `src` directory in a clean checkout of RTKLIB demo5 commit `75a2e56275485b21a67bd35bc94bbeb8936e1a74`, then run:

```sh
RTKLIB_SRC=/path/to/RTKLIB/src ./generate.sh
```

The script verifies the checkout revision and refuses a dirty RTKLIB `src` tree. It hashes the NAV fixture and both OBS fixtures into the JSON provenance block. Compilation and all five RTKLIB runs write to temporary paths; only a successful complete run atomically replaces the checked-in JSON file.

## Inputs and coverage

- NAV: `tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx`
- OBS: `tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx`
- OBS: `tests/fixtures/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx`
- Runs: ESBC with ionosphere/troposphere, troposphere only, and ionosphere only; WTZR with ionosphere/troposphere and ionosphere only.
- Initial states: geocentre, the RINEX approximate position, and that position rotated 12 degrees east or west.

The selection runs use GPS L1 C/A, broadcast ephemerides, a 10-degree elevation mask, and no RAIM exclusion. The helper compiles the pinned, unmodified `pntpos.c` into its own translation unit to call its static `rescode` at each returned solution; `pntpos.c` is therefore omitted from the separate object list. JSON records its commit identifier and hashes the NAV/OBS inputs; the generator source and compiler are not included in those input hashes.

## RAIM FDE oracle

`generate.sh` also writes `crates/sidereon-core/tests/fixtures/rtk/rtklib_spp_fde_oracle.json` from `rtklib_spp_oracle fde`. It takes every twelfth epoch of the ESBC fixture with the ionosphere and troposphere corrected, solves it with the unmodified `pntpos` from the approximate position, and then, for each satellite that solve used, adds +5000 m and then +300 m to that satellite's L1 pseudorange and hands the faulted epoch to RTKLIB's own `raim_fde` with the `satposs` states `pntpos` would compute.

demo5's `pntpos` reaches `raim_fde` only when `estpos` fails, because `valsol`'s chi-square rejection is commented out, so the oracle calls `raim_fde` directly. Its exclusion is the reference, not whether demo5 would have run it.

`raim_fde` does not report which satellite it removed or its final least-squares state. The helper replays its loop, calling the unmodified `estpos` in the same order with the same shared solution state, and requires the replay to equal `raim_fde`'s result bit for bit (position, clock and used satellites) before it records the excluded satellite, every candidate's status and residual RMS, and the chosen solution in the same form as a selection case, including the captured least-squares state and the satellite rows at the solution. The Rust test `fde_spp_matches_rtklib_raim_fde_on_faulted_epochs` requires `fde_spp` to exclude the same satellite in every case and certifies its solution against RTKLIB's with the selection oracle's endpoint certificate.
