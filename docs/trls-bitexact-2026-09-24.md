# TRLS bit-exact validation — 2026-09-24

The unmodified `crates/trust-region-least-squares/scripts/bitexact_gate.sh`
completed with `bitexact_gate: PASS` and exit status 0 at 17:46:15 UTC.
All 17 tests in its seven integration suites passed; none were ignored or
filtered out.

## Revision and environment

- Commit: `08f0fea926e2467167c7474bfba774619eb0f154`.
- TRLS subtree: `8abbe7fd706e887931554e342b08951c0c03e5ff`.
- Includes the TRLS changes in `acf19d3e`, `98c0a8b9`, and `f4af8c79`.
- Temporary Vultr AMD EPYC-Milan VM, Linux x86_64; `avx512f` absent from
  every reported CPU flag set.
- Ubuntu 24.04.5 LTS, kernel `6.8.0-139-generic`, glibc 2.39.
- Rust `1.98.1 (48a229cea 2026-09-01)`, LLVM 22.1.8.
- Python 3.12.3, NumPy 2.5.0, SciPy 1.18.0.
- SciPy wheel's `libscipy_openblas-5f890258.so`,
  `OPENBLAS_CORETYPE=HASWELL`, `OPENBLAS_NUM_THREADS=1`.
- `SIDEREON_BITEXACT=1`, set by the gate itself.

The source came from the recorded commit, with all workspace crates and its
Cargo lockfile. Cargo added the workspace's `sidereon-core` self-dependency
edge to the lockfile; no package versions, package checksums, or TRLS
dependency entries changed. The gate regenerated its CPU-sensitive NumPy
power fixture as designed. No other reference fixtures were regenerated.

## Results

| Integration suite | Passed |
| --- | ---: |
| `data_problem_fixtures` | 1 |
| `general_fixtures` | 1 |
| `host_backend_power` | 8 |
| `hostlapack_fixtures` | 3 |
| `loss_fixtures` | 2 |
| `numdiff_fixtures` | 1 |
| `trf_fixtures` | 1 |

These results establish equality for the gate's pinned reference checks in
the recorded environment, not unrestricted equality across all machines,
compiler versions, or numerical backends.

## Evidence digests

SHA-256:

| Artifact | Digest |
| --- | --- |
| Gate script | `8d35bb59d9b5c2f3168a19a634a071a5cc56e6e2b8da9865978bde25306805e4` |
| Original Cargo.lock | `08a789aefef9bfb2075b066c2cb34791a3e6b81755161b97cd9386f7bf19efe9` |
| Cargo.lock after the run | `93e20cfadb8bd0e4cb9b471e46a5cd4edd68ddaa11dcc48f9c9d1f083588e10f` |
| SciPy OpenBLAS library | `2a5b4c1a6311873c59132b2ceaaf3c26258a17350b7cd27ea8782a47e3322981` |
| Regenerated NumPy power fixture | `acbf4e5c71236abb4aa943771858b75a6d2895fa5fe277f8bb4a9abbb1903731` |
| Complete run log | `081e5fe7e5689156f168cf60880b6fda363e77083447211db7464cf97de18b68` |

## Resource cleanup

A two-hour automatic poweroff was configured and verified before the run.
A separate two-hour deletion watchdog was also armed. After retrieving the
evidence, the VM was manually deleted; Vultr returned HTTP 204, followed by
HTTP 404 when queried. All earlier provisioning attempts were also deleted.
The watchdog was stopped after deletion was confirmed. No permanent runner
was created, and the GitHub workflow was not changed.
