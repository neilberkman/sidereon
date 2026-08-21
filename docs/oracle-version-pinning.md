# Python oracle version pinning

Sidereon's numerical fixtures record the Python package versions and numerical
substrate that produced them. Package upgrades are evaluated as new oracle
environments. They do not silently refresh an existing fixture.

This note records two transition checks performed on 2026-08-21. The complete
artifact URLs, SHA-256 digests, host details, commands, and measured results are
in
`crates/trust-region-least-squares/tests/fixtures/oracle_version_pins.json`.
The committed
`crates/trust-region-least-squares/fixtures-generators/verify_oracle_version_pins.py`
script runs both checks.

## SciPy 1.17.1 to 1.18.0

SciPy 1.18.0 [ported FITPACK from Fortran to
C](https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html). A reported
consequence was deterministic zero filling of the unused coefficient tail where
older builds could expose uninitialized slots. That difference did not
reproduce with the pinned CPython 3.14 macOS arm64 wheels on this host.

For `splrep` on nine deterministic samples, both SciPy 1.17.1 and 1.18.0
returned four zero trailing coefficients. Poisoning and reusing the allocator
through 10,000 fits per version still found no nonzero tail. With interpolation
(`s=0`), the nine meaningful coefficients and 17 evaluations were bit-identical.
With smoothing (`s=0.01`), the seven meaningful coefficients moved by at most 21
ULP and the evaluations moved by at most 23 ULP. Therefore the broader claim
that evaluations never move across this version pair did not hold for the
smoothed probe.

The storage rule remains independent of that host result. FITPACK evaluation
uses only the first `n_knots - k - 1` coefficients, and SciPy documents the
additional `k + 1` entries as ignored. A coefficient fixture or digest hashes
only the meaningful prefix. An evaluation fixture also pins the SciPy version,
and changing that version creates a new fixture.

One-line reproduction from the repository root:

```sh
UV_CACHE_DIR=/tmp/sidereon-oracle-uv uv venv --python 3.14 /tmp/sidereon-oracle-old && UV_CACHE_DIR=/tmp/sidereon-oracle-uv uv pip install --python /tmp/sidereon-oracle-old/bin/python numpy==2.4.6 scipy==1.17.1 && UV_CACHE_DIR=/tmp/sidereon-oracle-uv uv venv --python 3.14 /tmp/sidereon-oracle-new && UV_CACHE_DIR=/tmp/sidereon-oracle-uv uv pip install --python /tmp/sidereon-oracle-new/bin/python numpy==2.5.0 scipy==1.18.0 && SIDEREON_ORACLE_OLD_PYTHON=/tmp/sidereon-oracle-old/bin/python SIDEREON_ORACLE_NEW_PYTHON=/tmp/sidereon-oracle-new/bin/python /tmp/sidereon-oracle-new/bin/python crates/trust-region-least-squares/fixtures-generators/verify_oracle_version_pins.py
```

## NumPy 2.4.6 to 2.5.0

A reported NumPy 2.5 pseudo-inverse change of up to a few hundred ULP also did
not reproduce with the pinned macOS arm64 wheels using Apple Accelerate. For
`default_rng(20260821).normal(size=(9, 6))`, NumPy 2.4.6 and 2.5.0 produced the
same pseudo-inverse bits and the same SHA-256 digest,
`742760f0f41566d7b1070be2b56f29f5520934a634d0283a5782ddf5f1698b09`.
A 285-case deterministic sweep across random, Hilbert, and nearly dependent
matrices of dimensions 2 through 20 also had zero differing cases. The measured
maximum was 0 ULP, not a few hundred ULP.

Pseudo-inverse fixtures still pin the NumPy version and BLAS/LAPACK substrate.
This measured equality is specific to the recorded environment and inputs. A
fixture generated under another NumPy version is a new fixture, not a refresh
of the old fixture.

The one-line reproduction is the same command above. Its `numpy_pinv` section
prints the fixed-matrix digests, maximum ULP difference, sweep case count, and
differing case count. NumPy's official [2.5.0 release
notes](https://numpy.org/doc/2.5/release/2.5.0-notes.html) identify the release
used by the check.
