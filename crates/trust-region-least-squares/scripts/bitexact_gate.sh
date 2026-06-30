#!/usr/bin/env bash
# Reproducible bit-exact gate for trust-region-least-squares.
#
# The committed parity fixtures reproduce scipy's trust-region-reflective
# least_squares bit-for-bit ONLY in a specific environment. The binding numbers
# come from the LAPACK/BLAS scipy itself loads, and numpy's array `pow` matches
# glibc `pow` (which Rust's powf calls) ONLY on a CPU without AVX-512: numpy
# 2.5.0 dispatches a wider SVML `pow` on AVX-512 that differs by 1 ULP, and that
# path cannot be disabled at runtime (NPY_DISABLE_CPU_FEATURES does not affect
# it). So this gate must run on a NON-AVX-512 x86_64 Linux host.
#
# Canonical env: Linux x86_64 (no AVX-512), glibc 2.39, Python 3.12,
# scipy 1.18.0 / numpy 2.5.0 (wheel-bundled OpenBLAS), OPENBLAS_NUM_THREADS=1,
# OPENBLAS_CORETYPE=HASWELL.
set -euo pipefail

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$crate_dir"

# 1. Refuse to run where the result would be a false negative.
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "bitexact_gate: requires Linux x86_64 (got $(uname -s)/$(uname -m)); the parity replays are platform-pinned." >&2
  exit 2
fi
if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
  echo "bitexact_gate: this CPU reports AVX-512; numpy 2.5.0's SVML pow diverges 1 ULP and cannot be disabled. Use a non-AVX-512 host." >&2
  exit 2
fi

# 2. Pinned Python toolchain in a throwaway venv.
venv="$(mktemp -d)/venv"
python3.12 -m venv "$venv"
# shellcheck disable=SC1091
. "$venv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet "numpy==2.5.0" "scipy==1.18.0"

# 3. Point the host-LAPACK backend at the wheel's bundled OpenBLAS.
lapack_so="$(python - <<'PY'
import glob, os, numpy
roots = [os.path.join(os.path.dirname(os.path.dirname(numpy.__file__)), d)
         for d in ("scipy.libs", "numpy.libs")]
for root in roots:
    hits = sorted(glob.glob(os.path.join(root, "libopenblas*.so")) +
                  glob.glob(os.path.join(root, "libscipy_openblas*.so")))
    if hits:
        print(hits[0]); break
PY
)"
if [ -z "${lapack_so:-}" ]; then
  echo "bitexact_gate: could not locate the wheel-bundled OpenBLAS .so" >&2
  exit 1
fi

export TRUST_REGION_LEAST_SQUARES_LAPACK_PATH="$lapack_so"
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=HASWELL

# Opt in to the platform-pinned hex-bit replays. A default `cargo test` skips
# them (green on every platform); this gate runs them on the canonical host.
export SIDEREON_BITEXACT=1

echo "bitexact_gate: scipy $(python -c 'import scipy;print(scipy.__version__)') / numpy $(python -c 'import numpy;print(numpy.__version__)')"
echo "bitexact_gate: LAPACK -> $lapack_so"

# 4. Run the bit-exact replays. These assert hex-bit equality vs scipy 1.18.0.
cargo test -p trust-region-least-squares \
  --test loss_fixtures \
  --test general_fixtures \
  --test hostlapack_fixtures \
  --test data_problem_fixtures \
  --test numdiff_fixtures \
  --test trf_fixtures

echo "bitexact_gate: PASS"
