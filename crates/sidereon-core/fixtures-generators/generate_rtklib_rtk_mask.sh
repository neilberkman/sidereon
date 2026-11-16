#!/usr/bin/env bash
# RTKLIB_SRC names the RTKLIB src directory; RTKLIB_LIBRARY may override its
# oracle_harness/librtklib.a. Run with Bash, a C compiler, Cargo, and Python.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${RTKLIB_SRC:?set RTKLIB_SRC to the RTKLIB source directory}
LIB=${RTKLIB_LIBRARY:-${SRC%/src}/oracle_harness/librtklib.a}
BUILD=$(mktemp -d)
trap 'rm -f "$BUILD/native" "$BUILD/portable" "$BUILD/common.o"; rmdir "$BUILD"' EXIT
CFLAGS="-std=gnu99 -O0 -include stdio.h"
cc $CFLAGS -I"$SRC" -DRTKPOS_SOURCE='"rtkpos.c"' "$HERE/rtklib_rtk_mask_oracle.c" "$LIB" -lm -o "$BUILD/native"
cargo build --release --manifest-path "$HERE/portable-math/Cargo.toml"
cc $CFLAGS -I"$SRC" -c "$HERE/portable-math/rtkcmn.c" -o "$BUILD/common.o"
cc $CFLAGS -I"$SRC" -DRTKPOS_SOURCE='"rtkpos.c"' "$HERE/rtklib_rtk_mask_oracle.c" "$BUILD/common.o" \
    "${CARGO_TARGET_DIR:-$HERE/portable-math/target}/release/libgeometry_oracle_math.a" "$LIB" -lm -o "$BUILD/portable"
"${PYTHON:-python3}" - "$BUILD/native" "$BUILD/portable" "$HERE" <<'PY'
import copy, hashlib, json, math, struct, subprocess, sys
from pathlib import Path
native, portable, here = sys.argv[1], sys.argv[2], Path(sys.argv[3])
doc = json.loads(subprocess.check_output([native], text=True))
positions = [row["satellite_ecef_m"] for row in doc["epochs"]]
def execute(binary, positions):
    values = " ".join(repr(value) for epoch in positions for satellite in epoch for value in satellite)
    return json.loads(subprocess.check_output([binary, "ecef"], input=values, text=True))
bits = lambda value: struct.unpack('Q', struct.pack('d', value))[0]
# Require an actual one-ulp gap in both arithmetic implementations. Adjacent
# ECEF operands can otherwise collapse to a tie in just one implementation.
found = False
for axis in range(3):
    if found: break
    for direction in (-math.inf, math.inf):
        if found: break
        candidate = copy.deepcopy(positions)
        candidate[7][2] = list(candidate[7][1])
        for step in range(64):
            candidate[7][2][axis] = math.nextafter(candidate[7][2][axis], direction)
            a, b = execute(native, candidate), execute(portable, candidate)
            if all(bits(row["epochs"][7]["elevation_rad"][1])-bits(row["epochs"][7]["elevation_rad"][2]) == 1
                   and row["epochs"][7]["reference"] == "G02" for row in (a,b)):
                doc, alternate, found = a, b, True
                break
assert found, "no common one-ulp input"
for row, other in zip(doc["epochs"], alternate["epochs"], strict=True):
    row["portable_elevation_rad"] = other["elevation_rad"]
    row["portable_reference"] = other["reference"]
doc["provenance"].update({"generator": "fixtures-generators/generate_rtklib_rtk_mask.sh",
    "generator_sha256": hashlib.sha256((here/"generate_rtklib_rtk_mask.sh").read_bytes()).hexdigest(),
    "harness_sha256": hashlib.sha256((here/"rtklib_rtk_mask_oracle.c").read_bytes()).hexdigest(),
    "build": "RTKLIB_SRC=RTKLIB/src RTKLIB_LIBRARY=RTKLIB/oracle_harness/librtklib.a bash fixtures-generators/generate_rtklib_rtk_mask.sh",
    "boundary_math": "zdres, selsat and ddres executed with platform libm and Rust libm 0.2.16 on identical ECEF operands; exact-tie and common one-ulp cases"})
(here.parent/"tests/fixtures/rtklib_rtk_mask_oracle.json").write_text(json.dumps(doc, separators=(',',':'))+'\n')
PY
