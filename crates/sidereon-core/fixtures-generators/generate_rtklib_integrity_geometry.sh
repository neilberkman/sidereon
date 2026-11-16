#!/usr/bin/env bash
# Regenerate RTKLIB integrity-geometry and antimeridian DOP fixtures.
# usage: RTKLIB_SRC=<rtklib-source-directory> ./generate_rtklib_integrity_geometry.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../tests/fixtures"
SRC="${RTKLIB_SRC:?set RTKLIB_SRC to RTKLIB demo5 src at commit 75a2e562}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
OPTS="-DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE"
CFLAGS="-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -w -I$SRC $OPTS"
for unit in rtkcmn trace rinex rtkpos postpos solution lambda geoid sbas preceph \
    ephemeris options ppp ppp_ar rtcm rtcm2 rtcm3 rtcm3e ionex tides sofa; do
    cc $CFLAGS -c "$SRC/$unit.c" -o "$BUILD/$unit.o"
done
cc $CFLAGS -o "$BUILD/oracle" "$HERE/rtklib_integrity_geometry_oracle.c" \
    "$BUILD"/*.o -lm
"$BUILD/oracle" > "$FIX/rtklib_integrity_geometry.json"
"$BUILD/oracle" antimeridian > "$FIX/rtklib_antimeridian_dops.json"
cargo build --release --manifest-path "$HERE/portable-math/Cargo.toml" --target-dir "$BUILD/math"
cc $CFLAGS -c "$HERE/portable-math/rtkcmn.c" -o "$BUILD/rtkcmn.o"
cc $CFLAGS -o "$BUILD/portable-oracle" "$HERE/rtklib_integrity_geometry_oracle.c" \
    "$BUILD"/*.o "$BUILD/math/release/libgeometry_oracle_math.a" -lm
"${PYTHON:-python3}" - "$FIX/rtklib_integrity_geometry.json" "$BUILD/oracle" "$BUILD/portable-oracle" <<'PY'
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
path = Path(sys.argv[1])
document = json.loads(path.read_text())
cases = document["elevation_mask_cases"]
inputs = "".join(" ".join(map(repr, case["receiver_ecef_m"] + case["satellite_ecef_m"] + [case["mask_rad"]])) + "\n" for case in cases)
for implementation, executable in zip(("platform", "portable"), sys.argv[2:]):
    lines = subprocess.check_output([executable, "mask-ecef"], input=inputs, text=True).splitlines()
    for case, line in zip(cases, lines, strict=True):
        elevation, kept = line.split()
        case[implementation + "_ecef_elevation_rad"] = float(elevation)
        case[implementation + "_kept"] = bool(int(kept))
# Resolve actual one-ulp boundaries too, rather than only geometrically
# constructed angles: choose masks between the two independently calculated
# elevations, retaining the identical ECEF inputs.
extra = []
for case in cases:
    elevations = [case[name + "_ecef_elevation_rad"] for name in ("platform", "portable")]
    if elevations[0] == elevations[1]:
        continue
    for elevation in elevations:
        degrees = elevation / (math.pi / 180.0)
        for degrees in (math.nextafter(degrees, -math.inf), degrees, math.nextafter(degrees, math.inf)):
            radian_mask = degrees * (math.pi / 180.0)
            if (elevations[0] >= radian_mask) == (elevations[1] >= radian_mask):
                continue
            boundary = dict(case, mask_deg=degrees, mask_rad=radian_mask)
            line = " ".join(map(repr, boundary["receiver_ecef_m"] + boundary["satellite_ecef_m"] + [radian_mask])) + "\n"
            for name, executable in zip(("platform", "portable"), sys.argv[2:]):
                actual, kept = subprocess.check_output([executable, "mask-ecef"], input=line, text=True).split()
                boundary[name + "_ecef_elevation_rad"] = float(actual)
                boundary[name + "_kept"] = bool(int(kept))
            if boundary not in extra:
                extra.append(boundary)
cases.extend(extra)
document["provenance"]["boundary_math"] = "RTKLIB ecef2pos, geodist and satazel executed on identical stored ECEF inputs, linked separately with platform libm and Rust libm 0.2.16; direct elevation >= radian mask in both executions"
document["provenance"]["platform_math"] = platform.platform()
path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
PY
# PYTHON must provide mpmath for the independent conditioning calculation.
"${PYTHON:-python3}" - "$FIX/rtklib_antimeridian_dops.json" <<'PY'
import json
import sys
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80
path = Path(sys.argv[1])
document = json.loads(path.read_text())
rows = []
for azimuth, elevation in document["azel_rad"]:
    azimuth, elevation = mp.mpf(azimuth), mp.mpf(elevation)
    rows.append([mp.cos(elevation)*mp.sin(azimuth),
                 mp.cos(elevation)*mp.cos(azimuth), mp.sin(elevation), mp.mpf(1)])
h = mp.matrix(rows)
eigenvalues = mp.eigsy(h.T*h, eigvals_only=True)
document["normal_condition_number"] = float(eigenvalues[-1, 0]/eigenvalues[0, 0])
document["provenance"]["conditioning"] = "mpmath 80-digit eigenvalues of the RTKLIB azimuth/elevation normal matrix"
path.write_text(json.dumps(document, separators=(",", ":"))+"\n")
PY
