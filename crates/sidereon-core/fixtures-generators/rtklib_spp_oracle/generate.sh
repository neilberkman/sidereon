#!/usr/bin/env bash
# Regenerate tests/fixtures/rtk/rtklib_spp_selection_oracle.json: RTKLIB `pntpos`
# single-point solutions of the ESBC and WTZR 120-epoch RINEX fixtures from four
# initial positions each (see rtklib_spp_oracle.c), with the troposphere corrected
# and uncorrected. Also regenerate tests/fixtures/rtk/rtklib_spp_fde_oracle.json:
# RTKLIB `raim_fde` exclusions of faulted copies (single and paired faults) of every
# twelfth ESBC epoch.
#
# usage: RTKLIB_SRC=/path/to/RTKLIB/src ./generate.sh
#
# RTKLIB_SRC is the `src` directory of RTKLIB demo5 at commit
# 75a2e56275485b21a67bd35bc94bbeb8936e1a74. It is compiled with the options of the
# demo5 `rnx2rtkp` makefile, less tracing; the fixtures are read by the Rust tests
# `spp_selection_matches_rtklib_pntpos_from_every_initial_position` and
# `fde_spp_matches_rtklib_raim_fde_on_faulted_epochs`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../../tests/fixtures"
OUT="$FIX/rtk/rtklib_spp_selection_oracle.json"
FDE_OUT="$FIX/rtk/rtklib_spp_fde_oracle.json"
QUALITY_OUT="$FIX/rtk/rtklib_quality_fde_100m.json"
MULTI_OUT="$FIX/rtk/rtklib_multiconstellation_fde_100m.json"
EDGE_OUT="$FIX/rtk/rtklib_spp_edge_cases.json"
FDE_STRIDE=12
SRC="${RTKLIB_SRC:?set RTKLIB_SRC to the RTKLIB demo5 src directory}"
PIN="75a2e56275485b21a67bd35bc94bbeb8936e1a74"
RECORDED_PIN="${RTKLIB_COMMIT:-$PIN}"
if [[ "$RECORDED_PIN" != "$PIN" ]]; then
    printf 'RTKLIB_COMMIT is %s, expected %s\n' "$RECORDED_PIN" "$PIN" >&2
    exit 1
fi
if [[ ! -f "$SRC/rtklib.h" || ! -f "$SRC/pntpos.c" ]]; then
    printf 'RTKLIB_SRC must name an RTKLIB src directory\n' >&2
    exit 1
fi
BUILD="$(mktemp -d)"
OUT_TMP="$(mktemp "$FIX/rtk/.rtklib_spp_selection_oracle.json.XXXXXX")"
FDE_OUT_TMP="$(mktemp "$FIX/rtk/.rtklib_spp_fde_oracle.json.XXXXXX")"
QUALITY_RAW="$(mktemp "$FIX/rtk/.rtklib_quality_raw.json.XXXXXX")"
QUALITY_OUT_TMP="$(mktemp "$FIX/rtk/.rtklib_quality_fde_100m.json.XXXXXX")"
MULTI_RAW="$(mktemp "$FIX/rtk/.rtklib_multi_raw.json.XXXXXX")"
MULTI_OUT_TMP="$(mktemp "$FIX/rtk/.rtklib_multiconstellation_fde_100m.json.XXXXXX")"
EDGE_RAW="$(mktemp "$FIX/rtk/.rtklib_spp_edges_raw.json.XXXXXX")"
EDGE_OUT_TMP="$(mktemp "$FIX/rtk/.rtklib_spp_edge_cases.json.XXXXXX")"
trap 'rm -rf "$BUILD"; rm -f "$OUT_TMP" "$FDE_OUT_TMP" "$QUALITY_RAW" "$QUALITY_OUT_TMP" "$MULTI_RAW" "$MULTI_OUT_TMP" "$EDGE_RAW" "$EDGE_OUT_TMP"' EXIT

# rtkcmn.c defines _POSIX_C_SOURCE, which on macOS hides snprintf unless
# _DARWIN_C_SOURCE is also defined; elsewhere the macro has no effect.
OPTS="-DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE"
CFLAGS="-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -w -I$SRC $OPTS"
for unit in rtkcmn trace rinex rtkpos postpos solution lambda geoid sbas preceph \
    ephemeris options ppp ppp_ar rtcm rtcm2 rtcm3 rtcm3e ionex tides sofa; do
    cc $CFLAGS -c "$SRC/$unit.c" -o "$BUILD/$unit.o"
done
cc $CFLAGS -o "$BUILD/rtklib_spp_oracle" "$HERE/rtklib_spp_oracle.c" "$BUILD"/*.o -lm

NAV="$FIX/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
ESBC_OBS="$FIX/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx"
WTZR_OBS="$FIX/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}
{
    printf '{"generator": "fixtures-generators/rtklib_spp_oracle/generate.sh",\n'
    printf ' "rtklib": "rtklibexplorer/RTKLIB demo5 75a2e56275485b21a67bd35bc94bbeb8936e1a74",\n'
    printf ' "nav": "nav/ESBC00DNK_R_20201770000_01D_MN.rnx",\n'
    printf ' "input_sha256": {"nav": "%s", "obs": {"ESBC": "%s", "WTZR": "%s"}},\n' \
        "$(sha256_file "$NAV")" "$(sha256_file "$ESBC_OBS")" "$(sha256_file "$WTZR_OBS")"
    printf ' "runs": [\n'
    "$BUILD/rtklib_spp_oracle" esbc_iono_tropo \
        "$ESBC_OBS" "$NAV" 1 1
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" wtzr_iono_tropo \
        "$WTZR_OBS" "$NAV" 1 1
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" esbc_tropo \
        "$ESBC_OBS" "$NAV" 0 1
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" esbc_iono \
        "$ESBC_OBS" "$NAV" 1 0
    printf ',\n'
    "$BUILD/rtklib_spp_oracle" wtzr_iono \
        "$WTZR_OBS" "$NAV" 1 0
    printf ']}\n'
} > "$OUT_TMP"
{
    printf '{"generator": "fixtures-generators/rtklib_spp_oracle/generate.sh",\n'
    printf ' "rtklib": "rtklibexplorer/RTKLIB demo5 75a2e56275485b21a67bd35bc94bbeb8936e1a74",\n'
    printf ' "nav": "nav/ESBC00DNK_R_20201770000_01D_MN.rnx",\n'
    printf ' "obs": "obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx",\n'
    printf ' "input_sha256": {"nav": "%s", "obs": "%s"},\n' \
        "$(sha256_file "$NAV")" "$(sha256_file "$ESBC_OBS")"
    printf ' "run": '
    "$BUILD/rtklib_spp_oracle" fde esbc_iono_tropo \
        "$ESBC_OBS" "$NAV" 1 1 "$FDE_STRIDE"
    printf '}\n'
} > "$FDE_OUT_TMP"
"$BUILD/rtklib_spp_oracle" quality_fde esbc_iono_tropo \
    "$ESBC_OBS" "$NAV" 1 1 120 > "$QUALITY_RAW"
python3 "$HERE/select_quality_cases.py" "$QUALITY_RAW" "$QUALITY_OUT_TMP" \
    "$ESBC_OBS" "$NAV"
"$BUILD/rtklib_spp_oracle" multi_fde esbc_multi \
    "$ESBC_OBS" "$NAV" 1 1 120 > "$MULTI_RAW"
python3 "$HERE/select_quality_cases.py" "$MULTI_RAW" "$MULTI_OUT_TMP" \
    "$ESBC_OBS" "$NAV" --compact-multiconstellation
"$BUILD/rtklib_spp_oracle" edges esbc_edges "$ESBC_OBS" "$NAV" > "$EDGE_RAW"
python3 - "$EDGE_RAW" "$EDGE_OUT_TMP" "$ESBC_OBS" "$NAV" "$HERE/rtklib_spp_oracle.c" <<'PY'
import hashlib, json, pathlib, sys
raw, out, obs, nav, source = map(pathlib.Path, sys.argv[1:])
document = json.loads(raw.read_text())
document["provenance"] = {
    "tool": "rtklibexplorer/RTKLIB demo5",
    "commit": "75a2e56275485b21a67bd35bc94bbeb8936e1a74",
    "generator": "fixtures-generators/rtklib_spp_oracle/rtklib_spp_oracle.c edges",
    "generator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "compiler_flags": "-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE",
    "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (obs, nav)},
}
out.write_text(json.dumps(document, indent=2) + "\n")
PY
mv -f "$OUT_TMP" "$OUT"
mv -f "$FDE_OUT_TMP" "$FDE_OUT"
mv -f "$QUALITY_OUT_TMP" "$QUALITY_OUT"
mv -f "$MULTI_OUT_TMP" "$MULTI_OUT"
mv -f "$EDGE_OUT_TMP" "$EDGE_OUT"
echo "wrote $OUT"
echo "wrote $FDE_OUT"
echo "wrote $QUALITY_OUT"
echo "wrote $MULTI_OUT"
echo "wrote $EDGE_OUT"
