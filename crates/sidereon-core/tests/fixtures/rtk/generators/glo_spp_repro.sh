#!/usr/bin/env bash
# Reproduce the GLONASS SPP oracle reference `esbc_gps_glonass_spp_l1_demo5.pos`.
#
# Reference solver: RTKLIB-demo5 (rtklibexplorer / EX) rnx2rtkp, version
# "EX 2.5.1", built locally from commit 968da9a.
#
# IMPORTANT GLONASS-nav workaround (reference generation only):
# The committed nav file ESBC00DNK_R_20201770000_01D_RN.rnx is a real RINEX 3.05
# GLONASS product: a "3.05" header and FIVE physical lines per record (epoch +
# FOUR broadcast-orbit lines). The fourth orbit line is the one 3.05 added over
# 3.04 (status flags, the L1/L2 group-delay difference dtaun, URAI, health), and
# gfzrnx wrote its dtaun field as the "unavailable" sentinel .999999999999e+09.
#
# RTKLIB EX 2.5.1 reading the 3.05 GLONASS header parses dtaun from that fourth
# orbit line; the ~1e9 sentinel becomes a garbage L1/L2 group delay that
# corrupts the GLONASS pseudorange (prange() applies -dtaun*c) and fails the
# epoch. So the reference was generated against a COPY of the RN file with only
# its header version edited 3.05 -> 3.04. With a 3.04 header RTKLIB reads three
# GLONASS orbit lines and skips the fourth physical line, leaving dtaun = 0 --
# which is exactly what sidereon does (the basic single-frequency GLONASS user
# applies no L1/L2 group delay). sidereon parses the COMMITTED, unedited 3.05 RN
# file and never reads dtaun (it ignores the fourth orbit line entirely; see the
# test committed_rn_fixture_is_rinex_305_five_line_layout_parsed_correctly), so
# both sides consume the identical broadcast GLONASS ephemeris.
#
# The committed RN file is NOT modified by this script. Only a throwaway copy is
# edited. The .pos is a committed artifact; this script documents how it was made
# (it requires a local RTKLIB-demo5 build, not present in this repo).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/../.."          # tests/fixtures
RNX="${RNX:-rnx2rtkp}"     # path to the demo5 rnx2rtkp binary
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

OBS="$FIX/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
MN="$FIX/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
RN="$FIX/nav/ESBC00DNK_R_20201770000_01D_RN.rnx"

# Header-only 3.05 -> 3.04 edit on a throwaway copy (only the version field on
# the "RINEX VERSION / TYPE" line changes; the record body is untouched).
RN_304="$WORK/RN_304.rnx"
sed '1 s/^     3.05/     3.04/' "$RN" > "$RN_304"

"$RNX" -x 2 -k "$HERE/glo_spp.conf" \
  -o "$HERE/esbc_gps_glonass_spp_l1_demo5.pos" \
  "$OBS" "$MN" "$RN_304"

echo "Wrote $HERE/esbc_gps_glonass_spp_l1_demo5.pos"
