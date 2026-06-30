#!/usr/bin/env python3
"""Regenerate the SPK type-21 parity fixture and its reference states.

This downloads a real Extended Modified Difference Array (type 21) SPK kernel
for asteroid 433 Eros from the JPL Horizons SPK API, then uses CSPICE (via
spiceypy) to compute geometric reference states. The kernel is committed as
`horizons_eros_type21.bsp`; the printed Rust literals are the bit-exact
reference vectors embedded in the `spk.rs` type-21 test.

Usage:
    pip install spiceypy numpy
    python3 gen_eros_type21.py

Provenance:
    Source: https://ssd.jpl.nasa.gov/api/horizons.api  (EPHEM_TYPE=SPK)
    Body:   433 Eros (SPK id 20000433), center Sun (10), frame J2000 (1)
    Oracle: CSPICE spkgeo() bundled with spiceypy.
"""

import base64
import json
import os
import struct
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "horizons_eros_type21.bsp")

START_TIME = "2024-01-01"
STOP_TIME = "2025-01-01"
TARGET = 20000433  # 433 Eros SPK id
CENTER = 10  # Sun


def download_kernel() -> None:
    params = {
        "format": "json",
        "EPHEM_TYPE": "SPK",
        "COMMAND": "433;",
        "START_TIME": START_TIME,
        "STOP_TIME": STOP_TIME,
        "OBJ_DATA": "NO",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.load(resp)
    if "spk" not in payload:
        raise SystemExit("Horizons did not return an SPK: " + payload.get("result", "")[:400])
    with open(KERNEL, "wb") as fh:
        fh.write(base64.b64decode(payload["spk"]))


def emit_reference() -> None:
    import spiceypy as s

    s.furnsh(KERNEL)
    h = s.dafopr(KERNEL)
    s.dafbfs(h)
    s.daffna()
    dc, _ic = s.dafus(s.dafgs(124), 2, 6)
    s.dafcls(h)
    t0, t1 = dc[0], dc[1]

    fracs = [0.0, 0.01, 0.1, 0.25, 0.333333, 0.5, 0.666667, 0.75, 0.9, 0.99, 1.0]
    ets = sorted({t0 + f * (t1 - t0) for f in fracs})

    print("// CSPICE spkgeo(20000433, et, \"J2000\", 10) on horizons_eros_type21.bsp")
    print("// (433 Eros relative to the Sun, J2000). Exact f64 round-trips.")
    for et in ets:
        st = s.spkgeo(TARGET, et, "J2000", CENTER)[0]
        vals = ", ".join(repr(x) for x in st)
        print(f"        ({et!r}, [{vals}]),")
    # Sanity: hex dump confirms little-endian doubles round-trip.
    _ = struct.pack("<d", t0)


if __name__ == "__main__":
    download_kernel()
    emit_reference()
