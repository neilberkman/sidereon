#!/usr/bin/env python3
"""Multi-system (per-constellation clock) DOP parity reference.

The single-system DOP golden (`dop_golden.json`) pins the 4x4 cofactor inverse
to 0 ULP. This generator covers the *multi-system* path: one receiver-clock
column per GNSS, so the cofactor matrix is `(3 + n_clocks) x (3 + n_clocks)` and
the per-constellation TDOP is `sqrt(Q[3+i][3+i])`.

`sidereon-core`'s `dop_multi` forms `Q` with a dense symmetric inverse
(`invert_symmetric_pd`, a Cholesky solve), NOT the explicit cofactor expansion,
so this is a TIGHT-TOLERANCE agreement target (~1e-12 relative), not a 0-ULP
target. The reference here uses `numpy.linalg.inv` (an LU solve); the two dense
inverses agree to a few ULP of the values, well inside the asserted tolerance.

Recipe (matches `dop_multi` term for term):
  H row k = [-ex, -ey, -ez, <one-hot over the n_clocks clock columns>]
  A = H^T W H   (W = diag(weights))
  Q = A^-1
  R(lat,lon) = [[-sl, cl, 0], [-sp*cl, -sp*sl, cp], [cp*cl, cp*sl, sp]]
  ENU position block = R Q[0:3,0:3] R^T
  GDOP = sqrt(trace(Q))                 (position block + every clock)
  PDOP = sqrt(qE + qN + qU)
  HDOP = sqrt(qE + qN)
  VDOP = sqrt(qU)
  TDOP = sqrt(Q[3][3])                  (reference clock)
  per-system TDOP[i] = sqrt(Q[3+i][3+i])

Line-of-sight unit vectors are built from topocentric az/el through the same
ENU->ECEF rotation `sidereon`'s `line_of_sight_from_az_el_deg` uses, so the
emitted ECEF vectors are exactly what the Rust port consumes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dop_multi_golden.json"


def hx(value: float) -> str:
    """Python `float.hex()` rendering (round-trips the exact f64)."""
    return float(value).hex()


def enu_to_ecef_rotation(lat_rad: float, lon_rad: float):
    """ECEF->ENU rotation R; ENU->ECEF is R^T (used to map LOS into ECEF)."""
    sp, cp = math.sin(lat_rad), math.cos(lat_rad)
    sl, cl = math.sin(lon_rad), math.cos(lon_rad)
    return np.array(
        [
            [-sl, cl, 0.0],
            [-sp * cl, -sp * sl, cp],
            [cp * cl, cp * sl, sp],
        ]
    )


def los_ecef_from_az_el(az_deg: float, el_deg: float, r_ecef_to_enu: np.ndarray):
    az, el = math.radians(az_deg), math.radians(el_deg)
    cos_el = math.cos(el)
    enu = np.array([cos_el * math.sin(az), cos_el * math.cos(az), math.sin(el)])
    # e = R^T @ enu  (sidereon line_of_sight_from_az_el_deg)
    return r_ecef_to_enu.T @ enu


def build_case(name: str, note: str, lat_deg: float, lon_deg: float, sats, n_clocks: int):
    """sats: list of (az_deg, el_deg, weight, clock_index)."""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    r = enu_to_ecef_rotation(lat_rad, lon_rad)

    los = [los_ecef_from_az_el(az, el, r) for (az, el, _w, _c) in sats]
    weights = [w for (_az, _el, w, _c) in sats]
    clock_index = [c for (_az, _el, _w, c) in sats]

    p = 3 + n_clocks
    a = np.zeros((p, p))
    for k, e in enumerate(los):
        row = np.zeros(p)
        row[0], row[1], row[2] = -e[0], -e[1], -e[2]
        row[3 + clock_index[k]] = 1.0
        a += weights[k] * np.outer(row, row)

    q = np.linalg.inv(a)

    qpos = q[0:3, 0:3]
    enu = r @ qpos @ r.T
    qe, qn, qu = enu[0, 0], enu[1, 1], enu[2, 2]

    gdop = math.sqrt(np.trace(q))
    pdop = math.sqrt(qe + qn + qu)
    hdop = math.sqrt(qe + qn)
    vdop = math.sqrt(qu)
    tdop = math.sqrt(q[3, 3])
    per_system_tdop = [math.sqrt(q[3 + i, 3 + i]) for i in range(n_clocks)]

    return {
        "name": name,
        "note": note,
        "inputs": {
            "lat_deg_repr": repr(lat_deg),
            "lon_deg_repr": repr(lon_deg),
            "lat_rad": hx(lat_rad),
            "lon_rad": hx(lon_rad),
            "n_clocks": n_clocks,
            "clock_index": clock_index,
            "weights": [hx(w) for w in weights],
            "los_ecef": [[hx(e[0]), hx(e[1]), hx(e[2])] for e in los],
        },
        "expect": {
            "gdop": hx(gdop),
            "pdop": hx(pdop),
            "hdop": hx(hdop),
            "vdop": hx(vdop),
            "tdop": hx(tdop),
            "per_system_tdop": [hx(v) for v in per_system_tdop],
        },
    }


def main() -> None:
    cases = []

    # GPS (clock 0) + Galileo (clock 1): 7 satellites, 3 + 2 = 5 parameters.
    cases.append(
        build_case(
            "gps_galileo_seven",
            "GPS+Galileo, two clock columns, mixed weights",
            lat_deg=45.0,
            lon_deg=10.0,
            sats=[
                (10.0, 78.0, 1.0, 0),
                (75.0, 35.0, 0.9, 0),
                (140.0, 22.0, 1.1, 0),
                (200.0, 48.0, 0.7, 0),
                (250.0, 30.0, 1.3, 1),
                (305.0, 60.0, 0.8, 1),
                (340.0, 18.0, 1.0, 1),
            ],
            n_clocks=2,
        )
    )

    # GPS (clock 0) + BeiDou (clock 1): 8 satellites, well spread.
    cases.append(
        build_case(
            "gps_beidou_eight",
            "GPS+BeiDou, two clock columns, mixed weights",
            lat_deg=-22.5,
            lon_deg=133.0,
            sats=[
                (5.0, 70.0, 1.0, 0),
                (55.0, 40.0, 1.2, 0),
                (110.0, 25.0, 0.8, 0),
                (165.0, 55.0, 1.0, 0),
                (210.0, 33.0, 0.9, 1),
                (260.0, 65.0, 1.1, 1),
                (300.0, 20.0, 0.7, 1),
                (350.0, 45.0, 1.0, 1),
            ],
            n_clocks=2,
        )
    )

    # GPS + Galileo + BeiDou: three clock columns, 9 satellites.
    cases.append(
        build_case(
            "gps_galileo_beidou_nine",
            "GPS+Galileo+BeiDou, three clock columns, unit weights",
            lat_deg=51.5,
            lon_deg=-0.13,
            sats=[
                (20.0, 72.0, 1.0, 0),
                (90.0, 38.0, 1.0, 0),
                (150.0, 28.0, 1.0, 0),
                (190.0, 55.0, 1.0, 1),
                (240.0, 31.0, 1.0, 1),
                (290.0, 62.0, 1.0, 1),
                (330.0, 24.0, 1.0, 2),
                (0.0, 84.0, 1.0, 2),
                (120.0, 47.0, 1.0, 2),
            ],
            n_clocks=3,
        )
    )

    doc = {
        "schema": "orbis-gnss-parity/dop_multi.v1",
        "purpose": (
            "Multi-system DOP: per-constellation TDOP = sqrt(Q[3+i][3+i]) from "
            "Q = (H^T W H)^-1 with one receiver-clock column per GNSS. Reference "
            "uses numpy.linalg.inv (a dense LU inverse); the Rust dop_multi uses a "
            "dense symmetric (Cholesky) inverse, so this is a TIGHT-TOLERANCE "
            "agreement target (~1e-12 relative), NOT 0 ULP. The single-system "
            "scalar DOP stays 0-ULP in dop_golden.json."
        ),
        "recipe": (
            "H row = [-ex,-ey,-ez, one-hot clock columns]; A = H^T W H; Q = A^-1; "
            "R(lat,lon)=[[-sl,cl,0],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]]; "
            "ENU = R Q[0:3,0:3] R^T; GDOP=sqrt(trace Q); PDOP=sqrt(qE+qN+qU); "
            "HDOP=sqrt(qE+qN); VDOP=sqrt(qU); TDOP=sqrt(Q[3][3]); "
            "per-system TDOP[i]=sqrt(Q[3+i][3+i])."
        ),
        "generator": {"numpy": np.__version__, "script": "generate_dop_multi.py"},
        "cases": cases,
    }

    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
