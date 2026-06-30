#!/usr/bin/env python3
"""Generate pyproj/PROJ ECEF-to-geodetic fixtures."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path

import pyproj


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_array(values) -> list[str]:
    return [f64_bits(v) for v in values]


ECEF = pyproj.CRS.from_epsg(4978)
GEODETIC = pyproj.CRS.from_epsg(4979)
TO_ECEF = pyproj.Transformer.from_crs(GEODETIC, ECEF, always_xy=True)
FROM_ECEF = pyproj.Transformer.from_crs(ECEF, GEODETIC, always_xy=True)


def lla_to_ecef(lon: float, lat: float, alt: float) -> tuple[float, float, float]:
    x, y, z = TO_ECEF.transform(lon, lat, alt)
    return float(x), float(y), float(z)


def make_inputs() -> list[tuple[float, float, float]]:
    rng = random.Random(0x47_33)
    inputs: list[tuple[float, float, float]] = []

    for _ in range(64):
        lon = rng.uniform(-180.0, 180.0)
        lat = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))
        alt = rng.choice(
            [
                rng.uniform(-500.0, 9000.0),
                rng.uniform(9000.0, 2_000_000.0),
                rng.uniform(2_000_000.0, 42_000_000.0),
                rng.uniform(-1_000_000.0, -500.0),
            ]
        )
        inputs.append(lla_to_ecef(lon, lat, alt))

    edge_lla = [
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 0.0),
        (-90.0, 0.0, 0.0),
        (180.0, 0.0, 0.0),
        (-180.0, 0.0, 0.0),
        (45.0, 89.999999, 0.0),
        (-135.0, -89.999999, 0.0),
        (12.0, 89.999999999, 1000.0),
        (-12.0, -89.999999999, -1000.0),
        (179.999999, 1e-9, 123.0),
        (-179.999999, -1e-9, -123.0),
        (0.0, 45.0, 42_000_000.0),
        (0.0, -45.0, -1_000_000.0),
        (123.456789, 0.0, 500.0),
    ]
    inputs.extend(lla_to_ecef(*entry) for entry in edge_lla)

    inputs.extend(
        [
            (0.0, 0.0, 0.0),
            (6_378_137.0, 0.0, 0.0),
            (-6_378_137.0, 0.0, 0.0),
            (0.0, 6_378_137.0, 0.0),
            (0.0, -6_378_137.0, 0.0),
            (0.0, 0.0, 6_356_752.314245179),
        ]
    )
    return inputs


def main() -> None:
    cases = []
    for case_id, xyz in enumerate(make_inputs()):
        lon, lat, alt = FROM_ECEF.transform(*xyz)
        cases.append(
            {
                "id": case_id,
                "ecef_bits": bits_array(xyz),
                "lonlatalt_bits": bits_array((float(lon), float(lat), float(alt))),
            }
        )

    payload = {
        "schema": "astrodynamics-geodetic-proj-v1",
        "pyproj_version": pyproj.__version__,
        "proj_version": pyproj.proj_version_str,
        "operation": "EPSG:4978 to EPSG:4979, always_xy=True",
        "cases": cases,
    }
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "geodetic" / "geodetic_proj.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
