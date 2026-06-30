#!/usr/bin/env python3
"""Generate ZWD troposphere fixtures from public formulas."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path

import pyproj


EARTH_RADIUS_M = 6_371_000.0
TO_ECEF = pyproj.Transformer.from_crs(4979, 4978, always_xy=True)


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def bits_array(values) -> list[str]:
    return [f64_bits(v) for v in values]


def ecef(lon: float, lat: float, alt: float) -> tuple[float, float, float]:
    x, y, z = TO_ECEF.transform(lon, lat, alt)
    return float(x), float(y), float(z)


def dot_three(a, b) -> float:
    return a[2] * b[2] + (a[1] * b[1] + a[0] * b[0])


def unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n)


def mapping_fraction(a: float, b: float, c: float, sin_e: float) -> float:
    numerator = 1.0 + a / (1.0 + b / (1.0 + c))
    denominator = sin_e + a / (sin_e + b / (sin_e + c))
    return numerator / denominator


def niell(elevation_rad: float, latitude_deg: float, day_of_year: int, height_m: float):
    sin_e = max(elevation_rad.sin() if hasattr(elevation_rad, "sin") else math.sin(elevation_rad), 0.01)
    lat_rad = latitude_deg * math.pi / 180.0
    sin_lat = math.sin(lat_rad)
    phi = 2.0 * math.pi * (day_of_year - 1.0) / 365.25

    a_h = 2.65e-3 * (1.0 - 0.0025 * sin_lat * sin_lat)
    b_h = 2.3e-4 * (1.0 - 0.0025 * sin_lat * sin_lat)
    c_h = 1.2e-4 * (1.0 - 0.0025 * sin_lat * sin_lat)
    a_h += 0.0005 * math.cos(phi) * (1.0 - 0.5 * sin_lat * sin_lat)
    b_h += 0.0001 * math.cos(phi) * (1.0 - 0.5 * sin_lat * sin_lat)
    c_h += 0.00005 * math.cos(phi) * (1.0 - 0.5 * sin_lat * sin_lat)

    a_w = 1.5e-2 * (1.0 - 0.01 * abs(sin_lat))
    b_w = 8.3e-3 * (1.0 - 0.01 * abs(sin_lat))
    c_w = 1.0e-3 * (1.0 - 0.01 * abs(sin_lat))
    a_w += 0.005 * math.cos(phi) * (1.0 - abs(sin_lat))
    b_w += 0.003 * math.cos(phi) * (1.0 - abs(sin_lat))
    c_w += 0.0005 * math.cos(phi) * (1.0 - abs(sin_lat))

    if height_m > 0.0:
        height_factor = math.exp(-height_m / 8000.0)
        a_h *= height_factor
        b_h *= height_factor
        c_h *= height_factor

    return mapping_fraction(a_h, b_h, c_h, sin_e), mapping_fraction(a_w, b_w, c_w, sin_e)


def delay(day_of_year: int, sat_xyz, receiver_xyz, receiver_lla) -> float:
    vec = tuple(sat_xyz[i] - receiver_xyz[i] for i in range(3))
    elevation_rad = math.asin(dot_three(unit(vec), unit(receiver_xyz)))
    latitude = receiver_lla[1]
    altitude = min(max(receiver_lla[2], -500.0), 9000.0)
    pressure = 1013.25 * (1.0 - 2.25577e-5 * altitude) ** 5.2559
    lat_rad = latitude * math.pi / 180.0
    zhd = 0.0022768 * pressure / (1.0 - 0.00266 * math.cos(2.0 * lat_rad) - 2.8e-7 * altitude)
    zwd = 0.25 * math.exp(-altitude / 2000.0)
    map_h, map_w = niell(elevation_rad, latitude, day_of_year, altitude)
    return zhd * map_h + zwd * map_w


def main() -> None:
    rng = random.Random(0x5A_57)
    cases = []
    for idx in range(80):
        lon = rng.uniform(-170.0, 170.0)
        lat = rng.uniform(-70.0, 70.0)
        alt = rng.uniform(-250.0, 4500.0)
        day = 1 + (idx * 17) % 365
        receiver_xyz = ecef(lon, lat, alt)
        az = math.radians((idx * 37) % 360)
        el = math.radians(5.0 + (idx * 11) % 80)
        east = (-math.sin(math.radians(lon)), math.cos(math.radians(lon)), 0.0)
        north = (
            -math.sin(math.radians(lat)) * math.cos(math.radians(lon)),
            -math.sin(math.radians(lat)) * math.sin(math.radians(lon)),
            math.cos(math.radians(lat)),
        )
        up = unit(receiver_xyz)
        direction = tuple(
            math.cos(el) * (math.cos(az) * north[i] + math.sin(az) * east[i]) + math.sin(el) * up[i]
            for i in range(3)
        )
        sat_xyz = tuple(receiver_xyz[i] + direction[i] * (EARTH_RADIUS_M + 20_200_000.0) for i in range(3))
        cases.append(
            {
                "name": f"zwd_{idx:02d}",
                "day_of_year": day,
                "sat_xyz_bits": bits_array(sat_xyz),
                "receiver_xyz_bits": bits_array(receiver_xyz),
                "receiver_lonlatalt_bits": bits_array((lon, lat, alt)),
                "delay_bits": f64_bits(delay(day, sat_xyz, receiver_xyz, (lon, lat, alt))),
            }
        )

    payload = {
        "schema": "gnss-tropo-zwd-v1",
        "reference": "Davis-Bevis-style ZWD with Niell mapping",
        "pyproj_version": pyproj.__version__,
        "cases": cases,
    }
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tropo_zwd" / "tropo_zwd.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
