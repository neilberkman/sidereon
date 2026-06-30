#!/usr/bin/env python3
"""Generate DTED lookup fixtures from public terrain tile identifiers."""

from __future__ import annotations

import json
import struct
from pathlib import Path

SYNTHETIC_TILE_ID = "n36_w107"
SYNTHETIC_TILE_NAME = "n36_w107_1arc_v3.dt2"
UHL_SIZE = 80
DSI_SIZE = 648
ACC_SIZE = 2700
DATA_OFFSET = UHL_SIZE + DSI_SIZE + ACC_SIZE


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def signed_magnitude(raw: int) -> int:
    if raw & 0x8000:
        return -int(raw & 0x7FFF)
    return int(raw)


def parse_ascii_int(buf: bytes) -> int:
    return int(buf.decode("ascii").strip())


def parse_coord(text: str) -> float:
    hemi = text[-1]
    sign = -1.0 if hemi in {"S", "W"} else 1.0
    coord = text[:-1]
    sec_idx = len(coord) - 4 if coord[-2] == "." else len(coord) - 2
    min_idx = sec_idx - 2
    degree = int(coord[:min_idx])
    minute = int(coord[min_idx:sec_idx])
    second = float(coord[sec_idx:])
    return sign * (degree + (minute + second / 60.0) / 60.0)


def dted_coord(value: float, is_lon: bool) -> bytes:
    hemi = ("E" if value >= 0.0 else "W") if is_lon else ("N" if value >= 0.0 else "S")
    value = abs(value)
    degree = int(value)
    minute_float = (value - degree) * 60.0
    minute = int(minute_float)
    second = (minute_float - minute) * 60.0
    return f"{degree:03}{minute:02}{second:02.0f}{hemi}".encode("ascii")


def signed_magnitude_bytes(value: int) -> bytes:
    if value < 0:
        raw = 0x8000 | abs(value)
    else:
        raw = value
    return raw.to_bytes(2, byteorder="big", signed=False)


def synthetic_elevation(lon_i: int, lat_i: int) -> int:
    return -20 + 7 * lon_i - 5 * lat_i + lon_i * lat_i


def write_synthetic_tile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lon_count = 5
    lat_count = 5
    uhl = bytearray(b" " * UHL_SIZE)
    uhl[0:4] = b"UHL1"
    uhl[4:12] = dted_coord(-107.0, is_lon=True)
    uhl[12:20] = dted_coord(36.0, is_lon=False)
    uhl[47:51] = f"{lon_count:04}".encode("ascii")
    uhl[51:55] = f"{lat_count:04}".encode("ascii")

    data = bytearray()
    for lon_i in range(lon_count):
        block = bytearray(12 + 2 * lat_count)
        block[0] = 0xAA
        block[4:6] = lon_i.to_bytes(2, byteorder="big")
        for lat_i in range(lat_count):
            sample = 8 + lat_i * 2
            block[sample : sample + 2] = signed_magnitude_bytes(
                synthetic_elevation(lon_i, lat_i)
            )
        checksum = sum(block[:-4])
        block[-4:] = checksum.to_bytes(4, byteorder="big", signed=True)
        data.extend(block)

    path.write_bytes(bytes(uhl) + (b" " * DSI_SIZE) + (b" " * ACC_SIZE) + bytes(data))


class Tile:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        if self.data[:4] != b"UHL1":
            raise ValueError(f"{path} missing UHL1 header")
        self.lon0 = parse_coord(self.data[4:12].decode("ascii"))
        self.lat0 = parse_coord(self.data[12:20].decode("ascii"))
        self.lon_count = parse_ascii_int(self.data[47:51])
        self.lat_count = parse_ascii_int(self.data[51:55])
        self.data_offset = 80 + 648 + 2700
        self.block_len = 12 + 2 * self.lat_count

    def elevation(self, lon: float, lat: float) -> int:
        lon_idx = round((lon - self.lon0) * (self.lon_count - 1))
        lat_idx = round((lat - self.lat0) * (self.lat_count - 1))
        block = self.data[self.data_offset + lon_idx * self.block_len : self.data_offset + (lon_idx + 1) * self.block_len]
        sample = 8 + lat_idx * 2
        raw = int.from_bytes(block[sample : sample + 2], byteorder="big", signed=False)
        return signed_magnitude(raw)

    def bilinear(self, lon: float, lat: float) -> float:
        lon_idx = (lon - self.lon0) * (self.lon_count - 1)
        lat_idx = (lat - self.lat0) * (self.lat_count - 1)
        lon_lo = int(lon_idx // 1)
        lat_lo = int(lat_idx // 1)
        fx = lon_idx - lon_lo
        fy = lat_idx - lat_lo
        z = 0.0
        for di, wx in ((0, 1.0 - fx), (1, fx)):
            for dj, wy in ((0, 1.0 - fy), (1, fy)):
                posting_lon = self.lon0 + (lon_lo + di) / (self.lon_count - 1)
                posting_lat = self.lat0 + (lat_lo + dj) / (self.lat_count - 1)
                z += (wx * wy) * self.elevation(posting_lon, posting_lat)
        return z


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dted"
    tile_path = out_dir / "tiles" / SYNTHETIC_TILE_NAME
    write_synthetic_tile(tile_path)

    tile = Tile(tile_path)
    nearest_cases = []
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    for lon_frac in fractions:
        for lat_frac in fractions:
            lon = tile.lon0 + lon_frac
            lat = tile.lat0 + lat_frac
            nearest_cases.append(
                {
                    "tile_id": SYNTHETIC_TILE_ID,
                    "longitude_bits": f64_bits(lon),
                    "latitude_bits": f64_bits(lat),
                    "elevation_bits": f64_bits(float(tile.elevation(lon, lat))),
                }
            )

    bilinear_cases = []
    for lon_frac, lat_frac in ((0.125, 0.125), (0.375, 0.625), (0.625, 0.375), (0.875, 0.875)):
        lon = tile.lon0 + lon_frac
        lat = tile.lat0 + lat_frac
        bilinear_cases.append(
            {
                "tile_id": SYNTHETIC_TILE_ID,
                "longitude_bits": f64_bits(lon),
                "latitude_bits": f64_bits(lat),
                "elevation_bits": f64_bits(tile.bilinear(lon, lat)),
            }
        )

    payload = {
        "schema": "gnss-dted-points-v1",
        "source": {
            "tile_id": SYNTHETIC_TILE_ID,
            "tile_path": f"tiles/{SYNTHETIC_TILE_NAME}",
            "format": "DTED UHL/DSI/ACC/data-record byte layout",
            "elevation_formula": "z_m = -20 + 7 * lon_i - 5 * lat_i + lon_i * lat_i",
        },
        "nearest_cases": nearest_cases,
        "bilinear_cases": bilinear_cases,
    }
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dted" / "dted_points.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
