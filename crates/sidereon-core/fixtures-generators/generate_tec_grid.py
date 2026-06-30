#!/usr/bin/env python3
"""Generate TEC grid fixtures from public IONEX files."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import struct
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


DEFAULT_IONEX_URLS = [
    "ftp://gssc.esa.int/gnss/products/ionex/2024/001/IGS0OPSFIN_20240010000_01D_02H_GIM.INX.gz",
]


def f64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def numbers(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", line[:60])]


def epoch_ns(line: str) -> int:
    year, month, day, hour, minute, second = [int(x) for x in line[:60].split()[:6]]
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def fetch(url: str, work: Path) -> Path:
    name = url.rsplit("/", 1)[-1]
    out = work / name
    urllib.request.urlretrieve(url, out)
    return out


def parse_ionex(path: Path) -> list[tuple[int, float, float, float]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    exponent = None
    for line in lines:
        if "EXPONENT" in line:
            exponent = int(line[:60].split()[0])
            break
    if exponent is None:
        raise ValueError(f"{path} has no EXPONENT header")
    scale = 10.0**exponent

    rows = []
    i = 0
    while i < len(lines):
        if "START OF TEC MAP" not in lines[i]:
            i += 1
            continue

        stamp = epoch_ns(lines[i + 1])
        i += 2
        while i < len(lines) and "END OF TEC MAP" not in lines[i]:
            if "LAT/LON1/LON2/DLON/H" not in lines[i]:
                i += 1
                continue

            lat, lon_first, lon_last, dlon, _height = numbers(lines[i])[:5]
            lons = np.arange(lon_first, lon_last + (0.1 if dlon > 0 else -0.1), dlon)
            i += 1
            vals = []
            while (
                i < len(lines)
                and "LAT/LON1/LON2/DLON/H" not in lines[i]
                and "END OF TEC MAP" not in lines[i]
            ):
                vals.extend(int(v) for v in lines[i].split())
                i += 1
            if len(vals) != len(lons):
                raise ValueError(f"{path} {stamp} latitude {lat} has {len(vals)} values")
            for lon, raw in zip(lons, vals):
                rows.append((stamp, float(lat), float(lon), float(raw) * scale))
        i += 1

    return rows


def build_grid(paths: list[Path]):
    first_by_key = {}
    for path in paths:
        for stamp, lat, lon, tec in parse_ionex(path):
            first_by_key.setdefault((stamp, lat, lon), tec)

    epochs = sorted({key[0] for key in first_by_key})
    lats = sorted({key[1] for key in first_by_key})
    lons = sorted({key[2] for key in first_by_key})
    values = np.empty((len(epochs), len(lats), len(lons)), dtype="f8")
    for epoch_idx, stamp in enumerate(epochs):
        for lat_idx, lat in enumerate(lats):
            for lon_idx, lon in enumerate(lons):
                values[epoch_idx, lat_idx, lon_idx] = first_by_key[(stamp, lat, lon)]
    return epochs, lats, lons, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ionex-url", action="append", default=None)
    args = parser.parse_args()
    urls = args.ionex_url or DEFAULT_IONEX_URLS

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        paths = [fetch(url, work) for url in urls]
        epochs, lats, lons, values = build_grid(paths)

    interp = RegularGridInterpolator(
        (np.asarray(epochs, dtype="f8"), np.asarray(lats, dtype="f8"), np.asarray(lons, dtype="f8")),
        values,
    )
    probes = []
    for idx, point in enumerate(
        [
            (epochs[0], lats[0], lons[0]),
            (epochs[-1], lats[-1], lons[-1]),
            ((epochs[0] + epochs[-1]) / 2.0, 0.0, 0.0),
        ]
    ):
        value = float(interp([point])[0])
        probes.append({"name": f"probe_{idx}", "point_bits": [f64_bits(v) for v in point], "value_bits": f64_bits(value)})

    payload = {
        "schema": "gnss-tec-grid-v1",
        "source_urls": urls,
        "epochs_bits": [f64_bits(v) for v in epochs],
        "lats_bits": [f64_bits(v) for v in lats],
        "lons_bits": [f64_bits(v) for v in lons],
        "values_bits": [f64_bits(v) for v in values.ravel(order="C")],
        "regular_grid_probes": probes,
    }
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tec_grid" / "tec_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
