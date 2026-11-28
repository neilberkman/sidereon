#!/usr/bin/env python3
"""Weighted DOP at the recorded SPP solution geometry."""

import argparse
import json
from math import sqrt
from pathlib import Path

from rust_libm_port import bits, cos, fb, sin

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def decode(value):
    return fb(int(value, 16))


def ordered_sum(terms):
    value = 0.0
    for term in terms:
        value += term
    return value


def invert_normal_matrix(matrix):
    """Expand the determinant along row zero, then divide the adjugate by it."""

    def minor(row, column):
        [[a, b, c], [d, e, f], [g, h, i]] = [
            [value for j, value in enumerate(values) if j != column]
            for k, values in enumerate(matrix) if k != row
        ]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    a, b, c, d = [matrix[0][column] * minor(0, column) for column in range(4)]
    det = a - b + c - d
    return [
        [(-1.0 if (row + column) % 2 else 1.0) * minor(column, row) / det
         for column in range(4)]
        for row in range(4)
    ]


def enu_variance(cofactor, direction):
    # Form R Q first, then multiply the row by the corresponding column of R^T.
    rq = [ordered_sum(direction[i] * cofactor[i][j] for i in range(3))
          for j in range(3)]
    return ordered_sum(rq[i] * direction[i] for i in range(3))


def dop(fixture):
    geometry = fixture["dop_geometry"]
    design = [[-decode(x) for x in los] + [1.0] for los in geometry["los_ecef"]]
    weights = [decode(fixture["used_sat_geometry"][sat]["weight"])
               for sat in fixture["used_sats"]]
    # Left-to-right satellite order for every (i, j) of H^T W H.
    normal = [
        [ordered_sum(row[i] * w * row[j] for row, w in zip(design, weights, strict=True))
         for j in range(4)]
        for i in range(4)
    ]
    q = invert_normal_matrix(normal)
    latitude = decode(geometry["latitude_rad"])
    longitude = decode(geometry["longitude_rad"])
    sphi, cphi = sin(latitude), cos(latitude)
    slam, clam = sin(longitude), cos(longitude)
    qe = enu_variance(q, [-slam, clam, 0.0])
    qn = enu_variance(q, [-sphi * clam, -sphi * slam, cphi])
    qu = enu_variance(q, [cphi * clam, cphi * slam, sphi])
    qt = q[3][3]
    return {
        name: f"0x{bits(sqrt(variance)):016x}"
        for name, variance in {
            "gdop": q[0][0] + q[1][1] + q[2][2] + qt,
            "pdop": qe + qn + qu,
            "hdop": qe + qn,
            "vdop": qu,
            "tdop": qt,
        }.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    geometries = json.loads(args.geometry.read_text()) if args.geometry else {}
    for path in sorted(FIXTURES.glob("spp_trace_*.json")):
        doc = json.loads(path.read_text())
        fixture = doc["fixture"]
        if fixture["level"] in geometries:
            fixture["dop_geometry"] = geometries[fixture["level"]]
        fixture["dop"] = dop(fixture)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
