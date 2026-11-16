#!/usr/bin/env python3
"""Derive per-cell discretization and arithmetic bounds for the CDF grid.

Expected CDFs remain the independent parallel-normal integrals. The ideal
64/128-by-16 perpendicular-radius rule is evaluated at 90 digits only to bound
its discretization error, including its finite radial cutoff. No engine is
loaded or executed. Run after generate_noncentral_grid.py.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp
from generate_noncentral_grid import DEST, normal_tail

mp.mp.dps = 90
NODES, WEIGHTS = mp.gauss_quadrature(16, "legendre")


def folded(r, d, upper):
    if upper:
        return normal_tail(r-d)+normal_tail(r+d)
    return (mp.erfc((d-r)/mp.sqrt(2))-mp.erfc((d+r)/mp.sqrt(2)))/2


def bound(row):
    if "quadrature_log_tail_error_bound" in row:
        return row
    x, lam, dof = mp.mpf(float(row["x"])), mp.mpf(float(row["lambda"])), row["dof"]
    error = mp.mpf(0)
    if dof > 1 and lam > 100 and row["tail"] != "chernoff_underflow":
        r, d, nu = mp.sqrt(x), mp.sqrt(lam), dof-1
        precision = max(1, d/r) if row["tail"] == "lower" else mp.mpf(1)
        limit = min(r, (mp.sqrt(nu)+40)/mp.sqrt(precision))
        angle_limit = mp.asin(limit/r)
        panels = 128 if nu > 16384 else 64
        half_width = angle_limit/(2*panels)
        density_constant = (1-mp.mpf(nu)/2)*mp.log(2)-mp.loggamma(mp.mpf(nu)/2)
        upper = row["tail"] == "upper"
        logs = []
        for panel in range(panels):
            midpoint = (2*panel+1)*half_width
            for node, weight in zip(NODES, WEIGHTS):
                angle = midpoint+half_width*node
                radius = r*mp.sin(angle)
                remaining = r*mp.cos(angle)
                logs.append(mp.log(weight*half_width)+density_constant+(nu-1)*mp.log(radius)
                            - radius*radius/2+mp.log(r*mp.cos(angle))
                            +mp.log(folded(remaining, d, upper)))
        if upper:
            outside = mp.gammainc(mp.mpf(nu)/2, x/2, mp.inf, regularized=True)
            if outside:
                logs.append(mp.log(outside))
        largest = max(logs)
        approximate_log_tail = largest+mp.log(mp.fsum(mp.exp(value-largest) for value in logs))
        error = abs(approximate_log_tail-mp.mpf(row["log_small_tail_or_bound"]))
    # The adaptive integral's relative error is <1e-35. A 30-significant-digit
    # serialized logarithm adds <1e-29*max(1,|log tail|) absolute error.
    # Retain four further decimal guard orders for both reference operations.
    guard = mp.mpf("1e-25")*max(1, abs(mp.mpf(row["log_small_tail_or_bound"])))
    row["quadrature_log_tail_error_bound"] = mp.nstr(error+guard, 25)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEST)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    doc = json.loads(args.path.read_text())
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if doc["provenance"].get("error_bound_generator_sha256") != digest:
        for row in doc["cases"]:
            row.pop("quadrature_log_tail_error_bound", None)
    if args.cache:
        previous_doc = json.loads(args.cache.read_text())
        cache = {(r["dof"], r["x"], r["lambda"]): r for r in previous_doc["cases"]} if previous_doc["provenance"].get("error_bound_generator_sha256") == digest else {}
        for row in doc["cases"]:
            previous = cache.get((row["dof"], row["x"], row["lambda"]))
            if previous and previous["log_cdf"] == row["log_cdf"] and "quadrature_log_tail_error_bound" in previous:
                row["quadrature_log_tail_error_bound"] = previous["quadrature_log_tail_error_bound"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(bound, doc["cases"])):
            doc["cases"][index] = row
            if index % 10 == 0:
                args.path.write_text(json.dumps(doc, indent=2)+"\n")
                print(index+1, len(doc["cases"]), flush=True)
    doc["provenance"]["error_bound_generator"] = "fixtures-generators/derive_noncentral_bounds.py"
    doc["provenance"]["error_bound_generator_sha256"] = digest
    doc["provenance"]["discretization_bound"] = "90-digit absolute difference in log smaller tail between the independent parallel-normal integral and the ideal 64/128-panel 16-node perpendicular-radius rule, including its conditionally scaled radial cutoff, plus 1e-25*max(1,abs(log tail)) for reference integration, summation and 30-digit serialization"
    args.path.write_text(json.dumps(doc, indent=2)+"\n")


if __name__ == "__main__":
    main()
