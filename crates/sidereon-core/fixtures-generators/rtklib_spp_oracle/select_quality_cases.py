#!/usr/bin/env python3
"""Select the compact 100 m real-epoch FDE corpus from the RTKLIB dump."""

import hashlib
import json
import pathlib
import sys
from collections import defaultdict


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
obs = pathlib.Path(sys.argv[3])
nav = pathlib.Path(sys.argv[4])
document = json.loads(source.read_text())
cases = [
    case
    for case in document["cases"]
    if case["faults"] and case["faults"][0]["bias_m"] == 100.0
]
if not any(case["mode"] == "single_fault" for case in cases):
    raise SystemExit("missing single 100 m fault")
if not any(case["mode"] == "two_fault" for case in cases):
    raise SystemExit("missing paired 100 m fault")
if len(sys.argv) == 6 and sys.argv[5] == "--compact-multiconstellation":
    primary = cases[0]
    primary = {
        key: primary[key]
        for key in (
            "epoch",
            "faults",
            "mode",
            "full_estpos_stat",
            "pntpos_stat",
            "stat",
            "excluded",
            "position_m",
            "geodetic_rad_m",
            "clock_m",
            "used",
        )
    } | {
        "candidates": [],
        "satellite_states": [
            {
                key: state[key]
                for key in (
                    "sat",
                    "variance_m2",
                    "az_rad",
                    "el_rad",
                    "reference_variance_m2",
                )
            }
            for state in cases[0]["satellite_states"]
        ],
    }
    tie_case = None
    for case in cases:
        by_rms = defaultdict(list)
        for candidate in case["candidates"]:
            if candidate["status"] == "solved":
                by_rms[float(candidate["rms_m"]).hex()].append(candidate["sat"])
        if any(len(satellites) > 1 for satellites in by_rms.values()):
            tie_case = {
                "epoch": case["epoch"],
                "faults": case["faults"],
                "mode": case["mode"],
                "candidates": [
                    {
                        "sat": candidate["sat"],
                        "status": candidate["status"],
                        "rms_m": candidate["rms_m"],
                    }
                    for candidate in case["candidates"]
                ],
            }
            break
    if tie_case is None:
        raise SystemExit("missing exact candidate RMS tie")
    cases = [primary, tie_case]
elif len(sys.argv) != 5:
    raise SystemExit(
        "usage: select_quality_cases.py input output obs nav [--compact-multiconstellation]"
    )
result = {
    "provenance": {
        "tool": "RTKLIB demo5",
        "commit": "75a2e56275485b21a67bd35bc94bbeb8936e1a74",
        "generator": "fixtures-generators/rtklib_spp_oracle/rtklib_spp_oracle.c",
        "generator_sha256": digest(pathlib.Path(__file__).with_name("rtklib_spp_oracle.c")),
        "selector": "fixtures-generators/rtklib_spp_oracle/select_quality_cases.py",
        "selector_sha256": digest(pathlib.Path(__file__)),
        "cflags": "-std=gnu99 -O2 -ffp-contract=off -fno-fast-math -w -DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=4 -DNEXOBS=3 -D_DARWIN_C_SOURCE",
        "inputs": {
            "obs": f"tests/fixtures/obs/{obs.name}",
            "obs_sha256": digest(obs),
            "nav": f"tests/fixtures/nav/{nav.name}",
            "nav_sha256": digest(nav),
        },
    },
    "guess": document["guess"],
    "cases": cases,
}
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
