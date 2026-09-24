#!/usr/bin/env python3
"""Reproduce the retained CODE boundary rows from AIUB's pinned listing."""

from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen


URL = "https://www.aiub.unibe.ch/download/full_listing.csv"
SHA256 = "7ed6281e10693634669c31d1440459ed968c1d104bb0b8108c37190884d48278"
EXPECTED_BYTES = 41_575_382
HERE = Path(__file__).resolve().parent
BOUNDARIES = HERE / "aiub-code-legacy-boundaries-20260924.csv"
REQUIRED_PATHS = {
    "CODE/1995/CODG0010.95I.Z",
    "CODE/1997/CODG0320.97I.Z",
    "CODE/1997/CODG0330.97I.Z",
    "CODE/1997/CODG0340.97I.Z",
    "CODE/1997/CODG0540.97I.Z",
    "CODE/1997/CODG0550.97I.Z",
    "CODE/1997/CODG0560.97I.Z",
    "CODE/1998/CODG0860.98I.Z",
    "CODE/1998/CODG0870.98I.Z",
    "CODE/1998/CODG0880.98I.Z",
    "CODE/2002/CODG3060.02I.Z",
    "CODE/2002/CODG3070.02I.Z",
    "CODE/2002/CODG3080.02I.Z",
    "CODE/2014/CODG2910.14I.Z",
    "CODE/2014/CODG2920.14I.Z",
    "CODE/2014/CODG2930.14I.Z",
    "CODE/2022/CODG3290.22I.Z",
    "CODE/2022/CODG3300.22I.Z",
    "CODE_MGEX/BSWUSER52/2014/COM17733.CLK.Z",
    "CODE_MGEX/BSWUSER52/2022/COM22375.CLK.Z",
    "CODE_MGEX/BSWUSER52/2022/COM22376.CLK.Z",
    "CODE_MGEX/CODE/2014/COM17733.CLK.Z",
    "CODE_MGEX/CODE/2014/COM17733.EPH.Z",
    "CODE_MGEX/CODE/2017/COM19610.EPH.Z",
    "CODE_MGEX/CODE/2017/COM19620.CLK.Z",
    "CODE_MGEX/CODE/2022/COM22375.CLK.Z",
    "CODE_MGEX/CODE/2022/COM22375.EPH.Z",
    "CODE_MGEX/CODE/2022/COM22376.CLK.Z",
    "CODE_MGEX/CODE/2022/COM22376.EPH.Z",
}
ABSENT = {
    "CODE/1994/CODG3650.94I.Z",
    "CODE/2022/CODG3310.22I.Z",
    "CODE_MGEX/CODE/2013/COM17732.CLK.Z",
    "CODE_MGEX/CODE/2013/COM17732.EPH.Z",
    "CODE_MGEX/CODE/2022/COM22377.CLK.Z",
    "CODE_MGEX/CODE/2022/COM22377.EPH.Z",
}


def main() -> None:
    request = Request(URL, headers={"User-Agent": "Sidereon CODE listing audit"})
    with urlopen(request, timeout=120) as response:
        listing = response.read()
    digest = sha256(listing).hexdigest()
    if len(listing) != EXPECTED_BYTES or digest != SHA256:
        raise SystemExit(
            f"unexpected AIUB listing: {len(listing)} bytes, SHA-256 {digest}"
        )

    expected = BOUNDARIES.read_bytes()
    expected_paths = {line.split(b";", 1)[0].decode("ascii") for line in expected.splitlines()}
    if expected_paths != REQUIRED_PATHS:
        raise SystemExit(
            f"retained path set mismatch: missing={sorted(REQUIRED_PATHS - expected_paths)}, "
            f"unexpected={sorted(expected_paths - REQUIRED_PATHS)}"
        )

    rows = listing.splitlines(keepends=True)
    paths = {line.split(b";", 1)[0].decode("ascii"): line for line in rows}
    missing = expected_paths - paths.keys()
    unexpected_absent = ABSENT & paths.keys()
    if missing or unexpected_absent:
        raise SystemExit(
            f"boundary mismatch: missing={sorted(missing)}, "
            f"unexpected_absent={sorted(unexpected_absent)}"
        )
    retained = b"".join(
        line for line in rows if line.split(b";", 1)[0].decode("ascii") in expected_paths
    )
    if retained != expected:
        raise SystemExit("retained boundary rows differ from the pinned listing")
    print(f"verified {len(expected_paths)} retained paths; listing SHA-256 {digest}")


if __name__ == "__main__":
    main()
