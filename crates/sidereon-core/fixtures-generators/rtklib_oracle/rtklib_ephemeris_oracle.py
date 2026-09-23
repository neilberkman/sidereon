#!/usr/bin/env python3
"""Evaluate the committed broadcast goldens with RTKLIB's own ephemeris functions.

Writes one case per line for `rtklib_ephemeris_oracle.c` (RTKLIB `eph2pos`,
`eph2clk`, `geph2pos`, `geph2clk`, `seph2pos`, `seph2clk`, built against the Rust
`libm` crate), runs it, and writes a JSON report that pairs each RTKLIB output with
the value sidereon's golden states for it, as binary64 bit patterns, with the
difference in ULP. Cases:

- every legacy Keplerian case of `tests/fixtures/broadcast_golden.json` (CNAV has no
  RTKLIB counterpart): position, `eph2pos` clock (the golden's polynomial plus
  relativistic term) and `eph2clk` at the same epoch;
- every case of `tests/fixtures/glonass_golden.json`: `geph2pos` position (the
  golden's final state), `geph2pos` clock and `geph2clk`;
- the first SBAS frame of `tests/fixtures/nav/KMS300DNK_R_20221591000_01H_MN.rnx` at
  0, 60 and 360 s from `t0`: `seph2pos` (sidereon's value is formed here with the same
  expression `SbasRecord::position_at` uses) and `seph2clk`.

Build and run from this directory, on a host with a Rust toolchain and clang:

    cargo build --release --manifest-path libm_shim/Cargo.toml
    clang -std=c99 -O2 -ffp-contract=off -fno-fast-math -o rtklib_ephemeris_oracle \\
        rtklib_ephemeris_oracle.c libm_shim/target/release/libsidereon_libm_shim.a \\
        -lm -lpthread -ldl
    python3 rtklib_ephemeris_oracle.py --harness ./rtklib_ephemeris_oracle \\
        --output ../../tests/fixtures/rtklib_ephemeris_oracle.json

The committed report, `tests/fixtures/rtklib_ephemeris_oracle.json`, is read by the
Rust test `goldens_equal_the_rtklib_ephemeris_oracle`.

On macOS drop `-lpthread -ldl`. `-ffp-contract=off` keeps clang from fusing RTKLIB's
multiplies and adds, which it does by default on arm64.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BROADCAST = ROOT / "tests/fixtures/broadcast_golden.json"
GLONASS = ROOT / "tests/fixtures/glonass_golden.json"
SBAS_NAV = ROOT / "tests/fixtures/nav/KMS300DNK_R_20221591000_01H_MN.rnx"
SYSTEM_LETTER = {"GPS": "G", "GAL": "E", "BDS": "C", "QZS": "J", "IRN": "I"}
ELEMENT_ORDER = [
    "sqrt_a", "e", "m0", "delta_n", "omega0", "i0", "omega", "omega_dot", "idot",
    "cuc", "cus", "crc", "crs", "cic", "cis",
]
SBAS_OFFSETS_S = [0.0, 60.0, 360.0]


def to_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def hex_bits(value: float) -> str:
    return "0x%016x" % to_bits(value)


def from_hex_float(text: str) -> float:
    """A golden value: a `0x...` bit pattern or a Python `float.hex` string."""
    if "p" in text:
        return float.fromhex(text)
    return from_bits(int(text, 16))


def ulp_distance(a: float, b: float) -> int:
    def ordered(x: float) -> int:
        u = to_bits(x)
        return u if u < (1 << 63) else (1 << 63) - (u - (1 << 63)) - 1

    return abs(ordered(a) - ordered(b))


def keplerian_cases():
    data = json.loads(BROADCAST.read_text())
    for case in data["cases"]:
        letter = SYSTEM_LETTER[case["system"]]
        prn = int(case["sat"][1:])
        el = case["elements_hex"]
        ck = case["clock_hex"]
        fields = [case["t_sow_hex"], el["toe_sow"], ck["toc_sow"]]
        fields += [el[k] for k in ELEMENT_ORDER]
        fields += [ck["af0"], ck["af1"], ck["af2"]]
        line = " ".join(["K", case["name"], letter, str(prn)] + fields)
        exp = case["expect_hex"]
        poly = from_hex_float(exp["dt_clock_poly_s"])
        rel = from_hex_float(exp["dt_rel_s"])
        sidereon = {
            "x_m": from_hex_float(exp["x_m"]),
            "y_m": from_hex_float(exp["y_m"]),
            "z_m": from_hex_float(exp["z_m"]),
            "eph2pos_dts": poly + rel,
        }
        yield case["name"], line, sidereon


def glonass_cases():
    data = json.loads(GLONASS.read_text())
    for case in data["cases"]:
        inp = case["inputs"]
        values = [inp["tk_s"], inp["clk_bias"], inp["gamma_n"]]
        values += inp["pos_m"] + inp["vel_m_s"] + inp["acc_m_s2"]
        line = " ".join(
            ["G", case["name"]] + [hex_bits(float.fromhex(v)) for v in values]
        )
        exp = case["expect"]
        final = [float.fromhex(v) for v in exp["final_state"]]
        sidereon = {
            "x_m": final[0],
            "y_m": final[1],
            "z_m": final[2],
            "geph2pos_dts": float.fromhex(exp["position_clock_offset_s"]),
            "geph2clk_dts": float.fromhex(exp["clock_offset_s"]),
        }
        yield case["name"], line, sidereon


def sbas_cases():
    lines = SBAS_NAV.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("> EPH S"))
    block = lines[start + 1 : start + 5]

    def field(line: str, index: int) -> float:
        begin = 23 + 19 * (index - 1) if line is block[0] else 4 + 19 * index
        return float(line[begin : begin + 19].replace("D", "E"))

    af0 = field(block[0], 1)
    af1 = field(block[0], 2)
    # RTKLIB `decode_seph`: pos/vel/acc in km, km/s, km/s^2 times 1E3.
    pos = [field(block[k], 0) * 1e3 for k in (1, 2, 3)]
    vel = [field(block[k], 1) * 1e3 for k in (1, 2, 3)]
    acc = [field(block[k], 2) * 1e3 for k in (1, 2, 3)]
    sat = block[0][:3]
    for offset in SBAS_OFFSETS_S:
        name = f"{sat.lower()}_t0_plus_{int(offset)}s"
        values = [offset] + pos + vel + acc + [af0, af1]
        line = " ".join(["S", name] + [hex_bits(v) for v in values])
        sidereon = {
            f"{axis}_m": pos[i] + vel[i] * offset + acc[i] * offset * offset / 2.0
            for i, axis in enumerate("xyz")
        }
        sidereon["seph2pos_dts"] = af0 + af1 * offset
        yield name, line, sidereon


def main() -> int:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--harness", required=True, help="the built oracle binary")
    parser.add_argument("--output", required=True, help="the JSON report to write")
    args = parser.parse_args()

    cases = list(keplerian_cases()) + list(glonass_cases()) + list(sbas_cases())
    stdin = "".join(line + "\n" for _, line, _ in cases)
    run = subprocess.run(
        [args.harness], input=stdin, capture_output=True, text=True, check=True
    )
    rtklib = {}
    for out in run.stdout.splitlines():
        kind, name, *pairs = out.split()
        rtklib[name] = {k: int(v, 16) for k, v in (p.split("=") for p in pairs)}

    report = {
        "rtklib": "demo5 75a2e56275485b21a67bd35bc94bbeb8936e1a74, Rust libm 0.2.16, "
        "-ffp-contract=off",
        "cases": [],
    }
    mismatches = 0
    for name, _, sidereon in cases:
        outputs = {}
        for key, bits in rtklib[name].items():
            entry = {"rtklib": "0x%016x" % bits}
            if key in sidereon:
                ours = sidereon[key]
                ulp = ulp_distance(from_bits(bits), ours)
                entry.update(sidereon=hex_bits(ours), ulp=ulp)
                mismatches += ulp != 0
            outputs[key] = entry
        report["cases"].append({"name": name, "outputs": outputs})
    report["mismatches"] = mismatches
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(cases)} cases, {mismatches} output(s) differ from sidereon's values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
