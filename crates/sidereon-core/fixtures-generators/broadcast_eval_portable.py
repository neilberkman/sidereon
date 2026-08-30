#!/usr/bin/env python3
"""Audit broadcast goldens with an independent high-precision reference.

The committed fixture already contains the exact binary64 inputs selected from
the vendored navigation products.  This generator keeps those inputs and
recomputes only the expected intermediates and outputs with
``portable_math.py``.  That makes the missing historical recipe
reconstructible without depending on a system libm or a Python ``math`` call.

The committed fixture is canonicalized to the Rust ``libm`` crate because that
is the production evaluator.  mpmath is intentionally retained here as an
independent audit: it is portable and high precision, but it does not match
Rust ``libm`` on every transcendental input in the fixture.  Run this script
when auditing those differences; do not use its output as the engine fixture
without comparing it to the Rust evaluator first.
"""

from __future__ import annotations

import json
import platform
import struct
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from portable_math import MPMATH_PRECISION_DIGITS
from portable_math import add, atan2, cos, div, mul, sin, sqrt, sub


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "tests/fixtures/broadcast_golden.json"
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "sidereon-broadcast-mpmath-audit.json"
PI = 3.141592653589793
TAU = 6.283185307179586
SECONDS_PER_WEEK = 604800.0
HALF_WEEK_S = 302400.0
KEPLER_TOL = 1.0e-12
KEPLER_MAX_ITER = 30
CLOCK_MAX_ITER = 2


def bits(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def value(s: str) -> float:
    return struct.unpack(">d", struct.pack(">Q", int(s.removeprefix("0x"), 16)))[0]


def folded_time(t_sow: float, reference_sow: float) -> float:
    dt = sub(t_sow, reference_sow)
    if dt > HALF_WEEK_S:
        dt = sub(dt, SECONDS_PER_WEEK)
    if dt < -HALF_WEEK_S:
        dt = add(dt, SECONDS_PER_WEEK)
    return dt


def eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> tuple[float, int]:
    current = mean_anomaly
    iterations = 0
    while iterations < KEPLER_MAX_ITER:
        previous = current
        current = add(mean_anomaly, mul(eccentricity, sin(previous)))
        iterations += 1
        if abs(current - previous) <= KEPLER_TOL:
            break
    return current, iterations


def orbit_state(
    elements: dict[str, float],
    consts: dict[str, float],
    t_sow: float,
    is_geo: bool,
) -> tuple[dict[str, float], int]:
    sqrt_a = elements["sqrt_a"]
    e = elements["e"]
    omega_e = consts["omega_e"]
    a = mul(sqrt_a, sqrt_a)
    a2 = mul(a, a)
    a3 = mul(a2, a)
    n0 = sqrt(div(consts["gm"], a3))
    n = add(n0, elements["delta_n"])
    tk = folded_time(t_sow, elements["toe_sow"])
    mk = add(elements["m0"], mul(n, tk))
    ecc, iterations = eccentric_anomaly(mk, e)
    sin_e = sin(ecc)
    cos_e = cos(ecc)
    e2 = mul(e, e)
    nu = atan2(mul(sqrt(sub(1.0, e2)), sin_e), sub(cos_e, e))
    phi = add(nu, elements["omega"])
    two_phi = mul(2.0, phi)
    s2 = sin(two_phi)
    c2 = cos(two_phi)
    du = add(mul(elements["cus"], s2), mul(elements["cuc"], c2))
    dr = add(mul(elements["crs"], s2), mul(elements["crc"], c2))
    di = add(mul(elements["cis"], s2), mul(elements["cic"], c2))
    u = add(phi, du)
    r = add(mul(a, sub(1.0, mul(e, cos_e))), dr)
    i = add(add(elements["i0"], di), mul(elements["idot"], tk))
    xp = mul(r, cos(u))
    yp = mul(r, sin(u))
    if is_geo:
        omega_k = sub(
            add(elements["omega0"], mul(elements["omega_dot"], tk)),
            mul(omega_e, elements["toe_sow"]),
        )
    else:
        omega_k = sub(
            add(elements["omega0"], mul(sub(elements["omega_dot"], omega_e), tk)),
            mul(omega_e, elements["toe_sow"]),
        )
    sin_o = sin(omega_k)
    cos_o = cos(omega_k)
    sin_i = sin(i)
    cos_i = cos(i)
    xg = sub(mul(xp, cos_o), mul(mul(yp, cos_i), sin_o))
    yg = add(mul(xp, sin_o), mul(mul(yp, cos_i), cos_o))
    zg = mul(yp, sin_i)
    if is_geo:
        deg5 = div(mul(5.0, PI), 180.0)
        cos_phi = cos(deg5)
        sin_phi = -sin(deg5)
        z_ang = mul(omega_e, tk)
        cos_z = cos(z_ang)
        sin_z = sin(z_ang)
        yr = add(mul(yg, cos_phi), mul(zg, sin_phi))
        zr = add(mul(-yg, sin_phi), mul(zg, cos_phi))
        x = add(mul(xg, cos_z), mul(yr, sin_z))
        y = add(mul(-xg, sin_z), mul(yr, cos_z))
        z = zr
    else:
        x, y, z = xg, yg, zg
    return (
        {
            "a": a,
            "n0": n0,
            "n": n,
            "tk": tk,
            "mk": mk,
            "eccentric_anomaly": ecc,
            "sin_e": sin_e,
            "cos_e": cos_e,
            "nu": nu,
            "phi": phi,
            "s2": s2,
            "c2": c2,
            "du": du,
            "dr": dr,
            "di": di,
            "u": u,
            "r": r,
            "i": i,
            "xp": xp,
            "yp": yp,
            "omega_k": omega_k,
            "x_m": x,
            "y_m": y,
            "z_m": z,
        },
        iterations,
    )


def clock_offset(
    clock: dict[str, float],
    elements: dict[str, float],
    consts: dict[str, float],
    sin_e: float,
    t_sow: float,
    tgd: float,
) -> dict[str, float]:
    dt0 = folded_time(t_sow, clock["toc_sow"])
    dt = dt0
    for _ in range(CLOCK_MAX_ITER):
        dt = sub(
            dt0,
            add(
                add(clock["af0"], mul(clock["af1"], dt)),
                mul(clock["af2"], mul(dt, dt)),
            ),
        )
    dt_poly = add(
        add(clock["af0"], mul(clock["af1"], dt)),
        mul(clock["af2"], mul(dt, dt)),
    )
    dt_rel = mul(mul(mul(consts["dtr"], elements["e"]), elements["sqrt_a"]), sin_e)
    return {
        "dt_clock_poly_s": dt_poly,
        "dt_rel_s": dt_rel,
        "tgd_s": tgd,
        "dt_clock_total_s": sub(add(dt_poly, dt_rel), tgd),
    }


def hex_map(values: dict[str, float]) -> dict[str, str]:
    return {key: bits(val) for key, val in values.items()}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"audit JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    doc = json.loads(INPUT.read_text())
    for case in doc["cases"]:
        elements = {key: value(val) for key, val in case["elements_hex"].items()}
        clock = {key: value(val) for key, val in case["clock_hex"].items()}
        t_sow = value(case["t_sow_hex"])
        tgd = value(case["tgd_s_hex"])
        system = case["system"]
        consts = {
            "gm": value(doc["constellations"][system]["gm_m3_s2_hex"]),
            "omega_e": value(doc["constellations"][system]["omega_e_rad_s_hex"]),
            "dtr": value(doc["constellations"][system]["dtr_f_hex"]),
        }
        orbit, iterations = orbit_state(elements, consts, t_sow, case.get("is_geo", False))
        case["kepler_iterations"] = iterations
        expect = dict(orbit)
        expect.update(clock_offset(clock, elements, consts, orbit["sin_e"], t_sow, tgd))
        case["expect_hex"] = hex_map(expect)

    doc["recipe"] = (
        "broadcast_eval_portable.py audit: IS-GPS-200 / Galileo OS ICD "
        "Keplerian orbit + clock; mpmath high-precision audit rounded to "
        "binary64 after every operation; no FMA; explicit-multiply powers; "
        "Kepler fixed-point E=M+e*sin(E), tol 1e-12 cap 30; clock RTKLIB "
        "time-arg refinement x2 + relativistic + group delay"
    )
    doc["python_version"] = platform.python_version()
    doc["mpmath_audit"] = {
        "library": "mpmath",
        "version": str(__import__("mpmath").__version__),
        "precision_digits": MPMATH_PRECISION_DIGITS,
        "round_after_every_operation": True,
    }
    doc["engine_reference"] = {
        "library": "Rust libm crate",
        "version": "0.2.16",
        "portable": True,
    }
    doc.pop("reference", None)
    doc.pop("numpy_version", None)
    args.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
