#!/usr/bin/env python3
"""Rebuild the expected values of the Keplerian broadcast goldens.

Reads `tests/fixtures/broadcast_golden.json` (legacy messages) and
`tests/fixtures/cnav_broadcast_golden.json` (GPS/QZSS CNAV and CNAV-2), keeps every
case's inputs, and recomputes each case's intermediates and outputs with the recipe
the Rust evaluator follows: RTKLIB `eph2pos` statement order, binary64 rounding after
every operation, no fused multiply-add, and the Rust `libm` crate's `sin`, `cos` and
`atan2` through `rust_libm_port.py`.

    python3 broadcast_goldens_rust_libm.py           # report cases that differ
    python3 broadcast_goldens_rust_libm.py --write   # rewrite the expected values

The recipe:

- `A = sqrtA^2`, `n0 = sqrt(mu / (A*A*A))`, `n = n0 + dn` (CNAV: `A = A0 + Adot*tk`,
  `n = n0 + dn0 + 0.5*dn0dot*tk` with `n0` from `A0`), `M = M0 + n*tk`;
- Kepler by Newton, `E -= (E - e*sin(E) - M) / (1 - e*cos(E))`, seeded `E = M` with a
  previous value of 0, while `|E - E_prev| > 1e-13` and fewer than 30 steps;
- `u = atan2(sqrt(1 - e*e)*sinE, cosE - e) + omega`, the harmonic corrections from
  `sin(2u)`, `cos(2u)`, `i = (i0 + idot*tk) + di`;
- the node `OMEGA0 + (OMEGAdot - we)*tk - we*toe` (BeiDou GEO: `OMEGA0 + OMEGAdot*tk -
  we*toe`, then RTKLIB's rotation with its `SIN_5`/`COS_5` literals);
- the clock polynomial at `t - toc` without refinement, the relativistic term
  `-(2*sqrt(mu*A)*e*sinE / (c*c))`, less the group delay.
"""

from __future__ import annotations

import json
import struct
import sys
from argparse import ArgumentParser
from math import sqrt
from pathlib import Path

import rust_libm_port as rlibm

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = [
    (ROOT / "tests/fixtures/broadcast_golden.json", False),
    (ROOT / "tests/fixtures/cnav_broadcast_golden.json", True),
]
SECONDS_PER_WEEK = 604800.0
HALF_WEEK_S = 302400.0
SPEED_OF_LIGHT = 299792458.0
KEPLER_TOL = 1.0e-13
KEPLER_MAX_ITER = 30
# RTKLIB `ephemeris.c`: sin(-5 deg) and cos(-5 deg) as it writes them.
SIN_5 = -0.0871557427476582
COS_5 = 0.9961946980917456
RECIPE = (
    "RTKLIB eph2pos statement order: Kepler Newton E-=(E-e*sin(E)-M)/(1-e*cos(E)) "
    "seeded E=M, Ek=0, while |E-Ek|>1e-13 and n<30; i=(i0+idot*tk)+di; BeiDou GEO "
    "rotation with SIN_5/COS_5 literals; clock polynomial at t-toc without refinement, "
    "relativistic -2*sqrt(mu*A)*e*sinE/(c*c), minus group delay"
)


def from_hex(text: str) -> float:
    return struct.unpack(">d", struct.pack(">Q", int(text, 16)))[0]


def to_hex(value: float) -> str:
    return "0x%016x" % struct.unpack(">Q", struct.pack(">d", value))[0]


def fold(t: float, reference: float) -> float:
    dt = t - reference
    if dt > HALF_WEEK_S:
        dt -= SECONDS_PER_WEEK
    if dt < -HALF_WEEK_S:
        dt += SECONDS_PER_WEEK
    return dt


def kepler(m: float, e: float) -> tuple[float, int]:
    ecc = m
    previous = 0.0
    n = 0
    while abs(ecc - previous) > KEPLER_TOL and n < KEPLER_MAX_ITER:
        previous = ecc
        ecc -= (ecc - e * rlibm.sin(ecc) - m) / (1.0 - e * rlibm.cos(ecc))
        n += 1
    return ecc, n


def orbit(el, gm, we, t, is_geo, rates):
    e = el["e"]
    tk = fold(t, el["toe_sow"])
    if rates is not None:
        a0 = el["sqrt_a"] * el["sqrt_a"]
        n0 = sqrt(gm / (a0 * a0 * a0))
        a = a0 + rates["adot_m_s"] * tk
        dna = el["delta_n"] + 0.5 * rates["delta_n0_dot_rad_s2"] * tk
        n = n0 + dna
    else:
        a = el["sqrt_a"] * el["sqrt_a"]
        n0 = sqrt(gm / (a * a * a))
        n = n0 + el["delta_n"]
    mk = el["m0"] + n * tk
    ecc, iterations = kepler(mk, e)
    sin_e = rlibm.sin(ecc)
    cos_e = rlibm.cos(ecc)
    nu = rlibm.atan2(sqrt(1.0 - e * e) * sin_e, cos_e - e)
    phi = nu + el["omega"]
    s2 = rlibm.sin(2.0 * phi)
    c2 = rlibm.cos(2.0 * phi)
    du = el["cus"] * s2 + el["cuc"] * c2
    dr = el["crs"] * s2 + el["crc"] * c2
    di = el["cis"] * s2 + el["cic"] * c2
    u = phi + du
    r = a * (1.0 - e * cos_e) + dr
    i = el["i0"] + el["idot"] * tk + di
    xp = r * rlibm.cos(u)
    yp = r * rlibm.sin(u)
    if is_geo:
        omega_k = el["omega0"] + el["omega_dot"] * tk - we * el["toe_sow"]
    else:
        omega_k = el["omega0"] + (el["omega_dot"] - we) * tk - we * el["toe_sow"]
    sin_o = rlibm.sin(omega_k)
    cos_o = rlibm.cos(omega_k)
    cos_i = rlibm.cos(i)
    xg = xp * cos_o - yp * cos_i * sin_o
    yg = xp * sin_o + yp * cos_i * cos_o
    zg = yp * rlibm.sin(i)
    if is_geo:
        sino = rlibm.sin(we * tk)
        coso = rlibm.cos(we * tk)
        x = xg * coso + yg * sino * COS_5 + zg * sino * SIN_5
        y = -xg * sino + yg * coso * COS_5 + zg * coso * SIN_5
        z = -yg * SIN_5 + zg * COS_5
    else:
        x, y, z = xg, yg, zg
    values = dict(
        a=a, n0=n0, n=n, tk=tk, mk=mk, eccentric_anomaly=ecc, sin_e=sin_e, cos_e=cos_e,
        nu=nu, phi=phi, s2=s2, c2=c2, du=du, dr=dr, di=di, u=u, r=r, i=i, xp=xp, yp=yp,
        omega_k=omega_k, x_m=x, y_m=y, z_m=z,
    )
    return values, iterations


def clock(ck, el, gm, sin_e, t, tgd):
    dt = fold(t, ck["toc_sow"])
    poly = ck["af0"] + ck["af1"] * dt + ck["af2"] * dt * dt
    a = el["sqrt_a"] * el["sqrt_a"]
    rel = -(2.0 * sqrt(gm * a) * el["e"] * sin_e / (SPEED_OF_LIGHT * SPEED_OF_LIGHT))
    return dict(
        dt_clock_poly_s=poly, dt_rel_s=rel, tgd_s=tgd, dt_clock_total_s=poly + rel - tgd
    )


def rebuild(path: Path, cnav: bool, write: bool) -> int:
    data = json.loads(path.read_text())
    constants = data["constellations"]
    changed = 0
    for case in data["cases"]:
        system = constants[case["system"]]
        gm = from_hex(system["gm_m3_s2_hex"])
        we = from_hex(system["omega_e_rad_s_hex"])
        el = {k: from_hex(v) for k, v in case["elements_hex"].items()}
        ck = {k: from_hex(v) for k, v in case["clock_hex"].items()}
        rates = {k: from_hex(v) for k, v in case["rates_hex"].items()} if cnav else None
        t = from_hex(case["t_sow_hex"])
        tgd = from_hex(case["tgd_s_hex"])
        values, iterations = orbit(el, gm, we, t, case.get("is_geo", False), rates)
        values.update(clock(ck, el, gm, values["sin_e"], t, tgd))
        for key in case["expect_hex"]:
            new = to_hex(values[key])
            if new != case["expect_hex"][key]:
                changed += 1
                print(f"{path.name}: {case['name']}.{key} {case['expect_hex'][key]} -> {new}")
            case["expect_hex"][key] = new
        if case.get("kepler_iterations") != iterations:
            changed += 1
        case["kepler_iterations"] = iterations
    data["kepler_tol_hex"] = to_hex(KEPLER_TOL)
    if RECIPE not in data["recipe"]:
        data["recipe"] = RECIPE
    if write:
        path.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def main() -> int:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the fixtures")
    args = parser.parse_args()
    changed = sum(rebuild(path, cnav, args.write) for path, cnav in FIXTURES)
    print(f"{changed} value(s) differ from the committed fixtures")
    return 0 if args.write or changed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
