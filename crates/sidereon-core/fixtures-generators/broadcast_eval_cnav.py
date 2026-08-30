#!/usr/bin/env python3
"""Audit CNAV broadcast goldens with a portable high-precision reference.

The production fixture is canonicalized to the Rust ``libm`` crate.  This
script reconstructs the CNAV recipe with mpmath so its result can be compared
independently; mpmath is not assumed to produce the same bits as Rust ``libm``.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import struct
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from portable_math import MPMATH_PRECISION_DIGITS
from portable_math import add, atan2, cos, div, mul, sin, sqrt, sub


SECONDS_PER_WEEK = 604800.0
HALF_WEEK_S = 302400.0
SECONDS_PER_HOUR = 3600.0
KEPLER_TOL = 1.0e-12
KEPLER_MAX_ITER = 30
CLOCK_MAX_ITER = 2

GPS_GM_M3_S2 = 3.9860050e14
GPS_OMEGA_E_RAD_S = 7.2921151467e-5
GPS_DTR_F = -0.000000000444280763339306

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAV = ROOT / "tests/fixtures/nav/BRD400DLR_S_20261800000_01H_MN_trim.rnx"
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "sidereon-cnav-broadcast-mpmath-audit.json"


def bits(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def field(line: str, index: int) -> float | None:
    ranges = [(4, 23), (23, 42), (42, 61), (61, 80)]
    start, end = ranges[index]
    text = line[start:end].strip()
    if not text:
        return None
    return float(text.replace("D", "E").replace("d", "e"))


def required(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"missing required field {name}")
    return value


def gps_week_sow(year: int, month: int, day: int, hour: int, minute: int, second: int) -> tuple[int, float]:
    epoch = dt.datetime(1980, 1, 6)
    value = dt.datetime(year, month, day, hour, minute, second)
    delta = value - epoch
    week = delta.days // 7
    sow = (delta - dt.timedelta(days=week * 7)).total_seconds()
    return week, float(sow)


def time_from_reference_s(t_sow_s: float, reference_sow_s: float) -> float:
    value = sub(t_sow_s, reference_sow_s)
    if value > HALF_WEEK_S:
        value = sub(value, SECONDS_PER_WEEK)
    if value < -HALF_WEEK_S:
        value = add(value, SECONDS_PER_WEEK)
    return value


def eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> tuple[float, int]:
    value = mean_anomaly
    iterations = 0
    while iterations < KEPLER_MAX_ITER:
        previous = value
        value = add(mean_anomaly, mul(eccentricity, sin(previous)))
        iterations += 1
        if abs(sub(value, previous)) <= KEPLER_TOL:
            break
    return value, iterations


def clock_offset(clock: dict[str, float], elements: dict[str, float], sin_e: float, t_sow_s: float, tgd_s: float) -> dict[str, float]:
    af0 = clock["af0"]
    af1 = clock["af1"]
    af2 = clock["af2"]
    dt0 = time_from_reference_s(t_sow_s, clock["toc_sow"])
    arg = dt0
    for _ in range(CLOCK_MAX_ITER):
        arg = sub(dt0, add(add(af0, mul(af1, arg)), mul(af2, mul(arg, arg))))
    dt_poly = add(add(af0, mul(af1, arg)), mul(af2, mul(arg, arg)))
    dt_rel = mul(mul(mul(GPS_DTR_F, elements["e"]), elements["sqrt_a"]), sin_e)
    return {
        "dt_clock_poly_s": dt_poly,
        "dt_rel_s": dt_rel,
        "tgd_s": tgd_s,
        "dt_clock_total_s": sub(add(dt_poly, dt_rel), tgd_s),
    }


def orbit_state(elements: dict[str, float], rates: dict[str, float], t_sow_s: float) -> tuple[dict[str, float], int]:
    sqrt_a = elements["sqrt_a"]
    e = elements["e"]
    a0 = mul(sqrt_a, sqrt_a)
    n0 = sqrt(div(GPS_GM_M3_S2, mul(mul(a0, a0), a0)))
    tk = time_from_reference_s(t_sow_s, elements["toe_sow"])
    a = add(a0, mul(rates["adot_m_s"], tk))
    delta_n_a = add(elements["delta_n"], mul(mul(0.5, rates["delta_n0_dot_rad_s2"]), tk))
    n = add(n0, delta_n_a)
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
    omega_k = sub(
        add(elements["omega0"], mul(sub(elements["omega_dot"], GPS_OMEGA_E_RAD_S), tk)),
        mul(GPS_OMEGA_E_RAD_S, elements["toe_sow"]),
    )
    sin_o = sin(omega_k)
    cos_o = cos(omega_k)
    sin_i = sin(i)
    cos_i = cos(i)
    x = sub(mul(xp, cos_o), mul(mul(yp, cos_i), sin_o))
    y = add(mul(xp, sin_o), mul(mul(yp, cos_i), cos_o))
    z = mul(yp, sin_i)
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


def read_frames(path: Path) -> list[tuple[str, str, str, list[str]]]:
    lines = path.read_text().splitlines()
    frames: list[tuple[str, str, str, list[str]]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("> EPH "):
            index += 1
            continue
        parts = line.split()
        sv = parts[2]
        message = parts[3]
        body: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].startswith("> "):
            body.append(lines[index])
            index += 1
        frames.append((line, sv, message, body))
    return frames


def parse_cnav_frame(sv: str, message: str, body: list[str]) -> dict[str, object]:
    is_cnav2 = message == "CNV2"
    year = int(body[0][4:8])
    month = int(body[0][9:11])
    day = int(body[0][12:14])
    hour = int(body[0][15:17])
    minute = int(body[0][18:20])
    second = int(body[0][21:23])
    week, sow = gps_week_sow(year, month, day, hour, minute, second)
    orbit = [[field(line, i) for i in range(4)] for line in body[1:]]
    elements = {
        "crs": required(orbit[0][1], "crs"),
        "delta_n": required(orbit[0][2], "delta_n"),
        "m0": required(orbit[0][3], "m0"),
        "cuc": required(orbit[1][0], "cuc"),
        "e": required(orbit[1][1], "e"),
        "cus": required(orbit[1][2], "cus"),
        "sqrt_a": required(orbit[1][3], "sqrt_a"),
        "toe_sow": sow,
        "cic": required(orbit[2][1], "cic"),
        "omega0": required(orbit[2][2], "omega0"),
        "cis": required(orbit[2][3], "cis"),
        "i0": required(orbit[3][0], "i0"),
        "crc": required(orbit[3][1], "crc"),
        "omega": required(orbit[3][2], "omega"),
        "omega_dot": required(orbit[3][3], "omega_dot"),
        "idot": required(orbit[4][0], "idot"),
    }
    clock = {
        "af0": required(field(body[0], 1), "af0"),
        "af1": required(field(body[0], 2), "af1"),
        "af2": required(field(body[0], 3), "af2"),
        "toc_sow": sow,
    }
    rates = {
        "adot_m_s": required(orbit[0][0], "adot_m_s"),
        "delta_n0_dot_rad_s2": required(orbit[4][1], "delta_n0_dot_rad_s2"),
    }
    line_ttm = 8 if is_cnav2 else 7
    params = {
        "top_sow": required(orbit[2][0], "top_sow"),
        "ura_ed_index": int(required(orbit[5][0], "ura_ed_index")),
        "ura_ned0_index": int(required(orbit[4][2], "ura_ned0_index")),
        "ura_ned1_index": int(required(orbit[4][3], "ura_ned1_index")),
        "ura_ned2_index": int(required(orbit[5][3], "ura_ned2_index")),
        "transmission_time_sow": required(orbit[line_ttm][0], "transmission_time_sow"),
        "wn_op": int(required(orbit[line_ttm][1], "wn_op")),
    }
    tgd = orbit[5][2] or 0.0
    isc_l1ca = orbit[6][0] or 0.0
    return {
        "sat": sv,
        "system": "QZS" if sv.startswith("J") else "GPS",
        "message": "QZSS_CNAV2" if sv.startswith("J") and is_cnav2 else "QZSS_CNAV" if sv.startswith("J") else "GPS_CNAV2" if is_cnav2 else "GPS_CNAV",
        "week": week,
        "elements": elements,
        "clock": clock,
        "rates": rates,
        "params": params,
        "tgd_s": tgd - isc_l1ca,
    }


def hex_map(values: dict[str, float]) -> dict[str, str]:
    return {key: bits(value) for key, value in values.items()}


def make_case(record: dict[str, object], suffix: str, offset_s: float) -> dict[str, object]:
    elements = record["elements"]
    rates = record["rates"]
    clock = record["clock"]
    t_sow = add(elements["toe_sow"], offset_s)
    orbit, iterations = orbit_state(elements, rates, t_sow)
    clk = clock_offset(clock, elements, orbit["sin_e"], t_sow, record["tgd_s"])
    expect = dict(orbit)
    expect.update(clk)
    return {
        "name": f"{record['sat'].lower()}_{record['message'].lower()}_{suffix}",
        "system": record["system"],
        "sat": record["sat"],
        "message": record["message"],
        "t_sow_hex": bits(t_sow),
        "elements_hex": hex_map(elements),
        "rates_hex": hex_map(rates),
        "clock_hex": hex_map(clock),
        "tgd_s_hex": bits(record["tgd_s"]),
        "kepler_iterations": iterations,
        "expect_hex": hex_map(expect),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"audit JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    records = [
        parse_cnav_frame(sv, message, body)
        for _, sv, message, body in read_frames(SOURCE_NAV)
        if sv[0] in {"G", "J"} and message in {"CNAV", "CNV2"}
    ]
    offsets = [
        ("toe", 0.0),
        ("minus_45m", -2700.0),
        ("plus_45m", 2700.0),
        ("minus_fit_edge", -1.5 * SECONDS_PER_HOUR),
        ("plus_fit_edge", 1.5 * SECONDS_PER_HOUR),
    ]
    cases = [make_case(record, suffix, offset) for record in records for suffix, offset in offsets]
    doc = {
        "schema": "cnav_broadcast_ephemeris.v1",
        "source_nav": "BRD400DLR_S_20261800000_01H_MN_trim.rnx",
        "source_url": "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2026/180/BRD400DLR_S_20261800000_01D_MN.rnx.gz",
        "trim": "Header plus selected G01/G03 LNAV+CNAV, J02 LNAV+CNAV+CNV2, and C19 CNV2 frames.",
        "recipe": "IS-GPS-200/705/800 CNAV audit: mpmath high-precision reference rounded to binary64 after every operation; no FMA; explicit-multiply powers; fixed-point Kepler E=M+e*sin(E), tol 1e-12 cap 30; two clock time-argument refinements; relativistic term uses sqrt(A0).",
        "mpmath_audit": {
            "library": "mpmath",
            "version": str(__import__("mpmath").__version__),
            "precision_digits": MPMATH_PRECISION_DIGITS,
            "round_after_every_operation": True,
        },
        "engine_reference": {
            "library": "Rust libm crate",
            "version": "0.2.16",
            "portable": True,
        },
        "python_version": platform.python_version(),
        "kepler_tol_hex": bits(KEPLER_TOL),
        "kepler_max_iter": KEPLER_MAX_ITER,
        "clock_max_iter": CLOCK_MAX_ITER,
        "seconds_per_week_hex": bits(SECONDS_PER_WEEK),
        "half_week_s_hex": bits(HALF_WEEK_S),
        "constellations": {
            "GPS": {
                "gm_m3_s2_hex": bits(GPS_GM_M3_S2),
                "omega_e_rad_s_hex": bits(GPS_OMEGA_E_RAD_S),
                "dtr_f_hex": bits(GPS_DTR_F),
            },
            "QZS": {
                "gm_m3_s2_hex": bits(GPS_GM_M3_S2),
                "omega_e_rad_s_hex": bits(GPS_OMEGA_E_RAD_S),
                "dtr_f_hex": bits(GPS_DTR_F),
            },
        },
        "cases": cases,
    }
    args.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
