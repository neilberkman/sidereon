#!/usr/bin/env python3
"""Write the small ICGEM fixture fields in tests/fixtures/icgem.

usage: python3 make_fields.py

All static values come from the EGM96 table embedded in the crate
(src/astro/forces/egm96_to_36.txt; Lemoine et al., 1998, NASA/TP-1998-206861),
a work of NASA and NIMA in the public domain. The time-variable records of
synthetic_icgem2.gfc and synthetic_icgem1.gfc are synthetic values of the
size of GRACE-era variations, written to exercise every ICGEM record type;
they describe no published model.

- egm96_d36.gfc: EGM96 through degree and order 36 in ICGEM form, with the
  table's formal sigmas.
- synthetic_icgem2.gfc: format icgem2.0, degree 6, calibrated_and_formal
  errors, zero_tide; gfct, trnd, acos and asin records over validity
  intervals, including periodic records that span several gfct intervals and
  degree-1 records.
- synthetic_icgem1.gfc: format icgem1.0 (no format line), degree 5, formal
  errors, tide_free; gfct with a yyyymmdd epoch, trnd, dot, acos and asin.
- unnormalized_d8.gfc: EGM96 through degree 8 converted to unnormalized
  coefficients, errors no, no tide_system line.
"""
from decimal import Decimal, getcontext
from math import factorial
from pathlib import Path

getcontext().prec = 50
HERE = Path(__file__).resolve().parent
CRATE = HERE.parent.parent
OUT = CRATE / "tests" / "fixtures" / "icgem"
TABLE = CRATE / "src" / "astro" / "forces" / "egm96_to_36.txt"

EGM96 = {}
for row in TABLE.read_text().splitlines():
    if not row.strip() or row.lstrip().startswith("#"):
        continue
    n, m, c, s, sc, ss = row.split()
    EGM96[(int(n), int(m))] = (c, s, sc, ss)


def header(lines, **kv):
    for key, value in kv.items():
        lines.append(f"{key:<24}{value}")


def egm96_d36():
    lines = [
        "EGM96 (Lemoine et al., 1998, NASA/TP-1998-206861) through degree and order 36,",
        "fully normalized, with formal sigmas. NASA GSFC and NIMA; public domain.",
        "begin_of_head =======================================================",
    ]
    header(
        lines,
        product_type="gravity_field",
        modelname="EGM96_d36",
        earth_gravity_constant="0.3986004415E+15",
        radius="0.6378136300E+07",
        max_degree="36",
        errors="formal",
        norm="fully_normalized",
        tide_system="tide_free",
    )
    lines.append("key     L    M     C                   S                   sigma C          sigma S")
    lines.append("end_of_head =========================================================")
    lines.append("gfc     0    0  1.000000000000E+00  0.000000000000E+00  0.00000000E+00  0.00000000E+00")
    lines.append("gfc     1    0  0.000000000000E+00  0.000000000000E+00  0.00000000E+00  0.00000000E+00")
    lines.append("gfc     1    1  0.000000000000E+00  0.000000000000E+00  0.00000000E+00  0.00000000E+00")
    for (n, m), (c, s, sc, ss) in sorted(EGM96.items()):
        lines.append(f"gfc  {n:4d} {m:4d} {c:>19} {s:>19} {sc:>15} {ss:>15}")
    (OUT / "egm96_d36.gfc").write_text("\n".join(lines) + "\n")


def static_rows(max_degree, skip, sigma_columns):
    rows = []
    for n in range(2, max_degree + 1):
        for m in range(n + 1):
            if (n, m) in skip:
                continue
            c, s, sc, ss = EGM96[(n, m)]
            sig = " ".join([sc, ss] * (sigma_columns // 2))
            rows.append(f"gfc  {n:3d} {m:3d} {c:>19} {s:>19} {sig}".rstrip())
    return rows


def synthetic_icgem2():
    # Validity intervals, yyyymmdd.hhmm.
    c20_spans = [
        ("20000101.0000", "20050615.1230"),
        ("20050615.1230", "20100101.0000"),
        ("20100101.0000", "20300101.0000"),
    ]
    long_span = ("20000101.0000", "20300101.0000")
    sig4 = "1.0E-13 0.0E+00 2.0E-13 0.0E+00"
    sig4ns = "1.0E-13 1.1E-13 2.0E-13 2.1E-13"
    lines = [
        "Synthetic time-variable field for reader tests: static part EGM96 through",
        "degree 6 (public domain), time-variable records synthetic.",
        "begin_of_head =====================================",
    ]
    header(
        lines,
        format="icgem2.0",
        product_type="gravity_field",
        modelname="SYNTHETIC_ICGEM2",
        earth_gravity_constant="0.3986004415E+15",
        radius="0.6378136300E+07",
        max_degree="6",
        errors="calibrated_and_formal",
        norm="fully_normalized",
        tide_system="zero_tide",
    )
    lines.append("key L M C S sigmaC_cal sigmaS_cal sigmaC_formal sigmaS_formal t0 t1 period")
    lines.append("end_of_head ==========================================")
    lines.append("gfc    0    0 1.00000000000E+00 0.00000000000E+00 0.0E+00 0.0E+00 0.0E+00 0.0E+00")
    # Degree 1: geocentre motion, outside the perturbation sum.
    lines.append(f"gfct   1    0 7.0E-11 0.0E+00 {sig4} {long_span[0]} {long_span[1]}")
    lines.append(f"acos   1    0 7.8E-11 0.0E+00 {sig4} {long_span[0]} {long_span[1]} 1.0")
    lines.append(f"gfct   1    1 -1.2E-11 3.4E-11 {sig4ns} {long_span[0]} {long_span[1]}")
    c20 = ["-4.84169317366E-04", "-4.84169303100E-04", "-4.84169280500E-04"]
    trends = ["1.16E-11", "-2.2E-11", "4.4E-12"]
    for (t0, t1), value, trend in zip(c20_spans, c20, trends):
        lines.append(f"gfct   2    0 {value} 0.00000000000E+00 {sig4} {t0} {t1}")
        lines.append(f"trnd   2    0 {trend} 0.00000000000E+00 {sig4} {t0} {t1}")
    lines.append(f"acos   2    0 3.87E-11 0.0E+00 {sig4} {long_span[0]} {long_span[1]} 1.0")
    lines.append(f"asin   2    0 5.92E-11 0.0E+00 {sig4} {long_span[0]} {long_span[1]} 1.0")
    lines.append(f"acos   2    0 3.94E-11 0.0E+00 {sig4} {long_span[0]} {long_span[1]} 0.5")
    lines.append(f"asin   2    0 -2.19E-11 0.0E+00 {sig4} {long_span[0]} {long_span[1]} 0.5")
    # C21/S21 over one long interval, with a 161-day period.
    span21 = ("19900101.0000", "20300101.0000")
    c, s, _, _ = EGM96[(2, 1)]
    lines.append(f"gfct   2    1 {c} {s} {sig4ns} {span21[0]} {span21[1]}")
    lines.append(f"trnd   2    1 -3.3E-12 1.5E-12 {sig4ns} {span21[0]} {span21[1]}")
    lines.append(f"acos   2    1 2.5E-11 -1.8E-11 {sig4ns} {span21[0]} {span21[1]} 0.4408")
    # C22/S22 over two intervals; its periodic record covers only the second.
    c, s, _, _ = EGM96[(2, 2)]
    lines.append(f"gfct   2    2 {c} {s} {sig4ns} 19950101.0000 20080301.0600")
    lines.append(f"gfct   2    2 2.43938357328E-06 -1.40027370385E-06 {sig4ns} 20080301.0600 20300101.0000")
    lines.append(f"trnd   2    2 -1.0E-12 2.0E-12 {sig4ns} 19950101.0000 20080301.0600")
    lines.append(f"trnd   2    2 1.0E-12 -2.0E-12 {sig4ns} 20080301.0600 20300101.0000")
    lines.append(f"asin   2    2 1.5E-11 -2.5E-11 {sig4ns} 20080301.0600 20300101.0000 1.0")
    # C30 and C43/S43, trend written after a periodic record.
    c, s, _, _ = EGM96[(3, 0)]
    lines.append(f"gfct   3    0 {c} {s} {sig4} 20000101.0000 20300101.0000")
    lines.append(f"asin   3    0 9.5E-11 0.0E+00 {sig4} 20000101.0000 20300101.0000 1.0")
    lines.append(f"trnd   3    0 -8.4E-12 0.0E+00 {sig4} 20000101.0000 20300101.0000")
    c, s, _, _ = EGM96[(4, 3)]
    lines.append(f"gfct   4    3 {c} {s} {sig4ns} 20000101.0000 20300101.0000")
    lines.append(f"acos   4    3 7.3E-13 -6.8E-13 {sig4ns} 20000101.0000 20300101.0000 0.5")
    lines.append(f"asin   4    3 4.9E-13 -1.6E-13 {sig4ns} 20000101.0000 20300101.0000 0.5")
    skip = {(2, 0), (2, 1), (2, 2), (3, 0), (4, 3)}
    lines.extend(static_rows(6, skip, 4))
    (OUT / "synthetic_icgem2.gfc").write_text("\n".join(lines) + "\n")


def synthetic_icgem1():
    lines = [
        "Synthetic icgem1.0 field: static part EGM96 through degree 5 (public domain),",
        "time-variable records synthetic, reference epoch 2005-01-01.",
        "begin_of_head =====================================",
    ]
    header(
        lines,
        product_type="gravity_field",
        modelname="SYNTHETIC_ICGEM1",
        earth_gravity_constant="0.3986004415E+15",
        radius="0.6378136300E+07",
        max_degree="5",
        errors="formal",
        tide_system="tide_free",
    )
    lines.append("key L M C S sigma_C sigma_S t0[yyyymmdd] period[y]")
    lines.append("end_of_head ==========================================")
    lines.append("gfc    0    0 1.00000000000e+00 0.000000000000e+00 0.0000e+00 0.0000e+00")
    lines.append("gfc    1    0 0.00000000000e+00 0.000000000000e+00 0.0000e+00 0.0000e+00")
    lines.append("gfc    1    1 0.00000000000e+00 0.000000000000e+00 0.0000e+00 0.0000e+00")
    lines.append("gfct   2    0 -4.84165299806e-04 0.000000000000e+00 1.9482e-13 0.0000e+00 20050101")
    lines.append("trnd   2    0 -1.26060242677e-11 0.000000000000e+00 3.2284e-14 0.0000e+00")
    lines.append("acos   2    0 4.10012162817e-11 0.000000000000e+00 1.8915e-13 0.0000e+00 1.0")
    lines.append("asin   2    0 5.32328946063e-11 0.000000000000e+00 1.9618e-13 0.0000e+00 1.0")
    lines.append("acos   2    0 3.33917546745e-11 0.000000000000e+00 1.8829e-13 0.0000e+00 0.5")
    lines.append("asin   2    0 -2.44339926664e-11 0.000000000000e+00 1.9158e-13 0.0000e+00 0.5")
    c, s, _, _ = EGM96[(3, 1)]
    lines.append(f"gfct   3    1 {c} {s} 1.6575e-13 1.6575e-13 20050101")
    lines.append("dot    3    1 -8.37409795344e-12 3.1e-12 5.0047e-14 5.0047e-14")
    lines.append("asin   3    1 9.47586646659e-11 -2.2e-11 2.2725e-13 2.2725e-13 1.0")
    c, s, _, _ = EGM96[(4, 4)]
    lines.append(f"gfct   4    4 {c} {s} 6.2820e-14 6.2820e-14 20050101")
    lines.append("acos   4    4 -7.55236874480e-12 8.1e-12 7.4141e-14 7.4141e-14 1.0")
    skip = {(2, 0), (3, 1), (4, 4)}
    lines.extend(static_rows(5, skip, 2))
    (OUT / "synthetic_icgem1.gfc").write_text("\n".join(lines) + "\n")


def fmt(value):
    return "0.0e+00" if value == 0 else f"{value:.15e}"


def unnormalized_d8():
    lines = [
        "EGM96 through degree 8 (public domain) converted to unnormalized coefficients",
        "C = Cbar N_nm, N_nm = sqrt((2 - delta_m0) (2n + 1) (n - m)! / (n + m)!), 16 digits.",
        "begin_of_head =====================================",
    ]
    header(
        lines,
        product_type="gravity_field",
        modelname="EGM96_d8_unnormalized",
        earth_gravity_constant="0.3986004415E+15",
        radius="0.6378136300E+07",
        max_degree="8",
        errors="no",
        norm="unnormalized",
    )
    lines.append("end_of_head ==========================================")
    lines.append("gfc    0    0 1.0e+00 0.0e+00")
    lines.append("gfc    1    0 0.0e+00 0.0e+00")
    lines.append("gfc    1    1 0.0e+00 0.0e+00")
    for n in range(2, 9):
        for m in range(n + 1):
            c, s, _, _ = EGM96[(n, m)]
            delta = 1 if m == 0 else 2
            norm = (Decimal(delta * (2 * n + 1) * factorial(n - m)) / Decimal(factorial(n + m))).sqrt()
            uc = Decimal(c.replace("D", "E")) * norm
            us = Decimal(s.replace("D", "E")) * norm
            lines.append(f"gfc  {n:3d} {m:3d} {fmt(uc)} {fmt(us)}")
    (OUT / "unnormalized_d8.gfc").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    egm96_d36()
    synthetic_icgem2()
    synthetic_icgem1()
    unnormalized_d8()
