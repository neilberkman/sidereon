#!/usr/bin/env python3
"""Rebuild the expected values of `tests/fixtures/glonass_golden.json`.

Keeps every case's inputs and recomputes, with binary64 rounding after every
operation and no fused multiply-add, the recipe the Rust port follows:

- RTKLIB `deq`: `r2 = x^2 + y^2 + z^2` (a zero derivative when `r2 <= 0`),
  `r3 = r2*sqrt(r2)`, `a = 1.5*J2*MU*(Re*Re)/r2/r3`, `b = 5*z*z/r2`,
  `c = -MU/r3 - a*(1 - b)`, and the accelerations with the Coriolis terms;
- RTKLIB `glorbit`: one RK4 step; `geph2pos` steps of 60 s toward `tk` with a final
  partial step, until `|t| <= 1e-9`;
- the clock of RTKLIB `geph2clk`: `t = ts = tk`, twice `t = ts - (clk + gamma*t)`, then
  `clk + gamma*t`, with `clk` the stated `-TauN`; and of `geph2pos`,
  `clk + gamma*tk` without iteration.

The script also adds the cases with a non-zero `GammaN` when they are missing.

    python3 glonass_golden.py           # report values that differ
    python3 glonass_golden.py --write   # rewrite the expected values
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/glonass_golden.json"
RECIPE = (
    "deq (RTKLIB): r2=x^2+y^2+z^2 (zero derivative if r2<=0); r3=r2*sqrt(r2); "
    "a=3/2*J2*MU*Re^2/r2/r3; b=5*z^2/r2; c=-MU/r3 - a*(1-b); accel = (c+w^2)*x + 2*w*vy "
    "+ als_x (symmetric y; z uses (c-2a) and no Coriolis). RK4 fixed step 60 s with a "
    "final partial step, direction by sign(tk). Clock (RTKLIB geph2clk): t=ts=tk, twice "
    "t=ts-(-TauN-field + GammaN*t), then -TauN-field + GammaN*t; position clock (RTKLIB "
    "geph2pos): -TauN-field + GammaN*tk. No FMA; integer powers as explicit multiplies; "
    "math.sqrt."
)
# Cases with a non-zero GammaN, built from the first case's state: (name, tk, TauN field,
# GammaN, note).
GAMMA_CASES = [
    (
        "gamma_forward_age_limit",
        1800.0,
        -1.0e-3,
        9.094947017729282e-13,
        "tk at RTKLIB MAXDTOE_GLO with a positive GammaN and a large -TauN field",
    ),
    (
        "gamma_backward_partial",
        -937.5,
        5.8e-5,
        -2.7e-12,
        "backward propagation ending on a partial step with a negative GammaN",
    ),
    (
        "gamma_velocity_step",
        1800.001,
        6.761029362679e-05,
        1.818989403546e-12,
        "tk one RTKLIB ephpos step past the age limit, 31 RK4 steps",
    ),
]


def run(constants, inputs):
    mu, j2, we, re, tstep = (constants[k] for k in ("mu", "j2", "omega_e", "r_e", "tstep_s"))

    def deq(s, acc):
        x, y, z, vx, vy, vz = s
        r2 = x * x + y * y + z * z
        if r2 <= 0.0:
            return [0.0] * 6
        r3 = r2 * sqrt(r2)
        omg2 = we * we
        a = 1.5 * j2 * mu * (re * re) / r2 / r3
        b = 5.0 * z * z / r2
        c = -mu / r3 - a * (1.0 - b)
        return [
            vx,
            vy,
            vz,
            (c + omg2) * x + 2.0 * we * vy + acc[0],
            (c + omg2) * y - 2.0 * we * vx + acc[1],
            (c - 2.0 * a) * z + acc[2],
        ]

    def glorbit(t, s, acc):
        k1 = deq(s, acc)
        w = [s[i] + k1[i] * t / 2.0 for i in range(6)]
        k2 = deq(w, acc)
        w = [s[i] + k2[i] * t / 2.0 for i in range(6)]
        k3 = deq(w, acc)
        w = [s[i] + k3[i] * t for i in range(6)]
        k4 = deq(w, acc)
        return [s[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * t / 6.0 for i in range(6)]

    state = [float.fromhex(v) for v in inputs["pos_m"]] + [
        float.fromhex(v) for v in inputs["vel_m_s"]
    ]
    acc = [float.fromhex(v) for v in inputs["acc_m_s2"]]
    tk = float.fromhex(inputs["tk_s"])
    steps = []
    t = tk
    step = -tstep if t < 0.0 else tstep
    while abs(t) > 1e-9:
        if abs(t) < tstep:
            step = t
        state = glorbit(step, state, acc)
        steps.append({"step_s": step.hex(), "state": [v.hex() for v in state]})
        t -= step
    clk = float.fromhex(inputs["clk_bias"])
    gamma = float.fromhex(inputs["gamma_n"])
    ts = tk
    tc = ts
    for _ in range(2):
        tc = ts - (clk + gamma * tc)
    return {
        "steps": steps,
        "final_state": [v.hex() for v in state],
        "clock_offset_s": (clk + gamma * tc).hex(),
        "position_clock_offset_s": (clk + gamma * tk).hex(),
    }


def main() -> int:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the fixture")
    args = parser.parse_args()
    data = json.loads(FIXTURE.read_text())
    constants = {k: float.fromhex(v) for k, v in data["constants"].items()}
    names = {case["name"] for case in data["cases"]}
    template = data["cases"][0]
    for name, tk, clk, gamma, note in GAMMA_CASES:
        if name in names:
            continue
        inputs = dict(template["inputs"])
        inputs.update(
            tk_s=tk.hex(), tk_s_repr=repr(tk), clk_bias=clk.hex(), gamma_n=gamma.hex()
        )
        data["cases"].append(
            {"name": name, "note": note, "sat": template["sat"], "inputs": inputs, "expect": {}}
        )
    changed = 0
    for case in data["cases"]:
        expect = run(constants, case["inputs"])
        for key, value in expect.items():
            if case["expect"].get(key) != value:
                changed += 1
                print(f"{case['name']}.{key} differs")
        case["expect"] = expect
    data["recipe"] = RECIPE
    if args.write:
        FIXTURE.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{changed} value(s) differ from the committed fixture")
    return 0 if args.write or changed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
