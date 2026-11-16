#!/usr/bin/env python3
"""Add independent ADD v3.0 roots to the MAAST matrix fixture.

The calculation forms the weighted least-squares gains and ADD Eqs. (2),
(15)-(18), (26)-(27), and (31)-(33) directly.  SciPy Brent roots are used;
no Sidereon code or protection-level search is invoked.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import mpmath as mp
import scipy
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

PATH = Path(__file__).parents[1] / "tests/fixtures/araim_maast_matrix.json"
PTHRES = 8e-8
PHMI_V, PHMI_H = 9.8e-8, 2e-9
PFA_V, PFA_H = 3.9e-6, 9e-8


def design(case: dict) -> np.ndarray:
    if "design_matrix" in case:
        return np.asarray(case["design_matrix"], dtype=float)
    rows = []
    for i, elevation in enumerate(case["elevation_rad"]):
        azimuth = math.radians(137.50776 * i)
        clocks = [float(i % case["n_const"] == system) for system in range(case["n_const"])]
        rows.append([
            -math.cos(elevation) * math.sin(azimuth),
            -math.cos(elevation) * math.cos(azimuth),
            -math.sin(elevation),
            *clocks,
        ])
    return np.asarray(rows)


def enumerate_modes(case: dict) -> tuple[list[dict], float]:
    mp.mp.dps = 60
    n, nc = case["n_sat"], case["n_const"]
    events = [("sat", i, case["p_sat"]) for i in range(n)]
    events += [("const", i, case["p_const"]) for i in range(nc)]
    mp_probabilities = [mp.mpf(str(event[2])) for event in events]
    mp_p0 = mp.fprod(1 - probability for probability in mp_probabilities)
    modes = [{"events": (), "excluded": set(), "consts": set(), "prior": float(mp_p0)}]
    candidates = []
    for order in range(1, case["max_fault_order"] + 1):
        for selected in itertools.combinations(range(len(events)), order):
            if any(events[index][2] == 0.0 for index in selected):
                continue
            prior = mp.fprod(
                probability if index in selected else 1 - probability
                for index, probability in enumerate(mp_probabilities)
            )
            excluded, constellations = set(), set()
            for index in selected:
                kind, value, _ = events[index]
                if kind == "sat":
                    excluded.add(value)
                else:
                    constellations.add(value)
            candidates.append({"order": order, "events": selected, "excluded": excluded,
                               "consts": constellations, "prior": float(prior),
                               "exact_prior": prior})
    candidates.sort(key=lambda mode: (mode["order"], -mode["prior"]))
    # Eq. (2) is evaluated directly for every subset in 60-digit arithmetic;
    # no fault-order recurrence or engine probability is used.
    remaining = 1 - mp_p0
    for candidate in candidates:
        if remaining <= case["p_threshold_unmonitored"]:
            break
        remaining = max(mp.mpf("0"), remaining - candidate["exact_prior"])
        modes.append(candidate)

    # ADD Eq. (16) compares the sum, then Eq. (17) consolidates the whole set.
    removed = set()
    for representative, mode in enumerate(modes):
        if len(mode["consts"]) != 1 or mode["excluded"]:
            continue
        system = next(iter(mode["consts"]))
        members = [i for i, other in enumerate(modes[1:], 1)
                   if i != representative and other["consts"].issubset({system}) and other["excluded"]
                   and all(satellite % nc == system for satellite in other["excluded"])]
        mass = mp.fsum(mp.mpf(str(modes[i]["prior"])) for i in members)
        if mass <= mp.mpf("0.01") * mp.mpf(str(mode["prior"])):
            mode["prior"] = float(mp.mpf(str(mode["prior"])) + mass)
            removed.update(members)
    selected_modes = [mode for i, mode in enumerate(modes) if i not in removed]
    for mode in selected_modes:
        mode.pop("exact_prior", None)
    return selected_modes, float(remaining)


def solve_case(case: dict) -> dict:
    h = design(case)
    n, nc = case["n_sat"], case["n_const"]
    sigma_i = np.sqrt(case["cint_m2"])
    sigma_a = np.sqrt(case["cacc_m2"])
    bias = np.full(n, case["bias_m"])
    system_index = case.get("system_index", [index % nc for index in range(n)])
    weights = 1.0 / np.square(sigma_i)
    modes, remaining = enumerate_modes(case)
    maximum_normal_condition = 1.0

    def gain(mode: dict) -> np.ndarray | None:
        nonlocal maximum_normal_condition
        kept = np.ones(n, dtype=bool)
        for satellite in mode["excluded"]:
            kept[satellite] = False
        for constellation in mode["consts"]:
            kept &= np.asarray([system_index[i] != constellation for i in range(n)])
        active = [system for system in range(nc) if any(kept[i] and system_index[i] == system for i in range(n))]
        columns = [0, 1, 2, *(3 + system for system in active)]
        if kept.sum() < len(columns) or not active:
            return None
        hk = h[:, columns]
        wk = weights * kept
        normal = hk.T @ (wk[:, None] * hk)
        if np.linalg.matrix_rank(normal) < len(columns):
            return None
        maximum_normal_condition = max(maximum_normal_condition, float(np.linalg.cond(normal)))
        state_gain = np.linalg.inv(normal) @ hk.T @ np.diag(wk)
        return state_gain[:3]

    gains = [gain(mode) for mode in modes]
    observable = sum(value is not None for value in gains[1:])
    kh = norm.isf(PFA_H / (4.0 * observable))
    kv = norm.isf(PFA_V / (2.0 * observable))
    g0 = gains[0]
    terms = []
    unmonitored = remaining
    for index, (mode, matrix) in enumerate(zip(modes, gains)):
        if matrix is None:
            unmonitored += mode["prior"]
            continue
        sigma = np.sqrt(np.sum(np.square(matrix * sigma_i), axis=1))
        nominal_bias = np.sum(np.abs(matrix) * bias, axis=1)
        separation = np.sqrt(np.sum(np.square((matrix - g0) * sigma_a), axis=1))
        threshold = separation * np.asarray([kh, kh, kv]) if index else np.zeros(3)
        terms.append((mode["prior"], sigma, nominal_bias, threshold, mode))

    scale = 1.0 - unmonitored / (PHMI_V + PHMI_H)

    def root(axis: int, target: float) -> float:
        def equation(value: float) -> float:
            _, sigma0, bias0, _, _ = terms[0]
            total = 2.0 * ndtr(-(value - bias0[axis]) / sigma0[axis])
            for prior, sigma, nominal_bias, threshold, _ in terms[1:]:
                argument = (value - threshold[axis] - nominal_bias[axis]) / sigma[axis]
                total += prior * (1.0 if argument <= 0.0 else ndtr(-argument))
            return total - target
        upper = 1.0
        while equation(upper) > 0.0:
            upper *= 2.0
        return brentq(equation, 0.0, upper, xtol=5e-15, rtol=4*np.finfo(float).eps)

    east = root(0, 0.5 * PHMI_H * scale)
    north = root(1, 0.5 * PHMI_H * scale)
    vertical = root(2, PHMI_V * scale)
    emt = max(threshold[2] for prior, _, _, threshold, _ in terms if prior >= 1e-5)
    sigma_acc_v = math.sqrt(math.fsum((g0[2, i] * sigma_a[i]) ** 2 for i in range(n)))
    maximum_threshold = max(abs(value) for _, _, _, threshold, _ in terms for value in threshold)
    threshold_rounding_bound = (
        256.0 * n * np.finfo(float).eps * maximum_normal_condition
        * max(1.0, maximum_threshold)
    )
    return {
        "sigma_acc_v_m": sigma_acc_v,
        "emt_m": emt,
        "vpl_root_m": vertical,
        "east_pl_root_m": east,
        "north_pl_root_m": north,
        "hpl_root_m": math.hypot(east, north),
        "p_unmonitored": unmonitored,
        "mode_count": len(modes),
        "monitorable_mode_count": len(terms),
        "threshold_enu_m": [threshold.tolist() for _, _, _, threshold, _ in terms],
        "threshold_rounding_bound_m": threshold_rounding_bound,
        "constellation_priors": [
            next(
                (mode["prior"] for mode in modes if mode["consts"] == {system} and not mode["excluded"]),
                None,
            )
            for system in range(nc)
        ],
    }


def main() -> None:
    document = json.loads(PATH.read_text())
    selected = {
        "n06_c1_nominal",
        "n12_c2_consolidation",
        "n12_c2_low_prior",
        "n24_c2_varied",
    }
    quantile_probabilities = set()
    for case in document["cases"]:
        if case["name"] not in selected:
            continue
        reference = solve_case(case)
        observable = reference["monitorable_mode_count"] - 1
        quantile_probabilities.update((PFA_H / (4.0 * observable), PFA_V / (2.0 * observable)))
        reference["maast_vpl_departure_m"] = case["vpl_m"] - reference["vpl_root_m"]
        reference["maast_hpl_departure_m"] = case["hpl_m"] - reference["hpl_root_m"]
        reference["maast_emt_departure_m"] = case["emt_m"] - reference["emt_m"]
        reference["maast_p_unmonitored_departure"] = case["p_unmonitored"] - reference["p_unmonitored"]
        reference["maast_mode_count_departure"] = (
            len(np.atleast_2d(case["excluded"])) - reference["monitorable_mode_count"]
        )
        if case["name"] == "n24_c2_varied":
            reference["maast_gps_constellation_prior_departure"] = (
                case["mode_probability"][1] - reference["constellation_priors"][0]
            )
            reference["maast_galileo_constellation_prior_departure"] = (
                case["mode_probability"][2] - reference["constellation_priors"][1]
            )
        maast_thresholds = np.atleast_2d(case["threshold_enu_m"])
        add_thresholds = np.asarray(reference["threshold_enu_m"])
        if maast_thresholds.shape == add_thresholds.shape:
            reference["max_maast_threshold_departure_m"] = float(
                np.max(np.abs(maast_thresholds - add_thresholds))
            )
        case["add_reference"] = reference
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    document["provenance"]["add_reference"] = {
        "tool": f"Python; NumPy {np.__version__}; SciPy {scipy.__version__}",
        "method": "ADD v3.0 equations; scipy.optimize.brentq",
        "generator_sha256": script_hash,
        "section_4_6_sentences": [
            "Cj [is] the set of fault modes that are formed of satellite faults included in constellation j (and included in the list established above).",
            "If a fault cannot be monitored, it is not included in the list of fault modes and we move to the next one.",
            "In this case, these events must be removed from the list of faults (and their integrity risk subtracted from the available budget).",
        ],
        "open_reading": "MAAST's reading is used: a constellation-plus-satellite mode belongs to Cj when all affected satellites are in j; unobservable selected modes are filtered and charged through Eq. (18).",
        "consolidation_case_pl_difference": {
            "case": "n12_c2_consolidation",
            "cause": "The selected mode sets and Eq. (17) priors agree. MAAST applies the optional Eqs. (35)-(37) second pass, removes modes whose summed coordinate allocation reaches one, and recomputes thresholds and PLs. The independent reference retains the first-pass Eqs. (31)-(33) roots. Both choices are allowed by the ADD.",
            "add_sentence": "This possible loss of performance can be mitigated by: first, identifying the modes for which we are overestimating the integrity risk, second, by excluding them from the list of monitored faults, and, third, by recomputing the thresholds and Protection Levels with the new list.",
        },
    }
    probabilities = sorted(quantile_probabilities)
    octave_expression = "printf('%.17g\\n'," + ");printf('%.17g\\n',".join(
        f"sqrt(2)*erfcinv(2*{probability:.17g})" for probability in probabilities
    ) + ");"
    octave_values = [
        float(value)
        for value in subprocess.run(
            ["octave", "--quiet", "--eval", octave_expression],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    ]
    document["provenance"]["add_reference"]["maast_norminv_check"] = [
        {
            "tail_probability": probability,
            "octave_erfcinv_quantile": octave,
            "mpmath_60_digit_quantile": float(
                mp.sqrt(2) * mp.erfinv(1 - 2 * mp.mpf(str(probability)))
            ),
            "octave_minus_mpmath": octave - float(
                mp.sqrt(2) * mp.erfinv(1 - 2 * mp.mpf(str(probability)))
            ),
        }
        for probability, octave in zip(probabilities, octave_values)
    ]
    PATH.write_text(json.dumps(document, separators=(",", ":"), ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
