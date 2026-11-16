#!/usr/bin/env python3
"""Generate independent WG-C ADD and SciPy ARAIM oracle values."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import ncx2, norm
from generate_araim_add_matrix import solve_case

SOURCE = "https://web.stanford.edu/group/scpnt/gpslab/website_files/maast/ARAIM_TSG_Reference_ADD_v3.0.pdf"
OUT = Path(__file__).parents[1] / "tests/fixtures/araim_oracles.json"


def add_sigma_variance(elevation_deg: float, aad_b: bool) -> float:
    elevation = math.radians(elevation_deg)
    tropo = 0.12 * 1.001 / math.sqrt(0.002001 + math.sin(elevation) ** 2)
    b0, b1 = (0.11, 0.13) if aad_b else (0.15, 0.43)
    noise = b0 + b1 * math.exp(-elevation_deg / 6.9)
    multipath = 0.13 + 0.53 * math.exp(-elevation_deg / 10.0)
    gamma = (1575.42**4 + 1176.45**4) / (1575.42**2 - 1176.45**2) ** 2
    return tropo**2 + gamma * (noise**2 + multipath**2)


GALILEO_USER_SIGMA_M = {
    5.0: 0.4529, 10.0: 0.3553, 15.0: 0.3063, 20.0: 0.2638,
    25.0: 0.2593, 30.0: 0.2555, 35.0: 0.2504, 40.0: 0.2438,
    45.0: 0.2396, 50.0: 0.2359, 55.0: 0.2339, 60.0: 0.2302,
    65.0: 0.2295, 70.0: 0.2278, 75.0: 0.2297, 80.0: 0.2310,
    85.0: 0.2274, 90.0: 0.2277,
}

APP_D_ROWS = [
    ([0.0225, 0.9951, -0.0966], 3.8865, 3.5740),
    ([0.6750, -0.6900, -0.2612], 1.4377, 1.1252),
    ([0.0723, -0.6601, -0.7477], 0.8604, 0.5479),
    ([-0.9398, 0.2553, -0.2269], 1.6383, 1.3258),
    ([-0.5907, -0.7539, -0.2877], 1.3229, 1.0104),
    ([-0.3236, -0.0354, -0.9455], 0.8434, 0.5309),
    ([-0.6748, 0.4356, -0.5957], 0.8963, 0.5838),
    ([0.0938, -0.7004, -0.7075], 0.8669, 0.5544),
    ([0.5571, 0.3088, -0.7709], 0.8573, 0.5448),
    ([0.6622, 0.6958, -0.2780], 1.3616, 1.0491),
]


def synthetic_case(n: int, nc: int, p_sat: float, p_const: float) -> dict:
    elevations = [math.radians(15.0 + 60.0 * ((0.61803 * i) % 1.0)) for i in range(n)]
    local = [0.25 + 2.0 * math.exp(-math.degrees(value) / 15.0) for value in elevations]
    return {
        "n_sat": n, "n_const": nc, "p_sat": p_sat, "p_const": p_const,
        "ura_m": 0.75, "ure_m": 0.5, "bias_m": 0.75,
        "elevation_rad": elevations,
        "cint_m2": [0.75**2 + value for value in local],
        "cacc_m2": [0.5**2 + value for value in local],
        "max_fault_order": 2, "p_threshold_unmonitored": 8e-8,
    }


def appendix_case(nc: int, p_const: float) -> dict:
    system_index = [0] * 10 if nc == 1 else [0] * 5 + [1] * 5
    design_matrix = [row + [float(system == clock) for clock in range(nc)]
                     for (row, _, _), system in zip(APP_D_ROWS, system_index)]
    return {
        "n_sat": 10, "n_const": nc, "p_sat": 1e-5, "p_const": p_const,
        "ura_m": 0.75, "ure_m": 0.5, "bias_m": 0.5,
        "elevation_rad": [math.asin(-row[2]) for row, _, _ in APP_D_ROWS],
        "cint_m2": [cint for _, cint, _ in APP_D_ROWS],
        "cacc_m2": [cacc for _, _, cacc in APP_D_ROWS],
        "design_matrix": design_matrix, "system_index": system_index,
        "max_fault_order": 2, "p_threshold_unmonitored": 8e-8,
    }


def main() -> None:
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    octave_generator = Path(__file__).with_name("generate_araim_maast.m")
    alpha, beta = 1.0e-3, 0.2
    delta = norm.isf(alpha / 2.0) + norm.isf(beta)
    # ADD v3.0 section 4.6 first selects the ten observable satellite-only
    # modes. The constellation-only mode is unobservable, so Eq. (18) charges
    # it once; it must not also be added to the residual Eq. (15) mass.
    getcontext().prec = 80
    p_sat = Decimal("1e-5")
    p_const = Decimal("1e-8")
    p0 = (1 - p_sat) ** 10 * (1 - p_const)
    p_sat_only = p0 * p_sat / (1 - p_sat)
    low_prior_unmonitored = float(1 - p0 - 10 * p_sat_only)
    appendix = solve_case(appendix_case(2, 1.0e-4))
    synthetic_two = solve_case(synthetic_case(22, 2, 1.0e-5, 1.0e-4))
    synthetic_fifteen = solve_case(synthetic_case(15, 3, 1.0e-5, 1.0e-4))
    single_constellation = solve_case(appendix_case(1, 9.0e-8))
    unobservable_constellation = solve_case(appendix_case(1, 1.0e-8))
    varied_case = synthetic_case(24, 2, 1.2e-5, 1.0e-4)
    varied_case["cint_m2"] = [1.4**2] * 24
    varied_case["cacc_m2"] = [1.1**2] * 24
    varied = solve_case(varied_case)
    reliability_cases = []
    for case_alpha, case_beta in ((1.0e-3, 0.2), (1.0e-5, 0.1), (0.05, 0.4)):
        critical = norm.isf(case_alpha / 2.0)
        case_delta = critical + norm.isf(case_beta)
        case_lambda = case_delta * case_delta
        reliability_cases.append(
            {
                "alpha": case_alpha,
                "beta": case_beta,
                "delta0": case_delta,
                "lambda0": case_lambda,
                "chi_square_critical": critical * critical,
                "ncx2_cdf_df1": ncx2.cdf(critical * critical, 1, case_lambda),
                "ncx2_sf_df1": ncx2.sf(critical * critical, 1, case_lambda),
            }
        )
    data = {
        "provenance": {
            "tool": f"Python {'.'.join(map(str, __import__('sys').version_info[:3]))}; NumPy {np.__version__}; SciPy {scipy.__version__}",
            "source_url": SOURCE,
            "generator_sha256": script_hash,
            "maast": {
                "tool": "MAAST ARAIM 0.4; GNU Octave 11.3.0",
                "source_url": "https://web.stanford.edu/group/scpnt/gpslab/website_files/maast/maast_araimpb0.4.zip",
                "generator_sha256": hashlib.sha256(octave_generator.read_bytes()).hexdigest(),
            },
            "float_format": "17 significant digits",
        },
        "appendix_d_v3_0": {
            "inputs": {"p_sat": 1.0e-5, "p_const": [1.0e-4, 1.0e-4]},
            "add_roots": {"vpl_m": appendix["vpl_root_m"], "hpl_m": appendix["hpl_root_m"]},
            "threshold_rounding_bound_m": appendix["threshold_rounding_bound_m"],
            "maast_upper_bisection": {"vpl_m": 19.223340, "hpl_m": 14.464131},
            "emt_m": 7.8193787,
            "sigma_acc_v_m": 1.469759,
            "mode_count": 12,
        },
        "edge_cases": {
            "synthetic_two_constellations": {"vpl_m": synthetic_two["vpl_root_m"], "hpl_m": synthetic_two["hpl_root_m"], "emt_m": synthetic_two["emt_m"], "threshold_rounding_bound_m": synthetic_two["threshold_rounding_bound_m"]},
            "synthetic_fifteen_satellites_three_constellations": {"mode_count": synthetic_fifteen["mode_count"] - 1, "vpl_m": synthetic_fifteen["vpl_root_m"], "hpl_m": synthetic_fifteen["hpl_root_m"], "threshold_rounding_bound_m": synthetic_fifteen["threshold_rounding_bound_m"]},
            "single_constellation_unmonitored_mass": {"vpl_m": single_constellation["vpl_root_m"], "hpl_m": single_constellation["hpl_root_m"], "p_unmonitored": single_constellation["p_unmonitored"], "threshold_rounding_bound_m": single_constellation["threshold_rounding_bound_m"]},
            "low_constellation_prior": {"mode_count": 10, "p_unmonitored": low_prior_unmonitored, "first_order_satellite_prior": float(p_sat_only), "emt_m": 0.0},
            "unobservable_constellation": {"vpl_m": unobservable_constellation["vpl_root_m"], "hpl_m": unobservable_constellation["hpl_root_m"], "threshold_rounding_bound_m": unobservable_constellation["threshold_rounding_bound_m"]},
            "varied_prior_two_constellations": {"vpl_m": varied["vpl_root_m"], "hpl_m": varied["hpl_root_m"], "p_unmonitored": varied["p_unmonitored"], "mode_count": varied["mode_count"], "threshold_rounding_bound_m": varied["threshold_rounding_bound_m"]},
            "large_fault_prior": {"fault_free": 0.16807, "single_mode": 0.07203, "unmonitored": 0.47178},
        },
        "sigma_model": [
            {"elevation_deg": x, "v3_0_variance_m2": add_sigma_variance(x, False), "v3_1_variance_m2": add_sigma_variance(x, True)}
            for x in (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
        ],
        "galileo_user_sigma_model": [
            {"elevation_deg": elevation, "user_sigma_m": sigma}
            for elevation, sigma in GALILEO_USER_SIGMA_M.items()
        ],
        "reliability": {
            "inputs": {"alpha": alpha, "beta": beta},
            "delta0": delta,
            "lambda0": delta * delta,
            "ncx2_cdf_df1_at_lambda": ncx2.cdf(delta * delta, 1, delta * delta),
            "negative_delta_inputs": {"alpha": 0.9, "beta": 0.9},
            "noncentral_chi_square_cases": reliability_cases,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
