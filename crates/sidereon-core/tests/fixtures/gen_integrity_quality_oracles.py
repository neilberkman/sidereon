#!/usr/bin/env python3
"""Generate SciPy chi-square and Skyfield topocentric reference values."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mpmath as mp
import scipy
from scipy.stats import chi2
from skyfield.api import load, wgs84
from skyfield.constants import AU_KM
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric
import skyfield

ROOT = Path(__file__).resolve().parent
def main():
    mp.mp.dps = 400
    lower = [10.0 ** -e for e in range(1, 16)]
    probabilities = sorted(set(lower + [0.5] + [1.0 - p for p in lower]))
    dofs = [1, 2, 3, 4, 10, 30, 100, 500, 1000]
    chi_cases = [
        {"p": format(p, ".17g"), "dof": k, "ppf": format(float(chi2.ppf(p, k)), ".17g")}
        for p in probabilities for k in dofs
    ]
    quality_edge_chi2 = {
        "raim_tiny_tail": {
            "p_fa": "1e-17",
            "dof": 2,
            "isf": format(float(chi2.isf(1e-17, 2)), ".17g"),
        },
        "lower_tail": [
            {"p": "1e-20", "dof": 1,
             "ppf": format(float(chi2.ppf(1e-20, 1)), ".17g")},
            {"p": "1e-300", "dof": 2,
             "ppf": format(float(chi2.ppf(1e-300, 2)), ".17g")},
        ],
        "million_dof": [
            {"p": format(p, ".17g"), "dof": 1_000_000,
             "ppf": format(float(chi2.ppf(p, 1_000_000)), ".17g")}
            for p in (0.5, 0.05, 0.001)
        ],
    }
    chi2_dof1_lower_tail = [
        {
            "p": format(p, ".17g"),
            "ppf": format(float(2 * mp.erfinv(mp.mpf(p)) ** 2), ".17g"),
        }
        for p in sorted({10.0 ** -e for e in range(3, 301)} | {1.1e-15, 1.0e-14})
    ]
    chi2_dof1_upper_tail = [
        {
            "p": format(p, ".17g"),
            "isf": format(float(2 * mp.erfinv(1 - mp.mpf(p)) ** 2), ".17g"),
        }
        for p in sorted({10.0 ** -e for e in range(1, 301)} | {0.5})
    ]
    switch = 1.0e-3
    chi2_dof1_switch = [
        {
            "p": format(p, ".17g"),
            "ppf": format(float(2 * mp.erfinv(mp.mpf(p)) ** 2), ".17g"),
        }
        for p in (np.nextafter(switch, 0.0), switch, np.nextafter(switch, 1.0))
    ]
    erfc_inv_grid = []
    erfc_inputs = {np.nextafter(0.0, 1.0), 0.5, 1.0, 1.5,
                   np.nextafter(2.0, 0.0)}
    erfc_inputs.update(10.0 ** -e for e in range(1, 301))
    erfc_inputs.update(2.0 - 10.0 ** -e for e in range(1, 16))
    for value in sorted(erfc_inputs):
        y = mp.mpf(value)
        erfc_inv_grid.append({
            "y": format(value, ".17g"),
            "value": format(float(mp.erfinv(1 - y)), ".17g"),
        })
    grid_probabilities = sorted({0.5} | {10.0 ** -e for e in
                                        (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300)})
    chi2_mpmath_grid = []
    for dof in (2, 3, 5, 10, 30):
        shape = mp.mpf(dof) / 2
        for probability in grid_probabilities:
            p = mp.mpf(format(probability, ".17g"))
            lower_guess = mp.mpf(float(chi2.ppf(probability, dof)))
            upper_guess = mp.mpf(float(chi2.isf(probability, dof)))
            lower_root = mp.findroot(
                lambda x: mp.gammainc(shape, 0, x / 2, regularized=True) - p,
                (lower_guess * (1 - mp.mpf("1e-12")),
                 lower_guess * (1 + mp.mpf("1e-12"))))
            upper_root = mp.findroot(
                lambda x: mp.gammainc(shape, x / 2, mp.inf, regularized=True) - p,
                (upper_guess * (1 - mp.mpf("1e-12")),
                 upper_guess * (1 + mp.mpf("1e-12"))))
            chi2_mpmath_grid.append({
                "p": format(probability, ".17g"), "dof": dof,
                "ppf": format(float(lower_root), ".17g"),
                "isf": format(float(upper_root), ".17g"),
            })
    ceiling_dof, ceiling_x, ceiling_lambda = 100_000, mp.mpf("100250"), mp.mpf("25")
    ceiling_cdf = mp.fsum(
        mp.exp(-ceiling_lambda / 2) * (ceiling_lambda / 2) ** index / mp.factorial(index)
        * mp.gammainc(mp.mpf(ceiling_dof) / 2 + index, 0, ceiling_x / 2,
                      regularized=True)
        for index in range(200)
    )
    noncentral_log_cdf_cases = []
    for dof, x, noncentrality in (
        (2, "1000", "50"),
        (2, "503.995", "100"),
        (2, "520.646", "150"),
        (2, "704.99", "100"),
        (2, "704.99", "100.0000001"),
        (20, "1e-40", "1"),
        (1000, "100", "100"),
    ):
        x_mp = mp.mpf(x)
        lambda_mp = mp.mpf(noncentrality)
        mean = lambda_mp / 2
        terms = []
        weight = mp.exp(-mean)
        for index in range(10000):
            terms.append(weight * mp.gammainc(
                mp.mpf(dof) / 2 + index, 0, x_mp / 2, regularized=True))
            weight *= mean / (index + 1)
            if index > mean + 100 and weight < mp.mpf("1e-390"):
                break
        noncentral_log_cdf_cases.append({
            "dof": dof, "x": x, "lambda": noncentrality,
            "log_cdf": mp.nstr(mp.log(mp.fsum(terms)), 52),
        })
    def ncx2_log_cdf_mpmath(dof, x_value, lambda_value):
        x_mp = mp.mpf(x_value)
        lambda_mp = mp.mpf(lambda_value)
        if lambda_mp == 0:
            return mp.log(mp.gammainc(mp.mpf(dof) / 2, 0, x_mp / 2,
                                      regularized=True))
        if dof == 1:
            radius = mp.sqrt(x_mp)
            displacement = mp.sqrt(lambda_mp)
            lower_tail = mp.erfc((radius - displacement) / mp.sqrt(2)) / 2
            upper_tail = mp.erfc((radius + displacement) / mp.sqrt(2)) / 2
            return mp.log(1 - lower_tail - upper_tail)
        if lambda_mp > 1000:
            displacement = mp.sqrt(lambda_mp)
            radius = mp.sqrt(x_mp)
            lower = max(-mp.mpf("40"), -displacement - radius)
            upper = min(mp.mpf("40"), radius - displacement)
            if lower >= upper:
                return mp.ninf
            shape = mp.mpf(dof - 1) / 2
            normalizer = mp.sqrt(2 * mp.pi)
            def conditional_integrand(z):
                remainder = x_mp - (displacement + z) ** 2
                if remainder <= 0:
                    return mp.mpf("0")
                perpendicular = mp.gammainc(shape, 0, remainder / 2,
                                            regularized=True)
                return mp.exp(-z * z / 2) * perpendicular / normalizer
            knots = [lower]
            knots.extend(point for point in (-12, -6, -3, 0, 3, 6, 12)
                         if lower < point < upper)
            knots.append(upper)
            return mp.log(mp.quad(conditional_integrand, knots))
        poisson_mean = lambda_mp / 2
        weight = mp.exp(-poisson_mean)
        total = mp.mpf("0")
        maximum_index = int(mp.ceil(poisson_mean + 40 * mp.sqrt(poisson_mean + 1) + 200))
        for index in range(maximum_index + 1):
            total += weight * mp.gammainc(
                mp.mpf(dof) / 2 + index, 0, x_mp / 2, regularized=True)
            weight *= poisson_mean / (index + 1)
        return mp.log(total)

    noncentral_log_cdf_grid = []
    grid_dofs = (1, 2, 3, 4, 8, 20, 100, 1000, 100000)
    grid_x = ("1e-40", "1e-6", "1", "100", "1e6", "1e12")
    lambda_switch = mp.mpf("100")
    grid_lambda = (
        "0", "1e-12", "1",
        mp.nstr(mp.mpf(np.nextafter(100.0, 0.0)), 25),
        "100",
        mp.nstr(mp.mpf(np.nextafter(100.0, np.inf)), 25),
        "150",
    )
    for dof in grid_dofs:
        for x_value in grid_x:
            for lambda_value in grid_lambda:
                log_cdf = ncx2_log_cdf_mpmath(dof, x_value, lambda_value)
                noncentral_log_cdf_grid.append({
                    "dof": dof, "x": str(x_value), "lambda": str(lambda_value),
                    "log_cdf": mp.nstr(log_cdf, 80),
                })
    for dof in grid_dofs:
        noncentral_log_cdf_grid.append({
            "dof": dof, "x": "1e12", "lambda": "1e12",
            "log_cdf": mp.nstr(ncx2_log_cdf_mpmath(dof, "1e12", "1e12"), 80),
        })

    ts = load.timescale(builtin=True)
    t = ts.utc(2024, 1, 1, 12, 0, 0)
    sites = [(51.5, -0.1, 11.0), (40.7, -74.0, 10.0), (-33.86, 151.21, 58.0)]
    sky_cases = []
    for site_index, (lat, lon, height) in enumerate(sites):
        observer = wgs84.latlon(lat, lon, elevation_m=height)
        observer_itrs = observer.at(t).frame_xyz(itrs).km
        for mask in [11.0, 15.0, 22.0, 30.0, 44.0, 60.0, 69.0, 88.0]:
            for offset in [-1e-6, 0.0, 1e-6]:
                wanted = mask + offset
                azimuth = 37.0 + site_index * 71.0
                azr, elr = np.deg2rad([azimuth, wanted])
                enu = 20000.0 * np.array([
                    np.cos(elr) * np.sin(azr),
                    np.cos(elr) * np.cos(azr), np.sin(elr)])
                ph, la = np.deg2rad([lat, lon])
                ecef_to_enu = np.array([
                    [-np.sin(la), np.cos(la), 0.0],
                    [-np.sin(ph)*np.cos(la), -np.sin(ph)*np.sin(la), np.cos(ph)],
                    [np.cos(ph)*np.cos(la), np.cos(ph)*np.sin(la), np.sin(ph)]])
                target_itrs = observer_itrs + ecef_to_enu.T @ enu
                target_gcrs = itrs.rotation_at(t).T @ target_itrs
                target = Geocentric(target_gcrs / AU_KM, t=t, center=399)
                alt, az, distance = (target - observer.at(t)).altaz(
                    temperature_C=None, pressure_mbar=0.0)
                sky_cases.append({
                    "site_deg_m": [lat, lon, height],
                    "observer_itrs_km": [format(float(x), ".17g") for x in observer_itrs],
                    "target_itrs_km": [format(float(x), ".17g") for x in target_itrs],
                    "mask_deg": format(mask, ".17g"), "offset_deg": format(offset, ".17g"),
                    "azimuth_deg": format(float(az.degrees), ".17g"),
                    "elevation_deg": format(float(alt.degrees), ".17g"),
                    "range_km": format(float(distance.km), ".17g"),
                    "visible": bool(alt.degrees >= mask),
                })
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    doc = {
        "provenance": {"tool": "SciPy, mpmath and Skyfield", "scipy_version": scipy.__version__,
                       "mpmath_version": mp.__version__,
                       "numpy_version": np.__version__, "skyfield_version": skyfield.__version__,
                       "generator": Path(__file__).name, "generator_sha256": source_hash,
                       "refraction": "disabled (temperature_C=None, pressure_mbar=0.0)"},
        "chi2_cases": chi_cases,
        "quality_edge_chi2": quality_edge_chi2,
        "chi2_dof1_lower_tail": chi2_dof1_lower_tail,
        "chi2_dof1_upper_tail": chi2_dof1_upper_tail,
        "chi2_dof1_switch": chi2_dof1_switch,
        "erfc_inv_grid": erfc_inv_grid,
        "chi2_mpmath_grid": chi2_mpmath_grid,
        "noncentral_supported_dof_ceiling": {
            "dof": ceiling_dof, "x": str(ceiling_x), "lambda": str(ceiling_lambda),
            "log_cdf": mp.nstr(mp.log(ceiling_cdf), 52),
        },
        "noncentral_log_cdf_cases": noncentral_log_cdf_cases,
        "noncentral_log_cdf_grid": noncentral_log_cdf_grid,
        "skyfield_cases": sky_cases,
    }
    (ROOT / "integrity_quality_scipy_skyfield.json").write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
