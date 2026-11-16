"""Generate extended chi-square quantiles with mpmath gamma-tail roots.

SciPy supplies starting guesses only; each root is refined independently at
90 decimal digits. Probabilities are exact binary64 inputs.
"""
import json
import hashlib
from pathlib import Path
import mpmath as mp
import numpy as np
import scipy
from scipy.stats import chi2

mp.mp.dps = 90
cases = []
switches = []
probabilities = sorted({10.0**-e for e in (1, 3, 6, 15, 41, 100, 180, 300)}
                       | {float.fromhex('0x0.0000000000001p-1022'), 1e-323, 1e-321, 1e-320,
                          1e-315, 1e-312, 1e-310, np.finfo(float).tiny, 0.01}
                       | {np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), np.nextafter(1.0, 0.0)}
                       | {1.0-10.0**-e for e in (3, 6, 12, 15)})
for dof in (1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 30, 31, 32, 33, 47, 50, 62, 63, 64, 65,
            100, 341, 342, 500, 1000, 10000, 100000):
    a = mp.mpf(dof)/2
    # The series handover tests its leading-term seed, not the final root.
    boundary = float(mp.exp(a*mp.log(mp.mpf('0.01'))-mp.loggamma(a+1)))
    band = {np.nextafter(1e-6, 0.0), 1e-6, np.nextafter(1e-6, 1.0),
            boundary, np.nextafter(boundary, 0.0), np.nextafter(boundary, 1.0),
            1.6666e-7, 8.5e-9, 9.99999e-7}
    for probability in sorted(set(probabilities) | band):
        if not 0 < probability < 1:
            continue
        p = mp.mpf(probability)
        a = mp.mpf(dof)/2
        result = {"dof": dof, "p": repr(float(probability))}
        for upper, name in ((False, "ppf"), (True, "isf")):
            if dof == 1:
                with mp.workdps(400):
                    root = 2*mp.erfinv(1-p if upper else p)**2
            else:
                guess = chi2.isf(probability, dof) if upper else chi2.ppf(probability, dof)
                x = mp.mpf(float(guess))
                if not mp.isfinite(x) or x <= 0:
                    x = 2*mp.exp((mp.log(p)+mp.loggamma(a+1))/a) if not upper else 2*(-mp.log(p)+a)
                target = mp.log(p)
                for _ in range(100):
                    tail = mp.gammainc(a, x/2, mp.inf, regularized=True) if upper else mp.gammainc(a, 0, x/2, regularized=True)
                    density = mp.exp((a-1)*mp.log(x/2)-x/2-mp.loggamma(a))/2
                    delta = (mp.log(tail)-target)*tail/density
                    x += delta if upper else -delta
                    if abs(delta) < abs(x)*mp.mpf("1e-80"):
                        break
                else:
                    raise ArithmeticError((dof, probability, name))
                root = x
            result[name] = repr(float(root))
        cases.append(result)
noncentral_minimum_x = []
for dof in (1, 2, 3, 20, 100, 1000, 100000):
    for noncentrality in (0, 1, 100):
        x = mp.mpf(float(np.nextafter(0.0, 1.0)))
        a = mp.mpf(dof)/2
        mean = mp.mpf(noncentrality)/2
        terms = [mp.exp(-mean)*mp.gammainc(a, 0, x/2, regularized=True)]
        weight = mp.exp(-mean)
        for index in range(1, 10):
            weight *= mean/index
            terms.append(weight*mp.gammainc(a+index, 0, x/2, regularized=True))
        noncentral_minimum_x.append({"x": repr(float(x)), "dof": dof,
            "lambda": noncentrality, "log_cdf": repr(float(mp.log(mp.fsum(terms))))})
out = {"provenance": {"generator": "fixtures-generators/generate_chi2_extended.py",
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "method": "mpmath direct regularized gamma tail roots",
                       "precision_decimal_digits": 90,
                       "mpmath_version": mp.__version__, "scipy_version": scipy.__version__,
                       "numpy_version": np.__version__}, "cases": cases,
       "noncentral_minimum_x": noncentral_minimum_x}
for dof in range(1, 65):
    a = mp.mpf(dof)/2
    boundary = float(mp.exp(a*mp.log(mp.mpf('0.01'))-mp.loggamma(a+1)))
    switches.extend({"dof": dof, "p": repr(float(p))}
                    for p in (1e-10, 1e-6, boundary, 0.001, 0.01, 0.5, 0.9) if p > 0)
out["lower_tail_series_switches"] = switches
# Include every tail-equation or direction boundary, its mirror through
# 1-p, and the leading-seed handover. Adjacent-input scans cover both public
# inverses, including shapes on either side of the finite-product boundary.
out["equation_switches"] = [
    {"dof": dof, "p": repr(float(p))}
    for dof in (*range(1, 66), 100, 341, 342, 1000, 10000, 100000)
    for p in sorted({1e-10, 1e-8, 0.1, 0.5, 0.9, 1-1e-8, 1-1e-10}
                    | {float(mp.exp(mp.mpf(dof)/2*mp.log(mp.mpf(dof)/20)
                                    - mp.loggamma(mp.mpf(dof)/2+1)))})
    if 0 < p < 1
]
out["equation_switch_roots"] = []
for dof in (64, 65, 100, 341, 342, 1000, 10000, 100000):
    a = mp.mpf(dof)/2
    for p_float in (np.nextafter(0.1, 0.0), 0.1, np.nextafter(0.1, 1.0)):
        p = mp.mpf(float(p_float))
        x = mp.mpf(float(chi2.isf(p_float, dof)))
        for _ in range(100):
            tail = mp.gammainc(a, x/2, mp.inf, regularized=True)
            density = mp.exp((a-1)*mp.log(x/2)-x/2-mp.loggamma(a))/2
            delta = (tail-p)/density
            x += delta
            if abs(delta) < abs(x)*mp.mpf('1e-80'):
                break
        else:
            raise ArithmeticError((dof, p_float))
        endpoint = float(x)
        if mp.mpf(endpoint) < x:
            endpoint = np.nextafter(endpoint, np.inf)
        out["equation_switch_roots"].append({"dof": dof, "p": repr(float(p_float)),
            "root_decimal": mp.nstr(x, 80), "upper_endpoint": repr(float(endpoint))})
out["provenance"]["switch_roots"] = (
    "90-digit direct upper regularized-gamma Newton roots; upper_endpoint is "
    "the least binary64 value not below the root, not nearest rounding")
near_one = sorted({np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)}
                  | {1.0011016224500742}
                  | {1+sign*float(z) for z in np.linspace(0.001, 0.011, 101) for sign in (-1, 1)}
                  | {1.0+sign*10.0**-e for e in range(3, 16) for sign in (-1, 1)})
out["erfc_near_one"] = [{"y": repr(float(y)),
    "value": repr(float(mp.erfinv(1-mp.mpf(float(y)))))} for y in near_one]
out["gamma_scale_large_argument"] = [
    {"a": a, "x": repr(float(x)),
     "value": repr(float(mp.exp(mp.mpf(a)*mp.log(x)-mp.mpf(x)-mp.loggamma(a))))}
    for a in (32, 64, 500, 50000)
    for x in (1e15, 5e19, 1e20, 1e100, np.finfo(float).max)]
destination = Path(__file__).resolve().parents[1]/"tests/fixtures/chi2_extended_mpmath.json"
destination.write_text(json.dumps(out, indent=2)+"\n")
