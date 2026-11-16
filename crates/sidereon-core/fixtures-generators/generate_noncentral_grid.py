#!/usr/bin/env python3
"""Noncentral chi-square reference grid, using positive high-precision integrals.

The large-noncentrality integral conditions on the *parallel* normal
coordinate, integrating a central chi-square CDF over that coordinate. It
does not use perpendicular-radius Gauss-Legendre panels. Small noncentrality
uses a positive Poisson mixture of mpmath incomplete gamma functions.

Run with a Python installation providing mpmath. Completed cells are cached
in the output so generation can be resumed without changing their values.
Then run derive_noncentral_bounds.py to attach the independent error budgets.
"""
import hashlib
import json
import math
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mpmath as mp

mp.mp.dps = 90
DEST = Path(__file__).resolve().parents[1] / "tests/fixtures/noncentral_log_cdf_mpmath.json"
DOFS = (1, 2, 3, 4, 8, 20, 100, 1000, 100000)
LAMBDAS = (0.0, 1e-12, 0.01, 1.0, math.nextafter(100.0, 0.0),
           100.0, math.nextafter(100.0, math.inf), 150.0, 1000.0, 1e6, 1e12)


def normal_tail(z):
    return mp.erfc(z / mp.sqrt(2)) / 2


@lru_cache(maxsize=None)
def log_gamma_70(a):
    with mp.workdps(70):
        return mp.loggamma(a)


def large_shape_upper_gamma(a, z):
    """DLMF 8.9.2, unit-denominator fraction with an enclosed tail.

    https://dlmf.nist.gov/8.9.E2
    For z>a and k<a, the odd tail 1+(k-a)/z/(1+k/z/(...)) lies between
    1-a/z and 1. Propagating those endpoints backwards encloses the fraction,
    independently of a floating-point continued-fraction stopping test.
    """
    with mp.workdps(70):
        for depth in (16, 32, 64, 128, 256, 512):
            assert depth+1 < a
            lower, upper = 1-a/z, mp.mpf(1)
            for index in range(2*depth, 0, -1):
                k = (index+1)//2
                coefficient = (k-a)/z if index % 2 else k/z
                if coefficient >= 0:
                    lower, upper = 1+coefficient/upper, 1+coefficient/lower
                else:
                    lower, upper = 1+coefficient/lower, 1+coefficient/upper
                assert lower > 0
            if (upper-lower)/lower < mp.mpf("1e-45"):
                factor = 2/(z*(lower+upper))
                return factor*mp.exp(a*mp.log(z)-z-log_gamma_70(a))
    raise ArithmeticError("upper-gamma fraction interval did not contract")


def gamma_tail(a, z, upper):
    if a == mp.mpf("0.5"):
        return mp.erfc(mp.sqrt(z)) if upper else mp.erf(mp.sqrt(z))
    if a == 1:
        return mp.exp(-z) if upper else -mp.expm1(-z)
    if a >= 1000 and z > a+8*mp.sqrt(a):
        q = large_shape_upper_gamma(a, z)
        return q if upper else 1-q
    return mp.gammainc(a, z, mp.inf, regularized=True) if upper else mp.gammainc(a, 0, z, regularized=True)


def chernoff(x, dof, lam):
    s = (dof + mp.sqrt(dof*dof + 4*lam*x)) / (2*x)
    t = (1-s)/2
    return -mp.mpf(dof)/2*mp.log(s) + lam*t/s - t*x


def mixture(x, dof, lam, upper):
    a, z, mean = mp.mpf(dof)/2, x/2, lam/2
    count = 1200
    weights = [mp.exp(-mean)]
    for j in range(count):
        weights.append(weights[-1]*mean/(j+1))
    if upper:
        gamma = mp.gammainc(a, z, mp.inf, regularized=True)
        increment = mp.exp(a*mp.log(z)-z-mp.loggamma(a+1))
        terms = []
        for j, weight in enumerate(weights):
            terms.append(weight*gamma)
            gamma += increment
            increment *= z/(a+j+1)
    else:
        gamma = mp.gammainc(a+count, 0, z, regularized=True)
        increment = mp.exp((a+count-1)*mp.log(z)-z-mp.loggamma(a+count))
        terms = []
        for j in range(count, -1, -1):
            terms.append(weights[j]*gamma)
            gamma += increment
            increment *= (a+j-1)/z
    total = mp.fsum(terms)
    # For the lower mixture each incomplete gamma decreases with its shape;
    # for the upper mixture it is bounded by one. The remaining Poisson mass
    # is bounded by its first omitted term divided by 1 - mean/(count+2).
    remainder = weights[-1]*mean/(count+1)/(1-mean/(count+2))
    if not upper:
        remainder *= mp.gammainc(a+count+1, 0, z, regularized=True)
    assert remainder < total*mp.mpf("1e-70")
    return mp.log(total)


def parallel_integral(x, dof, lam, upper):
    r, d, a = mp.sqrt(x), mp.sqrt(lam), mp.mpf(dof-1)/2
    def log_integrand(y):
        if y >= r:
            return mp.ninf
        z = (r-y)*(r+y)/2
        tail = gamma_tail(a, z, upper)
        return -(y-d)**2/2 - mp.log(2*mp.pi)/2 + mp.log1p(mp.exp(-2*d*y)) + mp.log(tail)
    # Locate the maximum independently by golden-section search. Localization
    # matters when r is a million but the integrand width is less than one.
    lo, hi = mp.mpf(0), r
    ratio = (mp.sqrt(5)-1)/2
    p, q = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fp, fq = log_integrand(p), log_integrand(q)
    for _ in range(160):
        if fp > fq:
            hi, q, fq = q, p, fp
            p = hi-ratio*(hi-lo)
            fp = log_integrand(p)
        else:
            lo, p, fp = p, q, fq
            q = lo+ratio*(hi-lo)
            fq = log_integrand(q)
    mode = (lo+hi)/2
    peak = log_integrand(mode)
    curvature = mp.diff(log_integrand, mode, 2)
    slope = mp.diff(log_integrand, mode)
    # At an endpoint maximum (notably dof=3 in the upper tail), the
    # logarithm can be linear: its slope, not its curvature, sets the width.
    width = 1/max(abs(slope), mp.sqrt(abs(curvature)), 1/r)
    knots = {mp.mpf(0), r, mode}
    for offset in (mp.mpf("0.25"), mp.mpf("0.5"), 1, 2, 4, 8, 16, 32, 64):
        for sign in (-1, 1):
            point = mode+sign*offset*width
            if 0 < point < r:
                knots.add(point)
            # A sharp central-gamma transition can give a narrow peak with
            # a broad normal flank. Resolve the normal core independently.
            normal_point = d+sign*offset
            if 0 < normal_point < r:
                knots.add(normal_point)
    with mp.workdps(50):
        for degree in (5, 6, 7, 8, 9):
            integral, error = mp.quad(lambda y: mp.exp(log_integrand(y)-peak), sorted(knots), error=True,
                                      method="gauss-legendre" if dof >= 100 else "tanh-sinh",
                                      maxdegree=degree)
            if error < integral*mp.mpf("1e-35"):
                break
    assert error < integral*mp.mpf("1e-35"), (dof, x, lam, error/integral)
    result = peak+mp.log(integral)
    if upper:
        outside = normal_tail(r-d)+normal_tail(r+d)
        result = mp.log(mp.exp(result)+outside)
    return result


def reference(x_float, dof, lam_float):
    x, lam = mp.mpf(x_float), mp.mpf(lam_float)
    upper = x >= dof+lam
    bound = chernoff(x, dof, lam)
    # Mark only tails proved to round to zero in binary64. Log-CDF values in
    # the lower tail themselves never underflow, even when the CDF does.
    if upper and bound < mp.log(mp.mpf(2)**-1075):
        return "0.0", mp.nstr(bound, 30), "chernoff_underflow"
    if dof == 1:
        r, d = mp.sqrt(x), mp.sqrt(lam)
        if upper:
            log_tail = mp.log(normal_tail(r-d)+normal_tail(r+d))
        else:
            log_tail = mp.log((mp.erfc((d-r)/mp.sqrt(2))-mp.erfc((d+r)/mp.sqrt(2)))/2)
    elif lam == 0:
        a = mp.mpf(dof)/2
        log_tail = mp.log(mp.gammainc(a, x/2, mp.inf, regularized=True) if upper
                          else mp.gammainc(a, 0, x/2, regularized=True))
    elif lam <= 150:
        log_tail = mixture(x, dof, lam, upper)
    elif not upper and lam*x < 4*(mp.mpf(dof)/2+1):
        # Conditioned Poisson indices are small even when lambda/2 is huge.
        # P(a+j+1,z)/P(a+j,z) <= z/(a+j+1), so the next term's geometric
        # majorant bounds the remaining positive sum without a normal integral.
        a, mean = mp.mpf(dof)/2, lam/2
        logs = []
        for j in range(1000):
            term = j*mp.log(mean)-mp.loggamma(j+1)+mp.log(mp.gammainc(a+j, 0, x/2, regularized=True))
            logs.append(term)
            largest = max(logs)
            total_log = largest+mp.log(mp.fsum(mp.exp(value-largest) for value in logs))
            ratio = mean*x/(2*(j+1)*(a+j+1))
            if ratio < 1 and term+mp.log(ratio/(1-ratio)) < total_log+mp.log(mp.mpf("1e-70")):
                break
        else:
            raise ArithmeticError("conditional Poisson remainder did not converge")
        log_tail = total_log-mean
    else:
        log_tail = parallel_integral(x, dof, lam, upper)
    log_cdf = mp.log1p(-mp.exp(log_tail)) if upper else log_tail
    return mp.nstr(log_cdf, 30), mp.nstr(log_tail, 30), "upper" if upper else "lower"


def cell(key):
    dof, x, lam = key
    value, tail, side = reference(float(x), dof, float(lam))
    return key, {"dof": dof, "x": x, "lambda": lam, "log_cdf": value,
                 "log_small_tail_or_bound": tail, "tail": side}


def main():
    old = json.loads(DEST.read_text()) if DEST.exists() else {"cases": []}
    cache = {(row["dof"], row["x"], row["lambda"]): row for row in old["cases"]}
    for row in cache.values():
        row["log_cdf"] = mp.nstr(mp.mpf(row["log_cdf"]), 30)
        row["log_small_tail_or_bound"] = mp.nstr(mp.mpf(row["log_small_tail_or_bound"]), 30)
        row.pop("quadrature_log_tail_error_bound", None)
    cells = set()
    for dof in DOFS:
        for lam in LAMBDAS:
            xs = {1e-40, 1e-30, 1e-20, 1e-10, 1e-6, 0.01, 1.0, 100.0, 1e3, 1e6, 1e9, 1e12,
                  1e15, 5e19, 1e20, 1e100, float.fromhex('0x1.fffffffffffffp+1023')}
            mean, sigma = dof+lam, math.sqrt(2*dof+4*lam)
            xs.update(mean+z*sigma for z in (-38, -20, -8, -2, -0.01, 0, 0.01, 2, 8, 20, 38)
                      if 1e-40 <= mean+z*sigma <= 1e12)
            xs.update((math.nextafter(mean, 0.0), mean, math.nextafter(mean, math.inf)))
            # Central-gamma series / continued-fraction boundary.
            if lam == 0:
                xs.update((math.nextafter(dof+2.0, 0.0), dof+2.0,
                           math.nextafter(dof+2.0, math.inf)))
            if dof > 1 and lam > 100:
                # The perpendicular-radius integration cutoff reaches sqrt(x).
                cutoff = (math.sqrt(dof-1)+40.0)**2
                xs.update((math.nextafter(cutoff, 0.0), cutoff,
                           math.nextafter(cutoff, math.inf)))
                scaled_cutoff = cutoff*cutoff/lam
                if scaled_cutoff < lam:
                    xs.update((math.nextafter(scaled_cutoff, 0.0), scaled_cutoff,
                               math.nextafter(scaled_cutoff, math.inf)))
            if dof == 1:
                # Straddle the folded-interval series switch r*max(d,1)=.01.
                switch = 1e-4/max(lam, 1.0)
                xs.update((math.nextafter(switch, 0.0), switch, math.nextafter(switch, math.inf)))
                xs.update((math.nextafter(lam, 0.0), lam, math.nextafter(lam, math.inf)))
                # The normal log-CDF's Mills-ratio boundary, r-d = -10.
                if lam > 100:
                    normal_switch = (math.sqrt(lam)-10.0)**2
                    xs.update((math.nextafter(normal_switch, 0.0), normal_switch,
                               math.nextafter(normal_switch, math.inf)))
            for x in xs:
                if 1e-40 <= x <= 1e12 or x in (1e15, 5e19, 1e20, 1e100,
                                              float.fromhex('0x1.fffffffffffffp+1023')):
                    cells.add((dof, repr(x), repr(lam)))
    # These folded-normal values replace a self-referential erfc expectation.
    cells.update((1, "100.0", repr(lam)) for lam in (0.0, 0.01, 0.1, 1.0))
    # A central two-dof survival probability at the smallest binary64 value,
    # and its adjacent x operands; the abscissa is derived, not a literal root.
    subnormal_tail_x = float(2*1074*mp.log(2))
    cells.update((2, repr(x), "0.0") for x in
                 (math.nextafter(subnormal_tail_x, 0.0), subnormal_tail_x,
                  math.nextafter(subnormal_tail_x, math.inf)))
    # Resolve sums of subnormal weighted terms, not only a central gamma tail.
    # The target is ten least subnormals; solve its survival equation directly
    # with the positive high-precision Poisson mixture.
    tiny_tail = 10*mp.mpf(2)**-1074
    for dof, lam in ((2, 50.0), (8, 100.0), (100, 100.0)):
        guess = (mp.sqrt(lam)+mp.sqrt(-2*mp.log(tiny_tail)))**2+dof
        root = mp.findroot(lambda x: mixture(x, dof, mp.mpf(lam), True)-mp.log(tiny_tail),
                           (guess-20, guess), tol=mp.mpf("1e-60"))
        x = float(root)
        cells.update((dof, repr(value), repr(lam)) for value in
                     (math.nextafter(x, 0.0), x, math.nextafter(x, math.inf)))
    # Straddle the perpendicular-degree panel-count boundary.
    for dof in (16385, 16386):
        mean = dof+150.0
        cells.update((dof, repr(x), "150.0") for x in
                     (1e-40, math.nextafter(mean, 0.0), mean,
                      math.nextafter(mean, math.inf)))
    gamma_checks = []
    for a in (mp.mpf("1000.5"), mp.mpf("49999.5")):
        for standardized in (9, 40):
            z = a+standardized*mp.sqrt(a)
            direct = mp.gammainc(a, z, mp.inf, regularized=True)
            difference = abs(large_shape_upper_gamma(a, z)/direct-1)
            assert difference < mp.mpf("1e-44")
            gamma_checks.append({"shape": mp.nstr(a, 20), "argument": mp.nstr(z, 50),
                                 "relative_difference_from_mpmath_gammainc": mp.nstr(difference, 25)})
    document = {"provenance": {
        "generator": "fixtures-generators/generate_noncentral_grid.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precision_decimal_digits": 90,
        "integration_precision_decimal_digits": 50,
        "case_count": len(cells),
        "large_shape_gamma": "DLMF 8.9.2 evaluated with mpmath at 70 digits, backward propagation of the odd-tail interval [1-a/z,1], relative interval width <1e-45; https://dlmf.nist.gov/8.9.E2",
        "large_shape_gamma_checks": gamma_checks,
        "mpmath_version": mp.__version__,
        "method": "positive incomplete-gamma Poisson mixture or adaptive parallel-normal conditioning integral; direct smaller tail; Chernoff certificate for binary64 upper-tail underflow",
        "reference_relative_error": "Poisson remainder < 1e-70 of the smaller tail; adaptive integration estimated relative error < 1e-35 at 50 digits (large-shape gamma fraction at 70 digits); algebra outside integration at 90 digits; logarithms serialized to 30 significant digits"
    }, "cases": []}
    keys = sorted(cells, key=lambda key: (key[0], float(key[2]), float(key[1])))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cell, key) for key in keys if key not in cache]
        for future in as_completed(futures):
            key, row = future.result()
            cache[key] = row
            document["cases"] = [cache[key] for key in keys if key in cache]
            DEST.write_text(json.dumps(document, indent=2)+"\n")
            print(len(cache), len(cells), key, flush=True)
    document["cases"] = [cache[key] for key in keys]
    DEST.write_text(json.dumps(document, indent=2)+"\n")


if __name__ == "__main__":
    main()
