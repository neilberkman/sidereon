#!/usr/bin/env python3
"""Generate Kalman-gate and IGG-III references.

Kalman equations follow FilterPy's update/mahalanobis definitions. IGG-III is
Yang, Song, and Xu's published weight: 1; k0/a*((k1-a)/(k1-k0))**2; 0.
"""
import hashlib,json
from math import sqrt
from struct import pack
from pathlib import Path
import numpy as np, scipy
from scipy.stats import chi2
from rust_libm_port import log
HERE=Path(__file__).resolve(); OUT=HERE.parent.parent/"tests/fixtures/fusion_gate_oracles.json"
k0,k1=2.,5.
def weight(a):
    if a<=k0:return 1.
    if a<=k1:return (k0/a)*((k1-a)/(k1-k0))**2
    return 0.
points=[np.nextafter(k0,0),k0,np.nextafter(k0,np.inf),3.,4.,4.952790301466122,np.nextafter(k1,0),k1,np.nextafter(k1,np.inf),10.]
grid=np.linspace(k0,k1,301)

def dot(left, right):
    total = 0.0
    for a, b in zip(left, right, strict=True):
        total += a * b
    return total


def transpose(matrix):
    return [list(column) for column in zip(*matrix, strict=True)]


def multiply(left, right):
    return [[dot(row, column) for column in transpose(right)] for row in left]


def add(left, right):
    return [[x + y for x, y in zip(a, b, strict=True)]
            for a, b in zip(left, right, strict=True)]


def hex_bits(value):
    return "0x" + pack(">d", value).hex()


def cholesky(matrix):
    # Factor by lower-triangular row, subtracting each product in index order.
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i, row in enumerate(matrix):
        for j in range(i + 1):
            value = row[j]
            for k in range(j):
                value -= lower[i][k] * lower[j][k]
            lower[i][j] = sqrt(value) if i == j else value / lower[j][j]
    return lower


def solve_cholesky(lower, rhs):
    n = len(rhs)
    forward = [0.0] * n
    solution = [0.0] * n
    for i, value in enumerate(rhs):
        for k in range(i):
            value -= lower[i][k] * forward[k]
        forward[i] = value / lower[i][i]
    for i in reversed(range(n)):
        value = forward[i]
        for k in range(i + 1, n):
            value -= lower[k][i] * solution[k]
        solution[i] = value / lower[i][i]
    return solution


def nis(innovation, covariance):
    return dot(innovation, solve_cholesky(cholesky(covariance), innovation))


def unscented_nis(p, h, r, innovation):
    # Scaled transform with alpha=0.5, beta=2, kappa=0. Scale the factor
    # columns and visit the centre, positive columns, then negative columns.
    n = len(p)
    alpha, beta = 0.5, 2.0
    lam = alpha * alpha * float(n) - float(n)
    scale = float(n) + lam
    gamma = sqrt(scale)
    columns = transpose(cholesky(p))
    points = [[0.0] * n] + [[sign * entry for entry in column]
                                for sign in (gamma, -gamma) for column in columns]
    values = [[dot(row, point) for row in h] for point in points]
    weights = [lam / scale] + [0.5 / scale] * (2 * n)
    mean = [dot(weights, column) for column in transpose(values)]
    weights[0] += 1.0 - alpha * alpha + beta

    # Add each weighted residual product to R from left to right in the
    # same point order, then average the two triangles.
    covariance = [row.copy() for row in r]
    for weight, value in zip(weights, values, strict=True):
        residual = [a - b for a, b in zip(value, mean, strict=True)]
        for i, a in enumerate(residual):
            for j, b in enumerate(residual):
                covariance[i][j] += weight * a * b
    for i, row in enumerate(covariance):
        for j in range(i + 1, len(covariance)):
            row[j] = covariance[j][i] = 0.5 * (row[j] + covariance[j][i])
    return nis([a - b for a, b in zip(innovation, mean, strict=True)], covariance)


def correlated_gate():
    # Both marginals pass a 3-sigma gate, while anticorrelation increases NIS.
    n = 15
    p = [[float(i == j) for j in range(n)] for i in range(n)]
    p[0][:2] = [0.5, -0.45]
    p[1][:2] = [-0.45, 0.5]
    h = [[float(i == j) for j in range(n)] for i in range(2)]
    r = [[0.5, -0.45], [-0.45, 0.5]]
    innovation = [2.5, 2.5]
    s = add(multiply(multiply(h, p), transpose(h)), r)
    probability = 0.01
    # For two degrees of freedom, the survival function is exp(-x / 2).
    threshold = -2.0 * log(probability)
    ekf_d2 = nis(innovation, s)
    return {
        "innovation_bits": [hex_bits(value) for value in innovation],
        "p_block_bits": [[hex_bits(value) for value in row[:2]] for row in p[:2]],
        "r_bits": [[hex_bits(value) for value in row] for row in r],
        "row_gate_pass": all(abs(value) / sqrt(s[i][i]) <= 3.0
                             for i, value in enumerate(innovation)),
        "joint_ekf_nis_bits": hex_bits(ekf_d2),
        "joint_ukf_nis_bits": hex_bits(unscented_nis(p, h, r, innovation)),
        "joint_false_alarm_probability_bits": hex_bits(probability),
        "joint_threshold_bits": hex_bits(threshold),
        "joint_pass": ekf_d2 <= threshold,
    }


def gated_joseph_update():
    p = [[2.0, 0.0], [0.0, 3.0]]
    h = [[1.0, 0.0], [0.5, 1.0]]
    r = [[1.0, 0.2], [0.2, 2.0]]
    innovation = [1.25, -4.0]
    gate = 1.5
    innovation_covariance = add(multiply(multiply(h, p), transpose(h)), r)
    normalized = [abs(value) / sqrt(innovation_covariance[i][i])
                  for i, value in enumerate(innovation)]
    kept = [i for i, value in enumerate(normalized) if value <= gate]
    [row] = kept
    hk = [h[row]]
    rk = [[r[row][row]]]
    sk = [[innovation_covariance[row][row]]]
    # The scalar Cholesky solve divides by the factor in each substitution.
    root = sqrt(sk[0][0])
    gain = [[(entry / root) / root] for [entry] in multiply(p, transpose(hk))]
    correction = [dot(gain_row, [innovation[row]]) for gain_row in gain]
    identity_minus_kh = [[float(i == j) - value for j, value in enumerate(row)]
                         for i, row in enumerate(multiply(gain, hk))]
    stabilized = multiply(multiply(identity_minus_kh, p), transpose(identity_minus_kh))
    noise = multiply(multiply(gain, rk), transpose(gain))
    posterior = add(stabilized, noise)
    nominal_ecef_m = [6378137.0, 0.0]
    return {
        "P": p, "H": h, "R": r, "innovation": innovation, "threshold": gate,
        "normalized": normalized, "accepted": kept,
        "nominal_position_ecef_m_bits": [
            hex_bits(value - delta) for value, delta in zip(nominal_ecef_m, correction, strict=True)
        ],
        "covariance_bits": [[hex_bits(value) for value in row] for row in posterior],
        "S": sk, "K": gain,
    }


igg_scan = np.concatenate([np.linspace(4.9, 5.1, 41), [10.]])
mixed_v = [0., 0., 0., .2, .2, .2]
mixed_p = [1.0] * 3 + [0.001] * 3
mixed_r = [99.0] * 3 + [0.009] * 3
mixed_variances = [p + r for p, r in zip(mixed_p, mixed_r, strict=True)]
mixed_standardized = [value * value / variance
                      for value, variance in zip(mixed_v, mixed_variances, strict=True)]
# The velocity-group trace statistic sets the predicted covariance scale.
velocity_statistic = dot(mixed_v[3:], mixed_v[3:]) / dot(mixed_variances[3:], [1.0] * 3)
scaled_p = velocity_statistic * mixed_p[3]
r_velocity = mixed_r[3]
velocity_factor = cholesky([[scaled_p + r_velocity]])
[gain] = solve_cholesky(velocity_factor, [scaled_p])
stabilizer = 1.0 - gain
posterior_velocity_variance = (stabilizer * scaled_p) * stabilizer + (gain * r_velocity) * gain

raw_gate_innovation = [value * sqrt(2.0) for value in [6.0, 4.0, 6.0, 6.0, 6.0, 6.0]]
raw_gate_s = np.diag([2.] * 6)
data = {
    "provenance": {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "generator_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
        "references": [
            "FilterPy KalmanFilter.update and mahalanobis equations",
            "Yang, Song, Xu IGG III weight function",
            "Exponential survival function for chi-square with two degrees of freedom",
            "SciPy chi-square quantiles",
        ],
    },
    "inputs": {"k0": k0, "k1": k1},
    "weights": [[float(x), float(weight(x))] for x in points],
    "transition_grid": {"points": grid.tolist(), "weights": [float(weight(x)) for x in grid]},
    "applied_scale_scan": {
        "standardized": igg_scan.tolist(), "oracle_weights": [float(weight(x)) for x in igg_scan],
    },
    "raw_gate_after_igg_removal": {
        "innovation_bits": [hex_bits(value) for value in raw_gate_innovation],
        "S": raw_gate_s.tolist(),
        "raw_standardized": [abs(value) / sqrt(2.0) for value in raw_gate_innovation],
        "igg_retained_rows": [1],
        "retained_nis_bits": hex_bits(nis([raw_gate_innovation[1]], [[2.0]])),
        "gate_threshold_sigma": 3.,
    },
    "yang_mixed_units": {
        "innovation_bits": [hex_bits(value) for value in mixed_v],
        "predicted_variance_bits": [hex_bits(value) for value in mixed_p],
        "measurement_variance_bits": [hex_bits(value) for value in mixed_r],
        "full_trace_statistic_bits": hex_bits(dot(mixed_v, mixed_v) / dot(mixed_variances, [1.] * 6)),
        "unit_consistent_mean_squared_standardized": dot(mixed_standardized, [1.] * 6) / 6.,
        "velocity_block_statistic_bits": hex_bits(velocity_statistic),
        "posterior_velocity_variance_bits": hex_bits(posterior_velocity_variance),
    },
    "yang_chi2_0p99": [float(chi2.ppf(.99, n)) for n in range(1, 13)],
    "correlated_verdict": correlated_gate(),
    "update": gated_joseph_update(),
}
OUT.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
