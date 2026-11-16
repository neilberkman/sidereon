"""Generate a single-gauge least-squares reference using 80-digit arithmetic.

Run with Python and mpmath; the output defaults to the adjacent core fixtures.
The code/phase design measures two ambiguity differences with unequal noise.
Changing the gauge preserves both differences and their covariance.
"""
import json
import hashlib
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80
# Dyadic sigmas make every information entry exactly representable, isolating
# gauge handling from conditioning in decimal-to-binary input conversion.
rows = [([1, -1, 0], "0.70", "0.125"),
        ([1, 0, -1], "1.20", "0.25"),
        ([0, 1, -1], "0.50", "0.125"),
        ([1, 0, 0], "0.08", "0.0001220703125")]
h = mp.matrix([row for row, _, _ in rows])
w = mp.diag([1 / mp.mpf(sigma)**2 for _, _, sigma in rows])
y = mp.matrix([mp.mpf(value) for _, value, _ in rows])
information = h.T * w * h
covariance = information**-1
estimate = covariance * h.T * w * y
contrast = mp.matrix([[1, -1, 0]])
out = {
    "provenance": {"generator": "fixtures-generators/generate_rtk_reference_gauge.py",
                   "precision_decimal_digits": 80,
                   "method": "weighted least squares with a single gauge direction",
                   "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    "information": [float(v) for v in information],
    "estimate": [float(v) for v in estimate],
    "dd": float((contrast * estimate)[0]),
    "dd_variance": float((contrast * covariance * contrast.T)[0]),
    "gauge_information": float(1 / mp.mpf(rows[-1][2])**2),
    "fixed_cycles": {"G02": -7, "G03": -12},
    "referred_fixed_cycles": {"G01": 7, "G03": -5},
}
fixture_dir = Path(__file__).resolve().parents[1] / "tests/fixtures"
arc = json.loads((fixture_dir/"rtklib_rtk_mask_oracle.json").read_text())
out["provenance"]["input_sha256"] = hashlib.sha256(
    (fixture_dir/"rtklib_rtk_mask_oracle.json").read_bytes()).hexdigest()
dimension = 9
normal = mp.diag([mp.mpf("0.0001")]*3+[mp.mpf("0.000001")]*6)
first = arc["epochs"][1]
second = arc["epochs"][2]
first_reference = 0  # configured G01, which is unmasked in the first epoch
second_reference = int(second["reference"][1:])-1
for epoch_index, epoch, reference in ((1, first, first_reference), (2, second, second_reference)):
    rover = [mp.mpf(value) for value in arc["rover_ecef_m"][epoch_index]]
    unit_vectors = []
    for position in epoch["satellite_ecef_m"]:
        delta = mp.matrix([mp.mpf(value)-receiver for value, receiver in zip(position, rover)])
        unit_vectors.append(delta/mp.norm(delta))
    others = [int(id[1:])-1 for id in epoch["used"] if int(id[1:])-1 != reference]
    for phase, sigma in ((False, mp.mpf(float(0.3))), (True, mp.mpf(float(0.003)))):
        design = mp.matrix(len(others), dimension)
        for row, satellite in enumerate(others):
            for axis in range(3):
                design[row, axis] = unit_vectors[reference][axis]-unit_vectors[satellite][axis]
            if phase:
                design[row, 3+satellite] = 1
                design[row, 3+reference] = -1
        noise = 2*sigma**2*(mp.eye(len(others))+mp.ones(len(others)))
        normal += design.T*(noise**-1)*design
# The four first-epoch fixes enter the second float solve as DD hold rows.
# Keep one gauge direction (G01); reinforcing it at epoch two is harmless.
hold_weight = mp.mpf("1e8")
normal[3, 3] += 2*hold_weight
held_satellites = {int(id[1:])-1 for id in first["used"]}-{first_reference}
assert second_reference in held_satellites
translated_holds = (held_satellites-{second_reference})|{first_reference}
for satellite in sorted(translated_holds):
    row = mp.matrix(1, dimension)
    row[0, 3+satellite], row[0, 3+second_reference] = 1, -1
    normal += hold_weight*row.T*row
covariance = normal**-1
contrast = mp.matrix(1, dimension)
contrast[0, 3+first_reference], contrast[0, 3+second_reference] = 1, -1
norm_one = lambda matrix: max(mp.fsum(abs(matrix[i,j]) for i in range(matrix.rows))
                              for j in range(matrix.cols))
condition = norm_one(normal)*norm_one(covariance)
wavelength = mp.mpf(float(299792458.0/1575420000.0))
out["moving_arc"] = {
    "epochs": [1, 2], "reference_ids": ["G01", second["reference"]],
    "dd_m": float((first_reference-second_reference)*wavelength),
    "dd_variance_m2": float((contrast*covariance*contrast.T)[0]),
    "normal_condition_one": float(condition),
    "matrix_dimension": dimension,
    "first_epoch_held_ids": [f"G{satellite+1:02d}" for satellite in sorted(held_satellites)],
    "translated_held_cycles": {f"G{satellite+1:02d}": satellite-second_reference
                               for satellite in sorted(translated_holds)},
    "roundoff_position_scale_m": max(abs(value) for epoch in (first, second)
                                      for position in epoch["satellite_ecef_m"] for value in position),
}
# Newly fixed constraints must survive a change to a reference not yet fixed.
# Form a complete weighted least-squares problem, then change coordinates on
# its posterior. No sequential-filter recurrence is used for the reference.
pending_cases = []
for count in (1, 2, 3):
    n = 8
    design = mp.eye(n)
    values = mp.matrix([mp.mpf('0.003'), mp.mpf('-0.002'), mp.mpf('0.001'),
                        0, mp.mpf('0.58'), mp.mpf('1.01'), mp.mpf('0.35'), mp.mpf('-0.1')])
    weights = mp.diag([mp.mpf('100')]*3+[1]*5)
    normal = design.T*weights*design
    rhs = design.T*weights*values
    for epoch in range(count):
        for satellite in range(1, 5):
            geometry = [mp.mpf(satellite)/5, mp.mpf(satellite*satellite)/25,
                        (-1)**satellite*mp.mpf('0.4')]
            for phase, sigma in ((False, mp.mpf('0.03')), (True, mp.mpf('0.003'))):
                h = mp.matrix(1, n)
                for axis in range(3):
                    h[0, axis] = geometry[axis]
                if phase:
                    h[0, 3+satellite], h[0, 3] = 1, -1
                target = (mp.mpf('0.19')*[0, 3, 5, 2, -1][satellite]+mp.mpf('0.05')) if phase else mp.mpf('0.002')*(satellite-2)
                normal += h.T*h/sigma**2
                rhs += h.T*target/sigma**2
    before = normal**-1*rhs
    before_normal = normal.copy()
    holds = {"G02": 3, "G03": 5}
    sigma = mp.mpf('0.0001')
    for id, integer in holds.items():
        h = mp.matrix(1, n)
        h[0, 3+int(id[1:])-1], h[0, 3] = 1, -1
        normal += h.T*h/sigma**2
        rhs += h.T*(mp.mpf('0.19')*integer+mp.mpf('0.05'))/sigma**2
    covariance = normal**-1
    after = covariance*rhs
    # New reference G04: the old gauge is transported, not a second pin.
    transform = mp.eye(n)
    for row in range(3, n):
        transform[row, 3] += 1
        transform[row, 6] -= 1
    cov = transform*covariance*transform.T
    estimate = transform*after
    pending_cases.append({"information": list(map(float, before_normal)),
        "estimate": list(map(float, before)), "fixed_cycles": holds,
        "fixed_m": {id: float(mp.mpf('0.19')*integer+mp.mpf('0.05')) for id, integer in holds.items()},
        "hold_sigma_m": float(sigma), "baseline_sigma_m": float(mp.sqrt(cov[0,0])),
        "dd_m": float(estimate[4]-estimate[6]),
        "dd_variance_m2": float(cov[4,4]+cov[6,6]-2*cov[4,6]),
        "condition_one": float(max(norm_one(normal)*norm_one(covariance),
                                   norm_one(cov**-1)*norm_one(cov),
                                   norm_one(before_normal)*norm_one(before_normal**-1))),
        "state_scale_m": float(max(1, *map(abs, before), *map(abs, estimate))),
        "dimension": n})
out["pending_holds"] = pending_cases
destination = Path(__file__).resolve().parents[1] / "tests/fixtures/rtk_reference_gauge.json"
destination.write_text(json.dumps(out, indent=2) + "\n")
