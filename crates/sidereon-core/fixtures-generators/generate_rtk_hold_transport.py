"""Reference-relative RTK ambiguities and geometric observations at 80 digits.

The elevations and reference decisions come from the RTKLIB mask fixture.
Ranges, carrier observations and translated integer/offset pairs are formed
independently with mpmath, without executing the positioning library.
"""
import hashlib
import json
import copy
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80
root = Path(__file__).resolve().parents[1]
source = root / "tests/fixtures/rtklib_rtk_mask_oracle.json"
mask = json.loads(source.read_text())
base = list(map(mp.mpf, mask["base_ecef_m"]))
cycles = [0, 2, 6, 5, 8, 9]
f1, f2, speed = mp.mpf(1575420000), mp.mpf(1227600000), mp.mpf(299792458)
scenarios = []
for offset in (0.05, 0.11):
    wavelength = mp.mpf(float(0.19))
    offsets = [mp.mpf(0)] + [mp.mpf(offset)]*5
    epochs = []
    for index in (1, 2, 1):
        row = mask["epochs"][index]
        rover = list(map(mp.mpf, mask["rover_ecef_m"][index]))
        reference = "G01" if "G01" not in row["rover_masked"] else row["reference"]
        origin = int(reference[1:])-1
        observations = []
        for satellite, position in enumerate(row["satellite_ecef_m"]):
            position = list(map(mp.mpf, position))
            rb = mp.sqrt(mp.fsum((x-y)**2 for x,y in zip(position, base)))
            rr = mp.sqrt(mp.fsum((x-y)**2 for x,y in zip(position, rover)))
            n1, n2 = 10+cycles[satellite], 5+2*cycles[satellite]
            observations.append({"id": f"G{satellite+1:02d}",
                "base_code": float(rb), "rover_code": float(rr),
                "base_phase": float(rb),
                "rover_phase": float(rr+wavelength*cycles[satellite]+offsets[satellite]),
                "base_phi1": float(rb*f1/speed), "base_phi2": float(rb*f2/speed),
                "rover_phi1": float(rr*f1/speed+n1), "rover_phi2": float(rr*f2/speed+n2)})
        epochs.append({"rover": list(map(float, rover)), "positions": row["satellite_ecef_m"],
            "reference": reference, "observations": observations,
            "expected": {f"G{i+1:02d}": {"cycles": cycles[i]-cycles[origin],
                "offset_m": float(offsets[i]-offsets[origin]),
                "metres": float(wavelength*(cycles[i]-cycles[origin])+offsets[i]-offsets[origin])}
                for i in range(6) if i != origin}})
    scenarios.append({"offset_m": offset, "wavelength_m": float(wavelength), "epochs": epochs})
fdma = copy.deepcopy(scenarios[0])
fdma["wavelengths_m"] = {f"G{i+1:02d}": 0.24 if i==3 else 0.19 for i in range(6)}
for epoch in fdma["epochs"]:
    for i, observation in enumerate(epoch["observations"]):
        position = epoch["positions"][i]
        radius = mp.sqrt(mp.fsum((mp.mpf(x)-mp.mpf(y))**2 for x,y in zip(position, epoch["rover"])))
        observation["rover_phase"] = float(radius+mp.mpf(fdma["wavelengths_m"][observation["id"]])*cycles[i]
                                             + (mp.mpf(0.05) if i else 0))
out = {"provenance": {"generator": "fixtures-generators/generate_rtk_hold_transport.py",
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "geometry_reference": "rtklib_rtk_mask_oracle.json",
    "geometry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "precision_decimal_digits": 80,
    "method": "Euclidean ranges and exact integer subtraction; reference offset is zero"},
    "base_m": list(map(float, base)), "mask_deg": mask["mask_deg"], "scenarios": scenarios,
    "narrow_lane_wavelength_m": float(speed/(f1+f2)),
    "fdma_scenario": fdma}
(root / "tests/fixtures/rtk_hold_transport.json").write_text(json.dumps(out, indent=2)+"\n")
