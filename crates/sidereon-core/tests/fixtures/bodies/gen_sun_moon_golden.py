"""Generate a sun/moon geocentric ITRS (ECEF) reference for sun_moon_ecef.

Independent oracle: Skyfield + JPL DE440 (high-precision), GEOMETRIC geocentric
positions (no light-time, no aberration) rotated ICRF -> ITRS via Skyfield's
Earth-orientation frame. The analytic sun_moon_ecef (RTKLIB sunmoonpos series +
the crate's IAU 2000A/2006 GCRS->ITRS transform) approximates these to its model
accuracy (sun direction ~arcmin, moon direction ~0.1-0.3 deg), so the Rust test
gates on direction/distance tolerance, not bit-exactness. The value of the
golden is catching frame/units/series-coefficient regressions: a wrong ECI frame
interpretation would show as arcmin-to-degree ITRS errors that grow over the
sampled year.
"""

import json

from skyfield.api import load
from skyfield.framelib import itrs

ts = load.timescale()
eph = load("de440.bsp")
earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]

# Epochs spanning a year (catch precession/nutation/frame drift) plus a couple of
# intra-day samples (catch the Earth-rotation term in the GCRS->ITRS rotation).
samples = [
    (2026, 1, 1, 0, 0, 0),
    (2026, 3, 20, 6, 0, 0),
    (2026, 5, 13, 0, 0, 0),
    (2026, 5, 13, 12, 0, 0),
    (2026, 5, 13, 18, 30, 0),
    (2026, 6, 21, 12, 0, 0),
    (2026, 9, 22, 18, 0, 0),
    (2026, 12, 21, 23, 59, 30),
]

cases = []
for (y, mo, d, h, mi, s) in samples:
    t = ts.utc(y, mo, d, h, mi, s)
    sun_itrs = (sun - earth).at(t).frame_xyz(itrs).m
    moon_itrs = (moon - earth).at(t).frame_xyz(itrs).m
    cases.append(
        {
            "utc": {"year": y, "month": mo, "day": d, "hour": h, "minute": mi, "second": s},
            "sun_itrs_m": [float(v) for v in sun_itrs],
            "moon_itrs_m": [float(v) for v in moon_itrs],
        }
    )

doc = {
    "source": "skyfield + JPL DE440, geometric geocentric (sun-earth)/(moon-earth), frame_xyz(itrs)",
    "note": "GEOMETRIC positions (no light-time/aberration); reference for analytic sun_moon_ecef",
    "cases": cases,
}

print(json.dumps(doc, indent=2))
