"""Generate skyfield_passes.json: Skyfield 1.54's view of three satellites
from two ground stations, for the pass and look-angle APIs.

Requires skyfield 1.54 and sgp4 2.22 (its compiled extension) with the
built-in timescale, which applies no polar motion.

Satellites, each an `EarthSatellite(line1, line2)` (python-sgp4's
`Satrec.twoline2rv`, opsmode 'i'):
  25544  the ISS element set of 2018-07-03 the crate's SGP4 tests use;
  08195  a 12-hour Molniya orbit of Vallado's verification set;
  23599  a 6.9-degree, e = 0.58 deep-space orbit of the same set, where
         opsmode 'a' and 'i' put the satellite up to 1.03 km apart
         within a day of epoch, so a pass computed in the wrong mode moves.
Stations: London (51.5074, -0.1278, 11 m) and Kampala (0.3136, 32.5811,
1190 m), each `wgs84.latlon`, and London at 80 m for the two ISS cases of the
crate's older tests below.

`epochs`: for each satellite and station, 30 instants 397 s apart from
17.123456 s past the first whole minute after the element epoch, each built
with `ts.from_datetime`. Each records
  * the TEME state `EarthSatellite._position_and_velocity_TEME_km` returns,
    with the per-component bound on how far a build of SGP4 that calls a
    different libm can move it (`fixtures-generators/sgp4_libm_bound.py`,
    at the minutes since epoch python-sgp4 forms from Skyfield's split);
  * `(sat - station).at(t).altaz()`: azimuth, elevation, range;
  * `wgs84.geographic_position_of(sat.at(t))`: latitude, longitude, height;
  * Skyfield's GCRS-to-TEME rotation (`sgp4lib.TEME.rotation_at`), its
    GCRS-to-ITRS rotation (`framelib.itrs.rotation_at`) and the station's
    ITRS position.

The ISS rows also hold the ten whole minutes from 2018-07-03 19:30 from
London at 80 m, the arc of tests/sgp4_topocentric_arc.rs.

`passes`: for each satellite and station, the passes in one day from the
element epoch's first whole hour, at a 0-degree and a 10-degree mask, and
the ISS from London at 80 m from 2018-07-03 12:00 to 07-04 12:00 at a
0-degree mask, the window of tests/pass_finder_arc.rs.
Skyfield's `find_events` brackets each event; the generator then refines it
on the Skyfield elevation (`altaz`, no shortcuts) at whole microseconds: a
rise or set by bisection to the two microseconds either side of the mask
crossing and linear interpolation between them, a culmination by bisection
on the sign of the elevation rate (a central difference of the elevation
10 ms either side). A pass that
crests twice (the Molniya from Kampala) takes the higher crest as its
culmination, and records how many crests Skyfield reports. Each event
records its time in Unix seconds, the elevation rate there (a crossing) or
the elevation's second and third derivatives (a culmination, from the rate
there and one second either side), the range, and the TEME position bound at
that instant.

Each event also keeps the time `find_events` itself reported
(`skyfield_unix_s`), which its search places within half a second of the
event. Two more ISS windows from London start and end mid-pass: a pass
already up at the window start has no rise, is marked `clamped`, and records
Skyfield's elevation, its rate, the range and the TEME bound at the start; a
pass still up at the window end is dropped, as `find_events` gives it no
set.

`visible`: every satellite from each station at four instants, as
`visible_from_constellation` lists them: the look angle and TEME position
of each satellite, with the TEME bound, or the message python-sgp4 gives
where it returns an error code.

Run from this directory: python gen_skyfield_passes.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import sgp4
import skyfield
from sgp4.api import WGS72, Satrec
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.constants import DAY_S
from skyfield.framelib import itrs
from skyfield.sgp4lib import TEME

HERE = Path(__file__).resolve().parent
GENERATORS = HERE.parent.parent.parent / "fixtures-generators"
sys.path.insert(0, str(GENERATORS))
from generate_sgp4_verification import TrackedSatellite  # noqa: E402

OUTPUT = HERE / "skyfield_passes.json"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

SATELLITES = {
    "25544": (
        "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
        "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
    ),
    "08195": (
        "1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813",
        "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656",
    ),
    "23599": (
        "1 23599U 95029B   06171.76535463  .00085586  12891-6  12956-2 0  2905",
        "2 23599   6.9327   0.2849 5782022 274.4436  25.2425  4.47796565123555",
    ),
}
STATIONS = {
    "london": (51.5074, -0.1278, 11.0),
    "kampala": (0.3136, 32.5811, 1190.0),
    "london_80m": (51.5074, -0.1278, 80.0),
}
MAIN_STATIONS = ["london", "kampala"]
EPOCH_SAMPLES = 30
EPOCH_STEP_US = 397_000_000
EPOCH_OFFSET_US = 17_123_456
MASKS = [0.0, 10.0]
EXTRA_EPOCHS = [
    ("25544", "london_80m", datetime(2018, 7, 3, 19, 30 + i, tzinfo=timezone.utc)) for i in range(10)
]
# Windows cut mid-pass, as fractions of the ISS's refined passes from London
# at a 0-degree mask: (pass index, fraction from rise to set) for the start
# and the end.
PARTIAL_WINDOWS = [((2, 0.25), (4, 0.5)), ((2, 0.75), (3, 0.02))]
EXTRA_WINDOWS = [
    (
        "25544",
        "london_80m",
        0.0,
        datetime(2018, 7, 3, 12, tzinfo=timezone.utc),
        datetime(2018, 7, 4, 12, tzinfo=timezone.utc),
    )
]
VISIBLE_INSTANTS = [
    datetime(2006, 6, 25, 9, 0, 0, 250_000, tzinfo=timezone.utc),
    datetime(2006, 6, 25, 15, 30, 0, tzinfo=timezone.utc),
    datetime(2006, 6, 21, 3, 17, 42, 5, tzinfo=timezone.utc),
    datetime(2006, 6, 26, 0, 0, 0, tzinfo=timezone.utc),
]


def hexf(x: float) -> str:
    return float.hex(float(x))


def unix_us(moment: datetime) -> int:
    delta = moment - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def at_us(ts, micros: int):
    return ts.from_datetime(EPOCH + timedelta(microseconds=micros))


class Satellite:
    def __init__(self, ts, name: str, lines):
        self.name = name
        self.lines = lines
        self.earth = EarthSatellite(*lines, ts=ts)
        self.satrec = Satrec.twoline2rv(*lines, WGS72)
        self.tracked = None

    def tsince(self, t) -> float:
        # python-sgp4's Satrec.sgp4(jd, fr) at Skyfield's split.
        jd = t.whole
        fr = t.tai_fraction - t._leap_seconds() / DAY_S
        return (jd - self.satrec.jdsatepoch) * 1440.0 + (fr - self.satrec.jdsatepochF) * 1440.0

    def teme(self, t):
        r, v, message = self.earth._position_and_velocity_TEME_km(t)
        return r, v, message

    def teme_bound(self, t):
        if self.tracked is None:
            self.tracked = TrackedSatellite(self.satrec)
        return self.tracked.bounds(self.tsince(t))


def elevation_deg(sat: Satellite, topo, t) -> float:
    return float((sat.earth - topo).at(t).altaz()[0].degrees)


RATE_HALF_STEP_US = 10_000


def elevation_rate_deg_s(ts, sat: Satellite, topo, micros: int) -> float:
    """Central difference of the altaz elevation, 10 ms either side.

    `frame_latlon_and_rates` is not used: its altitude rate differs from
    the derivative of `altaz` elevation by about 2e-7 deg/s (at the ISS's
    culminations), which moves a culmination by a millisecond for the ISS
    and by seconds for a Molniya's flat crest.
    """
    before = elevation_deg(sat, topo, at_us(ts, micros - RATE_HALF_STEP_US))
    after = elevation_deg(sat, topo, at_us(ts, micros + RATE_HALF_STEP_US))
    return (after - before) / (2.0 * RATE_HALF_STEP_US / 1e6)


def bracket(f, guess_us, same_side):
    """Widen [guess - w, guess + w] until `same_side` says the ends differ."""
    width = 2_000_000
    while width <= 3_600_000_000:
        lo_us, hi_us = guess_us - width, guess_us + width
        flo, fhi = f(lo_us), f(hi_us)
        if not same_side(flo, fhi):
            return lo_us, hi_us, flo, fhi
        width *= 2
    raise RuntimeError(f"no bracket near {guess_us}")


def refine_crossing(ts, sat, topo, mask, guess_us):
    """Bisect `elevation - mask` on whole microseconds, then interpolate."""
    f = lambda us: elevation_deg(sat, topo, at_us(ts, us)) - mask
    lo_us, hi_us, flo, fhi = bracket(f, guess_us, lambda a, b: (a < 0.0) == (b < 0.0))
    while hi_us - lo_us > 1:
        mid = (lo_us + hi_us) // 2
        fm = f(mid)
        if (fm < 0.0) == (flo < 0.0):
            lo_us, flo = mid, fm
        else:
            hi_us, fhi = mid, fm
    return lo_us + (0.0 - flo) / (fhi - flo)


def refine_culmination(ts, sat, topo, guess_us):
    """Bisect the sign of the elevation rate on whole microseconds."""
    g = lambda us: elevation_rate_deg_s(ts, sat, topo, us)
    lo_us, hi_us, glo, ghi = bracket(g, guess_us, lambda a, b: not (a > 0.0 > b))
    while hi_us - lo_us > 1:
        mid = (lo_us + hi_us) // 2
        gm = g(mid)
        if gm > 0.0:
            lo_us, glo = mid, gm
        else:
            hi_us, ghi = mid, gm
    return lo_us + glo / (glo - ghi)


def teme_position_bound_km(sat: Satellite, t) -> float:
    b = sat.teme_bound(t)
    return math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)


def unix_s(t) -> float:
    moment = t.utc_datetime()
    return unix_us(moment) / 1e6


def event_record(ts, sat, topo, kind, time_us_float, skyfield_unix_s):
    micros = int(round(time_us_float))
    t = at_us(ts, micros)
    alt, _, distance = (sat.earth - topo).at(t).altaz()
    row = {
        "kind": kind,
        "unix_s": time_us_float / 1e6,
        "skyfield_unix_s": skyfield_unix_s,
        "elevation_deg": float(alt.degrees),
        "range_km": float(distance.km),
        "teme_position_bound_km": teme_position_bound_km(sat, t),
    }
    if kind == "culmination":
        before = elevation_rate_deg_s(ts, sat, topo, micros - 1_000_000)
        at = elevation_rate_deg_s(ts, sat, topo, micros)
        after = elevation_rate_deg_s(ts, sat, topo, micros + 1_000_000)
        row["elevation_second_derivative_deg_s2"] = (after - before) / 2.0
        row["elevation_third_derivative_deg_s3"] = after - 2.0 * at + before
    else:
        row["elevation_rate_deg_s"] = elevation_rate_deg_s(ts, sat, topo, micros)
    return row


def passes_for(ts, sat, topo, mask, start_us, end_us):
    """The passes `find_events` reports in the window, refined.

    A pass already above the mask at the window start (no rise) is kept with
    `"clamped": true` and no rise; its culmination is the highest crest
    Skyfield reports after the start, or None when the elevation only falls
    from the start. A pass still up at the window end (no set) is dropped.
    Each event keeps the time `find_events` itself gave, `skyfield_unix_s`.
    """
    t0, t1 = at_us(ts, start_us), at_us(ts, end_us)
    times, events = sat.earth.find_events(topo, t0, t1, altitude_degrees=mask)
    out = []
    current = None
    if elevation_deg(sat, topo, t0) >= mask:
        current = {"rise": None, "crests": []}
    for t, event in zip(times, events):
        guess = unix_us(t.utc_datetime())
        if event == 0:
            current = {"rise": (refine_crossing(ts, sat, topo, mask, guess), unix_s(t)), "crests": []}
        elif event == 1 and current is not None:
            # A long pass can crest twice; the pass's culmination is the
            # higher crest.
            current["crests"].append((refine_culmination(ts, sat, topo, guess), unix_s(t)))
        elif event == 2 and current is not None:
            crest = max(
                current["crests"],
                key=lambda c: elevation_deg(sat, topo, at_us(ts, int(round(c[0])))),
                default=None,
            )
            set_us = refine_crossing(ts, sat, topo, mask, guess)
            row = {
                "rise": None
                if current["rise"] is None
                else event_record(ts, sat, topo, "rise", *current["rise"]),
                "culmination": None
                if crest is None
                else event_record(ts, sat, topo, "culmination", *crest),
                "set": event_record(ts, sat, topo, "set", set_us, unix_s(t)),
                "crests": len(current["crests"]),
            }
            if current["rise"] is None:
                row["clamped"] = True
                alt, _, distance = (sat.earth - topo).at(t0).altaz()
                row["start"] = {
                    "elevation_deg": float(alt.degrees),
                    "elevation_rate_deg_s": elevation_rate_deg_s(ts, sat, topo, start_us),
                    "range_km": float(distance.km),
                    "teme_position_bound_km": teme_position_bound_km(sat, t0),
                }
            out.append(row)
            current = None
    return out


def matrix_hex(m) -> list:
    return [[hexf(x) for x in row] for row in np.asarray(m)]


def epoch_row(ts, sat, station, topo, micros):
    t = at_us(ts, micros)
    r, v, message = sat.teme(t)
    assert message is None, (sat.name, micros, message)
    alt, az, distance = (sat.earth - topo).at(t).altaz()
    geo = wgs84.geographic_position_of(sat.earth.at(t))
    return {
        "satellite": sat.name,
        "station": station,
        "unix_us": micros,
        "teme_position_km": [hexf(x) for x in r],
        "teme_velocity_km_s": [hexf(x) for x in v],
        "teme_bound": sat.teme_bound(t),
        "azimuth_deg": hexf(az.degrees),
        "elevation_deg": hexf(alt.degrees),
        "range_km": hexf(distance.km),
        "latitude_rad": hexf(geo.latitude.radians),
        "longitude_rad": hexf(geo.longitude.radians),
        "height_m": hexf(geo.elevation.m),
        "gcrs_to_teme": matrix_hex(TEME.rotation_at(t)),
        "gcrs_to_itrs": matrix_hex(itrs.rotation_at(t)),
        "station_itrs_km": [hexf(x) for x in topo.itrs_xyz.km],
    }


def pass_case(name, station, mask, start_us, end_us, found):
    return {
        "satellite": name,
        "station": station,
        "mask_deg": mask,
        "start_unix_us": start_us,
        "end_unix_us": end_us,
        "passes": found,
    }


def main():
    if skyfield.__version__ != "1.54" or sgp4.__version__ != "2.22":
        raise SystemExit("needs skyfield 1.54 and sgp4 2.22")
    ts = load.timescale(builtin=True)
    sats = {name: Satellite(ts, name, lines) for name, lines in SATELLITES.items()}
    topos = {name: wgs84.latlon(*site) for name, site in STATIONS.items()}

    epochs = []
    for name, sat in sats.items():
        epoch_dt = sat.earth.epoch.utc_datetime().replace(second=0, microsecond=0) + timedelta(minutes=1)
        first_us = unix_us(epoch_dt) + EPOCH_OFFSET_US
        for station in MAIN_STATIONS:
            topo = topos[station]
            for k in range(EPOCH_SAMPLES):
                epochs.append(epoch_row(ts, sat, station, topo, first_us + k * EPOCH_STEP_US))
    for name, station, moment in EXTRA_EPOCHS:
        epochs.append(epoch_row(ts, sats[name], station, topos[station], unix_us(moment)))

    passes = []
    for name, sat in sats.items():
        start_dt = sat.earth.epoch.utc_datetime().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        start_us = unix_us(start_dt)
        end_us = start_us + 86_400_000_000
        for station in MAIN_STATIONS:
            topo = topos[station]
            for mask in MASKS:
                found = passes_for(ts, sat, topo, mask, start_us, end_us)
                passes.append(pass_case(name, station, mask, start_us, end_us, found))
    full = next(
        c for c in passes if c["satellite"] == "25544" and c["station"] == "london" and c["mask_deg"] == 0.0
    )["passes"]
    for (first, f0), (last, f1) in PARTIAL_WINDOWS:
        def at_fraction(index, fraction):
            rise = full[index]["rise"]["unix_s"]
            set_ = full[index]["set"]["unix_s"]
            return int(round((rise + fraction * (set_ - rise)) * 1e6))

        start_us, end_us = at_fraction(first, f0), at_fraction(last, f1)
        found = passes_for(ts, sats["25544"], topos["london"], 0.0, start_us, end_us)
        passes.append(pass_case("25544", "london", 0.0, start_us, end_us, found))
    for name, station, mask, start, end in EXTRA_WINDOWS:
        start_us, end_us = unix_us(start), unix_us(end)
        found = passes_for(ts, sats[name], topos[station], mask, start_us, end_us)
        passes.append(pass_case(name, station, mask, start_us, end_us, found))

    visible = []
    for moment in VISIBLE_INSTANTS:
        micros = unix_us(moment)
        t = at_us(ts, micros)
        for station in MAIN_STATIONS:
            topo = topos[station]
            rows = []
            for name, sat in sats.items():
                r, v, message = sat.teme(t)
                if message is not None:
                    rows.append({"satellite": name, "error": message})
                    continue
                alt, az, distance = (sat.earth - topo).at(t).altaz()
                rows.append(
                    {
                        "satellite": name,
                        "teme_position_km": [hexf(x) for x in r],
                        "teme_bound": sat.teme_bound(t),
                        "azimuth_deg": hexf(az.degrees),
                        "elevation_deg": hexf(alt.degrees),
                        "range_km": hexf(distance.km),
                    }
                )
            visible.append({"unix_us": micros, "station": station, "satellites": rows})

    doc = {
        "generator": "tests/fixtures/skyfield_passes/gen_skyfield_passes.py",
        "skyfield": skyfield.__version__,
        "sgp4": sgp4.__version__,
        "numpy": np.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}, "
        f"{platform.python_implementation()} {platform.python_version()}",
        "timescale": "load.timescale(builtin=True)",
        "satellites": {name: {"line1": l[0], "line2": l[1]} for name, l in SATELLITES.items()},
        "stations": {
            name: {"latitude_deg": s[0], "longitude_deg": s[1], "altitude_m": s[2]}
            for name, s in STATIONS.items()
        },
        "epochs": epochs,
        "passes": passes,
        "visible": visible,
    }
    OUTPUT.write_text(json.dumps(doc, indent=1) + "\n")
    n_passes = sum(len(p["passes"]) for p in passes)
    print(f"wrote {OUTPUT.name}: {len(epochs)} epochs, {n_passes} passes, {len(visible)} visibility lists")


if __name__ == "__main__":
    main()
