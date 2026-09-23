"""Generate sgp4_tsince.json: the minutes since epoch python-sgp4 forms from a
split Julian date, at Skyfield's splits for instants around UTC noon.

Requires skyfield 1.54 with its built-in timescale and sgp4 2.22. The element
set is the ISS TLE fixture (../omm/25544.tle) read by python-sgp4's pure
Python `sgp4.model.Satrec.twoline2rv`. For each instant, given as integer Unix
microseconds, the generator takes Skyfield's split (`t.whole`,
`t.tai_fraction - t._leap_seconds() / DAY_S`, as
`EarthSatellite._position_and_velocity_TEME_km` passes it) and records the
minutes since epoch `Satrec.sgp4(jd, fr)` forms from it,
`(jd - jdsatepoch) * 1440 + (fr - jdsatepochF) * 1440`, evaluated by
`sgp4.model.Satrec.sgp4`'s own Python expression, with `jdsatepoch` and
`jdsatepochF`. Every double is written as hexadecimal bits.

Instants: every second from 60 s before to 10 s after UTC noon on days from
1975 to 2045 (TAI - UTC from 14 s to 37 s), where the fraction turns negative,
and random instants from 1960 to 2059.

Run from this directory: python gen_sgp4_tsince.py
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random
import struct

import sgp4
import skyfield
from sgp4 import model
from skyfield.api import load
from skyfield.constants import DAY_S

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "sgp4_tsince.json"
SEED = 20260926
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
NOON_DAYS = ["1975-03-14", "1990-08-01", "2003-12-31", "2018-07-03", "2045-06-15"]


def bits(value):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def unix_us(moment):
    delta = moment - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def instants(rng):
    out = []
    for day in NOON_DAYS:
        noon = datetime.fromisoformat(day + "T12:00:00+00:00")
        for offset in range(-60, 11):
            out.append(unix_us(noon) + offset * 1_000_000)
    start = datetime(1960, 1, 1, tzinfo=timezone.utc)
    span_us = 100 * 365 * 86_400 * 1_000_000
    out += [unix_us(start) + rng.randrange(span_us) for _ in range(300)]
    return out


def main():
    if sgp4.__version__ != "2.22" or skyfield.__version__ != "1.54":
        raise SystemExit("needs sgp4 2.22 and skyfield 1.54")
    ts = load.timescale(builtin=True)
    lines = [
        line
        for line in (HERE.parent / "omm" / "25544.tle").read_bytes().decode("ascii").splitlines()
        if line.startswith(("1 ", "2 "))
    ]
    satrec = model.Satrec.twoline2rv(lines[0], lines[1])
    rows = []
    for micros in instants(random.Random(SEED)):
        t = ts.from_datetime(EPOCH + timedelta(microseconds=micros))
        jd = float(t.whole)
        fr = float(t.tai_fraction - t._leap_seconds() / DAY_S)
        tsince = ((jd - satrec.jdsatepoch) * model.minutes_per_day +
                  (fr - satrec.jdsatepochF) * model.minutes_per_day)
        rows.append([micros, bits(jd), bits(fr), bits(tsince)])
    OUTPUT.write_text(
        '{"sgp4": "%s", "skyfield": "%s", "jdsatepoch": "%s", "jdsatepochF": "%s", '
        '"instants": [\n%s\n]}\n'
        % (
            sgp4.__version__,
            skyfield.__version__,
            bits(satrec.jdsatepoch),
            bits(satrec.jdsatepochF),
            ",\n".join(json.dumps(row) for row in rows),
        )
    )


if __name__ == "__main__":
    main()
