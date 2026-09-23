"""Generate skyfield_sgp4_times.json: the split Julian date Skyfield hands
SGP4 for UTC instants.

Requires skyfield 1.54 with its built-in timescale. For each instant, given
as integer Unix microseconds, the generator builds `ts.from_datetime(datetime)`
and records the two numbers `EarthSatellite._position_and_velocity_TEME_km`
passes to `Satrec.sgp4`, `t.whole` and
`t.tai_fraction - t._leap_seconds() / DAY_S`, as hexadecimal bits. Those are
SGP4's only time inputs; the propagated state is not recorded, since the
compiled SGP4 extension's arithmetic depends on the platform it was built for.

Instants: random microseconds from 1960 through 2059, a band of seconds
either side of UTC noon (where TAI has passed noon and UTC has not), the
last seconds before and the first after 2016-12-31's leap second, and
2017-01-01 00:00:00.

Run from this directory: python gen_skyfield_sgp4_times.py
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random
import struct

import skyfield
from skyfield.api import load
from skyfield.constants import DAY_S

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "skyfield_sgp4_times.json"
RANDOM_INSTANTS = 3_000
SEED = 20260925
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def bits(value):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def unix_us(moment):
    delta = moment - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def instants(rng):
    out = []
    start = datetime(1960, 1, 1, tzinfo=timezone.utc)
    span_us = 100 * 365 * 86_400 * 1_000_000
    for _ in range(RANDOM_INSTANTS):
        out.append(unix_us(start) + rng.randrange(span_us))
    for _ in range(300):
        day = start + timedelta(days=rng.randrange(100 * 365))
        noon = day.replace(hour=12, minute=0, second=0, microsecond=0)
        out.append(unix_us(noon) + rng.randrange(-60_000_000, 60_000_000))
    leap = unix_us(datetime(2017, 1, 1, tzinfo=timezone.utc))
    out += [leap + d for d in (-2_000_000, -1_000_000, -1, 0, 1, 999_999)]
    return out


def main():
    if skyfield.__version__ != "1.54":
        raise SystemExit("needs skyfield 1.54")
    ts = load.timescale(builtin=True)
    rows = []
    for micros in instants(random.Random(SEED)):
        moment = EPOCH + timedelta(microseconds=micros)
        t = ts.from_datetime(moment)
        whole = t.whole
        fraction = t.tai_fraction - t._leap_seconds() / DAY_S
        rows.append([micros, bits(whole), bits(fraction)])
    OUTPUT.write_text(
        '{"skyfield": "%s", "instants": [\n%s\n]}\n'
        % (skyfield.__version__, ",\n".join(json.dumps(row) for row in rows))
    )


if __name__ == "__main__":
    main()
