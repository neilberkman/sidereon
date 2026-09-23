"""Generate python_sgp4_epochs.json: the SGP4 split Julian date python-sgp4
gives OMM epochs.

Requires sgp4 2.22 with its compiled extension (sgp4.api.accelerated), the
Satrec Skyfield 1.54 builds in EarthSatellite.from_omm. For each EPOCH the
fixture records `jdsatepoch` and `jdsatepochF` of a Satrec initialised by
sgp4.omm.initialize, as the hexadecimal bits of each double.

Epochs: the three committed CelesTrak OMM fixtures, then random epochs from
year 1 to 9999 with one to six fractional second digits (python-sgp4 reads
EPOCH with '%Y-%m-%dT%H:%M:%S.%f', which needs at least one), weighted toward
the satellite era and toward values whose day count has at most eight
decimals, where sgp4init rounds the fraction.

Run from this directory: python gen_python_sgp4_epochs.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import random
import struct

import sgp4
from sgp4 import omm
from sgp4.api import Satrec, accelerated

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "python_sgp4_epochs.json"
FIXTURES = ["24876.kvn", "25544.kvn", "28884.kvn"]
RANDOM_EPOCHS = 5_000
SEED = 20260923


def bits(value):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", value))[0]


def kvn_fields(path):
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def template():
    fields = kvn_fields(HERE / "25544.kvn")
    fields.setdefault("CLASSIFICATION_TYPE", "U")
    fields.setdefault("EPHEMERIS_TYPE", "0")
    return fields


def epoch_text(moment, digits):
    text = "%04d-%02d-%02dT%02d:%02d:%02d" % (
        moment.year, moment.month, moment.day,
        moment.hour, moment.minute, moment.second,
    )
    return text + "." + ("%06d" % moment.microsecond)[:digits]


def random_epochs(rng):
    epochs = []
    for _ in range(RANDOM_EPOCHS):
        if rng.random() < 0.7:
            start, end = datetime(1957, 1, 1), datetime(2100, 1, 1)
        else:
            start, end = datetime(1, 1, 1), datetime(9999, 12, 31)
        span = int((end - start).total_seconds())
        moment = start + timedelta(seconds=rng.randrange(span))
        digits = rng.randrange(1, 7)
        if rng.random() < 0.2:
            # Whole multiples of 0.864 s give a day count with at most eight
            # decimals.
            moment = moment.replace(second=0) + timedelta(
                microseconds=864_000 * rng.randrange(70)
            )
            digits = 6
        else:
            micro = rng.randrange(10**6)
            micro -= micro % (10 ** (6 - digits))
            moment = moment.replace(microsecond=micro)
        epochs.append(epoch_text(moment, digits))
    return epochs


def main():
    if sgp4.__version__ != "2.22" or not accelerated:
        raise SystemExit("needs sgp4 2.22 with its compiled extension")
    fields = template()
    epochs = [kvn_fields(HERE / name)["EPOCH"] for name in FIXTURES]
    epochs += random_epochs(random.Random(SEED))
    rows = []
    for epoch in epochs:
        fields["EPOCH"] = epoch
        satrec = Satrec()
        omm.initialize(satrec, fields)
        rows.append([epoch, bits(satrec.jdsatepoch), bits(satrec.jdsatepochF)])
    lines = ",\n".join(json.dumps(row) for row in rows)
    OUTPUT.write_text(
        '{"sgp4": "%s", "epochs": [\n%s\n]}\n' % (sgp4.__version__, lines)
    )


if __name__ == "__main__":
    main()
