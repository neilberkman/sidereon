"""Generate python_sgp4_omm_init.json: what python-sgp4 hands SGP4 for OMMs.

Requires sgp4 2.22 with its compiled extension (sgp4.api.accelerated), the
Satrec Skyfield 1.54 builds in EarthSatellite.from_omm. Each case is the text
of an OMM in KVN; the generator reads its fields as sgp4.omm.initialize does,
initialises a Satrec with it and records, as the hexadecimal bits of each
double, the day count since 1949-12-31 `omm.initialize` passes to `sgp4init`
and the element record's `jdsatepoch`, `jdsatepochF`, `no_kozai`, `bstar`,
`ndot`, `nddot`, `ecco`, `argpo`, `inclo`, `mo` and `nodeo`.

These are every input `sgp4init` receives. Propagated states are not
recorded: the compiled extension's states depend on its build (of the 183
states first captured in tests/sgp4_verification.json, sgp4 2.22 on arm64
macOS reproduces 180 and the sgp4 2.25 wheel 17), so what is checked against
python-sgp4 is the element record SGP4 starts from.

Cases: the three committed CelesTrak OMM fixtures, then generated OMMs from
the ISS fixture's layout with random epochs (one to six fractional second
digits) and elements: near-Earth orbits, 12-hour and 24-hour deep-space
orbits (the resonance branches) and Molniya-like orbits of high
eccentricity, with B* stated to between one and nine significant digits.

Run from this directory: python gen_python_sgp4_init.py
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
OUTPUT = HERE / "python_sgp4_omm_init.json"
FIXTURES = ["24876.kvn", "25544.kvn", "28884.kvn"]
GENERATED = 200
SEED = 20260924
EPOCH0 = datetime(1949, 12, 31)
FIELDS = ["jdsatepoch", "jdsatepochF", "no_kozai", "bstar", "ndot", "nddot",
          "ecco", "argpo", "inclo", "mo", "nodeo"]


def bits(value):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", value))[0]


def kvn_fields(text):
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def decimal(rng, low, high, digits):
    return "%.*f" % (digits, rng.uniform(low, high))


def generated_case(rng, template):
    start = datetime(1960, 1, 1)
    moment = start + timedelta(seconds=rng.randrange(95 * 365 * 86400))
    digits = rng.randrange(1, 7)
    micro = rng.randrange(10**6)
    micro -= micro % (10 ** (6 - digits))
    epoch = "%04d-%02d-%02dT%02d:%02d:%02d.%s" % (
        moment.year, moment.month, moment.day,
        moment.hour, moment.minute, moment.second,
        ("%06d" % micro)[:digits],
    )
    kind = rng.choice(["near", "near", "semi", "geo", "molniya"])
    if kind == "near":
        motion = decimal(rng, 11.0, 16.4, 8)
        ecc = decimal(rng, 0.0, 0.05, 7)
    elif kind == "semi":
        motion = decimal(rng, 1.9, 2.1, 8)
        ecc = decimal(rng, 0.0, 0.05, 7)
    elif kind == "geo":
        motion = decimal(rng, 0.98, 1.02, 8)
        ecc = decimal(rng, 0.0, 0.01, 7)
    else:
        motion = decimal(rng, 2.0, 2.02, 8)
        ecc = decimal(rng, 0.65, 0.75, 7)
    bstar_digits = rng.randrange(1, 10)
    bstar = "%.*E" % (bstar_digits - 1, rng.uniform(-5e-4, 5e-4))
    values = {
        "EPOCH": epoch,
        "MEAN_MOTION": motion,
        "ECCENTRICITY": ecc,
        "INCLINATION": decimal(rng, 0.0, 110.0, 4),
        "RA_OF_ASC_NODE": decimal(rng, 0.0, 360.0, 4),
        "ARG_OF_PERICENTER": decimal(rng, 0.0, 360.0, 4),
        "MEAN_ANOMALY": decimal(rng, 0.0, 360.0, 4),
        "BSTAR": bstar,
        "MEAN_MOTION_DOT": "%.8f" % rng.uniform(-1e-5, 1e-4),
    }
    lines = []
    for line in template.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in values:
            lines.append("%-14s = %s" % (key, values[key]))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def inputs(text):
    fields = kvn_fields(text)
    satrec = Satrec()
    omm.initialize(satrec, fields)
    # The day count `omm.initialize` computes and passes to `sgp4init`.
    moment = datetime.strptime(fields["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")
    epoch = (moment - EPOCH0).total_seconds() / 86400.0
    return [bits(epoch)] + [bits(getattr(satrec, name)) for name in FIELDS]


def main():
    if sgp4.__version__ != "2.22" or not accelerated:
        raise SystemExit("needs sgp4 2.22 with its compiled extension")
    rng = random.Random(SEED)
    template = (HERE / "25544.kvn").read_text()
    texts = [(HERE / name).read_bytes().decode("ascii") for name in FIXTURES]
    texts += [generated_case(rng, template) for _ in range(GENERATED)]
    cases = [[text] + inputs(text) for text in texts]
    OUTPUT.write_text(
        '{"sgp4": "%s", "cases": [\n%s\n]}\n'
        % (sgp4.__version__, ",\n".join(json.dumps(case) for case in cases))
    )


if __name__ == "__main__":
    main()
