"""Exact-rational reference for the civil day-fraction producers.

Each case is a civil clock label (hour, minute, second as decimal text). The
expected day fraction is the double nearest to the exact rational
(hour * 3600 + minute * 60 + second) / 86400, ties to even, computed with
fractions.Fraction and float(), which rounds a Fraction correctly. The pre-3.0.0
arithmetic, (hour * 3600.0 + minute * 60.0 + second) / 86400.0 in doubles, is
recorded beside it (fraction_bits_2x) so the cases where rounding once moves
the result are counted.

Cases: random labels with 0 to 15 fractional second digits, whole-second
labels, and labels constructed within 1e-15 s of the seconds at which the day
fraction is exactly halfway between two doubles (the cases a double rounding
gets wrong most often).

Run: python3 generate_day_fraction_exact.py > ../tests/fixtures/time/day_fraction_exact.json
"""

import json
import math
import random
import struct
import sys
from decimal import Decimal
from fractions import Fraction

random.seed(20260923)


def exact_fraction(hour, minute, second_text):
    return (Fraction(hour * 3600 + minute * 60) + Fraction(Decimal(second_text))) / 86400


def bits(value):
    return "%016x" % struct.unpack(">Q", struct.pack(">d", value))[0]


def case(hour, minute, second_text):
    # The label is the shortest decimal of its double (Python's repr), the
    # reading the producers take of an f64 second.
    value = float(second_text)
    text = repr(value)
    if "e" in text:
        return None
    expected = float(exact_fraction(hour, minute, text))
    old = (hour * 3600.0 + minute * 60.0 + value) / 86400.0
    return [hour, minute, text, bits(expected), bits(old)]


cases = []
for _ in range(1500):
    digits = random.randint(0, 15)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    whole = random.randint(0, 59)
    if digits == 0:
        text = "%d" % whole
    else:
        text = "%d.%0*d" % (whole, digits, random.randint(0, 10**digits - 1))
    c = case(hour, minute, text)
    if c:
        cases.append(c)

# Near halfway cases.
for _ in range(1500):
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    base = hour * 3600 + minute * 60
    target = Fraction(base, 86400) + Fraction(random.randint(0, 59 * 10**6), 86400 * 10**6)
    f = float(target)
    up = math.nextafter(f, 2.0)
    mid = (Fraction(f) + Fraction(up)) / 2
    seconds = mid * 86400 - base
    if not (0 <= seconds < 60):
        continue
    scaled = seconds * 10**15
    for n in (math.floor(scaled), math.ceil(scaled)):
        text = str(Decimal(n) / Decimal(10**15))
        c = case(hour, minute, text)
        if c:
            cases.append(c)

json.dump(
    {
        "schema": "sidereon-core/day_fraction_exact.v1",
        "reference": "fractions.Fraction exact rational, float() correctly rounded",
        "python": sys.version.split()[0],
        "columns": ["hour", "minute", "second", "fraction_bits", "fraction_bits_2x"],
        "cases": cases,
    },
    sys.stdout,
    separators=(",", ":"),
)
sys.stdout.write("\n")
