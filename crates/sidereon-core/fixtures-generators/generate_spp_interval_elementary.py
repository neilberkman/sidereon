"""Generate directed 100-digit Decimal references for bounded SPP intervals."""

import decimal
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


LOWER = decimal.Context(prec=100, rounding=decimal.ROUND_FLOOR)
UPPER = decimal.Context(prec=100, rounding=decimal.ROUND_CEILING)
NEAREST = decimal.Context(prec=100, rounding=decimal.ROUND_HALF_EVEN)
TERMS = 80


class Enclosure:
    def __init__(self, lower, upper):
        assert lower.is_finite() and upper.is_finite() and lower <= upper
        self.lower = lower
        self.upper = upper

    @classmethod
    def integer(cls, value):
        exact = decimal.Decimal(value)
        return cls(exact, exact)

    def add(self, other):
        return Enclosure(
            LOWER.add(self.lower, other.lower),
            UPPER.add(self.upper, other.upper),
        )

    def subtract(self, other):
        return Enclosure(
            LOWER.subtract(self.lower, other.upper),
            UPPER.subtract(self.upper, other.lower),
        )

    def multiply(self, other):
        pairs = [
            (left, right)
            for left in (self.lower, self.upper)
            for right in (other.lower, other.upper)
        ]
        return Enclosure(
            min(LOWER.multiply(left, right) for left, right in pairs),
            max(UPPER.multiply(left, right) for left, right in pairs),
        )

    def divide(self, other):
        assert not other.lower <= 0 <= other.upper
        pairs = [
            (left, right)
            for left in (self.lower, self.upper)
            for right in (other.lower, other.upper)
        ]
        return Enclosure(
            min(LOWER.divide(left, right) for left, right in pairs),
            max(UPPER.divide(left, right) for left, right in pairs),
        )


def trigonometric(argument, sine):
    exact = decimal.Decimal.from_float(argument)
    interval = Enclosure(exact, exact)
    squared = interval.multiply(interval)
    term = interval if sine else Enclosure.integer(1)
    total = Enclosure.integer(0)
    for index in range(TERMS):
        total = total.add(term)
        first = 2 * index + (2 if sine else 1)
        denominator = Enclosure.integer(first * (first + 1))
        term = term.multiply(squared).divide(denominator)
        term = term.multiply(Enclosure.integer(-1))
    first_omitted = max(term.lower.copy_abs(), term.upper.copy_abs())
    next_factor = 2 * TERMS + (2 if sine else 1)
    ratio = Enclosure.integer(64).divide(
        Enclosure.integer(next_factor * (next_factor + 1))
    )
    remainder = Enclosure(first_omitted, first_omitted).divide(
        Enclosure.integer(1).subtract(ratio)
    ).upper
    return total.add(Enclosure(remainder.copy_negate(), remainder))


def reference(function, argument):
    if function in ("sin", "cos"):
        return trigonometric(argument, function == "sin")
    exact = decimal.Decimal.from_float(argument)
    if function == "exp":
        result = NEAREST.exp(exact)
    elif function == "ln":
        result = NEAREST.ln(exact)
    else:
        raise ValueError(function)
    return Enclosure(LOWER.next_minus(result), UPPER.next_plus(result))


def bits(value):
    return "0x" + struct.pack(">d", value).hex()


def generate():
    cases = []
    arguments = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    for function in ("sin", "cos", "exp", "ln"):
        domain = [0.5, 1.0, 1.5, 2.0] if function == "ln" else arguments
        for argument in domain:
            enclosure = reference(function, argument)
            lower = math.nextafter(float(enclosure.lower), -math.inf)
            upper = math.nextafter(float(enclosure.upper), math.inf)
            cases.append(
                {
                    "function": function,
                    "argument_bits": bits(argument),
                    "lower_bits": bits(lower),
                    "upper_bits": bits(upper),
                }
            )
    return {
        "decimal_precision": NEAREST.prec,
        "trigonometric_terms": TERMS,
        "trigonometric_remainder": "directed geometric bound after first omitted term",
        "exp_log_reference": "correctly rounded Decimal operations, enclosed by adjacent Decimal values",
        "binary64_conversion": "one adjacent value outward after nearest conversion",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }


if __name__ == "__main__":
    json.dump(generate(), sys.stdout, indent=2)
    sys.stdout.write("\n")
