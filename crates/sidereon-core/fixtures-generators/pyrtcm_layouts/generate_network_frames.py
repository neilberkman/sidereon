"""Write RTCM 3 frames of the system, text, network RTK and transformation
messages from pyrtcm's message layouts, as a reference independent of
sidereon's codec.

For each of 1013-1017, 1021-1027, 1029-1032, 1034, 1035 and 1037-1039 this
script walks pyrtcm's payload definition (RTCM_PAYLOADS_GET) and data-field
table (RTCM_DATA_FIELDS), gives every field a value drawn from a fixed-seed
generator over the field's whole range, writes the fields in pyrtcm's order
and widths, and frames the body with its CRC-24Q. It checks that pyrtcm reads
every frame back to the values written, then writes:

- tests/fixtures/rtcm/network/pyrtcm_network_rtk.rtcm3: the frames;
- tests/fixtures/rtcm/network/pyrtcm_network_rtk.json: for each frame, the
  message number and every field after DF002 in wire order as
  [data field, raw integer];
- tests/fixtures/rtcm/families/text_1029.rtcm3: 1029 frames whose character
  count (DF138) states the characters of their UTF-8 text, ASCII and
  multi-byte, for the RTKLIB oracle (generate.sh in rtklib_rtcm_oracle).

usage: python3 generate_network_frames.py   (pyrtcm==1.2.0, see requirements)
"""

import json
import os
import random

from pyrtcm import RTCMReader
from pyrtcm.rtcmtypes_core import RTCM_DATA_FIELDS, INT, INTS, CHA, STR
from pyrtcm.rtcmtypes_get import RTCM_PAYLOADS_GET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "tests", "fixtures", "rtcm", "network")
MESSAGES = [1013, 1014, 1015, 1016, 1017, 1021, 1022, 1023, 1024, 1025, 1026,
            1027, 1029, 1030, 1031, 1032, 1034, 1035, 1037, 1038, 1039]
# Counts of repeating groups: records and name characters, in range.
COUNTS = {"DF067": (0, 15), "DF234": (0, 15), "DF006": (0, 31), "DF035": (0, 31),
          "DF143": (0, 31), "DF145": (0, 31), "DF053": (0, 31), "DF139": (0, 255)}
TEXTS = ["sidereon", "BKG NTRIP caster test, 12:00 UTC", "Zürich Hbf", "Kraków ↔ Łódź",
         "", "日本語のテキスト"]


def crc24q(data):
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    return crc & 0xFFFFFF


def draw(rng, name, repeat):
    """A raw value for data field `name` over its whole range; the extremes
    are drawn on some repeats."""
    typ, width, _, _ = RTCM_DATA_FIELDS[name]
    if name in COUNTS:
        low, high = COUNTS[name]
        return rng.randint(low, min(high, 5) if repeat < 3 else high)
    if typ in (CHA, STR):
        return rng.randint(0, 255)
    if typ in (INT, INTS):
        low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if typ == INTS:
            low = -high
    else:
        low, high = 0, (1 << width) - 1
    edge = rng.random()
    if edge < 0.1:
        return low
    if edge < 0.2:
        return high
    return rng.randint(low, high)


def fields(rng, payload, repeat, values):
    """Walk a pyrtcm payload definition, drawing values; yields
    (field, raw) in wire order."""
    for key, item in payload.items():
        if key.startswith("group"):
            count_ref, group = item
            count = count_ref if isinstance(count_ref, int) else values[count_ref]
            for _ in range(count):
                yield from fields(rng, group, repeat, values)
            continue
        if key == "DF002":
            continue
        value = draw(rng, key, repeat)
        values[key] = value
        yield key, value


def body(number, entries):
    bits = [((number >> (11 - i)) & 1) for i in range(12)]
    for name, value in entries:
        typ, width, _, _ = RTCM_DATA_FIELDS[name]
        if typ == INTS:
            raw = ((1 << (width - 1)) if value < 0 else 0) | abs(value)
        else:
            raw = value & ((1 << width) - 1)
        bits += [(raw >> (width - 1 - i)) & 1 for i in range(width)]
    bits += [0] * (-len(bits) % 8)
    return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


def frame(payload):
    head = bytes([0xD3, (len(payload) >> 8) & 0x03, len(payload) & 0xFF]) + payload
    return head + crc24q(head).to_bytes(3, "big")


def main():
    rng = random.Random(20260924)
    stream = b""
    records = []
    for repeat in range(4):
        for number in MESSAGES:
            values = {}
            entries = list(fields(rng, RTCM_PAYLOADS_GET[str(number)], repeat, values))
            data = frame(body(number, entries))
            parsed = RTCMReader.parse(data, labelmsm=1)
            assert parsed.identity == str(number), (number, parsed.identity)
            read_back = [(k, v) for k, v in vars(parsed).items()
                         if k.startswith("DF") and k != "DF002"]
            if number == 1029:
                # pyrtcm joins the code units into one DF140 string, a zero
                # byte adding nothing.
                units = [raw for name, raw in entries if name == "DF140"]
                assert vars(parsed).get("DF140", "") == "".join(chr(u) for u in units if u), number
                entries_checked = [e for e in entries if e[0] != "DF140"]
                read_back = [e for e in read_back if e[0] != "DF140"]
            else:
                entries_checked = entries
            assert len(read_back) == len(entries_checked), number
            for (name, raw), (attr, value) in zip(entries_checked, read_back):
                assert attr.split("_")[0] == name, (number, attr, name)
                _, _, scale, _ = RTCM_DATA_FIELDS[name]
                expected = chr(raw) if RTCM_DATA_FIELDS[name][0] == CHA else (
                    raw * scale if scale not in (0, 1) else raw)
                assert value == expected, (number, name, raw, value)
            stream += data
            records.append({"type": number, "fields": entries})
    with open(os.path.join(OUT, "pyrtcm_network_rtk.rtcm3"), "wb") as fp:
        fp.write(stream)
    with open(os.path.join(OUT, "pyrtcm_network_rtk.json"), "w") as fp:
        json.dump({"generator": "fixtures-generators/pyrtcm_layouts/generate_network_frames.py",
                   "pyrtcm": "1.2.0", "frames": records}, fp)
        fp.write("\n")
    print(f"wrote {len(records)} frames, {len(stream)} bytes")
    text = b""
    for index, message in enumerate(TEXTS):
        units = message.encode("utf-8")
        entries = [("DF003", 17 + index), ("DF051", 61000 + index), ("DF052", 3600 * index),
                   ("DF138", len(message)), ("DF139", len(units))]
        entries += [("DF140", u) for u in units]
        text += frame(body(1029, entries))
    with open(os.path.join(HERE, "..", "..", "tests", "fixtures", "rtcm", "families",
                           "text_1029.rtcm3"), "wb") as fp:
        fp.write(text)
    print(f"wrote {len(TEXTS)} text frames")


if __name__ == "__main__":
    main()
