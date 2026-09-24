#!/usr/bin/env python3
"""Generate tests/sgp4_verification.json from python-sgp4.

Every state in the fixture is python-sgp4's own output: the compiled
extension (`sgp4.api.Satrec`, Vallado's C++ of 2020-07-13) with WGS72 and
opsmode 'i'. The satellites are the 33 element sets of Vallado's
verification file `SGP4-VER.TLE`, read from the copy python-sgp4 ships. Each
satellite is propagated

* at the times Vallado's verification driver prints (`tcppver.out`): 0, then
  the start/stop/step grid on the element set's second line, ending at the
  first error, exactly as python-sgp4's own `generate_satellite_output` walks
  it, and
* at 0, 120, 360, 720, 1080 and 1440 minutes.

Every time is propagated from a freshly initialised record, as sidereon-core
does, and the error code python-sgp4 returns is recorded with it.

Next to each error-free state the generator records a per-component bound
on how far a build that runs the same operations with a different libm may
drift from these values. It is derived in `sgp4_libm_bound.py` from the
documented accuracy of `sin`, `cos`, `atan2` and `pow` (under 1 ulp in each
library) and from SGP4's own first-order sensitivity to each call, not from
sidereon-core's output.

Each state, error or not, also records under `rust_libm` what python-sgp4's
model computes in the C++'s operation order (`sgp4_vallado_order.py`) with a
statement-for-statement port of the Rust `libm` crate's `sin`, `cos`,
`atan2` and `pow` (`rust_libm_port.py`): the error code, or the state's
bits. Those are the operations sidereon-core's kernel performs, so it must
reproduce them bit for bit; the prediction is built from python-sgp4's
model and the crate's published algorithms, not from sidereon-core's output.

The ISS states at split Julian dates are python-sgp4's `Satrec.sgp4(jd, fr)`.

`far_times` holds each satellite far from epoch, where python-sgp4 still
propagates or returns its own error code: 1e7 to 1e8 minutes either way for
every satellite, and for the orbits without a resonance integrator also 1e9
minutes up to the largest finite double. A resonant orbit steps its
integrator 720 minutes at a time, so python-sgp4 needs about 1.4e6 steps
per 1e9 minutes; those times are left out, and so is every time past 2^63
minutes, where the integrator never ends. The far states of a resonant orbit
carry no bound: summed over the thousands of integrator steps, the bound
overflows to NaN. A far state of any other orbit carries its bound, or none
where a comparison or an angle reduction could go either way; every far
state carries its `rust_libm` expectation.

`drag_free_iss` is the ISS element set with B* = 0 at the far times of an
orbit without resonance and at +-1e100: python-sgp4 returns a finite state
with code 0 up to 1e19 minutes, and from 1e100 code 0 with a NaN state,
which sidereon-core reports as a non-finite output instead.

The build that produced the committed file:
    sgp4 2.22, wheel sgp4-2.22-cp311-cp311-macosx_11_0_arm64 (no fused
    multiply-add instructions in its extension), CPython 3.11, macOS 26 on
    arm64. The extension's SHA-256 is recorded in the file.

Run from crates/sidereon-core with sgp4 2.22 installed:
    python fixtures-generators/generate_sgp4_verification.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import sysconfig
from pathlib import Path

import copy

import sgp4
from sgp4.api import WGS72, Satrec, accelerated
from sgp4.model import Satrec as PureSatrec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rust_libm_port  # noqa: E402
import sgp4_libm_bound as bound  # noqa: E402
from sgp4_vallado_order import vallado_order_module  # noqa: E402

TRACKED = bound.tracked_module()
RUST_LIBM = vallado_order_module(
    "sgp4_rust_libm",
    rust_libm_port.sin,
    rust_libm_port.cos,
    rust_libm_port.atan2,
    rust_libm_port.pow,
)

EXPECTED_SGP4_VERSION = "2.22"
EXPECTED_SGP4_WHEEL = "sgp4-2.22-cp311-cp311-macosx_11_0_arm64"
EXPECTED_SGP4_EXTENSION_SHA256 = "f69b3fd82f2dd6fd99f708466bef3a3798d508840f90c959cf9c7851b9ad5f3b"
EXPECTED_PLATFORM = ("Darwin", "arm64")
EXPECTED_PYTHON = ("CPython", "cpython-311", "cpython-311-darwin")
FAR_TIMES = [1.0e7 + 1.0, -(1.0e7 + 1.0), 2.0e7, 1.0e8, -1.0e8]
FAR_TIMES_WITHOUT_RESONANCE = [
    1.0e9,
    -1.0e9,
    1.0e10,
    1.0e12,
    1.0e15,
    2.0**63,
    1.0e19,
    1.0e100,
    1.0e300,
    -1.0e300,
    sys.float_info.max,
    -sys.float_info.max,
]
FIXED_GRID = [0.0, 120.0, 360.0, 720.0, 1080.0, 1440.0]
LABELS = ["px", "py", "pz", "vx", "vy", "vz"]
ISS_LINE1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
ISS_LINE2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"
DRAG_FREE_ISS_LINE1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  00000-0 0  9999"
DRAG_FREE_ISS_TIMES = FAR_TIMES + FAR_TIMES_WITHOUT_RESONANCE + [-1.0e100]
ISS_SPLITS = [
    ("2018-07-04T00:00:00", 2458303.0, 0.5),
    ("2018-07-04T00:30:00", 2458303.0, 0.520833333333),
    ("2018-07-04T01:00:00", 2458303.0, 0.541666666667),
    ("2018-07-05T00:00:00", 2458304.0, 0.5),
]
OUT = Path(__file__).resolve().parent.parent / "tests" / "sgp4_verification.json"


def hexf(x: float) -> str:
    return float.hex(x)


def verification_sets():
    """(line1, line2, (start, stop, step)) from the SGP4-VER.TLE python-sgp4 ships."""
    path = Path(sgp4.__file__).resolve().parent / "SGP4-VER.TLE"
    lines = iter(path.read_text(encoding="ascii").replace("\r", "").splitlines())
    for line1 in lines:
        if not line1.startswith("1"):
            continue
        line2 = next(lines)
        grid = tuple(float(field) for field in line2[69:].split())
        yield line1[:69], line2[:69], grid


def driver_times(grid, reference: Satrec) -> list[float]:
    """The times python-sgp4's tcppver driver visits, ending at its first error."""
    times = [0.0]
    error, r, _ = reference.sgp4_tsince(0.0)
    if all(math.isnan(c) for c in r):
        return times
    start, stop, step = grid
    tsince = start
    while tsince <= stop:
        if tsince == start == 0.0:
            tsince += step
            continue
        times.append(tsince)
        error, _, _ = reference.sgp4_tsince(tsince)
        if error != 0:
            break
        tsince += step
    return times


def sgp4init_inputs(reference: Satrec):
    """The inputs python-sgp4's `twoline2rv` hands `sgp4init`."""
    return (
        reference.satnum,
        reference.jdsatepoch + reference.jdsatepochF - 2433281.5,
        reference.bstar,
        reference.ndot,
        reference.nddot,
        reference.ecco,
        reference.argpo,
        reference.inclo,
        reference.mo,
        reference.no_kozai,
        reference.nodeo,
    )


class TrackedSatellite:
    """python-sgp4's model in the C++ order, run on `bound.Tracked` values."""

    def __init__(self, reference: Satrec):
        self.record = PureSatrec()
        bound.BRANCH_FLAGS.clear()
        satnum, *elements = sgp4init_inputs(reference)
        TRACKED.sgp4init(
            TRACKED.getgravconst("wgs72"),
            "i",
            satnum,
            *[bound.Tracked(x) for x in elements],
            self.record,
        )
        self.init_flags = list(bound.BRANCH_FLAGS)

    def bounds(self, tsince: float):
        # One record serves every time, as python-sgp4 itself allows: the
        # resonance integrator resumes from its last step, which repeats the
        # operations a fresh record would make.
        bound.BRANCH_FLAGS.clear()
        r, v = TRACKED.sgp4(self.record, bound.Tracked(tsince))
        flags = self.init_flags + list(bound.BRANCH_FLAGS)
        if flags:
            raise RuntimeError(f"a branch may differ between libm builds: {flags[:3]}")
        return [bound.lift(c).radius() for c in list(r) + list(v)]

    def bounds_where_derivable(self, tsince: float):
        """The bound, or None where a comparison could go either way or the
        bound is not finite."""
        try:
            radii = self.bounds(tsince)
        except (RuntimeError, ArithmeticError, ValueError, OverflowError):
            return None
        return radii if all(math.isfinite(x) for x in radii) else None

    def resonant(self) -> bool:
        return self.record.irez != 0


class RustLibmSatellite:
    """python-sgp4's model in the C++ order with the Rust libm port."""

    def __init__(self, reference: Satrec):
        self.initial = PureSatrec()
        satnum, *elements = sgp4init_inputs(reference)
        RUST_LIBM.sgp4init(RUST_LIBM.getgravconst("wgs72"), "i", satnum, *elements, self.initial)
        self.record = None
        self.last = None

    def expect(self, tsince: float) -> dict:
        """The error code, or the state bits, sidereon-core must give.

        A time further from epoch on the same side resumes the resonance
        integrator from the last one, which repeats the operations a fresh
        record makes; any other time starts from a fresh record.
        """
        resume = (
            self.record is not None
            and self.last is not None
            and tsince * self.last > 0.0
            and abs(tsince) >= abs(self.last)
        )
        if not resume:
            self.record = copy.deepcopy(self.initial)
        self.last = tsince
        r, v = RUST_LIBM.sgp4(self.record, tsince)
        if self.record.error != 0:
            return {"error": self.record.error}
        state_vector = list(r) + list(v)
        out = {label: hexf(value) for label, value in zip(LABELS, state_vector)}
        if not all(math.isfinite(x) for x in state_vector):
            out["non_finite"] = True
        return out


def python_state(reference_lines, tsince: float) -> tuple[dict, list | None]:
    line1, line2 = reference_lines
    reference = Satrec.twoline2rv(line1, line2, WGS72)
    error, r, v = reference.sgp4_tsince(tsince)
    out = {"tsince": tsince}
    if error != 0:
        out["error"] = error
        return out, None
    state_vector = list(r) + list(v)
    for label, value in zip(LABELS, state_vector):
        out[label] = hexf(value)
    return out, state_vector


def state(reference_lines, tsince, tracked, rust) -> dict:
    out, _ = python_state(reference_lines, tsince)
    if "error" not in out:
        out["bound"] = tracked.bounds(tsince)
    out["rust_libm"] = rust.expect(tsince)
    return out


def far_state(reference_lines, tsince, tracked, rust, allow_non_finite=False) -> dict:
    out, state_vector = python_state(reference_lines, tsince)
    if state_vector is not None:
        if all(math.isfinite(x) for x in state_vector):
            out["bound"] = None if tracked.resonant() else tracked.bounds_where_derivable(tsince)
        elif allow_non_finite:
            out["non_finite"] = True
        else:
            raise RuntimeError(f"python-sgp4 returned a non-finite state without an error at {tsince}")
    out["rust_libm"] = rust.expect(tsince)
    return out


def far_order(times):
    """Each side of epoch in increasing distance, so the resonance
    integrator resumes rather than restarts."""
    return sorted(times, key=lambda t: (t < 0.0, abs(t)))


def element_record(reference: Satrec) -> dict:
    return {
        "jdsatepoch": hexf(reference.jdsatepoch),
        "jdsatepochF": hexf(reference.jdsatepochF),
        "bstar": hexf(reference.bstar),
        "ndot": hexf(reference.ndot),
        "nddot": hexf(reference.nddot),
        "ecco": hexf(reference.ecco),
        "argpo": hexf(reference.argpo),
        "inclo": hexf(reference.inclo),
        "mo": hexf(reference.mo),
        "no_kozai": hexf(reference.no_kozai),
        "nodeo": hexf(reference.nodeo),
    }


def main() -> None:
    if sgp4.__version__ != EXPECTED_SGP4_VERSION or not accelerated:
        raise SystemExit(f"needs the compiled extension of sgp4 {EXPECTED_SGP4_VERSION}")
    extension = Path(sgp4.__file__).resolve().parent.glob("vallado_cpp*")
    extension_path = next(extension, None)
    if extension_path is None:
        raise SystemExit("sgp4 reports an accelerated build but its Vallado extension is missing")
    extension_sha256 = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    actual_platform = (platform.system(), platform.machine())
    actual_python = (
        platform.python_implementation(),
        sys.implementation.cache_tag,
        sysconfig.get_config_var("SOABI"),
    )
    if actual_platform != EXPECTED_PLATFORM or actual_python != EXPECTED_PYTHON:
        raise SystemExit(
            "reference build platform/ABI mismatch; refusing to regenerate the pinned oracle.\n"
            f"  expected platform: {EXPECTED_PLATFORM}; got {actual_platform}\n"
            f"  expected Python: {EXPECTED_PYTHON}; got {actual_python}\n"
            f"  observed OS/Python releases: {platform.release()} / {platform.python_version()}\n"
            "To accept a different provenance, deliberately review and update the EXPECTED_* "
            "pins and the fixture metadata together."
        )
    if extension_sha256 != EXPECTED_SGP4_EXTENSION_SHA256:
        raise SystemExit(
            "python-sgp4 extension hash mismatch; refusing to regenerate the pinned oracle.\n"
            f"  expected {EXPECTED_SGP4_EXTENSION_SHA256}; got {extension_sha256} "
            f"({extension_path.name})\n"
            "To accept a different provenance, deliberately review and update the EXPECTED_* "
            "pins and the fixture metadata together."
        )

    satellites = []
    for line1, line2, grid in verification_sets():
        reference = Satrec.twoline2rv(line1, line2, WGS72)
        times = driver_times(grid, reference)
        for t in FIXED_GRID:
            if t not in times:
                times.append(t)
        tracked = TrackedSatellite(reference)
        rust = RustLibmSatellite(reference)
        propagations = [state((line1, line2), t, tracked, rust) for t in times]
        far = TrackedSatellite(reference)
        far_times = FAR_TIMES + ([] if far.resonant() else FAR_TIMES_WITHOUT_RESONANCE)
        far_propagations = [
            far_state((line1, line2), t, far, rust) for t in far_order(far_times)
        ]
        print(line1[2:7], "done", flush=True)
        satellites.append(
            {
                "line1": line1,
                "line2": line2,
                "norad": line1[2:7],
                "verification_grid": list(grid),
                **element_record(reference),
                "propagations": propagations,
                "resonant": far.resonant(),
                "far_times": far_propagations,
            }
        )

    iss = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2, WGS72)
    iss_tracked = TrackedSatellite(iss)
    iss_rust = RustLibmSatellite(iss)
    iss_states = []
    for label, jd, fr in ISS_SPLITS:
        error, r, v = iss.sgp4(jd, fr)
        assert error == 0
        tsince = (jd - iss.jdsatepoch) * 1440.0 + (fr - iss.jdsatepochF) * 1440.0
        row = {"label": label, "jd_whole": jd, "jd_fraction": fr}
        for key, value in zip(LABELS, list(r) + list(v)):
            row[key] = hexf(value)
        row["bound"] = iss_tracked.bounds(tsince)
        row["rust_libm"] = iss_rust.expect(tsince)
        iss_states.append(row)

    drag_free = Satrec.twoline2rv(DRAG_FREE_ISS_LINE1, ISS_LINE2, WGS72)
    drag_free_tracked = TrackedSatellite(drag_free)
    drag_free_rust = RustLibmSatellite(drag_free)
    drag_free_states = [
        far_state(
            (DRAG_FREE_ISS_LINE1, ISS_LINE2),
            t,
            drag_free_tracked,
            drag_free_rust,
            allow_non_finite=True,
        )
        for t in far_order(DRAG_FREE_ISS_TIMES)
    ]

    fixture = {
        "reference": "python-sgp4 compiled extension (Vallado C++ 2020-07-13), Satrec.twoline2rv, sgp4_tsince",
        "sgp4_version": sgp4.__version__,
        "sgp4_wheel": EXPECTED_SGP4_WHEEL,
        "sgp4_extension_sha256": extension_sha256,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}, "
        f"{platform.python_implementation()} {platform.python_version()}",
        "gravity": "WGS72",
        "opsmode": "i",
        "generator": "fixtures-generators/generate_sgp4_verification.py",
        "bound": "per-component libm-difference bound from fixtures-generators/sgp4_libm_bound.py, km and km/s",
        "rust_libm": "python-sgp4's model in the C++ operation order (fixtures-generators/sgp4_vallado_order.py) with the Rust libm crate 0.2.16 port (fixtures-generators/rust_libm_port.py)",
        "num_satellites": len(satellites),
        "satellites": satellites,
        "iss_split_jd_tests": iss_states,
        "drag_free_iss": {
            "line1": DRAG_FREE_ISS_LINE1,
            "line2": ISS_LINE2,
            "far_times": drag_free_states,
        },
    }
    OUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    n = sum(len(s["propagations"]) for s in satellites)
    errors = sum(1 for s in satellites for p in s["propagations"] if "error" in p)
    print(f"wrote {OUT.name}: {len(satellites)} satellites, {n} states, {errors} error states")
    far = [p for s in satellites for p in s["far_times"]]
    far_errors = sum(1 for p in far if "error" in p)
    far_bounded = sum(1 for p in far if p.get("bound") is not None)
    print(f"far times: {len(far)} states, {far_errors} error states, {far_bounded} with a bound")
    agree = sum(
        1
        for s in satellites
        for p in s["propagations"] + s["far_times"]
        if ("error" in p) == ("error" in p["rust_libm"])
        and all(p.get(k) == p["rust_libm"].get(k) for k in LABELS)
    )
    print(f"python-sgp4 and the Rust-libm model agree bit for bit on {agree} states")


if __name__ == "__main__":
    main()
