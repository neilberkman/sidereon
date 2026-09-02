#!/usr/bin/env python3
"""Check Criterion means against the committed hotpath performance baseline."""

import argparse
import json
from pathlib import Path


THRESHOLD = 0.25


def estimates(root: Path) -> dict[str, float]:
    values = {}
    for path in root.glob("hotpath_linalg*/**/new/estimates.json"):
        relative = path.relative_to(root)
        name = "/".join(relative.parts[:-2])
        if not ("/portable_" in name or name.endswith("/portable_jacobian_2000x200")):
            continue
        with path.open() as stream:
            values[name] = json.load(stream)["mean"]["point_estimate"]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("criterion_dir", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace the baseline with the current Criterion estimates",
    )
    args = parser.parse_args()
    current = estimates(args.criterion_dir)
    if not current:
        parser.error("no expanded hotpath estimates found")

    if args.update:
        document = {
            "schema": "sidereon-core/hotpath-baseline.v1",
            "threshold": THRESHOLD,
            "benchmarks": dict(sorted(current.items())),
        }
        args.baseline.write_text(json.dumps(document, indent=2) + "\n")
        for name, value in sorted(current.items()):
            print(f"updated {name}: {value:.3f} ns")
        return 0

    with args.baseline.open() as stream:
        document = json.load(stream)
    baseline = document["benchmarks"]
    threshold = float(document.get("threshold", THRESHOLD))
    failures = []
    for name, before in sorted(baseline.items()):
        if name not in current:
            failures.append(f"missing benchmark {name}")
            continue
        after = current[name]
        ratio = after / float(before)
        print(f"{name}: {after:.3f} ns / {float(before):.3f} ns = {ratio:.3f}x")
        if ratio > 1.0 + threshold:
            failures.append(
                f"{name} regressed {ratio:.3f}x (limit {1.0 + threshold:.3f}x)"
            )
    for name in sorted(set(current) - set(baseline)):
        failures.append(f"baseline missing {name}")
    if failures:
        print("hotpath performance gate failed:")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
