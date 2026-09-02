#!/usr/bin/env python3
"""Compare Criterion means from a merge base and its branch."""

import argparse
import json
from pathlib import Path


LIMIT = 1.25
APPLICATION_GROUPS = (
    "hotpath_bundle/",
    "hotpath_bundle_into/",
    "hotpath_bundle_cached/",
    "hotpath_solve/",
)


def is_gate_case(name: str) -> bool:
    return name.startswith(APPLICATION_GROUPS) or (
        name.startswith("hotpath_linalg_") and "/portable_" in name
    )


def estimates(root: Path) -> dict[str, float]:
    values = {}
    for path in root.glob("**/new/estimates.json"):
        relative = path.relative_to(root)
        name = "/".join(relative.parts[:-2])
        if not is_gate_case(name):
            continue
        with path.open() as stream:
            values[name] = json.load(stream)["mean"]["point_estimate"]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_criterion_dir", type=Path)
    parser.add_argument("branch_criterion_dir", type=Path)
    args = parser.parse_args()
    base = estimates(args.base_criterion_dir)
    branch = estimates(args.branch_criterion_dir)
    if not base or not branch:
        parser.error("both Criterion trees must contain hotpath estimates")
    failures = []
    for name, before in sorted(base.items()):
        if name not in branch:
            failures.append(f"branch missing benchmark {name}")
            continue
        after = branch[name]
        ratio = after / float(before)
        print(f"{name}: branch {after:.3f} ns / base {float(before):.3f} ns = {ratio:.3f}x")
        if ratio > LIMIT:
            failures.append(
                f"{name} regressed {ratio:.3f}x (limit {LIMIT:.3f}x)"
            )
    for name in sorted(set(branch) - set(base)):
        failures.append(f"base missing benchmark {name}")
    if failures:
        print("hotpath performance gate failed:")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
