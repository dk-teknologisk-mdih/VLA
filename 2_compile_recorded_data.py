#!/usr/bin/env python3
"""
Run openteach/visualize_demo.py on every demonstration in vla_data/pickle that
matches a given glob pattern.

Usage:
    python 2_compile_recorded_data.py "real_*"
    python 2_compile_recorded_data.py "*"

The glob is matched against entries inside vla_data/pickle. Each matching
demonstration folder (named "<demo_number>") is visualized by invoking
visualize_demo.py with the folder name as --demo_number.
"""

import argparse
import glob
import os
import subprocess
import sys

VLA_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(VLA_DIR, "vla_data", "pickle")
VISUALIZE_SCRIPT = os.path.join(VLA_DIR, "openteach", "visualize_demo.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", type=str, help="Glob pattern to match demonstrations in vla_data/pickle (e.g. 'real_*').")
    args = parser.parse_args()

    if not os.path.isdir(PICKLE_DIR):
        sys.exit(f"pickle folder not found: {PICKLE_DIR}")

    matches = sorted(glob.glob(os.path.join(PICKLE_DIR, args.pattern)))

    # Collect unique demo numbers from matching demonstration folders.
    demo_numbers = []
    seen = set()
    for path in matches:
        if not os.path.isdir(path):
            continue
        demo_number = os.path.basename(path.rstrip(os.sep))
        if demo_number and demo_number not in seen:
            seen.add(demo_number)
            demo_numbers.append(demo_number)

    if not demo_numbers:
        sys.exit(f"No demonstration folders matched pattern '{args.pattern}' in {PICKLE_DIR}")

    print(f"Found {len(demo_numbers)} demonstration(s) to visualize: {', '.join(demo_numbers)}")

    failures = []
    for demo_number in demo_numbers:
        print(f"\n=== Visualizing demo '{demo_number}' ===")
        result = subprocess.run(
            [sys.executable, VISUALIZE_SCRIPT, "--demo_number", demo_number],
            cwd=os.path.join(VLA_DIR, "openteach"),
        )
        if result.returncode != 0:
            failures.append(demo_number)
            print(f"visualize_demo.py failed for demo '{demo_number}' (exit code {result.returncode})")

    if failures:
        sys.exit(f"\n{len(failures)} demonstration(s) failed: {', '.join(failures)}")

    print(f"\nDone. Visualized {len(demo_numbers)} demonstration(s).")


if __name__ == "__main__":
    main()
