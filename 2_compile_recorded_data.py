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
from concurrent.futures import ThreadPoolExecutor, as_completed

VLA_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(VLA_DIR, "vla_data", "pickle")
VISUALIZE_SCRIPT = os.path.join(VLA_DIR, "openteach", "visualize_demo.py")


def visualize_one(demo_number):
    """Run visualize_demo.py for a single demo. Returns (demo_number, returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, VISUALIZE_SCRIPT, "--demo_number", demo_number],
        cwd=os.path.join(VLA_DIR, "openteach"),
        capture_output=True,
        text=True,
    )
    return demo_number, result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", type=str, help="Glob pattern to match demonstrations in vla_data/pickle (e.g. 'real_*').")
    parser.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="Number of demonstrations to visualize in parallel.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Recompile demonstrations even if they have already been compiled.")
    args = parser.parse_args()

    if not os.path.isdir(PICKLE_DIR):
        sys.exit(f"pickle folder not found: {PICKLE_DIR}")

    matches = sorted(glob.glob(os.path.join(PICKLE_DIR, args.pattern)))

    # Collect unique demo numbers from matching demonstration folders.
    demo_numbers = []
    seen = set()
    skipped = []
    for path in matches:
        if not os.path.isdir(path):
            continue
        demo_number = os.path.basename(path.rstrip(os.sep))
        if not demo_number or demo_number in seen:
            continue
        seen.add(demo_number)
        # visualize_demo.py writes the compiled output to demo_<demo>.pkl inside
        # the demonstration folder. Skip demos that already have this file
        # unless --force is given.
        compiled_pkl = os.path.join(path, f"demo_{demo_number}.pkl")
        if not args.force and os.path.isfile(compiled_pkl):
            skipped.append(demo_number)
            continue
        demo_numbers.append(demo_number)

    if skipped:
        print(f"Skipping {len(skipped)} already-compiled demonstration(s): "
              f"{', '.join(skipped)}")

    if not demo_numbers:
        if skipped:
            sys.exit("All matching demonstrations are already compiled. "
                     "Use --force to recompile.")
        sys.exit(f"No demonstration folders matched pattern '{args.pattern}' in {PICKLE_DIR}")

    jobs = max(1, args.jobs)
    print(f"Found {len(demo_numbers)} demonstration(s) to visualize "
          f"({jobs} in parallel): {', '.join(demo_numbers)}")

    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(visualize_one, d): d for d in demo_numbers}
        for future in as_completed(futures):
            demo_number, returncode, stdout, stderr = future.result()
            print(f"\n=== Finished demo '{demo_number}' (exit code {returncode}) ===")
            if stdout:
                print(stdout, end="")
            if returncode != 0:
                failures.append(demo_number)
                if stderr:
                    print(stderr, end="")
                print(f"visualize_demo.py failed for demo '{demo_number}' (exit code {returncode})")

    if failures:
        sys.exit(f"\n{len(failures)} demonstration(s) failed: {', '.join(failures)}")

    print(f"\nDone. Visualized {len(demo_numbers)} demonstration(s).")


if __name__ == "__main__":
    main()
