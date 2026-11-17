"""Run the selected evaluation suites and summarize their exit statuses."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SUITES = {"manual": "evaluation.py", "deepeval": "deepeval_rag.py"}


def run_suites(selected: list[str]) -> int:
    base_dir = Path(__file__).resolve().parent
    results: list[tuple[str, str, float]] = []
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        started = time.monotonic()
        try:
            result = subprocess.run([sys.executable, str(base_dir / SUITES[name])], cwd=base_dir, check=False)
            status = "PASS" if result.returncode == 0 else "FAIL" if result.returncode == 1 else "ERROR"
        except OSError as error:
            print(f"Could not start {name}: {error}", file=sys.stderr, flush=True)
            status = "ERROR"
        results.append((name, status, time.monotonic() - started))

    print("\n=== Suite summary ===")
    for name, status, elapsed in results:
        print(f"{name}: {status} ({elapsed:.2f}s)")
    passed = sum(status == "PASS" for _, status, _ in results)
    print(f"{passed}/{len(results)} suites passed")
    return 0 if passed == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all",
                        help="Select evaluation suite (default: all; both may call model APIs).")
    selection = parser.parse_args().suite
    return run_suites(list(SUITES) if selection == "all" else [selection])


if __name__ == "__main__":
    raise SystemExit(main())
