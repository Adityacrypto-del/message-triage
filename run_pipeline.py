#!/usr/bin/env python3
"""CLI entry point.

    python run_pipeline.py --data data --out outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from msgiq.pipeline import export, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the message triage pipeline.")
    ap.add_argument("--data", default="data", help="folder holding messages.csv")
    ap.add_argument("--out", default="outputs", help="folder for generated JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not (Path(args.data) / "messages.csv").exists():
        print(f"error: {args.data}/messages.csv not found.", file=sys.stderr)
        print("The dataset is intentionally not committed to this repository; "
              "place it in the data/ folder locally.", file=sys.stderr)
        return 1

    res = run(args.data, verbose=not args.quiet)
    written = export(res, args.out)

    if not args.quiet:
        print("\nWrote:")
        for name, path in written.items():
            print(f"  {path}  ({path.stat().st_size:,} bytes)")
        print(f"\n{len(res.flagged)} message(s) flagged for human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
