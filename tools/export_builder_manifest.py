#!/usr/bin/env python3
"""Export the canonical benchmark manifest for the static website builder."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Destination builder-manifest.js")
    args = parser.parse_args()

    source = ROOT / "benchmark_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    args.output.write_text(
        "/* Generated from GLNBench/benchmark_manifest.json. Do not edit by hand. */\n"
        f"window.GLNBENCH_MANIFEST = {payload};\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
