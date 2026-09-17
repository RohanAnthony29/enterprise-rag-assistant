#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.pipeline import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a metadata-rich handbook corpus")
    parser.add_argument("--config", type=Path, default=ROOT / "config/corpus.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/corpus-v1")
    args = parser.parse_args()
    print(json.dumps(build_corpus(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()

