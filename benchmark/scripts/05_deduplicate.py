#!/usr/bin/env python3
"""Step 5: Deduplicate and split QA pairs into train/dev/test."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.deduplicator import deduplicate_and_split


def main():
    parser = argparse.ArgumentParser(description="Deduplicate and split QA pairs")
    parser.add_argument("--input", default="data/generated/qa_pairs/bitpqa_generated_zh.json")
    parser.add_argument("--output-dir", default="data/validated")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_dir = Path(__file__).parent.parent
    deduplicate_and_split(
        input_path=base_dir / args.input,
        output_dir=base_dir / args.output_dir,
        similarity_threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
