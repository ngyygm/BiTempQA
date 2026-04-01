#!/usr/bin/env python3
"""Step 3: Validate generated scenarios and QA pairs."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.validator import validate_qa_file, validate_scenarios_file


def main():
    parser = argparse.ArgumentParser(description="Validate generated data")
    parser.add_argument("--scenarios", default="data/generated/scenarios/all_scenarios.json")
    parser.add_argument("--qa", default="data/generated/qa_pairs/bitpqa_generated_zh.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if Path(args.scenarios).exists():
        results = validate_scenarios_file(Path(args.scenarios))
        print(json.dumps(results, ensure_ascii=False, indent=2))

    if Path(args.qa).exists():
        results = validate_qa_file(Path(args.qa))
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
