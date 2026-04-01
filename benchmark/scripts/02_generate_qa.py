#!/usr/bin/env python3
"""Step 2: Generate QA pairs from scenarios using LLM.

Usage:
    python scripts/02_generate_qa.py [--scenarios data/generated/scenarios/all_scenarios.json]
    python scripts/02_generate_qa.py --workers 3 --timeout 300
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.qa_generator import generate_qa_for_scenarios
from src.schemas import Scenario


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from scenarios")
    parser.add_argument("--config", default="configs/generation_config.yaml")
    parser.add_argument("--scenarios", default="data/generated/scenarios/all_scenarios.json")
    parser.add_argument("--output", default="data/generated/qa_pairs/bitpqa_generated_zh.json")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent API requests")
    parser.add_argument("--timeout", type=int, default=300, help="API timeout in seconds")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load(open(base_dir / args.config))
    llm_cfg = config["llm"]

    # Load scenarios
    scenarios_data = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    scenarios = [Scenario(**s) for s in scenarios_data]
    logging.info(f"Loaded {len(scenarios)} scenarios")

    seed_prompt_path = base_dir / "data/raw/seed_prompts/qa_generation_zh.txt"

    generate_qa_for_scenarios(
        scenarios=scenarios,
        output_path=base_dir / args.output,
        seed_prompt_path=seed_prompt_path,
        api_base=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg["generation_model"],
        max_workers=args.workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
