#!/usr/bin/env python3
"""Step 1: Generate scenarios using LLM.

Usage:
    python scripts/01_generate_scenarios.py [--config configs/generation_config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.scenario_generator import generate_all_scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate BiTempQA scenarios")
    parser.add_argument("--config", default="configs/generation_config.yaml", help="Config file path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--start-index", type=int, default=1, help="Starting scenario index")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent API requests")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    base_dir = Path(__file__).parent.parent
    config_path = base_dir / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config["llm"]
    output_dir = base_dir / (args.output_dir or "data/generated/scenarios")
    output_dir.mkdir(parents=True, exist_ok=True)

    skeletons_dir = base_dir / "data/raw/scenarios"
    seed_prompt_path = base_dir / "data/raw/seed_prompts/scenario_generation_zh.txt"

    generate_all_scenarios(
        output_dir=output_dir,
        skeletons_dir=skeletons_dir,
        seed_prompt_path=seed_prompt_path,
        api_base=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg["generation_model"],
        start_index=args.start_index,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
