#!/usr/bin/env python3
"""Step 4: Translate QA pairs from Chinese to English."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.translator import Translator
from src.schemas import QADataset


def main():
    parser = argparse.ArgumentParser(description="Translate QA pairs to English")
    parser.add_argument("--input", default="data/validated/bitpqa_test_zh.json")
    parser.add_argument("--output", default="data/validated/bitpqa_test_bilingual.json")
    parser.add_argument("--config", default="configs/generation_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load(open(base_dir / args.config))
    llm_cfg = config["llm"]

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    dataset = QADataset(**data)

    translator = Translator(
        api_base=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=config["translation"]["model"],
    )
    translated = translator.translate_dataset(dataset)

    out_path = base_dir / args.output
    out_path.write_text(translated.model_dump_json(indent=2), encoding="utf-8")
    logging.info(f"Translated dataset saved to {out_path}")


if __name__ == "__main__":
    main()
