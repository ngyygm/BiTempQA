#!/usr/bin/env python3
"""Step 6: Ingest scenarios into TMG for evaluation (helper script)."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import Scenario


def main():
    parser = argparse.ArgumentParser(description="Ingest scenarios into TMG")
    parser.add_argument("--scenarios", default="data/generated/scenarios/all_scenarios.json")
    parser.add_argument("--api-base", default="http://localhost:8732")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_dir = Path(__file__).parent.parent
    from src.systems.tmg_client import TMGClient

    client = TMGClient(api_base=args.api_base)
    scenarios_data = json.loads((base_dir / args.scenarios).read_text(encoding="utf-8"))

    for s_data in scenarios_data:
        scenario = Scenario(**s_data)
        client.reset()
        logging.info(f"Ingesting {scenario.scenario_id}: {scenario.title_zh}")
        for w in scenario.memory_writes:
            client.remember(text=w.text, event_time=w.event_time)
        logging.info(f"  Ingested {len(scenario.memory_writes)} writes")

    logging.info("Done")


if __name__ == "__main__":
    main()
