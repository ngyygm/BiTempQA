#!/usr/bin/env python3
"""Step 8: Analyze evaluation results and generate report."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import EvaluationRun


def analyze_results(results_dir: Path) -> dict:
    results_files = list(results_dir.glob("*.json"))
    if not results_files:
        logging.error(f"No result files found in {results_dir}")
        return {}

    runs = []
    for f in results_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            runs.append(EvaluationRun(**data))
        except Exception as e:
            logging.warning(f"Failed to load {f}: {e}")

    report = {
        "systems": {},
        "comparisons": {},
    }

    for run in runs:
        report["systems"][run.system_name] = {
            "overall_accuracy": run.overall_accuracy,
            "overall_f1": run.overall_f1,
            "temporal_reasoning_accuracy": run.temporal_reasoning_accuracy,
            "version_recall": run.version_recall,
            "accuracy_by_level": run.accuracy_by_level,
            "accuracy_by_scenario_type": run.accuracy_by_scenario_type,
            "latency_stats": run.latency_stats,
        }

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results-dir", default="data/eval_results")
    parser.add_argument("--output", default="data/eval_results/analysis_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_dir = Path(__file__).parent.parent
    report = analyze_results(base_dir / args.results_dir)

    out_path = base_dir / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"Analysis report saved to {out_path}")

    # Print summary
    print("\n=== Evaluation Analysis ===\n")
    for name, metrics in report["systems"].items():
        print(f"  {name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"    {k}: {v}")
        print()


if __name__ == "__main__":
    main()
