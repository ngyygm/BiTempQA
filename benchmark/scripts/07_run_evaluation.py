#!/usr/bin/env python3
"""Step 7: Run evaluation on all systems."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.runner import run_evaluation
from src.systems.base import MemorySystem


def create_systems(config: dict, only_systems: list = None) -> list:
    systems = []
    sys_config = config.get("systems", {})

    def _should_create(name: str) -> bool:
        if only_systems:
            lower_names = [n.lower() for n in only_systems]
            return name.lower() in lower_names
        return True

    if _should_create("tmg") and sys_config.get("tmg", {}).get("enabled", False):
        from src.systems.tmg_client import TMGClient
        systems.append(TMGClient(
            api_base=sys_config["tmg"].get("api_base", "http://localhost:8732"),
        ))

    if _should_create("faiss") and sys_config.get("faiss", {}).get("enabled", False):
        from src.systems.faiss_baseline import FAISSBaseline
        systems.append(FAISSBaseline(
            embedding_model=sys_config["faiss"].get("embedding_model"),
        ))

    if _should_create("chroma") and sys_config.get("chroma", {}).get("enabled", False):
        from src.systems.chroma_baseline import ChromaBaseline
        systems.append(ChromaBaseline(
            embedding_model=sys_config["chroma"].get("embedding_model"),
        ))

    if _should_create("simple_kg") and sys_config.get("simple_kg", {}).get("enabled", False):
        from src.systems.simple_kg_baseline import SimpleKGBaseline
        systems.append(SimpleKGBaseline())

    if _should_create("naive_rag") and sys_config.get("naive_rag", {}).get("enabled", False):
        from src.systems.naive_rag_baseline import NaiveRAGBaseline
        systems.append(NaiveRAGBaseline())

    if _should_create("bm25") and sys_config.get("bm25", {}).get("enabled", False):
        from src.systems.bm25_baseline import BM25Baseline
        systems.append(BM25Baseline())

    if _should_create("mem0") and sys_config.get("mem0", {}).get("enabled", False):
        from src.systems.mem0_baseline import Mem0Baseline
        mem0_cfg = sys_config["mem0"]
        systems.append(Mem0Baseline(
            llm_model=mem0_cfg.get("llm_model"),
            llm_base_url=mem0_cfg.get("llm_base_url"),
            llm_api_key=mem0_cfg.get("llm_api_key"),
        ))

    if _should_create("graphiti") and sys_config.get("graphiti", {}).get("enabled", False):
        from src.systems.graphiti_baseline import GraphitiBaseline
        graphiti_cfg = sys_config["graphiti"]
        systems.append(GraphitiBaseline(
            neo4j_uri=graphiti_cfg.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=graphiti_cfg.get("neo4j_user", "neo4j"),
            neo4j_password=graphiti_cfg.get("neo4j_password", "password"),
        ))

    return systems


def main():
    parser = argparse.ArgumentParser(description="Run BiTempQA evaluation")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--scenarios", default="data/generated/scenarios/all_scenarios.json")
    parser.add_argument("--qa", default="data/validated/bitpqa_test_zh.json")
    parser.add_argument("--output-dir", default="data/eval_results")
    parser.add_argument("--systems", nargs="*", default=None, help="Limit to specific system names")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM Judge")
    parser.add_argument("--eval-mode", default=None, choices=["unified", "legacy"],
                        help="Evaluation mode (overrides config)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load(open(base_dir / args.config))

    systems = create_systems(config, args.systems)
    if args.systems:
        args_lower = [n.lower() for n in args.systems]
        systems = [s for s in systems if any(n.lower() in s.name.lower() for n in args.systems)]

    logging.info(f"Systems to evaluate: {[s.name for s in systems]}")

    eval_mode = args.eval_mode or config.get("eval_mode", "unified")
    logging.info(f"Evaluation mode: {eval_mode}")

    # LLM Judge config
    llm_judge_config = None if args.no_judge else config.get("llm_judge")
    if llm_judge_config:
        logging.info(f"LLM Judge enabled (mode={llm_judge_config.get('mode', 'answer_judge')})")

    # Answer Generator config (for unified mode, independent of judge)
    answer_generator_config = None
    if eval_mode == "unified":
        answer_generator_config = config.get("answer_generator")
        if answer_generator_config:
            logging.info(f"Answer Generator enabled (model={answer_generator_config.get('model')})")

    results = run_evaluation(
        scenarios_path=base_dir / args.scenarios,
        qa_path=base_dir / args.qa,
        systems=systems,
        output_dir=base_dir / args.output_dir,
        llm_judge_config=llm_judge_config,
        answer_generator_config=answer_generator_config,
        eval_mode=eval_mode,
    )

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'System':<25} {'Accuracy':>10} {'F1':>8} {'Temporal':>10} {'Version':>10}")
    print("-" * 90)
    for run in results:
        print(
            f"{run.system_name:<25} {run.overall_accuracy:>10.3f} {run.overall_f1:>8.3f} "
            f"{run.temporal_reasoning_accuracy:>10.3f} {run.version_recall:>10.3f}"
        )
    print("=" * 90)

    # Per-level breakdown
    if results:
        print(f"\n{'System':<25}", end="")
        for level in ["level_1", "level_2", "level_3"]:
            print(f" {level:>10}", end="")
        print()
        print("-" * 65)
        for run in results:
            print(f"{run.system_name:<25}", end="")
            for level in ["level_1", "level_2", "level_3"]:
                acc = run.accuracy_by_level.get(level, 0.0)
                print(f" {acc:>10.3f}", end="")
            print()
        print()


if __name__ == "__main__":
    main()
