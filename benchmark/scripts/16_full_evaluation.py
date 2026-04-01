#!/usr/bin/env python3
"""Step 16: Full evaluation across all benchmarks.

Runs the complete evaluation pipeline:
  Part A: LoCoMo benchmark (1,986 MC questions, 10 long conversations)
  Part B: Novel-based long text (temporal QA from Chinese novels)
  Part C: BiTempQA v2 (dual-timestamp reasoning)

All systems use unified pipeline: Retrieve → LLM Generate → LLM Judge

Usage:
    python scripts/16_full_evaluation.py --systems faiss bm25 naive_rag
    python scripts/16_full_evaluation.py --all --no-judge
    python scripts/16_full_evaluation.py --part locomo --systems faiss bm25
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.base import ingest_conversation
from src.benchmarks.locomo_loader import LoCoMoLoader
from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.systems.base import MemorySystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# System creation (shared across all benchmarks)
# ============================================================

def create_systems(config: dict, only_systems: list = None) -> list:
    """Create memory systems from config."""
    systems = []
    sys_config = config.get("systems", {})

    def _ok(name: str) -> bool:
        if only_systems:
            return name.lower() in [n.lower() for n in only_systems]
        return True

    if _ok("tmg") and sys_config.get("tmg", {}).get("enabled", False):
        from src.systems.tmg_client import TMGClient
        systems.append(TMGClient(
            api_base=sys_config["tmg"].get("api_base", "http://localhost:8732"),
        ))

    if _ok("faiss") and sys_config.get("faiss", {}).get("enabled", False):
        from src.systems.faiss_baseline import FAISSBaseline
        systems.append(FAISSBaseline(
            embedding_model=sys_config["faiss"].get("embedding_model"),
        ))

    if _ok("chroma") and sys_config.get("chroma", {}).get("enabled", False):
        from src.systems.chroma_baseline import ChromaBaseline
        systems.append(ChromaBaseline(
            embedding_model=sys_config["chroma"].get("embedding_model"),
        ))

    if _ok("simple_kg") and sys_config.get("simple_kg", {}).get("enabled", False):
        from src.systems.simple_kg_baseline import SimpleKGBaseline
        systems.append(SimpleKGBaseline())

    if _ok("naive_rag") and sys_config.get("naive_rag", {}).get("enabled", False):
        from src.systems.naive_rag_baseline import NaiveRAGBaseline
        systems.append(NaiveRAGBaseline())

    if _ok("bm25") and sys_config.get("bm25", {}).get("enabled", False):
        from src.systems.bm25_baseline import BM25Baseline
        systems.append(BM25Baseline())

    if _ok("mem0") and sys_config.get("mem0", {}).get("enabled", False):
        from src.systems.mem0_baseline import Mem0Baseline
        m = sys_config["mem0"]
        systems.append(Mem0Baseline(
            llm_model=m.get("llm_model"),
            llm_base_url=m.get("llm_base_url"),
            llm_api_key=m.get("llm_api_key"),
        ))

    if _ok("graphiti") and sys_config.get("graphiti", {}).get("enabled", False):
        from src.systems.graphiti_baseline import GraphitiBaseline
        g = sys_config["graphiti"]
        systems.append(GraphitiBaseline(
            neo4j_uri=g.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=g.get("neo4j_user", "neo4j"),
            neo4j_password=g.get("neo4j_password", "password"),
        ))

    return systems


# ============================================================
# Part A: LoCoMo
# ============================================================

def run_locomo(
    systems: list,
    answer_generator: AnswerGenerator,
    judge: LLMJudge,
    config: dict,
    no_judge: bool,
    max_questions: int = None,
) -> dict:
    """Run LoCoMo benchmark evaluation."""
    logger.info("=" * 60)
    logger.info("PART A: LoCoMo Benchmark")
    logger.info("=" * 60)

    loader = LoCoMoLoader()
    stats = loader.get_stats()
    logger.info(f"LoCoMo: {stats['total_questions']} questions, "
                f"{stats['total_conversations']} conversations")

    conversations = loader.get_conversations()
    questions = loader.get_questions()
    if max_questions:
        questions = questions[:max_questions]

    conv_map = {c.conversation_id: c for c in conversations}
    qs_by_conv = defaultdict(list)
    for q in questions:
        qs_by_conv[q.conversation_id].append(q)

    all_results = {}
    for system in systems:
        logger.info(f"\n--- [{system.name}] LoCoMo ---")
        sys_results = {}
        total = len(questions)
        done = 0

        for conv_id, conv_qs in qs_by_conv.items():
            conv = conv_map.get(conv_id)
            if not conv:
                continue

            t0 = time.time()
            system.reset()
            n = ingest_conversation(system, conv)
            logger.info(f"  Ingested {n} turns for {conv_id} ({time.time()-t0:.1f}s)")

            for q in conv_qs:
                done += 1
                if done % 100 == 0:
                    logger.info(f"  [{system.name}] {done}/{total}")

                qr = system.query(question=q.question)

                generated = answer_generator.generate(
                    question=q.question,
                    retrieved_context=qr.retrieved_context,
                    choices=q.choices,
                )

                if no_judge or judge is None:
                    from src.evaluation.answer_generator import parse_mc_answer
                    sel = parse_mc_answer(generated, len(q.choices))
                    correct = sel == q.correct_choice_index if sel is not None else False
                else:
                    jr = judge.judge_single({
                        "question": q.question,
                        "gold_answer": q.gold_answer,
                        "generated_answer": generated,
                        "question_id": q.question_id,
                        "system_name": system.name,
                        "choices": q.choices,
                        "correct_choice_index": q.correct_choice_index,
                    })
                    correct = jr.get("is_correct", False)

                sys_results[q.question_id] = {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "is_correct": correct,
                    "system_name": system.name,
                }

        # Compute metrics
        by_type = defaultdict(lambda: {"c": 0, "t": 0})
        c_total = sum(1 for r in sys_results.values() if r["is_correct"])
        t_total = len(sys_results)
        for r in sys_results.values():
            qt = r["question_type"]
            by_type[qt]["t"] += 1
            if r["is_correct"]:
                by_type[qt]["c"] += 1

        metrics = {
            "overall": c_total / t_total if t_total else 0,
            "correct": c_total,
            "total": t_total,
            "by_type": {
                k: {"accuracy": v["c"]/v["t"] if v["t"] else 0, "correct": v["c"], "total": v["t"]}
                for k, v in by_type.items()
            },
        }
        all_results[system.name] = {"metrics": metrics, "results": sys_results}
        logger.info(f"  [{system.name}] Overall: {metrics['overall']:.3f} ({c_total}/{t_total})")

    return all_results


# ============================================================
# Part C: BiTempQA v2
# ============================================================

def run_bitpqa(
    systems: list,
    answer_generator: AnswerGenerator,
    judge: LLMJudge,
    config: dict,
    no_judge: bool,
) -> dict:
    """Run BiTempQA v2 evaluation using existing evaluation runner."""
    logger.info("=" * 60)
    logger.info("PART C: BiTempQA v2")
    logger.info("=" * 60)

    from src.evaluation.runner import run_evaluation
    from src.schemas import QAPair, QADataset

    # Load QA dataset
    qa_path = Path(__file__).parent.parent / config.get("dataset", {}).get(
        "test_path", "data/validated/bitpqa_test_zh.json"
    )
    if not qa_path.exists():
        logger.warning(f"BiTempQA QA file not found: {qa_path}")
        return {}

    # Load scenarios
    scenarios_path = Path(__file__).parent.parent / "data/generated/scenarios/all_scenarios.json"
    if not scenarios_path.exists():
        logger.warning(f"Scenarios file not found: {scenarios_path}")
        return {}

    with open(scenarios_path) as f:
        scenarios_data = json.load(f)

    from src.schemas import Scenario
    scenarios = [Scenario(**s) for s in scenarios_data]

    with open(qa_path) as f:
        qa_data = json.load(f)

    if isinstance(qa_data, list):
        qa_pairs = [QAPair(**q) for q in qa_data]
    elif isinstance(qa_data, dict):
        qa_pairs = [QAPair(**q) for q in qa_data.get("qa_pairs", [])]
    else:
        logger.error(f"Unknown QA data format: {type(qa_data)}")
        return {}

    eval_mode = config.get("eval_mode", "unified")
    llm_judge_config = None if no_judge else config.get("llm_judge")
    ag_config = config.get("answer_generator") if eval_mode == "unified" else None

    output_dir = Path(__file__).parent.parent / config.get("evaluation", {}).get(
        "output_dir", "data/eval_results"
    )

    runs = run_evaluation(
        scenarios_path=scenarios_path,
        qa_path=qa_path,
        systems=systems,
        output_dir=output_dir,
        llm_judge_config=llm_judge_config,
        answer_generator_config=ag_config,
        eval_mode=eval_mode,
    )

    results = {}
    for run in runs:
        results[run.system_name] = {
            "metrics": {
                "overall": run.overall_accuracy,
                "f1": run.overall_f1,
                "temporal": run.temporal_reasoning_accuracy,
                "version_recall": run.version_recall,
                "by_level": run.accuracy_by_level,
                "by_scenario_type": run.accuracy_by_scenario_type,
            },
        }
    return results


# ============================================================
# Report generation
# ============================================================

def print_report(locomo_results: dict, bitpqa_results: dict, novel_results: dict = None):
    """Print comprehensive evaluation report."""
    all_systems = set()
    all_systems.update(locomo_results.keys())
    all_systems.update(bitpqa_results.keys())
    if novel_results:
        all_systems.update(novel_results.keys())

    print("\n" + "=" * 100)
    print(f"{'COMPREHENSIVE EVALUATION REPORT':^100}")
    print(f"{'Generated: ' + time.strftime('%Y-%m-%d %H:%M'):^100}")
    print("=" * 100)

    # Part A: LoCoMo
    if locomo_results:
        print(f"\n{'Part A: LoCoMo Benchmark (1,986 MC questions)':^100}")
        print("-" * 100)
        print(f"{'System':<20} {'Overall':>10} {'Single':>8} {'Multi':>8} "
              f"{'Temporal':>10} {'Open':>8} {'Adversarial':>12}")
        print("-" * 100)

        for sys_name in sorted(locomo_results.keys()):
            m = locomo_results[sys_name]["metrics"]
            bt = m.get("by_type", {})
            s = bt.get("single_hop", {}).get("accuracy", 0)
            mh = bt.get("multi_hop", {}).get("accuracy", 0)
            t = bt.get("temporal_reasoning", {}).get("accuracy", 0)
            o = bt.get("open_domain", {}).get("accuracy", 0)
            a = bt.get("adversarial", {}).get("accuracy", 0)
            print(f"{sys_name:<20} {m['overall']:>10.3f} {s:>8.3f} {mh:>8.3f} "
                  f"{t:>10.3f} {o:>8.3f} {a:>12.3f}")

    # Part C: BiTempQA
    if bitpqa_results:
        print(f"\n{'Part C: BiTempQA v2 (Dual-Timestamp Reasoning)':^100}")
        print("-" * 100)
        print(f"{'System':<20} {'Accuracy':>10} {'F1':>8} {'Temporal':>10} "
              f"{'Version':>10} {'L1':>8} {'L2':>8} {'L3':>8}")
        print("-" * 100)

        for sys_name in sorted(bitpqa_results.keys()):
            m = bitpqa_results[sys_name]["metrics"]
            bl = m.get("by_level", {})
            l1 = bl.get("level_1", 0)
            l2 = bl.get("level_2", 0)
            l3 = bl.get("level_3", 0)
            print(f"{sys_name:<20} {m['overall']:>10.3f} {m.get('f1', 0):>8.3f} "
                  f"{m.get('temporal', 0):>10.3f} {m.get('version_recall', 0):>10.3f} "
                  f"{l1:>8.3f} {l2:>8.3f} {l3:>8.3f}")

    # Part B: Novels
    if novel_results:
        print(f"\n{'Part B: Novel Long-Text Benchmark':^100}")
        print("-" * 100)
        for key, data in sorted(novel_results.items()):
            m = data["metrics"]
            sys_name = data.get("system_name", key)
            print(f"  {sys_name}: {m['overall']:.3f} ({m['correct']}/{m['total']})")

    print(f"\n{'=' * 100}\n")


def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="Run all enabled systems")
    parser.add_argument("--part", nargs="*", default=None,
                        choices=["locomo", "novel", "bitpqa", "all"],
                        help="Which benchmarks to run")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--output-dir", default="data/eval_results/full")
    parser.add_argument("--max-locomo-questions", type=int, default=None)
    parser.add_argument("--skip-bitpqa", action="store_true",
                        help="Skip BiTempQA (useful if data not generated)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load(open(base_dir / args.config))

    # Determine which parts to run
    parts = set()
    if args.part:
        if "all" in args.part:
            parts = {"locomo", "novel", "bitpqa"}
        else:
            parts = set(args.part)
    else:
        parts = {"locomo", "bitpqa"}

    # Create systems
    if args.all or args.systems:
        systems = create_systems(config, args.systems)
    else:
        systems = create_systems(config, ["faiss", "bm25", "naive_rag"])

    if not systems:
        logger.error("No systems to evaluate.")
        sys.exit(1)

    logger.info(f"Systems: {[s.name for s in systems]}")
    logger.info(f"Parts: {parts}")

    # Setup shared components
    ag_config = config.get("answer_generator", {})
    answer_generator = AnswerGenerator(
        base_url=ag_config.get("base_url"),
        api_key=ag_config.get("api_key"),
        model=ag_config.get("model", "deepseek-ai/DeepSeek-V3"),
    )

    judge = None
    if not args.no_judge:
        jc = config.get("llm_judge", {})
        if jc:
            judge = LLMJudge(
                base_url=jc.get("base_url", ag_config.get("base_url")),
                api_key=jc.get("api_key", ag_config.get("api_key")),
                model=jc.get("model", ag_config.get("model")),
                mode=jc.get("mode", "answer_judge"),
            )
            logger.info(f"LLM Judge: {judge.mode}")

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    locomo_results = {}
    bitpqa_results = {}
    novel_results = {}

    if "locomo" in parts:
        locomo_results = run_locomo(
            systems, answer_generator, judge, config, args.no_judge,
            max_questions=args.max_locomo_questions,
        )

    if "bitpqa" in parts and not args.skip_bitpqa:
        bitpqa_results = run_bitpqa(
            systems, answer_generator, judge, config, args.no_judge,
        )

    # Save all results
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "systems": [s.name for s in systems],
        "locomo": locomo_results,
        "bitpqa": bitpqa_results,
        "novel": novel_results,
    }
    results_path = output_dir / "full_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Print report
    print_report(locomo_results, bitpqa_results, novel_results)


if __name__ == "__main__":
    main()
