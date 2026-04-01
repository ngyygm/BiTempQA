#!/usr/bin/env python3
"""Step 14: Run LoCoMo benchmark evaluation.

Evaluates memory systems on the LoCoMo benchmark (ACL 2024):
  - 10 long conversations (~300 turns each)
  - 1,986 multi-choice questions (10 choices each)
  - 5 question types: single_hop, multi_hop, temporal_reasoning, open_domain, adversarial

Pipeline: Ingest conversation → Query → LLM Generate → LLM Judge

Usage:
    python scripts/14_run_locomo_eval.py --systems faiss bm25
    python scripts/14_run_locomo_eval.py --systems mem0 --question-type temporal_reasoning
    python scripts/14_run_locomo_eval.py --all --no-judge
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


def create_systems(config: dict, only_systems: list = None) -> list:
    """Create memory systems from config."""
    systems = []
    sys_config = config.get("systems", {})

    def _should_create(name: str) -> bool:
        if only_systems:
            return name.lower() in [n.lower() for n in only_systems]
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


def evaluate_system(
    system: MemorySystem,
    conversations: list,
    questions: list,
    answer_generator: AnswerGenerator,
    judge: LLMJudge,
    no_judge: bool = False,
) -> dict:
    """Evaluate a single system on LoCoMo questions.

    Returns per-question results dict.
    """
    # Group questions by conversation for efficient ingestion
    qs_by_conv = defaultdict(list)
    for q in questions:
        qs_by_conv[q.conversation_id].append(q)

    # Map conversation_id to Conversation object
    conv_map = {c.conversation_id: c for c in conversations}

    results = {}
    total = len(questions)
    processed = 0

    for conv_id, conv_questions in qs_by_conv.items():
        logger.info(
            f"[{system.name}] Processing {conv_id} "
            f"({len(conv_questions)} questions)..."
        )

        # Ingest conversation
        conv = conv_map.get(conv_id)
        if conv is None:
            logger.warning(f"Conversation {conv_id} not found, skipping")
            continue

        t0 = time.time()
        system.reset()
        n_turns = ingest_conversation(system, conv)
        logger.info(
            f"[{system.name}] Ingested {n_turns} turns for {conv_id} "
            f"in {time.time()-t0:.1f}s"
        )

        # Answer each question
        for q in conv_questions:
            processed += 1
            if processed % 50 == 0:
                logger.info(
                    f"[{system.name}] Progress: {processed}/{total}"
                )

            # Query
            query_result = system.query(question=q.question)

            # Generate answer (MC: pick from 10 choices)
            if q.choices:
                generated = answer_generator.generate(
                    question=q.question,
                    retrieved_context=query_result.retrieved_context,
                    choices=q.choices,
                )
            else:
                generated = answer_generator.generate(
                    question=q.question,
                    retrieved_context=query_result.retrieved_context,
                )

            # Judge
            if no_judge:
                # Direct matching: extract choice letter and compare
                from src.evaluation.answer_generator import parse_mc_answer
                selected = parse_mc_answer(generated, len(q.choices))
                is_correct = (
                    selected == q.correct_choice_index
                    if selected is not None and q.correct_choice_index is not None
                    else False
                )
            else:
                judge_item = {
                    "question": q.question,
                    "gold_answer": q.gold_answer,
                    "generated_answer": generated,
                    "question_id": q.question_id,
                    "system_name": system.name,
                    "choices": q.choices,
                    "correct_choice_index": q.correct_choice_index,
                }
                judge_result = judge.judge_single(judge_item)
                is_correct = judge_result.get("is_correct", False)

            results[q.question_id] = {
                "question_id": q.question_id,
                "conversation_id": conv_id,
                "question_type": q.question_type,
                "question": q.question,
                "gold_answer": q.gold_answer,
                "generated_answer": generated,
                "correct_choice_index": q.correct_choice_index,
                "is_correct": is_correct,
                "system_name": system.name,
                "latency_ms": query_result.latency_ms,
            }

    return results


def compute_metrics(results: dict) -> dict:
    """Compute metrics from results dict."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = len(results)

    for r in results.values():
        qt = r.get("question_type", "unknown")
        by_type[qt]["total"] += 1
        if r["is_correct"]:
            by_type[qt]["correct"] += 1
            overall_correct += 1

    metrics = {
        "overall_accuracy": overall_correct / overall_total if overall_total else 0,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "by_type": {},
    }
    for qt, v in by_type.items():
        metrics["by_type"][qt] = {
            "accuracy": v["correct"] / v["total"] if v["total"] else 0,
            "correct": v["correct"],
            "total": v["total"],
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo evaluation")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--output-dir", default="data/eval_results/locomo")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="Run all enabled systems")
    parser.add_argument("--question-type", default=None,
                        help="Filter by question type (e.g., temporal_reasoning)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM Judge")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions for quick test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load(open(base_dir / args.config))

    # Load LoCoMo
    logger.info("Loading LoCoMo benchmark...")
    loader = LoCoMoLoader()
    stats = loader.get_stats()
    logger.info(f"LoCoMo: {stats['total_questions']} questions, "
                f"{stats['total_conversations']} conversations, "
                f"{stats['avg_turns_per_conv']:.0f} avg turns/conv")
    logger.info(f"Question types: {stats['question_types']}")

    conversations = loader.get_conversations()
    questions = loader.get_questions(question_type=args.question_type)

    if args.max_questions:
        questions = questions[:args.max_questions]
        logger.info(f"Limited to {len(questions)} questions")

    logger.info(f"Evaluating on {len(questions)} questions")

    # Create systems
    if args.all or args.systems:
        systems = create_systems(config, args.systems)
    else:
        # Default: run simple baselines only
        systems = create_systems(config, ["faiss", "bm25", "naive_rag"])

    if not systems:
        logger.error("No systems to evaluate. Check config or use --systems.")
        sys.exit(1)

    logger.info(f"Systems: {[s.name for s in systems]}")

    # Setup answer generator and judge
    ag_config = config.get("answer_generator", {})
    answer_generator = AnswerGenerator(
        base_url=ag_config.get("base_url"),
        api_key=ag_config.get("api_key"),
        model=ag_config.get("model", "deepseek-ai/DeepSeek-V3"),
        temperature=ag_config.get("temperature", 0.0),
    )

    judge = None
    if not args.no_judge:
        judge_config = config.get("llm_judge", {})
        if judge_config:
            judge = LLMJudge(
                base_url=judge_config.get("base_url", ag_config.get("base_url")),
                api_key=judge_config.get("api_key", ag_config.get("api_key")),
                model=judge_config.get("model", ag_config.get("model")),
                mode=judge_config.get("mode", "answer_judge"),
            )
            logger.info(f"LLM Judge enabled (mode={judge.mode})")
        else:
            logger.warning("No llm_judge config found, using direct matching")

    # Run evaluation
    all_system_results = {}
    for system in systems:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {system.name}")
        logger.info(f"{'='*60}")

        system_results = evaluate_system(
            system=system,
            conversations=conversations,
            questions=questions,
            answer_generator=answer_generator,
            judge=judge,
            no_judge=args.no_judge or judge is None,
        )
        metrics = compute_metrics(system_results)

        all_system_results[system.name] = {
            "metrics": metrics,
            "results": system_results,
        }

        logger.info(
            f"[{system.name}] Overall: {metrics['overall_accuracy']:.3f} "
            f"({metrics['overall_correct']}/{metrics['overall_total']})"
        )

    # Save results
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results per system
    for sys_name, sys_data in all_system_results.items():
        out_path = output_dir / f"locomo_{sys_name.lower()}_results.json"
        with open(out_path, "w") as f:
            json.dump(sys_data["results"], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {out_path}")

    # Save summary
    summary = {}
    for sys_name, sys_data in all_system_results.items():
        summary[sys_name] = sys_data["metrics"]

    summary_path = output_dir / "locomo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # Print results table
    print(f"\n{'='*90}")
    print(f"{'LoCoMo Benchmark Results':^90}")
    print(f"{'='*90}")
    print(f"{'System':<20} {'Overall':>10}", end="")
    for qt in ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]:
        print(f" {qt:>12}", end="")
    print()
    print("-" * 90)

    for sys_name, sys_data in all_system_results.items():
        m = sys_data["metrics"]
        print(f"{sys_name:<20} {m['overall_accuracy']:>10.3f}", end="")
        for qt_key in ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]:
            acc = m["by_type"].get(qt_key, {}).get("accuracy", 0.0)
            print(f" {acc:>12.3f}", end="")
        print()

    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
