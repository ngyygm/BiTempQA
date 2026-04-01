#!/usr/bin/env python3
"""Step 15: Generate QA from novels and evaluate memory systems.

Uses Chinese novels as long-text sources:
  - Segments novel into paragraphs (~50-500 chars each)
  - Generates temporal QA pairs via LLM
  - Evaluates memory systems on retrieval + QA accuracy

Usage:
    python scripts/15_run_novel_eval.py --novels "活着.txt" --max-questions 50
    python scripts/15_run_novel_eval.py --all-novels --generate-qa --systems faiss bm25
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

from src.benchmarks.novel_loader import NovelLoader, NovelQAGenerator, NovelQA
from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.systems.base import MemorySystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_systems(config: dict, only_systems: list = None) -> list:
    """Create memory systems from config (same as 07_run_evaluation.py)."""
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


def generate_qa_for_novel(
    loader: NovelLoader,
    novel_file: str,
    qa_generator: NovelQAGenerator,
    questions_per_chapter: int = 10,
    max_questions: int = 100,
) -> list:
    """Generate QA pairs for a novel."""
    segments = loader.load_novel(novel_file)
    if not segments:
        logger.warning(f"No segments loaded from {novel_file}")
        return []

    # Group segments by chapter
    chapters = defaultdict(list)
    for seg in segments:
        chapters[seg.chapter].append(seg)

    all_qas = []
    novel_title = Path(novel_file).stem

    for chapter_name, chapter_segments in chapters.items():
        if len(all_qas) >= max_questions:
            break

        n_q = min(questions_per_chapter, max_questions - len(all_qas))
        logger.info(
            f"Generating {n_q} QA for '{novel_title}' chapter '{chapter_name}' "
            f"({len(chapter_segments)} segments)..."
        )

        try:
            qas = qa_generator.generate_qa_for_chapter(
                segments=chapter_segments,
                novel_title=novel_title,
                num_questions=n_q,
            )
            all_qas.extend(qas)
            logger.info(f"  Generated {len(qas)} QA pairs")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return all_qas


def evaluate_system_on_novel(
    system: MemorySystem,
    segments: list,
    questions: list,
    answer_generator: AnswerGenerator,
    judge: LLMJudge,
    no_judge: bool = False,
) -> dict:
    """Evaluate a system on novel-based QA."""
    # Ingest all segments
    t0 = time.time()
    system.reset()
    for seg in segments:
        system.remember(
            text=seg.text,
            event_time=seg.timestamp_hint,
            source_name=f"novel_{seg.novel_title}",
        )
    logger.info(
        f"[{system.name}] Ingested {len(segments)} segments in {time.time()-t0:.1f}s"
    )

    results = {}
    for i, q in enumerate(questions):
        if (i + 1) % 20 == 0:
            logger.info(f"[{system.name}] Progress: {i+1}/{len(questions)}")

        # Query
        query_result = system.query(question=q.question)

        # Generate answer
        generated = answer_generator.generate(
            question=q.question,
            retrieved_context=query_result.retrieved_context,
            choices=q.choices if q.choices else None,
        )

        # Judge
        if no_judge or judge is None:
            from src.evaluation.answer_generator import parse_mc_answer
            selected = parse_mc_answer(generated, len(q.choices)) if q.choices else None
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
                "question_id": q.qa_id,
                "system_name": system.name,
                "choices": q.choices,
                "correct_choice_index": q.correct_choice_index,
            }
            judge_result = judge.judge_single(judge_item)
            is_correct = judge_result.get("is_correct", False)

        results[q.qa_id] = {
            "qa_id": q.qa_id,
            "novel_title": q.novel_title,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "question": q.question,
            "gold_answer": q.gold_answer,
            "generated_answer": generated,
            "is_correct": is_correct,
            "system_name": system.name,
            "latency_ms": query_result.latency_ms,
        }

    return results


def compute_metrics(results: dict) -> dict:
    """Compute metrics from novel QA results."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_diff = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = sum(1 for r in results.values() if r["is_correct"])
    total = len(results)

    for r in results.values():
        qt = r.get("question_type", "unknown")
        diff = r.get("difficulty", "unknown")
        by_type[qt]["total"] += 1
        by_diff[diff]["total"] += 1
        if r["is_correct"]:
            by_type[qt]["correct"] += 1
            by_diff[diff]["correct"] += 1

    return {
        "overall_accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "by_type": {k: {"accuracy": v["correct"]/v["total"], **v} for k, v in by_type.items()},
        "by_difficulty": {k: {"accuracy": v["correct"]/v["total"], **v} for k, v in by_diff.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Novel-based evaluation")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--books-dir", default="/home/linkco/exa/llm-usefulEeb/书籍")
    parser.add_argument("--novels", nargs="*", default=None, help="Specific novel files")
    parser.add_argument("--all-novels", action="store_true", help="Use all novels")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--output-dir", default="data/eval_results/novels")
    parser.add_argument("--generate-qa", action="store_true", help="Generate QA pairs via LLM")
    parser.add_argument("--questions-per-chapter", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=100, help="Max QA per novel")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--qa-only", action="store_true", help="Only generate QA, skip evaluation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load(open(base_dir / args.config))

    # Setup loader
    loader = NovelLoader(books_dir=args.books_dir)

    # Select novels
    if args.novels:
        novel_files = args.novels
    elif args.all_novels:
        novel_files = loader.list_books()
        # Prefer novels with narrative structure
        preferred = ["活着.txt", "三体1疯狂年代.txt", "文化苦旅.txt", "从你的全世界路过.txt"]
        novel_files = [f for f in preferred if f in novel_files] + [
            f for f in novel_files if f not in preferred
        ]
    else:
        # Default: use 活着 (short, well-known, good narrative)
        novel_files = ["活着.txt"]

    logger.info(f"Novels: {novel_files}")

    # Setup QA generator
    ag_config = config.get("answer_generator", {})
    qa_generator = NovelQAGenerator(
        api_base=ag_config.get("base_url"),
        api_key=ag_config.get("api_key"),
        model=ag_config.get("model", "deepseek-ai/DeepSeek-V3"),
    )

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate QA pairs
    all_qa = {}
    for novel_file in novel_files:
        novel_stem = Path(novel_file).stem
        qa_path = output_dir / f"{novel_stem}_qa.json"

        if qa_path.exists() and not args.generate_qa:
            logger.info(f"Loading existing QA from {qa_path}")
            with open(qa_path) as f:
                novel_qas = [NovelQA(**q) for q in json.load(f)]
            all_qa[novel_file] = novel_qas
            logger.info(f"  Loaded {len(novel_qas)} QA pairs")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Generating QA for: {novel_file}")
        logger.info(f"{'='*60}")

        novel_qas = generate_qa_for_novel(
            loader=loader,
            novel_file=novel_file,
            qa_generator=qa_generator,
            questions_per_chapter=args.questions_per_chapter,
            max_questions=args.max_questions,
        )

        # Save QA
        with open(qa_path, "w") as f:
            json.dump(
                [vars(q) for q in novel_qas],
                f, ensure_ascii=False, indent=2
            )
        logger.info(f"Saved {len(novel_qas)} QA pairs to {qa_path}")

        all_qa[novel_file] = novel_qas

    if args.qa_only:
        logger.info("QA generation complete. Skipping evaluation.")
        return

    # Phase 2: Evaluate systems
    answer_generator = AnswerGenerator(
        base_url=ag_config.get("base_url"),
        api_key=ag_config.get("api_key"),
        model=ag_config.get("model", "deepseek-ai/DeepSeek-V3"),
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

    # Create systems
    systems = create_systems(config, args.systems)
    if not systems:
        systems = create_systems(config, ["faiss", "bm25"])
    logger.info(f"Systems: {[s.name for s in systems]}")

    all_results = {}
    for novel_file in novel_files:
        novel_stem = Path(novel_file).stem
        novel_qas = all_qa.get(novel_file, [])
        if not novel_qas:
            continue

        segments = loader.load_novel(novel_file)

        for system in systems:
            logger.info(f"\n[{system.name}] Evaluating on '{novel_stem}'...")
            sys_results = evaluate_system_on_novel(
                system=system,
                segments=segments,
                questions=novel_qas,
                answer_generator=answer_generator,
                judge=judge,
                no_judge=args.no_judge or judge is None,
            )
            metrics = compute_metrics(sys_results)

            key = f"{system.name}_{novel_stem}"
            all_results[key] = {"metrics": metrics, "results": sys_results}

            logger.info(
                f"[{key}] Accuracy: {metrics['overall_accuracy']:.3f} "
                f"({metrics['correct']}/{metrics['total']})"
            )

    # Save and print results
    for key, data in all_results.items():
        out_path = output_dir / f"{key}_results.json"
        with open(out_path, "w") as f:
            json.dump(data["results"], f, ensure_ascii=False, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Novel Benchmark Results':^80}")
    print(f"{'='*80}")
    print(f"{'System':<20} {'Novel':<20} {'Overall':>10} {'Temporal':>10} {'L1':>8} {'L2':>8} {'L3':>8}")
    print("-" * 84)

    for key, data in all_results.items():
        m = data["metrics"]
        sys_name, novel = key.rsplit("_", 1)
        temporal = m["by_type"].get("temporal", {}).get("accuracy", 0.0)
        l1 = m["by_difficulty"].get("level_1", {}).get("accuracy", 0.0)
        l2 = m["by_difficulty"].get("level_2", {}).get("accuracy", 0.0)
        l3 = m["by_difficulty"].get("level_3", {}).get("accuracy", 0.0)
        print(f"{sys_name:<20} {novel:<20} {m['overall_accuracy']:>10.3f} {temporal:>10.3f} {l1:>8.3f} {l2:>8.3f} {l3:>8.3f}")

    print(f"{'='*84}\n")


if __name__ == "__main__":
    main()
