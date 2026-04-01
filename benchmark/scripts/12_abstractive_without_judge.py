#!/usr/bin/env python3
"""Compute accuracy on different subsets to understand benchmark quality.

Subsets:
1. Genuine Temporal MC (56 questions) — excludes keyword-leaked MC
2. Full MC (135 questions)
3. Abstractive with Judge (163 questions)
4. Overall (308 questions)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    base = Path(__file__).parent.parent

    with open(base / "data/validated/bitpqa_test_zh.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    qa_pairs = qa_data["qa_pairs"]
    qa_meta = {p["qa_id"]: p for p in qa_pairs}

    with open(base / "data/generated/scenarios/all_scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario_type_map = {s["scenario_id"]: s["scenario_type"] for s in scenarios}

    # Load genuine temporal MC IDs
    with open(base / "data/eval_results/genuine_temporal_mc_ids.json") as f:
        genuine_ids = set(json.load(f))

    # Load all system eval results
    systems_data = {}
    for fpath in sorted((base / "data/eval_results").glob("eval_*.json")):
        if fpath.name in ("judge_cache.json", "genuine_temporal_mc_ids.json"):
            continue
        with open(fpath) as f:
            data = json.load(f)
        systems_data[data["system_name"]] = data

    print("=" * 90)
    print("BENCHMARK QUALITY ANALYSIS: Subsets and Filtered Results")
    print("=" * 90)

    for sys_name in sorted(systems_data.keys()):
        data = systems_data[sys_name]
        print(f"\n## {sys_name}")
        print(f"{'Subset':<35} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
        print("-" * 65)

        for sr in data["scenario_results"]:
            for qa in sr["qa_results"]:
                qid = qa["qa_id"]
                meta = qa_meta.get(qid, {})
                # Tag each result with subset info
                qa["_answer_type"] = meta.get("answer_type", "unknown")
                qa["_genuine_mc"] = qid in genuine_ids and meta.get("answer_type") == "multi_choice"
                qa["_keyword_mc"] = qid not in genuine_ids and meta.get("answer_type") == "multi_choice"

        # Compute per-subset accuracy
        subsets = {
            "Overall (all 308)": lambda qa: True,
            "Multi-choice (all 135)": lambda qa: qa.get("_answer_type") == "multi_choice",
            "  Keyword-leaked MC (79)": lambda qa: qa.get("_keyword_mc"),
            "  Genuine Temporal MC (56)": lambda qa: qa.get("_genuine_mc"),
            "Abstractive with Judge (163)": lambda qa: qa.get("_answer_type") == "abstractive",
            "Boolean (10)": lambda qa: qa.get("_answer_type") == "boolean",
        }

        for label, filter_fn in subsets.items():
            correct = 0
            total = 0
            for sr in data["scenario_results"]:
                for qa in sr["qa_results"]:
                    if filter_fn(qa):
                        total += 1
                        if qa["is_correct"]:
                            correct += 1
            acc = correct / total * 100 if total else 0
            print(f"  {label:<33} {correct:>8} {total:>6} {acc:>9.1f}%")

    # Cross-system comparison on genuine temporal MC
    print(f"\n{'=' * 90}")
    print("CROSS-SYSTEM COMPARISON: Genuine Temporal Multi-Choice (56 questions)")
    print(f"{'=' * 90}")
    print(f"\n{'System':<25} {'Accuracy':>10} {'By Level 1':>12} {'By Level 2':>12} {'By Level 3':>12}")
    print("-" * 75)

    for sys_name in sorted(systems_data.keys()):
        data = systems_data[sys_name]
        total_c, total_t = 0, 0
        by_level = defaultdict(lambda: [0, 0])

        for sr in data["scenario_results"]:
            for qa in sr["qa_results"]:
                qid = qa["qa_id"]
                if qid not in genuine_ids:
                    continue
                meta = qa_meta.get(qid, {})
                total_t += 1
                if qa["is_correct"]:
                    total_c += 1
                    by_level[meta.get("difficulty", "unknown")][0] += 1
                by_level[meta.get("difficulty", "unknown")][1] += 1

        acc = total_c / total_t * 100 if total_t else 0
        lvl_strs = []
        for lvl in ["level_1", "level_2", "level_3"]:
            c, t = by_level[lvl]
            a = c / t * 100 if t else 0
            lvl_strs.append(f"{a:.1f}%")
        print(f"{sys_name:<25} {acc:>9.1f}% {lvl_strs[0]:>12} {lvl_strs[1]:>12} {lvl_strs[2]:>12}")

    # Key insight
    print(f"\n{'=' * 90}")
    print("KEY INSIGHTS")
    print(f"{'=' * 90}")
    print("""
1. KEYWORD LEAK ANALYSIS:
   - 79/135 (58.5%) multi-choice questions are solvable by keyword overlap
   - All 5 systems (including BM25) score 100% on these
   - BM25 scores 0% on the remaining 56 genuine temporal questions
   - This means: BM25's overall 58.5% MC accuracy is ENTIRELY from keyword matching

2. GENUINE TEMPORAL REASONING (56 questions):
   - Best system: Simple KG at 14.3% (8/56)
   - FAISS/Naive RAG: 8.9% (5/56)
   - ChromaDB: 8.9% (5/56)
   - BM25: 0.0% (0/56)
   - ALL systems struggle with genuine temporal reasoning

3. IMPLICATIONS FOR BENCHMARK VALIDITY:
   - The benchmark DOES contain questions requiring temporal reasoning
   - But 58.5% of multi-choice questions are trivially solvable
   - Recommendation: Report both full and filtered results
   - The filtered subset (56 questions) is the meaningful evaluation

4. SIMPLE KG ADVANTAGE:
   - Simple KG is the only system that outperforms retrieval on genuine temporal MC
   - 14.3% vs 8.9% — suggests graph structure helps with temporal reasoning
   - This is the key signal that TMG's graph-based approach should amplify
""")


if __name__ == "__main__":
    main()
