#!/usr/bin/env python3
"""Evaluate all systems on the 'genuine temporal' subset of multi-choice questions.

This subset excludes questions where keyword matching alone can solve them,
providing a cleaner measure of temporal reasoning ability.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def identify_genuine_mc(qa_pairs, bm25_data):
    """Identify multi-choice QA pairs that are not keyword-solvable."""

    # Build BM25 results lookup
    bm25_results = {}
    for sr in bm25_data["scenario_results"]:
        for qa in sr["qa_results"]:
            bm25_results[qa["qa_id"]] = qa

    genuine_ids = set()

    for qa in qa_pairs:
        if qa.get("answer_type") != "multi_choice":
            continue
        qid = qa["qa_id"]
        question = qa["question_zh"]
        choices = qa.get("choices", [])
        correct_idx = qa.get("correct_choice_index", 0)
        correct_text = choices[correct_idx] if choices else ""

        bm25_r = bm25_results.get(qid, {})
        bm25_correct = bm25_r.get("is_correct", False)

        if bm25_correct:
            # BM25 solved it — check for keyword leak
            correct_substring = False
            if correct_text and len(correct_text) >= 2:
                for i in range(len(correct_text) - 1):
                    substr = correct_text[i:i+2]
                    if substr in question and len(substr) >= 2:
                        correct_substring = True
                        break

            # Check distractor overlap
            distractor_overlap = False
            if correct_text and len(correct_text) >= 2:
                correct_chars = set(correct_text)
                for j, c in enumerate(choices):
                    if j == correct_idx:
                        continue
                    d_chars = set(c)
                    overlap = correct_chars & d_chars
                    if len(overlap) / max(len(correct_chars), 1) > 0.5:
                        distractor_overlap = True
                        break

            if correct_substring or distractor_overlap:
                continue  # Keyword-leaked, skip

        # BM25 failed OR BM25 passed without keyword leak → genuine
        # Actually, we want questions where BM25 specifically FAILED
        # because those are the ones that require genuine temporal reasoning
        if not bm25_correct:
            genuine_ids.add(qid)

    return genuine_ids


def evaluate_on_subset(systems_data, genuine_ids, qa_pairs, scenario_type_map):
    """Evaluate each system on the genuine temporal subset."""

    qa_meta = {p["qa_id"]: p for p in qa_pairs}

    print("=" * 80)
    print("EVALUATION ON GENUINE TEMPORAL MULTI-CHOICE SUBSET")
    print("=" * 80)
    print(f"\nSubset size: {len(genuine_ids)} / 135 multi-choice questions")
    print(f"(Excluded: {135 - len(genuine_ids)} questions solvable by keyword matching)")
    print()

    results = {}
    for sys_name, data in systems_data.items():
        correct = 0
        total = 0
        by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
        by_qtype = defaultdict(lambda: {"correct": 0, "total": 0})
        by_scenario = defaultdict(lambda: {"correct": 0, "total": 0})

        for sr in data["scenario_results"]:
            stype = scenario_type_map.get(sr["scenario_id"], "unknown")
            for qa in sr["qa_results"]:
                qid = qa["qa_id"]
                if qid not in genuine_ids:
                    continue
                meta = qa_meta.get(qid, {})
                total += 1
                if qa["is_correct"]:
                    correct += 1
                    by_difficulty[meta.get("difficulty", "unknown")]["correct"] += 1
                    by_scenario[stype]["correct"] += 1
                    by_qtype[meta.get("question_type", "unknown")]["correct"] += 1
                by_difficulty[meta.get("difficulty", "unknown")]["total"] += 1
                by_scenario[stype]["total"] += 1
                by_qtype[meta.get("question_type", "unknown")]["total"] += 1

        acc = correct / total * 100 if total else 0
        results[sys_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "by_difficulty": dict(by_difficulty),
            "by_qtype": dict(by_qtype),
            "by_scenario": dict(by_scenario),
        }

    # Print results
    print(f"{'System':<25} {'Accuracy':>10} {'Correct':>8} {'Total':>6}")
    print("-" * 55)
    for sys_name in sorted(results.keys()):
        r = results[sys_name]
        print(f"{sys_name:<25} {r['accuracy']:>9.1f}% {r['correct']:>8} {r['total']:>6}")

    # By difficulty
    print(f"\n### By Difficulty Level")
    for level in ["level_1", "level_2", "level_3"]:
        print(f"\n  {level}:")
        print(f"  {'System':<25} {'Accuracy':>10}")
        for sys_name in sorted(results.keys()):
            d = results[sys_name]["by_difficulty"].get(level, {"correct": 0, "total": 0})
            acc = d["correct"] / d["total"] * 100 if d["total"] else 0
            print(f"  {sys_name:<25} {acc:>9.1f}% ({d['correct']}/{d['total']})")

    # By question type
    print(f"\n### By Question Type")
    all_qtypes = set()
    for sys_name in results:
        all_qtypes.update(results[sys_name]["by_qtype"].keys())
    for qt in sorted(all_qtypes):
        print(f"\n  {qt}:")
        for sys_name in sorted(results.keys()):
            d = results[sys_name]["by_qtype"].get(qt, {"correct": 0, "total": 0})
            acc = d["correct"] / d["total"] * 100 if d["total"] else 0
            print(f"    {sys_name:<25} {acc:>9.1f}% ({d['correct']}/{d['total']})")

    # Comparison with full multi-choice
    print(f"\n### Comparison: Full MC vs Genuine Temporal MC")
    print(f"{'System':<25} {'Full MC':>10} {'Genuine MC':>12} {'Delta':>8}")
    print("-" * 60)

    base = Path(__file__).parent.parent
    for sys_name in sorted(results.keys()):
        # Load full eval data for MC accuracy
        full_mc_correct = 0
        full_mc_total = 0
        eval_files = sorted((base / "data/eval_results").glob(f"eval_{sys_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_*.json"))
        if not eval_files:
            # Try other patterns
            eval_files = sorted((base / "data/eval_results").glob("eval_*.json"))
        for fpath in eval_files:
            with open(fpath) as f:
                ed = json.load(f)
            if ed["system_name"] != sys_name:
                continue
            for sr in ed["scenario_results"]:
                for qa in sr["qa_results"]:
                    meta = qa_meta.get(qa["qa_id"], {})
                    if meta.get("answer_type") == "multi_choice":
                        full_mc_total += 1
                        if qa["is_correct"]:
                            full_mc_correct += 1
            break

        full_acc = full_mc_correct / full_mc_total * 100 if full_mc_total else 0
        genuine_acc = results[sys_name]["accuracy"]
        delta = genuine_acc - full_acc
        print(f"{sys_name:<25} {full_acc:>9.1f}% {genuine_acc:>11.1f}% {delta:>+7.1f}%")

    return results


def main():
    base = Path(__file__).parent.parent

    with open(base / "data/validated/bitpqa_test_zh.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    qa_pairs = qa_data["qa_pairs"]

    with open(base / "data/generated/scenarios/all_scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario_type_map = {s["scenario_id"]: s["scenario_type"] for s in scenarios}

    # Load BM25 results
    bm25_files = sorted((base / "data/eval_results").glob("eval_bm25_*.json"))
    with open(bm25_files[-1]) as f:
        bm25_data = json.load(f)

    genuine_ids = identify_genuine_mc(qa_pairs, bm25_data)

    # Save genuine IDs for future use
    with open(base / "data/eval_results/genuine_temporal_mc_ids.json", "w") as f:
        json.dump(sorted(genuine_ids), f, indent=2)

    # Load all system results
    systems_data = {}
    for fpath in sorted((base / "data/eval_results").glob("eval_*.json")):
        if fpath.name == "judge_cache.json":
            continue
        with open(fpath) as f:
            data = json.load(f)
        systems_data[data["system_name"]] = data

    evaluate_on_subset(systems_data, genuine_ids, qa_pairs, scenario_type_map)


if __name__ == "__main__":
    main()
