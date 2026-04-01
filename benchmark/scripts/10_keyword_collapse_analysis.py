#!/usr/bin/env python3
"""Analyze keyword collapse: which multi-choice questions are trivially solvable?

For each multi-choice QA pair, compare:
1. BM25's answer vs correct answer
2. Whether the correct answer appears in the question text (keyword leak)
3. Whether distractors share keywords with the correct answer
4. Whether temporal reasoning is actually required (vs keyword matching)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data():
    base = Path(__file__).parent.parent
    with open(base / "data/validated/bitpqa_test_zh.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    with open(base / "data/generated/scenarios/all_scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)

    qa_pairs = qa_data["qa_pairs"]
    scenario_map = {s["scenario_id"]: s for s in scenarios}

    # Load BM25 eval results
    bm25_files = sorted((base / "data/eval_results").glob("eval_bm25_*.json"))
    if not bm25_files:
        print("ERROR: No BM25 eval results found")
        sys.exit(1)
    with open(bm25_files[-1]) as f:
        bm25_data = json.load(f)

    return qa_pairs, scenario_map, bm25_data


def analyze_keyword_leak(qa_pairs, scenario_map, bm25_data):
    """Check for keyword leaks in multi-choice questions."""

    # Build BM25 QA results lookup
    bm25_results = {}
    for sr in bm25_data["scenario_results"]:
        for qa in sr["qa_results"]:
            bm25_results[qa["qa_id"]] = qa

    categories = {
        "keyword_leak": [],       # Correct answer text appears in question
        "temporal_keyword": [],   # Question has temporal keywords but answer is in text
        "genuine_temporal": [],   # Requires actual temporal reasoning
        "distractor_overlap": [], # Distractors share keywords with correct answer
    }

    mc_pairs = [qa for qa in qa_pairs if qa.get("answer_type") == "multi_choice"]

    for qa in mc_pairs:
        qid = qa["qa_id"]
        question = qa["question_zh"]
        choices = qa.get("choices", [])
        correct_idx = qa.get("correct_choice_index", 0)
        correct_text = choices[correct_idx] if choices else ""

        bm25_r = bm25_results.get(qid, {})
        bm25_correct = bm25_r.get("is_correct", False)

        analysis = {
            "qa_id": qid,
            "question": question[:80],
            "correct_choice": correct_text,
            "distractors": [c for i, c in enumerate(choices) if i != correct_idx],
            "bm25_correct": bm25_correct,
            "difficulty": qa.get("difficulty", "unknown"),
            "question_type": qa.get("question_type", "unknown"),
            "requires_event_time": qa.get("requires_event_time_reasoning", False),
            "requires_record_time": qa.get("requires_record_time_reasoning", False),
        }

        # Check: does the correct answer appear verbatim in the question?
        correct_in_question = correct_text and any(
            phrase in question for phrase in correct_text.split() if len(phrase) > 1
        )
        # More lenient: any 2-char+ substring of correct answer in question
        correct_substring = False
        if correct_text and len(correct_text) >= 2:
            for i in range(len(correct_text) - 1):
                substr = correct_text[i:i+2]
                if substr in question and len(substr) >= 2:
                    correct_substring = True
                    break

        # Check: do distractors share significant keywords with correct answer?
        distractor_overlap = False
        if correct_text and len(correct_text) >= 2:
            correct_chars = set(correct_text)
            for d in analysis["distractors"]:
                d_chars = set(d)
                overlap = correct_chars & d_chars
                if len(overlap) / max(len(correct_chars), 1) > 0.5:
                    distractor_overlap = True
                    break

        analysis["correct_in_question"] = correct_in_question
        analysis["correct_substring"] = correct_substring
        analysis["distractor_overlap"] = distractor_overlap

        # Categorize
        if correct_substring and bm25_correct:
            categories["keyword_leak"].append(analysis)
        elif distractor_overlap and bm25_correct:
            categories["distractor_overlap"].append(analysis)
        elif bm25_correct:
            categories["temporal_keyword"].append(analysis)
        else:
            categories["genuine_temporal"].append(analysis)

    return categories


def analyze_by_question_type(categories):
    """Break down by question type and difficulty."""

    print("\n## Multi-Choice Question Analysis")
    print("=" * 80)

    total = sum(len(v) for v in categories.values())
    print(f"\nTotal multi-choice QA pairs: {total}")
    print(f"\n### Distribution by Category")
    print(f"{'Category':<25} {'Count':>6} {'%':>6} {'BM25 Acc':>10}")
    print("-" * 50)

    for cat_name, items in categories.items():
        n = len(items)
        pct = n / total * 100 if total else 0
        bm25_acc = sum(1 for i in items if i["bm25_correct"]) / n * 100 if n else 0
        print(f"{cat_name:<25} {n:>6} {pct:>5.1f}% {bm25_acc:>9.1f}%")

    # By difficulty within each category
    print(f"\n### By Difficulty Level")
    for cat_name, items in categories.items():
        if not items:
            continue
        print(f"\n  {cat_name}:")
        for level in ["level_1", "level_2", "level_3"]:
            lvl_items = [i for i in items if i["difficulty"] == level]
            n = len(lvl_items)
            if n > 0:
                acc = sum(1 for i in lvl_items if i["bm25_correct"]) / n * 100
                print(f"    {level}: {n} items, BM25 acc: {acc:.1f}%")

    # By question type
    print(f"\n### By Question Type")
    qt_totals = defaultdict(int)
    qt_bm25_correct = defaultdict(int)
    for cat_name, items in categories.items():
        for item in items:
            qt = item["question_type"]
            qt_totals[qt] += 1
            if item["bm25_correct"]:
                qt_bm25_correct[qt] += 1

    print(f"{'Question Type':<30} {'Total':>6} {'BM25 Correct':>14} {'BM25 Acc':>10}")
    print("-" * 65)
    for qt in sorted(qt_totals.keys()):
        t = qt_totals[qt]
        c = qt_bm25_correct[qt]
        acc = c / t * 100 if t else 0
        print(f"{qt:<30} {t:>6} {c:>14} {acc:>9.1f}%")

    # Analyze "genuine temporal" failures — what makes them hard?
    print(f"\n### Genuine Temporal Failures (BM25 incorrect, not keyword-leaked)")
    genuine_fail = [i for i in categories["genuine_temporal"] if not i["bm25_correct"]]
    genuine_pass = [i for i in categories["genuine_temporal"] if i["bm25_correct"]]

    print(f"  Failed: {len(genuine_fail)}, Passed: {len(genuine_pass)}")
    print(f"\n  Failure by question type:")
    fail_by_qt = defaultdict(int)
    for item in genuine_fail:
        fail_by_qt[item["question_type"]] += 1
    for qt, count in sorted(fail_by_qt.items(), key=lambda x: -x[1]):
        print(f"    {qt}: {count}")

    print(f"\n  Failure by difficulty:")
    fail_by_lvl = defaultdict(int)
    for item in genuine_fail:
        fail_by_lvl[item["difficulty"]] += 1
    for lvl in ["level_1", "level_2", "level_3"]:
        print(f"    {lvl}: {fail_by_lvl.get(lvl, 0)}")

    # Percentage that require event_time and record_time
    print(f"\n### Temporal Reasoning Requirements")
    for cat_name, items in categories.items():
        if not items:
            continue
        n = len(items)
        evt = sum(1 for i in items if i["requires_event_time"]) / n * 100
        rec = sum(1 for i in items if i["requires_record_time"]) / n * 100
        both = sum(1 for i in items if i["requires_event_time"] and i["requires_record_time"]) / n * 100
        print(f"  {cat_name:.<30} event_time: {evt:5.1f}%  record_time: {rec:5.1f}%  both: {both:5.1f}%")


def main():
    qa_pairs, scenario_map, bm25_data = load_data()
    categories = analyze_keyword_leak(qa_pairs, scenario_map, bm25_data)
    analyze_by_question_type(categories)

    # Summary for reviewer
    total_mc = sum(len(v) for v in categories.values())
    keyword_leak_count = len(categories["keyword_leak"])
    print(f"\n{'=' * 80}")
    print(f"KEY FINDING: {keyword_leak_count}/{total_mc} ({keyword_leak_count/total_mc*100:.1f}%) multi-choice questions")
    print(f"have potential keyword leaks (correct answer substrings appear in question).")
    print(f"\nHowever, {len(categories['genuine_temporal'])} questions show no keyword leak pattern.")
    print(f"Of these, BM25 correctly answers {sum(1 for i in categories['genuine_temporal'] if i['bm25_correct'])}")
    print(f"and fails on {sum(1 for i in categories['genuine_temporal'] if not i['bm25_correct'])}.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
