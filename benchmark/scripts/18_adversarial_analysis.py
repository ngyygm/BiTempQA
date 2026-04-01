#!/usr/bin/env python3
"""Analyze LoCoMo adversarial questions to explain Simple KG's 97.5% accuracy."""

import json
from collections import defaultdict
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    locomo_dir = base_dir / "data" / "eval_results" / "locomo"

    # Load all system results
    systems = {}
    for f in locomo_dir.glob("locomo_*_results.json"):
        if "summary" in f.name:
            continue
        name = f.stem.replace("locomo_", "").replace("_results", "")
        systems[name] = json.loads(f.read_text(encoding="utf-8"))

    # Find adversarial questions
    all_qids = set()
    for name, data in systems.items():
        all_qids.update(data.keys())

    adversarial = {}
    for qid in all_qids:
        types = set()
        for name, data in systems.items():
            if qid in data:
                types.add(data[qid].get("question_type", ""))
        if "adversarial" in types:
            adversarial[qid] = types

    print(f"Total adversarial questions: {len(adversarial)}")
    print(f"Systems: {list(systems.keys())}")

    # Per-system accuracy on adversarial
    print("\n## Adversarial Accuracy by System")
    for name, data in systems.items():
        correct = sum(1 for qid in adversarial if qid in data and data[qid]["is_correct"])
        total = sum(1 for qid in adversarial if qid in data)
        print(f"  {name}: {correct}/{total} = {correct/total*100:.1f}%")

    # Find questions where Simple KG is correct but others are wrong
    print("\n## Adversarial Questions: Simple KG correct, others wrong (sample of 10)")
    kg_data = systems.get("simple kg", {})
    count = 0
    for qid in sorted(adversarial.keys()):
        if qid not in kg_data or not kg_data[qid]["is_correct"]:
            continue
        # Check if at least 2 other systems got it wrong
        wrong_systems = []
        for name, data in systems.items():
            if name == "simple kg":
                continue
            if qid in data and not data[qid]["is_correct"]:
                wrong_systems.append(name)
        if len(wrong_systems) >= 2:
            q = kg_data[qid]
            print(f"\n  [{qid}]")
            print(f"    Q: {q['question'][:120]}")
            print(f"    Gold: {q['gold_answer'][:80]}")
            print(f"    KG chose: {q['generated_answer']} (correct_idx={q['correct_choice_index']})")
            for ws in wrong_systems:
                wd = systems[ws].get(qid, {})
                print(f"    {ws} chose: {wd.get('generated_answer', 'N/A')} (correct_idx={wd.get('correct_choice_index', 'N/A')})")
            count += 1
            if count >= 10:
                break

    # Find questions where Simple KG is wrong
    print("\n## Adversarial Questions: Simple KG WRONG (all failures)")
    count = 0
    for qid in sorted(adversarial.keys()):
        if qid not in kg_data or kg_data[qid]["is_correct"]:
            continue
        q = kg_data[qid]
        print(f"\n  [{qid}]")
        print(f"    Q: {q['question'][:120]}")
        print(f"    Gold: {q['gold_answer'][:80]}")
        print(f"    KG chose: {q['generated_answer']} (correct_idx={q['correct_choice_index']})")
        count += 1

    # Cross-system agreement analysis
    print("\n## Cross-System Agreement on Adversarial Questions")
    for qid in sorted(list(adversarial.keys())[:50]):
        answers = {}
        for name, data in systems.items():
            if qid in data:
                answers[name] = data[qid]["is_correct"]
        correct_names = [n for n, c in answers.items() if c]
        wrong_names = [n for n, c in answers.items() if not c]
        all_correct = len(correct_names) == len(systems)
        all_wrong = len(wrong_names) == len(systems)
        kg_only = len(correct_names) == 1 and "simple kg" in correct_names
        if all_correct or all_wrong or kg_only:
            flag = "ALL-CORRECT" if all_correct else "ALL-WRONG" if all_wrong else "KG-ONLY"
            q_text = systems[list(systems.keys())[0]][qid]["question"][:80]
            print(f"  [{flag}] {qid}: {q_text}")

    # Summary statistics
    print("\n## Adversarial Error Analysis Summary")
    # Count: how many adversarial questions each system gets uniquely right/wrong
    for name, data in systems.items():
        only_right = 0
        also_right = 0
        for qid in adversarial:
            if qid not in data or not data[qid]["is_correct"]:
                continue
            others_correct = sum(
                1 for n, d in systems.items()
                if n != name and qid in d and d[qid]["is_correct"]
            )
            if others_correct == 0:
                only_right += 1
            else:
                also_right += 1
        print(f"  {name}: uniquely correct={only_right}, correct with others={also_right}")


if __name__ == "__main__":
    main()
