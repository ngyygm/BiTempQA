#!/usr/bin/env python3
"""Statistical analysis on the genuine temporal MC subset (56 questions).

Computes McNemar's tests and bootstrap CIs for the filtered evaluation.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def mcnemar_test(a_correct, b_correct):
    """McNemar's test with continuity correction."""
    assert len(a_correct) == len(b_correct)
    a_correct = np.array(a_correct)
    b_correct = np.array(b_correct)
    b_count = int(np.sum((a_correct == 1) & (b_correct == 0)))
    c_count = int(np.sum((a_correct == 0) & (b_correct == 1)))
    if b_count + c_count == 0:
        return 1.0
    from scipy import stats
    statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return p_value


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Bootstrap CI for accuracy."""
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(data)), lower, upper


def main():
    base = Path(__file__).parent.parent

    with open(base / "data/validated/bitpqa_test_zh.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    qa_meta = {p["qa_id"]: p for p in qa_data["qa_pairs"]}

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

    # Build per-system correct arrays for genuine temporal MC
    print("=" * 80)
    print("STATISTICAL ANALYSIS: Genuine Temporal Multi-Choice (56 questions)")
    print("=" * 80)

    system_correct = {}
    for sys_name, data in systems_data.items():
        correct = []
        for sr in data["scenario_results"]:
            for qa in sr["qa_results"]:
                if qa["qa_id"] in genuine_ids:
                    correct.append(1 if qa["is_correct"] else 0)
        system_correct[sys_name] = np.array(correct)

    # 1. Bootstrap CIs
    print("\n## 1. Bootstrap 95% CIs (10,000 resamples)")
    print("-" * 60)
    for sn in sorted(system_correct.keys()):
        mean, lo, hi = bootstrap_ci(system_correct[sn])
        n_correct = int(np.sum(system_correct[sn]))
        print(f"  {sn:<25}: {mean:.4f} [{lo:.4f}, {hi:.4f}] ({n_correct}/56)")

    # 2. McNemar's pairwise tests
    print("\n## 2. McNemar's Test — Pairwise Comparisons")
    print("-" * 60)
    sys_names = sorted(system_correct.keys())
    for i, s1 in enumerate(sys_names):
        for j, s2 in enumerate(sys_names):
            if i < j:
                p = mcnemar_test(system_correct[s1], system_correct[s2])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                delta = np.mean(system_correct[s1]) - np.mean(system_correct[s2])
                b_count = int(np.sum((system_correct[s1] == 1) & (system_correct[s2] == 0)))
                c_count = int(np.sum((system_correct[s1] == 0) & (system_correct[s2] == 1)))
                print(f"  {s1:<25} vs {s2:<25}: p={p:.4f} {sig}  (Δ={delta:+.4f}, b={b_count}, c={c_count})")

    # 3. Difficulty level analysis
    print("\n## 3. Per-Difficulty Level — Bootstrap CIs")
    print("-" * 60)
    for level in ["level_1", "level_2", "level_3"]:
        print(f"\n  {level}:")
        for sn in sys_names:
            correct_for_level = []
            data = systems_data[sn]
            for sr in data["scenario_results"]:
                for qa in sr["qa_results"]:
                    if qa["qa_id"] in genuine_ids:
                        meta = qa_meta.get(qa["qa_id"], {})
                        if meta.get("difficulty") == level:
                            correct_for_level.append(1 if qa["is_correct"] else 0)
            if correct_for_level:
                arr = np.array(correct_for_level)
                mean, lo, hi = bootstrap_ci(arr)
                print(f"    {sn:<25}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (n={len(correct_for_level)})")

    # 4. Question type analysis
    print("\n## 4. Per-Question-Type — Accuracy")
    print("-" * 60)
    for sn in sys_names:
        data = systems_data[sn]
        qt_acc = {}
        for sr in data["scenario_results"]:
            for qa in sr["qa_results"]:
                if qa["qa_id"] in genuine_ids:
                    meta = qa_meta.get(qa["qa_id"], {})
                    qt = meta.get("question_type", "unknown")
                    if qt not in qt_acc:
                        qt_acc[qt] = {"correct": 0, "total": 0}
                    qt_acc[qt]["total"] += 1
                    if qa["is_correct"]:
                        qt_acc[qt]["correct"] += 1
        print(f"\n  {sn}:")
        for qt in sorted(qt_acc.keys()):
            c, t = qt_acc[qt]["correct"], qt_acc[qt]["total"]
            print(f"    {qt:<25}: {c}/{t} = {c/t*100:.1f}%")

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
