#!/usr/bin/env python3
"""Statistical significance analysis for BiTempQA benchmark results.

Computes bootstrap confidence intervals, McNemar's tests, and
discriminative power analysis per scenario type and question type.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95):
    """Compute bootstrap confidence interval for mean accuracy."""
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(data)), lower, upper


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray):
    """McNemar's test for paired binary outcomes with continuity correction."""
    assert len(a_correct) == len(b_correct)
    b_count = int(np.sum((a_correct == 1) & (b_correct == 0)))
    c_count = int(np.sum((a_correct == 0) & (b_correct == 1)))
    if b_count + c_count == 0:
        return 1.0
    from scipy import stats
    statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return p_value


def load_results(results_dir: Path, qa_path: Path):
    """Load evaluation results and QA metadata."""
    with open(qa_path, encoding='utf-8') as f:
        qa_dataset = json.load(f)
    qa_meta = {pair['qa_id']: pair for pair in qa_dataset['qa_pairs']}

    # Load scenario types
    scenarios_path = results_dir.parent / 'generated' / 'scenarios' / 'all_scenarios.json'
    with open(scenarios_path, encoding='utf-8') as f:
        all_scenarios = json.load(f)
    scenario_type_map = {s['scenario_id']: s['scenario_type'] for s in all_scenarios}

    # Load all eval result files
    systems_data = {}
    for fpath in sorted(results_dir.glob('eval_*.json')):
        with open(fpath) as f:
            data = json.load(f)
        systems_data[data['system_name']] = data

    return qa_meta, scenario_type_map, systems_data


def build_correct_arrays(systems_data, qa_meta, scenario_type_map, filter_fn=None):
    """Build per-system binary correct arrays with optional filtering."""
    system_correct = {}
    for sys_name, data in systems_data.items():
        correct = []
        for sr in data['scenario_results']:
            for qa in sr['qa_results']:
                meta = qa_meta.get(qa['qa_id'], {})
                record = {
                    'correct': 1 if qa['is_correct'] else 0,
                    'answer_type': meta.get('answer_type', 'unknown'),
                    'difficulty': meta.get('difficulty', 'unknown'),
                    'question_type': meta.get('question_type', 'unknown'),
                    'scenario_type': scenario_type_map.get(sr['scenario_id'], 'unknown'),
                    'requires_version': meta.get('requires_version_tracking', False),
                    'requires_event_time': meta.get('requires_event_time_reasoning', False),
                    'requires_record_time': meta.get('requires_record_time_reasoning', False),
                }
                if filter_fn is None or filter_fn(record):
                    correct.append(record['correct'])
        system_correct[sys_name] = np.array(correct)
    return system_correct


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for BiTempQA")
    parser.add_argument("--results-dir", default="data/eval_results")
    parser.add_argument("--qa", default="data/validated/bitpqa_test_zh.json")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    qa_meta, scenario_type_map, systems_data = load_results(
        base_dir / args.results_dir, base_dir / args.qa
    )
    sys_names = sorted(systems_data.keys())

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS — BiTempQA Benchmark")
    print("=" * 80)

    # 1. Overall bootstrap CIs
    print("\n## 1. Overall Accuracy — Bootstrap 95% CIs (10,000 resamples)")
    print("-" * 60)
    correct_all = build_correct_arrays(systems_data, qa_meta, scenario_type_map)
    for sn in sys_names:
        mean, lo, hi = bootstrap_ci(correct_all[sn])
        print(f"  {sn:<30}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (n={len(correct_all[sn])})")

    # 2. McNemar pairwise tests
    print("\n## 2. McNemar's Test — Pairwise Comparisons (Overall)")
    print("-" * 60)
    for i, s1 in enumerate(sys_names):
        for j, s2 in enumerate(sys_names):
            if i < j:
                p = mcnemar_test(correct_all[s1], correct_all[s2])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                delta = np.mean(correct_all[s1]) - np.mean(correct_all[s2])
                print(f"  {s1:<30} vs {s2:<30}: p={p:.6f} {sig}  (Δ={delta:+.4f})")

    # 3. Multi-choice only
    print("\n## 3. Multi-Choice Only — Bootstrap CIs + McNemar")
    print("-" * 60)
    mc_correct = build_correct_arrays(systems_data, qa_meta, scenario_type_map,
                                       filter_fn=lambda r: r['answer_type'] == 'multi_choice')
    for sn in sys_names:
        mean, lo, hi = bootstrap_ci(mc_correct[sn])
        print(f"  {sn:<30}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (n={len(mc_correct[sn])})")

    print("\n  Pairwise McNemar (multi-choice):")
    for i, s1 in enumerate(sys_names):
        for j, s2 in enumerate(sys_names):
            if i < j:
                p = mcnemar_test(mc_correct[s1], mc_correct[s2])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                delta = np.mean(mc_correct[s1]) - np.mean(mc_correct[s2])
                print(f"    {s1:<30} vs {s2:<30}: p={p:.6f} {sig}  (Δ={delta:+.4f})")

    # 4. Per-difficulty-level
    print("\n## 4. Per-Difficulty Level — Bootstrap CIs")
    print("-" * 60)
    for level in ['level_1', 'level_2', 'level_3']:
        print(f"\n  {level}:")
        lvl_correct = build_correct_arrays(systems_data, qa_meta, scenario_type_map,
                                            filter_fn=lambda r, l=level: r['difficulty'] == l)
        for sn in sys_names:
            if len(lvl_correct[sn]) > 0:
                mean, lo, hi = bootstrap_ci(lvl_correct[sn])
                print(f"    {sn:<30}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (n={len(lvl_correct[sn])})")

    # 5. Discriminative power per scenario type
    print("\n## 5. Discriminative Power per Scenario Type")
    print("-" * 60)
    stype_sys_acc = defaultdict(lambda: defaultdict(list))
    for sys_name, data in systems_data.items():
        for sr in data['scenario_results']:
            sid = sr['scenario_id']
            stype = scenario_type_map.get(sid, 'unknown')
            n_qa = len(sr['qa_results'])
            n_correct = sum(1 for qa in sr['qa_results'] if qa['is_correct'])
            stype_sys_acc[stype][sys_name].append(n_correct / n_qa if n_qa > 0 else 0)

    print(f"\n  {'Scenario Type':<45} {'Range':>8} {'Mean Std':>10} {'Discriminative':>15}")
    print("  " + "-" * 85)
    for stype in sorted(stype_sys_acc.keys()):
        all_accs = []
        for sn in sys_names:
            all_accs.extend(stype_sys_acc[stype][sn])
        if not all_accs:
            continue

        # Compute per-scenario mean accuracy per system
        sys_means = []
        for sn in sys_names:
            accs = stype_sys_acc[stype][sn]
            if accs:
                sys_means.append(np.mean(accs))

        if len(sys_means) >= 2:
            range_acc = max(sys_means) - min(sys_means)
            std_acc = np.std(sys_means)
            disc = "High" if range_acc > 0.10 else "Medium" if range_acc > 0.05 else "Low"
            print(f"  {stype:<45} {range_acc:>8.3f} {std_acc:>10.3f} {disc:>15}")

    # 6. Version tracking analysis
    print("\n## 6. Version Tracking — Bootstrap CIs")
    print("-" * 60)
    for label, fn in [("Requires version", lambda r: r['requires_version']),
                      ("No version", lambda r: not r['requires_version'])]:
        print(f"\n  {label}:")
        v_correct = build_correct_arrays(systems_data, qa_meta, scenario_type_map, filter_fn=fn)
        for sn in sys_names:
            if len(v_correct[sn]) > 0:
                mean, lo, hi = bootstrap_ci(v_correct[sn])
                print(f"    {sn:<30}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (n={len(v_correct[sn])})")

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
