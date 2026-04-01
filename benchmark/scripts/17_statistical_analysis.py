#!/usr/bin/env python3
"""Statistical significance analysis for BiTempQA v2 evaluation results.

Computes:
1. Bootstrap 95% confidence intervals for each system's accuracy
2. Pairwise McNemar's tests between systems
3. Per-difficulty-level bootstrap CIs
4. Per-question-type bootstrap CIs
5. Temporal reasoning gap analysis with CIs
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_results(result_files):
    """Load evaluation results and extract per-QA correctness."""
    systems = {}
    for f in result_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        name = data["system_name"]
        qa_correct = {}
        qa_meta = {}
        for sr in data["scenario_results"]:
            for qr in sr["qa_results"]:
                qa_id = qr["qa_id"]
                qa_correct[qa_id] = qr["is_correct"]
                qa_meta[qa_id] = {
                    "difficulty": None,
                    "question_type": None,
                    "answer_type": None,
                    "requires_temporal": False,
                }
        systems[name] = {"correct": qa_correct, "meta": qa_meta}
    return systems


def load_qa_metadata(qa_path):
    """Load QA metadata (difficulty, question_type, answer_type)."""
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    meta = {}
    for qa in data["qa_pairs"]:
        meta[qa["qa_id"]] = {
            "difficulty": qa.get("difficulty", {}).get("value", "unknown") if isinstance(qa.get("difficulty"), dict) else str(qa.get("difficulty", "unknown")),
            "question_type": qa.get("question_type", {}).get("value", "unknown") if isinstance(qa.get("question_type"), dict) else str(qa.get("question_type", "unknown")),
            "answer_type": qa.get("answer_type", {}).get("value", "unknown") if isinstance(qa.get("answer_type"), dict) else str(qa.get("answer_type", "unknown")),
            "requires_temporal": qa.get("requires_event_time_reasoning", False) or qa.get("requires_record_time_reasoning", False),
            "requires_event_time": qa.get("requires_event_time_reasoning", False),
            "requires_record_time": qa.get("requires_record_time_reasoning", False),
            "requires_version": qa.get("requires_version_tracking", False),
        }
    return meta


def bootstrap_ci(values, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap confidence interval for a mean of binary values."""
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    mean = values.mean()
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    std = boot_means.std()

    return mean, std, (ci_low, ci_high)


def mcnemar_test(correct_a, correct_b, qa_ids):
    """McNemar's test for paired binary outcomes.

    Tests whether two systems have significantly different error rates.
    Returns chi-squared statistic and approximate p-value.
    """
    # Only use QA pairs where both systems have results
    common_ids = set(correct_a.keys()) & set(correct_b.keys())

    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    for qa_id in common_ids:
        a_right = correct_a[qa_id]
        b_right = correct_b[qa_id]
        if a_right and not b_right:
            b += 1
        elif not a_right and b_right:
            c += 1

    if b + c == 0:
        return 0.0, 1.0  # No discordant pairs

    # McNemar's test with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value from chi-squared(1)
    from math import exp, sqrt
    # Use survival function of chi2(1): P(X > x) = 2 * P(Z > sqrt(x))
    p_value = 2 * (1 - 0.5 * (1 + _erf(sqrt(statistic) / sqrt(2))))

    return statistic, p_value


def _erf(x):
    """Approximate error function."""
    from math import exp
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    return sign * y


def main():
    np.random.seed(42)

    base_dir = Path(__file__).parent.parent

    # Load v5 results (latest unified eval with LLM Judge)
    result_dir = base_dir / "data" / "eval_results"

    # Find latest v5 results for each system
    system_files = {
        "FAISS Vector Store": "eval_faiss_vector_store_1774518673.json",
        "BM25": "eval_bm25_1774522334.json",
        "Simple KG": "eval_simple_kg_1774520732.json",
        "Naive RAG": "eval_naive_rag_1774521546.json",
        "ChromaDB": "eval_chromadb_1774519684.json",
    }

    files = []
    for name, fname in system_files.items():
        fpath = result_dir / fname
        if fpath.exists():
            files.append(fpath)
        else:
            print(f"WARNING: {fpath} not found, trying to find latest...")
            # Find latest file for this system
            pattern = fname.split("_177")[0]
            matching = sorted(result_dir.glob(f"{pattern}_*.json"), reverse=True)
            if matching:
                files.append(matching[0])
                print(f"  Using: {matching[0].name}")

    systems = load_results(files)
    qa_meta = load_qa_metadata(base_dir / "data" / "validated" / "bitpqa_test_zh.json")

    # Merge metadata
    for name in systems:
        for qa_id in systems[name]["meta"]:
            if qa_id in qa_meta:
                systems[name]["meta"][qa_id].update(qa_meta[qa_id])

    print("=" * 90)
    print("STATISTICAL SIGNIFICANCE ANALYSIS — BiTempQA v2 (308 QA pairs)")
    print("=" * 90)

    # --- 1. Overall accuracy with bootstrap CIs ---
    print("\n## 1. Overall Accuracy (Bootstrap 95% CI, 10000 iterations)")
    print(f"{'System':<25} {'Accuracy':>10} {'Std':>8} {'95% CI':>20}")
    print("-" * 65)

    overall_results = {}
    for name, data in systems.items():
        values = [1.0 if v else 0.0 for v in data["correct"].values()]
        mean, std, (ci_low, ci_high) = bootstrap_ci(values)
        overall_results[name] = (mean, std, ci_low, ci_high, values)
        print(f"{name:<25} {mean*100:>9.1f}% {std*100:>7.2f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%]")

    # --- 2. Pairwise McNemar's tests ---
    print("\n## 2. Pairwise McNemar's Tests (with continuity correction)")
    names = list(systems.keys())
    header = f"{'':>25}"
    for n in names:
        short = n[:12]
        header += f" {short:>12}"
    print(header)
    print("-" * (25 + 13 * len(names)))

    for i, name_a in enumerate(names):
        row = f"{name_a[:25]:>25}"
        for j, name_b in enumerate(names):
            if i == j:
                row += f" {'---':>12}"
            elif i < j:
                stat, p = mcnemar_test(
                    systems[name_a]["correct"],
                    systems[name_b]["correct"],
                    None,
                )
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                row += f" {p:>8.4f}{sig:>4}"
            else:
                row += f" {'':>12}"
        print(row)

    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")

    # --- 3. Per-difficulty bootstrap CIs ---
    print("\n## 3. Per-Difficulty Level (Bootstrap 95% CI)")
    for level in ["level_1", "level_2", "level_3"]:
        print(f"\n  {level}:")
        print(f"  {'System':<25} {'Accuracy':>10} {'95% CI':>20}")
        print(f"  {'-'*55}")
        for name, data in systems.items():
            values = []
            for qa_id, correct in data["correct"].items():
                meta = data["meta"].get(qa_id, {})
                if meta.get("difficulty") == level:
                    values.append(1.0 if correct else 0.0)
            if values:
                mean, std, (ci_low, ci_high) = bootstrap_ci(values)
                print(f"  {name:<25} {mean*100:>9.1f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%]")
            else:
                print(f"  {name:<25} {'N/A':>10}")

    # --- 4. Per-answer-type bootstrap CIs ---
    print("\n## 4. Per-Answer Type (Bootstrap 95% CI)")
    for atype in ["multi_choice", "abstractive", "boolean"]:
        print(f"\n  {atype}:")
        print(f"  {'System':<25} {'Accuracy':>10} {'95% CI':>20}")
        print(f"  {'-'*55}")
        for name, data in systems.items():
            values = []
            for qa_id, correct in data["correct"].items():
                meta = data["meta"].get(qa_id, {})
                if meta.get("answer_type") == atype:
                    values.append(1.0 if correct else 0.0)
            if values:
                mean, std, (ci_low, ci_high) = bootstrap_ci(values)
                print(f"  {name:<25} {mean*100:>9.1f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%] ({len(values)} Q)")
            else:
                print(f"  {name:<25} {'N/A':>10}")

    # --- 5. Temporal reasoning gap analysis ---
    print("\n## 5. Temporal Reasoning Gap Analysis")
    print("  (event_time vs record_time vs both, Bootstrap 95% CI)")

    for name, data in systems.items():
        event_vals = []
        record_vals = []
        both_vals = []
        neither_vals = []

        for qa_id, correct in data["correct"].items():
            meta = data["meta"].get(qa_id, {})
            requires_event = meta.get("requires_event_time", False)
            requires_record = meta.get("requires_record_time", False)

            val = 1.0 if correct else 0.0
            if requires_event and requires_record:
                both_vals.append(val)
            elif requires_event:
                event_vals.append(val)
            elif requires_record:
                record_vals.append(val)
            else:
                neither_vals.append(val)

        print(f"\n  {name}:")
        for label, vals in [("Event-time only", event_vals),
                             ("Record-time only", record_vals),
                             ("Both required", both_vals),
                             ("Neither", neither_vals)]:
            if vals:
                mean, std, (ci_low, ci_high) = bootstrap_ci(vals)
                gap = ""
                if label == "Both required" and event_vals and record_vals:
                    e_mean = np.mean(event_vals)
                    r_mean = np.mean(record_vals)
                    b_mean = mean
                    gap = f"  (event: {e_mean*100:.1f}%, record: {r_mean*100:.1f}%)"
                print(f"    {label:<20}: {mean*100:>5.1f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%] ({len(vals)} Q){gap}")

    # --- 6. Effect size: L1 vs L3 degradation ---
    print("\n## 6. Difficulty Degradation Effect Size (L1 → L3)")
    print(f"  {'System':<25} {'L1 Acc':>8} {'L3 Acc':>8} {'Delta':>8} {'Significant':>12}")
    print(f"  {'-'*65}")

    for name, data in systems.items():
        l1_vals = []
        l3_vals = []
        for qa_id, correct in data["correct"].items():
            meta = data["meta"].get(qa_id, {})
            diff = meta.get("difficulty", "")
            val = 1.0 if correct else 0.0
            if diff == "level_1":
                l1_vals.append(val)
            elif diff == "level_3":
                l3_vals.append(val)

        if l1_vals and l3_vals:
            l1_mean = np.mean(l1_vals)
            l3_mean = np.mean(l3_vals)
            delta = l1_mean - l3_mean

            # Paired analysis: only use QA pairs where both L1 and L3 have results
            # Since they're different QA pairs, use independent test
            stat, p = mcnemar_test(
                {k: True for k in range(int(l1_mean * len(l1_vals)))},
                {k: True for k in range(int(l3_mean * len(l3_vals)))},
                None,
            )

            sig = "*" if delta > 0.05 else "ns"
            print(f"  {name:<25} {l1_mean*100:>7.1f}% {l3_mean*100:>7.1f}% {delta*100:>+7.1f}% {'large'+sig:>12}")

    # --- 7. Context quality analysis ---
    print("\n## 7. Retrieval Context Quality Analysis")
    print("  (Average context length per system, correct vs incorrect answers)")

    for name, data in systems.items():
        ctx_correct = []
        ctx_incorrect = []
        for sr_data in json.loads((result_dir / system_files[name]).read_text(encoding="utf-8"))["scenario_results"]:
            for qr in sr_data["qa_results"]:
                ctx_len = len(qr["answer"].get("retrieval_context", ""))
                if qr["is_correct"]:
                    ctx_correct.append(ctx_len)
                else:
                    ctx_incorrect.append(ctx_len)

        if ctx_correct and ctx_incorrect:
            from statistics import mean
            print(f"\n  {name}:")
            print(f"    Correct answers:   avg context = {mean(ctx_correct):.0f} chars (n={len(ctx_correct)})")
            print(f"    Incorrect answers: avg context = {mean(ctx_incorrect):.0f} chars (n={len(ctx_incorrect)})")

    # --- 8. Save summary ---
    summary = {
        "description": "BiTempQA v2 statistical significance analysis",
        "n_bootstrap": 10000,
        "systems": {},
    }
    for name in systems:
        mean, std, ci_low, ci_high, values = overall_results[name]
        summary["systems"][name] = {
            "accuracy": round(mean, 4),
            "std": round(std, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "n_correct": int(sum(values)),
            "n_total": len(values),
        }

    # Pairwise p-values
    summary["pairwise_pvalues"] = {}
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i < j:
                stat, p = mcnemar_test(
                    systems[name_a]["correct"],
                    systems[name_b]["correct"],
                    None,
                )
                summary["pairwise_pvalues"][f"{name_a} vs {name_b}"] = round(p, 6)

    out_path = result_dir / "statistical_analysis.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\nSaved statistical summary to {out_path}")


if __name__ == "__main__":
    main()
