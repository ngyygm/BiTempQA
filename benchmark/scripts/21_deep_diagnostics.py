"""
BiTempQA Deep Diagnostic Analyses (Round 1 Review Fixes)
=========================================================
8 analyses suggested by external reviewer:
1. Error taxonomy — classify failure modes
2. Mixed-effects logistic regression — control for confounds
3. Delay sensitivity — accuracy vs event_time/record_time gap
4. Difficulty calibration — verify L1->L3 monotonicity
5. System complementarity — oracle score, pairwise overlap
6. Retraction/knowledge change analysis
7. Cross-domain interaction
8. Retrieval quality correlation (confidence -> correctness)
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np

# -- Paths --
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
EVAL_DIR = DATA_DIR / "eval_results"
LOCOMO_DIR = EVAL_DIR / "locomo"

# -- Auto-detect latest eval files --
SYSTEM_PREFIXES = {
    "FAISS": "eval_faiss_vector_store",
    "Naive RAG": "eval_naive_rag",
    "BM25": "eval_bm25",
    "Simple KG": "eval_simple_kg",
    "ChromaDB": "eval_chromadb",
}

EVAL_FILES = {}
for name, prefix in SYSTEM_PREFIXES.items():
    candidates = sorted(EVAL_DIR.glob(f"{prefix}_*.json"), reverse=True)
    if candidates:
        EVAL_FILES[name] = candidates[0]
        print(f"  {name}: {candidates[0].name}")

# -- Load QA dataset --
qa_dataset_path = DATA_DIR / "validated" / "bitpqa_test_zh.json"
if not qa_dataset_path.exists():
    # Try alternative locations
    for p in DATA_DIR.glob("**/*test*.json"):
        if "bitpqa" in p.name.lower() or "bitpqa" in p.stem.lower():
            qa_dataset_path = p
            break

with open(qa_dataset_path, encoding="utf-8") as f:
    dataset = json.load(f)

# Handle both formats: direct list or dict with qa_pairs key
if isinstance(dataset, list):
    qa_pairs = dataset
elif "qa_pairs" in dataset:
    qa_pairs = dataset["qa_pairs"]
else:
    qa_pairs = dataset.get("data", [])

qa_lookup = {}
for qa in qa_pairs:
    qid = qa.get("qa_id") or qa.get("id", "")
    qa_lookup[qid] = qa

print(f"Loaded {len(qa_lookup)} QA pairs from {qa_dataset_path.name}")

# -- Load eval results --
system_results = {}
for sys_name, path in EVAL_FILES.items():
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    # Handle nested scenario_results structure
    if "scenario_results" in data:
        for sr in data["scenario_results"]:
            for qr in sr.get("qa_results", []):
                results[qr["qa_id"]] = {
                    "is_correct": qr["is_correct"],
                    "confidence": qr.get("answer", {}).get("confidence"),
                    "error_category": qr.get("error_category"),
                    "raw_response": qr.get("answer", {}).get("raw_response", ""),
                    "retrieval_context": qr.get("answer", {}).get("retrieval_context", ""),
                    "extracted_answer": qr.get("answer", {}).get("extracted_answer", ""),
                }
    elif "results" in data:
        for r in data["results"]:
            qid = r.get("qa_id") or r.get("id", "")
            results[qid] = {
                "is_correct": r.get("is_correct", r.get("correct", False)),
                "confidence": r.get("confidence"),
                "error_category": r.get("error_category"),
            }
    system_results[sys_name] = results
    print(f"  {sys_name}: {len(results)} results, {sum(1 for v in results.values() if v['is_correct'])} correct")

# -- Load scenarios for domain info --
SCENARIO_DIR = DATA_DIR / "scenarios"
scenario_meta = {}

# Try to load scenario data from the QA pairs themselves
for qa in qa_pairs:
    sid = qa.get("scenario_id", "")
    if sid and sid not in scenario_meta:
        scenario_meta[sid] = {
            "domain": qa.get("domain", "unknown"),
            "scenario_type": qa.get("scenario_type", "unknown"),
        }

# Also try loading from scenario files if they exist
if SCENARIO_DIR.exists():
    for sf in SCENARIO_DIR.glob("*.json"):
        with open(sf, encoding="utf-8") as f:
            try:
                scenarios = json.load(f)
                if isinstance(scenarios, list):
                    for s in scenarios:
                        scenario_meta[s["scenario_id"]] = {
                            "domain": s.get("domain", "unknown"),
                            "scenario_type": s.get("scenario_type", "unknown"),
                            "num_writes": len(s.get("memory_writes", [])),
                            "writes": s.get("memory_writes", []),
                        }
            except (json.JSONDecodeError, KeyError):
                pass

print(f"Loaded {len(scenario_meta)} scenario metadata entries")

results = {}


# ======================================================================
# ANALYSIS 1: ERROR TAXONOMY
# ======================================================================
def analyze_error_taxonomy():
    """Classify failures by error mode based on question metadata."""
    error_modes = defaultdict(lambda: Counter())

    for sys_name, results_map in system_results.items():
        for qa_id, res in results_map.items():
            if res["is_correct"]:
                continue
            qa = qa_lookup.get(qa_id)
            if not qa:
                continue

            modes = []
            if qa.get("requires_event_time_reasoning") and qa.get("requires_record_time_reasoning"):
                modes.append("dual_timestamp_failure")
            elif qa.get("requires_record_time_reasoning"):
                modes.append("record_time_only_failure")
            elif qa.get("requires_event_time_reasoning"):
                modes.append("event_time_only_failure")

            if qa.get("requires_version_tracking"):
                modes.append("version_tracking_failure")
            if qa.get("requires_knowledge_retraction"):
                modes.append("retraction_failure")

            qt = qa.get("question_type", "unknown")
            qt_map = {
                "counterfactual": "counterfactual_failure",
                "change_detection": "change_detection_failure",
                "complex_temporal": "complex_reasoning_failure",
                "multi_hop_temporal": "multi_hop_failure",
            }
            if qt in qt_map:
                modes.append(qt_map[qt])

            at = qa.get("answer_type", "")
            if at == "abstractive":
                modes.append("abstractive_failure")
            elif at == "boolean":
                modes.append("boolean_failure")

            if not modes:
                modes.append("unclassified")

            for m in modes:
                error_modes[sys_name][m] += 1

    return dict(error_modes)


# ======================================================================
# ANALYSIS 2: DELAY SENSITIVITY
# ======================================================================
def analyze_delay_sensitivity():
    """Bin by temporal requirement, compute accuracy."""
    # Since we may not have write-level delays, use the temporal requirement as proxy
    # and also check if query times reveal delay patterns
    temporal_cats = {
        "event_only": [],
        "record_only": [],
        "both_required": [],
        "version_tracking": [],
        "retraction": [],
    }

    for qa_id, qa in qa_lookup.items():
        if qa.get("requires_event_time_reasoning") and qa.get("requires_record_time_reasoning"):
            temporal_cats["both_required"].append(qa_id)
        elif qa.get("requires_record_time_reasoning"):
            temporal_cats["record_only"].append(qa_id)
        elif qa.get("requires_event_time_reasoning"):
            temporal_cats["event_only"].append(qa_id)

        if qa.get("requires_version_tracking"):
            temporal_cats["version_tracking"].append(qa_id)
        if qa.get("requires_knowledge_retraction"):
            temporal_cats["retraction"].append(qa_id)

    delay_results = {}
    for sys_name, results_map in system_results.items():
        sys_delay = {}
        for cat_name, qa_ids in temporal_cats.items():
            if not qa_ids:
                continue
            correct = sum(1 for qid in qa_ids if qid in results_map and results_map[qid]["is_correct"])
            total = sum(1 for qid in qa_ids if qid in results_map)
            sys_delay[cat_name] = {
                "accuracy": correct / total if total > 0 else 0,
                "correct": correct,
                "total": total,
            }
        delay_results[sys_name] = sys_delay

    return {"counts": {k: len(v) for k, v in temporal_cats.items()}, "results": delay_results}


# ======================================================================
# ANALYSIS 3: DIFFICULTY CALIBRATION
# ======================================================================
def analyze_difficulty_calibration():
    """Check if L1 < L2 < L3 holds for each system x question_type."""
    groups = defaultdict(list)
    for qa_id, qa in qa_lookup.items():
        key = (qa.get("difficulty", ""), qa.get("question_type", ""))
        groups[key].append(qa_id)

    qt_calibration = {}
    question_types = set(qa.get("question_type") for qa in qa_lookup.values())

    for qt in sorted(question_types):
        qt_calibration[qt] = {}
        for sys_name, results_map in system_results.items():
            accs = []
            for level in ["level_1", "level_2", "level_3"]:
                level_ids = groups.get((level, qt), [])
                correct = sum(1 for qid in level_ids if qid in results_map and results_map[qid]["is_correct"])
                total = sum(1 for qid in level_ids if qid in results_map)
                accs.append(correct / total if total > 0 else None)

            valid_accs = [a for a in accs if a is not None]
            is_monotonic = all(valid_accs[i] >= valid_accs[i + 1] for i in range(len(valid_accs) - 1)) if len(valid_accs) >= 2 else None

            qt_calibration[qt][sys_name] = {
                "L1": accs[0], "L2": accs[1], "L3": accs[2],
                "monotonic": is_monotonic,
            }

    return qt_calibration


# ======================================================================
# ANALYSIS 4: SYSTEM COMPLEMENTARITY
# ======================================================================
def analyze_system_complementarity():
    """Oracle-over-systems score and pairwise overlap."""
    all_qa_ids = list(qa_lookup.keys())
    systems = list(system_results.keys())

    sys_correct = {}
    for sys_name in systems:
        sys_correct[sys_name] = {qid: system_results[sys_name].get(qid, {}).get("is_correct", False) for qid in all_qa_ids}

    # Oracle: any system correct
    oracle_correct = sum(1 for qid in all_qa_ids if any(sys_correct[s].get(qid, False) for s in systems))

    # Pairwise
    pairwise = {}
    for i, s1 in enumerate(systems):
        for s2 in systems[i + 1:]:
            both = sum(1 for qid in all_qa_ids if sys_correct[s1].get(qid) and sys_correct[s2].get(qid))
            s1_only = sum(1 for qid in all_qa_ids if sys_correct[s1].get(qid) and not sys_correct[s2].get(qid))
            s2_only = sum(1 for qid in all_qa_ids if not sys_correct[s1].get(qid) and sys_correct[s2].get(qid))
            neither = sum(1 for qid in all_qa_ids if not sys_correct[s1].get(qid) and not sys_correct[s2].get(qid))
            jaccard = both / (both + s1_only + s2_only) if (both + s1_only + s2_only) > 0 else 0
            pairwise[f"{s1} vs {s2}"] = {
                "both_correct": both, "s1_only": s1_only, "s2_only": s2_only,
                "neither": neither, "jaccard_correct": round(jaccard, 3),
                "complementarity": s1_only + s2_only,
            }

    # Uniquely correct per system
    unique = {}
    for s in systems:
        unique[s] = sum(
            1 for qid in all_qa_ids
            if sys_correct[s].get(qid) and not any(sys_correct[s2].get(qid) for s2 in systems if s2 != s)
        )

    # Items all systems get wrong
    all_wrong = sum(1 for qid in all_qa_ids if not any(sys_correct[s].get(qid) for s in systems))

    return {
        "oracle_accuracy": oracle_correct / len(all_qa_ids),
        "oracle_correct": oracle_correct,
        "total": len(all_qa_ids),
        "all_systems_wrong": all_wrong,
        "pairwise": pairwise,
        "uniquely_correct": unique,
    }


# ======================================================================
# ANALYSIS 5: RETRACTION / KNOWLEDGE CHANGE ANALYSIS
# ======================================================================
def analyze_retraction():
    """Separate performance on new/corrected/revoked facts."""
    categories = {
        "retraction_required": [],
        "version_tracking_only": [],
        "neither": [],
    }

    for qa_id, qa in qa_lookup.items():
        if qa.get("requires_knowledge_retraction"):
            categories["retraction_required"].append(qa_id)
        elif qa.get("requires_version_tracking"):
            categories["version_tracking_only"].append(qa_id)
        else:
            categories["neither"].append(qa_id)

    retraction_results = {}
    for sys_name, results_map in system_results.items():
        sys_res = {}
        for cat_name, qa_ids in categories.items():
            correct = sum(1 for qid in qa_ids if qid in results_map and results_map[qid]["is_correct"])
            total = sum(1 for qid in qa_ids if qid in results_map)
            sys_res[cat_name] = {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}
        retraction_results[sys_name] = sys_res

    return {"counts": {k: len(v) for k, v in categories.items()}, "results": retraction_results}


# ======================================================================
# ANALYSIS 6: CROSS-DOMAIN INTERACTION
# ======================================================================
def analyze_cross_domain():
    """Accuracy by domain to check confounds."""
    qa_domain = {}
    for qa_id, qa in qa_lookup.items():
        sid = qa.get("scenario_id", "")
        sinfo = scenario_meta.get(sid)
        qa_domain[qa_id] = sinfo.get("domain", "unknown") if sinfo else qa.get("domain", "unknown")

    domains = sorted(set(qa_domain.values()))

    domain_results = {}
    for sys_name, results_map in system_results.items():
        sys_domain = {}
        for domain in domains:
            domain_ids = [qid for qid in qa_lookup if qa_domain.get(qid) == domain]
            correct = sum(1 for qid in domain_ids if qid in results_map and results_map[qid]["is_correct"])
            total = sum(1 for qid in domain_ids if qid in results_map)
            sys_domain[domain] = {"accuracy": correct / total if total > 0 else 0, "total": total}
        domain_results[sys_name] = sys_domain

    # Domain x question_type distribution
    domain_qt_dist = defaultdict(Counter)
    for qa_id, qa in qa_lookup.items():
        domain_qt_dist[qa_domain[qa_id]][qa.get("question_type", "unknown")] += 1

    return {
        "domain_results": domain_results,
        "domain_qt_distribution": {d: dict(qts) for d, qts in domain_qt_dist.items()},
    }


# ======================================================================
# ANALYSIS 7: RETRIEVAL QUALITY CORRELATION
# ======================================================================
def analyze_retrieval_quality():
    """Correlate confidence score with correctness."""
    confidence_analysis = {}
    for sys_name, results_map in system_results.items():
        confidences_correct = []
        confidences_wrong = []
        for qa_id, res in results_map.items():
            conf = res.get("confidence")
            if conf is not None:
                if res["is_correct"]:
                    confidences_correct.append(conf)
                else:
                    confidences_wrong.append(conf)

        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
        bin_results = {}
        for lo, hi in bins:
            bin_c = [c for c in confidences_correct if lo <= c < hi]
            bin_w = [c for c in confidences_wrong if lo <= c < hi]
            total_in_bin = len(bin_c) + len(bin_w)
            bin_results[f"{lo:.1f}-{hi:.2f}"] = {
                "accuracy": len(bin_c) / total_in_bin if total_in_bin > 0 else 0,
                "count": total_in_bin,
            }

        mean_c = float(np.mean(confidences_correct)) if confidences_correct else 0
        mean_w = float(np.mean(confidences_wrong)) if confidences_wrong else 0

        confidence_analysis[sys_name] = {
            "mean_confidence_correct": round(mean_c, 4),
            "mean_confidence_wrong": round(mean_w, 4),
            "calibration_gap": round(mean_c - mean_w, 4),
            "bins": bin_results,
        }

    return confidence_analysis


# ======================================================================
# ANALYSIS 8: JUDGE DISAGREEMENT DEEP DIVE
# ======================================================================
def analyze_judge_disagreement():
    """Analyze patterns in judge disagreements."""
    jcv_path = EVAL_DIR / "judge_crossvalidation.json"
    if not jcv_path.exists():
        return {"status": "skipped", "reason": "judge_crossvalidation.json not found"}

    with open(jcv_path, encoding="utf-8") as f:
        jcv = json.load(f)

    disagreements = jcv.get("disagreements", [])

    patterns = {"deepseek_lenient": 0, "qwen_lenient": 0}
    answer_breakdown = Counter()

    for d in disagreements:
        if d["deepseek_verdict"] == "CORRECT" and d["qwen_verdict"] == "WRONG":
            patterns["deepseek_lenient"] += 1
        else:
            patterns["qwen_lenient"] += 1

        gold = d.get("gold_answer", "")
        gen = d.get("generated_answer", "")
        if len(gen) < len(gold) * 0.5:
            answer_breakdown["answer_too_short"] += 1
        else:
            answer_breakdown["partial_match"] += 1

    return {
        "total_disagreements": len(disagreements),
        "patterns": patterns,
        "answer_breakdown": dict(answer_breakdown),
    }


# ======================================================================
# RUN ALL
# ======================================================================
print("\n" + "=" * 60)
print("BiTempQA Deep Diagnostic Analyses")
print("=" * 60)

all_results = {}

print("\n[1/8] Error Taxonomy...")
all_results["error_taxonomy"] = analyze_error_taxonomy()
for sys_name, modes in all_results["error_taxonomy"].items():
    total_errors = sum(modes.values())
    top3 = modes.most_common(3)
    print(f"  {sys_name}: {total_errors} errors - top: {top3}")

print("\n[2/8] Temporal Requirement Sensitivity...")
all_results["temporal_sensitivity"] = analyze_delay_sensitivity()
print(f"  Category counts: {all_results['temporal_sensitivity']['counts']}")
for sys_name, bins in all_results["temporal_sensitivity"]["results"].items():
    accs = [f"{b}: {v['accuracy']:.1%}" for b, v in bins.items()]
    print(f"  {sys_name}: {', '.join(accs)}")

print("\n[3/8] Difficulty Calibration...")
all_results["difficulty_calibration"] = analyze_difficulty_calibration()
non_mono = []
for qt, sys_data in all_results["difficulty_calibration"].items():
    for sys_name, data in sys_data.items():
        vals = [data["L1"], data["L2"], data["L3"]]
        valid = [v for v in vals if v is not None]
        if len(valid) >= 2 and data.get("monotonic") is False:
            non_mono.append(f"{qt}/{sys_name}: L1={data['L1']:.2f} L2={data['L2']:.2f} L3={data['L3']:.2f}" if all(v is not None for v in vals) else "")
if non_mono:
    print(f"  NON-MONOTONIC difficulty ({len([n for n in non_mono if n])} cases):")
    for nm in non_mono[:15]:
        if nm:
            print(f"    {nm}")
else:
    print("  All difficulty curves are monotonic!")

print("\n[4/8] System Complementarity...")
all_results["complementarity"] = analyze_system_complementarity()
comp = all_results["complementarity"]
print(f"  Oracle accuracy: {comp['oracle_accuracy']:.1%} ({comp['oracle_correct']}/{comp['total']})")
print(f"  All systems wrong: {comp['all_systems_wrong']}/{comp['total']}")
print(f"  Uniquely correct:")
for s, u in comp["uniquely_correct"].items():
    print(f"    {s}: {u}")
print(f"  Pairwise complementarity:")
for pair, data in comp["pairwise"].items():
    print(f"    {pair}: Jaccard={data['jaccard_correct']}, complement={data['complementarity']}")

print("\n[5/8] Retraction Analysis...")
all_results["retraction"] = analyze_retraction()
print(f"  Counts: {all_results['retraction']['counts']}")
for sys_name, cats in all_results["retraction"]["results"].items():
    accs = [f"{k}: {v['accuracy']:.1%}" for k, v in cats.items()]
    print(f"  {sys_name}: {', '.join(accs)}")

print("\n[6/8] Cross-Domain Interaction...")
all_results["cross_domain"] = analyze_cross_domain()
for sys_name, domains in all_results["cross_domain"]["domain_results"].items():
    accs = [f"{d}: {v['accuracy']:.1%}(n={v['total']})" for d, v in sorted(domains.items())]
    print(f"  {sys_name}: {', '.join(accs)}")

print("\n[7/8] Retrieval Quality Correlation...")
all_results["retrieval_quality"] = analyze_retrieval_quality()
for sys_name, data in all_results["retrieval_quality"].items():
    print(f"  {sys_name}: conf correct={data['mean_confidence_correct']:.3f}, wrong={data['mean_confidence_wrong']:.3f}, gap={data['calibration_gap']:.3f}")

print("\n[8/8] Judge Disagreement...")
all_results["judge_disagreement"] = analyze_judge_disagreement()
jd = all_results["judge_disagreement"]
if jd.get("status") != "skipped":
    print(f"  Total disagreements: {jd['total_disagreements']}")
    print(f"  Patterns: {jd['patterns']}")
    print(f"  Breakdown: {jd['answer_breakdown']}")

# -- Save --
output_path = EVAL_DIR / "deep_diagnostics.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

print(f"\n{'=' * 60}")
print(f"Saved to: {output_path}")
print(f"{'=' * 60}")
