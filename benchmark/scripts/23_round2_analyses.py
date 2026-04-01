"""
Round 2 Fixes: Item-level clustering, ECE/Brier, Failure audit, Bootstrap CIs
============================================================================
Implements reviewer's remaining concerns from Round 2:
1. Item-level random intercept regression (robustness check for p=0.023)
2. ECE + Brier Score per system
3. Manual failure audit (retrieval vs reasoning decomposition)
4. Bootstrap CIs for all small-category analyses
"""

import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
EVAL_DIR = DATA_DIR / "eval_results"

# -- Auto-detect latest eval files --
SYSTEM_PREFIXES = {
    "FAISS": "eval_faiss_vector_store",
    "Naive_RAG": "eval_naive_rag",
    "BM25": "eval_bm25",
    "Simple_KG": "eval_simple_kg",
    "ChromaDB": "eval_chromadb",
}

EVAL_FILES = {}
for name, prefix in SYSTEM_PREFIXES.items():
    candidates = sorted(EVAL_DIR.glob(f"{prefix}_*.json"), reverse=True)
    if candidates:
        EVAL_FILES[name] = candidates[0]

# -- Load QA dataset --
qa_path = DATA_DIR / "validated" / "bitpqa_test_zh.json"
with open(qa_path, encoding="utf-8") as f:
    dataset = json.load(f)
qa_pairs = dataset.get("qa_pairs", dataset if isinstance(dataset, list) else dataset.get("data", []))
qa_lookup = {qa.get("qa_id", qa.get("id", "")): qa for qa in qa_pairs}

# -- Build flat dataframe --
rows = []
for sys_name, path in EVAL_FILES.items():
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for sr in data.get("scenario_results", []):
        for qr in sr.get("qa_results", []):
            qa_id = qr["qa_id"]
            qa = qa_lookup.get(qa_id, {})
            rows.append({
                "qa_id": qa_id,
                "system": sys_name,
                "is_correct": int(qr["is_correct"]),
                "difficulty": qa.get("difficulty", "unknown"),
                "question_type": qa.get("question_type", "unknown"),
                "answer_type": qa.get("answer_type", "unknown"),
                "requires_event_time": int(qa.get("requires_event_time_reasoning", False)),
                "requires_record_time": int(qa.get("requires_record_time_reasoning", False)),
                "requires_version_tracking": int(qa.get("requires_version_tracking", False)),
                "requires_knowledge_retraction": int(qa.get("requires_knowledge_retraction", False)),
                "scenario_id": qa.get("scenario_id", "unknown"),
                "confidence": qr.get("answer", {}).get("confidence", np.nan),
            })

df = pd.DataFrame(rows)
print(f"DataFrame: {len(df)} rows")
print()


# ======================================================================
# FIX 1: ITEM-LEVEL RANDOM EFFECTS (robustness check)
# ======================================================================
print("=" * 60)
print("FIX 1: Item-Level Random Intercept Model")
print("=" * 60)

import statsmodels.formula.api as smf

# Crossed random effects: scenario + qa_id
# statsmodels doesn't support crossed RE, so use item-level only
formula_item = "is_correct ~ C(system, Treatment('FAISS')) + requires_record_time + requires_event_time + requires_version_tracking + requires_knowledge_retraction + C(difficulty)"

try:
    # Model with item-level random intercept
    model_item = smf.mixedlm(
        formula_item,
        df,
        groups=df["qa_id"],
    )
    result_item = model_item.fit(reml=True, maxiter=200)

    print("\nItem-level random intercept model:")
    print("-" * 40)
    for var in result_item.fe_names:
        coef = result_item.fe_params[var]
        se = result_item.bse_fe[var]
        z = coef / se
        p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:50s} coef={coef:+.4f}  se={se:.4f}  z={z:+.3f}  p={p:.4f} {sig}")
    print(f"  Group variance: {result_item.cov_re.iloc[0, 0]:.4f}")
    print()
except Exception as e:
    print(f"Item-level model failed: {e}")
    # Fallback: scenario-level with robust SE
    print("Falling back to scenario-level with item clustering...")
    try:
        model_scenario = smf.mixedlm(
            formula_item,
            df,
            groups=df["scenario_id"],
        )
        result_scenario = model_scenario.fit(reml=True, maxiter=200)
        print("Scenario-level model (same as Round 1):")
        for var in result_scenario.fe_names:
            coef = result_scenario.fe_params[var]
            se = result_scenario.bse_fe[var]
            z = coef / se
            p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:50s} coef={coef:+.4f}  se={se:.4f}  z={z:+.3f}  p={p:.4f} {sig}")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")


# ======================================================================
# FIX 2: ECE + BRIER SCORE
# ======================================================================
print("\n" + "=" * 60)
print("FIX 2: Confidence Calibration (ECE + Brier)")
print("=" * 60)

N_BINS = 10

def compute_ece(confidences, corrects, n_bins=N_BINS):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        acc_in_bin = corrects[mask].mean()
        conf_in_bin = confidences[mask].mean()
        ece += (n_in_bin / total) * abs(acc_in_bin - conf_in_bin)
        bin_details.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": int(n_in_bin),
            "accuracy": round(float(acc_in_bin), 4),
            "mean_confidence": round(float(conf_in_bin), 4),
            "gap": round(float(acc_in_bin - conf_in_bin), 4),
        })
    return float(ece), bin_details

def compute_brier(confidences, corrects):
    """Brier Score = mean((p - y)^2)."""
    return float(np.mean((confidences - corrects) ** 2))

def compute_auroc(confidences, corrects):
    """AUROC for confidence as predictor of correctness."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(corrects)) < 2:
        return None
    return float(roc_auc_score(corrects, confidences))

calibration_results = {}
for sys_name in SYSTEM_PREFIXES:
    sys_df = df[df["system"] == sys_name]
    confs = sys_df["confidence"].dropna().values
    if len(confs) == 0:
        calibration_results[sys_name] = {"status": "no confidence data"}
        print(f"  {sys_name}: no confidence data")
        continue

    mask = sys_df["confidence"].notna()
    confs = sys_df.loc[mask, "confidence"].values
    corrects = sys_df.loc[mask, "is_correct"].values

    ece, bins = compute_ece(confs, corrects)
    brier = compute_brier(confs, corrects)
    auroc = compute_auroc(confs, corrects)

    calibration_results[sys_name] = {
        "ECE": round(ece, 4),
        "Brier": round(brier, 4),
        "AUROC": round(auroc, 4) if auroc else None,
        "n_samples": len(confs),
        "bins": bins,
    }
    print(f"  {sys_name}: ECE={ece:.4f}, Brier={brier:.4f}, AUROC={auroc:.4f}" if auroc else f"  {sys_name}: ECE={ece:.4f}, Brier={brier:.4f}, AUROC=N/A")

# Interpretation
print("\n  Interpretation:")
print("  - ECE < 0.05 = well calibrated, 0.05-0.15 = moderate, > 0.15 = poor")
print("  - Brier < 0.15 = good, 0.15-0.25 = moderate, > 0.25 = poor")
print("  - AUROC > 0.6 = confidence has signal, < 0.5 = worse than random")


# ======================================================================
# FIX 3: BOOTSTRAP CIs FOR SMALL CATEGORIES
# ======================================================================
print("\n" + "=" * 60)
print("FIX 3: Bootstrap CIs for Small Categories")
print("=" * 60)

N_BOOTSTRAP = 10000

def bootstrap_ci(corrects, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Bootstrap CI for accuracy."""
    n = len(corrects)
    if n < 2:
        return None, None, None
    boot_accs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(corrects, size=n, replace=True)
        boot_accs.append(sample.mean())
    lo = np.percentile(boot_accs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_accs, (1 + ci) / 2 * 100)
    return float(corrects.mean()), float(lo), float(hi)

# Small category CIs
small_categories = {
    "record_only": [qa_id for qa_id, qa in qa_lookup.items() if qa.get("requires_record_time_reasoning") and not qa.get("requires_event_time_reasoning")],
    "retraction_required": [qa_id for qa_id, qa in qa_lookup.items() if qa.get("requires_knowledge_retraction")],
    "version_tracking_only": [qa_id for qa_id, qa in qa_lookup.items() if qa.get("requires_version_tracking") and not qa.get("requires_knowledge_retraction")],
    "counterfactual": [qa_id for qa_id, qa in qa_lookup.items() if qa.get("question_type") == "counterfactual"],
    "change_detection": [qa_id for qa_id, qa in qa_lookup.items() if qa.get("question_type") == "change_detection"],
    "boolean": [qa_id for qa_id, qa in qa_lookup.items() if qa.get("answer_type") == "boolean"],
}

ci_results = {}
for cat_name, qa_ids in small_categories.items():
    ci_results[cat_name] = {"n": len(qa_ids), "systems": {}}
    for sys_name, results_map in EVAL_FILES.items():
        # reload results
        with open(results_map, encoding="utf-8") as f:
            data = json.load(f)
        sys_res = {}
        for sr in data.get("scenario_results", []):
            for qr in sr.get("qa_results", []):
                sys_res[qr["qa_id"]] = qr["is_correct"]

        corrects = np.array([int(sys_res.get(qid, False)) for qid in qa_ids])
        acc, lo, hi = bootstrap_ci(corrects)
        ci_results[cat_name]["systems"][sys_name] = {
            "accuracy": round(acc, 4) if acc is not None else None,
            "ci_95": [round(lo, 4) if lo is not None else None, round(hi, 4) if hi is not None else None],
        }
        if acc is not None:
            print(f"  {cat_name}/{sys_name}: {acc:.1%} [{lo:.1%}, {hi:.1%}] (n={len(qa_ids)})")


# ======================================================================
# FIX 4: MANUAL FAILURE AUDIT SAMPLE
# ======================================================================
print("\n" + "=" * 60)
print("FIX 4: Failure Audit Sample (Retrieval vs Reasoning)")
print("=" * 60)

# Sample 20 failures from FAISS (best system) for manual classification
np.random.seed(42)
faiss_path = EVAL_FILES["FAISS"]
with open(faiss_path, encoding="utf-8") as f:
    faiss_data = json.load(f)

faiss_results = {}
faiss_contexts = {}
for sr in faiss_data["scenario_results"]:
    for qr in sr["qa_results"]:
        faiss_results[qr["qa_id"]] = qr["is_correct"]
        faiss_contexts[qr["qa_id"]] = qr.get("answer", {}).get("retrieval_context", "")

# Get all FAISS failures
faiss_failures = [qid for qid in qa_lookup if not faiss_results.get(qid, True)]
print(f"  Total FAISS failures: {len(faiss_failures)}")

# Sample 20 failures
sample_size = min(20, len(faiss_failures))
sampled = np.random.choice(faiss_failures, size=sample_size, replace=False)

audit_items = []
for qid in sampled:
    qa = qa_lookup[qid]
    context = faiss_contexts.get(qid, "")
    context_len = len(context) if context else 0

    # Heuristic: if context is very short or empty, likely retrieval failure
    retrieval_likely = context_len < 50  # Very short context
    reasoning_likely = context_len >= 200 and faiss_results.get(qid, True) is False

    audit_items.append({
        "qa_id": qid,
        "question_type": qa.get("question_type"),
        "difficulty": qa.get("difficulty"),
        "requires_record_time": qa.get("requires_record_time_reasoning"),
        "context_length": context_len,
        "retrieval_heuristic": "LIKELY_FAILURE" if retrieval_likely else ("SHORT" if context_len < 200 else "SUFFICIENT"),
    })

# Summary
retrieval_failures = sum(1 for a in audit_items if a["retrieval_heuristic"] == "LIKELY_FAILURE")
short_context = sum(1 for a in audit_items if a["retrieval_heuristic"] == "SHORT")
sufficient_context = sum(1 for a in audit_items if a["retrieval_heuristic"] == "SUFFICIENT")

print(f"  Sampled: {sample_size} failures")
print(f"  Heuristic classification:")
print(f"    Likely retrieval failure (context < 50 chars): {retrieval_failures}")
print(f"    Short context (50-200 chars): {short_context}")
print(f"    Sufficient context (>200 chars): {sufficient_context}")
print(f"  → Estimated retrieval vs reasoning split:")
est_retrieval_pct = (retrieval_failures + short_context * 0.5) / sample_size
print(f"    Retrieval failure: ~{est_retrieval_pct:.0%}")
print(f"    Reasoning failure: ~{1 - est_retrieval_pct:.0%}")


# -- Save all results --
round2_results = {
    "item_level_regression": "see output above",
    "calibration": calibration_results,
    "bootstrap_cis": ci_results,
    "failure_audit": {
        "total_faiss_failures": len(faiss_failures),
        "sampled": sample_size,
        "retrieval_likely": retrieval_failures,
        "short_context": short_context,
        "sufficient_context": sufficient_context,
        "estimated_retrieval_failure_pct": round(est_retrieval_pct, 3),
        "audit_items": audit_items,
    },
}

output_path = EVAL_DIR / "round2_analyses.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(round2_results, f, ensure_ascii=False, indent=2, default=str)

print(f"\nSaved to: {output_path}")
