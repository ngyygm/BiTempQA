"""
Mixed-effects logistic regression for BiTempQA
=================================================
Controls for: system, question_type, difficulty, temporal requirement,
              answer_type, retraction requirement, version tracking
Dependent variable: is_correct (binary)
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
                "requires_both": int(qa.get("requires_event_time_reasoning", False) and qa.get("requires_record_time_reasoning", False)),
                "requires_version_tracking": int(qa.get("requires_version_tracking", False)),
                "requires_retraction": int(qa.get("requires_knowledge_retraction", False)),
                "scenario_id": qa.get("scenario_id", "unknown"),
                "confidence": qr.get("answer", {}).get("confidence", np.nan),
            })

df = pd.DataFrame(rows)
print(f"DataFrame: {len(df)} rows, {df['is_correct'].mean():.3f} mean accuracy")
print(f"Columns: {list(df.columns)}")
print()

# -- Model 1: Full fixed effects --
print("=" * 60)
print("MODEL 1: Full Fixed Effects Logistic Regression")
print("=" * 60)
formula1 = "is_correct ~ C(system, Treatment('FAISS')) + C(difficulty) + C(question_type) + C(answer_type) + requires_event_time + requires_record_time + requires_version_tracking + requires_retraction"
try:
    model1 = smf.logit(formula1, data=df).fit(disp=0, maxiter=200)
    print(model1.summary2().tables[1].to_string())
    print(f"\nPseudo R-squared: {model1.prsquared:.4f}")
    print(f"Log-Likelihood: {model1.llf:.2f}")
    print(f"AIC: {model1.aic:.2f}")
except Exception as e:
    print(f"Model 1 failed: {e}")

print()

# -- Model 2: Interaction model --
print("=" * 60)
print("MODEL 2: System x Temporal Requirement Interactions")
print("=" * 60)
formula2 = "is_correct ~ C(system, Treatment('FAISS')) * (requires_event_time + requires_record_time + requires_both) + C(difficulty) + C(question_type)"
try:
    model2 = smf.logit(formula2, data=df).fit(disp=0, maxiter=200)
    print(model2.summary2().tables[1].to_string())
    print(f"\nPseudo R-squared: {model2.prsquared:.4f}")
except Exception as e:
    print(f"Model 2 failed: {e}")

print()

# -- Model 3: Mixed effects with scenario random intercept --
print("=" * 60)
print("MODEL 3: Mixed Effects (scenario random intercept)")
print("=" * 60)
formula3 = "is_correct ~ C(system, Treatment('FAISS')) + C(difficulty) + requires_event_time + requires_record_time + requires_version_tracking + requires_retraction"
try:
    model3 = smf.mixedlm(formula3, df, groups=df["scenario_id"])
    result3 = model3.fit(reml=True)
    print(result3.summary().tables[1].to_string())
    print(f"\nGroup variance (scenario): {result3.cov_re.iloc[0, 0]:.6f}")
except Exception as e:
    print(f"Model 3 failed: {e}")

print()

# -- Marginal effects for Model 1 --
print("=" * 60)
print("MARGINAL EFFECTS (Model 1)")
print("=" * 60)
try:
    mfx = model1.get_margeff(at="overall")
    print(mfx.summary().tables[1].to_string())
except Exception as e:
    print(f"Marginal effects failed: {e}")

print()

# -- Key contrasts --
print("=" * 60)
print("KEY HYPOTHESIS TESTS")
print("=" * 60)

# Test: Is record_time reasoning harder after controlling for everything?
try:
    hypotheses = {
        "record_time_harder": "requires_record_time = 0",
        "event_time_effect": "requires_event_time = 0",
        "version_tracking_helps": "requires_version_tracking = 0",
        "retraction_harder": "requires_retraction = 0",
        "kg_worse_than_faiss": "C(system, Treatment('FAISS'))[T.Simple_KG] = 0",
        "chroma_worse_than_faiss": "C(system, Treatment('FAISS'))[T.ChromaDB] = 0",
        "level3_harder": "C(difficulty)[T.level_3] = 0",
        "abstractive_harder": "C(answer_type)[T.abstractive] = 0",
    }
    for name, hyp in hypotheses.items():
        try:
            t_test = model1.t_test(hyp)
            coef = t_test.result[0][0]
            pval = t_test.result[0][3]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            direction = "+" if coef > 0 else "-"
            print(f"  {name}: coef={coef:+.4f}, p={pval:.4f} {sig}  ({direction}effect)")
        except Exception as e:
            print(f"  {name}: test failed ({e})")
except Exception as e:
    print(f"Hypothesis tests failed: {e}")

# -- Save results --
output = {
    "n_observations": len(df),
    "n_systems": df["system"].nunique(),
    "n_scenarios": df["scenario_id"].nunique(),
    "overall_accuracy": float(df["is_correct"].mean()),
}

output_path = EVAL_DIR / "mixed_effects_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved metadata to: {output_path}")
