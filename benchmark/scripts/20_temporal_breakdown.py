#!/usr/bin/env python3
"""
Script 20: Per-system accuracy broken down by temporal reasoning requirements.

For each system, computes accuracy for:
  - event_time_only  (requires_event_time_reasoning=True, requires_record_time_reasoning=False)
  - record_time_only (requires_event_time_reasoning=False, requires_record_time_reasoning=True)
  - both_required    (both=True)
  - version_tracking (requires_version_tracking=True)
"""

import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval_results")
METADATA_PATH = os.path.join(BASE_DIR, "data", "validated", "bitpqa_test_zh.json")
OUTPUT_PATH = os.path.join(EVAL_DIR, "temporal_breakdown.json")

EVAL_FILES = [
    "eval_faiss_vector_store_1774518673.json",
    "eval_bm25_1774522334.json",
    "eval_simple_kg_1774520732.json",
    "eval_naive_rag_1774521546.json",
    "eval_chromadb_1774519684.json",
]

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Build lookup: qa_id -> temporal flags
qa_meta = {}
for pair in metadata["qa_pairs"]:
    qa_meta[pair["qa_id"]] = {
        "requires_event_time_reasoning": pair.get("requires_event_time_reasoning", False),
        "requires_record_time_reasoning": pair.get("requires_record_time_reasoning", False),
        "requires_version_tracking": pair.get("requires_version_tracking", False),
    }

# Classify QA IDs into temporal categories
event_time_only_ids = set()
record_time_only_ids = set()
both_required_ids = set()
version_tracking_ids = set()

for qa_id, flags in qa_meta.items():
    et = flags["requires_event_time_reasoning"]
    rt = flags["requires_record_time_reasoning"]
    vt = flags["requires_version_tracking"]

    if et and rt:
        both_required_ids.add(qa_id)
    elif et and not rt:
        event_time_only_ids.add(qa_id)
    elif not et and rt:
        record_time_only_ids.add(qa_id)

    if vt:
        version_tracking_ids.add(qa_id)

print(f"QA category sizes (from metadata, total {len(qa_meta)}):")
print(f"  event_time_only:  {len(event_time_only_ids)}")
print(f"  record_time_only: {len(record_time_only_ids)}")
print(f"  both_required:    {len(both_required_ids)}")
print(f"  version_tracking: {len(version_tracking_ids)}")
print()

# ---------------------------------------------------------------------------
# Load eval results
# ---------------------------------------------------------------------------
def load_eval_results(filepath):
    """Load eval JSON and return a flat dict: qa_id -> is_correct."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    system_name = data.get("system_name", "Unknown")
    for scenario in data.get("scenario_results", []):
        for qa in scenario.get("qa_results", []):
            results[qa["qa_id"]] = qa["is_correct"]
    return system_name, results


# ordered dict to preserve insertion order (we insert in eval file order)
systems = {}
for fname in EVAL_FILES:
    fpath = os.path.join(EVAL_DIR, fname)
    if not os.path.exists(fpath):
        print(f"WARNING: {fpath} not found, skipping.")
        continue
    name, results = load_eval_results(fpath)
    systems[name] = results
    print(f"Loaded {len(results)} QA results for {name}")

print()

# ---------------------------------------------------------------------------
# Compute per-system accuracy for each category
# ---------------------------------------------------------------------------
def accuracy_for_category(qa_ids, system_results):
    """Return (correct, total) for the given qa_ids."""
    correct = 0
    total = 0
    for qa_id in qa_ids:
        if qa_id in system_results:
            total += 1
            if system_results[qa_id]:
                correct += 1
    return correct, total


categories = {
    "event_time_only": event_time_only_ids,
    "record_time_only": record_time_only_ids,
    "both_required": both_required_ids,
    "version_tracking": version_tracking_ids,
}

# Overall accuracy too
all_qa_ids = set(qa_meta.keys())

# Build results structure for JSON output
output = {
    "metadata": {
        "total_qa_pairs": len(qa_meta),
        "category_sizes": {
            "event_time_only": len(event_time_only_ids),
            "record_time_only": len(record_time_only_ids),
            "both_required": len(both_required_ids),
            "version_tracking": len(version_tracking_ids),
        },
    },
    "systems": {},
}

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
# Column widths
col_system = 22
col_acc = 14

header = (
    f"{'System':<{col_system}}"
    f"{'Overall':<{col_acc}}"
    f"{'EventTime':<{col_acc}}"
    f"{'RecordTime':<{col_acc}}"
    f"{'Both':<{col_acc}}"
    f"{'VersionTrack':<{col_acc}}"
)
sep = "-" * len(header)
print("=" * len(header))
print("Per-System Accuracy by Temporal Reasoning Requirement")
print("=" * len(header))
print(header)
print(sep)

for sys_name, sys_results in systems.items():
    row = {"system": sys_name, "categories": {}}

    # Overall
    o_corr, o_tot = accuracy_for_category(all_qa_ids, sys_results)
    o_acc = o_corr / o_tot if o_tot > 0 else 0.0
    row["overall"] = {"correct": o_corr, "total": o_tot, "accuracy": round(o_acc, 4)}

    # Temporal categories
    parts = []
    for cat_name, cat_ids in categories.items():
        c_corr, c_tot = accuracy_for_category(cat_ids, sys_results)
        c_acc = c_corr / c_tot if c_tot > 0 else 0.0
        row["categories"][cat_name] = {
            "correct": c_corr,
            "total": c_tot,
            "accuracy": round(c_acc, 4),
        }
        parts.append(f"{c_corr}/{c_tot} ({c_acc:.1%})")

    overall_str = f"{o_corr}/{o_tot} ({o_acc:.1%})"
    line = (
        f"{sys_name:<{col_system}}"
        f"{overall_str:<{col_acc}}"
        + "".join(f"{p:<{col_acc}}" for p in parts)
    )
    print(line)
    output["systems"][sys_name] = row

print(sep)
print()

# ---------------------------------------------------------------------------
# Summary table: just percentages for easy comparison
# ---------------------------------------------------------------------------
print("=" * len(header))
print("Summary Table (Accuracy %)")
print("=" * len(header))

# Build a list of (system, {cat: acc}) for sorting
summary_rows = []
for sys_name, row_data in output["systems"].items():
    accs = {}
    accs["overall"] = row_data["overall"]["accuracy"]
    for cat_name, cat_data in row_data["categories"].items():
        accs[cat_name] = cat_data["accuracy"]
    summary_rows.append((sys_name, accs))

short_header = (
    f"{'System':<{col_system}}"
    f"{'Overall':<{col_acc}}"
    f"{'EventTime':<{col_acc}}"
    f"{'RecordTime':<{col_acc}}"
    f"{'Both':<{col_acc}}"
    f"{'VersionTrack':<{col_acc}}"
)
print(short_header)
print(sep)

for sys_name, accs in summary_rows:
    def pct_pad(acc, width):
        s = f"{acc:.1%}"
        return f"{s:<{width}}"

    line = (
        f"{sys_name:<{col_system}}"
        + pct_pad(accs['overall'], col_acc)
        + pct_pad(accs['event_time_only'], col_acc)
        + pct_pad(accs['record_time_only'], col_acc)
        + pct_pad(accs['both_required'], col_acc)
        + pct_pad(accs['version_tracking'], col_acc)
    )
    print(line)

print(sep)
print()

# ---------------------------------------------------------------------------
# Find best system per category
# ---------------------------------------------------------------------------
print("Best system per category:")
for cat_key in ["overall", "event_time_only", "record_time_only", "both_required", "version_tracking"]:
    best_name = None
    best_acc = -1
    for sys_name, row_data in output["systems"].items():
        if cat_key == "overall":
            acc = row_data["overall"]["accuracy"]
        else:
            acc = row_data["categories"][cat_key]["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_name = sys_name
    label = cat_key.replace("_", " ").title()
    print(f"  {label:<22s}: {best_name} ({best_acc:.1%})")

print()

# ---------------------------------------------------------------------------
# Save JSON output
# ---------------------------------------------------------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved to {OUTPUT_PATH}")
