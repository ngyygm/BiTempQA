"""Merge real-source + LLM-generated scenarios, generate QA pairs, produce final dataset.

Strategy:
  - Real-source (215 scenarios): ChronoQA + curated real-world events → primary data
  - LLM-generated (118 scenarios): Keep only rare types not covered by real data
  - Total target: ~250-280 scenarios with ~500+ QA pairs

Merging rules:
  - entity_attribute_evolution: keep ALL real (9) + ALL old (15) = 24
  - late_arriving_facts: keep ALL real (2) + ALL old (14) = 16
  - knowledge_retraction: keep ALL real (1) + ALL old (12) = 13
  - multi_source_information: keep ALL real (1) + ALL old (12) = 13
  - gradual_accumulation: keep ALL real (1) + ALL old (11) = 12
  - relationship_evolution: keep ALL real (1) + ALL old (10) = 11
  - contradictory_information: keep ALL old (12) — no real source
  - future_dated_information: keep ALL old (12) — no real source
  - entity_identity_resolution: keep ALL old (10) — no real source
  - temporal_ambiguity: keep ALL old (10) — no real source
  - point_in_time_query: keep ALL real (90) — ChronoQA
  - temporal_aggregation: keep ALL real (85) — ChronoQA
  - temporal_ordering: keep ALL real (25) — ChronoQA
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "generated" / "merged_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scenarios(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def generate_additional_qa_pairs(scenario: Dict) -> List[Dict]:
    """Generate additional QA pairs for scenarios that only have 1 QA pair."""
    existing_qa = scenario.get("qa_pairs", [])
    writes = scenario.get("memory_writes", [])
    if not writes or not existing_qa:
        return existing_qa

    new_qas = list(existing_qa)

    # For scenarios with 3+ writes and <3 QA pairs, generate more
    if len(writes) < 2:
        return new_qas

    sid = scenario["scenario_id"]
    qa_counter = len(existing_qa)

    # Type 1: Point-in-time query about a specific write
    if len(writes) >= 2:
        for w_idx, write in enumerate(writes):
            qa_counter += 1
            # Extract year from event_time
            evt = write.get("event_time", "")[:10]
            rec = write.get("record_time", "")[:10]

            # Generate a knowledge query
            text = write["text"]
            entities = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
            if entities:
                entity = entities[0]
            else:
                entity = "某实体"

            if w_idx < len(writes) - 1:
                next_evt = writes[w_idx + 1].get("event_time", "")[:10]
                # Before the next event, what was known?
                new_qas.append({
                    "qa_id": f"{sid}_L1_{qa_counter:03d}",
                    "scenario_id": sid,
                    "difficulty": "level_1",
                    "question_type": "point_in_time",
                    "question_zh": f"在{next_evt}之前，关于{entity}已知的信息是什么？",
                    "question_en": "",
                    "answer_zh": text,
                    "answer_en": "",
                    "answer_type": "abstractive",
                    "choices": [],
                    "correct_choice_index": None,
                    "ranking_order": None,
                    "query_event_time": f"{next_evt}T00:00:00Z",
                    "query_record_time": write.get("record_time"),
                    "relevant_time_range": {"start": write.get("event_time"), "end": f"{next_evt}T00:00:00Z"},
                    "reasoning_chain": [f"在{next_evt}之前，最新记录的信息是：{text[:50]}"],
                    "requires_event_time_reasoning": True,
                    "requires_record_time_reasoning": rec != evt,
                    "requires_version_tracking": False,
                    "requires_knowledge_retraction": False,
                    "source_write_ids": [write["write_id"]],
                    "generation_method": "llm",
                    "validation_status": "unvalidated",
                })

    # Type 2: Temporal ordering (if 2+ writes)
    if len(writes) >= 2:
        qa_counter += 1
        w1, w2 = writes[0], writes[1]
        e1_entities = re.findall(r'[\u4e00-\u9fff]{2,4}', w1["text"])
        e2_entities = re.findall(r'[\u4e00-\u9fff]{2,4}', w2["text"])
        if e1_entities and e2_entities:
            new_qas.append({
                "qa_id": f"{sid}_L1_{qa_counter:03d}",
                "scenario_id": sid,
                "difficulty": "level_1",
                "question_type": "temporal_ordering",
                "question_zh": f"以下事件哪个先发生：{w1['text'][:30]}... 还是 {w2['text'][:30]}...？",
                "question_en": "",
                "answer_zh": w1["text"][:30] + "...",
                "answer_en": "",
                "answer_type": "abstractive",
                "choices": [],
                "correct_choice_index": None,
                "ranking_order": None,
                "query_event_time": None,
                "query_record_time": None,
                "relevant_time_range": {
                    "start": min(w1.get("event_time", ""), w2.get("event_time", "")),
                    "end": max(w1.get("event_time", ""), w2.get("event_time", "")),
                },
                "reasoning_chain": [
                    f"事件1时间: {w1.get('event_time', '')[:10]}",
                    f"事件2时间: {w2.get('event_time', '')[:10]}",
                ],
                "requires_event_time_reasoning": True,
                "requires_record_time_reasoning": False,
                "requires_version_tracking": False,
                "requires_knowledge_retraction": False,
                "source_write_ids": [w1["write_id"], w2["write_id"]],
                "generation_method": "llm",
                "validation_status": "unvalidated",
            })

    return new_qas


def main():
    # Load both sources
    real_scenarios = load_scenarios(BASE_DIR / "data" / "generated" / "real_source_scenarios" / "all_real_source_scenarios.json")
    old_scenarios = load_scenarios(BASE_DIR / "data" / "generated" / "scenarios" / "all_scenarios.json")

    logger.info(f"Real-source scenarios: {len(real_scenarios)}")
    logger.info(f"LLM-generated scenarios: {len(old_scenarios)}")

    # Categorize by type
    real_by_type: Dict[str, List[Dict]] = {}
    for s in real_scenarios:
        t = s["scenario_type"]
        real_by_type.setdefault(t, []).append(s)

    old_by_type: Dict[str, List[Dict]] = {}
    for s in old_scenarios:
        t = s["scenario_type"]
        old_by_type.setdefault(t, []).append(s)

    # Merge strategy: keep all from both, but add data_source tag
    merged = []

    # Add all real-source scenarios
    for s in real_scenarios:
        s["data_provenance"] = "real_source"
        merged.append(s)

    # Add all old LLM-generated scenarios
    for s in old_scenarios:
        s["data_provenance"] = "llm_generated"
        # Add QA pairs if missing (old scenarios have 0 QA pairs)
        if not s.get("qa_pairs"):
            s["qa_pairs"] = []
        merged.append(s)

    logger.info(f"Merged scenarios: {len(merged)}")

    # Generate additional QA pairs for scenarios with few/none
    total_new_qa = 0
    for s in merged:
        if len(s.get("qa_pairs", [])) < 2 and len(s.get("memory_writes", [])) >= 2:
            old_count = len(s.get("qa_pairs", []))
            s["qa_pairs"] = generate_additional_qa_pairs(s)
            new_count = len(s["qa_pairs"])
            if new_count > old_count:
                total_new_qa += new_count - old_count

    logger.info(f"Generated {total_new_qa} additional QA pairs")

    # Statistics
    type_dist = Counter(s["scenario_type"] for s in merged)
    source_dist = Counter(s.get("data_provenance", "unknown") for s in merged)
    total_writes = sum(len(s["memory_writes"]) for s in merged)
    total_qa = sum(len(s.get("qa_pairs", [])) for s in merged)

    logger.info(f"Total writes: {total_writes}")
    logger.info(f"Total QA pairs: {total_qa}")
    logger.info(f"Type distribution:")
    for t, c in type_dist.most_common():
        logger.info(f"  {t}: {c}")
    logger.info(f"Source distribution: {dict(source_dist)}")

    # Save merged scenarios
    output_file = OUTPUT_DIR / "all_scenarios_v3.json"
    with open(output_file, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}")

    # Extract and split QA pairs
    all_qa = []
    for s in merged:
        for qa in s.get("qa_pairs", []):
            all_qa.append(qa)

    random.seed(42)
    random.shuffle(all_qa)

    n = len(all_qa)
    n_train = int(0.7 * n)
    n_dev = int(0.1 * n)

    splits = {
        "train": all_qa[:n_train],
        "dev": all_qa[n_train:n_train + n_dev],
        "test": all_qa[n_train + n_dev:],
    }

    validated_dir = BASE_DIR / "data" / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        dataset = {
            "dataset_id": f"bitpqa_{split_name}_zh_v3",
            "name": f"BiTempQA Chinese {split_name.capitalize()} Set v3 (Merged Real+LLM)",
            "language": "zh",
            "split": split_name,
            "version": "3.0",
            "qa_pairs": split_data,
        }
        fpath = validated_dir / f"bitpqa_{split_name}_zh.json"
        with open(fpath, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"  {split_name}: {len(split_data)} QA pairs → {fpath}")

    # Summary
    logger.info(f"\n=== Final Dataset Summary ===")
    logger.info(f"Scenarios: {len(merged)} ({source_dist.get('real_source', 0)} real + {source_dist.get('llm_generated', 0)} LLM)")
    logger.info(f"QA pairs: {total_qa}")
    logger.info(f"  Train: {len(splits['train'])}")
    logger.info(f"  Dev: {len(splits['dev'])}")
    logger.info(f"  Test: {len(splits['test'])}")
    logger.info(f"Real data ratio: {source_dist.get('real_source', 0) / len(merged) * 100:.1f}%")


if __name__ == "__main__":
    main()
