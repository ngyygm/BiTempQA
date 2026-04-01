"""Deduplicator and dataset splitter for QA pairs.

Uses embedding similarity to detect near-duplicate QA pairs,
then splits into train/dev/test sets with stratified sampling.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def deduplicate_by_question(
    qa_pairs: list, similarity_threshold: float = 0.9
) -> Tuple[list, list]:
    """Deduplicate QA pairs based on question text similarity.

    Uses simple character n-gram Jaccard similarity to avoid
    requiring an embedding model for basic deduplication.

    Returns (unique_pairs, duplicate_pairs).
    """
    def char_ngrams(text: str, n: int = 3) -> set:
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def jaccard(s1: set, s2: set) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    seen_signatures: List[Tuple[str, set]] = []
    unique = []
    duplicates = []

    for qa in qa_pairs:
        question = getattr(qa, "question_zh", "") or (qa.get("question_zh", "") if isinstance(qa, dict) else "")
        ngrams = char_ngrams(question)

        is_dup = False
        for sig_id, sig_ngrams in seen_signatures:
            sim = jaccard(ngrams, sig_ngrams)
            if sim >= similarity_threshold:
                is_dup = True
                duplicates.append(qa)
                logger.debug(f"Duplicate: {getattr(qa, 'qa_id', '?')} ~ {sig_id} (sim={sim:.2f})")
                break

        if not is_dup:
            seen_signatures.append((getattr(qa, "qa_id", "?"), ngrams))
            unique.append(qa)

    logger.info(f"Deduplication: {len(unique)} unique, {len(duplicates)} duplicates removed")
    return unique, duplicates


def stratified_split(
    qa_pairs: list,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, list]:
    """Split QA pairs into train/dev/test with stratification by difficulty and scenario_type."""

    rng = random.Random(seed)

    # Group by (difficulty, scenario_type) for stratification
    groups: Dict[Tuple[str, str], list] = defaultdict(list)
    for qa in qa_pairs:
        difficulty = getattr(qa, "difficulty", "unknown")
        if hasattr(difficulty, "value"):
            difficulty = difficulty.value
        scenario_id = getattr(qa, "scenario_id", "")
        scenario_type = scenario_id.split("_")[0] if scenario_id else "unknown"
        groups[(difficulty, scenario_type)].append(qa)

    splits = {"train": [], "dev": [], "test": []}

    for key, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_dev = max(1, int(n * dev_ratio))

        splits["train"].extend(group[:n_train])
        splits["dev"].extend(group[n_train:n_train + n_dev])
        splits["test"].extend(group[n_train + n_dev:])

    # Shuffle each split
    for split_name in splits:
        rng.shuffle(splits[split_name])

    total = sum(len(v) for v in splits.values())
    logger.info(f"Split: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])} (total={total})")
    return splits


def deduplicate_and_split(
    input_path: Path,
    output_dir: Path,
    similarity_threshold: float = 0.9,
    seed: int = 42,
) -> Dict[str, list]:
    """Load QA pairs, deduplicate, and split into train/dev/test."""

    data = json.loads(input_path.read_text(encoding="utf-8"))
    pairs = data if isinstance(data, list) else data.get("qa_pairs", [])

    logger.info(f"Loaded {len(pairs)} QA pairs from {input_path}")

    # Deduplicate
    unique, dups = deduplicate_by_question(pairs, similarity_threshold)

    # Stratified split
    splits = stratified_split(unique, seed=seed)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_pairs in splits.items():
        out_path = output_dir / f"bitpqa_{split_name}_zh.json"
        out_data = {
            "dataset_id": f"bitpqa_{split_name}_zh",
            "name": f"BiTempQA Chinese {split_name.title()} Set",
            "language": "zh",
            "split": split_name,
            "qa_pairs": [p.model_dump() if hasattr(p, "model_dump") else p for p in split_pairs],
            "total_count": len(split_pairs),
        }
        out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(split_pairs)} pairs to {out_path}")

    # Save duplicates log
    dup_log = output_dir / "duplicates_log.json"
    dup_log.write_text(
        json.dumps(
            [{"qa_id": getattr(p, "qa_id", p.get("qa_id")), "question": getattr(p, "question_zh", p.get("question_zh", ""))} for p in dups],
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    return splits
