"""LoCoMo benchmark loader.

LoCoMo (Long Conversations, Multiple Topics) is a long-context conversational
QA benchmark from ACL 2024. It contains 10 long conversations (~300 turns each)
with 1,986 multi-choice questions across 5 question types:
  single_hop, multi_hop, temporal_reasoning, open_domain, adversarial

Reference: https://arxiv.org/abs/2406.01308
Dataset: https://huggingface.co/datasets/Percena/locomo-mc10
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.benchmarks.base import (
    BenchmarkData,
    BenchmarkQuestion,
    Conversation,
)

logger = logging.getLogger(__name__)


class LoCoMoLoader(BenchmarkData):
    """Load LoCoMo benchmark from HuggingFace or local cache.

    Usage:
        loader = LoCoMoLoader()  # auto-downloads from HuggingFace
        loader = LoCoMoLoader(local_path="data/locomo_mc10.json")

        conversations = loader.get_conversations()
        questions = loader.get_questions()

        # Filter by question type
        temporal_qs = loader.get_questions(question_type="temporal_reasoning")
    """

    # LoCoMo always uses 10 choices
    NUM_CHOICES = 10

    def __init__(
        self,
        local_path: Optional[str] = None,
        hf_repo: str = "Percena/locomo-mc10",
        hf_file: str = "data/locomo_mc10.json",
    ):
        self.hf_repo = hf_repo
        self.hf_file = hf_file
        self._items: List[dict] = []
        self._conversations: Dict[str, Conversation] = {}
        self._questions: List[BenchmarkQuestion] = []

        if local_path and Path(local_path).exists():
            self._load_local(local_path)
        else:
            self._load_from_hf()

        self._parse()

    def _load_local(self, path: str) -> None:
        """Load from local JSONL file."""
        logger.info(f"Loading LoCoMo from local path: {path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._items.append(json.loads(line))
        logger.info(f"Loaded {len(self._items)} items")

    def _load_from_hf(self) -> None:
        """Download and load from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading LoCoMo from {self.hf_repo}...")
        path = hf_hub_download(self.hf_repo, self.hf_file, repo_type="dataset")
        self._load_local(path)

    def _parse(self) -> None:
        """Parse raw items into conversations and questions."""
        # Group by conversation
        conv_ids: Set[str] = set()
        for item in self._items:
            conv_id = item["question_id"].rsplit("_q", 1)[0]
            conv_ids.add(conv_id)

        # Build conversations (deduplicate — all questions from same conv share sessions)
        for conv_id in conv_ids:
            # Get the first item for this conv to extract sessions
            first_item = None
            for item in self._items:
                if item["question_id"].rsplit("_q", 1)[0] == conv_id:
                    first_item = item
                    break

            if first_item is None:
                continue

            turns: List[Dict[str, str]] = []
            sessions = first_item.get("haystack_sessions", [])
            datetimes = first_item.get("haystack_session_datetimes", [])

            for i, session in enumerate(sessions):
                ts = datetimes[i] if i < len(datetimes) else ""
                for turn in session:
                    turns.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                        "timestamp": ts,
                    })

            self._conversations[conv_id] = Conversation(
                conversation_id=conv_id,
                turns=turns,
            )

        # Build questions
        for item in self._items:
            conv_id = item["question_id"].rsplit("_q", 1)[0]
            self._questions.append(BenchmarkQuestion(
                question_id=item["question_id"],
                conversation_id=conv_id,
                question=item["question"],
                choices=item.get("choices", []),
                correct_choice_index=item.get("correct_choice_index"),
                gold_answer=item.get("answer", ""),
                question_type=item.get("question_type", ""),
            ))

        logger.info(
            f"Parsed {len(self._conversations)} conversations, "
            f"{len(self._questions)} questions"
        )

    def get_conversations(self) -> List[Conversation]:
        return list(self._conversations.values())

    def get_questions(
        self,
        question_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> List[BenchmarkQuestion]:
        """Get questions, optionally filtered by type or conversation."""
        qs = self._questions
        if question_type:
            qs = [q for q in qs if q.question_type == question_type]
        if conversation_id:
            qs = [q for q in qs if q.conversation_id == conversation_id]
        return qs

    def get_question_types(self) -> List[str]:
        """Return unique question types."""
        return sorted(set(q.question_type for q in self._questions))

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        from collections import Counter

        qt = Counter(q.question_type for q in self._questions)
        conv_turns = {cid: len(c.turns) for cid, c in self._conversations.items()}

        return {
            "total_questions": len(self._questions),
            "total_conversations": len(self._conversations),
            "question_types": dict(qt),
            "avg_turns_per_conv": (
                sum(conv_turns.values()) / len(conv_turns) if conv_turns else 0
            ),
            "total_turns": sum(conv_turns.values()),
        }
