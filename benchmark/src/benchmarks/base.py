"""Base interfaces for external benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Conversation:
    """A conversation to ingest into a memory system."""

    conversation_id: str
    turns: List[Dict[str, str]] = field(default_factory=list)
    # Each turn: {"role": "user"/"assistant", "content": "...", "timestamp": "..."}


@dataclass
class BenchmarkQuestion:
    """A question from an external benchmark."""

    question_id: str
    conversation_id: str
    question: str
    choices: List[str] = field(default_factory=list)
    correct_choice_index: Optional[int] = None
    gold_answer: str = ""
    question_type: str = ""  # e.g., single_hop, multi_hop, temporal_reasoning


class BenchmarkData(ABC):
    """Abstract interface for an external benchmark dataset."""

    @abstractmethod
    def get_conversations(self) -> List[Conversation]:
        """Return conversations to ingest into memory systems."""
        ...

    @abstractmethod
    def get_questions(self) -> List[BenchmarkQuestion]:
        """Return questions to evaluate."""
        ...


def ingest_conversation(system, conversation: Conversation) -> int:
    """Ingest a Conversation into a MemorySystem.

    Returns the number of turns ingested.
    """
    count = 0
    for turn in conversation.turns:
        system.remember(
            text=turn["content"],
            event_time=turn.get("timestamp", ""),
            source_name=f"benchmark_{conversation.conversation_id}",
        )
        count += 1
    return count
