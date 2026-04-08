"""Base interface for memory systems under evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """Result from a memory system query."""
    answer: str = ""
    retrieved_context: str = ""
    retrieved_facts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemorySystem(ABC):
    """Abstract interface for a memory system under evaluation."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def remember(
        self,
        text: str,
        event_time: str,
        record_time: Optional[str] = None,
        source_name: str = "scenario_trace",
    ) -> str:
        """Store a memory with event_time and record_time.

        Args:
            text: The natural language content to remember.
            event_time: When the described event occurred (valid time).
            record_time: When the system recorded this fact (transaction time).
            source_name: Provenance tag for the memory.

        Returns a write_id or task identifier.
        """
        ...

    @abstractmethod
    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        """Query the memory system.

        Args:
            question: Natural language question
            query_event_time: The event time the question refers to
            query_record_time: The record time to query as-of
            time_before: Only consider memories with event_time before this
            time_after: Only consider memories with event_time after this

        Returns:
            QueryResult with answer and context
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memories."""
        ...

    def ingest_scenario(self, scenario) -> None:
        """Ingest all memory writes from a scenario, ordered by record_time."""
        writes = sorted(scenario.memory_writes, key=lambda w: w.record_time)
        for write in writes:
            self.remember(
                text=write.text,
                event_time=write.event_time,
                record_time=write.record_time,
                source_name=write.source_name,
            )
