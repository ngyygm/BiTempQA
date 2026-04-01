"""Simple KG baseline — entity/relation extraction with overwrite updates, no versioning."""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class SimpleKGBaseline(MemorySystem):
    """Simple knowledge graph with overwrite updates, no version tracking."""

    def __init__(self, name: str = "Simple KG"):
        super().__init__(name)
        # entity_name -> {attr: value}  (always latest version)
        self.entities: Dict[str, Dict[str, str]] = defaultdict(dict)
        # (entity1, entity2) -> relation_description
        self.relations: Dict[tuple, str] = {}
        # Raw texts for retrieval
        self.texts: List[str] = []
        # Entity mentions in each text
        self.text_entities: List[Set[str]] = []

    def remember(self, text: str, event_time: str, source_name: str = "scenario_trace") -> str:
        self.texts.append(text)
        self._extract_and_update(text)
        return f"kg_{len(self.texts)}"

    def _extract_and_update(self, text: str) -> None:
        """Simple rule-based entity extraction and KG update.

        Uses heuristic patterns common in Chinese scenario texts.
        """
        # Pattern: X是/担任/在Y做Z
        patterns = [
            r"(\S+?)(?:是|担任|在)(\S+?)(?:的|，|。|$)",
            r"(\S+?)(?:加入|跳槽到|就职于)(\S+?)(?:[，。]|$)",
            r"(\S+?)和(\S+?)(?:建立了|是|成为|结束)(.+?)[。]",
        ]

        mentioned: Set[str] = set()

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    entity, value = match
                    mentioned.add(entity)
                    self.entities[entity]["last_known"] = value
                elif len(match) == 3:
                    e1, e2, rel = match
                    mentioned.add(e1)
                    mentioned.add(e2)
                    self.relations[(e1, e2)] = rel

        self.text_entities.append(mentioned)

    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        start = time.time()

        # Find relevant texts by entity overlap
        q_entities: Set[str] = set()
        for entity in self.entities:
            if entity in question:
                q_entities.add(entity)

        relevant_texts = []
        if q_entities:
            for i, text in enumerate(self.texts):
                if q_entities & self.text_entities[i]:
                    relevant_texts.append(text)

        # If no entity match, return all texts
        if not relevant_texts:
            relevant_texts = self.texts[-3:]  # Last 3 as fallback

        # Build KG context
        kg_context_parts = []
        for entity in q_entities or list(self.entities.keys())[:3]:
            attrs = self.entities.get(entity, {})
            if attrs:
                kg_context_parts.append(f"{entity}: {attrs}")
            for (e1, e2), rel in self.relations.items():
                if entity in (e1, e2):
                    kg_context_parts.append(f"{e1}-{e2}: {rel}")

        context = "\n".join(relevant_texts + kg_context_parts)

        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=relevant_texts,
            latency_ms=(time.time() - start) * 1000,
        )

    def reset(self) -> None:
        self.entities.clear()
        self.relations.clear()
        self.texts.clear()
        self.text_entities.clear()
