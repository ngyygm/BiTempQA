"""Simple KG baseline — entity/relation extraction with versioned storage."""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class SimpleKGBaseline(MemorySystem):
    """Simple knowledge graph with versioned entity attribute storage.

    Each entity attribute keeps a full history of (value, valid_from, valid_until)
    instead of overwriting, enabling temporal queries.
    """

    def __init__(self, name: str = "Simple KG"):
        super().__init__(name)
        # entity_name -> {attr: [(value, valid_from, valid_until, source_write_id), ...]}
        self.entities: Dict[str, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
        # (entity1, entity2) -> [(relation_desc, valid_from, valid_until), ...]
        self.relations: Dict[tuple, List[tuple]] = defaultdict(list)
        # Raw texts for retrieval, with event_time and record_time
        self.texts: List[str] = []
        self.text_metadata: List[Dict] = []  # [{event_time, record_time, source}, ...]
        # Entity mentions in each text
        self.text_entities: List[Set[str]] = []
        # Track write order for versioning
        self._write_counter: int = 0

    def remember(
        self,
        text: str,
        event_time: str,
        record_time: Optional[str] = None,
        source_name: str = "scenario_trace",
    ) -> str:
        self._write_counter += 1
        write_id = f"kg_{self._write_counter}"
        self.texts.append(text)
        self.text_metadata.append({
            "event_time": event_time,
            "record_time": record_time or event_time,
            "source": source_name,
        })
        self._extract_and_update(text, event_time, record_time or event_time)
        return write_id

    def _extract_and_update(self, text: str, event_time: str, record_time: str) -> None:
        """Rule-based entity extraction with versioned KG update."""
        patterns = [
            (r"(\S+?)(?:是|担任|在)(\S+?)(?:的|，|。|$)", "attr"),
            (r"(\S+?)(?:加入|跳槽到|就职于)(\S+?)(?:[，。]|$)", "attr"),
            (r"(\S+?)和(\S+?)(?:建立了|是|成为|结束)(.+?)[。]", "rel"),
        ]

        mentioned: Set[str] = set()

        for pattern, ptype in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if ptype == "attr" and len(match) == 2:
                    entity, value = match
                    mentioned.add(entity)
                    # Close previous version if exists
                    attr_versions = self.entities[entity]["main"]
                    if attr_versions:
                        last = attr_versions[-1]
                        # Update valid_until of previous version
                        attr_versions[-1] = (last[0], last[1], event_time, last[3])
                    # Add new version
                    attr_versions.append((value, event_time, None, text))
                elif ptype == "rel" and len(match) == 3:
                    e1, e2, rel = match
                    mentioned.add(e1)
                    mentioned.add(e2)
                    key = (e1, e2)
                    # Close previous relation if exists
                    if self.relations[key]:
                        last = self.relations[key][-1]
                        self.relations[key][-1] = (last[0], last[1], event_time)
                    self.relations[key].append((rel, event_time, None))

        self.text_entities.append(mentioned)

    def _get_entity_state_at(self, entity: str, as_of_time: Optional[str] = None) -> Dict[str, str]:
        """Get entity attributes as of a given time (by record_time)."""
        result = {}
        for attr, versions in self.entities[entity].items():
            for value, valid_from, valid_until, _source in reversed(versions):
                if as_of_time is None or valid_from <= as_of_time:
                    if valid_until is None or valid_until > as_of_time:
                        result[attr] = value
                        break
        return result

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
                    # Apply temporal filter if specified
                    meta = self.text_metadata[i]
                    if time_after and meta.get("record_time", "") < time_after:
                        continue
                    if time_before and meta.get("record_time", "") > time_before:
                        continue
                    relevant_texts.append(text)

        # If no entity match, return recent texts
        if not relevant_texts:
            relevant_texts = self.texts[-3:]

        # Build KG context using temporal-aware entity state
        as_of = query_record_time or time_before
        kg_context_parts = []
        for entity in q_entities or list(self.entities.keys())[:3]:
            state = self._get_entity_state_at(entity, as_of)
            if state:
                kg_context_parts.append(f"{entity}: {state}")
            for (e1, e2), versions in self.relations.items():
                if entity in (e1, e2):
                    for rel_desc, valid_from, valid_until in reversed(versions):
                        if as_of is None or valid_from <= as_of:
                            kg_context_parts.append(f"{e1}-{e2}: {rel_desc}")
                            break

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
        self.text_metadata.clear()
        self.text_entities.clear()
        self._write_counter = 0
