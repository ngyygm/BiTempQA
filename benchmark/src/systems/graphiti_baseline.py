"""Graphiti baseline — wraps graphiti-core library as a MemorySystem.

Graphiti is Zep's open-source temporal knowledge graph.
Requires Neo4j database running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List, Optional

from src.systems.base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class GraphitiBaseline(MemorySystem):
    """Graphiti (Zep) baseline using graphiti-core library.

    Graphiti builds a temporal knowledge graph from episodes,
    with nodes for entities and edges for relationships.
    Requires a running Neo4j instance.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
    ):
        super().__init__(name="Graphiti")

        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Ensure OpenAI env vars are set for Graphiti's LLM client
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ.get(
                "SILICONFLOW_API_KEY", ""
            )
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"

        self._init_graphiti()

    def _init_graphiti(self):
        """Initialize or re-initialize the Graphiti instance."""
        from graphiti_core import Graphiti

        self.graphiti = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
        )
        logger.info("Graphiti initialized")

    def remember(self, text: str, event_time: str = "", source_name: str = "scenario_trace") -> str:
        """Add an episode to Graphiti."""
        try:
            ref_time = event_time if event_time else None
            _run_async(
                self.graphiti.add_episode(
                    name=f"ep_{event_time or int(time.time())}",
                    episode_body=text,
                    source=source_name,
                    reference_time=ref_time,
                )
            )
            return f"graphiti_ep_{event_time}"
        except Exception as e:
            logger.warning(f"Graphiti add_episode error: {e}")
            return ""

    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        """Search Graphiti for relevant context."""
        start_time = time.time()

        try:
            results = _run_async(
                self.graphiti.search(
                    query=question,
                    num_results=10,
                )
            )

            facts: List[str] = []
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        # Graphiti returns fact-like results
                        fact = r.get("fact", r.get("content", r.get("text", "")))
                        if fact:
                            facts.append(fact)
                    elif isinstance(r, str):
                        facts.append(r)

            retrieved_context = "\n".join(facts) if facts else ""

            return QueryResult(
                answer=retrieved_context,
                retrieved_context=retrieved_context,
                retrieved_facts=facts,
                confidence=1.0 if facts else 0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"num_results": len(facts)},
            )

        except Exception as e:
            logger.warning(f"Graphiti search error: {e}")
            return QueryResult(
                answer="",
                retrieved_context="",
                retrieved_facts=[],
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    def reset(self) -> None:
        """Close and reinitialize Graphiti (clears graph)."""
        try:
            _run_async(self.graphiti.close())
        except Exception as e:
            logger.warning(f"Graphiti close error: {e}")

        # Reinitialize with fresh connection (clears graph state)
        self._init_graphiti()
        logger.info("Graphiti reset complete")
