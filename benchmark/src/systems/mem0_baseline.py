"""Mem0 baseline — wraps mem0ai library as a MemorySystem.

Uses Mem0's vector store + LLM-based memory extraction pipeline.
Configured to use SiliconFlow API for LLM backend.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from src.systems.base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class Mem0Baseline(MemorySystem):
    """Mem0 baseline using mem0ai library.

    Mem0 extracts structured memories from conversations using LLM,
    stores them in a vector DB (Qdrant by default), and retrieves
    relevant memories via semantic search.
    """

    USER_ID = "benchmark_user"

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedder_model: Optional[str] = None,
    ):
        super().__init__(name="Mem0")

        # Set environment variables for Mem0's OpenAI client
        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        if llm_base_url:
            os.environ["OPENAI_BASE_URL"] = llm_base_url

        # Build config dict for Mem0
        from mem0.configs.base import LlmConfig, EmbedderConfig

        llm_config = LlmConfig(
            provider="openai",
            config={
                "model": llm_model or "deepseek-ai/DeepSeek-V3",
                "openai_base_url": llm_base_url or "https://api.siliconflow.cn/v1",
                "api_key": llm_api_key or os.environ.get("OPENAI_API_KEY", ""),
                "temperature": 0.0,
            },
        )

        embedder_config = None
        if embedder_model:
            embedder_config = EmbedderConfig(
                provider="openai",
                config={"model": embedder_model},
            )

        from mem0 import Memory
        self.m = Memory(config=None)  # type: ignore

        # Override the LLM config
        self.m.config.llm = llm_config
        if embedder_config:
            self.m.config.embedder = embedder_config

        logger.info(f"Mem0 initialized (model={llm_model or 'default'})")

    def remember(self, text: str, event_time: str = "", source_name: str = "scenario_trace") -> str:
        """Store a memory using Mem0's add() API."""
        try:
            result = self.m.add(
                messages=[{"role": "user", "content": text}],
                user_id=self.USER_ID,
                metadata={
                    "event_time": event_time,
                    "source": source_name,
                },
                # Disable inference for speed in benchmark
                infer=False,
            )
            # result is a dict with "results" key
            if isinstance(result, dict) and "results" in result:
                memories = result["results"]
                if memories:
                    return str(memories[0].get("id", ""))
            return f"mem0_{int(time.time())}"
        except Exception as e:
            logger.warning(f"Mem0 add error: {e}")
            return ""

    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        """Query memories using Mem0's search() API."""
        start_time = time.time()

        try:
            # Build filters for temporal queries
            filters = {}
            if time_before or time_after:
                metadata_filter = {}
                if time_before:
                    metadata_filter["event_time"] = {"$lt": time_before}
                if time_after:
                    metadata_filter["event_time"] = {"$gt": time_after}
                filters["metadata"] = metadata_filter

            result = self.m.search(
                query=question,
                user_id=self.USER_ID,
                limit=10,
                filters=filters if filters else None,
            )

            facts: List[str] = []
            if isinstance(result, dict) and "results" in result:
                for r in result["results"]:
                    memory_text = r.get("memory", "")
                    if memory_text:
                        facts.append(memory_text)
            elif isinstance(result, list):
                for r in result:
                    if isinstance(r, dict):
                        facts.append(r.get("memory", r.get("text", "")))

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
            logger.warning(f"Mem0 search error: {e}")
            return QueryResult(
                answer="",
                retrieved_context="",
                retrieved_facts=[],
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    def reset(self) -> None:
        """Clear all stored memories for the benchmark user."""
        try:
            self.m.delete_all(user_id=self.USER_ID)
            logger.info("Mem0 reset complete")
        except Exception as e:
            logger.warning(f"Mem0 reset error: {e}")
