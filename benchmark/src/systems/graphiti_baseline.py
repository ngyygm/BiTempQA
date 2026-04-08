"""Graphiti baseline -- wraps graphiti-core library as a MemorySystem.

Graphiti is Zep's open-source temporal knowledge graph.
Requires Neo4j database running. Uses SiliconFlow API for LLM backend.
Uses a patched LLM client to avoid beta.chat.completions.parse.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, List, Optional

from src.systems.base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


# Persistent event loop for Graphiti async operations
# Graphiti internally uses semaphore_gather which needs consistent event loop
_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def _run_async(coro):
    """Run an async coroutine on the persistent event loop."""
    loop = _get_loop()
    return loop.run_until_complete(coro)


def _parse_datetime(time_str: str) -> datetime | None:
    """Parse ISO 8601 datetime string."""
    if not time_str:
        return None
    try:
        for fmt in [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(time_str.rstrip("Z"), fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(time_str.rstrip("Z"))
    except Exception:
        return None


class SiliconFlowRerankerClient:
    """Simple cross-encoder using embedding similarity for SiliconFlow compat."""

    def __init__(self, api_key: str, base_url: str, model: str = "Pro/BAAI/bge-m3"):
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        config = OpenAIEmbedderConfig(api_key=api_key, embedding_model=model, base_url=base_url)
        self.embedder = OpenAIEmbedder(config=config)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        import numpy as np
        from graphiti_core.helpers import semaphore_gather

        q_emb = await self.embedder.create(query)
        p_embs = await semaphore_gather(*[self.embedder.create(p) for p in passages])

        q = np.array(q_emb)
        scores = []
        for i, p_emb in enumerate(p_embs):
            p = np.array(p_emb)
            sim = float(np.dot(q, p) / (np.linalg.norm(q) * np.linalg.norm(p) + 1e-8))
            scores.append((passages[i], sim))

        scores.sort(reverse=True, key=lambda x: x[1])
        return scores


class SiliconFlowLLMClient:
    """Custom LLM client using chat.completions.create for SiliconFlow compat."""

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-ai/DeepSeek-V3"):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = 0.0
        self._tracer = None

    @staticmethod
    def _flatten_nested(obj):
        """Recursively flatten nested dicts in entity attributes to primitive types.

        LLMs sometimes output entity attributes as nested JSON Schema objects like:
          {"summary": {"description": "...", "title": "Summary", "type": "string"}}
        Neo4j only accepts primitive types, so we flatten these to simple strings.
        """
        if isinstance(obj, dict):
            return {k: SiliconFlowLLMClient._flatten_nested(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SiliconFlowLLMClient._flatten_nested(item) for item in obj]
        elif isinstance(obj, dict):
            return str(obj)
        else:
            return obj

    @staticmethod
    def _flatten_for_neo4j(data):
        """Walk parsed JSON and flatten any nested dict values that Neo4j can't store.

        For entity attribute dicts, convert nested dict values to their most useful
        string representation.
        """
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    # Try to extract the most descriptive field
                    for field in ("description", "summary", "value", "content", "text"):
                        if field in v and isinstance(v[field], str):
                            result[k] = v[field]
                            break
                    else:
                        # Fallback: recurse or stringify
                        if any(isinstance(vv, dict) for vv in v.values()):
                            result[k] = SiliconFlowLLMClient._flatten_for_neo4j(v)
                        else:
                            result[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, list):
                    result[k] = [SiliconFlowLLMClient._flatten_for_neo4j(item) for item in v]
                else:
                    result[k] = v
            return result
        elif isinstance(data, list):
            return [SiliconFlowLLMClient._flatten_for_neo4j(item) for item in data]
        else:
            return data

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=16384,
    ):
        openai_messages = []
        for m in messages:
            content = m.content if hasattr(m, "content") else str(m)
            role = m.role if hasattr(m, "role") else "user"
            if isinstance(content, str):
                clean = content.replace("\\n", "\n")
            else:
                clean = content
            openai_messages.append({"role": role, "content": clean})

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        if response_model is not None:
            from pydantic import BaseModel

            if issubclass(response_model, BaseModel):
                # Use json_object instead of json_schema — SiliconFlow
                # ignores json_schema constraints, but json_object at least forces valid JSON.
                kwargs["response_format"] = {"type": "json_object"}
                # Inject schema hint + explicit instruction for flat attribute values
                schema = response_model.model_json_schema()
                schema_str = json.dumps(schema, ensure_ascii=False)
                schema_hint = (
                    f"\n\nIMPORTANT: You MUST respond with valid JSON matching this exact schema:\n"
                    f"{schema_str}\n"
                    f"Do NOT use different field names. Use the exact field names from the schema above.\n"
                    f"CRITICAL: All attribute values MUST be simple strings, NOT nested objects or JSON Schema. "
                    f"For example, write {{\"summary\": \"some text\"}}, NOT {{\"summary\": {{\"description\": \"some text\", \"type\": \"string\"}}}}.\n"
                )
                if openai_messages and openai_messages[0]["role"] == "system":
                    openai_messages[0]["content"] += schema_hint
                else:
                    openai_messages.insert(0, {"role": "system", "content": schema_hint})

        response = await self.client.chat.completions.create(**kwargs)
        result_text = response.choices[0].message.content or ""

        # Clean markdown code blocks if present
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        if response_model is not None:
            try:
                parsed = json.loads(cleaned)
                # Flatten any nested dict values that Neo4j can't store
                parsed = self._flatten_for_neo4j(parsed)
                # Validate against Pydantic model
                from pydantic import BaseModel
                if issubclass(response_model, BaseModel):
                    try:
                        validated = response_model.model_validate(parsed)
                        return validated.model_dump()
                    except Exception as val_err:
                        logger.debug(f"Pydantic validation failed: {val_err}, returning raw parsed")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse structured response: {e}")
                raise
        else:
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                return {"content": result_text, "usage": {}}

    async def generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=16384,
    ):
        """Public interface called by Graphiti."""
        return await self._generate_response(messages, response_model, max_tokens)

    def set_tracer(self, tracer):
        self._tracer = tracer


class GraphitiBaseline(MemorySystem):
    """Graphiti (Zep) baseline using graphiti-core library.

    Builds a temporal knowledge graph from episodes using LLM entity extraction.
    Uses Neo4j as graph store and SiliconFlow as LLM backend.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "tmg2024secure",
    ):
        super().__init__(name="Graphiti")
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self._init_graphiti()

    def _init_graphiti(self):
        """Initialize the Graphiti instance with SiliconFlow LLM client."""
        from graphiti_core import Graphiti
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

        llm_client = SiliconFlowLLMClient(
            api_key=api_key,
            base_url=base_url,
            model="Qwen/Qwen2.5-32B-Instruct",
        )

        embedder_config = OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model="Pro/BAAI/bge-m3",
            base_url=base_url,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        cross_encoder = SiliconFlowRerankerClient(
            api_key=api_key,
            base_url=base_url,
            model="Pro/BAAI/bge-m3",
        )

        self.graphiti = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

        try:
            _run_async(self.graphiti.build_indices_and_constraints())
        except Exception as e:
            logger.warning(f"Graphiti build_indices error: {e}")

        logger.info("Graphiti initialized")

    def remember(self, text: str, event_time: str = "", record_time: Optional[str] = None, source_name: str = "scenario_trace") -> str:
        """Add an episode to Graphiti's knowledge graph.

        Uses record_time as reference_time (when the system learned about this fact).
        Falls back to event_time if record_time is not provided.
        """
        try:
            # reference_time = when the episode was observed by the system = record_time
            ref_time = _parse_datetime(record_time) if record_time else _parse_datetime(event_time)
            if ref_time is None:
                ref_time = datetime.now()
            _run_async(
                self.graphiti.add_episode(
                    name=f"ep_{record_time or event_time or int(time.time())}",
                    episode_body=text,
                    source_description=source_name,
                    reference_time=ref_time,
                )
            )
            return f"graphiti_ep_{record_time or event_time}"
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
        """Search Graphiti for relevant facts."""
        start_time = time.time()

        try:
            search_kwargs = {
                "query": question,
                "num_results": 5,
            }
            # Pass temporal filters to Graphiti's temporal search
            if time_after:
                start_dt = _parse_datetime(time_after)
                if start_dt:
                    search_kwargs["start_time"] = start_dt
            if time_before:
                end_dt = _parse_datetime(time_before)
                if end_dt:
                    search_kwargs["end_time"] = end_dt

            results = _run_async(
                self.graphiti.search(**search_kwargs)
            )

            facts: List[str] = []
            if isinstance(results, list):
                for edge in results:
                    if hasattr(edge, "fact") and edge.fact:
                        facts.append(edge.fact)
                    elif isinstance(edge, str):
                        facts.append(edge)

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
        """Clear graph data and reinitialize."""
        try:
            _run_async(self.graphiti.close())
        except Exception as e:
            logger.warning(f"Graphiti close error: {e}")

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            with driver.session() as session:
                # Batch delete to avoid OOM on large graphs
                for _ in range(100):
                    result = session.run(
                        "MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(n) as cnt"
                    )
                    if result.single()["cnt"] == 0:
                        break
            driver.close()
        except Exception as e:
            logger.warning(f"Graphiti Neo4j clear error: {e}")

        self._init_graphiti()
        logger.info("Graphiti reset complete")
