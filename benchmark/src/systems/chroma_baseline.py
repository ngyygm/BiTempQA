"""ChromaDB baseline — metadata filtering with limited temporal support."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class ChromaBaseline(MemorySystem):
    """ChromaDB with time metadata filtering (limited temporal awareness)."""

    def __init__(
        self,
        name: str = "ChromaDB",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
    ):
        super().__init__(name)
        self.encoder = SentenceTransformer(embedding_model)
        self.top_k = top_k

        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="bitpqa_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._all_docs: List[str] = []

    def remember(self, text: str, event_time: str, source_name: str = "scenario_trace") -> str:
        write_id = f"chroma_{self.collection.count()}"
        emb = self.encoder.encode([text]).tolist()
        # Store event_time as ISO string AND as numeric timestamp for filtering
        ts = self._iso_to_timestamp(event_time)
        self.collection.add(
            ids=[write_id],
            documents=[text],
            embeddings=emb,
            metadatas=[{
                "event_time": event_time,
                "event_ts": ts,
                "source": source_name,
            }],
        )
        self._all_docs.append(text)
        return write_id

    @staticmethod
    def _iso_to_timestamp(iso_str: str) -> float:
        """Convert ISO 8601 string to Unix timestamp for numeric comparison."""
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0

    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        start = time.time()

        where_filter = {}
        if time_before and time_after:
            where_filter["$and"] = [
                {"event_ts": {"$lt": self._iso_to_timestamp(time_before)}},
                {"event_ts": {"$gt": self._iso_to_timestamp(time_after)}},
            ]
        elif time_before:
            where_filter["event_ts"] = {"$lt": self._iso_to_timestamp(time_before)}
        elif time_after:
            where_filter["event_ts"] = {"$gt": self._iso_to_timestamp(time_after)}

        kwargs = {
            "query_texts": [question],
            "n_results": min(self.top_k, self.collection.count() or 1),
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = self.collection.query(**kwargs)
            docs = results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            docs = self._all_docs[-self.top_k:]  # Fallback to recent docs

        context = "\n".join(docs)
        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=docs,
            latency_ms=(time.time() - start) * 1000,
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection("bitpqa_memory")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="bitpqa_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._all_docs = []
