"""FAISS vector store baseline — pure semantic retrieval, no temporal awareness."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class FAISSBaseline(MemorySystem):
    """FAISS-based vector store with no temporal reasoning."""

    def __init__(
        self,
        name: str = "FAISS Vector Store",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
    ):
        super().__init__(name)
        self.encoder = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

        # Lazy import faiss
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def remember(self, text: str, event_time: str, source_name: str = "scenario_trace") -> str:
        self.texts.append(text)
        emb = self.encoder.encode([text], normalize_embeddings=True)
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self._rebuild_index()
        return f"faiss_{len(self.texts)}"

    def _rebuild_index(self) -> None:
        dim = self.embeddings.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        start = time.time()
        if self.index is None or len(self.texts) == 0:
            return QueryResult(answer="", latency_ms=(time.time() - start) * 1000)

        q_emb = self.encoder.encode([question], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), min(self.top_k, len(self.texts)))

        retrieved = []
        for idx in indices[0]:
            if idx >= 0:
                retrieved.append(self.texts[idx])

        return QueryResult(
            answer="\n".join(retrieved),
            retrieved_context="\n".join(retrieved),
            retrieved_facts=retrieved,
            confidence=float(scores[0][0]) if len(scores[0]) > 0 else 0.0,
            latency_ms=(time.time() - start) * 1000,
        )

    def reset(self) -> None:
        self.texts = []
        self.embeddings = None
        self.index = None
