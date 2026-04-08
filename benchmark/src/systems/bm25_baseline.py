"""BM25 baseline — term-frequency-based retrieval using rank_bm25."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from rank_bm25 import BM25Okapi

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False
    logger.warning("jieba not installed, falling back to character-level tokenization")


class BM25Baseline(MemorySystem):
    """BM25 retrieval baseline using jieba tokenization for Chinese text."""

    def __init__(self, name: str = "BM25"):
        super().__init__(name)
        self.raw_texts: List[str] = []
        self.tokenized_texts: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.top_k: int = 5

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text using jieba for Chinese, whitespace for non-CJK."""
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text)
        if has_cjk:
            if _JIEBA_AVAILABLE:
                return list(jieba.cut(text))
            else:
                # Fallback: character-level + bigrams
                chars = list(text)
                bigrams = [text[i:i+2] for i in range(len(text) - 1)]
                return chars + bigrams
        else:
            return text.split()

    def remember(self, text: str, event_time: str, record_time: Optional[str] = None, source_name: str = "scenario_trace") -> str:
        self.raw_texts.append(text)
        self.tokenized_texts.append(self._tokenize(text))
        # Rebuild BM25 index (rank_bm25 doesn't support incremental updates)
        self.bm25 = BM25Okapi(self.tokenized_texts)
        return f"bm25_{len(self.raw_texts)}"

    def query(
        self,
        question: str,
        query_event_time: Optional[str] = None,
        query_record_time: Optional[str] = None,
        time_before: Optional[str] = None,
        time_after: Optional[str] = None,
    ) -> QueryResult:
        start = time.time()
        if not self.bm25 or not self.raw_texts:
            return QueryResult(answer="", latency_ms=(time.time() - start) * 1000)

        query_tokens = self._tokenize(question)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        retrieved = [self.raw_texts[i] for i in top_indices if scores[i] > 0]

        if not retrieved:
            # Fallback: return all texts if no BM25 matches
            retrieved = self.raw_texts[:self.top_k]

        context = "\n".join(retrieved)
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0
        confidence = min(max_score / 10.0, 1.0)  # Normalize to [0, 1]

        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=retrieved,
            confidence=confidence,
            latency_ms=(time.time() - start) * 1000,
        )

    def reset(self) -> None:
        self.raw_texts.clear()
        self.tokenized_texts.clear()
        self.bm25 = None
