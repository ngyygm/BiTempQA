"""Answer extractor — extracts system answers and maps to choice indices."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class AnswerExtractor:
    """Extract structured answers from system responses for evaluation."""

    def extract_choice_index(
        self, response: str, choices: Optional[List[str]] = None
    ) -> Optional[int]:
        """Extract the selected choice index from a system response.

        Strategies:
        1. Look for explicit option markers (A/B/C/D or 1/2/3/4)
        2. Fuzzy match against choice text
        3. Find choice text directly in response
        """
        if not choices or not response:
            return None

        # Strategy 1: Explicit markers
        # Try Chinese markers: A、B、C、D or 甲、乙、丙、丁
        markers = [
            r"选\s*[项]?([A-D])",
            r"([A-D])\s*[选项]",
            r"答案[是为：:]\s*([A-D])",
            r"选择\s*[：:]\s*([A-D])",
            r"\(([A-D])\)",
            r"（([A-D])）",
            r"第([1-4])个",
        ]
        for pattern in markers:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                marker = match.group(1).upper()
                if marker in ("A", "B", "C", "D"):
                    idx = ord(marker) - ord("A")
                    if idx < len(choices):
                        return idx
                elif marker in ("1", "2", "3", "4"):
                    idx = int(marker) - 1
                    if idx < len(choices):
                        return idx

        # Strategy 2: Fuzzy match choice text
        response_lower = response.lower()
        best_idx = None
        best_len = 0
        for i, choice in enumerate(choices):
            # Check for exact substring match
            if choice in response:
                if len(choice) > best_len:
                    best_idx = i
                    best_len = len(choice)

        if best_idx is not None:
            return best_idx

        # Strategy 3: Partial match (at least 2 chars overlap)
        for i, choice in enumerate(choices):
            if len(choice) >= 2:
                # Check if key terms from choice appear in response
                for j in range(len(choice) - 1):
                    if choice[j:j+2] in response:
                        return i

        return None

    def extract_boolean(self, response: str) -> Optional[bool]:
        """Extract a boolean answer from response."""
        positive_patterns = [
            r"(是|对|正确|true|yes|确实|会的)",
            r"(不是|不对|错误|false|no|不会|没有)",
        ]
        for pattern in positive_patterns[:1]:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        for pattern in positive_patterns[1:]:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        return None

    def extract_text_answer(self, response: str) -> str:
        """Extract the core answer text from response."""
        # Try to find a concise answer
        patterns = [
            r"答案[是为：:]\s*(.+?)(?:\n|。|$)",
            r"(?:因此|所以|综上)[，,]?\s*(.+?)(?:\n|。|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        # Fallback: first sentence
        first_sentence = re.split(r"[。\n]", response, maxsplit=1)[0]
        return first_sentence.strip()
