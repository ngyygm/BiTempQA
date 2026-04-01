"""LLM-based answer generation from retrieved context.

All systems use the same LLM to generate answers from retrieved context,
ensuring fair comparison of retrieval quality across systems.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# --- Prompt Templates ---

MC_SYSTEM_PROMPT = """你是一个选择题答题助手。你只能根据提供的记忆上下文来选择正确答案。
你必须以JSON格式回复，格式为：{"answer": "X"}
其中X是选项字母（A/B/C/D等）。不要输出任何其他内容。"""

MC_USER_PROMPT = """## 记忆上下文
{retrieved_context}

## 问题
{question}

## 选项
{choices_text}

请根据记忆上下文选择正确答案，以JSON格式回复：{{"answer": "字母"}}
只回复JSON，不要有任何其他文字。"""

ABSTRACTIVE_SYSTEM_PROMPT = """你是一个记忆助手。你只能根据提供的记忆上下文来回答问题。
回答要简洁，不超过5-6个词。"""

ABSTRACTIVE_USER_PROMPT = """# 指令
1. 仔细分析所有提供的记忆
2. 特别注意时间戳来确定答案
3. 如果记忆包含矛盾信息，优先使用最新的记忆
4. 将相对时间引用转换为具体日期
5. 回答不超过5-6个词

## 记忆上下文
{retrieved_context}

## 问题
{question}

## 回答"""


def _format_choices(choices: List[str]) -> str:
    """Format choices as A/B/C/D list."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, choice in enumerate(choices):
        if i < len(labels):
            lines.append(f"{labels[i]}. {choice}")
    return "\n".join(lines)


def parse_mc_answer(response: str, num_choices: int) -> Optional[int]:
    """Parse multi-choice answer from LLM response.

    Returns 0-indexed choice index, or None if unparseable.
    """
    if not response:
        return None

    response = response.strip()

    max_letter = chr(ord("A") + num_choices - 1)
    valid_range = f"A-{max_letter}"

    def _to_idx(letter: str) -> Optional[int]:
        idx = ord(letter.upper()) - ord("A")
        return idx if 0 <= idx < num_choices else None

    # Strategy 1: Single letter only (A / B / C / ...)
    match = re.search(rf"^([{valid_range}])$", response, re.IGNORECASE)
    if match:
        idx = _to_idx(match.group(1))
        if idx is not None:
            return idx

    # Strategy 2: Letter with punctuation prefix (A. / A) / A） / A: )
    match = re.search(rf"^([{valid_range}])[\.\)）:\s]", response, re.IGNORECASE)
    if match:
        idx = _to_idx(match.group(1))
        if idx is not None:
            return idx

    # Strategy 3: Chinese patterns - "答案是X" / "选择X" / "选X" / "正确答案是X"
    match = re.search(r"(?:正确)?答案(?:是|为|：|:)?\s*([{valid_range}])", response, re.IGNORECASE)
    if match:
        idx = _to_idx(match.group(1))
        if idx is not None:
            return idx

    match = re.search(r"(?:选择|选|应该选)(?:的)?(?:答案)?(?:是|为|：|:)?\s*([{valid_range}])", response, re.IGNORECASE)
    if match:
        idx = _to_idx(match.group(1))
        if idx is not None:
            return idx

    # Strategy 4: "选项X" / "option X" pattern
    match = re.search(r"(?:选项|option)\s*([{valid_range}])", response, re.IGNORECASE)
    if match:
        idx = _to_idx(match.group(1))
        if idx is not None:
            return idx

    # Strategy 5: Last letter in the response (likely the final answer)
    # Scan from end to find the last valid letter
    for m in re.finditer(rf"[{valid_range}]", response, re.IGNORECASE):
        idx = _to_idx(m.group(0))
        if idx is not None:
            return idx

    # Strategy 6: First valid letter in response (fallback)
    match = re.search(rf"[{valid_range}]", response, re.IGNORECASE)
    if match:
        idx = _to_idx(match.group(0))
        if idx is not None:
            return idx

    logger.warning(f"Could not parse MC answer from: {response[:100]}")
    return None


class AnswerGenerator:
    """Generates answers from retrieved context using LLM."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        question: str,
        retrieved_context: str,
        choices: Optional[List[str]] = None,
    ) -> str:
        """Generate answer from retrieved context.

        Args:
            question: The question to answer
            retrieved_context: Context retrieved by the memory system
            choices: For multi-choice, the list of options

        Returns:
            Generated answer string (letter for MC, text for abstractive)
        """
        is_mc = choices is not None and len(choices) > 0

        if is_mc:
            system_prompt = MC_SYSTEM_PROMPT
            user_prompt = MC_USER_PROMPT.format(
                retrieved_context=retrieved_context[:3000],
                question=question,
                choices_text=_format_choices(choices),
            )
            # Use small max_tokens to force brief output, no JSON mode (causes truncation on some APIs)
            raw = self._call_llm(system_prompt, user_prompt, max_tokens=50)
            return self._extract_mc_letter(raw, len(choices))
        else:
            system_prompt = ABSTRACTIVE_SYSTEM_PROMPT
            user_prompt = ABSTRACTIVE_USER_PROMPT.format(
                retrieved_context=retrieved_context[:3000],
                question=question,
            )

        for attempt in range(3):
            try:
                raw = self._call_llm(system_prompt, user_prompt)
                return raw.strip()
            except Exception as e:
                logger.warning(f"Answer generation error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return ""

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict = None,
        max_tokens: int = 256,
    ) -> str:
        """Make an LLM API call."""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _extract_mc_letter(self, raw: str, num_choices: int) -> str:
        """Extract MC letter from JSON or plain text response."""
        # Try JSON parse first
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                letter = data.get("answer", "")
                if letter and len(letter) <= 2:
                    return letter.strip().upper()
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: use regex parser
        idx = parse_mc_answer(raw, num_choices)
        if idx is not None:
            return chr(ord("A") + idx)

        return raw.strip()  # Return raw for fallback parsing

    def generate_mc(
        self,
        question: str,
        retrieved_context: str,
        choices: List[str],
    ) -> Optional[int]:
        """Generate and parse multi-choice answer. Returns 0-indexed choice."""
        raw = self.generate(question, retrieved_context, choices)
        return parse_mc_answer(raw, len(choices))
