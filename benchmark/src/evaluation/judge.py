"""LLM Judge — scores answers via LLM-as-a-judge.

Supports two modes:
1. context_judge (legacy): Checks if retrieved context supports ground truth
2. answer_judge (new): Compares LLM-generated answer against ground truth

The answer_judge mode is used in the unified evaluation pipeline
(retrieve -> LLM generate -> judge), following Mem0/Zep methodology.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


# --- Context Judge (legacy mode) ---

CONTEXT_JUDGE_PROMPT = """你是一个公正的评判者。请判断系统的回答是否正确。

## 问题
{question}

## 标准答案
{ground_truth}

## 系统检索到的上下文信息
{retrieved_context}

## 判断标准
- 如果系统检索到的上下文信息**包含或隐含**标准答案的核心内容，判定为正确
- 如果上下文信息**与标准答案矛盾**或**完全无关**，判定为错误
- 如果上下文信息**部分支持**标准答案（涵盖核心要点但不完整），也判定为正确
- 只关注上下文是否包含正确信息，不评判上下文的质量或完整性

## 请按以下格式回复
判断：正确/错误
置信度：高/中/低
理由：一句话说明"""


# --- Answer Judge (new mode, based on Mem0 paper Appendix A) ---

ANSWER_JUDGE_SYSTEM = """你是一个公正严格的评判者。你的任务是判断一个系统生成的答案是否正确。
请严格按照格式回复。"""

ANSWER_JUDGE_PROMPT = """你的任务是判断系统生成的答案是否正确。

## 问题
{question}

## 标准答案
{gold_answer}

## 系统生成的答案
{generated_answer}

## 判断标准
- 只要系统答案涉及与标准答案相同的主题，即判定为正确
- 对于时间相关问题，只要指代相同的日期或时间段，即使格式不同也应判定为正确（如"5月7日"和"7 May"应视为相同）
- 如果系统答案比标准答案长但包含正确核心信息，应判定为正确
- 如果系统答案与标准答案无关或矛盾，判定为错误

请先提供一句推理，然后输出 CORRECT 或 WRONG。
以 JSON 格式返回，key 为 "label"。"""


def parse_context_judge_response(response: str) -> Tuple[bool, str, str]:
    """Parse context judge response (legacy Chinese format)."""
    if not response:
        return False, "low", "empty response"

    is_correct = False
    judge_match = re.search(r"判断[：:]\s*(正确|错误)", response)
    if judge_match:
        is_correct = judge_match.group(1) == "正确"
    else:
        if "正确" in response[:50] and "错误" not in response[:50]:
            is_correct = True
        elif "错误" in response[:50]:
            is_correct = False

    confidence = "low"
    conf_match = re.search(r"置信度[：:]\s*(高|中|低)", response)
    if conf_match:
        confidence = conf_match.group(1)

    reason = ""
    reason_match = re.search(r"理由[：:]\s*(.+?)(?:\n|$)", response)
    if reason_match:
        reason = reason_match.group(1).strip()

    return is_correct, confidence, reason


def parse_answer_judge_response(response: str) -> Tuple[bool, str]:
    """Parse answer judge response (JSON format with 'label' key).

    Returns (is_correct, reason).
    """
    if not response:
        return False, "empty response"

    # Try JSON extraction
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^}]*"label"\s*:\s*"(CORRECT|WRONG)"[^}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            label = data.get("label", "WRONG")
            return label == "CORRECT", response[:200]
    except json.JSONDecodeError:
        pass

    # Fallback: look for CORRECT/WRONG keywords
    if "CORRECT" in response.upper() and "WRONG" not in response.upper()[:50]:
        return True, response[:200]
    elif "WRONG" in response.upper():
        return False, response[:200]

    # Last resort
    if "正确" in response[:50]:
        return True, response[:200]
    return False, response[:200]


class LLMJudge:
    """LLM-as-a-judge for QA evaluation.

    Supports two judging modes:
    - context_judge: Checks if retrieved context supports ground truth (legacy)
    - answer_judge: Compares LLM-generated answer vs ground truth (new, default)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.0,
        max_workers: int = 3,
        timeout: int = 60,
        cache_path: Optional[Path] = None,
        mode: str = "answer_judge",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
        self.mode = mode  # "context_judge" or "answer_judge"
        self.cache: Dict[str, dict] = {}
        self.cache_path = cache_path
        self._load_cache()

    def _cache_key(self, qa_id: str, system_name: str) -> str:
        return f"{qa_id}|||{system_name}"

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.cache = data
                logger.info(f"Loaded {len(data)} cached judge results")
            except Exception as e:
                logger.warning(f"Failed to load judge cache: {e}")

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # --- Context Judge (legacy API, backward compatible) ---

    def judge_single(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        qa_id: str,
        system_name: str,
    ) -> dict:
        """Judge a single QA pair using context judge mode (legacy).

        Returns dict with is_correct, confidence, reason.
        """
        key = self._cache_key(qa_id, system_name)
        if key in self.cache:
            return self.cache[key]

        prompt = CONTEXT_JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            retrieved_context=retrieved_context[:1000],
        )

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个公正严格的评判者。请严格按照格式回复。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=256,
                )
                response_text = resp.choices[0].message.content or ""
                is_correct, confidence, reason = parse_context_judge_response(response_text)

                result = {
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "reason": reason,
                    "raw_response": response_text,
                    "mode": "context_judge",
                }
                self.cache[key] = result
                return result

            except Exception as e:
                logger.warning(f"Judge API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        result = {
            "is_correct": False,
            "confidence": "low",
            "reason": "API error after 3 retries",
            "raw_response": "",
            "mode": "context_judge",
        }
        self.cache[key] = result
        return result

    # --- Answer Judge (new API) ---

    def judge_answer(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
        qa_id: str,
        system_name: str,
    ) -> dict:
        """Judge a single QA pair using answer judge mode.

        Compares the LLM-generated answer against the gold answer.

        Returns dict with is_correct, reason, raw_response.
        """
        key = self._cache_key(qa_id, system_name)
        if key in self.cache:
            return self.cache[key]

        prompt = ANSWER_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ANSWER_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=256,
                )
                response_text = resp.choices[0].message.content or ""
                is_correct, reason = parse_answer_judge_response(response_text)

                result = {
                    "is_correct": is_correct,
                    "confidence": "high" if is_correct else "low",
                    "reason": reason,
                    "raw_response": response_text,
                    "mode": "answer_judge",
                }
                self.cache[key] = result
                return result

            except Exception as e:
                logger.warning(f"Answer judge API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        result = {
            "is_correct": False,
            "confidence": "low",
            "reason": "API error after 3 retries",
            "raw_response": "",
            "mode": "answer_judge",
        }
        self.cache[key] = result
        return result

    # --- Batch judging ---

    def judge_batch(
        self,
        items: list,
        progress_callback=None,
    ) -> Dict[str, dict]:
        """Judge multiple QA pairs concurrently.

        Args:
            items: list of dicts. For context_judge mode:
                question, ground_truth, retrieved_context, qa_id, system_name
                For answer_judge mode (auto-detected):
                question, gold_answer, generated_answer, qa_id, system_name
            progress_callback: optional callback(completed, total)

        Returns:
            dict mapping cache_key -> result
        """
        to_judge = []
        for item in items:
            key = self._cache_key(item["qa_id"], item["system_name"])
            if key not in self.cache:
                to_judge.append(item)

        if not to_judge:
            logger.info("All items already cached")
            return self.cache

        logger.info(f"Judging {len(to_judge)} items ({len(items) - len(to_judge)} cached)")

        completed = 0
        total = len(to_judge)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for item in to_judge:
                # Auto-detect mode from item keys
                if "generated_answer" in item:
                    future = executor.submit(
                        self.judge_answer,
                        item["question"],
                        item["gold_answer"],
                        item["generated_answer"],
                        item["qa_id"],
                        item["system_name"],
                    )
                else:
                    future = executor.submit(
                        self.judge_single,
                        item["question"],
                        item["ground_truth"],
                        item["retrieved_context"],
                        item["qa_id"],
                        item["system_name"],
                    )
                futures[future] = item

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    item = futures[future]
                    key = self._cache_key(item["qa_id"], item["system_name"])
                    self.cache[key] = {
                        "is_correct": False,
                        "confidence": "low",
                        "reason": f"Thread error: {e}",
                        "raw_response": "",
                        "mode": "unknown",
                    }
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                elif completed % 10 == 0:
                    logger.info(f"Judge progress: {completed}/{total}")

        self._save_cache()
        return self.cache
