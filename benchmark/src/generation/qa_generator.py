"""QA pair generator using SiliconFlow LLM API.

Generates question-answer pairs from Scenario objects for the BiTempQA benchmark.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from src.schemas import QAPair, QADataset, Scenario

logger = logging.getLogger(__name__)

# Thread-safe lock for intermediate file writes
_write_lock = threading.Lock()


class QAGenerator:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 16384,
        timeout: int = 300,
        max_retries: int = 5,
        retry_delay: float = 10.0,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _build_prompt(self, scenario: Scenario, seed_prompt: str) -> str:
        scenario_json = json.dumps(
            scenario.model_dump(), ensure_ascii=False, indent=2
        )

        return f"""{seed_prompt}

---

## 待生成QA的场景

{scenario_json}

## 生成要求

为以上场景生成12-15个QA对。确保：
1. 每个qa_id格式为"{scenario.scenario_id}_L{{N}}_{{NNN}}"
2. 难度分布: ~5个Level 1, ~4个Level 2, ~3个Level 3
3. 答案类型: ~80%多选题, ~15%抽象题, ~5%判断题
4. 每个QA对必须有完整的reasoning_chain
5. 正确标注requires_event_time_reasoning和requires_record_time_reasoning
6. 多选题的干扰选项要基于错误的时间推理设计
7. 至少30%的QA对需要record_time推理

只输出JSON数组，格式:
```json
{{"qa_pairs": [...]}}
```"""

    def _call_llm(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的基准测试QA生成专家。只输出JSON格式数据，不要输出其他文字。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def _parse_qa_pairs(self, raw: str) -> List[QAPair]:
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

        # Try direct parse first
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt repair of truncated JSON
            raw = self._repair_truncated_json(raw)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"JSON repair failed: {e}")
                return []

        pairs_data = data if isinstance(data, list) else data["qa_pairs"]

        pairs = []
        for i, pd in enumerate(pairs_data):
            try:
                pair = QAPair(**pd)
                pairs.append(pair)
            except Exception as e:
                logger.warning(f"Failed to parse QA pair {i}: {e}")
                continue

        return pairs

    @staticmethod
    def _repair_truncated_json(raw: str) -> str:
        """Attempt to repair truncated JSON by closing open structures."""
        # Remove trailing incomplete content after the last complete QA pair
        # Find the last complete "}," or "}" pattern that ends a qa pair object
        # Strategy: try to close open brackets/braces

        # Count open/close braces and brackets
        open_braces = raw.count("{")
        close_braces = raw.count("}")
        open_brackets = raw.count("[")
        close_brackets = raw.count("]")

        repaired = raw.rstrip()

        # Remove trailing incomplete string
        # If there's an odd number of quotes in the tail, we're mid-string
        # Simple heuristic: remove everything after the last complete value
        while True:
            stripped = repaired.rstrip()
            if not stripped:
                break
            last_char = stripped[-1]
            if last_char in ",: \t\n\r":
                repaired = stripped[:-1]
            else:
                break

        # If we still have unbalanced structures, try removing the last incomplete object
        if repaired.count("{") > repaired.count("}"):
            # Find the last complete "}," and truncate there
            last_complete = repaired.rfind("},")
            if last_complete > 0:
                repaired = repaired[:last_complete + 1]  # keep the closing brace
            elif repaired.rfind("]") > 0:
                # Try truncating at last ]
                repaired = repaired[:repaired.rfind("]") + 1]

        # Close remaining open structures
        deficit_braces = repaired.count("{") - repaired.count("}")
        deficit_brackets = repaired.count("[") - repaired.count("]")

        # Remove trailing comma before closing
        repaired = repaired.rstrip().rstrip(",")

        repaired += "}" * deficit_braces + "]" * deficit_brackets
        return repaired

    def generate_for_scenario(
        self, scenario: Scenario, seed_prompt: str
    ) -> List[QAPair]:
        prompt = self._build_prompt(scenario, seed_prompt)
        raw = self._call_llm(prompt)
        pairs = self._parse_qa_pairs(raw)

        # Assign IDs if not set
        level_counters = {"level_1": 0, "level_2": 0, "level_3": 0}
        for pair in pairs:
            if not pair.qa_id or not pair.qa_id.startswith(scenario.scenario_id):
                level_counters[pair.difficulty.value] += 1
                level_num = pair.difficulty.value.split("_")[1]
                pair.qa_id = (
                    f"{scenario.scenario_id}_L{level_num}"
                    f"_{level_counters[pair.difficulty.value]:03d}"
                )
            pair.scenario_id = scenario.scenario_id

        return pairs


def _load_intermediate(output_path: Path) -> Dict[str, List[dict]]:
    """Load previously saved intermediate results for resume support."""
    if not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        pairs_by_scenario: Dict[str, List[dict]] = {}
        for p in data.get("qa_pairs", []):
            sid = p.get("scenario_id", "")
            pairs_by_scenario.setdefault(sid, []).append(p)
        return pairs_by_scenario
    except Exception:
        return {}


def generate_qa_for_scenarios(
    scenarios: List[Scenario],
    output_path: Path,
    seed_prompt_path: Path,
    api_base: str,
    api_key: str,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    max_workers: int = 3,
    timeout: int = 300,
    skip_completed: bool = True,
) -> QADataset:
    generator = QAGenerator(api_base=api_base, api_key=api_key, model=model, timeout=timeout)
    seed_prompt = seed_prompt_path.read_text(encoding="utf-8")

    # Load existing results for resume
    existing = _load_intermediate(output_path) if skip_completed else {}
    completed_ids = set(existing.keys())
    logger.info(f"Loaded {len(existing)} scenarios with existing QA pairs")

    all_pairs: List[QAPair] = []
    # Re-add existing pairs
    for sid, pairs_data in existing.items():
        for pd in pairs_data:
            try:
                all_pairs.append(QAPair(**pd))
            except Exception as e:
                logger.warning(f"Failed to reload QA pair: {e}")

    # Filter scenarios to process
    to_process = [s for s in scenarios if s.scenario_id not in completed_ids]
    if skip_completed and len(to_process) < len(scenarios):
        logger.info(f"Skipping {len(scenarios) - len(to_process)} already completed scenarios")
    if not to_process:
        logger.info("All scenarios already processed")

    def _gen_qa(scenario: Scenario) -> Optional[List[QAPair]]:
        logger.info(f"Generating QA for {scenario.scenario_id}: {scenario.title_zh}")
        try:
            pairs = generator.generate_for_scenario(scenario, seed_prompt)
            logger.info(f"  {scenario.scenario_id}: {len(pairs)} QA pairs generated")
            return pairs
        except Exception as e:
            logger.error(f"  {scenario.scenario_id} failed: {e}")
            return None

    def _save_intermediate(pairs: List[QAPair]) -> None:
        """Thread-safe intermediate save after each scenario completes."""
        with _write_lock:
            current_pairs = [p.model_dump() for p in all_pairs]
            for p in pairs:
                current_pairs.append(p.model_dump())
            # Deduplicate by qa_id
            seen = set()
            unique = []
            for p in current_pairs:
                if p["qa_id"] not in seen:
                    seen.add(p["qa_id"])
                    unique.append(p)
            unique.sort(key=lambda p: p["qa_id"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"qa_pairs": unique}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_gen_qa, s): s for s in to_process}
        completed_count = 0
        for future in as_completed(futures):
            scenario = futures[future]
            pairs = future.result()
            if pairs:
                all_pairs.extend(pairs)
                _save_intermediate(pairs)
                completed_count += 1
                logger.info(
                    f"  Progress: {completed_count}/{len(to_process)} scenarios done, "
                    f"{len(all_pairs)} total QA pairs"
                )

    # Final sort and save
    all_pairs.sort(key=lambda p: p.qa_id)

    dataset = QADataset(
        dataset_id="bitpqa_generated_zh",
        name="BiTempQA Chinese Generated",
        language="zh",
        split="test",
        qa_pairs=all_pairs,
    )
    dataset.compute_stats()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        dataset.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(f"Total: {len(all_pairs)} QA pairs saved to {output_path}")
    logger.info(f"Difficulty distribution: {dataset.difficulty_counts}")

    return dataset
