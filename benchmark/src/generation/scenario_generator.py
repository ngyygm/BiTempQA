"""Scenario generator using SiliconFlow LLM API.

Generates complete Scenario objects from the 10 scenario type skeletons,
using Qwen2.5-72B-Instruct for data generation.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from src.schemas import Scenario, ScenarioType

logger = logging.getLogger(__name__)

SCENARIO_TYPE_CONFIG = {
    ScenarioType.ENTITY_ATTRIBUTE_EVOLUTION: {"count": 15, "file": "S01_entity_attribute_evolution.md"},
    ScenarioType.RELATIONSHIP_EVOLUTION: {"count": 12, "file": "S02_relationship_evolution.md"},
    ScenarioType.CONTRADICTORY_INFORMATION: {"count": 12, "file": "S03_contradictory_information.md"},
    ScenarioType.LATE_ARRIVING_FACTS: {"count": 15, "file": "S04_late_arriving_facts.md"},
    ScenarioType.FUTURE_DATED_INFORMATION: {"count": 12, "file": "S05_future_dated_information.md"},
    ScenarioType.ENTITY_IDENTITY_RESOLUTION: {"count": 10, "file": "S06_entity_identity_resolution.md"},
    ScenarioType.KNOWLEDGE_RETRACTION: {"count": 12, "file": "S07_knowledge_retraction.md"},
    ScenarioType.MULTI_SOURCE_INFORMATION: {"count": 12, "file": "S08_multi_source_information.md"},
    ScenarioType.GRADUAL_ACCUMULATION: {"count": 12, "file": "S09_gradual_accumulation.md"},
    ScenarioType.TEMPORAL_AMBIGUITY: {"count": 10, "file": "S10_temporal_ambiguity.md"},
}

DOMAINS = ["corporate", "academic", "social", "fictional", "historical", "scientific"]


class ScenarioGenerator:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _load_prompt_template(self, template_path: Path) -> str:
        return template_path.read_text(encoding="utf-8")

    def _load_scenario_skeleton(self, skeleton_path: Path) -> str:
        return skeleton_path.read_text(encoding="utf-8")

    def _build_generation_prompt(
        self,
        skeleton: str,
        scenario_type: ScenarioType,
        index: int,
        domain: Optional[str] = None,
        seed_prompt: str = "",
    ) -> str:
        type_name = scenario_type.value.replace("_", " ")
        type_code = f"S{list(SCENARIO_TYPE_CONFIG.keys()).index(scenario_type) + 1:02d}"

        domain_hint = ""
        if domain:
            domain_hint = f"\n\n**本次生成的领域**: {domain}（请围绕该领域构建场景）"

        return f"""{seed_prompt}

---

## 本次生成任务

**场景类型**: {scenario_type.value} ({type_name})
**场景编号**: {type_code}_{index:03d}
**domain**: {domain or "从以下领域中选择一个: corporate, academic, social, fictional, historical, scientific"}{domain_hint}

## 场景类型骨架

{skeleton}

## 要求

请严格按照上面的JSON Schema输出一个完整的Scenario对象。确保：
1. scenario_id = "{type_code}_{index:03d}"
2. scenario_type = "{scenario_type.value}"
3. memory_writes中的text字段是自然流畅的中文
4. event_time和record_time使用ISO 8601格式
5. world_states准确反映每个record_time时刻的Agent认知
6. entity_ground_truth包含所有实体的完整版本链

只输出JSON，不要输出其他文字。"""

    def _call_llm(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的基准测试数据生成专家。只输出JSON格式数据，不要输出其他文字。"},
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

    def _parse_scenario(self, raw: str) -> Scenario:
        # Extract JSON from potential markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

        data = json.loads(raw)
        self._normalize_ground_truth(data)
        return Scenario(**data)

    @staticmethod
    def _normalize_ground_truth(data: dict) -> None:
        """Fix common LLM formatting errors in ground truth fields."""
        # relation_ground_truth values should be lists
        rgt = data.get("relation_ground_truth", {})
        if isinstance(rgt, dict):
            for k, v in rgt.items():
                if isinstance(v, dict):
                    rgt[k] = [v]
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "relation_id" not in item:
                            item["relation_id"] = k
        # entity_ground_truth values should be lists
        egt = data.get("entity_ground_truth", {})
        if isinstance(egt, dict):
            for k, v in egt.items():
                if isinstance(v, dict):
                    egt[k] = [v]

    def generate_one(
        self,
        scenario_type: ScenarioType,
        index: int,
        skeleton: str,
        seed_prompt: str,
        domain: Optional[str] = None,
    ) -> Scenario:
        prompt = self._build_generation_prompt(skeleton, scenario_type, index, domain, seed_prompt)
        raw = self._call_llm(prompt)
        scenario = self._parse_scenario(raw)

        # Override IDs to ensure consistency
        type_code = f"S{list(SCENARIO_TYPE_CONFIG.keys()).index(scenario_type) + 1:02d}"
        scenario.scenario_id = f"{type_code}_{index:03d}"
        scenario.scenario_type = scenario_type

        return scenario

    def generate_batch(
        self,
        scenario_type: ScenarioType,
        count: int,
        start_index: int,
        skeletons_dir: Path,
        seed_prompt_path: Path,
        domains: Optional[List[str]] = None,
        max_workers: int = 3,
    ) -> List[Scenario]:
        skeleton_file = SCENARIO_TYPE_CONFIG[scenario_type]["file"]
        skeleton = self._load_scenario_skeleton(skeletons_dir / skeleton_file)
        seed_prompt = self._load_prompt_template(seed_prompt_path)

        def _gen_one(idx: int) -> Optional[Scenario]:
            domain = domains[idx % len(domains)] if domains else None
            logger.info(f"Generating {scenario_type.value} scenario {idx}...")
            try:
                s = self.generate_one(scenario_type, idx, skeleton, seed_prompt, domain)
                logger.info(f"  Generated {s.scenario_id}: {s.title_zh}")
                return s
            except Exception as e:
                logger.error(f"  Failed to generate scenario {idx}: {e}")
                return None

        scenarios = []
        indices = list(range(start_index, start_index + count))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_one, i): i for i in indices}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scenarios.append(result)

        # Sort by index
        scenarios.sort(key=lambda s: int(s.scenario_id.split("_")[1]))
        return scenarios


def generate_all_scenarios(
    output_dir: Path,
    skeletons_dir: Path,
    seed_prompt_path: Path,
    api_base: str,
    api_key: str,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    start_index: int = 1,
    skip_completed: bool = True,
    max_workers: int = 3,
) -> List[Scenario]:
    generator = ScenarioGenerator(api_base=api_base, api_key=api_key, model=model)

    all_scenarios = []
    for stype, config in SCENARIO_TYPE_CONFIG.items():
        type_code = f"S{list(SCENARIO_TYPE_CONFIG.keys()).index(stype) + 1:02d}"
        out_file = output_dir / f"{type_code}_{stype.value}.json"

        # Skip if already completed and has enough scenarios
        if skip_completed and out_file.exists():
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            if len(existing) >= config["count"]:
                logger.info(f"=== Skipping {stype.value} (already {len(existing)} scenarios) ===")
                all_scenarios.extend([Scenario(**s) for s in existing])
                continue

        logger.info(f"=== Generating {stype.value} ({config['count']} scenarios) ===")
        scenarios = generator.generate_batch(
            scenario_type=stype,
            count=config["count"],
            start_index=start_index,
            skeletons_dir=skeletons_dir,
            seed_prompt_path=seed_prompt_path,
            domains=DOMAINS,
            max_workers=max_workers,
        )
        all_scenarios.extend(scenarios)

        # Save intermediate results
        type_code = f"S{list(SCENARIO_TYPE_CONFIG.keys()).index(stype) + 1:02d}"
        out_file = output_dir / f"{type_code}_{stype.value}.json"
        out_file.write_text(
            json.dumps([s.model_dump() for s in scenarios], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"  Saved {len(scenarios)} scenarios to {out_file}")

    # Save all scenarios
    all_file = output_dir / "all_scenarios.json"
    all_file.write_text(
        json.dumps([s.model_dump() for s in all_scenarios], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Total: {len(all_scenarios)} scenarios saved to {all_file}")

    return all_scenarios
