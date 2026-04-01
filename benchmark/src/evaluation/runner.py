"""Evaluation runner — main evaluation loop for all systems.

Supports two evaluation modes:
1. legacy: System returns answer directly, text matching for MC + LLM judge for abstractive
2. unified (default): Retrieve -> LLM Generate -> LLM Judge for ALL question types

The unified mode follows Mem0/Zep methodology for fair comparison.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.evaluation.answer_extractor import AnswerExtractor
from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import compute_scenario_result, score_qa_pair
from src.schemas import (
    AnswerType,
    EvaluationRun,
    QADataset,
    QAPair,
    Scenario,
    ScenarioResult,
    SystemAnswer,
)
from src.systems.base import MemorySystem

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates the evaluation of memory systems on QA datasets."""

    def __init__(
        self,
        systems: List[MemorySystem],
        scenarios: List[Scenario],
        qa_dataset: QADataset,
        llm_judge_config: Optional[dict] = None,
        answer_generator_config: Optional[dict] = None,
        eval_mode: str = "unified",
    ):
        """Initialize runner.

        Args:
            systems: List of memory systems to evaluate
            scenarios: List of scenarios with memory writes
            qa_dataset: QA pairs to evaluate on
            llm_judge_config: Config for LLM Judge (base_url, api_key, model, etc.)
            answer_generator_config: Config for Answer Generator (base_url, api_key, model)
            eval_mode: "unified" (retrieve->generate->judge) or "legacy" (direct answer)
        """
        self.systems = systems
        self.scenarios = {s.scenario_id: s for s in scenarios}
        self.qa_dataset = qa_dataset
        self.extractor = AnswerExtractor()
        self.eval_mode = eval_mode
        self.answer_generator: Optional[AnswerGenerator] = None
        self.llm_judge_config = llm_judge_config
        self.judge: Optional[LLMJudge] = None

        # Initialize answer generator
        if answer_generator_config and eval_mode == "unified":
            self.answer_generator = AnswerGenerator(
                base_url=answer_generator_config["base_url"],
                api_key=answer_generator_config["api_key"],
                model=answer_generator_config.get("model", "deepseek-ai/DeepSeek-V3"),
                temperature=answer_generator_config.get("temperature", 0.0),
            )

        # Initialize LLM Judge
        if llm_judge_config:
            judge_mode = llm_judge_config.get("mode", "answer_judge")
            self.judge = LLMJudge(
                base_url=llm_judge_config["base_url"],
                api_key=llm_judge_config["api_key"],
                model=llm_judge_config.get("model", "deepseek-ai/DeepSeek-V3"),
                temperature=llm_judge_config.get("temperature", 0.0),
                max_workers=llm_judge_config.get("max_workers", 3),
                timeout=llm_judge_config.get("timeout", 60),
                cache_path=Path("data/eval_results/judge_cache.json"),
                mode=judge_mode,
            )

    def run(self, system: MemorySystem) -> EvaluationRun:
        """Run full evaluation for a single system."""
        run_id = f"eval_{system.name.replace(' ', '_').lower()}_{int(time.time())}"
        logger.info(f"Starting evaluation: {run_id} for {system.name} (mode={self.eval_mode})")

        run = EvaluationRun(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            system_name=system.name,
        )

        # Group QA pairs by scenario
        qa_by_scenario: Dict[str, List[QAPair]] = {}
        for qa in self.qa_dataset.qa_pairs:
            qa_by_scenario.setdefault(qa.scenario_id, []).append(qa)

        # Phase 1: Query all QA pairs and collect answers
        judge_items = []  # Items needing LLM judge
        scenario_qa_results: Dict[str, tuple] = {}  # scenario_id -> (qa_pairs, results)

        for scenario_id, qa_pairs in qa_by_scenario.items():
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                logger.warning(f"Scenario {scenario_id} not found, skipping")
                continue

            logger.info(
                f"  Scenario {scenario_id}: {scenario.title_zh} ({len(qa_pairs)} QA pairs)"
            )

            system.reset()
            system.ingest_scenario(scenario)

            results = []
            for qa in qa_pairs:
                result, sys_answer = self._evaluate_qa(system, qa)
                results.append(result)

                # Collect items for batch judging
                if self.judge:
                    if self.eval_mode == "unified":
                        # Only judge non-MC questions (MC uses letter parsing)
                        if qa.answer_type != AnswerType.MULTI_CHOICE:
                            judge_items.append({
                                "question": qa.question_zh,
                                "gold_answer": qa.answer_zh,
                                "generated_answer": sys_answer.raw_response,
                                "qa_id": qa.qa_id,
                                "system_name": system.name,
                            })
                    elif qa.answer_type == AnswerType.ABSTRACTIVE:
                        # Legacy mode: only judge abstractive
                        judge_items.append({
                            "question": qa.question_zh,
                            "ground_truth": qa.answer_zh,
                            "retrieved_context": sys_answer.retrieval_context,
                            "qa_id": qa.qa_id,
                            "system_name": system.name,
                        })

            scenario_qa_results[scenario_id] = (qa_pairs, results)

        # Phase 2: Batch LLM judge
        if judge_items and self.judge:
            logger.info(f"Running LLM Judge on {len(judge_items)} QA pairs...")
            self.judge.judge_batch(judge_items)
            logger.info("LLM Judge complete. Re-scoring with judge results...")

            # Re-score results with judge
            for scenario_id, (qa_pairs, results) in scenario_qa_results.items():
                scenario = self.scenarios.get(scenario_id)
                new_results = []
                for qa, result in zip(qa_pairs, results):
                    key = self.judge._cache_key(qa.qa_id, system.name)
                    judge_result = self.judge.cache.get(key)
                    if judge_result:
                        new_result = score_qa_pair(
                            qa_id=qa.qa_id,
                            system_name=system.name,
                            system_answer=result.answer,
                            correct_choice_index=qa.correct_choice_index,
                            ground_truth_answer=qa.answer_zh,
                            answer_type=qa.answer_type,
                            llm_judge_result=judge_result,
                        )
                        new_results.append(new_result)
                    else:
                        new_results.append(result)
                scenario_qa_results[scenario_id] = (qa_pairs, new_results)

        # Phase 3: Compute aggregates
        for scenario_id, (qa_pairs, results) in scenario_qa_results.items():
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                continue
            sr = compute_scenario_result(scenario, qa_pairs, results, system.name)
            run.scenario_results.append(sr)

        run.compute_aggregates()
        logger.info(
            f"  {system.name}: accuracy={run.overall_accuracy:.3f}, "
            f"f1={run.overall_f1:.3f}, "
            f"temporal={run.temporal_reasoning_accuracy:.3f}, "
            f"version={run.version_recall:.3f}"
        )
        return run

    def _evaluate_qa(self, system: MemorySystem, qa: QAPair):
        """Evaluate a single QA pair. Returns (QAResult, SystemAnswer)."""
        # Determine query parameters
        time_before = qa.query_event_time
        time_after = None
        if qa.relevant_time_range:
            time_before = qa.relevant_time_range.get("end") or time_before
            time_after = qa.relevant_time_range.get("start")

        # Query the memory system
        query_result = system.query(
            question=qa.question_zh,
            query_event_time=qa.query_event_time,
            query_record_time=qa.query_record_time,
            time_before=time_before,
            time_after=time_after,
        )

        if self.eval_mode == "unified" and self.answer_generator:
            return self._evaluate_qa_unified(system, qa, query_result)
        else:
            return self._evaluate_qa_legacy(system, qa, query_result)

    def _evaluate_qa_unified(self, system, qa, query_result):
        """Unified mode: retrieve -> LLM generate -> (judge later in batch)."""
        retrieved_context = query_result.retrieved_context or ""

        # Generate answer using LLM
        generated_answer = self.answer_generator.generate(
            question=qa.question_zh,
            retrieved_context=retrieved_context,
            choices=qa.choices,
        )

        # For multi-choice, also try to parse the letter
        selected_idx = None
        if qa.choices:
            from src.evaluation.answer_generator import parse_mc_answer
            selected_idx = parse_mc_answer(generated_answer, len(qa.choices))

        system_answer = SystemAnswer(
            qa_id=qa.qa_id,
            system_name=system.name,
            raw_response=generated_answer,
            extracted_answer=generated_answer,
            selected_choice_index=selected_idx,
            confidence=query_result.confidence,
            retrieval_context=retrieved_context[:2000],
            latency_ms=query_result.latency_ms,
        )

        # Initial scoring (will be overridden by judge in Phase 2)
        result = score_qa_pair(
            qa_id=qa.qa_id,
            system_name=system.name,
            system_answer=system_answer,
            correct_choice_index=qa.correct_choice_index,
            ground_truth_answer=qa.answer_zh,
            answer_type=qa.answer_type,
        )

        return result, system_answer

    def _evaluate_qa_legacy(self, system, qa, query_result):
        """Legacy mode: text matching for MC + judge for abstractive."""
        selected_idx = None
        if qa.answer_type.value == "multi_choice" and qa.choices:
            selected_idx = self.extractor.extract_choice_index(
                query_result.answer, qa.choices
            )

        system_answer = SystemAnswer(
            qa_id=qa.qa_id,
            system_name=system.name,
            raw_response=query_result.answer[:500],
            extracted_answer=self.extractor.extract_text_answer(query_result.answer),
            selected_choice_index=selected_idx,
            confidence=query_result.confidence,
            retrieval_context=query_result.retrieved_context[:500],
            latency_ms=query_result.latency_ms,
        )

        result = score_qa_pair(
            qa_id=qa.qa_id,
            system_name=system.name,
            system_answer=system_answer,
            correct_choice_index=qa.correct_choice_index,
            ground_truth_answer=qa.answer_zh,
            answer_type=qa.answer_type,
        )

        return result, system_answer

    def run_all(self) -> List[EvaluationRun]:
        """Run evaluation for all systems."""
        results = []
        for system in self.systems:
            try:
                run = self.run(system)
                results.append(run)
            except Exception as e:
                logger.error(f"Evaluation failed for {system.name}: {e}")
                continue
        return results


def run_evaluation(
    scenarios_path: Path,
    qa_path: Path,
    systems: List[MemorySystem],
    output_dir: Path,
    llm_judge_config: Optional[dict] = None,
    answer_generator_config: Optional[dict] = None,
    eval_mode: str = "unified",
) -> List[EvaluationRun]:
    """Convenience function to run full evaluation pipeline."""
    # Load data
    scenarios_data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios = [Scenario(**s) for s in scenarios_data]

    qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
    qa_dataset = QADataset(**qa_data)

    runner = EvaluationRunner(
        systems, scenarios, qa_dataset,
        llm_judge_config=llm_judge_config,
        answer_generator_config=answer_generator_config,
        eval_mode=eval_mode,
    )
    results = runner.run_all()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in results:
        out_path = output_dir / f"{run.run_id}.json"
        out_path.write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(f"Saved results to {out_path}")

    return results
