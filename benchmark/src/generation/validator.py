"""Validator for generated scenarios and QA pairs.

Provides LLM-assisted validation with configurable checks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from src.schemas import QAPair, Scenario

logger = logging.getLogger(__name__)


class ScenarioValidator:
    """Validates a single Scenario for structural and logical correctness."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, scenario: Scenario) -> bool:
        self.errors.clear()
        self.warnings.clear()

        self._check_basic_structure(scenario)
        self._check_time_consistency(scenario)
        self._check_ground_truth(scenario)
        self._check_world_states(scenario)

        is_valid = len(self.errors) == 0
        if not is_valid:
            logger.warning(f"{scenario.scenario_id}: {len(self.errors)} errors found")
            for e in self.errors:
                logger.warning(f"  ERROR: {e}")
        if self.warnings:
            for w in self.warnings:
                logger.info(f"  WARN: {w}")

        return is_valid

    def _check_basic_structure(self, s: Scenario) -> None:
        if not s.scenario_id:
            self.errors.append("Missing scenario_id")
        if not s.memory_writes:
            self.errors.append("No memory_writes")
        if len(s.memory_writes) < 2:
            self.warnings.append("Less than 2 memory_writes — may be too simple")

        write_ids = set()
        for w in s.memory_writes:
            if w.write_id in write_ids:
                self.errors.append(f"Duplicate write_id: {w.write_id}")
            write_ids.add(w.write_id)
            if not w.text:
                self.errors.append(f"Empty text in {w.write_id}")
            if not w.event_time:
                self.errors.append(f"Missing event_time in {w.write_id}")
            if not w.record_time:
                self.errors.append(f"Missing record_time in {w.write_id}")

    def _check_time_consistency(self, s: Scenario) -> None:
        # Check record_time is monotonically increasing
        prev_rt = ""
        for w in s.memory_writes:
            if prev_rt and w.record_time < prev_rt:
                self.errors.append(
                    f"record_time not monotonic: {w.write_id} ({w.record_time}) < previous ({prev_rt})"
                )
            prev_rt = w.record_time

        # Check for at least one write with event_time != record_time
        has_gap = any(w.event_time != w.record_time for w in s.memory_writes)
        if not has_gap:
            self.warnings.append("No event_time/record_time gap found")

    def _check_ground_truth(self, s: Scenario) -> None:
        if not s.entity_ground_truth and not s.relation_ground_truth:
            self.warnings.append("No ground truth entities or relations")

        for entity_name, snapshots in s.entity_ground_truth.items():
            for snap in snapshots:
                if not snap.entity_id:
                    self.errors.append(f"Missing entity_id in ground truth for {entity_name}")
                if not snap.valid_from:
                    self.errors.append(f"Missing valid_from in ground truth for {entity_name}")
                if snap.source_write_id not in {w.write_id for w in s.memory_writes}:
                    self.errors.append(
                        f"Invalid source_write_id {snap.source_write_id} for {entity_name}"
                    )

    def _check_world_states(self, s: Scenario) -> None:
        if not s.world_states:
            self.warnings.append("No world_states provided")

        for ws in s.world_states:
            if not ws.as_of_record_time:
                self.errors.append("WorldState missing as_of_record_time")
            if ws.known_facts and len(ws.known_facts) < len(s.memory_writes):
                # Only warn if world state is supposed to have all writes up to that time
                pass


class QAPairValidator:
    """Validates QA pairs for structural correctness."""

    def __init__(self) -> None:
        self.errors: List[str] = []

    def validate(self, qa: QAPair, scenario: Optional[Scenario] = None) -> bool:
        self.errors.clear()

        if not qa.qa_id:
            self.errors.append("Missing qa_id")
        if not qa.question_zh:
            self.errors.append("Missing question_zh")
        if not qa.answer_zh:
            self.errors.append("Missing answer_zh")

        if qa.answer_type.value == "multi_choice":
            if not qa.choices or len(qa.choices) < 2:
                self.errors.append("Multi-choice question needs at least 2 choices")
            if qa.correct_choice_index is None:
                self.errors.append("Multi-choice question needs correct_choice_index")
            elif qa.correct_choice_index >= len(qa.choices or []):
                self.errors.append("correct_choice_index out of range")

        if not qa.reasoning_chain:
            self.errors.append("Missing reasoning_chain")

        return len(self.errors) == 0


def validate_scenarios_file(scenarios_path: Path) -> dict:
    validator = ScenarioValidator()
    results = {"total": 0, "valid": 0, "invalid": 0, "errors_by_id": {}}

    data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    for s_data in data:
        scenario = Scenario(**s_data)
        results["total"] += 1
        if validator.validate(scenario):
            results["valid"] += 1
        else:
            results["invalid"] += 1
            results["errors_by_id"][scenario.scenario_id] = list(validator.errors)

    logger.info(
        f"Validation: {results['valid']}/{results['total']} valid, "
        f"{results['invalid']} invalid"
    )
    return results


def validate_qa_file(qa_path: Path) -> dict:
    validator = QAPairValidator()
    results = {"total": 0, "valid": 0, "invalid": 0, "errors_by_id": {}}

    data = json.loads(qa_path.read_text(encoding="utf-8"))
    pairs = data if isinstance(data, list) else data.get("qa_pairs", [])
    for p_data in pairs:
        qa = QAPair(**p_data)
        results["total"] += 1
        if validator.validate(qa):
            results["valid"] += 1
        else:
            results["invalid"] += 1
            results["errors_by_id"][qa.qa_id] = list(validator.errors)

    logger.info(
        f"QA Validation: {results['valid']}/{results['total']} valid, "
        f"{results['invalid']} invalid"
    )
    return results
