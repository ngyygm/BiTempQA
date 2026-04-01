"""Metrics computation for BiTempQA evaluation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.schemas import (
    AnswerType,
    QAResult,
    Scenario,
    ScenarioResult,
    SystemAnswer,
)

logger = logging.getLogger(__name__)


def compute_exact_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer exactly matches ground truth."""
    return predicted.strip() == ground_truth.strip()


def compute_f1(predicted: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = set(predicted.strip())
    gt_tokens = set(ground_truth.strip())

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def score_qa_pair(
    qa_id: str,
    system_name: str,
    system_answer: SystemAnswer,
    correct_choice_index: Optional[int],
    ground_truth_answer: str,
    answer_type: AnswerType,
    llm_judge_result: Optional[dict] = None,
) -> QAResult:
    """Score a single QA pair against ground truth."""

    if answer_type == AnswerType.MULTI_CHOICE:
        if llm_judge_result is not None:
            # LLM Judge mode: use judge's verdict
            is_correct = llm_judge_result.get("is_correct", False)
        else:
            # Direct matching mode
            is_correct = system_answer.selected_choice_index == correct_choice_index
        extracted = ""
        if system_answer.selected_choice_index is not None and correct_choice_index is not None:
            extracted = f"choice_{system_answer.selected_choice_index}"
        return QAResult(
            qa_id=qa_id,
            system_name=system_name,
            is_correct=is_correct,
            exact_match=is_correct,
            f1_score=1.0 if is_correct else 0.0,
            answer=system_answer,
        )

    elif answer_type == AnswerType.BOOLEAN:
        if llm_judge_result is not None:
            is_correct = llm_judge_result.get("is_correct", False)
            return QAResult(
                qa_id=qa_id,
                system_name=system_name,
                is_correct=is_correct,
                exact_match=is_correct,
                f1_score=1.0 if is_correct else 0.0,
                answer=system_answer,
            )
        from src.evaluation.answer_extractor import AnswerExtractor
        extractor = AnswerExtractor()
        predicted_bool = extractor.extract_boolean(system_answer.raw_response)
        gt_bool = ground_truth_answer in ("是", "对", "正确", "True", "true", "yes", "Yes")
        is_correct = predicted_bool == gt_bool if predicted_bool is not None else False
        return QAResult(
            qa_id=qa_id,
            system_name=system_name,
            is_correct=is_correct,
            exact_match=is_correct,
            f1_score=1.0 if is_correct else 0.0,
            answer=system_answer,
        )

    elif llm_judge_result is not None:
        # LLM Judge scoring for abstractive/extractive/exact
        is_correct = llm_judge_result.get("is_correct", False)
        confidence = llm_judge_result.get("confidence", "low")
        # Use confidence as f1 proxy: high=1.0, medium=0.5, low=0.0
        f1 = {"高": 1.0, "中": 0.5, "low": 0.0, "高": 1.0}.get(confidence, 0.0)
        if is_correct:
            f1 = max(f1, 1.0)
        return QAResult(
            qa_id=qa_id,
            system_name=system_name,
            is_correct=is_correct,
            exact_match=is_correct,
            f1_score=f1,
            answer=system_answer,
            error_category=f"judge_{confidence}",
        )

    else:  # exact, extractive, abstractive without LLM judge — heuristic only
        extracted = system_answer.extracted_answer
        em = compute_exact_match(extracted, ground_truth_answer)
        f1 = compute_f1(extracted, ground_truth_answer)
        return QAResult(
            qa_id=qa_id,
            system_name=system_name,
            is_correct=em,
            exact_match=em,
            f1_score=f1,
            answer=system_answer,
        )


def compute_scenario_result(
    scenario: Scenario,
    qa_pairs: list,
    qa_results: List[QAResult],
    system_name: str,
) -> ScenarioResult:
    """Compute aggregated metrics for a scenario."""
    sr = ScenarioResult(scenario_id=scenario.scenario_id, system_name=system_name)

    scenario_type = scenario.scenario_type.value if scenario.scenario_type else "unknown"
    for qa, result in zip(qa_pairs, qa_results):
        result._difficulty = qa.difficulty.value
        result._question_type = qa.question_type.value
        result._scenario_type = scenario_type
        result._requires_temporal = qa.requires_event_time_reasoning or qa.requires_record_time_reasoning
        result._requires_version = qa.requires_version_tracking
        sr.qa_results.append(result)

    sr.compute_aggregates(scenario)
    return sr
