"""
BiTempQA + EvolBench: Data schemas for dual-timestamp agent memory benchmark.

Core data models:
- MemoryWrite: A single remember call with dual timestamps
- Scenario: An agent interaction trace with knowledge evolution
- QAPair: A question-answer pair for evaluation
- EvaluationResult: System evaluation output
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# Enums
# ============================================================

class ScenarioType(str, Enum):
    ENTITY_ATTRIBUTE_EVOLUTION = "entity_attribute_evolution"
    RELATIONSHIP_EVOLUTION = "relationship_evolution"
    CONTRADICTORY_INFORMATION = "contradictory_information"
    LATE_ARRIVING_FACTS = "late_arriving_facts"
    FUTURE_DATED_INFORMATION = "future_dated_information"
    ENTITY_IDENTITY_RESOLUTION = "entity_identity_resolution"
    KNOWLEDGE_RETRACTION = "knowledge_retraction"
    MULTI_SOURCE_INFORMATION = "multi_source_information"
    GRADUAL_ACCUMULATION = "gradual_accumulation"
    TEMPORAL_AMBIGUITY = "temporal_ambiguity"


class DifficultyLevel(str, Enum):
    LEVEL_1_EASY = "level_1"
    LEVEL_2_MEDIUM = "level_2"
    LEVEL_3_HARD = "level_3"


class QuestionType(str, Enum):
    POINT_IN_TIME = "point_in_time"               # Level 1
    FIRST_RECORDED = "first_recorded"              # Level 1
    TEMPORAL_ORDERING = "temporal_ordering"        # Level 1
    PERIOD_QUERY = "period_query"                  # Level 2
    CHANGE_DETECTION = "change_detection"          # Level 2
    MULTI_HOP_TEMPORAL = "multi_hop_temporal"      # Level 2
    COUNTERFACTUAL = "counterfactual"              # Level 3
    VERSION_CONFLICT = "version_conflict"          # Level 3
    COMPLEX_TEMPORAL = "complex_temporal"          # Level 3


class AnswerType(str, Enum):
    EXACT = "exact"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BOOLEAN = "boolean"
    MULTI_CHOICE = "multi_choice"
    RANKING = "ranking"


# ============================================================
# Scenario / Trace models
# ============================================================

class MemoryWrite(BaseModel):
    """A single remember call in the agent trace."""

    write_id: str = Field(description="Unique ID within scenario, e.g. w1, w2")
    text: str = Field(description="Natural language text to remember")
    source_name: str = Field(default="scenario_trace", description="Provenance tag")
    event_time: str = Field(description="ISO 8601. When the event occurred (valid time).")
    record_time: str = Field(description="ISO 8601. When info was recorded (transaction time).")


class EntitySnapshot(BaseModel):
    """Ground truth entity state at a specific time."""

    entity_name: str
    entity_id: str = Field(description="Logical ID consistent across versions")
    attributes: Dict[str, str] = Field(default_factory=dict)
    valid_from: str = Field(description="ISO 8601")
    valid_until: Optional[str] = Field(default=None, description="ISO 8601, None=current")
    source_write_id: str


class RelationSnapshot(BaseModel):
    """Ground truth relation state at a specific time."""

    relation_id: str
    entity1_name: str
    entity2_name: str
    description: str = Field(description="Natural language relation description")
    valid_from: str
    valid_until: Optional[str] = None
    source_write_id: str


class WorldState(BaseModel):
    """Complete world state snapshot at a given record_time."""

    as_of_record_time: str
    entities: List[EntitySnapshot] = Field(default_factory=list)
    relations: List[RelationSnapshot] = Field(default_factory=list)
    known_facts: List[str] = Field(default_factory=list)


class Scenario(BaseModel):
    """A complete agent interaction scenario with knowledge evolution."""

    scenario_id: str = Field(description="e.g. S01_001")
    scenario_type: ScenarioType
    title_zh: str
    title_en: str = ""
    description_zh: str
    description_en: str = ""
    domain: str = Field(description="corporate, academic, social, fictional, historical, scientific")
    language: Literal["zh", "en"] = "zh"

    memory_writes: List[MemoryWrite] = Field(description="Ordered by record_time")
    world_states: List[WorldState] = Field(default_factory=list)

    entity_ground_truth: Dict[str, List[EntitySnapshot]] = Field(
        default_factory=dict, description="entity_name -> version list"
    )
    relation_ground_truth: Dict[str, List[RelationSnapshot]] = Field(
        default_factory=dict, description="relation_id -> version list"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# QA Pair models
# ============================================================

class QAPair(BaseModel):
    """A single question-answer pair for evaluation."""

    qa_id: str = Field(description="e.g. S01_001_L1_003")
    scenario_id: str
    difficulty: DifficultyLevel
    question_type: QuestionType

    # Question
    question_zh: str
    question_en: Optional[str] = None

    # Ground truth answer
    answer_zh: str
    answer_en: Optional[str] = None
    answer_type: AnswerType = AnswerType.MULTI_CHOICE

    # For multi-choice
    choices: Optional[List[str]] = None
    correct_choice_index: Optional[int] = None

    # For ranking
    ranking_order: Optional[List[str]] = None

    # Temporal context
    query_event_time: Optional[str] = None
    query_record_time: Optional[str] = None
    relevant_time_range: Optional[Dict[str, Optional[str]]] = None  # {"start": ISO, "end": ISO}

    # Reasoning annotation
    reasoning_chain: List[str] = Field(default_factory=list)
    requires_event_time_reasoning: bool = False
    requires_record_time_reasoning: bool = False
    requires_version_tracking: bool = False
    requires_knowledge_retraction: bool = False

    # Metadata
    source_write_ids: List[str] = Field(default_factory=list)
    generation_method: Literal["llm", "human", "llm_human"] = "llm"
    validation_status: Literal["unvalidated", "validated", "rejected"] = "unvalidated"
    validator_notes: Optional[str] = None


class QADataset(BaseModel):
    """A collection of QA pairs forming a dataset split."""

    dataset_id: str
    name: str
    language: Literal["zh", "en", "both"] = "zh"
    split: Literal["train", "dev", "test"]
    qa_pairs: List[QAPair]
    total_count: int = 0
    difficulty_counts: Dict[str, int] = Field(default_factory=dict)
    scenario_type_counts: Dict[str, int] = Field(default_factory=dict)

    def compute_stats(self) -> None:
        """Recompute statistics from qa_pairs."""
        self.total_count = len(self.qa_pairs)
        self.difficulty_counts = {}
        self.scenario_type_counts = {}
        for qa in self.qa_pairs:
            d = qa.difficulty.value
            self.difficulty_counts[d] = self.difficulty_counts.get(d, 0) + 1


# ============================================================
# Evaluation result models
# ============================================================

class SystemAnswer(BaseModel):
    """A system's answer to a single QA pair."""

    qa_id: str
    system_name: str
    raw_response: str = ""
    extracted_answer: str = ""
    selected_choice_index: Optional[int] = None
    confidence: Optional[float] = None
    retrieval_context: str = ""
    latency_ms: float = 0.0


class QAResult(BaseModel):
    """Evaluation result for a single QA pair."""

    qa_id: str
    system_name: str
    is_correct: bool = False
    exact_match: bool = False
    f1_score: float = 0.0
    answer: Optional[SystemAnswer] = None
    error_category: Optional[str] = None  # wrong_entity, wrong_time, etc.


class ScenarioResult(BaseModel):
    """Aggregated results for a single scenario."""

    scenario_id: str
    system_name: str
    qa_results: List[QAResult] = Field(default_factory=list)
    accuracy_by_level: Dict[str, float] = Field(default_factory=dict)
    accuracy_by_question_type: Dict[str, float] = Field(default_factory=dict)
    temporal_reasoning_accuracy: float = 0.0
    version_recall: float = 0.0

    def compute_aggregates(self, scenario: Optional[Scenario] = None) -> None:
        """Compute aggregated metrics from qa_results."""
        if not self.qa_results:
            return

        # By level
        level_correct: Dict[str, List[bool]] = {}
        type_correct: Dict[str, List[bool]] = {}
        temporal_flags: List[bool] = []
        version_flags: List[bool] = []

        for r in self.qa_results:
            # We need the original QA pair to know its difficulty/type
            # Store as metadata on QAResult if available
            level = getattr(r, '_difficulty', 'unknown')
            qtype = getattr(r, '_question_type', 'unknown')

            level_correct.setdefault(level, []).append(r.is_correct)
            type_correct.setdefault(qtype, []).append(r.is_correct)

            is_temporal = getattr(r, '_requires_temporal', False)
            is_version = getattr(r, '_requires_version', False)
            if is_temporal:
                temporal_flags.append(r.is_correct)
            if is_version:
                version_flags.append(r.is_correct)

        self.accuracy_by_level = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in level_correct.items()
        }
        self.accuracy_by_question_type = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in type_correct.items()
        }
        self.temporal_reasoning_accuracy = (
            sum(temporal_flags) / len(temporal_flags) if temporal_flags else 0.0
        )
        self.version_recall = (
            sum(version_flags) / len(version_flags) if version_flags else 0.0
        )


class EvaluationRun(BaseModel):
    """Complete evaluation run for one system."""

    run_id: str
    timestamp: str
    system_name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    scenario_results: List[ScenarioResult] = Field(default_factory=list)
    overall_accuracy: float = 0.0
    overall_f1: float = 0.0
    temporal_reasoning_accuracy: float = 0.0
    version_recall: float = 0.0
    accuracy_by_level: Dict[str, float] = Field(default_factory=dict)
    accuracy_by_scenario_type: Dict[str, float] = Field(default_factory=dict)
    latency_stats: Dict[str, float] = Field(default_factory=dict)

    def compute_aggregates(self) -> None:
        """Compute overall metrics from scenario results."""
        all_results: List[QAResult] = []
        for sr in self.scenario_results:
            all_results.extend(sr.qa_results)

        if not all_results:
            return

        self.overall_accuracy = sum(r.is_correct for r in all_results) / len(all_results)
        self.overall_f1 = sum(r.f1_score for r in all_results) / len(all_results)

        level_results: Dict[str, List[QAResult]] = {}
        type_results: Dict[str, List[QAResult]] = {}
        temporal: List[bool] = []
        version: List[bool] = []
        latencies: List[float] = []

        for r in all_results:
            level = getattr(r, '_difficulty', 'unknown')
            stype = getattr(r, '_scenario_type', 'unknown')
            level_results.setdefault(level, []).append(r)
            type_results.setdefault(stype, []).append(r)

            if getattr(r, '_requires_temporal', False):
                temporal.append(r.is_correct)
            if getattr(r, '_requires_version', False):
                version.append(r.is_correct)
            if r.answer:
                latencies.append(r.answer.latency_ms)

        self.accuracy_by_level = {
            k: sum(r.is_correct for r in v) / len(v)
            for k, v in level_results.items() if v
        }
        self.accuracy_by_scenario_type = {
            k: sum(r.is_correct for r in v) / len(v)
            for k, v in type_results.items() if v
        }
        self.temporal_reasoning_accuracy = (
            sum(temporal) / len(temporal) if temporal else 0.0
        )
        self.version_recall = sum(version) / len(version) if version else 0.0

        if latencies:
            latencies.sort()
            self.latency_stats = {
                "mean": sum(latencies) / len(latencies),
                "median": latencies[len(latencies) // 2],
                "p95": latencies[int(len(latencies) * 0.95)] if len(latencies) > 20 else latencies[-1],
                "max": latencies[-1],
                "count": len(latencies),
            }


# ============================================================
# Dataset metadata
# ============================================================

class DatasetMetadata(BaseModel):
    """Dataset card with statistics."""

    name: str = "BiTempQA + EvolBench"
    version: str = "1.0"
    description_zh: str = "双时间轴Agent记忆基准：评估记忆系统的事件时间与记录时间推理能力"
    description_en: str = "Bitemporal Agent Memory Benchmark: Evaluating event-time and record-time reasoning"
    total_scenarios: int = 0
    total_qa_pairs: int = 0
    languages: List[str] = ["zh", "en"]
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    scenario_type_distribution: Dict[str, int] = Field(default_factory=dict)
    answer_type_distribution: Dict[str, int] = Field(default_factory=dict)
    systems_evaluated: List[str] = Field(default_factory=list)
    created_at: str = ""
    license: str = "CC BY-NC-SA 4.0"
