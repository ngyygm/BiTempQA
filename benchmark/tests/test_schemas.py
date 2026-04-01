"""Unit tests for BiTempQA + EvolBench data schemas."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class TestEnums:
    def test_scenario_type_values(self):
        from src.schemas import ScenarioType
        assert ScenarioType.ENTITY_ATTRIBUTE_EVOLUTION == "entity_attribute_evolution"
        assert ScenarioType.LATE_ARRIVING_FACTS == "late_arriving_facts"
        assert len(ScenarioType) == 10

    def test_difficulty_level_values(self):
        from src.schemas import DifficultyLevel
        assert DifficultyLevel.LEVEL_1_EASY == "level_1"
        assert len(DifficultyLevel) == 3

    def test_question_type_values(self):
        from src.schemas import QuestionType
        assert QuestionType.POINT_IN_TIME == "point_in_time"
        assert QuestionType.COUNTERFACTUAL == "counterfactual"
        assert len(QuestionType) == 9

    def test_answer_type_values(self):
        from src.schemas import AnswerType
        assert AnswerType.MULTI_CHOICE == "multi_choice"
        assert len(AnswerType) == 6


class TestMemoryWrite:
    def test_from_dict(self):
        from src.schemas import MemoryWrite
        w = MemoryWrite(
            write_id="w1",
            text="张三在字节跳动工作。",
            event_time="2024-01-15T09:00:00",
            record_time="2024-01-15T10:30:00",
        )
        assert w.write_id == "w1"
        assert w.source_name == "scenario_trace"

    def test_missing_required_fields(self):
        from src.schemas import MemoryWrite
        with pytest.raises(Exception):
            MemoryWrite(write_id="w1")  # missing text, event_time, record_time


class TestEntitySnapshot:
    def test_with_optional_fields(self):
        from src.schemas import EntitySnapshot
        e = EntitySnapshot(
            entity_name="张三",
            entity_id="e1",
            valid_from="2024-01-01",
            source_write_id="w1",
        )
        assert e.attributes == {}
        assert e.valid_until is None

    def test_full_fields(self):
        from src.schemas import EntitySnapshot
        e = EntitySnapshot(
            entity_name="张三",
            entity_id="e1",
            attributes={"职业": "工程师"},
            valid_from="2024-01-01",
            valid_until="2024-06-01",
            source_write_id="w1",
        )
        assert e.attributes["职业"] == "工程师"
        assert e.valid_until == "2024-06-01"


class TestScenario:
    def test_from_fixture(self):
        from src.schemas import Scenario
        data = json.loads((FIXTURES / "sample_scenario.json").read_text())
        s = Scenario(**data)
        assert s.scenario_id == "S01_001"
        assert s.scenario_type.value == "entity_attribute_evolution"
        assert len(s.memory_writes) == 3
        assert s.memory_writes[0].event_time == "2024-01-15T09:00:00"
        assert len(s.world_states) == 3
        assert "张三" in s.entity_ground_truth
        assert len(s.entity_ground_truth["张三"]) == 3

    def test_late_arriving_scenario(self):
        from src.schemas import Scenario
        data = json.loads((FIXTURES / "sample_late_arriving.json").read_text())
        s = Scenario(**data)
        assert s.scenario_type.value == "late_arriving_facts"
        assert s.memory_writes[0].event_time == "2024-03-01T09:00:00"
        assert s.memory_writes[0].record_time == "2024-04-10T16:00:00"
        # event_time << record_time — late arriving
        assert s.memory_writes[0].event_time < s.memory_writes[0].record_time

    def test_roundtrip_json(self):
        from src.schemas import Scenario
        data = json.loads((FIXTURES / "sample_scenario.json").read_text())
        s = Scenario(**data)
        json_str = s.model_dump_json()
        s2 = Scenario.model_validate_json(json_str)
        assert s2.scenario_id == s.scenario_id
        assert len(s2.memory_writes) == len(s.memory_writes)


class TestQAPair:
    def test_multi_choice(self):
        from src.schemas import QAPair
        qa = QAPair(
            qa_id="S01_001_L1_001",
            scenario_id="S01_001",
            difficulty="level_1",
            question_type="point_in_time",
            question_zh="张三在哪里工作？",
            answer_zh="字节跳动",
            answer_type="multi_choice",
            choices=["字节跳动", "阿里巴巴", "腾讯", "未记录"],
            correct_choice_index=0,
        )
        assert qa.choices[0] == "字节跳动"
        assert qa.correct_choice_index == 0

    def test_abstractive(self):
        from src.schemas import QAPair
        qa = QAPair(
            qa_id="S01_001_L2_001",
            scenario_id="S01_001",
            difficulty="level_2",
            question_type="period_query",
            question_zh="张三的工作经历是什么？",
            answer_zh="先在字节跳动，后去阿里巴巴",
            answer_type="abstractive",
        )
        assert qa.choices is None
        assert qa.correct_choice_index is None

    def test_reasoning_flags(self):
        from src.schemas import QAPair
        qa = QAPair(
            qa_id="test",
            scenario_id="S01",
            difficulty="level_3",
            question_type="counterfactual",
            question_zh="反事实问题",
            answer_zh="答案",
            requires_event_time_reasoning=True,
            requires_record_time_reasoning=True,
            requires_version_tracking=True,
        )
        assert qa.requires_event_time_reasoning is True
        assert qa.requires_record_time_reasoning is True


class TestQADataset:
    def test_from_fixture(self):
        from src.schemas import QADataset
        data = json.loads((FIXTURES / "sample_qa_pairs.json").read_text())
        ds = QADataset(**data)
        assert ds.dataset_id == "bitpqa_test_zh_v1"
        assert ds.split == "test"
        assert len(ds.qa_pairs) == 4

    def test_compute_stats(self):
        from src.schemas import QADataset
        data = json.loads((FIXTURES / "sample_qa_pairs.json").read_text())
        ds = QADataset(**data)
        ds.compute_stats()
        assert ds.total_count == 4
        assert ds.difficulty_counts["level_1"] == 2
        assert ds.difficulty_counts["level_2"] == 1
        assert ds.difficulty_counts["level_3"] == 1


class TestSystemAnswer:
    def test_fields(self):
        from src.schemas import SystemAnswer
        a = SystemAnswer(
            qa_id="q1",
            system_name="tmg",
            raw_response="张三在字节跳动",
            selected_choice_index=0,
            latency_ms=150.5,
        )
        assert a.selected_choice_index == 0
        assert a.latency_ms == 150.5


class TestQAResult:
    def test_correct(self):
        from src.schemas import QAResult
        r = QAResult(qa_id="q1", system_name="tmg", is_correct=True, f1_score=1.0)
        assert r.is_correct is True
        assert r.error_category is None


class TestScenarioResult:
    def test_compute_aggregates_empty(self):
        from src.schemas import ScenarioResult
        sr = ScenarioResult(scenario_id="S01", system_name="tmg")
        sr.compute_aggregates()
        assert sr.accuracy_by_level == {}

    def test_compute_aggregates_with_results(self):
        from src.schemas import ScenarioResult, QAResult
        sr = ScenarioResult(scenario_id="S01", system_name="tmg")
        r1 = QAResult(qa_id="q1", system_name="tmg", is_correct=True, f1_score=1.0)
        r1._difficulty = "level_1"
        r1._requires_temporal = True
        r2 = QAResult(qa_id="q2", system_name="tmg", is_correct=False, f1_score=0.0)
        r2._difficulty = "level_1"
        r2._requires_temporal = True
        sr.qa_results = [r1, r2]
        sr.compute_aggregates()
        assert sr.accuracy_by_level["level_1"] == 0.5
        assert sr.temporal_reasoning_accuracy == 0.5


class TestEvaluationRun:
    def test_compute_aggregates_from_fixture(self):
        from src.schemas import EvaluationRun
        data = json.loads((FIXTURES / "sample_evaluation.json").read_text())
        ev = EvaluationRun(**data)
        ev.compute_aggregates()
        # 4 QA results: 3 correct, 1 incorrect
        assert ev.overall_accuracy == 0.75
        assert ev.overall_f1 == pytest.approx(0.9)
        assert len(ev.latency_stats) > 0
        assert ev.latency_stats["mean"] == pytest.approx(275.275, rel=0.01)

    def test_empty_run(self):
        from src.schemas import EvaluationRun
        ev = EvaluationRun(run_id="test", timestamp="2024-01-01", system_name="test")
        ev.compute_aggregates()
        assert ev.overall_accuracy == 0.0

    def test_roundtrip_json(self):
        from src.schemas import EvaluationRun
        data = json.loads((FIXTURES / "sample_evaluation.json").read_text())
        ev = EvaluationRun(**data)
        json_str = ev.model_dump_json()
        ev2 = EvaluationRun.model_validate_json(json_str)
        assert ev2.run_id == ev.run_id
        assert len(ev2.scenario_results) == len(ev.scenario_results)


class TestDatasetMetadata:
    def test_defaults(self):
        from src.schemas import DatasetMetadata
        m = DatasetMetadata()
        assert m.name == "BiTempQA + EvolBench"
        assert m.version == "1.0"
        assert "zh" in m.languages
        assert "en" in m.languages

    def test_custom(self):
        from src.schemas import DatasetMetadata
        m = DatasetMetadata(total_scenarios=122, total_qa_pairs=1586)
        assert m.total_scenarios == 122
        assert m.total_qa_pairs == 1586
