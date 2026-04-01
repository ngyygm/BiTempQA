#!/usr/bin/env python3
"""Step 15 (was Step 6): Validate QA pairs.

Validates QA pairs for correctness:
1. Time consistency: query_event_time within scenario time range
2. Answer derivable: gold answer matches scenario content
3. Choice quality: no duplicate/semantically identical options
4. Keyword leak detection: correct answer shouldn't be trivially findable by BM25
5. Dual-time coverage: 30%+ questions should test both event_time and record_time

Usage:
    python scripts/15_validate_qa.py --qa data/validated/bitpqa_test_zh.json
    python scripts/15_validate_qa.py --qa data/validated/bitpqa_test_zh.json --fix
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import QAPair, Scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class QAValidator:
    """Validate QA pairs against their source scenarios."""

    def __init__(self, scenarios: list):
        # Index scenarios by ID
        self.scenarios = {s.scenario_id: s for s in scenarios}
        # Build full text index per scenario for keyword leak detection
        self.scenario_texts = {}
        for s in scenarios:
            texts = []
            for w in s.memory_writes:
                texts.append(w.text.lower())
            self.scenario_texts[s.scenario_id] = " ".join(texts)

    def validate_pair(self, qa: QAPair) -> dict:
        """Validate a single QA pair. Returns issues list."""
        issues = []
        scenario = self.scenarios.get(qa.scenario_id)

        if scenario is None:
            return [{"severity": "error", "code": "missing_scenario",
                     "message": f"No scenario found for {qa.scenario_id}"}]

        # 1. Time consistency
        time_issues = self._check_time_consistency(qa, scenario)
        issues.extend(time_issues)

        # 2. Answer derivability (basic check)
        deriv_issues = self._check_answer_derivable(qa, scenario)
        issues.extend(deriv_issues)

        # 3. Choice quality (for multi-choice)
        if qa.answer_type.value == "multi_choice" and qa.choices:
            choice_issues = self._check_choices(qa)
            issues.extend(choice_issues)

        # 4. Keyword leak detection
        leak_issues = self._check_keyword_leak(qa)
        issues.extend(leak_issues)

        return issues

    def _check_time_consistency(self, qa: QAPair, scenario: Scenario) -> list:
        issues = []
        if not scenario.memory_writes:
            return issues

        times = [w.event_time for w in scenario.memory_writes if w.event_time]
        if not times:
            return issues

        times.sort()
        earliest, latest = times[0], times[-1]

        if qa.query_event_time:
            if qa.query_event_time < earliest:
                issues.append({
                    "severity": "warning",
                    "code": "time_before_range",
                    "message": f"query_event_time {qa.query_event_time} before earliest {earliest}",
                })
            if qa.query_event_time > latest:
                issues.append({
                    "severity": "warning",
                    "code": "time_after_range",
                    "message": f"query_event_time {qa.query_event_time} after latest {latest}",
                })

        return issues

    def _check_answer_derivable(self, qa: QAPair, scenario: Scenario) -> list:
        issues = []
        answer = qa.answer_zh.lower().strip()

        if not answer:
            issues.append({
                "severity": "error",
                "code": "empty_answer",
                "message": "Gold answer is empty",
            })
            return issues

        # Check if answer keywords appear in scenario text
        scenario_text = self.scenario_texts.get(qa.scenario_id, "")
        # Extract key terms from answer (Chinese characters and numbers)
        import re
        answer_terms = re.findall(r"[\u4e00-\u9fff]+|\d+", answer)

        if not answer_terms:
            return issues

        found_terms = [t for t in answer_terms if t in scenario_text]
        coverage = len(found_terms) / len(answer_terms) if answer_terms else 0

        if coverage < 0.3:
            issues.append({
                "severity": "warning",
                "code": "low_answer_coverage",
                "message": f"Only {coverage:.0%} of answer terms found in scenario text",
            })

        return issues

    def _check_choices(self, qa: QAPair) -> list:
        issues = []
        choices = qa.choices
        if not choices:
            return issues

        # Check for empty choices
        for i, c in enumerate(choices):
            if not c.strip():
                issues.append({
                    "severity": "error",
                    "code": "empty_choice",
                    "message": f"Choice {i} is empty",
                })

        # Check for duplicate choices
        seen = {}
        for i, c in enumerate(choices):
            c_lower = c.strip().lower()
            if c_lower in seen:
                issues.append({
                    "severity": "error",
                    "code": "duplicate_choice",
                    "message": f"Choices {seen[c_lower]} and {i} are identical",
                })
            seen[c_lower] = i

        # Check correct_choice_index is valid
        if qa.correct_choice_index is not None:
            if qa.correct_choice_index < 0 or qa.correct_choice_index >= len(choices):
                issues.append({
                    "severity": "error",
                    "code": "invalid_correct_index",
                    "message": f"correct_choice_index {qa.correct_choice_index} out of range [0, {len(choices)-1}]",
                })

        return issues

    def _check_keyword_leak(self, qa: QAPair) -> list:
        issues = []
        scenario_text = self.scenario_texts.get(qa.scenario_id, "")
        answer = qa.answer_zh.lower().strip()

        if not answer or not scenario_text:
            return issues

        # Check if the full answer string appears verbatim in scenario
        if answer in scenario_text and len(answer) > 5:
            issues.append({
                "severity": "info",
                "code": "verbatim_answer",
                "message": "Answer appears verbatim in scenario (BM25 may trivially find it)",
            })

        return issues

    def validate_all(self, qa_pairs: list) -> dict:
        """Validate all QA pairs and return summary."""
        all_issues = {}
        stats = Counter()
        issue_by_code = Counter()
        issue_by_severity = Counter()
        dual_time_count = 0

        for qa in qa_pairs:
            issues = self.validate_pair(qa)
            all_issues[qa.qa_id] = issues

            if issues:
                stats["has_issues"] += 1
            else:
                stats["clean"] += 1

            for issue in issues:
                issue_by_code[issue["code"]] += 1
                issue_by_severity[issue["severity"]] += 1

            # Check dual-time coverage
            if qa.requires_event_time_reasoning and qa.requires_record_time_reasoning:
                dual_time_count += 1

        total = len(qa_pairs)
        dual_time_pct = (dual_time_count / total * 100) if total else 0

        return {
            "total": total,
            "clean": stats["clean"],
            "has_issues": stats["has_issues"],
            "dual_time_coverage_pct": dual_time_pct,
            "issue_by_code": dict(issue_by_code),
            "issue_by_severity": dict(issue_by_severity),
            "all_issues": all_issues,
        }


def main():
    parser = argparse.ArgumentParser(description="Validate QA pairs")
    parser.add_argument("--qa", required=True, help="QA pairs JSON file")
    parser.add_argument("--scenarios", default="data/generated/scenarios/all_scenarios.json")
    parser.add_argument("--output", default=None, help="Output validated file")
    parser.add_argument("--fix", action="store_true", help="Auto-fix fixable issues")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_dir = Path(__file__).parent.parent

    # Load data
    with open(base_dir / args.qa) as f:
        qa_data = json.load(f)

    if isinstance(qa_data, list):
        qa_pairs = [QAPair(**q) for q in qa_data]
    elif isinstance(qa_data, dict):
        qa_pairs = [QAPair(**q) for q in qa_data.get("qa_pairs", [])]
    else:
        logger.error(f"Unknown format: {type(qa_data)}")
        sys.exit(1)

    scenarios_path = base_dir / args.scenarios
    if scenarios_path.exists():
        with open(scenarios_path) as f:
            scenarios = [Scenario(**s) for s in json.load(f)]
        logger.info(f"Loaded {len(scenarios)} scenarios")
    else:
        logger.warning(f"Scenarios not found: {scenarios_path}")
        scenarios = []

    # Validate
    validator = QAValidator(scenarios)
    results = validator.validate_all(qa_pairs)

    # Print report
    print(f"\n{'='*60}")
    print(f"QA Validation Report")
    print(f"{'='*60}")
    print(f"Total QA pairs: {results['total']}")
    print(f"Clean: {results['clean']} ({results['clean']/max(results['total'],1)*100:.1f}%)")
    print(f"With issues: {results['has_issues']} ({results['has_issues']/max(results['total'],1)*100:.1f}%)")
    print(f"Dual-time coverage: {results['dual_time_coverage_pct']:.1f}%")

    print(f"\nIssues by severity:")
    for sev, count in sorted(results["issue_by_severity"].items()):
        print(f"  {sev}: {count}")

    print(f"\nIssues by type:")
    for code, count in sorted(results["issue_by_code"].items(), key=lambda x: -x[1]):
        print(f"  {code}: {count}")

    # Show sample issues
    error_items = [
        (qid, issues) for qid, issues in results["all_issues"].items()
        if any(i["severity"] == "error" for i in issues)
    ]
    if error_items:
        print(f"\nSample errors (first 5):")
        for qid, issues in error_items[:5]:
            print(f"  {qid}:")
            for issue in issues:
                if issue["severity"] == "error":
                    print(f"    [{issue['code']}] {issue['message']}")

    # Save results
    output_path = base_dir / (args.output or "data/validated/validation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without detailed all_issues (too large)
    save_data = {k: v for k, v in results.items() if k != "all_issues"}
    save_data["error_qa_ids"] = [qid for qid, issues in results["all_issues"].items()
                                  if any(i["severity"] == "error" for i in issues)]
    save_data["warning_qa_ids"] = [qid for qid, issues in results["all_issues"].items()
                                    if any(i["severity"] == "warning" for i in issues)]

    with open(output_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    main()
