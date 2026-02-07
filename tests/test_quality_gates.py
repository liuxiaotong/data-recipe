"""Tests for Quality Gate mechanism (Upgrade 4)."""

import pytest
from datarecipe.quality_metrics import (
    AIDetectionMetrics,
    ComplexityMetrics,
    ConsistencyMetrics,
    DEFAULT_QUALITY_GATES,
    DiversityMetrics,
    GateResult,
    QualityAnalyzer,
    QualityGateReport,
    QualityGateRule,
    QualityReport,
)


def _make_report(
    overall_score=75.0,
    unique_token_ratio=0.15,
    format_consistency=0.9,
    field_completeness=0.95,
    ai_probability=None,
) -> QualityReport:
    """Helper to create a QualityReport with given values."""
    diversity = DiversityMetrics(
        unique_token_ratio=unique_token_ratio,
        vocabulary_size=1000,
        semantic_diversity=0.6,
    )
    consistency = ConsistencyMetrics(
        format_consistency=format_consistency,
        structure_score=0.8,
        field_completeness=field_completeness,
        length_variance=0.3,
    )
    complexity = ComplexityMetrics(
        avg_length=200.0,
        avg_tokens=50.0,
        vocabulary_richness=0.3,
        avg_sentence_length=15.0,
        readability_score=60.0,
    )
    ai_detection = None
    if ai_probability is not None:
        ai_detection = AIDetectionMetrics(
            ai_probability=ai_probability,
            confidence=0.7,
            indicators=[],
        )
    return QualityReport(
        diversity=diversity,
        consistency=consistency,
        complexity=complexity,
        ai_detection=ai_detection,
        overall_score=overall_score,
        sample_size=100,
    )


class TestQualityGateRule:
    def test_to_dict(self):
        rule = QualityGateRule("g1", "Test Gate", "overall_score", ">=", 60, "blocker")
        d = rule.to_dict()
        assert d["gate_id"] == "g1"
        assert d["threshold"] == 60
        assert d["severity"] == "blocker"


class TestQualityAnalyzerEvaluateGates:
    def setup_method(self):
        self.analyzer = QualityAnalyzer()

    def test_all_pass(self):
        report = _make_report(overall_score=80, unique_token_ratio=0.2, format_consistency=0.9)
        gate_report = self.analyzer.evaluate_gates(report)
        assert gate_report.passed is True
        assert len(gate_report.blocking_failures) == 0

    def test_blocker_failure(self):
        report = _make_report(overall_score=40)  # below 60 threshold
        gate_report = self.analyzer.evaluate_gates(report)
        assert gate_report.passed is False
        assert len(gate_report.blocking_failures) > 0
        assert gate_report.blocking_failures[0].gate.gate_id == "min_overall_score"

    def test_warning_not_blocking(self):
        report = _make_report(overall_score=80, ai_probability=0.8)  # above 0.5 warning threshold
        gate_report = self.analyzer.evaluate_gates(report)
        # ai_probability warning doesn't block
        assert gate_report.passed is True
        assert len(gate_report.warnings) > 0

    def test_missing_metric_skipped(self):
        """When ai_detection is None, ai-related gates should be skipped."""
        report = _make_report(overall_score=80, ai_probability=None)
        gate_report = self.analyzer.evaluate_gates(report)
        assert gate_report.passed is True

    def test_custom_gates(self):
        custom_gates = [
            QualityGateRule("custom1", "High Score", "overall_score", ">", 90, "blocker"),
        ]
        report = _make_report(overall_score=85)
        gate_report = self.analyzer.evaluate_gates(report, gates=custom_gates)
        assert gate_report.passed is False
        assert gate_report.blocking_failures[0].gate.gate_id == "custom1"

    def test_all_operators(self):
        report = _make_report(overall_score=60)
        operators_and_expected = [
            (">=", 60, True),
            (">=", 61, False),
            ("<=", 60, True),
            ("<=", 59, False),
            (">", 59, True),
            (">", 60, False),
            ("<", 61, True),
            ("<", 60, False),
            ("==", 60, True),
            ("==", 61, False),
            ("!=", 61, True),
            ("!=", 60, False),
        ]
        for op, threshold, expected in operators_and_expected:
            gate = QualityGateRule("test", "test", "overall_score", op, threshold, "blocker")
            result = self.analyzer.evaluate_gates(report, gates=[gate])
            assert result.passed == expected, f"Failed for {op} {threshold}: expected {expected}"

    def test_nested_metric_extraction(self):
        report = _make_report(unique_token_ratio=0.03)
        gate = QualityGateRule(
            "low_div", "Low diversity", "diversity.unique_token_ratio", ">=", 0.05, "blocker"
        )
        gate_report = self.analyzer.evaluate_gates(report, gates=[gate])
        assert gate_report.passed is False
        assert gate_report.blocking_failures[0].actual_value == pytest.approx(0.03, abs=0.001)

    def test_report_attached(self):
        """evaluate_gates should attach gate_report to the QualityReport."""
        report = _make_report()
        self.analyzer.evaluate_gates(report)
        assert report.gate_report is not None
        assert isinstance(report.gate_report, QualityGateReport)


class TestQualityGateReportToDict:
    def test_to_dict(self):
        gate = QualityGateRule("g1", "Test", "overall_score", ">=", 60, "blocker")
        gr = GateResult(gate=gate, actual_value=50.0, passed=False, message="fail")
        report = QualityGateReport(
            passed=False,
            results=[gr],
            blocking_failures=[gr],
            warnings=[],
        )
        d = report.to_dict()
        assert d["passed"] is False
        assert len(d["blocking_failures"]) == 1
        assert d["blocking_failures"][0]["actual_value"] == 50.0


class TestDefaultQualityGates:
    def test_default_gates_exist(self):
        assert len(DEFAULT_QUALITY_GATES) == 5

    def test_default_gates_have_correct_types(self):
        for gate in DEFAULT_QUALITY_GATES:
            assert isinstance(gate, QualityGateRule)
            assert gate.operator in (">=", "<=", ">", "<", "==", "!=")
            assert gate.severity in ("blocker", "warning")
