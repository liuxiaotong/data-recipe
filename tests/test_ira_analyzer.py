"""Tests for inter-rater agreement (IRA) analyzer."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from datarecipe.cli import main
from datarecipe.ira_analyzer import (
    AnnotatorStats,
    DisagreementPattern,
    IRAAnalyzer,
    IRAReport,
    PairwiseAgreement,
)


# ==================== Test data factories ====================


def _make_perfect_agreement(n=20):
    """Two annotators with perfect agreement."""
    labels = ["pos", "neg"] * (n // 2)
    data = []
    for i, label in enumerate(labels):
        data.append({"item_id": str(i), "annotator_id": "ann_A", "label": label})
        data.append({"item_id": str(i), "annotator_id": "ann_B", "label": label})
    return data


def _make_no_agreement(n=20):
    """Two annotators that always disagree."""
    data = []
    for i in range(n):
        data.append({"item_id": str(i), "annotator_id": "ann_A", "label": "pos"})
        data.append({"item_id": str(i), "annotator_id": "ann_B", "label": "neg"})
    return data


def _make_partial_agreement():
    """Two annotators with ~70% agreement."""
    # 10 items: agree on 7, disagree on 3
    data = []
    labels_a = ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "pos", "neg", "pos"]
    labels_b = ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
    for i in range(10):
        data.append({"item_id": str(i), "annotator_id": "ann_A", "label": labels_a[i]})
        data.append({"item_id": str(i), "annotator_id": "ann_B", "label": labels_b[i]})
    return data


def _make_multi_annotator():
    """Three annotators with varying agreement."""
    data = []
    labels = {
        "ann_A": ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "neg", "pos", "neg"],
        "ann_B": ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "pos", "neg", "neg"],
        "ann_C": ["pos", "neg", "neg", "pos", "pos", "neg", "pos", "neg", "pos", "neg"],
    }
    for ann_id, ann_labels in labels.items():
        for i, label in enumerate(ann_labels):
            data.append({"item_id": str(i), "annotator_id": ann_id, "label": label})
    return data


def _make_wide_format():
    """Wide-format data: one row per item, annotators as columns."""
    return [
        {"item_id": "0", "ann_A": "pos", "ann_B": "pos", "ann_C": "pos"},
        {"item_id": "1", "ann_A": "pos", "ann_B": "neg", "ann_C": "pos"},
        {"item_id": "2", "ann_A": "neg", "ann_B": "neg", "ann_C": "neg"},
        {"item_id": "3", "ann_A": "neg", "ann_B": "pos", "ann_C": "neg"},
        {"item_id": "4", "ann_A": "pos", "ann_B": "pos", "ann_C": "neg"},
    ]


def _make_three_label():
    """Data with three labels for disagreement pattern testing."""
    data = []
    configs = [
        ("0", "ann_A", "pos"), ("0", "ann_B", "neg"),
        ("1", "ann_A", "pos"), ("1", "ann_B", "neutral"),
        ("2", "ann_A", "neg"), ("2", "ann_B", "neutral"),
        ("3", "ann_A", "pos"), ("3", "ann_B", "neg"),
        ("4", "ann_A", "pos"), ("4", "ann_B", "pos"),
    ]
    for item_id, ann, label in configs:
        data.append({"item_id": item_id, "annotator_id": ann, "label": label})
    return data


# ==================== Cohen's Kappa tests ====================


class TestCohenKappa(unittest.TestCase):

    def setUp(self):
        self.analyzer = IRAAnalyzer()

    def test_perfect_agreement(self):
        labels = ["pos", "neg", "pos", "neg", "pos"]
        kappa = self.analyzer._cohen_kappa(labels, labels)
        self.assertAlmostEqual(kappa, 1.0, places=4)

    def test_complete_disagreement(self):
        # Alternating labels with systematic disagreement → negative kappa
        labels_a = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
        labels_b = ["neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos"]
        kappa = self.analyzer._cohen_kappa(labels_a, labels_b)
        self.assertLess(kappa, 0.0)

    def test_partial_agreement(self):
        labels_a = ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "pos", "neg", "pos"]
        labels_b = ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
        kappa = self.analyzer._cohen_kappa(labels_a, labels_b)
        self.assertGreater(kappa, 0.0)
        self.assertLess(kappa, 1.0)

    def test_empty_labels(self):
        kappa = self.analyzer._cohen_kappa([], [])
        self.assertEqual(kappa, 0.0)

    def test_single_label_both_agree(self):
        labels = ["pos"] * 10
        kappa = self.analyzer._cohen_kappa(labels, labels)
        # All same label → pe=1.0, special case returns 1.0
        self.assertEqual(kappa, 1.0)

    def test_mismatched_lengths(self):
        kappa = self.analyzer._cohen_kappa(["pos", "neg"], ["pos"])
        self.assertEqual(kappa, 0.0)


# ==================== Fleiss' Kappa tests ====================


class TestFleissKappa(unittest.TestCase):

    def setUp(self):
        self.analyzer = IRAAnalyzer()

    def test_perfect_agreement(self):
        items = {
            "0": {"A": "pos", "B": "pos", "C": "pos"},
            "1": {"A": "neg", "B": "neg", "C": "neg"},
            "2": {"A": "pos", "B": "pos", "C": "pos"},
        }
        kappa = self.analyzer._fleiss_kappa(items, ["neg", "pos"])
        self.assertAlmostEqual(kappa, 1.0, places=4)

    def test_no_agreement(self):
        # Each item has one of each label
        items = {
            "0": {"A": "pos", "B": "neg", "C": "neutral"},
            "1": {"A": "neg", "B": "neutral", "C": "pos"},
            "2": {"A": "neutral", "B": "pos", "C": "neg"},
        }
        kappa = self.analyzer._fleiss_kappa(items, ["neg", "neutral", "pos"])
        self.assertLess(kappa, 0.1)

    def test_empty_items(self):
        kappa = self.analyzer._fleiss_kappa({}, ["pos", "neg"])
        self.assertEqual(kappa, 0.0)

    def test_single_item(self):
        items = {"0": {"A": "pos", "B": "pos"}}
        kappa = self.analyzer._fleiss_kappa(items, ["pos"])
        # All agree on single label
        self.assertGreaterEqual(kappa, 0.0)


# ==================== Krippendorff's Alpha tests ====================


class TestKrippendorffAlpha(unittest.TestCase):

    def setUp(self):
        self.analyzer = IRAAnalyzer()

    def test_perfect_agreement(self):
        annotations = [
            ("0", "A", "pos"), ("0", "B", "pos"),
            ("1", "A", "neg"), ("1", "B", "neg"),
            ("2", "A", "pos"), ("2", "B", "pos"),
        ]
        alpha = self.analyzer._krippendorff_alpha(annotations)
        self.assertAlmostEqual(alpha, 1.0, places=4)

    def test_no_agreement(self):
        annotations = [
            ("0", "A", "pos"), ("0", "B", "neg"),
            ("1", "A", "neg"), ("1", "B", "pos"),
        ]
        alpha = self.analyzer._krippendorff_alpha(annotations)
        self.assertLess(alpha, 0.1)

    def test_empty_annotations(self):
        alpha = self.analyzer._krippendorff_alpha([])
        self.assertEqual(alpha, 0.0)

    def test_partial_agreement(self):
        annotations = [
            ("0", "A", "pos"), ("0", "B", "pos"),
            ("1", "A", "pos"), ("1", "B", "pos"),
            ("2", "A", "neg"), ("2", "B", "neg"),
            ("3", "A", "pos"), ("3", "B", "neg"),
        ]
        alpha = self.analyzer._krippendorff_alpha(annotations)
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)


# ==================== Format detection tests ====================


class TestFormatDetection(unittest.TestCase):

    def setUp(self):
        self.analyzer = IRAAnalyzer()

    def test_detect_long_format(self):
        data = [{"item_id": "0", "annotator_id": "A", "label": "pos"}]
        fmt = self.analyzer._detect_format(data, "item_id", "annotator_id")
        self.assertEqual(fmt, "long")

    def test_detect_wide_format(self):
        data = [{"item_id": "0", "ann_A": "pos", "ann_B": "neg"}]
        fmt = self.analyzer._detect_format(data, "item_id", "annotator_id")
        self.assertEqual(fmt, "wide")

    def test_detect_empty_data(self):
        fmt = self.analyzer._detect_format([], "item_id", "annotator_id")
        self.assertEqual(fmt, "long")

    def test_wide_to_long_conversion(self):
        data = _make_wide_format()
        tuples = self.analyzer._wide_to_long(data, "label")
        self.assertGreater(len(tuples), 0)
        # Each item should have 3 annotations (3 annotator columns)
        item_ids = {t[0] for t in tuples}
        self.assertEqual(len(item_ids), 5)

    def test_parse_long_format(self):
        data = _make_perfect_agreement(4)
        tuples = self.analyzer._parse_long(data, "item_id", "annotator_id", "label")
        self.assertEqual(len(tuples), 8)  # 4 items × 2 annotators

    def test_parse_long_skips_missing_fields(self):
        data = [
            {"item_id": "0", "annotator_id": "A", "label": "pos"},
            {"item_id": "1", "annotator_id": "B"},  # missing label
            {"item_id": "2", "label": "neg"},  # missing annotator
        ]
        tuples = self.analyzer._parse_long(data, "item_id", "annotator_id", "label")
        self.assertEqual(len(tuples), 1)


# ==================== IRAAnalyzer main tests ====================


class TestIRAAnalyzer(unittest.TestCase):

    def test_perfect_agreement_report(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_perfect_agreement())
        self.assertEqual(report.quality_level, "excellent")
        self.assertAlmostEqual(report.avg_pairwise_kappa, 1.0, places=4)
        self.assertEqual(report.n_annotators, 2)

    def test_no_agreement_report(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_no_agreement())
        # All "pos" vs all "neg" → pe=0, po=0 → kappa=0
        self.assertLessEqual(report.avg_pairwise_kappa, 0.0)
        self.assertEqual(report.quality_level, "poor")

    def test_partial_agreement_report(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_partial_agreement())
        self.assertGreater(report.avg_pairwise_kappa, 0.0)
        self.assertLess(report.avg_pairwise_kappa, 1.0)

    def test_multi_annotator(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_multi_annotator())
        self.assertEqual(report.n_annotators, 3)
        # 3 annotators → 3 pairwise comparisons
        self.assertEqual(len(report.pairwise_agreements), 3)
        self.assertGreater(report.fleiss_kappa, -1.0)

    def test_wide_format(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_wide_format(), data_format="wide")
        self.assertEqual(report.n_annotators, 3)
        self.assertGreater(report.total_items, 0)

    def test_wide_format_auto_detect(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_wide_format())
        self.assertEqual(report.n_annotators, 3)

    def test_empty_data(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample([])
        self.assertEqual(report.total_items, 0)
        self.assertEqual(report.quality_level, "poor")

    def test_single_annotator_filtered(self):
        data = [
            {"item_id": "0", "annotator_id": "A", "label": "pos"},
            {"item_id": "1", "annotator_id": "A", "label": "neg"},
        ]
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(data)
        # No items have 2+ annotators
        self.assertEqual(report.total_items, 0)

    def test_min_overlap(self):
        analyzer = IRAAnalyzer(min_overlap=3)
        # 2-annotator data → all filtered out
        report = analyzer.analyze_sample(_make_perfect_agreement())
        self.assertEqual(report.total_items, 0)

    def test_min_overlap_invalid(self):
        with self.assertRaises(ValueError):
            IRAAnalyzer(min_overlap=1)

    def test_disagreement_patterns_found(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_three_label())
        self.assertGreater(len(report.disagreement_patterns), 0)
        # "pos" vs "neg" should be most common (2 times)
        top = report.disagreement_patterns[0]
        self.assertEqual(top.count, 2)

    def test_annotator_stats(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_multi_annotator())
        self.assertEqual(len(report.annotator_stats), 3)
        for stat in report.annotator_stats:
            self.assertGreater(stat.n_annotations, 0)
            self.assertIsInstance(stat.label_distribution, dict)


# ==================== Report tests ====================


class TestIRAReport(unittest.TestCase):

    def test_quality_level_excellent(self):
        analyzer = IRAAnalyzer()
        self.assertEqual(analyzer._determine_quality_level(0.85), "excellent")

    def test_quality_level_good(self):
        analyzer = IRAAnalyzer()
        self.assertEqual(analyzer._determine_quality_level(0.65), "good")

    def test_quality_level_moderate(self):
        analyzer = IRAAnalyzer()
        self.assertEqual(analyzer._determine_quality_level(0.45), "moderate")

    def test_quality_level_fair(self):
        analyzer = IRAAnalyzer()
        self.assertEqual(analyzer._determine_quality_level(0.25), "fair")

    def test_quality_level_poor(self):
        analyzer = IRAAnalyzer()
        self.assertEqual(analyzer._determine_quality_level(0.1), "poor")

    def test_to_dict(self):
        report = IRAReport(
            total_items=100,
            total_annotations=300,
            n_annotators=3,
            labels=["pos", "neg"],
            fleiss_kappa=0.65,
            krippendorff_alpha=0.62,
            avg_pairwise_kappa=0.68,
            percent_agreement=0.82,
            quality_level="good",
            recommendations=["Good agreement"],
        )
        d = report.to_dict()
        self.assertEqual(d["total_items"], 100)
        self.assertEqual(d["quality_level"], "good")
        self.assertEqual(d["fleiss_kappa"], 0.65)
        self.assertIsInstance(d["pairwise_agreements"], list)

    def test_recommendations_generated(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_partial_agreement())
        self.assertGreater(len(report.recommendations), 0)

    def test_recommendations_mention_annotator_count(self):
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_sample(_make_perfect_agreement())
        # Only 2 annotators → should recommend adding a third
        rec_text = " ".join(report.recommendations)
        self.assertIn("2 annotators", rec_text)


# ==================== File loading tests ====================


class TestIRAFromFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv_long(self):
        path = os.path.join(self.tmpdir, "annotations.csv")
        rows = _make_partial_agreement()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["item_id", "annotator_id", "label"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _create_csv_wide(self):
        path = os.path.join(self.tmpdir, "annotations_wide.csv")
        rows = _make_wide_format()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _create_jsonl(self):
        path = os.path.join(self.tmpdir, "annotations.jsonl")
        rows = _make_partial_agreement()
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_analyze_from_csv_long(self):
        path = self._create_csv_long()
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_from_file(path)
        self.assertGreater(report.total_items, 0)
        self.assertEqual(report.n_annotators, 2)

    def test_analyze_from_csv_wide(self):
        path = self._create_csv_wide()
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_from_file(path, data_format="wide")
        self.assertGreater(report.total_items, 0)

    def test_analyze_from_jsonl(self):
        path = self._create_jsonl()
        analyzer = IRAAnalyzer()
        report = analyzer.analyze_from_file(path)
        self.assertGreater(report.total_items, 0)

    def test_file_not_found(self):
        analyzer = IRAAnalyzer()
        with self.assertRaises(FileNotFoundError):
            analyzer.analyze_from_file("/nonexistent/data.csv")


# ==================== CLI tests ====================


class TestIRACLI(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv(self):
        path = os.path.join(self.tmpdir, "test.csv")
        rows = _make_partial_agreement()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["item_id", "annotator_id", "label"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_ira_help(self):
        result = self.runner.invoke(main, ["ira", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("inter-rater agreement", result.output.lower())

    def test_ira_local_csv(self):
        path = self._create_csv()
        result = self.runner.invoke(main, ["ira", path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Inter-Rater Agreement Report", result.output)

    def test_ira_json_output(self):
        path = self._create_csv()
        result = self.runner.invoke(main, ["ira", path, "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("quality_level", data)
        self.assertIn("fleiss_kappa", data)

    def test_ira_nonexistent_file(self):
        result = self.runner.invoke(main, ["ira", "/nonexistent/data.csv"])
        self.assertNotEqual(result.exit_code, 0)


# ==================== Dataclass tests ====================


class TestIRADataclasses(unittest.TestCase):

    def test_pairwise_agreement_to_dict(self):
        p = PairwiseAgreement(
            annotator_a="A", annotator_b="B",
            cohen_kappa=0.75, percent_agreement=0.85,
            n_items=50, confusion_matrix={("pos", "pos"): 30, ("pos", "neg"): 5},
        )
        d = p.to_dict()
        self.assertEqual(d["annotator_a"], "A")
        self.assertEqual(d["cohen_kappa"], 0.75)
        self.assertIn("pos|pos", d["confusion_matrix"])

    def test_annotator_stats_to_dict(self):
        s = AnnotatorStats(
            annotator_id="A", n_annotations=100,
            label_distribution={"pos": 60, "neg": 40},
            avg_kappa=0.72,
        )
        d = s.to_dict()
        self.assertEqual(d["n_annotations"], 100)
        self.assertEqual(d["avg_kappa"], 0.72)

    def test_disagreement_pattern_to_dict(self):
        dp = DisagreementPattern(
            label_a="pos", label_b="neg", count=15,
            examples=["item_1", "item_2"],
        )
        d = dp.to_dict()
        self.assertEqual(d["count"], 15)
        self.assertEqual(len(d["examples"]), 2)


if __name__ == "__main__":
    unittest.main()
