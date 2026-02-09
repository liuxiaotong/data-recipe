"""Tests for PII detection module."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from datarecipe.cli import main
from datarecipe.pii_detector import (
    PIIDetector,
    PIIMatch,
    PIIReport,
    PIITypeSummary,
    _luhn_check,
)


# ==================== Test data factories ====================


def _make_clean_data(n=10):
    return [{"text": f"Hello world {i}", "label": "pos"} for i in range(n)]


def _make_pii_data():
    return [
        {"text": "Contact me at user@example.com", "label": "pos"},
        {"text": "Call 13812345678 for info", "label": "neg"},
        {"text": "ID: 110101199003077735", "label": "pos"},
        {"text": "Card: 4111 1111 1111 1111", "name": "John"},
        {"text": "Server at 192.168.1.100", "label": "pos"},
        {"text": "Clean text here", "label": "pos"},
        {"text": "SSN: 123-45-6789", "label": "neg"},
        {"text": "Creds: https://admin:pass123@example.com/api", "label": "neg"},
    ]


def _make_nested_data():
    return [
        {
            "messages": [
                {"role": "user", "content": "My email is test@example.com"},
                {"role": "assistant", "content": "Got it!"},
            ]
        },
    ]


# ==================== Pattern tests ====================


class TestPIIPatterns(unittest.TestCase):

    def setUp(self):
        self.detector = PIIDetector()

    def test_email_detected(self):
        matches = self.detector._scan_text("email: user@example.com", "text", 0)
        types = [m.pii_type for m in matches]
        self.assertIn("email", types)

    def test_email_not_detected_in_plain_text(self):
        matches = self.detector._scan_text("no email here", "text", 0)
        email_matches = [m for m in matches if m.pii_type == "email"]
        self.assertEqual(len(email_matches), 0)

    def test_phone_cn_detected(self):
        matches = self.detector._scan_text("call 13912345678", "text", 0)
        types = [m.pii_type for m in matches]
        self.assertIn("phone_cn", types)

    def test_id_card_cn_detected(self):
        matches = self.detector._scan_text("ID 110101199003077735", "text", 0)
        types = [m.pii_type for m in matches]
        self.assertIn("id_card_cn", types)

    def test_credit_card_luhn_valid(self):
        # 4111 1111 1111 1111 passes Luhn
        matches = self.detector._scan_text("card 4111 1111 1111 1111", "text", 0)
        cc_matches = [m for m in matches if m.pii_type == "credit_card"]
        self.assertGreater(len(cc_matches), 0)

    def test_credit_card_luhn_invalid_rejected(self):
        # 1234 5678 9012 3456 fails Luhn
        matches = self.detector._scan_text("card 1234 5678 9012 3456", "text", 0)
        cc_matches = [m for m in matches if m.pii_type == "credit_card"]
        self.assertEqual(len(cc_matches), 0)

    def test_ip_address_detected(self):
        matches = self.detector._scan_text("host 192.168.1.100", "text", 0)
        types = [m.pii_type for m in matches]
        self.assertIn("ip_address", types)

    def test_ssn_us_detected(self):
        matches = self.detector._scan_text("SSN: 123-45-6789", "text", 0)
        types = [m.pii_type for m in matches]
        self.assertIn("ssn_us", types)

    def test_url_with_credentials_detected(self):
        matches = self.detector._scan_text(
            "https://admin:pass@example.com", "text", 0
        )
        types = [m.pii_type for m in matches]
        self.assertIn("url_with_credentials", types)


# ==================== Luhn tests ====================


class TestLuhn(unittest.TestCase):

    def test_valid_card(self):
        self.assertTrue(_luhn_check("4111111111111111"))

    def test_invalid_card(self):
        self.assertFalse(_luhn_check("1234567890123456"))

    def test_too_short(self):
        self.assertFalse(_luhn_check("123"))


# ==================== Masking tests ====================


class TestPIIMasking(unittest.TestCase):

    def setUp(self):
        self.detector = PIIDetector()

    def test_mask_email(self):
        result = self.detector._mask_value("user@example.com", "email")
        self.assertIn("***", result)
        self.assertIn("@example.com", result)
        self.assertNotIn("user", result)

    def test_mask_phone_cn(self):
        result = self.detector._mask_value("13812345678", "phone_cn")
        self.assertIn("138", result)
        self.assertIn("5678", result)
        self.assertIn("****", result)

    def test_mask_id_card(self):
        result = self.detector._mask_value("110101199003077735", "id_card_cn")
        self.assertTrue(result.startswith("110101"))
        self.assertTrue(result.endswith("7735"))
        self.assertIn("****", result)

    def test_mask_credit_card(self):
        result = self.detector._mask_value("4111 1111 1111 1111", "credit_card")
        self.assertIn("4111", result)
        self.assertIn("1111", result)
        self.assertIn("****", result)

    def test_mask_ssn(self):
        result = self.detector._mask_value("123-45-6789", "ssn_us")
        self.assertTrue(result.endswith("6789"))
        self.assertIn("***", result)


# ==================== Detector tests ====================


class TestPIIDetector(unittest.TestCase):

    def test_clean_data_no_pii(self):
        detector = PIIDetector()
        report = detector.analyze_sample(_make_clean_data())
        self.assertEqual(report.risk_level, "none")
        self.assertEqual(report.samples_with_pii, 0)
        self.assertAlmostEqual(report.pii_ratio, 0.0)

    def test_pii_data_detected(self):
        detector = PIIDetector()
        report = detector.analyze_sample(_make_pii_data())
        self.assertGreater(report.samples_with_pii, 0)
        self.assertGreater(len(report.type_summaries), 0)
        self.assertNotEqual(report.risk_level, "none")

    def test_empty_data(self):
        detector = PIIDetector()
        report = detector.analyze_sample([])
        self.assertEqual(report.total_samples, 0)
        self.assertEqual(report.risk_level, "none")

    def test_specific_pii_types(self):
        detector = PIIDetector(pii_types=["email"])
        report = detector.analyze_sample(_make_pii_data())
        types_found = {s.pii_type for s in report.type_summaries}
        self.assertIn("email", types_found)
        self.assertNotIn("phone_cn", types_found)

    def test_unknown_pii_type_raises(self):
        with self.assertRaises(ValueError):
            PIIDetector(pii_types=["unknown_type"])

    def test_nested_data_scanned(self):
        detector = PIIDetector()
        report = detector.analyze_sample(_make_nested_data())
        self.assertGreater(report.samples_with_pii, 0)
        types_found = {s.pii_type for s in report.type_summaries}
        self.assertIn("email", types_found)

    def test_specific_text_fields(self):
        detector = PIIDetector()
        data = [{"text": "user@example.com", "safe": "hello"}]
        report = detector.analyze_sample(data, text_fields=["safe"])
        self.assertEqual(report.samples_with_pii, 0)

    def test_specific_text_fields_with_pii(self):
        detector = PIIDetector()
        data = [{"text": "user@example.com", "safe": "hello"}]
        report = detector.analyze_sample(data, text_fields=["text"])
        self.assertEqual(report.samples_with_pii, 1)

    def test_all_string_fields_scanned_by_default(self):
        detector = PIIDetector(pii_types=["email"])
        data = [{"name": "user@test.com", "bio": "hello"}]
        report = detector.analyze_sample(data)
        self.assertEqual(report.samples_with_pii, 1)

    def test_pii_ratio_correct(self):
        detector = PIIDetector(pii_types=["email"])
        data = [
            {"text": "user@example.com"},
            {"text": "clean text"},
            {"text": "another@test.org"},
            {"text": "safe content"},
        ]
        report = detector.analyze_sample(data)
        self.assertAlmostEqual(report.pii_ratio, 0.5)


# ==================== Report tests ====================


class TestPIIReport(unittest.TestCase):

    def test_risk_level_none(self):
        detector = PIIDetector()
        self.assertEqual(detector._determine_risk_level(0.0, 0), "none")

    def test_risk_level_low(self):
        detector = PIIDetector()
        self.assertEqual(detector._determine_risk_level(0.005, 1), "low")

    def test_risk_level_medium(self):
        detector = PIIDetector()
        self.assertEqual(detector._determine_risk_level(0.05, 2), "medium")

    def test_risk_level_high(self):
        detector = PIIDetector()
        self.assertEqual(detector._determine_risk_level(0.15, 5), "high")

    def test_to_dict(self):
        report = PIIReport(
            total_samples=100,
            samples_with_pii=10,
            pii_ratio=0.1,
            type_summaries=[
                PIITypeSummary("email", 5, ["text"], ["a***@test.com"]),
            ],
            risk_level="medium",
            recommendations=["Review data"],
        )
        d = report.to_dict()
        self.assertEqual(d["total_samples"], 100)
        self.assertEqual(d["risk_level"], "medium")
        self.assertEqual(len(d["type_summaries"]), 1)

    def test_recommendations_generated(self):
        detector = PIIDetector()
        report = detector.analyze_sample(_make_pii_data())
        self.assertGreater(len(report.recommendations), 0)


# ==================== File loading tests ====================


class TestPIIFromFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv(self, rows=None):
        if rows is None:
            rows = [
                {"text": "Contact user@example.com", "label": "pos"},
                {"text": "Call 13812345678", "label": "neg"},
                {"text": "Clean text", "label": "pos"},
            ]
        path = os.path.join(self.tmpdir, "test.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _create_jsonl(self, rows=None):
        if rows is None:
            rows = [
                {"text": "Email: test@domain.org", "label": "pos"},
                {"text": "Safe text", "label": "neg"},
            ]
        path = os.path.join(self.tmpdir, "test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_analyze_from_csv(self):
        path = self._create_csv()
        detector = PIIDetector()
        report = detector.analyze_from_file(path)
        self.assertGreater(report.samples_with_pii, 0)

    def test_analyze_from_jsonl(self):
        path = self._create_jsonl()
        detector = PIIDetector()
        report = detector.analyze_from_file(path)
        self.assertGreater(report.samples_with_pii, 0)

    def test_file_not_found(self):
        detector = PIIDetector()
        with self.assertRaises(FileNotFoundError):
            detector.analyze_from_file("/nonexistent/data.csv")

    def test_clean_file_no_pii(self):
        rows = [{"text": f"Hello {i}", "label": "pos"} for i in range(5)]
        path = self._create_csv(rows)
        detector = PIIDetector()
        report = detector.analyze_from_file(path)
        self.assertEqual(report.risk_level, "none")


# ==================== CLI tests ====================


class TestPIICLI(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv(self):
        path = os.path.join(self.tmpdir, "test.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "user@example.com", "label": "pos"})
            writer.writerow({"text": "safe text", "label": "neg"})
        return path

    def test_pii_help(self):
        result = self.runner.invoke(main, ["pii", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("PII", result.output)

    def test_pii_local_csv(self):
        path = self._create_csv()
        result = self.runner.invoke(main, ["pii", path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("PII Detection Report", result.output)

    def test_pii_json_output(self):
        path = self._create_csv()
        result = self.runner.invoke(main, ["pii", path, "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("risk_level", data)

    def test_pii_nonexistent_file(self):
        result = self.runner.invoke(main, ["pii", "/nonexistent/data.csv"])
        self.assertNotEqual(result.exit_code, 0)


# ==================== Dataclass tests ====================


class TestPIIDataclasses(unittest.TestCase):

    def test_pii_match_to_dict(self):
        m = PIIMatch("email", "text", "u***@test.com", 0, "high")
        d = m.to_dict()
        self.assertEqual(d["pii_type"], "email")
        self.assertEqual(d["confidence"], "high")

    def test_pii_type_summary_to_dict(self):
        s = PIITypeSummary("phone_cn", 3, ["text", "bio"], ["138****5678"])
        d = s.to_dict()
        self.assertEqual(d["count"], 3)
        self.assertEqual(len(d["affected_fields"]), 2)


if __name__ == "__main__":
    unittest.main()
