"""Tests for local file source support (CSV, Parquet, JSONL)."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.cli import main
from datarecipe.schema import GenerationType, SourceType
from datarecipe.sources.local import (
    SUPPORTED_EXTENSIONS,
    LocalFileExtractor,
    detect_format,
)


class TestDetectFormat(unittest.TestCase):
    """Test the detect_format() helper."""

    def test_csv(self):
        self.assertEqual(detect_format(Path("data.csv")), "csv")

    def test_parquet(self):
        self.assertEqual(detect_format(Path("data.parquet")), "parquet")

    def test_jsonl(self):
        self.assertEqual(detect_format(Path("data.jsonl")), "json")

    def test_json(self):
        self.assertEqual(detect_format(Path("data.json")), "json")

    def test_uppercase_extension(self):
        self.assertEqual(detect_format(Path("DATA.CSV")), "csv")

    def test_unsupported_extension(self):
        with self.assertRaises(ValueError) as ctx:
            detect_format(Path("data.xlsx"))
        self.assertIn("Unsupported file format", str(ctx.exception))

    def test_supported_extensions_set(self):
        self.assertIn(".csv", SUPPORTED_EXTENSIONS)
        self.assertIn(".parquet", SUPPORTED_EXTENSIONS)
        self.assertIn(".jsonl", SUPPORTED_EXTENSIONS)
        self.assertIn(".json", SUPPORTED_EXTENSIONS)


class TestLocalFileExtractor(unittest.TestCase):
    """Test LocalFileExtractor with real temp files."""

    def setUp(self):
        self.extractor = LocalFileExtractor()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv(self, filename="test.csv", rows=None):
        if rows is None:
            rows = [
                {"text": "Hello world", "label": "positive"},
                {"text": "Bad day", "label": "negative"},
                {"text": "Great news", "label": "positive"},
            ]
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _create_jsonl(self, filename="test.jsonl", rows=None):
        if rows is None:
            rows = [
                {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"},
                {"instruction": "Summarize", "input": "Long text...", "output": "Short."},
                {"instruction": "Classify", "input": "Great product", "output": "positive"},
            ]
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _create_preference_jsonl(self, filename="pref.jsonl"):
        rows = [
            {"prompt": "What is AI?", "chosen": "AI is...", "rejected": "I dunno"},
            {"prompt": "Explain ML", "chosen": "ML is...", "rejected": "No idea"},
        ]
        return self._create_jsonl(filename, rows)

    def _create_conversation_jsonl(self, filename="conv.jsonl"):
        rows = [
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
            {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]},
        ]
        return self._create_jsonl(filename, rows)

    def test_extract_csv(self):
        path = self._create_csv()
        recipe = self.extractor.extract(path)

        self.assertEqual(recipe.source_type, SourceType.LOCAL)
        self.assertEqual(recipe.name, "test")
        self.assertIn("local", recipe.tags)
        self.assertIn("csv", recipe.tags)
        self.assertIsNotNone(recipe.size)
        self.assertIsNotNone(recipe.num_examples)
        self.assertIn("text", recipe.description)

    def test_extract_jsonl(self):
        path = self._create_jsonl()
        recipe = self.extractor.extract(path)

        self.assertEqual(recipe.source_type, SourceType.LOCAL)
        self.assertEqual(recipe.name, "test")
        self.assertIn("instruction", recipe.description)

    def test_detect_preference_type(self):
        path = self._create_preference_jsonl()
        recipe = self.extractor.extract(path)

        self.assertIn("RLHF", recipe.description)
        self.assertIn("preference", recipe.tags)

    def test_detect_conversation_type(self):
        path = self._create_conversation_jsonl()
        recipe = self.extractor.extract(path)

        self.assertIn("conversation", recipe.tags)

    def test_detect_instruction_type(self):
        path = self._create_jsonl()
        recipe = self.extractor.extract(path)

        self.assertIn("instruction-tuning", recipe.tags)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract("/nonexistent/path/data.csv")

    def test_unsupported_format(self):
        path = os.path.join(self.tmpdir, "data.xlsx")
        with open(path, "w") as f:
            f.write("dummy")
        with self.assertRaises(ValueError):
            self.extractor.extract(path)

    def test_not_a_file(self):
        with self.assertRaises(ValueError):
            self.extractor.extract(self.tmpdir)

    def test_source_id_is_absolute_path(self):
        path = self._create_csv()
        recipe = self.extractor.extract(path)

        self.assertTrue(os.path.isabs(recipe.source_id))

    def test_generation_type_for_preference(self):
        path = self._create_preference_jsonl()
        recipe = self.extractor.extract(path)

        self.assertEqual(recipe.generation_type, GenerationType.MIXED)


class TestAnalyzerLocalRouting(unittest.TestCase):
    """Test that DatasetAnalyzer correctly routes local files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv(self):
        path = os.path.join(self.tmpdir, "test.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "hello", "label": "pos"})
        return path

    def test_parse_dataset_input_csv(self):
        path = self._create_csv()
        analyzer = DatasetAnalyzer()
        resolved, source_type = analyzer._parse_dataset_input(path)

        self.assertEqual(source_type, SourceType.LOCAL)
        self.assertTrue(os.path.isabs(resolved))

    def test_parse_dataset_input_nonexistent_csv(self):
        analyzer = DatasetAnalyzer()
        with self.assertRaises(FileNotFoundError):
            analyzer._parse_dataset_input("/nonexistent/data.csv")

    def test_parse_dataset_input_hf_id_unchanged(self):
        analyzer = DatasetAnalyzer()
        dataset_id, source_type = analyzer._parse_dataset_input("org/dataset-name")

        self.assertEqual(dataset_id, "org/dataset-name")
        self.assertIsNone(source_type)

    def test_detect_source_type_local_file(self):
        path = self._create_csv()
        analyzer = DatasetAnalyzer()
        source_type = analyzer._detect_source_type(path)

        self.assertEqual(source_type, SourceType.LOCAL)

    def test_analyze_local_csv(self):
        path = self._create_csv()
        analyzer = DatasetAnalyzer()
        recipe = analyzer.analyze(path)

        self.assertEqual(recipe.source_type, SourceType.LOCAL)
        self.assertEqual(recipe.name, "test")

    def test_local_extractor_registered(self):
        analyzer = DatasetAnalyzer()
        self.assertIn(SourceType.LOCAL, analyzer.extractors)


class TestLocalCLIAnalyze(unittest.TestCase):
    """Test CLI commands with local files."""

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
            for i in range(5):
                writer.writerow({"text": f"Sample text {i}", "label": "positive"})
        return path

    def _create_jsonl(self):
        path = os.path.join(self.tmpdir, "test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({"instruction": f"Task {i}", "input": "data", "output": "result"}) + "\n")
        return path

    def test_analyze_local_csv(self):
        path = self._create_csv()
        result = self.runner.invoke(main, ["analyze", path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test", result.output)

    def test_analyze_local_jsonl(self):
        path = self._create_jsonl()
        result = self.runner.invoke(main, ["analyze", path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test", result.output)

    def test_analyze_local_csv_json_output(self):
        path = self._create_csv()
        result = self.runner.invoke(main, ["analyze", path, "--json"])
        self.assertEqual(result.exit_code, 0)

    def test_analyze_nonexistent_file(self):
        result = self.runner.invoke(main, ["analyze", "/nonexistent/data.csv"])
        self.assertNotEqual(result.exit_code, 0)


class TestLocalQualityAnalysis(unittest.TestCase):
    """Test quality analysis from local files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_csv(self):
        path = os.path.join(self.tmpdir, "test.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            for i in range(20):
                writer.writerow({"text": f"This is sample text number {i} for quality analysis testing.", "label": "pos"})
        return path

    def test_analyze_from_file(self):
        from datarecipe.quality_metrics import QualityAnalyzer

        path = self._create_csv()
        analyzer = QualityAnalyzer()
        report = analyzer.analyze_from_file(path, text_field="text", sample_size=10)

        self.assertGreater(report.sample_size, 0)

    def test_analyze_from_file_auto_detect_field(self):
        from datarecipe.quality_metrics import QualityAnalyzer

        path = self._create_csv()
        analyzer = QualityAnalyzer()
        # Use a non-existent field name to trigger auto-detection
        report = analyzer.analyze_from_file(path, text_field="nonexistent", sample_size=10)

        self.assertGreater(report.sample_size, 0)


class TestDeepAnalyzerLocalFile(unittest.TestCase):
    """Test DeepAnalyzerCore with local file paths."""

    def test_local_path_detection(self):
        """Test that local file path is correctly detected."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            f.write(b'{"text": "hello"}\n')
            tmp_path = f.name

        try:
            p = Path(tmp_path)
            self.assertTrue(p.exists())
            self.assertTrue(p.is_file())
            self.assertEqual(detect_format(p), "json")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
