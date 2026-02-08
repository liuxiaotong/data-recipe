"""Unit tests for integrated_report module.

Tests DatasetEntry, WeeklyReport, and IntegratedReportGenerator classes.
"""

import json
import os
import unittest
from datetime import datetime, timedelta

from datarecipe.reports.integrated_report import (
    DatasetEntry,
    IntegratedReportGenerator,
    WeeklyReport,
)

# ==================== DatasetEntry ====================


class TestDatasetEntryDefaults(unittest.TestCase):
    """Test DatasetEntry dataclass defaults."""

    def test_required_field(self):
        entry = DatasetEntry(dataset_id="test/ds")
        self.assertEqual(entry.dataset_id, "test/ds")

    def test_default_values(self):
        entry = DatasetEntry(dataset_id="test/ds")
        self.assertEqual(entry.org, "")
        self.assertEqual(entry.category, "")
        self.assertEqual(entry.downloads, 0)
        self.assertEqual(entry.discovered_date, "")
        self.assertFalse(entry.analyzed)
        self.assertEqual(entry.dataset_type, "")
        self.assertEqual(entry.reproduction_cost, 0.0)
        self.assertEqual(entry.human_percentage, 0.0)
        self.assertEqual(entry.difficulty, "")
        self.assertEqual(entry.sample_count, 0)
        self.assertEqual(entry.guide_path, "")
        self.assertEqual(entry.report_path, "")

    def test_custom_values(self):
        entry = DatasetEntry(
            dataset_id="Anthropic/hh-rlhf",
            org="Anthropic",
            category="preference",
            downloads=50000,
            discovered_date="2025-01-01",
            analyzed=True,
            dataset_type="preference",
            reproduction_cost=15000.0,
            human_percentage=70.0,
            difficulty="medium",
            sample_count=10000,
            guide_path="/path/to/guide.md",
            report_path="/path/to/report.md",
        )
        self.assertEqual(entry.dataset_id, "Anthropic/hh-rlhf")
        self.assertEqual(entry.org, "Anthropic")
        self.assertEqual(entry.category, "preference")
        self.assertEqual(entry.downloads, 50000)
        self.assertEqual(entry.discovered_date, "2025-01-01")
        self.assertTrue(entry.analyzed)
        self.assertEqual(entry.dataset_type, "preference")
        self.assertEqual(entry.reproduction_cost, 15000.0)
        self.assertEqual(entry.human_percentage, 70.0)
        self.assertEqual(entry.difficulty, "medium")
        self.assertEqual(entry.sample_count, 10000)
        self.assertEqual(entry.guide_path, "/path/to/guide.md")
        self.assertEqual(entry.report_path, "/path/to/report.md")


# ==================== WeeklyReport ====================


class TestWeeklyReportDefaults(unittest.TestCase):
    """Test WeeklyReport dataclass defaults."""

    def test_required_fields(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        self.assertEqual(report.period_start, "2025-01-01")
        self.assertEqual(report.period_end, "2025-01-07")

    def test_default_values(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        self.assertEqual(report.generated_at, "")
        self.assertEqual(report.total_discovered, 0)
        self.assertEqual(report.discoveries_by_org, {})
        self.assertEqual(report.discoveries_by_category, {})
        self.assertEqual(report.total_analyzed, 0)
        self.assertEqual(report.analysis_by_type, {})
        self.assertEqual(report.total_reproduction_cost, 0.0)
        self.assertEqual(report.avg_human_percentage, 0.0)
        self.assertEqual(report.datasets, [])
        self.assertEqual(report.insights, [])
        self.assertEqual(report.trends, [])

    def test_mutable_defaults_are_independent(self):
        """Ensure mutable default fields are independent between instances."""
        report1 = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report2 = WeeklyReport(period_start="2025-01-08", period_end="2025-01-14")
        report1.discoveries_by_org["org1"] = 5
        report1.datasets.append(DatasetEntry(dataset_id="ds1"))
        self.assertEqual(report2.discoveries_by_org, {})
        self.assertEqual(report2.datasets, [])


# ==================== IntegratedReportGenerator.__init__ ====================


class TestIntegratedReportGeneratorInit(unittest.TestCase):
    """Test IntegratedReportGenerator initialization."""

    def test_default_init(self):
        gen = IntegratedReportGenerator()
        self.assertIsNone(gen.radar_reports_dir)
        self.assertEqual(gen.recipe_output_dir, "./projects")

    def test_custom_init(self):
        gen = IntegratedReportGenerator(
            radar_reports_dir="/path/to/radar",
            recipe_output_dir="/path/to/recipes",
        )
        self.assertEqual(gen.radar_reports_dir, "/path/to/radar")
        self.assertEqual(gen.recipe_output_dir, "/path/to/recipes")


# ==================== IntegratedReportGenerator.load_radar_report ====================


class TestLoadRadarReport(unittest.TestCase):
    """Test IntegratedReportGenerator.load_radar_report()."""

    def setUp(self):
        self.gen = IntegratedReportGenerator()

    def test_load_valid_report(self, tmp_path=None):
        """Test loading a valid JSON radar report."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"datasets": [{"id": "test/ds"}]}, f)
            path = f.name

        try:
            result = self.gen.load_radar_report(path)
            self.assertEqual(result, {"datasets": [{"id": "test/ds"}]})
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.gen.load_radar_report("/nonexistent/path/report.json")

    def test_load_invalid_json_raises(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("not valid json{{{")
            path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                self.gen.load_radar_report(path)
        finally:
            os.unlink(path)


# ==================== IntegratedReportGenerator.load_recipe_summary ====================


class TestLoadRecipeSummary(unittest.TestCase):
    """Test IntegratedReportGenerator.load_recipe_summary()."""

    def test_load_existing_summary(self, tmp_path=None):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)

        # Create the expected directory structure: {recipe_output_dir}/{safe_name}/recipe_summary.json
        safe_name = "org_dataset"
        os.makedirs(os.path.join(tmpdir, safe_name))
        summary_data = {"dataset_type": "preference", "sample_count": 1000}
        with open(
            os.path.join(tmpdir, safe_name, "recipe_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary_data, f)

        result = gen.load_recipe_summary("org/dataset")
        self.assertEqual(result, summary_data)

        import shutil

        shutil.rmtree(tmpdir)

    def test_load_nonexistent_summary_returns_none(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)

        result = gen.load_recipe_summary("nonexistent/dataset")
        self.assertIsNone(result)

        import shutil

        shutil.rmtree(tmpdir)

    def test_safe_name_replaces_slashes(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)

        # Create directory with forward slash replaced
        safe_name = "org_dataset"
        os.makedirs(os.path.join(tmpdir, safe_name))
        with open(
            os.path.join(tmpdir, safe_name, "recipe_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"found": True}, f)

        result = gen.load_recipe_summary("org/dataset")
        self.assertIsNotNone(result)
        self.assertTrue(result["found"])

        import shutil

        shutil.rmtree(tmpdir)

    def test_safe_name_replaces_backslashes(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)

        safe_name = "org_dataset"
        os.makedirs(os.path.join(tmpdir, safe_name))
        with open(
            os.path.join(tmpdir, safe_name, "recipe_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"found": True}, f)

        result = gen.load_recipe_summary("org\\dataset")
        self.assertIsNotNone(result)

        import shutil

        shutil.rmtree(tmpdir)


# ==================== IntegratedReportGenerator.generate_weekly_report ====================


class TestGenerateWeeklyReportBasic(unittest.TestCase):
    """Test generate_weekly_report() basic behavior."""

    def test_no_args_returns_weekly_report(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)
        report = gen.generate_weekly_report()
        self.assertIsInstance(report, WeeklyReport)

        import shutil

        shutil.rmtree(tmpdir)

    def test_default_date_range(self):
        """When no dates provided, defaults to last 7 days."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)

        now = datetime.now()
        report = gen.generate_weekly_report()

        expected_end = now.strftime("%Y-%m-%d")
        expected_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        self.assertEqual(report.period_end, expected_end)
        self.assertEqual(report.period_start, expected_start)

        import shutil

        shutil.rmtree(tmpdir)

    def test_custom_date_range(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)

        report = gen.generate_weekly_report(
            start_date="2025-01-01", end_date="2025-01-07"
        )
        self.assertEqual(report.period_start, "2025-01-01")
        self.assertEqual(report.period_end, "2025-01-07")

        import shutil

        shutil.rmtree(tmpdir)

    def test_generated_at_set(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        gen = IntegratedReportGenerator(recipe_output_dir=tmpdir)
        report = gen.generate_weekly_report()
        self.assertNotEqual(report.generated_at, "")
        # Should be an ISO format datetime string
        self.assertIn("T", report.generated_at)

        import shutil

        shutil.rmtree(tmpdir)


class TestGenerateWeeklyReportWithRadar(unittest.TestCase):
    """Test generate_weekly_report() with radar data."""

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.recipe_dir = os.path.join(self.tmpdir, "recipes")
        os.makedirs(self.recipe_dir)
        self.gen = IntegratedReportGenerator(recipe_output_dir=self.recipe_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def _write_radar_report(self, datasets):
        path = os.path.join(self.tmpdir, "radar_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"datasets": datasets}, f)
        return path

    def _write_recipe_summary(self, dataset_id, summary_data):
        safe_name = dataset_id.replace("/", "_").replace("\\", "_")
        ds_dir = os.path.join(self.recipe_dir, safe_name)
        os.makedirs(ds_dir, exist_ok=True)
        with open(os.path.join(ds_dir, "recipe_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f)

    def test_radar_datasets_loaded(self):
        radar_path = self._write_radar_report(
            [
                {"id": "org1/ds1", "category": "preference", "downloads": 5000},
                {"id": "org2/ds2", "category": "sft", "downloads": 3000},
            ]
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.total_discovered, 2)
        self.assertEqual(len(report.datasets), 2)

    def test_org_extraction(self):
        radar_path = self._write_radar_report(
            [
                {"id": "Anthropic/ds1", "downloads": 1000},
                {"id": "Anthropic/ds2", "downloads": 2000},
                {"id": "OpenAI/ds3", "downloads": 3000},
            ]
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.discoveries_by_org, {"Anthropic": 2, "OpenAI": 1})

    def test_org_extraction_no_slash(self):
        radar_path = self._write_radar_report(
            [{"id": "standalone-dataset", "downloads": 1000}]
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.discoveries_by_org, {"": 1})

    def test_category_aggregation(self):
        radar_path = self._write_radar_report(
            [
                {"id": "org/ds1", "category": "preference", "downloads": 100},
                {"id": "org/ds2", "category": "sft", "downloads": 200},
                {"id": "org/ds3", "category": "preference", "downloads": 300},
            ]
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.discoveries_by_category, {"preference": 2, "sft": 1})

    def test_empty_category_not_counted(self):
        radar_path = self._write_radar_report(
            [{"id": "org/ds1", "downloads": 100}]
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.discoveries_by_category, {})

    def test_datasets_sorted_by_downloads_descending(self):
        radar_path = self._write_radar_report(
            [
                {"id": "org/ds1", "downloads": 100},
                {"id": "org/ds2", "downloads": 5000},
                {"id": "org/ds3", "downloads": 1000},
            ]
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        downloads = [d.downloads for d in report.datasets]
        self.assertEqual(downloads, [5000, 1000, 100])

    def test_recipe_analysis_merged(self):
        """Test that recipe summaries are merged into radar datasets."""
        radar_path = self._write_radar_report(
            [{"id": "org/analyzed-ds", "category": "preference", "downloads": 5000}]
        )
        self._write_recipe_summary(
            "org/analyzed-ds",
            {
                "dataset_type": "preference",
                "reproduction_cost": {"total": 15000},
                "human_percentage": 70.0,
                "difficulty": "medium",
                "sample_count": 10000,
                "guide_path": "/path/to/guide.md",
                "report_path": "/path/to/report.md",
            },
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.total_analyzed, 1)
        self.assertEqual(report.total_reproduction_cost, 15000.0)

        entry = report.datasets[0]
        self.assertTrue(entry.analyzed)
        self.assertEqual(entry.dataset_type, "preference")
        self.assertEqual(entry.reproduction_cost, 15000.0)
        self.assertEqual(entry.human_percentage, 70.0)
        self.assertEqual(entry.difficulty, "medium")
        self.assertEqual(entry.sample_count, 10000)
        self.assertEqual(entry.guide_path, "/path/to/guide.md")
        self.assertEqual(entry.report_path, "/path/to/report.md")

    def test_analysis_by_type_aggregation(self):
        radar_path = self._write_radar_report(
            [
                {"id": "org/ds1", "downloads": 100},
                {"id": "org/ds2", "downloads": 200},
                {"id": "org/ds3", "downloads": 300},
            ]
        )
        self._write_recipe_summary("org/ds1", {"dataset_type": "preference", "reproduction_cost": {"total": 1000}})
        self._write_recipe_summary("org/ds2", {"dataset_type": "sft", "reproduction_cost": {"total": 2000}})
        self._write_recipe_summary("org/ds3", {"dataset_type": "preference", "reproduction_cost": {"total": 3000}})

        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertEqual(report.analysis_by_type, {"preference": 2, "sft": 1})
        self.assertEqual(report.total_reproduction_cost, 6000.0)

    def test_avg_human_percentage(self):
        radar_path = self._write_radar_report(
            [
                {"id": "org/ds1", "downloads": 100},
                {"id": "org/ds2", "downloads": 200},
            ]
        )
        self._write_recipe_summary(
            "org/ds1",
            {"human_percentage": 60.0, "reproduction_cost": {"total": 1000}},
        )
        self._write_recipe_summary(
            "org/ds2",
            {"human_percentage": 80.0, "reproduction_cost": {"total": 2000}},
        )
        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        self.assertAlmostEqual(report.avg_human_percentage, 70.0)

    def test_nonexistent_radar_path_skipped(self):
        """When radar_report_path does not exist, it is silently skipped."""
        report = self.gen.generate_weekly_report(
            radar_report_path="/nonexistent/radar_report.json"
        )
        self.assertEqual(report.total_discovered, 0)
        self.assertEqual(report.datasets, [])


class TestGenerateWeeklyReportRecipeScan(unittest.TestCase):
    """Test generate_weekly_report() scanning recipe_output_dir for extra analyses."""

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.recipe_dir = os.path.join(self.tmpdir, "recipes")
        os.makedirs(self.recipe_dir)
        self.gen = IntegratedReportGenerator(recipe_output_dir=self.recipe_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def _write_recipe_summary(self, dir_name, summary_data):
        ds_dir = os.path.join(self.recipe_dir, dir_name)
        os.makedirs(ds_dir, exist_ok=True)
        with open(os.path.join(ds_dir, "recipe_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f)

    def test_recipe_dir_scanned_for_extra_analyses(self):
        """Analyses in recipe_output_dir are added even without radar report."""
        self._write_recipe_summary(
            "org_extra-ds",
            {
                "dataset_id": "org/extra-ds",
                "dataset_type": "sft",
                "reproduction_cost": {"total": 5000},
                "human_percentage": 50.0,
                "difficulty": "easy",
                "sample_count": 2000,
            },
        )

        report = self.gen.generate_weekly_report()
        self.assertEqual(report.total_analyzed, 1)
        self.assertEqual(len(report.datasets), 1)
        self.assertEqual(report.datasets[0].dataset_id, "org/extra-ds")
        self.assertTrue(report.datasets[0].analyzed)

    def test_no_duplicate_from_radar_and_scan(self):
        """Datasets already loaded from radar should not be duplicated during scan."""
        radar_path = os.path.join(self.tmpdir, "radar.json")
        with open(radar_path, "w", encoding="utf-8") as f:
            json.dump({"datasets": [{"id": "org/ds1", "downloads": 100}]}, f)

        self._write_recipe_summary(
            "org_ds1",
            {
                "dataset_id": "org/ds1",
                "dataset_type": "preference",
                "reproduction_cost": {"total": 1000},
                "human_percentage": 60.0,
            },
        )

        report = self.gen.generate_weekly_report(radar_report_path=radar_path)
        # Should have only 1 dataset, not 2
        ids = [d.dataset_id for d in report.datasets]
        self.assertEqual(ids.count("org/ds1"), 1)

    def test_fallback_dataset_id_from_dirname(self):
        """When recipe_summary.json has no dataset_id, it falls back to dir name."""
        self._write_recipe_summary(
            "org_fallback-ds",
            {
                "dataset_type": "sft",
                "reproduction_cost": {"total": 3000},
            },
        )

        report = self.gen.generate_weekly_report()
        self.assertEqual(report.total_analyzed, 1)
        # Falls back: name.replace("_", "/", 1) → "org/fallback-ds"
        self.assertEqual(report.datasets[0].dataset_id, "org/fallback-ds")

    def test_invalid_recipe_summary_skipped(self):
        """Malformed recipe_summary.json files are silently skipped."""
        ds_dir = os.path.join(self.recipe_dir, "bad_ds")
        os.makedirs(ds_dir)
        with open(os.path.join(ds_dir, "recipe_summary.json"), "w", encoding="utf-8") as f:
            f.write("not valid json{{{")

        report = self.gen.generate_weekly_report()
        self.assertEqual(report.total_analyzed, 0)
        self.assertEqual(report.datasets, [])

    def test_nonexistent_recipe_dir_handled(self):
        """When recipe_output_dir does not exist, scan is skipped gracefully."""
        gen = IntegratedReportGenerator(recipe_output_dir="/nonexistent/dir")
        report = gen.generate_weekly_report()
        self.assertEqual(report.total_analyzed, 0)

    def test_recipe_dir_scanned_aggregates(self):
        """Verify analysis_by_type and total_reproduction_cost from scanned recipes."""
        self._write_recipe_summary(
            "org_ds1",
            {
                "dataset_id": "org/ds1",
                "dataset_type": "preference",
                "reproduction_cost": {"total": 1000},
                "human_percentage": 40.0,
            },
        )
        self._write_recipe_summary(
            "org_ds2",
            {
                "dataset_id": "org/ds2",
                "dataset_type": "preference",
                "reproduction_cost": {"total": 2000},
                "human_percentage": 60.0,
            },
        )

        report = self.gen.generate_weekly_report()
        self.assertEqual(report.total_analyzed, 2)
        self.assertEqual(report.analysis_by_type, {"preference": 2})
        self.assertEqual(report.total_reproduction_cost, 3000.0)
        self.assertAlmostEqual(report.avg_human_percentage, 50.0)

    def test_empty_dataset_type_not_counted_in_analysis_by_type(self):
        """Empty dataset_type should not appear in analysis_by_type."""
        self._write_recipe_summary(
            "org_ds1",
            {
                "dataset_id": "org/ds1",
                "dataset_type": "",
                "reproduction_cost": {"total": 500},
            },
        )
        report = self.gen.generate_weekly_report()
        self.assertEqual(report.analysis_by_type, {})


# ==================== IntegratedReportGenerator._generate_insights ====================


class TestGenerateInsights(unittest.TestCase):
    """Test _generate_insights() logic."""

    def setUp(self):
        self.gen = IntegratedReportGenerator()

    def test_empty_report_no_insights(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        insights = self.gen._generate_insights(report)
        self.assertEqual(insights, [])

    def test_top_org_insight(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.discoveries_by_org = {"Anthropic": 5, "OpenAI": 3}
        insights = self.gen._generate_insights(report)
        org_insight = [i for i in insights if "最活跃组织" in i]
        self.assertEqual(len(org_insight), 1)
        self.assertIn("Anthropic", org_insight[0])
        self.assertIn("5", org_insight[0])

    def test_top_category_insight(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.discoveries_by_category = {"preference": 10, "sft": 3}
        insights = self.gen._generate_insights(report)
        cat_insight = [i for i in insights if "热门数据集类型" in i]
        self.assertEqual(len(cat_insight), 1)
        self.assertIn("preference", cat_insight[0])

    def test_analysis_coverage_insight(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.total_discovered = 10
        report.total_analyzed = 7
        insights = self.gen._generate_insights(report)
        coverage_insight = [i for i in insights if "覆盖率" in i]
        self.assertEqual(len(coverage_insight), 1)
        self.assertIn("70%", coverage_insight[0])
        self.assertIn("7/10", coverage_insight[0])

    def test_avg_cost_insight(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.total_analyzed = 4
        report.total_reproduction_cost = 40000.0
        insights = self.gen._generate_insights(report)
        cost_insight = [i for i in insights if "成本" in i]
        self.assertEqual(len(cost_insight), 1)
        self.assertIn("$10,000", cost_insight[0])

    def test_human_percentage_insight(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.avg_human_percentage = 65.0
        insights = self.gen._generate_insights(report)
        human_insight = [i for i in insights if "人工占比" in i]
        self.assertEqual(len(human_insight), 1)
        self.assertIn("65%", human_insight[0])

    def test_zero_human_percentage_no_insight(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.avg_human_percentage = 0.0
        insights = self.gen._generate_insights(report)
        human_insight = [i for i in insights if "人工占比" in i]
        self.assertEqual(len(human_insight), 0)

    def test_all_insights_present(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.discoveries_by_org = {"Org1": 5}
        report.discoveries_by_category = {"cat1": 3}
        report.total_discovered = 5
        report.total_analyzed = 3
        report.total_reproduction_cost = 30000.0
        report.avg_human_percentage = 50.0
        insights = self.gen._generate_insights(report)
        self.assertEqual(len(insights), 5)


# ==================== IntegratedReportGenerator._generate_trends ====================


class TestGenerateTrends(unittest.TestCase):
    """Test _generate_trends() logic."""

    def setUp(self):
        self.gen = IntegratedReportGenerator()

    def test_empty_report_no_trends(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        trends = self.gen._generate_trends(report)
        self.assertEqual(trends, [])

    def test_type_distribution_trends(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.analysis_by_type = {"preference": 6, "sft": 4}
        trends = self.gen._generate_trends(report)
        self.assertEqual(len(trends), 2)

    def test_trends_sorted_by_count_descending(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.analysis_by_type = {"sft": 2, "preference": 8}
        trends = self.gen._generate_trends(report)
        self.assertIn("preference", trends[0])
        self.assertIn("sft", trends[1])

    def test_trend_percentages(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.analysis_by_type = {"preference": 3, "sft": 7}
        trends = self.gen._generate_trends(report)
        # preference: 3/10 = 30%, sft: 7/10 = 70%
        sft_trend = [t for t in trends if "sft" in t][0]
        self.assertIn("70%", sft_trend)
        pref_trend = [t for t in trends if "preference" in t][0]
        self.assertIn("30%", pref_trend)

    def test_single_type_100_percent(self):
        report = WeeklyReport(period_start="2025-01-01", period_end="2025-01-07")
        report.analysis_by_type = {"preference": 5}
        trends = self.gen._generate_trends(report)
        self.assertEqual(len(trends), 1)
        self.assertIn("100%", trends[0])
        self.assertIn("5 个", trends[0])


# ==================== IntegratedReportGenerator.to_markdown ====================


class TestToMarkdown(unittest.TestCase):
    """Test to_markdown() output."""

    def setUp(self):
        self.gen = IntegratedReportGenerator()

    def _make_report(self, **kwargs):
        defaults = {
            "period_start": "2025-01-01",
            "period_end": "2025-01-07",
            "generated_at": "2025-01-07T12:00:00",
        }
        defaults.update(kwargs)
        return WeeklyReport(**defaults)

    def test_header(self):
        report = self._make_report()
        md = self.gen.to_markdown(report)
        self.assertIn("# AI 数据集周报", md)
        self.assertIn("2025-01-01 ~ 2025-01-07", md)
        self.assertIn("2025-01-07 12:00", md)
        self.assertIn("DataRecipe + ai-dataset-radar", md)

    def test_executive_summary_table(self):
        report = self._make_report(
            total_discovered=10,
            total_analyzed=7,
            total_reproduction_cost=50000.0,
            avg_human_percentage=65.0,
        )
        md = self.gen.to_markdown(report)
        self.assertIn("执行摘要", md)
        self.assertIn("| 发现数据集 | 10 |", md)
        self.assertIn("| 已分析 | 7 |", md)
        self.assertIn("$50,000", md)
        self.assertIn("65%", md)

    def test_insights_section(self):
        report = self._make_report(insights=["Insight A", "Insight B"])
        md = self.gen.to_markdown(report)
        self.assertIn("关键洞察", md)
        self.assertIn("- Insight A", md)
        self.assertIn("- Insight B", md)

    def test_no_insights_section_when_empty(self):
        report = self._make_report(insights=[])
        md = self.gen.to_markdown(report)
        self.assertNotIn("关键洞察", md)

    def test_org_distribution_table(self):
        report = self._make_report(
            discoveries_by_org={"Anthropic": 5, "OpenAI": 3, "Meta": 2}
        )
        md = self.gen.to_markdown(report)
        self.assertIn("组织分布", md)
        self.assertIn("Anthropic", md)
        self.assertIn("5", md)

    def test_org_distribution_sorted_descending(self):
        report = self._make_report(
            discoveries_by_org={"Small": 1, "Big": 10, "Medium": 5}
        )
        md = self.gen.to_markdown(report)
        big_pos = md.index("Big")
        medium_pos = md.index("Medium")
        small_pos = md.index("Small")
        self.assertLess(big_pos, medium_pos)
        self.assertLess(medium_pos, small_pos)

    def test_org_distribution_limited_to_10(self):
        orgs = {f"Org{i:02d}": 100 - i for i in range(15)}
        report = self._make_report(discoveries_by_org=orgs)
        md = self.gen.to_markdown(report)
        # Count data rows in the org table (lines starting with "| Org")
        org_rows = [line for line in md.split("\n") if line.startswith("| Org")]
        self.assertEqual(len(org_rows), 10)

    def test_no_org_distribution_when_empty(self):
        report = self._make_report(discoveries_by_org={})
        md = self.gen.to_markdown(report)
        self.assertNotIn("组织分布", md)

    def test_type_distribution_table(self):
        report = self._make_report(analysis_by_type={"preference": 6, "sft": 4})
        md = self.gen.to_markdown(report)
        self.assertIn("类型分布", md)
        self.assertIn("preference", md)
        self.assertIn("60%", md)

    def test_no_type_distribution_when_empty(self):
        report = self._make_report(analysis_by_type={})
        md = self.gen.to_markdown(report)
        self.assertNotIn("类型分布", md)

    def test_analyzed_datasets_table(self):
        entry = DatasetEntry(
            dataset_id="org/ds1",
            analyzed=True,
            dataset_type="preference",
            reproduction_cost=15000.0,
            human_percentage=70.0,
            difficulty="medium difficulty",
            guide_path="/path/to/guide.md",
        )
        report = self._make_report(datasets=[entry])
        md = self.gen.to_markdown(report)
        self.assertIn("已分析", md)
        self.assertIn("[org/ds1](/path/to/guide.md)", md)
        self.assertIn("$15,000", md)
        self.assertIn("70%", md)
        self.assertIn("medium", md)

    def test_difficulty_first_word_only(self):
        entry = DatasetEntry(
            dataset_id="org/ds1",
            analyzed=True,
            difficulty="hard because complex",
        )
        report = self._make_report(datasets=[entry])
        md = self.gen.to_markdown(report)
        self.assertIn("hard", md)
        # "because" should not be in the difficulty column
        # (hard to check precisely, but the difficulty column uses first word)

    def test_empty_difficulty_shows_dash(self):
        entry = DatasetEntry(dataset_id="org/ds1", analyzed=True, difficulty="")
        report = self._make_report(datasets=[entry])
        md = self.gen.to_markdown(report)
        # The row should contain a dash for difficulty
        self.assertIn("| - |", md)

    def test_analyzed_datasets_limited_to_20(self):
        entries = [
            DatasetEntry(dataset_id=f"org/ds{i}", analyzed=True) for i in range(25)
        ]
        report = self._make_report(datasets=entries)
        md = self.gen.to_markdown(report)
        self.assertIn("还有 5 个", md)

    def test_not_analyzed_datasets_table(self):
        entry = DatasetEntry(
            dataset_id="org/ds1",
            analyzed=False,
            org="org",
            category="preference",
            downloads=5000,
        )
        report = self._make_report(datasets=[entry])
        md = self.gen.to_markdown(report)
        self.assertIn("待分析", md)
        self.assertIn("org/ds1", md)
        self.assertIn("5,000", md)

    def test_not_analyzed_limited_to_10(self):
        entries = [
            DatasetEntry(dataset_id=f"org/ds{i}", analyzed=False) for i in range(15)
        ]
        report = self._make_report(datasets=entries)
        md = self.gen.to_markdown(report)
        self.assertIn("还有 5 个", md)

    def test_cost_analysis_section(self):
        entries = [
            DatasetEntry(
                dataset_id="org/ds1",
                analyzed=True,
                dataset_type="preference",
                reproduction_cost=10000.0,
            ),
            DatasetEntry(
                dataset_id="org/ds2",
                analyzed=True,
                dataset_type="preference",
                reproduction_cost=20000.0,
            ),
            DatasetEntry(
                dataset_id="org/ds3",
                analyzed=True,
                dataset_type="sft",
                reproduction_cost=5000.0,
            ),
        ]
        report = self._make_report(datasets=entries)
        md = self.gen.to_markdown(report)
        self.assertIn("成本分析", md)
        # preference avg = 15000, min = 10000, max = 20000
        self.assertIn("$15,000", md)
        self.assertIn("$10,000", md)
        self.assertIn("$20,000", md)

    def test_cost_analysis_unknown_type_label(self):
        entries = [
            DatasetEntry(
                dataset_id="org/ds1",
                analyzed=True,
                dataset_type="",
                reproduction_cost=5000.0,
            ),
        ]
        report = self._make_report(datasets=entries)
        md = self.gen.to_markdown(report)
        self.assertIn("unknown", md)

    def test_no_cost_section_when_no_analyzed(self):
        report = self._make_report(datasets=[])
        md = self.gen.to_markdown(report)
        # The cost analysis section should not have actual cost data
        # (it still has the --- separator)
        self.assertNotIn("成本分析", md)

    def test_footer(self):
        report = self._make_report()
        md = self.gen.to_markdown(report)
        self.assertIn("报告由 DataRecipe 自动生成", md)

    def test_mixed_analyzed_and_unanalyzed(self):
        entries = [
            DatasetEntry(dataset_id="org/a1", analyzed=True, dataset_type="sft", reproduction_cost=1000.0),
            DatasetEntry(dataset_id="org/u1", analyzed=False, org="org", downloads=500),
        ]
        report = self._make_report(datasets=entries)
        md = self.gen.to_markdown(report)
        self.assertIn("已分析", md)
        self.assertIn("待分析", md)

    def test_empty_category_in_unanalyzed_shows_dash(self):
        entry = DatasetEntry(dataset_id="org/ds", analyzed=False, category="")
        report = self._make_report(datasets=[entry])
        md = self.gen.to_markdown(report)
        self.assertIn("| - |", md)


# ==================== IntegratedReportGenerator.to_json ====================


class TestToJson(unittest.TestCase):
    """Test to_json() output."""

    def setUp(self):
        self.gen = IntegratedReportGenerator()

    def _make_report(self, **kwargs):
        defaults = {
            "period_start": "2025-01-01",
            "period_end": "2025-01-07",
            "generated_at": "2025-01-07T12:00:00",
        }
        defaults.update(kwargs)
        return WeeklyReport(**defaults)

    def test_valid_json(self):
        report = self._make_report()
        json_str = self.gen.to_json(report)
        data = json.loads(json_str)
        self.assertIsInstance(data, dict)

    def test_period_structure(self):
        report = self._make_report()
        data = json.loads(self.gen.to_json(report))
        self.assertEqual(data["period"]["start"], "2025-01-01")
        self.assertEqual(data["period"]["end"], "2025-01-07")

    def test_generated_at(self):
        report = self._make_report()
        data = json.loads(self.gen.to_json(report))
        self.assertEqual(data["generated_at"], "2025-01-07T12:00:00")

    def test_summary_section(self):
        report = self._make_report(
            total_discovered=10,
            total_analyzed=7,
            total_reproduction_cost=50000.0,
            avg_human_percentage=65.0,
        )
        data = json.loads(self.gen.to_json(report))
        self.assertEqual(data["summary"]["total_discovered"], 10)
        self.assertEqual(data["summary"]["total_analyzed"], 7)
        self.assertEqual(data["summary"]["total_reproduction_cost"], 50000.0)
        self.assertEqual(data["summary"]["avg_human_percentage"], 65.0)

    def test_distributions(self):
        report = self._make_report(
            discoveries_by_org={"Org1": 5},
            discoveries_by_category={"cat1": 3},
            analysis_by_type={"pref": 2},
        )
        data = json.loads(self.gen.to_json(report))
        self.assertEqual(data["distributions"]["by_org"], {"Org1": 5})
        self.assertEqual(data["distributions"]["by_category"], {"cat1": 3})
        self.assertEqual(data["distributions"]["by_type"], {"pref": 2})

    def test_insights_and_trends(self):
        report = self._make_report(
            insights=["Insight 1"],
            trends=["Trend 1"],
        )
        data = json.loads(self.gen.to_json(report))
        self.assertEqual(data["insights"], ["Insight 1"])
        self.assertEqual(data["trends"], ["Trend 1"])

    def test_datasets_serialization(self):
        entry = DatasetEntry(
            dataset_id="org/ds1",
            org="org",
            category="preference",
            downloads=5000,
            analyzed=True,
            dataset_type="preference",
            reproduction_cost=15000.0,
            human_percentage=70.0,
            difficulty="medium",
        )
        report = self._make_report(datasets=[entry])
        data = json.loads(self.gen.to_json(report))
        self.assertEqual(len(data["datasets"]), 1)
        ds = data["datasets"][0]
        self.assertEqual(ds["id"], "org/ds1")
        self.assertEqual(ds["org"], "org")
        self.assertEqual(ds["category"], "preference")
        self.assertEqual(ds["downloads"], 5000)
        self.assertTrue(ds["analyzed"])
        self.assertEqual(ds["type"], "preference")
        self.assertEqual(ds["cost"], 15000.0)
        self.assertEqual(ds["human_pct"], 70.0)
        self.assertEqual(ds["difficulty"], "medium")

    def test_ensure_ascii_false(self):
        """Unicode characters should be preserved, not escaped."""
        report = self._make_report(insights=["中文洞察"])
        json_str = self.gen.to_json(report)
        self.assertIn("中文洞察", json_str)
        self.assertNotIn("\\u", json_str)

    def test_indented_format(self):
        report = self._make_report()
        json_str = self.gen.to_json(report)
        self.assertIn("  ", json_str)  # 2-space indent


# ==================== IntegratedReportGenerator.save_report ====================


class TestSaveReport(unittest.TestCase):
    """Test save_report() file output."""

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.gen = IntegratedReportGenerator()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def _make_report(self, **kwargs):
        defaults = {
            "period_start": "2025-01-01",
            "period_end": "2025-01-07",
            "generated_at": "2025-01-07T12:00:00",
        }
        defaults.update(kwargs)
        return WeeklyReport(**defaults)

    def test_default_formats_md_and_json(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir)
        self.assertIn("md", paths)
        self.assertIn("json", paths)
        self.assertTrue(os.path.exists(paths["md"]))
        self.assertTrue(os.path.exists(paths["json"]))

    def test_filename_uses_period_end(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir)
        self.assertIn("weekly_report_20250107.md", paths["md"])
        self.assertIn("weekly_report_20250107.json", paths["json"])

    def test_md_only_format(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir, formats=["md"])
        self.assertIn("md", paths)
        self.assertNotIn("json", paths)

    def test_json_only_format(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir, formats=["json"])
        self.assertIn("json", paths)
        self.assertNotIn("md", paths)

    def test_md_file_content(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir, formats=["md"])
        with open(paths["md"], encoding="utf-8") as f:
            content = f.read()
        self.assertIn("# AI 数据集周报", content)

    def test_json_file_content(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir, formats=["json"])
        with open(paths["json"], encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["period"]["start"], "2025-01-01")

    def test_creates_output_dir_if_not_exists(self):
        nested_dir = os.path.join(self.tmpdir, "nested", "output")
        report = self._make_report()
        paths = self.gen.save_report(report, nested_dir)
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(paths["md"]))

    def test_empty_formats_list_treated_as_default(self):
        """Empty list is falsy, so `formats or [...]` falls back to defaults."""
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir, formats=[])
        # Empty list is falsy in Python, so defaults to ["md", "json"]
        self.assertIn("md", paths)
        self.assertIn("json", paths)

    def test_unknown_format_ignored(self):
        report = self._make_report()
        paths = self.gen.save_report(report, self.tmpdir, formats=["xml"])
        self.assertEqual(paths, {})


# ==================== Integration: generate + to_markdown/to_json ====================


class TestIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests combining generate, render, and save."""

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.recipe_dir = os.path.join(self.tmpdir, "recipes")
        os.makedirs(self.recipe_dir)
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.gen = IntegratedReportGenerator(recipe_output_dir=self.recipe_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_full_pipeline(self):
        """Test full pipeline: create data, generate report, render, save."""
        # Create radar report
        radar_path = os.path.join(self.tmpdir, "radar.json")
        with open(radar_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "datasets": [
                        {"id": "Anthropic/hh-rlhf", "category": "preference", "downloads": 50000},
                        {"id": "OpenAI/sft-data", "category": "sft", "downloads": 10000},
                        {"id": "Meta/unanalyzed", "category": "other", "downloads": 5000},
                    ]
                },
                f,
            )

        # Create recipe summaries for 2 of the 3 datasets
        for ds_id, ds_data in [
            (
                "Anthropic/hh-rlhf",
                {
                    "dataset_type": "preference",
                    "reproduction_cost": {"total": 20000},
                    "human_percentage": 70.0,
                    "difficulty": "hard",
                    "sample_count": 50000,
                    "guide_path": "/guides/hh-rlhf.md",
                    "report_path": "/reports/hh-rlhf.md",
                },
            ),
            (
                "OpenAI/sft-data",
                {
                    "dataset_type": "sft",
                    "reproduction_cost": {"total": 10000},
                    "human_percentage": 40.0,
                    "difficulty": "medium",
                    "sample_count": 20000,
                },
            ),
        ]:
            safe_name = ds_id.replace("/", "_")
            ds_dir = os.path.join(self.recipe_dir, safe_name)
            os.makedirs(ds_dir, exist_ok=True)
            with open(os.path.join(ds_dir, "recipe_summary.json"), "w", encoding="utf-8") as f:
                json.dump(ds_data, f)

        # Generate report
        report = self.gen.generate_weekly_report(
            radar_report_path=radar_path,
            start_date="2025-01-01",
            end_date="2025-01-07",
        )

        # Verify report contents
        self.assertEqual(report.total_discovered, 3)
        self.assertEqual(report.total_analyzed, 2)
        self.assertEqual(report.total_reproduction_cost, 30000.0)
        self.assertAlmostEqual(report.avg_human_percentage, 55.0)
        self.assertEqual(report.analysis_by_type, {"preference": 1, "sft": 1})
        self.assertEqual(len(report.insights), 5)  # All 5 insight types
        self.assertGreater(len(report.trends), 0)

        # Render markdown
        md = self.gen.to_markdown(report)
        self.assertIn("# AI 数据集周报", md)
        self.assertIn("Anthropic/hh-rlhf", md)
        self.assertIn("待分析", md)  # Meta/unanalyzed should be in "not analyzed"

        # Render JSON
        json_str = self.gen.to_json(report)
        data = json.loads(json_str)
        self.assertEqual(data["summary"]["total_discovered"], 3)

        # Save report
        paths = self.gen.save_report(report, self.output_dir)
        self.assertTrue(os.path.exists(paths["md"]))
        self.assertTrue(os.path.exists(paths["json"]))


if __name__ == "__main__":
    unittest.main()
