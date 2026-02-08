"""File system watcher for automated analysis triggers."""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    """Configuration for analysis triggers."""

    # Filter settings
    orgs: list[str] = field(default_factory=list)  # Filter by organizations
    categories: list[str] = field(default_factory=list)  # Filter by categories
    min_downloads: int = 0  # Minimum download count

    # Analysis settings
    sample_size: int = 200
    use_llm: bool = False
    region: str = "china"

    # Limits
    max_datasets_per_report: int = 10  # Max datasets to analyze per report

    # Notification (placeholder for future)
    notify_on_complete: bool = False
    notify_webhook: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TriggerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "TriggerConfig":
        """Load config from YAML file."""
        try:
            import yaml

            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data.get("triggers", data))
        except ImportError:
            raise ImportError("Please install: pip install pyyaml")


class RadarWatcher:
    """Watch for new Radar reports and trigger analysis."""

    def __init__(
        self,
        watch_dir: str,
        output_dir: str = "./projects",
        config: TriggerConfig = None,
        callback: Callable[[str, dict], None] = None,
    ):
        """Initialize watcher.

        Args:
            watch_dir: Directory to watch for Radar reports
            output_dir: Output directory for analysis results
            config: Trigger configuration
            callback: Optional callback function(dataset_id, result)
        """
        self.watch_dir = Path(watch_dir)
        self.output_dir = output_dir
        self.config = config or TriggerConfig()
        self.callback = callback

        self.processed_files: dict[str, str] = {}  # file -> last_modified
        self.state_file = self.watch_dir / ".datarecipe_watcher_state.json"
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._load_state()

    def _load_state(self):
        """Load processed files state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.processed_files = json.load(f)
            except Exception:
                self.processed_files = {}

    def _save_state(self):
        """Save processed files state."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception:
            pass

    def _find_new_reports(self) -> list[Path]:
        """Find new or updated Radar reports."""
        new_reports = []

        if not self.watch_dir.exists():
            return []

        # Look for intel_report_*.json files
        for fpath in self.watch_dir.glob("intel_report_*.json"):
            mtime = str(fpath.stat().st_mtime)
            fname = str(fpath)

            if fname not in self.processed_files or self.processed_files[fname] != mtime:
                new_reports.append(fpath)

        return new_reports

    def _process_report(self, report_path: Path) -> dict:
        """Process a single Radar report.

        Args:
            report_path: Path to the report file

        Returns:
            Processing result dict
        """
        from datarecipe.integrations.radar import RadarIntegration

        result = {
            "report": str(report_path),
            "timestamp": datetime.now().isoformat(),
            "datasets_analyzed": 0,
            "datasets_failed": 0,
            "summaries": [],
        }

        try:
            # Load and filter datasets
            integration = RadarIntegration()
            datasets = integration.load_radar_report(str(report_path))

            # Apply filters
            datasets = integration.filter_datasets(
                orgs=self.config.orgs or None,
                categories=self.config.categories or None,
                min_downloads=self.config.min_downloads,
                limit=self.config.max_datasets_per_report,
            )

            if not datasets:
                result["message"] = "No datasets matched filters"
                return result

            # Analyze each dataset
            from datarecipe.cache import AnalysisCache

            cache = AnalysisCache()

            for ds in datasets:
                try:
                    # Check cache first
                    cached = cache.get(ds.id, check_freshness=True)
                    if cached:
                        result["summaries"].append(
                            {
                                "dataset_id": ds.id,
                                "status": "cached",
                                "type": cached.dataset_type,
                            }
                        )
                        result["datasets_analyzed"] += 1
                        continue

                    # Run analysis
                    analysis_result = self._analyze_dataset(ds.id)

                    if analysis_result.get("success"):
                        result["datasets_analyzed"] += 1
                        result["summaries"].append(
                            {
                                "dataset_id": ds.id,
                                "status": "analyzed",
                                "type": analysis_result.get("type", "unknown"),
                                "cost": analysis_result.get("cost", 0),
                            }
                        )

                        # Callback
                        if self.callback:
                            self.callback(ds.id, analysis_result)
                    else:
                        result["datasets_failed"] += 1
                        result["summaries"].append(
                            {
                                "dataset_id": ds.id,
                                "status": "failed",
                                "error": analysis_result.get("error", "Unknown error"),
                            }
                        )

                except Exception as e:
                    result["datasets_failed"] += 1
                    result["summaries"].append(
                        {
                            "dataset_id": ds.id,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

            # Mark as processed
            self.processed_files[str(report_path)] = str(report_path.stat().st_mtime)
            self._save_state()

        except Exception as e:
            result["error"] = str(e)

        return result

    def _analyze_dataset(self, dataset_id: str) -> dict:
        """Run analysis on a single dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Analysis result dict
        """

        try:
            from datasets import load_dataset

            from datarecipe.cache import AnalysisCache
            from datarecipe.generators import HumanMachineSplitter, TaskType
            from datarecipe.integrations.radar import RadarIntegration

            # Setup
            safe_name = dataset_id.replace("/", "_").replace("\\", "_")
            output_dir = os.path.join(self.output_dir, safe_name)
            os.makedirs(output_dir, exist_ok=True)

            # Load dataset
            try:
                ds = load_dataset(dataset_id, split="train", streaming=True)
            except ValueError:
                ds = load_dataset(dataset_id, split="test", streaming=True)

            # Collect samples
            schema_info = {}
            sample_items = []
            rubrics = []
            messages = []
            sample_count = 0

            for i, item in enumerate(ds):
                if i >= self.config.sample_size:
                    break
                sample_count = i + 1

                if i < 5:
                    for field, value in item.items():
                        if field not in schema_info:
                            schema_info[field] = {"type": type(value).__name__}
                    sample_items.append(item)

                for field in ["rubrics", "rubric", "criteria"]:
                    if field in item:
                        v = item[field]
                        if isinstance(v, list):
                            rubrics.extend(v)
                        elif isinstance(v, str):
                            rubrics.append(v)

                if "messages" in item:
                    messages.extend(item.get("messages", []))

            # Detect type
            is_preference = "chosen" in schema_info and "rejected" in schema_info
            is_swe = "repo" in schema_info and "patch" in schema_info

            dataset_type = ""
            if is_preference:
                dataset_type = "preference"
            elif is_swe:
                dataset_type = "swe_bench"
            elif rubrics:
                dataset_type = "evaluation"

            # Allocation
            splitter = HumanMachineSplitter(region=self.config.region)
            allocation = splitter.analyze(
                dataset_size=sample_count,
                task_types=[
                    TaskType.CONTEXT_CREATION,
                    TaskType.TASK_DESIGN,
                    TaskType.RUBRICS_WRITING,
                    TaskType.DATA_GENERATION,
                    TaskType.QUALITY_REVIEW,
                ],
            )

            # Create summary
            summary = RadarIntegration.create_summary(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                allocation=allocation,
                schema_info=schema_info,
                sample_count=sample_count,
                output_dir=output_dir,
            )
            RadarIntegration.save_summary(summary, output_dir)

            # Update cache
            cache = AnalysisCache()
            cache.put(
                dataset_id=dataset_id,
                output_dir=output_dir,
                dataset_type=dataset_type,
                sample_count=sample_count,
            )

            return {
                "success": True,
                "type": dataset_type,
                "cost": allocation.total_cost,
                "sample_count": sample_count,
                "output_dir": output_dir,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def check_once(self) -> list[dict]:
        """Check for new reports once and process them.

        Returns:
            List of processing results
        """
        results = []
        new_reports = self._find_new_reports()

        for report in new_reports:
            result = self._process_report(report)
            results.append(result)

        return results

    def watch(self, interval: int = 60, max_iterations: int = 0):
        """Start watching for new reports.

        Args:
            interval: Check interval in seconds
            max_iterations: Maximum iterations (0 = infinite)
        """
        self._running = True
        iterations = 0

        logger.info(f"Watching {self.watch_dir} for new Radar reports...")
        logger.info(f"Check interval: {interval}s")
        logger.info("Press Ctrl+C to stop")

        while self._running:
            try:
                results = self.check_once()

                for r in results:
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Processed: {r['report']}")
                    logger.info(f"  Analyzed: {r['datasets_analyzed']}, Failed: {r['datasets_failed']}")

                if not results:
                    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] No new reports")

                iterations += 1
                if max_iterations > 0 and iterations >= max_iterations:
                    break

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Stopping watcher...")
                break

        self._running = False

    def start_background(self, interval: int = 60):
        """Start watching in a background thread.

        Args:
            interval: Check interval in seconds
        """
        if self._thread and self._thread.is_alive():
            return

        self._running = True
        self._thread = threading.Thread(
            target=self.watch,
            args=(interval,),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
