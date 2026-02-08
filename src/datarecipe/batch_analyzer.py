"""Batch analysis for multiple datasets."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.schema import Recipe


@dataclass
class BatchResult:
    """Result of analyzing a single dataset in a batch."""

    dataset_id: str
    success: bool
    recipe: Recipe | None = None
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "dataset_id": self.dataset_id,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 2),
        }
        if self.recipe:
            result["recipe"] = self.recipe.to_dict()
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis across multiple datasets."""

    results: list[BatchResult] = field(default_factory=list)
    successful: int = 0
    failed: int = 0
    total_duration_seconds: float = 0.0

    def get_recipes(self) -> list[Recipe]:
        """Get all successful recipes."""
        return [r.recipe for r in self.results if r.success and r.recipe]

    def get_failed(self) -> list[BatchResult]:
        """Get all failed results."""
        return [r for r in self.results if not r.success]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "summary": {
                "total": len(self.results),
                "successful": self.successful,
                "failed": self.failed,
                "total_duration_seconds": round(self.total_duration_seconds, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class BatchAnalyzer:
    """Analyzer for processing multiple datasets in parallel."""

    def __init__(
        self,
        max_workers: int = 4,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ):
        """Initialize the batch analyzer.

        Args:
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback for progress updates
                              Signature: (dataset_id, completed, total) -> None
        """
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.analyzer = DatasetAnalyzer()

    def analyze_batch(
        self,
        dataset_ids: list[str],
        continue_on_error: bool = True,
    ) -> BatchAnalysisResult:
        """Analyze multiple datasets in parallel.

        Args:
            dataset_ids: List of dataset IDs to analyze
            continue_on_error: Whether to continue if a dataset fails

        Returns:
            BatchAnalysisResult with all results
        """
        import time

        start_time = time.time()
        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._analyze_single, dataset_id): dataset_id
                for dataset_id in dataset_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_id):
                dataset_id = future_to_id[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if not continue_on_error:
                        raise
                    results.append(
                        BatchResult(
                            dataset_id=dataset_id,
                            success=False,
                            error=str(e),
                        )
                    )

                # Progress callback
                if self.progress_callback:
                    self.progress_callback(dataset_id, completed, len(dataset_ids))

        # Sort results by original order
        id_order = {id: i for i, id in enumerate(dataset_ids)}
        results.sort(key=lambda r: id_order.get(r.dataset_id, 999))

        total_duration = time.time() - start_time

        return BatchAnalysisResult(
            results=results,
            successful=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success),
            total_duration_seconds=total_duration,
        )

    def analyze_from_file(
        self,
        file_path: str,
        continue_on_error: bool = True,
    ) -> BatchAnalysisResult:
        """Analyze datasets listed in a file.

        Args:
            file_path: Path to file with dataset IDs (one per line)
            continue_on_error: Whether to continue if a dataset fails

        Returns:
            BatchAnalysisResult with all results
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        dataset_ids = []

        if path.suffix == ".json":
            # JSON file: expect a list of IDs or a dict with "datasets" key
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset_ids = data
                elif isinstance(data, dict) and "datasets" in data:
                    dataset_ids = data["datasets"]
                else:
                    raise ValueError("JSON file must contain a list or {datasets: [...]} object")

        elif path.suffix == ".csv":
            # CSV file: first column is dataset ID
            import csv

            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header if present

                # Check if first row looks like a header
                if header and not header[0].startswith(("http", "/")):
                    # Check if it's a valid HuggingFace ID
                    if "/" not in header[0] or any(
                        keyword in header[0].lower() for keyword in ["dataset", "id", "name", "url"]
                    ):
                        pass  # Skip header
                    else:
                        dataset_ids.append(header[0])

                for row in reader:
                    if row and row[0].strip():
                        dataset_ids.append(row[0].strip())

        else:
            # Plain text file: one ID per line
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        dataset_ids.append(line)

        if not dataset_ids:
            raise ValueError(f"No dataset IDs found in {file_path}")

        return self.analyze_batch(dataset_ids, continue_on_error)

    def _analyze_single(self, dataset_id: str) -> BatchResult:
        """Analyze a single dataset."""
        import time

        start_time = time.time()

        try:
            recipe = self.analyzer.analyze(dataset_id)
            duration = time.time() - start_time

            return BatchResult(
                dataset_id=dataset_id,
                success=True,
                recipe=recipe,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time

            return BatchResult(
                dataset_id=dataset_id,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    def export_results(
        self,
        result: BatchAnalysisResult,
        output_dir: str,
        format: str = "yaml",
    ) -> list[str]:
        """Export batch results to files.

        Args:
            result: The batch analysis result
            output_dir: Directory to write output files
            format: Output format ('yaml', 'json', 'markdown')

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Export individual recipes
        for batch_result in result.results:
            if not batch_result.success or not batch_result.recipe:
                continue

            recipe = batch_result.recipe
            safe_name = recipe.name.replace("/", "-").replace(" ", "-").lower()

            if format == "yaml":
                file_path = output_path / f"{safe_name}.yaml"
                file_path.write_text(recipe.to_yaml(), encoding="utf-8")
            elif format == "json":
                file_path = output_path / f"{safe_name}.json"
                file_path.write_text(
                    json.dumps(recipe.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif format == "markdown":
                from datarecipe.cli import recipe_to_markdown

                file_path = output_path / f"{safe_name}.md"
                file_path.write_text(recipe_to_markdown(recipe), encoding="utf-8")

            created_files.append(str(file_path))

        # Export summary
        summary_path = output_path / "batch_summary.json"
        summary_path.write_text(result.to_json(), encoding="utf-8")
        created_files.append(str(summary_path))

        return created_files
