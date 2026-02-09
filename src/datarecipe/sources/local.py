"""Extract recipe information from local data files (CSV, Parquet, JSONL)."""

import os
from pathlib import Path

from datarecipe.schema import (
    GenerationType,
    Recipe,
    SourceType,
)

# Supported file extensions and their datasets library format names
FORMAT_MAP = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".jsonl": "json",
    ".json": "json",
}

SUPPORTED_EXTENSIONS = set(FORMAT_MAP.keys())

# Field patterns for dataset type detection
_PREFERENCE_FIELDS = {"chosen", "rejected"}
_CONVERSATION_FIELDS = {"messages", "conversations", "dialogue"}
_INSTRUCTION_FIELDS = {"instruction", "input", "output"}
_QA_FIELDS = {"question", "answer"}
_SWE_FIELDS = {"repo", "patch", "instance_id"}


def detect_format(path: Path) -> str:
    """Detect the datasets library format name from file extension.

    Args:
        path: Path to the data file.

    Returns:
        Format string for datasets.load_dataset() (e.g. "csv", "json", "parquet").

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
    fmt = FORMAT_MAP.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return fmt


class LocalFileExtractor:
    """Extract dataset recipe information from local CSV/Parquet/JSONL files."""

    def extract(self, file_path: str) -> Recipe:
        """Extract recipe information from a local data file.

        Args:
            file_path: Absolute or relative path to a CSV, Parquet, or JSONL file.

        Returns:
            Recipe object with inferred metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        fmt = detect_format(path)

        # Load a small sample to infer schema
        samples = self._load_samples(path, fmt, max_samples=100)

        # Infer metadata
        schema = self._infer_schema(samples)
        dataset_type = self._detect_dataset_type(samples, schema)
        file_size = path.stat().st_size
        row_estimate = self._estimate_row_count(path, fmt, samples, file_size)

        # Build recipe
        recipe = Recipe(
            name=path.stem,
            source_type=SourceType.LOCAL,
            source_id=str(path.resolve()),
            description=self._build_description(path, schema, dataset_type, row_estimate),
            tags=self._build_tags(schema, dataset_type, fmt),
        )

        recipe.size = file_size
        recipe.num_examples = row_estimate

        # Detect generation type heuristics
        recipe.generation_type, recipe.synthetic_ratio, recipe.human_ratio = (
            self._detect_generation_type(samples, schema, dataset_type)
        )

        return recipe

    def _load_samples(self, path: Path, fmt: str, max_samples: int = 100) -> list[dict]:
        """Load a small sample of rows from the file."""
        try:
            from datasets import load_dataset

            ds = load_dataset(
                fmt, data_files=str(path), split="train", streaming=True
            )
            samples = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                samples.append(item)
            return samples
        except Exception:
            # Fallback: try basic Python loading
            return self._fallback_load(path, fmt, max_samples)

    def _fallback_load(self, path: Path, fmt: str, max_samples: int) -> list[dict]:
        """Fallback loader without datasets library."""
        import csv
        import json

        samples = []
        if fmt == "csv":
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= max_samples:
                        break
                    samples.append(dict(row))
        elif fmt == "json":
            with open(path, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return samples

    def _infer_schema(self, samples: list[dict]) -> dict[str, str]:
        """Infer field names and types from samples."""
        schema = {}
        for sample in samples[:10]:
            for field, value in sample.items():
                if field not in schema:
                    schema[field] = type(value).__name__
        return schema

    def _detect_dataset_type(self, samples: list[dict], schema: dict[str, str]) -> str:
        """Detect the dataset type from field names and content."""
        fields = set(schema.keys())
        fields_lower = {f.lower() for f in fields}

        if _SWE_FIELDS.issubset(fields_lower):
            return "swe_bench"
        if _PREFERENCE_FIELDS.issubset(fields_lower):
            return "preference"
        if fields_lower & _CONVERSATION_FIELDS:
            return "conversation"
        if fields_lower & _INSTRUCTION_FIELDS:
            return "instruction_tuning"
        if fields_lower & _QA_FIELDS:
            return "question_answering"

        # Check content patterns
        if samples:
            first = samples[0]
            for value in first.values():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict) and "role" in value[0]:
                        return "conversation"

        return "unknown"

    def _estimate_row_count(
        self, path: Path, fmt: str, samples: list[dict], file_size: int
    ) -> int:
        """Estimate total row count from file size and sample."""
        if not samples:
            return 0

        if fmt == "parquet":
            # For parquet, try loading metadata
            try:
                import pyarrow.parquet as pq

                meta = pq.read_metadata(str(path))
                return meta.num_rows
            except Exception:
                pass

        # Estimate from file size and average row size
        if fmt == "csv" or fmt == "json":
            # Count lines as a quick estimate
            try:
                with open(path, "rb") as f:
                    # Read first 64KB to estimate bytes per line
                    chunk = f.read(65536)
                    lines_in_chunk = chunk.count(b"\n")
                    if lines_in_chunk > 0:
                        bytes_per_line = len(chunk) / lines_in_chunk
                        return max(1, int(file_size / bytes_per_line))
            except Exception:
                pass

        return len(samples)

    def _detect_generation_type(
        self, samples: list[dict], schema: dict[str, str], dataset_type: str
    ) -> tuple[GenerationType, float | None, float | None]:
        """Heuristic detection of generation type from content."""
        if dataset_type == "preference":
            return GenerationType.MIXED, 0.5, 0.5

        # Check for known synthetic indicators in field values
        synthetic_indicators = 0
        human_indicators = 0
        checked = 0

        for sample in samples[:20]:
            for value in sample.values():
                if not isinstance(value, str):
                    continue
                text = value.lower()
                checked += 1
                if any(kw in text for kw in ("gpt", "claude", "llama", "generated by")):
                    synthetic_indicators += 1
                if any(kw in text for kw in ("annotated by", "labeled by", "human")):
                    human_indicators += 1

        if checked == 0:
            return GenerationType.UNKNOWN, None, None

        if synthetic_indicators > human_indicators and synthetic_indicators > 2:
            ratio = min(0.9, synthetic_indicators / checked)
            return GenerationType.SYNTHETIC, ratio, 1.0 - ratio
        if human_indicators > synthetic_indicators and human_indicators > 2:
            ratio = min(0.9, human_indicators / checked)
            return GenerationType.HUMAN, 1.0 - ratio, ratio

        return GenerationType.UNKNOWN, None, None

    def _build_description(
        self, path: Path, schema: dict[str, str], dataset_type: str, row_count: int
    ) -> str:
        """Generate a description from inferred metadata."""
        type_labels = {
            "preference": "偏好/RLHF",
            "conversation": "对话",
            "instruction_tuning": "指令微调",
            "question_answering": "问答",
            "swe_bench": "软件工程",
            "unknown": "通用",
        }
        type_label = type_labels.get(dataset_type, dataset_type)
        fields_str = ", ".join(list(schema.keys())[:8])
        size_mb = path.stat().st_size / (1024 * 1024)
        return (
            f"本地{type_label}数据集 ({path.suffix.lstrip('.')} 格式, "
            f"~{row_count:,} 行, {size_mb:.1f} MB). "
            f"字段: {fields_str}"
        )

    def _build_tags(self, schema: dict[str, str], dataset_type: str, fmt: str) -> list[str]:
        """Generate tags from metadata."""
        tags = ["local", fmt]
        if dataset_type != "unknown":
            tags.append(dataset_type.replace("_", "-"))
        return tags
