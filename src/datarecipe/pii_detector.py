"""PII (Personally Identifiable Information) detection for datasets."""

import re
from dataclasses import dataclass, field


# ==================== Dataclasses ====================


@dataclass
class PIIMatch:
    """A single PII match found in the data."""

    pii_type: str
    field: str
    value: str  # masked
    row_index: int
    confidence: str  # "high", "medium"

    def to_dict(self) -> dict:
        return {
            "pii_type": self.pii_type,
            "field": self.field,
            "value": self.value,
            "row_index": self.row_index,
            "confidence": self.confidence,
        }


@dataclass
class PIITypeSummary:
    """Summary of a single PII type across the dataset."""

    pii_type: str
    count: int
    affected_fields: list[str]
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pii_type": self.pii_type,
            "count": self.count,
            "affected_fields": self.affected_fields,
            "examples": self.examples,
        }


@dataclass
class PIIReport:
    """Complete PII detection report."""

    total_samples: int
    samples_with_pii: int
    pii_ratio: float
    type_summaries: list[PIITypeSummary] = field(default_factory=list)
    risk_level: str = "none"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "samples_with_pii": self.samples_with_pii,
            "pii_ratio": round(self.pii_ratio, 4),
            "risk_level": self.risk_level,
            "type_summaries": [s.to_dict() for s in self.type_summaries],
            "recommendations": self.recommendations,
        }


# ==================== Patterns ====================

# PII type → (compiled regex, confidence, word-boundary flag)
_PATTERNS: dict[str, tuple[str, str]] = {
    "email": (
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "high",
    ),
    "phone_cn": (
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "high",
    ),
    "phone_intl": (
        r"\+\d{1,3}[-.\s]?\d{4,14}",
        "medium",
    ),
    "id_card_cn": (
        r"(?<!\d)\d{17}[\dXx](?!\d)",
        "high",
    ),
    "credit_card": (
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "medium",
    ),
    "ip_address": (
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "medium",
    ),
    "url_with_credentials": (
        r"https?://[^:\s]+:[^@\s]+@",
        "high",
    ),
    "ssn_us": (
        r"\b\d{3}-\d{2}-\d{4}\b",
        "high",
    ),
}


def _luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ==================== Detector ====================


class PIIDetector:
    """Detect PII in dataset samples using regex patterns."""

    def __init__(self, pii_types: list[str] | None = None):
        """Initialize the detector.

        Args:
            pii_types: List of PII types to detect. None means detect all.
        """
        if pii_types:
            unknown = set(pii_types) - set(_PATTERNS.keys())
            if unknown:
                raise ValueError(
                    f"Unknown PII types: {', '.join(sorted(unknown))}. "
                    f"Available: {', '.join(sorted(_PATTERNS.keys()))}"
                )
            self.pii_types = list(pii_types)
        else:
            self.pii_types = list(_PATTERNS.keys())

        self._compiled = {
            t: re.compile(_PATTERNS[t][0]) for t in self.pii_types
        }

    def analyze_sample(
        self,
        data: list[dict],
        text_fields: list[str] | None = None,
    ) -> PIIReport:
        """Analyze a list of data samples for PII.

        Args:
            data: List of dict samples.
            text_fields: Fields to scan. None means scan all string fields.

        Returns:
            PIIReport with detection results.
        """
        if not data:
            return PIIReport(
                total_samples=0, samples_with_pii=0, pii_ratio=0.0,
                risk_level="none", recommendations=["No data to analyze"],
            )

        all_matches: list[PIIMatch] = []
        rows_with_pii: set[int] = set()

        for row_idx, sample in enumerate(data):
            texts = self._extract_texts(sample, text_fields)
            for field_name, text in texts:
                matches = self._scan_text(text, field_name, row_idx)
                if matches:
                    all_matches.extend(matches)
                    rows_with_pii.add(row_idx)

        total = len(data)
        pii_count = len(rows_with_pii)
        pii_ratio = pii_count / total if total > 0 else 0.0

        type_summaries = self._build_type_summaries(all_matches)
        risk_level = self._determine_risk_level(pii_ratio, len(type_summaries))
        recommendations = self._generate_recommendations(
            pii_ratio, type_summaries, risk_level,
        )

        return PIIReport(
            total_samples=total,
            samples_with_pii=pii_count,
            pii_ratio=pii_ratio,
            type_summaries=type_summaries,
            risk_level=risk_level,
            recommendations=recommendations,
        )

    def analyze_from_file(
        self,
        file_path: str,
        sample_size: int = 1000,
        text_fields: list[str] | None = None,
    ) -> PIIReport:
        """Analyze a local file for PII.

        Args:
            file_path: Path to CSV, Parquet, or JSONL file.
            sample_size: Number of samples to check.
            text_fields: Fields to scan. None means scan all string fields.
        """
        from pathlib import Path

        from datarecipe.sources.local import LocalFileExtractor, detect_format

        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        fmt = detect_format(p)
        extractor = LocalFileExtractor()
        samples = extractor._load_samples(p, fmt, max_samples=sample_size)
        return self.analyze_sample(samples, text_fields)

    def analyze_from_huggingface(
        self,
        dataset_id: str,
        sample_size: int = 1000,
        split: str = "train",
        text_fields: list[str] | None = None,
    ) -> PIIReport:
        """Analyze a HuggingFace dataset for PII.

        Args:
            dataset_id: HuggingFace dataset identifier.
            sample_size: Number of samples to check.
            split: Dataset split to use.
            text_fields: Fields to scan. None means scan all string fields.
        """
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split, streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            samples.append(item)
        return self.analyze_sample(samples, text_fields)

    # ==================== Internal methods ====================

    def _extract_texts(
        self,
        sample: dict,
        text_fields: list[str] | None,
    ) -> list[tuple[str, str]]:
        """Extract (field_name, text) pairs from a sample.

        If text_fields is None, extracts all string values recursively.
        """
        results = []
        if text_fields:
            for f in text_fields:
                val = sample.get(f)
                if isinstance(val, str):
                    results.append((f, val))
        else:
            self._extract_all_strings(sample, "", results)
        return results

    def _extract_all_strings(
        self,
        obj: object,
        prefix: str,
        results: list[tuple[str, str]],
    ) -> None:
        """Recursively extract all string values from nested structures."""
        if isinstance(obj, str):
            results.append((prefix or "value", obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                self._extract_all_strings(v, key, results)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                key = f"{prefix}[{i}]"
                self._extract_all_strings(item, key, results)

    def _scan_text(
        self, text: str, field_name: str, row_index: int,
    ) -> list[PIIMatch]:
        """Scan a single text for all active PII patterns."""
        matches = []
        for pii_type in self.pii_types:
            pattern = self._compiled[pii_type]
            confidence = _PATTERNS[pii_type][1]
            for m in pattern.finditer(text):
                raw = m.group()
                # Credit card: extra Luhn validation
                if pii_type == "credit_card" and not _luhn_check(raw):
                    continue
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    field=field_name,
                    value=self._mask_value(raw, pii_type),
                    row_index=row_index,
                    confidence=confidence,
                ))
        return matches

    def _mask_value(self, value: str, pii_type: str) -> str:
        """Mask a PII value for safe display."""
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0]
                return f"{local[0]}***@{parts[1]}" if local else f"***@{parts[1]}"
            return "***"
        if pii_type in ("phone_cn", "phone_intl"):
            digits = "".join(c for c in value if c.isdigit())
            if len(digits) >= 7:
                return digits[:3] + "****" + digits[-4:]
            return "***"
        if pii_type == "id_card_cn":
            return value[:6] + "********" + value[-4:]
        if pii_type == "credit_card":
            digits = "".join(c for c in value if c.isdigit())
            if len(digits) >= 8:
                return digits[:4] + " **** **** " + digits[-4:]
            return "****"
        if pii_type == "ssn_us":
            return "***-**-" + value[-4:]
        if pii_type == "ip_address":
            parts = value.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.*.*"
            return "***"
        if pii_type == "url_with_credentials":
            return re.sub(r"://[^:]+:[^@]+@", "://***:***@", value)
        return "***"

    def _build_type_summaries(
        self, matches: list[PIIMatch],
    ) -> list[PIITypeSummary]:
        """Group matches by PII type into summaries."""
        by_type: dict[str, list[PIIMatch]] = {}
        for m in matches:
            by_type.setdefault(m.pii_type, []).append(m)

        summaries = []
        for pii_type in sorted(by_type.keys()):
            type_matches = by_type[pii_type]
            fields = sorted({m.field for m in type_matches})
            examples = []
            seen = set()
            for m in type_matches:
                if m.value not in seen and len(examples) < 3:
                    examples.append(m.value)
                    seen.add(m.value)
            summaries.append(PIITypeSummary(
                pii_type=pii_type,
                count=len(type_matches),
                affected_fields=fields,
                examples=examples,
            ))
        return summaries

    def _determine_risk_level(self, pii_ratio: float, type_count: int) -> str:
        """Determine overall risk level from PII statistics."""
        if pii_ratio == 0:
            return "none"
        if pii_ratio > 0.1 or type_count >= 4:
            return "high"
        if pii_ratio > 0.01 or type_count >= 2:
            return "medium"
        return "low"

    def _generate_recommendations(
        self,
        pii_ratio: float,
        type_summaries: list[PIITypeSummary],
        risk_level: str,
    ) -> list[str]:
        """Generate actionable recommendations based on findings."""
        if risk_level == "none":
            return ["No PII detected in the sampled data"]

        recs = []

        if risk_level == "high":
            recs.append(
                "HIGH RISK: This dataset contains significant PII. "
                "Do not distribute without thorough anonymization."
            )
        elif risk_level == "medium":
            recs.append(
                "MEDIUM RISK: Some PII detected. Review and anonymize before sharing."
            )
        else:
            recs.append(
                "LOW RISK: Minor PII detected. Consider reviewing flagged items."
            )

        type_names = {s.pii_type for s in type_summaries}

        if "email" in type_names:
            recs.append("Replace email addresses with synthetic ones or hash them")
        if "phone_cn" in type_names or "phone_intl" in type_names:
            recs.append("Remove or mask phone numbers")
        if "id_card_cn" in type_names:
            recs.append("CRITICAL: Remove Chinese ID card numbers immediately")
        if "credit_card" in type_names:
            recs.append("CRITICAL: Remove credit card numbers immediately")
        if "ssn_us" in type_names:
            recs.append("CRITICAL: Remove US Social Security Numbers immediately")
        if "ip_address" in type_names:
            recs.append("Consider anonymizing IP addresses")
        if "url_with_credentials" in type_names:
            recs.append("Remove URLs containing embedded credentials")

        recs.append(
            f"PII found in {pii_ratio * 100:.1f}% of samples — "
            f"increase sample size to validate full dataset"
        )

        return recs
