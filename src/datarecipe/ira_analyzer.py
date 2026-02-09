"""Inter-Rater Agreement (IRA) analysis for annotation datasets."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations


# ==================== Dataclasses ====================


@dataclass
class PairwiseAgreement:
    """Agreement metrics between two annotators."""

    annotator_a: str
    annotator_b: str
    cohen_kappa: float
    percent_agreement: float
    n_items: int
    confusion_matrix: dict  # {(label_a, label_b): count}

    def to_dict(self) -> dict:
        return {
            "annotator_a": self.annotator_a,
            "annotator_b": self.annotator_b,
            "cohen_kappa": round(self.cohen_kappa, 4),
            "percent_agreement": round(self.percent_agreement, 4),
            "n_items": self.n_items,
            "confusion_matrix": {
                f"{k[0]}|{k[1]}": v for k, v in self.confusion_matrix.items()
            },
        }


@dataclass
class AnnotatorStats:
    """Per-annotator statistics."""

    annotator_id: str
    n_annotations: int
    label_distribution: dict[str, int]
    avg_kappa: float

    def to_dict(self) -> dict:
        return {
            "annotator_id": self.annotator_id,
            "n_annotations": self.n_annotations,
            "label_distribution": self.label_distribution,
            "avg_kappa": round(self.avg_kappa, 4),
        }


@dataclass
class DisagreementPattern:
    """A common disagreement between two labels."""

    label_a: str
    label_b: str
    count: int
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "count": self.count,
            "examples": self.examples,
        }


@dataclass
class IRAReport:
    """Complete inter-rater agreement report."""

    total_items: int
    total_annotations: int
    n_annotators: int
    labels: list[str]
    fleiss_kappa: float
    krippendorff_alpha: float
    avg_pairwise_kappa: float
    percent_agreement: float
    pairwise_agreements: list[PairwiseAgreement] = field(default_factory=list)
    annotator_stats: list[AnnotatorStats] = field(default_factory=list)
    disagreement_patterns: list[DisagreementPattern] = field(default_factory=list)
    quality_level: str = "poor"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "total_annotations": self.total_annotations,
            "n_annotators": self.n_annotators,
            "labels": self.labels,
            "fleiss_kappa": round(self.fleiss_kappa, 4),
            "krippendorff_alpha": round(self.krippendorff_alpha, 4),
            "avg_pairwise_kappa": round(self.avg_pairwise_kappa, 4),
            "percent_agreement": round(self.percent_agreement, 4),
            "pairwise_agreements": [p.to_dict() for p in self.pairwise_agreements],
            "annotator_stats": [a.to_dict() for a in self.annotator_stats],
            "disagreement_patterns": [d.to_dict() for d in self.disagreement_patterns],
            "quality_level": self.quality_level,
            "recommendations": self.recommendations,
        }


# ==================== Analyzer ====================


class IRAAnalyzer:
    """Analyze inter-rater agreement in annotation datasets."""

    def __init__(self, min_overlap: int = 2):
        """Initialize the analyzer.

        Args:
            min_overlap: Minimum annotators per item to include in analysis.
        """
        if min_overlap < 2:
            raise ValueError("min_overlap must be at least 2")
        self.min_overlap = min_overlap

    def analyze_sample(
        self,
        data: list[dict],
        item_field: str = "item_id",
        annotator_field: str = "annotator_id",
        label_field: str = "label",
        data_format: str = "auto",
    ) -> IRAReport:
        """Analyze a list of annotation samples for inter-rater agreement.

        Args:
            data: List of annotation records.
            item_field: Field name for item identifier (long format).
            annotator_field: Field name for annotator identifier (long format).
            label_field: Field name for the annotation label (long format).
            data_format: "auto", "long", or "wide".

        Returns:
            IRAReport with agreement metrics.
        """
        if not data:
            return IRAReport(
                total_items=0,
                total_annotations=0,
                n_annotators=0,
                labels=[],
                fleiss_kappa=0.0,
                krippendorff_alpha=0.0,
                avg_pairwise_kappa=0.0,
                percent_agreement=0.0,
                quality_level="poor",
                recommendations=["No data to analyze"],
            )

        # Detect and normalize to long format: list of (item_id, annotator_id, label)
        if data_format == "auto":
            data_format = self._detect_format(data, item_field, annotator_field)

        if data_format == "wide":
            annotations = self._wide_to_long(data, label_field)
        else:
            annotations = self._parse_long(data, item_field, annotator_field, label_field)

        if not annotations:
            return IRAReport(
                total_items=0,
                total_annotations=0,
                n_annotators=0,
                labels=[],
                fleiss_kappa=0.0,
                krippendorff_alpha=0.0,
                avg_pairwise_kappa=0.0,
                percent_agreement=0.0,
                quality_level="poor",
                recommendations=["No valid annotations found"],
            )

        # Group by item
        items: dict[str, dict[str, str]] = {}  # item_id → {annotator: label}
        for item_id, annotator_id, label in annotations:
            items.setdefault(item_id, {})[annotator_id] = label

        # Filter items with enough overlap
        items = {k: v for k, v in items.items() if len(v) >= self.min_overlap}

        if not items:
            return IRAReport(
                total_items=0,
                total_annotations=0,
                n_annotators=0,
                labels=[],
                fleiss_kappa=0.0,
                krippendorff_alpha=0.0,
                avg_pairwise_kappa=0.0,
                percent_agreement=0.0,
                quality_level="poor",
                recommendations=[
                    f"No items with at least {self.min_overlap} annotators"
                ],
            )

        all_annotators = sorted({a for anns in items.values() for a in anns})
        all_labels = sorted({l for anns in items.values() for l in anns.values()})
        total_annotations = sum(len(anns) for anns in items.values())

        # Pairwise agreements
        pairwise = self._compute_pairwise(items, all_annotators)

        # Fleiss' kappa
        fleiss = self._fleiss_kappa(items, all_labels)

        # Krippendorff's alpha
        alpha = self._krippendorff_alpha(annotations)

        # Average pairwise kappa
        kappas = [p.cohen_kappa for p in pairwise]
        avg_kappa = sum(kappas) / len(kappas) if kappas else 0.0

        # Overall percent agreement
        agreements = [p.percent_agreement for p in pairwise]
        avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0

        # Annotator stats
        ann_stats = self._compute_annotator_stats(items, pairwise, all_annotators)

        # Disagreement patterns
        disagreements = self._find_disagreement_patterns(items)

        quality = self._determine_quality_level(avg_kappa)
        recommendations = self._generate_recommendations(
            avg_kappa, quality, pairwise, disagreements, len(all_annotators),
        )

        return IRAReport(
            total_items=len(items),
            total_annotations=total_annotations,
            n_annotators=len(all_annotators),
            labels=all_labels,
            fleiss_kappa=fleiss,
            krippendorff_alpha=alpha,
            avg_pairwise_kappa=avg_kappa,
            percent_agreement=avg_agreement,
            pairwise_agreements=pairwise,
            annotator_stats=ann_stats,
            disagreement_patterns=disagreements,
            quality_level=quality,
            recommendations=recommendations,
        )

    def analyze_from_file(
        self,
        file_path: str,
        sample_size: int = 1000,
        **kwargs,
    ) -> IRAReport:
        """Analyze a local file for inter-rater agreement.

        Args:
            file_path: Path to CSV, Parquet, or JSONL file.
            sample_size: Number of rows to load.
            **kwargs: Passed to analyze_sample.
        """
        from pathlib import Path

        from datarecipe.sources.local import LocalFileExtractor, detect_format

        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        fmt = detect_format(p)
        extractor = LocalFileExtractor()
        samples = extractor._load_samples(p, fmt, max_samples=sample_size)
        return self.analyze_sample(samples, **kwargs)

    def analyze_from_huggingface(
        self,
        dataset_id: str,
        sample_size: int = 1000,
        split: str = "train",
        **kwargs,
    ) -> IRAReport:
        """Analyze a HuggingFace dataset for inter-rater agreement.

        Args:
            dataset_id: HuggingFace dataset identifier.
            sample_size: Number of samples to check.
            split: Dataset split to use.
            **kwargs: Passed to analyze_sample.
        """
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split, streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            samples.append(item)
        return self.analyze_sample(samples, **kwargs)

    # ==================== Format detection ====================

    def _detect_format(
        self, data: list[dict], item_field: str, annotator_field: str,
    ) -> str:
        """Detect whether data is in long or wide format."""
        if not data:
            return "long"
        first = data[0]
        if item_field in first and annotator_field in first:
            return "long"
        return "wide"

    def _parse_long(
        self,
        data: list[dict],
        item_field: str,
        annotator_field: str,
        label_field: str,
    ) -> list[tuple[str, str, str]]:
        """Parse long-format data into (item_id, annotator_id, label) tuples."""
        results = []
        for row in data:
            item_id = row.get(item_field)
            annotator_id = row.get(annotator_field)
            label = row.get(label_field)
            if item_id is not None and annotator_id is not None and label is not None:
                results.append((str(item_id), str(annotator_id), str(label)))
        return results

    def _wide_to_long(
        self, data: list[dict], label_field: str,
    ) -> list[tuple[str, str, str]]:
        """Convert wide-format data to (item_id, annotator_id, label) tuples.

        Wide format: each row is an item, columns are annotator names with label values.
        Non-annotator columns (like 'text', 'id') are skipped by checking if
        the column has values that look like labels (appear in multiple columns).
        """
        if not data:
            return []

        # Find annotator columns: columns with non-None values
        # Exclude common non-label fields
        _skip_fields = {
            "text", "id", "item_id", "content", "question", "context",
            "input", "output", "source", "metadata",
        }

        first = data[0]
        # Candidate annotator columns: string values, not in skip list
        annotator_cols = []
        for col in first:
            if col.lower() in _skip_fields:
                continue
            if col == label_field:
                continue
            val = first[col]
            if val is not None:
                annotator_cols.append(col)

        if len(annotator_cols) < 2:
            return []

        results = []
        for i, row in enumerate(data):
            item_id = str(row.get("item_id", row.get("id", i)))
            for col in annotator_cols:
                label = row.get(col)
                if label is not None:
                    results.append((item_id, col, str(label)))
        return results

    # ==================== Core calculations ====================

    def _cohen_kappa(self, labels_a: list[str], labels_b: list[str]) -> float:
        """Compute Cohen's Kappa for two annotators.

        κ = (po - pe) / (1 - pe)
        po = observed agreement
        pe = expected agreement by chance
        """
        if len(labels_a) != len(labels_b) or len(labels_a) == 0:
            return 0.0

        n = len(labels_a)
        all_labels = sorted(set(labels_a) | set(labels_b))

        # Observed agreement
        po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

        # Expected agreement
        pe = 0.0
        for label in all_labels:
            p_a = sum(1 for x in labels_a if x == label) / n
            p_b = sum(1 for x in labels_b if x == label) / n
            pe += p_a * p_b

        if pe == 1.0:
            return 1.0 if po == 1.0 else 0.0
        return (po - pe) / (1 - pe)

    def _fleiss_kappa(
        self, items: dict[str, dict[str, str]], all_labels: list[str],
    ) -> float:
        """Compute Fleiss' Kappa for multiple annotators.

        Uses the items dict {item_id: {annotator: label}}.
        """
        if not items or not all_labels:
            return 0.0

        n_items = len(items)
        n_labels = len(all_labels)
        label_idx = {l: i for i, l in enumerate(all_labels)}

        # Build matrix: n_items × n_labels, counts per label per item
        matrix = []
        for item_id in sorted(items.keys()):
            row = [0] * n_labels
            for label in items[item_id].values():
                if label in label_idx:
                    row[label_idx[label]] += 1
            matrix.append(row)

        # Number of raters per item
        n_per_item = [sum(row) for row in matrix]

        # P_i for each item
        p_items = []
        for i, row in enumerate(matrix):
            n_i = n_per_item[i]
            if n_i < 2:
                continue
            sum_sq = sum(r * r for r in row)
            p_i = (sum_sq - n_i) / (n_i * (n_i - 1))
            p_items.append(p_i)

        if not p_items:
            return 0.0

        p_bar = sum(p_items) / len(p_items)

        # P_e: expected agreement
        total_ratings = sum(n_per_item)
        if total_ratings == 0:
            return 0.0

        p_j = []
        for j in range(n_labels):
            col_sum = sum(matrix[i][j] for i in range(n_items))
            p_j.append(col_sum / total_ratings)

        pe = sum(p * p for p in p_j)

        if pe == 1.0:
            return 1.0 if p_bar == 1.0 else 0.0
        return (p_bar - pe) / (1 - pe)

    def _krippendorff_alpha(
        self, annotations: list[tuple[str, str, str]],
    ) -> float:
        """Compute Krippendorff's Alpha (nominal scale).

        Args:
            annotations: List of (item_id, annotator_id, label) tuples.
        """
        if not annotations:
            return 0.0

        # Group by item
        items: dict[str, list[str]] = {}
        for item_id, _, label in annotations:
            items.setdefault(item_id, []).append(label)

        # Filter items with 2+ annotations
        items = {k: v for k, v in items.items() if len(v) >= 2}
        if not items:
            return 0.0

        # Observed disagreement (Do)
        do_num = 0.0
        do_den = 0.0
        for labels in items.values():
            m = len(labels)
            if m < 2:
                continue
            counts = Counter(labels)
            # Number of disagreeing pairs
            for c_k in counts.values():
                do_num += c_k * (m - c_k)
            do_den += m * (m - 1)

        if do_den == 0:
            return 0.0
        do = do_num / do_den

        # Expected disagreement (De)
        total_labels = []
        for labels in items.values():
            total_labels.extend(labels)
        n_total = len(total_labels)
        if n_total < 2:
            return 0.0

        label_counts = Counter(total_labels)
        de = 0.0
        for c_k in label_counts.values():
            de += c_k * (n_total - c_k)
        de /= n_total * (n_total - 1)

        if de == 0:
            return 1.0 if do == 0 else 0.0
        return 1.0 - do / de

    # ==================== Pairwise computation ====================

    def _compute_pairwise(
        self,
        items: dict[str, dict[str, str]],
        all_annotators: list[str],
    ) -> list[PairwiseAgreement]:
        """Compute pairwise agreement for all annotator pairs."""
        results = []
        for ann_a, ann_b in combinations(all_annotators, 2):
            # Find items both annotated
            common_items = [
                item_id for item_id, anns in items.items()
                if ann_a in anns and ann_b in anns
            ]
            if not common_items:
                continue

            labels_a = [items[item_id][ann_a] for item_id in common_items]
            labels_b = [items[item_id][ann_b] for item_id in common_items]

            kappa = self._cohen_kappa(labels_a, labels_b)
            n = len(common_items)
            agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
            pct = agree / n if n > 0 else 0.0

            cm = self._build_confusion_matrix(labels_a, labels_b)

            results.append(PairwiseAgreement(
                annotator_a=ann_a,
                annotator_b=ann_b,
                cohen_kappa=kappa,
                percent_agreement=pct,
                n_items=n,
                confusion_matrix=cm,
            ))
        return results

    def _build_confusion_matrix(
        self, labels_a: list[str], labels_b: list[str],
    ) -> dict[tuple[str, str], int]:
        """Build confusion matrix as {(label_a, label_b): count}."""
        cm: dict[tuple[str, str], int] = {}
        for a, b in zip(labels_a, labels_b):
            key = (a, b)
            cm[key] = cm.get(key, 0) + 1
        return cm

    # ==================== Annotator stats ====================

    def _compute_annotator_stats(
        self,
        items: dict[str, dict[str, str]],
        pairwise: list[PairwiseAgreement],
        all_annotators: list[str],
    ) -> list[AnnotatorStats]:
        """Compute per-annotator statistics."""
        stats = []
        for ann in all_annotators:
            labels = [
                anns[ann] for anns in items.values() if ann in anns
            ]
            n = len(labels)
            dist = dict(Counter(labels))

            # Average kappa with other annotators
            kappas = []
            for p in pairwise:
                if p.annotator_a == ann or p.annotator_b == ann:
                    kappas.append(p.cohen_kappa)
            avg_k = sum(kappas) / len(kappas) if kappas else 0.0

            stats.append(AnnotatorStats(
                annotator_id=ann,
                n_annotations=n,
                label_distribution=dist,
                avg_kappa=avg_k,
            ))
        return stats

    # ==================== Disagreement patterns ====================

    def _find_disagreement_patterns(
        self, items: dict[str, dict[str, str]],
    ) -> list[DisagreementPattern]:
        """Find the most common label disagreement patterns."""
        pair_counts: dict[tuple[str, str], list[str]] = {}

        for item_id, anns in items.items():
            labels = list(anns.values())
            if len(set(labels)) <= 1:
                continue
            # All disagreeing label pairs
            for la, lb in combinations(sorted(set(labels)), 2):
                key = (la, lb)
                pair_counts.setdefault(key, []).append(str(item_id))

        patterns = []
        for (la, lb), example_ids in sorted(
            pair_counts.items(), key=lambda x: -len(x[1]),
        ):
            patterns.append(DisagreementPattern(
                label_a=la,
                label_b=lb,
                count=len(example_ids),
                examples=example_ids[:3],
            ))

        return patterns[:10]  # Top 10 patterns

    # ==================== Quality & recommendations ====================

    def _determine_quality_level(self, avg_kappa: float) -> str:
        """Determine agreement quality level from average kappa."""
        if avg_kappa >= 0.8:
            return "excellent"
        if avg_kappa >= 0.6:
            return "good"
        if avg_kappa >= 0.4:
            return "moderate"
        if avg_kappa >= 0.2:
            return "fair"
        return "poor"

    def _generate_recommendations(
        self,
        avg_kappa: float,
        quality: str,
        pairwise: list[PairwiseAgreement],
        disagreements: list[DisagreementPattern],
        n_annotators: int,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        if quality == "excellent":
            recs.append(
                "EXCELLENT agreement — annotations are highly reliable"
            )
        elif quality == "good":
            recs.append(
                "GOOD agreement — annotations are generally reliable, "
                "minor calibration may help"
            )
        elif quality == "moderate":
            recs.append(
                "MODERATE agreement — review annotation guidelines "
                "and provide additional training"
            )
        elif quality == "fair":
            recs.append(
                "FAIR agreement — significant inconsistencies detected, "
                "revise guidelines and retrain annotators"
            )
        else:
            recs.append(
                "POOR agreement — annotations are unreliable, "
                "consider redesigning the task or guidelines"
            )

        # Find weak annotator pairs
        weak_pairs = [
            p for p in pairwise if p.cohen_kappa < 0.4
        ]
        if weak_pairs:
            names = [
                f"{p.annotator_a}-{p.annotator_b}" for p in weak_pairs[:3]
            ]
            recs.append(
                f"Low agreement between: {', '.join(names)} — "
                f"consider calibration sessions"
            )

        # Disagreement-focused recommendation
        if disagreements:
            top = disagreements[0]
            recs.append(
                f"Most common confusion: '{top.label_a}' vs '{top.label_b}' "
                f"({top.count} items) — clarify boundary in guidelines"
            )

        if n_annotators < 3:
            recs.append(
                "Only 2 annotators — add a third for tie-breaking "
                "and more robust agreement metrics"
            )

        return recs
