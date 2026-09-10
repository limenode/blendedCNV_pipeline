"""Scoring call sets against the merged benchmark -- what `consensuscnv benchmark` does.

Writes one metrics row per call set and, on request, the TP / FP / FN rows
for each call set. Every figure in the paper is built from these tables by
the scripts under `manuscript/scripts/`.

Every consensus set in the grid is scored. The individual callers and the control
are scored too, since those are the comparators the paper puts beside the consensus sets.

Notes:

  * **Recall has a ceiling.** It is scored against the whole merged benchmark,
    which is larger than every query set, so a set's recall is bounded by how many
    calls it holds: `n_query / n_truth`, capped at one. Where the matching is
    one-to-one, recall divided by that ceiling is equal to the precision. The
    `recall_ceiling` column is reported alongside recall for that reason.
  * **`n_true_positive` and `n_truth_found` are different numbers.** Matching is
    many-to-many, so the count of query calls that matched something and the count
    of benchmark intervals that were matched only coincide when the matching is
    one-to-one. Both are reported, along with `one_to_one`.
"""

from __future__ import annotations

import csv
import glob
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from consensuscnv.callsets import collect_callsets, merge_components, read_bed_calls
from consensuscnv.callsets.registry import seed_chromosomes
from consensuscnv.classification.classify import Classification, classify, match_topology
from consensuscnv.classification.intervals import IntervalSet
from consensuscnv.classification.labels import write_labels
from consensuscnv.classification.pairs import build_candidates
from consensuscnv.consensus import iter_consensus_sets
from consensuscnv.output_layout import overlap_slug, slug
from consensuscnv.parsing.parser_utils import load_sample_list
from consensuscnv.utils import PipelineConfig

METRICS_FILE = "metrics.csv"


@dataclass(frozen=True)
class QuerySet:
    """One thing to score, and the labels that identify it in the metrics table."""

    call_set: str
    kind: str  # "consensus", "caller", or "control"
    name: str  # "2of3", a tool label, or the control's name
    intervals: IntervalSet
    reciprocal_overlap: float | None = None  # consensus sets only
    level: int | None = None
    n_sources: int | None = None

    @property
    def stem(self) -> str:
        """Filename stem for this set's label files, unique across the grid.

        Empty components are dropped: a control has no call set, and leaving the
        gap in would start the filename with a dot and hide the file.
        """
        parts = [slug(self.call_set), self.kind]
        if self.reciprocal_overlap is not None:
            parts.append(overlap_slug(self.reciprocal_overlap))
        parts.append(slug(self.name))
        return ".".join(part for part in parts if part)


@dataclass(frozen=True)
class MetricsRow:
    """One row of `evaluation/metrics.csv`. Field order is the column order."""

    call_set: str
    kind: str
    name: str
    reciprocal_overlap: float | None
    level: int | None
    n_sources: int | None
    n_query: int
    n_truth: int
    n_true_positive: int
    n_false_positive: int
    n_truth_found: int
    n_false_negative: int
    precision: float
    recall: float
    recall_ceiling: float
    f1: float
    one_to_one: bool
    labels: tuple[Path, ...] = field(default=(), repr=False)


def bed_paths(directory: Path, samples: frozenset[str] | None = None) -> list[str]:
    """Per-sample BEDs in a directory, restricted to an allowlist if given.

    Each BED is one sample and its stem is the sample id.
    Both sides of a comparison have to cover the same samples or precision and
    recall cannot be calculated.
    """
    paths = sorted(glob.glob(str(directory / "*.bed")))
    if samples is None:
        return paths
    return [path for path in paths if Path(path).stem in samples]


def load_truth(config: PipelineConfig, samples: frozenset[str] | None = None) -> IntervalSet:
    """The merged benchmark: every configured truth set pooled, then merged.

    Pooled before merging, so a variant that two sources both report becomes one
    interval rather than two. Merged on padding rather than overlap, which is what
    keeps the result internally disjoint within a sample, chromosome and variant
    class.
    """
    layout = config.layout
    paths = [
        bed
        for name in config.benchmark  # pinned to configured names, never a wildcard
        for bed in bed_paths(layout.benchmark_dir(name), samples)
    ]
    if not paths:
        raise ValueError(
            f"No parsed benchmark BEDs found under {layout.benchmark}. Run "
            "`consensuscnv benchmark` without --reuse-beds to parse them first."
        )

    merged = merge_components(
        collect_callsets(
            (read_bed_calls(path) for path in paths), chromosome_order=config.chromosomes
        ),
        max_padding=config.evaluation.benchmark_padding,
    )
    truth = IntervalSet.from_merged(merged)
    return truth.filter_by_size(min_size=config.consensus.min_size)


def iter_query_sets(
    config: PipelineConfig, samples: frozenset[str] | None = None
) -> Iterator[QuerySet]:
    """Everything to be scored: the consensus grid, the callers, and the controls."""
    floor = config.consensus.min_size
    layout = config.layout

    for consensus_set in iter_consensus_sets(config):
        yield QuerySet(
            call_set=consensus_set.call_set,
            kind="consensus",
            name=consensus_set.label,
            intervals=IntervalSet.from_merged(consensus_set.merged),
            reciprocal_overlap=consensus_set.reciprocal_overlap,
            level=consensus_set.level,
            n_sources=consensus_set.n_sources,
        )

    # Callers read raw from their own directories
    for call_set, tools in config.experimental.items():
        for tool in tools:
            paths = bed_paths(layout.bed_tool_dir(call_set, tool), samples)
            if not paths:
                continue
            intervals = IntervalSet.from_callset(
                collect_callsets(
                    (read_bed_calls(path) for path in paths),
                    chromosome_order=config.chromosomes,
                )
            )
            yield QuerySet(call_set, "caller", tool, intervals.filter_by_size(min_size=floor))

    for control in config.control:
        paths = bed_paths(layout.control_bed_dir(control), samples)
        if not paths:
            continue
        intervals = IntervalSet.from_callset(
            collect_callsets(
                (read_bed_calls(path) for path in paths), chromosome_order=config.chromosomes
            )
        )
        yield QuerySet("", "control", control, intervals.filter_by_size(min_size=floor))


def score(
    query: QuerySet, truth: IntervalSet, config: PipelineConfig
) -> tuple[MetricsRow, Classification]:
    """One classification, reduced to a metrics row."""
    classification = classify(
        build_candidates(query.intervals, truth),
        min_reciprocal_overlap=config.evaluation.reciprocal_overlap,
        validate=False,
    )
    summary = classification.summary()
    ceiling = min(1.0, summary.n_query / summary.n_truth) if summary.n_truth else float("nan")

    row = MetricsRow(
        call_set=query.call_set,
        kind=query.kind,
        name=query.name,
        reciprocal_overlap=query.reciprocal_overlap,
        level=query.level,
        n_sources=query.n_sources,
        n_query=summary.n_query,
        n_truth=summary.n_truth,
        n_true_positive=summary.n_true_positive,
        n_false_positive=summary.n_false_positive,
        n_truth_found=summary.n_truth_found,
        n_false_negative=summary.n_false_negative,
        precision=summary.precision,
        recall=summary.recall,
        recall_ceiling=ceiling,
        f1=summary.f1,
        one_to_one=match_topology(classification).is_one_to_one,
    )
    return row, classification


def run_benchmark(config: PipelineConfig, *, write_label_files: bool = False) -> list[MetricsRow]:
    """Score every call set against the merged benchmark and write the tables.

    Assumes the BEDs are on disk. Writes `evaluation/metrics.csv`, and with
    `write_label_files` the TP / FP / FN rows behind every row of it under
    `evaluation/labels/`.
    """
    seed_chromosomes(config.chromosomes)

    samples = load_sample_list(config.sample_list_file)
    truth = load_truth(config, samples)
    print(
        f"\nMerged benchmark: {len(truth):,} intervals at or above "
        f"{config.consensus.min_size:,} bp "
        f"(padding {config.evaluation.benchmark_padding}, "
        f"crediting at {config.evaluation.reciprocal_overlap} reciprocal overlap)"
    )

    rows: list[MetricsRow] = []
    for query in iter_query_sets(config, samples):
        row, classification = score(query, truth, config)
        if write_label_files:
            written = write_labels(classification, config.layout.evaluation / "labels", query.stem)
            row = MetricsRow(**{**asdict(row), "labels": tuple(written.values())})
        rows.append(row)

    write_metrics(rows, config.layout.evaluation / METRICS_FILE)
    return rows


def write_metrics(rows: list[MetricsRow], path: Path) -> None:
    """One row per scored call set, as CSV. Deliberately not pandas: this is the
    handoff to whatever the reader plots with."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [name for name in MetricsRow.__dataclass_fields__ if name != "labels"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in columns})
    print(f"Wrote {len(rows)} metrics rows to {path}")


def format_summary(rows: list[MetricsRow], *, only_consensus: bool = False) -> str:
    """The metrics table as text, for the end of a run."""
    shown = [row for row in rows if not only_consensus or row.kind == "consensus"]
    if not shown:
        return "No call sets were scored."

    header = (
        f"{'Call set':<15}{'Set':>10}{'Overlap':>9}{'Calls':>9}"
        f"{'TP':>8}{'FP':>8}{'Prec.':>8}{'Recall':>8}{'Max.R':>8}{'F1':>8}"
    )
    lines = ["", header]
    for row in shown:
        overlap = "-" if row.reciprocal_overlap is None else f"{row.reciprocal_overlap:.2f}"
        lines.append(
            f"{row.call_set or '-':<15}{row.name:>10}{overlap:>9}{row.n_query:>9,}"
            f"{row.n_true_positive:>8,}{row.n_false_positive:>8,}{row.precision:>8.3f}"
            f"{row.recall:>8.3f}{row.recall_ceiling:>8.3f}{row.f1:>8.3f}"
        )
    return "\n".join(lines)
