from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import glob
from pathlib import Path
from typing import List, Tuple
import os

from consensuscnv.calls import Call
from consensuscnv.output_layout import OutputLayout
from consensuscnv.overlap_graph import generate_graph_from_calls, read_bed_file
from consensuscnv.utils import PipelineConfig


@dataclass(frozen=True)
class ClassificationResult:
    """Binary classification of a tested call set against a truth set."""

    true_positives: List[Tuple[Call, Call]]
    false_positives: List[Call]
    false_negatives: List[Call]


def _qualifies(weight: float, reciprocal_threshold: float) -> bool:
    """Whether an overlap-graph edge weight clears the matching threshold.
    
        Graph edges are weighted by reciprocal overlap. Checks for non-zero
        overlap and that the weight is >= the threshold.
    """
    return weight > 0.0 and weight >= reciprocal_threshold


def classify_calls(
    query_calls: List[Call],
    truth_calls: List[Call],
    reciprocal_threshold: float,
) -> ClassificationResult:
    """Classify a tested call set against a truth set by cross-boundary overlap.

    Builds a single overlap graph over both sets and labels each call by its
    edges that cross the tested/truth boundary.
    - a tested call with >= 1 qualifying truth overlap is a true positive,
      emitted once per ``(tested, truth)`` pair;
    - a tested call with no qualifying truth overlap is a false positive;
    - a truth call with no qualifying tested overlap is a false negative.

    An overlap "qualifies" when the calls share at least 1bp *and* their
    reciprocal overlap (the graph edge weight) is >= ``reciprocal_threshold``,
    equivalent to ``bedtools intersect -f t -r``. Passing ``0.0`` therefore means
    "any non-zero overlap" rather than "everything" — see `_qualifies`.

    The graph partitions by ``(sample_id, svtype, chrom)``, so calls only ever
    match within the same sample, SV type, and chromosome.
    """
    combined = query_calls + truth_calls
    graph = generate_graph_from_calls(combined)

    # Nodes are enumerated in `combined` order, so the tested calls own the
    # first `len(tested_calls)` node ids.
    query_ids = set(range(len(query_calls)))

    true_positives: List[Tuple[Call, Call]] = []
    false_positives: List[Call] = []
    false_negatives: List[Call] = []

    for node_id in graph.nodes:
        call: Call = graph.nodes[node_id]["call"]
        is_tested = node_id in query_ids

        matches = [
            graph.nodes[neighbor]["call"]
            for neighbor in graph.neighbors(node_id)
            # Only edges that cross the tested/truth boundary and clear the threshold.
            if (neighbor in query_ids) != is_tested
            and _qualifies(graph.edges[node_id, neighbor].get("weight", 0.0), reciprocal_threshold)
        ]

        if is_tested:
            if matches:
                true_positives.extend((call, truth) for truth in matches)
            else:
                false_positives.append(call)
        elif not matches:
            false_negatives.append(call)

    return ClassificationResult(true_positives, false_positives, false_negatives)


def _bed5(call: Call) -> str:
    """Format a call as the pipeline's 5-column BED row (no trailing newline)."""
    return (
        f"{call.chrom}\t{call.start}\t{call.end}\t{call.svtype}\t{'|'.join(sorted(call.sources))}"
    )



def _read_calls_from_dir(dir_path: Path, membership: str, sample_ids: set[str] | None = None) -> List[Call]:
    """Read every ``*.bed`` in a directory into `Call`s."""
    calls: List[Call] = []
    for bed_path in sorted(dir_path.glob("*.bed")):
        if bed_path.is_file() and (sample_ids is None or bed_path.stem in sample_ids):
            calls.extend(read_bed_file(bed_path, membership=membership))
    return calls


def _write_classification(
    result: ClassificationResult,
    output_dir: Path,
    chrom_order: List[str],
) -> None:
    """Write TP/FP/FN BEDs, one file per ``(sample, label)``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def coord_key(call: Call):
        return call.sort_key(chrom_order)

    buckets: dict[Tuple[str, str], List[str]] = defaultdict(list)

    for pred, truth in sorted(
        result.true_positives, key=lambda pair: (coord_key(pair[0]), coord_key(pair[1]))
    ):
        buckets[(pred.sample_id, "TP")].append(f"{_bed5(pred)}\t{_bed5(truth)}")
    for call in sorted(result.false_positives, key=coord_key):
        buckets[(call.sample_id, "FP")].append(_bed5(call))
    for call in sorted(result.false_negatives, key=coord_key):
        buckets[(call.sample_id, "FN")].append(_bed5(call))

    for (sample, label), rows in buckets.items():
        out_path = output_dir / OutputLayout.classification_bed(sample, None, label)
        with open(out_path, "w") as f:
            f.write("\n".join(rows) + "\n")


def _classify_set(
    query_dir: Path,
    truth_dir: Path,
    output_dir: Path,
    reciprocal_threshold: float,
    chrom_order: List[str],
    common_samples: set[str] | None = None,
) -> None:
    """Classify one tested call set against the merged benchmark and write BEDs."""
    query_calls = _read_calls_from_dir(query_dir, membership="tested")
    if not query_calls:
        return

    truth_calls = _read_calls_from_dir(truth_dir, membership="truth", sample_ids=common_samples)
    if not truth_calls:
        return

    result = classify_calls(query_calls, truth_calls, reciprocal_threshold=reciprocal_threshold)
    _write_classification(result, output_dir, chrom_order)

def run_binary_classification_script(
    config: PipelineConfig,
    sets_for_classification: List[Tuple[str, str, str, float]]
):
    """Classify each tested call set against the merged benchmark.

    Args:
        config: The pipeline configuration.
        sets_for_classification: ``(input_dir, output_dir)`` pairs, where
            ``input_dir`` holds the tested call set's per-sample BEDs.
    """
    
    max_workers = os.cpu_count() or 1
    max_workers = max(
        min(
            ((max_workers * 2) // 3), 
            len(sets_for_classification)
        ),
        1
    )
    
    def resolve_sets(items: Tuple[str, str, str, float]) -> Tuple[Path, Path, Path, float]:
        if len(items) == 4:
            query_dir, output_path, truth_dir, reciprocal_threshold = items
        else:
            raise ValueError(
                f"Each set for classification must be a 2-tuple or 3-tuple, got {items}"
            )
        
        query_path = Path(query_dir)
        truth_path = Path(truth_dir)
        output_path = Path(output_path)
        
        if not query_path.exists():
            raise FileNotFoundError(f"Query directory {query_dir} does not exist.")
        if not truth_path.exists():
            raise FileNotFoundError(f"Truth directory {truth_dir} does not exist.")

        return Path(query_dir), Path(truth_dir), Path(output_path), reciprocal_threshold
    
    # First pass: Find common samples
    common_samples: set[str] = set()
    
    for items in sets_for_classification:
        query_dir, truth_dir, _, _ = resolve_sets(items)
        
        query_samples = {bed_path.stem for bed_path in query_dir.glob("*.bed") if bed_path.is_file()}
        truth_samples = {bed_path.stem for bed_path in truth_dir.glob("*.bed") if bed_path.is_file()}
        
        if not common_samples:
            common_samples = query_samples.intersection(truth_samples)
        else:
            common_samples.intersection_update(query_samples.intersection(truth_samples))
    
    print(f"Common samples across all sets: {common_samples}")
    
    tasks: List[Tuple[Path, Path, Path, float, List[str], set[str]]] = []
    for items in sets_for_classification:
        query_dir, truth_dir, output_path, reciprocal_threshold = resolve_sets(items)
        
        tasks.append(
            (
                query_dir,
                truth_dir,
                output_path,
                reciprocal_threshold,
                config.chromosome_order,
                common_samples
            )
        )

    if not tasks:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_classify_set, *zip(*tasks)))
