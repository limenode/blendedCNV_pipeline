from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    tested_calls: List[Call],
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
    combined = tested_calls + truth_calls
    graph = generate_graph_from_calls(combined)

    # Nodes are enumerated in `combined` order, so the tested calls own the
    # first `len(tested_calls)` node ids.
    tested_ids = set(range(len(tested_calls)))

    true_positives: List[Tuple[Call, Call]] = []
    false_positives: List[Call] = []
    false_negatives: List[Call] = []

    for node_id in graph.nodes:
        call: Call = graph.nodes[node_id]["call"]
        is_tested = node_id in tested_ids

        matches = [
            graph.nodes[neighbor]["call"]
            for neighbor in graph.neighbors(node_id)
            # Only edges that cross the tested/truth boundary and clear the threshold.
            if (neighbor in tested_ids) != is_tested
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


def _read_calls_from_dir(dir_path: Path, membership: str) -> List[Call]:
    """Read every ``*.bed`` in a directory into `Call`s."""
    calls: List[Call] = []
    for bed_path in sorted(dir_path.glob("*.bed")):
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
    input_dir: Path,
    output_dir: Path,
    truth_dir: Path,
    reciprocal_threshold: float,
    chrom_order: List[str],
) -> None:
    """Classify one tested call set against the merged benchmark and write BEDs."""
    tested_calls = _read_calls_from_dir(input_dir, membership="tested")
    if not tested_calls:
        return

    # The benchmark covers every sample; restrict truth to samples this set
    # actually contains, so recall isn't charged for samples that weren't tested.
    tested_samples = {call.sample_id for call in tested_calls}
    truth_calls = [
        call
        for call in _read_calls_from_dir(truth_dir, membership="benchmark")
        if call.sample_id in tested_samples
    ]

    result = classify_calls(tested_calls, truth_calls, reciprocal_threshold=reciprocal_threshold)
    _write_classification(result, output_dir, chrom_order)

def run_binary_classification_script(
    config: PipelineConfig,
    sets_for_classification: List[Tuple[str, str] | Tuple[str, str, str]],
    default_truth_dir: Path | None = None,
    reciprocal_threshold: float | None = None,
):
    """Classify each tested call set against the merged benchmark.

    Args:
        config: The pipeline configuration.
        sets_for_classification: ``(input_dir, output_dir)`` pairs, where
            ``input_dir`` holds the tested call set's per-sample BEDs.
    """

    tasks: List[Tuple[Path, Path, Path, float, List[str]]] = []    
    
    for items in sets_for_classification:
        # Resolve input/output/truth paths from the tuple
        if len(items) == 2:
            input_path, output_path = items
            if default_truth_dir is None:
                raise ValueError(
                    "A default truth directory must be provided when sets are 2-tuples."
                )
            truth_dir = default_truth_dir
        elif len(items) == 3:
            input_path, output_path, truth_dir = items
        else:
            raise ValueError(
                f"Each set for classification must be a 2-tuple or 3-tuple, got {items}"
            )
        
        if not Path(input_path).exists():
            print(
                f"Input path {input_path} does not exist. Skipping binary classification for this set."
            )
            continue
        tasks.append(
            (
                Path(input_path),
                Path(output_path),
                Path(truth_dir),
                (
                    config.matching_reciprocal_threshold
                    if reciprocal_threshold is None
                    else reciprocal_threshold
                ),
                config.chromosome_order,
            )
        )

    if not tasks:
        return

    with ProcessPoolExecutor() as executor:
        list(executor.map(_classify_set, *zip(*tasks)))
