"""Archived from ``src/consensuscnv/analysis/analysis_functions.py`` (commit 70e52b4).

``discover_distances_between_benchmark_cnvs`` was the only consumer of the old
``consensuscnv.calls`` / ``consensuscnv.overlap_graph`` modules outside the
computation package, and it had no callers of its own -- it was only re-exported
from ``consensuscnv.analysis.__init__``. Archived rather than ported so that
``calls.py`` and ``overlap_graph.py`` could be retired with it.

To revive: rewrite against ``consensuscnv.callsets`` -- ``read_bed_calls`` replaces
``read_bed_file``, and the gap distances this computes are already available as
``CallSet.gap_key`` (see ``build_callset``), so the whole body is likely
unnecessary.
"""

import glob
from collections import defaultdict
from pathlib import Path

from consensuscnv.calls import Call  # archived alongside this file
from consensuscnv.overlap_graph import read_bed_file  # archived alongside this file
from consensuscnv.utils import PipelineConfig


def discover_distances_between_benchmark_cnvs(
    config: PipelineConfig,
) -> defaultdict[str, list[int]]:
    # Retrieve merged benchmark CNVs per sample
    merged_benchmark_dir = config.layout.benchmark_dir("merged")

    string_paths = glob.glob(str(merged_benchmark_dir / "*.bed"))

    paths = [Path(p) for p in string_paths]

    sample_calls_dict: defaultdict[str, list[Call]] = defaultdict(list)
    for path in paths:
        sample_name = Path(path).stem.split(".")[0]
        calls = read_bed_file(path, membership="merged")
        sample_calls_dict[sample_name] = calls

    sample_distances_dict: defaultdict[str, list[int]] = defaultdict(list)
    for sample, calls in sample_calls_dict.items():
        distances = []

        for svtype in ["DEL", "DUP"]:
            svtype_calls = [c for c in calls if c.svtype == svtype]
            if len(svtype_calls) < 2:
                continue  # Need at least two calls to compute distances

            sorted_calls = sorted(svtype_calls, key=lambda c: (c.chrom, c.start))

            # Get shortest distance from one call to another
            for i in range(1, len(sorted_calls) - 1):
                if sorted_calls[i].chrom != sorted_calls[i - 1].chrom or sorted_calls[i].chrom != sorted_calls[i + 1].chrom:
                    continue  # Skip if not on the same chromosome as neighbors

                distance_to_prev = sorted_calls[i].start - sorted_calls[i - 1].end
                distance_to_next = sorted_calls[i + 1].start - sorted_calls[i].end
                min_distance = min(distance_to_prev, distance_to_next)
                if min_distance < 0:
                    print(f"Warning: Overlapping calls detected for sample {sample}")

                distances.append(min_distance)

        sample_distances_dict[sample] = distances

    return sample_distances_dict
