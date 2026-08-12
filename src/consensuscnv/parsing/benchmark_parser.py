import os
from collections import Counter
from typing import TextIO

from cyvcf2 import VCF
from liftover import get_lifter

from consensuscnv.parsing.parser_utils import ExclusionMask, discover_samples_of_interest
from consensuscnv.utils import (
    LiftoverStatus,
    PipelineConfig,
    ensure_chr_prefix,
    lift_interval,
    sanitize_svtype,
)

BENCHMARK_STAT_KEYS = (
    "total_record_count",
    "total_call_count",
    "total_del_call_count",
    "total_dup_call_count",
    "total_base_count",
    "records_dropped",
    "records_dropped_unmapped",
    "records_dropped_size_change",
    "records_removed_excluded",
    "calls_removed_excluded",
    "calls_del_removed_excluded",
    "calls_dup_removed_excluded",
    "bases_removed_excluded",
    "bases_masked_excluded",
)


def process_benchmarks_to_beds(
    config: PipelineConfig,
    excluded_regions: ExclusionMask | None = None,
    common_only: bool = True,
    max_excluded_fraction: float = 0.0,
) -> dict:
    """Convert benchmark VCFs to per-benchmark, per-sample BED files."""
    excluded_regions = excluded_regions or ExclusionMask({})
    if not config.benchmark:
        print("No benchmark map found in config. Skipping benchmark parsing.")
        return {}

    layout = config.layout

    samples_of_interest = discover_samples_of_interest(config) if common_only else set()

    liftover_stats: dict = {}

    for bench_name, bench_path in config.benchmark.items():
        print(f"Processing benchmark {bench_name} at {bench_path}")
        vcf = VCF(bench_path, samples=list(samples_of_interest), threads=2)
        source = bench_name.replace(" ", "_").lower()

        output_dir = layout.benchmark_dir(bench_name)
        os.makedirs(output_dir, exist_ok=True)

        liftover_dict = config.liftover.get(bench_name)
        lifter = (
            get_lifter(liftover_dict["from"], liftover_dict["to"])
            if liftover_dict else None
        )

        stats: Counter[str] = Counter(dict.fromkeys(BENCHMARK_STAT_KEYS, 0))
        handles: dict[str, TextIO] = {}  # sample_id -> open file
        try:
            for record in vcf:
                chrom = ensure_chr_prefix(record.CHROM)
                if chrom not in config.valid_chromosomes:
                    continue

                if not record.ALT or len(record.ALT) == 0:
                    continue

                start = record.POS - 1  # Convert to 0-based
                record_id = record.ID if record.ID else "."

                # Extract END - try INFO field first, then calculate from SVLEN
                end = record.INFO.get("END")
                if end is not None:
                    end = int(end)
                else:
                    svlen = record.INFO.get("SVLEN")
                    if svlen is not None:
                        end = record.POS + abs(int(svlen))
                if end is None:
                    continue  # Skip records without END or SVLEN

                # Extract and sanitize SVTYPE
                raw_svtype = record.INFO.get("SVTYPE")
                svtype = sanitize_svtype(raw_svtype, record_id)
                if svtype == "NA":
                    continue  # Skip records with unrecognized SVTYPE

                # Perform liftover if necessary
                if lifter:
                    status, lifted = lift_interval(lifter, chrom, start, end)
                    if lifted is None:
                        stats["records_dropped"] += 1
                        if status is LiftoverStatus.UNMAPPED:
                            stats["records_dropped_unmapped"] += 1
                        else:  # LiftoverStatus.SIZE_CHANGE
                            stats["records_dropped_size_change"] += 1
                        continue
                    start, end = lifted

                # One VCF record becomes one BED line per carrier.
                carriers = [
                    vcf.samples[idx]
                    for idx, gt in enumerate(record.genotypes)
                    if not (gt[0] == 0 and gt[1] == 0)
                ]
                if not carriers:
                    continue

                kind = svtype.lower() if svtype in ("DEL", "DUP") else None
                size = end - start
                stats["total_record_count"] += 1
                stats["total_call_count"] += len(carriers)
                stats["total_base_count"] += size * len(carriers)
                if kind:
                    stats[f"total_{kind}_call_count"] += len(carriers)

                # After liftover, so the mask sees target-assembly coordinates.
                if excluded_regions.is_excluded(chrom, start, end, max_excluded_fraction):
                    masked = excluded_regions.overlap_bp(chrom, start, end)
                    stats["records_removed_excluded"] += 1
                    stats["calls_removed_excluded"] += len(carriers)
                    stats["bases_removed_excluded"] += size * len(carriers)
                    stats["bases_masked_excluded"] += masked * len(carriers)
                    if kind:
                        stats[f"calls_{kind}_removed_excluded"] += len(carriers)
                    continue

                # Open one handle per (sample_id) lazily, so no empty files are made.
                for sample_id in carriers:
                    fh = handles.get(sample_id)
                    if fh is None:
                        fh = open(output_dir / f"{sample_id}.bed", "w")
                        handles[sample_id] = fh
                    fh.write(f"{chrom}\t{start}\t{end}\t{svtype}\t{source}\n")
        finally:
            for fh in handles.values():
                fh.close()

        if liftover_dict:
            print(
                f"  {bench_name}: dropped {stats['records_dropped']} records that failed "
                f"liftover ({stats['records_dropped_unmapped']} unmapped, "
                f"{stats['records_dropped_size_change']} size change)"
            )

        if stats["calls_removed_excluded"]:
            print(
                f"  {bench_name}: dropped {stats['calls_removed_excluded']:,} calls "
                f"({stats['records_removed_excluded']:,} records) overlapping the "
                f"exclusion mask, {stats['bases_removed_excluded'] / 1e6:,.1f} Mb removed, "
                f"{stats['bases_masked_excluded'] / 1e6:,.1f} Mb of it inside the mask"
            )

        # Recorded unconditionally: a benchmark that lost nothing still needs a
        # row, otherwise "no exclusions" and "never ran" look identical.
        liftover_stats[bench_name] = {
            "liftover_from": liftover_dict["from"] if liftover_dict else "",
            "liftover_to": liftover_dict["to"] if liftover_dict else "",
            **dict(stats),
        }
        print(f"  Benchmark '{bench_name}' processing complete.\n")

    return liftover_stats
