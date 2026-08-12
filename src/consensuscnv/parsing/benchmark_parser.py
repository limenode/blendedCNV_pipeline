import os
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


def process_benchmarks_to_beds(
    config: PipelineConfig,
    excluded_regions: ExclusionMask | None = None,
    common_only: bool = True,
    max_excluded_fraction: float = 0.0,
) -> dict | None:
    """Convert benchmark VCFs to per-benchmark, per-sample BED files."""
    excluded_regions = excluded_regions or ExclusionMask({})
    if not config.benchmark:
        print("No benchmark map found in config. Skipping benchmark parsing.")
        return None

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

        dropped_unmapped = 0
        dropped_size_change = 0
        dropped_excluded = 0
        bases_removed_excluded = 0
        bases_masked_excluded = 0
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
                        if status is LiftoverStatus.UNMAPPED:
                            dropped_unmapped += 1
                        else:  # LiftoverStatus.SIZE_CHANGE
                            dropped_size_change += 1
                        continue
                    start, end = lifted

                # After liftover, so the mask sees target-assembly coordinates.
                if excluded_regions.is_excluded(chrom, start, end, max_excluded_fraction):
                    dropped_excluded += 1
                    bases_removed_excluded += end - start
                    bases_masked_excluded += excluded_regions.overlap_bp(chrom, start, end)
                    continue

                # Open one handle per (sample_id) lazily, so no empty files are made.
                for idx, gt in enumerate(record.genotypes):
                    if gt[0] == 0 and gt[1] == 0:
                        continue  # Skip homozygous reference samples
                    sample_id = vcf.samples[idx]

                    fh = handles.get(sample_id)
                    if fh is None:
                        fh = open(output_dir / f"{sample_id}.bed", "w")
                        handles[sample_id] = fh
                    fh.write(f"{chrom}\t{start}\t{end}\t{svtype}\t{source}\n")
        finally:
            for fh in handles.values():
                fh.close()

        if liftover_dict:
            dropped = dropped_unmapped + dropped_size_change
            print(
                f"  {bench_name}: dropped {dropped} records that failed liftover "
                f"({dropped_unmapped} unmapped, {dropped_size_change} size change)"
            )
            liftover_stats[bench_name] = {
                "from": liftover_dict["from"],
                "to": liftover_dict["to"],
                "records_dropped": dropped,
                "records_dropped_unmapped": dropped_unmapped,
                "records_dropped_size_change": dropped_size_change,
            }

        if dropped_excluded:
            print(
                f"  {bench_name}: dropped {dropped_excluded:,} records overlapping the "
                f"exclusion mask ({bases_removed_excluded / 1e6:,.1f} Mb removed, "
                f"{bases_masked_excluded / 1e6:,.1f} Mb of it inside the mask)"
            )
            liftover_stats.setdefault(bench_name, {}).update(
                {
                    "records_dropped_excluded": dropped_excluded,
                    "bases_removed_excluded": bases_removed_excluded,
                    "bases_masked_excluded": bases_masked_excluded,
                }
            )
        print(f"  Benchmark '{bench_name}' processing complete.\n")

    return liftover_stats if liftover_stats else None
