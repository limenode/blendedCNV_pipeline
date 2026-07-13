import os
from typing import TextIO

from cyvcf2 import VCF
from liftover import get_lifter

from consensuscnv.parsing.parser_utils import discover_samples_of_interest
from consensuscnv.utils import (
    LiftoverStatus,
    PipelineConfig,
    ensure_chr_prefix,
    lift_interval,
    sanitize_svtype,
)


def process_benchmarks_to_beds(
    config: PipelineConfig, common_only: bool = True
) -> dict | None:
    """Convert benchmark VCFs to per-benchmark, per-sample BED files."""
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

                # Write one entry per sample that carries a non-reference genotype.
                # Handles are opened lazily so samples with no calls make no file.
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
        print(f"  Benchmark '{bench_name}' processing complete.\n")

    return liftover_stats if liftover_stats else None
