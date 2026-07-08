import os
import glob
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Tuple
from cyvcf2 import VCF

from liftover import get_lifter

from utils import (
    PipelineConfig,
    lift_interval,
    LiftoverStatus,
    ensure_chr_prefix,
    sanitize_svtype,
)


def _merge_intervals(
    records: List[Tuple[str, int, int, str]],
    chromosome_order: List[str] | None = None,
) -> Iterator[Tuple[str, int, int, str]]:
    """Collapse overlapping/book-ended intervals, combining their sources.

    `records` is a list of (chrom, start, end, source) for a single
    (sample, svtype) group. Mirrors `bedtools merge -c 4 -o distinct`: sort,
    then sweep, extending the running interval while the next one starts at or
    before the current end. Yields (chrom, start, end, combined_source) where
    combined_source is the sorted, de-duplicated set of merged sources.

    `chromosome_order` (the genome-file contig order) sets the output ordering;
    contigs not in it sort after the known ones in lexicographic order.
    """
    rank = {chrom: i for i, chrom in enumerate(chromosome_order or [])}
    ordered = sorted(
        records, key=lambda r: (rank.get(r[0], len(rank)), r[0], r[1], r[2], r[3])
    )
    current: Tuple[str, int, int, set[str]] | None = None
    for r_chrom, r_start, r_end, r_source in ordered:
        if current is not None and current[0] == r_chrom and r_start <= current[2]:
            chrom, start, end, sources = current
            sources.add(r_source)
            current = (chrom, start, max(end, r_end), sources)
        else:
            if current is not None:
                chrom, start, end, sources = current
                yield chrom, start, end, ",".join(sorted(sources))
            current = (r_chrom, r_start, r_end, {r_source})
    if current is not None:
        chrom, start, end, sources = current
        yield chrom, start, end, ",".join(sorted(sources))


def process_benchmarks_to_beds(
    config: PipelineConfig, common_only: bool = True
) -> dict | None:
    """Convert gold-standard benchmark VCFs to per-sample DEL/DUP BED files.

    Every benchmark in `benchmark_map` is read exactly once; its per-sample
    calls are buffered by (sample_id, svtype) so overlapping intervals from
    different benchmarks can be merged into one record set (with sources
    combined) before writing, matching the "merged" benchmark semantics.
    Returns a per-benchmark summary of records dropped by liftover, or None.
    """
    if not config.benchmark_map:
        print("No benchmark map found in config. Skipping benchmark parsing.")
        return None

    layout = config.layout

    sample_ids: set[str] = set()
    if common_only:
        for key in config.input.keys():
            bed_paths = glob.glob(str(layout.set_dir(key)) + "/consensus*/*/*.bed")
            sample_ids |= {Path(path).name.split(".")[0] for path in bed_paths}

        if not sample_ids:
            print(
                "Warning: No samples found in consensus call sets. Skipping benchmark processing."
            )
            return None
        print(f"Found {len(sample_ids)} samples of interest from consensus call sets")

    sample_id_list = list(sample_ids) if common_only else None

    liftover_stats: dict = {}
    # (sample_id, svtype) -> list of (chrom, start, end, source) across all benchmarks
    buffers: dict[tuple[str, str], List[Tuple[str, int, int, str]]] = defaultdict(list)

    for bench_name, bench_path in config.benchmark_map.items():
        print(f"Processing benchmark {bench_name} at {bench_path}")
        vcf = VCF(bench_path, samples=sample_id_list, threads=2)
        source = bench_name.replace(" ", "_").lower()

        liftover_dict = config.liftover.get(bench_name)
        lifter = (
            get_lifter(liftover_dict["from"], liftover_dict["to"])
            if liftover_dict else None
        )

        dropped_unmapped = 0
        dropped_size_change = 0

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

            # Buffer one entry per sample that carries a non-reference genotype.
            for idx, gt in enumerate(record.genotypes):
                if gt[0] == 0 and gt[1] == 0:
                    continue  # Skip homozygous reference samples
                sample_id = vcf.samples[idx]
                buffers[(sample_id, svtype)].append((chrom, start, end, source))

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

    # Merge across benchmarks per (sample, svtype), then write one BED each.
    output_dir = layout.benchmark
    os.makedirs(output_dir, exist_ok=True)
    for (sample_id, svtype), records in buffers.items():
        with open(output_dir / f"{sample_id}.{svtype}.bed", "w") as fh:
            for chrom, start, end, combined_source in _merge_intervals(
                records, config.chromosome_order
            ):
                fh.write(f"{chrom}\t{start}\t{end}\t{svtype}\t{combined_source}\n")

    return liftover_stats if liftover_stats else None
