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
    "records_dropped_svtype",
    "records_dropped_no_span",
    "records_removed_excluded",
    "calls_removed_excluded",
    "calls_del_removed_excluded",
    "calls_dup_removed_excluded",
    "bases_removed_excluded",
    "bases_masked_excluded",
)

# --------------------------------------------------------------------------- #
# SVTYPE handling, specific to the benchmark sets.
#
# The benchmarks do not agree on vocabulary, so the mapping is spelled out
# rather than inferred. `utils.sanitize_svtype` is deliberately not used here:
# it folds every insertion class into DUP, which is right for a caller that
# reports gains as DUP but wrong for a truth set (see INSERTION_TYPES).
# --------------------------------------------------------------------------- #

# Losses of reference sequence. 1000G phase 3 names deletions of mobile elements
# DEL_ALU / DEL_LINE1 / DEL_SVA / DEL_HERV; they are genuine deletions.
DELETION_TYPES = frozenset({"DEL", "DEL_ALU", "DEL_LINE1", "DEL_SVA", "DEL_HERV"})

DUPLICATION_TYPES = frozenset({"DUP", "DUP:TANDEM", "DUP:INT"})

# Novel-sequence insertions. These occupy no reference interval, so a read-depth
# caller cannot detect them and reciprocal overlap against them is undefined.
# Dropped rather than folded into DUP. Listed for documentation and for the
# dropped-record counter; anything not in DELETION_TYPES or DUPLICATION_TYPES is
# dropped regardless, which also covers INV and BND.
INSERTION_TYPES = frozenset(
    {"INS", "ALU", "LINE1", "SVA", "MEI", "INS:ME", "INS:MT", "HERV"}
)


def benchmark_svtype(raw_svtype: str | None) -> str | None:
    """DEL or DUP for a benchmark record, or None to drop it.

    None covers insertions, inversions, breakends, and the multi-allelic
    copy-number records, which are resolved per allele by `copy_number_alleles`.
    """
    svtype = (raw_svtype or "").upper()
    if svtype in DELETION_TYPES:
        return "DEL"
    if svtype in DUPLICATION_TYPES:
        return "DUP"
    return None


def copy_number_alleles(record) -> dict[int, str]:
    """Map ALT allele index -> DEL/DUP for ``<CNn>`` records; empty if not one.

    1000G phase 3 carries multi-allelic copy-number records whose ALTs are
    absolute copy numbers against a diploid reference (``<CN0>,<CN2>``). The
    direction is therefore a property of the allele a sample carries, not of the
    record: taking it from INFO/SVTYPE or the record ID assigns one direction to
    every carrier and mislabels everyone holding the other allele. ``<CN2>`` is
    reference copy number and contributes nothing.
    """
    alleles: dict[int, str] = {}
    for index, alt in enumerate(record.ALT, start=1):
        if alt.startswith("<CN") and alt.endswith(">") and alt[3:-1].isdigit():
            copies = int(alt[3:-1])
            if copies != 2:
                alleles[index] = "DEL" if copies < 2 else "DUP"
    return alleles


def benchmark_end(record) -> int | None:
    """End coordinate for a benchmark record, or None when it cannot be derived.

    INFO/END is authoritative. The SVLEN fallback is only meaningful for
    deletions, where SVLEN is the length of removed reference -- HGSVC3 carries
    no END field at all and relies on it. For an insertion the same arithmetic
    would fabricate a reference interval equal to the *inserted* length, which is
    why insertions are dropped on type before this is ever called.
    """
    end = record.INFO.get("END")
    if end is not None:
        return int(end)
    svlen = record.INFO.get("SVLEN")
    if svlen is None:
        return None
    if isinstance(svlen, (tuple, list)):  # Number=. in some headers
        svlen = svlen[0]
    return record.POS + abs(int(svlen))


def process_benchmarks_to_beds(
    config: PipelineConfig,
    excluded_regions: ExclusionMask | None = None,
    common_only: bool = True,
    max_excluded_fraction: float = 0.0,
    samples: frozenset[str] | None = None,
) -> dict:
    """Convert benchmark VCFs to per-benchmark, per-sample BED files."""
    excluded_regions = excluded_regions or ExclusionMask({})
    if not config.benchmark:
        print("No benchmark map found in config. Skipping benchmark parsing.")
        return {}

    layout = config.layout

    samples_of_interest = discover_samples_of_interest(config, samples, common_only)

    liftover_stats: dict = {}

    for bench_name, bench_path in config.benchmark.items():
        print(f"Processing benchmark {bench_name} at {bench_path}")
        vcf = VCF(
            bench_path,
            samples=None if samples_of_interest is None else list(samples_of_interest),
            threads=2,
        )
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

                # Carriers grouped by the call type their own genotype implies.
                # A multi-allelic copy-number record emits both directions, so
                # this cannot be one svtype for the whole record.
                allele_types = copy_number_alleles(record)
                if allele_types:
                    carriers_by_type: dict[str, list[str]] = {}
                    for idx, gt in enumerate(record.genotypes):
                        for allele in {a for a in gt[:2] if a > 0}:
                            if svtype := allele_types.get(allele):
                                carriers_by_type.setdefault(svtype, []).append(
                                    vcf.samples[idx]
                                )
                else:
                    svtype = benchmark_svtype(record.INFO.get("SVTYPE"))
                    if svtype is None:
                        # Insertion, inversion, or breakend: not a copy-number
                        # change a read-depth caller can be scored against.
                        stats["records_dropped_svtype"] += 1
                        continue
                    carriers_by_type = {
                        svtype: [
                            vcf.samples[idx]
                            for idx, gt in enumerate(record.genotypes)
                            if not (gt[0] == 0 and gt[1] == 0)
                        ]
                    }

                carriers_by_type = {k: v for k, v in carriers_by_type.items() if v}
                if not carriers_by_type:
                    continue  # no carrier among the selected samples

                end = benchmark_end(record)
                if end is None or end <= start:
                    stats["records_dropped_no_span"] += 1
                    continue

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

                n_calls = sum(len(ids) for ids in carriers_by_type.values())
                size = end - start
                stats["total_record_count"] += 1
                stats["total_call_count"] += n_calls
                stats["total_base_count"] += size * n_calls
                for svtype, sample_ids in carriers_by_type.items():
                    stats[f"total_{svtype.lower()}_call_count"] += len(sample_ids)

                # After liftover, so the mask sees target-assembly coordinates.
                if excluded_regions.is_excluded(chrom, start, end, max_excluded_fraction):
                    masked = excluded_regions.overlap_bp(chrom, start, end)
                    stats["records_removed_excluded"] += 1
                    stats["calls_removed_excluded"] += n_calls
                    stats["bases_removed_excluded"] += size * n_calls
                    stats["bases_masked_excluded"] += masked * n_calls
                    for svtype, sample_ids in carriers_by_type.items():
                        stats[f"calls_{svtype.lower()}_removed_excluded"] += len(sample_ids)
                    continue

                # Open one handle per (sample_id) lazily, so no empty files are made.
                for svtype, sample_ids in carriers_by_type.items():
                    for sample_id in sample_ids:
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
