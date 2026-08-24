import os
from collections import Counter
from collections.abc import Iterator
from typing import TextIO

from liftover import get_lifter

from consensuscnv.parsing.parser_utils import ExclusionMask, discover_samples_of_interest
from consensuscnv.utils import LiftoverStatus, PipelineConfig, lift_interval


def iter_penncnv_records(penncnv_file: str) -> Iterator[tuple[str, int, int, str, str]]:
    """Yield (chrom, start, end, svtype, sample_id) for each CNV call in a PennCNV file.

    Example Record:
        chr1:112494946-112506104  numsnp=5  length=11,159  state1,cn=0  /path/HG00096.sig.tsv  startsnp=...  endsnp=...
    """
    with open(penncnv_file) as f:
        for line in f:
            parts = line.split()
            # Skip blanks, comments, and any line whose first token isn't a locus.
            if (
                len(parts) < 5
                or not parts[0].startswith("chr")
                or "cn=" not in parts[3]
            ):
                continue

            chrom, _, span = parts[0].partition(":")  # chr1 : 112494946-112506104
            start_str, _, end_str = span.partition("-")

            cn = int(parts[3].rsplit("cn=", 1)[1])  # state1,cn=0 -> 0
            if cn <= 1:
                svtype = "DEL"
            elif cn >= 3:
                svtype = "DUP"
            else:
                continue  # cn == 2, normal

            sample_id = os.path.basename(parts[4]).split(".")[0]
            yield chrom, int(start_str) - 1, int(end_str), svtype, sample_id


# PennCNV records are already per-sample, so one record is one call.
PENNCNV_STAT_KEYS = (
    "total_call_count",
    "total_del_call_count",
    "total_dup_call_count",
    "total_base_count",
    "records_dropped",
    "records_dropped_unmapped",
    "records_dropped_size_change",
    "calls_removed_excluded",
    "calls_del_removed_excluded",
    "calls_dup_removed_excluded",
    "bases_removed_excluded",
    "bases_masked_excluded",
)


def process_penncnv_to_beds(
    config: PipelineConfig,
    excluded_regions: ExclusionMask | None = None,
    common_only: bool = True,
    max_excluded_fraction: float = 0.0,
    samples: frozenset[str] | None = None,
) -> dict:
    """Convert control PennCNV datasets to per-sample DEL/DUP BED files."""
    excluded_regions = excluded_regions or ExclusionMask({})
    if not config.control:
        print("No control datasets found in config. Skipping control processing.")
        return {}

    layout = config.layout

    liftover_stats: dict = {}

    for control_name, control_path in config.control.items():
        print(f"Processing control dataset: {control_name}")
        source = control_name.replace(" ", "_").lower()

        output_dir = layout.control_bed_dir(control_name)
        os.makedirs(output_dir, exist_ok=True)

        # Build the lifter once, only if liftover was requested for this control.
        liftover_dict = config.liftover.get(control_name)
        lifter = (
            get_lifter(liftover_dict["from"], liftover_dict["to"])
            if liftover_dict
            else None
        )

        stats: Counter[str] = Counter(dict.fromkeys(PENNCNV_STAT_KEYS, 0))
        handles: dict[str, TextIO] = {}  # (sample_id) -> open file
        try:
            for chrom, start, end, svtype, sample_id in iter_penncnv_records(
                control_path
            ):
                if config.valid_chromosomes and chrom not in config.valid_chromosomes:
                    continue

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

                kind = svtype.lower() if svtype in ("DEL", "DUP") else None
                size = end - start
                stats["total_call_count"] += 1
                stats["total_base_count"] += size
                if kind:
                    stats[f"total_{kind}_call_count"] += 1

                # After liftover, so the mask sees target-assembly coordinates.
                if excluded_regions.is_excluded(chrom, start, end, max_excluded_fraction):
                    stats["calls_removed_excluded"] += 1
                    stats["bases_removed_excluded"] += size
                    stats["bases_masked_excluded"] += excluded_regions.overlap_bp(
                        chrom, start, end
                    )
                    if kind:
                        stats[f"calls_{kind}_removed_excluded"] += 1
                    continue

                # Open one handle per (sample_id) lazily, so no empty files are made.
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
                f"  {control_name}: dropped {stats['records_dropped']} records that failed "
                f"liftover ({stats['records_dropped_unmapped']} unmapped, "
                f"{stats['records_dropped_size_change']} size change)"
            )

        if stats["calls_removed_excluded"]:
            print(
                f"  {control_name}: dropped {stats['calls_removed_excluded']:,} calls "
                f"overlapping the exclusion mask, "
                f"{stats['bases_removed_excluded'] / 1e6:,.1f} Mb removed, "
                f"{stats['bases_masked_excluded'] / 1e6:,.1f} Mb of it inside the mask"
            )

        # Recorded unconditionally: a control that lost nothing still needs a row,
        # otherwise "no exclusions" and "never ran" look identical.
        liftover_stats[control_name] = {
            "liftover_from": liftover_dict["from"] if liftover_dict else "",
            "liftover_to": liftover_dict["to"] if liftover_dict else "",
            **dict(stats),
        }
        print(f"  Control dataset '{control_name}' processing complete.\n")

    return liftover_stats
