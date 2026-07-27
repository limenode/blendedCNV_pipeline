import os
from typing import Iterator, TextIO, Tuple

from liftover import get_lifter

from consensuscnv.parsing.parser_utils import discover_samples_of_interest
from consensuscnv.utils import LiftoverStatus, PipelineConfig, lift_interval


def iter_penncnv_records(penncnv_file: str) -> Iterator[Tuple[str, int, int, str, str]]:
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


def process_penncnv_to_beds(
    config: PipelineConfig, common_only: bool = True
) -> dict | None:
    """Convert control PennCNV datasets to per-sample DEL/DUP BED files."""
    if not config.control:
        print("No control datasets found in config. Skipping control processing.")
        return None

    layout = config.layout

    samples_of_interest = discover_samples_of_interest(config) if common_only else set()

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

        dropped_unmapped = 0
        dropped_size_change = 0
        handles: dict[str, TextIO] = {}  # (sample_id) -> open file
        try:
            for chrom, start, end, svtype, sample_id in iter_penncnv_records(
                control_path
            ):
                if common_only and sample_id not in samples_of_interest:
                    continue
                if config.valid_chromosomes and chrom not in config.valid_chromosomes:
                    continue

                if lifter:
                    status, lifted = lift_interval(lifter, chrom, start, end)
                    if lifted is None:
                        if status is LiftoverStatus.UNMAPPED:
                            dropped_unmapped += 1
                        else:  # LiftoverStatus.SIZE_CHANGE
                            dropped_size_change += 1
                        continue
                    start, end = lifted

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
            dropped = dropped_unmapped + dropped_size_change
            print(
                f"  {control_name}: dropped {dropped} records that failed liftover "
                f"({dropped_unmapped} unmapped, {dropped_size_change} size change)"
            )
            liftover_stats[control_name] = {
                "from": liftover_dict["from"],
                "to": liftover_dict["to"],
                "records_dropped": dropped,
                "records_dropped_unmapped": dropped_unmapped,
                "records_dropped_size_change": dropped_size_change,
            }
        print(f"  Control dataset '{control_name}' processing complete.\n")

    return liftover_stats if liftover_stats else None
