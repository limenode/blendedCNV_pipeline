import glob
import re
from pathlib import Path
from typing import Callable
import pandas as pd

from cyvcf2 import VCF

from utils import PipelineConfig
from liftover import get_lifter, ChainFile

valid_chromosomes = (
    [i for i in range(1, 23)]
    + [f"chr{i}" for i in range(1, 23)]
    + ["X", "Y", "M", "chrX", "chrY", "chrM"]
)


def expand_pattern(pattern: str) -> dict[str, Path]:
    """Find every file matching `pattern` and key it by sample id.

    `{id}` is a sample-id placeholder; `*` is an ordinary glob wildcard. Files
    are located by globbing (treating `{id}` as `*`), then each file's id is
    recovered from the text that `{id}` matched. If the pattern has no `{id}`,
    the id is read from the VCF's own sample header instead.
    """
    search_glob = pattern.replace("{id}", "*")
    paths = sorted(glob.glob(search_glob, recursive=True))
    if not paths:
        print(f"Warning: no files match pattern {search_glob}")

    if "{id}" not in pattern:
        return {sample_id_from_vcf(path): Path(path) for path in paths}

    id_regex = pattern_to_regex(pattern)
    return {extract_id(path, id_regex, pattern): Path(path) for path in paths}


def pattern_to_regex(pattern: str) -> re.Pattern:
    """Compile an `{id}`/`*` path pattern into a regex with one id capture group.

    The first `{id}` becomes the capture group; any repeats become a
    backreference, so a multi-`{id}` pattern only matches when every occurrence
    holds the same text. `*` matches within a single path segment.
    """
    regex = ""
    captured = False
    for part in re.split(r"(\{id\}|\*)", pattern):
        if part == "{id}":
            regex += r"\1" if captured else r"([^/]+)"
            captured = True
        elif part == "*":
            regex += r"[^/]*"
        else:
            regex += re.escape(part)
    return re.compile(regex)


def extract_id(path: str, id_regex: re.Pattern, pattern: str) -> str:
    """Recover the sample id from `path` using a compiled `{id}` regex."""
    match = id_regex.search(path)
    if match:
        return match.group(1)
    print(f"Warning: could not extract id from {path} using pattern {pattern}")
    return Path(path).stem


def sample_id_from_vcf(path: str) -> str:
    """Read the first sample name from a VCF header, falling back to the stem."""
    try:
        vcf = VCF(path)
        if vcf.samples:
            return vcf.samples[0]
        print(f"Warning: no samples found in {path}")
    except Exception as e:
        print(f"Error reading VCF {path}: {e}")
    return Path(path).stem


def extract_end_position(record) -> int:
    """Extract END position from VCF record."""
    end = record.INFO.get("END")
    if end is not None:
        return int(end)
    return record.POS  # If END is not available, use POS as a fallback


def sanitize_svtype(svtype: str | None) -> str:
    """Sanitize SVTYPE to DEL, DUP, or NA."""
    if svtype is None:
        return "NA"
    svtype = svtype.upper()
    if svtype in {"DEL", "DELETION"}:
        return "DEL"
    elif svtype in {"DUP", "DUPLICATION", "INS", "INSERTION"}:
        return "DUP"
    else:
        return "NA"


def extract_svtype_from_info(record) -> str:
    """Extract SVTYPE from INFO field."""
    svtype = record.INFO.get("SVTYPE")
    return sanitize_svtype(svtype)


def extract_svtype_from_alt(record) -> str:
    """Extract SVTYPE from ALT field."""
    if record.ALT and len(record.ALT) > 0:
        alt_str = str(record.ALT[0]).strip()
        if alt_str.startswith("<") and alt_str.endswith(">"):
            return sanitize_svtype(alt_str[1:-1])  # Remove angle brackets
        return sanitize_svtype(alt_str)
    return "NA"


def determine_sex_threshold(record) -> float:
    """
    Determine sex chromosome threshold (1.0 for XY, 2.0 for XX) based on sex chromosome RDCN values.
    """
    sex_chrom_rdcn = []
    if record.CHROM in ["chrX", "chrY", "X", "Y"]:
        # Get RDCN from FORMAT field of first sample
        rdcn = record.format("RDCN")
        if rdcn is not None and len(rdcn) > 0:
            sex_chrom_rdcn.append(float(rdcn[0][0]))

    if not sex_chrom_rdcn:
        return 2.0  # Default to XX if no sex chromosome data

    # Calculate median RDCN for sex chromosomes
    median_rdcn = sorted(sex_chrom_rdcn)[len(sex_chrom_rdcn) // 2]

    # If median is closer to 1.0, likely XY; if closer to 2.0, likely XX
    return 1.0 if median_rdcn < 1.5 else 2.0


def extract_svtype_from_rdcn(record, sex_threshold: float) -> str:
    """Extract SVTYPE from RDCN FORMAT field."""
    # Get RDCN from FORMAT field of first sample
    rdcn = record.format("RDCN")
    if rdcn is None or len(rdcn) == 0:
        return "NA"

    rdcn = float(rdcn[0][0])
    chrom = record.CHROM

    if chrom in ["chrX", "chrY", "X", "Y"]:
        threshold = sex_threshold
    else:
        threshold = 2.0  # Autosomal threshold

    return "DUP" if rdcn > threshold else "DEL"


def determine_svtype_method(record) -> Callable:
    # Check INFO SVTYPE
    if record.INFO.get("SVTYPE") is not None:
        if sanitize_svtype(record.INFO.get("SVTYPE")) != "NA":
            return extract_svtype_from_info

    if record.ALT and len(record.ALT) > 0:
        alt_str = str(record.ALT[0]).strip()
        if alt_str.startswith("<") and alt_str.endswith(">"):
            if sanitize_svtype(alt_str[1:-1]) != "NA":
                return extract_svtype_from_alt

    # Check RDCN in FORMAT field
    rdcn = record.format("RDCN")
    if rdcn is not None and len(rdcn) > 0:
        sex_threshold = determine_sex_threshold(record)
        return lambda rec: extract_svtype_from_rdcn(rec, sex_threshold)

    # Default to INFO SVTYPE if nothing else is available
    return extract_svtype_from_info


def _process_single_vcf_to_df(
    vcf_path: Path, lifter: ChainFile | None = None, size_change_treshold: float = 0.1
) -> tuple[pd.DataFrame, dict]:
    """Process a single VCF file and convert it to a DataFrame with BED-like format.
    If a lifter is provided, apply liftover to the coordinates.
    Returns a tuple of (DataFrame, statistics).
    """

    records = []
    vcf = VCF(vcf_path)
    svtype_method: Callable | None = None

    total_call_count = 0
    total_del_call_count = 0
    total_dup_call_count = 0
    total_base_count = 0
    total_del_base_count = 0
    total_dup_base_count = 0
    calls_removed_from_failed_liftover = 0
    calls_del_removed_from_failed_liftover = 0
    calls_dup_removed_from_failed_liftover = 0
    bases_removed_from_failed_liftover = 0
    bases_removed_from_failed_liftover_del = 0
    bases_removed_from_failed_liftover_dup = 0

    for record in vcf:
        if not record.ALT or len(record.ALT) == 0:
            continue

        chrom = record.CHROM
        if chrom not in valid_chromosomes:
            continue

        # Check first valid record to determine SVTYPE extraction method
        if svtype_method is None:
            svtype_method = determine_svtype_method(record)

        start = record.POS - 1
        end = extract_end_position(record)
        svtype = svtype_method(record)

        total_call_count += 1
        total_base_count += end - start

        if svtype == "DEL":
            total_del_call_count += 1
            total_del_base_count += end - start
        elif svtype == "DUP":
            total_dup_call_count += 1
            total_dup_base_count += end - start

        if lifter:
            old_size = end - start

            new_start = lifter[chrom][start][0][1]
            new_end = lifter[chrom][end][0][1]

            new_size = new_end - new_start

            if (abs(new_size - old_size) / old_size) > size_change_treshold:
                print(
                    f"Warning: Size change >{size_change_treshold * 100}% after liftover for {chrom}:{start}-{end}"
                )

                calls_removed_from_failed_liftover += 1
                bases_removed_from_failed_liftover += old_size

                if svtype == "DEL":
                    calls_del_removed_from_failed_liftover += 1
                    bases_removed_from_failed_liftover_del += old_size
                elif svtype == "DUP":
                    calls_dup_removed_from_failed_liftover += 1
                    bases_removed_from_failed_liftover_dup += old_size

            else:
                start, end = new_start, new_end

        records.append((chrom, start, end, svtype))

    df = pd.DataFrame(records, columns=["chrom", "start", "end", "svtype"])

    statistics = {
        "total_call_count": total_call_count,
        "total_del_call_count": total_del_call_count,
        "total_dup_call_count": total_dup_call_count,
        "total_base_count": total_base_count,
        "total_del_base_count": total_del_base_count,
        "total_dup_base_count": total_dup_base_count,
        "calls_removed_from_failed_liftover": calls_removed_from_failed_liftover,
        "bases_removed_from_failed_liftover": bases_removed_from_failed_liftover,
        "calls_del_removed_from_failed_liftover": calls_del_removed_from_failed_liftover,
        "calls_dup_removed_from_failed_liftover": calls_dup_removed_from_failed_liftover,
        "bases_removed_from_failed_liftover_del": bases_removed_from_failed_liftover_del,
        "bases_removed_from_failed_liftover_dup": bases_removed_from_failed_liftover_dup,
    }

    return df, statistics


def parse_input_map(config: PipelineConfig) -> dict[str, dict[str, dict[str, Path]]]:
    """Expand every input set's tool patterns into {sample_id: file_path} maps.

    Returns a nested dict:
        {input_name: {tool_label: {sample_id: path}}}
    """
    return {
        input_name: {tool: expand_pattern(pattern) for tool, pattern in tools.items()}
        for input_name, tools in config.input.items()
    }


def process_vcfs_to_beds(config: PipelineConfig, type: str, common_only: bool = True) -> pd.DataFrame:
    """Convert all input VCFs to BED format, applying liftover if needed.
    Returns a DataFrame of liftover statistics for each input set, tool, and sample.
    """

    layout = config.layout

    all_statistics = []

    if type == "input":
        io_map = parse_input_map(config)        
        liftover_map = config.liftover

        for input_name, tools in io_map.items():
            
            common_samples = set()
            
            if common_only:
                # Check if all tools have a given sample, if not then drop that sample from the other tools
                for tool, sample_map in tools.items():
                    if len(common_samples) == 0:
                        common_samples = set(sample_map.keys())
                    else:
                        common_samples = common_samples.intersection(set(sample_map.keys()))

            for tool, sample_map in tools.items():
                if common_only:
                    sample_map = {sample_id: vcf_path for sample_id, vcf_path in sample_map.items() if sample_id in common_samples}
                liftover_dict = liftover_map.get(tool, None)
                lifter = (
                    get_lifter(liftover_dict.get("from"), liftover_dict.get("to"))
                    if liftover_dict
                    else None
                )

                for sample_id, vcf_path in sample_map.items():
                    layout.bed_tool_dir(input_name, tool).mkdir(parents=True, exist_ok=True)
                    bed_path_del = (
                        layout.bed_tool_dir(input_name, tool) / f"{sample_id}.DEL.bed"
                    )
                    bed_path_dup = (
                        layout.bed_tool_dir(input_name, tool) / f"{sample_id}.DUP.bed"
                    )

                    df, statistics = _process_single_vcf_to_df(vcf_path, lifter)

                    statistics["input_name"] = input_name
                    statistics["sample_id"] = sample_id
                    statistics["tool"] = tool
                    all_statistics.append(statistics)

                    df["source"] = f"{tool}"
                    df[df["svtype"] == "DEL"].to_csv(
                        bed_path_del, sep="\t", index=False, header=False
                    )
                    df[df["svtype"] == "DUP"].to_csv(
                        bed_path_dup, sep="\t", index=False, header=False
                    )

    return pd.DataFrame(all_statistics)
