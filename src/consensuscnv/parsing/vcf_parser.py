import glob
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from cyvcf2 import VCF
from liftover import ChainFile, get_lifter

from consensuscnv.parsing.parser_utils import ExclusionMask
from consensuscnv.utils import (
    LiftoverStatus,
    PipelineConfig,
    ensure_chr_prefix,
    lift_interval,
    sanitize_svtype,
)

valid_chromosomes = [f"chr{i}" for i in range(1, 23)]

# Every counter `_process_single_vcf_to_df` reports. Removal counters are counted
# against `total_call_count`, which is tallied before any filtering.
# `bases_removed_excluded` is the full span of the dropped calls;
# `bases_masked_excluded` is only the part actually inside the mask. Their ratio
# is the collateral cost of dropping whole calls instead of trimming them.
PARSING_STAT_KEYS = (
    "total_call_count",
    "total_del_call_count",
    "total_dup_call_count",
    "total_base_count",
    "total_del_base_count",
    "total_dup_base_count",
    "calls_removed_from_failed_liftover",
    "calls_del_removed_from_failed_liftover",
    "calls_dup_removed_from_failed_liftover",
    "bases_removed_from_failed_liftover",
    "bases_removed_from_failed_liftover_del",
    "bases_removed_from_failed_liftover_dup",
    "calls_removed_unmapped",
    "bases_removed_unmapped",
    "calls_removed_size_change",
    "bases_removed_size_change",
    "calls_removed_excluded",
    "calls_del_removed_excluded",
    "calls_dup_removed_excluded",
    "bases_removed_excluded",
    "bases_del_removed_excluded",
    "bases_dup_removed_excluded",
    "bases_masked_excluded",
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
    if record.INFO.get("SVTYPE") is not None and sanitize_svtype(record.INFO.get("SVTYPE")) != "NA":
            return extract_svtype_from_info

    if record.ALT and len(record.ALT) > 0:
        alt_str = str(record.ALT[0]).strip()
        if alt_str.startswith("<") and alt_str.endswith(">") and sanitize_svtype(alt_str[1:-1]) != "NA":
                return extract_svtype_from_alt

    # Check RDCN in FORMAT field
    rdcn = record.format("RDCN")
    if rdcn is not None and len(rdcn) > 0:
        sex_threshold = determine_sex_threshold(record)
        return lambda rec: extract_svtype_from_rdcn(rec, sex_threshold)

    # Default to INFO SVTYPE if nothing else is available
    return extract_svtype_from_info


def _process_single_vcf_to_df(
    vcf_path: Path,
    excluded_regions: ExclusionMask,
    lifter: ChainFile | None = None,
    size_change_treshold: float = 0.1,
    max_excluded_fraction: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Process a single VCF file and convert it to a DataFrame with BED-like format.
    If a lifter is provided, apply liftover to the coordinates. Calls overlapping
    `excluded_regions` by more than `max_excluded_fraction` of their length are
    dropped whole (0.0 drops on any overlap).
    Returns a tuple of (DataFrame, statistics).
    """

    records = []
    vcf = VCF(vcf_path, threads=2)
    svtype_method: Callable | None = None

    stats: Counter[str] = Counter(dict.fromkeys(PARSING_STAT_KEYS, 0))

    for record in vcf:
        if not record.ALT or len(record.ALT) == 0:
            continue

        chrom = ensure_chr_prefix(record.CHROM)
        if chrom not in valid_chromosomes:
            continue

        # Check first valid record to determine SVTYPE extraction method
        if svtype_method is None:
            svtype_method = determine_svtype_method(record)

        start = record.POS - 1
        end = extract_end_position(record)
        svtype = svtype_method(record)

        kind = svtype.lower() if svtype in ("DEL", "DUP") else None
        size = end - start

        stats["total_call_count"] += 1
        stats["total_base_count"] += size
        if kind:
            stats[f"total_{kind}_call_count"] += 1
            stats[f"total_{kind}_base_count"] += size

        if lifter:
            old_size = end - start
            status, lifted = lift_interval(lifter, chrom, start, end, size_change_treshold)

            # Drop records that fail to map or whose size drifts past the threshold.
            if lifted is None:
                stats["calls_removed_from_failed_liftover"] += 1
                stats["bases_removed_from_failed_liftover"] += old_size
                if kind:
                    stats[f"calls_{kind}_removed_from_failed_liftover"] += 1
                    stats[f"bases_removed_from_failed_liftover_{kind}"] += old_size

                if status is LiftoverStatus.UNMAPPED:
                    stats["calls_removed_unmapped"] += 1
                    stats["bases_removed_unmapped"] += old_size
                else:  # LiftoverStatus.SIZE_CHANGE
                    stats["calls_removed_size_change"] += 1
                    stats["bases_removed_size_change"] += old_size

                continue

            start, end = lifted

        if excluded_regions.is_excluded(chrom, start, end, max_excluded_fraction):
            stats["calls_removed_excluded"] += 1
            stats["bases_removed_excluded"] += end - start
            stats["bases_masked_excluded"] += excluded_regions.overlap_bp(chrom, start, end)
            if kind:
                stats[f"calls_{kind}_removed_excluded"] += 1
                stats[f"bases_{kind}_removed_excluded"] += end - start
            continue

        records.append((chrom, start, end, svtype))

    df = pd.DataFrame(records, columns=["chrom", "start", "end", "svtype"])

    return df, dict(stats)


def get_experimental_sets_from_config(
    config: PipelineConfig, samples: frozenset[str] | None = None
) -> dict[str, dict[str, dict[str, Path]]]:
    """Expand every experimental set's tool patterns into {sample_id: file_path} maps.

    Returns a nested dict:
        {experimental_name: {tool_label: {sample_id: path}}}

    `samples` restricts every map to an allowlist; ``None`` keeps all.
    """

    def expand(pattern: str) -> dict[str, Path]:
        found = expand_pattern(pattern)
        if samples is None:
            return found
        return {
            sample_id: path for sample_id, path in found.items() if sample_id in samples
        }

    return {
        experimental_name: {tool: expand(pattern) for tool, pattern in tools.items()}
        for experimental_name, tools in config.experimental.items()
    }


def process_vcfs_to_beds(
    config: PipelineConfig,
    excluded_regions: ExclusionMask,
    common_only: bool = True,
    max_excluded_fraction: float = 0.0,
    samples: frozenset[str] | None = None,
) -> list[dict]:
    """Convert all experimental VCFs to BED format, applying liftover if needed.
    Returns a list of parsing statistics for each experimental set, tool, and sample.
    """

    layout = config.layout

    all_statistics = []

    experimental_map = get_experimental_sets_from_config(config, samples)
    liftover_map = config.liftover

    for experimental_name, tools in experimental_map.items():

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
                get_lifter(liftover_dict["from"], liftover_dict["to"])
                if liftover_dict and liftover_dict.get("from") and liftover_dict.get("to")
                else None
            )

            for sample_id, vcf_path in sample_map.items():
                layout.bed_tool_dir(experimental_name, tool).mkdir(parents=True, exist_ok=True)
                bed_path = (
                    layout.bed_tool_dir(experimental_name, tool) / f"{sample_id}.bed"
                )

                df, statistics = _process_single_vcf_to_df(
                    vcf_path,
                    excluded_regions,
                    lifter,
                    max_excluded_fraction=max_excluded_fraction,
                )

                statistics["experimental_name"] = experimental_name
                statistics["sample_id"] = sample_id
                statistics["tool"] = tool
                all_statistics.append(statistics)

                df["source"] = f"{tool}"
                df.to_csv(bed_path, sep="\t", index=False, header=False)

    return all_statistics
