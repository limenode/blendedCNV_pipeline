"""Parsing phase: normalize VCF / PennCNV / benchmark sources into BED files."""

from consensuscnv.parsing.benchmark_parser import process_benchmarks_to_beds
from consensuscnv.parsing.parser_utils import (
    discover_samples_of_interest,
    load_sample_list,
)
from consensuscnv.parsing.parsing_driver import parse_input_files
from consensuscnv.parsing.penncnv_parser import (
    iter_penncnv_records,
    process_penncnv_to_beds,
)
from consensuscnv.parsing.vcf_parser import (
    expand_pattern,
    parse_experimental_map,
    process_vcfs_to_beds,
    sample_id_from_vcf,
)

__all__ = [
    "discover_samples_of_interest",
    "expand_pattern",
    "iter_penncnv_records",
    "load_sample_list",
    "parse_experimental_map",
    "parse_input_files",
    "process_benchmarks_to_beds",
    "process_penncnv_to_beds",
    "process_vcfs_to_beds",
    "sample_id_from_vcf",
]
