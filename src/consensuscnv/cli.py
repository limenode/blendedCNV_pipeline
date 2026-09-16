"""Command line interface.

Three subcommands:

    consensuscnv init [dir]               write a config template + hg38 files
    consensuscnv call <config.yaml>       parse + consensus -> BED files
    consensuscnv benchmark <config.yaml>  the same, plus the truth-set comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from consensuscnv.utils import PipelineConfig

EPILOG = """\
examples:
  consensuscnv init myrun
  consensuscnv call config.yaml
  consensuscnv call config.yaml --reuse-beds --overlap 0.25,0.5,0.75
  consensuscnv call config.yaml --per-sample -o /scratch/out
  consensuscnv benchmark config.yaml
  consensuscnv benchmark config.yaml --reuse-beds --write-labels

Every consensus level is written, from 1 up to the number of tools configured for
a call set, so three tools gives 1of3.bed, 2of3.bed and 3of3.bed. Output lands in
<output_dir>/consensus/<call set>/overlap_<value>/.
"""


def overlap_values(text: str) -> list[float]:
    """One --overlap argument: a value, or several separated by commas."""
    values = []
    for part in text.split(","):
        part = part.strip()
        try:
            values.append(float(part))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{part!r} is not a number; --overlap takes a reciprocal overlap "
                "in [0.0, 1.0], optionally several separated by commas"
            ) from None
    if not values:
        raise argparse.ArgumentTypeError("--overlap given no value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consensuscnv",
        description="Graph-based consensus CNV calling from per-sample caller output.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"consensuscnv {_version()}")

    subcommands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    init = subcommands.add_parser(
        "init",
        help="write a config template and the hg38 genome and excluded-region files",
        description=(
            "Write config.yaml (the fields `call` needs), config.full.yaml (every "
            "option, documented) and the files they point at into a directory: the "
            "hg38 chromosome lengths (all primary chromosomes, and the autosomes "
            "only) and the excluded regions (centromeres and assembly gaps). Both "
            "configs get the absolute paths of the written genome and "
            "excluded-region files, so they work from any directory; the input "
            "and output paths are left for you."
        ),
    )
    init.add_argument(
        "directory", type=Path, nargs="?", default=Path("."),
        help="where to write; created if missing (default: the current directory)",
    )
    init.add_argument(
        "--force", action="store_true",
        help="overwrite files that already exist",
    )
    init.set_defaults(handler=cmd_init)

    call = subcommands.add_parser(
        "call",
        help="parse the input call sets and write consensus BED files",
        description="Parse the experimental VCFs and write every consensus level.",
    )
    _add_common(call)
    call.add_argument(
        "--reuse-beds",
        action="store_true",
        help="skip parsing and use the per-tool BED files already in the output "
             "directory. Much faster when only the consensus parameters changed.",
    )
    call.set_defaults(handler=cmd_call)

    benchmark = subcommands.add_parser(
        "benchmark",
        help="reproduce the paper's evaluation against control and truth sets",
        description=(
            "Parse the experimental, control and benchmark call sets, build the "
            "consensus sets, and compare them against the truth sets. Benchmark "
            "sources given as URLs are downloaded on first use."
        ),
    )
    _add_common(benchmark)
    benchmark.add_argument(
        "--reuse-beds",
        action="store_true",
        help="skip parsing and use the BED files already in the output directory, "
             "for the truth and control sets as well as the callers",
    )
    benchmark.add_argument(
        "--write-labels",
        action="store_true",
        help="also write the TP / FP / FN rows behind every metrics row, as BED "
             "files under <output_dir>/evaluation/labels/. One set of three per "
             "scored call set, so the whole grid is a lot of files.",
    )
    benchmark.set_defaults(handler=cmd_benchmark)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Arguments both subcommands take. Every one overrides the config file."""
    parser.add_argument("config", type=Path, help="path to the configuration YAML")
    parser.add_argument(
        "-o", "--output-dir", type=Path, metavar="DIR",
        help="override output_dir",
    )
    parser.add_argument(
        "--overlap", type=overlap_values, action="append", metavar="F",
        help="override consensus.reciprocal_overlap. Repeatable, and each may be a "
             "comma-separated list. One consensus run is written per value.",
    )
    parser.add_argument(
        "--min-size", type=int, metavar="BP",
        help="override consensus.min_size, the smallest consensus call kept",
    )
    parser.add_argument(
        "--per-sample", action="store_true",
        help="write one BED per sample instead of one carrying a sample column",
    )


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("consensuscnv")
    except PackageNotFoundError:  # running from a source tree, not installed
        return "unknown"


def load_config(args: argparse.Namespace) -> PipelineConfig:
    """Build the config, with any flag overrides folded in before validation."""
    from consensuscnv.utils import build_config

    overrides: dict = {}
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)

    consensus: dict = {}
    if args.overlap:
        # action="append" over a list-valued type gives a list of lists.
        consensus["reciprocal_overlap"] = [value for group in args.overlap for value in group]
    if args.min_size is not None:
        consensus["min_size"] = args.min_size
    if consensus:
        overrides["consensus"] = consensus

    return build_config(args.config, overrides=overrides or None)


def describe(config: PipelineConfig) -> None:
    print(f"Output directory: {config.output_dir}")
    print(f"Call sets:        {', '.join(config.experimental)}")
    print(
        f"Chromosomes:      {len(config.chromosomes)} "
        f"({config.chromosomes[0]}..{config.chromosomes[-1]})"
    )
    print(
        "Consensus:        overlaps "
        + ", ".join(f"{value:g}" for value in config.consensus.reciprocal_overlaps)
        + f"; min size {config.consensus.min_size} bp"
    )


# What `init` writes, in the order it reports them. The short config comes first;
# the two configs are the only files edited on the way out.
TEMPLATE_FILES = (
    "config.yaml",
    "config.full.yaml",
    "genome_primary_hg38.txt",
    "genome_autosome_hg38.txt",
    "excluded_regions_hg38.bed",
    "included_regions_hg38.bed",
)

# Config fields whose template value is the bare name of another template file;
# `init` replaces the value with that file's path in the written directory.
TEMPLATE_PATH_FIELDS = ("genome_file", "excluded_regions_file")


def write_templates(directory: Path, *, force: bool = False) -> list[Path]:
    """Write the template files into `directory` and return their paths.

    Refuses to overwrite an existing file unless `force`, and checks every
    target before writing any, so a refusal leaves the directory untouched.
    """
    from importlib.resources import files

    templates = files("consensuscnv") / "templates"
    targets = [directory / name for name in TEMPLATE_FILES]
    if not force:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite " + ", ".join(str(p) for p in existing)
                + " (pass --force to replace them)"
            )

    directory.mkdir(parents=True, exist_ok=True)
    for name, target in zip(TEMPLATE_FILES, targets, strict=True):
        text = (templates / name).read_text()
        if name.endswith(".yaml"):
            text = _fill_template_paths(text, directory.resolve())
        target.write_text(text)
    return targets


def _fill_template_paths(config_text: str, directory: Path) -> str:
    """Rewrite `field: "name"` to the path of `name` under `directory`."""
    lines = []
    for line in config_text.splitlines(keepends=True):
        for field in TEMPLATE_PATH_FIELDS:
            prefix = f"{field}: "
            if line.startswith(prefix):
                name = line[len(prefix):].strip().strip('"')
                if name in TEMPLATE_FILES:
                    line = f'{prefix}"{directory / name}"\n'
        lines.append(line)
    return "".join(lines)


def cmd_init(args: argparse.Namespace) -> int:
    written = write_templates(args.directory, force=args.force)
    for path in written:
        print(f"wrote {path}")
    print(
        f"\nNext: edit {written[0]} -- point `experimental` at your caller output "
        "and `output_dir` somewhere writable, then run\n"
        f"  consensuscnv call {written[0]}"
    )
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    from consensuscnv.consensus import format_summary, run_consensus

    config = load_config(args)
    describe(config)

    runs = run_consensus(config, reuse_beds=args.reuse_beds, per_sample=args.per_sample)
    if not runs:
        print("\nNo consensus sets were written.", file=sys.stderr)
        return 1

    print(format_summary(runs))
    print(f"\nWrote {len(runs)} consensus sets under {config.layout.consensus}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from consensuscnv.evaluation import format_summary, run_benchmark
    from consensuscnv.parsing import parse_input_files

    config = load_config(args)
    if not config.benchmark:
        print(
            "error: `benchmark` needs a `benchmark:` block in the config, naming at "
            "least one truth set as a path or a URL.",
            file=sys.stderr,
        )
        return 2

    describe(config)
    print(f"Benchmarks:       {', '.join(config.benchmark)}")
    print(f"Controls:         {', '.join(config.control) or '(none)'}")
    print(
        f"Evaluation:       padding {config.evaluation.benchmark_padding}; "
        f"crediting at {config.evaluation.reciprocal_overlap} reciprocal overlap"
    )

    if not args.reuse_beds:
        # All three parsers.
        # Benchmark sources given as URLs download here.
        parse_input_files(config)

    rows = run_benchmark(config, write_label_files=args.write_labels)
    if not rows:
        print("\nNothing was scored.", file=sys.stderr)
        return 1

    print(format_summary(rows))
    print(f"\nScored {len(rows)} call sets; tables under {config.layout.evaluation}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as error:
        # Config validation or missing input files.
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
