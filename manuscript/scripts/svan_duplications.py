"""Recover tandem duplications from the SVAN annotation of the ONT Vienna release.

Where used
----------
Methods -> "Benchmark Dataset Preparation", the ONT Vienna paragraph, and the
ONT Vienna row of Table 1 (Results -> "Input Call Sets After Parsing"). The
file this writes is what `internal.config.yaml` names as the `ont_vienna`
benchmark; the benchmark parser itself knows nothing about SVAN.

The 1KG ONT Vienna v1.1 release ships its svim-asm calls twice: a genotyped
BCF (908 samples, INS/DEL only) and a sites-only copy annotated with SVAN 1.3,
which classifies each insertion by mechanism. Four SVAN classes are tandem
duplications seen from the insertion side -- `DUP`, `INV_DUP`, `COMPLEX_DUP`
(all with `DUP_COORD`, the reference segment the inserted sequence aligns to)
and `DUP_INTERSPERSED` (no coordinates in the release, left as INS). This
script joins the two files on record ID and rewrites the first three classes
as `<DUP>` records over their reference segment, keeping every sample's
genotype, so the result is one ordinary multi-sample VCF.

Interval rule. `DUP_COORD` lists the chained alignment hits of the insert
against the reference, as `chr:beg-end_strand` (genomic, 0-based half-open PAF
coordinates) or, for hits found by SVAN's local realignment step, as
`flank_<ID>_chr:pos:beg-end_strand`, positions inside a flank sequence that
SVAN fetched as `chr:(pos - INS_LEN - 100)-(pos + INS_LEN + 100)` (SVAN-INS.py,
`align2flankingRegion`, offset 100). The duplicated interval is the span from
the lowest hit start to the highest hit end.

Multi-allelic records (several inserted sequences at one site) collapse to the
single `<DUP>` allele: any non-reference allele index becomes 1, and the
per-allele AC/AF are summed. INFO tags whose meaning does not survive the
collapse (AC_Hom, AC_Het, AC_Hemi, HWE, ExcHet, MAF) are dropped from the
rewritten records only.

Requires the two release files already cached by the pipeline under
`out/downloads/` (the SVAN BCF is not a config entry; download it once from
the same FTP directory as the genotyped BCF).

    pixi run python manuscript/scripts/svan_duplications.py
    # then point internal.config.yaml's `ont_vienna` at the output
"""

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

from cyvcf2 import VCF, Writer

ROOT = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline")
DOWNLOADS = ROOT / "out" / "downloads"

DUPLICATION_CLASSES = frozenset({"DUP", "INV_DUP", "COMPLEX_DUP"})
FLANK_OFFSET = 100  # SVAN-INS.py: align2flankingRegion(variant, refIndex, 100, outDir)

GENOMIC_HIT = re.compile(r"^(?P<chrom>[^:]+):(?P<beg>\d+)-(?P<end>\d+)_[+-]$")
FLANK_HIT = re.compile(r"^flank_[^_]+_(?P<chrom>[^:]+):(?P<pos>\d+):(?P<beg>\d+)-(?P<end>\d+)_[+-]$")

# INFO tags copied from the source record onto the rewritten one. Everything
# else on the source record is per-allele or per-genotype-class and does not
# survive collapsing to one <DUP> allele.
KEPT_INFO = ("NS", "F_MISSING", "AN")
SUMMED_INFO = ("AC", "AF")


def duplicated_interval(chrom: str, pos: int, ins_len: int, dup_coord: str) -> tuple[int, int]:
    """0-based half-open reference interval spanned by the chained hits."""
    starts, ends = [], []
    for hit in dup_coord.split(","):
        if match := FLANK_HIT.match(hit):
            if match["chrom"] != chrom or int(match["pos"]) != pos:
                raise ValueError(f"flank hit does not belong to its record: {hit}")
            flank_start = max(pos - ins_len - FLANK_OFFSET, 1) - 1  # 0-based
            starts.append(flank_start + int(match["beg"]))
            ends.append(flank_start + int(match["end"]))
        elif match := GENOMIC_HIT.match(hit):
            if match["chrom"] != chrom:
                raise ValueError(f"hit on another chromosome: {hit}")
            starts.append(int(match["beg"]))
            ends.append(int(match["end"]))
        else:
            raise ValueError(f"unrecognised DUP_COORD hit: {hit}")
    return min(starts), max(ends)


def read_annotations(path: Path) -> dict[str, tuple[int, int, str, str, int]]:
    """Record ID -> (start, end, class, DUP_COORD, INS_LEN) for the duplication classes."""
    intervals = {}
    for record in VCF(str(path)):
        svan_class = record.INFO.get("ITYPE_N")
        if svan_class not in DUPLICATION_CLASSES:
            continue
        dup_coord = record.INFO.get("DUP_COORD")
        if dup_coord is None:
            continue
        ins_len = int(record.INFO["INS_LEN"])
        start, end = duplicated_interval(record.CHROM, record.POS, ins_len, dup_coord)
        intervals[record.ID] = (start, end, svan_class, dup_coord, ins_len)
    return intervals


def parse_info(text: str) -> dict[str, str | None]:
    info: dict[str, str | None] = {}
    for item in text.split(";"):
        key, _, value = item.partition("=")
        info[key] = value if value else None
    return info


def sum_per_allele(value: str | None, cast):
    if value is None:
        return None
    return cast(sum(cast(v) for v in value.split(",")))


def rewrite(line: str, start: int, end: int, svan_class: str, dup_coord: str, ins_len: int) -> str:
    """Turn one genotyped INS line into a <DUP> line over `[start, end)`."""
    fields = line.rstrip("\n").split("\t")
    source_pos = int(fields[1])
    info = parse_info(fields[7])

    new_info = [
        "SVTYPE=DUP",
        f"END={end}",
        f"SVLEN={end - start}",
        f"ITYPE_N={svan_class}",
        f"DUP_COORD={dup_coord}",
        f"INS_POS={source_pos}",
        f"INS_LEN={ins_len}",
    ]
    for key in KEPT_INFO:
        if info.get(key) is not None:
            new_info.append(f"{key}={info[key]}")
    for key, cast in zip(SUMMED_INFO, (int, float)):
        if (total := sum_per_allele(info.get(key), cast)) is not None:
            new_info.append(f"{key}={total:g}" if cast is float else f"{key}={total}")

    fields[1] = str(start + 1)  # VCF POS is 1-based
    fields[3] = "N"
    fields[4] = "<DUP>"
    fields[7] = ";".join(new_info)
    if "," in line.split("\t", 5)[4]:  # multi-allelic: collapse allele indices
        fields[9:] = [
            re.sub(r"\d+", lambda m: "1" if int(m.group()) > 0 else "0", gt) for gt in fields[9:]
        ]
    return "\t".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--genotypes",
        type=Path,
        default=DOWNLOADS / "ont_vienna_svim.asm.hg38.bcf",
        help="genotyped svim-asm BCF as cached by the pipeline",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DOWNLOADS / "svim.asm.hg38.noGt.SVAN_1.3.bcf",
        help="sites-only SVAN 1.3 BCF from the same release directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOWNLOADS / "ont_vienna.svan_dups.vcf.gz",
        help="bgzipped VCF to write; name it in the config as the ont_vienna benchmark",
    )
    args = parser.parse_args()

    intervals = read_annotations(args.annotations)
    print(f"{len(intervals):,} duplication-class insertions with coordinates in {args.annotations.name}")

    # Pass 1: the rewritten lines, keyed for placement. A duplicated segment
    # usually lies upstream of the insertion breakpoint, so the record moves
    # backwards and cannot be emitted in a single pass over the source.
    source = VCF(str(args.genotypes))
    contig_rank = {name: i for i, name in enumerate(source.seqnames)}
    pending: list[tuple[tuple[int, int], str]] = []
    for record in source:
        if record.ID in intervals:
            start, end, svan_class, dup_coord, ins_len = intervals[record.ID]
            line = rewrite(str(record), start, end, svan_class, dup_coord, ins_len)
            pending.append(((contig_rank[record.CHROM], start + 1), line))
    pending.sort(key=lambda item: item[0])
    missing = len(intervals) - len(pending)
    if missing:
        print(f"  {missing:,} annotated IDs absent from the genotyped file")

    # Pass 2: stream the source, dropping the originals and merging the
    # rewritten lines in at their new positions.
    source = VCF(str(args.genotypes))
    source.add_info_to_header(
        {"ID": "ITYPE_N", "Number": "1", "Type": "String", "Description": "SVAN 1.3 insertion class"}
    )
    source.add_info_to_header(
        {"ID": "DUP_COORD", "Number": ".", "Type": "String", "Description": "SVAN 1.3 alignment hits of the inserted sequence, as released"}
    )
    source.add_info_to_header(
        {"ID": "INS_POS", "Number": "1", "Type": "Integer", "Description": "Breakpoint of the insertion this duplication was derived from"}
    )
    source.add_info_to_header(
        {"ID": "INS_LEN", "Number": "1", "Type": "Integer", "Description": "Length of the inserted sequence"}
    )
    source.add_to_header(
        f"##svan_duplications=<Date={datetime.now(tz=UTC).date().isoformat()},Genotypes={args.genotypes.name},"
        f"Annotations={args.annotations.name},Classes={'|'.join(sorted(DUPLICATION_CLASSES))},"
        f'Rule="INS with SVAN DUP_COORD rewritten as <DUP> over the span of its reference hits; '
        f'flank hits translated with offset {FLANK_OFFSET}; multi-allelic records collapsed to one allele">'
    )
    writer = Writer(str(args.output), source, mode="wz")

    n_written = n_dropped = 0
    i = 0
    for record in source:
        key = (contig_rank[record.CHROM], record.POS)
        while i < len(pending) and pending[i][0] <= key:
            writer.write_record(writer.variant_from_string(pending[i][1]))
            n_written += 1
            i += 1
        if record.ID in intervals:
            n_dropped += 1
            continue
        writer.write_record(record)
    for _, line in pending[i:]:
        writer.write_record(writer.variant_from_string(line))
        n_written += 1
    writer.close()
    print(f"wrote {args.output}: {n_written:,} <DUP> records replacing {n_dropped:,} INS records")


if __name__ == "__main__":
    main()
