"""The interval call, and the fields a graph may be partitioned on."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Call:
    chrom: str
    start: int
    end: int
    svtype: str
    source: str
    sample_id: str


def calls_from_records(
    records: Iterable[Mapping | tuple],
    *,
    svtype: str = "CNV",
    source: str = "user",
    sample_id: str = "sample",
) -> Iterator[Call]:
    """Calls from plain interval records, filling in whatever a record leaves out.

    A record is either a sequence ``(chrom, start, end[, svtype[, source[,
    sample_id]]])`` -- a tuple, a list, or a row from ``DataFrame.itertuples
    (index=False)`` with the columns in that order -- or a mapping with those
    keys, such as one from ``DataFrame.to_dict("records")``. Fields absent from a
    record take the keyword defaults; the defaults exist so that three-column
    intervals can go straight into `build_callset`.
    """
    defaults = (svtype, source, sample_id)
    for record in records:
        if isinstance(record, Mapping):
            yield Call(
                chrom=str(record["chrom"]),
                start=int(record["start"]),
                end=int(record["end"]),
                svtype=str(record.get("svtype", svtype)),
                source=str(record.get("source", source)),
                sample_id=str(record.get("sample_id", sample_id)),
            )
        else:
            if not 3 <= len(record) <= 6:
                raise ValueError(
                    f"a record needs 3 to 6 fields (chrom, start, end, svtype, source, "
                    f"sample_id); got {len(record)}: {record!r}"
                )
            chrom, start, end, *rest = record
            rest = tuple(str(value) for value in rest) + defaults[len(rest):]
            yield Call(str(chrom), int(start), int(end), *rest)


# The Call fields an edge cannot cross, in canonical order. `chrom`
# is not among them because it always partitions: the sweep in `build_callset`
# runs one chromosome at a time. The default is what consensus calling
# needs, since joining two samples' calls would merge two people's genomes.
PARTITION_FIELDS: tuple[str, ...] = ("svtype", "sample_id")


def normalize_partition(partition_by: Iterable[str]) -> tuple[str, ...]:
    """Validate a `partition_by` argument and put it into canonical order.

    Accepts any iterable of field names drawn from `PARTITION_FIELDS`, including
    the empty tuple, which partitions on chromosome alone.
    """
    chosen = set(partition_by)
    unknown = chosen - set(PARTITION_FIELDS)
    if unknown:
        raise ValueError(
            f"partition_by names {sorted(unknown)}; the fields a graph can be "
            f"partitioned on are {PARTITION_FIELDS} (chrom always partitions)"
        )
    return tuple(field for field in PARTITION_FIELDS if field in chosen)
