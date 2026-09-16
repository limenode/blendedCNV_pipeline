"""The interval call, and the fields a graph may be partitioned on."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Call:
    chrom: str
    start: int
    end: int
    svtype: str
    source: str
    sample_id: str


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
