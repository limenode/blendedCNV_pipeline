from dataclasses import dataclass


@dataclass(frozen=True)
class Call:
    """A single interval call, tagged with the source and sample it came from."""

    chrom: str
    start: int
    end: int
    svtype: str
    sources: frozenset[str]
    sample_id: str
    membership: str
    parent_calls: tuple[int, ...] = ()

    def sort_key(self, chromosome_order: list[str] | None = None) -> tuple:
        """Return a key suitable for sorting calls by chromosome, start, end"""
        rank = {chrom: i for i, chrom in enumerate(chromosome_order or [])}
        return (rank.get(self.chrom, len(rank)), self.chrom, self.start, self.end)

    def overlaps(self, other: "Call", padding: int = 0) -> bool:
        """Return True if this call overlaps `other`, with optional padding."""
        return (
            self.chrom == other.chrom
            and self.start - padding < other.end
            and other.start - padding < self.end
        )

    def bed_str(self) -> str:
        """Return a string representation of the call in BED format."""
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.svtype}\t{'|'.join(sorted(self.sources))}\t{self.sample_id}"

    def get_parent_calls(self) -> tuple[int, ...]:
        """Return the parent calls of the call, or an empty tuple if none."""
        return self.parent_calls

    @property
    def size(self) -> int:
        """Return the size of the call (end - start)."""
        return self.end - self.start
    
    def to_record(self, **extra) -> dict:
        """Return a dictionary representation of the call, including any extra fields."""
        return {
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "svtype": self.svtype,
            "sources": sorted(self.sources),
            "n_sources": len(self.sources),
            "sample_id": self.sample_id,
            "membership": self.membership,
            **extra,
        }
