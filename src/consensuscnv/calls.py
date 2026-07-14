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
    members: tuple[int, ...] = ()

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

    def get_members(self) -> tuple[int, ...]:
        """Return the members of the call, or an empty tuple if none."""
        return self.members
