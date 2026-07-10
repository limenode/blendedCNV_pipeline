
from dataclasses import dataclass

@dataclass(frozen=True)
class Call:
    """A single interval call, tagged with the source and sample it came from."""

    chrom: str
    start: int
    end: int
    svtype: str
    source: str
    sample_id: str

    def sort_key(self, chromosome_order: list[str] | None = None) -> tuple:
        """Return a key suitable for sorting calls by chromosome, start, end"""
        rank = {chrom: i for i, chrom in enumerate(chromosome_order or [])}
        return (rank.get(self.chrom, len(rank)), self.chrom, self.start, self.end)

    def overlaps(self, other: "Call", padding: int = 0) -> bool:
        """Return True if this call overlaps `other`, with optional padding."""
        return self.chrom == other.chrom and self.start - padding < other.end and other.start - padding < self.end
    