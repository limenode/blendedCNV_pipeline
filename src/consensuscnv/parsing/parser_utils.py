import glob
from dataclasses import dataclass
from pathlib import Path

from consensuscnv.utils import PipelineConfig, ensure_chr_prefix


def discover_samples_of_interest(config: PipelineConfig) -> set[str]:
    layout = config.layout

    sample_ids: set[str] = set()
    for key in config.experimental:
        bed_paths = glob.glob(str(layout.call_set_dir(key)) + "/*/*.bed")
        sample_ids |= {Path(path).name.split(".")[0] for path in bed_paths}

    if not sample_ids:
        print("Warning: No samples found in consensus call sets. Skipping control processing.")

    else:
        print(f"Found {len(sample_ids)} samples of interest from consensus call sets")

    return sample_ids

def _merge_regions(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and union one chromosome's mask regions."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True, slots=True)
class ExclusionMask:
    """Genome regions for subtraction/exclusion from other CNV call sets."""

    regions: dict[str, list[tuple[int, int]]]

    @classmethod
    def load(cls, path: str | Path | None) -> "ExclusionMask":
        """Read a BED of excluded regions. A missing path yields an empty mask."""
        if path is None:
            return cls(regions={})

        path = Path(path)
        if not path.exists():
            print(
                f"Warning: excluded regions file {path} does not exist. "
                "No regions will be excluded."
            )
            return cls(regions={})

        excluded_regions: dict[str, list[tuple[int, int]]] = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                chrom, start, end = line.split("\t")[:3]
                excluded_regions.setdefault(ensure_chr_prefix(chrom), []).append(
                    (int(start), int(end))
                )

        merged = {chrom: _merge_regions(spans) for chrom, spans in excluded_regions.items()}
        n_regions = sum(len(spans) for spans in merged.values())
        masked_bp = sum(e - s for spans in merged.values() for s, e in spans)
        print(
            f"Loaded exclusion mask: {n_regions:,} regions over {len(merged)} chromosomes, "
            f"{masked_bp / 1e6:,.1f} Mb"
        )
        return cls(regions=merged)

    def __bool__(self) -> bool:
        """False when the mask is empty, so callers can skip the work entirely."""
        return bool(self.regions)

    def overlapping(self, chrom: str, start: int, end: int):
        return [(s, e) for s, e in self.regions.get(chrom, ()) if start < e and end > s]

    def overlap_bp(self, chrom: str, start: int, end: int) -> int:
        return sum(
            min(end, e) - max(start, s) for s, e in self.overlapping(chrom, start, end)
        )

    def excluded_fraction(self, chrom: str, start: int, end: int) -> float:
        length = end - start
        return self.overlap_bp(chrom, start, end) / length if length > 0 else 0.0

    def subtract(self, chrom: str, start: int, end: int) -> list[tuple[int, int]]:
        """The parts of [start, end) not covered by the mask. Analogous to `bedtools subtract`"""
        fragments, cursor = [], start
        for region_start, region_end in self.overlapping(chrom, start, end):
            if region_start > cursor:
                fragments.append((cursor, region_start))
            cursor = max(cursor, region_end)
        if cursor < end:
            fragments.append((cursor, end))
        return fragments

    def is_excluded(
        self, chrom: str, start: int, end: int, max_excluded_fraction: float = 0.0
    ) -> bool:
        """True when the mask covers more than `max_excluded_fraction` of the call.

        The 0.0 default drops on any overlap at all (equivalent to `bedtools subtract -A`).
        """
        covered = self.overlap_bp(chrom, start, end)
        if covered == 0:
            return False
        length = end - start
        return length <= 0 or covered > max_excluded_fraction * length
