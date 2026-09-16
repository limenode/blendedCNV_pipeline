"""Process-wide name -> id tables."""

import threading
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

from consensuscnv.genome import read_genome_file

MAX_SOURCES = 63


class Registry:
    """A name -> id table."""

    __slots__ = ("_ids", "_label", "_lock", "_max_ids", "_names")

    def __init__(self, label: str, seed=(), *, max_ids: int | None = None):
        self._ids: dict[str, int] = {}
        self._names: list[str] = []
        self._lock = threading.Lock()
        self._label = label
        self._max_ids = max_ids

        for name in seed:
            self.intern(name)

    def intern(self, name: str) -> int:
        """Return the id for `name`, assigning one if it is new."""
        index = self._ids.get(name)
        if index is not None:
            return index

        with self._lock:
            index = self._ids.get(name)  # Check again inside the lock
            if index is not None:
                return index

            index = len(self._names)
            if self._max_ids is not None and index >= self._max_ids:
                raise ValueError(f"{self._label} registry is full (max {self._max_ids})")

            self._ids[name] = index
            self._names.append(name)

        return index

    def get(self, name: str) -> int:
        """Return the id for `name`, or -1 if it is not present."""
        return self._ids.get(name, -1)

    @property
    def names(self) -> list[str]:
        """Return the list of names in order of their ids."""
        return self._names

    def __len__(self) -> int:
        """Return the number of names in the registry."""
        return len(self._names)

    def __repr__(self) -> str:
        return f"<Registry {self._label} ({len(self)} names)>"

    def clear(self, seed: Iterable[str] = ()) -> None:
        """Forget every name, then intern `seed` in order.

        Ids handed out before this are meaningless afterwards; see
        `reset_registries`.
        """
        with self._lock:
            self._ids.clear()
            self._names.clear()
        for name in seed:
            self.intern(name)


# Seeded from the config's genome file by `seed_chromosomes`, never at import:
# the registry should hold the chromosomes of the genome being analysed and
# nothing else.
CHROMOSOMES = Registry("chromosome")
SVTYPES = Registry("svtype", ["DEL", "DUP"])
SAMPLES = Registry("sample")
SOURCES = Registry("source", max_ids=MAX_SOURCES)


def seed_chromosomes(names: Iterable[str] | str | Path) -> None:
    """Intern the analysis chromosomes into the registry, in genome order.

    Call once, before any `build_callset`, with a genome file or the names. Ids
    order the genome, so a second call must agree with the first about the order
    of any name they share: a subset, or a superset that only appends, is fine
    and interns nothing new for the names already present. A different order
    raises before anything is interned -- a genome that cannot be ordered by
    the ids already handed out would sort correctly but be written out in the
    old order. `reset_registries` starts over.
    """
    if isinstance(names, (Path, str)):
        names = read_genome_file(names)
    names = tuple(names)

    # The id each name would end up with, without interning anything yet.
    next_id = len(CHROMOSOMES)
    ids = []
    for name in names:
        index = CHROMOSOMES.get(name)
        if index < 0:
            index, next_id = next_id, next_id + 1
        ids.append(index)

    if any(later <= earlier for earlier, later in pairwise(ids)):
        raise ValueError(
            "The chromosome registry already orders these names differently: it holds "
            f"{CHROMOSOMES.names} and the genome given is {list(names)}. Ids are "
            "assigned once per process, so a genome loaded after another must agree "
            "with it on the order of every shared name (or the list has a duplicate). "
            "Call reset_registries() to start over."
        )

    for name in names:
        CHROMOSOMES.intern(name)


def reset_registries() -> None:
    """Forget every interned chromosome, svtype, sample and source."""
    CHROMOSOMES.clear()
    SVTYPES.clear(["DEL", "DUP"])
    SAMPLES.clear()
    SOURCES.clear()


__all__ = [
    "CHROMOSOMES",
    "MAX_SOURCES",
    "SAMPLES",
    "SOURCES",
    "SVTYPES",
    "Registry",
    "read_genome_file",
    "reset_registries",
    "seed_chromosomes",
]
