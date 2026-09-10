"""Process-wide name -> id tables."""

import threading
from collections.abc import Iterable

MAX_SOURCES = 63

class Registry:
    """A name -> id table."""

    __slots__ = ("_ids", "_label", "_lock", "_max_ids", "_names")

    def __init__(
        self,
        label: str,
        seed=(),
        *,
        max_ids: int | None = None
    ):
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
            index = self._ids.get(name) # Check again inside the lock
            if index is not None:
                return index

            index = len(self._names)
            if self._max_ids is not None and index >= self._max_ids:
                raise ValueError(f"{self._label} registry is full (max {self._max_ids})")

            self._ids[name] = index
            self._names.append(name)

        return index

    def get(self, name: str) -> int:
        """Return the id for `name`, or None if it is not present."""
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

# Seeded from the config's genome file by `seed_chromosomes`, never at import:
# the registry should hold the chromosomes of the genome being analysed and
# nothing else.
CHROMOSOMES = Registry("chromosome")
SVTYPES = Registry("svtype", ["DEL", "DUP"])
SAMPLES = Registry("sample")
SOURCES = Registry("source", max_ids=MAX_SOURCES)

def seed_chromosomes(names: Iterable[str]) -> None:
    """Intern the analysis chromosomes into local Registry, in genome order.

    Call once, before any `build_callset`. Seeding a name the registry already
    holds is a no-op; names already present keep the ids they have.
    """
    for name in names:
        CHROMOSOMES.intern(name)


__all__ = [
    "CHROMOSOMES",
    "MAX_SOURCES",
    "SAMPLES",
    "SOURCES",
    "SVTYPES",
    "Registry",
    "seed_chromosomes",
]
