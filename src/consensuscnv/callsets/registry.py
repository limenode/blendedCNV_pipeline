"""Process-wide name -> id tables."""

import threading

DEFAULT_CHROMOSOME_ORDER = [f"chr{i}" for i in range(1, 23)]

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

CHROMOSOMES = Registry("chromosome", DEFAULT_CHROMOSOME_ORDER)
SVTYPES = Registry("svtype", ["DEL", "DUP"])
SAMPLES = Registry("sample")
SOURCES = Registry("source", max_ids=MAX_SOURCES)

__all__ = ["CHROMOSOMES", "DEFAULT_CHROMOSOME_ORDER", "MAX_SOURCES", "SAMPLES", "SOURCES", "SVTYPES", "Registry"]
