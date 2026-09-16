"""The registry lifecycle: ids order the genome, a later genome must agree, and
`reset_registries` starts over.

The registries are process-global and the session fixtures hold ids from them,
so nothing here may reset in-process; the reset is exercised in a subprocess.
"""

import subprocess
import sys

import pytest

from consensuscnv.callsets.registry import CHROMOSOMES, Registry, seed_chromosomes
from consensuscnv.classification import IntervalSet


def test_registry_clear_forgets_and_reseeds():
    registry = Registry("test", ["a", "b"])
    assert registry.get("b") == 1
    registry.clear(["z"])
    assert registry.names == ["z"] and registry.get("b") == -1


def test_reseeding_a_compatible_genome_is_a_no_op(seeded_registry):
    before = list(CHROMOSOMES.names)
    seed_chromosomes(seeded_registry)          # the same genome again
    seed_chromosomes(seeded_registry[1:])      # a subset, in the same order
    assert list(CHROMOSOMES.names) == before


def test_reseeding_a_conflicting_order_raises_before_interning(seeded_registry):
    """A genome the existing ids cannot order would sort right but write wrong."""
    before = list(CHROMOSOMES.names)
    with pytest.raises(ValueError, match="orders these names differently"):
        seed_chromosomes(tuple(reversed(seeded_registry)))
    with pytest.raises(ValueError, match="orders these names differently"):
        seed_chromosomes((seeded_registry[0], "chrNew", seeded_registry[1]))  # insertion
    with pytest.raises(ValueError, match="orders these names differently"):
        IntervalSet.from_records([("chr1", 1, 2)], genome=tuple(reversed(seeded_registry)))
    assert list(CHROMOSOMES.names) == before   # nothing was interned on the way to raising


def test_reset_registries_starts_over():
    script = """
from consensuscnv.callsets.registry import CHROMOSOMES, SAMPLES, SVTYPES, reset_registries
from consensuscnv.classification import IntervalSet
IntervalSet.from_records([("chr2", 1, 2, "DEL", "x", "S1")], genome=("chr2", "chr1"))
assert CHROMOSOMES.names == ["chr2", "chr1"] and SAMPLES.names == ["S1"]
reset_registries()
assert CHROMOSOMES.names == [] and SAMPLES.names == [] and SVTYPES.names == ["DEL", "DUP"]
IntervalSet.from_records([("chr2", 1, 2)], genome=("chr1", "chr2"))   # would have raised before the reset
assert CHROMOSOMES.names == ["chr1", "chr2"]
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
