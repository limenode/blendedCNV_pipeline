"""Shared fixtures: one small synthetic call set, built once per session.

The registries in `callsets.registry` are **process-global**, so seeding the
chromosome registry is a session-level side effect rather than something each
test can do for itself. `seeded_registry` is autouse and every fixture that
builds a CallSet depends on it explicitly, so the ordering is guaranteed rather
than incidental.

Everything here is synthetic. The suite needs no `out/`, no config and no network,
which is what lets it run in a fraction of a second anywhere.
"""

import pytest

from consensuscnv.callsets import Call, collect_callsets
from consensuscnv.callsets.registry import CHROMOSOMES, seed_chromosomes

GENOME = ("chr1", "chr2")


@pytest.fixture(scope="session", autouse=True)
def seeded_registry() -> tuple[str, ...]:
    """Seed the process-global chromosome registry, exactly as a run would.

    The assertion is the point as much as the seeding: if anything interned a
    chromosome before this ran, ids would no longer order the genome and the row
    order tests below would be checking nothing.
    """
    seed_chromosomes(GENOME)
    assert list(CHROMOSOMES.names) == list(GENOME), (
        f"expected a registry holding exactly {GENOME}, found {CHROMOSOMES.names}. "
        "Something interned a chromosome before the registry was seeded."
    )
    return GENOME


@pytest.fixture(scope="session")
def synthetic_calls() -> list[Call]:
    """A call set built so every consensus level is non-empty at 0.5 overlap.

    chr2 is listed first on purpose: row order has to come from the genome file,
    not from the order the calls were handed over.
    """
    calls = [
        # chr2: one private call, plus a pair overlapping by only ~9% reciprocal,
        # which must stay two components at 0.5 and become one at 0.0.
        Call("chr2", 5_000, 7_000, "DEL", "gatk", "S1"),
        Call("chr2", 30_000, 31_000, "DEL", "cnvpytor", "S1"),
        Call("chr2", 30_900, 32_000, "DEL", "delly", "S1"),
    ]
    for sample in ("S1", "S2"):
        calls += [
            # all three callers agree: reciprocal overlap 1.00 and 0.975
            Call("chr1", 1_000, 3_000, "DEL", "cnvpytor", sample),
            Call("chr1", 1_000, 3_000, "DEL", "delly", sample),
            Call("chr1", 1_050, 3_050, "DEL", "gatk", sample),
        ]
    calls += [
        # exactly two callers agree: reciprocal overlap 0.95
        Call("chr1", 10_000, 12_000, "DUP", "cnvpytor", "S1"),
        Call("chr1", 10_100, 12_100, "DUP", "delly", "S1"),
        # one caller alone
        Call("chr1", 20_000, 22_000, "DEL", "cnvpytor", "S1"),
        # Three calls, two callers: CNVpytor reports the same event twice. This
        # component is the only thing separating `min_sources` from `min_calls`,
        # so without it a merge filtering on the wrong one passes every test.
        Call("chr1", 40_000, 42_000, "DEL", "cnvpytor", "S1"),
        Call("chr1", 40_100, 42_100, "DEL", "cnvpytor", "S1"),
        Call("chr1", 40_050, 42_050, "DEL", "delly", "S1"),
    ]
    return calls


@pytest.fixture(scope="session")
def callset(seeded_registry, synthetic_calls):
    """The synthetic calls as one built CallSet, with its overlap graph."""
    return collect_callsets([synthetic_calls], chromosome_order=seeded_registry)
