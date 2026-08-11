"""Variance-based sensitivity analysis of a parameter sweep.

A sweep over k parameters on a complete factorial grid produces a k-dimensional
field of metric values. This module answers two questions about such a field:

    - How much of the variation does each parameter account for?
      (Sobol' indices, `sobol_indices`)
    - Can the field be written as a sum of one-parameter curves, so that
      marginal plots are honest rather than merely convenient?
      (the functional ANOVA decomposition, `decompose`)

References
----------
Hoeffding (1948), *A class of statistics with asymptotically normal
    distribution*, Ann. Math. Statist. 19(3):293-325 -- the original
    decomposition.
Sobol' (1993), *Sensitivity estimates for nonlinear mathematical models*,
    Math. Model. Comput. Exp. 1(4):407-414 -- the variance indices.
Saltelli et al. (2008), *Global Sensitivity Analysis: The Primer* -- the
    standard practitioner treatment.
Owen (2013), *Variance components and generalized Sobol' indices*,
    SIAM/ASA J. Uncertain. Quantif. 1(1):19-41 -- the tidiest modern statement.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

# Beyond this many parameters the 2**k subset enumeration in `all_indices`
# stops being free. First- and total-order indices are linear in k and are
# unaffected.
MAX_FULL_DECOMPOSITION_NDIM = 12

# Columns of the TSV written by `SensitivityIndices.to_tsv`. One row per Sobol'
# term: `term_type` separates main effects from interactions from the derived
# aggregates, so a reader can filter to a clean rectangle of whichever it wants.
TSV_HEADER = (
    "metric",
    "term_type",
    "term",
    "order",
    "sobol_index",
    "total_index",
    "interaction_index",
)


def _validate_field(field: np.ndarray) -> np.ndarray:
    """Reject fields the decomposition would silently mis-handle."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim == 0:
        raise ValueError("field must have at least one axis; got a scalar")
    if not np.isfinite(field).all():
        n_bad = int((~np.isfinite(field)).sum())
        raise ValueError(
            f"field has {n_bad} non-finite cells. The decomposition needs a complete "
            "grid -- every marginal mean would be contaminated. Fill the gaps or "
            "slice them out before calling."
        )
    return field


def _axis_labels(ndim: int, factor_names: tuple[str, ...] | None) -> tuple[str, ...]:
    if factor_names is None:
        return tuple(f"x{i}" for i in range(ndim))
    if len(factor_names) != ndim:
        raise ValueError(f"got {len(factor_names)} names for a {ndim}-axis field")
    return tuple(factor_names)


# ---------------------------------------------------------------------------
# variance-based sensitivity analysis (Sobol' indices)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdditiveModel:
    """The field approximated as a grand mean plus one curve per parameter.

    `values` is the reconstruction mu + A(a) + B(b) + ...; `mains[i]` holds the
    centred effect of axis `i`, one number per level, and sums to zero.
    """

    grand: float
    mains: tuple[np.ndarray, ...]
    values: np.ndarray
    interaction_fraction: float

    @property
    def r_squared(self) -> float:
        """Fraction of variance the additive model reproduces."""
        return 1.0 - self.interaction_fraction

    @property
    def n_free_parameters(self) -> int:
        """Numbers needed to rebuild the field, after the sum-to-zero constraints."""
        return 1 + sum(len(m) - 1 for m in self.mains)

    def partial_dependence(self, axis: int) -> np.ndarray:
        """The marginal profile of one axis, on the scale of the field itself.

        This is the partial dependence function of the ML interpretability
        literature, and it is what a one-parameter line plot should show.
        """
        return self.grand + self.mains[axis]

    def residuals(self, field: np.ndarray) -> np.ndarray:
        """field - values: everything the one-parameter curves cannot express.

        The residual of an interaction is not confined to the cells that
        interact, because a single anomalous cell shifts the marginal means of
        its whole row and column. Read the summed square, not single entries.
        """
        return np.asarray(field, dtype=np.float64) - self.values


def main_effects(field: np.ndarray) -> tuple[float, tuple[np.ndarray, ...]]:
    """Grand mean and the centred main effect of each axis.

    `mean(axis=...)` collapses the axes named; the tuple here names every axis
    *except* `ax`, so what survives is one number per level of `ax`.
    """
    field = _validate_field(field)
    grand = float(field.mean())
    mains = tuple(
        field.mean(axis=tuple(a for a in range(field.ndim) if a != ax)) - grand
        for ax in range(field.ndim)
    )
    return grand, mains


def decompose(field: np.ndarray) -> AdditiveModel:
    """Split the field into a grand mean, one curve per axis, and a remainder."""
    field = _validate_field(field)
    grand, mains = main_effects(field)

    # reshape writes -1 in the axis's own slot and 1 elsewhere, so each length-n
    # vector becomes (n,1,1)-shaped and broadcasting adds them into a full grid.
    values = grand + sum(
        effect.reshape([-1 if a == ax else 1 for a in range(field.ndim)])
        for ax, effect in enumerate(mains)
    )

    total_ss = float(((field - grand) ** 2).sum())
    resid_ss = float(((field - values) ** 2).sum())
    interaction_fraction = resid_ss / total_ss if total_ss > 0.0 else 0.0
    return AdditiveModel(grand, mains, values, interaction_fraction)   # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Sobol' indices
# ---------------------------------------------------------------------------


def closed_index(field: np.ndarray, subset: tuple[int, ...]) -> float:
    """Var(E[Y | X_subset]) / Var(Y), the *closed* index of a set of axes.

    Averaging over the complement is exactly conditioning on `subset`, because
    every cell of a complete factorial grid carries equal weight.
    """
    field = _validate_field(field)
    total_variance = float(field.var())
    if total_variance == 0.0:
        return 0.0
    complement = tuple(a for a in range(field.ndim) if a not in subset)
    conditional = field.mean(axis=complement) if complement else field
    return float(conditional.var() / total_variance)  # ddof=0: population variance


def first_order_indices(field: np.ndarray) -> np.ndarray:
    """S_i for each axis: the variance that parameter explains acting alone."""
    field = _validate_field(field)
    return np.array([closed_index(field, (i,)) for i in range(field.ndim)])


def total_order_indices(field: np.ndarray) -> np.ndarray:
    """S_Ti for each axis: variance involving it, alone or in any interaction.

    S_Ti - S_i is the share a parameter carries only in company. Where that gap
    is small, the one-parameter plot of that axis tells its whole story.
    """
    field = _validate_field(field)
    return np.array(
        [
            1.0 - closed_index(field, tuple(a for a in range(field.ndim) if a != i))
            for i in range(field.ndim)
        ]
    )


def all_indices(field: np.ndarray) -> dict[tuple[int, ...], float]:
    """Every S_u, for all non-empty subsets of the axes. Sums to exactly 1.

    Moebius inversion of the closed indices. Enumerates 2**ndim subsets, so it
    is meant for the handful of parameters a sweep like this has.
    """
    field = _validate_field(field)
    if field.ndim > MAX_FULL_DECOMPOSITION_NDIM:
        raise ValueError(
            f"full decomposition enumerates 2**{field.ndim} subsets; use "
            "first_order_indices/total_order_indices instead"
        )
    axes = tuple(range(field.ndim))
    closed: dict[tuple[int, ...], float] = {(): 0.0}
    for size in range(1, field.ndim + 1):
        for subset in combinations(axes, size):
            closed[subset] = closed_index(field, subset)
    return {
        subset: sum(
            (-1) ** (len(subset) - len(lower)) * closed[lower]
            for size in range(len(subset) + 1)
            for lower in combinations(subset, size)
        )
        for subset in closed
        if subset
    }


@dataclass(frozen=True, slots=True)
class SensitivityIndices:
    """Sobol' indices for one metric field, ready to report."""

    factor_names: tuple[str, ...]
    first_order: np.ndarray
    total_order: np.ndarray
    interactions: dict[tuple[int, ...], float]

    @property
    def additive_fraction(self) -> float:
        """Variance explained by the parameters acting independently."""
        return float(self.first_order.sum())

    @property
    def interaction_fraction(self) -> float:
        """Variance that only appears when parameters are considered jointly."""
        return 1.0 - self.additive_fraction

    @property
    def dominant(self) -> str:
        """Name of the parameter with the largest first-order index."""
        return self.factor_names[int(np.argmax(self.first_order))]

    def label(self, subset: tuple[int, ...]) -> str:
        return " x ".join(self.factor_names[i] for i in subset)

    def rows(self, metric_name: str = "metric") -> list[tuple[str, ...]]:
        """One row per Sobol' term, in `TSV_HEADER` order, formatted as strings.

        Main-effect rows carry all three of S_i, S_Ti and their difference.
        Interaction rows have no total-order counterpart, so those two fields
        are left empty rather than filled with a misleading zero.
        """
        rows: list[tuple[str, ...]] = []
        for i, name in enumerate(self.factor_names):
            s, t = float(self.first_order[i]), float(self.total_order[i])
            rows.append(
                (metric_name, "main", name, "1", f"{s:.6f}", f"{t:.6f}", f"{t - s:.6f}")
            )
        for subset, value in sorted(self.interactions.items(), key=lambda kv: -kv[1]):
            rows.append(
                (metric_name, "interaction", self.label(subset), str(len(subset)),
                 f"{value:.6f}", "", "")
            )
        for name, value in (
            ("sum of first-order", self.additive_fraction),
            ("total interaction", self.interaction_fraction),
        ):
            rows.append((metric_name, "summary", name, "", f"{value:.6f}", "", ""))
        return rows

    def to_tsv(self, path: str | Path, metric_name: str = "metric") -> int:
        """Write the indices to `path` as TSV, returning the number of data rows.

        Indices are written as fractions of total variance, not percentages, so
        that they stay directly usable downstream; the main-effect rows and the
        interaction rows together sum to 1.
        """
        return write_sensitivity_tsv(path, {metric_name: self})

    def __str__(self) -> str:
        shares = ", ".join(
            f"{n} {s:.1%}" for n, s in zip(self.factor_names, self.first_order)
        )
        return f"first-order: {shares}; interactions {self.interaction_fraction:.1%}"


def write_sensitivity_tsv(
    path: str | Path,
    indices: Mapping[str, SensitivityIndices],
) -> int:
    """Write one or more metrics' indices to a single TSV, returning row count.

    Keys of `indices` become the `metric` column, so several metrics can share
    one supplementary table. Parent directories are not created.
    """
    rows = [row for metric_name, idx in indices.items() for row in idx.rows(metric_name)]
    lines = ["\t".join(TSV_HEADER)] + ["\t".join(row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n")
    return len(rows)


def sobol_indices(
    field: np.ndarray,
    factor_names: tuple[str, ...] | None = None,
    include_interactions: bool = True,
) -> SensitivityIndices:
    """First-order, total-order, and interaction Sobol' indices for a sweep field.

    Parameters
    ----------
    field:
        One metric evaluated on a complete factorial grid, one axis per swept
        parameter. Must be finite everywhere.
    factor_names:
        Parameter names in axis order, used for reporting. Defaults to x0, x1...
    include_interactions:
        Enumerate the 2**ndim subsets to break the interaction share down by
        which parameters are involved. Turn off for a wide sweep.

    Notes
    -----
    A constant field has no variance to attribute, and every index comes back
    zero rather than 0/0.
    """
    field = _validate_field(field)
    interactions: dict[tuple[int, ...], float] = {}
    if include_interactions:
        interactions = {u: v for u, v in all_indices(field).items() if len(u) > 1}
    return SensitivityIndices(
        factor_names=_axis_labels(field.ndim, factor_names),
        first_order=first_order_indices(field),
        total_order=total_order_indices(field),
        interactions=interactions,
    )
