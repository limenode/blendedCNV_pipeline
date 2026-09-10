"""One table describing what the exclusion mask removed, across every input type.

The three parsers count slightly different things -- experimental VCFs are
per-sample, benchmark VCFs are multi-sample and expand one record into one call
per carrier, PennCNV records are already per-sample -- so they cannot simply be
concatenated. `exclusion_summary` normalises all three onto a common per-call
schema, one row per (input type, dataset, source).

`collateral` is total bases removed divided by the bases actually inside the mask.
Whole calls are dropped rather than trimmed, so this is what that policy costs.
A value near 1.0 means the dropped calls sat almost entirely inside the mask to
begin with and the policy is close to free; a large value would mean the mask is
amputating mostly-clean calls.
"""

import pandas as pd

# Per-call columns every parser reports under its own name, mapped to the common
# schema. `total_base_count` is tallied before liftover for experimental VCFs and
# after it for the other two, so treat cross-type base comparisons as approximate.
_COMMON_COLUMNS = (
    "total_call_count",
    "total_del_call_count",
    "total_dup_call_count",
    "total_base_count",
    "calls_removed_excluded",
    "calls_del_removed_excluded",
    "calls_dup_removed_excluded",
    "bases_removed_excluded",
    "bases_masked_excluded",
)

SUMMARY_COLUMNS = (
    "input_type",
    "dataset",
    "source",
    "n_calls",
    "n_removed",
    "pct_removed",
    "pct_del_removed",
    "pct_dup_removed",
    "mb_removed",
    "mb_masked",
    "pct_bases_removed",
    "collateral",
    "mean_removed_kb",
)

# The compact view. The rest stay on the frame for anyone who wants them.
_DISPLAY_COLUMNS = (
    "input_type", "dataset", "source", "n_calls", "n_removed", "pct_removed",
    "pct_del_removed", "pct_dup_removed", "mb_removed", "collateral",
)


def _safe_div(numerator, denominator):
    """Elementwise divide, yielding NaN rather than inf where the denominator is 0."""
    return numerator / denominator.where(denominator != 0)


def _normalise(frame: pd.DataFrame, input_type: str) -> pd.DataFrame:
    """Reindex onto `_COMMON_COLUMNS`, filling in counters a parser never reported."""
    out = frame.reindex(columns=list(_COMMON_COLUMNS)).fillna(0)
    out.insert(0, "input_type", input_type)
    return out


def exclusion_summary(
    vcf_statistics: pd.DataFrame,
    penncnv_statistics: pd.DataFrame,
    benchmark_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the three parsers' statistics into one exclusion-mask table.

    Experimental VCFs arrive one row per sample and are summed per tool, since
    the mask is a property of the genome rather than of any one sample. The
    benchmark and control frames arrive one row per dataset already.
    """
    parts = []

    if not vcf_statistics.empty:
        grouped = (
            vcf_statistics.groupby(["experimental_name", "tool"], as_index=False)
            .sum(numeric_only=True)
            .rename(columns={"experimental_name": "dataset", "tool": "source"})
        )
        parts.append(
            pd.concat(
                [grouped[["dataset", "source"]], _normalise(grouped, "experimental")], axis=1
            )
        )

    for frame, input_type in (
        (penncnv_statistics, "control"),
        (benchmark_statistics, "benchmark"),
    ):
        if frame.empty:
            continue
        # These two are indexed by dataset name; the name is both dataset and source.
        named = frame.reset_index(names="dataset")
        named["source"] = named["dataset"]
        parts.append(
            pd.concat(
                [named[["dataset", "source"]], _normalise(named, input_type)], axis=1
            )
        )

    if not parts:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    table = pd.concat(parts, ignore_index=True)

    table["n_calls"] = table["total_call_count"]
    table["n_removed"] = table["calls_removed_excluded"]
    table["pct_removed"] = _safe_div(table["n_removed"], table["n_calls"])
    table["pct_del_removed"] = _safe_div(
        table["calls_del_removed_excluded"], table["total_del_call_count"]
    )
    table["pct_dup_removed"] = _safe_div(
        table["calls_dup_removed_excluded"], table["total_dup_call_count"]
    )
    table["mb_removed"] = table["bases_removed_excluded"] / 1e6
    table["mb_masked"] = table["bases_masked_excluded"] / 1e6
    table["pct_bases_removed"] = _safe_div(
        table["bases_removed_excluded"], table["total_base_count"]
    )
    # How many bases leave per base of masked sequence -- the cost of dropping
    # whole calls instead of trimming them.
    table["collateral"] = _safe_div(
        table["bases_removed_excluded"], table["bases_masked_excluded"]
    )
    table["mean_removed_kb"] = _safe_div(table["bases_removed_excluded"], table["n_removed"]) / 1e3

    ordered = table[list(SUMMARY_COLUMNS)].copy()
    for column in ("n_calls", "n_removed"):
        ordered[column] = ordered[column].astype("int64")
    return ordered.sort_values(["input_type", "dataset", "source"], ignore_index=True)


def format_exclusion_summary(summary: pd.DataFrame, full: bool = False) -> str:
    """Render `exclusion_summary` for a terminal or a log file."""
    if summary.empty:
        return "Exclusion mask: nothing removed (no mask configured, or no overlaps)."

    shown = summary if full else summary[list(_DISPLAY_COLUMNS)]
    formatters = {
        "n_calls": "{:,.0f}".format,
        "n_removed": "{:,.0f}".format,
        "mb_removed": "{:,.1f}".format,
        "mb_masked": "{:,.1f}".format,
        "mean_removed_kb": "{:,.1f}".format,
        "collateral": "{:.2f}".format,
        **{c: "{:.1%}".format for c in
           ("pct_removed", "pct_del_removed", "pct_dup_removed", "pct_bases_removed")},
    }
    body = shown.to_string(
        index=False,
        na_rep="-",
        formatters={k: v for k, v in formatters.items() if k in shown.columns},
    )
    total_removed = summary["n_removed"].sum()
    total_calls = summary["n_calls"].sum()
    return (
        f"Exclusion mask removed {total_removed:,} of {total_calls:,} calls "
        f"({total_removed / max(total_calls, 1):.1%}) across "
        f"{len(summary)} input sets\n\n{body}"
    )
