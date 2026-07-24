import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    ## imports
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import glob
    from line_profiler import LineProfiler
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from pathlib import Path

    # Jupyter-like light figures: marimo's dark theme otherwise shows through
    # matplotlib's transparent figure patch and leaves axis text unreadable.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    return Path, np, os, pd, plt, sns


@app.cell
def _():
    import consensuscnv
    import consensuscnv.utils
    import consensuscnv.analysis_assist as analysis_assist
    from consensuscnv.analysis_assist import SampleClassification

    return SampleClassification, analysis_assist, consensuscnv


@app.cell
def _():
    from consensuscnv import (
        parsing,
    )

    return (parsing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## prep
    """)
    return


@app.cell
def _(Path, consensuscnv, os, parsing):
    config = consensuscnv.utils.build_config(
            config_path=Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/internal.config.yaml"),
            do_processing=True,
            do_analysis=True,
            do_computation=True,
        )
    layout = config.layout

    # Check path for if parsing is needed 
    path_to_test_parsing = "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage/bed/cnvpytor"
    if os.path.exists(path_to_test_parsing):
        print("Parsing is not needed, files already exist.")
    else:
        print("Parsing is needed, running parsing function.")
        # Run parsing function
        parsing.parse_input_files(config=config)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## binary classification statistics
    """)
    return


@app.cell
def _(Path, analysis_assist):
    binary_classification_root = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/binary_classification")
    classification_tree = analysis_assist.load_binary_classification(binary_classification_root)
    raw_summary = analysis_assist.summarize(classification_tree)
    return classification_tree, raw_summary


@app.cell
def _(raw_summary):
    raw_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## metrics vs padding
    """)
    return


@app.cell
def _(raw_summary):
    # get list of samples that appear in every call set
    def get_common_samples(summary):
        call_sets = summary["call_set"].unique()
        sample_sets = [set(summary[summary["call_set"] == cs]["sample"]) for cs in call_sets]
        common_samples = set.intersection(*sample_sets)
        return common_samples

    summary = raw_summary[raw_summary["sample"].isin(get_common_samples(raw_summary))]
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.function
def padding_bp(setting: str) -> int:
    """Numeric padding (bp) parsed from a 'pad<N>' setting name."""
    return int(setting.split("_")[1].removeprefix("pad") or 0)


@app.cell
def _(summary):
    # Pool per-sample counts into one row
    padding_metrics = summary.groupby(
        ["benchmark_setting","classification_setting", "input_set", "call_set"], as_index=False
    ).agg(TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"))
    padding_metrics["padding"] = padding_metrics["benchmark_setting"].map(padding_bp)
    padding_metrics["precision"] = padding_metrics["TP"] / (
        padding_metrics["TP"] + padding_metrics["FP"]
    )
    padding_metrics["recall"] = padding_metrics["TP"] / (
        padding_metrics["TP"] + padding_metrics["FN"]
    )
    padding_metrics["f1"] = 2 * (padding_metrics["precision"] * padding_metrics["recall"]) / (
        padding_metrics["precision"] + padding_metrics["recall"]
    )
    padding_metrics = padding_metrics.sort_values(
        ["input_set", "call_set", "padding"]
    ).reset_index(drop=True)
    return (padding_metrics,)


@app.cell
def _(padding_metrics):
    padding_metrics
    return


@app.cell
def select_rows(pd):
    def select_rows(
        df,
        input_set: str | list | None = None,
        classification_setting: str | list | None = None,
        call_set: str | list | None = None,
    ) -> pd.DataFrame:
        """Filter a padding_metrics-style DataFrame by the three sweep dimensions.

        Each argument is a single value, a list of values, or None (keep all). E.g.
        select_rows(padding_metrics, input_set="30x_Coverage", classification_setting="classify_recip0.5")
        keeps 30x @ recip0.5 across every padding and call set.
        """
        wanted = {
            "input_set": input_set,
            "classification_setting": classification_setting,
            "call_set": call_set,
        }
        mask = df["padding"].notna()  # all-True seed
        for column, value in wanted.items():
            if value is None:
                continue
            allowed = [value] if isinstance(value, str) else list(value)
            mask &= df[column].isin(allowed)
        return df[mask]

    return (select_rows,)


@app.cell
def _(plt, sns):
    def plot_metric_vs_padding(df, metric, hue=None, style=None) -> plt.Figure:
        """One metric vs padding; colour/dash encode whichever dims still vary.

        `hue` and `style` may name any sweep dimension. When left None they are
        auto-assigned to the *free* dimensions — those with more than one value
        left in `df` — so after select_rows() pins a dimension it stops cluttering
        the legend. Filter first, then hand the slice here.
        """
        sweep_dims = ["input_set", "call_set", "classification_setting"]
        free = [d for d in sweep_dims if df[d].nunique() > 1]
        if hue is None:
            hue = free[0] if free else sweep_dims[0]
        if style is None:
            rest = [d for d in free if d != hue]
            style = rest[0] if rest else None

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = sns.color_palette("tab10", n_colors=max(df[hue].nunique(), 1))
        sns.lineplot(
            data=df,
            x="padding",
            y=metric,
            hue=hue,
            style=style,
            markers=True,
            dashes=True,
            ax=ax,
            palette=colors,
        )
        style_note = f", dash={style}" if style else ""
        ax.set_title(f"{metric} vs benchmark padding (colour={hue}{style_note})")
        ax.set_xlabel("padding (bp)")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(df["padding"].unique()))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        fig.tight_layout()
        return fig

    return (plot_metric_vs_padding,)


@app.function
def recip_value(setting: str) -> float:
    """Numeric reciprocal threshold parsed from a 'classify_recip<x>' setting name."""
    digits = "".join(ch for ch in setting if (ch.isdigit() or ch == "."))
    return float(digits) if digits else 0.0


@app.cell
def _(padding_metrics, plot_metric_vs_padding):
    padding_figures = {
        metric: plot_metric_vs_padding(padding_metrics, metric)
        for metric in ("TP", "FP", "FN", "precision", "recall", "f1")
    }
    return (padding_figures,)


@app.cell
def _(Path, os, padding_figures):
    # output to figures directory
    figures_dir = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/figures")
    for _metric, _fig in padding_figures.items():
        _fig_path = figures_dir / "vs_padding" / f"{_metric}_vs_padding.png"
        os.makedirs(_fig_path.parent, exist_ok=True)
        _fig.savefig(_fig_path)
    return (figures_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### flexible slices: metric vs padding

    `select_rows(padding_metrics, input_set=..., classification_setting=..., call_set=...)`
    pins any of the three sweep dimensions (single value, list, or None=all). Whatever
    is still varying gets auto-encoded as colour (hue) then dash (style) by
    `plot_metric_vs_padding`; padding stays on the x-axis.
    """)
    return


@app.cell
def _(padding_metrics, plot_metric_vs_padding, select_rows):
    # Scenario 1: one input set + one classification setting, ALL call sets.
    _slice = select_rows(
        padding_metrics,
        input_set="30x_Coverage",
        classification_setting="classify_recip0.5",
    )
    # Only call_set is free -> it becomes the colour.
    plot_metric_vs_padding(_slice, "f1")
    return


@app.cell
def _(padding_metrics, plot_metric_vs_padding, select_rows):
    # Scenario 2: one input set + a few call sets, ALL classification settings.
    _slice = select_rows(
        padding_metrics,
        input_set="30x_Coverage",
        call_set=["consensus_2of3_w0.5", "calls"],
    )
    # call_set and classification_setting are free -> colour + dash.
    plot_metric_vs_padding(_slice, "f1", hue="call_set", style="classification_setting")
    return


@app.cell
def _(padding_metrics, plot_metric_vs_padding, select_rows):
    # Scenario 3: one classification setting + a few call sets, ALL input sets.
    _slice = select_rows(
        padding_metrics,
        classification_setting="classify_recip0.5",
        call_set=["consensus_2of3_w0.5", "calls"],
    )
    plot_metric_vs_padding(_slice, "f1", hue="input_set", style="call_set")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### interactive 3D: padding × reciprocal threshold × metric
    """)
    return


@app.cell
def _(padding_metrics, select_rows):
    def plot_metric_3d(df, metric):
        """3D lines over the (padding, reciprocal-threshold) grid.

        One Scatter3d trace per (input_set, call_set, classification_setting) — each
        line holds its reciprocal threshold fixed (constant y) and walks padding, so
        the threshold levels read as separate parallel curves instead of one line
        zig-zagging through them all. Returns a plotly Figure; marimo renders it live.
        """
        import plotly.graph_objects as go

        df = df.copy()
        df["recip"] = df["classification_setting"].map(recip_value)
        fig = go.Figure()
        group_cols = ["input_set", "call_set", "classification_setting"]
        for (input_set, call_set, _setting), group in df.groupby(group_cols):
            group = group.sort_values("padding")
            recip = group["recip"].iloc[0]
            fig.add_trace(
                go.Scatter3d(
                    x=group["padding"],
                    y=group["recip"],
                    z=group[metric],
                    mode="lines+markers",
                    marker={"size": 3},
                    line={"width": 3},
                    name=f"{input_set} / {call_set} @ recip{recip:g}",
                )
            )
        fig.update_layout(
            title=f"{metric}: padding × reciprocal threshold",
            scene={
                "xaxis": {"title": "padding (bp)"},
                "yaxis": {"title": "reciprocal threshold"},
                "zaxis": {"title": metric},
            },
            height=700,
            width=1000,
            legend={"font": {"size": 9}},
        )
        return fig

    # A few sets keep the 3D view readable.
    _slice = select_rows(
        padding_metrics,
        input_set=["30x_Coverage", "6x_Coverage", "4x_Coverage", "2x_Coverage"],
        call_set=["consensus_2of3_w0.5", "calls"],
    )
    plot_metric_3d(_slice, "f1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## metrics vs CNV size
    """)
    return


@app.cell
def _(np, pd):
    def interval_sizes(df: pd.DataFrame, prefix=""):
        """CNV sizes (bp) from a calls DataFrame; prefix='truth_' for matched truth intervals."""
        start, end = f"{prefix}start", f"{prefix}end"
        if df.empty or start not in df.columns:
            return np.empty(0, dtype=int)
        return (df[end] - df[start]).to_numpy()

    return (interval_sizes,)


@app.cell
def _(SampleClassification, interval_sizes, np):
    def pool_sizes(samples: list[SampleClassification]) -> dict[str, np.ndarray]:
        """Pool CNV sizes (bp) across a call set's samples.

        Precision is call-based (TP query size vs FP); recall is truth-based
        (TP truth size vs FN). Returns arrays keyed tp_query, fp, tp_truth, fn.
        """
        def cat(arrs):
            arrs = [a for a in arrs if a.size]
            return np.concatenate(arrs) if arrs else np.empty(0, dtype=int)

        return {
            "tp_query": np.sort(cat([interval_sizes(s.tp) for s in samples])),
            "fp": np.sort(cat([interval_sizes(s.fp) for s in samples])),
            "tp_truth": np.sort(cat([interval_sizes(s.tp, "truth_") for s in samples])),
            "fn": np.sort(cat([interval_sizes(s.fn) for s in samples])),
        }

    return (pool_sizes,)


@app.cell
def _(np):
    def precision(tp, fp) -> float:
        """Precision = TP / (TP + FP)"""
        return tp / (tp + fp) if tp + fp > 0 else np.nan

    def recall(tp, fn) -> float:
        """Recall = TP / (TP + FN)"""
        return tp / (tp + fn) if tp + fn > 0 else np.nan

    def f1_score(tp, fp, fn) -> float:
        """F1 = 2 * TP / (2 * TP + FP + FN)"""
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan


    return f1_score, precision, recall


@app.cell
def _(f1_score, np, precision, recall):
    def sweep_size_thresholds(sizes) -> dict[str, list[tuple[int, float]]]:
        """Cumulative precision/recall/F1 across CNV size thresholds.

        Walks the pooled sizes in ascending order; the counters track calls with
        size <= the current threshold, and each metric is recorded whenever one of
        its inputs changes (TP/FP for precision, TP/FN for recall, any for F1).
        Returns lists keyed precision, recall, f1.
        """
        tp_counter, fp_counter, fn_counter = 0, 0, 0
        precision_list, recall_list, f1_list = [], [], []
        tp_sizes, fp_sizes, fn_sizes = sizes["tp_query"], sizes["fp"], sizes["fn"]

        while True:
            tp_next_size = tp_sizes[tp_counter] if len(tp_sizes) > tp_counter else np.inf
            fp_next_size = fp_sizes[fp_counter] if len(fp_sizes) > fp_counter else np.inf
            fn_next_size = fn_sizes[fn_counter] if len(fn_sizes) > fn_counter else np.inf
            min_next_size = min(tp_next_size, fp_next_size, fn_next_size)
            if min_next_size == np.inf:
                break

            tp_change, fp_change, fn_change = False, False, False
            if tp_next_size == min_next_size:
                tp_change = True
                while len(tp_sizes) > tp_counter and tp_sizes[tp_counter] == min_next_size:
                    tp_counter += 1
            if fp_next_size == min_next_size:
                fp_change = True
                while len(fp_sizes) > fp_counter and fp_sizes[fp_counter] == min_next_size:
                    fp_counter += 1
            if fn_next_size == min_next_size:
                fn_change = True
                while len(fn_sizes) > fn_counter and fn_sizes[fn_counter] == min_next_size:
                    fn_counter += 1

            if tp_change or fp_change:
                precision_list.append((min_next_size, precision(tp_counter, fp_counter)))
            if tp_change or fn_change:
                recall_list.append((min_next_size, recall(tp_counter, fn_counter)))
            if tp_change or fp_change or fn_change:
                f1_list.append((min_next_size, f1_score(tp_counter, fp_counter, fn_counter)))

        return {"precision": precision_list, "recall": recall_list, "f1": f1_list}

    return (sweep_size_thresholds,)


@app.cell
def _(classification_tree, pool_sizes, sweep_size_thresholds):
    # Walk the 4-level tree (bench -> classify -> input_set -> call_set) to one
    # representative leaf without hardcoding slug names.
    _bench = next(iter(classification_tree))
    _classify = next(iter(classification_tree[_bench]))
    _input_set = next(iter(classification_tree[_bench][_classify]))
    _call_set = next(iter(classification_tree[_bench][_classify][_input_set]))
    _sizes = pool_sizes(classification_tree[_bench][_classify][_input_set][_call_set])
    _distributions = sweep_size_thresholds(_sizes)
    precision_list = _distributions["precision"]
    recall_list = _distributions["recall"]
    f1_list = _distributions["f1"]
    return f1_list, precision_list, recall_list


@app.cell
def _(f1_list, precision_list, recall_list):
    print(f"precision_list length: {len(precision_list)}")
    print(f"recall_list length: {len(recall_list)}")
    print(f"f1_list length: {len(f1_list)}")
    return


@app.cell
def _(classification_tree, pool_sizes, sweep_size_thresholds):
    metrics_dict = {}
    for benchmark_setting, classify_dict in classification_tree.items():
        for classification_setting, input_set_dict in classify_dict.items():
            for input_set_name, call_set_dict in input_set_dict.items():
                for call_set_name, samples in call_set_dict.items():
                    _sizes = pool_sizes(samples)
                    distributions = sweep_size_thresholds(_sizes)
                    key = (benchmark_setting, classification_setting, input_set_name, call_set_name)
                    metrics_dict[key] = distributions
    return (metrics_dict,)


@app.function
def select_metrics(
    metrics_by_combo: dict,
    benchmark_setting: str | list | None = None,
    classification_setting: str | list | None = None,
    input_set: str | list | None = None,
    call_set: str | list | None = None,
) -> dict:
    """Slice the (benchmark_setting, classification_setting, input_set, call_set) -> distributions dict.

    Each filter is a single value, a list of values, or None (keep all). E.g.
    select_metrics(metrics_dict, input_set="30x_Coverage") keeps only 30x across
    every padding, classification setting, and call set.
    """
    def keep(value, wanted):
        if wanted is None:
            return True
        return value == wanted if isinstance(wanted, str) else value in wanted

    return {
        key: distributions
        for key, distributions in metrics_by_combo.items()
        if keep(key[0], benchmark_setting)
        and keep(key[1], classification_setting)
        and keep(key[2], input_set)
        and keep(key[3], call_set)
    }


@app.cell
def _(plt):
    def plot_size_vs_metric(
        metrics_by_combo: dict, hue="input_set", style="call_set", marker=None, title=""
    ) -> dict:
        """Size vs. precision/recall/F1, each metric on its OWN figure.

        Returns ``{metric: Figure}`` (keys 'precision', 'recall', 'f1'). Up to three
        key dimensions can be encoded: `hue` -> colour, `style` -> dash pattern,
        `marker` -> point symbol (optional). Each names one of 'benchmark_setting',
        'classification_setting', 'input_set', or 'call_set'. Any dimension not
        mapped is left overlaid. Pre-filter the dict with select_metrics to thin
        the lines.

        Layout is handed to matplotlib's constrained layout with figure-level
        "outside right" legends, so the axes shrink to fit the legends and
        nothing clips regardless of how long the setting slugs are.
        """
        from matplotlib.lines import Line2D

        dimension_index = {
            "benchmark_setting": 0,
            "classification_setting": 1,
            "input_set": 2,
            "call_set": 3,
        }
        hue_i, style_i = dimension_index[hue], dimension_index[style]
        marker_i = dimension_index[marker] if marker else None

        def natural_key(value):
            # Order 'padding0 < padding500 < ...' and '2x < 4x < 30x' numerically.
            digits = "".join(ch for ch in value if ch.isdigit())
            return (int(digits) if digits else 0, value)

        def sorted_values(index):
            return sorted({key[index] for key in metrics_by_combo}, key=natural_key)

        def short_labels(values):
            """Drop the '_'-delimited tokens every value in a dimension shares.

            'bench_pad0_mn1_mw0_lssT' / 'bench_pad500_mn1_mw0_lssT' -> 'pad0' /
            'pad500'; '2x_Coverage' / '30x_Coverage' -> '2x' / '30x'. Values with
            nothing in common are left alone. The legend title already names the
            dimension, so the shared tokens are pure noise.
            """
            parts = [v.split("_") for v in values]
            if len(parts) < 2:
                return dict(zip(values, values))
            lead = 0
            while lead < min(len(p) for p in parts) - 1 and len({p[lead] for p in parts}) == 1:
                lead += 1
            trail = 0
            while (
                trail < min(len(p) for p in parts) - lead - 1
                and len({p[-1 - trail] for p in parts}) == 1
            ):
                trail += 1
            kept = [p[lead : len(p) - trail] for p in parts]
            return {v: "_".join(k) for v, k in zip(values, kept)}

        hue_values = sorted_values(hue_i)
        style_values = sorted_values(style_i)
        colors = dict(zip(hue_values, plt.cm.tab10.colors))
        dash_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
        dashes = dict(zip(style_values, dash_styles))

        marker_symbols = ["o", "s", "^", "D", "v", "P", "X", "*"]
        marker_values = sorted_values(marker_i) if marker_i is not None else []
        markers = dict(zip(marker_values, marker_symbols))

        hue_labels = short_labels(hue_values)
        style_labels = short_labels(style_values)
        marker_labels = short_labels(marker_values)

        color_handles = [
            Line2D([], [], color=colors[v], label=hue_labels[v]) for v in hue_values
        ]
        dash_handles = [
            Line2D([], [], color="0.35", linestyle=dashes[v], label=style_labels[v])
            for v in style_values
        ]
        marker_handles = [
            Line2D([], [], color="0.35", linestyle="none", marker=markers[v], label=marker_labels[v])
            for v in marker_values
        ]

        # (handles, title, loc) for each legend, stacked down the right of the figure.
        # Constrained layout reserves the space these need.
        legend_specs = [(color_handles, hue, "outside right upper")]
        if marker_handles:
            legend_specs.append((marker_handles, marker, "outside right center"))
        legend_specs.append((dash_handles, style, "outside right lower"))

        figures = {}
        for metric in ["precision", "recall", "f1"]:
            fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
            for key, distributions in metrics_by_combo.items():
                pairs = distributions[metric]
                if not pairs:
                    continue
                # pairs are (size, value) in ascending size order from the sweep.
                ax.plot(
                    [size for size, _ in pairs],
                    [value for _, value in pairs],
                    color=colors[key[hue_i]],
                    linestyle=dashes[key[style_i]],
                    marker=markers[key[marker_i]] if marker_i is not None else None,
                    markersize=5,
                    markevery=0.12,
                    linewidth=1.0,
                    alpha=0.6,
                )
            ax.set_xscale("log")
            ax.set_title(f"{title} — {metric}" if title else metric)
            ax.set_xlabel("CNV size (bp, log scale)")
            ax.set_ylabel(metric)

            # Figure-level legends accumulate, so no add_artist bookkeeping is needed.
            for handles, ttl, loc in legend_specs:
                fig.legend(handles=handles, title=ttl, loc=loc, fontsize=9)

            figures[metric] = fig
        return figures

    return (plot_size_vs_metric,)


@app.cell
def _(figures_dir, metrics_dict, mo, os, plot_size_vs_metric):
    # Example slice: padding (colour), input set (dash), reciprocal threshold (marker).
    size_figures = plot_size_vs_metric(
        select_metrics(
            metrics_dict,
            benchmark_setting=["bench_pad0_mn1_mw0_lssT", "bench_pad500_mn1_mw0_lssT"],
            input_set=["30x_Coverage", "2x_Coverage"],
            call_set=["consensus_2of3_w0.5", "calls"],
            classification_setting=["classify_recip0.5", "classify_recip0.3", "classify_recip0.7", "classify_recip0"],),
        hue="benchmark_setting",
        style="input_set",
        marker="classification_setting",
        title="size vs. metric across padding values",
    )
    mo.vstack(list(size_figures.values()))

    # Output to figures directory
    for _metric, _fig in size_figures.items():
        _fig_path = figures_dir / "size_vs_metric" / f"{_metric}_vs_size.png"
        os.makedirs(_fig_path.parent, exist_ok=True)
        _fig.savefig(_fig_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
