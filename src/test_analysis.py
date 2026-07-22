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
def padding_bp(setting):
    """Numeric padding (bp) parsed from a 'padding<N>' setting name."""
    return int("".join(ch for ch in setting if ch.isdigit()) or 0)


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
def _(plt, sns):
    def plot_metric_vs_padding(df, metric) -> plt.Figure:
        """One metric vs padding, one line per (input_set, call_set) tuple.

        Colour encodes input_set, line/marker style encodes call_set.
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = sns.color_palette("tab10", n_colors=df["input_set"].nunique())

        sns.lineplot(
            data=df,
            x="padding",
            y=metric,
            hue="input_set",
            style="call_set",
            markers=True,
            dashes=True,
            ax=ax,
            palette=colors
        )
        ax.set_title(f"{metric} vs benchmark padding")
        ax.set_xlabel("padding (bp)")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(df["padding"].unique()))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        fig.tight_layout()
        return fig

    return (plot_metric_vs_padding,)


@app.cell
def _(Path, padding_metrics, plot_metric_vs_padding):
    padding_figures = {
        metric: plot_metric_vs_padding(padding_metrics, metric)
        for metric in ("TP", "FP", "FN", "precision", "recall", "f1")
    }

    # output to figures directory
    figures_dir = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/figures")
    for _metric, _fig in padding_figures.items():
        fig_path = figures_dir / f"{_metric}_vs_padding.png"
        _fig.savefig(fig_path)
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
    _sizes = pool_sizes(classification_tree["padding0"]["30x_Coverage"]["consensus_2"])
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
    for benchmark_setting, input_set_dict in classification_tree.items():
        for input_set_name, call_set_dict in input_set_dict.items():
            for call_set_name, samples in call_set_dict.items():
                _sizes = pool_sizes(samples)
                distributions = sweep_size_thresholds(_sizes)
                metrics_dict[(benchmark_setting, input_set_name, call_set_name)] = distributions
    return (metrics_dict,)


@app.cell
def _(precision_list, sns):
    # Plot size vs. precision
    precision_line_plot = sns.lineplot(x=[x[0] for x in precision_list], y=[x[1] for x in precision_list])
    precision_line_plot.set(xscale='log')
    precision_line_plot
    return


@app.cell
def _(recall_list, sns):
    # Plot size vs. recall
    recall_line_plot = sns.lineplot(x=[x[0] for x in recall_list], y=[x[1] for x in recall_list])
    recall_line_plot.set(xscale='log')
    recall_line_plot
    return


@app.cell
def _(f1_list, sns):
    # Plot size vs. f1
    f1_line_plot = sns.lineplot(x=[x[0] for x in f1_list], y=[x[1] for x in f1_list])
    f1_line_plot.set(xscale='log')
    f1_line_plot
    return


@app.function
def select_metrics(
    metrics_by_combo: dict, 
    benchmark_setting: str | list | None = None, 
    input_set: str | list | None = None, 
    call_set: str | list | None = None
) -> dict:
    """Slice the (benchmark_setting, input_set, call_set) -> distributions dict.

    Each filter is a single value, a list of values, or None (keep all). E.g.
    select_metrics(metrics_dict, input_set="30x_Coverage") keeps only 30x across
    every padding and call set.
    """
    def keep(value, wanted):
        if wanted is None:
            return True
        return value == wanted if isinstance(wanted, str) else value in wanted

    return {
        key: distributions
        for key, distributions in metrics_by_combo.items()
        if keep(key[0], benchmark_setting)
        and keep(key[1], input_set)
        and keep(key[2], call_set)
    }


@app.cell
def _(metrics_dict, plt):
    def plot_size_vs_metric(metrics_by_combo: dict, hue="input_set", style="call_set", title="") -> plt.Figure:
        """Size vs. precision/recall/F1, one line per combination on one figure.

        `hue` (colour) and `style` (dash) each name a key dimension to encode:
        'benchmark_setting', 'input_set', or 'call_set'. Any dimension not mapped
        is left overlaid. Pre-filter the dict with select_metrics to thin the lines.
        """
        from matplotlib.lines import Line2D

        dimension_index = {"benchmark_setting": 0, "input_set": 1, "call_set": 2}
        hue_i, style_i = dimension_index[hue], dimension_index[style]

        def natural_key(value):
            # Order 'padding0 < padding500 < ...' and '2x < 4x < 30x' numerically.
            digits = "".join(ch for ch in value if ch.isdigit())
            return (int(digits) if digits else 0, value)

        keys = list(metrics_by_combo)
        hue_values = sorted({key[hue_i] for key in keys}, key=natural_key)
        style_values = sorted({key[style_i] for key in keys}, key=natural_key)
        colors = dict(zip(hue_values, plt.cm.tab10.colors))
        dash_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
        dashes = dict(zip(style_values, dash_styles))

        metric_names = ["precision", "recall", "f1"]
        fig, axes = plt.subplots(1, len(metric_names), figsize=(24, 6))
        for ax, metric in zip(axes, metric_names):
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
                    linewidth=1.0,
                    alpha=0.6,
                )
            ax.set_xscale("log")
            ax.set_title(metric)
            ax.set_xlabel("CNV size (bp, log scale)")
            ax.set_ylabel(metric)

        color_handles = [Line2D([], [], color=colors[v], label=v) for v in hue_values]
        dash_handles = [Line2D([], [], color="0.35", linestyle=dashes[v], label=v) for v in style_values]
        color_legend = fig.legend(
            handles=color_handles, title=hue, loc="upper left", bbox_to_anchor=(0.91, 0.95)
        )
        fig.legend(
            handles=dash_handles, title=style, loc="lower left", bbox_to_anchor=(0.91, 0.05)
        )
        fig.add_artist(color_legend)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        return fig

    # Example slice: 30x coverage only, comparing padding values (colour) per caller (dash).
    plot_size_vs_metric(
        select_metrics(metrics_dict, input_set=["30x_Coverage", "2x_Coverage", "SNP_Array"], call_set=["consensus_1", "calls"]),
        hue="benchmark_setting",
        style="input_set",
        title="size vs. metric across padding values",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
