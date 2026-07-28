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
            # figure.dpi drives inline rendering, savefig.dpi drives files on disk —
            # they are separate rcParams so the notebook can stay light while saved
            # figures are print-resolution. savefig.dpi defaults to "figure", i.e. 100.
            "figure.dpi": 100,
            "savefig.dpi": 300,
            # Crop the canvas reserved for the legend instead of banding it in white.
            "savefig.bbox": "tight",
            # Embed TrueType rather than Type 3 outlines: Type 3 keeps text
            # unselectable and is rejected by many journal submission systems.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return Path, np, os, pd, plt, sns


@app.cell
def _():
    import consensuscnv
    import consensuscnv.utils
    from consensuscnv import analysis_assist
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
    path_to_test_parsing = "/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/out/4x_Coverage/cnvpytor"
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
    # get list of samples that appear in every source
    def get_common_samples(summary):
        sources = summary["source"].unique()
        sample_sets = [set(summary[summary["source"] == cs]["sample"]) for cs in sources]
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


@app.function
def save_figure(fig, path, formats=("png", "pdf")) -> list:
    """Write `fig` to `path` once per format, creating parent directories.

    PNG is the raster copy for viewing; PDF is vector, so it has no resolution at all
    and stays sharp at any zoom — that is the one to hand to a paper or a poster.
    Resolution and cropping come from rcParams (savefig.dpi / savefig.bbox), so they
    stay set in one place rather than at each call site.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, format=fmt)
        written.append(out_path)
    return written


@app.cell
def _(summary):
    # Pool per-sample counts into one row
    padding_metrics = summary.groupby(
        ["benchmark_setting","classification_setting", "query", "source"], as_index=False
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
        ["query", "source", "padding"]
    ).reset_index(drop=True)
    return (padding_metrics,)


@app.cell
def _(padding_metrics):
    padding_metrics
    return


@app.cell
def _(padding_metrics):
    padding_metrics[padding_metrics["query"].isin(["4x_Coverage"]) & padding_metrics["source"].isin(["consensus_2_weight_0.0", "consensus_2_weight_0.5"])]
    return


@app.cell
def select_rows(pd):
    def select_rows(
        df,
        query: str | list | None = None,
        classification_setting: str | list | None = None,
        source: str | list | None = None,
    ) -> pd.DataFrame:
        """Filter a padding_metrics-style DataFrame by the three sweep dimensions.

        Each argument is a single value, a list of values, or None (keep all). E.g.
        select_rows(padding_metrics, query="30x_Coverage", classification_setting="classify_recip0.5")
        keeps 30x @ recip0.5 across every padding and source.
        """
        wanted = {
            "query": query,
            "classification_setting": classification_setting,
            "source": source,
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
    # Channel capacities. 8 is the categorical order's length — past it, extra hues
    # break CVD separation. 4 is where dash patterns stop being tellable apart.
    # Panels have no equivalent cap, which is why the widest dimension goes there.
    _MAX_HUE, _MAX_STYLE = 8, 4

    def plot_metric_vs_padding(
        df, metric, hue=None, style=None, facet=None, sharey=True
    ) -> plt.Figure:
        """One metric vs padding, as small multiples over the third sweep dimension.

        Panels share both axes by default so heights are comparable across them —
        pass sharey=False only when a panel's range would otherwise flatten the rest.
        """
        sweep_dims = ["query", "source", "classification_setting"]
        free = [d for d in sweep_dims if df[d].nunique() > 1]
        pool = sorted(
            (d for d in free if d not in (hue, style, facet)), key=lambda d: df[d].nunique()
        )

        if hue is None and "query" in pool and df["query"].nunique() <= _MAX_HUE:
            hue = "query"
            pool.remove("query")
        if facet is None and len(pool) > (0 if hue else 1):
            facet = pool.pop()  # widest
        if hue is None and pool and df[pool[0]].nunique() <= _MAX_HUE:
            hue = pool.pop(0)
        if style is None and pool and df[pool[0]].nunique() <= _MAX_STYLE:
            style = pool.pop(0)
        if facet is None and pool:
            facet = pool.pop()

        unencoded = [d for d in free if d not in {hue, style, facet}]
        if unencoded:
            raise ValueError(
                f"{unencoded} still vary but are mapped to nothing — seaborn would "
                "average over them and draw a bootstrap CI ribbon. Only one dimension "
                "can be the facet, so narrow these with select_rows() or plot them in "
                "separate figures."
            )
        for channel, dimension, cap in (("hue", hue, _MAX_HUE), ("style", style, _MAX_STYLE)):
            if dimension and df[dimension].nunique() > cap:
                raise ValueError(
                    f"{channel}={dimension!r} has {df[dimension].nunique()} values but "
                    f"only {cap} are distinguishable. Pass facet={dimension!r} to give it "
                    "panels instead, or narrow it with select_rows()."
                )
        # Everything free may have gone to the facet; fall back to a pinned dimension so
        # each panel still gets one consistently-coloured line and a named legend entry.
        if hue is None:
            hue = next((d for d in sweep_dims if d not in (style, facet)), sweep_dims[0])

        hue_values = sweep_order(hue, df[hue])
        style_values = sweep_order(style, df[style]) if style else None
        panels = sweep_order(facet, df[facet]) if facet else [None]
        colors = dict(zip(hue_values, hue_colors(hue, hue_values)))

        ncols = 1 if len(panels) == 1 else 2 if len(panels) <= 4 else 3
        nrows = -(-len(panels) // ncols)
        # Reserve canvas for the legend rather than anchoring it at a fixed fraction —
        # marimo renders the figure as-is, so an overhanging legend gets clipped.
        legend_in = 2.8
        fig_w = 5.5 * ncols + legend_in
        plot_frac = 1 - legend_in / fig_w
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, 4.0 * nrows),
            sharex=True,
            sharey=sharey,
            squeeze=False,
        )
        flat = axes.ravel()
        xticks = sorted(df["padding"].unique())

        for ax, panel in zip(flat, panels):
            panel_df = df if panel is None else df[df[facet] == panel]
            sns.lineplot(
                data=panel_df.sort_values("padding"),
                x="padding",
                y=metric,
                hue=hue,
                hue_order=hue_values,
                style=style,
                style_order=style_values,
                markers=True,
                dashes=True,
                estimator=None,  # one row per point; never collapse to a mean
                errorbar=None,  # and never draw a band around it
                ax=ax,
                palette=colors,
                legend=(ax is flat[0]),
                # seaborn ignores markers=True when style is unmapped, but padding is
                # sampled at only a handful of points — keep them visible.
                **({} if style else {"marker": "o"}),
            )
            ax.set_title("" if panel is None else f"{facet} = {panel}", fontsize=10)
            ax.set_xlabel("padding (bp)")
            ax.set_ylabel(metric)
            ax.set_xticks(xticks)

        for ax in flat[len(panels):]:
            ax.set_visible(False)

        # One figure-level legend rather than a copy per panel.
        handles, labels = flat[0].get_legend_handles_labels()
        if flat[0].get_legend() is not None:
            flat[0].get_legend().remove()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(plot_frac + 0.01, 0.97),
            loc="upper left",
            fontsize="small",
        )

        style_note = f", dash={style}" if style else ""
        facet_note = f", panels={facet}" if facet else ""
        fig.suptitle(f"{metric} vs benchmark padding (colour={hue}{style_note}{facet_note})")
        fig.tight_layout(rect=(0, 0, plot_frac, 0.96))
        return fig

    return (plot_metric_vs_padding,)


@app.function
def recip_value(setting: str) -> float:
    """Numeric reciprocal threshold parsed from a 'classify_recip<x>' setting name."""
    digits = "".join(ch for ch in setting if (ch.isdigit() or ch == "."))
    return float(digits) if digits else 0.0


@app.function
def coverage_value(query: str) -> float:
    """Numeric depth parsed from the leading digits of a '30x_Coverage' query."""
    digits = ""
    for char in query:
        if char.isdigit() or char == ".":
            digits += char
        else:
            break
    return float(digits) if digits else 0.0


@app.function
def sweep_order(dimension: str, values) -> list:
    """Values of a sweep dimension in their natural order, not alphabetical.

    Plain sorting puts '30x_Coverage' second and 'classify_recip0.5' before
    'classify_recip0.7' only by luck; both dimensions are numeric underneath.
    """
    if dimension == "query":
        return sorted(set(values), key=coverage_value)
    if dimension == "classification_setting":
        return sorted(set(values), key=recip_value)
    return sorted(set(values))


@app.function
def coverage_ramp(n: int) -> list[str]:
    """`n` evenly spaced steps of a single-hue blue ramp, light -> dark.

    Coverage is an ordered quantity, so it takes an ordinal ramp rather than
    arbitrary categorical hues — depth then reads directly as colour weight. Steps
    come from the blue 250->700 scale; 250 is the lightest step that still clears
    2:1 against a light surface. Validated: lightness monotone, adjacent dL >= 0.06,
    single hue (4 degree spread).
    """
    steps = [
        "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
    ]
    if n <= 1:
        return [steps[len(steps) // 2]]
    last = len(steps) - 1
    return [steps[round(i * last / (n - 1))] for i in range(n)]


@app.function
def categorical_colors(n: int) -> list[str]:
    """First `n` slots of the fixed categorical order — assigned in order, never cycled.

    Validated on the adjacent pairlist (the one that applies to line charts): worst
    adjacent CVD dE 9.1, worst normal-vision dE 22.9 on a light surface. Two slots
    sit below 3:1 contrast against the surface, which the notebook's
    `padding_metrics` table view relieves.
    """
    slots = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]
    if n > len(slots):
        raise ValueError(
            f"{n} series exceeds the {len(slots)}-slot categorical order. Generating "
            "extra hues would break CVD separation — facet, or fold the tail into "
            "'Other', instead."
        )
    return slots[:n]


@app.function
def hue_colors(dimension: str, values: list) -> list[str]:
    """Colours for `values`: an ordinal ramp for coverage, categorical otherwise."""
    return coverage_ramp(len(values)) if dimension == "query" else categorical_colors(len(values))


@app.cell
def _(padding_metrics, plot_metric_vs_padding):
    padding_figures = {
        metric: plot_metric_vs_padding(padding_metrics, metric)
        for metric in ("TP", "FP", "FN", "precision", "recall", "f1")
    }
    return (padding_figures,)


@app.cell
def _(Path, padding_figures):
    # output to figures directory
    figures_dir = Path("/lab01/Projects/Lionel_Projects/blendedCNV_pipeline/figures")
    for _metric, _fig in padding_figures.items():
        save_figure(_fig, figures_dir / "vs_padding" / f"{_metric}_vs_padding.png")
    return (figures_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### flexible slices: metric vs padding

    `select_rows(padding_metrics, query=..., classification_setting=..., source=...)`
    pins any of the three sweep dimensions (single value, list, or None=all). Whatever
    is still varying gets auto-encoded as colour (hue) then dash (style) by
    `plot_metric_vs_padding`; padding stays on the x-axis.
    """)
    return


@app.cell
def _(padding_metrics, plot_metric_vs_padding, select_rows):
    # Scenario 1: one query + one classification setting, ALL sources.
    _slice = select_rows(
        padding_metrics,
        query="30x_Coverage",
        classification_setting="classify_recip0.5",
    )
    # Only source is free -> it becomes the colour.
    plot_metric_vs_padding(_slice, "f1")
    return


@app.cell
def _(padding_metrics, plot_metric_vs_padding, select_rows):
    # Scenario 2: one query + a few sources, ALL classification settings.
    _slice = select_rows(
        padding_metrics,
        query="30x_Coverage",
        source=["consensus_2_weight_0.5", "calls"],
    )
    # source and classification_setting are free -> colour + dash.
    plot_metric_vs_padding(_slice, "f1", hue="classification_setting", style="source")
    return


@app.cell
def _(padding_metrics, plot_metric_vs_padding, select_rows):
    # Scenario 3: one classification setting + a few sources, ALL queries.
    _slice = select_rows(
        padding_metrics,
        classification_setting="classify_recip0.5",
        source=["consensus_2_weight_0.5", "calls"],
    )
    plot_metric_vs_padding(_slice, "f1", hue="query", style="source")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### interactive 3D: padding × reciprocal threshold × metric
    """)
    return


@app.cell
def _(padding_metrics, select_rows):
    _SOURCE_DASHES = ["solid", "dash", "dot", "dashdot"]
    _SOURCE_SYMBOLS = ["circle", "diamond", "square", "cross"]
    _NEUTRAL = "#8a8a85"

    def plot_metric_3d(df, metric):
        """3D lines over the (padding, reciprocal-threshold) grid.

        One Scatter3d trace per (query, source, classification_setting) — each
        line holds its reciprocal threshold fixed (constant y) and walks padding, so
        the threshold levels read as separate parallel curves instead of one line
        zig-zagging through them all. Returns a plotly Figure; marimo renders it live.

        Colour encodes the query (coverage depth) alone, so every line from the same
        coverage shares one colour and shallow-to-deep reads off the ramp. Source is
        carried by dash pattern + marker symbol and the reciprocal threshold is
        already positional (the y axis), so neither leans on colour.
        """
        import plotly.graph_objects as go

        df = df.copy()
        df["recip"] = df["classification_setting"].map(recip_value)

        queries = sweep_order("query", df["query"])
        sources = sweep_order("source", df["source"])
        settings = sweep_order("classification_setting", df["classification_setting"])
        query_color = dict(zip(queries, hue_colors("query", queries)))
        source_dash = {s: _SOURCE_DASHES[i % len(_SOURCE_DASHES)] for i, s in enumerate(sources)}
        source_symbol = {s: _SOURCE_SYMBOLS[i % len(_SOURCE_SYMBOLS)] for i, s in enumerate(sources)}

        groups = dict(tuple(df.groupby(["query", "source", "classification_setting"])))

        fig = go.Figure()
        # Walk queries in coverage order so the legend reads shallow -> deep, and let
        # only the first trace of each coverage carry a legend entry.
        for query in queries:
            for source in sources:
                for setting in settings:
                    group = groups.get((query, source, setting))
                    if group is None or group.empty:
                        continue
                    group = group.sort_values("padding")
                    fig.add_trace(
                        go.Scatter3d(
                            x=group["padding"],
                            y=group["recip"],
                            z=group[metric],
                            mode="lines+markers",
                            marker={
                                "size": 4,
                                "color": query_color[query],
                                "symbol": source_symbol[source],
                            },
                            line={
                                "width": 3,
                                "color": query_color[query],
                                "dash": source_dash[source],
                            },
                            legendgroup=query,
                            showlegend=(source == sources[0] and setting == settings[0]),
                            name=query,
                            hovertemplate=(
                                f"<b>{query}<br>{source}</b><br>"
                                "padding %{x:,} bp<br>"
                                "recip %{y:g}<br>"
                                f"{metric} %{{z:.4g}}<extra></extra>"
                            ),
                        )
                    )

        # Neutral swatches document the dash/symbol encoding without repainting series.
        for source in sources:
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="lines+markers",
                    line={"width": 3, "color": _NEUTRAL, "dash": source_dash[source]},
                    marker={"size": 4, "color": _NEUTRAL, "symbol": source_symbol[source]},
                    name=source,
                    legendgroup=f"source::{source}",
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            title=f"{metric}: padding × reciprocal threshold",
            scene={
                "xaxis": {"title": "padding (bp)"},
                "yaxis": {"title": "classification reciprocal threshold"},
                "zaxis": {"title": metric},
            },
            height=700,
            width=1000,
            legend={"font": {"size": 9}, "title": {"text": "coverage / source"}},
        )
        return fig

    # A few sets keep the 3D view readable.
    _slice = select_rows(
        padding_metrics,
        query=["30x_Coverage", "6x_Coverage", "4x_Coverage", "2x_Coverage"],
        source=["consensus_2_weight_0.5", "calls"],
    )
    plot_metric_3d(_slice, "recall")
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
        """Pool CNV sizes (bp) across a source's samples.

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
    # Walk the 4-level tree (bench -> classify -> query -> source) to one
    # representative leaf without hardcoding slug names.
    _bench = next(iter(classification_tree))
    _classify = next(iter(classification_tree[_bench]))
    _query = next(iter(classification_tree[_bench][_classify]))
    _source = next(iter(classification_tree[_bench][_classify][_query]))
    _sizes = pool_sizes(classification_tree[_bench][_classify][_query][_source])
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
        benchmark_setting = benchmark_setting.split("=")[1]
        for classification_setting, query_dict in classify_dict.items():
            classification_setting = classification_setting.split("=")[1]
            for query_name, source_dict in query_dict.items():
                query_name = query_name.split("=")[1]
                for source_name, samples in source_dict.items():
                    source_name = source_name.split("=")[1]

                    _sizes = pool_sizes(samples)
                    distributions = sweep_size_thresholds(_sizes)
                    key = (benchmark_setting, classification_setting, query_name, source_name)
                    metrics_dict[key] = distributions
    return (metrics_dict,)


@app.function
def select_metrics(
    metrics_by_combo: dict,
    benchmark_setting: str | list | None = None,
    classification_setting: str | list | None = None,
    query: str | list | None = None,
    source: str | list | None = None,
) -> dict:
    """Slice the (benchmark_setting, classification_setting, query, source) -> distributions dict.

    Each filter is a single value, a list of values, or None (keep all). E.g.
    select_metrics(metrics_dict, query="30x_Coverage") keeps only 30x across
    every padding, classification setting, and source.
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
        and keep(key[2], query)
        and keep(key[3], source)
    }


@app.cell
def _(plt):
    def plot_size_vs_metric(
        metrics_by_combo: dict,
        hue: str = "query",
        style: str = "source",
        marker: str | None = None,
        title: str = "",
    ) -> dict[str, plt.Figure]:
        """Size vs. precision/recall/F1, each metric on its OWN figure.

        Returns ``{metric: Figure}`` (keys 'precision', 'recall', 'f1'). Up to three
        key dimensions can be encoded: `hue` -> colour, `style` -> dash pattern,
        `marker` -> point symbol (optional). Each names one of 'benchmark_setting',
        'classification_setting', 'query', or 'source'. Any dimension not
        mapped is left overlaid. Pre-filter the dict with select_metrics to thin
        the lines.

        Layout is handed to matplotlib's constrained layout with figure-level
        "outside right" legends, so the axes shrink to fit the legends and
        nothing clips regardless of how long the setting slugs are.
        """
        from matplotlib import color_sequences
        from matplotlib.lines import Line2D
        from matplotlib.typing import LegendLocType, LineStyleType

        dimension_index = {
            "benchmark_setting": 0,
            "classification_setting": 1,
            "query": 2,
            "source": 3,
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
        # color_sequences["tab10"] is the same 10 RGB tuples as plt.cm.tab10.colors,
        # but typed as list[ColorType] — the cm registry is typed Mapping[str, Colormap],
        # and the base Colormap has no .colors (only the ListedColormap subclass does).
        colors = dict(zip(hue_values, color_sequences["tab10"]))
        dash_styles: list[LineStyleType] = [
            "-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))
        ]
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
        # The LegendLocType annotation matters: without it the loc strings widen to
        # plain `str` inside the list and no fig.legend() overload matches.
        legend_specs: list[tuple[list[Line2D], str | None, LegendLocType]] = [
            (color_handles, hue, "outside right upper")
        ]
        if marker_handles:
            legend_specs.append((marker_handles, marker, "outside right center"))
        legend_specs.append((dash_handles, style, "outside right lower"))

        figures: dict[str, plt.Figure] = {}
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
def _(figures_dir, metrics_dict, mo, plot_size_vs_metric):
    # Example slice: padding (colour), query (dash), reciprocal threshold (marker).
    size_figures = plot_size_vs_metric(
        select_metrics(
            metrics_dict,
            benchmark_setting=["bench_pad0_mn1_mw0_lssT", "bench_pad500_mn1_mw0_lssT"],
            query=["30x_Coverage", "2x_Coverage"],
            source=["consensus_2_weight_0.5", "calls"],
            classification_setting=["classify_recip0.5", "classify_recip0.3", "classify_recip0.7", "classify_recip0"],),
        hue="benchmark_setting",
        style="query",
        marker="classification_setting",
        title="size vs. metric across padding values",
    )
    mo.vstack(list(size_figures.values()))

    # Output to figures directory
    for _metric, _fig in size_figures.items():
        save_figure(_fig, figures_dir / "size_vs_metric" / f"{_metric}_vs_size.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
