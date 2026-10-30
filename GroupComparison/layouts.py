# analysis/groupcomparison/layouts.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Helpers.DataHelpers as DataHelpers
from analysis import psychometric as Psychometric
from analysis.uncertainty import ci95_yerr, summarize_animal_mean_ci

from .config import ViewSpec, GroupComparisonConfig, PlotStyle, OverlaySpec
from .plots import (
    style_axes, apply_50_tick_labels, _maybe_shift,
    plot_rt_on_ax, plot_mt_on_ax, plot_psy_on_ax,
    add_jnd_inset_abl_colored, add_jnd_inset_single_abl
)

def plot_views_3x3(
    prepared,
    views,
    cfg,
    style,
    overlay,
    group_jnd_by_view,
    view_colors,
    add_jnd_inset=True,
    rt_shaded_sem=False,
) -> plt.Figure:
    """
    Rows = views (genotypes), Cols = RT/MT/Psy
    Each axis shows multiple ABLs colored by ABL.
    """
    view_names = [v.name for v in views]
    abls = sorted(set().union(*[
        set(prepared[vn]["rt_group"]["ABL"].unique()) for vn in view_names
    ]))
    abl_colors = {abl: f"C{i % 10}" for i, abl in enumerate(abls)}

    fig, axes = plt.subplots(len(view_names), 3, figsize=(22, 7 * len(view_names)), squeeze=False)

    for r, vn in enumerate(view_names):
        tables = prepared[vn]
        ax_rt, ax_mt, ax_psy = axes[r]

        abls_v = sorted(tables["rt_group"]["ABL"].unique())
        for abl in abls_v:
            c = abl_colors.get(abl, "gray")
            if rt_shaded_sem:
                rt_df = tables["rt_group"]
                _plot_chrono_mean_shaded(
                    ax_rt, rt_df[rt_df["ABL"] == abl], color=c, cfg=cfg,
                )
            else:
                plot_rt_on_ax(ax_rt, tables, abl, c, cfg)
            plot_mt_on_ax(ax_mt, tables, abl, c, cfg)
            plot_psy_on_ax(ax_psy, tables, abl, c, cfg)

            # overlays (old neurotypical)
            if overlay.makefig1_chrono is not None:
                DataHelpers.overlay_makefig1_rt(ax_rt, abl, overlay.makefig1_chrono, color=overlay.overlay_color, zorder=-1)
            if overlay.makefig1_data is not None and abl != 50:
                DataHelpers.overlay_makefig1_psychometrics(
                    ax_psy, overlay.makefig1_data, abl=abl,
                    color="black", show_individuals=False, use_abl_colors=False
                )

        style_axes(ax_rt, style, f"{vn} — RT", "ILD (dB)", "Mean RT (s)")
        style_axes(ax_mt, style, f"{vn} — MT", "ILD (dB)", "Mean MT (s)")
        style_axes(ax_psy, style, f"{vn} — Psychometric", "ILD (dB)", "P(Right)")

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_rt.set_ylim(*cfg.ylim_rt)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_mt.set_ylim(*cfg.ylim_mt)
        ax_psy.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

        # JND inset colored by ABL (matches curve colors)
        if add_jnd_inset:
            add_jnd_inset_abl_colored(
                ax_parent=ax_psy,
                group_jnd_df=group_jnd_by_view.get(vn),
                abl_colors=abl_colors,
                style=style,
                inset_rect=(0.70, 0.15, 0.30, 0.30),
            )

    # legend = ABL colors
    handles, labels = [], []
    for abl, c in abl_colors.items():
        handles.append(plt.Line2D([], [], color=c, marker="o", linestyle="None"))
        labels.append(f"ABL {abl} dB")

    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(handles)), fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.92])


    return fig


def plot_abls_4x3(
    prepared: Dict[str, dict],
    views: List[ViewSpec],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    overlay: OverlaySpec,
    view_colors: Dict[str, str],
    group_jnd_by_view: Dict[str, "pd.DataFrame"],
    add_inset: bool = True,
    view_styles: Optional[Dict[str, dict]] = None,
) -> plt.Figure:
    """
    Rows = ABLs, Cols = RT/MT/Psy
    Each row axis shows multiple views in view colors.
    JND inset stays inside the psychometrics (bottom-right psychometric axis).
    """
    import pandas as pd

    view_names = [v.name for v in views]
    abl_rows = sorted(set().union(*[
        set(prepared[vn]["rt_group"]["ABL"].unique()) for vn in view_names
    ]))

    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.8 * len(abl_rows)), squeeze=False)

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r]

        for vn in view_names:
            c = view_colors.get(vn, "gray")
            style_cfg = (view_styles or {}).get(vn, {})
            tables = prepared[vn]
            plot_rt_on_ax(ax_rt, tables, abl, c, cfg, **style_cfg)
            plot_mt_on_ax(ax_mt, tables, abl, c, cfg, **style_cfg)
            plot_psy_on_ax(ax_psy, tables, abl, c, cfg, **style_cfg)

        # overlays
        if overlay.makefig1_chrono is not None:
            DataHelpers.overlay_makefig1_rt(ax_rt, abl, overlay.makefig1_chrono, color=overlay.overlay_color,force_black=True, zorder=-1)

        if overlay.makefig1_data is not None and abl != 50:
            DataHelpers.overlay_makefig1_psychometrics(
                ax_psy, overlay.makefig1_data, abl=abl,
                color="black", show_individuals=False, use_abl_colors=False
            )

        style_axes(ax_rt, style, f"ABL {abl} — RT", "ILD (dB)", "Mean RT (s)")
        style_axes(ax_mt, style, f"ABL {abl} — MT", "ILD (dB)", "Mean MT (s)")
        style_axes(ax_psy, style, f"ABL {abl} — Psychometric", "ILD (dB)", "P(Right)")

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_rt.set_ylim(*cfg.ylim_rt)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_mt.set_ylim(*cfg.ylim_mt)
        ax_psy.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

    # legend = views
    handles = [
        plt.Line2D(
            [],
            [],
            color=view_colors[vn],
            marker=(view_styles or {}).get(vn, {}).get("marker", "o"),
            linestyle=(view_styles or {}).get(vn, {}).get("linestyle", "None"),
            markerfacecolor=(
                view_colors[vn]
                if (view_styles or {}).get(vn, {}).get("markerfacecolor", None) is None
                else (view_styles or {}).get(vn, {}).get("markerfacecolor")
            ),
            markeredgecolor=view_colors[vn],
        )
        for vn in view_names
    ]
    fig.legend(
        handles,
        view_names,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=style.legend_fs,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    return fig


def _pooled_chrono_group(tables: dict, metric: str) -> pd.DataFrame:
    if metric == "rt":
        per_subj = tables["rt_per_subj"]
        value_col = "mean_rt"
    elif metric == "mt":
        per_subj = tables["mt_per_subj"]
        value_col = "mean_mt"
    else:
        raise ValueError("metric must be 'rt' or 'mt'.")

    if per_subj.empty:
        return pd.DataFrame(columns=["ILD", "mean", "sem", "ci95_low", "ci95_high", "n", "n_strata"])

    subject_cols = ["dataset_key", "animal", "ILD"] if "dataset_key" in per_subj.columns else ["animal", "ILD"]
    pooled_subject = (
        per_subj.groupby(subject_cols, as_index=False, dropna=False)[value_col]
        .mean()
    )
    return summarize_animal_mean_ci(
        pooled_subject,
        value_col=value_col,
        group_cols=["ILD"],
    )


def _pooled_psychometric_group(tables: dict) -> pd.DataFrame:
    points = tables["psy_points"].copy()
    if points.empty:
        return pd.DataFrame(columns=["ILD", "mean", "sem", "ci95_low", "ci95_high", "n", "n_strata"])
    subject_cols = ["dataset_key", "subject", "ILD"] if "dataset_key" in points.columns else ["subject", "ILD"]
    per_subject = (
        points.groupby(subject_cols, dropna=False)["PropLeft"]
        .mean()
        .reset_index()
    )
    return summarize_animal_mean_ci(
        per_subject.rename(columns={"subject": "animal"}),
        value_col="PropLeft",
        group_cols=["ILD"],
    )


def _plot_pooled_psychometric_on_ax(
    ax,
    group_df: pd.DataFrame,
    *,
    color: str,
    cfg: GroupComparisonConfig,
    linestyle: str = "-",
    marker: str = "o",
    markerfacecolor=None,
) -> None:
    if group_df.empty:
        return
    markerfacecolor = color if markerfacecolor is None else markerfacecolor
    df = group_df.sort_values("ILD").copy()
    # Fit in physical ILD units; compress ±50 only for displayed points.
    x = pd.to_numeric(df["ILD"], errors="coerce").to_numpy(dtype=float)
    display_x = _maybe_shift(x, cfg)
    y = pd.to_numeric(df["mean"], errors="coerce").to_numpy(dtype=float)
    yerr = ci95_yerr(df)
    ax.errorbar(display_x, y, yerr=yerr, fmt=marker, color=color, markerfacecolor=markerfacecolor,
                markeredgecolor=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 4:
        return
    try:
        _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
            x[finite], y[finite], model="my_psycho",
            n_trials=np.clip(df.loc[finite, "n"].to_numpy(dtype=float), 1, None).astype(int),
            show_plot=False,
            fit_method="least_squares",
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return
    ax.plot(xx, yy, color=color, linewidth=2.5, linestyle=linestyle)


def _plot_chrono_mean_shaded(
    ax,
    group_df: pd.DataFrame,
    *,
    color: str,
    cfg: GroupComparisonConfig,
    linestyle: str = "-",
    marker: str = "o",
    markerfacecolor=None,
) -> None:
    if group_df is None or group_df.empty:
        return
    markerfacecolor = color if markerfacecolor is None else markerfacecolor
    df = group_df.sort_values("ILD").copy()
    x = _maybe_shift(df["ILD"], cfg)
    y = df["mean"].to_numpy(dtype=float)
    errors = ci95_yerr(df)
    low = y - errors[0]
    high = y + errors[1]
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.5,
        linestyle=linestyle,
        marker=marker,
        markersize=7,
        markerfacecolor=markerfacecolor,
        markeredgecolor=color,
    )
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)


def _chrono_ylim(frames: List[pd.DataFrame]) -> tuple[float, float] | None:
    lows = []
    highs = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        y = pd.to_numeric(frame["mean"], errors="coerce")
        errors = ci95_yerr(frame)
        low = pd.Series(y.to_numpy(dtype=float) - errors[0]).replace([np.inf, -np.inf], np.nan).dropna()
        high = pd.Series(y.to_numpy(dtype=float) + errors[1]).replace([np.inf, -np.inf], np.nan).dropna()
        if not low.empty:
            lows.append(float(low.min()))
        if not high.empty:
            highs.append(float(high.max()))

    if not lows or not highs:
        return None

    lo = min(lows)
    hi = max(highs)
    span = hi - lo
    pad = max(span * 0.12, 0.02)
    lower = max(0, lo - pad)
    upper = hi + pad
    if upper <= lower:
        upper = lower + 0.05
    return lower, upper


def _empty_chronometric_figure(title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)
    ax.text(
        0.5, 0.5,
        "No prepared data for the selected groups.\n"
        "Check group selections and filters, then rerun preparation.",
        ha="center", va="center", fontsize=14, transform=ax.transAxes,
    )
    fig.tight_layout()
    return fig


def plot_chronometrics(
    prepared: Dict[str, dict],
    views: List[ViewSpec],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    view_colors: Dict[str, str],
    *,
    separate_abls: bool = True,
    view_styles: Optional[Dict[str, dict]] = None,
) -> plt.Figure:
    """
    Chronometric-only RT/MT figure.

    separate_abls=True: rows are ABLs, columns are RT and MT.
    separate_abls=False: one pooled row, columns are RT and MT.
    Always plots group means with shaded animal-level 95% CI, without individual traces.
    """
    style = replace(style, title_fs=14, label_fs=14, tick_fs=14, legend_fs=14)
    cfg = replace(cfg, xlim_sym=(-17, 17))
    view_names = [v.name for v in views]
    if not any(not prepared[name]["rt_group"].empty for name in view_names):
        return _empty_chronometric_figure("Chronometrics: no usable trials")
    if separate_abls:
        abl_rows = sorted(set().union(*[
            set(prepared[vn]["rt_group"]["ABL"].unique()) for vn in view_names
        ]))
        # Hide the ABL 50 panel only; do not remove ±50 ILD fit points.
        abl_rows = [abl for abl in abl_rows if float(abl) != 50]
        if not abl_rows:
            return _empty_chronometric_figure("No ABL panels other than 50 dB")
    else:
        abl_rows = ["All ABLs"]

    fig, axes = plt.subplots(
        len(abl_rows),
        2,
        figsize=(12, 4.8 * len(abl_rows)),
        squeeze=False,
    )

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt = axes[r]
        rt_frames = []
        mt_frames = []

        for vn in view_names:
            c = view_colors.get(vn, "gray")
            style_cfg = (view_styles or {}).get(vn, {})
            tables = prepared[vn]

            if separate_abls:
                rt_df = tables["rt_group"][tables["rt_group"]["ABL"] == abl]
                mt_df = tables["mt_group"][tables["mt_group"]["ABL"] == abl]
            else:
                rt_df = _pooled_chrono_group(tables, "rt")
                mt_df = _pooled_chrono_group(tables, "mt")

            rt_df = rt_df[pd.to_numeric(rt_df["ILD"], errors="coerce").abs().ne(50)].copy()
            mt_df = mt_df[pd.to_numeric(mt_df["ILD"], errors="coerce").abs().ne(50)].copy()
            rt_frames.append(rt_df)
            mt_frames.append(mt_df)
            _plot_chrono_mean_shaded(ax_rt, rt_df, color=c, cfg=cfg, **style_cfg)
            _plot_chrono_mean_shaded(ax_mt, mt_df, color=c, cfg=cfg, **style_cfg)

        row_label = f"ABL {abl}" if separate_abls else "All ABLs"
        style_axes(ax_rt, style, f"{row_label} - RT", "|ILD| (dB)", "Mean RT (s)")
        style_axes(ax_mt, style, f"{row_label} - MT", "ILD (dB)", "Mean MT (s)")
        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_rt.set_xticks([0, 5, 10, 15])
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_mt.set_xticks([-16, -4, 0, 4, 16])
        rt_ylim = _chrono_ylim(rt_frames)
        mt_ylim = _chrono_ylim(mt_frames)
        if rt_ylim is not None:
            ax_rt.set_ylim(*rt_ylim)
        if mt_ylim is not None:
            ax_mt.set_ylim(*mt_ylim)

    handles = [
        plt.Line2D(
            [],
            [],
            color=view_colors[vn],
            marker=(view_styles or {}).get(vn, {}).get("marker", "o"),
            linestyle=(view_styles or {}).get(vn, {}).get("linestyle", "-"),
            markerfacecolor=(
                view_colors[vn]
                if (view_styles or {}).get(vn, {}).get("markerfacecolor", None) is None
                else (view_styles or {}).get(vn, {}).get("markerfacecolor")
            ),
            markeredgecolor=view_colors[vn],
        )
        for vn in view_names
    ]
    fig.legend(handles, view_names, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=1, fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    return fig


def plot_chronometric_psychometrics(
    prepared: Dict[str, dict],
    views: List[ViewSpec],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    view_colors: Dict[str, str],
    *,
    separate_abls: bool = True,
    view_styles: Optional[Dict[str, dict]] = None,
) -> plt.Figure:
    """Plot psychometrics using the views and ABL treatment of the chronometric layout."""
    style = replace(style, title_fs=14, label_fs=14, tick_fs=14, legend_fs=14)
    cfg = replace(cfg, xlim_sym=(-17, 17))
    view_names = [view.name for view in views]
    if not any(not prepared[name]["psy_group"].empty for name in view_names):
        return _empty_chronometric_figure("Psychometrics: no usable trials")
    if separate_abls:
        abl_rows = sorted(set().union(*[
            set(prepared[name]["psy_group"]["ABL"].unique()) for name in view_names
        ]))
        # Hide the ABL 50 panel only; do not remove ±50 ILD fit points.
        abl_rows = [abl for abl in abl_rows if float(abl) != 50]
        if not abl_rows:
            return _empty_chronometric_figure("No ABL panels other than 50 dB")
    else:
        abl_rows = ["All ABLs"]

    fig, axes = plt.subplots(len(abl_rows), 1, figsize=(7.2, 4.8 * len(abl_rows)), squeeze=False)
    group_cfg = replace(cfg, error_mode="sem")
    for row, abl in enumerate(abl_rows):
        ax = axes[row, 0]
        for name in view_names:
            tables = prepared[name]
            color = view_colors.get(name, "gray")
            style_cfg = (view_styles or {}).get(name, {})
            if separate_abls:
                plot_psy_on_ax(ax, tables, abl, color, group_cfg, **style_cfg)
            else:
                _plot_pooled_psychometric_on_ax(
                    ax,
                    _pooled_psychometric_group(tables),
                    color=color,
                    cfg=cfg,
                    **style_cfg,
                )
                ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
                ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        row_label = f"ABL {abl}" if separate_abls else "All ABLs"
        style_axes(ax, style, f"{row_label} - Psychometric", "ILD (dB)", "P(Right)")
        ax.set_xlim(*cfg.xlim_sym)
        ax.set_ylim(0, 1)
        ax.set_xticks([-16, -4, 0, 4, 16])
        ax.set_xticklabels(["-16", "-4", "0", "4", "16"])

    handles = [
        plt.Line2D(
            [], [], color=view_colors[name], marker=(view_styles or {}).get(name, {}).get("marker", "o"),
            linestyle=(view_styles or {}).get(name, {}).get("linestyle", "-"),
            markerfacecolor=(view_colors[name] if (view_styles or {}).get(name, {}).get("markerfacecolor") is None else (view_styles or {}).get(name, {}).get("markerfacecolor")),
            markeredgecolor=view_colors[name],
        )
        for name in view_names
    ]
    fig.legend(handles, view_names, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=1, fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def plot_chronometric_jnds(
    group_jnd_by_view: Dict[str, pd.DataFrame],
    views: List[ViewSpec],
    view_colors: Dict[str, str],
    *,
    view_styles: Optional[Dict[str, dict]] = None,
) -> plt.Figure:
    """Compact animal-mean JND with 95% CI by ABL, reusing prepared fits.

    Keep ABLs separate even when the chronometric curves pool ABLs: these
    estimates come from each animal's per-ABL psychometric fits.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    abls = set()
    for view in views:
        table = group_jnd_by_view.get(view.name)
        if table is None or table.empty:
            continue
        table = table.sort_values("ABL")
        table = table[np.isfinite(table["mean"])]
        if table.empty:
            continue
        abls.update(table["ABL"])
        color = view_colors.get(view.name, "gray")
        appearance = (view_styles or {}).get(view.name, {})
        face = appearance.get("markerfacecolor")
        ax.errorbar(
            table["ABL"], table["mean"],
            yerr=ci95_yerr(table),
            color=color, label=view.name,
            linestyle=appearance.get("linestyle", "-"),
            marker=appearance.get("marker", "o"),
            markerfacecolor=color if face is None else face,
            markeredgecolor=color, capsize=3,
        )
    style = PlotStyle(title_fs=14, label_fs=14, tick_fs=14, legend_fs=14)
    style_axes(ax, style, "JND by ABL (mean and 95% CI)", "ABL (dB)", "JND (dB)", square=False)
    # Autoscale to the selected groups, including their confidence intervals.
    ax.margins(y=0.15)
    ax.set_xticks(sorted(abls))
    if abls:
        ax.legend(fontsize=14, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        ax.text(0.5, 0.5, "No JND estimates for selected groups", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig
