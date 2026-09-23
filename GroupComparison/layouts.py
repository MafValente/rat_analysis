# analysis/groupcomparison/layouts.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Helpers.DataHelpers as DataHelpers
from analysis import psychometric as Psychometric

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
        style_axes(ax_psy, style, f"{vn} — Psychometric", "ILD (dB)", "P(Left)")

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
        style_axes(ax_psy, style, f"ABL {abl} — Psychometric", "ILD (dB)", "P(Left)")

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
        return pd.DataFrame(columns=["ILD", "mean", "sem", "n"])

    pooled_subject = (
        per_subj.groupby(["animal", "ILD"], as_index=False)[value_col]
        .mean()
    )
    return (
        pooled_subject.groupby("ILD")[value_col]
        .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 1 else np.nan, n="count")
        .reset_index()
    )


def _pooled_psychometric_group(tables: dict) -> pd.DataFrame:
    points = tables["psy_points"].copy()
    if points.empty:
        return pd.DataFrame(columns=["ILD", "mean", "sem", "n"])
    per_subject = (
        points.groupby(["subject", "ILD"], dropna=False)["PropLeft"]
        .mean()
        .reset_index()
    )
    return (
        per_subject.groupby("ILD", dropna=False)["PropLeft"]
        .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 1 else np.nan, n="count")
        .reset_index()
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
    x = _maybe_shift(df["ILD"], cfg)
    y = pd.to_numeric(df["mean"], errors="coerce").to_numpy(dtype=float)
    yerr = pd.to_numeric(df["sem"], errors="coerce").to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, markerfacecolor=markerfacecolor,
                markeredgecolor=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 4:
        return
    try:
        _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
            x[finite], y[finite], model="my_psycho",
            n_trials=np.clip(df.loc[finite, "n"].to_numpy(dtype=float), 1, None).astype(int),
            show_plot=False,
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
    sem = df["sem"].fillna(0).to_numpy(dtype=float)
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
    ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)


def _chrono_ylim(frames: List[pd.DataFrame]) -> tuple[float, float] | None:
    lows = []
    highs = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        y = pd.to_numeric(frame["mean"], errors="coerce")
        sem = pd.to_numeric(frame["sem"], errors="coerce").fillna(0)
        low = (y - sem).replace([np.inf, -np.inf], np.nan).dropna()
        high = (y + sem).replace([np.inf, -np.inf], np.nan).dropna()
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
    Always plots group means with shaded SEM, without individual animal traces.
    """
    view_names = [v.name for v in views]
    if separate_abls:
        abl_rows = sorted(set().union(*[
            set(prepared[vn]["rt_group"]["ABL"].unique()) for vn in view_names
        ]))
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
        ax_mt.set_xticks([-15, -10, -5, 0, 5, 10, 15])
        rt_ylim = _chrono_ylim(rt_frames)
        mt_ylim = _chrono_ylim(mt_frames)
        if rt_ylim is not None:
            ax_rt.set_ylim(*rt_ylim)
        if mt_ylim is not None:
            ax_mt.set_ylim(*mt_ylim)
        ax_mt.set_xticklabels(["-15", "-10", "-5", "0", "5", "10", "15"])

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
    view_names = [view.name for view in views]
    if separate_abls:
        abl_rows = sorted(set().union(*[
            set(prepared[name]["psy_group"]["ABL"].unique()) for name in view_names
        ]))
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
        style_axes(ax, style, f"{row_label} - Psychometric", "ILD (dB)", "P(Left)")
        ax.set_xlim(*cfg.xlim_sym)
        ax.set_ylim(0, 1)
        apply_50_tick_labels(ax, cfg.xlim_sym)

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
