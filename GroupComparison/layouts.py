# analysis/groupcomparison/layouts.py
from __future__ import annotations

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

import Helpers.DataHelpers as DataHelpers

from .config import ViewSpec, GroupComparisonConfig, PlotStyle, OverlaySpec
from .plots import (
    style_axes, apply_50_tick_labels,
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
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax_mt)
        apply_50_tick_labels(ax_psy)

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
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax_mt)
        apply_50_tick_labels(ax_psy)

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
    fig.legend(handles, view_names, loc="upper center", ncol=min(6, len(view_names)), fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig
