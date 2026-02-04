# analysis/groupcomparison/runner.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    ViewSpec, FilterConfig, PlotStyle, GroupComparisonConfig,
    OverlaySpec, JNDOverlaySpec
)
from .prepare import (
    apply_filters, build_prepared,
    compute_jnd_individuals_by_view, compute_group_jnd_by_view
)
from .layouts import plot_views_3x3, plot_abls_4x3
from .plots import plot_jnd_comparison_per_view
from Helpers.DataHelpers import prepare_data


def run_groupcomparison(
    cohort_csv: str,
    views: List[ViewSpec],
    cfg: GroupComparisonConfig = GroupComparisonConfig(),
    fcfg: FilterConfig = FilterConfig(),
    style: PlotStyle = PlotStyle(),
    overlay: OverlaySpec = OverlaySpec(),
    jnd_overlay: JNDOverlaySpec = JNDOverlaySpec(),
    layout: str = "views_3x3",   # "views_3x3" or "abls_4x3"
    view_colors: Optional[Dict[str, str]] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """
    - layout="views_3x3": returns main 3x3 fig + separate per-view JND comparison fig (old vs new individuals)
    - layout="abls_4x3": returns main 4x3 fig with JND inset inside psychometrics (group mean±SEM), no separate JND fig
    """
    df = pd.read_csv(cohort_csv)

    df = prepare_data(df, session_col="session", trial_col="trial")
    df = df[df["trial_is_repeat"] == False].copy()

    df = df[df["training_level"]==16].copy()


    sess = pd.to_numeric(df["session_type"], errors="coerce")
    sd   = pd.to_numeric(df["stim_dur"], errors="coerce")

    df = df[(sess == 1) | (sd == 6000)].copy()

    df = apply_filters(df, fcfg)

    prepared = build_prepared(df, views, cfg)

    jnd_indiv_by_view = compute_jnd_individuals_by_view(prepared, skip_abl=50)
    group_jnd_by_view = compute_group_jnd_by_view(jnd_indiv_by_view)

    if view_colors is None:
        base = ["C0", "C1", "C2", "C3", "C4"]
        view_colors = {v.name: base[i % len(base)] for i, v in enumerate(views)}

    figs: Dict[str, plt.Figure] = {}

    if layout == "views_3x3":
        fig_main = plot_views_3x3(
        prepared=prepared,
        views=views,
        cfg=cfg,
        style=style,
        overlay=overlay,
        group_jnd_by_view=group_jnd_by_view,
        view_colors=view_colors,
        add_jnd_inset=True,
    )
        figs["main"] = fig_main

        # JND comparison fig: one subplot per view, showing old + new individual points
        view_names = [v.name for v in views]
        fig_jnd, axes = plt.subplots(
            1, len(view_names),
            figsize=(4.2 * len(view_names), 4.0),
            squeeze=False
        )
        plot_jnd_comparison_per_view(
            fig=fig_jnd,
            axes=list(axes[0]),
            view_names=view_names,
            jnd_indiv_by_view=jnd_indiv_by_view,
            jnd_overlay=jnd_overlay,
            style=style,
        )
        fig_jnd.tight_layout()
        figs["jnd_comparison"] = fig_jnd

    elif layout == "abls_4x3":
        fig_main = plot_abls_4x3(
            prepared, views, cfg, style, overlay,
            view_colors=view_colors,
            group_jnd_by_view=group_jnd_by_view,
            add_inset=True,
        )
        figs["main"] = fig_main

    else:
        raise ValueError(f"Unknown layout='{layout}'. Use 'views_3x3' or 'abls_4x3'.")

    if show:
        plt.show()

    return dict(
        prepared=prepared,
        jnd_indiv_by_view=jnd_indiv_by_view,
        group_jnd_by_view=group_jnd_by_view,
        figures=figs,
    )
