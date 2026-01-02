from __future__ import annotations
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

from .config import ViewSpec, StimDurSpec, PlotStyle, StimDurComparisonConfig
from .plots import (
    style_axes, apply_50_tick_labels,
    plot_rt_on_ax, plot_mt_on_ax, plot_psy_on_ax,
)

def plot_stimdur_4x3_for_view(
    prepared_for_view: Dict[str, dict],          # stimdur_name -> tables
    group_jnd_for_view: Dict[str, "pd.DataFrame"],# stimdur_name -> df
    stimdur_specs: List[StimDurSpec],
    stimdur_colors: Dict[str, str],
    view_name: str,
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
) -> plt.Figure:
    

    # ABL rows present in this view (across stimdur splits)
    abl_rows = sorted(set().union(*[
        set(t["rt_group"]["ABL"].unique())
        for t in prepared_for_view.values()
        if t["rt_group"] is not None and not t["rt_group"].empty
    ]))
    if not abl_rows:
        raise ValueError(f"No ABL rows found for view '{view_name}' (check filters / stim_dur values).")

    names = [s.name for s in stimdur_specs]

    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.2 * len(abl_rows)), sharex="col")
    if len(abl_rows) == 1:
        axes = axes.reshape(1, 3)

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r, 0], axes[r, 1], axes[r, 2]

        for s in stimdur_specs:
            nm = s.name
            tables = prepared_for_view[nm]
            color = stimdur_colors[nm]

            plot_rt_on_ax(ax_rt, tables, abl=abl, color=color, cfg=cfg)
            plot_mt_on_ax(ax_mt, tables, abl=abl, color=color, cfg=cfg)
            plot_psy_on_ax(ax_psy, tables, abl=abl, color=color, cfg=cfg)

        ax_rt.set_title(f"{view_name} — ABL {abl} RT", fontsize=style.title_fs, pad=style.title_pad)
        ax_mt.set_title(f"{view_name} — ABL {abl} MT", fontsize=style.title_fs, pad=style.title_pad)
        ax_psy.set_title(f"{view_name} — ABL {abl} Psychometric", fontsize=style.title_fs, pad=style.title_pad)

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)

        style_axes(ax_rt, style); style_axes(ax_mt, style); style_axes(ax_psy, style)
        apply_50_tick_labels(ax_rt, cfg.xlim_abs)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

    # legend: stim_dur

    pretty = {
    "15": "SD = 15 ms",
    "60": "SD = 60 ms",
    "120": "SD = 120 ms",
    "6000": "SD = RT",
}

    legend_labels = [pretty.get(nm, str(nm)) for nm in names]

    handles = [
        plt.Line2D([], [], color=stimdur_colors[nm], marker="o", linestyle="None")
        for nm in names
    ]
    fig.legend(handles, legend_labels, loc="upper right", fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    return fig
