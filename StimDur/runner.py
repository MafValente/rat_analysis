from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from StimDur.config import (
    ViewSpec, StimDurSpec,
    FilterConfig, PlotStyle, StimDurComparisonConfig,
)
from StimDur.prepare import (
    apply_filters,
    build_prepared_by_view_and_stimdur,
    compute_group_jnd_by_view_and_stimdur,
)
from StimDur.layouts import plot_stimdur_4x3_for_view

def run_stimdur_comparison(
    cohort_csv: str,
    views: List[ViewSpec],
    stimdur_specs: List[StimDurSpec],
    cfg: StimDurComparisonConfig = StimDurComparisonConfig(),
    fcfg: FilterConfig = FilterConfig(),
    style: PlotStyle = PlotStyle(),
    stimdur_colors: Optional[Dict[str, str]] = None,
    show: bool = True,
) -> Dict[str, Any]:
    df = pd.read_csv(cohort_csv)
    df = apply_filters(df, fcfg)

    if stimdur_colors is None:
        stimdur_colors = {s.name: f"C{i % 10}" for i, s in enumerate(stimdur_specs)}

    prepared = build_prepared_by_view_and_stimdur(df, views, stimdur_specs, cfg)
    group_jnd = compute_group_jnd_by_view_and_stimdur(prepared, skip_abl=50)

    figs: Dict[str, plt.Figure] = {}
    for v in views:
        figs[v.name] = plot_stimdur_4x3_for_view(
            prepared_for_view=prepared[v.name],
            group_jnd_for_view=group_jnd[v.name],
            stimdur_specs=stimdur_specs,
            stimdur_colors=stimdur_colors,
            view_name=v.name,
            cfg=cfg,
            style=style,
        )
        if show:
            plt.show()

    return dict(prepared=prepared, group_jnd=group_jnd, figures=figs)
